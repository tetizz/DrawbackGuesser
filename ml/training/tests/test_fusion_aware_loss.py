from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import _bootstrap  # noqa: F401

from drawback_ml import streaming_training, symbolic, training
from drawback_ml.checkpoint import checkpoint_metadata
from drawback_ml.rank_preserving_fusion import rank_preserving_fusion
from drawback_ml.records import parse_dataset_row
from drawback_ml.streaming_training import _claim_run
from drawback_ml.symbolic import (
    FUSION_AWARE_LOSS_ALPHA_GRID,
    FUSION_AWARE_LOSS_METHOD,
    FUSION_AWARE_LOSS_VERSION,
    SYMBOLIC_RULE_COUNT,
    SYMBOLIC_RULE_IDS,
    SYMBOLIC_FEATURE_VERSION,
    fusion_aware_drawback_loss,
    fusion_aware_loss_metadata,
    torch_rank_preserving_fusion,
)
from drawback_ml.training import TrainingConfig
from test_records import row

try:
    import torch
except ImportError:
    torch = None


def _features(
    priors: tuple[float, ...],
    eliminated: tuple[bool, ...],
):
    if len(priors) != SYMBOLIC_RULE_COUNT or len(eliminated) != SYMBOLIC_RULE_COUNT:
        raise ValueError("test symbolic rows must use the complete vocabulary")
    value = row("white", SYMBOLIC_RULE_IDS[0])
    value.update(
        {
            "symbolicFeatureVersion": SYMBOLIC_FEATURE_VERSION,
            "symbolicWhiteRuleProbabilities": list(priors),
            "symbolicBlackRuleProbabilities": list(priors),
            "symbolicWhiteEliminated": list(eliminated),
            "symbolicBlackEliminated": list(eliminated),
        }
    )
    return parse_dataset_row(value).features


def _row(
    positive_priors: tuple[float, ...],
    *,
    zero_survivor: int | None = None,
):
    priors = [0.0] * SYMBOLIC_RULE_COUNT
    eliminated = [True] * SYMBOLIC_RULE_COUNT
    for index, prior in enumerate(positive_priors):
        priors[index] = prior
        eliminated[index] = False
    if zero_survivor is not None:
        eliminated[zero_survivor] = False
    return _features(tuple(priors), tuple(eliminated))


@unittest.skipUnless(torch is not None, "PyTorch is not installed")
class FusionAwareLossTests(unittest.TestCase):
    def test_torch_forward_matches_scalar_production_fusion(self) -> None:
        assert torch is not None
        priors = (0.45, 0.45, 0.1)
        features = _row(priors)
        residuals = [3.0, -1.0, 7.0] + [1000.0] * (
            SYMBOLIC_RULE_COUNT - len(priors)
        )
        neural = torch.tensor([residuals], dtype=torch.float64)

        fused, hard_mask = torch_rank_preserving_fusion(
            torch,
            neural,
            [features],
            SYMBOLIC_RULE_IDS,
            "white",
            alpha=0.5,
        )
        scalar = rank_preserving_fusion(
            residuals,
            [*priors, *([0.0] * (SYMBOLIC_RULE_COUNT - len(priors)))],
            [False, False, False]
            + [True] * (SYMBOLIC_RULE_COUNT - len(priors)),
            alpha=0.5,
        )

        self.assertEqual(
            hard_mask[0].tolist(),
            [False, False, False]
            + [True] * (SYMBOLIC_RULE_COUNT - len(priors)),
        )
        for index in range(SYMBOLIC_RULE_COUNT):
            self.assertAlmostEqual(
                fused[0, index].item(),
                scalar.logits[index],
                places=8,
            )
        probabilities = torch.softmax(
            fused.masked_fill(hard_mask, float("-inf")),
            dim=-1,
        )[0]
        for actual, expected in zip(
            probabilities.tolist(),
            scalar.probabilities,
            strict=True,
        ):
            self.assertAlmostEqual(actual, expected, places=8)

    def test_equal_tier_has_finite_learning_gradient(self) -> None:
        assert torch is not None
        features = _row((0.5, 0.5))
        neural = torch.zeros(
            (1, SYMBOLIC_RULE_COUNT),
            dtype=torch.float64,
            requires_grad=True,
        )
        target = torch.tensor([1], dtype=torch.long)
        supervised = torch.tensor([True], dtype=torch.bool)

        loss = fusion_aware_drawback_loss(
            torch,
            neural,
            [features],
            SYMBOLIC_RULE_IDS,
            "white",
            target,
            supervised,
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        assert neural.grad is not None
        self.assertTrue(torch.isfinite(neural.grad).all())
        self.assertGreater(neural.grad[0, 0].item(), 0.0)
        self.assertLess(neural.grad[0, 1].item(), 0.0)

    def test_three_constant_survivors_are_an_exact_no_op_at_every_alpha(
        self,
    ) -> None:
        assert torch is not None
        features = _row((0.5, 0.3, 0.2))
        devices = ["cpu"]
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability(0)
            compiled_architecture = f"sm_{capability[0]}{capability[1]}"
            if compiled_architecture in torch.cuda.get_arch_list():
                devices.append("cuda")

        for device in devices:
            with self.subTest(device=device):
                neural = torch.full(
                    (1, SYMBOLIC_RULE_COUNT),
                    7.0,
                    dtype=torch.float32,
                    device=device,
                )
                priors, eliminated = symbolic._ordered_symbolic_rows(
                    [features],
                    SYMBOLIC_RULE_IDS,
                    "white",
                )
                base, perturbation, _ = (
                    symbolic._torch_rank_preserving_components(
                        torch,
                        neural,
                        priors,
                        eliminated,
                    )
                )
                self.assertTrue(
                    torch.equal(
                        perturbation,
                        torch.zeros_like(perturbation),
                    )
                )
                symbolic_only, _ = torch_rank_preserving_fusion(
                    torch,
                    neural,
                    [features],
                    SYMBOLIC_RULE_IDS,
                    "white",
                    alpha=0.0,
                )
                self.assertTrue(torch.equal(symbolic_only, base))
                for alpha in FUSION_AWARE_LOSS_ALPHA_GRID:
                    fused, _ = torch_rank_preserving_fusion(
                        torch,
                        neural,
                        [features],
                        SYMBOLIC_RULE_IDS,
                        "white",
                        alpha=alpha,
                    )
                    self.assertTrue(torch.equal(fused, symbolic_only))

    def test_extreme_residual_cannot_invert_strict_symbolic_tiers(self) -> None:
        assert torch is not None
        features = _row((0.9, 0.1))
        neural = torch.tensor(
            [[-1000.0, 1000.0] + [0.0] * (SYMBOLIC_RULE_COUNT - 2)],
            dtype=torch.float64,
        )

        for alpha in FUSION_AWARE_LOSS_ALPHA_GRID:
            fused, _ = torch_rank_preserving_fusion(
                torch,
                neural,
                [features],
                SYMBOLIC_RULE_IDS,
                "white",
                alpha=alpha,
            )
            self.assertGreater(fused[0, 0].item(), fused[0, 1].item())

        higher = 0.5
        lower = math.nextafter(higher, 0.0)
        near_tie = _row((higher, lower))
        fused, _ = torch_rank_preserving_fusion(
            torch,
            neural,
            [near_tie],
            SYMBOLIC_RULE_IDS,
            "white",
            alpha=1.0,
        )
        self.assertGreater(fused[0, 0].item(), fused[0, 1].item())

    def test_eliminated_and_zero_prior_true_labels_fail_closed(self) -> None:
        assert torch is not None
        neural = torch.zeros((1, SYMBOLIC_RULE_COUNT))
        supervised = torch.tensor([True], dtype=torch.bool)
        cases = (
            (
                _row((1.0,)),
                1,
                "eliminated true white drawback",
            ),
            (
                _row((1.0,), zero_survivor=1),
                1,
                "zero mass to true white drawback",
            ),
        )
        for features, target_index, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                ValueError,
                message,
            ):
                fusion_aware_drawback_loss(
                    torch,
                    neural,
                    [features],
                    SYMBOLIC_RULE_IDS,
                    "white",
                    torch.tensor([target_index], dtype=torch.long),
                    supervised,
                )

    def test_loss_is_exact_mean_over_every_frozen_nonzero_alpha(self) -> None:
        assert torch is not None
        features = _row((0.4, 0.4, 0.2))
        neural = torch.tensor(
            [[1.5, -0.5, 0.25] + [0.0] * (SYMBOLIC_RULE_COUNT - 3)],
            dtype=torch.float64,
        )
        target = torch.tensor([1], dtype=torch.long)
        supervised = torch.tensor([True], dtype=torch.bool)
        actual = fusion_aware_drawback_loss(
            torch,
            neural,
            [features],
            SYMBOLIC_RULE_IDS,
            "black",
            target,
            supervised,
        )

        per_alpha = []
        for alpha in FUSION_AWARE_LOSS_ALPHA_GRID:
            fused, hard_mask = torch_rank_preserving_fusion(
                torch,
                neural,
                [features],
                SYMBOLIC_RULE_IDS,
                "black",
                alpha=alpha,
            )
            per_alpha.append(
                torch.nn.functional.cross_entropy(
                    fused.masked_fill(hard_mask, float("-inf")),
                    target,
                )
            )
        expected = torch.stack(per_alpha).mean()

        self.assertTrue(torch.equal(actual, expected))
        self.assertNotIn(0.0, FUSION_AWARE_LOSS_ALPHA_GRID)
        self.assertEqual(
            FUSION_AWARE_LOSS_ALPHA_GRID,
            (0.125, 0.25, 0.5, 1.0),
        )

    def test_grid_loss_prepares_shared_fusion_components_once(self) -> None:
        assert torch is not None
        features = _row((0.5, 0.5))
        neural = torch.zeros(
            (1, SYMBOLIC_RULE_COUNT),
            dtype=torch.float64,
            requires_grad=True,
        )
        target = torch.tensor([1], dtype=torch.long)
        supervised = torch.tensor([True], dtype=torch.bool)

        with patch.object(
            symbolic,
            "_torch_rank_preserving_components",
            wraps=symbolic._torch_rank_preserving_components,
        ) as prepare:
            loss = fusion_aware_drawback_loss(
                torch,
                neural,
                [features],
                SYMBOLIC_RULE_IDS,
                "white",
                target,
                supervised,
            )

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(prepare.call_count, 1)

    def test_both_trainers_bind_the_identical_helper(self) -> None:
        self.assertIs(
            training.fusion_aware_drawback_loss,
            streaming_training.fusion_aware_drawback_loss,
        )
        for trainer in (
            training.train_baseline,
            streaming_training._train_streaming_baseline,
        ):
            source = inspect.getsource(trainer)
            self.assertIn("fusion_aware_drawback_loss(", source)
            self.assertNotIn("combine_with_symbolic_prior(", source)


class FusionAwareMetadataTests(unittest.TestCase):
    def test_v21_metadata_binds_method_version_grid_and_production_policy(
        self,
    ) -> None:
        metadata = fusion_aware_loss_metadata("v21-hybrid")
        assert metadata is not None
        self.assertEqual(metadata["method"], FUSION_AWARE_LOSS_METHOD)
        self.assertEqual(metadata["version"], FUSION_AWARE_LOSS_VERSION)
        self.assertEqual(
            metadata["alpha_grid"],
            list(FUSION_AWARE_LOSS_ALPHA_GRID),
        )
        self.assertEqual(
            metadata["production_fusion_method"],
            "rank-preserving-bounded-residual-plus-symbolic-prior-v1",
        )
        self.assertIsNone(fusion_aware_loss_metadata("v2-gru"))

        checkpoint = checkpoint_metadata(
            seed=1,
            epoch=1,
            drawback_vocabulary=list(SYMBOLIC_RULE_IDS),
            parameter_vocabulary=[],
            training_metadata={"drawback_loss_objective": metadata},
        )
        self.assertEqual(
            checkpoint["training_metadata"]["drawback_loss_objective"],
            metadata,
        )

    def test_streaming_claim_and_config_freeze_the_objective(self) -> None:
        config = TrainingConfig(seed=7, model_variant="v21-hybrid")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "run"
            _claim_run(
                directory,
                config,
                {"device": "cpu"},
                {"policy": "fixture"},
            )
            claim = json.loads(
                (directory / "run.claim.json").read_text(encoding="utf-8")
            )
        self.assertEqual(
            claim["config"]["fusion_aware_loss_method"],
            FUSION_AWARE_LOSS_METHOD,
        )
        self.assertEqual(
            claim["config"]["fusion_aware_loss_version"],
            FUSION_AWARE_LOSS_VERSION,
        )
        self.assertEqual(
            claim["config"]["fusion_aware_loss_alpha_grid"],
            list(FUSION_AWARE_LOSS_ALPHA_GRID),
        )

        with self.assertRaisesRegex(ValueError, "frozen"):
            TrainingConfig(
                seed=7,
                model_variant="v21-hybrid",
                fusion_aware_loss_alpha_grid=(1.0,),
            )


if __name__ == "__main__":
    unittest.main()
