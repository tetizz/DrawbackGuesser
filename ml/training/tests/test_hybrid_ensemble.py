from __future__ import annotations

from dataclasses import replace
import math
import unittest

import _bootstrap  # noqa: F401

from drawback_ml.checkpoint import (
    FUSION_GRID_DRAWBACK_OBJECTIVE,
    LEGACY_ADDITIVE_DRAWBACK_OBJECTIVE,
)
from drawback_ml.ensemble import (
    EnsembleError,
    HybridEnsemblePredictor,
    PROTOCOL_V2_ENSEMBLE_SEEDS,
)
from drawback_ml.inference import InferenceOutput
from drawback_ml.model import ModelConfig
from drawback_ml.records import parse_dataset_row
from drawback_ml.sequence import SanTokenizer
from drawback_ml.symbolic_schema import (
    SYMBOLIC_FEATURE_DIMENSION,
    SYMBOLIC_FEATURE_VERSION,
    SYMBOLIC_RULE_IDS,
)
from test_records import row


PARAMETERS = ("none", "rank=1")
LEGAL_DIMENSION = 3
TOKENIZER = SanTokenizer.fit((("e4",),), max_history=8)
CONFIG = ModelConfig(
    input_dimension=10,
    drawback_classes=len(SYMBOLIC_RULE_IDS),
    parameter_classes=len(PARAMETERS),
    legal_mask_dimension=LEGAL_DIMENSION,
    hidden_dimension=8,
    model_variant="v21-hybrid",
    san_vocabulary_size=len(TOKENIZER.vocabulary),
    san_embedding_dimension=4,
    sequence_hidden_dimension=5,
    symbolic_dimension=SYMBOLIC_FEATURE_DIMENSION,
    symbolic_hidden_dimension=6,
)


def public_features():
    value = row("white", "vegan")
    count = len(SYMBOLIC_RULE_IDS)
    white_prior = [0.0] * count
    white_prior[0] = 0.8
    white_prior[1] = 0.2
    black_prior = [1.0 / count] * count
    white_eliminated = [True] * count
    white_eliminated[0] = False
    white_eliminated[1] = False
    value.update(
        {
            "symbolicFeatureVersion": SYMBOLIC_FEATURE_VERSION,
            "symbolicWhiteRuleProbabilities": white_prior,
            "symbolicBlackRuleProbabilities": black_prior,
            "symbolicWhiteEliminated": white_eliminated,
            "symbolicBlackEliminated": [False] * count,
        }
    )
    return parse_dataset_row(value).features


def member_output(
    residual: float,
    *,
    trigger: float,
    white_mask: tuple[bool, ...] | None = None,
) -> InferenceOutput:
    count = len(SYMBOLIC_RULE_IDS)
    features = public_features()
    if white_mask is None:
        white_mask = features.symbolic_white_eliminated
    black_mask = features.symbolic_black_eliminated
    raw = tuple(residual for _ in range(count))
    return InferenceOutput(
        white_drawback_probabilities={
            drawback_id: 1.0 / count for drawback_id in SYMBOLIC_RULE_IDS
        },
        black_drawback_probabilities={
            drawback_id: 1.0 / count for drawback_id in SYMBOLIC_RULE_IDS
        },
        white_parameter_probabilities={
            PARAMETERS[0]: trigger,
            PARAMETERS[1]: 1.0 - trigger,
        },
        black_parameter_probabilities={
            PARAMETERS[0]: 1.0 - trigger,
            PARAMETERS[1]: trigger,
        },
        trigger_probability=trigger,
        legal_mask_probabilities=(trigger, 0.5, 1.0 - trigger),
        white_neural_residual_logits=raw,
        black_neural_residual_logits=raw,
        white_fused_logits=raw,
        black_fused_logits=raw,
        white_hard_eliminated=white_mask,
        black_hard_eliminated=black_mask,
    )


class StubMember:
    def __init__(self, seed: int, output: InferenceOutput) -> None:
        self.checkpoint_seed = seed
        self.checkpoint_epoch = 1
        self.training_run_id = f"{seed:064x}"
        self.drawback_vocabulary = SYMBOLIC_RULE_IDS
        self.parameter_vocabulary = PARAMETERS
        self.legal_mask_dimension = LEGAL_DIMENSION
        self.model_config = CONFIG
        self.san_tokenizer = TOKENIZER
        self.symbolic_enabled = True
        self.drawback_loss_objective = LEGACY_ADDITIVE_DRAWBACK_OBJECTIVE
        self.output = output

    def predict_batch(self, features):
        return tuple(self.output for _ in features)


def ensemble(
    outputs: tuple[InferenceOutput, InferenceOutput, InferenceOutput],
    *,
    fusion_alpha: float = 1.0,
) -> HybridEnsemblePredictor:
    members = tuple(
        StubMember(seed, output)
        for seed, output in zip(
            PROTOCOL_V2_ENSEMBLE_SEEDS, outputs, strict=True
        )
    )
    return HybridEnsemblePredictor(  # type: ignore[arg-type]
        members,
        fusion_alpha=fusion_alpha,
    )


class HybridEnsembleTests(unittest.TestCase):
    def test_averages_raw_residuals_then_applies_rank_preserving_fusion(
        self,
    ) -> None:
        outputs = (
            member_output(-3.0, trigger=0.0),
            member_output(0.0, trigger=0.5),
            member_output(3.0, trigger=1.0),
        )
        result = ensemble(outputs).predict(public_features())

        self.assertEqual(
            result.white_neural_residual_logits,
            (0.0,) * len(SYMBOLIC_RULE_IDS),
        )
        self.assertAlmostEqual(result.white_drawback_probabilities["vegan"], 0.8)
        self.assertAlmostEqual(result.white_drawback_probabilities["lame-duck"], 0.2)
        self.assertTrue(
            all(
                result.white_drawback_probabilities[drawback_id] == 0.0
                for drawback_id in SYMBOLIC_RULE_IDS[2:]
            )
        )
        assert result.white_fused_logits is not None
        self.assertAlmostEqual(result.white_fused_logits[0], math.log(0.8))
        self.assertAlmostEqual(result.white_fused_logits[1], math.log(0.2))

    def test_auxiliary_outputs_are_arithmetic_probability_means(self) -> None:
        result = ensemble(
            (
                member_output(-1.0, trigger=0.0),
                member_output(0.0, trigger=0.3),
                member_output(1.0, trigger=0.9),
            )
        ).predict(public_features())

        self.assertAlmostEqual(result.trigger_probability, 0.4)
        self.assertAlmostEqual(result.white_parameter_probabilities["none"], 0.4)
        self.assertAlmostEqual(result.black_parameter_probabilities["none"], 0.6)
        for actual, expected in zip(
            result.legal_mask_probabilities, (0.4, 0.5, 0.6), strict=True
        ):
            self.assertAlmostEqual(actual, expected)

    def test_extreme_residual_cannot_reverse_symbolic_survivor_order(
        self,
    ) -> None:
        output = member_output(0.0, trigger=0.5)
        residual = list(output.white_neural_residual_logits or ())
        residual[0] = -1000.0
        residual[1] = 1000.0
        adversarial = replace(
            output,
            white_neural_residual_logits=tuple(residual),
        )

        result = ensemble(
            (adversarial, adversarial, adversarial)
        ).predict(public_features())

        self.assertGreater(
            result.white_drawback_probabilities["vegan"],
            result.white_drawback_probabilities["lame-duck"],
        )
        self.assertEqual(
            result.white_drawback_probabilities[SYMBOLIC_RULE_IDS[2]],
            0.0,
        )

    def test_validates_configured_fusion_alpha(self) -> None:
        output = member_output(0.0, trigger=0.5)
        predictor = ensemble(
            (output, output, output),
            fusion_alpha=0,
        )
        result = predictor.predict(public_features())
        self.assertAlmostEqual(
            result.white_drawback_probabilities["vegan"],
            0.8,
        )
        with self.assertRaisesRegex(
            ValueError,
            "between zero and one",
        ):
            ensemble(
                (output, output, output),
                fusion_alpha=1.01,
            )

    def test_rejects_wrong_seed_order_or_incompatible_contract(self) -> None:
        output = member_output(0.0, trigger=0.5)
        wrong_order = tuple(
            StubMember(seed, output)
            for seed in reversed(PROTOCOL_V2_ENSEMBLE_SEEDS)
        )
        with self.assertRaisesRegex(EnsembleError, "ordered by fixed seeds"):
            HybridEnsemblePredictor(  # type: ignore[arg-type]
                wrong_order,
                fusion_alpha=1.0,
            )

        members = [
            StubMember(seed, output) for seed in PROTOCOL_V2_ENSEMBLE_SEEDS
        ]
        members[2].san_tokenizer = replace(TOKENIZER, max_history=9)
        with self.assertRaisesRegex(EnsembleError, "incompatible"):
            HybridEnsemblePredictor(  # type: ignore[arg-type]
                tuple(members),
                fusion_alpha=1.0,
            )

    def test_rejects_missing_raw_residual_or_mask_disagreement(self) -> None:
        valid = member_output(0.0, trigger=0.5)
        missing = replace(valid, white_neural_residual_logits=None)
        with self.assertRaisesRegex(EnsembleError, "raw neural residuals"):
            ensemble((valid, missing, valid)).predict(public_features())

        changed_mask = list(public_features().symbolic_white_eliminated)
        changed_mask[2] = False
        inconsistent = member_output(
            0.0, trigger=0.5, white_mask=tuple(changed_mask)
        )
        with self.assertRaisesRegex(EnsembleError, "inconsistent hard masks"):
            ensemble((valid, inconsistent, valid)).predict(public_features())

    def test_rejects_non_hybrid_member_and_invalid_auxiliary_probability(self) -> None:
        valid = member_output(0.0, trigger=0.5)
        members = [
            StubMember(seed, valid) for seed in PROTOCOL_V2_ENSEMBLE_SEEDS
        ]
        members[1].symbolic_enabled = False
        with self.assertRaisesRegex(EnsembleError, "incompatible"):
            HybridEnsemblePredictor(  # type: ignore[arg-type]
                tuple(members),
                fusion_alpha=1.0,
            )

        invalid = replace(valid, trigger_probability=float("nan"))
        with self.assertRaisesRegex(EnsembleError, "trigger probability"):
            ensemble((valid, invalid, valid)).predict(public_features())

    def test_empty_batch_is_deterministic_and_does_not_call_members(self) -> None:
        predictor = ensemble(
            tuple(
                member_output(0.0, trigger=0.5)
                for _ in PROTOCOL_V2_ENSEMBLE_SEEDS
            )  # type: ignore[arg-type]
        )
        self.assertEqual(predictor.predict_batch(()), ())

    def test_rejects_mixed_legacy_and_fusion_grid_objectives(self) -> None:
        output = member_output(0.0, trigger=0.5)
        members = [
            StubMember(seed, output) for seed in PROTOCOL_V2_ENSEMBLE_SEEDS
        ]
        members[1].drawback_loss_objective = FUSION_GRID_DRAWBACK_OBJECTIVE
        with self.assertRaisesRegex(EnsembleError, "incompatible"):
            HybridEnsemblePredictor(  # type: ignore[arg-type]
                tuple(members),
                fusion_alpha=1.0,
            )


if __name__ == "__main__":
    unittest.main()
