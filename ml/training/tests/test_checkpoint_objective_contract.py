from __future__ import annotations

import copy
import io
import math
import unittest
from unittest.mock import patch

import _bootstrap  # noqa: F401

from drawback_ml.checkpoint import (
    FUSION_GRID_DRAWBACK_OBJECTIVE,
    LEGACY_ADDITIVE_DRAWBACK_OBJECTIVE,
    fusion_grid_drawback_objective_metadata,
)
from drawback_ml.features import FEATURE_DIMENSION
from drawback_ml.inference import CheckpointError, load_checkpoint_predictor
from drawback_ml.model import ModelConfig, create_sequence_model
from drawback_ml.records import parse_dataset_row
from drawback_ml.sequence import (
    ObservationTokenizerV2,
    PublicSequenceObservation,
    SanTokenizer,
)
from drawback_ml.symbolic_schema import (
    SYMBOLIC_FEATURE_DIMENSION,
    SYMBOLIC_FEATURE_VERSION,
    SYMBOLIC_RULE_IDS,
)
from test_records import row


class CheckpointObjectiveContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import torch
        except ImportError as error:
            raise unittest.SkipTest("PyTorch is not installed") from error
        cls.torch = torch
        cls.tokenizer = SanTokenizer.fit((("e4",),), max_history=8)
        cls.config = ModelConfig(
            input_dimension=FEATURE_DIMENSION,
            drawback_classes=len(SYMBOLIC_RULE_IDS),
            parameter_classes=1,
            legal_mask_dimension=3,
            hidden_dimension=8,
            model_variant="v21-hybrid",
            san_vocabulary_size=len(cls.tokenizer.vocabulary),
            san_embedding_dimension=4,
            sequence_hidden_dimension=5,
            symbolic_dimension=SYMBOLIC_FEATURE_DIMENSION,
            symbolic_hidden_dimension=6,
        )
        cls.model = create_sequence_model(cls.config)

    def payload(self, objective: object = ...) -> dict[str, object]:
        metadata: dict[str, object] = {
            "feature_schema_version": 1,
            "run_id": "a" * 64,
            "san_tokenizer": self.tokenizer.metadata(),
            "symbolic_feature_version": SYMBOLIC_FEATURE_VERSION,
            "symbolic_rule_ids": list(SYMBOLIC_RULE_IDS),
            "corpus_provenance": None,
        }
        if objective is not ...:
            metadata["drawback_loss_objective"] = objective
        return {
            "format_version": 3,
            "seed": 20260811,
            "epoch": 1,
            "drawback_vocabulary": list(SYMBOLIC_RULE_IDS),
            "parameter_vocabulary": ["none"],
            "model_config": {
                "input_dimension": self.config.input_dimension,
                "drawback_classes": self.config.drawback_classes,
                "parameter_classes": self.config.parameter_classes,
                "legal_mask_dimension": self.config.legal_mask_dimension,
                "hidden_dimension": self.config.hidden_dimension,
                "model_variant": self.config.model_variant,
                "san_vocabulary_size": self.config.san_vocabulary_size,
                "san_embedding_dimension": self.config.san_embedding_dimension,
                "sequence_hidden_dimension": (
                    self.config.sequence_hidden_dimension
                ),
                "symbolic_dimension": self.config.symbolic_dimension,
                "symbolic_hidden_dimension": (
                    self.config.symbolic_hidden_dimension
                ),
            },
            "training_metadata": metadata,
            "model_state": self.model.state_dict(),
        }

    def load(self, payload: dict[str, object]):
        stream = io.BytesIO()
        self.torch.save(payload, stream)
        stream.seek(0)
        return load_checkpoint_predictor(stream)

    def features_with_zero_prior_survivor(self):
        value = row("white", SYMBOLIC_RULE_IDS[0])
        count = len(SYMBOLIC_RULE_IDS)
        white_prior = [0.0] * count
        white_prior[0] = 1.0
        white_mask = [True] * count
        white_mask[0] = False
        white_mask[1] = False
        value.update(
            {
                "symbolicFeatureVersion": SYMBOLIC_FEATURE_VERSION,
                "symbolicWhiteRuleProbabilities": white_prior,
                "symbolicBlackRuleProbabilities": [1.0 / count] * count,
                "symbolicWhiteEliminated": white_mask,
                "symbolicBlackEliminated": [False] * count,
            }
        )
        return parse_dataset_row(value).features

    def features_with_adjacent_binary64_priors(self):
        value = row("white", SYMBOLIC_RULE_IDS[0])
        count = len(SYMBOLIC_RULE_IDS)
        white_prior = [0.0] * count
        white_prior[0] = 0.5
        white_prior[1] = math.nextafter(0.5, 0.0)
        white_mask = [True] * count
        white_mask[0] = False
        white_mask[1] = False
        value.update(
            {
                "symbolicFeatureVersion": SYMBOLIC_FEATURE_VERSION,
                "symbolicWhiteRuleProbabilities": white_prior,
                "symbolicBlackRuleProbabilities": [1.0 / count] * count,
                "symbolicWhiteEliminated": white_mask,
                "symbolicBlackEliminated": [False] * count,
            }
        )
        return parse_dataset_row(value).features

    def test_exact_new_metadata_selects_rank_preserving_inference(self) -> None:
        predictor = self.load(
            self.payload(fusion_grid_drawback_objective_metadata())
        )
        self.assertEqual(
            predictor.drawback_loss_objective,
            FUSION_GRID_DRAWBACK_OBJECTIVE,
        )
        output = predictor.predict(self.features_with_zero_prior_survivor())
        self.assertEqual(
            output.white_drawback_probabilities[SYMBOLIC_RULE_IDS[1]],
            0.0,
        )

    def test_missing_metadata_is_explicit_legacy_additive_behavior(self) -> None:
        predictor = self.load(self.payload())
        self.assertEqual(
            predictor.drawback_loss_objective,
            LEGACY_ADDITIVE_DRAWBACK_OBJECTIVE,
        )
        output = predictor.predict(self.features_with_zero_prior_survivor())
        self.assertGreater(
            output.white_drawback_probabilities[SYMBOLIC_RULE_IDS[1]],
            0.0,
        )

    def test_new_inference_preserves_adjacent_binary64_symbolic_tiers(
        self,
    ) -> None:
        predictor = self.load(
            self.payload(fusion_grid_drawback_objective_metadata())
        )
        output = predictor.predict(
            self.features_with_adjacent_binary64_priors()
        )
        self.assertGreater(
            output.white_drawback_probabilities[SYMBOLIC_RULE_IDS[0]],
            output.white_drawback_probabilities[SYMBOLIC_RULE_IDS[1]],
        )
        assert output.white_fused_logits is not None
        self.assertGreater(
            output.white_fused_logits[0],
            output.white_fused_logits[1],
        )

    def test_rejects_partial_unknown_or_type_coerced_metadata(self) -> None:
        valid = fusion_grid_drawback_objective_metadata()
        invalid = []
        partial = copy.deepcopy(valid)
        partial.pop("aggregation")
        invalid.append(partial)
        unknown = copy.deepcopy(valid)
        unknown["method"] = "unknown"
        invalid.append(unknown)
        coerced = copy.deepcopy(valid)
        coerced["alpha_grid"][0] = 0
        invalid.append(coerced)
        for objective in invalid:
            with self.subTest(objective=objective):
                with self.assertRaisesRegex(
                    CheckpointError,
                    "objective metadata",
                ):
                    self.load(self.payload(objective))


class V22CheckpointObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import torch
        except ImportError as error:
            raise unittest.SkipTest("PyTorch is not installed") from error
        cls.torch = torch
        cls.tokenizer = ObservationTokenizerV2.fit(
            (PublicSequenceObservation(("e4",), "e7e5"),),
            max_sequence=4,
        )

    def payload(
        self,
        *,
        mode: str = "exact-current-v2",
        include_objective: bool = True,
        training_mode: str | None = None,
    ) -> dict[str, object]:
        config = ModelConfig(
            input_dimension=FEATURE_DIMENSION,
            drawback_classes=len(SYMBOLIC_RULE_IDS),
            parameter_classes=1,
            legal_mask_dimension=3,
            hidden_dimension=8,
            model_variant="v22-hybrid",
            sequence_observation_mode=mode,  # type: ignore[arg-type]
            san_vocabulary_size=len(self.tokenizer.vocabulary),
            san_embedding_dimension=4,
            sequence_hidden_dimension=5,
            symbolic_dimension=SYMBOLIC_FEATURE_DIMENSION,
            symbolic_hidden_dimension=6,
        )
        metadata: dict[str, object] = {
            "feature_schema_version": 1,
            "run_id": "b" * 64,
            "san_tokenizer": self.tokenizer.metadata(),
            "sequence_observation_mode": (
                mode if training_mode is None else training_mode
            ),
            "symbolic_feature_version": SYMBOLIC_FEATURE_VERSION,
            "symbolic_rule_ids": list(SYMBOLIC_RULE_IDS),
            "corpus_provenance": None,
        }
        if include_objective:
            metadata["drawback_loss_objective"] = (
                fusion_grid_drawback_objective_metadata()
            )
        return {
            "format_version": 3,
            "seed": 20260901,
            "epoch": 1,
            "drawback_vocabulary": list(SYMBOLIC_RULE_IDS),
            "parameter_vocabulary": ["none"],
            "model_config": {
                "input_dimension": config.input_dimension,
                "drawback_classes": config.drawback_classes,
                "parameter_classes": config.parameter_classes,
                "legal_mask_dimension": config.legal_mask_dimension,
                "hidden_dimension": config.hidden_dimension,
                "model_variant": config.model_variant,
                "sequence_observation_mode": config.sequence_observation_mode,
                "san_vocabulary_size": config.san_vocabulary_size,
                "san_embedding_dimension": config.san_embedding_dimension,
                "sequence_hidden_dimension": config.sequence_hidden_dimension,
                "symbolic_dimension": config.symbolic_dimension,
                "symbolic_hidden_dimension": config.symbolic_hidden_dimension,
            },
            "training_metadata": metadata,
            "model_state": create_sequence_model(config).state_dict(),
        }

    def load(self, payload: dict[str, object]):
        stream = io.BytesIO()
        self.torch.save(payload, stream)
        stream.seek(0)
        return load_checkpoint_predictor(stream)

    def features(self):
        value = row("black", SYMBOLIC_RULE_IDS[0])
        count = len(SYMBOLIC_RULE_IDS)
        value.update(
            {
                "move": "e7e5",
                "historySan": ["e4"],
                "symbolicFeatureVersion": SYMBOLIC_FEATURE_VERSION,
                "symbolicWhiteRuleProbabilities": [1.0 / count] * count,
                "symbolicBlackRuleProbabilities": [1.0 / count] * count,
                "symbolicWhiteEliminated": [False] * count,
                "symbolicBlackEliminated": [False] * count,
            }
        )
        return parse_dataset_row(value).features

    def test_inference_uses_the_checkpoint_mode_for_the_same_tokenizer(
        self,
    ) -> None:
        observed_masks: list[bool] = []
        original = ObservationTokenizerV2.encode

        def capture(
            tokenizer: ObservationTokenizerV2,
            observation: PublicSequenceObservation,
            *,
            mask_current: bool = False,
        ):
            observed_masks.append(mask_current)
            return original(
                tokenizer,
                observation,
                mask_current=mask_current,
            )

        exact = self.load(self.payload(mode="exact-current-v2"))
        masked = self.load(self.payload(mode="masked-current-v2"))
        self.assertEqual(exact.san_tokenizer, masked.san_tokenizer)
        with patch.object(ObservationTokenizerV2, "encode", new=capture):
            exact.predict(self.features())
            masked.predict(self.features())
        self.assertEqual(observed_masks, [False, True])
        self.assertEqual(
            exact.drawback_loss_objective,
            FUSION_GRID_DRAWBACK_OBJECTIVE,
        )

    def test_v22_rejects_missing_objective_or_mismatched_mode(self) -> None:
        with self.assertRaisesRegex(CheckpointError, "objective"):
            self.load(self.payload(include_objective=False))
        with self.assertRaisesRegex(CheckpointError, "mode"):
            self.load(
                self.payload(
                    mode="exact-current-v2",
                    training_mode="masked-current-v2",
                )
            )

    def test_v22_rejects_the_legacy_san_tokenizer_contract(self) -> None:
        payload = self.payload()
        metadata = payload["training_metadata"]
        assert isinstance(metadata, dict)
        metadata["san_tokenizer"] = SanTokenizer.fit(
            (("e4",),),
            max_history=4,
        ).metadata()
        with self.assertRaisesRegex(CheckpointError, "tokenizer"):
            self.load(payload)

if __name__ == "__main__":
    unittest.main()
