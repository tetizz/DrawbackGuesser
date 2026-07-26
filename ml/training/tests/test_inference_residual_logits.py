from __future__ import annotations

import importlib.util
import unittest

import _bootstrap  # noqa: F401

from drawback_ml.features import FEATURE_DIMENSION, MOVE_VOCABULARY_SIZE
from drawback_ml.inference import CheckpointPredictor
from drawback_ml.model import ModelConfig, create_model
from drawback_ml.records import parse_dataset_row
from test_records import row


@unittest.skipUnless(
    importlib.util.find_spec("torch") is not None,
    "PyTorch is not installed",
)
class InferenceResidualLogitTests(unittest.TestCase):
    def test_exposes_raw_head_logits_without_changing_probabilities(self) -> None:
        import torch

        torch.manual_seed(123)
        model = create_model(
            ModelConfig(
                input_dimension=FEATURE_DIMENSION,
                drawback_classes=2,
                parameter_classes=1,
                legal_mask_dimension=MOVE_VOCABULARY_SIZE,
                hidden_dimension=8,
            )
        )
        predictor = CheckpointPredictor(
            torch_module=torch,
            model=model,
            drawback_vocabulary=("a", "b"),
            parameter_vocabulary=("none",),
            legal_mask_dimension=MOVE_VOCABULARY_SIZE,
            device="cpu",
            checkpoint_seed=123,
            checkpoint_epoch=1,
            training_run_id="a" * 64,
        )
        result = predictor.predict(
            parse_dataset_row(row("white", "a")).features
        )

        assert result.white_neural_residual_logits is not None
        expected = torch.softmax(
            torch.tensor(result.white_neural_residual_logits), dim=-1
        ).tolist()
        self.assertAlmostEqual(
            result.white_drawback_probabilities["a"], expected[0]
        )
        self.assertAlmostEqual(
            result.white_drawback_probabilities["b"], expected[1]
        )
        self.assertIsNone(result.white_fused_logits)


if __name__ == "__main__":
    unittest.main()
