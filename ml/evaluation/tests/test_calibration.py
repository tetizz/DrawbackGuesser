from __future__ import annotations

import json
import math
import unittest

from ml.evaluation.calibration import (
    MAXIMUM_CALIBRATION_TEMPERATURE,
    MINIMUM_CALIBRATION_TEMPERATURE,
    CalibrationExample,
    TemperatureCalibration,
    apply_temperature_calibration,
    fit_validation_temperature,
    masked_temperature_softmax,
    multiclass_nll,
)


class TemperatureScalingTests(unittest.TestCase):
    def test_default_search_interval_is_frozen(self) -> None:
        self.assertEqual(MINIMUM_CALIBRATION_TEMPERATURE, 0.05)
        self.assertEqual(MAXIMUM_CALIBRATION_TEMPERATURE, 10.0)

        examples = (
            CalibrationExample((100.0, 0.0), 1, (False, False)),
            CalibrationExample((0.0, 100.0), 0, (False, False)),
        )
        fitted = fit_validation_temperature(examples, split="validation")
        self.assertLessEqual(
            fitted.temperature,
            MAXIMUM_CALIBRATION_TEMPERATURE,
        )

    def test_masked_softmax_preserves_exact_zero_for_eliminated_classes(self) -> None:
        probabilities = masked_temperature_softmax(
            (2.0, 1_000_000.0, -2.0),
            (False, True, False),
            3.0,
        )
        self.assertEqual(probabilities[1], 0.0)
        self.assertEqual(math.fsum(probabilities), 1.0)
        self.assertGreater(probabilities[0], probabilities[2])

        calibration = TemperatureCalibration(
            temperature=10_000.0,
            example_count=1,
            nll_before=1.0,
            nll_after=0.5,
        )
        applied = apply_temperature_calibration(
            (-100.0, 1_000_000.0, 100.0),
            (False, True, False),
            calibration,
        )
        self.assertEqual(applied[1], 0.0)

    def test_fit_is_deterministic_and_reduces_overconfident_validation_nll(self) -> None:
        examples = (
            CalibrationExample((8.0, 0.0, -1.0), 1, (False, False, False)),
            CalibrationExample((0.0, 8.0, -1.0), 0, (False, False, False)),
            CalibrationExample((8.0, 0.0, -1.0), 0, (False, False, False)),
            CalibrationExample((0.0, 8.0, -1.0), 1, (False, False, False)),
        )
        first = fit_validation_temperature(examples, split="validation")
        second = fit_validation_temperature(examples, split="validation")
        self.assertEqual(first, second)
        self.assertGreater(first.temperature, 1.0)
        self.assertLessEqual(first.nll_after, first.nll_before)
        self.assertAlmostEqual(
            first.nll_after,
            multiclass_nll(examples, first.temperature),
        )

    def test_fit_respects_per_example_hard_eliminations(self) -> None:
        examples = (
            CalibrationExample((3.0, 100.0, 0.0), 0, (False, True, False)),
            CalibrationExample((0.0, -3.0, 100.0), 1, (False, False, True)),
        )
        fitted = fit_validation_temperature(examples, split="validation")
        for example in examples:
            probabilities = apply_temperature_calibration(
                example.logits, example.eliminated, fitted
            )
            for probability, eliminated in zip(
                probabilities, example.eliminated, strict=True
            ):
                if eliminated:
                    self.assertEqual(probability, 0.0)

    def test_rejects_test_labels_true_elimination_and_invalid_masks(self) -> None:
        valid = (CalibrationExample((1.0, 0.0), 0, (False, False)),)
        with self.assertRaisesRegex(ValueError, "test labels are forbidden"):
            fit_validation_temperature(valid, split="test")
        with self.assertRaisesRegex(ValueError, "hard-eliminated"):
            CalibrationExample((1.0, 0.0), 1, (False, True))
        with self.assertRaisesRegex(ValueError, "eliminate every class"):
            masked_temperature_softmax((1.0, 0.0), (True, True), 1.0)
        with self.assertRaisesRegex(ValueError, "must align"):
            masked_temperature_softmax((1.0, 0.0), (False,), 1.0)
        with self.assertRaisesRegex(ValueError, "booleans"):
            masked_temperature_softmax((1.0, 0.0), (False, 0), 1.0)

    def test_metadata_is_json_roundtrip_safe_and_fails_closed(self) -> None:
        calibration = TemperatureCalibration(
            temperature=1.75,
            example_count=12,
            nll_before=2.0,
            nll_after=1.5,
        )
        metadata = json.loads(json.dumps(calibration.to_metadata()))
        self.assertEqual(
            TemperatureCalibration.from_metadata(metadata),
            calibration,
        )
        restored = dict(metadata)
        restored["preserves_hard_eliminations"] = False
        with self.assertRaisesRegex(ValueError, "preserve hard eliminations"):
            TemperatureCalibration.from_metadata(restored)
        test_fitted = dict(metadata)
        test_fitted["fitted_split"] = "test"
        with self.assertRaisesRegex(ValueError, "validation only"):
            TemperatureCalibration.from_metadata(test_fitted)

    def test_temperature_scaling_preserves_survivor_ranking(self) -> None:
        logits = (3.0, 2.0, 999.0, -1.0)
        eliminated = (False, False, True, False)
        before = masked_temperature_softmax(logits, eliminated, 1.0)
        after = masked_temperature_softmax(logits, eliminated, 7.0)
        self.assertEqual(
            sorted(range(len(logits)), key=before.__getitem__, reverse=True),
            sorted(range(len(logits)), key=after.__getitem__, reverse=True),
        )
        self.assertEqual(after[2], 0.0)


if __name__ == "__main__":
    unittest.main()
