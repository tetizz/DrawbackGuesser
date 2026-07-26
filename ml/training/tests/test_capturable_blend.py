from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import _bootstrap  # noqa: F401

from capturable_fixture import capturable_row
from drawback_ml.capturable_baseline import _canonical_json
from drawback_ml.capturable_blend import (
    BLEND_WEIGHTS,
    CAPTURABLE_BLEND_FORMAT,
    CAPTURABLE_BLEND_VERSION,
    PROTOCOL_COMMIT,
    PROTOCOL_FILE,
    PROTOCOL_SHA256,
    SELECTION_METRIC,
    ComponentPredictions,
    _EXPECTED_INPUTS,
    blend_components,
    blend_reliability_checks,
    load_blend_validation,
)
from drawback_ml.capturable_records import (
    CAPTURABLE_RULE_COUNT,
    CAPTURABLE_RULE_IDS,
    CapturableDatasetError,
    parse_capturable_dataset_row,
)


class CapturableBlendTests(unittest.TestCase):
    @staticmethod
    def _rows():
        return (
            parse_capturable_dataset_row(
                capturable_row(game_id="blend-game")
            ),
        )

    @staticmethod
    def _component(
        first_survivor: float,
        *,
        mask: tuple[bool, ...] | None = None,
        digest: str = "0" * 64,
    ) -> ComponentPredictions:
        hard_mask = mask or (True,) + (False,) * (
            CAPTURABLE_RULE_COUNT - 1
        )
        tail = (1.0 - first_survivor) / (
            CAPTURABLE_RULE_COUNT - 2
        )
        probabilities = (0.0, first_survivor) + (tail,) * (
            CAPTURABLE_RULE_COUNT - 2
        )
        return ComponentPredictions(
            drawback=(probabilities,),
            trigger=(0.2,),
            forced=(0.8,),
            parameters=((0.75, 0.25),),
            hard_masks=(hard_mask,),
            sha256=digest,
        )

    @staticmethod
    def _metrics() -> dict[str, object]:
        hybrid = {
            "accuracy_after_moves": {
                "5": 0.10,
                "10": 0.20,
                "15": 0.30,
                "20": 0.40,
            },
            "expected_calibration_error": 0.10,
            "game_normalized_brier_score": 0.80,
            "game_normalized_negative_log_likelihood": 2.0,
            "game_normalized_top_1_accuracy": 0.30,
            "game_normalized_top_3_accuracy": 0.50,
            "game_normalized_top_5_accuracy": 0.60,
            "hidden_parameter_accuracy": 0.50,
            "metrics_per_drawback": {
                drawback_id: {"top_1_accuracy": 0.30}
                for drawback_id in CAPTURABLE_RULE_IDS
            },
            "probability_diagnostics": {
                "checked_count": 10,
                "hard_elimination_violation_count": 0,
                "hard_mask_checked_count": 10,
                "maximum_eliminated_probability": 0.0,
                "missing_hard_mask_count": 0,
            },
        }
        color = {
            "game_normalized_negative_log_likelihood": 2.0,
            "game_normalized_top_1_accuracy": 0.30,
            "game_normalized_top_3_accuracy": 0.50,
        }
        binary = {
            "accuracy": 0.75,
            "brier_score": 0.20,
            "negative_log_likelihood": 0.50,
        }
        return {
            "forced": dict(binary),
            "hybrid": hybrid,
            "hybridByColor": {
                "black": dict(color),
                "white": dict(color),
            },
            "trigger": dict(binary),
        }

    def test_convex_mix_is_exact_and_preserves_hard_zero(self) -> None:
        rows = self._rows()
        control = self._component(0.6)
        treatment = self._component(0.2, digest="1" * 64)

        blended = blend_components(rows, control, treatment, 0.25)

        self.assertEqual(blended.drawback[0][0], 0.0)
        self.assertAlmostEqual(blended.drawback[0][1], 0.5)
        self.assertAlmostEqual(sum(blended.drawback[0]), 1.0)
        self.assertEqual(blended.trigger, (0.2,))
        self.assertEqual(blended.forced, (0.8,))
        self.assertEqual(blended.parameters, ((0.75, 0.25),))
        self.assertNotEqual(blended.sha256, control.sha256)
        self.assertNotEqual(blended.sha256, treatment.sha256)

    def test_blend_rejects_mask_mismatch_and_invalid_weight(self) -> None:
        rows = self._rows()
        control = self._component(0.6)
        mismatched = self._component(
            1.0,
            mask=(True, False, True) + (False,) * (
                CAPTURABLE_RULE_COUNT - 3
            ),
        )

        with self.assertRaisesRegex(
            CapturableDatasetError,
            "different rows or hard masks",
        ):
            blend_components(rows, control, mismatched, 0.5)
        with self.assertRaisesRegex(
            CapturableDatasetError,
            "finite in",
        ):
            blend_components(rows, control, control, float("nan"))

    def test_extended_gate_catches_each_new_regression_class(self) -> None:
        control = self._metrics()
        self.assertTrue(
            all(
                blend_reliability_checks(
                    control,
                    deepcopy(control),
                    True,
                ).values()
            )
        )

        cases = (
            ("top5NonRegression", ("hybrid", "game_normalized_top_5_accuracy"), 0.59),
            (
                "bothColorsNonRegression",
                ("hybridByColor", "white", "game_normalized_top_1_accuracy"),
                0.29,
            ),
            (
                "auxiliaryCalibrationNonRegression",
                ("trigger", "brier_score"),
                0.21,
            ),
            (
                "parameterAccuracyNonRegression",
                ("hybrid", "hidden_parameter_accuracy"),
                0.49,
            ),
        )
        for expected, keys, value in cases:
            with self.subTest(check=expected):
                candidate = deepcopy(control)
                target = candidate
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = value
                checks = blend_reliability_checks(
                    control,
                    candidate,
                    True,
                )
                self.assertFalse(checks[expected])

    def test_loader_recomputes_selection_and_rejects_tamper(self) -> None:
        metrics = self._metrics()
        candidates = [
            {
                "metrics": deepcopy(metrics),
                "predictionsSha256": f"{index + 1:064x}",
                "weight": weight,
            }
            for index, weight in enumerate(BLEND_WEIGHTS)
        ]
        selected = candidates[0]
        inputs = {
            "control": dict(_EXPECTED_INPUTS["control"]),
            "priorComparison": {
                **_EXPECTED_INPUTS["priorComparison"],
                "releaseDecision": "retain-control",
            },
            "treatment": dict(_EXPECTED_INPUTS["treatment"]),
            "validation": dict(_EXPECTED_INPUTS["validation"]),
        }
        checks = blend_reliability_checks(metrics, metrics, False)
        artifact = {
            "candidates": candidates,
            "control": {
                "metrics": deepcopy(metrics),
                "predictionsSha256": "a" * 64,
                "weight": 0.0,
            },
            "execution": {
                "cleanWorktree": True,
                "repository": "DrawbackGuesser",
                "revision": "f" * 40,
            },
            "format": CAPTURABLE_BLEND_FORMAT,
            "inputs": inputs,
            "primaryDecision": "reject-blend",
            "protocol": {
                "commit": PROTOCOL_COMMIT,
                "file": PROTOCOL_FILE,
                "sha256": PROTOCOL_SHA256,
            },
            "releaseDecision": "retain-control",
            "reliabilityChecks": checks,
            "sealedTestStatus": "unopened",
            "selected": selected,
            "selectionMetric": SELECTION_METRIC,
            "treatment": candidates[-1],
            "version": CAPTURABLE_BLEND_VERSION,
            "weightGrid": list(BLEND_WEIGHTS),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blend.json"
            path.write_bytes(_canonical_json(artifact))

            loaded, _ = load_blend_validation(path)

            self.assertEqual(loaded, artifact)
            tampered = deepcopy(artifact)
            tampered["selected"] = candidates[1]
            path.write_bytes(_canonical_json(tampered))
            with self.assertRaisesRegex(
                CapturableDatasetError,
                "selected blend is inconsistent",
            ):
                load_blend_validation(path)

    def test_loader_rejects_noncanonical_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blend.json"
            path.write_text(
                json.dumps({"format": CAPTURABLE_BLEND_FORMAT}),
                encoding="utf-8",
            )
            with self.assertRaises(CapturableDatasetError):
                load_blend_validation(path)


if __name__ == "__main__":
    unittest.main()
