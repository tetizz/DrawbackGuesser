from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import _bootstrap  # noqa: F401

from capturable_fixture import capturable_row
from drawback_ml.capturable_baseline import (
    CapturableTrainingConfig,
    SOURCE_WEIGHTING_OBJECTIVE,
    _checked_train_source_weights,
    _parser as _baseline_parser,
    main as baseline_main,
    run_training,
)
from drawback_ml.capturable_candidate_selection import (
    load_treatment_comparison,
)
from drawback_ml.capturable_experiment import (
    _load_bound_selection_checkpoint,
    _load_selection_checkpoint,
    _parser as _experiment_parser,
    main as experiment_main,
    run_candidate_selection,
    run_paired_sealed_evaluation,
    run_sealed_evaluation,
    run_selection,
    run_treatment_comparison,
)
from drawback_ml.capturable_records import (
    CAPTURABLE_RULE_IDS,
    CapturableDatasetError,
)
from drawback_ml.capturable_reliability import (
    validation_reliability_checks,
)


class CapturableExperimentTests(unittest.TestCase):
    @staticmethod
    def _reliability_metrics(
        top1: float,
        top3: float,
    ) -> dict[str, object]:
        return {
            "hybrid": {
                "game_normalized_top_1_accuracy": top1,
                "game_normalized_top_3_accuracy": top3,
                "game_normalized_top_5_accuracy": max(top3, 0.60),
                "game_normalized_negative_log_likelihood": 2.0,
                "game_normalized_brier_score": 0.8,
                "expected_calibration_error": 0.1,
                "accuracy_after_moves": {
                    "5": 0.1,
                    "10": 0.2,
                    "15": 0.3,
                    "20": 0.4,
                },
                "metrics_per_drawback": {
                    drawback_id: {
                        "top_1_accuracy": top1,
                    }
                    for drawback_id in CAPTURABLE_RULE_IDS
                },
                "probability_diagnostics": {
                    "hard_elimination_violation_count": 0,
                    "missing_hard_mask_count": 0,
                    "checked_count": 1,
                    "hard_mask_checked_count": 1,
                    "maximum_eliminated_probability": 0.0,
                },
            },
            "trigger": {"accuracy": 0.75},
            "forced": {"accuracy": 0.95},
        }

    def test_validation_release_rejects_a_primary_win_with_top3_loss(
        self,
    ) -> None:
        report = self._reliability_metrics

        checks = validation_reliability_checks(
            report(0.30, 0.52),
            report(0.31, 0.51),
            True,
        )

        self.assertTrue(checks["primaryRankingConfirmed"])
        self.assertTrue(checks["top1NonRegression"])
        self.assertFalse(checks["top3NonRegression"])
        self.assertTrue(checks["perDrawbackTop1WithinOnePoint"])
        self.assertTrue(checks["symbolicAuthorityPreserved"])
        self.assertFalse(all(checks.values()))

        malformed = report(0.31, 0.53)
        malformed_hybrid = malformed["hybrid"]
        self.assertIsInstance(malformed_hybrid, dict)
        malformed_hybrid["accuracy_after_moves"] = {
            "5": 0.1,
            "10": float("nan"),
            "15": 0.3,
            "20": 0.4,
        }
        with self.assertRaisesRegex(
            CapturableDatasetError,
            "must be finite",
        ):
            validation_reliability_checks(
                report(0.30, 0.52),
                malformed,
                True,
            )

        drawback_regression = report(0.31, 0.53)
        drawback_metrics = drawback_regression["hybrid"]
        self.assertIsInstance(drawback_metrics, dict)
        per_drawback = drawback_metrics["metrics_per_drawback"]
        self.assertIsInstance(per_drawback, dict)
        per_drawback["vegan"]["top_1_accuracy"] = 0.28
        checks = validation_reliability_checks(
            report(0.30, 0.52),
            drawback_regression,
            True,
        )
        self.assertFalse(checks["perDrawbackTop1WithinOnePoint"])

        symbolic_regression = report(0.31, 0.53)
        symbolic_hybrid = symbolic_regression["hybrid"]
        self.assertIsInstance(symbolic_hybrid, dict)
        diagnostics = symbolic_hybrid["probability_diagnostics"]
        self.assertIsInstance(diagnostics, dict)
        diagnostics["hard_elimination_violation_count"] = 1
        checks = validation_reliability_checks(
            report(0.30, 0.52),
            symbolic_regression,
            True,
        )
        self.assertFalse(checks["symbolicAuthorityPreserved"])

        with self.assertRaisesRegex(
            CapturableDatasetError,
            "must be a boolean",
        ):
            validation_reliability_checks(
                report(0.30, 0.52),
                report(0.31, 0.53),
                "yes",  # type: ignore[arg-type]
            )
        with self.assertRaisesRegex(
            CapturableDatasetError,
            "Top-3 cannot be below Top-1",
        ):
            validation_reliability_checks(
                report(0.40, 0.30),
                report(0.41, 0.42),
                True,
            )
        invalid_top5 = report(0.30, 0.52)
        invalid_top5_hybrid = invalid_top5["hybrid"]
        self.assertIsInstance(invalid_top5_hybrid, dict)
        invalid_top5_hybrid["game_normalized_top_5_accuracy"] = 0.51
        with self.assertRaisesRegex(
            CapturableDatasetError,
            "Top-5 cannot be below Top-3",
        ):
            validation_reliability_checks(
                report(0.30, 0.52),
                invalid_top5,
                True,
            )
        wrong_horizons = report(0.31, 0.53)
        wrong_hybrid = wrong_horizons["hybrid"]
        self.assertIsInstance(wrong_hybrid, dict)
        wrong_hybrid["accuracy_after_moves"] = {
            "5": 0.1,
            "10": 0.2,
            "15": 0.3,
        }
        with self.assertRaisesRegex(
            CapturableDatasetError,
            "move horizons are incompatible",
        ):
            validation_reliability_checks(
                report(0.30, 0.52),
                wrong_horizons,
                True,
            )
        missing_drawback = report(0.31, 0.53)
        missing_hybrid = missing_drawback["hybrid"]
        self.assertIsInstance(missing_hybrid, dict)
        missing_metrics = missing_hybrid["metrics_per_drawback"]
        self.assertIsInstance(missing_metrics, dict)
        del missing_metrics[CAPTURABLE_RULE_IDS[-1]]
        with self.assertRaisesRegex(
            CapturableDatasetError,
            "drawback metrics are incompatible",
        ):
            validation_reliability_checks(
                report(0.30, 0.52),
                missing_drawback,
                True,
            )
        vacuous_symbolic = report(0.31, 0.53)
        vacuous_hybrid = vacuous_symbolic["hybrid"]
        self.assertIsInstance(vacuous_hybrid, dict)
        vacuous_diagnostics = vacuous_hybrid["probability_diagnostics"]
        self.assertIsInstance(vacuous_diagnostics, dict)
        vacuous_diagnostics["checked_count"] = 0
        vacuous_diagnostics["hard_mask_checked_count"] = 0
        checks = validation_reliability_checks(
            report(0.30, 0.52),
            vacuous_symbolic,
            True,
        )
        self.assertFalse(checks["symbolicAuthorityPreserved"])

        mismatched_symbolic = report(0.31, 0.53)
        mismatched_hybrid = mismatched_symbolic["hybrid"]
        self.assertIsInstance(mismatched_hybrid, dict)
        mismatched_diagnostics = mismatched_hybrid[
            "probability_diagnostics"
        ]
        self.assertIsInstance(mismatched_diagnostics, dict)
        mismatched_diagnostics["hard_mask_checked_count"] = 0
        checks = validation_reliability_checks(
            report(0.30, 0.52),
            mismatched_symbolic,
            True,
        )
        self.assertFalse(checks["symbolicAuthorityPreserved"])

    def test_treatment_comparison_retains_control_on_reliability_loss(
        self,
    ) -> None:
        def candidate(
            directory: str,
            seed: int,
            top1: float,
            top3: float,
        ) -> dict[str, object]:
            return {
                "selectionDirectory": directory,
                "selectionReport": "selection.json",
                "selectionReportSha256": f"{seed:064x}",
                "checkpointFile": "model.pt",
                "checkpointSha256": f"{seed + 10:064x}",
                "seed": seed,
                "triggerRowMultiplier": 1.0,
                "validationGameNormalizedTop1": top1,
                "validationGameNormalizedTop3": top3,
                "validationGameNormalizedNll": 2.0,
            }

        def report(
            seed: int,
            train_sha256: str,
            top1: float,
            top3: float,
        ) -> dict[str, object]:
            return {
                "inputs": {
                    "train": {"sha256": train_sha256},
                    "validation": {"sha256": "shared-validation"},
                },
                "config": {
                    "seed": seed,
                    "trigger_row_multiplier": 1.0,
                    "epochs": 1,
                },
                "validation": self._reliability_metrics(top1, top3),
            }

        control_candidate = candidate("control", 1, 0.30, 0.52)
        treatment_candidate = candidate("treatment", 2, 0.31, 0.51)
        control_report = report(1, "control-train", 0.30, 0.52)
        treatment_report = report(2, "treatment-train", 0.31, 0.51)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "comparison.json"
            with patch(
                "drawback_ml.capturable_candidate_selection."
                "_validated_candidate",
                side_effect=(
                    (control_candidate, control_report),
                    (treatment_candidate, treatment_report),
                ),
            ):
                result = run_treatment_comparison(
                    Path("control") / "selection.json",
                    (Path("treatment") / "selection.json",),
                    output,
                )
            artifact = json.loads(output.read_text("utf-8"))
            legacy = json.loads(json.dumps(artifact))
            legacy["version"] = 1
            legacy["promotionMetric"] = (
                "strictly better validation game-normalized Top-1, "
                "then Top-3, then lowest NLL; parameter tie-breaks "
                "cannot promote"
            )
            del legacy["primaryDecision"]
            del legacy["reliabilityChecks"]
            del legacy["releaseDecision"]
            legacy["decision"] = "promote-treatment"
            legacy["selected"] = legacy["bestTreatment"]
            legacy_path = Path(directory) / "legacy-comparison.json"
            legacy_path.write_text(
                json.dumps(
                    legacy,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
                newline="\n",
            )
            with patch(
                "drawback_ml.capturable_candidate_selection."
                "_validated_candidate",
                side_effect=(
                    (control_candidate, control_report),
                    (treatment_candidate, treatment_report),
                ),
            ):
                normalized_legacy, _ = load_treatment_comparison(
                    legacy_path
                )

        self.assertEqual(
            result["primaryDecision"],
            "confirm-treatment",
        )
        self.assertEqual(result["releaseDecision"], "retain-control")
        self.assertFalse(
            artifact["reliabilityChecks"]["top3NonRegression"]
        )
        self.assertEqual(artifact["selected"], artifact["control"])
        self.assertEqual(
            normalized_legacy["primaryDecision"],
            "confirm-treatment",
        )
        self.assertEqual(
            normalized_legacy["releaseDecision"],
            "retain-control",
        )
        self.assertEqual(
            normalized_legacy["selected"],
            normalized_legacy["control"],
        )

    def test_selection_never_opens_test_and_frozen_checkpoint_evaluates_once(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = self._write_split(root, "train")
            validation = self._write_split(root, "validation")
            test = self._write_split(root, "test")
            selection = root / "selection"
            config = CapturableTrainingConfig(
                seed=20260726,
                epochs=1,
                batch_size=2,
                hidden_dimension=8,
                torch_threads=1,
            )

            selected = run_selection(
                (train,),
                validation,
                selection,
                config,
            )
            self.assertIn("validationGameNormalizedTop1", selected)
            self.assertIn("validationMoveWeightedTop1", selected)
            self.assertNotIn("validationHybridTop1", selected)
            alternate_selection = root / "selection-alternate"
            run_selection(
                (train,),
                validation,
                alternate_selection,
                CapturableTrainingConfig(
                    seed=20260727,
                    epochs=1,
                    batch_size=2,
                    hidden_dimension=8,
                    torch_threads=1,
                ),
            )
            candidate_path = root / "candidate-selection.json"
            chosen = run_candidate_selection(
                (
                    selection / "selection.json",
                    alternate_selection / "selection.json",
                ),
                candidate_path,
            )
            candidate_report = json.loads(
                candidate_path.read_text("utf-8")
            )
            self.assertEqual(
                candidate_report["sealedTestStatus"],
                "unopened",
            )
            self.assertEqual(len(candidate_report["candidates"]), 2)
            self.assertIn(
                chosen["selectedDirectory"],
                {"selection", "selection-alternate"},
            )
            self.assertIn("validationGameNormalizedTop1", chosen)
            self.assertNotIn("validationHybridTop1", chosen)
            selection_report = json.loads(
                (selection / "selection.json").read_text("utf-8")
            )
            tampered = dict(selection_report)
            tampered["selectedFusionAlpha"] = (
                1.0
                if selection_report["selectedFusionAlpha"] != 1.0
                else 2.0
            )
            tampered_path = selection / "tampered-selection.json"
            tampered_path.write_text(
                json.dumps(
                    tampered,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
                newline="\n",
            )
            with self.assertRaisesRegex(
                CapturableDatasetError,
                "checkpoint does not bind",
            ):
                run_candidate_selection(
                    (
                        tampered_path,
                        alternate_selection / "selection.json",
                    ),
                    root / "tampered-candidate-selection.json",
                )

            self.assertEqual(
                selection_report["sealedTestStatus"],
                "unopened",
            )
            self.assertNotIn("test", selection_report)
            self.assertEqual(selected["selectedEpoch"], 1)
            sealed_path = root / "sealed-evaluation.json"
            evaluated = run_sealed_evaluation(
                selection / "model.pt",
                test,
                sealed_path,
            )
            sealed_report = json.loads(sealed_path.read_text("utf-8"))
            self.assertEqual(
                evaluated["checkpointSha256"],
                selection_report["checkpoint"]["sha256"],
            )
            self.assertEqual(
                sealed_report["test"]["input"]["games"],
                2,
            )

            overlap_output = root / "overlap.json"
            with self.assertRaisesRegex(
                CapturableDatasetError,
                "overlaps selection games",
            ):
                run_sealed_evaluation(
                    selection / "model.pt",
                    train,
                    overlap_output,
                )
            self.assertFalse(overlap_output.exists())

    def test_selection_authenticates_explicit_train_source_weights(self) -> None:
        options = _experiment_parser().parse_args(
            (
                "select",
                "--train",
                "baseline.ndjson",
                "--train-source-weight",
                "1",
                "--train",
                "diagnostic.ndjson",
                "--train-source-weight",
                "0.1",
                "--validation",
                "validation.ndjson",
                "--output",
                "selection",
            )
        )
        self.assertEqual(
            options.train,
            [Path("baseline.ndjson"), Path("diagnostic.ndjson")],
        )
        self.assertEqual(options.train_source_weight, [1.0, 0.1])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = self._write_split(root, "baseline")
            diagnostic = self._write_split(root, "diagnostic")
            validation = self._write_split(root, "validation")
            test = self._write_split(root, "test")
            output = root / "weighted-selection"
            config = CapturableTrainingConfig(
                seed=20260728,
                epochs=1,
                batch_size=2,
                hidden_dimension=8,
                torch_threads=1,
            )

            run_selection(
                (baseline, diagnostic),
                validation,
                output,
                config,
                (1.0, 0.1),
            )
            report_path = output / "selection.json"
            report = json.loads(report_path.read_text("utf-8"))

            self.assertEqual(
                [
                    source["weight"]
                    for source in report["inputs"]["train"]["sources"]
                ],
                [1.0, 0.1],
            )
            self.assertEqual(
                report["inputs"]["train"]["sourceWeightingObjective"],
                SOURCE_WEIGHTING_OBJECTIVE,
            )

            tampered = json.loads(json.dumps(report))
            tampered["inputs"]["train"]["sources"][1]["weight"] = 1.0
            tampered_path = output / "tampered-weight.json"
            tampered_path.write_text(
                json.dumps(
                    tampered,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
                newline="\n",
            )
            with self.assertRaisesRegex(
                CapturableDatasetError,
                "checkpoint does not bind",
            ):
                run_candidate_selection(
                    (tampered_path, report_path),
                    root / "tampered-choice.json",
                )

            tampered_objective = json.loads(json.dumps(report))
            tampered_objective["inputs"]["train"].pop(
                "sourceWeightingObjective"
            )
            tampered_objective_path = output / "tampered-objective.json"
            tampered_objective_path.write_text(
                json.dumps(
                    tampered_objective,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
                newline="\n",
            )
            with self.assertRaisesRegex(
                CapturableDatasetError,
                "checkpoint does not bind",
            ):
                run_candidate_selection(
                    (tampered_objective_path, report_path),
                    root / "tampered-objective-choice.json",
                )

            import torch

            checkpoint_path = output / "model.pt"
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            checkpoint["metadata"]["inputs"]["train"].pop(
                "sourceWeightingObjective"
            )
            tampered_checkpoint_path = output / "tampered-objective.pt"
            torch.save(checkpoint, tampered_checkpoint_path)
            with self.assertRaisesRegex(
                CapturableDatasetError,
                "source weighting objective",
            ):
                _load_selection_checkpoint(tampered_checkpoint_path)

            for weights in ((1.0,), (1.0, 0.0), (1.0, float("nan"))):
                with self.subTest(weights=weights):
                    invalid_output = (
                        root / f"invalid-{len(weights)}-{str(weights[-1])}"
                    )
                    with self.assertRaises(ValueError):
                        run_selection(
                            (baseline, diagnostic),
                            validation,
                            invalid_output,
                            config,
                            weights,
                        )
                    self.assertFalse(invalid_output.exists())

            oversized_output = root / "invalid-oversized"
            with self.assertRaises(ValueError):
                run_selection(
                    (baseline, diagnostic),
                    validation,
                    oversized_output,
                    config,
                    (1.0, 10**10000),
                )
            self.assertFalse(oversized_output.exists())

            extreme_selection_output = root / "invalid-relative-selection"
            with self.assertRaisesRegex(ValueError, "relative ratios"):
                run_selection(
                    (baseline, diagnostic),
                    validation,
                    extreme_selection_output,
                    config,
                    (1e300, 1e-300),
                )
            self.assertFalse(extreme_selection_output.exists())

            training_output = root / "invalid-training"
            with self.assertRaises(ValueError):
                run_training(
                    (baseline, diagnostic),
                    validation,
                    test,
                    training_output,
                    config,
                    (1.0,),
                )
            self.assertFalse(training_output.exists())

            extreme_training_output = root / "invalid-relative-training"
            with self.assertRaisesRegex(ValueError, "relative ratios"):
                run_training(
                    (baseline, diagnostic),
                    validation,
                    test,
                    extreme_training_output,
                    config,
                    (1e300, 1e-300),
                )
            self.assertFalse(extreme_training_output.exists())

    def test_source_weight_cli_pairing_contract(self) -> None:
        common = (
            "--train",
            "baseline.ndjson",
            "--train",
            "diagnostic.ndjson",
            "--train-source-weight",
            "1",
            "--train-source-weight",
            "0.1",
            "--validation",
            "validation.ndjson",
            "--output",
            "output",
        )
        selection = _experiment_parser().parse_args(("select", *common))
        training = _baseline_parser().parse_args(
            (*common, "--test", "test.ndjson")
        )

        for options in (selection, training):
            self.assertEqual(
                options.train,
                [Path("baseline.ndjson"), Path("diagnostic.ndjson")],
            )
            self.assertEqual(options.train_source_weight, [1.0, 0.1])

        omitted = _experiment_parser().parse_args(
            (
                "select",
                "--train",
                "baseline.ndjson",
                "--validation",
                "validation.ndjson",
                "--output",
                "output",
            )
        )
        self.assertIsNone(omitted.train_source_weight)
        with self.assertRaisesRegex(ValueError, "one value per train"):
            _checked_train_source_weights(
                (Path("baseline.ndjson"), Path("diagnostic.ndjson")),
                (1.0,),
            )

    def test_source_weight_cli_mains_forward_or_fail_closed(self) -> None:
        train = (
            "--train",
            "baseline.ndjson",
            "--train",
            "diagnostic.ndjson",
        )
        cases = (
            (
                (
                    "--train",
                    "baseline.ndjson",
                    "--train-source-weight",
                    "1",
                    "--train",
                    "diagnostic.ndjson",
                    "--train-source-weight",
                    "0.1",
                ),
                [1.0, 0.1],
            ),
            (
                (
                    *train,
                    "--train-source-weight",
                    "1",
                    "--train-source-weight",
                    "0.1",
                ),
                [1.0, 0.1],
            ),
            (train, None),
        )
        for train_arguments, expected_weights in cases:
            shared = (
                *train_arguments,
                "--validation",
                "validation.ndjson",
                "--output",
                "output",
            )
            with (
                self.subTest(entrypoint="selection", args=train_arguments),
                patch(
                    "drawback_ml.capturable_experiment.run_selection",
                    return_value={},
                ) as selection,
                patch("builtins.print"),
            ):
                self.assertEqual(
                    experiment_main(("select", *shared)),
                    0,
                )
                self.assertEqual(
                    selection.call_args.args[4],
                    expected_weights,
                )
            with (
                self.subTest(entrypoint="training", args=train_arguments),
                patch(
                    "drawback_ml.capturable_baseline.run_training",
                    return_value={},
                ) as training,
                patch("builtins.print"),
            ):
                self.assertEqual(
                    baseline_main((*shared, "--test", "test.ndjson")),
                    0,
                )
                self.assertEqual(
                    training.call_args.args[5],
                    expected_weights,
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            partial = (
                *train,
                "--train-source-weight",
                "1",
                "--validation",
                str(root / "validation.ndjson"),
                "--output",
                str(root / "output"),
            )
            with patch(
                "drawback_ml.capturable_experiment."
                "_load_stable_capturable_dataset"
            ) as selection_load:
                with self.assertRaisesRegex(ValueError, "one value per train"):
                    experiment_main(("select", *partial))
                selection_load.assert_not_called()
            with patch(
                "drawback_ml.capturable_baseline.load_capturable_dataset"
            ) as training_load:
                with self.assertRaisesRegex(ValueError, "one value per train"):
                    baseline_main(
                        (*partial, "--test", str(root / "test.ndjson"))
                    )
                training_load.assert_not_called()
            self.assertFalse((root / "output").exists())

    def test_treatment_comparison_allows_train_only_intervention(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            control_train = self._write_full_catalog_split(
                root,
                "control-train",
            )
            treatment_train = self._write_full_catalog_split(
                root,
                "treatment-train",
            )
            validation = self._write_full_catalog_split(
                root,
                "shared-validation",
            )
            other_validation = self._write_full_catalog_split(
                root,
                "other-validation",
            )
            paired_test = self._write_full_catalog_split(
                root,
                "paired-test",
            )
            config = CapturableTrainingConfig(
                seed=20260726,
                epochs=1,
                batch_size=2,
                hidden_dimension=8,
                torch_threads=1,
            )
            control = root / "control"
            treatment = root / "treatment"
            wrong_validation = root / "wrong-validation"
            run_selection((control_train,), validation, control, config)
            run_selection(
                (treatment_train,),
                validation,
                treatment,
                config,
            )
            run_selection(
                (treatment_train,),
                other_validation,
                wrong_validation,
                config,
            )
            with self.assertRaisesRegex(
                CapturableDatasetError,
                "changed after comparison authentication",
            ):
                _load_bound_selection_checkpoint(
                    control / "model.pt",
                    "0" * 64,
                )

            output = root / "treatment-comparison.json"
            result = run_treatment_comparison(
                control / "selection.json",
                (treatment / "selection.json",),
                output,
            )
            report = json.loads(output.read_text("utf-8"))
            self.assertIn(
                result["decision"],
                {"promote-treatment", "retain-control"},
            )
            self.assertEqual(
                result["releaseDecision"],
                result["decision"],
            )
            self.assertIn(
                result["primaryDecision"],
                {"confirm-treatment", "reject-treatment"},
            )
            self.assertEqual(
                report["releaseDecision"],
                report["decision"],
            )
            self.assertEqual(
                report["decision"] == "promote-treatment",
                all(report["reliabilityChecks"].values()),
            )
            self.assertEqual(
                set(report["reliabilityChecks"]),
                {
                    "primaryRankingConfirmed",
                    "top1NonRegression",
                    "top3NonRegression",
                    "negativeLogLikelihoodNonRegression",
                    "brierNonRegression",
                    "calibrationNonRegression",
                    "allMoveHorizonsNonRegression",
                    "triggerAccuracyNonRegression",
                    "forcedAccuracyNonRegression",
                    "perDrawbackTop1WithinOnePoint",
                    "symbolicAuthorityPreserved",
                },
            )
            self.assertEqual(report["sealedTestStatus"], "unopened")
            self.assertEqual(
                report["validationInput"]["sha256"],
                json.loads(
                    (control / "selection.json").read_text("utf-8")
                )["inputs"]["validation"]["sha256"],
            )
            self.assertNotEqual(
                report["control"]["trainInput"],
                report["treatments"][0]["trainInput"],
            )
            for field, replacement in (
                (
                    "primaryDecision",
                    (
                        "reject-treatment"
                        if report["primaryDecision"]
                        == "confirm-treatment"
                        else "confirm-treatment"
                    ),
                ),
                (
                    "releaseDecision",
                    (
                        "retain-control"
                        if report["releaseDecision"]
                        == "promote-treatment"
                        else "promote-treatment"
                    ),
                ),
                (
                    "decision",
                    (
                        "retain-control"
                        if report["decision"] == "promote-treatment"
                        else "promote-treatment"
                    ),
                ),
                (
                    "selected",
                    (
                        report["control"]
                        if report["selected"]
                        != report["control"]
                        else report["bestTreatment"]
                    ),
                ),
            ):
                tampered = json.loads(json.dumps(report))
                tampered[field] = replacement
                tampered_path = root / f"tampered-{field}.json"
                tampered_path.write_text(
                    json.dumps(
                        tampered,
                        allow_nan=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                    newline="\n",
                )
                with self.assertRaisesRegex(
                    CapturableDatasetError,
                    "decision is inconsistent",
                ):
                    load_treatment_comparison(tampered_path)

            tampered = json.loads(json.dumps(report))
            first_check = next(iter(tampered["reliabilityChecks"]))
            tampered["reliabilityChecks"][first_check] = not tampered[
                "reliabilityChecks"
            ][first_check]
            tampered_path = root / "tampered-reliability.json"
            tampered_path.write_text(
                json.dumps(
                    tampered,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
                newline="\n",
            )
            with self.assertRaisesRegex(
                CapturableDatasetError,
                "decision is inconsistent",
            ):
                load_treatment_comparison(tampered_path)

            malformed_boolean = json.loads(json.dumps(report))
            malformed_boolean["version"] = True
            malformed_float = json.loads(json.dumps(report))
            malformed_float["version"] = 2.0
            duplicate_treatment = json.loads(json.dumps(report))
            duplicate_treatment["treatments"].append(
                duplicate_treatment["treatments"][0]
            )
            reused_control = json.loads(json.dumps(report))
            reused_control["treatments"] = [reused_control["control"]]
            for name, malformed, message in (
                (
                    "boolean-version",
                    malformed_boolean,
                    "not a compatible treatment comparison",
                ),
                (
                    "float-version",
                    malformed_float,
                    "not a compatible treatment comparison",
                ),
                (
                    "duplicate-treatment",
                    duplicate_treatment,
                    "reuses a comparison candidate",
                ),
                (
                    "reused-control",
                    reused_control,
                    "reuses a comparison candidate",
                ),
            ):
                malformed_path = root / f"{name}.json"
                malformed_path.write_text(
                    json.dumps(
                        malformed,
                        allow_nan=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                    newline="\n",
                )
                with self.assertRaisesRegex(
                    CapturableDatasetError,
                    message,
                ):
                    load_treatment_comparison(malformed_path)

            legacy = json.loads(json.dumps(report))
            legacy["version"] = 1
            legacy["promotionMetric"] = (
                "strictly better validation game-normalized Top-1, "
                "then Top-3, then lowest NLL; parameter tie-breaks "
                "cannot promote"
            )
            del legacy["primaryDecision"]
            del legacy["reliabilityChecks"]
            del legacy["releaseDecision"]
            primary_confirmed = (
                report["primaryDecision"] == "confirm-treatment"
            )
            legacy["decision"] = (
                "promote-treatment"
                if primary_confirmed
                else "retain-control"
            )
            legacy["selected"] = (
                report["bestTreatment"]
                if primary_confirmed
                else report["control"]
            )
            legacy_path = root / "legacy-comparison.json"
            legacy_path.write_text(
                json.dumps(
                    legacy,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
                newline="\n",
            )
            normalized_legacy, _ = load_treatment_comparison(
                legacy_path
            )
            self.assertEqual(
                normalized_legacy["releaseDecision"],
                report["releaseDecision"],
            )
            self.assertEqual(
                normalized_legacy["selected"],
                report["selected"],
            )
            paired_output = root / "paired-evaluation.json"
            if result["releaseDecision"] == "promote-treatment":
                paired = run_paired_sealed_evaluation(
                    output,
                    paired_test,
                    paired_output,
                )
                paired_report = json.loads(
                    paired_output.read_text("utf-8")
                )
                self.assertIn(
                    paired["primaryDecision"],
                    {"confirm-treatment", "reject-treatment"},
                )
                self.assertIn(
                    paired["releaseDecision"],
                    {"promote-treatment", "retain-control"},
                )
                self.assertEqual(
                    paired_report["test"]["input"]["games"],
                    2,
                )
                self.assertEqual(
                    paired_report["sealedTestStatus"],
                    "consumed",
                )
                self.assertEqual(
                    set(paired_report["test"]["reliabilityChecks"]),
                    set(report["reliabilityChecks"]),
                )
                with self.assertRaisesRegex(
                    CapturableDatasetError,
                    "overlaps selection games",
                ):
                    run_paired_sealed_evaluation(
                        output,
                        control_train,
                        root / "overlap-paired-evaluation.json",
                    )
            else:
                with patch(
                    "drawback_ml.capturable_experiment."
                    "_load_stable_capturable_dataset"
                ) as test_loader:
                    with self.assertRaisesRegex(
                        CapturableDatasetError,
                        "did not authorize sealed test",
                    ):
                        run_paired_sealed_evaluation(
                            output,
                            paired_test,
                            paired_output,
                        )
                    test_loader.assert_not_called()
                self.assertFalse(paired_output.exists())

            with self.assertRaisesRegex(
                CapturableDatasetError,
                "different validation inputs",
            ):
                run_treatment_comparison(
                    control / "selection.json",
                    (wrong_validation / "selection.json",),
                    root / "wrong-comparison.json",
                )
            self.assertFalse((root / "wrong-comparison.json").exists())

    @staticmethod
    def _write_split(root: Path, split: str) -> Path:
        path = root / f"{split}.ndjson"
        rows = (
            capturable_row(
                game_id=f"{split}-white",
                drawback="vegan",
            ),
            capturable_row(
                game_id=f"{split}-black",
                color="black",
                drawback="checkers",
            ),
        )
        path.write_bytes(
            "".join(
                json.dumps(
                    row,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                + "\n"
                for row in rows
            ).encode("utf-8")
        )
        return path

    @staticmethod
    def _write_full_catalog_split(root: Path, split: str) -> Path:
        path = root / f"{split}.ndjson"
        rows = tuple(
            capturable_row(
                game_id=f"{split}-{drawback_id}",
                color="white" if index % 2 == 0 else "black",
                drawback=drawback_id,
            )
            for index, drawback_id in enumerate(CAPTURABLE_RULE_IDS)
        )
        path.write_bytes(
            "".join(
                json.dumps(
                    row,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                + "\n"
                for row in rows
            ).encode("utf-8")
        )
        return path


if __name__ == "__main__":
    unittest.main()
