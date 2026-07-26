from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import _bootstrap  # noqa: F401

from capturable_fixture import capturable_row
from drawback_ml.capturable_baseline import (
    CapturableTrainingConfig,
)
from drawback_ml.capturable_experiment import (
    run_candidate_selection,
    run_paired_sealed_evaluation,
    run_sealed_evaluation,
    run_selection,
    run_treatment_comparison,
)
from drawback_ml.capturable_records import CapturableDatasetError


class CapturableExperimentTests(unittest.TestCase):
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

    def test_treatment_comparison_allows_train_only_intervention(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            control_train = self._write_split(root, "control-train")
            treatment_train = self._write_split(root, "treatment-train")
            validation = self._write_split(root, "shared-validation")
            other_validation = self._write_split(root, "other-validation")
            paired_test = self._write_split(root, "paired-test")
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
            paired_output = root / "paired-evaluation.json"
            paired = run_paired_sealed_evaluation(
                output,
                paired_test,
                paired_output,
            )
            paired_report = json.loads(
                paired_output.read_text("utf-8")
            )
            self.assertIn(
                paired["decision"],
                {"confirm-treatment", "reject-treatment"},
            )
            self.assertEqual(
                paired_report["test"]["input"]["games"],
                2,
            )
            self.assertEqual(
                paired_report["sealedTestStatus"],
                "consumed",
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


if __name__ == "__main__":
    unittest.main()
