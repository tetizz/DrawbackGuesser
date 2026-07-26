from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace

from ml.evaluation.calibration import CalibrationExample
from ml.evaluation.calibration_release import (
    CalibrationObservation,
    CalibrationSidecarHeader,
    CalibrationSidecarStream,
    ContentAddressedFile,
    fit_calibration_release,
    load_calibration_observation_sidecar,
    write_calibration_observation_sidecar,
)
from ml.evaluation.calibration_receipt import (
    CalibrationReceiptInputs,
    CalibrationReceiptReference,
    write_calibration_receipt,
)
from ml.evaluation.cli import main
from ml.evaluation.runner import evaluate_held_out
from ml.evaluation.release_selection_bundle import ContentAddressedJson
from ml.evaluation.splits import SplitManifest
from ml.evaluation.tests.test_runner import FakePredictor, row
from ml.evaluation.validation_partition import VALIDATION_PARTITION_IDENTITY
from ml.evaluation.tests.training_corpus_set_fixture import (
    training_corpus_set_fixture,
)


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def observations() -> tuple[CalibrationObservation, ...]:
    eliminated = (False, False, True)
    white = (
        CalibrationExample((8.0, 0.0, 100.0), 1, eliminated),
        CalibrationExample((0.0, 8.0, 100.0), 0, eliminated),
        CalibrationExample((8.0, 0.0, 100.0), 0, eliminated),
        CalibrationExample((0.0, 8.0, 100.0), 1, eliminated),
    )
    black = (
        CalibrationExample((0.2, 0.0, 100.0), 0, eliminated),
        CalibrationExample((0.0, 0.2, 100.0), 1, eliminated),
        CalibrationExample((0.2, 0.0, 100.0), 0, eliminated),
        CalibrationExample((0.0, 0.2, 100.0), 1, eliminated),
    )
    return tuple(
        [
            *(CalibrationObservation("white", item) for item in white),
            *(CalibrationObservation("black", item) for item in black),
        ]
    )


def release_inputs(
    directory: Path,
) -> tuple[
    ContentAddressedFile,
    ContentAddressedFile,
    ContentAddressedFile,
    ContentAddressedJson,
    CalibrationReceiptReference,
]:
    checkpoint_path = directory / "selected.pt"
    checkpoint_payload = b"selected checkpoint"
    checkpoint_path.write_bytes(checkpoint_payload)
    checkpoint = ContentAddressedFile(
        checkpoint_path,
        digest(checkpoint_payload),
    )
    corpus_set = training_corpus_set_fixture()
    run_material = {
        "format": "drawbacktrainer-streaming-run",
        "version": 1,
        "config": {
            "seed": 20260811,
            "epochs": 1,
            "corpus_provenance": {
                "training_corpus_set": corpus_set,
                "training_corpus_set_sha256": corpus_set["sha256"],
            },
        },
        "runtime": {"device": "cpu"},
        "sampling": {"policy": "game-balanced-v1"},
    }
    run_id = digest(
        json.dumps(
            run_material, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    )
    run_path = directory / "run.claim.json"
    run_payload = (
        json.dumps(
            {"run_id": run_id, **run_material},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    run_path.write_bytes(run_payload)
    training_run = ContentAddressedJson(run_path, digest(run_payload))

    report_path = directory / "selection-report.json"
    report_payload = (
        json.dumps(
            {
                "evaluation": {
                    "validationPartition": {
                        "identity": VALIDATION_PARTITION_IDENTITY,
                        "name": "selection",
                        "seedSha256": "1" * 64,
                    }
                },
                "provenance": {
                    "release_root_sha256": "4" * 64,
                    "corpus_run_id": "5" * 64,
                    "manifest_sha256": "6" * 64,
                    "dataset_sha256": "7" * 64,
                    "checkpoint_file": checkpoint.path.name,
                    "checkpoint_sha256": checkpoint.sha256,
                    "checkpoint_seed": 20260811,
                    "checkpoint_epoch": 1,
                    "training_run_id": run_id,
                },
                "metrics": {
                    "white_drawback": {"negative_log_likelihood": 1.0},
                    "black_drawback": {"negative_log_likelihood": 1.0},
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    report_path.write_bytes(report_payload)
    report_sha256 = digest(report_payload)

    summary_path = directory / "summary.json"
    summary_payload = (
        json.dumps(
            {
                "format_version": 3,
                "training_seed": 20260811,
                "epoch": 1,
                "provenance": {
                    "release_root_sha256": "4" * 64,
                    "corpus_run_id": "5" * 64,
                    "training_corpus_set_sha256": corpus_set["sha256"],
                    "private_validation_manifest_sha256": "6" * 64,
                    "validation_dataset_sha256": "7" * 64,
                    "model_run_config_sha256": training_run.sha256,
                    "planned_epoch_count": 1,
                },
                "partition": {
                    "identity": VALIDATION_PARTITION_IDENTITY,
                    "name": "selection",
                    "seed_sha256": "1" * 64,
                },
                "checkpoint": {
                    "file": checkpoint.path.name,
                    "sha256": checkpoint.sha256,
                },
                "evaluation_report": {
                    "file": report_path.name,
                    "sha256": report_sha256,
                },
                "metrics": {"white_nll": 1.0, "black_nll": 1.0},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    summary_path.write_bytes(summary_payload)
    summary_sha256 = digest(summary_payload)

    selection_path = directory / "selection.json"
    selection_value = {
        "format_version": 3,
        "method": {
            "metric": "arithmetic-mean-white-black-nll",
            "tie_tolerance": 0.005,
            "tie_break": "earlier-epoch",
        },
        "partition": {
            "identity": VALIDATION_PARTITION_IDENTITY,
            "name": "selection",
            "seed_sha256": "1" * 64,
        },
        "provenance": {
            "release_root_sha256": "4" * 64,
            "corpus_run_id": "5" * 64,
            "training_corpus_set_sha256": corpus_set["sha256"],
            "private_validation_manifest_sha256": "6" * 64,
            "validation_dataset_sha256": "7" * 64,
            "model_run_config_sha256": training_run.sha256,
            "planned_epoch_count": 1,
        },
        "training_seed": 20260811,
        "candidates": [{
            "epoch": 1,
            "white_nll": 1.0,
            "black_nll": 1.0,
            "mean_nll": 1.0,
            "checkpoint_file": checkpoint.path.name,
            "checkpoint_sha256": checkpoint.sha256,
            "evaluation_report_file": "selection-report.json",
            "evaluation_report_sha256": report_sha256,
            "summary_file": "summary.json",
            "summary_sha256": summary_sha256,
        }],
        "selected": {
            "epoch": 1,
            "mean_nll": 1.0,
            "checkpoint_file": checkpoint.path.name,
            "checkpoint_sha256": checkpoint.sha256,
            "evaluation_report_file": "selection-report.json",
            "evaluation_report_sha256": report_sha256,
            "summary_file": "summary.json",
            "summary_sha256": summary_sha256,
        },
    }
    selection_payload = (
        json.dumps(selection_value, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")
    selection_path.write_bytes(selection_payload)
    selection = ContentAddressedFile(
        selection_path,
        digest(selection_payload),
    )

    sidecar_path = directory / "calibration.ndjson"
    sidecar = write_calibration_observation_sidecar(
        sidecar_path,
        CalibrationSidecarHeader(
            calibration_seed_sha256="2" * 64,
            checkpoint_sha256=checkpoint.sha256,
            selection_artifact_sha256=selection.sha256,
            symbolic_schema_sha256="3" * 64,
            symbolic_feature_version=6,
            class_count=3,
        ),
        observations(),
    )
    calibration_report_path = directory / "calibration-report.json"
    calibration_report_payload = (
        json.dumps(
            {
                "evaluation": {
                    "validationPartition": {
                        "identity": VALIDATION_PARTITION_IDENTITY,
                        "name": "calibration-fit",
                        "seedSha256": "2" * 64,
                    },
                    "calibrationObservationSidecar": {
                        "file": sidecar.path.name,
                        "sha256": sidecar.sha256,
                    },
                },
                "provenance": {
                    "release_root_sha256": "4" * 64,
                    "corpus_run_id": "5" * 64,
                    "manifest_sha256": "6" * 64,
                    "dataset_sha256": "7" * 64,
                    "checkpoint_file": checkpoint.path.name,
                    "checkpoint_sha256": checkpoint.sha256,
                    "checkpoint_seed": 20260811,
                    "checkpoint_epoch": 1,
                    "training_run_id": run_id,
                },
                "metrics": {},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    calibration_report_path.write_bytes(calibration_report_payload)
    receipt = write_calibration_receipt(
        directory / "calibration-receipt.json",
        CalibrationReceiptInputs(
            evaluation_report=ContentAddressedJson(
                calibration_report_path,
                digest(calibration_report_payload),
            ),
            sidecar=ContentAddressedJson(sidecar.path, sidecar.sha256),
            checkpoint=ContentAddressedJson(
                checkpoint.path, checkpoint.sha256
            ),
            selection_artifact=ContentAddressedJson(
                selection.path, selection.sha256
            ),
            training_run=training_run,
        ),
    )
    return sidecar, checkpoint, selection, training_run, receipt


class CalibrationReleaseTest(unittest.TestCase):
    def test_streaming_sidecar_is_atomic_and_content_addressed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            _sidecar, checkpoint, selection, _run, _receipt = release_inputs(directory)
            output = directory / "streamed.ndjson"
            stream = CalibrationSidecarStream(
                output,
                CalibrationSidecarHeader(
                    calibration_seed_sha256="4" * 64,
                    checkpoint_sha256=checkpoint.sha256,
                    selection_artifact_sha256=selection.sha256,
                    symbolic_schema_sha256="5" * 64,
                    symbolic_feature_version=6,
                    class_count=3,
                ),
            )
            for observation in observations():
                stream.add(observation)
            published = stream.finalize()
            self.assertEqual(
                published.sha256,
                digest(output.read_bytes()),
            )
            loaded = load_calibration_observation_sidecar(published)
            self.assertEqual((len(loaded.white), len(loaded.black)), (4, 4))
            self.assertEqual(list(directory.glob("*.tmp")), [])

    def test_runner_streams_only_mover_scores_without_changing_report(self) -> None:
        class ScorePredictor(FakePredictor):
            def predict(self, features: object):
                output = super().predict(features)
                return replace(
                    output,
                    white_fused_logits=(2.0, 0.0),
                    black_fused_logits=(0.0, 2.0),
                    white_hard_eliminated=(False, False),
                    black_hard_eliminated=(False, False),
                )

        rows = [
            row("white", "A", None, ply=0, move="e2e4"),
            row("black", "B", None, ply=1, move="e7e5"),
        ]
        manifest = SplitManifest(
            train=(1,), validation=(101,), test=(201,)
        )
        baseline = evaluate_held_out(
            rows, predictor=ScorePredictor(), split="validation", manifest=manifest
        )
        observed: list[CalibrationObservation] = []
        with_sidecar = evaluate_held_out(
            rows,
            predictor=ScorePredictor(),
            split="validation",
            manifest=manifest,
            calibration_sink=observed.append,
        )
        self.assertEqual(baseline, with_sidecar)
        self.assertEqual([item.color for item in observed], ["white", "black"])
        self.assertEqual(observed[0].example.logits, (2.0, 0.0))
        self.assertEqual(observed[1].example.logits, (0.0, 2.0))

    def test_sidecar_roundtrip_preserves_fused_logits_and_exact_masks(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            sidecar, _checkpoint, _selection, _run, _receipt = release_inputs(Path(raw))
            loaded = load_calibration_observation_sidecar(sidecar)
            self.assertEqual(len(loaded.white), 4)
            self.assertEqual(len(loaded.black), 4)
            self.assertEqual(loaded.white[0].logits, (8.0, 0.0, 100.0))
            self.assertEqual(
                loaded.white[0].eliminated,
                (False, False, True),
            )

    def test_fits_separate_heads_and_binds_every_source(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            sidecar, checkpoint, selection, training_run, receipt = release_inputs(directory)
            output = directory / "calibration.json"
            fit_calibration_release(
                sidecar=sidecar,
                checkpoint=checkpoint,
                selection_artifact=selection,
                training_run=training_run,
                calibration_receipt=receipt,
                output=output,
            )
            artifact = json.loads(output.read_text(encoding="utf-8"))
            self.assertGreater(artifact["white"]["temperature"], 1.0)
            self.assertLess(artifact["black"]["temperature"], 1.0)
            self.assertGreaterEqual(artifact["white"]["temperature"], 0.05)
            self.assertLessEqual(artifact["white"]["temperature"], 10.0)
            self.assertEqual(
                artifact["checkpoint"]["sha256"],
                checkpoint.sha256,
            )
            self.assertEqual(
                artifact["selection_artifact"]["sha256"],
                selection.sha256,
            )
            self.assertEqual(
                artifact["observation_sidecar"]["sha256"],
                sidecar.sha256,
            )
            self.assertEqual(
                artifact["calibration_receipt"]["sha256"],
                receipt.sha256,
            )
            self.assertEqual(
                artifact["calibration_evaluation_report"]["file"],
                "calibration-report.json",
            )
            self.assertEqual(
                artifact["partition"]["name"],
                "calibration-fit",
            )
            self.assertLess(
                artifact["mean_nll_after"],
                artifact["mean_nll_before"],
            )
            self.assertIs(
                artifact["method"]["preserves_hard_eliminations"],
                True,
            )

    def test_rejects_selection_gate_and_test_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            sidecar, _checkpoint, _selection, _run, _receipt = release_inputs(directory)
            original = sidecar.path.read_text(encoding="utf-8")
            for forbidden in ("selection", "validation-gate", "test"):
                with self.subTest(partition=forbidden):
                    changed = original.replace(
                        '"name":"calibration-fit"',
                        f'"name":"{forbidden}"',
                        1,
                    )
                    path = directory / f"{forbidden}.ndjson"
                    path.write_text(changed, encoding="utf-8", newline="\n")
                    reference = ContentAddressedFile(
                        path,
                        digest(path.read_bytes()),
                    )
                    with self.assertRaisesRegex(
                        ValueError,
                        "calibration-fit data only",
                    ):
                        load_calibration_observation_sidecar(reference)

    def test_rejects_tampered_bindings_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            sidecar, checkpoint, selection, training_run, receipt = release_inputs(directory)
            output = directory / "calibration.json"
            output.write_text("preserve", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                fit_calibration_release(
                    sidecar=sidecar,
                    checkpoint=checkpoint,
                    selection_artifact=selection,
                    training_run=training_run,
                    calibration_receipt=receipt,
                    output=output,
                )
            self.assertEqual(output.read_text(encoding="utf-8"), "preserve")

            checkpoint.path.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "checkpoint sha256"):
                fit_calibration_release(
                    sidecar=sidecar,
                    checkpoint=checkpoint,
                    selection_artifact=selection,
                    training_run=training_run,
                    calibration_receipt=receipt,
                    output=directory / "unused.json",
                )

    def test_rejects_forged_minimal_selection_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            sidecar, checkpoint, _selection, training_run, receipt = release_inputs(directory)
            forged_path = directory / "forged-selection.json"
            forged_payload = json.dumps(
                {
                    "partition": {"name": "selection"},
                    "selected": {
                        "checkpoint_sha256": checkpoint.sha256,
                    },
                },
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8") + b"\n"
            forged_path.write_bytes(forged_payload)
            forged = ContentAddressedFile(
                forged_path, digest(forged_payload)
            )
            with self.assertRaisesRegex(
                ValueError, "different selection artifact"
            ):
                fit_calibration_release(
                    sidecar=sidecar,
                    checkpoint=checkpoint,
                    selection_artifact=forged,
                    training_run=training_run,
                    calibration_receipt=receipt,
                    output=directory / "calibration.json",
                )

    def test_cli_publishes_content_address_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            sidecar, checkpoint, selection, training_run, receipt = release_inputs(directory)
            output = directory / "calibration.json"
            captured = io.StringIO()
            with redirect_stdout(captured):
                result = main(
                    [
                        "fit-calibration",
                        str(sidecar.path),
                        str(checkpoint.path),
                        str(selection.path),
                        str(training_run.path),
                        str(receipt.path),
                        str(output),
                        "--sidecar-sha256",
                        sidecar.sha256,
                        "--checkpoint-sha256",
                        checkpoint.sha256,
                        "--selection-sha256",
                        selection.sha256,
                        "--training-run-sha256",
                        training_run.sha256,
                        "--calibration-receipt-sha256",
                        receipt.sha256,
                    ]
                )
            self.assertEqual(result, 0)
            receipt = json.loads(captured.getvalue())
            self.assertEqual(
                receipt["sha256"],
                digest(output.read_bytes()),
            )


if __name__ == "__main__":
    unittest.main()
