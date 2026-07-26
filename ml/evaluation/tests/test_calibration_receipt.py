from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from ml.evaluation.calibration import CalibrationExample
from ml.evaluation.calibration_receipt import (
    CalibrationReceiptInputs,
    CalibrationReceiptReference,
    load_calibration_receipt,
    write_calibration_receipt,
)
from ml.evaluation.calibration_release import (
    CalibrationObservation,
    CalibrationSidecarHeader,
    CalibrationSidecarStream,
)
from ml.evaluation.release_selection_bundle import ContentAddressedJson
from ml.evaluation.tests.test_release_selection_bundle import build_bundle
from ml.evaluation.validation_partition import VALIDATION_PARTITION_IDENTITY


def canonical(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: object) -> ContentAddressedJson:
    payload = canonical(value)
    path.write_bytes(payload)
    return ContentAddressedJson(path, digest(payload))


def build_inputs(directory: Path) -> CalibrationReceiptInputs:
    selection, training_run = build_bundle(directory)
    selection_value = json.loads(selection.path.read_text(encoding="utf-8"))
    selected = selection_value["selected"]
    checkpoint_path = directory / selected["checkpoint_file"]
    checkpoint = ContentAddressedJson(
        checkpoint_path, selected["checkpoint_sha256"]
    )
    calibration_seed = "1" * 64
    stream = CalibrationSidecarStream(
        directory / "calibration.ndjson",
        CalibrationSidecarHeader(
            calibration_seed_sha256=calibration_seed,
            checkpoint_sha256=checkpoint.sha256,
            selection_artifact_sha256=selection.sha256,
            symbolic_schema_sha256="2" * 64,
            symbolic_feature_version=6,
            class_count=1,
        ),
    )
    example = CalibrationExample((0.0,), 0, (False,))
    stream.add(CalibrationObservation("white", example))
    stream.add(CalibrationObservation("black", example))
    sidecar_file = stream.finalize()
    sidecar = ContentAddressedJson(sidecar_file.path, sidecar_file.sha256)
    run_value = json.loads(training_run.path.read_text(encoding="utf-8"))
    provenance = selection_value["provenance"]
    report = write_json(
        directory / "calibration-report.json",
        {
            "formatVersion": 1,
            "serialization": {"nonFiniteMetricPolicy": "null"},
            "evaluation": {
                "batchSize": 16,
                "validationPartition": {
                    "identity": VALIDATION_PARTITION_IDENTITY,
                    "name": "calibration-fit",
                    "seedSha256": calibration_seed,
                },
                "calibrationObservationSidecar": {
                    "file": sidecar.path.name,
                    "sha256": sidecar.sha256,
                },
            },
            "provenance": {
                "release_root_sha256": provenance["release_root_sha256"],
                "corpus_run_id": provenance["corpus_run_id"],
                "manifest_sha256": provenance[
                    "private_validation_manifest_sha256"
                ],
                "dataset_sha256": provenance["validation_dataset_sha256"],
                "checkpoint_file": checkpoint.path.name,
                "checkpoint_sha256": checkpoint.sha256,
                "checkpoint_seed": run_value["config"]["seed"],
                "checkpoint_epoch": selected["epoch"],
                "training_run_id": run_value["run_id"],
            },
            "metrics": {},
        },
    )
    return CalibrationReceiptInputs(
        evaluation_report=report,
        sidecar=sidecar,
        checkpoint=checkpoint,
        selection_artifact=selection,
        training_run=training_run,
    )


class CalibrationReceiptTest(unittest.TestCase):
    def test_round_trip_binds_completed_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            inputs = build_inputs(directory)
            reference = write_calibration_receipt(
                directory / "calibration-receipt.json", inputs
            )
            loaded = load_calibration_receipt(reference)
            self.assertEqual(loaded.validation_seed_sha256, "1" * 64)
            value = json.loads(reference.path.read_text(encoding="utf-8"))
            self.assertEqual(
                value["evaluation_report"]["sha256"],
                inputs.evaluation_report.sha256,
            )
            self.assertEqual(
                value["training_run"]["run_id"],
                json.loads(
                    inputs.training_run.path.read_text(encoding="utf-8")
                )["run_id"],
            )

    def test_loader_rejects_duplicate_keys_and_noncanonical_bytes(self) -> None:
        for mutation, expected in (
            (
                lambda payload: payload.replace(
                    b'{\n  "checkpoint"',
                    b'{\n  "format_version": 1,\n  "checkpoint"',
                    1,
                ),
                "duplicate key",
            ),
            (
                lambda payload: json.dumps(
                    json.loads(payload), sort_keys=True
                ).encode("utf-8"),
                "not canonical",
            ),
        ):
            with self.subTest(expected=expected):
                with tempfile.TemporaryDirectory() as raw:
                    directory = Path(raw)
                    reference = write_calibration_receipt(
                        directory / "calibration-receipt.json",
                        build_inputs(directory),
                    )
                    payload = mutation(reference.path.read_bytes())
                    tampered = directory / "tampered-receipt.json"
                    tampered.write_bytes(payload)
                    with self.assertRaisesRegex(ValueError, expected):
                        load_calibration_receipt(
                            CalibrationReceiptReference(
                                tampered, digest(payload)
                            )
                        )

    def test_loader_rejects_missing_or_altered_report_and_sidecar(self) -> None:
        for field, action, expected in (
            (
                "evaluation_report",
                lambda path: path.write_text("altered", encoding="utf-8"),
                "evaluation report sha256",
            ),
            (
                "evaluation_report",
                lambda path: path.unlink(),
                "evaluation report file is missing",
            ),
            (
                "sidecar",
                lambda path: path.write_text("altered", encoding="utf-8"),
                "sidecar sha256",
            ),
            (
                "sidecar",
                lambda path: path.unlink(),
                "sidecar file is missing",
            ),
        ):
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory() as raw:
                    directory = Path(raw)
                    inputs = build_inputs(directory)
                    reference = write_calibration_receipt(
                        directory / "calibration-receipt.json", inputs
                    )
                    action(getattr(inputs, field).path)
                    with self.assertRaisesRegex(ValueError, expected):
                        load_calibration_receipt(reference)

    def test_loader_rejects_rehashed_mismatched_binding(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            reference = write_calibration_receipt(
                directory / "calibration-receipt.json",
                build_inputs(directory),
            )
            value = json.loads(reference.path.read_text(encoding="utf-8"))
            value["checkpoint"]["sha256"] = "9" * 64
            payload = canonical(value)
            tampered = directory / "mismatched-receipt.json"
            tampered.write_bytes(payload)
            with self.assertRaisesRegex(ValueError, "checkpoint sha256"):
                load_calibration_receipt(
                    CalibrationReceiptReference(tampered, digest(payload))
                )

    def test_writer_rejects_report_and_sidecar_binding_mismatches(self) -> None:
        for mutate, expected in (
            (
                lambda value: value["evaluation"][
                    "calibrationObservationSidecar"
                ].__setitem__("sha256", "9" * 64),
                "sidecar binding",
            ),
            (
                lambda value: value["provenance"].__setitem__(
                    "release_root_sha256", "9" * 64
                ),
                "release_root_sha256",
            ),
            (
                lambda value: value["evaluation"][
                    "validationPartition"
                ].__setitem__("seedSha256", "9" * 64),
                "validation partition",
            ),
        ):
            with self.subTest(expected=expected):
                with tempfile.TemporaryDirectory() as raw:
                    directory = Path(raw)
                    inputs = build_inputs(directory)
                    report_value = json.loads(
                        inputs.evaluation_report.path.read_text(encoding="utf-8")
                    )
                    mutate(report_value)
                    report = write_json(
                        inputs.evaluation_report.path, report_value
                    )
                    altered = CalibrationReceiptInputs(
                        evaluation_report=report,
                        sidecar=inputs.sidecar,
                        checkpoint=inputs.checkpoint,
                        selection_artifact=inputs.selection_artifact,
                        training_run=inputs.training_run,
                    )
                    with self.assertRaisesRegex(ValueError, expected):
                        write_calibration_receipt(
                            directory / "receipt.json", altered
                        )

    def test_rejects_non_sibling_input_and_output_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            inputs = build_inputs(directory)
            outside = directory / "nested"
            outside.mkdir()
            copied = outside / inputs.evaluation_report.path.name
            copied.write_bytes(inputs.evaluation_report.path.read_bytes())
            displaced = CalibrationReceiptInputs(
                evaluation_report=ContentAddressedJson(
                    copied, inputs.evaluation_report.sha256
                ),
                sidecar=inputs.sidecar,
                checkpoint=inputs.checkpoint,
                selection_artifact=inputs.selection_artifact,
                training_run=inputs.training_run,
            )
            with self.assertRaisesRegex(ValueError, "sibling"):
                write_calibration_receipt(directory / "receipt.json", displaced)
            output = directory / "receipt.json"
            output.write_text("preserve", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                write_calibration_receipt(output, inputs)
            self.assertEqual(output.read_text(encoding="utf-8"), "preserve")


if __name__ == "__main__":
    unittest.main()
