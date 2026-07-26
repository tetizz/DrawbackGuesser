"""Immutable receipt for a completed calibration-fit evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Mapping

from .calibration_release import (
    ContentAddressedFile,
    load_calibration_observation_sidecar,
)
from .release_selection_bundle import (
    ContentAddressedJson,
    verify_release_selection_bundle,
)
from .validation_partition import (
    VALIDATION_PARTITION_IDENTITY,
    ValidationPartition,
)


CALIBRATION_RECEIPT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class CalibrationReceiptReference:
    path: Path
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "sha256", _digest(self.sha256, "receipt sha256"))


@dataclass(frozen=True)
class CalibrationReceiptInputs:
    evaluation_report: ContentAddressedJson
    sidecar: ContentAddressedJson
    checkpoint: ContentAddressedJson
    selection_artifact: ContentAddressedJson
    training_run: ContentAddressedJson


@dataclass(frozen=True)
class LoadedCalibrationReceipt:
    reference: CalibrationReceiptReference
    inputs: CalibrationReceiptInputs
    validation_seed_sha256: str
    release_root_sha256: str
    corpus_run_id: str


def write_calibration_receipt(
    output: Path,
    inputs: CalibrationReceiptInputs,
) -> CalibrationReceiptReference:
    """Verify completed local artifacts, then publish one non-circular receipt."""

    directory = output.parent.resolve()
    bindings = _verify_inputs(directory, inputs)
    value = {
        "format_version": CALIBRATION_RECEIPT_FORMAT_VERSION,
        **bindings,
    }
    payload = _canonical_json(value)
    _write_atomic_no_clobber(output, payload)
    actual = hashlib.sha256(output.read_bytes()).hexdigest()
    expected = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise ValueError("published calibration receipt bytes changed")
    return CalibrationReceiptReference(output, expected)


def load_calibration_receipt(
    reference: CalibrationReceiptReference,
) -> LoadedCalibrationReceipt:
    """Strictly load a receipt and reverify every sibling artifact it binds."""

    payload = _verified_bytes(reference.path, reference.sha256, "receipt")
    value = _strict_json(payload, "receipt")
    if _canonical_json(value) != payload:
        raise ValueError("calibration receipt JSON bytes are not canonical")
    expected_keys = {
        "format_version",
        "evaluation_report",
        "sidecar",
        "checkpoint",
        "selection_artifact",
        "training_run",
        "validation_partition",
        "provenance",
    }
    _exact_keys(value, expected_keys, "calibration receipt")
    if value["format_version"] != CALIBRATION_RECEIPT_FORMAT_VERSION:
        raise ValueError("unsupported calibration receipt format version")
    directory = reference.path.parent.resolve()
    inputs = CalibrationReceiptInputs(
        evaluation_report=_reference(
            directory, value["evaluation_report"], "evaluation report"
        ),
        sidecar=_reference(directory, value["sidecar"], "sidecar"),
        checkpoint=_reference(directory, value["checkpoint"], "checkpoint"),
        selection_artifact=_reference(
            directory, value["selection_artifact"], "selection artifact"
        ),
        training_run=_reference(
            directory, value["training_run"], "training run"
        ),
    )
    expected = {
        "format_version": CALIBRATION_RECEIPT_FORMAT_VERSION,
        **_verify_inputs(directory, inputs),
    }
    if value != expected:
        raise ValueError("calibration receipt bindings do not match local artifacts")
    partition = _object(value["validation_partition"], "validation partition")
    provenance = _object(value["provenance"], "receipt provenance")
    return LoadedCalibrationReceipt(
        reference=reference,
        inputs=inputs,
        validation_seed_sha256=str(partition["seed_sha256"]),
        release_root_sha256=str(provenance["release_root_sha256"]),
        corpus_run_id=str(provenance["corpus_run_id"]),
    )


def _verify_inputs(
    directory: Path,
    inputs: CalibrationReceiptInputs,
) -> dict[str, object]:
    for name, reference in (
        ("evaluation report", inputs.evaluation_report),
        ("sidecar", inputs.sidecar),
        ("checkpoint", inputs.checkpoint),
        ("selection artifact", inputs.selection_artifact),
        ("training run", inputs.training_run),
    ):
        _require_sibling(directory, reference.path, name)

    report_payload = _verified_bytes(
        inputs.evaluation_report.path,
        inputs.evaluation_report.sha256,
        "evaluation report",
    )
    report = _strict_json(report_payload, "evaluation report")
    if _canonical_json(report) != report_payload:
        raise ValueError("evaluation report JSON bytes are not canonical")
    _verified_bytes(
        inputs.checkpoint.path, inputs.checkpoint.sha256, "checkpoint"
    )
    sidecar = load_calibration_observation_sidecar(
        ContentAddressedFile(inputs.sidecar.path, inputs.sidecar.sha256)
    )
    selection = verify_release_selection_bundle(
        inputs.selection_artifact,
        inputs.training_run,
    )
    selected = selection.artifact
    run = selection.training_run
    if (
        selected.selected_checkpoint_sha256 != inputs.checkpoint.sha256
        or selected.selected_checkpoint_file != inputs.checkpoint.path.name
    ):
        raise ValueError("checkpoint disagrees with selected checkpoint")
    if sidecar.header.checkpoint_sha256 != inputs.checkpoint.sha256:
        raise ValueError("sidecar binds a different checkpoint")
    if (
        sidecar.header.selection_artifact_sha256
        != inputs.selection_artifact.sha256
    ):
        raise ValueError("sidecar binds a different selection artifact")

    evaluation = _object(report.get("evaluation"), "report evaluation")
    partition = _object(
        evaluation.get("validationPartition"), "report validation partition"
    )
    expected_partition = {
        "identity": VALIDATION_PARTITION_IDENTITY,
        "name": ValidationPartition.CALIBRATION_FIT.value,
        "seed_sha256": sidecar.header.calibration_seed_sha256,
    }
    report_partition = {
        "identity": partition.get("identity"),
        "name": partition.get("name"),
        "seed_sha256": partition.get("seedSha256"),
    }
    if report_partition != expected_partition or set(partition) != {
        "identity", "name", "seedSha256"
    }:
        raise ValueError("evaluation report validation partition is inconsistent")
    report_sidecar = _object(
        evaluation.get("calibrationObservationSidecar"),
        "report calibration sidecar",
    )
    expected_report_sidecar = {
        "file": inputs.sidecar.path.name,
        "sha256": inputs.sidecar.sha256,
    }
    if dict(report_sidecar) != expected_report_sidecar:
        raise ValueError("evaluation report sidecar binding is inconsistent")

    provenance = _object(report.get("provenance"), "report provenance")
    expected_provenance = {
        "checkpoint_file": inputs.checkpoint.path.name,
        "checkpoint_sha256": inputs.checkpoint.sha256,
        "checkpoint_seed": selection.artifact.training_seed,
        "checkpoint_epoch": selection.artifact.selected_epoch,
        "training_run_id": run.run_id,
        "release_root_sha256": selected.provenance.release_root_sha256,
        "corpus_run_id": selected.provenance.corpus_run_id,
        "manifest_sha256": (
            selected.provenance.private_validation_manifest_sha256
        ),
        "dataset_sha256": selected.provenance.validation_dataset_sha256,
    }
    for key, expected_value in expected_provenance.items():
        if provenance.get(key) != expected_value:
            raise ValueError(f"evaluation report provenance mismatch: {key}")

    return {
        "evaluation_report": _binding(inputs.evaluation_report),
        "sidecar": _binding(inputs.sidecar),
        "checkpoint": _binding(inputs.checkpoint),
        "selection_artifact": _binding(inputs.selection_artifact),
        "training_run": {
            **_binding(inputs.training_run),
            "run_id": run.run_id,
        },
        "validation_partition": expected_partition,
        "provenance": {
            "release_root_sha256": selected.provenance.release_root_sha256,
            "corpus_run_id": selected.provenance.corpus_run_id,
            "private_validation_manifest_sha256": (
                selected.provenance.private_validation_manifest_sha256
            ),
            "validation_dataset_sha256": (
                selected.provenance.validation_dataset_sha256
            ),
        },
    }


def _binding(reference: ContentAddressedJson) -> dict[str, str]:
    return {"file": reference.path.name, "sha256": reference.sha256}


def _reference(
    directory: Path, value: object, name: str
) -> ContentAddressedJson:
    binding = _object(value, f"{name} binding")
    expected = {"file", "sha256"}
    if name == "training run":
        expected.add("run_id")
        _digest(binding.get("run_id"), "training run_id")
    _exact_keys(binding, expected, f"{name} binding")
    filename = _basename(binding.get("file"), f"{name} file")
    path = directory / filename
    if not path.is_file():
        raise ValueError(f"{name} file is missing: {filename}")
    return ContentAddressedJson(
        path, _digest(binding.get("sha256"), f"{name} sha256")
    )


def _require_sibling(directory: Path, path: Path, name: str) -> None:
    if path.is_symlink() or path.parent.resolve() != directory:
        raise ValueError(f"{name} must be a basename-only sibling file")
    if not path.is_file():
        raise ValueError(f"{name} file is missing: {path.name}")


def _verified_bytes(path: Path, expected: str, name: str) -> bytes:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read {name}: {path}") from error
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(f"{name} sha256 does not match")
    return payload


def _strict_json(payload: bytes, name: str) -> Mapping[str, object]:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"{name} contains duplicate key {key!r}")
            result[key] = value
        return result

    def constant(token: str) -> None:
        raise ValueError(f"{name} contains non-finite constant {token}")

    try:
        value = json.loads(
            payload, object_pairs_hook=pairs, parse_constant=constant
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not strict UTF-8 JSON") from error
    _require_finite(value, name)
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} root must be an object")
    return value


def _canonical_json(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_atomic_no_clobber(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ValueError(
                f"refusing to overwrite calibration receipt: {path}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)


def _exact_keys(
    value: Mapping[str, object], expected: set[str], name: str
) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} fields are not canonical")


def _object(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _basename(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise ValueError(f"{name} must be a non-empty basename")
    return value


def _digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_finite(value: object, name: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} contains a non-finite number")
    if isinstance(value, Mapping):
        for item in value.values():
            _require_finite(item, name)
    elif isinstance(value, list):
        for item in value:
            _require_finite(item, name)
