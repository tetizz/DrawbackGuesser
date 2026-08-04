"""Canonical calibration evidence for the fixed three-member ensemble.

This is intentionally a separate protocol from the legacy single-checkpoint
calibration formats.  Every stage repeats the immutable ensemble and corpus
identity, making member substitution and cross-corpus evidence mixing fail
closed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Iterable, Mapping
from ml.training.drawback_ml.durable_publish import (
    abort_staged_file_safely,
    publish_bytes_durable,
    publish_bytes_durable_exact,
    publish_staged_file_durable,
)
from ml.training.drawback_ml.path_validation import is_portable_safe_basename
from ml.training.drawback_ml.rank_preserving_fusion import (
    RANK_PRESERVING_FUSION_METHOD,
)

from .calibration import (
    MAXIMUM_CALIBRATION_TEMPERATURE,
    MINIMUM_CALIBRATION_TEMPERATURE,
    CalibrationExample,
    apply_temperature_calibration,
    fit_validation_temperature,
)
from .ensemble_release import (
    ENSEMBLE_TRAINING_SEEDS,
    LoadedEnsembleRelease,
    verify_ensemble_release,
)
from .fusion_selection import (
    FusionSelectionIdentity,
    load_fusion_selection_artifact,
)
from .release_selection_bundle import ContentAddressedJson
from .validation_partition import (
    VALIDATION_PARTITION_IDENTITY,
    ValidationPartition,
)


SIDECAR_FORMAT = "drawbacktrainer-ensemble-calibration-sidecar"
REPORT_FORMAT = "drawbacktrainer-ensemble-evaluation-report"
RECEIPT_FORMAT = "drawbacktrainer-ensemble-calibration-receipt"
ARTIFACT_FORMAT = "drawbacktrainer-ensemble-calibration"
FORMAT_VERSION = 3
SYMBOLIC_FEATURE_VERSION = 6
CLASS_COUNT = 182
FUSION_METHOD = RANK_PRESERVING_FUSION_METHOD


@dataclass(frozen=True)
class ContentAddressedFile:
    path: Path
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "sha256", _digest(self.sha256, "file sha256"))


@dataclass(frozen=True)
class EnsembleCalibrationMember:
    seed: int
    selection_sha256: str
    training_claim_sha256: str
    training_run_id: str
    checkpoint_sha256: str
    checkpoint_epoch: int

    def __post_init__(self) -> None:
        if self.seed not in ENSEMBLE_TRAINING_SEEDS:
            raise ValueError("calibration member uses an unsupported seed")
        for name in (
            "selection_sha256",
            "training_claim_sha256",
            "training_run_id",
            "checkpoint_sha256",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        _positive_int(self.checkpoint_epoch, "checkpoint epoch")


@dataclass(frozen=True)
class EnsembleCalibrationIdentity:
    ensemble_release_sha256: str
    fusion_selection_sha256: str
    selected_alpha: float
    members: tuple[EnsembleCalibrationMember, ...]
    release_root_sha256: str
    corpus_run_id: str
    private_validation_manifest_sha256: str
    validation_dataset_sha256: str
    training_corpus_set_sha256: str
    calibration_seed_sha256: str
    symbolic_schema_sha256: str
    training_corpus_set: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        for name in (
            "ensemble_release_sha256",
            "fusion_selection_sha256",
            "release_root_sha256",
            "corpus_run_id",
            "private_validation_manifest_sha256",
            "validation_dataset_sha256",
            "training_corpus_set_sha256",
            "calibration_seed_sha256",
            "symbolic_schema_sha256",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if (
            isinstance(self.selected_alpha, bool)
            or not isinstance(self.selected_alpha, (int, float))
            or not math.isfinite(float(self.selected_alpha))
            or not 0.0 <= float(self.selected_alpha) <= 1.0
        ):
            raise ValueError("selected fusion alpha must be between zero and one")
        object.__setattr__(self, "selected_alpha", float(self.selected_alpha))
        if tuple(member.seed for member in self.members) != ENSEMBLE_TRAINING_SEEDS:
            raise ValueError("calibration members are not in fixed seed order")
        if len({member.checkpoint_sha256 for member in self.members}) != 3:
            raise ValueError("calibration members reuse a checkpoint")
        if self.training_corpus_set is not None:
            normalized = _corpus_mapping(self.training_corpus_set)
            if (
                hashlib.sha256(_canonical_compact(normalized)).hexdigest()
                != self.training_corpus_set_sha256
            ):
                raise ValueError(
                    "training corpus set mapping disagrees with release hash"
                )
            object.__setattr__(self, "training_corpus_set", normalized)


@dataclass(frozen=True)
class EnsembleCalibrationObservation:
    color: str
    example: CalibrationExample

    def __post_init__(self) -> None:
        if self.color not in {"white", "black"}:
            raise ValueError("observation color must be white or black")
        if len(self.example.logits) != CLASS_COUNT:
            raise ValueError("observation must contain exactly 182 fused logits")


@dataclass(frozen=True)
class LoadedEnsembleCalibrationSidecar:
    identity: EnsembleCalibrationIdentity
    white: tuple[CalibrationExample, ...]
    black: tuple[CalibrationExample, ...]
    source: ContentAddressedFile


@dataclass(frozen=True)
class EnsembleCalibrationReceipt:
    source: ContentAddressedFile
    report: ContentAddressedFile
    sidecar: ContentAddressedFile
    ensemble_release: ContentAddressedJson
    fusion_selection: ContentAddressedJson
    identity: EnsembleCalibrationIdentity


class EnsembleCalibrationSidecarStream:
    """Bounded-memory, atomic writer for canonical observation records."""

    def __init__(
        self, output: Path, identity: EnsembleCalibrationIdentity
    ) -> None:
        self.output = output
        self.identity = identity
        output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
        )
        self._temporary = Path(temporary_name)
        self._file = os.fdopen(descriptor, "wb")
        self._hash = hashlib.sha256()
        self._counts = {"white": 0, "black": 0}
        self._index = 0
        self._closed = False
        self._write(_canonical_compact(_header_value(identity)))

    def _write(self, record: bytes) -> None:
        payload = record + b"\n"
        self._file.write(payload)
        self._hash.update(payload)

    def add(self, observation: EnsembleCalibrationObservation) -> None:
        if self._closed:
            raise ValueError("ensemble calibration sidecar stream is closed")
        self._counts[observation.color] += 1
        self._write(
            _canonical_compact(
                {
                    "record_type": "observation",
                    "observation_index": self._index,
                    "color": observation.color,
                    "ensemble_fused_logits": list(observation.example.logits),
                    "hard_eliminated": list(observation.example.eliminated),
                    "true_index": observation.example.true_index,
                }
            )
        )
        self._index += 1

    def finalize(self, *, recover_exact: bool = False) -> ContentAddressedFile:
        if self._closed:
            raise ValueError("ensemble calibration sidecar stream is closed")
        if any(count == 0 for count in self._counts.values()):
            error = ValueError("sidecar requires White and Black observations")
            try:
                self.abort()
            except BaseException as cleanup_error:
                error.add_note(
                    "ensemble calibration sidecar cleanup also failed: "
                    f"{cleanup_error!r}"
                )
            raise error
        self._file.flush()
        os.fsync(self._file.fileno())
        self._file.close()
        expected_sha256 = self._hash.hexdigest()
        try:
            publish_staged_file_durable(
                self.output,
                self._temporary,
                expected_sha256,
                label="ensemble calibration sidecar",
                recover_exact=recover_exact,
            )
        except FileExistsError as error:
            self._closed = True
            raise ValueError(
                "refusing to overwrite ensemble calibration sidecar: "
                f"{self.output}"
            ) from error
        except BaseException:
            self._closed = True
            raise
        self._closed = True
        return ContentAddressedFile(self.output, expected_sha256)

    def abort(self) -> None:
        if not self._closed:
            try:
                abort_staged_file_safely(
                    self._temporary,
                    self._file,
                    label="ensemble calibration sidecar",
                )
            finally:
                self._closed = True


def identity_from_release(
    ensemble_release: ContentAddressedJson,
    *,
    calibration_seed_sha256: str,
    symbolic_schema_sha256: str,
    fusion_selection: ContentAddressedJson,
    training_corpus_set: Mapping[str, str] | None = None,
) -> EnsembleCalibrationIdentity:
    """Recursively verify release and fusion policy, then derive the identity."""

    loaded = verify_ensemble_release(ensemble_release)
    loaded_fusion = load_fusion_selection_artifact(
        fusion_selection,
        expected_identity=FusionSelectionIdentity(
            ensemble_release_sha256=ensemble_release.sha256,
            private_validation_manifest_sha256=(
                loaded.private_validation_manifest_sha256
            ),
            validation_dataset_sha256=loaded.validation_dataset_sha256,
            validation_seed_sha256=loaded.partition_seed_sha256,
            training_corpus_set_sha256=loaded.training_corpus_set_sha256,
            symbolic_schema_sha256=symbolic_schema_sha256,
        ),
    )
    return _identity_from_loaded(
        loaded,
        ensemble_release.sha256,
        calibration_seed_sha256,
        symbolic_schema_sha256,
        fusion_selection.sha256,
        loaded_fusion.selected_alpha,
        training_corpus_set,
    )


def write_ensemble_calibration_sidecar(
    output: Path,
    identity: EnsembleCalibrationIdentity,
    observations: Iterable[EnsembleCalibrationObservation],
) -> ContentAddressedFile:
    stream = EnsembleCalibrationSidecarStream(output, identity)
    try:
        for observation in observations:
            stream.add(observation)
        return stream.finalize()
    except BaseException as error:
        try:
            stream.abort()
        except BaseException as cleanup_error:
            error.add_note(
                "ensemble calibration sidecar cleanup also failed: "
                f"{cleanup_error!r}"
            )
        raise


def load_ensemble_calibration_sidecar(
    reference: ContentAddressedFile,
) -> LoadedEnsembleCalibrationSidecar:
    payload = _verified_bytes(reference.path, reference.sha256, "sidecar")
    if not payload.endswith(b"\n") or b"\r" in payload:
        raise ValueError("sidecar must use canonical LF-terminated records")
    raw_lines = payload[:-1].split(b"\n")
    if len(raw_lines) < 3 or any(not line for line in raw_lines):
        raise ValueError("sidecar must contain a header and both color heads")
    values = [_strict_json(line, "sidecar record") for line in raw_lines]
    for raw, value in zip(raw_lines, values, strict=True):
        if raw != _canonical_compact(value):
            raise ValueError("sidecar record bytes are not canonical")
    identity = _load_header(values[0])
    white: list[CalibrationExample] = []
    black: list[CalibrationExample] = []
    for index, value in enumerate(values[1:]):
        _exact_keys(
            value,
            {
                "record_type",
                "observation_index",
                "color",
                "ensemble_fused_logits",
                "hard_eliminated",
                "true_index",
            },
            "sidecar observation",
        )
        if value["record_type"] != "observation":
            raise ValueError("sidecar contains a non-observation record")
        if value["observation_index"] != index:
            raise ValueError("sidecar observation indices are not contiguous")
        color = value["color"]
        if color not in {"white", "black"}:
            raise ValueError("sidecar observation color is invalid")
        logits = value["ensemble_fused_logits"]
        mask = value["hard_eliminated"]
        if (
            not isinstance(logits, list)
            or not isinstance(mask, list)
            or len(logits) != CLASS_COUNT
            or len(mask) != CLASS_COUNT
        ):
            raise ValueError("sidecar observation dimensions are invalid")
        example = CalibrationExample(
            tuple(_finite_float(item, "fused logit") for item in logits),
            _index(value["true_index"], "true index"),
            tuple(mask),
        )
        (white if color == "white" else black).append(example)
    if not white or not black:
        raise ValueError("sidecar requires White and Black observations")
    return LoadedEnsembleCalibrationSidecar(
        identity, tuple(white), tuple(black), reference
    )


def write_ensemble_calibration_receipt(
    output: Path,
    *,
    report: ContentAddressedFile,
    sidecar: ContentAddressedFile,
    ensemble_release: ContentAddressedJson,
    fusion_selection: ContentAddressedJson,
    recover_exact: bool = False,
) -> ContentAddressedFile:
    """Publish a non-circular receipt after independently verifying all inputs."""

    _require_sibling(output.parent, report.path, "report")
    _require_sibling(output.parent, sidecar.path, "sidecar")
    _require_sibling(output.parent, ensemble_release.path, "ensemble release")
    _require_sibling(
        output.parent,
        fusion_selection.path,
        "fusion selection",
    )
    loaded_sidecar = load_ensemble_calibration_sidecar(sidecar)
    expected_identity = identity_from_release(
        ensemble_release,
        calibration_seed_sha256=loaded_sidecar.identity.calibration_seed_sha256,
        symbolic_schema_sha256=loaded_sidecar.identity.symbolic_schema_sha256,
        fusion_selection=fusion_selection,
        training_corpus_set=loaded_sidecar.identity.training_corpus_set,
    )
    if loaded_sidecar.identity != expected_identity:
        raise ValueError("sidecar identity disagrees with ensemble release")
    _verify_report(report, sidecar, expected_identity)
    value = {
        "format": RECEIPT_FORMAT,
        "version": FORMAT_VERSION,
        "report": _binding(report),
        "sidecar": _binding(sidecar),
        "ensemble_release": _binding(
            ContentAddressedFile(
                ensemble_release.path, ensemble_release.sha256
            )
        ),
        "fusion_selection": _binding(
            ContentAddressedFile(
                fusion_selection.path,
                fusion_selection.sha256,
            )
        ),
        "identity": _identity_value(expected_identity),
    }
    payload = _canonical_pretty(value)
    if recover_exact:
        publish_bytes_durable_exact(
            output,
            payload,
            label="ensemble calibration receipt",
        )
    else:
        _write_atomic_no_clobber(output, payload, "ensemble calibration receipt")
    return ContentAddressedFile(output, hashlib.sha256(payload).hexdigest())


def load_ensemble_calibration_receipt(
    reference: ContentAddressedFile,
) -> EnsembleCalibrationReceipt:
    value = _load_canonical_document(reference, "receipt")
    _exact_keys(
        value,
        {
            "format",
            "version",
            "report",
            "sidecar",
            "ensemble_release",
            "fusion_selection",
            "identity",
        },
        "receipt",
    )
    if value["format"] != RECEIPT_FORMAT or value["version"] != FORMAT_VERSION:
        raise ValueError("unsupported ensemble calibration receipt")
    directory = reference.path.parent
    report = _reference(directory, value["report"], "report")
    sidecar = _reference(directory, value["sidecar"], "sidecar")
    release_file = _reference(
        directory, value["ensemble_release"], "ensemble release"
    )
    ensemble_release = ContentAddressedJson(
        release_file.path, release_file.sha256
    )
    fusion_file = _reference(
        directory,
        value["fusion_selection"],
        "fusion selection",
    )
    fusion_selection = ContentAddressedJson(
        fusion_file.path,
        fusion_file.sha256,
    )
    loaded_sidecar = load_ensemble_calibration_sidecar(sidecar)
    declared = _load_identity(value["identity"])
    expected = identity_from_release(
        ensemble_release,
        calibration_seed_sha256=declared.calibration_seed_sha256,
        symbolic_schema_sha256=declared.symbolic_schema_sha256,
        fusion_selection=fusion_selection,
        training_corpus_set=declared.training_corpus_set,
    )
    if declared != expected or loaded_sidecar.identity != expected:
        raise ValueError("receipt, sidecar, and ensemble identities disagree")
    _verify_report(report, sidecar, expected)
    return EnsembleCalibrationReceipt(
        reference,
        report,
        sidecar,
        ensemble_release,
        fusion_selection,
        expected,
    )


def fit_ensemble_calibration(
    output: Path,
    receipt: ContentAddressedFile,
    *,
    recover_exact: bool = False,
) -> ContentAddressedFile:
    """Fit both validation heads and publish the final verified artifact."""

    loaded_receipt = load_ensemble_calibration_receipt(receipt)
    loaded_sidecar = load_ensemble_calibration_sidecar(loaded_receipt.sidecar)
    white = fit_validation_temperature(
        loaded_sidecar.white,
        split="validation",
        minimum_temperature=MINIMUM_CALIBRATION_TEMPERATURE,
        maximum_temperature=MAXIMUM_CALIBRATION_TEMPERATURE,
    )
    black = fit_validation_temperature(
        loaded_sidecar.black,
        split="validation",
        minimum_temperature=MINIMUM_CALIBRATION_TEMPERATURE,
        maximum_temperature=MAXIMUM_CALIBRATION_TEMPERATURE,
    )
    for name, fitted, examples in (
        ("white", white, loaded_sidecar.white),
        ("black", black, loaded_sidecar.black),
    ):
        if not fitted.nll_after < fitted.nll_before:
            raise ValueError(f"{name} head NLL did not improve")
        if not (
            MINIMUM_CALIBRATION_TEMPERATURE
            <= fitted.temperature
            <= MAXIMUM_CALIBRATION_TEMPERATURE
        ):
            raise ValueError(f"{name} temperature is outside release bounds")
        for example in examples:
            probabilities = apply_temperature_calibration(
                example.logits, example.eliminated, fitted
            )
            if any(
                probabilities[index] != 0.0
                for index, eliminated in enumerate(example.eliminated)
                if eliminated
            ):
                raise RuntimeError("calibration restored a hard elimination")
    value = {
        "format": ARTIFACT_FORMAT,
        "version": FORMAT_VERSION,
        "method": {
            "name": "per-head-multiclass-temperature-scaling",
            "fusion": FUSION_METHOD,
            "selected_alpha": loaded_receipt.identity.selected_alpha,
            "minimum_temperature": MINIMUM_CALIBRATION_TEMPERATURE,
            "maximum_temperature": MAXIMUM_CALIBRATION_TEMPERATURE,
            "preserves_hard_eliminations": True,
            "requires_each_head_nll_improvement": True,
        },
        "receipt": _binding(receipt),
        "report": _binding(loaded_receipt.report),
        "sidecar": _binding(loaded_receipt.sidecar),
        "ensemble_release": _binding(
            ContentAddressedFile(
                loaded_receipt.ensemble_release.path,
                loaded_receipt.ensemble_release.sha256,
            )
        ),
        "fusion_selection": _binding(
            ContentAddressedFile(
                loaded_receipt.fusion_selection.path,
                loaded_receipt.fusion_selection.sha256,
            )
        ),
        "identity": _identity_value(loaded_receipt.identity),
        "white": white.to_metadata(),
        "black": black.to_metadata(),
    }
    payload = _canonical_pretty(value)
    if recover_exact:
        publish_bytes_durable_exact(
            output,
            payload,
            label="ensemble calibration artifact",
        )
    else:
        _write_atomic_no_clobber(output, payload, "ensemble calibration artifact")
    return ContentAddressedFile(output, hashlib.sha256(payload).hexdigest())


def load_ensemble_calibration(
    reference: ContentAddressedFile,
) -> Mapping[str, object]:
    """Strictly reload and recursively verify a final artifact."""

    value = _load_canonical_document(reference, "calibration artifact")
    _exact_keys(
        value,
        {
            "format",
            "version",
            "method",
            "receipt",
            "report",
            "sidecar",
            "ensemble_release",
            "fusion_selection",
            "identity",
            "white",
            "black",
        },
        "calibration artifact",
    )
    if value["format"] != ARTIFACT_FORMAT or value["version"] != FORMAT_VERSION:
        raise ValueError("unsupported ensemble calibration artifact")
    receipt_ref = _reference(reference.path.parent, value["receipt"], "receipt")
    receipt = load_ensemble_calibration_receipt(receipt_ref)
    expected_bindings = {
        "report": _binding(receipt.report),
        "sidecar": _binding(receipt.sidecar),
        "ensemble_release": _binding(
            ContentAddressedFile(
                receipt.ensemble_release.path, receipt.ensemble_release.sha256
            )
        ),
        "fusion_selection": _binding(
            ContentAddressedFile(
                receipt.fusion_selection.path,
                receipt.fusion_selection.sha256,
            )
        ),
    }
    if any(value[name] != binding for name, binding in expected_bindings.items()):
        raise ValueError("calibration artifact input binding is inconsistent")
    if _load_identity(value["identity"]) != receipt.identity:
        raise ValueError("calibration artifact identity disagrees with receipt")
    method = _object(value["method"], "calibration method")
    expected_method = {
        "name": "per-head-multiclass-temperature-scaling",
        "fusion": FUSION_METHOD,
        "selected_alpha": receipt.identity.selected_alpha,
        "minimum_temperature": MINIMUM_CALIBRATION_TEMPERATURE,
        "maximum_temperature": MAXIMUM_CALIBRATION_TEMPERATURE,
        "preserves_hard_eliminations": True,
        "requires_each_head_nll_improvement": True,
    }
    if dict(method) != expected_method:
        raise ValueError("calibration method is not canonical")
    sidecar = load_ensemble_calibration_sidecar(receipt.sidecar)
    expected_heads = {
        "white": fit_validation_temperature(
            sidecar.white,
            split="validation",
            minimum_temperature=MINIMUM_CALIBRATION_TEMPERATURE,
            maximum_temperature=MAXIMUM_CALIBRATION_TEMPERATURE,
        ),
        "black": fit_validation_temperature(
            sidecar.black,
            split="validation",
            minimum_temperature=MINIMUM_CALIBRATION_TEMPERATURE,
            maximum_temperature=MAXIMUM_CALIBRATION_TEMPERATURE,
        ),
    }
    for head, expected_fit in expected_heads.items():
        metadata = _object(value[head], f"{head} calibration")
        if dict(metadata) != expected_fit.to_metadata():
            raise ValueError(
                f"{head} calibration does not match the bound sidecar"
            )
        temperature = _finite_float(metadata.get("temperature"), "temperature")
        before = _finite_float(metadata.get("nll_before"), "nll before")
        after = _finite_float(metadata.get("nll_after"), "nll after")
        if not MINIMUM_CALIBRATION_TEMPERATURE <= temperature <= MAXIMUM_CALIBRATION_TEMPERATURE:
            raise ValueError("calibration temperature is outside release bounds")
        if not after < before:
            raise ValueError("calibration head NLL did not improve")
        if metadata.get("preserves_hard_eliminations") is not True:
            raise ValueError("calibration does not preserve hard eliminations")
    return value


def _identity_from_loaded(
    release: LoadedEnsembleRelease,
    ensemble_sha256: str,
    calibration_seed_sha256: str,
    symbolic_schema_sha256: str,
    fusion_selection_sha256: str,
    selected_alpha: float,
    training_corpus_set: Mapping[str, str] | None,
) -> EnsembleCalibrationIdentity:
    return EnsembleCalibrationIdentity(
        ensemble_release_sha256=ensemble_sha256,
        fusion_selection_sha256=fusion_selection_sha256,
        selected_alpha=selected_alpha,
        members=tuple(
            EnsembleCalibrationMember(
                seed=member.training_seed,
                selection_sha256=member.selection_sha256,
                training_claim_sha256=member.training_run_sha256,
                training_run_id=member.training_run_id,
                checkpoint_sha256=member.checkpoint_sha256,
                checkpoint_epoch=member.checkpoint_epoch,
            )
            for member in release.members
        ),
        release_root_sha256=release.release_root_sha256,
        corpus_run_id=release.corpus_run_id,
        private_validation_manifest_sha256=(
            release.private_validation_manifest_sha256
        ),
        validation_dataset_sha256=release.validation_dataset_sha256,
        training_corpus_set_sha256=release.training_corpus_set_sha256,
        calibration_seed_sha256=calibration_seed_sha256,
        symbolic_schema_sha256=symbolic_schema_sha256,
        training_corpus_set=training_corpus_set,
    )


def _header_value(identity: EnsembleCalibrationIdentity) -> dict[str, object]:
    return {
        "record_type": "header",
        "format": SIDECAR_FORMAT,
        "version": FORMAT_VERSION,
        "partition": {
            "identity": VALIDATION_PARTITION_IDENTITY,
            "name": ValidationPartition.CALIBRATION_FIT.value,
            "seed_sha256": identity.calibration_seed_sha256,
        },
        "symbolic": {
            "feature_version": SYMBOLIC_FEATURE_VERSION,
            "schema_sha256": identity.symbolic_schema_sha256,
            "class_count": CLASS_COUNT,
        },
        "fusion": FUSION_METHOD,
        "identity": _identity_value(identity, include_partition=False),
    }


def _identity_value(
    identity: EnsembleCalibrationIdentity,
    *,
    include_partition: bool = True,
) -> dict[str, object]:
    value: dict[str, object] = {
        "ensemble_release_sha256": identity.ensemble_release_sha256,
        "fusion_selection_sha256": identity.fusion_selection_sha256,
        "selected_alpha": identity.selected_alpha,
        "members": [
            {
                "seed": member.seed,
                "selection_sha256": member.selection_sha256,
                "training_claim_sha256": member.training_claim_sha256,
                "training_run_id": member.training_run_id,
                "checkpoint_sha256": member.checkpoint_sha256,
                "checkpoint_epoch": member.checkpoint_epoch,
            }
            for member in identity.members
        ],
        "release_root_sha256": identity.release_root_sha256,
        "corpus_run_id": identity.corpus_run_id,
        "private_validation_manifest_sha256": (
            identity.private_validation_manifest_sha256
        ),
        "validation_dataset_sha256": identity.validation_dataset_sha256,
        "training_corpus_set_sha256": (
            identity.training_corpus_set_sha256
        ),
        "training_corpus_set": _corpus_set_value(identity.training_corpus_set),
    }
    if include_partition:
        value.update(
            {
                "calibration_seed_sha256": identity.calibration_seed_sha256,
                "symbolic_schema_sha256": identity.symbolic_schema_sha256,
                "symbolic_feature_version": SYMBOLIC_FEATURE_VERSION,
                "class_count": CLASS_COUNT,
                "fusion": FUSION_METHOD,
                "partition_identity": VALIDATION_PARTITION_IDENTITY,
                "partition_name": ValidationPartition.CALIBRATION_FIT.value,
            }
        )
    return value


def _load_header(value: Mapping[str, object]) -> EnsembleCalibrationIdentity:
    _exact_keys(
        value,
        {
            "record_type",
            "format",
            "version",
            "partition",
            "symbolic",
            "fusion",
            "identity",
        },
        "sidecar header",
    )
    if (
        value["record_type"] != "header"
        or value["format"] != SIDECAR_FORMAT
        or value["version"] != FORMAT_VERSION
    ):
        raise ValueError("unsupported ensemble sidecar header")
    partition = _object(value["partition"], "sidecar partition")
    if dict(partition) != {
        "identity": VALIDATION_PARTITION_IDENTITY,
        "name": ValidationPartition.CALIBRATION_FIT.value,
        "seed_sha256": partition.get("seed_sha256"),
    }:
        raise ValueError("sidecar partition is invalid")
    symbolic = _object(value["symbolic"], "sidecar symbolic schema")
    if (
        set(symbolic) != {"feature_version", "schema_sha256", "class_count"}
        or symbolic["feature_version"] != SYMBOLIC_FEATURE_VERSION
        or symbolic["class_count"] != CLASS_COUNT
        or value["fusion"] != FUSION_METHOD
    ):
        raise ValueError("sidecar model contract is invalid")
    base = _load_identity(
        {
            **dict(_object(value["identity"], "sidecar identity")),
            "calibration_seed_sha256": partition["seed_sha256"],
            "symbolic_schema_sha256": symbolic["schema_sha256"],
            "symbolic_feature_version": symbolic["feature_version"],
            "class_count": symbolic["class_count"],
            "fusion": value["fusion"],
            "partition_identity": partition["identity"],
            "partition_name": partition["name"],
        }
    )
    return base


def _load_identity(value: object) -> EnsembleCalibrationIdentity:
    identity = _object(value, "calibration identity")
    _exact_keys(
        identity,
        {
            "ensemble_release_sha256",
            "fusion_selection_sha256",
            "selected_alpha",
            "members",
            "release_root_sha256",
            "corpus_run_id",
            "private_validation_manifest_sha256",
            "validation_dataset_sha256",
            "training_corpus_set_sha256",
            "training_corpus_set",
            "calibration_seed_sha256",
            "symbolic_schema_sha256",
            "symbolic_feature_version",
            "class_count",
            "fusion",
            "partition_identity",
            "partition_name",
        },
        "calibration identity",
    )
    if (
        identity["symbolic_feature_version"] != SYMBOLIC_FEATURE_VERSION
        or identity["class_count"] != CLASS_COUNT
        or identity["fusion"] != FUSION_METHOD
        or identity["partition_identity"] != VALIDATION_PARTITION_IDENTITY
        or identity["partition_name"] != ValidationPartition.CALIBRATION_FIT.value
    ):
        raise ValueError("calibration identity contract is invalid")
    members_value = identity["members"]
    if not isinstance(members_value, list) or len(members_value) != 3:
        raise ValueError("calibration identity requires exactly three members")
    members: list[EnsembleCalibrationMember] = []
    for item in members_value:
        member = _object(item, "calibration member")
        _exact_keys(
            member,
            {
                "seed",
                "selection_sha256",
                "training_claim_sha256",
                "training_run_id",
                "checkpoint_sha256",
                "checkpoint_epoch",
            },
            "calibration member",
        )
        members.append(
            EnsembleCalibrationMember(
                seed=_nonnegative_int(member["seed"], "member seed"),
                selection_sha256=member["selection_sha256"],  # type: ignore[arg-type]
                training_claim_sha256=member["training_claim_sha256"],  # type: ignore[arg-type]
                training_run_id=member["training_run_id"],  # type: ignore[arg-type]
                checkpoint_sha256=member["checkpoint_sha256"],  # type: ignore[arg-type]
                checkpoint_epoch=member["checkpoint_epoch"],  # type: ignore[arg-type]
            )
        )
    return EnsembleCalibrationIdentity(
        ensemble_release_sha256=identity["ensemble_release_sha256"],  # type: ignore[arg-type]
        fusion_selection_sha256=identity["fusion_selection_sha256"],  # type: ignore[arg-type]
        selected_alpha=_finite_float(
            identity["selected_alpha"],
            "selected fusion alpha",
        ),
        members=tuple(members),
        release_root_sha256=identity["release_root_sha256"],  # type: ignore[arg-type]
        corpus_run_id=identity["corpus_run_id"],  # type: ignore[arg-type]
        private_validation_manifest_sha256=identity["private_validation_manifest_sha256"],  # type: ignore[arg-type]
        validation_dataset_sha256=identity["validation_dataset_sha256"],  # type: ignore[arg-type]
        training_corpus_set_sha256=identity["training_corpus_set_sha256"],  # type: ignore[arg-type]
        calibration_seed_sha256=identity["calibration_seed_sha256"],  # type: ignore[arg-type]
        symbolic_schema_sha256=identity["symbolic_schema_sha256"],  # type: ignore[arg-type]
        training_corpus_set=_load_corpus_set(identity["training_corpus_set"]),
    )


def _corpus_set_value(mapping: Mapping[str, str] | None) -> object:
    if mapping is None:
        return None
    normalized = _corpus_mapping(mapping)
    material = _canonical_compact(normalized)
    return {
        "mapping": normalized,
        "sha256": hashlib.sha256(material).hexdigest(),
    }


def _load_corpus_set(value: object) -> Mapping[str, str] | None:
    if value is None:
        return None
    wrapper = _object(value, "training corpus set")
    _exact_keys(wrapper, {"mapping", "sha256"}, "training corpus set")
    mapping_value = _object(wrapper["mapping"], "training corpus set mapping")
    mapping = _corpus_mapping(mapping_value)
    expected = hashlib.sha256(_canonical_compact(mapping)).hexdigest()
    if _digest(wrapper["sha256"], "training corpus set sha256") != expected:
        raise ValueError("training corpus set hash is invalid")
    return mapping


def _corpus_mapping(value: Mapping[object, object]) -> dict[str, str]:
    if not value:
        raise ValueError("training corpus set mapping must not be empty")
    result: dict[str, str] = {}
    for key, digest in value.items():
        if (
            not isinstance(key, str)
            or not key
            or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in key)
        ):
            raise ValueError("training corpus set keys must be canonical identifiers")
        result[key] = _digest(digest, f"training corpus set {key}")
    return dict(sorted(result.items()))


def _binding(reference: ContentAddressedFile) -> dict[str, str]:
    return {"file": reference.path.name, "sha256": reference.sha256}


def _verify_report(
    report: ContentAddressedFile,
    sidecar: ContentAddressedFile,
    identity: EnsembleCalibrationIdentity,
) -> None:
    value = _load_canonical_document(report, "report")
    _exact_keys(
        value,
        {"format", "version", "evaluation", "identity", "metrics"},
        "ensemble evaluation report",
    )
    if value["format"] != REPORT_FORMAT or value["version"] != FORMAT_VERSION:
        raise ValueError("unsupported ensemble evaluation report")
    evaluation = _object(value["evaluation"], "report evaluation")
    _exact_keys(
        evaluation,
        {"validation_partition", "calibration_sidecar"},
        "report evaluation",
    )
    partition = _object(
        evaluation["validation_partition"], "report validation partition"
    )
    if dict(partition) != {
        "identity": VALIDATION_PARTITION_IDENTITY,
        "name": ValidationPartition.CALIBRATION_FIT.value,
        "seed_sha256": identity.calibration_seed_sha256,
    }:
        raise ValueError("report validation partition is inconsistent")
    if evaluation["calibration_sidecar"] != _binding(sidecar):
        raise ValueError("report calibration sidecar binding is inconsistent")
    if _load_identity(value["identity"]) != identity:
        raise ValueError("report calibration identity is inconsistent")
    _object(value["metrics"], "report metrics")


def _reference(directory: Path, value: object, name: str) -> ContentAddressedFile:
    binding = _object(value, f"{name} binding")
    _exact_keys(binding, {"file", "sha256"}, f"{name} binding")
    filename = binding["file"]
    if not is_portable_safe_basename(filename):
        raise ValueError(f"{name} file must be a safe basename")
    path = directory / filename
    _require_sibling(directory, path, name)
    return ContentAddressedFile(path, _digest(binding["sha256"], f"{name} sha256"))


def _load_canonical_document(
    reference: ContentAddressedFile, name: str
) -> Mapping[str, object]:
    payload = _verified_bytes(reference.path, reference.sha256, name)
    value = _strict_json(payload, name)
    if payload != _canonical_pretty(value):
        raise ValueError(f"{name} JSON bytes are not canonical")
    return value


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


def _canonical_compact(value: Mapping[object, object]) -> bytes:
    return json.dumps(
        value, separators=(",", ":"), sort_keys=True, allow_nan=False
    ).encode("utf-8")


def _canonical_pretty(value: Mapping[object, object]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_atomic_no_clobber(path: Path, payload: bytes, name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        publish_bytes_durable(path, payload)
    except FileExistsError as error:
        raise ValueError(f"refusing to overwrite {name}: {path}") from error


def _require_sibling(directory: Path, path: Path, name: str) -> None:
    if path.is_symlink() or path.parent.resolve() != directory.resolve():
        raise ValueError(f"{name} must be a sibling file")
    if not path.is_file():
        raise ValueError(f"{name} file is missing")


def _exact_keys(
    value: Mapping[object, object], expected: set[str], name: str
) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} fields are not canonical")


def _object(value: object, name: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_int(value: object, name: str) -> int:
    result = _nonnegative_int(value, name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _index(value: object, name: str) -> int:
    return _nonnegative_int(value, name)


def _require_finite(value: object, name: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} contains a non-finite number")
    if isinstance(value, Mapping):
        for item in value.values():
            _require_finite(item, name)
    elif isinstance(value, list):
        for item in value:
            _require_finite(item, name)
