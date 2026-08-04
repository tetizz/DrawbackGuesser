"""Content-addressed calibration observations and immutable release artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, Iterable, Mapping

from ml.training.drawback_ml.durable_publish import (
    abort_staged_file_safely,
    publish_bytes_durable,
    publish_staged_file_durable,
)

from .calibration import (
    MAXIMUM_CALIBRATION_TEMPERATURE,
    MINIMUM_CALIBRATION_TEMPERATURE,
    CalibrationExample,
    apply_temperature_calibration,
    fit_validation_temperature,
)
from .validation_partition import (
    VALIDATION_PARTITION_IDENTITY,
    ValidationPartition,
)
from .release_selection_bundle import (
    ContentAddressedJson,
    verify_release_selection_bundle,
)

if TYPE_CHECKING:
    from .calibration_receipt import CalibrationReceiptReference


CALIBRATION_SIDECAR_FORMAT_VERSION = 1
CALIBRATION_ARTIFACT_FORMAT_VERSION = 1


def canonical_symbolic_schema_sha256(
    feature_version: int,
    rule_ids: Iterable[str],
) -> str:
    material = _canonical_json(
        {
            "feature_version": feature_version,
            "rule_ids": list(rule_ids),
        }
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


@dataclass(frozen=True)
class ContentAddressedFile:
    path: Path
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "sha256", _digest(self.sha256, "file sha256"))


@dataclass(frozen=True)
class CalibrationSidecarHeader:
    calibration_seed_sha256: str
    checkpoint_sha256: str
    selection_artifact_sha256: str
    symbolic_schema_sha256: str
    symbolic_feature_version: int
    class_count: int

    def __post_init__(self) -> None:
        for field_name in (
            "calibration_seed_sha256",
            "checkpoint_sha256",
            "selection_artifact_sha256",
            "symbolic_schema_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _digest(getattr(self, field_name), field_name),
            )
        _positive_int(
            self.symbolic_feature_version,
            "symbolic_feature_version",
        )
        _positive_int(self.class_count, "class_count")


@dataclass(frozen=True)
class CalibrationObservation:
    color: str
    example: CalibrationExample

    def __post_init__(self) -> None:
        if self.color not in {"white", "black"}:
            raise ValueError("calibration observation color must be white or black")


@dataclass(frozen=True)
class LoadedCalibrationSidecar:
    header: CalibrationSidecarHeader
    white: tuple[CalibrationExample, ...]
    black: tuple[CalibrationExample, ...]
    sha256: str


class CalibrationSidecarStream:
    """Bounded-memory atomic sidecar sink for held-out evaluation."""

    def __init__(self, output: Path, header: CalibrationSidecarHeader) -> None:
        self.output = output
        self.header = header
        output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
        )
        self._temporary = Path(temporary_name)
        self._file = os.fdopen(
            descriptor, "w", encoding="utf-8", newline="\n"
        )
        self._hash = hashlib.sha256()
        self._index = 0
        self._counts = {"white": 0, "black": 0}
        self._closed = False
        self._write(
            _canonical_json(
                {
                    "record_type": "header",
                    "format_version": CALIBRATION_SIDECAR_FORMAT_VERSION,
                    "partition": {
                        "identity": VALIDATION_PARTITION_IDENTITY,
                        "name": ValidationPartition.CALIBRATION_FIT.value,
                        "seed_sha256": header.calibration_seed_sha256,
                    },
                    "checkpoint_sha256": header.checkpoint_sha256,
                    "selection_artifact_sha256": header.selection_artifact_sha256,
                    "symbolic_schema_sha256": header.symbolic_schema_sha256,
                    "symbolic_feature_version": header.symbolic_feature_version,
                    "class_count": header.class_count,
                }
            )
        )

    def _write(self, line: str) -> None:
        rendered = line + "\n"
        self._file.write(rendered)
        self._hash.update(rendered.encode("utf-8"))

    def add(self, observation: CalibrationObservation) -> None:
        if self._closed:
            raise ValueError("calibration sidecar stream is closed")
        if len(observation.example.logits) != self.header.class_count:
            raise ValueError("calibration observation class count is invalid")
        self._counts[observation.color] += 1
        self._write(
            _canonical_json(
                {
                    "record_type": "observation",
                    "observation_index": self._index,
                    "color": observation.color,
                    "fused_logits": list(observation.example.logits),
                    "hard_eliminated": list(observation.example.eliminated),
                    "true_index": observation.example.true_index,
                }
            )
        )
        self._index += 1

    def finalize(self, *, recover_exact: bool = False) -> ContentAddressedFile:
        if self._closed:
            raise ValueError("calibration sidecar stream is closed")
        if any(count == 0 for count in self._counts.values()):
            error = ValueError(
                "calibration sidecar requires White and Black observations"
            )
            try:
                self.abort()
            except BaseException as cleanup_error:
                error.add_note(
                    "calibration sidecar cleanup also failed: "
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
                label="calibration observation sidecar",
                recover_exact=recover_exact,
            )
        except FileExistsError as error:
            self._closed = True
            raise ValueError(
                "refusing to overwrite calibration observation sidecar: "
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
                    label="calibration sidecar",
                )
            finally:
                self._closed = True


def write_calibration_observation_sidecar(
    output: Path,
    header: CalibrationSidecarHeader,
    observations: Iterable[CalibrationObservation],
) -> ContentAddressedFile:
    """Publish actual fused logits and masks without deriving them from probabilities."""

    lines = [
        _canonical_json(
            {
                "record_type": "header",
                "format_version": CALIBRATION_SIDECAR_FORMAT_VERSION,
                "partition": {
                    "identity": VALIDATION_PARTITION_IDENTITY,
                    "name": ValidationPartition.CALIBRATION_FIT.value,
                    "seed_sha256": header.calibration_seed_sha256,
                },
                "checkpoint_sha256": header.checkpoint_sha256,
                "selection_artifact_sha256": (
                    header.selection_artifact_sha256
                ),
                "symbolic_schema_sha256": header.symbolic_schema_sha256,
                "symbolic_feature_version": (
                    header.symbolic_feature_version
                ),
                "class_count": header.class_count,
            }
        )
    ]
    counts = {"white": 0, "black": 0}
    for index, observation in enumerate(observations):
        if len(observation.example.logits) != header.class_count:
            raise ValueError(
                "calibration observation class count disagrees with header"
            )
        counts[observation.color] += 1
        lines.append(
            _canonical_json(
                {
                    "record_type": "observation",
                    "observation_index": index,
                    "color": observation.color,
                    "fused_logits": list(observation.example.logits),
                    "hard_eliminated": list(
                        observation.example.eliminated
                    ),
                    "true_index": observation.example.true_index,
                }
            )
        )
    if any(count == 0 for count in counts.values()):
        raise ValueError(
            "calibration sidecar requires White and Black observations"
        )
    rendered = "\n".join(lines) + "\n"
    _write_atomic_no_clobber(
        output,
        rendered,
        "calibration observation sidecar",
    )
    digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
    if hashlib.sha256(output.read_bytes()).hexdigest() != digest:
        raise ValueError("published calibration sidecar bytes changed")
    return ContentAddressedFile(output, digest)


def load_calibration_observation_sidecar(
    reference: ContentAddressedFile,
) -> LoadedCalibrationSidecar:
    try:
        payload = reference.path.read_bytes()
    except OSError as error:
        raise ValueError(
            f"cannot read calibration sidecar: {reference.path}"
        ) from error
    actual = hashlib.sha256(payload).hexdigest()
    if actual != reference.sha256:
        raise ValueError("calibration sidecar sha256 does not match")
    raw_lines = payload.splitlines()
    if not raw_lines:
        raise ValueError("calibration sidecar is empty")
    header_value = _strict_json_object(raw_lines[0], "calibration header")
    _exact_keys(
        header_value,
        {
            "record_type",
            "format_version",
            "partition",
            "checkpoint_sha256",
            "selection_artifact_sha256",
            "symbolic_schema_sha256",
            "symbolic_feature_version",
            "class_count",
        },
        "calibration header",
    )
    if (
        header_value["record_type"] != "header"
        or header_value["format_version"]
        != CALIBRATION_SIDECAR_FORMAT_VERSION
    ):
        raise ValueError("unsupported calibration sidecar header")
    partition = _object(header_value["partition"], "calibration partition")
    _exact_keys(
        partition,
        {"identity", "name", "seed_sha256"},
        "calibration partition",
    )
    if partition["identity"] != VALIDATION_PARTITION_IDENTITY:
        raise ValueError("calibration sidecar partition identity is invalid")
    if partition["name"] != ValidationPartition.CALIBRATION_FIT.value:
        raise ValueError("calibration fitting requires calibration-fit data only")
    header = CalibrationSidecarHeader(
        calibration_seed_sha256=_digest(
            partition["seed_sha256"], "calibration seed sha256"
        ),
        checkpoint_sha256=_digest(
            header_value["checkpoint_sha256"], "checkpoint sha256"
        ),
        selection_artifact_sha256=_digest(
            header_value["selection_artifact_sha256"],
            "selection artifact sha256",
        ),
        symbolic_schema_sha256=_digest(
            header_value["symbolic_schema_sha256"],
            "symbolic schema sha256",
        ),
        symbolic_feature_version=_positive_int(
            header_value["symbolic_feature_version"],
            "symbolic feature version",
        ),
        class_count=_positive_int(
            header_value["class_count"], "class count"
        ),
    )
    by_color: dict[str, list[CalibrationExample]] = {
        "white": [],
        "black": [],
    }
    for expected_index, raw_line in enumerate(raw_lines[1:]):
        value = _strict_json_object(
            raw_line,
            f"calibration observation {expected_index}",
        )
        _exact_keys(
            value,
            {
                "record_type",
                "observation_index",
                "color",
                "fused_logits",
                "hard_eliminated",
                "true_index",
            },
            "calibration observation",
        )
        if value["record_type"] != "observation":
            raise ValueError("calibration sidecar contains an unknown record")
        if value["observation_index"] != expected_index:
            raise ValueError("calibration observations are not contiguous")
        color = value["color"]
        if color not in by_color:
            raise ValueError("calibration observation color is invalid")
        logits = value["fused_logits"]
        eliminated = value["hard_eliminated"]
        if not isinstance(logits, list) or not isinstance(eliminated, list):
            raise ValueError("calibration logits and masks must be arrays")
        if len(logits) != header.class_count:
            raise ValueError("calibration observation class count is invalid")
        by_color[color].append(
            CalibrationExample(
                tuple(logits),
                value["true_index"],
                tuple(eliminated),
            )
        )
    if any(not values for values in by_color.values()):
        raise ValueError(
            "calibration sidecar requires White and Black observations"
        )
    return LoadedCalibrationSidecar(
        header=header,
        white=tuple(by_color["white"]),
        black=tuple(by_color["black"]),
        sha256=actual,
    )


def fit_calibration_release(
    *,
    sidecar: ContentAddressedFile,
    checkpoint: ContentAddressedFile,
    selection_artifact: ContentAddressedFile,
    training_run: ContentAddressedJson,
    calibration_receipt: "CalibrationReceiptReference",
    output: Path,
) -> Path:
    """Fit and publish separate head temperatures from calibration-fit only."""

    checkpoint_payload = _verified_bytes(checkpoint, "checkpoint")
    from .calibration_receipt import load_calibration_receipt

    loaded_receipt = load_calibration_receipt(calibration_receipt)
    receipt_inputs = loaded_receipt.inputs
    for name, actual, expected in (
        ("sidecar", sidecar, receipt_inputs.sidecar),
        ("checkpoint", checkpoint, receipt_inputs.checkpoint),
        (
            "selection artifact",
            selection_artifact,
            receipt_inputs.selection_artifact,
        ),
        ("training run", training_run, receipt_inputs.training_run),
    ):
        if (
            actual.path.resolve() != expected.path.resolve()
            or actual.sha256 != expected.sha256
        ):
            raise ValueError(
                f"calibration receipt binds a different {name}"
            )
    verified_selection = verify_release_selection_bundle(
        ContentAddressedJson(
            selection_artifact.path, selection_artifact.sha256
        ),
        training_run,
    )
    selected_artifact = verified_selection.artifact
    if selected_artifact.selected_checkpoint_sha256 != checkpoint.sha256:
        raise ValueError(
            "selected checkpoint digest disagrees with checkpoint bytes"
        )
    loaded = load_calibration_observation_sidecar(sidecar)
    if loaded.header.checkpoint_sha256 != checkpoint.sha256:
        raise ValueError("calibration sidecar binds a different checkpoint")
    if (
        loaded.header.selection_artifact_sha256
        != selection_artifact.sha256
    ):
        raise ValueError(
            "calibration sidecar binds a different selection artifact"
        )
    white = fit_validation_temperature(
        loaded.white,
        split="validation",
        minimum_temperature=MINIMUM_CALIBRATION_TEMPERATURE,
        maximum_temperature=MAXIMUM_CALIBRATION_TEMPERATURE,
    )
    black = fit_validation_temperature(
        loaded.black,
        split="validation",
        minimum_temperature=MINIMUM_CALIBRATION_TEMPERATURE,
        maximum_temperature=MAXIMUM_CALIBRATION_TEMPERATURE,
    )
    mean_before = (white.nll_before + black.nll_before) / 2.0
    mean_after = (white.nll_after + black.nll_after) / 2.0
    if not mean_after < mean_before:
        raise ValueError(
            "calibration is rejected because mean head NLL did not improve"
        )
    for examples, fitted in ((loaded.white, white), (loaded.black, black)):
        for example in examples:
            probabilities = apply_temperature_calibration(
                example.logits,
                example.eliminated,
                fitted,
            )
            if any(
                probabilities[index] != 0.0
                for index, eliminated in enumerate(example.eliminated)
                if eliminated
            ):
                raise RuntimeError(
                    "calibration restored a hard-eliminated hypothesis"
                )
    artifact = {
        "format_version": CALIBRATION_ARTIFACT_FORMAT_VERSION,
        "method": {
            "name": "per-head-multiclass-temperature-scaling",
            "minimum_temperature": MINIMUM_CALIBRATION_TEMPERATURE,
            "maximum_temperature": MAXIMUM_CALIBRATION_TEMPERATURE,
            "preserves_hard_eliminations": True,
        },
        "partition": {
            "identity": VALIDATION_PARTITION_IDENTITY,
            "name": ValidationPartition.CALIBRATION_FIT.value,
            "seed_sha256": loaded.header.calibration_seed_sha256,
        },
        "checkpoint": {
            "file": checkpoint.path.name,
            "sha256": checkpoint.sha256,
            "bytes": len(checkpoint_payload),
        },
        "selection_artifact": {
            "file": selection_artifact.path.name,
            "sha256": selection_artifact.sha256,
        },
        "training_run": {
            "file": training_run.path.name,
            "sha256": training_run.sha256,
            "run_id": verified_selection.training_run.run_id,
        },
        "calibration_receipt": {
            "file": calibration_receipt.path.name,
            "sha256": calibration_receipt.sha256,
        },
        "calibration_evaluation_report": {
            "file": receipt_inputs.evaluation_report.path.name,
            "sha256": receipt_inputs.evaluation_report.sha256,
        },
        "observation_sidecar": {
            "file": sidecar.path.name,
            "sha256": sidecar.sha256,
        },
        "symbolic_schema": {
            "feature_version": loaded.header.symbolic_feature_version,
            "sha256": loaded.header.symbolic_schema_sha256,
        },
        "white": white.to_metadata(),
        "black": black.to_metadata(),
        "mean_nll_before": mean_before,
        "mean_nll_after": mean_after,
    }
    rendered = (
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )
    _write_atomic_no_clobber(output, rendered, "calibration artifact")
    return output


def _verified_bytes(reference: ContentAddressedFile, name: str) -> bytes:
    try:
        payload = reference.path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read {name}: {reference.path}") from error
    if hashlib.sha256(payload).hexdigest() != reference.sha256:
        raise ValueError(f"{name} sha256 does not match")
    return payload


def _canonical_json(value: Mapping[str, object]) -> str:
    return json.dumps(
        value,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _strict_json_object(payload: bytes, name: str) -> Mapping[str, object]:
    def reject_constant(token: str) -> None:
        raise ValueError(f"{name} contains non-finite JSON constant {token}")

    try:
        value = json.loads(payload, parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not strict UTF-8 JSON") from error
    _require_finite(value, name)
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} root must be an object")
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


def _object(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _exact_keys(
    value: Mapping[object, object],
    expected: set[str],
    name: str,
) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} fields do not match format version 1")


def _digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _positive_int(value: object, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return value


def _write_atomic_no_clobber(
    path: Path,
    rendered: str,
    name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        publish_bytes_durable(path, rendered.encode("utf-8"))
    except FileExistsError as error:
        raise ValueError(f"refusing to overwrite {name}: {path}") from error
