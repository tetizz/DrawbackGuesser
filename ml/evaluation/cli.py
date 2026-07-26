"""Run measured validation/test evaluation from a real checkpoint and corpus."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields, is_dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Mapping

from ml.training.drawback_ml.checkpoint import (
    FUSION_GRID_DRAWBACK_OBJECTIVE,
    LEGACY_ADDITIVE_DRAWBACK_OBJECTIVE,
)
from ml.training.drawback_ml.inference import (
    InferenceOutput,
    load_checkpoint_predictor,
)
from ml.training.drawback_ml.records import TrainingExample
from ml.training.drawback_ml.ensemble import load_hybrid_ensemble
from ml.training.drawback_ml.corpus_contract import (
    AuditedPrivateCorpusLease,
    audit_corpus_split,
    audit_private_corpus_split,
    open_audited_private_corpus_split,
)
from ml.training.drawback_ml.symbolic_schema import SYMBOLIC_RULE_IDS
from ml.training.drawback_ml.symbolic_schema import SYMBOLIC_FEATURE_VERSION
from ml.training.drawback_ml.training_corpus_set import (
    verify_training_corpus_set,
)

from .runner import (
    evaluate_held_out,
    load_rule_families,
    read_ndjson,
    read_ndjson_stream,
)
from .browser_ensemble_artifact import export_browser_ensemble_artifact
from .calibration_release import (
    ContentAddressedFile,
    CalibrationSidecarHeader,
    CalibrationSidecarStream,
    canonical_symbolic_schema_sha256,
    fit_calibration_release,
)
from .calibration_receipt import (
    CalibrationReceiptInputs,
    CalibrationReceiptReference,
    write_calibration_receipt,
)
from .ensemble_calibration import (
    CLASS_COUNT as ENSEMBLE_CLASS_COUNT,
    FORMAT_VERSION as ENSEMBLE_CALIBRATION_VERSION,
    FUSION_METHOD as ENSEMBLE_FUSION_METHOD,
    REPORT_FORMAT as ENSEMBLE_REPORT_FORMAT,
    ContentAddressedFile as EnsembleContentAddressedFile,
    EnsembleCalibrationObservation,
    EnsembleCalibrationSidecarStream,
    fit_ensemble_calibration,
    identity_from_release,
    write_ensemble_calibration_receipt,
)
from .ensemble_release import (
    resolve_member_checkpoint,
    verify_ensemble_release,
    write_ensemble_release,
)
from .fusion_selection import (
    FusionSelectionAccumulator,
    FusionSelectionIdentity,
    FusionSelectionObservation,
    load_fusion_selection_artifact,
    write_fusion_selection_accumulator,
)
from .selection import (
    SELECTION_SUMMARY_FORMAT_VERSION,
    ContentAddressedSummary,
    FusionGridEpochScorer,
    fusion_grid_selection_objective_metadata,
    validate_fusion_grid_head_counts,
    write_selection_artifact,
)
from .release_selection_bundle import (
    ContentAddressedJson,
    load_training_run,
    verify_checkpoint_training_identity,
    verify_release_selection_bundle,
)
from .splits import SplitManifest
from .validation_partition import (
    VALIDATION_PARTITION_IDENTITY,
    ValidationPartition,
    assign_validation_partition,
    validation_seed_sha256,
)

EXPECTED_SPLIT_SALT = "drawbacktrainer-v1"
SUPPORTED_CORPUS_SCHEMA_VERSIONS = frozenset({4, 5, 6})
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
FULL_VALIDATION_PARTITION_NAME = "full-validation-non-selection"


@dataclass(frozen=True)
class ValidationEvaluationContext:
    metadata: Mapping[str, str]
    allowed_seeds: frozenset[int]
    game_assignments: Mapping[str, tuple[str, str]]


def _checkpoint_corpus_requirement(audited: Any) -> dict[str, object]:
    expected = audited.provenance()
    requirement = {
        "split": "train",
        "engine_fingerprint": expected["engine_fingerprint"],
        "evaluator_policy_id": expected["evaluator_policy_id"],
        "evaluator_policy_version": expected["evaluator_policy_version"],
        "rule_ids": expected["rule_ids"],
        "symbolic_feature_version": expected["symbolic_feature_version"],
    }
    if "release_root_sha256" in expected and "corpus_run_id" in expected:
        requirement.update(
            {
                "release_root_sha256": expected["release_root_sha256"],
                "corpus_run_id": expected["corpus_run_id"],
            }
        )
    else:
        requirement["manifest_sha256"] = expected["manifest_sha256"]
    return requirement


def _require_checkpoint_corpus_provenance(
    predictor: Any,
    audited: Any,
) -> None:
    provenance = getattr(predictor, "corpus_provenance", None)
    if not isinstance(provenance, Mapping):
        raise ValueError(
            "checkpoint lacks authenticated training-corpus provenance"
        )
    if (
        "training_corpus_set" in provenance
        or "training_corpus_set_sha256" in provenance
    ):
        if set(provenance) != {
            "training_corpus_set",
            "training_corpus_set_sha256",
        }:
            raise ValueError(
                "checkpoint aggregate training-corpus provenance fields are invalid"
            )
        try:
            corpus_set = verify_training_corpus_set(
                provenance["training_corpus_set"]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "checkpoint aggregate training-corpus provenance is invalid"
            ) from error
        if (
            corpus_set["sha256"]
            != provenance["training_corpus_set_sha256"]
        ):
            raise ValueError(
                "checkpoint aggregate training-corpus provenance hash does not match"
            )
        primary = corpus_set["primary"]
        if not isinstance(primary, Mapping):
            raise ValueError(
                "checkpoint aggregate training-corpus primary is invalid"
            )
        expected = audited.provenance()
        for key in (
            "release_root_sha256",
            "corpus_run_id",
            "engine_fingerprint",
            "evaluator_policy_id",
            "evaluator_policy_version",
            "symbolic_feature_version",
        ):
            if primary.get(key) != expected.get(key):
                raise ValueError(
                    "checkpoint aggregate training-corpus provenance "
                    f"does not match manifest: {key}"
                )
        return
    for key, value in _checkpoint_corpus_requirement(audited).items():
        if provenance.get(key) != value:
            raise ValueError(
                f"checkpoint corpus provenance does not match manifest: {key}"
            )


def _write_report_atomic_no_clobber(path: Path, rendered: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            output.write(rendered)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ValueError(f"refusing to overwrite evaluation report: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _non_empty_manifest_string(
    value: Mapping[str, Any],
    key: str,
) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"corpus manifest {key} must be a non-empty string")
    return item


def _validate_schema_five_evaluator_manifest(
    value: Mapping[str, Any],
) -> None:
    if value.get("evaluatorCoverage") != "uniform-required":
        raise ValueError(
            "evaluator corpus manifest requires uniform evaluator coverage"
        )
    _non_empty_manifest_string(value, "evaluatorPolicyId")
    policy_version = value.get("evaluatorPolicyVersion")
    if (
        isinstance(policy_version, bool)
        or not isinstance(policy_version, int)
        or policy_version <= 0
    ):
        raise ValueError(
            "corpus manifest evaluatorPolicyVersion must be a positive integer"
        )
    _non_empty_manifest_string(value, "engineFingerprint")
    binary_digest = value.get("engineBinarySha256")
    if (
        not isinstance(binary_digest, str)
        or SHA256_PATTERN.fullmatch(binary_digest) is None
    ):
        raise ValueError(
            "corpus manifest engineBinarySha256 must be a lowercase SHA-256 digest"
        )
    for key in (
        "evaluatorRequestSchemaVersion",
        "evaluatorCacheSchemaVersion",
    ):
        schema = value.get(key)
        if (
            isinstance(schema, bool)
            or not isinstance(schema, int)
            or schema != 1
        ):
            raise ValueError(f"corpus manifest {key} must be 1")
    search_limit = value.get("evaluatorSearchLimit")
    if not isinstance(search_limit, Mapping):
        raise ValueError(
            "corpus manifest evaluatorSearchLimit must be a canonical object"
        )
    expected_limit_keys = {"kind", "value"}
    if set(search_limit) != expected_limit_keys:
        raise ValueError(
            "corpus manifest evaluatorSearchLimit must contain only kind and value"
        )
    if search_limit != {"kind": "nodes", "value": 10_000}:
        raise ValueError(
            "corpus manifest evaluatorSearchLimit must be exactly 10,000 nodes"
        )


def _split_manifest_mapping(
    value: Mapping[str, Any],
) -> Mapping[str, list[int]]:
    """Accept either a bare seed manifest or the simulator corpus manifest."""

    if all(isinstance(value.get(name), list) for name in ("train", "validation", "test")):
        return {
            name: value[name]
            for name in ("train", "validation", "test")
        }
    schema_version = value.get("schemaVersion")
    if (
        schema_version is not None
        and (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version not in SUPPORTED_CORPUS_SCHEMA_VERSIONS
        )
    ):
        supported = ", ".join(
            str(version) for version in sorted(SUPPORTED_CORPUS_SCHEMA_VERSIONS)
        )
        raise ValueError(
            f"unsupported corpus manifest schemaVersion; expected {supported}"
        )
    if schema_version in {5, 6}:
        _validate_schema_five_evaluator_manifest(value)
    if (
        schema_version == 6
        and value.get("observationPolicy")
        != "single-attempt-allow-partial-v1"
    ):
        raise ValueError(
            "corpus manifest observationPolicy is unsupported"
        )
    split_salt = value.get("splitSalt")
    if split_salt != EXPECTED_SPLIT_SALT:
        raise ValueError(
            f"corpus manifest splitSalt must be {EXPECTED_SPLIT_SALT}"
        )
    splits = value.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("corpus manifest must contain a splits object")
    result: dict[str, list[int]] = {}
    for name in ("train", "validation", "test"):
        entry = splits.get(name)
        if not isinstance(entry, Mapping):
            raise ValueError(f"corpus manifest split {name} must be an object")
        seeds = entry.get("seeds")
        if not isinstance(seeds, list):
            raise ValueError(f"corpus manifest split {name} must contain seeds")
        result[name] = seeds
    return result


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m ml.evaluation.cli")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "manifest",
        type=Path,
        nargs="?",
        help="legacy monolithic manifest for non-selection research only",
    )
    parser.add_argument("--public-root", type=Path)
    parser.add_argument("--private-validation", type=Path)
    parser.add_argument("--split", choices=("validation",), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--catalog", action="append", type=Path, default=[])
    parser.add_argument(
        "--validation-partition",
        choices=tuple(partition.value for partition in ValidationPartition),
        help=(
            "evaluate one frozen complete-game validation subpartition; "
            "omitting this produces a labeled non-selection research report"
        ),
    )
    parser.add_argument("--calibration-sidecar-output", type=Path)
    parser.add_argument("--selection-artifact", type=Path)
    parser.add_argument("--selection-sha256")
    parser.add_argument("--training-run", type=Path)
    parser.add_argument("--training-run-sha256")
    parser.add_argument("--calibration-receipt-output", type=Path)
    return parser


def _release_evaluation_inputs(
    arguments: argparse.Namespace,
    release_lease: AuditedPrivateCorpusLease | None = None,
) -> tuple[Any, Mapping[str, Any], SplitManifest, Any]:
    release_values = (arguments.public_root, arguments.private_validation)
    release_mode = all(value is not None for value in release_values)
    if any(value is not None for value in release_values) and not release_mode:
        raise ValueError(
            "release evaluation requires both --public-root and "
            "--private-validation"
        )
    if release_mode and arguments.manifest is not None:
        raise ValueError(
            "release evaluation cannot also consume a legacy monolithic manifest"
        )
    if not release_mode and arguments.manifest is None:
        raise ValueError(
            "provide release manifests or an explicitly legacy research manifest"
        )
    if not release_mode and arguments.validation_partition is not None:
        raise ValueError(
            "selection, calibration-fit, and gate evaluation require "
            "split-private release manifests"
        )
    if release_mode:
        audited = (
            release_lease.audited
            if release_lease is not None
            else audit_private_corpus_split(
                arguments.public_root,
                arguments.private_validation,
                arguments.dataset,
                "validation",
            )
        )
        corpus_value = {"maxPlies": audited.max_plies}
        manifest = SplitManifest(
            train=(),
            validation=audited.seeds,
            test=(),
        )

        def reaudit() -> Any:
            if release_lease is not None:
                release_lease.verify_dataset_unchanged()
                return release_lease.audited
            return audit_private_corpus_split(
                arguments.public_root,
                arguments.private_validation,
                arguments.dataset,
                "validation",
            )

        return audited, corpus_value, manifest, reaudit

    audited = audit_corpus_split(arguments.manifest, arguments.split)
    if arguments.dataset.resolve() != audited.dataset_path:
        raise ValueError(
            "dataset path must be the exact split file bound by the manifest"
        )
    manifest_value = json.loads(arguments.manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest_value, Mapping):
        raise ValueError("legacy corpus manifest must be an object")
    manifest = SplitManifest.from_mapping(_split_manifest_mapping(manifest_value))

    def reaudit() -> Any:
        return audit_corpus_split(arguments.manifest, arguments.split)

    return audited, manifest_value, manifest, reaudit


def build_emit_selection_summary_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ml.evaluation.cli emit-selection-summary"
    )
    parser.add_argument("report", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--training-run-sha256", required=True)
    return parser


def build_select_epoch_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ml.evaluation.cli select-epoch"
    )
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--summary",
        action="append",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--summary-sha256",
        action="append",
        required=True,
    )
    return parser


def build_fit_calibration_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ml.evaluation.cli fit-calibration"
    )
    parser.add_argument("sidecar", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("selection_artifact", type=Path)
    parser.add_argument("training_run", type=Path)
    parser.add_argument("calibration_receipt", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sidecar-sha256", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--selection-sha256", required=True)
    parser.add_argument("--training-run-sha256", required=True)
    parser.add_argument("--calibration-receipt-sha256", required=True)
    return parser


def build_evaluate_ensemble_calibration_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ml.evaluation.cli evaluate-ensemble-calibration"
    )
    parser.add_argument("ensemble_release", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--ensemble-sha256", required=True)
    parser.add_argument("--fusion-selection", type=Path, required=True)
    parser.add_argument("--fusion-selection-sha256", required=True)
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--private-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sidecar-output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--catalog", action="append", type=Path, default=[])
    return parser


def build_select_ensemble_fusion_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ml.evaluation.cli select-ensemble-fusion",
        description=(
            "Select the frozen rank-preserving fusion alpha using only the "
            "authenticated validation selection partition."
        ),
    )
    parser.add_argument("ensemble_release", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--ensemble-sha256", required=True)
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--private-validation", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    return parser


def build_fit_ensemble_calibration_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ml.evaluation.cli fit-ensemble-calibration"
    )
    parser.add_argument("sidecar", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("ensemble_release", type=Path)
    parser.add_argument("receipt_output", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sidecar-sha256", required=True)
    parser.add_argument("--report-sha256", required=True)
    parser.add_argument("--ensemble-sha256", required=True)
    parser.add_argument("--fusion-selection", type=Path, required=True)
    parser.add_argument("--fusion-selection-sha256", required=True)
    return parser


def build_create_ensemble_release_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ml.evaluation.cli create-ensemble-release",
        description=(
            "Verify and publish the fixed three-member ensemble release. "
            "Repeat each member option exactly three times in training-seed "
            "order: 20260811, 20260812, 20260813."
        ),
    )
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--selection",
        action="append",
        type=Path,
        required=True,
        metavar="JSON",
        help=(
            "selected-epoch release JSON; repeat exactly three times in "
            "fixed seed order"
        ),
    )
    parser.add_argument(
        "--selection-sha256",
        action="append",
        required=True,
        metavar="SHA256",
        help="expected SHA-256 for the corresponding --selection",
    )
    parser.add_argument(
        "--training-run",
        action="append",
        type=Path,
        required=True,
        metavar="JSON",
        help=(
            "authenticated training-run claim; repeat exactly three times "
            "in fixed seed order"
        ),
    )
    parser.add_argument(
        "--training-run-sha256",
        action="append",
        required=True,
        metavar="SHA256",
        help="expected SHA-256 for the corresponding --training-run",
    )
    return parser


def build_export_browser_ensemble_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ml.evaluation.cli export-browser-ensemble"
    )
    parser.add_argument("ensemble_release", type=Path)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--ensemble-sha256", required=True)
    parser.add_argument("--calibration-sha256", required=True)
    return parser


def _strict_json_object(payload: bytes, name: str) -> Mapping[str, Any]:
    def reject_constant(token: str) -> None:
        raise ValueError(f"{name} contains non-finite JSON constant {token}")

    try:
        value = json.loads(payload, parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not strict UTF-8 JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} root must be an object")
    _require_finite_json(value, name)
    return value


def _require_finite_json(value: object, name: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} contains a non-finite number")
    if isinstance(value, Mapping):
        for item in value.values():
            _require_finite_json(item, name)
    elif isinstance(value, list):
        for item in value:
            _require_finite_json(item, name)


def _required_object(
    value: Mapping[str, Any],
    key: str,
    context: str,
) -> Mapping[str, Any]:
    item = value.get(key)
    if not isinstance(item, Mapping):
        raise ValueError(f"{context}.{key} must be an object")
    return item


def _selection_nll(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _selection_digest(value: object, name: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _selection_basename(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or Path(value).name != value
    ):
        raise ValueError(f"{name} must be a non-empty basename")
    return value


def _emit_selection_summary(arguments: argparse.Namespace) -> int:
    if (
        isinstance(arguments.training_seed, bool)
        or arguments.training_seed < 0
    ):
        raise ValueError("training seed must be non-negative")
    if isinstance(arguments.epoch, bool) or arguments.epoch <= 0:
        raise ValueError("epoch must be positive")
    try:
        checkpoint_before = arguments.checkpoint.read_bytes()
    except OSError as error:
        raise ValueError(
            f"cannot read checkpoint: {arguments.checkpoint}"
        ) from error
    checkpoint_sha256 = hashlib.sha256(checkpoint_before).hexdigest()
    try:
        report_payload = arguments.report.read_bytes()
    except OSError as error:
        raise ValueError(
            f"cannot read selection report: {arguments.report}"
        ) from error
    report = _strict_json_object(report_payload, "selection report")
    evaluation = _required_object(report, "evaluation", "selection report")
    partition = _required_object(
        evaluation,
        "validationPartition",
        "selection report.evaluation",
    )
    if set(partition) != {"identity", "name", "seedSha256"}:
        raise ValueError("selection report partition fields are not canonical")
    if partition["identity"] != VALIDATION_PARTITION_IDENTITY:
        raise ValueError("selection report uses the wrong partition identity")
    if partition["name"] != ValidationPartition.SELECTION.value:
        raise ValueError("epoch summaries require a selection-only report")
    seed_sha256 = _selection_digest(
        partition["seedSha256"],
        "selection report seedSha256",
    )
    provenance = _required_object(report, "provenance", "selection report")
    release_root_sha256 = _selection_digest(
        provenance.get("release_root_sha256"), "release root sha256"
    )
    corpus_run_id = _selection_digest(
        provenance.get("corpus_run_id"), "corpus run id"
    )
    private_validation_manifest_sha256 = _selection_digest(
        provenance.get("manifest_sha256"),
        "private validation manifest sha256",
    )
    validation_dataset_sha256 = _selection_digest(
        provenance.get("dataset_sha256"), "validation dataset sha256"
    )
    training_run = load_training_run(
        ContentAddressedJson(
            arguments.training_run,
            arguments.training_run_sha256,
        )
    )
    if arguments.training_seed != training_run.seed:
        raise ValueError("training seed disagrees with authenticated training run")
    if arguments.epoch > training_run.epochs:
        raise ValueError("epoch exceeds planned epoch count")
    if provenance.get("checkpoint_seed") != arguments.training_seed:
        raise ValueError("evaluation report checkpoint seed is mislabeled")
    if provenance.get("checkpoint_epoch") != arguments.epoch:
        raise ValueError("evaluation report checkpoint epoch is mislabeled")
    if provenance.get("training_run_id") != training_run.run_id:
        raise ValueError("evaluation report training run is mislabeled")
    declared_checkpoint_file = _selection_basename(
        provenance.get("checkpoint_file"),
        "selection report checkpoint_file",
    )
    if declared_checkpoint_file != arguments.checkpoint.name:
        raise ValueError("selection report names a different checkpoint")
    declared_checkpoint_sha256 = _selection_digest(
        provenance.get("checkpoint_sha256"),
        "selection report checkpoint_sha256",
    )
    if declared_checkpoint_sha256 != checkpoint_sha256:
        raise ValueError(
            "checkpoint bytes do not match the SHA-256 declared by the report"
        )
    report_version = report.get("formatVersion")
    objective_id = training_run.drawback_loss_objective
    if report_version == 1:
        if objective_id != LEGACY_ADDITIVE_DRAWBACK_OBJECTIVE:
            raise ValueError(
                "fusion-grid checkpoint cannot use a legacy additive "
                "selection report"
            )
        legacy_predictor = load_checkpoint_predictor(
            io.BytesIO(checkpoint_before)
        )
        verify_checkpoint_training_identity(
            legacy_predictor,
            training_run,
            expected_seed=arguments.training_seed,
            expected_epoch=arguments.epoch,
            expected_objective=objective_id,
        )
        metrics = _required_object(report, "metrics", "selection report")
        white = _required_object(
            metrics,
            "white_drawback",
            "selection report.metrics",
        )
        black = _required_object(
            metrics,
            "black_drawback",
            "selection report.metrics",
        )
        white_nll = _selection_nll(
            white.get("negative_log_likelihood"),
            "White NLL",
        )
        black_nll = _selection_nll(
            black.get("negative_log_likelihood"),
            "Black NLL",
        )
    elif report_version == 2:
        if objective_id != FUSION_GRID_DRAWBACK_OBJECTIVE:
            raise ValueError(
                "legacy checkpoint cannot use a fusion-grid selection report"
            )
        if (
            provenance.get("drawback_loss_objective")
            != FUSION_GRID_DRAWBACK_OBJECTIVE
        ):
            raise ValueError(
                "selection report drawback objective is invalid"
            )
        predictor = load_checkpoint_predictor(io.BytesIO(checkpoint_before))
        verify_checkpoint_training_identity(
            predictor,
            training_run,
            expected_seed=arguments.training_seed,
            expected_epoch=arguments.epoch,
            expected_objective=objective_id,
        )
        selection = _required_object(
            report,
            "epochSelectionObjective",
            "selection report",
        )
        if set(selection) != {"identity", "metrics"}:
            raise ValueError(
                "selection report epoch objective fields are invalid"
            )
        identity = _required_object(
            selection,
            "identity",
            "selection report.epochSelectionObjective",
        )
        if dict(identity) != fusion_grid_selection_objective_metadata():
            raise ValueError("selection report epoch objective is invalid")
        head_metrics = _required_object(
            selection,
            "metrics",
            "selection report.epochSelectionObjective",
        )
        if set(head_metrics) != {"white", "black"}:
            raise ValueError("selection report objective heads are invalid")
        expected_head_fields = {
            "observation_count",
            "player_game_count",
            "player_game_normalized_nll",
        }
        white = _required_object(
            head_metrics,
            "white",
            "selection report objective metrics",
        )
        black = _required_object(
            head_metrics,
            "black",
            "selection report objective metrics",
        )
        if set(white) != expected_head_fields or set(black) != expected_head_fields:
            raise ValueError(
                "selection report objective metric fields are invalid"
            )
        for color, head in (("White", white), ("Black", black)):
            validate_fusion_grid_head_counts(head, color)
        white_nll = _selection_nll(
            white.get("player_game_normalized_nll"),
            "White player-game-normalized fusion-grid NLL",
        )
        black_nll = _selection_nll(
            black.get("player_game_normalized_nll"),
            "Black player-game-normalized fusion-grid NLL",
        )
    else:
        raise ValueError("selection report format version is unsupported")
    try:
        checkpoint_after = arguments.checkpoint.read_bytes()
    except OSError as error:
        raise ValueError("checkpoint disappeared during summary creation") from error
    if checkpoint_after != checkpoint_before:
        raise ValueError("checkpoint changed during summary creation")
    summary_provenance = {
        "release_root_sha256": release_root_sha256,
        "corpus_run_id": corpus_run_id,
        "private_validation_manifest_sha256": (
            private_validation_manifest_sha256
        ),
        "validation_dataset_sha256": validation_dataset_sha256,
        "model_run_config_sha256": training_run.sha256,
        "planned_epoch_count": training_run.epochs,
        "training_corpus_set_sha256": (
            training_run.training_corpus_set_sha256
        ),
    }
    summary = {
        "format_version": (
            SELECTION_SUMMARY_FORMAT_VERSION
            if objective_id == FUSION_GRID_DRAWBACK_OBJECTIVE
            else 3
        ),
        "training_seed": arguments.training_seed,
        "epoch": arguments.epoch,
        "provenance": summary_provenance,
        "partition": {
            "identity": VALIDATION_PARTITION_IDENTITY,
            "name": ValidationPartition.SELECTION.value,
            "seed_sha256": seed_sha256,
        },
        "checkpoint": {
            "file": declared_checkpoint_file,
            "sha256": checkpoint_sha256,
        },
        "evaluation_report": {
            "file": arguments.report.name,
            "sha256": hashlib.sha256(report_payload).hexdigest(),
        },
        "metrics": {
            "white_nll": white_nll,
            "black_nll": black_nll,
        },
    }
    if objective_id == FUSION_GRID_DRAWBACK_OBJECTIVE:
        summary["objective"] = fusion_grid_selection_objective_metadata()
    rendered = json.dumps(
        summary,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    _write_report_atomic_no_clobber(arguments.output, rendered)
    published = arguments.output.read_bytes()
    published_sha256 = hashlib.sha256(published).hexdigest()
    expected_sha256 = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
    if published_sha256 != expected_sha256:
        raise ValueError("published epoch summary bytes changed unexpectedly")
    print(
        json.dumps(
            {
                "file": arguments.output.name,
                "sha256": published_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


def _select_epoch(arguments: argparse.Namespace) -> int:
    if len(arguments.summary) != len(arguments.summary_sha256):
        raise ValueError(
            "--summary and --summary-sha256 must have equal counts"
        )
    references = tuple(
        ContentAddressedSummary(path, digest)
        for path, digest in zip(
            arguments.summary,
            arguments.summary_sha256,
            strict=True,
        )
    )
    write_selection_artifact(arguments.output, references)
    artifact_sha256 = hashlib.sha256(arguments.output.read_bytes()).hexdigest()
    print(
        json.dumps(
            {
                "file": arguments.output.name,
                "sha256": artifact_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


def _fit_calibration(arguments: argparse.Namespace) -> int:
    output = fit_calibration_release(
        sidecar=ContentAddressedFile(
            arguments.sidecar,
            arguments.sidecar_sha256,
        ),
        checkpoint=ContentAddressedFile(
            arguments.checkpoint,
            arguments.checkpoint_sha256,
        ),
        selection_artifact=ContentAddressedFile(
            arguments.selection_artifact,
            arguments.selection_sha256,
        ),
        training_run=ContentAddressedJson(
            arguments.training_run,
            arguments.training_run_sha256,
        ),
        calibration_receipt=CalibrationReceiptReference(
            arguments.calibration_receipt,
            arguments.calibration_receipt_sha256,
        ),
        output=arguments.output,
    )
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(
        json.dumps(
            {"file": output.name, "sha256": digest},
            sort_keys=True,
        )
    )
    return 0


def _validation_evaluation_context(
    audited: Any,
    manifest: SplitManifest,
    requested: str | None,
) -> ValidationEvaluationContext:
    audited_seeds = tuple(getattr(audited, "seeds", manifest.validation))
    if audited_seeds != manifest.validation:
        raise ValueError("audited validation seeds disagree with the manifest")
    assignments = tuple(audited.game_assignments)
    if len(assignments) != len(audited_seeds):
        raise ValueError(
            "authenticated game assignments do not align with validation seeds"
        )
    if requested is None:
        selected_seeds = audited_seeds
        name = FULL_VALIDATION_PARTITION_NAME
    else:
        partition = ValidationPartition(requested)
        selected_seeds = tuple(
            seed
            for seed in audited_seeds
            if assign_validation_partition(seed) is partition
        )
        name = partition.value
    if not selected_seeds:
        raise ValueError(f"validation partition {name} contains no games")
    observed = frozenset(
        getattr(audited, "observed_seeds", audited_seeds)
    )
    if not observed.issubset(audited_seeds):
        raise ValueError("observed validation seeds are outside the manifest")
    selected_set = frozenset(selected_seeds)
    if not selected_set.intersection(observed):
        raise ValueError(
            f"validation partition {name} contains no move-bearing games"
        )
    game_assignments = {
        game_id: (white, black)
        for seed, (game_id, white, black) in zip(
            audited_seeds,
            assignments,
            strict=True,
        )
        if seed in selected_set
    }
    return ValidationEvaluationContext(
        metadata={
            "identity": VALIDATION_PARTITION_IDENTITY,
            "name": name,
            "seedSha256": validation_seed_sha256(selected_seeds),
        },
        allowed_seeds=selected_set,
        game_assignments=game_assignments,
    )


def _filter_validation_partition_rows(
    rows: Any,
    context: ValidationEvaluationContext,
) -> Any:
    """Filter authenticated rows by complete game seed before inference."""

    game_seeds: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("validation row must be an object")
        seed = row.get("seed")
        game_id = row.get("gameId")
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed < 0
        ):
            raise ValueError("validation row seed must be non-negative")
        if not isinstance(game_id, str) or not game_id:
            raise ValueError("validation row gameId must be non-empty")
        prior = game_seeds.get(game_id)
        if prior is not None and prior != seed:
            raise ValueError(
                f"validation game {game_id} is split across complete-game seeds"
            )
        game_seeds[game_id] = seed
        if seed in context.allowed_seeds:
            if game_id not in context.game_assignments:
                raise ValueError(
                    "selected validation row has no partition assignment"
                )
            yield row


def _ensemble_member_checkpoint_payloads(
    ensemble_reference: ContentAddressedJson,
    loaded_ensemble: Any,
) -> tuple[bytes, bytes, bytes]:
    payloads: list[bytes] = []
    for member in loaded_ensemble.members:
        try:
            resolved = resolve_member_checkpoint(
                ensemble_reference,
                member,
            )
            payload = resolved.read_bytes()
        except (OSError, ValueError) as error:
            raise ValueError(
                "ensemble checkpoint source authentication failed"
            ) from error
        if hashlib.sha256(payload).hexdigest() != member.checkpoint_sha256:
            raise ValueError("ensemble checkpoint SHA-256 does not match")
        payloads.append(payload)
    if len(payloads) != 3:
        raise ValueError("ensemble release does not contain exactly three members")
    return payloads[0], payloads[1], payloads[2]


def _ensemble_report_identity(identity: Any) -> dict[str, object]:
    return {
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
        "calibration_seed_sha256": identity.calibration_seed_sha256,
        "symbolic_schema_sha256": identity.symbolic_schema_sha256,
        "symbolic_feature_version": SYMBOLIC_FEATURE_VERSION,
        "class_count": ENSEMBLE_CLASS_COUNT,
        "fusion": ENSEMBLE_FUSION_METHOD,
        "training_corpus_set": None,
        "partition_identity": VALIDATION_PARTITION_IDENTITY,
        "partition_name": ValidationPartition.CALIBRATION_FIT.value,
    }


def _verify_ensemble_training_corpus_set(
    predictor: Any,
    expected_sha256: str,
) -> None:
    members = getattr(predictor, "members", None)
    if not isinstance(members, tuple) or len(members) != 3:
        raise ValueError("loaded ensemble does not expose exactly three members")
    verified_sets: list[dict[str, object]] = []
    for member in members:
        provenance = getattr(member, "corpus_provenance", None)
        if not isinstance(provenance, Mapping):
            raise ValueError(
                "ensemble member lacks authenticated corpus-set provenance"
            )
        declared = provenance.get("training_corpus_set_sha256")
        corpus_set = provenance.get("training_corpus_set")
        if declared != expected_sha256:
            raise ValueError(
                "ensemble member training corpus-set hash disagrees with release"
            )
        try:
            verified = verify_training_corpus_set(corpus_set)
        except ValueError as error:
            raise ValueError(
                "ensemble member training corpus set is invalid"
            ) from error
        if verified.get("sha256") != expected_sha256:
            raise ValueError(
                "ensemble member full training corpus set disagrees with release"
            )
        verified_sets.append(verified)
    if any(item != verified_sets[0] for item in verified_sets[1:]):
        raise ValueError("ensemble members use different training corpus sets")


def _evaluate_ensemble_calibration(
    arguments: argparse.Namespace,
    release_lease: AuditedPrivateCorpusLease,
) -> int:
    """Evaluate only the frozen validation calibration-fit partition."""

    if arguments.batch_size <= 0:
        raise ValueError("evaluation batch size must be positive")
    evidence_directory = arguments.ensemble_release.parent.resolve()
    for output, name in (
        (arguments.output, "ensemble evaluation report"),
        (arguments.sidecar_output, "ensemble calibration sidecar"),
        (arguments.fusion_selection, "fusion selection artifact"),
    ):
        if output.parent.resolve() != evidence_directory:
            raise ValueError(f"{name} must be beside the ensemble release")
    ensemble_reference = ContentAddressedJson(
        arguments.ensemble_release,
        arguments.ensemble_sha256,
    )
    loaded_ensemble = verify_ensemble_release(ensemble_reference)
    checkpoint_payloads = _ensemble_member_checkpoint_payloads(
        ensemble_reference, loaded_ensemble
    )
    audited, manifest_value, manifest, reaudit = _release_evaluation_inputs(
        argparse.Namespace(
            public_root=arguments.public_root,
            private_validation=arguments.private_validation,
            dataset=arguments.dataset,
            manifest=None,
            split="validation",
            validation_partition=ValidationPartition.CALIBRATION_FIT.value,
        ),
        release_lease,
    )
    expected_provenance = (
        getattr(audited, "release_root_sha256", None),
        getattr(audited, "corpus_run_id", None),
        audited.manifest_sha256,
        audited.dataset_sha256,
    )
    ensemble_provenance = (
        loaded_ensemble.release_root_sha256,
        loaded_ensemble.corpus_run_id,
        loaded_ensemble.private_validation_manifest_sha256,
        loaded_ensemble.validation_dataset_sha256,
    )
    if ensemble_provenance != expected_provenance:
        raise ValueError(
            "ensemble release provenance disagrees with held-out corpus"
        )
    max_plies = manifest_value.get("maxPlies")
    if (
        isinstance(max_plies, bool)
        or not isinstance(max_plies, int)
        or max_plies <= 0
    ):
        raise ValueError("evaluation manifest maxPlies must be positive")
    validation_context = _validation_evaluation_context(
        audited,
        manifest,
        ValidationPartition.CALIBRATION_FIT.value,
    )
    selection_context = _validation_evaluation_context(
        audited,
        manifest,
        ValidationPartition.SELECTION.value,
    )
    if (
        selection_context.metadata["seedSha256"]
        != loaded_ensemble.partition_seed_sha256
    ):
        raise ValueError(
            "ensemble release selection partition disagrees with held-out corpus"
        )
    symbolic_schema_sha256 = canonical_symbolic_schema_sha256(
        SYMBOLIC_FEATURE_VERSION,
        SYMBOLIC_RULE_IDS,
    )
    fusion_selection_reference = ContentAddressedJson(
        arguments.fusion_selection,
        arguments.fusion_selection_sha256,
    )
    loaded_fusion_selection = load_fusion_selection_artifact(
        fusion_selection_reference,
        expected_identity=FusionSelectionIdentity(
            ensemble_release_sha256=ensemble_reference.sha256,
            private_validation_manifest_sha256=(
                loaded_ensemble.private_validation_manifest_sha256
            ),
            validation_dataset_sha256=(
                loaded_ensemble.validation_dataset_sha256
            ),
            validation_seed_sha256=(
                selection_context.metadata["seedSha256"]
            ),
            training_corpus_set_sha256=(
                loaded_ensemble.training_corpus_set_sha256
            ),
            symbolic_schema_sha256=symbolic_schema_sha256,
        ),
    )
    predictor = load_hybrid_ensemble(
        tuple(io.BytesIO(payload) for payload in checkpoint_payloads),
        device=arguments.device,
        fusion_alpha=loaded_fusion_selection.selected_alpha,
        required_corpus_provenance={
            "training_corpus_set_sha256": (
                loaded_ensemble.training_corpus_set_sha256
            )
        },
    )
    _verify_ensemble_training_corpus_set(
        predictor, loaded_ensemble.training_corpus_set_sha256
    )
    if tuple(predictor.drawback_vocabulary) != tuple(SYMBOLIC_RULE_IDS):
        raise ValueError(
            "ensemble vocabulary must equal the canonical ordered catalog"
        )
    identity = identity_from_release(
        ensemble_reference,
        calibration_seed_sha256=validation_context.metadata["seedSha256"],
        symbolic_schema_sha256=symbolic_schema_sha256,
        fusion_selection=fusion_selection_reference,
        training_corpus_set=None,
    )
    rows = _filter_validation_partition_rows(
        read_ndjson_stream(
            release_lease.dataset,
            label=str(arguments.dataset),
        ),
        validation_context,
    )
    if reaudit() != audited:
        raise ValueError("held-out corpus changed while it was being loaded")
    catalogs = arguments.catalog or [
        Path("engine/data/catalog/observed-drawbacks.json")
    ]
    families = load_rule_families(catalogs)
    missing_families = set(SYMBOLIC_RULE_IDS).difference(families)
    if missing_families:
        raise ValueError(
            "rule-family catalog is incomplete: "
            + ", ".join(sorted(missing_families))
        )
    stream = EnsembleCalibrationSidecarStream(
        arguments.sidecar_output, identity
    )

    def capture(observation: Any) -> None:
        stream.add(
            EnsembleCalibrationObservation(
                observation.color, observation.example
            )
        )

    try:
        report = evaluate_held_out(
            rows,
            predictor=predictor,
            split="validation",
            manifest=manifest,
            rule_families=families,
            game_assignments=validation_context.game_assignments,
            max_rows_per_game=max_plies,
            batch_size=arguments.batch_size,
            calibration_sink=capture,
        )
        if reaudit() != audited:
            raise ValueError("held-out corpus changed during ensemble evaluation")
        if verify_ensemble_release(ensemble_reference) != loaded_ensemble:
            raise ValueError("ensemble release changed during evaluation")
        if (
            load_fusion_selection_artifact(
                fusion_selection_reference,
                expected_identity=loaded_fusion_selection.identity,
            )
            != loaded_fusion_selection
        ):
            raise ValueError(
                "fusion selection artifact changed during evaluation"
            )
        sidecar = stream.finalize()
    except BaseException:
        stream.abort()
        raise
    envelope = {
        "format": ENSEMBLE_REPORT_FORMAT,
        "version": ENSEMBLE_CALIBRATION_VERSION,
        "evaluation": {
            "validation_partition": {
                "identity": validation_context.metadata["identity"],
                "name": validation_context.metadata["name"],
                "seed_sha256": validation_context.metadata["seedSha256"],
            },
            "calibration_sidecar": {
                "file": sidecar.path.name,
                "sha256": sidecar.sha256,
            },
        },
        "identity": _ensemble_report_identity(identity),
        "metrics": _json_value(report),
    }
    rendered = (
        json.dumps(envelope, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    _write_report_atomic_no_clobber(arguments.output, rendered)
    print(
        json.dumps(
            {
                "report": {
                    "file": arguments.output.name,
                    "sha256": hashlib.sha256(
                        arguments.output.read_bytes()
                    ).hexdigest(),
                },
                "sidecar": {
                    "file": sidecar.path.name,
                    "sha256": sidecar.sha256,
                },
            },
            sort_keys=True,
        )
    )
    return 0


def _select_ensemble_fusion(
    arguments: argparse.Namespace,
    release_lease: AuditedPrivateCorpusLease,
) -> int:
    """Select fusion strength on the authenticated selection partition only."""

    if arguments.batch_size <= 0:
        raise ValueError("evaluation batch size must be positive")
    evidence_directory = arguments.ensemble_release.parent.resolve()
    if arguments.output.parent.resolve() != evidence_directory:
        raise ValueError(
            "fusion selection output must be beside the ensemble release"
        )
    if arguments.output.exists():
        raise ValueError(
            "refusing to overwrite fusion selection artifact: "
            f"{arguments.output}"
        )
    ensemble_reference = ContentAddressedJson(
        arguments.ensemble_release,
        arguments.ensemble_sha256,
    )
    loaded_ensemble = verify_ensemble_release(ensemble_reference)
    checkpoint_payloads = _ensemble_member_checkpoint_payloads(
        ensemble_reference,
        loaded_ensemble,
    )
    audited, manifest_value, manifest, reaudit = _release_evaluation_inputs(
        argparse.Namespace(
            public_root=arguments.public_root,
            private_validation=arguments.private_validation,
            dataset=arguments.dataset,
            manifest=None,
            split="validation",
            validation_partition=ValidationPartition.SELECTION.value,
        ),
        release_lease,
    )
    expected_provenance = (
        getattr(audited, "release_root_sha256", None),
        getattr(audited, "corpus_run_id", None),
        audited.manifest_sha256,
        audited.dataset_sha256,
    )
    ensemble_provenance = (
        loaded_ensemble.release_root_sha256,
        loaded_ensemble.corpus_run_id,
        loaded_ensemble.private_validation_manifest_sha256,
        loaded_ensemble.validation_dataset_sha256,
    )
    if ensemble_provenance != expected_provenance:
        raise ValueError(
            "ensemble release provenance disagrees with held-out corpus"
        )
    max_plies = manifest_value.get("maxPlies")
    if (
        isinstance(max_plies, bool)
        or not isinstance(max_plies, int)
        or max_plies <= 0
    ):
        raise ValueError("evaluation manifest maxPlies must be positive")
    validation_context = _validation_evaluation_context(
        audited,
        manifest,
        ValidationPartition.SELECTION.value,
    )
    if (
        validation_context.metadata["seedSha256"]
        != loaded_ensemble.partition_seed_sha256
    ):
        raise ValueError(
            "ensemble release selection partition disagrees with held-out corpus"
        )
    identity = FusionSelectionIdentity(
        ensemble_release_sha256=ensemble_reference.sha256,
        private_validation_manifest_sha256=(
            loaded_ensemble.private_validation_manifest_sha256
        ),
        validation_dataset_sha256=loaded_ensemble.validation_dataset_sha256,
        validation_seed_sha256=validation_context.metadata["seedSha256"],
        training_corpus_set_sha256=(
            loaded_ensemble.training_corpus_set_sha256
        ),
        symbolic_schema_sha256=canonical_symbolic_schema_sha256(
            SYMBOLIC_FEATURE_VERSION,
            SYMBOLIC_RULE_IDS,
        ),
    )
    predictor = load_hybrid_ensemble(
        tuple(io.BytesIO(payload) for payload in checkpoint_payloads),
        device=arguments.device,
        fusion_alpha=0.0,
        required_corpus_provenance={
            "training_corpus_set_sha256": (
                loaded_ensemble.training_corpus_set_sha256
            )
        },
    )
    _verify_ensemble_training_corpus_set(
        predictor,
        loaded_ensemble.training_corpus_set_sha256,
    )
    if tuple(predictor.drawback_vocabulary) != tuple(SYMBOLIC_RULE_IDS):
        raise ValueError(
            "ensemble vocabulary must equal the canonical ordered catalog"
        )
    rows = _filter_validation_partition_rows(
        read_ndjson_stream(
            release_lease.dataset,
            label=str(arguments.dataset),
        ),
        validation_context,
    )
    if reaudit() != audited:
        raise ValueError("held-out corpus changed while it was being loaded")
    accumulator = FusionSelectionAccumulator(identity)
    vocabulary_indices = {
        drawback_id: index
        for index, drawback_id in enumerate(
            predictor.drawback_vocabulary
        )
    }

    def capture(
        example: TrainingExample,
        output: InferenceOutput,
    ) -> None:
        color = example.features.player_color
        if color == "white":
            residual = output.white_neural_residual_logits
            prior = example.features.symbolic_white_rule_probabilities
            hard_eliminated = output.white_hard_eliminated
            truth = example.white_drawback
        elif color == "black":
            residual = output.black_neural_residual_logits
            prior = example.features.symbolic_black_rule_probabilities
            hard_eliminated = output.black_hard_eliminated
            truth = example.black_drawback
        else:
            raise ValueError("fusion selection color must be white or black")
        if residual is None or hard_eliminated is None:
            raise ValueError(
                "fusion selection requires genuine residual logits and hard masks"
            )
        try:
            true_index = vocabulary_indices[truth]
        except KeyError as error:
            raise ValueError(
                "fusion selection truth is outside predictor vocabulary"
            ) from error
        accumulator.add(
            FusionSelectionObservation(
                identity=identity,
                partition=ValidationPartition.SELECTION.value,
                game_id=example.game_id,
                color=color,
                observed_ply=example.features.ply,
                true_index=true_index,
                residual_logits=residual,
                symbolic_prior=prior,
                hard_eliminated=hard_eliminated,
            )
        )

    report = evaluate_held_out(
        rows,
        predictor=predictor,
        split="validation",
        manifest=manifest,
        game_assignments=validation_context.game_assignments,
        max_rows_per_game=max_plies,
        batch_size=arguments.batch_size,
        prediction_sink=capture,
    )
    if reaudit() != audited:
        raise ValueError("held-out corpus changed during fusion selection")
    if verify_ensemble_release(ensemble_reference) != loaded_ensemble:
        raise ValueError("ensemble release changed during fusion selection")
    reference = write_fusion_selection_accumulator(
        arguments.output,
        accumulator,
    )
    loaded = load_fusion_selection_artifact(
        reference,
        expected_identity=identity,
    )
    selected = next(
        candidate
        for candidate in loaded.candidates
        if candidate.alpha == loaded.selected_alpha
    )
    expected_counts = (
        report.white_drawback.count,
        report.white_drawback.player_game_count,
        report.black_drawback.count,
        report.black_drawback.player_game_count,
    )
    artifact_counts = (
        selected.white.observation_count,
        selected.white.player_game_count,
        selected.black.observation_count,
        selected.black.player_game_count,
    )
    if artifact_counts != expected_counts:
        raise ValueError(
            "fusion selection evidence counts disagree with held-out report"
        )
    print(
        json.dumps(
            {
                "file": reference.path.name,
                "sha256": reference.sha256,
                "selected_alpha": loaded.selected_alpha,
            },
            sort_keys=True,
        )
    )
    return 0


def _fit_ensemble_calibration(arguments: argparse.Namespace) -> int:
    evidence_directory = arguments.ensemble_release.parent.resolve()
    for path, name in (
        (arguments.sidecar, "sidecar"),
        (arguments.report, "report"),
        (arguments.fusion_selection, "fusion selection"),
        (arguments.receipt_output, "receipt output"),
        (arguments.output, "calibration output"),
    ):
        if path.parent.resolve() != evidence_directory:
            raise ValueError(f"{name} must be beside the ensemble release")
    sidecar = EnsembleContentAddressedFile(
        arguments.sidecar, arguments.sidecar_sha256
    )
    report = EnsembleContentAddressedFile(
        arguments.report, arguments.report_sha256
    )
    ensemble = ContentAddressedJson(
        arguments.ensemble_release, arguments.ensemble_sha256
    )
    fusion_selection = ContentAddressedJson(
        arguments.fusion_selection,
        arguments.fusion_selection_sha256,
    )
    receipt = write_ensemble_calibration_receipt(
        arguments.receipt_output,
        report=report,
        sidecar=sidecar,
        ensemble_release=ensemble,
        fusion_selection=fusion_selection,
    )
    artifact = fit_ensemble_calibration(arguments.output, receipt)
    print(
        json.dumps(
            {
                "receipt": {
                    "file": receipt.path.name,
                    "sha256": receipt.sha256,
                },
                "calibration": {
                    "file": artifact.path.name,
                    "sha256": artifact.sha256,
                },
            },
            sort_keys=True,
        )
    )
    return 0


def _create_ensemble_release(arguments: argparse.Namespace) -> int:
    bindings = (
        ("--selection", arguments.selection),
        ("--selection-sha256", arguments.selection_sha256),
        ("--training-run", arguments.training_run),
        ("--training-run-sha256", arguments.training_run_sha256),
    )
    invalid = tuple(name for name, values in bindings if len(values) != 3)
    if invalid:
        raise ValueError(
            "ensemble release requires exactly three values for "
            + ", ".join(invalid)
        )
    selections = tuple(
        ContentAddressedJson(path, digest)
        for path, digest in zip(
            arguments.selection,
            arguments.selection_sha256,
            strict=True,
        )
    )
    training_runs = tuple(
        ContentAddressedJson(path, digest)
        for path, digest in zip(
            arguments.training_run,
            arguments.training_run_sha256,
            strict=True,
        )
    )
    output = write_ensemble_release(
        arguments.output,
        selections,
        training_runs,
    )
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    verify_ensemble_release(ContentAddressedJson(output, digest))
    print(
        json.dumps(
            {"file": output.name, "sha256": digest},
            sort_keys=True,
        )
    )
    return 0


def _export_browser_ensemble(arguments: argparse.Namespace) -> int:
    output = export_browser_ensemble_artifact(
        ContentAddressedJson(
            arguments.ensemble_release,
            arguments.ensemble_sha256,
        ),
        EnsembleContentAddressedFile(
            arguments.calibration,
            arguments.calibration_sha256,
        ),
        arguments.output,
    )
    print(str(output))
    return 0


def main(argv: list[str] | None = None) -> int:
    raw_arguments = sys.argv[1:] if argv is None else argv
    if raw_arguments and raw_arguments[0] == "emit-selection-summary":
        arguments = build_emit_selection_summary_parser().parse_args(
            raw_arguments[1:]
        )
        return _emit_selection_summary(arguments)
    if raw_arguments and raw_arguments[0] == "select-epoch":
        arguments = build_select_epoch_parser().parse_args(raw_arguments[1:])
        return _select_epoch(arguments)
    if raw_arguments and raw_arguments[0] == "fit-calibration":
        arguments = build_fit_calibration_parser().parse_args(
            raw_arguments[1:]
        )
        return _fit_calibration(arguments)
    if (
        raw_arguments
        and raw_arguments[0] == "evaluate-ensemble-calibration"
    ):
        arguments = build_evaluate_ensemble_calibration_parser().parse_args(
            raw_arguments[1:]
        )
        with open_audited_private_corpus_split(
            arguments.public_root,
            arguments.private_validation,
            arguments.dataset,
            "validation",
        ) as release_lease:
            return _evaluate_ensemble_calibration(arguments, release_lease)
    if raw_arguments and raw_arguments[0] == "select-ensemble-fusion":
        arguments = build_select_ensemble_fusion_parser().parse_args(
            raw_arguments[1:]
        )
        with open_audited_private_corpus_split(
            arguments.public_root,
            arguments.private_validation,
            arguments.dataset,
            "validation",
        ) as release_lease:
            return _select_ensemble_fusion(arguments, release_lease)
    if raw_arguments and raw_arguments[0] == "fit-ensemble-calibration":
        arguments = build_fit_ensemble_calibration_parser().parse_args(
            raw_arguments[1:]
        )
        return _fit_ensemble_calibration(arguments)
    if raw_arguments and raw_arguments[0] == "create-ensemble-release":
        arguments = build_create_ensemble_release_parser().parse_args(
            raw_arguments[1:]
        )
        return _create_ensemble_release(arguments)
    if raw_arguments and raw_arguments[0] == "export-browser-ensemble":
        arguments = build_export_browser_ensemble_parser().parse_args(
            raw_arguments[1:]
        )
        return _export_browser_ensemble(arguments)
    arguments = build_parser().parse_args(raw_arguments)
    if (
        arguments.public_root is not None
        and arguments.private_validation is not None
    ):
        with open_audited_private_corpus_split(
            arguments.public_root,
            arguments.private_validation,
            arguments.dataset,
            "validation",
        ) as release_lease:
            return _evaluate(arguments, release_lease)
    return _evaluate(arguments, None)


def _evaluate(
    arguments: argparse.Namespace,
    release_lease: AuditedPrivateCorpusLease | None,
) -> int:
    if arguments.batch_size <= 0:
        raise ValueError("evaluation batch size must be positive")
    audited, manifest_value, manifest, reaudit = _release_evaluation_inputs(
        arguments,
        release_lease,
    )
    max_plies = manifest_value.get("maxPlies")
    if (
        isinstance(max_plies, bool)
        or not isinstance(max_plies, int)
        or max_plies <= 0
    ):
        raise ValueError("evaluation manifest maxPlies must be positive")
    validation_context = _validation_evaluation_context(
        audited,
        manifest,
        arguments.validation_partition,
    )
    calibration_requested = arguments.validation_partition == "calibration-fit"
    calibration_arguments = (
        arguments.calibration_sidecar_output,
        arguments.selection_artifact,
        arguments.selection_sha256,
        arguments.training_run,
        arguments.training_run_sha256,
        arguments.calibration_receipt_output,
    )
    if calibration_requested and any(item is None for item in calibration_arguments):
        raise ValueError(
            "calibration-fit evaluation requires sidecar output, selection artifact, "
            "selection SHA-256, training run, training-run SHA-256, and "
            "a receipt output"
        )
    if not calibration_requested and any(
        item is not None for item in calibration_arguments
    ):
        raise ValueError(
            "calibration sidecars may be emitted for calibration-fit only"
        )
    if calibration_requested and arguments.output is None:
        raise ValueError(
            "calibration-fit evaluation requires an evaluation report output"
        )
    catalogs = arguments.catalog or [
        Path("engine/data/catalog/observed-drawbacks.json")
    ]
    checkpoint_payload = arguments.checkpoint.read_bytes()
    checkpoint_before = hashlib.sha256(checkpoint_payload).hexdigest()
    predictor = load_checkpoint_predictor(
        io.BytesIO(checkpoint_payload),
        device=arguments.device,
        required_corpus_provenance={},
    )
    _require_checkpoint_corpus_provenance(predictor, audited)
    if tuple(predictor.drawback_vocabulary) != tuple(SYMBOLIC_RULE_IDS):
        raise ValueError(
            "checkpoint drawback vocabulary must equal the canonical ordered catalog"
        )
    calibration_stream = None
    calibration_stream_factory = None
    selection_json_reference = None
    training_run_reference = None
    if calibration_requested:
        selection_reference = ContentAddressedFile(
            arguments.selection_artifact,
            arguments.selection_sha256,
        )
        selection_payload = selection_reference.path.read_bytes()
        if hashlib.sha256(
            selection_payload
        ).hexdigest() != selection_reference.sha256:
            raise ValueError("selection artifact SHA-256 does not match")
        verified_selection = verify_release_selection_bundle(
            ContentAddressedJson(
                selection_reference.path, selection_reference.sha256
            ),
            ContentAddressedJson(
                arguments.training_run, arguments.training_run_sha256
            ),
        )
        selection_json_reference = ContentAddressedJson(
            selection_reference.path, selection_reference.sha256
        )
        training_run_reference = ContentAddressedJson(
            arguments.training_run, arguments.training_run_sha256
        )
        selected_artifact = verified_selection.artifact
        if selected_artifact.selected_checkpoint_sha256 != checkpoint_before:
            raise ValueError("selection artifact names a different checkpoint")
        if (
            selected_artifact.provenance.release_root_sha256
            != getattr(audited, "release_root_sha256", None)
            or selected_artifact.provenance.corpus_run_id
            != getattr(audited, "corpus_run_id", None)
            or selected_artifact.provenance.private_validation_manifest_sha256
            != audited.manifest_sha256
            or selected_artifact.provenance.validation_dataset_sha256
            != audited.dataset_sha256
        ):
            raise ValueError(
                "selection artifact provenance disagrees with release-safe audit"
            )
        calibration_stream_factory = lambda: CalibrationSidecarStream(
            arguments.calibration_sidecar_output,
            CalibrationSidecarHeader(
                calibration_seed_sha256=validation_context.metadata[
                    "seedSha256"
                ],
                checkpoint_sha256=checkpoint_before,
                selection_artifact_sha256=selection_reference.sha256,
                symbolic_schema_sha256=canonical_symbolic_schema_sha256(
                    SYMBOLIC_FEATURE_VERSION,
                    SYMBOLIC_RULE_IDS,
                ),
                symbolic_feature_version=SYMBOLIC_FEATURE_VERSION,
                class_count=len(SYMBOLIC_RULE_IDS),
            ),
        )
    rows = _filter_validation_partition_rows(
        (
            read_ndjson_stream(
                release_lease.dataset,
                label=str(arguments.dataset),
            )
            if release_lease is not None
            else read_ndjson(arguments.dataset)
        ),
        validation_context,
    )
    if reaudit() != audited:
        raise ValueError("held-out corpus changed while it was being loaded")
    families = load_rule_families(catalogs)
    missing_families = set(SYMBOLIC_RULE_IDS).difference(families)
    if missing_families:
        raise ValueError(
            "rule-family catalog is incomplete: "
            + ", ".join(sorted(missing_families))
        )
    if calibration_stream_factory is not None:
        calibration_stream = calibration_stream_factory()
    epoch_scorer = (
        FusionGridEpochScorer()
        if (
            arguments.validation_partition
            == ValidationPartition.SELECTION.value
            and predictor.drawback_loss_objective
            == FUSION_GRID_DRAWBACK_OBJECTIVE
        )
        else None
    )

    def score_epoch_prediction(
        example: TrainingExample,
        output: InferenceOutput,
    ) -> None:
        if epoch_scorer is None:
            return
        color = example.features.player_color
        if color == "white":
            truth = example.white_drawback
            residuals = output.white_neural_residual_logits
            mask = output.white_hard_eliminated
            prior = example.features.symbolic_white_rule_probabilities
        elif color == "black":
            truth = example.black_drawback
            residuals = output.black_neural_residual_logits
            mask = output.black_hard_eliminated
            prior = example.features.symbolic_black_rule_probabilities
        else:
            raise ValueError("selection example player color is invalid")
        if residuals is None or mask is None:
            raise ValueError(
                "fusion-grid selection requires raw residuals and hard masks"
            )
        try:
            true_index = tuple(predictor.drawback_vocabulary).index(truth)
        except ValueError as error:
            raise ValueError(
                "fusion-grid truth is outside checkpoint vocabulary"
            ) from error
        epoch_scorer.add(
            game_id=example.game_id,
            color=color,
            observed_ply=example.features.ply,
            true_index=true_index,
            residual_logits=residuals,
            symbolic_prior=prior,
            hard_eliminated=mask,
        )
    try:
        report = evaluate_held_out(
            rows,
            predictor=predictor,
            split=arguments.split,
            manifest=manifest,
            rule_families=families,
            game_assignments=validation_context.game_assignments,
            max_rows_per_game=max_plies,
            batch_size=arguments.batch_size,
            calibration_sink=(
                None if calibration_stream is None else calibration_stream.add
            ),
            prediction_sink=(
                None if epoch_scorer is None else score_epoch_prediction
            ),
        )
    except BaseException:
        if calibration_stream is not None:
            calibration_stream.abort()
        raise
    if reaudit() != audited:
        if calibration_stream is not None:
            calibration_stream.abort()
        raise ValueError("held-out corpus changed during evaluation")
    calibration_sidecar = (
        None
        if calibration_stream is None
        else calibration_stream.finalize()
    )
    epoch_metrics = None if epoch_scorer is None else epoch_scorer.report()
    envelope = {
        "formatVersion": 2 if epoch_metrics is not None else 1,
        "serialization": {"nonFiniteMetricPolicy": "null"},
        "evaluation": {
            "batchSize": arguments.batch_size,
            "validationPartition": dict(validation_context.metadata),
            "calibrationObservationSidecar": (
                None
                if calibration_sidecar is None
                else {
                    "file": calibration_sidecar.path.name,
                    "sha256": calibration_sidecar.sha256,
                }
            ),
        },
        "provenance": {
            **audited.provenance(),
            "checkpoint_file": arguments.checkpoint.name,
            "checkpoint_sha256": checkpoint_before,
            "checkpoint_seed": predictor.checkpoint_seed,
            "checkpoint_epoch": predictor.checkpoint_epoch,
            "training_run_id": predictor.training_run_id,
        },
        "metrics": _json_value(report),
    }
    if epoch_metrics is not None:
        envelope["provenance"]["drawback_loss_objective"] = (
            predictor.drawback_loss_objective
        )
        envelope["epochSelectionObjective"] = {
            "identity": fusion_grid_selection_objective_metadata(),
            "metrics": {
                color: {
                    "observation_count": metric.observation_count,
                    "player_game_count": metric.player_game_count,
                    "player_game_normalized_nll": (
                        metric.player_game_normalized_nll
                    ),
                }
                for color, metric in epoch_metrics.items()
            },
        }
    rendered = json.dumps(
        envelope,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    if arguments.output is None:
        print(rendered, end="")
    else:
        _write_report_atomic_no_clobber(arguments.output, rendered)
        if calibration_requested:
            if (
                calibration_sidecar is None
                or selection_json_reference is None
                or training_run_reference is None
            ):
                raise RuntimeError("calibration receipt inputs were not finalized")
            report_payload = arguments.output.read_bytes()
            report_reference = ContentAddressedJson(
                arguments.output,
                hashlib.sha256(report_payload).hexdigest(),
            )
            write_calibration_receipt(
                arguments.calibration_receipt_output,
                CalibrationReceiptInputs(
                    evaluation_report=report_reference,
                    sidecar=ContentAddressedJson(
                        calibration_sidecar.path,
                        calibration_sidecar.sha256,
                    ),
                    checkpoint=ContentAddressedJson(
                        arguments.checkpoint,
                        checkpoint_before,
                    ),
                    selection_artifact=selection_json_reference,
                    training_run=training_run_reference,
                ),
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
