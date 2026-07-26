"""Independent clean-process reproduction of frozen validation-gate evidence."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import tempfile
from typing import Callable, Mapping, Sequence

from ml.training.drawback_ml.corpus_contract import (
    open_audited_private_corpus_split,
)

from .ensemble_calibration import ContentAddressedFile
from .training_frequency import (
    ContentAddressedFile as TrainingFrequencyReference,
)
from .validation_gate import (
    DECISION_FORMAT,
    PROTOCOL_ID,
    REPORT_FORMAT,
    VERSION as VALIDATION_GATE_VERSION,
    _canonical_pretty,
    load_validation_gate_document,
)


FORMAT = "drawbacktrainer-validation-reproduction-receipt"
VERSION = 1
FLOAT_TOLERANCE = 1e-6
SOURCE_PATHS = (
    "ml",
    "packages",
    "apps",
    "scripts",
    "engine",
    "package.json",
    "pnpm-lock.yaml",
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class EnvironmentAttestation:
    source_revision: str
    pnpm_lock_sha256: str
    python_requirements_sha256: str
    python_project_sha256: str
    python_executable_sha256: str
    python_version: str


@dataclass(frozen=True)
class ComparisonResult:
    float_count: int
    maximum_absolute_float_difference: float


def _digest_file(path: Path, label: str) -> str:
    try:
        source = path.open("rb")
    except OSError as error:
        raise ValueError(f"cannot read {label}: {path}") from error
    digest = hashlib.sha256()
    with source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_digest(value: str, label: str) -> str:
    if SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _strict_json(payload: bytes, label: str) -> Mapping[str, object]:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(token: str) -> None:
        raise ValueError(f"{label} contains non-finite constant {token}")

    try:
        value = json.loads(
            payload,
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} root must be an object")
    return value


def load_validation_reproduction_receipt(
    reference: ContentAddressedFile,
) -> Mapping[str, object]:
    """Strictly reload a receipt and recursively verify original evidence."""

    try:
        payload = reference.path.read_bytes()
    except OSError as error:
        raise ValueError("cannot read validation reproduction receipt") from error
    if hashlib.sha256(payload).hexdigest() != reference.sha256:
        raise ValueError("validation reproduction receipt SHA-256 does not match")
    value = _strict_json(payload, "validation reproduction receipt")
    if payload != _canonical_pretty(value):
        raise ValueError("validation reproduction receipt is not canonical")
    expected_keys = {
        "format",
        "version",
        "protocol_id",
        "original",
        "candidate_inputs",
        "validation_corpus",
        "environment",
        "evaluator",
        "fresh_process",
        "comparison",
    }
    if set(value) != expected_keys:
        raise ValueError("validation reproduction receipt fields are invalid")
    if (
        value.get("format") != FORMAT
        or value.get("version") != VERSION
        or value.get("protocol_id") != PROTOCOL_ID
    ):
        raise ValueError("unsupported validation reproduction receipt")
    original = value.get("original")
    candidate_inputs = value.get("candidate_inputs")
    comparison = value.get("comparison")
    if (
        not isinstance(original, Mapping)
        or not isinstance(candidate_inputs, Mapping)
        or not isinstance(comparison, Mapping)
    ):
        raise ValueError("validation reproduction receipt sections are invalid")
    if set(candidate_inputs) != {
        "ensemble_release_sha256",
        "calibration_sha256",
        "training_frequency_sha256",
        "catalogs",
    }:
        raise ValueError("validation reproduction candidate inputs are invalid")
    catalogs = candidate_inputs.get("catalogs")
    if not isinstance(catalogs, list) or not catalogs:
        raise ValueError("validation reproduction catalogs are absent")
    for binding in catalogs:
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"file", "sha256"}
            or not isinstance(binding.get("file"), str)
            or not binding.get("file")
            or Path(str(binding.get("file"))).name != binding.get("file")
        ):
            raise ValueError("validation reproduction catalog binding is invalid")
        _expected_digest(
            str(binding.get("sha256")), "receipt catalog sha256"
        )
    if (
        comparison.get("float_tolerance") != FLOAT_TOLERANCE
        or comparison.get("exact_candidate_and_input_hashes") is not True
        or comparison.get("exact_transcript_sha256") is not True
    ):
        raise ValueError("validation reproduction comparison is not exact")
    maximum = comparison.get("maximum_absolute_float_difference")
    if (
        isinstance(maximum, bool)
        or not isinstance(maximum, (int, float))
        or not math.isfinite(float(maximum))
        or float(maximum) > FLOAT_TOLERANCE
    ):
        raise ValueError("validation reproduction metric difference is invalid")
    loaded_original: dict[str, Mapping[str, object]] = {}
    references: dict[str, ContentAddressedFile] = {}
    for key, expected_format in (
        ("report", REPORT_FORMAT),
        ("decision", DECISION_FORMAT),
    ):
        binding = original.get(key)
        if not isinstance(binding, Mapping) or set(binding) != {"file", "sha256"}:
            raise ValueError(f"original {key} binding is invalid")
        filename = binding.get("file")
        digest = binding.get("sha256")
        if (
            not isinstance(filename, str)
            or not filename
            or Path(filename).name != filename
        ):
            raise ValueError(f"original {key} file must be a basename")
        bound = ContentAddressedFile(
            reference.path.parent / filename,
            _expected_digest(str(digest), f"original {key} sha256"),
        )
        references[key] = bound
        loaded_original[key] = load_validation_gate_document(
            bound, expected_format
        )
    _exact_original_evidence(
        loaded_original["report"],
        loaded_original["decision"],
        report_reference=references["report"],
        ensemble_sha256=_expected_digest(
            str(candidate_inputs.get("ensemble_release_sha256")),
            "receipt ensemble sha256",
        ),
        calibration_sha256=_expected_digest(
            str(candidate_inputs.get("calibration_sha256")),
            "receipt calibration sha256",
        ),
        training_frequency_sha256=_expected_digest(
            str(candidate_inputs.get("training_frequency_sha256")),
            "receipt training-frequency sha256",
        ),
    )
    return value


def _run_git(
    repository: Path, arguments: Sequence[str]
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except OSError as error:
        raise ValueError("cannot execute Git for source attestation") from error


def attest_clean_environment(
    repository: Path,
    *,
    expected_source_revision: str,
    expected_pnpm_lock_sha256: str,
    expected_python_requirements_sha256: str,
    expected_python_project_sha256: str,
) -> EnvironmentAttestation:
    """Require the exact committed source and dependency lock identities."""

    root = repository.resolve()
    expected_revision = expected_source_revision.lower()
    if REVISION_PATTERN.fullmatch(expected_revision) is None:
        raise ValueError("expected source revision must be a full Git SHA")
    head = _run_git(root, ["rev-parse", "HEAD"])
    if head.returncode != 0:
        raise ValueError("repository root is not a readable Git checkout")
    revision = head.stdout.strip().lower()
    if revision != expected_revision:
        raise ValueError("current source revision differs from the approved revision")
    tracked_status = _run_git(
        root,
        ["status", "--porcelain=v1", "--untracked-files=no"],
    )
    if tracked_status.returncode != 0:
        raise ValueError("cannot inspect tracked worktree status")
    if tracked_status.stdout.strip():
        raise ValueError("tracked worktree is not clean")
    status = _run_git(
        root,
        [
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            *SOURCE_PATHS,
        ],
    )
    if status.returncode != 0:
        raise ValueError("cannot inspect source worktree status")
    if status.stdout.strip():
        raise ValueError("source worktree is not clean")
    locks = (
        (
            root / "pnpm-lock.yaml",
            expected_pnpm_lock_sha256,
            "pnpm lock",
        ),
        (
            root / "ml" / "requirements.txt",
            expected_python_requirements_sha256,
            "Python requirements",
        ),
        (
            root / "ml" / "pyproject.toml",
            expected_python_project_sha256,
            "Python project",
        ),
    )
    observed: list[str] = []
    for path, expected, label in locks:
        expected_hash = _expected_digest(expected, f"expected {label} sha256")
        actual = _digest_file(path, label)
        if actual != expected_hash:
            raise ValueError(f"{label} differs from its approved SHA-256")
        observed.append(actual)
    executable = Path(sys.executable).resolve()
    return EnvironmentAttestation(
        source_revision=revision,
        pnpm_lock_sha256=observed[0],
        python_requirements_sha256=observed[1],
        python_project_sha256=observed[2],
        python_executable_sha256=_digest_file(
            executable, "Python executable"
        ),
        python_version=sys.version,
    )


def _exact_original_evidence(
    report: Mapping[str, object],
    decision: Mapping[str, object],
    *,
    report_reference: ContentAddressedFile,
    ensemble_sha256: str,
    calibration_sha256: str,
    training_frequency_sha256: str,
) -> None:
    protocol = report.get("protocol")
    bindings = report.get("bindings")
    promotion = report.get("promotion")
    report_binding = decision.get("validation_report")
    results = decision.get("results")
    if (
        not isinstance(protocol, Mapping)
        or protocol.get("id") != PROTOCOL_ID
        or not isinstance(bindings, Mapping)
        or not isinstance(promotion, Mapping)
        or promotion.get("partition") != "validation-gate"
        or not isinstance(report_binding, Mapping)
        or report_binding
        != {
            "file": report_reference.path.name,
            "sha256": report_reference.sha256,
        }
    ):
        raise ValueError("original validation evidence bindings are invalid")
    expected_bindings = {
        "ensemble_release_sha256": ensemble_sha256,
        "calibration_sha256": calibration_sha256,
        "training_frequency_sha256": training_frequency_sha256,
    }
    if dict(bindings) != expected_bindings:
        raise ValueError("original report binds different candidate inputs")
    if (
        decision.get("protocol_id") != PROTOCOL_ID
        or decision.get("passed") is not True
        or decision.get("missing_count") != 0
        or decision.get("failed_count") != 0
        or not isinstance(results, list)
        or not results
    ):
        raise ValueError("original validation decision is not a passing decision")
    gate_ids: set[str] = set()
    for item in results:
        if not isinstance(item, Mapping):
            raise ValueError("original decision contains a malformed gate result")
        gate_id = item.get("gate_id")
        if (
            not isinstance(gate_id, str)
            or not gate_id
            or gate_id in gate_ids
            or item.get("status") != "passed"
        ):
            raise ValueError("original decision gate results are not all passing")
        gate_ids.add(gate_id)


def _compare_values(
    expected: object,
    actual: object,
    *,
    path: str,
    state: list[float],
) -> None:
    if (
        isinstance(expected, (int, float))
        and not isinstance(expected, bool)
        and isinstance(actual, (int, float))
        and not isinstance(actual, bool)
        and (isinstance(expected, float) or isinstance(actual, float))
    ):
        left = float(expected)
        right = float(actual)
        if not math.isfinite(left) or not math.isfinite(right):
            raise ValueError(f"{path} contains a non-finite metric")
        difference = abs(left - right)
        state[0] += 1
        state[1] = max(state[1], difference)
        if difference > FLOAT_TOLERANCE:
            raise ValueError(
                f"{path} differs by more than {FLOAT_TOLERANCE}"
            )
        return
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        if set(expected) != set(actual):
            raise ValueError(f"{path} object fields differ")
        for key in sorted(expected):
            _compare_values(
                expected[key],
                actual[key],
                path=f"{path}.{key}",
                state=state,
            )
        return
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            raise ValueError(f"{path} array lengths differ")
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            _compare_values(
                left,
                right,
                path=f"{path}[{index}]",
                state=state,
            )
        return
    if type(expected) is not type(actual) or expected != actual:
        raise ValueError(f"{path} differs")


def compare_validation_evidence(
    original_report: Mapping[str, object],
    original_decision: Mapping[str, object],
    reproduced_report: Mapping[str, object],
    reproduced_decision: Mapping[str, object],
) -> ComparisonResult:
    """Require exact identities/transcript and metrics within one micro-unit."""

    original_promotion = original_report.get("promotion")
    reproduced_promotion = reproduced_report.get("promotion")
    if not isinstance(original_promotion, Mapping) or not isinstance(
        reproduced_promotion, Mapping
    ):
        raise ValueError("validation report lacks promotion evidence")
    original_transcript = original_promotion.get("transcript")
    reproduced_transcript = reproduced_promotion.get("transcript")
    if (
        not isinstance(original_transcript, Mapping)
        or not isinstance(reproduced_transcript, Mapping)
        or original_transcript.get("sha256")
        != reproduced_transcript.get("sha256")
    ):
        raise ValueError("reproduced inference transcript SHA-256 differs")
    state = [0.0, 0.0]
    _compare_values(
        original_promotion,
        reproduced_promotion,
        path="promotion",
        state=state,
    )
    original_normalized = dict(original_decision)
    reproduced_normalized = dict(reproduced_decision)
    original_normalized.pop("validation_report", None)
    reproduced_normalized.pop("validation_report", None)
    _compare_values(
        original_normalized,
        reproduced_normalized,
        path="decision",
        state=state,
    )
    return ComparisonResult(
        float_count=int(state[0]),
        maximum_absolute_float_difference=state[1],
    )


def _write_atomic_no_clobber(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ValueError(
                f"refusing to overwrite validation reproduction receipt: {path}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)


def _temporary_output_paths(directory: Path) -> tuple[Path, Path]:
    for _ in range(10):
        nonce = secrets.token_hex(16)
        report = directory / f".reproduction-{nonce}.report.json"
        decision = directory / f".reproduction-{nonce}.decision.json"
        if not report.exists() and not decision.exists():
            return report, decision
    raise ValueError("cannot allocate fresh validation reproduction outputs")


def _command(
    *,
    repository: Path,
    ensemble: ContentAddressedFile,
    calibration: ContentAddressedFile,
    frequency: TrainingFrequencyReference,
    dataset: Path,
    public_root: Path,
    private_validation: Path,
    report_output: Path,
    decision_output: Path,
    catalogs: Sequence[Path],
    batch_size: int,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "ml.evaluation.validation_gate",
        str(ensemble.path.resolve()),
        str(calibration.path.resolve()),
        str(frequency.path.resolve()),
        str(dataset.resolve()),
        "--ensemble-sha256",
        ensemble.sha256,
        "--calibration-sha256",
        calibration.sha256,
        "--training-frequency-sha256",
        frequency.sha256,
        "--public-root",
        str(public_root.resolve()),
        "--private-validation",
        str(private_validation.resolve()),
        "--report-output",
        str(report_output.resolve()),
        "--decision-output",
        str(decision_output.resolve()),
        "--batch-size",
        str(batch_size),
    ]
    for catalog in catalogs:
        command.extend(("--catalog", str(catalog.resolve())))
    return command


def reproduce_validation(
    *,
    repository: Path,
    original_report: ContentAddressedFile,
    original_decision: ContentAddressedFile,
    ensemble: ContentAddressedFile,
    calibration: ContentAddressedFile,
    training_frequency: TrainingFrequencyReference,
    public_root: Path,
    private_validation: Path,
    validation_dataset: Path,
    catalogs: Sequence[Path],
    receipt_output: Path,
    environment: EnvironmentAttestation,
    expected_engine_binary_sha256: str,
    expected_engine_fingerprint: str,
    batch_size: int,
    process_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> ContentAddressedFile:
    """Rerun the gate in a fresh process and publish an immutable receipt."""

    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    if receipt_output.exists():
        raise ValueError("validation reproduction receipt already exists")
    if (
        attest_clean_environment(
            repository,
            expected_source_revision=environment.source_revision,
            expected_pnpm_lock_sha256=environment.pnpm_lock_sha256,
            expected_python_requirements_sha256=(
                environment.python_requirements_sha256
            ),
            expected_python_project_sha256=environment.python_project_sha256,
        )
        != environment
    ):
        raise ValueError("clean environment identity is not reproducible")
    evidence_directory = ensemble.path.parent.resolve()
    for path, label in (
        (original_report.path, "original report"),
        (original_decision.path, "original decision"),
        (calibration.path, "calibration"),
        (training_frequency.path, "training frequency"),
        (receipt_output, "receipt output"),
    ):
        if path.parent.resolve() != evidence_directory:
            raise ValueError(f"{label} must be beside the ensemble release")
    expected_binary = _expected_digest(
        expected_engine_binary_sha256, "expected engine binary sha256"
    )
    report = load_validation_gate_document(original_report, REPORT_FORMAT)
    decision = load_validation_gate_document(original_decision, DECISION_FORMAT)
    _exact_original_evidence(
        report,
        decision,
        report_reference=original_report,
        ensemble_sha256=ensemble.sha256,
        calibration_sha256=calibration.sha256,
        training_frequency_sha256=training_frequency.sha256,
    )
    with open_audited_private_corpus_split(
        public_root,
        private_validation,
        validation_dataset,
        "validation",
    ) as lease:
        audited = lease.audited
        if (
            audited.engine_binary_sha256 != expected_binary
            or audited.engine_fingerprint != expected_engine_fingerprint
            or audited.evaluator_policy_id != "stockfish-bestmove-v1"
            or audited.evaluator_policy_version != 1
            or audited.evaluator_nodes != 10_000
        ):
            raise ValueError("authenticated validation evaluator identity differs")
        report_output, decision_output = _temporary_output_paths(
            ensemble.path.parent.resolve()
        )
        command = _command(
            repository=repository,
            ensemble=ensemble,
            calibration=calibration,
            frequency=training_frequency,
            dataset=validation_dataset,
            public_root=public_root,
            private_validation=private_validation,
            report_output=report_output,
            decision_output=decision_output,
            catalogs=catalogs,
            batch_size=batch_size,
        )
        child_environment = dict(os.environ)
        child_environment.update(
            {"PYTHONHASHSEED": "0", "PYTHONNOUSERSITE": "1"}
        )
        try:
            completed = process_runner(
                command,
                cwd=repository.resolve(),
                env=child_environment,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            if completed.returncode != 0:
                raise ValueError(
                    "fresh validation process failed: "
                    + completed.stderr[-2_000:]
                )
            if not report_output.is_file() or not decision_output.is_file():
                raise ValueError(
                    "fresh validation process did not publish both outputs"
                )
            reproduced_report_ref = ContentAddressedFile(
                report_output, _digest_file(report_output, "reproduced report")
            )
            reproduced_decision_ref = ContentAddressedFile(
                decision_output,
                _digest_file(decision_output, "reproduced decision"),
            )
            reproduced_report = load_validation_gate_document(
                reproduced_report_ref, REPORT_FORMAT
            )
            reproduced_decision = load_validation_gate_document(
                reproduced_decision_ref, DECISION_FORMAT
            )
            if (
                reproduced_decision.get("passed") is not True
                or reproduced_decision.get("missing_count") != 0
                or reproduced_decision.get("failed_count") != 0
            ):
                raise ValueError("fresh validation decision is not passing")
            comparison = compare_validation_evidence(
                report,
                decision,
                reproduced_report,
                reproduced_decision,
            )
            lease.verify_dataset_unchanged()
            if (
                attest_clean_environment(
                    repository,
                    expected_source_revision=environment.source_revision,
                    expected_pnpm_lock_sha256=environment.pnpm_lock_sha256,
                    expected_python_requirements_sha256=(
                        environment.python_requirements_sha256
                    ),
                    expected_python_project_sha256=(
                        environment.python_project_sha256
                    ),
                )
                != environment
            ):
                raise ValueError(
                    "clean environment identity changed during reproduction"
                )
            catalog_bindings = [
                {
                    "file": catalog.name,
                    "sha256": _digest_file(catalog, "rule catalog"),
                }
                for catalog in catalogs
            ]
            receipt_value = {
                "format": FORMAT,
                "version": VERSION,
                "protocol_id": PROTOCOL_ID,
                "original": {
                    "report": {
                        "file": original_report.path.name,
                        "sha256": original_report.sha256,
                    },
                    "decision": {
                        "file": original_decision.path.name,
                        "sha256": original_decision.sha256,
                    },
                },
                "candidate_inputs": {
                    "ensemble_release_sha256": ensemble.sha256,
                    "calibration_sha256": calibration.sha256,
                    "training_frequency_sha256": training_frequency.sha256,
                    "catalogs": catalog_bindings,
                },
                "validation_corpus": {
                    "release_root_sha256": audited.release_root_sha256,
                    "corpus_run_id": audited.corpus_run_id,
                    "private_validation_manifest_sha256": audited.manifest_sha256,
                    "validation_dataset_sha256": audited.dataset_sha256,
                    "partition_seed_sha256": report["promotion"][
                        "partition_seed_sha256"
                    ],
                },
                "environment": {
                    "source_revision": environment.source_revision,
                    "pnpm_lock_sha256": environment.pnpm_lock_sha256,
                    "python_requirements_sha256": (
                        environment.python_requirements_sha256
                    ),
                    "python_project_sha256": environment.python_project_sha256,
                    "python_executable_sha256": (
                        environment.python_executable_sha256
                    ),
                    "python_version": environment.python_version,
                },
                "evaluator": {
                    "engine_binary_sha256": audited.engine_binary_sha256,
                    "engine_fingerprint": audited.engine_fingerprint,
                    "policy_id": audited.evaluator_policy_id,
                    "policy_version": audited.evaluator_policy_version,
                    "nodes": audited.evaluator_nodes,
                },
                "fresh_process": {
                    "command": command,
                    "command_sha256": hashlib.sha256(
                        _canonical_pretty(command)
                    ).hexdigest(),
                    "report_sha256": reproduced_report_ref.sha256,
                    "decision_sha256": reproduced_decision_ref.sha256,
                    "transcript_sha256": reproduced_report["promotion"][
                        "transcript"
                    ]["sha256"],
                },
                "comparison": {
                    "float_tolerance": FLOAT_TOLERANCE,
                    "float_count": comparison.float_count,
                    "maximum_absolute_float_difference": (
                        comparison.maximum_absolute_float_difference
                    ),
                    "exact_candidate_and_input_hashes": True,
                    "exact_transcript_sha256": True,
                },
            }
            payload = _canonical_pretty(receipt_value)
            _write_atomic_no_clobber(receipt_output, payload)
            receipt = ContentAddressedFile(
                receipt_output, hashlib.sha256(payload).hexdigest()
            )
            load_validation_reproduction_receipt(receipt)
            return receipt
        finally:
            report_output.unlink(missing_ok=True)
            decision_output.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ml.evaluation.validation_reproduction"
    )
    parser.add_argument("original_report", type=Path)
    parser.add_argument("original_decision", type=Path)
    parser.add_argument("ensemble_release", type=Path)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("training_frequency", type=Path)
    parser.add_argument("validation_dataset", type=Path)
    parser.add_argument("--original-report-sha256", required=True)
    parser.add_argument("--original-decision-sha256", required=True)
    parser.add_argument("--ensemble-sha256", required=True)
    parser.add_argument("--calibration-sha256", required=True)
    parser.add_argument("--training-frequency-sha256", required=True)
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--private-validation", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--expected-source-revision", required=True)
    parser.add_argument("--pnpm-lock-sha256", required=True)
    parser.add_argument("--python-requirements-sha256", required=True)
    parser.add_argument("--python-project-sha256", required=True)
    parser.add_argument("--engine-binary-sha256", required=True)
    parser.add_argument("--engine-fingerprint", required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--catalog", action="append", type=Path, default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    environment = attest_clean_environment(
        arguments.repository_root,
        expected_source_revision=arguments.expected_source_revision,
        expected_pnpm_lock_sha256=arguments.pnpm_lock_sha256,
        expected_python_requirements_sha256=(
            arguments.python_requirements_sha256
        ),
        expected_python_project_sha256=arguments.python_project_sha256,
    )
    catalogs = tuple(arguments.catalog) or (
        arguments.repository_root
        / "engine/data/catalog/observed-drawbacks.json",
    )
    receipt = reproduce_validation(
        repository=arguments.repository_root,
        original_report=ContentAddressedFile(
            arguments.original_report, arguments.original_report_sha256
        ),
        original_decision=ContentAddressedFile(
            arguments.original_decision, arguments.original_decision_sha256
        ),
        ensemble=ContentAddressedFile(
            arguments.ensemble_release, arguments.ensemble_sha256
        ),
        calibration=ContentAddressedFile(
            arguments.calibration, arguments.calibration_sha256
        ),
        training_frequency=TrainingFrequencyReference(
            arguments.training_frequency,
            arguments.training_frequency_sha256,
        ),
        public_root=arguments.public_root,
        private_validation=arguments.private_validation,
        validation_dataset=arguments.validation_dataset,
        catalogs=catalogs,
        receipt_output=arguments.receipt_output,
        environment=environment,
        expected_engine_binary_sha256=arguments.engine_binary_sha256,
        expected_engine_fingerprint=arguments.engine_fingerprint,
        batch_size=arguments.batch_size,
    )
    print(json.dumps({"file": receipt.path.name, "sha256": receipt.sha256}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
