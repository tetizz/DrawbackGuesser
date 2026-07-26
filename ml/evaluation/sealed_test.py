"""One-shot, fail-closed execution of the signed sealed-test plan."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Mapping, Sequence

from ml.training.drawback_ml.corpus_contract import open_audited_private_corpus_split

from .ensemble_calibration import ContentAddressedFile
from .promotion_evaluator import (
    BROWSER_VIEW,
    PREPARED_VIEW,
    PromotionReport,
    evaluate_candidate_partition,
)
from .release_selection_bundle import ContentAddressedJson
from .review_authorization import (
    SEALED_REPORT_FORMAT,
    ReviewAuthorizationError,
    load_authorization_receipt,
)
from .training_frequency import ContentAddressedFile as TrainingFrequencyReference
from .validation_gate import (
    BOOTSTRAP_SEED,
    REPORT_FORMAT as VALIDATION_REPORT_FORMAT,
    decide_validation_gate,
    load_validation_gate_document,
)


VERSION = 1
DECISION_FORMAT = "drawbacktrainer-sealed-test-decision"
USE_FORMAT = "drawbacktrainer-sealed-test-use"
AUTHORIZATION_FILE_ENV = "DRAWBACKTRAINER_AUTHORIZATION_FILE"
AUTHORIZATION_SHA_ENV = "DRAWBACKTRAINER_AUTHORIZATION_SHA256"
_SHA256 = frozenset("0123456789abcdef")


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2)
        + "\n"
    ).encode("utf-8")


def _binding(value: object, label: str) -> tuple[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"file", "sha256"}:
        raise ReviewAuthorizationError(f"{label} binding is invalid")
    name, digest = value["file"], value["sha256"]
    if not isinstance(name, str) or Path(name).name != name or name in {".", ".."}:
        raise ReviewAuthorizationError(f"{label} file must be a safe basename")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in _SHA256 for character in digest)
    ):
        raise ReviewAuthorizationError(f"{label} SHA-256 is invalid")
    return name, digest


def _write_once(path: Path, payload: bytes, label: str) -> None:
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
            raise ReviewAuthorizationError(f"refusing to reuse {label}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _publish_pair(
    report_path: Path,
    report_payload: bytes,
    decision_path: Path,
    decision_payload: bytes,
) -> None:
    """Publish decision last as the completion marker, without partial evidence."""

    if report_path.exists() or decision_path.exists():
        raise ReviewAuthorizationError("sealed outputs already exist")
    staged: list[Path] = []
    published_report = False
    try:
        for path, payload in (
            (report_path, report_payload),
            (decision_path, decision_payload),
        ):
            descriptor, name = tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
            )
            staged_path = Path(name)
            staged.append(staged_path)
            with os.fdopen(descriptor, "wb") as target:
                target.write(payload)
                target.flush()
                os.fsync(target.fileno())
        os.link(staged[0], report_path)
        published_report = True
        os.link(staged[1], decision_path)
    except FileExistsError as error:
        if published_report:
            report_path.unlink(missing_ok=True)
        raise ReviewAuthorizationError("refusing to overwrite sealed outputs") from error
    except BaseException:
        if published_report:
            report_path.unlink(missing_ok=True)
        raise
    finally:
        for path in staged:
            path.unlink(missing_ok=True)


def _file_sha256(path: Path, label: str) -> str:
    hasher = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                hasher.update(chunk)
    except OSError as error:
        raise ReviewAuthorizationError(f"cannot read {label}") from error
    return hasher.hexdigest()


def _digest(path: Path, expected: str, label: str) -> None:
    actual = _file_sha256(path, label)
    if actual != expected:
        raise ReviewAuthorizationError(f"{label} SHA-256 does not match")


def _source_identity(directory: Path) -> tuple[Path, str]:
    process = subprocess.run(
        ("git", "rev-parse", "--show-toplevel", "HEAD"),
        cwd=directory,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.returncode != 0:
        raise ReviewAuthorizationError("cannot authenticate running source revision")
    cleanliness = subprocess.run(
        ("git", "status", "--porcelain", "--untracked-files=all"),
        cwd=directory,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if cleanliness.returncode != 0 or cleanliness.stdout:
        raise ReviewAuthorizationError("sealed test requires a clean source tree")
    lines = process.stdout.splitlines()
    if len(lines) != 2:
        raise ReviewAuthorizationError("running source identity is invalid")
    return Path(lines[0]).resolve(), lines[1]


@contextmanager
def _pinned_sealed_inputs(
    directory: Path, inputs: Mapping[str, object]
):
    with tempfile.TemporaryDirectory(
        prefix=".sealed-inputs.", dir=directory
    ) as raw_pinned:
        pinned = Path(raw_pinned)
        resolved: dict[str, tuple[Path, str]] = {}
        for key in ("public_root", "private_test", "dataset"):
            name, digest = _binding(inputs[key], f"test_plan.inputs.{key}")
            source = directory / name
            target = pinned / name
            hasher = hashlib.sha256()
            try:
                with source.open("rb") as sealed, target.open("xb") as copy:
                    while chunk := sealed.read(1024 * 1024):
                        hasher.update(chunk)
                        copy.write(chunk)
                    copy.flush()
                    os.fsync(copy.fileno())
            except OSError as error:
                raise ReviewAuthorizationError(f"cannot read sealed {key}") from error
            if hasher.hexdigest() != digest:
                raise ReviewAuthorizationError(f"sealed {key} SHA-256 does not match")
            resolved[key] = (target, digest)
        yield resolved


def _candidate_metrics(report: PromotionReport) -> dict[str, object]:
    result: dict[str, object] = {}
    for view_name in (PREPARED_VIEW, BROWSER_VIEW):
        view = report.views[view_name]
        system = view.systems["calibrated-ensemble"]
        colors: dict[str, object] = {}
        for color in ("white", "black"):
            head = getattr(system, color)
            if head is None:
                raise ReviewAuthorizationError("sealed metrics are incomplete")
            colors[color] = {
                "game_normalized_top_1_accuracy": head.game_normalized_top_1_accuracy,
                "game_normalized_top_3_accuracy": head.game_normalized_top_3_accuracy,
                "game_normalized_top_5_accuracy": head.game_normalized_top_5_accuracy,
                "negative_log_likelihood": head.negative_log_likelihood,
                "brier_score": head.brier_score,
                "expected_calibration_error": head.expected_calibration_error,
            }
        result[view_name] = colors
    return result


def _validation_metrics(document: Mapping[str, object]) -> Mapping[str, object]:
    promotion = document.get("promotion")
    if not isinstance(promotion, Mapping):
        raise ReviewAuthorizationError("validation report has no promotion evidence")
    views = promotion.get("views")
    if not isinstance(views, Mapping):
        raise ReviewAuthorizationError("validation report views are invalid")
    result: dict[str, object] = {}
    for view_name in (PREPARED_VIEW, BROWSER_VIEW):
        view = views.get(view_name)
        if not isinstance(view, Mapping):
            raise ReviewAuthorizationError("validation report view is missing")
        systems = view.get("systems")
        system = systems.get("calibrated-ensemble") if isinstance(systems, Mapping) else None
        if not isinstance(system, Mapping):
            raise ReviewAuthorizationError("validation candidate metrics are missing")
        result[view_name] = system
    return result


def _test_decision(
    report: PromotionReport, validation: Mapping[str, object]
) -> tuple[bool, list[dict[str, object]]]:
    # Reuse the frozen gate implementation.  Only identity, hard exactness,
    # absolute and horizon gates are test requirements; comparator/bootstrap
    # selection gates remain validation-only.
    gate_report = replace(
        report, partition="validation-gate", bootstrap_seed=BOOTSTRAP_SEED
    )
    frozen = decide_validation_gate(gate_report)
    prefixes = ("identity.", "prepared-182.white.absolute.", "prepared-182.black.absolute.",
                "browser-180.white.absolute.", "browser-180.black.absolute.",
                "prepared-182.white.horizon.", "prepared-182.black.horizon.",
                "exactness.")
    results = [
        {"gate_id": item.gate_id, "status": item.status}
        for item in frozen.results
        if item.gate_id.startswith(prefixes)
        and item.gate_id not in {"identity.partition", "identity.bootstrap-seed"}
    ]
    test_metrics = _candidate_metrics(report)
    validation_metrics = _validation_metrics(validation)
    limits = {
        "game_normalized_top_1_accuracy": (-0.03, "minimum"),
        "game_normalized_top_3_accuracy": (-0.04, "minimum"),
        "negative_log_likelihood": (0.20, "maximum"),
        "brier_score": (0.020, "maximum"),
        "expected_calibration_error": (0.030, "maximum"),
    }
    for view in (PREPARED_VIEW, BROWSER_VIEW):
        for color in ("white", "black"):
            test_color = test_metrics[view][color]
            validation_color = validation_metrics[view][color]
            if not isinstance(test_color, Mapping) or not isinstance(validation_color, Mapping):
                raise ReviewAuthorizationError("stability metrics are invalid")
            for metric, (limit, direction) in limits.items():
                test_value = float(test_color[metric])
                validation_value = float(validation_color[metric])
                passed = (
                    test_value >= validation_value + limit
                    if direction == "minimum"
                    else test_value <= validation_value + limit
                )
                results.append(
                    {
                        "gate_id": f"{view}.{color}.stability.{metric}",
                        "status": "passed" if passed else "failed",
                    }
                )
    return all(item["status"] == "passed" for item in results), results


def execute_authorized_test(
    *,
    authorization: ContentAddressedFile,
    invocation: Sequence[str],
    directory: Path,
) -> tuple[Path, Path, bool]:
    """Authenticate, consume once, then and only then resolve sealed inputs."""

    receipt = load_authorization_receipt(authorization)
    plan = receipt["authorized_test_plan"]
    if not isinstance(plan, Mapping) or list(invocation) != plan["argv"]:
        raise ReviewAuthorizationError("process argv is not the exact signed test plan")
    directory = directory.resolve()
    if authorization.path.resolve().parent != directory or Path.cwd().resolve() != directory:
        raise ReviewAuthorizationError("sealed test must run beside its authorization")

    dependencies = receipt["dependencies"]
    if not isinstance(dependencies, Mapping):
        raise ReviewAuthorizationError("authorization dependencies are invalid")
    source_root, source_revision = _source_identity(directory)
    if source_revision != dependencies["source_revision"]:
        raise ReviewAuthorizationError("running source revision is not authorized")
    runtime_dependencies = {
        "pnpm_lock": source_root / "pnpm-lock.yaml",
        "python_requirements": source_root / "ml" / "requirements.txt",
        "python_project": source_root / "ml" / "pyproject.toml",
    }
    for key, runtime_path in runtime_dependencies.items():
        _name, expected = _binding(dependencies[key], f"dependencies.{key}")
        _digest(runtime_path, expected, f"runtime {key}")
    python_runtime = receipt["authorized_python_runtime"]
    if not isinstance(python_runtime, Mapping):
        raise ReviewAuthorizationError("authorized Python runtime is invalid")
    python_sha = _file_sha256(Path(sys.executable), "Python executable")
    if python_sha != python_runtime["executable_sha256"]:
        raise ReviewAuthorizationError("Python executable is not authorized")
    if sys.version != python_runtime["version"]:
        raise ReviewAuthorizationError("Python version is not authorized")

    outputs = plan["output_basenames"]
    inputs = plan["inputs"]
    if not isinstance(outputs, Mapping) or not isinstance(inputs, Mapping):
        raise ReviewAuthorizationError("authorized test plan is invalid")
    report_name = str(outputs["report"])
    decision_name = str(outputs["decision"])
    report_path, decision_path = directory / report_name, directory / decision_name
    if report_path.exists() or decision_path.exists():
        raise ReviewAuthorizationError("sealed outputs already exist")

    use_path = directory / f".sealed-test-use-{authorization.sha256}.json"
    _write_once(
        use_path,
        _canonical(
            {
                "format": USE_FORMAT,
                "version": VERSION,
                "authorization_sha256": authorization.sha256,
                "plan_id": plan["plan_id"],
                "argv_sha256": hashlib.sha256(_canonical(list(invocation))).hexdigest(),
            }
        ),
        "sealed-test authorization",
    )

    candidate = receipt["authorized_candidate"]
    validation = receipt["authorized_validation"]
    if not isinstance(candidate, Mapping) or not isinstance(validation, Mapping):
        raise ReviewAuthorizationError("authorization bindings are invalid")
    ensemble_name, ensemble_sha = _binding(candidate["ensemble_release"], "ensemble release")
    calibration_name, calibration_sha = _binding(candidate["calibration"], "calibration")
    frequency_name, frequency_sha = _binding(candidate["training_frequency"], "training frequency")
    validation_name, validation_sha = _binding(validation["report"], "validation report")
    validation_path = directory / validation_name
    validation_document = load_validation_gate_document(
        ContentAddressedFile(validation_path, validation_sha),
        VALIDATION_REPORT_FORMAT,
    )

    # This is the first point at which sealed basenames are resolved or read.
    with _pinned_sealed_inputs(directory, inputs) as resolved:
        with open_audited_private_corpus_split(
            resolved["public_root"][0],
            resolved["private_test"][0],
            resolved["dataset"][0],
            "test",
        ) as lease:
            report = evaluate_candidate_partition(
                lease=lease,
                partition="test",
                ensemble_release=ContentAddressedJson(directory / ensemble_name, ensemble_sha),
                calibration=ContentAddressedFile(directory / calibration_name, calibration_sha),
                training_frequency=TrainingFrequencyReference(directory / frequency_name, frequency_sha),
                catalogs=(
                    source_root
                    / "engine"
                    / "data"
                    / "catalog"
                    / "observed-drawbacks.json",
                ),
                bootstrap_seed=int(plan["bootstrap_seed"]),
                batch_size=256,
            )
    passed, gates = _test_decision(report, validation_document)
    metrics = _candidate_metrics(report)
    evaluator = receipt["evaluator"]
    browser_runtime = receipt["authorized_browser_runtime"]
    report_value = {
        "format": SEALED_REPORT_FORMAT,
        "version": VERSION,
        "protocol_id": receipt["protocol_id"],
        "plan_id": plan["plan_id"],
        "authorization_sha256": authorization.sha256,
        "bindings": {
            "source_revision": source_revision,
            "dependency_sha256": {
                key: _binding(dependencies[key], f"dependencies.{key}")[1]
                for key in sorted(runtime_dependencies)
            },
            "sealed_input_sha256": {
                key: _binding(inputs[key], f"test_plan.inputs.{key}")[1]
                for key in sorted(inputs)
            },
            "test_plan_sha256": hashlib.sha256(_canonical(plan)).hexdigest(),
            "argv_sha256": hashlib.sha256(_canonical(list(invocation))).hexdigest(),
            "ensemble_release_sha256": report.ensemble_release_sha256,
            "calibration_sha256": report.calibration_sha256,
            "training_frequency_sha256": report.training_frequency_sha256,
            "partition_seed_sha256": report.partition_seed_sha256,
            "evaluator_fingerprint": evaluator["fingerprint"],
            "browser_binary_sha256": browser_runtime["binary_sha256"],
            "browser_version": browser_runtime["version"],
            "transcript_sha256": report.transcript.sha256,
            "python_executable_sha256": python_sha,
            "python_version": sys.version,
        },
        "bootstrap": {
            "seed": report.bootstrap_seed,
            "method": "complete-game-paired-percentile-linear-v1",
            "replicates": 10_000,
        },
        "metrics": metrics,
        "counts": {
            "move_examples": report.move_examples,
            "transcript_records": report.transcript.record_count,
        },
    }
    report_payload = _canonical(report_value)
    decision_value = {
        "format": DECISION_FORMAT,
        "version": VERSION,
        "protocol_id": receipt["protocol_id"],
        "sealed_report": {
            "file": report_name,
            "sha256": hashlib.sha256(report_payload).hexdigest(),
        },
        "passed": passed,
        "gates": gates,
    }
    _publish_pair(
        report_path,
        report_payload,
        decision_path,
        _canonical(decision_value),
    )
    return report_path, decision_path, passed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m ml.evaluation.sealed_test")
    parser.add_argument("public_root", type=Path)
    parser.add_argument("private_test", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--decision-output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    build_parser().parse_args(raw)  # syntax only; signed plan is authoritative
    authorization_file = os.environ.get(AUTHORIZATION_FILE_ENV)
    authorization_sha = os.environ.get(AUTHORIZATION_SHA_ENV)
    if not authorization_file or not authorization_sha:
        raise ReviewAuthorizationError("authorization capability environment is missing")
    invocation = ["python", "-m", "ml.evaluation.sealed_test", *raw]
    report, decision, passed = execute_authorized_test(
        authorization=ContentAddressedFile(Path(authorization_file), authorization_sha),
        invocation=invocation,
        directory=Path.cwd(),
    )
    print(json.dumps({"report": report.name, "decision": decision.name, "passed": passed}))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
