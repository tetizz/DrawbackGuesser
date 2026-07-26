"""Two-reviewer authorization for a future one-shot sealed-test opener.

This module authenticates release evidence and a precommitted test plan.  It
has no argument for a sealed manifest or dataset path and never resolves the
three sealed input bindings recorded in the approval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePath
import re
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

from .ensemble_calibration import ContentAddressedFile
from .validation_reproduction import (
    FLOAT_TOLERANCE as REPRODUCTION_FLOAT_TOLERANCE,
    FORMAT as REPRODUCTION_FORMAT,
    load_validation_reproduction_receipt,
)


FORMAT = "drawbacktrainer-sealed-test-review-approval"
RECEIPT_FORMAT = "drawbacktrainer-sealed-test-authorization"
VERSION = 1
PROTOCOL_ID = "current-catalog-182-v2"
SIGNATURE_NAMESPACE = "drawbacktrainer-sealed-test-review-v1"
VALIDATION_DECISION_FORMAT = "drawbacktrainer-validation-gate-decision"
VALIDATION_REPORT_FORMAT = "drawbacktrainer-validation-gate-report"
PARITY_FORMAT = "drawbacktrainer-browser-parity-evidence"
SEALED_REPORT_FORMAT = "drawbacktrainer-sealed-test-report"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
SOURCE_REVISION = re.compile(r"^[0-9a-f]{40}$")
FINGERPRINT = re.compile(r"SHA256:[A-Za-z0-9+/]{43}")
_CANDIDATE_KEYS = frozenset(
    {"ensemble_release", "calibration", "browser_artifact", "training_frequency"}
)
_TEST_INPUT_KEYS = frozenset({"public_root", "private_test", "dataset"})
_TEST_OUTPUT_KEYS = frozenset({"report", "decision"})


class ReviewAuthorizationError(ValueError):
    """Raised before authorization when review evidence is not exact."""


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ReviewAuthorizationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            separators=(",", ": "),
        )
        + "\n"
    ).encode("utf-8")


def _strict_document(payload: bytes, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(payload, object_pairs_hook=_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReviewAuthorizationError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, Mapping):
        raise ReviewAuthorizationError(f"{label} must contain a JSON object")
    try:
        canonical = _canonical(value)
    except (TypeError, ValueError) as error:
        raise ReviewAuthorizationError(
            f"{label} contains noncanonical JSON values"
        ) from error
    if payload != canonical:
        raise ReviewAuthorizationError(f"{label} is not canonical JSON")
    return value


def _exact(value: Mapping[str, Any], keys: set[str] | frozenset[str], label: str) -> None:
    if set(value) != keys:
        raise ReviewAuthorizationError(f"{label} fields are not exact")


def _object(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReviewAuthorizationError(f"{label} must be an object")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReviewAuthorizationError(f"{label} must be a non-empty string")
    return value


def _sha(value: object, label: str) -> str:
    result = _string(value, label)
    if SHA256.fullmatch(result) is None:
        raise ReviewAuthorizationError(f"{label} must be a lowercase SHA-256")
    return result


def _basename(value: object, label: str) -> str:
    result = _string(value, label)
    path = PurePath(result)
    if (
        path.name != result
        or result in {".", ".."}
        or "/" in result
        or "\\" in result
        or "\x00" in result
    ):
        raise ReviewAuthorizationError(f"{label} must be a safe basename")
    return result


def _binding(value: object, label: str) -> tuple[str, str]:
    mapping = _object(value, label)
    _exact(mapping, {"file", "sha256"}, label)
    return _basename(mapping["file"], f"{label}.file"), _sha(
        mapping["sha256"], f"{label}.sha256"
    )


def _read_exact(path: Path, expected_sha256: str, label: str) -> bytes:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ReviewAuthorizationError(f"cannot read {label}") from error
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise ReviewAuthorizationError(f"{label} SHA-256 does not match")
    return payload


def _bound_bytes(
    directory: Path, value: object, label: str
) -> tuple[Path, str, bytes]:
    name, digest = _binding(value, label)
    path = directory / name
    return path, digest, _read_exact(path, digest, label)


def _verify_validation_decision(payload: bytes, approval: Mapping[str, Any]) -> None:
    value = _strict_document(payload, "validation decision")
    _exact(
        value,
        {
            "format",
            "version",
            "protocol_id",
            "validation_report",
            "passed",
            "missing_count",
            "failed_count",
            "threshold_contract_sha256",
            "bootstrap",
            "results",
        },
        "validation decision",
    )
    if (
        value.get("format") != VALIDATION_DECISION_FORMAT
        or value.get("version") != VERSION
        or value.get("protocol_id") != PROTOCOL_ID
        or value.get("passed") is not True
        or value.get("missing_count") != 0
        or value.get("failed_count") != 0
    ):
        raise ReviewAuthorizationError(
            "validation decision is not an exact passing protocol decision"
        )
    threshold_sha = _sha(
        value.get("threshold_contract_sha256"),
        "validation decision threshold contract",
    )
    results = value.get("results")
    if not isinstance(results, list) or not results:
        raise ReviewAuthorizationError(
            "validation decision must contain exhaustive passing results"
        )
    gate_ids: set[str] = set()
    for index, candidate in enumerate(results):
        result = _object(candidate, f"validation result {index}")
        _exact(
            result,
            {"gate_id", "status", "actual", "requirement", "reason"},
            f"validation result {index}",
        )
        gate_id = _string(result.get("gate_id"), f"validation result {index}.gate_id")
        _string(
            result.get("requirement"),
            f"validation result {index}.requirement",
        )
        if (
            gate_id in gate_ids
            or result.get("status") != "passed"
            or result.get("reason") is not None
        ):
            raise ReviewAuthorizationError(
                "validation decision results are duplicate or not passing"
            )
        gate_ids.add(gate_id)
    contract = [
        {
            "gate_id": result["gate_id"],
            "requirement": result["requirement"],
        }
        for result in results
    ]
    recomputed = hashlib.sha256(
        json.dumps(
            contract, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    if threshold_sha != recomputed:
        raise ReviewAuthorizationError(
            "validation threshold contract SHA-256 is stale"
        )
    report = _object(value.get("validation_report"), "validation decision report")
    report_name, report_sha = _binding(
        approval["validation_report"], "approval.validation_report"
    )
    if report.get("file") != report_name or report.get("sha256") != report_sha:
        raise ReviewAuthorizationError(
            "validation decision binds a different validation report"
        )


def _verify_validation_report(payload: bytes, approval: Mapping[str, Any]) -> None:
    value = _strict_document(payload, "validation report")
    _exact(
        value,
        {"format", "version", "protocol", "bindings", "promotion"},
        "validation report",
    )
    protocol = _object(value["protocol"], "validation report protocol")
    bindings = _object(value["bindings"], "validation report bindings")
    _exact(
        bindings,
        {
            "ensemble_release_sha256",
            "calibration_sha256",
            "training_frequency_sha256",
        },
        "validation report bindings",
    )
    candidates = _object(approval["candidate"], "approval.candidate")
    if (
        value["format"] != VALIDATION_REPORT_FORMAT
        or value["version"] != VERSION
        or protocol.get("id") != PROTOCOL_ID
        or bindings["ensemble_release_sha256"]
        != _binding(
            candidates["ensemble_release"], "candidate.ensemble_release"
        )[1]
        or bindings["calibration_sha256"]
        != _binding(candidates["calibration"], "candidate.calibration")[1]
        or bindings["training_frequency_sha256"]
        != _binding(
            candidates["training_frequency"], "candidate.training_frequency"
        )[1]
    ):
        raise ReviewAuthorizationError(
            "validation report binds a different protocol or candidate"
        )


def _verify_reproduction(
    path: Path,
    digest: str,
    approval: Mapping[str, Any],
) -> Mapping[str, Any]:
    try:
        value = load_validation_reproduction_receipt(
            ContentAddressedFile(path, digest)
        )
    except ValueError as error:
        raise ReviewAuthorizationError(
            "validation reproduction receipt is invalid"
        ) from error
    if value.get("format") != REPRODUCTION_FORMAT:
        raise ReviewAuthorizationError(
            "validation reproduction receipt format is not authoritative"
        )
    candidates = _object(approval["candidate"], "approval.candidate")
    candidate_inputs = _object(
        value.get("candidate_inputs"), "reproduction candidate_inputs"
    )
    original = _object(value.get("original"), "reproduction original")
    expected_original = {
        "report": {
            "file": _binding(
                approval["validation_report"], "approval.validation_report"
            )[0],
            "sha256": _binding(
                approval["validation_report"], "approval.validation_report"
            )[1],
        },
        "decision": {
            "file": _binding(
                approval["validation_decision"], "approval.validation_decision"
            )[0],
            "sha256": _binding(
                approval["validation_decision"], "approval.validation_decision"
            )[1],
        },
    }
    if dict(original) != expected_original:
        raise ReviewAuthorizationError(
            "reproduction receipt binds different validation evidence"
        )
    if (
        candidate_inputs.get("ensemble_release_sha256")
        != _binding(
            candidates["ensemble_release"], "candidate.ensemble_release"
        )[1]
        or candidate_inputs.get("calibration_sha256")
        != _binding(candidates["calibration"], "candidate.calibration")[1]
        or candidate_inputs.get("training_frequency_sha256")
        != _binding(
            candidates["training_frequency"], "candidate.training_frequency"
        )[1]
    ):
        raise ReviewAuthorizationError(
            "reproduction receipt binds different candidate evidence"
        )
    comparison = _object(value.get("comparison"), "reproduction comparison")
    maximum = comparison.get("maximum_absolute_float_difference")
    if (
        comparison.get("float_tolerance") != REPRODUCTION_FLOAT_TOLERANCE
        or isinstance(maximum, bool)
        or not isinstance(maximum, (int, float))
        or not 0.0 <= float(maximum) <= REPRODUCTION_FLOAT_TOLERANCE
    ):
        raise ReviewAuthorizationError(
            "reproduction receipt exceeds the frozen tolerance"
        )

    dependencies = _object(approval["dependencies"], "approval.dependencies")
    environment = _object(value.get("environment"), "reproduction environment")
    if (
        environment.get("source_revision") != dependencies["source_revision"]
        or environment.get("pnpm_lock_sha256")
        != _binding(dependencies["pnpm_lock"], "dependencies.pnpm_lock")[1]
        or environment.get("python_requirements_sha256")
        != _binding(
            dependencies["python_requirements"],
            "dependencies.python_requirements",
        )[1]
        or environment.get("python_project_sha256")
        != _binding(
            dependencies["python_project"], "dependencies.python_project"
        )[1]
    ):
        raise ReviewAuthorizationError(
            "reproduction receipt binds different dependency identities"
        )
    approved_evaluator = _object(approval["evaluator"], "approval.evaluator")
    reproduced_evaluator = _object(
        value.get("evaluator"), "reproduction evaluator"
    )
    if (
        reproduced_evaluator.get("engine_binary_sha256")
        != _binding(approved_evaluator["binary"], "evaluator.binary")[1]
        or reproduced_evaluator.get("engine_fingerprint")
        != approved_evaluator["fingerprint"]
        or reproduced_evaluator.get("policy_id")
        != approved_evaluator["policy_id"]
        or reproduced_evaluator.get("policy_version")
        != approved_evaluator["policy_version"]
        or reproduced_evaluator.get("nodes") != approved_evaluator["nodes"]
    ):
        raise ReviewAuthorizationError(
            "reproduction receipt binds a different evaluator identity"
        )
    _sha(
        environment.get("python_executable_sha256"),
        "reproduction environment.python_executable_sha256",
    )
    _string(
        environment.get("python_version"),
        "reproduction environment.python_version",
    )
    return value


def _verify_parity(
    payload: bytes, approval: Mapping[str, Any]
) -> Mapping[str, Any]:
    value = _strict_document(payload, "browser parity evidence")
    _exact(
        value,
        {
            "format",
            "version",
            "protocol_id",
            "browser_artifact_sha256",
            "calibration_sha256",
            "passed",
            "max_absolute_difference",
            "top_k_identical",
            "hard_zero_sets_identical",
            "worker_e2e_passed",
            "parity_input_sha256",
            "transcript_sha256",
            "fixture_sha256",
            "partition_selection_sha256",
            "ensemble_sha256",
            "source_revision",
            "pnpm_lock_sha256",
            "browser_binary_sha256",
            "browser_version",
            "public_fixture_sha256",
        },
        "browser parity evidence",
    )
    difference = value["max_absolute_difference"]
    if (
        value["format"] != PARITY_FORMAT
        or value["version"] != VERSION
        or value["protocol_id"] != PROTOCOL_ID
        or value["passed"] is not True
        or value["top_k_identical"] is not True
        or value["hard_zero_sets_identical"] is not True
        or value["worker_e2e_passed"] is not True
        or isinstance(difference, bool)
        or not isinstance(difference, (int, float))
        or not 0.0 <= float(difference) <= 1e-6
    ):
        raise ReviewAuthorizationError("browser parity evidence is not passing")
    candidates = _object(approval["candidate"], "approval.candidate")
    dependencies = _object(approval["dependencies"], "approval.dependencies")
    for name in (
        "parity_input_sha256",
        "transcript_sha256",
        "fixture_sha256",
        "partition_selection_sha256",
        "ensemble_sha256",
        "pnpm_lock_sha256",
        "browser_binary_sha256",
        "public_fixture_sha256",
    ):
        _sha(value[name], f"browser parity evidence.{name}")
    if SOURCE_REVISION.fullmatch(
        _string(value["source_revision"], "browser parity evidence.source_revision")
    ) is None:
        raise ReviewAuthorizationError("browser parity source revision is invalid")
    _string(value["browser_version"], "browser parity evidence.browser_version")
    if value["browser_artifact_sha256"] != _binding(
        candidates["browser_artifact"], "candidate.browser_artifact"
    )[1] or value["calibration_sha256"] != _binding(
        candidates["calibration"], "candidate.calibration"
    )[1] or value["ensemble_sha256"] != _binding(
        candidates["ensemble_release"], "candidate.ensemble_release"
    )[1] or value["source_revision"] != dependencies["source_revision"] or value[
        "pnpm_lock_sha256"
    ] != _binding(dependencies["pnpm_lock"], "dependencies.pnpm_lock")[1]:
        raise ReviewAuthorizationError(
            "browser parity evidence binds different release evidence"
        )
    return value


def _verify_test_plan(value: object) -> Mapping[str, Any]:
    plan = _object(value, "test_plan")
    _exact(
        plan,
        {
            "plan_id",
            "bootstrap_seed",
            "report_schema",
            "argv",
            "inputs",
            "output_basenames",
        },
        "test_plan",
    )
    _string(plan["plan_id"], "test_plan.plan_id")
    bootstrap = plan["bootstrap_seed"]
    if (
        not isinstance(bootstrap, int)
        or isinstance(bootstrap, bool)
        or bootstrap < 0
        or bootstrap > 9_007_199_254_740_991
    ):
        raise ReviewAuthorizationError("test_plan.bootstrap_seed is invalid")
    schema = _object(plan["report_schema"], "test_plan.report_schema")
    _exact(schema, {"format", "version"}, "test_plan.report_schema")
    if schema["format"] != SEALED_REPORT_FORMAT:
        raise ReviewAuthorizationError("test report schema format is not frozen")
    if (
        not isinstance(schema["version"], int)
        or isinstance(schema["version"], bool)
        or schema["version"] != VERSION
    ):
        raise ReviewAuthorizationError("test report schema version is invalid")
    argv = plan["argv"]
    if (
        not isinstance(argv, list)
        or not argv
        or any(
            not isinstance(item, str) or not item or "\x00" in item
            for item in argv
        )
    ):
        raise ReviewAuthorizationError(
            "test_plan.argv must be an exact non-empty string array"
        )
    inputs = _object(plan["inputs"], "test_plan.inputs")
    _exact(inputs, _TEST_INPUT_KEYS, "test_plan.inputs")
    input_names = [
        _binding(inputs[name], f"test_plan.inputs.{name}")[0]
        for name in sorted(_TEST_INPUT_KEYS)
    ]
    if len(set(input_names)) != len(input_names):
        raise ReviewAuthorizationError("test input basenames must be distinct")
    outputs = _object(plan["output_basenames"], "test_plan.output_basenames")
    _exact(outputs, _TEST_OUTPUT_KEYS, "test_plan.output_basenames")
    output_names = [
        _basename(outputs[name], f"test_plan.output_basenames.{name}")
        for name in sorted(_TEST_OUTPUT_KEYS)
    ]
    if len(set(output_names)) != len(output_names):
        raise ReviewAuthorizationError("test output basenames must be distinct")
    if set(input_names).intersection(output_names):
        raise ReviewAuthorizationError(
            "test input and output basenames must not overlap"
        )
    for name in (*input_names, *output_names):
        if argv.count(name) != 1:
            raise ReviewAuthorizationError(
                f"test_plan.argv must contain bound basename exactly once: {name}"
            )
    return plan


def _validate_approval(value: Mapping[str, Any]) -> None:
    _exact(
        value,
        {
            "format",
            "version",
            "protocol",
            "candidate",
            "validation_report",
            "validation_decision",
            "reproduction_receipt",
            "parity_evidence",
            "test_plan",
            "dependencies",
            "evaluator",
            "review_policy",
        },
        "approval",
    )
    if value["format"] != FORMAT or value["version"] != VERSION:
        raise ReviewAuthorizationError("unsupported review approval format")
    protocol = _object(value["protocol"], "approval.protocol")
    _exact(protocol, {"id", "document"}, "approval.protocol")
    if protocol["id"] != PROTOCOL_ID:
        raise ReviewAuthorizationError("approval protocol ID is not frozen")
    _binding(protocol["document"], "approval.protocol.document")
    candidates = _object(value["candidate"], "approval.candidate")
    _exact(candidates, _CANDIDATE_KEYS, "approval.candidate")
    for name in sorted(_CANDIDATE_KEYS):
        _binding(candidates[name], f"approval.candidate.{name}")
    for name in (
        "validation_report",
        "validation_decision",
        "reproduction_receipt",
        "parity_evidence",
    ):
        _binding(value[name], f"approval.{name}")
    _verify_test_plan(value["test_plan"])

    dependencies = _object(value["dependencies"], "approval.dependencies")
    _exact(
        dependencies,
        {
            "source_revision",
            "pnpm_lock",
            "python_requirements",
            "python_project",
        },
        "approval.dependencies",
    )
    if SOURCE_REVISION.fullmatch(
        _string(dependencies["source_revision"], "dependencies.source_revision")
    ) is None:
        raise ReviewAuthorizationError(
            "dependencies.source_revision must be a full lowercase Git SHA"
        )
    _binding(dependencies["pnpm_lock"], "dependencies.pnpm_lock")
    _binding(
        dependencies["python_requirements"], "dependencies.python_requirements"
    )
    _binding(dependencies["python_project"], "dependencies.python_project")

    evaluator = _object(value["evaluator"], "approval.evaluator")
    _exact(
        evaluator,
        {
            "binary",
            "options",
            "fingerprint",
            "policy_id",
            "policy_version",
            "nodes",
        },
        "approval.evaluator",
    )
    binary_sha = _binding(evaluator["binary"], "evaluator.binary")[1]
    options_sha = _binding(evaluator["options"], "evaluator.options")[1]
    fingerprint = _string(evaluator["fingerprint"], "evaluator.fingerprint")
    parts = fingerprint.split(":")
    if (
        len(parts) != 4
        or parts[2] != binary_sha
        or parts[3] != options_sha
    ):
        raise ReviewAuthorizationError(
            "evaluator fingerprint does not bind binary and options identities"
        )
    _string(evaluator["policy_id"], "evaluator.policy_id")
    for name in ("policy_version", "nodes"):
        if (
            not isinstance(evaluator[name], int)
            or isinstance(evaluator[name], bool)
            or evaluator[name] <= 0
        ):
            raise ReviewAuthorizationError(
                f"evaluator.{name} must be a positive integer"
            )

    policy = _object(value["review_policy"], "approval.review_policy")
    _exact(
        policy,
        {"signature_namespace", "allowed_signers_sha256", "required_reviewers"},
        "approval.review_policy",
    )
    if (
        policy["signature_namespace"] != SIGNATURE_NAMESPACE
        or policy["required_reviewers"] != 2
    ):
        raise ReviewAuthorizationError("review signature policy is not frozen")
    _sha(policy["allowed_signers_sha256"], "review_policy.allowed_signers_sha256")


def _verify_bound_evidence(
    approval: Mapping[str, Any], directory: Path
) -> Mapping[str, Any]:
    protocol = _object(approval["protocol"], "approval.protocol")
    _bound_bytes(directory, protocol["document"], "protocol document")
    candidates = _object(approval["candidate"], "approval.candidate")
    for name in sorted(_CANDIDATE_KEYS):
        _bound_bytes(directory, candidates[name], f"candidate {name}")
    _, _, validation_report = _bound_bytes(
        directory, approval["validation_report"], "validation report"
    )
    _verify_validation_report(validation_report, approval)
    _, _, validation_decision = _bound_bytes(
        directory, approval["validation_decision"], "validation decision"
    )
    _verify_validation_decision(validation_decision, approval)
    reproduction_path, reproduction_sha, _reproduction = _bound_bytes(
        directory, approval["reproduction_receipt"], "reproduction receipt"
    )
    reproduction_value = _verify_reproduction(
        reproduction_path, reproduction_sha, approval
    )
    _, _, parity = _bound_bytes(
        directory, approval["parity_evidence"], "browser parity evidence"
    )
    parity_value = _verify_parity(parity, approval)
    dependencies = _object(approval["dependencies"], "approval.dependencies")
    _bound_bytes(directory, dependencies["pnpm_lock"], "pnpm dependency lock")
    _bound_bytes(
        directory,
        dependencies["python_requirements"],
        "Python dependency lock",
    )
    _bound_bytes(
        directory,
        dependencies["python_project"],
        "Python project lock",
    )
    evaluator = _object(approval["evaluator"], "approval.evaluator")
    _bound_bytes(directory, evaluator["binary"], "evaluator binary")
    _bound_bytes(directory, evaluator["options"], "evaluator options")
    return {
        "parity": parity_value,
        "reproduction": reproduction_value,
    }


def _verify_signature(
    *,
    approval_payload: bytes,
    allowed_signers: Path,
    identity: str,
    signature: Path,
) -> str:
    if (
        not identity
        or "\x00" in identity
        or "\n" in identity
        or "\r" in identity
    ):
        raise ReviewAuthorizationError("reviewer identity is invalid")
    try:
        process = subprocess.run(
            (
                "ssh-keygen",
                "-Y",
                "verify",
                "-f",
                str(allowed_signers),
                "-I",
                identity,
                "-n",
                SIGNATURE_NAMESPACE,
                "-s",
                str(signature),
            ),
            input=approval_payload,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as error:
        raise ReviewAuthorizationError("cannot execute ssh-keygen") from error
    if process.returncode != 0:
        raise ReviewAuthorizationError(
            f"OpenSSH signature verification failed for {identity}"
        )
    output = (process.stdout + process.stderr).decode("utf-8", errors="replace")
    match = FINGERPRINT.search(output)
    if match is None:
        raise ReviewAuthorizationError(
            f"OpenSSH did not report a signing fingerprint for {identity}"
        )
    return match.group(0)


def _write_atomic_no_clobber(path: Path, payload: bytes) -> None:
    path = path.resolve()
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
            raise ReviewAuthorizationError(
                f"refusing to overwrite authorization receipt: {path}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)


def authorize_review(
    *,
    approval_path: Path,
    approval_sha256: str,
    allowed_signers_path: Path,
    allowed_signers_sha256: str,
    reviewers: Sequence[tuple[str, Path]],
    output: Path,
) -> Mapping[str, Any]:
    """Authenticate two reviews and publish a no-clobber authorization receipt."""

    approval_path = approval_path.resolve()
    directory = approval_path.parent
    if output.resolve().parent != directory:
        raise ReviewAuthorizationError(
            "authorization receipt must be beside the approval"
        )
    approval_sha256 = _sha(approval_sha256, "approval_sha256")
    approval_payload = _read_exact(
        approval_path, approval_sha256, "review approval"
    )
    approval = _strict_document(approval_payload, "review approval")
    _validate_approval(approval)
    verified = _verify_bound_evidence(approval, directory)
    parity_value = _object(verified["parity"], "verified parity")
    reproduction_value = _object(
        verified["reproduction"], "verified reproduction"
    )
    reproduction_environment = _object(
        reproduction_value["environment"], "reproduction environment"
    )

    allowed_signers_sha256 = _sha(
        allowed_signers_sha256, "allowed_signers_sha256"
    )
    allowed_signers_path = allowed_signers_path.resolve()
    allowed_payload = _read_exact(
        allowed_signers_path, allowed_signers_sha256, "allowed signers"
    )
    policy = _object(approval["review_policy"], "approval.review_policy")
    if policy["allowed_signers_sha256"] != allowed_signers_sha256:
        raise ReviewAuthorizationError(
            "approval binds a different allowed-signers policy"
        )
    if len(reviewers) != 2:
        raise ReviewAuthorizationError("exactly two reviewers are required")

    reviewer_receipts: list[dict[str, object]] = []
    identities: set[str] = set()
    fingerprints: set[str] = set()
    signature_hashes: set[str] = set()
    with tempfile.TemporaryDirectory(
        prefix=".review-signatures.", dir=directory
    ) as raw_pinned:
        pinned = Path(raw_pinned)
        pinned_allowed = pinned / "allowed_signers"
        pinned_allowed.write_bytes(allowed_payload)
        for index, (identity, signature_path) in enumerate(reviewers):
            signature_path = signature_path.resolve()
            try:
                signature_payload = signature_path.read_bytes()
            except OSError as error:
                raise ReviewAuthorizationError(
                    f"cannot read signature for {identity}"
                ) from error
            signature_sha = hashlib.sha256(signature_payload).hexdigest()
            pinned_signature = pinned / f"review-{index}.sig"
            pinned_signature.write_bytes(signature_payload)
            fingerprint = _verify_signature(
                approval_payload=approval_payload,
                allowed_signers=pinned_allowed,
                identity=identity,
                signature=pinned_signature,
            )
            if (
                identity in identities
                or fingerprint in fingerprints
                or signature_sha in signature_hashes
            ):
                raise ReviewAuthorizationError(
                    "reviewers, signing keys, and signatures must be distinct"
                )
            identities.add(identity)
            fingerprints.add(fingerprint)
            signature_hashes.add(signature_sha)
            reviewer_receipts.append(
                {
                    "identity": identity,
                    "key_fingerprint": fingerprint,
                    "signature": {
                        "file": signature_path.name,
                        "sha256": signature_sha,
                    },
                }
            )

    receipt = {
        "format": RECEIPT_FORMAT,
        "version": VERSION,
        "protocol_id": PROTOCOL_ID,
        "approval": {
            "file": approval_path.name,
            "sha256": approval_sha256,
        },
        "review_policy": {
            "signature_namespace": SIGNATURE_NAMESPACE,
            "allowed_signers": {
                "file": allowed_signers_path.name,
                "sha256": hashlib.sha256(allowed_payload).hexdigest(),
            },
            "required_reviewers": 2,
        },
        "reviewers": reviewer_receipts,
        "authorized_candidate": approval["candidate"],
        "authorized_validation": {
            "report": approval["validation_report"],
            "decision": approval["validation_decision"],
            "reproduction_receipt": approval["reproduction_receipt"],
            "parity_evidence": approval["parity_evidence"],
        },
        "authorized_browser_runtime": {
            "binary_sha256": parity_value["browser_binary_sha256"],
            "version": parity_value["browser_version"],
        },
        "authorized_python_runtime": {
            "executable_sha256": reproduction_environment[
                "python_executable_sha256"
            ],
            "version": reproduction_environment["python_version"],
        },
        "authorized_test_plan": approval["test_plan"],
        "dependencies": approval["dependencies"],
        "evaluator": approval["evaluator"],
    }
    payload = _canonical(receipt)
    _write_atomic_no_clobber(output, payload)
    published = _read_exact(
        output.resolve(), hashlib.sha256(payload).hexdigest(), "authorization receipt"
    )
    if published != payload:
        raise ReviewAuthorizationError("published authorization receipt changed")
    return receipt


def load_authorization_receipt(
    reference: ContentAddressedFile,
) -> Mapping[str, Any]:
    """Recursively authenticate a published authorization capability.

    This is the only entry point the sealed-test opener needs.  In particular,
    it verifies the approval and both detached signatures before returning the
    authorized plan; callers must not resolve sealed basenames first.
    """

    path = reference.path.resolve()
    payload = _read_exact(path, _sha(reference.sha256, "authorization sha256"), "authorization receipt")
    value = _strict_document(payload, "authorization receipt")
    _exact(
        value,
        {
            "format", "version", "protocol_id", "approval", "review_policy",
            "reviewers", "authorized_candidate", "authorized_validation",
            "authorized_browser_runtime", "authorized_python_runtime",
            "authorized_test_plan", "dependencies", "evaluator",
        },
        "authorization receipt",
    )
    if (
        value["format"] != RECEIPT_FORMAT
        or value["version"] != VERSION
        or value["protocol_id"] != PROTOCOL_ID
    ):
        raise ReviewAuthorizationError("unsupported authorization receipt")
    approval_name, approval_sha = _binding(value["approval"], "receipt.approval")
    approval_payload = _read_exact(path.parent / approval_name, approval_sha, "review approval")
    approval = _strict_document(approval_payload, "review approval")
    _validate_approval(approval)
    verified = _verify_bound_evidence(approval, path.parent)
    parity_value = _object(verified["parity"], "verified parity")
    reproduction_value = _object(
        verified["reproduction"], "verified reproduction"
    )
    reproduction_environment = _object(
        reproduction_value["environment"], "reproduction environment"
    )

    policy = _object(value["review_policy"], "receipt.review_policy")
    _exact(
        policy,
        {"signature_namespace", "allowed_signers", "required_reviewers"},
        "receipt.review_policy",
    )
    if (
        policy["signature_namespace"] != SIGNATURE_NAMESPACE
        or policy["required_reviewers"] != 2
    ):
        raise ReviewAuthorizationError("authorization review policy changed")
    allowed_name, allowed_sha = _binding(
        policy["allowed_signers"], "receipt.review_policy.allowed_signers"
    )
    approval_policy = _object(approval["review_policy"], "approval.review_policy")
    if allowed_sha != approval_policy["allowed_signers_sha256"]:
        raise ReviewAuthorizationError("authorization allowed-signers binding changed")
    allowed_payload = _read_exact(
        path.parent / allowed_name, allowed_sha, "allowed signers"
    )

    reviewers = value["reviewers"]
    if not isinstance(reviewers, list) or len(reviewers) != 2:
        raise ReviewAuthorizationError("authorization requires exactly two reviewers")
    identities: set[str] = set()
    fingerprints: set[str] = set()
    signatures: set[str] = set()
    with tempfile.TemporaryDirectory(
        prefix=".authorization-reload.", dir=path.parent
    ) as raw_pinned:
        pinned = Path(raw_pinned)
        pinned_allowed = pinned / "allowed_signers"
        pinned_allowed.write_bytes(allowed_payload)
        for index, item in enumerate(reviewers):
            reviewer = _object(item, "receipt.reviewer")
            _exact(
                reviewer,
                {"identity", "key_fingerprint", "signature"},
                "receipt.reviewer",
            )
            identity = _string(reviewer["identity"], "receipt.reviewer.identity")
            signature_name, signature_sha = _binding(
                reviewer["signature"], "receipt.reviewer.signature"
            )
            signature_payload = _read_exact(
                path.parent / signature_name, signature_sha, "review signature"
            )
            pinned_signature = pinned / f"review-{index}.sig"
            pinned_signature.write_bytes(signature_payload)
            fingerprint = _verify_signature(
                approval_payload=approval_payload,
                allowed_signers=pinned_allowed,
                identity=identity,
                signature=pinned_signature,
            )
            if fingerprint != reviewer["key_fingerprint"]:
                raise ReviewAuthorizationError(
                    "authorization signing fingerprint changed"
                )
            if (
                identity in identities
                or fingerprint in fingerprints
                or signature_sha in signatures
            ):
                raise ReviewAuthorizationError(
                    "authorization reviewers are not distinct"
                )
            identities.add(identity)
            fingerprints.add(fingerprint)
            signatures.add(signature_sha)

    expected = {
        "authorized_candidate": approval["candidate"],
        "authorized_validation": {
            "report": approval["validation_report"],
            "decision": approval["validation_decision"],
            "reproduction_receipt": approval["reproduction_receipt"],
            "parity_evidence": approval["parity_evidence"],
        },
        "authorized_browser_runtime": {
            "binary_sha256": parity_value["browser_binary_sha256"],
            "version": parity_value["browser_version"],
        },
        "authorized_python_runtime": {
            "executable_sha256": reproduction_environment[
                "python_executable_sha256"
            ],
            "version": reproduction_environment["python_version"],
        },
        "authorized_test_plan": approval["test_plan"],
        "dependencies": approval["dependencies"],
        "evaluator": approval["evaluator"],
    }
    for key, expected_value in expected.items():
        if value[key] != expected_value:
            raise ReviewAuthorizationError(f"authorization {key} changed")
    _verify_test_plan(value["authorized_test_plan"])
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ml.evaluation.review_authorization"
    )
    parser.add_argument("approval", type=Path)
    parser.add_argument("allowed_signers", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--approval-sha256", required=True)
    parser.add_argument("--allowed-signers-sha256", required=True)
    parser.add_argument(
        "--reviewer",
        action="append",
        nargs=2,
        metavar=("IDENTITY", "SIGNATURE"),
        required=True,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    receipt = authorize_review(
        approval_path=arguments.approval,
        approval_sha256=arguments.approval_sha256,
        allowed_signers_path=arguments.allowed_signers,
        allowed_signers_sha256=arguments.allowed_signers_sha256,
        reviewers=tuple(
            (identity, Path(signature))
            for identity, signature in arguments.reviewer
        ),
        output=arguments.output,
    )
    payload = _canonical(receipt)
    print(
        json.dumps(
            {
                "file": arguments.output.name,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "reviewers": [
                    item["identity"] for item in receipt["reviewers"]
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
