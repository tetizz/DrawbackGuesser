from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from ml.evaluation.review_authorization import (
    FORMAT,
    PARITY_FORMAT,
    PROTOCOL_ID,
    RECEIPT_FORMAT,
    REPRODUCTION_FORMAT,
    SIGNATURE_NAMESPACE,
    ReviewAuthorizationError,
    authorize_review,
    main,
)


def canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, separators=(",", ": "))
        + "\n"
    ).encode("utf-8")


def validation_canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")


def write(root: Path, name: str, payload: bytes) -> dict[str, str]:
    (root / name).write_bytes(payload)
    return {"file": name, "sha256": hashlib.sha256(payload).hexdigest()}


def make_key(root: Path, name: str) -> tuple[Path, str]:
    key = root / name
    subprocess.run(
        ("ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key)),
        check=True,
    )
    public = (root / f"{name}.pub").read_text(encoding="utf-8").strip()
    return key, public


def sign(root: Path, key: Path, approval: Path, name: str, namespace: str = SIGNATURE_NAMESPACE) -> Path:
    message = root / f"{name}.approval"
    message.write_bytes(approval.read_bytes())
    subprocess.run(
        (
            "ssh-keygen",
            "-Y",
            "sign",
            "-q",
            "-f",
            str(key),
            "-n",
            namespace,
            str(message),
        ),
        check=True,
    )
    return root / f"{name}.approval.sig"


def fixture(root: Path) -> tuple[Path, Path, list[tuple[str, Path]], dict[str, object]]:
    alice_key, alice_public = make_key(root, "alice")
    bob_key, bob_public = make_key(root, "bob")
    allowed_payload = (
        f"alice {alice_public}\n"
        f"bob {bob_public}\n"
    ).encode("utf-8")
    allowed = root / "allowed_signers"
    allowed.write_bytes(allowed_payload)
    allowed_sha = hashlib.sha256(allowed_payload).hexdigest()

    protocol = write(root, "protocol.md", b"frozen protocol\n")
    ensemble = write(root, "ensemble.json", b"ensemble\n")
    calibration = write(root, "calibration.json", b"calibration\n")
    browser = write(root, "browser.json", b"browser\n")
    frequency = write(root, "frequency.json", b"frequency\n")
    validation_report = write(
        root,
        "validation-report.json",
        validation_canonical(
            {
                "format": "drawbacktrainer-validation-gate-report",
                "version": 1,
                "protocol": {"id": PROTOCOL_ID},
                "bindings": {
                    "ensemble_release_sha256": ensemble["sha256"],
                    "calibration_sha256": calibration["sha256"],
                    "training_frequency_sha256": frequency["sha256"],
                },
                "promotion": {
                    "partition": "validation-gate",
                    "partition_seed_sha256": "10" * 32,
                    "transcript": {"sha256": "11" * 32},
                },
            }
        ),
    )
    gate_results = [
        {
            "gate_id": "all-required-evidence",
            "status": "passed",
            "actual": True,
            "requirement": "all required evidence exists",
            "reason": None,
        }
    ]
    threshold_contract = [
        {
            "gate_id": item["gate_id"],
            "requirement": item["requirement"],
        }
        for item in gate_results
    ]
    decision_value = {
        "format": "drawbacktrainer-validation-gate-decision",
        "version": 1,
        "protocol_id": PROTOCOL_ID,
        "validation_report": validation_report,
        "passed": True,
        "missing_count": 0,
        "failed_count": 0,
        "threshold_contract_sha256": hashlib.sha256(
            json.dumps(
                threshold_contract, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest(),
        "bootstrap": {},
        "results": gate_results,
    }
    decision = write(
        root, "validation-decision.json", validation_canonical(decision_value)
    )
    parity = write(
        root,
        "parity.json",
        canonical(
            {
                "format": PARITY_FORMAT,
                "version": 1,
                "protocol_id": PROTOCOL_ID,
                "browser_artifact_sha256": browser["sha256"],
                "calibration_sha256": calibration["sha256"],
                "passed": True,
                "max_absolute_difference": 0.0000003,
                "top_k_identical": True,
                "hard_zero_sets_identical": True,
                "worker_e2e_passed": True,
                "parity_input_sha256": "61" * 32,
                "transcript_sha256": "62" * 32,
                "fixture_sha256": "63" * 32,
                "partition_selection_sha256": "64" * 32,
                "ensemble_sha256": ensemble["sha256"],
                "source_revision": "55" * 20,
                "pnpm_lock_sha256": hashlib.sha256(
                    b"lockfileVersion: '9.0'\n"
                ).hexdigest(),
                "browser_binary_sha256": "65" * 32,
                "browser_version": "Chromium fixture",
                "public_fixture_sha256": "66" * 32,
            }
        ),
    )
    pnpm = write(root, "pnpm-lock.yaml", b"lockfileVersion: '9.0'\n")
    requirements = write(root, "requirements.txt", b"torch==2.4.0\n")
    python_project = write(root, "pyproject.toml", b"[project]\nname='fixture'\n")
    engine = write(root, "stockfish.exe", b"stockfish-binary")
    options = write(root, "stockfish-options.json", canonical({"Threads": 1}))
    catalog = write(root, "catalog.json", canonical({"rules": []}))
    source_revision = "55" * 20
    evaluator_fingerprint = (
        f"stockfish:18:{engine['sha256']}:{options['sha256']}"
    )
    reproduction = write(
        root,
        "reproduction.json",
        validation_canonical(
            {
                "format": REPRODUCTION_FORMAT,
                "version": 1,
                "protocol_id": PROTOCOL_ID,
                "original": {
                    "report": validation_report,
                    "decision": decision,
                },
                "candidate_inputs": {
                    "ensemble_release_sha256": ensemble["sha256"],
                    "calibration_sha256": calibration["sha256"],
                    "training_frequency_sha256": frequency["sha256"],
                    "catalogs": [catalog],
                },
                "validation_corpus": {
                    "release_root_sha256": "12" * 32,
                    "corpus_run_id": "13" * 32,
                    "private_validation_manifest_sha256": "14" * 32,
                    "validation_dataset_sha256": "15" * 32,
                    "partition_seed_sha256": "10" * 32,
                },
                "environment": {
                    "source_revision": source_revision,
                    "pnpm_lock_sha256": pnpm["sha256"],
                    "python_requirements_sha256": requirements["sha256"],
                    "python_project_sha256": python_project["sha256"],
                    "python_executable_sha256": hashlib.sha256(
                        Path(sys.executable).read_bytes()
                    ).hexdigest(),
                    "python_version": sys.version,
                },
                "evaluator": {
                    "engine_binary_sha256": engine["sha256"],
                    "engine_fingerprint": evaluator_fingerprint,
                    "policy_id": "stockfish-bestmove-v1",
                    "policy_version": 1,
                    "nodes": 10_000,
                },
                "fresh_process": {
                    "command": ["python", "-m", "ml.evaluation.validation_gate"],
                    "command_sha256": "17" * 32,
                    "report_sha256": validation_report["sha256"],
                    "decision_sha256": decision["sha256"],
                    "transcript_sha256": "11" * 32,
                },
                "comparison": {
                    "float_tolerance": 1e-6,
                    "float_count": 1,
                    "maximum_absolute_float_difference": 0.0000004,
                    "exact_candidate_and_input_hashes": True,
                    "exact_transcript_sha256": True,
                },
            }
        ),
    )

    approval_value: dict[str, object] = {
        "format": FORMAT,
        "version": 1,
        "protocol": {"id": PROTOCOL_ID, "document": protocol},
        "candidate": {
            "ensemble_release": ensemble,
            "calibration": calibration,
            "browser_artifact": browser,
            "training_frequency": frequency,
        },
        "validation_report": validation_report,
        "validation_decision": decision,
        "reproduction_receipt": reproduction,
        "parity_evidence": parity,
        "test_plan": {
            "plan_id": "sealed-test-one-shot-v1",
            "bootstrap_seed": 20260831,
            "report_schema": {
                "format": "drawbacktrainer-sealed-test-report",
                "version": 1,
            },
            "argv": [
                "python",
                "-m",
                "ml.evaluation.sealed_test",
                "release.public.json",
                "test.private.json",
                "test.ndjson",
                "--report-output",
                "sealed-report.json",
                "--decision-output",
                "sealed-decision.json",
            ],
            "inputs": {
                "public_root": {"file": "release.public.json", "sha256": "22" * 32},
                "private_test": {"file": "test.private.json", "sha256": "33" * 32},
                "dataset": {"file": "test.ndjson", "sha256": "44" * 32},
            },
            "output_basenames": {
                "report": "sealed-report.json",
                "decision": "sealed-decision.json",
            },
        },
        "dependencies": {
            "source_revision": source_revision,
            "pnpm_lock": pnpm,
            "python_requirements": requirements,
            "python_project": python_project,
        },
        "evaluator": {
            "binary": engine,
            "options": options,
            "fingerprint": evaluator_fingerprint,
            "policy_id": "stockfish-bestmove-v1",
            "policy_version": 1,
            "nodes": 10_000,
        },
        "review_policy": {
            "signature_namespace": SIGNATURE_NAMESPACE,
            "allowed_signers_sha256": allowed_sha,
            "required_reviewers": 2,
        },
    }
    approval = root / "approval.json"
    approval.write_bytes(canonical(approval_value))
    reviewers = [
        ("alice", sign(root, alice_key, approval, "alice")),
        ("bob", sign(root, bob_key, approval, "bob")),
    ]
    return approval, allowed, reviewers, approval_value


class ReviewAuthorizationTests(unittest.TestCase):
    def test_rejects_authoritative_reproduction_dependency_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            approval, allowed, _reviewers, value = fixture(root)
            reproduction_path = root / "reproduction.json"
            reproduction = json.loads(reproduction_path.read_text("utf-8"))
            self.assertEqual(
                reproduction["format"],
                "drawbacktrainer-validation-reproduction-receipt",
            )
            reproduction["environment"]["source_revision"] = "99" * 20
            reproduction_payload = validation_canonical(reproduction)
            reproduction_path.write_bytes(reproduction_payload)
            value["reproduction_receipt"]["sha256"] = hashlib.sha256(
                reproduction_payload
            ).hexdigest()
            approval.write_bytes(canonical(value))
            with self.assertRaisesRegex(
                ReviewAuthorizationError, "dependency identities"
            ):
                authorize_review(
                    approval_path=approval,
                    approval_sha256=hashlib.sha256(
                        approval.read_bytes()
                    ).hexdigest(),
                    allowed_signers_path=allowed,
                    allowed_signers_sha256=hashlib.sha256(
                        allowed.read_bytes()
                    ).hexdigest(),
                    reviewers=(),
                    output=root / "authorization.json",
                )

    def test_authorizes_two_distinct_signatures_without_opening_sealed_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            approval, allowed, reviewers, value = fixture(root)
            forbidden = {
                item["file"]
                for item in value["test_plan"]["inputs"].values()
            }
            original = Path.read_bytes

            def guarded(path: Path) -> bytes:
                if path.name in forbidden:
                    raise AssertionError("sealed input was opened")
                return original(path)

            with patch.object(Path, "read_bytes", guarded):
                receipt = authorize_review(
                    approval_path=approval,
                    approval_sha256=hashlib.sha256(approval.read_bytes()).hexdigest(),
                    allowed_signers_path=allowed,
                    allowed_signers_sha256=hashlib.sha256(allowed.read_bytes()).hexdigest(),
                    reviewers=reviewers,
                    output=root / "authorization.json",
                )
            self.assertEqual(receipt["format"], RECEIPT_FORMAT)
            self.assertEqual(
                [item["identity"] for item in receipt["reviewers"]],
                ["alice", "bob"],
            )
            self.assertEqual(
                len({item["key_fingerprint"] for item in receipt["reviewers"]}), 2
            )
            self.assertEqual(
                receipt["authorized_test_plan"]["inputs"],
                value["test_plan"]["inputs"],
            )
            self.assertEqual(
                receipt["authorized_browser_runtime"],
                {
                    "binary_sha256": "65" * 32,
                    "version": "Chromium fixture",
                },
            )
            self.assertEqual(
                receipt["authorized_python_runtime"],
                {
                    "executable_sha256": hashlib.sha256(
                        Path(sys.executable).read_bytes()
                    ).hexdigest(),
                    "version": sys.version,
                },
            )

    def test_fails_closed_on_mutation_duplicate_reviewer_and_no_clobber(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            approval, allowed, reviewers, _value = fixture(root)
            approval_sha = hashlib.sha256(approval.read_bytes()).hexdigest()
            allowed_sha = hashlib.sha256(allowed.read_bytes()).hexdigest()
            output = root / "authorization.json"
            authorize_review(
                approval_path=approval,
                approval_sha256=approval_sha,
                allowed_signers_path=allowed,
                allowed_signers_sha256=allowed_sha,
                reviewers=reviewers,
                output=output,
            )
            with self.assertRaisesRegex(ReviewAuthorizationError, "overwrite"):
                authorize_review(
                    approval_path=approval,
                    approval_sha256=approval_sha,
                    allowed_signers_path=allowed,
                    allowed_signers_sha256=allowed_sha,
                    reviewers=reviewers,
                    output=output,
                )
            with self.assertRaisesRegex(ReviewAuthorizationError, "distinct"):
                authorize_review(
                    approval_path=approval,
                    approval_sha256=approval_sha,
                    allowed_signers_path=allowed,
                    allowed_signers_sha256=allowed_sha,
                    reviewers=(reviewers[0], reviewers[0]),
                    output=root / "duplicate.json",
                )
            (root / "ensemble.json").write_bytes(b"mutated\n")
            with self.assertRaisesRegex(ReviewAuthorizationError, "SHA-256"):
                authorize_review(
                    approval_path=approval,
                    approval_sha256=approval_sha,
                    allowed_signers_path=allowed,
                    allowed_signers_sha256=allowed_sha,
                    reviewers=reviewers,
                    output=root / "mutated.json",
                )

    def test_rejects_noncanonical_path_traversal_and_failing_decision(self) -> None:
        for label, mutate, expected in (
            (
                "traversal",
                lambda value: value["test_plan"]["inputs"]["dataset"].__setitem__(
                    "file", "../test.ndjson"
                ),
                "safe basename",
            ),
            (
                "duplicate-input",
                lambda value: value["test_plan"]["inputs"]["dataset"].__setitem__(
                    "file",
                    value["test_plan"]["inputs"]["public_root"]["file"],
                ),
                "input basenames must be distinct",
            ),
            (
                "failing",
                lambda value: value["validation_decision"].__setitem__(
                    "file", "failing-decision.json"
                ),
                "passing",
            ),
        ):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                approval, allowed, _reviewers, value = fixture(root)
                if label == "failing":
                    decision = json.loads(
                        (root / "validation-decision.json").read_text("utf-8")
                    )
                    decision["passed"] = False
                    payload = validation_canonical(decision)
                    (root / "failing-decision.json").write_bytes(payload)
                    value["validation_decision"]["sha256"] = hashlib.sha256(
                        payload
                    ).hexdigest()
                mutate(value)
                approval.write_bytes(canonical(value))
                with self.assertRaisesRegex(ReviewAuthorizationError, expected):
                    authorize_review(
                        approval_path=approval,
                        approval_sha256=hashlib.sha256(
                            approval.read_bytes()
                        ).hexdigest(),
                        allowed_signers_path=allowed,
                        allowed_signers_sha256=hashlib.sha256(
                            allowed.read_bytes()
                        ).hexdigest(),
                        reviewers=(),
                        output=root / "authorization.json",
                    )

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            approval, allowed, _reviewers, value = fixture(root)
            approval.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ReviewAuthorizationError, "canonical"):
                authorize_review(
                    approval_path=approval,
                    approval_sha256=hashlib.sha256(approval.read_bytes()).hexdigest(),
                    allowed_signers_path=allowed,
                    allowed_signers_sha256=hashlib.sha256(allowed.read_bytes()).hexdigest(),
                    reviewers=(),
                    output=root / "authorization.json",
                )

    def test_rejects_wrong_signature_namespace_and_cli_publishes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            approval, allowed, reviewers, _value = fixture(root)
            wrong = sign(root, root / "alice", approval, "wrong", "wrong-namespace")
            with self.assertRaisesRegex(ReviewAuthorizationError, "verification failed"):
                authorize_review(
                    approval_path=approval,
                    approval_sha256=hashlib.sha256(approval.read_bytes()).hexdigest(),
                    allowed_signers_path=allowed,
                    allowed_signers_sha256=hashlib.sha256(allowed.read_bytes()).hexdigest(),
                    reviewers=(("alice", wrong), reviewers[1]),
                    output=root / "wrong.json",
                )
            result = main(
                [
                    str(approval),
                    str(allowed),
                    str(root / "authorization.json"),
                    "--approval-sha256",
                    hashlib.sha256(approval.read_bytes()).hexdigest(),
                    "--allowed-signers-sha256",
                    hashlib.sha256(allowed.read_bytes()).hexdigest(),
                    "--reviewer",
                    reviewers[0][0],
                    str(reviewers[0][1]),
                    "--reviewer",
                    reviewers[1][0],
                    str(reviewers[1][1]),
                ]
            )
            self.assertEqual(result, 0)
            self.assertTrue((root / "authorization.json").is_file())

    def test_rejects_parity_release_binding_mutations(self) -> None:
        for field, replacement in (
            ("ensemble_sha256", "71" * 32),
            ("source_revision", "72" * 20),
            ("pnpm_lock_sha256", "73" * 32),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                approval, allowed, _reviewers, value = fixture(root)
                parity_path = root / "parity.json"
                parity = json.loads(parity_path.read_text(encoding="utf-8"))
                parity[field] = replacement
                parity_payload = canonical(parity)
                parity_path.write_bytes(parity_payload)
                value["parity_evidence"]["sha256"] = hashlib.sha256(
                    parity_payload
                ).hexdigest()
                approval.write_bytes(canonical(value))
                with self.assertRaisesRegex(
                    ReviewAuthorizationError, "different release evidence"
                ):
                    authorize_review(
                        approval_path=approval,
                        approval_sha256=hashlib.sha256(
                            approval.read_bytes()
                        ).hexdigest(),
                        allowed_signers_path=allowed,
                        allowed_signers_sha256=hashlib.sha256(
                            allowed.read_bytes()
                        ).hexdigest(),
                        reviewers=(),
                        output=root / "authorization.json",
                    )


if __name__ == "__main__":
    unittest.main()
