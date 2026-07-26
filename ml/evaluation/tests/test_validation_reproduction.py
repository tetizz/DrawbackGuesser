from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ml.evaluation.ensemble_calibration import ContentAddressedFile
from ml.evaluation.training_frequency import (
    ContentAddressedFile as TrainingFrequencyReference,
)
from ml.evaluation.validation_reproduction import (
    DECISION_FORMAT,
    EnvironmentAttestation,
    FLOAT_TOLERANCE,
    REPORT_FORMAT,
    _canonical_pretty,
    _exact_original_evidence,
    attest_clean_environment,
    build_parser,
    compare_validation_evidence,
    load_validation_reproduction_receipt,
    reproduce_validation,
)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def report_value(metric: float = 0.25) -> dict[str, object]:
    return {
        "format": REPORT_FORMAT,
        "version": 1,
        "protocol": {
            "id": "current-catalog-182-v2",
            "validation_partition_identity": "current-catalog-182-v2",
            "bootstrap_seed": 20260814,
            "bootstrap_replicates": 10_000,
        },
        "bindings": {
            "ensemble_release_sha256": "a" * 64,
            "calibration_sha256": "b" * 64,
            "training_frequency_sha256": "c" * 64,
        },
        "promotion": {
            "partition": "validation-gate",
            "partition_seed_sha256": "d" * 64,
            "ensemble_release_sha256": "a" * 64,
            "calibration_sha256": "b" * 64,
            "training_frequency_sha256": "c" * 64,
            "transcript": {
                "algorithm": "sha256-domain-separated-canonical-json-v1",
                "sha256": "e" * 64,
                "record_count": 3,
            },
            "views": {"prepared-182": {"top_3_accuracy": metric}},
        },
    }


def decision_value(
    report_file: str, report_sha256: str, metric: float = 0.25
) -> dict[str, object]:
    return {
        "format": DECISION_FORMAT,
        "version": 1,
        "protocol_id": "current-catalog-182-v2",
        "validation_report": {
            "file": report_file,
            "sha256": report_sha256,
        },
        "passed": True,
        "missing_count": 0,
        "failed_count": 0,
        "threshold_contract_sha256": "f" * 64,
        "bootstrap": {"interval": metric},
        "results": [
            {
                "gate_id": "gate.one",
                "status": "passed",
                "actual": metric,
                "requirement": "frozen",
                "reason": None,
            }
        ],
    }


def write_document(root: Path, name: str, value: object) -> ContentAddressedFile:
    payload = _canonical_pretty(value)
    path = root / name
    path.write_bytes(payload)
    return ContentAddressedFile(path, hashlib.sha256(payload).hexdigest())


class ValidationReproductionTests(unittest.TestCase):
    def test_metric_comparison_allows_one_micro_unit_but_not_more(self) -> None:
        original_report = report_value(0.25)
        original_decision = decision_value("original.json", "a" * 64, 0.25)
        reproduced_report = report_value(0.25 + FLOAT_TOLERANCE)
        reproduced_decision = decision_value(
            "reproduced.json", "b" * 64, 0.25 + FLOAT_TOLERANCE
        )
        comparison = compare_validation_evidence(
            original_report,
            original_decision,
            reproduced_report,
            reproduced_decision,
        )
        self.assertGreater(comparison.float_count, 0)
        self.assertAlmostEqual(
            comparison.maximum_absolute_float_difference, FLOAT_TOLERANCE
        )
        reproduced_report["promotion"]["views"]["prepared-182"][  # type: ignore[index]
            "top_3_accuracy"
        ] = 0.25 + FLOAT_TOLERANCE * 1.1
        with self.assertRaisesRegex(ValueError, "differs by more"):
            compare_validation_evidence(
                original_report,
                original_decision,
                reproduced_report,
                reproduced_decision,
            )

    def test_comparison_requires_exact_transcript_and_nonfloat_identity(self) -> None:
        original_report = report_value()
        original_decision = decision_value("one.json", "a" * 64)
        changed = report_value()
        changed["promotion"]["transcript"]["sha256"] = "9" * 64  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "transcript"):
            compare_validation_evidence(
                original_report,
                original_decision,
                changed,
                decision_value("two.json", "b" * 64),
            )
        changed = report_value()
        changed["promotion"]["partition"] = "test"  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "partition"):
            compare_validation_evidence(
                original_report,
                original_decision,
                changed,
                decision_value("two.json", "b" * 64),
            )

    def test_original_evidence_must_be_passing_and_bind_exact_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            report_ref = write_document(
                root, "report.json", report_value()
            )
            decision = decision_value(
                report_ref.path.name, report_ref.sha256
            )
            _exact_original_evidence(
                report_value(),
                decision,
                report_reference=report_ref,
                ensemble_sha256="a" * 64,
                calibration_sha256="b" * 64,
                training_frequency_sha256="c" * 64,
            )
            decision["passed"] = False
            with self.assertRaisesRegex(ValueError, "not a passing"):
                _exact_original_evidence(
                    report_value(),
                    decision,
                    report_reference=report_ref,
                    ensemble_sha256="a" * 64,
                    calibration_sha256="b" * 64,
                    training_frequency_sha256="c" * 64,
                )

    def test_clean_environment_binds_head_and_all_dependency_locks(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "ml").mkdir()
            files = {
                root / "pnpm-lock.yaml": b"pnpm",
                root / "ml/requirements.txt": b"requirements",
                root / "ml/pyproject.toml": b"project",
            }
            for path, payload in files.items():
                path.write_bytes(payload)
            revision = "1" * 40

            def git_result(
                _repository: Path, arguments: object
            ) -> subprocess.CompletedProcess[str]:
                if arguments == ["rev-parse", "HEAD"]:
                    return subprocess.CompletedProcess([], 0, revision + "\n", "")
                return subprocess.CompletedProcess([], 0, "", "")

            with patch(
                "ml.evaluation.validation_reproduction._run_git",
                side_effect=git_result,
            ):
                attestation = attest_clean_environment(
                    root,
                    expected_source_revision=revision,
                    expected_pnpm_lock_sha256=hashlib.sha256(b"pnpm").hexdigest(),
                    expected_python_requirements_sha256=hashlib.sha256(
                        b"requirements"
                    ).hexdigest(),
                    expected_python_project_sha256=hashlib.sha256(
                        b"project"
                    ).hexdigest(),
                )
            self.assertEqual(attestation.source_revision, revision)
            self.assertRegex(attestation.python_executable_sha256, r"^[0-9a-f]{64}$")

    def test_fresh_process_orchestration_publishes_bound_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            report = report_value()
            report_ref = write_document(root, "report.json", report)
            decision = decision_value(report_ref.path.name, report_ref.sha256)
            decision_ref = write_document(root, "decision.json", decision)
            for name in (
                "ensemble.json",
                "calibration.json",
                "frequency.json",
                "validation.ndjson",
                "public.json",
                "private.json",
                "catalog.json",
            ):
                (root / name).write_text("{}\n", encoding="utf-8")
            environment = EnvironmentAttestation(
                source_revision="1" * 40,
                pnpm_lock_sha256="2" * 64,
                python_requirements_sha256="3" * 64,
                python_project_sha256="4" * 64,
                python_executable_sha256="5" * 64,
                python_version="3.11-test",
            )
            audited = SimpleNamespace(
                engine_binary_sha256="6" * 64,
                engine_fingerprint=f"stockfish:18:{'6' * 64}:{'7' * 64}",
                evaluator_policy_id="stockfish-bestmove-v1",
                evaluator_policy_version=1,
                evaluator_nodes=10_000,
                release_root_sha256="8" * 64,
                corpus_run_id="9" * 64,
                manifest_sha256="a" * 64,
                dataset_sha256="b" * 64,
            )
            lease = SimpleNamespace(
                audited=audited,
                verify_dataset_unchanged=lambda: None,
            )

            class LeaseContext:
                def __enter__(self) -> object:
                    return lease

                def __exit__(self, *_args: object) -> None:
                    return None

            def runner(
                command: list[str], **_kwargs: object
            ) -> subprocess.CompletedProcess[str]:
                report_output = Path(
                    command[command.index("--report-output") + 1]
                )
                decision_output = Path(
                    command[command.index("--decision-output") + 1]
                )
                report_output.write_bytes(_canonical_pretty(report))
                reproduced_decision = decision_value(
                    report_output.name,
                    hashlib.sha256(report_output.read_bytes()).hexdigest(),
                )
                decision_output.write_bytes(
                    _canonical_pretty(reproduced_decision)
                )
                return subprocess.CompletedProcess(command, 0, "ok", "")

            receipt_path = root / "receipt.json"
            with patch(
                "ml.evaluation.validation_reproduction."
                "open_audited_private_corpus_split",
                return_value=LeaseContext(),
            ), patch(
                "ml.evaluation.validation_reproduction."
                "attest_clean_environment",
                return_value=environment,
            ):
                receipt = reproduce_validation(
                    repository=root,
                    original_report=report_ref,
                    original_decision=decision_ref,
                    ensemble=ContentAddressedFile(
                        root / "ensemble.json", "a" * 64
                    ),
                    calibration=ContentAddressedFile(
                        root / "calibration.json", "b" * 64
                    ),
                    training_frequency=TrainingFrequencyReference(
                        root / "frequency.json", "c" * 64
                    ),
                    public_root=root / "public.json",
                    private_validation=root / "private.json",
                    validation_dataset=root / "validation.ndjson",
                    catalogs=(root / "catalog.json",),
                    receipt_output=receipt_path,
                    environment=environment,
                    expected_engine_binary_sha256="6" * 64,
                    expected_engine_fingerprint=audited.engine_fingerprint,
                    batch_size=16,
                    process_runner=runner,
                )
            value = json.loads(receipt_path.read_bytes())
            self.assertEqual(value["original"]["report"]["sha256"], report_ref.sha256)
            self.assertTrue(value["comparison"]["exact_transcript_sha256"])
            self.assertEqual(
                hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
                receipt.sha256,
            )
            self.assertEqual(
                load_validation_reproduction_receipt(receipt)["format"],
                "drawbacktrainer-validation-reproduction-receipt",
            )
            self.assertEqual(
                list(root.glob(".reproduction-*.json")),
                [],
            )

    def test_cli_has_no_test_partition_or_test_path(self) -> None:
        destinations = {action.dest for action in build_parser()._actions}
        self.assertNotIn("partition", destinations)
        self.assertNotIn("test", destinations)
        self.assertNotIn("private_test", destinations)
        self.assertIn("private_validation", destinations)


if __name__ == "__main__":
    unittest.main()
