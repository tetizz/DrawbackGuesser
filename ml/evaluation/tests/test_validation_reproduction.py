from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import ml.evaluation.validation_runtime as validation_runtime
from ml.evaluation.ensemble_calibration import ContentAddressedFile
from ml.evaluation.training_frequency import (
    ContentAddressedFile as TrainingFrequencyReference,
)
from ml.evaluation.validation_reproduction import (
    DECISION_FORMAT,
    EnvironmentAttestation,
    FLOAT_TOLERANCE,
    REPORT_FORMAT,
    SOURCE_PATHS,
    _RUNTIME_BOOTSTRAP,
    _canonical_pretty,
    _cleanup_reproduction_outputs,
    _exact_original_evidence,
    _isolated_python_environment,
    _load_runtime_manifest,
    _python_runtime_configuration,
    _run_bounded_process,
    _run_git,
    _scratch_identity,
    _write_atomic_no_clobber,
    attest_clean_environment,
    build_parser,
    compare_validation_evidence,
    load_validation_reproduction_receipt,
    reproduce_validation,
)
from ml.training.drawback_ml.durable_publish import publish_bytes_durable_exact


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


def write_runtime_manifest(command: list[str], target: Path) -> None:
    bootstrap = command.index(_RUNTIME_BOOTSTRAP)
    repository = command[bootstrap + 1]
    import_roots = json.loads(command[bootstrap + 2])
    output = Path(command[bootstrap + 3])
    configured_sys_path = [repository, *import_roots]
    output.write_text(
        json.dumps(
            {
                "format": "drawbacktrainer-python-runtime-manifest",
                "version": 2,
                "python_executable": command[0],
                "isolated": False,
                "ignore_environment": False,
                "no_site": True,
                "no_user_site": True,
                "safe_path": True,
                "dont_write_bytecode": True,
                "hash_randomization": False,
                "python_controls": {"PYTHONHASHSEED": "0"},
                "sitecustomize_loaded": False,
                "usercustomize_loaded": False,
                "configured_sys_path": configured_sys_path,
                "final_sys_path": configured_sys_path,
                "modules": [
                    {"name": "builtins", "kind": "built-in", "files": []},
                    {
                        "name": "ml.evaluation.validation_gate",
                        "kind": "file",
                        "files": [
                            {
                                "role": "origin",
                                "path": str(target.resolve()),
                                "sha256": hashlib.sha256(
                                    target.read_bytes()
                                ).hexdigest(),
                            }
                        ],
                    },
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )


class ValidationReproductionTests(unittest.TestCase):
    def test_cleanup_retains_replacement_and_preserves_primary_error(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / ".reproduction-fixture.report.json"
            output.write_bytes(b"authenticated child output")
            identity = _scratch_identity(output, "reproduced report")
            output.unlink()
            output.write_bytes(b"replacement that belongs to another writer")
            primary = ValueError("fresh validation process failed")

            _cleanup_reproduction_outputs(
                ((output, identity, "reproduced report"),),
                primary,
            )

            self.assertEqual(
                output.read_bytes(),
                b"replacement that belongs to another writer",
            )
            self.assertTrue(
                any(
                    "pathname changed" in note and "retained" in note
                    for note in primary.__notes__
                )
            )

    def test_cleanup_replacement_fails_successful_reproduction_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / ".reproduction-fixture.runtime.json"
            output.write_bytes(b"authenticated child output")
            identity = _scratch_identity(output, "reproduced runtime manifest")
            output.unlink()
            output.write_bytes(b"replacement that must be retained")

            with self.assertRaisesRegex(
                OSError, "pathname changed.*retained"
            ):
                _cleanup_reproduction_outputs(
                    (
                        (
                            output,
                            identity,
                            "reproduced runtime manifest",
                        ),
                    ),
                    None,
                )

            self.assertEqual(
                output.read_bytes(), b"replacement that must be retained"
            )

    def test_publication_retry_accepts_only_exact_committed_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "reproduction.json"
            payload = b'{"reproduced":true}\n'

            def fail_after_publication(
                path: Path,
                value: bytes,
                *,
                label: str,
            ) -> None:
                publish_bytes_durable_exact(path, value, label=label)
                raise OSError("simulated post-publication failure")

            with patch(
                "ml.evaluation.validation_reproduction."
                "publish_bytes_durable_exact",
                side_effect=fail_after_publication,
            ):
                with self.assertRaisesRegex(OSError, "post-publication"):
                    _write_atomic_no_clobber(output, payload)

            _write_atomic_no_clobber(output, payload)
            with self.assertRaisesRegex(ValueError, "overwrite"):
                _write_atomic_no_clobber(output, b"different\n")

    def test_taskkill_nonzero_fails_closed_after_identity_recheck(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            taskkill = root / "taskkill.exe"
            taskkill.write_bytes(b"authenticated taskkill fixture")
            with (
                patch.object(
                    validation_runtime,
                    "_windows_known_directory",
                    return_value=root,
                ),
                patch.object(
                    validation_runtime,
                    "_isolated_python_environment",
                    return_value={},
                ),
                patch.object(
                    validation_runtime.subprocess,
                    "run",
                    return_value=subprocess.CompletedProcess([], 7),
                ),
                self.assertRaisesRegex(OSError, "exit code 7"),
            ):
                validation_runtime._windows_taskkill(12345)

    def test_cleanup_failure_preserves_primary_process_error(self) -> None:
        primary = subprocess.TimeoutExpired(["fixture"], 1)
        cleanup = OSError("tree cleanup failed")
        with self.assertRaises(subprocess.TimeoutExpired) as raised:
            validation_runtime._raise_primary_with_cleanup(primary, cleanup)
        self.assertIs(raised.exception, primary)
        self.assertIs(raised.exception.__cause__, cleanup)
        self.assertTrue(
            any("cleanup also failed" in note for note in primary.__notes__)
        )

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
            git_calls: list[list[str]] = []

            def git_result(
                _repository: Path, arguments: object, **_kwargs: object
            ) -> subprocess.CompletedProcess[str]:
                self.assertIsInstance(arguments, (list, tuple))
                assert isinstance(arguments, (list, tuple))
                git_calls.append([str(item) for item in arguments])
                if arguments == ["--version"]:
                    return subprocess.CompletedProcess(
                        [], 0, "git version 2.test\n", ""
                    )
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
            self.assertRegex(attestation.git_executable_sha256, r"^[0-9a-f]{64}$")
            self.assertEqual(attestation.git_version, "git version 2.test")
            self.assertEqual(
                git_calls[:6],
                [
                    ["--version"],
                    ["rev-parse", "HEAD"],
                    list(validation_runtime._GIT_FILTER_CONFIG_ARGUMENTS),
                    ["status", "--porcelain=v1", "--untracked-files=no"],
                    list(validation_runtime._GIT_FILTER_CONFIG_ARGUMENTS),
                    [
                        "status",
                        "--porcelain=v1",
                        "--untracked-files=all",
                        "--",
                        *SOURCE_PATHS,
                    ],
                ],
            )
            self.assertRegex(
                attestation.python_distributions_sha256, r"^[0-9a-f]{64}$"
            )

    def test_clean_environment_rejects_worktree_filter_before_status(self) -> None:
        located = shutil.which("git")
        if located is None:
            self.skipTest("Git is unavailable")
        git = Path(located).resolve(strict=True)
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            marker = root / "filter-ran"
            filter_hook = root / "filter-hook"
            filter_hook.write_text(
                "#!/bin/sh\n"
                'printf x > "$DRAWBACK_TEST_FILTER_MARKER"\n'
                "cat\n",
                encoding="utf-8",
                newline="\n",
            )
            filter_hook.chmod(0o755)
            subprocess.run([str(git), "init", "-q"], cwd=root, check=True)
            (root / "ml").mkdir()
            files = {
                root / "pnpm-lock.yaml": b"pnpm\n",
                root / "ml/requirements.txt": b"requirements\n",
                root / "ml/pyproject.toml": b"project\n",
            }
            for path, payload in files.items():
                path.write_bytes(payload)
            (root / ".gitattributes").write_text(
                "*.txt filter=marker\n", encoding="utf-8"
            )
            tracked = root / "ml/tracked.txt"
            tracked.write_text("tracked\n", encoding="utf-8")
            subprocess.run([str(git), "add", "."], cwd=root, check=True)
            subprocess.run(
                [
                    str(git),
                    "-c",
                    "user.name=fixture",
                    "-c",
                    "user.email=fixture@example.invalid",
                    "commit",
                    "-q",
                    "-m",
                    "fixture",
                ],
                cwd=root,
                check=True,
            )
            subprocess.run(
                [
                    str(git),
                    "config",
                    "--local",
                    "extensions.worktreeConfig",
                    "true",
                ],
                cwd=root,
                check=True,
            )
            subprocess.run(
                [
                    str(git),
                    "config",
                    "--worktree",
                    "filter.marker.clean",
                    f'"{filter_hook.as_posix()}"',
                ],
                cwd=root,
                check=True,
            )
            tracked.write_text("changed\n", encoding="utf-8")
            revision = subprocess.run(
                [str(git), "rev-parse", "HEAD"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            ).stdout.strip()
            fixture_environment = dict(os.environ)
            fixture_environment["DRAWBACK_TEST_FILTER_MARKER"] = str(marker)
            subprocess.run(
                [str(git), "status", "--porcelain"],
                cwd=root,
                env=fixture_environment,
                check=True,
                capture_output=True,
            )
            self.assertTrue(marker.exists(), "filter fixture did not run")
            marker.unlink()
            with patch.dict(
                os.environ,
                {"DRAWBACK_TEST_FILTER_MARKER": str(marker)},
                clear=False,
            ), self.assertRaisesRegex(ValueError, "executable Git filter"):
                attest_clean_environment(
                    root,
                    expected_source_revision=revision,
                    expected_pnpm_lock_sha256=hashlib.sha256(
                        files[root / "pnpm-lock.yaml"]
                    ).hexdigest(),
                    expected_python_requirements_sha256=hashlib.sha256(
                        files[root / "ml/requirements.txt"]
                    ).hexdigest(),
                    expected_python_project_sha256=hashlib.sha256(
                        files[root / "ml/pyproject.toml"]
                    ).hexdigest(),
                )
            self.assertFalse(marker.exists())

    def test_filter_preflight_rejects_every_executable_driver_key(self) -> None:
        for suffix in ("clean", "smudge", "process"):
            with self.subTest(suffix=suffix), self.assertRaisesRegex(
                ValueError, "executable Git filter"
            ):
                validation_runtime._assert_no_executable_git_filters(
                    f"worktree\tfilter.marker.{suffix}\n"
                )

    def test_fresh_process_orchestration_publishes_bound_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            target = root / "ml/evaluation/validation_gate.py"
            target.parent.mkdir(parents=True)
            target.write_text("# authenticated evaluator fixture\n", encoding="utf-8")
            runtime = _python_runtime_configuration(root)
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
                git_executable_path_sha256="5" * 64,
                git_executable_sha256="6" * 64,
                git_version="git version 2.test",
                python_executable_path_sha256=hashlib.sha256(
                    os.path.normcase(str(runtime.executable.path)).encode("utf-8")
                ).hexdigest(),
                python_executable_sha256=runtime.executable.sha256,
                python_version=sys.version,
                python_import_roots_sha256=runtime.import_roots_sha256,
                python_distributions_sha256=runtime.distributions_sha256,
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
                command: list[str], **kwargs: object
            ) -> subprocess.CompletedProcess[str]:
                self.assertEqual(
                    command[1:6], ["-B", "-s", "-S", "-P", "-c"]
                )
                child_environment = kwargs["env"]
                self.assertIsInstance(child_environment, dict)
                assert isinstance(child_environment, dict)
                self.assertNotIn("PYTHONPATH", child_environment)
                self.assertNotIn("PYTHONHOME", child_environment)
                self.assertNotIn("GIT_DIR", child_environment)
                self.assertNotIn("GIT_WORK_TREE", child_environment)
                self.assertEqual(
                    {
                        key: value
                        for key, value in child_environment.items()
                        if key.upper().startswith("PYTHON")
                    },
                    {"PYTHONHASHSEED": "0"},
                )
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
                write_runtime_manifest(command, target)
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
                value["environment"]["python_runtime"]["algorithm"],
                "sha256-canonical-loaded-python-modules-v1",
            )
            self.assertEqual(value["environment"]["python_runtime"]["file_count"], 1)
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

    def test_git_attestation_ignores_forged_path_and_git_environment(self) -> None:
        poison = {
            "PATH": str(Path.cwd() / "forged-tools"),
            "GIT_DIR": "forged-directory",
            "GIT_WORK_TREE": "forged-worktree",
            "GIT_CONFIG_SYSTEM": "forged-system-config",
            "GIT_CONFIG_GLOBAL": "forged-global-config",
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "core.fsmonitor",
            "GIT_CONFIG_VALUE_0": "forged-hook",
        }
        completed = subprocess.CompletedProcess([], 0, "ok\n", "")
        with patch.dict(os.environ, poison, clear=False), patch(
            "ml.evaluation.validation_reproduction.subprocess.run",
            return_value=completed,
        ) as process:
            self.assertIs(_run_git(Path.cwd(), ["rev-parse", "HEAD"]), completed)
        command = process.call_args.args[0]
        kwargs = process.call_args.kwargs
        self.assertTrue(Path(command[0]).is_absolute())
        self.assertNotEqual(command[0], "git")
        self.assertIn("--no-replace-objects", command)
        self.assertIn("--no-pager", command)
        self.assertIn(f"core.hooksPath={os.devnull}", command)
        self.assertIn(f"core.askPass={os.devnull}", command)
        self.assertIn("credential.helper=", command)
        self.assertNotIn(poison["PATH"], kwargs["env"]["PATH"])
        self.assertEqual(kwargs["env"]["GIT_ASKPASS"], os.devnull)
        self.assertEqual(kwargs["env"]["SSH_ASKPASS"], os.devnull)
        for key in (
            "GIT_DIR",
            "GIT_WORK_TREE",
            "GIT_CONFIG_SYSTEM",
            "GIT_CONFIG_KEY_0",
            "GIT_CONFIG_VALUE_0",
        ):
            self.assertNotIn(key, kwargs["env"])
        self.assertEqual(kwargs["env"]["GIT_CONFIG_COUNT"], "0")
        self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)

    def test_isolated_python_environment_blocks_sitecustomize_and_overrides(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            poison = root / "poison"
            poison.mkdir()
            marker = root / "sitecustomize-ran"
            user_marker = root / "usercustomize-ran"
            (poison / "sitecustomize.py").write_text(
                "from pathlib import Path\n"
                f"Path({str(marker)!r}).write_text('ran', encoding='utf-8')\n",
                encoding="utf-8",
            )
            (poison / "usercustomize.py").write_text(
                "from pathlib import Path\n"
                f"Path({str(user_marker)!r}).write_text('ran', encoding='utf-8')\n",
                encoding="utf-8",
            )
            hostile = {
                "PYTHONPATH": str(poison),
                "PYTHONHOME": str(poison),
                "PYTHONUSERBASE": str(poison),
                "GIT_DIR": "forged-directory",
            }
            with patch.dict(os.environ, hostile, clear=False):
                environment = _isolated_python_environment()
            for key in hostile:
                self.assertNotIn(key, environment)
            self.assertEqual(
                {
                    key: value
                    for key, value in environment.items()
                    if key.upper().startswith("PYTHON")
                },
                {"PYTHONHASHSEED": "0"},
            )
            probe = (
                "import json,os,sys\n"
                "print(json.dumps({"
                "'hash':hash('drawbacktrainer-closed-python-v1'),"
                "'python_controls':{name:value for name,value in "
                "os.environ.items() if name.upper().startswith('PYTHON')},"
                "'flags':{"
                "'isolated':sys.flags.isolated,"
                "'ignore_environment':sys.flags.ignore_environment,"
                "'no_site':sys.flags.no_site,"
                "'no_user_site':sys.flags.no_user_site,"
                "'safe_path':sys.flags.safe_path,"
                "'dont_write_bytecode':sys.flags.dont_write_bytecode,"
                "'hash_randomization':sys.flags.hash_randomization},"
                "'customizations':[name for name in "
                "('sitecustomize','usercustomize') if name in sys.modules],"
                "'sys_path':sys.path},sort_keys=True))"
            )
            outputs: list[str] = []
            for _ in range(2):
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-B",
                        "-s",
                        "-S",
                        "-P",
                        "-c",
                        probe,
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    env=environment,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                outputs.append(completed.stdout)
            self.assertEqual(outputs[0], outputs[1])
            observed = json.loads(outputs[0])
            self.assertEqual(
                observed["python_controls"], {"PYTHONHASHSEED": "0"}
            )
            self.assertEqual(
                observed["flags"],
                {
                    "isolated": 0,
                    "ignore_environment": 0,
                    "no_site": 1,
                    "no_user_site": 1,
                    "safe_path": True,
                    "dont_write_bytecode": 1,
                    "hash_randomization": 0,
                },
            )
            self.assertEqual(observed["customizations"], [])
            self.assertNotIn(str(poison.resolve()), observed["sys_path"])
            self.assertFalse(marker.exists())
            self.assertFalse(user_marker.exists())
            self.assertEqual(list(root.rglob("__pycache__")), [])

    def test_runtime_manifest_rejects_a_module_outside_authenticated_roots(
        self,
    ) -> None:
        with (
            tempfile.TemporaryDirectory() as raw,
            tempfile.TemporaryDirectory() as other,
        ):
            root = Path(raw)
            target = root / "ml/evaluation/validation_gate.py"
            target.parent.mkdir(parents=True)
            target.write_text("# evaluator\n", encoding="utf-8")
            outsider = Path(other) / "shadow.py"
            outsider.write_text("# shadow\n", encoding="utf-8")
            runtime = _python_runtime_configuration(root)
            manifest = root / "runtime.json"
            command = [
                str(runtime.executable.path),
                "-B",
                "-s",
                "-S",
                "-P",
                "-c",
                _RUNTIME_BOOTSTRAP,
                str(root.resolve()),
                json.dumps([str(item.path) for item in runtime.import_roots]),
                str(manifest.resolve()),
            ]
            write_runtime_manifest(command, target)
            value = json.loads(manifest.read_text("utf-8"))
            value["modules"].append(
                {
                    "name": "shadow",
                    "kind": "file",
                    "files": [
                        {
                            "role": "origin",
                            "path": str(outsider.resolve()),
                            "sha256": hashlib.sha256(
                                outsider.read_bytes()
                            ).hexdigest(),
                        }
                    ],
                }
            )
            manifest.write_text(
                json.dumps(value, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "outside authenticated roots"):
                _load_runtime_manifest(
                    manifest,
                    repository=root,
                    runtime=runtime,
                )

    def test_bounded_process_timeout_terminates_descendant_tree(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            heartbeat = root / "heartbeat"
            child = (
                "from pathlib import Path; import sys,time\n"
                "path=Path(sys.argv[1])\n"
                "while True:\n"
                " path.open('ab', buffering=0).write(b'x')\n"
                " time.sleep(0.02)\n"
            )
            parent = (
                "import subprocess,sys,time\n"
                "subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]])\n"
                "while True: time.sleep(1)\n"
            )
            with self.assertRaises(subprocess.TimeoutExpired):
                _run_bounded_process(
                    [sys.executable, "-c", parent, child, str(heartbeat)],
                    cwd=root,
                    env=dict(os.environ),
                    check=False,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    timeout_seconds=0.75,
                )
            self.assertTrue(heartbeat.exists(), "descendant never started")
            time.sleep(0.2)
            settled_size = heartbeat.stat().st_size
            time.sleep(0.3)
            self.assertEqual(heartbeat.stat().st_size, settled_size)

    def test_cli_has_no_test_partition_or_test_path(self) -> None:
        destinations = {action.dest for action in build_parser()._actions}
        self.assertNotIn("partition", destinations)
        self.assertNotIn("test", destinations)
        self.assertNotIn("private_test", destinations)
        self.assertIn("private_validation", destinations)


if __name__ == "__main__":
    unittest.main()
