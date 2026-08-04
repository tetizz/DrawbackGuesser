from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from ml.evaluation.browser_parity import (
    EVIDENCE_FORMAT,
    INPUT_FORMAT,
    TRANSCRIPT_FORMAT,
    authenticate_runtime_bindings,
    load_authenticated_input,
    publish_evidence,
    verify_transcript_bindings,
    _authenticated_git,
    main as browser_parity_main,
    _publish_browser_outputs,
    _preflight_recursive_git_filters,
    _public_feature_record,
    _reauthenticate_git,
    _reject_executable_git_filters,
    _run_process,
    _sanitized_git_environment,
    _windows_runtime_paths,
    _windows_taskkill,
)
from ml.evaluation.tests.test_release_workflow import (
    _nested_filter_fixture,
    _redirected_worktree_fixture,
)
from ml.evaluation.validation_gate import PROTOCOL_ID, _canonical_pretty
from ml.training.drawback_ml.symbolic_schema import SYMBOLIC_RULE_IDS
from ml.training.drawback_ml.durable_publish import publish_bytes_durable

PUBLIC_GENERATOR_PROTOCOL = {
    "id": "drawbacktrainer-public-pgn-parity-v1",
    "seedDomain": "public-parity-v1",
    "rootSeed": 0x5A17_2026,
    "gameCount": 8,
    "maxPlies": 320,
    "agentSchedule": ["random-legal", "human-like-weak", "greedy-material"],
}


def _authenticated_git_environment() -> dict[str, str]:
    located = shutil.which("git")
    if located is None:
        raise unittest.SkipTest("Git is unavailable")
    git = Path(located).resolve(strict=True)
    return {
        "DRAWBACK_AUTHENTICATED_GIT": str(git),
        "DRAWBACK_AUTHENTICATED_GIT_SHA256": hashlib.sha256(
            git.read_bytes()
        ).hexdigest(),
    }


class BrowserParityTest(unittest.TestCase):
    def test_browser_preflight_rejects_redirected_root_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            git, repository, _redirected, environment = (
                _redirected_worktree_fixture(Path(temporary))
            )
            with self.assertRaisesRegex(ValueError, "different worktree root"):
                _preflight_recursive_git_filters(
                    git,
                    repository=repository,
                    environment=environment,
                )

    def test_browser_preflight_rejects_nested_worktree_filter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            git, repository, _nested, marker, environment = (
                _nested_filter_fixture(Path(temporary))
            )
            with self.assertRaisesRegex(ValueError, "executable Git filter"):
                _preflight_recursive_git_filters(
                    git,
                    repository=repository,
                    environment=environment,
                )
            self.assertFalse(marker.exists())

    def test_browser_nonzero_preserves_child_error_when_cleanup_fails(
        self,
    ) -> None:
        process = type(
            "Process",
            (),
            {
                "returncode": 11,
                "communicate": lambda self, *, timeout: (
                    "output",
                    "browser failed",
                ),
            },
        )()
        with (
            patch(
                "ml.evaluation.browser_parity._popen_contained",
                return_value=(process, None),
            ),
            patch(
                "ml.evaluation.browser_parity._terminate_process_tree",
                side_effect=OSError("cleanup failed"),
            ),
            self.assertRaises(subprocess.CalledProcessError) as raised,
        ):
            _run_process(
                ["child"],
                check=True,
                capture_output=True,
                timeout=10,
            )
        self.assertEqual(raised.exception.returncode, 11)
        self.assertEqual(raised.exception.stderr, "browser failed")
        self.assertIsInstance(raised.exception.__cause__, OSError)
        self.assertIn(
            "cleanup failed",
            " ".join(getattr(raised.exception, "__notes__", ())),
        )

    def test_unchecked_browser_nonzero_preserves_error_when_cleanup_fails(
        self,
    ) -> None:
        process = type(
            "Process",
            (),
            {
                "returncode": 12,
                "communicate": lambda self, *, timeout: (
                    "output",
                    "unchecked browser failed",
                ),
            },
        )()
        with (
            patch(
                "ml.evaluation.browser_parity._popen_contained",
                return_value=(process, None),
            ),
            patch(
                "ml.evaluation.browser_parity._terminate_process_tree",
                side_effect=OSError("cleanup failed"),
            ),
            self.assertRaises(subprocess.CalledProcessError) as raised,
        ):
            _run_process(
                ["child"],
                check=False,
                capture_output=True,
                timeout=10,
            )
        self.assertEqual(raised.exception.returncode, 12)
        self.assertEqual(raised.exception.stderr, "unchecked browser failed")
        self.assertIsInstance(raised.exception.__cause__, OSError)
        self.assertIn(
            "cleanup failed",
            " ".join(getattr(raised.exception, "__notes__", ())),
        )

    def test_browser_git_probe_blocks_local_filter_command(self) -> None:
        located = shutil.which("git")
        if located is None:
            self.skipTest("Git is unavailable")
        git = Path(located).resolve(strict=True)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = root / "repository"
            repository.mkdir()
            marker = root / "filter-ran"
            hook = root / "filter-hook"
            hook.write_text(
                "#!/bin/sh\n"
                'printf x > "$DRAWBACK_TEST_FILTER_MARKER"\n'
                "cat\n",
                encoding="utf-8",
                newline="\n",
            )
            hook.chmod(0o755)
            subprocess.run(
                [str(git), "init"],
                cwd=repository,
                check=True,
                capture_output=True,
            )
            (repository / ".gitattributes").write_text(
                "*.txt filter=marker\n", encoding="utf-8"
            )
            tracked = repository / "tracked.txt"
            tracked.write_text("tracked\n", encoding="utf-8")
            subprocess.run(
                [str(git), "add", ".gitattributes", "tracked.txt"],
                cwd=repository,
                check=True,
            )
            subprocess.run(
                [
                    str(git),
                    "-c",
                    "user.name=test",
                    "-c",
                    "user.email=test@example.invalid",
                    "commit",
                    "-m",
                    "fixture",
                ],
                cwd=repository,
                check=True,
                capture_output=True,
            )
            subprocess.run(
                [
                    str(git),
                    "config",
                    "--local",
                    "filter.marker.clean",
                    f'"{hook.as_posix()}"',
                ],
                cwd=repository,
                check=True,
            )
            tracked.write_text("changed\n", encoding="utf-8")
            environment = dict(os.environ)
            environment["DRAWBACK_TEST_FILTER_MARKER"] = str(marker)
            subprocess.run(
                [str(git), "status", "--porcelain"],
                cwd=repository,
                env=environment,
                check=True,
                capture_output=True,
            )
            self.assertTrue(marker.exists(), "filter fixture did not run")
            marker.unlink()
            sanitized = dict(_sanitized_git_environment())
            sanitized["DRAWBACK_TEST_FILTER_MARKER"] = str(marker)
            with self.assertRaisesRegex(ValueError, "executable Git filter"):
                _reject_executable_git_filters(
                    git,
                    repository=repository,
                    environment=sanitized,
                )
            self.assertFalse(marker.exists())

    @unittest.skipUnless(os.name == "nt", "Windows-only cleanup contract")
    def test_taskkill_nonzero_fails_closed(self) -> None:
        hostile = str(Path.cwd() / "attacker-windows")
        with (
            patch.dict(
                os.environ,
                {"SystemRoot": hostile, "PATH": hostile, "PATHEXT": ".EVIL"},
                clear=False,
            ),
            patch(
                "ml.evaluation.browser_parity._stat_identity",
                return_value=(1, 2, 3, 4, 5),
            ),
            patch(
                "ml.evaluation.browser_parity.subprocess.run",
                return_value=subprocess.CompletedProcess([], 7),
            ) as run_process,
            self.assertRaisesRegex(OSError, "exit code 7"),
        ):
            _windows_taskkill(12345)
        environment = run_process.call_args.kwargs["env"]
        windows, system, command = _windows_runtime_paths()
        self.assertEqual(environment["SystemRoot"], str(windows))
        self.assertEqual(environment["WINDIR"], str(windows))
        self.assertEqual(environment["ComSpec"], str(command))
        self.assertEqual(environment["PATH"], str(system))
        self.assertEqual(environment["PATHEXT"], ".COM;.EXE;.BAT;.CMD")

    def test_browser_timeout_terminates_descendant_process(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
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
                _run_process(
                    [sys.executable, "-c", parent, child, str(heartbeat)],
                    cwd=root,
                    check=True,
                    capture_output=True,
                    timeout=1,
                    environment=dict(os.environ),
                )
            self.assertTrue(heartbeat.exists(), "descendant never started")
            time.sleep(0.2)
            settled_size = heartbeat.stat().st_size
            time.sleep(0.3)
            self.assertEqual(heartbeat.stat().st_size, settled_size)

    def test_browser_fast_parent_success_terminates_grandchild(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            heartbeat = root / "fast-parent-heartbeat"
            child = (
                "from pathlib import Path; import sys,time\n"
                "path=Path(sys.argv[1])\n"
                "while True:\n"
                " path.open('ab', buffering=0).write(b'x')\n"
                " time.sleep(0.02)\n"
            )
            parent = (
                "import os,subprocess,sys,time\n"
                "subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]],"
                "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,"
                "stderr=subprocess.DEVNULL)\n"
                "deadline=time.monotonic()+5\n"
                "while not os.path.exists(sys.argv[2]) and time.monotonic()<deadline:"
                " time.sleep(0.01)\n"
            )
            completed = _run_process(
                [sys.executable, "-c", parent, child, str(heartbeat)],
                cwd=root,
                check=True,
                capture_output=True,
                timeout=10,
                environment=dict(os.environ),
            )
            self.assertEqual(completed.returncode, 0)
            self.assertTrue(heartbeat.exists(), "grandchild never started")
            time.sleep(0.2)
            settled_size = heartbeat.stat().st_size
            time.sleep(0.3)
            self.assertEqual(heartbeat.stat().st_size, settled_size)

    def test_git_binding_ignores_path_and_sanitizes_git_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trusted = root / "trusted-git"
            forged_directory = root / "forged"
            forged_directory.mkdir()
            (forged_directory / "git").write_bytes(b"forged")
            trusted.write_bytes(b"trusted")
            environment = {
                "DRAWBACK_AUTHENTICATED_GIT": str(trusted.resolve()),
                "DRAWBACK_AUTHENTICATED_GIT_SHA256": hashlib.sha256(
                    trusted.read_bytes()
                ).hexdigest(),
                "GIT_DIR": str(root / "attacker-repository"),
                "GIT_CONFIG_GLOBAL": str(root / "attacker-config"),
                "SSH_ASKPASS": str(root / "attacker-askpass"),
                "LD_PRELOAD": str(root / "attacker-preload.so"),
                "LD_LIBRARY_PATH": str(root / "attacker-libraries"),
                "DYLD_INSERT_LIBRARIES": str(root / "attacker-insert.dylib"),
                "DYLD_LIBRARY_PATH": str(root / "attacker-dyld-libraries"),
                "PATH": str(forged_directory),
            }
            with patch.dict(os.environ, environment, clear=False):
                identity = _authenticated_git()
                sanitized = _sanitized_git_environment()
                self.assertEqual(identity.path, trusted.resolve())
                self.assertNotIn("GIT_DIR", sanitized)
                self.assertEqual(sanitized["GIT_CONFIG_GLOBAL"], os.devnull)
                self.assertEqual(sanitized["GIT_ASKPASS"], os.devnull)
                self.assertEqual(sanitized["SSH_ASKPASS"], os.devnull)
                self.assertEqual(sanitized["GCM_INTERACTIVE"], "never")
                self.assertNotEqual(sanitized["PATH"], str(forged_directory))
                self.assertNotIn("LD_PRELOAD", sanitized)
                self.assertNotIn("LD_LIBRARY_PATH", sanitized)
                self.assertNotIn("DYLD_INSERT_LIBRARIES", sanitized)
                self.assertNotIn("DYLD_LIBRARY_PATH", sanitized)
                trusted.write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "digest differs"):
                    _reauthenticate_git(identity)

    def test_browser_output_interruption_retains_exact_partial(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            transcript = root / "transcript.json"
            evidence = root / "evidence.json"
            calls = 0

            def interrupted(path: Path, payload: bytes) -> None:
                nonlocal calls
                calls += 1
                if calls == 1:
                    publish_bytes_durable(path, payload)
                    return
                raise KeyboardInterrupt("injected publication interruption")

            with (
                patch(
                    "ml.evaluation.browser_parity.publish_bytes_durable",
                    side_effect=interrupted,
                ),
                self.assertRaisesRegex(
                    KeyboardInterrupt, "publication interruption"
                ) as raised,
            ):
                _publish_browser_outputs(
                    transcript, b"transcript\n", evidence, b"evidence\n"
                )
            self.assertEqual(transcript.read_bytes(), b"transcript\n")
            self.assertFalse(evidence.exists())
            self.assertEqual(list(root.glob("*.tmp-*")), [])
            self.assertIn(
                "exact partial publication",
                " ".join(getattr(raised.exception, "__notes__", ())),
            )

    def test_browser_output_race_preserves_competing_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            transcript = root / "transcript.json"
            evidence = root / "evidence.json"
            competitor = b"competing evidence\n"
            calls = 0

            def racing(path: Path, payload: bytes) -> None:
                nonlocal calls
                calls += 1
                if calls == 1:
                    publish_bytes_durable(path, payload)
                    return
                path.write_bytes(competitor)
                raise FileExistsError("competing publisher won")

            with (
                patch(
                    "ml.evaluation.browser_parity.publish_bytes_durable",
                    side_effect=racing,
                ),
                self.assertRaisesRegex(ValueError, "bytes do not match"),
            ):
                _publish_browser_outputs(
                    transcript, b"transcript\n", evidence, b"evidence\n"
                )
            self.assertEqual(transcript.read_bytes(), b"transcript\n")
            self.assertEqual(evidence.read_bytes(), competitor)

    def test_browser_output_rollback_retains_pathname_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            transcript = root / "transcript.json"
            evidence = root / "evidence.json"
            replacement = b"replacement transcript\n"
            calls = 0

            def racing(path: Path, payload: bytes) -> None:
                nonlocal calls
                calls += 1
                if calls == 1:
                    publish_bytes_durable(path, payload)
                    return
                transcript.write_bytes(replacement)
                raise RuntimeError("second publication failed")

            with (
                patch(
                    "ml.evaluation.browser_parity.publish_bytes_durable",
                    side_effect=racing,
                ),
                self.assertRaisesRegex(
                    RuntimeError, "second publication failed"
                ) as raised,
            ):
                _publish_browser_outputs(
                    transcript, b"transcript\n", evidence, b"evidence\n"
                )
            self.assertEqual(transcript.read_bytes(), replacement)
            self.assertFalse(evidence.exists())
            self.assertIn(
                "changed after publication",
                " ".join(getattr(raised.exception, "__notes__", ())),
            )

    def test_browser_output_recovers_exact_crash_partial(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            transcript = root / "transcript.json"
            evidence = root / "evidence.json"
            transcript_payload = b"transcript\n"
            evidence_payload = b"evidence\n"
            publish_bytes_durable(transcript, transcript_payload)

            _publish_browser_outputs(
                transcript,
                transcript_payload,
                evidence,
                evidence_payload,
            )
            self.assertEqual(transcript.read_bytes(), transcript_payload)
            self.assertEqual(evidence.read_bytes(), evidence_payload)
            # A completed exact pair is idempotent and remains no-clobber.
            _publish_browser_outputs(
                transcript,
                transcript_payload,
                evidence,
                evidence_payload,
            )

    def test_browser_cli_recovers_an_exact_partial_output_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            transcript = root / "transcript.json"
            evidence = root / "evidence.json"
            transcript_payload = b"transcript\n"
            evidence_payload = b"evidence\n"
            browser_artifact = root / "browser.json"
            calibration = root / "calibration.json"
            browser_artifact.write_bytes(b"browser artifact\n")
            calibration.write_bytes(b"calibration\n")
            publish_bytes_durable(transcript, transcript_payload)
            with (
                patch(
                    "ml.evaluation.browser_parity.load_authenticated_input",
                    return_value={},
                ),
                patch(
                    "ml.evaluation.browser_parity.authenticate_runtime_bindings"
                ),
                patch(
                    "ml.evaluation.browser_parity.run_real_worker",
                    return_value=transcript_payload,
                ),
                patch(
                    "ml.evaluation.browser_parity.verify_transcript_bindings"
                ),
                patch(
                    "ml.evaluation.browser_parity._build_evidence_payload",
                    return_value=({}, evidence_payload),
                ),
            ):
                result = browser_parity_main([
                    "--repository",
                    str(root),
                    "--browser",
                    str(root / "browser.exe"),
                    "--browser-artifact",
                    str(browser_artifact),
                    "--calibration",
                    str(calibration),
                    "--input",
                    str(root / "input.json"),
                    "--input-sha256",
                    "b" * 64,
                    "--transcript-output",
                    str(transcript),
                    "--evidence-output",
                    str(evidence),
                ])
            self.assertEqual(result, 0)
            self.assertEqual(transcript.read_bytes(), transcript_payload)
            self.assertEqual(evidence.read_bytes(), evidence_payload)

    def test_browser_output_rejects_mismatched_crash_partial(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            transcript = root / "transcript.json"
            evidence = root / "evidence.json"
            transcript.write_bytes(b"different transcript\n")

            with self.assertRaisesRegex(ValueError, "bytes do not match"):
                _publish_browser_outputs(
                    transcript,
                    b"transcript\n",
                    evidence,
                    b"evidence\n",
                )
            self.assertEqual(transcript.read_bytes(), b"different transcript\n")
            self.assertFalse(evidence.exists())

    def test_authenticated_input_rejects_hidden_truth(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.json"
            value = {
                "format": INPUT_FORMAT,
                "version": 1,
                "protocolId": PROTOCOL_ID,
                "browserArtifactSha256": "a" * 64,
                "fixtureSha256": "",
                "partition": {
                    "id": "public-validation-parity-v1",
                    "split": "validation-parity",
                    "selectionSha256": "b" * 64,
                    "publicExampleCount": 1,
                },
                "bindings": {
                    "ensembleSha256": "c" * 64,
                    "calibrationSha256": "d" * 64,
                    "fusionSelectionSha256": "9" * 64,
                    "sourceRevision": "e" * 40,
                    "pnpmLockSha256": "f" * 64,
                },
                "publicFixture": {
                    "file": "fixture.json",
                    "sha256": "1" * 64,
                    "generatorProtocol": PUBLIC_GENERATOR_PROTOCOL,
                },
                "cases": [{"id": "case-1", "pgn": "1. e4", "truth": "vegan"}],
            }
            value["fixtureSha256"] = hashlib.sha256(
                _canonical_pretty(
                    {"partition": value["partition"], "cases": value["cases"]}
                )
            ).hexdigest()
            payload = _canonical_pretty(value)
            path.write_bytes(payload)
            with self.assertRaisesRegex(ValueError, "hidden data"):
                load_authenticated_input(
                    path, hashlib.sha256(payload).hexdigest(), "a" * 64
                )

    def test_publishes_exact_review_projection_no_clobber(self) -> None:
        transcript = _canonical_pretty(
            {
                "format": TRANSCRIPT_FORMAT,
                "version": 1,
                "browserArtifactSha256": "a" * 64,
                "fixtureSha256": "b" * 64,
                "workerE2ePassed": True,
                "maximumAbsoluteDifference": 0.0000004,
                "topKIdentical": True,
                "hardZeroSetsIdentical": True,
                "cases": [],
                "browserRuntime": {
                    "binarySha256": "3" * 64,
                    "version": "Browser 1.2.3",
                },
            }
        )
        parity_input = {
            "fixtureSha256": "b" * 64,
            "partition": {"selectionSha256": "d" * 64},
            "bindings": {
                "ensembleSha256": "e" * 64,
                "fusionSelectionSha256": "9" * 64,
                "sourceRevision": "f" * 40,
                "pnpmLockSha256": "1" * 64,
            },
            "publicFixture": {
                "file": "fixture.json",
                "sha256": "1" * 64,
                "generatorProtocol": PUBLIC_GENERATOR_PROTOCOL,
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "parity.json"
            received = publish_evidence(
                transcript_payload=transcript,
                browser_artifact_sha256="a" * 64,
                calibration_sha256="c" * 64,
                parity_input=parity_input,
                parity_input_sha256="2" * 64,
                output=output,
            )
            self.assertEqual(received["format"], EVIDENCE_FORMAT)
            self.assertEqual(received["max_absolute_difference"], 0.0000004)
            with self.assertRaises(FileExistsError):
                publish_evidence(
                    transcript_payload=transcript,
                    browser_artifact_sha256="a" * 64,
                    calibration_sha256="c" * 64,
                    parity_input=parity_input,
                    parity_input_sha256="2" * 64,
                    output=output,
                )

    def test_rejects_order_or_hard_zero_mismatch(self) -> None:
        for field in ("topKIdentical", "hardZeroSetsIdentical"):
            transcript = {
                "format": TRANSCRIPT_FORMAT,
                "version": 1,
                "browserArtifactSha256": "a" * 64,
                "fixtureSha256": "b" * 64,
                "workerE2ePassed": True,
                "maximumAbsoluteDifference": 0.0,
                "topKIdentical": True,
                "hardZeroSetsIdentical": True,
                "cases": [],
                "browserRuntime": {
                    "binarySha256": "3" * 64,
                    "version": "Browser 1.2.3",
                },
            }
            transcript[field] = False
            parity_input = {
                "fixtureSha256": "b" * 64,
                "partition": {"selectionSha256": "d" * 64},
                "bindings": {
                    "ensembleSha256": "e" * 64,
                    "fusionSelectionSha256": "9" * 64,
                    "sourceRevision": "f" * 40,
                    "pnpmLockSha256": "1" * 64,
                },
            }
            with tempfile.TemporaryDirectory() as temporary:
                with self.assertRaisesRegex(ValueError, "not passing"):
                    publish_evidence(
                        transcript_payload=_canonical_pretty(transcript),
                        browser_artifact_sha256="a" * 64,
                        calibration_sha256="c" * 64,
                        parity_input=parity_input,
                        parity_input_sha256="2" * 64,
                        output=Path(temporary) / "out.json",
                    )

    def test_transcript_must_echo_authenticated_bindings(self) -> None:
        parity_input = {
            "protocolId": PROTOCOL_ID,
            "browserArtifactSha256": "a" * 64,
            "fixtureSha256": "b" * 64,
            "partition": {"id": "parity"},
            "bindings": {"sourceRevision": "c" * 40},
        }
        transcript = {
            "protocolId": PROTOCOL_ID,
            "browserArtifactSha256": "a" * 64,
            "fixtureSha256": "b" * 64,
            "partition": {"id": "different"},
            "bindings": {"sourceRevision": "c" * 40},
        }
        with self.assertRaisesRegex(ValueError, "partition binding differs"):
            verify_transcript_bindings(
                _canonical_pretty(transcript), parity_input
            )

    def test_partition_and_cases_are_not_caller_weakenable(self) -> None:
        probabilities = {
            f"rule-{index}": (1.0 if index == 0 else 0.0)
            for index in range(6)
        }
        expected = {
            color: {
                "probabilities": probabilities,
                "topIds": [f"rule-{index}" for index in range(5)],
                "hardZeroIds": [f"rule-{index}" for index in range(1, 6)],
            }
            for color in ("white", "black")
        }
        base = {
            "format": INPUT_FORMAT,
            "version": 1,
            "protocolId": PROTOCOL_ID,
            "browserArtifactSha256": "a" * 64,
            "fixtureSha256": "",
            "partition": {
                "id": "validation-parity-v1",
                "split": "validation-parity",
                "selectionSha256": "b" * 64,
                "publicExampleCount": 1,
            },
            "bindings": {
                "ensembleSha256": "c" * 64,
                "calibrationSha256": "d" * 64,
                "fusionSelectionSha256": "9" * 64,
                "sourceRevision": "e" * 40,
                "pnpmLockSha256": "f" * 64,
            },
            "publicFixture": {
                "file": "fixture.json",
                "sha256": "1" * 64,
                "generatorProtocol": PUBLIC_GENERATOR_PROTOCOL,
            },
            "cases": [
                {
                    "id": "case-1",
                    "pgn": "1. e4",
                    "pgnSha256": hashlib.sha256(b"1. e4").hexdigest(),
                    "expected": expected,
                }
            ],
        }

        def write_and_load(value: dict[str, object]) -> None:
            value["fixtureSha256"] = hashlib.sha256(
                _canonical_pretty(
                    {"partition": value["partition"], "cases": value["cases"]}
                )
            ).hexdigest()
            payload = _canonical_pretty(value)
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "input.json"
                path.write_bytes(payload)
                load_authenticated_input(
                    path, hashlib.sha256(payload).hexdigest(), "a" * 64
                )

        for mutate, message in (
            (
                lambda value: value["partition"].update({"id": ""}),
                "partition identity",
            ),
            (
                lambda value: value["partition"].update(
                    {"selectionSha256": "NOT-A-DIGEST"}
                ),
                "partition identity",
            ),
            (
                lambda value: value["partition"].update(
                    {"publicExampleCount": 2}
                ),
                "partition identity",
            ),
            (
                lambda value: (
                    value["cases"].append(value["cases"][0]),
                    value["partition"].update({"publicExampleCount": 2}),
                ),
                "not unique",
            ),
            (
                lambda value: value["cases"][0]["expected"]["white"].update(
                    {"topIds": ["rule-0"]}
                ),
                "head is invalid",
            ),
            (
                lambda value: value["bindings"].pop(
                    "fusionSelectionSha256"
                ),
                "authentication metadata",
            ),
        ):
            value = json.loads(json.dumps(base))
            mutate(value)
            with self.assertRaisesRegex(ValueError, message):
                write_and_load(value)

    def test_clean_source_rejects_untracked_source_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            (repository / "ml").mkdir()
            (repository / "pnpm-lock.yaml").write_text("lock\n", encoding="utf-8")
            calibration = repository / "calibration.json"
            calibration.write_bytes(b"calibration\n")
            calibration_sha = hashlib.sha256(calibration.read_bytes()).hexdigest()
            artifact = repository / "artifact.json"
            artifact.write_bytes(
                _canonical_pretty(
                    {
                        "ensemble": {
                            "sourceEnsembleReleaseSha256": "a" * 64,
                            "sourceFusionSelectionSha256": "9" * 64,
                        },
                        "calibration": {
                            "sourceCalibrationSha256": calibration_sha
                        },
                    }
                )
            )
            subprocess.run(["git", "init"], cwd=repository, check=True, capture_output=True)
            subprocess.run(
                ["git", "config", "user.name", "test"],
                cwd=repository,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.email", "test@example.invalid"],
                cwd=repository,
                check=True,
            )
            subprocess.run(["git", "add", "."], cwd=repository, check=True)
            subprocess.run(
                ["git", "commit", "-m", "fixture"],
                cwd=repository,
                check=True,
                capture_output=True,
            )
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            marker = repository / "fsmonitor-ran"
            hook = repository / "fsmonitor-hook"
            hook.write_text(
                "#!/bin/sh\n"
                'printf x > "$DRAWBACK_TEST_FSMONITOR_MARKER"\n',
                encoding="utf-8",
                newline="\n",
            )
            hook.chmod(0o755)
            subprocess.run(
                [
                    "git",
                    "config",
                    "--local",
                    "core.fsmonitor",
                    hook.as_posix(),
                ],
                cwd=repository,
                check=True,
            )
            baseline_environment = dict(os.environ)
            baseline_environment["DRAWBACK_TEST_FSMONITOR_MARKER"] = str(
                marker
            )
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repository,
                env=baseline_environment,
                check=True,
                capture_output=True,
            )
            self.assertTrue(marker.exists(), "fsmonitor fixture did not run")
            marker.unlink()
            parity_input = {
                "bindings": {
                    "ensembleSha256": "a" * 64,
                    "calibrationSha256": calibration_sha,
                    "fusionSelectionSha256": "9" * 64,
                    "sourceRevision": revision,
                    "pnpmLockSha256": hashlib.sha256(
                        (repository / "pnpm-lock.yaml").read_bytes()
                    ).hexdigest(),
                }
            }
            (repository / "ml" / "untracked.py").write_text(
                "raise SystemExit\n", encoding="utf-8"
            )
            with (
                patch.dict(
                    os.environ,
                    {
                        **_authenticated_git_environment(),
                        "DRAWBACK_TEST_FSMONITOR_MARKER": str(marker),
                    },
                    clear=False,
                ),
                self.assertRaisesRegex(ValueError, "clean HEAD"),
            ):
                authenticate_runtime_bindings(
                    repository, artifact, calibration, parity_input
                )
            self.assertFalse(marker.exists())

    def test_public_observation_rejects_replay_and_symbolic_mutations(self) -> None:
        count = len(SYMBOLIC_RULE_IDS)
        probability = 1.0 / count
        base = {
            "fenBefore": (
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/"
                "RNBQKBNR w KQkq - 0 1"
            ),
            "move": "e2e4",
            "moveNumber": 1,
            "ply": 0,
            "playerColor": "white",
            "historySan": [],
            "ordinaryLegalMoves": [
                "a2a3", "a2a4", "b1a3", "b1c3", "b2b3", "b2b4",
                "c2c3", "c2c4", "d2d3", "d2d4", "e2e3", "e2e4",
                "f2f3", "f2f4", "g1f3", "g1h3", "g2g3", "g2g4",
                "h2h3", "h2h4",
            ],
            "symbolicFeatureVersion": 6,
            "symbolic": {
                "ruleIds": list(SYMBOLIC_RULE_IDS),
                "whiteProbabilities": [probability] * count,
                "blackProbabilities": [probability] * count,
                "whiteEliminated": [False] * count,
                "blackEliminated": [False] * count,
            },
        }
        arguments = {
            "final_before": base["fenBefore"],
            "final_move": "e2e4",
            "final_history": (),
            "final_legal": tuple(base["ordinaryLegalMoves"]),
            "final_ply": 0,
        }
        self.assertEqual(
            _public_feature_record(base, **arguments).move, "e2e4"
        )
        mutations = []
        for field, replacement in (
            ("fenBefore", "bad-fen"),
            ("historySan", ["e4"]),
            ("ordinaryLegalMoves", ["e2e4"]),
        ):
            changed = json.loads(json.dumps(base))
            changed[field] = replacement
            mutations.append((changed, "disagrees"))
        bad_mask = json.loads(json.dumps(base))
        bad_mask["symbolic"]["whiteEliminated"][0] = True
        mutations.append((bad_mask, "probability is invalid"))
        bad_dimension = json.loads(json.dumps(base))
        bad_dimension["symbolic"]["blackProbabilities"].pop()
        mutations.append((bad_dimension, "dimensions differ"))
        for changed, message in mutations:
            with self.assertRaisesRegex(ValueError, message):
                _public_feature_record(changed, **arguments)


if __name__ == "__main__":
    unittest.main()
