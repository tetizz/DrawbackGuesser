from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from ml.evaluation.release_workflow import (
    FORMAT,
    ReleaseWorkflowError,
    _SelectionWave,
    _assert_no_executable_git_filters,
    build_plan,
    _confined,
    _closed_environment,
    _hardened_git_command,
    _preflight_recursive_git_filters,
    _reject_executable_git_filters,
    _run_capture_process,
    _run_step_process,
    _rollback_new_selection_outputs,
    _windows_runtime_paths,
    _windows_taskkill,
    load_workflow,
    run,
)


def _ref(path: str, digest: str = "c" * 64) -> dict[str, str]:
    return {"path": path, "sha256": digest}


def _external_path(name: str) -> str:
    return (
        f"C:/release-tools/{name}"
        if os.name == "nt"
        else f"/opt/release-tools/{name}"
    )


def _workflow() -> dict[str, object]:
    candidates = []
    for seed in (20260811, 20260812, 20260813):
        candidates.append({
            "seed": seed,
            "trainingRun": _ref(f"release/{seed}/run.json"),
            "epochs": [
                {
                    "epoch": epoch,
                    "checkpoint": _ref(
                        f"release/{seed}/epoch-{epoch}.pt"
                    ),
                    "report": f"release/{seed}/epoch-{epoch}.selection.json",
                    "summary": f"release/{seed}/epoch-{epoch}.summary.json",
                }
                for epoch in range(1, 9)
            ],
        })
    return {
        "format": FORMAT,
        "version": 3,
        "sourceRevision": "a" * 40,
        "realDomainExecution": "external-isolated-only",
        "tools": {
            "browser": {
                "path": _external_path("browser"),
                "sha256": "b" * 64,
            },
            "git": {
                "path": _external_path("git"),
                "sha256": "d" * 64,
            },
            "node": {
                "path": _external_path("node"),
                "sha256": "f" * 64,
            },
            "pnpm": {
                "path": _external_path("pnpm"),
                "sha256": "e" * 64,
            },
        },
        "shared": {
            "dataset": _ref("release/validation.ndjson"),
            "publicRoot": _ref("release/public.json"),
            "privateValidation": _ref("release/private-validation.json"),
        },
        "candidates": candidates,
        "selectionOutputs": {
            "20260811": "release/20260811/selected.json",
            "20260812": "release/20260812/selected.json",
            "20260813": "release/20260813/selected.json",
        },
        "ensemble": {"output": "release/ensemble.json"},
        "calibration": {
            "report": "release/calibration-report.json",
            "sidecar": "release/calibration-sidecar.ndjson",
            "receipt": "release/calibration-receipt.json",
            "output": "release/calibration.json",
        },
        "trainingFrequency": _ref("release/training-frequency.json"),
        "validation": {
            "report": "release/validation-report.json",
            "decision": "release/validation-decision.json",
        },
        "browserRelease": {
            "fixture": _ref("release/public-parity-fixture.json"),
            "artifact": "release/browser-model.json",
            "parityInput": "release/parity-input.json",
            "transcript": "release/parity-transcript.json",
            "evidence": "release/parity-evidence.json",
        },
        "transcriptOutput": "release/workflow-transcript.json",
    }


def _write(path: Path, value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode() + b"\n"
    path.write_bytes(payload)
    import hashlib
    return hashlib.sha256(payload).hexdigest()


def _nested_filter_fixture(
    root: Path,
) -> tuple[Path, Path, Path, Path, dict[str, str]]:
    located = shutil.which("git")
    if located is None:
        raise unittest.SkipTest("Git is unavailable")
    git = Path(located).resolve(strict=True)

    def git_run(
        repository: Path,
        *arguments: str,
        check: bool = True,
        environment: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(git), *arguments],
            cwd=repository,
            check=check,
            capture_output=True,
            text=True,
            env=environment,
        )

    def initialize(repository: Path) -> None:
        repository.mkdir()
        git_run(repository, "init")
        (repository / "seed.txt").write_text("seed\n", encoding="utf-8")
        git_run(repository, "add", "seed.txt")
        git_run(
            repository,
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-m",
            "fixture",
        )

    nested_source = root / "nested-source"
    initialize(nested_source)
    (nested_source / ".gitattributes").write_text(
        "*.txt filter=nested-marker\n", encoding="utf-8"
    )
    (nested_source / "tracked.txt").write_text("base\n", encoding="utf-8")
    git_run(nested_source, "add", ".gitattributes", "tracked.txt")
    git_run(
        nested_source,
        "-c",
        "user.name=test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "tracked fixture",
    )

    child_source = root / "child-source"
    initialize(child_source)
    git_run(
        child_source,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(nested_source),
        "deps/nested",
    )
    git_run(
        child_source,
        "-c",
        "user.name=test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-am",
        "nested submodule",
    )

    repository = root / "repository"
    initialize(repository)
    git_run(
        repository,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(child_source),
        "deps/child",
    )
    git_run(
        repository,
        "-c",
        "user.name=test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-am",
        "child submodule",
    )
    git_run(
        repository,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "update",
        "--init",
        "--recursive",
    )

    nested = repository / "deps" / "child" / "deps" / "nested"
    marker = root / "nested-filter-ran"
    hook = root / "nested-filter-hook"
    hook.write_text(
        "#!/bin/sh\n"
        'printf x > "$DRAWBACK_TEST_FILTER_MARKER"\n'
        "cat\n",
        encoding="utf-8",
        newline="\n",
    )
    hook.chmod(0o755)
    git_run(
        nested,
        "config",
        "--local",
        "filter.nested-marker.clean",
        f'"{hook.as_posix()}"',
    )
    (nested / "tracked.txt").write_text("changed\n", encoding="utf-8")
    baseline = dict(os.environ)
    baseline["DRAWBACK_TEST_FILTER_MARKER"] = str(marker)
    git_run(nested, "status", "--porcelain", environment=baseline)
    if not marker.exists():
        raise AssertionError("nested filter fixture did not execute")
    marker.unlink()
    environment = dict(_closed_environment(repository, {"git": git}))
    environment["DRAWBACK_TEST_FILTER_MARKER"] = str(marker)
    return git, repository, nested, marker, environment


def _redirected_worktree_fixture(
    root: Path,
) -> tuple[Path, Path, Path, dict[str, str]]:
    located = shutil.which("git")
    if located is None:
        raise unittest.SkipTest("Git is unavailable")
    git = Path(located).resolve(strict=True)
    repository = root / "repository"
    redirected = root / "redirected-worktree"
    repository.mkdir()
    redirected.mkdir()

    def git_run(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(git), *arguments],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )

    git_run("init")
    tracked = repository / "tracked.txt"
    tracked.write_bytes(b"committed\n")
    git_run("add", "tracked.txt")
    git_run(
        "-c",
        "user.name=test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "fixture",
    )
    committed = subprocess.run(
        [str(git), "show", "HEAD:tracked.txt"],
        cwd=repository,
        check=True,
        capture_output=True,
    ).stdout
    (redirected / "tracked.txt").write_bytes(committed)
    git_run("config", "--local", "core.worktree", str(redirected))
    tracked.write_bytes(b"dirty source\n")
    environment = dict(_closed_environment(repository, {"git": git}))
    return git, repository, redirected, environment


class ReleaseWorkflowTests(unittest.TestCase):
    def test_recursive_filter_preflight_rejects_redirected_root_worktree(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            git, repository, redirected, environment = (
                _redirected_worktree_fixture(Path(directory))
            )
            actual_top_level = _run_capture_process(
                _hardened_git_command(git, "rev-parse", "--show-toplevel"),
                cwd=repository,
                environment=environment,
                timeout=30,
            ).stdout.strip()
            self.assertEqual(Path(actual_top_level).resolve(), redirected.resolve())
            with self.assertRaisesRegex(
                ReleaseWorkflowError, "different worktree root"
            ):
                _preflight_recursive_git_filters(
                    git,
                    repository=repository,
                    environment=environment,
                )

    def test_recursive_filter_preflight_rejects_nested_worktree_filter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            git, repository, _nested, marker, environment = (
                _nested_filter_fixture(Path(directory))
            )
            with self.assertRaisesRegex(
                ReleaseWorkflowError, "executable Git filter"
            ):
                _preflight_recursive_git_filters(
                    git,
                    repository=repository,
                    environment=environment,
                )
            self.assertFalse(marker.exists())

    def test_selection_rollback_retains_pathname_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "selection.json"
            replacement = root / "replacement.json"
            output.write_bytes(b"owned\n")
            replacement.write_bytes(b"replacement\n")
            original_lstat = Path.lstat
            swapped = False

            def swap_before_inspection(path: Path):
                nonlocal swapped
                if path == output and not swapped:
                    swapped = True
                    replacement.replace(output)
                return original_lstat(path)

            with (
                patch.object(
                    Path,
                    "lstat",
                    autospec=True,
                    side_effect=swap_before_inspection,
                ),
                patch.object(Path, "unlink", autospec=True) as unlink,
            ):
                retained = _rollback_new_selection_outputs((output,))
            unlink.assert_not_called()
            self.assertEqual(output.read_bytes(), b"replacement\n")
            self.assertIn("pathname race", " ".join(retained))

    def test_nonzero_runners_preserve_child_error_when_cleanup_fails(
        self,
    ) -> None:
        step_process = SimpleNamespace(
            returncode=7,
            wait=lambda *, timeout: 7,
        )
        with (
            patch(
                "ml.evaluation.release_workflow._popen_contained",
                return_value=(step_process, None),
            ),
            patch(
                "ml.evaluation.release_workflow._terminate_process_tree",
                side_effect=OSError("cleanup failed"),
            ),
            self.assertRaises(subprocess.CalledProcessError) as raised,
        ):
            _run_step_process(
                ["child"],
                cwd=Path.cwd(),
                environment={},
                cancel=None,
            )
        self.assertEqual(raised.exception.returncode, 7)
        self.assertIsInstance(raised.exception.__cause__, OSError)
        self.assertIn(
            "cleanup failed",
            " ".join(getattr(raised.exception, "__notes__", ())),
        )

        capture_process = SimpleNamespace(
            returncode=9,
            communicate=lambda *, timeout: ("output", "child failed"),
        )
        with (
            patch(
                "ml.evaluation.release_workflow._popen_contained",
                return_value=(capture_process, None),
            ),
            patch(
                "ml.evaluation.release_workflow._terminate_process_tree",
                side_effect=OSError("capture cleanup failed"),
            ),
            self.assertRaises(subprocess.CalledProcessError) as raised,
        ):
            _run_capture_process(
                ["child"],
                cwd=Path.cwd(),
                environment={},
                timeout=10,
            )
        self.assertEqual(raised.exception.returncode, 9)
        self.assertEqual(raised.exception.stderr, "child failed")
        self.assertIsInstance(raised.exception.__cause__, OSError)
        self.assertIn(
            "capture cleanup failed",
            " ".join(getattr(raised.exception, "__notes__", ())),
        )

    @unittest.skipUnless(os.name == "nt", "Windows-only environment contract")
    def test_closed_environment_ignores_hostile_windows_variables(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hostile = str(root / "attacker-windows")
            with patch.dict(
                os.environ,
                {
                    "SystemRoot": hostile,
                    "WINDIR": hostile,
                    "ComSpec": str(root / "attacker.cmd"),
                    "PATH": hostile,
                    "PATHEXT": ".EVIL",
                    "TEMP": hostile,
                    "TMP": hostile,
                },
                clear=False,
            ):
                environment = _closed_environment(
                    root, {"git": Path(sys.executable).resolve()}
                )
            windows, system, command = _windows_runtime_paths()
            self.assertEqual(environment["SystemRoot"], str(windows))
            self.assertEqual(environment["WINDIR"], str(windows))
            self.assertEqual(environment["ComSpec"], str(command))
            self.assertIn(str(system), environment["PATH"].split(os.pathsep))
            self.assertNotIn(hostile, environment["PATH"])
            self.assertEqual(environment["PATHEXT"], ".COM;.EXE;.BAT;.CMD")
            self.assertEqual(environment["TEMP"], str(root))
            self.assertEqual(environment["TMP"], str(root))

    def test_hardened_git_status_blocks_local_fsmonitor_command(self) -> None:
        located = shutil.which("git")
        if located is None:
            self.skipTest("Git is unavailable")
        git = Path(located).resolve(strict=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "repository"
            repository.mkdir()
            marker = root / "fsmonitor-ran"
            hook = root / "fsmonitor-hook"
            hook.write_text(
                "#!/bin/sh\n"
                'printf x > "$DRAWBACK_TEST_FSMONITOR_MARKER"\n',
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
            (repository / "tracked.txt").write_text(
                "tracked\n", encoding="utf-8"
            )
            (repository / ".gitattributes").write_text(
                "*.txt filter=marker\n", encoding="utf-8"
            )
            subprocess.run(
                [str(git), "add", "tracked.txt", ".gitattributes"],
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
                [str(git), "status", "--porcelain"],
                cwd=repository,
                env=baseline_environment,
                check=True,
                capture_output=True,
            )
            self.assertTrue(marker.exists(), "fsmonitor fixture did not run")
            marker.unlink()

            environment = dict(_closed_environment(repository, {"git": git}))
            environment["DRAWBACK_TEST_FSMONITOR_MARKER"] = str(marker)
            self.assertEqual(environment["GIT_CONFIG_NOSYSTEM"], "1")
            self.assertEqual(environment["GIT_CONFIG_GLOBAL"], os.devnull)
            self.assertEqual(environment["GIT_ASKPASS"], os.devnull)
            self.assertEqual(environment["SSH_ASKPASS"], os.devnull)
            self.assertEqual(environment["GCM_INTERACTIVE"], "never")
            completed = _run_capture_process(
                _hardened_git_command(
                    git, "status", "--porcelain", "--untracked-files=all"
                ),
                cwd=repository,
                environment=environment,
                timeout=30,
            )
            self.assertEqual(completed.returncode, 0)
            self.assertFalse(marker.exists())
            self.assertIn("core.fsmonitor=false", completed.args)
            self.assertIn(f"core.hooksPath={os.devnull}", completed.args)
            self.assertIn(f"core.askPass={os.devnull}", completed.args)
            self.assertIn("credential.helper=", completed.args)

            excludes = root / "caller-excludes"
            excludes.write_text("hidden.txt\n", encoding="utf-8")
            (repository / "hidden.txt").write_text(
                "hidden\n", encoding="utf-8"
            )
            subprocess.run(
                [
                    str(git),
                    "config",
                    "--local",
                    "core.excludesFile",
                    excludes.as_posix(),
                ],
                cwd=repository,
                check=True,
            )
            unsafe_status = subprocess.run(
                [str(git), "status", "--porcelain", "--untracked-files=all"],
                cwd=repository,
                env=baseline_environment,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            self.assertNotIn("hidden.txt", unsafe_status)
            marker.unlink(missing_ok=True)
            hardened_status = _run_capture_process(
                _hardened_git_command(
                    git, "status", "--porcelain", "--untracked-files=all"
                ),
                cwd=repository,
                environment=environment,
                timeout=30,
            ).stdout
            self.assertIn("hidden.txt", hardened_status)
            self.assertFalse(marker.exists())

            filter_marker = root / "filter-ran"
            filter_hook = root / "filter-hook"
            filter_hook.write_text(
                "#!/bin/sh\n"
                'printf x > "$DRAWBACK_TEST_FILTER_MARKER"\n'
                "cat\n",
                encoding="utf-8",
                newline="\n",
            )
            filter_hook.chmod(0o755)
            subprocess.run(
                [
                    str(git),
                    "config",
                    "--local",
                    "extensions.worktreeConfig",
                    "true",
                ],
                cwd=repository,
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
                cwd=repository,
                check=True,
            )
            (repository / "tracked.txt").write_text(
                "changed\n", encoding="utf-8"
            )
            baseline_environment["DRAWBACK_TEST_FILTER_MARKER"] = str(
                filter_marker
            )
            subprocess.run(
                [str(git), "status", "--porcelain"],
                cwd=repository,
                env=baseline_environment,
                check=True,
                capture_output=True,
            )
            self.assertTrue(filter_marker.exists(), "filter fixture did not run")
            filter_marker.unlink()
            marker.unlink(missing_ok=True)
            environment["DRAWBACK_TEST_FILTER_MARKER"] = str(filter_marker)
            with self.assertRaisesRegex(
                ReleaseWorkflowError, "executable Git filter"
            ):
                _reject_executable_git_filters(
                    git,
                    repository=repository,
                    environment=environment,
                )
            self.assertFalse(filter_marker.exists())
            self.assertFalse(marker.exists())
            with self.assertRaisesRegex(
                ReleaseWorkflowError, "executable Git filter"
            ):
                _assert_no_executable_git_filters(
                    "worktree\tfilter.marker.process\n"
                )
            _assert_no_executable_git_filters(
                "command\tfilter.marker.process\n"
                "global\tfilter.marker.clean\n"
            )

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
                "ml.evaluation.release_workflow._stable_file_identity",
                return_value=(1, 2, 3, 4, 5),
            ),
            patch(
                "ml.evaluation.release_workflow.subprocess.run",
                return_value=SimpleNamespace(returncode=5),
            ) as run_process,
            self.assertRaisesRegex(OSError, "exit code 5"),
        ):
            _windows_taskkill(12345)
        environment = run_process.call_args.kwargs["env"]
        windows, system, command = _windows_runtime_paths()
        self.assertEqual(environment["SystemRoot"], str(windows))
        self.assertEqual(environment["WINDIR"], str(windows))
        self.assertEqual(environment["ComSpec"], str(command))
        self.assertEqual(environment["PATH"], str(system))
        self.assertEqual(environment["PATHEXT"], ".COM;.EXE;.BAT;.CMD")

    def test_builds_only_approved_typed_commands(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "workflow.json"
            digest = _write(path, _workflow())
            workflow, loaded_digest = load_workflow(path)
            self.assertEqual(loaded_digest, digest)
            plan = build_plan(workflow, root)
            self.assertEqual(len(plan), 59)
            evaluations = [
                step for step in plan if step.stage == "selection-fit-evaluation"
            ]
            self.assertEqual(len(evaluations), 24)
            self.assertEqual(
                {(step.seed, step.epoch) for step in evaluations},
                {
                    (seed, epoch)
                    for seed in (20260811, 20260812, 20260813)
                    for epoch in range(1, 9)
                },
            )
            for step in evaluations:
                self.assertIn("--validation-partition", step.argv)
                index = step.argv.index("--validation-partition")
                self.assertEqual(step.argv[index + 1], "selection")
                self.assertEqual(step.argv[0], sys.executable)
                self.assertEqual(
                    step.argv[1:6], ("-B", "-s", "-S", "-P", "-c")
                )
                self.assertIn("ml.evaluation.cli", step.argv)
                self.assertEqual(len(step.inputs), 4)
                self.assertEqual(step.generated_inputs, ())
            expected_generated_counts = {
                "selection-summary": 1,
                "epoch-selection": 8,
                "ensemble-release": 3,
                "fusion-selection": 1,
                "calibration-evaluation": 2,
                "calibration-fit": 4,
                "validation-gate": 2,
                "browser-artifact": 2,
                "browser-parity-input": 3,
                "browser-parity": 3,
            }
            for step in plan:
                if step.stage in expected_generated_counts:
                    self.assertEqual(
                        len(step.generated_inputs),
                        expected_generated_counts[step.stage],
                    )
            stages = [step.stage for step in plan]
            ensemble_index = stages.index("ensemble-release")
            self.assertEqual(
                stages[ensemble_index : ensemble_index + 3],
                [
                    "ensemble-release",
                    "fusion-selection",
                    "calibration-evaluation",
                ],
            )
            fusion = plan[ensemble_index + 1]
            module_index = fusion.argv.index("ml.evaluation.cli")
            self.assertEqual(
                fusion.argv[module_index + 1], "select-ensemble-fusion"
            )
            self.assertNotIn("--validation-partition", fusion.argv)
            self.assertEqual(
                fusion.outputs,
                (Path("release/fusion-selection.json"),),
            )
            self.assertEqual(
                fusion.generated_inputs,
                (Path("release/ensemble.json"),),
            )
            calibration = plan[ensemble_index + 2]
            self.assertEqual(
                calibration.argv[
                    calibration.argv.index("--fusion-selection") + 1
                ],
                str(Path("release/fusion-selection.json")),
            )
            self.assertEqual(
                calibration.argv[
                    calibration.argv.index(
                        "--fusion-selection-sha256"
                    )
                    + 1
                ],
                "<sha256:release/fusion-selection.json>",
            )
            transcript = run(workflow, digest, root, execute=False)
            self.assertIs(transcript["evidence"], False)
            self.assertEqual(transcript["workflowSha256"], digest)
            self.assertEqual(
                transcript["realDomainExecution"], "external-isolated-only"
            )
            self.assertFalse(any(
                "sealed_test" in " ".join(record["argv"])
                for record in transcript["steps"]
            ))

    def test_rejects_path_seed_epoch_and_schema_bypasses(self) -> None:
        mutations = {
            "absolute": lambda value: value["shared"].__setitem__(
                "dataset", _ref("C:/secret/test.ndjson")
            ),
            "traversal": lambda value: value["shared"].__setitem__(
                "dataset", _ref("../secret/test.ndjson")
            ),
            "backslash": lambda value: value["shared"].__setitem__(
                "dataset", _ref("release\\validation.ndjson")
            ),
            "reserved-device": lambda value: value["shared"].__setitem__(
                "dataset", _ref("release/NUL")
            ),
            "trailing-dot": lambda value: value[
                "selectionOutputs"
            ].__setitem__("20260811", "release/selected.json."),
            "duplicate-seed": lambda value: value["candidates"][1].__setitem__(
                "seed", 20260811
            ),
            "reordered-epoch": lambda value: value["candidates"][0]["epochs"][0].__setitem__(
                "epoch", 2
            ),
            "duplicate-path": lambda value: value["candidates"][0]["epochs"][1].__setitem__(
                "checkpoint", _ref("release/20260811/epoch-1.pt")
            ),
            "output-overwrites-input": lambda value: value[
                "selectionOutputs"
            ].__setitem__(
                "20260811", "release/validation.ndjson"
            ),
            "case-alias-overwrites-input": lambda value: value[
                "selectionOutputs"
            ].__setitem__(
                "20260811", "RELEASE/VALIDATION.NDJSON"
            ),
            "invalid-input-sha": lambda value: value["shared"].__setitem__(
                "dataset", _ref("release/validation.ndjson", "NOT-A-SHA")
            ),
            "real-domain-run": lambda value: value.__setitem__(
                "realDomainExecution", "run"
            ),
            "unknown-field": lambda value: value.__setitem__(
                "command", ["cmd.exe", "/c", "anything"]
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                value = _workflow()
                mutate(value)
                path = Path(directory) / "workflow.json"
                _write(path, value)
                with self.assertRaises(ReleaseWorkflowError):
                    workflow, _ = load_workflow(path)
                    build_plan(workflow, Path(directory))

    def test_rejects_noncanonical_manifest_and_symlink_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "workflow.json"
            path.write_text(json.dumps(_workflow(), indent=2), encoding="utf-8")
            with self.assertRaisesRegex(ReleaseWorkflowError, "canonical"):
                load_workflow(path)
            _write(path, _workflow())
            workflow, _ = load_workflow(path)
            with (
                patch.object(Path, "exists", return_value=True),
                patch.object(Path, "is_symlink", return_value=True),
            ):
                with self.assertRaisesRegex(ReleaseWorkflowError, "symlink"):
                    _confined(root, Path("release/input.json"), must_exist=False)

    def test_execute_preflights_all_outputs_before_spawning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "release").mkdir()
            workflow_path = root / "workflow.json"
            digest = _write(workflow_path, _workflow())
            workflow, _ = load_workflow(workflow_path)
            late_output = root / "release" / "parity-evidence.json"
            late_output.write_text("already exists", encoding="utf-8")
            with patch(
                "ml.evaluation.release_workflow._run_capture_process"
            ) as process:
                with self.assertRaisesRegex(
                    ReleaseWorkflowError, "outputs already exist"
                ):
                    run(workflow, digest, root, execute=True)
                process.assert_not_called()

    def test_executes_selection_evaluations_four_at_a_time_in_plan_order(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "release").mkdir()
            workflow_path = root / "workflow.json"
            digest = _write(workflow_path, _workflow())
            workflow, _ = load_workflow(workflow_path)
            lock = threading.Lock()
            active = 0
            maximum = 0

            def execute_step(step, *_args):
                nonlocal active, maximum
                if step.stage == "selection-fit-evaluation":
                    with lock:
                        active += 1
                        maximum = max(maximum, active)
                    time.sleep(0.02)
                    with lock:
                        active -= 1
                return (
                    step.argv,
                    {
                        path: "2" * 64
                        for path in step.generated_inputs
                    },
                    [
                        {
                            "path": path.as_posix(),
                            "sha256": "1" * 64,
                        }
                        for path in step.outputs
                    ],
                )

            external = {
                name: Path(value["path"]).resolve()
                for name, value in workflow["tools"].items()
            }

            def process(args, **_kwargs):
                if "rev-parse" in args:
                    if "--show-toplevel" in args:
                        return SimpleNamespace(stdout=str(root) + "\n")
                    return SimpleNamespace(
                        stdout=workflow["sourceRevision"] + "\n"
                    )
                return SimpleNamespace(stdout="")

            with (
                patch(
                    "ml.evaluation.release_workflow._authenticate_external",
                    side_effect=lambda reference, _label: reference.path.resolve(),
                ),
                patch(
                    "ml.evaluation.release_workflow._closed_environment",
                    return_value={"PATH": "closed"},
                ),
                patch(
                    "ml.evaluation.release_workflow.shutil.which",
                    side_effect=lambda name, **_kwargs: str(external[name]),
                ),
                patch(
                    "ml.evaluation.release_workflow._input_references",
                    return_value=(),
                ),
                patch(
                    "ml.evaluation.release_workflow._run_capture_process",
                    side_effect=process,
                ),
                patch(
                    "ml.evaluation.release_workflow._execute_step",
                    side_effect=execute_step,
                ),
            ):
                with patch(
                    "ml.evaluation.release_workflow._stable_file_identity",
                    return_value=(1, 2, 3, 4, 5),
                ):
                    transcript = run(
                        workflow,
                        digest,
                        root,
                        execute=True,
                    )

            self.assertEqual(maximum, 4)
            selection = [
                (record["seed"], record["epoch"])
                for record in transcript["steps"]
                if record["stage"] == "selection-fit-evaluation"
            ]
            self.assertEqual(
                selection,
                [
                    (seed, epoch)
                    for seed in (20260811, 20260812, 20260813)
                    for epoch in range(1, 9)
                ],
            )

    def test_selection_failure_stops_queued_evaluations_promptly(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "release").mkdir()
            workflow_path = root / "workflow.json"
            digest = _write(workflow_path, _workflow())
            workflow, _ = load_workflow(workflow_path)
            called: list[tuple[int | None, int | None]] = []
            lock = threading.Lock()
            first_finished = threading.Event()

            def execute_step(step, *_args):
                with lock:
                    called.append((step.seed, step.epoch))
                if (
                    step.stage == "selection-fit-evaluation"
                    and step.seed == 20260811
                    and step.epoch == 1
                ):
                    for output in step.outputs:
                        path = root / output
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text("finished", encoding="utf-8")
                    first_finished.set()
                if (
                    step.stage == "selection-fit-evaluation"
                    and step.seed == 20260811
                    and step.epoch == 4
                ):
                    if not first_finished.wait(timeout=1):
                        raise AssertionError("first worker did not finish")
                    raise RuntimeError("later-plan failure")
                time.sleep(0.05)
                return (
                    step.argv,
                    {},
                    [
                        {
                            "path": path.as_posix(),
                            "sha256": "1" * 64,
                        }
                        for path in step.outputs
                    ],
                )

            def cancellation_first(futures):
                pending = tuple(futures)
                while not all(future.done() for future in pending):
                    time.sleep(0.001)
                return iter(sorted(
                    pending,
                    key=lambda future: not isinstance(
                        future.exception(), ReleaseWorkflowError
                    ),
                ))

            def rollback_with_report(paths):
                return (
                    *_rollback_new_selection_outputs(paths),
                    "injected cleanup failure",
                )

            external = {
                name: Path(value["path"]).resolve()
                for name, value in workflow["tools"].items()
            }

            def process(args, **_kwargs):
                if "rev-parse" in args:
                    if "--show-toplevel" in args:
                        return SimpleNamespace(stdout=str(root) + "\n")
                    return SimpleNamespace(
                        stdout=workflow["sourceRevision"] + "\n"
                    )
                return SimpleNamespace(stdout="")

            with (
                patch(
                    "ml.evaluation.release_workflow._authenticate_external",
                    side_effect=lambda reference, _label: reference.path.resolve(),
                ),
                patch(
                    "ml.evaluation.release_workflow._closed_environment",
                    return_value={"PATH": "closed"},
                ),
                patch(
                    "ml.evaluation.release_workflow.shutil.which",
                    side_effect=lambda name, **_kwargs: str(external[name]),
                ),
                patch(
                    "ml.evaluation.release_workflow._input_references",
                    return_value=(),
                ),
                patch(
                    "ml.evaluation.release_workflow._run_capture_process",
                    side_effect=process,
                ),
                patch(
                    "ml.evaluation.release_workflow._execute_step",
                    side_effect=execute_step,
                ),
                patch(
                    "ml.evaluation.release_workflow.as_completed",
                    side_effect=cancellation_first,
                ),
                patch(
                    "ml.evaluation.release_workflow."
                    "_rollback_new_selection_outputs",
                    side_effect=rollback_with_report,
                ),
            ):
                with (
                    patch(
                        "ml.evaluation.release_workflow._stable_file_identity",
                        return_value=(1, 2, 3, 4, 5),
                    ),
                    self.assertRaisesRegex(
                        RuntimeError,
                        "later-plan failure",
                    ) as raised,
                ):
                    run(
                        workflow,
                        digest,
                        root,
                        execute=True,
                    )

            self.assertEqual(
                set(called),
                {
                    (20260811, 1),
                    (20260811, 2),
                    (20260811, 3),
                    (20260811, 4),
                },
            )
            self.assertTrue(
                (root / "release/20260811/epoch-1.selection.json").exists(),
                getattr(raised.exception, "__notes__", ()),
            )
            self.assertIn(
                "injected cleanup failure",
                " ".join(getattr(raised.exception, "__notes__", ())),
            )
            self.assertFalse(
                (root / "release/20260811/epoch-1.summary.json").exists()
            )
            self.assertIn(
                "portable Python cannot unlink",
                " ".join(getattr(raised.exception, "__notes__", ())),
            )

    def test_selection_cancellation_terminates_descendant_process(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
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
            cancel = threading.Event()
            errors: list[BaseException] = []

            def execute() -> None:
                try:
                    _run_step_process(
                        [sys.executable, "-c", parent, child, str(heartbeat)],
                        cwd=root,
                        environment=dict(os.environ),
                        cancel=cancel,
                    )
                except BaseException as error:
                    errors.append(error)

            worker = threading.Thread(target=execute)
            worker.start()
            deadline = time.monotonic() + 10
            while (
                (not heartbeat.exists() or heartbeat.stat().st_size < 2)
                and time.monotonic() < deadline
            ):
                time.sleep(0.02)
            self.assertTrue(heartbeat.exists(), "descendant never started")
            cancel.set()
            worker.join(timeout=15)
            self.assertFalse(worker.is_alive(), "cancelled runner did not settle")
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], ReleaseWorkflowError)
            time.sleep(0.2)
            settled_size = heartbeat.stat().st_size
            time.sleep(0.3)
            self.assertEqual(heartbeat.stat().st_size, settled_size)

    def test_successful_fast_parent_still_terminates_grandchild(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
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
            _run_step_process(
                [sys.executable, "-c", parent, child, str(heartbeat)],
                cwd=root,
                environment=dict(os.environ),
                cancel=None,
            )
            self.assertTrue(heartbeat.exists(), "grandchild never started")
            time.sleep(0.2)
            settled_size = heartbeat.stat().st_size
            time.sleep(0.3)
            self.assertEqual(heartbeat.stat().st_size, settled_size)

    def test_first_worker_failure_outranks_a_second_real_failure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "release").mkdir()
            workflow_path = root / "workflow.json"
            digest = _write(workflow_path, _workflow())
            workflow, _ = load_workflow(workflow_path)
            primary_recorded = threading.Event()
            original_record_failure = _SelectionWave.record_failure

            def record_failure(wave, error):
                original_record_failure(wave, error)
                if str(error) == "primary plan failure":
                    primary_recorded.set()

            def execute_step(step, *_args):
                if (
                    step.stage == "selection-fit-evaluation"
                    and step.seed == 20260811
                    and step.epoch == 1
                ):
                    raise RuntimeError("primary plan failure")
                if (
                    step.stage == "selection-fit-evaluation"
                    and step.seed == 20260811
                    and step.epoch == 2
                ):
                    if not primary_recorded.wait(timeout=1):
                        raise AssertionError("primary failure was not recorded")
                    raise RuntimeError("second plan failure")
                time.sleep(0.05)
                return (
                    step.argv,
                    {},
                    [
                        {
                            "path": path.as_posix(),
                            "sha256": "1" * 64,
                        }
                        for path in step.outputs
                    ],
                )

            def second_failure_first(futures):
                pending = tuple(futures)
                while not all(future.done() for future in pending):
                    time.sleep(0.001)
                return iter(sorted(
                    pending,
                    key=lambda future: str(future.exception())
                    != "second plan failure",
                ))

            def interrupt_after_primary(_futures):
                if not primary_recorded.wait(timeout=1):
                    raise AssertionError("primary failure was not recorded")
                raise KeyboardInterrupt("main-thread interruption")

            external = {
                name: Path(value["path"]).resolve()
                for name, value in workflow["tools"].items()
            }

            def process(args, **_kwargs):
                if "rev-parse" in args:
                    if "--show-toplevel" in args:
                        return SimpleNamespace(stdout=str(root) + "\n")
                    return SimpleNamespace(
                        stdout=workflow["sourceRevision"] + "\n"
                    )
                return SimpleNamespace(stdout="")

            with (
                patch(
                    "ml.evaluation.release_workflow._authenticate_external",
                    side_effect=lambda reference, _label: reference.path.resolve(),
                ),
                patch(
                    "ml.evaluation.release_workflow._stable_file_identity",
                    return_value=(1, 2, 3, 4, 5),
                ),
                patch(
                    "ml.evaluation.release_workflow._closed_environment",
                    return_value={"PATH": "closed"},
                ),
                patch(
                    "ml.evaluation.release_workflow.shutil.which",
                    side_effect=lambda name, **_kwargs: str(external[name]),
                ),
                patch(
                    "ml.evaluation.release_workflow._input_references",
                    return_value=(),
                ),
                patch(
                    "ml.evaluation.release_workflow._run_capture_process",
                    side_effect=process,
                ),
                patch(
                    "ml.evaluation.release_workflow._SelectionWave.record_failure",
                    autospec=True,
                    side_effect=record_failure,
                ),
                patch(
                    "ml.evaluation.release_workflow._execute_step",
                    side_effect=execute_step,
                ),
            ):
                with (
                    patch(
                        "ml.evaluation.release_workflow.as_completed",
                        side_effect=second_failure_first,
                    ),
                    self.assertRaisesRegex(
                        RuntimeError,
                        "primary plan failure",
                    ),
                ):
                    run(
                        workflow,
                        digest,
                        root,
                        execute=True,
                    )
                primary_recorded.clear()
                with (
                    patch(
                        "ml.evaluation.release_workflow.as_completed",
                        side_effect=interrupt_after_primary,
                    ),
                    self.assertRaisesRegex(
                        KeyboardInterrupt,
                        "main-thread interruption",
                    ),
                ):
                    run(
                        workflow,
                        digest,
                        root,
                        execute=True,
                    )

    def test_closed_python_bootstrap_is_deterministic_and_ignores_customization(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bad = root / "bad-marker"
            user_bad = root / "user-bad-marker"
            probe_output = root / "probe.json"
            hostile = root / "hostile"
            hostile.mkdir()
            (root / "sitecustomize.py").write_text(
                "from pathlib import Path\n"
                f"Path({str(bad)!r}).write_text('bad')\n",
                encoding="utf-8",
            )
            (root / "usercustomize.py").write_text(
                "from pathlib import Path\n"
                f"Path({str(user_bad)!r}).write_text('bad')\n",
                encoding="utf-8",
            )
            (hostile / "schema9_hostile_only.py").write_text(
                "VALUE = 'hostile'\n",
                encoding="utf-8",
            )
            (root / "approved_probe.py").write_text(
                "import importlib.util\n"
                "import json\n"
                "import os\n"
                "from pathlib import Path\n"
                "import sys\n"
                f"Path({str(probe_output)!r}).write_text(json.dumps({{\n"
                "    'hash': hash('drawbacktrainer-release-python-v1'),\n"
                "    'python_controls': {name: value for name, value in "
                "os.environ.items() if name.upper().startswith('PYTHON')},\n"
                "    'flags': {\n"
                "        'isolated': sys.flags.isolated,\n"
                "        'ignore_environment': sys.flags.ignore_environment,\n"
                "        'no_site': sys.flags.no_site,\n"
                "        'no_user_site': sys.flags.no_user_site,\n"
                "        'safe_path': sys.flags.safe_path,\n"
                "        'dont_write_bytecode': sys.flags.dont_write_bytecode,\n"
                "        'hash_randomization': sys.flags.hash_randomization,\n"
                "    },\n"
                "    'customizations': [name for name in "
                "('sitecustomize', 'usercustomize') if name in sys.modules],\n"
                "    'hostile_import_found': "
                "importlib.util.find_spec('schema9_hostile_only') is not None,\n"
                "}, sort_keys=True), encoding='utf-8')\n",
                encoding="utf-8",
            )
            workflow_path = root / "workflow.json"
            _write(workflow_path, _workflow())
            workflow, _ = load_workflow(workflow_path)
            command = list(build_plan(workflow, root)[0].argv)
            module_index = command.index("ml.evaluation.cli")
            command = command[: module_index + 1]
            command[module_index] = "approved_probe"
            hostile_environment = {
                "PYTHONPATH": str(hostile),
                "PYTHONHOME": str(hostile),
                "PYTHONUSERBASE": str(hostile),
                "PYTHONDONTWRITEBYTECODE": "0",
            }
            with patch.dict(os.environ, hostile_environment, clear=False):
                environment = dict(_closed_environment(root, {}))
            self.assertEqual(
                {
                    key: value
                    for key, value in environment.items()
                    if key.upper().startswith("PYTHON")
                },
                {"PYTHONHASHSEED": "0"},
            )
            observations: list[dict[str, object]] = []
            for _ in range(2):
                subprocess.run(
                    command,
                    cwd=root,
                    check=True,
                    env=environment,
                )
                observations.append(json.loads(probe_output.read_bytes()))
            self.assertEqual(observations[0], observations[1])
            self.assertEqual(
                observations[0]["python_controls"],
                {"PYTHONHASHSEED": "0"},
            )
            self.assertEqual(
                observations[0]["flags"],
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
            self.assertEqual(observations[0]["customizations"], [])
            self.assertFalse(observations[0]["hostile_import_found"])
            self.assertFalse(bad.exists())
            self.assertFalse(user_bad.exists())
            self.assertEqual(list(root.rglob("__pycache__")), [])
