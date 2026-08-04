from __future__ import annotations

from copy import deepcopy
import hashlib
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

import _bootstrap  # noqa: F401

from capturable_fixture import capturable_row
from drawback_ml.capturable_baseline import _canonical_json
from drawback_ml.capturable_fixed_corpus import (
    AUDIT_TOOL_ID,
    CONVERSION_ID,
    ENGINE_SUBMODULE_COMMIT,
    ENGINE_REQUIRED_DIST_FILES,
    GENERATOR_DIRECTORY,
    GENERATOR_ENGINE_COMMIT,
    GENERATOR_LOCK_SHA256,
    GUESSER_ENGINE_REQUIRED_DIST_FILES,
    GUESSER_REQUIRED_DIST_FILES,
    REQUIRED_COREPACK_VERSION,
    REQUIRED_NODE_VERSION,
    REQUIRED_PNPM_VERSION,
    PNPM_RUNTIME_TREE_FILES,
    PNPM_RUNTIME_TREE_SHA256,
    ResolvedToolchain,
    STANDARD_INITIAL_FEN,
    TRACE_AGENT,
    TRACE_AUTHORITY,
    TRACE_FORMAT,
    TRACE_HYPOTHESIS_POLICY,
    TRACE_RANDOM_POLICY,
    TRACE_RESULT_KINDS,
    TRACE_RULESET,
    _audit_trace,
    _authenticate_active_toolchain,
    _authenticate_expected_environment,
    _authenticate_git_repository,
    _construct_corpus_verification,
    _distinct_regular_files_equal,
    _git_ignored,
    _is_link_or_junction,
    _isolated_package_environment,
    _load_pinned_capturable_dataset,
    _reproduce_trace,
    _resolve_toolchain,
    _run,
    _run_bounded,
    _sanitized_environment,
    _scrub_ignored_paths,
    _tracked_blob_sha256,
    _trusted_package_shell,
    _validate_receipt,
    _verify_conversion,
    _windows_known_directory,
    _windows_taskkill,
    main as corpus_main,
    reauthenticate_fixed_corpus_files,
    require_private_regular_file,
    require_private_root,
    require_isolated_python_runtime,
)
from drawback_ml.capturable_fixed_blend_contract import (
    CONFIRMATION_TEST_FILE,
    CONFIRMATION_TRACE_FILE,
    CORPUS_RECEIPT_FORMAT,
    CORPUS_RECEIPT_VERSION,
    FIXED_PROTOCOL_COMMIT,
    FIXED_PROTOCOL_FILE,
    FIXED_PROTOCOL_SHA256,
    GENERATION_SCHEDULE,
    PRIOR_REGISTRY_FILE,
    PRIOR_REGISTRY_GAME_COUNT,
    PRIOR_REGISTRY_SHA256,
    PRIOR_REGISTRY_SOURCE_COUNT,
)
from drawback_ml.capturable_fixed_schedule import (
    EXPECTED_FIXED_ASSIGNMENTS,
)
from drawback_ml.capturable_records import CapturableDatasetError


def _trace_record(assignment):
    secrets = {
        "black": {
            "drawbackId": assignment.black_rule_id,
            "drawbackInternalState": {},
            "hiddenParameters": {},
        },
        "white": {
            "drawbackId": assignment.white_rule_id,
            "drawbackInternalState": {},
            "hiddenParameters": {},
        },
    }
    return {
        "agents": {
            "black": deepcopy(TRACE_AGENT),
            "white": deepcopy(TRACE_AGENT),
        },
        "authorityId": TRACE_AUTHORITY,
        "finalPosition": {"fen": "fixture"},
        "format": TRACE_FORMAT,
        "gameId": assignment.game_id,
        "gameIndex": assignment.game_index,
        "hypothesisPolicy": TRACE_HYPOTHESIS_POLICY,
        "initialPosition": {
            "authorityId": TRACE_AUTHORITY,
            "fen": STANDARD_INITIAL_FEN,
            "terminal": None,
        },
        "parameterSeeds": {
            "black": assignment.black_parameter_seed,
            "white": assignment.white_parameter_seed,
        },
        "plies": [
            {"color": "white", "ply": 0},
            {"color": "black", "ply": 1},
        ],
        "plyLimit": 60,
        "randomPolicy": TRACE_RANDOM_POLICY,
        "result": {"kind": "draw"},
        "ruleset": TRACE_RULESET,
        "schemaVersion": 2,
        "secrets": {"final": secrets, "initial": secrets},
        "seed": assignment.gameplay_seed,
        "stoppedAtPlyLimit": False,
    }


def _write_trace(path: Path, tamper=None) -> None:
    payload = bytearray()
    for index, assignment in enumerate(EXPECTED_FIXED_ASSIGNMENTS):
        record = _trace_record(assignment)
        if index == 17 and tamper is not None:
            tamper(record)
        payload.extend(_canonical_json(record))
    path.write_bytes(bytes(payload))


def _receipt(trace_sha: str = "a" * 64, dataset_sha: str = "b" * 64):
    return {
        "audit": {
            "clean": True,
            "repository": "DrawbackGuesser",
            "revision": "f" * 40,
            "toolId": AUDIT_TOOL_ID,
        },
        "conversion": {
            "buildSha256": "c" * 64,
            "byteExact": True,
            "engineSubmoduleCommit": ENGINE_SUBMODULE_COMMIT,
            "id": CONVERSION_ID,
            "inputTraceSha256": trace_sha,
        },
        "dataset": {
            "authorityId": TRACE_AUTHORITY,
            "bytes": 2_000,
            "file": CONFIRMATION_TEST_FILE,
            "games": 625,
            "rows": 1_250,
            "schemaVersion": 8,
            "sha256": dataset_sha,
            "trueHypothesisSurvivalRows": 1_250,
            "twoColorGames": 625,
        },
        "engineSubmodule": {"commit": ENGINE_SUBMODULE_COMMIT},
        "format": CORPUS_RECEIPT_FORMAT,
        "generator": {
            "buildSha256": "d" * 64,
            "clean": True,
            "commit": GENERATOR_ENGINE_COMMIT,
            "lockfileSha256": GENERATOR_LOCK_SHA256,
            "repository": "DrawbackEngine",
        },
        "priorRegistry": {
            "file": PRIOR_REGISTRY_FILE,
            "games": PRIOR_REGISTRY_GAME_COUNT,
            "overlap": 0,
            "sha256": PRIOR_REGISTRY_SHA256,
            "sources": PRIOR_REGISTRY_SOURCE_COUNT,
        },
        "protocol": {
            "commit": FIXED_PROTOCOL_COMMIT,
            "file": FIXED_PROTOCOL_FILE,
            "sha256": FIXED_PROTOCOL_SHA256,
        },
        "schedule": GENERATION_SCHEDULE,
        "toolchain": {
            "corepack": {
                "sha256": "e" * 64,
                "version": REQUIRED_COREPACK_VERSION,
            },
            "node": {
                "sha256": "1" * 64,
                "version": REQUIRED_NODE_VERSION,
            },
            "pnpm": {
                "files": PNPM_RUNTIME_TREE_FILES,
                "sha256": PNPM_RUNTIME_TREE_SHA256,
                "version": REQUIRED_PNPM_VERSION,
            },
            "shell": {
                "sha256": "2" * 64,
            },
        },
        "trace": {
            "activeAtPlyLimit": 0,
            "authorityId": TRACE_AUTHORITY,
            "bytes": 1_000,
            "countPerLabelColorCell": 25,
            "file": CONFIRMATION_TRACE_FILE,
            "firstGameIndex": 0,
            "format": TRACE_FORMAT,
            "games": 625,
            "labelColorCells": 50,
            "lastGameIndex": 624,
            "orderedPairs": 625,
            "plies": 1_250,
            "policyRegenerationMatch": True,
            "randomPolicy": TRACE_RANDOM_POLICY,
            "resultKindCounts": {
                kind: 625 if kind == "draw" else 0
                for kind in TRACE_RESULT_KINDS
            },
            "ruleset": TRACE_RULESET,
            "schemaVersion": 2,
            "semanticReplayGames": 625,
            "semanticReplayPlies": 1_250,
            "sha256": trace_sha,
            "terminalGames": 625,
        },
        "version": CORPUS_RECEIPT_VERSION,
    }


def _toolchain() -> ResolvedToolchain:
    artifact = _receipt()["toolchain"]
    return ResolvedToolchain(
        node=Path("system-node"),
        corepack=Path("corepack.js"),
        pnpm_entrypoint=Path("pnpm.mjs"),
        pnpm_store=Path("pnpm-store"),
        shell=Path("shell"),
        environment={"PATH": "isolated"},
        artifact=artifact,
        runtime_owner=None,
    )


class CapturableFixedCorpusTests(unittest.TestCase):
    def test_nonzero_command_preserves_failure_when_cleanup_fails(self) -> None:
        process = SimpleNamespace(
            returncode=13,
            communicate=lambda *, timeout: ("output", "child failed"),
        )
        with (
            patch(
                "drawback_ml.capturable_fixed_corpus._popen_contained",
                return_value=(process, None),
            ),
            patch(
                "drawback_ml.capturable_fixed_corpus._terminate_process_tree",
                side_effect=OSError("cleanup failed"),
            ),
            self.assertRaises(subprocess.CalledProcessError) as raised,
        ):
            _run_bounded(
                ["child"],
                cwd=Path.cwd(),
                environment={},
                timeout=10,
            )
        self.assertEqual(raised.exception.returncode, 13)
        self.assertEqual(raised.exception.stderr, "child failed")
        self.assertIsInstance(raised.exception.__cause__, OSError)
        self.assertIn(
            "cleanup failed",
            " ".join(getattr(raised.exception, "__notes__", ())),
        )

    @unittest.skipUnless(os.name == "nt", "Windows-only cleanup contract")
    def test_taskkill_nonzero_fails_closed(self) -> None:
        with (
            patch.dict(
                os.environ,
                {
                    "SystemRoot": str(Path.cwd() / "attacker-windows"),
                    "PATH": str(Path.cwd() / "attacker-bin"),
                    "PATHEXT": ".EVIL",
                },
                clear=False,
            ),
            patch(
                "drawback_ml.capturable_fixed_corpus.subprocess.run",
                return_value=SimpleNamespace(returncode=9),
            ) as run_process,
            self.assertRaisesRegex(OSError, "exit code 9"),
        ):
            _windows_taskkill(12345)
        environment = run_process.call_args.kwargs["env"]
        system = _windows_known_directory("system")
        windows = _windows_known_directory("windows")
        self.assertEqual(environment["SystemRoot"], str(windows))
        self.assertEqual(environment["WINDIR"], str(windows))
        self.assertEqual(environment["ComSpec"], str(system / "cmd.exe"))
        self.assertEqual(environment["PATH"], str(system))
        self.assertEqual(environment["PATHEXT"], ".COM;.EXE;.BAT;.CMD")

    def test_fixed_entrypoints_require_isolated_python_311(self) -> None:
        valid_flags = SimpleNamespace(
            ignore_environment=1,
            no_user_site=1,
        )
        with (
            patch(
                "drawback_ml.capturable_fixed_corpus.sys.version_info",
                (3, 11, 9),
            ),
            patch(
                "drawback_ml.capturable_fixed_corpus."
                "sys.dont_write_bytecode",
                True,
            ),
            patch(
                "drawback_ml.capturable_fixed_corpus.sys.flags",
                valid_flags,
            ),
        ):
            require_isolated_python_runtime()

        invalid_runtimes = (
            ((3, 12, 0), True, valid_flags),
            ((3, 11, 9), False, valid_flags),
            (
                (3, 11, 9),
                True,
                SimpleNamespace(
                    ignore_environment=0,
                    no_user_site=1,
                ),
            ),
            (
                (3, 11, 9),
                True,
                SimpleNamespace(
                    ignore_environment=1,
                    no_user_site=0,
                ),
            ),
        )
        for version, dont_write_bytecode, flags in invalid_runtimes:
            with self.subTest(
                version=version,
                dont_write_bytecode=dont_write_bytecode,
                flags=flags,
            ):
                with (
                    patch(
                        "drawback_ml.capturable_fixed_corpus."
                        "sys.version_info",
                        version,
                    ),
                    patch(
                        "drawback_ml.capturable_fixed_corpus."
                        "sys.dont_write_bytecode",
                        dont_write_bytecode,
                    ),
                    patch(
                        "drawback_ml.capturable_fixed_corpus.sys.flags",
                        flags,
                    ),
                ):
                    with self.assertRaisesRegex(
                        CapturableDatasetError,
                        "Python 3.11 with -B -E -s",
                    ):
                        require_isolated_python_runtime()

        with (
            patch(
                "drawback_ml.capturable_fixed_corpus."
                "require_isolated_python_runtime",
                side_effect=CapturableDatasetError("runtime rejected"),
            ) as require_runtime,
            patch(
                "drawback_ml.capturable_fixed_corpus._parser",
            ) as parser,
        ):
            with self.assertRaisesRegex(
                CapturableDatasetError,
                "runtime rejected",
            ):
                corpus_main([])

        require_runtime.assert_called_once_with()
        parser.assert_not_called()

    def test_protocol_binding_matches_the_frozen_git_blob(self) -> None:
        repository = Path(__file__).resolve().parents[3]
        payload = subprocess.run(
            [
                "git",
                "cat-file",
                "blob",
                (
                    f"{FIXED_PROTOCOL_COMMIT}:docs/research/"
                    f"{FIXED_PROTOCOL_FILE}"
                ),
            ],
            cwd=repository,
            check=True,
            capture_output=True,
        ).stdout
        self.assertEqual(
            hashlib.sha256(payload).hexdigest(),
            FIXED_PROTOCOL_SHA256,
        )
        ancestor = subprocess.run(
            [
                "git",
                "merge-base",
                "--is-ancestor",
                FIXED_PROTOCOL_COMMIT,
                "HEAD",
            ],
            cwd=repository,
            check=False,
            capture_output=True,
        )
        self.assertEqual(ancestor.returncode, 0)

    def test_regenerated_outputs_must_be_distinct_real_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            submitted = root / "submitted.ndjson"
            regenerated = root / "regenerated.ndjson"
            hard_link = root / "hard-link.ndjson"
            symbolic_link = root / "symbolic-link.ndjson"
            submitted.write_bytes(b"same\n")
            regenerated.write_bytes(b"same\n")

            self.assertTrue(
                _distinct_regular_files_equal(
                    submitted,
                    regenerated,
                    "fixture",
                )
            )
            os.link(submitted, hard_link)
            with self.assertRaisesRegex(
                CapturableDatasetError,
                "aliases its submitted input",
            ):
                _distinct_regular_files_equal(
                    submitted,
                    hard_link,
                    "fixture",
                )
            try:
                symbolic_link.symlink_to(submitted)
            except OSError:
                return
            with self.assertRaisesRegex(
                CapturableDatasetError,
                "real regular files",
            ):
                _distinct_regular_files_equal(
                    submitted,
                    symbolic_link,
                    "fixture",
                )

    def test_private_inputs_reject_hardlinks_and_windows_junctions(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            submitted = root / CONFIRMATION_TRACE_FILE
            alias = root / "outside-alias.ndjson"
            submitted.write_bytes(b"trace\n")
            os.link(submitted, alias)
            with self.assertRaisesRegex(
                CapturableDatasetError,
                "single-link private regular file",
            ):
                require_private_regular_file(
                    root,
                    submitted,
                    CONFIRMATION_TRACE_FILE,
                    "confirmation trace",
                )

        if os.name != "nt":
            return
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            target = workspace / "target"
            junction = workspace / "junction"
            target.mkdir()
            shell = _trusted_package_shell()
            subprocess.run(
                [
                    str(shell),
                    "/d",
                    "/c",
                    "mklink",
                    "/J",
                    str(junction),
                    str(target),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            try:
                self.assertTrue(_is_link_or_junction(junction))
                with self.assertRaisesRegex(
                    CapturableDatasetError,
                    "real private directory",
                ):
                    require_private_root(junction)
            finally:
                os.rmdir(junction)

    def test_absent_ignored_runtime_directories_match_directory_rules(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(
                ["git", "init", "--quiet"],
                cwd=root,
                check=True,
                capture_output=True,
            )
            (root / ".gitignore").write_text(
                "node_modules/\n**/dist/\n",
                encoding="utf-8",
                newline="\n",
            )

            self.assertTrue(_git_ignored(root, "node_modules"))
            self.assertTrue(
                _git_ignored(root, "packages/example/dist")
            )

    def test_git_source_auth_rejects_ignored_source_and_index_flags(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(
                ["git", "init", "--quiet"],
                cwd=root,
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "fixture"],
                cwd=root,
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "config", "user.email", "fixture@example.test"],
                cwd=root,
                check=True,
                capture_output=True,
            )
            (root / ".gitignore").write_text(
                ".npmrc\nnode_modules/\n",
                encoding="utf-8",
                newline="\n",
            )
            tracked = root / "tracked.txt"
            tracked.write_text("tracked\n", encoding="utf-8", newline="\n")
            subprocess.run(
                ["git", "add", ".gitignore", "tracked.txt"],
                cwd=root,
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "commit", "--quiet", "-m", "fixture"],
                cwd=root,
                check=True,
                capture_output=True,
            )
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            _authenticate_git_repository(
                root,
                expected_commit=revision,
                require_detached=False,
                label="fixture",
                allowed_runtime_paths=("node_modules",),
            )

            (root / ".npmrc").write_text(
                "script-shell=attacker\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                CapturableDatasetError,
                "unauthenticated runtime path",
            ):
                _authenticate_git_repository(
                    root,
                    expected_commit=revision,
                    require_detached=False,
                    label="fixture",
                    allowed_runtime_paths=("node_modules",),
                )
            (root / ".npmrc").unlink()

            subprocess.run(
                ["git", "update-index", "--skip-worktree", "tracked.txt"],
                cwd=root,
                check=True,
                capture_output=True,
            )
            try:
                with self.assertRaisesRegex(
                    CapturableDatasetError,
                    "non-default tracking flags",
                ):
                    _authenticate_git_repository(
                        root,
                        expected_commit=revision,
                        require_detached=False,
                        label="fixture",
                        allowed_runtime_paths=("node_modules",),
                    )
            finally:
                subprocess.run(
                    [
                        "git",
                        "update-index",
                        "--no-skip-worktree",
                        "tracked.txt",
                    ],
                    cwd=root,
                    check=True,
                    capture_output=True,
                )

    def test_tracked_blob_hash_ignores_worktree_line_endings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(
                ["git", "init", "--quiet"],
                cwd=root,
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "fixture"],
                cwd=root,
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "config", "user.email", "fixture@example.test"],
                cwd=root,
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "config", "core.autocrlf", "true"],
                cwd=root,
                check=True,
                capture_output=True,
            )
            lockfile = root / "pnpm-lock.yaml"
            committed = b"lockfileVersion: '9.0'\nsettings:\n  autoInstallPeers: true\n"
            lockfile.write_bytes(committed)
            subprocess.run(
                ["git", "add", "pnpm-lock.yaml"],
                cwd=root,
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "commit", "--quiet", "-m", "fixture"],
                cwd=root,
                check=True,
                capture_output=True,
            )
            lockfile.write_bytes(committed.replace(b"\n", b"\r\n"))

            self.assertNotEqual(
                hashlib.sha256(lockfile.read_bytes()).hexdigest(),
                hashlib.sha256(committed).hexdigest(),
            )
            self.assertEqual(
                _tracked_blob_sha256(
                    root,
                    "pnpm-lock.yaml",
                    "fixture lockfile",
                ),
                hashlib.sha256(committed).hexdigest(),
            )

            with self.assertRaisesRegex(
                CapturableDatasetError,
                "tracked path is invalid",
            ):
                _tracked_blob_sha256(
                    root,
                    "../pnpm-lock.yaml",
                    "fixture lockfile",
                )

    def test_runtime_scrub_tolerates_only_already_missing_entries(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(
                ["git", "init", "--quiet"],
                cwd=root,
                check=True,
                capture_output=True,
            )
            (root / ".gitignore").write_text(
                "node_modules/\n",
                encoding="utf-8",
                newline="\n",
            )
            target = root / "node_modules"
            target.mkdir()
            (target / "package.js").write_text(
                "fixture\n",
                encoding="utf-8",
            )
            real_rmtree = shutil.rmtree

            def missing_during_walk(path, *, onerror):
                real_rmtree(path)
                try:
                    raise FileNotFoundError("entry already removed")
                except FileNotFoundError:
                    onerror(os.unlink, str(path / "package.js"), sys.exc_info())

            with patch(
                "drawback_ml.capturable_fixed_corpus.shutil.rmtree",
                side_effect=missing_during_walk,
            ):
                _scrub_ignored_paths(root, ("node_modules",))

            target.mkdir()
            with patch(
                "drawback_ml.capturable_fixed_corpus.shutil.rmtree",
                side_effect=lambda _path, *, onerror: onerror(
                    os.unlink,
                    str(target / "missing.js"),
                    (
                        FileNotFoundError,
                        FileNotFoundError("entry already removed"),
                        None,
                    ),
                ),
            ):
                with self.assertRaisesRegex(
                    CapturableDatasetError,
                    "could not scrub ignored runtime path",
                ):
                    _scrub_ignored_paths(root, ("node_modules",))

            with patch(
                "drawback_ml.capturable_fixed_corpus.shutil.rmtree",
                side_effect=lambda _path, *, onerror: onerror(
                    os.unlink,
                    str(target / "protected.js"),
                    (
                        PermissionError,
                        PermissionError("permission denied"),
                        None,
                    ),
                ),
            ):
                with self.assertRaisesRegex(
                    CapturableDatasetError,
                    "could not scrub ignored runtime path",
                ):
                    _scrub_ignored_paths(root, ("node_modules",))

            real_rmtree(target)
            nested = (
                target
                / ("a" * 90)
                / ("b" * 90)
                / ("c" * 90)
            )
            long_file = nested / "package.d.ts"
            if os.name == "nt":
                self.assertGreater(len(str(long_file)), 260)
                creation_nested = Path(f"\\\\?\\{nested.absolute()}")
            else:
                creation_nested = nested
            creation_nested.mkdir(parents=True)
            (creation_nested / "package.d.ts").write_text(
                "fixture\n",
                encoding="utf-8",
            )

            _scrub_ignored_paths(root, ("node_modules",))
            self.assertFalse(os.path.lexists(target))

    def test_private_root_rejects_both_public_repositories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            guesser = workspace / "DrawbackGuesser"
            submodule = guesser / "engine"
            sibling_engine = workspace / "DrawbackEngine"
            private = workspace / "DrawbackTrainingData"
            for path in (submodule, sibling_engine / "child", private):
                path.mkdir(parents=True)

            with patch(
                "drawback_ml.capturable_fixed_corpus.REPOSITORY_ROOT",
                guesser,
            ):
                for public_path in (
                    guesser,
                    submodule,
                    sibling_engine,
                    sibling_engine / "child",
                ):
                    with self.subTest(path=public_path):
                        with self.assertRaisesRegex(
                            CapturableDatasetError,
                            "outside both public repositories",
                        ):
                            require_private_root(public_path)
                self.assertEqual(require_private_root(private), private)

    def test_independent_trace_audit_authenticates_every_assignment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / CONFIRMATION_TRACE_FILE
            _write_trace(path)

            summary, identity = _audit_trace(path)

            self.assertEqual(summary["games"], 625)
            self.assertEqual(summary["orderedPairs"], 625)
            self.assertEqual(summary["plies"], 1_250)
            self.assertEqual(summary["semanticReplayGames"], 625)
            self.assertEqual(
                identity.sha256,
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )

    def test_trace_audit_rejects_seed_policy_and_framing_forgery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / CONFIRMATION_TRACE_FILE
            _write_trace(
                path,
                lambda record: record.__setitem__(
                    "seed",
                    int(record["seed"]) ^ 1,
                ),
            )
            with self.assertRaisesRegex(
                CapturableDatasetError,
                "schedule or policy mismatch",
            ):
                _audit_trace(path)

            _write_trace(
                path,
                lambda record: record["agents"]["white"][
                    "searchPolicy"
                ].__setitem__("maxNodes", 4_999),
            )
            with self.assertRaisesRegex(
                CapturableDatasetError,
                "schedule or policy mismatch",
            ):
                _audit_trace(path)

            _write_trace(path)
            path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n", 1))
            with self.assertRaisesRegex(
                CapturableDatasetError,
                "exact LF framing",
            ):
                _audit_trace(path)

    def test_receipt_contract_is_closed_and_cross_binds_hashes(self) -> None:
        execution = {"revision": "f" * 40}
        artifact = _receipt()
        _validate_receipt(artifact, execution)

        tampered = deepcopy(artifact)
        tampered["conversion"]["inputTraceSha256"] = "c" * 64
        with self.assertRaisesRegex(
            CapturableDatasetError,
            "conversion evidence",
        ):
            _validate_receipt(tampered, execution)
        tampered = deepcopy(artifact)
        tampered["toolchain"]["node"]["version"] = "v24.15.1"
        with self.assertRaisesRegex(
            CapturableDatasetError,
            "toolchain or build identity",
        ):
            _validate_receipt(tampered, execution)
        tampered = deepcopy(artifact)
        tampered["generator"]["buildSha256"] = "not-a-hash"
        with self.assertRaisesRegex(
            CapturableDatasetError,
            "toolchain or build identity",
        ):
            _validate_receipt(tampered, execution)
        tampered = deepcopy(artifact)
        tampered["unknown"] = True
        with self.assertRaisesRegex(
            CapturableDatasetError,
            "identity is invalid",
        ):
            _validate_receipt(tampered, execution)
        tampered = deepcopy(artifact)
        tampered["version"] = True
        with self.assertRaisesRegex(
            CapturableDatasetError,
            "identity is invalid",
        ):
            _validate_receipt(tampered, execution)
        tampered = deepcopy(artifact)
        tampered["trace"]["activeAtPlyLimit"] = -1
        tampered["trace"]["terminalGames"] = 626
        with self.assertRaisesRegex(
            CapturableDatasetError,
            "trace evidence",
        ):
            _validate_receipt(tampered, execution)
        tampered = deepcopy(artifact)
        tampered["dataset"]["rows"] = 1_250.0
        tampered["dataset"]["trueHypothesisSurvivalRows"] = 1_250.0
        with self.assertRaisesRegex(
            CapturableDatasetError,
            "conversion evidence",
        ):
            _validate_receipt(tampered, execution)

    def test_child_process_source_drift_fails_before_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / CONFIRMATION_TRACE_FILE).write_bytes(b"trace\n")
            (root / CONFIRMATION_TEST_FILE).write_bytes(b"dataset\n")
            dirty = False
            authentications = 0

            def authenticate(
                _root,
                _execution,
                _expected_toolchain=None,
                *,
                create_runtime=False,
            ):
                del create_runtime
                nonlocal authentications
                authentications += 1
                if dirty:
                    raise CapturableDatasetError(
                        "requires a clean committed worktree"
                    )
                return (
                    {"generator": True},
                    {"engine": True},
                    _toolchain(),
                )

            def reproduce(_root, _trace, _execution, _runtime):
                nonlocal dirty
                dirty = True

            with (
                patch(
                    "drawback_ml.capturable_fixed_corpus."
                    "authenticate_corpus_environment",
                    side_effect=authenticate,
                ),
                patch(
                    "drawback_ml.capturable_fixed_corpus._audit_trace",
                    return_value=({"plies": 2}, object()),
                ),
                patch(
                    "drawback_ml.capturable_fixed_corpus._reproduce_trace",
                    side_effect=reproduce,
                ),
                patch(
                    "drawback_ml.capturable_fixed_corpus."
                    "_authenticate_active_toolchain",
                ),
                patch(
                    "drawback_ml.capturable_fixed_corpus._verify_conversion",
                ) as converter,
            ):
                with self.assertRaisesRegex(
                    CapturableDatasetError,
                    "clean committed worktree",
                ):
                    _construct_corpus_verification(
                        root,
                        {"revision": "f" * 40},
                        {"gameIds": []},
                    )

            self.assertEqual(authentications, 3)
            converter.assert_not_called()

    def test_dataset_rows_and_digest_come_from_one_open_handle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / CONFIRMATION_TEST_FILE
            first = capturable_row(
                game_id="pinned-handle",
                color="white",
                triggered=False,
            )
            second = capturable_row(
                game_id="pinned-handle",
                color="black",
                triggered=False,
            )
            payload = _canonical_json(first) + _canonical_json(second)
            path.write_bytes(payload)

            rows, identity = _load_pinned_capturable_dataset(path)

            self.assertEqual(len(rows), 2)
            self.assertEqual(identity.bytes, len(payload))
            self.assertEqual(
                identity.sha256,
                hashlib.sha256(payload).hexdigest(),
            )

    def test_toolchain_is_explicit_versioned_and_environment_is_scrubbed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            install = Path(directory)
            node = install / "node.exe"
            corepack = (
                install
                / "node_modules"
                / "corepack"
                / "dist"
                / "corepack.js"
            )
            pnpm_root = install / "pnpm-runtime"
            pnpm_entrypoint = pnpm_root / "bin" / "pnpm.mjs"
            pnpm_entrypoint.parent.mkdir(parents=True)
            pnpm_entrypoint.write_bytes(b"fixed-pnpm")
            pnpm_store = install / "pnpm-store"
            pnpm_store.mkdir()
            corepack.parent.mkdir(parents=True)
            node.write_bytes(b"fixed-node")
            corepack.write_bytes(b"fixed-corepack")
            with (
                patch.dict(
                    os.environ,
                    {
                        "ComSpec": "poison-shell.exe",
                        "HTTPS_PROXY": "https://poison.invalid",
                        "PATH": "poison-path",
                        "PYTHONPATH": "poison",
                        "SystemRoot": "C:\\poison-windows",
                        "WINDIR": "C:\\poison-windows",
                    },
                    clear=False,
                ),
                patch(
                    "drawback_ml.capturable_fixed_corpus."
                    "_trusted_node_executable",
                    return_value=node,
                ),
                patch(
                    "drawback_ml.capturable_fixed_corpus._capture_output",
                    side_effect=(
                        REQUIRED_NODE_VERSION,
                        REQUIRED_COREPACK_VERSION,
                        REQUIRED_PNPM_VERSION,
                        str(pnpm_store),
                    ),
                ) as capture,
                patch(
                    "drawback_ml.capturable_fixed_corpus."
                    "_resolve_pnpm_runtime",
                    return_value=(
                        pnpm_root,
                        pnpm_entrypoint,
                        PNPM_RUNTIME_TREE_SHA256,
                    ),
                ),
                patch(
                    "drawback_ml.capturable_fixed_corpus."
                    "_hash_pnpm_runtime_tree",
                    return_value=(
                        PNPM_RUNTIME_TREE_SHA256,
                        PNPM_RUNTIME_TREE_FILES,
                    ),
                ),
            ):
                toolchain = _resolve_toolchain(install)

            self.assertEqual(toolchain.node, node)
            self.assertEqual(toolchain.corepack, corepack)
            self.assertEqual(
                toolchain.artifact["node"]["sha256"],
                hashlib.sha256(b"fixed-node").hexdigest(),
            )
            self.assertEqual(
                toolchain.artifact["corepack"]["sha256"],
                hashlib.sha256(b"fixed-corepack").hexdigest(),
            )
            self.assertEqual(
                capture.call_args_list[2].args[0],
                [
                    str(node),
                    str(pnpm_entrypoint),
                    f"--config.globalconfig={os.devnull}",
                    f"--config.userconfig={os.devnull}",
                    "--version",
                ],
            )
            self.assertEqual(
                toolchain.artifact["pnpm"],
                {
                    "files": PNPM_RUNTIME_TREE_FILES,
                    "sha256": PNPM_RUNTIME_TREE_SHA256,
                    "version": REQUIRED_PNPM_VERSION,
                },
            )
            self.assertNotEqual(
                toolchain.environment.get("ComSpec"),
                "poison-shell.exe",
            )
            self.assertNotIn("HTTPS_PROXY", toolchain.environment)
            self.assertNotIn("PYTHONPATH", toolchain.environment)
            self.assertNotEqual(toolchain.environment["PATH"], "poison-path")
            if os.name == "nt":
                self.assertNotEqual(
                    toolchain.environment["SystemRoot"],
                    "C:\\poison-windows",
                )
                self.assertNotEqual(
                    toolchain.environment["WINDIR"],
                    "C:\\poison-windows",
                )
                self.assertEqual(
                    toolchain.environment[
                        "NoDefaultCurrentDirectoryInExePath"
                    ],
                    "1",
                )
            runtime_bin = Path(
                toolchain.environment["PATH"].split(os.pathsep)[0]
            )
            shim = runtime_bin / ("pnpm.cmd" if os.name == "nt" else "pnpm")
            self.assertTrue(shim.is_file())
            self.assertIn(
                str(pnpm_entrypoint),
                shim.read_text(encoding="utf-8"),
            )
            self.assertIsNotNone(toolchain.runtime_owner)
            if toolchain.runtime_owner is not None:
                toolchain.runtime_owner.cleanup()
            with (
                patch(
                    "drawback_ml.capturable_fixed_corpus."
                    "_trusted_node_executable",
                    return_value=node,
                ),
                patch(
                    "drawback_ml.capturable_fixed_corpus._capture_output",
                    side_effect=(
                        "v24.15.1",
                        REQUIRED_COREPACK_VERSION,
                        REQUIRED_PNPM_VERSION,
                    ),
                ),
                patch(
                    "drawback_ml.capturable_fixed_corpus."
                    "_resolve_pnpm_runtime",
                    return_value=(
                        pnpm_root,
                        pnpm_entrypoint,
                        PNPM_RUNTIME_TREE_SHA256,
                    ),
                ),
                patch(
                    "drawback_ml.capturable_fixed_corpus."
                    "_hash_pnpm_runtime_tree",
                    return_value=(
                        PNPM_RUNTIME_TREE_SHA256,
                        PNPM_RUNTIME_TREE_FILES,
                    ),
                ),
            ):
                with self.assertRaisesRegex(
                    CapturableDatasetError,
                    "requires Node v24.15.0",
                ):
                    _resolve_toolchain(install)

        environment = _sanitized_environment(
            {
                "PATH": "trusted",
                "NODE_OPTIONS": "--require=poison.js",
                "node_path": "poison",
                "COREPACK_HOME": "poison",
                "npm_config_registry": "https://poison.invalid",
                "PNPM_HOME": "poison",
                "YARN_ENABLE_SCRIPTS": "poison",
                "GIT_CONFIG_GLOBAL": "poison",
                "git_exec_path": "poison",
                "GIT_WORK_TREE": "poison",
                "SAFE_VALUE": "retained",
            }
        )
        self.assertEqual(environment["PATH"], "trusted")
        self.assertEqual(environment["SAFE_VALUE"], "retained")
        self.assertEqual(environment["CI"], "1")
        self.assertEqual(environment["COREPACK_ENABLE_NETWORK"], "0")
        self.assertFalse(
            {
                "NODE_OPTIONS",
                "node_path",
                "COREPACK_HOME",
                "npm_config_registry",
                "PNPM_HOME",
                "YARN_ENABLE_SCRIPTS",
                "GIT_CONFIG_GLOBAL",
                "git_exec_path",
                "GIT_WORK_TREE",
            }
            & set(environment)
        )

    def test_expected_toolchain_hash_mismatch_never_executes_candidate(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            install = Path(directory)
            node = install / "node.exe"
            corepack = (
                install
                / "node_modules"
                / "corepack"
                / "dist"
                / "corepack.js"
            )
            pnpm_root = install / "pnpm-runtime"
            pnpm_entrypoint = pnpm_root / "bin" / "pnpm.mjs"
            pnpm_store = install / "pnpm-store"
            corepack.parent.mkdir(parents=True)
            pnpm_entrypoint.parent.mkdir(parents=True)
            pnpm_store.mkdir()
            node.write_bytes(b"candidate-node")
            corepack.write_bytes(b"candidate-corepack")
            pnpm_entrypoint.write_bytes(b"candidate-pnpm")
            shell = _trusted_package_shell()
            expected = {
                "corepack": {
                    "sha256": hashlib.sha256(
                        corepack.read_bytes()
                    ).hexdigest(),
                    "version": REQUIRED_COREPACK_VERSION,
                },
                "node": {
                    "sha256": "0" * 64,
                    "version": REQUIRED_NODE_VERSION,
                },
                "pnpm": {
                    "files": PNPM_RUNTIME_TREE_FILES,
                    "sha256": PNPM_RUNTIME_TREE_SHA256,
                    "version": REQUIRED_PNPM_VERSION,
                },
                "shell": {
                    "sha256": hashlib.sha256(
                        shell.read_bytes()
                    ).hexdigest(),
                },
            }
            with (
                patch(
                    "drawback_ml.capturable_fixed_corpus."
                    "_trusted_node_executable",
                    return_value=node,
                ),
                patch(
                    "drawback_ml.capturable_fixed_corpus."
                    "_resolve_pnpm_runtime",
                    return_value=(
                        pnpm_root,
                        pnpm_entrypoint,
                        PNPM_RUNTIME_TREE_SHA256,
                    ),
                ),
                patch(
                    "drawback_ml.capturable_fixed_corpus."
                    "_hash_pnpm_runtime_tree",
                    return_value=(
                        PNPM_RUNTIME_TREE_SHA256,
                        PNPM_RUNTIME_TREE_FILES,
                    ),
                ),
                patch(
                    "drawback_ml.capturable_fixed_corpus._capture_output",
                ) as capture,
            ):
                with self.assertRaisesRegex(
                    CapturableDatasetError,
                    "changed before execution",
                ):
                    _resolve_toolchain(
                        install,
                        expected,
                        create_runtime=False,
                    )
            capture.assert_not_called()

    def test_isolated_runtime_rejects_path_separator_in_private_root(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            unsafe_root = Path(directory) / f"unsafe{os.pathsep}root"
            unsafe_root.mkdir()
            with self.assertRaisesRegex(
                CapturableDatasetError,
                "not safe for the isolated environment",
            ):
                _isolated_package_environment(
                    private_root=unsafe_root,
                    node=Path(directory) / "node",
                    pnpm_entrypoint=Path(directory) / "pnpm.mjs",
                    pnpm_store=Path(directory) / "store",
                    shell=_trusted_package_shell(),
                )

    def test_active_toolchain_rejects_laundering_config_and_shim_drift(
        self,
    ) -> None:
        active = _toolchain()
        measured = ResolvedToolchain(
            node=Path("different-node"),
            corepack=active.corepack,
            pnpm_entrypoint=active.pnpm_entrypoint,
            pnpm_store=active.pnpm_store,
            shell=active.shell,
            environment={},
            artifact=active.artifact,
            runtime_owner=None,
        )
        with (
            patch(
                "drawback_ml.capturable_fixed_corpus."
                "authenticate_corpus_environment",
                return_value=({}, {}, measured),
            ),
            patch(
                "drawback_ml.capturable_fixed_corpus."
                "_authenticate_active_toolchain",
            ) as authenticate_active,
        ):
            with self.assertRaisesRegex(
                CapturableDatasetError,
                "toolchain path changed",
            ):
                _authenticate_expected_environment(
                    Path("private"),
                    {"revision": "f" * 40},
                    active,
                )
        authenticate_active.assert_not_called()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            node = root / "node.exe"
            corepack = root / "corepack.js"
            pnpm_root = root / "pnpm-runtime"
            pnpm_entrypoint = pnpm_root / "bin" / "pnpm.mjs"
            pnpm_store = root / "pnpm-store"
            pnpm_entrypoint.parent.mkdir(parents=True)
            pnpm_store.mkdir()
            node.write_bytes(b"active-node")
            corepack.write_bytes(b"active-corepack")
            pnpm_entrypoint.write_bytes(b"active-pnpm")
            shell = _trusted_package_shell()
            environment, owner = _isolated_package_environment(
                private_root=root,
                node=node,
                pnpm_entrypoint=pnpm_entrypoint,
                pnpm_store=pnpm_store,
                shell=shell,
            )
            artifact = {
                "corepack": {
                    "sha256": hashlib.sha256(
                        corepack.read_bytes()
                    ).hexdigest(),
                    "version": REQUIRED_COREPACK_VERSION,
                },
                "node": {
                    "sha256": hashlib.sha256(
                        node.read_bytes()
                    ).hexdigest(),
                    "version": REQUIRED_NODE_VERSION,
                },
                "pnpm": {
                    "files": PNPM_RUNTIME_TREE_FILES,
                    "sha256": PNPM_RUNTIME_TREE_SHA256,
                    "version": REQUIRED_PNPM_VERSION,
                },
                "shell": {
                    "sha256": hashlib.sha256(
                        shell.read_bytes()
                    ).hexdigest(),
                },
            }
            toolchain = ResolvedToolchain(
                node=node.resolve(strict=True),
                corepack=corepack.resolve(strict=True),
                pnpm_entrypoint=pnpm_entrypoint.resolve(strict=True),
                pnpm_store=pnpm_store.resolve(strict=True),
                shell=shell,
                environment=environment,
                artifact=artifact,
                runtime_owner=owner,
            )
            runtime_root = Path(owner.name)
            try:
                with (
                    patch(
                        "drawback_ml.capturable_fixed_corpus."
                        "_trusted_node_executable",
                        return_value=toolchain.node,
                    ),
                    patch(
                        "drawback_ml.capturable_fixed_corpus."
                        "_hash_pnpm_runtime_tree",
                        return_value=(
                            PNPM_RUNTIME_TREE_SHA256,
                            PNPM_RUNTIME_TREE_FILES,
                        ),
                    ),
                    patch(
                        "drawback_ml.capturable_fixed_corpus."
                        "_capture_output",
                        side_effect=(
                            REQUIRED_NODE_VERSION,
                            REQUIRED_COREPACK_VERSION,
                            REQUIRED_PNPM_VERSION,
                            str(toolchain.pnpm_store),
                        ),
                    ),
                ):
                    _authenticate_active_toolchain(toolchain)

                poisoned_config = runtime_root / "home" / ".npmrc"
                poisoned_config.write_text(
                    "script-shell=attacker\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    CapturableDatasetError,
                    "configuration directory changed",
                ):
                    _authenticate_active_toolchain(toolchain)
                poisoned_config.unlink()

                shadow = runtime_root / "bin" / "pnpm.exe"
                shadow.write_bytes(b"shadow")
                with (
                    patch(
                        "drawback_ml.capturable_fixed_corpus."
                        "_trusted_node_executable",
                        return_value=toolchain.node,
                    ),
                    patch(
                        "drawback_ml.capturable_fixed_corpus."
                        "_hash_pnpm_runtime_tree",
                        return_value=(
                            PNPM_RUNTIME_TREE_SHA256,
                            PNPM_RUNTIME_TREE_FILES,
                        ),
                    ),
                    self.assertRaisesRegex(
                        CapturableDatasetError,
                        "shim directory changed",
                    ),
                ):
                    _authenticate_active_toolchain(toolchain)
                shadow.unlink()

                shim = runtime_root / "bin" / (
                    "pnpm.cmd" if os.name == "nt" else "pnpm"
                )
                shim.write_bytes(b"poisoned shim")
                with (
                    patch(
                        "drawback_ml.capturable_fixed_corpus."
                        "_trusted_node_executable",
                        return_value=toolchain.node,
                    ),
                    patch(
                        "drawback_ml.capturable_fixed_corpus."
                        "_hash_pnpm_runtime_tree",
                        return_value=(
                            PNPM_RUNTIME_TREE_SHA256,
                            PNPM_RUNTIME_TREE_FILES,
                        ),
                    ),
                    self.assertRaisesRegex(
                        CapturableDatasetError,
                        "pnpm shim identity changed",
                    ),
                ):
                    _authenticate_active_toolchain(toolchain)
            finally:
                owner.cleanup()

    def test_pre_report_reauthentication_binds_rebuilt_outputs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace_payload = b"trace\n"
            dataset_payload = b"dataset\n"
            (root / CONFIRMATION_TRACE_FILE).write_bytes(trace_payload)
            (root / CONFIRMATION_TEST_FILE).write_bytes(dataset_payload)
            receipt = _receipt(
                trace_sha=hashlib.sha256(trace_payload).hexdigest(),
                dataset_sha=hashlib.sha256(dataset_payload).hexdigest(),
            )
            receipt["trace"]["bytes"] = len(trace_payload)
            receipt["trace"]["games"] = 1
            receipt["dataset"]["bytes"] = len(dataset_payload)
            receipt["dataset"]["rows"] = 1
            execution = {"revision": "f" * 40}

            with (
                patch(
                    "drawback_ml.capturable_fixed_corpus."
                    "authenticate_corpus_environment",
                    return_value=({}, {}, _toolchain()),
                ) as authenticate,
                patch(
                    "drawback_ml.capturable_fixed_corpus._hash_dist_tree",
                    side_effect=("d" * 64, "a" * 64, "b" * 64),
                ),
                patch(
                    "drawback_ml.capturable_fixed_corpus."
                    "_combine_build_tree_hashes",
                    return_value="c" * 64,
                ),
            ):
                reauthenticate_fixed_corpus_files(
                    root,
                    receipt,
                    execution,
                )

            self.assertEqual(authenticate.call_count, 2)
            self.assertEqual(
                authenticate.call_args_list[0].args,
                (root, execution, receipt["toolchain"]),
            )

            with (
                patch(
                    "drawback_ml.capturable_fixed_corpus."
                    "authenticate_corpus_environment",
                    return_value=({}, {}, _toolchain()),
                ),
                patch(
                    "drawback_ml.capturable_fixed_corpus._hash_dist_tree",
                    side_effect=("e" * 64, "a" * 64, "b" * 64),
                ),
                patch(
                    "drawback_ml.capturable_fixed_corpus."
                    "_combine_build_tree_hashes",
                    return_value="c" * 64,
                ),
            ):
                with self.assertRaisesRegex(
                    CapturableDatasetError,
                    "rebuilt outputs changed",
                ):
                    reauthenticate_fixed_corpus_files(
                        root,
                        receipt,
                        execution,
                    )

    def test_timed_out_command_terminates_descendant_process(self) -> None:
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
            with self.assertRaisesRegex(
                CapturableDatasetError, "timed process could not run"
            ):
                _run(
                    [sys.executable, "-c", parent, child, str(heartbeat)],
                    cwd=root,
                    operation="timed process",
                    timeout=1,
                    environment=dict(os.environ),
                )
            self.assertTrue(heartbeat.exists(), "descendant never started")
            time.sleep(0.2)
            settled_size = heartbeat.stat().st_size
            time.sleep(0.3)
            self.assertEqual(heartbeat.stat().st_size, settled_size)

    def test_successful_fast_parent_terminates_descendant_process(self) -> None:
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
                "from pathlib import Path; import subprocess,sys,time\n"
                "heartbeat=Path(sys.argv[2])\n"
                "subprocess.Popen([sys.executable,'-c',sys.argv[1],"
                "sys.argv[2]],stdin=subprocess.DEVNULL,"
                "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\n"
                "deadline=time.monotonic()+5\n"
                "while not heartbeat.exists():\n"
                " if time.monotonic() >= deadline: raise SystemExit(2)\n"
                " time.sleep(0.01)\n"
            )
            _run(
                [sys.executable, "-c", parent, child, str(heartbeat)],
                cwd=root,
                operation="fast parent process",
                timeout=10,
                environment=dict(os.environ),
            )
            self.assertTrue(heartbeat.exists(), "descendant never started")
            time.sleep(0.2)
            settled_size = heartbeat.stat().st_size
            time.sleep(0.3)
            self.assertEqual(heartbeat.stat().st_size, settled_size)

    def test_poisoned_ignored_outputs_are_scrubbed_and_rebuilt(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            root = workspace / "private"
            guesser = workspace / "DrawbackGuesser"
            generator = root / GENERATOR_DIRECTORY
            engine_submodule = guesser / "engine"
            for path in (generator, engine_submodule):
                path.mkdir(parents=True)
            trace = root / CONFIRMATION_TRACE_FILE
            dataset = root / CONFIRMATION_TEST_FILE
            trace.write_bytes(b'{"trace":true}\n')
            dataset.write_bytes(b'{"dataset":true}\n')
            poisoned_engine = (
                generator
                / "packages"
                / "simulation-trace"
                / "dist"
                / "poison.js"
            )
            poisoned_converter = (
                guesser
                / "apps"
                / "dataset-cli"
                / "dist"
                / "poison.js"
            )
            for poisoned in (poisoned_engine, poisoned_converter):
                poisoned.parent.mkdir(parents=True)
                poisoned.write_text("poison", encoding="utf-8")
            (generator / "node_modules").mkdir()
            (generator / "node_modules" / "poison").write_text(
                "poison",
                encoding="utf-8",
            )
            calls: list[tuple[str, ...]] = []

            def build_dist(base: Path, required_files) -> None:
                for relative in required_files:
                    output = base / relative
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_text(
                        f"rebuilt:{relative}\n",
                        encoding="utf-8",
                    )

            def run(arguments, **kwargs):
                values = tuple(arguments)
                calls.append(values)
                operation = kwargs["operation"]
                if operation == "full frozen Engine build":
                    build_dist(generator, ENGINE_REQUIRED_DIST_FILES)
                elif operation == "frozen dataset converter dependency build":
                    build_dist(guesser, GUESSER_REQUIRED_DIST_FILES)
                    build_dist(
                        engine_submodule,
                        GUESSER_ENGINE_REQUIRED_DIST_FILES,
                    )
                elif operation == "frozen Engine trace regeneration":
                    output_index = values.index("633450514") + 1
                    Path(values[output_index]).write_bytes(
                        trace.read_bytes()
                    )
                elif operation == (
                    "semantic replay and deterministic conversion"
                ):
                    output = Path(values[values.index("--output") + 1])
                    output.write_bytes(dataset.read_bytes())

            runtime = _toolchain()
            with (
                patch(
                    "drawback_ml.capturable_fixed_corpus.REPOSITORY_ROOT",
                    guesser,
                ),
                patch(
                    "drawback_ml.capturable_fixed_corpus._run",
                    side_effect=run,
                ),
                patch(
                    "drawback_ml.capturable_fixed_corpus."
                    "_authenticate_expected_environment",
                ),
                patch(
                    "drawback_ml.capturable_fixed_corpus._git_ignored",
                    return_value=True,
                ),
            ):
                engine_build_sha256 = _reproduce_trace(
                    root,
                    trace,
                    {"revision": "f" * 40},
                    runtime,
                )
                identity, converter_build_sha256 = _verify_conversion(
                    root,
                    trace,
                    dataset,
                    {"revision": "f" * 40},
                    runtime,
                )

            self.assertEqual(identity.sha256, hashlib.sha256(
                dataset.read_bytes()
            ).hexdigest())
            self.assertEqual(len(engine_build_sha256), 64)
            self.assertEqual(len(converter_build_sha256), 64)
            self.assertFalse(poisoned_engine.exists())
            self.assertFalse(poisoned_converter.exists())
            generation = next(
                call
                for call in calls
                if any(
                    value.endswith("player-private-batch-cli.js")
                    for value in call
                )
            )
            self.assertEqual(
                generation[:2],
                (
                    str(runtime.node),
                    str(
                        generator
                        / "apps"
                        / "engine-cli"
                        / "dist"
                        / "player-private-batch-cli.js"
                    ),
                ),
            )
            test_index = generation.index("test")
            self.assertEqual(
                generation[test_index : test_index + 8],
                (
                    "test",
                    "0",
                    "0",
                    "625",
                    "15",
                    "633442320",
                    "633446417",
                    "633450514",
                ),
            )
            output_index = generation.index("633450514") + 1
            self.assertEqual(
                generation[output_index + 1 :],
                ("60", "30", "1", "5000", "35", "standard"),
            )
            pnpm_calls = [
                call
                for call in calls
                if len(call) >= 2
                and call[:2]
                == (
                    str(runtime.node),
                    str(runtime.pnpm_entrypoint),
                )
            ]
            self.assertTrue(
                all(
                    call[:2]
                    == (
                        str(runtime.node),
                        str(runtime.pnpm_entrypoint),
                    )
                    for call in pnpm_calls
                )
            )
            conversion = next(
                call for call in calls if "--require-authority" in call
            )
            self.assertEqual(conversion[0], str(runtime.node))
            self.assertNotEqual(conversion[1], str(runtime.corepack))
            self.assertIn("--require-authority", conversion)
            self.assertIn("--require-evaluator", conversion)


if __name__ == "__main__":
    unittest.main()
