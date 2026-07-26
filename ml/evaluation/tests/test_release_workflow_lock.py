from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from ml.evaluation.release_workflow_builder import (
    WorkflowBuilderError,
    load_builder_lock,
)
from ml.evaluation.release_workflow_lock import (
    build_builder_lock,
    main,
    write_builder_lock,
)
from ml.training.drawback_ml.checkpoint import (
    checkpoint_path,
    write_checkpoint_index,
)


def create_file(path: Path, value: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return path


def create_checkpoint_index(root: Path, seed: int) -> Path:
    directory = root / f"data/generated/models/{seed}"
    existing = directory / "checkpoint-index.claim.json"
    if existing.is_file():
        return existing
    directory.mkdir(parents=True, exist_ok=True)
    material = {
        "format": "drawbacktrainer-streaming-run",
        "version": 1,
        "config": {"seed": seed, "epochs": 8},
        "runtime": {"device": "cpu"},
        "sampling": {"policy": "fixture"},
    }
    run_id = hashlib.sha256(
        json.dumps(
            material,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    (directory / "run.claim.json").write_text(
        json.dumps(
            {"run_id": run_id, **material},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    for epoch in range(1, 9):
        checkpoint_path(directory, seed, epoch).write_bytes(
            f"{seed}:{epoch}".encode()
        )
    return write_checkpoint_index(directory, seed=seed, epochs=8)


def inputs(root: Path) -> dict[str, object]:
    tools = {
        name: create_file(root / "tools" / f"{name}.exe", name.encode())
        for name in ("browser", "git", "node", "pnpm")
    }
    return {
        "repository": root,
        "source_revision": "a" * 40,
        "tools": tools,
        "dataset": create_file(
            root / "data/generated/corpus/validation.ndjson",
            b"validation",
        ),
        "public_root": create_file(
            root / "data/releases/current/public/manifest.json",
            b"public",
        ),
        "private_validation": create_file(
            root / "data/releases/current/private/validation/manifest.json",
            b"private",
        ),
        "checkpoint_indexes": [
            create_checkpoint_index(root, seed)
            for seed in (20260811, 20260812, 20260813)
        ],
        "training_frequency": create_file(
            root / "data/generated/release/training-frequency.json",
            b"frequency",
        ),
        "browser_fixture": create_file(
            root / "data/generated/release/public-parity.json",
            b"fixture",
        ),
        "output_root": "data/generated/frozen-release",
    }


class ReleaseWorkflowLockTests(unittest.TestCase):
    def test_builds_only_from_explicit_files_in_fixed_seed_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arguments = inputs(root)
            with (
                patch.object(Path, "glob", side_effect=AssertionError("glob")),
                patch.object(
                    Path,
                    "iterdir",
                    side_effect=AssertionError("iterdir"),
                ),
            ):
                lock = build_builder_lock(**arguments)  # type: ignore[arg-type]

            candidates = lock["candidates"]
            assert isinstance(candidates, list)
            self.assertEqual(
                [candidate["seed"] for candidate in candidates],
                [20260811, 20260812, 20260813],
            )
            first = candidates[0]["checkpointIndex"]
            indexes = arguments["checkpoint_indexes"]
            assert isinstance(indexes, list)
            self.assertEqual(
                first["sha256"],
                hashlib.sha256(
                    indexes[0].read_bytes()
                ).hexdigest(),
            )

    def test_publishes_canonical_no_clobber_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock = build_builder_lock(
                **inputs(root)  # type: ignore[arg-type]
            )
            output = root / "data/generated/release/builder-lock.json"
            published, digest = write_builder_lock(
                output,
                lock,
                repository=root,
            )
            loaded = load_builder_lock(published)

            self.assertEqual(loaded, lock)
            self.assertEqual(
                digest,
                hashlib.sha256(published.read_bytes()).hexdigest(),
            )
            with self.assertRaises(FileExistsError):
                write_builder_lock(output, lock, repository=root)

    def test_rejects_missing_wrong_count_and_outside_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arguments = inputs(root)
            arguments["checkpoint_indexes"] = arguments[
                "checkpoint_indexes"
            ][:2]
            with self.assertRaisesRegex(
                WorkflowBuilderError,
                "exactly three",
            ):
                build_builder_lock(**arguments)  # type: ignore[arg-type]

            arguments = inputs(root)
            arguments["dataset"] = Path(temporary).parent / "outside.ndjson"
            with self.assertRaisesRegex(
                WorkflowBuilderError,
                "inside the repository",
            ):
                build_builder_lock(**arguments)  # type: ignore[arg-type]

            arguments = inputs(root)
            arguments["output_root"] = "docs/release"
            with self.assertRaisesRegex(
                WorkflowBuilderError,
                "inside data/generated",
            ):
                build_builder_lock(**arguments)  # type: ignore[arg-type]

            arguments = inputs(root)
            indexes = arguments["checkpoint_indexes"]
            assert isinstance(indexes, list)
            indexes.reverse()
            with self.assertRaisesRegex(
                WorkflowBuilderError,
                "does not belong",
            ):
                build_builder_lock(**arguments)  # type: ignore[arg-type]

    def test_output_is_confined_and_failure_leaves_no_partial_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock = build_builder_lock(
                **inputs(root)  # type: ignore[arg-type]
            )
            with self.assertRaisesRegex(
                WorkflowBuilderError,
                "data/generated",
            ):
                write_builder_lock(
                    root / "docs/builder-lock.json",
                    lock,
                    repository=root,
                )
            output = root / "data/generated/release/builder-lock.json"
            with patch(
                "ml.evaluation.release_workflow_lock.os.fsync",
                side_effect=OSError("disk failure"),
            ):
                with self.assertRaisesRegex(OSError, "disk failure"):
                    write_builder_lock(output, lock, repository=root)
            self.assertFalse(output.exists())
            self.assertFalse(
                tuple(output.parent.glob(".release-builder-lock.*.tmp"))
            )

    def test_concurrent_publication_has_one_complete_winner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock = build_builder_lock(
                **inputs(root)  # type: ignore[arg-type]
            )
            output = root / "data/generated/release/builder-lock.json"
            barrier = threading.Barrier(2)
            results: list[str] = []

            def publish() -> None:
                barrier.wait()
                try:
                    _path, digest = write_builder_lock(
                        output,
                        lock,
                        repository=root,
                    )
                    results.append(digest)
                except FileExistsError:
                    results.append("exists")

            first = threading.Thread(target=publish)
            second = threading.Thread(target=publish)
            first.start()
            second.start()
            first.join()
            second.join()

            self.assertEqual(results.count("exists"), 1)
            loaded = load_builder_lock(output)
            self.assertEqual(loaded, lock)
            self.assertIn(
                hashlib.sha256(output.read_bytes()).hexdigest(),
                results,
            )

    def test_post_commit_cleanup_error_does_not_report_false_failure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock = build_builder_lock(
                **inputs(root)  # type: ignore[arg-type]
            )
            output = root / "data/generated/release/builder-lock.json"
            original_unlink = Path.unlink

            def fail_temporary_unlink(
                path: Path,
                missing_ok: bool = False,
            ) -> None:
                if path.name.startswith(".release-builder-lock."):
                    raise OSError("cleanup failure")
                original_unlink(path, missing_ok=missing_ok)

            with patch.object(Path, "unlink", fail_temporary_unlink):
                published, digest = write_builder_lock(
                    output,
                    lock,
                    repository=root,
                )
            self.assertEqual(published, output)
            self.assertEqual(
                digest,
                hashlib.sha256(output.read_bytes()).hexdigest(),
            )

    def test_cli_resolves_project_paths_against_repository(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "repository"
            root.mkdir()
            arguments = inputs(root)
            output = Path("data/generated/release/builder-lock.json")
            indexes = arguments["checkpoint_indexes"]
            tools = arguments["tools"]
            assert isinstance(indexes, list)
            assert isinstance(tools, dict)
            argv = [
                str(output),
                "--repository",
                str(root),
                "--source-revision",
                "a" * 40,
            ]
            for name in ("browser", "git", "node", "pnpm"):
                argv.extend((f"--{name}", str(tools[name])))
            argv.extend(
                (
                    "--dataset",
                    "data/generated/corpus/validation.ndjson",
                    "--public-root",
                    "data/releases/current/public/manifest.json",
                    "--private-validation",
                    "data/releases/current/private/validation/manifest.json",
                )
            )
            for index in indexes:
                argv.extend(
                    (
                        "--checkpoint-index",
                        str(index.relative_to(root)),
                    )
                )
            argv.extend(
                (
                    "--training-frequency",
                    "data/generated/release/training-frequency.json",
                    "--browser-fixture",
                    "data/generated/release/public-parity.json",
                    "--output-root",
                    "data/generated/frozen-release",
                )
            )
            outside = Path(temporary) / "outside"
            outside.mkdir()
            previous = Path.cwd()
            try:
                os.chdir(outside)
                self.assertEqual(main(argv), 0)
            finally:
                os.chdir(previous)
            self.assertTrue((root / output).is_file())


if __name__ == "__main__":
    unittest.main()
