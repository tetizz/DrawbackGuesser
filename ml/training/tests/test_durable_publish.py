from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

import _bootstrap  # noqa: F401

from drawback_ml import durable_publish
from drawback_ml.durable_publish import (
    publish_bytes_durable,
    publish_bytes_durable_exact,
    publish_staged_file_durable,
)


class DurablePublishTests(unittest.TestCase):
    def test_staged_publication_is_bounded_memory_and_create_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staged = root / ".stream.ndjson.unique.tmp"
            destination = root / "stream.ndjson"
            payload = b"first\nsecond\n"
            staged.write_bytes(payload)

            publish_staged_file_durable(
                destination,
                staged,
                hashlib.sha256(payload).hexdigest(),
                label="stream",
            )

            self.assertEqual(destination.read_bytes(), payload)
            self.assertFalse(staged.exists())

    def test_staged_exact_recovery_rejects_different_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "stream.ndjson"
            destination.write_bytes(b"competitor\n")
            staged = root / ".stream.ndjson.unique.tmp"
            payload = b"expected\n"
            staged.write_bytes(payload)

            with self.assertRaisesRegex(ValueError, "do not match"):
                publish_staged_file_durable(
                    destination,
                    staged,
                    hashlib.sha256(payload).hexdigest(),
                    label="stream",
                    recover_exact=True,
                )

            self.assertEqual(destination.read_bytes(), b"competitor\n")

    def test_windows_staged_exact_recovery_retains_unique_scratch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "stream.ndjson"
            staged = root / ".stream.ndjson.unique.tmp"
            payload = b"expected\n"
            destination.write_bytes(payload)
            staged.write_bytes(payload)

            with (
                patch.object(durable_publish, "_PLATFORM", "nt"),
                patch.object(
                    durable_publish,
                    "_move_windows_write_through",
                    side_effect=FileExistsError(destination),
                ),
            ):
                publish_staged_file_durable(
                    destination,
                    staged,
                    hashlib.sha256(payload).hexdigest(),
                    label="stream",
                    recover_exact=True,
                )

            self.assertEqual(destination.read_bytes(), payload)
            self.assertEqual(staged.read_bytes(), payload)

    def test_exact_recovery_accepts_only_identical_stable_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "marker.json"
            publish_bytes_durable_exact(
                destination,
                b"published\n",
                label="marker",
            )
            publish_bytes_durable_exact(
                destination,
                b"published\n",
                label="marker",
            )
            with self.assertRaisesRegex(ValueError, "do not match"):
                publish_bytes_durable_exact(
                    destination,
                    b"different\n",
                    label="marker",
                )
            self.assertEqual(destination.read_bytes(), b"published\n")

    def test_exact_recovery_authenticates_concurrent_creator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "marker.json"

            def concurrent_publish(path: Path, payload: bytes) -> None:
                path.write_bytes(payload)
                raise FileExistsError(path)

            with patch.object(
                durable_publish,
                "publish_bytes_durable",
                side_effect=concurrent_publish,
            ):
                publish_bytes_durable_exact(
                    destination,
                    b"published\n",
                    label="marker",
                )

            self.assertEqual(destination.read_bytes(), b"published\n")

    def test_exact_recovery_rejects_mutation_during_final_path_check(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "marker.json"
            original = b"original\n"
            replacement = b"tampered\n"
            destination.write_bytes(original)
            real_stat = Path.stat
            mutated = False

            def mutate_before_final_fstat(path: Path, *args, **kwargs):
                nonlocal mutated
                if (
                    path == destination
                    and kwargs.get("follow_symlinks") is False
                    and not mutated
                ):
                    mutated = True
                    with destination.open("r+b", buffering=0) as stream:
                        stream.write(replacement)
                return real_stat(path, *args, **kwargs)

            with (
                patch.object(
                    Path,
                    "stat",
                    autospec=True,
                    side_effect=mutate_before_final_fstat,
                ),
                self.assertRaisesRegex(ValueError, "changed while"),
            ):
                publish_bytes_durable_exact(
                    destination,
                    original,
                    label="marker",
                )

            self.assertTrue(mutated)
            self.assertEqual(destination.read_bytes(), replacement)

    def test_temp_path_replacement_is_retained_not_unlinked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "marker.json"
            replacement = root / "replacement"
            replacement.write_bytes(b"attacker replacement\n")
            original_stat = Path.stat
            original_unlink = Path.unlink
            swapped = False

            def swap_temp(path: Path, *args, **kwargs):
                nonlocal swapped
                if ".tmp-" in path.name and not swapped:
                    swapped = True
                    original_unlink(path)
                    replacement.replace(path)
                return original_stat(path, *args, **kwargs)

            with (
                patch.object(durable_publish, "_PLATFORM", "posix"),
                patch.object(
                    durable_publish,
                    "_fsync_parent_directory",
                ),
                patch.object(
                    Path,
                    "stat",
                    autospec=True,
                    side_effect=swap_temp,
                ),
                self.assertRaisesRegex(
                    OSError, "temporary cleanup failed"
                ),
            ):
                publish_bytes_durable(destination, b"published\n")

            self.assertEqual(destination.read_bytes(), b"published\n")
            leftovers = list(root.glob("marker.json.tmp-*"))
            self.assertEqual(len(leftovers), 1)
            if os.name == "nt":
                self.assertEqual(leftovers[0].read_bytes(), b"published\n")
                self.assertEqual(
                    replacement.read_bytes(), b"attacker replacement\n"
                )
            else:
                self.assertEqual(
                    leftovers[0].read_bytes(), b"attacker replacement\n"
                )

    def test_posix_temp_cleanup_failure_is_reported_after_publication(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "marker.json"
            original_unlink = Path.unlink

            def fail_temp_cleanup(path: Path, *args, **kwargs) -> None:
                if ".tmp-" in path.name:
                    raise OSError("temp unlink failed")
                original_unlink(path, *args, **kwargs)

            with (
                patch.object(durable_publish, "_PLATFORM", "posix"),
                patch.object(
                    durable_publish,
                    "_fsync_parent_directory",
                ),
                patch.object(
                    Path,
                    "unlink",
                    autospec=True,
                    side_effect=fail_temp_cleanup,
                ),
                self.assertRaisesRegex(
                    OSError, "temporary cleanup failed"
                ),
            ):
                publish_bytes_durable(destination, b"published\n")

            self.assertEqual(destination.read_bytes(), b"published\n")
            self.assertEqual(len(list(root.glob("marker.json.tmp-*"))), 1)

    def test_primary_publish_error_keeps_cleanup_failure_as_note(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "marker.json"

            with (
                patch.object(durable_publish, "_PLATFORM", "posix"),
                patch.object(
                    durable_publish,
                    "_fsync_parent_directory",
                    side_effect=OSError("directory sync failed"),
                ),
                patch.object(
                    Path,
                    "unlink",
                    autospec=True,
                    side_effect=OSError("temp unlink failed"),
                ),
                self.assertRaisesRegex(
                    OSError, "directory sync failed"
                ) as raised,
            ):
                publish_bytes_durable(destination, b"published\n")

            self.assertEqual(destination.read_bytes(), b"published\n")
            self.assertIn(
                "temporary cleanup failed",
                " ".join(getattr(raised.exception, "__notes__", ())),
            )

    def test_create_only_publication_preserves_bytes_and_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "marker.json"
            publish_bytes_durable(destination, b'{"state":"consumed"}\n')

            self.assertEqual(
                destination.read_bytes(),
                b'{"state":"consumed"}\n',
            )
            if os.name != "nt":
                self.assertEqual(
                    stat.S_IMODE(destination.stat().st_mode),
                    0o600,
                )
            with self.assertRaises(FileExistsError):
                publish_bytes_durable(destination, b"replacement\n")
            self.assertEqual(
                destination.read_bytes(),
                b'{"state":"consumed"}\n',
            )
            self.assertEqual(
                len(list(root.glob("marker.json.tmp-*"))),
                1 if os.name == "nt" else 0,
            )

    def test_file_fsync_failure_prevents_publication_and_cleans_temp(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "marker.json"
            with patch.object(
                durable_publish.os,
                "fsync",
                side_effect=OSError("file fsync failed"),
            ):
                with self.assertRaisesRegex(
                    OSError, "file fsync failed"
                ) as raised:
                    publish_bytes_durable(destination, b"marker\n")

            self.assertFalse(destination.exists())
            self.assertEqual(
                len(list(root.glob("marker.json.tmp-*"))),
                1 if os.name == "nt" else 0,
            )
            if os.name == "nt":
                self.assertIn(
                    "temporary cleanup failed",
                    " ".join(getattr(raised.exception, "__notes__", ())),
                )

    def test_link_sync_failure_is_fail_closed_and_keeps_destination(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "marker.json"
            with (
                patch.object(durable_publish, "_PLATFORM", "posix"),
                patch.object(
                    durable_publish,
                    "_fsync_parent_directory",
                    side_effect=OSError("directory sync failed"),
                ),
            ):
                with self.assertRaisesRegex(OSError, "directory sync failed"):
                    publish_bytes_durable(destination, b"consumed\n")

            self.assertEqual(destination.read_bytes(), b"consumed\n")
            self.assertEqual(list(root.glob("marker.json.tmp-*")), [])
            with self.assertRaises(FileExistsError):
                publish_bytes_durable(destination, b"second\n")

    def test_parent_directory_is_synced_only_after_link(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "marker.json"
            events: list[str] = []
            real_link = durable_publish.os.link

            def link(source: Path, target: Path) -> None:
                real_link(source, target)
                events.append("link")

            def sync(parent: Path) -> None:
                self.assertEqual(parent, root)
                events.append("sync")

            with (
                patch.object(durable_publish, "_PLATFORM", "posix"),
                patch.object(durable_publish.os, "link", side_effect=link),
                patch.object(
                    durable_publish,
                    "_fsync_parent_directory",
                    side_effect=sync,
                ),
            ):
                publish_bytes_durable(destination, b"ordered\n")

            self.assertEqual(events, ["link", "sync"])

    def test_windows_move_uses_only_write_through_flag(self) -> None:
        source = Path("marker.tmp")
        destination = Path("marker.json")
        with patch.object(
            durable_publish,
            "_call_windows_move_file_ex",
        ) as move:
            durable_publish._move_windows_write_through(
                source,
                destination,
            )

        move.assert_called_once_with(
            source,
            destination,
            durable_publish._MOVEFILE_WRITE_THROUGH,
        )
        self.assertEqual(durable_publish._MOVEFILE_WRITE_THROUGH, 0x8)

    def test_windows_branch_moves_then_verifies_without_posix_calls(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "marker.json"
            events: list[str] = []

            def move(source: Path, target: Path) -> None:
                self.assertTrue(source.exists())
                source.replace(target)
                events.append("move")

            def verify(path: Path, expected: bytes) -> None:
                self.assertEqual(path.read_bytes(), expected)
                events.append("verify")

            with (
                patch.object(durable_publish, "_PLATFORM", "nt"),
                patch.object(
                    durable_publish,
                    "_move_windows_write_through",
                    side_effect=move,
                ),
                patch.object(
                    durable_publish,
                    "_verify_published_bytes",
                    side_effect=verify,
                ),
                patch.object(durable_publish.os, "link") as posix_link,
                patch.object(
                    durable_publish,
                    "_fsync_parent_directory",
                ) as directory_sync,
            ):
                publish_bytes_durable(destination, b"ordered\n")

            self.assertEqual(events, ["move", "verify"])
            posix_link.assert_not_called()
            directory_sync.assert_not_called()
            self.assertEqual(list(root.glob("marker.json.tmp-*")), [])

    def test_windows_move_failure_prevents_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "marker.json"
            with (
                patch.object(durable_publish, "_PLATFORM", "nt"),
                patch.object(
                    durable_publish,
                    "_move_windows_write_through",
                    side_effect=OSError("MoveFileExW failed"),
                ),
            ):
                with self.assertRaisesRegex(
                    OSError, "MoveFileExW failed"
                ) as raised:
                    publish_bytes_durable(destination, b"marker\n")

            self.assertFalse(destination.exists())
            self.assertEqual(len(list(root.glob("marker.json.tmp-*"))), 1)
            self.assertIn(
                "retained temporary after closed-handle failure",
                " ".join(getattr(raised.exception, "__notes__", ())),
            )

    def test_windows_verification_failure_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "marker.json"

            def move(source: Path, target: Path) -> None:
                if target.exists():
                    raise FileExistsError(target)
                source.replace(target)

            with (
                patch.object(durable_publish, "_PLATFORM", "nt"),
                patch.object(
                    durable_publish,
                    "_move_windows_write_through",
                    side_effect=move,
                ),
                patch.object(
                    durable_publish,
                    "_verify_published_bytes",
                    side_effect=OSError("verification failed"),
                ),
            ):
                with self.assertRaisesRegex(OSError, "verification failed"):
                    publish_bytes_durable(destination, b"consumed\n")

                self.assertEqual(destination.read_bytes(), b"consumed\n")
                with self.assertRaises(FileExistsError):
                    publish_bytes_durable(destination, b"second\n")

            self.assertEqual(len(list(root.glob("marker.json.tmp-*"))), 1)

if __name__ == "__main__":
    unittest.main()
