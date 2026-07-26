from __future__ import annotations

import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

import _bootstrap  # noqa: F401

from drawback_ml import durable_publish
from drawback_ml.durable_publish import publish_bytes_durable


class DurablePublishTests(unittest.TestCase):
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
            self.assertEqual(list(root.glob("marker.json.tmp-*")), [])

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
                with self.assertRaisesRegex(OSError, "file fsync failed"):
                    publish_bytes_durable(destination, b"marker\n")

            self.assertFalse(destination.exists())
            self.assertEqual(list(root.glob("marker.json.tmp-*")), [])

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
                with self.assertRaisesRegex(OSError, "MoveFileExW failed"):
                    publish_bytes_durable(destination, b"marker\n")

            self.assertFalse(destination.exists())
            self.assertEqual(list(root.glob("marker.json.tmp-*")), [])

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

            self.assertEqual(list(root.glob("marker.json.tmp-*")), [])

if __name__ == "__main__":
    unittest.main()
