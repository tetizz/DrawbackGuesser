from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

import _bootstrap  # noqa: F401

from drawback_ml.corpus_contract import (
    CorpusContractError,
    open_audited_private_corpus_split,
)
from test_private_corpus_contract import write_release


class PrivateCorpusLeaseTests(unittest.TestCase):
    def test_path_replacement_keeps_authenticated_dataset_handle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            root, private, datasets = write_release(directory)
            original = datasets["train"].read_bytes()
            replacement = directory / "replacement.ndjson"
            replacement.write_bytes(b'{"replacement":true}\n')
            with open_audited_private_corpus_split(
                root,
                private["train"],
                datasets["train"],
                "train",
                require_complete_catalog=False,
            ) as lease:
                try:
                    os.replace(replacement, datasets["train"])
                except PermissionError:
                    self.skipTest(
                        "platform sharing policy disallows replacing an open file"
                    )
                self.assertEqual(lease.dataset.read(), original)
                lease.dataset.seek(0)
            self.assertEqual(datasets["train"].read_bytes(), b'{"replacement":true}\n')

    def test_in_place_mutation_is_rejected_on_finalization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            root, private, datasets = write_release(directory)
            with self.assertRaisesRegex(CorpusContractError, "changed"):
                with open_audited_private_corpus_split(
                    root,
                    private["train"],
                    datasets["train"],
                    "train",
                    require_complete_catalog=False,
                ):
                    try:
                        with datasets["train"].open("r+b") as writer:
                            writer.seek(0)
                            writer.write(b"!")
                            writer.flush()
                    except PermissionError:
                        self.skipTest(
                            "platform sharing policy disallows concurrent writes"
                        )

    def test_root_and_private_handles_remain_pinned_after_path_replacement(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            root, private, datasets = write_release(directory)
            root_bytes = root.read_bytes()
            private_bytes = private["train"].read_bytes()
            replacement_root = directory / "replacement-root.json"
            replacement_private = directory / "replacement-private.json"
            replacement_root.write_bytes(b"{}\n")
            replacement_private.write_bytes(b"{}\n")
            with open_audited_private_corpus_split(
                root,
                private["train"],
                datasets["train"],
                "train",
                require_complete_catalog=False,
            ) as lease:
                try:
                    os.replace(replacement_root, root)
                    os.replace(replacement_private, private["train"])
                except PermissionError:
                    self.skipTest(
                        "platform sharing policy disallows replacing open manifests"
                    )
                self.assertEqual(lease.root.read(), root_bytes)
                self.assertEqual(lease.private_manifest.read(), private_bytes)
                lease.root.seek(0)
                lease.private_manifest.seek(0)

    def test_lease_owns_and_closes_every_handle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            root, private, datasets = write_release(directory)
            with open_audited_private_corpus_split(
                root,
                private["train"],
                datasets["train"],
                "train",
                require_complete_catalog=False,
            ) as lease:
                handles = (lease.root, lease.private_manifest, lease.dataset)
                self.assertTrue(all(not handle.closed for handle in handles))
                lease.verify_dataset_unchanged(chunk_size=7)
            self.assertTrue(all(handle.closed for handle in handles))

    def test_handles_close_when_consumer_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            root, private, datasets = write_release(directory)
            with self.assertRaisesRegex(RuntimeError, "consumer"):
                with open_audited_private_corpus_split(
                    root,
                    private["train"],
                    datasets["train"],
                    "train",
                    require_complete_catalog=False,
                ) as lease:
                    handles = (lease.root, lease.private_manifest, lease.dataset)
                    raise RuntimeError("consumer failed")
            self.assertTrue(all(handle.closed for handle in handles))


if __name__ == "__main__":
    unittest.main()
