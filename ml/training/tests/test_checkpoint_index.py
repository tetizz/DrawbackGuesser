from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import _bootstrap  # noqa: F401

from drawback_ml.checkpoint import (
    CHECKPOINT_INDEX_FORMAT,
    checkpoint_path,
    verify_checkpoint_index,
    write_checkpoint_index,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_claim(root: Path, *, seed: object, epochs: object) -> Path:
    material = {
        "format": "drawbacktrainer-streaming-run",
        "version": 1,
        "config": {"seed": seed, "epochs": epochs},
        "runtime": {"device": "cpu"},
        "sampling": {"policy": "fixture"},
    }
    run_id = hashlib.sha256(
        json.dumps(
            material,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    path = root / "run.claim.json"
    path.write_text(
        json.dumps(
            {"run_id": run_id, **material},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


class CheckpointIndexTests(unittest.TestCase):
    def test_writes_exact_ordered_content_addresses(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            claim = write_claim(root, seed=7, epochs=2)
            first = checkpoint_path(root, 7, 1)
            second = checkpoint_path(root, 7, 2)
            first.write_bytes(b"epoch-one")
            second.write_bytes(b"epoch-two")

            output = write_checkpoint_index(root, seed=7, epochs=2)
            value = json.loads(output.read_text(encoding="utf-8"))

            self.assertEqual(value["format"], CHECKPOINT_INDEX_FORMAT)
            self.assertEqual(value["version"], 1)
            self.assertEqual(value["seed"], 7)
            self.assertEqual(value["epochs"], 2)
            self.assertEqual(
                value["runClaim"],
                {"file": "run.claim.json", "sha256": digest(claim)},
            )
            self.assertEqual(
                value["checkpoints"],
                [
                    {
                        "epoch": 1,
                        "file": first.name,
                        "sha256": digest(first),
                    },
                    {
                        "epoch": 2,
                        "file": second.name,
                        "sha256": digest(second),
                    },
                ],
            )
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                write_checkpoint_index(root, seed=7, epochs=2)
            verified = verify_checkpoint_index(output, digest(output))
            self.assertEqual(verified["seed"], 7)

            second.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "authentication failed"):
                verify_checkpoint_index(output, digest(output))

    def test_rejects_missing_claim_or_epoch_without_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "claim"):
                write_checkpoint_index(root, seed=1, epochs=1)
            write_claim(root, seed=1, epochs=1)
            with self.assertRaisesRegex(ValueError, "epoch 1"):
                write_checkpoint_index(root, seed=1, epochs=1)
            self.assertFalse((root / "checkpoint-index.claim.json").exists())
            self.assertFalse(tuple(root.glob(".checkpoint-index.*.tmp")))

    def test_rejects_invalid_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "seed"):
                write_checkpoint_index(root, seed=-1, epochs=1)
            with self.assertRaisesRegex(ValueError, "epochs"):
                write_checkpoint_index(root, seed=1, epochs=0)
            with self.assertRaisesRegex(ValueError, "seed"):
                write_checkpoint_index(root, seed=True, epochs=1)
            with self.assertRaisesRegex(ValueError, "epochs"):
                write_checkpoint_index(root, seed=1, epochs=True)

    def test_rejects_run_claim_identity_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_claim(root, seed=8, epochs=2)
            checkpoint_path(root, 7, 1).write_bytes(b"epoch-one")
            with self.assertRaisesRegex(ValueError, "disagree"):
                write_checkpoint_index(root, seed=7, epochs=1)

    def test_rejects_type_invalid_run_claim_identity(self) -> None:
        for bad_seed, bad_epochs in ((True, 1), (1.0, 1), (1, True), (1, 1.0)):
            with self.subTest(
                seed=bad_seed,
                epochs=bad_epochs,
            ), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                write_claim(root, seed=bad_seed, epochs=bad_epochs)
                checkpoint_path(root, 1, 1).write_bytes(b"epoch-one")
                with self.assertRaisesRegex(ValueError, "invalid"):
                    write_checkpoint_index(root, seed=1, epochs=1)

    def test_unique_temporary_does_not_delete_unowned_stale_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_claim(root, seed=7, epochs=1)
            checkpoint_path(root, 7, 1).write_bytes(b"epoch-one")
            stale = root / "checkpoint-index.claim.json.tmp"
            stale.write_text("unowned", encoding="utf-8")

            write_checkpoint_index(root, seed=7, epochs=1)

            self.assertEqual(stale.read_text(encoding="utf-8"), "unowned")

    def test_verifier_rejects_noncanonical_or_wrong_index_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "checkpoint-index.claim.json"
            path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "authentication"):
                verify_checkpoint_index(path, "0" * 64)
            with self.assertRaisesRegex(ValueError, "fields"):
                verify_checkpoint_index(path, digest(path))


if __name__ == "__main__":
    unittest.main()
