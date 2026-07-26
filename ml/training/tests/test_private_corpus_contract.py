from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import _bootstrap  # noqa: F401

from drawback_ml.corpus_contract import (
    CorpusContractError,
    audit_private_corpus_split,
)
from test_corpus_contract import write_fixture


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def write_release(directory: Path) -> tuple[Path, dict[str, Path], dict[str, Path]]:
    monolithic_path = write_fixture(directory)
    monolithic = json.loads(monolithic_path.read_text(encoding="utf-8"))
    corpus_run_id = hashlib.sha256(monolithic_path.read_bytes()).hexdigest()
    corpus = {
        key: value
        for key, value in monolithic.items()
        if key not in {"rootSeed", "splits"}
    }
    private_paths: dict[str, Path] = {}
    dataset_paths: dict[str, Path] = {}
    commitments: dict[str, object] = {}
    for split in ("train", "validation", "test"):
        entry = monolithic["splits"][split]
        private = {
            "manifestVersion": 1,
            "corpusRunId": corpus_run_id,
            "split": split,
            "dataset": entry,
        }
        private_payload = canonical_bytes(private)
        private_path = directory / f"{split}.private.json"
        private_path.write_bytes(private_payload)
        private_paths[split] = private_path
        dataset_paths[split] = directory / entry["file"]
        commitments[split] = {
            "games": entry["games"],
            "rows": entry["rows"],
            "datasetBytes": entry["bytes"],
            "datasetSha256": entry["sha256"],
            "privateManifestSha256": hashlib.sha256(private_payload).hexdigest(),
        }
    root = {
        "releaseManifestVersion": 1,
        "corpusRunId": corpus_run_id,
        "corpus": corpus,
        "splits": commitments,
    }
    root_path = directory / "release.public.json"
    root_path.write_bytes(canonical_bytes(root))
    return root_path, private_paths, dataset_paths


class PrivateCorpusContractTests(unittest.TestCase):
    def test_authenticated_release_still_rejects_semantic_san_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, private, datasets = write_release(Path(temporary))
            lines = datasets["train"].read_text(encoding="utf-8").splitlines()
            first = json.loads(lines[0])
            first["san"] = "d4"
            lines[0] = json.dumps(first, separators=(",", ":"))
            payload = ("\n".join(lines) + "\n").encode("utf-8")
            datasets["train"].write_bytes(payload)

            private_value = json.loads(
                private["train"].read_text(encoding="utf-8")
            )
            private_value["dataset"]["bytes"] = len(payload)
            private_value["dataset"]["sha256"] = hashlib.sha256(payload).hexdigest()
            private_payload = canonical_bytes(private_value)
            private["train"].write_bytes(private_payload)

            root_value = json.loads(root.read_text(encoding="utf-8"))
            commitment = root_value["splits"]["train"]
            commitment["datasetBytes"] = len(payload)
            commitment["datasetSha256"] = hashlib.sha256(payload).hexdigest()
            commitment["privateManifestSha256"] = hashlib.sha256(
                private_payload
            ).hexdigest()
            root.write_bytes(canonical_bytes(root_value))

            with self.assertRaisesRegex(CorpusContractError, "semantic replay.*SAN"):
                audit_private_corpus_split(
                    root,
                    private["train"],
                    datasets["train"],
                    "train",
                    require_complete_catalog=False,
                )

    def test_train_audit_opens_only_authorized_manifest_and_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, private, datasets = write_release(Path(temporary))
            opened: list[Path] = []
            original_read_bytes = Path.read_bytes
            original_open = Path.open

            def tracked_read_bytes(path: Path) -> bytes:
                opened.append(path.resolve())
                return original_read_bytes(path)

            def tracked_open(path: Path, *args: object, **kwargs: object):
                opened.append(path.resolve())
                return original_open(path, *args, **kwargs)

            with patch.object(Path, "read_bytes", tracked_read_bytes), patch.object(
                Path, "open", tracked_open
            ):
                audited = audit_private_corpus_split(
                    root,
                    private["train"],
                    datasets["train"],
                    "train",
                    require_complete_catalog=False,
                )

            self.assertEqual(audited.split, "train")
            self.assertNotIn(private["test"].resolve(), opened)
            self.assertNotIn(datasets["test"].resolve(), opened)
            provenance = audited.provenance()
            self.assertNotIn("test", json.dumps(provenance))
            self.assertEqual(audited.seeds, (audited.seeds[0],))

    def test_rejects_private_manifest_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, private, datasets = write_release(Path(temporary))
            value = json.loads(private["train"].read_text(encoding="utf-8"))
            value["dataset"]["rows"] += 1
            private["train"].write_bytes(canonical_bytes(value))
            with self.assertRaisesRegex(CorpusContractError, "digest"):
                audit_private_corpus_split(
                    root,
                    private["train"],
                    datasets["train"],
                    "train",
                    require_complete_catalog=False,
                )

    def test_rejects_split_swap_without_resolving_test_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, private, datasets = write_release(Path(temporary))
            with self.assertRaises(CorpusContractError):
                audit_private_corpus_split(
                    root,
                    private["validation"],
                    datasets["validation"],
                    "train",
                    require_complete_catalog=False,
                )

    def test_rejects_noncanonical_release_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, private, datasets = write_release(Path(temporary))
            value = json.loads(root.read_text(encoding="utf-8"))
            root.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(CorpusContractError, "canonical"):
                audit_private_corpus_split(
                    root,
                    private["train"],
                    datasets["train"],
                    "train",
                    require_complete_catalog=False,
                )


if __name__ == "__main__":
    unittest.main()
