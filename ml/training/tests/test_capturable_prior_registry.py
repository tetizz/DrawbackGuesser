from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import _bootstrap  # noqa: F401

from drawback_ml.capturable_baseline import _canonical_json
from drawback_ml.capturable_prior_registry import (
    PRIOR_CORPUS_REGISTRY_FORMAT,
    build_prior_corpus_registry,
    load_prior_corpus_registry,
)
from drawback_ml.capturable_records import CapturableDatasetError


class CapturablePriorRegistryTests(unittest.TestCase):
    @staticmethod
    def _write_rows(path: Path, rows: list[object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            b"".join(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                + b"\n"
                for row in rows
            )
        )

    def test_builds_canonical_union_for_trace_and_dataset_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "private"
            self._write_rows(
                root / "z-trace.ndjson",
                [{"gameId": "game-b"}, {"gameId": "game-a"}],
            )
            self._write_rows(
                root / "nested" / "a-dataset.ndjson",
                [
                    {"evaluation": {"gameId": "game-a"}},
                    {"evaluation": {"gameId": "game-c"}},
                ],
            )
            output = root / "prior-registry.json"

            summary = build_prior_corpus_registry(root, output)
            artifact, digest = load_prior_corpus_registry(output)

            self.assertEqual(summary["artifactSha256"], digest)
            self.assertEqual(
                artifact["format"],
                PRIOR_CORPUS_REGISTRY_FORMAT,
            )
            self.assertEqual(
                artifact["gameIds"],
                ["game-a", "game-b", "game-c"],
            )
            self.assertEqual(
                [source["file"] for source in artifact["sources"]],
                ["nested/a-dataset.ndjson", "z-trace.ndjson"],
            )
            self.assertEqual(artifact["sourceCount"], 2)
            self.assertEqual(artifact["uniqueGameCount"], 3)

            with self.assertRaises(FileExistsError):
                build_prior_corpus_registry(root, output)

    def test_builder_rejects_duplicate_keys_and_missing_game_id(self) -> None:
        invalid_rows = (
            b'{"gameId":"first","gameId":"second"}\n',
            b'{"notGameId":"missing"}\n',
            b"\n",
        )
        for index, payload in enumerate(invalid_rows):
            with self.subTest(case=index), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "private"
                root.mkdir()
                (root / "invalid.ndjson").write_bytes(payload)
                with self.assertRaises(CapturableDatasetError):
                    build_prior_corpus_registry(
                        root,
                        root / "registry.json",
                    )

    def test_loader_rejects_rehashed_structural_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "private"
            self._write_rows(root / "a.ndjson", [{"gameId": "game-a"}])
            self._write_rows(root / "b.ndjson", [{"gameId": "game-b"}])
            output = root / "registry.json"
            build_prior_corpus_registry(root, output)
            artifact, _ = load_prior_corpus_registry(output)

            tampered = deepcopy(artifact)
            tampered["sources"].reverse()
            output.write_bytes(_canonical_json(tampered))
            with self.assertRaisesRegex(
                CapturableDatasetError,
                "not ordered and unique",
            ):
                load_prior_corpus_registry(output)


if __name__ == "__main__":
    unittest.main()
