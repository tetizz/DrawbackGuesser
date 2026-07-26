from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

from ml.evaluation.tests.training_corpus_set_fixture import (
    training_corpus_set_fixture,
)
from ml.evaluation.training_frequency import (
    COUNTING_UNIT,
    FORMAT,
    VERSION,
    ContentAddressedFile,
    _canonical_compact,
    _canonical_pretty,
    _count_observed_player_games,
    _write_atomic_no_clobber,
    build_parser,
    load_training_frequency_artifact,
    verify_training_frequency_sources,
    write_training_frequency_artifact,
    HardNegativeBinding,
)
from ml.training.drawback_ml.symbolic_schema import SYMBOLIC_RULE_IDS
from ml.training.drawback_ml.training_corpus_set import (
    FROZEN_SUPPLEMENT_PROFILES,
)


def row(
    game_id: str,
    color: str,
    drawback: str,
    *,
    ply: int,
) -> dict[str, object]:
    move = "e2e4" if color == "white" else "e7e5"
    return {
        "gameId": game_id,
        "seed": 101,
        "fenBefore": (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/"
            "RNBQKBNR w KQkq - 0 1"
        ),
        "move": move,
        "moveNumber": ply // 2 + 1,
        "ply": ply,
        "playerColor": color,
        "historySan": [],
        "trueDrawback": drawback,
        "hiddenParameters": {},
        "drawbackInternalState": None,
        "ordinaryLegalMoves": [move],
        "drawbackLegalMoves": [move],
        "ruleTriggered": False,
        "forced": False,
        "clockMs": None,
        "result": {"kind": "active"},
    }


def artifact_value() -> dict[str, object]:
    corpus_set = training_corpus_set_fixture()
    primary = corpus_set["primary"]
    supplements = corpus_set["supplements"]
    assert isinstance(primary, dict)
    assert isinstance(supplements, list)
    white = [0] * len(SYMBOLIC_RULE_IDS)
    black = [0] * len(SYMBOLIC_RULE_IDS)
    white[0] = 7
    black[SYMBOLIC_RULE_IDS.index("checkers")] = 9
    return {
        "format": FORMAT,
        "version": VERSION,
        "counting_unit": COUNTING_UNIT,
        "training_corpus_set": corpus_set,
        "training_corpus_set_sha256": corpus_set["sha256"],
        "rule_ids": list(SYMBOLIC_RULE_IDS),
        "rule_ids_sha256": hashlib.sha256(
            _canonical_compact(list(SYMBOLIC_RULE_IDS))
        ).hexdigest(),
        "counts": {
            "white": white,
            "black": black,
            "white_total": 7,
            "black_total": 9,
        },
        "sources": {
            "primary": {
                "public_root": {
                    "file": "manifest.json",
                    "sha256": primary["release_root_sha256"],
                },
                "private_manifest": {
                    "file": "manifest.json",
                    "sha256": primary["private_train_manifest_sha256"],
                },
                "dataset": {
                    "file": "train.ndjson",
                    "sha256": primary["dataset_sha256"],
                    "bytes": primary["dataset_bytes"],
                },
                "corpus_run_id": primary["corpus_run_id"],
                "outcomes_sha256": primary["outcomes_sha256"],
            },
            "supplements": [
                {
                    "profile_id": item["profile_id"],
                    "profile_offset": item["profile_offset"],
                    "manifest": {
                        "file": f"{item['profile_id']}-manifest.json",
                        "sha256": item["manifest_sha256"],
                    },
                    "dataset": {
                        "file": f"{item['profile_id']}.ndjson",
                        "sha256": item["dataset_sha256"],
                        "bytes": item["dataset_bytes"],
                    },
                    "plan": {
                        "file": f"{item['profile_id']}-plan.json",
                        "sha256": item["plan_sha256"],
                    },
                    "generation_run_id": item["generation_run_id"],
                    "outcomes_sha256": item["outcomes_sha256"],
                }
                for item in supplements
            ],
        },
    }


def write_artifact(root: Path, value: object) -> ContentAddressedFile:
    payload = _canonical_pretty(value)
    path = root / "frequency.json"
    path.write_bytes(payload)
    return ContentAddressedFile(path, hashlib.sha256(payload).hexdigest())


class TrainingFrequencyTests(unittest.TestCase):
    def test_counts_each_observed_player_color_once_per_game(self) -> None:
        values = [
            row("one", "white", "vegan", ply=0),
            row("one", "black", "checkers", ply=1),
            row("one", "white", "vegan", ply=2),
            row("one", "black", "checkers", ply=3),
            row("two", "white", "truant", ply=0),
        ]
        source = io.BytesIO(
            b"".join(
                json.dumps(value, separators=(",", ":")).encode("utf-8")
                + b"\n"
                for value in values
            )
        )
        white, black = _count_observed_player_games(source)
        self.assertEqual(white[SYMBOLIC_RULE_IDS.index("vegan")], 1)
        self.assertEqual(white[SYMBOLIC_RULE_IDS.index("truant")], 1)
        self.assertEqual(sum(white), 2)
        self.assertEqual(black[SYMBOLIC_RULE_IDS.index("checkers")], 1)
        self.assertEqual(sum(black), 1)
        self.assertEqual(source.tell(), 0)

    def test_loads_canonical_artifact_and_returns_color_priors(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            reference = write_artifact(Path(raw), artifact_value())
            loaded = load_training_frequency_artifact(reference)
        self.assertEqual(loaded.rule_ids, tuple(SYMBOLIC_RULE_IDS))
        self.assertEqual(loaded.white_total, 7)
        self.assertEqual(loaded.black_total, 9)
        self.assertEqual(loaded.probabilities("white")[0], 1.0)
        self.assertEqual(
            loaded.probabilities("black")[
                SYMBOLIC_RULE_IDS.index("checkers")
            ],
            1.0,
        )
        self.assertEqual(
            tuple(loaded.promotion_priors()["white"]),
            tuple(SYMBOLIC_RULE_IDS),
        )
        self.assertEqual(
            sum(loaded.promotion_priors()["black"].values()), 1.0
        )
        with self.assertRaisesRegex(ValueError, "color"):
            loaded.probabilities("green")

    def test_rejects_wrong_expected_corpus_and_noncanonical_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            reference = write_artifact(root, artifact_value())
            with self.assertRaisesRegex(ValueError, "different corpus set"):
                load_training_frequency_artifact(reference, "f" * 64)
            value = artifact_value()
            payload = json.dumps(value).encode("utf-8")
            path = root / "noncanonical.json"
            path.write_bytes(payload)
            with self.assertRaisesRegex(ValueError, "not canonical"):
                load_training_frequency_artifact(
                    ContentAddressedFile(
                        path, hashlib.sha256(payload).hexdigest()
                    )
                )

    def test_rejects_count_source_and_rule_order_tampering(self) -> None:
        mutations = []
        wrong_total = artifact_value()
        wrong_total["counts"]["white_total"] = 8  # type: ignore[index]
        mutations.append((wrong_total, "do not sum"))
        wrong_source = artifact_value()
        wrong_source["sources"]["supplements"][0]["dataset"]["sha256"] = (  # type: ignore[index]
            "f" * 64
        )
        mutations.append((wrong_source, "disagrees with corpus set"))
        wrong_order = artifact_value()
        wrong_order["rule_ids"][0], wrong_order["rule_ids"][1] = (  # type: ignore[index]
            wrong_order["rule_ids"][1],  # type: ignore[index]
            wrong_order["rule_ids"][0],  # type: ignore[index]
        )
        mutations.append((wrong_order, "rule order"))
        for index, (value, message) in enumerate(mutations):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as raw:
                reference = write_artifact(Path(raw), value)
                with self.assertRaisesRegex(ValueError, message):
                    load_training_frequency_artifact(reference)

    def test_rejects_duplicate_json_keys_and_digest_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            duplicate = (
                b'{"format":"a","format":"b","version":1}\n'
            )
            path = root / "duplicate.json"
            path.write_bytes(duplicate)
            with self.assertRaisesRegex(ValueError, "duplicate key"):
                load_training_frequency_artifact(
                    ContentAddressedFile(
                        path, hashlib.sha256(duplicate).hexdigest()
                    )
                )
            reference = write_artifact(root, artifact_value())
            with self.assertRaisesRegex(ValueError, "sha256 does not match"):
                load_training_frequency_artifact(
                    ContentAddressedFile(reference.path, "f" * 64)
                )

    def test_recursive_verifier_requires_byte_identical_reproduction(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            reference = write_artifact(root, artifact_value())
            changed = root / "changed.json"
            changed.write_bytes(b"changed")
            reproduced = ContentAddressedFile(
                changed, hashlib.sha256(b"changed").hexdigest()
            )
            with patch(
                "ml.evaluation.training_frequency."
                "write_training_frequency_artifact",
                return_value=reproduced,
            ), self.assertRaisesRegex(ValueError, "not reproducible"):
                verify_training_frequency_sources(
                    reference,
                    public_root=root / "public.json",
                    private_train=root / "private.json",
                    primary_dataset=root / "train.ndjson",
                    hard_negatives=(),
                    expected_training_corpus_set_sha256=str(
                        artifact_value()["training_corpus_set_sha256"]
                    ),
                )

    def test_atomic_writer_never_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "artifact.json"
            _write_atomic_no_clobber(path, b"first")
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                _write_atomic_no_clobber(path, b"second")
            self.assertEqual(path.read_bytes(), b"first")
            self.assertEqual(list(path.parent.glob(".artifact.json.*.tmp")), [])

    def test_writer_aggregates_the_pinned_primary_and_six_supplements(self) -> None:
        corpus_set = training_corpus_set_fixture()
        primary_set = corpus_set["primary"]
        supplement_sets = corpus_set["supplements"]
        assert isinstance(primary_set, dict)
        assert isinstance(supplement_sets, list)

        def stream(game: str) -> io.BytesIO:
            values = [
                row(game, "white", "vegan", ply=0),
                row(game, "black", "checkers", ply=1),
                row(game, "white", "vegan", ply=2),
            ]
            return io.BytesIO(
                b"".join(
                    json.dumps(value, separators=(",", ":")).encode("utf-8")
                    + b"\n"
                    for value in values
                )
            )

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            public = root / "public.json"
            private = root / "private.json"
            primary_dataset = root / "primary.ndjson"
            primary_audited = SimpleNamespace(
                manifest_path=private,
                dataset_path=primary_dataset,
            )
            primary_lease = SimpleNamespace(
                audited=primary_audited,
                dataset=stream("primary"),
            )
            primary_identity = SimpleNamespace(
                release_root_sha256=primary_set["release_root_sha256"],
                private_train_manifest_sha256=primary_set[
                    "private_train_manifest_sha256"
                ],
                dataset_sha256=primary_set["dataset_sha256"],
                dataset_bytes=primary_set["dataset_bytes"],
                corpus_run_id=primary_set["corpus_run_id"],
                outcomes_sha256=primary_set["outcomes_sha256"],
            )
            bindings = tuple(
                HardNegativeBinding(
                    profile_id=item["profile_id"],
                    manifest=root / f"{item['profile_id']}.manifest.json",
                    dataset=root / f"{item['profile_id']}.ndjson",
                    plan=root / f"{item['profile_id']}.plan.json",
                )
                for item in supplement_sets
            )
            leases = [
                SimpleNamespace(dataset=stream(f"supplement-{index}"))
                for index in range(6)
            ]
            identities = [
                SimpleNamespace(
                    profile_id=item["profile_id"],
                    profile_offset=item["profile_offset"],
                    manifest_sha256=item["manifest_sha256"],
                    dataset_sha256=item["dataset_sha256"],
                    dataset_bytes=item["dataset_bytes"],
                    plan_sha256=item["plan_sha256"],
                    generation_run_id=item["generation_run_id"],
                    outcomes_sha256=item["outcomes_sha256"],
                )
                for item in supplement_sets
            ]
            with patch(
                "ml.evaluation.training_frequency."
                "open_audited_private_corpus_split",
                return_value=nullcontext(primary_lease),
            ), patch(
                "ml.evaluation.training_frequency."
                "open_audited_hard_negative_train_corpus",
                side_effect=[nullcontext(lease) for lease in leases],
            ), patch(
                "ml.evaluation.training_frequency._primary_identity",
                return_value=primary_identity,
            ), patch(
                "ml.evaluation.training_frequency._supplement_identity",
                side_effect=[
                    (identity, {}) for identity in identities
                ],
            ), patch(
                "ml.evaluation.training_frequency.create_training_corpus_set",
                return_value=corpus_set,
            ):
                output = root / "frequency.json"
                reference = write_training_frequency_artifact(
                    output,
                    public_root=public,
                    private_train=private,
                    primary_dataset=primary_dataset,
                    hard_negatives=bindings,
                    expected_training_corpus_set_sha256=str(
                        corpus_set["sha256"]
                    ),
                )
            loaded = load_training_frequency_artifact(reference)
            self.assertEqual(loaded.white_total, 7)
            self.assertEqual(loaded.black_total, 7)
            self.assertEqual(
                loaded.white_counts[SYMBOLIC_RULE_IDS.index("vegan")], 7
            )
            self.assertEqual(
                loaded.black_counts[SYMBOLIC_RULE_IDS.index("checkers")], 7
            )
            self.assertEqual(
                loaded.sources["primary"]["public_root"]["file"],  # type: ignore[index]
                public.name,
            )

    def test_cli_requires_explicit_release_inputs_and_six_bindings(self) -> None:
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["out.json"])
        arguments = parser.parse_args(
            [
                "out.json",
                "--public-root",
                "root.json",
                "--private-train",
                "private.json",
                "--primary-dataset",
                "train.ndjson",
                "--expected-training-corpus-set-sha256",
                "a" * 64,
                *sum(
                    (
                        [
                            "--hard-negative",
                            profile_id,
                            f"{profile_id}.manifest.json",
                            f"{profile_id}.ndjson",
                            f"{profile_id}.plan.json",
                        ]
                        for _offset, profile_id, _rules
                        in FROZEN_SUPPLEMENT_PROFILES
                    ),
                    [],
                ),
            ]
        )
        self.assertEqual(len(arguments.hard_negative), 6)


if __name__ == "__main__":
    unittest.main()
