from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import _bootstrap  # noqa: F401

from drawback_ml.corpus_contract import (
    CorpusContractError,
    PREPARED_RULE_IDS,
    _evaluator_request_digest,
    _normalize_fen,
    audit_corpus_split,
)
from drawback_ml.splits import Split, assign_split
from drawback_ml.symbolic_schema import (
    SYMBOLIC_FEATURE_VERSION,
    SYMBOLIC_RULE_IDS,
)


START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
AFTER_E4_FEN = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
AFTER_E4_E5_FEN = (
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
)
WHITE_ROOTS = (
    "g1h3", "g1f3", "b1c3", "b1a3", "h2h3", "g2g3", "f2f3",
    "e2e3", "d2d3", "c2c3", "b2b3", "a2a3", "h2h4", "g2g4",
    "f2f4", "e2e4", "d2d4", "c2c4", "b2b4", "a2a4",
)
BLACK_ROOTS_AFTER_E4 = (
    "g8h6", "g8f6", "b8c6", "b8a6", "h7h6", "g7g6", "f7f6",
    "e7e6", "d7d6", "c7c6", "b7b6", "a7a6", "h7h5", "g7g5",
    "f7f5", "e7e5", "d7d5", "c7c5", "b7b5", "a7a5",
)
BINARY_DIGEST = "11" * 32
OPTIONS_DIGEST = "22" * 32
ENGINE_FINGERPRINT = f"stockfish:18:{BINARY_DIGEST}:{OPTIONS_DIGEST}"
POLICY_ID = "stockfish-bestmove-v1"
POLICY_VERSION = 1
SEARCH_LIMIT = {"kind": "nodes", "value": 10_000}


def seed_for(split: Split) -> int:
    seed = 0
    while assign_split(seed) is not split:
        seed += 1
    return seed


def row(seed: int, color: str, drawback: str) -> dict[str, object]:
    is_white = color == "white"
    fen = START_FEN if is_white else AFTER_E4_FEN
    roots = WHITE_ROOTS if is_white else BLACK_ROOTS_AFTER_E4
    move = "e2e4" if is_white else "e7e5"
    request_digest = _evaluator_request_digest(
        fen=fen,
        ordinary_moves=roots,
        policy_id=POLICY_ID,
        policy_version=POLICY_VERSION,
        engine_fingerprint=ENGINE_FINGERPRINT,
        search_limit=SEARCH_LIMIT,
    )
    probability = 1.0 / len(SYMBOLIC_RULE_IDS)
    return {
        "gameId": f"{seed:08x}-000000",
        "seed": seed,
        "fenBefore": fen,
        "move": move,
        "san": "e4" if is_white else "e5",
        "moveNumber": 1,
        "ply": 0 if color == "white" else 1,
        "playerColor": color,
        "botAgentId": "random-legal",
        "botStyle": "random",
        "botStrength": 100,
        "historySan": [] if is_white else ["e4"],
        "ordinaryLegalMoves": list(roots),
        "clockMs": None,
        "symbolicFeatureVersion": SYMBOLIC_FEATURE_VERSION,
        "symbolicWhiteRuleProbabilities": [probability] * len(SYMBOLIC_RULE_IDS),
        "symbolicBlackRuleProbabilities": [probability] * len(SYMBOLIC_RULE_IDS),
        "symbolicWhiteEliminated": [False] * len(SYMBOLIC_RULE_IDS),
        "symbolicBlackEliminated": [False] * len(SYMBOLIC_RULE_IDS),
        "publicEvaluatorConstraint": {
            "provider": "uci-best-move",
            "policyId": POLICY_ID,
            "positionKey": json.dumps(
                [fen, sorted(roots)], separators=(",", ":")
            ),
            "requestDigest": request_digest,
            "bestMoveUci": move,
            "engineFingerprint": ENGINE_FINGERPRINT,
        },
        "trueDrawback": drawback,
        "hiddenParameters": {},
        "drawbackInternalState": {},
        "drawbackLegalMoves": list(roots),
        "ruleTriggered": False,
        "forced": False,
        "result": {"kind": "active"},
    }


def coverage() -> dict[str, object]:
    assigned_white = {rule_id: 0 for rule_id in SYMBOLIC_RULE_IDS}
    assigned_black = {rule_id: 0 for rule_id in SYMBOLIC_RULE_IDS}
    observed_white = {rule_id: 0 for rule_id in SYMBOLIC_RULE_IDS}
    observed_black = {rule_id: 0 for rule_id in SYMBOLIC_RULE_IDS}
    assigned_white["vegan"] = 1
    assigned_black["checkers"] = 1
    observed_white["vegan"] = 1
    observed_black["checkers"] = 1
    return {
        "assignedGames": {
            "white": assigned_white,
            "black": assigned_black,
        },
        "observedRows": {
            "white": observed_white,
            "black": observed_black,
        },
    }


def write_fixture(directory: Path) -> Path:
    split_rows: dict[str, tuple[int, bytes]] = {}
    split_names = {
        "train": Split.TRAIN,
        "validation": Split.VALIDATION,
        "test": Split.TEST,
    }
    for name, split in split_names.items():
        seed = seed_for(split)
        payload = b"".join(
            (
                json.dumps(
                    row(seed, "white", "vegan"), separators=(",", ":")
                ).encode("utf-8")
                + b"\n",
                json.dumps(
                    row(seed, "black", "checkers"), separators=(",", ":")
                ).encode("utf-8")
                + b"\n",
            )
        )
        (directory / f"{name}.ndjson").write_bytes(payload)
        split_rows[name] = (seed, payload)
    splits = {}
    for name, (seed, payload) in split_rows.items():
        outcomes = [
            {
                "seed": seed,
                "splitIndex": 0,
                "whiteRuleId": "vegan",
                "blackRuleId": "checkers",
                "whiteAgentId": "random-legal",
                "blackAgentId": "random-legal",
                "plyCount": 2,
                "finalFen": AFTER_E4_E5_FEN,
                "result": {"kind": "active"},
                "stoppedAtPlyLimit": True,
            }
        ]
        splits[name] = {
            "file": f"{name}.ndjson",
            "games": 1,
            "rowBearingGames": 1,
            "zeroPlyGames": 0,
            "oneSidedGames": 0,
            "rows": 2,
            "seeds": [seed],
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "outcomesSha256": hashlib.sha256(
                json.dumps(outcomes, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "outcomes": outcomes,
            "coverage": coverage(),
        }
    manifest = {
        "schemaVersion": 6,
        "generator": "@drawbacktrainer/simulation",
        "rootSeed": 7,
        "seedPolicy": "BLAKE2b-64(drawbacktrainer-v1:gameSeed)",
        "splitFractions": {"train": 0.8, "validation": 0.1, "test": 0.1},
        "splitSalt": "drawbacktrainer-v1",
        "workers": 1,
        "maxPlies": 2,
        "ruleIds": list(PREPARED_RULE_IDS),
        "ruleIdsSha256": hashlib.sha256(
            json.dumps(
                list(PREPARED_RULE_IDS), separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest(),
        "symbolicFeatureVersion": SYMBOLIC_FEATURE_VERSION,
        "symbolicRuleIds": list(SYMBOLIC_RULE_IDS),
        "symbolicRuleIdsSha256": hashlib.sha256(
            json.dumps(
                list(SYMBOLIC_RULE_IDS), separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest(),
        "agentIds": ["random-legal"],
        "evaluatorCoverage": "uniform-required",
        "evaluatorRequestSchemaVersion": 1,
        "evaluatorCacheSchemaVersion": 1,
        "evaluatorPolicyId": POLICY_ID,
        "evaluatorPolicyVersion": POLICY_VERSION,
        "engineFingerprint": ENGINE_FINGERPRINT,
        "engineBinarySha256": BINARY_DIGEST,
        "evaluatorSearchLimit": SEARCH_LIMIT,
        "ruleAssignmentPolicy": "seed-random-v1",
        "observationPolicy": "single-attempt-allow-partial-v1",
        "splitSizes": {"train": 1, "validation": 1, "test": 1},
        "totalGames": 3,
        "totalRows": 6,
        "splits": splits,
    }
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


class CorpusContractTests(unittest.TestCase):
    def test_matches_typescript_evaluator_canonicalization_vector(self) -> None:
        self.assertEqual(
            _normalize_fen(
                "  rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR "
                "w qKQk - 000 01 "
            ),
            START_FEN,
        )
        digest = _evaluator_request_digest(
            fen=START_FEN,
            ordinary_moves=("e2e4", "d2d4"),
            policy_id="stockfish-bestmove",
            policy_version=POLICY_VERSION,
            engine_fingerprint=f"stockfish:17.1:{BINARY_DIGEST}:{'ab' * 32}",
            search_limit=SEARCH_LIMIT,
        )
        self.assertEqual(
            digest,
            "aca9948290fe7ff478fa1035c6a06f17f79cc14ad3d4ecb5c8c5d0a5e64e9949",
        )

    def test_rejects_noncanonical_evaluator_inputs(self) -> None:
        for invalid_fen in (
            "8/8/8/8/8/8/8/K6k w - - 0",
            "8/8/8/8/8/8/8/8 w - - 0 1",
            "44/8/8/8/8/8/8/K6k w - - 0 1",
            "8/8/8/8/8/8/8/K6k x - - 0 1",
            "8/8/8/8/8/8/8/K6k w - a4 0 1",
            "8/8/8/8/8/8/8/K6k w KK - 0 1",
            "8/8/8/8/8/8/8/K6k w - - 0 0",
            "8/8/8/8/8/8/8/K6k w - - 0 9007199254740992",
        ):
            with self.assertRaises(CorpusContractError):
                _normalize_fen(invalid_fen)
        with self.assertRaisesRegex(CorpusContractError, "canonical identifier"):
            _evaluator_request_digest(
                fen=START_FEN,
                ordinary_moves=("e2e4",),
                policy_id="stockfish bestmove",
                policy_version=POLICY_VERSION,
                engine_fingerprint=ENGINE_FINGERPRINT,
                search_limit=SEARCH_LIMIT,
            )
        with self.assertRaisesRegex(CorpusContractError, "invalid UCI"):
            _evaluator_request_digest(
                fen=START_FEN,
                ordinary_moves=("e2e9",),
                policy_id=POLICY_ID,
                policy_version=POLICY_VERSION,
                engine_fingerprint=ENGINE_FINGERPRINT,
                search_limit=SEARCH_LIMIT,
            )
        with self.assertRaisesRegex(CorpusContractError, "positive safe integer"):
            _evaluator_request_digest(
                fen=START_FEN,
                ordinary_moves=("e2e4",),
                policy_id=POLICY_ID,
                policy_version=9_007_199_254_740_992,
                engine_fingerprint=ENGINE_FINGERPRINT,
                search_limit=SEARCH_LIMIT,
            )
        with self.assertRaisesRegex(CorpusContractError, "engineFingerprint"):
            _evaluator_request_digest(
                fen=START_FEN,
                ordinary_moves=("e2e4",),
                policy_id=POLICY_ID,
                policy_version=POLICY_VERSION,
                engine_fingerprint=(
                    f"stock:fish:18:{BINARY_DIGEST}:{OPTIONS_DIGEST}"
                ),
                search_limit=SEARCH_LIMIT,
            )

    def test_audits_exact_bytes_evaluator_and_empirical_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = write_fixture(Path(temporary))
            audited = audit_corpus_split(
                manifest, "train", require_complete_catalog=False
            )

            self.assertEqual(audited.rows, 2)
            self.assertEqual(audited.games, 1)
            self.assertEqual(dict(audited.white_assigned_games)["vegan"], 1)
            self.assertEqual(dict(audited.black_assigned_games)["checkers"], 1)
            self.assertEqual(dict(audited.white_observed_rows)["vegan"], 1)
            self.assertEqual(dict(audited.black_observed_rows)["checkers"], 1)
            self.assertEqual(
                audited.provenance()["dataset_sha256"],
                hashlib.sha256(audited.dataset_path.read_bytes()).hexdigest(),
            )

    def test_complete_catalog_gate_rejects_random_assignment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = write_fixture(Path(temporary))
            with self.assertRaisesRegex(
                CorpusContractError, "balanced-symmetric-v1"
            ):
                audit_corpus_split(manifest, "train")

    def test_authenticates_a_one_sided_scheduled_game(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest_path = write_fixture(directory)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            dataset_path = directory / "train.ndjson"
            payload = dataset_path.read_bytes().splitlines(keepends=True)[0]
            dataset_path.write_bytes(payload)
            entry = manifest["splits"]["train"]
            entry["rows"] = 1
            entry["bytes"] = len(payload)
            entry["sha256"] = hashlib.sha256(payload).hexdigest()
            entry["oneSidedGames"] = 1
            entry["outcomes"][0]["plyCount"] = 1
            entry["outcomes"][0]["finalFen"] = AFTER_E4_FEN
            entry["outcomesSha256"] = hashlib.sha256(
                json.dumps(
                    entry["outcomes"], separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest()
            entry["coverage"]["observedRows"]["black"]["checkers"] = 0
            manifest["totalRows"] = 5
            manifest_path.write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )

            audited = audit_corpus_split(
                manifest_path, "train", require_complete_catalog=False
            )
            self.assertEqual(audited.games, 1)
            self.assertEqual(audited.rows, 1)
            self.assertEqual(audited.one_sided_games, 1)
            self.assertEqual(audited.observed_seeds, audited.seeds)
            self.assertEqual(
                audited.game_assignments[0][1:],
                ("vegan", "checkers"),
            )

    def test_rejects_tampered_outcome_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = write_fixture(Path(temporary))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["splits"]["train"]["outcomes"][0]["plyCount"] = 1
            manifest_path.write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                CorpusContractError, "outcomes|ledger"
            ):
                audit_corpus_split(
                    manifest_path, "train", require_complete_catalog=False
                )

    def test_forged_balanced_policy_must_match_the_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = write_fixture(Path(temporary))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["ruleAssignmentPolicy"] = "balanced-symmetric-v1"
            manifest_path.write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                CorpusContractError, "balanced schedule"
            ):
                audit_corpus_split(
                    manifest_path, "train", require_complete_catalog=False
                )

    def test_rejects_mutated_split_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = write_fixture(directory)
            with (directory / "train.ndjson").open("ab") as target:
                target.write(b"\n")
            with self.assertRaisesRegex(CorpusContractError, "blank|bytes"):
                audit_corpus_split(
                    manifest, "train", require_complete_catalog=False
                )

    def test_rejects_stale_vocabulary_digest_and_search_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest_path = write_fixture(directory)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["ruleIdsSha256"] = "00" * 32
            manifest_path.write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(CorpusContractError, "ruleIdsSha256"):
                audit_corpus_split(
                    manifest_path, "train", require_complete_catalog=False
                )

            original_manifest = json.loads(
                write_fixture(directory).read_text(encoding="utf-8")
            )
            manifest["ruleIds"] = original_manifest["ruleIds"]
            manifest["ruleIdsSha256"] = original_manifest["ruleIdsSha256"]
            manifest["evaluatorSearchLimit"] = {"kind": "nodes", "value": 100}
            manifest_path.write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(CorpusContractError, "10,000 nodes"):
                audit_corpus_split(
                    manifest_path, "train", require_complete_catalog=False
                )

    def test_rejects_reordered_prepared_sampling_vocabulary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = write_fixture(Path(temporary))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            rule_ids = manifest["ruleIds"]
            rule_ids[0], rule_ids[1] = rule_ids[1], rule_ids[0]
            manifest["ruleIdsSha256"] = hashlib.sha256(
                json.dumps(rule_ids, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            manifest_path.write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(CorpusContractError, "prepared sampling"):
                audit_corpus_split(
                    manifest_path, "train", require_complete_catalog=False
                )

    def test_recomputes_public_evaluator_request_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest_path = write_fixture(directory)
            dataset_path = directory / "train.ndjson"
            rows = [
                json.loads(line)
                for line in dataset_path.read_text(encoding="utf-8").splitlines()
            ]
            rows[0]["publicEvaluatorConstraint"]["requestDigest"] = "00" * 32
            payload = (
                "\n".join(
                    json.dumps(item, separators=(",", ":")) for item in rows
                )
                + "\n"
            ).encode("utf-8")
            dataset_path.write_bytes(payload)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["splits"]["train"]["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest["splits"]["train"]["bytes"] = len(payload)
            manifest_path.write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(CorpusContractError, "request digest"):
                audit_corpus_split(
                    manifest_path, "train", require_complete_catalog=False
                )

    def test_rejects_symbolic_elimination_of_the_known_truth(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest_path = write_fixture(directory)
            dataset_path = directory / "train.ndjson"
            rows = [
                json.loads(line)
                for line in dataset_path.read_text(encoding="utf-8").splitlines()
            ]
            vegan_index = SYMBOLIC_RULE_IDS.index("vegan")
            rows[0]["symbolicWhiteEliminated"][vegan_index] = True
            payload = (
                "\n".join(
                    json.dumps(item, separators=(",", ":")) for item in rows
                )
                + "\n"
            ).encode("utf-8")
            dataset_path.write_bytes(payload)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["splits"]["train"]["sha256"] = hashlib.sha256(
                payload
            ).hexdigest()
            manifest["splits"]["train"]["bytes"] = len(payload)
            manifest_path.write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(
                CorpusContractError, "hard-eliminates.*true drawback"
            ):
                audit_corpus_split(
                    manifest_path, "train", require_complete_catalog=False
                )


if __name__ == "__main__":
    unittest.main()
