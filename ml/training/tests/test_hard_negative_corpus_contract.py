from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import _bootstrap  # noqa: F401

from drawback_ml.corpus_contract import (
    CorpusContractError,
    HARD_NEGATIVE_AGENT_IDS,
    HARD_NEGATIVE_PROFILES,
    _canonical_json,
    _expected_hard_negative_slots,
    _expected_hard_negative_train_seeds,
    _hard_negative_plan_shards,
    _hard_negative_slot_material,
    open_audited_hard_negative_train_corpus,
)
from drawback_ml.splits import Split, assign_split
from drawback_ml.symbolic_schema import (
    SYMBOLIC_FEATURE_VERSION,
    SYMBOLIC_RULE_IDS,
)
from test_corpus_contract import (
    BINARY_DIGEST,
    ENGINE_FINGERPRINT,
    POLICY_ID,
    POLICY_VERSION,
    SEARCH_LIMIT,
    AFTER_E4_E5_FEN,
    START_FEN,
    row,
)


PROFILE_ID = "checkers-pacman"
SOURCE_REVISION = "ab" * 20
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def exact_json(value: object) -> bytes:
    return (json.dumps(value, indent=2) + "\n").encode("utf-8")


def derive_game_seed(batch_seed: int, game_index: int) -> int:
    def imul(left: int, right: int) -> int:
        return ((left & 0xFFFF_FFFF) * (right & 0xFFFF_FFFF)) & 0xFFFF_FFFF

    value = (batch_seed ^ imul(game_index + 1, 0x9E37_79B9)) & 0xFFFF_FFFF
    value ^= value >> 16
    value = imul(value, 0x21F0_AAAD)
    value ^= value >> 15
    value = imul(value, 0x735A_2D97)
    return (value ^ (value >> 15)) & 0xFFFF_FFFF


def training_seeds(root_seed: int) -> list[int]:
    result: list[int] = []
    index = 0
    while len(result) < 1_000:
        seed = derive_game_seed(root_seed, index)
        if assign_split(seed) is Split.TRAIN:
            result.append(seed)
        index += 1
    return result


def digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def write_fixture(directory: Path) -> tuple[Path, Path, Path]:
    description, rule_ids, evidence, root_seed = HARD_NEGATIVE_PROFILES[PROFILE_ID]
    seeds = training_seeds(root_seed)
    slots = _expected_hard_negative_slots(root_seed, tuple(seeds), rule_ids)
    slot_material = [
        _hard_negative_slot_material("train", index, seed, slots[index])
        for index, seed in enumerate(seeds)
    ]

    white = row(seeds[0], "white", slots[0].white_rule_id)
    white["gameId"] = f"{seeds[0]:08x}-000000"
    white["botAgentId"] = slots[0].white_agent_id
    black = row(seeds[0], "black", slots[0].black_rule_id)
    black["gameId"] = f"{seeds[0]:08x}-000000"
    black["botAgentId"] = slots[0].black_agent_id
    dataset_payload = (
        _canonical_json(white) + "\n" + _canonical_json(black) + "\n"
    ).encode("utf-8")
    dataset_path = directory / "train.ndjson"
    dataset_path.write_bytes(dataset_payload)

    outcomes: list[dict[str, object]] = []
    assigned_white = {rule_id: 0 for rule_id in rule_ids}
    assigned_black = {rule_id: 0 for rule_id in rule_ids}
    observed_white = {rule_id: 0 for rule_id in rule_ids}
    observed_black = {rule_id: 0 for rule_id in rule_ids}
    observed_white[slots[0].white_rule_id] += 1
    observed_black[slots[0].black_rule_id] += 1
    for index, (seed, slot) in enumerate(zip(seeds, slots, strict=True)):
        assigned_white[slot.white_rule_id] += 1
        assigned_black[slot.black_rule_id] += 1
        first = index == 0
        outcomes.append(
            {
                "seed": seed,
                "splitIndex": index,
                "whiteRuleId": slot.white_rule_id,
                "blackRuleId": slot.black_rule_id,
                "whiteAgentId": slot.white_agent_id,
                "blackAgentId": slot.black_agent_id,
                "plyCount": 2 if first else 0,
                "finalFen": AFTER_E4_E5_FEN if first else START_FEN,
                "result": (
                    {"kind": "active"}
                    if first
                    else {
                        "kind": "drawback-loss",
                        "loss": {
                            "ruleId": slot.white_rule_id,
                            "color": "white",
                            "reason": "fixture",
                        },
                    }
                ),
                "stoppedAtPlyLimit": first,
            }
        )
    profile = {
        "id": PROFILE_ID,
        "description": description,
        "ruleIds": list(rule_ids),
        "evidence": evidence,
    }
    split_sizes = {"train": 1_000, "validation": 0, "test": 0}
    metadata = {
        "schemaVersion": 6,
        "generator": "@drawbacktrainer/simulation",
        "rootSeed": root_seed,
        "seedPolicy": "BLAKE2b-64(drawbacktrainer-v1:gameSeed)",
        "splitFractions": {"train": 0.8, "validation": 0.1, "test": 0.1},
        "splitSalt": "drawbacktrainer-v1",
        "maxPlies": 80,
        "ruleIds": list(rule_ids),
        "symbolicFeatureVersion": SYMBOLIC_FEATURE_VERSION,
        "symbolicRuleIds": list(SYMBOLIC_RULE_IDS),
        "agentIds": list(HARD_NEGATIVE_AGENT_IDS),
        "evaluatorCoverage": "uniform-required",
        "evaluatorRequestSchemaVersion": 1,
        "evaluatorCacheSchemaVersion": 1,
        "evaluatorPolicyId": POLICY_ID,
        "evaluatorPolicyVersion": POLICY_VERSION,
        "engineFingerprint": ENGINE_FINGERPRINT,
        "engineBinarySha256": BINARY_DIGEST,
        "evaluatorSearchLimit": SEARCH_LIMIT,
        "ruleAssignmentPolicy": "balanced-symmetric-v1",
        "observationPolicy": "single-attempt-allow-partial-v1",
        "splitSizes": split_sizes,
        "totalGames": 1_000,
        "hardNegativeProfile": profile,
    }
    workers = 8
    split_seeds = {"train": seeds, "validation": [], "test": []}
    schedule = {
        "policyId": "balanced-symmetric-v1",
        "splits": {"train": slot_material, "validation": [], "test": []},
    }
    config_material = {
        "sourceRevision": SOURCE_REVISION,
        "schemaVersion": metadata["schemaVersion"],
        "generator": metadata["generator"],
        "rootSeed": metadata["rootSeed"],
        "seedPolicy": metadata["seedPolicy"],
        "splitFractions": metadata["splitFractions"],
        "splitSalt": metadata["splitSalt"],
        "maxPlies": metadata["maxPlies"],
        "ruleIds": metadata["ruleIds"],
        "symbolicFeatureVersion": metadata["symbolicFeatureVersion"],
        "symbolicRuleIds": metadata["symbolicRuleIds"],
        "agentIds": metadata["agentIds"],
        "evaluatorCoverage": metadata["evaluatorCoverage"],
        "evaluatorRequestSchemaVersion": metadata[
            "evaluatorRequestSchemaVersion"
        ],
        "evaluatorCacheSchemaVersion": metadata["evaluatorCacheSchemaVersion"],
        "evaluatorPolicyId": metadata["evaluatorPolicyId"],
        "evaluatorPolicyVersion": metadata["evaluatorPolicyVersion"],
        "engineFingerprint": metadata["engineFingerprint"],
        "engineBinarySha256": metadata["engineBinarySha256"],
        "evaluatorSearchLimit": metadata["evaluatorSearchLimit"],
        "ruleAssignmentPolicy": metadata["ruleAssignmentPolicy"],
        "observationPolicy": metadata["observationPolicy"],
        "workers": workers,
        "splitSizes": metadata["splitSizes"],
        "totalGames": metadata["totalGames"],
        "hardNegativeProfile": profile,
        "splitSeeds": split_seeds,
    }
    corpus_config_sha256 = digest(config_material)
    schedule_sha256 = digest(
        {"policyId": "balanced-symmetric-v1", "slots": slot_material}
    )
    shards = _hard_negative_plan_shards(250, 1_000)
    for shard in shards:
        start = int(shard["splitStart"])
        end = int(shard["splitEnd"])
        shard["seedAssignmentSha256"] = digest(slot_material[start:end])
    run_material = {
        "schemaVersion": 1,
        "corpusConfigSha256": corpus_config_sha256,
        "scheduleSha256": schedule_sha256,
        "ruleIds": list(rule_ids),
        "agentIds": list(HARD_NEGATIVE_AGENT_IDS),
        "shardSize": 250,
        "totalGames": 1_000,
        "shards": shards,
    }
    run_id = digest(run_material)
    plan = {
        "schemaVersion": 1,
        "runPlan": {**run_material, "runId": run_id},
        "sourceRevision": SOURCE_REVISION,
        "metadata": metadata,
        "splitSeeds": split_seeds,
        "schedule": schedule,
    }
    plan_path = directory / "plan.json"
    plan_payload = exact_json(plan)
    plan_path.write_bytes(plan_payload)

    zero = {rule_id: 0 for rule_id in rule_ids}
    empty_split = {
        "file": None,
        "games": 0,
        "rowBearingGames": 0,
        "zeroPlyGames": 0,
        "oneSidedGames": 0,
        "rows": 0,
        "seeds": [],
        "bytes": 0,
        "sha256": EMPTY_SHA256,
        "outcomesSha256": digest([]),
        "outcomes": [],
        "coverage": {
            "assignedGames": {"white": zero, "black": zero},
            "observedRows": {"white": zero, "black": zero},
        },
    }
    manifest = {
        **metadata,
        "workers": workers,
        "totalRows": 2,
        "ruleIdsSha256": digest(list(rule_ids)),
        "symbolicRuleIdsSha256": digest(list(SYMBOLIC_RULE_IDS)),
        "hardNegativeGeneration": {
            "version": 1,
            "sourceRevision": SOURCE_REVISION,
            "runId": run_id,
            "corpusConfigSha256": corpus_config_sha256,
            "planSha256": hashlib.sha256(plan_payload).hexdigest(),
        },
        "splits": {
            "train": {
                "file": "train.ndjson",
                "games": 1_000,
                "rowBearingGames": 1,
                "zeroPlyGames": 999,
                "oneSidedGames": 0,
                "rows": 2,
                "seeds": seeds,
                "bytes": len(dataset_payload),
                "sha256": hashlib.sha256(dataset_payload).hexdigest(),
                "outcomesSha256": digest(outcomes),
                "outcomes": outcomes,
                "coverage": {
                    "assignedGames": {
                        "white": assigned_white,
                        "black": assigned_black,
                    },
                    "observedRows": {
                        "white": observed_white,
                        "black": observed_black,
                    },
                },
            },
            "validation": empty_split,
            "test": empty_split,
        },
    }
    manifest_path = directory / "manifest.json"
    manifest_path.write_bytes(exact_json(manifest))
    return manifest_path, dataset_path, plan_path


class HardNegativeCorpusContractTests(unittest.TestCase):
    def test_matches_typescript_repeated_cycle_vector(self) -> None:
        seeds = _expected_hard_negative_train_seeds(20260911)
        self.assertEqual(
            seeds[:5],
            (907543878, 2076016098, 3288708667, 108531045, 3141817514),
        )
        slots = _expected_hard_negative_slots(
            20260911, seeds, ("checkers", "pacman")
        )
        self.assertEqual(
            [
                (
                    slot.white_rule_id,
                    slot.black_rule_id,
                    slot.white_agent_id,
                    slot.black_agent_id,
                )
                for slot in slots[:6]
            ],
            [
                (
                    "pacman",
                    "checkers",
                    "human-like-strong",
                    "human-like-medium",
                ),
                ("checkers", "pacman", "random-legal", "random-legal"),
                ("pacman", "checkers", "random-legal", "greedy-material"),
                (
                    "checkers",
                    "pacman",
                    "greedy-material",
                    "human-like-medium",
                ),
                (
                    "pacman",
                    "checkers",
                    "greedy-material",
                    "human-like-strong",
                ),
                (
                    "checkers",
                    "pacman",
                    "human-like-weak",
                    "greedy-material",
                ),
            ],
        )

    def test_authenticates_exact_profile_plan_and_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = write_fixture(Path(temporary))
            with open_audited_hard_negative_train_corpus(
                *paths, PROFILE_ID
            ) as lease:
                self.assertEqual(lease.audited.profile_id, PROFILE_ID)
                self.assertEqual(lease.audited.rule_ids, ("checkers", "pacman"))
                self.assertEqual(lease.audited.games, 1_000)
                self.assertEqual(lease.audited.rows, 2)
                self.assertEqual(len(lease.audited.observed_seeds), 1)
                self.assertEqual(len(lease.audited.game_assignments), 1_000)
                self.assertEqual(lease.audited.row_bearing_games, 1)
                self.assertEqual(lease.audited.max_plies, 80)
                self.assertEqual(
                    lease.audited.agent_ids, HARD_NEGATIVE_AGENT_IDS
                )
                self.assertEqual(
                    lease.audited.engine_binary_sha256, BINARY_DIGEST
                )
                self.assertEqual(
                    lease.audited.evaluator_nodes, 10_000
                )
                lease.verify_unchanged(chunk_size=17)

    def test_rejects_wrong_profile_and_nonempty_heldout_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = write_fixture(Path(temporary))
            with self.assertRaisesRegex(CorpusContractError, "named profile"):
                with open_audited_hard_negative_train_corpus(
                    *paths, "gambler-truant"
                ):
                    pass

            manifest = json.loads(paths[0].read_text(encoding="utf-8"))
            manifest["splits"]["validation"]["file"] = "validation.ndjson"
            paths[0].write_bytes(exact_json(manifest))
            with self.assertRaisesRegex(CorpusContractError, "canonical empty"):
                with open_audited_hard_negative_train_corpus(
                    *paths, PROFILE_ID
                ):
                    pass

    def test_rejects_plan_and_generation_identity_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = write_fixture(Path(temporary))
            plan = json.loads(paths[2].read_text(encoding="utf-8"))
            plan["runPlan"]["runId"] = "ff" * 32
            plan_payload = exact_json(plan)
            paths[2].write_bytes(plan_payload)
            manifest = json.loads(paths[0].read_text(encoding="utf-8"))
            manifest["hardNegativeGeneration"]["planSha256"] = hashlib.sha256(
                plan_payload
            ).hexdigest()
            manifest["hardNegativeGeneration"]["runId"] = "ff" * 32
            paths[0].write_bytes(exact_json(manifest))
            with self.assertRaisesRegex(CorpusContractError, "run identity"):
                with open_audited_hard_negative_train_corpus(
                    *paths, PROFILE_ID
                ):
                    pass

    def test_rejects_coverage_outside_pair_and_row_label_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = write_fixture(Path(temporary))
            manifest = json.loads(paths[0].read_text(encoding="utf-8"))
            manifest["splits"]["train"]["coverage"]["observedRows"]["white"][
                "vegan"
            ] = 0
            paths[0].write_bytes(exact_json(manifest))
            with self.assertRaisesRegex(CorpusContractError, "coverage keys"):
                with open_audited_hard_negative_train_corpus(
                    *paths, PROFILE_ID
                ):
                    pass

        with tempfile.TemporaryDirectory() as temporary:
            paths = write_fixture(Path(temporary))
            rows = [
                json.loads(line)
                for line in paths[1].read_text(encoding="utf-8").splitlines()
            ]
            rows[0]["trueDrawback"] = "vegan"
            payload = (
                "\n".join(_canonical_json(item) for item in rows) + "\n"
            ).encode("utf-8")
            paths[1].write_bytes(payload)
            manifest = json.loads(paths[0].read_text(encoding="utf-8"))
            manifest["splits"]["train"]["bytes"] = len(payload)
            manifest["splits"]["train"]["sha256"] = hashlib.sha256(payload).hexdigest()
            paths[0].write_bytes(exact_json(manifest))
            with self.assertRaisesRegex(CorpusContractError, "scheduled assignment"):
                with open_audited_hard_negative_train_corpus(
                    *paths, PROFILE_ID
                ):
                    pass

    def test_rejects_duplicate_json_keys_and_nonfinite_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = write_fixture(Path(temporary))
            payload = paths[0].read_bytes().replace(
                b'{\n  "schemaVersion": 6,',
                b'{\n  "schemaVersion": 6,\n  "schemaVersion": 6,',
                1,
            )
            paths[0].write_bytes(payload)
            with self.assertRaisesRegex(CorpusContractError, "duplicate JSON"):
                with open_audited_hard_negative_train_corpus(
                    *paths, PROFILE_ID
                ):
                    pass

        with tempfile.TemporaryDirectory() as temporary:
            paths = write_fixture(Path(temporary))
            payload = paths[1].read_bytes().replace(
                b'"botStrength":100',
                b'"botStrength":NaN',
                1,
            )
            paths[1].write_bytes(payload)
            manifest = json.loads(paths[0].read_text(encoding="utf-8"))
            manifest["splits"]["train"]["bytes"] = len(payload)
            manifest["splits"]["train"]["sha256"] = hashlib.sha256(payload).hexdigest()
            paths[0].write_bytes(exact_json(manifest))
            with self.assertRaisesRegex(CorpusContractError, "non-finite"):
                with open_audited_hard_negative_train_corpus(
                    *paths, PROFILE_ID
                ):
                    pass

    def test_detects_pinned_dataset_mutation_before_release(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = write_fixture(Path(temporary))
            with self.assertRaisesRegex(CorpusContractError, "changed"):
                with open_audited_hard_negative_train_corpus(
                    *paths, PROFILE_ID
                ) as lease:
                    with paths[1].open("ab") as target:
                        target.write(b"\n")
                    lease.verify_unchanged()

    def test_detects_pinned_manifest_and_plan_mutation(self) -> None:
        for path_index, label in ((0, "manifest"), (2, "plan")):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                paths = write_fixture(Path(temporary))
                with self.assertRaisesRegex(CorpusContractError, label):
                    with open_audited_hard_negative_train_corpus(
                        *paths, PROFILE_ID
                    ) as lease:
                        with paths[path_index].open("ab") as target:
                            target.write(b" ")
                        lease.verify_unchanged()


if __name__ == "__main__":
    unittest.main()
