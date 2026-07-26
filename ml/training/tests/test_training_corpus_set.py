from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json
import unittest

import _bootstrap  # noqa: F401

from drawback_ml.training_corpus_set import (
    AGENT_DOMAIN,
    CorpusIdentity,
    SupplementIdentity,
    TrainingCorpusSetError,
    create_training_corpus_set,
    recompute_training_corpus_set_sha256,
    verify_training_corpus_set,
)


def digest(index: int) -> str:
    return f"{index:064x}"


def content(index: int) -> dict[str, object]:
    return {
        "outcomes_sha256": digest(index * 10 + 4),
        "dataset_sha256": digest(index * 10 + 5),
        "dataset_bytes": 1_000 + index,
        "games": 1_000 + index,
        "rows": 10_000 + index,
        "schema_version": 6,
        "symbolic_feature_version": 6,
        "max_plies": 80,
        "observation_policy": "single-attempt-allow-partial-v1",
        "evaluator_policy_id": "stockfish-bestmove-v1",
        "evaluator_policy_version": 1,
        "evaluator_nodes": 10_000,
        "engine_binary_sha256": digest(900),
        "engine_fingerprint": f"stockfish:18:{digest(900)}:{digest(901)}",
        "agent_domain": AGENT_DOMAIN,
    }


PROFILES = (
    (101, "checkers-pacman", ("checkers", "pacman")),
    (102, "truant-spice-of-life", ("truant", "spice-of-life")),
    (103, "oddball-even-keeled", ("oddball", "even-keeled")),
    (
        104,
        "quit-horsing-around-forward-march",
        ("quit-horsing-around", "forward-march"),
    ),
    (
        105,
        "horse-tranquilizer-conscientious-objectors",
        ("horse-tranquilizer", "conscientious-objectors"),
    ),
    (106, "gambler-truant", ("gambler", "truant")),
)


def identities() -> tuple[CorpusIdentity, list[SupplementIdentity]]:
    primary = CorpusIdentity(
        release_root_sha256=digest(1),
        corpus_run_id=digest(2),
        private_train_manifest_sha256=digest(3),
        **content(1),  # type: ignore[arg-type]
    )
    supplements = [
        SupplementIdentity(
            source_revision=digest(800 + index),
            generation_run_id=digest((index + 2) * 10 + 1),
            manifest_sha256=digest((index + 2) * 10 + 2),
            plan_sha256=digest((index + 2) * 10 + 3),
            **content(index + 2),  # type: ignore[arg-type]
            profile_offset=offset,
            profile_id=profile_id,
            rule_ids=rule_ids,
            profile_sha256=digest(700 + index),
        )
        for index, (offset, profile_id, rule_ids) in enumerate(PROFILES)
    ]
    return primary, supplements


class TrainingCorpusSetTests(unittest.TestCase):
    def test_builds_path_independent_canonical_identity(self) -> None:
        primary, supplements = identities()
        result = create_training_corpus_set(primary, list(reversed(supplements)))
        self.assertEqual(
            [item["profile_offset"] for item in result["supplements"]],  # type: ignore[index]
            [101, 102, 103, 104, 105, 106],
        )
        self.assertEqual(result, verify_training_corpus_set(result))
        self.assertEqual(
            result["sha256"], recompute_training_corpus_set_sha256(result)
        )
        serialized = json.dumps(result, sort_keys=True)
        self.assertNotIn("path", serialized.lower())
        self.assertNotIn("\\\\", serialized)
        primary_wire = result["primary"]  # type: ignore[assignment]
        self.assertNotIn("source_revision", primary_wire)
        self.assertNotIn("plan_sha256", primary_wire)
        self.assertNotIn("generation_run_id", primary_wire)

    def test_input_identities_are_frozen(self) -> None:
        primary, _ = identities()
        with self.assertRaises(FrozenInstanceError):
            primary.rows = 1  # type: ignore[misc]

    def test_hash_binds_every_nested_identity(self) -> None:
        primary, supplements = identities()
        result = create_training_corpus_set(primary, supplements)
        tampered = json.loads(json.dumps(result))
        tampered["supplements"][0]["profile_sha256"] = digest(999)
        with self.assertRaisesRegex(TrainingCorpusSetError, "does not match"):
            verify_training_corpus_set(tampered)

    def test_primary_identity_binds_release_safe_roots(self) -> None:
        primary, supplements = identities()
        baseline = create_training_corpus_set(primary, supplements)["sha256"]
        for field in (
            "release_root_sha256",
            "corpus_run_id",
            "private_train_manifest_sha256",
        ):
            changed = replace(primary, **{field: digest(990 + len(field))})
            self.assertNotEqual(
                baseline,
                create_training_corpus_set(changed, supplements)["sha256"],
            )

    def test_primary_cannot_represent_held_out_or_monolithic_artifacts(self) -> None:
        primary, supplements = identities()
        result = create_training_corpus_set(primary, supplements)
        forbidden = (
            "source_revision",
            "generation_run_id",
            "manifest_sha256",
            "plan_sha256",
            "private_validation_manifest_sha256",
            "private_test_manifest_sha256",
            "validation_dataset_sha256",
            "test_dataset_sha256",
            "held_out_seeds",
        )
        for field in forbidden:
            tampered = json.loads(json.dumps(result))
            tampered["primary"][field] = digest(998)
            with self.assertRaisesRegex(TrainingCorpusSetError, "unknown"):
                verify_training_corpus_set(tampered)

    def test_primary_and_supplement_wire_shapes_are_strict(self) -> None:
        primary, supplements = identities()
        result = create_training_corpus_set(primary, supplements)

        missing_primary = json.loads(json.dumps(result))
        del missing_primary["primary"]["private_train_manifest_sha256"]
        with self.assertRaisesRegex(
            TrainingCorpusSetError,
            "missing private_train_manifest_sha256",
        ):
            verify_training_corpus_set(missing_primary)

        primary_field_on_supplement = json.loads(json.dumps(result))
        primary_field_on_supplement["supplements"][0][
            "release_root_sha256"
        ] = digest(997)
        with self.assertRaisesRegex(
            TrainingCorpusSetError,
            "unknown release_root_sha256",
        ):
            verify_training_corpus_set(primary_field_on_supplement)

        missing_supplement = json.loads(json.dumps(result))
        del missing_supplement["supplements"][0]["source_revision"]
        with self.assertRaisesRegex(
            TrainingCorpusSetError,
            "missing source_revision",
        ):
            verify_training_corpus_set(missing_supplement)

    def test_rejects_unknown_fields_and_paths(self) -> None:
        primary, supplements = identities()
        result = create_training_corpus_set(primary, supplements)
        for target, field in (
            (result, "output_path"),
            (result["primary"], "manifest_path"),  # type: ignore[index]
            (result["supplements"][0], "dataset_file"),  # type: ignore[index]
        ):
            tampered = json.loads(json.dumps(result))
            if target is result:
                tampered[field] = "C:/secret"
            elif target is result["primary"]:  # type: ignore[index]
                tampered["primary"][field] = "C:/secret"
            else:
                tampered["supplements"][0][field] = "C:/secret"
            with self.assertRaisesRegex(TrainingCorpusSetError, "unknown"):
                verify_training_corpus_set(tampered)

    def test_requires_exact_profiles_once_each(self) -> None:
        primary, supplements = identities()
        with self.assertRaisesRegex(TrainingCorpusSetError, "exactly six"):
            create_training_corpus_set(primary, supplements[:-1])
        duplicate = [*supplements[:-1], supplements[0]]
        with self.assertRaisesRegex(TrainingCorpusSetError, "offsets must be unique"):
            create_training_corpus_set(primary, duplicate)
        wrong_pair = replace(supplements[0], rule_ids=("checkers", "truant"))
        with self.assertRaisesRegex(TrainingCorpusSetError, "frozen profile"):
            create_training_corpus_set(primary, [wrong_pair, *supplements[1:]])

    def test_rejects_duplicate_content_identities(self) -> None:
        primary, supplements = identities()
        duplicate_dataset = replace(
            supplements[0], dataset_sha256=primary.dataset_sha256
        )
        with self.assertRaisesRegex(
            TrainingCorpusSetError, "dataset_sha256 values must be unique"
        ):
            create_training_corpus_set(
                primary, [duplicate_dataset, *supplements[1:]]
            )

    def test_rejects_compatibility_drift_and_nonpositive_counts(self) -> None:
        primary, supplements = identities()
        drifted = replace(supplements[0], engine_fingerprint="stockfish:other")
        with self.assertRaisesRegex(
            TrainingCorpusSetError, "engine_fingerprint does not match primary"
        ):
            create_training_corpus_set(primary, [drifted, *supplements[1:]])
        different_source = replace(supplements[0], source_revision=digest(899))
        different_source_set = create_training_corpus_set(
            primary, [different_source, *supplements[1:]]
        )
        self.assertNotEqual(
            different_source_set["sha256"],
            create_training_corpus_set(primary, supplements)["sha256"],
        )
        empty = replace(supplements[0], rows=0)
        with self.assertRaisesRegex(
            TrainingCorpusSetError, "positive interoperable integer"
        ):
            create_training_corpus_set(primary, [empty, *supplements[1:]])
        wrong_schema = replace(primary, schema_version=5)
        with self.assertRaisesRegex(TrainingCorpusSetError, "frozen value"):
            create_training_corpus_set(wrong_schema, supplements)

    def test_strict_verifier_rejects_noncanonical_order_and_bad_outer_shape(
        self,
    ) -> None:
        primary, supplements = identities()
        result = create_training_corpus_set(primary, supplements)
        reordered = json.loads(json.dumps(result))
        reordered["supplements"].reverse()
        with self.assertRaisesRegex(TrainingCorpusSetError, "canonical profile order"):
            recompute_training_corpus_set_sha256(reordered)
        extra = json.loads(json.dumps(result))
        extra["model"] = "v21"
        with self.assertRaisesRegex(TrainingCorpusSetError, "unknown model"):
            verify_training_corpus_set(extra)

    def test_rejects_mutable_or_noncanonical_wire_shapes(self) -> None:
        primary, supplements = identities()
        result = create_training_corpus_set(primary, supplements)
        bad_agents = json.loads(json.dumps(result))
        bad_agents["primary"]["agent_domain"].reverse()
        with self.assertRaisesRegex(TrainingCorpusSetError, "frozen release domain"):
            verify_training_corpus_set(bad_agents)
        tuple_wire = dict(result)
        tuple_wire["supplements"] = tuple(result["supplements"])  # type: ignore[arg-type]
        with self.assertRaisesRegex(TrainingCorpusSetError, "must be an array"):
            verify_training_corpus_set(tuple_wire)
        path_fingerprint = replace(primary, engine_fingerprint="C:\\stockfish.exe")
        with self.assertRaisesRegex(TrainingCorpusSetError, "filesystem path"):
            create_training_corpus_set(path_fingerprint, supplements)


if __name__ == "__main__":
    unittest.main()
