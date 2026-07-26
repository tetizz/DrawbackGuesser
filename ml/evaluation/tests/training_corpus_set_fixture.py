"""Canonical training-corpus-set fixtures for evaluation release tests."""

from __future__ import annotations

from ml.training.drawback_ml.training_corpus_set import (
    AGENT_DOMAIN,
    FROZEN_SUPPLEMENT_PROFILES,
    CorpusIdentity,
    SupplementIdentity,
    create_training_corpus_set,
)


def training_corpus_set_fixture(
    variant: int = 0,
    *,
    release_root_sha256: str | None = None,
) -> dict[str, object]:
    def digest(index: int) -> str:
        return f"{variant * 1_000 + index:064x}"

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
            "engine_fingerprint": (
                f"stockfish:18:{digest(900)}:{digest(901)}"
            ),
            "agent_domain": AGENT_DOMAIN,
        }

    primary = CorpusIdentity(
        release_root_sha256=release_root_sha256 or digest(1),
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
        for index, (offset, profile_id, rule_ids) in enumerate(
            FROZEN_SUPPLEMENT_PROFILES
        )
    ]
    return create_training_corpus_set(primary, supplements)
