from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ml.evaluation import fusion_selection
from ml.evaluation.fusion_selection import (
    FROZEN_ALPHA_GRID,
    FUSION_SELECTION_CLASS_COUNT,
    FUSION_SELECTION_FORMAT,
    FusionSelectionError,
    FusionSelectionAccumulator,
    FusionSelectionIdentity,
    FusionSelectionObservation,
    build_fusion_selection_artifact,
    load_fusion_selection_artifact,
    write_fusion_selection_artifact,
    write_fusion_selection_accumulator,
)
from ml.evaluation.release_selection_bundle import ContentAddressedJson
from ml.evaluation.validation_partition import VALIDATION_PARTITION_IDENTITY
from ml.training.drawback_ml.rank_preserving_fusion import (
    RANK_PRESERVING_FUSION_METHOD,
)


def identity(offset: int = 0) -> FusionSelectionIdentity:
    characters = "abcdef123456"
    return FusionSelectionIdentity(
        ensemble_release_sha256=characters[offset % len(characters)] * 64,
        private_validation_manifest_sha256=(
            characters[(offset + 1) % len(characters)] * 64
        ),
        validation_dataset_sha256=(
            characters[(offset + 2) % len(characters)] * 64
        ),
        validation_seed_sha256=(
            characters[(offset + 3) % len(characters)] * 64
        ),
        training_corpus_set_sha256=(
            characters[(offset + 4) % len(characters)] * 64
        ),
        symbolic_schema_sha256=(
            characters[(offset + 5) % len(characters)] * 64
        ),
    )


def observation(
    *,
    evidence_identity: FusionSelectionIdentity,
    game_id: str,
    color: str,
    observed_ply: int,
    true_index: int = 0,
    residual_advantage: float = 0.0,
    partition: str = "selection",
    prior: tuple[float, ...] | None = None,
    mask: tuple[bool, ...] | None = None,
) -> FusionSelectionObservation:
    probabilities = prior or tuple(
        0.5 if index in {0, 1} else 0.0
        for index in range(FUSION_SELECTION_CLASS_COUNT)
    )
    residuals = tuple(
        residual_advantage if index == true_index else 0.0
        for index in range(FUSION_SELECTION_CLASS_COUNT)
    )
    return FusionSelectionObservation(
        identity=evidence_identity,
        partition=partition,
        game_id=game_id,
        color=color,
        observed_ply=observed_ply,
        true_index=true_index,
        residual_logits=residuals,
        symbolic_prior=probabilities,
        hard_eliminated=mask
        or tuple(False for _ in range(FUSION_SELECTION_CLASS_COUNT)),
    )


def complete_rows(
    evidence_identity: FusionSelectionIdentity,
) -> tuple[FusionSelectionObservation, ...]:
    return (
        observation(
            evidence_identity=evidence_identity,
            game_id="short",
            color="white",
            observed_ply=1,
            residual_advantage=2.0,
        ),
        observation(
            evidence_identity=evidence_identity,
            game_id="short",
            color="black",
            observed_ply=1,
            residual_advantage=2.0,
        ),
        observation(
            evidence_identity=evidence_identity,
            game_id="long",
            color="white",
            observed_ply=1,
            residual_advantage=-2.0,
        ),
        observation(
            evidence_identity=evidence_identity,
            game_id="long",
            color="white",
            observed_ply=2,
            residual_advantage=-2.0,
        ),
        observation(
            evidence_identity=evidence_identity,
            game_id="long",
            color="white",
            observed_ply=3,
            residual_advantage=-2.0,
        ),
        observation(
            evidence_identity=evidence_identity,
            game_id="long",
            color="black",
            observed_ply=1,
            residual_advantage=-2.0,
        ),
        observation(
            evidence_identity=evidence_identity,
            game_id="long",
            color="black",
            observed_ply=2,
            residual_advantage=-2.0,
        ),
        observation(
            evidence_identity=evidence_identity,
            game_id="long",
            color="black",
            observed_ply=3,
            residual_advantage=-2.0,
        ),
    )


def canonical(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


class FusionSelectionTest(unittest.TestCase):
    def test_prepares_each_streamed_observation_once_for_all_alphas(
        self,
    ) -> None:
        expected_identity = identity()
        rows = complete_rows(expected_identity)
        with patch(
            "ml.evaluation.fusion_selection.prepare_rank_preserving_fusion",
            wraps=fusion_selection.prepare_rank_preserving_fusion,
        ) as prepare:
            build_fusion_selection_artifact(
                expected_identity,
                iter(rows),
            )
        self.assertEqual(prepare.call_count, len(rows))

    def test_builds_frozen_content_addressed_artifact_and_round_trips(
        self,
    ) -> None:
        expected_identity = identity()
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "fusion-selection.json"
            reference = write_fusion_selection_artifact(
                output,
                expected_identity,
                complete_rows(expected_identity),
            )
            self.assertEqual(
                reference.sha256,
                hashlib.sha256(output.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                reference.sha256,
                "f13f7463fa8dda112824c4da508488e86d6de34f3eeec23db568f993e4188711",
            )
            value = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(value["format"], FUSION_SELECTION_FORMAT)
            self.assertEqual(value["version"], 1)
            self.assertEqual(value["method"], RANK_PRESERVING_FUSION_METHOD)
            self.assertEqual(value["alpha_grid"], list(FROZEN_ALPHA_GRID))
            self.assertEqual(value["identity"]["class_count"], 182)
            self.assertEqual(value["identity"]["partition"], "selection")
            self.assertEqual(
                value["identity"]["partition_identity"],
                VALIDATION_PARTITION_IDENTITY,
            )
            self.assertEqual(
                [candidate["alpha"] for candidate in value["candidates"]],
                list(FROZEN_ALPHA_GRID),
            )
            self.assertEqual(output.read_bytes(), canonical(value))

            loaded = load_fusion_selection_artifact(
                reference,
                expected_identity=expected_identity,
            )
            self.assertEqual(loaded.identity, expected_identity)
            self.assertEqual(
                tuple(candidate.alpha for candidate in loaded.candidates),
                FROZEN_ALPHA_GRID,
            )
            self.assertIn(loaded.selected_alpha, FROZEN_ALPHA_GRID)
            first = loaded.candidates[0]
            self.assertEqual(first.white.observation_count, 4)
            self.assertEqual(first.black.observation_count, 4)
            self.assertEqual(first.white.player_game_count, 2)
            self.assertEqual(first.black.player_game_count, 2)

    def test_accumulator_writer_matches_iterable_writer(self) -> None:
        expected_identity = identity()
        rows = complete_rows(expected_identity)
        accumulator = FusionSelectionAccumulator(expected_identity)
        for row in rows:
            accumulator.add(row)
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            streamed = write_fusion_selection_accumulator(
                root / "streamed.json",
                accumulator,
            )
            wrapped = write_fusion_selection_artifact(
                root / "wrapped.json",
                expected_identity,
                rows,
            )
            self.assertEqual(streamed.sha256, wrapped.sha256)
            self.assertEqual(
                streamed.path.read_bytes(),
                wrapped.path.read_bytes(),
            )

    def test_accumulator_fails_closed_after_rejected_observation(
        self,
    ) -> None:
        expected_identity = identity()
        accumulator = FusionSelectionAccumulator(expected_identity)
        invalid = replace(
            complete_rows(expected_identity)[0],
            symbolic_prior=tuple(
                0.0 for _index in range(FUSION_SELECTION_CLASS_COUNT)
            ),
        )
        with self.assertRaisesRegex(
            FusionSelectionError,
            "sum to one",
        ):
            accumulator.add(invalid)
        with self.assertRaisesRegex(
            FusionSelectionError,
            "already finalized",
        ):
            accumulator.add(complete_rows(expected_identity)[0])
        with self.assertRaisesRegex(
            FusionSelectionError,
            "already finalized",
        ):
            accumulator.finalize()

    def test_accumulator_identity_is_read_only(self) -> None:
        accumulator = FusionSelectionAccumulator(identity())
        with self.assertRaises(AttributeError):
            accumulator.identity = identity(1)  # type: ignore[misc]

    def test_player_games_are_equally_weighted_not_moves(self) -> None:
        expected_identity = identity()
        value = json.loads(
            build_fusion_selection_artifact(
                expected_identity,
                complete_rows(expected_identity),
            )
        )
        alpha_zero = value["candidates"][0]
        self.assertAlmostEqual(
            alpha_zero["white"]["game_normalized_nll"],
            math.log(2.0),
        )
        self.assertAlmostEqual(
            alpha_zero["black"]["game_normalized_nll"],
            math.log(2.0),
        )
        self.assertEqual(alpha_zero["white"]["player_game_count"], 2)
        self.assertEqual(alpha_zero["white"]["observation_count"], 4)

    def test_deterministically_selects_smaller_alpha_on_equal_metrics(
        self,
    ) -> None:
        expected_identity = identity()
        uniform_residuals = tuple(
            observation(
                evidence_identity=expected_identity,
                game_id="game",
                color=color,
                observed_ply=1,
            )
            for color in ("white", "black")
        )
        value = json.loads(
            build_fusion_selection_artifact(
                expected_identity, uniform_residuals
            )
        )
        self.assertEqual(value["selected_alpha"], 0.0)
        self.assertTrue(
            all(
                candidate["mean_white_black_game_normalized_nll"]
                == value["candidates"][0][
                    "mean_white_black_game_normalized_nll"
                ]
                for candidate in value["candidates"]
            )
        )

    def test_rejects_non_selection_and_identity_mismatch(self) -> None:
        expected_identity = identity()
        valid = complete_rows(expected_identity)
        wrong_partition = replace(valid[0], partition="test")
        with self.assertRaisesRegex(
            FusionSelectionError, "selection partition"
        ):
            build_fusion_selection_artifact(
                expected_identity, (wrong_partition, *valid[1:])
            )
        wrong_identity = replace(valid[0], identity=identity(1))
        with self.assertRaisesRegex(FusionSelectionError, "identity mismatch"):
            build_fusion_selection_artifact(
                expected_identity, (wrong_identity, *valid[1:])
            )

    def test_accepts_one_sided_games_but_requires_both_colors_globally(
        self,
    ) -> None:
        expected_identity = identity()
        one_sided_games = (
            observation(
                evidence_identity=expected_identity,
                game_id="white-only",
                color="white",
                observed_ply=0,
            ),
            observation(
                evidence_identity=expected_identity,
                game_id="complete",
                color="white",
                observed_ply=0,
            ),
            observation(
                evidence_identity=expected_identity,
                game_id="complete",
                color="black",
                observed_ply=1,
            ),
        )
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "one-sided.json"
            reference = write_fusion_selection_artifact(
                output,
                expected_identity,
                one_sided_games,
            )
            loaded = load_fusion_selection_artifact(
                reference,
                expected_identity=expected_identity,
            )
        first = loaded.candidates[0]
        self.assertEqual(first.white.player_game_count, 2)
        self.assertEqual(first.black.player_game_count, 1)

        with self.assertRaisesRegex(
            FusionSelectionError,
            "observations for both colors",
        ):
            build_fusion_selection_artifact(
                expected_identity,
                one_sided_games[:1],
            )
        with self.assertRaisesRegex(
            FusionSelectionError,
            "opening White",
        ):
            build_fusion_selection_artifact(
                expected_identity,
                (
                    observation(
                        evidence_identity=expected_identity,
                        game_id="black-only",
                        color="black",
                        observed_ply=0,
                    ),
                ),
            )

    def test_rejects_mixed_or_duplicate_player_games(self) -> None:
        expected_identity = identity()
        valid = complete_rows(expected_identity)
        mixed_truth = replace(valid[3], true_index=1)
        with self.assertRaisesRegex(FusionSelectionError, "mixed truth"):
            build_fusion_selection_artifact(
                expected_identity, (*valid[:3], mixed_truth, *valid[4:])
            )

        duplicate = replace(valid[0])
        with self.assertRaisesRegex(FusionSelectionError, "duplicate observed"):
            build_fusion_selection_artifact(
                expected_identity, (*valid, duplicate)
            )

    def test_observed_ply_is_zero_based_and_strictly_non_negative(
        self,
    ) -> None:
        expected_identity = identity()
        valid = complete_rows(expected_identity)
        zero_based = replace(valid[0], observed_ply=0)
        self.assertEqual(
            json.loads(
                build_fusion_selection_artifact(
                    expected_identity,
                    (zero_based, *valid[1:]),
                )
            )["format"],
            FUSION_SELECTION_FORMAT,
        )
        for invalid in (-1, True):
            with self.subTest(observed_ply=invalid):
                rejected = replace(valid[0], observed_ply=invalid)
                with self.assertRaisesRegex(
                    FusionSelectionError,
                    "non-negative integer",
                ):
                    build_fusion_selection_artifact(
                        expected_identity,
                        (rejected, *valid[1:]),
                    )

    def test_rejects_hard_eliminated_truth_and_invalid_probabilities(
        self,
    ) -> None:
        expected_identity = identity()
        valid = complete_rows(expected_identity)
        truth_mask = tuple(
            index == 0 for index in range(FUSION_SELECTION_CLASS_COUNT)
        )
        eliminated_truth = replace(valid[0], hard_eliminated=truth_mask)
        with self.assertRaisesRegex(FusionSelectionError, "hard-eliminated"):
            build_fusion_selection_artifact(
                expected_identity, (eliminated_truth, *valid[1:])
            )

        invalid_sum = replace(
            valid[0],
            symbolic_prior=tuple(
                0.25 if index in {0, 1} else 0.0
                for index in range(FUSION_SELECTION_CLASS_COUNT)
            ),
        )
        with self.assertRaisesRegex(FusionSelectionError, "sum to one"):
            build_fusion_selection_artifact(
                expected_identity, (invalid_sum, *valid[1:])
            )

        nonfinite = replace(
            valid[0],
            residual_logits=(
                float("nan"),
                *valid[0].residual_logits[1:],
            ),
        )
        with self.assertRaisesRegex(FusionSelectionError, "must be finite"):
            build_fusion_selection_artifact(
                expected_identity, (nonfinite, *valid[1:])
            )

    def test_discards_stale_mass_for_nontruth_eliminations(self) -> None:
        expected_identity = identity()
        valid = complete_rows(expected_identity)
        mask = tuple(
            index == 1 for index in range(FUSION_SELECTION_CLASS_COUNT)
        )
        stale = replace(valid[0], hard_eliminated=mask)

        payload = build_fusion_selection_artifact(
            expected_identity,
            (stale, *valid[1:]),
        )

        self.assertEqual(json.loads(payload)["format"], FUSION_SELECTION_FORMAT)

    def test_loader_rejects_hash_identity_method_grid_and_metric_tampering(
        self,
    ) -> None:
        expected_identity = identity()
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            output = directory / "selection.json"
            reference = write_fusion_selection_artifact(
                output,
                expected_identity,
                complete_rows(expected_identity),
            )
            output.write_bytes(output.read_bytes() + b" ")
            with self.assertRaisesRegex(FusionSelectionError, "sha256"):
                load_fusion_selection_artifact(
                    reference,
                    expected_identity=expected_identity,
                )

            original = json.loads(
                build_fusion_selection_artifact(
                    expected_identity,
                    complete_rows(expected_identity),
                )
            )
            mutations = (
                (
                    lambda value: value.__setitem__("method", "legacy"),
                    "fusion method",
                ),
                (
                    lambda value: value["alpha_grid"].__setitem__(1, 0.1),
                    "alpha grid",
                ),
                (
                    lambda value: value["identity"].__setitem__(
                        "validation_dataset_sha256", "9" * 64
                    ),
                    "identity does not match",
                ),
                (
                    lambda value: value["candidates"][0]["white"].__setitem__(
                        "observation_count", 99
                    ),
                    "counts are inconsistent",
                ),
                (
                    lambda value: value.__setitem__("selected_alpha", 0.0),
                    "selected alpha is inconsistent",
                ),
            )
            for index, (mutate, message) in enumerate(mutations):
                value = json.loads(json.dumps(original))
                mutate(value)
                payload = canonical(value)
                path = directory / f"tampered-{index}.json"
                path.write_bytes(payload)
                with self.assertRaisesRegex(FusionSelectionError, message):
                    load_fusion_selection_artifact(
                        ContentAddressedJson(
                            path,
                            hashlib.sha256(payload).hexdigest(),
                        ),
                        expected_identity=expected_identity,
                    )

    def test_loader_rejects_duplicate_nonfinite_and_noncanonical_json(
        self,
    ) -> None:
        expected_identity = identity()
        payload = build_fusion_selection_artifact(
            expected_identity, complete_rows(expected_identity)
        )
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            duplicate = payload.replace(
                b'{\n  "alpha_grid":',
                b'{\n  "format": "duplicate",\n  "alpha_grid":',
                1,
            )
            duplicate_path = directory / "duplicate.json"
            duplicate_path.write_bytes(duplicate)
            with self.assertRaisesRegex(FusionSelectionError, "duplicate key"):
                load_fusion_selection_artifact(
                    ContentAddressedJson(
                        duplicate_path,
                        hashlib.sha256(duplicate).hexdigest(),
                    ),
                    expected_identity=expected_identity,
                )

            nonfinite = payload.replace(
                b'"game_normalized_nll": ',
                b'"game_normalized_nll": NaN, "removed": ',
                1,
            )
            nonfinite_path = directory / "nonfinite.json"
            nonfinite_path.write_bytes(nonfinite)
            with self.assertRaisesRegex(FusionSelectionError, "non-finite"):
                load_fusion_selection_artifact(
                    ContentAddressedJson(
                        nonfinite_path,
                        hashlib.sha256(nonfinite).hexdigest(),
                    ),
                    expected_identity=expected_identity,
                )

            value = json.loads(payload)
            noncanonical = (
                json.dumps(value, separators=(",", ":"), sort_keys=True)
                + "\n"
            ).encode("utf-8")
            noncanonical_path = directory / "noncanonical.json"
            noncanonical_path.write_bytes(noncanonical)
            with self.assertRaisesRegex(FusionSelectionError, "not canonical"):
                load_fusion_selection_artifact(
                    ContentAddressedJson(
                        noncanonical_path,
                        hashlib.sha256(noncanonical).hexdigest(),
                    ),
                    expected_identity=expected_identity,
                )

    def test_loader_refuses_unexpected_identity_and_writer_no_clobber(
        self,
    ) -> None:
        expected_identity = identity()
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "fusion-selection.json"
            reference = write_fusion_selection_artifact(
                output,
                expected_identity,
                complete_rows(expected_identity),
            )
            with self.assertRaisesRegex(FusionSelectionError, "identity"):
                load_fusion_selection_artifact(
                    reference,
                    expected_identity=identity(1),
                )
            original = output.read_bytes()
            with self.assertRaisesRegex(FusionSelectionError, "overwrite"):
                write_fusion_selection_artifact(
                    output,
                    expected_identity,
                    complete_rows(expected_identity),
                )
            self.assertEqual(output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
