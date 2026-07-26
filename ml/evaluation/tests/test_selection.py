from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from ml.evaluation.selection import (
    ContentAddressedSummary,
    FusionGridEpochScorer,
    choose_epoch,
    fusion_grid_selection_objective_metadata,
    load_selection_artifact,
    load_selection_summary,
    validate_fusion_grid_head_counts,
    write_selection_artifact,
)
from ml.evaluation.validation_partition import VALIDATION_PARTITION_IDENTITY
from ml.training.drawback_ml.rank_preserving_fusion import (
    rank_preserving_fusion,
)
from ml.training.drawback_ml.symbolic import FUSION_AWARE_LOSS_ALPHA_GRID


DIGEST_A = "a" * 64
DIGEST_B = "b" * 64


def summary_value(
    *,
    epoch: int,
    white_nll: float = 1.0,
    black_nll: float = 1.0,
    partition_name: str = "selection",
    seed_hash: str = DIGEST_A,
    model_config_hash: str = "c" * 64,
    planned_epoch_count: int = 2,
    fusion_grid: bool = False,
) -> dict[str, object]:
    value = {
        "format_version": 4 if fusion_grid else 3,
        "training_seed": 20260811,
        "epoch": epoch,
        "provenance": {
            "release_root_sha256": "d" * 64,
            "corpus_run_id": "e" * 64,
            "training_corpus_set_sha256": "2" * 64,
            "private_validation_manifest_sha256": "f" * 64,
            "validation_dataset_sha256": "1" * 64,
            "model_run_config_sha256": model_config_hash,
            "planned_epoch_count": planned_epoch_count,
        },
        "partition": {
            "identity": VALIDATION_PARTITION_IDENTITY,
            "name": partition_name,
            "seed_sha256": seed_hash,
        },
        "checkpoint": {
            "file": f"epoch-{epoch}.pt",
            "sha256": f"{epoch:064x}",
        },
        "evaluation_report": {
            "file": f"evaluation-{epoch}.json",
            "sha256": f"{epoch + 100:064x}",
        },
        "metrics": {
            "white_nll": white_nll,
            "black_nll": black_nll,
        },
    }
    if fusion_grid:
        value["objective"] = fusion_grid_selection_objective_metadata()
    return value


def write_summary(
    directory: Path,
    value: dict[str, object],
) -> ContentAddressedSummary:
    path = directory / f"summary-{value['epoch']}.json"
    payload = (
        json.dumps(value, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    path.write_bytes(payload)
    return ContentAddressedSummary(
        path=path,
        sha256=hashlib.sha256(payload).hexdigest(),
    )


class EpochSelectionTest(unittest.TestCase):
    def test_fusion_grid_counts_fail_closed(self) -> None:
        validate_fusion_grid_head_counts(
            {"observation_count": 3, "player_game_count": 2},
            "White",
        )
        invalid = (
            {"observation_count": True, "player_game_count": 1},
            {"observation_count": -1, "player_game_count": 1},
            {"observation_count": 1, "player_game_count": 2},
        )
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_fusion_grid_head_counts(value, "White")

    def test_fusion_grid_selection_defeats_move_weighted_adversary(self) -> None:
        prior = (0.5, 0.5)
        mask = (False, False)
        good = (10.0, -10.0)
        bad = (-10.0, 10.0)
        neutral = (0.0, 0.0)

        def move_loss(residuals: tuple[float, float]) -> float:
            return math.fsum(
                -math.log(
                    rank_preserving_fusion(
                        residuals,
                        prior,
                        mask,
                        alpha=alpha,
                    ).probabilities[0]
                )
                for alpha in FUSION_AWARE_LOSS_ALPHA_GRID
            ) / len(FUSION_AWARE_LOSS_ALPHA_GRID)

        adversarial = FusionGridEpochScorer()
        balanced = FusionGridEpochScorer()
        for color in ("white", "black"):
            for ply in range(1, 11):
                adversarial.add(
                    game_id=f"{color}-long",
                    color=color,
                    observed_ply=ply,
                    true_index=0,
                    residual_logits=good,
                    symbolic_prior=prior,
                    hard_eliminated=mask,
                )
                balanced.add(
                    game_id=f"{color}-long",
                    color=color,
                    observed_ply=ply,
                    true_index=0,
                    residual_logits=neutral,
                    symbolic_prior=prior,
                    hard_eliminated=mask,
                )
            adversarial.add(
                game_id=f"{color}-short",
                color=color,
                observed_ply=1,
                true_index=0,
                residual_logits=bad,
                symbolic_prior=prior,
                hard_eliminated=mask,
            )
            balanced.add(
                game_id=f"{color}-short",
                color=color,
                observed_ply=1,
                true_index=0,
                residual_logits=neutral,
                symbolic_prior=prior,
                hard_eliminated=mask,
            )

        adversarial_metrics = adversarial.report()
        balanced_metrics = balanced.report()
        move_weighted_adversarial = (
            10 * move_loss(good) + move_loss(bad)
        ) / 11
        self.assertLess(move_weighted_adversarial, move_loss(neutral))
        self.assertGreater(
            adversarial_metrics["white"].player_game_normalized_nll,
            balanced_metrics["white"].player_game_normalized_nll,
        )

        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            first = write_summary(
                directory,
                summary_value(
                    epoch=1,
                    white_nll=adversarial_metrics[
                        "white"
                    ].player_game_normalized_nll,
                    black_nll=adversarial_metrics[
                        "black"
                    ].player_game_normalized_nll,
                    fusion_grid=True,
                ),
            )
            second = write_summary(
                directory,
                summary_value(
                    epoch=2,
                    white_nll=balanced_metrics[
                        "white"
                    ].player_game_normalized_nll,
                    black_nll=balanced_metrics[
                        "black"
                    ].player_game_normalized_nll,
                    fusion_grid=True,
                ),
            )
            artifact = directory / "selection.json"
            write_selection_artifact(artifact, (first, second))
            value = json.loads(artifact.read_text(encoding="utf-8"))
            self.assertEqual(value["format_version"], 4)
            self.assertEqual(value["selected"]["epoch"], 2)

    def test_selects_minimum_mean_head_nll(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            references = (
                write_summary(
                    directory,
                    summary_value(epoch=1, white_nll=1.0, black_nll=1.0),
                ),
                write_summary(
                    directory,
                    summary_value(epoch=2, white_nll=0.7, black_nll=0.9),
                ),
            )
            selected = choose_epoch(
                tuple(load_selection_summary(item) for item in references)
            )
            self.assertEqual(selected.epoch, 2)
            self.assertEqual(selected.mean_nll, 0.8)

    def test_earlier_epoch_wins_within_absolute_point_zero_zero_five(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            earlier = load_selection_summary(
                write_summary(
                    directory,
                    summary_value(
                        epoch=1,
                        white_nll=1.004,
                        black_nll=1.004,
                    ),
                )
            )
            later = load_selection_summary(
                write_summary(
                    directory,
                    summary_value(epoch=2, white_nll=1.0, black_nll=1.0),
                )
            )
            self.assertEqual(choose_epoch((later, earlier)).epoch, 1)

    def test_later_epoch_wins_outside_tolerance(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            earlier = load_selection_summary(
                write_summary(
                    directory,
                    summary_value(
                        epoch=1,
                        white_nll=1.006,
                        black_nll=1.006,
                    ),
                )
            )
            later = load_selection_summary(
                write_summary(
                    directory,
                    summary_value(epoch=2, white_nll=1.0, black_nll=1.0),
                )
            )
            self.assertEqual(choose_epoch((earlier, later)).epoch, 2)

    def test_rejects_wrong_partition_hash_mismatch_and_nonfinite_json(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            wrong_partition = write_summary(
                directory,
                summary_value(epoch=1, partition_name="calibration-fit"),
            )
            with self.assertRaisesRegex(ValueError, "selection summaries only"):
                load_selection_summary(wrong_partition)

            valid = write_summary(directory, summary_value(epoch=2))
            valid.path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sha256"):
                load_selection_summary(valid)

            nonfinite_path = directory / "nonfinite.json"
            payload = json.dumps(
                summary_value(epoch=3), sort_keys=True
            ).replace("1.0", "NaN", 1).encode("utf-8")
            nonfinite_path.write_bytes(payload)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                load_selection_summary(
                    ContentAddressedSummary(
                        nonfinite_path,
                        hashlib.sha256(payload).hexdigest(),
                    )
                )

    def test_rejects_mixed_seed_sets_and_duplicate_epochs(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            first = load_selection_summary(
                write_summary(directory, summary_value(epoch=1))
            )
            second = load_selection_summary(
                write_summary(
                    directory,
                    summary_value(epoch=2, seed_hash=DIGEST_B),
                )
            )
            with self.assertRaisesRegex(ValueError, "seed sets"):
                choose_epoch((first, second))
            with self.assertRaisesRegex(ValueError, "duplicate epochs"):
                choose_epoch((first, first))

    def test_requires_complete_epoch_plan_and_identical_config(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            first = load_selection_summary(
                write_summary(directory, summary_value(epoch=1))
            )
            with self.assertRaisesRegex(ValueError, "every planned epoch"):
                choose_epoch((first,))
            second = load_selection_summary(
                write_summary(
                    directory,
                    summary_value(epoch=2, model_config_hash="9" * 64),
                )
            )
            with self.assertRaisesRegex(ValueError, "mixed release provenance"):
                choose_epoch((first, second))

    def test_writes_atomic_no_clobber_artifact_with_all_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            references = (
                write_summary(
                    directory,
                    summary_value(epoch=1, white_nll=0.9, black_nll=0.9),
                ),
                write_summary(
                    directory,
                    summary_value(epoch=2, white_nll=0.7, black_nll=0.7),
                ),
            )
            output = directory / "selection.json"
            self.assertEqual(
                write_selection_artifact(output, references),
                output,
            )
            artifact = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(artifact["selected"]["epoch"], 2)
            self.assertEqual(
                artifact["partition"]["identity"],
                VALIDATION_PARTITION_IDENTITY,
            )
            self.assertEqual(
                [item["summary_sha256"] for item in artifact["candidates"]],
                [item.sha256 for item in references],
            )
            self.assertEqual(
                artifact["provenance"]["training_corpus_set_sha256"],
                "2" * 64,
            )
            self.assertEqual(
                [
                    item["evaluation_report_sha256"]
                    for item in artifact["candidates"]
                ],
                [f"{epoch:064x}" for epoch in (101, 102)],
            )
            original = output.read_bytes()
            with self.assertRaisesRegex(ValueError, "overwrite"):
                write_selection_artifact(output, references)
            self.assertEqual(output.read_bytes(), original)

    def test_loader_rejects_rehashed_selected_and_metric_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            references = (
                write_summary(
                    directory,
                    summary_value(epoch=1, white_nll=0.9, black_nll=0.9),
                ),
                write_summary(
                    directory,
                    summary_value(epoch=2, white_nll=0.7, black_nll=0.7),
                ),
            )
            output = directory / "selection.json"
            write_selection_artifact(output, references)

            for mutate, expected_error in (
                (
                    lambda value: value["selected"].__setitem__(
                        "checkpoint_sha256", "9" * 64
                    ),
                    "disagrees with its candidate",
                ),
                (
                    lambda value: value["selected"].__setitem__("epoch", 1),
                    "disagrees with its candidate",
                ),
                (
                    lambda value: value["candidates"][0].__setitem__(
                        "mean_nll", 0.1
                    ),
                    "mean NLL is inconsistent",
                ),
            ):
                value = json.loads(output.read_text(encoding="utf-8"))
                mutate(value)
                payload = (
                    json.dumps(
                        value,
                        separators=(",", ":"),
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n"
                ).encode("utf-8")
                tampered = directory / f"tampered-{expected_error[:4]}.json"
                tampered.write_bytes(payload)
                with self.assertRaisesRegex(ValueError, expected_error):
                    load_selection_artifact(
                        ContentAddressedSummary(
                            tampered,
                            hashlib.sha256(payload).hexdigest(),
                        )
                    )

    def test_loader_rejects_reused_candidate_content_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            references = (
                write_summary(directory, summary_value(epoch=1)),
                write_summary(directory, summary_value(epoch=2)),
            )
            output = directory / "selection.json"
            write_selection_artifact(output, references)
            value = json.loads(output.read_text(encoding="utf-8"))
            value["candidates"][1]["evaluation_report_sha256"] = (
                value["candidates"][0]["evaluation_report_sha256"]
            )
            payload = (
                json.dumps(
                    value,
                    separators=(",", ":"),
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
            tampered = directory / "duplicate-report.json"
            tampered.write_bytes(payload)
            with self.assertRaisesRegex(ValueError, "reuse"):
                load_selection_artifact(
                    ContentAddressedSummary(
                        tampered,
                        hashlib.sha256(payload).hexdigest(),
                    )
                )

    def test_loader_requires_canonical_json_without_duplicate_keys(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            references = (
                write_summary(directory, summary_value(epoch=1)),
                write_summary(directory, summary_value(epoch=2)),
            )
            output = directory / "selection.json"
            write_selection_artifact(output, references)
            value = json.loads(output.read_text(encoding="utf-8"))
            noncanonical_payload = (
                json.dumps(value, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")
            noncanonical = directory / "noncanonical.json"
            noncanonical.write_bytes(noncanonical_payload)
            with self.assertRaisesRegex(ValueError, "canonical JSON"):
                load_selection_artifact(
                    ContentAddressedSummary(
                        noncanonical,
                        hashlib.sha256(noncanonical_payload).hexdigest(),
                    )
                )

            original = output.read_text(encoding="utf-8")
            duplicate_payload = original.replace(
                '{"candidates":',
                '{"training_seed":20260811,"candidates":',
                1,
            ).encode("utf-8")
            duplicate = directory / "duplicate.json"
            duplicate.write_bytes(duplicate_payload)
            with self.assertRaisesRegex(ValueError, "duplicate key training_seed"):
                load_selection_artifact(
                    ContentAddressedSummary(
                        duplicate,
                        hashlib.sha256(duplicate_payload).hexdigest(),
                    )
                )

    def test_rejects_legacy_summary_without_training_corpus_set(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            value = summary_value(epoch=1)
            provenance = value["provenance"]
            assert isinstance(provenance, dict)
            provenance.pop("training_corpus_set_sha256")
            reference = write_summary(directory, value)
            with self.assertRaisesRegex(
                ValueError,
                "selection provenance fields",
            ):
                load_selection_summary(reference)


if __name__ == "__main__":
    unittest.main()
