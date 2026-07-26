from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from ml.evaluation.release_selection_bundle import (
    ContentAddressedJson,
    load_training_run,
    verify_checkpoint_training_identity,
    verify_release_selection_bundle,
)
from ml.evaluation.selection import (
    ContentAddressedSummary,
    fusion_grid_selection_objective_metadata,
    write_selection_artifact,
)
from ml.evaluation.validation_partition import VALIDATION_PARTITION_IDENTITY
from ml.evaluation.tests.training_corpus_set_fixture import (
    training_corpus_set_fixture,
)
from ml.evaluation.tests.checkpoint_fixture import (
    write_fusion_checkpoint,
    write_legacy_checkpoint,
)
from ml.training.drawback_ml.checkpoint import (
    FUSION_GRID_DRAWBACK_OBJECTIVE,
    LEGACY_ADDITIVE_DRAWBACK_OBJECTIVE,
)
from ml.training.drawback_ml.symbolic import (
    FUSION_AWARE_LOSS_ALPHA_GRID,
    FUSION_AWARE_LOSS_METHOD,
    FUSION_AWARE_LOSS_VERSION,
)


def canonical(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write(path: Path, value: object) -> str:
    payload = canonical(value)
    path.write_bytes(payload)
    return sha(payload)


def build_bundle(
    directory: Path,
    *,
    separate_evidence_directory: bool = False,
    fusion_grid: bool = False,
) -> tuple[ContentAddressedJson, ContentAddressedJson]:
    training_directory = (
        directory / "training" if separate_evidence_directory else directory
    )
    evidence_directory = (
        directory / "frozen" if separate_evidence_directory else directory
    )
    training_directory.mkdir(parents=True, exist_ok=True)
    evidence_directory.mkdir(parents=True, exist_ok=True)
    corpus_set = training_corpus_set_fixture()
    run_config = {
        "seed": 20260811,
        "epochs": 2,
        "model_variant": "v21-hybrid",
        "corpus_provenance": {
            "training_corpus_set": corpus_set,
            "training_corpus_set_sha256": corpus_set["sha256"],
        },
    }
    if fusion_grid:
        run_config.update(
            {
                "fusion_aware_loss_method": FUSION_AWARE_LOSS_METHOD,
                "fusion_aware_loss_version": FUSION_AWARE_LOSS_VERSION,
                "fusion_aware_loss_alpha_grid": list(
                    FUSION_AWARE_LOSS_ALPHA_GRID
                ),
            }
        )
    run_material = {
        "format": "drawbacktrainer-streaming-run",
        "version": 1,
        "config": run_config,
        "runtime": {"device": "cpu"},
        "sampling": {"policy": "game-balanced-v1"},
    }
    run_id = sha(
        json.dumps(
            run_material, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    )
    run_value = {"run_id": run_id, **run_material}
    run_path = training_directory / "run.claim.json"
    run_sha = write(run_path, run_value)
    provenance = {
        "release_root_sha256": "b" * 64,
        "corpus_run_id": "c" * 64,
        "training_corpus_set_sha256": corpus_set["sha256"],
        "private_validation_manifest_sha256": "d" * 64,
        "validation_dataset_sha256": "e" * 64,
        "model_run_config_sha256": run_sha,
        "planned_epoch_count": 2,
    }
    references: list[ContentAddressedSummary] = []
    for epoch, nll in ((1, 0.9), (2, 0.7)):
        checkpoint_path = training_directory / f"epoch-{epoch}.pt"
        checkpoint_writer = (
            write_fusion_checkpoint
            if fusion_grid
            else write_legacy_checkpoint
        )
        checkpoint_sha = checkpoint_writer(
            checkpoint_path,
            seed=20260811,
            epoch=epoch,
            run_id=run_id,
            training_corpus_set=corpus_set,
        )
        report_path = evidence_directory / f"report-{epoch}.json"
        report_value: dict[str, object] = {
            "formatVersion": 2 if fusion_grid else 1,
            "evaluation": {
                "validationPartition": {
                    "identity": VALIDATION_PARTITION_IDENTITY,
                    "name": "selection",
                    "seedSha256": "f" * 64,
                }
            },
            "provenance": {
                "release_root_sha256": provenance["release_root_sha256"],
                "corpus_run_id": provenance["corpus_run_id"],
                "manifest_sha256": provenance[
                    "private_validation_manifest_sha256"
                ],
                "dataset_sha256": provenance["validation_dataset_sha256"],
                "checkpoint_file": checkpoint_path.name,
                "checkpoint_sha256": checkpoint_sha,
                "checkpoint_seed": 20260811,
                "checkpoint_epoch": epoch,
                "training_run_id": run_id,
            },
            "metrics": {
                "white_drawback": {"negative_log_likelihood": nll},
                "black_drawback": {"negative_log_likelihood": nll},
            },
        }
        if fusion_grid:
            report_value["provenance"][
                "drawback_loss_objective"
            ] = FUSION_GRID_DRAWBACK_OBJECTIVE
            report_value["epochSelectionObjective"] = {
                "identity": fusion_grid_selection_objective_metadata(),
                "metrics": {
                    "white": {
                        "observation_count": 10,
                        "player_game_count": 2,
                        "player_game_normalized_nll": nll,
                    },
                    "black": {
                        "observation_count": 9,
                        "player_game_count": 2,
                        "player_game_normalized_nll": nll,
                    },
                },
            }
        report_sha = write(report_path, report_value)
        summary_path = evidence_directory / f"summary-{epoch}.json"
        summary_value: dict[str, object] = {
            "format_version": 4 if fusion_grid else 3,
            "training_seed": 20260811,
            "epoch": epoch,
            "provenance": provenance,
            "partition": {
                "identity": VALIDATION_PARTITION_IDENTITY,
                "name": "selection",
                "seed_sha256": "f" * 64,
            },
            "checkpoint": {
                "file": checkpoint_path.name,
                "sha256": checkpoint_sha,
            },
            "evaluation_report": {
                "file": report_path.name,
                "sha256": report_sha,
            },
            "metrics": {"white_nll": nll, "black_nll": nll},
        }
        if fusion_grid:
            summary_value["objective"] = (
                fusion_grid_selection_objective_metadata()
            )
        summary_sha = write(summary_path, summary_value)
        references.append(ContentAddressedSummary(summary_path, summary_sha))
    artifact_path = evidence_directory / "selection.json"
    write_selection_artifact(artifact_path, references)
    return (
        ContentAddressedJson(artifact_path, sha(artifact_path.read_bytes())),
        ContentAddressedJson(run_path, run_sha),
    )


class ReleaseSelectionBundleTest(unittest.TestCase):
    def test_verifies_complete_fusion_grid_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            artifact, run = build_bundle(Path(raw), fusion_grid=True)
            verified = verify_release_selection_bundle(artifact, run)
            self.assertEqual(
                verified.artifact.objective_id,
                FUSION_GRID_DRAWBACK_OBJECTIVE,
            )
            self.assertEqual(verified.candidate_count, 2)

    def test_checkpoint_identity_binds_run_seed_epoch_objective_and_corpus(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            _artifact, run_reference = build_bundle(Path(raw))
            training_run = load_training_run(run_reference)
            corpus_set = training_corpus_set_fixture()
            values = {
                "training_run_id": training_run.run_id,
                "checkpoint_seed": training_run.seed,
                "checkpoint_epoch": 1,
                "drawback_loss_objective": (
                    LEGACY_ADDITIVE_DRAWBACK_OBJECTIVE
                ),
                "corpus_provenance": {
                    "training_corpus_set": corpus_set,
                    "training_corpus_set_sha256": corpus_set["sha256"],
                },
            }

            def verify(candidate: SimpleNamespace) -> None:
                verify_checkpoint_training_identity(
                    candidate,  # type: ignore[arg-type]
                    training_run,
                    expected_seed=training_run.seed,
                    expected_epoch=1,
                    expected_objective=LEGACY_ADDITIVE_DRAWBACK_OBJECTIVE,
                )

            verify(SimpleNamespace(**values))
            mutations = (
                ("training_run_id", "0" * 64, "training run"),
                ("checkpoint_seed", training_run.seed + 1, "seed"),
                ("checkpoint_epoch", 2, "epoch"),
                ("drawback_loss_objective", "wrong", "objective"),
                (
                    "corpus_provenance",
                    {
                        "training_corpus_set": corpus_set,
                        "training_corpus_set_sha256": "0" * 64,
                    },
                    "corpus",
                ),
            )
            for key, value, message in mutations:
                with self.subTest(key=key), self.assertRaisesRegex(
                    ValueError,
                    message,
                ):
                    verify(SimpleNamespace(**{**values, key: value}))

    def test_verifies_complete_local_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            artifact, run = build_bundle(Path(raw))
            verified = verify_release_selection_bundle(artifact, run)
            self.assertEqual(verified.candidate_count, 2)
            self.assertEqual(verified.artifact.selected_epoch, 2)
            self.assertEqual(verified.training_run.seed, 20260811)
            self.assertEqual(
                verified.training_run.training_corpus_set_sha256,
                training_corpus_set_fixture()["sha256"],
            )
            self.assertEqual(
                verified.artifact.provenance.training_corpus_set_sha256,
                training_corpus_set_fixture()["sha256"],
            )

    def test_verifies_workflow_layout_with_separate_training_and_evidence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            artifact, run = build_bundle(
                Path(raw),
                separate_evidence_directory=True,
            )
            verified = verify_release_selection_bundle(artifact, run)
            self.assertEqual(verified.candidate_count, 2)
            self.assertEqual(verified.artifact.selected_epoch, 2)
            self.assertEqual(artifact.path.parent.name, "frozen")
            self.assertEqual(run.path.parent.name, "training")

    def test_rejects_invented_hash_and_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            artifact, run = build_bundle(directory)
            with self.assertRaisesRegex(ValueError, "sha256"):
                verify_release_selection_bundle(
                    ContentAddressedJson(artifact.path, "0" * 64), run
                )
            (directory / "epoch-1.pt").unlink()
            with self.assertRaisesRegex(ValueError, "missing"):
                verify_release_selection_bundle(artifact, run)

    def test_rejects_altered_report_and_checkpoint(self) -> None:
        for filename in ("report-1.json", "epoch-1.pt"):
            with self.subTest(filename=filename):
                with tempfile.TemporaryDirectory() as raw:
                    directory = Path(raw)
                    artifact, run = build_bundle(directory)
                    (directory / filename).write_bytes(b"altered")
                    with self.assertRaisesRegex(ValueError, "sha256"):
                        verify_release_selection_bundle(artifact, run)

    def test_rejects_selection_bound_to_different_training_corpus_set(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            artifact, run = build_bundle(directory)
            value = json.loads(artifact.path.read_bytes())
            value["provenance"]["training_corpus_set_sha256"] = "9" * 64
            payload = (
                json.dumps(
                    value,
                    separators=(",", ":"),
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
            artifact.path.write_bytes(payload)
            changed = ContentAddressedJson(artifact.path, sha(payload))
            with self.assertRaisesRegex(
                ValueError,
                "different training corpus set",
            ):
                verify_release_selection_bundle(changed, run)

    def test_rejects_duplicate_run_keys(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "run.claim.json"
            payload = (
                '{"config":{"epochs":2,"epochs":3,"seed":1},'
                '"format":"drawbacktrainer-streaming-run","run_id":"'
                + "a" * 64
                + '","runtime":{},"sampling":{},"version":1}\n'
            ).encode("utf-8")
            path.write_bytes(payload)
            with self.assertRaisesRegex(ValueError, "duplicate key"):
                load_training_run(ContentAddressedJson(path, sha(payload)))

    def test_rejects_noncanonical_run_json(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "run.claim.json"
            material = {
                "format": "drawbacktrainer-streaming-run",
                "version": 1,
                "config": {
                    "seed": 1,
                    "epochs": 2,
                    "corpus_provenance": {
                        "training_corpus_set": training_corpus_set_fixture(),
                        "training_corpus_set_sha256": (
                            training_corpus_set_fixture()["sha256"]
                        ),
                    },
                },
                "runtime": {},
                "sampling": {},
            }
            run_id = sha(
                json.dumps(
                    material, separators=(",", ":"), sort_keys=True
                ).encode("utf-8")
            )
            payload = json.dumps(
                {"run_id": run_id, **material}, sort_keys=True
            ).encode("utf-8")
            path.write_bytes(payload)
            with self.assertRaisesRegex(ValueError, "not canonical"):
                load_training_run(ContentAddressedJson(path, sha(payload)))

    def test_rejects_run_id_that_does_not_match_claim_material(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "run.claim.json"
            value = {
                "run_id": "a" * 64,
                "format": "drawbacktrainer-streaming-run",
                "version": 1,
                "config": {
                    "seed": 1,
                    "epochs": 2,
                    "corpus_provenance": {
                        "training_corpus_set": training_corpus_set_fixture(),
                        "training_corpus_set_sha256": (
                            training_corpus_set_fixture()["sha256"]
                        ),
                    },
                },
                "runtime": {},
                "sampling": {},
            }
            payload = canonical(value)
            path.write_bytes(payload)
            with self.assertRaisesRegex(ValueError, "claim material"):
                load_training_run(ContentAddressedJson(path, sha(payload)))

    def test_rejects_legacy_or_invalid_training_corpus_set_identity(self) -> None:
        for corpus_provenance in (
            None,
            {},
            {"training_corpus_set_sha256": "not-a-digest"},
            {
                "training_corpus_set": training_corpus_set_fixture(),
                "training_corpus_set_sha256": "9" * 64,
            },
        ):
            with self.subTest(corpus_provenance=corpus_provenance):
                with tempfile.TemporaryDirectory() as raw:
                    path = Path(raw) / "run.claim.json"
                    config: dict[str, object] = {"seed": 1, "epochs": 2}
                    if corpus_provenance is not None:
                        config["corpus_provenance"] = corpus_provenance
                    material = {
                        "format": "drawbacktrainer-streaming-run",
                        "version": 1,
                        "config": config,
                        "runtime": {},
                        "sampling": {},
                    }
                    run_id = sha(
                        json.dumps(
                            material,
                            separators=(",", ":"),
                            sort_keys=True,
                        ).encode("utf-8")
                    )
                    payload = canonical({"run_id": run_id, **material})
                    path.write_bytes(payload)
                    with self.assertRaisesRegex(
                        ValueError,
                        "corpus_provenance|training corpus set|corpus-set",
                    ):
                        load_training_run(
                            ContentAddressedJson(path, sha(payload))
                        )


if __name__ == "__main__":
    unittest.main()
