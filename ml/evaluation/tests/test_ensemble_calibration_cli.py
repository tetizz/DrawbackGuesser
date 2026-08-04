from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ml.evaluation.calibration import CalibrationExample
from ml.evaluation.calibration_release import CalibrationObservation
from ml.evaluation.cli import (
    _ensemble_member_checkpoint_payloads,
    _ensemble_report_identity,
    _evaluate_ensemble_calibration,
    _select_ensemble_fusion,
    _verify_ensemble_training_corpus_set,
    build_evaluate_ensemble_calibration_parser,
    build_select_ensemble_fusion_parser,
    main,
)
from ml.evaluation.ensemble_calibration import (
    CLASS_COUNT,
    FORMAT_VERSION,
    FUSION_METHOD,
    REPORT_FORMAT,
    ContentAddressedFile,
    EnsembleCalibrationIdentity,
    EnsembleCalibrationMember,
    fit_ensemble_calibration as fit_ensemble_calibration_real,
    load_ensemble_calibration,
    load_ensemble_calibration_sidecar,
)
from ml.evaluation.ensemble_release import (
    ENSEMBLE_TRAINING_SEEDS,
    EnsembleMember,
    LoadedEnsembleRelease,
)
from ml.evaluation.fusion_selection import (
    FusionSelectionAccumulator,
    FusionSelectionObservation,
)
from ml.evaluation.release_selection_bundle import ContentAddressedJson
from ml.evaluation.splits import SplitManifest
from ml.evaluation.validation_partition import (
    VALIDATION_PARTITION_IDENTITY,
    ValidationPartition,
    assign_validation_partition,
    validation_seed_sha256,
)
from ml.training.drawback_ml.symbolic_schema import SYMBOLIC_RULE_IDS
from ml.evaluation.tests.training_corpus_set_fixture import (
    training_corpus_set_fixture,
)


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def calibration_seed() -> int:
    return next(
        seed
        for seed in range(100_000)
        if assign_validation_partition(seed)
        is ValidationPartition.CALIBRATION_FIT
    )


def selection_seed() -> int:
    return next(
        seed
        for seed in range(100_000)
        if assign_validation_partition(seed) is ValidationPartition.SELECTION
    )


class FakeAudited:
    def __init__(
        self,
        seeds: tuple[int, ...],
        *,
        release_root: str,
        corpus_run: str,
        manifest_sha256: str,
        dataset_sha256: str,
    ) -> None:
        self.seeds = seeds
        self.observed_seeds = seeds
        self.game_assignments = tuple(
            (f"game-{index}", "vegan", "checkers")
            for index, _seed in enumerate(seeds, start=1)
        )
        self.max_plies = 80
        self.release_root_sha256 = release_root
        self.corpus_run_id = corpus_run
        self.manifest_sha256 = manifest_sha256
        self.dataset_sha256 = dataset_sha256

    def provenance(self) -> dict[str, object]:
        return {
            "split": "validation",
            "engine_fingerprint": "engine",
            "evaluator_policy_id": "policy",
            "evaluator_policy_version": 1,
            "rule_ids": list(SYMBOLIC_RULE_IDS),
            "symbolic_feature_version": 6,
            "release_root_sha256": self.release_root_sha256,
            "corpus_run_id": self.corpus_run_id,
        }


class FakeLease:
    def __init__(self, audited: FakeAudited) -> None:
        self.audited = audited
        self.dataset = io.BytesIO(b"")

    def verify_dataset_unchanged(self) -> None:
        return None


class FakePredictor:
    drawback_vocabulary = tuple(SYMBOLIC_RULE_IDS)
    parameter_vocabulary = ("none",)
    legal_mask_dimension = 1

    def __init__(self, corpus_set: dict[str, object]) -> None:
        self.members = tuple(
            SimpleNamespace(
                corpus_provenance={
                    "training_corpus_set": corpus_set,
                    "training_corpus_set_sha256": corpus_set["sha256"],
                }
            )
            for _index in range(3)
        )


def build_release(
    root: Path,
) -> tuple[
    ContentAddressedJson,
    LoadedEnsembleRelease,
    EnsembleCalibrationIdentity,
]:
    release_path = root / "ensemble.json"
    release_path.write_text(
        json.dumps({"format": "fixture"}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    reference = ContentAddressedJson(
        release_path, sha(release_path.read_bytes())
    )
    release_root = "a" * 64
    corpus_run = "b" * 64
    manifest_sha = "c" * 64
    dataset_sha = "d" * 64
    corpus_set_sha = str(training_corpus_set_fixture()["sha256"])
    members: list[EnsembleMember] = []
    identity_members: list[EnsembleCalibrationMember] = []
    for index, seed in enumerate(ENSEMBLE_TRAINING_SEEDS):
        directory = root / f"member-{index}"
        directory.mkdir()
        selection = directory / "selection.json"
        selection.write_text("{}\n", encoding="utf-8")
        checkpoint = directory / f"epoch-{index + 1}.pt"
        checkpoint.write_bytes(f"checkpoint-{index}".encode())
        (directory / "run.json").write_bytes(f"run-{index}".encode())
        selection_sha = sha(f"selection-{index}".encode())
        run_sha = sha(f"run-{index}".encode())
        run_id = sha(f"run-id-{index}".encode())
        checkpoint_sha = sha(checkpoint.read_bytes())
        members.append(
            EnsembleMember(
                training_seed=seed,
                selection_file=f"member-{index}/selection.json",
                selection_sha256=selection_sha,
                training_run_file=f"member-{index}/run.json",
                training_run_sha256=run_sha,
                training_run_id=run_id,
                checkpoint_file=checkpoint.name,
                checkpoint_sha256=checkpoint_sha,
                checkpoint_epoch=index + 1,
            )
        )
        identity_members.append(
            EnsembleCalibrationMember(
                seed=seed,
                selection_sha256=selection_sha,
                training_claim_sha256=run_sha,
                training_run_id=run_id,
                checkpoint_sha256=checkpoint_sha,
                checkpoint_epoch=index + 1,
            )
        )
    loaded = LoadedEnsembleRelease(
        source=reference,
        release_root_sha256=release_root,
        corpus_run_id=corpus_run,
        training_corpus_set_sha256=corpus_set_sha,
        private_validation_manifest_sha256=manifest_sha,
        validation_dataset_sha256=dataset_sha,
        partition_seed_sha256=validation_seed_sha256((selection_seed(),)),
        members=tuple(members),
    )
    seed = calibration_seed()
    identity = EnsembleCalibrationIdentity(
        ensemble_release_sha256=reference.sha256,
        fusion_selection_sha256="0" * 64,
        selected_alpha=0.5,
        members=tuple(identity_members),
        release_root_sha256=release_root,
        corpus_run_id=corpus_run,
        training_corpus_set_sha256=corpus_set_sha,
        private_validation_manifest_sha256=manifest_sha,
        validation_dataset_sha256=dataset_sha,
        calibration_seed_sha256=validation_seed_sha256((seed,)),
        symbolic_schema_sha256="1" * 64,
    )
    return reference, loaded, identity


class EnsembleCalibrationCliTests(unittest.TestCase):
    def test_checkpoint_payload_loader_hashes_bytes_after_resolution(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            reference, loaded, _identity = build_release(root)
            member = loaded.members[0]
            checkpoint = (
                reference.path.parent
                / member.training_run_file
            ).parent / member.checkpoint_file
            checkpoint.write_bytes(b"tampered-after-release-load")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                _ensemble_member_checkpoint_payloads(reference, loaded)

    def test_evaluate_command_is_calibration_fit_only(self) -> None:
        parser = build_evaluate_ensemble_calibration_parser()
        actions = {action.dest for action in parser._actions}
        self.assertNotIn("split", actions)
        self.assertNotIn("validation_partition", actions)
        self.assertIn("fusion_selection", actions)
        self.assertIn("fusion_selection_sha256", actions)
        self.assertNotIn("fusion_alpha", actions)
        self.assertNotIn("test", parser.format_help())

    def test_fusion_selection_command_is_structurally_selection_only(
        self,
    ) -> None:
        parser = build_select_ensemble_fusion_parser()
        actions = {action.dest for action in parser._actions}
        self.assertNotIn("split", actions)
        self.assertNotIn("validation_partition", actions)
        self.assertNotIn("fusion_alpha", actions)
        self.assertNotIn("test", parser.format_help())

    def test_fusion_selection_rejects_existing_output_before_loading(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            output = root / "fusion-selection.json"
            original = b"existing-authenticated-evidence\n"
            output.write_bytes(original)
            arguments = argparse.Namespace(
                ensemble_release=root / "ensemble.json",
                ensemble_sha256="0" * 64,
                output=output,
                batch_size=32,
            )
            with patch(
                "ml.evaluation.cli.verify_ensemble_release",
            ) as verify:
                with self.assertRaisesRegex(
                    ValueError,
                    "refusing to overwrite",
                ):
                    _select_ensemble_fusion(
                        arguments,
                        object(),
                    )
            verify.assert_not_called()
            self.assertEqual(output.read_bytes(), original)

    def test_fusion_selection_uses_residuals_and_public_symbolic_state(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            release, loaded, _identity = build_release(root)
            seed = selection_seed()
            audited = FakeAudited(
                (seed,),
                release_root=loaded.release_root_sha256,
                corpus_run=loaded.corpus_run_id,
                manifest_sha256=loaded.private_validation_manifest_sha256,
                dataset_sha256=loaded.validation_dataset_sha256,
            )
            lease = FakeLease(audited)
            output = root / "fusion-selection.json"
            arguments = argparse.Namespace(
                ensemble_release=release.path,
                ensemble_sha256=release.sha256,
                dataset=root / "validation.ndjson",
                public_root=root / "public.json",
                private_validation=root / "validation.private.json",
                output=output,
                device="cpu",
                batch_size=32,
            )
            probability = 1.0 / CLASS_COUNT
            prior = tuple(probability for _index in range(CLASS_COUNT))
            residual = tuple(float(index) for index in range(CLASS_COUNT))
            hard_eliminated = tuple(False for _index in range(CLASS_COUNT))
            example = SimpleNamespace(
                game_id="game-1",
                features=SimpleNamespace(
                    player_color="white",
                    ply=0,
                    symbolic_white_rule_probabilities=prior,
                ),
                white_drawback="vegan",
            )
            prediction = SimpleNamespace(
                white_neural_residual_logits=residual,
                white_hard_eliminated=hard_eliminated,
            )

            def evaluate(_rows: object, **kwargs: object) -> object:
                sink = kwargs["prediction_sink"]
                sink(example, prediction)
                self.assertEqual(kwargs["split"], "validation")
                return SimpleNamespace(
                    white_drawback=SimpleNamespace(
                        count=1,
                        player_game_count=1,
                    ),
                    black_drawback=SimpleNamespace(
                        count=0,
                        player_game_count=0,
                    ),
                )

            def release_inputs(
                _arguments: argparse.Namespace, _lease: object
            ) -> tuple[object, object, object, object]:
                return (
                    audited,
                    {"maxPlies": 80},
                    SplitManifest(train=(), validation=(seed,), test=()),
                    lambda: audited,
                )

            reference = ContentAddressedJson(output, "9" * 64)
            corpus_set = training_corpus_set_fixture()
            captured = io.StringIO()
            captured_observations: list[FusionSelectionObservation] = []

            class CapturingAccumulator(FusionSelectionAccumulator):
                def add(
                    self,
                    observation: FusionSelectionObservation,
                ) -> None:
                    captured_observations.append(observation)
                    super().add(observation)

            selected_candidate = SimpleNamespace(
                alpha=0.5,
                white=SimpleNamespace(
                    observation_count=1,
                    player_game_count=1,
                ),
                black=SimpleNamespace(
                    observation_count=0,
                    player_game_count=0,
                ),
            )
            with patch(
                "ml.evaluation.cli.verify_ensemble_release",
                return_value=loaded,
            ), patch(
                "ml.evaluation.cli.load_hybrid_ensemble",
                return_value=FakePredictor(corpus_set),
            ) as load_ensemble, patch(
                "ml.evaluation.cli._release_evaluation_inputs",
                side_effect=release_inputs,
            ), patch(
                "ml.evaluation.cli.evaluate_held_out",
                side_effect=evaluate,
            ), patch(
                "ml.evaluation.cli.FusionSelectionAccumulator",
                CapturingAccumulator,
            ), patch(
                "ml.evaluation.cli.write_fusion_selection_accumulator",
                return_value=reference,
            ) as write_selection, patch(
                "ml.evaluation.cli.load_fusion_selection_artifact",
                return_value=SimpleNamespace(
                    selected_alpha=0.5,
                    candidates=(selected_candidate,),
                ),
            ), redirect_stdout(captured):
                result = _select_ensemble_fusion(arguments, lease)

            self.assertEqual(result, 0)
            self.assertEqual(
                load_ensemble.call_args.kwargs["fusion_alpha"],
                0.0,
            )
            self.assertEqual(len(write_selection.call_args.args), 2)
            self.assertEqual(len(captured_observations), 1)
            captured_observation = captured_observations[0]
            self.assertEqual(captured_observation.game_id, "game-1")
            self.assertEqual(captured_observation.observed_ply, 0)
            self.assertEqual(captured_observation.residual_logits, residual)
            self.assertEqual(captured_observation.symbolic_prior, prior)
            self.assertEqual(
                captured_observation.hard_eliminated,
                hard_eliminated,
            )
            self.assertEqual(
                json.loads(captured.getvalue())["selected_alpha"],
                0.5,
            )

    def test_evaluate_emits_genuine_ensemble_sidecar_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            release, loaded, identity = build_release(root)
            seed = calibration_seed()
            select_seed = selection_seed()
            audited = FakeAudited(
                (select_seed, seed),
                release_root=loaded.release_root_sha256,
                corpus_run=loaded.corpus_run_id,
                manifest_sha256=loaded.private_validation_manifest_sha256,
                dataset_sha256=loaded.validation_dataset_sha256,
            )
            lease = FakeLease(audited)
            output = root / "report.json"
            sidecar_output = root / "sidecar.ndjson"
            fusion_selection = root / "fusion-selection.json"
            fusion_selection.write_text("{}\n", encoding="utf-8")
            arguments = argparse.Namespace(
                ensemble_release=release.path,
                ensemble_sha256=release.sha256,
                fusion_selection=fusion_selection,
                fusion_selection_sha256="0" * 64,
                dataset=root / "validation.ndjson",
                public_root=root / "public.json",
                private_validation=root / "validation.private.json",
                output=output,
                sidecar_output=sidecar_output,
                device="cpu",
                batch_size=32,
                catalog=[],
            )
            logits = (3.0, 0.0) + (-1.0,) * (CLASS_COUNT - 2)
            eliminated = (False,) * CLASS_COUNT
            example = CalibrationExample(logits, 0, eliminated)

            def evaluate(_rows: object, **kwargs: object) -> dict[str, int]:
                sink = kwargs["calibration_sink"]
                sink(CalibrationObservation("white", example))
                sink(CalibrationObservation("black", example))
                self.assertEqual(kwargs["split"], "validation")
                return {"move_examples": 2}

            def release_inputs(
                _arguments: argparse.Namespace, _lease: object
            ) -> tuple[object, object, object, object]:
                return (
                    audited,
                    {"maxPlies": 80},
                    SplitManifest(
                        train=(),
                        validation=(select_seed, seed),
                        test=(),
                    ),
                    lambda: audited,
                )

            captured = io.StringIO()
            report_attempts = 0

            from ml.evaluation.cli import (
                _write_report_atomic_no_clobber as write_report_real,
            )

            def fail_first_report(
                path: Path,
                rendered: str,
                *,
                recover_exact: bool = False,
            ) -> None:
                nonlocal report_attempts
                report_attempts += 1
                if report_attempts == 1:
                    raise OSError("injected report publication failure")
                write_report_real(
                    path,
                    rendered,
                    recover_exact=recover_exact,
                )

            corpus_set = training_corpus_set_fixture()
            loaded_fusion_selection = SimpleNamespace(
                selected_alpha=0.5,
                identity=object(),
            )
            with patch(
                "ml.evaluation.cli.verify_ensemble_release",
                return_value=loaded,
            ), patch(
                "ml.evaluation.cli.identity_from_release",
                return_value=identity,
            ), patch(
                "ml.evaluation.cli.load_hybrid_ensemble",
                return_value=FakePredictor(corpus_set),
            ) as load_ensemble, patch(
                "ml.evaluation.cli.load_fusion_selection_artifact",
                return_value=loaded_fusion_selection,
            ) as load_fusion_selection, patch(
                "ml.evaluation.cli._release_evaluation_inputs",
                side_effect=release_inputs,
            ), patch(
                "ml.evaluation.cli.load_rule_families",
                return_value={rule_id: "family" for rule_id in SYMBOLIC_RULE_IDS},
            ), patch(
                "ml.evaluation.cli.evaluate_held_out",
                side_effect=evaluate,
            ), patch(
                "ml.evaluation.cli._write_report_atomic_no_clobber",
                side_effect=fail_first_report,
            ), redirect_stdout(captured):
                with self.assertRaisesRegex(OSError, "injected report"):
                    _evaluate_ensemble_calibration(arguments, lease)
                self.assertTrue(sidecar_output.is_file())
                self.assertFalse(output.exists())
                result = _evaluate_ensemble_calibration(arguments, lease)

            self.assertEqual(result, 0)
            loaded_sidecar = load_ensemble_calibration_sidecar(
                type("Reference", (), {
                    "path": sidecar_output,
                    "sha256": sha(sidecar_output.read_bytes()),
                })()
            )
            self.assertEqual(loaded_sidecar.white[0].logits, logits)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(
                report["format"],
                REPORT_FORMAT,
            )
            self.assertEqual(report["identity"]["fusion"], FUSION_METHOD)
            self.assertEqual(
                report["evaluation"]["validation_partition"]["name"],
                "calibration-fit",
            )
            self.assertEqual(
                report["identity"]["training_corpus_set_sha256"],
                loaded.training_corpus_set_sha256,
            )
            self.assertEqual(
                report["identity"]["fusion_selection_sha256"],
                "0" * 64,
            )
            self.assertEqual(report["identity"]["selected_alpha"], 0.5)
            streams = load_ensemble.call_args.args[0]
            self.assertEqual(len(streams), 3)
            self.assertTrue(all(isinstance(item, io.BytesIO) for item in streams))
            self.assertEqual(
                load_ensemble.call_args.kwargs["fusion_alpha"],
                0.5,
            )
            self.assertEqual(load_fusion_selection.call_count, 4)

    def test_union_training_corpus_set_must_match_release_hash(self) -> None:
        corpus_set = training_corpus_set_fixture()
        predictor = FakePredictor(corpus_set)
        _verify_ensemble_training_corpus_set(
            predictor, str(corpus_set["sha256"])
        )
        predictor.members[1].corpus_provenance[
            "training_corpus_set_sha256"
        ] = "0" * 64
        with self.assertRaisesRegex(ValueError, "disagrees with release"):
            _verify_ensemble_training_corpus_set(
                predictor, str(corpus_set["sha256"])
            )

    def test_fit_command_writes_receipt_and_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            release, loaded, identity = build_release(root)
            from ml.evaluation.ensemble_calibration import (
                EnsembleCalibrationObservation,
                write_ensemble_calibration_sidecar,
            )

            logits = (3.0, 0.0) + (-1.0,) * (CLASS_COUNT - 2)
            example = CalibrationExample(
                logits, 0, (False,) * CLASS_COUNT
            )
            sidecar = write_ensemble_calibration_sidecar(
                root / "sidecar.ndjson",
                identity,
                (
                    EnsembleCalibrationObservation("white", example),
                    EnsembleCalibrationObservation("black", example),
                ),
            )
            report = root / "report.json"
            report_value = {
                "format": REPORT_FORMAT,
                "version": FORMAT_VERSION,
                "evaluation": {
                    "validation_partition": {
                        "identity": VALIDATION_PARTITION_IDENTITY,
                        "name": "calibration-fit",
                        "seed_sha256": identity.calibration_seed_sha256,
                    },
                    "calibration_sidecar": {
                        "file": sidecar.path.name,
                        "sha256": sidecar.sha256,
                    },
                },
                "identity": _ensemble_report_identity(identity),
                "metrics": {},
            }
            report.write_bytes(
                (
                    json.dumps(report_value, indent=2, sort_keys=True)
                    + "\n"
                ).encode("utf-8")
            )
            receipt = root / "receipt.json"
            artifact = root / "calibration.json"
            fusion_selection = root / "fusion-selection.json"
            fusion_selection.write_text("{}\n", encoding="utf-8")
            captured = io.StringIO()
            artifact_attempts = 0

            def fail_first_artifact(
                output: Path,
                receipt_reference: ContentAddressedFile,
                *,
                recover_exact: bool = False,
            ) -> ContentAddressedFile:
                nonlocal artifact_attempts
                artifact_attempts += 1
                if artifact_attempts == 1:
                    raise OSError("injected artifact publication failure")
                return fit_ensemble_calibration_real(
                    output,
                    receipt_reference,
                    recover_exact=recover_exact,
                )

            with patch(
                "ml.evaluation.ensemble_calibration.verify_ensemble_release",
                return_value=loaded,
            ), patch(
                "ml.evaluation.ensemble_calibration.load_fusion_selection_artifact",
                return_value=SimpleNamespace(selected_alpha=0.5),
            ), patch(
                "ml.evaluation.cli.fit_ensemble_calibration",
                side_effect=fail_first_artifact,
            ), redirect_stdout(captured):
                with self.assertRaisesRegex(OSError, "injected artifact"):
                    main(
                        [
                            "fit-ensemble-calibration",
                            str(sidecar.path),
                            str(report),
                            str(release.path),
                            str(receipt),
                            str(artifact),
                            "--sidecar-sha256",
                            sidecar.sha256,
                            "--report-sha256",
                            sha(report.read_bytes()),
                            "--ensemble-sha256",
                            release.sha256,
                            "--fusion-selection",
                            str(fusion_selection),
                            "--fusion-selection-sha256",
                            "0" * 64,
                        ]
                    )
                self.assertTrue(receipt.is_file())
                self.assertFalse(artifact.exists())
                result = main(
                    [
                        "fit-ensemble-calibration",
                        str(sidecar.path),
                        str(report),
                        str(release.path),
                        str(receipt),
                        str(artifact),
                        "--sidecar-sha256",
                        sidecar.sha256,
                        "--report-sha256",
                        sha(report.read_bytes()),
                        "--ensemble-sha256",
                        release.sha256,
                        "--fusion-selection",
                        str(fusion_selection),
                        "--fusion-selection-sha256",
                        "0" * 64,
                    ]
                )
                loaded_artifact = load_ensemble_calibration(
                    type("Reference", (), {
                        "path": artifact,
                        "sha256": sha(artifact.read_bytes()),
                    })()
                )
            self.assertEqual(result, 0)
            self.assertTrue(receipt.is_file())
            self.assertLess(
                loaded_artifact["white"]["nll_after"],
                loaded_artifact["white"]["nll_before"],
            )
            emitted = json.loads(captured.getvalue())
            self.assertEqual(
                emitted["calibration"]["sha256"], sha(artifact.read_bytes())
            )


if __name__ == "__main__":
    unittest.main()
