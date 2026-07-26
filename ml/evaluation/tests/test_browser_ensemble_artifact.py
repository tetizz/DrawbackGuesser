from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ml.evaluation.browser_ensemble_artifact import (
    BROWSER_ENSEMBLE_ARTIFACT_VERSION,
    BROWSER_ENSEMBLE_MODEL_VARIANT,
    export_browser_ensemble_artifact,
)
from ml.evaluation.cli import main
from ml.evaluation.ensemble_calibration import (
    FUSION_METHOD,
    ContentAddressedFile,
)
from ml.evaluation.ensemble_release import (
    ENSEMBLE_TRAINING_SEEDS,
    EnsembleMember,
    LoadedEnsembleRelease,
)
from ml.evaluation.release_selection_bundle import ContentAddressedJson
from ml.training.drawback_ml.browser_artifact import BrowserArtifactError


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class BrowserEnsembleArtifactTests(unittest.TestCase):
    def inputs(
        self,
        root: Path,
    ) -> tuple[
        ContentAddressedJson,
        ContentAddressedFile,
        LoadedEnsembleRelease,
        dict[str, object],
    ]:
        release_path = root / "ensemble.json"
        release_path.write_bytes(b"ensemble\n")
        release_ref = ContentAddressedJson(
            release_path,
            sha(release_path.read_bytes()),
        )
        members: list[EnsembleMember] = []
        for index, seed in enumerate(ENSEMBLE_TRAINING_SEEDS):
            selection_dir = root / f"member-{index}"
            selection_dir.mkdir()
            checkpoint = selection_dir / f"epoch-{index + 1}.pt"
            checkpoint.write_bytes(f"checkpoint-{index}".encode("ascii"))
            training_run = selection_dir / "run.json"
            training_run.write_bytes(f"run-{index}".encode())
            members.append(
                EnsembleMember(
                    training_seed=seed,
                    selection_file=f"member-{index}/selection.json",
                    selection_sha256=sha(f"selection-{index}".encode()),
                    training_run_file=f"member-{index}/run.json",
                    training_run_sha256=sha(training_run.read_bytes()),
                    training_run_id=sha(f"run-id-{index}".encode()),
                    checkpoint_file=checkpoint.name,
                    checkpoint_sha256=sha(checkpoint.read_bytes()),
                    checkpoint_epoch=index + 1,
                )
            )
        loaded = LoadedEnsembleRelease(
            source=release_ref,
            release_root_sha256="a" * 64,
            corpus_run_id="b" * 64,
            training_corpus_set_sha256="c" * 64,
            private_validation_manifest_sha256="d" * 64,
            validation_dataset_sha256="e" * 64,
            partition_seed_sha256="f" * 64,
            members=tuple(members),
        )
        calibration_path = root / "calibration.json"
        calibration_path.write_bytes(b"calibration\n")
        calibration_ref = ContentAddressedFile(
            calibration_path,
            sha(calibration_path.read_bytes()),
        )
        head = {
            "format_version": 1,
            "method": "multiclass-temperature-scaling",
            "fitted_split": "validation",
            "temperature": 1.25,
            "example_count": 50,
            "nll_before": 4.0,
            "nll_after": 3.0,
            "preserves_hard_eliminations": True,
        }
        calibration: dict[str, object] = {
            "method": {
                "name": "per-head-multiclass-temperature-scaling",
                "fusion": FUSION_METHOD,
                "minimum_temperature": 0.05,
                "maximum_temperature": 10.0,
                "preserves_hard_eliminations": True,
                "requires_each_head_nll_improvement": True,
            },
            "identity": {
                "ensemble_release_sha256": release_ref.sha256,
                "fusion_selection_sha256": "9" * 64,
                "selected_alpha": 0.5,
                "members": [
                    {
                        "seed": member.training_seed,
                        "checkpoint_sha256": member.checkpoint_sha256,
                    }
                    for member in members
                ],
            },
            "white": head,
            "black": {**head, "temperature": 0.75},
        }
        return release_ref, calibration_ref, loaded, calibration

    @staticmethod
    def member_artifact(payload: bytes) -> dict[str, object]:
        index = int(payload.decode("ascii").rsplit("-", 1)[1])
        return {
            "format": "drawbacktrainer-browser-model",
            "formatVersion": 2,
            "modelVariant": "v21-hybrid",
            "featureSchemaVersion": 1,
            "symbolicFeatureVersion": 6,
            "sourceCheckpointSha256": sha(payload),
            "drawbackVocabulary": ["vegan", "checkers"],
            "symbolicRuleIds": ["vegan", "checkers"],
            "tokenizer": {
                "kind": "exact-san-token",
                "version": 1,
                "vocabulary": ["<pad>", "<unk>"],
                "max_history": 8,
                "padding": "right",
                "truncation": "keep-most-recent",
            },
            "tensorEncoding": "float32-le-base64",
            "dimensions": {
                "input": 792,
                "boardHidden": 2,
                "sanVocabulary": 2,
                "sanEmbedding": 2,
                "sequenceHidden": 2,
                "symbolicInput": 8,
                "symbolicHidden": 2,
                "drawbackClasses": 2,
            },
            "tensors": {"member.bias": {"shape": [1], "data": str(index)}},
        }

    def test_exports_canonical_compact_bound_v4_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            release, calibration_ref, loaded, calibration = self.inputs(root)
            output = root / "browser.json"
            with patch(
                "ml.evaluation.browser_ensemble_artifact.verify_ensemble_release",
                return_value=loaded,
            ) as verify_release, patch(
                "ml.evaluation.browser_ensemble_artifact.load_ensemble_calibration",
                return_value=calibration,
            ) as verify_calibration, patch(
                "ml.evaluation.browser_ensemble_artifact.build_browser_artifact",
                side_effect=self.member_artifact,
            ):
                export_browser_ensemble_artifact(
                    release,
                    calibration_ref,
                    output,
                )

            payload = output.read_bytes()
            value = json.loads(payload)
            self.assertTrue(payload.endswith(b"\n"))
            self.assertNotIn(b"\n ", payload)
            self.assertEqual(value["formatVersion"], BROWSER_ENSEMBLE_ARTIFACT_VERSION)
            self.assertEqual(value["modelVariant"], BROWSER_ENSEMBLE_MODEL_VARIANT)
            self.assertEqual(value["ensemble"]["method"], FUSION_METHOD)
            self.assertEqual(
                value["ensemble"]["sourceFusionSelectionSha256"],
                "9" * 64,
            )
            self.assertEqual(value["ensemble"]["selectedAlpha"], 0.5)
            self.assertEqual(
                value["ensemble"]["seedOrder"],
                list(ENSEMBLE_TRAINING_SEEDS),
            )
            self.assertEqual(
                [
                    member["selectedEpoch"]
                    for member in value["ensemble"]["members"]
                ],
                [1, 2, 3],
            )
            self.assertEqual(
                value["calibration"]["white"],
                {
                    "temperature": 1.25,
                    "exampleCount": 50,
                    "nllBefore": 4.0,
                    "nllAfter": 3.0,
                },
            )
            self.assertEqual(value["calibration"]["black"]["temperature"], 0.75)
            self.assertEqual(verify_release.call_count, 2)
            self.assertEqual(verify_calibration.call_count, 2)

    def test_rejects_mixed_contract_binding_and_oversized_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            release, calibration_ref, loaded, calibration = self.inputs(root)

            def mixed(payload: bytes) -> dict[str, object]:
                artifact = self.member_artifact(payload)
                if payload.endswith(b"1"):
                    artifact["tokenizer"] = {
                        **artifact["tokenizer"],  # type: ignore[arg-type]
                        "max_history": 9,
                    }
                return artifact

            with patch(
                "ml.evaluation.browser_ensemble_artifact.verify_ensemble_release",
                return_value=loaded,
            ), patch(
                "ml.evaluation.browser_ensemble_artifact.load_ensemble_calibration",
                return_value=calibration,
            ), patch(
                "ml.evaluation.browser_ensemble_artifact.build_browser_artifact",
                side_effect=mixed,
            ):
                with self.assertRaisesRegex(
                    BrowserArtifactError,
                    "incompatible shared contract",
                ):
                    export_browser_ensemble_artifact(
                        release, calibration_ref, root / "mixed.json"
                    )

            def unsupported_encoding(payload: bytes) -> dict[str, object]:
                artifact = self.member_artifact(payload)
                artifact["tensorEncoding"] = "future-encoding"
                return artifact

            with patch(
                "ml.evaluation.browser_ensemble_artifact.verify_ensemble_release",
                return_value=loaded,
            ), patch(
                "ml.evaluation.browser_ensemble_artifact.load_ensemble_calibration",
                return_value=calibration,
            ), patch(
                "ml.evaluation.browser_ensemble_artifact.build_browser_artifact",
                side_effect=unsupported_encoding,
            ):
                with self.assertRaisesRegex(
                    BrowserArtifactError,
                    "must be v21-hybrid",
                ):
                    export_browser_ensemble_artifact(
                        release, calibration_ref, root / "encoding.json"
                    )

            def v22_member(payload: bytes) -> dict[str, object]:
                artifact = self.member_artifact(payload)
                artifact.update(
                    {
                        "formatVersion": 3,
                        "modelVariant": "v22-hybrid",
                        "sequenceObservationMode": "exact-current-v2",
                    }
                )
                return artifact

            with patch(
                "ml.evaluation.browser_ensemble_artifact.verify_ensemble_release",
                return_value=loaded,
            ), patch(
                "ml.evaluation.browser_ensemble_artifact.load_ensemble_calibration",
                return_value=calibration,
            ), patch(
                "ml.evaluation.browser_ensemble_artifact.build_browser_artifact",
                side_effect=v22_member,
            ):
                with self.assertRaisesRegex(
                    BrowserArtifactError,
                    "must be v21-hybrid",
                ):
                    export_browser_ensemble_artifact(
                        release,
                        calibration_ref,
                        root / "v22.json",
                    )

            with patch(
                "ml.evaluation.browser_ensemble_artifact.verify_ensemble_release",
                return_value=loaded,
            ), patch(
                "ml.evaluation.browser_ensemble_artifact.load_ensemble_calibration",
                return_value=calibration,
            ), patch(
                "ml.evaluation.browser_ensemble_artifact.build_browser_artifact",
                side_effect=self.member_artifact,
            ), patch(
                "ml.evaluation.browser_ensemble_artifact."
                "MAX_BROWSER_ENSEMBLE_ARTIFACT_BYTES",
                1,
            ):
                with self.assertRaisesRegex(
                    BrowserArtifactError,
                    "exceeds 1 bytes",
                ):
                    export_browser_ensemble_artifact(
                        release, calibration_ref, root / "large.json"
                    )

    def test_rejects_calibration_drift_mutation_and_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            release, calibration_ref, loaded, calibration = self.inputs(root)
            wrong = {
                **calibration,
                "identity": {
                    **calibration["identity"],  # type: ignore[arg-type]
                    "ensemble_release_sha256": "0" * 64,
                },
            }
            with patch(
                "ml.evaluation.browser_ensemble_artifact.verify_ensemble_release",
                return_value=loaded,
            ), patch(
                "ml.evaluation.browser_ensemble_artifact.load_ensemble_calibration",
                return_value=wrong,
            ):
                with self.assertRaisesRegex(
                    BrowserArtifactError,
                    "does not bind",
                ):
                    export_browser_ensemble_artifact(
                        release, calibration_ref, root / "wrong.json"
                    )

            output = root / "existing.json"
            output.write_bytes(b"keep")
            with patch(
                "ml.evaluation.browser_ensemble_artifact.verify_ensemble_release",
                return_value=loaded,
            ), patch(
                "ml.evaluation.browser_ensemble_artifact.load_ensemble_calibration",
                return_value=calibration,
            ), patch(
                "ml.evaluation.browser_ensemble_artifact.build_browser_artifact",
                side_effect=self.member_artifact,
            ):
                with self.assertRaisesRegex(BrowserArtifactError, "overwrite"):
                    export_browser_ensemble_artifact(
                        release, calibration_ref, output
                    )
            self.assertEqual(output.read_bytes(), b"keep")

            changed = RuntimeError("release changed")
            with patch(
                "ml.evaluation.browser_ensemble_artifact.verify_ensemble_release",
                side_effect=[loaded, changed],
            ), patch(
                "ml.evaluation.browser_ensemble_artifact.load_ensemble_calibration",
                return_value=calibration,
            ), patch(
                "ml.evaluation.browser_ensemble_artifact.build_browser_artifact",
                side_effect=self.member_artifact,
            ):
                with self.assertRaisesRegex(RuntimeError, "release changed"):
                    export_browser_ensemble_artifact(
                        release, calibration_ref, root / "changed.json"
                    )
            self.assertFalse((root / "changed.json").exists())

    def test_cli_wires_content_addresses_and_prints_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "browser.json"
            stdout = StringIO()
            with patch(
                "ml.evaluation.cli.export_browser_ensemble_artifact",
                return_value=output,
            ) as export, redirect_stdout(stdout):
                result = main(
                    [
                        "export-browser-ensemble",
                        str(root / "ensemble.json"),
                        str(root / "calibration.json"),
                        str(output),
                        "--ensemble-sha256",
                        "a" * 64,
                        "--calibration-sha256",
                        "b" * 64,
                    ]
                )
            self.assertEqual(result, 0)
            self.assertEqual(stdout.getvalue().strip(), str(output))
            ensemble, calibration, destination = export.call_args.args
            self.assertEqual(ensemble.sha256, "a" * 64)
            self.assertEqual(calibration.sha256, "b" * 64)
            self.assertEqual(destination, output)


if __name__ == "__main__":
    unittest.main()
