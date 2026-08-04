from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ml.evaluation.calibration import CalibrationExample
from ml.evaluation.ensemble_calibration import (
    CLASS_COUNT,
    FORMAT_VERSION,
    FUSION_METHOD,
    REPORT_FORMAT,
    ContentAddressedFile,
    EnsembleCalibrationObservation,
    EnsembleCalibrationSidecarStream,
    fit_ensemble_calibration,
    identity_from_release,
    load_ensemble_calibration,
    load_ensemble_calibration_receipt,
    load_ensemble_calibration_sidecar,
    write_ensemble_calibration_receipt,
    write_ensemble_calibration_sidecar,
)
from ml.evaluation.ensemble_release import (
    ENSEMBLE_TRAINING_SEEDS,
    EnsembleMember,
    LoadedEnsembleRelease,
)
from ml.evaluation.fusion_selection import (
    FusionSelectionIdentity,
    FusionSelectionObservation,
    write_fusion_selection_artifact,
)
from ml.evaluation.release_selection_bundle import ContentAddressedJson
from ml.evaluation.validation_partition import VALIDATION_PARTITION_IDENTITY


def digest(character: str) -> str:
    return character * 64


def canonical(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def compact(value: object) -> bytes:
    return json.dumps(
        value, separators=(",", ":"), sort_keys=True, allow_nan=False
    ).encode("utf-8")


TRAINING_CORPUS_SET_SHA256 = hashlib.sha256(
    compact({"primary": digest("9")})
).hexdigest()


def reference(path: Path) -> ContentAddressedFile:
    return ContentAddressedFile(
        path, hashlib.sha256(path.read_bytes()).hexdigest()
    )


def release_reference(path: Path) -> ContentAddressedJson:
    return ContentAddressedJson(
        path, hashlib.sha256(path.read_bytes()).hexdigest()
    )


def fake_release(source: ContentAddressedJson) -> LoadedEnsembleRelease:
    members = tuple(
        EnsembleMember(
            training_seed=seed,
            selection_file=f"selection-{index}.json",
            selection_sha256=hashlib.sha256(
                f"selection-{index}".encode()
            ).hexdigest(),
            training_run_file=f"run-{index}.json",
            training_run_sha256=hashlib.sha256(
                f"run-{index}".encode()
            ).hexdigest(),
            training_run_id=hashlib.sha256(
                f"run-id-{index}".encode()
            ).hexdigest(),
            checkpoint_file=f"epoch-{index + 1}.pt",
            checkpoint_sha256=hashlib.sha256(
                f"checkpoint-{index}".encode()
            ).hexdigest(),
            checkpoint_epoch=index + 1,
        )
        for index, seed in enumerate(ENSEMBLE_TRAINING_SEEDS)
    )
    return LoadedEnsembleRelease(
        source=source,
        release_root_sha256=digest("a"),
        corpus_run_id=digest("b"),
        private_validation_manifest_sha256=digest("c"),
        validation_dataset_sha256=digest("d"),
        training_corpus_set_sha256=TRAINING_CORPUS_SET_SHA256,
        partition_seed_sha256=digest("e"),
        members=members,
    )


def observations() -> list[EnsembleCalibrationObservation]:
    logits = (3.0, 0.0) + (-1.0,) * (CLASS_COUNT - 2)
    eliminated = (False, False, True) + (False,) * (CLASS_COUNT - 3)
    example = CalibrationExample(logits, 0, eliminated)
    return [
        EnsembleCalibrationObservation("white", example),
        EnsembleCalibrationObservation("black", example),
    ]


def fusion_reference(
    root: Path,
    release: ContentAddressedJson,
    loaded: LoadedEnsembleRelease,
) -> ContentAddressedJson:
    identity = FusionSelectionIdentity(
        ensemble_release_sha256=release.sha256,
        private_validation_manifest_sha256=(
            loaded.private_validation_manifest_sha256
        ),
        validation_dataset_sha256=loaded.validation_dataset_sha256,
        validation_seed_sha256=loaded.partition_seed_sha256,
        training_corpus_set_sha256=loaded.training_corpus_set_sha256,
        symbolic_schema_sha256=digest("2"),
    )
    probability = 1.0 / CLASS_COUNT
    prior = tuple(probability for _index in range(CLASS_COUNT))
    residual = tuple(0.0 for _index in range(CLASS_COUNT))
    mask = tuple(False for _index in range(CLASS_COUNT))
    rows = tuple(
        FusionSelectionObservation(
            identity=identity,
            partition="selection",
            game_id="game-1",
            color=color,
            observed_ply=index,
            true_index=0,
            residual_logits=residual,
            symbolic_prior=prior,
            hard_eliminated=mask,
        )
        for index, color in enumerate(("white", "black"), start=1)
    )
    return write_fusion_selection_artifact(
        root / "fusion-selection.json",
        identity,
        rows,
    )


class EnsembleCalibrationTests(unittest.TestCase):
    def test_abort_never_deletes_a_replacement_pathname(self) -> None:
        class ReplaceOnClose:
            def __init__(self, stream: object, path: Path) -> None:
                self.stream = stream
                self.path = path

            def fileno(self) -> int:
                return self.stream.fileno()  # type: ignore[attr-defined]

            def close(self) -> None:
                self.stream.close()  # type: ignore[attr-defined]
                self.path.unlink(missing_ok=True)
                self.path.write_bytes(b"replacement\n")

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            _, _, published, _, _ = self.build_inputs(root)
            identity = load_ensemble_calibration_sidecar(published).identity
            stream = EnsembleCalibrationSidecarStream(
                root / "aborted.ndjson",
                identity,
            )
            temporary = stream._temporary
            stream._file = ReplaceOnClose(stream._file, temporary)  # type: ignore[assignment]
            if os.name == "nt":
                with self.assertRaisesRegex(OSError, "retained replacement"):
                    stream.abort()
            else:
                stream.abort()

            self.assertTrue(stream._closed)
            self.assertEqual(temporary.read_bytes(), b"replacement\n")

    def test_sidecar_writer_preserves_primary_when_abort_fails(self) -> None:
        primary = ValueError("observation failed")

        class FailingStream:
            def __init__(self, output: Path, identity: object) -> None:
                del output, identity

            @staticmethod
            def add(observation: object) -> None:
                del observation
                raise primary

            @staticmethod
            def abort() -> None:
                raise OSError("sidecar abort failed")

        with patch(
            "ml.evaluation.ensemble_calibration."
            "EnsembleCalibrationSidecarStream",
            FailingStream,
        ):
            with self.assertRaisesRegex(ValueError, "observation failed") as raised:
                write_ensemble_calibration_sidecar(
                    Path("unused.ndjson"),
                    object(),  # type: ignore[arg-type]
                    (object(),),  # type: ignore[arg-type]
                )

        self.assertIs(raised.exception, primary)
        self.assertIn(
            "sidecar abort failed",
            " ".join(getattr(raised.exception, "__notes__", ())),
        )

    def test_missing_head_error_survives_abort_failure(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            _, _, published, _, _ = self.build_inputs(root)
            identity = load_ensemble_calibration_sidecar(published).identity
            stream = EnsembleCalibrationSidecarStream(
                root / "incomplete-sidecar.ndjson",
                identity,
            )
            with patch.object(
                stream,
                "abort",
                side_effect=OSError("sidecar abort failed"),
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "White and Black",
                ) as raised:
                    stream.finalize()

            self.assertIn(
                "sidecar abort failed",
                " ".join(getattr(raised.exception, "__notes__", ())),
            )
            stream._file.close()
            stream._temporary.unlink(missing_ok=True)

    def build_inputs(
        self, root: Path, *, corpus_set: dict[str, str] | None = None
    ) -> tuple[
        ContentAddressedJson,
        LoadedEnsembleRelease,
        ContentAddressedFile,
        ContentAddressedFile,
        ContentAddressedJson,
    ]:
        release_path = root / "ensemble.json"
        release_path.write_bytes(canonical({"format": "fake-ensemble"}))
        release_ref = release_reference(release_path)
        loaded = fake_release(release_ref)
        fusion = fusion_reference(root, release_ref, loaded)
        with patch(
            "ml.evaluation.ensemble_calibration.verify_ensemble_release",
            return_value=loaded,
        ):
            identity = identity_from_release(
                release_ref,
                calibration_seed_sha256=digest("1"),
                symbolic_schema_sha256=digest("2"),
                fusion_selection=fusion,
                training_corpus_set=corpus_set,
            )
        sidecar = write_ensemble_calibration_sidecar(
            root / "calibration.ndjson", identity, observations()
        )
        report_path = root / "report.json"
        report_path.write_bytes(
            canonical(
                {
                    "format": REPORT_FORMAT,
                    "version": FORMAT_VERSION,
                    "evaluation": {
                        "validation_partition": {
                            "identity": VALIDATION_PARTITION_IDENTITY,
                            "name": "calibration-fit",
                            "seed_sha256": digest("1"),
                        },
                        "calibration_sidecar": {
                            "file": sidecar.path.name,
                            "sha256": sidecar.sha256,
                        },
                    },
                    "identity": json.loads(
                        sidecar.path.read_bytes().splitlines()[0]
                    )["identity"]
                    | {
                        "calibration_seed_sha256": digest("1"),
                        "symbolic_schema_sha256": digest("2"),
                        "symbolic_feature_version": 6,
                        "class_count": 182,
                        "fusion": FUSION_METHOD,
                        "partition_identity": VALIDATION_PARTITION_IDENTITY,
                        "partition_name": "calibration-fit",
                    },
                    "metrics": {},
                }
            )
        )
        return release_ref, loaded, sidecar, reference(report_path), fusion

    def test_sidecar_round_trip_binds_complete_v3_identity(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            _, loaded, sidecar, _, _ = self.build_inputs(
                root, corpus_set={"primary": digest("9")}
            )
            result = load_ensemble_calibration_sidecar(sidecar)
            self.assertEqual(
                tuple(member.seed for member in result.identity.members),
                ENSEMBLE_TRAINING_SEEDS,
            )
            self.assertEqual(result.identity.release_root_sha256, digest("a"))
            self.assertEqual(result.identity.symbolic_schema_sha256, digest("2"))
            self.assertEqual(
                result.identity.training_corpus_set, {"primary": digest("9")}
            )
            self.assertEqual(len(result.white[0].logits), 182)
            self.assertTrue(result.white[0].eliminated[2])
            self.assertFalse(result.white[0].eliminated[0])
            self.assertEqual(
                loaded.members[0].checkpoint_sha256,
                result.identity.members[0].checkpoint_sha256,
            )

    def test_stream_is_atomic_bounded_and_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            release, loaded, _, _, fusion = self.build_inputs(root)
            with patch(
                "ml.evaluation.ensemble_calibration.verify_ensemble_release",
                return_value=loaded,
            ):
                identity = identity_from_release(
                    release,
                    calibration_seed_sha256=digest("1"),
                    symbolic_schema_sha256=digest("2"),
                    fusion_selection=fusion,
                )
            output = root / "stream.ndjson"
            stream = EnsembleCalibrationSidecarStream(output, identity)
            stream.add(observations()[0])
            self.assertFalse(output.exists())
            with self.assertRaisesRegex(ValueError, "White and Black") as raised:
                stream.finalize()
            self.assertFalse(output.exists())
            leftovers = list(root.glob(".stream.ndjson.*.tmp"))
            if os.name == "nt":
                self.assertEqual(len(leftovers), 1)
                self.assertIn(
                    "safe handle-bound unlink is unavailable on Windows",
                    " ".join(getattr(raised.exception, "__notes__", ())),
                )
                leftovers[0].unlink()
            else:
                self.assertEqual(leftovers, [])

            complete = EnsembleCalibrationSidecarStream(output, identity)
            for observation in observations():
                complete.add(observation)
            published = complete.finalize()
            self.assertTrue(published.path.is_file())
            with self.assertRaisesRegex(ValueError, "closed"):
                complete.add(observations()[0])

    def test_receipt_and_artifact_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            release, loaded, sidecar, report, fusion = self.build_inputs(root)
            with patch(
                "ml.evaluation.ensemble_calibration.verify_ensemble_release",
                return_value=loaded,
            ):
                receipt = write_ensemble_calibration_receipt(
                    root / "receipt.json",
                    report=report,
                    sidecar=sidecar,
                    ensemble_release=release,
                    fusion_selection=fusion,
                )
                loaded_receipt = load_ensemble_calibration_receipt(receipt)
                artifact = fit_ensemble_calibration(
                    root / "calibration.json", receipt
                )
                value = load_ensemble_calibration(artifact)
            self.assertEqual(
                loaded_receipt.identity.ensemble_release_sha256, release.sha256
            )
            self.assertEqual(value["version"], FORMAT_VERSION)
            self.assertEqual(value["method"]["fusion"], FUSION_METHOD)
            for head in ("white", "black"):
                self.assertLess(
                    value[head]["nll_after"], value[head]["nll_before"]
                )
                self.assertGreaterEqual(value[head]["temperature"], 0.05)
                self.assertLessEqual(value[head]["temperature"], 10.0)
            with patch(
                "ml.evaluation.ensemble_calibration.verify_ensemble_release",
                return_value=loaded,
            ):
                with self.assertRaisesRegex(ValueError, "overwrite"):
                    fit_ensemble_calibration(
                        root / "calibration.json", receipt
                    )
            changed = json.loads(artifact.path.read_bytes())
            changed["white"]["temperature"] = 1.0
            changed_path = root / "changed-calibration.json"
            changed_path.write_bytes(canonical(changed))
            with patch(
                "ml.evaluation.ensemble_calibration.verify_ensemble_release",
                return_value=loaded,
            ):
                with self.assertRaisesRegex(
                    ValueError, "does not match the bound sidecar"
                ):
                    load_ensemble_calibration(reference(changed_path))

    def test_sidecar_rejects_noncanonical_and_duplicate_json(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            _, _, sidecar, _, _ = self.build_inputs(root)
            lines = sidecar.path.read_bytes().splitlines()
            header = json.loads(lines[0])
            noncanonical = root / "noncanonical.ndjson"
            noncanonical.write_bytes(
                json.dumps(header).encode() + b"\n" + b"\n".join(lines[1:]) + b"\n"
            )
            with self.assertRaisesRegex(ValueError, "not canonical"):
                load_ensemble_calibration_sidecar(reference(noncanonical))

            duplicate = root / "duplicate.ndjson"
            duplicate_header = (
                b'{"record_type":"header","record_type":"header",'
                + compact(header)[1:]
            )
            duplicate.write_bytes(
                duplicate_header + b"\n" + b"\n".join(lines[1:]) + b"\n"
            )
            with self.assertRaisesRegex(ValueError, "duplicate key"):
                load_ensemble_calibration_sidecar(reference(duplicate))

            legacy = root / "legacy-v2.ndjson"
            legacy_header = json.loads(lines[0])
            legacy_header["version"] = 2
            legacy.write_bytes(
                compact(legacy_header)
                + b"\n"
                + b"\n".join(lines[1:])
                + b"\n"
            )
            with self.assertRaisesRegex(ValueError, "unsupported"):
                load_ensemble_calibration_sidecar(reference(legacy))

    def test_sidecar_rejects_member_reorder_and_corpus_set_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            _, _, sidecar, _, _ = self.build_inputs(
                root, corpus_set={"primary": digest("9")}
            )
            lines = sidecar.path.read_bytes().splitlines()
            header = json.loads(lines[0])
            header["identity"]["members"][0], header["identity"]["members"][1] = (
                header["identity"]["members"][1],
                header["identity"]["members"][0],
            )
            reordered = root / "reordered.ndjson"
            reordered.write_bytes(
                compact(header) + b"\n" + b"\n".join(lines[1:]) + b"\n"
            )
            with self.assertRaisesRegex(ValueError, "fixed seed order"):
                load_ensemble_calibration_sidecar(reference(reordered))

            header = json.loads(lines[0])
            header["identity"]["training_corpus_set"]["mapping"]["primary"] = (
                digest("8")
            )
            corpus_tamper = root / "corpus-tamper.ndjson"
            corpus_tamper.write_bytes(
                compact(header) + b"\n" + b"\n".join(lines[1:]) + b"\n"
            )
            with self.assertRaisesRegex(ValueError, "hash is invalid"):
                load_ensemble_calibration_sidecar(reference(corpus_tamper))

    def test_sidecar_rejects_bad_dimensions_nonfinite_and_eliminated_truth(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            _, _, sidecar, _, _ = self.build_inputs(root)
            lines = sidecar.path.read_bytes().splitlines()
            for mutation, message in (
                (
                    lambda row: row["ensemble_fused_logits"].pop(),
                    "dimensions",
                ),
                (
                    lambda row: row["hard_eliminated"].__setitem__(0, True),
                    "hard-eliminated",
                ),
            ):
                row = json.loads(lines[1])
                mutation(row)
                changed = root / f"bad-{message}.ndjson"
                changed.write_bytes(
                    lines[0]
                    + b"\n"
                    + compact(row)
                    + b"\n"
                    + lines[2]
                    + b"\n"
                )
                with self.assertRaisesRegex(ValueError, message):
                    load_ensemble_calibration_sidecar(reference(changed))

            nonfinite = root / "nonfinite.ndjson"
            row_bytes = lines[1].replace(b"3.0", b"NaN", 1)
            nonfinite.write_bytes(
                lines[0] + b"\n" + row_bytes + b"\n" + lines[2] + b"\n"
            )
            with self.assertRaisesRegex(ValueError, "non-finite"):
                load_ensemble_calibration_sidecar(reference(nonfinite))

    def test_receipt_rejects_release_swap_and_legacy_report(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            release, loaded, sidecar, report, fusion = self.build_inputs(root)
            legacy_path = root / "legacy.json"
            legacy_path.write_bytes(
                canonical({"format": "drawbacktrainer-calibration"})
            )
            with patch(
                "ml.evaluation.ensemble_calibration.verify_ensemble_release",
                return_value=loaded,
            ):
                with self.assertRaisesRegex(
                    ValueError, "fields are not canonical"
                ):
                    write_ensemble_calibration_receipt(
                        root / "bad-receipt.json",
                        report=reference(legacy_path),
                        sidecar=sidecar,
                        ensemble_release=release,
                        fusion_selection=fusion,
                    )
                receipt = write_ensemble_calibration_receipt(
                    root / "receipt.json",
                    report=report,
                    sidecar=sidecar,
                    ensemble_release=release,
                    fusion_selection=fusion,
                )

            fabricated_fusion = root / "fabricated-fusion.json"
            fabricated_fusion.write_bytes(canonical({"format": "fabricated"}))
            fabricated_receipt = json.loads(receipt.path.read_bytes())
            fabricated_receipt["fusion_selection"] = {
                "file": fabricated_fusion.name,
                "sha256": hashlib.sha256(
                    fabricated_fusion.read_bytes()
                ).hexdigest(),
            }
            fabricated_receipt_path = root / "fabricated-receipt.json"
            fabricated_receipt_path.write_bytes(canonical(fabricated_receipt))
            with patch(
                "ml.evaluation.ensemble_calibration.verify_ensemble_release",
                return_value=loaded,
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "fusion selection",
                ):
                    load_ensemble_calibration_receipt(
                        reference(fabricated_receipt_path)
                    )

            replacement_path = root / "replacement.json"
            replacement_path.write_bytes(canonical({"format": "replacement"}))
            receipt_value = json.loads(receipt.path.read_bytes())
            receipt_value["ensemble_release"] = {
                "file": replacement_path.name,
                "sha256": hashlib.sha256(
                    replacement_path.read_bytes()
                ).hexdigest(),
            }
            tampered_path = root / "tampered-receipt.json"
            tampered_path.write_bytes(canonical(receipt_value))
            replacement_ref = release_reference(replacement_path)
            replacement_loaded = fake_release(replacement_ref)
            with patch(
                "ml.evaluation.ensemble_calibration.verify_ensemble_release",
                return_value=replacement_loaded,
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "identity|identities disagree",
                ):
                    load_ensemble_calibration_receipt(reference(tampered_path))


if __name__ == "__main__":
    unittest.main()
