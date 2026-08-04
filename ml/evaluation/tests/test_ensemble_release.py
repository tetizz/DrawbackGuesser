from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ml.evaluation.ensemble_release import (
    ENSEMBLE_TRAINING_SEEDS,
    EnsembleMember,
    _write_atomic_no_clobber,
    load_ensemble_release,
    resolve_member_checkpoint,
    verify_ensemble_release,
    write_ensemble_release,
)
from ml.evaluation.cli import (
    build_create_ensemble_release_parser,
    main,
)
from ml.evaluation.release_selection_bundle import ContentAddressedJson
from ml.evaluation.selection import (
    ContentAddressedSummary,
    write_selection_artifact,
)
from ml.evaluation.validation_partition import VALIDATION_PARTITION_IDENTITY
from ml.evaluation.tests.training_corpus_set_fixture import (
    training_corpus_set_fixture,
)
from ml.training.drawback_ml.durable_publish import publish_bytes_durable_exact


def canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def pretty_canonical(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write(path: Path, value: object, *, compact: bool = False) -> str:
    payload = canonical(value) if compact else pretty_canonical(value)
    path.write_bytes(payload)
    return sha(payload)


def build_member(
    root: Path,
    seed: int,
    *,
    corpus_run_id: str = "c" * 64,
    training_corpus_variant: int = 0,
    checkpoint_payload: bytes | None = None,
) -> tuple[ContentAddressedJson, ContentAddressedJson]:
    directory = root / str(seed)
    directory.mkdir()
    corpus_set = training_corpus_set_fixture(training_corpus_variant)
    run_material = {
        "format": "drawbacktrainer-streaming-run",
        "version": 1,
        "config": {
            "seed": seed,
            "epochs": 1,
            "model_variant": "v21-hybrid",
            "corpus_provenance": {
                "training_corpus_set": corpus_set,
                "training_corpus_set_sha256": corpus_set["sha256"],
            },
        },
        "runtime": {"device": "cpu"},
        "sampling": {"policy": "game-balanced-v1"},
    }
    run_id = sha(
        json.dumps(
            run_material, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    )
    run_path = directory / "run.claim.json"
    run_sha = write(run_path, {"run_id": run_id, **run_material})
    checkpoint_path = directory / "epoch-1.pt"
    checkpoint_bytes = (
        checkpoint_payload
        if checkpoint_payload is not None
        else f"checkpoint-{seed}".encode("ascii")
    )
    checkpoint_path.write_bytes(checkpoint_bytes)
    checkpoint_sha = sha(checkpoint_bytes)
    provenance = {
        "release_root_sha256": "b" * 64,
        "corpus_run_id": corpus_run_id,
        "training_corpus_set_sha256": corpus_set["sha256"],
        "private_validation_manifest_sha256": "d" * 64,
        "validation_dataset_sha256": "e" * 64,
        "model_run_config_sha256": run_sha,
        "planned_epoch_count": 1,
    }
    report_path = directory / "report-1.json"
    report_sha = write(
        report_path,
        {
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
                "checkpoint_seed": seed,
                "checkpoint_epoch": 1,
                "training_run_id": run_id,
            },
            "metrics": {
                "white_drawback": {"negative_log_likelihood": 0.7},
                "black_drawback": {"negative_log_likelihood": 0.8},
            },
        },
    )
    summary_path = directory / "summary-1.json"
    summary_sha = write(
        summary_path,
        {
            "format_version": 3,
            "training_seed": seed,
            "epoch": 1,
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
            "metrics": {"white_nll": 0.7, "black_nll": 0.8},
        },
    )
    selection_path = directory / "selection.json"
    write_selection_artifact(
        selection_path,
        [ContentAddressedSummary(summary_path, summary_sha)],
    )
    return (
        ContentAddressedJson(
            selection_path, sha(selection_path.read_bytes())
        ),
        ContentAddressedJson(run_path, run_sha),
    )


def build_ensemble(
    root: Path,
) -> tuple[Path, list[ContentAddressedJson], list[ContentAddressedJson]]:
    pairs = [build_member(root, seed) for seed in ENSEMBLE_TRAINING_SEEDS]
    selections = [item[0] for item in pairs]
    runs = [item[1] for item in pairs]
    output = root / "ensemble.json"
    write_ensemble_release(output, selections, runs)
    return output, selections, runs


def rewrite_reference(path: Path, value: object) -> ContentAddressedJson:
    payload = canonical(value)
    path.write_bytes(payload)
    return ContentAddressedJson(path, sha(payload))


class EnsembleReleaseTest(unittest.TestCase):
    def test_publication_retry_accepts_only_exact_committed_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "ensemble.json"
            payload = b'{"ensemble":true}\n'

            def fail_after_publication(
                path: Path,
                value: bytes,
                *,
                label: str,
            ) -> None:
                publish_bytes_durable_exact(path, value, label=label)
                raise OSError("simulated post-publication failure")

            with patch(
                "ml.evaluation.ensemble_release.publish_bytes_durable_exact",
                side_effect=fail_after_publication,
            ):
                with self.assertRaisesRegex(OSError, "post-publication"):
                    _write_atomic_no_clobber(output, payload)

            _write_atomic_no_clobber(output, payload)
            with self.assertRaisesRegex(ValueError, "overwrite"):
                _write_atomic_no_clobber(output, b"different\n")

    def test_create_release_cli_help_documents_fixed_member_order(self) -> None:
        help_text = build_create_ensemble_release_parser().format_help()
        normalized_help = " ".join(help_text.split())
        self.assertIn("exactly three times", help_text)
        self.assertIn(
            "20260811, 20260812, 20260813",
            normalized_help,
        )
        self.assertIn("--selection-sha256", help_text)
        self.assertIn("--training-run-sha256", help_text)

    def test_create_release_cli_publishes_verified_release_and_digest(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            pairs = [
                build_member(root, seed) for seed in ENSEMBLE_TRAINING_SEEDS
            ]
            output = root / "ensemble.json"
            arguments = ["create-ensemble-release", str(output)]
            for selection, _ in pairs:
                arguments.extend(
                    [
                        "--selection",
                        str(selection.path),
                        "--selection-sha256",
                        selection.sha256,
                    ]
                )
            for _, training_run in pairs:
                arguments.extend(
                    [
                        "--training-run",
                        str(training_run.path),
                        "--training-run-sha256",
                        training_run.sha256,
                    ]
                )
            stdout = StringIO()
            with redirect_stdout(stdout):
                result = main(arguments)
            self.assertEqual(result, 0)
            emitted = json.loads(stdout.getvalue())
            self.assertEqual(emitted["file"], output.name)
            self.assertEqual(emitted["sha256"], sha(output.read_bytes()))
            loaded = verify_ensemble_release(
                ContentAddressedJson(output, emitted["sha256"])
            )
            self.assertEqual(
                tuple(member.training_seed for member in loaded.members),
                ENSEMBLE_TRAINING_SEEDS,
            )

    def test_create_release_cli_rejects_incomplete_bindings(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            arguments = ["create-ensemble-release", str(root / "ensemble.json")]
            for index in range(2):
                arguments.extend(
                    [
                        "--selection",
                        str(root / f"selection-{index}.json"),
                        "--selection-sha256",
                        "a" * 64,
                        "--training-run",
                        str(root / f"run-{index}.json"),
                        "--training-run-sha256",
                        "b" * 64,
                    ]
                )
            with self.assertRaisesRegex(
                ValueError,
                "exactly three values",
            ):
                main(arguments)
            self.assertFalse((root / "ensemble.json").exists())

    def test_publishes_and_recursively_verifies_three_members(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output, _, _ = build_ensemble(Path(raw))
            reference = ContentAddressedJson(
                output, sha(output.read_bytes())
            )
            loaded = verify_ensemble_release(reference)
            self.assertEqual(
                tuple(member.training_seed for member in loaded.members),
                ENSEMBLE_TRAINING_SEEDS,
            )
            self.assertEqual(loaded.corpus_run_id, "c" * 64)
            self.assertEqual(
                loaded.training_corpus_set_sha256,
                training_corpus_set_fixture()["sha256"],
            )
            self.assertEqual(loaded.partition_seed_sha256, "f" * 64)
            self.assertEqual(
                len({member.checkpoint_sha256 for member in loaded.members}), 3
            )

    def test_release_resolves_authenticated_sources_outside_output_directory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / ".git").mkdir()
            models = root / "models"
            models.mkdir()
            pairs = [
                build_member(models, seed)
                for seed in ENSEMBLE_TRAINING_SEEDS
            ]
            selections = [selection for selection, _run in pairs]
            runs = [run for _selection, run in pairs]
            output = root / "release" / "frozen" / "ensemble.json"
            output.parent.mkdir(parents=True)
            write_ensemble_release(output, selections, runs)
            reference = ContentAddressedJson(
                output,
                sha(output.read_bytes()),
            )
            loaded = verify_ensemble_release(reference)
            self.assertEqual(
                tuple(member.training_seed for member in loaded.members),
                ENSEMBLE_TRAINING_SEEDS,
            )
            self.assertTrue(
                all(
                    member.training_run_file.startswith("../../")
                    for member in loaded.members
                )
            )

    def test_writer_requires_exact_fixed_seed_order(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            pairs = [
                build_member(root, seed) for seed in ENSEMBLE_TRAINING_SEEDS
            ]
            selections = [item[0] for item in pairs]
            runs = [item[1] for item in pairs]
            with self.assertRaisesRegex(ValueError, "exactly three"):
                write_ensemble_release(
                    root / "short.json", selections[:2], runs[:2]
                )
            with self.assertRaisesRegex(ValueError, "fixed seed order"):
                write_ensemble_release(
                    root / "reordered.json",
                    [selections[1], selections[0], selections[2]],
                    [runs[1], runs[0], runs[2]],
                )

    def test_writer_rejects_mixed_corpus_provenance(self) -> None:
        for changed_field in ("corpus_run_id", "training_corpus_set_sha256"):
            with self.subTest(changed_field=changed_field):
                with tempfile.TemporaryDirectory() as raw:
                    root = Path(raw)
                    pairs = [
                        build_member(
                            root,
                            seed,
                            corpus_run_id=(
                                "9" * 64
                                if changed_field == "corpus_run_id" and index == 2
                                else "c" * 64
                            ),
                            training_corpus_variant=(
                                1
                                if changed_field
                                == "training_corpus_set_sha256" and index == 2
                                else 0
                            ),
                        )
                        for index, seed in enumerate(ENSEMBLE_TRAINING_SEEDS)
                    ]
                    with self.assertRaisesRegex(
                        ValueError,
                        "mixed corpus provenance",
                    ):
                        write_ensemble_release(
                            root / "ensemble.json",
                            [item[0] for item in pairs],
                            [item[1] for item in pairs],
                        )

    def test_writer_rejects_reused_checkpoint_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            pairs = [
                build_member(
                    root,
                    seed,
                    checkpoint_payload=(
                        b"same"
                        if index < 2
                        else f"checkpoint-{seed}".encode("ascii")
                    ),
                )
                for index, seed in enumerate(ENSEMBLE_TRAINING_SEEDS)
            ]
            with self.assertRaisesRegex(ValueError, "selected checkpoint"):
                write_ensemble_release(
                    root / "ensemble.json",
                    [item[0] for item in pairs],
                    [item[1] for item in pairs],
                )

    def test_writer_is_atomic_and_refuses_to_clobber(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            output, selections, runs = build_ensemble(root)
            original = output.read_bytes()
            self.assertEqual(
                write_ensemble_release(output, selections, runs),
                output,
            )
            self.assertEqual(output.read_bytes(), original)
            output.write_bytes(b"competitor\n")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                write_ensemble_release(output, selections, runs)
            self.assertEqual(output.read_bytes(), b"competitor\n")
            self.assertEqual(list(root.glob(".ensemble.json.*.tmp")), [])

    def test_loader_rejects_duplicate_keys_and_noncanonical_json(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            output, _, _ = build_ensemble(root)
            original = output.read_bytes()
            duplicate = original.replace(
                b'"version":3',
                b'"version":3,"version":3',
                1,
            )
            duplicate_path = root / "duplicate.json"
            duplicate_path.write_bytes(duplicate)
            with self.assertRaisesRegex(ValueError, "duplicate key"):
                load_ensemble_release(
                    ContentAddressedJson(duplicate_path, sha(duplicate))
                )
            value = json.loads(original)
            noncanonical = pretty_canonical(value)
            noncanonical_path = root / "noncanonical.json"
            noncanonical_path.write_bytes(noncanonical)
            with self.assertRaisesRegex(ValueError, "not canonical"):
                load_ensemble_release(
                    ContentAddressedJson(noncanonical_path, sha(noncanonical))
                )

    def test_loader_rejects_extra_fields_missing_or_reordered_seeds(self) -> None:
        mutations = ("extra", "missing", "reordered")
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                with tempfile.TemporaryDirectory() as raw:
                    root = Path(raw)
                    output, _, _ = build_ensemble(root)
                    value = json.loads(output.read_bytes())
                    if mutation == "extra":
                        value["unexpected"] = True
                    elif mutation == "missing":
                        value["members"].pop()
                    else:
                        value["members"][0], value["members"][1] = (
                            value["members"][1],
                            value["members"][0],
                        )
                    reference = rewrite_reference(output, value)
                    with self.assertRaisesRegex(
                        ValueError, "canonical|three|reordered"
                    ):
                        load_ensemble_release(reference)

    def test_loader_rejects_legacy_or_missing_training_corpus_set(self) -> None:
        for mutation in ("legacy-version", "missing-hash"):
            with self.subTest(mutation=mutation):
                with tempfile.TemporaryDirectory() as raw:
                    root = Path(raw)
                    output, _, _ = build_ensemble(root)
                    value = json.loads(output.read_bytes())
                    if mutation == "legacy-version":
                        value["version"] = 1
                        expected = "format is unsupported"
                    else:
                        value["provenance"].pop(
                            "training_corpus_set_sha256"
                        )
                        expected = "provenance fields"
                    reference = rewrite_reference(output, value)
                    with self.assertRaisesRegex(ValueError, expected):
                        load_ensemble_release(reference)

    def test_loader_rejects_noncanonical_or_absolute_source_paths(self) -> None:
        for bad_path in (
            "x/./selection.json",
            "/tmp/selection.json",
            "x\\y",
            "C:/selection.json",
            "C:selection.json",
        ):
            with self.subTest(path=bad_path):
                with tempfile.TemporaryDirectory() as raw:
                    root = Path(raw)
                    output, _, _ = build_ensemble(root)
                    value = json.loads(output.read_bytes())
                    value["members"][0]["selection"]["file"] = bad_path
                    reference = rewrite_reference(output, value)
                    with self.assertRaisesRegex(ValueError, "safe relative"):
                        load_ensemble_release(reference)

    def test_verifier_rejects_source_path_outside_repository(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / ".git").mkdir()
            models = root / "models"
            models.mkdir()
            pairs = [
                build_member(models, seed)
                for seed in ENSEMBLE_TRAINING_SEEDS
            ]
            output = root / "release" / "frozen" / "ensemble.json"
            output.parent.mkdir(parents=True)
            write_ensemble_release(
                output,
                [selection for selection, _run in pairs],
                [run for _selection, run in pairs],
            )
            value = json.loads(output.read_bytes())
            value["members"][0]["selection"]["file"] = (
                "../../../../outside-selection.json"
            )
            reference = rewrite_reference(output, value)
            with self.assertRaisesRegex(ValueError, "allowed source root"):
                verify_ensemble_release(reference)

    def test_writer_rejects_symlink_source(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            pairs = [
                build_member(root, seed)
                for seed in ENSEMBLE_TRAINING_SEEDS
            ]
            target = pairs[0][1].path
            link = root / "linked-run.claim.json"
            try:
                link.symlink_to(target)
            except OSError:
                self.skipTest("symbolic links are unavailable")
            runs = [run for _selection, run in pairs]
            runs[0] = ContentAddressedJson(link, runs[0].sha256)
            with self.assertRaisesRegex(ValueError, "non-symlink"):
                write_ensemble_release(
                    root / "ensemble.json",
                    [selection for selection, _run in pairs],
                    runs,
                )

            selection_target = pairs[0][0].path
            selection_link = root / "linked-selection.json"
            selection_link.symlink_to(selection_target)
            selections = [selection for selection, _run in pairs]
            selections[0] = ContentAddressedJson(
                selection_link,
                selections[0].sha256,
            )
            with self.assertRaisesRegex(ValueError, "non-symlink"):
                write_ensemble_release(
                    root / "ensemble.json",
                    selections,
                    [run for _selection, run in pairs],
                )

    def test_checkpoint_resolver_rejects_escape_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            output, _selections, _runs = build_ensemble(root)
            reference = ContentAddressedJson(
                output,
                sha(output.read_bytes()),
            )
            loaded = verify_ensemble_release(reference)
            member = loaded.members[0]
            escaped = EnsembleMember(
                **{
                    **member.__dict__,
                    "checkpoint_file": "../outside.pt",
                }
            )
            (root / "outside.pt").write_bytes(b"outside")
            with self.assertRaisesRegex(ValueError, "escapes"):
                resolve_member_checkpoint(reference, escaped)

            training_run = root / member.training_run_file
            checkpoint = training_run.parent / member.checkpoint_file
            target = checkpoint.with_name("checkpoint-target.pt")
            checkpoint.replace(target)
            try:
                checkpoint.symlink_to(target.name)
            except OSError:
                target.replace(checkpoint)
                self.skipTest("symbolic links are unavailable")
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                resolve_member_checkpoint(reference, member)

    def test_verifier_rejects_tampered_nested_artifacts(self) -> None:
        for relative in (
            f"{ENSEMBLE_TRAINING_SEEDS[0]}/report-1.json",
            f"{ENSEMBLE_TRAINING_SEEDS[1]}/epoch-1.pt",
            f"{ENSEMBLE_TRAINING_SEEDS[2]}/run.claim.json",
        ):
            with self.subTest(relative=relative):
                with tempfile.TemporaryDirectory() as raw:
                    root = Path(raw)
                    output, _, _ = build_ensemble(root)
                    (root / relative).write_bytes(b"tampered")
                    with self.assertRaisesRegex(ValueError, "sha256"):
                        verify_ensemble_release(
                            ContentAddressedJson(
                                output, sha(output.read_bytes())
                            )
                        )

    def test_verifier_rejects_tampered_member_binding(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            output, _, _ = build_ensemble(root)
            value = json.loads(output.read_bytes())
            value["members"][0]["selected_checkpoint"]["epoch"] = 2
            reference = rewrite_reference(output, value)
            with self.assertRaisesRegex(ValueError, "checkpoint binding"):
                verify_ensemble_release(reference)

    def test_verifier_rejects_tampered_declared_training_corpus_set(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            output, _, _ = build_ensemble(root)
            value = json.loads(output.read_bytes())
            value["provenance"]["training_corpus_set_sha256"] = "9" * 64
            reference = rewrite_reference(output, value)
            with self.assertRaisesRegex(
                ValueError,
                "declared provenance",
            ):
                verify_ensemble_release(reference)

    def test_loader_rejects_reused_run_and_selection_identities(self) -> None:
        for field in ("selection", "training_run_sha", "run_id"):
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory() as raw:
                    root = Path(raw)
                    output, _, _ = build_ensemble(root)
                    value = json.loads(output.read_bytes())
                    if field == "selection":
                        value["members"][1]["selection"]["sha256"] = value[
                            "members"
                        ][0]["selection"]["sha256"]
                    elif field == "training_run_sha":
                        value["members"][1]["training_run"]["sha256"] = value[
                            "members"
                        ][0]["training_run"]["sha256"]
                    else:
                        value["members"][1]["training_run"]["run_id"] = value[
                            "members"
                        ][0]["training_run"]["run_id"]
                    reference = rewrite_reference(output, value)
                    with self.assertRaisesRegex(ValueError, "reuse"):
                        load_ensemble_release(reference)


if __name__ == "__main__":
    unittest.main()
