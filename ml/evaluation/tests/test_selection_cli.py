from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from ml.evaluation.cli import FULL_VALIDATION_PARTITION_NAME, main
from ml.evaluation.validation_partition import VALIDATION_PARTITION_IDENTITY
from ml.evaluation.tests.training_corpus_set_fixture import (
    training_corpus_set_fixture,
)
from ml.evaluation.tests.checkpoint_fixture import write_legacy_checkpoint
from ml.training.drawback_ml.inference import CheckpointError


def report_value(
    checkpoint: Path,
    checkpoint_sha256: str,
    *,
    partition: str = "selection",
    white_nll: float = 1.0,
    black_nll: float = 1.0,
    checkpoint_seed: int = 1,
    checkpoint_epoch: int = 1,
    training_run_id: str = "f" * 64,
) -> dict[str, object]:
    return {
        "formatVersion": 1,
        "evaluation": {
            "validationPartition": {
                "identity": VALIDATION_PARTITION_IDENTITY,
                "name": partition,
                "seedSha256": "a" * 64,
            },
        },
        "provenance": {
            "checkpoint_file": checkpoint.name,
            "checkpoint_sha256": checkpoint_sha256,
            "release_root_sha256": "b" * 64,
            "corpus_run_id": "c" * 64,
            "manifest_sha256": "d" * 64,
            "dataset_sha256": "e" * 64,
            "checkpoint_seed": checkpoint_seed,
            "checkpoint_epoch": checkpoint_epoch,
            "training_run_id": training_run_id,
        },
        "metrics": {
            "white_drawback": {
                "negative_log_likelihood": white_nll,
            },
            "black_drawback": {
                "negative_log_likelihood": black_nll,
            },
        },
    }


def write_training_run(
    directory: Path,
    *,
    seed: int,
    epochs: int,
) -> tuple[Path, str, str]:
    corpus_set = training_corpus_set_fixture()
    material = {
        "format": "drawbacktrainer-streaming-run",
        "version": 1,
        "config": {
            "seed": seed,
            "epochs": epochs,
            "corpus_provenance": {
                "training_corpus_set": corpus_set,
                "training_corpus_set_sha256": corpus_set["sha256"],
            },
        },
        "runtime": {"device": "cpu"},
        "sampling": {"policy": "game-balanced-v1"},
    }
    run_id = hashlib.sha256(
        json.dumps(
            material, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    ).hexdigest()
    path = directory / "run.claim.json"
    payload = (
        json.dumps(
            {"run_id": run_id, **material},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    path.write_bytes(payload)
    return path, hashlib.sha256(payload).hexdigest(), run_id


def emit(
    directory: Path,
    *,
    epoch: int,
    white_nll: float,
    black_nll: float,
) -> tuple[Path, str]:
    training_run, training_run_sha256, training_run_id = write_training_run(
        directory, seed=20260811, epochs=2
    )
    checkpoint = directory / f"epoch-{epoch}.pt"
    checkpoint_sha256 = write_legacy_checkpoint(
        checkpoint,
        seed=20260811,
        epoch=epoch,
        run_id=training_run_id,
        training_corpus_set=training_corpus_set_fixture(),
    )
    report = directory / f"report-{epoch}.json"
    report.write_text(
        json.dumps(
            report_value(
                checkpoint,
                checkpoint_sha256,
                white_nll=white_nll,
                black_nll=black_nll,
                checkpoint_seed=20260811,
                checkpoint_epoch=epoch,
                training_run_id=training_run_id,
            ),
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    output = directory / f"summary-{epoch}.json"
    captured = io.StringIO()
    with redirect_stdout(captured):
        result = main(
            [
                "emit-selection-summary",
                str(report),
                str(checkpoint),
                str(output),
                "--training-seed",
                "20260811",
                "--epoch",
                str(epoch),
                "--training-run",
                str(training_run),
                "--training-run-sha256",
                training_run_sha256,
            ]
        )
    assert result == 0
    receipt = json.loads(captured.getvalue())
    return output, receipt["sha256"]


class SelectionCliTest(unittest.TestCase):
    def test_rejects_unloadable_legacy_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            checkpoint = directory / "epoch-1.pt"
            checkpoint.write_bytes(b"not a checkpoint")
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            training_run, training_run_sha256, run_id = write_training_run(
                directory,
                seed=1,
                epochs=1,
            )
            report = directory / "report.json"
            report.write_text(
                json.dumps(
                    report_value(
                        checkpoint,
                        digest,
                        training_run_id=run_id,
                    )
                ),
                encoding="utf-8",
            )
            output = directory / "summary.json"
            with self.assertRaisesRegex(
                CheckpointError,
                "loaded safely",
            ):
                main(
                    [
                        "emit-selection-summary",
                        str(report),
                        str(checkpoint),
                        str(output),
                        "--training-seed",
                        "1",
                        "--epoch",
                        "1",
                        "--training-run",
                        str(training_run),
                        "--training-run-sha256",
                        training_run_sha256,
                    ]
                )
            self.assertFalse(output.exists())

    def test_emits_content_addressable_summary_and_selects_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            first, first_sha = emit(
                directory,
                epoch=1,
                white_nll=0.9,
                black_nll=0.9,
            )
            second, second_sha = emit(
                directory,
                epoch=2,
                white_nll=0.7,
                black_nll=0.7,
            )
            output = directory / "selection.json"
            captured = io.StringIO()
            with redirect_stdout(captured):
                result = main(
                    [
                        "select-epoch",
                        str(output),
                        "--summary",
                        str(first),
                        "--summary-sha256",
                        first_sha,
                        "--summary",
                        str(second),
                        "--summary-sha256",
                        second_sha,
                    ]
                )
            self.assertEqual(result, 0)
            artifact = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(artifact["selected"]["epoch"], 2)
            self.assertEqual(
                artifact["selected"]["evaluation_report_sha256"],
                hashlib.sha256(
                    (directory / "report-2.json").read_bytes()
                ).hexdigest(),
            )
            receipt = json.loads(captured.getvalue())
            self.assertEqual(
                receipt["sha256"],
                hashlib.sha256(output.read_bytes()).hexdigest(),
            )

    def test_rejects_checkpoint_digest_mismatch_before_publication(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            checkpoint = directory / "epoch-1.pt"
            checkpoint.write_bytes(b"real checkpoint")
            report = directory / "report.json"
            training_run, training_run_sha256, run_id = write_training_run(
                directory, seed=1, epochs=1
            )
            report.write_text(
                json.dumps(
                    report_value(
                        checkpoint,
                        "0" * 64,
                        training_run_id=run_id,
                    )
                ),
                encoding="utf-8",
            )
            output = directory / "summary.json"
            with self.assertRaisesRegex(ValueError, "checkpoint bytes"):
                main(
                    [
                        "emit-selection-summary",
                        str(report),
                        str(checkpoint),
                        str(output),
                        "--training-seed",
                        "1",
                        "--epoch",
                        "1",
                        "--training-run",
                        str(training_run),
                        "--training-run-sha256",
                        training_run_sha256,
                    ]
                )
            self.assertFalse(output.exists())

    def test_rejects_non_selection_nonfinite_and_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            checkpoint = directory / "epoch-1.pt"
            checkpoint.write_bytes(b"checkpoint")
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            report = directory / "report.json"
            output = directory / "summary.json"
            training_run, training_run_sha256, _run_id = write_training_run(
                directory, seed=1, epochs=1
            )

            report.write_text(
                json.dumps(
                    report_value(
                        checkpoint,
                        digest,
                        partition="validation-gate",
                    )
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "selection-only"):
                main(
                    [
                        "emit-selection-summary",
                        str(report),
                        str(checkpoint),
                        str(output),
                        "--training-seed",
                        "1",
                        "--epoch",
                        "1",
                        "--training-run",
                        str(training_run),
                        "--training-run-sha256",
                        training_run_sha256,
                    ]
                )

            report.write_text(
                json.dumps(
                    report_value(
                        checkpoint,
                        digest,
                        partition=FULL_VALIDATION_PARTITION_NAME,
                    )
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "selection-only"):
                main(
                    [
                        "emit-selection-summary",
                        str(report),
                        str(checkpoint),
                        str(output),
                        "--training-seed",
                        "1",
                        "--epoch",
                        "1",
                        "--training-run",
                        str(training_run),
                        "--training-run-sha256",
                        training_run_sha256,
                    ]
                )

            nonfinite = json.dumps(
                report_value(checkpoint, digest)
            ).replace("1.0", "1e999", 1)
            report.write_text(nonfinite, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-finite"):
                main(
                    [
                        "emit-selection-summary",
                        str(report),
                        str(checkpoint),
                        str(output),
                        "--training-seed",
                        "1",
                        "--epoch",
                        "1",
                        "--training-run",
                        str(training_run),
                        "--training-run-sha256",
                        training_run_sha256,
                    ]
                )

            report.write_text(
                json.dumps(
                    report_value(
                        checkpoint,
                        digest,
                        training_run_id=_run_id,
                    )
                ),
                encoding="utf-8",
            )
            digest = write_legacy_checkpoint(
                checkpoint,
                seed=1,
                epoch=1,
                run_id=_run_id,
                training_corpus_set=training_corpus_set_fixture(),
            )
            report.write_text(
                json.dumps(
                    report_value(
                        checkpoint,
                        digest,
                        training_run_id=_run_id,
                    )
                ),
                encoding="utf-8",
            )
            output.write_text("preserve me", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                main(
                    [
                        "emit-selection-summary",
                        str(report),
                        str(checkpoint),
                        str(output),
                        "--training-seed",
                        "1",
                        "--epoch",
                        "1",
                        "--training-run",
                        str(training_run),
                        "--training-run-sha256",
                        training_run_sha256,
                    ]
                )
            self.assertEqual(output.read_text(encoding="utf-8"), "preserve me")

    def test_select_requires_parallel_summary_hash_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            with self.assertRaisesRegex(ValueError, "equal counts"):
                main(
                    [
                        "select-epoch",
                        str(directory / "selection.json"),
                        "--summary",
                        str(directory / "summary.json"),
                        "--summary-sha256",
                        "a" * 64,
                        "--summary-sha256",
                        "b" * 64,
                    ]
                )


if __name__ == "__main__":
    unittest.main()
