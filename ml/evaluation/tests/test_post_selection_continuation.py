from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from ml.evaluation.post_selection_continuation import (
    FORMAT,
    RESUME_VERSION,
    _authenticate_calibration_prefix,
    _exclusive_copy,
    _continued_workflow,
    _stage_seed,
    load_manifest,
    run,
)
from ml.evaluation.release_workflow import (
    ReleaseWorkflowError,
    Step,
    build_plan,
)
from ml.evaluation.tests.test_release_workflow import _workflow


def write(path: Path, value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


class PostSelectionContinuationTests(unittest.TestCase):
    def test_loads_narrow_resume_manifest_with_exact_prefix_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resume.json"
            value = self._manifest()
            value["version"] = RESUME_VERSION
            value["resumeAfterCalibration"] = {
                "ensembleSha256": "1" * 64,
                "fusionSelectionSha256": "6" * 64,
                "calibrationReportSha256": "2" * 64,
                "calibrationSidecarSha256": "3" * 64,
                "calibrationReceiptSha256": "4" * 64,
                "calibrationSha256": "5" * 64,
            }
            expected = write(path, value)
            loaded, digest = load_manifest(path)
            self.assertEqual(loaded, value)
            self.assertEqual(digest, expected)
            value["resumeAfterCalibration"]["extra"] = "6" * 64
            write(path, value)
            with self.assertRaisesRegex(
                ReleaseWorkflowError,
                "resumeAfterCalibration fields",
            ):
                load_manifest(path)

    def test_authenticates_recursive_calibration_prefix_and_records_reuse(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "data" / "generated" / "resume"
            output.mkdir(parents=True)
            names = {
                "ensembleSha256": "ensemble.json",
                "fusionSelectionSha256": "fusion-selection.json",
                "calibrationReportSha256": "calibration-report.json",
                "calibrationSidecarSha256": "calibration-sidecar.ndjson",
                "calibrationReceiptSha256": "calibration-receipt.json",
                "calibrationSha256": "calibration.json",
            }
            resume = {
                field: write(output / filename, {"field": field})
                for field, filename in names.items()
            }
            events: list[str] = []
            with (
                patch(
                    "ml.evaluation.post_selection_continuation."
                    "verify_ensemble_release",
                    side_effect=lambda _reference: (
                        events.append("ensemble")
                        or SimpleNamespace(
                            private_validation_manifest_sha256="a" * 64,
                            validation_dataset_sha256="b" * 64,
                            partition_seed_sha256="c" * 64,
                            training_corpus_set_sha256="d" * 64,
                        )
                    ),
                ),
                patch(
                    "ml.evaluation.post_selection_continuation."
                    "load_fusion_selection_artifact",
                    side_effect=lambda *_arguments, **_keywords: (
                        events.append("fusion")
                        or SimpleNamespace(selected_alpha=0.25)
                    ),
                ),
                patch(
                    "ml.evaluation.post_selection_continuation."
                    "load_ensemble_calibration",
                    side_effect=lambda _reference: (
                        events.append("calibration")
                        or {
                            "ensemble_release": {
                                "file": "ensemble.json",
                                "sha256": resume["ensembleSha256"],
                            },
                            "identity": {
                                "fusion_selection_sha256": resume[
                                    "fusionSelectionSha256"
                                ],
                                "selected_alpha": 0.25,
                            },
                            "report": {
                                "file": "calibration-report.json",
                                "sha256": resume[
                                    "calibrationReportSha256"
                                ],
                            },
                            "sidecar": {
                                "file": "calibration-sidecar.ndjson",
                                "sha256": resume[
                                    "calibrationSidecarSha256"
                                ],
                            },
                            "receipt": {
                                "file": "calibration-receipt.json",
                                "sha256": resume[
                                    "calibrationReceiptSha256"
                                ],
                            },
                        }
                    ),
                ),
            ):
                records = _authenticate_calibration_prefix(
                    root,
                    output,
                    resume,
                )
            self.assertEqual(
                events,
                ["ensemble", "fusion", "calibration"],
            )
            self.assertEqual(
                [record["status"] for record in records],
                ["reused-authenticated"] * 4,
            )
            with patch(
                "ml.evaluation.post_selection_continuation."
                "load_ensemble_calibration",
                return_value={
                    "ensemble_release": {
                        "file": "other-ensemble.json",
                        "sha256": resume["ensembleSha256"],
                    },
                    "identity": {
                        "fusion_selection_sha256": resume[
                            "fusionSelectionSha256"
                        ],
                        "selected_alpha": 0.25,
                    },
                    "report": {
                        "file": "calibration-report.json",
                        "sha256": resume["calibrationReportSha256"],
                    },
                    "sidecar": {
                        "file": "calibration-sidecar.ndjson",
                        "sha256": resume["calibrationSidecarSha256"],
                    },
                    "receipt": {
                        "file": "calibration-receipt.json",
                        "sha256": resume["calibrationReceiptSha256"],
                    },
                },
            ), patch(
                "ml.evaluation.post_selection_continuation."
                "verify_ensemble_release",
                return_value=SimpleNamespace(
                    private_validation_manifest_sha256="a" * 64,
                    validation_dataset_sha256="b" * 64,
                    partition_seed_sha256="c" * 64,
                    training_corpus_set_sha256="d" * 64,
                ),
            ), patch(
                "ml.evaluation.post_selection_continuation."
                "load_fusion_selection_artifact",
                return_value=SimpleNamespace(selected_alpha=0.25),
            ):
                with self.assertRaisesRegex(
                    ReleaseWorkflowError,
                    "bindings disagree",
                ):
                    _authenticate_calibration_prefix(root, output, resume)
            with (
                patch(
                    "ml.evaluation.post_selection_continuation."
                    "verify_ensemble_release",
                    return_value=SimpleNamespace(
                        private_validation_manifest_sha256="a" * 64,
                        validation_dataset_sha256="b" * 64,
                        partition_seed_sha256="c" * 64,
                        training_corpus_set_sha256="d" * 64,
                    ),
                ),
                patch(
                    "ml.evaluation.post_selection_continuation."
                    "load_fusion_selection_artifact",
                    return_value=SimpleNamespace(selected_alpha=0.25),
                ),
                patch(
                    "ml.evaluation.post_selection_continuation."
                    "load_ensemble_calibration",
                    return_value={
                        "ensemble_release": {
                            "file": "ensemble.json",
                            "sha256": resume["ensembleSha256"],
                        },
                        "identity": {
                            "fusion_selection_sha256": resume[
                                "fusionSelectionSha256"
                            ],
                            "selected_alpha": 0.5,
                        },
                        "report": {
                            "file": "calibration-report.json",
                            "sha256": resume[
                                "calibrationReportSha256"
                            ],
                        },
                        "sidecar": {
                            "file": "calibration-sidecar.ndjson",
                            "sha256": resume[
                                "calibrationSidecarSha256"
                            ],
                        },
                        "receipt": {
                            "file": "calibration-receipt.json",
                            "sha256": resume[
                                "calibrationReceiptSha256"
                            ],
                        },
                    },
                ),
            ):
                with self.assertRaisesRegex(
                    ReleaseWorkflowError,
                    "fusion binding disagrees",
                ):
                    _authenticate_calibration_prefix(root, output, resume)
            (output / "calibration.json").write_bytes(b"tampered")
            with self.assertRaisesRegex(
                ReleaseWorkflowError,
                "authentication failed",
            ):
                _authenticate_calibration_prefix(root, output, resume)

    def test_resume_boundary_requires_fusion_selection_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workflow.json").write_text("{}\n", encoding="utf-8")
            (root / "archive").mkdir()
            output = root / "data" / "generated" / "out"
            (output / "reused-selection-inputs").mkdir(parents=True)
            for name in (
                "ensemble.json",
                "calibration-report.json",
                "calibration-sidecar.ndjson",
                "calibration-receipt.json",
                "calibration.json",
            ):
                (output / name).write_bytes(b"evidence")
            manifest = self._manifest()
            manifest["version"] = RESUME_VERSION
            manifest["resumeAfterCalibration"] = {
                "ensembleSha256": "1" * 64,
                "fusionSelectionSha256": "2" * 64,
                "calibrationReportSha256": "3" * 64,
                "calibrationSidecarSha256": "4" * 64,
                "calibrationReceiptSha256": "5" * 64,
                "calibrationSha256": "6" * 64,
            }
            with patch(
                "ml.evaluation.post_selection_continuation.load_workflow",
                return_value=({"candidates": []}, "a" * 64),
            ):
                with self.assertRaisesRegex(
                    ReleaseWorkflowError,
                    "exact calibration-fit boundary",
                ):
                    run(manifest, "f" * 64, root, execute=True)

    def test_loads_strict_content_addressed_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "continue.json"
            value = {
                "format": FORMAT,
                "version": 1,
                "sourceWorkflow": {
                    "path": "release/workflow.json",
                    "sha256": "a" * 64,
                },
                "executionSourceRevision": "e" * 40,
                "archiveRoot": "release/frozen-interrupted",
                "outputRoot": "release/continuation-1",
                "selectionSha256": {
                    "20260811": "b" * 64,
                    "20260812": "c" * 64,
                    "20260813": "d" * 64,
                },
                "transcriptOutput": "release/continuation-1/transcript.json",
            }
            expected = write(path, value)
            loaded, digest = load_manifest(path)
            self.assertEqual(loaded, value)
            self.assertEqual(digest, expected)

    def test_rejects_nonfresh_or_ambiguous_manifest_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "continue.json"
            value = {
                "format": FORMAT,
                "version": 1,
                "sourceWorkflow": {
                    "path": "release/workflow.json",
                    "sha256": "a" * 64,
                },
                "executionSourceRevision": "e" * 40,
                "archiveRoot": "../archive",
                "outputRoot": "release/continuation-1",
                "selectionSha256": {
                    "20260811": "b" * 64,
                    "20260812": "c" * 64,
                    "20260813": "d" * 64,
                },
                "transcriptOutput": "release/continuation-1/transcript.json",
            }
            write(path, value)
            with self.assertRaisesRegex(ReleaseWorkflowError, "escapes"):
                load_manifest(path)

    def test_authenticated_staging_is_no_clobber(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            destination = root / "staged" / "source.json"
            expected = write(source, {"evidence": True})
            _exclusive_copy(source, destination, expected)
            self.assertEqual(destination.read_bytes(), source.read_bytes())
            with self.assertRaisesRegex(
                ReleaseWorkflowError, "already exists"
            ):
                _exclusive_copy(source, destination, expected)
            destination.unlink()
            with self.assertRaisesRegex(
                ReleaseWorkflowError, "authentication failed"
            ):
                _exclusive_copy(source, destination, "0" * 64)

    def test_stages_only_archived_evidence_and_records_checkpoint_source(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "archive"
            source = archive / "20260811"
            source.mkdir(parents=True)
            checkpoint = root / "training" / "epoch-1.pt"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"checkpoint")
            checkpoint_sha = hashlib.sha256(
                checkpoint.read_bytes()
            ).hexdigest()
            report_sha = write(
                source / "epoch-1.selection-report.json",
                {"report": 1},
            )
            summary_sha = write(
                source / "epoch-1.selection-summary.json",
                {"summary": 1},
            )
            selection_sha = write(
                source / "selection.json",
                {
                    "candidates": [
                        {
                            "checkpoint_file": "epoch-1.pt",
                            "checkpoint_sha256": checkpoint_sha,
                            "evaluation_report_file": (
                                "epoch-1.selection-report.json"
                            ),
                            "evaluation_report_sha256": report_sha,
                            "summary_file": (
                                "epoch-1.selection-summary.json"
                            ),
                            "summary_sha256": summary_sha,
                        }
                    ]
                },
            )
            candidate = {
                "seed": 20260811,
                "epochs": [
                    {
                        "epoch": 1,
                        "checkpoint": {
                            "path": "training/epoch-1.pt",
                            "sha256": checkpoint_sha,
                        },
                    }
                ],
                "trainingRun": {
                    "path": "training/run.json",
                    "sha256": "a" * 64,
                },
            }
            staging = root / "data" / "generated" / "continuation"
            with patch(
                "ml.evaluation.post_selection_continuation."
                "verify_release_selection_bundle",
            ):
                evidence = _stage_seed(
                    root,
                    archive,
                    staging,
                    candidate,
                    selection_sha,
                )
            checkpoint_record = next(
                item
                for item in evidence
                if item["kind"] == "checkpoint"
            )
            self.assertEqual(
                checkpoint_record,
                {
                    "kind": "checkpoint",
                    "source": "training/epoch-1.pt",
                    "sha256": checkpoint_sha,
                },
            )
            self.assertFalse(
                (staging / "20260811" / "epoch-1.pt").exists()
            )
            self.assertEqual(
                {
                    path.name
                    for path in (staging / "20260811").iterdir()
                },
                {
                    "epoch-1.selection-report.json",
                    "epoch-1.selection-summary.json",
                    "selection.json",
                },
            )

    def test_validation_frequency_is_staged_beside_ensemble(self) -> None:
        workflow = _workflow()
        continued = _continued_workflow(
            workflow,
            Path("data/generated/out/reused-selection-inputs"),
            Path("data/generated/out"),
            Path("data/generated/out/transcript.json"),
        )
        self.assertEqual(
            continued["trainingFrequency"],
            {
                "path": "data/generated/out/training-frequency.json",
                "sha256": workflow["trainingFrequency"]["sha256"],
            },
        )
        plan = build_plan(continued, Path.cwd())
        fusion = next(
            step
            for step in plan
            if step.stage == "fusion-selection"
        )
        calibration = next(
            step
            for step in plan
            if step.stage == "calibration-evaluation"
        )
        self.assertEqual(
            fusion.outputs,
            (Path("data/generated/out/fusion-selection.json"),),
        )
        self.assertEqual(
            calibration.argv[
                calibration.argv.index("--fusion-selection") + 1
            ],
            str(Path("data/generated/out/fusion-selection.json")),
        )
        validation = next(
            step
            for step in plan
            if step.stage == "validation-gate"
        )
        ensemble = Path(
            validation.argv[
                validation.argv.index(
                    "ml.evaluation.validation_gate"
                ) + 1
            ]
        )
        frequency = Path(
            validation.argv[
                validation.argv.index(
                    "ml.evaluation.validation_gate"
                ) + 3
            ]
        )
        self.assertEqual(ensemble.parent, frequency.parent)
        self.assertEqual(
            frequency,
            Path("data/generated/out/training-frequency.json"),
        )

    def test_strict_json_rejects_duplicates_nonfinite_and_invalid_json(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "continue.json"
            invalid = (
                b'{"format":"a","format":"b"}',
                b'{"value":NaN}',
                b"\xff",
                b"[]",
            )
            for payload in invalid:
                with self.subTest(payload=payload):
                    path.write_bytes(payload)
                    with self.assertRaises(ReleaseWorkflowError):
                        load_manifest(path)

    def test_run_verifies_before_ensemble_and_never_runs_selection(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "workflow.json"
            source.write_text("{}", encoding="utf-8")
            archive = root / "archive"
            archive.mkdir()
            frequency_sha = write(
                root / "training-frequency.json",
                {"frequency": True},
            )
            events: list[str] = []
            workflow = {
                "sourceRevision": "1" * 40,
                "trainingFrequency": {
                    "path": "training-frequency.json",
                    "sha256": frequency_sha,
                },
                "candidates": [
                    {
                        "seed": seed,
                        "epochs": [],
                        "trainingRun": {
                            "path": "training-run.json",
                            "sha256": "9" * 64,
                        },
                    }
                    for seed in (20260811, 20260812, 20260813)
                ],
            }
            manifest = self._manifest()
            selection_sha: dict[str, str] = {}
            for seed in (20260811, 20260812, 20260813):
                directory = archive / str(seed)
                directory.mkdir()
                selection_sha[str(seed)] = write(
                    directory / "selection.json",
                    {"candidates": []},
                )
            manifest["selectionSha256"] = selection_sha
            plan = (
                Step("selection-fit-evaluation", ("selection",), ()),
                Step("ensemble-release", ("ensemble",), (Path("out/e"),)),
                Step(
                    "fusion-selection",
                    ("fusion",),
                    (Path("out/f"),),
                ),
                Step("validation-gate", ("validation",), (Path("out/v"),)),
            )

            def process(arguments, **_kwargs):
                if "rev-parse" in arguments:
                    events.append("revision")
                    return SimpleNamespace(stdout="2" * 40 + "\n")
                events.append("dirty")
                return SimpleNamespace(stdout="")

            def verify(selection, _training):
                seed = int(selection.path.parent.name)
                events.append(f"verify-{seed}")

            def execute(step, *_arguments):
                events.append(f"execute-{step.stage}")
                return step.argv, {}, []

            tools = {
                name: object()
                for name in ("browser", "git", "node", "pnpm")
            }
            binaries = {
                name: root / f"{name}.exe"
                for name in tools
            }
            with (
                patch(
                    "ml.evaluation.post_selection_continuation.load_workflow",
                    return_value=(workflow, "a" * 64),
                ),
                patch(
                    "ml.evaluation.post_selection_continuation._external_tools",
                    return_value=tools,
                ),
                patch(
                    "ml.evaluation.post_selection_continuation._authenticate_external",
                    side_effect=lambda _reference, label: (
                        events.append(f"tool-{label.rsplit(' ', 1)[-1]}")
                        or binaries[label.rsplit(" ", 1)[-1]]
                    ),
                ),
                patch(
                    "ml.evaluation.post_selection_continuation._closed_environment",
                    return_value={"PATH": "closed"},
                ),
                patch(
                    "ml.evaluation.post_selection_continuation.shutil.which",
                    side_effect=lambda name, **_kwargs: str(binaries[name]),
                ),
                patch(
                    "ml.evaluation.post_selection_continuation.subprocess.run",
                    side_effect=process,
                ),
                patch(
                    "ml.evaluation.post_selection_continuation._input_references",
                    return_value=("input",),
                ),
                patch(
                    "ml.evaluation.release_workflow._authenticate_input",
                    side_effect=lambda *_args: events.append("input"),
                ),
                patch(
                    "ml.evaluation.post_selection_continuation."
                    "verify_release_selection_bundle",
                    side_effect=verify,
                ),
                patch(
                    "ml.evaluation.post_selection_continuation._continued_workflow",
                    return_value=workflow,
                ),
                patch(
                    "ml.evaluation.post_selection_continuation.build_plan",
                    return_value=plan,
                ),
                patch(
                    "ml.evaluation.post_selection_continuation._execute_step",
                    side_effect=execute,
                ),
            ):
                transcript = run(
                    manifest,
                    "f" * 64,
                    root,
                    execute=True,
                )

            first_verify = events.index("verify-20260811")
            for name in ("browser", "git", "node", "pnpm"):
                self.assertLess(events.index(f"tool-{name}"), first_verify)
            self.assertLess(events.index("revision"), first_verify)
            self.assertLess(events.index("dirty"), first_verify)
            self.assertLess(events.index("input"), first_verify)
            first_execute = events.index("execute-ensemble-release")
            self.assertTrue(
                all(
                    events.index(f"verify-{seed}") < first_execute
                    for seed in (20260811, 20260812, 20260813)
                )
            )
            self.assertEqual(
                [
                    event
                    for event in events
                    if event.startswith("execute-")
                ],
                [
                    "execute-ensemble-release",
                    "execute-fusion-selection",
                    "execute-validation-gate",
                ],
            )
            self.assertTrue(transcript["reusedEvidence"])
            self.assertEqual(
                transcript["selectionSourceRevision"],
                "1" * 40,
            )
            self.assertEqual(
                transcript["executionSourceRevision"],
                "2" * 40,
            )
            self.assertIn(
                {
                    "kind": "training-frequency",
                    "source": "training-frequency.json",
                    "staged": (
                        "data/generated/out/training-frequency.json"
                    ),
                    "sha256": frequency_sha,
                },
                transcript["reusedInputs"],
            )

    def test_failure_writes_no_transcript_and_existing_root_is_untouched(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "workflow.json"
            source.write_text("{}", encoding="utf-8")
            (root / "archive").mkdir()
            existing = root / "data" / "generated" / "out"
            existing.mkdir(parents=True)
            marker = existing / "marker"
            marker.write_text("keep", encoding="utf-8")
            with patch(
                "ml.evaluation.post_selection_continuation.load_workflow",
                return_value=({}, "a" * 64),
            ):
                with self.assertRaisesRegex(
                    ReleaseWorkflowError,
                    "already exists",
                ):
                    run(
                        self._manifest(),
                        "f" * 64,
                        root,
                        execute=True,
                    )
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")
            self.assertFalse((existing / "transcript.json").exists())

    def test_stage_failure_publishes_no_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workflow.json").write_text("{}", encoding="utf-8")
            (root / "archive").mkdir()
            frequency_sha = write(
                root / "training-frequency.json",
                {"frequency": True},
            )
            workflow = {
                "trainingFrequency": {
                    "path": "training-frequency.json",
                    "sha256": frequency_sha,
                },
                "candidates": [
                    {"seed": seed, "epochs": [], "trainingRun": {}}
                    for seed in (20260811, 20260812, 20260813)
                ],
            }
            tools = {
                name: object()
                for name in ("browser", "git", "node", "pnpm")
            }
            binaries = {
                name: root / f"{name}.exe"
                for name in tools
            }

            def process(arguments, **_kwargs):
                output = (
                    "2" * 40 + "\n"
                    if "rev-parse" in arguments
                    else ""
                )
                return SimpleNamespace(stdout=output)

            def stage(*_arguments, **_keyword_arguments):
                marker = (
                    root / "data" / "generated" / "out" / "staged-marker"
                )
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text("staged", encoding="utf-8")
                return []

            plan = (Step("ensemble-release", ("ensemble",), ()),)
            with (
                patch(
                    "ml.evaluation.post_selection_continuation.load_workflow",
                    return_value=(workflow, "a" * 64),
                ),
                patch(
                    "ml.evaluation.post_selection_continuation._external_tools",
                    return_value=tools,
                ),
                patch(
                    "ml.evaluation.post_selection_continuation._authenticate_external",
                    side_effect=lambda _reference, label: binaries[
                        label.rsplit(" ", 1)[-1]
                    ],
                ),
                patch(
                    "ml.evaluation.post_selection_continuation._closed_environment",
                    return_value={"PATH": "closed"},
                ),
                patch(
                    "ml.evaluation.post_selection_continuation.shutil.which",
                    side_effect=lambda name, **_kwargs: str(binaries[name]),
                ),
                patch(
                    "ml.evaluation.post_selection_continuation.subprocess.run",
                    side_effect=process,
                ),
                patch(
                    "ml.evaluation.post_selection_continuation._input_references",
                    return_value=(),
                ),
                patch(
                    "ml.evaluation.post_selection_continuation._stage_seed",
                    side_effect=stage,
                ),
                patch(
                    "ml.evaluation.post_selection_continuation._continued_workflow",
                    return_value=workflow,
                ),
                patch(
                    "ml.evaluation.post_selection_continuation.build_plan",
                    return_value=plan,
                ),
                patch(
                    "ml.evaluation.post_selection_continuation._execute_step",
                    side_effect=RuntimeError("ensemble failed"),
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "ensemble failed",
                ):
                    run(
                        self._manifest(),
                        "f" * 64,
                        root,
                        execute=True,
                    )
            self.assertEqual(
                (
                    root
                    / "data"
                    / "generated"
                    / "out"
                    / "staged-marker"
                ).read_text(
                    encoding="utf-8"
                ),
                "staged",
            )
            self.assertFalse(
                (
                    root
                    / "data"
                    / "generated"
                    / "out"
                    / "transcript.json"
                ).exists()
            )

    def test_dry_run_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "workflow.json"
            source.write_text("{}", encoding="utf-8")
            (root / "archive").mkdir()
            workflow = {
                "sourceRevision": "1" * 40,
                "candidates": [],
            }
            plan = (Step("ensemble-release", ("ensemble",), ()),)
            with (
                patch(
                    "ml.evaluation.post_selection_continuation.load_workflow",
                    return_value=(workflow, "a" * 64),
                ),
                patch(
                    "ml.evaluation.post_selection_continuation._continued_workflow",
                    return_value=workflow,
                ),
                patch(
                    "ml.evaluation.post_selection_continuation.build_plan",
                    return_value=plan,
                ),
            ):
                transcript = run(
                    self._manifest(),
                    "f" * 64,
                    root,
                    execute=False,
                )
            self.assertFalse(
                (root / "data" / "generated" / "out").exists()
            )
            self.assertEqual(transcript["steps"], [])
            self.assertEqual(transcript["reusedInputs"], [])

    def test_rejects_execution_revision_mismatch_before_staging(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workflow.json").write_text("{}", encoding="utf-8")
            (root / "archive").mkdir()
            workflow = {"sourceRevision": "1" * 40, "candidates": []}
            tools = {
                name: object()
                for name in ("browser", "git", "node", "pnpm")
            }
            binaries = {
                name: root / f"{name}.exe"
                for name in tools
            }
            with (
                patch(
                    "ml.evaluation.post_selection_continuation.load_workflow",
                    return_value=(workflow, "a" * 64),
                ),
                patch(
                    "ml.evaluation.post_selection_continuation._external_tools",
                    return_value=tools,
                ),
                patch(
                    "ml.evaluation.post_selection_continuation._authenticate_external",
                    side_effect=lambda _reference, label: binaries[
                        label.rsplit(" ", 1)[-1]
                    ],
                ),
                patch(
                    "ml.evaluation.post_selection_continuation._closed_environment",
                    return_value={"PATH": "closed"},
                ),
                patch(
                    "ml.evaluation.post_selection_continuation.shutil.which",
                    side_effect=lambda name, **_kwargs: str(binaries[name]),
                ),
                patch(
                    "ml.evaluation.post_selection_continuation.subprocess.run",
                    return_value=SimpleNamespace(stdout="3" * 40 + "\n"),
                ),
                patch(
                    "ml.evaluation.post_selection_continuation._stage_seed"
                ) as stage,
            ):
                with self.assertRaisesRegex(
                    ReleaseWorkflowError,
                    "source revision differs",
                ):
                    run(
                        self._manifest(),
                        "f" * 64,
                        root,
                        execute=True,
                    )
                stage.assert_not_called()

    def test_rejects_output_outside_generated_and_archived_symlink(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workflow.json").write_text("{}", encoding="utf-8")
            archive = root / "archive"
            seed = archive / "20260811"
            seed.mkdir(parents=True)
            manifest = self._manifest()
            manifest["outputRoot"] = "elsewhere/out"
            manifest["transcriptOutput"] = "elsewhere/out/transcript.json"
            with patch(
                "ml.evaluation.post_selection_continuation.load_workflow",
                return_value=({"candidates": []}, "a" * 64),
            ):
                with self.assertRaisesRegex(
                    ReleaseWorkflowError,
                    "child of data/generated",
                ):
                    run(
                        manifest,
                        "f" * 64,
                        root,
                        execute=False,
                    )

            target = root / "selection-target.json"
            target.write_text('{"candidates":[]}', encoding="utf-8")
            link = seed / "selection.json"
            try:
                link.symlink_to(target)
            except OSError:
                self.skipTest("symbolic links are unavailable")
            candidate = {
                "seed": 20260811,
                "epochs": [],
                "trainingRun": {
                    "path": "training.json",
                    "sha256": "a" * 64,
                },
            }
            with self.assertRaisesRegex(
                ReleaseWorkflowError,
                "symlink",
            ):
                _stage_seed(
                    root,
                    archive,
                    root / "data" / "generated" / "out",
                    candidate,
                    "a" * 64,
                )

    @staticmethod
    def _manifest() -> dict[str, object]:
        return {
            "format": FORMAT,
            "version": 1,
            "sourceWorkflow": {
                "path": "workflow.json",
                "sha256": "a" * 64,
            },
            "executionSourceRevision": "2" * 40,
            "archiveRoot": "archive",
            "outputRoot": "data/generated/out",
            "selectionSha256": {
                "20260811": "b" * 64,
                "20260812": "c" * 64,
                "20260813": "d" * 64,
            },
            "transcriptOutput": "data/generated/out/transcript.json",
        }


if __name__ == "__main__":
    unittest.main()
