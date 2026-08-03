from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from ml.evaluation.release_workflow import (
    FORMAT,
    ReleaseWorkflowError,
    _SelectionWave,
    build_plan,
    _confined,
    _closed_environment,
    load_workflow,
    run,
)


def _ref(path: str, digest: str = "c" * 64) -> dict[str, str]:
    return {"path": path, "sha256": digest}


def _external_path(name: str) -> str:
    return (
        f"C:/release-tools/{name}"
        if os.name == "nt"
        else f"/opt/release-tools/{name}"
    )


def _workflow() -> dict[str, object]:
    candidates = []
    for seed in (20260811, 20260812, 20260813):
        candidates.append({
            "seed": seed,
            "trainingRun": _ref(f"release/{seed}/run.json"),
            "epochs": [
                {
                    "epoch": epoch,
                    "checkpoint": _ref(
                        f"release/{seed}/epoch-{epoch}.pt"
                    ),
                    "report": f"release/{seed}/epoch-{epoch}.selection.json",
                    "summary": f"release/{seed}/epoch-{epoch}.summary.json",
                }
                for epoch in range(1, 9)
            ],
        })
    return {
        "format": FORMAT,
        "version": 3,
        "sourceRevision": "a" * 40,
        "realDomainExecution": "external-isolated-only",
        "tools": {
            "browser": {
                "path": _external_path("browser"),
                "sha256": "b" * 64,
            },
            "git": {
                "path": _external_path("git"),
                "sha256": "d" * 64,
            },
            "node": {
                "path": _external_path("node"),
                "sha256": "f" * 64,
            },
            "pnpm": {
                "path": _external_path("pnpm"),
                "sha256": "e" * 64,
            },
        },
        "shared": {
            "dataset": _ref("release/validation.ndjson"),
            "publicRoot": _ref("release/public.json"),
            "privateValidation": _ref("release/private-validation.json"),
        },
        "candidates": candidates,
        "selectionOutputs": {
            "20260811": "release/20260811/selected.json",
            "20260812": "release/20260812/selected.json",
            "20260813": "release/20260813/selected.json",
        },
        "ensemble": {"output": "release/ensemble.json"},
        "calibration": {
            "report": "release/calibration-report.json",
            "sidecar": "release/calibration-sidecar.ndjson",
            "receipt": "release/calibration-receipt.json",
            "output": "release/calibration.json",
        },
        "trainingFrequency": _ref("release/training-frequency.json"),
        "validation": {
            "report": "release/validation-report.json",
            "decision": "release/validation-decision.json",
        },
        "browserRelease": {
            "fixture": _ref("release/public-parity-fixture.json"),
            "artifact": "release/browser-model.json",
            "parityInput": "release/parity-input.json",
            "transcript": "release/parity-transcript.json",
            "evidence": "release/parity-evidence.json",
        },
        "transcriptOutput": "release/workflow-transcript.json",
    }


def _write(path: Path, value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode() + b"\n"
    path.write_bytes(payload)
    import hashlib
    return hashlib.sha256(payload).hexdigest()


class ReleaseWorkflowTests(unittest.TestCase):
    def test_builds_only_approved_typed_commands(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "workflow.json"
            digest = _write(path, _workflow())
            workflow, loaded_digest = load_workflow(path)
            self.assertEqual(loaded_digest, digest)
            plan = build_plan(workflow, root)
            self.assertEqual(len(plan), 59)
            evaluations = [
                step for step in plan if step.stage == "selection-fit-evaluation"
            ]
            self.assertEqual(len(evaluations), 24)
            self.assertEqual(
                {(step.seed, step.epoch) for step in evaluations},
                {
                    (seed, epoch)
                    for seed in (20260811, 20260812, 20260813)
                    for epoch in range(1, 9)
                },
            )
            for step in evaluations:
                self.assertIn("--validation-partition", step.argv)
                index = step.argv.index("--validation-partition")
                self.assertEqual(step.argv[index + 1], "selection")
                self.assertEqual(step.argv[0], sys.executable)
                self.assertEqual(step.argv[1:4], ("-I", "-S", "-c"))
                self.assertEqual(step.argv[10], "ml.evaluation.cli")
                self.assertEqual(len(step.inputs), 4)
                self.assertEqual(step.generated_inputs, ())
            expected_generated_counts = {
                "selection-summary": 1,
                "epoch-selection": 8,
                "ensemble-release": 3,
                "fusion-selection": 1,
                "calibration-evaluation": 2,
                "calibration-fit": 4,
                "validation-gate": 2,
                "browser-artifact": 2,
                "browser-parity-input": 3,
                "browser-parity": 3,
            }
            for step in plan:
                if step.stage in expected_generated_counts:
                    self.assertEqual(
                        len(step.generated_inputs),
                        expected_generated_counts[step.stage],
                    )
            stages = [step.stage for step in plan]
            ensemble_index = stages.index("ensemble-release")
            self.assertEqual(
                stages[ensemble_index : ensemble_index + 3],
                [
                    "ensemble-release",
                    "fusion-selection",
                    "calibration-evaluation",
                ],
            )
            fusion = plan[ensemble_index + 1]
            self.assertEqual(fusion.argv[10], "ml.evaluation.cli")
            self.assertEqual(fusion.argv[11], "select-ensemble-fusion")
            self.assertNotIn("--validation-partition", fusion.argv)
            self.assertEqual(
                fusion.outputs,
                (Path("release/fusion-selection.json"),),
            )
            self.assertEqual(
                fusion.generated_inputs,
                (Path("release/ensemble.json"),),
            )
            calibration = plan[ensemble_index + 2]
            self.assertEqual(
                calibration.argv[
                    calibration.argv.index("--fusion-selection") + 1
                ],
                str(Path("release/fusion-selection.json")),
            )
            self.assertEqual(
                calibration.argv[
                    calibration.argv.index(
                        "--fusion-selection-sha256"
                    )
                    + 1
                ],
                "<sha256:release/fusion-selection.json>",
            )
            transcript = run(workflow, digest, root, execute=False)
            self.assertIs(transcript["evidence"], False)
            self.assertEqual(transcript["workflowSha256"], digest)
            self.assertEqual(
                transcript["realDomainExecution"], "external-isolated-only"
            )
            self.assertFalse(any(
                "sealed_test" in " ".join(record["argv"])
                for record in transcript["steps"]
            ))

    def test_rejects_path_seed_epoch_and_schema_bypasses(self) -> None:
        mutations = {
            "absolute": lambda value: value["shared"].__setitem__(
                "dataset", _ref("C:/secret/test.ndjson")
            ),
            "traversal": lambda value: value["shared"].__setitem__(
                "dataset", _ref("../secret/test.ndjson")
            ),
            "backslash": lambda value: value["shared"].__setitem__(
                "dataset", _ref("release\\validation.ndjson")
            ),
            "duplicate-seed": lambda value: value["candidates"][1].__setitem__(
                "seed", 20260811
            ),
            "reordered-epoch": lambda value: value["candidates"][0]["epochs"][0].__setitem__(
                "epoch", 2
            ),
            "duplicate-path": lambda value: value["candidates"][0]["epochs"][1].__setitem__(
                "checkpoint", _ref("release/20260811/epoch-1.pt")
            ),
            "output-overwrites-input": lambda value: value[
                "selectionOutputs"
            ].__setitem__(
                "20260811", "release/validation.ndjson"
            ),
            "invalid-input-sha": lambda value: value["shared"].__setitem__(
                "dataset", _ref("release/validation.ndjson", "NOT-A-SHA")
            ),
            "real-domain-run": lambda value: value.__setitem__(
                "realDomainExecution", "run"
            ),
            "unknown-field": lambda value: value.__setitem__(
                "command", ["cmd.exe", "/c", "anything"]
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                value = _workflow()
                mutate(value)
                path = Path(directory) / "workflow.json"
                _write(path, value)
                with self.assertRaises(ReleaseWorkflowError):
                    workflow, _ = load_workflow(path)
                    build_plan(workflow, Path(directory))

    def test_rejects_noncanonical_manifest_and_symlink_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "workflow.json"
            path.write_text(json.dumps(_workflow(), indent=2), encoding="utf-8")
            with self.assertRaisesRegex(ReleaseWorkflowError, "canonical"):
                load_workflow(path)
            _write(path, _workflow())
            workflow, _ = load_workflow(path)
            with (
                patch.object(Path, "exists", return_value=True),
                patch.object(Path, "is_symlink", return_value=True),
            ):
                with self.assertRaisesRegex(ReleaseWorkflowError, "symlink"):
                    _confined(root, Path("release/input.json"), must_exist=False)

    def test_execute_preflights_all_outputs_before_spawning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "release").mkdir()
            workflow_path = root / "workflow.json"
            digest = _write(workflow_path, _workflow())
            workflow, _ = load_workflow(workflow_path)
            late_output = root / "release" / "parity-evidence.json"
            late_output.write_text("already exists", encoding="utf-8")
            with patch(
                "ml.evaluation.release_workflow.subprocess.run"
            ) as process:
                with self.assertRaisesRegex(
                    ReleaseWorkflowError, "outputs already exist"
                ):
                    run(workflow, digest, root, execute=True)
                process.assert_not_called()

    def test_executes_selection_evaluations_four_at_a_time_in_plan_order(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "release").mkdir()
            workflow_path = root / "workflow.json"
            digest = _write(workflow_path, _workflow())
            workflow, _ = load_workflow(workflow_path)
            lock = threading.Lock()
            active = 0
            maximum = 0

            def execute_step(step, *_args):
                nonlocal active, maximum
                if step.stage == "selection-fit-evaluation":
                    with lock:
                        active += 1
                        maximum = max(maximum, active)
                    time.sleep(0.02)
                    with lock:
                        active -= 1
                return (
                    step.argv,
                    {
                        path: "2" * 64
                        for path in step.generated_inputs
                    },
                    [
                        {
                            "path": path.as_posix(),
                            "sha256": "1" * 64,
                        }
                        for path in step.outputs
                    ],
                )

            external = {
                name: Path(value["path"]).resolve()
                for name, value in workflow["tools"].items()
            }

            def process(args, **_kwargs):
                if args[1:3] == ["rev-parse", "HEAD"]:
                    return SimpleNamespace(
                        stdout=workflow["sourceRevision"] + "\n"
                    )
                return SimpleNamespace(stdout="")

            with (
                patch(
                    "ml.evaluation.release_workflow._authenticate_external",
                    side_effect=lambda reference, _label: reference.path.resolve(),
                ),
                patch(
                    "ml.evaluation.release_workflow._closed_environment",
                    return_value={"PATH": "closed"},
                ),
                patch(
                    "ml.evaluation.release_workflow.shutil.which",
                    side_effect=lambda name, **_kwargs: str(external[name]),
                ),
                patch(
                    "ml.evaluation.release_workflow._input_references",
                    return_value=(),
                ),
                patch(
                    "ml.evaluation.release_workflow.subprocess.run",
                    side_effect=process,
                ),
                patch(
                    "ml.evaluation.release_workflow._execute_step",
                    side_effect=execute_step,
                ),
            ):
                transcript = run(
                    workflow,
                    digest,
                    root,
                    execute=True,
                )

            self.assertEqual(maximum, 4)
            selection = [
                (record["seed"], record["epoch"])
                for record in transcript["steps"]
                if record["stage"] == "selection-fit-evaluation"
            ]
            self.assertEqual(
                selection,
                [
                    (seed, epoch)
                    for seed in (20260811, 20260812, 20260813)
                    for epoch in range(1, 9)
                ],
            )

    def test_selection_failure_stops_queued_evaluations_promptly(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "release").mkdir()
            workflow_path = root / "workflow.json"
            digest = _write(workflow_path, _workflow())
            workflow, _ = load_workflow(workflow_path)
            called: list[tuple[int | None, int | None]] = []
            lock = threading.Lock()

            def execute_step(step, *_args):
                with lock:
                    called.append((step.seed, step.epoch))
                if (
                    step.stage == "selection-fit-evaluation"
                    and step.seed == 20260811
                    and step.epoch == 4
                ):
                    raise RuntimeError("later-plan failure")
                time.sleep(0.05)
                return (
                    step.argv,
                    {},
                    [
                        {
                            "path": path.as_posix(),
                            "sha256": "1" * 64,
                        }
                        for path in step.outputs
                    ],
                )

            def cancellation_first(futures):
                pending = tuple(futures)
                while not all(future.done() for future in pending):
                    time.sleep(0.001)
                return iter(sorted(
                    pending,
                    key=lambda future: not isinstance(
                        future.exception(), ReleaseWorkflowError
                    ),
                ))

            external = {
                name: Path(value["path"]).resolve()
                for name, value in workflow["tools"].items()
            }

            def process(args, **_kwargs):
                if args[1:3] == ["rev-parse", "HEAD"]:
                    return SimpleNamespace(
                        stdout=workflow["sourceRevision"] + "\n"
                    )
                return SimpleNamespace(stdout="")

            with (
                patch(
                    "ml.evaluation.release_workflow._authenticate_external",
                    side_effect=lambda reference, _label: reference.path.resolve(),
                ),
                patch(
                    "ml.evaluation.release_workflow._closed_environment",
                    return_value={"PATH": "closed"},
                ),
                patch(
                    "ml.evaluation.release_workflow.shutil.which",
                    side_effect=lambda name, **_kwargs: str(external[name]),
                ),
                patch(
                    "ml.evaluation.release_workflow._input_references",
                    return_value=(),
                ),
                patch(
                    "ml.evaluation.release_workflow.subprocess.run",
                    side_effect=process,
                ),
                patch(
                    "ml.evaluation.release_workflow._execute_step",
                    side_effect=execute_step,
                ),
                patch(
                    "ml.evaluation.release_workflow.as_completed",
                    side_effect=cancellation_first,
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "later-plan failure",
                ):
                    run(
                        workflow,
                        digest,
                        root,
                        execute=True,
                    )

            self.assertEqual(
                set(called),
                {
                    (20260811, 1),
                    (20260811, 2),
                    (20260811, 3),
                    (20260811, 4),
                },
            )

    def test_first_worker_failure_outranks_a_second_real_failure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "release").mkdir()
            workflow_path = root / "workflow.json"
            digest = _write(workflow_path, _workflow())
            workflow, _ = load_workflow(workflow_path)
            primary_recorded = threading.Event()
            original_record_failure = _SelectionWave.record_failure

            def record_failure(wave, error):
                original_record_failure(wave, error)
                if str(error) == "primary plan failure":
                    primary_recorded.set()

            def execute_step(step, *_args):
                if (
                    step.stage == "selection-fit-evaluation"
                    and step.seed == 20260811
                    and step.epoch == 1
                ):
                    raise RuntimeError("primary plan failure")
                if (
                    step.stage == "selection-fit-evaluation"
                    and step.seed == 20260811
                    and step.epoch == 2
                ):
                    if not primary_recorded.wait(timeout=1):
                        raise AssertionError("primary failure was not recorded")
                    raise RuntimeError("second plan failure")
                time.sleep(0.05)
                return (
                    step.argv,
                    {},
                    [
                        {
                            "path": path.as_posix(),
                            "sha256": "1" * 64,
                        }
                        for path in step.outputs
                    ],
                )

            def second_failure_first(futures):
                pending = tuple(futures)
                while not all(future.done() for future in pending):
                    time.sleep(0.001)
                return iter(sorted(
                    pending,
                    key=lambda future: str(future.exception())
                    != "second plan failure",
                ))

            def interrupt_after_primary(_futures):
                if not primary_recorded.wait(timeout=1):
                    raise AssertionError("primary failure was not recorded")
                raise KeyboardInterrupt("main-thread interruption")

            external = {
                name: Path(value["path"]).resolve()
                for name, value in workflow["tools"].items()
            }

            def process(args, **_kwargs):
                if args[1:3] == ["rev-parse", "HEAD"]:
                    return SimpleNamespace(
                        stdout=workflow["sourceRevision"] + "\n"
                    )
                return SimpleNamespace(stdout="")

            with (
                patch(
                    "ml.evaluation.release_workflow._authenticate_external",
                    side_effect=lambda reference, _label: reference.path.resolve(),
                ),
                patch(
                    "ml.evaluation.release_workflow._closed_environment",
                    return_value={"PATH": "closed"},
                ),
                patch(
                    "ml.evaluation.release_workflow.shutil.which",
                    side_effect=lambda name, **_kwargs: str(external[name]),
                ),
                patch(
                    "ml.evaluation.release_workflow._input_references",
                    return_value=(),
                ),
                patch(
                    "ml.evaluation.release_workflow.subprocess.run",
                    side_effect=process,
                ),
                patch(
                    "ml.evaluation.release_workflow._SelectionWave.record_failure",
                    autospec=True,
                    side_effect=record_failure,
                ),
                patch(
                    "ml.evaluation.release_workflow._execute_step",
                    side_effect=execute_step,
                ),
            ):
                with (
                    patch(
                        "ml.evaluation.release_workflow.as_completed",
                        side_effect=second_failure_first,
                    ),
                    self.assertRaisesRegex(
                        RuntimeError,
                        "primary plan failure",
                    ),
                ):
                    run(
                        workflow,
                        digest,
                        root,
                        execute=True,
                    )
                primary_recorded.clear()
                with (
                    patch(
                        "ml.evaluation.release_workflow.as_completed",
                        side_effect=interrupt_after_primary,
                    ),
                    self.assertRaisesRegex(
                        KeyboardInterrupt,
                        "main-thread interruption",
                    ),
                ):
                    run(
                        workflow,
                        digest,
                        root,
                        execute=True,
                    )

    def test_isolated_python_bootstrap_ignores_sitecustomize(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bad = root / "bad-marker"
            good = root / "good-marker"
            (root / "sitecustomize.py").write_text(
                "from pathlib import Path\n"
                f"Path({str(bad)!r}).write_text('bad')\n",
                encoding="utf-8",
            )
            (root / "approved_probe.py").write_text(
                "from pathlib import Path\n"
                f"Path({str(good)!r}).write_text('good')\n",
                encoding="utf-8",
            )
            workflow_path = root / "workflow.json"
            _write(workflow_path, _workflow())
            workflow, _ = load_workflow(workflow_path)
            command = list(build_plan(workflow, root)[0].argv[:11])
            command[10] = "approved_probe"
            subprocess.run(
                command,
                cwd=root,
                check=True,
                env=_closed_environment(root, {}),
            )
            self.assertTrue(good.is_file())
            self.assertFalse(bad.exists())
