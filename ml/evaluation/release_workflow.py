"""Closed, post-training release orchestration with no sealed-test access."""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import subprocess
import sys
import sysconfig
import threading
from typing import Mapping, Sequence


FORMAT = "drawbacktrainer-post-training-release-workflow"
TRANSCRIPT_FORMAT = "drawbacktrainer-release-workflow-transcript"
VERSION = 3
SEEDS = (20260811, 20260812, 20260813)
EPOCHS = tuple(range(1, 9))
SELECTION_EVALUATION_WORKERS = 4
SHA256 = frozenset("0123456789abcdef")
ISOLATED_MODULE_BOOTSTRAP = (
    "import runpy,sys;"
    "root,purelib,platlib,stdlib,dynlib,module=sys.argv[1:7];"
    "sys.path[:]=[root,purelib,platlib,stdlib,dynlib];"
    "sys.argv=[module,*sys.argv[7:]];"
    "runpy.run_module(module,run_name='__main__')"
)


class ReleaseWorkflowError(ValueError):
    pass


@dataclass(frozen=True)
class Step:
    stage: str
    argv: tuple[str, ...]
    outputs: tuple[Path, ...]
    inputs: tuple[ArtifactRef, ...] = ()
    generated_inputs: tuple[Path, ...] = ()
    seed: int | None = None
    epoch: int | None = None


@dataclass(frozen=True)
class ArtifactRef:
    path: Path
    sha256: str


@dataclass(frozen=True)
class ExternalRef:
    path: Path
    sha256: str


def _object(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
        raise ReleaseWorkflowError(f"{label} must be an object")
    return value


def _exact(value: Mapping[str, object], fields: set[str], label: str) -> None:
    if set(value) != fields:
        raise ReleaseWorkflowError(f"{label} fields are invalid")


def _pairs(items: Sequence[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in items:
        if key in value:
            raise ReleaseWorkflowError(
                f"workflow JSON repeats key {key!r}"
            )
        value[key] = item
    return value


def _constant(token: str) -> object:
    raise ReleaseWorkflowError(
        f"workflow JSON contains non-finite constant {token}"
    )


def _relative(value: object, label: str) -> Path:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or ":" in value
    ):
        raise ReleaseWorkflowError(f"{label} must be a normalized POSIX path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        raise ReleaseWorkflowError(f"{label} escapes the repository")
    normalized = pure.as_posix()
    if normalized != value:
        raise ReleaseWorkflowError(f"{label} is not normalized")
    return Path(*pure.parts)


def _digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in SHA256 for character in value)
    ):
        raise ReleaseWorkflowError(f"{label} must be a lowercase SHA-256")
    return value


def _reference(value: object, label: str) -> ArtifactRef:
    item = _object(value, label)
    _exact(item, {"path", "sha256"}, label)
    return ArtifactRef(
        path=_relative(item["path"], f"{label}.path"),
        sha256=_digest(item["sha256"], f"{label}.sha256"),
    )


def _external_reference(value: object, label: str) -> ExternalRef:
    item = _object(value, label)
    _exact(item, {"path", "sha256"}, label)
    raw_path = item["path"]
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise ReleaseWorkflowError(f"{label}.path must be absolute")
    return ExternalRef(
        path=Path(raw_path),
        sha256=_digest(item["sha256"], f"{label}.sha256"),
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _confined(root: Path, relative: Path, *, must_exist: bool) -> Path:
    candidate = root.joinpath(relative)
    current = root
    for part in relative.parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise ReleaseWorkflowError(f"workflow path traverses symlink: {relative}")
    resolved = candidate.resolve(strict=must_exist)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ReleaseWorkflowError(f"workflow path escapes repository: {relative}") from error
    return resolved


def load_workflow(path: Path) -> tuple[Mapping[str, object], str]:
    try:
        payload = path.read_bytes()
        value = _object(
            json.loads(
                payload,
                object_pairs_hook=_pairs,
                parse_constant=_constant,
            ),
            "workflow",
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReleaseWorkflowError(
            f"cannot load strict UTF-8 workflow JSON: {path}"
        ) from error
    _exact(value, {
        "format", "version", "sourceRevision", "realDomainExecution",
        "tools", "shared", "candidates", "selectionOutputs", "ensemble",
        "calibration", "trainingFrequency", "validation", "browserRelease",
        "transcriptOutput",
    }, "workflow")
    if value["format"] != FORMAT or value["version"] != VERSION:
        raise ReleaseWorkflowError("workflow identity is invalid")
    revision = value["sourceRevision"]
    if not isinstance(revision, str) or len(revision) != 40 or any(
        c not in "0123456789abcdef" for c in revision
    ):
        raise ReleaseWorkflowError("sourceRevision must be a full lowercase Git SHA")
    if value["realDomainExecution"] != "external-isolated-only":
        raise ReleaseWorkflowError(
            "real-domain execution is outside this runner and requires external isolation"
        )
    canonical = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode() + b"\n"
    if payload != canonical:
        raise ReleaseWorkflowError("workflow must be canonical JSON")
    return value, hashlib.sha256(payload).hexdigest()


def _path_fields(value: object, fields: set[str], label: str) -> dict[str, Path]:
    obj = _object(value, label)
    _exact(obj, fields, label)
    return {field: _relative(obj[field], f"{label}.{field}") for field in fields}


def build_plan(workflow: Mapping[str, object], root: Path) -> tuple[Step, ...]:
    shared_raw = _object(workflow["shared"], "shared")
    _exact(
        shared_raw,
        {"dataset", "publicRoot", "privateValidation"},
        "shared",
    )
    shared = {
        name: _reference(shared_raw[name], f"shared.{name}")
        for name in ("dataset", "publicRoot", "privateValidation")
    }
    candidates = workflow["candidates"]
    if not isinstance(candidates, list) or len(candidates) != 3:
        raise ReleaseWorkflowError("candidates must contain exactly three seeds")
    candidate_data: dict[
        int, tuple[ArtifactRef, list[Mapping[str, object]]]
    ] = {}
    candidate_paths: set[Path] = set()
    input_references: list[ArtifactRef] = list(shared.values())
    for index, raw in enumerate(candidates):
        item = _object(raw, f"candidate {index}")
        _exact(item, {"seed", "trainingRun", "epochs"}, f"candidate {index}")
        seed = item["seed"]
        if seed != SEEDS[index]:
            raise ReleaseWorkflowError("candidate seed order is invalid")
        training_run = _reference(
            item["trainingRun"], f"candidate {index}.trainingRun"
        )
        input_references.append(training_run)
        epochs = item["epochs"]
        if not isinstance(epochs, list) or len(epochs) != 8:
            raise ReleaseWorkflowError("candidate must contain eight epochs")
        for epoch_index, raw_epoch in enumerate(epochs, 1):
            epoch = _object(raw_epoch, "epoch")
            _exact(epoch, {"epoch", "checkpoint", "report", "summary"}, "epoch")
            if epoch["epoch"] != epoch_index:
                raise ReleaseWorkflowError("epoch order is invalid")
            checkpoint = _reference(
                epoch["checkpoint"],
                f"candidate {index}.epoch {epoch_index}.checkpoint",
            )
            input_references.append(checkpoint)
            for name in ("report", "summary"):
                candidate_path = _relative(epoch[name], f"epoch.{name}")
                if candidate_path in candidate_paths:
                    raise ReleaseWorkflowError(
                        "checkpoint/report/summary paths must be distinct"
                    )
                candidate_paths.add(candidate_path)
            if checkpoint.path in candidate_paths:
                raise ReleaseWorkflowError(
                    "checkpoint/report/summary paths must be distinct"
                )
            candidate_paths.add(checkpoint.path)
        candidate_data[seed] = (training_run, epochs)
    selection_raw = _object(workflow["selectionOutputs"], "selectionOutputs")
    _exact(selection_raw, {str(seed) for seed in SEEDS}, "selectionOutputs")
    selections = {
        seed: _relative(selection_raw[str(seed)], f"selectionOutputs.{seed}")
        for seed in SEEDS
    }
    if len(set(selections.values())) != len(SEEDS):
        raise ReleaseWorkflowError("selection output paths must be distinct")
    ensemble = _path_fields(workflow["ensemble"], {"output"}, "ensemble")
    fusion_selection = ensemble["output"].with_name("fusion-selection.json")
    calibration = _path_fields(
        workflow["calibration"],
        {"report", "sidecar", "receipt", "output"},
        "calibration",
    )
    frequency = _reference(
        workflow["trainingFrequency"], "trainingFrequency"
    )
    input_references.append(frequency)
    validation = _path_fields(
        workflow["validation"], {"report", "decision"}, "validation"
    )
    browser_release_raw = _object(
        workflow["browserRelease"], "browserRelease"
    )
    _exact(
        browser_release_raw,
        {"fixture", "artifact", "parityInput", "transcript", "evidence"},
        "browserRelease",
    )
    fixture = _reference(
        browser_release_raw["fixture"], "browserRelease.fixture"
    )
    input_references.append(fixture)
    browser_release = {
        name: _relative(
            browser_release_raw[name], f"browserRelease.{name}"
        )
        for name in ("artifact", "parityInput", "transcript", "evidence")
    }
    transcript = _relative(workflow["transcriptOutput"], "transcriptOutput")
    tools = _object(workflow["tools"], "tools")
    _exact(tools, {"browser", "git", "node", "pnpm"}, "tools")
    browser = _external_reference(tools["browser"], "tools.browser")
    _external_reference(tools["git"], "tools.git")
    _external_reference(tools["node"], "tools.node")
    _external_reference(tools["pnpm"], "tools.pnpm")

    def p(path: Path) -> str:
        return str(path)

    def digest(path: Path) -> str:
        absolute = _confined(root, path, must_exist=False)
        return _sha(absolute) if absolute.is_file() else f"<sha256:{path.as_posix()}>"

    purelib = sysconfig.get_path("purelib")
    platlib = sysconfig.get_path("platlib")
    stdlib = sysconfig.get_path("stdlib")
    if not purelib or not platlib or not stdlib:
        raise ReleaseWorkflowError("Python library paths are unavailable")
    configured_dynlib = sysconfig.get_config_var("DESTSHARED")
    dynlib = (
        str(configured_dynlib)
        if isinstance(configured_dynlib, str) and configured_dynlib
        else str(Path(stdlib).parent / "DLLs")
    )

    def command(module: str, *arguments: str) -> tuple[str, ...]:
        return (
            sys.executable,
            "-I",
            "-S",
            "-c",
            ISOLATED_MODULE_BOOTSTRAP,
            str(root.resolve()),
            purelib,
            platlib,
            stdlib,
            dynlib,
            module,
            *arguments,
        )

    steps: list[Step] = []
    for seed in SEEDS:
        training_run, epochs = candidate_data[seed]
        for epoch in epochs:
            number = int(epoch["epoch"])
            checkpoint = _reference(epoch["checkpoint"], "checkpoint")
            report = _relative(epoch["report"], "report")
            steps.append(Step(
                "selection-fit-evaluation",
                command("ml.evaluation.cli", p(checkpoint.path), p(shared["dataset"].path),
                 "--public-root", p(shared["publicRoot"].path), "--private-validation",
                 p(shared["privateValidation"].path), "--split", "validation",
                 "--validation-partition", "selection", "--output", p(report)),
                (report,),
                (checkpoint, *shared.values()),
                seed=seed,
                epoch=number,
            ))
        for epoch in epochs:
            number = int(epoch["epoch"])
            checkpoint = _reference(epoch["checkpoint"], "checkpoint")
            report = _relative(epoch["report"], "report")
            summary = _relative(epoch["summary"], "summary")
            steps.append(Step(
                "selection-summary",
                command("ml.evaluation.cli", "emit-selection-summary",
                 p(report), p(checkpoint.path), p(summary), "--training-seed", str(seed),
                 "--epoch", str(number), "--training-run", p(training_run.path),
                 "--training-run-sha256", training_run.sha256),
                (summary,),
                (checkpoint, training_run),
                (report,),
                seed,
                number,
            ))
    for seed in SEEDS:
        training_run, epochs = candidate_data[seed]
        argv = list(command(
            "ml.evaluation.cli", "select-epoch", p(selections[seed])
        ))
        for epoch in epochs:
            summary = _relative(epoch["summary"], "summary")
            argv += ["--summary", p(summary), "--summary-sha256", digest(summary)]
        steps.append(Step(
            "epoch-selection",
            tuple(argv),
            (selections[seed],),
            (),
            tuple(
                _relative(epoch["summary"], "summary")
                for epoch in epochs
            ),
            seed,
        ))
    argv = list(command(
        "ml.evaluation.cli",
        "create-ensemble-release",
        p(ensemble["output"]),
    ))
    for seed in SEEDS:
        training_run, _ = candidate_data[seed]
        argv += ["--selection", p(selections[seed]), "--selection-sha256", digest(selections[seed])]
        argv += ["--training-run", p(training_run.path), "--training-run-sha256", training_run.sha256]
    steps.append(Step(
        "ensemble-release",
        tuple(argv),
        (ensemble["output"],),
        tuple(
            candidate_data[seed][0] for seed in SEEDS
        ),
        tuple(selections[seed] for seed in SEEDS),
    ))
    steps.append(Step(
        "fusion-selection",
        command(
            "ml.evaluation.cli",
            "select-ensemble-fusion",
            p(ensemble["output"]),
            p(shared["dataset"].path),
            p(fusion_selection),
            "--ensemble-sha256",
            digest(ensemble["output"]),
            "--public-root",
            p(shared["publicRoot"].path),
            "--private-validation",
            p(shared["privateValidation"].path),
        ),
        (fusion_selection,),
        tuple(shared.values()),
        (ensemble["output"],),
    ))
    steps.append(Step(
        "calibration-evaluation",
        command("ml.evaluation.cli", "evaluate-ensemble-calibration",
         p(ensemble["output"]), p(shared["dataset"].path), "--ensemble-sha256",
         digest(ensemble["output"]), "--public-root", p(shared["publicRoot"].path),
         "--private-validation", p(shared["privateValidation"].path),
         "--fusion-selection", p(fusion_selection),
         "--fusion-selection-sha256", digest(fusion_selection), "--output",
         p(calibration["report"]), "--sidecar-output", p(calibration["sidecar"])),
        (calibration["report"], calibration["sidecar"]),
        tuple(shared.values()),
        (ensemble["output"], fusion_selection),
    ))
    steps.append(Step(
        "calibration-fit",
        command("ml.evaluation.cli", "fit-ensemble-calibration",
         p(calibration["sidecar"]), p(calibration["report"]), p(ensemble["output"]),
         p(calibration["receipt"]), p(calibration["output"]), "--sidecar-sha256",
         digest(calibration["sidecar"]), "--report-sha256", digest(calibration["report"]),
         "--ensemble-sha256", digest(ensemble["output"]),
         "--fusion-selection", p(fusion_selection),
         "--fusion-selection-sha256", digest(fusion_selection)),
        (calibration["receipt"], calibration["output"]),
        (),
        (
            calibration["sidecar"],
            calibration["report"],
            ensemble["output"],
            fusion_selection,
        ),
    ))
    steps.append(Step(
        "validation-gate",
        command("ml.evaluation.validation_gate", p(ensemble["output"]),
         p(calibration["output"]), p(frequency.path), p(shared["dataset"].path),
         "--ensemble-sha256", digest(ensemble["output"]), "--calibration-sha256",
         digest(calibration["output"]), "--training-frequency-sha256",
         frequency.sha256, "--public-root", p(shared["publicRoot"].path),
         "--private-validation", p(shared["privateValidation"].path), "--report-output",
         p(validation["report"]), "--decision-output", p(validation["decision"])),
        (validation["report"], validation["decision"]),
        (*shared.values(), frequency),
        (ensemble["output"], calibration["output"]),
    ))
    steps.append(Step(
        "browser-artifact",
        command("ml.evaluation.cli", "export-browser-ensemble",
         p(ensemble["output"]), p(calibration["output"]), p(browser_release["artifact"]),
         "--ensemble-sha256", digest(ensemble["output"]), "--calibration-sha256",
         digest(calibration["output"])),
        (browser_release["artifact"],),
        (),
        (ensemble["output"], calibration["output"]),
    ))
    steps.append(Step(
        "browser-parity-input",
        command("ml.evaluation.browser_parity_input", p(fixture.path),
         p(ensemble["output"]), p(calibration["output"]), p(browser_release["artifact"]),
         "--fixture-sha256", fixture.sha256, "--ensemble-sha256",
         digest(ensemble["output"]), "--calibration-sha256", digest(calibration["output"]),
         "--browser-artifact-sha256", digest(browser_release["artifact"]),
         "--repository", ".", "--output", p(browser_release["parityInput"])),
        (browser_release["parityInput"],),
        (fixture,),
        (
            ensemble["output"],
            calibration["output"],
            browser_release["artifact"],
        ),
    ))
    steps.append(Step(
        "browser-parity",
        command("ml.evaluation.browser_parity", "--repository", ".",
         "--browser", str(browser.path), "--browser-artifact", p(browser_release["artifact"]),
         "--calibration", p(calibration["output"]), "--input",
         p(browser_release["parityInput"]), "--input-sha256",
         digest(browser_release["parityInput"]), "--transcript-output",
         p(browser_release["transcript"]), "--evidence-output",
         p(browser_release["evidence"])),
        (browser_release["transcript"], browser_release["evidence"]),
        (),
        (
            browser_release["artifact"],
            calibration["output"],
            browser_release["parityInput"],
        ),
    ))
    input_paths = [reference.path for reference in input_references]
    output_paths = [
        *(output for step in steps for output in step.outputs),
        transcript,
    ]
    if len(set(input_paths)) != len(input_paths):
        raise ReleaseWorkflowError("input artifact paths must be distinct")
    if len(set(output_paths)) != len(output_paths):
        raise ReleaseWorkflowError("output artifact paths must be distinct")
    if set(input_paths).intersection(output_paths):
        raise ReleaseWorkflowError(
            "release outputs must not overwrite input artifacts"
        )
    for declared_path in (*input_paths, *output_paths):
        _confined(root, declared_path, must_exist=False)
    # Real-domain Stage A/B intentionally remain outside this process. Their
    # mandatory no-label mount receipt is an external isolation boundary.
    return tuple(steps)


def _input_references(workflow: Mapping[str, object]) -> tuple[ArtifactRef, ...]:
    shared = _object(workflow["shared"], "shared")
    references = [
        _reference(shared[name], f"shared.{name}")
        for name in ("dataset", "publicRoot", "privateValidation")
    ]
    candidates = workflow["candidates"]
    if not isinstance(candidates, list):
        raise ReleaseWorkflowError("candidates are unavailable")
    for index, raw in enumerate(candidates):
        candidate = _object(raw, f"candidate {index}")
        references.append(
            _reference(
                candidate["trainingRun"],
                f"candidate {index}.trainingRun",
            )
        )
        epochs = candidate["epochs"]
        if not isinstance(epochs, list):
            raise ReleaseWorkflowError("candidate epochs are unavailable")
        for epoch_index, raw_epoch in enumerate(epochs, 1):
            epoch = _object(raw_epoch, f"candidate {index}.epoch {epoch_index}")
            references.append(
                _reference(
                    epoch["checkpoint"],
                    f"candidate {index}.epoch {epoch_index}.checkpoint",
                )
            )
    references.append(
        _reference(workflow["trainingFrequency"], "trainingFrequency")
    )
    browser_release = _object(
        workflow["browserRelease"], "browserRelease"
    )
    references.append(
        _reference(
            browser_release["fixture"], "browserRelease.fixture"
        )
    )
    return tuple(references)


def _external_tools(
    workflow: Mapping[str, object],
) -> Mapping[str, ExternalRef]:
    tools = _object(workflow["tools"], "tools")
    return {
        name: _external_reference(tools[name], f"tools.{name}")
        for name in ("browser", "git", "node", "pnpm")
    }


def _authenticate_input(root: Path, reference: ArtifactRef) -> Path:
    try:
        absolute = _confined(root, reference.path, must_exist=True)
    except OSError as error:
        raise ReleaseWorkflowError(
            f"release input is missing: {reference.path}"
        ) from error
    if (
        not absolute.is_file()
        or absolute.is_symlink()
        or _sha(absolute) != reference.sha256
    ):
        raise ReleaseWorkflowError(
            f"release input authentication failed: {reference.path}"
        )
    return absolute


def _authenticate_external(reference: ExternalRef, label: str) -> Path:
    try:
        path = reference.path.resolve(strict=True)
    except OSError as error:
        raise ReleaseWorkflowError(f"{label} is missing") from error
    if not path.is_file() or _sha(path) != reference.sha256:
        raise ReleaseWorkflowError(f"{label} authentication failed")
    return path


def _authenticate_generated(root: Path, relative: Path) -> tuple[Path, str]:
    try:
        path = _confined(root, relative, must_exist=True)
    except OSError as error:
        raise ReleaseWorkflowError(
            f"generated release input is missing: {relative}"
        ) from error
    if not path.is_file() or path.is_symlink():
        raise ReleaseWorkflowError(
            f"generated release input is invalid: {relative}"
        )
    return path, _sha(path)


def _closed_environment(
    root: Path,
    tools: Mapping[str, Path],
) -> Mapping[str, str]:
    system_root = os.environ.get("SystemRoot", "")
    path_entries = {
        str(path.parent) for path in tools.values()
    }
    if system_root:
        path_entries.add(str(Path(system_root) / "System32"))
    return {
        "PATH": os.pathsep.join(sorted(path_entries)),
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": "",
        **(
            {"PATHEXT": os.environ["PATHEXT"]}
            if "PATHEXT" in os.environ
            else {}
        ),
        "TEMP": os.environ.get("TEMP", str(root)),
        "TMP": os.environ.get("TMP", str(root)),
        **({"SystemRoot": system_root} if system_root else {}),
    }


def run(
    workflow: Mapping[str, object],
    workflow_sha: str,
    root: Path,
    *,
    execute: bool,
) -> Mapping[str, object]:
    plan = build_plan(workflow, root)
    records: list[dict[str, object]] = []
    transcript_path = _confined(
        root,
        _relative(workflow["transcriptOutput"], "transcriptOutput"),
        must_exist=False,
    )
    all_outputs = [
        *(
            _confined(root, output, must_exist=False)
            for step in plan
            for output in step.outputs
        ),
        transcript_path,
    ]
    if execute:
        if any(path.exists() for path in all_outputs):
            raise ReleaseWorkflowError(
                "one or more declared release outputs already exist"
            )
        external = {
            name: _authenticate_external(reference, f"external {name}")
            for name, reference in _external_tools(workflow).items()
        }
        environment = _closed_environment(root, external)
        for executable_name in ("git", "node", "pnpm"):
            resolved = shutil.which(
                executable_name,
                path=environment["PATH"],
            )
            if (
                resolved is None
                or Path(resolved).resolve() != external[executable_name]
            ):
                raise ReleaseWorkflowError(
                    f"closed PATH resolves the wrong {executable_name}"
                )
        revision = subprocess.run(
            [str(external["git"]), "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True, text=True,
            env=environment,
        ).stdout.strip()
        if revision != workflow["sourceRevision"]:
            raise ReleaseWorkflowError("source revision differs from workflow")
        dirty = subprocess.run(
            [
                str(external["git"]),
                "status",
                "--porcelain",
                "--untracked-files=all",
            ],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        ).stdout
        if dirty:
            raise ReleaseWorkflowError(
                "tracked source changes are present during release execution"
            )
        for reference in _input_references(workflow):
            _authenticate_input(root, reference)
    else:
        environment = {}
    parallel_selection: dict[
        tuple[int, int],
        tuple[tuple[str, ...], dict[Path, str], list[dict[str, str]]],
    ] = {}
    if execute:
        selection_steps = tuple(
            step
            for step in plan
            if step.stage == "selection-fit-evaluation"
        )
        futures: dict[
            Future[
                tuple[
                    tuple[str, ...],
                    dict[Path, str],
                    list[dict[str, str]],
                ]
            ],
            tuple[int, int],
        ] = {}
        stop = threading.Event()
        executor = ThreadPoolExecutor(
            max_workers=SELECTION_EVALUATION_WORKERS,
            thread_name_prefix="selection-evaluation",
        )
        try:
            for step in selection_steps:
                assert step.seed is not None
                assert step.epoch is not None
                future = executor.submit(
                    _execute_selection_step,
                    step,
                    workflow,
                    root,
                    environment,
                    stop,
                )
                futures[future] = (step.seed, step.epoch)
            for future in as_completed(futures):
                parallel_selection[futures[future]] = future.result()
        except BaseException:
            stop.set()
            for future in futures:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)
    for step in plan:
        output_paths = [_confined(root, item, must_exist=False) for item in step.outputs]
        if execute and any(path.exists() for path in output_paths):
            if step.stage != "selection-fit-evaluation":
                raise ReleaseWorkflowError(f"{step.stage} output already exists")
        recorded_argv = step.argv
        if execute:
            if step.stage == "selection-fit-evaluation":
                assert step.seed is not None
                assert step.epoch is not None
                recorded_argv, generated_before, outputs = (
                    parallel_selection[(step.seed, step.epoch)]
                )
            else:
                recorded_argv, generated_before, outputs = _execute_step(
                    step,
                    workflow,
                    root,
                    environment,
                )
        else:
            outputs = [{"path": rel.as_posix(), "sha256": None} for rel in step.outputs]
            generated_before = {
                relative: None for relative in step.generated_inputs
            }
        records.append({
            "stage": step.stage, "seed": step.seed, "epoch": step.epoch,
            "argv": list(recorded_argv),
            "inputs": [
                {
                    "path": reference.path.as_posix(),
                    "sha256": reference.sha256,
                }
                for reference in step.inputs
            ] + [
                {
                    "path": relative.as_posix(),
                    "sha256": generated_before[relative],
                }
                for relative in step.generated_inputs
            ],
            "outputs": outputs,
        })
    transcript = {
        "format": TRANSCRIPT_FORMAT, "version": VERSION, "evidence": False,
        "workflowSha256": workflow_sha,
        "sourceRevision": workflow["sourceRevision"],
        "mode": "execute" if execute else "dry-run",
        "realDomainExecution": "external-isolated-only",
        "externalTools": {
            name: {
                "path": str(reference.path),
                "sha256": reference.sha256,
            }
            for name, reference in _external_tools(workflow).items()
        },
        "steps": records,
    }
    if execute:
        relative = _relative(
            workflow["transcriptOutput"], "transcriptOutput"
        )
        output = transcript_path
        rendered = json.dumps(
            transcript, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode() + b"\n"
        output.parent.mkdir(parents=False, exist_ok=True)
        output.write_bytes(rendered)
    return transcript


def _execute_selection_step(
    step: Step,
    workflow: Mapping[str, object],
    root: Path,
    environment: Mapping[str, str],
    stop: threading.Event,
) -> tuple[tuple[str, ...], dict[Path, str], list[dict[str, str]]]:
    if stop.is_set():
        raise ReleaseWorkflowError(
            "selection evaluation wave was cancelled"
        )
    try:
        return _execute_step(
            step,
            workflow,
            root,
            environment,
        )
    except BaseException:
        stop.set()
        raise


def _execute_step(
    step: Step,
    workflow: Mapping[str, object],
    root: Path,
    environment: Mapping[str, str],
) -> tuple[tuple[str, ...], dict[Path, str], list[dict[str, str]]]:
    output_paths = [
        _confined(root, item, must_exist=False)
        for item in step.outputs
    ]
    if any(path.exists() for path in output_paths):
        raise ReleaseWorkflowError(f"{step.stage} output already exists")
    for reference in step.inputs:
        _authenticate_input(root, reference)
    generated_before = {
        relative: _authenticate_generated(root, relative)[1]
        for relative in step.generated_inputs
    }
    # Rebuild immediately before execution so digests of prior outputs
    # replace dry-run placeholders.
    current = next(
        item
        for item in build_plan(workflow, root)
        if (
            item.stage == step.stage
            and item.seed == step.seed
            and item.epoch == step.epoch
        )
    )
    subprocess.run(
        current.argv,
        cwd=root,
        check=True,
        shell=False,
        env=environment,
    )
    for reference in step.inputs:
        _authenticate_input(root, reference)
    for relative, expected_sha256 in generated_before.items():
        if (
            _authenticate_generated(root, relative)[1]
            != expected_sha256
        ):
            raise ReleaseWorkflowError(
                f"generated input changed during {step.stage}: "
                f"{relative}"
            )
    if any(
        not path.is_file() or path.is_symlink()
        for path in output_paths
    ):
        raise ReleaseWorkflowError(
            f"{step.stage} did not publish declared outputs"
        )
    outputs = [
        {"path": rel.as_posix(), "sha256": _sha(path)}
        for rel, path in zip(step.outputs, output_paths, strict=True)
    ]
    return current.argv, generated_before, outputs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ml.evaluation.release_workflow")
    parser.add_argument("workflow", type=Path)
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    root = args.repository.resolve(strict=True)
    workflow, workflow_sha = load_workflow(args.workflow)
    print(json.dumps(run(workflow, workflow_sha, root, execute=args.execute),
                     sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
