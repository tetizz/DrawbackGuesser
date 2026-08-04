"""Authenticated continuation of a release after epoch selection completed."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
from typing import Iterator, Mapping, Sequence

from ml.training.drawback_ml.durable_publish import publish_bytes_durable_exact
from ml.training.drawback_ml.path_validation import is_portable_safe_basename
from ml.training.drawback_ml.symbolic_schema import (
    SYMBOLIC_FEATURE_VERSION,
    SYMBOLIC_RULE_IDS,
)

from .calibration_release import canonical_symbolic_schema_sha256
from .ensemble_calibration import ContentAddressedFile, load_ensemble_calibration
from .ensemble_release import verify_ensemble_release
from .fusion_selection import (
    FusionSelectionIdentity,
    load_fusion_selection_artifact,
)
from .release_selection_bundle import (
    ContentAddressedJson,
    verify_release_selection_bundle,
)
from .release_workflow import (
    ExternalRef,
    GIT_TIMEOUT_SECONDS,
    ReleaseWorkflowError,
    _authenticate_external,
    _closed_environment,
    _confined,
    _execute_step,
    _external_tools,
    _hardened_git_command,
    _input_references,
    _preflight_recursive_git_filters,
    _run_capture_process,
    _stable_file_identity,
    build_plan,
    load_workflow,
)


FORMAT = "drawbacktrainer-post-selection-continuation"
VERSION = 1
RESUME_VERSION = 3
SEEDS = (20260811, 20260812, 20260813)
FIRST_STAGE = "ensemble-release"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextmanager
def _stable_external_git(
    reference: ExternalRef,
    authenticated: Path,
) -> Iterator[None]:
    identity = _stable_file_identity(authenticated)

    def reauthenticate() -> None:
        if (
            _authenticate_external(reference, "external git") != authenticated
            or _stable_file_identity(authenticated) != identity
        ):
            raise ReleaseWorkflowError(
                "external git identity changed during source authentication"
            )

    try:
        yield
    except BaseException as primary:
        try:
            reauthenticate()
        except BaseException as authentication_error:
            primary.add_note(
                "external Git reauthentication also failed: "
                f"{authentication_error!r}"
            )
        raise
    reauthenticate()


def _relative(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ReleaseWorkflowError(f"{label} must be a normalized POSIX path")
    path = Path(*value.split("/"))
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ReleaseWorkflowError(f"{label} escapes the repository")
    if path.as_posix() != value:
        raise ReleaseWorkflowError(f"{label} is not normalized")
    return path


def _digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ReleaseWorkflowError(f"{label} must be a lowercase SHA-256")
    return value


def _revision(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ReleaseWorkflowError(f"{label} must be a lowercase Git revision")
    return value


def _strict_json(path: Path) -> tuple[Mapping[str, object], str]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ReleaseWorkflowError(f"cannot read strict JSON: {path}") from error

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in items:
            if key in value:
                raise ReleaseWorkflowError(f"JSON repeats key {key!r}")
            value[key] = item
        return value

    def constant(token: str) -> object:
        raise ReleaseWorkflowError(
            f"strict JSON contains non-finite constant {token}"
        )

    try:
        value = json.loads(
            payload,
            object_pairs_hook=pairs,
            parse_constant=constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReleaseWorkflowError(f"invalid strict JSON: {path}") from error
    if not isinstance(value, dict):
        raise ReleaseWorkflowError("continuation manifest must be an object")
    return value, hashlib.sha256(payload).hexdigest()


def load_manifest(path: Path) -> tuple[Mapping[str, object], str]:
    value, digest = _strict_json(path)
    base_fields = {
        "format",
        "version",
        "sourceWorkflow",
        "executionSourceRevision",
        "archiveRoot",
        "outputRoot",
        "selectionSha256",
        "transcriptOutput",
    }
    version = value.get("version")
    fields = (
        base_fields
        if version == VERSION
        else base_fields | {"resumeAfterCalibration"}
    )
    if set(value) != fields:
        raise ReleaseWorkflowError("continuation manifest fields are invalid")
    if value["format"] != FORMAT or version not in {VERSION, RESUME_VERSION}:
        raise ReleaseWorkflowError("continuation manifest format is unsupported")
    if version == RESUME_VERSION:
        resume = value["resumeAfterCalibration"]
        resume_fields = {
            "ensembleSha256",
            "fusionSelectionSha256",
            "calibrationReportSha256",
            "calibrationSidecarSha256",
            "calibrationReceiptSha256",
            "calibrationSha256",
        }
        if not isinstance(resume, dict) or set(resume) != resume_fields:
            raise ReleaseWorkflowError(
                "resumeAfterCalibration fields are invalid"
            )
        for name in resume_fields:
            _digest(resume[name], f"resumeAfterCalibration.{name}")
    source = value["sourceWorkflow"]
    if not isinstance(source, dict) or set(source) != {"path", "sha256"}:
        raise ReleaseWorkflowError("sourceWorkflow fields are invalid")
    _relative(source["path"], "sourceWorkflow.path")
    _digest(source["sha256"], "sourceWorkflow.sha256")
    _revision(
        value["executionSourceRevision"],
        "executionSourceRevision",
    )
    _relative(value["archiveRoot"], "archiveRoot")
    _relative(value["outputRoot"], "outputRoot")
    _relative(value["transcriptOutput"], "transcriptOutput")
    selections = value["selectionSha256"]
    if (
        not isinstance(selections, dict)
        or set(selections) != {str(seed) for seed in SEEDS}
    ):
        raise ReleaseWorkflowError("selectionSha256 identities are invalid")
    for seed in SEEDS:
        _digest(selections[str(seed)], f"selectionSha256.{seed}")
    return value, digest


def _exclusive_copy(source: Path, destination: Path, expected: str) -> None:
    try:
        payload = source.read_bytes()
    except OSError as error:
        raise ReleaseWorkflowError(
            f"cannot read continuation evidence: {source}"
        ) from error
    if source.is_symlink() or hashlib.sha256(payload).hexdigest() != expected:
        raise ReleaseWorkflowError(
            f"reused evidence authentication failed: {source}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        publish_bytes_durable_exact(
            destination,
            payload,
            label="continuation output",
        )
    except (OSError, ValueError) as error:
        raise ReleaseWorkflowError(
            f"cannot publish continuation output: {destination}"
        ) from error


def _authenticate_reused_file(source: Path, expected: str) -> None:
    if (
        not source.is_file()
        or source.is_symlink()
        or _sha(source) != expected
    ):
        raise ReleaseWorkflowError(
            f"reused evidence authentication failed: {source}"
        )


def _basename(value: object, label: str) -> str:
    if not is_portable_safe_basename(value):
        raise ReleaseWorkflowError(f"{label} must be a safe basename")
    return value


def _archived_file(
    archive: Path,
    relative: Path,
    label: str,
) -> Path:
    try:
        path = _confined(archive, relative, must_exist=True)
    except OSError as error:
        raise ReleaseWorkflowError(
            f"{label} is missing from the archive"
        ) from error
    if not path.is_file() or path.is_symlink():
        raise ReleaseWorkflowError(
            f"{label} must be a real archived file"
        )
    return path


def _stage_seed(
    root: Path,
    archive: Path,
    staging: Path,
    candidate: Mapping[str, object],
    selection_sha: str,
    *,
    reuse_existing: bool = False,
) -> list[dict[str, str]]:
    seed = candidate["seed"]
    assert isinstance(seed, int)
    source_relative = Path(str(seed))
    source_dir = _confined(
        archive,
        source_relative,
        must_exist=True,
    )
    if not source_dir.is_dir() or source_dir.is_symlink():
        raise ReleaseWorkflowError(
            f"archived seed {seed} must be a real directory"
        )
    target_dir = staging / str(seed)
    selection_source = _archived_file(
        archive,
        source_relative / "selection.json",
        "selection",
    )
    selection_value, actual_selection_sha = _strict_json(selection_source)
    if actual_selection_sha != selection_sha:
        raise ReleaseWorkflowError(f"selection authentication failed for seed {seed}")
    evidence: list[dict[str, str]] = []
    epochs = candidate["epochs"]
    if not isinstance(epochs, list):
        raise ReleaseWorkflowError("source workflow epochs are invalid")
    selection_candidates = selection_value.get("candidates")
    if (
        not isinstance(selection_candidates, list)
        or len(selection_candidates) != len(epochs)
    ):
        raise ReleaseWorkflowError("selection candidate count is invalid")
    for epoch, selected in zip(epochs, selection_candidates, strict=True):
        if not isinstance(epoch, dict) or not isinstance(selected, dict):
            raise ReleaseWorkflowError("selection candidate is invalid")
        checkpoint = epoch["checkpoint"]
        if not isinstance(checkpoint, dict):
            raise ReleaseWorkflowError("checkpoint reference is invalid")
        checkpoint_source = root / _relative(
            checkpoint["path"],
            "checkpoint.path",
        )
        checkpoint_sha = _digest(
            checkpoint["sha256"],
            "checkpoint.sha256",
        )
        _authenticate_reused_file(
            checkpoint_source,
            checkpoint_sha,
        )
        evidence.append({
            "kind": "checkpoint",
            "source": checkpoint_source.relative_to(root).as_posix(),
            "sha256": checkpoint_sha,
        })
        bindings = (
            (
                _archived_file(
                    archive,
                    source_relative / _basename(
                        selected["evaluation_report_file"],
                        "evaluation report file",
                    ),
                    "selection report",
                ),
                target_dir / _basename(
                    selected["evaluation_report_file"], "evaluation report file"
                ),
                _digest(selected["evaluation_report_sha256"], "report sha256"),
                "selection-report",
            ),
            (
                _archived_file(
                    archive,
                    source_relative / _basename(
                        selected["summary_file"],
                        "summary file",
                    ),
                    "selection summary",
                ),
                target_dir / _basename(selected["summary_file"], "summary file"),
                _digest(selected["summary_sha256"], "summary sha256"),
                "selection-summary",
            ),
        )
        for source, destination, expected, kind in bindings:
            if reuse_existing:
                _authenticate_reused_file(source, expected)
                _authenticate_reused_file(destination, expected)
            else:
                _exclusive_copy(source, destination, expected)
            evidence.append({
                "kind": kind,
                "source": source.relative_to(root).as_posix(),
                "staged": destination.relative_to(root).as_posix(),
                "sha256": expected,
            })
    selection_target = target_dir / "selection.json"
    if reuse_existing:
        _authenticate_reused_file(selection_source, selection_sha)
        _authenticate_reused_file(selection_target, selection_sha)
    else:
        _exclusive_copy(selection_source, selection_target, selection_sha)
    evidence.append({
        "kind": "selection",
        "source": selection_source.relative_to(root).as_posix(),
        "staged": selection_target.relative_to(root).as_posix(),
        "sha256": selection_sha,
    })
    training = candidate["trainingRun"]
    if not isinstance(training, dict):
        raise ReleaseWorkflowError("trainingRun reference is invalid")
    verify_release_selection_bundle(
        ContentAddressedJson(selection_target, selection_sha),
        ContentAddressedJson(
            root / _relative(training["path"], "trainingRun.path"),
            _digest(training["sha256"], "trainingRun.sha256"),
        ),
    )
    if reuse_existing:
        expected_names = {"selection.json"}
        for selected in selection_candidates:
            assert isinstance(selected, dict)
            expected_names.add(
                _basename(
                    selected["evaluation_report_file"],
                    "evaluation report file",
                )
            )
            expected_names.add(
                _basename(selected["summary_file"], "summary file")
            )
        if (
            not target_dir.is_dir()
            or target_dir.is_symlink()
            or {item.name for item in target_dir.iterdir()} != expected_names
        ):
            raise ReleaseWorkflowError(
                f"staged seed {seed} is not the exact selection boundary"
            )
    return evidence


def _continued_workflow(
    source: Mapping[str, object],
    staging: Path,
    output: Path,
    transcript: Path,
) -> dict[str, object]:
    value = json.loads(json.dumps(source))
    assert isinstance(value, dict)
    candidates = value["candidates"]
    assert isinstance(candidates, list)
    for candidate in candidates:
        assert isinstance(candidate, dict)
        seed = candidate["seed"]
        assert isinstance(seed, int)
        directory = staging / str(seed)
        epochs = candidate["epochs"]
        assert isinstance(epochs, list)
        for epoch in epochs:
            assert isinstance(epoch, dict)
            number = epoch["epoch"]
            epoch["report"] = (
                directory / f"epoch-{number}.selection-report.json"
            ).as_posix()
            epoch["summary"] = (
                directory / f"epoch-{number}.selection-summary.json"
            ).as_posix()
    value["selectionOutputs"] = {
        str(seed): (staging / str(seed) / "selection.json").as_posix()
        for seed in SEEDS
    }
    training_frequency = value["trainingFrequency"]
    assert isinstance(training_frequency, dict)
    training_frequency["path"] = (
        output / "training-frequency.json"
    ).as_posix()
    value["ensemble"] = {"output": (output / "ensemble.json").as_posix()}
    value["calibration"] = {
        "report": (output / "calibration-report.json").as_posix(),
        "sidecar": (output / "calibration-sidecar.ndjson").as_posix(),
        "receipt": (output / "calibration-receipt.json").as_posix(),
        "output": (output / "calibration.json").as_posix(),
    }
    value["validation"] = {
        "report": (output / "validation-report.json").as_posix(),
        "decision": (output / "validation-decision.json").as_posix(),
    }
    value["browserRelease"] = {
        **value["browserRelease"],
        "artifact": (output / "browser-model.json").as_posix(),
        "parityInput": (output / "parity-input.json").as_posix(),
        "transcript": (output / "parity-transcript.json").as_posix(),
        "evidence": (output / "parity-evidence.json").as_posix(),
    }
    value["transcriptOutput"] = transcript.as_posix()
    return value


def _authenticate_calibration_prefix(
    root: Path,
    output: Path,
    resume: Mapping[str, object],
) -> list[dict[str, object]]:
    bindings = {
        "ensemble-release": (
            output / "ensemble.json",
            _digest(resume["ensembleSha256"], "ensembleSha256"),
        ),
        "fusion-selection": (
            output / "fusion-selection.json",
            _digest(
                resume["fusionSelectionSha256"],
                "fusionSelectionSha256",
            ),
        ),
        "calibration-report": (
            output / "calibration-report.json",
            _digest(
                resume["calibrationReportSha256"],
                "calibrationReportSha256",
            ),
        ),
        "calibration-sidecar": (
            output / "calibration-sidecar.ndjson",
            _digest(
                resume["calibrationSidecarSha256"],
                "calibrationSidecarSha256",
            ),
        ),
        "calibration-receipt": (
            output / "calibration-receipt.json",
            _digest(
                resume["calibrationReceiptSha256"],
                "calibrationReceiptSha256",
            ),
        ),
        "calibration": (
            output / "calibration.json",
            _digest(resume["calibrationSha256"], "calibrationSha256"),
        ),
    }
    for path, expected in bindings.values():
        _authenticate_reused_file(path, expected)
    try:
        ensemble = verify_ensemble_release(
            ContentAddressedJson(*bindings["ensemble-release"])
        )
        fusion_selection = load_fusion_selection_artifact(
            ContentAddressedJson(*bindings["fusion-selection"]),
            expected_identity=FusionSelectionIdentity(
                ensemble_release_sha256=bindings["ensemble-release"][1],
                private_validation_manifest_sha256=(
                    ensemble.private_validation_manifest_sha256
                ),
                validation_dataset_sha256=(
                    ensemble.validation_dataset_sha256
                ),
                validation_seed_sha256=ensemble.partition_seed_sha256,
                training_corpus_set_sha256=(
                    ensemble.training_corpus_set_sha256
                ),
                symbolic_schema_sha256=canonical_symbolic_schema_sha256(
                    SYMBOLIC_FEATURE_VERSION,
                    SYMBOLIC_RULE_IDS,
                ),
            ),
        )
        calibration = load_ensemble_calibration(
            ContentAddressedFile(*bindings["calibration"])
        )
    except (OSError, ValueError) as error:
        raise ReleaseWorkflowError(
            "resume calibration prefix failed recursive verification"
        ) from error
    expected_bindings = {
        "ensemble_release": {
            "file": bindings["ensemble-release"][0].name,
            "sha256": bindings["ensemble-release"][1],
        },
        "report": {
            "file": bindings["calibration-report"][0].name,
            "sha256": bindings["calibration-report"][1],
        },
        "sidecar": {
            "file": bindings["calibration-sidecar"][0].name,
            "sha256": bindings["calibration-sidecar"][1],
        },
        "receipt": {
            "file": bindings["calibration-receipt"][0].name,
            "sha256": bindings["calibration-receipt"][1],
        },
    }
    if any(
        calibration.get(name) != expected
        for name, expected in expected_bindings.items()
    ):
        raise ReleaseWorkflowError(
            "resume calibration prefix bindings disagree with the manifest"
        )
    calibration_identity = calibration.get("identity")
    if (
        not isinstance(calibration_identity, Mapping)
        or calibration_identity.get("fusion_selection_sha256")
        != bindings["fusion-selection"][1]
        or calibration_identity.get("selected_alpha")
        != fusion_selection.selected_alpha
    ):
        raise ReleaseWorkflowError(
            "resume calibration prefix fusion binding disagrees with the manifest"
        )

    def output_record(name: str) -> dict[str, str]:
        path, digest = bindings[name]
        return {
            "path": path.relative_to(root).as_posix(),
            "sha256": digest,
        }

    return [
        {
            "stage": "ensemble-release",
            "status": "reused-authenticated",
            "outputs": [output_record("ensemble-release")],
        },
        {
            "stage": "fusion-selection",
            "status": "reused-authenticated",
            "outputs": [output_record("fusion-selection")],
        },
        {
            "stage": "calibration-evaluation",
            "status": "reused-authenticated",
            "outputs": [
                output_record("calibration-report"),
                output_record("calibration-sidecar"),
            ],
        },
        {
            "stage": "calibration-fit",
            "status": "reused-authenticated",
            "outputs": [
                output_record("calibration-receipt"),
                output_record("calibration"),
            ],
        },
    ]


def run(
    manifest: Mapping[str, object],
    manifest_sha: str,
    root: Path,
    *,
    execute: bool,
) -> Mapping[str, object]:
    source_ref = manifest["sourceWorkflow"]
    assert isinstance(source_ref, dict)
    source_path = _confined(
        root, _relative(source_ref["path"], "sourceWorkflow.path"), must_exist=True
    )
    expected_source_sha = _digest(source_ref["sha256"], "sourceWorkflow.sha256")
    source_workflow, actual_source_sha = load_workflow(source_path)
    if actual_source_sha != expected_source_sha:
        raise ReleaseWorkflowError("source workflow authentication failed")
    archive = _confined(
        root, _relative(manifest["archiveRoot"], "archiveRoot"), must_exist=True
    )
    if not archive.is_dir() or archive.is_symlink():
        raise ReleaseWorkflowError(
            "archiveRoot must be a real non-symlink directory"
        )
    output = _relative(manifest["outputRoot"], "outputRoot")
    staging = output / "reused-selection-inputs"
    transcript_path = _relative(manifest["transcriptOutput"], "transcriptOutput")
    if (
        len(output.parts) < 3
        or output.parts[:2] != ("data", "generated")
        or ".git" in output.parts
    ):
        raise ReleaseWorkflowError(
            "outputRoot must be a fresh child of data/generated"
        )
    if not transcript_path.is_relative_to(output):
        raise ReleaseWorkflowError("transcriptOutput must be inside outputRoot")
    absolute_output = _confined(root, output, must_exist=False)
    resume = manifest.get("resumeAfterCalibration")
    resume_mode = isinstance(resume, dict)
    if (
        absolute_output.is_relative_to(archive)
        or archive.is_relative_to(absolute_output)
        or source_path.is_relative_to(absolute_output)
    ):
        raise ReleaseWorkflowError(
            "outputRoot must not overlap archived or source evidence"
        )
    if execute and absolute_output.exists() and not resume_mode:
        raise ReleaseWorkflowError("fresh continuation outputRoot already exists")
    if execute and resume_mode:
        if not absolute_output.is_dir() or absolute_output.is_symlink():
            raise ReleaseWorkflowError(
                "resume outputRoot must be an existing real directory"
            )
        required = {
            "reused-selection-inputs",
            "ensemble.json",
            "fusion-selection.json",
            "calibration-report.json",
            "calibration-sidecar.ndjson",
            "calibration-receipt.json",
            "calibration.json",
        }
        actual = frozenset(item.name for item in absolute_output.iterdir())
        if actual not in {
            frozenset(required),
            frozenset(required | {"training-frequency.json"}),
        }:
            raise ReleaseWorkflowError(
                "resume outputRoot is not the exact calibration-fit boundary"
            )
        staged_root = absolute_output / "reused-selection-inputs"
        if (
            not staged_root.is_dir()
            or staged_root.is_symlink()
            or {item.name for item in staged_root.iterdir()}
            != {str(seed) for seed in SEEDS}
        ):
            raise ReleaseWorkflowError(
                "resume staged selections are not the exact seed boundary"
            )

    external_references = _external_tools(source_workflow) if execute else {}
    external = {
        name: _authenticate_external(reference, f"external {name}")
        for name, reference in external_references.items()
    } if execute else {}
    environment = _closed_environment(root, external) if execute else {}
    if execute:
        with _stable_external_git(
            external_references["git"], external["git"]
        ):
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
            revision = _run_capture_process(
                _hardened_git_command(
                    external["git"], "rev-parse", "HEAD"
                ),
                cwd=root,
                environment=environment,
                timeout=GIT_TIMEOUT_SECONDS,
            ).stdout.strip()
            if revision != manifest["executionSourceRevision"]:
                raise ReleaseWorkflowError(
                    "source revision differs from continuation manifest"
                )
            _preflight_recursive_git_filters(
                external["git"],
                repository=root,
                environment=environment,
            )
            dirty = _run_capture_process(
                _hardened_git_command(
                    external["git"],
                    "status",
                    "--porcelain",
                    "--untracked-files=all",
                    "--ignore-submodules=none",
                ),
                cwd=root,
                environment=environment,
                timeout=GIT_TIMEOUT_SECONDS,
            ).stdout
            if dirty:
                raise ReleaseWorkflowError(
                    "tracked source changes are present during continuation"
                )
        for reference in _input_references(source_workflow):
            from .release_workflow import _authenticate_input
            _authenticate_input(root, reference)

    reused: list[dict[str, str]] = []
    selections = manifest["selectionSha256"]
    assert isinstance(selections, dict)
    candidates = source_workflow["candidates"]
    assert isinstance(candidates, list)
    prefix_records: list[dict[str, object]] = []
    if execute:
        for candidate in candidates:
            assert isinstance(candidate, dict)
            seed = candidate["seed"]
            assert isinstance(seed, int)
            reused.extend(_stage_seed(
                root, archive, root / staging, candidate,
                _digest(selections[str(seed)], f"selectionSha256.{seed}"),
                reuse_existing=resume_mode,
            ))
        if resume_mode:
            assert isinstance(resume, Mapping)
            prefix_records = _authenticate_calibration_prefix(
                root,
                absolute_output,
                resume,
            )
        training_frequency = source_workflow["trainingFrequency"]
        assert isinstance(training_frequency, dict)
        frequency_source = _confined(
            root,
            _relative(
                training_frequency["path"],
                "trainingFrequency.path",
            ),
            must_exist=True,
        )
        frequency_sha = _digest(
            training_frequency["sha256"],
            "trainingFrequency.sha256",
        )
        frequency_target = absolute_output / "training-frequency.json"
        if resume_mode:
            _authenticate_reused_file(frequency_source, frequency_sha)
            if frequency_target.exists():
                _authenticate_reused_file(frequency_target, frequency_sha)
            else:
                _exclusive_copy(
                    frequency_source,
                    frequency_target,
                    frequency_sha,
                )
        else:
            _exclusive_copy(
                frequency_source,
                frequency_target,
                frequency_sha,
            )
        reused.append({
            "kind": "training-frequency",
            "source": frequency_source.relative_to(root).as_posix(),
            "staged": frequency_target.relative_to(root).as_posix(),
            "sha256": frequency_sha,
        })
    continued = _continued_workflow(source_workflow, staging, output, transcript_path)
    plan = build_plan(continued, root)
    first_stage = "validation-gate" if resume_mode else FIRST_STAGE
    first = next(
        index for index, step in enumerate(plan) if step.stage == first_stage
    )
    records: list[dict[str, object]] = []
    if execute:
        records.extend(prefix_records)
        for step in plan[first:]:
            argv, generated, outputs = _execute_step(step, continued, root, environment)
            records.append({
                "stage": step.stage,
                "status": "executed",
                "argv": list(argv),
                "inputs": [
                    {"path": item.path.as_posix(), "sha256": item.sha256}
                    for item in step.inputs
                ] + [
                    {"path": path.as_posix(), "sha256": generated[path]}
                    for path in step.generated_inputs
                ],
                "outputs": outputs,
            })
    transcript = {
        "format": FORMAT,
        "version": manifest["version"],
        "evidence": False,
        "reusedEvidence": True,
        "manifestSha256": manifest_sha,
        "selectionSourceRevision": source_workflow["sourceRevision"],
        "executionSourceRevision": manifest["executionSourceRevision"],
        "sourceWorkflow": {
            "path": _relative(source_ref["path"], "sourceWorkflow.path").as_posix(),
            "sha256": expected_source_sha,
        },
        "reusedInputs": reused,
        "steps": records,
    }
    if execute:
        target = root / transcript_path
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            transcript,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode() + b"\n"
        try:
            publish_bytes_durable_exact(
                target,
                payload,
                label="continuation transcript",
            )
        except (OSError, ValueError) as error:
            raise ReleaseWorkflowError(
                f"cannot publish continuation transcript: {target}"
            ) from error
    return transcript


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args(argv)
    manifest, digest = load_manifest(arguments.manifest)
    result = run(
        manifest,
        digest,
        arguments.repository.resolve(),
        execute=arguments.execute,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
