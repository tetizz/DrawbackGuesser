"""Build a closed release workflow from authenticated checkpoint indexes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

from ml.training.drawback_ml.checkpoint import verify_checkpoint_index

from .browser_parity import load_public_parity_fixture
from .release_workflow import (
    FORMAT as WORKFLOW_FORMAT,
    VERSION as WORKFLOW_VERSION,
    SEEDS,
    ArtifactRef,
    _authenticate_external,
    _authenticate_input,
    _confined,
    _constant,
    _digest,
    _exact,
    _external_reference,
    _object,
    _pairs,
    _reference,
    _relative,
    build_plan,
)
from .release_selection_bundle import (
    ContentAddressedJson,
    load_training_run,
)
from .training_frequency import (
    ContentAddressedFile,
    load_training_frequency_artifact,
)


LOCK_FORMAT = "drawbacktrainer-release-workflow-builder-lock"
LOCK_VERSION = 1


class WorkflowBuilderError(ValueError):
    pass


def _canonical(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8") + b"\n"


def _validate_builder_lock_value(value: Mapping[str, object]) -> None:
    try:
        _exact(
            value,
            {
                "format",
                "version",
                "sourceRevision",
                "tools",
                "shared",
                "candidates",
                "trainingFrequency",
                "browserFixture",
                "outputRoot",
            },
            "builder lock",
        )
    except ValueError as error:
        raise WorkflowBuilderError(str(error)) from error
    if (
        value["format"] != LOCK_FORMAT
        or type(value["version"]) is not int
        or value["version"] != LOCK_VERSION
    ):
        raise WorkflowBuilderError("builder lock identity is invalid")
    revision = value["sourceRevision"]
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise WorkflowBuilderError(
            "builder sourceRevision must be a full lowercase Git SHA"
        )


def load_builder_lock(path: Path) -> Mapping[str, object]:
    try:
        payload = path.read_bytes()
        value = _object(
            json.loads(
                payload,
                object_pairs_hook=_pairs,
                parse_constant=_constant,
            ),
            "builder lock",
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise WorkflowBuilderError(
            f"cannot load strict UTF-8 builder lock: {path}"
        ) from error
    _validate_builder_lock_value(value)
    if payload != _canonical(value):
        raise WorkflowBuilderError("builder lock must be canonical JSON")
    return value


def build_release_workflow(
    lock: Mapping[str, object],
    repository: Path,
) -> Mapping[str, object]:
    _validate_builder_lock_value(lock)
    root = repository.resolve(strict=True)
    tools = _object(lock["tools"], "builder tools")
    _exact(tools, {"browser", "git", "node", "pnpm"}, "builder tools")
    tool_values: dict[str, dict[str, str]] = {}
    for name in ("browser", "git", "node", "pnpm"):
        reference = _external_reference(tools[name], f"builder tools.{name}")
        _authenticate_external(reference, f"builder external {name}")
        tool_values[name] = {
            "path": str(reference.path),
            "sha256": reference.sha256,
        }

    shared_raw = _object(lock["shared"], "builder shared")
    _exact(
        shared_raw,
        {"dataset", "publicRoot", "privateValidation"},
        "builder shared",
    )
    shared = {
        name: _reference(shared_raw[name], f"builder shared.{name}")
        for name in ("dataset", "publicRoot", "privateValidation")
    }
    for reference in shared.values():
        _authenticate_input(root, reference)

    frequency = _reference(
        lock["trainingFrequency"],
        "builder trainingFrequency",
    )
    fixture = _reference(lock["browserFixture"], "builder browserFixture")
    _authenticate_input(root, frequency)
    _authenticate_input(root, fixture)
    load_public_parity_fixture(
        _confined(root, fixture.path, must_exist=True),
        fixture.sha256,
    )

    output_root = _relative(lock["outputRoot"], "builder outputRoot")
    if not output_root.parts or output_root.parts[:2] != ("data", "generated"):
        raise WorkflowBuilderError(
            "builder outputRoot must be inside data/generated"
        )

    raw_candidates = lock["candidates"]
    if not isinstance(raw_candidates, list) or len(raw_candidates) != len(SEEDS):
        raise WorkflowBuilderError("builder requires exactly three candidates")
    candidates: list[dict[str, object]] = []
    corpus_set_digests: set[str] = set()
    release_root_digests: set[str] = set()
    for index, raw_candidate in enumerate(raw_candidates):
        candidate = _object(raw_candidate, f"builder candidate {index}")
        _exact(
            candidate,
            {"seed", "checkpointIndex"},
            f"builder candidate {index}",
        )
        seed = candidate["seed"]
        if seed != SEEDS[index]:
            raise WorkflowBuilderError("builder candidate seed order is invalid")
        index_reference = _reference(
            candidate["checkpointIndex"],
            f"builder candidate {index}.checkpointIndex",
        )
        index_path = _authenticate_input(root, index_reference)
        index_value = verify_checkpoint_index(
            index_path,
            index_reference.sha256,
        )
        if index_value["seed"] != seed or index_value["epochs"] != 8:
            raise WorkflowBuilderError(
                "builder checkpoint index seed or epoch plan is invalid"
            )
        run_claim = _object(
            index_value["runClaim"],
            f"builder candidate {index}.runClaim",
        )
        run_path = index_reference.path.parent / str(run_claim["file"])
        loaded_run = load_training_run(
            ContentAddressedJson(
                _confined(root, run_path, must_exist=True),
                str(run_claim["sha256"]),
            )
        )
        corpus_set_digests.add(loaded_run.training_corpus_set_sha256)
        run_value = _strict_manifest(
            _confined(root, run_path, must_exist=True),
            f"builder candidate {index} training run",
        )
        run_config = run_value.get("config")
        provenance = (
            run_config.get("corpus_provenance")
            if isinstance(run_config, dict)
            else None
        )
        corpus_set = (
            provenance.get("training_corpus_set")
            if isinstance(provenance, dict)
            else None
        )
        primary = (
            corpus_set.get("primary")
            if isinstance(corpus_set, dict)
            else None
        )
        release_root_sha256 = (
            primary.get("release_root_sha256")
            if isinstance(primary, dict)
            else None
        )
        release_root_digests.add(
            _digest(
                release_root_sha256,
                f"builder candidate {index} release root sha256",
            )
        )
        epochs = index_value["checkpoints"]
        assert isinstance(epochs, list)
        candidates.append(
            {
                "seed": seed,
                "trainingRun": {
                    "path": run_path.as_posix(),
                    "sha256": run_claim["sha256"],
                },
                "epochs": [
                    {
                        "epoch": epoch,
                        "checkpoint": {
                            "path": (
                                index_reference.path.parent
                                / str(item["file"])
                            ).as_posix(),
                            "sha256": item["sha256"],
                        },
                        "report": (
                            output_root
                            / str(seed)
                            / f"epoch-{epoch}.selection-report.json"
                        ).as_posix(),
                        "summary": (
                            output_root
                            / str(seed)
                            / f"epoch-{epoch}.selection-summary.json"
                        ).as_posix(),
                    }
                    for epoch, item in enumerate(epochs, 1)
                    if isinstance(item, dict)
                ],
            }
        )

    if len(corpus_set_digests) != 1 or len(release_root_digests) != 1:
        raise WorkflowBuilderError(
            "builder candidates do not share one training corpus identity"
        )
    expected_corpus_set = next(iter(corpus_set_digests))
    expected_release_root = next(iter(release_root_digests))
    _validate_validation_bindings(
        root,
        shared,
        expected_release_root,
    )
    load_training_frequency_artifact(
        ContentAddressedFile(
            _confined(root, frequency.path, must_exist=True),
            frequency.sha256,
        ),
        expected_corpus_set,
    )

    workflow: dict[str, object] = {
        "format": WORKFLOW_FORMAT,
        "version": WORKFLOW_VERSION,
        "sourceRevision": lock["sourceRevision"],
        "realDomainExecution": "external-isolated-only",
        "tools": tool_values,
        "shared": {
            name: {
                "path": reference.path.as_posix(),
                "sha256": reference.sha256,
            }
            for name, reference in shared.items()
        },
        "candidates": candidates,
        "selectionOutputs": {
            str(seed): (output_root / str(seed) / "selection.json").as_posix()
            for seed in SEEDS
        },
        "ensemble": {
            "output": (output_root / "ensemble.json").as_posix(),
        },
        "calibration": {
            "report": (output_root / "calibration-report.json").as_posix(),
            "sidecar": (output_root / "calibration-sidecar.ndjson").as_posix(),
            "receipt": (output_root / "calibration-receipt.json").as_posix(),
            "output": (output_root / "calibration.json").as_posix(),
        },
        "trainingFrequency": {
            "path": frequency.path.as_posix(),
            "sha256": frequency.sha256,
        },
        "validation": {
            "report": (output_root / "validation-report.json").as_posix(),
            "decision": (output_root / "validation-decision.json").as_posix(),
        },
        "browserRelease": {
            "fixture": {
                "path": fixture.path.as_posix(),
                "sha256": fixture.sha256,
            },
            "artifact": (output_root / "browser-model.json").as_posix(),
            "parityInput": (output_root / "parity-input.json").as_posix(),
            "transcript": (output_root / "parity-transcript.json").as_posix(),
            "evidence": (output_root / "parity-evidence.json").as_posix(),
        },
        "transcriptOutput": (
            output_root / "workflow-transcript.json"
        ).as_posix(),
    }
    build_plan(workflow, root)
    return workflow


def _strict_manifest(path: Path, label: str) -> Mapping[str, object]:
    try:
        payload = path.read_bytes()
        value = json.loads(
            payload,
            object_pairs_hook=_pairs,
            parse_constant=_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise WorkflowBuilderError(f"{label} is not strict JSON") from error
    if not isinstance(value, dict):
        raise WorkflowBuilderError(f"{label} must be an object")
    return value


def _validate_validation_bindings(
    root: Path,
    shared: Mapping[str, ArtifactRef],
    expected_release_root_sha256: str,
) -> None:
    dataset = shared["dataset"]
    public = shared["publicRoot"]
    private = shared["privateValidation"]
    if (
        public.path.as_posix()
        != "data/releases/current/public/manifest.json"
        or private.path.as_posix()
        != "data/releases/current/private/validation/manifest.json"
    ):
        raise WorkflowBuilderError(
            "builder validation manifests must use the current release roots"
        )
    dataset_path = _confined(root, dataset.path, must_exist=True)
    public_value = _strict_manifest(
        _confined(root, public.path, must_exist=True),
        "public release manifest",
    )
    private_value = _strict_manifest(
        _confined(root, private.path, must_exist=True),
        "private validation manifest",
    )
    public_splits = public_value.get("splits")
    public_validation = (
        public_splits.get("validation")
        if isinstance(public_splits, dict)
        else None
    )
    private_dataset = private_value.get("dataset")
    if (
        type(public_value.get("releaseManifestVersion")) is not int
        or public_value.get("releaseManifestVersion") != 1
        or type(private_value.get("manifestVersion")) is not int
        or private_value.get("manifestVersion") != 1
        or private_value.get("split") != "validation"
        or public.sha256 != expected_release_root_sha256
        or not isinstance(public_validation, dict)
        or not isinstance(private_dataset, dict)
        or public_value.get("corpusRunId") != private_value.get("corpusRunId")
        or public_validation.get("privateManifestSha256") != private.sha256
        or public_validation.get("datasetSha256") != dataset.sha256
        or public_validation.get("datasetBytes") != dataset_path.stat().st_size
        or private_dataset.get("file") != dataset.path.name
        or private_dataset.get("sha256") != dataset.sha256
        or private_dataset.get("bytes") != dataset_path.stat().st_size
    ):
        raise WorkflowBuilderError(
            "builder validation dataset is not bound by the current release"
        )


def write_release_workflow(
    lock_path: Path,
    output: Path,
    *,
    repository: Path,
) -> tuple[Path, str]:
    lock = load_builder_lock(lock_path)
    workflow = build_release_workflow(lock, repository)
    root = repository.resolve(strict=True)
    try:
        relative_output = output.resolve(strict=False).relative_to(root)
    except ValueError as error:
        raise WorkflowBuilderError(
            "workflow output must be inside the repository"
        ) from error
    if relative_output.parts[:2] != ("data", "generated"):
        raise WorkflowBuilderError(
            "workflow output must be inside data/generated"
        )
    confined = _confined(root, relative_output, must_exist=False)
    plan = build_plan(workflow, root)
    reserved_paths = {
        path
        for step in plan
        for path in (*step.outputs, *step.generated_inputs)
    } | {
        reference.path
        for step in plan
        for reference in step.inputs
    }
    if relative_output in reserved_paths:
        raise WorkflowBuilderError(
            "workflow output collides with a release input or output"
        )
    if not confined.parent.is_dir():
        raise WorkflowBuilderError(
            "workflow output parent must already exist"
        )
    payload = _canonical(workflow)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".release-workflow.",
        suffix=".tmp",
        dir=confined.parent,
    )
    temporary = Path(temporary_name)
    committed = False
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())
        try:
            os.link(temporary, confined)
        except FileExistsError as error:
            raise FileExistsError(
                f"release workflow already exists: {confined}"
            ) from error
        committed = True
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            if not committed:
                raise
    return confined, hashlib.sha256(payload).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ml.evaluation.release_workflow_builder"
    )
    parser.add_argument("lock", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repository", type=Path, default=Path("."))
    arguments = parser.parse_args(argv)
    output, digest = write_release_workflow(
        arguments.lock,
        arguments.output,
        repository=arguments.repository,
    )
    print(
        json.dumps(
            {"file": str(output), "sha256": digest},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
