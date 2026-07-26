"""Publish a canonical workflow-builder lock from explicit release inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

from ml.training.drawback_ml.checkpoint import verify_checkpoint_index

from .release_workflow import SEEDS, _confined, _relative
from .release_workflow_builder import (
    LOCK_FORMAT,
    LOCK_VERSION,
    WorkflowBuilderError,
    _canonical,
    _validate_builder_lock_value,
)


TOOL_NAMES = ("browser", "git", "node", "pnpm")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_reference(
    root: Path,
    path: Path,
    label: str,
) -> dict[str, str]:
    candidate = path if path.is_absolute() else root / path
    try:
        resolved = candidate.resolve(strict=True)
        relative = resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise WorkflowBuilderError(
            f"{label} must be an existing file inside the repository"
        ) from error
    normalized = _relative(relative.as_posix(), label)
    try:
        resolved = _confined(root, normalized, must_exist=True)
    except ValueError as error:
        raise WorkflowBuilderError(
            f"{label} must be a confined regular file"
        ) from error
    if not resolved.is_file():
        raise WorkflowBuilderError(f"{label} must be a regular file")
    return {
        "path": normalized.as_posix(),
        "sha256": _hash_file(resolved),
    }


def _external_reference(path: Path, label: str) -> dict[str, str]:
    if not path.is_absolute():
        raise WorkflowBuilderError(f"{label} must be an absolute path")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise WorkflowBuilderError(
            f"{label} must be an existing external tool"
        ) from error
    if resolved.is_symlink() or not resolved.is_file():
        raise WorkflowBuilderError(f"{label} must be a regular file")
    return {"path": str(resolved), "sha256": _hash_file(resolved)}


def build_builder_lock(
    *,
    repository: Path,
    source_revision: str,
    tools: Mapping[str, Path],
    dataset: Path,
    public_root: Path,
    private_validation: Path,
    checkpoint_indexes: Sequence[Path],
    training_frequency: Path,
    browser_fixture: Path,
    output_root: str,
) -> Mapping[str, object]:
    root = repository.resolve(strict=True)
    if set(tools) != set(TOOL_NAMES):
        raise WorkflowBuilderError(
            "lock writer requires browser, git, node, and pnpm tools"
        )
    if len(checkpoint_indexes) != len(SEEDS):
        raise WorkflowBuilderError(
            "lock writer requires exactly three checkpoint indexes"
        )
    normalized_output_root = _relative(output_root, "output root")
    if normalized_output_root.parts[:2] != ("data", "generated"):
        raise WorkflowBuilderError(
            "lock writer output root must be inside data/generated"
        )
    candidates: list[dict[str, object]] = []
    for seed, checkpoint_index in zip(
        SEEDS,
        checkpoint_indexes,
        strict=True,
    ):
        reference = _project_reference(
            root,
            checkpoint_index,
            f"seed {seed} checkpoint index",
        )
        try:
            index = verify_checkpoint_index(
                root / reference["path"],
                reference["sha256"],
            )
        except ValueError as error:
            raise WorkflowBuilderError(
                f"seed {seed} checkpoint index is invalid"
            ) from error
        if index["seed"] != seed or index["epochs"] != 8:
            raise WorkflowBuilderError(
                f"checkpoint index does not belong to fixed seed {seed}"
            )
        candidates.append(
            {"seed": seed, "checkpointIndex": reference}
        )
    lock: dict[str, object] = {
        "format": LOCK_FORMAT,
        "version": LOCK_VERSION,
        "sourceRevision": source_revision,
        "tools": {
            name: _external_reference(tools[name], f"tool {name}")
            for name in TOOL_NAMES
        },
        "shared": {
            "dataset": _project_reference(root, dataset, "validation dataset"),
            "publicRoot": _project_reference(
                root, public_root, "public release root"
            ),
            "privateValidation": _project_reference(
                root,
                private_validation,
                "private validation manifest",
            ),
        },
        "candidates": candidates,
        "trainingFrequency": _project_reference(
            root,
            training_frequency,
            "training-frequency comparator",
        ),
        "browserFixture": _project_reference(
            root,
            browser_fixture,
            "public browser fixture",
        ),
        "outputRoot": normalized_output_root.as_posix(),
    }
    _validate_builder_lock_value(lock)
    return lock


def write_builder_lock(
    output: Path,
    lock: Mapping[str, object],
    *,
    repository: Path,
) -> tuple[Path, str]:
    root = repository.resolve(strict=True)
    candidate_output = output if output.is_absolute() else root / output
    try:
        relative_output = candidate_output.resolve(
            strict=False
        ).relative_to(root)
    except ValueError as error:
        raise WorkflowBuilderError(
            "builder lock output must be inside the repository"
        ) from error
    if relative_output.parts[:2] != ("data", "generated"):
        raise WorkflowBuilderError(
            "builder lock output must be inside data/generated"
        )
    destination = root / relative_output
    if not destination.parent.is_dir() or destination.parent.is_symlink():
        raise WorkflowBuilderError(
            "builder lock output parent must be an existing regular directory"
        )
    _validate_builder_lock_value(lock)
    payload = _canonical(lock)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".release-builder-lock.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    committed = False
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise FileExistsError(
                f"release builder lock already exists: {destination}"
            ) from error
        committed = True
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            if not committed:
                raise
    return destination, hashlib.sha256(payload).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ml.evaluation.release_workflow_lock"
    )
    parser.add_argument("output", type=Path)
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument("--source-revision", required=True)
    for name in TOOL_NAMES:
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--private-validation", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-index",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--training-frequency", type=Path, required=True)
    parser.add_argument("--browser-fixture", type=Path, required=True)
    parser.add_argument("--output-root", required=True)
    arguments = parser.parse_args(argv)
    lock = build_builder_lock(
        repository=arguments.repository,
        source_revision=arguments.source_revision,
        tools={name: getattr(arguments, name) for name in TOOL_NAMES},
        dataset=arguments.dataset,
        public_root=arguments.public_root,
        private_validation=arguments.private_validation,
        checkpoint_indexes=arguments.checkpoint_index,
        training_frequency=arguments.training_frequency,
        browser_fixture=arguments.browser_fixture,
        output_root=arguments.output_root,
    )
    output, digest = write_builder_lock(
        arguments.output,
        lock,
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
