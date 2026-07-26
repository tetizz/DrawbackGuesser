"""Strict, content-addressed release bundle for a three-member ensemble."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Mapping, Sequence

from .release_selection_bundle import (
    ContentAddressedJson,
    VerifiedSelectionBundle,
    verify_release_selection_bundle,
)
from .validation_partition import VALIDATION_PARTITION_IDENTITY


ENSEMBLE_RELEASE_FORMAT = "drawbacktrainer-ensemble-release"
ENSEMBLE_RELEASE_VERSION = 3
ENSEMBLE_TRAINING_SEEDS = (20260811, 20260812, 20260813)


@dataclass(frozen=True)
class EnsembleMember:
    training_seed: int
    selection_file: str
    selection_sha256: str
    training_run_file: str
    training_run_sha256: str
    training_run_id: str
    checkpoint_file: str
    checkpoint_sha256: str
    checkpoint_epoch: int


@dataclass(frozen=True)
class LoadedEnsembleRelease:
    source: ContentAddressedJson
    release_root_sha256: str
    corpus_run_id: str
    training_corpus_set_sha256: str
    private_validation_manifest_sha256: str
    validation_dataset_sha256: str
    partition_seed_sha256: str
    members: tuple[EnsembleMember, ...]


def write_ensemble_release(
    output: Path,
    selection_references: Sequence[ContentAddressedJson],
    training_run_references: Sequence[ContentAddressedJson],
) -> Path:
    """Verify and atomically publish the fixed three-member release bundle."""

    if len(selection_references) != 3:
        raise ValueError("ensemble release requires exactly three selections")
    if len(training_run_references) != 3:
        raise ValueError("ensemble release requires exactly three training runs")
    for selection in selection_references:
        _require_regular_source(selection.path, "selection")
    for training_run in training_run_references:
        _require_regular_source(training_run.path, "training run")
    verified = tuple(
        verify_release_selection_bundle(selection, training_run)
        for selection, training_run in zip(
            selection_references, training_run_references, strict=True
        )
    )
    _verify_members(verified)
    first = verified[0].artifact
    members = [
        {
            "training_seed": seed,
            "selection": {
                "file": _relative_source(output, selection.path, "selection"),
                "sha256": selection.sha256,
            },
            "training_run": {
                "file": _relative_source(
                    output, training_run.path, "training run"
                ),
                "sha256": training_run.sha256,
                "run_id": bundle.training_run.run_id,
            },
            "selected_checkpoint": {
                "file": bundle.artifact.selected_checkpoint_file,
                "sha256": bundle.artifact.selected_checkpoint_sha256,
                "epoch": bundle.artifact.selected_epoch,
            },
        }
        for seed, selection, training_run, bundle in zip(
            ENSEMBLE_TRAINING_SEEDS,
            selection_references,
            training_run_references,
            verified,
            strict=True,
        )
    ]
    value = {
        "format": ENSEMBLE_RELEASE_FORMAT,
        "version": ENSEMBLE_RELEASE_VERSION,
        "method": {
            "member_count": 3,
            "seed_order": list(ENSEMBLE_TRAINING_SEEDS),
        },
        "provenance": {
            "release_root_sha256": first.provenance.release_root_sha256,
            "corpus_run_id": first.provenance.corpus_run_id,
            "training_corpus_set_sha256": (
                first.provenance.training_corpus_set_sha256
            ),
            "private_validation_manifest_sha256": (
                first.provenance.private_validation_manifest_sha256
            ),
            "validation_dataset_sha256": (
                first.provenance.validation_dataset_sha256
            ),
            "validation_partition": {
                "identity": VALIDATION_PARTITION_IDENTITY,
                "name": "selection",
                "seed_sha256": first.partition_seed_sha256,
            },
        },
        "members": members,
    }
    _write_atomic_no_clobber(output, _canonical_json(value))
    return output


def load_ensemble_release(
    reference: ContentAddressedJson,
) -> LoadedEnsembleRelease:
    """Load and structurally verify one immutable ensemble artifact."""

    payload = _verified_bytes(reference, "ensemble release")
    value = _strict_json(payload, "ensemble release")
    if payload != _canonical_json(value):
        raise ValueError("ensemble release JSON bytes are not canonical")
    _exact_keys(
        value,
        {"format", "version", "method", "provenance", "members"},
        "ensemble release",
    )
    if (
        value["format"] != ENSEMBLE_RELEASE_FORMAT
        or value["version"] != ENSEMBLE_RELEASE_VERSION
    ):
        raise ValueError("ensemble release format is unsupported")
    method = _object(value["method"], "ensemble method")
    _exact_keys(method, {"member_count", "seed_order"}, "ensemble method")
    if method != {
        "member_count": 3,
        "seed_order": list(ENSEMBLE_TRAINING_SEEDS),
    }:
        raise ValueError("ensemble method or seed order is invalid")
    provenance = _object(value["provenance"], "ensemble provenance")
    _exact_keys(
        provenance,
        {
            "release_root_sha256",
            "corpus_run_id",
            "training_corpus_set_sha256",
            "private_validation_manifest_sha256",
            "validation_dataset_sha256",
            "validation_partition",
        },
        "ensemble provenance",
    )
    partition = _object(
        provenance["validation_partition"], "ensemble partition"
    )
    _exact_keys(
        partition, {"identity", "name", "seed_sha256"}, "ensemble partition"
    )
    if (
        partition["identity"] != VALIDATION_PARTITION_IDENTITY
        or partition["name"] != "selection"
    ):
        raise ValueError("ensemble validation partition is invalid")
    members_value = value["members"]
    if not isinstance(members_value, list) or len(members_value) != 3:
        raise ValueError("ensemble release requires exactly three members")
    members = tuple(
        _load_member(item, expected_seed)
        for item, expected_seed in zip(
            members_value, ENSEMBLE_TRAINING_SEEDS, strict=True
        )
    )
    if len({item.selection_sha256 for item in members}) != 3:
        raise ValueError("ensemble members reuse a selection artifact")
    if len({item.training_run_sha256 for item in members}) != 3:
        raise ValueError("ensemble members reuse a training run claim")
    if len({item.training_run_id for item in members}) != 3:
        raise ValueError("ensemble members reuse a training run ID")
    if len({item.checkpoint_sha256 for item in members}) != 3:
        raise ValueError("ensemble members reuse a selected checkpoint")
    return LoadedEnsembleRelease(
        source=reference,
        release_root_sha256=_digest(
            provenance["release_root_sha256"], "release root sha256"
        ),
        corpus_run_id=_digest(provenance["corpus_run_id"], "corpus run ID"),
        training_corpus_set_sha256=_digest(
            provenance["training_corpus_set_sha256"],
            "training corpus set sha256",
        ),
        private_validation_manifest_sha256=_digest(
            provenance["private_validation_manifest_sha256"],
            "private validation manifest sha256",
        ),
        validation_dataset_sha256=_digest(
            provenance["validation_dataset_sha256"],
            "validation dataset sha256",
        ),
        partition_seed_sha256=_digest(
            partition["seed_sha256"], "partition seed sha256"
        ),
        members=members,
    )


def verify_ensemble_release(
    reference: ContentAddressedJson,
) -> LoadedEnsembleRelease:
    """Recursively verify the ensemble and all member release evidence."""

    loaded = load_ensemble_release(reference)
    directory = reference.path.parent.resolve()
    verified: list[VerifiedSelectionBundle] = []
    for member in loaded.members:
        selection_path = _local_source(
            directory, member.selection_file, "selection"
        )
        training_run_path = _local_source(
            directory, member.training_run_file, "training run"
        )
        bundle = verify_release_selection_bundle(
            ContentAddressedJson(selection_path, member.selection_sha256),
            ContentAddressedJson(training_run_path, member.training_run_sha256),
        )
        if bundle.artifact.training_seed != member.training_seed:
            raise ValueError("ensemble member training seed disagrees with selection")
        if bundle.training_run.run_id != member.training_run_id:
            raise ValueError("ensemble member training run ID is invalid")
        if (
            bundle.artifact.selected_checkpoint_file != member.checkpoint_file
            or bundle.artifact.selected_checkpoint_sha256
            != member.checkpoint_sha256
            or bundle.artifact.selected_epoch != member.checkpoint_epoch
        ):
            raise ValueError("ensemble selected checkpoint binding is invalid")
        verified.append(bundle)
    _verify_members(tuple(verified))
    first = verified[0].artifact
    expected = (
        first.provenance.release_root_sha256,
        first.provenance.corpus_run_id,
        first.provenance.training_corpus_set_sha256,
        first.provenance.private_validation_manifest_sha256,
        first.provenance.validation_dataset_sha256,
        first.partition_seed_sha256,
    )
    declared = (
        loaded.release_root_sha256,
        loaded.corpus_run_id,
        loaded.training_corpus_set_sha256,
        loaded.private_validation_manifest_sha256,
        loaded.validation_dataset_sha256,
        loaded.partition_seed_sha256,
    )
    if declared != expected:
        raise ValueError("ensemble declared provenance disagrees with members")
    return loaded


def resolve_member_checkpoint(
    reference: ContentAddressedJson,
    member: EnsembleMember,
) -> Path:
    """Resolve one selected checkpoint through its authenticated run location."""

    directory = reference.path.parent.resolve()
    training_run = _local_source(
        directory,
        member.training_run_file,
        "training run",
    )
    checkpoint = training_run.parent / member.checkpoint_file
    if checkpoint.is_symlink():
        raise ValueError("ensemble checkpoint must not be a symbolic link")
    try:
        resolved = checkpoint.resolve(strict=True)
        resolved.relative_to(training_run.parent.resolve())
    except (OSError, ValueError) as error:
        raise ValueError(
            "ensemble checkpoint is missing or escapes its training run directory"
        ) from error
    if not resolved.is_file():
        raise ValueError("ensemble checkpoint is not a regular file")
    return resolved


def _verify_members(
    bundles: tuple[VerifiedSelectionBundle, ...],
) -> None:
    if len(bundles) != 3:
        raise ValueError("ensemble release requires exactly three members")
    if tuple(item.artifact.training_seed for item in bundles) != (
        ENSEMBLE_TRAINING_SEEDS
    ):
        raise ValueError("ensemble selections are not in fixed seed order")
    if tuple(item.training_run.seed for item in bundles) != (
        ENSEMBLE_TRAINING_SEEDS
    ):
        raise ValueError("ensemble training runs are not in fixed seed order")
    common = {
        (
            item.artifact.provenance.release_root_sha256,
            item.artifact.provenance.corpus_run_id,
            item.artifact.provenance.training_corpus_set_sha256,
            item.artifact.provenance.private_validation_manifest_sha256,
            item.artifact.provenance.validation_dataset_sha256,
            item.artifact.partition_seed_sha256,
        )
        for item in bundles
    }
    if len(common) != 1:
        raise ValueError("ensemble members use mixed corpus provenance")
    if len({item.artifact.sha256 for item in bundles}) != 3:
        raise ValueError("ensemble members reuse a selection artifact")
    if len({item.training_run.sha256 for item in bundles}) != 3:
        raise ValueError("ensemble members reuse a training run claim")
    if len({item.training_run.run_id for item in bundles}) != 3:
        raise ValueError("ensemble members reuse a training run ID")
    if (
        len(
            {
                item.artifact.selected_checkpoint_sha256
                for item in bundles
            }
        )
        != 3
    ):
        raise ValueError("ensemble members reuse a selected checkpoint")


def _load_member(value: object, expected_seed: int) -> EnsembleMember:
    member = _object(value, "ensemble member")
    _exact_keys(
        member,
        {
            "training_seed",
            "selection",
            "training_run",
            "selected_checkpoint",
        },
        "ensemble member",
    )
    seed = _nonnegative_int(member["training_seed"], "member training seed")
    if seed != expected_seed:
        raise ValueError("ensemble members are reordered or use an invalid seed")
    selection = _object(member["selection"], "member selection")
    _exact_keys(selection, {"file", "sha256"}, "member selection")
    training_run = _object(member["training_run"], "member training run")
    _exact_keys(
        training_run, {"file", "sha256", "run_id"}, "member training run"
    )
    checkpoint = _object(
        member["selected_checkpoint"], "member selected checkpoint"
    )
    _exact_keys(
        checkpoint,
        {"file", "sha256", "epoch"},
        "member selected checkpoint",
    )
    return EnsembleMember(
        training_seed=seed,
        selection_file=_safe_source_relative(
            selection["file"], "selection file"
        ),
        selection_sha256=_digest(
            selection["sha256"], "selection sha256"
        ),
        training_run_file=_safe_source_relative(
            training_run["file"], "training run file"
        ),
        training_run_sha256=_digest(
            training_run["sha256"], "training run sha256"
        ),
        training_run_id=_digest(
            training_run["run_id"], "training run ID"
        ),
        checkpoint_file=_basename(
            checkpoint["file"], "selected checkpoint file"
        ),
        checkpoint_sha256=_digest(
            checkpoint["sha256"], "selected checkpoint sha256"
        ),
        checkpoint_epoch=_positive_int(
            checkpoint["epoch"], "selected checkpoint epoch"
        ),
    )


def _relative_source(output: Path, source: Path, name: str) -> str:
    output_directory = output.parent.resolve()
    resolved = _require_regular_source(source, name)
    try:
        relative = resolved.relative_to(output_directory)
        return _safe_relative(relative.as_posix(), f"{name} file")
    except ValueError:
        repository = _repository_root(output_directory)
        try:
            resolved.relative_to(repository)
        except ValueError as error:
            raise ValueError(
                f"{name} must be inside the ensemble repository"
            ) from error
        rendered = Path(os.path.relpath(resolved, output_directory)).as_posix()
        return _safe_source_relative(rendered, f"{name} file")


def _require_regular_source(source: Path, name: str) -> Path:
    try:
        resolved = source.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{name} source is missing") from error
    if source.is_symlink() or not resolved.is_file():
        raise ValueError(f"{name} source must be a regular non-symlink file")
    return resolved


def _local_source(directory: Path, value: str, name: str) -> Path:
    relative = _safe_source_relative(value, f"{name} file")
    candidate = directory / Path(*PurePosixPath(relative).parts)
    if candidate.is_symlink():
        raise ValueError(f"{name} file must not be a symbolic link")
    try:
        path = candidate.resolve(strict=True)
        confinement = (
            _repository_root(directory)
            if ".." in PurePosixPath(relative).parts
            else directory
        )
        path.relative_to(confinement)
    except (OSError, ValueError) as error:
        raise ValueError(f"{name} path escapes its allowed source root") from error
    if not path.is_file():
        raise ValueError(f"{name} file is missing: {relative}")
    return path


def _repository_root(path: Path) -> Path:
    resolved = path.resolve(strict=True)
    for candidate in (resolved, *resolved.parents):
        marker = candidate / ".git"
        if marker.exists() and not marker.is_symlink():
            return candidate
    raise ValueError("ensemble repository root cannot be authenticated")


def _safe_source_relative(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or ":" in value
    ):
        raise ValueError(f"{name} must be a safe relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or value != path.as_posix() or "." in path.parts:
        raise ValueError(f"{name} must be a safe relative POSIX path")
    return value


def _safe_relative(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{name} must be a safe relative POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value != path.as_posix()
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise ValueError(f"{name} must be a safe relative POSIX path")
    return value


def _basename(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or Path(value).name != value
        or "/" in value
        or "\\" in value
    ):
        raise ValueError(f"{name} must be a non-empty basename")
    return value


def _verified_bytes(reference: ContentAddressedJson, name: str) -> bytes:
    try:
        payload = reference.path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read {name}: {reference.path}") from error
    if hashlib.sha256(payload).hexdigest() != reference.sha256:
        raise ValueError(f"{name} sha256 does not match")
    return payload


def _strict_json(payload: bytes, name: str) -> Mapping[str, object]:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"{name} contains duplicate key {key!r}")
            result[key] = value
        return result

    def constant(token: str) -> None:
        raise ValueError(f"{name} contains non-finite constant {token}")

    try:
        value = json.loads(
            payload,
            object_pairs_hook=pairs,
            parse_constant=constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not strict UTF-8 JSON") from error
    _require_finite(value, name)
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} root must be an object")
    return value


def _canonical_json(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            value,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_atomic_no_clobber(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ValueError(
                f"refusing to overwrite ensemble release: {path}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)


def _require_finite(value: object, name: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} contains a non-finite number")
    if isinstance(value, Mapping):
        for item in value.values():
            _require_finite(item, name)
    elif isinstance(value, list):
        for item in value:
            _require_finite(item, name)


def _exact_keys(
    value: Mapping[object, object], expected: set[str], name: str
) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} fields are not canonical")


def _object(value: object, name: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_int(value: object, name: str) -> int:
    result = _nonnegative_int(value, name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result
