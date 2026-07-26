"""Deterministic checkpoint naming and metadata."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from .rank_preserving_fusion import RANK_PRESERVING_FUSION_METHOD
from .symbolic import (
    FUSION_AWARE_LOSS_ALPHA_GRID,
    FUSION_AWARE_LOSS_METHOD,
    FUSION_AWARE_LOSS_VERSION,
)


LEGACY_ADDITIVE_DRAWBACK_OBJECTIVE = (
    "legacy-additive-symbolic-log-prior-nll-v1"
)
FUSION_GRID_DRAWBACK_OBJECTIVE = FUSION_AWARE_LOSS_METHOD
FUSION_GRID_DRAWBACK_OBJECTIVE_METADATA = {
    "method": FUSION_AWARE_LOSS_METHOD,
    "version": FUSION_AWARE_LOSS_VERSION,
    "production_fusion_method": RANK_PRESERVING_FUSION_METHOD,
    "alpha_grid": list(FUSION_AWARE_LOSS_ALPHA_GRID),
    "aggregation": "mean-cross-entropy-across-alpha-grid-v1",
}


def parse_checkpoint_drawback_objective(
    model_variant: str,
    training_metadata: Mapping[str, Any],
) -> str:
    """Return the exact drawback objective declared by a checkpoint.

    Historical v21 checkpoints predate objective metadata. They remain
    loadable under an explicit legacy identity; metadata that is present must
    match the frozen rank-preserving contract exactly.
    """

    raw = training_metadata.get("drawback_loss_objective")
    if model_variant not in {"v21-hybrid", "v22-hybrid"}:
        if raw is not None:
            raise ValueError(
                "non-hybrid checkpoint cannot declare a hybrid drawback objective"
            )
        return LEGACY_ADDITIVE_DRAWBACK_OBJECTIVE
    if raw is None:
        if model_variant == "v22-hybrid":
            raise ValueError(
                "v22-hybrid checkpoint requires the fusion-grid objective"
            )
        return LEGACY_ADDITIVE_DRAWBACK_OBJECTIVE
    _require_exact_fusion_objective_metadata(raw)
    return FUSION_GRID_DRAWBACK_OBJECTIVE


def parse_training_run_drawback_objective(
    config: Mapping[str, object],
) -> str:
    """Authenticate the objective frozen into a streaming training claim."""

    model_variant = config.get("model_variant")
    if model_variant not in {"v21-hybrid", "v22-hybrid"}:
        return LEGACY_ADDITIVE_DRAWBACK_OBJECTIVE
    fields = (
        "fusion_aware_loss_method",
        "fusion_aware_loss_version",
        "fusion_aware_loss_alpha_grid",
    )
    present = tuple(field in config for field in fields)
    if not any(present) and model_variant == "v21-hybrid":
        return LEGACY_ADDITIVE_DRAWBACK_OBJECTIVE
    if not all(present):
        raise ValueError("training run has partial fusion objective metadata")
    if (
        type(config["fusion_aware_loss_method"]) is not str
        or config["fusion_aware_loss_method"] != FUSION_AWARE_LOSS_METHOD
        or type(config["fusion_aware_loss_version"]) is not int
        or config["fusion_aware_loss_version"] != FUSION_AWARE_LOSS_VERSION
    ):
        raise ValueError("training run fusion objective metadata is invalid")
    raw_grid = config["fusion_aware_loss_alpha_grid"]
    if (
        not isinstance(raw_grid, (list, tuple))
        or len(raw_grid) != len(FUSION_AWARE_LOSS_ALPHA_GRID)
        or any(type(value) is not float for value in raw_grid)
        or tuple(raw_grid) != FUSION_AWARE_LOSS_ALPHA_GRID
    ):
        raise ValueError("training run fusion objective alpha grid is invalid")
    return FUSION_GRID_DRAWBACK_OBJECTIVE


def fusion_grid_drawback_objective_metadata() -> dict[str, object]:
    """Return a defensive copy of the frozen objective metadata."""

    return {
        **FUSION_GRID_DRAWBACK_OBJECTIVE_METADATA,
        "alpha_grid": list(FUSION_AWARE_LOSS_ALPHA_GRID),
    }


def _require_exact_fusion_objective_metadata(raw: object) -> None:
    if not isinstance(raw, Mapping):
        raise ValueError("checkpoint drawback objective metadata is invalid")
    if set(raw) != set(FUSION_GRID_DRAWBACK_OBJECTIVE_METADATA):
        raise ValueError("checkpoint drawback objective metadata is incomplete")
    if (
        type(raw["method"]) is not str
        or raw["method"] != FUSION_AWARE_LOSS_METHOD
        or type(raw["version"]) is not int
        or raw["version"] != FUSION_AWARE_LOSS_VERSION
        or type(raw["production_fusion_method"]) is not str
        or raw["production_fusion_method"] != RANK_PRESERVING_FUSION_METHOD
        or type(raw["aggregation"]) is not str
        or raw["aggregation"]
        != FUSION_GRID_DRAWBACK_OBJECTIVE_METADATA["aggregation"]
    ):
        raise ValueError("checkpoint drawback objective metadata is invalid")
    raw_grid = raw["alpha_grid"]
    if (
        not isinstance(raw_grid, (list, tuple))
        or len(raw_grid) != len(FUSION_AWARE_LOSS_ALPHA_GRID)
        or any(type(value) is not float for value in raw_grid)
        or tuple(raw_grid) != FUSION_AWARE_LOSS_ALPHA_GRID
    ):
        raise ValueError("checkpoint drawback objective alpha grid is invalid")


CHECKPOINT_INDEX_FORMAT = "drawbacktrainer-checkpoint-index"
CHECKPOINT_INDEX_VERSION = 1


def checkpoint_path(directory: Path, seed: int, epoch: int) -> Path:
    if seed < 0 or epoch < 0:
        raise ValueError("seed and epoch must be non-negative")
    return directory / f"baseline-seed-{seed:010d}-epoch-{epoch:04d}.pt"


def write_run_metadata(directory: Path, metadata: Mapping[str, Any]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "run.json"
    temporary = directory / "run.json.tmp"
    temporary.write_text(
        json.dumps(dict(metadata), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _pairs(items: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in items:
        if key in value:
            raise ValueError(f"checkpoint index repeats key {key!r}")
        value[key] = item
    return value


def _constant(token: str) -> object:
    raise ValueError(f"checkpoint index contains non-finite constant {token}")


def _digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def write_checkpoint_index(
    directory: Path,
    *,
    seed: int,
    epochs: int,
    checkpoint_sha256s: tuple[str, ...] | None = None,
) -> Path:
    """Publish an ordered content-addressed inventory for one completed run."""

    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be non-negative")
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs <= 0:
        raise ValueError("epochs must be positive")
    if checkpoint_sha256s is not None and len(checkpoint_sha256s) != epochs:
        raise ValueError("checkpoint digest count must equal epochs")
    claim = directory / "run.claim.json"
    if not claim.is_file() or claim.is_symlink():
        raise ValueError("training run claim is missing or invalid")
    claim_value, claim_payload = _load_run_claim(claim)
    config = claim_value["config"]
    assert isinstance(config, dict)
    if config.get("seed") != seed or config.get("epochs") != epochs:
        raise ValueError("training run claim seed or epochs disagree")
    run_id = claim_value["run_id"]
    assert isinstance(run_id, str)
    checkpoints: list[dict[str, object]] = []
    for epoch in range(1, epochs + 1):
        path = checkpoint_path(directory, seed, epoch)
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"checkpoint epoch {epoch} is missing or invalid")
        checkpoint_digest = (
            file_sha256(path)
            if checkpoint_sha256s is None
            else _digest(
                checkpoint_sha256s[epoch - 1],
                f"checkpoint epoch {epoch} sha256",
            )
        )
        checkpoints.append(
            {
                "epoch": epoch,
                "file": path.name,
                "sha256": checkpoint_digest,
            }
        )
    value = {
        "format": CHECKPOINT_INDEX_FORMAT,
        "version": CHECKPOINT_INDEX_VERSION,
        "seed": seed,
        "epochs": epochs,
        "runId": run_id,
        "runClaim": {
            "file": claim.name,
            "sha256": hashlib.sha256(claim_payload).hexdigest(),
        },
        "checkpoints": checkpoints,
    }
    rendered = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    path = directory / "checkpoint-index.claim.json"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".checkpoint-index.",
        suffix=".tmp",
        dir=directory,
    )
    temporary = Path(temporary_name)
    committed = False
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as target:
            target.write(rendered)
            target.flush()
            os.fsync(target.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(
                f"checkpoint index already exists: {path}"
            ) from error
        committed = True
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            if not committed:
                raise
    return path


def verify_checkpoint_index(
    path: Path,
    expected_sha256: str,
) -> Mapping[str, object]:
    """Authenticate an index and every exact local artifact it binds."""

    expected = _digest(expected_sha256, "checkpoint index sha256")
    if not path.is_file() or path.is_symlink():
        raise ValueError("checkpoint index authentication failed")
    try:
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError("checkpoint index authentication failed")
        value = json.loads(
            payload,
            object_pairs_hook=_pairs,
            parse_constant=_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("checkpoint index is not strict UTF-8 JSON") from error
    if not isinstance(value, dict) or set(value) != {
        "format",
        "version",
        "seed",
        "epochs",
        "runId",
        "runClaim",
        "checkpoints",
    }:
        raise ValueError("checkpoint index fields are invalid")
    rendered = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    if rendered != payload:
        raise ValueError("checkpoint index is not canonical")
    seed = value.get("seed")
    epochs = value.get("epochs")
    run_id = value.get("runId")
    if (
        value.get("format") != CHECKPOINT_INDEX_FORMAT
        or value.get("version") != CHECKPOINT_INDEX_VERSION
        or not isinstance(seed, int)
        or isinstance(seed, bool)
        or seed < 0
        or not isinstance(epochs, int)
        or isinstance(epochs, bool)
        or epochs <= 0
        or _digest(run_id, "checkpoint index runId") != run_id
    ):
        raise ValueError("checkpoint index identity is invalid")
    claim = value.get("runClaim")
    if (
        not isinstance(claim, dict)
        or set(claim) != {"file", "sha256"}
        or claim.get("file") != "run.claim.json"
    ):
        raise ValueError("checkpoint index run claim is invalid")
    claim_path = _verify_indexed_file(
        path.parent,
        "run.claim.json",
        _digest(claim.get("sha256"), "run claim sha256"),
    )
    claim_value, _claim_payload = _load_run_claim(claim_path)
    claim_config = claim_value["config"]
    assert isinstance(claim_config, dict)
    if (
        claim_value.get("run_id") != run_id
        or claim_config.get("seed") != seed
        or claim_config.get("epochs") != epochs
    ):
        raise ValueError("checkpoint index disagrees with training run claim")
    checkpoints = value.get("checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != epochs:
        raise ValueError("checkpoint index epoch count is invalid")
    for epoch, item in enumerate(checkpoints, 1):
        expected_file = checkpoint_path(path.parent, seed, epoch).name
        if (
            not isinstance(item, dict)
            or set(item) != {"epoch", "file", "sha256"}
            or item.get("epoch") != epoch
            or item.get("file") != expected_file
        ):
            raise ValueError("checkpoint index epoch identity is invalid")
        _verify_indexed_file(
            path.parent,
            expected_file,
            _digest(item.get("sha256"), f"checkpoint epoch {epoch} sha256"),
        )
    return value


def _verify_indexed_file(
    directory: Path,
    name: str,
    expected_sha256: str,
) -> Path:
    candidate = directory / name
    if (
        candidate.parent != directory
        or not candidate.is_file()
        or candidate.is_symlink()
        or file_sha256(candidate) != expected_sha256
    ):
        raise ValueError(f"indexed artifact authentication failed: {name}")
    return candidate


def _load_run_claim(path: Path) -> tuple[Mapping[str, object], bytes]:
    try:
        payload = path.read_bytes()
        value = json.loads(
            payload,
            object_pairs_hook=_pairs,
            parse_constant=_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("training run claim is not strict UTF-8 JSON") from error
    if not isinstance(value, dict) or set(value) != {
        "run_id",
        "format",
        "version",
        "config",
        "runtime",
        "sampling",
    }:
        raise ValueError("training run claim fields are invalid")
    if (
        value.get("format") != "drawbacktrainer-streaming-run"
        or value.get("version") != 1
        or not isinstance(value.get("config"), dict)
        or not isinstance(value.get("runtime"), dict)
        or not isinstance(value.get("sampling"), dict)
    ):
        raise ValueError("training run claim identity is invalid")
    config = value["config"]
    assert isinstance(config, dict)
    claim_seed = config.get("seed")
    claim_epochs = config.get("epochs")
    if (
        not isinstance(claim_seed, int)
        or isinstance(claim_seed, bool)
        or claim_seed < 0
        or not isinstance(claim_epochs, int)
        or isinstance(claim_epochs, bool)
        or claim_epochs <= 0
    ):
        raise ValueError("training run claim seed or epochs are invalid")
    canonical = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    if payload != canonical:
        raise ValueError("training run claim is not canonical")
    material = {
        "format": value["format"],
        "version": value["version"],
        "config": value["config"],
        "runtime": value["runtime"],
        "sampling": value["sampling"],
    }
    computed = hashlib.sha256(
        json.dumps(
            material,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if _digest(value.get("run_id"), "training run id") != computed:
        raise ValueError("training run id does not match claim material")
    return value, payload


def checkpoint_metadata(
    *,
    seed: int,
    epoch: int,
    drawback_vocabulary: list[str],
    parameter_vocabulary: list[str],
    model_config: Mapping[str, Any] | None = None,
    training_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "format_version": 3,
        "seed": seed,
        "epoch": epoch,
        "drawback_vocabulary": list(drawback_vocabulary),
        "parameter_vocabulary": list(parameter_vocabulary),
        "model_config": None if model_config is None else dict(model_config),
        "training_metadata": (
            None if training_metadata is None else dict(training_metadata)
        ),
    }


def save_checkpoint(
    directory: Path,
    *,
    model: Any,
    optimizer: Any,
    seed: int,
    epoch: int,
    drawback_vocabulary: list[str],
    parameter_vocabulary: list[str],
    model_config: Mapping[str, Any],
    training_metadata: Mapping[str, Any],
) -> Path:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required to save a training checkpoint") from error
    directory.mkdir(parents=True, exist_ok=True)
    path = checkpoint_path(directory, seed, epoch)
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            **checkpoint_metadata(
                seed=seed,
                epoch=epoch,
                drawback_vocabulary=drawback_vocabulary,
                parameter_vocabulary=parameter_vocabulary,
                model_config=model_config,
                training_metadata=training_metadata,
            ),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        temporary,
    )
    temporary.replace(path)
    return path
