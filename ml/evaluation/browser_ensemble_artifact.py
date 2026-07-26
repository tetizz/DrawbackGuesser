"""Canonical browser export for a verified calibrated three-member ensemble."""

from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import BinaryIO, Mapping

from ml.training.drawback_ml.browser_artifact import (
    BROWSER_ARTIFACT_FORMAT,
    BROWSER_HYBRID_ARTIFACT_VERSION,
    BROWSER_TENSOR_ENCODING,
    BrowserArtifactError,
    build_browser_artifact,
)

from .ensemble_calibration import (
    FUSION_METHOD,
    ContentAddressedFile,
    load_ensemble_calibration,
)
from .ensemble_release import (
    ENSEMBLE_TRAINING_SEEDS,
    LoadedEnsembleRelease,
    resolve_member_checkpoint,
    verify_ensemble_release,
)
from .release_selection_bundle import ContentAddressedJson


BROWSER_ENSEMBLE_ARTIFACT_VERSION = 4
BROWSER_ENSEMBLE_MODEL_VARIANT = "v21-hybrid-ensemble"
MAX_BROWSER_ENSEMBLE_ARTIFACT_BYTES = 32 * 1024 * 1024
MEMBER_SHARED_FIELDS = (
    "featureSchemaVersion",
    "symbolicFeatureVersion",
    "drawbackVocabulary",
    "symbolicRuleIds",
    "tokenizer",
    "tensorEncoding",
    "dimensions",
)


def export_browser_ensemble_artifact(
    ensemble_reference: ContentAddressedJson,
    calibration_reference: ContentAddressedFile,
    output: Path,
) -> Path:
    """Verify, snapshot, render, reverify, and atomically publish format v4."""

    loaded_ensemble = verify_ensemble_release(ensemble_reference)
    calibration = load_ensemble_calibration(calibration_reference)
    _verify_calibration_binding(
        calibration,
        ensemble_reference,
        loaded_ensemble,
    )
    with ExitStack() as stack:
        checkpoint_sources = _open_checkpoint_sources(
            stack,
            ensemble_reference,
            loaded_ensemble,
        )
        checkpoint_payloads = tuple(
            _read_and_verify_checkpoint(source, member.checkpoint_sha256)
            for source, member in zip(
                checkpoint_sources,
                loaded_ensemble.members,
                strict=True,
            )
        )
        artifact = _build_artifact(
            ensemble_reference,
            calibration_reference,
            loaded_ensemble,
            calibration,
            checkpoint_payloads,
        )
        rendered = _canonical_bytes(artifact)
        if len(rendered) > MAX_BROWSER_ENSEMBLE_ARTIFACT_BYTES:
            raise BrowserArtifactError(
                "browser ensemble artifact exceeds "
                f"{MAX_BROWSER_ENSEMBLE_ARTIFACT_BYTES} bytes"
            )
        for source, member in zip(
            checkpoint_sources,
            loaded_ensemble.members,
            strict=True,
        ):
            _read_and_verify_checkpoint(source, member.checkpoint_sha256)
        if verify_ensemble_release(ensemble_reference) != loaded_ensemble:
            raise BrowserArtifactError(
                "ensemble release changed during browser export"
            )
        if load_ensemble_calibration(calibration_reference) != calibration:
            raise BrowserArtifactError(
                "ensemble calibration changed during browser export"
            )
        _write_atomic_no_clobber(output, rendered)
    return output


def _build_artifact(
    ensemble_reference: ContentAddressedJson,
    calibration_reference: ContentAddressedFile,
    loaded_ensemble: LoadedEnsembleRelease,
    calibration: Mapping[str, object],
    checkpoint_payloads: tuple[bytes, bytes, bytes],
) -> dict[str, object]:
    members = tuple(
        build_browser_artifact(payload) for payload in checkpoint_payloads
    )
    first = members[0]
    if (
        first.get("format") != BROWSER_ARTIFACT_FORMAT
        or first.get("formatVersion") != BROWSER_HYBRID_ARTIFACT_VERSION
        or first.get("modelVariant") != "v21-hybrid"
        or first.get("tensorEncoding") != BROWSER_TENSOR_ENCODING
    ):
        raise BrowserArtifactError("ensemble members must be v21-hybrid")
    for index, member in enumerate(members):
        if (
            member.get("format") != BROWSER_ARTIFACT_FORMAT
            or member.get("formatVersion") != BROWSER_HYBRID_ARTIFACT_VERSION
            or member.get("modelVariant") != "v21-hybrid"
            or member.get("tensorEncoding") != BROWSER_TENSOR_ENCODING
        ):
            raise BrowserArtifactError("ensemble members must be v21-hybrid")
        if any(member.get(field) != first.get(field) for field in MEMBER_SHARED_FIELDS):
            raise BrowserArtifactError(
                f"ensemble member {index} has an incompatible shared contract"
            )
        expected_digest = loaded_ensemble.members[index].checkpoint_sha256
        if member.get("sourceCheckpointSha256") != expected_digest:
            raise BrowserArtifactError(
                f"ensemble member {index} checkpoint digest is inconsistent"
            )
    method = _mapping(calibration.get("method"), "calibration method")
    identity = _mapping(calibration.get("identity"), "calibration identity")
    fusion_selection_sha256 = identity.get("fusion_selection_sha256")
    selected_alpha = identity.get("selected_alpha")
    if (
        not isinstance(fusion_selection_sha256, str)
        or len(fusion_selection_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in fusion_selection_sha256
        )
        or isinstance(selected_alpha, bool)
        or not isinstance(selected_alpha, (int, float))
        or not 0.0 <= float(selected_alpha) <= 1.0
    ):
        raise BrowserArtifactError(
            "calibration identity lacks a valid fusion selection policy"
        )
    white = _mapping(calibration.get("white"), "White calibration")
    black = _mapping(calibration.get("black"), "Black calibration")
    return {
        "format": BROWSER_ARTIFACT_FORMAT,
        "formatVersion": BROWSER_ENSEMBLE_ARTIFACT_VERSION,
        "modelVariant": BROWSER_ENSEMBLE_MODEL_VARIANT,
        **{field: first[field] for field in MEMBER_SHARED_FIELDS},
        "ensemble": {
            "method": FUSION_METHOD,
            "memberCount": 3,
            "seedOrder": list(ENSEMBLE_TRAINING_SEEDS),
            "sourceEnsembleReleaseSha256": ensemble_reference.sha256,
            "sourceFusionSelectionSha256": fusion_selection_sha256,
            "selectedAlpha": float(selected_alpha),
            "members": [
                {
                    "trainingSeed": release_member.training_seed,
                    "selectedEpoch": release_member.checkpoint_epoch,
                    "trainingRunId": release_member.training_run_id,
                    "sourceSelectionSha256": release_member.selection_sha256,
                    "sourceCheckpointSha256": release_member.checkpoint_sha256,
                    "tensors": member["tensors"],
                }
                for release_member, member in zip(
                    loaded_ensemble.members,
                    members,
                    strict=True,
                )
            ],
        },
        "calibration": {
            "sourceCalibrationSha256": calibration_reference.sha256,
            "method": method["name"],
            "preservesHardEliminations": method[
                "preserves_hard_eliminations"
            ],
            "white": _calibration_head(white, "White"),
            "black": _calibration_head(black, "Black"),
        },
    }


def _verify_calibration_binding(
    calibration: Mapping[str, object],
    ensemble_reference: ContentAddressedJson,
    loaded_ensemble: LoadedEnsembleRelease,
) -> None:
    identity = _mapping(calibration.get("identity"), "calibration identity")
    if identity.get("ensemble_release_sha256") != ensemble_reference.sha256:
        raise BrowserArtifactError(
            "calibration does not bind the selected ensemble release"
        )
    fusion_selection_sha256 = identity.get("fusion_selection_sha256")
    selected_alpha = identity.get("selected_alpha")
    if (
        not isinstance(fusion_selection_sha256, str)
        or len(fusion_selection_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in fusion_selection_sha256
        )
        or isinstance(selected_alpha, bool)
        or not isinstance(selected_alpha, (int, float))
        or not 0.0 <= float(selected_alpha) <= 1.0
    ):
        raise BrowserArtifactError(
            "calibration does not bind a valid fusion selection policy"
        )
    raw_members = identity.get("members")
    if not isinstance(raw_members, list) or len(raw_members) != 3:
        raise BrowserArtifactError(
            "calibration identity must bind exactly three members"
        )
    for raw, release_member, seed in zip(
        raw_members,
        loaded_ensemble.members,
        ENSEMBLE_TRAINING_SEEDS,
        strict=True,
    ):
        member = _mapping(raw, "calibration member")
        if (
            member.get("seed") != seed
            or member.get("checkpoint_sha256")
            != release_member.checkpoint_sha256
        ):
            raise BrowserArtifactError(
                "calibration member binding disagrees with ensemble release"
            )
    method = _mapping(calibration.get("method"), "calibration method")
    if (
        method.get("name") != "per-head-multiclass-temperature-scaling"
        or method.get("fusion") != FUSION_METHOD
        or method.get("preserves_hard_eliminations") is not True
    ):
        raise BrowserArtifactError("calibration method is incompatible")


def _open_checkpoint_sources(
    stack: ExitStack,
    ensemble_reference: ContentAddressedJson,
    loaded: LoadedEnsembleRelease,
) -> tuple[BinaryIO, BinaryIO, BinaryIO]:
    sources: list[BinaryIO] = []
    for member in loaded.members:
        try:
            resolved = resolve_member_checkpoint(
                ensemble_reference,
                member,
            )
            source = stack.enter_context(resolved.open("rb"))
        except (OSError, ValueError) as error:
            raise BrowserArtifactError(
                "ensemble checkpoint source authentication failed"
            ) from error
        sources.append(source)
    if len(sources) != 3:
        raise BrowserArtifactError("ensemble release must contain three checkpoints")
    return sources[0], sources[1], sources[2]


def _read_and_verify_checkpoint(source: BinaryIO, expected_sha256: str) -> bytes:
    source.seek(0)
    payload = source.read()
    source.seek(0)
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise BrowserArtifactError("ensemble checkpoint SHA-256 does not match")
    return payload


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise BrowserArtifactError(f"{label} must be an object")
    return value


def _calibration_head(
    value: Mapping[str, object],
    label: str,
) -> dict[str, object]:
    expected = {
        "format_version",
        "method",
        "fitted_split",
        "temperature",
        "example_count",
        "nll_before",
        "nll_after",
        "preserves_hard_eliminations",
    }
    if set(value) != expected:
        raise BrowserArtifactError(f"{label} calibration fields are incompatible")
    if (
        value.get("method") != "multiclass-temperature-scaling"
        or value.get("fitted_split") != "validation"
        or value.get("preserves_hard_eliminations") is not True
    ):
        raise BrowserArtifactError(f"{label} calibration metadata is incompatible")
    return {
        "temperature": value["temperature"],
        "exampleCount": value["example_count"],
        "nllBefore": value["nll_before"],
        "nllAfter": value["nll_after"],
    }


def _canonical_bytes(value: Mapping[str, object]) -> bytes:
    try:
        rendered = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise BrowserArtifactError(
            "browser ensemble artifact is not canonical JSON data"
        ) from error
    return (rendered + "\n").encode("utf-8")


def _write_atomic_no_clobber(output: Path, payload: bytes) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise BrowserArtifactError(
                f"refusing to overwrite browser ensemble artifact: {output}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)
