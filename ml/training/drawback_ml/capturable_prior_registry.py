"""Canonical inventory of every private corpus that predates a sealed test."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from .capturable_baseline import _canonical_json, _publish_bytes
from .capturable_candidate_selection import _selection_report
from .capturable_records import CapturableDatasetError


PRIOR_CORPUS_REGISTRY_FORMAT = (
    "drawbackguesser-capturable-prior-corpus-registry"
)
PRIOR_CORPUS_REGISTRY_VERSION = 1


def _object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CapturableDatasetError(
                f"prior corpus row contains duplicate key {key!r}"
            )
        result[key] = value
    return result


def _row_game_id(value: object, label: str) -> str:
    if not isinstance(value, Mapping):
        raise CapturableDatasetError(f"{label} must be a JSON object")
    game_id = value.get("gameId")
    if game_id is None:
        evaluation = value.get("evaluation")
        game_id = (
            evaluation.get("gameId")
            if isinstance(evaluation, Mapping)
            else None
        )
    if not isinstance(game_id, str) or not game_id:
        raise CapturableDatasetError(f"{label} has no valid gameId")
    return game_id


def _safe_relative_name(root: Path, path: Path) -> str:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise CapturableDatasetError(
            "prior corpus escaped the registry root"
        ) from error
    name = relative.as_posix()
    pure = PurePosixPath(name)
    if (
        pure.is_absolute()
        or name in {"", ".", ".."}
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise CapturableDatasetError(
            "prior corpus has an unsafe relative name"
        )
    return name


def _inventory_source(
    root: Path,
    path: Path,
) -> tuple[Mapping[str, Any], set[str]]:
    if path.is_symlink() or not path.is_file():
        raise CapturableDatasetError(
            f"{path.name} is not a regular prior corpus file"
        )
    digest = hashlib.sha256()
    game_ids: set[str] = set()
    line_count = 0
    byte_count = 0
    try:
        with path.open("rb") as source:
            for raw_line in source:
                line_count += 1
                byte_count += len(raw_line)
                digest.update(raw_line)
                if not raw_line.strip():
                    raise CapturableDatasetError(
                        f"{path.name}:{line_count} is blank"
                    )
                try:
                    value = json.loads(
                        raw_line.decode("utf-8"),
                        object_pairs_hook=_object_without_duplicate_keys,
                        parse_constant=lambda token: (_ for _ in ()).throw(
                            CapturableDatasetError(
                                f"{path.name}:{line_count} contains {token}"
                            )
                        ),
                    )
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise CapturableDatasetError(
                        f"{path.name}:{line_count} is not strict UTF-8 JSON"
                    ) from error
                game_ids.add(
                    _row_game_id(
                        value,
                        f"{path.name}:{line_count}",
                    )
                )
    except OSError as error:
        raise CapturableDatasetError(
            f"cannot read prior corpus {path.name}"
        ) from error
    if line_count == 0 or not game_ids:
        raise CapturableDatasetError(
            f"{path.name} is an empty prior corpus"
        )
    return (
        {
            "bytes": byte_count,
            "file": _safe_relative_name(root, path),
            "games": len(game_ids),
            "lines": line_count,
            "sha256": digest.hexdigest(),
        },
        game_ids,
    )


def build_prior_corpus_registry(
    root: Path,
    output_path: Path,
) -> Mapping[str, Any]:
    """Freeze all existing NDJSON corpus bytes and their game-ID union."""

    if output_path.exists():
        raise FileExistsError("prior corpus registry output already exists")
    resolved_root = root.resolve()
    if not resolved_root.is_dir():
        raise CapturableDatasetError(
            "prior corpus registry root must be a directory"
        )
    sources = sorted(
        (
            path
            for path in resolved_root.rglob("*.ndjson")
            if path.resolve() != output_path.resolve()
        ),
        key=lambda path: _safe_relative_name(resolved_root, path),
    )
    if not sources:
        raise CapturableDatasetError(
            "prior corpus registry found no NDJSON sources"
        )

    identities: list[Mapping[str, Any]] = []
    all_game_ids: set[str] = set()
    for source in sources:
        identity, game_ids = _inventory_source(resolved_root, source)
        identities.append(identity)
        all_game_ids.update(game_ids)
    artifact = {
        "createdBeforeFreshTest": True,
        "format": PRIOR_CORPUS_REGISTRY_FORMAT,
        "gameIds": sorted(all_game_ids),
        "rootName": resolved_root.name,
        "sourceCount": len(identities),
        "sourcePattern": "**/*.ndjson",
        "sources": identities,
        "uniqueGameCount": len(all_game_ids),
        "version": PRIOR_CORPUS_REGISTRY_VERSION,
    }
    payload = _canonical_json(artifact)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _publish_bytes(output_path, payload)
    return {
        "artifactPath": str(output_path),
        "artifactSha256": hashlib.sha256(payload).hexdigest(),
        "sourceCount": artifact["sourceCount"],
        "uniqueGameCount": artifact["uniqueGameCount"],
    }


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(token in "0123456789abcdef" for token in value)
    )


def load_prior_corpus_registry(
    path: Path,
) -> tuple[Mapping[str, Any], str]:
    """Authenticate canonical registry structure without reopening sources."""

    artifact, sha256 = _selection_report(path)
    expected_keys = {
        "createdBeforeFreshTest",
        "format",
        "gameIds",
        "rootName",
        "sourceCount",
        "sourcePattern",
        "sources",
        "uniqueGameCount",
        "version",
    }
    sources = artifact.get("sources")
    game_ids = artifact.get("gameIds")
    if (
        set(artifact) != expected_keys
        or artifact.get("format") != PRIOR_CORPUS_REGISTRY_FORMAT
        or artifact.get("version") != PRIOR_CORPUS_REGISTRY_VERSION
        or artifact.get("createdBeforeFreshTest") is not True
        or artifact.get("sourcePattern") != "**/*.ndjson"
        or not isinstance(artifact.get("rootName"), str)
        or not artifact["rootName"]
        or not isinstance(sources, list)
        or not sources
        or not isinstance(game_ids, list)
        or not game_ids
        or game_ids != sorted(set(game_ids))
        or any(not isinstance(game_id, str) or not game_id for game_id in game_ids)
        or artifact.get("sourceCount") != len(sources)
        or artifact.get("uniqueGameCount") != len(game_ids)
    ):
        raise CapturableDatasetError(
            f"{path.name} is not a compatible prior corpus registry"
        )
    names: list[str] = []
    for source in sources:
        if not isinstance(source, Mapping) or set(source) != {
            "bytes",
            "file",
            "games",
            "lines",
            "sha256",
        }:
            raise CapturableDatasetError(
                f"{path.name} has an invalid source identity"
            )
        name = source.get("file")
        pure = PurePosixPath(name) if isinstance(name, str) else None
        if (
            pure is None
            or pure.is_absolute()
            or not pure.parts
            or any(part in {"", ".", ".."} for part in pure.parts)
            or isinstance(source.get("bytes"), bool)
            or not isinstance(source.get("bytes"), int)
            or source["bytes"] <= 0
            or isinstance(source.get("lines"), bool)
            or not isinstance(source.get("lines"), int)
            or source["lines"] <= 0
            or isinstance(source.get("games"), bool)
            or not isinstance(source.get("games"), int)
            or source["games"] <= 0
            or source["games"] > source["lines"]
            or not _is_sha256(source.get("sha256"))
        ):
            raise CapturableDatasetError(
                f"{path.name} has an invalid source identity"
            )
        names.append(str(name))
    if names != sorted(set(names)):
        raise CapturableDatasetError(
            f"{path.name} source inventory is not ordered and unique"
        )
    return artifact, sha256


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze every existing private NDJSON corpus before a sealed "
            "capturable test is generated."
        )
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    options = _parser().parse_args(arguments)
    result = build_prior_corpus_registry(options.root, options.output)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
