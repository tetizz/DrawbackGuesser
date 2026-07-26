"""Model-free preparation and verification of the fixed confirmation corpus."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any, Iterator, Mapping, Sequence

from .capturable_baseline import _canonical_json
from .capturable_blend import (
    _authenticated_execution_identity,
    _authenticated_git_runner,
    _windows_known_directory,
)
from .capturable_candidate_selection import _selection_report
from .capturable_fixed_blend_contract import (
    CONFIRMATION_TEST_FILE,
    CONFIRMATION_TRACE_FILE,
    CORPUS_RECEIPT_FILE,
    CORPUS_RECEIPT_FORMAT,
    CORPUS_RECEIPT_VERSION,
    FIXED_PROTOCOL_COMMIT,
    FIXED_PROTOCOL_FILE,
    FIXED_PROTOCOL_SHA256,
    GENERATION_SCHEDULE,
    PRIOR_REGISTRY_FILE,
    PRIOR_REGISTRY_GAME_COUNT,
    PRIOR_REGISTRY_SHA256,
    PRIOR_REGISTRY_SOURCE_COUNT,
    audit_confirmation_rows,
)
from .capturable_fixed_schedule import EXPECTED_FIXED_ASSIGNMENTS
from .capturable_prior_registry import load_prior_corpus_registry
from .capturable_records import (
    CapturableDatasetError,
    CapturableDatasetRow,
    parse_capturable_dataset_row,
    strict_json_object,
)
from .durable_publish import publish_bytes_durable


AUDIT_TOOL_ID = "capturable-fixed-confirmation-audit/v1"
CONVERSION_ID = "player-private-trace-to-schema8/v1"
GENERATOR_DIRECTORY = "engine-generator-74eb6fc"
GENERATOR_ENGINE_COMMIT = (
    "74eb6fc95571994bd96b7a351278f3f74f0972e3"
)
GENERATOR_LOCK_SHA256 = (
    "b2a2d051ce12f6313eb3692969a46cad0675fbe22c63ea71d8feaeb39bee8307"
)
ENGINE_SUBMODULE_COMMIT = (
    "436407b51b983ba9c173f93f6c6d08920a36825f"
)
STANDARD_INITIAL_FEN = (
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
)
TRACE_FORMAT = "drawbackengine-player-private-simulation-trace"
TRACE_AUTHORITY = "capturable-king/v1"
TRACE_RANDOM_POLICY = {
    "kind": "explicit-parameter-seeds-domain-agent-mulberry32",
    "version": 1,
}
TRACE_RULESET = {"kind": "audited-player-private", "version": 2}
TRACE_HYPOTHESIS_POLICY = {
    "kind": "unrestricted-baseline",
    "version": 1,
}
TRACE_SEARCH_POLICY = {
    "evaluatorId": "drawback-material/v1",
    "leafCacheEntries": 16_384,
    "leafCacheHistoryMode": "full",
    "maxDepth": 1,
    "maxNodes": 5_000,
    "opponentAggregation": "worst-case",
    "policyId": "material-player-private-corpus/v1",
    "temperatureCp": 35,
    "topK": 8,
}
TRACE_AGENT = {
    "id": "material-player-private-corpus/v1",
    "searchPolicy": TRACE_SEARCH_POLICY,
    "strength": None,
    "style": "drawback-search",
}
TRACE_TOP_LEVEL_KEYS = {
    "agents",
    "authorityId",
    "finalPosition",
    "format",
    "gameId",
    "gameIndex",
    "hypothesisPolicy",
    "initialPosition",
    "parameterSeeds",
    "plies",
    "plyLimit",
    "randomPolicy",
    "result",
    "ruleset",
    "schemaVersion",
    "secrets",
    "seed",
    "stoppedAtPlyLimit",
}
TRACE_RESULT_KINDS = (
    "active",
    "checkmate",
    "draw",
    "drawback-loss",
    "king-capture",
    "no-legal-moves",
)
MAX_TRACE_LINE_BYTES = 64 * 1024 * 1024
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
REQUIRED_NODE_VERSION = "v24.15.0"
REQUIRED_COREPACK_VERSION = "0.34.6"
REQUIRED_PNPM_VERSION = "11.9.0"
PNPM_RUNTIME_TREE_HASH_ID = "pnpm-runtime-tree/v1"
PNPM_RUNTIME_TREE_FILES = 451
PNPM_RUNTIME_TREE_SHA256 = (
    "07a0dd6cd5047b0fba062374b548723df3e969494a1d633c203c9ddf2b8e937f"
)
BUILD_TREE_HASH_ID = "relative-path-size-content-sha256/v1"
ENGINE_WORKSPACE_DIRECTORIES = (
    "apps/engine-cli",
    "packages/chess-core",
    "packages/chess-evaluator",
    "packages/drawback-engine",
    "packages/drawback-search",
    "packages/probe-search",
    "packages/shared",
    "packages/simulation-arena",
    "packages/simulation-trace",
)
ENGINE_SCRUB_PATHS = (
    "node_modules",
    *(
        relative
        for workspace in ENGINE_WORKSPACE_DIRECTORIES
        for relative in (f"{workspace}/node_modules", f"{workspace}/dist")
    ),
)
ENGINE_DIST_PATHS = tuple(
    f"{workspace}/dist" for workspace in ENGINE_WORKSPACE_DIRECTORIES
)
ENGINE_REQUIRED_DIST_FILES = (
    "apps/engine-cli/dist/player-private-batch-cli.js",
    *(
        f"{workspace}/dist/index.js"
        for workspace in ENGINE_WORKSPACE_DIRECTORIES
        if workspace != "apps/engine-cli"
    ),
)
GUESSER_WORKSPACE_DIRECTORIES = (
    "apps/dataset-cli",
    "packages/dataset-contract",
    "packages/predictor",
    "packages/trace-to-dataset",
)
GUESSER_ENGINE_WORKSPACE_DIRECTORIES = (
    "packages/chess-core",
    "packages/drawback-engine",
    "packages/shared",
    "packages/simulation-trace",
)
GUESSER_SCRUB_PATHS = (
    "node_modules",
    *(
        relative
        for workspace in GUESSER_WORKSPACE_DIRECTORIES
        for relative in (f"{workspace}/node_modules", f"{workspace}/dist")
    ),
)
GUESSER_ALLOWED_RUNTIME_PATHS = (
    *GUESSER_SCRUB_PATHS,
    "apps/web/dist",
    "apps/web/node_modules",
)
GUESSER_ENGINE_SCRUB_PATHS = (
    "node_modules",
    *(
        relative
        for workspace in GUESSER_ENGINE_WORKSPACE_DIRECTORIES
        for relative in (f"{workspace}/node_modules", f"{workspace}/dist")
    ),
)
GUESSER_ENGINE_ALLOWED_RUNTIME_PATHS = (
    "node_modules",
    *(
        relative
        for workspace in ENGINE_WORKSPACE_DIRECTORIES
        for relative in (f"{workspace}/node_modules", f"{workspace}/dist")
    ),
)
GUESSER_DIST_PATHS = tuple(
    f"{workspace}/dist" for workspace in GUESSER_WORKSPACE_DIRECTORIES
)
GUESSER_REQUIRED_DIST_FILES = (
    "apps/dataset-cli/dist/cli.js",
    *(
        f"{workspace}/dist/index.js"
        for workspace in GUESSER_WORKSPACE_DIRECTORIES
        if workspace != "apps/dataset-cli"
    ),
)
GUESSER_ENGINE_DIST_PATHS = tuple(
    f"{workspace}/dist"
    for workspace in GUESSER_ENGINE_WORKSPACE_DIRECTORIES
)
GUESSER_ENGINE_REQUIRED_DIST_FILES = tuple(
    f"{workspace}/dist/index.js"
    for workspace in GUESSER_ENGINE_WORKSPACE_DIRECTORIES
)
PACKAGE_MANAGER_ENVIRONMENT_PREFIXES = (
    "COREPACK",
    "NPM_",
    "PNPM_",
    "YARN_",
)


@dataclass(frozen=True)
class FileIdentity:
    bytes: int
    lines: int
    sha256: str


@dataclass(frozen=True)
class CorpusVerification:
    artifact: Mapping[str, Any]
    rows: tuple[CapturableDatasetRow, ...]
    test_sha256: str
    corpus: Mapping[str, Any]


@dataclass(frozen=True)
class ResolvedToolchain:
    node: Path
    corepack: Path
    pnpm_entrypoint: Path
    pnpm_store: Path
    shell: Path
    environment: Mapping[str, str]
    artifact: Mapping[str, Any]
    runtime_owner: tempfile.TemporaryDirectory[str] | None

    def pnpm(self, *arguments: str) -> list[str]:
        return [
            str(self.node),
            str(self.pnpm_entrypoint),
            f"--config.globalconfig={os.devnull}",
            f"--config.store-dir={self.pnpm_store}",
            f"--config.userconfig={os.devnull}",
            "--config.offline=true",
            *arguments,
        ]


def require_isolated_python_runtime() -> None:
    if (
        sys.version_info[:2] != (3, 11)
        or not sys.dont_write_bytecode
        or sys.flags.ignore_environment != 1
        or sys.flags.no_user_site != 1
    ):
        raise CapturableDatasetError(
            "fixed confirmation requires Python 3.11 with -B -E -s"
        )


def _is_link_or_junction(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return path.is_symlink()
    return path.is_symlink() or (
        os.name == "nt"
        and bool(
            getattr(metadata, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x0400)
        )
    )


def require_private_root(path: Path) -> Path:
    root = path.resolve()
    public_roots = (
        REPOSITORY_ROOT,
        (REPOSITORY_ROOT / "engine").resolve(),
        (REPOSITORY_ROOT.parent / "DrawbackEngine").resolve(),
    )
    if (
        not root.is_dir()
        or _is_link_or_junction(path)
        or any(
            root == public_root or root.is_relative_to(public_root)
            for public_root in public_roots
        )
    ):
        raise CapturableDatasetError(
            "fixed confirmation root must be a real private directory "
            "outside both public repositories"
        )
    return root


def require_private_regular_file(
    root: Path,
    path: Path,
    filename: str,
    label: str,
) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise CapturableDatasetError(
            f"{label} is not the fixed private regular file"
        ) from error
    if (
        resolved.name != filename
        or not resolved.is_file()
        or not resolved.is_relative_to(root)
    ):
        raise CapturableDatasetError(
            f"{label} is not the fixed private regular file"
        )
    current = path.absolute()
    while True:
        if _is_link_or_junction(current):
            raise CapturableDatasetError(
                f"{label} may not traverse a link or junction"
            )
        if current.resolve() == root:
            break
        parent = current.parent
        if parent == current or not parent.resolve().is_relative_to(root):
            raise CapturableDatasetError(
                f"{label} escapes the private root"
            )
        current = parent
    descriptor = -1
    try:
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        current_identity = path.stat(follow_symlinks=False)
    except OSError as error:
        raise CapturableDatasetError(
            f"{label} is not a stable private regular file"
        ) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if (
        not stat.S_ISREG(opened.st_mode)
        or opened.st_nlink != 1
        or not os.path.samestat(opened, current_identity)
    ):
        raise CapturableDatasetError(
            f"{label} must be a single-link private regular file"
        )
    return resolved


def _strict_json(payload: bytes, label: str) -> Mapping[str, Any]:
    def pairs(items: Sequence[tuple[str, Any]]) -> Mapping[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise CapturableDatasetError(
                    f"{label} contains duplicate key {key}"
                )
            result[key] = value
        return result

    def reject_constant(token: str) -> Any:
        raise CapturableDatasetError(
            f"{label} contains non-finite number {token}"
        )

    try:
        value = json.loads(
            payload,
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, UnicodeError) as error:
        raise CapturableDatasetError(f"{label} is invalid JSON") from error
    if not isinstance(value, Mapping):
        raise CapturableDatasetError(f"{label} must be a JSON object")
    return value


def _strict_file_identity(path: Path, label: str) -> FileIdentity:
    digest = hashlib.sha256()
    byte_count = 0
    line_count = 0
    with path.open("rb") as source:
        while True:
            line = source.readline(MAX_TRACE_LINE_BYTES + 1)
            if not line:
                break
            if len(line) > MAX_TRACE_LINE_BYTES:
                raise CapturableDatasetError(
                    f"{label} contains an oversized line"
                )
            if (
                not line.endswith(b"\n")
                or b"\r" in line
                or line in {b"\n", b""}
                or (line_count == 0 and line.startswith(b"\xef\xbb\xbf"))
            ):
                raise CapturableDatasetError(
                    f"{label} must be nonblank UTF-8 with exact LF framing"
                )
            try:
                line.decode("utf-8", errors="strict")
            except UnicodeError as error:
                raise CapturableDatasetError(
                    f"{label} is not strict UTF-8"
                ) from error
            digest.update(line)
            byte_count += len(line)
            line_count += 1
    if line_count == 0:
        raise CapturableDatasetError(f"{label} is empty")
    return FileIdentity(byte_count, line_count, digest.hexdigest())


def _load_pinned_capturable_dataset(
    path: Path,
) -> tuple[tuple[CapturableDatasetRow, ...], FileIdentity]:
    digest = hashlib.sha256()
    rows: list[CapturableDatasetRow] = []
    byte_count = 0
    with path.open("rb") as source:
        before = os.fstat(source.fileno())
        for line_number, raw_line in enumerate(source, start=1):
            label = f"{path.name}:{line_number}"
            if (
                not raw_line.endswith(b"\n")
                or raw_line.endswith(b"\r\n")
            ):
                raise CapturableDatasetError(
                    f"{label} must use canonical LF framing"
                )
            try:
                line = raw_line[:-1].decode(
                    "utf-8",
                    errors="strict",
                )
            except UnicodeDecodeError as error:
                raise CapturableDatasetError(
                    f"{label} is not UTF-8"
                ) from error
            if not line:
                raise CapturableDatasetError(f"{label} is blank")
            rows.append(
                parse_capturable_dataset_row(
                    strict_json_object(line, label)
                )
            )
            digest.update(raw_line)
            byte_count += len(raw_line)
        after = os.fstat(source.fileno())
    if not rows:
        raise CapturableDatasetError(
            "confirmation dataset must contain rows"
        )
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity or byte_count != before.st_size:
        raise CapturableDatasetError(
            "confirmation dataset changed through its pinned handle"
        )
    return (
        tuple(rows),
        FileIdentity(
            bytes=byte_count,
            lines=len(rows),
            sha256=digest.hexdigest(),
        ),
    )


def _trace_records(path: Path) -> Iterator[Mapping[str, Any]]:
    with path.open("rb") as source:
        for line_number, framed in enumerate(source, start=1):
            raw = framed[:-1]
            record = _strict_json(
                raw,
                f"{path.name}:{line_number}",
            )
            canonical = json.dumps(
                record,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            if canonical != raw:
                raise CapturableDatasetError(
                    f"{path.name}:{line_number} is not canonical trace JSON"
                )
            yield record


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CapturableDatasetError(f"{label} must be an object")
    return value


def _audit_trace(path: Path) -> tuple[Mapping[str, Any], FileIdentity]:
    before = _strict_file_identity(path, "confirmation trace")
    if before.lines != len(EXPECTED_FIXED_ASSIGNMENTS):
        raise CapturableDatasetError(
            "confirmation trace must contain exactly 625 records"
        )
    pairs: Counter[tuple[str, str]] = Counter()
    cells: Counter[tuple[str, str]] = Counter()
    total_plies = 0
    active_at_limit = 0
    terminal_games = 0
    result_kind_counts: Counter[str] = Counter()
    for expected, record in zip(
        EXPECTED_FIXED_ASSIGNMENTS,
        _trace_records(path),
        strict=True,
    ):
        if (
            set(record) != TRACE_TOP_LEVEL_KEYS
            or record.get("format") != TRACE_FORMAT
            or record.get("schemaVersion") != 2
            or record.get("authorityId") != TRACE_AUTHORITY
            or record.get("ruleset") != TRACE_RULESET
            or record.get("randomPolicy") != TRACE_RANDOM_POLICY
            or record.get("gameIndex") != expected.game_index
            or record.get("gameId") != expected.game_id
            or record.get("seed") != expected.gameplay_seed
            or record.get("parameterSeeds")
            != {
                "black": expected.black_parameter_seed,
                "white": expected.white_parameter_seed,
            }
            or record.get("plyLimit") != 60
            or record.get("hypothesisPolicy")
            != TRACE_HYPOTHESIS_POLICY
            or record.get("agents")
            != {"black": TRACE_AGENT, "white": TRACE_AGENT}
        ):
            raise CapturableDatasetError(
                f"confirmation trace schedule or policy mismatch at "
                f"game {expected.game_index}"
            )
        initial = _mapping(
            record.get("initialPosition"),
            "trace initialPosition",
        )
        initial_secrets = _mapping(
            _mapping(record.get("secrets"), "trace secrets").get("initial"),
            "trace initial secrets",
        )
        if (
            initial.get("fen") != STANDARD_INITIAL_FEN
            or initial.get("authorityId") != TRACE_AUTHORITY
            or initial.get("terminal") is not None
            or _mapping(
                initial_secrets.get("white"),
                "trace white secret",
            ).get("drawbackId")
            != expected.white_rule_id
            or _mapping(
                initial_secrets.get("black"),
                "trace black secret",
            ).get("drawbackId")
            != expected.black_rule_id
        ):
            raise CapturableDatasetError(
                f"confirmation trace initial assignment mismatch at "
                f"game {expected.game_index}"
            )
        plies = record.get("plies")
        if not isinstance(plies, list) or not 2 <= len(plies) <= 60:
            raise CapturableDatasetError(
                f"confirmation trace game {expected.game_index} "
                "does not retain both player trajectories"
            )
        colors: set[str] = set()
        for index, ply_value in enumerate(plies):
            ply = _mapping(ply_value, "trace ply")
            color = "white" if index % 2 == 0 else "black"
            if ply.get("ply") != index or ply.get("color") != color:
                raise CapturableDatasetError(
                    f"confirmation trace ply order mismatch at "
                    f"game {expected.game_index}"
                )
            colors.add(color)
        if colors != {"white", "black"}:
            raise CapturableDatasetError(
                f"confirmation trace game {expected.game_index} "
                "is missing one color"
            )
        result = _mapping(record.get("result"), "trace result")
        result_kind = result.get("kind")
        if result_kind not in TRACE_RESULT_KINDS:
            raise CapturableDatasetError(
                f"confirmation trace has unknown result kind at "
                f"game {expected.game_index}"
            )
        result_kind_counts[str(result_kind)] += 1
        stopped = record.get("stoppedAtPlyLimit")
        if result_kind == "active":
            if stopped is not True or len(plies) != 60:
                raise CapturableDatasetError(
                    "active trace is not censored at exactly 60 plies"
                )
            active_at_limit += 1
        else:
            if stopped is not False:
                raise CapturableDatasetError(
                    "terminal trace is marked as ply-limit censored"
                )
            terminal_games += 1
        total_plies += len(plies)
        pairs[(expected.white_rule_id, expected.black_rule_id)] += 1
        cells[("white", expected.white_rule_id)] += 1
        cells[("black", expected.black_rule_id)] += 1
    after = _strict_file_identity(path, "confirmation trace")
    if before != after:
        raise CapturableDatasetError(
            "confirmation trace changed during schedule verification"
        )
    if (
        len(pairs) != 625
        or set(pairs.values()) != {1}
        or len(cells) != 50
        or set(cells.values()) != {25}
    ):
        raise CapturableDatasetError(
            "confirmation trace pair or marginal coverage is invalid"
        )
    return (
        {
            "activeAtPlyLimit": active_at_limit,
            "authorityId": TRACE_AUTHORITY,
            "bytes": before.bytes,
            "countPerLabelColorCell": 25,
            "file": CONFIRMATION_TRACE_FILE,
            "firstGameIndex": 0,
            "format": TRACE_FORMAT,
            "games": 625,
            "labelColorCells": 50,
            "lastGameIndex": 624,
            "orderedPairs": 625,
            "plies": total_plies,
            "policyRegenerationMatch": True,
            "randomPolicy": TRACE_RANDOM_POLICY,
            "resultKindCounts": {
                kind: result_kind_counts[kind]
                for kind in TRACE_RESULT_KINDS
            },
            "ruleset": TRACE_RULESET,
            "schemaVersion": 2,
            "semanticReplayGames": 625,
            "semanticReplayPlies": total_plies,
            "sha256": before.sha256,
            "terminalGames": terminal_games,
        },
        before,
    )


def _sanitized_environment(
    source: Mapping[str, str] | None = None,
) -> Mapping[str, str]:
    environment = dict(os.environ if source is None else source)
    for variable in tuple(environment):
        normalized = variable.upper()
        if (
            normalized.startswith("GIT_")
            or normalized in {"NODE_OPTIONS", "NODE_PATH"}
            or any(
                normalized.startswith(prefix)
                for prefix in PACKAGE_MANAGER_ENVIRONMENT_PREFIXES
            )
        ):
            environment.pop(variable, None)
    environment["CI"] = "1"
    environment["NO_COLOR"] = "1"
    environment["COREPACK_ENABLE_NETWORK"] = "0"
    return environment


def _sha256_regular_file(path: Path, label: str) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            before = os.fstat(source.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise CapturableDatasetError(
                    f"{label} is not a regular file"
                )
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
            after = os.fstat(source.fileno())
    except OSError as error:
        raise CapturableDatasetError(
            f"cannot authenticate {label}"
        ) from error
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after:
        raise CapturableDatasetError(
            f"{label} changed while it was authenticated"
        )
    return digest.hexdigest()


def _hash_pnpm_runtime_tree(root: Path) -> tuple[str, int]:
    if not root.is_dir() or _is_link_or_junction(root):
        raise CapturableDatasetError(
            "pnpm runtime root must be a real directory"
        )
    resolved_root = root.resolve(strict=True)
    digest = hashlib.sha256()
    digest.update(PNPM_RUNTIME_TREE_HASH_ID.encode("ascii"))
    digest.update(b"\n")
    files: list[Path] = []
    for current_root, directories, names in os.walk(
        resolved_root,
        topdown=True,
        followlinks=False,
    ):
        current = Path(current_root)
        directories.sort()
        for name in directories:
            if _is_link_or_junction(current / name):
                raise CapturableDatasetError(
                    "pnpm runtime tree contains a link or junction"
                )
        for name in sorted(names):
            child = current / name
            if _is_link_or_junction(child) or not child.is_file():
                raise CapturableDatasetError(
                    "pnpm runtime tree contains a non-regular file"
                )
            files.append(child)
    for child in sorted(
        files,
        key=lambda path: path.relative_to(resolved_root).as_posix(),
    ):
        relative = child.relative_to(resolved_root).as_posix()
        child_sha256 = _sha256_regular_file(
            child,
            f"pnpm runtime file {relative}",
        )
        size = child.stat().st_size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(child_sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), len(files)


def _pnpm_runtime_candidates(
    source: Mapping[str, str],
) -> tuple[Path, ...]:
    cache_homes: list[Path] = []
    if source.get("COREPACK_HOME"):
        cache_homes.append(Path(source["COREPACK_HOME"]))
    if source.get("XDG_CACHE_HOME"):
        cache_homes.append(Path(source["XDG_CACHE_HOME"]) / "node" / "corepack")
    if source.get("LOCALAPPDATA"):
        cache_homes.append(Path(source["LOCALAPPDATA"]) / "node" / "corepack")
    home_value = source.get("USERPROFILE") or source.get("HOME")
    if home_value:
        home = Path(home_value)
        cache_homes.append(
            home
            / (
                Path("AppData") / "Local" / "node" / "corepack"
                if os.name == "nt"
                else Path(".cache") / "node" / "corepack"
            )
        )
    candidates: dict[Path, Path] = {}
    for cache_home in cache_homes:
        candidate = (
            cache_home
            / "v1"
            / "pnpm"
            / REQUIRED_PNPM_VERSION
        )
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        candidates.setdefault(resolved, candidate)
    return tuple(candidates.values())


def _resolve_pnpm_runtime(
    source: Mapping[str, str],
) -> tuple[Path, Path, str]:
    matches: list[tuple[Path, str]] = []
    for candidate in _pnpm_runtime_candidates(source):
        try:
            tree_sha256, file_count = _hash_pnpm_runtime_tree(candidate)
        except CapturableDatasetError:
            continue
        if (
            tree_sha256 == PNPM_RUNTIME_TREE_SHA256
            and file_count == PNPM_RUNTIME_TREE_FILES
        ):
            matches.append((candidate.resolve(strict=True), tree_sha256))
    if len(matches) != 1:
        raise CapturableDatasetError(
            "exact pinned pnpm runtime tree is unavailable or ambiguous"
        )
    root, tree_sha256 = matches[0]
    entrypoint = root / "bin" / "pnpm.mjs"
    if not entrypoint.is_file() or _is_link_or_junction(entrypoint):
        raise CapturableDatasetError(
            "pinned pnpm entrypoint is unavailable"
        )
    return root, entrypoint, tree_sha256


def _trusted_node_executable() -> Path:
    if os.name == "nt":
        candidate = (
            _windows_known_directory("program-files")
            / "nodejs"
            / "node.exe"
        )
    else:
        candidates = {
            path.resolve(strict=True)
            for path in (Path("/usr/bin/node"), Path("/usr/local/bin/node"))
            if path.is_file()
        }
        if len(candidates) != 1:
            raise CapturableDatasetError(
                "trusted system Node executable is unavailable"
            )
        candidate = candidates.pop()
    try:
        node = candidate.resolve(strict=True)
    except OSError as error:
        raise CapturableDatasetError(
            "trusted system Node executable is unavailable"
        ) from error
    if not node.is_file() or _is_link_or_junction(node):
        raise CapturableDatasetError(
            "trusted system Node executable is invalid"
        )
    return node


def _trusted_package_shell() -> Path:
    candidate = (
        _windows_known_directory("system") / "cmd.exe"
        if os.name == "nt"
        else Path("/bin/sh")
    )
    try:
        shell = candidate.resolve(strict=True)
    except OSError as error:
        raise CapturableDatasetError(
            "trusted package command shell is unavailable"
        ) from error
    if not shell.is_file() or _is_link_or_junction(shell):
        raise CapturableDatasetError(
            "trusted package command shell is invalid"
        )
    return shell


def _pnpm_shim_payload(
    node: Path,
    pnpm_entrypoint: Path,
    pnpm_store: Path,
) -> tuple[str, bytes]:
    unsafe = '%!"^&|<>\r\n'
    serialized_paths = (
        str(node),
        str(pnpm_entrypoint),
        str(pnpm_store),
    )
    if os.name == "nt":
        if any(
            any(token in value for token in unsafe)
            for value in serialized_paths
        ):
            raise CapturableDatasetError(
                "authenticated package runtime path is not shell-safe"
            )
        return (
            "pnpm.cmd",
            (
                "@echo off\r\n"
                f'"{node}" "{pnpm_entrypoint}" '
                f'--config.globalconfig="{os.devnull}" '
                f'--config.store-dir="{pnpm_store}" '
                f'--config.userconfig="{os.devnull}" '
                "--config.offline=true %*\r\n"
            ).encode("utf-8"),
        )
    return (
        "pnpm",
        (
            "#!/bin/sh\nexec "
            f"{shlex.quote(str(node))} "
            f"{shlex.quote(str(pnpm_entrypoint))} "
            f"--config.globalconfig={shlex.quote(os.devnull)} "
            f"--config.store-dir={shlex.quote(str(pnpm_store))} "
            f"--config.userconfig={shlex.quote(os.devnull)} "
            "--config.offline=true \"$@\"\n"
        ).encode("utf-8"),
    )


def _validate_environment_path(path: Path, label: str) -> None:
    serialized = str(path)
    unsafe = {"\0", "\r", "\n", os.pathsep}
    if os.name == "nt":
        unsafe.update('%!"^&|<>')
    if any(token in serialized for token in unsafe):
        raise CapturableDatasetError(
            f"{label} is not safe for the isolated environment"
        )


def _expected_package_environment(
    *,
    runtime_root: Path,
    node: Path,
    shell: Path,
) -> Mapping[str, str]:
    for path, label in (
        (runtime_root, "package runtime root"),
        (node.parent, "Node executable directory"),
        (shell.parent, "package shell directory"),
    ):
        _validate_environment_path(path, label)
    home = runtime_root / "home"
    bin_directory = runtime_root / "bin"
    cache = runtime_root / "cache"
    configuration = runtime_root / "config"
    data = runtime_root / "data"
    temporary = runtime_root / "tmp"
    environment = {
        "APPDATA": str(configuration),
        "CI": "1",
        "COREPACK_ENABLE_NETWORK": "0",
        "HOME": str(home),
        "LOCALAPPDATA": str(cache),
        "NPM_CONFIG_GLOBALCONFIG": os.devnull,
        "NPM_CONFIG_USERCONFIG": os.devnull,
        "TEMP": str(temporary),
        "TMP": str(temporary),
        "TMPDIR": str(temporary),
        "USERPROFILE": str(home),
        "XDG_CACHE_HOME": str(cache),
        "XDG_CONFIG_HOME": str(configuration),
        "XDG_DATA_HOME": str(data),
    }
    if os.name == "nt":
        windows_directory = _windows_known_directory("windows")
        system_directory = _windows_known_directory("system")
        if shell != (system_directory / "cmd.exe").resolve(strict=True):
            raise CapturableDatasetError(
                "trusted package command shell identity changed"
            )
        environment.update(
            {
                "ComSpec": str(shell),
                "NoDefaultCurrentDirectoryInExePath": "1",
                "PATHEXT": ".COM;.EXE;.BAT;.CMD",
                "PATH": os.pathsep.join(
                    str(path)
                    for path in (
                        bin_directory,
                        node.parent,
                        system_directory,
                        windows_directory,
                    )
                ),
                "SystemRoot": str(windows_directory),
                "WINDIR": str(windows_directory),
            }
        )
    else:
        environment.update(
            {
                "PATH": os.pathsep.join(
                    str(path)
                    for path in (
                        bin_directory,
                        node.parent,
                        Path("/usr/bin"),
                        Path("/bin"),
                    )
                ),
                "SHELL": str(shell),
            }
        )
    return environment


def _isolated_package_environment(
    *,
    private_root: Path,
    node: Path,
    pnpm_entrypoint: Path,
    pnpm_store: Path,
    shell: Path,
) -> tuple[Mapping[str, str], tempfile.TemporaryDirectory[str]]:
    if not private_root.is_dir() or _is_link_or_junction(private_root):
        raise CapturableDatasetError(
            "package runtime root must be a real private directory"
        )
    resolved_private_root = private_root.resolve(strict=True)
    _validate_environment_path(
        resolved_private_root,
        "package runtime root",
    )
    owner = tempfile.TemporaryDirectory(
        prefix=".fixed-package-environment-",
        dir=resolved_private_root,
    )
    runtime_root = Path(owner.name).resolve(strict=True)
    home = runtime_root / "home"
    bin_directory = runtime_root / "bin"
    cache = runtime_root / "cache"
    configuration = runtime_root / "config"
    data = runtime_root / "data"
    temporary = runtime_root / "tmp"
    for directory in (
        home,
        bin_directory,
        cache,
        configuration,
        data,
        temporary,
    ):
        directory.mkdir()
    shim_name, shim_payload = _pnpm_shim_payload(
        node,
        pnpm_entrypoint,
        pnpm_store,
    )
    shim = bin_directory / shim_name
    shim.write_bytes(shim_payload)
    if os.name != "nt":
        shim.chmod(0o700)
    environment = _expected_package_environment(
        runtime_root=runtime_root,
        node=node,
        shell=shell,
    )
    return environment, owner


def _capture_output(
    arguments: Sequence[str],
    *,
    cwd: Path,
    operation: str,
    environment: Mapping[str, str] | None = None,
) -> str:
    try:
        completed = subprocess.run(
            list(arguments),
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            env=(
                dict(environment)
                if environment is not None
                else _sanitized_environment()
            ),
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise CapturableDatasetError(
            f"{operation} could not run"
        ) from error
    if completed.returncode != 0:
        raise CapturableDatasetError(f"{operation} failed")
    return completed.stdout.strip()


def _resolve_toolchain(
    private_root: Path,
    expected_artifact: Mapping[str, Any] | None = None,
    *,
    create_runtime: bool = True,
) -> ResolvedToolchain:
    source_environment = dict(os.environ)
    node = _trusted_node_executable()
    corepack_candidates = (
        node.parent / "node_modules" / "corepack" / "dist" / "corepack.js",
        node.parent.parent
        / "lib"
        / "node_modules"
        / "corepack"
        / "dist"
        / "corepack.js",
    )
    resolved_corepack = {
        candidate.resolve()
        for candidate in corepack_candidates
        if candidate.is_file()
    }
    if len(resolved_corepack) != 1:
        raise CapturableDatasetError(
            "system Node must have one adjacent Corepack JavaScript entrypoint"
        )
    corepack = resolved_corepack.pop()
    node_sha256 = _sha256_regular_file(node, "system Node executable")
    corepack_sha256 = _sha256_regular_file(
        corepack,
        "system Corepack JavaScript entrypoint",
    )
    shell = _trusted_package_shell()
    shell_sha256 = _sha256_regular_file(
        shell,
        "trusted package command shell",
    )
    _pnpm_root, pnpm_entrypoint, pnpm_tree_sha256 = (
        _resolve_pnpm_runtime(source_environment)
    )
    measured_tree_sha256, measured_tree_files = _hash_pnpm_runtime_tree(
        pnpm_entrypoint.parents[1]
    )
    if (
        measured_tree_sha256 != pnpm_tree_sha256
        or measured_tree_sha256 != PNPM_RUNTIME_TREE_SHA256
        or measured_tree_files != PNPM_RUNTIME_TREE_FILES
    ):
        raise CapturableDatasetError(
            "fixed pnpm runtime changed during authentication"
        )
    if expected_artifact is not None:
        try:
            expected_identity = (
                expected_artifact["node"]["sha256"],
                expected_artifact["node"]["version"],
                expected_artifact["corepack"]["sha256"],
                expected_artifact["corepack"]["version"],
                expected_artifact["pnpm"]["sha256"],
                expected_artifact["pnpm"]["files"],
                expected_artifact["pnpm"]["version"],
                expected_artifact["shell"]["sha256"],
            )
        except (KeyError, TypeError) as error:
            raise CapturableDatasetError(
                "expected corpus toolchain identity is invalid"
            ) from error
        if expected_identity != (
            node_sha256,
            REQUIRED_NODE_VERSION,
            corepack_sha256,
            REQUIRED_COREPACK_VERSION,
            pnpm_tree_sha256,
            PNPM_RUNTIME_TREE_FILES,
            REQUIRED_PNPM_VERSION,
            shell_sha256,
        ):
            raise CapturableDatasetError(
                "corpus audit toolchain changed before execution"
            )
    node_version = _capture_output(
        [str(node), "--version"],
        cwd=REPOSITORY_ROOT,
        operation="system Node version authentication",
    )
    corepack_version = _capture_output(
        [str(node), str(corepack), "--version"],
        cwd=REPOSITORY_ROOT,
        operation="system Corepack version authentication",
    )
    pnpm_version = _capture_output(
        [
            str(node),
            str(pnpm_entrypoint),
            f"--config.globalconfig={os.devnull}",
            f"--config.userconfig={os.devnull}",
            "--version",
        ],
        cwd=REPOSITORY_ROOT,
        operation="pinned pnpm version authentication",
    )
    if (
        node_version != REQUIRED_NODE_VERSION
        or corepack_version != REQUIRED_COREPACK_VERSION
        or pnpm_version != REQUIRED_PNPM_VERSION
    ):
        raise CapturableDatasetError(
            "fixed corpus requires Node v24.15.0, Corepack 0.34.6, "
            "and pnpm 11.9.0"
        )
    if (
        _sha256_regular_file(node, "system Node executable")
        != node_sha256
        or _sha256_regular_file(
            corepack,
            "system Corepack JavaScript entrypoint",
        )
        != corepack_sha256
    ):
        raise CapturableDatasetError(
            "fixed corpus toolchain changed during authentication"
        )
    pnpm_store_output = _capture_output(
        [
            str(node),
            str(pnpm_entrypoint),
            f"--config.globalconfig={os.devnull}",
            f"--config.userconfig={os.devnull}",
            "store",
            "path",
        ],
        cwd=REPOSITORY_ROOT,
        operation="pnpm offline store resolution",
    )
    try:
        pnpm_store = Path(pnpm_store_output).resolve(strict=True)
    except OSError as error:
        raise CapturableDatasetError(
            "pnpm offline store is unavailable"
        ) from error
    if (
        not pnpm_store.is_dir()
        or _is_link_or_junction(pnpm_store)
    ):
        raise CapturableDatasetError(
            "pnpm offline store must be a real directory"
        )
    if create_runtime:
        environment, runtime_owner = _isolated_package_environment(
            private_root=private_root,
            node=node,
            pnpm_entrypoint=pnpm_entrypoint,
            pnpm_store=pnpm_store,
            shell=shell,
        )
    else:
        environment = {}
        runtime_owner = None
    return ResolvedToolchain(
        node=node,
        corepack=corepack,
        pnpm_entrypoint=pnpm_entrypoint,
        pnpm_store=pnpm_store,
        shell=shell,
        environment=environment,
        artifact={
            "corepack": {
                "sha256": corepack_sha256,
                "version": corepack_version,
            },
            "node": {
                "sha256": node_sha256,
                "version": node_version,
            },
            "pnpm": {
                "files": PNPM_RUNTIME_TREE_FILES,
                "sha256": pnpm_tree_sha256,
                "version": pnpm_version,
            },
            "shell": {
                "sha256": shell_sha256,
            },
        },
        runtime_owner=runtime_owner,
    )


def _authenticate_active_toolchain(toolchain: ResolvedToolchain) -> None:
    if toolchain.runtime_owner is None:
        raise CapturableDatasetError(
            "active corpus toolchain has no isolated runtime"
        )
    try:
        runtime_root = Path(toolchain.runtime_owner.name).resolve(strict=True)
    except OSError as error:
        raise CapturableDatasetError(
            "active corpus toolchain runtime is unavailable"
        ) from error
    if not runtime_root.is_dir() or _is_link_or_junction(runtime_root):
        raise CapturableDatasetError(
            "active corpus toolchain runtime is invalid"
        )
    for directory in (
        runtime_root / "home",
        runtime_root / "bin",
        runtime_root / "cache",
        runtime_root / "config",
        runtime_root / "data",
        runtime_root / "tmp",
    ):
        if not directory.is_dir() or _is_link_or_junction(directory):
            raise CapturableDatasetError(
                "active corpus toolchain runtime directory changed"
            )
    if any(
        any((runtime_root / role).iterdir())
        for role in ("home", "config")
    ):
        raise CapturableDatasetError(
            "active corpus toolchain configuration directory changed"
        )
    for path, label in (
        (toolchain.node, "active system Node executable"),
        (toolchain.corepack, "active Corepack JavaScript entrypoint"),
        (toolchain.shell, "active package command shell"),
    ):
        if (
            not path.is_file()
            or _is_link_or_junction(path)
            or path.resolve(strict=True) != path
        ):
            raise CapturableDatasetError(f"{label} identity changed")
    if (
        not toolchain.pnpm_store.is_dir()
        or _is_link_or_junction(toolchain.pnpm_store)
        or toolchain.pnpm_store.resolve(strict=True) != toolchain.pnpm_store
    ):
        raise CapturableDatasetError(
            "active pnpm store identity changed"
        )
    pnpm_root = toolchain.pnpm_entrypoint.parents[1]
    if (
        toolchain.pnpm_entrypoint
        != pnpm_root / "bin" / "pnpm.mjs"
        or not toolchain.pnpm_entrypoint.is_file()
        or _is_link_or_junction(toolchain.pnpm_entrypoint)
    ):
        raise CapturableDatasetError(
            "active pnpm entrypoint identity changed"
        )
    pnpm_sha256, pnpm_files = _hash_pnpm_runtime_tree(pnpm_root)
    measured_artifact = {
        "corepack": {
            "sha256": _sha256_regular_file(
                toolchain.corepack,
                "active Corepack JavaScript entrypoint",
            ),
            "version": REQUIRED_COREPACK_VERSION,
        },
        "node": {
            "sha256": _sha256_regular_file(
                toolchain.node,
                "active system Node executable",
            ),
            "version": REQUIRED_NODE_VERSION,
        },
        "pnpm": {
            "files": pnpm_files,
            "sha256": pnpm_sha256,
            "version": REQUIRED_PNPM_VERSION,
        },
        "shell": {
            "sha256": _sha256_regular_file(
                toolchain.shell,
                "active package command shell",
            ),
        },
    }
    if (
        measured_artifact != toolchain.artifact
        or pnpm_files != PNPM_RUNTIME_TREE_FILES
        or pnpm_sha256 != PNPM_RUNTIME_TREE_SHA256
        or toolchain.shell != _trusted_package_shell()
        or toolchain.node != _trusted_node_executable()
    ):
        raise CapturableDatasetError(
            "active corpus toolchain bytes changed"
        )
    expected_environment = _expected_package_environment(
        runtime_root=runtime_root,
        node=toolchain.node,
        shell=toolchain.shell,
    )
    if dict(toolchain.environment) != expected_environment:
        raise CapturableDatasetError(
            "active corpus toolchain environment changed"
        )
    shim_name, shim_payload = _pnpm_shim_payload(
        toolchain.node,
        toolchain.pnpm_entrypoint,
        toolchain.pnpm_store,
    )
    bin_directory = runtime_root / "bin"
    try:
        bin_entries = tuple(bin_directory.iterdir())
    except OSError as error:
        raise CapturableDatasetError(
            "active pnpm shim directory is unavailable"
        ) from error
    if (
        {entry.name for entry in bin_entries} != {shim_name}
        or len(bin_entries) != 1
    ):
        raise CapturableDatasetError(
            "active pnpm shim directory changed"
        )
    shim = bin_directory / shim_name
    try:
        shim_identity = shim.stat(follow_symlinks=False)
        actual_shim = shim.read_bytes()
    except OSError as error:
        raise CapturableDatasetError(
            "active pnpm shim is unavailable"
        ) from error
    if (
        not stat.S_ISREG(shim_identity.st_mode)
        or shim_identity.st_nlink != 1
        or _is_link_or_junction(shim)
        or actual_shim != shim_payload
    ):
        raise CapturableDatasetError(
            "active pnpm shim identity changed"
        )
    node_version = _capture_output(
        [str(toolchain.node), "--version"],
        cwd=REPOSITORY_ROOT,
        operation="active system Node version authentication",
        environment=toolchain.environment,
    )
    corepack_version = _capture_output(
        [str(toolchain.node), str(toolchain.corepack), "--version"],
        cwd=REPOSITORY_ROOT,
        operation="active Corepack version authentication",
        environment=toolchain.environment,
    )
    pnpm_version = _capture_output(
        [
            str(toolchain.node),
            str(toolchain.pnpm_entrypoint),
            f"--config.globalconfig={os.devnull}",
            f"--config.userconfig={os.devnull}",
            "--version",
        ],
        cwd=REPOSITORY_ROOT,
        operation="active pnpm version authentication",
        environment=toolchain.environment,
    )
    store_output = _capture_output(
        [
            str(toolchain.node),
            str(toolchain.pnpm_entrypoint),
            f"--config.globalconfig={os.devnull}",
            f"--config.store-dir={toolchain.pnpm_store}",
            f"--config.userconfig={os.devnull}",
            "--config.offline=true",
            "store",
            "path",
        ],
        cwd=REPOSITORY_ROOT,
        operation="active pnpm store authentication",
        environment=toolchain.environment,
    )
    try:
        measured_store = Path(store_output).resolve(strict=True)
    except OSError as error:
        raise CapturableDatasetError(
            "active pnpm store identity changed"
        ) from error
    if (
        node_version != REQUIRED_NODE_VERSION
        or corepack_version != REQUIRED_COREPACK_VERSION
        or pnpm_version != REQUIRED_PNPM_VERSION
        or measured_store != toolchain.pnpm_store
    ):
        raise CapturableDatasetError(
            "active corpus toolchain behavior changed"
        )


def _run(
    arguments: Sequence[str],
    *,
    cwd: Path,
    operation: str,
    timeout: int,
    environment: Mapping[str, str] | None = None,
) -> None:
    try:
        completed = subprocess.run(
            list(arguments),
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            env=(
                dict(environment)
                if environment is not None
                else _sanitized_environment()
            ),
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise CapturableDatasetError(f"{operation} could not run") from error
    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()
        suffix = f": {detail[-1]}" if detail else ""
        raise CapturableDatasetError(f"{operation} failed{suffix}")


_LOCAL_REPOSITORY_REDIRECTION_PATTERN = (
    r"^(include\.path|includeif\..*\.path|"
    r"url\..*\.(insteadof|pushinsteadof)|"
    r"http\..*|credential\..*|filter\..*|"
    r"core\.(attributesfile|excludesfile|fsmonitor|sshcommand)|"
    r"remote\..*\.(proxy|proxycommand|receivepack|uploadpack))$"
)


def _git_raw_blob_id(
    path: Path,
    algorithm: str,
) -> str:
    try:
        payload = path.read_bytes()
    except (OSError, ValueError) as error:
        raise CapturableDatasetError(
            "cannot authenticate tracked source bytes"
        ) from error
    try:
        digest = hashlib.new(algorithm)
    except ValueError as error:
        raise CapturableDatasetError(
            "cannot authenticate tracked source bytes"
        ) from error
    digest.update(f"blob {len(payload)}\0".encode("ascii"))
    digest.update(payload)
    return digest.hexdigest()


def _tracked_blob_sha256(
    repository: Path,
    relative: str,
    label: str,
) -> str:
    relative_path = Path(relative)
    if (
        relative_path.is_absolute()
        or not relative_path.parts
        or any(part in {"", ".", ".."} for part in relative_path.parts)
        or "\\" in relative
        or relative_path.as_posix() != relative
    ):
        raise CapturableDatasetError(
            f"{label} tracked path is invalid"
        )
    try:
        repository_root = repository.resolve(strict=True)
        completed = _authenticated_git_runner(repository_root)(
            "cat-file",
            "blob",
            f"HEAD:{relative}",
        )
        payload = completed.stdout.encode("utf-8")
    except (
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        UnicodeError,
        ValueError,
    ) as error:
        raise CapturableDatasetError(
            f"{label} tracked blob is unavailable"
        ) from error
    return hashlib.sha256(payload).hexdigest()


def _validate_tracked_worktree(
    *,
    git: Any,
    root: Path,
    allowed_runtime_paths: Sequence[str],
    label: str,
) -> None:
    object_format = git(
        "rev-parse",
        "--show-object-format",
    ).stdout.strip()
    if object_format not in {"sha1", "sha256"}:
        raise CapturableDatasetError(
            f"{label} Git object format is unsupported"
        )
    tree_records = git(
        "ls-tree",
        "-r",
        "-z",
        "--full-tree",
        "HEAD",
    ).stdout.split("\0")
    tracked_paths: set[str] = set()
    for record in tree_records:
        if not record:
            continue
        try:
            header, relative = record.split("\t", 1)
            mode, object_type, object_id = header.split(" ", 2)
        except ValueError as error:
            raise CapturableDatasetError(
                f"{label} tracked-tree record is invalid"
            ) from error
        relative_path = Path(relative)
        if (
            relative_path.is_absolute()
            or not relative_path.parts
            or any(part in {"", ".", ".."} for part in relative_path.parts)
            or "\\" in relative
            or relative in tracked_paths
        ):
            raise CapturableDatasetError(
                f"{label} tracked-tree path is invalid"
            )
        tracked_paths.add(relative)
        if mode == "160000" and object_type == "commit":
            continue
        path = root.joinpath(*relative_path.parts)
        if (
            mode not in {"100644", "100755"}
            or object_type != "blob"
            or _is_link_or_junction(path)
            or not path.is_file()
        ):
            raise CapturableDatasetError(
                f"{label} tracked source bytes differ from HEAD"
            )
        raw_object_id = _git_raw_blob_id(path, object_format)
        if object_id != raw_object_id:
            filtered_object_id = git(
                "hash-object",
                f"--path={relative}",
                "--",
                relative,
            ).stdout.strip()
            if object_id != filtered_object_id:
                raise CapturableDatasetError(
                    f"{label} tracked source bytes differ from HEAD"
                )
        if os.name != "nt":
            executable = bool(path.stat().st_mode & stat.S_IXUSR)
            if executable != (mode == "100755"):
                raise CapturableDatasetError(
                    f"{label} tracked source mode differs from HEAD"
                )
    index_records = git("ls-files", "-v", "-z").stdout.split("\0")
    if any(
        record
        and (
            len(record) < 3
            or record[1] != " "
            or record[0] != "H"
        )
        for record in index_records
    ):
        raise CapturableDatasetError(
            f"{label} index contains non-default tracking flags"
        )
    allowed = tuple(
        Path(value).as_posix().rstrip("/")
        for value in allowed_runtime_paths
    )
    for ignored in (False, True):
        arguments = [
            "ls-files",
            "--others",
            "--directory",
            "-z",
        ]
        if ignored:
            arguments.extend(("--ignored", "--exclude-standard"))
        else:
            arguments.append("--exclude-standard")
        for value in git(*arguments).stdout.split("\0"):
            relative = value.rstrip("/")
            if not relative:
                continue
            if not any(
                relative == prefix or relative.startswith(f"{prefix}/")
                for prefix in allowed
            ):
                raise CapturableDatasetError(
                    f"{label} contains an unauthenticated runtime path"
                )


def _authenticate_git_repository(
    root: Path,
    *,
    expected_commit: str,
    require_detached: bool,
    label: str,
    allowed_runtime_paths: Sequence[str],
) -> None:
    resolved_root = root.resolve(strict=True)
    git = _authenticated_git_runner(resolved_root)
    try:
        top_level = Path(
            git("rev-parse", "--show-toplevel").stdout.strip()
        ).resolve(strict=True)
        revision = git("rev-parse", "HEAD").stdout.strip()
        branch = git(
            "rev-parse",
            "--abbrev-ref",
            "HEAD",
        ).stdout.strip()
        status = git(
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignore-submodules=none",
        ).stdout
        redirections = git(
            "config",
            "--local",
            "--no-includes",
            "--name-only",
            "--get-regexp",
            _LOCAL_REPOSITORY_REDIRECTION_PATTERN,
            check=False,
        )
        if redirections.returncode not in (0, 1):
            raise subprocess.CalledProcessError(
                redirections.returncode,
                redirections.args,
                output=redirections.stdout,
                stderr=redirections.stderr,
            )
        _validate_tracked_worktree(
            git=git,
            root=resolved_root,
            allowed_runtime_paths=allowed_runtime_paths,
            label=label,
        )
    except CapturableDatasetError:
        raise
    except (
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        UnicodeError,
        ValueError,
    ) as error:
        raise CapturableDatasetError(
            f"cannot authenticate Git repository {label}"
        ) from error
    if (
        top_level != resolved_root
        or revision != expected_commit
        or (require_detached and branch != "HEAD")
        or status
        or redirections.stdout
    ):
        raise CapturableDatasetError(
            f"{label} Git repository identity is invalid"
        )


def authenticate_corpus_environment(
    root: Path,
    execution: Mapping[str, Any],
    expected_toolchain: Mapping[str, Any] | None = None,
    *,
    create_runtime: bool = False,
) -> tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    ResolvedToolchain,
]:
    measured_execution = _authenticated_execution_identity(
        protocol_commit=FIXED_PROTOCOL_COMMIT,
        protocol_file=FIXED_PROTOCOL_FILE,
        protocol_sha256=FIXED_PROTOCOL_SHA256,
        operation="fixed corpus environment authentication",
    )
    if measured_execution != execution:
        raise CapturableDatasetError(
            "corpus audit execution revision changed"
        )
    revision = execution.get("revision")
    if not isinstance(revision, str):
        raise CapturableDatasetError(
            "corpus audit execution revision is invalid"
        )
    _authenticate_git_repository(
        REPOSITORY_ROOT,
        expected_commit=revision,
        require_detached=False,
        label="Guesser source worktree",
        allowed_runtime_paths=GUESSER_ALLOWED_RUNTIME_PATHS,
    )
    generator = root / GENERATOR_DIRECTORY
    if (
        not generator.is_dir()
        or _is_link_or_junction(generator)
    ):
        raise CapturableDatasetError(
            "fixed generator worktree is not detached and clean at 74eb"
        )
    _authenticate_git_repository(
        generator,
        expected_commit=GENERATOR_ENGINE_COMMIT,
        require_detached=True,
        label="fixed generator worktree",
        allowed_runtime_paths=ENGINE_SCRUB_PATHS,
    )
    lock_path = generator / "pnpm-lock.yaml"
    if (
        not lock_path.is_file()
        or _is_link_or_junction(lock_path)
        or _tracked_blob_sha256(
            generator,
            "pnpm-lock.yaml",
            "fixed generator lockfile",
        )
        != GENERATOR_LOCK_SHA256
    ):
        raise CapturableDatasetError(
            "fixed generator lockfile identity is invalid"
        )
    engine_submodule = REPOSITORY_ROOT / "engine"
    if not engine_submodule.is_dir() or _is_link_or_junction(engine_submodule):
        raise CapturableDatasetError(
            "Guesser Engine submodule identity is invalid"
        )
    _authenticate_git_repository(
        engine_submodule,
        expected_commit=ENGINE_SUBMODULE_COMMIT,
        require_detached=False,
        label="Guesser Engine submodule",
        allowed_runtime_paths=GUESSER_ENGINE_ALLOWED_RUNTIME_PATHS,
    )
    toolchain = _resolve_toolchain(
        root,
        expected_toolchain,
        create_runtime=create_runtime,
    )
    if (
        expected_toolchain is not None
        and toolchain.artifact != expected_toolchain
    ):
        if toolchain.runtime_owner is not None:
            toolchain.runtime_owner.cleanup()
        raise CapturableDatasetError(
            "corpus audit toolchain changed"
        )
    return (
        {
            "clean": True,
            "commit": GENERATOR_ENGINE_COMMIT,
            "lockfileSha256": GENERATOR_LOCK_SHA256,
            "repository": "DrawbackEngine",
        },
        {"commit": ENGINE_SUBMODULE_COMMIT},
        toolchain,
    )


def _distinct_regular_files_equal(
    left: Path,
    right: Path,
    label: str,
) -> bool:
    if (
        _is_link_or_junction(left)
        or _is_link_or_junction(right)
        or not left.is_file()
        or not right.is_file()
    ):
        raise CapturableDatasetError(
            f"{label} must compare two real regular files"
        )
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    left_descriptor = -1
    right_descriptor = -1
    try:
        left_descriptor = os.open(left, flags)
        right_descriptor = os.open(right, flags)
        left_stat = os.fstat(left_descriptor)
        right_stat = os.fstat(right_descriptor)
        if (
            not stat.S_ISREG(left_stat.st_mode)
            or not stat.S_ISREG(right_stat.st_mode)
            or os.path.samestat(left_stat, right_stat)
        ):
            raise CapturableDatasetError(
                f"{label} output aliases its submitted input"
            )
        if left_stat.st_size != right_stat.st_size:
            return False
        with (
            os.fdopen(left_descriptor, "rb", closefd=True) as first,
            os.fdopen(
                right_descriptor,
                "rb",
                closefd=True,
            ) as second,
        ):
            left_descriptor = -1
            right_descriptor = -1
            while True:
                left_chunk = first.read(1024 * 1024)
                right_chunk = second.read(1024 * 1024)
                if left_chunk != right_chunk:
                    return False
                if not left_chunk:
                    return True
    except OSError as error:
        raise CapturableDatasetError(
            f"cannot authenticate {label} output"
        ) from error
    finally:
        if left_descriptor >= 0:
            os.close(left_descriptor)
        if right_descriptor >= 0:
            os.close(right_descriptor)


def _git_ignored(root: Path, relative: str) -> bool:
    directory_pattern = f"{relative.rstrip('/')}/"
    git = _authenticated_git_runner(root.resolve(strict=True))
    try:
        completed = git(
            "check-ignore",
            "--quiet",
            "--no-index",
            "--",
            directory_pattern,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired, UnicodeError) as error:
        raise CapturableDatasetError(
            f"cannot verify ignored runtime path {relative}"
        ) from error
    if completed.returncode not in (0, 1):
        raise CapturableDatasetError(
            f"cannot verify ignored runtime path {relative}"
        )
    return completed.returncode == 0


def _validated_runtime_target(root: Path, relative: str) -> Path:
    relative_path = Path(relative)
    if (
        relative_path.is_absolute()
        or not relative_path.parts
        or any(part in {"", ".", ".."} for part in relative_path.parts)
    ):
        raise CapturableDatasetError(
            f"unsafe runtime scrub target {relative}"
        )
    if not root.is_dir() or _is_link_or_junction(root):
        raise CapturableDatasetError(
            "runtime scrub root must be a real directory"
        )
    root_resolved = root.resolve(strict=True)
    target = root_resolved.joinpath(*relative_path.parts)
    if (
        target.absolute() == root_resolved.absolute()
        or not target.absolute().is_relative_to(root_resolved.absolute())
        or not _git_ignored(root_resolved, relative_path.as_posix())
    ):
        raise CapturableDatasetError(
            f"runtime scrub target is not an explicit ignored path: "
            f"{relative}"
        )
    current = target.parent
    while current.absolute() != root_resolved.absolute():
        if current.exists() and _is_link_or_junction(current):
            raise CapturableDatasetError(
                f"runtime scrub target traverses a link or junction: "
                f"{relative}"
            )
        parent = current.parent
        if parent == current or not parent.absolute().is_relative_to(
            root_resolved.absolute()
        ):
            raise CapturableDatasetError(
                f"runtime scrub target escapes repository: {relative}"
            )
        current = parent
    if _is_link_or_junction(target):
        raise CapturableDatasetError(
            f"runtime scrub target is a link or junction: {relative}"
        )
    return target


def _scrub_ignored_paths(
    root: Path,
    relatives: Sequence[str],
) -> None:
    def handle_remove_error(
        _function: Any,
        _path: str,
        error_info: tuple[type[BaseException], BaseException, Any],
    ) -> None:
        error = error_info[1]
        if isinstance(error, FileNotFoundError):
            return
        raise error

    for relative in relatives:
        target = _validated_runtime_target(root, relative)
        if not target.exists():
            continue
        if not target.is_dir():
            raise CapturableDatasetError(
                f"runtime scrub target is not a directory: {relative}"
            )
        removal_target = target
        if os.name == "nt":
            resolved = str(target.resolve(strict=True))
            removal_target = Path(
                (
                    f"\\\\?\\UNC\\{resolved[2:]}"
                    if resolved.startswith("\\\\")
                    else f"\\\\?\\{resolved}"
                )
            )
        try:
            shutil.rmtree(removal_target, onerror=handle_remove_error)
        except FileNotFoundError:
            pass
        except OSError as error:
            raise CapturableDatasetError(
                f"could not scrub ignored runtime path {relative}"
            ) from error
        if os.path.lexists(target):
            raise CapturableDatasetError(
                f"could not scrub ignored runtime path {relative}"
            )


def _hash_dist_tree(
    root: Path,
    relatives: Sequence[str],
    required_files: Sequence[str],
) -> str:
    root_resolved = root.resolve(strict=True)
    digest = hashlib.sha256()
    digest.update(BUILD_TREE_HASH_ID.encode("ascii"))
    digest.update(b"\n")
    seen_files: set[str] = set()
    for relative in relatives:
        directory = _validated_runtime_target(root_resolved, relative)
        if not directory.is_dir():
            raise CapturableDatasetError(
                f"rebuilt dist directory is missing: {relative}"
            )
        file_count = 0
        for current_root, directories, files in os.walk(
            directory,
            topdown=True,
            followlinks=False,
        ):
            current = Path(current_root)
            directories.sort()
            for name in tuple(directories):
                child = current / name
                if _is_link_or_junction(child):
                    raise CapturableDatasetError(
                        "rebuilt dist tree contains a link or junction"
                    )
            for name in sorted(files):
                child = current / name
                if _is_link_or_junction(child) or not child.is_file():
                    raise CapturableDatasetError(
                        "rebuilt dist tree contains a non-regular file"
                    )
                relative_file = child.relative_to(root_resolved).as_posix()
                if relative_file in seen_files:
                    raise CapturableDatasetError(
                        "rebuilt dist tree contains a duplicate file"
                    )
                seen_files.add(relative_file)
                child_sha256 = _sha256_regular_file(
                    child,
                    f"rebuilt dist file {relative_file}",
                )
                size = child.stat().st_size
                digest.update(relative_file.encode("utf-8"))
                digest.update(b"\0")
                digest.update(str(size).encode("ascii"))
                digest.update(b"\0")
                digest.update(child_sha256.encode("ascii"))
                digest.update(b"\n")
                file_count += 1
        if file_count == 0:
            raise CapturableDatasetError(
                f"rebuilt dist directory is empty: {relative}"
            )
    for relative in required_files:
        required = root_resolved.joinpath(*Path(relative).parts)
        if (
            not required.is_file()
            or _is_link_or_junction(required)
            or required.relative_to(root_resolved).as_posix()
            not in seen_files
        ):
            raise CapturableDatasetError(
                f"rebuilt distribution entrypoint is missing: {relative}"
            )
    return digest.hexdigest()


def _combine_build_tree_hashes(
    components: Sequence[tuple[str, str]],
) -> str:
    digest = hashlib.sha256()
    digest.update(BUILD_TREE_HASH_ID.encode("ascii"))
    digest.update(b"\n")
    for label, component_sha256 in components:
        if not _is_sha256(component_sha256):
            raise CapturableDatasetError(
                "build-tree component digest is invalid"
            )
        digest.update(label.encode("ascii"))
        digest.update(b"\0")
        digest.update(component_sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _authenticate_expected_environment(
    root: Path,
    execution: Mapping[str, Any],
    toolchain: ResolvedToolchain,
) -> None:
    _generator, _engine_submodule, measured = authenticate_corpus_environment(
        root,
        execution,
        toolchain.artifact,
    )
    if (
        measured.node != toolchain.node
        or measured.corepack != toolchain.corepack
        or measured.pnpm_entrypoint != toolchain.pnpm_entrypoint
        or measured.pnpm_store != toolchain.pnpm_store
        or measured.shell != toolchain.shell
    ):
        raise CapturableDatasetError(
            "active corpus toolchain path changed"
        )
    _authenticate_active_toolchain(toolchain)


def _run_authenticated(
    arguments: Sequence[str],
    *,
    root: Path,
    execution: Mapping[str, Any],
    toolchain: ResolvedToolchain,
    cwd: Path,
    operation: str,
    timeout: int,
) -> None:
    _authenticate_expected_environment(root, execution, toolchain)
    _run(
        arguments,
        cwd=cwd,
        operation=operation,
        timeout=timeout,
        environment=toolchain.environment,
    )
    _authenticate_expected_environment(root, execution, toolchain)


def _reproduce_trace(
    root: Path,
    trace_path: Path,
    execution: Mapping[str, Any],
    toolchain: ResolvedToolchain,
) -> str:
    generator = root / GENERATOR_DIRECTORY
    _authenticate_expected_environment(root, execution, toolchain)
    _scrub_ignored_paths(generator, ENGINE_SCRUB_PATHS)
    _authenticate_expected_environment(root, execution, toolchain)
    _run_authenticated(
        toolchain.pnpm(
            "install",
            "--frozen-lockfile",
            "--ignore-scripts",
            "--offline",
        ),
        root=root,
        execution=execution,
        toolchain=toolchain,
        cwd=generator,
        operation="frozen offline Engine dependency installation",
        timeout=60 * 60,
    )
    _run_authenticated(
        toolchain.pnpm("build"),
        root=root,
        execution=execution,
        toolchain=toolchain,
        cwd=generator,
        operation="full frozen Engine build",
        timeout=60 * 60,
    )
    build_sha256 = _hash_dist_tree(
        generator,
        ENGINE_DIST_PATHS,
        ENGINE_REQUIRED_DIST_FILES,
    )
    with tempfile.TemporaryDirectory(
        prefix=".fixed-confirmation-regeneration-",
        dir=root,
    ) as directory:
        regenerated = Path(directory) / CONFIRMATION_TRACE_FILE
        _run_authenticated(
            [
                str(toolchain.node),
                str(
                    generator
                    / "apps"
                    / "engine-cli"
                    / "dist"
                    / "player-private-batch-cli.js"
                ),
                "test",
                "0",
                "0",
                "625",
                "15",
                "633442320",
                "633446417",
                "633450514",
                str(regenerated),
                "60",
                "30",
                "1",
                "5000",
                "35",
                "standard",
            ],
            root=root,
            execution=execution,
            toolchain=toolchain,
            cwd=generator,
            operation="frozen Engine trace regeneration",
            timeout=4 * 60 * 60,
        )
        if not _distinct_regular_files_equal(
            trace_path,
            regenerated,
            "frozen Engine trace regeneration",
        ):
            raise CapturableDatasetError(
                "submitted trace is not byte-identical to frozen Engine output"
            )
    if (
        _hash_dist_tree(
            generator,
            ENGINE_DIST_PATHS,
            ENGINE_REQUIRED_DIST_FILES,
        )
        != build_sha256
    ):
        raise CapturableDatasetError(
            "frozen Engine build changed during trace regeneration"
        )
    return build_sha256


def _verify_conversion(
    root: Path,
    trace_path: Path,
    test_path: Path,
    execution: Mapping[str, Any],
    toolchain: ResolvedToolchain,
) -> tuple[FileIdentity, str]:
    before = _strict_file_identity(test_path, "confirmation dataset")
    engine_submodule = REPOSITORY_ROOT / "engine"
    _authenticate_expected_environment(root, execution, toolchain)
    _scrub_ignored_paths(REPOSITORY_ROOT, GUESSER_SCRUB_PATHS)
    _scrub_ignored_paths(
        engine_submodule,
        GUESSER_ENGINE_SCRUB_PATHS,
    )
    _authenticate_expected_environment(root, execution, toolchain)
    _run_authenticated(
        toolchain.pnpm(
            "install",
            "--frozen-lockfile",
            "--ignore-scripts",
            "--offline",
            "--filter",
            "@drawbackguesser/dataset-cli...",
        ),
        root=root,
        execution=execution,
        toolchain=toolchain,
        cwd=REPOSITORY_ROOT,
        operation="frozen offline converter dependency installation",
        timeout=60 * 60,
    )
    _run_authenticated(
        toolchain.pnpm(
            "--filter",
            "@drawbackguesser/dataset-cli...",
            "build",
        ),
        root=root,
        execution=execution,
        toolchain=toolchain,
        cwd=REPOSITORY_ROOT,
        operation="frozen dataset converter dependency build",
        timeout=60 * 60,
    )
    conversion_build_sha256 = _combine_build_tree_hashes(
        (
            (
                "DrawbackGuesser",
                _hash_dist_tree(
                    REPOSITORY_ROOT,
                    GUESSER_DIST_PATHS,
                    GUESSER_REQUIRED_DIST_FILES,
                ),
            ),
            (
                "DrawbackEngine-submodule",
                _hash_dist_tree(
                    engine_submodule,
                    GUESSER_ENGINE_DIST_PATHS,
                    GUESSER_ENGINE_REQUIRED_DIST_FILES,
                ),
            ),
        )
    )
    with tempfile.TemporaryDirectory(
        prefix=".fixed-confirmation-conversion-",
        dir=root,
    ) as directory:
        converted = Path(directory) / CONFIRMATION_TEST_FILE
        _run_authenticated(
            [
                str(toolchain.node),
                str(
                    REPOSITORY_ROOT
                    / "apps"
                    / "dataset-cli"
                    / "dist"
                    / "cli.js"
                ),
                "--input",
                str(trace_path),
                "--output",
                str(converted),
                "--require-authority",
                TRACE_AUTHORITY,
                "--require-evaluator",
                "none",
            ],
            root=root,
            execution=execution,
            toolchain=toolchain,
            cwd=REPOSITORY_ROOT,
            operation="semantic replay and deterministic conversion",
            timeout=60 * 60,
        )
        if not _distinct_regular_files_equal(
            test_path,
            converted,
            "deterministic trace conversion",
        ):
            raise CapturableDatasetError(
                "confirmation dataset is not the exact trace conversion"
            )
    after = _strict_file_identity(test_path, "confirmation dataset")
    if before != after:
        raise CapturableDatasetError(
            "confirmation dataset changed during conversion verification"
        )
    if (
        _combine_build_tree_hashes(
            (
                (
                    "DrawbackGuesser",
                    _hash_dist_tree(
                        REPOSITORY_ROOT,
                        GUESSER_DIST_PATHS,
                        GUESSER_REQUIRED_DIST_FILES,
                    ),
                ),
                (
                    "DrawbackEngine-submodule",
                    _hash_dist_tree(
                        engine_submodule,
                        GUESSER_ENGINE_DIST_PATHS,
                        GUESSER_ENGINE_REQUIRED_DIST_FILES,
                    ),
                ),
            )
        )
        != conversion_build_sha256
    ):
        raise CapturableDatasetError(
            "dataset converter build changed during conversion"
        )
    return before, conversion_build_sha256


def _construct_corpus_verification(
    root: Path,
    execution: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> CorpusVerification:
    generator, engine_submodule, toolchain = (
        authenticate_corpus_environment(
            root,
            execution,
            create_runtime=True,
        )
    )
    toolchain_artifact = toolchain.artifact
    _authenticate_expected_environment(
        root,
        execution,
        toolchain,
    )
    trace_path = require_private_regular_file(
        root,
        root / CONFIRMATION_TRACE_FILE,
        CONFIRMATION_TRACE_FILE,
        "confirmation trace",
    )
    test_path = require_private_regular_file(
        root,
        root / CONFIRMATION_TEST_FILE,
        CONFIRMATION_TEST_FILE,
        "confirmation dataset",
    )
    trace_summary, trace_identity = _audit_trace(trace_path)
    generator_build_sha256 = _reproduce_trace(
        root,
        trace_path,
        execution,
        toolchain,
    )
    _authenticate_expected_environment(root, execution, toolchain)
    dataset_identity, conversion_build_sha256 = _verify_conversion(
        root,
        trace_path,
        test_path,
        execution,
        toolchain,
    )
    _authenticate_expected_environment(root, execution, toolchain)
    rows, loaded_identity = _load_pinned_capturable_dataset(test_path)
    test_sha256 = loaded_identity.sha256
    corpus = audit_confirmation_rows(
        rows,
        set(registry["gameIds"]),
    )
    if (
        test_sha256 != dataset_identity.sha256
        or loaded_identity != dataset_identity
        or dataset_identity.lines != len(rows)
        or trace_summary["plies"] != len(rows)
        or trace_identity != _strict_file_identity(
            trace_path,
            "confirmation trace",
        )
        or dataset_identity != _strict_file_identity(
            test_path,
            "confirmation dataset",
        )
    ):
        raise CapturableDatasetError(
            "confirmation corpus changed across complete verification"
        )
    _authenticate_expected_environment(root, execution, toolchain)
    artifact = {
        "audit": {
            "clean": True,
            "repository": "DrawbackGuesser",
            "revision": execution["revision"],
            "toolId": AUDIT_TOOL_ID,
        },
        "conversion": {
            "buildSha256": conversion_build_sha256,
            "byteExact": True,
            "engineSubmoduleCommit": ENGINE_SUBMODULE_COMMIT,
            "id": CONVERSION_ID,
            "inputTraceSha256": trace_identity.sha256,
        },
        "dataset": {
            "authorityId": TRACE_AUTHORITY,
            "bytes": dataset_identity.bytes,
            "file": CONFIRMATION_TEST_FILE,
            "games": corpus["games"],
            "rows": len(rows),
            "schemaVersion": 8,
            "sha256": dataset_identity.sha256,
            "trueHypothesisSurvivalRows": len(rows),
            "twoColorGames": corpus["games"],
        },
        "engineSubmodule": engine_submodule,
        "format": CORPUS_RECEIPT_FORMAT,
        "generator": {
            **generator,
            "buildSha256": generator_build_sha256,
        },
        "priorRegistry": {
            "file": PRIOR_REGISTRY_FILE,
            "games": PRIOR_REGISTRY_GAME_COUNT,
            "overlap": 0,
            "sha256": PRIOR_REGISTRY_SHA256,
            "sources": PRIOR_REGISTRY_SOURCE_COUNT,
        },
        "protocol": {
            "commit": FIXED_PROTOCOL_COMMIT,
            "file": FIXED_PROTOCOL_FILE,
            "sha256": FIXED_PROTOCOL_SHA256,
        },
        "schedule": GENERATION_SCHEDULE,
        "trace": trace_summary,
        "toolchain": toolchain_artifact,
        "version": CORPUS_RECEIPT_VERSION,
    }
    verification = CorpusVerification(
        artifact=artifact,
        rows=tuple(rows),
        test_sha256=test_sha256,
        corpus=corpus,
    )
    if toolchain.runtime_owner is not None:
        toolchain.runtime_owner.cleanup()
    return verification


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(token in "0123456789abcdef" for token in value)
    )


def _exact_wire_value(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, Mapping):
        return (
            isinstance(actual, Mapping)
            and set(actual) == set(expected)
            and all(
                _exact_wire_value(actual[key], expected[key])
                for key in expected
            )
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                _exact_wire_value(left, right)
                for left, right in zip(actual, expected, strict=True)
            )
        )
    return actual == expected


def _validate_receipt(
    artifact: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> None:
    if (
        set(artifact)
        != {
            "audit",
            "conversion",
            "dataset",
            "engineSubmodule",
            "format",
            "generator",
            "priorRegistry",
            "protocol",
            "schedule",
            "toolchain",
            "trace",
            "version",
        }
        or artifact.get("format") != CORPUS_RECEIPT_FORMAT
        or type(artifact.get("version")) is not int
        or artifact.get("version") != CORPUS_RECEIPT_VERSION
        or not _exact_wire_value(
            artifact.get("protocol"),
            {
            "commit": FIXED_PROTOCOL_COMMIT,
            "file": FIXED_PROTOCOL_FILE,
            "sha256": FIXED_PROTOCOL_SHA256,
            },
        )
        or not _exact_wire_value(
            artifact.get("audit"),
            {
            "clean": True,
            "repository": "DrawbackGuesser",
            "revision": execution.get("revision"),
            "toolId": AUDIT_TOOL_ID,
            },
        )
        or not _exact_wire_value(
            artifact.get("engineSubmodule"),
            {"commit": ENGINE_SUBMODULE_COMMIT},
        )
        or not _exact_wire_value(
            artifact.get("schedule"),
            GENERATION_SCHEDULE,
        )
        or not _exact_wire_value(
            artifact.get("priorRegistry"),
            {
            "file": PRIOR_REGISTRY_FILE,
            "games": PRIOR_REGISTRY_GAME_COUNT,
            "overlap": 0,
            "sha256": PRIOR_REGISTRY_SHA256,
            "sources": PRIOR_REGISTRY_SOURCE_COUNT,
            },
        )
    ):
        raise CapturableDatasetError(
            "fixed corpus receipt identity is invalid"
        )
    trace = _mapping(artifact.get("trace"), "receipt trace")
    generator = _mapping(
        artifact.get("generator"),
        "receipt generator",
    )
    toolchain = _mapping(
        artifact.get("toolchain"),
        "receipt toolchain",
    )
    node_toolchain = _mapping(
        toolchain.get("node"),
        "receipt Node toolchain",
    )
    corepack_toolchain = _mapping(
        toolchain.get("corepack"),
        "receipt Corepack toolchain",
    )
    pnpm_toolchain = _mapping(
        toolchain.get("pnpm"),
        "receipt pnpm toolchain",
    )
    shell_toolchain = _mapping(
        toolchain.get("shell"),
        "receipt shell toolchain",
    )
    if (
        set(generator)
        != {
            "buildSha256",
            "clean",
            "commit",
            "lockfileSha256",
            "repository",
        }
        or generator.get("clean") is not True
        or generator.get("commit") != GENERATOR_ENGINE_COMMIT
        or generator.get("lockfileSha256") != GENERATOR_LOCK_SHA256
        or generator.get("repository") != "DrawbackEngine"
        or not _is_sha256(generator.get("buildSha256"))
        or set(toolchain) != {"corepack", "node", "pnpm", "shell"}
        or set(node_toolchain) != {"sha256", "version"}
        or node_toolchain.get("version") != REQUIRED_NODE_VERSION
        or not _is_sha256(node_toolchain.get("sha256"))
        or set(corepack_toolchain) != {"sha256", "version"}
        or corepack_toolchain.get("version") != REQUIRED_COREPACK_VERSION
        or not _is_sha256(corepack_toolchain.get("sha256"))
        or not _exact_wire_value(
            pnpm_toolchain,
            {
                "files": PNPM_RUNTIME_TREE_FILES,
                "sha256": PNPM_RUNTIME_TREE_SHA256,
                "version": REQUIRED_PNPM_VERSION,
            },
        )
        or set(shell_toolchain) != {"sha256"}
        or not _is_sha256(shell_toolchain.get("sha256"))
    ):
        raise CapturableDatasetError(
            "fixed corpus receipt toolchain or build identity is invalid"
        )
    result_kind_counts = _mapping(
        trace.get("resultKindCounts"),
        "receipt trace resultKindCounts",
    )
    dataset = _mapping(artifact.get("dataset"), "receipt dataset")
    conversion = _mapping(
        artifact.get("conversion"),
        "receipt conversion",
    )
    if (
        set(trace)
        != {
            "activeAtPlyLimit",
            "authorityId",
            "bytes",
            "countPerLabelColorCell",
            "file",
            "firstGameIndex",
            "format",
            "games",
            "labelColorCells",
            "lastGameIndex",
            "orderedPairs",
            "plies",
            "policyRegenerationMatch",
            "randomPolicy",
            "resultKindCounts",
            "ruleset",
            "schemaVersion",
            "semanticReplayGames",
            "semanticReplayPlies",
            "sha256",
            "terminalGames",
        }
        or trace.get("file") != CONFIRMATION_TRACE_FILE
        or trace.get("format") != TRACE_FORMAT
        or type(trace.get("schemaVersion")) is not int
        or trace.get("schemaVersion") != 2
        or trace.get("authorityId") != TRACE_AUTHORITY
        or not _exact_wire_value(
            trace.get("randomPolicy"),
            TRACE_RANDOM_POLICY,
        )
        or not _exact_wire_value(trace.get("ruleset"), TRACE_RULESET)
        or set(result_kind_counts) != set(TRACE_RESULT_KINDS)
        or any(
            type(result_kind_counts.get(kind)) is not int
            or not 0 <= result_kind_counts[kind] <= 625
            for kind in TRACE_RESULT_KINDS
        )
        or sum(result_kind_counts.values()) != 625
        or type(trace.get("games")) is not int
        or trace.get("games") != 625
        or type(trace.get("firstGameIndex")) is not int
        or trace.get("firstGameIndex") != 0
        or type(trace.get("lastGameIndex")) is not int
        or trace.get("lastGameIndex") != 624
        or type(trace.get("orderedPairs")) is not int
        or trace.get("orderedPairs") != 625
        or type(trace.get("labelColorCells")) is not int
        or trace.get("labelColorCells") != 50
        or type(trace.get("countPerLabelColorCell")) is not int
        or trace.get("countPerLabelColorCell") != 25
        or type(trace.get("semanticReplayGames")) is not int
        or trace.get("semanticReplayGames") != 625
        or trace.get("policyRegenerationMatch") is not True
        or not _is_sha256(trace.get("sha256"))
        or type(trace.get("bytes")) is not int
        or trace["bytes"] <= 0
        or type(trace.get("plies")) is not int
        or not 1_250 <= trace["plies"] <= 37_500
        or trace.get("semanticReplayPlies") != trace["plies"]
        or type(trace.get("semanticReplayPlies")) is not int
        or type(trace.get("activeAtPlyLimit")) is not int
        or not 0 <= trace["activeAtPlyLimit"] <= 625
        or type(trace.get("terminalGames")) is not int
        or not 0 <= trace["terminalGames"] <= 625
        or trace["activeAtPlyLimit"] + trace["terminalGames"] != 625
        or result_kind_counts["active"] != trace["activeAtPlyLimit"]
    ):
        raise CapturableDatasetError(
            "fixed corpus receipt trace evidence is invalid"
        )
    if (
        set(dataset)
        != {
            "authorityId",
            "bytes",
            "file",
            "games",
            "rows",
            "schemaVersion",
            "sha256",
            "trueHypothesisSurvivalRows",
            "twoColorGames",
        }
        or dataset.get("file") != CONFIRMATION_TEST_FILE
        or dataset.get("authorityId") != TRACE_AUTHORITY
        or type(dataset.get("schemaVersion")) is not int
        or dataset.get("schemaVersion") != 8
        or type(dataset.get("games")) is not int
        or dataset.get("games") != 625
        or type(dataset.get("twoColorGames")) is not int
        or dataset.get("twoColorGames") != 625
        or type(dataset.get("rows")) is not int
        or dataset.get("rows") != trace["plies"]
        or type(dataset.get("trueHypothesisSurvivalRows")) is not int
        or dataset.get("trueHypothesisSurvivalRows") != dataset["rows"]
        or type(dataset.get("bytes")) is not int
        or dataset["bytes"] <= 0
        or not _is_sha256(dataset.get("sha256"))
        or not _exact_wire_value(
            conversion,
            {
            "buildSha256": conversion.get("buildSha256"),
            "byteExact": True,
            "engineSubmoduleCommit": ENGINE_SUBMODULE_COMMIT,
            "id": CONVERSION_ID,
            "inputTraceSha256": trace["sha256"],
            },
        )
        or not _is_sha256(conversion.get("buildSha256"))
    ):
        raise CapturableDatasetError(
            "fixed corpus receipt conversion evidence is invalid"
        )


def load_fixed_corpus_receipt(
    path: Path,
    execution: Mapping[str, Any],
) -> tuple[Mapping[str, Any], str]:
    if path.name != CORPUS_RECEIPT_FILE or _is_link_or_junction(path):
        raise CapturableDatasetError(
            "fixed corpus receipt path is invalid"
        )
    artifact, sha256 = _selection_report(path)
    _validate_receipt(artifact, execution)
    return artifact, sha256


def verify_fixed_corpus_receipt(
    root: Path,
    receipt: Mapping[str, Any],
    execution: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> CorpusVerification:
    measured = _construct_corpus_verification(
        root,
        execution,
        registry,
    )
    if measured.artifact != receipt:
        raise CapturableDatasetError(
            "fixed corpus no longer matches its bound receipt"
        )
    return measured


def reauthenticate_fixed_corpus_files(
    root: Path,
    receipt: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> None:
    _generator, _engine_submodule, toolchain = (
        authenticate_corpus_environment(
            root,
            execution,
            _mapping(receipt.get("toolchain"), "receipt toolchain"),
        )
    )
    trace_path = require_private_regular_file(
        root,
        root / CONFIRMATION_TRACE_FILE,
        CONFIRMATION_TRACE_FILE,
        "confirmation trace",
    )
    test_path = require_private_regular_file(
        root,
        root / CONFIRMATION_TEST_FILE,
        CONFIRMATION_TEST_FILE,
        "confirmation dataset",
    )
    trace_identity = _strict_file_identity(
        trace_path,
        "confirmation trace",
    )
    dataset_identity = _strict_file_identity(
        test_path,
        "confirmation dataset",
    )
    generator_build_sha256 = _hash_dist_tree(
        root / GENERATOR_DIRECTORY,
        ENGINE_DIST_PATHS,
        ENGINE_REQUIRED_DIST_FILES,
    )
    engine_submodule = REPOSITORY_ROOT / "engine"
    conversion_build_sha256 = _combine_build_tree_hashes(
        (
            (
                "DrawbackGuesser",
                _hash_dist_tree(
                    REPOSITORY_ROOT,
                    GUESSER_DIST_PATHS,
                    GUESSER_REQUIRED_DIST_FILES,
                ),
            ),
            (
                "DrawbackEngine-submodule",
                _hash_dist_tree(
                    engine_submodule,
                    GUESSER_ENGINE_DIST_PATHS,
                    GUESSER_ENGINE_REQUIRED_DIST_FILES,
                ),
            ),
        )
    )
    if (
        trace_identity.sha256 != receipt["trace"]["sha256"]
        or trace_identity.bytes != receipt["trace"]["bytes"]
        or trace_identity.lines != receipt["trace"]["games"]
        or dataset_identity.sha256 != receipt["dataset"]["sha256"]
        or dataset_identity.bytes != receipt["dataset"]["bytes"]
        or dataset_identity.lines != receipt["dataset"]["rows"]
        or generator_build_sha256
        != receipt["generator"]["buildSha256"]
        or conversion_build_sha256
        != receipt["conversion"]["buildSha256"]
    ):
        raise CapturableDatasetError(
            "fixed corpus files or rebuilt outputs changed after sealed "
            "verification"
        )
    authenticate_corpus_environment(
        root,
        execution,
        toolchain.artifact,
    )


def prepare_fixed_corpus_receipt(
    output_directory: Path,
) -> Mapping[str, Any]:
    root = require_private_root(output_directory)
    receipt_path = root / CORPUS_RECEIPT_FILE
    if receipt_path.exists():
        raise FileExistsError("fixed corpus receipt already exists")
    registry_path = require_private_regular_file(
        root,
        root / PRIOR_REGISTRY_FILE,
        PRIOR_REGISTRY_FILE,
        "prior corpus registry",
    )
    registry, registry_sha256 = load_prior_corpus_registry(registry_path)
    if (
        registry_sha256 != PRIOR_REGISTRY_SHA256
        or registry["sourceCount"] != PRIOR_REGISTRY_SOURCE_COUNT
        or registry["uniqueGameCount"] != PRIOR_REGISTRY_GAME_COUNT
    ):
        raise CapturableDatasetError(
            "prior corpus registry does not match the frozen inventory"
        )
    execution = _authenticated_execution_identity(
        protocol_commit=FIXED_PROTOCOL_COMMIT,
        protocol_file=FIXED_PROTOCOL_FILE,
        protocol_sha256=FIXED_PROTOCOL_SHA256,
        operation="fixed corpus preparation",
    )
    verified = _construct_corpus_verification(
        root,
        execution,
        registry,
    )
    payload = _canonical_json(verified.artifact)
    publish_bytes_durable(receipt_path, payload)
    return {
        "datasetSha256": verified.test_sha256,
        "file": str(receipt_path),
        "receiptSha256": hashlib.sha256(payload).hexdigest(),
        "rows": len(verified.rows),
        "traceSha256": verified.artifact["trace"]["sha256"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "Reproduce and audit the frozen confirmation corpus without "
            "loading any model."
        ),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        required=True,
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    require_isolated_python_runtime()
    options = _parser().parse_args(arguments)
    print(
        json.dumps(
            prepare_fixed_corpus_receipt(options.output_directory),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
