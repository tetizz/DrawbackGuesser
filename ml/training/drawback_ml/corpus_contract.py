"""Strict, content-addressed contract for schema-6 evaluator corpora."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, BinaryIO, Iterator, Mapping

from .records import DatasetSchemaError, parse_dataset_row
from .semantic_replay import (
    SEMANTIC_REPLAY_PUBLIC_KEYS,
    SemanticReplayError,
    StreamingSemanticReplayVerifier,
)
from .evaluator_schedule_contract import (
    ExpectedEvaluatorSlot,
    expected_balanced_slots,
)
from .path_validation import is_portable_safe_basename, portable_basename_key
from .splits import Split, assign_split
from .symbolic_schema import SYMBOLIC_FEATURE_VERSION, SYMBOLIC_RULE_IDS


SHA256 = re.compile(r"^[0-9a-f]{64}$")
IDENTIFIER = re.compile(r"^[a-z0-9](?:[a-z0-9._:/+-]{0,127})$", re.IGNORECASE)
COMPOSITE_IDENTIFIER = re.compile(
    r"^[a-z0-9](?:[a-z0-9._/+-]{0,127})$", re.IGNORECASE
)
UCI_MOVE = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$")
FEN_PIECES = re.compile(r"^[prnbqkPRNBQK1-8]+$")
MAX_SAFE_INTEGER = 9_007_199_254_740_991
PREPARED_RULE_IDS_SHA256 = (
    "2176c14065f2dfde853d37017f16e6ff00165b8c7704be596c6aad85c428f304"
)
PREPARED_RULE_IDS = (
    "vegan",
    "true-gentleman",
    "trophy-wife",
    "lame-duck",
    "cess",
    "forward-march",
    "checkers",
    "pacman",
    "oddball",
    "even-keeled",
    "truant",
    "spice-of-life",
    *SYMBOLIC_RULE_IDS[12:],
)
SPLITS = ("train", "validation", "test")
SYMBOLIC_RULE_INDEX = {
    drawback_id: index
    for index, drawback_id in enumerate(SYMBOLIC_RULE_IDS)
}
EXPECTED_TOP_LEVEL_KEYS = frozenset(
    {
        "schemaVersion",
        "generator",
        "rootSeed",
        "seedPolicy",
        "splitFractions",
        "splitSalt",
        "workers",
        "maxPlies",
        "ruleIds",
        "ruleIdsSha256",
        "symbolicFeatureVersion",
        "symbolicRuleIds",
        "symbolicRuleIdsSha256",
        "agentIds",
        "evaluatorCoverage",
        "evaluatorRequestSchemaVersion",
        "evaluatorCacheSchemaVersion",
        "evaluatorPolicyId",
        "evaluatorPolicyVersion",
        "engineFingerprint",
        "engineBinarySha256",
        "evaluatorSearchLimit",
        "ruleAssignmentPolicy",
        "observationPolicy",
        "splitSizes",
        "totalGames",
        "totalRows",
        "splits",
    }
)
EXPECTED_SPLIT_KEYS = frozenset(
    {
        "file",
        "games",
        "rowBearingGames",
        "zeroPlyGames",
        "oneSidedGames",
        "rows",
        "seeds",
        "sha256",
        "bytes",
        "outcomesSha256",
        "outcomes",
        "coverage",
    }
)
EXPECTED_RELEASE_ROOT_KEYS = frozenset(
    {"releaseManifestVersion", "corpusRunId", "corpus", "splits"}
)
EXPECTED_PRIVATE_MANIFEST_KEYS = frozenset(
    {"manifestVersion", "corpusRunId", "split", "dataset"}
)
EXPECTED_ROOT_SPLIT_KEYS = frozenset(
    {"games", "rows", "datasetBytes", "datasetSha256", "privateManifestSha256"}
)
EXPECTED_PUBLIC_CORPUS_KEYS = EXPECTED_TOP_LEVEL_KEYS.difference(
    {"rootSeed", "splits"}
)
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
HARD_NEGATIVE_AGENT_IDS = (
    "random-legal",
    "greedy-material",
    "human-like-weak",
    "human-like-medium",
    "human-like-strong",
)
HARD_NEGATIVE_PROFILES: Mapping[
    str, tuple[str, tuple[str, str], str, int]
] = {
    "checkers-pacman": (
        "Forced-capture rules that differ in which captures remain legal.",
        ("checkers", "pacman"),
        "requested",
        20260911,
    ),
    "truant-spice-of-life": (
        "History restrictions over recently moved piece types.",
        ("truant", "spice-of-life"),
        "requested",
        20260912,
    ),
    "oddball-even-keeled": (
        "Move-number parity restrictions.",
        ("oddball", "even-keeled"),
        "requested",
        20260913,
    ),
    "quit-horsing-around-forward-march": (
        "Knight movement restriction versus pawn-forward pressure.",
        ("quit-horsing-around", "forward-march"),
        "measured-confusion",
        20260914,
    ),
    "horse-tranquilizer-conscientious-objectors": (
        "Knight-capture prohibition versus global capture prohibition.",
        ("horse-tranquilizer", "conscientious-objectors"),
        "measured-confusion",
        20260915,
    ),
    "gambler-truant": (
        "Per-turn hidden piece-type restriction versus move-history restriction.",
        ("gambler", "truant"),
        "measured-confusion",
        20260916,
    ),
}
EXPECTED_HARD_NEGATIVE_TOP_LEVEL_KEYS = EXPECTED_TOP_LEVEL_KEYS.union(
    {"hardNegativeProfile", "hardNegativeGeneration"}
)
EXPECTED_HARD_NEGATIVE_PROFILE_KEYS = frozenset(
    {"id", "description", "ruleIds", "evidence"}
)
EXPECTED_HARD_NEGATIVE_GENERATION_KEYS = frozenset(
    {
        "version",
        "sourceRevision",
        "runId",
        "corpusConfigSha256",
        "planSha256",
    }
)
EXPECTED_HARD_NEGATIVE_PLAN_KEYS = frozenset(
    {"schemaVersion", "runPlan", "sourceRevision", "metadata", "splitSeeds", "schedule"}
)
EXPECTED_HARD_NEGATIVE_RUN_PLAN_KEYS = frozenset(
    {
        "schemaVersion",
        "runId",
        "corpusConfigSha256",
        "scheduleSha256",
        "ruleIds",
        "agentIds",
        "shardSize",
        "totalGames",
        "shards",
    }
)
EXPECTED_HARD_NEGATIVE_SHARD_KEYS = frozenset(
    {
        "id",
        "split",
        "shardIndex",
        "splitShardIndex",
        "splitStart",
        "splitEnd",
        "globalStart",
        "globalEnd",
        "gameCount",
        "seedAssignmentSha256",
    }
)
EXPECTED_HARD_NEGATIVE_METADATA_KEYS = EXPECTED_TOP_LEVEL_KEYS.difference(
    {"workers", "totalRows", "ruleIdsSha256", "symbolicRuleIdsSha256", "splits"}
).union({"hardNegativeProfile"})


class CorpusContractError(ValueError):
    """Raised before training when corpus identity or coverage is invalid."""


@dataclass(frozen=True)
class AuditedCorpusSplit:
    manifest_path: Path
    manifest_sha256: str
    split: str
    dataset_path: Path
    dataset_sha256: str
    dataset_bytes: int
    rows: int
    games: int
    seeds: tuple[int, ...]
    observed_seeds: tuple[int, ...]
    game_assignments: tuple[tuple[str, str, str], ...]
    outcomes_sha256: str
    row_bearing_games: int
    zero_ply_games: int
    one_sided_games: int
    white_assigned_games: tuple[tuple[str, int], ...]
    black_assigned_games: tuple[tuple[str, int], ...]
    white_observed_rows: tuple[tuple[str, int], ...]
    black_observed_rows: tuple[tuple[str, int], ...]
    engine_fingerprint: str
    evaluator_policy_id: str
    evaluator_policy_version: int
    release_root_sha256: str | None = None
    corpus_run_id: str | None = None
    max_plies: int | None = None

    def provenance(self) -> dict[str, object]:
        value: dict[str, object] = {
            "manifest_path": self.manifest_path.name,
            "manifest_sha256": self.manifest_sha256,
            "split": self.split,
            "dataset_file": self.dataset_path.name,
            "dataset_sha256": self.dataset_sha256,
            "dataset_bytes": self.dataset_bytes,
            "rows": self.rows,
            "games": self.games,
            "seeds": list(self.seeds),
            "observed_seeds": list(self.observed_seeds),
            "outcomes_sha256": self.outcomes_sha256,
            "row_bearing_games": self.row_bearing_games,
            "zero_ply_games": self.zero_ply_games,
            "one_sided_games": self.one_sided_games,
            "white_assigned_games": dict(self.white_assigned_games),
            "black_assigned_games": dict(self.black_assigned_games),
            "white_observed_rows": dict(self.white_observed_rows),
            "black_observed_rows": dict(self.black_observed_rows),
            "engine_fingerprint": self.engine_fingerprint,
            "evaluator_policy_id": self.evaluator_policy_id,
            "evaluator_policy_version": self.evaluator_policy_version,
            "rule_ids": list(SYMBOLIC_RULE_IDS),
            "symbolic_feature_version": SYMBOLIC_FEATURE_VERSION,
        }
        if self.release_root_sha256 is not None:
            value["release_root_sha256"] = self.release_root_sha256
        if self.corpus_run_id is not None:
            value["corpus_run_id"] = self.corpus_run_id
        return value


@dataclass
class AuditedPrivateCorpusLease:
    """Caller-owned lifetime for one authenticated set of pinned file handles."""

    audited: AuditedCorpusSplit
    root: BinaryIO
    private_manifest: BinaryIO
    dataset: BinaryIO

    def verify_dataset_unchanged(self, *, chunk_size: int = 1024 * 1024) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.dataset.seek(0)
        digest = hashlib.sha256()
        byte_count = 0
        while True:
            chunk = self.dataset.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
        self.dataset.seek(0)
        if (
            digest.hexdigest() != self.audited.dataset_sha256
            or byte_count != self.audited.dataset_bytes
        ):
            raise CorpusContractError(
                "pinned dataset changed after authentication"
            )


@dataclass(frozen=True)
class AuditedHardNegativeTrainCorpus:
    manifest_path: Path
    manifest_sha256: str
    dataset_path: Path
    dataset_sha256: str
    dataset_bytes: int
    plan_path: Path
    plan_sha256: str
    profile_id: str
    rule_ids: tuple[str, str]
    root_seed: int
    source_revision: str
    run_id: str
    corpus_config_sha256: str
    rows: int
    games: int
    outcomes_sha256: str
    observed_seeds: tuple[int, ...]
    game_assignments: tuple[tuple[str, str, str], ...]
    row_bearing_games: int
    max_plies: int
    agent_ids: tuple[str, ...]
    engine_binary_sha256: str
    engine_fingerprint: str
    evaluator_policy_id: str
    evaluator_policy_version: int
    evaluator_nodes: int
    observation_policy: str

    def provenance(self) -> dict[str, object]:
        return {
            "manifest_path": self.manifest_path.name,
            "manifest_sha256": self.manifest_sha256,
            "dataset_file": self.dataset_path.name,
            "dataset_sha256": self.dataset_sha256,
            "dataset_bytes": self.dataset_bytes,
            "plan_path": self.plan_path.name,
            "plan_sha256": self.plan_sha256,
            "profile_id": self.profile_id,
            "rule_ids": list(self.rule_ids),
            "root_seed": self.root_seed,
            "source_revision": self.source_revision,
            "run_id": self.run_id,
            "corpus_config_sha256": self.corpus_config_sha256,
            "rows": self.rows,
            "games": self.games,
            "outcomes_sha256": self.outcomes_sha256,
            "observed_seeds": list(self.observed_seeds),
            "row_bearing_games": self.row_bearing_games,
            "max_plies": self.max_plies,
            "agent_ids": list(self.agent_ids),
            "engine_binary_sha256": self.engine_binary_sha256,
            "engine_fingerprint": self.engine_fingerprint,
            "evaluator_policy_id": self.evaluator_policy_id,
            "evaluator_policy_version": self.evaluator_policy_version,
            "evaluator_nodes": self.evaluator_nodes,
            "observation_policy": self.observation_policy,
            "symbolic_feature_version": SYMBOLIC_FEATURE_VERSION,
        }


@dataclass
class AuditedHardNegativeCorpusLease:
    """Pinned lifetime for one authenticated training-only supplement."""

    audited: AuditedHardNegativeTrainCorpus
    manifest: BinaryIO
    dataset: BinaryIO
    plan: BinaryIO

    def verify_unchanged(self, *, chunk_size: int = 1024 * 1024) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        for source, expected_digest, expected_bytes, label in (
            (
                self.manifest,
                self.audited.manifest_sha256,
                None,
                "manifest",
            ),
            (
                self.dataset,
                self.audited.dataset_sha256,
                self.audited.dataset_bytes,
                "dataset",
            ),
            (self.plan, self.audited.plan_sha256, None, "plan"),
        ):
            source.seek(0)
            digest = hashlib.sha256()
            byte_count = 0
            while True:
                chunk = source.read(chunk_size)
                if not chunk:
                    break
                digest.update(chunk)
                byte_count += len(chunk)
            source.seek(0)
            if digest.hexdigest() != expected_digest or (
                expected_bytes is not None and byte_count != expected_bytes
            ):
                raise CorpusContractError(
                    f"pinned hard-negative {label} changed after authentication"
                )


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CorpusContractError(f"{label} must be an object")
    return value


def _exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    actual = set(value)
    missing = expected.difference(actual)
    unknown = actual.difference(expected)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        if unknown:
            details.append("unknown " + ", ".join(sorted(unknown)))
        raise CorpusContractError(f"{label} fields are invalid: {'; '.join(details)}")


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CorpusContractError(f"{label} must be a positive integer")
    return value


def _positive_safe_int(value: object, label: str) -> int:
    result = _positive_int(value, label)
    if result > MAX_SAFE_INTEGER:
        raise CorpusContractError(f"{label} must be a positive safe integer")
    return result


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CorpusContractError(f"{label} must be a non-negative integer")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise CorpusContractError(f"{label} must be a non-empty string")
    return value


def _identifier(value: object, label: str) -> str:
    identifier = _string(value, label).strip()
    if IDENTIFIER.fullmatch(identifier) is None:
        raise CorpusContractError(
            f"{label} must be a non-empty canonical identifier"
        )
    return identifier


def _composite_identifier(value: object, label: str) -> str:
    identifier = _string(value, label).strip()
    if COMPOSITE_IDENTIFIER.fullmatch(identifier) is None:
        raise CorpusContractError(
            f"{label} must be a canonical identifier without a colon"
        )
    return identifier


def _digest(value: object, label: str) -> str:
    digest = _string(value, label)
    if SHA256.fullmatch(digest) is None:
        raise CorpusContractError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _ordered_strings(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise CorpusContractError(f"{label} must be an array of non-empty strings")
    result = tuple(value)
    if len(set(result)) != len(result):
        raise CorpusContractError(f"{label} must not contain duplicates")
    return result


def _seed_list(value: object, label: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise CorpusContractError(f"{label} must be an array")
    seeds = tuple(_nonnegative_int(item, label) for item in value)
    if len(set(seeds)) != len(seeds):
        raise CorpusContractError(f"{label} must not contain duplicates")
    return seeds


def _coverage(
    value: object,
    label: str,
) -> tuple[tuple[str, int], ...]:
    coverage = _mapping(value, label)
    if set(coverage) != set(SYMBOLIC_RULE_IDS):
        raise CorpusContractError(
            f"{label} keys must exactly match the canonical drawback catalog"
        )
    return tuple(
        (rule_id, _nonnegative_int(coverage[rule_id], f"{label}.{rule_id}"))
        for rule_id in SYMBOLIC_RULE_IDS
    )


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _session_result(value: object, label: str) -> Mapping[str, Any]:
    result = _mapping(value, label)
    kind = result.get("kind")
    if kind == "active":
        _exact_keys(result, frozenset({"kind"}), label)
    elif kind == "checkmate":
        _exact_keys(result, frozenset({"kind", "winner"}), label)
        if result["winner"] not in {"white", "black"}:
            raise CorpusContractError(f"{label}.winner must be white or black")
    elif kind == "draw":
        _exact_keys(result, frozenset({"kind", "reason"}), label)
        _string(result["reason"], f"{label}.reason")
    elif kind == "drawback-loss":
        _exact_keys(result, frozenset({"kind", "loss"}), label)
        loss = _mapping(result["loss"], f"{label}.loss")
        _exact_keys(
            loss,
            frozenset({"ruleId", "color", "reason"}),
            f"{label}.loss",
        )
        _string(loss["ruleId"], f"{label}.loss.ruleId")
        if loss["color"] not in {"white", "black"}:
            raise CorpusContractError(
                f"{label}.loss.color must be white or black"
            )
        _string(loss["reason"], f"{label}.loss.reason")
    else:
        raise CorpusContractError(f"{label}.kind is unsupported")
    return result


def _outcome_ledger(
    entry: Mapping[str, Any],
    *,
    split: str,
    seeds: tuple[int, ...],
    rows: int,
) -> tuple[Mapping[str, Any], ...]:
    value = entry["outcomes"]
    if not isinstance(value, list) or len(value) != len(seeds):
        raise CorpusContractError(
            f"splits.{split}.outcomes must cover every scheduled game"
        )
    outcomes: list[Mapping[str, Any]] = []
    ply_total = 0
    for index, candidate in enumerate(value):
        label = f"splits.{split}.outcomes[{index}]"
        outcome = _mapping(candidate, label)
        _exact_keys(
            outcome,
            frozenset(
                {
                    "seed",
                    "splitIndex",
                    "whiteRuleId",
                    "blackRuleId",
                    "whiteAgentId",
                    "blackAgentId",
                    "plyCount",
                    "finalFen",
                    "result",
                    "stoppedAtPlyLimit",
                }
            ),
            label,
        )
        if outcome["seed"] != seeds[index]:
            raise CorpusContractError(f"{label}.seed disagrees with schedule")
        if outcome["splitIndex"] != index:
            raise CorpusContractError(f"{label}.splitIndex is not canonical")
        for field in (
            "whiteRuleId",
            "blackRuleId",
            "whiteAgentId",
            "blackAgentId",
        ):
            _string(outcome[field], f"{label}.{field}")
        ply_count = _nonnegative_int(outcome["plyCount"], f"{label}.plyCount")
        if ply_total > MAX_SAFE_INTEGER - ply_count:
            raise CorpusContractError("outcome ply counts exceed safe integers")
        ply_total += ply_count
        final_fen = _string(outcome["finalFen"], f"{label}.finalFen")
        if _normalize_fen(final_fen) != final_fen:
            raise CorpusContractError(f"{label}.finalFen is not canonical")
        result = _session_result(outcome["result"], f"{label}.result")
        stopped = outcome["stoppedAtPlyLimit"]
        if not isinstance(stopped, bool):
            raise CorpusContractError(
                f"{label}.stoppedAtPlyLimit must be boolean"
            )
        if (result["kind"] == "active") != stopped:
            raise CorpusContractError(
                f"{label} active result and ply-limit flag disagree"
            )
        outcomes.append(outcome)
    if ply_total != rows:
        raise CorpusContractError(
            f"splits.{split}.outcomes do not account for every row"
        )
    outcomes_sha256 = hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()
    if _digest(
        entry["outcomesSha256"], f"splits.{split}.outcomesSha256"
    ) != outcomes_sha256:
        raise CorpusContractError(
            f"splits.{split}.outcomesSha256 does not match outcomes"
        )
    row_bearing = sum(outcome["plyCount"] > 0 for outcome in outcomes)
    zero_ply = sum(outcome["plyCount"] == 0 for outcome in outcomes)
    one_sided = sum(outcome["plyCount"] == 1 for outcome in outcomes)
    for field, expected in (
        ("rowBearingGames", row_bearing),
        ("zeroPlyGames", zero_ply),
        ("oneSidedGames", one_sided),
    ):
        if _nonnegative_int(entry[field], f"splits.{split}.{field}") != expected:
            raise CorpusContractError(
                f"splits.{split}.{field} disagrees with outcomes"
            )
    return tuple(outcomes)


def _normalize_fen(fen: str) -> str:
    fields = fen.strip().split()
    if len(fields) != 6:
        raise CorpusContractError("FEN must contain exactly six fields")
    board, turn, castling, en_passant, halfmove, fullmove = fields
    ranks = board.split("/")
    if len(ranks) != 8:
        raise CorpusContractError("FEN board must contain exactly eight ranks")
    white_kings = 0
    black_kings = 0
    for rank in ranks:
        if FEN_PIECES.fullmatch(rank) is None or re.search(r"[1-8]{2}", rank):
            raise CorpusContractError("FEN board contains an invalid rank")
        squares = 0
        for symbol in rank:
            if "1" <= symbol <= "8":
                squares += int(symbol)
            else:
                squares += 1
                white_kings += int(symbol == "K")
                black_kings += int(symbol == "k")
        if squares != 8:
            raise CorpusContractError(
                "each FEN rank must describe exactly eight squares"
            )
    if white_kings != 1 or black_kings != 1:
        raise CorpusContractError(
            "FEN must contain exactly one king of each color"
        )
    if turn not in {"w", "b"}:
        raise CorpusContractError("FEN active color must be w or b")
    if castling != "-":
        if (
            re.fullmatch(r"[KQkq]+", castling) is None
            or len(set(castling)) != len(castling)
        ):
            raise CorpusContractError("FEN castling rights are invalid")
        castling = "".join(right for right in "KQkq" if right in castling)
    if en_passant != "-" and re.fullmatch(r"[a-h][36]", en_passant) is None:
        raise CorpusContractError("FEN en-passant target is invalid")
    if re.fullmatch(r"[0-9]+", halfmove) is None:
        raise CorpusContractError(
            "FEN halfmove clock must be a non-negative integer"
        )
    if re.fullmatch(r"[0-9]+", fullmove) is None or int(fullmove) < 1:
        raise CorpusContractError(
            "FEN fullmove number must be a positive integer"
        )
    halfmove_number = int(halfmove)
    fullmove_number = int(fullmove)
    if (
        halfmove_number > MAX_SAFE_INTEGER
        or fullmove_number > MAX_SAFE_INTEGER
    ):
        raise CorpusContractError("FEN move counters must be safe integers")
    return " ".join(
        (
            board,
            turn,
            castling,
            en_passant,
            str(halfmove_number),
            str(fullmove_number),
        )
    )


def _evaluator_request_digest(
    *,
    fen: str,
    ordinary_moves: tuple[str, ...],
    policy_id: str,
    policy_version: int,
    engine_fingerprint: str,
    search_limit: Mapping[str, Any],
) -> str:
    parts = engine_fingerprint.split(":")
    if (
        len(parts) != 4
        or SHA256.fullmatch(parts[2]) is None
        or SHA256.fullmatch(parts[3]) is None
    ):
        raise CorpusContractError(
            "engineFingerprint must contain engine, version, binary SHA-256, "
            "and options SHA-256"
        )
    policy_id = _identifier(policy_id, "evaluatorPolicyId")
    policy_version = _positive_safe_int(policy_version, "evaluatorPolicyVersion")
    # The persisted public fingerprint is already the canonical output of the
    # TypeScript request layer: digest components are lowercase and the two
    # colon-delimited identifiers cannot themselves contain a colon.
    engine = _composite_identifier(parts[0], "engineFingerprint.engine")
    engine_version = _composite_identifier(
        parts[1], "engineFingerprint.version"
    )
    root_moves = sorted({move.strip().lower() for move in ordinary_moves})
    if any(UCI_MOVE.fullmatch(move) is None for move in root_moves):
        raise CorpusContractError("ordinaryLegalMoves contains invalid UCI")
    material = {
        "schemaVersion": 1,
        "policy": {"id": policy_id, "version": policy_version},
        "fingerprint": {
            "engine": engine,
            "version": engine_version,
            "optionsDigest": parts[3],
        },
        "fen": _normalize_fen(fen),
        "rootMoves": root_moves,
        "limit": {
            "kind": search_limit["kind"],
            "value": search_limit["value"],
        },
    }
    return hashlib.sha256(_canonical_json(material).encode("utf-8")).hexdigest()


def _load_manifest(path: Path) -> tuple[Mapping[str, Any], str]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise CorpusContractError(f"cannot read corpus manifest: {path}") from error
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as error:
        raise CorpusContractError("corpus manifest is not valid JSON") from error
    manifest = _mapping(value, "corpus manifest")
    _exact_keys(manifest, EXPECTED_TOP_LEVEL_KEYS, "corpus manifest")
    return manifest, hashlib.sha256(payload).hexdigest()


def _validate_corpus_metadata(
    manifest: Mapping[str, Any],
    *,
    require_complete_catalog: bool,
    require_root_seed: bool,
) -> None:
    if manifest["schemaVersion"] != 6:
        raise CorpusContractError("corpus schemaVersion must be 6")
    if manifest["generator"] != "@drawbacktrainer/simulation":
        raise CorpusContractError("corpus generator identity is invalid")
    if manifest["seedPolicy"] != "BLAKE2b-64(drawbacktrainer-v1:gameSeed)":
        raise CorpusContractError("corpus seed policy is invalid")
    if manifest["splitSalt"] != "drawbacktrainer-v1":
        raise CorpusContractError("corpus split salt is invalid")
    if require_root_seed:
        root_seed = _nonnegative_int(manifest["rootSeed"], "rootSeed")
        if root_seed > 0xFFFF_FFFF:
            raise CorpusContractError("rootSeed must be an unsigned 32-bit integer")
    _positive_int(manifest["workers"], "workers")
    max_plies = manifest["maxPlies"]
    if max_plies is not None:
        _positive_int(max_plies, "maxPlies")
    if not _ordered_strings(manifest["agentIds"], "agentIds"):
        raise CorpusContractError("agentIds cannot be empty")
    if manifest["symbolicFeatureVersion"] != SYMBOLIC_FEATURE_VERSION:
        raise CorpusContractError(
            f"symbolicFeatureVersion must be {SYMBOLIC_FEATURE_VERSION}"
        )
    canonical_ids = tuple(SYMBOLIC_RULE_IDS)
    rule_ids = _ordered_strings(manifest["ruleIds"], "ruleIds")
    if rule_ids != PREPARED_RULE_IDS:
        raise CorpusContractError(
            "ruleIds must equal the frozen prepared sampling order"
        )
    rule_ids_digest = hashlib.sha256(
        _canonical_json(list(rule_ids)).encode("utf-8")
    ).hexdigest()
    if (
        _digest(manifest["ruleIdsSha256"], "ruleIdsSha256")
        != rule_ids_digest
        or rule_ids_digest != PREPARED_RULE_IDS_SHA256
    ):
        raise CorpusContractError("ruleIdsSha256 does not match ruleIds")
    if _ordered_strings(manifest["symbolicRuleIds"], "symbolicRuleIds") != canonical_ids:
        raise CorpusContractError(
            "symbolicRuleIds must equal the canonical ordered catalog"
        )
    if _digest(
        manifest["symbolicRuleIdsSha256"], "symbolicRuleIdsSha256"
    ) != hashlib.sha256(
        _canonical_json(list(canonical_ids)).encode("utf-8")
    ).hexdigest():
        raise CorpusContractError(
            "symbolicRuleIdsSha256 does not match symbolicRuleIds"
        )
    if manifest["evaluatorCoverage"] != "uniform-required":
        raise CorpusContractError("uniform evaluator coverage is required")
    _identifier(manifest["evaluatorPolicyId"], "evaluatorPolicyId")
    _positive_safe_int(
        manifest["evaluatorPolicyVersion"], "evaluatorPolicyVersion"
    )
    if manifest["evaluatorRequestSchemaVersion"] != 1:
        raise CorpusContractError("evaluator request schema version must be 1")
    if manifest["evaluatorCacheSchemaVersion"] != 1:
        raise CorpusContractError("evaluator cache schema version must be 1")
    _digest(manifest["engineBinarySha256"], "engineBinarySha256")
    fingerprint = _string(manifest["engineFingerprint"], "engineFingerprint")
    fingerprint_parts = fingerprint.split(":")
    if (
        len(fingerprint_parts) != 4
        or fingerprint_parts[2] != manifest["engineBinarySha256"]
    ):
        raise CorpusContractError(
            "engineFingerprint does not contain engineBinarySha256"
        )
    fractions = _mapping(manifest["splitFractions"], "splitFractions")
    if fractions != {"train": 0.8, "validation": 0.1, "test": 0.1}:
        raise CorpusContractError("splitFractions must be the frozen 80/10/10 policy")
    search_limit = _mapping(
        manifest["evaluatorSearchLimit"], "evaluatorSearchLimit"
    )
    if search_limit != {"kind": "nodes", "value": 10_000}:
        raise CorpusContractError(
            "evaluatorSearchLimit must be exactly 10,000 nodes"
        )
    assignment_policy = manifest["ruleAssignmentPolicy"]
    if assignment_policy not in {
        "seed-random-v1",
        "balanced-symmetric-v1",
    }:
        raise CorpusContractError("ruleAssignmentPolicy is unsupported")
    if (
        require_complete_catalog
        and assignment_policy != "balanced-symmetric-v1"
    ):
        raise CorpusContractError(
            "complete-catalog training requires balanced-symmetric-v1"
        )
    if manifest["observationPolicy"] != "single-attempt-allow-partial-v1":
        raise CorpusContractError("corpus observationPolicy is unsupported")


def _validate_manifest_identity(
    manifest: Mapping[str, Any],
    *,
    require_complete_catalog: bool,
) -> None:
    _validate_corpus_metadata(
        manifest,
        require_complete_catalog=require_complete_catalog,
        require_root_seed=True,
    )
    split_sizes = _mapping(manifest["splitSizes"], "splitSizes")
    split_entries = _mapping(manifest["splits"], "splits")
    if set(split_sizes) != set(SPLITS) or set(split_entries) != set(SPLITS):
        raise CorpusContractError("manifest must declare exactly three splits")
    total_games = 0
    total_rows = 0
    seen_seeds: set[int] = set()
    seen_split_files: set[str] = set()
    for split in SPLITS:
        entry = _mapping(split_entries[split], f"splits.{split}")
        _exact_keys(entry, EXPECTED_SPLIT_KEYS, f"splits.{split}")
        file_name = entry["file"]
        if not is_portable_safe_basename(file_name):
            raise CorpusContractError(
                f"splits.{split}.file must be a portable manifest-adjacent basename"
            )
        file_key = portable_basename_key(file_name)
        if file_key in seen_split_files:
            raise CorpusContractError("corpus split files are not portable-distinct")
        seen_split_files.add(file_key)
        games = _positive_int(entry["games"], f"splits.{split}.games")
        rows = _nonnegative_int(entry["rows"], f"splits.{split}.rows")
        if games != _positive_int(split_sizes[split], f"splitSizes.{split}"):
            raise CorpusContractError(f"splits.{split}.games disagrees with splitSizes")
        seeds = _seed_list(entry["seeds"], f"splits.{split}.seeds")
        if len(seeds) != games:
            raise CorpusContractError(f"splits.{split}.seeds length is invalid")
        outcomes = _outcome_ledger(
            entry, split=split, seeds=seeds, rows=rows
        )
        rule_domain = set(manifest["ruleIds"])
        agent_domain = set(manifest["agentIds"])
        for outcome in outcomes:
            if (
                outcome["whiteRuleId"] not in rule_domain
                or outcome["blackRuleId"] not in rule_domain
            ):
                raise CorpusContractError(
                    f"splits.{split}.outcomes contains an unknown rule"
                )
            if (
                outcome["whiteAgentId"] not in agent_domain
                or outcome["blackAgentId"] not in agent_domain
            ):
                raise CorpusContractError(
                    f"splits.{split}.outcomes contains an unknown agent"
                )
        expected_split = Split(split)
        if any(assign_split(seed) is not expected_split for seed in seeds):
            raise CorpusContractError(
                f"splits.{split}.seeds violates the hash split contract"
            )
        overlap = seen_seeds.intersection(seeds)
        if overlap:
            raise CorpusContractError("corpus split seeds overlap")
        seen_seeds.update(seeds)
        total_games += games
        total_rows += rows
    if manifest["totalGames"] != total_games or manifest["totalRows"] != total_rows:
        raise CorpusContractError("manifest totals disagree with split totals")


def _audit_loaded_corpus_split(
    manifest_path: Path,
    manifest_sha256: str,
    manifest: Mapping[str, Any],
    entry: Mapping[str, Any],
    split: str,
    *,
    require_complete_catalog: bool = True,
    verify_balanced_schedule: bool = True,
    dataset_path_override: Path | None = None,
    release_root_sha256: str | None = None,
    corpus_run_id: str | None = None,
    dataset_source: BinaryIO | None = None,
    verify_semantic_replay: bool = False,
) -> AuditedCorpusSplit:
    """Audit an already authenticated split without consulting sibling manifests."""
    file_name = _string(entry["file"], f"splits.{split}.file")
    if not is_portable_safe_basename(file_name):
        raise CorpusContractError(
            "split file must be a portable manifest-adjacent basename"
        )
    dataset_path = (
        dataset_path_override.resolve()
        if dataset_path_override is not None
        else manifest_path.parent / file_name
    )
    if (
        not is_portable_safe_basename(dataset_path.name)
        or dataset_path.name != file_name
    ):
        raise CorpusContractError("dataset basename disagrees with private manifest")
    expected_digest = _digest(entry["sha256"], f"splits.{split}.sha256")
    expected_bytes = _nonnegative_int(entry["bytes"], f"splits.{split}.bytes")
    expected_rows = _nonnegative_int(entry["rows"], f"splits.{split}.rows")
    expected_games = _positive_int(entry["games"], f"splits.{split}.games")
    expected_seeds = _seed_list(entry["seeds"], f"splits.{split}.seeds")
    outcomes = _outcome_ledger(
        entry,
        split=split,
        seeds=expected_seeds,
        rows=expected_rows,
    )
    semantic_replay = (
        StreamingSemanticReplayVerifier(
            max_plies=_positive_int(manifest["maxPlies"], "maxPlies"),
        )
        if verify_semantic_replay
        else None
    )
    expected_slots = None
    if (
        verify_balanced_schedule
        and manifest["ruleAssignmentPolicy"] == "balanced-symmetric-v1"
    ):
        try:
            expected_slots = expected_balanced_slots(
                root_seed=manifest["rootSeed"],
                split=split,
                seeds=expected_seeds,
                rule_ids=manifest["ruleIds"],
                agent_ids=manifest["agentIds"],
            )
        except ValueError as error:
            raise CorpusContractError("balanced schedule is invalid") from error
        for index, slot in enumerate(expected_slots):
            outcome = outcomes[index]
            if (
                outcome["whiteRuleId"] != slot.white_rule_id
                or outcome["blackRuleId"] != slot.black_rule_id
                or outcome["whiteAgentId"] != slot.white_agent_id
                or outcome["blackAgentId"] != slot.black_agent_id
            ):
                raise CorpusContractError(
                    "outcome assignments disagree with balanced schedule"
                )
    expected_game_ids = tuple(
        f"{seed:08x}-{index:06d}"
        for index, seed in enumerate(expected_seeds)
    )
    game_index_by_id = {
        game_id: index for index, game_id in enumerate(expected_game_ids)
    }
    expected_coverage = _mapping(entry["coverage"], f"splits.{split}.coverage")
    if set(expected_coverage) != {"assignedGames", "observedRows"}:
        raise CorpusContractError(
            "split coverage must contain assignedGames and observedRows"
        )
    expected_assigned = _mapping(
        expected_coverage["assignedGames"],
        f"splits.{split}.coverage.assignedGames",
    )
    expected_observed = _mapping(
        expected_coverage["observedRows"],
        f"splits.{split}.coverage.observedRows",
    )
    if set(expected_assigned) != {"white", "black"}:
        raise CorpusContractError("assignedGames must contain white and black")
    if set(expected_observed) != {"white", "black"}:
        raise CorpusContractError("observedRows must contain white and black")
    expected_white_assigned = _coverage(
        expected_assigned["white"],
        f"splits.{split}.coverage.assignedGames.white",
    )
    expected_black_assigned = _coverage(
        expected_assigned["black"],
        f"splits.{split}.coverage.assignedGames.black",
    )
    expected_white_rows = _coverage(
        expected_observed["white"],
        f"splits.{split}.coverage.observedRows.white",
    )
    expected_black_rows = _coverage(
        expected_observed["black"],
        f"splits.{split}.coverage.observedRows.black",
    )

    policy_id = _string(manifest["evaluatorPolicyId"], "evaluatorPolicyId")
    policy_version = _positive_int(
        manifest["evaluatorPolicyVersion"], "evaluatorPolicyVersion"
    )
    engine_fingerprint = _string(
        manifest["engineFingerprint"], "engineFingerprint"
    )
    search_limit = _mapping(
        manifest["evaluatorSearchLimit"], "evaluatorSearchLimit"
    )

    digest = hashlib.sha256()
    byte_count = 0
    row_count = 0
    game_order: list[str] = []
    game_seed: dict[str, int] = {}
    game_labels: dict[str, dict[str, str]] = {}
    game_agents: dict[str, dict[str, str]] = {}
    game_results: dict[str, Mapping[str, Any]] = {}
    observed_seed_order: list[int] = []
    next_ply: dict[str, int] = {}
    previous_game_index = -1
    white_row_counts = {rule_id: 0 for rule_id in SYMBOLIC_RULE_IDS}
    black_row_counts = {rule_id: 0 for rule_id in SYMBOLIC_RULE_IDS}
    if dataset_source is None:
        try:
            source = dataset_path.open("rb")
        except OSError as error:
            raise CorpusContractError(
                f"cannot read corpus split: {dataset_path}"
            ) from error
        source_context = source
    else:
        dataset_source.seek(0)
        source = dataset_source
        source_context = nullcontext(source)
    with source_context:
        for line_number, raw_line in enumerate(source, start=1):
            digest.update(raw_line)
            byte_count += len(raw_line)
            if not raw_line.strip():
                raise CorpusContractError(
                    f"{dataset_path.name}:{line_number} is blank"
                )
            try:
                value = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise CorpusContractError(
                    f"{dataset_path.name}:{line_number} is not valid UTF-8 JSON"
                ) from error
            if not isinstance(value, Mapping):
                raise CorpusContractError(
                    f"{dataset_path.name}:{line_number} must contain an object"
                )
            try:
                row = parse_dataset_row(value)
            except DatasetSchemaError as error:
                raise CorpusContractError(
                    f"{dataset_path.name}:{line_number}: {error}"
                ) from error
            if row.features.symbolic_feature_version != SYMBOLIC_FEATURE_VERSION:
                raise CorpusContractError("every row must use symbolic feature version 6")
            constraint = row.features.public_evaluator_constraint
            if constraint is None:
                raise CorpusContractError("every row must have a public evaluator fact")
            if constraint.policy_id != policy_id:
                raise CorpusContractError("row evaluator policy does not match manifest")
            if constraint.engine_fingerprint != engine_fingerprint:
                raise CorpusContractError(
                    "row evaluator fingerprint does not match manifest"
                )
            ordinary_moves = tuple(row.features.ordinary_legal_moves)
            roots = sorted({move.strip().lower() for move in ordinary_moves})
            expected_position_key = _canonical_json([row.features.fen_before, roots])
            if constraint.position_key != expected_position_key:
                raise CorpusContractError("row evaluator position key is stale")
            if constraint.best_move_uci not in roots:
                raise CorpusContractError(
                    "row evaluator best move is outside ordinary legal moves"
                )
            expected_request_digest = _evaluator_request_digest(
                fen=row.features.fen_before,
                ordinary_moves=ordinary_moves,
                policy_id=policy_id,
                policy_version=policy_version,
                engine_fingerprint=engine_fingerprint,
                search_limit=search_limit,
            )
            if constraint.request_digest != expected_request_digest:
                raise CorpusContractError("row evaluator request digest is stale")
            if row.labels.true_drawback not in SYMBOLIC_RULE_IDS:
                raise CorpusContractError("row drawback is outside the canonical catalog")
            true_index = SYMBOLIC_RULE_INDEX[row.labels.true_drawback]
            active_eliminated = (
                row.features.symbolic_white_eliminated
                if row.features.player_color == "white"
                else row.features.symbolic_black_eliminated
            )
            if active_eliminated[true_index]:
                raise CorpusContractError(
                    "symbolic evidence hard-eliminates the active player's "
                    "known true drawback"
                )
            if assign_split(row.seed).value != split:
                raise CorpusContractError("row seed violates the requested split")
            game_index = game_index_by_id.get(row.game_id)
            if game_index is None or row.seed != expected_seeds[game_index]:
                raise CorpusContractError(
                    "row game ID or seed is outside the scheduled corpus"
                )
            if semantic_replay is not None:
                try:
                    semantic_replay.observe(
                        {
                            key: value[key]
                            for key in SEMANTIC_REPLAY_PUBLIC_KEYS
                        },
                        line_number=line_number,
                        expected_final_fen=outcomes[game_index]["finalFen"],
                    )
                except SemanticReplayError as error:
                    raise CorpusContractError(
                        f"{dataset_path.name}:{line_number}: semantic replay: {error}"
                    ) from error
            if game_index < previous_game_index:
                raise CorpusContractError(
                    "observed games are not a canonical schedule subsequence"
                )
            previous_game_index = game_index
            expected_ply = next_ply.get(row.game_id, 0)
            if (
                row.features.ply != expected_ply
                or row.features.player_color
                != ("white" if expected_ply % 2 == 0 else "black")
            ):
                raise CorpusContractError("row ply sequence is not canonical")
            next_ply[row.game_id] = expected_ply + 1
            outcome = outcomes[game_index]
            row_result = _session_result(
                value.get("result"),
                f"{dataset_path.name}:{line_number}.result",
            )
            previous_result = game_results.get(row.game_id)
            if previous_result is not None and previous_result != row_result:
                raise CorpusContractError(
                    "one game contains inconsistent terminal results"
                )
            if row_result != outcome["result"]:
                raise CorpusContractError(
                    "row result differs from its outcome ledger"
                )
            game_results[row.game_id] = row_result
            expected_rule = (
                outcome["whiteRuleId"]
                if row.labels.player_color == "white"
                else outcome["blackRuleId"]
            )
            expected_agent = (
                outcome["whiteAgentId"]
                if row.labels.player_color == "white"
                else outcome["blackAgentId"]
            )
            if row.labels.true_drawback != expected_rule:
                raise CorpusContractError(
                    "row drawback differs from its scheduled assignment"
                )
            if row.game_id not in game_seed:
                game_seed[row.game_id] = row.seed
                game_labels[row.game_id] = {}
                game_agents[row.game_id] = {}
                game_order.append(row.game_id)
                observed_seed_order.append(row.seed)
            elif game_seed[row.game_id] != row.seed:
                raise CorpusContractError("one game ID contains multiple seeds")
            labels = game_labels[row.game_id]
            previous = labels.get(row.labels.player_color)
            if previous is not None and previous != row.labels.true_drawback:
                raise CorpusContractError("one game/color contains multiple drawbacks")
            labels[row.labels.player_color] = row.labels.true_drawback
            agent_id = value.get("botAgentId")
            if (
                not isinstance(agent_id, str)
                or agent_id not in manifest["agentIds"]
                or agent_id != expected_agent
            ):
                raise CorpusContractError(
                    "row botAgentId differs from its scheduled assignment"
                )
            agents = game_agents[row.game_id]
            previous_agent = agents.get(row.labels.player_color)
            if previous_agent is not None and previous_agent != agent_id:
                raise CorpusContractError("one game/color contains multiple agents")
            agents[row.labels.player_color] = agent_id
            if row.labels.player_color == "white":
                white_row_counts[row.labels.true_drawback] += 1
            else:
                black_row_counts[row.labels.true_drawback] += 1
            row_count += 1
    if dataset_source is not None:
        dataset_source.seek(0)
    if semantic_replay is not None:
        try:
            semantic_replay.finish(
                expected_game_count=sum(
                    outcome["plyCount"] > 0 for outcome in outcomes
                )
            )
        except SemanticReplayError as error:
            raise CorpusContractError(f"semantic replay: {error}") from error

    actual_digest = digest.hexdigest()
    if actual_digest != expected_digest or byte_count != expected_bytes:
        raise CorpusContractError("split bytes do not match manifest identity")
    if row_count != expected_rows:
        raise CorpusContractError("split row count does not match manifest")
    expected_observed_seeds = tuple(
        expected_seeds[index]
        for index, outcome in enumerate(outcomes)
        if outcome["plyCount"] > 0
    )
    if tuple(observed_seed_order) != expected_observed_seeds:
        raise CorpusContractError(
            "observed games do not match the authenticated outcome ledger"
        )
    for index, game_id in enumerate(expected_game_ids):
        if next_ply.get(game_id, 0) != outcomes[index]["plyCount"]:
            raise CorpusContractError(
                "serialized rows disagree with outcome ledger ply counts"
            )
    white_counts = {rule_id: 0 for rule_id in SYMBOLIC_RULE_IDS}
    black_counts = {rule_id: 0 for rule_id in SYMBOLIC_RULE_IDS}
    for outcome in outcomes:
        white_rule = outcome["whiteRuleId"]
        black_rule = outcome["blackRuleId"]
        if white_rule not in white_counts or black_rule not in black_counts:
            raise CorpusContractError(
                "outcome assignment is outside the canonical rule catalog"
            )
        white_counts[white_rule] += 1
        black_counts[black_rule] += 1
    actual_white_assigned = tuple(
        (rule_id, white_counts[rule_id]) for rule_id in SYMBOLIC_RULE_IDS
    )
    actual_black_assigned = tuple(
        (rule_id, black_counts[rule_id]) for rule_id in SYMBOLIC_RULE_IDS
    )
    actual_white_rows = tuple(
        (rule_id, white_row_counts[rule_id]) for rule_id in SYMBOLIC_RULE_IDS
    )
    actual_black_rows = tuple(
        (rule_id, black_row_counts[rule_id]) for rule_id in SYMBOLIC_RULE_IDS
    )
    if (
        actual_white_assigned != expected_white_assigned
        or actual_black_assigned != expected_black_assigned
        or actual_white_rows != expected_white_rows
        or actual_black_rows != expected_black_rows
    ):
        raise CorpusContractError("empirical rule coverage does not match manifest")
    if require_complete_catalog and (
        any(count == 0 for _, count in actual_white_rows)
        or any(count == 0 for _, count in actual_black_rows)
    ):
        raise CorpusContractError(
            "complete-catalog training requires every rule in both colors"
        )
    return AuditedCorpusSplit(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        split=split,
        dataset_path=dataset_path,
        dataset_sha256=actual_digest,
        dataset_bytes=byte_count,
        rows=row_count,
        games=expected_games,
        seeds=expected_seeds,
        observed_seeds=tuple(observed_seed_order),
        game_assignments=tuple(
            (
                expected_game_ids[index],
                outcome["whiteRuleId"],
                outcome["blackRuleId"],
            )
            for index, outcome in enumerate(outcomes)
        ),
        outcomes_sha256=_digest(
            entry["outcomesSha256"],
            f"splits.{split}.outcomesSha256",
        ),
        row_bearing_games=_nonnegative_int(
            entry["rowBearingGames"],
            f"splits.{split}.rowBearingGames",
        ),
        zero_ply_games=_nonnegative_int(
            entry["zeroPlyGames"],
            f"splits.{split}.zeroPlyGames",
        ),
        one_sided_games=_nonnegative_int(
            entry["oneSidedGames"],
            f"splits.{split}.oneSidedGames",
        ),
        white_assigned_games=actual_white_assigned,
        black_assigned_games=actual_black_assigned,
        white_observed_rows=actual_white_rows,
        black_observed_rows=actual_black_rows,
        engine_fingerprint=engine_fingerprint,
        evaluator_policy_id=policy_id,
        evaluator_policy_version=policy_version,
        release_root_sha256=release_root_sha256,
        corpus_run_id=corpus_run_id,
        max_plies=manifest["maxPlies"],
    )


def _load_canonical_release_json_stream(
    source: BinaryIO, label: str
) -> tuple[Mapping[str, Any], bytes, str]:
    source.seek(0)
    try:
        payload = source.read()
    except OSError as error:
        raise CorpusContractError(f"cannot read {label}") from error
    finally:
        source.seek(0)
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CorpusContractError(f"{label} is not valid UTF-8 JSON") from error
    manifest = _mapping(value, label)
    canonical = (
        json.dumps(manifest, ensure_ascii=False, indent=2, separators=(",", ": "))
        + "\n"
    ).encode("utf-8")
    if payload != canonical:
        raise CorpusContractError(f"{label} is not canonical exact JSON")
    return manifest, payload, hashlib.sha256(payload).hexdigest()


def _reject_duplicate_pairs(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CorpusContractError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> object:
    raise CorpusContractError(f"non-finite JSON number is forbidden: {value}")


def _strict_json_bytes(payload: bytes, label: str) -> object:
    try:
        return json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_json,
        )
    except CorpusContractError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CorpusContractError(f"{label} is not valid UTF-8 JSON") from error


def _load_canonical_exact_json_stream(
    source: BinaryIO, label: str
) -> tuple[Mapping[str, Any], bytes, str]:
    source.seek(0)
    try:
        payload = source.read()
    except OSError as error:
        raise CorpusContractError(f"cannot read {label}") from error
    finally:
        source.seek(0)
    value = _strict_json_bytes(payload, label)
    document = _mapping(value, label)
    canonical = (
        json.dumps(
            document,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            separators=(",", ": "),
        )
        + "\n"
    ).encode("utf-8")
    if payload != canonical:
        raise CorpusContractError(f"{label} is not canonical exact JSON")
    return document, payload, hashlib.sha256(payload).hexdigest()


def _hard_negative_profile(
    value: object, expected_profile_id: str, label: str
) -> tuple[str, tuple[str, str], str, int]:
    profile = _mapping(value, label)
    _exact_keys(profile, EXPECTED_HARD_NEGATIVE_PROFILE_KEYS, label)
    frozen = HARD_NEGATIVE_PROFILES.get(expected_profile_id)
    if frozen is None:
        raise CorpusContractError("expected hard-negative profile ID is unknown")
    description, rule_ids, evidence, root_seed = frozen
    expected = {
        "id": expected_profile_id,
        "description": description,
        "ruleIds": list(rule_ids),
        "evidence": evidence,
    }
    if profile != expected:
        raise CorpusContractError(
            "hard-negative profile does not equal the frozen named profile"
        )
    return description, rule_ids, evidence, root_seed


def _hard_negative_empty_split(
    value: object, split: str, rule_ids: tuple[str, str]
) -> None:
    entry = _mapping(value, f"splits.{split}")
    _exact_keys(entry, EXPECTED_SPLIT_KEYS, f"splits.{split}")
    zero = {rule_id: 0 for rule_id in rule_ids}
    expected = {
        "file": None,
        "games": 0,
        "rowBearingGames": 0,
        "zeroPlyGames": 0,
        "oneSidedGames": 0,
        "rows": 0,
        "seeds": [],
        "bytes": 0,
        "sha256": EMPTY_SHA256,
        "outcomesSha256": hashlib.sha256(b"[]").hexdigest(),
        "outcomes": [],
        "coverage": {
            "assignedGames": {"white": zero, "black": zero},
            "observedRows": {"white": zero, "black": zero},
        },
    }
    if entry != expected:
        raise CorpusContractError(
            f"splits.{split} must be the canonical empty hard-negative split"
        )


def _hard_negative_coverage(
    entry: Mapping[str, Any],
    rule_ids: tuple[str, str],
) -> Mapping[str, Any]:
    coverage = _mapping(entry["coverage"], "splits.train.coverage")
    _exact_keys(
        coverage,
        frozenset({"assignedGames", "observedRows"}),
        "splits.train.coverage",
    )
    expanded: dict[str, object] = {}
    for group_name in ("assignedGames", "observedRows"):
        group = _mapping(
            coverage[group_name], f"splits.train.coverage.{group_name}"
        )
        _exact_keys(
            group,
            frozenset({"white", "black"}),
            f"splits.train.coverage.{group_name}",
        )
        colors: dict[str, object] = {}
        for color in ("white", "black"):
            counts = _mapping(
                group[color],
                f"splits.train.coverage.{group_name}.{color}",
            )
            if tuple(counts) != rule_ids:
                raise CorpusContractError(
                    "hard-negative coverage keys must equal the ordered profile pair"
                )
            full = {rule_id: 0 for rule_id in SYMBOLIC_RULE_IDS}
            for rule_id in rule_ids:
                full[rule_id] = _nonnegative_int(
                    counts[rule_id],
                    f"splits.train.coverage.{group_name}.{color}.{rule_id}",
                )
            colors[color] = full
        expanded[group_name] = colors
    return expanded


def _hard_negative_plan_shards(
    shard_size: int, total_games: int
) -> list[dict[str, object]]:
    shards: list[dict[str, object]] = []
    start = 0
    while start < total_games:
        end = min(total_games, start + shard_size)
        shards.append(
            {
                "id": f"train-{start}-{end}",
                "split": "train",
                "shardIndex": len(shards),
                "splitShardIndex": len(shards),
                "splitStart": start,
                "splitEnd": end,
                "globalStart": start,
                "globalEnd": end,
                "gameCount": end - start,
            }
        )
        start = end
    return shards


def _derive_hard_negative_game_seed(root_seed: int, game_index: int) -> int:
    def imul(left: int, right: int) -> int:
        return (
            (left & 0xFFFF_FFFF) * (right & 0xFFFF_FFFF)
        ) & 0xFFFF_FFFF

    value = (
        root_seed ^ imul(game_index + 1, 0x9E37_79B9)
    ) & 0xFFFF_FFFF
    value ^= value >> 16
    value = imul(value, 0x21F0_AAAD)
    value ^= value >> 15
    value = imul(value, 0x735A_2D97)
    return (value ^ (value >> 15)) & 0xFFFF_FFFF


def _expected_hard_negative_train_seeds(
    root_seed: int,
) -> tuple[int, ...]:
    seeds: list[int] = []
    game_index = 0
    while len(seeds) < 1_000:
        seed = _derive_hard_negative_game_seed(root_seed, game_index)
        if assign_split(seed) is Split.TRAIN:
            seeds.append(seed)
        game_index += 1
    return tuple(seeds)


def _hard_negative_slot_material(
    split: str,
    split_index: int,
    seed: int,
    slot: ExpectedEvaluatorSlot,
) -> dict[str, object]:
    return {
        "split": split,
        "splitIndex": split_index,
        "seed": seed,
        "whiteRuleId": slot.white_rule_id,
        "blackRuleId": slot.black_rule_id,
        "whiteAgentId": slot.white_agent_id,
        "blackAgentId": slot.black_agent_id,
    }


class _HardNegativeMulberry32:
    def __init__(self, seed: int) -> None:
        self._state = seed & 0xFFFF_FFFF

    def integer(self, maximum: int) -> int:
        self._state = (self._state + 0x6D2B_79F5) & 0xFFFF_FFFF
        value = self._state
        value = ((value ^ (value >> 15)) * (value | 1)) & 0xFFFF_FFFF
        value ^= (
            value
            + (((value ^ (value >> 7)) * (value | 61)) & 0xFFFF_FFFF)
        ) & 0xFFFF_FFFF
        value &= 0xFFFF_FFFF
        unit = ((value ^ (value >> 14)) & 0xFFFF_FFFF) / 4_294_967_296
        return int(unit * maximum)


def _hard_negative_permutation(
    values: tuple[str, ...], seed: int
) -> tuple[str, ...]:
    shuffled = list(values)
    rng = _HardNegativeMulberry32(seed)
    for index in range(len(shuffled) - 1, 0, -1):
        selected = rng.integer(index + 1)
        shuffled[index], shuffled[selected] = (
            shuffled[selected],
            shuffled[index],
        )
    return tuple(shuffled)


def _expected_hard_negative_slots(
    root_seed: int,
    seeds: tuple[int, ...],
    rule_ids: tuple[str, str],
) -> tuple[ExpectedEvaluatorSlot, ...]:
    if len(seeds) % len(rule_ids) != 0:
        raise CorpusContractError(
            "hard-negative schedule must be a rule-count multiple"
        )
    rounds = len(seeds) // len(rule_ids)
    if rounds % len(HARD_NEGATIVE_AGENT_IDS) != 0 or rounds % 2 != 0:
        raise CorpusContractError(
            "hard-negative schedule cannot balance rules and agents"
        )
    split_domain = 0x243F_6A88
    rules = _hard_negative_permutation(
        rule_ids,
        (root_seed ^ split_domain ^ 0x9E37_79B9) & 0xFFFF_FFFF,
    )
    white_agents = _hard_negative_permutation(
        HARD_NEGATIVE_AGENT_IDS,
        (root_seed ^ split_domain ^ 0xA409_3822) & 0xFFFF_FFFF,
    )
    black_agents = _hard_negative_permutation(
        HARD_NEGATIVE_AGENT_IDS,
        (root_seed ^ split_domain ^ 0x299F_31D0) & 0xFFFF_FFFF,
    )
    slots: list[ExpectedEvaluatorSlot] = []
    rule_count = len(rules)
    for split_index in range(len(seeds)):
        round_index = split_index // rule_count
        rule_index = split_index % rule_count
        magnitude = (round_index // 2) % (rule_count - 1) + 1
        offset = magnitude if round_index % 2 == 0 else -magnitude
        black_index = (rule_index + offset + rule_count) % rule_count
        slots.append(
            ExpectedEvaluatorSlot(
                white_rule_id=rules[rule_index],
                black_rule_id=rules[black_index],
                white_agent_id=white_agents[
                    (round_index + rule_index) % len(white_agents)
                ],
                black_agent_id=black_agents[
                    (round_index + black_index) % len(black_agents)
                ],
            )
        )
    return tuple(slots)


def _audit_hard_negative_plan(
    plan: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    profile: Mapping[str, Any],
    rule_ids: tuple[str, str],
    root_seed: int,
    source_revision: str,
    workers: int,
    generation: Mapping[str, Any],
) -> None:
    _exact_keys(plan, EXPECTED_HARD_NEGATIVE_PLAN_KEYS, "hard-negative plan")
    if plan["schemaVersion"] != 1:
        raise CorpusContractError("hard-negative plan schemaVersion must be 1")
    if plan["sourceRevision"] != source_revision:
        raise CorpusContractError("hard-negative source revision disagrees with plan")
    metadata = _mapping(plan["metadata"], "hard-negative plan metadata")
    _exact_keys(
        metadata,
        EXPECTED_HARD_NEGATIVE_METADATA_KEYS,
        "hard-negative plan metadata",
    )
    expected_metadata = {
        key: manifest[key]
        for key in metadata
    }
    if metadata != expected_metadata:
        raise CorpusContractError("hard-negative plan metadata disagrees with manifest")

    split_seeds = _mapping(plan["splitSeeds"], "hard-negative plan splitSeeds")
    _exact_keys(split_seeds, frozenset(SPLITS), "hard-negative plan splitSeeds")
    train_seeds = _seed_list(split_seeds["train"], "plan splitSeeds.train")
    if list(train_seeds) != manifest["splits"]["train"]["seeds"]:
        raise CorpusContractError("hard-negative plan seeds disagree with manifest")
    if train_seeds != _expected_hard_negative_train_seeds(root_seed):
        raise CorpusContractError(
            "hard-negative train seeds do not equal the frozen root-seed search"
        )
    if split_seeds["validation"] != [] or split_seeds["test"] != []:
        raise CorpusContractError("hard-negative plan may contain only train seeds")

    schedule = _mapping(plan["schedule"], "hard-negative plan schedule")
    _exact_keys(
        schedule,
        frozenset({"policyId", "splits"}),
        "hard-negative plan schedule",
    )
    if schedule["policyId"] != "balanced-symmetric-v1":
        raise CorpusContractError("hard-negative schedule policy is invalid")
    schedule_splits = _mapping(
        schedule["splits"], "hard-negative plan schedule.splits"
    )
    _exact_keys(
        schedule_splits,
        frozenset(SPLITS),
        "hard-negative plan schedule.splits",
    )
    if schedule_splits["validation"] != [] or schedule_splits["test"] != []:
        raise CorpusContractError("hard-negative schedule may contain only train slots")
    expected_slots = _expected_hard_negative_slots(
        root_seed, train_seeds, rule_ids
    )
    expected_train_slots = [
        _hard_negative_slot_material("train", index, seed, expected_slots[index])
        for index, seed in enumerate(train_seeds)
    ]
    if schedule_splits["train"] != expected_train_slots:
        raise CorpusContractError(
            "hard-negative plan assignments disagree with balanced schedule"
        )

    run_plan = _mapping(plan["runPlan"], "hard-negative plan runPlan")
    _exact_keys(
        run_plan,
        EXPECTED_HARD_NEGATIVE_RUN_PLAN_KEYS,
        "hard-negative plan runPlan",
    )
    if run_plan["schemaVersion"] != 1:
        raise CorpusContractError("hard-negative run plan schemaVersion must be 1")
    shard_size = _positive_safe_int(run_plan["shardSize"], "runPlan.shardSize")
    expected_shards = _hard_negative_plan_shards(shard_size, 1_000)
    actual_shards = run_plan["shards"]
    if (
        not isinstance(actual_shards, list)
        or len(actual_shards) != len(expected_shards)
    ):
        raise CorpusContractError("hard-negative run plan shards are invalid")
    for index, expected_shard in enumerate(expected_shards):
        actual = _mapping(actual_shards[index], f"runPlan.shards[{index}]")
        _exact_keys(
            actual,
            EXPECTED_HARD_NEGATIVE_SHARD_KEYS,
            f"runPlan.shards[{index}]",
        )
        slot_slice = expected_train_slots[
            int(expected_shard["splitStart"]):int(expected_shard["splitEnd"])
        ]
        assignment_digest = hashlib.sha256(
            _canonical_json(slot_slice).encode("utf-8")
        ).hexdigest()
        expected_shard["seedAssignmentSha256"] = assignment_digest
        if actual != expected_shard:
            raise CorpusContractError(
                "hard-negative run plan shard identity is invalid"
            )

    config_material = {
        "sourceRevision": source_revision,
        "schemaVersion": metadata["schemaVersion"],
        "generator": metadata["generator"],
        "rootSeed": metadata["rootSeed"],
        "seedPolicy": metadata["seedPolicy"],
        "splitFractions": metadata["splitFractions"],
        "splitSalt": metadata["splitSalt"],
        "maxPlies": metadata["maxPlies"],
        "ruleIds": metadata["ruleIds"],
        "symbolicFeatureVersion": metadata["symbolicFeatureVersion"],
        "symbolicRuleIds": metadata["symbolicRuleIds"],
        "agentIds": metadata["agentIds"],
        "evaluatorCoverage": metadata["evaluatorCoverage"],
        "evaluatorRequestSchemaVersion": metadata["evaluatorRequestSchemaVersion"],
        "evaluatorCacheSchemaVersion": metadata["evaluatorCacheSchemaVersion"],
        "evaluatorPolicyId": metadata["evaluatorPolicyId"],
        "evaluatorPolicyVersion": metadata["evaluatorPolicyVersion"],
        "engineFingerprint": metadata["engineFingerprint"],
        "engineBinarySha256": metadata["engineBinarySha256"],
        "evaluatorSearchLimit": metadata["evaluatorSearchLimit"],
        "ruleAssignmentPolicy": metadata["ruleAssignmentPolicy"],
        "observationPolicy": metadata["observationPolicy"],
        "workers": workers,
        "splitSizes": metadata["splitSizes"],
        "totalGames": metadata["totalGames"],
        "hardNegativeProfile": profile,
        "splitSeeds": {
            "train": list(train_seeds),
            "validation": [],
            "test": [],
        },
    }
    corpus_config_sha256 = hashlib.sha256(
        _canonical_json(config_material).encode("utf-8")
    ).hexdigest()
    schedule_material = {
        "policyId": "balanced-symmetric-v1",
        "slots": expected_train_slots,
    }
    schedule_sha256 = hashlib.sha256(
        _canonical_json(schedule_material).encode("utf-8")
    ).hexdigest()
    run_material = {
        "schemaVersion": 1,
        "corpusConfigSha256": corpus_config_sha256,
        "scheduleSha256": schedule_sha256,
        "ruleIds": list(rule_ids),
        "agentIds": list(HARD_NEGATIVE_AGENT_IDS),
        "shardSize": shard_size,
        "totalGames": 1_000,
        "shards": expected_shards,
    }
    run_id = hashlib.sha256(
        _canonical_json(run_material).encode("utf-8")
    ).hexdigest()
    expected_run_plan = {**run_material, "runId": run_id}
    if run_plan != expected_run_plan:
        raise CorpusContractError("hard-negative run identity is invalid")
    if (
        generation["runId"] != run_id
        or generation["corpusConfigSha256"] != corpus_config_sha256
    ):
        raise CorpusContractError(
            "hard-negative generation identity disagrees with exact plan"
        )


def _audit_hard_negative_handles(
    manifest_path: Path,
    dataset_path: Path,
    plan_path: Path,
    expected_profile_id: str,
    manifest_source: BinaryIO,
    dataset_source: BinaryIO,
    plan_source: BinaryIO,
) -> AuditedHardNegativeTrainCorpus:
    manifest, _manifest_payload, manifest_sha256 = (
        _load_canonical_exact_json_stream(
            manifest_source, "hard-negative manifest"
        )
    )
    plan, _plan_payload, plan_sha256 = _load_canonical_exact_json_stream(
        plan_source, "hard-negative plan"
    )
    _exact_keys(
        manifest,
        EXPECTED_HARD_NEGATIVE_TOP_LEVEL_KEYS,
        "hard-negative manifest",
    )
    _description, rule_ids, _evidence, root_seed = _hard_negative_profile(
        manifest["hardNegativeProfile"],
        expected_profile_id,
        "hardNegativeProfile",
    )
    profile = _mapping(manifest["hardNegativeProfile"], "hardNegativeProfile")
    if manifest["schemaVersion"] != 6:
        raise CorpusContractError("hard-negative corpus schemaVersion must be 6")
    if manifest["generator"] != "@drawbacktrainer/simulation":
        raise CorpusContractError("hard-negative corpus generator is invalid")
    if manifest["rootSeed"] != root_seed:
        raise CorpusContractError("hard-negative root seed is not frozen")
    if manifest["seedPolicy"] != "BLAKE2b-64(drawbacktrainer-v1:gameSeed)":
        raise CorpusContractError("hard-negative seed policy is invalid")
    if manifest["splitFractions"] != {
        "train": 0.8,
        "validation": 0.1,
        "test": 0.1,
    } or manifest["splitSalt"] != "drawbacktrainer-v1":
        raise CorpusContractError("hard-negative split policy is invalid")
    workers = _positive_safe_int(manifest["workers"], "workers")
    if manifest["maxPlies"] != 80:
        raise CorpusContractError("hard-negative maxPlies must be 80")
    if manifest["ruleIds"] != list(rule_ids):
        raise CorpusContractError("hard-negative ruleIds must equal the profile pair")
    if _digest(manifest["ruleIdsSha256"], "ruleIdsSha256") != hashlib.sha256(
        _canonical_json(list(rule_ids)).encode("utf-8")
    ).hexdigest():
        raise CorpusContractError("hard-negative ruleIdsSha256 is invalid")
    if manifest["symbolicFeatureVersion"] != SYMBOLIC_FEATURE_VERSION:
        raise CorpusContractError("hard-negative symbolic feature version must be 6")
    if manifest["symbolicRuleIds"] != list(SYMBOLIC_RULE_IDS):
        raise CorpusContractError(
            "hard-negative symbolicRuleIds must equal the 182-rule model order"
        )
    if _digest(
        manifest["symbolicRuleIdsSha256"], "symbolicRuleIdsSha256"
    ) != hashlib.sha256(
        _canonical_json(list(SYMBOLIC_RULE_IDS)).encode("utf-8")
    ).hexdigest():
        raise CorpusContractError("hard-negative symbolicRuleIdsSha256 is invalid")
    if manifest["agentIds"] != list(HARD_NEGATIVE_AGENT_IDS):
        raise CorpusContractError("hard-negative agents must equal the frozen five")
    if (
        manifest["evaluatorCoverage"] != "uniform-required"
        or manifest["evaluatorRequestSchemaVersion"] != 1
        or manifest["evaluatorCacheSchemaVersion"] != 1
        or manifest["evaluatorPolicyId"] != "stockfish-bestmove-v1"
        or manifest["evaluatorPolicyVersion"] != 1
        or manifest["evaluatorSearchLimit"] != {"kind": "nodes", "value": 10_000}
    ):
        raise CorpusContractError("hard-negative evaluator policy is invalid")
    binary_digest = _digest(
        manifest["engineBinarySha256"], "engineBinarySha256"
    )
    fingerprint = _string(manifest["engineFingerprint"], "engineFingerprint")
    parts = fingerprint.split(":")
    if len(parts) != 4 or parts[2] != binary_digest:
        raise CorpusContractError("hard-negative engine fingerprint is invalid")
    if (
        manifest["ruleAssignmentPolicy"] != "balanced-symmetric-v1"
        or manifest["observationPolicy"] != "single-attempt-allow-partial-v1"
        or manifest["splitSizes"]
        != {"train": 1_000, "validation": 0, "test": 0}
        or manifest["totalGames"] != 1_000
    ):
        raise CorpusContractError("hard-negative corpus schedule metadata is invalid")
    _nonnegative_int(manifest["totalRows"], "totalRows")

    splits = _mapping(manifest["splits"], "splits")
    _exact_keys(splits, frozenset(SPLITS), "splits")
    _hard_negative_empty_split(splits["validation"], "validation", rule_ids)
    _hard_negative_empty_split(splits["test"], "test", rule_ids)
    train = _mapping(splits["train"], "splits.train")
    _exact_keys(train, EXPECTED_SPLIT_KEYS, "splits.train")
    if (
        train["games"] != 1_000
        or manifest["totalRows"] != train["rows"]
        or manifest["splitSizes"]["train"] != train["games"]
    ):
        raise CorpusContractError("hard-negative train totals are invalid")
    if _positive_safe_int(train["rows"], "splits.train.rows") <= 0:
        raise CorpusContractError("hard-negative training corpus must contain rows")
    train_file = train["file"]
    if (
        not is_portable_safe_basename(train_file)
        or not is_portable_safe_basename(dataset_path.name)
        or train_file != dataset_path.name
    ):
        raise CorpusContractError("hard-negative dataset basename is invalid")
    seeds = _seed_list(train["seeds"], "splits.train.seeds")
    if len(seeds) != 1_000 or any(
        assign_split(seed) is not Split.TRAIN for seed in seeds
    ):
        raise CorpusContractError("hard-negative train seeds violate split assignment")
    outcomes = _outcome_ledger(
        train,
        split="train",
        seeds=seeds,
        rows=train["rows"],
    )
    for outcome in outcomes:
        if (
            outcome["whiteRuleId"] not in rule_ids
            or outcome["blackRuleId"] not in rule_ids
            or outcome["whiteAgentId"] not in HARD_NEGATIVE_AGENT_IDS
            or outcome["blackAgentId"] not in HARD_NEGATIVE_AGENT_IDS
        ):
            raise CorpusContractError(
                "hard-negative outcome is outside the profile or agent domain"
            )
    expanded_coverage = _hard_negative_coverage(train, rule_ids)

    generation = _mapping(
        manifest["hardNegativeGeneration"], "hardNegativeGeneration"
    )
    _exact_keys(
        generation,
        EXPECTED_HARD_NEGATIVE_GENERATION_KEYS,
        "hardNegativeGeneration",
    )
    if generation["version"] != 1:
        raise CorpusContractError("hardNegativeGeneration.version must be 1")
    source_revision = _string(
        generation["sourceRevision"], "hardNegativeGeneration.sourceRevision"
    )
    if re.fullmatch(r"[0-9a-f]{40}", source_revision) is None:
        raise CorpusContractError(
            "hard-negative sourceRevision must be a lowercase Git commit"
        )
    for field in ("runId", "corpusConfigSha256", "planSha256"):
        _digest(generation[field], f"hardNegativeGeneration.{field}")
    if generation["planSha256"] != plan_sha256:
        raise CorpusContractError("hard-negative plan digest disagrees with manifest")
    _audit_hard_negative_plan(
        plan,
        manifest,
        profile=profile,
        rule_ids=rule_ids,
        root_seed=root_seed,
        source_revision=source_revision,
        workers=workers,
        generation=generation,
    )

    dataset_source.seek(0)
    for line_number, raw_line in enumerate(dataset_source, start=1):
        if not raw_line.endswith(b"\n") or raw_line.endswith(b"\r\n"):
            raise CorpusContractError(
                f"{dataset_path.name}:{line_number} must use canonical LF framing"
            )
        if not raw_line[:-1]:
            raise CorpusContractError(
                f"{dataset_path.name}:{line_number} is blank"
            )
        value = _strict_json_bytes(
            raw_line[:-1], f"{dataset_path.name}:{line_number}"
        )
        if not isinstance(value, Mapping):
            raise CorpusContractError(
                f"{dataset_path.name}:{line_number} must contain an object"
            )
    dataset_source.seek(0)

    transformed_train = dict(train)
    transformed_train["coverage"] = expanded_coverage
    audited_split = _audit_loaded_corpus_split(
        manifest_path,
        manifest_sha256,
        manifest,
        transformed_train,
        "train",
        require_complete_catalog=False,
        verify_balanced_schedule=False,
        dataset_path_override=dataset_path,
        dataset_source=dataset_source,
        verify_semantic_replay=True,
    )
    return AuditedHardNegativeTrainCorpus(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        dataset_path=dataset_path,
        dataset_sha256=audited_split.dataset_sha256,
        dataset_bytes=audited_split.dataset_bytes,
        plan_path=plan_path,
        plan_sha256=plan_sha256,
        profile_id=expected_profile_id,
        rule_ids=rule_ids,
        root_seed=root_seed,
        source_revision=source_revision,
        run_id=_digest(generation["runId"], "hardNegativeGeneration.runId"),
        corpus_config_sha256=_digest(
            generation["corpusConfigSha256"],
            "hardNegativeGeneration.corpusConfigSha256",
        ),
        rows=audited_split.rows,
        games=audited_split.games,
        outcomes_sha256=audited_split.outcomes_sha256,
        observed_seeds=audited_split.observed_seeds,
        game_assignments=audited_split.game_assignments,
        row_bearing_games=audited_split.row_bearing_games,
        max_plies=80,
        agent_ids=HARD_NEGATIVE_AGENT_IDS,
        engine_binary_sha256=binary_digest,
        engine_fingerprint=fingerprint,
        evaluator_policy_id="stockfish-bestmove-v1",
        evaluator_policy_version=1,
        evaluator_nodes=10_000,
        observation_policy="single-attempt-allow-partial-v1",
    )


@contextmanager
def open_audited_hard_negative_train_corpus(
    manifest_path: Path,
    dataset_path: Path,
    plan_path: Path,
    expected_profile_id: str,
) -> Iterator[AuditedHardNegativeCorpusLease]:
    """Authenticate exactly one named, training-only hard-negative corpus.

    The three explicit paths are pinned for the full lease. No adjacent file is
    discovered, and this entrypoint is intentionally separate from release
    corpus validation.
    """

    resolved_manifest = manifest_path.resolve()
    resolved_dataset = dataset_path.resolve()
    resolved_plan = plan_path.resolve()
    try:
        manifest_source = resolved_manifest.open("rb")
        dataset_source = resolved_dataset.open("rb")
        plan_source = resolved_plan.open("rb")
    except OSError as error:
        for source in (
            locals().get("plan_source"),
            locals().get("dataset_source"),
            locals().get("manifest_source"),
        ):
            if source is not None:
                source.close()
        raise CorpusContractError(
            "cannot open pinned hard-negative corpus inputs"
        ) from error
    try:
        audited = _audit_hard_negative_handles(
            resolved_manifest,
            resolved_dataset,
            resolved_plan,
            expected_profile_id,
            manifest_source,
            dataset_source,
            plan_source,
        )
        lease = AuditedHardNegativeCorpusLease(
            audited=audited,
            manifest=manifest_source,
            dataset=dataset_source,
            plan=plan_source,
        )
        try:
            yield lease
        finally:
            lease.verify_unchanged()
    finally:
        plan_source.close()
        dataset_source.close()
        manifest_source.close()


def _audit_private_corpus_handles(
    root_manifest_path: Path,
    private_manifest_path: Path,
    dataset_path: Path,
    split: str,
    root_source: BinaryIO,
    private_source: BinaryIO,
    dataset_source: BinaryIO,
    *,
    require_complete_catalog: bool = True,
) -> AuditedCorpusSplit:
    """Audit one authorized private split against its public root commitment.

    Only the two explicitly supplied manifest files and dataset are opened.
    Sibling private manifests are neither resolved nor inspected.
    """

    if split not in SPLITS:
        raise CorpusContractError("split must be train, validation, or test")
    root_path = root_manifest_path.resolve()
    private_path = private_manifest_path.resolve()
    root, _root_payload, root_sha256 = _load_canonical_release_json_stream(
        root_source, "public release manifest"
    )
    _exact_keys(root, EXPECTED_RELEASE_ROOT_KEYS, "public release manifest")
    if root["releaseManifestVersion"] != 1:
        raise CorpusContractError("releaseManifestVersion must be 1")
    corpus_run_id = _digest(root["corpusRunId"], "corpusRunId")
    corpus = _mapping(root["corpus"], "public release corpus")
    _exact_keys(corpus, EXPECTED_PUBLIC_CORPUS_KEYS, "public release corpus")
    _validate_corpus_metadata(
        corpus,
        require_complete_catalog=require_complete_catalog,
        require_root_seed=False,
    )
    split_sizes = _mapping(corpus["splitSizes"], "splitSizes")
    commitments = _mapping(root["splits"], "public release splits")
    if set(split_sizes) != set(SPLITS) or set(commitments) != set(SPLITS):
        raise CorpusContractError("public release must commit exactly three splits")
    total_games = 0
    total_rows = 0
    parsed_commitments: dict[str, Mapping[str, Any]] = {}
    for split_name in SPLITS:
        commitment = _mapping(
            commitments[split_name], f"public release splits.{split_name}"
        )
        _exact_keys(
            commitment,
            EXPECTED_ROOT_SPLIT_KEYS,
            f"public release splits.{split_name}",
        )
        games = _positive_safe_int(
            commitment["games"], f"splits.{split_name}.games"
        )
        rows = _nonnegative_int(commitment["rows"], f"splits.{split_name}.rows")
        if games != _positive_safe_int(
            split_sizes[split_name], f"splitSizes.{split_name}"
        ):
            raise CorpusContractError("public split games disagree with splitSizes")
        _nonnegative_int(
            commitment["datasetBytes"], f"splits.{split_name}.datasetBytes"
        )
        _digest(
            commitment["datasetSha256"], f"splits.{split_name}.datasetSha256"
        )
        _digest(
            commitment["privateManifestSha256"],
            f"splits.{split_name}.privateManifestSha256",
        )
        total_games += games
        total_rows += rows
        parsed_commitments[split_name] = commitment
    if corpus["totalGames"] != total_games or corpus["totalRows"] != total_rows:
        raise CorpusContractError("public release totals disagree with commitments")

    private, _private_payload, private_sha256 = _load_canonical_release_json_stream(
        private_source, f"{split} private manifest"
    )
    commitment = parsed_commitments[split]
    if private_sha256 != commitment["privateManifestSha256"]:
        raise CorpusContractError("private manifest digest disagrees with public root")
    _exact_keys(private, EXPECTED_PRIVATE_MANIFEST_KEYS, "private split manifest")
    if private["manifestVersion"] != 1:
        raise CorpusContractError("private manifestVersion must be 1")
    if private["corpusRunId"] != corpus_run_id:
        raise CorpusContractError("private manifest corpusRunId disagrees with root")
    if private["split"] != split:
        raise CorpusContractError("private manifest declares another split")
    entry = _mapping(private["dataset"], "private split dataset")
    _exact_keys(entry, EXPECTED_SPLIT_KEYS, "private split dataset")
    if (
        entry["games"] != commitment["games"]
        or entry["rows"] != commitment["rows"]
        or entry["bytes"] != commitment["datasetBytes"]
        or entry["sha256"] != commitment["datasetSha256"]
    ):
        raise CorpusContractError("private dataset identity disagrees with public root")
    if entry["games"] != split_sizes[split]:
        raise CorpusContractError("private split size disagrees with public corpus")

    # The public release intentionally omits rootSeed. All remaining metadata,
    # ledger, coverage, row-schema, evaluator, and content checks are identical
    # to the legacy audit.
    return _audit_loaded_corpus_split(
        private_path,
        private_sha256,
        corpus,
        entry,
        split,
        require_complete_catalog=require_complete_catalog,
        verify_balanced_schedule=False,
        dataset_path_override=dataset_path,
        release_root_sha256=root_sha256,
        corpus_run_id=corpus_run_id,
        dataset_source=dataset_source,
        verify_semantic_replay=True,
    )


@contextmanager
def open_audited_private_corpus_split(
    root_manifest_path: Path,
    private_manifest_path: Path,
    dataset_path: Path,
    split: str,
    *,
    require_complete_catalog: bool = True,
) -> Iterator[AuditedPrivateCorpusLease]:
    """Pin, authenticate, and retain exactly one authorized private split."""

    root_path = root_manifest_path.resolve()
    private_path = private_manifest_path.resolve()
    resolved_dataset = dataset_path.resolve()
    try:
        root_source = root_path.open("rb")
        private_source = private_path.open("rb")
        dataset_source = resolved_dataset.open("rb")
    except OSError as error:
        for source in (
            locals().get("dataset_source"),
            locals().get("private_source"),
            locals().get("root_source"),
        ):
            if source is not None:
                source.close()
        raise CorpusContractError("cannot open pinned private corpus inputs") from error
    try:
        audited = _audit_private_corpus_handles(
            root_path,
            private_path,
            resolved_dataset,
            split,
            root_source,
            private_source,
            dataset_source,
            require_complete_catalog=require_complete_catalog,
        )
        lease = AuditedPrivateCorpusLease(
            audited=audited,
            root=root_source,
            private_manifest=private_source,
            dataset=dataset_source,
        )
        try:
            yield lease
        finally:
            lease.verify_dataset_unchanged()
    finally:
        dataset_source.close()
        private_source.close()
        root_source.close()


def audit_private_corpus_split(
    root_manifest_path: Path,
    private_manifest_path: Path,
    dataset_path: Path,
    split: str,
    *,
    require_complete_catalog: bool = True,
) -> AuditedCorpusSplit:
    """Compatibility wrapper that authenticates and closes a pinned lease."""

    with open_audited_private_corpus_split(
        root_manifest_path,
        private_manifest_path,
        dataset_path,
        split,
        require_complete_catalog=require_complete_catalog,
    ) as lease:
        return lease.audited


def audit_corpus_split(
    manifest_path: Path,
    split: str,
    *,
    require_complete_catalog: bool = True,
) -> AuditedCorpusSplit:
    """Validate one legacy monolithic split.

    Release-candidate workflows must use :func:`audit_private_corpus_split`.
    This entrypoint remains available only for explicitly legacy research flows.
    """

    if split not in SPLITS:
        raise CorpusContractError("split must be train, validation, or test")
    manifest_path = manifest_path.resolve()
    manifest, manifest_sha256 = _load_manifest(manifest_path)
    _validate_manifest_identity(
        manifest,
        require_complete_catalog=require_complete_catalog,
    )
    entries = _mapping(manifest["splits"], "splits")
    entry = _mapping(entries[split], f"splits.{split}")
    return _audit_loaded_corpus_split(
        manifest_path,
        manifest_sha256,
        manifest,
        entry,
        split,
        require_complete_catalog=require_complete_catalog,
    )
