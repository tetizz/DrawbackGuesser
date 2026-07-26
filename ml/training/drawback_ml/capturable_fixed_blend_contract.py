"""Immutable calculations and artifact contract for fixed-blend confirmation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .capturable_baseline import _top_k_vector_credit
from .capturable_blend_contract import (
    ComponentPredictions,
    _EXPECTED_INPUTS,
    _is_sha256,
    blend_reliability_checks,
    performance_order,
)
from .capturable_blend import _authenticated_recorded_revision_identity
from .capturable_candidate_selection import _selection_report
from .capturable_fixed_schedule import (
    EXPECTED_FIXED_ASSIGNMENTS,
    EXPECTED_FIXED_ASSIGNMENTS_BY_GAME_ID,
    SCHEDULE_SHA256,
)
from .capturable_records import (
    CAPTURABLE_RULE_IDS,
    CAPTURABLE_RULE_INDEX,
    CapturableDatasetError,
    CapturableDatasetRow,
)


FIXED_BLEND_FORMAT = "drawbackguesser-capturable-fixed-blend-confirmation"
FIXED_BLEND_VERSION = 1
FIXED_TREATMENT_WEIGHT = 0.1
FIXED_PROTOCOL_COMMIT = "2a3f68e69e22db45142a2fcd08121843b44a6660"
FIXED_PROTOCOL_FILE = "capturable-25-fixed-blend-confirmation-protocol.md"
FIXED_PROTOCOL_SHA256 = (
    "1d6bdcdc27778970c9b12ee30202cbdbd48e9a026ed3a6ac7053b388651bc3f8"
)
GRID_FILE = "capturable25-v3-convex-blend-validation.json"
GRID_SHA256 = (
    "e7721460b8ab94e2bd4e8ee293efdc36e25d49c92c2f9660dae3dba532ad6375"
)
GRID_EXECUTION_REVISION = "02ee847f9d6791a5eb09a281026ce537f33e922c"
FIXED_VALIDATION_PREDICTIONS_SHA256 = (
    "86299d8cb6c79973f7e675d495254a7a7f265a2bcf218758d46cfd048b7f223b"
)
PRIOR_REGISTRY_FILE = "capturable25-prior-corpus-registry-v1.json"
PRIOR_REGISTRY_SHA256 = (
    "af97da7cf0e790fc50747898141d348cf016855e6158e2bc1cd6a835c66aa1a0"
)
PRIOR_REGISTRY_SOURCE_COUNT = 54
PRIOR_REGISTRY_GAME_COUNT = 9_037
CONFIRMATION_TRACE_FILE = (
    "capturable25-v4-fixed-blend-confirmation-trace.ndjson"
)
CONFIRMATION_TEST_FILE = (
    "capturable25-v4-fixed-blend-confirmation-schema8.ndjson"
)
CORPUS_RECEIPT_FILE = (
    "capturable25-v4-fixed-blend-confirmation-corpus-receipt.json"
)
CONSUMPTION_FILE = (
    "capturable25-v4-fixed-blend-confirmation-consumption.json"
)
REPORT_PREFIX = "capturable25-v4-fixed-blend-confirmation-report-"

BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 633_454_611
BOOTSTRAP_DRAWS_PER_REPLICATE = 625
BOOTSTRAP_LOWER_INDEX = 999
MINIMUM_OBSERVED_TOP1_DELTA = 0.001
SAMPLER_ID = "splitmix64-rejection/v1"

CONSUMPTION_FORMAT = "drawbackguesser-capturable-fixed-blend-consumption"
CONSUMPTION_VERSION = 1
CORPUS_RECEIPT_FORMAT = (
    "drawbackguesser-capturable-fixed-confirmation-corpus-receipt"
)
CORPUS_RECEIPT_VERSION = 1

_UINT64_RANGE = 1 << 64
_UINT64_MASK = _UINT64_RANGE - 1
_SPLITMIX_INCREMENT = 0x9E3779B97F4A7C15
_SPLITMIX_MULTIPLIER_ONE = 0xBF58476D1CE4E5B9
_SPLITMIX_MULTIPLIER_TWO = 0x94D049BB133111EB

GENERATION_SCHEDULE = {
    "engineCommit": "74eb6fc95571994bd96b7a351278f3f74f0972e3",
    "evaluatorId": "drawback-material/v1",
    "gameplaySeed": 633_446_417,
    "labelSeed": 633_442_320,
    "leafCacheEntries": 16_384,
    "leafCacheHistoryMode": "full",
    "maxDepth": 1,
    "maxNodes": 5_000,
    "maxPlies": 60,
    "opponentAggregation": "worst-case",
    "opponentHypotheses": {
        "kind": "unrestricted-baseline",
        "version": 1,
    },
    "parameterSeed": 633_450_514,
    "policyId": "material-player-private-corpus/v1",
    "profile": "standard",
    "scheduleSha256": SCHEDULE_SHA256,
    "splitCounts": {"test": 625, "train": 0, "validation": 0},
    "temperatureCp": 35,
    "topK": 8,
    "traceFile": CONFIRMATION_TRACE_FILE,
    "window": 30,
    "workers": 15,
}


@dataclass(frozen=True)
class PairedTop1:
    game_ids: tuple[str, ...]
    deltas: tuple[float, ...]
    observed_delta: float


@dataclass(frozen=True)
class BootstrapResult:
    lower_bound: float
    rejected_draws: int


def fixed_validation_candidate(
    grid: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Authenticate the sole fixed hypothesis inside the rejected grid."""

    execution = grid.get("execution")
    candidates = grid.get("candidates")
    control = grid.get("control")
    if (
        not isinstance(execution, Mapping)
        or execution.get("revision") != GRID_EXECUTION_REVISION
        or grid.get("releaseDecision") != "retain-control"
        or not isinstance(grid.get("selected"), Mapping)
        or grid["selected"].get("weight") != 0.7
        or not isinstance(candidates, list)
        or not isinstance(control, Mapping)
        or not isinstance(control.get("metrics"), Mapping)
    ):
        raise CapturableDatasetError(
            "fixed confirmation grid identity is invalid"
        )
    matches = [
        candidate
        for candidate in candidates
        if isinstance(candidate, Mapping)
        and candidate.get("weight") == FIXED_TREATMENT_WEIGHT
    ]
    if len(matches) != 1:
        raise CapturableDatasetError(
            "fixed confirmation grid has no unique weight-0.1 candidate"
        )
    candidate = matches[0]
    metrics = candidate.get("metrics")
    if (
        candidate.get("predictionsSha256")
        != FIXED_VALIDATION_PREDICTIONS_SHA256
        or not isinstance(metrics, Mapping)
    ):
        raise CapturableDatasetError(
            "fixed validation candidate is not the frozen hypothesis"
        )
    primary = performance_order(metrics) > performance_order(
        control["metrics"]
    )
    checks = blend_reliability_checks(
        control["metrics"],
        metrics,
        primary,
    )
    if not all(checks.values()):
        raise CapturableDatasetError(
            "fixed validation candidate did not pass the complete gate"
        )
    return candidate


def audit_confirmation_rows(
    rows: Sequence[CapturableDatasetRow],
    prior_game_ids: set[str],
) -> Mapping[str, Any]:
    """Require the frozen balanced schedule and complete two-color trajectories."""

    if not rows:
        raise CapturableDatasetError("confirmation corpus has no rows")
    labels: dict[tuple[str, str], str] = {}
    seeds: dict[str, int] = {}
    colors: dict[str, set[str]] = {}
    results: dict[str, Any] = {}
    plies: dict[str, list[int]] = {}
    encountered_game_ids: list[str] = []
    closed_game_ids: set[str] = set()
    current_game_id: str | None = None
    for row in rows:
        game_id = row.evaluation.game_id
        color = row.features.player_color
        expected = EXPECTED_FIXED_ASSIGNMENTS_BY_GAME_ID.get(game_id)
        if expected is None:
            raise CapturableDatasetError(
                f"confirmation game ID is outside the frozen schedule: {game_id}"
            )
        if game_id != current_game_id:
            if current_game_id is not None:
                closed_game_ids.add(current_game_id)
            if game_id in closed_game_ids:
                raise CapturableDatasetError(
                    f"confirmation rows interleave game {game_id}"
                )
            encountered_game_ids.append(game_id)
            current_game_id = game_id
        game_plies = plies.setdefault(game_id, [])
        if row.features.ply != len(game_plies):
            raise CapturableDatasetError(
                f"confirmation ply trajectory is not consecutive in {game_id}"
            )
        expected_color = (
            "white" if row.features.ply % 2 == 0 else "black"
        )
        if color != expected_color:
            raise CapturableDatasetError(
                f"confirmation color does not alternate in {game_id}"
            )
        if row.features.move_number != row.features.ply // 2 + 1:
            raise CapturableDatasetError(
                f"confirmation move number disagrees with ply in {game_id}"
            )
        if (
            row.evaluation.seed != expected.gameplay_seed
            or row.evaluation.bot_agent_id
            != "material-player-private-corpus/v1"
            or row.evaluation.bot_style != "drawback-search"
            or row.evaluation.bot_strength is not None
        ):
            raise CapturableDatasetError(
                f"confirmation provenance disagrees with schedule in {game_id}"
            )
        expected_label = (
            expected.white_rule_id
            if color == "white"
            else expected.black_rule_id
        )
        if row.labels.true_drawback != expected_label:
            raise CapturableDatasetError(
                f"confirmation label disagrees with schedule in {game_id}/{color}"
            )
        game_plies.append(row.features.ply)
        key = (game_id, color)
        prior_label = labels.setdefault(key, row.labels.true_drawback)
        if prior_label != row.labels.true_drawback:
            raise CapturableDatasetError(
                f"confirmation label changed inside {game_id}/{color}"
            )
        prior_seed = seeds.setdefault(game_id, row.evaluation.seed)
        if prior_seed != row.evaluation.seed:
            raise CapturableDatasetError(
                f"confirmation seed changed inside {game_id}"
            )
        colors.setdefault(game_id, set()).add(color)
        if game_id not in results:
            results[game_id] = row.labels.result
        elif results[game_id] != row.labels.result:
            raise CapturableDatasetError(
                f"confirmation result changed inside {game_id}"
            )

    expected_game_ids = [
        assignment.game_id for assignment in EXPECTED_FIXED_ASSIGNMENTS
    ]
    if encountered_game_ids != expected_game_ids:
        raise CapturableDatasetError(
            "confirmation game order does not match the frozen schedule"
        )
    game_ids = sorted(colors)
    overlap = sorted(set(game_ids) & prior_game_ids)
    if overlap:
        raise CapturableDatasetError(
            "confirmation corpus overlaps prior corpus registry: "
            + ", ".join(overlap[:5])
        )
    if (
        len(game_ids) != BOOTSTRAP_DRAWS_PER_REPLICATE
        or any(colors[game_id] != {"white", "black"} for game_id in game_ids)
        or len(labels) != 2 * BOOTSTRAP_DRAWS_PER_REPLICATE
    ):
        raise CapturableDatasetError(
            "confirmation corpus must contain 625 two-color games"
        )
    active_at_limit = 0
    for game_id in game_ids:
        result = results[game_id]
        if not isinstance(result, Mapping) or not isinstance(
            result.get("kind"),
            str,
        ):
            raise CapturableDatasetError(
                f"confirmation result is invalid in {game_id}"
            )
        ply_count = len(plies[game_id])
        if not 2 <= ply_count <= GENERATION_SCHEDULE["maxPlies"]:
            raise CapturableDatasetError(
                f"confirmation row count is outside the ply limit in {game_id}"
            )
        if result["kind"] == "active":
            if ply_count != GENERATION_SCHEDULE["maxPlies"]:
                raise CapturableDatasetError(
                    f"active confirmation game is not censored at the limit in {game_id}"
                )
            active_at_limit += 1
    pairs = Counter(
        (
            labels[(game_id, "white")],
            labels[(game_id, "black")],
        )
        for game_id in game_ids
    )
    expected_pairs = {
        (white, black)
        for white in CAPTURABLE_RULE_IDS
        for black in CAPTURABLE_RULE_IDS
    }
    if set(pairs) != expected_pairs or any(count != 1 for count in pairs.values()):
        raise CapturableDatasetError(
            "confirmation corpus does not contain each ordered pair once"
        )
    marginals = Counter(
        (color, drawback_id)
        for (_game_id, color), drawback_id in labels.items()
    )
    if any(
        marginals[(color, drawback_id)] != 25
        for color in ("white", "black")
        for drawback_id in CAPTURABLE_RULE_IDS
    ):
        raise CapturableDatasetError(
            "confirmation corpus label/color marginals are not balanced"
        )
    return {
        "games": len(game_ids),
        "orderedPairs": len(pairs),
        "playerGames": len(labels),
        "rows": len(rows),
        "stoppedAtPlyLimit": active_at_limit,
        "terminalGames": len(game_ids) - active_at_limit,
    }


def _prediction_top1_credit(
    row: CapturableDatasetRow,
    probabilities: Sequence[float],
) -> float:
    if len(probabilities) != len(CAPTURABLE_RULE_IDS):
        raise CapturableDatasetError(
            "fixed confirmation posterior dimension is invalid"
        )
    return _top_k_vector_credit(
        probabilities,
        CAPTURABLE_RULE_INDEX[row.labels.true_drawback],
        1,
    )


def paired_game_top1_deltas(
    rows: Sequence[CapturableDatasetRow],
    control: ComponentPredictions,
    fixed: ComponentPredictions,
) -> PairedTop1:
    """Build one paired Top-1 delta per physical game."""

    if (
        not rows
        or len(rows) != len(control.drawback)
        or len(rows) != len(fixed.drawback)
    ):
        raise CapturableDatasetError(
            "paired Top-1 rows and predictions do not align"
        )
    credits: dict[
        tuple[str, str],
        dict[str, list[float]],
    ] = {}
    for row, control_probabilities, fixed_probabilities in zip(
        rows,
        control.drawback,
        fixed.drawback,
        strict=True,
    ):
        key = (row.evaluation.game_id, row.features.player_color)
        entry = credits.setdefault(
            key,
            {"control": [], "fixed": []},
        )
        entry["control"].append(
            _prediction_top1_credit(row, control_probabilities)
        )
        entry["fixed"].append(
            _prediction_top1_credit(row, fixed_probabilities)
        )
    player_values: dict[tuple[str, str], tuple[float, float]] = {}
    for key, values in credits.items():
        count = len(values["control"])
        if count == 0 or len(values["fixed"]) != count:
            raise CapturableDatasetError(
                "paired Top-1 player-game rows are incomplete"
            )
        player_values[key] = (
            math.fsum(values["control"]) / count,
            math.fsum(values["fixed"]) / count,
        )
    game_ids = tuple(sorted({game_id for game_id, _color in player_values}))
    if (
        len(game_ids) != BOOTSTRAP_DRAWS_PER_REPLICATE
        or any(
            (game_id, color) not in player_values
            for game_id in game_ids
            for color in ("white", "black")
        )
    ):
        raise CapturableDatasetError(
            "paired Top-1 requires both colors in exactly 625 games"
        )
    deltas = tuple(
        math.fsum(
            (
                player_values[(game_id, "white")][1]
                - player_values[(game_id, "white")][0],
                player_values[(game_id, "black")][1]
                - player_values[(game_id, "black")][0],
            )
        )
        / 2.0
        for game_id in game_ids
    )
    return PairedTop1(
        game_ids=game_ids,
        deltas=deltas,
        observed_delta=math.fsum(deltas) / len(deltas),
    )


def _splitmix64_next(state: int) -> tuple[int, int]:
    state = (state + _SPLITMIX_INCREMENT) & _UINT64_MASK
    value = state
    value = (
        (value ^ (value >> 30)) * _SPLITMIX_MULTIPLIER_ONE
    ) & _UINT64_MASK
    value = (
        (value ^ (value >> 27)) * _SPLITMIX_MULTIPLIER_TWO
    ) & _UINT64_MASK
    return state, value ^ (value >> 31)


def _rejection_index(
    state: int,
    size: int,
) -> tuple[int, int, int]:
    if size <= 0:
        raise ValueError("sampler size must be positive")
    rejected = 0
    while True:
        state, value = _splitmix64_next(state)
        index = _reduce_to_index(value, size)
        if index is not None:
            return state, index, rejected
        rejected += 1


def _reduce_to_index(value: int, size: int) -> int | None:
    if (
        size <= 0
        or value < 0
        or value >= _UINT64_RANGE
    ):
        raise ValueError("sampler reduction inputs are invalid")
    limit = _UINT64_RANGE - (_UINT64_RANGE % size)
    return value % size if value < limit else None


def fixed_paired_bootstrap(deltas: Sequence[float]) -> BootstrapResult:
    """Run the exact preregistered whole-game percentile bootstrap."""

    if (
        len(deltas) != BOOTSTRAP_DRAWS_PER_REPLICATE
        or any(not math.isfinite(value) for value in deltas)
    ):
        raise CapturableDatasetError(
            "fixed bootstrap requires 625 finite game deltas"
        )
    state = BOOTSTRAP_SEED
    rejected_draws = 0
    replicate_means: list[float] = []
    for _replicate in range(BOOTSTRAP_REPLICATES):
        sampled: list[float] = []
        for _draw in range(BOOTSTRAP_DRAWS_PER_REPLICATE):
            state, index, rejected = _rejection_index(
                state,
                len(deltas),
            )
            rejected_draws += rejected
            sampled.append(deltas[index])
        replicate_means.append(
            math.fsum(sampled) / BOOTSTRAP_DRAWS_PER_REPLICATE
        )
    replicate_means.sort()
    return BootstrapResult(
        lower_bound=replicate_means[BOOTSTRAP_LOWER_INDEX],
        rejected_draws=rejected_draws,
    )


def fixed_release_checks(
    control: Mapping[str, Any],
    fixed: Mapping[str, Any],
    paired: PairedTop1,
    bootstrap: BootstrapResult,
) -> Mapping[str, bool]:
    """Combine the complete reliability and paired uncertainty gates."""

    primary = performance_order(fixed) > performance_order(control)
    checks = dict(blend_reliability_checks(control, fixed, primary))
    control_hybrid = control.get("hybrid")
    fixed_hybrid = fixed.get("hybrid")
    if not isinstance(control_hybrid, Mapping) or not isinstance(
        fixed_hybrid,
        Mapping,
    ):
        raise CapturableDatasetError(
            "fixed confirmation hybrid metrics are invalid"
        )
    metric_delta = (
        float(fixed_hybrid["game_normalized_top_1_accuracy"])
        - float(control_hybrid["game_normalized_top_1_accuracy"])
    )
    checks["pairedTop1MetricAgreement"] = math.isclose(
        paired.observed_delta,
        metric_delta,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    checks["pairedTop1MinimumObservedGain"] = (
        paired.observed_delta >= MINIMUM_OBSERVED_TOP1_DELTA
    )
    checks["pairedTop1BootstrapLowerPositive"] = (
        bootstrap.lower_bound > 0.0
    )
    return checks


def _validate_metric_entry(
    entry: object,
    label: str,
) -> tuple[Mapping[str, Any], str]:
    if (
        not isinstance(entry, Mapping)
        or set(entry) != {"metrics", "predictionsSha256"}
        or not isinstance(entry.get("metrics"), Mapping)
        or not _is_sha256(entry.get("predictionsSha256"))
    ):
        raise CapturableDatasetError(
            f"fixed confirmation {label} metrics are invalid"
        )
    return entry["metrics"], str(entry["predictionsSha256"])


def _is_revision(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(token in "0123456789abcdef" for token in value)
    )


def fixed_expected_inputs() -> Mapping[str, Any]:
    return {
        "control": dict(_EXPECTED_INPUTS["control"]),
        "grid": {
            "executionRevision": GRID_EXECUTION_REVISION,
            "file": GRID_FILE,
            "sha256": GRID_SHA256,
        },
        "registry": {
            "file": PRIOR_REGISTRY_FILE,
            "games": PRIOR_REGISTRY_GAME_COUNT,
            "sha256": PRIOR_REGISTRY_SHA256,
            "sources": PRIOR_REGISTRY_SOURCE_COUNT,
        },
        "treatment": dict(_EXPECTED_INPUTS["treatment"]),
    }


def fixed_consumption_artifact(
    *,
    execution: Mapping[str, Any],
    inputs: Mapping[str, Any],
    receipt_sha256: str,
) -> Mapping[str, Any]:
    if inputs != fixed_expected_inputs() or not _is_sha256(receipt_sha256):
        raise CapturableDatasetError(
            "fixed consumption inputs are invalid"
        )
    return {
        "corpusReceipt": {
            "file": CORPUS_RECEIPT_FILE,
            "sha256": receipt_sha256,
        },
        "execution": execution,
        "fixedTreatmentWeight": FIXED_TREATMENT_WEIGHT,
        "format": CONSUMPTION_FORMAT,
        "inputs": inputs,
        "protocol": {
            "commit": FIXED_PROTOCOL_COMMIT,
            "file": FIXED_PROTOCOL_FILE,
            "sha256": FIXED_PROTOCOL_SHA256,
        },
        "schedule": GENERATION_SCHEDULE,
        "state": "consumed",
        "testFile": CONFIRMATION_TEST_FILE,
        "traceFile": CONFIRMATION_TRACE_FILE,
        "version": CONSUMPTION_VERSION,
    }


def load_fixed_blend_confirmation(
    path: Path,
) -> tuple[Mapping[str, Any], str]:
    """Load a canonical report and recompute every promotion decision."""

    from .capturable_fixed_corpus import (
        load_fixed_corpus_receipt,
        require_private_regular_file,
        require_private_root,
    )

    root = require_private_root(path.absolute().parent)
    resolved_report = require_private_regular_file(
        root,
        path,
        path.name,
        "fixed confirmation report",
    )
    artifact, sha256 = _selection_report(resolved_report)
    expected_keys = {
        "consumption",
        "control",
        "corpusReceipt",
        "execution",
        "fixedBlend",
        "fixedTreatmentWeight",
        "format",
        "inputs",
        "pairedTop1",
        "primaryDecision",
        "protocol",
        "releaseDecision",
        "reliabilityChecks",
        "sealedTestStatus",
        "test",
        "trace",
        "version",
    }
    if (
        set(artifact) != expected_keys
        or artifact.get("format") != FIXED_BLEND_FORMAT
        or type(artifact.get("version")) is not int
        or artifact.get("version") != FIXED_BLEND_VERSION
        or artifact.get("fixedTreatmentWeight") != FIXED_TREATMENT_WEIGHT
        or artifact.get("sealedTestStatus") != "consumed"
        or artifact.get("protocol")
        != {
            "commit": FIXED_PROTOCOL_COMMIT,
            "file": FIXED_PROTOCOL_FILE,
            "sha256": FIXED_PROTOCOL_SHA256,
        }
    ):
        raise CapturableDatasetError(
            f"{path.name} is not a compatible fixed confirmation"
        )
    execution = artifact.get("execution")
    consumption = artifact.get("consumption")
    corpus_receipt = artifact.get("corpusReceipt")
    trace = artifact.get("trace")
    test = artifact.get("test")
    inputs = artifact.get("inputs")
    if (
        not isinstance(execution, Mapping)
        or set(execution) != {"cleanWorktree", "repository", "revision"}
        or execution.get("cleanWorktree") is not True
        or execution.get("repository") != "DrawbackGuesser"
        or not _is_revision(execution.get("revision"))
        or not isinstance(consumption, Mapping)
        or set(consumption) != {"file", "sha256"}
        or consumption.get("file") != CONSUMPTION_FILE
        or not _is_sha256(consumption.get("sha256"))
        or not isinstance(corpus_receipt, Mapping)
        or set(corpus_receipt) != {"file", "sha256"}
        or corpus_receipt.get("file") != CORPUS_RECEIPT_FILE
        or not _is_sha256(corpus_receipt.get("sha256"))
        or not isinstance(trace, Mapping)
        or set(trace) != {"file", "sha256"}
        or trace.get("file") != CONFIRMATION_TRACE_FILE
        or not _is_sha256(trace.get("sha256"))
        or not isinstance(test, Mapping)
        or set(test) != {"file", "games", "rows", "sha256"}
        or test.get("file") != CONFIRMATION_TEST_FILE
        or test.get("games") != BOOTSTRAP_DRAWS_PER_REPLICATE
        or isinstance(test.get("rows"), bool)
        or not isinstance(test.get("rows"), int)
        or test["rows"] <= 0
        or not _is_sha256(test.get("sha256"))
        or not isinstance(inputs, Mapping)
    ):
        raise CapturableDatasetError(
            f"{path.name} fixed confirmation identity is invalid"
        )
    expected_inputs = fixed_expected_inputs()
    if inputs != expected_inputs:
        raise CapturableDatasetError(
            f"{path.name} fixed confirmation inputs are invalid"
        )
    control_metrics, _control_sha = _validate_metric_entry(
        artifact.get("control"),
        "control",
    )
    fixed_metrics, _fixed_sha = _validate_metric_entry(
        artifact.get("fixedBlend"),
        "blend",
    )
    paired = artifact.get("pairedTop1")
    if not isinstance(paired, Mapping) or set(paired) != {
        "bootstrapLowerBound",
        "bootstrapReplicates",
        "drawsPerReplicate",
        "gameDeltas",
        "minimumObservedDelta",
        "observedDelta",
        "rejectedDraws",
        "sampler",
        "seed",
    }:
        raise CapturableDatasetError(
            f"{path.name} paired Top-1 evidence is invalid"
        )
    entries = paired.get("gameDeltas")
    if (
        paired.get("bootstrapReplicates") != BOOTSTRAP_REPLICATES
        or paired.get("drawsPerReplicate")
        != BOOTSTRAP_DRAWS_PER_REPLICATE
        or paired.get("minimumObservedDelta")
        != MINIMUM_OBSERVED_TOP1_DELTA
        or paired.get("sampler") != SAMPLER_ID
        or paired.get("seed") != BOOTSTRAP_SEED
        or isinstance(paired.get("observedDelta"), bool)
        or not isinstance(paired.get("observedDelta"), (int, float))
        or not math.isfinite(float(paired["observedDelta"]))
        or isinstance(paired.get("bootstrapLowerBound"), bool)
        or not isinstance(
            paired.get("bootstrapLowerBound"),
            (int, float),
        )
        or not math.isfinite(float(paired["bootstrapLowerBound"]))
        or isinstance(paired.get("rejectedDraws"), bool)
        or not isinstance(paired.get("rejectedDraws"), int)
        or paired["rejectedDraws"] < 0
        or not isinstance(entries, list)
        or len(entries) != BOOTSTRAP_DRAWS_PER_REPLICATE
    ):
        raise CapturableDatasetError(
            f"{path.name} paired Top-1 contract is invalid"
        )
    game_ids: list[str] = []
    deltas: list[float] = []
    for entry in entries:
        if (
            not isinstance(entry, Mapping)
            or set(entry) != {"delta", "gameId"}
            or not isinstance(entry.get("gameId"), str)
            or not entry["gameId"]
            or isinstance(entry.get("delta"), bool)
            or not isinstance(entry.get("delta"), (int, float))
            or not math.isfinite(float(entry["delta"]))
        ):
            raise CapturableDatasetError(
                f"{path.name} game-delta entry is invalid"
            )
        game_ids.append(entry["gameId"])
        deltas.append(float(entry["delta"]))
    if game_ids != sorted(set(game_ids)):
        raise CapturableDatasetError(
            f"{path.name} game-delta entries are not ordered and unique"
        )
    if game_ids != sorted(
        assignment.game_id for assignment in EXPECTED_FIXED_ASSIGNMENTS
    ):
        raise CapturableDatasetError(
            f"{path.name} game-delta IDs are outside the frozen corpus"
        )
    observed = math.fsum(deltas) / len(deltas)
    bootstrap = fixed_paired_bootstrap(deltas)
    if (
        paired.get("observedDelta") != observed
        or paired.get("bootstrapLowerBound") != bootstrap.lower_bound
        or paired.get("rejectedDraws") != bootstrap.rejected_draws
    ):
        raise CapturableDatasetError(
            f"{path.name} paired Top-1 evidence is inconsistent"
        )
    paired_value = PairedTop1(
        game_ids=tuple(game_ids),
        deltas=tuple(deltas),
        observed_delta=observed,
    )
    checks = fixed_release_checks(
        control_metrics,
        fixed_metrics,
        paired_value,
        bootstrap,
    )
    primary = performance_order(fixed_metrics) > performance_order(
        control_metrics
    )
    release = "promote-fixed-blend" if all(checks.values()) else "retain-control"
    if (
        artifact.get("primaryDecision")
        != ("confirm-fixed-blend" if primary else "reject-fixed-blend")
        or artifact.get("reliabilityChecks") != checks
        or artifact.get("releaseDecision") != release
    ):
        raise CapturableDatasetError(
            f"{path.name} fixed confirmation decision is inconsistent"
        )
    receipt_path = require_private_regular_file(
        root,
        root / CORPUS_RECEIPT_FILE,
        CORPUS_RECEIPT_FILE,
        "fixed corpus receipt",
    )
    marker_path = require_private_regular_file(
        root,
        root / CONSUMPTION_FILE,
        CONSUMPTION_FILE,
        "fixed consumption marker",
    )
    receipt, receipt_sha256 = load_fixed_corpus_receipt(
        receipt_path,
        execution,
    )
    marker, marker_sha256 = _selection_report(marker_path)
    if (
        receipt_sha256 != corpus_receipt["sha256"]
        or trace
        != {
            "file": CONFIRMATION_TRACE_FILE,
            "sha256": receipt["trace"]["sha256"],
        }
        or test["sha256"] != receipt["dataset"]["sha256"]
        or test["rows"] != receipt["dataset"]["rows"]
        or consumption.get("sha256") != marker_sha256
        or marker != fixed_consumption_artifact(
            execution=execution,
            inputs=inputs,
            receipt_sha256=receipt_sha256,
        )
    ):
        raise CapturableDatasetError(
            f"{path.name} corpus or consumption evidence is inconsistent"
        )
    if resolved_report.name != f"{REPORT_PREFIX}{sha256}.json":
        raise CapturableDatasetError(
            f"{path.name} is not named by its report digest"
        )
    _authenticated_recorded_revision_identity(
        revision=str(execution["revision"]),
        protocol_commit=FIXED_PROTOCOL_COMMIT,
        protocol_file=FIXED_PROTOCOL_FILE,
        protocol_sha256=FIXED_PROTOCOL_SHA256,
        operation="fixed confirmation report",
    )
    return artifact, sha256
