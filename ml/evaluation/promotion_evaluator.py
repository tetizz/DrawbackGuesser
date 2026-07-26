"""One-pass scoring core for the frozen current-catalog promotion candidate.

This module deliberately does not authorize opening the sealed test split and
does not publish reports.  It evaluates an already-authorized, authenticated
lease and returns deterministic in-memory evidence that a future publication
boundary can bind.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import BinaryIO, Iterable, Iterator, Mapping, Protocol, Sequence

from ml.training.drawback_ml.corpus_contract import AuditedPrivateCorpusLease
from ml.training.drawback_ml.ensemble import (
    PROTOCOL_V2_ENSEMBLE_SEEDS,
    load_hybrid_ensemble,
)
from ml.training.drawback_ml.inference import InferenceOutput
from ml.training.drawback_ml.records import (
    DatasetSchemaError,
    FeatureRecord,
    TrainingExample,
    group_training_examples,
    parse_dataset_row,
)
from ml.training.drawback_ml.rank_preserving_fusion import (
    RANK_PRESERVING_FUSION_METHOD,
    RankPreservingFusionError,
    RankPreservingFusionResult,
    rank_preserving_fusion,
)
from ml.training.drawback_ml.symbolic_schema import SYMBOLIC_RULE_IDS

from .calibration import masked_temperature_softmax
from .ensemble_calibration import (
    ContentAddressedFile,
    load_ensemble_calibration,
)
from .ensemble_release import (
    LoadedEnsembleRelease,
    resolve_member_checkpoint,
    verify_ensemble_release,
)
from .metrics import EvaluationReport, PredictionExample, StreamingEvaluation
from .release_selection_bundle import ContentAddressedJson
from .runner import (
    EvaluationDataError,
    decode_predicted_parameter,
    decode_true_parameter,
    load_rule_families,
    read_ndjson_stream,
)
from .training_frequency import (
    ContentAddressedFile as TrainingFrequencyReference,
    load_training_frequency_artifact,
)
from .validation_partition import (
    ValidationPartition,
    assign_validation_partition,
    validation_seed_sha256,
)


PROMOTION_REPORT_FORMAT = "drawbacktrainer-promotion-report"
PROMOTION_REPORT_VERSION = 2
PREPARED_VIEW = "prepared-182"
BROWSER_VIEW = "browser-180"
UNAVAILABLE_BROWSER_RULE_IDS = frozenset(
    {"hand-and-gigabrain", "ichtyophobe"}
)
SYSTEM_NAMES = (
    "uniform",
    "training-frequency",
    "symbolic-only",
    "member-20260811",
    "member-20260812",
    "member-20260813",
    "uncalibrated-ensemble",
    "calibrated-ensemble",
)
_TRANSCRIPT_DOMAIN = b"drawbacktrainer-promotion-inference-transcript-v2\x00"


class FrozenMemberPredictor(Protocol):
    drawback_vocabulary: Sequence[str]
    parameter_vocabulary: Sequence[str]
    checkpoint_seed: int

    def predict_batch(
        self, features: Sequence[FeatureRecord]
    ) -> Sequence[InferenceOutput]: ...


@dataclass(frozen=True)
class PromotionTemperatures:
    white: float
    black: float

    def __post_init__(self) -> None:
        for name, value in (("white", self.white), ("black", self.black)):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} temperature must be finite and positive")
            object.__setattr__(self, name, float(value))


@dataclass(frozen=True)
class SystemPredictionRow:
    """Predictions for one public prefix; truth is evaluation metadata only."""

    game_id: str
    seed: int
    color: str
    observed_ply: int
    truth: str
    hard_eliminated: tuple[bool, ...]
    member_residual_logits: tuple[
        tuple[float, ...], tuple[float, ...], tuple[float, ...]
    ]
    ensemble_fused_logits: tuple[float, ...]
    true_parameter_token: str | None
    parameter_observed: bool
    true_parameters: Mapping[str, object] | None
    predicted_parameters: Mapping[str, object] | None
    parameter_unscorable_reason: str | None
    probabilities: Mapping[str, tuple[float, ...]]
    rule_triggered: bool = False
    bot_agent_id: str | None = None
    bot_style: str | None = None
    bot_strength: int | None = None
    agent_metadata_present: bool = False
    evaluator_backed: bool = False

    def __post_init__(self) -> None:
        if not self.game_id:
            raise ValueError("promotion row game_id must not be empty")
        if self.color not in {"white", "black"}:
            raise ValueError("promotion row color must be white or black")
        if self.observed_ply <= 0:
            raise ValueError("promotion row observed_ply must be positive")
        if type(self.rule_triggered) is not bool:
            raise ValueError("promotion row rule_triggered must be boolean")
        if type(self.agent_metadata_present) is not bool:
            raise ValueError(
                "promotion row agent_metadata_present must be boolean"
            )
        if type(self.evaluator_backed) is not bool:
            raise ValueError("promotion row evaluator_backed must be boolean")
        if self.agent_metadata_present:
            if not self.bot_agent_id:
                raise ValueError(
                    "complete agent metadata requires a non-empty agent ID"
                )
        elif any(
            value is not None
            for value in (
                self.bot_agent_id,
                self.bot_style,
                self.bot_strength,
            )
        ):
            raise ValueError(
                "absent agent metadata cannot carry partial values"
            )
        dimension = len(SYMBOLIC_RULE_IDS)
        if self.truth not in SYMBOLIC_RULE_IDS:
            raise ValueError("promotion row truth is outside the frozen catalog")
        if (
            len(self.hard_eliminated) != dimension
            or any(type(value) is not bool for value in self.hard_eliminated)
            or all(self.hard_eliminated)
        ):
            raise ValueError("promotion row has an invalid hard mask")
        if len(self.member_residual_logits) != 3 or any(
            len(row) != dimension
            or any(not math.isfinite(value) for value in row)
            for row in self.member_residual_logits
        ):
            raise ValueError("promotion row has invalid member residuals")
        if len(self.ensemble_fused_logits) != dimension or any(
            not math.isfinite(value) for value in self.ensemble_fused_logits
        ):
            raise ValueError("promotion row has invalid ensemble logits")
        if tuple(self.probabilities) != SYSTEM_NAMES:
            raise ValueError("promotion row systems are missing or reordered")
        if self.true_parameter_token is None:
            if (
                self.true_parameters is not None
                or self.predicted_parameters is not None
                or self.parameter_unscorable_reason is not None
            ):
                raise ValueError(
                    "unparameterized rows cannot contain parameter scoring data"
                )
        elif self.parameter_unscorable_reason is None:
            if self.true_parameters is None:
                raise ValueError(
                    "scorable parameter rows require decoded true values"
                )
        elif self.true_parameters is not None or self.predicted_parameters is not None:
            raise ValueError(
                "unscorable parameter rows cannot contain decoded values"
            )
        for name in ("true_parameters", "predicted_parameters"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    MappingProxyType(dict(value)),
                )
        normalized: dict[str, tuple[float, ...]] = {}
        for system, values in self.probabilities.items():
            row = tuple(values)
            _validate_distribution(row, self.hard_eliminated, system)
            normalized[system] = row
        object.__setattr__(
            self,
            "probabilities",
            MappingProxyType(normalized),
        )


@dataclass(frozen=True)
class InferenceTranscript:
    algorithm: str
    sha256: str
    record_count: int
    first_record: tuple[str, str, int] | None
    last_record: tuple[str, str, int] | None


@dataclass(frozen=True)
class PlayerGameSummary:
    """Sufficient per-player-game statistics for paired bootstrap resampling."""

    view: str
    system: str
    game_id: str
    seed: int
    color: str
    example_count: int
    top_1_sum: float
    top_3_sum: float
    top_5_sum: float
    nll_sum: float
    brier_sum: float


@dataclass(frozen=True)
class PromotionSystemReport:
    white: EvaluationReport | None
    black: EvaluationReport | None


@dataclass(frozen=True)
class ParameterEvaluationReport:
    eligible_examples: int
    scorable_examples: int
    unscorable_examples: int
    coverage: float | None
    unscorable_rate: float | None
    whole_object_accuracy: float | None
    component_accuracy: float | None
    component_count: int
    unscorable_by_reason: Mapping[str, int]


@dataclass(frozen=True)
class RuleOpportunityReport:
    support_examples: int
    support_player_games: int
    trigger_opportunities: int
    unscorable_examples: int


@dataclass(frozen=True)
class AgentProfileReport:
    agent_id: str
    style: str | None
    strength: int | None
    example_count: int
    white: EvaluationReport | None
    black: EvaluationReport | None


@dataclass(frozen=True)
class EvaluatorModeReport:
    mode: str
    example_count: int
    white: EvaluationReport | None
    black: EvaluationReport | None


@dataclass(frozen=True)
class PromotionViewReport:
    name: str
    class_ids: tuple[str, ...]
    unavailable_rule_ids: tuple[str, ...]
    scored_examples: int
    unscorable_examples: int
    unscorable_player_games: int
    systems: Mapping[str, PromotionSystemReport]
    calibrated_parameters: Mapping[str, ParameterEvaluationReport]
    rule_opportunities: Mapping[str, Mapping[str, RuleOpportunityReport]]
    agent_profiles: Mapping[str, AgentProfileReport]
    evaluator_modes: Mapping[str, EvaluatorModeReport]
    evaluation_slices_complete: bool
    unsupported_protocol_slices: tuple[str, ...]


@dataclass(frozen=True)
class PromotionReport:
    format: str
    version: int
    partition: str
    partition_seed_sha256: str
    bootstrap_seed: int
    ensemble_release_sha256: str
    calibration_sha256: str
    training_frequency_sha256: str
    move_examples: int
    promotion_gate_complete: bool
    unsupported_protocol_slices: tuple[str, ...]
    transcript: InferenceTranscript
    views: Mapping[str, PromotionViewReport]
    player_game_summaries: tuple[PlayerGameSummary, ...]


class _TranscriptBuilder:
    def __init__(
        self,
        partition: str,
        fusion_policy: tuple[str, float] | None = None,
    ) -> None:
        self._digest = hashlib.sha256()
        self._digest.update(_TRANSCRIPT_DOMAIN)
        fusion = (
            None
            if fusion_policy is None
            else {
                "method": RANK_PRESERVING_FUSION_METHOD,
                "selection_sha256": fusion_policy[0],
                "alpha": fusion_policy[1],
            }
        )
        self._append(
            {
                "partition": partition,
                "symbolic_rule_ids": list(SYMBOLIC_RULE_IDS),
                "systems": list(SYSTEM_NAMES),
                "fusion": fusion,
            }
        )
        self._count = 0
        self._first: tuple[str, str, int] | None = None
        self._last: tuple[str, str, int] | None = None

    def _append(self, value: object) -> None:
        payload = json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        self._digest.update(len(payload).to_bytes(8, "big"))
        self._digest.update(payload)

    def add(self, row: SystemPredictionRow) -> None:
        key = (row.game_id, row.color, row.observed_ply)
        if self._first is None:
            self._first = key
        self._last = key
        self._count += 1
        self._append(
            {
                "game_id": row.game_id,
                "seed": row.seed,
                "color": row.color,
                "observed_ply": row.observed_ply,
                "truth": row.truth,
                "rule_triggered": row.rule_triggered,
                "agent": {
                    "id": row.bot_agent_id,
                    "style": row.bot_style,
                    "strength": row.bot_strength,
                    "present": row.agent_metadata_present,
                },
                "evaluator_backed": row.evaluator_backed,
                "hard_eliminated": list(row.hard_eliminated),
                "member_residual_logits": [
                    list(values) for values in row.member_residual_logits
                ],
                "ensemble_fused_logits": list(row.ensemble_fused_logits),
                "calibrated_probabilities": list(
                    row.probabilities["calibrated-ensemble"]
                ),
                "parameter": {
                    "true_token": row.true_parameter_token,
                    "observed": row.parameter_observed,
                    "true": (
                        None
                        if row.true_parameters is None
                        else dict(row.true_parameters)
                    ),
                    "predicted": (
                        None
                        if row.predicted_parameters is None
                        else dict(row.predicted_parameters)
                    ),
                    "unscorable_reason": row.parameter_unscorable_reason,
                },
            }
        )

    def finish(self) -> InferenceTranscript:
        return InferenceTranscript(
            algorithm="sha256-domain-separated-canonical-json-v1",
            sha256=self._digest.hexdigest(),
            record_count=self._count,
            first_record=self._first,
            last_record=self._last,
        )


class _ViewAccumulator:
    def __init__(
        self,
        name: str,
        class_ids: tuple[str, ...],
        unavailable: tuple[str, ...],
        rule_families: Mapping[str, str],
    ) -> None:
        self.name = name
        self.class_ids = class_ids
        self.unavailable = unavailable
        self._indices = tuple(SYMBOLIC_RULE_IDS.index(item) for item in class_ids)
        self._metrics = {
            system: {
                color: StreamingEvaluation(15)
                for color in ("white", "black")
            }
            for system in SYSTEM_NAMES
        }
        self._color_counts = {"white": 0, "black": 0}
        self._summaries: dict[
            tuple[str, str, int, str],
            list[float],
        ] = {}
        self._unscorable_examples = 0
        self._unscorable_games: set[tuple[str, str]] = set()
        self._scored_examples = 0
        self._families = rule_families
        self._parameter_totals = {
            color: {
                "eligible": 0,
                "scorable": 0,
                "whole_correct": 0,
                "component_correct": 0,
                "component_count": 0,
            }
            for color in ("white", "black")
        }
        self._parameter_reasons: dict[str, dict[str, int]] = {
            "white": {},
            "black": {},
        }
        self._rule_totals: dict[
            str,
            dict[str, dict[str, object]],
        ] = {
            color: {
                rule_id: {
                    "support": 0,
                    "games": set(),
                    "triggers": 0,
                    "unscorable": 0,
                }
                for rule_id in SYMBOLIC_RULE_IDS
            }
            for color in ("white", "black")
        }
        self._agent_contracts: dict[str, tuple[str | None, int | None]] = {}
        self._agent_metrics: dict[
            str, dict[str, StreamingEvaluation]
        ] = {}
        self._agent_counts: dict[str, int] = {}
        self._agent_color_counts: dict[str, dict[str, int]] = {}
        self._mode_metrics = {
            mode: {
                color: StreamingEvaluation(15)
                for color in ("white", "black")
            }
            for mode in ("evaluator-backed", "synchronous")
        }
        self._mode_counts = {
            mode: {"white": 0, "black": 0}
            for mode in ("evaluator-backed", "synchronous")
        }
        self._slice_issues: set[str] = set()

    def add(self, row: SystemPredictionRow) -> None:
        rule_totals = self._rule_totals[row.color][row.truth]
        rule_totals["triggers"] = int(rule_totals["triggers"]) + int(
            row.rule_triggered
        )
        if row.truth not in self.class_ids:
            self._unscorable_examples += 1
            self._unscorable_games.add((row.game_id, row.color))
            rule_totals["unscorable"] = int(
                rule_totals["unscorable"]
            ) + 1
            return
        rule_totals["support"] = int(rule_totals["support"]) + 1
        games = rule_totals["games"]
        if not isinstance(games, set):
            raise RuntimeError("rule player-game support is invalid")
        games.add((row.seed, row.game_id))
        projected_mask = tuple(
            row.hard_eliminated[index] for index in self._indices
        )
        self._scored_examples += 1
        self._color_counts[row.color] += 1
        if row.true_parameter_token is not None:
            parameter_totals = self._parameter_totals[row.color]
            parameter_totals["eligible"] += 1
            if row.parameter_unscorable_reason is not None:
                reasons = self._parameter_reasons[row.color]
                reasons[row.parameter_unscorable_reason] = (
                    reasons.get(row.parameter_unscorable_reason, 0) + 1
                )
            else:
                if row.true_parameters is None:
                    raise EvaluationDataError(
                        "scorable parameter row lacks decoded true values"
                    )
                predicted = row.predicted_parameters or {}
                parameter_totals["scorable"] += 1
                parameter_totals["whole_correct"] += (
                    predicted == row.true_parameters
                )
                for name, value in row.true_parameters.items():
                    parameter_totals["component_count"] += 1
                    parameter_totals["component_correct"] += (
                        predicted.get(name) == value
                    )
        candidate_example: PredictionExample | None = None
        for system in SYSTEM_NAMES:
            projected = _project_distribution(
                row.probabilities[system],
                self._indices,
                projected_mask,
            )
            probabilities = dict(
                zip(self.class_ids, projected, strict=True)
            )
            hard_mask = dict(
                zip(self.class_ids, projected_mask, strict=True)
            )
            example = PredictionExample(
                game_id=row.game_id,
                move_number=row.observed_ply,
                observed_ply=row.observed_ply,
                player_color=row.color,
                true_drawback=row.truth,
                probabilities=probabilities,
                rule_family=self._families.get(row.truth),
                true_parameters=(
                    row.true_parameters
                    if system == "calibrated-ensemble"
                    and row.parameter_unscorable_reason is None
                    else None
                ),
                predicted_parameters=(
                    row.predicted_parameters
                    if system == "calibrated-ensemble"
                    and row.parameter_unscorable_reason is None
                    else None
                ),
                hard_eliminated=hard_mask,
            )
            self._metrics[system][row.color].add(example)
            if system == "calibrated-ensemble":
                candidate_example = example
            values = self._summaries.setdefault(
                (system, row.game_id, row.seed, row.color),
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            )
            values[0] += 1.0
            values[1] += _top_k_credit(probabilities, row.truth, 1)
            values[2] += _top_k_credit(probabilities, row.truth, 3)
            values[3] += _top_k_credit(probabilities, row.truth, 5)
            truth_probability = probabilities[row.truth]
            values[4] += (
                math.inf
                if truth_probability == 0.0
                else -math.log(truth_probability)
            )
            values[5] += math.fsum(
                (
                    probability
                    - (1.0 if label == row.truth else 0.0)
                )
                ** 2
                for label, probability in probabilities.items()
            )
        if candidate_example is None:
            raise RuntimeError("calibrated candidate example was not scored")
        if not row.agent_metadata_present or row.bot_agent_id is None:
            self._slice_issues.add("agent-profile metrics")
        else:
            contract = (row.bot_style, row.bot_strength)
            prior_contract = self._agent_contracts.get(row.bot_agent_id)
            if prior_contract is not None and prior_contract != contract:
                raise EvaluationDataError(
                    f"agent profile {row.bot_agent_id} changes style or strength"
                )
            self._agent_contracts[row.bot_agent_id] = contract
            metrics = self._agent_metrics.setdefault(
                row.bot_agent_id,
                {
                    color: StreamingEvaluation(15)
                    for color in ("white", "black")
                },
            )
            metrics[row.color].add(candidate_example)
            color_counts = self._agent_color_counts.setdefault(
                row.bot_agent_id,
                {"white": 0, "black": 0},
            )
            color_counts[row.color] += 1
            self._agent_counts[row.bot_agent_id] = (
                self._agent_counts.get(row.bot_agent_id, 0) + 1
            )
        mode = "evaluator-backed" if row.evaluator_backed else "synchronous"
        self._mode_metrics[mode][row.color].add(candidate_example)
        self._mode_counts[mode][row.color] += 1

    def finish(
        self,
    ) -> tuple[PromotionViewReport, tuple[PlayerGameSummary, ...]]:
        if self._scored_examples == 0:
            raise EvaluationDataError(
                f"{self.name} view contains no scorable examples"
            )
        reports = MappingProxyType(
            {
                system: PromotionSystemReport(
                    white=(
                        None
                        if self._color_counts["white"] == 0
                        else self._metrics[system]["white"].report()
                    ),
                    black=(
                        None
                        if self._color_counts["black"] == 0
                        else self._metrics[system]["black"].report()
                    ),
                )
                for system in SYSTEM_NAMES
            }
        )
        summaries = tuple(
            PlayerGameSummary(
                view=self.name,
                system=system,
                game_id=game_id,
                seed=seed,
                color=color,
                example_count=int(values[0]),
                top_1_sum=values[1],
                top_3_sum=values[2],
                top_5_sum=values[3],
                nll_sum=values[4],
                brier_sum=values[5],
            )
            for (system, game_id, seed, color), values in sorted(
                self._summaries.items()
            )
        )
        parameter_reports = MappingProxyType(
            {
                color: _parameter_report(
                    self._parameter_totals[color],
                    self._parameter_reasons[color],
                )
                for color in ("white", "black")
            }
        )
        rule_reports = MappingProxyType(
            {
                color: MappingProxyType(
                    {
                        rule_id: RuleOpportunityReport(
                            support_examples=int(values["support"]),
                            support_player_games=len(values["games"]),
                            trigger_opportunities=int(values["triggers"]),
                            unscorable_examples=int(values["unscorable"]),
                        )
                        for rule_id, values in self._rule_totals[color].items()
                    }
                )
                for color in ("white", "black")
            }
        )
        agent_reports = MappingProxyType(
            {
                agent_id: AgentProfileReport(
                    agent_id=agent_id,
                    style=self._agent_contracts[agent_id][0],
                    strength=self._agent_contracts[agent_id][1],
                    example_count=self._agent_counts[agent_id],
                    white=(
                        None
                        if self._agent_color_counts[agent_id]["white"] == 0
                        else self._agent_metrics[agent_id]["white"].report()
                    ),
                    black=(
                        None
                        if self._agent_color_counts[agent_id]["black"] == 0
                        else self._agent_metrics[agent_id]["black"].report()
                    ),
                )
                for agent_id in sorted(self._agent_contracts)
            }
        )
        mode_reports = MappingProxyType(
            {
                mode: EvaluatorModeReport(
                    mode=mode,
                    example_count=sum(self._mode_counts[mode].values()),
                    white=(
                        None
                        if self._mode_counts[mode]["white"] == 0
                        else self._mode_metrics[mode]["white"].report()
                    ),
                    black=(
                        None
                        if self._mode_counts[mode]["black"] == 0
                        else self._mode_metrics[mode]["black"].report()
                    ),
                )
                for mode in ("evaluator-backed", "synchronous")
            }
        )
        unsupported = tuple(sorted(self._slice_issues))
        return (
            PromotionViewReport(
                name=self.name,
                class_ids=self.class_ids,
                unavailable_rule_ids=self.unavailable,
                scored_examples=self._scored_examples,
                unscorable_examples=self._unscorable_examples,
                unscorable_player_games=len(self._unscorable_games),
                systems=reports,
                calibrated_parameters=parameter_reports,
                rule_opportunities=rule_reports,
                agent_profiles=agent_reports,
                evaluator_modes=mode_reports,
                evaluation_slices_complete=not unsupported,
                unsupported_protocol_slices=unsupported,
            ),
            summaries,
        )


class _PromotionScorer:
    def __init__(
        self,
        partition: str,
        rule_families: Mapping[str, str],
        fusion_policy: tuple[str, float] | None = None,
    ) -> None:
        prepared = tuple(SYMBOLIC_RULE_IDS)
        browser = tuple(
            rule_id
            for rule_id in SYMBOLIC_RULE_IDS
            if rule_id not in UNAVAILABLE_BROWSER_RULE_IDS
        )
        if len(prepared) != 182 or len(browser) != 180:
            raise EvaluationDataError("frozen promotion views are invalid")
        self._views = (
            _ViewAccumulator(
                PREPARED_VIEW,
                prepared,
                (),
                rule_families,
            ),
            _ViewAccumulator(
                BROWSER_VIEW,
                browser,
                tuple(
                    rule_id
                    for rule_id in SYMBOLIC_RULE_IDS
                    if rule_id in UNAVAILABLE_BROWSER_RULE_IDS
                ),
                rule_families,
            ),
        )
        self._transcript = _TranscriptBuilder(partition, fusion_policy)
        self.count = 0

    def add(self, row: SystemPredictionRow) -> None:
        self._transcript.add(row)
        for view in self._views:
            view.add(row)
        self.count += 1

    def finish(
        self,
    ) -> tuple[
        Mapping[str, PromotionViewReport],
        tuple[PlayerGameSummary, ...],
        InferenceTranscript,
    ]:
        reports: dict[str, PromotionViewReport] = {}
        summaries: list[PlayerGameSummary] = []
        for accumulator in self._views:
            report, view_summaries = accumulator.finish()
            reports[report.name] = report
            summaries.extend(view_summaries)
        return (
            MappingProxyType(reports),
            tuple(summaries),
            self._transcript.finish(),
        )


def predict_all_systems_batch(
    *,
    members: Sequence[FrozenMemberPredictor],
    examples: Sequence[TrainingExample],
    training_frequency: Mapping[str, Mapping[str, float]],
    temperatures: PromotionTemperatures,
    fusion_alpha: float,
) -> tuple[SystemPredictionRow, ...]:
    """Infer every frozen system once from the same three member forward passes."""

    if len(members) != 3:
        raise EvaluationDataError("promotion evaluation requires three members")
    if tuple(member.checkpoint_seed for member in members) != (
        PROTOCOL_V2_ENSEMBLE_SEEDS
    ):
        raise EvaluationDataError("promotion members are reordered")
    vocabulary = tuple(members[0].drawback_vocabulary)
    if vocabulary != tuple(SYMBOLIC_RULE_IDS) or any(
        tuple(member.drawback_vocabulary) != vocabulary
        for member in members[1:]
    ):
        raise EvaluationDataError(
            "promotion members must use the ordered 182-rule vocabulary"
        )
    parameter_vocabulary = tuple(members[0].parameter_vocabulary)
    if not parameter_vocabulary or any(
        tuple(member.parameter_vocabulary) != parameter_vocabulary
        for member in members[1:]
    ):
        raise EvaluationDataError(
            "promotion members use incompatible parameter vocabularies"
        )
    parameter_vocabulary_set = frozenset(parameter_vocabulary)
    priors = _validate_training_frequency(training_frequency)
    features = tuple(example.features for example in examples)
    member_batches = tuple(
        tuple(member.predict_batch(features)) for member in members
    )
    if any(len(batch) != len(examples) for batch in member_batches):
        raise EvaluationDataError(
            "promotion member output count does not match input count"
        )
    rows: list[SystemPredictionRow] = []
    for index, example in enumerate(examples):
        outputs = tuple(batch[index] for batch in member_batches)
        color = example.features.player_color
        truth = (
            example.white_drawback if color == "white"
            else example.black_drawback
        )
        residual_attribute = f"{color}_neural_residual_logits"
        mask_attribute = f"{color}_hard_eliminated"
        member_residuals = tuple(
            _required_logits(
                getattr(output, residual_attribute),
                f"{color} member residual logits",
            )
            for output in outputs
        )
        output_masks = tuple(
            _required_mask(
                getattr(output, mask_attribute),
                f"{color} member hard mask",
            )
            for output in outputs
        )
        feature_mask, symbolic_prior = _symbolic_inputs(
            example.features,
            color,
        )
        if any(mask != feature_mask for mask in output_masks):
            raise EvaluationDataError(
                "member hard mask disagrees with public symbolic input"
            )
        member_fused = tuple(
            _fuse(
                residual,
                symbolic_prior,
                feature_mask,
                fusion_alpha,
            )
            for residual in member_residuals
        )
        ensemble_residual = tuple(
            math.fsum(values) / 3.0
            for values in zip(*member_residuals, strict=True)
        )
        ensemble_fused = _fuse(
            ensemble_residual,
            symbolic_prior,
            feature_mask,
            fusion_alpha,
        )
        temperature = temperatures.white if color == "white" else temperatures.black
        parameter_rows = tuple(
            (
                output.white_parameter_probabilities
                if color == "white"
                else output.black_parameter_probabilities
            )
            for output in outputs
        )
        mean_parameter_probabilities = _mean_parameter_probabilities(
            parameter_rows,
            parameter_vocabulary,
        )
        true_parameter_token = (
            example.white_parameters
            if color == "white"
            else example.black_parameters
        )
        parameter_observed = (
            example.white_parameters_observed
            if color == "white"
            else example.black_parameters_observed
        )
        true_parameters: Mapping[str, object] | None = None
        predicted_parameters: Mapping[str, object] | None = None
        parameter_unscorable_reason: str | None = None
        if true_parameter_token is not None:
            if not parameter_observed:
                parameter_unscorable_reason = "unobserved"
            else:
                true_parameters, unknown = decode_true_parameter(
                    true_parameter_token,
                    parameter_vocabulary_set,
                )
                if unknown:
                    parameter_unscorable_reason = "out-of-vocabulary"
                elif true_parameters is None:
                    parameter_unscorable_reason = "no-supervised-components"
                else:
                    predicted_parameters = decode_predicted_parameter(
                        mean_parameter_probabilities
                    )
        systems: dict[str, tuple[float, ...]] = {
            "uniform": _uniform(feature_mask),
            "training-frequency": _renormalize(priors[color], feature_mask),
            "symbolic-only": _renormalize(symbolic_prior, feature_mask),
            "member-20260811": member_fused[0].probabilities,
            "member-20260812": member_fused[1].probabilities,
            "member-20260813": member_fused[2].probabilities,
            "uncalibrated-ensemble": ensemble_fused.probabilities,
            "calibrated-ensemble": masked_temperature_softmax(
                ensemble_fused.logits,
                feature_mask,
                temperature,
            ),
        }
        rows.append(
            SystemPredictionRow(
                game_id=example.game_id,
                seed=example.seed,
                color=color,
                observed_ply=example.features.ply // 2 + 1,
                truth=truth,
                hard_eliminated=feature_mask,
                member_residual_logits=(
                    member_residuals[0],
                    member_residuals[1],
                    member_residuals[2],
                ),
                ensemble_fused_logits=ensemble_fused.logits,
                true_parameter_token=true_parameter_token,
                parameter_observed=parameter_observed,
                true_parameters=true_parameters,
                predicted_parameters=predicted_parameters,
                parameter_unscorable_reason=parameter_unscorable_reason,
                rule_triggered=example.rule_triggered,
                bot_agent_id=example.evaluation.bot_agent_id,
                bot_style=example.evaluation.bot_style,
                bot_strength=example.evaluation.bot_strength,
                agent_metadata_present=(
                    example.evaluation.agent_metadata_present
                ),
                evaluator_backed=(
                    example.features.public_evaluator_constraint is not None
                ),
                probabilities=systems,
            )
        )
    return tuple(rows)


def predict_calibrated_two_heads(
    *,
    members: Sequence[FrozenMemberPredictor],
    features: FeatureRecord,
    temperatures: PromotionTemperatures,
    fusion_alpha: float,
) -> Mapping[str, tuple[float, ...]]:
    """Run the exact promotion ensemble/calibration path for both public heads."""

    if len(members) != 3 or tuple(
        member.checkpoint_seed for member in members
    ) != PROTOCOL_V2_ENSEMBLE_SEEDS:
        raise EvaluationDataError("promotion members are reordered")
    vocabulary = tuple(members[0].drawback_vocabulary)
    if vocabulary != tuple(SYMBOLIC_RULE_IDS) or any(
        tuple(member.drawback_vocabulary) != vocabulary for member in members[1:]
    ):
        raise EvaluationDataError("promotion members use a different vocabulary")
    outputs = tuple(member.predict_batch((features,))[0] for member in members)
    result: dict[str, tuple[float, ...]] = {}
    for color, temperature in (
        ("white", temperatures.white),
        ("black", temperatures.black),
    ):
        feature_mask, symbolic_prior = _symbolic_inputs(features, color)
        residual_attribute = f"{color}_neural_residual_logits"
        mask_attribute = f"{color}_hard_eliminated"
        residuals = tuple(
            _required_logits(
                getattr(output, residual_attribute),
                f"{color} member residual logits",
            )
            for output in outputs
        )
        masks = tuple(
            _required_mask(
                getattr(output, mask_attribute), f"{color} member hard mask"
            )
            for output in outputs
        )
        if any(mask != feature_mask for mask in masks):
            raise EvaluationDataError(
                "member hard mask disagrees with public symbolic input"
            )
        ensemble_residual = tuple(
            math.fsum(values) / 3.0
            for values in zip(*residuals, strict=True)
        )
        result[color] = masked_temperature_softmax(
            _fuse(
                ensemble_residual,
                symbolic_prior,
                feature_mask,
                fusion_alpha,
            ).logits,
            feature_mask,
            temperature,
        )
    return MappingProxyType(result)


def score_views(
    rows: Iterable[SystemPredictionRow],
    *,
    partition: str,
    rule_families: Mapping[str, str] | None = None,
) -> tuple[
    Mapping[str, PromotionViewReport],
    tuple[PlayerGameSummary, ...],
    InferenceTranscript,
]:
    """Score prepared and browser views in one streaming pass."""

    scorer = _PromotionScorer(partition, rule_families or {})
    for row in rows:
        scorer.add(row)
    if scorer.count == 0:
        raise EvaluationDataError("promotion evaluation contains no examples")
    return scorer.finish()


def evaluate_candidate_partition(
    *,
    lease: AuditedPrivateCorpusLease,
    partition: str,
    ensemble_release: ContentAddressedJson,
    calibration: ContentAddressedFile,
    training_frequency: TrainingFrequencyReference,
    catalogs: Iterable[Path],
    bootstrap_seed: int,
    batch_size: int,
) -> PromotionReport:
    """Evaluate an authenticated partition without publishing or authorizing it."""

    if partition not in {
        ValidationPartition.VALIDATION_GATE.value,
        "test",
    }:
        raise ValueError("partition must be validation-gate or test")
    if (
        isinstance(bootstrap_seed, bool)
        or not isinstance(bootstrap_seed, int)
        or bootstrap_seed < 0
    ):
        raise ValueError("bootstrap_seed must be a non-negative integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    audited = lease.audited
    expected_split = "validation" if partition != "test" else "test"
    if audited.split != expected_split:
        raise EvaluationDataError(
            f"{partition} requires an authenticated {expected_split} lease"
        )
    loaded_release = verify_ensemble_release(ensemble_release)
    loaded_calibration = load_ensemble_calibration(calibration)
    loaded_frequency = load_training_frequency_artifact(
        training_frequency,
        loaded_release.training_corpus_set_sha256,
    )
    _verify_calibration_binding(
        loaded_calibration,
        ensemble_release,
    )
    fusion_alpha, fusion_selection_sha256 = _calibration_fusion_policy(
        loaded_calibration
    )
    _verify_corpus_binding(loaded_release, audited, partition)
    selected_seeds, assignments = _selected_partition(audited, partition)
    temperatures = PromotionTemperatures(
        white=_calibration_temperature(loaded_calibration, "white"),
        black=_calibration_temperature(loaded_calibration, "black"),
    )
    families = load_rule_families(catalogs)
    scorer = _PromotionScorer(
        partition,
        families,
        fusion_policy=(fusion_selection_sha256, fusion_alpha),
    )
    with ExitStack() as stack:
        checkpoints = _open_checkpoint_sources(
            stack,
            ensemble_release,
            loaded_release,
        )
        checkpoint_payloads = tuple(
            _verified_checkpoint_bytes(source, member.checkpoint_sha256)
            for source, member in zip(
                checkpoints,
                loaded_release.members,
                strict=True,
            )
        )
        ensemble = load_hybrid_ensemble(
            checkpoints,
            device="cpu",
            fusion_alpha=fusion_alpha,
            required_corpus_provenance={
                "training_corpus_set_sha256": (
                    loaded_release.training_corpus_set_sha256
                )
            },
        )
        examples = _partition_examples(
            lease.dataset,
            selected_seeds,
            assignments,
            max_rows_per_game=audited.max_plies or 10_000,
        )
        for batch in _batched(examples, batch_size):
            for row in predict_all_systems_batch(
                members=ensemble.members,
                examples=batch,
                training_frequency=loaded_frequency.promotion_priors(),
                temperatures=temperatures,
                fusion_alpha=fusion_alpha,
            ):
                scorer.add(row)
        for source, expected_payload in zip(
            checkpoints,
            checkpoint_payloads,
            strict=True,
        ):
            source.seek(0)
            if source.read() != expected_payload:
                raise EvaluationDataError(
                    "ensemble checkpoint changed during promotion evaluation"
                )
            source.seek(0)
    if scorer.count == 0:
        raise EvaluationDataError("promotion partition contains no examples")
    lease.verify_dataset_unchanged()
    if verify_ensemble_release(ensemble_release) != loaded_release:
        raise EvaluationDataError(
            "ensemble release changed during promotion evaluation"
        )
    if load_ensemble_calibration(calibration) != loaded_calibration:
        raise EvaluationDataError(
            "ensemble calibration changed during promotion evaluation"
        )
    if (
        load_training_frequency_artifact(
            training_frequency,
            loaded_release.training_corpus_set_sha256,
        )
        != loaded_frequency
    ):
        raise EvaluationDataError(
            "training-frequency artifact changed during promotion evaluation"
        )
    views, summaries, transcript = scorer.finish()
    unsupported_slices = tuple(
        sorted(
            {
                issue
                for view in views.values()
                for issue in view.unsupported_protocol_slices
            }
        )
    )
    return PromotionReport(
        format=PROMOTION_REPORT_FORMAT,
        version=PROMOTION_REPORT_VERSION,
        partition=partition,
        partition_seed_sha256=validation_seed_sha256(selected_seeds),
        bootstrap_seed=bootstrap_seed,
        ensemble_release_sha256=ensemble_release.sha256,
        calibration_sha256=calibration.sha256,
        training_frequency_sha256=training_frequency.sha256,
        move_examples=scorer.count,
        promotion_gate_complete=not unsupported_slices,
        unsupported_protocol_slices=unsupported_slices,
        transcript=transcript,
        views=views,
        player_game_summaries=summaries,
    )


def _validate_training_frequency(
    value: Mapping[str, Mapping[str, float]],
) -> Mapping[str, tuple[float, ...]]:
    if set(value) != {"white", "black"}:
        raise EvaluationDataError(
            "training frequency must contain exactly white and black"
        )
    result: dict[str, tuple[float, ...]] = {}
    for color in ("white", "black"):
        mapping = value[color]
        if tuple(mapping) != tuple(SYMBOLIC_RULE_IDS):
            raise EvaluationDataError(
                f"{color} training frequency is missing or reorders classes"
            )
        frequencies = tuple(float(mapping[item]) for item in SYMBOLIC_RULE_IDS)
        if any(
            not math.isfinite(item) or item < 0.0 for item in frequencies
        ) or math.fsum(frequencies) <= 0.0:
            raise EvaluationDataError(
                f"{color} training frequency contains invalid values"
            )
        result[color] = frequencies
    return MappingProxyType(result)


def _required_logits(
    value: tuple[float, ...] | None,
    label: str,
) -> tuple[float, ...]:
    if value is None or len(value) != len(SYMBOLIC_RULE_IDS) or any(
        not math.isfinite(item) for item in value
    ):
        raise EvaluationDataError(f"{label} are missing or invalid")
    return tuple(value)


def _required_mask(
    value: tuple[bool, ...] | None,
    label: str,
) -> tuple[bool, ...]:
    if (
        value is None
        or len(value) != len(SYMBOLIC_RULE_IDS)
        or any(type(item) is not bool for item in value)
        or all(value)
    ):
        raise EvaluationDataError(f"{label} is missing or invalid")
    return tuple(value)


def _symbolic_inputs(
    features: FeatureRecord,
    color: str,
) -> tuple[tuple[bool, ...], tuple[float, ...]]:
    if color == "white":
        mask = features.symbolic_white_eliminated
        prior = features.symbolic_white_rule_probabilities
    else:
        mask = features.symbolic_black_eliminated
        prior = features.symbolic_black_rule_probabilities
    rendered_mask = _required_mask(tuple(mask), f"{color} symbolic hard mask")
    if len(prior) != len(SYMBOLIC_RULE_IDS) or any(
        not math.isfinite(value) or value < 0.0 for value in prior
    ):
        raise EvaluationDataError(f"{color} symbolic prior is invalid")
    return rendered_mask, tuple(prior)


def _fuse(
    residual: Sequence[float],
    prior: Sequence[float],
    eliminated: Sequence[bool],
    alpha: float,
) -> RankPreservingFusionResult:
    try:
        return rank_preserving_fusion(
            residual,
            prior,
            eliminated,
            alpha=alpha,
        )
    except RankPreservingFusionError as error:
        raise EvaluationDataError(
            "public symbolic input cannot satisfy rank-preserving fusion"
        ) from error


def _mean_parameter_probabilities(
    rows: Sequence[Mapping[str, float]],
    vocabulary: Sequence[str],
) -> Mapping[str, float]:
    if any(tuple(row) != tuple(vocabulary) for row in rows):
        raise EvaluationDataError(
            "member parameter output vocabulary is incompatible"
        )
    values: dict[str, float] = {}
    for token in vocabulary:
        probabilities = tuple(float(row[token]) for row in rows)
        if any(
            not math.isfinite(value) or value < 0.0 or value > 1.0
            for value in probabilities
        ):
            raise EvaluationDataError(
                "member parameter probability is invalid"
            )
        values[token] = math.fsum(probabilities) / len(probabilities)
    if abs(math.fsum(values.values()) - 1.0) > 1e-6:
        raise EvaluationDataError(
            "ensemble parameter probabilities are not normalized"
        )
    return MappingProxyType(values)


def _masked_softmax(
    logits: Sequence[float],
    eliminated: Sequence[bool],
) -> tuple[float, ...]:
    return masked_temperature_softmax(logits, eliminated, 1.0)


def _uniform(eliminated: Sequence[bool]) -> tuple[float, ...]:
    survivor_count = sum(not value for value in eliminated)
    if survivor_count == 0:
        raise EvaluationDataError("hard mask eliminates every class")
    return tuple(
        0.0 if value else 1.0 / survivor_count for value in eliminated
    )


def _renormalize(
    values: Sequence[float],
    eliminated: Sequence[bool],
) -> tuple[float, ...]:
    retained = tuple(
        0.0 if is_eliminated else float(value)
        for value, is_eliminated in zip(values, eliminated, strict=True)
    )
    total = math.fsum(retained)
    if not math.isfinite(total) or total <= 0.0:
        raise EvaluationDataError(
            "comparator has no probability mass on surviving classes"
        )
    return tuple(value / total for value in retained)


def _validate_distribution(
    probabilities: Sequence[float],
    eliminated: Sequence[bool],
    system: str,
) -> None:
    if len(probabilities) != len(SYMBOLIC_RULE_IDS) or any(
        not math.isfinite(value) or value < 0.0 or value > 1.0
        for value in probabilities
    ):
        raise ValueError(f"{system} distribution is invalid")
    if abs(math.fsum(probabilities) - 1.0) > 1e-12:
        raise ValueError(f"{system} distribution is not normalized")
    if any(
        probability != 0.0
        for probability, is_eliminated in zip(
            probabilities, eliminated, strict=True
        )
        if is_eliminated
    ):
        raise ValueError(f"{system} violates the exact hard mask")


def _project_distribution(
    probabilities: Sequence[float],
    indices: Sequence[int],
    projected_mask: Sequence[bool],
) -> tuple[float, ...]:
    selected = tuple(probabilities[index] for index in indices)
    return _renormalize(selected, projected_mask)


def _top_k_credit(
    probabilities: Mapping[str, float],
    truth: str,
    k: int,
) -> float:
    true_probability = probabilities[truth]
    greater = sum(value > true_probability for value in probabilities.values())
    tied = sum(value == true_probability for value in probabilities.values())
    slots = min(k, len(probabilities)) - greater
    return 0.0 if slots <= 0 else min(1.0, slots / tied)


def _parameter_report(
    totals: Mapping[str, int],
    reasons: Mapping[str, int],
) -> ParameterEvaluationReport:
    eligible = totals["eligible"]
    scorable = totals["scorable"]
    components = totals["component_count"]
    unscorable = eligible - scorable
    if sum(reasons.values()) != unscorable:
        raise EvaluationDataError(
            "parameter unscorable reasons do not match coverage totals"
        )
    return ParameterEvaluationReport(
        eligible_examples=eligible,
        scorable_examples=scorable,
        unscorable_examples=unscorable,
        coverage=None if eligible == 0 else scorable / eligible,
        unscorable_rate=None if eligible == 0 else unscorable / eligible,
        whole_object_accuracy=(
            None
            if scorable == 0
            else totals["whole_correct"] / scorable
        ),
        component_accuracy=(
            None
            if components == 0
            else totals["component_correct"] / components
        ),
        component_count=components,
        unscorable_by_reason=MappingProxyType(dict(sorted(reasons.items()))),
    )


def _verify_calibration_binding(
    calibration: Mapping[str, object],
    ensemble: ContentAddressedJson,
) -> None:
    binding = calibration.get("ensemble_release")
    if not isinstance(binding, Mapping) or binding.get("sha256") != ensemble.sha256:
        raise EvaluationDataError(
            "calibration does not bind the selected ensemble release"
        )


def _calibration_fusion_policy(
    calibration: Mapping[str, object],
) -> tuple[float, str]:
    """Read the immutable fusion policy from a verified calibration artifact."""

    identity = calibration.get("identity")
    method = calibration.get("method")
    if not isinstance(identity, Mapping) or not isinstance(method, Mapping):
        raise EvaluationDataError(
            "calibration lacks an authenticated fusion selection policy"
        )
    selection_sha256 = identity.get("fusion_selection_sha256")
    identity_alpha = identity.get("selected_alpha")
    method_alpha = method.get("selected_alpha")
    if (
        not isinstance(selection_sha256, str)
        or len(selection_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in selection_sha256
        )
        or isinstance(identity_alpha, bool)
        or not isinstance(identity_alpha, (int, float))
        or not math.isfinite(float(identity_alpha))
        or not 0.0 <= float(identity_alpha) <= 1.0
        or method.get("fusion") != RANK_PRESERVING_FUSION_METHOD
        or method_alpha != identity_alpha
    ):
        raise EvaluationDataError(
            "calibration fusion selection policy is invalid"
        )
    return float(identity_alpha), selection_sha256


def _calibration_temperature(
    calibration: Mapping[str, object],
    color: str,
) -> float:
    head = calibration.get(color)
    if not isinstance(head, Mapping):
        raise EvaluationDataError(f"calibration lacks the {color} head")
    value = head.get("temperature")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationDataError(f"{color} calibration temperature is invalid")
    return float(value)


def _verify_corpus_binding(
    release: LoadedEnsembleRelease,
    audited: object,
    partition: str,
) -> None:
    if (
        release.release_root_sha256
        != getattr(audited, "release_root_sha256", None)
        or release.corpus_run_id != getattr(audited, "corpus_run_id", None)
    ):
        raise EvaluationDataError(
            "ensemble and held-out corpus release identities disagree"
        )
    if partition == ValidationPartition.VALIDATION_GATE.value and (
        release.private_validation_manifest_sha256
        != getattr(audited, "manifest_sha256", None)
        or release.validation_dataset_sha256
        != getattr(audited, "dataset_sha256", None)
    ):
        raise EvaluationDataError(
            "ensemble and validation corpus bindings disagree"
        )


def _selected_partition(
    audited: object,
    partition: str,
) -> tuple[frozenset[int], Mapping[str, tuple[str, str]]]:
    seeds = tuple(getattr(audited, "seeds"))
    assignments = tuple(getattr(audited, "game_assignments"))
    if len(seeds) != len(assignments):
        raise EvaluationDataError(
            "authenticated game assignments do not align with seeds"
        )
    selected = frozenset(
        seed
        for seed in seeds
        if partition == "test"
        or assign_validation_partition(seed)
        is ValidationPartition.VALIDATION_GATE
    )
    if not selected:
        raise EvaluationDataError(f"{partition} contains no games")
    mapping = {
        game_id: (white, black)
        for seed, (game_id, white, black) in zip(
            seeds, assignments, strict=True
        )
        if seed in selected
    }
    return selected, MappingProxyType(mapping)


def _partition_examples(
    source: BinaryIO,
    selected_seeds: frozenset[int],
    assignments: Mapping[str, tuple[str, str]],
    *,
    max_rows_per_game: int,
) -> Iterator[TrainingExample]:
    if max_rows_per_game <= 0:
        raise ValueError("max_rows_per_game must be positive")
    current_game: str | None = None
    current_rows: list[Mapping[str, object]] = []
    completed: set[str] = set()

    def flush() -> Iterator[TrainingExample]:
        if not current_rows:
            return
        try:
            yield from group_training_examples(current_rows, assignments)
        except DatasetSchemaError as error:
            raise EvaluationDataError(str(error)) from error

    for raw in read_ndjson_stream(source, label="authenticated promotion corpus"):
        try:
            parsed = parse_dataset_row(raw)
        except DatasetSchemaError as error:
            raise EvaluationDataError(str(error)) from error
        if parsed.seed not in selected_seeds:
            continue
        if parsed.game_id not in assignments:
            raise EvaluationDataError(
                f"game {parsed.game_id} has no authenticated assignment"
            )
        if current_game is not None and parsed.game_id != current_game:
            completed.add(current_game)
            yield from flush()
            current_rows.clear()
        if parsed.game_id in completed:
            raise EvaluationDataError(
                f"game {parsed.game_id} is not contiguous in the dataset"
            )
        current_game = parsed.game_id
        current_rows.append(raw)
        if len(current_rows) > max_rows_per_game:
            raise EvaluationDataError(
                f"game {parsed.game_id} exceeds the evaluation row bound"
            )
    yield from flush()


def _batched(
    examples: Iterable[TrainingExample],
    batch_size: int,
) -> Iterator[tuple[TrainingExample, ...]]:
    batch: list[TrainingExample] = []
    for example in examples:
        batch.append(example)
        if len(batch) == batch_size:
            yield tuple(batch)
            batch.clear()
    if batch:
        yield tuple(batch)


def _open_checkpoint_sources(
    stack: ExitStack,
    reference: ContentAddressedJson,
    release: LoadedEnsembleRelease,
) -> tuple[BinaryIO, BinaryIO, BinaryIO]:
    sources: list[BinaryIO] = []
    for member in release.members:
        try:
            resolved = resolve_member_checkpoint(reference, member)
            sources.append(stack.enter_context(resolved.open("rb")))
        except (OSError, ValueError) as error:
            raise EvaluationDataError(
                "ensemble checkpoint source authentication failed"
            ) from error
    if len(sources) != 3:
        raise EvaluationDataError("ensemble release must contain three checkpoints")
    return sources[0], sources[1], sources[2]


def _verified_checkpoint_bytes(
    source: BinaryIO,
    expected_sha256: str,
) -> bytes:
    source.seek(0)
    payload = source.read()
    source.seek(0)
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise EvaluationDataError("ensemble checkpoint SHA-256 does not match")
    return payload
