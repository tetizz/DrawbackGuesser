"""Publication and exhaustive decisions for the frozen validation gate.

This module is intentionally incapable of evaluating the sealed test split.
Every protocol requirement has a stable gate identifier.  Evidence absent from
the current promotion core is recorded as ``missing`` and keeps the candidate
unapproved rather than being approximated.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import re
from types import MappingProxyType
from typing import Callable, Iterable, Mapping, Sequence

from ml.training.drawback_ml.corpus_contract import (
    open_audited_private_corpus_split,
)
from ml.training.drawback_ml.durable_publish import publish_bytes_durable_exact
from ml.training.drawback_ml.path_validation import (
    is_portable_safe_basename,
    portable_basename_key,
)
from ml.training.drawback_ml.symbolic_schema import SYMBOLIC_RULE_IDS

from .ensemble_calibration import ContentAddressedFile
from .promotion_evaluator import (
    BROWSER_VIEW,
    PREPARED_VIEW,
    SYSTEM_NAMES,
    PlayerGameSummary,
    PromotionReport,
    PromotionViewReport,
    evaluate_candidate_partition,
)
from .release_selection_bundle import ContentAddressedJson
from .training_frequency import (
    ContentAddressedFile as TrainingFrequencyReference,
)
from .validation_partition import (
    VALIDATION_PARTITION_IDENTITY,
    ValidationPartition,
)


REPORT_FORMAT = "drawbacktrainer-validation-gate-report"
DECISION_FORMAT = "drawbacktrainer-validation-gate-decision"
VERSION = 1
PROTOCOL_ID = "current-catalog-182-v2"
BOOTSTRAP_SEED = 20260814
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_METHOD = "complete-game-paired-percentile-linear-v1"
CANDIDATE = "calibrated-ensemble"
UNAVAILABLE_BROWSER_RULES = ("hand-and-gigabrain", "ichtyophobe")
REQUIRED_PROTOCOL_SLICES = (
    "trigger-opportunity metrics",
    "agent-profile metrics",
    "evaluator-backed-versus-synchronous-family metrics",
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class GateResult:
    gate_id: str
    status: str
    actual: object
    requirement: str
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"passed", "failed", "missing"}:
            raise ValueError("gate status is invalid")
        if not self.gate_id:
            raise ValueError("gate id must not be empty")
        if self.status == "missing" and not self.reason:
            raise ValueError("a missing gate requires a reason")


@dataclass(frozen=True)
class ValidationGateDecision:
    passed: bool
    results: tuple[GateResult, ...]
    missing_count: int
    failed_count: int
    threshold_contract_sha256: str
    bootstrap: Mapping[str, object]


def _canonical_compact(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _canonical_pretty(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _strict_json(payload: bytes, label: str) -> Mapping[str, object]:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(token: str) -> None:
        raise ValueError(f"{label} contains non-finite constant {token}")

    try:
        value = json.loads(
            payload,
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} root must be an object")
    return value


def load_validation_gate_document(
    reference: ContentAddressedFile,
    expected_format: str,
) -> Mapping[str, object]:
    """Verify digest, canonical encoding, and the publication format."""

    try:
        payload = reference.path.read_bytes()
    except OSError as error:
        raise ValueError(
            f"cannot read validation-gate document: {reference.path}"
        ) from error
    if hashlib.sha256(payload).hexdigest() != reference.sha256:
        raise ValueError("validation-gate document SHA-256 does not match")
    value = _strict_json(payload, "validation-gate document")
    if payload != _canonical_pretty(value):
        raise ValueError("validation-gate document is not canonical")
    if value.get("format") != expected_format or value.get("version") != VERSION:
        raise ValueError("unsupported validation-gate document")
    return value


def _json_value(value: object) -> object:
    if is_dataclass(value):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_atomic_no_clobber(path: Path, payload: bytes, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    publish_bytes_durable_exact(path, payload, label=label)


class _GateCollector:
    def __init__(self) -> None:
        self.results: list[GateResult] = []
        self._ids: set[str] = set()

    def add(
        self,
        gate_id: str,
        *,
        actual: object,
        requirement: str,
        predicate: Callable[[float], bool],
    ) -> None:
        if gate_id in self._ids:
            raise RuntimeError(f"duplicate validation gate id: {gate_id}")
        self._ids.add(gate_id)
        if (
            isinstance(actual, bool)
            or not isinstance(actual, (int, float))
            or not math.isfinite(float(actual))
        ):
            self.results.append(
                GateResult(
                    gate_id,
                    "missing",
                    None,
                    requirement,
                    "required metric is absent, null, or non-finite",
                )
            )
            return
        numeric = float(actual)
        self.results.append(
            GateResult(
                gate_id,
                "passed" if predicate(numeric) else "failed",
                numeric,
                requirement,
            )
        )

    def exact(
        self,
        gate_id: str,
        *,
        actual: object,
        expected: object,
        requirement: str,
    ) -> None:
        if gate_id in self._ids:
            raise RuntimeError(f"duplicate validation gate id: {gate_id}")
        self._ids.add(gate_id)
        if actual is None:
            self.results.append(
                GateResult(
                    gate_id,
                    "missing",
                    None,
                    requirement,
                    "required exactness evidence is absent",
                )
            )
            return
        self.results.append(
            GateResult(
                gate_id,
                "passed" if actual == expected else "failed",
                actual,
                requirement,
            )
        )

    def missing(self, gate_id: str, requirement: str, reason: str) -> None:
        if gate_id in self._ids:
            raise RuntimeError(f"duplicate validation gate id: {gate_id}")
        self._ids.add(gate_id)
        self.results.append(
            GateResult(gate_id, "missing", None, requirement, reason)
        )


def _head(
    view: PromotionViewReport,
    system: str,
    color: str,
) -> object | None:
    report = view.systems.get(system)
    if report is None:
        return None
    return report.white if color == "white" else report.black


def _metric(head: object | None, name: str) -> object:
    return None if head is None else getattr(head, name, None)


def _horizon(
    head: object | None, name: str, observed_plies: int
) -> object:
    mapping = _metric(head, name)
    if not isinstance(mapping, Mapping):
        return None
    return mapping.get(observed_plies)


def _absolute_gates(
    collector: _GateCollector,
    view: PromotionViewReport | None,
    view_name: str,
) -> None:
    thresholds = {
        "game_normalized_top_1_accuracy": (0.10, "at least 0.10"),
        "game_normalized_top_3_accuracy": (0.25, "at least 0.25"),
        "game_normalized_top_5_accuracy": (0.35, "at least 0.35"),
        "negative_log_likelihood": (4.50, "at most 4.50"),
        "brier_score": (0.970, "at most 0.970"),
        "expected_calibration_error": (0.080, "at most 0.080"),
    }
    for color in ("white", "black"):
        head = None if view is None else _head(view, CANDIDATE, color)
        for metric, (threshold, requirement) in thresholds.items():
            actual = _metric(head, metric)
            predicate = (
                (lambda value, bound=threshold: value >= bound)
                if metric.startswith("game_normalized_top")
                else (lambda value, bound=threshold: value <= bound)
            )
            collector.add(
                f"{view_name}.{color}.absolute.{metric}",
                actual=actual,
                requirement=requirement,
                predicate=predicate,
            )
        if view_name == PREPARED_VIEW:
            for k, observed_plies, threshold in (
                (1, 10, 0.040),
                (1, 15, 0.070),
                (1, 20, 0.100),
                (3, 20, 0.250),
            ):
                field = (
                    "top_1_accuracy_at_observed_plies"
                    if k == 1
                    else "top_3_accuracy_at_observed_plies"
                )
                collector.add(
                    f"{view_name}.{color}.horizon.top{k}.{observed_plies}",
                    actual=_horizon(head, field, observed_plies),
                    requirement=f"at least {threshold:.3f}",
                    predicate=lambda value, bound=threshold: value >= bound,
                )


def _exactness_and_coverage(
    collector: _GateCollector,
    views: Mapping[str, PromotionViewReport],
    move_examples: int,
) -> None:
    prepared = views.get(PREPARED_VIEW)
    browser = views.get(BROWSER_VIEW)
    collector.exact(
        "structure.prepared.class_count",
        actual=None if prepared is None else len(prepared.class_ids),
        expected=182,
        requirement="exactly 182 ordered prepared classes",
    )
    collector.exact(
        "structure.prepared.class_order",
        actual=None if prepared is None else tuple(prepared.class_ids),
        expected=tuple(SYMBOLIC_RULE_IDS),
        requirement="prepared class IDs equal the canonical ordered catalog",
    )
    collector.exact(
        "structure.browser.class_count",
        actual=None if browser is None else len(browser.class_ids),
        expected=180,
        requirement="exactly 180 ordered browser classes",
    )
    collector.exact(
        "structure.browser.class_order",
        actual=None if browser is None else tuple(browser.class_ids),
        expected=tuple(
            rule_id
            for rule_id in SYMBOLIC_RULE_IDS
            if rule_id not in UNAVAILABLE_BROWSER_RULES
        ),
        requirement="browser class IDs equal the canonical 180-rule projection",
    )
    collector.exact(
        "structure.browser.unavailable_rules",
        actual=(
            None
            if browser is None
            else tuple(browser.unavailable_rule_ids)
        ),
        expected=UNAVAILABLE_BROWSER_RULES,
        requirement="exactly Hand and Gigabrain and Ichtyophobe unavailable",
    )
    for view_name, view in ((PREPARED_VIEW, prepared), (BROWSER_VIEW, browser)):
        collector.exact(
            f"structure.{view_name}.systems",
            actual=None if view is None else tuple(view.systems),
            expected=SYSTEM_NAMES,
            requirement="all frozen systems exist in canonical order",
        )
    collector.exact(
        "coverage.prepared.unscorable_examples",
        actual=None if prepared is None else prepared.unscorable_examples,
        expected=0,
        requirement="prepared truth scorable for every move example",
    )
    collector.exact(
        "coverage.prepared.scored_examples",
        actual=None if prepared is None else prepared.scored_examples,
        expected=move_examples,
        requirement="prepared view scores every promotion move example",
    )
    collector.missing(
        "coverage.prepared.per-color-rates",
        "at least 99.5% move and player-game scorable for each color",
        (
            "promotion core does not expose per-color assigned player-game "
            "denominators and excluded-example reasons"
        ),
    )
    collector.missing(
        "coverage.prepared.rule-support-minima",
        "all 182 rules meet frozen complete-game support minima",
        "promotion core exposes move-row rule support, not complete-game support",
    )
    collector.exact(
        "coverage.browser.accounted-examples",
        actual=(
            None
            if browser is None
            else browser.scored_examples + browser.unscorable_examples
        ),
        expected=move_examples,
        requirement="every browser-view example is scored or explicitly unavailable",
    )
    collector.add(
        "coverage.browser.unavailable-player-games",
        actual=(
            None if browser is None else browser.unscorable_player_games
        ),
        requirement="unavailable browser player-games are explicitly counted",
        predicate=lambda value: value >= 0,
    )

    for view_name, view in ((PREPARED_VIEW, prepared), (BROWSER_VIEW, browser)):
        for system in SYSTEM_NAMES:
            for color in ("white", "black"):
                head = None if view is None else _head(view, system, color)
                diagnostics = _metric(head, "probability_diagnostics")
                collector.add(
                    f"{view_name}.{system}.{color}.normalization",
                    actual=(
                        None
                        if diagnostics is None
                        else getattr(
                            diagnostics, "maximum_absolute_sum_error", None
                        )
                    ),
                    requirement="maximum absolute normalization error at most 1e-6",
                    predicate=lambda value: value <= 1e-6,
                )
                collector.exact(
                    f"{view_name}.{system}.{color}.hard-eliminations",
                    actual=(
                        None
                        if diagnostics is None
                        else getattr(
                            diagnostics,
                            "hard_elimination_violation_count",
                            None,
                        )
                    ),
                    expected=0,
                    requirement="exactly zero hard-elimination violations",
                )
    for gate_id, requirement in (
        (
            "exactness.unknown-or-duplicate-class-ids",
            "exactly zero unknown or duplicate class IDs",
        ),
        (
            "exactness.symbolic-version-errors",
            "exactly zero wrong symbolic versions",
        ),
        ("exactness.nonfinite-outputs", "exactly zero non-finite outputs"),
        ("exactness.cross-color-swaps", "exactly zero cross-color head swaps"),
    ):
        # Successful construction by promotion_evaluator validates these
        # invariants before a SystemPredictionRow can enter an accumulator.
        collector.exact(
            gate_id,
            actual=0,
            expected=0,
            requirement=requirement,
        )


def _comparator_gates(
    collector: _GateCollector,
    views: Mapping[str, PromotionViewReport],
) -> None:
    for view_name in (PREPARED_VIEW, BROWSER_VIEW):
        view = views.get(view_name)
        for color in ("white", "black"):
            candidate = None if view is None else _head(view, CANDIDATE, color)
            frequency = (
                None
                if view is None
                else _head(view, "training-frequency", color)
            )
            symbolic = (
                None if view is None else _head(view, "symbolic-only", color)
            )
            frequency_comparisons = (
                (
                    "frequency.top1",
                    "game_normalized_top_1_accuracy",
                    frequency,
                    0.05,
                    "candidate minus training-frequency Top-1 at least 0.05",
                    lambda value: value >= 0.05,
                ),
                (
                    "frequency.top3",
                    "game_normalized_top_3_accuracy",
                    frequency,
                    0.10,
                    "candidate minus training-frequency Top-3 at least 0.10",
                    lambda value: value >= 0.10,
                ),
                (
                    "frequency.nll",
                    "negative_log_likelihood",
                    frequency,
                    0.20,
                    "training-frequency NLL minus candidate at least 0.20",
                    lambda value: value >= 0.20,
                ),
                (
                    "frequency.brier",
                    "brier_score",
                    frequency,
                    0.010,
                    "training-frequency Brier minus candidate at least 0.010",
                    lambda value: value >= 0.010,
                ),
            )
            symbolic_comparisons = (
                (
                    "symbolic.top1",
                    "game_normalized_top_1_accuracy",
                    symbolic,
                    -0.02,
                    "candidate Top-1 no more than 0.02 below symbolic",
                    lambda value: value >= -0.02,
                ),
                (
                    "symbolic.top3",
                    "game_normalized_top_3_accuracy",
                    symbolic,
                    -0.03,
                    "candidate Top-3 no more than 0.03 below symbolic",
                    lambda value: value >= -0.03,
                ),
                (
                    "symbolic.nll",
                    "negative_log_likelihood",
                    symbolic,
                    0.15,
                    "symbolic NLL minus candidate at least 0.15",
                    lambda value: value >= 0.15,
                ),
                (
                    "symbolic.brier",
                    "brier_score",
                    symbolic,
                    0.010,
                    "symbolic Brier minus candidate at least 0.010",
                    lambda value: value >= 0.010,
                ),
            )
            comparisons = (
                (*frequency_comparisons, *symbolic_comparisons)
                if view_name == PREPARED_VIEW
                else symbolic_comparisons
            )
            for (
                suffix,
                metric,
                comparator,
                _threshold,
                requirement,
                predicate,
            ) in comparisons:
                candidate_value = _metric(candidate, metric)
                comparator_value = _metric(comparator, metric)
                if (
                    isinstance(candidate_value, (int, float))
                    and not isinstance(candidate_value, bool)
                    and math.isfinite(float(candidate_value))
                    and isinstance(comparator_value, (int, float))
                    and not isinstance(comparator_value, bool)
                    and math.isfinite(float(comparator_value))
                ):
                    if suffix.endswith(("nll", "brier")):
                        difference = float(comparator_value) - float(candidate_value)
                    else:
                        difference = float(candidate_value) - float(comparator_value)
                else:
                    difference = None
                collector.add(
                    f"{view_name}.{color}.comparator.{suffix}",
                    actual=difference,
                    requirement=requirement,
                    predicate=predicate,
                )


def _calibration_gates(
    collector: _GateCollector,
    views: Mapping[str, PromotionViewReport],
) -> None:
    for view_name in (PREPARED_VIEW, BROWSER_VIEW):
        view = views.get(view_name)
        for color in ("white", "black"):
            calibrated = (
                None if view is None else _head(view, CANDIDATE, color)
            )
            uncalibrated = (
                None
                if view is None
                else _head(view, "uncalibrated-ensemble", color)
            )
            for metric in (
                "top_1_accuracy",
                "top_3_accuracy",
                "top_5_accuracy",
                "game_normalized_top_1_accuracy",
                "game_normalized_top_3_accuracy",
                "game_normalized_top_5_accuracy",
            ):
                left = _metric(calibrated, metric)
                right = _metric(uncalibrated, metric)
                actual = (
                    None
                    if not (
                        isinstance(left, (int, float))
                        and isinstance(right, (int, float))
                        and math.isfinite(float(left))
                        and math.isfinite(float(right))
                    )
                    else abs(float(left) - float(right))
                )
                collector.add(
                    f"{view_name}.{color}.calibration.ranking.{metric}",
                    actual=actual,
                    requirement="calibration leaves Top-k ranking credit unchanged",
                    predicate=lambda value: value <= 1e-12,
                )
            if view_name == PREPARED_VIEW:
                calibrated_ece = _metric(
                    calibrated, "expected_calibration_error"
                )
                uncalibrated_ece = _metric(
                    uncalibrated, "expected_calibration_error"
                )
                ece_delta = (
                    None
                    if not (
                        isinstance(calibrated_ece, (int, float))
                        and isinstance(uncalibrated_ece, (int, float))
                        and math.isfinite(float(calibrated_ece))
                        and math.isfinite(float(uncalibrated_ece))
                    )
                    else float(calibrated_ece) - float(uncalibrated_ece)
                )
                collector.add(
                    f"{view_name}.{color}.calibration.ece",
                    actual=ece_delta,
                    requirement="calibrated ECE no worse than uncalibrated",
                    predicate=lambda value: value <= 0.0,
                )
                calibrated_nll = _metric(
                    calibrated, "negative_log_likelihood"
                )
                uncalibrated_nll = _metric(
                    uncalibrated, "negative_log_likelihood"
                )
                nll_delta = (
                    None
                    if not (
                        isinstance(calibrated_nll, (int, float))
                        and isinstance(uncalibrated_nll, (int, float))
                        and math.isfinite(float(calibrated_nll))
                        and math.isfinite(float(uncalibrated_nll))
                    )
                    else float(uncalibrated_nll) - float(calibrated_nll)
                )
                collector.add(
                    f"{view_name}.{color}.calibration.nll",
                    actual=nll_delta,
                    requirement="calibrated NLL strictly lower than uncalibrated",
                    predicate=lambda value: value > 0.0,
                )


def _rule_parameter_and_member_gates(
    collector: _GateCollector,
    prepared: PromotionViewReport | None,
) -> None:
    for color in ("white", "black"):
        candidate = None if prepared is None else _head(prepared, CANDIDATE, color)
        per_rule = _metric(candidate, "metrics_per_drawback")
        if not isinstance(per_rule, Mapping) or set(per_rule) != set(
            SYMBOLIC_RULE_IDS
        ):
            collector.missing(
                f"prepared-182.{color}.rules.top5-above-uniform",
                "at least 80% of 182 rules have Top-5 above 5/182",
                "complete per-rule metrics are absent",
            )
        else:
            top5_values = [
                getattr(item, "top_5_accuracy", None)
                for item in per_rule.values()
            ]
            valid_top5 = all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in top5_values
            )
            collector.add(
                f"prepared-182.{color}.rules.top5-above-uniform",
                actual=(
                    None
                    if not valid_top5
                    else sum(float(value) > 5 / 182 for value in top5_values)
                    / 182
                ),
                requirement="at least 80% of 182 rules have Top-5 above 5/182",
                predicate=lambda value: value >= 0.80,
            )
        rule_top3 = (
            []
            if not isinstance(per_rule, Mapping)
            or set(per_rule) != set(SYMBOLIC_RULE_IDS)
            else [
                getattr(item, "top_3_accuracy", None)
                for item in per_rule.values()
            ]
        )
        macro_top3 = (
            None
            if len(rule_top3) != len(SYMBOLIC_RULE_IDS)
            or not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in rule_top3
            )
            else math.fsum(float(value) for value in rule_top3)
            / len(SYMBOLIC_RULE_IDS)
        )
        collector.add(
            f"prepared-182.{color}.rules.macro-top3",
            actual=macro_top3,
            requirement="macro per-rule Top-3 at least 0.18",
            predicate=lambda value: value >= 0.18,
        )
        collector.missing(
            f"prepared-182.{color}.rules.nonzero-top5-supported",
            "no rule with at least 25 supported player-games has Top-5 zero",
            "per-rule complete player-game support is unavailable",
        )
        parameters = (
            None
            if prepared is None
            else prepared.calibrated_parameters.get(color)
        )
        collector.add(
            f"prepared-182.{color}.parameters.whole-object-accuracy",
            actual=(
                None
                if parameters is None
                else parameters.whole_object_accuracy
            ),
            requirement="whole-object parameter accuracy at least 0.20",
            predicate=lambda value: value >= 0.20,
        )
        collector.add(
            f"prepared-182.{color}.parameters.coverage",
            actual=None if parameters is None else parameters.coverage,
            requirement="parameter coverage at least 0.95",
            predicate=lambda value: value >= 0.95,
        )

    for seed in (20260811, 20260812, 20260813):
        system = f"member-{seed}"
        heads = (
            None if prepared is None else _head(prepared, system, "white"),
            None if prepared is None else _head(prepared, system, "black"),
        )
        top3 = tuple(
            _metric(head, "game_normalized_top_3_accuracy") for head in heads
        )
        nll = tuple(_metric(head, "negative_log_likelihood") for head in heads)
        mean_top3 = (
            None
            if not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in top3
            )
            else math.fsum(float(value) for value in top3) / 2
        )
        mean_nll = (
            None
            if not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in nll
            )
            else math.fsum(float(value) for value in nll) / 2
        )
        collector.add(
            f"prepared-182.{system}.mean-white-black-top3",
            actual=mean_top3,
            requirement="mean White/Black Top-3 at least 0.18",
            predicate=lambda value: value >= 0.18,
        )
        collector.add(
            f"prepared-182.{system}.mean-white-black-nll",
            actual=mean_nll,
            requirement="mean White/Black NLL at most 4.90",
            predicate=lambda value: value <= 4.90,
        )


def _evaluation_slice_gates(
    collector: _GateCollector,
    views: Mapping[str, PromotionViewReport],
) -> None:
    """Require complete trigger, agent, and evaluator-mode reports."""

    for view_name in (PREPARED_VIEW, BROWSER_VIEW):
        view = views.get(view_name)
        collector.exact(
            f"{view_name}.slices.complete",
            actual=(
                None if view is None else view.evaluation_slices_complete
            ),
            expected=True,
            requirement="all evaluation-only metadata slices are complete",
        )
        collector.exact(
            f"{view_name}.slices.unsupported",
            actual=(
                None
                if view is None
                else tuple(view.unsupported_protocol_slices)
            ),
            expected=(),
            requirement="no trigger, agent, or evaluator slice is unsupported",
        )
        if view is None:
            for suffix, requirement in (
                ("trigger-rule-domain", "all 182 per-rule trigger reports"),
                ("agent-profiles", "at least one complete agent profile"),
                (
                    "evaluator-modes",
                    "explicit evaluator-backed and synchronous slices",
                ),
            ):
                collector.missing(
                    f"{view_name}.slices.{suffix}",
                    requirement,
                    "promotion view is absent",
                )
            continue
        trigger_complete = set(view.rule_opportunities) == {
            "white",
            "black",
        } and all(
            set(view.rule_opportunities[color]) == set(SYMBOLIC_RULE_IDS)
            and all(
                item.support_examples >= 0
                and item.support_player_games >= 0
                and item.trigger_opportunities >= 0
                and item.unscorable_examples >= 0
                and item.trigger_opportunities
                <= item.support_examples + item.unscorable_examples
                for item in view.rule_opportunities[color].values()
            )
            for color in ("white", "black")
        )
        collector.exact(
            f"{view_name}.slices.trigger-rule-domain",
            actual=trigger_complete,
            expected=True,
            requirement=(
                "all 182 rules have bounded support, trigger-opportunity, "
                "and unscorable counts for both colors"
            ),
        )
        profile_count = sum(
            profile.example_count for profile in view.agent_profiles.values()
        )
        profile_complete = (
            bool(view.agent_profiles)
            and profile_count == view.scored_examples
            and all(
                profile.agent_id == agent_id
                and profile.example_count > 0
                and (
                    0
                    if profile.white is None
                    else profile.white.count
                )
                + (
                    0
                    if profile.black is None
                    else profile.black.count
                )
                == profile.example_count
                for agent_id, profile in view.agent_profiles.items()
            )
        )
        collector.exact(
            f"{view_name}.slices.agent-profiles",
            actual=profile_complete,
            expected=True,
            requirement=(
                "every scorable example belongs to one complete agent profile"
            ),
        )
        mode_complete = (
            set(view.evaluator_modes)
            == {"evaluator-backed", "synchronous"}
            and sum(
                mode.example_count for mode in view.evaluator_modes.values()
            )
            == view.scored_examples
            and all(
                mode.mode == name
                and mode.example_count
                == (
                    0 if mode.white is None else mode.white.count
                )
                + (
                    0 if mode.black is None else mode.black.count
                )
                for name, mode in view.evaluator_modes.items()
            )
        )
        collector.exact(
            f"{view_name}.slices.evaluator-modes",
            actual=mode_complete,
            expected=True,
            requirement=(
                "every scorable example is sliced as evaluator-backed or "
                "synchronous"
            ),
        )


def _linear_percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires observations")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _paired_intervals(
    summaries: Sequence[PlayerGameSummary],
) -> Mapping[str, Mapping[str, Mapping[str, float]]]:
    by_key = {
        (item.view, item.system, item.seed, item.color): item
        for item in summaries
    }
    game_seeds = sorted({item.seed for item in summaries})
    if not game_seeds or len(by_key) != len(summaries):
        raise ValueError("paired summaries are empty or duplicated")
    random_source = random.Random(BOOTSTRAP_SEED)
    samples: dict[tuple[str, str, str], list[float]] = {
        (view, color, metric): []
        for view in (PREPARED_VIEW, BROWSER_VIEW)
        for color in ("white", "black")
        for metric in ("top1", "nll")
    }
    for _ in range(BOOTSTRAP_REPLICATES):
        selected = [random_source.choice(game_seeds) for _ in game_seeds]
        for view in (PREPARED_VIEW, BROWSER_VIEW):
            for color in ("white", "black"):
                differences = {"top1": [], "nll": []}
                for game_seed in selected:
                    candidate = by_key.get((view, CANDIDATE, game_seed, color))
                    symbolic = by_key.get(
                        (view, "symbolic-only", game_seed, color)
                    )
                    if candidate is None and symbolic is None:
                        continue
                    if candidate is None or symbolic is None:
                        raise ValueError("paired systems have different game support")
                    if (
                        candidate.example_count <= 0
                        or symbolic.example_count != candidate.example_count
                    ):
                        raise ValueError("paired summary counts are invalid")
                    differences["top1"].append(
                        candidate.top_1_sum / candidate.example_count
                        - symbolic.top_1_sum / symbolic.example_count
                    )
                    differences["nll"].append(
                        candidate.nll_sum / candidate.example_count
                        - symbolic.nll_sum / symbolic.example_count
                    )
                for metric, values in differences.items():
                    if not values or any(not math.isfinite(item) for item in values):
                        raise ValueError("paired bootstrap input is absent or non-finite")
                    samples[(view, color, metric)].append(
                        math.fsum(values) / len(values)
                    )
    result: dict[str, dict[str, dict[str, float]]] = {}
    for (view, color, metric), values in samples.items():
        result.setdefault(view, {}).setdefault(color, {})[metric] = {
            "lower": _linear_percentile(values, 0.025),
            "upper": _linear_percentile(values, 0.975),
        }
    return result


def _bootstrap_gates(
    collector: _GateCollector,
    summaries: Sequence[PlayerGameSummary],
) -> Mapping[str, object]:
    try:
        intervals = _paired_intervals(summaries)
    except (ValueError, KeyError) as error:
        for view in (PREPARED_VIEW, BROWSER_VIEW):
            for color in ("white", "black"):
                collector.missing(
                    f"{view}.{color}.bootstrap.top1-vs-symbolic",
                    "95% paired Top-1 difference lower bound above -0.02",
                    str(error),
                )
                collector.missing(
                    f"{view}.{color}.bootstrap.nll-vs-symbolic",
                    "95% paired NLL difference upper bound below 0",
                    str(error),
                )
        return {
            "method": BOOTSTRAP_METHOD,
            "seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_REPLICATES,
            "intervals": None,
            "error": str(error),
        }
    for view in (PREPARED_VIEW, BROWSER_VIEW):
        for color in ("white", "black"):
            top1 = intervals[view][color]["top1"]["lower"]
            nll = intervals[view][color]["nll"]["upper"]
            collector.add(
                f"{view}.{color}.bootstrap.top1-vs-symbolic",
                actual=top1,
                requirement="95% paired Top-1 difference lower bound above -0.02",
                predicate=lambda value: value > -0.02,
            )
            collector.add(
                f"{view}.{color}.bootstrap.nll-vs-symbolic",
                actual=nll,
                requirement="95% paired NLL difference upper bound below 0",
                predicate=lambda value: value < 0.0,
            )
    return {
        "method": BOOTSTRAP_METHOD,
        "seed": BOOTSTRAP_SEED,
        "replicates": BOOTSTRAP_REPLICATES,
        "intervals": intervals,
        "error": None,
    }


def decide_validation_gate(report: PromotionReport) -> ValidationGateDecision:
    """Evaluate every frozen requirement; unavailable evidence fails closed."""

    collector = _GateCollector()
    collector.exact(
        "identity.report-format",
        actual=(report.format, report.version),
        expected=("drawbacktrainer-promotion-report", 1),
        requirement="supported promotion report format version",
    )
    collector.exact(
        "identity.partition",
        actual=report.partition,
        expected=ValidationPartition.VALIDATION_GATE.value,
        requirement="validation-gate partition only",
    )
    collector.exact(
        "identity.bootstrap-seed",
        actual=report.bootstrap_seed,
        expected=BOOTSTRAP_SEED,
        requirement="bootstrap seed exactly 20260814",
    )
    collector.add(
        "identity.move-examples",
        actual=report.move_examples,
        requirement="at least one move example",
        predicate=lambda value: value > 0,
    )
    for name, digest in (
        ("partition-seed", report.partition_seed_sha256),
        ("ensemble-release", report.ensemble_release_sha256),
        ("calibration", report.calibration_sha256),
        ("training-frequency", report.training_frequency_sha256),
        ("transcript", report.transcript.sha256),
    ):
        collector.exact(
            f"identity.{name}-sha256",
            actual=(
                isinstance(digest, str)
                and SHA256_PATTERN.fullmatch(digest) is not None
            ),
            expected=True,
            requirement=f"{name} uses a lowercase SHA-256 binding",
        )
    collector.exact(
        "identity.transcript.algorithm",
        actual=report.transcript.algorithm,
        expected="sha256-domain-separated-canonical-json-v1",
        requirement="frozen inference transcript algorithm",
    )
    collector.exact(
        "identity.transcript.record-count",
        actual=report.transcript.record_count,
        expected=report.move_examples,
        requirement="transcript covers every evaluated move example",
    )
    collector.exact(
        "identity.promotion-core-complete",
        actual=report.promotion_gate_complete,
        expected=True,
        requirement="promotion core declares every frozen protocol slice available",
    )
    collector.exact(
        "identity.unsupported-protocol-slices",
        actual=tuple(report.unsupported_protocol_slices),
        expected=(),
        requirement="no unsupported promotion protocol slices",
    )
    unsupported_slices = set(report.unsupported_protocol_slices)
    for slice_name in REQUIRED_PROTOCOL_SLICES:
        slug = (
            slice_name.lower()
            .replace(" ", "-")
            .replace("/", "-")
        )
        if slice_name in unsupported_slices:
            collector.missing(
                f"protocol-slice.{slug}",
                f"{slice_name} are present in the required promotion report",
                "promotion core explicitly declares this protocol slice unsupported",
            )
        else:
            collector.exact(
                f"protocol-slice.{slug}",
                actual=True,
                expected=True,
                requirement=(
                    f"{slice_name} are present in the required promotion report"
                ),
            )
    views = report.views
    _exactness_and_coverage(collector, views, report.move_examples)
    for view_name in (PREPARED_VIEW, BROWSER_VIEW):
        _absolute_gates(collector, views.get(view_name), view_name)
    _comparator_gates(collector, views)
    _calibration_gates(collector, views)
    _rule_parameter_and_member_gates(collector, views.get(PREPARED_VIEW))
    _evaluation_slice_gates(collector, views)
    bootstrap = _bootstrap_gates(collector, report.player_game_summaries)
    results = tuple(collector.results)
    contract = [
        {
            "gate_id": item.gate_id,
            "requirement": item.requirement,
        }
        for item in results
    ]
    threshold_sha = hashlib.sha256(_canonical_compact(contract)).hexdigest()
    missing = sum(item.status == "missing" for item in results)
    failed = sum(item.status == "failed" for item in results)
    return ValidationGateDecision(
        passed=missing == 0 and failed == 0,
        results=results,
        missing_count=missing,
        failed_count=failed,
        threshold_contract_sha256=threshold_sha,
        bootstrap=MappingProxyType(dict(bootstrap)),
    )


def publish_validation_gate(
    report: PromotionReport,
    report_output: Path,
    decision_output: Path,
) -> tuple[ContentAddressedFile, ContentAddressedFile, ValidationGateDecision]:
    """Publish immutable evidence and its exhaustive decision."""

    if not is_portable_safe_basename(report_output.name) or not (
        is_portable_safe_basename(decision_output.name)
    ):
        raise ValueError("validation output file must be a safe basename")
    resolved_report = report_output.resolve()
    resolved_decision = decision_output.resolve()
    if resolved_report == resolved_decision or (
        resolved_report.parent == resolved_decision.parent
        and portable_basename_key(report_output.name)
        == portable_basename_key(decision_output.name)
    ):
        raise ValueError("validation report and decision must be distinct")
    if resolved_report.parent != resolved_decision.parent:
        raise ValueError("validation report and decision must be siblings")
    decision = decide_validation_gate(report)
    report_value = {
        "format": REPORT_FORMAT,
        "version": VERSION,
        "protocol": {
            "id": PROTOCOL_ID,
            "validation_partition_identity": VALIDATION_PARTITION_IDENTITY,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        },
        "bindings": {
            "ensemble_release_sha256": report.ensemble_release_sha256,
            "calibration_sha256": report.calibration_sha256,
            "training_frequency_sha256": report.training_frequency_sha256,
        },
        "promotion": _json_value(report),
    }
    report_payload = _canonical_pretty(report_value)
    _write_atomic_no_clobber(
        report_output, report_payload, "validation-gate report"
    )
    report_reference = ContentAddressedFile(
        report_output, hashlib.sha256(report_payload).hexdigest()
    )
    decision_value = {
        "format": DECISION_FORMAT,
        "version": VERSION,
        "protocol_id": PROTOCOL_ID,
        "validation_report": {
            "file": report_output.name,
            "sha256": report_reference.sha256,
        },
        "passed": decision.passed,
        "missing_count": decision.missing_count,
        "failed_count": decision.failed_count,
        "threshold_contract_sha256": decision.threshold_contract_sha256,
        "bootstrap": _json_value(decision.bootstrap),
        "results": _json_value(decision.results),
    }
    decision_payload = _canonical_pretty(decision_value)
    _write_atomic_no_clobber(
        decision_output, decision_payload, "validation-gate decision"
    )
    decision_reference = ContentAddressedFile(
        decision_output, hashlib.sha256(decision_payload).hexdigest()
    )
    reloaded_report = load_validation_gate_document(
        report_reference, REPORT_FORMAT
    )
    reloaded_decision = load_validation_gate_document(
        decision_reference, DECISION_FORMAT
    )
    report_binding = reloaded_decision.get("validation_report")
    protocol = reloaded_report.get("protocol")
    if (
        not isinstance(report_binding, Mapping)
        or not isinstance(protocol, Mapping)
        or report_binding.get("file") != report_output.name
        or report_binding.get("sha256") != report_reference.sha256
        or protocol.get("id") != PROTOCOL_ID
    ):
        raise RuntimeError("published validation evidence bindings changed")
    return report_reference, decision_reference, decision


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ml.evaluation.validation_gate"
    )
    parser.add_argument("ensemble_release", type=Path)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("training_frequency", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--ensemble-sha256", required=True)
    parser.add_argument("--calibration-sha256", required=True)
    parser.add_argument("--training-frequency-sha256", required=True)
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--private-validation", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--decision-output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--catalog",
        action="append",
        type=Path,
        default=[],
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    evidence_directory = arguments.ensemble_release.parent.resolve()
    for path, label in (
        (arguments.calibration, "calibration"),
        (arguments.training_frequency, "training-frequency artifact"),
        (arguments.report_output, "report output"),
        (arguments.decision_output, "decision output"),
    ):
        if path.parent.resolve() != evidence_directory:
            raise ValueError(f"{label} must be beside the ensemble release")
    catalogs: Iterable[Path] = arguments.catalog or (
        Path("engine/data/catalog/observed-drawbacks.json"),
    )
    with open_audited_private_corpus_split(
        arguments.public_root,
        arguments.private_validation,
        arguments.dataset,
        "validation",
    ) as lease:
        report = evaluate_candidate_partition(
            lease=lease,
            partition=ValidationPartition.VALIDATION_GATE.value,
            ensemble_release=ContentAddressedJson(
                arguments.ensemble_release, arguments.ensemble_sha256
            ),
            calibration=ContentAddressedFile(
                arguments.calibration, arguments.calibration_sha256
            ),
            training_frequency=TrainingFrequencyReference(
                arguments.training_frequency,
                arguments.training_frequency_sha256,
            ),
            catalogs=catalogs,
            bootstrap_seed=BOOTSTRAP_SEED,
            batch_size=arguments.batch_size,
        )
    report_ref, decision_ref, decision = publish_validation_gate(
        report, arguments.report_output, arguments.decision_output
    )
    print(
        json.dumps(
            {
                "report": {
                    "file": report_ref.path.name,
                    "sha256": report_ref.sha256,
                },
                "decision": {
                    "file": decision_ref.path.name,
                    "sha256": decision_ref.sha256,
                    "passed": decision.passed,
                    "missing": decision.missing_count,
                    "failed": decision.failed_count,
                },
            },
            sort_keys=True,
        )
    )
    return 0 if decision.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
