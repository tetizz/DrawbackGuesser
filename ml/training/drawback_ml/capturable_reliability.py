"""Shared release gates for capturable validation and sealed-test metrics."""

from __future__ import annotations

import math
from typing import Any, Mapping

from .capturable_records import (
    CAPTURABLE_RULE_IDS,
    CapturableDatasetError,
)


def _metric(
    section: Mapping[str, Any],
    key: str,
    label: str,
    *,
    maximum: float | None = None,
) -> float:
    value = section.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
        or (maximum is not None and float(value) > maximum)
    ):
        upper = "unbounded" if maximum is None else str(maximum)
        raise CapturableDatasetError(
            f"{label} {key} must be finite between 0.0 and {upper}"
        )
    return float(value)


def validation_reliability_checks(
    control: Mapping[str, Any],
    treatment: Mapping[str, Any],
    primary_confirmed: bool,
) -> Mapping[str, bool]:
    """Compare two metric bundles using the common non-regression gate."""

    if not isinstance(primary_confirmed, bool):
        raise CapturableDatasetError(
            "primary confirmation must be a boolean"
        )

    def sections(
        metrics: Mapping[str, Any],
        label: str,
    ) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
        hybrid = metrics.get("hybrid")
        trigger = metrics.get("trigger")
        forced = metrics.get("forced")
        if (
            not isinstance(hybrid, Mapping)
            or not isinstance(trigger, Mapping)
            or not isinstance(forced, Mapping)
        ):
            raise CapturableDatasetError(
                f"{label} reliability metrics are invalid"
            )
        return hybrid, trigger, forced

    control_hybrid, control_trigger, control_forced = sections(
        control,
        "control",
    )
    treatment_hybrid, treatment_trigger, treatment_forced = sections(
        treatment,
        "treatment",
    )
    control_horizons = control_hybrid.get("accuracy_after_moves")
    treatment_horizons = treatment_hybrid.get("accuracy_after_moves")
    if (
        not isinstance(control_horizons, Mapping)
        or not isinstance(treatment_horizons, Mapping)
        or set(control_horizons) != set(treatment_horizons)
        or set(control_horizons) != {"5", "10", "15", "20"}
    ):
        raise CapturableDatasetError(
            "control and treatment move horizons are incompatible"
        )
    for label, hybrid in (
        ("control", control_hybrid),
        ("treatment", treatment_hybrid),
    ):
        top1 = _metric(
            hybrid,
            "game_normalized_top_1_accuracy",
            label,
            maximum=1.0,
        )
        top3 = _metric(
            hybrid,
            "game_normalized_top_3_accuracy",
            label,
            maximum=1.0,
        )
        top5 = _metric(
            hybrid,
            "game_normalized_top_5_accuracy",
            label,
            maximum=1.0,
        )
        if top3 < top1:
            raise CapturableDatasetError(
                f"{label} Top-3 cannot be below Top-1"
            )
        if top5 < top3:
            raise CapturableDatasetError(
                f"{label} Top-5 cannot be below Top-3"
            )
    control_per_drawback = control_hybrid.get("metrics_per_drawback")
    treatment_per_drawback = treatment_hybrid.get("metrics_per_drawback")
    if (
        not isinstance(control_per_drawback, Mapping)
        or not isinstance(treatment_per_drawback, Mapping)
        or set(control_per_drawback) != set(treatment_per_drawback)
        or set(control_per_drawback) != set(CAPTURABLE_RULE_IDS)
    ):
        raise CapturableDatasetError(
            "control and treatment drawback metrics are incompatible"
        )

    per_drawback_non_regression = True
    for drawback_id, control_metrics in control_per_drawback.items():
        treatment_metrics = treatment_per_drawback.get(drawback_id)
        if not isinstance(control_metrics, Mapping) or not isinstance(
            treatment_metrics,
            Mapping,
        ):
            raise CapturableDatasetError(
                f"{drawback_id} metrics are invalid"
            )
        control_top1 = _metric(
            control_metrics,
            "top_1_accuracy",
            f"control {drawback_id}",
            maximum=1.0,
        )
        treatment_top1 = _metric(
            treatment_metrics,
            "top_1_accuracy",
            f"treatment {drawback_id}",
            maximum=1.0,
        )
        per_drawback_non_regression = (
            per_drawback_non_regression
            and treatment_top1 >= control_top1 - 0.01
        )

    def exact_symbolic_authority(
        hybrid: Mapping[str, Any],
        label: str,
    ) -> bool:
        diagnostics = hybrid.get("probability_diagnostics")
        if not isinstance(diagnostics, Mapping):
            raise CapturableDatasetError(
                f"{label} probability diagnostics are invalid"
            )
        violations = diagnostics.get("hard_elimination_violation_count")
        missing_masks = diagnostics.get("missing_hard_mask_count")
        checked = diagnostics.get("checked_count")
        hard_mask_checked = diagnostics.get("hard_mask_checked_count")
        if (
            isinstance(violations, bool)
            or not isinstance(violations, int)
            or violations < 0
            or isinstance(missing_masks, bool)
            or not isinstance(missing_masks, int)
            or missing_masks < 0
            or isinstance(checked, bool)
            or not isinstance(checked, int)
            or checked < 0
            or isinstance(hard_mask_checked, bool)
            or not isinstance(hard_mask_checked, int)
            or hard_mask_checked < 0
        ):
            raise CapturableDatasetError(
                f"{label} symbolic counts are invalid"
            )
        return (
            checked > 0
            and hard_mask_checked == checked
            and violations == 0
            and missing_masks == 0
            and _metric(
                diagnostics,
                "maximum_eliminated_probability",
                label,
                maximum=1.0,
            )
            == 0.0
        )

    control_symbolic_authority = exact_symbolic_authority(
        control_hybrid,
        "control",
    )
    treatment_symbolic_authority = exact_symbolic_authority(
        treatment_hybrid,
        "treatment",
    )

    return {
        "primaryRankingConfirmed": primary_confirmed,
        "top1NonRegression": (
            _metric(
                treatment_hybrid,
                "game_normalized_top_1_accuracy",
                "treatment",
                maximum=1.0,
            )
            >= _metric(
                control_hybrid,
                "game_normalized_top_1_accuracy",
                "control",
                maximum=1.0,
            )
        ),
        "top3NonRegression": (
            _metric(
                treatment_hybrid,
                "game_normalized_top_3_accuracy",
                "treatment",
                maximum=1.0,
            )
            >= _metric(
                control_hybrid,
                "game_normalized_top_3_accuracy",
                "control",
                maximum=1.0,
            )
        ),
        "negativeLogLikelihoodNonRegression": (
            _metric(
                treatment_hybrid,
                "game_normalized_negative_log_likelihood",
                "treatment",
            )
            <= _metric(
                control_hybrid,
                "game_normalized_negative_log_likelihood",
                "control",
            )
        ),
        "brierNonRegression": (
            _metric(
                treatment_hybrid,
                "game_normalized_brier_score",
                "treatment",
            )
            <= _metric(
                control_hybrid,
                "game_normalized_brier_score",
                "control",
            )
        ),
        "calibrationNonRegression": (
            _metric(
                treatment_hybrid,
                "expected_calibration_error",
                "treatment",
                maximum=1.0,
            )
            <= _metric(
                control_hybrid,
                "expected_calibration_error",
                "control",
                maximum=1.0,
            )
        ),
        "allMoveHorizonsNonRegression": all(
            _metric(
                treatment_horizons,
                str(horizon),
                "treatment horizon",
                maximum=1.0,
            )
            >= _metric(
                control_horizons,
                str(horizon),
                "control horizon",
                maximum=1.0,
            )
            for horizon in control_horizons
        ),
        "triggerAccuracyNonRegression": (
            _metric(
                treatment_trigger,
                "accuracy",
                "treatment trigger",
                maximum=1.0,
            )
            >= _metric(
                control_trigger,
                "accuracy",
                "control trigger",
                maximum=1.0,
            )
        ),
        "forcedAccuracyNonRegression": (
            _metric(
                treatment_forced,
                "accuracy",
                "treatment forced",
                maximum=1.0,
            )
            >= _metric(
                control_forced,
                "accuracy",
                "control forced",
                maximum=1.0,
            )
        ),
        "perDrawbackTop1WithinOnePoint": per_drawback_non_regression,
        "symbolicAuthorityPreserved": (
            control_symbolic_authority
            and treatment_symbolic_authority
        ),
    }
