"""Validation-only comparison for frozen capturable model candidates."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .capturable_baseline import (
    _canonical_json,
    _opportunity_contract,
    _publish_bytes,
)
from .capturable_experiment import (
    CAPTURABLE_OPPORTUNITY_SELECTION_VERSION,
    CAPTURABLE_SELECTION_FORMAT,
    CAPTURABLE_SELECTION_VERSION,
    _load_selection_checkpoint,
)
from .capturable_records import (
    CAPTURABLE_FEATURE_DIMENSION,
    CAPTURABLE_RULE_IDS,
    CapturableDatasetError,
)
from .capturable_reliability import validation_reliability_checks
from .path_validation import is_portable_safe_basename, portable_basename_key

CAPTURABLE_TREATMENT_COMPARISON_FORMAT = (
    "drawbackguesser-capturable-treatment-comparison"
)
CAPTURABLE_TREATMENT_COMPARISON_VERSION = 2
_TREATMENT_SELECTION_METRIC = (
    "validation game-normalized Top-1, then Top-3, then lowest "
    "NLL, then lower trigger multiplier, then lower seed"
)
_TREATMENT_PROMOTION_METRIC = (
    "strictly better validation game-normalized Top-1, then Top-3, "
    "then lowest NLL, with no regression in Top-1, Top-3, NLL, "
    "Brier score, calibration, move horizons, trigger accuracy, or "
    "forced accuracy, no per-drawback Top-1 loss above one absolute "
    "percentage point, and exact symbolic authority; parameter "
    "tie-breaks cannot promote"
)
_LEGACY_TREATMENT_PROMOTION_METRIC = (
    "strictly better validation game-normalized Top-1, then Top-3, "
    "then lowest NLL; parameter tie-breaks cannot promote"
)


def _selection_report(path: Path) -> tuple[Mapping[str, Any], str]:
    payload = path.resolve().read_bytes()
    if not payload.endswith(b"\n") or payload.endswith(b"\r\n"):
        raise CapturableDatasetError(
            f"{path.name} must use canonical LF framing"
        )

    def pairs(items: Sequence[tuple[str, Any]]) -> Mapping[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise CapturableDatasetError(
                    f"{path.name} contains duplicate key {key}"
                )
            result[key] = value
        return result

    def reject_constant(token: str) -> Any:
        raise CapturableDatasetError(
            f"{path.name} contains non-finite number {token}"
        )

    try:
        report = json.loads(
            payload,
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, UnicodeError) as error:
        raise CapturableDatasetError(
            f"{path.name} is invalid JSON"
        ) from error
    if not isinstance(report, Mapping) or payload != _canonical_json(report):
        raise CapturableDatasetError(
            f"{path.name} is not canonical selection JSON"
        )
    return report, hashlib.sha256(payload).hexdigest()


def _finite_metric(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise CapturableDatasetError(f"{label} must be finite")
    return float(value)


def _bounded_metric(
    value: Any,
    label: str,
    *,
    minimum: float,
    maximum: float | None = None,
) -> float:
    measured = _finite_metric(value, label)
    if measured < minimum or (
        maximum is not None and measured > maximum
    ):
        upper = "unbounded" if maximum is None else str(maximum)
        raise CapturableDatasetError(
            f"{label} must be between {minimum} and {upper}"
        )
    return measured


def _validated_candidate(
    path: Path,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    resolved_path = path.resolve()
    selection_directory = resolved_path.parent.name
    selection_report = resolved_path.name
    if not is_portable_safe_basename(
        selection_directory
    ) or not is_portable_safe_basename(selection_report):
        raise CapturableDatasetError("selection path is not portable")
    report, report_sha256 = _selection_report(path)
    report_version = report.get("version")
    if (
        report.get("format") != CAPTURABLE_SELECTION_FORMAT
        or type(report_version) is not int
        or report_version
        not in {
            CAPTURABLE_SELECTION_VERSION,
            CAPTURABLE_OPPORTUNITY_SELECTION_VERSION,
        }
        or report.get("freshStart") is not True
        or report.get("sealedTestStatus") != "unopened"
        or report.get("ruleIds") != list(CAPTURABLE_RULE_IDS)
        or report.get("featureDimension")
        != CAPTURABLE_FEATURE_DIMENSION
        or "test" in report
    ):
        raise CapturableDatasetError(
            f"{path.name} is not a compatible unopened selection"
        )
    config = report.get("config")
    validation = report.get("validation")
    checkpoint = report.get("checkpoint")
    inputs = report.get("inputs")
    if (
        not isinstance(config, Mapping)
        or not isinstance(validation, Mapping)
        or not isinstance(checkpoint, Mapping)
        or not isinstance(inputs, Mapping)
    ):
        raise CapturableDatasetError(
            f"{path.name} selection sections are invalid"
        )
    train_input = inputs.get("train")
    validation_input = inputs.get("validation")
    if not isinstance(train_input, Mapping) or not isinstance(
        validation_input,
        Mapping,
    ):
        raise CapturableDatasetError(
            f"{path.name} input identities are invalid"
        )
    seed = config.get("seed")
    trigger_multiplier = config.get("trigger_row_multiplier")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed <= 0xFFFF_FFFF
    ):
        raise CapturableDatasetError(
            f"{path.name} model seed is invalid"
        )
    trigger = _finite_metric(
        trigger_multiplier,
        f"{path.name} trigger multiplier",
    )
    hybrid = validation.get("hybrid")
    if not isinstance(hybrid, Mapping):
        raise CapturableDatasetError(
            f"{path.name} validation hybrid metrics are invalid"
        )
    top1 = _bounded_metric(
        hybrid.get("game_normalized_top_1_accuracy"),
        f"{path.name} Top-1",
        minimum=0.0,
        maximum=1.0,
    )
    top3 = _bounded_metric(
        hybrid.get("game_normalized_top_3_accuracy"),
        f"{path.name} Top-3",
        minimum=0.0,
        maximum=1.0,
    )
    nll = _bounded_metric(
        hybrid.get("game_normalized_negative_log_likelihood"),
        f"{path.name} NLL",
        minimum=0.0,
    )
    if top3 < top1:
        raise CapturableDatasetError(
            f"{path.name} Top-3 cannot be below Top-1"
        )
    checkpoint_file = checkpoint.get("file")
    checkpoint_sha256 = checkpoint.get("sha256")
    if (
        not is_portable_safe_basename(checkpoint_file)
        or portable_basename_key(checkpoint_file)
        == portable_basename_key(selection_report)
        or not isinstance(checkpoint_sha256, str)
        or len(checkpoint_sha256) != 64
    ):
        raise CapturableDatasetError(
            f"{path.name} checkpoint identity is invalid"
        )
    checkpoint_path = resolved_path.parent / checkpoint_file
    _, checkpoint_metadata, measured_checkpoint_sha256 = (
        _load_selection_checkpoint(checkpoint_path)
    )
    opportunity_keys = {
        "symbolicFeatureVersion",
        "opportunityFeatureVersion",
        "opportunityRuleIds",
        "opportunityFields",
        "opportunityShape",
        "opportunityMode",
    }
    opportunity_contract: Mapping[str, Any] = {}
    if report_version == CAPTURABLE_OPPORTUNITY_SELECTION_VERSION:
        try:
            opportunity_contract = _opportunity_contract(
                report.get("opportunityMode")
            )
        except ValueError as error:
            raise CapturableDatasetError(
                f"{path.name} opportunity contract is invalid"
            ) from error
        if (
            any(
                report.get(key) != value
                for key, value in opportunity_contract.items()
            )
            or any(
                checkpoint_metadata.get(key) != value
                for key, value in opportunity_contract.items()
            )
        ):
            raise CapturableDatasetError(
                f"{path.name} opportunity contract is invalid"
            )
    elif (
        any(key in report for key in opportunity_keys)
        or any(key in checkpoint_metadata for key in opportunity_keys)
    ):
        raise CapturableDatasetError(
            f"{path.name} legacy selection declares opportunities"
        )
    if (
        measured_checkpoint_sha256 != checkpoint_sha256
        or checkpoint_metadata.get("validation") != validation
        or checkpoint_metadata.get("inputs") != inputs
        or checkpoint_metadata.get("config") != config
        or checkpoint_metadata.get("selectedEpoch")
        != report.get("selectedEpoch")
        or checkpoint_metadata.get("selectedFusionAlpha")
        != report.get("selectedFusionAlpha")
        or checkpoint_metadata.get("selectedPriorSmoothing")
        != report.get("selectedPriorSmoothing")
    ):
        raise CapturableDatasetError(
            f"{path.name} checkpoint does not bind its selection report"
        )
    return (
        {
            "selectionDirectory": selection_directory,
            "selectionReport": selection_report,
            "selectionReportSha256": report_sha256,
            "checkpointFile": checkpoint_file,
            "checkpointSha256": checkpoint_sha256,
            "seed": seed,
            "triggerRowMultiplier": trigger,
            "validationGameNormalizedTop1": top1,
            "validationGameNormalizedTop3": top3,
            "validationGameNormalizedNll": nll,
            **(
                {}
                if not opportunity_contract
                else {"opportunityContract": dict(opportunity_contract)}
            ),
        },
        report,
    )


def _selection_order(candidate: Mapping[str, Any]) -> tuple[float, ...]:
    return _performance_order(candidate) + (
        -float(candidate["triggerRowMultiplier"]),
        -float(candidate["seed"]),
    )


def _performance_order(candidate: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        float(candidate["validationGameNormalizedTop1"]),
        float(candidate["validationGameNormalizedTop3"]),
        -float(candidate["validationGameNormalizedNll"]),
    )


def _comparable_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        key: value
        for key, value in config.items()
        if key not in {"seed", "trigger_row_multiplier"}
    }


def _opportunity_mode(candidate: Mapping[str, Any]) -> str | None:
    contract = candidate.get("opportunityContract")
    if not isinstance(contract, Mapping):
        return None
    mode = contract.get("opportunityMode")
    return mode if isinstance(mode, str) else None


def _validate_cross_mode_opportunity_ablation(
    control: Mapping[str, Any],
    control_report: Mapping[str, Any],
    treatment: Mapping[str, Any],
    treatment_report: Mapping[str, Any],
    *,
    label: str,
) -> None:
    control_mode = _opportunity_mode(control)
    treatment_mode = _opportunity_mode(treatment)
    if control_mode == treatment_mode:
        return
    if (
        control_mode != "zero-ablation"
        or treatment_mode != "public-exact"
    ):
        raise CapturableDatasetError(
            f"{label} cross-mode opportunity comparison must use "
            "zero-ablation control and public-exact treatment"
        )
    if control["seed"] != treatment["seed"]:
        raise CapturableDatasetError(
            f"{label} cross-mode opportunity comparison must use "
            "the same seed"
        )
    if control_report["config"] != treatment_report["config"]:
        raise CapturableDatasetError(
            f"{label} cross-mode opportunity comparison must use "
            "the same full training configuration"
        )
    control_inputs = control_report["inputs"]
    treatment_inputs = treatment_report["inputs"]
    if control_inputs["train"] != treatment_inputs["train"]:
        raise CapturableDatasetError(
            f"{label} cross-mode opportunity comparison must use "
            "the same training input"
        )
    if control_inputs["validation"] != treatment_inputs["validation"]:
        raise CapturableDatasetError(
            f"{label} cross-mode opportunity comparison must use "
            "the same validation input"
        )


def run_candidate_selection(
    report_paths: Sequence[Path],
    output_path: Path,
) -> Mapping[str, Any]:
    """Choose among frozen candidates using validation metrics only."""

    if output_path.exists():
        raise FileExistsError("candidate selection output already exists")
    if len(report_paths) < 2:
        raise ValueError("candidate selection requires at least two reports")
    candidates: list[dict[str, Any]] = []
    common_inputs: Any = None
    identities: set[tuple[int, float, str | None]] = set()
    for path in report_paths:
        candidate, report = _validated_candidate(path)
        inputs = report.get("inputs")
        if common_inputs is None:
            common_inputs = inputs
        elif inputs != common_inputs:
            raise CapturableDatasetError(
                "candidate reports use different train or validation inputs"
            )
        seed = candidate["seed"]
        trigger = candidate["triggerRowMultiplier"]
        opportunity_contract = candidate.get("opportunityContract")
        opportunity_mode = (
            opportunity_contract.get("opportunityMode")
            if isinstance(opportunity_contract, Mapping)
            else None
        )
        identity = (seed, trigger, opportunity_mode)
        if identity in identities:
            raise CapturableDatasetError(
                "candidate seed, trigger multiplier, and opportunity mode "
                "must be unique"
            )
        identities.add(identity)
        candidates.append(candidate)
    selected = max(
        candidates,
        key=_selection_order,
    )
    artifact = {
        "format": "drawbackguesser-capturable-candidate-selection",
        "version": 1,
        "selectionMetric": (
            "validation game-normalized Top-1, then Top-3, then lowest "
            "NLL, then lower trigger multiplier, then lower seed"
        ),
        "inputs": common_inputs,
        "candidates": candidates,
        "selected": selected,
        "sealedTestStatus": "unopened",
    }
    payload = _canonical_json(artifact)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _publish_bytes(output_path, payload)
    return {
        "artifactPath": str(output_path),
        "artifactSha256": hashlib.sha256(payload).hexdigest(),
        "selectedDirectory": selected["selectionDirectory"],
        "selectedCheckpointSha256": selected["checkpointSha256"],
        "validationGameNormalizedTop1": selected[
            "validationGameNormalizedTop1"
        ],
        "validationGameNormalizedTop3": selected[
            "validationGameNormalizedTop3"
        ],
    }


def run_treatment_comparison(
    control_report_path: Path,
    treatment_report_paths: Sequence[Path],
    output_path: Path,
) -> Mapping[str, Any]:
    """Compare training interventions on one unchanged validation corpus."""

    if output_path.exists():
        raise FileExistsError("treatment comparison output already exists")
    if not treatment_report_paths:
        raise ValueError("at least one treatment report is required")
    control, control_report = _validated_candidate(control_report_path)
    control_inputs = control_report["inputs"]
    control_validation = control_inputs["validation"]
    control_config = _comparable_config(control_report["config"])
    identities = {
        (
            control["selectionReportSha256"],
            control["checkpointSha256"],
        )
    }
    treatments: list[dict[str, Any]] = []
    treatment_reports: dict[str, Mapping[str, Any]] = {}
    for path in treatment_report_paths:
        treatment, report = _validated_candidate(path)
        _validate_cross_mode_opportunity_ablation(
            control,
            control_report,
            treatment,
            report,
            label="control and treatment",
        )
        identity = (
            treatment["selectionReportSha256"],
            treatment["checkpointSha256"],
        )
        if identity in identities:
            raise CapturableDatasetError(
                "control and treatment candidates must be distinct"
            )
        identities.add(identity)
        inputs = report["inputs"]
        if inputs["validation"] != control_validation:
            raise CapturableDatasetError(
                "control and treatment use different validation inputs"
            )
        if _comparable_config(report["config"]) != control_config:
            raise CapturableDatasetError(
                "control and treatment use different model configurations"
            )
        treatment["trainInput"] = inputs["train"]
        treatments.append(treatment)
        treatment_reports[treatment["selectionReportSha256"]] = report
    control["trainInput"] = control_inputs["train"]
    best_treatment = max(treatments, key=_selection_order)
    primary_confirmed = _performance_order(
        best_treatment
    ) > _performance_order(
        control
    )
    best_treatment_report = treatment_reports[
        best_treatment["selectionReportSha256"]
    ]
    reliability_checks = validation_reliability_checks(
        control_report["validation"],
        best_treatment_report["validation"],
        primary_confirmed,
    )
    promoted = all(reliability_checks.values())
    selected = best_treatment if promoted else control
    artifact = {
        "format": CAPTURABLE_TREATMENT_COMPARISON_FORMAT,
        "version": CAPTURABLE_TREATMENT_COMPARISON_VERSION,
        "treatmentSelectionMetric": _TREATMENT_SELECTION_METRIC,
        "promotionMetric": _TREATMENT_PROMOTION_METRIC,
        "validationInput": control_validation,
        "control": control,
        "treatments": treatments,
        "bestTreatment": best_treatment,
        "primaryDecision": (
            "confirm-treatment"
            if primary_confirmed
            else "reject-treatment"
        ),
        "reliabilityChecks": reliability_checks,
        "releaseDecision": (
            "promote-treatment" if promoted else "retain-control"
        ),
        "decision": (
            "promote-treatment" if promoted else "retain-control"
        ),
        "selected": selected,
        "sealedTestStatus": "unopened",
    }
    payload = _canonical_json(artifact)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _publish_bytes(output_path, payload)
    return {
        "artifactPath": str(output_path),
        "artifactSha256": hashlib.sha256(payload).hexdigest(),
        "primaryDecision": artifact["primaryDecision"],
        "releaseDecision": artifact["releaseDecision"],
        "decision": artifact["decision"],
        "selectedDirectory": selected["selectionDirectory"],
        "selectedCheckpointSha256": selected["checkpointSha256"],
        "validationGameNormalizedTop1": selected[
            "validationGameNormalizedTop1"
        ],
        "validationGameNormalizedTop3": selected[
            "validationGameNormalizedTop3"
        ],
    }


def load_treatment_comparison(
    path: Path,
) -> tuple[Mapping[str, Any], str]:
    """Authenticate a comparison and every selection checkpoint it binds."""

    artifact, artifact_sha256 = _selection_report(path)
    current_expected_keys = {
        "format",
        "version",
        "treatmentSelectionMetric",
        "promotionMetric",
        "validationInput",
        "control",
        "treatments",
        "bestTreatment",
        "primaryDecision",
        "reliabilityChecks",
        "releaseDecision",
        "decision",
        "selected",
        "sealedTestStatus",
    }
    legacy_expected_keys = current_expected_keys - {
        "primaryDecision",
        "reliabilityChecks",
        "releaseDecision",
    }
    version = artifact.get("version")
    legacy = version == 1
    expected_keys = (
        legacy_expected_keys if legacy else current_expected_keys
    )
    expected_promotion_metric = (
        _LEGACY_TREATMENT_PROMOTION_METRIC
        if legacy
        else _TREATMENT_PROMOTION_METRIC
    )
    if (
        set(artifact) != expected_keys
        or artifact.get("format")
        != CAPTURABLE_TREATMENT_COMPARISON_FORMAT
        or type(version) is not int
        or version not in {1, CAPTURABLE_TREATMENT_COMPARISON_VERSION}
        or artifact.get("treatmentSelectionMetric")
        != _TREATMENT_SELECTION_METRIC
        or artifact.get("promotionMetric") != expected_promotion_metric
        or artifact.get("sealedTestStatus") != "unopened"
    ):
        raise CapturableDatasetError(
            f"{path.name} is not a compatible treatment comparison"
        )
    control_entry = artifact.get("control")
    treatment_entries = artifact.get("treatments")
    if not isinstance(control_entry, Mapping) or (
        not isinstance(treatment_entries, list)
        or not treatment_entries
        or any(
            not isinstance(entry, Mapping)
            for entry in treatment_entries
        )
    ):
        raise CapturableDatasetError(
            f"{path.name} comparison candidates are invalid"
        )

    def candidate_location(
        entry: Mapping[str, Any],
    ) -> tuple[str, str]:
        directory = entry.get("selectionDirectory")
        report_name = entry.get("selectionReport")
        if (
            not is_portable_safe_basename(directory)
            or not is_portable_safe_basename(report_name)
        ):
            raise CapturableDatasetError(
                f"{path.name} contains an invalid candidate path"
            )
        return directory, report_name

    control_location = candidate_location(control_entry)
    treatment_locations = [
        candidate_location(entry) for entry in treatment_entries
    ]
    seen_locations: dict[tuple[str, str], tuple[str, str]] = {}
    for directory, report_name in (control_location, *treatment_locations):
        location = (directory, report_name)
        location_key = (
            portable_basename_key(directory),
            portable_basename_key(report_name),
        )
        prior_location = seen_locations.get(location_key)
        if prior_location is not None and prior_location != location:
            raise CapturableDatasetError(
                f"{path.name} contains a portable-colliding candidate path"
            )
        seen_locations[location_key] = location

    def authenticate(
        entry: Mapping[str, Any],
        location: tuple[str, str],
    ) -> tuple[dict[str, Any], Mapping[str, Any]]:
        directory, report_name = location
        candidate, report = _validated_candidate(
            path.resolve().parent / directory / report_name
        )
        candidate["trainInput"] = report["inputs"]["train"]
        if candidate != entry:
            raise CapturableDatasetError(
                f"{path.name} candidate identity does not match its report"
            )
        return candidate, report

    control, control_report = authenticate(control_entry, control_location)
    treatments_with_reports = [
        authenticate(entry, location)
        for entry, location in zip(
            treatment_entries,
            treatment_locations,
            strict=True,
        )
    ]
    treatments = [item[0] for item in treatments_with_reports]
    identities = {
        (
            control["selectionReportSha256"],
            control["checkpointSha256"],
        )
    }
    for treatment in treatments:
        identity = (
            treatment["selectionReportSha256"],
            treatment["checkpointSha256"],
        )
        if identity in identities:
            raise CapturableDatasetError(
                f"{path.name} reuses a comparison candidate"
            )
        identities.add(identity)
    validation_input = control_report["inputs"]["validation"]
    comparable_config = _comparable_config(control_report["config"])
    if artifact.get("validationInput") != validation_input:
        raise CapturableDatasetError(
            f"{path.name} validation identity is inconsistent"
        )
    for treatment, report in treatments_with_reports:
        _validate_cross_mode_opportunity_ablation(
            control,
            control_report,
            treatment,
            report,
            label=path.name,
        )
        if report["inputs"]["validation"] != validation_input:
            raise CapturableDatasetError(
                f"{path.name} candidates use different validation inputs"
            )
        if _comparable_config(report["config"]) != comparable_config:
            raise CapturableDatasetError(
                f"{path.name} candidates use different model configurations"
            )
    best_treatment = max(treatments, key=_selection_order)
    primary_confirmed = _performance_order(
        best_treatment
    ) > _performance_order(
        control
    )
    best_treatment_report = next(
        report
        for candidate, report in treatments_with_reports
        if candidate["selectionReportSha256"]
        == best_treatment["selectionReportSha256"]
    )
    reliability_checks = validation_reliability_checks(
        control_report["validation"],
        best_treatment_report["validation"],
        primary_confirmed,
    )
    promoted = all(reliability_checks.values())
    selected = best_treatment if promoted else control
    expected_primary_decision = (
        "confirm-treatment"
        if primary_confirmed
        else "reject-treatment"
    )
    expected_decision = (
        "promote-treatment" if promoted else "retain-control"
    )
    if legacy:
        legacy_decision = (
            "promote-treatment"
            if primary_confirmed
            else "retain-control"
        )
        legacy_selected = (
            best_treatment if primary_confirmed else control
        )
        consistent = (
            artifact.get("bestTreatment") == best_treatment
            and artifact.get("decision") == legacy_decision
            and artifact.get("selected") == legacy_selected
        )
    else:
        consistent = (
            artifact.get("bestTreatment") == best_treatment
            and artifact.get("primaryDecision")
            == expected_primary_decision
            and artifact.get("reliabilityChecks")
            == reliability_checks
            and artifact.get("releaseDecision") == expected_decision
            and artifact.get("decision") == expected_decision
            and artifact.get("selected") == selected
        )
    if not consistent:
        raise CapturableDatasetError(
            f"{path.name} comparison decision is inconsistent"
        )
    if not legacy:
        return artifact, artifact_sha256
    normalized = dict(artifact)
    normalized.update(
        {
            "primaryDecision": expected_primary_decision,
            "reliabilityChecks": reliability_checks,
            "releaseDecision": expected_decision,
            "decision": expected_decision,
            "selected": selected,
        }
    )
    return normalized, artifact_sha256
