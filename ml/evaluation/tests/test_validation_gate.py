from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import ml.evaluation.validation_gate as validation_gate
from ml.evaluation.promotion_evaluator import (
    PROMOTION_REPORT_FORMAT,
    PROMOTION_REPORT_VERSION,
    SYSTEM_NAMES,
    PromotionReport,
    SystemPredictionRow,
    score_views,
)
from ml.evaluation.validation_gate import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    BROWSER_VIEW,
    PREPARED_VIEW,
    build_parser,
    decide_validation_gate,
    load_validation_gate_document,
    publish_validation_gate,
)
from ml.evaluation.ensemble_calibration import ContentAddressedFile
from ml.training.drawback_ml.durable_publish import publish_bytes_durable_exact
from ml.training.drawback_ml.symbolic_schema import SYMBOLIC_RULE_IDS


def peaked(probability: float, truth: str = "vegan") -> tuple[float, ...]:
    remainder = (1.0 - probability) / (len(SYMBOLIC_RULE_IDS) - 1)
    return tuple(
        probability if rule_id == truth else remainder
        for rule_id in SYMBOLIC_RULE_IDS
    )


def promotion_report(
    *,
    partition: str = "validation-gate",
    bootstrap_seed: int = BOOTSTRAP_SEED,
    complete_slices: bool = False,
) -> PromotionReport:
    probabilities = {
        "uniform": tuple(1 / len(SYMBOLIC_RULE_IDS) for _ in SYMBOLIC_RULE_IDS),
        "training-frequency": peaked(0.10),
        "symbolic-only": peaked(0.20),
        "member-20260811": peaked(0.60),
        "member-20260812": peaked(0.60),
        "member-20260813": peaked(0.60),
        "uncalibrated-ensemble": peaked(0.90),
        "calibrated-ensemble": peaked(0.95),
    }
    self = SystemPredictionRow(
        game_id="game-1",
        seed=101,
        color="white",
        observed_ply=20,
        truth="vegan",
        hard_eliminated=tuple(False for _ in SYMBOLIC_RULE_IDS),
        member_residual_logits=(
            tuple(0.0 for _ in SYMBOLIC_RULE_IDS),
            tuple(0.0 for _ in SYMBOLIC_RULE_IDS),
            tuple(0.0 for _ in SYMBOLIC_RULE_IDS),
        ),
        ensemble_fused_logits=tuple(0.0 for _ in SYMBOLIC_RULE_IDS),
        true_parameter_token=None,
        parameter_observed=False,
        true_parameters=None,
        predicted_parameters=None,
        parameter_unscorable_reason=None,
        probabilities=probabilities,
        bot_agent_id=(
            "human-like-medium" if complete_slices else None
        ),
        bot_style="human-like" if complete_slices else None,
        bot_strength=1400 if complete_slices else None,
        agent_metadata_present=complete_slices,
        evaluator_backed=True,
    )
    black = replace(self, color="black")
    views, summaries, transcript = score_views(
        (self, black), partition=partition
    )
    return PromotionReport(
        format=PROMOTION_REPORT_FORMAT,
        version=PROMOTION_REPORT_VERSION,
        partition=partition,
        partition_seed_sha256="a" * 64,
        bootstrap_seed=bootstrap_seed,
        ensemble_release_sha256="b" * 64,
        calibration_sha256="c" * 64,
        training_frequency_sha256="d" * 64,
        move_examples=2,
        promotion_gate_complete=complete_slices,
        unsupported_protocol_slices=(
            ()
            if complete_slices
            else (
                "macro-player-game-per-rule",
                "per-rule-complete-game-support",
            )
        ),
        transcript=transcript,
        views=views,
        player_game_summaries=summaries,
    )


class ValidationGateTests(unittest.TestCase):
    def test_trigger_agent_and_evaluator_slices_are_consumed(self) -> None:
        decision = decide_validation_gate(
            promotion_report(complete_slices=True)
        )
        results = {item.gate_id: item for item in decision.results}
        for gate_id in (
            "protocol-slice.trigger-opportunity-metrics",
            "protocol-slice.agent-profile-metrics",
            (
                "protocol-slice."
                "evaluator-backed-versus-synchronous-family-metrics"
            ),
            "prepared-182.slices.trigger-rule-domain",
            "prepared-182.slices.agent-profiles",
            "prepared-182.slices.evaluator-modes",
            "browser-180.slices.trigger-rule-domain",
            "browser-180.slices.agent-profiles",
            "browser-180.slices.evaluator-modes",
        ):
            self.assertEqual(results[gate_id].status, "passed", gate_id)

    def test_exhaustive_decision_fails_closed_on_core_evidence_gaps(self) -> None:
        decision = decide_validation_gate(promotion_report())
        self.assertFalse(decision.passed)
        self.assertGreater(decision.missing_count, 0)
        results = {item.gate_id: item for item in decision.results}
        self.assertEqual(
            results[
                "prepared-182.white.parameters.whole-object-accuracy"
            ].status,
            "missing",
        )
        self.assertEqual(
            results[
                "prepared-182.white.rules.macro-top3"
            ].status,
            "missing",
        )
        self.assertEqual(
            results["prepared-182.white.absolute.negative_log_likelihood"].status,
            "passed",
        )
        self.assertEqual(
            results[
                "prepared-182.white.comparator.frequency.top1"
            ].status,
            "failed",
        )
        self.assertEqual(len(results), len(decision.results))

    def test_wrong_partition_and_bootstrap_seed_are_explicit_failures(self) -> None:
        decision = decide_validation_gate(
            promotion_report(partition="test", bootstrap_seed=7)
        )
        results = {item.gate_id: item for item in decision.results}
        self.assertEqual(results["identity.partition"].status, "failed")
        self.assertEqual(results["identity.bootstrap-seed"].status, "failed")
        self.assertFalse(decision.passed)

    def test_bootstrap_is_frozen_deterministic_and_complete_game_paired(self) -> None:
        first = decide_validation_gate(promotion_report())
        second = decide_validation_gate(promotion_report())
        self.assertEqual(first.bootstrap, second.bootstrap)
        self.assertEqual(first.bootstrap["seed"], BOOTSTRAP_SEED)
        self.assertEqual(
            first.bootstrap["replicates"], BOOTSTRAP_REPLICATES
        )
        results = {item.gate_id: item for item in first.results}
        self.assertEqual(
            results["prepared-182.white.bootstrap.top1-vs-symbolic"].status,
            "passed",
        )
        self.assertEqual(
            results["prepared-182.white.bootstrap.nll-vs-symbolic"].status,
            "passed",
        )

    def test_publishes_canonical_bound_report_and_recovers_exact_pair(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            report_path = root / "validation-report.json"
            decision_path = root / "validation-decision.json"
            report_ref, decision_ref, decision = publish_validation_gate(
                promotion_report(), report_path, decision_path
            )
            report_value = json.loads(report_path.read_bytes())
            decision_value = json.loads(decision_path.read_bytes())
            self.assertEqual(
                hashlib.sha256(report_path.read_bytes()).hexdigest(),
                report_ref.sha256,
            )
            self.assertEqual(
                hashlib.sha256(decision_path.read_bytes()).hexdigest(),
                decision_ref.sha256,
            )
            self.assertEqual(
                decision_value["validation_report"]["sha256"],
                report_ref.sha256,
            )
            self.assertEqual(
                report_value["bindings"]["calibration_sha256"], "c" * 64
            )
            self.assertEqual(decision_value["passed"], decision.passed)
            self.assertTrue(report_path.read_bytes().endswith(b"\n"))
            recovered_report, recovered_decision, recovered = (
                publish_validation_gate(
                    promotion_report(), report_path, decision_path
                )
            )
            self.assertEqual(recovered_report, report_ref)
            self.assertEqual(recovered_decision, decision_ref)
            self.assertEqual(recovered, decision)

    def test_pair_retry_converges_after_late_publication_failure(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            report_path = root / "validation-report.json"
            decision_path = root / "validation-decision.json"
            calls = 0

            def fail_after_second_publication(
                path: Path,
                payload: bytes,
                *,
                label: str,
            ) -> None:
                nonlocal calls
                calls += 1
                publish_bytes_durable_exact(path, payload, label=label)
                if calls == 2:
                    raise OSError("simulated late publication failure")

            with mock.patch.object(
                validation_gate,
                "publish_bytes_durable_exact",
                side_effect=fail_after_second_publication,
            ):
                with self.assertRaisesRegex(OSError, "simulated late"):
                    publish_validation_gate(
                        promotion_report(), report_path, decision_path
                    )

            self.assertTrue(report_path.is_file())
            self.assertTrue(decision_path.is_file())
            publish_validation_gate(
                promotion_report(), report_path, decision_path
            )

            decision_path.write_bytes(b"competitor\n")
            with self.assertRaisesRegex(ValueError, "do not match"):
                publish_validation_gate(
                    promotion_report(), report_path, decision_path
                )
            self.assertTrue(report_path.is_file())
            self.assertEqual(decision_path.read_bytes(), b"competitor\n")

    def test_pair_rejects_the_same_output_before_publication(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            for report, decision, expected in (
                ("validation.json", "validation.json", "must be distinct"),
                (
                    "Validation-Report.json",
                    "validation-report.JSON",
                    "must be distinct",
                ),
                (
                    "validation-report.json:secret",
                    "validation-decision.json",
                    "safe basename",
                ),
                ("NUL", "validation-decision.json", "safe basename"),
            ):
                with self.subTest(report=report, decision=decision):
                    report_output = root / report
                    decision_output = root / decision
                    with self.assertRaisesRegex(ValueError, expected):
                        publish_validation_gate(
                            promotion_report(),
                            report_output,
                            decision_output,
                        )
                    self.assertEqual(list(root.iterdir()), [])

    def test_missing_view_is_recorded_instead_of_raising_or_passing(self) -> None:
        report = promotion_report()
        without_browser = replace(
            report,
            views={PREPARED_VIEW: report.views[PREPARED_VIEW]},
        )
        decision = decide_validation_gate(without_browser)
        results = {item.gate_id: item for item in decision.results}
        self.assertFalse(decision.passed)
        self.assertEqual(
            results[f"{BROWSER_VIEW}.white.absolute.negative_log_likelihood"].status,
            "missing",
        )
        self.assertEqual(
            results["structure.browser.class_count"].status, "missing"
        )

    def test_parser_exposes_no_partition_threshold_or_bootstrap_override(self) -> None:
        parser = build_parser()
        destinations = {action.dest for action in parser._actions}
        self.assertNotIn("partition", destinations)
        self.assertNotIn("bootstrap_seed", destinations)
        self.assertNotIn("threshold", destinations)

    def test_document_loader_rejects_digest_and_noncanonical_json(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            report_path = root / "validation-report.json"
            decision_path = root / "validation-decision.json"
            report_ref, _decision_ref, _decision = publish_validation_gate(
                promotion_report(), report_path, decision_path
            )
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_validation_gate_document(
                    ContentAddressedFile(report_ref.path, "f" * 64),
                    "drawbacktrainer-validation-gate-report",
                )
            value = json.loads(report_path.read_bytes())
            noncanonical = json.dumps(value).encode("utf-8")
            changed = root / "changed.json"
            changed.write_bytes(noncanonical)
            with self.assertRaisesRegex(ValueError, "not canonical"):
                load_validation_gate_document(
                    ContentAddressedFile(
                        changed, hashlib.sha256(noncanonical).hexdigest()
                    ),
                    "drawbacktrainer-validation-gate-report",
                )


if __name__ == "__main__":
    unittest.main()
