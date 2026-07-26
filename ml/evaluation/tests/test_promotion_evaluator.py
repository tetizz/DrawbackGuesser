from __future__ import annotations

from dataclasses import replace
import math
from typing import Sequence
import unittest

from ml.evaluation.promotion_evaluator import (
    BROWSER_VIEW,
    PREPARED_VIEW,
    SYSTEM_NAMES,
    PromotionTemperatures,
    _TranscriptBuilder,
    _calibration_fusion_policy,
    predict_all_systems_batch,
    score_views,
)
from ml.training.drawback_ml.rank_preserving_fusion import (
    RANK_PRESERVING_FUSION_METHOD,
)
from ml.training.drawback_ml.inference import InferenceOutput
from ml.training.drawback_ml.records import (
    EvaluationMetadata,
    FeatureRecord,
    PublicEvaluatorConstraint,
    TrainingExample,
)
from ml.training.drawback_ml.symbolic_schema import SYMBOLIC_RULE_IDS


FUSION_ALPHA = 0.5
FUSION_SELECTION_SHA256 = "ab" * 32


def frequency() -> dict[str, dict[str, float]]:
    ordered = {
        rule_id: float(index + 1)
        for index, rule_id in enumerate(SYMBOLIC_RULE_IDS)
    }
    return {"white": dict(ordered), "black": dict(ordered)}


def feature(
    *,
    color: str = "white",
    ply: int = 8,
    eliminated_id: str = "vegan",
) -> FeatureRecord:
    mask = tuple(rule_id == eliminated_id for rule_id in SYMBOLIC_RULE_IDS)
    weights = tuple(
        0.0 if eliminated else float(index + 1)
        for index, eliminated in enumerate(mask)
    )
    total = math.fsum(weights)
    prior = tuple(value / total for value in weights)
    return FeatureRecord(
        fen_before="8/8/8/8/8/8/8/8 w - - 0 1",
        move="a2a3",
        move_number=5,
        ply=ply,
        player_color=color,
        history_san=("a3",),
        ordinary_legal_moves=("a2a3",),
        clock_ms=None,
        symbolic_feature_version=6,
        symbolic_white_rule_probabilities=prior,
        symbolic_black_rule_probabilities=prior,
        symbolic_white_eliminated=mask,
        symbolic_black_eliminated=mask,
        public_evaluator_constraint=None,
    )


def example(
    game_id: str,
    truth: str,
    *,
    color: str = "white",
    ply: int = 8,
    parameter_token: str | None = None,
    parameter_observed: bool = True,
    rule_triggered: bool = False,
    agent_id: str | None = None,
    agent_style: str | None = None,
    agent_strength: int | None = None,
    evaluator_backed: bool = False,
) -> TrainingExample:
    features = feature(color=color, ply=ply)
    if evaluator_backed:
        features = replace(
            features,
            public_evaluator_constraint=PublicEvaluatorConstraint(
                provider="uci-best-move",
                policy_id="stockfish-bestmove-v1",
                position_key="position",
                request_digest="ab" * 32,
                best_move_uci="a2a3",
                engine_fingerprint="stockfish:18:test",
            ),
        )
    return TrainingExample(
        game_id=game_id,
        seed=101,
        features=features,
        white_drawback=truth if color == "white" else "checkers",
        black_drawback=truth if color == "black" else "checkers",
        white_parameters=parameter_token if color == "white" else None,
        black_parameters=parameter_token if color == "black" else None,
        white_parameters_observed=(
            parameter_observed if color == "white" else True
        ),
        black_parameters_observed=(
            parameter_observed if color == "black" else True
        ),
        rule_triggered=rule_triggered,
        drawback_legal_moves=("a2a3",),
        evaluation=EvaluationMetadata(
            bot_agent_id=agent_id,
            bot_style=agent_style,
            bot_strength=agent_strength,
            agent_metadata_present=agent_id is not None,
        ),
    )


class FakeMember:
    drawback_vocabulary = SYMBOLIC_RULE_IDS
    parameter_vocabulary = ('{"rank":1}', '{"rank":2}')

    def __init__(self, seed: int, scale: float) -> None:
        self.checkpoint_seed = seed
        self.scale = scale
        self.calls = 0

    def predict_batch(
        self,
        features: Sequence[FeatureRecord],
    ) -> tuple[InferenceOutput, ...]:
        self.calls += 1
        residual = tuple(
            self.scale * index / len(SYMBOLIC_RULE_IDS)
            for index in range(len(SYMBOLIC_RULE_IDS))
        )
        outputs = []
        for item in features:
            outputs.append(
                InferenceOutput(
                    white_drawback_probabilities={},
                    black_drawback_probabilities={},
                    white_parameter_probabilities={
                        '{"rank":1}': 0.8,
                        '{"rank":2}': 0.2,
                    },
                    black_parameter_probabilities={
                        '{"rank":1}': 0.8,
                        '{"rank":2}': 0.2,
                    },
                    trigger_probability=0.5,
                    legal_mask_probabilities=(),
                    white_neural_residual_logits=residual,
                    black_neural_residual_logits=residual,
                    white_hard_eliminated=(
                        item.symbolic_white_eliminated
                    ),
                    black_hard_eliminated=(
                        item.symbolic_black_eliminated
                    ),
                )
            )
        return tuple(outputs)


def members() -> tuple[FakeMember, FakeMember, FakeMember]:
    return (
        FakeMember(20260811, 0.5),
        FakeMember(20260812, 1.0),
        FakeMember(20260813, 1.5),
    )


class PromotionEvaluatorTest(unittest.TestCase):
    def test_predicts_every_frozen_system_from_one_member_pass(self) -> None:
        predictors = members()
        rows = predict_all_systems_batch(
            members=predictors,
            examples=(example("game-a", "checkers"),),
            training_frequency=frequency(),
            temperatures=PromotionTemperatures(white=2.0, black=3.0),
            fusion_alpha=FUSION_ALPHA,
        )
        self.assertEqual(tuple(member.calls for member in predictors), (1, 1, 1))
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(tuple(row.probabilities), SYSTEM_NAMES)
        vegan_index = SYMBOLIC_RULE_IDS.index("vegan")
        for probabilities in row.probabilities.values():
            self.assertEqual(probabilities[vegan_index], 0.0)
            self.assertAlmostEqual(math.fsum(probabilities), 1.0, places=14)
        self.assertNotEqual(
            row.probabilities["calibrated-ensemble"],
            row.probabilities["uncalibrated-ensemble"],
        )
        expected_residual = tuple(
            index / len(SYMBOLIC_RULE_IDS)
            for index in range(len(SYMBOLIC_RULE_IDS))
        )
        for actual, expected in zip(
            row.member_residual_logits[1],
            expected_residual,
            strict=True,
        ):
            self.assertAlmostEqual(actual, expected)

    def test_scores_parameter_accuracy_and_unscorable_coverage(self) -> None:
        rows = predict_all_systems_batch(
            members=members(),
            examples=(
                example(
                    "correct",
                    "checkers",
                    parameter_token='{"rank":1}',
                ),
                example(
                    "incorrect",
                    "truant",
                    color="black",
                    parameter_token='{"rank":2}',
                ),
                example(
                    "unknown",
                    "spice-of-life",
                    parameter_token='{"rank":3}',
                ),
                example(
                    "unobserved",
                    "gambler",
                    parameter_token='{"rank":1}',
                    parameter_observed=False,
                ),
            ),
            training_frequency=frequency(),
            temperatures=PromotionTemperatures(1.5, 2.0),
            fusion_alpha=FUSION_ALPHA,
        )
        views, _summaries, _transcript = score_views(
            rows,
            partition="validation-gate",
        )
        prepared = views[PREPARED_VIEW]
        white = prepared.calibrated_parameters["white"]
        black = prepared.calibrated_parameters["black"]
        self.assertEqual(white.eligible_examples, 3)
        self.assertEqual(white.scorable_examples, 1)
        self.assertEqual(white.unscorable_examples, 2)
        self.assertEqual(white.coverage, 1 / 3)
        self.assertEqual(white.unscorable_rate, 2 / 3)
        self.assertEqual(white.whole_object_accuracy, 1.0)
        self.assertEqual(white.component_accuracy, 1.0)
        self.assertEqual(
            dict(white.unscorable_by_reason),
            {"out-of-vocabulary": 1, "unobserved": 1},
        )
        self.assertEqual(black.eligible_examples, 1)
        self.assertEqual(black.scorable_examples, 1)
        self.assertEqual(black.whole_object_accuracy, 0.0)
        self.assertEqual(black.component_accuracy, 0.0)
        calibrated = prepared.systems["calibrated-ensemble"]
        self.assertEqual(
            calibrated.white.hidden_parameter_accuracy,
            1.0,
        )
        self.assertEqual(
            calibrated.black.hidden_parameter_accuracy,
            0.0,
        )
        self.assertIn(
            "agent-profile metrics",
            prepared.unsupported_protocol_slices,
        )

    def test_scores_prepared_and_browser_views_in_one_pass(self) -> None:
        predictors = members()
        rows = predict_all_systems_batch(
            members=predictors,
            examples=(
                example("game-a", "checkers", ply=8),
                example("game-b", "hand-and-gigabrain", ply=10),
            ),
            training_frequency=frequency(),
            temperatures=PromotionTemperatures(white=2.0, black=3.0),
            fusion_alpha=FUSION_ALPHA,
        )
        views, summaries, transcript = score_views(
            iter(rows),
            partition="validation-gate",
            rule_families={"checkers": "forced-capture"},
        )
        prepared = views[PREPARED_VIEW]
        browser = views[BROWSER_VIEW]
        self.assertEqual(len(prepared.class_ids), 182)
        self.assertEqual(len(browser.class_ids), 180)
        self.assertEqual(prepared.scored_examples, 2)
        self.assertEqual(prepared.unscorable_examples, 0)
        self.assertEqual(browser.scored_examples, 1)
        self.assertEqual(browser.unscorable_examples, 1)
        self.assertEqual(browser.unscorable_player_games, 1)
        self.assertEqual(
            browser.unavailable_rule_ids,
            ("hand-and-gigabrain", "ichtyophobe"),
        )
        self.assertEqual(
            prepared.systems["calibrated-ensemble"].white.count,
            2,
        )
        self.assertEqual(
            browser.systems["calibrated-ensemble"].white.count,
            1,
        )
        self.assertIsNone(prepared.systems["calibrated-ensemble"].black)
        self.assertEqual(len(summaries), 24)
        self.assertTrue(all(summary.example_count == 1 for summary in summaries))
        self.assertEqual(transcript.record_count, 2)
        self.assertEqual(transcript.first_record, ("game-a", "white", 5))
        self.assertEqual(transcript.last_record, ("game-b", "white", 6))
        repeated = score_views(
            iter(rows),
            partition="validation-gate",
        )[2]
        reversed_transcript = score_views(
            reversed(rows),
            partition="validation-gate",
        )[2]
        self.assertEqual(transcript.sha256, repeated.sha256)
        self.assertNotEqual(transcript.sha256, reversed_transcript.sha256)

    def test_calibration_is_the_only_fusion_policy_carrier(self) -> None:
        calibration = {
            "identity": {
                "fusion_selection_sha256": FUSION_SELECTION_SHA256,
                "selected_alpha": FUSION_ALPHA,
            },
            "method": {
                "fusion": RANK_PRESERVING_FUSION_METHOD,
                "selected_alpha": FUSION_ALPHA,
            },
        }
        self.assertEqual(
            _calibration_fusion_policy(calibration),
            (FUSION_ALPHA, FUSION_SELECTION_SHA256),
        )
        for changed in (
            {
                **calibration,
                "method": {
                    **calibration["method"],
                    "selected_alpha": 0.25,
                },
            },
            {
                **calibration,
                "identity": {
                    **calibration["identity"],
                    "fusion_selection_sha256": "not-a-digest",
                },
            },
        ):
            with self.assertRaisesRegex(
                ValueError,
                "fusion selection policy is invalid",
            ):
                _calibration_fusion_policy(changed)

    def test_production_transcript_binds_selected_fusion_policy(self) -> None:
        selected = _TranscriptBuilder(
            "validation-gate",
            (FUSION_SELECTION_SHA256, FUSION_ALPHA),
        ).finish()
        changed_alpha = _TranscriptBuilder(
            "validation-gate",
            (FUSION_SELECTION_SHA256, 0.25),
        ).finish()
        changed_selection = _TranscriptBuilder(
            "validation-gate",
            ("cd" * 32, FUSION_ALPHA),
        ).finish()
        self.assertNotEqual(selected.sha256, changed_alpha.sha256)
        self.assertNotEqual(selected.sha256, changed_selection.sha256)

    def test_evaluation_metadata_cannot_change_model_inputs_or_predictions(
        self,
    ) -> None:
        first = example(
            "first",
            "checkers",
            rule_triggered=False,
            agent_id="random-legal",
            agent_style="random",
            agent_strength=100,
        )
        second = replace(
            first,
            game_id="second",
            rule_triggered=True,
            evaluation=EvaluationMetadata(
                bot_agent_id="human-like-strong",
                bot_style="human-like",
                bot_strength=2000,
                agent_metadata_present=True,
            ),
        )
        self.assertEqual(first.features, second.features)
        first_row = predict_all_systems_batch(
            members=members(),
            examples=(first,),
            training_frequency=frequency(),
            temperatures=PromotionTemperatures(1.5, 2.0),
            fusion_alpha=FUSION_ALPHA,
        )[0]
        second_row = predict_all_systems_batch(
            members=members(),
            examples=(second,),
            training_frequency=frequency(),
            temperatures=PromotionTemperatures(1.5, 2.0),
            fusion_alpha=FUSION_ALPHA,
        )[0]
        self.assertEqual(first_row.probabilities, second_row.probabilities)
        self.assertEqual(
            first_row.member_residual_logits,
            second_row.member_residual_logits,
        )
        self.assertNotEqual(first_row.bot_agent_id, second_row.bot_agent_id)
        self.assertNotEqual(first_row.rule_triggered, second_row.rule_triggered)

    def test_reports_trigger_agent_and_evaluator_slices(self) -> None:
        rows = predict_all_systems_batch(
            members=members(),
            examples=(
                example(
                    "sync",
                    "checkers",
                    rule_triggered=True,
                    agent_id="random-legal",
                    agent_style="random",
                    agent_strength=100,
                ),
                example(
                    "evaluator",
                    "checkers",
                    color="black",
                    agent_id="human-like-strong",
                    agent_style="human-like",
                    agent_strength=2000,
                    evaluator_backed=True,
                ),
            ),
            training_frequency=frequency(),
            temperatures=PromotionTemperatures(1.5, 2.0),
            fusion_alpha=FUSION_ALPHA,
        )
        views, _summaries, _transcript = score_views(
            rows,
            partition="validation-gate",
        )
        prepared = views[PREPARED_VIEW]
        self.assertTrue(prepared.evaluation_slices_complete)
        self.assertEqual(prepared.unsupported_protocol_slices, ())
        self.assertEqual(
            prepared.rule_opportunities["white"]["checkers"].support_examples,
            1,
        )
        self.assertEqual(
            prepared.rule_opportunities["white"][
                "checkers"
            ].trigger_opportunities,
            1,
        )
        self.assertEqual(
            set(prepared.agent_profiles),
            {"human-like-strong", "random-legal"},
        )
        self.assertEqual(
            prepared.evaluator_modes["synchronous"].example_count,
            1,
        )
        self.assertEqual(
            prepared.evaluator_modes["evaluator-backed"].example_count,
            1,
        )

    def test_rejects_member_mask_disagreement(self) -> None:
        predictors = list(members())
        original = predictors[2].predict_batch

        def inconsistent(
            features: Sequence[FeatureRecord],
        ) -> tuple[InferenceOutput, ...]:
            outputs = original(features)
            wrong = tuple(False for _ in SYMBOLIC_RULE_IDS)
            return tuple(
                replace(output, white_hard_eliminated=wrong)
                for output in outputs
            )

        predictors[2].predict_batch = inconsistent  # type: ignore[method-assign]
        with self.assertRaisesRegex(
            ValueError,
            "hard mask disagrees",
        ):
            predict_all_systems_batch(
                members=predictors,
                examples=(example("game-a", "checkers"),),
                training_frequency=frequency(),
                temperatures=PromotionTemperatures(1.0, 1.0),
                fusion_alpha=FUSION_ALPHA,
            )

    def test_rejects_reordered_training_frequency(self) -> None:
        invalid = frequency()
        invalid["white"] = dict(reversed(tuple(invalid["white"].items())))
        with self.assertRaisesRegex(ValueError, "reorders classes"):
            predict_all_systems_batch(
                members=members(),
                examples=(example("game-a", "checkers"),),
                training_frequency=invalid,
                temperatures=PromotionTemperatures(1.0, 1.0),
                fusion_alpha=FUSION_ALPHA,
            )


if __name__ == "__main__":
    unittest.main()
