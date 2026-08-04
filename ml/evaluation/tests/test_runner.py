from __future__ import annotations

import unittest
from pathlib import Path
import hashlib
import io
import json
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from ml.evaluation.cli import (
    _abort_sidecar_preserving_failure,
    _evaluate,
    _finalize_sidecar_preserving_failure,
    _json_value,
    _require_checkpoint_corpus_provenance,
    _split_manifest_mapping,
    _write_report_atomic_no_clobber,
    main as evaluation_main,
)
from ml.evaluation.calibration import CalibrationExample
from ml.evaluation.calibration_release import CalibrationObservation
from ml.evaluation.runner import (
    EvaluationDataError,
    decode_predicted_parameter,
    decode_true_parameter,
    evaluate_held_out,
    load_rule_families,
    read_ndjson_stream,
)
from ml.evaluation.splits import SplitManifest
from ml.training.drawback_ml.features import MOVE_VOCABULARY_SIZE, encode_move
from ml.training.drawback_ml.inference import InferenceOutput
from ml.training.drawback_ml.durable_publish import publish_bytes_durable_exact


def row(
    color: str,
    drawback: str,
    parameters: object,
    *,
    ply: int,
    move: str,
) -> dict[str, object]:
    return {
        "gameId": "game-1",
        "seed": 101,
        "fenBefore": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "move": move,
        "moveNumber": ply // 2 + 1,
        "ply": ply,
        "playerColor": color,
        "historySan": [],
        "trueDrawback": drawback,
        "hiddenParameters": parameters,
        "drawbackInternalState": None,
        "ordinaryLegalMoves": [move],
        "drawbackLegalMoves": [move],
        "ruleTriggered": ply == 4,
        "forced": True,
        "clockMs": None,
        "result": {"kind": "active"},
    }


class FakePredictor:
    drawback_vocabulary = ("A", "B")
    parameter_vocabulary = ('{"rank":3}',)
    legal_mask_dimension = MOVE_VOCABULARY_SIZE

    def __init__(self) -> None:
        self.feature_inputs: list[object] = []

    def predict(self, features: object) -> InferenceOutput:
        self.feature_inputs.append(features)
        move = getattr(features, "move")
        legal = [0.0] * self.legal_mask_dimension
        legal[encode_move(move)] = 0.9
        return InferenceOutput(
            white_drawback_probabilities={"A": 0.8, "B": 0.2},
            black_drawback_probabilities={"A": 0.1, "B": 0.9},
            white_parameter_probabilities={"{\"rank\":3}": 1.0},
            black_parameter_probabilities={"{\"rank\":3}": 1.0},
            trigger_probability=0.8 if getattr(features, "ply") == 4 else 0.2,
            legal_mask_probabilities=tuple(legal),
            white_hard_eliminated=(False, True),
            black_hard_eliminated=(True, False),
        )


class HeldOutRunnerTests(unittest.TestCase):
    def test_sidecar_abort_cannot_mask_the_primary_evaluation_failure(self) -> None:
        primary = ValueError("evaluation failed")

        class FailingStream:
            @staticmethod
            def abort() -> None:
                raise OSError("sidecar cleanup failed")

        _abort_sidecar_preserving_failure(FailingStream(), primary)
        self.assertEqual(str(primary), "evaluation failed")
        self.assertIn(
            "sidecar cleanup failed",
            " ".join(getattr(primary, "__notes__", ())),
        )

    def test_sidecar_finalize_failure_remains_primary_when_abort_fails(self) -> None:
        class FailingStream:
            @staticmethod
            def finalize(*, recover_exact: bool = False) -> None:
                if not recover_exact:
                    raise AssertionError("exact recovery was not requested")
                raise OSError("sidecar fsync failed")

            @staticmethod
            def abort() -> None:
                raise OSError("sidecar abort failed")

        with self.assertRaisesRegex(OSError, "sidecar fsync failed") as raised:
            _finalize_sidecar_preserving_failure(FailingStream())

        self.assertIn(
            "sidecar abort failed",
            " ".join(getattr(raised.exception, "__notes__", ())),
        )

    def test_public_parameter_decoders_apply_supervision_contract(self) -> None:
        decoded, unknown = decode_true_parameter(
            '{"rank":3,"seed":99}',
            frozenset({'{"rank":3}'}),
        )
        self.assertEqual(decoded, {"rank": 3})
        self.assertFalse(unknown)
        seed_only, seed_unknown = decode_true_parameter(
            '{"seed":99}',
            frozenset({'{"rank":3}'}),
        )
        self.assertIsNone(seed_only)
        self.assertFalse(seed_unknown)
        self.assertEqual(
            decode_predicted_parameter({'{"rank":3}': 1.0}),
            {"rank": 3},
        )

    def test_reads_authenticated_binary_stream_without_reopening_a_path(self) -> None:
        source = io.BytesIO(b'{"value":1}\n\n{"value":2}\n')
        self.assertEqual(
            list(read_ndjson_stream(source, label="pinned-validation")),
            [{"value": 1}, {"value": 2}],
        )
        self.assertFalse(source.closed)
        invalid = io.BytesIO(b'{"value":"\\xff"}\n'.replace(b"\\xff", b"\xff"))
        with self.assertRaisesRegex(EvaluationDataError, "UTF-8 JSON"):
            list(read_ndjson_stream(invalid, label="pinned-validation"))

    def setUp(self) -> None:
        self.rows = [
            row("white", "A", {"square": "h8"}, ply=4, move="e2e4"),
            row("black", "B", {"rank": 3}, ply=9, move="e7e5"),
        ]
        self.manifest = SplitManifest(
            train=(1,),
            validation=(101,),
            test=(201,),
        )

    def test_scores_only_the_mover_head_and_counts_unknown_parameters(self) -> None:
        predictor = FakePredictor()
        report = evaluate_held_out(
            self.rows,
            predictor=predictor,
            split="validation",
            manifest=self.manifest,
            rule_families={"A": "square", "B": "rank"},
        )
        self.assertEqual(report.move_examples, 2)
        self.assertEqual(report.white_drawback.top_1_accuracy, 1.0)
        self.assertEqual(report.black_drawback.top_1_accuracy, 1.0)
        self.assertEqual(report.white_drawback.count, 1)
        self.assertEqual(report.black_drawback.count, 1)
        self.assertEqual(
            dict(report.white_drawback.accuracy_per_rule_family),
            {"square": 1.0},
        )
        self.assertEqual(
            dict(report.black_drawback.accuracy_per_rule_family),
            {"rank": 1.0},
        )
        self.assertEqual(report.white_unscorable_parameter_examples, 1)
        self.assertEqual(report.black_unscorable_parameter_examples, 0)
        self.assertIsNone(report.white_drawback.hidden_parameter_accuracy)
        self.assertEqual(report.black_drawback.hidden_parameter_accuracy, 1.0)
        self.assertEqual(report.trigger.accuracy, 1.0)
        self.assertEqual(report.legal_mask.exact_match_accuracy, 1.0)
        self.assertEqual(
            report.white_drawback.probability_diagnostics.hard_mask_checked_count,
            1,
        )
        self.assertEqual(
            report.black_drawback.probability_diagnostics.hard_mask_checked_count,
            1,
        )
        self.assertEqual(
            report.white_drawback.probability_diagnostics.hard_elimination_violation_count,
            1,
        )
        self.assertEqual(
            report.black_drawback.probability_diagnostics.hard_elimination_violation_count,
            1,
        )
        self.assertEqual(
            dict(report.white_drawback.accuracy_after_moves),
            {5: 1.0, 10: 1.0, 15: 1.0, 20: 1.0},
        )
        self.assertTrue(
            all(type(value).__name__ == "FeatureRecord" for value in predictor.feature_inputs)
        )

    def test_does_not_score_an_unobserved_opponent_in_one_sided_games(self) -> None:
        black_row = row("black", "B", {"rank": 3}, ply=1, move="e7e5")
        black_row["gameId"] = "game-2"
        report = evaluate_held_out(
            [
                row("white", "A", {"rank": 3}, ply=0, move="e2e4"),
                black_row,
            ],
            predictor=FakePredictor(),
            split="validation",
            manifest=self.manifest,
            rule_families={"A": "rank", "B": "rank"},
            game_assignments={
                "game-1": ("A", "B"),
                "game-2": ("A", "B"),
            },
        )
        self.assertEqual(report.move_examples, 2)
        self.assertEqual(report.white_drawback.count, 1)
        self.assertEqual(report.black_drawback.count, 1)
        self.assertEqual(report.white_drawback.top_1_accuracy, 1.0)
        self.assertEqual(report.black_drawback.top_1_accuracy, 1.0)
        self.assertEqual(report.white_unscorable_parameter_examples, 0)
        self.assertEqual(report.black_unscorable_parameter_examples, 0)

    def test_streams_authenticated_rows_from_a_one_shot_iterable(self) -> None:
        class OneShot:
            def __init__(self, values: list[dict[str, object]]) -> None:
                self.values = values
                self.iterations = 0

            def __iter__(self):
                self.iterations += 1
                if self.iterations > 1:
                    raise AssertionError("rows were iterated more than once")
                yield from self.values

        rows = OneShot(self.rows)
        report = evaluate_held_out(
            rows,
            predictor=FakePredictor(),
            split="validation",
            manifest=self.manifest,
            game_assignments={"game-1": ("A", "B")},
        )
        self.assertEqual(report.move_examples, 2)
        self.assertEqual(rows.iterations, 1)

    def test_batch_predictor_preserves_report_and_order(self) -> None:
        class BatchPredictor(FakePredictor):
            def __init__(self) -> None:
                super().__init__()
                self.batch_sizes: list[int] = []

            def predict_batch(self, features):
                self.batch_sizes.append(len(features))
                return tuple(self.predict(item) for item in features)

        scalar = evaluate_held_out(
            self.rows,
            predictor=FakePredictor(),
            split="validation",
            manifest=self.manifest,
            game_assignments={"game-1": ("A", "B")},
            batch_size=1,
        )
        predictor = BatchPredictor()
        batched = evaluate_held_out(
            self.rows,
            predictor=predictor,
            split="validation",
            manifest=self.manifest,
            game_assignments={"game-1": ("A", "B")},
            batch_size=2,
        )
        self.assertEqual(batched, scalar)
        self.assertEqual(predictor.batch_sizes, [2])

    def test_requires_exact_checkpoint_manifest_provenance(self) -> None:
        expected = {
            "manifest_sha256": "a" * 64,
            "split": "validation",
            "engine_fingerprint": "stockfish:18:a:b",
            "evaluator_policy_id": "stockfish-bestmove-v1",
            "evaluator_policy_version": 1,
            "rule_ids": ["A", "B"],
            "symbolic_feature_version": 5,
        }

        class Audited:
            def provenance(self) -> dict[str, object]:
                return expected

        class PredictorWithProvenance:
            corpus_provenance = {**expected, "split": "train"}

        _require_checkpoint_corpus_provenance(
            PredictorWithProvenance(),
            Audited(),
        )
        for changed in (
            None,
            {**PredictorWithProvenance.corpus_provenance, "split": "test"},
            {
                **PredictorWithProvenance.corpus_provenance,
                "manifest_sha256": "b" * 64,
            },
        ):
            predictor = PredictorWithProvenance()
            predictor.corpus_provenance = changed
            with self.assertRaisesRegex(ValueError, "provenance"):
                _require_checkpoint_corpus_provenance(predictor, Audited())

    def test_publishes_report_atomically_without_clobbering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.json"
            _write_report_atomic_no_clobber(path, '{"ok":true}\n')
            self.assertEqual(path.read_text(encoding="utf-8"), '{"ok":true}\n')
            with self.assertRaisesRegex(ValueError, "overwrite"):
                _write_report_atomic_no_clobber(path, '{"ok":false}\n')
            self.assertEqual(path.read_text(encoding="utf-8"), '{"ok":true}\n')
            self.assertEqual(list(path.parent.glob("*.tmp")), [])
            recovered = Path(temporary) / "nested" / "report.json"
            _write_report_atomic_no_clobber(
                recovered,
                '{"ok":true}\n',
                recover_exact=True,
            )
            _write_report_atomic_no_clobber(
                recovered,
                '{"ok":true}\n',
                recover_exact=True,
            )
            self.assertEqual(
                recovered.read_text(encoding="utf-8"),
                '{"ok":true}\n',
            )

    def test_calibration_evaluation_recovers_after_third_publication_failure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "model.pt"
            checkpoint.write_bytes(b"checkpoint")
            checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            selection = root / "selection.json"
            selection.write_bytes(b"{}\n")
            selection_sha256 = hashlib.sha256(selection.read_bytes()).hexdigest()
            training_run = root / "training-run.json"
            training_run.write_bytes(b"{}\n")
            training_run_sha256 = hashlib.sha256(
                training_run.read_bytes()
            ).hexdigest()
            report = root / "report.json"
            sidecar = root / "sidecar.ndjson"
            receipt = root / "receipt.json"
            audited = SimpleNamespace(
                manifest_sha256="1" * 64,
                dataset_sha256="2" * 64,
                release_root_sha256="3" * 64,
                corpus_run_id="run-1",
                provenance=lambda: {"manifest_sha256": "1" * 64},
            )
            validation_context = SimpleNamespace(
                metadata={
                    "identity": "drawbacktrainer-validation-partition-v1",
                    "name": "calibration-fit",
                    "seedSha256": "4" * 64,
                },
                game_assignments={},
            )
            selected = SimpleNamespace(
                selected_checkpoint_sha256=checkpoint_sha256,
                provenance=SimpleNamespace(
                    release_root_sha256=audited.release_root_sha256,
                    corpus_run_id=audited.corpus_run_id,
                    private_validation_manifest_sha256=audited.manifest_sha256,
                    validation_dataset_sha256=audited.dataset_sha256,
                ),
            )
            predictor = SimpleNamespace(
                drawback_vocabulary=("A", "B"),
                drawback_loss_objective="legacy",
                checkpoint_seed=7,
                checkpoint_epoch=2,
                training_run_id="run-1",
            )
            arguments = SimpleNamespace(
                batch_size=8,
                validation_partition="calibration-fit",
                calibration_sidecar_output=sidecar,
                selection_artifact=selection,
                selection_sha256=selection_sha256,
                training_run=training_run,
                training_run_sha256=training_run_sha256,
                calibration_receipt_output=receipt,
                output=report,
                catalog=[],
                checkpoint=checkpoint,
                device="cpu",
                dataset=root / "validation.ndjson",
                split="validation",
            )
            example = CalibrationExample(
                (2.0, 0.0),
                0,
                (False, False),
            )

            def evaluate(_rows: object, **kwargs: object) -> dict[str, int]:
                sink = kwargs["calibration_sink"]
                sink(CalibrationObservation("white", example))
                sink(CalibrationObservation("black", example))
                return {"move_examples": 2}

            receipt_attempts = 0

            def publish_receipt(
                output: Path,
                _inputs: object,
                *,
                recover_exact: bool = False,
            ) -> None:
                nonlocal receipt_attempts
                receipt_attempts += 1
                self.assertTrue(recover_exact)
                if receipt_attempts == 1:
                    raise OSError("injected receipt publication failure")
                publish_bytes_durable_exact(
                    output,
                    b"receipt\n",
                    label="calibration receipt",
                )

            with (
                patch(
                    "ml.evaluation.cli._release_evaluation_inputs",
                    return_value=(
                        audited,
                        {"maxPlies": 80},
                        SplitManifest(train=(), validation=(101,), test=()),
                        lambda: audited,
                    ),
                ),
                patch(
                    "ml.evaluation.cli._validation_evaluation_context",
                    return_value=validation_context,
                ),
                patch(
                    "ml.evaluation.cli.load_checkpoint_predictor",
                    return_value=predictor,
                ),
                patch("ml.evaluation.cli._require_checkpoint_corpus_provenance"),
                patch("ml.evaluation.cli.SYMBOLIC_RULE_IDS", ("A", "B")),
                patch(
                    "ml.evaluation.cli.verify_release_selection_bundle",
                    return_value=SimpleNamespace(artifact=selected),
                ),
                patch("ml.evaluation.cli.read_ndjson", return_value=[]),
                patch(
                    "ml.evaluation.cli._filter_validation_partition_rows",
                    return_value=[],
                ),
                patch(
                    "ml.evaluation.cli.load_rule_families",
                    return_value={"A": "one", "B": "two"},
                ),
                patch(
                    "ml.evaluation.cli.evaluate_held_out",
                    side_effect=evaluate,
                ),
                patch(
                    "ml.evaluation.cli.write_calibration_receipt",
                    side_effect=publish_receipt,
                ),
            ):
                with self.assertRaisesRegex(OSError, "injected receipt"):
                    _evaluate(arguments, None)
                self.assertTrue(sidecar.is_file())
                self.assertTrue(report.is_file())
                self.assertFalse(receipt.exists())
                self.assertEqual(_evaluate(arguments, None), 0)

            self.assertEqual(receipt.read_bytes(), b"receipt\n")

    def test_non_finite_metrics_use_strict_json_null(self) -> None:
        value = _json_value(
            {"nll": float("inf"), "loss": float("-inf"), "bad": float("nan")}
        )
        self.assertEqual(value, {"nll": None, "loss": None, "bad": None})
        rendered = json.dumps(value, allow_nan=False)
        self.assertNotIn("Infinity", rendered)
        self.assertNotIn("NaN", rendered)

    def test_lazy_file_replacement_fails_before_report_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            dataset = directory / "validation.ndjson"
            dataset.write_text('{"original":true}\n', encoding="utf-8")
            checkpoint = directory / "model.pt"
            checkpoint.write_bytes(b"checkpoint")
            manifest = directory / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "train": [1],
                        "validation": [2],
                        "test": [3],
                        "maxPlies": 80,
                    }
                ),
                encoding="utf-8",
            )
            output = directory / "report.json"
            provenance = {
                "manifest_sha256": "a" * 64,
                "split": "validation",
                "engine_fingerprint": "stockfish:18:a:b",
                "evaluator_policy_id": "policy",
                "evaluator_policy_version": 1,
                "rule_ids": ["A", "B"],
                "symbolic_feature_version": 5,
            }
            audited = SimpleNamespace(
                dataset_path=dataset.resolve(),
                game_assignments=(("game-1", "A", "B"),),
                provenance=lambda: provenance,
            )
            changed = SimpleNamespace(
                dataset_path=dataset.resolve(),
                game_assignments=audited.game_assignments,
                provenance=lambda: {**provenance, "dataset_sha256": "b" * 64},
            )
            predictor = FakePredictor()
            predictor.drawback_vocabulary = ("A", "B")
            predictor.corpus_provenance = {
                **provenance,
                "split": "train",
            }

            def load_from_verified_bytes(source, **kwargs):
                self.assertEqual(source.read(), b"checkpoint")
                self.assertEqual(kwargs["required_corpus_provenance"], {})
                checkpoint.write_bytes(b"replacement checkpoint")
                source.seek(0)
                self.assertEqual(source.read(), b"checkpoint")
                return predictor

            def consume_and_replace(*_args, **_kwargs):
                dataset.write_text('{"replacement":true}\n', encoding="utf-8")
                return {"measured": True}

            with (
                patch(
                    "ml.evaluation.cli.audit_corpus_split",
                    side_effect=(audited, audited, changed),
                ),
                patch(
                    "ml.evaluation.cli.load_checkpoint_predictor",
                    side_effect=load_from_verified_bytes,
                ),
                patch(
                    "ml.evaluation.cli.load_rule_families",
                    return_value={"A": "one", "B": "two"},
                ),
                patch(
                    "ml.evaluation.cli.evaluate_held_out",
                    side_effect=consume_and_replace,
                ),
                patch("ml.evaluation.cli.SYMBOLIC_RULE_IDS", ("A", "B")),
            ):
                with self.assertRaisesRegex(ValueError, "changed during"):
                    evaluation_main(
                        [
                            str(checkpoint),
                            str(dataset),
                            str(manifest),
                            "--split",
                            "validation",
                            "--output",
                            str(output),
                        ]
                    )
            self.assertFalse(output.exists())

    def test_rejects_train_or_out_of_manifest_data_before_inference(self) -> None:
        predictor = FakePredictor()
        with self.assertRaisesRegex(EvaluationDataError, "validation or test"):
            evaluate_held_out(
                self.rows,
                predictor=predictor,
                split="train",
                manifest=self.manifest,
            )
        outside = [dict(value, seed=999) for value in self.rows]
        with self.assertRaisesRegex(EvaluationDataError, "outside its manifest"):
            evaluate_held_out(
                outside,
                predictor=predictor,
                split="validation",
                manifest=self.manifest,
            )
        self.assertEqual(predictor.feature_inputs, [])

    def test_rejects_unseen_drawbacks_instead_of_mapping_them(self) -> None:
        predictor = FakePredictor()
        unseen = [
            row("white", "UNSEEN", None, ply=4, move="e2e4"),
            self.rows[1],
        ]
        with self.assertRaisesRegex(EvaluationDataError, "unseen drawback"):
            evaluate_held_out(
                unseen,
                predictor=predictor,
                split="validation",
                manifest=self.manifest,
            )

    def test_reads_the_simulator_corpus_manifest_without_weakening_split_checks(
        self,
    ) -> None:
        mapping = _split_manifest_mapping(
            {
                "splitSalt": "drawbacktrainer-v1",
                "splits": {
                    "train": {"seeds": [1]},
                    "validation": {"seeds": [101]},
                    "test": {"seeds": [201]},
                },
            }
        )
        self.assertEqual(
            SplitManifest.from_mapping(mapping),
            self.manifest,
        )
        with self.assertRaisesRegex(ValueError, "splitSalt"):
            _split_manifest_mapping(
                {
                    "splitSalt": "different",
                    "splits": {
                        "train": {"seeds": [1]},
                        "validation": {"seeds": [101]},
                        "test": {"seeds": [201]},
                    },
                }
            )

    def test_reads_schema_five_evaluator_manifest_and_requires_uniform_coverage(
        self,
    ) -> None:
        evaluator_manifest = {
            "schemaVersion": 5,
            "splitSalt": "drawbacktrainer-v1",
            "evaluatorCoverage": "uniform-required",
            "evaluatorPolicyId": "stockfish-bestmove-v1",
            "evaluatorPolicyVersion": 1,
            "evaluatorRequestSchemaVersion": 1,
            "evaluatorCacheSchemaVersion": 1,
            "evaluatorSearchLimit": {"kind": "nodes", "value": 10_000},
            "engineFingerprint": "stockfish:17:options",
            "engineBinarySha256": "ab" * 32,
            "splits": {
                "train": {"seeds": [1]},
                "validation": {"seeds": [101]},
                "test": {"seeds": [201]},
            },
        }
        self.assertEqual(
            SplitManifest.from_mapping(
                _split_manifest_mapping(evaluator_manifest)
            ),
            self.manifest,
        )
        with self.assertRaisesRegex(ValueError, "uniform evaluator"):
            _split_manifest_mapping(
                {**evaluator_manifest, "evaluatorCoverage": "mixed"}
            )
        with self.assertRaisesRegex(ValueError, "schemaVersion"):
            _split_manifest_mapping(
                {**evaluator_manifest, "schemaVersion": 7}
            )
        invalid_provenance = {
            "evaluatorPolicyId": "",
            "evaluatorPolicyVersion": 0,
            "evaluatorRequestSchemaVersion": 1.0,
            "evaluatorCacheSchemaVersion": True,
            "evaluatorSearchLimit": {"nodes": 10_000},
            "engineFingerprint": " ",
            "engineBinarySha256": "not-a-digest",
        }
        for key, invalid in invalid_provenance.items():
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, key):
                    _split_manifest_mapping(
                        {**evaluator_manifest, key: invalid}
                    )

    def test_rejects_noncanonical_evaluator_search_limits(self) -> None:
        base = {
            "schemaVersion": 5,
            "splitSalt": "drawbacktrainer-v1",
            "evaluatorCoverage": "uniform-required",
            "evaluatorPolicyId": "stockfish-bestmove-v1",
            "evaluatorPolicyVersion": 1,
            "evaluatorRequestSchemaVersion": 1,
            "evaluatorCacheSchemaVersion": 1,
            "engineFingerprint": "stockfish:17:options",
            "engineBinarySha256": "ab" * 32,
            "splits": {
                "train": {"seeds": [1]},
                "validation": {"seeds": [101]},
                "test": {"seeds": [201]},
            },
        }
        invalid_limits = [
            None,
            {"kind": "nodes", "value": 0},
            {"kind": "nodes", "value": True},
            {"kind": "milliseconds", "value": 1},
            {"kind": ["nodes"], "value": 1},
            {"kind": "depth", "value": 1, "extra": False},
        ]
        for limit in invalid_limits:
            with self.subTest(limit=limit):
                with self.assertRaisesRegex(
                    ValueError, "evaluatorSearchLimit"
                ):
                    _split_manifest_mapping(
                        {**base, "evaluatorSearchLimit": limit}
                    )

    def test_loads_rule_families_from_the_canonical_catalog_envelope(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "catalog.json"
            path.write_text(
                json.dumps(
                    {
                        "catalogVersion": 1,
                        "entries": [
                            {"id": "vegan", "ruleFamily": "forbidden-capture"},
                            {"id": "truant", "ruleFamily": "history"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                load_rule_families([path]),
                {"vegan": "forbidden-capture", "truant": "history"},
            )
            with self.assertRaisesRegex(
                EvaluationDataError, "does not exist"
            ):
                load_rule_families([Path(temporary) / "missing.json"])


if __name__ == "__main__":
    unittest.main()
