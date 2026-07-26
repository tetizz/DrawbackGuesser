from __future__ import annotations

from copy import deepcopy
import inspect
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import _bootstrap  # noqa: F401

from capturable_fixture import capturable_row
from drawback_ml.capturable_blend_contract import (
    ComponentPredictions,
    _EXPECTED_INPUTS,
)
from drawback_ml.capturable_fixed_blend import (
    _parser,
    main,
    run_fixed_blend_confirmation,
)
from drawback_ml.capturable_fixed_blend_contract import (
    BOOTSTRAP_DRAWS_PER_REPLICATE,
    BOOTSTRAP_LOWER_INDEX,
    BOOTSTRAP_REPLICATES,
    CONFIRMATION_TEST_FILE,
    CONFIRMATION_TRACE_FILE,
    CONSUMPTION_FILE,
    CORPUS_RECEIPT_FILE,
    FIXED_TREATMENT_WEIGHT,
    FIXED_VALIDATION_PREDICTIONS_SHA256,
    GRID_EXECUTION_REVISION,
    PRIOR_REGISTRY_SHA256,
    REPORT_PREFIX,
    BootstrapResult,
    PairedTop1,
    _reduce_to_index,
    _splitmix64_next,
    audit_confirmation_rows,
    fixed_paired_bootstrap,
    fixed_release_checks,
    fixed_validation_candidate,
    load_fixed_blend_confirmation,
    paired_game_top1_deltas,
)
from drawback_ml.capturable_fixed_corpus import CorpusVerification
from drawback_ml.capturable_fixed_schedule import (
    EXPECTED_FIXED_ASSIGNMENTS,
)
from drawback_ml.capturable_records import (
    CAPTURABLE_RULE_COUNT,
    CAPTURABLE_RULE_IDS,
    CapturableDatasetError,
    parse_capturable_dataset_row,
)


def _metrics(
    *,
    top1: float = 0.30,
    nll: float = 2.0,
) -> dict[str, object]:
    hybrid = {
        "accuracy_after_moves": {
            "5": 0.10,
            "10": 0.20,
            "15": 0.30,
            "20": 0.40,
        },
        "expected_calibration_error": 0.10,
        "game_normalized_brier_score": 0.80,
        "game_normalized_negative_log_likelihood": nll,
        "game_normalized_top_1_accuracy": top1,
        "game_normalized_top_3_accuracy": 0.50,
        "game_normalized_top_5_accuracy": 0.60,
        "hidden_parameter_accuracy": 0.50,
        "metrics_per_drawback": {
            drawback_id: {"top_1_accuracy": 0.30}
            for drawback_id in CAPTURABLE_RULE_IDS
        },
        "probability_diagnostics": {
            "checked_count": 10,
            "hard_elimination_violation_count": 0,
            "hard_mask_checked_count": 10,
            "maximum_eliminated_probability": 0.0,
            "missing_hard_mask_count": 0,
        },
    }
    color = {
        "game_normalized_negative_log_likelihood": nll,
        "game_normalized_top_1_accuracy": top1,
        "game_normalized_top_3_accuracy": 0.50,
    }
    binary = {
        "accuracy": 0.75,
        "brier_score": 0.20,
        "negative_log_likelihood": 0.50,
    }
    return {
        "forced": dict(binary),
        "hybrid": hybrid,
        "hybridByColor": {
            "black": dict(color),
            "white": dict(color),
        },
        "trigger": dict(binary),
    }


def _balanced_rows():
    rows = []
    for assignment in EXPECTED_FIXED_ASSIGNMENTS:
        for color, drawback in (
            ("white", assignment.white_rule_id),
            ("black", assignment.black_rule_id),
        ):
            value = capturable_row(
                game_id=assignment.game_id,
                color=color,
                drawback=drawback,
                triggered=False,
            )
            value["seed"] = assignment.gameplay_seed
            value["result"] = {
                "kind": "draw",
                "reason": "fixture-terminal",
            }
            rows.append(parse_capturable_dataset_row(value))
    return tuple(rows)


def _predictions(rows, *, correct: bool, digest: str):
    drawback = []
    for row in rows:
        probabilities = [1.0 / CAPTURABLE_RULE_COUNT] * CAPTURABLE_RULE_COUNT
        if correct:
            probabilities = [0.0] * CAPTURABLE_RULE_COUNT
            probabilities[
                CAPTURABLE_RULE_IDS.index(row.labels.true_drawback)
            ] = 1.0
        drawback.append(tuple(probabilities))
    return ComponentPredictions(
        drawback=tuple(drawback),
        trigger=(0.5,) * len(rows),
        forced=(0.5,) * len(rows),
        parameters=((0.5, 0.5),) * len(rows),
        hard_masks=((False,) * CAPTURABLE_RULE_COUNT,) * len(rows),
        sha256=digest,
    )


class CapturableFixedBlendContractTests(unittest.TestCase):
    def test_fixed_candidate_is_independent_of_rejected_grid_winner(self) -> None:
        control = _metrics()
        fixed = _metrics(top1=0.302, nll=1.9)
        grid = {
            "candidates": [
                {
                    "metrics": fixed,
                    "predictionsSha256": (
                        FIXED_VALIDATION_PREDICTIONS_SHA256
                    ),
                    "weight": 0.1,
                }
            ],
            "control": {"metrics": control},
            "execution": {"revision": GRID_EXECUTION_REVISION},
            "releaseDecision": "retain-control",
            "selected": {"weight": 0.7},
        }

        selected = fixed_validation_candidate(grid)

        self.assertEqual(selected["weight"], 0.1)
        tampered = deepcopy(grid)
        tampered["candidates"][0]["predictionsSha256"] = "0" * 64
        with self.assertRaisesRegex(
            CapturableDatasetError,
            "not the frozen hypothesis",
        ):
            fixed_validation_candidate(tampered)

    def test_corpus_audit_and_paired_top1_use_physical_games(self) -> None:
        rows = _balanced_rows()
        summary = audit_confirmation_rows(rows, set())
        control = _predictions(rows, correct=False, digest="a" * 64)
        fixed = _predictions(rows, correct=True, digest="b" * 64)

        paired = paired_game_top1_deltas(rows, control, fixed)

        self.assertEqual(summary["games"], 625)
        self.assertEqual(summary["playerGames"], 1_250)
        self.assertEqual(len(paired.deltas), 625)
        self.assertEqual(
            paired.game_ids,
            tuple(
                sorted(
                    {
                        row.evaluation.game_id
                        for row in rows
                    }
                )
            ),
        )
        self.assertAlmostEqual(
            paired.observed_delta,
            1.0 - (1.0 / CAPTURABLE_RULE_COUNT),
        )
        with self.assertRaisesRegex(
            CapturableDatasetError,
            "overlaps prior corpus",
        ):
            audit_confirmation_rows(rows, {rows[0].evaluation.game_id})
        with self.assertRaisesRegex(
            CapturableDatasetError,
            "frozen schedule|625 two-color games",
        ):
            audit_confirmation_rows(rows[:-1], set())

    def test_corpus_audit_rejects_schedule_and_trajectory_forgery(
        self,
    ) -> None:
        rows = list(_balanced_rows())

        wrong_seed = deepcopy(rows)
        wrong_seed[0] = deepcopy(wrong_seed[0])
        object.__setattr__(
            wrong_seed[0].evaluation,
            "seed",
            wrong_seed[0].evaluation.seed ^ 1,
        )
        with self.assertRaisesRegex(
            CapturableDatasetError,
            "provenance disagrees",
        ):
            audit_confirmation_rows(wrong_seed, set())

        wrong_label = deepcopy(rows)
        object.__setattr__(
            wrong_label[0].labels,
            "true_drawback",
            "vegan"
            if wrong_label[0].labels.true_drawback != "vegan"
            else "checkers",
        )
        with self.assertRaisesRegex(
            CapturableDatasetError,
            "label disagrees",
        ):
            audit_confirmation_rows(wrong_label, set())

        reordered = rows[2:4] + rows[0:2] + rows[4:]
        with self.assertRaisesRegex(
            CapturableDatasetError,
            "game order",
        ):
            audit_confirmation_rows(reordered, set())

        wrong_ply = deepcopy(rows)
        object.__setattr__(wrong_ply[1].features, "ply", 2)
        with self.assertRaisesRegex(
            CapturableDatasetError,
            "ply trajectory",
        ):
            audit_confirmation_rows(wrong_ply, set())

    def test_splitmix_golden_rejection_and_frozen_bootstrap(self) -> None:
        state = 0
        outputs = []
        for _ in range(3):
            state, value = _splitmix64_next(state)
            outputs.append(value)
        self.assertEqual(
            outputs,
            [
                0xE220A8397B1DCDAF,
                0x6E789E6AA1B965F4,
                0x06C45D188009454F,
            ],
        )
        limit = (1 << 64) - ((1 << 64) % 625)
        self.assertIsNone(_reduce_to_index(limit, 625))
        self.assertEqual(_reduce_to_index(limit - 1, 625), 624)
        self.assertEqual(BOOTSTRAP_REPLICATES, 20_000)
        self.assertEqual(BOOTSTRAP_DRAWS_PER_REPLICATE, 625)
        self.assertEqual(BOOTSTRAP_LOWER_INDEX, 999)

        bootstrap = fixed_paired_bootstrap((0.002,) * 625)

        self.assertTrue(math.isclose(bootstrap.lower_bound, 0.002))
        self.assertEqual(bootstrap.rejected_draws, 0)

        nonuniform = tuple(
            (index - 312) / 1_000_000
            for index in range(625)
        )
        ordered = fixed_paired_bootstrap(nonuniform)
        reversed_order = fixed_paired_bootstrap(
            tuple(reversed(nonuniform))
        )
        self.assertEqual(
            ordered.lower_bound,
            -1.1721599999999998e-05,
        )
        self.assertEqual(reversed_order.lower_bound, -1.18272e-05)
        self.assertNotEqual(
            ordered.lower_bound,
            reversed_order.lower_bound,
        )

    def test_release_requires_minimum_observed_and_positive_lower_bound(
        self,
    ) -> None:
        control = _metrics()
        fixed = _metrics(top1=0.302, nll=1.9)
        paired = PairedTop1(
            game_ids=tuple(f"g-{index:03d}" for index in range(625)),
            deltas=(0.002,) * 625,
            observed_delta=0.002,
        )
        passing = fixed_release_checks(
            control,
            fixed,
            paired,
            BootstrapResult(lower_bound=0.001, rejected_draws=0),
        )
        self.assertTrue(all(passing.values()))

        below_minimum = fixed_release_checks(
            control,
            fixed,
            PairedTop1(
                game_ids=paired.game_ids,
                deltas=(0.0009,) * 625,
                observed_delta=0.0009,
            ),
            BootstrapResult(lower_bound=0.0001, rejected_draws=0),
        )
        self.assertFalse(
            below_minimum["pairedTop1MinimumObservedGain"]
        )
        zero_lower = fixed_release_checks(
            control,
            fixed,
            paired,
            BootstrapResult(lower_bound=0.0, rejected_draws=0),
        )
        self.assertFalse(
            zero_lower["pairedTop1BootstrapLowerPositive"]
        )


class CapturableFixedBlendOrchestrationTests(unittest.TestCase):
    def test_cli_checks_python_runtime_before_parsing_inputs(self) -> None:
        with (
            patch(
                "drawback_ml.capturable_fixed_blend."
                "require_isolated_python_runtime",
                side_effect=CapturableDatasetError("runtime rejected"),
            ) as require_runtime,
            patch(
                "drawback_ml.capturable_fixed_blend._parser",
            ) as parser,
        ):
            with self.assertRaisesRegex(
                CapturableDatasetError,
                "runtime rejected",
            ):
                main([])

        require_runtime.assert_called_once_with()
        parser.assert_not_called()

    def _layout(self, directory: str):
        root = Path(directory) / "private"
        root.mkdir()
        control = root / str(_EXPECTED_INPUTS["control"]["directory"])
        treatment = root / str(_EXPECTED_INPUTS["treatment"]["directory"])
        control.mkdir()
        treatment.mkdir()
        grid = root / "capturable25-v3-convex-blend-validation.json"
        registry = root / "capturable25-prior-corpus-registry-v1.json"
        test = root / CONFIRMATION_TEST_FILE
        trace = root / CONFIRMATION_TRACE_FILE
        receipt = root / CORPUS_RECEIPT_FILE
        grid.write_bytes(b"grid\n")
        registry.write_bytes(b"registry\n")
        test.write_bytes(b"sealed")
        trace.write_bytes(b"trace\n")
        receipt.write_bytes(b"receipt\n")
        (control / "selection.json").write_bytes(b"selection\n")
        (control / "model.pt").write_bytes(b"model\n")
        (treatment / "selection.json").write_bytes(b"selection\n")
        (treatment / "model.pt").write_bytes(b"model\n")
        return (
            root,
            grid,
            control / "selection.json",
            treatment / "selection.json",
            registry,
            test,
        )

    @staticmethod
    def _authenticated():
        return (
            object(),
            {"component": "control"},
            dict(_EXPECTED_INPUTS["control"]),
            object(),
            {"component": "treatment"},
            dict(_EXPECTED_INPUTS["treatment"]),
            {"grid": "authenticated"},
            {"gameIds": []},
            {
                "cleanWorktree": True,
                "repository": "DrawbackGuesser",
                "revision": "f" * 40,
            },
            PRIOR_REGISTRY_SHA256,
        )

    @staticmethod
    def _receipt():
        return {
            "dataset": {
                "rows": 1,
                "sha256": "c" * 64,
            },
            "trace": {"sha256": "e" * 64},
        }

    @staticmethod
    def _verified(rows):
        return CorpusVerification(
            artifact=CapturableFixedBlendOrchestrationTests._receipt(),
            rows=tuple(rows),
            test_sha256="c" * 64,
            corpus={"games": 625, "rows": len(rows)},
        )

    def _run_with_report_publisher(
        self,
        paths,
        publisher,
        *,
        after_inference=None,
    ):
        rows = (object(),)
        control_predictions = ComponentPredictions(
            drawback=(
                (1.0 / CAPTURABLE_RULE_COUNT,)
                * CAPTURABLE_RULE_COUNT,
            ),
            trigger=(0.5,),
            forced=(0.5,),
            parameters=((0.5, 0.5),),
            hard_masks=((False,) * CAPTURABLE_RULE_COUNT,),
            sha256="a" * 64,
        )
        fixed_predictions = ComponentPredictions(
            **{
                **control_predictions.__dict__,
                "sha256": "b" * 64,
            }
        )
        bootstrap = BootstrapResult(
            lower_bound=0.001,
            rejected_draws=0,
        )
        paired = PairedTop1(
            game_ids=tuple(sorted(
                assignment.game_id
                for assignment in EXPECTED_FIXED_ASSIGNMENTS
            )),
            deltas=(0.002,) * 625,
            observed_delta=0.002,
        )
        metric_values = iter(
            [_metrics(), _metrics(top1=0.302, nll=1.9)]
        )

        def evaluate(*_arguments):
            value = next(metric_values)
            if (
                value["hybrid"]["game_normalized_top_1_accuracy"]
                == 0.302
                and after_inference is not None
            ):
                after_inference()
            return value

        with (
            patch(
                "drawback_ml.capturable_fixed_blend."
                "_authenticate_fixed_inputs",
                return_value=self._authenticated(),
            ),
            patch(
                "drawback_ml.capturable_fixed_blend."
                "load_fixed_corpus_receipt",
                return_value=(self._receipt(), "d" * 64),
            ),
            patch(
                "drawback_ml.capturable_fixed_blend."
                "authenticate_corpus_environment",
            ),
            patch(
                "drawback_ml.capturable_fixed_blend."
                "verify_fixed_corpus_receipt",
                return_value=self._verified(rows),
            ),
            patch(
                "drawback_ml.capturable_fixed_blend."
                "reauthenticate_fixed_corpus_files",
            ),
            patch(
                "drawback_ml.capturable_fixed_blend.tensorize",
                return_value=object(),
            ),
            patch(
                "drawback_ml.capturable_fixed_blend."
                "component_predictions",
                side_effect=[
                    control_predictions,
                    deepcopy(control_predictions),
                ],
            ),
            patch(
                "drawback_ml.capturable_fixed_blend.blend_components",
                return_value=fixed_predictions,
            ),
            patch(
                "drawback_ml.capturable_fixed_blend."
                "evaluate_predictions",
                side_effect=evaluate,
            ),
            patch(
                "drawback_ml.capturable_fixed_blend."
                "paired_game_top1_deltas",
                return_value=paired,
            ),
            patch(
                "drawback_ml.capturable_fixed_blend."
                "fixed_paired_bootstrap",
                return_value=bootstrap,
            ),
            patch(
                "drawback_ml.capturable_fixed_blend."
                "publish_bytes_durable",
                side_effect=publisher,
            ),
        ):
            result = run_fixed_blend_confirmation(
                *paths[1:],
                paths[0],
            )
        return result, bootstrap

    def test_cli_and_function_expose_no_weight_parameter(self) -> None:
        self.assertNotIn(
            "weight",
            inspect.signature(
                run_fixed_blend_confirmation
            ).parameters,
        )
        with self.assertRaises(SystemExit):
            _parser().parse_args(
                [
                    "--grid",
                    "grid.json",
                    "--control-selection",
                    "control.json",
                    "--treatment-selection",
                    "treatment.json",
                    "--prior-registry",
                    "registry.json",
                    "--test",
                    "test.ndjson",
                    "--output-directory",
                    "output",
                    "--weight",
                    "0.2",
                ]
            )

    def test_consumption_marker_precedes_test_read_and_survives_failure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._layout(directory)
            root = paths[0]
            with (
                patch(
                    "drawback_ml.capturable_fixed_blend."
                    "_authenticate_fixed_inputs",
                    return_value=self._authenticated(),
                ),
                patch(
                    "drawback_ml.capturable_fixed_blend."
                    "load_fixed_corpus_receipt",
                    return_value=(self._receipt(), "d" * 64),
                ),
                patch(
                    "drawback_ml.capturable_fixed_blend."
                    "authenticate_corpus_environment",
                ),
                patch(
                    "drawback_ml.capturable_fixed_blend."
                    "verify_fixed_corpus_receipt",
                    side_effect=CapturableDatasetError("bad sealed test"),
                ) as verifier,
            ):
                with self.assertRaisesRegex(
                    CapturableDatasetError,
                    "bad sealed test",
                ):
                    run_fixed_blend_confirmation(*paths[1:], root)
                self.assertTrue((root / CONSUMPTION_FILE).is_file())
                verifier.assert_called_once()

            with patch(
                "drawback_ml.capturable_fixed_blend."
                "verify_fixed_corpus_receipt",
            ) as verifier:
                with self.assertRaisesRegex(
                    FileExistsError,
                    "already consumed",
                ):
                    run_fixed_blend_confirmation(*paths[1:], root)
                verifier.assert_not_called()

    def test_marker_publication_failure_prevents_test_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._layout(directory)
            root = paths[0]
            with (
                patch(
                    "drawback_ml.capturable_fixed_blend."
                    "_authenticate_fixed_inputs",
                    return_value=self._authenticated(),
                ),
                patch(
                    "drawback_ml.capturable_fixed_blend."
                    "load_fixed_corpus_receipt",
                    return_value=(self._receipt(), "d" * 64),
                ),
                patch(
                    "drawback_ml.capturable_fixed_blend."
                    "authenticate_corpus_environment",
                ),
                patch(
                    "drawback_ml.capturable_fixed_blend."
                    "publish_bytes_durable",
                    side_effect=OSError("marker publication failed"),
                ),
                patch(
                    "drawback_ml.capturable_fixed_blend."
                    "verify_fixed_corpus_receipt",
                ) as verifier,
            ):
                with self.assertRaisesRegex(
                    OSError,
                    "marker publication failed",
                ):
                    run_fixed_blend_confirmation(*paths[1:], root)
                verifier.assert_not_called()
                self.assertFalse((root / CONSUMPTION_FILE).exists())

    def test_post_link_marker_failure_is_irreversible_and_fail_closed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._layout(directory)
            root = paths[0]

            def linked_then_failed(path: Path, payload: bytes) -> None:
                path.write_bytes(payload)
                raise OSError("directory sync failed")

            with (
                patch(
                    "drawback_ml.capturable_fixed_blend."
                    "_authenticate_fixed_inputs",
                    return_value=self._authenticated(),
                ),
                patch(
                    "drawback_ml.capturable_fixed_blend."
                    "load_fixed_corpus_receipt",
                    return_value=(self._receipt(), "d" * 64),
                ),
                patch(
                    "drawback_ml.capturable_fixed_blend."
                    "authenticate_corpus_environment",
                ),
                patch(
                    "drawback_ml.capturable_fixed_blend."
                    "publish_bytes_durable",
                    side_effect=linked_then_failed,
                ),
                patch(
                    "drawback_ml.capturable_fixed_blend."
                    "verify_fixed_corpus_receipt",
                ) as verifier,
            ):
                with self.assertRaisesRegex(
                    OSError,
                    "directory sync failed",
                ):
                    run_fixed_blend_confirmation(*paths[1:], root)

            self.assertTrue((root / CONSUMPTION_FILE).is_file())
            verifier.assert_not_called()
            with self.assertRaisesRegex(
                FileExistsError,
                "already consumed",
            ):
                run_fixed_blend_confirmation(*paths[1:], root)

    def test_pre_link_report_failure_keeps_marker_and_no_report(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._layout(directory)
            root = paths[0]

            def marker_then_report_failure(
                path: Path,
                payload: bytes,
            ) -> None:
                if path.name == CONSUMPTION_FILE:
                    path.write_bytes(payload)
                    return
                raise OSError("report link failed")

            with self.assertRaisesRegex(
                OSError,
                "report link failed",
            ):
                self._run_with_report_publisher(
                    paths,
                    marker_then_report_failure,
                )

            self.assertTrue((root / CONSUMPTION_FILE).is_file())
            self.assertEqual(
                list(root.glob(f"{REPORT_PREFIX}*.json")),
                [],
            )
            with self.assertRaisesRegex(
                FileExistsError,
                "already consumed",
            ):
                run_fixed_blend_confirmation(*paths[1:], root)

    def test_marker_mutation_after_inference_prevents_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._layout(directory)
            root = paths[0]

            def publisher(path: Path, payload: bytes) -> None:
                path.write_bytes(payload)

            def mutate_marker() -> None:
                (root / CONSUMPTION_FILE).write_bytes(b"{}\n")

            with self.assertRaisesRegex(
                CapturableDatasetError,
                "changed during evaluation",
            ):
                self._run_with_report_publisher(
                    paths,
                    publisher,
                    after_inference=mutate_marker,
                )

            self.assertTrue((root / CONSUMPTION_FILE).is_file())
            self.assertEqual(
                list(root.glob(f"{REPORT_PREFIX}*.json")),
                [],
            )
            with self.assertRaisesRegex(
                FileExistsError,
                "already consumed",
            ):
                run_fixed_blend_confirmation(*paths[1:], root)

    def test_post_link_report_failure_keeps_valid_report_and_marker(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._layout(directory)
            root = paths[0]

            def linked_then_report_failure(
                path: Path,
                payload: bytes,
            ) -> None:
                path.write_bytes(payload)
                if path.name != CONSUMPTION_FILE:
                    raise OSError("report directory sync failed")

            bootstrap = BootstrapResult(
                lower_bound=0.001,
                rejected_draws=0,
            )
            with self.assertRaisesRegex(
                OSError,
                "report directory sync failed",
            ):
                self._run_with_report_publisher(
                    paths,
                    linked_then_report_failure,
                )

            marker_path = root / CONSUMPTION_FILE
            report_paths = list(root.glob(f"{REPORT_PREFIX}*.json"))
            self.assertTrue(marker_path.is_file())
            self.assertEqual(len(report_paths), 1)
            with (
                patch(
                    "drawback_ml.capturable_fixed_blend_contract."
                    "fixed_paired_bootstrap",
                    return_value=bootstrap,
                ),
                patch(
                    "drawback_ml.capturable_fixed_corpus."
                    "load_fixed_corpus_receipt",
                    return_value=(self._receipt(), "d" * 64),
                ),
                patch(
                    "drawback_ml.capturable_fixed_blend_contract."
                    "_authenticated_recorded_revision_identity",
                ),
            ):
                report, digest = load_fixed_blend_confirmation(
                    report_paths[0],
                )
            self.assertEqual(
                report_paths[0].name,
                f"{REPORT_PREFIX}{digest}.json",
            )
            self.assertEqual(
                report["releaseDecision"],
                "promote-fixed-blend",
            )
            with self.assertRaisesRegex(
                FileExistsError,
                "already consumed",
            ):
                run_fixed_blend_confirmation(*paths[1:], root)

    def test_report_loader_rejects_link_replaced_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._layout(directory)

            def publisher(path: Path, payload: bytes) -> None:
                path.write_bytes(payload)

            result, bootstrap = self._run_with_report_publisher(
                paths,
                publisher,
            )
            root = paths[0]
            marker_path = root / CONSUMPTION_FILE
            linked_directory = root / "linked-marker"
            linked_directory.mkdir()
            target = linked_directory / CONSUMPTION_FILE
            marker_path.replace(target)
            try:
                marker_path.symlink_to(target)
            except OSError:
                target.replace(marker_path)
                self.skipTest("symbolic links are unavailable")

            with (
                patch(
                    "drawback_ml.capturable_fixed_blend_contract."
                    "fixed_paired_bootstrap",
                    return_value=bootstrap,
                ),
                patch(
                    "drawback_ml.capturable_fixed_corpus."
                    "load_fixed_corpus_receipt",
                    return_value=(self._receipt(), "d" * 64),
                ),
                patch(
                    "drawback_ml.capturable_fixed_blend_contract."
                    "_authenticated_recorded_revision_identity",
                ),
                self.assertRaisesRegex(
                    CapturableDatasetError,
                    "link or junction",
                ),
            ):
                load_fixed_blend_confirmation(
                    Path(result["reportPath"]),
                )

    def test_success_runs_components_once_and_report_reauthenticates(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._layout(directory)
            root = paths[0]
            rows = (object(),)
            control_predictions = ComponentPredictions(
                drawback=(
                    (1.0 / CAPTURABLE_RULE_COUNT,)
                    * CAPTURABLE_RULE_COUNT,
                ),
                trigger=(0.5,),
                forced=(0.5,),
                parameters=((0.5, 0.5),),
                hard_masks=(
                    (False,) * CAPTURABLE_RULE_COUNT,
                ),
                sha256="a" * 64,
            )
            treatment_predictions = deepcopy(control_predictions)
            fixed_predictions = ComponentPredictions(
                **{
                    **control_predictions.__dict__,
                    "sha256": "b" * 64,
                }
            )
            control_metrics = _metrics()
            fixed_metrics = _metrics(top1=0.302, nll=1.9)
            paired = PairedTop1(
                game_ids=tuple(sorted(
                    assignment.game_id
                    for assignment in EXPECTED_FIXED_ASSIGNMENTS
                )),
                deltas=(0.002,) * 625,
                observed_delta=0.002,
            )
            bootstrap = BootstrapResult(
                lower_bound=0.001,
                rejected_draws=0,
            )
            with (
                patch(
                    "drawback_ml.capturable_fixed_blend."
                    "_authenticate_fixed_inputs",
                    return_value=self._authenticated(),
                ),
                patch(
                    "drawback_ml.capturable_fixed_blend."
                    "load_fixed_corpus_receipt",
                    return_value=(self._receipt(), "d" * 64),
                ),
                patch(
                    "drawback_ml.capturable_fixed_blend."
                    "authenticate_corpus_environment",
                ),
                patch(
                    "drawback_ml.capturable_fixed_blend."
                    "verify_fixed_corpus_receipt",
                    return_value=self._verified(rows),
                ),
                patch(
                    "drawback_ml.capturable_fixed_blend."
                    "reauthenticate_fixed_corpus_files",
                ),
                patch(
                    "drawback_ml.capturable_fixed_blend.tensorize",
                    return_value=object(),
                ) as tensorizer,
                patch(
                    "drawback_ml.capturable_fixed_blend."
                    "component_predictions",
                    side_effect=[
                        control_predictions,
                        treatment_predictions,
                    ],
                ) as component,
                patch(
                    "drawback_ml.capturable_fixed_blend.blend_components",
                    return_value=fixed_predictions,
                ) as blender,
                patch(
                    "drawback_ml.capturable_fixed_blend."
                    "evaluate_predictions",
                    side_effect=[control_metrics, fixed_metrics],
                ) as evaluator,
                patch(
                    "drawback_ml.capturable_fixed_blend."
                    "paired_game_top1_deltas",
                    return_value=paired,
                ),
                patch(
                    "drawback_ml.capturable_fixed_blend."
                    "fixed_paired_bootstrap",
                    return_value=bootstrap,
                ),
            ):
                summary = run_fixed_blend_confirmation(*paths[1:], root)

            tensorizer.assert_called_once()
            self.assertEqual(component.call_count, 2)
            self.assertEqual(evaluator.call_count, 2)
            self.assertEqual(
                blender.call_args.args[-1],
                FIXED_TREATMENT_WEIGHT,
            )
            report_path = Path(summary["reportPath"])
            with (
                patch(
                    "drawback_ml.capturable_fixed_blend_contract."
                    "fixed_paired_bootstrap",
                    return_value=bootstrap,
                ),
                patch(
                    "drawback_ml.capturable_fixed_corpus."
                    "load_fixed_corpus_receipt",
                    return_value=(self._receipt(), "d" * 64),
                ),
                patch(
                    "drawback_ml.capturable_fixed_blend_contract."
                    "_authenticated_recorded_revision_identity",
                ),
            ):
                report, digest = load_fixed_blend_confirmation(report_path)
            self.assertEqual(digest, summary["reportSha256"])
            self.assertEqual(
                report["releaseDecision"],
                "promote-fixed-blend",
            )
            self.assertNotIn("candidates", report)
            self.assertNotIn("weightGrid", report)
            marker_path = root / CONSUMPTION_FILE
            marker_payload = marker_path.read_bytes()
            marker_path.write_bytes(b"{}\n")
            with (
                patch(
                    "drawback_ml.capturable_fixed_blend_contract."
                    "fixed_paired_bootstrap",
                    return_value=bootstrap,
                ),
                patch(
                    "drawback_ml.capturable_fixed_corpus."
                    "load_fixed_corpus_receipt",
                    return_value=(self._receipt(), "d" * 64),
                ),
                self.assertRaisesRegex(
                    CapturableDatasetError,
                    "consumption evidence",
                ),
            ):
                load_fixed_blend_confirmation(report_path)
            marker_path.write_bytes(marker_payload)
            marker_path.unlink()
            with (
                patch(
                    "drawback_ml.capturable_fixed_blend_contract."
                    "fixed_paired_bootstrap",
                    return_value=bootstrap,
                ),
                patch(
                    "drawback_ml.capturable_fixed_corpus."
                    "load_fixed_corpus_receipt",
                    return_value=(self._receipt(), "d" * 64),
                ),
                self.assertRaisesRegex(
                    CapturableDatasetError,
                    "fixed private regular file",
                ),
            ):
                load_fixed_blend_confirmation(report_path)


if __name__ == "__main__":
    unittest.main()
