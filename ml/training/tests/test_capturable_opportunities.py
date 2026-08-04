from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import _bootstrap  # noqa: F401

from capturable_fixture import (
    capturable_opportunity_row,
    capturable_row,
)
from drawback_ml.capturable_baseline import (
    CAPTURABLE_OPPORTUNITY_BASELINE_VERSION,
    CapturableTrainingConfig,
    _parser as baseline_parser,
    create_capturable_model,
    create_capturable_opportunity_model,
    evaluate_capturable,
    tensorize,
    train_capturable_baseline,
)
from drawback_ml.capturable_candidate_selection import (
    load_treatment_comparison,
    run_treatment_comparison,
)
from drawback_ml.capturable_experiment import (
    CAPTURABLE_OPPORTUNITY_SELECTION_VERSION,
    _load_selection_checkpoint,
    _parser as experiment_parser,
    run_candidate_selection,
    run_sealed_evaluation,
    run_selection,
)
from drawback_ml.capturable_records import (
    CAPTURABLE_FEATURE_DIMENSION,
    CAPTURABLE_OPPORTUNITY_FIELDS,
    CAPTURABLE_OPPORTUNITY_SHAPE,
    CAPTURABLE_OPPORTUNITY_SYMBOLIC_FEATURE_VERSION,
    CAPTURABLE_RULE_INDEX,
    CAPTURABLE_RULE_IDS,
    CapturableDatasetError,
    parse_capturable_dataset_row,
    parse_capturable_opportunity_dataset_row,
)


class CapturableOpportunityTests(unittest.TestCase):
    def test_clis_expose_the_two_explicit_opportunity_modes(self) -> None:
        shared = (
            "--train",
            "train.ndjson",
            "--validation",
            "validation.ndjson",
            "--output",
            "output",
            "--opportunity-mode",
            "public-exact",
        )
        baseline = baseline_parser().parse_args(
            (*shared, "--test", "test.ndjson")
        )
        selection = experiment_parser().parse_args(("select", *shared))

        self.assertEqual(baseline.opportunity_mode, "public-exact")
        self.assertEqual(selection.opportunity_mode, "public-exact")
        self.assertEqual(
            baseline_parser().get_default("opportunity_mode"),
            None,
        )

    def test_tensorizes_public_exact_and_zero_ablation_separately(self) -> None:
        import torch

        rows = self._split("tensor")
        public = tensorize(rows, opportunity_mode="public-exact")
        ablation = tensorize(rows, opportunity_mode="zero-ablation")

        self.assertEqual(
            tuple(public.rule_opportunities.shape),
            (len(rows), *CAPTURABLE_OPPORTUNITY_SHAPE),
        )
        self.assertTrue(torch.any(public.rule_opportunities != 0.0))
        self.assertTrue(torch.equal(
            ablation.rule_opportunities,
            torch.zeros_like(ablation.rule_opportunities),
        ))
        self.assertTrue(torch.equal(public.inputs, ablation.inputs))

        schema_eight = (
            parse_capturable_dataset_row(capturable_row()),
        )
        with self.assertRaisesRegex(ValueError, "schema-9 rows"):
            tensorize(schema_eight, opportunity_mode="public-exact")
        with self.assertRaisesRegex(ValueError, "explicit opportunity mode"):
            tensorize(rows)

    def test_opportunity_model_has_zero_initialized_shared_residual(self) -> None:
        import torch

        torch.manual_seed(20260728)
        public_model = create_capturable_opportunity_model(8)
        torch.manual_seed(20260728)
        ablation_model = create_capturable_opportunity_model(8)
        torch.manual_seed(20260728)
        legacy_model = create_capturable_model(8)

        self.assertEqual(
            tuple(public_model.opportunity_weights.shape),
            CAPTURABLE_OPPORTUNITY_SHAPE,
        )
        self.assertTrue(torch.equal(
            public_model.opportunity_weights,
            torch.zeros_like(public_model.opportunity_weights),
        ))
        for key, value in public_model.state_dict().items():
            self.assertTrue(value.equal(ablation_model.state_dict()[key]))
            if key != "opportunity_weights":
                self.assertTrue(value.equal(legacy_model.state_dict()[key]))

        inputs = torch.zeros((2, CAPTURABLE_FEATURE_DIMENSION))
        zeros = torch.zeros((2, *CAPTURABLE_OPPORTUNITY_SHAPE))
        ones = torch.ones((2, *CAPTURABLE_OPPORTUNITY_SHAPE))
        zero_outputs = public_model(inputs, zeros)
        one_outputs = public_model(inputs, ones)
        for head in (
            "white_drawback",
            "black_drawback",
            "trigger",
            "forced",
            "triple_play_parameter",
        ):
            self.assertTrue(torch.equal(zero_outputs[head], one_outputs[head]))
        with self.assertRaisesRegex(ValueError, r"\[N, 25, 4\]"):
            public_model(inputs, zeros[:1])

    def test_residual_is_candidate_wise_and_leaves_auxiliary_heads_alone(
        self,
    ) -> None:
        import torch

        model = create_capturable_opportunity_model(8)
        model.eval()
        inputs = torch.zeros((1, CAPTURABLE_FEATURE_DIMENSION))
        opportunities = torch.zeros((1, *CAPTURABLE_OPPORTUNITY_SHAPE))
        rule_index = CAPTURABLE_RULE_INDEX["checkers"]
        opportunities[0, rule_index] = 1.0
        with torch.no_grad():
            baseline = model(inputs, torch.zeros_like(opportunities))
            model.opportunity_weights[rule_index] = torch.tensor(
                [1.0, 2.0, 3.0, 4.0]
            )
            changed = model(inputs, opportunities)

        for head in ("white_drawback", "black_drawback"):
            delta = changed[head] - baseline[head]
            expected = torch.zeros_like(delta)
            expected[0, rule_index] = 10.0
            torch.testing.assert_close(delta, expected)
        for head in ("trigger", "forced", "triple_play_parameter"):
            torch.testing.assert_close(changed[head], baseline[head])

    def test_extreme_opportunity_residual_cannot_restore_hard_elimination(
        self,
    ) -> None:
        import torch

        row_value = capturable_opportunity_row(
            eliminated_rule="checkers",
        )
        rule_index = CAPTURABLE_RULE_INDEX["checkers"]
        features = row_value["symbolicActiveRuleOpportunityFeatures"]
        field_count = len(CAPTURABLE_OPPORTUNITY_FIELDS)
        start = rule_index * field_count
        features[start : start + field_count] = [1.0] * field_count
        rows = (parse_capturable_opportunity_dataset_row(row_value),)
        model = create_capturable_opportunity_model(8)
        with torch.no_grad():
            model.opportunity_weights[rule_index].fill_(1e20)
        config = CapturableTrainingConfig(
            epochs=1,
            batch_size=1,
            hidden_dimension=8,
            torch_threads=1,
        )

        report = evaluate_capturable(
            model,
            rows,
            tensorize(rows, opportunity_mode="public-exact"),
            config,
            2.0,
            0.1,
        )

        diagnostics = report["hybrid"]["probability_diagnostics"]
        self.assertEqual(diagnostics["maximum_eliminated_probability"], 0.0)
        self.assertEqual(diagnostics["hard_elimination_violation_count"], 0)

    def test_opportunity_training_is_deterministic_for_both_modes(self) -> None:
        config = CapturableTrainingConfig(
            seed=20260728,
            epochs=1,
            batch_size=2,
            hidden_dimension=8,
            torch_threads=1,
        )
        for mode in ("public-exact", "zero-ablation"):
            with self.subTest(mode=mode):
                train = self._split(f"{mode}-train")
                validation = self._split(f"{mode}-validation")
                first_model, first_report = train_capturable_baseline(
                    train,
                    validation,
                    None,
                    config,
                    opportunity_mode=mode,
                )
                second_model, second_report = train_capturable_baseline(
                    train,
                    validation,
                    None,
                    config,
                    opportunity_mode=mode,
                )

                self.assertEqual(first_report, second_report)
                self.assertEqual(
                    first_report["version"],
                    CAPTURABLE_OPPORTUNITY_BASELINE_VERSION,
                )
                self.assertEqual(
                    first_report["symbolicFeatureVersion"],
                    CAPTURABLE_OPPORTUNITY_SYMBOLIC_FEATURE_VERSION,
                )
                self.assertEqual(first_report["opportunityMode"], mode)
                for key, value in first_model.state_dict().items():
                    self.assertTrue(
                        value.equal(second_model.state_dict()[key])
                    )

    def test_checkpoint_binds_full_opportunity_contract(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = self._write_split(root, "train", opportunities=True)
            validation = self._write_split(
                root,
                "validation",
                opportunities=True,
            )
            output = root / "selection"
            config = CapturableTrainingConfig(
                seed=20260728,
                epochs=1,
                batch_size=2,
                hidden_dimension=8,
                torch_threads=1,
            )
            run_selection(
                (train,),
                validation,
                output,
                config,
                opportunity_mode="public-exact",
            )

            checkpoint_path = output / "model.pt"
            _, metadata, _ = _load_selection_checkpoint(checkpoint_path)
            self.assertEqual(
                metadata["opportunityShape"],
                list(CAPTURABLE_OPPORTUNITY_SHAPE),
            )
            self.assertEqual(
                metadata["opportunityFields"],
                list(CAPTURABLE_OPPORTUNITY_FIELDS),
            )
            self.assertEqual(metadata["opportunityMode"], "public-exact")
            test = self._write_split(root, "test", opportunities=True)
            sealed = run_sealed_evaluation(
                checkpoint_path,
                test,
                root / "sealed.json",
                expected_test_sha256=hashlib.sha256(
                    test.read_bytes()
                ).hexdigest(),
                consumption_registry=root / "consumption-registry",
            )
            self.assertIn("testHybridTop1", sealed)

            ablation_output = root / "ablation-selection"
            run_selection(
                (train,),
                validation,
                ablation_output,
                CapturableTrainingConfig(
                    seed=config.seed,
                    epochs=1,
                    batch_size=2,
                    hidden_dimension=8,
                    torch_threads=1,
                ),
                opportunity_mode="zero-ablation",
            )
            choice = run_candidate_selection(
                (
                    output / "selection.json",
                    ablation_output / "selection.json",
                ),
                root / "candidate.json",
            )
            self.assertIn(
                choice["selectedDirectory"],
                {"selection", "ablation-selection"},
            )

            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(
                checkpoint["version"],
                CAPTURABLE_OPPORTUNITY_SELECTION_VERSION,
            )
            for field, invalid in (
                ("symbolicFeatureVersion", 9.0),
                ("opportunityFeatureVersion", 1.0),
                ("opportunityRuleIds", ["vegan"]),
                ("opportunityFields", ["knownMass"]),
                ("opportunityShape", [25.0, 4.0]),
                ("opportunityMode", "unknown"),
            ):
                with self.subTest(field=field):
                    tampered = {
                        **checkpoint,
                        "metadata": {
                            **checkpoint["metadata"],
                            field: invalid,
                        },
                    }
                    path = root / f"tampered-{field}.pt"
                    torch.save(tampered, path)
                    with self.assertRaisesRegex(
                        CapturableDatasetError,
                        "opportunity contract",
                    ):
                        _load_selection_checkpoint(path)
            for invalid_version in (2.0, True):
                with self.subTest(version=invalid_version):
                    invalid_outer = {
                        **checkpoint,
                        "version": invalid_version,
                    }
                    path = root / f"tampered-version-{invalid_version}.pt"
                    torch.save(invalid_outer, path)
                    with self.assertRaisesRegex(
                        CapturableDatasetError,
                        "outer contract",
                    ):
                        _load_selection_checkpoint(path)
            opportunity_as_legacy = {
                **checkpoint,
                "version": 1,
            }
            opportunity_as_legacy_path = root / "opportunity-as-legacy.pt"
            torch.save(opportunity_as_legacy, opportunity_as_legacy_path)
            with self.assertRaisesRegex(
                CapturableDatasetError,
                "selection metadata",
            ):
                _load_selection_checkpoint(opportunity_as_legacy_path)

            legacy_train = self._write_split(
                root,
                "legacy-train",
                opportunities=False,
            )
            legacy_validation = self._write_split(
                root,
                "legacy-validation",
                opportunities=False,
            )
            legacy_output = root / "legacy-selection"
            run_selection(
                (legacy_train,),
                legacy_validation,
                legacy_output,
                config,
            )
            legacy = torch.load(
                legacy_output / "model.pt",
                map_location="cpu",
                weights_only=True,
            )
            legacy["version"] = CAPTURABLE_OPPORTUNITY_SELECTION_VERSION
            legacy_as_opportunity = root / "legacy-as-opportunity.pt"
            torch.save(legacy, legacy_as_opportunity)
            with self.assertRaisesRegex(
                CapturableDatasetError,
                "selection metadata",
            ):
                _load_selection_checkpoint(legacy_as_opportunity)

    def test_cross_mode_ablation_requires_an_exact_paired_experiment(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = self._write_full_catalog_split(root, "train")
            other_train = self._write_full_catalog_split(root, "other-train")
            validation = self._write_full_catalog_split(root, "validation")
            config = CapturableTrainingConfig(
                seed=20260728,
                epochs=1,
                batch_size=2,
                hidden_dimension=8,
                torch_threads=1,
            )

            def selection(
                name: str,
                mode: str,
                selected_config: CapturableTrainingConfig = config,
                selected_train: Path = train,
            ) -> Path:
                output = root / name
                run_selection(
                    (selected_train,),
                    validation,
                    output,
                    selected_config,
                    opportunity_mode=mode,
                )
                return output / "selection.json"

            control = selection("control", "zero-ablation")
            treatment = selection("treatment", "public-exact")
            valid_path = root / "valid-comparison.json"
            run_treatment_comparison(control, (treatment,), valid_path)
            authenticated, _ = load_treatment_comparison(valid_path)
            self.assertEqual(
                authenticated["control"]["opportunityContract"][
                    "opportunityMode"
                ],
                "zero-ablation",
            )
            self.assertEqual(
                authenticated["bestTreatment"]["opportunityContract"][
                    "opportunityMode"
                ],
                "public-exact",
            )

            wrong_seed = selection(
                "wrong-seed",
                "public-exact",
                CapturableTrainingConfig(
                    seed=config.seed + 1,
                    epochs=1,
                    batch_size=2,
                    hidden_dimension=8,
                    torch_threads=1,
                ),
            )
            wrong_config = selection(
                "wrong-config",
                "public-exact",
                CapturableTrainingConfig(
                    seed=config.seed,
                    epochs=1,
                    batch_size=2,
                    hidden_dimension=8,
                    trigger_row_multiplier=2.0,
                    torch_threads=1,
                ),
            )
            wrong_train = selection(
                "wrong-train",
                "public-exact",
                selected_train=other_train,
            )
            invalid_pairs = (
                ("wrong-direction", treatment, control, "zero-ablation control"),
                ("wrong-seed", control, wrong_seed, "same seed"),
                (
                    "wrong-config",
                    control,
                    wrong_config,
                    "same full training configuration",
                ),
                ("wrong-train", control, wrong_train, "same training input"),
            )
            for name, invalid_control, invalid_treatment, message in (
                invalid_pairs
            ):
                with self.subTest(name=name, phase="creation"):
                    output = root / f"{name}.json"
                    with self.assertRaisesRegex(
                        CapturableDatasetError,
                        message,
                    ):
                        run_treatment_comparison(
                            invalid_control,
                            (invalid_treatment,),
                            output,
                        )
                    self.assertFalse(output.exists())

                with self.subTest(name=name, phase="authentication"):
                    unsafe_output = root / f"unsafe-{name}.json"
                    with patch(
                        "drawback_ml.capturable_candidate_selection."
                        "_validate_cross_mode_opportunity_ablation"
                    ):
                        run_treatment_comparison(
                            invalid_control,
                            (invalid_treatment,),
                            unsafe_output,
                        )
                    with self.assertRaisesRegex(
                        CapturableDatasetError,
                        message,
                    ):
                        load_treatment_comparison(unsafe_output)

    @staticmethod
    def _split(prefix: str):
        return (
            parse_capturable_opportunity_dataset_row(
                capturable_opportunity_row(
                    game_id=f"{prefix}-white",
                    drawback="vegan",
                )
            ),
            parse_capturable_opportunity_dataset_row(
                capturable_opportunity_row(
                    game_id=f"{prefix}-black",
                    color="black",
                    drawback="checkers",
                )
            ),
        )

    @staticmethod
    def _write_split(
        root: Path,
        split: str,
        *,
        opportunities: bool,
    ) -> Path:
        path = root / f"{split}.ndjson"
        factory = (
            capturable_opportunity_row
            if opportunities
            else capturable_row
        )
        rows = (
            factory(
                game_id=f"{split}-white",
                drawback="vegan",
            ),
            factory(
                game_id=f"{split}-black",
                color="black",
                drawback="checkers",
            ),
        )
        path.write_bytes(
            "".join(
                json.dumps(
                    row,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                + "\n"
                for row in rows
            ).encode("utf-8")
        )
        return path

    @staticmethod
    def _write_full_catalog_split(root: Path, split: str) -> Path:
        path = root / f"{split}.ndjson"
        rows = tuple(
            capturable_opportunity_row(
                game_id=f"{split}-{drawback_id}",
                color="white" if index % 2 == 0 else "black",
                drawback=drawback_id,
            )
            for index, drawback_id in enumerate(CAPTURABLE_RULE_IDS)
        )
        path.write_bytes(
            "".join(
                json.dumps(
                    row,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                + "\n"
                for row in rows
            ).encode("utf-8")
        )
        return path


if __name__ == "__main__":
    unittest.main()
