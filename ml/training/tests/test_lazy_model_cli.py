from __future__ import annotations

import builtins
from contextlib import contextmanager
from contextlib import redirect_stdout
from dataclasses import replace
import hashlib
from io import BytesIO, StringIO
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import _bootstrap  # noqa: F401

from drawback_ml.cli import (
    _training_config,
    _verify_clean_execution_revision,
    build_parser,
    main,
)
from drawback_ml.corpus_contract import (
    CorpusContractError,
    open_audited_private_corpus_split,
)
from drawback_ml.model import ModelConfig, create_model
from drawback_ml.streaming_training import staged_output_directory
from drawback_ml.training_corpus_set import (
    AGENT_DOMAIN,
    FROZEN_SUPPLEMENT_PROFILES,
    TrainingCorpusSetError,
)

from test_records import row
from test_corpus_contract import BINARY_DIGEST, ENGINE_FINGERPRINT, write_fixture
from test_private_corpus_contract import write_release


def _patched_authenticated_git(*results: SimpleNamespace):
    return patch(
        "drawback_ml.cli._authenticated_git_runner",
        return_value=Mock(side_effect=results),
    )


class LazyModelTests(unittest.TestCase):
    def test_release_revision_requires_matching_clean_head(self) -> None:
        revision = "a" * 40
        repository_root = Path.cwd().resolve()
        with _patched_authenticated_git(
            SimpleNamespace(stdout=revision + "\n"),
            SimpleNamespace(stdout=str(repository_root) + "\n"),
            SimpleNamespace(stdout=""),
            SimpleNamespace(stdout=revision + "\n"),
        ):
            self.assertEqual(
                _verify_clean_execution_revision(revision),
                revision,
            )
        with _patched_authenticated_git(
            SimpleNamespace(stdout="b" * 40 + "\n"),
        ):
            with self.assertRaisesRegex(ValueError, "differs"):
                _verify_clean_execution_revision(revision)
        with _patched_authenticated_git(
            SimpleNamespace(stdout=revision + "\n"),
            SimpleNamespace(stdout=str(repository_root) + "\n"),
            SimpleNamespace(stdout=" M source.py\0"),
            SimpleNamespace(stdout=revision + "\n"),
        ):
            with self.assertRaisesRegex(ValueError, "clean"):
                _verify_clean_execution_revision(revision)
        with self.assertRaisesRegex(ValueError, "full lowercase"):
            _verify_clean_execution_revision("not-a-revision")

        staging = repository_root / ".run.staging-fixture"
        with _patched_authenticated_git(
            SimpleNamespace(stdout=revision + "\n"),
            SimpleNamespace(stdout=str(repository_root) + "\n"),
            SimpleNamespace(
                stdout=(
                    "?? .run.staging-fixture/checkpoint.pt\0"
                    "?? .run.staging-fixture/run.json\0"
                )
            ),
            SimpleNamespace(stdout=revision + "\n"),
        ):
            self.assertEqual(
                _verify_clean_execution_revision(
                    revision,
                    ignored_untracked_paths=(staging,),
                ),
                revision,
            )

        with _patched_authenticated_git(
            SimpleNamespace(stdout=revision + "\n"),
            SimpleNamespace(stdout=str(repository_root) + "\n"),
            SimpleNamespace(stdout=""),
            SimpleNamespace(stdout="b" * 40 + "\n"),
        ):
            with self.assertRaisesRegex(ValueError, "changed during"):
                _verify_clean_execution_revision(revision)

        with _patched_authenticated_git(
            SimpleNamespace(stdout=revision + "\n"),
            SimpleNamespace(
                stdout=str(repository_root.parent) + "\n"
            ),
        ):
            with self.assertRaisesRegex(ValueError, "loaded training source"):
                _verify_clean_execution_revision(revision)

    def test_streaming_example_count_options_are_explicit_and_compatible(
        self,
    ) -> None:
        base = ["train-corpus", "manifest", "output", "--seed", "7"]
        default = _training_config(
            build_parser().parse_args(base),
            {},
        )
        self.assertEqual(default.game_examples_per_epoch, 16)
        self.assertIsNone(default.player_game_examples_per_epoch)

        explicit = _training_config(
            build_parser().parse_args(
                [*base, "--player-game-examples-per-epoch", "5"]
            ),
            {},
        )
        self.assertEqual(explicit.player_game_examples_per_epoch, 5)
        with self.assertRaisesRegex(ValueError, "choose either"):
            _training_config(
                build_parser().parse_args(
                    [
                        *base,
                        "--game-examples-per-epoch",
                        "10",
                        "--player-game-examples-per-epoch",
                        "5",
                    ]
                ),
                {},
            )

    def test_configuration_validates_without_torch(self) -> None:
        config = ModelConfig(
            input_dimension=10,
            drawback_classes=2,
            legal_mask_dimension=20,
        )
        self.assertEqual(config.hidden_dimension, 128)
        self.assertEqual(config.parameter_classes, 1)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            ModelConfig(input_dimension=0, drawback_classes=2, legal_mask_dimension=20)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            ModelConfig(
                input_dimension=10,
                drawback_classes=2,
                legal_mask_dimension=20,
                parameter_classes=0,
            )

    def test_missing_torch_has_an_actionable_error(self) -> None:
        original_import = builtins.__import__

        def import_without_torch(name: str, *args: object, **kwargs: object) -> object:
            if name == "torch.nn" or name.startswith("torch."):
                raise ImportError("simulated missing torch")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=import_without_torch):
            with self.assertRaisesRegex(RuntimeError, "PyTorch is required"):
                create_model(
                    ModelConfig(
                        input_dimension=10,
                        drawback_classes=2,
                        legal_mask_dimension=20,
                    )
                )


class CliTests(unittest.TestCase):
    _EXECUTION_SOURCE_REVISION = "a" * 40

    def setUp(self) -> None:
        verifier = patch(
            "drawback_ml.cli._verify_clean_execution_revision",
            return_value=self._EXECUTION_SOURCE_REVISION,
        )
        verifier.start()
        self.addCleanup(verifier.stop)

    @staticmethod
    def _hard_negative_arguments(directory: Path) -> list[str]:
        arguments: list[str] = []
        for _offset, profile_id, _rule_ids in reversed(
            FROZEN_SUPPLEMENT_PROFILES
        ):
            arguments.extend(
                [
                    "--hard-negative",
                    profile_id,
                    str(directory / f"{profile_id}.manifest.json"),
                    str(directory / f"{profile_id}.ndjson"),
                    str(directory / f"{profile_id}.plan.json"),
                ]
            )
        return arguments

    @staticmethod
    @contextmanager
    def _smoke_hard_negative_lease(
        manifest: Path,
        dataset: Path,
        plan: Path,
        profile_id: str,
    ):
        del manifest, dataset, plan
        offset, _profile_id, rule_ids = next(
            profile
            for profile in FROZEN_SUPPLEMENT_PROFILES
            if profile[1] == profile_id
        )

        def digest(label: str) -> str:
            return hashlib.sha256(
                f"{profile_id}:{label}".encode("utf-8")
            ).hexdigest()

        profile = {
            "id": profile_id,
            "description": "authenticated test supplement",
            "ruleIds": list(rule_ids),
            "evidence": ["fixture"],
        }
        manifest_source = BytesIO(
            json.dumps({"hardNegativeProfile": profile}).encode("utf-8")
        )
        audited = SimpleNamespace(
            source_revision=f"{offset:040x}",
            run_id=digest("run"),
            manifest_sha256=digest("manifest"),
            plan_sha256=digest("plan"),
            outcomes_sha256=digest("outcomes"),
            dataset_sha256=digest("dataset"),
            dataset_bytes=1,
            games=1,
            rows=1,
            max_plies=80,
            observation_policy="single-attempt-allow-partial-v1",
            evaluator_policy_id="stockfish-bestmove-v1",
            evaluator_policy_version=1,
            evaluator_nodes=10_000,
            engine_binary_sha256=BINARY_DIGEST,
            engine_fingerprint=ENGINE_FINGERPRINT,
            agent_ids=AGENT_DOMAIN,
            profile_id=profile_id,
            rule_ids=rule_ids,
            game_assignments=(),
        )
        yield SimpleNamespace(
            audited=audited,
            manifest=manifest_source,
            dataset=BytesIO(),
            plan=BytesIO(),
            verify_unchanged=Mock(),
        )

    def test_inspect_audits_and_counts_without_torch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary) / "dataset.ndjson"
            dataset.write_text(
                "\n".join(
                    json.dumps(value)
                    for value in (row("white", "vegan"), row("black", "checkers"))
                )
                + "\n",
                encoding="utf-8",
            )
            output = StringIO()
            with redirect_stdout(output):
                exit_code = main(["inspect", str(dataset)])
            self.assertEqual(exit_code, 0)
            result = json.loads(output.getvalue())
            self.assertEqual(result["examples"], 2)
            self.assertEqual(sum(result["splits"].values()), 2)

    def test_audit_corpus_reports_content_addressed_smoke_provenance(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = write_fixture(Path(temporary))
            output = StringIO()
            with redirect_stdout(output):
                exit_code = main(
                    [
                        "audit-corpus",
                        str(manifest),
                        "train",
                        "--allow-incomplete-catalog",
                    ]
                )
            self.assertEqual(exit_code, 0)
            result = json.loads(output.getvalue())
            self.assertEqual(result["split"], "train")
            self.assertEqual(result["rows"], 2)
            self.assertEqual(len(result["dataset_sha256"]), 64)

    @staticmethod
    @contextmanager
    def _smoke_release_lease(root: Path, private: Path, dataset: Path, split: str):
        with open_audited_private_corpus_split(
            root,
            private,
            dataset,
            split,
            require_complete_catalog=False,
        ) as lease:
            root_value = json.load(lease.root)
            lease.root.seek(0)
            root_value["corpus"]["agentIds"] = list(AGENT_DOMAIN)
            root_value["corpus"]["maxPlies"] = 80
            root_payload = (
                json.dumps(root_value, ensure_ascii=False, indent=2) + "\n"
            ).encode("utf-8")
            yield SimpleNamespace(
                audited=replace(
                    lease.audited,
                    release_root_sha256=hashlib.sha256(
                        root_payload
                    ).hexdigest(),
                ),
                root=BytesIO(root_payload),
                private_manifest=lease.private_manifest,
                dataset=lease.dataset,
                verify_dataset_unchanged=lease.verify_dataset_unchanged,
            )

    def test_train_release_reuses_pinned_dataset_after_path_replacement(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            root, private, datasets = write_release(directory)
            replacement = directory / "malicious.ndjson"
            replacement.write_bytes(b'{"gameId":"malicious"}\n')
            passes: list[list[str]] = []

            def fake_train(factory, output, config, *, final_validation):
                passes.append([item.game_id for item in factory()])
                try:
                    os.replace(replacement, datasets["train"])
                except PermissionError:
                    self.skipTest(
                        "platform sharing policy disallows replacing an open file"
                    )
                passes.append([item.game_id for item in factory()])
                final_validation(Path("fixture-staging"))

            with patch(
                "drawback_ml.cli.open_audited_private_corpus_split",
                side_effect=self._smoke_release_lease,
            ), patch(
                "drawback_ml.cli.open_audited_hard_negative_train_corpus",
                side_effect=self._smoke_hard_negative_lease,
            ), patch(
                "drawback_ml.cli.train_streaming_baseline",
                side_effect=fake_train,
            ):
                result = main(
                    [
                        "train-release",
                        str(root),
                        str(private["train"]),
                        str(datasets["train"]),
                        str(directory / "output"),
                        "--execution-source-revision",
                        self._EXECUTION_SOURCE_REVISION,
                        "--seed",
                        "7",
                        *self._hard_negative_arguments(directory),
                    ]
                )

            self.assertEqual(result, 0)
            self.assertEqual(passes[0], passes[1])
            self.assertTrue(passes[0])
            self.assertEqual(
                datasets["train"].read_bytes(), b'{"gameId":"malicious"}\n'
            )

    def test_train_release_builds_canonical_authenticated_union(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            root, private, datasets = write_release(directory)
            captured_sources = []
            captured_config = []
            supplement_leases = []

            @contextmanager
            def tracking_hard_negative(*args):
                with self._smoke_hard_negative_lease(*args) as lease:
                    supplement_leases.append(lease)
                    yield lease

            def fake_factory(sources):
                captured_sources.extend(sources)
                return lambda: iter(())

            def fake_train(factory, output, config, *, final_validation):
                del factory, output
                captured_config.append(config)
                final_validation(Path("fixture-staging"))

            with patch(
                "drawback_ml.cli.open_audited_private_corpus_split",
                side_effect=self._smoke_release_lease,
            ), patch(
                "drawback_ml.cli.open_audited_hard_negative_train_corpus",
                side_effect=tracking_hard_negative,
            ), patch(
                "drawback_ml.cli.pinned_multi_source_example_factory",
                side_effect=fake_factory,
            ), patch(
                "drawback_ml.cli.train_streaming_baseline",
                side_effect=fake_train,
            ):
                result = main(
                    [
                        "train-release",
                        str(root),
                        str(private["train"]),
                        str(datasets["train"]),
                        str(directory / "output"),
                        "--execution-source-revision",
                        self._EXECUTION_SOURCE_REVISION,
                        "--seed",
                        "7",
                        *self._hard_negative_arguments(directory),
                    ]
                )

            self.assertEqual(result, 0)
            self.assertEqual(
                [source.namespace for source in captured_sources],
                [
                    "primary",
                    *[
                        f"hard-negative-{offset}-{profile_id}"
                        for offset, profile_id, _rule_ids
                        in FROZEN_SUPPLEMENT_PROFILES
                    ],
                ],
            )
            provenance = captured_config[0].corpus_provenance
            self.assertIsNotNone(provenance)
            self.assertEqual(
                captured_config[0].execution_source_revision,
                self._EXECUTION_SOURCE_REVISION,
            )
            corpus_set = provenance["training_corpus_set"]
            self.assertEqual(
                provenance["training_corpus_set_sha256"],
                corpus_set["sha256"],
            )
            self.assertEqual(
                [
                    item["profile_offset"]
                    for item in corpus_set["supplements"]
                ],
                [101, 102, 103, 104, 105, 106],
            )
            for lease in supplement_leases:
                lease.verify_unchanged.assert_called_once_with()

    def test_train_release_rejects_bad_hard_negative_bindings(self) -> None:
        base = [
            "train-release",
            "root",
            "private",
            "dataset",
            "output",
            "--execution-source-revision",
            self._EXECUTION_SOURCE_REVISION,
            "--seed",
            "7",
        ]
        with self.assertRaisesRegex(ValueError, "exactly six"):
            main(base)

        bindings = self._hard_negative_arguments(Path("fixtures"))
        duplicate = list(bindings)
        duplicate[6] = duplicate[1]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            main([*base, *duplicate])

        unknown = list(bindings)
        unknown[1] = "unknown-profile"
        with self.assertRaisesRegex(ValueError, "unknown"):
            main([*base, *unknown])

    def test_train_release_rejects_supplement_collision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            root, private, datasets = write_release(directory)
            primary_digest = hashlib.sha256(
                datasets["train"].read_bytes()
            ).hexdigest()

            @contextmanager
            def colliding_hard_negative(*args):
                with self._smoke_hard_negative_lease(*args) as lease:
                    lease.audited = SimpleNamespace(
                        **{
                            **vars(lease.audited),
                            "dataset_sha256": primary_digest,
                        }
                    )
                    yield lease

            with patch(
                "drawback_ml.cli.open_audited_private_corpus_split",
                side_effect=self._smoke_release_lease,
            ), patch(
                "drawback_ml.cli.open_audited_hard_negative_train_corpus",
                side_effect=colliding_hard_negative,
            ):
                with self.assertRaisesRegex(
                    TrainingCorpusSetError,
                    "dataset_sha256 values must be unique",
                ):
                    main(
                        [
                            "train-release",
                            str(root),
                            str(private["train"]),
                            str(datasets["train"]),
                            str(directory / "output"),
                            "--execution-source-revision",
                            self._EXECUTION_SOURCE_REVISION,
                            "--seed",
                            "7",
                            *self._hard_negative_arguments(directory),
                        ]
                    )

    def test_train_release_rejects_supplement_mutation_before_publication(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            root, private, datasets = write_release(directory)
            lease_count = 0

            @contextmanager
            def changed_hard_negative(*args):
                nonlocal lease_count
                with self._smoke_hard_negative_lease(*args) as lease:
                    lease_count += 1
                    if lease_count == 1:
                        lease.verify_unchanged.side_effect = CorpusContractError(
                            "pinned supplement changed"
                        )
                    yield lease

            def fake_train(factory, output, config, *, final_validation):
                del factory, output, config
                final_validation(Path("fixture-staging"))

            with patch(
                "drawback_ml.cli.open_audited_private_corpus_split",
                side_effect=self._smoke_release_lease,
            ), patch(
                "drawback_ml.cli.open_audited_hard_negative_train_corpus",
                side_effect=changed_hard_negative,
            ), patch(
                "drawback_ml.cli.train_streaming_baseline",
                side_effect=fake_train,
            ):
                with self.assertRaisesRegex(
                    CorpusContractError,
                    "supplement changed",
                ):
                    main(
                        [
                            "train-release",
                            str(root),
                            str(private["train"]),
                            str(datasets["train"]),
                            str(directory / "output"),
                            "--execution-source-revision",
                            self._EXECUTION_SOURCE_REVISION,
                            "--seed",
                            "7",
                            *self._hard_negative_arguments(directory),
                        ]
                    )

    def test_train_release_rechecks_source_before_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            root, private, datasets = write_release(directory)

            def fake_train(factory, output, config, *, final_validation):
                del factory, output, config
                final_validation(Path("fixture-staging"))

            with patch(
                "drawback_ml.cli._verify_clean_execution_revision",
                side_effect=[
                    self._EXECUTION_SOURCE_REVISION,
                    ValueError("release training source changed"),
                ],
            ), patch(
                "drawback_ml.cli.open_audited_private_corpus_split",
                side_effect=self._smoke_release_lease,
            ), patch(
                "drawback_ml.cli.open_audited_hard_negative_train_corpus",
                side_effect=self._smoke_hard_negative_lease,
            ), patch(
                "drawback_ml.cli.train_streaming_baseline",
                side_effect=fake_train,
            ):
                with self.assertRaisesRegex(ValueError, "source changed"):
                    main(
                        [
                            "train-release",
                            str(root),
                            str(private["train"]),
                            str(datasets["train"]),
                            str(directory / "output"),
                            "--execution-source-revision",
                            self._EXECUTION_SOURCE_REVISION,
                            "--seed",
                            "7",
                            *self._hard_negative_arguments(directory),
                        ]
                    )

    def test_train_release_rejects_mutation_before_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            root, private, datasets = write_release(directory)
            output = directory / "output"

            def fake_train(factory, destination, config, *, final_validation):
                self.assertTrue(list(factory()))
                with staged_output_directory(destination) as staging:
                    (staging / "untrusted.pt").write_bytes(b"untrusted")
                    try:
                        with datasets["train"].open("r+b") as writer:
                            writer.seek(0)
                            writer.write(b"!")
                            writer.flush()
                    except PermissionError:
                        self.skipTest(
                            "platform sharing policy disallows concurrent writes"
                        )
                    final_validation(Path("fixture-staging"))

            with patch(
                "drawback_ml.cli.open_audited_private_corpus_split",
                side_effect=self._smoke_release_lease,
            ), patch(
                "drawback_ml.cli.open_audited_hard_negative_train_corpus",
                side_effect=self._smoke_hard_negative_lease,
            ), patch(
                "drawback_ml.cli.train_streaming_baseline",
                side_effect=fake_train,
            ):
                with self.assertRaisesRegex(CorpusContractError, "changed"):
                    main(
                        [
                            "train-release",
                            str(root),
                            str(private["train"]),
                            str(datasets["train"]),
                            str(output),
                            "--execution-source-revision",
                            self._EXECUTION_SOURCE_REVISION,
                            "--seed",
                            "7",
                            *self._hard_negative_arguments(directory),
                        ]
                    )
            self.assertFalse(output.exists())

    def test_restored_path_cannot_hide_pinned_dataset_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            root, private, datasets = write_release(directory)
            pristine = datasets["train"].read_bytes()
            restored = directory / "restored.ndjson"
            restored.write_bytes(pristine)

            def fake_train(factory, output, config, *, final_validation):
                self.assertTrue(list(factory()))
                try:
                    with datasets["train"].open("r+b") as writer:
                        writer.seek(0)
                        writer.write(b"!")
                        writer.flush()
                    os.replace(restored, datasets["train"])
                except PermissionError:
                    self.skipTest(
                        "platform sharing policy disallows replacing an open file"
                    )
                self.assertEqual(datasets["train"].read_bytes(), pristine)
                final_validation(Path("fixture-staging"))

            with patch(
                "drawback_ml.cli.open_audited_private_corpus_split",
                side_effect=self._smoke_release_lease,
            ), patch(
                "drawback_ml.cli.open_audited_hard_negative_train_corpus",
                side_effect=self._smoke_hard_negative_lease,
            ), patch(
                "drawback_ml.cli.train_streaming_baseline",
                side_effect=fake_train,
            ):
                with self.assertRaisesRegex(CorpusContractError, "changed"):
                    main(
                        [
                            "train-release",
                            str(root),
                            str(private["train"]),
                            str(datasets["train"]),
                            str(directory / "output"),
                            "--execution-source-revision",
                            self._EXECUTION_SOURCE_REVISION,
                            "--seed",
                            "7",
                            *self._hard_negative_arguments(directory),
                        ]
                    )


if __name__ == "__main__":
    unittest.main()
