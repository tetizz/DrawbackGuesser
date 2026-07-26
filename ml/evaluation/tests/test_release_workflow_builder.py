from __future__ import annotations

from contextlib import contextmanager
import chess
import hashlib
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ml.evaluation.browser_parity import (
    PUBLIC_FIXTURE_AGENTS,
    PUBLIC_FIXTURE_FORMAT,
    PUBLIC_FIXTURE_GAME_COUNT,
    PUBLIC_FIXTURE_MAX_PLIES,
    PUBLIC_FIXTURE_PROTOCOL_ID,
    PUBLIC_FIXTURE_ROOT_SEED,
    PUBLIC_FIXTURE_SEED_DOMAIN,
)
from ml.evaluation.release_workflow import build_plan, load_workflow
from ml.evaluation.release_workflow_builder import (
    LOCK_FORMAT,
    WorkflowBuilderError,
    build_release_workflow,
    load_builder_lock,
    write_release_workflow,
)
from ml.evaluation.tests.training_corpus_set_fixture import (
    training_corpus_set_fixture,
)
from ml.evaluation.training_frequency import (
    COUNTING_UNIT,
    FORMAT as FREQUENCY_FORMAT,
    VERSION as FREQUENCY_VERSION,
    _canonical_compact,
    _canonical_pretty,
)
from ml.training.drawback_ml.checkpoint import (
    checkpoint_path,
    write_checkpoint_index,
)
from ml.training.drawback_ml.symbolic_schema import SYMBOLIC_RULE_IDS


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference(root: Path, path: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256(path),
    }


def write_json(path: Path, value: object, *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        if compact
        else json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
    )
    path.write_text(rendered + "\n", encoding="utf-8", newline="\n")


def create_run(
    root: Path,
    seed: int,
    training_corpus_set: dict[str, object],
) -> Path:
    directory = root / "models" / str(seed)
    directory.mkdir(parents=True)
    material = {
        "format": "drawbacktrainer-streaming-run",
        "version": 1,
        "config": {
            "seed": seed,
            "epochs": 8,
            "corpus_provenance": {
                "training_corpus_set_sha256": training_corpus_set["sha256"],
                "training_corpus_set": training_corpus_set,
            },
        },
        "runtime": {"device": "cpu"},
        "sampling": {"policy": "fixture"},
    }
    run_id = hashlib.sha256(
        json.dumps(
            material,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    write_json(
        directory / "run.claim.json",
        {"run_id": run_id, **material},
    )
    for epoch in range(1, 9):
        checkpoint_path(directory, seed, epoch).write_bytes(
            f"{seed}:{epoch}".encode("ascii")
        )
    return write_checkpoint_index(directory, seed=seed, epochs=8)


def frequency_value(
    training_corpus_set: dict[str, object],
) -> dict[str, object]:
    primary = training_corpus_set["primary"]
    supplements = training_corpus_set["supplements"]
    assert isinstance(primary, dict)
    assert isinstance(supplements, list)
    counts = [1] * len(SYMBOLIC_RULE_IDS)
    return {
        "format": FREQUENCY_FORMAT,
        "version": FREQUENCY_VERSION,
        "counting_unit": COUNTING_UNIT,
        "training_corpus_set": training_corpus_set,
        "training_corpus_set_sha256": training_corpus_set["sha256"],
        "rule_ids": list(SYMBOLIC_RULE_IDS),
        "rule_ids_sha256": hashlib.sha256(
            _canonical_compact(list(SYMBOLIC_RULE_IDS))
        ).hexdigest(),
        "counts": {
            "white": counts,
            "black": counts,
            "white_total": len(counts),
            "black_total": len(counts),
        },
        "sources": {
            "primary": {
                "public_root": {
                    "file": "manifest.json",
                    "sha256": primary["release_root_sha256"],
                },
                "private_manifest": {
                    "file": "manifest.json",
                    "sha256": primary["private_train_manifest_sha256"],
                },
                "dataset": {
                    "file": "train.ndjson",
                    "sha256": primary["dataset_sha256"],
                    "bytes": primary["dataset_bytes"],
                },
                "corpus_run_id": primary["corpus_run_id"],
                "outcomes_sha256": primary["outcomes_sha256"],
            },
            "supplements": [
                {
                    "profile_id": item["profile_id"],
                    "profile_offset": item["profile_offset"],
                    "manifest": {
                        "file": f"{item['profile_id']}-manifest.json",
                        "sha256": item["manifest_sha256"],
                    },
                    "dataset": {
                        "file": f"{item['profile_id']}.ndjson",
                        "sha256": item["dataset_sha256"],
                        "bytes": item["dataset_bytes"],
                    },
                    "plan": {
                        "file": f"{item['profile_id']}-plan.json",
                        "sha256": item["plan_sha256"],
                    },
                    "generation_run_id": item["generation_run_id"],
                    "outcomes_sha256": item["outcomes_sha256"],
                }
                for item in supplements
                if isinstance(item, dict)
            ],
        },
    }


def parity_fixture_value() -> dict[str, object]:
    initial_fen = (
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/"
        "RNBQKBNR w KQkq - 0 1"
    )
    probability = 1.0 / len(SYMBOLIC_RULE_IDS)
    games: list[dict[str, object]] = []
    for index in range(PUBLIC_FIXTURE_GAME_COUNT):
        board = chess.Board()
        legal = [move.uci() for move in board.legal_moves]
        pgn = (
            f'[Event "Public parity {index + 1}"]\n'
            '[Result "*"]\n\n'
            "1. e4 *\n"
        )
        board.push(chess.Move.from_uci("e2e4"))
        games.append(
            {
                "id": f"public-parity-{index + 1:02d}",
                "seed": index + 1,
                "pgn": pgn,
                "pgnSha256": hashlib.sha256(pgn.encode("utf-8")).hexdigest(),
                "plyCount": 1,
                "initialFen": initial_fen,
                "finalFen": board.fen(en_passant="legal"),
                "result": "*",
                "finalPublicObservation": {
                    "fenBefore": initial_fen,
                    "move": "e2e4",
                    "moveNumber": 1,
                    "ply": 0,
                    "playerColor": "white",
                    "historySan": [],
                    "ordinaryLegalMoves": legal,
                    "symbolicFeatureVersion": 6,
                    "symbolic": {
                        "ruleIds": list(SYMBOLIC_RULE_IDS),
                        "whiteProbabilities": [
                            probability
                        ] * len(SYMBOLIC_RULE_IDS),
                        "blackProbabilities": [
                            probability
                        ] * len(SYMBOLIC_RULE_IDS),
                        "whiteEliminated": [
                            False
                        ] * len(SYMBOLIC_RULE_IDS),
                        "blackEliminated": [
                            False
                        ] * len(SYMBOLIC_RULE_IDS),
                    },
                },
            }
        )
    return {
        "format": PUBLIC_FIXTURE_FORMAT,
        "version": 1,
        "protocol": {
            "id": PUBLIC_FIXTURE_PROTOCOL_ID,
            "seedDomain": PUBLIC_FIXTURE_SEED_DOMAIN,
            "rootSeed": PUBLIC_FIXTURE_ROOT_SEED,
            "gameCount": PUBLIC_FIXTURE_GAME_COUNT,
            "maxPlies": PUBLIC_FIXTURE_MAX_PLIES,
            "agentSchedule": list(PUBLIC_FIXTURE_AGENTS),
        },
        "candidateInputs": [],
        "games": games,
    }


def create_lock(
    root: Path,
    *,
    real_artifacts: bool = False,
) -> tuple[Path, dict[str, object]]:
    tools: dict[str, object] = {}
    for name in ("browser", "git", "node", "pnpm"):
        path = root / "tools" / f"{name}.exe"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode("ascii"))
        tools[name] = {"path": str(path.resolve()), "sha256": sha256(path)}
    dataset = root / "data" / "generated" / "corpus" / "validation.ndjson"
    dataset.parent.mkdir(parents=True)
    dataset.write_bytes(b"validation")
    private = (
        root
        / "data"
        / "releases"
        / "current"
        / "private"
        / "validation"
        / "manifest.json"
    )
    write_json(
        private,
        {
            "manifestVersion": 1,
            "corpusRunId": "corpus",
            "split": "validation",
            "dataset": {
                "file": dataset.name,
                "bytes": dataset.stat().st_size,
                "sha256": sha256(dataset),
            },
        },
    )
    public = (
        root
        / "data"
        / "releases"
        / "current"
        / "public"
        / "manifest.json"
    )
    write_json(
        public,
        {
            "releaseManifestVersion": 1,
            "corpusRunId": "corpus",
            "splits": {
                "validation": {
                    "datasetBytes": dataset.stat().st_size,
                    "datasetSha256": sha256(dataset),
                    "privateManifestSha256": sha256(private),
                }
            },
        },
    )
    training_corpus_set = training_corpus_set_fixture(
        release_root_sha256=sha256(public),
    )
    shared: dict[str, object] = {
        "dataset": reference(root, dataset),
        "publicRoot": reference(root, public),
        "privateValidation": reference(root, private),
    }
    frequency = root / "inputs" / "training-frequency.json"
    fixture = root / "inputs" / "public-parity-fixture.json"
    frequency.parent.mkdir(parents=True)
    if real_artifacts:
        frequency.write_bytes(
            _canonical_pretty(frequency_value(training_corpus_set))
        )
        fixture.write_bytes(_canonical_pretty(parity_fixture_value()))
    else:
        frequency.write_bytes(b"frequency")
        fixture.write_bytes(b"fixture")
    value: dict[str, object] = {
        "format": LOCK_FORMAT,
        "version": 1,
        "sourceRevision": "a" * 40,
        "tools": tools,
        "shared": shared,
        "candidates": [
            {
                "seed": seed,
                "checkpointIndex": reference(
                    root,
                    create_run(root, seed, training_corpus_set),
                ),
            }
            for seed in (20260811, 20260812, 20260813)
        ],
        "trainingFrequency": reference(root, frequency),
        "browserFixture": reference(root, fixture),
        "outputRoot": "data/generated/frozen-release",
    }
    path = root / "builder-lock.json"
    write_json(path, value, compact=True)
    return path, value


@contextmanager
def mocked_artifact_loaders():
    with (
        patch(
            "ml.evaluation.release_workflow_builder.load_training_run",
            return_value=SimpleNamespace(
                training_corpus_set_sha256="c" * 64,
            ),
        ),
        patch(
            "ml.evaluation.release_workflow_builder."
            "load_training_frequency_artifact"
        ),
        patch(
            "ml.evaluation.release_workflow_builder."
            "load_public_parity_fixture"
        ),
    ):
        yield


class ReleaseWorkflowBuilderTests(unittest.TestCase):
    def test_real_loaders_compose_with_canonical_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock_path, _value = create_lock(root, real_artifacts=True)
            workflow = build_release_workflow(
                load_builder_lock(lock_path),
                root,
            )

            self.assertEqual(
                workflow["format"],
                "drawbacktrainer-post-training-release-workflow",
            )

    def test_builds_exact_closed_workflow_without_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock_path, _value = create_lock(root)
            lock = load_builder_lock(lock_path)
            with (
                mocked_artifact_loaders(),
                patch.object(Path, "glob", side_effect=AssertionError("glob")),
                patch.object(
                    Path,
                    "iterdir",
                    side_effect=AssertionError("iterdir"),
                ),
            ):
                workflow = build_release_workflow(lock, root)
            plan = build_plan(workflow, root)

            self.assertEqual(len(plan), 59)
            self.assertEqual(
                [
                    candidate["seed"]
                    for candidate in workflow["candidates"]  # type: ignore[index]
                ],
                [20260811, 20260812, 20260813],
            )
            evaluations = [
                step for step in plan if step.stage == "selection-fit-evaluation"
            ]
            self.assertEqual(len(evaluations), 24)
            self.assertTrue(
                all(
                    step.argv[step.argv.index("--validation-partition") + 1]
                    == "selection"
                    for step in evaluations
                )
            )
            self.assertFalse(
                any(
                    "sealed" in " ".join(step.argv).lower()
                    or "--split test" in " ".join(step.argv).lower()
                    for step in plan
                )
            )
            ensemble_step = next(
                step for step in plan if step.stage == "ensemble-release"
            )
            fusion_step = next(
                step for step in plan if step.stage == "fusion-selection"
            )
            calibration_step = next(
                step
                for step in plan
                if step.stage == "calibration-evaluation"
            )
            self.assertEqual(
                fusion_step.outputs,
                (
                    ensemble_step.outputs[0].with_name(
                        "fusion-selection.json"
                    ),
                ),
            )
            self.assertEqual(
                fusion_step.argv[11],
                "select-ensemble-fusion",
            )
            self.assertEqual(
                calibration_step.argv[
                    calibration_step.argv.index("--fusion-selection") + 1
                ],
                str(fusion_step.outputs[0]),
            )
            ensemble_directory = ensemble_step.outputs[0].parent
            training_run_paths = [
                Path(ensemble_step.argv[index + 1])
                for index, value in enumerate(ensemble_step.argv)
                if value == "--training-run"
            ]
            self.assertEqual(len(training_run_paths), 3)
            self.assertTrue(
                all(
                    path.parent != ensemble_directory
                    for path in training_run_paths
                )
            )

    def test_publishes_canonical_no_clobber_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock_path, _value = create_lock(root)
            output = root / "data" / "generated" / "workflow.json"
            output.parent.mkdir(parents=True, exist_ok=True)

            with mocked_artifact_loaders():
                published, digest = write_release_workflow(
                    lock_path,
                    output,
                    repository=root,
                )
            loaded, loaded_digest = load_workflow(published)

            self.assertEqual(digest, loaded_digest)
            self.assertEqual(loaded["sourceRevision"], "a" * 40)
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                with mocked_artifact_loaders():
                    write_release_workflow(
                        lock_path,
                        output,
                        repository=root,
                    )

    def test_publication_failure_is_clean_and_concurrent_winner_is_complete(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock_path, _value = create_lock(root)
            output = root / "data" / "generated" / "workflow.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            with patch(
                "ml.evaluation.release_workflow_builder.os.fsync",
                side_effect=OSError("disk failure"),
            ):
                with self.assertRaisesRegex(OSError, "disk failure"):
                    with mocked_artifact_loaders():
                        write_release_workflow(
                            lock_path,
                            output,
                            repository=root,
                        )
            self.assertFalse(output.exists())
            self.assertFalse(tuple(output.parent.glob(".release-workflow.*.tmp")))

            results: list[str] = []
            barrier = threading.Barrier(2)

            def publish() -> None:
                barrier.wait()
                try:
                    _path, digest = write_release_workflow(
                        lock_path,
                        output,
                        repository=root,
                    )
                    results.append(digest)
                except FileExistsError:
                    results.append("exists")

            first = threading.Thread(target=publish)
            second = threading.Thread(target=publish)
            with mocked_artifact_loaders():
                first.start()
                second.start()
                first.join()
                second.join()

            self.assertEqual(results.count("exists"), 1)
            self.assertEqual(len([item for item in results if item != "exists"]), 1)
            _workflow, digest = load_workflow(output)
            self.assertIn(digest, results)
            self.assertFalse(tuple(output.parent.glob(".release-workflow.*.tmp")))

    def test_output_must_use_an_existing_generated_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock_path, _value = create_lock(root)
            with self.assertRaisesRegex(
                WorkflowBuilderError,
                "data/generated",
            ):
                with mocked_artifact_loaders():
                    write_release_workflow(
                        lock_path,
                        root / "docs" / "workflow.json",
                        repository=root,
                    )
            with self.assertRaisesRegex(
                WorkflowBuilderError,
                "parent must already exist",
            ):
                with mocked_artifact_loaders():
                    write_release_workflow(
                        lock_path,
                        root
                        / "data"
                        / "generated"
                        / "missing"
                        / "workflow.json",
                        repository=root,
                    )
            self.assertFalse((root / "docs").exists())
            self.assertFalse(
                (root / "data" / "generated" / "missing").exists()
            )

    def test_rejects_output_collision_with_release_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock_path, _value = create_lock(root)
            output = (
                root
                / "data"
                / "generated"
                / "frozen-release"
                / "ensemble.json"
            )
            output.parent.mkdir(parents=True)

            with self.assertRaisesRegex(
                WorkflowBuilderError,
                "collides",
            ):
                with mocked_artifact_loaders():
                    write_release_workflow(
                        lock_path,
                        output,
                        repository=root,
                    )

    def test_rejects_validation_data_outside_authenticated_release(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock_path, value = create_lock(root)
            hostile = (
                root
                / "data"
                / "generated"
                / "hostile"
                / "validation.ndjson"
            )
            hostile.parent.mkdir(parents=True)
            hostile.write_bytes(b"hostile")
            shared = value["shared"]
            assert isinstance(shared, dict)
            shared["dataset"] = reference(root, hostile)
            write_json(lock_path, value, compact=True)

            lock = load_builder_lock(lock_path)
            with self.assertRaisesRegex(
                WorkflowBuilderError,
                "not bound by the current release",
            ):
                with mocked_artifact_loaders():
                    build_release_workflow(lock, root)

    def test_rejects_reorder_tamper_noncanonical_and_unknown_fields(self) -> None:
        mutations = {
            "reordered": lambda value: value["candidates"].reverse(),
            "unknown": lambda value: value.__setitem__(
                "sealedTest",
                {"path": "secret"},
            ),
            "boolean-version": lambda value: value.__setitem__(
                "version",
                True,
            ),
            "float-version": lambda value: value.__setitem__(
                "version",
                1.0,
            ),
            "wrong-index-hash": lambda value: value["candidates"][0][
                "checkpointIndex"
            ].__setitem__("sha256", "0" * 64),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                lock_path, value = create_lock(root)
                mutate(value)
                write_json(lock_path, value, compact=True)
                with self.assertRaises((ValueError, WorkflowBuilderError)):
                    lock = load_builder_lock(lock_path)
                    with mocked_artifact_loaders():
                        build_release_workflow(lock, root)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock_path, value = create_lock(root)
            write_json(lock_path, value, compact=False)
            with self.assertRaisesRegex(WorkflowBuilderError, "canonical"):
                load_builder_lock(lock_path)

    def test_direct_builder_call_rejects_unknown_lock_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock_path, _value = create_lock(root)
            lock = dict(load_builder_lock(lock_path))
            lock["sealedTest"] = {
                "path": "data/releases/current/private/test/manifest.json"
            }

            with self.assertRaisesRegex(
                WorkflowBuilderError,
                "fields are invalid",
            ):
                with mocked_artifact_loaders():
                    build_release_workflow(lock, root)


if __name__ == "__main__":
    unittest.main()
