from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import _bootstrap  # noqa: F401

from drawback_ml.capturable_baseline import _canonical_json
from drawback_ml.capturable_blend_contract import blend_reliability_checks
from drawback_ml.capturable_opportunity_workflow import (
    CORPUS_LEDGER_FORMAT,
    CORPUS_LEDGER_SPLITS,
    CORPUS_LEDGER_VERSION,
    FROZEN_CONFIG,
    LEDGER_VERIFICATION_FORMAT,
    LEDGER_VERIFICATION_VERSION,
    MODEL_SEEDS,
    OPPORTUNITY_WORKFLOW_VERSION,
    SEALED_TEST_FORMAT,
    SPLIT_SEED_ROOTS,
    _LoadedPair,
    _Pair,
    _delta,
    _load_corpus_ledger,
    _ledger_verification_input_set,
    _protocol,
    consumption_marker_path,
    ledger_verification_receipt_path,
    load_sealed_test,
    load_stage_a,
    load_stage_b,
    main,
    run_sealed_test,
    run_stage_a,
    run_stage_b,
)
from drawback_ml.capturable_records import (
    CAPTURABLE_OPPORTUNITY_FIELDS,
    CAPTURABLE_OPPORTUNITY_SHAPE,
    CAPTURABLE_RULE_IDS,
    CapturableDatasetError,
)


_COMMIT = "1" * 40
_CONVERTER_COMMIT = "2" * 40
_SPLIT_SHA = {
    "train": "b" * 64,
    "validation-a": "c" * 64,
    "validation-b": "e" * 64,
    "test": "9" * 64,
}
_SPLIT_SEED_BASE = {
    "train": 10_000,
    "validation-a": 20_000,
    "validation-b": 30_000,
    "test": 40_000,
}


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_set_sha(values: list[str] | list[int]) -> str:
    return hashlib.sha256(_canonical_json(values)).hexdigest()


def _assignment_sha(
    artifact: dict[str, object],
    field: str,
) -> str:
    splits = artifact["splits"]
    assert isinstance(splits, list)
    payload = [
        {
            "split": split["split"],
            "values": split["sourceTrace"][field],
        }
        for split in splits
    ]
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _resign_ledger(artifact: dict[str, object]) -> str:
    splits = artifact["splits"]
    assert isinstance(splits, list)
    for split in splits:
        for identity_name in ("sourceTrace", "converted"):
            identity = split[identity_name]
            identity["gameIds"].sort()
            identity["simulationSeeds"].sort()
            identity["gameIdSetSha256"] = _canonical_set_sha(
                identity["gameIds"]
            )
            identity["simulationSeedSetSha256"] = _canonical_set_sha(
                identity["simulationSeeds"]
            )
        source = split["sourceTrace"]
        source["parameterSeeds"].sort()
        source["parameterSeedSetSha256"] = _canonical_set_sha(
            source["parameterSeeds"]
        )
    partition = artifact["partition"]
    partition["games"] = sum(
        split["sourceTrace"]["games"] for split in splits
    )
    partition["gameIdAssignmentsSha256"] = _assignment_sha(
        artifact,
        "gameIds",
    )
    partition["simulationSeedAssignmentsSha256"] = _assignment_sha(
        artifact,
        "simulationSeeds",
    )
    partition["parameterSeedAssignmentsSha256"] = _assignment_sha(
        artifact,
        "parameterSeeds",
    )
    artifact.pop("contentSha256", None)
    artifact["contentSha256"] = hashlib.sha256(
        _canonical_json(artifact)
    ).hexdigest()
    return hashlib.sha256(_canonical_json(artifact)).hexdigest()


def _resign_runtime(identity: dict[str, object]) -> None:
    identity.pop("aggregateSha256", None)
    identity["aggregateSha256"] = hashlib.sha256(
        _canonical_json(identity)
    ).hexdigest()


def _ledger_artifact() -> dict[str, object]:
    component = {
        "entrypoint": "fixture-code",
        "files": 1,
        "bytes": 1,
        "sha256": "8" * 64,
    }
    execution: dict[str, object] = {
        "algorithm": "sha256-loaded-module-graph-v2",
        "runtime": {
            "nodeVersion": "v24.0.0",
            "platform": "win32",
            "architecture": "x64",
            "execArgv": [],
        },
        "parser": dict(component),
        "converter": dict(component),
        "scheduler": dict(component),
        "verifier": dict(component),
    }
    execution["aggregateSha256"] = hashlib.sha256(
        _canonical_json(execution)
    ).hexdigest()
    runtime_component = {
        "componentId": "schema9-coordinator/v1",
        "files": 17,
        "bytes": 1_234,
        "sha256": "1" * 64,
    }
    producer_runtime: dict[str, object] = {
        "format": "drawbackengine-schema9-producer-runtime",
        "version": 1,
        "algorithm": "sha256-engine-runtime-tree-v1",
        "runtime": {
            "nodeVersion": "v24.0.0",
            "platform": "win32",
            "architecture": "x64",
            "execArgv": [],
        },
        "coordinator": runtime_component,
        "parallelWorker": {
            "componentId": "player-private-parallel-worker/v1",
            "files": 13,
            "bytes": 987,
            "sha256": "2" * 64,
        },
    }
    producer_runtime["aggregateSha256"] = hashlib.sha256(
        _canonical_json(producer_runtime)
    ).hexdigest()
    splits: list[dict[str, object]] = []
    for split_index, split_name in enumerate(CORPUS_LEDGER_SPLITS):
        game_ids = [
            f"{split_name}-game-{index:04d}"
            for index in range(2_500)
        ]
        seeds = [
            _SPLIT_SEED_BASE[split_name] + index
            for index in range(2_500)
        ]
        parameter_seeds = [
            100_000 + (split_index * 10_000) + index
            for index in range(5_000)
        ]
        workflow_split = (
            "sealed-test" if split_name == "test" else split_name
        )
        label_counts = {
            rule_id: 100 for rule_id in CAPTURABLE_RULE_IDS
        }
        splits.append(
            {
                "split": split_name,
                "scheduleId": f"schema9-{split_name}",
                "seedRoots": list(SPLIT_SEED_ROOTS[workflow_split]),
                "producerEngineCommit": _CONVERTER_COMMIT,
                "producerRuntimeIdentity": deepcopy(producer_runtime),
                "generatorReceipts": {
                    "launch": {
                        "sha256": _sha(f"{split_name}-launch"),
                        "bytes": 100,
                    },
                    "completion": {
                        "sha256": _sha(f"{split_name}-completion"),
                        "bytes": 120,
                    },
                },
                "scheduleProfile": {
                    "id": "standard",
                    "policyId": "material-player-private-corpus/v1",
                },
                "sourceTrace": {
                    "sha256": _sha(f"{split_name}-trace"),
                    "bytes": 10_000,
                    "games": 2_500,
                    "zeroPlyGames": 0,
                    "gameIds": list(game_ids),
                    "simulationSeeds": list(seeds),
                    "parameterSeeds": parameter_seeds,
                    "gameIdSetSha256": "",
                    "simulationSeedSetSha256": "",
                    "parameterSeedSetSha256": "",
                    "labelCountsByColor": {
                        "white": dict(label_counts),
                        "black": dict(label_counts),
                    },
                },
                "converted": {
                    "sha256": _SPLIT_SHA[split_name],
                    "bytes": 20_000,
                    "rows": 2_500,
                    "games": 2_500,
                    "gameIds": list(game_ids),
                    "simulationSeeds": list(seeds),
                    "gameIdSetSha256": "",
                    "simulationSeedSetSha256": "",
                },
            }
        )
    artifact: dict[str, object] = {
        "format": CORPUS_LEDGER_FORMAT,
        "version": CORPUS_LEDGER_VERSION,
        "identity": {
            "guesserCommit": _COMMIT,
            "converterEngineCommit": _CONVERTER_COMMIT,
            "producerConverterPolicy": "exact/v1",
            "execution": execution,
            "producerRuntimeIdentity": producer_runtime,
        },
        "scheduleContract": {
            "authorityId": "capturable25-schema9-opportunity/v1",
            "seedStreams": ["label", "gameplay", "parameters"],
        },
        "opportunityContract": {
            "authorityId": "capturable-king/v1",
            "symbolicFeatureVersion": 9,
            "opportunityFeatureVersion": 1,
            "ruleIds": list(CAPTURABLE_RULE_IDS),
            "fields": list(CAPTURABLE_OPPORTUNITY_FIELDS),
            "shape": list(CAPTURABLE_OPPORTUNITY_SHAPE),
        },
        "splits": splits,
        "partition": {
            "games": 0,
            "gameIdAssignmentsSha256": "",
            "simulationSeedAssignmentsSha256": "",
            "parameterSeedAssignmentsSha256": "",
        },
    }
    _resign_ledger(artifact)
    return artifact


class _LedgerFixture:
    def __init__(
        self,
        root: Path,
        mutate=None,
    ) -> None:
        self.path = root / "schema9-corpus-ledger.json"
        self.artifact = _ledger_artifact()
        if mutate is not None:
            mutate(self.artifact)
            _resign_ledger(self.artifact)
        payload = _canonical_json(self.artifact)
        self.path.write_bytes(payload)
        self.sha256 = hashlib.sha256(payload).hexdigest()
        receipt: dict[str, object] = {
            "format": LEDGER_VERIFICATION_FORMAT,
            "version": LEDGER_VERIFICATION_VERSION,
            "ledger": {
                "sha256": self.sha256,
                "contentSha256": self.artifact["contentSha256"],
            },
            "repository": self.artifact["identity"],
            "inputSetSha256": hashlib.sha256(
                _canonical_json(_ledger_verification_input_set(self.artifact))
            ).hexdigest(),
            "verificationPolicy": {
                "repository": "head-clean-content-manifest/v1",
                "schedule": "engine-scheduler-replay/v1",
                "corpus": "full-byte-reauthentication/v1",
            },
        }
        receipt["contentSha256"] = hashlib.sha256(
            _canonical_json(receipt)
        ).hexdigest()
        self.verification_receipt_path = ledger_verification_receipt_path(
            self.path,
            self.sha256,
        )
        verification_receipt_payload = _canonical_json(receipt)
        self.verification_receipt_path.write_bytes(
            verification_receipt_payload
        )
        self.verification_receipt_sha256 = hashlib.sha256(
            verification_receipt_payload
        ).hexdigest()

    @property
    def reference(self) -> dict[str, object]:
        identity = self.artifact["identity"]
        return {
            "sha256": self.sha256,
            "contentSha256": self.artifact["contentSha256"],
            "verificationReceiptSha256": (
                self.verification_receipt_sha256
            ),
            "guesserCommit": identity["guesserCommit"],
            "converterEngineCommit": identity["converterEngineCommit"],
            "producerConverterPolicy": identity[
                "producerConverterPolicy"
            ],
            "executionAggregateSha256": identity["execution"][
                "aggregateSha256"
            ],
            "producerRuntimeAggregateSha256": identity[
                "producerRuntimeIdentity"
            ]["aggregateSha256"],
            "sealedCorpusIdentitySha256": self._sealed_identity(),
        }

    @property
    def workflow_auth_kwargs(self) -> dict[str, str]:
        return {
            "corpus_ledger_sha256": self.sha256,
            "corpus_ledger_verification_receipt_sha256": (
                self.verification_receipt_sha256
            ),
        }

    def _sealed_identity(self) -> str:
        test_split = self.split("test")
        converted = test_split["converted"]
        return hashlib.sha256(
            _canonical_json(
                {
                    "domain": "capturable25-schema9-opportunity-v1",
                    "workflowVersion": OPPORTUNITY_WORKFLOW_VERSION,
                    "ledgerSha256": self.sha256,
                    "ledgerContentSha256": self.artifact["contentSha256"],
                    "testConvertedSha256": converted["sha256"],
                    "testGameIdSetSha256": converted["gameIdSetSha256"],
                    "testSimulationSeedSetSha256": converted[
                        "simulationSeedSetSha256"
                    ],
                    "testParameterSeedSetSha256": test_split["sourceTrace"][
                        "parameterSeedSetSha256"
                    ],
                    "scheduleId": test_split["scheduleId"],
                    "seedRoots": test_split["seedRoots"],
                }
            )
        ).hexdigest()

    def split(self, name: str) -> dict[str, object]:
        return next(
            split
            for split in self.artifact["splits"]
            if split["split"] == name
        )

    def rows(
        self,
        name: str,
        *,
        first_seed: int | None = None,
    ) -> tuple[SimpleNamespace, ...]:
        converted = self.split(name)["converted"]
        seeds = list(converted["simulationSeeds"])
        if first_seed is not None:
            seeds[0] = first_seed
        return tuple(
            SimpleNamespace(
                evaluation=SimpleNamespace(game_id=game_id, seed=seed)
            )
            for game_id, seed in zip(
                converted["gameIds"],
                seeds,
                strict=True,
            )
        )

    def input_identity(self, name: str) -> dict[str, object]:
        converted = self.split(name)["converted"]
        return {
            "path": f"{name}-schema9.ndjson",
            "sha256": converted["sha256"],
            "rows": converted["rows"],
            "games": converted["games"],
        }

    def rewrite(self, mutate) -> None:
        mutate(self.artifact)
        self.artifact.pop("contentSha256", None)
        self.artifact["contentSha256"] = hashlib.sha256(
            _canonical_json(self.artifact)
        ).hexdigest()
        payload = _canonical_json(self.artifact)
        self.path.write_bytes(payload)
        self.sha256 = hashlib.sha256(payload).hexdigest()

    def bound_identity(
        self,
        name: str,
        *,
        file: str | None = None,
    ) -> dict[str, object]:
        split = self.split(name)
        converted = split["converted"]
        return {
            "file": file or f"{name}-schema9.ndjson",
            "sha256": converted["sha256"],
            "rows": converted["rows"],
            "games": converted["games"],
            "gameIdSetSha256": converted["gameIdSetSha256"],
            "simulationSeedSetSha256": converted[
                "simulationSeedSetSha256"
            ],
            "scheduleId": split["scheduleId"],
            "seedRoots": split["seedRoots"],
        }


def _metrics(
    *,
    top1: float,
    top3: float,
    top5: float,
    nll: float,
    brier: float = 0.8,
    calibration: float = 0.1,
    hard_violations: int = 0,
) -> dict[str, object]:
    hybrid = {
        "accuracy_after_moves": {
            "5": top1,
            "10": top1,
            "15": top1,
            "20": top1,
        },
        "expected_calibration_error": calibration,
        "game_normalized_brier_score": brier,
        "game_normalized_negative_log_likelihood": nll,
        "game_normalized_top_1_accuracy": top1,
        "game_normalized_top_3_accuracy": top3,
        "game_normalized_top_5_accuracy": top5,
        "hidden_parameter_accuracy": 0.5,
        "metrics_per_drawback": {
            drawback_id: {"top_1_accuracy": top1}
            for drawback_id in CAPTURABLE_RULE_IDS
        },
        "probability_diagnostics": {
            "checked_count": 100,
            "hard_elimination_violation_count": hard_violations,
            "hard_mask_checked_count": 100,
            "maximum_eliminated_probability": (
                0.0 if hard_violations == 0 else 0.1
            ),
            "missing_hard_mask_count": 0,
        },
    }
    color = {
        "game_normalized_negative_log_likelihood": nll,
        "game_normalized_top_1_accuracy": top1,
        "game_normalized_top_3_accuracy": top3,
    }
    binary = {
        "accuracy": 0.8,
        "brier_score": 0.2,
        "negative_log_likelihood": 0.5,
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


class _PairFixtures:
    def __init__(self, root: Path, ledger: _LedgerFixture) -> None:
        self.root = root
        self.ledger = ledger
        self.comparisons: dict[str, tuple[dict[str, object], str]] = {}
        self.candidates: dict[
            tuple[str, str],
            tuple[dict[str, object], dict[str, object]],
        ] = {}
        self.paths: list[Path] = []

    def add(
        self,
        seed: int,
        top1_delta: float,
        *,
        nll_delta: float = -0.01,
        top5_delta: float = 0.01,
        decision: str = "promote-treatment",
        white_top3_delta: float | None = None,
        black_nll_delta: float | None = None,
        calibration_delta: float | None = None,
        config_override: dict[str, object] | None = None,
        hard_violations: int = 0,
    ) -> Path:
        path = self.root / f"pair-{seed}.json"
        path.touch()
        self.paths.append(path)
        config = {
            "seed": seed,
            **FROZEN_CONFIG,
            "learning_rate": 0.001,
            "torch_threads": 1,
        }
        if config_override is not None:
            config.update(config_override)
        train_input = self.ledger.input_identity("train")
        validation_input = self.ledger.input_identity("validation-a")
        control_metrics = _metrics(
            top1=0.30,
            top3=0.50,
            top5=0.60,
            nll=2.0,
            hard_violations=hard_violations,
        )
        treatment_metrics = _metrics(
            top1=0.30 + top1_delta,
            top3=0.50 + top1_delta,
            top5=0.60 + top5_delta,
            nll=2.0 + nll_delta,
            brier=0.8 - max(top1_delta, 0.0),
            calibration=(
                0.1 - max(top1_delta, 0.0) / 2
                if calibration_delta is None
                else 0.1 + calibration_delta
            ),
            hard_violations=hard_violations,
        )
        if white_top3_delta is not None:
            treatment_metrics["hybridByColor"]["white"][
                "game_normalized_top_3_accuracy"
            ] = 0.50 + white_top3_delta
        if black_nll_delta is not None:
            treatment_metrics["hybridByColor"]["black"][
                "game_normalized_negative_log_likelihood"
            ] = 2.0 + black_nll_delta

        def candidate(
            mode: str,
            metrics: dict[str, object],
        ) -> tuple[dict[str, object], dict[str, object]]:
            directory = f"{mode}-{seed}"
            base = {
                "selectionDirectory": directory,
                "selectionReport": "selection.json",
                "selectionReportSha256": _sha(f"selection-{mode}-{seed}"),
                "checkpointFile": "model.pt",
                "checkpointSha256": _sha(f"checkpoint-{mode}-{seed}"),
                "seed": seed,
                "triggerRowMultiplier": 1.0,
                "validationGameNormalizedTop1": metrics["hybrid"][
                    "game_normalized_top_1_accuracy"
                ],
                "validationGameNormalizedTop3": metrics["hybrid"][
                    "game_normalized_top_3_accuracy"
                ],
                "validationGameNormalizedNll": metrics["hybrid"][
                    "game_normalized_negative_log_likelihood"
                ],
                "opportunityContract": {
                    "symbolicFeatureVersion": 9,
                    "opportunityFeatureVersion": 1,
                    "opportunityRuleIds": list(CAPTURABLE_RULE_IDS),
                    "opportunityFields": list(
                        CAPTURABLE_OPPORTUNITY_FIELDS
                    ),
                    "opportunityShape": list(
                        CAPTURABLE_OPPORTUNITY_SHAPE
                    ),
                    "opportunityMode": mode,
                },
            }
            report = {
                "config": dict(config),
                "inputs": {
                    "train": deepcopy(train_input),
                    "validation": deepcopy(validation_input),
                },
                "validation": deepcopy(metrics),
            }
            self.candidates[(directory, "selection.json")] = (
                deepcopy(base),
                report,
            )
            return ({**base, "trainInput": deepcopy(train_input)}, report)

        control, _ = candidate("zero-ablation", control_metrics)
        treatment, _ = candidate("public-exact", treatment_metrics)
        comparison = {
            "validationInput": validation_input,
            "control": control,
            "treatments": [treatment],
            "bestTreatment": treatment,
            "releaseDecision": decision,
        }
        self.comparisons[path.name] = (comparison, _sha(path.name))
        return path

    def patches(self):
        def load_comparison(path: Path):
            return deepcopy(self.comparisons[path.name])

        def validated_candidate(path: Path):
            candidate, report = self.candidates[
                (path.parent.name, path.name)
            ]
            return deepcopy(candidate), deepcopy(report)

        return (
            patch(
                "drawback_ml.capturable_opportunity_workflow."
                "load_treatment_comparison",
                side_effect=load_comparison,
            ),
            patch(
                "drawback_ml.capturable_opportunity_workflow."
                "_validated_candidate",
                side_effect=validated_candidate,
            ),
        )


def _paired_result(*, promote: bool = True) -> dict[str, object]:
    control_metrics = _metrics(
        top1=0.30,
        top3=0.50,
        top5=0.60,
        nll=2.0,
    )
    treatment_metrics = _metrics(
        top1=0.31 if promote else 0.29,
        top3=0.51 if promote else 0.49,
        top5=0.61 if promote else 0.59,
        nll=1.99 if promote else 2.01,
        brier=0.79 if promote else 0.81,
        calibration=0.09 if promote else 0.11,
    )
    synthetic = _Pair(
        comparison_path=Path("test"),
        comparison={"control": {"seed": MODEL_SEEDS[0]}},
        comparison_sha256="0" * 64,
        control_report={"validation": control_metrics},
        treatment_report={"validation": treatment_metrics},
    )
    deltas = _delta(synthetic)
    primary = float(deltas["top1"]) > 0.0
    checks = blend_reliability_checks(
        control_metrics,
        treatment_metrics,
        primary,
    )
    return {
        "control": {
            "checkpoint": {"file": "model.pt", "sha256": "1" * 64},
            "selection": {
                "selectedEpoch": 8,
                "selectedFusionAlpha": 1.0,
                "selectedPriorSmoothing": 0.1,
            },
            "metrics": control_metrics,
        },
        "treatment": {
            "checkpoint": {"file": "model.pt", "sha256": "2" * 64},
            "selection": {
                "selectedEpoch": 8,
                "selectedFusionAlpha": 1.0,
                "selectedPriorSmoothing": 0.1,
            },
            "metrics": treatment_metrics,
        },
        "deltas": deltas,
        "primaryDecision": (
            "confirm-treatment" if primary else "reject-treatment"
        ),
        "reliabilityChecks": checks,
        "decision": (
            "promote-treatment"
            if all(checks.values())
            else "retain-control"
        ),
    }


def _frozen_pair() -> dict[str, object]:
    return {
        "file": "pair.json",
        "control": {
            "checkpointFile": "model.pt",
            "checkpointSha256": "1" * 64,
        },
        "treatment": {
            "checkpointFile": "model.pt",
            "checkpointSha256": "2" * 64,
        },
    }


def _stage_b_fixture(
    ledger: _LedgerFixture,
    stage_b_path: Path,
    stage_b_sha256: str,
    *,
    authorization: str = "sealed-test-authorized",
) -> dict[str, object]:
    return {
        "authorization": authorization,
        "corpusLedger": ledger.reference,
        "frozenPair": _frozen_pair(),
        "stageA": {"file": "stage-a.json"},
        "validationBInput": ledger.bound_identity("validation-b"),
        "result": _paired_result(),
        "_stageBFile": stage_b_path.name,
        "_stageBSha256": stage_b_sha256,
    }


def _sealed_artifact(
    ledger: _LedgerFixture,
    stage_b_path: Path,
    stage_b_sha256: str,
    stage_b: dict[str, object],
) -> dict[str, object]:
    visible_stage_b = {
        key: value
        for key, value in stage_b.items()
        if not key.startswith("_")
    }
    return {
        "format": SEALED_TEST_FORMAT,
        "version": OPPORTUNITY_WORKFLOW_VERSION,
        "protocol": _protocol(),
        "corpusLedger": ledger.reference,
        "stageB": {
            "file": stage_b_path.name,
            "sha256": stage_b_sha256,
            "authorization": visible_stage_b["authorization"],
        },
        "frozenPair": visible_stage_b["frozenPair"],
        "testInput": ledger.bound_identity(
            "test",
            file="sealed-test.ndjson",
        ),
        "result": visible_stage_b["result"],
        "sealedTestStatus": "consumed",
    }


class OpportunityWorkflowTests(unittest.TestCase):
    def test_python_requires_and_reauthenticates_typescript_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = _LedgerFixture(root)
            ledger.verification_receipt_path.unlink()
            with self.assertRaisesRegex(
                CapturableDatasetError,
                "TypeScript verification receipt",
            ):
                _load_corpus_ledger(
                    ledger.path,
                    ledger.sha256,
                    ledger.verification_receipt_sha256,
                )

            second_root = root / "second"
            second_root.mkdir()
            ledger = _LedgerFixture(second_root)
            receipt = json.loads(
                ledger.verification_receipt_path.read_text("utf-8")
            )
            receipt["inputSetSha256"] = "0" * 64
            receipt.pop("contentSha256")
            receipt["contentSha256"] = hashlib.sha256(
                _canonical_json(receipt)
            ).hexdigest()
            rewritten_receipt = _canonical_json(receipt)
            ledger.verification_receipt_path.write_bytes(rewritten_receipt)
            with self.assertRaisesRegex(
                CapturableDatasetError,
                "verification receipt SHA-256 is inconsistent",
            ):
                _load_corpus_ledger(
                    ledger.path,
                    ledger.sha256,
                    ledger.verification_receipt_sha256,
                )
            with self.assertRaisesRegex(
                CapturableDatasetError,
                "does not bind the authenticated ledger",
            ):
                _load_corpus_ledger(
                    ledger.path,
                    ledger.sha256,
                    hashlib.sha256(rewritten_receipt).hexdigest(),
                )

    def test_stage_b_aliases_share_one_sealed_corpus_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_root = root / "copy-a"
            second_root = root / "copy-b"
            first_root.mkdir()
            second_root.mkdir()
            first_ledger = _LedgerFixture(first_root)
            second_ledger = _LedgerFixture(second_root)
            self.assertEqual(
                first_ledger.reference["sealedCorpusIdentitySha256"],
                second_ledger.reference["sealedCorpusIdentitySha256"],
            )
            registry = root / "consumption-registry"
            first_path = first_root / "stage-b.json"
            second_path = second_root / "stage-b.json"
            first_payload = _canonical_json({"alias": "one"})
            second_payload = _canonical_json({"alias": "two"})
            first_path.write_bytes(first_payload)
            second_path.write_bytes(second_payload)
            first_sha = hashlib.sha256(first_payload).hexdigest()
            second_sha = hashlib.sha256(second_payload).hexdigest()
            first = _stage_b_fixture(
                first_ledger,
                first_path,
                first_sha,
            )
            second = _stage_b_fixture(
                second_ledger,
                second_path,
                second_sha,
            )
            second["stageA"]["file"] = "stage-a-alias.json"
            second["frozenPair"]["file"] = "pair-two.json"
            second["frozenPair"]["control"]["checkpointSha256"] = "3" * 64
            second["frozenPair"]["treatment"]["checkpointSha256"] = "4" * 64
            marker = consumption_marker_path(
                first_path,
                first_ledger.reference["sealedCorpusIdentitySha256"],
                consumption_registry=registry,
            )
            self.assertEqual(
                marker,
                consumption_marker_path(
                    second_path,
                    second_ledger.reference[
                        "sealedCorpusIdentitySha256"
                    ],
                    consumption_registry=registry,
                ),
            )
            artifact = _sealed_artifact(
                first_ledger,
                first_path,
                first_sha,
                first,
            )
            with (
                patch(
                    "drawback_ml.capturable_opportunity_workflow."
                    "_load_stage_b_context",
                    return_value=(
                        first,
                        first_sha,
                        first_ledger.rows("validation-b"),
                    ),
                ),
                patch(
                    "drawback_ml.capturable_opportunity_workflow."
                    "_build_sealed_result",
                    return_value=artifact,
                ),
            ):
                run_sealed_test(
                    first_path,
                    first_root / "validation-b.ndjson",
                    first_root / "sealed-test.ndjson",
                    first_root / "report.json",
                    corpus_ledger_path=first_ledger.path,
                    consumption_registry=registry,
                    **first_ledger.workflow_auth_kwargs,
                )
            self.assertTrue(marker.exists())
            with (
                patch(
                    "drawback_ml.capturable_opportunity_workflow."
                    "_load_stage_b_context",
                    return_value=(
                        second,
                        second_sha,
                        second_ledger.rows("validation-b"),
                    ),
                ),
                patch(
                    "drawback_ml.capturable_opportunity_workflow."
                    "_build_sealed_result",
                ) as test_access,
            ):
                with self.assertRaises(FileExistsError):
                    run_sealed_test(
                        second_path,
                        second_root / "validation-b.ndjson",
                        second_root / "sealed-test.ndjson",
                        second_root / "report.json",
                        corpus_ledger_path=second_ledger.path,
                        consumption_registry=registry,
                        **second_ledger.workflow_auth_kwargs,
                    )
            test_access.assert_not_called()

    def test_stage_a_authenticates_ledger_and_selects_lower_median(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = _LedgerFixture(root)
            fixtures = _PairFixtures(root, ledger)
            fixtures.add(MODEL_SEEDS[0], 0.03)
            fixtures.add(MODEL_SEEDS[1], 0.01)
            fixtures.add(MODEL_SEEDS[2], 0.02)
            output = root / "stage-a.json"
            comparison_patch, candidate_patch = fixtures.patches()
            with comparison_patch, candidate_patch:
                created = run_stage_a(
                    tuple(reversed(fixtures.paths)),
                    output,
                    corpus_ledger_path=ledger.path,
                    **ledger.workflow_auth_kwargs,
                )
                authenticated, digest = load_stage_a(
                    output,
                    corpus_ledger_path=ledger.path,
                    **ledger.workflow_auth_kwargs,
                )
            self.assertEqual(created["artifactPath"], output.name)
            self.assertEqual(created["artifactSha256"], digest)
            self.assertEqual(
                authenticated["selectedPair"]["modelSeed"],
                MODEL_SEEDS[2],
            )
            self.assertEqual(
                authenticated["corpusLedger"],
                ledger.reference,
            )
            self.assertNotIn(str(root), output.read_text(encoding="utf-8"))

    def test_stage_a_color_gate_rejects_top3_and_nll_regressions(
        self,
    ) -> None:
        cases = (
            {"white_top3_delta": -0.01},
            {"black_nll_delta": 0.01},
        )
        for overrides in cases:
            with self.subTest(overrides=overrides):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    ledger = _LedgerFixture(root)
                    fixtures = _PairFixtures(root, ledger)
                    for seed in MODEL_SEEDS:
                        fixtures.add(seed, 0.01, **overrides)
                    output = root / "stage-a.json"
                    comparison_patch, candidate_patch = fixtures.patches()
                    with comparison_patch, candidate_patch:
                        created = run_stage_a(
                            fixtures.paths,
                            output,
                            corpus_ledger_path=ledger.path,
                            **ledger.workflow_auth_kwargs,
                        )
                        artifact, _ = load_stage_a(
                            output,
                            corpus_ledger_path=ledger.path,
                            **ledger.workflow_auth_kwargs,
                        )
                    self.assertEqual(created["decision"], "retain-control")
                    self.assertFalse(
                        artifact["aggregate"]["checks"][
                            "bothColorsTop1Top3NllNonRegression"
                        ]
                    )
                    self.assertEqual(artifact["nextStage"], "blocked")

    def test_stage_a_rejects_calibration_regression(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = _LedgerFixture(root)
            fixtures = _PairFixtures(root, ledger)
            for seed in MODEL_SEEDS:
                fixtures.add(seed, 0.01, calibration_delta=0.02)
            output = root / "stage-a.json"
            comparison_patch, candidate_patch = fixtures.patches()
            with comparison_patch, candidate_patch:
                created = run_stage_a(
                    fixtures.paths,
                    output,
                    corpus_ledger_path=ledger.path,
                    **ledger.workflow_auth_kwargs,
                )
                artifact, _ = load_stage_a(
                    output,
                    corpus_ledger_path=ledger.path,
                    **ledger.workflow_auth_kwargs,
                )
            self.assertEqual(created["decision"], "retain-control")
            self.assertFalse(
                artifact["aggregate"]["checks"][
                    "calibrationNonRegression"
                ]
            )

    def test_stage_a_rejects_model_seed_seven(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = _LedgerFixture(root)
            fixtures = _PairFixtures(root, ledger)
            fixtures.add(7, 0.01)
            fixtures.add(MODEL_SEEDS[1], 0.01)
            fixtures.add(MODEL_SEEDS[2], 0.01)
            comparison_patch, candidate_patch = fixtures.patches()
            with comparison_patch, candidate_patch:
                with self.assertRaisesRegex(
                    CapturableDatasetError,
                    "model seed",
                ):
                    run_stage_a(
                        fixtures.paths,
                        root / "stage-a.json",
                        corpus_ledger_path=ledger.path,
                        **ledger.workflow_auth_kwargs,
                    )

    def test_stage_a_rejects_dataset_identity_outside_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = _LedgerFixture(root)
            fixtures = _PairFixtures(root, ledger)
            for seed in MODEL_SEEDS:
                fixtures.add(seed, 0.01)
            for comparison, _ in fixtures.comparisons.values():
                comparison["validationInput"]["sha256"] = "0" * 64
            for _, report in fixtures.candidates.values():
                report["inputs"]["validation"]["sha256"] = "0" * 64
            comparison_patch, candidate_patch = fixtures.patches()
            with comparison_patch, candidate_patch:
                with self.assertRaisesRegex(
                    CapturableDatasetError,
                    "authenticated corpus ledger",
                ):
                    run_stage_a(
                        fixtures.paths,
                        root / "stage-a.json",
                        corpus_ledger_path=ledger.path,
                        **ledger.workflow_auth_kwargs,
                    )

    def test_stage_a_rejects_treatment_training_identity_mismatch(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = _LedgerFixture(root)
            fixtures = _PairFixtures(root, ledger)
            for seed in MODEL_SEEDS:
                fixtures.add(seed, 0.01)
            first = fixtures.comparisons[fixtures.paths[0].name][0]
            first["treatments"][0]["trainInput"]["sha256"] = "0" * 64
            first["bestTreatment"]["trainInput"]["sha256"] = "0" * 64
            for (selection_directory, _), (_, report) in (
                fixtures.candidates.items()
            ):
                if selection_directory.startswith("public-exact-"):
                    report["inputs"]["train"]["sha256"] = "0" * 64
                    break
            comparison_patch, candidate_patch = fixtures.patches()
            with comparison_patch, candidate_patch:
                with self.assertRaisesRegex(
                    CapturableDatasetError,
                    "same training input",
                ):
                    run_stage_a(
                        fixtures.paths,
                        root / "stage-a.json",
                        corpus_ledger_path=ledger.path,
                        **ledger.workflow_auth_kwargs,
                    )

    def test_ledger_rejects_wrong_sha_unknown_fields_and_identity(
        self,
    ) -> None:
        mutations = (
            (
                "unknown",
                lambda artifact: artifact["identity"].__setitem__(
                    "unknownField",
                    "opaque",
                ),
                "fields are invalid",
            ),
            (
                "producer",
                lambda artifact: artifact["splits"][0].__setitem__(
                    "producerEngineCommit",
                    "3" * 40,
                ),
                "producer identity",
            ),
            (
                "balance",
                lambda artifact: artifact["splits"][1]["sourceTrace"][
                    "labelCountsByColor"
                ]["white"].__setitem__(CAPTURABLE_RULE_IDS[0], 99),
                "label-balanced",
            ),
            (
                "roots",
                lambda artifact: artifact["splits"][2].__setitem__(
                    "seedRoots",
                    [7, 7, 7],
                ),
                "seedRoots",
            ),
        )
        for name, mutate, message in mutations:
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as directory:
                    ledger = _LedgerFixture(Path(directory), mutate)
                    with self.assertRaisesRegex(
                        CapturableDatasetError,
                        message,
                    ):
                        _load_corpus_ledger(
                            ledger.path,
                            ledger.sha256,
                            ledger.verification_receipt_sha256,
                        )
        with tempfile.TemporaryDirectory() as directory:
            ledger = _LedgerFixture(Path(directory))
            with self.assertRaisesRegex(
                CapturableDatasetError,
                "SHA-256 is inconsistent",
            ):
                _load_corpus_ledger(
                    ledger.path,
                    "0" * 64,
                    ledger.verification_receipt_sha256,
                )

    def test_ledger_rejects_cross_split_a_b_test_overlap(self) -> None:
        def overlap(artifact):
            validation_a = artifact["splits"][1]
            validation_b = artifact["splits"][2]
            test = artifact["splits"][3]
            for target in (validation_b, test):
                target["sourceTrace"]["gameIds"][0] = (
                    validation_a["sourceTrace"]["gameIds"][0]
                )
                target["sourceTrace"]["simulationSeeds"][0] = (
                    validation_a["sourceTrace"]["simulationSeeds"][0]
                )
                target["converted"]["gameIds"][0] = (
                    validation_a["converted"]["gameIds"][0]
                )
                target["converted"]["simulationSeeds"][0] = (
                    validation_a["converted"]["simulationSeeds"][0]
                )

        with tempfile.TemporaryDirectory() as directory:
            ledger = _LedgerFixture(Path(directory), overlap)
            with self.assertRaisesRegex(
                CapturableDatasetError,
                "sets overlap across splits",
            ):
                _load_corpus_ledger(
                    ledger.path,
                    ledger.sha256,
                    ledger.verification_receipt_sha256,
                )

    def test_ledger_accepts_typescript_runtime_string_domain(self) -> None:
        platform = "WIN_32.test-" + ("A9_.-" * 64)
        architecture = "X64_beta." + ("z0-_." * 64)

        def use_typescript_runtime_strings(artifact):
            identities = [
                artifact["identity"]["producerRuntimeIdentity"],
                *(
                    split["producerRuntimeIdentity"]
                    for split in artifact["splits"]
                ),
            ]
            for identity in identities:
                identity["runtime"]["platform"] = platform
                identity["runtime"]["architecture"] = architecture
                _resign_runtime(identity)

        with tempfile.TemporaryDirectory() as directory:
            ledger = _LedgerFixture(
                Path(directory),
                use_typescript_runtime_strings,
            )
            loaded = _load_corpus_ledger(
                ledger.path,
                ledger.sha256,
                ledger.verification_receipt_sha256,
            )

        runtime = loaded.artifact["identity"]["producerRuntimeIdentity"][
            "runtime"
        ]
        self.assertEqual(runtime["platform"], platform)
        self.assertEqual(runtime["architecture"], architecture)

    def test_ledger_rejects_invalid_producer_runtime_identity(self) -> None:
        def valid_split_mismatch(artifact):
            runtime = artifact["splits"][0]["producerRuntimeIdentity"]
            runtime["coordinator"]["sha256"] = "3" * 64
            _resign_runtime(runtime)

        mutations = (
            (
                "missing-global",
                lambda artifact: artifact["identity"].pop(
                    "producerRuntimeIdentity"
                ),
                "fields are invalid",
            ),
            (
                "old-ledger",
                lambda artifact: artifact.__setitem__("version", 2),
                "format or version is invalid",
            ),
            (
                "runtime-extra",
                lambda artifact: artifact["identity"][
                    "producerRuntimeIdentity"
                ]["runtime"].__setitem__("hook", "blocked"),
                "fields are invalid",
            ),
            (
                "runtime-flags",
                lambda artifact: artifact["identity"][
                    "producerRuntimeIdentity"
                ]["runtime"].__setitem__("execArgv", ["--inspect"]),
                "runtime is invalid",
            ),
            (
                "component-id",
                lambda artifact: artifact["identity"][
                    "producerRuntimeIdentity"
                ]["coordinator"].__setitem__(
                    "componentId",
                    "wrong/v1",
                ),
                "componentId is unsupported",
            ),
            (
                "component-count",
                lambda artifact: artifact["identity"][
                    "producerRuntimeIdentity"
                ]["parallelWorker"].__setitem__("files", False),
                "positive integer",
            ),
            (
                "component-count-above-max-safe",
                lambda artifact: artifact["identity"][
                    "producerRuntimeIdentity"
                ]["parallelWorker"].__setitem__(
                    "files",
                    9_007_199_254_740_992,
                ),
                "positive integer",
            ),
            (
                "aggregate",
                lambda artifact: artifact["identity"][
                    "producerRuntimeIdentity"
                ].__setitem__("aggregateSha256", "0" * 64),
                "aggregate is inconsistent",
            ),
            (
                "split-mismatch",
                valid_split_mismatch,
                "differs across splits",
            ),
        )
        for name, mutate, message in mutations:
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as directory:
                    ledger = _LedgerFixture(Path(directory), mutate)
                    with self.assertRaisesRegex(
                        CapturableDatasetError,
                        message,
                    ):
                        _load_corpus_ledger(
                            ledger.path,
                            ledger.sha256,
                            ledger.verification_receipt_sha256,
                        )

    def test_ledger_rejects_old_verification_receipt_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger = _LedgerFixture(Path(directory))
            receipt = json.loads(
                ledger.verification_receipt_path.read_text("utf8")
            )
            receipt["version"] = 1
            receipt.pop("contentSha256")
            receipt["contentSha256"] = hashlib.sha256(
                _canonical_json(receipt)
            ).hexdigest()
            payload = _canonical_json(receipt)
            ledger.verification_receipt_path.write_bytes(payload)
            with self.assertRaisesRegex(
                CapturableDatasetError,
                "does not bind the authenticated ledger",
            ):
                _load_corpus_ledger(
                    ledger.path,
                    ledger.sha256,
                    hashlib.sha256(payload).hexdigest(),
                )

    def test_ledger_rejects_cross_stream_parameter_seed_overlap(self) -> None:
        def overlap(artifact):
            train = artifact["splits"][0]["sourceTrace"]
            test = artifact["splits"][3]["sourceTrace"]
            test["parameterSeeds"][0] = train["simulationSeeds"][0]

        with tempfile.TemporaryDirectory() as directory:
            ledger = _LedgerFixture(Path(directory), overlap)
            with self.assertRaisesRegex(
                CapturableDatasetError,
                "sets overlap across splits or streams",
            ):
                _load_corpus_ledger(
                    ledger.path,
                    ledger.sha256,
                    ledger.verification_receipt_sha256,
                )

    def test_ledger_rejects_numeric_type_confusion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger = _LedgerFixture(Path(directory))
            ledger.rewrite(
                lambda artifact: artifact["opportunityContract"][
                    "shape"
                ].__setitem__(0, 25.0)
            )
            with self.assertRaisesRegex(
                CapturableDatasetError,
                "opportunity contract",
            ):
                _load_corpus_ledger(
                    ledger.path,
                    ledger.sha256,
                    ledger.verification_receipt_sha256,
                )
        with tempfile.TemporaryDirectory() as directory:
            ledger = _LedgerFixture(Path(directory))
            ledger.rewrite(
                lambda artifact: artifact["partition"].__setitem__(
                    "games",
                    float(artifact["partition"]["games"]),
                )
            )
            with self.assertRaisesRegex(
                CapturableDatasetError,
                "positive integer",
            ):
                _load_corpus_ledger(
                    ledger.path,
                    ledger.sha256,
                    ledger.verification_receipt_sha256,
                )

    def test_stage_b_rejects_seed_seven_false_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = _LedgerFixture(root)
            stage_a_path = root / "stage-a.json"
            stage_a_path.touch()
            stage_a = {
                "decision": "promote-treatment",
                "selectedPair": _frozen_pair(),
                "validationAInput": ledger.bound_identity("validation-a"),
                "corpusLedger": ledger.reference,
            }
            pair = _LoadedPair(
                control=(Path("control.pt"), object(), {}, "1" * 64),
                treatment=(Path("treatment.pt"), object(), {}, "2" * 64),
                source_game_ids=frozenset(
                    ledger.split("train")["converted"]["gameIds"]
                ),
            )
            with (
                patch(
                    "drawback_ml.capturable_opportunity_workflow.load_stage_a",
                    return_value=(stage_a, "d" * 64),
                ),
                patch(
                    "drawback_ml.capturable_opportunity_workflow."
                    "_load_frozen_pair",
                    return_value=pair,
                ),
                patch(
                    "drawback_ml.capturable_opportunity_workflow."
                    "_load_stable_capturable_dataset",
                    return_value=(
                        ledger.rows("validation-b", first_seed=7),
                        _SPLIT_SHA["validation-b"],
                    ),
                ),
                patch(
                    "drawback_ml.capturable_opportunity_workflow."
                    "_paired_result"
                ) as evaluation,
            ):
                with self.assertRaisesRegex(
                    CapturableDatasetError,
                    "authenticated corpus ledger",
                ):
                    run_stage_b(
                        stage_a_path,
                        root / "validation-b.ndjson",
                        root / "stage-b.json",
                        corpus_ledger_path=ledger.path,
                        **ledger.workflow_auth_kwargs,
                    )
                evaluation.assert_not_called()

    def test_stage_b_rejects_dataset_sha_outside_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = _LedgerFixture(root)
            stage_a_path = root / "stage-a.json"
            stage_a_path.touch()
            stage_a = {
                "decision": "promote-treatment",
                "selectedPair": _frozen_pair(),
                "validationAInput": ledger.bound_identity("validation-a"),
                "corpusLedger": ledger.reference,
            }
            pair = _LoadedPair(
                control=(Path("control.pt"), object(), {}, "1" * 64),
                treatment=(Path("treatment.pt"), object(), {}, "2" * 64),
                source_game_ids=frozenset(
                    ledger.split("train")["converted"]["gameIds"]
                ),
            )
            with (
                patch(
                    "drawback_ml.capturable_opportunity_workflow.load_stage_a",
                    return_value=(stage_a, "d" * 64),
                ),
                patch(
                    "drawback_ml.capturable_opportunity_workflow."
                    "_load_frozen_pair",
                    return_value=pair,
                ),
                patch(
                    "drawback_ml.capturable_opportunity_workflow."
                    "_load_stable_capturable_dataset",
                    return_value=(
                        ledger.rows("validation-b"),
                        "0" * 64,
                    ),
                ),
            ):
                with self.assertRaisesRegex(
                    CapturableDatasetError,
                    "authenticated corpus ledger",
                ):
                    run_stage_b(
                        stage_a_path,
                        root / "validation-b.ndjson",
                        root / "stage-b.json",
                        corpus_ledger_path=ledger.path,
                        **ledger.workflow_auth_kwargs,
                    )

    def test_stage_b_round_trip_and_blocked_stage_a(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = _LedgerFixture(root)
            stage_a_path = root / "stage-a.json"
            stage_a_path.touch()
            pair = _LoadedPair(
                control=(Path("control.pt"), object(), {}, "1" * 64),
                treatment=(Path("treatment.pt"), object(), {}, "2" * 64),
                source_game_ids=frozenset(
                    ledger.split("train")["converted"]["gameIds"]
                ),
            )
            blocked = {"decision": "retain-control"}
            with (
                patch(
                    "drawback_ml.capturable_opportunity_workflow.load_stage_a",
                    return_value=(blocked, "d" * 64),
                ),
                patch(
                    "drawback_ml.capturable_opportunity_workflow."
                    "_load_stable_capturable_dataset"
                ) as dataset_access,
            ):
                with self.assertRaisesRegex(
                    CapturableDatasetError,
                    "did not authorize validation B",
                ):
                    run_stage_b(
                        stage_a_path,
                        root / "validation-b.ndjson",
                        root / "blocked-stage-b.json",
                        corpus_ledger_path=ledger.path,
                        **ledger.workflow_auth_kwargs,
                    )
                dataset_access.assert_not_called()

            stage_a = {
                "decision": "promote-treatment",
                "selectedPair": _frozen_pair(),
                "validationAInput": ledger.bound_identity("validation-a"),
                "corpusLedger": ledger.reference,
            }
            output = root / "stage-b.json"
            with (
                patch(
                    "drawback_ml.capturable_opportunity_workflow.load_stage_a",
                    return_value=(stage_a, "d" * 64),
                ),
                patch(
                    "drawback_ml.capturable_opportunity_workflow."
                    "_load_frozen_pair",
                    return_value=pair,
                ),
                patch(
                    "drawback_ml.capturable_opportunity_workflow."
                    "_load_stable_capturable_dataset",
                    return_value=(
                        ledger.rows("validation-b"),
                        _SPLIT_SHA["validation-b"],
                    ),
                ),
                patch(
                    "drawback_ml.capturable_opportunity_workflow."
                    "_paired_result",
                    return_value=_paired_result(),
                ),
            ):
                created = run_stage_b(
                    stage_a_path,
                    root / "validation-b.ndjson",
                    output,
                    corpus_ledger_path=ledger.path,
                    **ledger.workflow_auth_kwargs,
                )
                artifact, digest = load_stage_b(
                    output,
                    root / "validation-b.ndjson",
                    corpus_ledger_path=ledger.path,
                    **ledger.workflow_auth_kwargs,
                )
            self.assertEqual(created["artifactPath"], output.name)
            self.assertEqual(created["artifactSha256"], digest)
            self.assertEqual(
                artifact["authorization"],
                "sealed-test-authorized",
            )

    def test_recursive_reference_paths_are_rejected_on_create_and_load(
        self,
    ) -> None:
        path_values = (
            r"C:\Users\private\selection.json",
            "/home/private/selection.json",
            "relative/sub/selection.json",
            r"relative\sub\selection.json",
            "selection.json:secret",
            "NUL",
            "selection.json.",
        )
        for path_value in path_values:
            with self.subTest(path_value=path_value):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    ledger = _LedgerFixture(root)
                    fixtures = _PairFixtures(root, ledger)
                    for seed in MODEL_SEEDS:
                        fixtures.add(seed, 0.01)
                    first = fixtures.comparisons[
                        fixtures.paths[0].name
                    ][0]
                    first["control"]["trainInput"]["path"] = path_value
                    comparison_patch, candidate_patch = fixtures.patches()
                    with comparison_patch, candidate_patch:
                        with self.assertRaisesRegex(
                            CapturableDatasetError,
                            "path-free basename|absolute path",
                        ):
                            run_stage_a(
                                fixtures.paths,
                                root / "stage-a.json",
                                corpus_ledger_path=ledger.path,
                                **ledger.workflow_auth_kwargs,
                            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = _LedgerFixture(root)
            fixtures = _PairFixtures(root, ledger)
            for seed in MODEL_SEEDS:
                fixtures.add(seed, 0.01)
            output = root / "stage-a.json"
            comparison_patch, candidate_patch = fixtures.patches()
            with comparison_patch, candidate_patch:
                run_stage_a(
                    fixtures.paths,
                    output,
                    corpus_ledger_path=ledger.path,
                    **ledger.workflow_auth_kwargs,
                )
                original = load_stage_a(
                    output,
                    corpus_ledger_path=ledger.path,
                    **ledger.workflow_auth_kwargs,
                )[0]
                for key, value in (
                    ("checkpointFile", "relative/checkpoint.pt"),
                    ("debugReport", "relative/private/report.json"),
                    ("debugMarker", "relative/private/marker.json"),
                ):
                    with self.subTest(key=key):
                        artifact = deepcopy(original)
                        artifact["selectedPair"]["control"][key] = value
                        output.write_bytes(_canonical_json(artifact))
                        with self.assertRaisesRegex(
                            CapturableDatasetError,
                            "path-free basename",
                        ):
                            load_stage_a(
                                output,
                                corpus_ledger_path=ledger.path,
                                **ledger.workflow_auth_kwargs,
                            )

    def test_sealed_test_does_not_touch_input_when_stage_b_blocks(
        self,
    ) -> None:
        class UntouchablePath:
            def __getattribute__(self, name: str):
                if name in {"exists", "open", "resolve", "stat"}:
                    raise AssertionError(
                        f"test input was touched via {name}"
                    )
                return object.__getattribute__(self, name)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = _LedgerFixture(root)
            with patch(
                "drawback_ml.capturable_opportunity_workflow."
                "_load_stage_b_context",
                return_value=({"authorization": "blocked"}, "f" * 64, ()),
            ):
                with self.assertRaisesRegex(
                    CapturableDatasetError,
                    "did not authorize",
                ):
                    run_sealed_test(
                        root / "stage-b.json",
                        root / "validation-b.ndjson",
                        UntouchablePath(),  # type: ignore[arg-type]
                        root / "sealed.json",
                        corpus_ledger_path=ledger.path,
                        consumption_registry=root / "consumption-registry",
                        **ledger.workflow_auth_kwargs,
                    )

    def test_one_canonical_marker_rejects_report_one_report_two(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = _LedgerFixture(root)
            stage_b_path = root / "stage-b.json"
            stage_b_payload = _canonical_json({"fixture": "stage-b"})
            stage_b_path.write_bytes(stage_b_payload)
            stage_b_sha = hashlib.sha256(stage_b_payload).hexdigest()
            stage_b = _stage_b_fixture(
                ledger,
                stage_b_path,
                stage_b_sha,
            )
            artifact = _sealed_artifact(
                ledger,
                stage_b_path,
                stage_b_sha,
                stage_b,
            )
            report_one = root / "report-one.json"
            report_two = root / "report-two.json"
            marker = consumption_marker_path(
                stage_b_path,
                ledger.reference["sealedCorpusIdentitySha256"],
                consumption_registry=root / "consumption-registry",
            )
            with (
                patch(
                    "drawback_ml.capturable_opportunity_workflow."
                    "_load_stage_b_context",
                    return_value=(
                        stage_b,
                        stage_b_sha,
                        ledger.rows("validation-b"),
                    ),
                ) as authorization,
                patch(
                    "drawback_ml.capturable_opportunity_workflow."
                    "_build_sealed_result",
                    return_value=artifact,
                ) as test_access,
            ):
                created = run_sealed_test(
                    stage_b_path,
                    root / "validation-b.ndjson",
                    root / "sealed-test.ndjson",
                    report_one,
                    corpus_ledger_path=ledger.path,
                    consumption_registry=root / "consumption-registry",
                    **ledger.workflow_auth_kwargs,
                )
                self.assertEqual(
                    created["consumptionMarker"],
                    marker.name,
                )
                self.assertTrue(marker.exists())
                with self.assertRaises(FileExistsError):
                    run_sealed_test(
                        stage_b_path,
                        root / "validation-b.ndjson",
                        root / "sealed-test.ndjson",
                        report_two,
                        corpus_ledger_path=ledger.path,
                        consumption_registry=root / "consumption-registry",
                        **ledger.workflow_auth_kwargs,
                    )
                self.assertEqual(authorization.call_count, 2)
                self.assertEqual(test_access.call_count, 1)
                self.assertFalse(report_two.exists())

    def test_failure_then_renamed_retry_is_irreversibly_consumed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = _LedgerFixture(root)
            stage_b_path = root / "stage-b.json"
            payload = _canonical_json({"fixture": "stage-b"})
            stage_b_path.write_bytes(payload)
            stage_b_sha = hashlib.sha256(payload).hexdigest()
            stage_b = _stage_b_fixture(
                ledger,
                stage_b_path,
                stage_b_sha,
            )
            marker = consumption_marker_path(
                stage_b_path,
                ledger.reference["sealedCorpusIdentitySha256"],
                consumption_registry=root / "consumption-registry",
            )
            with (
                patch(
                    "drawback_ml.capturable_opportunity_workflow."
                    "_load_stage_b_context",
                    return_value=(
                        stage_b,
                        stage_b_sha,
                        ledger.rows("validation-b"),
                    ),
                ),
                patch(
                    "drawback_ml.capturable_opportunity_workflow."
                    "_build_sealed_result",
                    side_effect=RuntimeError("inference failed"),
                ) as test_access,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "inference failed",
                ):
                    run_sealed_test(
                        stage_b_path,
                        root / "validation-b.ndjson",
                        root / "sealed-test.ndjson",
                        root / "report-one.json",
                        corpus_ledger_path=ledger.path,
                        consumption_registry=root / "consumption-registry",
                        **ledger.workflow_auth_kwargs,
                    )
                self.assertTrue(marker.exists())
                with self.assertRaises(FileExistsError):
                    run_sealed_test(
                        stage_b_path,
                        root / "validation-b.ndjson",
                        root / "sealed-test.ndjson",
                        root / "report-two.json",
                        corpus_ledger_path=ledger.path,
                        consumption_registry=root / "consumption-registry",
                        **ledger.workflow_auth_kwargs,
                    )
                self.assertEqual(test_access.call_count, 1)

    def test_marker_publication_failure_prevents_test_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = _LedgerFixture(root)
            stage_b_path = root / "stage-b.json"
            payload = _canonical_json({"fixture": "stage-b"})
            stage_b_path.write_bytes(payload)
            stage_b_sha = hashlib.sha256(payload).hexdigest()
            stage_b = _stage_b_fixture(
                ledger,
                stage_b_path,
                stage_b_sha,
            )
            with (
                patch(
                    "drawback_ml.capturable_opportunity_workflow."
                    "_load_stage_b_context",
                    return_value=(
                        stage_b,
                        stage_b_sha,
                        ledger.rows("validation-b"),
                    ),
                ),
                patch(
                    "drawback_ml.capturable_opportunity_workflow."
                    "publish_bytes_durable",
                    side_effect=OSError("publication failed"),
                ),
                patch(
                    "drawback_ml.capturable_opportunity_workflow."
                    "_build_sealed_result"
                ) as test_access,
            ):
                with self.assertRaisesRegex(
                    OSError,
                    "publication failed",
                ):
                    run_sealed_test(
                        stage_b_path,
                        root / "validation-b.ndjson",
                        root / "sealed-test.ndjson",
                        root / "sealed.json",
                        corpus_ledger_path=ledger.path,
                        consumption_registry=root / "consumption-registry",
                        **ledger.workflow_auth_kwargs,
                    )
                test_access.assert_not_called()

    def test_load_sealed_requires_authorized_stage_b(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = _LedgerFixture(root)
            stage_b_path = root / "stage-b.json"
            stage_b_path.touch()
            stage_b_sha = "f" * 64
            stage_b = _stage_b_fixture(
                ledger,
                stage_b_path,
                stage_b_sha,
                authorization="blocked",
            )
            artifact = _sealed_artifact(
                ledger,
                stage_b_path,
                stage_b_sha,
                stage_b,
            )
            artifact["consumption"] = {
                "file": consumption_marker_path(
                    stage_b_path,
                    ledger.reference["sealedCorpusIdentitySha256"],
                    consumption_registry=root / "consumption-registry",
                ).name,
                "sha256": "a" * 64,
            }
            report = root / "sealed.json"
            report.write_bytes(_canonical_json(artifact))
            report_sha = hashlib.sha256(report.read_bytes()).hexdigest()
            with patch(
                "drawback_ml.capturable_opportunity_workflow.load_stage_b",
                return_value=(stage_b, stage_b_sha),
            ):
                with self.assertRaisesRegex(
                    CapturableDatasetError,
                    "did not authorize",
                ):
                    load_sealed_test(
                        report,
                        stage_b_path,
                        root / "validation-b.ndjson",
                        report_sha256=report_sha,
                        corpus_ledger_path=ledger.path,
                        consumption_registry=root / "consumption-registry",
                        **ledger.workflow_auth_kwargs,
                    )

    def test_caller_report_sha_rejects_internally_rewritten_metrics(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = _LedgerFixture(root)
            stage_b_path = root / "stage-b.json"
            stage_b_payload = _canonical_json({"fixture": "stage-b"})
            stage_b_path.write_bytes(stage_b_payload)
            stage_b_sha = hashlib.sha256(stage_b_payload).hexdigest()
            stage_b = _stage_b_fixture(
                ledger,
                stage_b_path,
                stage_b_sha,
            )
            artifact = _sealed_artifact(
                ledger,
                stage_b_path,
                stage_b_sha,
                stage_b,
            )
            report = root / "sealed.json"
            with (
                patch(
                    "drawback_ml.capturable_opportunity_workflow."
                    "_load_stage_b_context",
                    return_value=(
                        stage_b,
                        stage_b_sha,
                        ledger.rows("validation-b"),
                    ),
                ),
                patch(
                    "drawback_ml.capturable_opportunity_workflow."
                    "_build_sealed_result",
                    return_value=artifact,
                ),
            ):
                created = run_sealed_test(
                    stage_b_path,
                    root / "validation-b.ndjson",
                    root / "sealed-test.ndjson",
                    report,
                    corpus_ledger_path=ledger.path,
                    consumption_registry=root / "consumption-registry",
                    **ledger.workflow_auth_kwargs,
                )
            with patch(
                "drawback_ml.capturable_opportunity_workflow.load_stage_b",
                return_value=(stage_b, stage_b_sha),
            ):
                authenticated, digest = load_sealed_test(
                    report,
                    stage_b_path,
                    root / "validation-b.ndjson",
                    report_sha256=created["artifactSha256"],
                    corpus_ledger_path=ledger.path,
                    consumption_registry=root / "consumption-registry",
                    **ledger.workflow_auth_kwargs,
                )
                self.assertEqual(digest, created["artifactSha256"])
                forged = deepcopy(authenticated)
                forged["result"] = _paired_result(promote=False)
                report.write_bytes(_canonical_json(forged))
                with self.assertRaisesRegex(
                    CapturableDatasetError,
                    "caller-authenticated final report SHA-256",
                ):
                    load_sealed_test(
                        report,
                        stage_b_path,
                        root / "validation-b.ndjson",
                        report_sha256=created["artifactSha256"],
                        corpus_ledger_path=ledger.path,
                        consumption_registry=root / "consumption-registry",
                        **ledger.workflow_auth_kwargs,
                    )

    def test_load_sealed_recursively_rejects_all_path_forms(self) -> None:
        values = (
            r"C:\private\model.pt",
            "/private/model.pt",
            "relative/sub/model.pt",
            r"relative\sub\model.pt",
        )
        for value in values:
            with self.subTest(value=value):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    ledger = _LedgerFixture(root)
                    stage_b_path = root / "stage-b.json"
                    stage_b_path.touch()
                    stage_b_sha = "f" * 64
                    stage_b = _stage_b_fixture(
                        ledger,
                        stage_b_path,
                        stage_b_sha,
                    )
                    artifact = _sealed_artifact(
                        ledger,
                        stage_b_path,
                        stage_b_sha,
                        stage_b,
                    )
                    artifact["result"]["control"]["checkpoint"][
                        "file"
                    ] = value
                    report = root / "sealed.json"
                    report.write_bytes(_canonical_json(artifact))
                    digest = hashlib.sha256(report.read_bytes()).hexdigest()
                    with self.assertRaisesRegex(
                        CapturableDatasetError,
                        "path-free basename|absolute path",
                    ):
                        load_sealed_test(
                            report,
                            stage_b_path,
                            root / "validation-b.ndjson",
                            report_sha256=digest,
                            corpus_ledger_path=ledger.path,
                            consumption_registry=(
                                root / "consumption-registry"
                            ),
                            **ledger.workflow_auth_kwargs,
                        )

    def test_cli_json_refuses_paths_and_emits_basename(self) -> None:
        output = io.StringIO()
        with (
            patch(
                "drawback_ml.capturable_opportunity_workflow.run_stage_a",
                return_value={
                    "artifactPath": "stage-a.json",
                    "artifactSha256": "a" * 64,
                    "decision": "retain-control",
                    "selectedModelSeed": None,
                },
            ) as run_stage_a_mock,
            redirect_stdout(output),
        ):
            result = main(
                [
                    "stage-a",
                    "--comparison",
                    "pair-1.json",
                    "--corpus-ledger",
                    "ledger.json",
                    "--corpus-ledger-sha256",
                    "a" * 64,
                    "--corpus-ledger-verification-receipt-sha256",
                    "b" * 64,
                    "--output",
                    "stage-a.json",
                ]
            )
        self.assertEqual(result, 0)
        self.assertIn('"artifactPath":"stage-a.json"', output.getvalue())
        self.assertEqual(
            run_stage_a_mock.call_args.kwargs[
                "corpus_ledger_verification_receipt_sha256"
            ],
            "b" * 64,
        )

        with patch(
            "drawback_ml.capturable_opportunity_workflow.run_stage_a",
            return_value={
                "artifactPath": r"C:\private\stage-a.json",
                "artifactSha256": "a" * 64,
                "decision": "retain-control",
                "selectedModelSeed": None,
            },
        ):
            with self.assertRaisesRegex(
                CapturableDatasetError,
                "path-free basename|absolute path",
            ):
                main(
                    [
                        "stage-a",
                        "--comparison",
                        "pair-1.json",
                        "--corpus-ledger",
                        "ledger.json",
                        "--corpus-ledger-sha256",
                        "a" * 64,
                        "--corpus-ledger-verification-receipt-sha256",
                        "b" * 64,
                        "--output",
                        "stage-a.json",
                    ]
                )


if __name__ == "__main__":
    unittest.main()
