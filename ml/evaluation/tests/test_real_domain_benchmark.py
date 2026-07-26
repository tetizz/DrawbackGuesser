from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

import chess

from ml.evaluation.real_domain_benchmark import (
    ANALYSIS_FORMAT,
    CORPUS_FORMAT,
    LABEL_FORMAT,
    ApprovedSubprocessAnalyzer,
    BenchmarkClaimConfig,
    ContentAddressedJson,
    RealDomainBenchmarkError,
    _build_release_claim_gate,
    _metrics,
    create_real_domain_prediction_bundle,
    load_corpus,
    publish_real_domain_benchmark_report,
)
from ml.evaluation.metrics import PredictionExample
from ml.training.drawback_ml.training_corpus_set import (
    AGENT_DOMAIN,
    CorpusIdentity,
    SupplementIdentity,
    create_training_corpus_set,
)


ROOT = Path(__file__).resolve().parents[3]
CATALOG = ROOT / "engine" / "data" / "catalog" / "observed-drawbacks.json"
CATALOG_SHA = hashlib.sha256(CATALOG.read_bytes()).hexdigest()
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def write_json(directory: Path, name: str, value: object) -> ContentAddressedJson:
    path = directory / name
    payload = canonical(value)
    path.write_bytes(payload)
    return ContentAddressedJson(path, hashlib.sha256(payload).hexdigest())


PGN = """[Event "Consented completed game"]
[Site "offline"]
[White "private-player-one"]
[Black "private-player-two"]
[Result "1-0"]

1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7# 1-0
"""


class FakeAnalyzer:
    def __init__(self, supported: tuple[str, ...], unsupported: tuple[str, ...]):
        self.supported = supported
        self.unsupported = unsupported
        self.calls: list[tuple[str, str]] = []

    def analyze_completed_game(
        self, *, pgn_sha256: str, pgn: str
    ) -> dict[str, object]:
        self.calls.append((pgn_sha256, pgn))
        probability = 1.0 / len(self.supported)
        posterior = {drawback_id: probability for drawback_id in self.supported}
        return {
            "format": ANALYSIS_FORMAT,
            "version": 1,
            "pgnSha256": pgn_sha256,
            "completed": True,
            "plyCount": 7,
            "classIds": list(self.supported),
            "unavailableSupportedIds": list(self.unsupported),
            "snapshots": [
                {"ply": 1, "white": posterior, "black": posterior},
                {"ply": 5, "white": posterior, "black": posterior},
                {"ply": 7, "white": posterior, "black": posterior},
            ],
            "predictor": {
                "mode": "hybrid-v21-ensemble",
                "browserArtifactSha256": SHA_A,
                "ensembleReleaseSha256": SHA_B,
                "calibrationSha256": SHA_C,
                "approvalEvidenceSha256": SHA_D,
            },
        }


class RealDomainBenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
        self.entries = catalog["entries"]
        self.supported = tuple(
            entry["id"]
            for entry in self.entries
            if entry["implementationStatus"] != "unsupported"
        )
        self.unsupported = tuple(
            entry["id"]
            for entry in self.entries
            if entry["implementationStatus"] == "unsupported"
        )
        self.unavailable = ("hand-and-gigabrain", "ichtyophobe")
        self.analyzable = tuple(
            drawback_id
            for drawback_id in self.supported
            if drawback_id not in self.unavailable
        )
        self.title_by_id = {
            entry["id"]: entry["observedName"] for entry in self.entries
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def runtime_dependencies(self) -> tuple[ContentAddressedJson, ...]:
        paths = (
            Path(sys.base_prefix)
            / f"python{sys.version_info.major}{sys.version_info.minor}.dll",
            Path(sys.executable).parent.parent / "pyvenv.cfg",
        )
        return tuple(
            ContentAddressedJson(
                path,
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            for path in paths
            if path.is_file()
        )

    def inputs(
        self,
        *,
        pgn: str = PGN,
        white_id: str = "vegan",
        black_id: str = "checkers",
        primary_training_moves: tuple[str, ...] = ("e2e4",),
    ) -> tuple[
        ContentAddressedJson,
        ContentAddressedJson,
        ContentAddressedJson,
        ContentAddressedJson,
        ContentAddressedJson,
    ]:
        digest = hashlib.sha256(pgn.encode()).hexdigest()
        corpus = write_json(
            self.directory,
            "corpus.json",
            {
                "format": CORPUS_FORMAT,
                "version": 1,
                "consent": {
                    "basis": "explicit",
                    "completedOnly": True,
                    "liveCollection": False,
                },
                "games": [
                    {
                        "pgn": pgn,
                        "pgnSha256": digest,
                        "completed": True,
                        "result": "1-0",
                        "simulationSeed": None,
                    }
                ],
            },
        )
        labels = write_json(
            self.directory,
            "labels.json",
            {
                "format": LABEL_FORMAT,
                "version": 1,
                "revealTiming": "after-game-completion",
                "labels": [
                    {
                        "pgnSha256": digest,
                        "white": {
                            "title": self.title_by_id[white_id],
                            "parameters": {},
                        },
                        "black": {
                            "title": self.title_by_id[black_id],
                            "parameters": {},
                        },
                    }
                ],
            },
        )
        corpus_run_id = "9" * 64
        self.training_datasets = []
        for index in range(7):
            path = self.directory / f"training-{index}.ndjson"
            board = chess.Board()
            rows = []
            moves = (
                primary_training_moves if index == 0 else ("e2e4",)
            )
            for ply, move_code in enumerate(moves):
                rows.append(
                    {
                        "fenBefore": board.fen(en_passant="legal"),
                        "gameId": f"training-{index}",
                        "move": move_code,
                        "ply": ply,
                        "seed": 20260811 + index,
                    }
                )
                board.push_uci(move_code)
            payload = b"".join(
                (
                    json.dumps(row, sort_keys=True, separators=(",", ":"))
                    + "\n"
                ).encode()
                for row in rows
            )
            path.write_bytes(payload)
            self.training_datasets.append(
                ContentAddressedJson(path, hashlib.sha256(payload).hexdigest())
            )
        private_training = write_json(
            self.directory,
            "private-training.json",
            {
                "corpusRunId": corpus_run_id,
                "dataset": {
                    "bytes": self.training_datasets[0].path.stat().st_size,
                    "file": "train.ndjson",
                    "sha256": self.training_datasets[0].sha256,
                },
                "manifestVersion": 1,
                "split": "train",
            },
        )
        public_training = write_json(
            self.directory,
            "public-training.json",
            {
                "corpus": {},
                "corpusRunId": corpus_run_id,
                "releaseManifestVersion": 1,
                "splits": {
                    "train": {
                        "privateManifestSha256": private_training.sha256,
                    }
                },
            },
        )
        def training_content(index: int) -> dict[str, object]:
            return {
                "outcomes_sha256": f"{index * 10 + 4:064x}",
                "dataset_sha256": self.training_datasets[index - 1].sha256,
                "dataset_bytes": self.training_datasets[
                    index - 1
                ].path.stat().st_size,
                "games": 1,
                "rows": (
                    len(primary_training_moves) if index == 1 else 1
                ),
                "schema_version": 6,
                "symbolic_feature_version": 6,
                "max_plies": 80,
                "observation_policy": "single-attempt-allow-partial-v1",
                "evaluator_policy_id": "stockfish-bestmove-v1",
                "evaluator_policy_version": 1,
                "evaluator_nodes": 10_000,
                "engine_binary_sha256": "e" * 64,
                "engine_fingerprint": "stockfish:18:test",
                "agent_domain": AGENT_DOMAIN,
            }

        profiles = (
            (101, "checkers-pacman", ("checkers", "pacman")),
            (102, "truant-spice-of-life", ("truant", "spice-of-life")),
            (103, "oddball-even-keeled", ("oddball", "even-keeled")),
            (104, "quit-horsing-around-forward-march", ("quit-horsing-around", "forward-march")),
            (105, "horse-tranquilizer-conscientious-objectors", ("horse-tranquilizer", "conscientious-objectors")),
            (106, "gambler-truant", ("gambler", "truant")),
        )
        profile_documents = [
            {
                "description": f"fixture {profile_id}",
                "evidence": "fixture",
                "id": profile_id,
                "ruleIds": list(rule_ids),
            }
            for _offset, profile_id, rule_ids in profiles
        ]
        source_revisions = [f"{800 + index:040x}" for index in range(6)]
        run_ids = [f"{index + 20:064x}" for index in range(6)]
        config_hashes = [f"{index + 60:064x}" for index in range(6)]
        self.supplement_plans = [
            write_json(
                self.directory,
                f"hn-plan-{index}.json",
                {
                    "metadata": {
                        "hardNegativeProfile": profile_documents[index],
                    },
                    "runPlan": {
                        "corpusConfigSha256": config_hashes[index],
                        "ruleIds": list(profiles[index][2]),
                        "runId": run_ids[index],
                    },
                    "schedule": {
                        "policyId": "balanced-symmetric-v1",
                        "splits": {"test": [], "train": [], "validation": []},
                    },
                    "schemaVersion": 1,
                    "sourceRevision": source_revisions[index],
                },
            )
            for index in range(6)
        ]
        self.supplement_manifests = [
            write_json(
                self.directory,
                f"hn-manifest-{index}.json",
                {
                    "agentIds": list(AGENT_DOMAIN),
                    "engineBinarySha256": "e" * 64,
                    "engineFingerprint": "stockfish:18:test",
                    "evaluatorPolicyId": "stockfish-bestmove-v1",
                    "evaluatorPolicyVersion": 1,
                    "hardNegativeGeneration": {
                        "corpusConfigSha256": config_hashes[index],
                        "planSha256": self.supplement_plans[index].sha256,
                        "runId": run_ids[index],
                        "sourceRevision": source_revisions[index],
                    },
                    "hardNegativeProfile": profile_documents[index],
                    "maxPlies": 80,
                    "observationPolicy": "single-attempt-allow-partial-v1",
                    "schemaVersion": 6,
                    "splits": {
                        "train": {
                            "bytes": self.training_datasets[index + 1].path.stat().st_size,
                            "games": 1,
                            "rows": 1,
                            "sha256": self.training_datasets[index + 1].sha256,
                        }
                    },
                    "symbolicFeatureVersion": 6,
                },
            )
            for index in range(6)
        ]
        primary = CorpusIdentity(
            release_root_sha256=public_training.sha256,
            corpus_run_id=corpus_run_id,
            private_train_manifest_sha256=private_training.sha256,
            **training_content(1),
        )
        supplements = [
            SupplementIdentity(
                source_revision=source_revisions[index],
                generation_run_id=run_ids[index],
                manifest_sha256=self.supplement_manifests[index].sha256,
                plan_sha256=self.supplement_plans[index].sha256,
                **training_content(index + 2),
                profile_offset=offset,
                profile_id=profile_id,
                rule_ids=rule_ids,
                profile_sha256=hashlib.sha256(
                    json.dumps(
                        profile_documents[index],
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest(),
            )
            for index, (offset, profile_id, rule_ids) in enumerate(profiles)
        ]
        candidate_value = create_training_corpus_set(primary, supplements)
        self.candidate_training_corpus_set = write_json(
            self.directory,
            "candidate-training-corpus-set.json",
            candidate_value,
        )
        unused = corpus
        return (
            corpus,
            labels,
            unused,
            public_training,
            private_training,
        )

    def execute(
        self,
        corpus: ContentAddressedJson,
        labels: ContentAddressedJson,
        unused: ContentAddressedJson,
        public_training: ContentAddressedJson,
        private_training: ContentAddressedJson,
        analyzer: FakeAnalyzer | None = None,
        claim_config: BenchmarkClaimConfig = BenchmarkClaimConfig(),
    ) -> tuple[dict[str, object], FakeAnalyzer]:
        selected = analyzer or FakeAnalyzer(self.analyzable, self.unavailable)
        bundle_path = self.directory / "predictions.json"
        bundle = create_real_domain_prediction_bundle(
            corpus=corpus,
            public_training_release=public_training,
            private_training_manifest=private_training,
            candidate_training_corpus_set=(
                self.candidate_training_corpus_set
            ),
            training_datasets=self.training_datasets,
            supplement_manifests=self.supplement_manifests,
            supplement_plans=self.supplement_plans,
            observed_catalog=ContentAddressedJson(CATALOG, CATALOG_SHA),
            analyzer=selected,
            output=bundle_path,
        )
        bundle_reference = ContentAddressedJson(
            bundle_path,
            hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
        )
        report = publish_real_domain_benchmark_report(
            prediction_bundle=bundle_reference,
            revealed_labels=labels,
            observed_catalog=ContentAddressedJson(CATALOG, CATALOG_SHA),
            output=self.directory / "report.json",
            claim_config=claim_config,
        )
        return dict(report), selected

    def test_publishes_aggregate_metrics_without_pgn_or_player_ids(self) -> None:
        corpus, labels, unused, public, private = self.inputs()
        report, analyzer = self.execute(
            corpus, labels, unused, public, private
        )
        rendered = (self.directory / "report.json").read_text(encoding="utf-8")

        self.assertEqual(report["catalogCoverage"]["observedTitleCount"], 194)
        self.assertEqual(report["catalogCoverage"]["supportedTitleCount"], 182)
        self.assertEqual(report["catalogCoverage"]["unsupportedTitleCount"], 12)
        self.assertEqual(report["labelCoverage"]["supportedPlayerLabels"], 2)
        self.assertEqual(
            report["metrics"]["combined"]["final"]["count"],
            2,
        )
        for color in ("white", "black"):
            self.assertEqual(
                report["metrics"]["byColor"][color]["final"]["count"],
                1,
            )
            for horizon in ("5", "10", "15", "20"):
                measured = report["metrics"]["byColor"][color][
                    "moveHorizons"
                ][horizon]
                self.assertEqual(measured["count"], 1)
                self.assertIn("top1Accuracy", measured)
                self.assertIn("top3Accuracy", measured)
                self.assertIn("top5Accuracy", measured)
                self.assertIn("negativeLogLikelihood", measured)
                self.assertIn("brierScore", measured)
                self.assertIn("expectedCalibrationError", measured)
        self.assertNotIn(PGN, rendered)
        self.assertNotIn("private-player-one", rendered)
        self.assertNotIn("private-player-two", rendered)
        self.assertNotIn(hashlib.sha256(PGN.encode()).hexdigest(), rendered)
        self.assertEqual(len(analyzer.calls), 1)
        self.assertNotIn("private-player-one", analyzer.calls[0][1])
        self.assertIn('[Result "1-0"]', analyzer.calls[0][1])
        original_join_digest = hashlib.sha256(PGN.encode()).hexdigest()
        sanitized_digest = hashlib.sha256(
            analyzer.calls[0][1].encode()
        ).hexdigest()
        self.assertNotEqual(sanitized_digest, original_join_digest)
        self.assertEqual(analyzer.calls[0][0], sanitized_digest)
        self.assertEqual(report["releaseClaimGate"]["mode"], "research")
        self.assertFalse(
            report["releaseClaimGate"]["releaseClaimPassing"]
        )

    def test_release_claim_rejects_one_game_report(self) -> None:
        corpus, labels, unused, public, private = self.inputs()

        with self.assertRaisesRegex(
            RealDomainBenchmarkError,
            "minimum-completed-games",
        ):
            self.execute(
                corpus,
                labels,
                unused,
                public,
                private,
                claim_config=BenchmarkClaimConfig(
                    mode="release-claim",
                    claimed_drawback_ids=("vegan",),
                ),
            )

    def test_release_claim_gate_accepts_exact_boundaries(self) -> None:
        gate = _build_release_claim_gate(
            config=BenchmarkClaimConfig(
                mode="release-claim",
                claimed_drawback_ids=("vegan", "checkers"),
            ),
            game_count=2_000,
            player_game_support={"vegan": 10, "checkers": 10},
            analyzable_ids=frozenset({"vegan", "checkers"}),
            zero_truth_probability_count=0,
        )

        self.assertTrue(gate["requirementsSatisfied"])
        self.assertTrue(gate["releaseClaimPassing"])
        self.assertEqual(gate["failures"], [])

    def test_release_claim_gate_rejects_game_and_rule_underflow(self) -> None:
        game_gate = _build_release_claim_gate(
            config=BenchmarkClaimConfig(
                mode="release-claim",
                claimed_drawback_ids=("vegan",),
            ),
            game_count=1_999,
            player_game_support={"vegan": 10},
            analyzable_ids=frozenset({"vegan"}),
            zero_truth_probability_count=0,
        )
        rule_gate = _build_release_claim_gate(
            config=BenchmarkClaimConfig(
                mode="release-claim",
                claimed_drawback_ids=("vegan", "checkers"),
            ),
            game_count=2_000,
            player_game_support={"vegan": 10, "checkers": 9},
            analyzable_ids=frozenset({"vegan", "checkers"}),
            zero_truth_probability_count=0,
        )

        self.assertIn("minimum-completed-games", game_gate["failures"])
        self.assertEqual(
            rule_gate["undercoveredClaimedRules"],
            {"checkers": 9},
        )
        self.assertIn(
            "minimum-player-games-per-claimed-rule",
            rule_gate["failures"],
        )

    def test_zero_truth_probability_is_explicit_and_fails_claim_gate(self) -> None:
        metrics = _metrics(
            [
                PredictionExample(
                    game_id="game",
                    move_number=1,
                    player_color="white",
                    true_drawback="vegan",
                    probabilities={"vegan": 0.0, "checkers": 1.0},
                )
            ]
        )
        assert metrics is not None
        gate = _build_release_claim_gate(
            config=BenchmarkClaimConfig(
                mode="release-claim",
                claimed_drawback_ids=("vegan",),
            ),
            game_count=2_000,
            player_game_support={"vegan": 10},
            analyzable_ids=frozenset({"vegan"}),
            zero_truth_probability_count=int(
                metrics["zeroTruthProbabilityCount"]
            ),
        )

        self.assertIsNone(metrics["negativeLogLikelihood"])
        self.assertFalse(metrics["negativeLogLikelihoodFinite"])
        self.assertEqual(metrics["zeroTruthProbabilityCount"], 1)
        self.assertIn(
            "nonfinite-truth-negative-log-likelihood",
            gate["failures"],
        )

    def test_reports_explicit_unsupported_and_unavailable_coverage(self) -> None:
        corpus, labels, unused, public, private = self.inputs(
            white_id=self.unsupported[0],
            black_id="hand-and-gigabrain",
        )
        analyzer = FakeAnalyzer(
            self.analyzable,
            self.unavailable,
        )
        report, _ = self.execute(
            corpus,
            labels,
            unused,
            public,
            private,
            analyzer,
        )

        self.assertEqual(report["labelCoverage"]["unsupportedPlayerLabels"], 1)
        self.assertEqual(report["labelCoverage"]["supportedPlayerLabels"], 1)
        self.assertEqual(
            report["labelCoverage"]["unavailableSupportedPlayerLabels"], 1
        )
        self.assertIsNone(report["metrics"]["combined"]["final"])

    def test_rejects_duplicate_semantic_game_and_training_overlap(self) -> None:
        corpus, labels, unused, public, private = self.inputs()
        corpus_value = json.loads(corpus.path.read_text(encoding="utf-8"))
        corpus_value["games"].append(dict(corpus_value["games"][0]))
        duplicate = write_json(self.directory, "duplicate.json", corpus_value)
        with self.assertRaisesRegex(
            RealDomainBenchmarkError, "duplicate PGN bytes"
        ):
            self.execute(
                duplicate, labels, unused, public, private
            )

        alternate_pgn = PGN.replace(
            '[Event "Consented completed game"]',
            '[Event "Same moves, different identifying headers"]',
        )
        semantic_value = json.loads(corpus.path.read_text(encoding="utf-8"))
        semantic_value["games"].append(
            {
                **semantic_value["games"][0],
                "pgn": alternate_pgn,
                "pgnSha256": hashlib.sha256(alternate_pgn.encode()).hexdigest(),
            }
        )
        semantic_duplicate = write_json(
            self.directory,
            "semantic-duplicate.json",
            semantic_value,
        )
        with self.assertRaisesRegex(RealDomainBenchmarkError, "semantically"):
            self.execute(
                semantic_duplicate,
                labels,
                unused,
                public,
                private,
            )

        wrong_public_value = json.loads(
            public.path.read_text(encoding="utf-8")
        )
        wrong_public_value["corpusRunId"] = "3" * 64
        wrong_public = write_json(
            self.directory,
            "wrong-public.json",
            wrong_public_value,
        )
        with self.assertRaisesRegex(
            RealDomainBenchmarkError,
            "incomplete or disagree",
        ):
            self.execute(
                corpus,
                labels,
                unused,
                wrong_public,
                private,
            )

        moves = ("e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7")
        corpus, labels, unused, public, private = self.inputs(
            primary_training_moves=moves
        )
        with self.assertRaisesRegex(
            RealDomainBenchmarkError,
            "replayed semantic training game",
        ):
            self.execute(corpus, labels, unused, public, private)

    def test_rejects_live_incomplete_seeded_or_label_leaking_inputs(self) -> None:
        corpus, labels, unused, public, private = self.inputs()
        value = json.loads(corpus.path.read_text(encoding="utf-8"))
        value["consent"]["liveCollection"] = True
        live = write_json(self.directory, "live.json", value)
        with self.assertRaisesRegex(RealDomainBenchmarkError, "consent"):
            self.execute(live, labels, unused, public, private)

        value = json.loads(corpus.path.read_text(encoding="utf-8"))
        value["games"][0]["simulationSeed"] = 20260811
        seeded = write_json(self.directory, "seeded.json", value)
        with self.assertRaisesRegex(RealDomainBenchmarkError, "real"):
            self.execute(seeded, labels, unused, public, private)

        leaking_pgn = PGN.replace(
            '[Event "Consented completed game"]',
            '[Drawback "Vegan"]',
        )
        (
            leaking,
            leak_labels,
            leak_unused,
            leak_public,
            leak_private,
        ) = self.inputs(pgn=leaking_pgn)
        with self.assertRaisesRegex(RealDomainBenchmarkError, "headers"):
            self.execute(
                leaking,
                leak_labels,
                leak_unused,
                leak_public,
                leak_private,
            )

    def test_stage_a_api_is_procedurally_label_separated(self) -> None:
        corpus, labels, unused, public, private = self.inputs()
        analyzer = FakeAnalyzer(self.analyzable, self.unavailable)
        labels.path.write_text("not canonical yet", encoding="utf-8")
        bundle_path = self.directory / "label-blind.json"
        create_real_domain_prediction_bundle(
            corpus=corpus,
            public_training_release=public,
            private_training_manifest=private,
            candidate_training_corpus_set=(
                self.candidate_training_corpus_set
            ),
            training_datasets=self.training_datasets,
            supplement_manifests=self.supplement_manifests,
            supplement_plans=self.supplement_plans,
            observed_catalog=ContentAddressedJson(CATALOG, CATALOG_SHA),
            analyzer=analyzer,
            output=bundle_path,
        )
        self.assertEqual(len(analyzer.calls), 1)
        bundle_text = bundle_path.read_text(encoding="utf-8")
        self.assertNotIn("Vegan", bundle_text)
        self.assertNotIn("Checkers", bundle_text)
        self.assertNotIn(
            "revealed_labels",
            inspect.signature(
                create_real_domain_prediction_bundle
            ).parameters,
        )

    def test_stage_a_rejects_omitted_hard_negative_source(self) -> None:
        corpus, _labels, unused, public, private = self.inputs()
        with self.assertRaisesRegex(
            RealDomainBenchmarkError,
            "incomplete or disagree",
        ):
            create_real_domain_prediction_bundle(
                corpus=corpus,
                public_training_release=public,
                private_training_manifest=private,
                candidate_training_corpus_set=(
                    self.candidate_training_corpus_set
                ),
                training_datasets=self.training_datasets[:-1],
                supplement_manifests=self.supplement_manifests,
                supplement_plans=self.supplement_plans,
                observed_catalog=ContentAddressedJson(CATALOG, CATALOG_SHA),
                analyzer=FakeAnalyzer(self.analyzable, self.unavailable),
                output=self.directory / "should-not-exist-dataset.json",
            )
        candidate = json.loads(
            self.candidate_training_corpus_set.path.read_text(encoding="utf-8")
        )
        candidate["supplements"].pop()
        incomplete = write_json(
            self.directory,
            "incomplete-training-corpus-set.json",
            candidate,
        )
        with self.assertRaisesRegex(
            RealDomainBenchmarkError,
            "invalid or incomplete",
        ):
            create_real_domain_prediction_bundle(
                corpus=corpus,
                public_training_release=public,
                private_training_manifest=private,
                candidate_training_corpus_set=incomplete,
                training_datasets=self.training_datasets,
                supplement_manifests=self.supplement_manifests,
                supplement_plans=self.supplement_plans,
                observed_catalog=ContentAddressedJson(CATALOG, CATALOG_SHA),
                analyzer=FakeAnalyzer(self.analyzable, self.unavailable),
                output=self.directory / "should-not-exist.json",
            )
        self.supplement_manifests[0].path.write_bytes(b"tampered\n")
        with self.assertRaisesRegex(RealDomainBenchmarkError, "SHA-256"):
            self.execute(corpus, _labels, unused, public, private)

    def test_semantic_identity_excludes_declared_result(self) -> None:
        corpus, _labels, _unused, _public, _private = self.inputs()
        original = load_corpus(corpus)[0]
        changed_result_pgn = PGN.replace("1-0", "0-1")
        changed, *_ = self.inputs(pgn=changed_result_pgn)
        changed_value = json.loads(changed.path.read_text(encoding="utf-8"))
        changed_value["games"][0]["result"] = "0-1"
        changed = write_json(
            self.directory,
            "changed-result.json",
            changed_value,
        )
        self.assertNotEqual(
            original.pgn_sha256,
            load_corpus(changed)[0].pgn_sha256,
        )
        self.assertEqual(
            original.semantic_sha256,
            load_corpus(changed)[0].semantic_sha256,
        )

    def test_rejects_wrong_analysis_identity_domain_and_no_clobber(self) -> None:
        corpus, labels, unused, public, private = self.inputs()

        class WrongAnalyzer(FakeAnalyzer):
            def analyze_completed_game(self, **kwargs: str) -> dict[str, object]:
                result = super().analyze_completed_game(**kwargs)
                result["predictor"]["mode"] = "symbolic-only"
                return result

        with self.assertRaisesRegex(RealDomainBenchmarkError, "approved"):
            self.execute(
                corpus,
                labels,
                unused,
                public,
                private,
                WrongAnalyzer(self.analyzable, self.unavailable),
            )

        report, _ = self.execute(
            corpus, labels, unused, public, private
        )
        self.assertEqual(report["scope"]["aggregateOnly"], True)
        bundle_path = self.directory / "predictions.json"
        bundle_reference = ContentAddressedJson(
            bundle_path,
            hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
        )
        with self.assertRaises(FileExistsError):
            publish_real_domain_benchmark_report(
                prediction_bundle=bundle_reference,
                revealed_labels=labels,
                observed_catalog=ContentAddressedJson(CATALOG, CATALOG_SHA),
                output=self.directory / "report.json",
            )

    def test_subprocess_uses_pinned_copies_and_authenticated_artifacts(self) -> None:
        artifacts: list[ContentAddressedJson] = []
        for index, name in enumerate(
            (
                "browser.json",
                "ensemble.json",
                "calibration.json",
                "approval.json",
            ),
            start=1,
        ):
            artifacts.append(
                write_json(
                    self.directory,
                    name,
                    {"artifact": index},
                )
            )
        launcher = self.directory / "launcher.py"
        launcher.write_text(
            "\n".join(
                (
                    "import hashlib, json, os",
                    "from pathlib import Path",
                    "assert 'PATH' not in os.environ",
                    "root = Path.cwd().resolve()",
                    "assert Path(__file__).resolve().parent == root",
                    "keys = {",
                    "'browserArtifactSha256': 'DRAWBACKTRAINER_BROWSER_ARTIFACT',",
                    "'ensembleReleaseSha256': 'DRAWBACKTRAINER_ENSEMBLE_RELEASE',",
                    "'calibrationSha256': 'DRAWBACKTRAINER_CALIBRATION',",
                    "'approvalEvidenceSha256': 'DRAWBACKTRAINER_APPROVAL_EVIDENCE',",
                    "}",
                    "predictor = {'mode': 'hybrid-v21-ensemble'}",
                    "for output_key, env_key in keys.items():",
                    " path = Path(os.environ[env_key]).resolve()",
                    " assert path.parent == root",
                    " predictor[output_key] = hashlib.sha256(path.read_bytes()).hexdigest()",
                    "value = {",
                    "'pgnSha256': os.environ['DRAWBACKTRAINER_EXPECTED_SANITIZED_PGN_SHA256'],",
                    "'predictor': predictor,",
                    "}",
                    "payload = (json.dumps(value, indent=2, sort_keys=True) + '\\n').encode()",
                    "Path(os.environ['DRAWBACKTRAINER_ANALYSIS_OUTPUT']).write_bytes(payload)",
                )
            )
            + "\n",
            encoding="utf-8",
        )
        runtime = Path(sys.executable)
        analyzer = ApprovedSubprocessAnalyzer(
            runtime=runtime,
            runtime_sha256=hashlib.sha256(runtime.read_bytes()).hexdigest(),
            launcher=launcher,
            launcher_sha256=hashlib.sha256(launcher.read_bytes()).hexdigest(),
            browser_artifact=artifacts[0],
            ensemble_release=artifacts[1],
            calibration=artifacts[2],
            approval_evidence=artifacts[3],
            runtime_dependencies=self.runtime_dependencies(),
        )
        result = analyzer.analyze_completed_game(
            pgn_sha256="4" * 64,
            pgn='[Result "1-0"]\n\n1. e4 1-0\n',
        )
        self.assertEqual(result["pgnSha256"], "4" * 64)
        artifacts[0].path.write_bytes(b"changed")
        with self.assertRaisesRegex(RealDomainBenchmarkError, "SHA-256"):
            analyzer.analyze_completed_game(
                pgn_sha256="4" * 64,
                pgn='[Result "1-0"]\n\n1. e4 1-0\n',
            )

    def test_subprocess_rejects_oversized_stdout_and_launcher_tamper(self) -> None:
        artifacts = [
            write_json(self.directory, f"artifact-{index}.json", {"i": index})
            for index in range(4)
        ]
        launcher = self.directory / "oversize.py"
        launcher.write_text(
            "import sys\nsys.stdout.buffer.write(b'x' * 4096)\n",
            encoding="utf-8",
        )
        runtime = Path(sys.executable)
        analyzer = ApprovedSubprocessAnalyzer(
            runtime=runtime,
            runtime_sha256=hashlib.sha256(runtime.read_bytes()).hexdigest(),
            launcher=launcher,
            launcher_sha256=hashlib.sha256(launcher.read_bytes()).hexdigest(),
            browser_artifact=artifacts[0],
            ensemble_release=artifacts[1],
            calibration=artifacts[2],
            approval_evidence=artifacts[3],
            runtime_dependencies=self.runtime_dependencies(),
            maximum_output_bytes=1024,
        )
        with self.assertRaisesRegex(RealDomainBenchmarkError, "exceeds"):
            analyzer.analyze_completed_game(
                pgn_sha256="4" * 64,
                pgn='[Result "1-0"]\n\n1. e4 1-0\n',
            )

    def test_subprocess_fast_parent_exit_still_kills_descendant(self) -> None:
        artifacts = [
            write_json(self.directory, f"tree-artifact-{index}.json", {"i": index})
            for index in range(4)
        ]
        marker = self.directory / "descendant-survived"
        child_source = (
            "import time\n"
            "from pathlib import Path\n"
            "time.sleep(1)\n"
            f"Path({str(marker)!r}).write_text('bad')\n"
        )
        launcher = self.directory / "tree-launcher.py"
        launcher.write_text(
            "\n".join(
                (
                    "import subprocess, sys, time",
                    "from pathlib import Path",
                    "child = Path.cwd() / 'child.py'",
                    f"child.write_text({child_source!r})",
                    "subprocess.Popen([sys.executable, str(child)])",
                    "sys.stdout.buffer.write(b'x' * 4096)",
                    "sys.stdout.buffer.flush()",
                )
            )
            + "\n",
            encoding="utf-8",
        )
        runtime = Path(sys.executable)
        analyzer = ApprovedSubprocessAnalyzer(
            runtime=runtime,
            runtime_sha256=hashlib.sha256(runtime.read_bytes()).hexdigest(),
            launcher=launcher,
            launcher_sha256=hashlib.sha256(launcher.read_bytes()).hexdigest(),
            browser_artifact=artifacts[0],
            ensemble_release=artifacts[1],
            calibration=artifacts[2],
            approval_evidence=artifacts[3],
            runtime_dependencies=self.runtime_dependencies(),
            maximum_output_bytes=1024,
        )
        with self.assertRaisesRegex(RealDomainBenchmarkError, "exceeds"):
            analyzer.analyze_completed_game(
                pgn_sha256="4" * 64,
                pgn='[Result "1-0"]\n\n1. e4 1-0\n',
            )
        time.sleep(1.3)
        self.assertFalse(marker.exists())
        launcher.write_text("raise SystemExit(0)\n", encoding="utf-8")
        with self.assertRaisesRegex(RealDomainBenchmarkError, "launcher changed"):
            analyzer.analyze_completed_game(
                pgn_sha256="4" * 64,
                pgn='[Result "1-0"]\n\n1. e4 1-0\n',
            )


if __name__ == "__main__":
    unittest.main()
