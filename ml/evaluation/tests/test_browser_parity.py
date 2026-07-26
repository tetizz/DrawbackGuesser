from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from ml.evaluation.browser_parity import (
    EVIDENCE_FORMAT,
    INPUT_FORMAT,
    TRANSCRIPT_FORMAT,
    authenticate_runtime_bindings,
    load_authenticated_input,
    publish_evidence,
    verify_transcript_bindings,
    _public_feature_record,
)
from ml.evaluation.validation_gate import PROTOCOL_ID, _canonical_pretty
from ml.training.drawback_ml.symbolic_schema import SYMBOLIC_RULE_IDS

PUBLIC_GENERATOR_PROTOCOL = {
    "id": "drawbacktrainer-public-pgn-parity-v1",
    "seedDomain": "public-parity-v1",
    "rootSeed": 0x5A17_2026,
    "gameCount": 8,
    "maxPlies": 320,
    "agentSchedule": ["random-legal", "human-like-weak", "greedy-material"],
}


class BrowserParityTest(unittest.TestCase):
    def test_authenticated_input_rejects_hidden_truth(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.json"
            value = {
                "format": INPUT_FORMAT,
                "version": 1,
                "protocolId": PROTOCOL_ID,
                "browserArtifactSha256": "a" * 64,
                "fixtureSha256": "",
                "partition": {
                    "id": "public-validation-parity-v1",
                    "split": "validation-parity",
                    "selectionSha256": "b" * 64,
                    "publicExampleCount": 1,
                },
                "bindings": {
                    "ensembleSha256": "c" * 64,
                    "calibrationSha256": "d" * 64,
                    "fusionSelectionSha256": "9" * 64,
                    "sourceRevision": "e" * 40,
                    "pnpmLockSha256": "f" * 64,
                },
                "publicFixture": {
                    "file": "fixture.json",
                    "sha256": "1" * 64,
                    "generatorProtocol": PUBLIC_GENERATOR_PROTOCOL,
                },
                "cases": [{"id": "case-1", "pgn": "1. e4", "truth": "vegan"}],
            }
            value["fixtureSha256"] = hashlib.sha256(
                _canonical_pretty(
                    {"partition": value["partition"], "cases": value["cases"]}
                )
            ).hexdigest()
            payload = _canonical_pretty(value)
            path.write_bytes(payload)
            with self.assertRaisesRegex(ValueError, "hidden data"):
                load_authenticated_input(
                    path, hashlib.sha256(payload).hexdigest(), "a" * 64
                )

    def test_publishes_exact_review_projection_no_clobber(self) -> None:
        transcript = _canonical_pretty(
            {
                "format": TRANSCRIPT_FORMAT,
                "version": 1,
                "browserArtifactSha256": "a" * 64,
                "fixtureSha256": "b" * 64,
                "workerE2ePassed": True,
                "maximumAbsoluteDifference": 0.0000004,
                "topKIdentical": True,
                "hardZeroSetsIdentical": True,
                "cases": [],
                "browserRuntime": {
                    "binarySha256": "3" * 64,
                    "version": "Browser 1.2.3",
                },
            }
        )
        parity_input = {
            "fixtureSha256": "b" * 64,
            "partition": {"selectionSha256": "d" * 64},
            "bindings": {
                "ensembleSha256": "e" * 64,
                "fusionSelectionSha256": "9" * 64,
                "sourceRevision": "f" * 40,
                "pnpmLockSha256": "1" * 64,
            },
            "publicFixture": {
                "file": "fixture.json",
                "sha256": "1" * 64,
                "generatorProtocol": PUBLIC_GENERATOR_PROTOCOL,
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "parity.json"
            received = publish_evidence(
                transcript_payload=transcript,
                browser_artifact_sha256="a" * 64,
                calibration_sha256="c" * 64,
                parity_input=parity_input,
                parity_input_sha256="2" * 64,
                output=output,
            )
            self.assertEqual(received["format"], EVIDENCE_FORMAT)
            self.assertEqual(received["max_absolute_difference"], 0.0000004)
            with self.assertRaises(FileExistsError):
                publish_evidence(
                    transcript_payload=transcript,
                    browser_artifact_sha256="a" * 64,
                    calibration_sha256="c" * 64,
                    parity_input=parity_input,
                    parity_input_sha256="2" * 64,
                    output=output,
                )

    def test_rejects_order_or_hard_zero_mismatch(self) -> None:
        for field in ("topKIdentical", "hardZeroSetsIdentical"):
            transcript = {
                "format": TRANSCRIPT_FORMAT,
                "version": 1,
                "browserArtifactSha256": "a" * 64,
                "fixtureSha256": "b" * 64,
                "workerE2ePassed": True,
                "maximumAbsoluteDifference": 0.0,
                "topKIdentical": True,
                "hardZeroSetsIdentical": True,
                "cases": [],
                "browserRuntime": {
                    "binarySha256": "3" * 64,
                    "version": "Browser 1.2.3",
                },
            }
            transcript[field] = False
            parity_input = {
                "fixtureSha256": "b" * 64,
                "partition": {"selectionSha256": "d" * 64},
                "bindings": {
                    "ensembleSha256": "e" * 64,
                    "fusionSelectionSha256": "9" * 64,
                    "sourceRevision": "f" * 40,
                    "pnpmLockSha256": "1" * 64,
                },
            }
            with tempfile.TemporaryDirectory() as temporary:
                with self.assertRaisesRegex(ValueError, "not passing"):
                    publish_evidence(
                        transcript_payload=_canonical_pretty(transcript),
                        browser_artifact_sha256="a" * 64,
                        calibration_sha256="c" * 64,
                        parity_input=parity_input,
                        parity_input_sha256="2" * 64,
                        output=Path(temporary) / "out.json",
                    )

    def test_transcript_must_echo_authenticated_bindings(self) -> None:
        parity_input = {
            "protocolId": PROTOCOL_ID,
            "browserArtifactSha256": "a" * 64,
            "fixtureSha256": "b" * 64,
            "partition": {"id": "parity"},
            "bindings": {"sourceRevision": "c" * 40},
        }
        transcript = {
            "protocolId": PROTOCOL_ID,
            "browserArtifactSha256": "a" * 64,
            "fixtureSha256": "b" * 64,
            "partition": {"id": "different"},
            "bindings": {"sourceRevision": "c" * 40},
        }
        with self.assertRaisesRegex(ValueError, "partition binding differs"):
            verify_transcript_bindings(
                _canonical_pretty(transcript), parity_input
            )

    def test_partition_and_cases_are_not_caller_weakenable(self) -> None:
        probabilities = {
            f"rule-{index}": (1.0 if index == 0 else 0.0)
            for index in range(6)
        }
        expected = {
            color: {
                "probabilities": probabilities,
                "topIds": [f"rule-{index}" for index in range(5)],
                "hardZeroIds": [f"rule-{index}" for index in range(1, 6)],
            }
            for color in ("white", "black")
        }
        base = {
            "format": INPUT_FORMAT,
            "version": 1,
            "protocolId": PROTOCOL_ID,
            "browserArtifactSha256": "a" * 64,
            "fixtureSha256": "",
            "partition": {
                "id": "validation-parity-v1",
                "split": "validation-parity",
                "selectionSha256": "b" * 64,
                "publicExampleCount": 1,
            },
            "bindings": {
                "ensembleSha256": "c" * 64,
                "calibrationSha256": "d" * 64,
                "fusionSelectionSha256": "9" * 64,
                "sourceRevision": "e" * 40,
                "pnpmLockSha256": "f" * 64,
            },
            "publicFixture": {
                "file": "fixture.json",
                "sha256": "1" * 64,
                "generatorProtocol": PUBLIC_GENERATOR_PROTOCOL,
            },
            "cases": [
                {
                    "id": "case-1",
                    "pgn": "1. e4",
                    "pgnSha256": hashlib.sha256(b"1. e4").hexdigest(),
                    "expected": expected,
                }
            ],
        }

        def write_and_load(value: dict[str, object]) -> None:
            value["fixtureSha256"] = hashlib.sha256(
                _canonical_pretty(
                    {"partition": value["partition"], "cases": value["cases"]}
                )
            ).hexdigest()
            payload = _canonical_pretty(value)
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "input.json"
                path.write_bytes(payload)
                load_authenticated_input(
                    path, hashlib.sha256(payload).hexdigest(), "a" * 64
                )

        for mutate, message in (
            (
                lambda value: value["partition"].update({"id": ""}),
                "partition identity",
            ),
            (
                lambda value: value["partition"].update(
                    {"selectionSha256": "NOT-A-DIGEST"}
                ),
                "partition identity",
            ),
            (
                lambda value: value["partition"].update(
                    {"publicExampleCount": 2}
                ),
                "partition identity",
            ),
            (
                lambda value: (
                    value["cases"].append(value["cases"][0]),
                    value["partition"].update({"publicExampleCount": 2}),
                ),
                "not unique",
            ),
            (
                lambda value: value["cases"][0]["expected"]["white"].update(
                    {"topIds": ["rule-0"]}
                ),
                "head is invalid",
            ),
            (
                lambda value: value["bindings"].pop(
                    "fusionSelectionSha256"
                ),
                "authentication metadata",
            ),
        ):
            value = json.loads(json.dumps(base))
            mutate(value)
            with self.assertRaisesRegex(ValueError, message):
                write_and_load(value)

    def test_clean_source_rejects_untracked_source_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            (repository / "ml").mkdir()
            (repository / "pnpm-lock.yaml").write_text("lock\n", encoding="utf-8")
            calibration = repository / "calibration.json"
            calibration.write_bytes(b"calibration\n")
            calibration_sha = hashlib.sha256(calibration.read_bytes()).hexdigest()
            artifact = repository / "artifact.json"
            artifact.write_bytes(
                _canonical_pretty(
                    {
                        "ensemble": {
                            "sourceEnsembleReleaseSha256": "a" * 64,
                            "sourceFusionSelectionSha256": "9" * 64,
                        },
                        "calibration": {
                            "sourceCalibrationSha256": calibration_sha
                        },
                    }
                )
            )
            subprocess.run(["git", "init"], cwd=repository, check=True, capture_output=True)
            subprocess.run(
                ["git", "config", "user.name", "test"],
                cwd=repository,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.email", "test@example.invalid"],
                cwd=repository,
                check=True,
            )
            subprocess.run(["git", "add", "."], cwd=repository, check=True)
            subprocess.run(
                ["git", "commit", "-m", "fixture"],
                cwd=repository,
                check=True,
                capture_output=True,
            )
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            parity_input = {
                "bindings": {
                    "ensembleSha256": "a" * 64,
                    "calibrationSha256": calibration_sha,
                    "fusionSelectionSha256": "9" * 64,
                    "sourceRevision": revision,
                    "pnpmLockSha256": hashlib.sha256(
                        (repository / "pnpm-lock.yaml").read_bytes()
                    ).hexdigest(),
                }
            }
            (repository / "ml" / "untracked.py").write_text(
                "raise SystemExit\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "clean HEAD"):
                authenticate_runtime_bindings(
                    repository, artifact, calibration, parity_input
                )

    def test_public_observation_rejects_replay_and_symbolic_mutations(self) -> None:
        count = len(SYMBOLIC_RULE_IDS)
        probability = 1.0 / count
        base = {
            "fenBefore": (
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/"
                "RNBQKBNR w KQkq - 0 1"
            ),
            "move": "e2e4",
            "moveNumber": 1,
            "ply": 0,
            "playerColor": "white",
            "historySan": [],
            "ordinaryLegalMoves": [
                "a2a3", "a2a4", "b1a3", "b1c3", "b2b3", "b2b4",
                "c2c3", "c2c4", "d2d3", "d2d4", "e2e3", "e2e4",
                "f2f3", "f2f4", "g1f3", "g1h3", "g2g3", "g2g4",
                "h2h3", "h2h4",
            ],
            "symbolicFeatureVersion": 6,
            "symbolic": {
                "ruleIds": list(SYMBOLIC_RULE_IDS),
                "whiteProbabilities": [probability] * count,
                "blackProbabilities": [probability] * count,
                "whiteEliminated": [False] * count,
                "blackEliminated": [False] * count,
            },
        }
        arguments = {
            "final_before": base["fenBefore"],
            "final_move": "e2e4",
            "final_history": (),
            "final_legal": tuple(base["ordinaryLegalMoves"]),
            "final_ply": 0,
        }
        self.assertEqual(
            _public_feature_record(base, **arguments).move, "e2e4"
        )
        mutations = []
        for field, replacement in (
            ("fenBefore", "bad-fen"),
            ("historySan", ["e4"]),
            ("ordinaryLegalMoves", ["e2e4"]),
        ):
            changed = json.loads(json.dumps(base))
            changed[field] = replacement
            mutations.append((changed, "disagrees"))
        bad_mask = json.loads(json.dumps(base))
        bad_mask["symbolic"]["whiteEliminated"][0] = True
        mutations.append((bad_mask, "probability is invalid"))
        bad_dimension = json.loads(json.dumps(base))
        bad_dimension["symbolic"]["blackProbabilities"].pop()
        mutations.append((bad_dimension, "dimensions differ"))
        for changed, message in mutations:
            with self.assertRaisesRegex(ValueError, message):
                _public_feature_record(changed, **arguments)


if __name__ == "__main__":
    unittest.main()
