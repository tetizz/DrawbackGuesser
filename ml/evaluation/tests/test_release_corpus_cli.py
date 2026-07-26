from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ml.evaluation.cli import (
    _checkpoint_corpus_requirement,
    _require_checkpoint_corpus_provenance,
    _release_evaluation_inputs,
)
from ml.evaluation.tests.training_corpus_set_fixture import (
    training_corpus_set_fixture,
)


class _Audited:
    seeds = (101, 202)
    max_plies = 64

    def provenance(self) -> dict[str, object]:
        return {
            "manifest_sha256": "11" * 32,
            "split": "validation",
            "engine_fingerprint": "stockfish:18:" + "22" * 32 + ":" + "33" * 32,
            "evaluator_policy_id": "stockfish-bestmove-v1",
            "evaluator_policy_version": 1,
            "rule_ids": ["vegan"],
            "symbolic_feature_version": 6,
            "release_root_sha256": "44" * 32,
            "corpus_run_id": "55" * 32,
        }


def arguments(directory: Path, *, partition: str | None) -> argparse.Namespace:
    return argparse.Namespace(
        public_root=directory / "public.json",
        private_validation=directory / "validation.private.json",
        dataset=directory / "validation.ndjson",
        manifest=None,
        split="validation",
        validation_partition=partition,
    )


class ReleaseCorpusCliTests(unittest.TestCase):
    def test_release_mode_uses_only_validation_private_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            values = arguments(directory, partition="selection")
            audited = _Audited()
            with patch(
                "ml.evaluation.cli.audit_private_corpus_split",
                return_value=audited,
            ) as private_audit, patch(
                "ml.evaluation.cli.audit_corpus_split"
            ) as legacy_audit:
                loaded, corpus, manifest, reaudit = _release_evaluation_inputs(
                    values
                )
                self.assertIs(loaded, audited)
                self.assertEqual(corpus["maxPlies"], 64)
                self.assertEqual(manifest.validation, audited.seeds)
                self.assertEqual(manifest.train, ())
                self.assertEqual(manifest.test, ())
                self.assertIs(reaudit(), audited)
            legacy_audit.assert_not_called()
            self.assertEqual(private_audit.call_count, 2)
            for call in private_audit.call_args_list:
                self.assertEqual(call.args[1], values.private_validation)
                self.assertEqual(call.args[2], values.dataset)
                self.assertEqual(call.args[3], "validation")

    def test_legacy_mode_cannot_select_calibrate_or_gate(self) -> None:
        for partition in ("selection", "calibration-fit", "gate"):
            values = argparse.Namespace(
                public_root=None,
                private_validation=None,
                dataset=Path("validation.ndjson"),
                manifest=Path("legacy.json"),
                split="validation",
                validation_partition=partition,
            )
            with patch("ml.evaluation.cli.audit_corpus_split") as legacy_audit:
                with self.assertRaisesRegex(ValueError, "split-private"):
                    _release_evaluation_inputs(values)
                legacy_audit.assert_not_called()

    def test_release_checkpoint_compatibility_uses_root_not_private_hash(self) -> None:
        requirement = _checkpoint_corpus_requirement(_Audited())
        self.assertEqual(requirement["release_root_sha256"], "44" * 32)
        self.assertEqual(requirement["corpus_run_id"], "55" * 32)
        self.assertNotIn("manifest_sha256", requirement)
        self.assertEqual(requirement["split"], "train")

    def test_release_checkpoint_accepts_authenticated_aggregate_training_set(
        self,
    ) -> None:
        corpus_set = training_corpus_set_fixture()
        primary = corpus_set["primary"]
        assert isinstance(primary, dict)
        audited = _Audited()
        expected = audited.provenance()
        for key in (
            "release_root_sha256",
            "corpus_run_id",
            "engine_fingerprint",
            "evaluator_policy_id",
            "evaluator_policy_version",
            "symbolic_feature_version",
        ):
            expected[key] = primary[key]
        audited.provenance = lambda: expected  # type: ignore[method-assign]
        predictor = type(
            "Predictor",
            (),
            {
                "corpus_provenance": {
                    "training_corpus_set": corpus_set,
                    "training_corpus_set_sha256": corpus_set["sha256"],
                }
            },
        )()

        _require_checkpoint_corpus_provenance(predictor, audited)

        for key in ("release_root_sha256", "corpus_run_id"):
            with self.subTest(key=key):
                changed = dict(expected)
                changed[key] = "9" * 64
                audited.provenance = (  # type: ignore[method-assign]
                    lambda changed=changed: changed
                )
                with self.assertRaisesRegex(ValueError, key):
                    _require_checkpoint_corpus_provenance(predictor, audited)

    def test_release_checkpoint_rejects_tampered_aggregate_training_set(
        self,
    ) -> None:
        corpus_set = training_corpus_set_fixture()
        predictor = type(
            "Predictor",
            (),
            {
                "corpus_provenance": {
                    "training_corpus_set": corpus_set,
                    "training_corpus_set_sha256": "9" * 64,
                }
            },
        )()
        with self.assertRaisesRegex(ValueError, "hash does not match"):
            _require_checkpoint_corpus_provenance(predictor, _Audited())

    def test_rejects_partial_or_mixed_release_references(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            values = arguments(directory, partition=None)
            values.private_validation = None
            with self.assertRaisesRegex(ValueError, "requires both"):
                _release_evaluation_inputs(values)
            values.private_validation = directory / "validation.private.json"
            values.manifest = directory / "legacy.json"
            with self.assertRaisesRegex(ValueError, "cannot also consume"):
                _release_evaluation_inputs(values)


if __name__ == "__main__":
    unittest.main()
