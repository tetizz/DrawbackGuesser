from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import _bootstrap  # noqa: F401

from drawback_ml.browser_artifact import (
    BROWSER_ARTIFACT_FORMAT,
    BROWSER_HYBRID_ARTIFACT_VERSION,
    BROWSER_OBSERVATION_ARTIFACT_VERSION,
    BROWSER_TENSOR_ENCODING,
    EXPORTED_HYBRID_TENSORS,
    EXPORTED_TENSORS,
    BrowserArtifactError,
    export_browser_artifact,
)
from drawback_ml.cli import main
from drawback_ml.checkpoint import fusion_grid_drawback_objective_metadata
from drawback_ml.features import FEATURE_DIMENSION, MOVE_VOCABULARY_SIZE
from drawback_ml.model import ModelConfig, create_model, create_sequence_model
from drawback_ml.sequence import (
    ObservationTokenizerV2,
    PublicSequenceObservation,
    SanTokenizer,
)
from drawback_ml.symbolic_schema import (
    SYMBOLIC_FEATURE_DIMENSION,
    SYMBOLIC_FEATURE_VERSION,
    SYMBOLIC_RULE_IDS,
)

try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, "PyTorch is not installed")
class BrowserArtifactTests(unittest.TestCase):
    def _payload(self) -> dict[str, object]:
        assert torch is not None
        config = ModelConfig(
            input_dimension=FEATURE_DIMENSION,
            drawback_classes=2,
            parameter_classes=1,
            legal_mask_dimension=MOVE_VOCABULARY_SIZE,
            hidden_dimension=3,
        )
        torch.manual_seed(91)
        model = create_model(config)
        return {
            "format_version": 3,
            "seed": 91,
            "epoch": 2,
            "drawback_vocabulary": ["checkers", "vegan"],
            "parameter_vocabulary": ["{}"],
            "model_config": {
                "input_dimension": config.input_dimension,
                "drawback_classes": config.drawback_classes,
                "parameter_classes": config.parameter_classes,
                "legal_mask_dimension": config.legal_mask_dimension,
                "hidden_dimension": config.hidden_dimension,
                "model_variant": "v1",
            },
            "training_metadata": {"feature_schema_version": 1},
            "model_state": model.state_dict(),
            "optimizer_state": {},
        }

    def _checkpoint(self, directory: Path, payload: object | None = None) -> Path:
        assert torch is not None
        path = directory / "checkpoint.pt"
        torch.save(self._payload() if payload is None else payload, path)
        return path

    def _hybrid_payload(self) -> dict[str, object]:
        assert torch is not None
        tokenizer = SanTokenizer.fit(
            [["e4", "e5"], ["Nf3", "Nc6"]],
            max_history=8,
        )
        config = ModelConfig(
            input_dimension=FEATURE_DIMENSION,
            drawback_classes=len(SYMBOLIC_RULE_IDS),
            parameter_classes=1,
            legal_mask_dimension=MOVE_VOCABULARY_SIZE,
            hidden_dimension=3,
            model_variant="v21-hybrid",
            san_vocabulary_size=len(tokenizer.vocabulary),
            san_embedding_dimension=2,
            sequence_hidden_dimension=4,
            symbolic_dimension=SYMBOLIC_FEATURE_DIMENSION,
            symbolic_hidden_dimension=5,
        )
        torch.manual_seed(92)
        model = create_sequence_model(config)
        return {
            "format_version": 3,
            "seed": 92,
            "epoch": 3,
            "drawback_vocabulary": list(reversed(SYMBOLIC_RULE_IDS)),
            "parameter_vocabulary": ["{}"],
            "model_config": {
                "input_dimension": config.input_dimension,
                "drawback_classes": config.drawback_classes,
                "parameter_classes": config.parameter_classes,
                "legal_mask_dimension": config.legal_mask_dimension,
                "hidden_dimension": config.hidden_dimension,
                "model_variant": config.model_variant,
                "san_vocabulary_size": config.san_vocabulary_size,
                "san_embedding_dimension": config.san_embedding_dimension,
                "sequence_hidden_dimension": config.sequence_hidden_dimension,
                "symbolic_dimension": config.symbolic_dimension,
                "symbolic_hidden_dimension": config.symbolic_hidden_dimension,
            },
            "training_metadata": {
                "feature_schema_version": 1,
                "symbolic_feature_version": SYMBOLIC_FEATURE_VERSION,
                "symbolic_rule_ids": list(SYMBOLIC_RULE_IDS),
                "san_tokenizer": tokenizer.metadata(),
            },
            "model_state": model.state_dict(),
            "optimizer_state": {},
        }

    def _observation_payload(
        self,
        mode: str = "exact-current-v2",
    ) -> dict[str, object]:
        assert torch is not None
        tokenizer = ObservationTokenizerV2.fit(
            (
                PublicSequenceObservation((), "e2e4"),
                PublicSequenceObservation(("e4",), "e7e5"),
            ),
            max_sequence=9,
        )
        config = ModelConfig(
            input_dimension=FEATURE_DIMENSION,
            drawback_classes=len(SYMBOLIC_RULE_IDS),
            parameter_classes=1,
            legal_mask_dimension=MOVE_VOCABULARY_SIZE,
            hidden_dimension=3,
            model_variant="v22-hybrid",
            sequence_observation_mode=mode,  # type: ignore[arg-type]
            san_vocabulary_size=len(tokenizer.vocabulary),
            san_embedding_dimension=2,
            sequence_hidden_dimension=4,
            symbolic_dimension=SYMBOLIC_FEATURE_DIMENSION,
            symbolic_hidden_dimension=5,
        )
        torch.manual_seed(93)
        model = create_sequence_model(config)
        return {
            "format_version": 3,
            "seed": 93,
            "epoch": 3,
            "drawback_vocabulary": list(reversed(SYMBOLIC_RULE_IDS)),
            "parameter_vocabulary": ["{}"],
            "model_config": {
                "input_dimension": config.input_dimension,
                "drawback_classes": config.drawback_classes,
                "parameter_classes": config.parameter_classes,
                "legal_mask_dimension": config.legal_mask_dimension,
                "hidden_dimension": config.hidden_dimension,
                "model_variant": config.model_variant,
                "sequence_observation_mode": config.sequence_observation_mode,
                "san_vocabulary_size": config.san_vocabulary_size,
                "san_embedding_dimension": config.san_embedding_dimension,
                "sequence_hidden_dimension": config.sequence_hidden_dimension,
                "symbolic_dimension": config.symbolic_dimension,
                "symbolic_hidden_dimension": config.symbolic_hidden_dimension,
            },
            "training_metadata": {
                "feature_schema_version": 1,
                "symbolic_feature_version": SYMBOLIC_FEATURE_VERSION,
                "symbolic_rule_ids": list(SYMBOLIC_RULE_IDS),
                "sequence_observation_mode": mode,
                "san_tokenizer": tokenizer.metadata(),
                "drawback_loss_objective": (
                    fusion_grid_drawback_objective_metadata()
                ),
            },
            "model_state": model.state_dict(),
            "optimizer_state": {},
        }

    def test_export_is_canonical_deterministic_and_provenanced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            checkpoint = self._checkpoint(directory)
            first = directory / "first.json"
            second = directory / "second.json"

            export_browser_artifact(checkpoint, first)
            export_browser_artifact(checkpoint, second)

            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertTrue(first.read_bytes().endswith(b"\n"))
            artifact = json.loads(first.read_text(encoding="utf-8"))
            self.assertEqual(artifact["format"], BROWSER_ARTIFACT_FORMAT)
            self.assertEqual(artifact["formatVersion"], 1)
            self.assertEqual(artifact["modelVariant"], "v1")
            self.assertEqual(artifact["featureSchemaVersion"], 1)
            self.assertEqual(
                artifact["sourceCheckpointSha256"],
                hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                artifact["drawbackVocabulary"], ["checkers", "vegan"]
            )
            self.assertEqual(
                artifact["dimensions"],
                {
                    "input": FEATURE_DIMENSION,
                    "hidden": 3,
                    "drawbackClasses": 2,
                },
            )
            self.assertEqual(set(artifact["tensors"]), set(EXPORTED_TENSORS))
            self.assertEqual(
                artifact["tensors"]["encoder.0.weight"]["shape"],
                [3, FEATURE_DIMENSION],
            )
            self.assertNotIn("legal_mask.weight", artifact["tensors"])
            self.assertNotIn("white_parameters.weight", artifact["tensors"])

    def test_export_accepts_exact_retry_without_replacing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            checkpoint = self._checkpoint(directory)
            output = directory / "artifact.json"

            export_browser_artifact(checkpoint, output)
            first_stat = output.stat()
            first_bytes = output.read_bytes()
            export_browser_artifact(checkpoint, output)

            self.assertEqual(output.read_bytes(), first_bytes)
            self.assertEqual(output.stat().st_ino, first_stat.st_ino)

    def test_export_preserves_different_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            checkpoint = self._checkpoint(directory)
            output = directory / "artifact.json"
            output.write_bytes(b"pre-existing artifact\n")

            with self.assertRaisesRegex(BrowserArtifactError, "cannot write"):
                export_browser_artifact(checkpoint, output)

            self.assertEqual(output.read_bytes(), b"pre-existing artifact\n")

    def test_cli_exports_the_same_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            checkpoint = self._checkpoint(directory)
            direct = directory / "direct.json"
            cli = directory / "cli.json"
            export_browser_artifact(checkpoint, direct)

            stdout = StringIO()
            with redirect_stdout(stdout):
                result = main(["export-browser", str(checkpoint), str(cli)])

            self.assertEqual(result, 0)
            self.assertEqual(stdout.getvalue().strip(), str(cli))
            self.assertEqual(direct.read_bytes(), cli.read_bytes())

    def test_hybrid_export_is_canonical_binary_and_provenanced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            checkpoint = self._checkpoint(directory, self._hybrid_payload())
            first = directory / "hybrid-first.json"
            second = directory / "hybrid-second.json"

            export_browser_artifact(checkpoint, first)
            export_browser_artifact(checkpoint, second)

            self.assertEqual(first.read_bytes(), second.read_bytes())
            artifact = json.loads(first.read_text(encoding="utf-8"))
            self.assertEqual(
                artifact["formatVersion"], BROWSER_HYBRID_ARTIFACT_VERSION
            )
            self.assertEqual(artifact["modelVariant"], "v21-hybrid")
            self.assertEqual(
                artifact["symbolicFeatureVersion"], SYMBOLIC_FEATURE_VERSION
            )
            self.assertEqual(
                artifact["sourceCheckpointSha256"],
                hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            )
            self.assertEqual(artifact["symbolicRuleIds"], list(SYMBOLIC_RULE_IDS))
            self.assertEqual(
                artifact["drawbackVocabulary"],
                list(reversed(SYMBOLIC_RULE_IDS)),
            )
            self.assertEqual(
                artifact["tensorEncoding"], BROWSER_TENSOR_ENCODING
            )
            self.assertNotIn("sequenceObservationMode", artifact)
            self.assertEqual(set(artifact["tensors"]), set(EXPORTED_HYBRID_TENSORS))
            self.assertEqual(
                artifact["tensors"]["history_encoder.weight_ih_l0"]["shape"],
                [12, 2],
            )
            encoded = artifact["tensors"]["history_encoder.weight_ih_l0"]["data"]
            self.assertIsInstance(encoded, str)
            self.assertNotIn("legal_mask.weight", artifact["tensors"])
            self.assertNotIn("white_parameters.weight", artifact["tensors"])

    def test_observation_export_is_format_three_and_binds_ablation_mode(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            exact_checkpoint = self._checkpoint(
                directory,
                self._observation_payload("exact-current-v2"),
            )
            exact_output = directory / "exact.json"
            export_browser_artifact(exact_checkpoint, exact_output)
            exact = json.loads(exact_output.read_text(encoding="utf-8"))

            self.assertEqual(
                exact["formatVersion"],
                BROWSER_OBSERVATION_ARTIFACT_VERSION,
            )
            self.assertEqual(exact["modelVariant"], "v22-hybrid")
            self.assertEqual(
                exact["sequenceObservationMode"],
                "exact-current-v2",
            )
            self.assertEqual(
                exact["tokenizer"],
                self._observation_payload("exact-current-v2")[
                    "training_metadata"
                ]["san_tokenizer"],  # type: ignore[index]
            )
            self.assertEqual(exact["tokenizer"]["version"], 2)
            self.assertEqual(exact["tokenizer"]["max_sequence"], 9)

            masked_payload = self._observation_payload(
                "masked-current-v2"
            )
            masked_checkpoint = directory / "masked.pt"
            assert torch is not None
            torch.save(masked_payload, masked_checkpoint)
            masked_output = directory / "masked.json"
            export_browser_artifact(masked_checkpoint, masked_output)
            masked = json.loads(masked_output.read_text(encoding="utf-8"))
            self.assertEqual(
                masked["sequenceObservationMode"],
                "masked-current-v2",
            )
            self.assertEqual(masked["tokenizer"], exact["tokenizer"])
            self.assertEqual(masked["tensors"], exact["tensors"])

    def test_observation_export_rejects_mode_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            payload = self._observation_payload()
            training = payload["training_metadata"]
            assert isinstance(training, dict)
            training["sequence_observation_mode"] = "masked-current-v2"
            checkpoint = self._checkpoint(directory, payload)
            with self.assertRaisesRegex(BrowserArtifactError, "mode"):
                export_browser_artifact(
                    checkpoint,
                    directory / "invalid.json",
                )

    def test_observation_export_rejects_missing_fusion_objective(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            payload = self._observation_payload()
            training = payload["training_metadata"]
            assert isinstance(training, dict)
            training.pop("drawback_loss_objective")
            checkpoint = self._checkpoint(directory, payload)
            with self.assertRaisesRegex(BrowserArtifactError, "objective"):
                export_browser_artifact(
                    checkpoint,
                    directory / "invalid-objective.json",
                )

    def test_observation_export_rejects_browser_oversized_san_token(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            payload = self._observation_payload()
            training = payload["training_metadata"]
            assert isinstance(training, dict)
            tokenizer = training["san_tokenizer"]
            assert isinstance(tokenizer, dict)
            vocabulary = tokenizer["vocabulary"]
            assert isinstance(vocabulary, list)
            vocabulary.append("N" * 33)
            checkpoint = self._checkpoint(directory, payload)
            with self.assertRaisesRegex(
                BrowserArtifactError,
                "tokenizer is incompatible",
            ):
                export_browser_artifact(
                    checkpoint,
                    directory / "oversized-token.json",
                )

    def test_hybrid_export_rejects_incompatible_public_schemas(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)

            bad_symbolic = self._hybrid_payload()
            bad_symbolic["training_metadata"] = {
                **bad_symbolic["training_metadata"],  # type: ignore[arg-type]
                "symbolic_rule_ids": list(reversed(SYMBOLIC_RULE_IDS)),
            }
            checkpoint = self._checkpoint(directory, bad_symbolic)
            with self.assertRaisesRegex(BrowserArtifactError, "symbolic schema"):
                export_browser_artifact(checkpoint, directory / "symbolic.json")

            bad_tokenizer = self._hybrid_payload()
            bad_tokenizer["training_metadata"] = {
                **bad_tokenizer["training_metadata"],  # type: ignore[arg-type]
                "san_tokenizer": {
                    "kind": "exact-san-token",
                    "version": 1,
                    "vocabulary": ["<pad>", "<unk>"],
                    "max_history": 8,
                    "padding": "left",
                    "truncation": "keep-most-recent",
                },
            }
            checkpoint = self._checkpoint(directory, bad_tokenizer)
            with self.assertRaisesRegex(BrowserArtifactError, "tokenizer"):
                export_browser_artifact(checkpoint, directory / "tokenizer.json")

            incomplete_vocabulary = self._hybrid_payload()
            incomplete_vocabulary["drawback_vocabulary"] = list(
                SYMBOLIC_RULE_IDS[:-1]
            )
            checkpoint = self._checkpoint(directory, incomplete_vocabulary)
            with self.assertRaisesRegex(BrowserArtifactError, "vocabulary"):
                export_browser_artifact(checkpoint, directory / "vocabulary.json")

    def test_hybrid_export_rejects_shape_nonfinite_and_unexpected_state(self) -> None:
        assert torch is not None
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)

            malformed = self._hybrid_payload()
            malformed_state = dict(malformed["model_state"])  # type: ignore[arg-type]
            malformed_state["history_encoder.weight_hh_l0"] = torch.zeros((1, 1))
            malformed["model_state"] = malformed_state
            checkpoint = self._checkpoint(directory, malformed)
            with self.assertRaisesRegex(BrowserArtifactError, "shape"):
                export_browser_artifact(checkpoint, directory / "shape-v2.json")

            non_finite = self._hybrid_payload()
            non_finite_state = dict(non_finite["model_state"])  # type: ignore[arg-type]
            bad_bias = non_finite_state["symbolic_encoder.2.bias"].clone()
            bad_bias[0] = float("inf")
            non_finite_state["symbolic_encoder.2.bias"] = bad_bias
            non_finite["model_state"] = non_finite_state
            checkpoint = self._checkpoint(directory, non_finite)
            with self.assertRaisesRegex(BrowserArtifactError, "non-finite"):
                export_browser_artifact(checkpoint, directory / "infinite-v2.json")

            invalid_auxiliary = self._hybrid_payload()
            invalid_auxiliary_state = dict(
                invalid_auxiliary["model_state"]  # type: ignore[arg-type]
            )
            bad_legal_bias = invalid_auxiliary_state["legal_mask.bias"].clone()
            bad_legal_bias[0] = float("nan")
            invalid_auxiliary_state["legal_mask.bias"] = bad_legal_bias
            invalid_auxiliary["model_state"] = invalid_auxiliary_state
            checkpoint = self._checkpoint(directory, invalid_auxiliary)
            with self.assertRaisesRegex(BrowserArtifactError, "non-finite"):
                export_browser_artifact(
                    checkpoint,
                    directory / "auxiliary-infinite-v2.json",
                )

            unexpected = self._hybrid_payload()
            unexpected_state = dict(unexpected["model_state"])  # type: ignore[arg-type]
            unexpected_state["untrusted.weight"] = unexpected_state[
                "board_encoder.0.bias"
            ]
            unexpected["model_state"] = unexpected_state
            checkpoint = self._checkpoint(directory, unexpected)
            with self.assertRaisesRegex(BrowserArtifactError, "unexpected"):
                export_browser_artifact(checkpoint, directory / "extra-v2.json")

    def test_hybrid_export_enforces_the_raw_per_tensor_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            checkpoint = self._checkpoint(directory, self._hybrid_payload())
            with patch(
                "drawback_ml.browser_artifact.MAX_BROWSER_TENSOR_BYTES",
                4,
            ):
                with self.assertRaisesRegex(
                    BrowserArtifactError,
                    "tensor .* exceeds 4 raw bytes",
                ):
                    export_browser_artifact(
                        checkpoint,
                        directory / "oversized-tensor.json",
                    )

    def test_rejects_unsupported_architecture_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            payload = self._payload()
            payload["model_config"] = {
                **payload["model_config"],  # type: ignore[arg-type]
                "model_variant": "v2-gru",
            }
            checkpoint = self._checkpoint(directory, payload)
            output = directory / "artifact.json"

            with self.assertRaisesRegex(BrowserArtifactError, "only.*v21-hybrid"):
                export_browser_artifact(checkpoint, output)
            self.assertFalse(output.exists())

    def test_rejects_models_larger_than_the_browser_hidden_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            payload = self._payload()
            payload["model_config"] = {
                **payload["model_config"],  # type: ignore[arg-type]
                "hidden_dimension": 257,
            }
            checkpoint = self._checkpoint(directory, payload)
            with self.assertRaisesRegex(BrowserArtifactError, "browser limit"):
                export_browser_artifact(checkpoint, directory / "large.json")

    def test_rejects_malformed_or_non_finite_tensors(self) -> None:
        assert torch is not None
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            malformed = self._payload()
            malformed_state = dict(malformed["model_state"])  # type: ignore[arg-type]
            malformed_state["encoder.0.weight"] = torch.zeros((1, 1))
            malformed["model_state"] = malformed_state
            checkpoint = self._checkpoint(directory, malformed)
            with self.assertRaisesRegex(BrowserArtifactError, "shape"):
                export_browser_artifact(checkpoint, directory / "shape.json")

            non_finite = self._payload()
            non_finite_state = dict(non_finite["model_state"])  # type: ignore[arg-type]
            bad_bias = non_finite_state["black_drawback.bias"].clone()
            bad_bias[0] = float("nan")
            non_finite_state["black_drawback.bias"] = bad_bias
            non_finite["model_state"] = non_finite_state
            checkpoint = self._checkpoint(directory, non_finite)
            with self.assertRaisesRegex(BrowserArtifactError, "non-finite"):
                export_browser_artifact(checkpoint, directory / "nan.json")

    def test_rejects_schema_vocabulary_and_unexpected_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            bad_schema = self._payload()
            bad_schema["training_metadata"] = {"feature_schema_version": 999}
            checkpoint = self._checkpoint(directory, bad_schema)
            with self.assertRaisesRegex(BrowserArtifactError, "schema"):
                export_browser_artifact(checkpoint, directory / "schema.json")

            bad_vocabulary = self._payload()
            bad_vocabulary["drawback_vocabulary"] = ["vegan"]
            checkpoint = self._checkpoint(directory, bad_vocabulary)
            with self.assertRaisesRegex(BrowserArtifactError, "vocabulary"):
                export_browser_artifact(checkpoint, directory / "vocab.json")

            unknown_vocabulary = self._payload()
            unknown_vocabulary["drawback_vocabulary"] = [
                "checkers",
                "not-a-real-drawback",
            ]
            checkpoint = self._checkpoint(directory, unknown_vocabulary)
            with self.assertRaisesRegex(BrowserArtifactError, "unknown IDs"):
                export_browser_artifact(checkpoint, directory / "unknown.json")

            unexpected = self._payload()
            unexpected_state = dict(unexpected["model_state"])  # type: ignore[arg-type]
            unexpected_state["untrusted.weight"] = unexpected_state[
                "encoder.0.bias"
            ]
            unexpected["model_state"] = unexpected_state
            checkpoint = self._checkpoint(directory, unexpected)
            with self.assertRaisesRegex(BrowserArtifactError, "unexpected"):
                export_browser_artifact(checkpoint, directory / "extra.json")


if __name__ == "__main__":
    unittest.main()
