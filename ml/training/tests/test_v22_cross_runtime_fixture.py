from __future__ import annotations

import hashlib
import importlib.util
from io import BytesIO
import json
from pathlib import Path
import unittest
import zipfile


try:
    import torch  # noqa: F401
except ImportError:
    torch = None


ROOT = Path(__file__).resolve().parents[3]
GENERATOR = ROOT / "scripts" / "generate-v22-cross-runtime-fixture.py"
FIXTURE = (
    ROOT / "apps" / "web" / "src" / "fixtures"
    / "v22-cross-runtime-golden.json"
)


def _load_generator() -> object:
    spec = importlib.util.spec_from_file_location(
        "generate_v22_cross_runtime_fixture",
        GENERATOR,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load v22 cross-runtime fixture generator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(torch is not None, "PyTorch is not installed")
class V22CrossRuntimeFixtureTests(unittest.TestCase):
    def test_committed_fixture_is_current_and_compact(self) -> None:
        generator = _load_generator()
        rendered = generator.generate_fixture_bytes()  # type: ignore[attr-defined]
        committed = (
            FIXTURE.read_text(encoding="utf-8")
            .replace("\r\n", "\n")
            .replace("\r", "\n")
            .encode("utf-8")
        )
        self.assertEqual(committed, rendered)
        self.assertLess(len(rendered), 64 * 1024)

        fixture = json.loads(rendered)
        self.assertEqual(
            fixture["format"],
            "drawbacktrainer-v22-cross-runtime-golden",
        )
        self.assertEqual(fixture["artifact"]["formatVersion"], 3)
        self.assertEqual(fixture["artifact"]["modelVariant"], "v22-hybrid")
        self.assertEqual(
            fixture["inputSpec"]["expectedProbabilityDecimalPlaces"],
            12,
        )
        for case in fixture["cases"]:
            for observation_mode in ("exact", "masked"):
                for color in ("white", "black"):
                    for probability in case["expected"][observation_mode][
                        color
                    ]:
                        self.assertEqual(
                            probability,
                            float(f"{probability:.12f}"),
                        )
        self.assertNotIn("model_state", fixture)
        self.assertNotIn("optimizer_state", fixture)
        self.assertNotIn("checkpoint", fixture)
        provenance = fixture["bindings"]["checkpointProvenance"]
        self.assertEqual(
            provenance["encoding"],
            "portable-torch-checkpoint-zip-v1",
        )
        self.assertEqual(
            provenance["exactSha256"],
            fixture["artifact"]["sourceCheckpointSha256"],
        )
        self.assertEqual(
            provenance["maskedSha256"],
            fixture["maskedArtifactDelta"]["sourceCheckpointSha256"],
        )

    def test_provenance_hashes_exact_portable_checkpoint_bytes(self) -> None:
        generator = _load_generator()
        tokenizer = generator._tokenizer()  # type: ignore[attr-defined]
        state = generator._intentional_state(  # type: ignore[attr-defined]
            tokenizer
        )
        payload = generator._checkpoint_payload(  # type: ignore[attr-defined]
            tokenizer,
            state,
            "exact-current-v2",
        )
        raw = generator._torch_checkpoint_bytes(  # type: ignore[attr-defined]
            payload
        )
        portable = generator._portable_checkpoint_bytes(  # type: ignore[attr-defined]
            raw
        )
        self.assertNotEqual(raw, portable)
        self.assertEqual(
            portable,
            generator._checkpoint_bytes(payload),  # type: ignore[attr-defined]
        )

        artifact = generator.build_fixture()["artifact"]  # type: ignore[attr-defined]
        self.assertEqual(
            artifact["sourceCheckpointSha256"],
            hashlib.sha256(portable).hexdigest(),
        )
        loaded = torch.load(
            BytesIO(portable),
            map_location="cpu",
            weights_only=True,
        )
        self.assertEqual(loaded["format_version"], 3)

        unexpected = BytesIO(raw)
        with zipfile.ZipFile(unexpected, "a") as archive:
            archive.writestr("archive/future-required-record", b"unsafe")
        with self.assertRaisesRegex(
            RuntimeError,
            "unsupported runtime records",
        ):
            generator._portable_checkpoint_bytes(  # type: ignore[attr-defined]
                unexpected.getvalue()
            )

    def test_known_runtime_metadata_variants_project_identically(self) -> None:
        generator = _load_generator()
        tokenizer = generator._tokenizer()  # type: ignore[attr-defined]
        state = generator._intentional_state(  # type: ignore[attr-defined]
            tokenizer
        )
        payload = generator._checkpoint_payload(  # type: ignore[attr-defined]
            tokenizer,
            state,
            "exact-current-v2",
        )
        portable = generator._checkpoint_bytes(payload)  # type: ignore[attr-defined]

        variants = (
            {
                ".data/serialization_id": b"runtime-specific-id",
            },
            {
                ".data/serialization_id": b"another-runtime-specific-id",
                ".format_version": b"1",
                ".storage_alignment": b"64",
            },
        )
        for runtime_records in variants:
            archive_bytes = BytesIO(portable)
            with zipfile.ZipFile(archive_bytes, "a") as archive:
                for name, value in runtime_records.items():
                    archive.writestr(f"archive/{name}", value)
            self.assertEqual(
                generator._portable_checkpoint_bytes(  # type: ignore[attr-defined]
                    archive_bytes.getvalue()
                ),
                portable,
            )

    def test_masked_artifact_is_losslessly_reconstructable(self) -> None:
        generator = _load_generator()
        fixture = generator.build_fixture()  # type: ignore[attr-defined]
        masked = json.loads(json.dumps(fixture["artifact"]))
        masked.update(fixture["maskedArtifactDelta"])
        digest = generator._sha256(  # type: ignore[attr-defined]
            generator.canonical_artifact_bytes(masked)  # type: ignore[attr-defined]
        )
        self.assertEqual(
            digest,
            fixture["bindings"]["maskedArtifactSha256"],
        )


if __name__ == "__main__":
    unittest.main()
