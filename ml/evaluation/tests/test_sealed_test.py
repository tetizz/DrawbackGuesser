from __future__ import annotations

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ml.evaluation.ensemble_calibration import ContentAddressedFile
from ml.evaluation.review_authorization import (
    ReviewAuthorizationError,
    authorize_review,
)
from ml.evaluation.sealed_test import (
    _pinned_sealed_inputs,
    _source_identity,
    execute_authorized_test,
)
from ml.evaluation.tests.test_review_authorization import fixture


class SealedTestBoundaryTests(unittest.TestCase):
    def _authorization(self, root: Path) -> tuple[ContentAddressedFile, dict[str, object]]:
        approval, allowed, reviewers, value = fixture(root)
        (root / "ml").mkdir()
        (root / "ml" / "requirements.txt").write_bytes(
            (root / "requirements.txt").read_bytes()
        )
        (root / "ml" / "pyproject.toml").write_bytes(
            (root / "pyproject.toml").read_bytes()
        )
        output = root / "authorization.json"
        authorize_review(
            approval_path=approval,
            approval_sha256=hashlib.sha256(approval.read_bytes()).hexdigest(),
            allowed_signers_path=allowed,
            allowed_signers_sha256=hashlib.sha256(allowed.read_bytes()).hexdigest(),
            reviewers=reviewers,
            output=output,
        )
        return (
            ContentAddressedFile(
                output, hashlib.sha256(output.read_bytes()).hexdigest()
            ),
            value,
        )

    @contextmanager
    def _cwd(self, path: Path):
        previous = Path.cwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(previous)

    def test_authorization_failure_occurs_before_any_sealed_read(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            reference, value = self._authorization(root)
            bad = ContentAddressedFile(reference.path, "00" * 32)
            with self._cwd(root), patch(
                "ml.evaluation.sealed_test._digest"
            ) as digest:
                with self.assertRaises(ReviewAuthorizationError):
                    execute_authorized_test(
                        authorization=bad,
                        invocation=value["test_plan"]["argv"],  # type: ignore[index]
                        directory=root,
                    )
            digest.assert_not_called()

    def test_exact_invocation_is_required_before_source_or_sealed_paths(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            reference, value = self._authorization(root)
            alternate = [*value["test_plan"]["argv"], "--batch-size", "1"]  # type: ignore[index]
            with self._cwd(root), patch(
                "ml.evaluation.sealed_test._source_identity"
            ) as process, patch("ml.evaluation.sealed_test._digest") as digest:
                with self.assertRaisesRegex(ReviewAuthorizationError, "exact signed"):
                    execute_authorized_test(
                        authorization=reference,
                        invocation=alternate,
                        directory=root,
                    )
            process.assert_not_called()
            digest.assert_not_called()

    def test_output_no_clobber_precedes_sealed_read(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            reference, value = self._authorization(root)
            (root / "sealed-report.json").write_text("existing", encoding="utf-8")
            with self._cwd(root), patch(
                "ml.evaluation.sealed_test._source_identity",
                return_value=(root, value["dependencies"]["source_revision"]),  # type: ignore[index]
            ), patch(
                "ml.evaluation.sealed_test._pinned_sealed_inputs"
            ) as sealed:
                with self.assertRaisesRegex(ReviewAuthorizationError, "already exist"):
                    execute_authorized_test(
                        authorization=reference,
                        invocation=value["test_plan"]["argv"],  # type: ignore[index]
                        directory=root,
                    )
            sealed.assert_not_called()

    def test_consumed_authorization_cannot_retry_after_failure(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            reference, value = self._authorization(root)
            for name in ("release.public.json", "test.private.json", "test.ndjson"):
                (root / name).write_bytes(b"mutated sealed input")
            invocation = value["test_plan"]["argv"]  # type: ignore[index]
            with self._cwd(root), patch(
                "ml.evaluation.sealed_test._source_identity",
                return_value=(root, value["dependencies"]["source_revision"]),  # type: ignore[index]
            ):
                with self.assertRaisesRegex(ReviewAuthorizationError, "SHA-256"):
                    execute_authorized_test(
                        authorization=reference,
                        invocation=invocation,
                        directory=root,
                    )
                with self.assertRaisesRegex(ReviewAuthorizationError, "refusing to reuse"):
                    execute_authorized_test(
                        authorization=reference,
                        invocation=invocation,
                        directory=root,
                    )

    def test_authorized_evaluation_uses_the_pinned_engine_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            reference, value = self._authorization(root)
            source_revision = value["dependencies"][  # type: ignore[index]
                "source_revision"
            ]
            invocation = value["test_plan"]["argv"]  # type: ignore[index]

            @contextmanager
            def pinned_inputs(*_args: object):
                yield {
                    "public_root": (root / "release.public.json", "22" * 32),
                    "private_test": (root / "test.private.json", "33" * 32),
                    "dataset": (root / "test.ndjson", "44" * 32),
                }

            @contextmanager
            def audited_split(*_args: object):
                yield object()

            report = SimpleNamespace(
                ensemble_release_sha256="50" * 32,
                calibration_sha256="51" * 32,
                training_frequency_sha256="52" * 32,
                partition_seed_sha256="53" * 32,
                transcript=SimpleNamespace(sha256="54" * 32, record_count=1),
                bootstrap_seed=20260831,
                move_examples=1,
            )
            with self._cwd(root), patch(
                "ml.evaluation.sealed_test._source_identity",
                return_value=(root, source_revision),
            ), patch(
                "ml.evaluation.sealed_test._pinned_sealed_inputs",
                side_effect=pinned_inputs,
            ), patch(
                "ml.evaluation.sealed_test.open_audited_private_corpus_split",
                side_effect=audited_split,
            ), patch(
                "ml.evaluation.sealed_test.evaluate_candidate_partition",
                return_value=report,
            ) as evaluate, patch(
                "ml.evaluation.sealed_test._test_decision",
                return_value=(True, []),
            ), patch(
                "ml.evaluation.sealed_test._candidate_metrics",
                return_value={},
            ):
                execute_authorized_test(
                    authorization=reference,
                    invocation=invocation,
                    directory=root,
                )

            self.assertEqual(
                evaluate.call_args.kwargs["catalogs"],
                (
                    root
                    / "engine"
                    / "data"
                    / "catalog"
                    / "observed-drawbacks.json",
                ),
            )

    def test_nested_ignored_evidence_resolves_clean_repository_root(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            subprocess.run(("git", "init", "-q"), cwd=root, check=True)
            subprocess.run(
                ("git", "config", "user.name", "fixture"), cwd=root, check=True
            )
            subprocess.run(
                ("git", "config", "user.email", "fixture@example.invalid"),
                cwd=root,
                check=True,
            )
            (root / ".gitignore").write_text("evidence/\n", encoding="utf-8")
            (root / "tracked.txt").write_text("bound source\n", encoding="utf-8")
            subprocess.run(("git", "add", "."), cwd=root, check=True)
            subprocess.run(
                ("git", "commit", "-q", "-m", "fixture"), cwd=root, check=True
            )
            evidence = root / "evidence" / "nested"
            evidence.mkdir(parents=True)
            (evidence / "authorization.json").write_text(
                "ignored evidence\n", encoding="utf-8"
            )
            resolved_root, revision = _source_identity(evidence)
            expected = subprocess.run(
                ("git", "rev-parse", "HEAD"),
                cwd=root,
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            ).stdout.strip()
            self.assertEqual(resolved_root, root.resolve())
            self.assertEqual(revision, expected)

    def test_sealed_inputs_are_streamed_without_read_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            inputs: dict[str, object] = {}
            for key, name in (
                ("public_root", "release.public.json"),
                ("private_test", "test.private.json"),
                ("dataset", "test.ndjson"),
            ):
                payload = (key.encode("ascii") + b"\n") * 4096
                (root / name).write_bytes(payload)
                inputs[key] = {
                    "file": name,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            with patch.object(
                Path, "read_bytes", side_effect=AssertionError("unbounded read")
            ):
                with _pinned_sealed_inputs(root, inputs) as pinned:
                    for path, digest in pinned.values():
                        with path.open("rb") as source:
                            self.assertEqual(
                                hashlib.sha256(source.read()).hexdigest(), digest
                            )


if __name__ == "__main__":
    unittest.main()
