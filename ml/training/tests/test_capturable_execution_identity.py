from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import unittest
from unittest import mock

import _bootstrap  # noqa: F401

from drawback_ml import capturable_blend
from drawback_ml.capturable_records import CapturableDatasetError


_STATUS_COMMAND = (
    "status",
    "--porcelain=v1",
    "--untracked-files=all",
    "--ignore-submodules=none",
)
_ORIGIN_COMMAND = (
    "config",
    "--local",
    "--no-includes",
    "--get-all",
    "remote.origin.url",
)
_REDIRECTION_COMMAND = (
    "config",
    "--no-includes",
    "--show-scope",
    "--name-only",
    "--get-regexp",
    capturable_blend._LOCAL_CONFIG_REDIRECTION_PATTERN,
)


class _GitStub:
    def __init__(
        self,
        *,
        repository: Path,
        protocol_file: str,
        protocol_text: str,
        protocol_commit: str,
        head_revision: str,
    ) -> None:
        self.repository = repository
        self.protocol_file = protocol_file
        self.protocol_text = protocol_text
        self.protocol_commit = protocol_commit
        self.head_revision = head_revision
        self.calls: list[
            tuple[list[str], dict[str, object]]
        ] = []
        self.overrides: dict[
            tuple[str, ...],
            tuple[int, str] | BaseException,
        ] = {}

    @staticmethod
    def logical_arguments(command: list[str]) -> tuple[str, ...]:
        index = 2
        while index < len(command) and command[index] == "-c":
            index += 2
        return tuple(command[index:])

    def _default_response(
        self,
        arguments: tuple[str, ...],
    ) -> tuple[int, str]:
        if arguments == _STATUS_COMMAND:
            return 0, ""
        if arguments == ("rev-parse", "HEAD"):
            return 0, f"{self.head_revision}\n"
        if arguments == ("rev-parse", "--show-toplevel"):
            return 0, f"{self.repository}\n"
        if arguments == _ORIGIN_COMMAND:
            return 0, (
                f"{capturable_blend._CANONICAL_REPOSITORY_URL}\n"
            )
        if arguments == _REDIRECTION_COMMAND:
            return 1, ""
        if arguments[:2] == ("merge-base", "--is-ancestor"):
            return 0, ""
        if arguments == (
            "ls-remote",
            capturable_blend._CANONICAL_REPOSITORY_URL,
            "refs/heads/main",
        ):
            return 0, (
                f"{self.head_revision}\trefs/heads/main\n"
            )
        if arguments[:1] == ("show",) and len(arguments) == 2:
            return 0, self.protocol_text
        if arguments[:3] == (
            "show",
            "-s",
            "--format=%an%x00%ae%x00%cn%x00%ce",
        ):
            identity = "\x00".join(
                capturable_blend._CANONICAL_COMMIT_IDENTITY
            )
            return 0, f"{identity}\n"
        raise AssertionError(f"unexpected Git command: {arguments!r}")

    def __call__(
        self,
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        copied_kwargs = dict(kwargs)
        self.calls.append((list(command), copied_kwargs))
        arguments = self.logical_arguments(command)
        response = self.overrides.get(arguments)
        if isinstance(response, BaseException):
            raise response
        returncode, stdout = (
            response
            if response is not None
            else self._default_response(arguments)
        )
        if returncode != 0 and kwargs.get("check"):
            raise subprocess.CalledProcessError(
                returncode,
                command,
                output=stdout,
                stderr="failure",
            )
        return subprocess.CompletedProcess(
            command,
            returncode,
            stdout,
            "",
        )


class CapturableExecutionIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repository = Path(
            capturable_blend.__file__
        ).resolve().parents[3]
        self.protocol_file = capturable_blend.PROTOCOL_FILE
        protocol_bytes = (
            self.repository
            / "docs"
            / "research"
            / self.protocol_file
        ).read_bytes()
        self.protocol_text = protocol_bytes.decode("utf-8")
        self.protocol_sha256 = hashlib.sha256(
            protocol_bytes
        ).hexdigest()
        self.protocol_commit = "b" * 40
        self.head_revision = "a" * 40
        self.recorded_revision = "c" * 40

    def _stub(self) -> _GitStub:
        return _GitStub(
            repository=self.repository,
            protocol_file=self.protocol_file,
            protocol_text=self.protocol_text,
            protocol_commit=self.protocol_commit,
            head_revision=self.head_revision,
        )

    def _execution(
        self,
        stub: _GitStub,
        *,
        protocol_sha256: str | None = None,
    ) -> dict[str, object]:
        with mock.patch.object(
            capturable_blend.subprocess,
            "run",
            side_effect=stub,
        ):
            return dict(
                capturable_blend._authenticated_execution_identity(
                    protocol_commit=self.protocol_commit,
                    protocol_file=self.protocol_file,
                    protocol_sha256=(
                        protocol_sha256
                        if protocol_sha256 is not None
                        else self.protocol_sha256
                    ),
                    operation="test execution",
                )
            )

    def _recorded(
        self,
        stub: _GitStub,
        *,
        revision: str | None = None,
    ) -> dict[str, object]:
        with mock.patch.object(
            capturable_blend.subprocess,
            "run",
            side_effect=stub,
        ):
            return dict(
                capturable_blend._authenticated_recorded_revision_identity(
                    revision=(
                        revision
                        if revision is not None
                        else self.recorded_revision
                    ),
                    protocol_commit=self.protocol_commit,
                    protocol_file=self.protocol_file,
                    protocol_sha256=self.protocol_sha256,
                    operation="test report",
                )
            )

    def test_clean_pushed_execution_has_hardened_git_contract(
        self,
    ) -> None:
        stub = self._stub()
        trusted_git = capturable_blend._trusted_git_executable()
        hostile_environment = {
            "ALL_PROXY": "http://hostile.invalid",
            "COMSPEC": "hostile-shell.exe",
            "CURL_CA_BUNDLE": "hostile-ca.pem",
            "GIT_DIR": "redirected",
            "git_work_tree": "redirected-too",
            "GIT_INDEX_FILE": "hostile-index",
            "GIT_OBJECT_DIRECTORY": "hostile-objects",
            "GIT_ALTERNATE_OBJECT_DIRECTORIES": "hostile-alternates",
            "GIT_REPLACE_REF_BASE": "refs/replace/hostile",
            "GIT_CONFIG_SYSTEM": "hostile-system-config",
            "GIT_CONFIG_GLOBAL": "hostile-global-config",
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "url.evil.insteadOf",
            "GIT_CONFIG_VALUE_0": "https://github.com/",
            "GIT_EXEC_PATH": "hostile-exec-path",
            "GIT_TERMINAL_PROMPT": "1",
            "GCM_INTERACTIVE": "Always",
            "HTTPS_PROXY": "http://hostile.invalid",
            "HTTP_PROXY": "http://hostile.invalid",
            "PATH": "hostile-path",
            "SSL_CERT_FILE": "hostile-ca.pem",
            "SSH_ASKPASS": "hostile-askpass",
            "SSH_ASKPASS_REQUIRE": "force",
            "SystemRoot": "C:\\hostile-windows",
        }
        with mock.patch.dict(
            capturable_blend.os.environ,
            hostile_environment,
            clear=False,
        ):
            measured = self._execution(stub)

        self.assertEqual(
            measured,
            {
                "cleanWorktree": True,
                "repository": "DrawbackGuesser",
                "revision": self.head_revision,
            },
        )
        logical_calls = [
            stub.logical_arguments(command)
            for command, _kwargs in stub.calls
        ]
        self.assertIn(_STATUS_COMMAND, logical_calls)
        self.assertIn(_ORIGIN_COMMAND, logical_calls)
        self.assertIn(_REDIRECTION_COMMAND, logical_calls)
        self.assertIn(
            (
                "ls-remote",
                capturable_blend._CANONICAL_REPOSITORY_URL,
                "refs/heads/main",
            ),
            logical_calls,
        )
        self.assertFalse(
            any(call[:2] == ("remote", "get-url") for call in logical_calls)
        )

        allowed_git_variables = {
            "GIT_CONFIG_COUNT",
            "GIT_CONFIG_GLOBAL",
            "GIT_CONFIG_NOSYSTEM",
            "GIT_OPTIONAL_LOCKS",
            "GIT_PAGER",
            "GIT_TERMINAL_PROMPT",
        }
        for command, kwargs in stub.calls:
            self.assertEqual(
                command[0:2],
                [str(trusted_git), "--no-replace-objects"],
            )
            environment = kwargs["env"]
            self.assertIsInstance(environment, dict)
            git_variables = {
                key.upper()
                for key in environment
                if key.upper().startswith("GIT_")
            }
            self.assertEqual(git_variables, allowed_git_variables)
            self.assertEqual(
                environment["GIT_CONFIG_GLOBAL"],
                os.devnull,
            )
            self.assertEqual(environment["GIT_CONFIG_NOSYSTEM"], "1")
            self.assertEqual(environment["GIT_CONFIG_COUNT"], "0")
            self.assertEqual(environment["GIT_TERMINAL_PROMPT"], "0")
            self.assertEqual(environment["GCM_INTERACTIVE"], "Never")
            self.assertNotIn("SSH_ASKPASS", environment)
            self.assertNotIn("SSH_ASKPASS_REQUIRE", environment)
            self.assertNotIn("ALL_PROXY", environment)
            self.assertNotIn("CURL_CA_BUNDLE", environment)
            self.assertNotIn("HTTPS_PROXY", environment)
            self.assertNotIn("HTTP_PROXY", environment)
            self.assertNotIn("SSL_CERT_FILE", environment)
            self.assertNotEqual(environment["PATH"], "hostile-path")
            if os.name == "nt":
                self.assertNotEqual(
                    environment["SystemRoot"],
                    "C:\\hostile-windows",
                )
                self.assertNotEqual(
                    environment["ComSpec"],
                    "hostile-shell.exe",
                )
            self.assertEqual(kwargs["cwd"], self.repository)
            self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)
            self.assertEqual(kwargs["timeout"], 30)
            self.assertTrue(kwargs["capture_output"])
            self.assertTrue(kwargs["text"])
            self.assertEqual(kwargs["encoding"], "utf-8")
            self.assertEqual(kwargs["errors"], "strict")

    def test_dirty_untracked_and_dirty_submodule_are_rejected(
        self,
    ) -> None:
        for status in (
            " M tracked.py\n",
            "?? untracked.txt\n",
            " m engine\n",
        ):
            with self.subTest(status=status):
                stub = self._stub()
                stub.overrides[_STATUS_COMMAND] = (0, status)
                with self.assertRaisesRegex(
                    CapturableDatasetError,
                    "clean committed worktree",
                ):
                    self._execution(stub)

    def test_repository_identity_mismatches_are_rejected(self) -> None:
        cases: list[
            tuple[str, tuple[str, ...], tuple[int, str]]
        ] = [
            (
                "root",
                ("rev-parse", "--show-toplevel"),
                (0, f"{self.repository.parent}\n"),
            ),
            (
                "protocol blob",
                (
                    "show",
                    (
                        f"{self.protocol_commit}:docs/research/"
                        f"{self.protocol_file}"
                    ),
                ),
                (0, "not the frozen protocol"),
            ),
            (
                "origin",
                _ORIGIN_COMMAND,
                (0, "https://github.com/attacker/repository.git\n"),
            ),
            (
                "remote head",
                (
                    "ls-remote",
                    capturable_blend._CANONICAL_REPOSITORY_URL,
                    "refs/heads/main",
                ),
                (0, f"{'d' * 40}\trefs/heads/main\n"),
            ),
            (
                "identity",
                (
                    "show",
                    "-s",
                    "--format=%an%x00%ae%x00%cn%x00%ce",
                    self.head_revision,
                ),
                (0, "Mallory\x00mallory@example.test\x00Mallory"
                    "\x00mallory@example.test\n"),
            ),
        ]
        for label, command, response in cases:
            with self.subTest(case=label):
                stub = self._stub()
                stub.overrides[command] = response
                with self.assertRaisesRegex(
                    CapturableDatasetError,
                    "repository identity is not the pushed release",
                ):
                    self._execution(stub)

    def test_working_protocol_mismatch_is_rejected(self) -> None:
        stub = self._stub()
        with self.assertRaisesRegex(
            CapturableDatasetError,
            "protocol bytes have changed",
        ):
            self._execution(stub, protocol_sha256="0" * 64)

    def test_local_include_or_url_redirection_is_rejected(self) -> None:
        for redirection in (
            "local\tinclude.path\n",
            "local\turl.https://evil.example/.insteadof\n",
            "local\thttp.https://github.com/.sslverify\n",
            "local\tcredential.helper\n",
            "worktree\tcore.excludesfile\n",
            "worktree\tfilter.hostile.process\n",
        ):
            with self.subTest(redirection=redirection):
                stub = self._stub()
                stub.overrides[_REDIRECTION_COMMAND] = (
                    0,
                    redirection,
                )
                with self.assertRaisesRegex(
                    CapturableDatasetError,
                    "repository identity is not the pushed release",
                ):
                    self._execution(stub)
                logical_calls = [
                    stub.logical_arguments(command)
                    for command, _ in stub.calls
                ]
                self.assertNotIn(_STATUS_COMMAND, logical_calls)

    def test_subprocess_timeout_and_failures_fail_closed(self) -> None:
        failures: tuple[BaseException, ...] = (
            subprocess.TimeoutExpired(["git", "status"], 30),
            subprocess.CalledProcessError(2, ["git", "status"]),
            OSError("git is unavailable"),
        )
        for failure in failures:
            with self.subTest(failure=type(failure).__name__):
                stub = self._stub()
                stub.overrides[_STATUS_COMMAND] = failure
                with self.assertRaisesRegex(
                    CapturableDatasetError,
                    "execution identity cannot be verified",
                ):
                    self._execution(stub)

    def test_recorded_revision_is_authenticated_without_clean_head(
        self,
    ) -> None:
        stub = self._stub()
        measured = self._recorded(stub)

        self.assertEqual(
            measured,
            {
                "repository": "DrawbackGuesser",
                "revision": self.recorded_revision,
                "pushedMainRevision": self.head_revision,
            },
        )
        logical_calls = [
            stub.logical_arguments(command)
            for command, _kwargs in stub.calls
        ]
        self.assertNotIn(_STATUS_COMMAND, logical_calls)
        self.assertIn(
            (
                "merge-base",
                "--is-ancestor",
                self.recorded_revision,
                self.head_revision,
            ),
            logical_calls,
        )
        self.assertIn(
            (
                "show",
                (
                    f"{self.recorded_revision}:docs/research/"
                    f"{self.protocol_file}"
                ),
            ),
            logical_calls,
        )

    def test_recorded_revision_fails_closed_on_proof_mismatch(
        self,
    ) -> None:
        cases: list[
            tuple[str, tuple[str, ...], tuple[int, str]]
        ] = [
            (
                "not reachable",
                (
                    "merge-base",
                    "--is-ancestor",
                    self.recorded_revision,
                    self.head_revision,
                ),
                (1, ""),
            ),
            (
                "predates protocol",
                (
                    "merge-base",
                    "--is-ancestor",
                    self.protocol_commit,
                    self.recorded_revision,
                ),
                (1, ""),
            ),
            (
                "wrong protocol",
                (
                    "show",
                    (
                        f"{self.recorded_revision}:docs/research/"
                        f"{self.protocol_file}"
                    ),
                ),
                (0, "wrong protocol"),
            ),
            (
                "wrong identity",
                (
                    "show",
                    "-s",
                    "--format=%an%x00%ae%x00%cn%x00%ce",
                    self.recorded_revision,
                ),
                (0, "Mallory\x00bad@example.test\x00Mallory"
                    "\x00bad@example.test\n"),
            ),
        ]
        for label, command, response in cases:
            with self.subTest(case=label):
                stub = self._stub()
                stub.overrides[command] = response
                with self.assertRaisesRegex(
                    CapturableDatasetError,
                    "recorded revision is not on the pushed release",
                ):
                    self._recorded(stub)

    def test_noncanonical_recorded_revision_does_not_invoke_git(
        self,
    ) -> None:
        stub = self._stub()
        with self.assertRaisesRegex(
            CapturableDatasetError,
            "recorded revision is not canonical",
        ):
            self._recorded(stub, revision="HEAD")
        self.assertEqual(stub.calls, [])


if __name__ == "__main__":
    unittest.main()
