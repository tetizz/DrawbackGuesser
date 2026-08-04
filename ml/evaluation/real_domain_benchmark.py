"""Offline benchmark for consented, completed DrawbackChess games.

The benchmark deliberately has no live-game input.  Label-blind Stage A
authenticates and replays a canonical PGN corpus before invoking an approved
post-game analyzer.  Label-aware Stage B scores the resulting content-addressed
prediction bundle without invoking inference.  The revealed-label artifact
must be absent from the Stage A access domain.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import threading
import time
from types import MappingProxyType
from typing import Iterator, Mapping, Protocol, Sequence
import unicodedata

import chess
import chess.pgn

from .metrics import PredictionExample, evaluate
from ml.training.drawback_ml.training_corpus_set import (
    TrainingCorpusSetError,
    verify_training_corpus_set,
)
from ml.training.drawback_ml.durable_publish import publish_bytes_durable_exact


CORPUS_FORMAT = "drawbacktrainer-real-domain-completed-pgn-corpus"
LABEL_FORMAT = "drawbacktrainer-real-domain-revealed-labels"
ANALYSIS_FORMAT = "drawbacktrainer-approved-postgame-analysis"
PREDICTION_BUNDLE_FORMAT = "drawbacktrainer-real-domain-prediction-bundle"
REPORT_FORMAT = "drawbacktrainer-real-domain-benchmark-report"
VERSION = 1
WINDOWS_JOB_BOOTSTRAP = """\
import os
import runpy
import sys
import time
import traceback
from pathlib import Path

gate = Path(sys.argv[1])
launcher = sys.argv[2]
finished = Path(sys.argv[3])
deadline = time.monotonic() + 30.0
while not gate.is_file():
    if time.monotonic() >= deadline:
        raise SystemExit(124)
    time.sleep(0.005)
status = 0
try:
    sys.argv = [launcher]
    runpy.run_path(launcher, run_name="__main__")
except BaseException:
    traceback.print_exc()
    status = 1
temporary = finished.with_suffix(".tmp")
temporary.write_bytes(f"{status}\\n".encode("ascii"))
os.replace(temporary, finished)
# Keep the assigned root alive so the controller can always terminate the
# complete Windows process tree by its still-existing root PID.
deadline = time.monotonic() + 30.0
while time.monotonic() < deadline:
    time.sleep(0.05)
raise SystemExit(125)
"""
HORIZONS = (5, 10, 15, 20)
TERMINAL_RESULTS = frozenset({"1-0", "0-1", "1/2-1/2"})
ENSEMBLE_MODE = "hybrid-v21-ensemble"
STANDARD_PGN_UNAVAILABLE_IDS = frozenset(
    {"hand-and-gigabrain", "ichtyophobe"}
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
RELEASE_CLAIM_MINIMUM_GAMES = 2_000
RELEASE_CLAIM_MINIMUM_PLAYER_GAMES_PER_RULE = 10
PROCESS_TERMINATION_GRACE_SECONDS = 5.0


class RealDomainBenchmarkError(ValueError):
    """Raised when benchmark evidence is invalid or incomplete."""


@dataclass(frozen=True)
class ContentAddressedJson:
    path: Path
    sha256: str


@dataclass(frozen=True)
class BenchmarkClaimConfig:
    """Select research reporting or an externally publishable release claim."""

    mode: str = "research"
    claimed_drawback_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in {"research", "release-claim"}:
            raise ValueError("benchmark claim mode must be research or release-claim")
        if len(set(self.claimed_drawback_ids)) != len(self.claimed_drawback_ids):
            raise ValueError("claimed drawback IDs must be unique")
        if any(not drawback_id for drawback_id in self.claimed_drawback_ids):
            raise ValueError("claimed drawback IDs must not be empty")
        if self.mode == "release-claim" and not self.claimed_drawback_ids:
            raise ValueError(
                "release-claim mode requires explicit claimed drawback IDs"
            )


@dataclass(frozen=True)
class CanonicalGame:
    pgn_sha256: str
    semantic_sha256: str
    safe_pgn: str
    ply_count: int


@dataclass(frozen=True)
class CatalogTitle:
    drawback_id: str
    supported: bool


@dataclass(frozen=True)
class RevealedSide:
    drawback_id: str
    parameters: Mapping[str, object]


@dataclass(frozen=True)
class RevealedGame:
    white: RevealedSide
    black: RevealedSide


@dataclass(frozen=True)
class AnalysisSnapshot:
    ply: int
    white: Mapping[str, float]
    black: Mapping[str, float]


@dataclass(frozen=True)
class ApprovedAnalysis:
    pgn_sha256: str
    ply_count: int
    class_ids: tuple[str, ...]
    unavailable_supported_ids: tuple[str, ...]
    snapshots: tuple[AnalysisSnapshot, ...]
    predictor_identity: Mapping[str, object]


class ApprovedPostGameAnalyzer(Protocol):
    """Production implementations invoke the approved browser Worker."""

    def analyze_completed_game(
        self,
        *,
        pgn_sha256: str,
        pgn: str,
    ) -> Mapping[str, object]: ...


@dataclass(frozen=True)
class ApprovedSubprocessAnalyzer:
    """Content-addressed adapter for the production browser-Worker launcher.

    The launcher receives private, authenticated input/output paths through a
    closed environment and writes one canonical ``ANALYSIS_FORMAT`` object.
    Standard output is drained under a strict byte bound. Runtime, launcher,
    dependencies, and predictor artifacts are pinned and reauthenticated for
    every game.
    """

    runtime: Path
    runtime_sha256: str
    launcher: Path
    launcher_sha256: str
    browser_artifact: ContentAddressedJson
    ensemble_release: ContentAddressedJson
    calibration: ContentAddressedJson
    approval_evidence: ContentAddressedJson
    runtime_dependencies: tuple[ContentAddressedJson, ...] = ()
    timeout_seconds: int = 300
    maximum_output_bytes: int = 64 * 1024 * 1024

    def __post_init__(self) -> None:
        for label, path, digest in (
            ("runtime", self.runtime, self.runtime_sha256),
            ("launcher", self.launcher, self.launcher_sha256),
        ):
            if not path.is_file():
                raise RealDomainBenchmarkError(
                    f"approved analyzer {label} does not exist"
                )
            if (
                not SHA256_PATTERN.fullmatch(digest)
                or _sha256(path.read_bytes()) != digest
            ):
                raise RealDomainBenchmarkError(
                    f"approved analyzer {label} SHA-256 does not match"
                )
        if self.timeout_seconds <= 0 or self.maximum_output_bytes <= 0:
            raise RealDomainBenchmarkError(
                "approved analyzer bounds must be positive"
            )
        for reference, label in (
            (self.browser_artifact, "browser artifact"),
            (self.ensemble_release, "ensemble release"),
            (self.calibration, "calibration"),
            (self.approval_evidence, "approval evidence"),
        ):
            _authenticate_bytes(reference, label)
        dependency_names: set[str] = set()
        for dependency in self.runtime_dependencies:
            _authenticate_bytes(dependency, "runtime dependency")
            if dependency.path.name in dependency_names:
                raise RealDomainBenchmarkError(
                    "approved analyzer runtime dependency names collide"
                )
            dependency_names.add(dependency.path.name)

    def analyze_completed_game(
        self,
        *,
        pgn_sha256: str,
        pgn: str,
    ) -> Mapping[str, object]:
        # Reauthenticate mutable paths immediately before every process.
        runtime_payload = self.runtime.read_bytes()
        if _sha256(runtime_payload) != self.runtime_sha256:
            raise RealDomainBenchmarkError(
                "approved analyzer runtime changed before execution"
            )
        launcher_payload = self.launcher.read_bytes()
        if _sha256(launcher_payload) != self.launcher_sha256:
            raise RealDomainBenchmarkError(
                "approved analyzer launcher changed before execution"
            )
        artifacts = (
            (self.browser_artifact, "browser-model.json"),
            (self.ensemble_release, "ensemble-release.json"),
            (self.calibration, "calibration.json"),
            (self.approval_evidence, "approval-evidence.json"),
        )
        artifact_payloads = tuple(
            (
                reference,
                name,
                _authenticate_bytes(reference, label),
            )
            for (reference, name), label in zip(
                artifacts,
                (
                    "browser artifact",
                    "ensemble release",
                    "calibration",
                    "approval evidence",
                ),
                strict=True,
            )
        )
        dependency_payloads = tuple(
            (
                dependency,
                _authenticate_bytes(dependency, "runtime dependency"),
            )
            for dependency in self.runtime_dependencies
        )
        with _temporary_analysis_directory() as directory:
            runtime = directory / self.runtime.name
            launcher = directory / self.launcher.name
            runtime.write_bytes(runtime_payload)
            runtime.chmod(self.runtime.stat().st_mode)
            launcher.write_bytes(launcher_payload)
            launcher.chmod(self.launcher.stat().st_mode)
            for dependency, payload in dependency_payloads:
                destination = directory / dependency.path.name
                destination.write_bytes(payload)
                if _sha256(destination.read_bytes()) != dependency.sha256:
                    raise RealDomainBenchmarkError(
                        "approved analyzer staged runtime dependency "
                        "authentication failed"
                    )
            if _sha256(runtime.read_bytes()) != self.runtime_sha256:
                raise RealDomainBenchmarkError(
                    "approved analyzer runtime copy failed authentication"
                )
            if _sha256(launcher.read_bytes()) != self.launcher_sha256:
                raise RealDomainBenchmarkError(
                    "approved analyzer launcher copy failed authentication"
                )
            copied: dict[str, Path] = {}
            for reference, name, payload in artifact_payloads:
                path = directory / name
                path.write_bytes(payload)
                if _sha256(path.read_bytes()) != reference.sha256:
                    raise RealDomainBenchmarkError(
                        "approved analyzer staged artifact authentication failed"
                    )
                copied[name] = path
            input_path = directory / "completed.pgn"
            output_path = directory / "analysis.json"
            input_path.write_text(pgn, encoding="utf-8", newline="\n")
            command = [str(runtime), str(launcher)]
            start_gate: Path | None = None
            launcher_finished: Path | None = None
            if os.name == "nt":
                # Popen starts executing before AssignProcessToJobObject can run.
                # Gate the approved launcher inside a tiny bootstrap so it cannot
                # create an uncontained descendant during that assignment race.
                bootstrap = directory / "windows-job-bootstrap.py"
                bootstrap.write_text(
                    WINDOWS_JOB_BOOTSTRAP,
                    encoding="utf-8",
                    newline="\n",
                )
                start_gate = directory / "windows-job-assigned"
                launcher_finished = directory / "windows-launcher-finished"
                command = [
                    str(runtime),
                    str(bootstrap),
                    str(start_gate),
                    str(launcher),
                    str(launcher_finished),
                ]
            environment = {
                **_trusted_windows_runtime_environment(),
                "TEMP": str(directory),
                "TMP": str(directory),
                "PYTHONHASHSEED": "0",
                "PYTHONNOUSERSITE": "1",
                "DRAWBACKTRAINER_POSTGAME_ONLY": "1",
                "DRAWBACKTRAINER_EXPECTED_SANITIZED_PGN_SHA256": pgn_sha256,
                "DRAWBACKTRAINER_PGN_INPUT": str(input_path),
                "DRAWBACKTRAINER_ANALYSIS_OUTPUT": str(output_path),
                "DRAWBACKTRAINER_BROWSER_ARTIFACT": str(
                    copied["browser-model.json"]
                ),
                "DRAWBACKTRAINER_ENSEMBLE_RELEASE": str(
                    copied["ensemble-release.json"]
                ),
                "DRAWBACKTRAINER_CALIBRATION": str(
                    copied["calibration.json"]
                ),
                "DRAWBACKTRAINER_APPROVAL_EVIDENCE": str(
                    copied["approval-evidence.json"]
                ),
            }
            process = subprocess.Popen(
                command,
                cwd=directory,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=environment,
                start_new_session=os.name != "nt",
                creationflags=(
                    subprocess.CREATE_NEW_PROCESS_GROUP
                    if os.name == "nt"
                    else 0
                ),
            )
            kill_job = _WindowsKillJob(process)
            if start_gate is not None:
                start_gate.write_bytes(b"assigned\n")
            stdout_size = 0
            stdout_oversized = threading.Event()

            def drain_stdout() -> None:
                nonlocal stdout_size
                assert process.stdout is not None
                try:
                    while chunk := process.stdout.read1(64 * 1024):
                        stdout_size += len(chunk)
                        if stdout_size > self.maximum_output_bytes:
                            stdout_oversized.set()
                            return
                finally:
                    process.stdout.close()

            reader = threading.Thread(target=drain_stdout, daemon=True)
            reader.start()
            deadline = time.monotonic() + self.timeout_seconds
            launcher_returncode: int | None = None
            tree_cleanup_attempted = False
            try:
                while process.poll() is None:
                    if time.monotonic() >= deadline:
                        raise RealDomainBenchmarkError(
                            "approved post-game analyzer timed out"
                        )
                    if (
                        stdout_oversized.is_set()
                        or (
                            output_path.exists()
                            and output_path.stat().st_size
                            > self.maximum_output_bytes
                        )
                    ):
                        raise RealDomainBenchmarkError(
                            "approved analyzer output exceeds its bound"
                        )
                    if (
                        launcher_finished is not None
                        and launcher_finished.is_file()
                    ):
                        status = launcher_finished.read_bytes()
                        if status not in {b"0\n", b"1\n"}:
                            raise RealDomainBenchmarkError(
                                "approved analyzer launcher status is invalid"
                            )
                        launcher_returncode = int(status.strip())
                        break
                    time.sleep(0.01)
                # A launcher can exit while one of its descendants still holds
                # the inherited stdout pipe. Terminate the secured process tree
                # before waiting for that pipe. On Windows the live bootstrap
                # root lets taskkill enumerate descendants, while the kill job
                # remains the fail-closed fallback.
                tree_cleanup_attempted = True
                _kill_process_tree(process, kill_job)
                reader.join(timeout=1)
                if reader.is_alive():
                    raise RealDomainBenchmarkError(
                        "approved analyzer stdout did not close"
                    )
                if stdout_oversized.is_set():
                    raise RealDomainBenchmarkError(
                        "approved analyzer output exceeds its bound"
                    )
                effective_returncode = (
                    launcher_returncode
                    if launcher_returncode is not None
                    else process.returncode
                )
                if effective_returncode != 0:
                    raise RealDomainBenchmarkError(
                        "approved post-game analyzer process failed"
                    )
                if (
                    not output_path.is_file()
                    or output_path.stat().st_size
                    > self.maximum_output_bytes
                ):
                    raise RealDomainBenchmarkError(
                        "approved analyzer output is missing or exceeds its bound"
                    )
                output_payload = output_path.read_bytes()
            except BaseException as primary_error:
                cleanup_errors: list[BaseException] = []
                if not tree_cleanup_attempted:
                    tree_cleanup_attempted = True
                    try:
                        _kill_process_tree(process, kill_job)
                    except BaseException as cleanup_error:
                        cleanup_errors.append(cleanup_error)
                reader.join(timeout=1)
                if reader.is_alive():
                    cleanup_errors.append(
                        RealDomainBenchmarkError(
                            "approved analyzer stdout remained open after cleanup"
                        )
                    )
                if cleanup_errors:
                    _raise_primary_with_cleanup(
                        primary_error,
                        cleanup_errors,
                    )
                raise
        value = _strict_json(output_payload, "approved analyzer output")
        if value.get("pgnSha256") != pgn_sha256:
            raise RealDomainBenchmarkError(
                "approved analyzer output names a different sanitized PGN"
            )
        expected_identity = {
            "mode": ENSEMBLE_MODE,
            "browserArtifactSha256": self.browser_artifact.sha256,
            "ensembleReleaseSha256": self.ensemble_release.sha256,
            "calibrationSha256": self.calibration.sha256,
            "approvalEvidenceSha256": self.approval_evidence.sha256,
        }
        if value.get("predictor") != expected_identity:
            raise RealDomainBenchmarkError(
                "approved analyzer output predictor identity changed"
            )
        return value


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _stable_file_identity(path: Path) -> tuple[int, int, int, int, int, str]:
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != after_identity:
        raise OSError("executable identity changed while it was authenticated")
    return (*after_identity, _sha256(payload))


def _windows_system_directory() -> Path:
    import ctypes
    from ctypes import wintypes

    buffer = ctypes.create_unicode_buffer(32_768)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_system_directory = kernel32.GetSystemDirectoryW
    get_system_directory.argtypes = (wintypes.LPWSTR, wintypes.UINT)
    get_system_directory.restype = wintypes.UINT
    length = get_system_directory(buffer, len(buffer))
    if length == 0 or length >= len(buffer):
        raise OSError("Windows system directory is unavailable")
    unresolved = Path(buffer.value)
    if not unresolved.is_absolute() or unresolved.is_symlink():
        raise OSError("Windows system directory identity is invalid")
    directory = unresolved.resolve(strict=True)
    if not directory.is_dir():
        raise OSError("Windows system directory is unavailable")
    return directory


def _trusted_windows_runtime_environment() -> Mapping[str, str]:
    if os.name != "nt":
        return {}
    system_directory = _windows_system_directory()
    windows_directory = system_directory.parent.resolve(strict=True)
    command_processor = (system_directory / "cmd.exe").resolve(strict=True)
    if (
        command_processor.parent != system_directory
        or not command_processor.is_file()
    ):
        raise OSError(
            "Windows command processor is outside the system directory"
        )
    return {
        "SystemRoot": str(windows_directory),
        "WINDIR": str(windows_directory),
        "ComSpec": str(command_processor),
        "PATH": str(system_directory),
        "PATHEXT": ".COM;.EXE;.BAT;.CMD",
    }


def _windows_taskkill(process_id: int) -> None:
    system_directory = _windows_system_directory()
    unresolved = system_directory / "taskkill.exe"
    if unresolved.is_symlink():
        raise OSError("taskkill must not be a symbolic link")
    taskkill = unresolved.resolve(strict=True)
    if taskkill.parent != system_directory or not taskkill.is_file():
        raise OSError("taskkill resolved outside the Windows system directory")
    taskkill_identity = _stable_file_identity(taskkill)
    completed: subprocess.CompletedProcess[bytes] | None = None
    invocation_error: BaseException | None = None
    try:
        completed = subprocess.run(
            [str(taskkill), "/PID", str(process_id), "/T", "/F"],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=PROCESS_TERMINATION_GRACE_SECONDS,
            env={
                "SystemRoot": str(system_directory.parent),
                "WINDIR": str(system_directory.parent),
            },
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except BaseException as error:
        invocation_error = error
    identity_error: BaseException | None = None
    try:
        if _stable_file_identity(taskkill) != taskkill_identity:
            raise OSError(
                "taskkill identity changed during process-tree cleanup"
            )
    except OSError as error:
        identity_error = error
    if identity_error is not None:
        if invocation_error is not None:
            identity_error.add_note(
                "taskkill invocation also failed: "
                f"{type(invocation_error).__name__}: {invocation_error}"
            )
        raise identity_error from invocation_error
    if invocation_error is not None:
        raise invocation_error
    assert completed is not None
    if completed.returncode != 0:
        raise OSError(
            f"taskkill failed with exit code {completed.returncode}"
        )


def _raise_primary_with_cleanup(
    primary_error: BaseException,
    cleanup_errors: Sequence[BaseException],
) -> None:
    cleanup_failure = RealDomainBenchmarkError(
        "approved analyzer cleanup also failed: "
        + "; ".join(
            f"{type(error).__name__}: {error}"
            for error in cleanup_errors
        )
    )
    primary_error.add_note(str(cleanup_failure))
    raise primary_error from cleanup_failure


def _kill_process_tree(
    process: subprocess.Popen[bytes],
    kill_job: _WindowsKillJob | None = None,
) -> None:
    cleanup_errors: list[BaseException] = []
    if os.name == "nt":
        if process.poll() is None:
            try:
                _windows_taskkill(process.pid)
            except BaseException as error:
                cleanup_errors.append(error)
        # Closing the kill-on-close job must happen before waiting for the
        # direct child. It is the trusted fallback that terminates descendants
        # when taskkill fails or times out.
        if kill_job is not None:
            try:
                kill_job.close()
            except BaseException as error:
                cleanup_errors.append(error)
    else:
        import signal

        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass
        if kill_job is not None:
            try:
                kill_job.close()
            except BaseException as error:
                cleanup_errors.append(error)
    try:
        process.wait(timeout=PROCESS_TERMINATION_GRACE_SECONDS)
    except subprocess.TimeoutExpired as error:
        try:
            process.kill()
        except BaseException as kill_error:
            if process.poll() is None:
                cleanup_errors.append(kill_error)
        try:
            process.wait(timeout=PROCESS_TERMINATION_GRACE_SECONDS)
        except subprocess.TimeoutExpired as final_error:
            cleanup_errors.extend((error, final_error))
        except BaseException as final_error:
            cleanup_errors.append(final_error)
    except BaseException as wait_error:
        cleanup_errors.append(wait_error)
        try:
            process.kill()
        except BaseException as kill_error:
            if process.poll() is None:
                cleanup_errors.append(kill_error)
        try:
            process.wait(timeout=PROCESS_TERMINATION_GRACE_SECONDS)
        except BaseException as final_error:
            cleanup_errors.append(final_error)
    if cleanup_errors:
        failure = RealDomainBenchmarkError(
            "approved analyzer process-tree cleanup failed: "
            + "; ".join(
                f"{type(error).__name__}: {error}"
                for error in cleanup_errors
            )
        )
        for error in cleanup_errors[1:]:
            failure.add_note(
                f"additional cleanup failure: {type(error).__name__}: {error}"
            )
        raise failure from cleanup_errors[0]


class _WindowsKillJob:
    def __init__(self, process: subprocess.Popen[bytes]) -> None:
        self._handle: int | None = None
        if os.name != "nt":
            return
        import ctypes
        from ctypes import wintypes

        class BasicLimit(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class IoCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in (
                "ReadOperationCount", "WriteOperationCount",
                "OtherOperationCount", "ReadTransferCount",
                "WriteTransferCount", "OtherTransferCount",
            )]

        class ExtendedLimit(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BasicLimit),
                ("IoInfo", IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.argtypes = [
            ctypes.c_void_p,
            wintypes.LPCWSTR,
        ]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [
            wintypes.HANDLE,
            wintypes.HANDLE,
        ]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            primary_error = RealDomainBenchmarkError(
                "could not create analyzer kill job"
            )
            try:
                _kill_process_tree(process)
            except BaseException as cleanup_error:
                _raise_primary_with_cleanup(
                    primary_error,
                    (cleanup_error,),
                )
            raise primary_error
        limits = ExtendedLimit()
        limits.BasicLimitInformation.LimitFlags = 0x00002000
        if not kernel32.SetInformationJobObject(
            handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)
        ) or not kernel32.AssignProcessToJobObject(
            handle, wintypes.HANDLE(process._handle)
        ):
            error_code = ctypes.get_last_error()
            kernel32.CloseHandle(handle)
            primary_error = RealDomainBenchmarkError(
                "could not secure analyzer process tree "
                f"(Windows error {error_code})"
            )
            try:
                _kill_process_tree(process)
            except BaseException as cleanup_error:
                _raise_primary_with_cleanup(
                    primary_error,
                    (cleanup_error,),
                )
            raise primary_error
        self._handle = int(handle)

    def close(self) -> None:
        if self._handle is None:
            return
        import ctypes
        from ctypes import wintypes

        class BasicAccounting(ctypes.Structure):
            _fields_ = [
                ("TotalUserTime", ctypes.c_longlong),
                ("TotalKernelTime", ctypes.c_longlong),
                ("ThisPeriodTotalUserTime", ctypes.c_longlong),
                ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
                ("TotalPageFaultCount", wintypes.DWORD),
                ("TotalProcesses", wintypes.DWORD),
                ("ActiveProcesses", wintypes.DWORD),
                ("TotalTerminatedProcesses", wintypes.DWORD),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.TerminateJobObject.argtypes = [
            wintypes.HANDLE,
            wintypes.UINT,
        ]
        kernel32.TerminateJobObject.restype = wintypes.BOOL
        kernel32.QueryInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        ]
        kernel32.QueryInformationJobObject.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = wintypes.HANDLE(self._handle)
        if not kernel32.TerminateJobObject(handle, 1):
            error_code = ctypes.get_last_error()
            kernel32.CloseHandle(handle)
            self._handle = None
            raise RealDomainBenchmarkError(
                "could not terminate analyzer process tree "
                f"(Windows error {error_code})"
            )
        deadline = time.monotonic() + 5.0
        accounting = BasicAccounting()
        while time.monotonic() < deadline:
            if not kernel32.QueryInformationJobObject(
                handle,
                1,
                ctypes.byref(accounting),
                ctypes.sizeof(accounting),
                None,
            ):
                break
            if accounting.ActiveProcesses == 0:
                break
            time.sleep(0.01)
        kernel32.CloseHandle(handle)
        self._handle = None


@contextmanager
def _temporary_analysis_directory() -> Iterator[Path]:
    directory = Path(
        tempfile.mkdtemp(prefix="drawback-real-domain-analysis-")
    )
    primary_error: BaseException | None = None
    try:
        yield directory
    except BaseException as error:
        primary_error = error
        raise
    finally:
        deadline = time.monotonic() + 5.0
        while True:
            try:
                shutil.rmtree(directory)
                break
            except FileNotFoundError:
                break
            except OSError as error:
                if time.monotonic() >= deadline:
                    cleanup_error = RealDomainBenchmarkError(
                        "approved analyzer temporary files remained locked"
                    )
                    if primary_error is not None:
                        _raise_primary_with_cleanup(
                            primary_error,
                            (cleanup_error,),
                        )
                    raise cleanup_error from error
                time.sleep(0.05)


def _authenticate_bytes(
    reference: ContentAddressedJson,
    label: str,
) -> bytes:
    if not SHA256_PATTERN.fullmatch(reference.sha256):
        raise RealDomainBenchmarkError(f"{label} SHA-256 is malformed")
    payload = reference.path.read_bytes()
    if _sha256(payload) != reference.sha256:
        raise RealDomainBenchmarkError(f"{label} SHA-256 does not match")
    return payload


def _canonical(value: object) -> bytes:
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


def _canonical_compact(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _strict_json(
    payload: bytes,
    label: str,
    *,
    require_canonical: bool = True,
) -> Mapping[str, object]:
    def pairs(items: Sequence[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise RealDomainBenchmarkError(
                    f"{label} contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(payload, object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RealDomainBenchmarkError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise RealDomainBenchmarkError(f"{label} must be a JSON object")
    if require_canonical and payload != _canonical(value):
        raise RealDomainBenchmarkError(f"{label} is not canonical JSON")
    return value


def _load(
    reference: ContentAddressedJson,
    label: str,
    *,
    require_canonical: bool = True,
) -> Mapping[str, object]:
    if not SHA256_PATTERN.fullmatch(reference.sha256):
        raise RealDomainBenchmarkError(f"{label} SHA-256 is malformed")
    payload = reference.path.read_bytes()
    if _sha256(payload) != reference.sha256:
        raise RealDomainBenchmarkError(f"{label} SHA-256 does not match")
    return _strict_json(
        payload,
        label,
        require_canonical=require_canonical,
    )


def _closed(
    value: Mapping[str, object],
    keys: frozenset[str],
    label: str,
) -> None:
    if set(value) != keys:
        raise RealDomainBenchmarkError(f"{label} has an unexpected schema")


def _format(value: Mapping[str, object], expected: str, label: str) -> None:
    if value.get("format") != expected or value.get("version") != VERSION:
        raise RealDomainBenchmarkError(f"{label} format/version is unsupported")


def _title_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.replace("’", "'").replace("‘", "'")
    return " ".join(normalized.split()).casefold()


def load_catalog(
    reference: ContentAddressedJson,
) -> tuple[Mapping[str, CatalogTitle], frozenset[str], frozenset[str]]:
    # The frozen generated catalog predates the canonical sort-key envelope.
    # Its exact bytes remain authenticated; changing them would change the
    # released catalog identity.
    value = _load(
        reference,
        "observed catalog",
        require_canonical=False,
    )
    entries = value.get("entries")
    counts = value.get("counts")
    if (
        value.get("catalogVersion") != 1
        or not isinstance(entries, list)
        or not isinstance(counts, dict)
        or counts.get("observed") != 194
        or counts.get("executable") != 182
        or counts.get("unsupported") != 12
        or len(entries) != 194
    ):
        raise RealDomainBenchmarkError(
            "observed catalog must contain frozen 194/182/12 coverage"
        )
    titles: dict[str, CatalogTitle] = {}
    supported: set[str] = set()
    unsupported: set[str] = set()
    ids: set[str] = set()
    for index, item in enumerate(entries):
        if not isinstance(item, dict):
            raise RealDomainBenchmarkError(
                f"observed catalog entry {index} must be an object"
            )
        drawback_id = item.get("id")
        title = item.get("observedName")
        status = item.get("implementationStatus")
        if (
            not isinstance(drawback_id, str)
            or not drawback_id
            or drawback_id in ids
            or not isinstance(title, str)
            or not title
            or status
            not in {
                "verified",
                "implemented-unverified",
                "partial",
                "unsupported",
            }
        ):
            raise RealDomainBenchmarkError("observed catalog entry is invalid")
        ids.add(drawback_id)
        key = _title_key(title)
        if key in titles:
            raise RealDomainBenchmarkError(
                "observed catalog titles are ambiguous after normalization"
            )
        executable = status != "unsupported"
        titles[key] = CatalogTitle(drawback_id, executable)
        (supported if executable else unsupported).add(drawback_id)
    if len(supported) != 182 or len(unsupported) != 12:
        raise RealDomainBenchmarkError(
            "observed catalog status coverage disagrees with counts"
        )
    return MappingProxyType(titles), frozenset(supported), frozenset(unsupported)


def _parse_completed_pgn(pgn: str, expected_result: str) -> CanonicalGame:
    if "\x00" in pgn:
        raise RealDomainBenchmarkError("PGN contains a NUL byte")
    stream = io.StringIO(pgn)
    game = chess.pgn.read_game(stream)
    if game is None:
        raise RealDomainBenchmarkError("PGN contains no game")
    if game.errors:
        raise RealDomainBenchmarkError("PGN failed legal mainline replay")
    if chess.pgn.read_game(stream) is not None:
        raise RealDomainBenchmarkError("each corpus PGN must contain one game")
    result = game.headers.get("Result")
    if (
        result not in TERMINAL_RESULTS
        or result != expected_result
        or game.end().board().ply() <= 0
    ):
        raise RealDomainBenchmarkError(
            "PGN must be completed, non-empty, and match its terminal result"
        )
    forbidden_headers = {
        key
        for key in game.headers
        if "drawback" in key.casefold()
        or "handicap" in key.casefold()
        or "parameter" in key.casefold()
        or "seed" in key.casefold()
    }
    if forbidden_headers:
        raise RealDomainBenchmarkError(
            "PGN headers may not contain drawback labels, parameters, or seeds"
        )
    if any(
        node.comment.strip() or node.nags
        for node in game.mainline()
    ):
        raise RealDomainBenchmarkError(
            "PGN comments and annotations are forbidden model inputs"
        )
    moves = tuple(move.uci() for move in game.mainline_moves())
    starting_fen = game.board().fen()
    semantic = _canonical(
        {
            "startingFen": starting_fen,
            "moves": moves,
        }
    )
    exporter = chess.pgn.StringExporter(
        headers=False,
        variations=False,
        comments=False,
    )
    body = game.accept(exporter).strip()
    safe_headers = [f'[Result "{result}"]']
    if game.headers.get("SetUp") == "1":
        fen = game.headers.get("FEN")
        if not fen:
            raise RealDomainBenchmarkError("SetUp PGN is missing FEN")
        safe_headers = ['[SetUp "1"]', f'[FEN "{fen}"]', *safe_headers]
    safe_pgn = "\n".join(safe_headers) + "\n\n" + body + "\n"
    return CanonicalGame(
        pgn_sha256=_sha256(pgn.encode()),
        semantic_sha256=_sha256(semantic),
        safe_pgn=safe_pgn,
        ply_count=len(moves),
    )


def load_corpus(reference: ContentAddressedJson) -> tuple[CanonicalGame, ...]:
    value = _load(reference, "real-domain corpus")
    _closed(
        value,
        frozenset({"format", "version", "consent", "games"}),
        "real-domain corpus",
    )
    _format(value, CORPUS_FORMAT, "real-domain corpus")
    consent = value["consent"]
    games = value["games"]
    if not isinstance(consent, dict):
        raise RealDomainBenchmarkError("corpus consent must be an object")
    _closed(
        consent,
        frozenset({"basis", "completedOnly", "liveCollection"}),
        "corpus consent",
    )
    if (
        consent.get("basis") != "explicit"
        or consent.get("completedOnly") is not True
        or consent.get("liveCollection") is not False
    ):
        raise RealDomainBenchmarkError(
            "corpus must attest explicit consent and completed offline use"
        )
    if not isinstance(games, list) or not games:
        raise RealDomainBenchmarkError("corpus games must be non-empty")
    parsed: list[CanonicalGame] = []
    pgn_hashes: set[str] = set()
    semantic_hashes: set[str] = set()
    for index, item in enumerate(games):
        if not isinstance(item, dict):
            raise RealDomainBenchmarkError(f"corpus game {index} is invalid")
        _closed(
            item,
            frozenset(
                {
                    "pgn",
                    "pgnSha256",
                    "completed",
                    "result",
                    "simulationSeed",
                }
            ),
            f"corpus game {index}",
        )
        pgn = item.get("pgn")
        digest = item.get("pgnSha256")
        result = item.get("result")
        if (
            not isinstance(pgn, str)
            or not isinstance(digest, str)
            or not SHA256_PATTERN.fullmatch(digest)
            or _sha256(pgn.encode()) != digest
            or item.get("completed") is not True
            or result not in TERMINAL_RESULTS
            or item.get("simulationSeed") is not None
        ):
            raise RealDomainBenchmarkError(
                "corpus game must be completed, content-addressed, and real"
            )
        game = _parse_completed_pgn(pgn, str(result))
        if game.pgn_sha256 in pgn_hashes:
            raise RealDomainBenchmarkError("corpus contains duplicate PGN bytes")
        if game.semantic_sha256 in semantic_hashes:
            raise RealDomainBenchmarkError(
                "corpus contains semantically duplicate games"
            )
        pgn_hashes.add(game.pgn_sha256)
        semantic_hashes.add(game.semantic_sha256)
        parsed.append(game)
    return tuple(parsed)


def audit_released_training(
    *,
    candidate_training_corpus_set: ContentAddressedJson,
    public_training_release: ContentAddressedJson,
    private_training_manifest: ContentAddressedJson,
    training_datasets: Sequence[ContentAddressedJson],
    supplement_manifests: Sequence[ContentAddressedJson],
    supplement_plans: Sequence[ContentAddressedJson],
    benchmark_games: Sequence[CanonicalGame],
) -> Mapping[str, object]:
    """Authenticate and replay the exact released primary + supplement union."""

    candidate_value = _load(
        candidate_training_corpus_set,
        "candidate training corpus set",
    )
    try:
        candidate = verify_training_corpus_set(candidate_value)
    except TrainingCorpusSetError as error:
        raise RealDomainBenchmarkError(
            "candidate training corpus set is invalid or incomplete"
        ) from error
    public = _strict_json(
        _authenticate_bytes(public_training_release, "public training release"),
        "public training release",
        require_canonical=False,
    )
    private = _strict_json(
        _authenticate_bytes(private_training_manifest, "private training manifest"),
        "private training manifest",
        require_canonical=False,
    )
    primary = candidate["primary"]
    supplements = candidate["supplements"]
    if (
        len(training_datasets) != 7
        or len(supplement_manifests) != 6
        or len(supplement_plans) != 6
        or primary["release_root_sha256"] != public_training_release.sha256
        or primary["corpus_run_id"] != public.get("corpusRunId")
        or primary["private_train_manifest_sha256"]
        != private_training_manifest.sha256
        or not isinstance(private.get("dataset"), dict)
        or primary["dataset_sha256"] != private["dataset"].get("sha256")
    ):
        raise RealDomainBenchmarkError(
            "released training sources are incomplete or disagree"
        )
    expected_datasets = [
        primary["dataset_sha256"],
        *[item["dataset_sha256"] for item in supplements],
    ]
    for index, (reference, expected) in enumerate(
        zip(training_datasets, expected_datasets, strict=True)
    ):
        if reference.sha256 != expected:
            raise RealDomainBenchmarkError(
                f"training dataset {index} disagrees with candidate corpus set"
            )
    generation_by_plan: list[Mapping[str, object]] = []
    for references, key, label in (
        (supplement_manifests, "manifest_sha256", "manifest"),
        (supplement_plans, "plan_sha256", "plan"),
    ):
        for index, (reference, supplement) in enumerate(
            zip(references, supplements, strict=True)
        ):
            if reference.sha256 != supplement[key]:
                raise RealDomainBenchmarkError(
                    f"hard-negative {label} {index} disagrees with candidate"
                )
            source = _strict_json(
                _authenticate_bytes(reference, f"hard-negative {label} {index}"),
                f"hard-negative {label} {index}",
                require_canonical=False,
            )
            if label == "manifest":
                profile = source.get("hardNegativeProfile")
                generation = source.get("hardNegativeGeneration")
                splits = source.get("splits")
                train = (
                    splits.get("train")
                    if isinstance(splits, dict)
                    else None
                )
                valid = (
                    source.get("schemaVersion") == supplement["schema_version"]
                    and source.get("symbolicFeatureVersion")
                    == supplement["symbolic_feature_version"]
                    and source.get("maxPlies") == supplement["max_plies"]
                    and source.get("observationPolicy")
                    == supplement["observation_policy"]
                    and source.get("evaluatorPolicyId")
                    == supplement["evaluator_policy_id"]
                    and source.get("evaluatorPolicyVersion")
                    == supplement["evaluator_policy_version"]
                    and source.get("engineBinarySha256")
                    == supplement["engine_binary_sha256"]
                    and source.get("engineFingerprint")
                    == supplement["engine_fingerprint"]
                    and source.get("agentIds")
                    == supplement["agent_domain"]
                    and isinstance(profile, dict)
                    and profile.get("id") == supplement["profile_id"]
                    and profile.get("ruleIds") == supplement["rule_ids"]
                    and _sha256(_canonical_compact(profile))
                    == supplement["profile_sha256"]
                    and isinstance(generation, dict)
                    and generation.get("planSha256")
                    == supplement["plan_sha256"]
                    and generation.get("sourceRevision")
                    == supplement["source_revision"]
                    and generation.get("runId")
                    == supplement["generation_run_id"]
                    and isinstance(train, dict)
                    and train.get("sha256") == supplement["dataset_sha256"]
                    and train.get("bytes") == supplement["dataset_bytes"]
                    and train.get("rows") == supplement["rows"]
                    and train.get("games") == supplement["games"]
                )
                if isinstance(generation, dict):
                    generation_by_plan.append(generation)
            else:
                metadata = source.get("metadata")
                profile = (
                    metadata.get("hardNegativeProfile")
                    if isinstance(metadata, dict)
                    else None
                )
                run_plan = source.get("runPlan")
                valid = (
                    source.get("schemaVersion") == 1
                    and source.get("sourceRevision")
                    == supplement["source_revision"]
                    and isinstance(metadata, dict)
                    and metadata.get("schemaVersion", supplement["schema_version"])
                    == supplement["schema_version"]
                    and metadata.get(
                        "symbolicFeatureVersion",
                        supplement["symbolic_feature_version"],
                    )
                    == supplement["symbolic_feature_version"]
                    and metadata.get("maxPlies", supplement["max_plies"])
                    == supplement["max_plies"]
                    and metadata.get(
                        "observationPolicy",
                        supplement["observation_policy"],
                    )
                    == supplement["observation_policy"]
                    and isinstance(profile, dict)
                    and profile.get("id") == supplement["profile_id"]
                    and profile.get("ruleIds") == supplement["rule_ids"]
                    and _sha256(_canonical_compact(profile))
                    == supplement["profile_sha256"]
                    and isinstance(run_plan, dict)
                    and run_plan.get("ruleIds") == supplement["rule_ids"]
                    and run_plan.get("runId")
                    == supplement["generation_run_id"]
                    and run_plan.get("corpusConfigSha256")
                    == generation_by_plan[index]["corpusConfigSha256"]
                    and isinstance(source.get("schedule"), dict)
                )
            if not valid:
                raise RealDomainBenchmarkError(
                    f"hard-negative {label} {index} content disagrees "
                    "with candidate identity"
                )

    semantics: set[str] = set()
    seeds: set[int] = set()
    total_rows = 0
    total_games = 0
    for dataset_index, (reference, identity) in enumerate(
        zip(training_datasets, [primary, *supplements], strict=True)
    ):
        digest = hashlib.sha256()
        byte_count = 0
        row_count = 0
        game_count = 0
        sealed: set[str] = set()
        current_id: str | None = None
        board: chess.Board | None = None
        starting_fen = ""
        moves: list[str] = []
        current_seed: int | None = None

        def seal() -> None:
            nonlocal game_count
            if current_id is None:
                return
            semantics.add(
                _sha256(
                    _canonical(
                        {"startingFen": starting_fen, "moves": tuple(moves)}
                    )
                )
            )
            sealed.add(current_id)
            game_count += 1

        with reference.path.open("rb") as source:
            for line_number, payload in enumerate(source, start=1):
                digest.update(payload)
                byte_count += len(payload)
                if not payload.endswith(b"\n"):
                    raise RealDomainBenchmarkError(
                        "training dataset rows must be newline terminated"
                    )
                row = _strict_json(
                    payload,
                    f"training dataset {dataset_index} line {line_number}",
                    require_canonical=False,
                )
                game_id = row.get("gameId")
                seed = row.get("seed")
                ply = row.get("ply")
                fen = row.get("fenBefore")
                move_code = row.get("move")
                if (
                    not isinstance(game_id, str)
                    or not game_id
                    or type(seed) is not int
                    or seed < 0
                    or type(ply) is not int
                    or ply < 0
                    or not isinstance(fen, str)
                    or not isinstance(move_code, str)
                ):
                    raise RealDomainBenchmarkError(
                        "training dataset replay fields are invalid"
                    )
                if game_id != current_id:
                    seal()
                    if game_id in sealed or ply != 0:
                        raise RealDomainBenchmarkError(
                            "training games must be contiguous and begin at ply zero"
                        )
                    try:
                        board = chess.Board(fen)
                    except ValueError as error:
                        raise RealDomainBenchmarkError(
                            "training game has invalid starting FEN"
                        ) from error
                    if not board.is_valid():
                        raise RealDomainBenchmarkError(
                            "training game has invalid starting position"
                        )
                    current_id = game_id
                    current_seed = seed
                    starting_fen = board.fen()
                    moves = []
                    seeds.add(seed)
                assert board is not None
                if (
                    seed != current_seed
                    or ply != len(moves)
                    or fen != board.fen(en_passant="legal")
                ):
                    raise RealDomainBenchmarkError(
                        "training ply/FEN sequence disagrees with replay"
                    )
                try:
                    move = chess.Move.from_uci(move_code)
                except ValueError as error:
                    raise RealDomainBenchmarkError(
                        "training move is not UCI"
                    ) from error
                if move not in board.legal_moves:
                    raise RealDomainBenchmarkError(
                        "training move is not legal"
                    )
                board.push(move)
                moves.append(move_code)
                row_count += 1
        seal()
        if (
            digest.hexdigest() != reference.sha256
            or byte_count != identity["dataset_bytes"]
            or row_count != identity["rows"]
            or game_count != identity["games"]
        ):
            raise RealDomainBenchmarkError(
                "training dataset bytes/counts disagree with candidate"
            )
        total_rows += row_count
        total_games += game_count
    benchmark_semantics = {game.semantic_sha256 for game in benchmark_games}
    if benchmark_semantics.intersection(semantics):
        raise RealDomainBenchmarkError(
            "real-domain corpus overlaps a replayed semantic training game"
        )
    return MappingProxyType(
        {
            "candidateTrainingCorpusSetArtifactSha256": (
                candidate_training_corpus_set.sha256
            ),
            "trainingCorpusSetSha256": candidate["sha256"],
            "datasetCount": 7,
            "supplementSourceCount": 6,
            "replayedGameCount": total_games,
            "replayedRowCount": total_rows,
            "distinctSeedCount": len(seeds),
        }
    )


def _probabilities(
    value: object,
    class_ids: tuple[str, ...],
    label: str,
) -> Mapping[str, float]:
    if not isinstance(value, dict) or set(value) != set(class_ids):
        raise RealDomainBenchmarkError(
            f"{label} posterior does not match declared classes"
        )
    parsed: dict[str, float] = {}
    for drawback_id in class_ids:
        probability = value.get(drawback_id)
        if (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not math.isfinite(float(probability))
            or float(probability) < 0.0
        ):
            raise RealDomainBenchmarkError(f"{label} probability is invalid")
        parsed[drawback_id] = float(probability)
    if not math.isclose(math.fsum(parsed.values()), 1.0, abs_tol=1e-9):
        raise RealDomainBenchmarkError(f"{label} probabilities must sum to one")
    return MappingProxyType(parsed)


def validate_analysis(
    value: Mapping[str, object],
    game: CanonicalGame,
    supported_ids: frozenset[str],
) -> ApprovedAnalysis:
    _closed(
        value,
        frozenset(
            {
                "format",
                "version",
                "pgnSha256",
                "completed",
                "plyCount",
                "classIds",
                "unavailableSupportedIds",
                "snapshots",
                "predictor",
            }
        ),
        "approved analysis",
    )
    _format(value, ANALYSIS_FORMAT, "approved analysis")
    class_ids_value = value["classIds"]
    unavailable_value = value["unavailableSupportedIds"]
    snapshots_value = value["snapshots"]
    predictor = value["predictor"]
    if (
        value.get("pgnSha256") != game.pgn_sha256
        or value.get("completed") is not True
        or value.get("plyCount") != game.ply_count
        or not isinstance(class_ids_value, list)
        or any(not isinstance(item, str) for item in class_ids_value)
        or len(set(class_ids_value)) != len(class_ids_value)
        or not isinstance(unavailable_value, list)
        or any(not isinstance(item, str) for item in unavailable_value)
        or len(set(unavailable_value)) != len(unavailable_value)
        or not isinstance(snapshots_value, list)
        or not snapshots_value
        or not isinstance(predictor, dict)
    ):
        raise RealDomainBenchmarkError("approved analysis binding is invalid")
    class_ids = tuple(class_ids_value)
    unavailable = tuple(sorted(unavailable_value))
    if (
        set(class_ids).union(unavailable) != set(supported_ids)
        or set(class_ids).intersection(unavailable)
        or len(class_ids) + len(unavailable) != 182
        or frozenset(unavailable) != STANDARD_PGN_UNAVAILABLE_IDS
        or len(class_ids) != 180
    ):
        raise RealDomainBenchmarkError(
            "analysis must expose the approved 180-class standard-PGN view "
            "and explicitly mark both evaluator-backed classes unavailable"
        )
    _closed(
        predictor,
        frozenset(
            {
                "mode",
                "browserArtifactSha256",
                "ensembleReleaseSha256",
                "calibrationSha256",
                "approvalEvidenceSha256",
            }
        ),
        "analysis predictor",
    )
    if (
        predictor.get("mode") != ENSEMBLE_MODE
        or any(
            not isinstance(predictor.get(key), str)
            or not SHA256_PATTERN.fullmatch(str(predictor[key]))
            for key in (
                "browserArtifactSha256",
                "ensembleReleaseSha256",
                "calibrationSha256",
                "approvalEvidenceSha256",
            )
        )
    ):
        raise RealDomainBenchmarkError(
            "analysis is not bound to an approved ensemble browser artifact"
        )
    snapshots: list[AnalysisSnapshot] = []
    previous_ply = 0
    for index, item in enumerate(snapshots_value):
        if not isinstance(item, dict):
            raise RealDomainBenchmarkError("analysis snapshot is invalid")
        _closed(
            item,
            frozenset({"ply", "white", "black"}),
            f"analysis snapshot {index}",
        )
        ply = item.get("ply")
        if (
            type(ply) is not int
            or ply <= previous_ply
            or ply > game.ply_count
        ):
            raise RealDomainBenchmarkError(
                "analysis snapshots must have increasing completed plies"
            )
        previous_ply = ply
        snapshots.append(
            AnalysisSnapshot(
                ply,
                _probabilities(item["white"], class_ids, "white"),
                _probabilities(item["black"], class_ids, "black"),
            )
        )
    if snapshots[-1].ply != game.ply_count:
        raise RealDomainBenchmarkError(
            "analysis must include the completed final position"
        )
    return ApprovedAnalysis(
        game.pgn_sha256,
        game.ply_count,
        class_ids,
        unavailable,
        tuple(snapshots),
        MappingProxyType(dict(predictor)),
    )


def _load_labels(
    reference: ContentAddressedJson,
    expected_join_digests: frozenset[str],
    titles: Mapping[str, CatalogTitle],
) -> Mapping[str, RevealedGame]:
    value = _load(reference, "revealed labels")
    _closed(
        value,
        frozenset({"format", "version", "revealTiming", "labels"}),
        "revealed labels",
    )
    _format(value, LABEL_FORMAT, "revealed labels")
    if value.get("revealTiming") != "after-game-completion":
        raise RealDomainBenchmarkError(
            "labels must be revealed only after game completion"
        )
    labels = value["labels"]
    if not isinstance(labels, list):
        raise RealDomainBenchmarkError("revealed labels must be an array")
    loaded: dict[str, RevealedGame] = {}

    def side(raw: object, label: str) -> RevealedSide:
        if not isinstance(raw, dict):
            raise RealDomainBenchmarkError(f"{label} reveal is invalid")
        _closed(raw, frozenset({"title", "parameters"}), label)
        title = raw.get("title")
        parameters = raw.get("parameters")
        if not isinstance(title, str) or not isinstance(parameters, dict):
            raise RealDomainBenchmarkError(f"{label} reveal is invalid")
        mapped = titles.get(_title_key(title))
        if mapped is None:
            raise RealDomainBenchmarkError(
                f"{label} title is not in the complete observed catalog"
            )
        # Canonical JSON validation already excludes non-JSON parameter values.
        return RevealedSide(
            mapped.drawback_id,
            MappingProxyType(dict(parameters)),
        )

    for index, item in enumerate(labels):
        if not isinstance(item, dict):
            raise RealDomainBenchmarkError(f"label {index} is invalid")
        _closed(
            item,
            frozenset({"pgnSha256", "white", "black"}),
            f"label {index}",
        )
        digest = item.get("pgnSha256")
        if (
            not isinstance(digest, str)
            or digest in loaded
            or digest not in expected_join_digests
        ):
            raise RealDomainBenchmarkError(
                "revealed label binding is duplicate or unknown"
            )
        loaded[digest] = RevealedGame(
            side(item["white"], "white"),
            side(item["black"], "black"),
        )
    if set(loaded) != expected_join_digests:
        raise RealDomainBenchmarkError(
            "revealed labels must cover every corpus game exactly once"
        )
    return MappingProxyType(loaded)


def _metrics(rows: Sequence[PredictionExample]) -> Mapping[str, object] | None:
    if not rows:
        return None
    report = evaluate(rows)
    zero_truth_probability_count = sum(
        row.probabilities[row.true_drawback] == 0.0 for row in rows
    )
    return {
        "count": report.count,
        "playerGameCount": report.player_game_count,
        "top1Accuracy": report.top_1_accuracy,
        "top3Accuracy": report.top_3_accuracy,
        "top5Accuracy": report.top_5_accuracy,
        "negativeLogLikelihood": (
            None
            if not math.isfinite(report.negative_log_likelihood)
            else report.negative_log_likelihood
        ),
        "negativeLogLikelihoodFinite": math.isfinite(
            report.negative_log_likelihood
        ),
        "zeroTruthProbabilityCount": zero_truth_probability_count,
        "brierScore": report.brier_score,
        "expectedCalibrationError": report.expected_calibration_error,
    }


def _build_release_claim_gate(
    *,
    config: BenchmarkClaimConfig,
    game_count: int,
    player_game_support: Mapping[str, int],
    analyzable_ids: frozenset[str],
    zero_truth_probability_count: int,
) -> Mapping[str, object]:
    claimed_ids = tuple(
        sorted(config.claimed_drawback_ids or player_game_support)
    )
    unknown_claims = tuple(
        drawback_id
        for drawback_id in claimed_ids
        if drawback_id not in analyzable_ids
    )
    undercovered = {
        drawback_id: player_game_support.get(drawback_id, 0)
        for drawback_id in claimed_ids
        if player_game_support.get(drawback_id, 0)
        < RELEASE_CLAIM_MINIMUM_PLAYER_GAMES_PER_RULE
    }
    failures: list[str] = []
    if game_count < RELEASE_CLAIM_MINIMUM_GAMES:
        failures.append("minimum-completed-games")
    if unknown_claims:
        failures.append("claimed-rule-not-standard-pgn-analyzable")
    if undercovered:
        failures.append("minimum-player-games-per-claimed-rule")
    if zero_truth_probability_count:
        failures.append("nonfinite-truth-negative-log-likelihood")
    requirements_satisfied = not failures and bool(claimed_ids)
    return {
        "mode": config.mode,
        "releaseClaimRequested": config.mode == "release-claim",
        "requirementsSatisfied": requirements_satisfied,
        "releaseClaimPassing": (
            config.mode == "release-claim" and requirements_satisfied
        ),
        "minimumCompletedGames": RELEASE_CLAIM_MINIMUM_GAMES,
        "minimumPlayerGamesPerClaimedRule": (
            RELEASE_CLAIM_MINIMUM_PLAYER_GAMES_PER_RULE
        ),
        "claimedDrawbackIds": list(claimed_ids),
        "claimedRuleSupport": {
            drawback_id: player_game_support.get(drawback_id, 0)
            for drawback_id in claimed_ids
        },
        "undercoveredClaimedRules": undercovered,
        "unanalyzableClaimedRules": list(unknown_claims),
        "zeroTruthProbabilityPlayerGames": zero_truth_probability_count,
        "failures": failures,
    }


def _latest(
    analysis: ApprovedAnalysis,
    horizon: int | None,
) -> AnalysisSnapshot:
    if horizon is None:
        return analysis.snapshots[-1]
    eligible = tuple(
        snapshot for snapshot in analysis.snapshots if snapshot.ply <= horizon
    )
    if not eligible:
        raise RealDomainBenchmarkError(
            "approved analysis omits the first completed ply"
        )
    return eligible[-1]


def _jsonable(value: object) -> object:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _publish_no_clobber(output: Path, value: Mapping[str, object]) -> bytes:
    payload = _canonical(_jsonable(value))
    publish_bytes_durable_exact(
        output,
        payload,
        label="real-domain benchmark artifact",
    )
    return payload


def create_real_domain_prediction_bundle(
    *,
    corpus: ContentAddressedJson,
    public_training_release: ContentAddressedJson,
    private_training_manifest: ContentAddressedJson,
    candidate_training_corpus_set: ContentAddressedJson,
    training_datasets: Sequence[ContentAddressedJson],
    supplement_manifests: Sequence[ContentAddressedJson],
    supplement_plans: Sequence[ContentAddressedJson],
    observed_catalog: ContentAddressedJson,
    analyzer: ApprovedPostGameAnalyzer,
    output: Path,
) -> Mapping[str, object]:
    """Stage A: run label-blind post-game inference and publish predictions.

    There is intentionally no revealed-label argument.  Run this stage in a
    process and access domain where the reveal file/path is absent.
    """

    _titles, supported_ids, _unsupported_ids = load_catalog(observed_catalog)
    games = load_corpus(corpus)
    training_identity = audit_released_training(
        candidate_training_corpus_set=candidate_training_corpus_set,
        public_training_release=public_training_release,
        private_training_manifest=private_training_manifest,
        training_datasets=training_datasets,
        supplement_manifests=supplement_manifests,
        supplement_plans=supplement_plans,
        benchmark_games=games,
    )
    cases: list[Mapping[str, object]] = []
    identity: Mapping[str, object] | None = None
    for game in games:
        sanitized_sha256 = _sha256(game.safe_pgn.encode())
        raw = analyzer.analyze_completed_game(
            pgn_sha256=sanitized_sha256,
            pgn=game.safe_pgn,
        )
        sanitized_game = CanonicalGame(
            pgn_sha256=sanitized_sha256,
            semantic_sha256=game.semantic_sha256,
            safe_pgn=game.safe_pgn,
            ply_count=game.ply_count,
        )
        analysis = validate_analysis(raw, sanitized_game, supported_ids)
        if identity is None:
            identity = analysis.predictor_identity
        elif analysis.predictor_identity != identity:
            raise RealDomainBenchmarkError(
                "approved analyzer identity changed within the corpus"
            )
        cases.append(
            {
                # The join digest stays in this private evaluator bundle.  It
                # is never given to the analyzer or emitted in the report.
                "joinSha256": game.pgn_sha256,
                "sanitizedPgnSha256": sanitized_sha256,
                "semanticSha256": game.semantic_sha256,
                "analysis": raw,
            }
        )
    if identity is None:
        raise RealDomainBenchmarkError("no approved analyses were produced")
    bundle: Mapping[str, object] = {
        "format": PREDICTION_BUNDLE_FORMAT,
        "version": VERSION,
        "stage": "label-blind-inference",
        "aggregateMetricsIncluded": False,
        "cases": cases,
        "provenance": {
            "corpusSha256": corpus.sha256,
            "observedCatalogSha256": observed_catalog.sha256,
            "trainingRelease": dict(training_identity),
            "predictor": dict(identity),
            "labelIsolationEnforcement": "external-access-domain",
        },
    }
    payload = _publish_no_clobber(output, bundle)
    return _strict_json(payload, "real-domain prediction bundle")


def _load_prediction_bundle(
    reference: ContentAddressedJson,
    supported_ids: frozenset[str],
) -> tuple[
    tuple[tuple[str, ApprovedAnalysis], ...],
    Mapping[str, object],
]:
    value = _load(reference, "real-domain prediction bundle")
    _closed(
        value,
        frozenset(
            {
                "format",
                "version",
                "stage",
                "aggregateMetricsIncluded",
                "cases",
                "provenance",
            }
        ),
        "real-domain prediction bundle",
    )
    _format(value, PREDICTION_BUNDLE_FORMAT, "real-domain prediction bundle")
    if (
        value.get("stage") != "label-blind-inference"
        or value.get("aggregateMetricsIncluded") is not False
        or not isinstance(value.get("cases"), list)
        or not value["cases"]
        or not isinstance(value.get("provenance"), dict)
    ):
        raise RealDomainBenchmarkError("prediction bundle is invalid")
    provenance = value["provenance"]
    _closed(
        provenance,
        frozenset(
            {
                "corpusSha256",
                "observedCatalogSha256",
                "trainingRelease",
                "predictor",
                "labelIsolationEnforcement",
            }
        ),
        "prediction bundle provenance",
    )
    if (
        not isinstance(provenance.get("predictor"), dict)
        or not isinstance(provenance.get("trainingRelease"), dict)
        or any(
            not isinstance(provenance.get(key), str)
            or not SHA256_PATTERN.fullmatch(str(provenance[key]))
            for key in ("corpusSha256", "observedCatalogSha256")
        )
        or provenance.get("labelIsolationEnforcement")
        != "external-access-domain"
    ):
        raise RealDomainBenchmarkError(
            "prediction bundle provenance is invalid"
        )
    loaded: list[tuple[str, ApprovedAnalysis]] = []
    joins: set[str] = set()
    sanitized: set[str] = set()
    semantics: set[str] = set()
    for index, item in enumerate(value["cases"]):
        if not isinstance(item, dict):
            raise RealDomainBenchmarkError("prediction bundle case is invalid")
        _closed(
            item,
            frozenset(
                {
                    "joinSha256",
                    "sanitizedPgnSha256",
                    "semanticSha256",
                    "analysis",
                }
            ),
            f"prediction bundle case {index}",
        )
        join = item.get("joinSha256")
        safe = item.get("sanitizedPgnSha256")
        semantic = item.get("semanticSha256")
        raw = item.get("analysis")
        if (
            not isinstance(join, str)
            or not SHA256_PATTERN.fullmatch(join)
            or join in joins
            or not isinstance(safe, str)
            or not SHA256_PATTERN.fullmatch(safe)
            or safe in sanitized
            or not isinstance(semantic, str)
            or not SHA256_PATTERN.fullmatch(semantic)
            or semantic in semantics
            or not isinstance(raw, dict)
            or type(raw.get("plyCount")) is not int
        ):
            raise RealDomainBenchmarkError(
                "prediction bundle case binding is invalid or duplicate"
            )
        game = CanonicalGame(
            safe,
            semantic,
            "",
            raw["plyCount"],
        )
        analysis = validate_analysis(raw, game, supported_ids)
        if analysis.predictor_identity != provenance["predictor"]:
            raise RealDomainBenchmarkError(
                "prediction case predictor identity differs from provenance"
            )
        joins.add(join)
        sanitized.add(safe)
        semantics.add(semantic)
        loaded.append((join, analysis))
    return tuple(loaded), MappingProxyType(dict(provenance))


def publish_real_domain_benchmark_report(
    *,
    prediction_bundle: ContentAddressedJson,
    revealed_labels: ContentAddressedJson,
    observed_catalog: ContentAddressedJson,
    output: Path,
    claim_config: BenchmarkClaimConfig = BenchmarkClaimConfig(),
) -> Mapping[str, object]:
    """Stage B: score an authenticated prediction bundle without inference."""

    titles, supported_ids, unsupported_ids = load_catalog(observed_catalog)
    cases, bundle_provenance = _load_prediction_bundle(
        prediction_bundle,
        supported_ids,
    )
    if bundle_provenance["observedCatalogSha256"] != observed_catalog.sha256:
        raise RealDomainBenchmarkError(
            "prediction bundle names a different observed catalog"
        )
    labels = _load_labels(
        revealed_labels,
        frozenset(join for join, _analysis in cases),
        titles,
    )
    label_count = len(cases) * 2
    supported_count = 0
    unsupported_count = 0
    analyzable_count = 0
    unavailable_supported_count = 0
    player_game_support: dict[str, int] = {}
    rows: dict[tuple[str, str], list[PredictionExample]] = {}
    for join_digest, analysis in cases:
        truth = labels[join_digest]
        for color in ("white", "black"):
            side = truth.white if color == "white" else truth.black
            if side.drawback_id in unsupported_ids:
                unsupported_count += 1
                continue
            supported_count += 1
            if side.drawback_id not in analysis.class_ids:
                if side.drawback_id not in analysis.unavailable_supported_ids:
                    raise RealDomainBenchmarkError(
                        "supported truth has no explicit analysis coverage"
                    )
                unavailable_supported_count += 1
                continue
            analyzable_count += 1
            player_game_support[side.drawback_id] = (
                player_game_support.get(side.drawback_id, 0) + 1
            )
            for horizon in (*HORIZONS, None):
                snapshot = _latest(analysis, horizon)
                probabilities = (
                    snapshot.white if color == "white" else snapshot.black
                )
                key = (color, "final" if horizon is None else str(horizon))
                rows.setdefault(key, []).append(
                    PredictionExample(
                        # Join digest stays in memory and is never reported.
                        game_id=join_digest,
                        move_number=snapshot.ply,
                        observed_ply=snapshot.ply,
                        player_color=color,
                        true_drawback=side.drawback_id,
                        probabilities=probabilities,
                    )
                )
    if supported_count + unsupported_count != label_count:
        raise RealDomainBenchmarkError("label coverage accounting failed")

    metrics_by_color: dict[str, object] = {}
    for color in ("white", "black"):
        metrics_by_color[color] = {
            "final": _metrics(rows.get((color, "final"), [])),
            "moveHorizons": {
                str(horizon): _metrics(rows.get((color, str(horizon)), []))
                for horizon in HORIZONS
            },
        }
    combined: dict[str, object] = {
        "final": _metrics(
            [
                *rows.get(("white", "final"), []),
                *rows.get(("black", "final"), []),
            ]
        ),
        "moveHorizons": {
            str(horizon): _metrics(
                [
                    *rows.get(("white", str(horizon)), []),
                    *rows.get(("black", str(horizon)), []),
                ]
            )
            for horizon in HORIZONS
        },
    }
    zero_truth_probability_count = len(
        {
            (row.game_id, row.player_color)
            for metric_rows in rows.values()
            for row in metric_rows
            if row.probabilities[row.true_drawback] == 0.0
        }
    )
    claim_gate = _build_release_claim_gate(
        config=claim_config,
        game_count=len(cases),
        player_game_support=player_game_support,
        analyzable_ids=frozenset(
            supported_ids.difference(STANDARD_PGN_UNAVAILABLE_IDS)
        ),
        zero_truth_probability_count=zero_truth_probability_count,
    )
    if (
        claim_config.mode == "release-claim"
        and claim_gate["releaseClaimPassing"] is not True
    ):
        failures = ", ".join(str(item) for item in claim_gate["failures"])
        raise RealDomainBenchmarkError(
            f"release-claim benchmark gate failed: {failures}"
        )
    report: Mapping[str, object] = {
        "format": REPORT_FORMAT,
        "version": VERSION,
        "scope": {
            "completedGamesOnly": True,
            "liveGameIntegration": False,
            "aggregateOnly": True,
            "rawPgnIncluded": False,
            "playerIdentifiersIncluded": False,
            "labelIsolationEnforcement": "external-access-domain",
            "releaseMetricsRequireNoLabelMountReceipt": True,
        },
        "catalogCoverage": {
            "mappingComplete": True,
            "observedTitleCount": len(titles),
            "supportedTitleCount": len(supported_ids),
            "unsupportedTitleCount": len(unsupported_ids),
        },
        "labelCoverage": {
            "gameCount": len(cases),
            "playerLabelCount": label_count,
            "supportedPlayerLabels": supported_count,
            "unsupportedPlayerLabels": unsupported_count,
            "supportedCoverage": supported_count / label_count,
            "analyzableSupportedPlayerLabels": analyzable_count,
            "unavailableSupportedPlayerLabels": unavailable_supported_count,
            "analyzableSupportedCoverage": (
                None
                if supported_count == 0
                else analyzable_count / supported_count
            ),
        },
        "releaseClaimGate": claim_gate,
        "metrics": {
            "combined": combined,
            "byColor": metrics_by_color,
        },
        "provenance": {
            "predictionBundleSha256": prediction_bundle.sha256,
            "revealedLabelsSha256": revealed_labels.sha256,
            "observedCatalogSha256": observed_catalog.sha256,
            "labelBlindInference": {
                "corpusSha256": bundle_provenance["corpusSha256"],
                "trainingRelease": bundle_provenance["trainingRelease"],
                "predictor": bundle_provenance["predictor"],
                "labelIsolationEnforcement": (
                    bundle_provenance["labelIsolationEnforcement"]
                ),
            },
        },
        "serialization": {
            "nonFiniteMetricPolicy": "null",
        },
    }
    payload = _publish_no_clobber(output, report)
    return _strict_json(payload, "real-domain benchmark report")
