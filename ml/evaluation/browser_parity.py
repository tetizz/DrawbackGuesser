"""Canonical Python/browser parity evidence using a production Web Worker."""

from __future__ import annotations

import argparse
import chess.pgn
from contextlib import ExitStack, contextmanager
import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import hashlib
from html.parser import HTMLParser
import json
import math
import os
from pathlib import Path, PurePosixPath
import signal
import shutil
import subprocess
import tempfile
from threading import Thread
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import Iterator, Mapping, Sequence
from ml.training.drawback_ml.records import FeatureRecord
from ml.training.drawback_ml.symbolic_schema import SYMBOLIC_RULE_IDS
from ml.training.drawback_ml.ensemble import load_hybrid_ensemble
from ml.training.drawback_ml.durable_publish import publish_bytes_durable
from ml.training.drawback_ml.path_validation import is_portable_safe_basename

from .validation_gate import PROTOCOL_ID, _canonical_pretty
from .ensemble_calibration import ContentAddressedFile, load_ensemble_calibration
from .release_selection_bundle import ContentAddressedJson
from .ensemble_release import verify_ensemble_release
from .promotion_evaluator import (
    PromotionTemperatures,
    UNAVAILABLE_BROWSER_RULE_IDS,
    _calibration_fusion_policy,
    _calibration_temperature,
    _open_checkpoint_sources,
    _verified_checkpoint_bytes,
    _verify_calibration_binding,
    predict_calibrated_two_heads,
)


INPUT_FORMAT = "drawbacktrainer-browser-parity-input"
TRANSCRIPT_FORMAT = "drawbacktrainer-browser-worker-transcript"
EVIDENCE_FORMAT = "drawbacktrainer-browser-parity-evidence"
VERSION = 1
TOLERANCE = 1e-6
TOP_K = 5
SOURCE_PATHS = (
    "apps",
    "packages",
    "ml",
    "scripts",
    "engine",
    "package.json",
    "pnpm-lock.yaml",
)
SHA256_KEYS = {
    "ensembleSha256",
    "calibrationSha256",
    "fusionSelectionSha256",
    "pnpmLockSha256",
}
PUBLIC_FIXTURE_FORMAT = "drawbacktrainer-public-pgn-parity-fixture"
PUBLIC_FIXTURE_PROTOCOL_ID = "drawbacktrainer-public-pgn-parity-v1"
PUBLIC_FIXTURE_SEED_DOMAIN = "public-parity-v1"
PUBLIC_FIXTURE_ROOT_SEED = 0x5A17_2026
PUBLIC_FIXTURE_GAME_COUNT = 8
PUBLIC_FIXTURE_MAX_PLIES = 320
PUBLIC_FIXTURE_AGENTS = (
    "random-legal",
    "human-like-weak",
    "greedy-material",
)
PROCESS_TERMINATION_GRACE_SECONDS = 5.0
GIT_TIMEOUT_SECONDS = 30
BUILD_TIMEOUT_SECONDS = 20 * 60
BROWSER_VERSION_TIMEOUT_SECONDS = 30
BROWSER_TIMEOUT_SECONDS = 180
CREATE_SUSPENDED = 0x00000004
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
WINDOWS_PATHEXT = ".COM;.EXE;.BAT;.CMD"


class _JobObjectBasicLimitInformation(ctypes.Structure):
    _fields_ = (
        ("per_process_user_time_limit", ctypes.c_int64),
        ("per_job_user_time_limit", ctypes.c_int64),
        ("limit_flags", wintypes.DWORD),
        ("minimum_working_set_size", ctypes.c_size_t),
        ("maximum_working_set_size", ctypes.c_size_t),
        ("active_process_limit", wintypes.DWORD),
        ("affinity", ctypes.c_size_t),
        ("priority_class", wintypes.DWORD),
        ("scheduling_class", wintypes.DWORD),
    )


class _IoCounters(ctypes.Structure):
    _fields_ = tuple(
        (name, ctypes.c_uint64)
        for name in (
            "read_operation_count",
            "write_operation_count",
            "other_operation_count",
            "read_transfer_count",
            "write_transfer_count",
            "other_transfer_count",
        )
    )


class _JobObjectExtendedLimitInformation(ctypes.Structure):
    _fields_ = (
        ("basic_limit_information", _JobObjectBasicLimitInformation),
        ("io_info", _IoCounters),
        ("process_memory_limit", ctypes.c_size_t),
        ("job_memory_limit", ctypes.c_size_t),
        ("peak_process_memory_used", ctypes.c_size_t),
        ("peak_job_memory_used", ctypes.c_size_t),
    )


class _WindowsJob:
    def __init__(self) -> None:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create_job = kernel32.CreateJobObjectW
        create_job.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
        create_job.restype = wintypes.HANDLE
        handle = create_job(None, None)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        self._handle: wintypes.HANDLE | None = handle
        information = _JobObjectExtendedLimitInformation()
        information.basic_limit_information.limit_flags = (
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
        set_information = kernel32.SetInformationJobObject
        set_information.argtypes = (
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        )
        set_information.restype = wintypes.BOOL
        if not set_information(
            handle,
            JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(information),
            ctypes.sizeof(information),
        ):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def assign_and_resume(self, process: subprocess.Popen[str]) -> None:
        if self._handle is None:
            raise OSError("process containment is already closed")
        raw_process_handle = getattr(process, "_handle", None)
        if raw_process_handle is None:
            raise OSError("spawned process handle is unavailable")
        process_handle = wintypes.HANDLE(int(raw_process_handle))
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        assign = kernel32.AssignProcessToJobObject
        assign.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
        assign.restype = wintypes.BOOL
        if not assign(self._handle, process_handle):
            raise ctypes.WinError(ctypes.get_last_error())
        ntdll = ctypes.WinDLL("ntdll")
        resume = ntdll.NtResumeProcess
        resume.argtypes = (wintypes.HANDLE,)
        resume.restype = ctypes.c_long
        status = int(resume(process_handle))
        if status < 0:
            raise OSError(
                "cannot resume contained process: "
                f"NTSTATUS 0x{status & 0xFFFF_FFFF:08x}"
            )

    def close(self) -> None:
        handle = self._handle
        if handle is None:
            return
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = (wintypes.HANDLE,)
        close_handle.restype = wintypes.BOOL
        if not close_handle(handle):
            raise ctypes.WinError(ctypes.get_last_error())
        self._handle = None


def _popen_contained(
    arguments: Sequence[str],
    **options: object,
) -> tuple[subprocess.Popen[str], _WindowsJob | None]:
    if os.name != "nt":
        options["start_new_session"] = True
        return subprocess.Popen(list(arguments), **options), None
    job = _WindowsJob()
    options["creationflags"] = int(options.get("creationflags", 0)) | getattr(
        subprocess, "CREATE_NEW_PROCESS_GROUP", 0
    ) | CREATE_SUSPENDED
    try:
        process: subprocess.Popen[str] = subprocess.Popen(
            list(arguments), **options
        )
    except BaseException:
        job.close()
        raise
    try:
        job.assign_and_resume(process)
    except BaseException as error:
        try:
            job.close()
        except BaseException as cleanup_error:
            error.add_note(f"job cleanup also failed: {cleanup_error!r}")
        try:
            process.kill()
            process.wait(timeout=PROCESS_TERMINATION_GRACE_SECONDS)
        except (OSError, subprocess.SubprocessError):
            pass
        _close_process_streams(process)
        raise
    return process, job


@dataclass(frozen=True)
class _ExecutableIdentity:
    path: Path
    sha256: str
    stat: tuple[int, int, int, int, int]


@dataclass(frozen=True)
class _PublishedIdentity:
    path: Path
    sha256: str
    stat: tuple[int, int, int, int, int]


def _stat_identity(path: Path) -> tuple[int, int, int, int, int]:
    status = path.stat()
    return (
        status.st_dev,
        status.st_ino,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def _authenticated_git() -> _ExecutableIdentity:
    raw_path = os.environ.get("DRAWBACK_AUTHENTICATED_GIT")
    expected_sha256 = os.environ.get("DRAWBACK_AUTHENTICATED_GIT_SHA256")
    if (
        raw_path is None
        or expected_sha256 is None
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ValueError("authenticated Git executable binding is unavailable")
    unresolved = Path(raw_path)
    if not unresolved.is_absolute() or unresolved.is_symlink():
        raise ValueError("authenticated Git executable path is invalid")
    try:
        path = unresolved.resolve(strict=True)
        payload = path.read_bytes()
        stat_identity = _stat_identity(path)
    except OSError as error:
        raise ValueError("authenticated Git executable is unavailable") from error
    if not path.is_file() or _sha256(payload) != expected_sha256:
        raise ValueError("authenticated Git executable digest differs")
    return _ExecutableIdentity(path, expected_sha256, stat_identity)


def _reauthenticate_git(identity: _ExecutableIdentity) -> None:
    if _authenticated_git() != identity:
        raise ValueError("authenticated Git executable changed during use")


def _sanitized_git_environment() -> Mapping[str, str]:
    environment = {
        "GIT_ASKPASS": os.devnull,
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "GCM_INTERACTIVE": "never",
        "SSH_ASKPASS": os.devnull,
        "SSH_ASKPASS_REQUIRE": "never",
    }
    if os.name == "nt":
        environment.update(_windows_process_environment())
    else:
        environment.update(
            {
                "LANG": "C",
                "LC_ALL": "C",
                "PATH": os.defpath,
            }
        )
    return environment


def _hardened_git_command(
    executable: Path,
    *arguments: str,
) -> list[str]:
    """Build a read-only Git command without executable config hooks."""

    command = [str(executable)]
    for key, value in (
        ("core.attributesFile", ""),
        ("core.excludesFile", ""),
        ("core.fsmonitor", "false"),
        ("core.hooksPath", os.devnull),
        ("core.askPass", os.devnull),
        ("credential.helper", ""),
        ("credential.interactive", "false"),
    ):
        command.extend(("-c", f"{key}={value}"))
    command.extend(arguments)
    return command


def _reject_executable_git_filters(
    executable: Path,
    *,
    repository: Path,
    environment: Mapping[str, str],
) -> None:
    """Reject effective local/worktree filters before Git reads the tree."""

    try:
        configured = _run_process(
            _hardened_git_command(
                executable,
                "-C",
                str(repository),
                "config",
                "--includes",
                "--show-scope",
                "--name-only",
                "--list",
            ),
            check=True,
            capture_output=True,
            timeout=GIT_TIMEOUT_SECONDS,
            environment=environment,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError(
            "cannot authenticate executable Git filter configuration"
        ) from error
    for line in configured.stdout.splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise ValueError(
                "Git filter configuration probe returned malformed output"
            )
        scope, key = fields[0].casefold(), fields[1].casefold()
        if (
            scope in {"local", "worktree"}
            and key.startswith("filter.")
            and key.rsplit(".", maxsplit=1)[-1]
            in {"clean", "smudge", "process"}
        ):
            raise ValueError(
                "repository config contains an executable Git filter"
            )


def _gitlink_paths(index: str) -> tuple[PurePosixPath, ...]:
    paths: list[PurePosixPath] = []
    for record in index.split("\0"):
        if not record:
            continue
        header, separator, raw_path = record.partition("\t")
        fields = header.split()
        if (
            not separator
            or len(fields) != 3
            or not raw_path
            or "\\" in raw_path
            or "\ufffd" in raw_path
        ):
            raise ValueError("Git index probe returned malformed output")
        if fields[0] != "160000" or fields[2] != "0":
            continue
        path = PurePosixPath(raw_path)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError("Git index contains an unsafe gitlink path")
        paths.append(path)
    return tuple(paths)


def _preflight_recursive_git_filters(
    executable: Path,
    *,
    repository: Path,
    environment: Mapping[str, str],
) -> tuple[Path, ...]:
    """Reject executable filters in every checked-out nested worktree."""

    try:
        root = repository.resolve(strict=True)
    except OSError as error:
        raise ValueError("cannot resolve repository for Git preflight") from error
    pending = [root]
    repositories: list[Path] = []
    seen: set[Path] = set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        try:
            top_level = _run_process(
                _hardened_git_command(
                    executable, "rev-parse", "--show-toplevel"
                ),
                cwd=current,
                check=True,
                capture_output=True,
                timeout=GIT_TIMEOUT_SECONDS,
                environment=environment,
            ).stdout.strip()
            authenticated_current = Path(top_level).resolve(strict=True)
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            raise ValueError(
                "cannot authenticate repository worktree root"
            ) from error
        if authenticated_current != current:
            raise ValueError("repository resolves to a different worktree root")
        _reject_executable_git_filters(
            executable,
            repository=current,
            environment=environment,
        )
        try:
            index = _run_process(
                _hardened_git_command(executable, "ls-files", "--stage", "-z"),
                cwd=current,
                check=True,
                capture_output=True,
                timeout=GIT_TIMEOUT_SECONDS,
                environment=environment,
            ).stdout
        except (OSError, subprocess.SubprocessError) as error:
            raise ValueError("cannot enumerate repository gitlinks") from error
        repositories.append(current)
        for relative in reversed(_gitlink_paths(index)):
            candidate = current.joinpath(*relative.parts)
            metadata = candidate / ".git"
            try:
                if not candidate.exists():
                    continue
                if (
                    not candidate.is_dir()
                    or candidate.is_symlink()
                ):
                    raise ValueError(
                        "checked-out gitlink has an unsafe worktree boundary"
                    )
                if not metadata.exists():
                    continue
                if metadata.is_symlink():
                    raise ValueError(
                        "checked-out gitlink has unsafe Git metadata"
                    )
                resolved = candidate.resolve(strict=True)
                metadata_resolved = metadata.resolve(strict=True)
            except OSError as error:
                raise ValueError(
                    "cannot authenticate checked-out gitlink worktree"
                ) from error
            if (
                resolved != Path(os.path.abspath(candidate))
                or not resolved.is_relative_to(root)
                or (
                    metadata.is_dir()
                    and metadata_resolved != Path(os.path.abspath(metadata))
                )
            ):
                raise ValueError("checked-out gitlink escapes the repository worktree")
            try:
                top_level = _run_process(
                    _hardened_git_command(
                        executable, "rev-parse", "--show-toplevel"
                    ),
                    cwd=resolved,
                    check=True,
                    capture_output=True,
                    timeout=GIT_TIMEOUT_SECONDS,
                    environment=environment,
                ).stdout.strip()
                authenticated = Path(top_level).resolve(strict=True)
            except (OSError, subprocess.SubprocessError) as error:
                raise ValueError(
                    "cannot authenticate checked-out gitlink repository"
                ) from error
            if authenticated != resolved:
                raise ValueError(
                    "checked-out gitlink resolves to a different worktree"
                )
            pending.append(resolved)
    return tuple(repositories)


def _close_process_streams(process: subprocess.Popen[str]) -> None:
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass


def _windows_runtime_paths() -> tuple[Path, Path, Path]:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    def known_directory(function_name: str) -> Path:
        buffer = ctypes.create_unicode_buffer(32_768)
        function = getattr(kernel32, function_name)
        function.argtypes = (wintypes.LPWSTR, wintypes.UINT)
        function.restype = wintypes.UINT
        length = function(buffer, len(buffer))
        if length == 0 or length >= len(buffer):
            raise OSError(f"{function_name} failed")
        directory = Path(buffer.value).resolve(strict=True)
        if not directory.is_dir():
            raise OSError(f"{function_name} returned a non-directory")
        return directory

    system_directory = known_directory("GetSystemDirectoryW")
    windows_directory = known_directory("GetWindowsDirectoryW")
    command_processor = (system_directory / "cmd.exe").resolve(strict=True)
    if (
        not command_processor.is_file()
        or command_processor.parent != system_directory
    ):
        raise OSError("Windows command processor is outside System32")
    return windows_directory, system_directory, command_processor


def _windows_process_environment(
    runtime_paths: tuple[Path, Path, Path] | None = None,
) -> dict[str, str]:
    windows_directory, system_directory, command_processor = (
        runtime_paths if runtime_paths is not None else _windows_runtime_paths()
    )
    return {
        "SystemRoot": str(windows_directory),
        "WINDIR": str(windows_directory),
        "ComSpec": str(command_processor),
        "PATH": str(system_directory),
        "PATHEXT": WINDOWS_PATHEXT,
    }


def _windows_taskkill(process_id: int) -> None:
    runtime_paths = _windows_runtime_paths()
    _, system_directory, _ = runtime_paths
    taskkill = (system_directory / "taskkill.exe").resolve(strict=True)
    if taskkill.parent != system_directory:
        raise OSError("taskkill resolved outside the Windows system directory")
    taskkill_identity = _stat_identity(taskkill)
    completed = subprocess.run(
        [str(taskkill), "/PID", str(process_id), "/T", "/F"],
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=PROCESS_TERMINATION_GRACE_SECONDS,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        env=_windows_process_environment(runtime_paths),
    )
    if _stat_identity(taskkill) != taskkill_identity:
        raise OSError("taskkill identity changed during process-tree cleanup")
    if completed.returncode != 0:
        raise OSError(
            f"taskkill failed with exit code {completed.returncode}"
        )


def _terminate_process_tree(
    process: subprocess.Popen[str],
    containment: _WindowsJob | None,
) -> None:
    cleanup_error: BaseException | None = None
    if os.name == "nt":
        try:
            if containment is None:
                _windows_taskkill(process.pid)
            else:
                containment.close()
        except (OSError, subprocess.SubprocessError) as error:
            cleanup_error = error
            try:
                process.kill()
            except OSError:
                pass
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except OSError:
            pass
        try:
            process.wait(timeout=PROCESS_TERMINATION_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            pass
    try:
        process.wait(timeout=PROCESS_TERMINATION_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except OSError:
            pass
        try:
            process.wait(timeout=PROCESS_TERMINATION_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            pass
    finally:
        _close_process_streams(process)
    if cleanup_error is not None:
        raise OSError("Windows process-tree cleanup failed") from cleanup_error


def _run_process(
    arguments: Sequence[str],
    *,
    cwd: Path | None = None,
    check: bool,
    capture_output: bool,
    timeout: int,
    environment: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    process, containment = _popen_contained(
        arguments,
        cwd=cwd,
        shell=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.PIPE if capture_output else None,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=dict(environment) if environment is not None else None,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except BaseException as error:
        try:
            _terminate_process_tree(process, containment)
        except BaseException as cleanup_error:
            error.add_note(f"process-tree cleanup also failed: {cleanup_error!r}")
        raise
    completed = subprocess.CompletedProcess(
        list(arguments), process.returncode, stdout, stderr
    )
    child_error = None
    if completed.returncode != 0:
        child_error = subprocess.CalledProcessError(
            completed.returncode,
            completed.args,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    try:
        _terminate_process_tree(process, containment)
    except BaseException as cleanup_error:
        if child_error is not None:
            child_error.add_note(
                f"process-tree cleanup also failed: {cleanup_error!r}"
            )
            raise child_error from cleanup_error
        raise
    if check and child_error is not None:
        raise child_error
    return completed


@dataclass(frozen=True)
class PublicParityGame:
    game_id: str
    seed: int
    pgn: str
    final_fen: str
    ply_count: int
    result: str
    features: FeatureRecord


def load_public_parity_fixture(
    path: Path, expected_sha256: str
) -> tuple[Mapping[str, object], tuple[PublicParityGame, ...]]:
    """Authenticate and independently replay the candidate-independent PGNs."""

    payload = path.read_bytes()
    if _sha256(payload) != expected_sha256:
        raise ValueError("public parity fixture SHA-256 does not match")
    value = _strict_json(payload, "public parity fixture")
    if payload != _canonical_pretty(value):
        raise ValueError("public parity fixture is not canonical")
    if set(value) != {
        "format",
        "version",
        "protocol",
        "candidateInputs",
        "games",
    } or value.get("format") != PUBLIC_FIXTURE_FORMAT or value.get("version") != 1:
        raise ValueError("public parity fixture contract is invalid")
    if value.get("candidateInputs") != []:
        raise ValueError("public parity fixture selection is candidate-dependent")
    protocol = value.get("protocol")
    expected_protocol = {
        "id": PUBLIC_FIXTURE_PROTOCOL_ID,
        "seedDomain": PUBLIC_FIXTURE_SEED_DOMAIN,
        "rootSeed": PUBLIC_FIXTURE_ROOT_SEED,
        "gameCount": PUBLIC_FIXTURE_GAME_COUNT,
        "maxPlies": PUBLIC_FIXTURE_MAX_PLIES,
        "agentSchedule": list(PUBLIC_FIXTURE_AGENTS),
    }
    if protocol != expected_protocol:
        raise ValueError("public parity generator protocol differs")
    games = value.get("games")
    if not isinstance(games, list) or len(games) != PUBLIC_FIXTURE_GAME_COUNT:
        raise ValueError("public parity game count differs")
    replayed: list[PublicParityGame] = []
    ids: set[str] = set()
    for raw in games:
        if not isinstance(raw, Mapping) or set(raw) != {
            "id",
            "seed",
            "pgn",
            "pgnSha256",
            "plyCount",
            "initialFen",
            "finalFen",
            "result",
            "finalPublicObservation",
        }:
            raise ValueError("public parity game fields are invalid")
        game_id = raw.get("id")
        pgn = raw.get("pgn")
        if (
            not isinstance(game_id, str)
            or not game_id
            or game_id in ids
            or not isinstance(pgn, str)
            or _sha256(pgn.encode("utf-8")) != raw.get("pgnSha256")
        ):
            raise ValueError("public parity game identity or PGN hash is invalid")
        parsed = chess.pgn.read_game(__import__("io").StringIO(pgn))
        if parsed is None:
            raise ValueError("public parity PGN cannot be parsed")
        board = parsed.board()
        plies = 0
        final_before = ""
        final_move = ""
        final_history: tuple[str, ...] = ()
        final_legal: tuple[str, ...] = ()
        history_san: list[str] = []
        try:
            for move in parsed.mainline_moves():
                if move not in board.legal_moves:
                    raise ValueError("public parity PGN contains an illegal move")
                final_before = board.fen(en_passant="legal")
                final_move = move.uci()
                final_history = tuple(history_san)
                final_legal = tuple(item.uci() for item in board.legal_moves)
                history_san.append(board.san(move))
                board.push(move)
                plies += 1
        except (ValueError, AssertionError) as error:
            raise ValueError("public parity PGN replay failed") from error
        if (
            parsed.errors
            or parsed.headers.get("Result") != raw.get("result")
            or parsed.headers.get("Result") not in {"1-0", "0-1", "1/2-1/2", "*"}
            or raw.get("initialFen")
            != "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
            or raw.get("finalFen") != board.fen(en_passant="legal")
            or raw.get("plyCount") != plies
            or plies > PUBLIC_FIXTURE_MAX_PLIES
        ):
            raise ValueError("public parity replay metadata differs")
        seed = raw.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("public parity seed is invalid")
        observation = _public_feature_record(
            raw.get("finalPublicObservation"),
            final_before=final_before,
            final_move=final_move,
            final_history=final_history,
            final_legal=final_legal,
            final_ply=plies - 1,
        )
        ids.add(game_id)
        replayed.append(
            PublicParityGame(
                game_id=game_id,
                seed=seed,
                pgn=pgn,
                final_fen=board.fen(en_passant="legal"),
                ply_count=plies,
                result=str(raw["result"]),
                features=observation,
            )
        )
    return value, tuple(replayed)


def _public_feature_record(
    value: object,
    *,
    final_before: str,
    final_move: str,
    final_history: tuple[str, ...],
    final_legal: tuple[str, ...],
    final_ply: int,
) -> FeatureRecord:
    if not isinstance(value, Mapping) or set(value) != {
        "fenBefore",
        "move",
        "moveNumber",
        "ply",
        "playerColor",
        "historySan",
        "ordinaryLegalMoves",
        "symbolicFeatureVersion",
        "symbolic",
    }:
        raise ValueError("public parity observation fields are invalid")
    if (
        value.get("fenBefore") != final_before
        or value.get("move") != final_move
        or value.get("historySan") != list(final_history)
        or not isinstance(value.get("ordinaryLegalMoves"), list)
        or set(value["ordinaryLegalMoves"]) != set(final_legal)
        or len(value["ordinaryLegalMoves"]) != len(final_legal)
        or value.get("ply") != final_ply
        or value.get("moveNumber") != final_ply // 2 + 1
        or value.get("playerColor") != ("white" if final_ply % 2 == 0 else "black")
        or value.get("symbolicFeatureVersion") != 6
    ):
        raise ValueError("public parity observation disagrees with PGN replay")
    symbolic = value.get("symbolic")
    if not isinstance(symbolic, Mapping) or set(symbolic) != {
        "ruleIds",
        "whiteProbabilities",
        "blackProbabilities",
        "whiteEliminated",
        "blackEliminated",
    } or symbolic.get("ruleIds") != list(SYMBOLIC_RULE_IDS):
        raise ValueError("public parity symbolic vocabulary differs")
    heads: dict[str, tuple[tuple[float, ...], tuple[bool, ...]]] = {}
    for color in ("white", "black"):
        probabilities = symbolic.get(f"{color}Probabilities")
        eliminated = symbolic.get(f"{color}Eliminated")
        if (
            not isinstance(probabilities, list)
            or not isinstance(eliminated, list)
            or len(probabilities) != len(SYMBOLIC_RULE_IDS)
            or len(eliminated) != len(SYMBOLIC_RULE_IDS)
            or any(not isinstance(item, bool) for item in eliminated)
        ):
            raise ValueError("public parity symbolic dimensions differ")
        converted: list[float] = []
        for probability, masked in zip(probabilities, eliminated, strict=True):
            if (
                isinstance(probability, bool)
                or not isinstance(probability, (int, float))
                or not math.isfinite(float(probability))
                or not 0 <= float(probability) <= 1
                or (masked and probability != 0)
            ):
                raise ValueError("public parity symbolic probability is invalid")
            converted.append(float(probability))
        if abs(math.fsum(converted) - 1.0) > TOLERANCE:
            raise ValueError("public parity symbolic probabilities are not normalized")
        heads[color] = (tuple(converted), tuple(eliminated))
    return FeatureRecord(
        fen_before=final_before,
        move=final_move,
        move_number=final_ply // 2 + 1,
        ply=final_ply,
        player_color="white" if final_ply % 2 == 0 else "black",
        history_san=final_history,
        ordinary_legal_moves=tuple(value["ordinaryLegalMoves"]),
        clock_ms=None,
        symbolic_feature_version=6,
        symbolic_white_rule_probabilities=heads["white"][0],
        symbolic_black_rule_probabilities=heads["black"][0],
        symbolic_white_eliminated=heads["white"][1],
        symbolic_black_eliminated=heads["black"][1],
        public_evaluator_constraint=None,
    )


def build_public_parity_input(
    *,
    fixture: ContentAddressedFile,
    ensemble_release: ContentAddressedJson,
    calibration: ContentAddressedFile,
    browser_artifact: ContentAddressedFile,
    repository: Path,
    output: Path,
) -> Mapping[str, object]:
    """Publish Python ensemble expectations over the immutable public fixture."""

    fixture_value, games = load_public_parity_fixture(
        fixture.path, fixture.sha256
    )
    release = verify_ensemble_release(ensemble_release)
    calibration_value = load_ensemble_calibration(calibration)
    _verify_calibration_binding(calibration_value, ensemble_release)
    fusion_alpha, fusion_selection_sha256 = _calibration_fusion_policy(
        calibration_value
    )
    artifact_payload = browser_artifact.path.read_bytes()
    if _sha256(artifact_payload) != browser_artifact.sha256:
        raise ValueError("browser artifact SHA-256 does not match")
    artifact = _strict_json(artifact_payload, "browser artifact")
    artifact_ensemble = artifact.get("ensemble")
    artifact_calibration = artifact.get("calibration")
    if (
        not isinstance(artifact_ensemble, Mapping)
        or artifact_ensemble.get("sourceEnsembleReleaseSha256")
        != ensemble_release.sha256
        or artifact_ensemble.get("sourceFusionSelectionSha256")
        != fusion_selection_sha256
        or artifact_ensemble.get("selectedAlpha") != fusion_alpha
        or not isinstance(artifact_calibration, Mapping)
        or artifact_calibration.get("sourceCalibrationSha256")
        != calibration.sha256
    ):
        raise ValueError("browser artifact binds a different candidate")
    temperatures = PromotionTemperatures(
        white=_calibration_temperature(calibration_value, "white"),
        black=_calibration_temperature(calibration_value, "black"),
    )
    cases: list[Mapping[str, object]] = []
    with ExitStack() as stack:
        checkpoints = _open_checkpoint_sources(stack, ensemble_release, release)
        payloads = tuple(
            _verified_checkpoint_bytes(source, member.checkpoint_sha256)
            for source, member in zip(checkpoints, release.members, strict=True)
        )
        loaded = load_hybrid_ensemble(
            checkpoints,
            device="cpu",
            fusion_alpha=fusion_alpha,
            required_corpus_provenance={
                "training_corpus_set_sha256": release.training_corpus_set_sha256
            },
        )
        represented = tuple(
            rule_id
            for rule_id in SYMBOLIC_RULE_IDS
            if rule_id not in UNAVAILABLE_BROWSER_RULE_IDS
        )
        represented_indices = tuple(
            index
            for index, rule_id in enumerate(SYMBOLIC_RULE_IDS)
            if rule_id in represented
        )
        for game in games:
            prepared = predict_calibrated_two_heads(
                members=loaded.members,
                features=game.features,
                temperatures=temperatures,
                fusion_alpha=fusion_alpha,
            )
            expected: dict[str, object] = {}
            for color in ("white", "black"):
                values = [prepared[color][index] for index in represented_indices]
                mass = math.fsum(values)
                probabilities = {
                    rule_id: value / mass
                    for rule_id, value in zip(represented, values, strict=True)
                }
                ordered = sorted(
                    represented,
                    key=lambda rule_id: (-probabilities[rule_id], rule_id),
                )
                expected[color] = {
                    "probabilities": probabilities,
                    "topIds": ordered[:TOP_K],
                    "hardZeroIds": sorted(
                        rule_id
                        for rule_id in represented
                        if probabilities[rule_id] == 0
                    ),
                }
            cases.append(
                {
                    "id": game.game_id,
                    "pgn": game.pgn,
                    "pgnSha256": _sha256(game.pgn.encode("utf-8")),
                    "expected": expected,
                }
            )
        for source, expected_payload in zip(checkpoints, payloads, strict=True):
            source.seek(0)
            if source.read() != expected_payload:
                raise ValueError("ensemble checkpoint changed during parity inference")
    partition: Mapping[str, object] = {
        "id": PUBLIC_FIXTURE_PROTOCOL_ID,
        "split": "validation-parity",
        "selectionSha256": _sha256(
            _canonical_pretty([game.game_id for game in games])
        ),
        "publicExampleCount": len(cases),
    }
    fixture_digest = _sha256(
        _canonical_pretty({"partition": partition, "cases": cases})
    )
    git = _authenticated_git()
    git_environment = _sanitized_git_environment()
    revision = _run_process(
        _hardened_git_command(
            git.path, "-C", str(repository), "rev-parse", "HEAD"
        ),
        check=True,
        capture_output=True,
        timeout=GIT_TIMEOUT_SECONDS,
        environment=git_environment,
    ).stdout.strip()
    _preflight_recursive_git_filters(
        git.path,
        repository=repository,
        environment=git_environment,
    )
    source_status = _run_process(
        _hardened_git_command(
            git.path,
            "-C",
            str(repository),
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--ignore-submodules=none",
            "--",
            *SOURCE_PATHS,
        ),
        check=True,
        capture_output=True,
        timeout=GIT_TIMEOUT_SECONDS,
        environment=git_environment,
    ).stdout.strip()
    _reauthenticate_git(git)
    if source_status:
        raise ValueError("parity input production requires a clean source HEAD")
    value: Mapping[str, object] = {
        "format": INPUT_FORMAT,
        "version": VERSION,
        "protocolId": PROTOCOL_ID,
        "browserArtifactSha256": browser_artifact.sha256,
        "fixtureSha256": fixture_digest,
        "partition": partition,
        "bindings": {
            "ensembleSha256": ensemble_release.sha256,
            "calibrationSha256": calibration.sha256,
            "fusionSelectionSha256": fusion_selection_sha256,
            "sourceRevision": revision,
            "pnpmLockSha256": _digest_file(repository / "pnpm-lock.yaml"),
        },
        "publicFixture": {
            "file": fixture.path.name,
            "sha256": fixture.sha256,
            "generatorProtocol": fixture_value["protocol"],
        },
        "cases": cases,
    }
    payload = _canonical_pretty(value)
    publish_bytes_durable(output, payload)
    return value


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _strict_json(payload: bytes, label: str) -> Mapping[str, object]:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(payload, object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} root must be an object")
    return value


def load_authenticated_input(
    path: Path, expected_sha256: str, artifact_sha256: str
) -> Mapping[str, object]:
    """Load a canonical, content-addressed validation-parity input."""

    payload = path.read_bytes()
    if _sha256(payload) != expected_sha256:
        raise ValueError("parity input SHA-256 does not match")
    value = _strict_json(payload, "parity input")
    if payload != _canonical_pretty(value):
        raise ValueError("parity input is not canonical")
    if set(value) != {
        "format",
        "version",
        "protocolId",
        "browserArtifactSha256",
        "fixtureSha256",
        "partition",
        "bindings",
        "publicFixture",
        "cases",
    }:
        raise ValueError("parity input fields are invalid")
    if (
        value.get("format") != INPUT_FORMAT
        or value.get("version") != VERSION
        or value.get("protocolId") != PROTOCOL_ID
        or value.get("browserArtifactSha256") != artifact_sha256
    ):
        raise ValueError("parity input identity is invalid")
    fixture_sha256 = value.get("fixtureSha256")
    if (
        not isinstance(fixture_sha256, str)
        or len(fixture_sha256) != 64
        or any(character not in "0123456789abcdef" for character in fixture_sha256)
    ):
        raise ValueError("parity fixture SHA-256 is invalid")
    partition = value.get("partition")
    bindings = value.get("bindings")
    public_fixture = value.get("publicFixture")
    cases = value.get("cases")
    if (
        not isinstance(partition, Mapping)
        or set(partition) != {
            "id",
            "split",
            "selectionSha256",
            "publicExampleCount",
        }
        or partition.get("split") != "validation-parity"
        or not isinstance(partition.get("publicExampleCount"), int)
        or int(partition["publicExampleCount"]) <= 0
        or not isinstance(bindings, Mapping)
        or set(bindings) != {
            "ensembleSha256",
            "calibrationSha256",
            "fusionSelectionSha256",
            "sourceRevision",
            "pnpmLockSha256",
        }
        or not isinstance(cases, list)
        or not cases
        or not isinstance(public_fixture, Mapping)
        or set(public_fixture) != {"file", "sha256", "generatorProtocol"}
        or public_fixture.get("generatorProtocol")
        != {
            "id": PUBLIC_FIXTURE_PROTOCOL_ID,
            "seedDomain": PUBLIC_FIXTURE_SEED_DOMAIN,
            "rootSeed": PUBLIC_FIXTURE_ROOT_SEED,
            "gameCount": PUBLIC_FIXTURE_GAME_COUNT,
            "maxPlies": PUBLIC_FIXTURE_MAX_PLIES,
            "agentSchedule": list(PUBLIC_FIXTURE_AGENTS),
        }
        or not is_portable_safe_basename(public_fixture.get("file"))
        or not isinstance(public_fixture.get("sha256"), str)
        or len(str(public_fixture["sha256"])) != 64
    ):
        raise ValueError("parity input authentication metadata is invalid")
    partition_id = partition.get("id")
    selection_sha256 = partition.get("selectionSha256")
    if (
        not isinstance(partition_id, str)
        or not partition_id.strip()
        or not isinstance(selection_sha256, str)
        or len(selection_sha256) != 64
        or any(character not in "0123456789abcdef" for character in selection_sha256)
        or partition.get("publicExampleCount") != len(cases)
    ):
        raise ValueError("parity partition identity or count is invalid")
    for key in SHA256_KEYS:
        digest = bindings.get(key)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"parity input {key} is invalid")
    revision = bindings.get("sourceRevision")
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise ValueError("parity input sourceRevision is invalid")
    expected_fixture = _sha256(
        _canonical_pretty({"partition": partition, "cases": cases})
    )
    if fixture_sha256 != expected_fixture:
        raise ValueError("parity fixture SHA-256 does not match partition and cases")
    # Inputs contain public PGN and Python posteriors only. Ground-truth
    # drawback labels, parameters, evaluator facts, and sealed metrics are
    # deliberately not accepted by this schema.
    forbidden = {"truth", "label", "hiddenParameters", "evaluatorFacts"}
    if any(key in forbidden for case in cases if isinstance(case, Mapping) for key in case):
        raise ValueError("parity input exposes sealed-test or hidden data")
    case_ids: set[str] = set()
    for case in cases:
        _validate_case(case)
        case_id = str(case["id"])
        if case_id in case_ids:
            raise ValueError("parity case ids are not unique")
        case_ids.add(case_id)
    return value


def _validate_case(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "id",
        "pgn",
        "pgnSha256",
        "expected",
    }:
        raise ValueError("parity case fields are invalid")
    if (
        not isinstance(value.get("id"), str)
        or not value["id"]
        or not isinstance(value.get("pgn"), str)
        or not value["pgn"].strip()
        or value.get("pgnSha256")
        != _sha256(str(value.get("pgn")).encode("utf-8"))
    ):
        raise ValueError("parity case identity or PGN is invalid")
    expected = value.get("expected")
    if not isinstance(expected, Mapping) or set(expected) != {"white", "black"}:
        raise ValueError("parity case expected heads are invalid")
    for color in ("white", "black"):
        head = expected.get(color)
        if (
            not isinstance(head, Mapping)
            or set(head) != {"probabilities", "topIds", "hardZeroIds"}
        ):
            raise ValueError(f"parity {color} head fields are invalid")
        probabilities = head.get("probabilities")
        top_ids = head.get("topIds")
        zero_ids = head.get("hardZeroIds")
        if (
            not isinstance(probabilities, Mapping)
            or not probabilities
            or not isinstance(top_ids, list)
            or len(top_ids) != TOP_K
            or not isinstance(zero_ids, list)
        ):
            raise ValueError(f"parity {color} head is invalid")
        ids = set(probabilities)
        if (
            any(not isinstance(rule_id, str) or not rule_id for rule_id in ids)
            or len(top_ids) != len(set(top_ids))
            or len(zero_ids) != len(set(zero_ids))
            or any(rule_id not in ids for rule_id in top_ids + zero_ids)
        ):
            raise ValueError(f"parity {color} rule ids are invalid")
        total = 0.0
        for rule_id, probability in probabilities.items():
            if (
                isinstance(probability, bool)
                or not isinstance(probability, (int, float))
                or not 0 <= float(probability) <= 1
            ):
                raise ValueError(f"parity {color} probability is invalid")
            total += float(probability)
            if (probability == 0) != (rule_id in zero_ids):
                raise ValueError(f"parity {color} hard-zero set is inconsistent")
        if abs(total - 1.0) > TOLERANCE:
            raise ValueError(f"parity {color} probabilities do not sum to one")
        ordered = sorted(ids, key=lambda rule_id: (-float(probabilities[rule_id]), rule_id))
        if top_ids != ordered[: len(top_ids)]:
            raise ValueError(f"parity {color} Top-k order is inconsistent")


def authenticate_runtime_bindings(
    repository: Path,
    artifact: Path,
    calibration: Path,
    parity_input: Mapping[str, object],
) -> None:
    bindings = parity_input["bindings"]
    if not isinstance(bindings, Mapping):
        raise ValueError("parity input bindings are invalid")
    artifact_value = _strict_json(artifact.read_bytes(), "browser artifact")
    ensemble = artifact_value.get("ensemble")
    calibration_section = artifact_value.get("calibration")
    if (
        not isinstance(ensemble, Mapping)
        or ensemble.get("sourceEnsembleReleaseSha256")
        != bindings.get("ensembleSha256")
        or ensemble.get("sourceFusionSelectionSha256")
        != bindings.get("fusionSelectionSha256")
        or not isinstance(calibration_section, Mapping)
        or calibration_section.get("sourceCalibrationSha256")
        != bindings.get("calibrationSha256")
        or _digest_file(calibration) != bindings.get("calibrationSha256")
        or _digest_file(repository / "pnpm-lock.yaml")
        != bindings.get("pnpmLockSha256")
    ):
        raise ValueError("candidate or dependency binding differs")
    git = _authenticated_git()
    git_environment = _sanitized_git_environment()
    revision = _run_process(
        _hardened_git_command(
            git.path, "-C", str(repository), "rev-parse", "HEAD"
        ),
        check=True,
        capture_output=True,
        timeout=GIT_TIMEOUT_SECONDS,
        environment=git_environment,
    ).stdout.strip()
    _preflight_recursive_git_filters(
        git.path,
        repository=repository,
        environment=git_environment,
    )
    dirty = _run_process(
        _hardened_git_command(
            git.path,
            "-C",
            str(repository),
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--ignore-submodules=none",
            "--",
            *SOURCE_PATHS,
        ),
        check=False,
        capture_output=True,
        timeout=GIT_TIMEOUT_SECONDS,
        environment=git_environment,
    )
    _reauthenticate_git(git)
    if (
        revision != bindings.get("sourceRevision")
        or dirty.returncode != 0
        or dirty.stdout.strip()
    ):
        raise ValueError("source revision is not the bound clean HEAD")


class _ResultParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.inside = False
        self.fragments: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag == "pre" and ("id", "browser-parity-result") in attrs:
            self.inside = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "pre":
            self.inside = False

    def handle_data(self, data: str) -> None:
        if self.inside:
            self.fragments.append(data)


@contextmanager
def _serve(directory: Path) -> Iterator[str]:
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, directory=str(directory), **kwargs)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/browser-parity.html"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def run_real_worker(
    repository: Path,
    browser: Path,
    artifact: Path,
    parity_input: Path,
) -> bytes:
    """Build the production app and capture a real-browser Worker transcript."""

    browser_payload = browser.read_bytes()
    browser_sha256 = _sha256(browser_payload)
    browser_identity = _stat_identity(browser)
    browser_version_process = _run_process(
        [str(browser), "--version"],
        check=True,
        capture_output=True,
        timeout=BROWSER_VERSION_TIMEOUT_SECONDS,
    )
    browser_version = (browser_version_process.stdout or "").strip()
    if not browser_version:
        raise ValueError("browser did not report a version")
    _run_process(
        ["pnpm", "--filter", "@drawbackguesser/web", "build"],
        cwd=repository,
        check=True,
        capture_output=False,
        timeout=BUILD_TIMEOUT_SECONDS,
    )
    dist = repository / "apps" / "web" / "dist"
    with tempfile.TemporaryDirectory(prefix="drawback-parity-") as temporary:
        webroot = Path(temporary) / "web"
        shutil.copytree(dist, webroot)
        shutil.copy2(artifact, webroot / "browser-model.json")
        shutil.copy2(parity_input, webroot / "browser-parity-input.json")
        with _serve(webroot) as url:
            completed = _run_process(
                [
                    str(browser),
                    "--headless=new",
                    "--disable-gpu",
                    "--no-first-run",
                    "--disable-background-networking",
                    "--virtual-time-budget=120000",
                    "--dump-dom",
                    url,
                ],
                check=False,
                capture_output=True,
                timeout=BROWSER_TIMEOUT_SECONDS,
            )
    if (
        _stat_identity(browser) != browser_identity
        or _sha256(browser.read_bytes()) != browser_sha256
    ):
        raise ValueError("browser executable changed during parity execution")
    if completed.returncode != 0:
        raise ValueError(
            f"headless browser failed with exit code {completed.returncode}"
        )
    parser = _ResultParser()
    parser.feed(completed.stdout or "")
    if not parser.fragments:
        raise ValueError("headless browser returned no parity transcript")
    value = dict(
        _strict_json("".join(parser.fragments).encode(), "worker transcript")
    )
    value["browserRuntime"] = {
        "binarySha256": browser_sha256,
        "version": browser_version,
    }
    return _canonical_pretty(value)


def _build_evidence_payload(
    *,
    transcript_payload: bytes,
    browser_artifact_sha256: str,
    calibration_sha256: str,
    parity_input: Mapping[str, object],
    parity_input_sha256: str,
) -> tuple[Mapping[str, object], bytes]:
    """Verify a Worker transcript and build its review projection."""

    transcript = _strict_json(transcript_payload, "worker transcript")
    if transcript_payload != _canonical_pretty(transcript):
        raise ValueError("worker transcript is not canonical")
    maximum = transcript.get("maximumAbsoluteDifference")
    browser_runtime = transcript.get("browserRuntime")
    passed = (
        transcript.get("format") == TRANSCRIPT_FORMAT
        and transcript.get("version") == VERSION
        and transcript.get("browserArtifactSha256")
        == browser_artifact_sha256
        and transcript.get("workerE2ePassed") is True
        and transcript.get("topKIdentical") is True
        and transcript.get("hardZeroSetsIdentical") is True
        and isinstance(maximum, (int, float))
        and not isinstance(maximum, bool)
        and 0 <= float(maximum) <= TOLERANCE
        and isinstance(browser_runtime, Mapping)
        and set(browser_runtime) == {"binarySha256", "version"}
        and isinstance(browser_runtime.get("binarySha256"), str)
        and len(str(browser_runtime["binarySha256"])) == 64
        and isinstance(browser_runtime.get("version"), str)
        and bool(str(browser_runtime["version"]).strip())
    )
    if not passed:
        raise ValueError("browser Worker parity transcript is not passing")
    evidence: Mapping[str, object] = {
        "format": EVIDENCE_FORMAT,
        "version": VERSION,
        "protocol_id": PROTOCOL_ID,
        "browser_artifact_sha256": browser_artifact_sha256,
        "calibration_sha256": calibration_sha256,
        "passed": True,
        "max_absolute_difference": float(maximum),
        "top_k_identical": True,
        "hard_zero_sets_identical": True,
        "worker_e2e_passed": True,
        "parity_input_sha256": parity_input_sha256,
        "transcript_sha256": _sha256(transcript_payload),
        "fixture_sha256": parity_input["fixtureSha256"],
        "partition_selection_sha256": parity_input["partition"][
            "selectionSha256"
        ],
        "ensemble_sha256": parity_input["bindings"]["ensembleSha256"],
        "source_revision": parity_input["bindings"]["sourceRevision"],
        "pnpm_lock_sha256": parity_input["bindings"]["pnpmLockSha256"],
        "browser_binary_sha256": browser_runtime["binarySha256"],
        "browser_version": browser_runtime["version"],
        "public_fixture_sha256": parity_input["publicFixture"]["sha256"],
    }
    payload = _canonical_pretty(evidence)
    return evidence, payload


def publish_evidence(
    *,
    transcript_payload: bytes,
    browser_artifact_sha256: str,
    calibration_sha256: str,
    parity_input: Mapping[str, object],
    parity_input_sha256: str,
    output: Path,
) -> Mapping[str, object]:
    """Verify and durably publish one create-only review projection."""

    evidence, payload = _build_evidence_payload(
        transcript_payload=transcript_payload,
        browser_artifact_sha256=browser_artifact_sha256,
        calibration_sha256=calibration_sha256,
        parity_input=parity_input,
        parity_input_sha256=parity_input_sha256,
    )
    publish_bytes_durable(output, payload)
    return evidence


def verify_transcript_bindings(
    transcript_payload: bytes, parity_input: Mapping[str, object]
) -> None:
    """Require the browser transcript to echo authenticated public bindings."""

    transcript = _strict_json(transcript_payload, "worker transcript")
    for transcript_key, input_key in (
        ("protocolId", "protocolId"),
        ("browserArtifactSha256", "browserArtifactSha256"),
        ("fixtureSha256", "fixtureSha256"),
        ("partition", "partition"),
        ("bindings", "bindings"),
        ("publicFixture", "publicFixture"),
    ):
        if transcript.get(transcript_key) != parity_input.get(input_key):
            raise ValueError(
                f"worker transcript {transcript_key} binding differs"
            )


def _digest_file(path: Path) -> str:
    return _sha256(path.read_bytes())


def _published_identity(path: Path, payload: bytes) -> _PublishedIdentity:
    if _digest_file(path) != _sha256(payload):
        raise OSError("published browser parity bytes changed immediately")
    return _PublishedIdentity(path, _sha256(payload), _stat_identity(path))


def _retain_owned_publication(identity: _PublishedIdentity) -> str:
    detail = "changed after publication"
    try:
        if (
            _stat_identity(identity.path) == identity.stat
            and _digest_file(identity.path) == identity.sha256
        ):
            detail = "is the exact partial publication"
    except OSError as error:
        detail = f"could not be reauthenticated: {error}"
    return (
        f"retained browser output ({detail}) because portable Python cannot "
        "unlink an authenticated object without a pathname race: "
        f"{identity.path}"
    )


def _authenticate_existing_publication(
    path: Path,
    payload: bytes,
    label: str,
) -> None:
    try:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"existing {label} is not a regular file")
        before = _stat_identity(path)
        observed = path.read_bytes()
        after = _stat_identity(path)
    except OSError as error:
        raise ValueError(f"existing {label} cannot be authenticated") from error
    if before != after or observed != payload:
        raise ValueError(f"existing {label} bytes do not match this run")


def _publish_browser_outputs(
    transcript_path: Path,
    transcript_payload: bytes,
    evidence_path: Path,
    evidence_payload: bytes,
) -> None:
    transcript_existed = transcript_path.exists()
    evidence_existed = evidence_path.exists()
    if transcript_existed:
        _authenticate_existing_publication(
            transcript_path, transcript_payload, "browser transcript"
        )
    if evidence_existed:
        _authenticate_existing_publication(
            evidence_path, evidence_payload, "browser evidence"
        )
    owned: list[_PublishedIdentity] = []
    try:
        if not transcript_existed:
            try:
                publish_bytes_durable(transcript_path, transcript_payload)
            except FileExistsError:
                _authenticate_existing_publication(
                    transcript_path,
                    transcript_payload,
                    "browser transcript",
                )
            else:
                owned.append(
                    _published_identity(transcript_path, transcript_payload)
                )
        if not evidence_existed:
            try:
                publish_bytes_durable(evidence_path, evidence_payload)
            except FileExistsError:
                _authenticate_existing_publication(
                    evidence_path,
                    evidence_payload,
                    "browser evidence",
                )
            else:
                owned.append(
                    _published_identity(evidence_path, evidence_payload)
                )
    except BaseException as error:
        for identity in reversed(owned):
            error.add_note(_retain_owned_publication(identity))
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run production browser Worker parity and publish evidence."
    )
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--browser", type=Path, required=True)
    parser.add_argument("--browser-artifact", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--transcript-output", type=Path, required=True)
    parser.add_argument("--evidence-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact_sha256 = _digest_file(args.browser_artifact)
    parity_input = load_authenticated_input(
        args.input, args.input_sha256, artifact_sha256
    )
    authenticate_runtime_bindings(
        args.repository.resolve(),
        args.browser_artifact.resolve(),
        args.calibration.resolve(),
        parity_input,
    )
    transcript = run_real_worker(
        args.repository.resolve(),
        args.browser.resolve(),
        args.browser_artifact.resolve(),
        args.input.resolve(),
    )
    verify_transcript_bindings(transcript, parity_input)
    _evidence, evidence_payload = _build_evidence_payload(
        transcript_payload=transcript,
        browser_artifact_sha256=artifact_sha256,
        calibration_sha256=_digest_file(args.calibration),
        parity_input=parity_input,
        parity_input_sha256=args.input_sha256,
    )
    _publish_browser_outputs(
        args.transcript_output,
        transcript,
        args.evidence_output,
        evidence_payload,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
