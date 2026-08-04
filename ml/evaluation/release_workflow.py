"""Closed, post-training release orchestration with no sealed-test access."""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
import signal
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import subprocess
import sys
import sysconfig
import threading
import time
from typing import Mapping, Sequence

from ml.training.drawback_ml.durable_publish import publish_bytes_durable
from ml.training.drawback_ml.path_validation import is_portable_safe_basename


FORMAT = "drawbacktrainer-post-training-release-workflow"
TRANSCRIPT_FORMAT = "drawbacktrainer-release-workflow-transcript"
VERSION = 3
SEEDS = (20260811, 20260812, 20260813)
EPOCHS = tuple(range(1, 9))
SELECTION_EVALUATION_WORKERS = 4
STEP_TIMEOUT_SECONDS = 6 * 60 * 60
PROCESS_POLL_SECONDS = 0.05
PROCESS_TERMINATION_GRACE_SECONDS = 5.0
GIT_TIMEOUT_SECONDS = 30
CREATE_SUSPENDED = 0x00000004
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
WINDOWS_PATHEXT = ".COM;.EXE;.BAT;.CMD"
SHA256 = frozenset("0123456789abcdef")
ISOLATED_MODULE_BOOTSTRAP = r"""
import os
import sys

python_controls = {
    name: value
    for name, value in os.environ.items()
    if name.upper().startswith("PYTHON")
}
if python_controls != {"PYTHONHASHSEED": "0"}:
    raise SystemExit("closed Python controls are invalid")
if (
    sys.flags.isolated != 0
    or sys.flags.ignore_environment != 0
    or sys.flags.no_site != 1
    or sys.flags.no_user_site != 1
    or sys.flags.safe_path != 1
    or sys.flags.dont_write_bytecode != 1
    or sys.flags.hash_randomization != 0
):
    raise SystemExit("closed Python flags are invalid")
if "sitecustomize" in sys.modules or "usercustomize" in sys.modules:
    raise SystemExit("Python startup customization was loaded")

sys.dont_write_bytecode = True
import runpy

root, purelib, platlib, stdlib, dynlib, module = sys.argv[1:7]
sys.path[:] = [root, purelib, platlib, stdlib, dynlib]
sys.argv = [module, *sys.argv[7:]]
runpy.run_module(module, run_name="__main__")
"""


class ReleaseWorkflowError(ValueError):
    pass


class _SelectionWaveCancelled(ReleaseWorkflowError):
    pass


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

    def assign_and_resume(
        self,
        process: subprocess.Popen[str] | subprocess.Popen[bytes],
    ) -> None:
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
) -> tuple[
    subprocess.Popen[str] | subprocess.Popen[bytes],
    _WindowsJob | None,
]:
    if os.name != "nt":
        options["start_new_session"] = True
        return subprocess.Popen(list(arguments), **options), None
    job = _WindowsJob()
    options["creationflags"] = int(options.get("creationflags", 0)) | getattr(
        subprocess, "CREATE_NEW_PROCESS_GROUP", 0
    ) | CREATE_SUSPENDED
    try:
        process = subprocess.Popen(list(arguments), **options)
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


class _SelectionWave:
    def __init__(self) -> None:
        self.stop = threading.Event()
        self._failure_lock = threading.Lock()
        self._primary_failure: BaseException | None = None

    def record_failure(self, error: BaseException) -> None:
        with self._failure_lock:
            if self._primary_failure is None:
                self._primary_failure = error
        self.stop.set()

    def primary_failure(self) -> BaseException | None:
        with self._failure_lock:
            return self._primary_failure


def _close_process_streams(
    process: subprocess.Popen[str] | subprocess.Popen[bytes],
) -> None:
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
    path_entries: Sequence[Path] = (),
    *,
    runtime_paths: tuple[Path, Path, Path] | None = None,
) -> dict[str, str]:
    windows_directory, system_directory, command_processor = (
        runtime_paths if runtime_paths is not None else _windows_runtime_paths()
    )
    paths = {str(path.resolve()) for path in path_entries}
    paths.add(str(system_directory))
    return {
        "SystemRoot": str(windows_directory),
        "WINDIR": str(windows_directory),
        "ComSpec": str(command_processor),
        "PATH": os.pathsep.join(sorted(paths, key=str.casefold)),
        "PATHEXT": WINDOWS_PATHEXT,
    }


def _windows_taskkill(process_id: int) -> None:
    runtime_paths = _windows_runtime_paths()
    _, system_directory, _ = runtime_paths
    taskkill = (system_directory / "taskkill.exe").resolve(strict=True)
    if taskkill.parent != system_directory:
        raise OSError("taskkill resolved outside the Windows system directory")
    taskkill_identity = _stable_file_identity(taskkill)
    completed = subprocess.run(
        [str(taskkill), "/PID", str(process_id), "/T", "/F"],
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=PROCESS_TERMINATION_GRACE_SECONDS,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        env=_windows_process_environment(runtime_paths=runtime_paths),
    )
    if _stable_file_identity(taskkill) != taskkill_identity:
        raise OSError("taskkill identity changed during process-tree cleanup")
    if completed.returncode != 0:
        raise OSError(
            f"taskkill failed with exit code {completed.returncode}"
        )


def _terminate_process_tree(
    process: subprocess.Popen[str] | subprocess.Popen[bytes],
    containment: _WindowsJob | None,
) -> None:
    cleanup_error: BaseException | None = None
    if os.name == "nt":
        try:
            if containment is None:
                _windows_taskkill(process.pid)
            else:
                containment.close()
        except (
            OSError,
            subprocess.SubprocessError,
            ReleaseWorkflowError,
        ) as error:
            cleanup_error = error
            try:
                process.kill()
            except OSError:
                pass
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass
        try:
            process.wait(timeout=PROCESS_TERMINATION_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            pass
        # The direct child can exit before a descendant that ignored SIGTERM.
        # Always force the remaining process group down before settlement.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
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


def _run_step_process(
    arguments: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    cancel: threading.Event | None,
) -> None:
    process, containment = _popen_contained(
        arguments,
        cwd=cwd,
        shell=False,
        env=dict(environment),
    )
    deadline = time.monotonic() + STEP_TIMEOUT_SECONDS
    try:
        while True:
            if cancel is not None and cancel.is_set():
                raise _SelectionWaveCancelled(
                    "selection evaluation wave was cancelled"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(arguments, STEP_TIMEOUT_SECONDS)
            try:
                return_code = process.wait(
                    timeout=min(PROCESS_POLL_SECONDS, remaining)
                )
            except subprocess.TimeoutExpired:
                continue
            break
    except BaseException as error:
        try:
            _terminate_process_tree(process, containment)
        except BaseException as cleanup_error:
            error.add_note(f"process-tree cleanup also failed: {cleanup_error!r}")
        raise
    child_error = (
        subprocess.CalledProcessError(return_code, arguments)
        if return_code != 0
        else None
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
    if child_error is not None:
        raise child_error


def _run_capture_process(
    arguments: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    process, containment = _popen_contained(
        arguments,
        cwd=cwd,
        shell=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=dict(environment),
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
    if child_error is not None:
        raise child_error
    return completed


@dataclass(frozen=True)
class Step:
    stage: str
    argv: tuple[str, ...]
    outputs: tuple[Path, ...]
    inputs: tuple[ArtifactRef, ...] = ()
    generated_inputs: tuple[Path, ...] = ()
    seed: int | None = None
    epoch: int | None = None


@dataclass(frozen=True)
class ArtifactRef:
    path: Path
    sha256: str


@dataclass(frozen=True)
class ExternalRef:
    path: Path
    sha256: str


def _object(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
        raise ReleaseWorkflowError(f"{label} must be an object")
    return value


def _exact(value: Mapping[str, object], fields: set[str], label: str) -> None:
    if set(value) != fields:
        raise ReleaseWorkflowError(f"{label} fields are invalid")


def _pairs(items: Sequence[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in items:
        if key in value:
            raise ReleaseWorkflowError(
                f"workflow JSON repeats key {key!r}"
            )
        value[key] = item
    return value


def _constant(token: str) -> object:
    raise ReleaseWorkflowError(
        f"workflow JSON contains non-finite constant {token}"
    )


def _relative(value: object, label: str) -> Path:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or ":" in value
    ):
        raise ReleaseWorkflowError(f"{label} must be a normalized POSIX path")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or ".." in pure.parts
        or "." in pure.parts
        or any(not is_portable_safe_basename(part) for part in pure.parts)
    ):
        raise ReleaseWorkflowError(f"{label} escapes the repository")
    normalized = pure.as_posix()
    if normalized != value:
        raise ReleaseWorkflowError(f"{label} is not normalized")
    return Path(*pure.parts)


def _portable_path_key(path: Path) -> tuple[str, ...]:
    if path.is_absolute() or not path.parts or any(
        not is_portable_safe_basename(part) for part in path.parts
    ):
        raise ReleaseWorkflowError("release artifact path is not portable")
    return tuple(part.casefold() for part in path.parts)


def _digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in SHA256 for character in value)
    ):
        raise ReleaseWorkflowError(f"{label} must be a lowercase SHA-256")
    return value


def _reference(value: object, label: str) -> ArtifactRef:
    item = _object(value, label)
    _exact(item, {"path", "sha256"}, label)
    return ArtifactRef(
        path=_relative(item["path"], f"{label}.path"),
        sha256=_digest(item["sha256"], f"{label}.sha256"),
    )


def _external_reference(value: object, label: str) -> ExternalRef:
    item = _object(value, label)
    _exact(item, {"path", "sha256"}, label)
    raw_path = item["path"]
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise ReleaseWorkflowError(f"{label}.path must be absolute")
    return ExternalRef(
        path=Path(raw_path),
        sha256=_digest(item["sha256"], f"{label}.sha256"),
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _confined(root: Path, relative: Path, *, must_exist: bool) -> Path:
    candidate = root.joinpath(relative)
    current = root
    for part in relative.parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise ReleaseWorkflowError(f"workflow path traverses symlink: {relative}")
    resolved = candidate.resolve(strict=must_exist)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ReleaseWorkflowError(f"workflow path escapes repository: {relative}") from error
    return resolved


def load_workflow(path: Path) -> tuple[Mapping[str, object], str]:
    try:
        payload = path.read_bytes()
        value = _object(
            json.loads(
                payload,
                object_pairs_hook=_pairs,
                parse_constant=_constant,
            ),
            "workflow",
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReleaseWorkflowError(
            f"cannot load strict UTF-8 workflow JSON: {path}"
        ) from error
    _exact(value, {
        "format", "version", "sourceRevision", "realDomainExecution",
        "tools", "shared", "candidates", "selectionOutputs", "ensemble",
        "calibration", "trainingFrequency", "validation", "browserRelease",
        "transcriptOutput",
    }, "workflow")
    if value["format"] != FORMAT or value["version"] != VERSION:
        raise ReleaseWorkflowError("workflow identity is invalid")
    revision = value["sourceRevision"]
    if not isinstance(revision, str) or len(revision) != 40 or any(
        c not in "0123456789abcdef" for c in revision
    ):
        raise ReleaseWorkflowError("sourceRevision must be a full lowercase Git SHA")
    if value["realDomainExecution"] != "external-isolated-only":
        raise ReleaseWorkflowError(
            "real-domain execution is outside this runner and requires external isolation"
        )
    canonical = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode() + b"\n"
    if payload != canonical:
        raise ReleaseWorkflowError("workflow must be canonical JSON")
    return value, hashlib.sha256(payload).hexdigest()


def _path_fields(value: object, fields: set[str], label: str) -> dict[str, Path]:
    obj = _object(value, label)
    _exact(obj, fields, label)
    return {field: _relative(obj[field], f"{label}.{field}") for field in fields}


def build_plan(workflow: Mapping[str, object], root: Path) -> tuple[Step, ...]:
    shared_raw = _object(workflow["shared"], "shared")
    _exact(
        shared_raw,
        {"dataset", "publicRoot", "privateValidation"},
        "shared",
    )
    shared = {
        name: _reference(shared_raw[name], f"shared.{name}")
        for name in ("dataset", "publicRoot", "privateValidation")
    }
    candidates = workflow["candidates"]
    if not isinstance(candidates, list) or len(candidates) != 3:
        raise ReleaseWorkflowError("candidates must contain exactly three seeds")
    candidate_data: dict[
        int, tuple[ArtifactRef, list[Mapping[str, object]]]
    ] = {}
    candidate_paths: set[tuple[str, ...]] = set()
    input_references: list[ArtifactRef] = list(shared.values())
    for index, raw in enumerate(candidates):
        item = _object(raw, f"candidate {index}")
        _exact(item, {"seed", "trainingRun", "epochs"}, f"candidate {index}")
        seed = item["seed"]
        if seed != SEEDS[index]:
            raise ReleaseWorkflowError("candidate seed order is invalid")
        training_run = _reference(
            item["trainingRun"], f"candidate {index}.trainingRun"
        )
        input_references.append(training_run)
        epochs = item["epochs"]
        if not isinstance(epochs, list) or len(epochs) != 8:
            raise ReleaseWorkflowError("candidate must contain eight epochs")
        for epoch_index, raw_epoch in enumerate(epochs, 1):
            epoch = _object(raw_epoch, "epoch")
            _exact(epoch, {"epoch", "checkpoint", "report", "summary"}, "epoch")
            if epoch["epoch"] != epoch_index:
                raise ReleaseWorkflowError("epoch order is invalid")
            checkpoint = _reference(
                epoch["checkpoint"],
                f"candidate {index}.epoch {epoch_index}.checkpoint",
            )
            input_references.append(checkpoint)
            for name in ("report", "summary"):
                candidate_path = _relative(epoch[name], f"epoch.{name}")
                candidate_key = _portable_path_key(candidate_path)
                if candidate_key in candidate_paths:
                    raise ReleaseWorkflowError(
                        "checkpoint/report/summary paths must be distinct"
                    )
                candidate_paths.add(candidate_key)
            checkpoint_key = _portable_path_key(checkpoint.path)
            if checkpoint_key in candidate_paths:
                raise ReleaseWorkflowError(
                    "checkpoint/report/summary paths must be distinct"
                )
            candidate_paths.add(checkpoint_key)
        candidate_data[seed] = (training_run, epochs)
    selection_raw = _object(workflow["selectionOutputs"], "selectionOutputs")
    _exact(selection_raw, {str(seed) for seed in SEEDS}, "selectionOutputs")
    selections = {
        seed: _relative(selection_raw[str(seed)], f"selectionOutputs.{seed}")
        for seed in SEEDS
    }
    if len({_portable_path_key(path) for path in selections.values()}) != len(
        SEEDS
    ):
        raise ReleaseWorkflowError("selection output paths must be distinct")
    ensemble = _path_fields(workflow["ensemble"], {"output"}, "ensemble")
    fusion_selection = ensemble["output"].with_name("fusion-selection.json")
    calibration = _path_fields(
        workflow["calibration"],
        {"report", "sidecar", "receipt", "output"},
        "calibration",
    )
    frequency = _reference(
        workflow["trainingFrequency"], "trainingFrequency"
    )
    input_references.append(frequency)
    validation = _path_fields(
        workflow["validation"], {"report", "decision"}, "validation"
    )
    browser_release_raw = _object(
        workflow["browserRelease"], "browserRelease"
    )
    _exact(
        browser_release_raw,
        {"fixture", "artifact", "parityInput", "transcript", "evidence"},
        "browserRelease",
    )
    fixture = _reference(
        browser_release_raw["fixture"], "browserRelease.fixture"
    )
    input_references.append(fixture)
    browser_release = {
        name: _relative(
            browser_release_raw[name], f"browserRelease.{name}"
        )
        for name in ("artifact", "parityInput", "transcript", "evidence")
    }
    transcript = _relative(workflow["transcriptOutput"], "transcriptOutput")
    tools = _object(workflow["tools"], "tools")
    _exact(tools, {"browser", "git", "node", "pnpm"}, "tools")
    browser = _external_reference(tools["browser"], "tools.browser")
    _external_reference(tools["git"], "tools.git")
    _external_reference(tools["node"], "tools.node")
    _external_reference(tools["pnpm"], "tools.pnpm")

    def p(path: Path) -> str:
        return str(path)

    def digest(path: Path) -> str:
        absolute = _confined(root, path, must_exist=False)
        return _sha(absolute) if absolute.is_file() else f"<sha256:{path.as_posix()}>"

    purelib = sysconfig.get_path("purelib")
    platlib = sysconfig.get_path("platlib")
    stdlib = sysconfig.get_path("stdlib")
    if not purelib or not platlib or not stdlib:
        raise ReleaseWorkflowError("Python library paths are unavailable")
    configured_dynlib = sysconfig.get_config_var("DESTSHARED")
    dynlib = (
        str(configured_dynlib)
        if isinstance(configured_dynlib, str) and configured_dynlib
        else str(Path(stdlib).parent / "DLLs")
    )

    def command(module: str, *arguments: str) -> tuple[str, ...]:
        return (
            sys.executable,
            "-B",
            "-s",
            "-S",
            "-P",
            "-c",
            ISOLATED_MODULE_BOOTSTRAP,
            str(root.resolve()),
            purelib,
            platlib,
            stdlib,
            dynlib,
            module,
            *arguments,
        )

    steps: list[Step] = []
    for seed in SEEDS:
        training_run, epochs = candidate_data[seed]
        for epoch in epochs:
            number = int(epoch["epoch"])
            checkpoint = _reference(epoch["checkpoint"], "checkpoint")
            report = _relative(epoch["report"], "report")
            steps.append(Step(
                "selection-fit-evaluation",
                command("ml.evaluation.cli", p(checkpoint.path), p(shared["dataset"].path),
                 "--public-root", p(shared["publicRoot"].path), "--private-validation",
                 p(shared["privateValidation"].path), "--split", "validation",
                 "--validation-partition", "selection", "--output", p(report)),
                (report,),
                (checkpoint, *shared.values()),
                seed=seed,
                epoch=number,
            ))
        for epoch in epochs:
            number = int(epoch["epoch"])
            checkpoint = _reference(epoch["checkpoint"], "checkpoint")
            report = _relative(epoch["report"], "report")
            summary = _relative(epoch["summary"], "summary")
            steps.append(Step(
                "selection-summary",
                command("ml.evaluation.cli", "emit-selection-summary",
                 p(report), p(checkpoint.path), p(summary), "--training-seed", str(seed),
                 "--epoch", str(number), "--training-run", p(training_run.path),
                 "--training-run-sha256", training_run.sha256),
                (summary,),
                (checkpoint, training_run),
                (report,),
                seed,
                number,
            ))
    for seed in SEEDS:
        training_run, epochs = candidate_data[seed]
        argv = list(command(
            "ml.evaluation.cli", "select-epoch", p(selections[seed])
        ))
        for epoch in epochs:
            summary = _relative(epoch["summary"], "summary")
            argv += ["--summary", p(summary), "--summary-sha256", digest(summary)]
        steps.append(Step(
            "epoch-selection",
            tuple(argv),
            (selections[seed],),
            (),
            tuple(
                _relative(epoch["summary"], "summary")
                for epoch in epochs
            ),
            seed,
        ))
    argv = list(command(
        "ml.evaluation.cli",
        "create-ensemble-release",
        p(ensemble["output"]),
    ))
    for seed in SEEDS:
        training_run, _ = candidate_data[seed]
        argv += ["--selection", p(selections[seed]), "--selection-sha256", digest(selections[seed])]
        argv += ["--training-run", p(training_run.path), "--training-run-sha256", training_run.sha256]
    steps.append(Step(
        "ensemble-release",
        tuple(argv),
        (ensemble["output"],),
        tuple(
            candidate_data[seed][0] for seed in SEEDS
        ),
        tuple(selections[seed] for seed in SEEDS),
    ))
    steps.append(Step(
        "fusion-selection",
        command(
            "ml.evaluation.cli",
            "select-ensemble-fusion",
            p(ensemble["output"]),
            p(shared["dataset"].path),
            p(fusion_selection),
            "--ensemble-sha256",
            digest(ensemble["output"]),
            "--public-root",
            p(shared["publicRoot"].path),
            "--private-validation",
            p(shared["privateValidation"].path),
        ),
        (fusion_selection,),
        tuple(shared.values()),
        (ensemble["output"],),
    ))
    steps.append(Step(
        "calibration-evaluation",
        command("ml.evaluation.cli", "evaluate-ensemble-calibration",
         p(ensemble["output"]), p(shared["dataset"].path), "--ensemble-sha256",
         digest(ensemble["output"]), "--public-root", p(shared["publicRoot"].path),
         "--private-validation", p(shared["privateValidation"].path),
         "--fusion-selection", p(fusion_selection),
         "--fusion-selection-sha256", digest(fusion_selection), "--output",
         p(calibration["report"]), "--sidecar-output", p(calibration["sidecar"])),
        (calibration["report"], calibration["sidecar"]),
        tuple(shared.values()),
        (ensemble["output"], fusion_selection),
    ))
    steps.append(Step(
        "calibration-fit",
        command("ml.evaluation.cli", "fit-ensemble-calibration",
         p(calibration["sidecar"]), p(calibration["report"]), p(ensemble["output"]),
         p(calibration["receipt"]), p(calibration["output"]), "--sidecar-sha256",
         digest(calibration["sidecar"]), "--report-sha256", digest(calibration["report"]),
         "--ensemble-sha256", digest(ensemble["output"]),
         "--fusion-selection", p(fusion_selection),
         "--fusion-selection-sha256", digest(fusion_selection)),
        (calibration["receipt"], calibration["output"]),
        (),
        (
            calibration["sidecar"],
            calibration["report"],
            ensemble["output"],
            fusion_selection,
        ),
    ))
    steps.append(Step(
        "validation-gate",
        command("ml.evaluation.validation_gate", p(ensemble["output"]),
         p(calibration["output"]), p(frequency.path), p(shared["dataset"].path),
         "--ensemble-sha256", digest(ensemble["output"]), "--calibration-sha256",
         digest(calibration["output"]), "--training-frequency-sha256",
         frequency.sha256, "--public-root", p(shared["publicRoot"].path),
         "--private-validation", p(shared["privateValidation"].path), "--report-output",
         p(validation["report"]), "--decision-output", p(validation["decision"])),
        (validation["report"], validation["decision"]),
        (*shared.values(), frequency),
        (ensemble["output"], calibration["output"]),
    ))
    steps.append(Step(
        "browser-artifact",
        command("ml.evaluation.cli", "export-browser-ensemble",
         p(ensemble["output"]), p(calibration["output"]), p(browser_release["artifact"]),
         "--ensemble-sha256", digest(ensemble["output"]), "--calibration-sha256",
         digest(calibration["output"])),
        (browser_release["artifact"],),
        (),
        (ensemble["output"], calibration["output"]),
    ))
    steps.append(Step(
        "browser-parity-input",
        command("ml.evaluation.browser_parity_input", p(fixture.path),
         p(ensemble["output"]), p(calibration["output"]), p(browser_release["artifact"]),
         "--fixture-sha256", fixture.sha256, "--ensemble-sha256",
         digest(ensemble["output"]), "--calibration-sha256", digest(calibration["output"]),
         "--browser-artifact-sha256", digest(browser_release["artifact"]),
         "--repository", ".", "--output", p(browser_release["parityInput"])),
        (browser_release["parityInput"],),
        (fixture,),
        (
            ensemble["output"],
            calibration["output"],
            browser_release["artifact"],
        ),
    ))
    steps.append(Step(
        "browser-parity",
        command("ml.evaluation.browser_parity", "--repository", ".",
         "--browser", str(browser.path), "--browser-artifact", p(browser_release["artifact"]),
         "--calibration", p(calibration["output"]), "--input",
         p(browser_release["parityInput"]), "--input-sha256",
         digest(browser_release["parityInput"]), "--transcript-output",
         p(browser_release["transcript"]), "--evidence-output",
         p(browser_release["evidence"])),
        (browser_release["transcript"], browser_release["evidence"]),
        (),
        (
            browser_release["artifact"],
            calibration["output"],
            browser_release["parityInput"],
        ),
    ))
    input_paths = [reference.path for reference in input_references]
    output_paths = [
        *(output for step in steps for output in step.outputs),
        transcript,
    ]
    input_keys = {_portable_path_key(path) for path in input_paths}
    output_keys = {_portable_path_key(path) for path in output_paths}
    if len(input_keys) != len(input_paths):
        raise ReleaseWorkflowError("input artifact paths must be distinct")
    if len(output_keys) != len(output_paths):
        raise ReleaseWorkflowError("output artifact paths must be distinct")
    if input_keys.intersection(output_keys):
        raise ReleaseWorkflowError(
            "release outputs must not overwrite input artifacts"
        )
    for declared_path in (*input_paths, *output_paths):
        _confined(root, declared_path, must_exist=False)
    # Real-domain Stage A/B intentionally remain outside this process. Their
    # mandatory no-label mount receipt is an external isolation boundary.
    return tuple(steps)


def _input_references(workflow: Mapping[str, object]) -> tuple[ArtifactRef, ...]:
    shared = _object(workflow["shared"], "shared")
    references = [
        _reference(shared[name], f"shared.{name}")
        for name in ("dataset", "publicRoot", "privateValidation")
    ]
    candidates = workflow["candidates"]
    if not isinstance(candidates, list):
        raise ReleaseWorkflowError("candidates are unavailable")
    for index, raw in enumerate(candidates):
        candidate = _object(raw, f"candidate {index}")
        references.append(
            _reference(
                candidate["trainingRun"],
                f"candidate {index}.trainingRun",
            )
        )
        epochs = candidate["epochs"]
        if not isinstance(epochs, list):
            raise ReleaseWorkflowError("candidate epochs are unavailable")
        for epoch_index, raw_epoch in enumerate(epochs, 1):
            epoch = _object(raw_epoch, f"candidate {index}.epoch {epoch_index}")
            references.append(
                _reference(
                    epoch["checkpoint"],
                    f"candidate {index}.epoch {epoch_index}.checkpoint",
                )
            )
    references.append(
        _reference(workflow["trainingFrequency"], "trainingFrequency")
    )
    browser_release = _object(
        workflow["browserRelease"], "browserRelease"
    )
    references.append(
        _reference(
            browser_release["fixture"], "browserRelease.fixture"
        )
    )
    return tuple(references)


def _external_tools(
    workflow: Mapping[str, object],
) -> Mapping[str, ExternalRef]:
    tools = _object(workflow["tools"], "tools")
    return {
        name: _external_reference(tools[name], f"tools.{name}")
        for name in ("browser", "git", "node", "pnpm")
    }


def _authenticate_input(root: Path, reference: ArtifactRef) -> Path:
    try:
        absolute = _confined(root, reference.path, must_exist=True)
    except OSError as error:
        raise ReleaseWorkflowError(
            f"release input is missing: {reference.path}"
        ) from error
    if (
        not absolute.is_file()
        or absolute.is_symlink()
        or _sha(absolute) != reference.sha256
    ):
        raise ReleaseWorkflowError(
            f"release input authentication failed: {reference.path}"
        )
    return absolute


def _authenticate_external(reference: ExternalRef, label: str) -> Path:
    try:
        path = reference.path.resolve(strict=True)
    except OSError as error:
        raise ReleaseWorkflowError(f"{label} is missing") from error
    if not path.is_file() or _sha(path) != reference.sha256:
        raise ReleaseWorkflowError(f"{label} authentication failed")
    return path


def _stat_result_identity(
    status: os.stat_result,
) -> tuple[int, int, int, int, int]:
    return (
        status.st_dev,
        status.st_ino,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def _stable_file_identity(path: Path) -> tuple[int, int, int, int, int]:
    try:
        status = path.stat()
    except OSError as error:
        raise ReleaseWorkflowError(
            "external executable identity cannot be measured"
        ) from error
    return _stat_result_identity(status)


def _rollback_new_selection_outputs(
    output_paths: Sequence[Path],
) -> tuple[str, ...]:
    """Retain partial outputs because portable handle-bound unlink is absent.

    A pathname can be replaced after any stat/open authentication and before
    ``unlink``. Retention is therefore the only portable fail-closed rollback;
    the caller reports every retained path for explicit operator cleanup.
    """

    retained: list[str] = []
    for path in output_paths:
        try:
            path.lstat()
        except FileNotFoundError:
            continue
        except OSError as error:
            retained.append(f"cannot inspect retained output {path}: {error}")
            continue
        retained.append(
            "retained selection output because portable Python cannot unlink "
            f"an authenticated object without a pathname race: {path}"
        )
    return tuple(retained)


def _authenticate_generated(root: Path, relative: Path) -> tuple[Path, str]:
    try:
        path = _confined(root, relative, must_exist=True)
    except OSError as error:
        raise ReleaseWorkflowError(
            f"generated release input is missing: {relative}"
        ) from error
    if not path.is_file() or path.is_symlink():
        raise ReleaseWorkflowError(
            f"generated release input is invalid: {relative}"
        )
    return path, _sha(path)


def _closed_environment(
    root: Path,
    tools: Mapping[str, Path],
) -> Mapping[str, str]:
    path_entries = {path.parent for path in tools.values()}
    environment = {
        "PYTHONHASHSEED": "0",
        "TEMP": str(root),
        "TMP": str(root),
        "GIT_ASKPASS": os.devnull,
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "GCM_INTERACTIVE": "never",
        "SSH_ASKPASS": os.devnull,
        "SSH_ASKPASS_REQUIRE": "never",
    }
    if os.name == "nt":
        environment.update(_windows_process_environment(tuple(path_entries)))
    else:
        environment["PATH"] = os.pathsep.join(
            sorted((str(path.resolve()) for path in path_entries))
        )
    git = tools.get("git")
    if git is not None:
        environment["DRAWBACK_AUTHENTICATED_GIT"] = str(git)
        environment["DRAWBACK_AUTHENTICATED_GIT_SHA256"] = _sha(git)
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
        configured = _run_capture_process(
            _git_filter_config_probe_command(executable),
            cwd=repository,
            environment=environment,
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ReleaseWorkflowError(
            "cannot authenticate executable Git filter configuration"
        ) from error
    _assert_no_executable_git_filters(configured.stdout)


def _git_filter_config_probe_command(executable: Path) -> list[str]:
    return _hardened_git_command(
        executable,
        "config",
        "--includes",
        "--show-scope",
        "--name-only",
        "--list",
    )


def _assert_no_executable_git_filters(configured: str) -> None:
    for line in configured.splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise ReleaseWorkflowError(
                "Git filter configuration probe returned malformed output"
            )
        scope, key = fields[0].casefold(), fields[1].casefold()
        if (
            scope in {"local", "worktree"}
            and key.startswith("filter.")
            and key.rsplit(".", maxsplit=1)[-1]
            in {"clean", "smudge", "process"}
        ):
            raise ReleaseWorkflowError(
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
            raise ReleaseWorkflowError("Git index probe returned malformed output")
        if fields[0] != "160000" or fields[2] != "0":
            continue
        path = PurePosixPath(raw_path)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise ReleaseWorkflowError("Git index contains an unsafe gitlink path")
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
        raise ReleaseWorkflowError(
            "cannot resolve repository for Git preflight"
        ) from error
    pending = [root]
    repositories: list[Path] = []
    seen: set[Path] = set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        try:
            top_level = _run_capture_process(
                _hardened_git_command(
                    executable, "rev-parse", "--show-toplevel"
                ),
                cwd=current,
                environment=environment,
                timeout=GIT_TIMEOUT_SECONDS,
            ).stdout.strip()
            authenticated_current = Path(top_level).resolve(strict=True)
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            raise ReleaseWorkflowError(
                "cannot authenticate repository worktree root"
            ) from error
        if authenticated_current != current:
            raise ReleaseWorkflowError(
                "repository resolves to a different worktree root"
            )
        _reject_executable_git_filters(
            executable,
            repository=current,
            environment=environment,
        )
        try:
            index = _run_capture_process(
                _hardened_git_command(executable, "ls-files", "--stage", "-z"),
                cwd=current,
                environment=environment,
                timeout=GIT_TIMEOUT_SECONDS,
            ).stdout
        except (OSError, subprocess.SubprocessError) as error:
            raise ReleaseWorkflowError(
                "cannot enumerate repository gitlinks"
            ) from error
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
                    raise ReleaseWorkflowError(
                        "checked-out gitlink has an unsafe worktree boundary"
                    )
                if not metadata.exists():
                    continue
                if metadata.is_symlink():
                    raise ReleaseWorkflowError(
                        "checked-out gitlink has unsafe Git metadata"
                    )
                resolved = candidate.resolve(strict=True)
                metadata_resolved = metadata.resolve(strict=True)
            except OSError as error:
                raise ReleaseWorkflowError(
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
                raise ReleaseWorkflowError(
                    "checked-out gitlink escapes the repository worktree"
                )
            try:
                top_level = _run_capture_process(
                    _hardened_git_command(
                        executable, "rev-parse", "--show-toplevel"
                    ),
                    cwd=resolved,
                    environment=environment,
                    timeout=GIT_TIMEOUT_SECONDS,
                ).stdout.strip()
                authenticated = Path(top_level).resolve(strict=True)
            except (OSError, subprocess.SubprocessError) as error:
                raise ReleaseWorkflowError(
                    "cannot authenticate checked-out gitlink repository"
                ) from error
            if authenticated != resolved:
                raise ReleaseWorkflowError(
                    "checked-out gitlink resolves to a different worktree"
                )
            pending.append(resolved)
    return tuple(repositories)


def run(
    workflow: Mapping[str, object],
    workflow_sha: str,
    root: Path,
    *,
    execute: bool,
) -> Mapping[str, object]:
    plan = build_plan(workflow, root)
    records: list[dict[str, object]] = []
    transcript_path = _confined(
        root,
        _relative(workflow["transcriptOutput"], "transcriptOutput"),
        must_exist=False,
    )
    all_outputs = [
        *(
            _confined(root, output, must_exist=False)
            for step in plan
            for output in step.outputs
        ),
        transcript_path,
    ]
    if execute:
        if any(path.exists() for path in all_outputs):
            raise ReleaseWorkflowError(
                "one or more declared release outputs already exist"
            )
        external_references = _external_tools(workflow)
        external = {
            name: _authenticate_external(reference, f"external {name}")
            for name, reference in external_references.items()
        }
        git_identity = _stable_file_identity(external["git"])
        environment = _closed_environment(root, external)
        for executable_name in ("git", "node", "pnpm"):
            resolved = shutil.which(
                executable_name,
                path=environment["PATH"],
            )
            if (
                resolved is None
                or Path(resolved).resolve() != external[executable_name]
            ):
                raise ReleaseWorkflowError(
                    f"closed PATH resolves the wrong {executable_name}"
                )
        revision = _run_capture_process(
            _hardened_git_command(
                external["git"], "rev-parse", "HEAD"
            ),
            cwd=root,
            environment=environment,
            timeout=GIT_TIMEOUT_SECONDS,
        ).stdout.strip()
        if revision != workflow["sourceRevision"]:
            raise ReleaseWorkflowError("source revision differs from workflow")
        _preflight_recursive_git_filters(
            external["git"],
            repository=root,
            environment=environment,
        )
        dirty = _run_capture_process(
            _hardened_git_command(
                external["git"],
                "status",
                "--porcelain",
                "--untracked-files=all",
                "--ignore-submodules=none",
            ),
            cwd=root,
            environment=environment,
            timeout=GIT_TIMEOUT_SECONDS,
        ).stdout
        if dirty:
            raise ReleaseWorkflowError(
                "tracked source changes are present during release execution"
            )
        if (
            _authenticate_external(
                external_references["git"], "external git"
            )
            != external["git"]
            or _stable_file_identity(external["git"]) != git_identity
        ):
            raise ReleaseWorkflowError(
                "external git identity changed during source authentication"
            )
        for reference in _input_references(workflow):
            _authenticate_input(root, reference)
    else:
        environment = {}
    parallel_selection: dict[
        tuple[int, int],
        tuple[tuple[str, ...], dict[Path, str], list[dict[str, str]]],
    ] = {}
    if execute:
        selection_steps = tuple(
            step
            for step in plan
            if step.stage == "selection-fit-evaluation"
        )
        selection_output_paths = tuple(
            _confined(root, output, must_exist=False)
            for step in selection_steps
            for output in step.outputs
        )
        futures: dict[
            Future[
                tuple[
                    tuple[str, ...],
                    dict[Path, str],
                    list[dict[str, str]],
                ]
            ],
            tuple[int, int],
        ] = {}
        wave = _SelectionWave()
        executor = ThreadPoolExecutor(
            max_workers=SELECTION_EVALUATION_WORKERS,
            thread_name_prefix="selection-evaluation",
        )
        failure_from_future = False
        try:
            for step in selection_steps:
                assert step.seed is not None
                assert step.epoch is not None
                future = executor.submit(
                    _execute_selection_step,
                    step,
                    workflow,
                    root,
                    environment,
                    wave,
                )
                futures[future] = (step.seed, step.epoch)
            for future in as_completed(futures):
                try:
                    result = future.result()
                except BaseException as error:
                    failure_from_future = (
                        not future.cancelled()
                        and future.exception() is error
                    )
                    raise
                parallel_selection[futures[future]] = result
        except BaseException as error:
            wave.stop.set()
            for future in futures:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            primary_failure = wave.primary_failure()
            reported_error = error
            if (
                failure_from_future
                and primary_failure is not None
                and primary_failure is not error
            ):
                reported_error = primary_failure
            elif primary_failure is not None and primary_failure is not error:
                reported_error.add_note(
                    "a selection worker also failed during cancellation: "
                    f"{primary_failure!r}"
                )
            cleanup_failures = _rollback_new_selection_outputs(
                selection_output_paths
            )
            if cleanup_failures:
                reported_error.add_note(
                    "selection output rollback was incomplete: "
                    + "; ".join(cleanup_failures)
                )
            if reported_error is not error:
                raise reported_error
            raise
        else:
            executor.shutdown(wait=True)
    for step in plan:
        output_paths = [_confined(root, item, must_exist=False) for item in step.outputs]
        if execute and any(path.exists() for path in output_paths):
            if step.stage != "selection-fit-evaluation":
                raise ReleaseWorkflowError(f"{step.stage} output already exists")
        recorded_argv = step.argv
        if execute:
            if step.stage == "selection-fit-evaluation":
                assert step.seed is not None
                assert step.epoch is not None
                recorded_argv, generated_before, outputs = (
                    parallel_selection[(step.seed, step.epoch)]
                )
            else:
                recorded_argv, generated_before, outputs = _execute_step(
                    step,
                    workflow,
                    root,
                    environment,
                )
        else:
            outputs = [{"path": rel.as_posix(), "sha256": None} for rel in step.outputs]
            generated_before = {
                relative: None for relative in step.generated_inputs
            }
        records.append({
            "stage": step.stage, "seed": step.seed, "epoch": step.epoch,
            "argv": list(recorded_argv),
            "inputs": [
                {
                    "path": reference.path.as_posix(),
                    "sha256": reference.sha256,
                }
                for reference in step.inputs
            ] + [
                {
                    "path": relative.as_posix(),
                    "sha256": generated_before[relative],
                }
                for relative in step.generated_inputs
            ],
            "outputs": outputs,
        })
    transcript = {
        "format": TRANSCRIPT_FORMAT, "version": VERSION, "evidence": False,
        "workflowSha256": workflow_sha,
        "sourceRevision": workflow["sourceRevision"],
        "mode": "execute" if execute else "dry-run",
        "realDomainExecution": "external-isolated-only",
        "externalTools": {
            name: {
                "path": str(reference.path),
                "sha256": reference.sha256,
            }
            for name, reference in _external_tools(workflow).items()
        },
        "steps": records,
    }
    if execute:
        output = transcript_path
        rendered = json.dumps(
            transcript, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode() + b"\n"
        output.parent.mkdir(parents=False, exist_ok=True)
        publish_bytes_durable(output, rendered)
    return transcript


def _execute_selection_step(
    step: Step,
    workflow: Mapping[str, object],
    root: Path,
    environment: Mapping[str, str],
    wave: _SelectionWave,
) -> tuple[tuple[str, ...], dict[Path, str], list[dict[str, str]]]:
    if wave.stop.is_set():
        raise _SelectionWaveCancelled(
            "selection evaluation wave was cancelled"
        )
    try:
        return _execute_step(
            step,
            workflow,
            root,
            environment,
            wave.stop,
        )
    except BaseException as error:
        wave.record_failure(error)
        raise


def _execute_step(
    step: Step,
    workflow: Mapping[str, object],
    root: Path,
    environment: Mapping[str, str],
    cancel: threading.Event | None = None,
) -> tuple[tuple[str, ...], dict[Path, str], list[dict[str, str]]]:
    output_paths = [
        _confined(root, item, must_exist=False)
        for item in step.outputs
    ]
    if any(path.exists() for path in output_paths):
        raise ReleaseWorkflowError(f"{step.stage} output already exists")
    for reference in step.inputs:
        _authenticate_input(root, reference)
    generated_before = {
        relative: _authenticate_generated(root, relative)[1]
        for relative in step.generated_inputs
    }
    # Rebuild immediately before execution so digests of prior outputs
    # replace dry-run placeholders.
    current = next(
        item
        for item in build_plan(workflow, root)
        if (
            item.stage == step.stage
            and item.seed == step.seed
            and item.epoch == step.epoch
        )
    )
    _run_step_process(
        current.argv,
        cwd=root,
        environment=environment,
        cancel=cancel,
    )
    for reference in step.inputs:
        _authenticate_input(root, reference)
    for relative, expected_sha256 in generated_before.items():
        if (
            _authenticate_generated(root, relative)[1]
            != expected_sha256
        ):
            raise ReleaseWorkflowError(
                f"generated input changed during {step.stage}: "
                f"{relative}"
            )
    if any(
        not path.is_file() or path.is_symlink()
        for path in output_paths
    ):
        raise ReleaseWorkflowError(
            f"{step.stage} did not publish declared outputs"
        )
    outputs = [
        {"path": rel.as_posix(), "sha256": _sha(path)}
        for rel, path in zip(step.outputs, output_paths, strict=True)
    ]
    return current.argv, generated_before, outputs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ml.evaluation.release_workflow")
    parser.add_argument("workflow", type=Path)
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    root = args.repository.resolve(strict=True)
    workflow, workflow_sha = load_workflow(args.workflow)
    print(json.dumps(run(workflow, workflow_sha, root, execute=args.execute),
                     sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
