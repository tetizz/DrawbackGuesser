"""Authenticated tool and process boundary for validation reproduction."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import sysconfig
import tempfile
from typing import Mapping, Sequence


RUNTIME_MANIFEST_FORMAT = "drawbacktrainer-python-runtime-manifest"
RUNTIME_MANIFEST_VERSION = 2
RUNTIME_CLOSURE_ALGORITHM = "sha256-canonical-loaded-python-modules-v1"
_REQUIRED_DISTRIBUTIONS = ("chess", "torch")
_GIT_TIMEOUT_SECONDS = 30
_REPRODUCTION_TIMEOUT_SECONDS = 6 * 60 * 60
_PROCESS_TERMINATION_GRACE_SECONDS = 5
_SHA256 = frozenset("0123456789abcdef")
_GIT_FILTER_CONFIG_ARGUMENTS = (
    "config",
    "--includes",
    "--show-scope",
    "--name-only",
    "--list",
)
_RUNTIME_BOOTSTRAP = r"""
import hashlib
import importlib
import json
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

repository = os.path.realpath(sys.argv.pop(1))
import_roots = json.loads(sys.argv.pop(1))
manifest_output = os.path.realpath(sys.argv.pop(1))
configured_sys_path = [repository, *[os.path.realpath(path) for path in import_roots]]
sys.path[:] = configured_sys_path
sys.argv[0] = "ml.evaluation.validation_gate"
sys.dont_write_bytecode = True

if os.name == "nt":
    import time
    gate = os.environ.get("DRAWBACK_REPRODUCTION_JOB_GATE")
    if not gate:
        raise SystemExit("missing authenticated Windows job gate")
    deadline = time.monotonic() + 30.0
    while not os.path.isfile(gate):
        if time.monotonic() >= deadline:
            raise SystemExit("authenticated Windows job gate timed out")
        time.sleep(0.005)

try:
    target = importlib.import_module("ml.evaluation.validation_gate")
    raise SystemExit(target.main())
finally:
    modules = []
    for name, module in sorted(sys.modules.items()):
        spec = getattr(module, "__spec__", None)
        origin = getattr(spec, "origin", None)
        if origin in {"built-in", "frozen"}:
            modules.append({"name": name, "kind": origin, "files": []})
            continue
        files = []
        candidates = (
            ("origin", origin),
            ("bytecode-cache", getattr(module, "__cached__", None)),
        )
        seen = set()
        for role, candidate in candidates:
            if not isinstance(candidate, str):
                continue
            resolved = os.path.realpath(candidate)
            if resolved in seen or not os.path.isfile(resolved):
                continue
            seen.add(resolved)
            digest = hashlib.sha256()
            with open(resolved, "rb") as source:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
            files.append({"role": role, "path": resolved, "sha256": digest.hexdigest()})
        modules.append({
            "name": name,
            "kind": "file" if files else "namespace",
            "files": files,
        })
    manifest = {
        "format": "drawbacktrainer-python-runtime-manifest",
        "version": 2,
        "python_executable": os.path.realpath(sys.executable),
        "isolated": sys.flags.isolated == 1,
        "ignore_environment": sys.flags.ignore_environment == 1,
        "no_site": sys.flags.no_site == 1,
        "no_user_site": sys.flags.no_user_site == 1,
        "safe_path": sys.flags.safe_path == 1,
        "dont_write_bytecode": sys.flags.dont_write_bytecode == 1,
        "hash_randomization": sys.flags.hash_randomization == 1,
        "python_controls": python_controls,
        "sitecustomize_loaded": "sitecustomize" in sys.modules,
        "usercustomize_loaded": "usercustomize" in sys.modules,
        "configured_sys_path": configured_sys_path,
        "final_sys_path": list(sys.path),
        "modules": modules,
    }
    payload = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    descriptor = os.open(
        manifest_output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    with os.fdopen(descriptor, "wb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
"""


@dataclass(frozen=True)
class _ExecutableIdentity:
    path: Path
    sha256: str
    size: int
    modified_ns: int
    changed_ns: int
    device: int
    inode: int


@dataclass(frozen=True)
class _PythonImportRoot:
    label: str
    path: Path


@dataclass(frozen=True)
class _PythonRuntimeConfiguration:
    executable: _ExecutableIdentity
    import_roots: tuple[_PythonImportRoot, ...]
    import_roots_sha256: str
    distributions_sha256: str


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _digest_file(path: Path, label: str) -> str:
    try:
        source = path.open("rb")
    except OSError as error:
        raise ValueError(f"cannot read {label}: {path}") from error
    digest = hashlib.sha256()
    with source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256 for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _strict_json(payload: bytes, label: str) -> Mapping[str, object]:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(token: str) -> None:
        raise ValueError(f"{label} contains non-finite constant {token}")

    try:
        value = json.loads(
            payload,
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} root must be an object")
    return value


def _is_link_or_reparse_point(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return path.is_symlink()
    return path.is_symlink() or (
        os.name == "nt"
        and bool(
            getattr(metadata, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x0400)
        )
    )


def _read_executable_identity(path: Path, label: str) -> _ExecutableIdentity:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{label} is unavailable") from error
    if not resolved.is_absolute() or not resolved.is_file():
        raise ValueError(f"{label} is not an absolute regular file")
    if _is_link_or_reparse_point(resolved):
        raise ValueError(f"{label} must not be a link or reparse point")
    digest = hashlib.sha256()
    try:
        with resolved.open("rb") as source:
            before = os.fstat(source.fileno())
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
            after = os.fstat(source.fileno())
        current = resolved.stat()
    except OSError as error:
        raise ValueError(f"cannot authenticate {label}") from error
    handle_attributes = (
        "st_dev",
        "st_ino",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    # Windows reports different creation/change-time semantics for fstat() and
    # path.stat() on the same file. Keep ctime in the same-handle stability
    # check, but compare the pathname using fields that are stable across both
    # APIs. The content hash remains the executable's primary identity.
    path_attributes = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(
        getattr(before, key) != getattr(after, key)
        for key in handle_attributes
    ) or any(
        getattr(after, key) != getattr(current, key)
        for key in path_attributes
    ):
        raise ValueError(f"{label} changed while it was authenticated")
    return _ExecutableIdentity(
        path=resolved,
        sha256=digest.hexdigest(),
        size=after.st_size,
        modified_ns=after.st_mtime_ns,
        changed_ns=after.st_ctime_ns,
        device=after.st_dev,
        inode=after.st_ino,
    )


def _path_identity_sha256(path: Path) -> str:
    canonical = os.path.normcase(str(path.resolve())).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _windows_known_directory(kind: str) -> Path:
    if os.name != "nt":
        raise ValueError("Windows known-directory lookup is unavailable")
    buffer = ctypes.create_unicode_buffer(32_768)
    if kind == "program-files":
        result = ctypes.windll.shell32.SHGetFolderPathW(
            None, 0x0026, None, 0, buffer
        )
        if result != 0:
            raise ValueError("trusted Windows Program Files is unavailable")
    else:
        function_name = {
            "system": "GetSystemDirectoryW",
            "windows": "GetSystemWindowsDirectoryW",
        }.get(kind)
        if function_name is None:
            raise ValueError("Windows known-directory role is invalid")
        length = getattr(ctypes.windll.kernel32, function_name)(buffer, len(buffer))
        if length <= 0 or length >= len(buffer):
            raise ValueError(f"trusted Windows {kind} directory is unavailable")
    try:
        directory = Path(buffer.value).resolve(strict=True)
    except OSError as error:
        raise ValueError(f"trusted Windows {kind} directory is unavailable") from error
    if not directory.is_dir() or _is_link_or_reparse_point(directory):
        raise ValueError(f"trusted Windows {kind} directory is invalid")
    return directory


def _trusted_git_executable() -> Path:
    if os.name == "nt":
        candidate = _windows_known_directory("program-files") / "Git/cmd/git.exe"
    else:
        candidates = {
            path.resolve(strict=True)
            for path in (Path("/usr/bin/git"), Path("/bin/git"))
            if path.is_file()
        }
        if len(candidates) != 1:
            raise ValueError("trusted system Git executable is unavailable")
        candidate = candidates.pop()
    return _read_executable_identity(candidate, "trusted Git executable").path


def _trusted_python_executable() -> Path:
    if not sys.executable:
        raise ValueError("current Python executable identity is unavailable")
    return _read_executable_identity(
        Path(sys.executable), "current Python executable"
    ).path


def _temporary_environment() -> dict[str, str]:
    environment: dict[str, str] = {}
    for key in ("TEMP", "TMP", "TMPDIR"):
        value = os.environ.get(key)
        if value:
            environment[key] = value
    return environment


def _authenticated_git_environment(git_executable: Path) -> dict[str, str]:
    environment = _temporary_environment()
    if os.name == "nt":
        windows_directory = _windows_known_directory("windows")
        system_directory = _windows_known_directory("system")
        shell = _read_executable_identity(
            system_directory / "cmd.exe", "trusted Windows command shell"
        ).path
        environment.update(
            {
                "ComSpec": str(shell),
                "PATHEXT": ".COM;.EXE;.BAT;.CMD",
                "PATH": os.pathsep.join(
                    (
                        str(git_executable.parent),
                        str(system_directory),
                        str(windows_directory),
                    )
                ),
                "SystemRoot": str(windows_directory),
                "WINDIR": str(windows_directory),
            }
        )
    else:
        environment["PATH"] = os.pathsep.join(("/usr/bin", "/bin"))
    environment.update(
        {
            "GCM_INTERACTIVE": "Never",
            "GIT_ASKPASS": os.devnull,
            "GIT_CONFIG_COUNT": "0",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_PAGER": "",
            "GIT_TERMINAL_PROMPT": "0",
            "LC_ALL": "C",
            "SSH_ASKPASS": os.devnull,
            "SSH_ASKPASS_REQUIRE": "never",
        }
    )
    return environment


def _isolated_python_environment() -> dict[str, str]:
    environment = _temporary_environment()
    if os.name == "nt":
        windows_directory = _windows_known_directory("windows")
        system_directory = _windows_known_directory("system")
        environment.update(
            {
                "ComSpec": str(
                    _read_executable_identity(
                        system_directory / "cmd.exe",
                        "trusted Windows command shell",
                    ).path
                ),
                "PATH": str(system_directory),
                "SystemRoot": str(windows_directory),
                "WINDIR": str(windows_directory),
            }
        )
    else:
        environment["PATH"] = os.pathsep.join(("/usr/bin", "/bin"))
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "GIT_CONFIG_COUNT": "0",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "LC_ALL": "C",
            "PYTHONHASHSEED": "0",
        }
    )
    return environment


def _close_process_streams(process: subprocess.Popen[str]) -> None:
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass


def _windows_taskkill(process_id: int) -> None:
    system_directory = _windows_known_directory("system")
    taskkill = _read_executable_identity(
        system_directory / "taskkill.exe", "trusted Windows taskkill executable"
    )
    completed: subprocess.CompletedProcess[str] | None = None
    invocation_error: BaseException | None = None
    try:
        completed = subprocess.run(
            [str(taskkill.path), "/PID", str(process_id), "/T", "/F"],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=_PROCESS_TERMINATION_GRACE_SECONDS,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            env=_isolated_python_environment(),
        )
    except BaseException as error:
        invocation_error = error
    identity_error: BaseException | None = None
    try:
        if (
            _read_executable_identity(
                taskkill.path, "trusted Windows taskkill executable"
            )
            != taskkill
        ):
            raise OSError("trusted Windows taskkill identity changed during cleanup")
    except BaseException as error:
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
        raise OSError(f"taskkill failed with exit code {completed.returncode}")


class _WindowsKillJob:
    """Kill-on-close containment for a Windows reproduction process tree."""

    def __init__(self) -> None:
        self._handle: int | None = None
        self._assigned = False
        if os.name != "nt":
            return
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
            _fields_ = [
                (name, ctypes.c_ulonglong)
                for name in (
                    "ReadOperationCount",
                    "WriteOperationCount",
                    "OtherOperationCount",
                    "ReadTransferCount",
                    "WriteTransferCount",
                    "OtherTransferCount",
                )
            ]

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
        kernel32.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = (
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        )
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise OSError(
                ctypes.get_last_error(), "cannot create Windows reproduction job"
            )
        limits = ExtendedLimit()
        limits.BasicLimitInformation.LimitFlags = 0x00002000
        if not kernel32.SetInformationJobObject(
            handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)
        ):
            error_code = ctypes.get_last_error()
            kernel32.CloseHandle(handle)
            raise OSError(error_code, "cannot configure Windows reproduction job")
        self._handle = int(handle)

    def assign(self, process: subprocess.Popen[str]) -> None:
        if self._handle is None:
            return
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.AssignProcessToJobObject.argtypes = (
            wintypes.HANDLE,
            wintypes.HANDLE,
        )
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        # subprocess exposes the owned OS handle on Windows. Keeping this cast
        # local avoids an unauthenticated PID-to-handle lookup.
        process_handle = wintypes.HANDLE(process._handle)  # type: ignore[attr-defined]
        if not kernel32.AssignProcessToJobObject(
            wintypes.HANDLE(self._handle), process_handle
        ):
            raise OSError(
                ctypes.get_last_error(), "cannot assign Windows reproduction job"
            )
        self._assigned = True

    def terminate(self) -> None:
        if self._handle is None or not self._assigned:
            raise OSError("Windows reproduction job does not contain the process")
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.TerminateJobObject.argtypes = (wintypes.HANDLE, wintypes.UINT)
        kernel32.TerminateJobObject.restype = wintypes.BOOL
        if not kernel32.TerminateJobObject(wintypes.HANDLE(self._handle), 1):
            raise OSError(
                ctypes.get_last_error(), "cannot terminate Windows reproduction job"
            )

    def close(self) -> None:
        if self._handle is None:
            return
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = self._handle
        self._handle = None
        if not kernel32.CloseHandle(wintypes.HANDLE(handle)):
            raise OSError(
                ctypes.get_last_error(), "cannot close Windows reproduction job"
            )


def _terminate_process_tree(
    process: subprocess.Popen[str], windows_job: _WindowsKillJob
) -> None:
    cleanup_error: BaseException | None = None
    if os.name == "nt":
        try:
            windows_job.terminate()
        except OSError as job_error:
            try:
                _windows_taskkill(process.pid)
            except (OSError, subprocess.SubprocessError) as tree_error:
                tree_error.add_note(
                    "Windows job termination also failed: "
                    f"{type(job_error).__name__}: {job_error}"
                )
                cleanup_error = tree_error
                try:
                    process.kill()
                except OSError:
                    pass
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        if process.poll() is None:
            try:
                process.wait(timeout=_PROCESS_TERMINATION_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError as error:
            cleanup_error = error
    if process.poll() is None:
        try:
            process.wait(timeout=_PROCESS_TERMINATION_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except OSError:
                pass
            try:
                process.wait(timeout=_PROCESS_TERMINATION_GRACE_SECONDS)
            except subprocess.TimeoutExpired as error:
                cleanup_error = OSError(
                    "reproduction process tree did not settle"
                )
                cleanup_error.__cause__ = error
    _close_process_streams(process)
    if cleanup_error is not None:
        raise OSError("reproduction process-tree cleanup failed") from cleanup_error


def _raise_primary_with_cleanup(
    primary_error: BaseException,
    cleanup_error: BaseException,
) -> None:
    primary_error.add_note(
        "reproduction process cleanup also failed: "
        f"{type(cleanup_error).__name__}: {cleanup_error}"
    )
    raise primary_error from cleanup_error


def _run_bounded_process(
    arguments: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    check: bool,
    capture_output: bool,
    text: bool,
    encoding: str,
    timeout_seconds: float = _REPRODUCTION_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    if timeout_seconds <= 0:
        raise ValueError("reproduction process timeout must be positive")
    environment = dict(env)
    gate_path: Path | None = None
    windows_job = _WindowsKillJob()
    popen_options: dict[str, object] = {}
    if os.name == "nt":
        descriptor, gate_name = tempfile.mkstemp(
            prefix="drawback-reproduction-job-", suffix=".gate"
        )
        os.close(descriptor)
        gate_path = Path(gate_name)
        gate_path.unlink(missing_ok=True)
        environment["DRAWBACK_REPRODUCTION_JOB_GATE"] = str(gate_path)
        popen_options["creationflags"] = getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0
        )
    else:
        popen_options["start_new_session"] = True
    process: subprocess.Popen[str] | None = None
    cleanup_attempted = False
    try:
        process = subprocess.Popen(
            list(arguments),
            cwd=cwd,
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE if capture_output else None,
            stderr=subprocess.PIPE if capture_output else None,
            text=text,
            encoding=encoding,
            errors="strict",
            env=environment,
            **popen_options,
        )
        try:
            windows_job.assign(process)
        except OSError as primary_error:
            cleanup_attempted = True
            try:
                _terminate_process_tree(process, windows_job)
            except BaseException as cleanup_error:
                _raise_primary_with_cleanup(primary_error, cleanup_error)
            raise
        if gate_path is not None:
            gate_path.write_bytes(b"assigned\n")
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
        except BaseException as primary_error:
            cleanup_attempted = True
            try:
                _terminate_process_tree(process, windows_job)
            except BaseException as cleanup_error:
                _raise_primary_with_cleanup(primary_error, cleanup_error)
            raise
        # Remove descendants even when the direct parent exits successfully.
        return_code = process.returncode
        cleanup_attempted = True
        _terminate_process_tree(process, windows_job)
        completed = subprocess.CompletedProcess(
            list(arguments), return_code, stdout, stderr
        )
        if check and completed.returncode != 0:
            raise subprocess.CalledProcessError(
                completed.returncode,
                completed.args,
                output=completed.stdout,
                stderr=completed.stderr,
            )
        return completed
    except BaseException as primary_error:
        if process is not None and not cleanup_attempted:
            try:
                _terminate_process_tree(process, windows_job)
            except BaseException as cleanup_error:
                _raise_primary_with_cleanup(primary_error, cleanup_error)
        raise
    finally:
        if gate_path is not None:
            gate_path.unlink(missing_ok=True)
        try:
            windows_job.close()
        finally:
            if process is not None and process.poll() is not None:
                _close_process_streams(process)


def _distribution_identity(name: str) -> tuple[Path, Mapping[str, object]]:
    try:
        distribution = importlib.metadata.distribution(name)
        root = Path(distribution.locate_file("")).resolve(strict=True)
    except (importlib.metadata.PackageNotFoundError, OSError) as error:
        raise ValueError(
            f"required Python distribution {name!r} is unavailable"
        ) from error
    if not root.is_dir():
        raise ValueError(f"required Python distribution {name!r} has no import root")
    metadata_files: list[dict[str, str]] = []
    for relative in distribution.files or ():
        if relative.name not in {"METADATA", "RECORD"}:
            continue
        path = Path(distribution.locate_file(relative))
        if ".dist-info" not in path.parent.name:
            continue
        metadata_files.append(
            {
                "file": relative.as_posix(),
                "sha256": _digest_file(path, f"{name} distribution metadata"),
            }
        )
    if not metadata_files:
        raise ValueError(f"required Python distribution {name!r} lacks metadata")
    return root, {
        "name": str(distribution.metadata["Name"] or name).lower(),
        "version": distribution.version,
        "metadata": sorted(metadata_files, key=lambda item: item["file"]),
    }


def _python_runtime_configuration(repository: Path) -> _PythonRuntimeConfiguration:
    try:
        root = repository.resolve(strict=True)
    except OSError as error:
        raise ValueError("repository root is unavailable") from error
    if not root.is_dir():
        raise ValueError("repository root must be a directory")
    executable = _read_executable_identity(
        _trusted_python_executable(), "current Python executable"
    )
    roots: list[_PythonImportRoot] = []
    seen: set[Path] = set()

    def add_root(label: str, candidate: Path) -> None:
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            return
        if resolved in seen or not (resolved.is_dir() or resolved.is_file()):
            return
        seen.add(resolved)
        roots.append(_PythonImportRoot(label, resolved))

    for label, key in (("stdlib", "stdlib"), ("platform-stdlib", "platstdlib")):
        configured = sysconfig.get_path(key)
        if configured:
            add_root(label, Path(configured))
    add_root("extension-modules", Path(sys.base_prefix) / "DLLs")
    add_root(
        "standard-library-zip",
        Path(sys.base_prefix)
        / f"python{sys.version_info.major}{sys.version_info.minor}.zip",
    )
    distributions: list[Mapping[str, object]] = []
    for name in _REQUIRED_DISTRIBUTIONS:
        distribution_root, identity = _distribution_identity(name)
        add_root(f"distribution-{name}", distribution_root)
        distributions.append(identity)
    if not roots:
        raise ValueError("Python runtime has no authenticated import roots")
    root_value = [
        {"label": item.label, "path_sha256": _path_identity_sha256(item.path)}
        for item in roots
    ]
    return _PythonRuntimeConfiguration(
        executable=executable,
        import_roots=tuple(roots),
        import_roots_sha256=hashlib.sha256(_canonical(root_value)).hexdigest(),
        distributions_sha256=hashlib.sha256(
            _canonical(sorted(distributions, key=lambda item: str(item["name"])))
        ).hexdigest(),
    )


def _run_git(
    repository: Path,
    arguments: Sequence[str],
    *,
    git_executable: Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    executable = git_executable or _trusted_git_executable()
    authenticated = _read_executable_identity(executable, "trusted Git executable")
    child_environment = (
        dict(environment)
        if environment is not None
        else _authenticated_git_environment(authenticated.path)
    )
    try:
        return subprocess.run(
            [
                str(authenticated.path),
                "--no-replace-objects",
                "--no-pager",
                "-c",
                "core.attributesFile=",
                "-c",
                "core.excludesFile=",
                "-c",
                "core.fsmonitor=false",
                "-c",
                f"core.hooksPath={os.devnull}",
                "-c",
                f"core.askPass={os.devnull}",
                "-c",
                "credential.helper=",
                "-c",
                "credential.interactive=false",
                "-c",
                "protocol.allow=never",
                "-C",
                str(repository),
                *arguments,
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            env=child_environment,
            stdin=subprocess.DEVNULL,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except (OSError, UnicodeError, subprocess.TimeoutExpired) as error:
        raise ValueError("cannot execute Git for source attestation") from error


def _assert_no_executable_git_filters(configured: str) -> None:
    """Reject command-bearing filter drivers in effective Git configuration."""

    for line in configured.splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise ValueError("Git filter configuration probe returned malformed output")
        key = fields[1].casefold()
        if (
            key.startswith("filter.")
            and key.rsplit(".", maxsplit=1)[-1]
            in {"clean", "smudge", "process"}
        ):
            raise ValueError("repository config contains an executable Git filter")


def _runtime_location(
    path: Path,
    *,
    repository: Path,
    import_roots: Sequence[_PythonImportRoot],
) -> str:
    roots = (_PythonImportRoot("repository", repository.resolve()), *import_roots)
    for root in roots:
        if not root.path.is_dir():
            continue
        try:
            relative = path.relative_to(root.path)
        except ValueError:
            continue
        if not relative.parts:
            raise ValueError("loaded Python runtime file has no relative identity")
        return f"{root.label}/{relative.as_posix()}"
    raise ValueError(
        f"loaded Python runtime file is outside authenticated roots: {path}"
    )


def _validate_runtime_closure(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "algorithm",
        "module_count",
        "file_count",
        "sha256",
        "modules",
    }:
        raise ValueError("validation reproduction runtime closure is invalid")
    modules = value.get("modules")
    if (
        value.get("algorithm") != RUNTIME_CLOSURE_ALGORITHM
        or not isinstance(modules, list)
        or not modules
    ):
        raise ValueError("validation reproduction runtime closure is unsupported")
    names: set[str] = set()
    observed_file_count = 0
    for module in modules:
        if not isinstance(module, Mapping) or set(module) != {
            "name",
            "kind",
            "files",
        }:
            raise ValueError("validation reproduction runtime module is invalid")
        name = module.get("name")
        kind = module.get("kind")
        files = module.get("files")
        if (
            not isinstance(name, str)
            or not name
            or name in names
            or kind not in {"built-in", "file", "frozen", "namespace"}
            or not isinstance(files, list)
            or (kind == "file") != bool(files)
        ):
            raise ValueError("validation reproduction runtime module is malformed")
        names.add(name)
        roles: set[tuple[str, str]] = set()
        for binding in files:
            if not isinstance(binding, Mapping) or set(binding) != {
                "role",
                "location",
                "sha256",
            }:
                raise ValueError("validation reproduction runtime file is invalid")
            role = binding.get("role")
            location = binding.get("location")
            if (
                role not in {"bytecode-cache", "origin"}
                or not isinstance(location, str)
                or not location
                or "\\" in location
                or Path(location).is_absolute()
                or any(part in {"", ".", ".."} for part in location.split("/"))
                or (str(role), location) in roles
            ):
                raise ValueError("validation reproduction runtime file is malformed")
            _digest(binding.get("sha256"), "runtime file sha256")
            roles.add((str(role), location))
            observed_file_count += 1
    if (
        value.get("module_count") != len(modules)
        or value.get("file_count") != observed_file_count
        or value.get("sha256") != hashlib.sha256(_canonical(modules)).hexdigest()
    ):
        raise ValueError("validation reproduction runtime closure digest differs")


def _load_runtime_manifest(
    path: Path,
    *,
    repository: Path,
    runtime: _PythonRuntimeConfiguration,
) -> Mapping[str, object]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValueError("cannot read fresh Python runtime manifest") from error
    manifest = _strict_json(payload, "fresh Python runtime manifest")
    expected_keys = {
        "format",
        "version",
        "python_executable",
        "isolated",
        "ignore_environment",
        "no_site",
        "no_user_site",
        "safe_path",
        "dont_write_bytecode",
        "hash_randomization",
        "python_controls",
        "sitecustomize_loaded",
        "usercustomize_loaded",
        "configured_sys_path",
        "final_sys_path",
        "modules",
    }
    if set(manifest) != expected_keys:
        raise ValueError("fresh Python runtime manifest fields are invalid")
    expected_sys_path = [
        str(repository.resolve()),
        *(str(root.path) for root in runtime.import_roots),
    ]
    if (
        manifest.get("format") != RUNTIME_MANIFEST_FORMAT
        or manifest.get("version") != RUNTIME_MANIFEST_VERSION
        or manifest.get("python_executable") != str(runtime.executable.path)
        or manifest.get("isolated") is not False
        or manifest.get("ignore_environment") is not False
        or manifest.get("no_site") is not True
        or manifest.get("no_user_site") is not True
        or manifest.get("safe_path") is not True
        or manifest.get("dont_write_bytecode") is not True
        or manifest.get("hash_randomization") is not False
        or manifest.get("python_controls") != {"PYTHONHASHSEED": "0"}
        or manifest.get("sitecustomize_loaded") is not False
        or manifest.get("usercustomize_loaded") is not False
        or manifest.get("configured_sys_path") != expected_sys_path
        or manifest.get("final_sys_path") != expected_sys_path
    ):
        raise ValueError("fresh Python process was not fully isolated")
    modules = manifest.get("modules")
    if not isinstance(modules, list) or not modules:
        raise ValueError("fresh Python runtime manifest has no loaded modules")
    normalized: list[dict[str, object]] = []
    names: set[str] = set()
    target_loaded = False
    file_count = 0
    for module in modules:
        if not isinstance(module, Mapping) or set(module) != {
            "name",
            "kind",
            "files",
        }:
            raise ValueError("fresh Python runtime module is invalid")
        name = module.get("name")
        kind = module.get("kind")
        files = module.get("files")
        if (
            not isinstance(name, str)
            or not name
            or name in names
            or kind not in {"built-in", "file", "frozen", "namespace"}
            or not isinstance(files, list)
            or (kind == "file") != bool(files)
        ):
            raise ValueError("fresh Python runtime module is malformed")
        names.add(name)
        normalized_files: list[dict[str, str]] = []
        observed: set[tuple[str, Path]] = set()
        for binding in files:
            if not isinstance(binding, Mapping) or set(binding) != {
                "role",
                "path",
                "sha256",
            }:
                raise ValueError("fresh Python runtime file is invalid")
            role = binding.get("role")
            if role not in {"origin", "bytecode-cache"}:
                raise ValueError("fresh Python runtime file role is invalid")
            raw_path = binding.get("path")
            if not isinstance(raw_path, str) or not raw_path:
                raise ValueError("fresh Python runtime file path is invalid")
            try:
                resolved = Path(raw_path).resolve(strict=True)
            except OSError as error:
                raise ValueError("fresh Python runtime file disappeared") from error
            if not resolved.is_file() or (str(role), resolved) in observed:
                raise ValueError("fresh Python runtime file identity is invalid")
            observed.add((str(role), resolved))
            expected = _digest(
                binding.get("sha256"), "fresh Python runtime file sha256"
            )
            if _digest_file(resolved, "fresh Python runtime file") != expected:
                raise ValueError("fresh Python runtime file changed after execution")
            location = _runtime_location(
                resolved,
                repository=repository,
                import_roots=runtime.import_roots,
            )
            if location == "repository/ml/evaluation/validation_gate.py":
                target_loaded = True
            normalized_files.append(
                {"role": str(role), "location": location, "sha256": expected}
            )
            file_count += 1
        normalized.append(
            {
                "name": name,
                "kind": kind,
                "files": sorted(
                    normalized_files,
                    key=lambda item: (item["role"], item["location"]),
                ),
            }
        )
    if [str(module["name"]) for module in normalized] != sorted(names):
        raise ValueError("fresh Python runtime modules are not canonical")
    if not target_loaded or "ml.evaluation.validation_gate" not in names:
        raise ValueError("fresh Python runtime did not load the approved evaluator")
    closure: Mapping[str, object] = {
        "algorithm": RUNTIME_CLOSURE_ALGORITHM,
        "module_count": len(normalized),
        "file_count": file_count,
        "sha256": hashlib.sha256(_canonical(normalized)).hexdigest(),
        "modules": normalized,
    }
    _validate_runtime_closure(closure)
    return closure
