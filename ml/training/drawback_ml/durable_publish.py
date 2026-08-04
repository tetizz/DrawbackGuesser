"""Crash-conscious, create-only publication for sealed research artifacts."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import hmac
import os
import secrets
import stat
from ctypes import wintypes
from pathlib import Path
from typing import Protocol


_PLATFORM = os.name

_MOVEFILE_WRITE_THROUGH = 0x00000008
_ERROR_FILE_EXISTS = 80
_ERROR_ALREADY_EXISTS = 183


class _OpenFile(Protocol):
    def fileno(self) -> int: ...

    def close(self) -> None: ...


def _stable_file_sha256(path: Path, label: str) -> tuple[str, int]:
    """Hash one regular file while authenticating the opened object.

    The pathname is checked again after the read. A symlink, replacement, or
    in-place mutation is rejected rather than being mistaken for an exact
    crash-recovery publication.
    """

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"existing {label} cannot be opened safely") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"existing {label} is not a regular file")
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(descriptor)
        try:
            current = path.stat(follow_symlinks=False)
        except OSError as error:
            raise ValueError(
                f"existing {label} changed while it was authenticated"
            ) from error
        final = os.fstat(descriptor)
        if (
            not stat.S_ISREG(current.st_mode)
            or not os.path.samestat(before, after)
            or not os.path.samestat(after, current)
            or not os.path.samestat(current, final)
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ctime_ns != after.st_ctime_ns
            or after.st_size != final.st_size
            or after.st_mtime_ns != final.st_mtime_ns
            or after.st_ctime_ns != final.st_ctime_ns
            or current.st_size != final.st_size
            or current.st_mtime_ns != final.st_mtime_ns
            # Windows can report the same file's creation/change timestamp at
            # slightly different precision through stat() and fstat(). The
            # two descriptor snapshots above still authenticate that field.
            or (
                _PLATFORM != "nt"
                and current.st_ctime_ns != final.st_ctime_ns
            )
            or size != final.st_size
        ):
            raise ValueError(
                f"existing {label} changed while it was authenticated"
            )
        return digest.hexdigest(), size
    finally:
        os.close(descriptor)


def authenticate_existing_sha256(
    path: Path,
    expected_sha256: str,
    label: str,
    *,
    expected_size: int | None = None,
) -> None:
    """Require an existing regular file to match one expected identity."""

    if (
        len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ValueError("expected SHA-256 must be a lowercase hexadecimal digest")
    actual_sha256, actual_size = _stable_file_sha256(path, label)
    if (
        not hmac.compare_digest(actual_sha256, expected_sha256)
        or (expected_size is not None and actual_size != expected_size)
    ):
        raise ValueError(f"existing {label} bytes do not match this run")


def publish_bytes_durable_exact(
    path: Path,
    payload: bytes,
    *,
    label: str,
) -> None:
    """Create one artifact or accept only an authenticated exact prior copy.

    This is the crash-recovery companion to :func:`publish_bytes_durable`.
    It never replaces or removes an existing pathname. A concurrent creator is
    accepted only when the stable bytes are exactly those requested here.
    """

    if not isinstance(payload, bytes):
        raise TypeError("durable publication payload must be bytes")
    expected_sha256 = hashlib.sha256(payload).hexdigest()
    try:
        path.lstat()
    except FileNotFoundError:
        pass
    except OSError as error:
        raise ValueError(f"existing {label} cannot be authenticated") from error
    else:
        authenticate_existing_sha256(
            path,
            expected_sha256,
            label,
            expected_size=len(payload),
        )
        return
    try:
        publish_bytes_durable(path, payload)
    except FileExistsError:
        authenticate_existing_sha256(
            path,
            expected_sha256,
            label,
            expected_size=len(payload),
        )


def publish_staged_file_durable(
    path: Path,
    staged: Path,
    expected_sha256: str,
    *,
    label: str,
    recover_exact: bool = False,
) -> None:
    """Publish a fully written same-directory scratch file create-only.

    This bounded-memory companion is intended for streaming artifacts. The
    caller must create ``staged`` with an unpredictable, exclusive name in the
    destination directory, flush it, fsync it, and close its writer first.
    Successful publication consumes the scratch path. If an exact destination
    is recovered after a competing or prior publication, the scratch path is
    retained because portable Windows Python cannot safely unlink a closed
    authenticated object without a pathname replacement race.
    """

    if staged.parent != path.parent or staged == path:
        raise ValueError(
            "durable staged publication requires distinct same-directory paths"
        )
    expected_sha256 = expected_sha256.casefold()
    actual_sha256, expected_size = _stable_file_sha256(staged, f"staged {label}")
    if not hmac.compare_digest(actual_sha256, expected_sha256):
        raise ValueError(f"staged {label} bytes do not match this run")

    if _PLATFORM == "nt":
        try:
            _move_windows_write_through(staged, path)
        except FileExistsError:
            if not recover_exact:
                raise
            authenticate_existing_sha256(
                path,
                expected_sha256,
                label,
                expected_size=expected_size,
            )
            return
        authenticate_existing_sha256(
            path,
            expected_sha256,
            label,
            expected_size=expected_size,
        )
        return

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(staged, flags)
    primary_error: BaseException | None = None
    opened: os.stat_result | None = None
    try:
        opened = os.fstat(descriptor)
        current = staged.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or not os.path.samestat(opened, current)
        ):
            raise ValueError(f"staged {label} changed before publication")
        try:
            os.link(staged, path)
        except FileExistsError:
            if not recover_exact:
                raise
            authenticate_existing_sha256(
                path,
                expected_sha256,
                label,
                expected_size=expected_size,
            )
        else:
            _fsync_parent_directory(path.parent)
            authenticate_existing_sha256(
                path,
                expected_sha256,
                label,
                expected_size=expected_size,
            )
    except BaseException as error:
        primary_error = error
        raise
    finally:
        cleanup_error: BaseException | None = None
        try:
            current = staged.stat(follow_symlinks=False)
        except FileNotFoundError:
            current = None
        except OSError as error:
            cleanup_error = error
            current = None
        if current is not None:
            if opened is not None and os.path.samestat(opened, current):
                try:
                    staged.unlink()
                except OSError as error:
                    cleanup_error = error
            else:
                cleanup_error = OSError(
                    "staged pathname changed; retained replacement"
                )
        try:
            os.close(descriptor)
        except OSError as error:
            if cleanup_error is None:
                cleanup_error = error
            else:
                cleanup_error.add_note(repr(error))
        if cleanup_error is not None:
            message = f"durable staged publication cleanup failed: {cleanup_error!r}"
            if primary_error is not None:
                primary_error.add_note(message)
            else:
                raise OSError(message) from cleanup_error


def abort_staged_file_safely(
    staged: Path,
    stream: _OpenFile,
    *,
    label: str,
) -> None:
    """Close a scratch writer without ever unlinking a replacement pathname.

    POSIX permits unlinking the authenticated object while its descriptor is
    still open. Windows CRT handles do not, so the unique scratch is retained
    after close and reported rather than risking a path-check/unlink race.
    """

    cleanup_errors: list[BaseException] = []
    opened: os.stat_result | None = None
    try:
        opened = os.fstat(stream.fileno())
    except BaseException as error:
        cleanup_errors.append(error)

    if _PLATFORM != "nt" and opened is not None:
        try:
            current = staged.stat(follow_symlinks=False)
        except FileNotFoundError:
            current = None
        except OSError as error:
            cleanup_errors.append(error)
            current = None
        if current is not None:
            if stat.S_ISREG(current.st_mode) and os.path.samestat(opened, current):
                try:
                    staged.unlink()
                except OSError as error:
                    cleanup_errors.append(error)
            else:
                cleanup_errors.append(
                    OSError(f"{label} pathname changed; retained replacement")
                )

    try:
        stream.close()
    except BaseException as error:
        cleanup_errors.append(error)

    if _PLATFORM == "nt":
        try:
            current = staged.stat(follow_symlinks=False)
        except FileNotFoundError:
            current = None
        except OSError as error:
            cleanup_errors.append(error)
            current = None
        if current is not None:
            if opened is not None and os.path.samestat(opened, current):
                cleanup_errors.append(
                    OSError(
                        f"retained {label} scratch because safe handle-bound "
                        "unlink is unavailable on Windows"
                    )
                )
            else:
                cleanup_errors.append(
                    OSError(f"{label} pathname changed; retained replacement")
                )

    if cleanup_errors:
        message = "; ".join(repr(error) for error in cleanup_errors)
        cleanup_error = OSError(f"{label} cleanup failed: {message}")
        for extra in cleanup_errors[1:]:
            cleanup_error.add_note(repr(extra))
        raise cleanup_error from cleanup_errors[0]


def _call_windows_move_file_ex(
    source: Path,
    destination: Path,
    flags: int,
) -> None:
    """Call ``MoveFileExW`` and preserve create-only error semantics."""

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    move_file_ex = kernel32.MoveFileExW
    move_file_ex.argtypes = (
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        wintypes.DWORD,
    )
    move_file_ex.restype = wintypes.BOOL

    if move_file_ex(str(source), str(destination), flags):
        return
    code = ctypes.get_last_error()
    message = (
        "cannot publish the durable artifact with MoveFileExW: "
        f"{ctypes.FormatError(code)}"
    )
    if code in {_ERROR_FILE_EXISTS, _ERROR_ALREADY_EXISTS}:
        raise FileExistsError(
            errno.EEXIST,
            message,
            str(destination),
        )
    raise OSError(code, message, str(destination))


def _move_windows_write_through(
    source: Path,
    destination: Path,
) -> None:
    """Move a same-directory temp file without permitting replacement.

    ``MOVEFILE_WRITE_THROUGH`` is the only flag. In particular,
    ``MOVEFILE_REPLACE_EXISTING`` is intentionally absent, so an existing
    sealed artifact makes publication fail rather than being overwritten.
    """

    _call_windows_move_file_ex(
        source,
        destination,
        _MOVEFILE_WRITE_THROUGH,
    )


def _verify_published_bytes(path: Path, expected: bytes) -> None:
    """Reopen a published artifact and compare its complete byte content."""

    with path.open("rb") as source:
        observed = source.read(len(expected) + 1)
    if observed != expected:
        raise OSError(
            f"durable publication verification failed for {path}"
        )


def _fsync_parent_directory(path: Path) -> None:
    """Persist the directory entry on platforms that expose directory fsync."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_bytes_durable(
    path: Path,
    payload: bytes,
) -> None:
    """Create ``path`` once and cross the platform durability barrier.

    On Windows, an already-fsynced same-directory temporary file is published
    with ``MoveFileExW(..., MOVEFILE_WRITE_THROUGH)``. The replacement flag is
    omitted. The destination is then reopened and byte-verified.
    On POSIX, publication uses a create-only hard link followed by an fsync of
    the parent directory.

    The destination is never replaced or removed. If a barrier or the mandatory
    post-publication verification fails after publication, the function raises
    but deliberately leaves the destination in place so a sealed input cannot
    be consumed again. Temporary-file close or unlink failures are also raised;
    if another error is already active, cleanup details are attached as notes.
    """

    if not isinstance(payload, bytes):
        raise TypeError("durable publication payload must be bytes")
    parent = path.parent
    if not parent.is_dir():
        raise NotADirectoryError(
            f"durable publication parent is not a directory: {parent}"
        )
    temporary = path.with_name(
        f"{path.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    )
    descriptor: int | None = None
    primary_error: BaseException | None = None
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_BINARY", 0)
        descriptor = os.open(temporary, flags, 0o600)
        with os.fdopen(descriptor, "wb", closefd=False) as destination:
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
        if _PLATFORM == "nt":
            os.close(descriptor)
            descriptor = None
            _move_windows_write_through(temporary, path)
            _verify_published_bytes(path, payload)
        else:
            os.link(temporary, path)
            _fsync_parent_directory(parent)
    except BaseException as error:
        primary_error = error
        raise
    finally:
        cleanup_errors: list[BaseException] = []
        if descriptor is not None:
            try:
                opened = os.fstat(descriptor)
            except OSError as error:
                cleanup_errors.append(error)
            else:
                try:
                    current = temporary.stat(follow_symlinks=False)
                except FileNotFoundError:
                    current = None
                except OSError as error:
                    cleanup_errors.append(error)
                    current = None
                if current is not None:
                    if os.path.samestat(opened, current):
                        if os.name == "nt" and _PLATFORM != "nt":
                            # Cross-platform tests exercise the POSIX branch on
                            # Windows, whose CRT handle denies unlink. Production
                            # POSIX keeps the authenticated handle open.
                            try:
                                os.close(descriptor)
                            except OSError as error:
                                cleanup_errors.append(error)
                            else:
                                descriptor = None
                        try:
                            # Keep the authenticated handle open through unlink;
                            # a mismatched pathname is retained, never deleted.
                            if descriptor is not None or os.name == "nt":
                                temporary.unlink()
                        except OSError as error:
                            cleanup_errors.append(error)
                    else:
                        cleanup_errors.append(
                            OSError(
                                "temporary pathname changed; retained replacement"
                            )
                        )
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError as error:
                    cleanup_errors.append(error)
        else:
            try:
                temporary.lstat()
            except FileNotFoundError:
                pass
            except OSError as error:
                cleanup_errors.append(error)
            else:
                # Windows publication closes the handle before MoveFileExW.
                # If the move failed, portable code cannot safely unlink the
                # remaining name because it may now identify a replacement.
                cleanup_errors.append(
                    OSError("retained temporary after closed-handle failure")
                )
        if cleanup_errors:
            detail = "; ".join(repr(error) for error in cleanup_errors)
            message = f"durable publication temporary cleanup failed: {detail}"
            if primary_error is not None:
                primary_error.add_note(message)
            else:
                cleanup_error = OSError(message)
                for extra in cleanup_errors[1:]:
                    cleanup_error.add_note(repr(extra))
                raise cleanup_error from cleanup_errors[0]
