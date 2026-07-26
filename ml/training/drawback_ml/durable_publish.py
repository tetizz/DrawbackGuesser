"""Crash-conscious, create-only publication for sealed research artifacts."""

from __future__ import annotations

import ctypes
import errno
import os
import secrets
from ctypes import wintypes
from pathlib import Path


_PLATFORM = os.name

_MOVEFILE_WRITE_THROUGH = 0x00000008
_ERROR_FILE_EXISTS = 80
_ERROR_ALREADY_EXISTS = 183


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
    be consumed again.
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
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_BINARY", 0)
        descriptor = os.open(temporary, flags, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as destination:
            descriptor = None
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
        if _PLATFORM == "nt":
            _move_windows_write_through(temporary, path)
            _verify_published_bytes(path, payload)
        else:
            os.link(temporary, path)
            _fsync_parent_directory(parent)
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
