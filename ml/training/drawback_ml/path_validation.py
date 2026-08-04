"""Portable path-component validation for authenticated artifacts.

Release manifests are consumed on both POSIX and Windows.  A name that looks
distinct on POSIX can refer to the same Windows object through case folding,
trailing-dot/space normalization, a device alias, or an alternate data stream.
Keep this structural policy independent from the host running validation.
"""

from __future__ import annotations

from pathlib import PurePosixPath, PureWindowsPath
import re
from typing import TypeGuard


_SAFE_BASENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}\Z")
_WINDOWS_RESERVED_BASENAME = re.compile(
    r"(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?\Z",
    re.IGNORECASE,
)


def is_portable_safe_basename(value: object) -> TypeGuard[str]:
    """Return whether *value* is one unambiguous portable file name.

    The deliberately narrow ASCII alphabet also excludes Windows 8.3 aliases,
    control characters, whitespace normalization, and Unicode-equivalence
    ambiguity.  Callers remain responsible for domain-specific privacy rules.
    """

    return (
        isinstance(value, str)
        and bool(value)
        and _SAFE_BASENAME.fullmatch(value) is not None
        and PureWindowsPath(value).name == value
        and PurePosixPath(value).name == value
        and not PureWindowsPath(value).drive
        and ":" not in value
        and "/" not in value
        and "\\" not in value
        and value[-1] not in {" ", "."}
        and value not in {".", ".."}
        and _WINDOWS_RESERVED_BASENAME.fullmatch(value) is None
    )


def portable_basename_key(value: str) -> str:
    """Return the portable collision key for an already validated basename."""

    if not is_portable_safe_basename(value):
        raise ValueError("portable basename key requires a safe basename")
    return value.casefold()
