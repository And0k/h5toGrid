"""Restricted filesystem-path detection for Markdown auto-linking.

A text span is linkable only when it is an **absolute** path — a file with a
**2–4 alnum extension**, or, under a known allowed directory, a directory on
disk.  The allowed directory (last value of the GUI path field, CLI default
included) is matched literally via ``re.escape``, so *its* spaces are fine;
nodes after it may not contain whitespace — a space terminates a candidate.
The allowed directory **itself** also matches and links (the device dir must
be clickable, not only the paths beneath it).  With no directory configured
any absolute path (drive / UNC / POSIX root) matches, files only —
unrestricted directory matching would over-link prose.
"""

from __future__ import annotations

import os
import re
from functools import cache
from pathlib import Path
from typing import Final

_SEP: Final = r"[\\/]+"
_EXT: Final = r"[A-Za-z0-9]{2,4}"  # linkable files only (.h5/.nc/.txt/.yaml …, not .c/.astro)

# After allowed_dir: no whitespace inside path nodes; Windows-invalid chars excluded.
_VALID: Final = r'[^<>:"/\\|?*\x00-\x1F\s]'
_SAFE_END: Final = r'[^<>:"/\\|?*\x00-\x1F\s.,;:!?)\]]'  # …nor trailing prose punctuation

_NODE: Final = rf"(?:{_VALID}*{_SAFE_END})"
# Mandatory file node: the 2–4 letter extension may not be a prefix of a longer
# word (``f.ncx`` does not match via ``nc``), prose punctuation after it may.
_FILE: Final = rf"{_NODE}\.{_EXT}(?![A-Za-z0-9_])"
_GUARD: Final = r"(?<![a-zA-Z0-9_\\/:@-])"  # no re-trigger mid-path / inside URLs


def _escaped_root(root: str) -> str:
    """Regex for the allowed-dir prefix: UNC ``\\\\srv\\…`` / rooted ``/…`` / plain ``C:\\…``."""
    if not root:
        return ""

    unc = root.startswith((r"\\", "//"))
    parts = [part for part in re.split(_SEP, root) if part]

    if not parts:
        return ""

    escaped = _SEP.join(re.escape(part) for part in parts)

    if unc:
        return rf"(?:\\\\|//){escaped}"

    if root.startswith(("/", "\\")):
        return rf"{_SEP}{escaped}"

    return escaped


@cache
def _compile(root: str) -> re.Pattern[str]:
    """Pattern for *root*; ``''`` → unrestricted branch (any absolute path).

    Restricted matches are node chains — files *and* directories, classified
    by :func:`path_kind`; unrestricted stays files-only (directories anywhere
    in prose would over-link).
    """
    if prefix := _escaped_root(root):
        return re.compile(rf"{_GUARD}{prefix}(?:{_SEP}(?:{_NODE}{_SEP})*{_NODE})?", re.IGNORECASE)

    # Optional drive letter (UNC/POSIX roots ride the bare-separator branch).
    return re.compile(rf"{_GUARD}(?:[A-Za-z]:)?{_SEP}(?:{_NODE}{_SEP})*{_FILE}", re.IGNORECASE)


def allowed_path_re(allowed_dir: str | os.PathLike[str]) -> re.Pattern[str]:
    """Pattern matching linkable absolute paths beneath *allowed_dir* (``''`` → any)."""
    return _compile(os.fspath(allowed_dir))


@cache
def _is_dir(path: str) -> bool:
    return Path(path).is_dir()


def path_kind(path: str) -> str:
    """``'file'`` | ``'dir'`` | ``''`` (not linkable), decided by the final node.

    A 2–4 alnum extension makes a file; a node without one is a directory —
    but only a real one on disk (guards the spaced-dir false positive
    ``…\\My`` out of ``…\\My Dir\\f.nc``); an odd extension stays plain.
    ``.`` / ``..`` are not linkable — they are relative path components, not
    directories to open (fixes the status-bar auto-linking bare "." to the
    project root).
    """
    if suffix := Path(path).suffix[1:]:
        return "file" if suffix.isalnum() and 2 <= len(suffix) <= 4 else ""
    return "dir" if _is_dir(path) and path not in (".", "..") else ""


def to_uri(path: str) -> str:
    """``file:`` URI (drive-less Windows roots fall back to the cwd anchor)."""
    try:
        return Path(path).as_uri()
    except ValueError:
        return Path(os.path.abspath(path)).as_uri()


def extract_allowed_paths(
    text: str,
    allowed_dir: str | os.PathLike[str],
) -> tuple[str, ...]:
    """All linkable paths occurring in *text* (test / extraction convenience)."""
    return tuple(m[0] for m in allowed_path_re(allowed_dir).finditer(text) if path_kind(m[0]))
