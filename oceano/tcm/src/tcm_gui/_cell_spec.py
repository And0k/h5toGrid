"""Classify Hydra structured-config field types for tksheet widget rendering.

Maps dataclass field types to ``CellSpec`` describing how each cell should
be rendered: ``bool`` → checkbox, ``Enum`` (including ``StrEnum``) →
dropdown, ``str``/``Path`` → left-aligned text, ``int``/``float`` →
right-aligned number, ``date``/``datetime`` → right-aligned date.

Resolution walks the dataclass tree along a dotted path (e.g.
``"program.return_"``) via :func:`resolve_dataclass_field`.
"""

from __future__ import annotations

import functools
import os
import types
from contextlib import suppress
from dataclasses import dataclass, fields, is_dataclass
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Final,
    Literal,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import numpy as np

CellKind = Literal["bool", "text", "number", "date", "enum"]


@dataclass(frozen=True, slots=True)
class CellSpec:
    kind: CellKind = "text"
    enum: type[Enum] | None = None


TEXT_SPEC: Final = CellSpec("text")
BOOL_SPEC: Final = CellSpec("bool")
NUMBER_SPEC: Final = CellSpec("number")
DATE_SPEC: Final = CellSpec("date")


# ── type unwrapping ──────────────────────────────────────────────────────────


def _unwrap(tp: Any) -> Any:
    """Unwrap ``Annotated`` / ``Optional`` / ``Union`` — return first concrete type."""
    while True:
        origin = get_origin(tp)
        if origin is Annotated:
            tp = get_args(tp)[0]
        elif origin is Union or isinstance(origin, types.UnionType):
            args = tuple(a for a in get_args(tp) if a is not type(None))
            if len(args) == 1:
                tp = args[0]
            else:
                return args[0] if args else tp
        else:
            return tp


# ── classification ───────────────────────────────────────────────────────────


def classify_type(tp: Any) -> CellSpec:
    """Map a (possibly wrapped) Python type → ``CellSpec``."""
    tp = _unwrap(tp)
    if isinstance(tp, type):
        if issubclass(tp, bool):
            return BOOL_SPEC
        if issubclass(tp, Enum):
            return CellSpec("enum", tp)
        if issubclass(tp, (str, os.PathLike, Path)):
            return TEXT_SPEC
        if issubclass(tp, (int, float, Decimal)):
            return NUMBER_SPEC
        if issubclass(tp, (date, datetime, time)):
            return DATE_SPEC
    return TEXT_SPEC


# ── dataclass path resolution ────────────────────────────────────────────────


def resolve_dataclass_field(root: Any, path: str) -> Any:
    """Walk the dataclass tree along *path* (e.g. ``"program.return_"``) and return the leaf type."""
    tp = root
    for part in filter(None, path.split(".")):
        tp = _unwrap(tp)
        if not is_dataclass(tp):
            return Any
        dc = tp if isinstance(tp, type) else type(tp)
        with suppress(NameError, TypeError):
            hints = get_type_hints(dc, include_extras=True)
        if "hints" not in locals() or part not in hints:
            hints = {f.name: f.type for f in fields(dc)}
        if part not in hints:
            return Any
        tp = hints[part]
    return _unwrap(tp)


@functools.lru_cache(maxsize=4096)
def spec_for_path(
    root: type | None,
    path: str,
    return_enum: type[Enum] | None,
) -> CellSpec:
    """Resolve *path* against *root* dataclass, with ``program.return_`` override."""
    if path == "program.return_" and return_enum is not None:
        return CellSpec("enum", return_enum)
    if root is None:
        return TEXT_SPEC
    return classify_type(resolve_dataclass_field(root, path))


# ── OmegaConf → dataclass type ──────────────────────────────────────────────


def schema_type(cfg: Any) -> type | None:
    """Extract the structured-config dataclass type from an OmegaConf DictConfig."""
    try:
        from omegaconf import OmegaConf

        return OmegaConf.get_type(cfg)
    except (ImportError, AttributeError, TypeError):
        return cfg if isinstance(cfg, type) else type(cfg)


# ── helpers for tksheet widgets ──────────────────────────────────────────────


def enum_values(enum: type[Enum]) -> list[str]:
    """Return display values for an Enum (values for StrEnum, names otherwise)."""
    return [str(e.value) for e in enum]


def as_bool(value: Any) -> bool:
    """Coerce a cell value to bool (handles str variants like 'true', 'on')."""
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def any2str(v: Any) -> str:
    """
    String formatting for cell display: :g for floats, str for other types.

    :param v: _description_
    :return: _description_
    """
    if v is None:
        return ""
    if isinstance(v, (float, np.floating)):
        return f"{v:g}"
    return str(v)


def parse_float(v: str) -> float | None:
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def as_date(s: str) -> bool:
    """Create datetime if s is an ISO date string or looks like an naive European dd.mm.yyyy"""
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        parts = s.split(".")
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            try:
                return datetime.strptime(s, "%d.%m.%Y")
            except ValueError:
                return None
