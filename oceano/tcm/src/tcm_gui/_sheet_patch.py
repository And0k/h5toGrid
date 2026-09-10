"""Generic sheet → YAML patch builder — dataclass-typed "write all non-default".

Pure-logic sidekick to :class:`tcm_gui.coef_sheet.ConfigSheet`: turns the
sheet's ``_meta`` + cell strings into the minimal override patch that
``config_yaml.update_run_yaml`` merges. No tkinter here — the caller injects
a ``cell_str(iid, col)`` reader, so the builder is unit-testable headless.

Type source is the Hydra structured-config dataclass (``_config_root``),
resolved via :func:`tcm_gui._cell_spec.resolve_dataclass_field` — never the
``is_string`` / ``max_col`` / ``"["`` display heuristics.  Date rows
(``check: "sorted"`` — ``input.time_ranges``, ``input.calib.time_ranges_*``)
are validated before writing: every value is canonicalized through
:func:`tcm_gui._cell_spec.iso_secs` (ISO ``T``-separated seconds) and a row
holding an unparseable cell is **skipped** with a warning — the stored YAML
value survives instead of a mixed-format list that ``pd.to_datetime`` would
reject at load (see :func:`tcm_gui._cell_spec.iso_secs`).  Special subtrees
keep their dedicated write paths and are skipped here:

* ``input.path`` + ``input.coefs`` (matrix/date/``dates`` machinery);
* ``metadata*`` rows — owned by the metadata node mixin;
* coef date cells (``has_date``).

``int`` vs ``float`` is strict: ``out.dt_bins: list[int]`` accepts only
canonical int spellings — ``"600.0"`` skips the row instead of silently
writing ``600.0`` (the old ``parse_float``-for-everything bug).
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import is_dataclass
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Final, get_args, get_origin

from tcm_gui._cell_spec import _unwrap, as_bool, iso_secs, parse_float, resolve_dataclass_field
from tcm_gui.cli_cfg import NO_DEFAULT, default_for_path

lf = logging.getLogger(__name__)

_LEAF_RE: Final = re.compile(r"\[\d+\]")
_SKIP_TOP: Final = frozenset({"defaults", "hydra", "metadata"})
_COEFS_PREFIX: Final = "input.coefs"
_INPUT_PATH: Final = "input.path"


def _base(path: str) -> str:
    """Strip ``[i]`` indices — ``out.dt_bins[2]`` → ``out.dt_bins``."""
    return _LEAF_RE.sub("", path)


def _leaf_kind(root: Any, base: str) -> tuple[str, type[Enum] | None]:
    """Dataclass leaf kind for *base*: ``bool|int|float|text|date|enum|dict|list|none``."""
    if root is None:
        return "text", None
    tp = _unwrap(resolve_dataclass_field(root, base))
    if tp is Any or tp is None:
        return ("none", None) if tp is None else ("text", None)
    if isinstance(tp, type):
        if issubclass(tp, bool):
            return "bool", None
        if issubclass(tp, Enum):
            return "enum", tp
        if issubclass(tp, (str, Path, os.PathLike)):
            return "text", None
        if issubclass(tp, int):
            return "int", None
        if issubclass(tp, (float, Decimal)):
            return "float", None
        if issubclass(tp, (date, datetime, time)):
            return "date", None
        if is_dataclass(tp):
            return "dict", None
    if get_origin(tp) is list:
        return "list", None
    if get_origin(tp) is dict:
        return "dict", None
    return "text", None


def _elem_kind(root: Any, base: str) -> str:
    """Element kind for ``list[T]`` at *base* — ``bool|int|float|text|date``."""
    tp = _unwrap(resolve_dataclass_field(root, base)) if root is not None else Any
    if get_origin(tp) is list and (args := get_args(tp)):
        inner = _unwrap(args[0])
        if isinstance(inner, type):
            if issubclass(inner, bool):
                return "bool"
            if issubclass(inner, int):
                return "int"
            if issubclass(inner, (float, Decimal)):
                return "float"
            if issubclass(inner, (date, datetime, time)):
                return "date"
            if issubclass(inner, (str, Path)):
                return "text"
    return "text"


def parse_int_strict(v: str) -> int | None:
    """Strict int parse — ``"600"`` ok; ``"600.0"``/``"1e3"``/``"abc"`` → None (row skipped)."""
    s = v.strip()
    if not s or not re.fullmatch(r"[+-]?\d+", s):
        return None
    with suppress(ValueError):
        return int(s)
    return None


def convert_cell(kind: str, v: str) -> Any:
    """Convert one cell string per leaf *kind*; None = unparseable (row skipped)."""
    if kind == "bool":
        return as_bool(v)
    if kind == "int":
        return parse_int_strict(v)
    if kind == "float":
        return parse_float(v)
    return v  # text/date/enum — verbatim (date validity vetted by _on_edit)


def _scalar_equal(kind: str, a: Any, b: Any) -> bool:
    """True when sheet-read *a* matches default *b* (int/float/bool-aware)."""
    if b is None:
        return False  # non-empty cell vs None default → changed
    if kind == "int":
        with suppress(TypeError, ValueError):
            return int(a) == int(b)
    if kind == "float":
        with suppress(TypeError, ValueError):
            return float(a) == float(b)
    if kind == "bool":
        return bool(a) == bool(b)
    return str(a) == str(b)


def _values_equal(kind: str, conv: list, dflt: Any) -> bool:
    """True when sheet-read *conv* matches default *dflt* (shape + value)."""
    if not isinstance(dflt, (list, tuple)):
        return len(conv) == 1 and _scalar_equal(kind, conv[0], dflt)
    return len(conv) == len(dflt) and all(_scalar_equal(kind, a, b) for a, b in zip(conv, dflt))


def assign_dotted(root: dict, path: str, value: Any) -> None:
    """Assign *value* into nested *root* along dotted *path* with ``[i]`` indices."""
    node: Any = root
    parts = path.split(".")
    for part in parts[:-1]:
        if "[" in part:
            name, idx = part[:-1].split("[", 1)
            lst = node.setdefault(name, [])
            i = int(idx.rstrip("]"))
            while len(lst) <= i:
                lst.append({})
            node = lst[i]
        else:
            nxt = node.get(part)
            if not isinstance(nxt, dict):
                nxt = node[part] = {}
            node = nxt
    last = parts[-1]
    if "[" in last:
        name, idx = last[:-1].split("[", 1)
        lst = node.setdefault(name, [])
        if not isinstance(lst, list):
            lst = node[name] = []
        i = int(idx.rstrip("]"))
        while len(lst) <= i:
            lst.append(None)
        lst[i] = value
    else:
        node[last] = value


def is_skipped(path: str, meta: Mapping[str, Any]) -> bool:
    """True when *path* keeps its dedicated write path (never generic)."""
    if not path or path.startswith("_") or path.split(".", 1)[0] in _SKIP_TOP:
        return True
    if meta.get("is_metadata") or meta.get("is_metadata_root") or meta.get("has_date"):
        return True
    return path == _INPUT_PATH or path == _COEFS_PREFIX or path.startswith(_COEFS_PREFIX + ".")


def is_leaf(root: Any, base: str, leaf: str, meta: Mapping[str, Any]) -> bool:
    """True when the row holds values itself (not a dict/2-D parent container)."""
    kind, _ = _leaf_kind(root, base)
    if kind == "dict":
        return False  # values live on children
    # bare list node w/o own width — children carry [i]; flat rows (g0xyz, dt_bins) have max_col
    return not (kind == "list" and "[" not in leaf and not meta.get("max_col"))


def row_values(cell_str: Callable[[Any, int], str], iid: Any, width: int) -> list[str]:
    """Read + right-trim empties; [] = empty row (at-default, omitted)."""
    vals = [cell_str(iid, j) for j in range(width)]
    while vals and not vals[-1]:
        vals.pop()
    return vals


def build_patch(
    meta_items: Mapping[Any, Mapping[str, Any]],
    cell_str: Callable[[Any, int], str],
    nv: int,
    config_root: Any = None,
    return_enum: type[Enum] | None = None,
    sections: tuple[str, ...] | None = None,
) -> dict:
    """Generic "write all non-default" patch over sheet rows.

    :param meta_items: ``{iid: meta}`` with ``path`` (+``max_col`` for width).
    :param cell_str: ``(iid, col) → trimmed str`` (ghost-aware).
    :param nv: fallback row width.
    :param config_root: Hydra structured-config dataclass for typing.
    :param return_enum: ``program.return_`` StrEnum override (unused — enums verbatim).
    :param sections: top-level sections to cover (None = all non-skipped).
    """
    patch: dict[str, Any] = {}
    for iid, m in meta_items.items():
        path = m.get("path") or ""
        if is_skipped(path, m):
            continue
        if sections is not None and path.split(".", 1)[0] not in sections:
            continue
        base = _base(path)
        leaf = path.rsplit(".", 1)[-1]
        if not is_leaf(config_root, base, leaf, m):
            continue
        if not (vals := row_values(cell_str, iid, int(m.get("max_col") or nv))):
            continue
        kind, _enum_tp = _leaf_kind(config_root, base)
        if kind in ("dict", "none"):
            continue
        if kind == "list":
            kind = _elem_kind(config_root, base)
        if m.get("check") == "sorted":
            # Date rows: validate format + write canonical ISO-T (see iso_secs) —
            # verbatim cell strings carry the space-separated display form, and a
            # mixed space/T list crashes pd.to_datetime at load.
            if any(v and iso_secs(v) is None for v in vals):
                lf.warning(
                    "Unparseable date cell(s) %s in %s — row write skipped, stored value kept", vals, path
                )
                continue
            vals_conv = [iso_secs(v) or v for v in vals]
        else:
            vals_conv = (
                list(vals) if kind in ("text", "date", "enum") else [convert_cell(kind, v) for v in vals]
            )
        if any(v is None for v in vals_conv):
            continue
        dflt = default_for_path(base)
        cmp_kind = kind if kind in ("int", "float", "bool") else "text"
        if dflt is not NO_DEFAULT and _values_equal(cmp_kind, vals_conv, dflt):
            continue  # at default — run YAMLs carry overrides only
        assign_dotted(
            patch, path, vals_conv if isinstance(dflt, (list, tuple)) or len(vals_conv) > 1 else vals_conv[0]
        )
    return patch
