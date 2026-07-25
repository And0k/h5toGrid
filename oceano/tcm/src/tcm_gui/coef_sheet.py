"""Config tree in tksheet ≥ 7 treeview."""
from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping
from contextlib import suppress
from datetime import datetime
from itertools import islice
from tkinter import TclError, ttk
from typing import Any, Final, Literal

import numpy as np
from tksheet import Sheet

from tcm.config import (
    ConfigFilter_InclProc,
    ConfigIn_InclProc,
    ConfigInCoefs_InclProc,
    ConfigOut_InclProc,
    ConfigProgram,
)
from tcm.to_omegaconf import get_field_default

from ._browse_button import BrowseButtonManager
from ._cell_spec import NUMBER_SPEC, CellSpec, _spec_for_path, as_bool, enum_values, schema_type

_lg = logging.getLogger(__name__)

# Derive field order from dataclass declaration — single source of truth.
# Exclude `dates` / `date` which are handled as tree-level metadata, not row items.
_COEF_FIELDS = [f.name for f in dataclasses.fields(ConfigInCoefs_InclProc) if f.name not in ("dates", "date")]


def _build_defaults(schema: type) -> dict[str, Any]:
    """Build `{field_name: default}` from a dataclass using :func:`get_field_default`.

    Nested dataclasses are recursively expanded into sub-dicts.
    """
    result: dict[str, Any] = {}
    for fld in dataclasses.fields(schema):
        d = get_field_default(fld)
        result[fld.name] = _build_defaults(type(d)) if dataclasses.is_dataclass(type(d)) else d
    return result


# Nested defaults for every config section — single source of truth for gray-out.
_CFG_DEFAULTS: dict[str, dict[str, Any]] = {
    section: _build_defaults(cls)
    for section, cls in [
        ("input", ConfigIn_InclProc),
        ("out", ConfigOut_InclProc),
        ("filter", ConfigFilter_InclProc),
        ("program", ConfigProgram),
    ]
}


def _default_for_path(path: str) -> Any:
    """Walk dotted config path through :data:`_CFG_DEFAULTS`, return default or `_NO_DEFAULT`.

    Handles array indices (e.g. ``Ag[0]``) and ``None`` → ``""``.
    """
    parts = path.split(".")
    if not parts or parts[0] not in _CFG_DEFAULTS:
        return _NO_DEFAULT
    current: Any = _CFG_DEFAULTS[parts[0]]
    for part in parts[1:]:
        if current is None:
            return ""
        # Handle array indices like "Ag[0]"
        if "[" in part:
            name, idx_str = part.split("[", 1)
            idx = int(idx_str.rstrip("]"))
        else:
            name, idx = part, None
        if isinstance(current, dict) and name in current:
            current = current[name]
        else:
            return _NO_DEFAULT
        if idx is not None:
            if current is None:
                return ""
            if isinstance(current, (list, tuple)) and idx < len(current):
                current = current[idx]
            else:
                return _NO_DEFAULT
    return current


_DEFAULT_FG: Final[str] = "#999999"  # cell value == config default
_BLUE_FG: Final[str] = "#0055CC"     # header text + node label when subtree at default
_NO_DEFAULT = object()

_COEF_SHAPES: dict[str, tuple[int, ...]] = {
    "Ag": (3, 3),
    "Cg": (3,),
    "Ah": (3, 3),
    "Ch": (3,),
    "Rz": (3, 3),
    "kVabs": (6,),
    "P_t": (3, 3),
    "P": (2,),
    "PBattery": (2,),
    "PTemp": (2,),
    "azimuth_shift_deg": (),
    "g0xyz": (3,),
}
_1D_WITH_DATES = {"kVabs"}  # единственное 1D с датами → parent+child
_SUB = "₁₂₃₄₅₆₇₈₉"
_DATE_COL = 2  # sheet col: 0=tree 1=₁ 2=₂/date 3=₃…

TkAlign = Literal["w", "center", "e"]
_ALIGN: Final[Mapping[str, TkAlign]] = {
    "left": "w",
    "right": "e",
    "center": "center",
    "w": "w",
    "e": "e",
}


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (float, np.floating)):
        return f"{v:g}"
    return str(v)


def _pf(v: str) -> float | None:
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _is_num(s: str) -> bool:
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def _is_date(s: str) -> bool:
    """Check if s looks like a date — fast path via `fromisoformat`."""
    try:
        datetime.fromisoformat(s)
        return True
    except ValueError:
        # European dd.mm.yyyy — validate structure only (no naive datetime created)
        parts = s.split(".")
        return len(parts) == 3 and all(p.isdigit() for p in parts)


def _resolve_bg(widget, color: str) -> str:
    """Convert any Tk color spec to hex — tksheet rejects system names like 'SystemButtonFace'."""
    try:
        r, g, b = widget.winfo_rgb(color)
        return f"#{r >> 8:02x}{g >> 8:02x}{b >> 8:02x}"
    except TclError:
        return color


class ConfigSheet:
    """Wraps tksheet.Sheet(treeview=True) for config display / editing.

    Row spaces — the central tksheet treeview subtlety:
      * internal rows: data-model indices, hidden (collapsed) rows included —
        what every cell API (``highlight_cells``, ``get_cell_data``, widgets)
        consumes; produced by :meth:`_row_map`.
      * display rows: visible-only positions, collapsed subtrees compressed
        out — what edit events report; decoded by :meth:`_iid_at_row` via
        :meth:`_walk_visible`, whose open oracle is the ``item()`` wrapper
        (7.6's getter exposes no ``"open"`` key — bookkeeping is the only
        reliable source).
    """

    # Meta columns are 1-based; tksheet data column = meta_col - DATA_COL_BASE
    DATA_COL_BASE: int = 1

    def __init__(self, parent) -> None:
        self.sh = Sheet(parent, treeview=True, show_horizontal_grid=False, show_vertical_grid=False)
        raw_bg = ttk.Style().lookup("TFrame", "background") or "#F0F0F0"
        bg = _resolve_bg(self.sh, raw_bg)
        self.sh.set_options(allow_cell_overflow=True, header_background=bg)
        self.sh.enable_bindings(["all"])
        self.sh.edit_validation(self._on_edit)
        self.sh.extra_bindings([
            ("begin_edit_cell", self._on_begin_edit_cell),
            ("end_edit_cell", self._on_end_edit_cell),
        ])
        self._meta: dict[Any, dict] = {}
        self._nv = 6
        self._full = False
        self._cfg: dict = {}
        # Hydra structured-config root type for cell classification
        self._config_root: type | None = None
        # StrEnum for program.return_ dropdown
        self._return_enum: type | None = None
        # Snapshot of all editable cells for dirty tracking — populated at end of load()
        self._snap: tuple = ()
        # Normal (non-default) text color — "clear" side of gray/blue toggles (_apply_styles)
        self._fg_default: str = "#000000"
        # Open-state bookkeeping: 7.6's item() getter has no "open" key, so
        # wrap item() and record every change — ours and tksheet's internal
        # arrow toggles, which route through the same public method.
        self._item_orig = self.sh.item
        self.sh.item = self._item_hook
        # BrowseButtonManager — injected by App after construction
        self._mgr: BrowseButtonManager | None = None

    # ── public API ──────────────────────────────────────────────────

    def load(
        self, cfg: dict, full: bool = False, config_root: type | None = None, return_enum: type | None = None
    ) -> None:
        self._cfg, self._full = cfg, full
        self._config_root = config_root if config_root is not None else schema_type(cfg)
        self._return_enum = return_enum
        self._meta.clear()
        self.sh.del_rows(rows=list(range(self.sh.total_rows())))
        self.sh.enable_bindings(["all"])
        self._nv = self._calc_nv(cfg, full)
        self.sh.headers(list(_SUB[: self._nv]))
        (self._build_full if full else self._build_coefs)(cfg)
        self._apply_open()
        self._apply_styles()
        self._apply_default_fg()
        self.sh.redraw()
        self._take_snapshot()

    def get_edited_coefs(self) -> dict[str, Any]:
        """Read leaf values → coefs dict for YAML write-back."""
        out: dict[str, Any] = {}
        for iid, m in self._meta.items():
            t = m.get("type")
            if t == "2d":
                nr, nc = m["shape"]
                out[m["key"]] = [
                    [_pf((self.sh.item(c).get("values") or ())[j]) or 0.0 for j in range(nc)]
                    for c in m["children"][:nr]
                ]
            elif t == "1d":  # kVabs: parent+child
                vals = self.sh.item(m["child"]).get("values") or ()
                out[m["key"]] = [_pf(vals[j]) or 0.0 for j in range(min(m["len"], len(vals)))]
            elif t == "1d_flat":  # Cg, Ch, …: одна строка
                vals = self.sh.item(iid).get("values") or ()
                out[m["key"]] = [_pf(vals[j]) or 0.0 for j in range(min(m["len"], len(vals)))]
            elif t == "scalar":
                if v := (self.sh.item(iid).get("values") or ("",))[0]:
                    out[m["key"]] = float(v)
        return out

    def get_edited_dates(self) -> dict[str, str]:
        return {
            m["key"]: d
            for iid, m in self._meta.items()
            if m.get("has_date")
            and (vals := self.sh.item(iid).get("values") or ())
            and len(vals) > 1
            and (d := vals[1])  # date in values[1]
        }

    def get_edited_input_path(self) -> str:
        for iid, m in self._meta.items():
            if m.get("type") == "input":
                return (self.sh.item(iid).get("values") or ("",))[0]
        return ""

    # ── dirty tracking ───────────────────────────────────────────────

    def _current_state(self) -> tuple[dict, dict, str]:
        """Return coefs/dates/path state for YAML write-back (app._write_coefs)."""
        return self.get_edited_coefs(), self.get_edited_dates(), self.get_edited_input_path()

    def _data_snapshot(self) -> tuple[tuple, ...]:
        """Hashable snapshot of ALL editable leaf cell values.

        Captures every row that has ``max_col > 0`` (i.e. contains editable
        data cells) — covers input, out, filter, program sections, and coefs.
        """
        parts = []
        for iid, m in self._meta.items():
            max_col = int(m.get("max_col") or m.get("len") or 0)
            if m.get("type") == "scalar":
                max_col = 1
            if max_col == 0:
                continue
            vals = self.sh.item(iid).get("values") or ()
            parts.append((iid, tuple(str(vals[j]) if j < len(vals) else "" for j in range(max_col))))
        return tuple(parts)

    def _take_snapshot(self) -> None:
        """Capture current cell data as the *clean* baseline."""
        self._snap = self._data_snapshot()

    @property
    def is_dirty(self) -> bool:
        """True when any editable cell differs from last load/save snapshot."""
        return self._data_snapshot() != self._snap

    def mark_clean(self) -> None:
        """Reset dirty flag after a successful write-back."""
        self._take_snapshot()

    # ── tree construction ───────────────────────────────────────────

    def _calc_nv(self, cfg: dict, full: bool) -> int:
        nv = 6
        for v in cfg.get("input", {}).get("coefs", {}).values():
            if isinstance(v, (list, np.ndarray)) and len(v):
                nv = max(nv, len(v[0]) if isinstance(v[0], (list, np.ndarray)) else len(v))
        if full:
            for sec in cfg.values():
                if isinstance(sec, dict):
                    for v in sec.values():
                        if isinstance(v, (list, np.ndarray)):
                            nv = max(nv, len(v))
        return nv

    def _build_coefs(self, cfg: dict) -> None:
        inp = cfg.get("input", {})
        path = _fmt(inp.get("path", ""))
        inp_iid = self._ins(
            "",
            "input",
            [path] + [""] * (self._nv - 1),
            "",
            meta={"key": "input", "type": "input", "is_string": True, "max_col": 1, "style": "node", "browse": True},
            open_=True,
        )
        if tr := inp.get("time_ranges", []):
            self._ins(
                inp_iid,
                "time_ranges",
                [_fmt(x) for x in tr] + [""] * (self._nv - len(tr)),
                "",
                meta={"is_string": True, "max_col": len(tr)},
            )
        coefs_path = _fmt(inp.get("coefs_path", ""))
        self._ins(
            inp_iid,
            "coefs_path",
            [coefs_path] + [""] * (self._nv - 1),
            "",
            meta={"key": "coefs_path", "is_string": True, "max_col": 1, "path": "input.coefs_path", "browse": True},
        )
        coefs = inp.get("coefs", {})
        dates = coefs.get("dates", {})
        cdate_src = coefs.get("date")
        cdate = cdate_src or (max(dates.values()) if dates else "")
        coefs_iid = self._ins(
            inp_iid,
            "coefs",
            [""] * self._nv,
            cdate,
            meta={
                "key": "coefs",
                "has_date": True,
                "max_col": 0,
                "date_style": "blue" if not cdate_src else None,
            },
            open_=False,
        )
        for name in _COEF_FIELDS:
            if name in coefs:
                self._ins_coef(coefs_iid, name, coefs.get(name), dates.get(name, ""))

    def _build_full(self, cfg: dict) -> None:
        for sec, val in cfg.items():
            if sec == "input":
                self._build_input(val)
            elif isinstance(val, dict):
                sid = self._ins("", sec, [""] * self._nv, "")
                for k, v in val.items():
                    self._ins_generic(sid, k, v)
            else:
                self._ins_leaf("", sec, val)

    def _build_input(self, inp: dict) -> None:
        path = _fmt(inp.get("path", ""))
        inp_iid = self._ins(
            "",
            "input",
            [path] + [""] * (self._nv - 1),
            "",
            meta={"key": "input", "type": "input", "is_string": True, "max_col": 1, "style": "node", "browse": True},
            open_=True,
        )
        coefs_path = _fmt(inp.get("coefs_path", ""))
        self._ins(
            inp_iid,
            "coefs_path",
            [coefs_path] + [""] * (self._nv - 1),
            "",
            meta={"key": "coefs_path", "is_string": True, "max_col": 1, "path": "input.coefs_path", "browse": True},
        )
        for k, v in inp.items():
            if k in ("path", "coefs_path"):
                continue
            if k == "coefs":
                dates = v.get("dates", {})
                cdate = v.get("date") or (max(dates.values()) if dates else "")
                cid = self._ins(
                    inp_iid,
                    "coefs",
                    [""] * self._nv,
                    cdate,
                    meta={"key": "coefs", "has_date": True, "max_col": 0},
                )
                for name in _COEF_FIELDS:
                    if name in v:
                        self._ins_coef(cid, name, v.get(name), dates.get(name, ""))
            else:
                self._ins_generic(inp_iid, k, v)

    # ── coef inserters ──────────────────────────────────────────────

    def _ins_coef(self, par: Any, name: str, value: Any, date: str) -> None:
        shape = _COEF_SHAPES.get(name, ())
        if len(shape) == 2:
            self._ins_2d(par, name, value, shape, date)
        elif len(shape) == 1 and name in _1D_WITH_DATES:
            self._ins_1d(par, name, value, shape[0], date)
        elif len(shape) == 1:
            self._ins_1d_flat(par, name, value, shape[0])
        else:
            self._ins_scalar(par, name, value)

    def _ins_2d(self, par, name, value, shape, date) -> None:
        nr, nc = shape
        children: list = []
        pid = self._ins(
            par,
            name,
            [""] * self._nv,
            date,
            meta={
                "key": name,
                "type": "2d",
                "children": children,
                "has_date": True,
                "shape": shape,
                "max_col": 0,
            },
        )
        parent_path = self._meta[pid]["path"]
        for i in range(nr):
            row = [_fmt(x) for x in value[i]] if value else [""] * nc
            children.append(
                self._ins(
                    pid,
                    f"{name}[{i}]",
                    row + [""] * (self._nv - len(row)),
                    "",
                    meta={"max_col": nc, "type": "_coef_child", "path": f"{parent_path}[{i}]"},
                )
            )

    def _ins_1d(self, par, name, value, n, date):
        """1D с датами (kVabs): parent + child."""
        pid = self._ins(
            par,
            name,
            [""] * self._nv,
            date,
            meta={"key": name, "type": "1d", "child": None, "has_date": True, "len": n, "max_col": 0},
        )
        row = [_fmt(x) for x in value] if value else [""] * n
        # Child holds the array values — same config path as parent
        self._meta[pid]["child"] = self._ins(
            pid,
            name,
            row + [""] * (self._nv - len(row)),
            "",
            meta={"max_col": n, "type": "_coef_child", "path": self._meta[pid]["path"]},
        )

    def _ins_1d_flat(self, par, name, value, n):
        """1D без дат (Cg, Ch, P, …): одна строка, без детей."""
        row = [_fmt(x) for x in value] if value else [""] * n
        self._ins(
            par,
            name,
            row + [""] * (self._nv - len(row)),
            "",
            meta={"key": name, "type": "1d_flat", "len": n, "max_col": n},
        )

    def _ins_scalar(self, par, name, value):
        self._ins(
            par,
            name,
            [_fmt(value)] + [""] * (self._nv - 1),
            "",
            meta={"key": name, "type": "scalar", "max_col": 1},
        )

    def _ins_generic(self, par: Any, key: str, value: Any) -> None:
        if isinstance(value, dict):
            sid = self._ins(par, key, [""] * self._nv, "")
            for k, v in value.items():
                self._ins_generic(sid, k, v)
        elif isinstance(value, (list, np.ndarray)):
            if not value:  # empty [] / np.array([]) → show nothing
                self._ins_leaf(par, key, None)
            elif isinstance(value[0], (list, np.ndarray)):
                sid = self._ins(par, key, [""] * self._nv, "")
                sid_path = self._meta[sid]["path"]
                for i, row in enumerate(value):
                    self._ins(
                        sid,
                        f"{key}[{i}]",
                        [_fmt(x) for x in row] + [""] * (self._nv - len(row)),
                        "",
                        meta={"path": f"{sid_path}[{i}]"},
                    )
            else:
                self._ins(
                    par,
                    key,
                    [_fmt(x) for x in value] + [""] * (self._nv - len(value)),
                    "",
                    meta={"is_string": True, "max_col": len(value)},
                )
        else:
            self._ins_leaf(par, key, value)

    def _ins_leaf(self, par: Any, key: str, value: Any) -> None:
        self._ins(par, key, [_fmt(value)] + [""] * (self._nv - 1), "", meta={"is_string": True, "max_col": 1})

    # ── helpers ─────────────────────────────────────────────────────

    def _item_hook(self, iid=None, *args, **kwargs):
        """``sh.item`` wrapper — the open-state oracle.

        7.6's getter returns no ``"open"`` key, so every state change
        (our calls and tksheet's internal arrow toggles, which route
        through this same public method) is recorded in ``meta["open"]``;
        :meth:`_walk_visible` consumes it.
        """
        if iid is not None and kwargs.get("open_") is not None and iid in self._meta:
            self._meta[iid]["open"] = bool(kwargs["open_"])
        return self._item_orig(iid, *args, **kwargs)

    def _ins(self, parent_iid, text, vals, date="", meta=None, open_=False):
        if meta is None:
            meta = {}
        meta.setdefault("open", open_)
        # Backlink for ancestor traversal (blue-label propagation in _on_end_edit)
        meta.setdefault("parent", parent_iid or None)
        # Compute Hydra config path from parent path + node text
        parent_path = self._meta.get(parent_iid, {}).get("path", "")
        meta.setdefault("path", f"{parent_path}.{text}" if parent_path else text)
        if date and len(vals) > 1:
            vals = list(vals)
            vals[1] = date
            if meta is not None:
                meta.setdefault("meta_date_cols", []).append(2)
        # даты по содержимому
        if dc := [c + 1 for c, v in enumerate(vals) if v and _is_date(str(v))]:
            meta.setdefault("date_cols", []).extend(dc)
        try:
            iid = self.sh.insert(parent=parent_iid, text=text, values=vals, open_=open_)
        except TypeError:
            iid = self.sh.insert(parent=parent_iid, text=text, values=vals)
            if open_:
                with suppress(TclError):
                    self.sh.item(iid, open_=True)  # hook records the state
        if meta:
            self._meta[iid] = meta
        return iid

    def _on_edit(self, event) -> str | None:
        """Validate incoming cell edit.

        ``event.row`` is a **display** row, ``event.column`` the 0-based
        **data** column (tree column lives on the index canvas and never
        reaches this callback).

        Date col: accept only date strings.
        Beyond ``max_col``: reject.
        ``is_string`` rows: accept any text.
        Otherwise: accept only numeric values.
        """
        c = event.column
        val = event.value
        if not val or not val.strip():
            return val
        iid = self._iid_at_row(event.row)
        m = self._meta.get(iid, {})
        if c == _DATE_COL - self.DATA_COL_BASE and m.get("has_date"):
            result = val if _is_date(val) else None
            _lg.debug(
                "edit r=%s c=%s iid=%s path=%s val=%r → %s (date)",
                event.row,
                c,
                iid,
                m.get("path"),
                val,
                "ok" if result else "REJECT",
            )
            return result
        if c >= m.get("max_col", self._nv):
            _lg.debug(
                "edit r=%s c=%s iid=%s path=%s → REJECT (beyond max_col=%s)",
                event.row,
                c,
                iid,
                m.get("path"),
                m.get("max_col"),
            )
            return None
        if m.get("is_string"):
            return val
        if _is_num(val):
            return val
        _lg.debug(
            "edit r=%s c=%s iid=%s path=%s val=%r → REJECT (not numeric)",
            event.row,
            c,
            iid,
            m.get("path"),
            val,
        )
        return None

    # ── edit lifecycle ──────────────────────────────────────────────

    def _on_begin_edit_cell(self, event) -> str | None:
        """Detach any previous browse button; attach for path-type rows."""
        iid = self._iid_at_row(event.row)
        m = self._meta.get(iid, {})
        if self._mgr is not None:
            self._mgr.detach()  # always clean up previous button first
        if m.get("browse") and self._mgr is not None:
            self._mgr.attach(self._row_map()[iid], event.column)
        return self.sh.get_cell_data(event.row, event.column)

    def _on_end_edit_cell(self, event) -> None:
        """Detach browse button (any row); reload coefs_path if changed."""
        if self._mgr is not None:
            self._mgr.detach()
        iid = self._iid_at_row(event.row)
        m = self._meta.get(iid, {})
        if m.get("key") == "coefs_path":
            new_path = str(event.value) if event.value else ""
            if new_path.strip() and self._mgr is not None:
                self.sh.after_idle(lambda p=new_path: self._mgr.notify_path_changed(p))
        # existing style logic
        self._apply_end_edit_style(event)

    def _style_header(self, bg: str, fg: str) -> None:
        """Color header row and top-left corner (node column styled per-row in :meth:`_apply_styles`)."""
        sh = self.sh
        # global header fg/bg — most reliable path for tksheet 7.x
        with suppress(AttributeError, TypeError):
            sh.set_options(header_fg=fg, header_bg=bg)
        # per-cell header highlight (more reliable than set_options in some builds)
        for c in range(sh.total_columns()):
            sh.highlight_cells(row=0, column=c, canvas="header", bg=bg, fg=fg, redraw=False)
        # top-left corner cell (intersection of header + index/tree)
        with suppress(AttributeError, TypeError, ValueError, KeyError):
            sh.highlight_cells(row=0, column=0, canvas="topleft", bg=bg, redraw=False)

    def _apply_open(self) -> None:
        """Re-apply desired open states stored in meta during construction.

        Workaround for tksheet 7.6 ``insert(open_=…)`` bug where styling
        operations may collapse/expand nodes unexpectedly.  Goes through
        the ``item()`` wrapper, so bookkeeping stays consistent.
        """
        for iid, m in list(self._meta.items()):
            if m.get("open"):
                with suppress(AttributeError, TclError, TypeError):
                    self.sh.item(iid, open_=True)

    def _row_map(self) -> dict[Any, int]:
        """Map treeview iid → **internal** tksheet row — the cell-API row space.

        DFS over *all* items: collapsed children keep their internal indices
        (``highlight_cells`` / ``get_cell_data`` / widgets address them even
        while hidden).  Never use for event rows — see :meth:`_iid_at_row`.
        """
        def _walk(parent: Any = ""):
            for cid in self.sh.get_children(parent):
                yield cid
                with suppress(AttributeError, TclError, TypeError, ValueError, KeyError):
                    yield from _walk(cid)

        return {iid: r for r, iid in enumerate(_walk()) if iid in self._meta}

    def _iid_at_row(self, r: int) -> Any | None:
        """Map **display** row (``event.row``) → tree iid.

        Display rows compress collapsed subtrees away, so walk visible
        items only; internal-row conversion is :meth:`_row_map` afterwards.
        """
        return next(islice(self._walk_visible(), r, r + 1), None)

    def _walk_visible(self, parent: Any = ""):
        """DFS over visible items — collapsed subtrees yield no rows."""
        for cid in self.sh.get_children(parent):
            yield cid
            if self._meta.get(cid, {}).get("open"):
                yield from self._walk_visible(cid)

    # ── type-aware cell widget helpers ──────────────────────────────

    # Coef meta types — always numeric; skip Hydra path resolution for these.
    # Also tag array/vector children with ``_coef_child`` during construction.
    _COEF_TYPES = frozenset({"2d", "1d", "1d_flat", "scalar", "_coef_child"})

    def _cell_spec_for(self, iid: str, m: Mapping[str, Any], meta_col: int) -> CellSpec:
        """Resolve ``CellSpec`` for a cell at *meta_col* in row *iid*."""
        # Coef parent rows are always numeric
        if m.get("type") in self._COEF_TYPES:
            return NUMBER_SPEC
        # Strip array indices (e.g. "input.coefs.Ag[0]" → "input.coefs.Ag")
        path = str(m.get("path", iid))
        clean = path.split("[")[0] if "[" in path else path
        return _spec_for_path(self._config_root, clean, self._return_enum)

    @staticmethod
    def _clear_cell_widgets(sh: Sheet, r: int, c: int) -> None:
        """Remove existing dropdown/checkbox at (r, c) before re-creating."""
        with suppress(AttributeError, KeyError, ValueError, TypeError):
            sh.delete_dropdown(r, c)
        with suppress(AttributeError, KeyError, ValueError, TypeError):
            sh.delete_checkbox(r, c)

    def _node_at_default(self, iid: Any) -> bool:
        """True iff every value in the node's subtree matches its config default.

        Drives blue node labels (blue = untouched).  Cells without a
        resolvable default never block the match; parent nodes aggregate
        their children via the ``parent`` backlink.
        """
        m = self._meta.get(iid, {})
        max_col = int(m.get("max_col") or m.get("len") or 0)
        if m.get("type") == "scalar":
            max_col = 1
        if max_col:
            vals = self.sh.item(iid).get("values") or ()
            return all(
                (dv := self._default_for_cell(iid, m, j)) is _NO_DEFAULT
                or _fmt(vals[j] if j < len(vals) else "") == _fmt(dv)
                for j in range(max_col)
            )
        if kids := [k for k, km in self._meta.items() if km.get("parent") == iid]:
            return all(self._node_at_default(k) for k in kids)
        return True

    def _apply_styles(self) -> None:
        sh = self.sh
        style = ttk.Style()
        bg = _resolve_bg(sh, style.lookup("TFrame", "background") or "#F0F0F0")
        self._fg_default = _resolve_bg(sh, style.lookup("TFrame", "foreground") or "#000000")
        # Index canvas (tree column) background — global fallback only; the 7.6.x
        # draw path consults per-cell highlights.  No index_foreground global:
        # per-node fg is state-driven (blue = at default, see step 1).
        with suppress(AttributeError, TypeError):
            sh.set_options(index_background=bg)
        self._style_header(bg, _BLUE_FG)
        row_of = self._row_map()
        first_data_col = self.DATA_COL_BASE - 1  # tksheet 0-based
        for iid, m in self._meta.items():
            if (r := row_of.get(iid)) is None:
                continue  # hidden/collapsed; style lazily on expand
            # ── 1) node bg + state fg — treeview column = "index" canvas ──
            sh.highlight_cells(
                row=r,
                column=0,
                canvas="index",
                bg=bg,
                fg=_BLUE_FG if self._node_at_default(iid) else self._fg_default,
                redraw=False,
            )
            date_cols = tuple(int(c) for c in (m.get("meta_date_cols") or ()))
            date_set = frozenset(date_cols)
            # ── 2) metadata row bg up to last date cell inclusive ────
            if date_cols and (last_tk := max(date_cols) - self.DATA_COL_BASE) >= first_data_col:
                for col in range(first_data_col, last_tk + 1):
                    sh.highlight_cells(row=r, column=col, bg=bg, redraw=False)
            # ── 3) date alignment + blue fg (independent of max_col) ─
            for dc in date_cols:
                col = dc - self.DATA_COL_BASE
                if col >= first_data_col:
                    sh.align_cells(r, col, align="e", redraw=False)
                    if m.get("date_style") == "blue":
                        sh.highlight_cells(row=r, column=col, fg=_BLUE_FG, highlight_fg=_BLUE_FG, redraw=False)
            max_col = int(m.get("max_col") or 0)
            for meta_col in range(1, max_col + 1):
                col = meta_col - self.DATA_COL_BASE
                if col < 0 or meta_col in date_set:
                    continue  # date cells already styled above
                # ── type-aware widget creation ───────────────────────
                spec = self._cell_spec_for(iid, m, meta_col)
                self._clear_cell_widgets(sh, r, col)
                if spec.kind == "bool":
                    checked = as_bool(sh.get_cell_data(r, col))
                    sh.create_checkbox(r, col, checked=checked, state="normal", redraw=False)
                    sh.set_cell_data(r, col, checked, redraw=False)
                elif spec.kind == "enum" and spec.enum is not None:
                    values = enum_values(spec.enum)
                    if values:
                        current = str(sh.get_cell_data(r, col) or "")
                        if current not in values:
                            current = values[0]
                        sh.create_dropdown(
                            r,
                            col,
                            values=values,
                            set_value=current,
                            state="normal",
                            redraw=False,
                        )
                    sh.align_cells(r, col, align="w", redraw=False)
                elif spec.kind == "text":
                    sh.align_cells(r, col, align="w", redraw=False)
                else:
                    # number / date — right-align (default)
                    sh.align_cells(r, col, align="e", redraw=False)
        sh.redraw()

    # ── default-value foreground coloring ─────────────────────────────

    def _default_for_cell(self, iid: Any, m: dict, col_idx: int) -> Any:
        """Return default value for cell at 0-based *col_idx*, or ``_NO_DEFAULT``.

        Walks the dotted ``meta["path"]`` through :data:`_CFG_DEFAULTS` —
        works for **any** config field (input, out, filter, program), not just coefs.
        Dict results (non-leaf nodes) are rejected — only scalar/list defaults apply.
        """
        path = m.get("path", "")
        if not path:
            return _NO_DEFAULT
        # input row: cell 0 holds input.path — the node itself is a section, not a value
        if m.get("type") == "input" and col_idx == 0:
            path += ".path"
        default = _default_for_path(path)
        if default is _NO_DEFAULT or isinstance(default, dict):
            return _NO_DEFAULT
        if default is None:
            return ""
        if isinstance(default, (list, tuple)):
            return default[col_idx] if col_idx < len(default) else _NO_DEFAULT
        return default if col_idx == 0 else _NO_DEFAULT

    def _apply_default_fg(self) -> None:
        """Gray-out cells whose values match config dataclass factory defaults.

        Uses dotted ``meta["path"]`` via :meth:`_default_for_cell` — covers all
        config sections (input, out, filter, program), not just coefs.

        Called from :meth:`load` after :meth:`_apply_styles`.  Does **not**
        call ``sh.redraw()`` — the caller is responsible.
        """
        sh = self.sh
        row_of = self._row_map()
        for iid, m in self._meta.items():
            if (r := row_of.get(iid)) is None:
                continue
            max_col = int(m.get("max_col") or m.get("len") or 0)
            if m.get("type") == "scalar":
                max_col = 1
            vals = sh.item(iid).get("values") or ()
            for j in range(max_col):
                if (
                    j < len(vals)
                    and (dv := self._default_for_cell(iid, m, j)) is not _NO_DEFAULT
                    and _fmt(vals[j]) == _fmt(dv)
                ):
                    sh.highlight_cells(row=r, column=j, fg=_DEFAULT_FG, redraw=False)

    def _apply_end_edit_style(self, event) -> None:
        """Toggle gray cell fg + blue node labels after a committed edit.

        ``event.row`` is a **display** row → resolve iid via the visible
        walk, then convert to the internal row (:meth:`_row_map`) for every
        cell-API call.  Gray: match → ``_DEFAULT_FG``, else theme fg —
        always explicit, since ``fg=None`` is a no-op under tksheet's
        per-key merge.  Blue: node label + ancestors re-checked — blue
        iff the whole subtree is at default.
        """
        c, r = event.column, event.row
        iid = self._iid_at_row(r)
        if iid is None:
            _lg.debug("end_edit r=%s c=%s → no iid (invalid display row)", r, c)
            return
        row_of = self._row_map()
        if (ri := row_of.get(iid)) is None:
            _lg.debug("end_edit r=%s c=%s iid=%s → no internal row", r, c, iid)
            return
        m = self._meta.get(iid, {})
        dv = self._default_for_cell(iid, m, c)
        if dv is _NO_DEFAULT:
            _lg.debug(
                "end_edit r=%s→%s c=%s iid=%s path=%s → no default, skipping",
                r,
                ri,
                c,
                iid,
                m.get("path"),
            )
            return
        new_val = str(event.value) if event.value is not None else ""
        match = _fmt(new_val) == _fmt(dv)
        _lg.debug(
            "end_edit r=%s→%s c=%s iid=%s path=%s val=%r default=%r match=%s → %s",
            r,
            ri,
            c,
            iid,
            m.get("path"),
            new_val,
            dv,
            match,
            "GRAY" if match else "CLEAR",
        )
        sh = self.sh
        sh.highlight_cells(row=ri, column=c, fg=_DEFAULT_FG if match else self._fg_default, redraw=False)
        # node labels: propagate at-default state up the ancestor chain
        node: Any = iid
        while node is not None:
            if (nr := row_of.get(node)) is not None:
                sh.highlight_cells(
                    row=nr,
                    column=0,
                    canvas="index",
                    fg=_BLUE_FG if self._node_at_default(node) else self._fg_default,
                    redraw=False,
                )
            node = self._meta.get(node, {}).get("parent")
        sh.redraw()