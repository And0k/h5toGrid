"""Config tree in tksheet ≥ 7 treeview."""

from __future__ import annotations

import dataclasses
import logging
import operator
from collections.abc import Callable, Mapping
from contextlib import suppress
from tkinter import TclError
from types import SimpleNamespace
from typing import Any, Final

import numpy as np
import tcm_gui.theme
from tksheet import Sheet

from tcm_gui._cell_spec import any2str, as_date, parse_float
from tcm_gui.cli_cfg import COEF_SHAPES, COEFS_TYPE, NO_DEFAULT, default_for_path
from tcm_gui import _help, _path_field
from ._browse_button import BrowseButtonManager, BrowseOverlay, _pointer_inside
from ._cell_spec import NUMBER_SPEC, CellSpec, as_bool, enum_values, schema_type, spec_for_path

_l = logging.getLogger(__name__)

# Derive field order from dataclass declaration — single source of truth.
# Exclude `dates` / `date` which are handled as tree-level metadata, not row items.
_COEF_FIELDS = [f.name for f in dataclasses.fields(COEFS_TYPE) if f.name not in ("dates", "date")]
_1D_WITH_DATES = {"kVabs"}  # единственное 1D с датами → parent+child
_DATE_COL = 2  # sheet col: 0=tree 1=₁ 2=₂/date 3=₃…
_INTENT_MS: Final[int] = 120  # hover-intent delay for floated PathField (ms)
_RESIZE_ZONE: Final[int] = 8  # px from cell boundary to activate resize cursor
_RESIZE_CURSOR: Final[str] = "sb_h_double_arrow"


class CellBoundaryColumnResize:
    """Column resizing from selected cell boundaries, without a visible header.

    ``resize_cells`` contains ``(row, column)`` tksheet-internal coordinates.
    A cell enables the boundary immediately to its RIGHT, i.e. ``(row, column)``
    enables resizing column ``column`` (the left column of that boundary).

    The resize operation changes the complete column, exactly as normal tksheet
    column resizing does — via :meth:`Sheet.column_width`.
    """

    def __init__(
        self,
        sheet: Sheet,
        resize_cells: set[tuple[int, int]] = frozenset(),
        on_complete: Callable | None = None,
    ):
        self.sheet = sheet
        self.mt = sheet.MT
        self.resize_cells: set[tuple[int, int]] = set(resize_cells)
        self._on_complete = on_complete

        self._col: int | None = None  # column being resized
        self._x0: int | None = None  # press x
        self._w0: int | None = None  # original width

        self._cursor_on = False

        self.mt.bind("<Motion>", self._on_motion, add="+")
        self.mt.bind("<Leave>", self._on_leave, add="+")
        self.mt.bind("<ButtonPress-1>", self._on_press, add="+")
        self.mt.bind("<B1-Motion>", self._on_drag, add="+")
        self.mt.bind("<ButtonRelease-1>", self._on_release, add="+")

    # -- public API -----------------------------------------------------------

    def set_resize_cells(self, cells: set[tuple[int, int]]) -> None:
        self.resize_cells = set(cells)
        self._reset_cursor()

    # -- hit testing ----------------------------------------------------------

    def _boundary_col_at(self, x: int, y: int) -> int | None:
        """Return the resizeable column for a boundary under the cursor, or ``None``.

        Column boundaries are global — any row with data in that column makes
        the boundary resizeable.  This avoids row-space ambiguity between
        ``identify_row`` (display) and ``_row_map`` (internal).
        """
        if not self.resize_cells:
            return None
        positions = self.sheet.get_column_widths(canvas_positions=True)
        cx = self.mt.canvasx(x)
        # Collect unique right-boundaries from resize cells: column c → positions[c+1]
        boundaries = {c: positions[c + 1] for _, c in self.resize_cells if c + 1 < len(positions)}
        for c, bx in boundaries.items():
            if abs(cx - bx) <= _RESIZE_ZONE:
                return c
        return None

    # -- mouse handlers -------------------------------------------------------

    def _on_motion(self, event) -> None:
        if self._col is not None:
            return
        (self._set_cursor if self._boundary_col_at(event.x, event.y) is not None else self._reset_cursor)()

    def _on_leave(self, _event) -> None:
        if self._col is None:
            self._reset_cursor()

    def _on_press(self, event) -> str | None:
        if (col := self._boundary_col_at(event.x, event.y)) is None:
            return None
        self._col, self._x0, self._w0 = col, event.x, self.sheet.column_width(col)
        self._set_cursor()
        # Deselect so the selection box doesn't lag during drag.
        with suppress(AttributeError, TclError):
            self.sheet.deselect()
        return "break"

    def _on_drag(self, event) -> str | None:
        if self._col is None:
            return None
        self.sheet.column_width(self._col, width=self._w0 + event.x - self._x0)
        # Force immediate visual update (redraw=True alone may not repaint fast enough).
        self.sheet.refresh()
        return "break"

    def _on_release(self, _event) -> str | None:
        if self._col is None:
            return None
        self._col = self._x0 = self._w0 = None
        self._set_cursor()
        if self._on_complete:
            self._on_complete()
        return "break"

    # -- cursor ---------------------------------------------------------------

    def _set_cursor(self) -> None:
        if not self._cursor_on:
            self.mt.configure(cursor=_RESIZE_CURSOR)
            self._cursor_on = True

    def _reset_cursor(self) -> None:
        if self._cursor_on:
            self.mt.configure(cursor="")
            self._cursor_on = False


class ConfigSheet:
    """Wraps tksheet.Sheet(treeview=True) for config display / editing.

    Row spaces:
      * internal rows — all rows, hidden included; cell APIs consume these.
      * display rows — visible rows only; edit events report these.

    Hover resolution detects the row space exposed by ``MT.identify_row``
    and enforces a visible-row invariant before status or overlay publication.
    """

    # Meta columns are 1-based; tksheet data column = meta_col - DATA_COL_BASE
    DATA_COL_BASE: int = 1

    # Coef meta types — always numeric; skip Hydra path resolution for these.
    _COEF_TYPES = frozenset({"2d", "1d", "1d_flat", "scalar", "_coef_child"})

    def __init__(self, parent, status_hint: str = "") -> None:
        self.sh = Sheet(
            parent,
            treeview=True,
            show_header=False,
            show_horizontal_grid=False,
            show_vertical_grid=False,
            allow_cell_overflow=True,
            scrollbar_theme_inheritance="default",
        )
        self._status_hint = status_hint
        # Apply dark theme to tksheet when system theme is dark.
        if tcm_gui.theme.THEME == "dark":
            self.sh.change_theme("dark")

        # Cell-boundary column resize — registered BEFORE enable_bindings
        # so resize handlers fire before tksheet's internal click/drag handlers.
        self._col_resize = CellBoundaryColumnResize(self.sh, on_complete=self._after_column_resize)

        self.sh.enable_bindings(["all"])
        self.sh.edit_validation(self._on_edit)
        self.sh.extra_bindings(
            [
                ("begin_edit_cell", self._on_begin_edit_cell),
                ("end_edit_cell", self._on_end_edit_cell),
                # Block selection on non-editable cells — fires inside tksheet's pipeline.
                ("cell_select", self._on_cell_select),
            ]
        )
        # Override rc-menu "Insert column/row" to append at end (idx=None → end).
        self.sh.popup_menu_add_command("Insert column", self._insert_col_at_end)
        self.sh.popup_menu_add_command("Insert row", lambda e=None: self.sh.insert_row())

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
        # Normal (non-default) text color — "clear" side of gray/blue toggles
        self._fg_default: str = tcm_gui.theme.FG_DEFAULT

        # BrowseButtonManager — injected by App after construction
        self._mgr: BrowseButtonManager | None = None

        # ── sheet-hover policy ────────────────────────────────────
        # Status-bar hook — injected by App: ``cs.on_hover_status = lambda msg, md: ...``.
        # Second arg ``md``: True when msg is Markdown (from config_reference.md),
        # False for plain strings (STR labels, Hydra paths).
        self.on_hover_status: Callable[[str, bool], None] | None = None
        self.hover_status: dict[str, str] = {}
        self._status_iid: Any = None
        # Track which canvas owns the current status: "tree" (RI) or "data" (MT).
        # Moving between tree column and data cell on the SAME row must re-publish.
        self._status_source: str | None = None

        # ── floated PathField: hover-edit surface for browse rows ──
        # One reusable instance — reposition + .set() per hover; created
        # lazily on first browse hover.  Focus is opt-in by construction.
        self._field_iid: Any = None  # commit target; survives hide (after_idle commits)
        self._field_row: int = 0  # hit_row for expand-on-edit
        self._field_y: int = 0  # fallback_y for expand-on-edit
        self._field_pending: tuple[Any, int, int] | None = None
        self._field_show_job: str | None = None
        self._field_hide_job: str | None = None
        self._hover_field: _path_field.PathField | None = None
        self._hover_btn: BrowseOverlay | None = None
        self._stretch_job: str | None = None

        # Row-space caches — rebuilt on load and expand/collapse.
        self._loading = False
        self._item_lock = False
        self._vis: tuple[Any, ...] = ()
        self._vis_index: dict[Any, int] = {}
        self._int_row_of: dict[Any, int] = {}
        self._iid_by_disp: dict[int, Any] = {}
        self._iid_by_int: dict[int, Any] = {}
        self._row_space: str = "unknown"  # "display" | "internal" | "unknown"

        mt = self.sh.MT
        mt.bind("<Motion>", self._on_sheet_motion, add="+")
        mt.bind("<Enter>", self._on_sheet_motion, add="+")  # RI→MT transition
        mt.bind("<Leave>", self._on_sheet_leave, add="+")
        for ev in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            mt.bind(ev, self._on_sheet_wheel, add="+")
        mt.bind("<Button-1>", self._redirect_overflow_click, add="+")
        mt.bind("<Double-Button-1>", self._redirect_overflow_double, add="+")
        self.sh.bind("<Configure>", self._stretch_last_col, add="+")
        # Tree column (index canvas / RI): hover shows section-level status.
        # MT <Motion> only fires for data cells; the tree column renders on RI.
        ri = getattr(self.sh, "RI", None)
        if ri is not None:
            ri.bind("<Motion>", self._on_tree_motion, add="+")
            ri.bind("<Leave>", self._on_sheet_leave, add="+")
        # Open-state oracle: hook both public Sheet.item and internal MT.item.
        self._sh_item_orig = self.sh.item
        self._mt_item_orig: Callable | None = None
        self.sh.item = self._item_hook_sh
        with suppress(AttributeError, TypeError):
            self._mt_item_orig = mt.item
            mt.item = self._item_hook_mt

    # ── public API ──────────────────────────────────────────────────

    def load(
        self,
        cfg: dict,
        full: bool = False,
        config_root: type | None = None,
        return_enum: type | None = None,
    ) -> None:
        self._loading = True
        try:
            self._cfg, self._full = cfg, full
            self._config_root = config_root if config_root is not None else schema_type(cfg)
            self._return_enum = return_enum

            self._meta.clear()
            self._hide_hover_field()  # rows are about to die
            self._clear_status()

            self.sh.del_rows(rows=list(range(self.sh.total_rows())))
            self.sh.enable_bindings(["all"])

            self._nv = self._calc_nv(cfg, full)
            self.sh.headers([""] * self._nv)

            (self._build_full if full else self._build_coefs)(cfg)
            self._apply_open()

            self._rebuild_row_caches()
            self._apply_styles()
            self._apply_default_fg()
            self.sh.redraw()

            # Geometry sync after redraw: row_positions may change.
            self._rebuild_row_caches()
        finally:
            self._loading = False

        self._take_snapshot()
        self._stretch_last_col()

    def get_edited_coefs(self) -> dict[str, Any]:
        """Read leaf values → coefs dict for YAML write-back."""
        out: dict[str, Any] = {}

        for iid, m in self._meta.items():
            t = m.get("type")

            if t == "2d":
                nr, nc = m["shape"]
                out[m["key"]] = [
                    [parse_float((self.sh.item(c).get("values") or ())[j]) or 0.0 for j in range(nc)]
                    for c in m["children"][:nr]
                ]

            elif t == "1d":  # kVabs: parent+child
                vals = self.sh.item(m["child"]).get("values") or ()
                out[m["key"]] = [parse_float(vals[j]) or 0.0 for j in range(min(m["len"], len(vals)))]

            elif t == "1d_flat":  # Cg, Ch, …: одна строка
                vals = self.sh.item(iid).get("values") or ()
                out[m["key"]] = [parse_float(vals[j]) or 0.0 for j in range(min(m["len"], len(vals)))]

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
        """Return coefs/dates/path state for YAML write-back."""
        return self.get_edited_coefs(), self.get_edited_dates(), self.get_edited_input_path()

    def _data_snapshot(self) -> tuple[tuple, ...]:
        """Hashable snapshot of ALL editable leaf cell values."""
        parts = []

        for iid, m in self._meta.items():
            max_col = int(m.get("max_col") or m.get("len") or 0)
            if m.get("type") == "scalar":
                max_col = 1
            if max_col == 0:
                continue

            with suppress(ValueError):
                vals = self.sh.item(iid).get("values") or ()
                parts.append((iid, tuple(str(vals[j]) if j < len(vals) else "" for j in range(max_col))))

        return tuple(parts)

    def _take_snapshot(self) -> None:
        """Capture current cell data as the clean baseline."""
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

        path = any2str(inp.get("path", ""))
        inp_iid = self._ins(
            "",
            "input",
            [path] + [""] * (self._nv - 1),
            "",
            meta={
                "key": "input",
                "type": "input",
                "is_string": True,
                "max_col": 1,
                "style": "node",
                "browse": True,
            },
            open_=True,
        )

        if tr := inp.get("time_ranges", []):
            self._ins(
                inp_iid,
                "time_ranges",
                [any2str(x) for x in tr] + [""] * (self._nv - len(tr)),
                "",
                meta={"is_string": True, "max_col": self._nv},
            )

        coefs_path = any2str(inp.get("coefs_path", ""))
        self._ins(
            inp_iid,
            "coefs_path",
            [coefs_path] + [""] * (self._nv - 1),
            "",
            meta={
                "key": "coefs_path",
                "is_string": True,
                "max_col": 1,
                "path": "input.coefs_path",
                "browse": True,
            },
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
        path = any2str(inp.get("path", ""))
        inp_iid = self._ins(
            "",
            "input",
            [path] + [""] * (self._nv - 1),
            "",
            meta={
                "key": "input",
                "type": "input",
                "is_string": True,
                "max_col": 1,
                "style": "node",
                "browse": True,
            },
            open_=True,
        )

        coefs_path = any2str(inp.get("coefs_path", ""))
        self._ins(
            inp_iid,
            "coefs_path",
            [coefs_path] + [""] * (self._nv - 1),
            "",
            meta={
                "key": "coefs_path",
                "is_string": True,
                "max_col": 1,
                "path": "input.coefs_path",
                "browse": True,
            },
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
        shape = COEF_SHAPES.get(name, ())

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
            row = [any2str(x) for x in value[i]] if value else [""] * nc
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

        row = [any2str(x) for x in value] if value else [""] * n

        # Child holds the array values — same config path as parent.
        self._meta[pid]["child"] = self._ins(
            pid,
            name,
            row + [""] * (self._nv - len(row)),
            "",
            meta={"max_col": n, "type": "_coef_child", "path": self._meta[pid]["path"]},
        )

    def _ins_1d_flat(self, par, name, value, n):
        """1D без дат (Cg, Ch, P, …): одна строка, без детей."""
        row = [any2str(x) for x in value] if value else [""] * n
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
            [any2str(value)] + [""] * (self._nv - 1),
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
                        [any2str(x) for x in row] + [""] * (self._nv - len(row)),
                        "",
                        meta={"path": f"{sid_path}[{i}]"},
                    )
            else:
                self._ins(
                    par,
                    key,
                    [any2str(x) for x in value] + [""] * (self._nv - len(value)),
                    "",
                    meta={"is_string": True, "max_col": self._nv},
                )
        else:
            self._ins_leaf(par, key, value)

    def _ins_leaf(self, par: Any, key: str, value: Any) -> None:
        self._ins(
            par,
            key,
            [any2str(value)] + [""] * (self._nv - 1),
            "",
            meta={"is_string": True, "max_col": 1},
        )

    # ── item() open-state oracle ────────────────────────────────────

    def _item_hook_sh(self, iid=None, *args, **kwargs):
        return self._item_call(self._sh_item_orig, iid, args, kwargs)

    def _item_hook_mt(self, iid=None, *args, **kwargs):
        if self._mt_item_orig is None:
            raise AttributeError("MT.item is unavailable")
        return self._item_call(self._mt_item_orig, iid, args, kwargs)

    def _item_call(self, orig: Callable, iid: Any, args: tuple, kwargs: dict) -> Any:
        if self._item_lock:
            return orig(iid, *args, **kwargs)

        op = self._open_intent(iid, kwargs)
        self._item_lock = True
        try:
            out = orig(iid, *args, **kwargs)
        finally:
            self._item_lock = False

        if op is not None and self._commit_open(iid, op):
            self._tree_shape_changed()

        return out

    def _open_intent(self, iid: Any, kwargs: dict) -> bool | None:
        if iid is None or (op := kwargs.get("open_")) is None:
            return None
        if iid not in self._meta:
            return None
        return bool(op)

    def _commit_open(self, iid: Any, op: bool) -> bool:
        m = self._meta.get(iid)
        if m is None or m.get("open") == op:
            return False
        m["open"] = op
        return True

    def _tree_shape_changed(self) -> None:
        if self._loading:
            return

        self._hide_hover_field()
        self._clear_status()

        self._rebuild_row_caches()
        with suppress(AttributeError, TclError):
            self.sh.after_idle(self._rebuild_row_caches)

    # ── helpers ─────────────────────────────────────────────────────

    def _ins(self, parent_iid, text, vals, date="", meta=None, open_=False):
        if meta is None:
            meta = {}

        meta.setdefault("label", text)
        meta.setdefault("open", open_)

        # Backlink for ancestor traversal (blue-label propagation in _on_end_edit)
        meta.setdefault("parent", parent_iid or None)

        # Compute Hydra config path from parent path + node text
        parent_path = self._meta.get(parent_iid, {}).get("path", "")
        meta.setdefault("path", f"{parent_path}.{text}" if parent_path else text)

        if date and len(vals) > 1:
            vals = list(vals)
            vals[1] = date
            meta.setdefault("meta_date_cols", []).append(2)

        # даты по содержимому
        if dc := [c + 1 for c, v in enumerate(vals) if v and as_date(str(v))]:
            meta.setdefault("date_cols", []).extend(dc)

        try:
            iid = self.sh.insert(parent=parent_iid, text=text, values=vals, open_=open_)
        except TypeError:
            iid = self.sh.insert(parent=parent_iid, text=text, values=vals)
            if open_:
                with suppress(TclError):
                    self.sh.item(iid, open_=True)

        self._meta[iid] = meta
        return iid

    def _on_edit(self, event) -> str | None:
        """Validate incoming cell edit.

        ``event.row`` is a display row, ``event.column`` the 0-based data column.
        """
        c = event.column
        val = event.value

        if not val or not val.strip():
            return val

        iid = self._iid_at_row(event.row)
        m = self._meta.get(iid, {})
        if m.get("browse"):
            return val  # overflow-column edits accepted; rerouted in end_edit

        if c == _DATE_COL - self.DATA_COL_BASE and m.get("has_date"):
            result = val if (parsed := as_date(val)) else None
            _l.debug(
                "edit r=%s c=%s iid=%s path=%s val=%r → %s (date)",
                event.row,
                c,
                iid,
                m.get("path"),
                val,
                str(parsed) if result else "REJECT",
            )
            return result

        if c >= m.get("max_col", self._nv):
            _l.debug(
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

        if parse_float(val) is not None:
            return val

        _l.debug(
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
        """Detach any previous browse button; attach for path-type rows.
        Browse rows always edit col 0 — overflow clicks rerouted here.
        Non-data cells (beyond max_col) are rejected — except the date cell."""
        self._hide_hover_field()
        # Unconditional detach — prevents ghost buttons from a previous edit.
        if self._mgr is not None:
            self._mgr.detach()
        iid = self._iid_at_row(event.row)
        m = self._meta.get(iid, {})
        # Determine editable column limit: explicit max_col > len > scalar=1 > _nv.
        raw = m.get("max_col", m.get("len"))
        max_col = int(raw) if raw is not None else (1 if m.get("type") == "scalar" else self._nv)
        # Date cell is always editable (handled by _on_edit validation).
        is_date = event.column == _DATE_COL - self.DATA_COL_BASE and m.get("has_date")
        if not is_date and event.column >= max_col:
            return None
        ri = self._internal_row(iid) if m.get("browse") else None
        if self._mgr is not None and ri is not None:
            self._mgr.attach(ri, 0, iid=iid)
        if ri is not None:
            return self.sh.get_cell_data(ri, 0)  # overflow click edits the path itself
        # Use internal row — display row ≠ data-model row when ancestors collapsed.
        if (int_row := self._internal_row(iid)) is not None:
            return self.sh.get_cell_data(int_row, event.column)
        return None

    def _on_end_edit_cell(self, event) -> None:
        """Detach browse button (any row); reroute overflow edits to col 0;
        reload coefs_path if changed."""
        if self._mgr is not None:
            self._mgr.detach()
        iid = self._iid_at_row(event.row)
        m = self._meta.get(iid, {})
        c = event.column
        val = str(event.value) if event.value is not None else ""
        if m.get("browse") and c > 0 and (ri := self._internal_row(iid)) is not None:
            # tksheet committed to the clicked cell — move the value to col 0,
            # blank the click target. Deferred: order-independent w.r.t. tksheet's
            # own commit; default args pin r/c/v before c → 0 below (closure trap).
            self.sh.after_idle(
                lambda r=ri, oc=c, v=val: (
                    self.sh.set_cell_data(r, 0, v, redraw=False),
                    self.sh.set_cell_data(r, oc, "", redraw=True),
                )
            )
            c = 0
        if m.get("key") == "coefs_path" and val.strip() and self._mgr is not None:
            self.sh.after_idle(lambda p=val: self._mgr.notify_path_changed(p))
        self._apply_end_edit_style(event, col=c)

    # ── overflow-click redirect ───────────────────────────────────

    def _after_column_resize(self) -> None:
        """Called after a column resize drag ends — update last-column stretch and scrollbars."""
        self._stretch_last_col()

    def _insert_col_at_end(self, _event=None) -> None:
        """Append a column and make it editable for string-list rows (e.g. time_ranges)."""
        self.sh.insert_column()
        self._nv += 1
        for m in self._meta.values():
            if m.get("is_string") and m.get("max_col") is not None:
                m["max_col"] = self._nv
        self._apply_styles()

    def _on_cell_select(self, event) -> None:
        """Deselect non-editable cells — fires inside tksheet's selection pipeline.

        ``event`` is an ``EventDataDict``.  The actually selected cell is in
        ``event['selected']`` which may be a ``Selected(row=…, column=…)``
        namedtuple or a plain tuple depending on the triggering action.
        """
        sel = event.get("selected") if isinstance(event, dict) else None
        if sel is None:
            return
        r = getattr(sel, "row", sel[0] if isinstance(sel, (tuple, list)) and len(sel) > 1 else None)
        c = getattr(sel, "column", sel[1] if isinstance(sel, (tuple, list)) and len(sel) > 1 else None)
        if r is None or c is None:
            return
        iid = self._iid_at_row(r)
        if iid is None:
            return
        m = self._meta.get(iid, {})
        raw = m.get("max_col", m.get("len"))
        max_col = int(raw) if raw is not None else (1 if m.get("type") == "scalar" else self._nv)
        is_date = c == _DATE_COL - self.DATA_COL_BASE and m.get("has_date")
        if not is_date and c >= max_col:
            if self._is_date_only_row(m):
                dc = _DATE_COL - self.DATA_COL_BASE
                self.sh.after(1, lambda rr=r, dcc=dc: self.sh.select_cell(rr, dcc))
            else:
                self.sh.after(1, self.sh.deselect)

    def _is_date_only_row(self, m: dict) -> bool:
        """Row whose only editable data cell is the date column (max_col=0, has_date)."""
        return bool(m.get("has_date") and m.get("max_col", self._nv) == 0)

    def _redirect_overflow_click(self, event) -> None:
        """Single-click on a path row's overflow cells → selection box follows
        to col 0.  On date-only rows → selection follows to the date cell.
        Bound add="+" → post-correction after tksheet's handler."""
        if (hit := self._hover_resolve(event)) is None:
            return
        iid, *_ = hit
        c = self._raw_col(event)
        if c is None:
            return
        m = self._meta.get(iid, {})
        dc = _DATE_COL - self.DATA_COL_BASE
        target = None
        if m.get("browse") and c > 0:
            target = 0
        elif self._is_date_only_row(m) and c != dc:
            target = dc
        if target is not None and (ri := self._internal_row(iid)) is not None:
            with suppress(AttributeError, TclError, TypeError, ValueError):
                self.sh.select_cell(ri, target)

    def _redirect_overflow_double(self, event) -> None:
        """Double-click on a path row's overflow cells → editor opens at col 0.
        On date-only rows → editor opens at the date cell.
        Post-correction replay: a synthetic double-click at the target col's x
        (same y) re-enters tksheet's own still-installed binding."""
        if (hit := self._hover_resolve(event)) is None:
            return
        iid, *_ = hit
        c = self._raw_col(event)
        if c is None:
            return
        m = self._meta.get(iid, {})
        dc = _DATE_COL - self.DATA_COL_BASE
        target_col = None
        if m.get("browse") and c > 0:
            target_col = 0
        elif self._is_date_only_row(m) and c != dc:
            target_col = dc
        if target_col is not None and (wx := self._col_widget_x(target_col)) is not None:
            with suppress(TclError):
                self.sh.MT.event_generate("<Double-Button-1>", x=wx, y=event.y, state=event.state)

    def _col_widget_x(self, col: int) -> int | None:
        """Widget-space x just inside *col* — event coords are widget-space,
        ``col_positions`` canvas-space (same convention as ``_hover_place_kw``)."""
        mt = self.sh.MT
        with suppress(AttributeError, TypeError, IndexError, TclError):
            return int(mt.col_positions[col] - mt.canvasx(0)) + 1
        return None

    def _col0_widget_x(self) -> int | None:
        """Shortcut for :meth:`_col_widget_x` at data col 0."""
        return self._col_widget_x(0)

    def _stretch_last_col(self, _event=None) -> None:
        """Stretch the last column to fill the sheet's visible width.

        Debounced via ``after`` to allow tksheet to settle its column positions
        after a Configure event (``after_idle`` fires too early in some builds).
        Skipped during ``load()``.
        """
        if self._loading:
            return
        if (job := getattr(self, "_stretch_job", None)) is not None:
            self.sh.after_cancel(job)
        self._stretch_job = self.sh.after(20, self._do_stretch_last_col)

    def _do_stretch_last_col(self) -> None:
        self._stretch_job = None
        mt = self.sh.MT
        with suppress(AttributeError, TypeError, IndexError, TclError):
            nc = len(mt.col_positions) - 1
            if nc < 1:
                return
            # Always resize last column to fill the visible width.
            cur_w = mt.col_positions[-1] - mt.col_positions[-2]
            new_w = mt.winfo_width() - mt.col_positions[-2]
            if new_w > 0 and abs(cur_w - new_w) > 1:
                self.sh.column_width(nc - 1, width=new_w, redraw=True)

        # Auto-hide scrollbars when content fits within the visible area.
        # 2-px margin prevents oscillation: showing the scrollbar reduces
        # winfo_width/height, which could otherwise immediately re-trigger show.
        with suppress(AttributeError, TclError):
            if len(mt.row_positions) > 1 and (vis_h := mt.winfo_height()) > 10:
                total_h = mt.row_positions[-1]
                (self.sh.hide if total_h <= vis_h + 2 else self.sh.show)("y_scrollbar")
            if len(mt.col_positions) > 1 and (vis_w := mt.winfo_width()) > 10:
                total_w = mt.col_positions[-1]
                (self.sh.hide if total_w <= vis_w + 2 else self.sh.show)("x_scrollbar")

    def _raw_col(self, event) -> int | None:
        # API drift: identify_col may take an event object or bare x.
        for arg in (event, event.x):
            with suppress(AttributeError, TypeError, TclError, ValueError):
                if (v := self.sh.MT.identify_col(arg)) is not None and (c := operator.index(v)) >= 0:
                    return c
        return None

    # ── sheet-hover overlay ──────────────────────────────────────────

    def _clear_status(self) -> None:
        """Reset hover status tracking and clear the status bar."""
        self._status_iid = None
        self._status_source = None
        self._publish_status(None)

    def _on_sheet_leave(self, _event) -> None:
        """``<Leave>`` also fires when the pointer steps onto the field —
        the delayed hide's pointer check decides; status preserved if the
        pointer merely moved onto the field (same row) or a show was pending."""
        had_pending_show = self._field_show_job is not None
        self._schedule_field_hide()
        if not self._pointer_in_field() and not had_pending_show:
            self._clear_status()

    def _on_sheet_wheel(self, _event) -> None:
        """Scroll changes row hit-testing — immediate hide, clear status."""
        self._hide_hover_field()
        self._clear_status()

    def _on_tree_motion(self, event) -> None:
        """Hover over tree column (index canvas) — show section-level status.

        The tree column renders on tksheet's RI canvas, which is separate from
        the MT canvas where ``_on_sheet_motion`` handles data-cell hovers.
        Tree-column hover always shows the section-level help text (e.g.
        "Data source & parameters" for the ``input`` node) — NOT the relocated
        field text (``input.path``) which belongs to the data cells.
        """
        if (hit := self._hover_resolve(event)) is None:
            self._clear_status()
            return

        iid, _row, _y = hit
        # Re-publish when source changes (tree ↔ data on same row).
        if iid == self._status_iid and self._status_source == "tree":
            return

        self._status_iid = iid
        self._status_source = "tree"
        m = self._meta.get(iid, {})
        path = str(m.get("path") or "")
        # Section-level: resolve the path as-is (no `.path` suffix).
        if path and (h := _help.help_for_path(path)) and h.short:
            self.on_hover_status(h.short, True)
        elif self.on_hover_status is not None:
            self.on_hover_status(str(m.get("key") or m.get("label") or path or ""), False)

    def _publish_status(self, iid: Any) -> None:
        """Status text for the hovered element (data cells on MT canvas).

        Override via :attr:`hover_status`, keyed by meta ``key``/``path``/``label``.
        Fallback chain: ``hover_status[ident]`` → ``help_for_path(path).short``
        (from ``config_reference.md``) → ``key`` → ``label`` → ``path``.

        Data cells on parent rows: the ``input`` node row displays ``input.path``
        in its data cell, and the ``coefs`` parent row shows the calibration
        date.  ``_meta[iid]["path"]`` is the section name (``input``, ``input.coefs``)
        rather than the field path.  For data-cell hover, the relocated field
        path is tried FIRST (``input.path``, ``input.coefs.date``), so the status
        describes the editable value, not the section.

        Tree-column hover is handled separately by ``_on_tree_motion`` which
        always uses the section-level path.
        """
        if self.on_hover_status is None:
            return

        if iid is None:
            self.on_hover_status("", False)
            return

        m = self._meta.get(iid, {})
        ident = str(m.get("key") or m.get("path") or m.get("label") or "")

        if (txt := self.hover_status.get(ident)) is not None:
            self.on_hover_status(txt, False)
            return

        # Doc-driven help: ``config_reference.md`` → short tooltip per field.
        # Array indices stripped by ``help_for_path`` (``Ag[0]`` → ``Ag``).
        # Data-cell priority: relocated field first, then section-level.
        # ``input`` row: ``input.path`` (relocated) → "File path, glob…"
        # ``coefs`` parent with date: ``input.coefs.date`` → "Overall calibration date"
        # ``Ag`` child with date: ``input.coefs.dates`` (parent) → "Per-component dates"
        if path := str(m.get("path") or ""):
            candidates: list[str] = []
            if m.get("has_date"):
                # Date field on this row (e.g. ``input.coefs.date``)
                candidates.append(f"{path}.date")
                candidates.append(f"{path}.dates")
                # Parent-level dates for child rows (e.g. Ag → input.coefs.dates)
                if (par := m.get("parent")) and (pp := self._meta.get(par, {}).get("path")):
                    candidates.append(f"{pp}.dates")
                    candidates.append(f"{pp}.date")
            # Relocated data field on input parent row (and similar)
            candidates.append(f"{path}.path")
            # Section / field-level (the path as-is)
            candidates.append(path)
            for candidate in candidates:
                if (h := _help.help_for_path(candidate)) and h.short:
                    self.on_hover_status(h.short, True)
                    return

        self.on_hover_status(str(m.get("key") or m.get("label") or m.get("path") or ""), False)

    def _hover_write(self, text: str) -> None:
        """Write path to column 0 of the hovered row + restyle."""
        iid = self._field_iid
        if iid is None:
            return

        if (r := self._internal_row(iid)) is not None:
            with suppress(TclError):
                self.sh.set_cell_data(r, 0, text)

        self._apply_edit_value(iid, 0, text)

        m = self._meta.get(iid, {})
        if m.get("key") == "coefs_path" and self._mgr is not None:
            self.sh.after_idle(lambda: self._mgr.notify_path_changed(text))

    def _hover_read(self) -> str:
        """Read column 0 of the hovered row (for dialog initialdir)."""
        iid = self._field_iid
        if iid is None:
            return ""

        if (r := self._internal_row(iid)) is not None:
            with suppress(TclError, IndexError):
                return self.sh.get_cell_data(r, 0) or ""

        return ""

    # ── floated PathField — hover-edit surface for browse rows ─────

    # No-op overlay — replaces PathField's internal BrowseOverlay so its
    # SheetHoverBinder never creates a second button.
    _NULL_OV = SimpleNamespace(
        visible=False,
        pending=False,
        show=lambda **_kw: None,
        hide=lambda: None,
        schedule_show=lambda _kw, **_a: None,
        schedule_hide=lambda **_a: None,
        cancel_show=lambda: None,
        cancel_hide=lambda: None,
    )

    def _ensure_hover_field(self) -> _path_field.PathField:
        """The single PathField instance for the text surface + a separate
        ``BrowseOverlay`` button at the sheet's right edge.  Both are
        created lazily on first browse hover.  Focus is opt-in."""
        if self._hover_field is None:
            f = _path_field.PathField(
                self.sh,
                align="e",  # floated field: always right-aligned
                on_commit=self._hover_write,
                on_begin_edit=self._on_field_edit_start,
                on_end_edit=self._on_field_edit_end,
                dir_title="Browse data path",
                files_title="Browse data files",
            )
            # Neuter PathField's own browse overlay — we use a separate one.
            # Replace overlay + binder so SheetHoverBinder never creates a
            # second button.
            f._ov.hide()
            f._ov = self._NULL_OV
            f._binder._ov = self._NULL_OV
            f._binder._last = None
            for ev in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                f.sh.MT.bind(ev, lambda _e: self._hide_hover_field(), add="+")
            self._hover_field = f
            # Status callback for browse button Shift hint — wraps
            # on_hover_status(msg, md) into the on_status(text) signature.
            _on_status = (
                (lambda text: self.on_hover_status(text, True)) if self.on_hover_status is not None else None
            )
            _hint = self._status_hint
            self._hover_btn = BrowseOverlay(
                self.sh,
                self._hover_write,
                self._hover_read,
                dir_title="Browse data path",
                files_title="Browse data files",
                on_status=_on_status,
                status_hint=_hint,
            )
        return self._hover_field

    def _show_hover_field(self, iid: Any, hit_row: int, fallback_y: int) -> None:
        f = self._ensure_hover_field()
        f.cancel_edit()  # stale editor from the previous row → Esc
        self._field_iid = iid
        self._field_row = hit_row  # stored for expand-on-edit
        self._field_y = fallback_y
        val = self._hover_read()
        f.set(val)
        f.place(**self._field_place_kw(hit_row, fallback_y, val))
        f.lift()
        if self._hover_btn is not None:
            self._hover_btn.show(**self._btn_place_kw(hit_row, fallback_y))
        # Publish status so the help text is visible even when the pointer
        # went straight to the overlay without lingering on the tksheet cell.
        self._status_iid = iid
        self._status_source = "data"
        self._publish_status(iid)

    def _field_place_kw(self, hit_row: int, fallback_y: int, val: str) -> dict[str, Any]:
        """Text surface: from col 0, top-aligned, row-matching height.

        Width ends where the browse button starts.  Height is read from
        ``MT.row_positions`` so the field matches the sheet's actual rows
        regardless of index-label chrome.
        """
        mt = self.sh.MT
        with suppress(AttributeError, TypeError, IndexError, TclError):
            y1, y2 = mt.row_positions[hit_row], mt.row_positions[hit_row + 1]
            x0 = self._col0_widget_x() or 0
            btn_w = self._hover_btn_w()
            return {
                "in_": mt,
                "x": x0,
                "anchor": "nw",
                "y": y1 - mt.canvasy(0),
                "width": max(mt.winfo_width() - x0 - btn_w, 50),
                "height": y2 - y1,
            }
        return {"in_": mt, "x": 0, "y": fallback_y, "anchor": "nw", "width": 320}

    def _field_full_width_kw(self) -> dict[str, Any]:
        """Full row width (no button subtraction) — used when editing starts."""
        mt = self.sh.MT
        with suppress(AttributeError, TypeError, IndexError, TclError):
            y1, y2 = mt.row_positions[self._field_row], mt.row_positions[self._field_row + 1]
            x0 = self._col0_widget_x() or 0
            return {
                "in_": mt,
                "x": x0,
                "anchor": "nw",
                "y": y1 - mt.canvasy(0),
                "width": mt.winfo_width() - x0,
                "height": y2 - y1,
            }
        return {"in_": mt, "x": 0, "y": self._field_y, "anchor": "nw", "width": 320}

    def _btn_place_kw(self, hit_row: int, fallback_y: int) -> dict[str, Any]:
        """Browse button: right edge of the visible row, top-aligned with text field."""
        mt = self.sh.MT
        with suppress(AttributeError, TypeError, IndexError, TclError):
            y1 = mt.row_positions[hit_row]
            return {
                "in_": mt,
                "x": mt.winfo_width(),
                "y": y1 - mt.canvasy(0),
                "anchor": "ne",
            }
        return {"in_": mt, "x": mt.winfo_width(), "y": fallback_y, "anchor": "ne"}

    def _hover_btn_w(self) -> int:
        """Pixel width of the hover browse button.

        Uses the actual rendered width (``winfo_width``) when the button
        exists and has been placed; falls back to ``winfo_reqwidth`` of a
        temporary button otherwise.
        """
        if self._hover_btn is not None and self._hover_btn._button is not None:
            with suppress(TclError):
                btn = self._hover_btn._button
                btn.update_idletasks()
                w = btn.winfo_width()
                if w > 1:
                    return w
        if not hasattr(self, "_btn_w_cache"):
            from ._browse_button import browse_button_width

            self._btn_w_cache = browse_button_width(self.sh)
        return self._btn_w_cache

    def _on_field_edit_start(self) -> None:
        """User clicked the overlay field to edit — hide button, expand field.

        Cancels any pending hide (armed by ``<Leave>`` when the pointer
        stepped onto the Entry) and forces geometry so ``PathField``
        reads the correct ``winfo_width()`` for its column constraint.
        """
        self._cancel_field_hide_job()
        if self._hover_btn is not None:
            self._hover_btn.hide()
        if (f := self._hover_field) is not None and f.winfo_ismapped():
            f.place(**self._field_full_width_kw())
            f.update_idletasks()

    def _on_field_edit_end(self) -> None:
        """Entry edit finished — restore hover width, re-show button."""
        self._restore_hover_placement()

    def _restore_hover_placement(self) -> None:
        if (f := self._hover_field) is not None and f.winfo_ismapped() and not f._editing:
            val = self._hover_read()
            f.place(**self._field_place_kw(self._field_row, self._field_y, val))
            f.update_idletasks()
            if self._hover_btn is not None:
                self._hover_btn.show(**self._btn_place_kw(self._field_row, self._field_y))

    def _schedule_field_show(self, iid: Any, hit_row: int, fallback_y: int) -> None:
        self._cancel_field_hide_job()
        if self._field_show_job is not None:
            self.sh.after_cancel(self._field_show_job)
        self._field_pending = (iid, hit_row, fallback_y)
        self._field_show_job = self.sh.after(_INTENT_MS, self._do_field_show)

    def _do_field_show(self) -> None:
        self._field_show_job = None
        if self._field_pending is not None:
            self._show_hover_field(*self._field_pending)

    def _schedule_field_hide(self) -> None:
        """Delayed hide — vetoed when the pointer has moved onto the field
        itself (MT fires ``<Leave>`` at exactly that crossing)."""
        if self._field_show_job is not None:
            self.sh.after_cancel(self._field_show_job)
            self._field_show_job = self._field_pending = None
        field_mapped = self._hover_field is not None and self._hover_field.winfo_ismapped()
        btn_visible = self._hover_btn is not None and self._hover_btn.visible
        if self._field_hide_job is None and (field_mapped or btn_visible):
            self._field_hide_job = self.sh.after(_INTENT_MS, self._do_field_hide)

    def _do_field_hide(self) -> None:
        self._field_hide_job = None
        if (f := self._hover_field) is not None and f._editing:
            return  # don't hide while editing — Entry fills the PathField
        if not self._pointer_in_field():
            self._hide_hover_field()

    def _hide_hover_field(self) -> None:
        """Immediate teardown: cancel jobs, unmap field + button.

        Deliberately keeps ``_field_iid`` — PathField commits via
        ``after_idle``, so a commit already queued must still land on its row.
        """
        for attr in ("_field_show_job", "_field_hide_job"):
            if (job := getattr(self, attr)) is not None:
                self.sh.after_cancel(job)
                setattr(self, attr, None)
        self._field_pending = None
        if (f := self._hover_field) is not None and f.winfo_ismapped():
            f.place_forget()
        if self._hover_btn is not None:
            self._hover_btn.hide()

    def _cancel_field_hide_job(self) -> None:
        job, self._field_hide_job = self._field_hide_job, None
        if job is not None:
            self.sh.after_cancel(job)

    def _pointer_in_field(self) -> bool:
        """True when the pointer is inside the floated PathField or its browse button."""
        f = self._hover_field
        if f is not None and f.winfo_ismapped() and _pointer_inside(f):
            return True
        btn = self._hover_btn
        return btn is not None and btn.visible and btn._button is not None and _pointer_inside(btn._button)

    def _on_sheet_motion(self, event) -> None:
        """Hover: status text for any visible row; floated field on browse rows."""
        if (hit := self._hover_resolve(event)) is None:
            self._schedule_field_hide()
            if self._status_iid is not None:
                self._clear_status()
            return

        iid, row, y = hit

        # Re-publish when row OR source (tree ↔ data) changes.
        if iid != self._status_iid or self._status_source != "data":
            self._status_iid = iid
            self._status_source = "data"
            self._publish_status(iid)

        if not self._meta.get(iid, {}).get("browse"):
            self._schedule_field_hide()
            return

        f = self._hover_field
        if iid == self._field_iid and (
            (f is not None and f.winfo_ismapped()) or self._field_show_job is not None
        ):
            self._cancel_field_hide_job()  # motion over target vetoes pending hide
            # During editing the field is at full width — don't shrink it back.
            if f is not None and f.winfo_ismapped() and not f._editing:
                val = self._hover_read()
                f.place(**self._field_place_kw(row, y, val))
                if self._hover_btn is not None:
                    self._hover_btn.show(**self._btn_place_kw(row, y))
            return

        self._schedule_field_show(iid, row, y)

    # ── row-space resolution ────────────────────────────────────────

    def _walk(self, parent: Any = "", *, visible: bool = False):
        with suppress(AttributeError, TclError, TypeError, ValueError):
            for cid in self.sh.get_children(parent):
                yield cid
                if not visible or self._is_open(cid):
                    yield from self._walk(cid, visible=visible)

    def _is_open(self, iid: Any) -> bool:
        # Prefer tksheet private truth when available; meta is fallback.
        with suppress(AttributeError, KeyError, TypeError, TclError):
            return bool(self.sh.MT.treeview[iid]["open"])
        return bool(self._meta.get(iid, {}).get("open"))

    def _rebuild_row_caches(self) -> None:
        self._int_row_of = {iid: r for r, iid in enumerate(self._walk()) if iid in self._meta}
        self._vis = tuple(iid for iid in self._walk(visible=True) if iid in self._meta)
        self._vis_index = {iid: i for i, iid in enumerate(self._vis)}
        self._iid_by_disp = dict(enumerate(self._vis))
        self._iid_by_int = {r: iid for iid, r in self._int_row_of.items()}
        self._row_space = self._detect_row_space()

    def _detect_row_space(self) -> str:
        with suppress(AttributeError, TypeError, TclError):
            n = max(0, len(self.sh.MT.row_positions) - 1)
            d, i = len(self._vis), len(self._int_row_of)

            if i != d:
                if n == i:
                    return "internal"
                if n == d:
                    return "display"
                if n > d and n >= i:
                    return "internal"
                if n < i and n >= d:
                    return "display"

            return "display" if n == d else "unknown"

        return "unknown"

    def _raw_hit(self, event) -> int | None:
        # API drift: 7.x identify_row may take an event object, older takes y.
        for arg in (event, event.y):
            with suppress(AttributeError, TypeError, TclError, ValueError):
                if (v := self.sh.MT.identify_row(arg)) is not None and (r := operator.index(v)) >= 0:
                    return r
        return None

    def _hover_resolve(self, event) -> tuple[Any, int, int] | None:
        """Return ``(iid, hit_row, fallback_y)`` only for visible rows."""
        if (r := self._raw_hit(event)) is None:
            return None

        if self._row_space == "internal" or (self._row_space == "unknown" and r >= len(self._vis)):
            iid = self._iid_by_int.get(r)
        else:
            iid = self._iid_by_disp.get(r)
            if iid not in self._vis_index and (alt := self._iid_by_int.get(r)) in self._vis_index:
                iid = alt

        if iid is None or iid not in self._vis_index:
            return None

        return iid, r, event.y

    def _row_map(self) -> dict[Any, int]:
        """Map treeview iid → internal tksheet row — the cell-API row space."""
        if self._int_row_of:
            return self._int_row_of

        return {iid: r for r, iid in enumerate(self._walk()) if iid in self._meta}

    def _iid_at_row(self, r: int) -> Any | None:
        """Map display row (``event.row``) → tree iid."""
        return self._vis[r] if 0 <= r < len(self._vis) else None

    def _internal_row(self, iid: Any) -> int | None:
        if iid is None:
            return None
        if (r := self._int_row_of.get(iid)) is None:
            r = self._row_map().get(iid)
        return r

    # ── styling ─────────────────────────────────────────────────────

    def _apply_open(self) -> None:
        """Re-apply desired open states stored in meta during construction."""
        for iid, m in list(self._meta.items()):
            if m.get("open"):
                with suppress(AttributeError, TclError, TypeError):
                    self.sh.item(iid, open_=True)

    def _cell_spec_for(self, iid: str, m: Mapping[str, Any], meta_col: int) -> CellSpec:
        """Resolve ``CellSpec`` for a cell at *meta_col* in row *iid*."""
        if m.get("type") in self._COEF_TYPES:
            return NUMBER_SPEC

        # Strip array indices (e.g. "input.coefs.Ag[0]" → "input.coefs.Ag")
        path = str(m.get("path", iid))
        clean = path.split("[")[0] if "[" in path else path
        return spec_for_path(self._config_root, clean, self._return_enum)

    @staticmethod
    def _clear_cell_widgets(sh: Sheet, r: int, c: int) -> None:
        """Remove existing dropdown/checkbox at (r, c) before re-creating."""
        with suppress(AttributeError, KeyError, ValueError, TypeError):
            sh.delete_dropdown(r, c)
        with suppress(AttributeError, KeyError, ValueError, TypeError):
            sh.delete_checkbox(r, c)

    def _node_at_default(self, iid: Any) -> bool:
        """True iff every value in the node's subtree matches its config default."""
        m = self._meta.get(iid, {})

        max_col = int(m.get("max_col") or m.get("len") or 0)
        if m.get("type") == "scalar":
            max_col = 1

        if max_col:
            vals = self.sh.item(iid).get("values") or ()
            return all(
                (dv := self._default_for_cell(iid, m, j)) is NO_DEFAULT
                or any2str(vals[j] if j < len(vals) else "") == any2str(dv)
                for j in range(max_col)
            )

        if kids := [k for k, km in self._meta.items() if km.get("parent") == iid]:
            return all(self._node_at_default(k) for k in kids)

        return True

    def _apply_styles(self) -> None:
        sh = self.sh

        bg = tcm_gui.theme.resolved_frame_bg(sh)
        self._fg_default = tcm_gui.theme.FG_DEFAULT

        with suppress(AttributeError, TypeError):
            sh.set_options(index_background=bg)

        row_of = self._row_map()
        first_data_col = self.DATA_COL_BASE - 1  # tksheet 0-based
        total_cols = sh.total_columns()
        resize_cells: set[tuple[int, int]] = set()

        for iid, m in self._meta.items():
            if (r := row_of.get(iid)) is None:
                continue

            is_input = m.get("type") == "input"
            is_browse = is_input or m.get("browse")

            # ── 1) node label — treeview column = "index" canvas ──
            # Input row: button-face bg + normal black fg; other rows: blue/black fg
            sh.highlight_cells(
                row=r,
                column=0,
                canvas="index",
                bg=bg,
                fg=(
                    tcm_gui.theme.FG_DEFAULT
                    if is_input
                    else (tcm_gui.theme.BLUE_FG if self._node_at_default(iid) else self._fg_default)
                ),
                redraw=False,
            )

            # ── 1b) browse/input rows: paint ALL columns uniform ──
            # Prevents colour mismatch between col-0 and overflow columns.
            if is_browse:
                for col in range(first_data_col, total_cols):
                    sh.highlight_cells(row=r, column=col, bg=bg, redraw=False)

            date_cols = tuple(int(c) for c in (m.get("meta_date_cols") or ()))
            date_set = frozenset(date_cols)

            # ── 2) metadata row bg up to last date cell inclusive ────
            if date_cols and (last_tk := max(date_cols) - self.DATA_COL_BASE) >= first_data_col:
                for col in range(first_data_col, last_tk + 1):
                    sh.highlight_cells(row=r, column=col, bg=bg, redraw=False)

            # ── 3) date alignment + blue fg ─────────────────────────
            for dc in date_cols:
                col = dc - self.DATA_COL_BASE
                if col >= first_data_col:
                    sh.align_cells(r, col, align="e", redraw=False)
                    if m.get("date_style") == "blue":
                        sh.highlight_cells(
                            row=r,
                            column=col,
                            fg=tcm_gui.theme.BLUE_FG,
                            highlight_fg=tcm_gui.theme.BLUE_FG,
                            redraw=False,
                        )

            max_col = int(m.get("max_col") or 0)

            # ── 4) build resize cells: non-browse data cells only ──
            # Browse rows use overflow — no column boundaries needed.
            if max_col > 0 and not is_browse:
                for col in range(max_col):
                    resize_cells.add((r, col))

            for meta_col in range(1, max_col + 1):
                col = meta_col - self.DATA_COL_BASE
                if col < 0 or meta_col in date_set:
                    continue

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
                    # Left-align to preserve allow_cell_overflow (extends RIGHT).
                    # The floated overlay PathField shows the right-aligned end.
                    sh.align_cells(r, col, align="w", redraw=False)

                else:
                    # number / date — right-align
                    sh.align_cells(r, col, align="e", redraw=False)

        self._col_resize.set_resize_cells(resize_cells)
        sh.redraw()

    # ── default-value foreground coloring ─────────────────────────────

    def _default_for_cell(self, iid: Any, m: dict, col_idx: int) -> Any:
        """Return default value for cell at 0-based *col_idx*, or ``NO_DEFAULT``."""
        path = m.get("path", "")
        if not path:
            return NO_DEFAULT

        # input row: cell 0 holds input.path — the node itself is a section, not a value
        if m.get("type") == "input" and col_idx == 0:
            path += ".path"

        default = default_for_path(path)

        if default is NO_DEFAULT or isinstance(default, dict):
            return NO_DEFAULT
        if default is None:
            return ""
        if isinstance(default, (list, tuple)):
            return default[col_idx] if col_idx < len(default) else NO_DEFAULT

        return default if col_idx == 0 else NO_DEFAULT

    def _apply_default_fg(self) -> None:
        """Gray-out cells whose values match config dataclass factory defaults."""
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
                    and (dv := self._default_for_cell(iid, m, j)) is not NO_DEFAULT
                    and any2str(vals[j]) == any2str(dv)
                ):
                    sh.highlight_cells(
                        row=r, column=j, fg=tcm_gui.theme.DEFAULT_FG, redraw=False, overwrite=False
                    )

    def _apply_edit_value(self, iid: Any, col: int, value: str) -> None:
        """Restyle cell + ancestors after a committed value."""
        row_of = self._row_map()

        if (ri := row_of.get(iid)) is None:
            return

        m = self._meta.get(iid, {})
        dv = self._default_for_cell(iid, m, col)

        if dv is NO_DEFAULT:
            return

        match = any2str(value) == any2str(dv)

        self.sh.highlight_cells(
            row=ri,
            column=col,
            fg=tcm_gui.theme.DEFAULT_FG if match else self._fg_default,
            redraw=False,
            overwrite=False,
        )

        # node labels: propagate at-default state up the ancestor chain
        # Input row: always normal fg — never blue/gray toggle
        node: Any = iid
        while node is not None:
            if (nr := row_of.get(node)) is not None:
                nm = self._meta.get(node, {})
                if nm.get("type") == "input":
                    node_fg = tcm_gui.theme.FG_DEFAULT
                else:
                    node_fg = tcm_gui.theme.BLUE_FG if self._node_at_default(node) else self._fg_default
                self.sh.highlight_cells(
                    row=nr,
                    column=0,
                    canvas="index",
                    fg=node_fg,
                    redraw=False,
                    overwrite=False,
                )
            node = self._meta.get(node, {}).get("parent")

        self.sh.redraw()

    def _apply_end_edit_style(self, event, col: int | None = None) -> None:
        """Toggle gray cell fg + blue node labels after a committed edit."""
        c = col if col is not None else event.column
        r = event.row
        iid = self._iid_at_row(r)

        if iid is None:
            _l.debug("end_edit r=%s c=%s → no iid (invalid display row)", r, c)
            return

        new_val = str(event.value) if event.value is not None else ""

        _l.debug(
            "end_edit r=%s c=%s iid=%s path=%s val=%r",
            r,
            c,
            iid,
            self._meta.get(iid, {}).get("path"),
            new_val,
        )

        self._apply_edit_value(iid, c, new_val)
