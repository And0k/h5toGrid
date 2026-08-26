"""Config tree in tksheet ≥ 7 treeview — composition root.

:class:`ConfigSheet` wires the sheet widget, builds the tree from a config
dict and owns row-space resolution + the edit lifecycle.  Cross-cutting
behavior lives in focused mixins (see their module docstrings):

* :mod:`tcm_gui._sheet_tint` — defaults, gray/blue tint, ghost placeholders,
  live ``time_ranges`` sync relation;
* :mod:`tcm_gui._sheet_styles` — alignment/widgets, node fg, path validation;
* :mod:`tcm_gui._sheet_status` — hover status bar, floated PathField overlay.

Row spaces:
  * internal rows — all rows, hidden included; cell APIs consume these.
  * display rows — visible rows only; edit events report these.

Hover resolution detects the row space exposed by ``MT.identify_row``
and enforces a visible-row invariant before status or overlay publication.
"""

from __future__ import annotations

import dataclasses
import logging
import operator
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from tkinter import TclError
from typing import Any, Final

import numpy as np
from tksheet import Sheet

import tcm_gui.theme
from tcm import _meta_pairs, _constants
from tcm_gui import _path_field
from tcm_gui._cell_spec import any2str, as_date, parse_float
from tcm_gui.cli_cfg import COEF_SHAPES, COEFS_TYPE

from ._browse_button import BrowseButtonManager
from ._i18n import STRINGS as _S
from ._placeholder import CellPlaceholder
from ._sheet_status import SheetHoverMixin
from ._sheet_styles import SheetStylesMixin, _path_exists
from ._sheet_tint import _DATE_COL, _DATE_PH_COL, SheetTintMixin

_l = logging.getLogger(__name__)

# Derive field order from dataclass declaration — single source of truth.
# Exclude `dates` / `date` which are handled as tree-level metadata, not row items.
_COEF_FIELDS = [f.name for f in dataclasses.fields(COEFS_TYPE) if f.name not in ("dates", "date")]
_1D_WITH_DATES = {"kVabs"}  # единственное 1D с датами → parent+child
_DATE_COL = _DATE_COL  # meta col: 1=₁ 2=₂/date 3=₃… (tksheet col = meta_col − DATA_COL_BASE)
_RESIZE_ZONE: Final[int] = 8  # px from cell boundary to activate resize cursor
_RESIZE_CURSOR: Final[str] = "sb_h_double_arrow"


def _safe_select(sheet: Any, row: int, col: int) -> None:
    """select_cell that swallows IndexError — row may be stale after tree changes."""
    with suppress(AttributeError, TclError, TypeError, ValueError, IndexError):
        sheet.select_cell(row, col)


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


class ConfigSheet(SheetTintMixin, SheetStylesMixin, SheetHoverMixin):
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
        self.sh.popup_menu_add_command(_S["sheet.insert_col"], self._insert_col_at_end)
        self.sh.popup_menu_add_command(_S["sheet.insert_row"], lambda e=None: self.sh.insert_row())

        self._meta: dict[Any, dict] = {}
        self._ph = CellPlaceholder()  # dim ISO-format hint for empty date cells
        self._nv = 6
        self._full = False
        self._cfg: dict = {}
        self._readonly = False  # blocks editing until scan finds configs (non-full mode)

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
        # Fired when the user starts editing any cell (double-click / keypress).
        # App wires this to hide progress widgets during editing.
        self.on_edit_begin: Callable[[], None] | None = None
        # Fired after every validation pass — App wires this to re-evaluate
        # the Run button enabled state across all config tabs.
        self.on_validity_change: Callable[[], None] | None = None
        # (hover_status dict removed — time_ranges detail is live via _time_ranges_detail)
        self._empty_area_hint: str = ""  # shown on hover below last row
        self._status_iid: Any = None
        # Track which canvas owns the current status: "tree" (RI) or "data" (MT).
        # Moving between tree column and data cell on the SAME row must re-publish.
        self._status_source: str | None = None
        # Detailed body text for dwell tooltip (App reads via _on_cell_status).
        # Set by _publish_status / _on_tree_motion; cleared by _clear_status.
        self._hover_detail: str = ""

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

        # Shift key on toplevel → re-publish status for coefs_path hover so
        # the status bar swaps dir↔file content on Shift toggle without
        # requiring pointer motion.  Bound once with add="+".
        root = self.sh.winfo_toplevel()
        root.bind("<KeyPress-Shift_L>", self._on_shift_toggle, add="+")
        root.bind("<KeyRelease-Shift_L>", self._on_shift_toggle, add="+")
        root.bind("<KeyRelease-Shift_R>", self._on_shift_toggle, add="+")
        # F1 — open the doc browser at the hovered row's config_reference heading.
        root.bind("<F1>", self._on_f1_help, add="+")

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
        # Hook hide_text_editor_and_dropdown — every editor-CLOSE path (Escape,
        # click-away, Enter, Tab) calls it; open_text_editor calls plain
        # hide_text_editor instead, so the hook never fires mid-open.
        # Covers the case tksheet never fires end_edit_cell: committing "" over
        # an already-"" cell is rejected by input_valid_for_cell (cell_equal_to).
        self._mt_close_editor_orig = mt.hide_text_editor_and_dropdown
        mt.hide_text_editor_and_dropdown = self._on_editor_closed

    def load(
        self,
        cfg: dict,
        full: bool = False,
        config_root: type | None = None,
        return_enum: type | None = None,
        *,
        metadata: list[Any] | None = None,
        sync_status: dict | None = None,
        metadata_path: str | None = None,
    ) -> None:
        self._loading = True
        try:
            self._cfg, self._full = cfg, full
            self._config_root = config_root if config_root is not None else schema_type(cfg)
            self._return_enum = return_enum
            self._metadata = metadata  # raw 11-array for metadata node (None → fallback)
            self._sync_status = sync_status  # {status, meta_tr, existing_tr} for time_ranges
            page_stem = cfg.get("_page_stem") or cfg.get("input", {}).get("_page_stem") or ""
            if page_stem:
                self._page_stem = str(page_stem)
            if metadata_path is not None:
                self._metadata_path = metadata_path

            self._meta.clear()
            self._hide_hover_field()  # rows are about to die
            self._clear_status()

            self.sh.del_rows(rows=list(range(self.sh.total_rows())))
            self.sh.enable_bindings(["all"])

            self._nv = self._calc_nv(cfg, full)
            self.sh.headers([""] * self._nv)

            (self._build_full if full else self._build_coefs)(cfg)
            self._build_metadata()
            self._apply_open()

            self._rebuild_row_caches()
            self._apply_styles()
            self._apply_placeholders()
            self._apply_default_fg()
            self._apply_time_ranges_tint()
            self._apply_validations()
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
        """Return non-empty date values from date-metadata rows.

        Placeholder cells (dim ``YYYY-MM-DDTHH:MM:SS``) are treated as empty
        via :meth:`CellPlaceholder.get` — the hint never leaks into YAML.
        Metadata rows (``is_metadata``) are not coef dates — skip them (they
        have no ``key`` in the coefs sense and would KeyError).
        """
        row_of = self._row_map()
        out: dict[str, str] = {}
        for iid, m in self._meta.items():
            if not m.get("has_date") or m.get("is_metadata"):
                continue
            if (r := row_of.get(iid)) is None:
                continue
            d = self._ph.get(self.sh, r, _DATE_PH_COL)
            if d and m.get("key"):
                out[m["key"]] = d
        return out

    def get_edited_input_path(self) -> str:
        for iid, m in self._meta.items():
            if m.get("type") == "input":
                return self._cell_str(iid, 0)
        return ""

    def is_path_valid(self) -> bool:
        """True iff ``input.path`` is non-empty and resolves to an existing file.

        Only the ``input.path`` row gates the Run button — ``input.coefs_path``
        is optional (coefficients may be entered manually), even though
        :meth:`_apply_validations` still red-flags it when the path is missing.
        """
        for iid, m in self._meta.items():
            if m.get("type") != "input":
                continue
            if not (s := self._cell_str(iid, 0)) or s.startswith("<"):
                return False
            return _path_exists(s)
        return False

    def get_metadata_path(self) -> str:
        for iid, m in self._meta.items():
            if m.get("is_metadata_root"):
                if not (s := self._cell_str(iid, 0)):
                    return str(getattr(self, "_metadata_path", "") or "")
                return s
        return str(getattr(self, "_metadata_path", "") or "")

    def get_edited_metadata(self) -> list[Any]:
        """Read metadata rows → 11-array for info_devices.yaml write-back.

        Ghost placeholders (``_ph``) read as ``""`` → ``None`` → ``~`` (required)
        or trimmed tail — identical to the coefs date extraction.  Guards
        ``_ph`` for test harnesses that construct ``ConfigSheet`` via
        ``__new__`` without ``__init__`` (no ``_ph`` attribute yet).
        """
        paired: dict[str, list[str]] = {}
        row_of = self._row_map()
        ph = getattr(self, "_ph", None)
        for iid, m in self._meta.items():
            if not m.get("is_metadata"):
                continue
            r = row_of.get(iid)
            n = int(m.get("max_col", 1))
            vals: list[str] = []
            for j in range(n):
                if r is not None and ph is not None and hasattr(ph, "has") and ph.has(r, j):
                    vals.append("")
                else:
                    raw = self.sh.item(iid).get("values") or ()
                    vals.append(str(raw[j]) if j < len(raw) else "")
            paired[m["label"]] = vals
        if not paired:
            return []
        base = list(self._metadata) if getattr(self, "_metadata", None) else None
        return _meta_pairs.to_storage(paired, base=base)

    def is_metadata_dirty(self) -> bool:
        """True when metadata rows differ from load snapshot (separate file)."""
        snap = getattr(self, "_snap_meta", None)
        if snap is None:
            return False
        cur = self.get_edited_metadata()
        cur_t = tuple("?" if v is None else str(v) for v in cur) if cur else ()
        return cur_t != snap

    def _take_metadata_snapshot(self) -> None:
        md = self.get_edited_metadata()
        self._snap_meta: tuple = tuple("?" if v is None else str(v) for v in md) if md else ()

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

    def mark_metadata_clean(self) -> None:
        self._take_metadata_snapshot()

    def set_readonly(self, readonly: bool) -> None:
        """Block/unblock cell editing.

        Used in non-full mode to disable editing until scan finds configs.
        When enabling readonly, also hides any active overlays.
        """
        self._readonly = readonly
        if readonly:
            # Hide any active overlays (hover PathField, browse buttons).
            self._hide_hover_field()
            if self._mgr is not None:
                self._mgr.detach()

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

    def _ins_coefs_path(self, inp_iid: Any, coefs_path: str) -> None:
        """Insert the ``input.coefs_path`` child row (shared by both build modes).

        ``check: "exists"`` marks the row for red-fg validation when the path
        doesn't exist on disk (see :meth:`_apply_validations`).
        """
        self._ins(
            inp_iid,
            "coefs_path",
            [coefs_path] + [""] * (self._nv - 1),
            "",
            meta={
                "key": "coefs_path",
                "check": "exists",
                "is_string": True,
                "max_col": 1,
                "path": "input.coefs_path",
                "browse": True,
            },
        )

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
                "check": "exists",
                "is_string": True,
                "max_col": 1,
                "style": "node",
                "browse": True,
            },
            open_=True,
        )

        # Always create time_ranges row — empty cells get ghost placeholders via unified _apply_placeholders
        tr = inp.get("time_ranges") or []
        self._ins(
            inp_iid,
            "time_ranges",
            [any2str(x) for x in tr] + [""] * (self._nv - len(tr)),
            "",
            meta={
                "label": "time_ranges",
                "is_string": True,
                "max_col": self._nv,
                "path": "input.time_ranges",
            },
        )

        self._ins_coefs_path(inp_iid, any2str(inp.get("coefs_path", "")))

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

        # ── process-stage calibration correction ── (DRY: same Annotated shape pattern as coefs)
        if calib := inp.get("calib"):
            try:
                from tcm_gui.cli_cfg import infer_coef_shapes
                from tcm.schema import ConfigInCalib_InclProc

                _CALIB_SHAPES_COEFS = infer_coef_shapes(ConfigInCalib_InclProc)
            except Exception:
                _CALIB_SHAPES_COEFS = {}
            calib_iid = self._ins(inp_iid, "calib", [""] * self._nv, "", meta={"path": "input.calib"})
            for ck, cv in calib.items():
                shape = _CALIB_SHAPES_COEFS.get(ck, ())
                if len(shape) == 1 and shape[0] > 0:
                    row = [any2str(x) for x in (cv or [])] if isinstance(cv, (list, tuple)) else []
                    row += [""] * (shape[0] - len(row))
                    self._ins(
                        calib_iid,
                        ck,
                        row + [""] * (self._nv - len(row)),
                        "",
                        meta={"path": f"input.calib.{ck}", "max_col": shape[0]},
                    )
                elif (
                    shape == () and isinstance(cv, (list, tuple, type(None))) and ck.startswith("time_ranges")
                ):
                    self._ins(
                        calib_iid,
                        ck,
                        [any2str(x) for x in (cv or [])] + [""] * (self._nv - len(cv or [])),
                        "",
                        meta={"path": f"input.calib.{ck}", "is_string": True, "max_col": self._nv},
                    )
                elif shape == ():
                    self._ins_generic(calib_iid, ck, cv)
                else:
                    self._ins_generic(calib_iid, ck, cv)

    def _build_metadata(self) -> None:
        """Top-level ``metadata`` node (sibling of ``input``) with paired rows.

        ``metadata`` itself is the device-file path (always editable, browseable
        — same floated field as ``input``).  Children are 6 paired rows from
        ``_meta_pairs.PAIRS``; empty cells show gray example ghosts that vanish
        on edit (``CellPlaceholder``), never persisted as ``"?"``.
        """
        md_list: list[Any] | None = getattr(self, "_metadata", None)
        if md_list is None or (len(md_list) < 8 and not any(md_list or [])):
            tr = (self._cfg.get("input", {}) or {}).get("time_ranges") or []
            if tr and len(tr) >= 2:
                base = [None] * 11
                base[6], base[7] = tr[0], tr[-1]
                md_list = base
            elif not md_list:
                md_list = [None] * 11
        if len(md_list) < 11:
            md_list = list(md_list) + [None] * (11 - len(md_list))
        paired = _meta_pairs.to_display(md_list)
        # Device-file path — always editable (default when file absent).
        _path = getattr(self, "_metadata_path", None)
        if _path is None:
            # Default: device_dir/info_devices.yaml (parent of _raw)
            try:
                from pathlib import Path as _P

                from tcm import paths as _paths

                # Judge the RAW string before resolving — ``Path("").absolute()``
                # is the cwd, so a no-path GUI launch would otherwise probe the
                # launch directory and ``find_dir_raw_absolute`` would log the
                # misleading "Not standard input path" warning on the project
                # root.  Repo-internal paths are skipped too: their default
                # device file would land in the code tree — the scan owns that
                # error verdict, not this default derivation.
                _probe_str = str((self._cfg.get("input", {}) or {}).get("path") or "").strip()
                _probe = _P(_probe_str).absolute() if _probe_str and _probe_str != "." else None
                if _probe is not None and (_probe == _constants.REPO_ROOT or _constants.REPO_ROOT in _probe.parents):
                    _probe = None
                _ddir = _paths.find_dir_raw_absolute(_probe).parent if _probe is not None else None
                _path = str(_ddir / "info_devices.yaml") if _ddir else ""
            except Exception:
                _path = ""
        meta_iid = self._ins(
            "",
            "metadata",
            [_path] + [""] * (self._nv - 1),
            "",
            meta={
                "key": "metadata",
                "path": "metadata",
                "style": "node",
                "max_col": 1,
                "is_metadata_root": True,
                "is_string": True,
                "browse": True,
                "check": "exists",
                "metadata_path": _path,
            },
            open_=True,
        )
        for label, idxs in _meta_pairs.PAIRS:
            vals = paired.get(label, ["?"] * len(idxs))
            # Empty "?" → ghost via CellPlaceholder, not literal "?".
            # Distinguish vacuous vs valued "?" by checking md_list indices.
            display_vals: list[str] = []
            ghost_cols: list[int] = []
            for j, v in enumerate(vals):
                idx = idxs[j]
                raw = md_list[idx] if idx < len(md_list) else None
                is_empty = v == "?" and _meta_pairs.is_placeholder(raw)
                if is_empty:
                    display_vals.append("")
                    ghost_cols.append(j)
                else:
                    display_vals.append(v)
            row_vals = display_vals + [""] * (self._nv - len(display_vals))
            is_time = label == "time_range"
            iid = self._ins(
                meta_iid,
                label,
                row_vals,
                "",
                meta={
                    "label": label,
                    "path": f"metadata.{label.replace(', ', '_').replace('/', '_')}",
                    "is_string": True,
                    "is_metadata": True,
                    "max_col": len(idxs),
                    "has_date": is_time,
                    "_ghost_cols": ghost_cols,
                },
            )
            # Remember ghosts for placeholder pass — _ph.show needs row index later
            if ghost_cols:
                self._meta[iid]["_ghost_example"] = [
                    _meta_pairs.EXAMPLES.get(label, ["?", "?"])[j] for j in range(len(idxs))
                ]
        self._take_metadata_snapshot()

    def _reload_metadata_from(self, path_str: str) -> None:
        """Load metadata from existing file on browse select."""
        try:
            from meta_finder.io_info_files import read_metadata_file
            from tcm import format as _fmt

            data = read_metadata_file(Path(path_str).expanduser())
            stem = getattr(self, "_page_stem", "") or ""
            pcid = _fmt.to_pcid_from_name(_fmt.stem_to_pcid(stem)) if stem else None
            ent = None
            if pcid is not None:
                for cand in (pcid, pcid.replace("_", "")):
                    if cand in data:
                        ent = data[cand]
                        break
            if ent is None and data:
                ent = next(iter(data.values()))
            if isinstance(ent, dict):
                arr = next((list(v) for v in ent.values() if isinstance(v, (list, tuple))), None)
            else:
                arr = list(ent) if isinstance(ent, (list, tuple)) else None
            if arr is not None:
                self._metadata = arr
                self._metadata_path = path_str
                self._rebuild_metadata_rows()
        except Exception:
            pass

    def _rebuild_metadata_rows(self) -> None:
        """Rebuild only the metadata subtree — keep node expanded and ghosts visible."""
        # Discard stale placeholders before structural change — row indices will shift
        self._ph.clear_all(self.sh)
        to_del = [
            iid for iid, m in list(self._meta.items()) if m.get("is_metadata") or m.get("is_metadata_root")
        ]
        for iid in to_del:
            with suppress(Exception):
                self.sh.delete_row(self._row_map().get(iid, -1))
            self._meta.pop(iid, None)
        self._build_metadata()
        self._apply_open()
        self._rebuild_row_caches()
        self._apply_styles()
        self._apply_placeholders()
        self._apply_default_fg()
        self._apply_validations()
        with suppress(Exception):
            self.sh.redraw()
        self._take_metadata_snapshot()

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
                "check": "exists",
                "is_string": True,
                "max_col": 1,
                "style": "node",
                "browse": True,
            },
            open_=True,
        )

        self._ins_coefs_path(inp_iid, any2str(inp.get("coefs_path", "")))

        # infer calib shapes same way as coefs (Annotated metadata)
        try:
            from tcm_gui.cli_cfg import infer_coef_shapes
            from tcm.schema import ConfigInCalib_InclProc

            _CALIB_SHAPES = infer_coef_shapes(ConfigInCalib_InclProc)
        except Exception:
            _CALIB_SHAPES = {}

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
            elif k == "calib" and isinstance(v, dict):
                calib_iid = self._ins(inp_iid, "calib", [""] * self._nv, "", meta={"path": "input.calib"})
                for ck, cv in v.items():
                    shape = _CALIB_SHAPES.get(ck, ())
                    if len(shape) == 1 and shape[0] > 0:
                        # 1-D numeric array like g0xyz[3], coordinates[2] — same pattern as Cg
                        row = [any2str(x) for x in (cv or [])] if isinstance(cv, (list, tuple)) else []
                        row += [""] * (shape[0] - len(row))
                        self._ins(
                            calib_iid,
                            ck,
                            row + [""] * (self._nv - len(row)),
                            "",
                            meta={"path": f"input.calib.{ck}", "max_col": shape[0]},
                        )
                    elif (
                        shape == ()
                        and isinstance(cv, (list, tuple, type(None)))
                        and ck.startswith("time_ranges")
                    ):
                        # date lists (time_ranges_*) — treat as is_string multi-col via shape-less fallback
                        self._ins(
                            calib_iid,
                            ck,
                            [any2str(x) for x in (cv or [])] + [""] * (self._nv - len(cv or [])),
                            "",
                            meta={"path": f"input.calib.{ck}", "is_string": True, "max_col": self._nv},
                        )
                    elif shape == ():
                        self._ins_generic(calib_iid, ck, cv)
                    else:
                        self._ins_generic(calib_iid, ck, cv)
            else:
                self._ins_generic(inp_iid, k, v)

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
        """1D without dates (Cg, Ch, P, …): single row without children"""
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

        # DRY: use meta_finder's field typology via _meta_pairs.
        # Try to interpret metadata fields with those indices as numeric.
        if m.get("is_metadata"):
            lbl = m.get("label", "")
            try:
                idxs = dict(_meta_pairs.PAIRS)[lbl]
                if 0 <= c < len(idxs) and idxs[c] in _meta_pairs.NUMERIC_IDXS:
                    ok = parse_float(val) is not None
                    if not ok:
                        _l.debug(
                            "edit r=%s c=%s iid=%s path=%s val=%r → REJECT (not numeric)",
                            event.row,
                            c,
                            iid,
                            m.get("path"),
                            val,
                        )
                    return val if ok else None
            except Exception:
                pass
            # text-like metadata (point/symbol/comment/time_range) — free-form
            if m.get("is_string"):
                return val
            return val

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

    def _on_begin_edit_cell(self, event) -> str | None:
        """Detach any previous browse button; attach for path-type rows.
        Browse rows always edit col 0 — overflow clicks rerouted here.
        Non-data cells (beyond max_col) are rejected — except the date cell."""
        if self._readonly:
            return None  # veto editing in readonly mode (non-full before scan)
        if self.on_edit_begin is not None:
            self.on_edit_begin()
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
        # Clear ghost/placeholder so the user starts empty; _on_editor_closed restores via _placeholder_for.
        _ph = getattr(self, "_ph", None)
        if _ph is not None and (int_row := self._internal_row(iid)) is not None:
            if _ph.has(int_row, event.column):
                _ph.clear(self.sh, int_row, event.column)
                return ""
        ri = self._internal_row(iid) if m.get("browse") else None
        if self._mgr is not None and ri is not None:
            self._mgr.attach(ri, 0, iid=iid)
            # Row-specific browse button status hint:
            # coefs_path → short files hint (manager overlay is files-only);
            # other rows (input.path) → static hint from ConfigSheet init.
            if (ov := self._mgr._ov) is not None:
                ov._status_hint = (
                    _S["browse_btn.status_files"] if m.get("key") == "coefs_path" else self._status_hint
                )
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
        # Re-validate the edited cell after commit — red fg if its check fails.
        if m.get("check"):
            self.sh.after_idle(lambda iid=iid: self._apply_validations(iid))

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
        self._rebuild_row_caches()
        self._apply_styles()
        self._apply_placeholders()
        self._apply_default_fg()
        self.sh.redraw()

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
            if self._is_date_only_row(m) and (ir := self._internal_row(iid)) is not None:
                dc = _DATE_COL - self.DATA_COL_BASE
                self.sh.after(1, lambda rr=ir, dcc=dc: _safe_select(self.sh, rr, dcc))
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
            _safe_select(self.sh, ri, target)

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
