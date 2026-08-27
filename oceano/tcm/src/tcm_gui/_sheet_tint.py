"""Default-value tinting, ghost placeholders and time_ranges sync state.

Mixin for :class:`tcm_gui.coef_sheet.ConfigSheet` — everything that decides
whether a cell/node is *at default* and paints it accordingly:

* **Cell gray** (:meth:`SheetTintMixin._apply_default_fg`) — value equals the
  config dataclass factory default (:func:`tcm_gui.cli_cfg.default_for_path`).
* **Node label blue** (:meth:`SheetTintMixin._node_at_default`) — every cell in
  the subtree matches its default.
* **Ghost placeholders** (:meth:`SheetTintMixin._apply_placeholders`) — dim
  example text in empty cells via :class:`tcm_gui._placeholder.CellPlaceholder`;
  ghost cells always read as ``""`` (:meth:`SheetTintMixin._cell_str`).
* **time_ranges sync** (:meth:`SheetTintMixin._time_ranges_relation`) — the
  relation of the *live* sheet window to the info_devices window is recomputed
  on every hover/edit; nothing is cached from the scan.

Default semantics (single source — :meth:`SheetTintMixin._default_for_cell`):

* config leaf → dataclass default; ``None``/empty-list default ⇒ ``""`` per
  cell (empty means default, e.g. ``input.time_ranges``,
  ``input.calib.time_ranges_*``);
* ``is_metadata`` rows → ghost-empty (``""``) except ``time_range`` which
  defaults to the live ``input.time_ranges[0, -1]``;
* ``is_metadata_root`` / containers → :data:`NO_DEFAULT` (state = children).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from contextlib import suppress
from tkinter import TclError
from typing import Any

import tcm_gui.theme
from tcm import _meta_pairs

from ._cell_spec import any2str
from ._i18n import STRINGS as _S
from .cli_cfg import NO_DEFAULT, default_for_path

_l = logging.getLogger(__name__)

_DATE_FMT = "YYYY-MM-DDTHH:MM:SS"  # ISO 8601 — ghost for empty date cells
_DATE_PH_COL = 1  # 0-based tksheet col of the date cell (meta col 2, not tree col 0)
_DATE_COL = _DATE_PH_COL + 1  # meta col (1-based): 1=col0, 2=date, 3=col2...

# Central placeholder registry — full Hydra path → example texts per column.
# Ghost shown only for empty cells; single source via _placeholder_for() +
# _apply_placeholders().  Type-derived generics cover unseen calib vectors;
# only hand-picked examples live here.
_PH_BY_FIELD: dict[str, list[str]] = {
    "input": [_S.get("path_field.placeholder", "D:/data/_raw")],
    "input.path": [_S.get("path_field.placeholder", "D:/data/_raw")],
    **{
        f"metadata.{lbl.replace(', ', '_').replace('/', '_')}": ex for lbl, ex in _meta_pairs.EXAMPLES.items()
    },
}


class SheetTintMixin:
    """Default tinting + ghost placeholders + live time_ranges sync relation."""

    # ── ph-aware reads ────────────────────────────────────────────────

    def _cell_str(self, iid: Any, j: int) -> str:
        """Cell value as trimmed string; ghost placeholder cells read as ``""``.

        Single read primitive for every consumer of sheet values (tint,
        validation, YAML write-back) — the dim hint never leaks as data.
        """
        vals = self.sh.item(iid).get("values") or ()
        cur = str(vals[j]) if j < len(vals) else ""
        if cur.strip() and (ph := getattr(self, "_ph", None)) is not None:
            with suppress(AttributeError):
                if (r := self._internal_row(iid)) is not None and ph.has(r, j):
                    return ""
        return cur.strip()

    def _own_cols(self, m: Mapping[str, Any], *, with_len: bool = True) -> int:
        """Editable value-cell count of a row: scalar→1, ``time_ranges``→``_nv``.

        ``with_len=False`` — container rows (``1d`` parents carrying ``len``
        for shape only) report 0; their child row holds the values.
        """
        if m.get("type") == "scalar":
            return 1
        if m.get("label") == "time_ranges":
            return int(m.get("max_col", self._nv))
        return int(m.get("max_col") or (m.get("len") if with_len else 0) or 0)

    # ── live time_ranges window ───────────────────────────────────────

    def _time_ranges_iid(self) -> Any | None:
        """iid of the ``input.time_ranges`` row (never the metadata ``time_range`` row)."""
        return next(
            (
                iid
                for iid, m in self._meta.items()
                if m.get("path") == "input.time_ranges"
                or (m.get("label") == "time_ranges" and not m.get("is_metadata"))
            ),
            None,
        )

    def _sheet_time_ranges(self) -> list[str]:
        """Live ``input.time_ranges`` values (ghost cells excluded); cfg fallback."""
        if (iid := self._time_ranges_iid()) is not None:
            if cleaned := [
                v for v in (self._cell_str(iid, j) for j in range(self._own_cols(self._meta[iid]))) if v
            ]:
                return cleaned
        tr = (getattr(self, "_cfg", {}).get("input", {}) or {}).get("time_ranges") or []
        return [str(v).strip() for v in tr if str(v).strip()]

    def _metadata_time_range_default(self, col_idx: int) -> str:
        """``input.time_ranges[[0, -1]]`` — DRY for cell and tree tint."""
        # Walrus-in-conditional is a trap: the condition reads ``tr`` before
        # the assignment runs — assign first (regression: UnboundLocalError
        # on load of a sheet with a metadata time_range row).
        tr = self._sheet_time_ranges()
        return tr[0 if col_idx == 0 else -1] if tr else ""

    def _time_ranges_relation(self) -> tuple[str, list[str]]:
        """``(status, meta_tr)`` — live relation of the sheet window to info_devices.

        ``""`` — window or info_devices range absent; ``"equal"``; ``"broader"``
        — extends beyond either end (review); ``"differs"`` — narrowed/shifted.
        """
        meta_tr = [str(t) for t in ((getattr(self, "_sync_status", None) or {}).get("meta_tr") or ()) if t]
        if (
            len(meta_tr) < 2
            and isinstance(md := getattr(self, "_metadata", None), (list, tuple))
            and len(md) >= 8
        ):
            meta_tr = [str(t) for t in (md[6], md[7]) if not _meta_pairs.is_placeholder(t)]
        if len(meta_tr) < 2 or len(cur := self._sheet_time_ranges()) < 2:
            return "", meta_tr
        s, e, ms, me = cur[0], cur[-1], meta_tr[0], meta_tr[-1]
        if (s, e) == (ms, me):
            return "equal", meta_tr
        return ("broader" if s < ms or e > me else "differs"), meta_tr

    def _time_ranges_detail(self) -> str:
        """Hover detail from the live relation (``""`` — nothing to report).

        Never cached: a narrowed window must not keep claiming it *matches*
        info_devices (regression: stale scan-time status lied on hover).
        """
        st, meta_tr = self._time_ranges_relation()
        if not st:
            return ""
        key = {"equal": "kept"}.get(st, st)
        return _S.get(f"time_ranges.hover.{key}", "{s} — {e}").format(s=meta_tr[0], e=meta_tr[-1])

    def apply_time_ranges_sync_status(self, status: dict | None) -> None:
        """Store the info_devices window; tint the live ``time_ranges`` row.

        The hover detail is derived live by :meth:`_time_ranges_detail` — the
        scan-time ``status`` classification is deliberately not trusted.
        """
        self._sync_status = status
        self._apply_time_ranges_tint()

    def _apply_time_ranges_tint(self) -> None:
        """Warning fg on window cells while the live range is broader than info_devices."""
        st = self._time_ranges_relation()[0]
        ph = getattr(self, "_ph", None)
        if (target := self._time_ranges_iid()) is None or (row := self._row_map().get(target)) is None:
            return
        for c in range(self._own_cols(self._meta[target])):
            if ph is not None and ph.has(row, c):
                continue  # ghost cells manage their own dim fg
            self.sh.highlight_cells(
                row=row,
                column=c,
                fg=tcm_gui.theme.TAG_COLORS["warning"] if st == "broader" else self._fg_default,
                redraw=False,
            )
        self.sh.redraw()

    # ── defaults ──────────────────────────────────────────────────────

    def _default_for_cell(self, iid: Any, m: dict, col_idx: int) -> Any:
        """Default value for cell at 0-based *col_idx*, or :data:`NO_DEFAULT`.

        ``is_metadata`` rows default to ghost-empty; ``time_range`` defaults to
        the live ``input.time_ranges[0, -1]`` (copy-paste from that row grays
        instantly, re-evaluated on every edit via ``_apply_edit_value``).
        ``None``/empty-list defaults ⇒ every cell defaults to ``""`` — so an
        emptied ``time_ranges_*`` row reads as at-default, not as modified.
        """
        if m.get("is_metadata_root"):
            return NO_DEFAULT
        if m.get("is_metadata"):
            if m.get("label") == "time_range" and 0 <= col_idx <= 1:
                return self._metadata_time_range_default(col_idx)
            return ""
        path = str(m.get("path") or "")
        if not path:
            return NO_DEFAULT
        if m.get("type") == "input" and col_idx == 0:
            path += ".path"
        default = default_for_path(path)
        if default is NO_DEFAULT or isinstance(default, dict):
            return NO_DEFAULT
        if default is None:
            return ""
        if isinstance(default, (list, tuple)):
            return any2str(default[col_idx]) if col_idx < len(default) else ""
        return default if col_idx == 0 else NO_DEFAULT

    def _node_at_default(self, iid: Any) -> bool:
        """True iff every cell in the node's subtree matches its default.

        Ghost placeholders count as ``""`` — empty metadata children (whose
        default IS empty) are at default → blue label; a child holding real
        data breaks the chain.  Parents without own defaults (``metadata``
        root, ``coefs`` container) defer to their children.
        """
        m = self._meta.get(iid, {})
        own_ok, has_defined = True, False
        for j in range(self._own_cols(m, with_len=False)):
            if (dv := self._default_for_cell(iid, m, j)) is NO_DEFAULT:
                continue
            has_defined = True
            if any2str(self._cell_str(iid, j)) != any2str(dv):
                own_ok = False
                break
        kids = [k for k, km in self._meta.items() if km.get("parent") == iid]
        if not has_defined and kids:
            return all(self._node_at_default(k) for k in kids)
        if kids:
            return own_ok and all(self._node_at_default(k) for k in kids)
        return own_ok and (has_defined or self._own_cols(m, with_len=False) == 0)

    # ── tint application ──────────────────────────────────────────────

    def _apply_default_fg(self) -> None:
        """Gray-out cells whose values match config dataclass factory defaults."""
        ph = getattr(self, "_ph", None)
        for iid, m in self._meta.items():
            if (r := self._row_map().get(iid)) is None:
                continue
            for j in range(self._own_cols(m)):
                if ph is not None and ph.has(r, j):
                    continue  # ghost already dim — its emptiness IS the default state
                if not (cur := self._cell_str(iid, j)):
                    continue
                if (dv := self._default_for_cell(iid, m, j)) is NO_DEFAULT:
                    continue
                if any2str(cur) == any2str(dv):
                    self.sh.highlight_cells(
                        row=r, column=j, fg=tcm_gui.theme.DEFAULT_FG, redraw=False, overwrite=False
                    )

    def _apply_edit_value(self, iid: Any, col: int, value: str) -> None:
        """Restyle cell + ancestors after a committed value; re-tint the sync row."""
        m = self._meta.get(iid, {})
        if m.get("is_metadata"):
            # Defer — sheet may not have committed yet; keeps ``metadata*`` label in sync
            self.sh.after_idle(self._apply_metadata_dirty_label)
        if m.get("label") == "time_ranges" and not m.get("is_metadata"):
            self.sh.after_idle(self._apply_time_ranges_tint)  # broader⇄normal may flip
        if (ri := self._row_map().get(iid)) is None:
            return

        ph = getattr(self, "_ph", None)
        if (dv := self._default_for_cell(iid, m, col)) is not NO_DEFAULT and not (
            ph is not None and ph.has(ri, col)
        ):
            self.sh.highlight_cells(
                row=ri,
                column=col,
                fg=tcm_gui.theme.DEFAULT_FG if any2str(value) == any2str(dv) else self._fg_default,
                redraw=False,
                overwrite=False,
            )

        # node labels: propagate at-default state up the ancestor chain;
        # the ``input`` row keeps normal fg — never in the blue/gray toggle
        node: Any = iid
        while node is not None:
            if (nr := self._row_map().get(node)) is not None:
                nm = self._meta.get(node, {})
                node_fg = (
                    tcm_gui.theme.FG_DEFAULT
                    if nm.get("type") == "input"
                    else (tcm_gui.theme.BLUE_FG if self._node_at_default(node) else self._fg_default)
                )
                self.sh.highlight_cells(
                    row=nr, column=0, canvas="index", fg=node_fg, redraw=False, overwrite=False
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

    # ── ghost placeholders ────────────────────────────────────────────

    def _ghost_fg(self) -> str:
        """Ghost fg above this sheet's own table bg — live, theme-aware.

        :func:`tcm_gui.theme.ghost_fg` blends ≈3× fainter than the gray
        default; the background comes from the rendered MT canvas so dark
        mode (``change_theme``) is honored without a second palette table.
        """
        with suppress(TclError, AttributeError):
            bg = tcm_gui.theme.tk_color_to_hex(self.sh.MT, self.sh.MT.cget("background"))
            return tcm_gui.theme.ghost_fg(bg)
        return tcm_gui.theme.ghost_fg(tcm_gui.theme.ENTRY_BG_FALLBACK)

    def _placeholder_for(self, iid: Any, col_idx: int) -> str | None:
        """Ghost text for (iid, col): full-Hydra-path registry → field type generic."""
        m = self._meta.get(iid, {})
        if m.get("label") == "time_range" and col_idx in (0, 1):
            return _meta_pairs.EXAMPLES["time_range"][col_idx]
        if m.get("has_date") and col_idx == _DATE_PH_COL:
            return _DATE_FMT
        path = str(m.get("path") or "")
        if exs := _PH_BY_FIELD.get(path):
            return exs[col_idx if col_idx < len(exs) else col_idx % len(exs)]
        if "time_ranges" in path:  # date lists classify as text
            return _DATE_FMT
        with suppress(Exception):
            spec = self._cell_spec_for(iid, m, col_idx + self.DATA_COL_BASE)
            if spec.kind == "date":
                return _DATE_FMT
            if spec.kind == "number":
                return "0"
            if spec.kind == "text" and m.get("is_string") and int(m.get("max_col", 0)) > 1:
                return _DATE_FMT
        return None

    def _apply_placeholders(self) -> None:
        """Show ghosts in empty cells of every row (left-aligned, dim)."""
        row_of = self._row_map()
        dim_fg = self._ghost_fg()
        for iid, m in self._meta.items():
            if (r := row_of.get(iid)) is None:
                continue
            # date-only parent (kVabs-style, max_col 0) → single date cell
            maxc = (
                _DATE_PH_COL + 1
                if m.get("has_date") and not (m.get("max_col") or m.get("len"))
                else self._own_cols(m)
            )
            vals = self.sh.item(iid).get("values") or ()
            for c in range(maxc):
                if c < len(vals) and str(vals[c]).strip():
                    continue
                if self._ph.has(r, c):
                    continue
                if (ph := self._placeholder_for(iid, c)) is None:
                    continue
                # left-aligned to be distinct from right-aligned values
                self._ph.show(self.sh, r, c, ph, dim_fg)
                with suppress(Exception):
                    self.sh.align_cells(r, c, align="w", redraw=False)

    def _restore_placeholder(self, iid: Any, r: int, c: int) -> None:
        """Show the ghost in (r, c) when the cell is empty — after a deletion commit."""
        ph = getattr(self, "_ph", None)
        if ph is None or ph.has(r, c):
            return
        if str(self.sh.get_cell_data(r, c) or "").strip():
            return
        if txt := self._placeholder_for(iid, c):
            ph.show(self.sh, r, c, txt, self._ghost_fg())
            with suppress(Exception):
                self.sh.align_cells(r, c, align="w", redraw=False)
            self.sh.redraw()

    def _on_editor_closed(self, redraw: bool = True) -> None:
        """Restore the ghost when an editor closed on an empty cell (unified hook).

        ``MT.hide_text_editor_and_dropdown`` is the single funnel EVERY edit
        close (Escape, Enter-commit, click-away, FocusOut) goes through —
        ``open_text_editor`` calls plain ``hide_text_editor``, so no false fire
        mid-open.  Also notifies ``on_edit_end`` (App releases the edit-time
        status freeze here).
        """
        self._mt_close_editor_orig(redraw=redraw)
        if (cb := self.on_edit_end) is not None:
            try:
                cb()
            except Exception:
                _l.exception("on_edit_end callback failed")
        with suppress(AttributeError, TypeError, IndexError, TclError):
            r, c = self.sh.MT.text_editor.coords
            if (iid := self._iid_at_row(r)) is not None and (int_row := self._internal_row(iid)) is not None:
                self._restore_placeholder(iid, int_row, c)
