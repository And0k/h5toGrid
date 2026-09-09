"""Sheet styling: per-cell alignment/widgets, node fg, path-existence validation.

Mixin for :class:`tcm_gui.coef_sheet.ConfigSheet`.

* :meth:`SheetStylesMixin._apply_styles` — tree-label fg (blue when the node
  is at default, see :mod:`tcm_gui._sheet_tint`), uniform bg for browse rows,
  date alignment + blue fg, dropdowns/checkboxes from the field's
  :class:`tcm_gui._cell_spec.CellSpec`, column-resize zones.
* :meth:`SheetStylesMixin._apply_validations` — red fg on rows failing their
  ``check``: ``"exists"`` path rows whose path resolves to nothing,
  ``"sorted"`` date rows (``time_ranges*``) breaking ascending order; gray fg
  when the value matches its config default via
  :meth:`SheetTintMixin._default_for_cell`; reads via
  :meth:`SheetTintMixin._cell_str` so a ghost placeholder (deleted value) is
  skipped instead of validated.
"""

from __future__ import annotations

import glob as _glob_mod
from collections.abc import Mapping
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from tkinter import TclError
from typing import Any

import tcm_gui.theme
from tcm import _meta_pairs

from ._cell_spec import (
    CellSpec,
    NUMBER_SPEC,
    TEXT_SPEC,
    any2str,
    as_bool,
    as_date,
    enum_values,
    spec_for_path,
)
from .cli_cfg import NO_DEFAULT


def _path_exists(path_str: str) -> bool:
    """True iff *path_str* (after ``~`` expansion) exists or matches files via glob."""
    return Path(path_str).expanduser().exists() or bool(_glob_mod.glob(path_str))


def _input_path_matches_config(path_str: str, page_stem: str) -> bool:
    """True iff *path_str* stem matches the config stem per ``format.pcid_key``.

    The input path "belongs" to this config when their canonical identities
    align (prefix/case/comment separators normalized — see
    ``docs/reference/io_formats.md``).  Missing/empty stems never match.
    """
    if not path_str or not page_stem:
        return False
    try:
        from tcm.format import pcid_key

        return pcid_key(Path(path_str).stem) == pcid_key(page_stem)
    except Exception:
        return False


class SheetStylesMixin:
    """Row/cell styling and validation for the config tree."""

    # Coef meta types — always numeric; skip Hydra path resolution for these.
    _COEF_TYPES = frozenset({"2d", "1d", "1d_flat", "scalar", "_coef_child"})

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
        # Metadata numeric columns: sea_depth/h_above/lat/lon/burst_dt/bursts_t → number.
        if m.get("is_metadata"):
            # paired label → column within pair (0/1) → 11-array index
            try:
                idxs = dict(_meta_pairs.PAIRS)[m.get("label", "")]
                col0 = meta_col - self.DATA_COL_BASE
                if 0 <= col0 < len(idxs) and idxs[col0] in _meta_pairs.NUMERIC_IDXS:
                    return NUMBER_SPEC
            except Exception:
                pass
            return TEXT_SPEC
        # Strip array indices (e.g. "input.coefs.Ag[0]" → "input.coefs.Ag")
        path = str(m.get("path", iid))
        clean = path.split("[")[0] if "[" in path else path
        return spec_for_path(self._config_root, clean, self._return_enum)

    @staticmethod
    def _clear_cell_widgets(sh: Any, r: int, c: int) -> None:
        """Remove existing dropdown/checkbox at (r, c) before re-creating."""
        with suppress(AttributeError, KeyError, ValueError, TypeError):
            sh.delete_dropdown(r, c)
        with suppress(AttributeError, KeyError, ValueError, TypeError):
            sh.delete_checkbox(r, c)

    def _apply_metadata_dirty_label(self) -> None:
        """Append ``*`` to ``metadata`` tree label when metadata dirty, remove when clean."""
        for iid, m in self._meta.items():
            if not m.get("is_metadata_root"):
                continue
            if (r := self._row_map().get(iid)) is None:
                continue
            dirty = getattr(self, "is_metadata_dirty", False) and self.is_metadata_dirty()
            label = "metadata*" if dirty else "metadata"
            # Tree label via highlight is fg only; update text via item
            with suppress(Exception):
                cur = self.sh.item(iid).get("text", "")
                if cur != label:
                    self.sh.item(iid, text=label)
            break

    def _apply_styles(self) -> None:
        sh = self.sh

        bg = tcm_gui.theme.resolved_frame_bg(sh)
        meta_bg = tcm_gui.theme.META_TREE_BG
        config_bg = tcm_gui.theme.CONFIG_TREE_BG
        self._fg_default = tcm_gui.theme.FG_DEFAULT

        with suppress(AttributeError, TypeError):
            sh.set_options(index_background=bg)

        row_of = self._row_map()
        first_data_col = self.DATA_COL_BASE - 1  # tksheet 0-based
        total_cols = sh.total_columns()
        resize_cells: set[tuple[int, int]] = set()

        # Metadata root + all its descendants get the tinted tree column.
        meta_roots = {iid for iid, m in self._meta.items() if m.get("is_metadata_root")}
        meta_descendants = set(meta_roots)
        for iid, m in self._meta.items():
            if iid in meta_descendants:
                continue
            cur = m.get("parent")
            while cur:
                if cur in meta_descendants:
                    meta_descendants.add(iid)
                    break
                cur = self._meta.get(cur, {}).get("parent")

        for iid, m in self._meta.items():
            if (r := row_of.get(iid)) is None:
                continue

            is_input = m.get("type") == "input"
            is_browse = is_input or m.get("browse")
            is_meta = iid in meta_descendants

            # ── 1) node label — treeview column = "index" canvas ──
            # Input row: button-face bg + normal black fg; other rows: blue/black fg
            # Metadata subtree gets a tinted tree column to visually separate it;
            # non-metadata config nodes get a cooler lavender tint.
            sh.highlight_cells(
                row=r,
                column=0,
                canvas="index",
                bg=meta_bg if is_meta else config_bg,
                fg=(
                    tcm_gui.theme.FG_DEFAULT
                    if is_input
                    else (
                        tcm_gui.theme.NODE_DEFAULT_VALS_FG if self._node_at_default(iid) else self._fg_default
                    )
                ),
                redraw=False,
            )

            # ── 1b) browse/input rows: paint ALL columns uniform ────
            # Prevents colour mismatch between col-0 and overflow columns.
            if is_browse:
                for col in range(first_data_col, total_cols):
                    sh.highlight_cells(row=r, column=col, bg=bg, redraw=False)

            date_cols = tuple(int(c) for c in (m.get("meta_date_cols") or ()))
            date_set = frozenset(date_cols)

            # ── 2) metadata row bg up to last date cell inclusive ───
            if date_cols and (last_tk := max(date_cols) - self.DATA_COL_BASE) >= first_data_col:
                for col in range(first_data_col, last_tk + 1):
                    sh.highlight_cells(row=r, column=col, bg=bg, redraw=False)

            # ── 3) date alignment + blue fg ─────────────────────────
            # Left-align so allow_cell_overflow extends to next empty cell
            # (right-aligned dates are clipped on the left when narrow, ghost
            # placeholders are left-aligned and overflow — make real dates match).
            for dc in date_cols:
                col = dc - self.DATA_COL_BASE
                if col >= first_data_col:
                    sh.align_cells(r, col, align="w", redraw=False)
                    if m.get("date_style") == "blue":
                        sh.highlight_cells(
                            row=r,
                            column=col,
                            fg=tcm_gui.theme.NODE_DEFAULT_VALS_FG,
                            highlight_fg=tcm_gui.theme.NODE_DEFAULT_VALS_FG,
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

    def _apply_validations(self, target_iid: Any = None) -> None:
        """Red fg on any cell whose ``check`` validation fails.

        Called after every edit commit and at the end of ``load()``.
        ``check: "exists"`` rows (``input.path``, ``input.coefs`` path cell) are
        marked red when the path doesn't exist on disk; glob patterns are red
        only when zero matches; ``~`` is expanded.  ``check: "sorted"`` rows
        (``input.time_ranges``, ``metadata.time_range``) — red on any date cell
        breaking ascending order, see :meth:`_validate_sorted_dates`.  Values
        are read through :meth:`_cell_str` — a deleted value (ghost
        placeholder) reads empty and is skipped, never validated against the
        placeholder text.  Sentinels (``<…>``) are never marked invalid.
        """
        sh = self.sh
        row_of = self._row_map()

        for iid, m in self._meta.items():
            if not (check := m.get("check")):
                continue
            if target_iid is not None and iid != target_iid:
                continue
            if (r := row_of.get(iid)) is None:
                continue

            if check == "exists":
                path_str = self._cell_str(iid, 0)
                if not path_str or path_str.startswith("<"):
                    continue

                if _path_exists(path_str):
                    # Restore normal fg: gray if value matches config default, else default fg.
                    dv = self._default_for_cell(iid, m, 0)
                    if dv is not NO_DEFAULT and any2str(path_str) == any2str(dv):
                        restore_fg = tcm_gui.theme.CELL_DEFAULT_VAL_FG
                    elif m.get("type") == "input" and _input_path_matches_config(
                        path_str, getattr(self, "_page_stem", "")
                    ):
                        # input.path "belongs" to this config — a different but same-probe
                        # file (stem matches per pcid_key) also reads as default.
                        restore_fg = tcm_gui.theme.CELL_DEFAULT_VAL_FG
                    else:
                        restore_fg = self._fg_default
                    sh.highlight_cells(row=r, column=0, fg=restore_fg, redraw=False)
                else:
                    sh.highlight_cells(row=r, column=0, fg=tcm_gui.theme.INVALID_FG, redraw=False)
            elif check == "sorted":
                self._validate_sorted_dates(iid, m, r)

        sh.redraw()
        if self.on_validity_change:
            self.on_validity_change()

    def _validate_sorted_dates(self, iid: Any, m: Mapping[str, Any], r: int) -> None:
        """Red fg on date cells of a ``check: "sorted"`` row when not ascending.

        ``input.time_ranges`` / ``metadata.time_range`` hold ordered ISO dates;
        a cell breaking the sequence against its predecessor or successor gets
        :data:`~tcm_gui.theme.INVALID_FG`.  Sorted cells restore gray (at
        default) / warning (broader input window — keeps the tint
        :meth:`SheetTintMixin._apply_time_ranges_tint` painted before this) /
        normal fg.  Unparseable cells are skipped — no verdict either way.
        """
        sh = self.sh
        # Collect (col, str, datetime) — non-empty, parseable, non-sentinel cells
        parsed: list[tuple[int, str, datetime]] = []
        for c in range(self._own_cols(m)):
            s = self._cell_str(iid, c)
            if not s or s.startswith("<") or (dt := as_date(s)) is None:
                continue
            parsed.append((c, s, dt))
        if not parsed:
            return

        # input.time_ranges — keep the broader-window warning tint on sorted cells
        broader = (
            not m.get("is_metadata")
            and m.get("label") == "time_ranges"
            and self._time_ranges_relation()[0] == "broader"
        )

        for i, (c, s, dt) in enumerate(parsed):
            is_unsorted = (i > 0 and dt < parsed[i - 1][2]) or (i < len(parsed) - 1 and dt > parsed[i + 1][2])
            if is_unsorted:
                sh.highlight_cells(row=r, column=c, fg=tcm_gui.theme.INVALID_FG, redraw=False)
            else:
                dv = self._default_for_cell(iid, m, c)
                if dv is not NO_DEFAULT and any2str(s) == any2str(dv):
                    fg = tcm_gui.theme.CELL_DEFAULT_VAL_FG
                elif broader:
                    fg = tcm_gui.theme.TAG_COLORS["warning"]
                else:
                    fg = self._fg_default
                sh.highlight_cells(row=r, column=c, fg=fg, redraw=False)
