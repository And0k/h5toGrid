"""Reusable dim-text placeholder for empty tksheet cells.

A placeholder shows a dim-format hint (e.g. ``YYYY-MM-DD``) in an **empty**
cell so the user knows what to type.  The text is written into the cell and
styled with a dim foreground — tksheet has no native placeholder concept.

Tracking which cells carry a placeholder avoids:
- treating the hint as real user data on read-back (``get`` returns ``""``)
- the hint surviving into the YAML save when the cell is committed empty

Usage (single-cell — PathField)::

    self._ph = CellPlaceholder()
    self._ph.show(self.sh, 0, 0, "D:/data/_raw/", dim_fg)
    ...
    val = self._ph.get(self.sh, 0, 0)   # "" while placeholder is active

Usage (multi-cell — ConfigSheet date columns)::

    self._ph = CellPlaceholder()
    for iid, m in self._meta.items():
        if m.get("has_date") and not real_date:
            self._ph.show(self.sh, row, _DATE_COL - DATA_COL_BASE,
                          "YYYY-MM-DD", dim_fg)
    ...
    dates = {
        m["key"]: self._ph.get(self.sh, row, date_col) or ""
        for iid, m in self._meta.items() if m.get("has_date")
    }
"""

from __future__ import annotations

from tkinter import TclError
from typing import Any

from contextlib import suppress

from tksheet import Sheet


class CellPlaceholder:
    """Track dim-placeholder text in empty tksheet cells.

    Stores a ``set[tuple[int, int]]`` of ``(row, col)`` positions that
    currently display a placeholder.  Methods are no-ops when the cell is
    not in the set (idempotent clear / re-show).
    """

    __slots__ = ("_cells",)

    def __init__(self) -> None:
        self._cells: set[tuple[int, int]] = set()

    # ── queries ───────────────────────────────────────────────────

    @property
    def active(self) -> bool:
        """True when at least one cell carries a placeholder."""
        return bool(self._cells)

    def has(self, row: int, col: int) -> bool:
        """True when ``(row, col)`` currently shows a placeholder."""
        return (row, col) in self._cells

    def get(self, sheet: Sheet, row: int, col: int) -> str:
        """Return cell value, or ``""`` when a placeholder is active.

        Use this instead of ``sheet.get_cell_data`` for data extraction
        so the dim hint never leaks into saved output.
        """
        if (row, col) in self._cells:
            return ""
        with suppress(TclError, IndexError):
            return str(sheet.get_cell_data(row, col) or "")
        return ""

    # ── mutations ─────────────────────────────────────────────────

    def show(self, sheet: Sheet, row: int, col: int, text: str, dim_fg: str) -> None:
        """Write *text* dimmed into ``(row, col)`` and mark it as placeholder.

        If the cell already holds real data, the call is a no-op (caller
        decides when a cell is *empty enough* to deserve a hint).
        """
        with suppress(TclError):
            sheet.set_cell_data(row, col, text, redraw=False)
            sheet.highlight_cells(row=row, column=col, fg=dim_fg, canvas="table", redraw=False)
        self._cells.add((row, col))

    def clear(self, sheet: Sheet, row: int, col: int, *, redraw: bool = False) -> None:
        """Remove placeholder: blank the cell and restore default fg.

        No-op when ``(row, col)`` is not a placeholder — safe to call
        unconditionally in ``begin_edit_cell`` hooks.
        """
        if (row, col) not in self._cells:
            return
        self._cells.discard((row, col))
        with suppress(TclError):
            sheet.dehighlight_cells(row=row, column=col, canvas="table", redraw=redraw)
            sheet.set_cell_data(row, col, "", redraw=redraw)

    def untrack(self, sheet: Sheet, row: int, col: int) -> None:
        """Drop placeholder tracking for ``(row, col)`` WITHOUT blanking the cell.

        Use when real data has been written into a previously-placeholdered
        cell (e.g. a hovered-field commit): :meth:`clear` would wipe the
        fresh value, here only the dim highlight is removed.
        """
        if (row, col) not in self._cells:
            return
        self._cells.discard((row, col))
        with suppress(TclError):
            sheet.dehighlight_cells(row=row, column=col, canvas="table", redraw=False)

    def restore(
        self, sheet: Sheet, row: int, col: int, text: str, dim_fg: str
    ) -> None:
        """Re-show placeholder after an edit committed an empty value.

        Reads the current cell content; if non-empty, does nothing (the
        user typed a real value).  If empty, calls :meth:`show`.
        """
        if (row, col) in self._cells:
            return  # already a placeholder
        with suppress(TclError, IndexError):
            val = str(sheet.get_cell_data(row, col) or "").strip()
            if not val:
                self.show(sheet, row, col, text, dim_fg)

    def clear_all(self, sheet: Sheet) -> None:
        """Remove all placeholders (e.g. before a full sheet rebuild)."""
        for row, col in list(self._cells):
            self.clear(sheet, row, col)
        self._cells.clear()
