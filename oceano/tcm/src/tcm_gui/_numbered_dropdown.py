"""Numbered, editable path dropdown for a tksheet cell with external number label.

The cell stores only the actual path::

    C:\\data\\file.nc

The ordinal number is displayed in an existing label (e.g. the ``path`` caption
in ``App._build``) as ``selected/total``::

    2/3 | C:\\data\\file.nc

The dropdown itself displays::

    1. C:\\data\\first.nc
    2. C:\\data\\file.nc
    3. C:\\data\\third.nc

The list is tksheet's inherent dropdown with the overflow patch (see
:mod:`tcm_gui._dropdown_overflow`): the path field is a 1×1 sheet ~18 px
tall and tksheet embeds the list inside the canvas viewport — the measured
list window is 1 px tall. The patch places the same ``Dropdown`` widget in
the toplevel below the field instead. Openers: arrow click (when reachable),
path-caption click (toggles the list) and Up/Down on the cell (``PathField``).

The user can either select a predefined path, edit the current path manually,
or type a completely new path. A manually entered value not present in
``paths`` restores the label default text.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from typing import Any

from ._dropdown_overflow import enable_dropdown_overflow


class NumberedPathDropdown:
    """Attach a numbered dropdown to one tksheet cell, reusing an external label.

    :param sheet: tksheet.Sheet containing the cell.
    :param row: data row of the target cell.
    :param column: data column of the target cell.
    :param paths: dropdown paths; position determines the displayed number.
    :param number_label: existing label widget showing ``selected/total`` (has
        ``configure(text=...)`` — e.g. ``ttk.Label`` from ``App._build``).
    :param default_text: label text when the cell value is not in ``paths``.
    :param on_select: optional ``callback(path)`` after dropdown selection
        (e.g. rescan the picked anchor). Cell data + number are already
        updated before the callback runs.
    """

    def __init__(
        self,
        sheet,
        row: int,
        column: int,
        paths: list[str],
        *,
        number_label,
        default_text: str = "",
        on_select: Callable[[str], None] | None = None,
    ) -> None:
        self.sheet = sheet
        self.row = row
        self.column = column
        self.paths = list(paths)
        self.number_by_path = {path: index for index, path in enumerate(self.paths, start=1)}
        self.number_label = number_label
        self.default_text = default_text
        self.on_select = on_select
        # The list must escape the 1-row canvas — patch once per sheet.
        enable_dropdown_overflow(sheet)
        # `extra_bindings` is a single slot — chain a previously registered
        # `end_edit_cell` handler (e.g. `PathField._on_end_edit`) instead of
        # silently dropping it.
        self._prior_end_edit = getattr(sheet.MT, "extra_end_edit_cell_func", None)
        if self.paths:
            self._attach_dropdown()
        # `end_edit_cell` fires before the edited value is committed, so
        # `event.value` is the about-to-be-stored value. Covers ordinary text
        # editing; programmatic `set()` callers must call `refresh()`.
        sheet.extra_bindings("end_edit_cell", self._end_edit_cell)
        self._update_number()

    def set_paths(self, paths: list[str]) -> None:
        """Replace the dropdown list (e.g. on parent ``scan_list``) and refresh."""
        self.paths = list(paths)
        self.number_by_path = {path: index for index, path in enumerate(self.paths, start=1)}
        if self.paths:
            self._attach_dropdown()
        else:
            with suppress(Exception):  # no anchors — the arrow must offer nothing stale
                self.sheet.del_dropdown(self.row, self.column)
        self._update_number()

    def refresh(self, value: str | None = None) -> None:
        """Re-evaluate the number from the current cell (call after programmatic `set`)."""
        self._update_number(value)

    def _attach_dropdown(self) -> None:
        """(Re)create the tksheet dropdown with numbered display values."""
        self.sheet.dropdown(
            self.row,
            self.column,
            values=self._dropdown_values(),
            # Display-only: attaching must not rewrite the cell with the
            # first numbered value — it keeps the bare path.
            edit_data=False,
            state="normal",
            validate_input=False,
            selection_function=self._dropdown_selected,
            modified_function=self._dropdown_modified,
        )

    def _dropdown_values(self) -> list[str]:
        """Display strings; the cell itself keeps the unnumbered path."""
        return [f"{index}. {path}" for index, path in enumerate(self.paths, start=1)]

    def _dropdown_selected(self, event: Any) -> None:
        """Strip the ordinal prefix, store the real path, update number, notify."""
        displayed = str(event.value)
        _, separator, path = displayed.partition(". ")
        if not separator:
            path = displayed
        # tksheet commits `event.value` into the cell after this returns —
        # hand it the bare path, or the cell ends up numbered again.
        event["value"] = path
        # Redraw now: the pick must be visible immediately, not after the
        # next hover-triggered repaint.
        self.sheet.set_cell_data(self.row, self.column, path, redraw=True)
        self._update_number(path)
        if self.on_select is not None:
            self.on_select(path)

    def _dropdown_modified(self, event: Any) -> None:
        """Typing into the editable dropdown editor — sync happens in `_end_edit_cell`."""
        return

    def _end_edit_cell(self, event: Any) -> None:
        """Update the ordinal after normal cell editing (guard to our cell)."""
        if event.row != self.row or event.column != self.column:
            return
        self._update_number(str(event.value))
        if self._prior_end_edit is not None:
            self._prior_end_edit(event)

    def _update_number(self, value: str | None = None) -> None:
        """Show ``selected/total`` for predefined paths, else the label default text."""
        if value is None:
            try:
                value = self.sheet.get_cell_data(self.row, self.column)
            except Exception:
                value = ""
        number = self.number_by_path.get(str(value))
        try:
            if number is None:
                self.number_label.configure(text=self.default_text)
            else:
                self.number_label.configure(text=f"{number}/{len(self.paths)}")
        except Exception:
            pass
