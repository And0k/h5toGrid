"""Path field as a 1×1 tksheet — the full cell contract outside the tree.

Drop-in replacement for the top path Entry: visually a field (headers,
index and grid hidden, entry-colored background, shrink-wrapped to one
row), contractually a cell — double-click / keypress edit, Enter commit,
Esc undo.  Display uses tksheet (right-aligned, scroll-to-filename).
Editing uses a plain ``ttk.Entry`` overlaid on the cell — native
single-line scroll, no wrapping, cursor always visible.

Reuses the floating-button stack: ``BrowseButtonManager`` (editor-
anchored, during edit) and ``BrowseOverlay`` + ``SheetHoverBinder``
(cell-anchored, on hover).
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from tkinter import TclError, ttk
from typing import Any

from tksheet import Sheet

from . import const
from ._browse_button import BrowseOverlay, SheetHoverBinder


class PathField(ttk.Frame):
    """A single-cell tksheet posing as the path field.

    Display: column=4096, ``align="e"``, ``xview_moveto(1.0)`` — filename
    visible at the right edge.

    Edit: a ``ttk.Entry`` is placed over the cell.  The Entry is the
    native Tk single-line text widget — horizontal scroll, cursor
    tracking, Home/End, selection — all built-in, no wrapping.
    """

    def __init__(
        self,
        parent,
        *,
        on_commit: Callable[[str], None] | None = None,
        on_begin_edit: Callable[[], None] | None = None,
        on_end_edit: Callable[[], None] | None = None,
        dir_title: str = "Browse data path",
        files_title: str = "Browse data files",
    ) -> None:
        super().__init__(parent)
        self._on_commit = on_commit
        self._on_begin_edit_cb = on_begin_edit
        self._on_end_edit_cb = on_end_edit
        self._pre_edit = ""
        self._editing = False
        self._entry: ttk.Entry | None = None
        bg = const.ENTRY_BG_FALLBACK
        fg = const.FG_DEFAULT
        self.sh = Sheet(
            self,
            total_rows=1,
            total_columns=1,
            show_header=False,
            show_row_index=False,
            show_top_left=False,
            show_x_scrollbar=False,
            show_y_scrollbar=False,
            show_horizontal_grid=False,
            show_vertical_grid=False,
            show_selected_cells_border=False,
            startup_focus=False,
            default_column_width=4096,
            empty_vertical=0,
            empty_horizontal=0,
            cell_auto_resize_enabled=False,
            align="e",
            table_wrap="",
            edit_cell_return="",
            table_bg=bg,
            table_fg=fg,
            table_editor_bg=bg,
            table_editor_fg=fg,
            table_selected_cells_bg=bg,
            table_selected_cells_fg=fg,
            table_selected_cells_border_fg=bg,
            outline_thickness=0,
        )
        self.sh.set_options(table_wrap="")
        self.sh.MT.config(highlightthickness=0, cursor="xterm")
        cur = self.sh.font()
        self.sh.font((cur[0], cur[1], "bold"))
        self.sh.pack(fill="x")

        self.sh.enable_bindings(["all"])
        # Intercept double-click → open Entry editor instead of tksheet's tk.Text
        self.sh.extra_bindings(
            [
                ("begin_edit_cell", self._on_begin_edit),
                ("end_edit_cell", self._on_end_edit),
            ]
        )
        self._ov = BrowseOverlay(
            self.winfo_toplevel(),
            self._hover_write,
            self._hover_read,
            dir_title=dir_title,
            files_title=files_title,
            leave_hides=True,
        )
        self._binder = SheetHoverBinder(self.sh, self._ov, lambda _e: self._place_kw())
        self.bind("<Configure>", self._on_configure, add="+")
        self.pack_propagate(False)
        self.sh.pack(fill="both", expand=True)
        with suppress(AttributeError, TclError):
            self.configure(height=self.sh.MT.row_positions[1])

    def _scroll_to_right(self) -> None:
        with suppress(TclError):
            self.sh.MT.xview_moveto(1.0)

    def _on_configure(self, _event) -> None:
        if not self._editing:
            self._scroll_to_right()

    # ── Entry-like API ────────────────────────────────────────────
    def get(self) -> str:
        return str(self.sh.get_cell_data(0, 0) or "")

    def set(self, value: str) -> None:
        self.sh.set_cell_data(0, 0, value)
        self.sh.redraw()
        self._scroll_to_right()

    def cancel_edit(self) -> None:
        """Close the Entry editor if open."""
        if self._entry is not None:
            self._commit_entry(cancel=True)

    # ── edit lifecycle (Entry overlay) ─────────────────────────────
    def _on_begin_edit(self, event) -> str | None:
        """Open a ``ttk.Entry`` over the cell instead of tksheet's editor.

        Returns ``None`` to veto tksheet's built-in editor — we handle
        editing entirely through the Entry.
        """
        self._editing = True
        self._pre_edit = self.get()
        self._ov.hide()
        if self._on_begin_edit_cb is not None:
            self._on_begin_edit_cb()
        self._open_entry()
        return None  # veto tksheet's tk.Text editor

    def _open_entry(self) -> None:
        """Create and place a ``ttk.Entry`` filling the PathField frame."""
        bold = self.sh.font()
        ent = ttk.Entry(self, font=bold, justify="right")
        ent.insert(0, self._pre_edit)
        ent.icursor("end")
        ent.xview_moveto(1.0)
        ent.place(relx=0, rely=0, relwidth=1, relheight=1)
        ent.bind("<Return>", lambda _e: self._commit_entry())
        ent.bind("<Escape>", lambda _e: self._commit_entry(cancel=True))
        ent.focus_set()
        self._entry = ent

    def _commit_entry(self, *, cancel: bool = False) -> None:
        """Read Entry value, update cell, destroy Entry."""
        ent, self._entry = self._entry, None
        if ent is None:
            return
        val = ent.get()
        with suppress(TclError):
            ent.destroy()
        self._editing = False
        if self._on_end_edit_cb is not None:
            self._on_end_edit_cb()
        if not cancel:
            self.sh.set_cell_data(0, 0, val)
        self.sh.redraw()
        self._scroll_to_right()
        if not cancel and val != self._pre_edit:
            self._notify(val)

    def _on_end_edit(self, event) -> None:
        """tksheet end-edit — no-op when Entry handles editing."""
        if self._entry is not None:
            return  # Entry is in charge
        self._editing = False
        self._scroll_to_right()

    def _notify(self, value: str) -> None:
        if self._on_commit is not None and value.strip():
            self.after_idle(lambda: self._on_commit(value))

    # ── hover policy ──────────────────────────────────────────────
    def _place_kw(self) -> dict[str, Any]:
        return {"in_": self, "relx": 1.0, "rely": 0.0, "x": -2, "anchor": "ne"}

    def _hover_write(self, text: str) -> None:
        self.set(text)
        self._notify(text)

    def _hover_read(self) -> str:
        return self.get()
