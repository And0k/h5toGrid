"""Path field as a 1×1 tksheet — the full cell contract outside the tree.

Drop-in replacement for the top path Entry: visually a field (headers,
index and grid hidden, entry-colored background, shrink-wrapped to one
row), contractually a cell — double-click / keypress edit, Enter commit,
Esc undo, all native tksheet.  Reuses the floating-button stack
verbatim: ``BrowseButtonManager`` (editor-anchored, during edit) and
``BrowseOverlay`` + ``SheetHoverBinder`` (cell-anchored, on hover).

Size it from the app's geometry manager; ``empty_vertical=False`` keeps
the canvas at exactly one row.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from tkinter import TclError, ttk
from typing import Any

from tksheet import Sheet

from ._browse_button import BrowseButtonManager, BrowseOverlay, SheetHoverBinder
from . import const

class PathField(ttk.Frame):
    """A single-cell tksheet posing as the path field.

    *on_commit* fires after every value change (edit commit or
    hover-browse write) via ``after_idle`` — the same async contract
    as ``notify_path_changed``.  Esc-cancel is silent by construction:
    the pre-edit snapshot compare swallows the unchanged value.
    """

    def __init__(
        self,
        parent,
        *,
        on_commit: Callable[[str], None] | None = None,
        dir_title: str = "Browse data path",
        files_title: str = "Browse data files",
    ) -> None:
        super().__init__(parent)
        self._on_commit, self._pre_edit = on_commit, ""
        # ── Entry metrics — measured from a throwaway probe, not guessed ──
        style = ttk.Style()
        probe = ttk.Entry(self)
        entry_h = probe.winfo_reqheight()
        probe.destroy()
        bg = const.tk_color_to_hex(
            self,
            style.lookup("TEntry", "fieldbackground") or style.lookup("TEntry", "background") or "#FFFFFF",
        )
        fg = const.BLUE_FG
        self.sh = Sheet(
            self,
            total_rows=1, total_columns=1,
            # ── chrome off — verified ctor kwargs ─────────────────
            show_header=False, show_row_index=False, show_top_left=False,
            show_x_scrollbar=False, show_y_scrollbar=False,
            show_horizontal_grid=False, show_vertical_grid=False,
            show_selected_cells_border=False,
            startup_focus=False,                    # form field: don't steal focus
            # ── Entry silhouette ──────────────────────────────────
            height=entry_h,                         # verified kwarg; no set_height probe
            default_row_height=entry_h,             # int ⇒ pixels; Sheet.default_row_height() exists post-hoc
            default_column_width=4096,              # cell spans field; x-scrollbar off ⇒ clipped, not scrollable
            empty_vertical=0, empty_horizontal=0,   # ints (px), not bools
            cell_auto_resize_enabled=False,         # long path must not stretch the row
            align="w",
            edit_cell_return="",                    # Enter commits; no row travel in a 1-row sheet
            allow_cell_overflow=True,
            # ── Entry face ────────────────────────────────────────
            table_bg=bg, table_fg=fg,               # not table_background — that was silently ignored
            table_editor_bg=bg, table_editor_fg=fg, # seamless edit transition
            table_selected_cells_bg=bg,             # selection paints nothing —
            table_selected_cells_fg=fg,             # keyboard-edit state stays, box invisible
            table_selected_cells_border_fg=bg,
            outline_thickness=0,
        )
        # self.sh.set_options(font=entry_font)        # special-handled → MT.set_table_font; string acceptance to verify
        self.sh.MT.config(highlightthickness=0, cursor="xterm")   # plain Tk — no version surface
        self.sh.pack(fill="x")                      # height is intrinsic now; width app-managed


        self.sh.enable_bindings(["all"])
        self.sh.edit_validation(lambda event: event.value)
        self.sh.extra_bindings([
            ("begin_edit_cell", self._on_begin_edit),
            ("end_edit_cell", self._on_end_edit),
        ])
        # floating-button stack — unchanged
        self._mgr = BrowseButtonManager(self.sh, self._notify)
        self._ov = BrowseOverlay(
            self.sh, self._hover_write, self._hover_read,
            dir_title=dir_title, files_title=files_title, leave_hides=True,
        )
        self._binder = SheetHoverBinder(self.sh, self._ov, lambda _e: self._place_kw())
        # single cell tracks field width — clicks anywhere hit the cell,
        # content never overflows, so scrollbars have nothing to appear for
        self.bind("<Configure>", self._on_resize, add="+")
        # Entry silhouette: self-managed height, app-managed width (fill="x")
        self.configure(height=entry_h)
        self.pack_propagate(False)
        self.grid_propagate(False)
        self.sh.pack(fill="both", expand=True)

    def _on_resize(self, event) -> None:
        with suppress(AttributeError, TclError, TypeError):
            self.sh.column_width(0, max(event.width - 2, 50))

    # ── Entry-like API ────────────────────────────────────────────
    def get(self) -> str:
        return str(self.sh.get_cell_data(0, 0) or "")

    def set(self, value: str) -> None:
        self.sh.set_cell_data(0, 0, value)
        self.sh.redraw()

    # ── edit lifecycle ────────────────────────────────────────────
    def _on_begin_edit(self, event) -> str | None:
        """Hover → edit handoff; snapshot for the Esc-undo compare."""
        self._pre_edit = self.get()
        self._ov.hide()
        self._mgr.detach()
        self._mgr.attach(0, 0)
        return self._pre_edit

    def _on_end_edit(self, event) -> None:
        self._mgr.detach()
        if (v := str(event.value or "")) != self._pre_edit:  # Esc ⇒ unchanged ⇒ silent
            self._notify(v)

    def _notify(self, value: str) -> None:
        if self._on_commit is not None and value.strip():
            self.after_idle(lambda: self._on_commit(value))

    # ── hover policy ──────────────────────────────────────────────
    def _place_kw(self) -> dict[str, Any]:
        """Right edge of the single cell, viewport-adjusted."""
        with suppress(AttributeError, TypeError, IndexError, TclError):
            mt = self.sh.MT
            y1, y2 = mt.row_positions[0], mt.row_positions[1]
            return {"in_": mt, "x": mt.winfo_width(), "y": y1 - mt.canvasy(0), "anchor": "ne", "height": y2 - y1}
        return {"in_": self.sh, "relx": 1.0, "x": 0, "rely": 0, "y": 0, "height": 20}

    def _hover_write(self, text: str) -> None:
        self.set(text)
        self._notify(text)

    def _hover_read(self) -> str:
        return self.get()
