"""Instance-level overflow patch for tksheet dropdowns on tiny sheets.

tksheet creates its ``Dropdown`` as a child of the application toplevel but
embeds it into ``MainTable`` with ``Canvas.create_window()`` — a canvas clips
child windows to its own rectangle. On the 1×1 path field (~18 px tall) the
measured list window is 1 px tall: the list opens but is permanently
invisible.

This patch keeps tksheet's ``Dropdown`` implementation completely intact and
only replaces the canvas embedding with ``place()`` in the sheet's toplevel,
so the list can extend below the sheet. Applied to one ``Sheet`` instance —
the installed tksheet package is untouched.

Explicit geometry rules (deviations from tksheet, caused by canvas space
being meaningless outside the sheet):

- the dropdown is not clipped by the ``Sheet`` (always opens downward);
- width is ``max(visible cell width, widest displayed item + padding)``,
  capped at the toplevel (window) width — the *visible* (not canvas) cell
  width is the floor, since a 4096 px column would otherwise span screens
  (for ordinary columns the two coincide);
- dropdown text is always left-aligned, independently of cell alignment;
- height fits **all** items (not tksheet's first-six / 500 px cap);
- height is capped by the space from the dropdown top to the **toplevel
  (window) bottom** — a ``place()`` child is clipped by its window, so
  screen room is unusable; a short list shows whole, a tall one reaches
  the window edge and scrolls internally;
- the same geometry is recalculated after zoom/resize/editing.
"""

from __future__ import annotations

import tkinter.font as tkfont
import types
from contextlib import suppress

import tksheet

# List-navigation keys: releases carry no text change, so filter research
# must skip them (else every release re-selects the editor-text match and
# arrow steps could never leave it).
_NAV_KEYS = ("Up", "Prior", "Down", "Next")
_NAV_STEPS = (("<Up>", -1), ("<Prior>", -1), ("<Down>", +1), ("<Next>", +1))


def _step_open_list(mt, delta: int) -> str | None:
    """Move the open list highlight (combobox arrows); ``None`` when closed.

    Highlight-only, like mouse hover — commit paths are untouched, so
    Return still commits the editor text (custom paths keep working).
    ``"break"`` keeps the editor cursor put while the list is open.
    Module-level (not a patched method) so tests can drive it directly —
    synthetic keys don't reach these widgets under a test runner.
    """
    win = mt.dropdown.window
    if not mt.dropdown.open or win is None:
        return None
    win.arrowkey_UP() if delta < 0 else win.arrowkey_DOWN()
    return "break"


def _dropdown_editor_key_release(mt, event) -> str | None:
    """tksheet filter research, except navigation-key releases (no text changed)."""
    if getattr(event, "keysym", "") in _NAV_KEYS:
        return None
    return mt.dropdown_text_editor_modified(event)


def enable_dropdown_overflow(sheet: tksheet.Sheet) -> None:
    """Make one sheet's dropdowns capable of extending outside the widget.

    Idempotent — re-applying to the same sheet is a no-op. The normal
    tksheet open path (arrow click, ``open_cell``) needs no extra wiring.
    The underlying tksheet ``Dropdown`` is retained, so search, keyboard
    navigation, editing, selection callbacks and styling keep working.
    """
    mt = sheet.MT

    # Keep the original methods available in case the patch needs to be
    # disabled or inspected later.
    if hasattr(mt, "_overflow_dropdown_originals"):
        return

    mt._overflow_dropdown_originals = {
        "open_dropdown_window": mt.open_dropdown_window,
        "hide_dropdown_window": mt.hide_dropdown_window,
        "refresh_open_window_positions": mt.refresh_open_window_positions,
        "text_editor_newline_binding": mt.text_editor_newline_binding,
    }

    def dropdown_values(self, r: int, c: int) -> list:
        """Current dropdown values for a displayed cell."""
        return self.get_cell_kwargs(self.datarn(r), self.datacn(c), key="dropdown")["values"]

    def dropdown_content_width(self, values: list) -> int:
        """Pixel width of the widest displayed item + padding/scrollbar/border room."""
        font = tkfont.Font(root=self.winfo_toplevel(), font=self.PAR.ops.table_font)
        return max((font.measure(str(v)) for v in values), default=0) + 40

    def dropdown_content_height(self, values: list) -> int:
        """Height for every dropdown item — deliberately no six-item / 500 px cap."""
        height = 5
        for value in (str(v) for v in values):
            if (lines := len(value.split("\n"))) > 1:
                height += 8 + lines * self.table_txt_height
            else:
                height += self.min_row_height
        return max(height, self.table_txt_height + 5)

    def dropdown_popup_geometry(
        self, r: int, c: int, *, text_editor_height: int = 0
    ) -> tuple[int, int, int, int]:
        """``(x, y, width, height)`` relative to the toplevel; downward, window-capped."""
        toplevel = self.winfo_toplevel()
        values = dropdown_values(self, r, c)
        # Visible left edge of the cell in toplevel coordinates (scroll-aware,
        # floored at the sheet edge — the popup aligns under the field, then
        # slides left if content width would clip it at the window edge).
        tl_w = toplevel.winfo_width()
        x = self.winfo_rootx() - toplevel.winfo_rootx() + max(self.col_positions[c] - self.canvasx(0), 0)
        cell_width = self.col_positions[c + 1] - self.col_positions[c] + 1
        vw = self.winfo_width()
        width = max(min(cell_width, vw) if vw > 1 else cell_width, dropdown_content_width(self, values))
        width = min(width, self.winfo_screenwidth(), tl_w if tl_w > 1 else width)
        x = min(max(x, 0), max(tl_w - width, 0))
        y = (
            self.winfo_rooty()
            - toplevel.winfo_rooty()
            + (self.row_positions[r] - self.canvasy(0))
            + text_editor_height
            - 1
        )
        available = toplevel.winfo_height() - y - 2
        height = max(self.table_txt_height + 5, min(dropdown_content_height(self, values), max(1, available)))
        return (int(x), int(y), int(width), int(height))

    def position_dropdown(self, r: int, c: int, *, text_editor_height: int = 0) -> None:
        """Position the existing Dropdown widget directly in the toplevel."""
        if not self.dropdown.window:
            return
        x, y, width, height = dropdown_popup_geometry(self, r, c, text_editor_height=text_editor_height)
        self.dropdown.window.place(x=x, y=y, width=width, height=height)
        self.dropdown.window.lift()

    def open_dropdown_window(self, r: int, c: int, event=None) -> None:
        """Replacement for MainTable.open_dropdown_window() — toplevel ``place()`` geometry."""
        self.hide_text_editor()
        datarn = self.datarn(r)
        datacn = self.datacn(c)
        kwargs = self.get_cell_kwargs(datarn, datacn, key="dropdown")
        if kwargs["state"] == "disabled":
            return
        if kwargs["state"] == "normal" and not self.open_text_editor(event=event, r=r, c=c, dropdown=True):
            return
        text_editor_height = 0
        if kwargs["state"] == "normal":
            self.text_editor.window.update_idletasks()
            text_editor_height = self.text_editor.window.winfo_height()
        _x, _y, width, height = dropdown_popup_geometry(self, r, c, text_editor_height=text_editor_height)
        reset_kwargs = {
            "r": r,
            "c": c,
            "bg": self.PAR.ops.table_editor_bg,
            "fg": self.PAR.ops.table_editor_fg,
            "select_bg": self.PAR.ops.table_editor_select_bg,
            "select_fg": self.PAR.ops.table_editor_select_fg,
            "width": width,
            "height": height,
            "font": self.PAR.ops.table_font,
            "ops": self.PAR.ops,
            "outline_color": self.get_selected_box_bg_fg(type_="cells")[1],
            # Deliberately independent of the cell's alignment.
            "align": "w",
            "values": kwargs["values"],
            "search_function": kwargs["search_function"],
            "modified_function": kwargs["modified_function"],
        }
        if self.dropdown.window:
            self.dropdown.window.reset(**reset_kwargs)
        else:
            self.dropdown.window = self.PAR._dropdown_cls(
                self.winfo_toplevel(),
                **reset_kwargs,
                close_dropdown_window=self.close_dropdown_window,
                arrowkey_RIGHT=self.arrowkey_RIGHT,
                arrowkey_LEFT=self.arrowkey_LEFT,
            )
        # The canvas_id is no longer used for the overflow dropdown.
        # Keep it as None so accidental canvas operations fail loudly
        # instead of manipulating an unrelated Canvas item.
        self.dropdown.canvas_id = None
        position_dropdown(self, r, c, text_editor_height=text_editor_height)
        self.update_idletasks()
        # The placed window skipped canvas-embedding Configure sequencing —
        # force row paint now that it has real size (else rows may stay blank
        # until hovered one by one).
        with suppress(Exception):
            self.dropdown.window.MT.main_table_redraw_grid_and_text(True, True, True, True, True)
        if kwargs["state"] == "normal":
            self.text_editor.tktext.bind("<KeyRelease>", lambda e: _dropdown_editor_key_release(self, e))
            for _seq, _delta in _NAV_STEPS:
                self.text_editor.tktext.bind(_seq, lambda e, _d=_delta: _step_open_list(self, _d))
            try:
                self.after(1, lambda: self.text_editor.tktext.focus())
                self.after(2, self.text_editor.window.scroll_to_bottom())
            except Exception:
                return
            redraw = False
        else:
            self.dropdown.window.bind("<FocusOut>", lambda _: self.close_dropdown_window(r, c))
            self.dropdown.window.bind("<Escape>", self.close_dropdown_window)
            self.dropdown.window.focus_set()
            redraw = True
        self.dropdown.open = True
        if redraw:
            self.main_table_redraw_grid_and_text(redraw_header=False, redraw_row_index=False)

    def hide_dropdown_window(self) -> None:
        """Replacement for MainTable.hide_dropdown_window() — ``place_forget()``, widget kept."""
        if self.dropdown.open:
            self.dropdown.window.unbind("<FocusOut>")
            self.dropdown.window.place_forget()
            self.dropdown.open = False

    def refresh_open_window_positions(self, zoom: str) -> None:
        """Replacement for MainTable.refresh_open_window_positions() — recalculated geometry."""
        if self.text_editor.open:
            r, c = self.text_editor.coords
            self.text_editor.window.config(height=self.row_positions[r + 1] - self.row_positions[r])
            self.text_editor.tktext.config(font=self.PAR.ops.table_font)
            self.coords(self.text_editor.canvas_id, self.col_positions[c], self.row_positions[r])
        if not self.dropdown.open:
            return
        if zoom == "in":
            self.dropdown.window.zoom_in()
        elif zoom == "out":
            self.dropdown.window.zoom_out()
        r, c = self.dropdown.get_coords()
        text_editor_height = self.text_editor.window.winfo_height() if self.text_editor.open else 0
        _x, _y, width, height = dropdown_popup_geometry(self, r, c, text_editor_height=text_editor_height)
        self.dropdown.window.config(width=width, height=height)
        position_dropdown(self, r, c, text_editor_height=text_editor_height)

    def text_editor_newline_binding(self, event=None, check_lines: bool = True) -> None:
        """Replacement for the part of tksheet's newline handler that repositions a dropdown."""
        r, c = self.text_editor.coords
        curr_height = self.text_editor.window.winfo_height()
        if curr_height < self.min_row_height:
            return
        if (
            not check_lines
            or self.get_lines_cell_height(
                self.text_editor.window.get_num_lines() + 1,
                font=self.text_editor.tktext.cget("font"),
            )
            > curr_height
        ):
            new_height = min(
                curr_height + self.table_txt_height,
                self.scrollregion[3] - self.scrollregion[1] - self.row_positions[r],
            )
            if new_height == curr_height:
                return
            self.text_editor.window.config(height=new_height)
            if self.dropdown.open and self.dropdown.get_coords() == (r, c):
                text_editor_height = self.text_editor.window.winfo_height()
                _x, _y, width, height = dropdown_popup_geometry(
                    self, r, c, text_editor_height=text_editor_height
                )
                self.dropdown.window.config(width=width, height=height)
                position_dropdown(self, r, c, text_editor_height=text_editor_height)

    # Bind the replacements to this particular MainTable instance.
    mt.open_dropdown_window = types.MethodType(open_dropdown_window, mt)
    mt.hide_dropdown_window = types.MethodType(hide_dropdown_window, mt)
    mt.refresh_open_window_positions = types.MethodType(refresh_open_window_positions, mt)
    mt.text_editor_newline_binding = types.MethodType(text_editor_newline_binding, mt)

    # `open_dropdown_window()` is normally reached through MainTable's
    # existing bindings, so no additional event binding is necessary.
