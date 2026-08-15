"""Path field as a 1×1 tksheet — the full cell contract outside the tree.

Drop-in replacement for the top path Entry: visually a field (headers,
index and grid hidden, entry-colored background, shrink-wrapped to one
row), contractually a cell — double-click / keypress edit, Enter commit,
Esc undo.  Display uses tksheet: left-aligned by default; on hover the
tksheet widget shrinks to ``viewport − button`` (same
``_field_place_kw`` pattern as ConfigSheet's floated field) and
switches to right-aligned scroll-to-filename.  Editing uses a plain
``ttk.Entry`` overlaid on the cell — native single-line scroll, no
wrapping, cursor always visible.

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

import tcm_gui.theme

from ._browse_button import (
    COEF_FILETYPES,
    BrowseOverlay,
    SheetHoverBinder,
    _is_shift_pressed,
    browse_button_width,
)
from ._help import help_for_path
from ._i18n import STRINGS as _S
from ._placeholder import CellPlaceholder

_COL_W = 4096  # wider than any viewport — scrollable when right-aligned


class PathField(ttk.Frame):
    """A single-cell tksheet posing as the path field.

    Display: left-aligned (``"w"``) by default; on hover the tksheet
    widget is shrunk to ``frame − button`` (``place(width=…)``) and
    switches to right-aligned — filename ends right before the browse
    button, same geometry as ConfigSheet's ``_field_place_kw``.
    Editing uses a plain ``ttk.Entry`` overlay with ``justify="right"``.

    When the cell is empty, a dim-gray *placeholder* text is shown
    (e.g. ``"D:/data/_raw/"``) — it vanishes on first keystroke or
    double-click.  Holding Shift swaps to the *advanced* placeholder
    (e.g. ``"D:/data/_raw/(i*raw_file1[.]txt|i*raw_file2[.]txt)"``)
    revealing pattern syntax; releasing Shift restores the simple one.

    Parameters
    ----------
    align : ``"w"`` | ``"e"``
        Initial cell alignment.  ``"w"`` (default) for standalone
        fields; ``"e"`` for floated fields managed by ConfigSheet.
    """

    _MIN_W = 50  # minimum tksheet width during hover

    def __init__(
        self,
        parent,
        *,
        align: str = "w",
        on_commit: Callable[[str], None] | None = None,
        on_begin_edit: Callable[[], None] | None = None,
        on_end_edit: Callable[[], None] | None = None,
        dir_title: str = _S["dialog.data_dir"],
        files_title: str = _S["dialog.data_files"],
        filetypes=COEF_FILETYPES,
        on_status: Callable[[str], None] | None = None,
        status_hint: str | Callable[[], str] = "",
        status_hint_files: str | Callable[[], str] = "",
        on_shift: Callable[[bool], None] | None = None,
        placeholder: str = _S.get("path_field.placeholder", ""),
        placeholder_shift: str = _S.get("path_field.placeholder_shift", ""),
        shift_swap: bool = True,  # False for floated fields in ConfigSheet
    ) -> None:
        super().__init__(parent)
        self._on_commit = on_commit
        self._on_begin_edit_cb = on_begin_edit
        self._on_end_edit_cb = on_end_edit
        self._pre_edit = ""
        self._editing = False
        self._entry: ttk.Entry | None = None
        self._hovering = False
        self._btn_w: int | None = None
        # Hover status: doc-driven (config_reference.md `input.path` search mode
        # short lines); fall back to str.yaml when the doc is absent.
        _h = help_for_path("input.path", mode="search")
        self._path_status = (
            _h.body if _h and isinstance(_h.body, str) and _h.body else _S.get("path_field.status", "")
        )
        self._shift_status = _S.get("path_field.status_shift", "")
        self._browse_hint = status_hint
        self._browse_hint_files = status_hint_files

        self._mouse_in = False  # True when pointer is inside PathField frame
        self._shift_swap = shift_swap
        # Status callback — shared with BrowseOverlay; also called by Shift handlers.
        # Two dedicated keys: normal status and Shift-held status (no separator parsing).
        self._on_status = on_status
        self._path_status_shift = _S.get("path_field.status_shift", self._path_status)
        # Placeholder state — two levels: simple (default) and advanced (Shift held).
        self._placeholder_simple = placeholder
        self._placeholder_shift = placeholder_shift or placeholder
        self._ph = CellPlaceholder()
        self._dim_fg = tcm_gui.theme.DEFAULT_FG






        bg = tcm_gui.theme.ENTRY_BG_FALLBACK
        fg = tcm_gui.theme.FG_DEFAULT
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
            default_column_width=_COL_W,
            empty_vertical=0,
            empty_horizontal=0,
            cell_auto_resize_enabled=False,
            align=align,
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
        # Shift key swaps placeholder: simple ↔ advanced pattern hint.
        # Bind on toplevel — tksheet canvas has no keyboard focus on hover.
        # Swap only fires when pointer is inside the PathField frame.
        # Skipped for floated fields (shift_swap=False).
        if self._shift_swap:
            root = self.winfo_toplevel()
            root.bind("<KeyPress-Shift_L>", self._on_shift_press, add="+")
            root.bind("<KeyPress-Shift_R>", self._on_shift_press, add="+")
            root.bind("<KeyRelease-Shift_L>", self._on_shift_release, add="+")
            root.bind("<KeyRelease-Shift_R>", self._on_shift_release, add="+")
            self.bind("<Enter>", self._on_frame_enter, add="+")
            self.bind("<Leave>", self._on_frame_leave, add="+")
        self._ov = BrowseOverlay(
            self.winfo_toplevel(),
            self._hover_write,
            self._hover_read,
            filetypes=filetypes,
            dir_title=dir_title,
            files_title=files_title,
            leave_hides=True,
            on_status=on_status,
            status_hint=status_hint,
            status_hint_files=status_hint_files,
            on_shift=on_shift,
        )
        # Patch overlay to shrink/restore tksheet widget on show/hide
        self._orig_ss = self._ov.schedule_show
        self._ov.schedule_show = self._patched_schedule_show
        self._orig_hide = self._ov.hide
        self._ov.hide = self._patched_hide
        self._binder = SheetHoverBinder(self.sh, self._ov, lambda _e: self._place_kw())
        self.bind("<Configure>", self._on_configure, add="+")
        self.pack_propagate(False)
        self.sh.pack(fill="both", expand=True)
        with suppress(AttributeError, TclError):
            self.configure(height=self.sh.MT.row_positions[1])
        # Show placeholder if no initial value was set.
        if self._placeholder_simple:
            self._show_placeholder()

    # ── hover widget shrink ───────────────────────────────────────
    def _get_btn_w(self) -> int:
        """Pixel width of the browse button (cached)."""
        if self._btn_w is None:
            self._btn_w = browse_button_width(self.sh)
        return self._btn_w

    def _hover_shrink_width(self) -> int:
        """Constrained tksheet width: frame minus button."""
        return max(self.winfo_width() - self._get_btn_w(), self._MIN_W)

    def _switch_to_hover_shrink(self) -> None:
        """Shrink tksheet widget + right-align + scroll-to-filename.

        Mirrors ``ConfigSheet._field_place_kw``: the field width ends
        where the browse button starts.
        """
        if self._hovering:
            return
        self._hovering = True
        self.sh.table_align("e", redraw=False)
        self.sh.pack_forget()
        self.sh.place(relx=0, rely=0, relheight=1, width=self._hover_shrink_width())
        self.update_idletasks()  # force geometry so scroll targets the new viewport
        self.sh.redraw()  # explicit render with right-align at new width
        self._scroll_to_right()

    def _restore_default_layout(self) -> None:
        """Restore left-align + full-width tksheet (pack)."""
        if not self._hovering:
            return
        self._hovering = False
        self.sh.place_forget()
        self.sh.pack(fill="both", expand=True)
        self.sh.table_align("w", redraw=False)
        self.update_idletasks()  # force geometry so subsequent redraw renders correctly
        self.sh.redraw()  # explicit render — Configure alone may not suffice
        self._scroll_to_left()  # reset viewport — was at xview_moveto(1.0) from hover

    def _patched_schedule_show(self, *a: Any, **kw: Any) -> None:
        self._switch_to_hover_shrink()
        self._orig_ss(*a, **kw)

    def _patched_hide(self) -> None:
        self._restore_default_layout()
        self._orig_hide()

    def _on_configure(self, _event) -> None:
        if self._editing:
            return
        if self._hovering:
            # viewport resized — re-constrain tksheet width
            self.sh.place(width=self._hover_shrink_width())
            self.update_idletasks()  # force geometry before scroll
            self._scroll_to_right()
        elif self.sh.MT.align == "ne":
            # right-aligned non-hover (floated field) — keep scrolled
            self._scroll_to_right()
        else:
            # left-align default — ensure viewport at left
            self._scroll_to_left()

    def _scroll_to_right(self) -> None:
        with suppress(TclError):
            self.sh.MT.xview_moveto(1.0)

    def _scroll_to_left(self) -> None:
        with suppress(TclError):
            self.sh.MT.xview_moveto(0.0)

    # ── placeholder ───────────────────────────────────────────────
    def _show_placeholder(self) -> None:
        """Render the placeholder text dimmed in the empty cell."""
        self._ph.show(self.sh, 0, 0, self._placeholder_simple, self._dim_fg)

    def _clear_placeholder(self) -> None:
        """Remove placeholder text and restore default foreground."""
        self._ph.clear(self.sh, 0, 0)

    def _swap_placeholder(self, text: str) -> None:
        """Swap displayed placeholder text (simple ↔ Shift variant) without changing tracking.

        Delegates to :meth:`CellPlaceholder.show` — the cell is already tracked
        (callers guard on ``self._ph.active``), so the re-add to ``_cells`` is
        an idempotent no-op.
        """
        self._ph.show(self.sh, 0, 0, text, self._dim_fg)
        self.sh.redraw()

    def _on_shift_press(self, _event) -> None:
        """Swap to advanced placeholder + YAML status while Shift is held and cell is empty.

        Only fires when the pointer is inside this PathField (``_mouse_in``) —
        prevents the search-path placeholder from visually changing while the
        user Shift-clicks elsewhere in the GUI.
        """
        if self._mouse_in and self._ph.active and not self._editing:
            self._swap_placeholder(self._placeholder_shift)
            if self._on_status:
                self._on_status(self._path_status_shift)

    def _on_shift_release(self, _event) -> None:
        """Restore simple placeholder + normal status when Shift is released.

        Same ``_mouse_in`` guard as :meth:`_on_shift_press`.
        """
        if self._mouse_in and self._ph.active and not self._editing:
            self._swap_placeholder(self._placeholder_simple)
            if self._on_status:
                self._on_status(self._path_status)

    def _on_frame_enter(self, _event) -> None:
        """Mouse entered PathField — if Shift already held, swap immediately."""
        self._mouse_in = True
        if self._ph.active and not self._editing and _is_shift_pressed():
            self._swap_placeholder(self._placeholder_shift)
            if self._on_status:
                self._on_status(self._path_status_shift)

    def _on_frame_leave(self, _event) -> None:
        """Mouse left PathField — restore simple placeholder and normal status."""
        self._mouse_in = False
        if self._ph.active and not self._editing:
            self._swap_placeholder(self._placeholder_simple)
            if self._on_status:
                self._on_status(self._path_status)

    # ── Entry-like API ────────────────────────────────────────────
    def get(self) -> str:
        return self._ph.get(self.sh, 0, 0)

    def set(self, value: str) -> None:
        if not value and self._placeholder_simple:
            self._show_placeholder()
        else:
            self._clear_placeholder()
            self.sh.set_cell_data(0, 0, value)
        self.sh.redraw()
        if self.sh.MT.align == "ne":
            self._scroll_to_right()
        else:
            self._scroll_to_left()

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
        # Clear placeholder so the user starts with an empty field.
        if self._ph.active:
            self._clear_placeholder()
            self.sh.redraw()
            self._pre_edit = ""
        else:
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
        # On cancel restore pre-edit value; on commit use the Entry value.
        new_val = self._pre_edit if cancel else val
        if new_val:
            self.sh.set_cell_data(0, 0, new_val)
        elif self._placeholder_simple:
            self._show_placeholder()
        self.sh.redraw()
        if self.sh.MT.align == "ne":
            self._scroll_to_right()
        else:
            self._scroll_to_left()
        if not cancel and val != self._pre_edit:
            self._notify(val)

    def _on_end_edit(self, event) -> None:
        """tksheet end-edit — no-op when Entry handles editing."""
        if self._entry is not None:
            return  # Entry is in charge
        self._editing = False

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
