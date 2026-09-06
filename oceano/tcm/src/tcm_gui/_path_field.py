"""Path field as a 1×1 tksheet — the full cell contract outside the tree.

Drop-in replacement for the top path Entry: visually a field (headers,
index and grid hidden, entry-colored background, shrink-wrapped to one
row), contractually a cell — double-click / keypress edit, Enter commit,
Esc undo.  Display uses tksheet: left-aligned by default; on hover the
tksheet widget shrinks to ``viewport − button`` (same
``_field_place_kw`` pattern as ConfigSheet's floated field) and
switches to right-aligned scroll-to-filename.  Editing uses a plain
``ttk.Entry`` overlaid on the cell — native single-line scroll, no
wrapping, cursor always visible, right-justified with the cursor at
the path end.

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


def _status_body(mode: str) -> str:
    """``path_field`` status = doc general + STR suffix (``path_field.status.<mode>``).

    General = ``### `path_field` `` short body (pre-``####``) from
    ``config_reference.md``; suffix = mode-specific GUI hint from ``STR``.
    Mirrors ``time_ranges.hover.*`` pattern — md-driven base plus STR augmentation.
    """
    e = help_for_path("path_field", mode=mode)
    base = e.body if e and isinstance(e.body, str) and e.body else ""
    # Fallback when mode body absent — help_for_path already falls back to
    # ``_NO_MODE`` short when the requested mode has no tagged section.
    if not base:
        if (ge := help_for_path("path_field")) and isinstance(ge.body, dict):
            from tcm_gui._help import _NO_MODE, ModeBody

            raw = ge.body.get(_NO_MODE)
            if isinstance(raw, ModeBody):
                base = raw.short
            elif isinstance(raw, str):
                base = raw
    suffix = _S.get(f"path_field.status.{mode}", "")
    if base and suffix:
        return f"{base} {suffix}"
    return base or suffix


class PathField(ttk.Frame):
    """A single-cell tksheet posing as the path field.

    Display: left-aligned (``"w"``) by default; on hover the tksheet
    widget is shrunk to ``frame − button`` (``place(width=…)``) and
    switches to right-aligned — filename ends right before the browse
    button, same geometry as ConfigSheet's ``_field_place_kw``.
    Editing uses a plain ``ttk.Entry`` overlay, always right-justified
    with the cursor at the path end; the tksheet editor behind an open
    dropdown list keeps the same right-justified edit text (only the
    list itself is left-aligned).

    When the cell is empty, a dim-gray *placeholder* text is shown
    (e.g. ``"D:/data/_raw/"``) — it vanishes on first keystroke or
    double-click.  Holding Shift swaps to the *advanced* placeholder
    (e.g. ``"D:/data/_raw/(i*raw_file1[.]txt|i*raw_file2[.]txt)"``)
    revealing pattern syntax; releasing Shift restores the simple one.

    Parameters
    ----------
    align : ``"w"`` | ``"e"``
        Initial cell alignment (the edit Entry is always right-justified).
        Standalone and ConfigSheet floated fields are ``"w"`` (left,
        matching the cells they cover); ``"e"`` is kept for the transient
        hover-shrink right-align-scroll-to-filename mode.
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
        on_browse_click: Callable[[], None] | None = None,
        placeholder: str = _S.get("path_field.placeholder", ""),
        placeholder_shift: str = _S.get("path_field.placeholder_files", ""),
        shift_swap: bool = True,  # False for floated fields in ConfigSheet
    ) -> None:
        super().__init__(parent)
        self._on_commit = on_commit
        self._on_begin_edit_cb = on_begin_edit
        self._on_end_edit_cb = on_end_edit
        self._align = align  # "w" standalone, "w" floated (display only)
        self._pre_edit = ""
        self._editing = False
        self._entry: ttk.Entry | None = None
        self._expand_once = False  # label/Up-Down expand bypasses the pointer-band check once
        self._hovering = False
        self._btn_w: int | None = None
        # Hover status: doc-driven from the config_reference ``path_field``
        # section — ``dir`` mode short body (changes to ``files`` on Shift-held).
        self._path_status = _status_body("dirs")
        self._shift_status = _status_body("files")
        self._browse_hint = status_hint
        self._browse_hint_files = status_hint_files

        self._mouse_in = False  # True when pointer is inside PathField frame
        self._shift_swap = shift_swap
        self._on_browse_click = on_browse_click
        # Status callback — shared with BrowseOverlay; also called by Shift handlers.
        self._on_status = on_status
        self._path_status_shift = self._shift_status or self._path_status
        # Placeholder state — two levels: simple (default) and advanced (Shift held).
        self._placeholder_simple = placeholder
        self._placeholder_shift = placeholder_shift or placeholder
        self._ph = CellPlaceholder()
        # Ghost fg comes from the ``_dim_fg`` property — never snapshotted:
        # apply_theme_defaults() swaps the theme globals AFTER widgets exist
        # (dark mode); a captured value would go stale and the ghost would
        # vanish against the entry background.

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
        # Combobox UX: Up/Down on the cell expands an attached dropdown.
        # tksheet's own Up/Down navigation is bound first and returns "break",
        # swallowing later handlers — drop it on this 1×1 cell (selection has
        # nowhere to move) and own the keys. Fires only with canvas focus
        # (Entry / open popup own their keys).
        for _seq in ("<Up>", "<Down>"):
            self.sh.MT.unbind(_seq)
            self.sh.MT.bind(_seq, self._on_expand_key, add="+")
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
            on_click=on_browse_click,
        )
        # Patch overlay to shrink/restore tksheet widget on show/hide
        self._orig_ss = self._ov.schedule_show
        self._ov.schedule_show = self._patched_schedule_show
        self._orig_hide = self._ov.hide
        self._ov.hide = self._patched_hide
        # Esc/close of the tksheet editor or dropdown must never leave a
        # blank field: opening either scrolls the 4096 px column into view
        # (viewport jumps left) while the shrunk layout keeps right-aligned
        # text at the far right — after close the data is intact but the
        # canvas shows empty space (only the next hover transition masked
        # this by re-scrolling). Restore the viewport on every close.
        self._orig_hide_editor_dropdown = self.sh.MT.hide_text_editor_and_dropdown
        self.sh.MT.hide_text_editor_and_dropdown = self._patched_hide_editor_dropdown
        self._binder = SheetHoverBinder(self.sh, self._ov, self._hover_place_kw)
        self.bind("<Configure>", self._on_configure, add="+")
        self.pack_propagate(False)
        self.sh.pack(fill="both", expand=True)
        with suppress(AttributeError, TclError):
            # Shrink the frame to the font's text height, not the padded
            # tksheet row height (min_row_height = max(6, font_h, index_h) + 6).
            # On some Windows display settings the +6 padding dwarfs the font,
            # leaving the path field much taller than its text.  Override
            # tksheet's min_row_height floor and reset the single row so it
            # shrinks to fit the font exactly — anchored at the top: the
            # browse button (rely=0.0, anchor="ne") and edit Entry
            # (relheight=1) follow the shorter frame, other grid elements
            # (label, separator, button bar) are unaffected.
            font_height = self.sh.MT.table_txt_height
            self.sh.MT.min_row_height = font_height
            self.sh.default_row_height(font_height)
            self.sh.MT.reset_row_positions()
            self.configure(height=font_height)
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
        # Frozen while the dropdown list is open — restoring would scroll
        # the canvas under it (the overflow list is positioned once, at open).
        # Edit-start always lands here list-closed.
        try:
            dd_open = bool(self.sh.MT.dropdown.open)
        except Exception:
            dd_open = False
        if not dd_open:
            self._restore_default_layout()
        self._orig_hide()

    def _patched_hide_editor_dropdown(self, redraw: bool = True) -> None:
        """Close tksheet editor/dropdown, then re-show the value (never blank).

        Covers every close path — Esc in the editor (``close_text_editor``),
        Esc/FocusOut on the list (``close_dropdown_window``), click-away
        commit: the original hides + refreshes, then the viewport is
        re-scrolled to the value end (shrunk layout) or start (default).
        Skipped while the custom Entry owns the canvas.
        """
        self._orig_hide_editor_dropdown(redraw=redraw)
        try:
            if self._editing or self._entry is not None:
                return
            if self._hovering or self.sh.MT.align == "ne":
                self._scroll_to_right()
            else:
                self._scroll_to_left()
            self.sh.redraw()
        except Exception:
            pass

    def _on_configure(self, _event) -> None:
        if self._editing:
            return
        if self._hovering:
            # viewport resized — re-constrain tksheet width
            self.sh.place(width=self._hover_shrink_width())
            self.update_idletasks()  # force geometry before scroll
            self._scroll_to_right()
        elif self.sh.MT.align == "ne":
            # right-aligned non-hover — keep scrolled
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
    @property
    def _dim_fg(self) -> str:
        """Ghost fg resolved live — theme globals may be swapped for dark mode."""
        return tcm_gui.theme.ghost_fg(tcm_gui.theme.ENTRY_BG_FALLBACK)

    def _set_font_weight(self, bold: bool) -> None:
        """tksheet has one table font — ghosts render regular, data bold."""
        cur = self.sh.font()
        want = "bold" if bold else "normal"
        if len(cur) < 3 or cur[2] != want:
            self.sh.font((cur[0], cur[1], want))

    def _show_placeholder(self) -> None:
        """Render the placeholder text dimmed in the empty cell."""
        self._set_font_weight(False)
        self._ph.show(self.sh, 0, 0, self._placeholder_simple, self._dim_fg)

    def _clear_placeholder(self) -> None:
        """Remove placeholder text and restore default foreground."""
        self._ph.clear(self.sh, 0, 0)
        self._set_font_weight(True)

    def _swap_placeholder(self, text: str) -> None:
        """Swap displayed placeholder text (simple ↔ Shift variant) without changing tracking.

        Delegates to :meth:`CellPlaceholder.show` — the cell is already tracked
        (callers guard on ``self._ph.active``), so the re-add to ``_cells`` is
        an idempotent no-op.
        """
        self._set_font_weight(False)
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

    def set_placeholder(self, text: str) -> None:
        """Replace the displayed placeholder hint (no-op while data is present).

        Lets the host surface the SAME ghost text as the cell underneath
        (ConfigSheet ``_placeholder_for``) instead of the generic one
        this field was constructed with.
        """
        if text and text != self._placeholder_simple:
            self._placeholder_simple = text
            self._placeholder_shift = text
            if self._ph.has(0, 0):
                self._show_placeholder()
                self.sh.redraw()

    def set_error(self, flag: bool) -> None:
        """Red fg on the single cell when *flag* (scan failed), else restore normal fg.

        Clearing the mark on a placeholder cell restores the dim ghost fg —
        ``FG_DEFAULT`` would render the hint full-strength (empty commit →
        ``set_error(False)`` lands AFTER the ghost is re-shown).
        """
        if flag:
            fg = tcm_gui.theme.INVALID_FG
        elif self._ph.has(0, 0):
            fg = self._dim_fg
        else:
            fg = tcm_gui.theme.FG_DEFAULT
        self.sh.highlight_cells(
            row=0,
            column=0,
            fg=fg,
            redraw=False,
        )
        self.sh.redraw()

    def cancel_edit(self) -> None:
        """Close the Entry editor if open."""
        if self._entry is not None:
            self._commit_entry(cancel=True)

    def expand_dropdown(self) -> None:
        """Programmatically expand the cell dropdown (label click, Up/Down keys).

        Bypasses the pointer-band check (`_is_dropdown_expand` would fail —
        the pointer is on the label, not the arrow) — tksheet's own ``"rc"``
        opener event drives ``open_dropdown_window`` through the normal gate.
        No-op without an attached dropdown, while editing, or already open.
        """
        try:
            mt = self.sh.MT
            if self._editing or self._entry is not None or mt.dropdown.open:
                return
            if not mt.get_cell_kwargs(mt.datarn(0), mt.datacn(0), key="dropdown"):
                return
        except Exception:
            return
        self._expand_once = True
        try:
            mt.open_dropdown_window(0, 0, event="rc")
        finally:
            self._expand_once = False
        # The 4096 px column embeds tksheet's editor as a canvas window —
        # ``open_text_editor`` re-shows the cell's left edge, clipping a
        # right-aligned tail. Re-scroll after open so the edit text (like
        # the custom Entry) shows its filename end.
        with suppress(TclError):
            self.update_idletasks()
            self._scroll_to_right()
            self.sh.redraw()

    def toggle_dropdown(self) -> None:
        """Label-click toggle: close the open list, else expand it."""
        try:
            if self.sh.MT.dropdown.open:
                self.sh.MT.close_dropdown_window()
                return
        except Exception:
            pass
        self.expand_dropdown()

    def _on_expand_key(self, _event) -> str | None:
        """Up/Down on the cell → expand an attached dropdown (combobox UX)."""
        try:
            mt = self.sh.MT
            if self._editing or self._entry is not None or mt.dropdown.open:
                return None
            if not mt.get_cell_kwargs(mt.datarn(0), mt.datacn(0), key="dropdown"):
                return None
        except Exception:
            return None
        self.expand_dropdown()
        return "break"

    # ── edit lifecycle (Entry overlay) ─────────────────────────────
    def _on_begin_edit(self, event) -> str | None:
        """Open a ``ttk.Entry`` over the cell instead of tksheet's editor.

        An arrow click on a dropdown cell expands the list instead (see
        :meth:`_is_dropdown_expand`): tksheet's ``open_dropdown_window``
        gates the list on ``open_text_editor`` succeeding, so vetoing here
        would open the Entry while the list never appears.

        Returns ``None`` to veto tksheet's built-in editor — we handle
        editing entirely through the Entry.
        """
        if self._is_dropdown_expand(event):
            # Double-click-like state: full-width layout, no button/overlay —
            # tksheet's editor opens over the cell with the current value
            # visible (a frozen shrunk layout hid it: viewport left while
            # right-aligned text sat at the far right of the wide column).
            # Keep edit text right-aligned like the custom Entry — only the
            # dropdown list itself stays left-aligned (overflow patch forces
            # ``align="w"`` there, independent of the cell).
            if self._hovering:
                self._hovering = False
                self.sh.place_forget()
                self.sh.pack(fill="both", expand=True)
                self.sh.table_align("e", redraw=False)
                self.update_idletasks()
                self.sh.redraw()
                self._scroll_to_right()
            elif self.sh.MT.align != "ne":
                self.sh.table_align("e", redraw=False)
                self.sh.redraw()
                self._scroll_to_right()
            # ``open_text_editor`` (gated next by tksheet) re-shows the cell's
            # left edge — re-scroll once it is open so the right-aligned tail
            # stays visible (same post-open step as ``expand_dropdown``).
            self.after_idle(self._scroll_to_right)
            self._orig_hide()  # button off, layout frozen full-width
            self._pre_edit = self.get()  # baseline for the end-edit commit check
            with suppress(Exception):
                self.sh.MT._anchor_pick_pending = None  # stale pick must not mute this edit
            return self.get()  # let tksheet open its editor + list — no Entry, no side effects
        self._editing = True
        with suppress(Exception):
            self.sh.MT._anchor_pick_pending = None
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
        try:
            self._open_entry()
        except TclError:
            # Editor failed to open — release the edit lock, otherwise
            # every ``_editing``-guarded path stays wedged until the
            # sheet is rebuilt (regression: justify="w" wedged the field).
            self._editing = False
            if self._on_end_edit_cb is not None:
                self._on_end_edit_cb()
        return None  # veto tksheet's tk.Text editor

    def _is_dropdown_expand(self, event) -> bool:
        """True for a programmatic expand or a click with the pointer on the cell's dropdown arrow.

        tksheet expands a list only from an arrow-band click (``MT.b1_release``
        → ``open_cell`` → ``open_dropdown_window``), while double-click /
        Return / typing must keep the custom Entry. The synthesized
        ``begin_edit_cell`` dict carries no pointer info, so re-derive the
        same band tksheet checks (right ``table_txt_height + 4`` px of the
        cell) from the live pointer — the handler runs synchronously inside
        the click, so the pointer is still there. EAFP: any anomaly → manual
        edit, never a spurious expand.
        """
        try:
            if self._expand_once:  # programmatic expand (label click, Up/Down) — no pointer needed
                return True
            if (event.key or "") != "??":  # keyboard activation → manual Entry edit
                return False
            row, column = event.row, event.column
            mt = self.sh.MT
            if not mt.get_cell_kwargs(mt.datarn(row), mt.datacn(column), key="dropdown"):
                return False
            pointer = (mt.winfo_pointerx() - mt.winfo_rootx(), mt.winfo_pointery() - mt.winfo_rooty())
            if mt.identify_row(y=pointer[1], allow_end=False) != row:
                return False
            if mt.identify_col(x=pointer[0], allow_end=False) != column:
                return False
            edge = mt.col_positions[column + 1]
            return edge - mt.table_txt_height - 4 < mt.canvasx(pointer[0]) < edge - 1
        except Exception:
            return False

    def _open_entry(self) -> None:
        """Create and place a ``ttk.Entry`` filling the PathField frame."""
        bold = self.sh.font()
        # always right-justified, cursor at the path end so the
        # filename stays visible while editing long paths.  (``justify`` takes
        # left/center/right — NOT tksheet's "w"/"e", which raised TclError.)
        ent = ttk.Entry(self, font=bold, justify="right")
        ent.insert(0, self._pre_edit)
        ent.icursor("end")
        ent.xview_moveto(1.0)
        ent.place(relx=0, rely=0, relwidth=1, relheight=1)
        ent.bind("<Return>", lambda _e: self._commit_entry())
        ent.bind("<Escape>", lambda _e: self._commit_entry(cancel=True))
        # Focus-loss commits — same contract as a tksheet cell (its editor
        # commits on FocusOut).  Without it a click-away leaves an orphaned
        # open Entry wedging every ``_editing``-guarded path.  ``_entry`` is
        # None by the time a re-entrant FocusOut (from destroy) fires.
        ent.bind("<FocusOut>", lambda _e: self._commit_entry() if self._entry is not None else None)
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
            self._clear_placeholder()
            self.sh.set_cell_data(0, 0, new_val)
        elif self._placeholder_simple:
            self._show_placeholder()
        else:
            self._clear_placeholder()
            self.sh.set_cell_data(0, 0, "")
        self.sh.redraw()
        if self.sh.MT.align == "ne":
            self._scroll_to_right()
        else:
            self._scroll_to_left()
        if not cancel and val != self._pre_edit:
            self._notify(val)

    def _on_end_edit(self, event) -> None:
        """tksheet editor commit — mirror the Entry contract (commit + verify + notify).

        Fires for the dropdown-gated editor only (the custom Entry returns
        early): typing a custom path into the open list's edit field and
        committing it (Return/Tab/FocusOut/click-away) must trigger
        ``on_commit`` exactly like the Entry does — otherwise the field
        shows text no scan ever loaded. A list pick is the exception: it
        already rescanned via ``on_select``, so a one-shot
        ``_anchor_pick_pending`` flag (set by the selection handler before
        this chain runs) mutes the duplicate commit.
        """
        if self._entry is not None:
            return  # Entry is in charge
        self._editing = False
        try:
            if getattr(event, "row", 0) != 0 or getattr(event, "column", 0) != 0:
                return
            raw = str(getattr(event, "value", ""))
        except Exception:
            return
        # Placeholder coherence first — a committed value must never hide
        # behind ghost tracking (tksheet wrote the cell already, so untrack
        # instead of blanking, unlike ``_commit_entry`` which sets after).
        if raw:
            if self._ph.has(0, 0):
                self._ph.untrack(self.sh, 0, 0)
            self._set_font_weight(True)
        elif self._placeholder_simple:
            self._show_placeholder()
        # Read-back check: the stored cell is the truth post-commit. Repair
        # once if tksheet stored something else, so the notified value and
        # the visible text can never diverge.
        actual = self.get()
        if raw and actual != raw:
            with suppress(TclError):
                self.sh.set_cell_data(0, 0, raw)
                actual = self.get()
        # One-shot pick mute — a list pick already rescanned via ``on_select``.
        pending = getattr(self.sh.MT, "_anchor_pick_pending", None)
        if pending is not None:
            with suppress(Exception):
                self.sh.MT._anchor_pick_pending = None
            if actual == pending:
                self._pre_edit = actual
                return
        if actual != self._pre_edit:
            self._pre_edit = actual
            self._notify(actual)

    def _notify(self, value: str) -> None:
        if self._on_commit is not None:
            self.after_idle(lambda v=value: self._on_commit(v))

    # ── hover policy ──────────────────────────────────────────────
    def _hover_place_kw(self, _event) -> dict[str, Any] | None:
        """Overlay anchor, or None (hide) while editing or the dropdown list is open.

        Motion during Entry editing would re-shrink the sheet under the open
        Entry; the button floats at the cell's top-right, above an open
        list's first rows. Hiding keeps full-width layout + bare canvas until
        commit/cancel/close; the next motion re-shows.
        `SheetHoverBinder` hides on None.
        """
        if self._editing:
            return None
        try:
            if self.sh.MT.dropdown.open:
                return None
        except Exception:
            pass
        return self._place_kw()

    def _place_kw(self) -> dict[str, Any]:
        return {"in_": self, "relx": 1.0, "rely": 0.0, "x": -2, "anchor": "ne"}

    def _hover_write(self, text: str) -> None:
        self.set(text)
        self._notify(text)

    def _hover_read(self) -> str:
        return self.get()
