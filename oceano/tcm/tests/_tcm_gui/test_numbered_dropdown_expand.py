"""Anchor dropdown on the path cell: triggers, commit contract, overlay coexistence.

tksheet's inherent dropdown embeds its list *inside* the canvas viewport —
on the 1×1 path field (~18 px tall) the measured list window is 1 px tall,
i.e. permanently invisible. ``NumberedPathDropdown`` therefore relies on the
overflow patch (``_dropdown_overflow.enable_dropdown_overflow``) placing
tksheet's own list in the toplevel below the field; the triggers below are
the only openers:

- click on the path caption (``App._path_lbl`` → ``PathField.expand_dropdown``),
- Up/Down on the cell (``PathField._on_expand_key`` → ``expand_dropdown``).

Pinned contracts:
- Attaching/replacing paths never touches the cell (bare path only).
- Picking ``N. path`` commits the bare path, updates the number, closes the
  list, and fires ``on_select`` (rescan).
- Esc / FocusOut / click-away close never blanks the field: opening the
  editor scrolls the 4096 px column left while the shrunk layout keeps
  right-aligned text at the far right, so the close path re-scrolls to the
  value (``_patched_hide_editor_dropdown``).
- Manual edit (double-click / Return / F2 / typing) still opens the custom
  Entry and vetoes tksheet's editor.
- Structural invariant arrow-visible ⟺ no Entry open: edit start restores
  full-width layout; binder motion neither re-shrinks nor re-shows while
  editing (else the open Entry swallows all canvas clicks — self-perpetuating).
"""

from __future__ import annotations

import time
import tkinter as tk
from contextlib import suppress
from tkinter import TclError, ttk
from types import SimpleNamespace

import pytest
from tksheet.functions import event_dict

from tcm_gui._numbered_dropdown import NumberedPathDropdown
from tcm_gui._path_field import PathField

PATHS = ["C:/data/first", "C:/data/second", "C:/data/third"]
CELL = PATHS[1]
LABEL_DEFAULT = "PATH"


@pytest.fixture()
def field(_session_tk_root):
    """Fresh PathField in the session interpreter, cell preset to ``CELL``."""
    if _session_tk_root is None:
        pytest.skip("Tk not available")
    from tcm_gui._path_field import PathField as _PF

    f = _PF(_session_tk_root)
    f.pack(fill="x")
    f.set(CELL)
    _session_tk_root.update_idletasks()
    yield f
    with suppress(TclError):
        f.destroy()


@pytest.fixture()
def rig(field, _session_tk_root):
    """Field + label + attached dropdown, pristine ``picks`` recorder."""
    lbl = ttk.Label(_session_tk_root)
    lbl.pack()
    picks: list[str] = []
    dd = NumberedPathDropdown(
        field.sh, 0, 0, [], number_label=lbl, default_text=LABEL_DEFAULT, on_select=picks.append
    )
    dd.set_paths(PATHS)
    dd.refresh()
    _session_tk_root.update_idletasks()
    ns = SimpleNamespace(root=_session_tk_root, field=field, dd=dd, lbl=lbl, picks=picks, mt=field.sh.MT)
    yield ns
    with suppress(TclError):
        lbl.destroy()


def _begin(key="??"):
    """Faithful ``begin_edit_cell`` dict as built by tksheet's ``open_text_editor``."""
    return event_dict(name="begin_edit_table", key=key, value=CELL, row=0, column=0)


@pytest.mark.parametrize(
    ("via", "desc"),
    [
        ("ctor", "constructor attaches with initial paths"),
        ("set", "set_paths replaces the list"),
    ],
    ids=["constructor", "set_paths"],
)
def test_attach_keeps_bare_path(field, via, desc):
    """Attaching the dropdown must not rewrite the cell with a numbered display value."""
    lbl = ttk.Label(field.master)
    dd = NumberedPathDropdown(
        field.sh,
        0,
        0,
        PATHS if via == "ctor" else [],
        number_label=lbl,
        default_text=LABEL_DEFAULT,
    )
    if via == "set":
        dd.set_paths(PATHS)
    actual = field.sh.get_cell_data(0, 0)
    assert actual == CELL, f"{desc}: attach rewrote the cell - {CELL=!r}, {actual=!r}"


@pytest.mark.parametrize(
    ("key", "desc"),
    [
        ("??", "click activation opens the custom Entry"),
        ("Return", "Return opens the custom Entry"),
        ("F2", "F2 opens the custom Entry"),
    ],
    ids=["click", "return-key", "f2-key"],
)
def test_manual_edit_keeps_entry(rig, key, desc):
    """Genuine edits still open the custom Entry and veto tksheet's editor."""
    try:
        ret = rig.mt.extra_begin_edit_cell_func(_begin(key))
        assert ret is None, f"{desc}: manual edit did not veto - {ret=!r}"
        assert rig.field._entry is not None, f"{desc}: manual edit opened no custom Entry"
    finally:
        rig.field.cancel_edit()


@pytest.fixture()
def mapped(_session_tk_root):
    """Mapped Toplevel (real geometry) in the session interpreter."""
    if _session_tk_root is None:
        pytest.skip("Tk not available")
    top = tk.Toplevel(_session_tk_root)
    top.geometry("400x300+0+0")  # tall: the height override needs room below the field
    top.deiconify()
    for _ in range(50):  # busy suite: the WM may lag behind a single update
        top.update_idletasks()
        top.update()
        if top.winfo_width() >= 390 and top.winfo_height() >= 290:
            break
        time.sleep(0.02)
    yield top
    with suppress(TclError):
        top.destroy()


def _mapped_field(mapped):
    """PathField + label + attached dropdown in the mapped window, cell preset to ``CELL``."""
    field = PathField(mapped)
    field.pack(fill="x")
    lbl = ttk.Label(mapped)
    picks: list[str] = []
    dd = NumberedPathDropdown(
        field.sh, 0, 0, [], number_label=lbl, default_text=LABEL_DEFAULT, on_select=picks.append
    )
    field.set(CELL)
    dd.set_paths(PATHS)
    dd.refresh()
    mapped.update_idletasks()
    mapped.update()
    return SimpleNamespace(field=field, dd=dd, lbl=lbl, picks=picks, mt=field.sh.MT)


def test_begin_edit_restores_layout(rig):
    """Edit start restores full-width layout — the Entry always opens with the arrow off-screen."""
    rig.field._switch_to_hover_shrink()
    assert rig.field._hovering is True, "hover shrink did not engage"
    try:
        rig.field._on_begin_edit(_begin("Return"))
        assert rig.field._entry is not None, "Entry did not open"
        assert rig.field._hovering is False, "edit start did not restore full-width layout"
    finally:
        rig.field.cancel_edit()


def test_motion_during_edit_keeps_layout(rig):
    """Binder motion while the Entry is open must not re-shrink the sheet nor show the button."""
    rig.field._editing = True
    try:
        rig.field._binder._on_motion(None)
        assert rig.field._hovering is False, "motion during edit re-shrunk the sheet under the Entry"
        assert not rig.field._ov.visible, "motion during edit showed the browse button"
    finally:
        rig.field._editing = False


def _popup_rows(dd):
    """tksheet dropdown kwargs values (fails loudly when unattached)."""
    return list(dd.sheet.MT.get_cell_kwargs(dd.row, dd.column, key="dropdown")["values"])


def test_dropdown_values_numbered(rig):
    """Attached values show ``N. path`` while the cell keeps the bare path."""
    assert _popup_rows(rig.dd) == [f"{i}. {p}" for i, p in enumerate(PATHS, start=1)], "values wrong"
    assert rig.field.sh.get_cell_data(0, 0) == CELL, "attach rewrote the cell"


def test_overflow_list_visible(mapped):
    """Overflow list: below the field, all rows shown, field-width floor, left-aligned text."""
    ns = _mapped_field(mapped)
    try:
        ns.field.expand_dropdown()
        mt = ns.mt
        assert mt.dropdown.open is True, "list did not open"
        win = mt.dropdown.window
        mapped.update_idletasks()
        mapped.update()
        h = win.winfo_height()
        assert h >= 2 * mt.table_txt_height, f"list too short to be usable - {h=!r}"
        field_bottom = ns.field.winfo_rooty() + ns.field.winfo_height()
        assert win.winfo_rooty() >= field_bottom - mt.table_txt_height, "list not below the field"
        assert win.winfo_width() >= mt.winfo_width() - 1, "list narrower than the field"
        assert win.winfo_rootx() + win.winfo_width() <= mt.winfo_screenwidth(), "list off the screen"
        assert win.MT.align in ("w", "nw"), "list text not left-aligned"
    finally:
        with suppress(Exception):
            mt.close_dropdown_window()
        with suppress(TclError):
            ns.field.destroy()
            ns.lbl.destroy()


def test_overflow_height_fits_all_items(mapped):
    """Height covers every item — no tksheet six-item / 500 px cap."""
    ns = _mapped_field(mapped)
    try:
        mt = ns.mt
        ns.dd.set_paths([f"C:/data/anchor-{i:02d}" for i in range(8)])
        ns.field.expand_dropdown()
        assert mt.dropdown.open is True, "list did not open"
        win = mt.dropdown.window
        mapped.update_idletasks()
        mapped.update()
        full = 5 + 8 * win.MT.min_row_height
        assert win.winfo_height() >= full, f"list capped below all 8 rows - {win.winfo_height()!r}"
    finally:
        with suppress(Exception):
            mt.close_dropdown_window()
        with suppress(TclError):
            ns.field.destroy()
            ns.lbl.destroy()


def test_overflow_height_capped_by_window(mapped):
    """A tall list reaches the window bottom (and scrolls) instead of clipping past it."""
    ns = _mapped_field(mapped)
    try:
        mt = ns.mt
        ns.dd.set_paths([f"C:/data/anchor-{i:02d}" for i in range(30)])
        mapped.geometry("400x120+0+0")  # short window: the cap must follow it, not the screen
        mapped.update_idletasks()
        mapped.update()
        ns.field.expand_dropdown()
        assert mt.dropdown.open is True, "list did not open"
        win = mt.dropdown.window
        mapped.update_idletasks()
        mapped.update()
        cap = mapped.winfo_height() - win.winfo_y() - 2
        assert win.winfo_height() <= cap + 1, f"list overflows the window - {win.winfo_height()!r}"
        assert win.winfo_height() >= mt.table_txt_height + 5, "capped list unusably short"
    finally:
        with suppress(Exception):
            mt.close_dropdown_window()
        with suppress(TclError):
            ns.field.destroy()
            ns.lbl.destroy()


def test_overflow_wide_content_stays_in_window(mapped):
    """Content wider than the window is capped at the window (no clipping past it)."""
    ns = _mapped_field(mapped)
    try:
        mt = ns.mt
        ns.dd.set_paths([f"C:/data/{'x' * 200}.nc"])
        ns.field.expand_dropdown()
        assert mt.dropdown.open is True, "list did not open"
        win = mt.dropdown.window
        mapped.update_idletasks()
        mapped.update()
        assert win.winfo_width() >= mt.winfo_width() - 1, "list narrower than the field"
        assert win.winfo_width() <= mapped.winfo_width(), "list wider than the window (clipped)"
        right = win.winfo_rootx() + win.winfo_width()
        assert right <= mapped.winfo_rootx() + mapped.winfo_width() + 1, "list past the window edge"
        assert win.MT.align in ("w", "nw"), "wide list text not left-aligned"
    finally:
        with suppress(Exception):
            mt.close_dropdown_window()
        with suppress(TclError):
            ns.field.destroy()
            ns.lbl.destroy()


def test_expand_restores_full_layout(mapped):
    """Expand from hover shows the double-click state: full width, value visible, no button."""
    ns = _mapped_field(mapped)
    try:
        mt = ns.mt
        ns.field._switch_to_hover_shrink()
        ns.field.expand_dropdown()
        assert mt.dropdown.open is True, "list did not open"
        mapped.update_idletasks()
        mapped.update()
        assert ns.field._hovering is False, "expand kept the shrunk layout"
        assert mt.align == "nw", "expand kept right-aligned text"
        assert mt.xview()[0] <= 0.01, f"expand left the viewport scrolled - xview={mt.xview()!r}"
        assert not ns.field._ov.visible, "expand left the browse button up"
        assert mt.text_editor.open is True, "no editor over the cell"
        assert mt.text_editor.get() == CELL, "editor lost the current value"
    finally:
        with suppress(Exception):
            mt.close_dropdown_window()
        with suppress(TclError):
            ns.field.destroy()
            ns.lbl.destroy()


def test_toggle_closes_and_reopens(mapped):
    """Caption-click toggle: close the open list, expand again on the next click."""
    ns = _mapped_field(mapped)
    try:
        mt = ns.mt
        ns.field.expand_dropdown()
        assert mt.dropdown.open is True, "list did not open"
        mapped.update_idletasks()
        mapped.update()
        ns.field.toggle_dropdown()
        mapped.update_idletasks()
        mapped.update()
        assert mt.dropdown.open is False, "toggle did not close the list"
        assert mt.text_editor.open is False, "toggle left the editor open"
        assert ns.field.sh.get_cell_data(0, 0) == CELL, "toggle rewrote the cell"
        ns.field.toggle_dropdown()
        mapped.update_idletasks()
        mapped.update()
        assert mt.dropdown.open is True, "toggle did not reopen the list"
    finally:
        with suppress(Exception):
            mt.close_dropdown_window()
        with suppress(TclError):
            ns.field.destroy()
            ns.lbl.destroy()


def test_editor_arrows_move_highlight(mapped):
    """Up/Down in the editor move the list highlight (combobox UX), not the cursor.

    Synthetic keys don't reach these widgets under a test runner (verified:
    even recorder binds stay silent), so routing is asserted via registration
    (stock tksheet binds no widget-level arrows on its editor) and the step
    handler is driven directly. Real keypresses deliver through the binding.
    """
    from tcm_gui._dropdown_overflow import _step_open_list

    ns = _mapped_field(mapped)
    try:
        mt = ns.mt
        ns.field.expand_dropdown()
        assert mt.dropdown.open is True, "list did not open"
        mapped.update_idletasks()
        mapped.update()
        ed = mt.text_editor.tktext
        assert ed.bind("<Down>"), "Down not routed to the list"
        assert ed.bind("<Up>"), "Up not routed to the list"
        win = mt.dropdown.window
        rows = [_step_open_list(mt, d) and win.row for d in (+1, +1, -1)]
        assert rows == [0, 1, 0], f"arrows did not walk the list - {rows!r}"
        assert win.get_selected_rows() == {0}, "highlight not on the walked row"
        assert mt.text_editor.get() == CELL, "arrow walk rewrote the filter text"
        assert mt.dropdown.open is True, "arrow walk closed the list"
        ns.mt.close_dropdown_window()
        assert _step_open_list(mt, +1) is None, "step acted on a closed list"
    finally:
        with suppress(Exception):
            mt.close_dropdown_window()
        with suppress(TclError):
            ns.field.destroy()
            ns.lbl.destroy()


def test_arrow_release_skips_research(mapped):
    """Navigation-key releases skip filter research; character releases still run it."""
    from tcm_gui._dropdown_overflow import _dropdown_editor_key_release, _step_open_list

    ns = _mapped_field(mapped)
    try:
        mt = ns.mt
        ns.field.expand_dropdown()  # editor text = CELL = row 1 match
        mapped.update_idletasks()
        mapped.update()
        win = mt.dropdown.window
        assert _step_open_list(mt, +1) == "break", "step did not claim the key"
        assert win.row == 0, "Down did not step to row 0"
        _dropdown_editor_key_release(mt, SimpleNamespace(keysym="Down", state=0))
        mapped.update_idletasks()
        mapped.update()
        assert win.row == 0, "release re-researched and snapped back to the text match"
        _dropdown_editor_key_release(mt, SimpleNamespace(keysym="x", state=0))
        mapped.update_idletasks()
        mapped.update()
        assert win.row == 1, "character release stopped researching the filter text"
        assert mt.dropdown.open is True, "release closed the list"
    finally:
        with suppress(Exception):
            mt.close_dropdown_window()
        with suppress(TclError):
            ns.field.destroy()
            ns.lbl.destroy()


def test_raise_overlays_dropdown_above_run(mapped):
    """Root-resize stacking: the open anchor list stays above the floating Run button."""
    from tcm_gui.app import App

    app = App.__new__(App)
    ns = _mapped_field(mapped)
    run = ttk.Button(mapped, text="Run")
    run.place(relx=1.0, rely=1.0, anchor="se")  # the app's bottom-right float
    app._run_btn = run
    app._path_field = ns.field
    try:
        ns.field.expand_dropdown()
        assert ns.mt.dropdown.open is True, "list did not open"
        mapped.update_idletasks()
        mapped.update()
        win = str(ns.mt.dropdown.window)
        stack = lambda: list(mapped.tk.call("winfo", "children", mapped))  # bottom-to-top
        run.lift()  # what a root <Configure> did blindly before the fix
        assert stack().index(str(run)) > stack().index(win), "setup: Run does not cover the list"
        app._raise_overlays()
        assert stack().index(win) > stack().index(str(run)), "Run covers the open list"
        ns.mt.close_dropdown_window()
        app._raise_overlays()  # closed list: no error, Run simply stays up
    finally:
        with suppress(Exception):
            ns.mt.close_dropdown_window()
        with suppress(TclError):
            run.destroy()
            ns.field.destroy()
            ns.lbl.destroy()


def test_overflow_wide_content_widens_narrow_cell(mapped):
    """On a narrow cell the list widens to fit the content (up to the window)."""
    import tkinter.font as tkfont

    from tksheet import Sheet

    from tcm_gui._dropdown_overflow import enable_dropdown_overflow

    sh = Sheet(mapped, total_rows=3, total_columns=1, default_column_width=150)
    sh.pack(fill="both", expand=True)
    long_value = f"C:/data/{'y' * 30}.nc"
    sh.dropdown(0, 0, values=["short", long_value], state="readonly")
    enable_dropdown_overflow(sh)
    mapped.update_idletasks()
    mapped.update()
    try:
        mt = sh.MT
        mt.open_dropdown_window(0, 0)
        assert mt.dropdown.open is True, "list did not open"
        win = mt.dropdown.window
        mapped.update_idletasks()
        mapped.update()
        cell_w = mt.col_positions[1] - mt.col_positions[0] + 1
        assert win.winfo_width() > cell_w, "list clipped to a narrow cell for wide content"
        assert win.winfo_width() <= mapped.winfo_width(), "list wider than the window (clipped)"
        font = tkfont.Font(root=mapped, font=mt.PAR.ops.table_font)
        assert win.winfo_width() >= font.measure(long_value), "longest value does not fit the list"
    finally:
        with suppress(Exception):
            mt.close_dropdown_window()
        with suppress(TclError):
            sh.destroy()


def test_overflow_pick_commits_bare(mapped):
    """Full tksheet pick chain on the overflow list: strip, commit bare, number, notify, close."""
    ns = _mapped_field(mapped)
    try:
        mt = ns.mt
        ns.field.expand_dropdown()
        assert mt.dropdown.open is True, "list did not open"
        mt.close_dropdown_window(0, 0, f"2. {CELL}")
        assert ns.field.sh.get_cell_data(0, 0) == CELL, "pick left the numbered display value"
        assert ns.lbl.cget("text") == "2/3", "number label wrong after pick"
        assert ns.picks == [CELL], "on_select not fired with the bare path"
        assert mt.dropdown.open is False, "list stayed open after pick"
    finally:
        with suppress(Exception):
            mt.close_dropdown_window()
        with suppress(TclError):
            ns.field.destroy()
            ns.lbl.destroy()


def test_expand_noop_without_paths(field):
    """Programmatic expand with no attached dropdown is a silent no-op."""
    field.expand_dropdown()
    assert field.sh.MT.dropdown.open is False, "expand opened a list with no dropdown attached"


def test_expand_noop_while_editing(rig):
    """Programmatic expand while the Entry is open leaves the edit alone."""
    rig.field._on_begin_edit(_begin("Return"))
    assert rig.field._entry is not None, "Entry did not open"
    try:
        rig.field.expand_dropdown()
        assert rig.mt.dropdown.open is False, "expand hijacked an active edit"
        assert rig.field._entry is not None, "expand closed the active Entry"
    finally:
        rig.field.cancel_edit()


def test_empty_paths_disables_expand(rig):
    """``set_paths([])`` clears the dropdown so expand becomes a no-op and the label resets."""
    rig.dd.set_paths([])
    assert rig.mt.get_cell_kwargs(0, 0, key="dropdown") == {}, "stale dropdown survived empty set_paths"
    rig.field.expand_dropdown()
    assert rig.mt.dropdown.open is False, "expand opened a list with no paths"
    assert rig.lbl.cget("text") == LABEL_DEFAULT, "label not restored after clearing paths"


@pytest.mark.parametrize(
    ("key", "desc"), [("<Up>", "Up opens the list"), ("<Down>", "Down opens the list")], ids=["up", "down"]
)
def test_expand_key_opens_list(rig, key, desc):
    """Up/Down on the cell expands the attached dropdown (combobox UX).

    Synthetic key events don't reach canvas bindings in this Tk build
    (verified: even trivial recorders stay silent), so delivery is asserted
    via registration (ours alone — tksheet nav unbound) and the handler is
    driven directly. Real keypresses deliver through the same binding.
    """
    bound = rig.mt.bind(key)
    assert "_on_expand_key" in bound, f"{desc} - our handler not bound for {key}"
    assert "arrowkey" not in bound, f"{desc} - tksheet nav still bound for {key}"
    try:
        assert rig.field._on_expand_key(None) == "break", f"{desc} - handler did not claim the key"
        assert rig.mt.dropdown.open is True, f"{desc} - {key} did not expand the list"
        assert rig.field._entry is None, f"{desc} - {key} opened the custom Entry"
    finally:
        with suppress(Exception):
            rig.mt.close_dropdown_window()


def test_expand_key_noop_without_paths(field):
    """Up/Down without an attached dropdown keeps stock behavior (no list, no error)."""
    field.sh.MT.event_generate("<Down>")
    assert field.sh.MT.dropdown.open is False, "Down opened a list with no dropdown attached"
    assert field._entry is None, "Down opened the custom Entry with no dropdown attached"


def _close_like_escape(mt, where):
    """Drive the real tksheet Esc handlers (synthetic keys don't reach these widgets).

    ``where="editor"`` mirrors the tktext ``<Escape>`` binding
    (``close_text_editor``); ``where="list"`` mirrors the list ``<Escape>``
    binding (``close_dropdown_window`` with the Tk event as ``r``).
    """
    if where == "editor":
        mt.close_text_editor(SimpleNamespace(keysym="Escape", widget=mt.text_editor.tktext))
    else:
        mt.close_dropdown_window(SimpleNamespace())


@pytest.mark.parametrize(("where", "desc"), [("editor", "Esc in editor"), ("list", "Esc on list")])
def test_esc_close_keeps_value_visible(mapped, where, desc):
    """Esc from a hover-shrunk expand leaves the double-click state: full width, value at left."""
    ns = _mapped_field(mapped)
    try:
        mt = ns.mt
        ns.field._switch_to_hover_shrink()  # the real pre-expand state: arrow needs hover
        ns.field.expand_dropdown()
        assert mt.dropdown.open is True, f"{desc}: list did not open"
        mapped.update_idletasks()
        mapped.update()
        if where == "editor":
            # `close_text_editor` bails without app focus (OS-dependent under
            # a test runner) — pin the focus query to the editor; the real
            # Escape branch (keysym check → hide both → focus_set) still runs.
            mt.focus_get = lambda *a, **k: mt.text_editor.tktext
            mt.text_editor.tktext.focus_set()
            mapped.update_idletasks()
            mapped.update()
        _close_like_escape(mt, where)
        mapped.update_idletasks()
        mapped.update()
        assert ns.field.sh.get_cell_data(0, 0) == CELL, f"{desc}: Esc rewrote the cell"
        assert mt.dropdown.open is False, f"{desc}: list stayed open"
        assert mt.text_editor.open is False, f"{desc}: editor stayed open"
        assert ns.field._entry is None, f"{desc}: Esc opened the custom Entry"
        assert ns.field._hovering is False, f"{desc}: Esc left the shrunk layout"
        assert mt.xview()[0] <= 0.01, f"{desc}: value scrolled out of view - xview={mt.xview()!r}"
    finally:
        with suppress(Exception):
            mt.close_dropdown_window()
        with suppress(TclError):
            ns.field.destroy()
            ns.lbl.destroy()


def test_hover_cycle_after_esc_keeps_value(mapped):
    """Hover out/in after an Esc close keeps showing the value at both ends."""
    ns = _mapped_field(mapped)
    try:
        mt = ns.mt
        ns.field._switch_to_hover_shrink()
        ns.field.expand_dropdown()
        mapped.update_idletasks()
        mapped.update()
        _close_like_escape(mt, "editor")
        assert mt.xview()[0] <= 0.01, f"Esc lost the value - xview={mt.xview()!r}"
        ns.field._switch_to_hover_shrink()  # hover back in from outer space
        mapped.update_idletasks()
        mapped.update()
        assert mt.xview()[1] >= 0.99, f"hover-in lost the value - xview={mt.xview()!r}"
        ns.field._restore_default_layout()  # hover back out
        mapped.update_idletasks()
        mapped.update()
        assert mt.xview()[0] <= 0.01, f"hover-out lost the value - xview={mt.xview()!r}"
        assert ns.field.sh.get_cell_data(0, 0) == CELL, "hover cycle rewrote the cell"
    finally:
        with suppress(Exception):
            mt.close_dropdown_window()
        with suppress(TclError):
            ns.field.destroy()
            ns.lbl.destroy()
