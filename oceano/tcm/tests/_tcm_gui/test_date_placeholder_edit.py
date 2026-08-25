"""Regression: dim date placeholder must survive cancelled cell edits.

Root cause (tksheet 7.x): committing ``""`` over an already-``""`` cell is
rejected by ``input_valid_for_cell`` (``cell_equal_to``), so ``end_edit_cell``
never fires on cancel — the begin-edit placeholder clear becomes permanent.
Fix under test: ConfigSheet hooks ``MT.hide_text_editor_and_dropdown`` (every
editor-CLOSE path funnels there; ``open_text_editor`` calls plain
``hide_text_editor``, so the hook cannot falsely restore mid-open) and re-shows
the placeholder via ``text_editor.coords``.

Event fidelity: editors are opened through the real ``b1_press``/``double_b1``
pipeline and closed through the real click-away path (``b1_press`` on another
cell → ``mouseclick_outside_editor_or_dropdown`` → hook) — the synthetic-event
traps (no ``num``/``keycode`` markers, ``focus_get() is None``) are avoided.
"""

from __future__ import annotations

import tkinter as tk
from contextlib import suppress

import pytest

import tcm_gui._sheet_tint as _sheet_tint
import tcm_gui.coef_sheet as coef_sheet
import tcm_gui.theme as theme
from tcm_gui.coef_sheet import ConfigSheet

FMT = _sheet_tint._DATE_FMT
PH_COL = _sheet_tint._DATE_PH_COL  # tksheet col of the date cell (0-based)


def _mouse_ev(mt, r: int, c: int) -> tk.Event:
    ev = tk.Event()
    ev.x = mt.col_positions[c] + 3
    ev.y = mt.row_positions[r] + 3
    ev.state = 0
    ev.num, ev.keycode = 1, "??"  # real-mouse markers — open_text_editor requires them
    return ev


@pytest.fixture(scope="module")
def cs(_session_tk_root):
    """Use the session root (not a new tk.Tk) so PhotoImage objects from
    tksheet's ``new_sheet_options`` and the Sheet widget share the same
    Tcl interpreter — multiple ``tk.Tk()`` instances create separate
    interpreters on Windows, making cross-interpreter PhotoImages invalid."""
    if _session_tk_root is None:
        pytest.skip("Tk not available")
        return
    root = _session_tk_root
    try:
        root.geometry("600x300+40+40")
        root.deiconify()
    except tk.TclError:
        pytest.skip("Tk not available")
        return
    sheet = ConfigSheet(root)
    sheet.sh.pack(fill="both", expand=True)
    d1 = sheet.sh.insert(parent="", text="dt_from", values=[""] * 6)
    r2 = sheet.sh.insert(parent="", text="plain", values=["v"] + [""] * 5)
    sheet._meta[d1] = {"has_date": True, "key": "dt_from", "max_col": 0}
    sheet._meta[r2] = {"key": "plain", "max_col": 1}
    sheet._rebuild_row_caches()
    sheet._apply_placeholders()
    root.update_idletasks()
    root.update()
    yield sheet
    # Clean up widgets but keep the root alive for the session fixture
    sheet.sh.destroy()


def _open_editor(sheet: ConfigSheet, r: int, c: int) -> None:
    mt = sheet.sh.MT
    mt.b1_press(_mouse_ev(mt, r, c))  # first click selects the cell
    mt.double_b1(_mouse_ev(mt, r, c))  # Double-Button-1 opens the editor
    _pump(sheet)
    # Real editing implies editor focus — also required: both the click-away
    # path (mouseclick_outside → close_text_editor) and Escape/Enter check
    # ``focus_get() is None → "break"``.  ``focus_force`` because the suite may
    # lack OS foreground (a plain focus_set no-ops on non-foreground windows).
    with suppress(AttributeError, tk.TclError):
        mt.text_editor.window.tktext.focus_force()
    _pump(sheet)


def _pump(sheet: ConfigSheet) -> None:
    root = sheet.sh.winfo_toplevel()
    root.update_idletasks()
    root.update()


def _close_ev(keysym: str) -> tk.Event:
    ev = tk.Event()
    ev.keysym, ev.num, ev.keycode = keysym, 1, "??"
    return ev


def _cell(sheet: ConfigSheet, r: int) -> str:
    return str(sheet.sh.get_cell_data(r, PH_COL) or "")


class TestDatePlaceholderSurvivesCancel:
    def test_placeholder_initially_shown(self, cs):
        assert _cell(cs, 0) == FMT
        assert cs._ph.has(0, PH_COL)

    def test_begin_edit_clears_without_false_restore(self, cs):
        """Editor open → cell blank; hook must NOT restore mid-open."""
        _open_editor(cs, 0, PH_COL)
        assert cs.sh.MT.text_editor.open
        assert _cell(cs, 0) == ""
        assert not cs._ph.has(0, PH_COL)
        cs.sh.MT.close_text_editor(_close_ev("Escape"))  # restore state for next tests
        _pump(cs)

    def test_clickaway_restores_placeholder(self, cs):
        """THE regression: double-click date cell → click another cell."""
        _open_editor(cs, 0, PH_COL)
        cs.sh.MT.b1_press(_mouse_ev(cs.sh.MT, 1, 0))
        _pump(cs)
        assert _cell(cs, 0) == FMT, "placeholder lost after click-away"
        assert cs._ph.has(0, PH_COL)
        assert not cs.sh.MT.text_editor.open

    def test_escape_restores_placeholder(self, cs):
        _open_editor(cs, 0, PH_COL)
        cs.sh.MT.close_text_editor(_close_ev("Escape"))
        _pump(cs)
        assert _cell(cs, 0) == FMT
        assert cs._ph.has(0, PH_COL)

    def test_enter_commits_real_date(self, cs):
        _open_editor(cs, 0, PH_COL)
        cs.sh.MT.text_editor.window.tktext.insert("1.0", "2020-01-02T03:04:05")
        cs.sh.MT.close_text_editor(_close_ev("Return"))
        _pump(cs)
        assert _cell(cs, 0) == "2020-01-02T03:04:05"
        assert not cs._ph.has(0, PH_COL)
        # cleanup: restore placeholder state for any later assertions
        cs.sh.set_cell_data(0, PH_COL, "")
        cs._ph.show(cs.sh, 0, PH_COL, FMT, theme.DEFAULT_FG)
