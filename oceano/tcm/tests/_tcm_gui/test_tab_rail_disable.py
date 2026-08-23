"""Inert (disabled) look + click veto of TabRail + overall caption.

Simple mode before the first successful scan: the rail + centered caption
must read as awaiting a data path, mirroring the readonly tksheet pages —
dim text, no selection accent, no hand cursor, clicks ignored.  Re-enabled
by a new search (``_on_path_changed``) or a successful scan.

Note: root is withdrawn (headless Tk) — the canvas never maps, so the rail
fixture fakes ``winfo_height`` and click/motion handlers are driven with
synthetic events carrying bare ``x``/``y``.
"""

from __future__ import annotations

import sys
import tkinter as tk

import pytest

_mod = sys.modules[__name__]
_mod._root = None


@pytest.fixture(autouse=True, scope="module")
def _tk_root():
    try:
        r = tk.Tk()
        r.withdraw()
    except tk.TclError:
        pytest.skip("Tk not available")
        return
    _mod._root = r
    yield
    r.destroy()
    _mod._root = None


def _event(x: int, y: int):
    return type("E", (), {"x": x, "y": y})()


@pytest.fixture()
def rail():
    """Two-tab rail with laid-out cells.

    The withdrawn root never maps the canvas — ``winfo_height()`` stays 1 and
    ``_layout`` would bail before creating cells — so the layout height is
    faked on the instance (also covers the internal ``add_tab`` → ``_layout``).
    """
    from tcm_gui._tab_rail import TabRail
    from tcm_gui.theme import scaled

    calls: list[str] = []  # on_select invocations — selection itself is App's job
    r: TabRail = TabRail(_mod._root, on_select=lambda name: calls.append(name))
    r.winfo_height = lambda: 200
    r.add_tab("alpha")
    r.add_tab("beta")
    r._tab_x = scaled(TabRail.PROG_W) + 5  # inside the tab column
    r._calls = calls  # on_select invocations — selection itself is App's job
    return r


def _text_fill(rail, name: str) -> str:
    return str(rail.itemcget(rail._cells[name]["text"], "fill"))


def _accent_state(rail, name: str) -> str:
    return str(rail.itemcget(rail._cells[name]["acc"], "state"))


def _click_tab(rail, name: str) -> None:
    c = rail._cells[name]
    rail._click(_event(rail._tab_x, (c["y0"] + c["y1"]) // 2))


class TestTabRailDisabled:
    def test_click_vetoed(self, rail):
        rail.set_disabled(True)
        _click_tab(rail, "beta")
        assert not rail._calls, "disabled rail must not fire on_select"

    def test_dim_text_and_hidden_accent(self, rail):
        rail.set_disabled(True)
        assert _text_fill(rail, "alpha") == rail._pal["dim"]
        assert _accent_state(rail, "alpha") == "hidden", "selected accent must hide when disabled"

    def test_no_hand_cursor(self, rail):
        rail.set_disabled(True)
        c = rail._cells["alpha"]
        rail._motion(_event(rail._tab_x, (c["y0"] + c["y1"]) // 2))
        assert str(rail.cget("cursor")) != "hand2"

    def test_reenable_restores_accent_and_click(self, rail):
        rail.set_disabled(True)
        rail.set_disabled(False)
        assert _accent_state(rail, "alpha") == "normal"
        assert _text_fill(rail, "alpha") == rail._pal["text"]
        _click_tab(rail, "beta")
        assert rail._calls == ["beta"]

    def test_hand_cursor_when_enabled(self, rail):
        c = rail._cells["alpha"]
        rail._motion(_event(rail._tab_x, (c["y0"] + c["y1"]) // 2))
        assert str(rail.cget("cursor")) == "hand2"

    def test_disabled_survives_relayout(self, rail):
        """Cells rebuilt after clear/add keep the disabled rendering."""
        rail.set_disabled(True)
        rail.clear()
        rail.add_tab("gamma")
        assert rail._disabled is True
        assert _text_fill(rail, "gamma") == rail._pal["dim"]


class TestAppCfgUiDisabled:
    """App._set_cfg_ui_disabled toggles rail state + caption foreground."""

    def test_fg_follows_disabled_state(self, rail):
        from tkinter import ttk

        import tcm_gui.theme
        from tcm_gui.app import App

        ns = type("NS", (), {})()
        ns._rail = rail
        ns._overall_lbl = ttk.Label(_mod._root, text="Default", anchor="center")
        fn = App._set_cfg_ui_disabled.__get__(ns)

        fn(True)
        assert str(ns._overall_lbl.cget("foreground")) == tcm_gui.theme.DEFAULT_FG
        assert rail._disabled is True
        fn(False)
        assert str(ns._overall_lbl.cget("foreground")) == tcm_gui.theme.FG_DEFAULT
        assert rail._disabled is False
        ns._overall_lbl.destroy()
