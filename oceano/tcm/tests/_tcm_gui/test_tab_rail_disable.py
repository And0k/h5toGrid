"""Inert (disabled) look + click veto of TabRail + overall caption.

Simple mode before the first successful scan: the rail + centered caption
must read as awaiting a data path, mirroring the readonly tksheet pages —
dim text, no selection accent, no hand cursor, clicks ignored.  Re-enabled
by a new search (``_on_path_changed``) or a successful scan.

Tab context ops (``TabRail.remove_tab`` / ``set_tab_disabled`` + App
``_remove_tab`` / ``_set_tab_muted`` / ``_on_rail_context``): right-click menu
with Remove (instant session-only drop) and Disable↔Enable (mute — dim ✕
label, kept page, skipped on Run); menu hidden while busy.

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


class TestTabContextOps:
    """Rail context ops: remove_tab neighbour + per-tab mute rendering."""

    def test_remove_middle_selects_next(self, rail):
        rail.add_tab("gamma")
        assert rail.remove_tab("beta") == "gamma"
        assert rail._names == ["alpha", "gamma"]

    def test_remove_last_falls_back_to_previous(self, rail):
        assert rail.remove_tab("beta") == "alpha"

    def test_remove_unknown_returns_none(self, rail):
        assert rail.remove_tab("nope") is None
        assert rail._names == ["alpha", "beta"]

    def test_mute_dims_and_hides_accent(self, rail):
        rail.set_tab_disabled("alpha", True)
        assert _text_fill(rail, "alpha") == rail._pal["dim"]
        assert _accent_state(rail, "alpha") == "hidden"

    def test_unmute_restores_accent(self, rail):
        rail.set_tab_disabled("alpha", True)
        rail.set_tab_disabled("alpha", False)
        assert _text_fill(rail, "alpha") == rail._pal["text"]
        assert _accent_state(rail, "alpha") == "normal"

    def test_mute_prefixes_cross_label(self, rail):
        rail.set_tab_disabled("beta", True)
        assert rail._full_label("beta").startswith("✕ ")

    def test_mute_does_not_block_click(self, rail):
        rail.set_tab_disabled("beta", True)
        _click_tab(rail, "beta")
        assert rail._calls == ["beta"]  # App filters Run, rail stays clickable


class TestAppTabContextMenu:
    """App._on_rail_context/_remove_tab/_set_tab_muted — menu + state sync."""

    def _app_ns(self):
        from tcm_gui.app import App

        ns = type("NS", (), {})()
        ns._tab_of = {"a": object(), "b": object()}
        ok_sheet = lambda: type(  # noqa: E731 — tiny test double
            "CS",
            (),
            {
                "is_path_valid": lambda self: True,
                "is_dates_valid": lambda self: True,
                "calib_blocking": lambda self: False,
            },
        )()
        ns._pages = {"a": ok_sheet(), "b": ok_sheet()}
        ns._yaml_paths = {"a": object(), "b": object()}
        ns._disabled_tabs = set()
        ns._current = "a"
        rail_calls: list = []
        ns._rail = type(
            "R",
            (),
            {
                "remove_tab": lambda self, n: rail_calls.append(("remove", n)) or "b",
                "set_tab_disabled": lambda self, n, d: rail_calls.append(("mute", n, d)),
            },
        )()
        ns.wk = type("W", (), {"busy": False})()
        ns.root = None
        ns._select_tab = lambda stem: setattr(ns, "_current", stem)
        ns._run_btn = type("B", (), {"config": lambda self, **k: rail_calls.append(("run_btn", k))})()
        ns._update_run_btn_state = App._update_run_btn_state.__get__(ns)
        ns._remove_tab = App._remove_tab.__get__(ns)
        ns._set_tab_muted = App._set_tab_muted.__get__(ns)
        ns._on_rail_context = App._on_rail_context.__get__(ns)
        return ns, rail_calls

    def _cap_menu(self, monkeypatch, captured):
        import tcm_gui.app as app_mod

        class CapMenu:
            def __init__(self, *a, **k):
                pass

            def add_command(self, label="", command=None):
                captured.append((label, command))

            def tk_popup(self, *a):
                pass

            def grab_release(self):
                pass

        monkeypatch.setattr(app_mod.tk, "Menu", CapMenu)
        return app_mod

    def test_menu_labels_remove_disable(self, monkeypatch):
        ns, _ = self._app_ns()
        captured: list = []
        app_mod = self._cap_menu(monkeypatch, captured)
        ns._on_rail_context("a", 0, 0)
        assert captured[0][0] == str(app_mod._S.get("rail_ctx.remove", "Remove"))
        assert captured[1][0] == str(app_mod._S.get("rail_ctx.disable", "Disable"))
        captured[0][1]()  # Remove → drops instantly
        assert "a" not in ns._tab_of

    def test_menu_label_flips_to_enable_when_muted(self, monkeypatch):
        ns, _ = self._app_ns()
        ns._disabled_tabs.add("a")
        captured: list = []
        app_mod = self._cap_menu(monkeypatch, captured)
        ns._on_rail_context("a", 0, 0)
        assert captured[0][0] == str(app_mod._S.get("rail_ctx.remove", "Remove"))
        assert captured[1][0] == str(app_mod._S.get("rail_ctx.enable", "Enable"))
        captured[1][1]()  # Enable → unmutes
        assert "a" not in ns._disabled_tabs

    def test_menu_suppressed_when_busy(self, monkeypatch):
        import tcm_gui.app as app_mod

        ns, _ = self._app_ns()
        ns.wk = type("W", (), {"busy": True})()
        called: list = []
        monkeypatch.setattr(app_mod.tk, "Menu", lambda *a, **k: called.append(1))
        ns._on_rail_context("a", 0, 0)
        assert not called

    def test_mute_skips_run_stems(self):
        ns, rail_calls = self._app_ns()
        ns._set_tab_muted("a", True)
        assert "a" in ns._disabled_tabs and ("mute", "a", True) in rail_calls
        assert [s for s in ns._pages if s not in ns._disabled_tabs] == ["b"]
        assert ("run_btn", {"state": "normal"}) in rail_calls  # one runnable left → Run on
        ns._set_tab_muted("b", True)
        assert ("run_btn", {"state": "disabled"}) in rail_calls  # all muted → Run off
        ns._set_tab_muted("a", False)
        assert [s for s in ns._pages if s not in ns._disabled_tabs] == ["a"]


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
        assert str(ns._overall_lbl.cget("foreground")) == tcm_gui.theme.CELL_DEFAULT_VAL_FG
        assert rail._disabled is True
        fn(False)
        assert str(ns._overall_lbl.cget("foreground")) == tcm_gui.theme.FG_DEFAULT
        assert rail._disabled is False
        ns._overall_lbl.destroy()
