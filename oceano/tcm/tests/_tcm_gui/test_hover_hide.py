"""TDD tests for the hover-hide behavior of stage status + progress floater.

Requirements:
1. ``<Enter>`` on ``_prog_status`` or ``_prog_floater`` → both hidden (grid_remove + place_forget)
2. Root ``<Motion>`` outside the recorded bbox → ``_status_hovering = False`` (poll restores)
3. ``_poll_progress`` with ``tot > 0`` does NOT show floater while ``_status_hovering``
4. ``_poll_progress`` with ``tot > 0`` DOES show floater when not hovering
5. ``_show_prog_floater`` skips placement while ``_status_hovering``
6. ``_error_active`` does NOT block hover-hide (error text is in the log)

Note: root is withdrawn (headless Tk), so ``winfo_ismapped()`` always returns False.
We check ``grid_info()`` / ``place_info()`` instead — ``grid_remove()`` clears
``grid_info()`` to ``{}``; ``place_forget()`` clears ``place_info()`` to ``{}``.
"""

from __future__ import annotations

import sys

import pytest
import tkinter as tk

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


class _FakeUI:
    """Minimal UIScale stand-in — provides font() used by MarkdownLabel."""

    def font(self):
        import tkinter.font as tkfont

        return tkfont.Font(family="TkDefaultFont", size=10)


class _FakeProgressStage:
    def __init__(self, cur=0, tot=0, desc=""):
        self._snap = (cur, tot, desc)

    def snapshot(self):
        return self._snap

    def consume_clear(self):
        return False

    def set(self, cur, tot, desc):
        self._snap = (cur, tot, desc)


class _FakeProgressOverall:
    def __init__(self, cur=0, tot=0, desc=""):
        self._snap = (cur, tot, desc)

    def snapshot(self):
        return self._snap

    def set(self, cur, tot, desc):
        self._snap = (cur, tot, desc)


class _FakeProgressBank:
    def snapshot_all(self):
        return {}


class _FakeRT:
    def __init__(self):
        self.progress_stage = _FakeProgressStage()
        self.progress_overall = _FakeProgressOverall()
        self.progress_bank = _FakeProgressBank()
        from queue import Queue

        self.log_queue = Queue()
        self.result_queue = Queue()
        self.pause_gate = type("G", (), {"paused": False})()


class _FakeWorker:
    busy = False


def _build_app_minimal(root):
    """Build a minimal App-like object with the _build layout + poll methods.

    Avoids the full App.__init__ (which starts threads, loads configs, etc).
    Instead constructs just the widgets and binds the methods we test.
    """
    import tcm_gui.theme
    from tcm_gui.const import configure_ui

    # Skip apply_theme_defaults — its DWM titlebar call fails on a stale HWND
    # after previous test modules destroyed their Tk roots.
    with __import__("contextlib").suppress(Exception):
        tcm_gui.theme.apply_theme_defaults(root)
    with __import__("contextlib").suppress(Exception):
        configure_ui(root)
    root.geometry("1100x800")

    from tkinter import ttk
    import tcm_gui.theme as theme

    r = root
    for w in r.winfo_children():
        w.destroy()

    r.grid_rowconfigure(2, weight=2)
    r.grid_rowconfigure(3, weight=1)
    r.grid_columnconfigure(0, weight=1)

    # §2 status row
    f1 = ttk.Frame(r)
    f1.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 0))
    f1.columnconfigure(0, weight=1)
    _overall_lbl = ttk.Label(f1, text="Default")
    _overall_lbl.grid(row=0, column=0, sticky="w")
    _prog_status = ttk.Label(f1, text="Loading…", anchor="e")
    _prog_status.grid(row=0, column=1, sticky="e", padx=(8, 0))

    # §6b floater
    _bg = theme.FRAME_BG_FALLBACK
    _prog_floater = tk.Frame(r, bg=_bg, bd=0, highlightthickness=0)
    _prog_stage_text = tk.Label(
        _prog_floater, text="", anchor="e", justify="right", bg=_bg, fg=theme.FG_DEFAULT, bd=0
    )
    _prog_stage_text.pack(side="top", anchor="e", fill="x")
    from tkinter import ttk as ttk_mod

    _prog_stage = ttk_mod.Progressbar(_prog_floater, mode="determinate", length=220)
    _prog_stage.pack(side="bottom", fill="x")

    # Place floater initially (simulate processing active)
    _prog_floater.place(relx=1.0, rely=1.0, anchor="se", x=-8, y=-4)

    # Build a namespace with the methods and state
    ns = type("NS", (), {})()
    ns.root = r
    ns._prog_status = _prog_status
    ns._prog_floater = _prog_floater
    ns._prog_stage_text = _prog_stage_text
    ns._prog_stage = _prog_stage
    ns._overall_lbl = _overall_lbl
    ns._status_hovering = False
    ns._status_bbox = None
    ns._error_active = False
    ns._initial_scan = False
    ns._prog_show_job = None
    ns._cfg_state = type("S", (), {"name": "DEFAULT"})()  # ScanStage.DEFAULT stand-in
    ns._cfg_detail = ""
    ns.rt = _FakeRT()
    ns.wk = _FakeWorker()
    ns._any_hovering = False
    ns._status_lbl = type("S", (), {"set_text": lambda self, t, raw=False: None})()
    ns.ui = _FakeUI()

    # Bind methods from App
    from tcm_gui.app import App

    ns._on_status_enter = App._on_status_enter.__get__(ns)
    ns._status_bounds = App._status_bounds.__get__(ns)
    ns._on_motion_check_hover = App._on_motion_check_hover.__get__(ns)
    ns._poll_progress = App._poll_progress.__get__(ns)
    ns._show_prog_floater = App._show_prog_floater.__get__(ns)
    # Static methods — no self binding
    ns._translate_desc = lambda desc="": App._translate_desc(desc)
    ns._translate_scan_stage = lambda stage=None: App._translate_scan_stage(stage)

    for _ in range(20):
        r.update()

    return ns


def _is_gridded(widget):
    """True when widget is managed by grid (not grid_remove'd)."""
    return bool(widget.grid_info())


def _is_placed(widget):
    """True when widget is managed by place (not place_forget'd)."""
    return bool(widget.place_info())


def _fire_enter(ns):
    """Simulate <Enter> on any status widget — call _on_status_enter directly.

    event_generate('<Enter>') does not fire Python callbacks on a withdrawn
    root, so we invoke the handler directly with a minimal event-like object.
    """
    ns._on_status_enter(type("E", (), {}))


def _fire_motion(ns, x_root, y_root):
    """Simulate root <Motion> at screen coordinates (x_root, y_root)."""
    event = type("E", (), {"x_root": x_root, "y_root": y_root, "x": 0, "y": 0, "widget": ns.root})
    ns._on_motion_check_hover(event)


class TestHoverHide:
    """``<Enter>`` on status widgets hides both; ``<Motion>`` outside restores."""

    def test_enter_prog_status_hides_both(self):
        """<Enter> on _prog_status → grid_remove + place_forget, _status_hovering=True."""
        ns = _build_app_minimal(_mod._root)
        assert _is_gridded(ns._prog_status), "premise: _prog_status starts gridded"
        assert _is_placed(ns._prog_floater), "premise: _prog_floater starts placed"
        _fire_enter(ns)
        assert ns._status_hovering is True, "_status_hovering should be True after <Enter>"
        assert not _is_gridded(ns._prog_status), "_prog_status should be grid-removed after <Enter>"
        assert not _is_placed(ns._prog_floater), "_prog_floater should be place-forgotten after <Enter>"

    def test_enter_floater_hides_both(self):
        """<Enter> on _prog_floater → grid_remove + place_forget, _status_hovering=True."""
        ns = _build_app_minimal(_mod._root)
        _fire_enter(ns)
        assert ns._status_hovering is True, "_status_hovering should be True after <Enter> on floater"
        assert not _is_gridded(ns._prog_status), "_prog_status should be grid-removed"
        assert not _is_placed(ns._prog_floater), "_prog_floater should be place-forgotten"

    def test_motion_outside_bbox_clears_flag(self):
        """Root <Motion> outside the recorded bbox → _status_hovering=False."""
        ns = _build_app_minimal(_mod._root)
        _fire_enter(ns)
        assert ns._status_hovering is True
        # Fire motion far outside the bbox (top-left corner)
        _fire_motion(ns, 0, 0)
        assert ns._status_hovering is False, (
            "_status_hovering should be False after motion outside bbox"
        )

    def test_motion_inside_bbox_keeps_flag(self):
        """Root <Motion> inside the recorded bbox → _status_hovering stays True."""
        ns = _build_app_minimal(_mod._root)
        _fire_enter(ns)
        assert ns._status_hovering is True
        assert ns._status_bbox is not None
        x0, y0, x1, y1 = ns._status_bbox
        # Fire motion at the center of the bbox
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        _fire_motion(ns, cx, cy)
        assert ns._status_hovering is True, "_status_hovering should stay True inside bbox"

    def test_error_active_does_not_block_hide(self):
        """_error_active=True → <Enter> still hides both (error is in the log)."""
        ns = _build_app_minimal(_mod._root)
        ns._error_active = True
        _fire_enter(ns)
        assert ns._status_hovering is True, "_error_active should NOT block hover-hide"
        assert not _is_gridded(ns._prog_status), "_prog_status should be hidden even with _error_active"
        assert not _is_placed(ns._prog_floater), "_prog_floater should be hidden even with _error_active"

    def test_poll_progress_no_show_while_hovering(self):
        """_poll_progress with tot>0 does NOT show floater while hovering."""
        ns = _build_app_minimal(_mod._root)
        _fire_enter(ns)  # hides both, sets _status_hovering=True
        ns.rt.progress_stage._snap = (5, 10, "stage.discovering")
        ns.rt.progress_overall._snap = (0, 0, "")
        ns._poll_progress()
        assert not _is_placed(ns._prog_floater), (
            "floater should NOT be shown while _status_hovering is True"
        )
        assert not _is_gridded(ns._prog_status), (
            "_prog_status should NOT be gridded while _status_hovering is True"
        )

    def test_poll_progress_shows_when_not_hovering(self):
        """_poll_progress with tot>0 DOES show floater when not hovering."""
        ns = _build_app_minimal(_mod._root)
        ns._status_hovering = False
        ns._prog_floater.place_forget()
        ns.rt.progress_stage._snap = (5, 10, "stage.discovering")
        ns.rt.progress_overall._snap = (0, 0, "")
        ns._poll_progress()
        assert ns._prog_show_job is not None or _is_placed(ns._prog_floater), (
            "floater should be scheduled or shown when not hovering and tot>0"
        )
        assert _is_gridded(ns._prog_status), "_prog_status should be gridded when not hovering"

    def test_show_prog_floater_skips_while_hovering(self):
        """_show_prog_floater does NOT place when _status_hovering."""
        ns = _build_app_minimal(_mod._root)
        ns._status_hovering = True
        ns._prog_floater.place_forget()
        ns.rt.progress_stage._snap = (5, 10, "test")
        ns._prog_show_job = None
        ns._show_prog_floater()
        assert not _is_placed(ns._prog_floater), (
            "_show_prog_floater should skip placement while _status_hovering"
        )

    def test_show_prog_floater_shows_when_not_hovering(self):
        """_show_prog_floater places floater when not hovering."""
        ns = _build_app_minimal(_mod._root)
        ns._status_hovering = False
        ns._prog_floater.place_forget()
        ns.rt.progress_stage._snap = (5, 10, "test")
        ns._prog_show_job = None
        ns._show_prog_floater()
        assert _is_placed(ns._prog_floater), (
            "_show_prog_floater should place floater when not hovering"
        )

    def test_full_cycle_enter_then_motion_then_poll(self):
        """Full cycle: Enter → hide → motion outside → poll restores."""
        ns = _build_app_minimal(_mod._root)
        # Enter → hide
        _fire_enter(ns)
        assert ns._status_hovering is True
        assert not _is_placed(ns._prog_floater)
        # Simulate processing active
        ns.rt.progress_stage._snap = (5, 10, "stage.discovering")
        ns.rt.progress_overall._snap = (0, 0, "")
        # Poll while hovering → no show
        ns._poll_progress()
        assert not _is_placed(ns._prog_floater), "floater should stay hidden while hovering"
        # Motion outside → flag clears
        _fire_motion(ns, 0, 0)
        assert ns._status_hovering is False
        # Poll → restores
        ns._poll_progress()
        assert _is_gridded(ns._prog_status), "_prog_status should be restored after mouse leaves"
