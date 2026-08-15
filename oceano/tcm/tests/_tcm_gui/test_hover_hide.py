"""TDD tests for the hover-hide behavior of stage status + progress floater.

Requirements:
1. Root ``<Motion>`` with the live pointer over a mapped status widget hides
   THAT widget only — ``_prog_status`` and the floater hide independently;
   shown always, motion elsewhere never hides
2. Once hidden, stays hidden after the pointer leaves — restoration is
   exclusively programmatic: a stage-snapshot change (progress advance) in
   ``_poll_progress`` clears both hover flags
3. ``_poll_progress`` with unchanged snapshot does NOT restore while hovering
4. ``_poll_progress`` with ``tot > 0`` DOES show floater when not hovering
5. ``_show_prog_floater`` skips placement while ``_floater_hovering``
6. ``_error_active`` does NOT block hover-hide — the frozen ``tot == 0``
   snapshot keeps the error floater hidden until a new scan/run changes it

Hide is driven entirely by root ``<Motion>`` with LIVE bounds — ``<Enter>``
bindings were removed because ``_poll_progress`` re-shows/lifts the floater
mid-motion (pointer already inside), so ``<Enter>`` can never fire there.

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
    ns._floater_hovering = False
    ns._stage_last = (0, 0, "")
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

    ns._on_status_motion = App._on_status_motion.__get__(ns)
    ns._pointer_inside = App._pointer_inside
    ns._lift_status_z = App._lift_status_z.__get__(ns)
    ns._place_floater = App._place_floater.__get__(ns)
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


def _fire_motion(ns, x_root, y_root):
    """Simulate root <Motion> at screen coordinates (x_root, y_root)."""
    event = type("E", (), {"x_root": x_root, "y_root": y_root, "x": 0, "y": 0, "widget": ns.root})
    ns._on_status_motion(event)


def _point_in(w):
    """Center of *w*'s live bbox (must have real area)."""
    x0, y0 = w.winfo_rootx(), w.winfo_rooty()
    return (2 * x0 + w.winfo_width()) // 2, (2 * y0 + w.winfo_height()) // 2


def _fire_motion_over(ns, w):
    """Simulate root <Motion> with the pointer over widget *w*."""
    _fire_motion(ns, *_point_in(w))


class TestHoverHide:
    """Each widget hides only under its own pointer; only advance restores."""

    def test_motion_over_status_hides_only_status(self):
        """Pointer over _prog_status hides it only — the floater stays visible."""
        ns = _build_app_minimal(_mod._root)
        assert _is_gridded(ns._prog_status), "premise: _prog_status starts gridded"
        assert _is_placed(ns._prog_floater), "premise: _prog_floater starts placed"
        _fire_motion_over(ns, ns._prog_status)
        assert ns._status_hovering is True
        assert not _is_gridded(ns._prog_status), "_prog_status should be grid-removed"
        assert ns._floater_hovering is False, "floater flag must stay clear"
        assert _is_placed(ns._prog_floater), "floater must NOT hide on status hover"

    def test_motion_over_floater_hides_only_floater(self):
        """Pointer over the floater hides it only — _prog_status stays visible."""
        ns = _build_app_minimal(_mod._root)
        _fire_motion_over(ns, ns._prog_floater)
        assert ns._floater_hovering is True
        assert not _is_placed(ns._prog_floater), "floater should be place-forgotten"
        assert ns._status_hovering is False, "status flag must stay clear"
        assert _is_gridded(ns._prog_status), "_prog_status must NOT hide on floater hover"

    def test_motion_elsewhere_keeps_both_visible(self):
        """<Motion> anywhere NOT over the status widgets leaves both shown."""
        ns = _build_app_minimal(_mod._root)
        _fire_motion(ns, 0, 0)  # top-left corner — far from the bottom-right floater
        assert ns._status_hovering is False, "motion elsewhere must not set hovering"
        assert _is_gridded(ns._prog_status), "_prog_status must stay visible on other motion"
        assert _is_placed(ns._prog_floater), "_prog_floater must stay visible on other motion"

    def test_motion_outside_keeps_hidden(self):
        """Pointer leave does NOT restore — hidden state persists until advance."""
        ns = _build_app_minimal(_mod._root)
        _fire_motion_over(ns, ns._prog_floater)
        assert ns._floater_hovering is True
        # Fire motion far outside (top-left corner) — must NOT restore
        _fire_motion(ns, 0, 0)
        assert ns._floater_hovering is True, "pointer leave must not end hover-hide"
        assert not _is_placed(ns._prog_floater), "floater must stay hidden after leave"

    def test_error_hides_and_restores_only_on_advance(self):
        """Error floater: hides on hover, stays hidden on leave, restores on new stage."""
        ns = _build_app_minimal(_mod._root)
        ns._error_active = True
        ns.rt.progress_stage._snap = (0, 0, "")  # error state: frozen snapshot
        ns.rt.progress_overall._snap = (0, 0, "")
        ns._stage_last = (0, 0, "")
        _fire_motion_over(ns, ns._prog_floater)
        assert ns._floater_hovering is True, "_error_active must not block hover-hide"
        assert not _is_placed(ns._prog_floater), "error floater should hide on hover"
        # Leave + poll with unchanged snapshot → stays hidden
        _fire_motion(ns, 0, 0)
        ns._poll_progress()
        assert not _is_placed(ns._prog_floater), "frozen snapshot must not restore"
        # New scan advances the snapshot → poll clears hovering and restores
        ns.rt.progress_stage._snap = (2, 5, "stage.discovering")
        ns._poll_progress()
        assert ns._floater_hovering is False
        assert ns._prog_show_job is not None or _is_placed(ns._prog_floater), (
            "advanced snapshot must restore the floater"
        )

    def test_motion_lift_keeps_error_floater_on_top(self):
        """_error_active → motion z-order lift re-lifts the floater above the wide tooltip."""
        ns = _build_app_minimal(_mod._root)
        ns._error_active = True
        order = []
        ns._status_lbl.lift = lambda: order.append("status")
        ns._prog_floater.lift = lambda: order.append("floater")
        ns._lift_status_z(type("E", (), {}))
        assert order == ["status", "floater"], "error floater must stay above the tooltip"

    def test_motion_lift_status_only_when_no_error(self):
        """No error → motion lifts only the status label (poll re-lifts the floater)."""
        ns = _build_app_minimal(_mod._root)
        order = []
        ns._status_lbl.lift = lambda: order.append("status")
        ns._prog_floater.lift = lambda: order.append("floater")
        ns._lift_status_z(type("E", (), {}))
        assert order == ["status"], "without error only the status label is lifted on motion"

    def test_poll_progress_no_show_while_hovering(self):
        """Unchanged snapshot: poll does NOT restore while hovering."""
        ns = _build_app_minimal(_mod._root)
        ns.rt.progress_stage._snap = (5, 10, "stage.discovering")
        ns.rt.progress_overall._snap = (0, 0, "")
        ns._poll_progress()  # establishes _stage_last, floater visible
        _fire_motion_over(ns, ns._prog_floater)  # hides the floater only
        ns._poll_progress()  # same snapshot → no restore
        assert not _is_placed(ns._prog_floater), (
            "floater should NOT be restored while hovering without advance"
        )
        assert _is_gridded(ns._prog_status), (
            "_prog_status hides independently — must stay visible here"
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
        """_show_prog_floater does NOT place when _floater_hovering."""
        ns = _build_app_minimal(_mod._root)
        ns._floater_hovering = True
        ns._prog_floater.place_forget()
        ns.rt.progress_stage._snap = (5, 10, "test")
        ns._prog_show_job = None
        ns._show_prog_floater()
        assert not _is_placed(ns._prog_floater), (
            "_show_prog_floater should skip placement while _floater_hovering"
        )

    def test_show_prog_floater_shows_when_not_hovering(self):
        """_show_prog_floater places floater when not hovering."""
        ns = _build_app_minimal(_mod._root)
        ns._floater_hovering = False
        ns._prog_floater.place_forget()
        ns.rt.progress_stage._snap = (5, 10, "test")
        ns._prog_show_job = None
        ns._show_prog_floater()
        assert _is_placed(ns._prog_floater), (
            "_show_prog_floater should place floater when not hovering"
        )

    def test_full_cycle_hide_leave_advance_restore(self):
        """Full cycle: hide on hover → leave keeps hidden → advance + poll restores."""
        ns = _build_app_minimal(_mod._root)
        ns.rt.progress_stage._snap = (5, 10, "stage.discovering")
        ns.rt.progress_overall._snap = (0, 0, "")
        ns._poll_progress()  # establishes _stage_last
        # Motion over the floater → hide it only
        _fire_motion_over(ns, ns._prog_floater)
        assert ns._floater_hovering is True
        assert not _is_placed(ns._prog_floater)
        # Leave + poll with the same snapshot → stays hidden
        _fire_motion(ns, 0, 0)
        ns._poll_progress()
        assert not _is_placed(ns._prog_floater), "floater should stay hidden without advance"
        # Advance → poll restores
        ns.rt.progress_stage._snap = (6, 10, "stage.discovering")
        ns._poll_progress()
        assert ns._floater_hovering is False
        assert ns._prog_show_job is not None or _is_placed(ns._prog_floater), (
            "floater should be restored after advance"
        )
