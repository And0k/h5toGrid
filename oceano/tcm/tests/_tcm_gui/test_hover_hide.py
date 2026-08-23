"""TDD tests for the collapsible stage progress row (§2) hover-hide behavior.

Requirements:
1. Root ``<Motion>`` with the live pointer over the stage widgets (bar or
   description) collapses the row — both grid_remove'd, overall caption
   re-centered across the full row; motion elsewhere never hides
2. Once hidden, stays hidden after the pointer leaves — restoration is
   exclusively programmatic: a stage-snapshot change (progress advance) in
   ``_poll_progress`` clears the hover flag
3. ``_poll_progress`` with unchanged snapshot does NOT restore while hovering
4. ``_poll_progress`` with ``tot > 0`` DOES schedule/show the row when not
   hovering
5. ``_show_prog_stage`` skips placement while ``_stage_hovering``
6. ``_error_active`` does NOT block hover-hide — the frozen ``tot == 0``
   snapshot keeps the error row hidden until a new scan/run changes it

Hide is driven entirely by root ``<Motion>`` with LIVE bounds — ``<Enter>``
bindings were removed because ``_poll_progress`` re-grids the row mid-motion
(pointer already inside), so ``<Enter>`` can never fire there.

Note: root is withdrawn (headless Tk), so ``winfo_ismapped()`` always returns False.
We check ``grid_info()`` instead — ``grid_remove()`` clears it to ``{}``.
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

    r = root
    for w in r.winfo_children():
        w.destroy()

    r.grid_rowconfigure(2, weight=2)
    r.grid_rowconfigure(3, weight=1)
    r.grid_columnconfigure(0, weight=1)

    # §2 status row — overall caption (col 0, weight=1) + stage widgets (cols 1–2)
    f1 = ttk.Frame(r)
    f1.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 0))
    f1.columnconfigure(0, weight=1)
    _overall_lbl = ttk.Label(f1, text="Default", anchor="center")
    _overall_lbl.grid(row=0, column=0, sticky="ew")
    _prog_stage = ttk.Progressbar(f1, mode="determinate", length=220)
    _prog_stage_text = ttk.Label(f1, text="Loading…", anchor="w", justify="left")

    from tcm_gui.app import App

    # _stage_shown is a property — must live on the class, not the instance.
    ns = type("NS", (), {"_stage_shown": App._stage_shown})()
    ns.root = r
    ns._prog_stage = _prog_stage
    ns._prog_stage_text = _prog_stage_text
    ns._overall_lbl = _overall_lbl
    ns._stage_hovering = False
    ns._status_hovering = False
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
    ns._on_status_motion = App._on_status_motion.__get__(ns)
    ns._pointer_inside = App._pointer_inside
    ns._hide_progress_widgets = App._hide_progress_widgets.__get__(ns)
    ns._show_stage_progress = App._show_stage_progress.__get__(ns)
    ns._hide_stage_progress = App._hide_stage_progress.__get__(ns)
    ns._poll_progress = App._poll_progress.__get__(ns)
    ns._show_prog_stage = App._show_prog_stage.__get__(ns)
    # Static methods — no self binding
    ns._translate_desc = lambda desc="": App._translate_desc(desc)
    ns._translate_scan_stage = lambda stage=None: App._translate_scan_stage(stage)
    ns._default_stage_text = lambda: "Default"

    # Simulate an active stage — row expanded (bar + text gridded, anchor="w").
    ns._show_stage_progress()

    for _ in range(20):
        r.update()

    return ns


def _is_gridded(widget):
    """True when widget is managed by grid (not grid_remove'd)."""
    return bool(widget.grid_info())


def _anchor(ns) -> str:
    """Current overall-label anchor — 'w' expanded, 'center' collapsed."""
    return str(ns._overall_lbl.cget("anchor"))


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
    """The row collapses under its own widgets only; only advance restores."""

    def test_motion_over_text_collapses_row(self):
        """Pointer over the stage description collapses the row + re-centers overall."""
        ns = _build_app_minimal(_mod._root)
        assert _is_gridded(ns._prog_stage), "premise: bar starts gridded"
        assert _is_gridded(ns._prog_stage_text), "premise: text starts gridded"
        _fire_motion_over(ns, ns._prog_stage_text)
        assert ns._stage_hovering is True
        assert not _is_gridded(ns._prog_stage), "bar should be grid-removed"
        assert not _is_gridded(ns._prog_stage_text), "text should be grid-removed"
        assert _anchor(ns) == "center", "overall caption must re-center when collapsed"

    def test_motion_over_bar_collapses_row(self):
        """Pointer over the bar collapses the row too — one group, one flag."""
        ns = _build_app_minimal(_mod._root)
        _fire_motion_over(ns, ns._prog_stage)
        assert ns._stage_hovering is True
        assert not _is_gridded(ns._prog_stage) and not _is_gridded(ns._prog_stage_text)
        assert _anchor(ns) == "center"

    def test_motion_elsewhere_keeps_row(self):
        """<Motion> anywhere NOT over the stage widgets leaves the row shown."""
        ns = _build_app_minimal(_mod._root)
        _fire_motion(ns, 0, 0)  # top-left corner — far from the row-1 widgets
        assert ns._stage_hovering is False, "motion elsewhere must not set hovering"
        assert _is_gridded(ns._prog_stage) and _is_gridded(ns._prog_stage_text)
        assert _anchor(ns) == "w"

    def test_motion_outside_keeps_hidden(self):
        """Pointer leave does NOT restore — hidden state persists until advance."""
        ns = _build_app_minimal(_mod._root)
        _fire_motion_over(ns, ns._prog_stage_text)
        assert ns._stage_hovering is True
        # Fire motion far outside (top-left corner) — must NOT restore
        _fire_motion(ns, 0, 0)
        assert ns._stage_hovering is True, "pointer leave must not end hover-hide"
        assert not _is_gridded(ns._prog_stage), "row must stay hidden after leave"

    def test_hide_progress_widgets_collapses_and_flags(self):
        """Edit/browse start collapses the row and sets both hover flags."""
        ns = _build_app_minimal(_mod._root)
        ns._hide_progress_widgets()
        assert ns._stage_hovering is True and ns._status_hovering is True
        assert not _is_gridded(ns._prog_stage) and not _is_gridded(ns._prog_stage_text)
        assert _anchor(ns) == "center"

    def test_error_hides_and_restores_only_on_advance(self):
        """Error row: hides on hover, stays hidden on leave, restores on new stage."""
        ns = _build_app_minimal(_mod._root)
        ns._error_active = True
        ns.rt.progress_stage._snap = (0, 0, "")  # error state: frozen snapshot
        ns.rt.progress_overall._snap = (0, 0, "")
        ns._stage_last = (0, 0, "")
        _fire_motion_over(ns, ns._prog_stage_text)
        assert ns._stage_hovering is True, "_error_active must not block hover-hide"
        assert not _is_gridded(ns._prog_stage), "error row should hide on hover"
        # Leave + poll with unchanged snapshot → stays hidden
        _fire_motion(ns, 0, 0)
        ns._poll_progress()
        assert not _is_gridded(ns._prog_stage), "frozen snapshot must not restore"
        # New scan advances the snapshot → poll clears hovering and restores
        ns.rt.progress_stage._snap = (2, 5, "stage.discovering")
        ns._poll_progress()
        assert ns._stage_hovering is False
        assert ns._prog_show_job is not None or _is_gridded(ns._prog_stage), (
            "advanced snapshot must restore the row"
        )

    def test_poll_progress_no_show_while_hovering(self):
        """Unchanged snapshot: poll does NOT restore while hovering."""
        ns = _build_app_minimal(_mod._root)
        ns.rt.progress_stage._snap = (5, 10, "stage.discovering")
        ns.rt.progress_overall._snap = (0, 0, "")
        ns._poll_progress()  # establishes _stage_last, row stays visible
        _fire_motion_over(ns, ns._prog_stage_text)  # collapses the row
        ns._poll_progress()  # same snapshot → no restore
        assert not _is_gridded(ns._prog_stage), "row should NOT be restored while hovering without advance"

    def test_poll_progress_shows_when_not_hovering(self):
        """_poll_progress with tot>0 DOES schedule the row when not hovering."""
        ns = _build_app_minimal(_mod._root)
        ns._stage_hovering = False
        ns._hide_stage_progress()
        ns.rt.progress_stage._snap = (5, 10, "stage.discovering")
        ns.rt.progress_overall._snap = (0, 0, "")
        ns._poll_progress()
        assert ns._prog_show_job is not None or _is_gridded(ns._prog_stage), (
            "row should be scheduled or shown when not hovering and tot>0"
        )

    def test_show_prog_stage_skips_while_hovering(self):
        """_show_prog_stage does NOT grid when _stage_hovering."""
        ns = _build_app_minimal(_mod._root)
        ns._stage_hovering = True
        ns._hide_stage_progress()
        ns.rt.progress_stage._snap = (5, 10, "test")
        ns._prog_show_job = None
        ns._show_prog_stage()
        assert not _is_gridded(ns._prog_stage), "_show_prog_stage should skip placement while _stage_hovering"

    def test_show_prog_stage_shows_when_not_hovering(self):
        """_show_prog_stage grids the row (and left-aligns overall) when not hovering."""
        ns = _build_app_minimal(_mod._root)
        ns._stage_hovering = False
        ns._hide_stage_progress()
        ns.rt.progress_stage._snap = (5, 10, "test")
        ns._prog_show_job = None
        ns._show_prog_stage()
        assert _is_gridded(ns._prog_stage) and _is_gridded(ns._prog_stage_text)
        assert _anchor(ns) == "w", "expanded row must left-align the overall caption"

    def test_full_cycle_hide_leave_advance_restore(self):
        """Full cycle: hide on hover → leave keeps hidden → advance + poll restores."""
        ns = _build_app_minimal(_mod._root)
        ns.rt.progress_stage._snap = (5, 10, "stage.discovering")
        ns.rt.progress_overall._snap = (0, 0, "")
        ns._poll_progress()  # establishes _stage_last
        # Motion over the bar → collapse
        _fire_motion_over(ns, ns._prog_stage)
        assert ns._stage_hovering is True
        assert not _is_gridded(ns._prog_stage)
        # Leave + poll with the same snapshot → stays hidden
        _fire_motion(ns, 0, 0)
        ns._poll_progress()
        assert not _is_gridded(ns._prog_stage), "row should stay hidden without advance"
        # Advance → poll restores
        ns.rt.progress_stage._snap = (6, 10, "stage.discovering")
        ns._poll_progress()
        assert ns._stage_hovering is False
        assert ns._prog_show_job is not None or _is_gridded(ns._prog_stage), (
            "row should be restored after advance"
        )
