"""``const.work_area`` / ``fit_to_workarea``: startup placement inside visible space."""

from __future__ import annotations

import tkinter as tk

import pytest

from tcm_gui import const


@pytest.fixture
def root():
    """Fresh Tk root (skips when no display is available)."""
    try:
        r = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    yield r
    r.destroy()


def _placed(root: tk.Tk) -> tuple[int, int, int, int, int, int]:
    """Map *root*, let the ``<Map>`` re-fit settle, return ``(w, h, x, y, dx, dy)``.

    ``dx``/``dy`` are the measured chrome px (side borders ×2 / title bar +
    bottom border) that ``fit_to_workarea`` subtracts from the work area.
    """
    root.deiconify()
    root.update()
    b = max(root.winfo_rootx() - root.winfo_x(), 0)
    dx, dy = 2 * b, max(root.winfo_rooty() - root.winfo_y(), 0) + b
    w, h, x, y = map(int, root.winfo_geometry().replace("x", "+").split("+"))
    return w, h, x, y, dx, dy


def test_clamps_oversized_window_inside_chrome(root: tk.Tk, monkeypatch: pytest.MonkeyPatch):
    """Size larger than work area → clamped to work area MINUS window chrome."""
    monkeypatch.setattr(const, "work_area", lambda _w: (100, 50, 900, 650))
    const.fit_to_workarea(root, 1100, 800)
    w, h, x, y, dx, dy = _placed(root)
    assert (w, h) == (800 - dx, 600 - dy)
    assert (x, y) == (100, 50)  # pre-map centered pos KEPT — position released


def test_centers_within_workarea(root: tk.Tk, monkeypatch: pytest.MonkeyPatch):
    """Fitting window → centered inside the taskbar-excluded area."""
    monkeypatch.setattr(const, "work_area", lambda _w: (100, 50, 900, 650))
    const.fit_to_workarea(root, 400, 300)
    w, h, x, y, _dx, _dy = _placed(root)
    assert (w, h, x, y) == (400, 300, 100 + (800 - 400) // 2, 50 + (600 - 300) // 2)


def test_no_maxsize_cap(root: tk.Tk, monkeypatch: pytest.MonkeyPatch):
    """No ``maxsize`` call: it feeds WM_GETMINMAXINFO → Windows edge-drag/
    Aero-Snap snaps the window back mid-drag (bottom below the screen)."""
    monkeypatch.setattr(const, "work_area", lambda _w: (0, 0, 1024, 768))
    orig, calls = root.maxsize, []
    root.maxsize = lambda *a: (calls.append(a), orig(*a))[1]  # type: ignore[method-assign]
    try:
        const.fit_to_workarea(root, 400, 300)
        _placed(root)  # <Map> recap pass included
    finally:
        del root.maxsize
    assert calls == []


def test_recap_releases_position(root: tk.Tk, monkeypatch: pytest.MonkeyPatch):
    """Final ``<Map>`` re-fit writes SIZE-ONLY geometry — a stored ``+x+y``
    would make Tk teleport the window back to it on every later resize."""
    monkeypatch.setattr(const, "work_area", lambda _w: (0, 0, 1024, 768))
    orig, calls = root.geometry, []

    def spy(g=None):
        calls.append(g)
        return orig(g)

    root.geometry = spy  # type: ignore[method-assign]
    try:
        const.fit_to_workarea(root, 400, 300)
        _placed(root)  # drive <Map> → after_idle recap
    finally:
        del root.geometry
    assert calls and calls[-1] and "+" not in calls[-1]
    assert any("+" in c for c in calls[:-1])  # initial pass DID center


def test_nudge_window_above_top(root: tk.Tk):
    """``nudge_window`` shifts freely — incl. parking the window off-top."""
    root.geometry("400x300+50+50")
    root.update()
    const.nudge_window(root, -80, -120)
    root.update()
    assert (root.winfo_x(), root.winfo_y()) == (-30, -70)


def test_nudge_after_position_release(root: tk.Tk):
    """Regression: once position was released (size-only geometry), the
    ``wm_geometry`` string carries NO ``+x+y`` — the nudge must read the
    live ``winfo_x/y`` instead of failing to parse it."""
    root.geometry("400x300+50+50")
    root.update()
    root.geometry("400x300")  # what fit_to_workarea's <Map> pass writes
    root.update()
    const.nudge_window(root, -80, -120)
    root.update()
    assert (root.winfo_x(), root.winfo_y()) == (-30, -70)
    # rapid repeats (key auto-repeat): each nudge flushes pending geometry
    # first, so consecutive steps accumulate instead of overwriting one step
    const.nudge_window(root, -10, -10)
    const.nudge_window(root, -10, -10)
    root.update()
    assert (root.winfo_x(), root.winfo_y()) == (-50, -90)


def test_workarea_sane(root: tk.Tk):
    """Real query returns an ordered non-empty rect on any platform."""
    left, top, right, bottom = const.work_area(root)
    assert 0 <= left < right <= root.winfo_screenwidth()
    assert 0 <= top < bottom <= root.winfo_screenheight()
