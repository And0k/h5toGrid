"""Immutable user settings and scaling utilities for tcm_gui.

:data:`UI_SCALE`, :data:`FONT_SCALE`, :data:`TTK_THEME`, :data:`COLOR_MODE`
are the application's configuration knobs — all ``Final``, all defaulting to
"no change from platform / theme defaults".

:class:`UIScale` sets ``tk scaling`` to ``platform_scaling × UI_SCALE``
so that ALL Tk geometry scales uniformly.  :data:`FONT_SCALE` is an
additional multiplier on named fonts.  :func:`configure_ui` selects the
ttk theme.

Mutable runtime state (colors, :data:`THEME`, :data:`TAG_COLORS`,
:data:`widget_meta`) lives in :mod:`tcm_gui.theme`.

"""

from __future__ import annotations

import sys
import tkinter as tk
import tkinter.font as tkfont
from collections.abc import Callable
from contextlib import suppress
from tkinter import ttk
from typing import Final

# ── UI scaling ───────────────────────────────────────────────────────────────
# Four independent user-facing settings — all default to "no change":
#
#   UI_SCALE    — application geometry (sets tk scaling = platform × UI_SCALE)
#   FONT_SCALE  — typography (additional multiplier on named fonts)
#   TTK_THEME   — ttk element rendering ("native" or "clam")
#   COLOR_MODE  — dark/light palette ("auto", "light", or "dark")

UI_SCALE: Final[float] = 1.5  # application geometry multiplier
FONT_SCALE: Final[float] = 1.0  # typography multiplier (1.0 = platform default)
TTK_THEME: Final[str] = "native"  # "native" (platform default) or "clam"
COLOR_MODE: Final[str] = "auto"  # "auto" (OS detection), "light", or "dark"
LANG: Final[str] = "auto"  # "auto" (OS locale), or explicit: "en", "ru", etc.

# Virtual key code for the physical ``C`` key (platform-specific).
# Used to detect Ctrl+C regardless of the active keyboard layout: on non-Latin
# layouts (Cyrillic, Greek, …) the physical ``C`` key produces a different
# character, so Tk's ``<<Copy>>`` virtual event never fires from its
# ``<Control-Key-c>`` mapping.  We detect the physical key by ``keycode`` and
# generate ``<<Copy>>`` ourselves.  See :meth:`App._on_ctrl_keypress``.
#
# Tk's ``keycode`` is the platform keycode (Windows VK, X11 keycode, macOS
# key ID) — NOT a Tk abstraction, so values differ per windowing system:
#   - win32:  VK_C = 0x43
#   - x11:    standard PC ``C`` key = 54
#   - aqua:   kVK_ANSI_C = 0x08
# Mapping uses ``sys.platform`` (Tk's ``windowingsystem`` maps to the same
# ``win32``/``linux``/``darwin`` keys for the platforms we target).
_VK_C_BY_PLATFORM: Final[dict[str, int]] = {"win32": 0x43, "darwin": 0x08, "linux": 54}
VK_C: Final[int] = _VK_C_BY_PLATFORM.get(sys.platform, 0x43)

# Saved once on first UIScale() call — prevents compounding when multiple
# UIScale instances are created (e.g. tests creating fresh roots).
_platform_scaling: float | None = None


class UIScale:
    """Application-level geometry and typography scaling.

    ``ui_scale`` scales ALL Tk geometry by setting ``tk scaling`` to
    ``platform_scaling × ui_scale``.  This affects every widget, padding,
    font, and measurement uniformly — no per-widget configuration needed.

    ``font_scale`` is an additional multiplier on named fonts only
    (``TkDefaultFont``, ``TkTextFont``, ``TkMenuFont``, ``TkHeadingFont``).
    ``font_scale=1.0`` is a no-op.

    Call :meth:`font` to get a scaled copy for custom widgets (e.g.
    :class:`MarkdownLabel`).
    """

    def __init__(
        self,
        root: tk.Misc,
        *,
        ui_scale: float = UI_SCALE,
        font_scale: float = FONT_SCALE,
    ) -> None:
        global _platform_scaling
        self.root = root
        self.ui_scale = ui_scale
        self.font_scale = font_scale
        # Save the original platform scaling once — prevents compounding
        # when multiple UIScale instances are created in the same process.
        if _platform_scaling is None:
            _platform_scaling = root.tk.call("tk", "scaling")
        root.tk.call("tk", "scaling", _platform_scaling * ui_scale)
        # FONT_SCALE is an additional multiplier on named fonts.
        if font_scale != 1.0:
            for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
                with suppress(tk.TclError):
                    base = tkfont.nametofont(name)
                    size = base.cget("size")
                    base.configure(
                        size=round(size * font_scale) if size > 0 else -round(abs(size) * font_scale)
                    )

    def font(
        self,
        base: str = "TkDefaultFont",
        *,
        size_diff: int = 0,
    ) -> tkfont.Font:
        """Copy a named font (already scaled by ``tk scaling`` and ``FONT_SCALE``).

        ``size_diff`` is applied **after** the copy (e.g. ``size_diff=2``
        means "scaled font + 2 points").
        """
        result = tkfont.nametofont(base).copy()
        if size_diff:
            result.configure(size=result.cget("size") + size_diff)
        return result

    def set_font(
        self,
        *widgets: tk.Misc,
        base: str = "TkDefaultFont",
        size_diff: int = 0,
    ) -> None:
        """Apply a scaled font copy to each *widget* (each gets its own copy).

        Centralizes the ``font=self.ui.font()`` pattern so callers don't
        repeat the scaling logic.  Every widget receives a private
        :class:`~tkinter.font.Font` copy — safe for per-widget mutation
        (e.g. ``MarkdownLabel.fit_to_height``).
        """
        for w in widgets:
            w.configure(font=self.font(base, size_diff=size_diff))


def configure_ui(root: tk.Misc) -> ttk.Style:
    """Select ttk theme based on :data:`TTK_THEME` policy.

    ``"native"`` → platform default (``vista`` on Windows, system default otherwise).
    ``"clam"`` → Tk's cross-platform theme.
    """
    style = ttk.Style(root)
    if TTK_THEME == "clam":
        style.theme_use("clam")
    elif sys.platform == "win32":
        style.theme_use("vista")
    return style


# ── screen work area / window placement ─────────────────────────────────────


def work_area(widget: tk.Misc) -> tuple[int, int, int, int]:
    """Work area ``(left, top, right, bottom)`` of the monitor holding *widget*.

    Per-monitor (``MonitorFromWindow`` + ``GetMonitorInfoW``), taskbar excluded
    — ``SPI_GETWORKAREA`` only knows the PRIMARY monitor, wrong on multi-monitor
    setups.  This process is DPI-unaware, so both Tk and WinAPI speak the same
    virtualized coordinates.
    """
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class _MONITORINFO(ctypes.Structure):
                _fields_ = [
                    ("cbSize", wintypes.DWORD),
                    ("rcMonitor", wintypes.RECT),
                    ("rcWork", wintypes.RECT),
                    ("dwFlags", wintypes.DWORD),
                ]

            user32 = ctypes.windll.user32
            hwnd = user32.GetAncestor(widget.winfo_id(), 2) or widget.winfo_id()  # GA_ROOT
            hmon = user32.MonitorFromWindow(hwnd, 2)  # MONITOR_DEFAULTTONEAREST
            info = _MONITORINFO(cbSize=ctypes.sizeof(_MONITORINFO))
            if user32.GetMonitorInfoW(hmon, ctypes.byref(info)):
                r = info.rcWork
                return r.left, r.top, r.right, r.bottom
        except OSError:
            pass
    return 0, 0, widget.winfo_screenwidth(), widget.winfo_screenheight()


def fit_to_workarea(root: tk.Misc, width: int, height: int) -> None:
    """Place *root* in the visible work area ONCE at startup — no handler left
    behind, so the user's mouse resize/move is never touched again.

    ``geometry`` sets the CLIENT rect while the work area bounds the OUTER
    window, so we map first, measure the chrome (title bar + borders) and
    re-clamp the client size to fit.  The last call writes SIZE-ONLY geometry:
    an explicit ``+x+y`` stays stored in Tk and is re-applied on every later
    content resize — the window would jump back after the user drags it.
    """
    left, top, right, bottom = work_area(root)

    def _center(_dx: int, _dy: int) -> tuple[int, int, int, int]:
        w = max(min(width, right - left - _dx), 1)
        h = max(min(height, bottom - top - _dy), 1)
        _x, _y = left + (right - left - w) // 2, top + (bottom - top - h) // 2
        return w, h, _x, _y

    w, h, x, y = _center(0, 0)  # chrome unknown pre-map — first guess
    root.geometry(f"{w}x{h}+{x}+{y}")
    root.deiconify()
    root.update()  # map → chrome measurable

    b = max(root.winfo_rootx() - root.winfo_x(), 0)  # side border px
    dy = max(root.winfo_rooty() - root.winfo_y(), 0) + b  # title bar + border
    w, h, _x, _y = _center(2 * b, dy)
    root.geometry(f"{w}x{h}+{_x}+{_y}")  # chrome-accurate final placement
    root.geometry(f"{w}x{h}")  # size-only → release position: free dragging


def nudge_window(win: tk.Misc, dx: int, dy: int) -> None:
    """Shift *win* by ``(dx, dy)`` px from its current position.

    ``update_idletasks`` first: Tk applies ``geometry`` requests in an idle
    pass, so without the flush rapid key repeats all read the PRE-MOVE
    position and the window stops after one step.  Live ``winfo_x/y`` then
    match the ``+x+y`` coordinate space exactly (verified on Windows).

    A mouse drag cannot carry the title bar above the screen top — the shell
    clamps every app's interactive drags there.  This can, so keyboard
    nudges (Alt+Arrows) park the window partly off-screen.
    """
    win.update_idletasks()
    win.geometry(f"+{win.winfo_x() + dx}+{win.winfo_y() + dy}")


# ── font helpers ────────────────────────────────────────────────────────────

def tk_font_family(widget: tk.Text) -> str:
    """Extract the real family name from the widget font (e.g. 'Consolas').

    ``cget('font')`` may return a Tk font spec like ``'Consolas 11'`` or a
    named font like ``'font1'`` (TkDefaultFont etc.).  Named fonts are
    resolved to their real family via ``tk.font.nametofont().actual()``.
    Falls back to ``'Consolas'`` if extraction fails.
    """
    spec = widget.cget("font") or ""
    token = (str(spec).split() or ["Consolas"])[0]
    # Named Tk fonts (e.g. 'font1', 'TkDefaultFont') aren't real family names.
    # Resolve via nametofont → actual['family'] to get the OS font name.
    try:
        import tkinter.font as tkfont

        return tkfont.nametofont(token).actual()["family"]
    except Exception:
        return token


# ── widget metadata registry ─────────────────────────────────────────────────
# Tkinter widgets have no built-in metadata store (no ``widget.tooltip=``).
# This dict is the central runtime registry for status-bar captions, future
# tooltip text, and translation keys — keyed by widget instance.
#
# Alternative key for non-widget rows (e.g. tksheet treeview items that lack
# a real tk widget): the ``path`` string from the row's meta dict.  Both keys
# coexist in the same dict — lookup tries widget first, then string identifier.
#
# Field values are either:
#   * ``str``         — static (tooltip text, fixed status caption);
#   * ``Callable[[]]``— dynamic status, evaluated at read time so it sees the
#                       current state (busy / paused) AND the current language
#                       (via STR); stored as the binding, never pre-resolved.


MetaValue = str | Callable[[], str]
widget_meta: dict[tk.Widget | str, dict[str, MetaValue]] = {}


def set_widget_meta(widget: tk.Widget | str, /, **kwargs: MetaValue) -> None:
    """Attach metadata (help, tooltip, translation_key) to a widget or string id.

    String ids are used for tksheet treeview rows that have no real tk widget
    instance — the ``path`` field from the row's meta dict serves as identifier.
    """
    widget_meta[widget] = kwargs


def get_widget_meta(widget: tk.Widget | str, key: str, default: str = "") -> str:
    """Retrieve a metadata value for *key* (``default`` if widget or key absent).

    Lookup order: widget instance → string identifier → *default*.  ``MetaValue``
    may be ``str`` or ``Callable[[], str]`` — the latter is invoked at read time so
    dynamic statuses reflect current widget/application state.
    """
    val = widget_meta.get(widget, {}).get(key, default)
    return val() if callable(val) else val
