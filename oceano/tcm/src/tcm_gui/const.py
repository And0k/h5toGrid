"""Centralized colors, style constants, Tk color helpers, and UI scaling for tcm_gui.

Every color and style token used in the GUI lives here — no hardcoded
hex values in widget modules.  Runtime theme colors (resolved from
``ttk.Style``) are exposed via :func:`resolved_frame_bg` and
:func:`resolved_entry_bg`; the ``*_FALLBACK`` constants are the
last-resort defaults when the theme lookup returns ``None``.

UI scaling (:func:`apply_ui_scale`) sets the global Tk DPI factor and
configures named fonts once at root creation, so all standard widgets
scale uniformly.  Theme detection (:func:`apply_theme_defaults`) reads
the Windows ``AppsUseLightTheme`` registry key to pick light/dark
colors for log output and function names.  Widget metadata
(:data:`widget_meta`) provides a central registry for help / tooltip /
status text — Tkinter widgets carry no built-in metadata store, so this
dict is the single source of truth for status bar captions and (future)
tooltip popups.
"""

from __future__ import annotations

import ctypes
import logging
import sys
import tkinter as tk
import tkinter.font as tkfont
from collections.abc import Callable
from contextlib import suppress
from tkinter import ttk
from typing import Final

_l = logging.getLogger(__name__)

# ── foreground colors ────────────────────────────────────────────────────────
# Mutable — all updated by apply_theme_defaults() for dark/light mode.

DEFAULT_FG: str = "#999999"  # cell value == config default
BLUE_FG: str = "#0055CC"  # header text + node label when subtree at default
FG_DEFAULT: str = "#000000"  # normal (non-default) text color
FUNC_COLOR: str = "#0070A0"  # function name in log bridge

# ── background fallbacks (when ttk.Style().lookup returns None) ──────────────
# Mutable — updated by apply_theme_defaults() for dark/light mode.

FRAME_BG_FALLBACK: str = "#F0F0F0"  # TFrame background fallback
ENTRY_BG_FALLBACK: str = "#FFFFFF"  # TEntry fieldbackground fallback
CELL_NON_DATA_BG: str = "#E8E8E8"  # subtle gray for read-only cells

# Detected theme — set by :func:`apply_theme_defaults`, read by widget modules
# to choose tksheet theme, configure ttk.Style, etc.
THEME: str = "light"

# ── UI scaling ───────────────────────────────────────────────────────────────
# ``tk scaling`` sets pts→px: 1.0 ≈ 96 DPI, 1.5 ≈ 144 DPI, 2.0 ≈ 192 DPI.
# Named fonts (TkDefaultFont, TkTextFont) cascade the size to every standard
# widget that inherits them — Button, Label, Entry, ttk.* all pick it up.

UI_SCALE: Final[float] = 2.0  # global Tk scaling factor (devicePixelRatio analog)
FONT_SIZE: Final[int] = 14  # base font size for named fonts


def apply_ui_scale(root: tk.Tk) -> None:
    """Apply global DPI scaling and named-font sizing to *root*.

    Call once immediately after the ``tk.Tk()`` constructor — before any
    widget is created, so every child inherits the scaled fonts.  Manual
    pixel dimensions (``width=``, ``height=``) and Canvas art are not
    affected and must be scaled separately where used.
    """
    root.tk.call("tk", "scaling", UI_SCALE)
    # Cascade font size through all standard widgets that reference named fonts;
    # some fonts may be absent on exotic themes — swallow TclError per-font.
    for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
        with suppress(tk.TclError):
            tkfont.nametofont(name).configure(size=FONT_SIZE)


# ── theme detection ─────────────────────────────────────────────────────────


def _detect_windows_theme() -> str:
    """Read the Windows ``AppsUseLightTheme`` registry value.

    Returns ``"dark"`` or ``"light"``.  Falls back to ``"light"`` on
    non-Windows platforms or when the key is absent.
    """
    if sys.platform != "win32":
        return "light"
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize",
        ) as key:
            val, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
            return "light" if val else "dark"
    except (OSError, FileNotFoundError):
        return "light"


# Dark-mode color overrides — keyed by the same names as the module globals.
_DARK: dict[str, str] = {
    "FUNC_COLOR": "#5CB8D6",
    "DEFAULT_FG": "#808080",
    "BLUE_FG": "#4DA6FF",
    "FG_DEFAULT": "#D4D4D4",
    "FRAME_BG_FALLBACK": "#2D2D2D",
    "ENTRY_BG_FALLBACK": "#1E1E1E",
    "CELL_NON_DATA_BG": "#383838",
    "THEME": "dark",
    "debug": "#707070",
    "info": "#D4D4D4",
    "warning": "#FFB84D",
    "error": "#FF6B6B",
    "critical": "#FF6B6B",
}

_LIGHT: dict[str, str] = {
    "FUNC_COLOR": "#0070A0",
    "DEFAULT_FG": "#999999",
    "BLUE_FG": "#0055CC",
    "FG_DEFAULT": "#000000",
    "FRAME_BG_FALLBACK": "#F0F0F0",
    "ENTRY_BG_FALLBACK": "#FFFFFF",
    "CELL_NON_DATA_BG": "#E8E8E8",
    "THEME": "light",
    "debug": "#808080",
    "info": "#1a1a1a",
    "warning": "#CC7000",
    "error": "#CC0000",
    "critical": "#CC0000",
}

# Keys from _DARK/_LIGHT that map to module-level globals (not TAG_COLORS).
_GLOBAL_KEYS = (
    "FUNC_COLOR",
    "DEFAULT_FG",
    "BLUE_FG",
    "FG_DEFAULT",
    "FRAME_BG_FALLBACK",
    "ENTRY_BG_FALLBACK",
    "CELL_NON_DATA_BG",
    "THEME",
)


def apply_theme_defaults(root: tk.Tk) -> str:
    """Detect Windows dark/light mode and set all color globals accordingly.

    Call once after :func:`apply_ui_scale` — before any widget uses
    :data:`FUNC_COLOR`, :data:`TAG_COLORS`, or the background fallbacks.
    Mutates the module-level dicts and globals in place so all importers
    see the updated values.  Also configures ``ttk.Style`` for dark mode
    so that ``resolved_frame_bg()`` and ``resolved_entry_bg()`` return
    dark colors, and all ttk widgets render with dark backgrounds.

    :returns: ``"dark"`` or ``"light"``.
    """
    theme = _detect_windows_theme()
    palette = _DARK if theme == "dark" else _LIGHT
    # Update module-level globals.
    import sys as _sys

    _self = _sys.modules[__name__]
    for key in _GLOBAL_KEYS:
        setattr(_self, key, palette[key])
    # Update TAG_COLORS dict.
    TAG_COLORS.update((k, palette[k]) for k in ("debug", "info", "warning", "error", "critical"))
    if theme == "dark":
        _apply_ttk_dark(root)
    # Opt into dark title bar unconditionally — DWMWA_USE_IMMERSIVE_DARK_MODE
    # means "honor system dark mode", not "force dark".  On a light system
    # this produces a light title bar; on a dark system, a dark one.
    _opt_into_dark_titlebar(root)
    _l.debug(
        "Theme detected: {} → FUNC_COLOR={}, FRAME_BG_FALLBACK={}, ENTRY_BG_FALLBACK={}",
        theme,
        FUNC_COLOR,
        FRAME_BG_FALLBACK,
        ENTRY_BG_FALLBACK,
    )
    return theme


def _apply_ttk_dark(root: tk.Tk) -> None:
    """Switch to ``clam`` ttk theme and configure dark-mode colors.

    Called by :func:`apply_theme_defaults` when the detected theme is ``"dark"``.
    The Windows native themes (``vista``, ``xpnative``) ignore
    ``ttk.Style().configure()`` for visual rendering — ``clam`` is the only
    cross-platform ttk theme that fully respects custom colors.

    Also sets the root window background so that ``root.cget("background")``
    returns the dark frame color (used by some widgets as implicit parent bg).
    """
    style = ttk.Style()
    style.theme_use("clam")
    bg, fg, entry_bg, non_data = FRAME_BG_FALLBACK, FG_DEFAULT, ENTRY_BG_FALLBACK, CELL_NON_DATA_BG
    # TFrame / TLabel — container-level widgets.
    style.configure(".", background=bg, foreground=fg)
    # TButton — slightly lighter bg for contrast.
    style.configure("TButton", background=non_data, foreground=fg)
    style.map(
        "TButton",
        background=[("active", bg), ("pressed", bg)],
        foreground=[("disabled", "#808080")],
    )
    # TNotebook + tabs.
    style.configure("TNotebook", background=bg)
    style.configure(
        "TNotebook.Tab",
        background=non_data,
        foreground=fg,
        padding=[8, 2],
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", bg)],
        foreground=[("selected", fg)],
    )
    # TEntry — input field.
    style.configure("TEntry", fieldbackground=entry_bg, foreground=fg, insertcolor=fg)
    style.map("TEntry", fieldbackground=[("focus", entry_bg)])
    # TProgressbar — trough and bar.
    style.configure("TProgressbar", troughcolor=bg, background=non_data)
    # Scrollbar — shared style for log ScrolledText replacement and tksheet.
    # tksheet uses scrollbar_theme_inheritance="clam" so its scrollbars match.
    style.configure(
        "App.Vertical.TScrollbar",
        troughcolor=bg,
        background=non_data,
        arrowcolor="#808080",
        bordercolor=bg,
        lightcolor=bg,
        darkcolor=bg,
    )
    style.map(
        "App.Vertical.TScrollbar",
        background=[("pressed", fg), ("active", non_data)],
        arrowcolor=[("pressed", fg), ("active", "#a0a0a0")],
    )
    # Root window bg — used by tk.Frame/Label as implicit parent color.
    root.configure(bg=bg)
    _l.debug(
        "ttk.Style switched to 'clam' for dark mode: bg={}, fg={}, entry_bg={}",
        bg,
        fg,
        entry_bg,
    )


def _opt_into_dark_titlebar(root: tk.Tk) -> None:
    """Tell Windows DWM to honor the system dark/light theme for the title bar.

    ``DWMWA_USE_IMMERSIVE_DARK_MODE = 20`` does **not** force dark — it means
    "this window supports dark mode; follow the system setting".  On a light
    system the title bar stays light; on a dark system it becomes dark.

    Requires Windows 11 build 22000+.  On older Windows or non-Win32 platforms
    this is a silent no-op.

    Tk's ``winfo_id()`` returns a **child** HWND (the internal Tk widget window),
    not the actual top-level frame that DWM renders.  ``GetAncestor(GA_ROOT)``
    walks up to the real toplevel HWND that DWM controls.
    """
    if sys.platform != "win32":
        return
    with suppress(OSError, ValueError):
        root.update_idletasks()
        tk_hwnd = root.winfo_id()
        if not tk_hwnd:
            return
        # Walk up to the real toplevel HWND — winfo_id() is a child widget.
        user32 = ctypes.WinDLL("user32")
        user32.GetAncestor.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        user32.GetAncestor.restype = ctypes.c_void_p
        GA_ROOT = 2
        hwnd = user32.GetAncestor(tk_hwnd, GA_ROOT) or tk_hwnd
        # Set DWMWA_USE_IMMERSIVE_DARK_MODE = TRUE on the real HWND.
        dwm = ctypes.WinDLL("dwmapi")
        dwm.DwmSetWindowAttribute.argtypes = [
            ctypes.c_void_p,  # HWND
            ctypes.c_uint,  # DWMWINDOWATTRIBUTE
            ctypes.c_void_p,  # LPCVOID
            ctypes.c_uint,  # cbAttribute
        ]
        dwm.DwmSetWindowAttribute.restype = ctypes.c_long  # HRESULT
        value = ctypes.c_int(1)
        hr = dwm.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(value), ctypes.sizeof(value))
        _l.debug(
            "DwmSetWindowAttribute(HWND={:#x}, DWMWA_USE_IMMERSIVE_DARK_MODE=TRUE) → HRESULT={:#010x}",
            hwnd,
            hr & 0xFFFFFFFF,
        )


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


# ── chrome help content (i18n substitution point) ────────────────────────────
# Stable string keys for chrome widgets (everything that is NOT a config cell).
# At build for another language, replace this dict wholesale with the same keys
# → localized strings; App auto-registration (``_register_chrome_help``) then
# reads already-translated values at startup.  Config-cell help comes from a
# different source (``config_reference.md``) and is resolved by ``_help.py``.

STR: dict[str, str] = {
    "path_lbl.tooltip": "Data search path",
    "path_field.tooltip": "Data search path",
    "path_field.status": "Changing data path rescans and resets all config tabs below",
    "cfg_lbl.status": "Current configuration state",
    "run.tooltip": "Run button",
    "run.start": "Start processing",
    "run.pause": "Pause processing",
    "run.resume": "Resume processing",
    "status_lbl.status": "Application status messages",
    "nb.status": "Configuration tabs — one per data file / deployment",
    "log.status": "Processing log — color-coded by severity (debug/info/warning/error)",
    "prog_all.status": "Overall progress across all configurations",
    "prog_stage.status": "Current stage progress within the active configuration",
    "tab.status": "Configuration: {path}",  # template — _add_page formats {path}
}


# ── log level → ScrolledText tag colors ──────────────────────────────────────
# Mutable — updated by apply_theme_defaults() for dark/light mode.

TAG_COLORS: dict[str, str] = {
    "debug": "#808080",
    "info": "#1a1a1a",
    "warning": "#CC7000",
    "error": "#CC0000",
    "critical": "#CC0000",
}


# ── Tk color helpers ────────────────────────────────────────────────────────


def tk_color_to_rgb(widget, color: str) -> tuple[int, int, int]:
    """Any Tk color spec → 8-bit RGB via the display's pixel mapping."""
    r, g, b = widget.winfo_rgb(color)
    return r >> 8, g >> 8, b >> 8


def tk_color_to_hex(widget, color: str) -> str:
    """Convert any Tk color spec to hex — tksheet rejects system names like 'SystemButtonFace'."""
    try:
        return "#{:02x}{:02x}{:02x}".format(*tk_color_to_rgb(widget, color))
    except tk.TclError:
        return color


# ── resolved theme colors (call after Tk root exists) ───────────────────────


def resolved_frame_bg(widget) -> str:
    """TFrame background resolved from the current ttk theme, hex-encoded."""
    return tk_color_to_hex(widget, ttk.Style().lookup("TFrame", "background") or FRAME_BG_FALLBACK)


def resolved_entry_bg(widget) -> str:
    """TEntry fieldbackground resolved from the current ttk theme, hex-encoded."""
    style = ttk.Style()
    return tk_color_to_hex(
        widget,
        style.lookup("TEntry", "fieldbackground")
        or style.lookup("TEntry", "background")
        or ENTRY_BG_FALLBACK,
    )


# ── font helpers ────────────────────────────────────────────────────────────


def tk_font_family(widget: tk.Text) -> str:
    """Extract the family name from the widget font (e.g. 'Consolas').

    ``cget('font')`` may return a Tk font spec like ``'Consolas 11'`` or a named font;
    we take the first whitespace-separated token as the family, falling back to
    ``'Consolas'`` (the production default) if extraction fails.
    """
    spec = widget.cget("font") or ""
    return (str(spec).split() or ["Consolas"])[0]
