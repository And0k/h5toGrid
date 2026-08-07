"""Centralized colors, style constants, Tk color helpers, and UI scaling for tcm_gui.

Every color and style token used in the GUI lives here — no hardcoded
hex values in widget modules.  Runtime theme colors (resolved from
``ttk.Style``) are exposed via :func:`resolved_frame_bg` and
:func:`resolved_entry_bg`; the ``*_FALLBACK`` constants are the
last-resort defaults when the theme lookup returns ``None``.

UI scaling (:func:`apply_ui_scale`) sets the global Tk DPI factor and
configures named fonts once at root creation, so all standard widgets
scale uniformly.  Widget metadata (:data:`widget_meta`) provides a
central registry for help / tooltip / status text — Tkinter widgets carry
no built-in metadata store, so this dict is the single source of truth
for status bar captions and (future) tooltip popups.
"""

from __future__ import annotations

import tkinter as tk
import tkinter.font as tkfont
from collections.abc import Callable
from contextlib import suppress
from tkinter import ttk
from typing import Final

# ── foreground colors ────────────────────────────────────────────────────────

DEFAULT_FG: Final[str] = "#999999"  # cell value == config default
BLUE_FG: Final[str] = "#0055CC"  # header text + node label when subtree at default
FG_DEFAULT: Final[str] = "#000000"  # normal (non-default) text color
FUNC_COLOR: Final[str] = "#0070A0"  # function name in log bridge

# ── background fallbacks (when ttk.Style().lookup returns None) ──────────────

FRAME_BG_FALLBACK: Final[str] = "#F0F0F0"  # TFrame background fallback
ENTRY_BG_FALLBACK: Final[str] = "#FFFFFF"  # TEntry fieldbackground fallback

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
