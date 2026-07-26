import tkinter as tk
from typing import Final

DEFAULT_FG: Final[str] = "#999999"  # cell value == config default
BLUE_FG: Final[str] = "#0055CC"  # header text + node label when subtree at default

TAG_COLORS: dict[str, str] = {
    "debug": "#808080",
    "info": "#1a1a1a",
    "warning": "#CC7000",
    "error": "#CC0000",
    "critical": "#CC0000",
}


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


def tk_font_family(widget: tk.Text) -> str:
    """Extract the family name from the widget font (e.g. 'Consolas').

    `cget('font')` may return a Tk font spec like 'Consolas 11' or a named font;
    we take the first whitespace-separated token as the family, falling back to
    'Consolas' (the production default) if extraction fails.
    """
    spec = widget.cget("font") or ""
    return (str(spec).split() or ["Consolas"])[0]
