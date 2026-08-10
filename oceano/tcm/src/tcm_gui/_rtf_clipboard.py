"""Rich-text clipboard: ScrolledText → RTF + plain text (Ctrl+C with colors).

Places both RTF and plain text on the clipboard so apps like Word or Outlook
preserve foreground colors while plain-text apps fall back automatically.
Falls back to plain-text-only copy if ``pywin32`` is absent or the OS clipboard
is momentarily locked by another viewer (:func:`copy_rich`).
"""

from __future__ import annotations

import logging

import itertools
import tkinter as tk

from tcm_gui.const import tk_font_family
from tcm_gui.theme import tk_color_to_rgb

lf = logging.getLogger(__name__)


def _esc(text: str) -> str:
    """Escape one chunk for RTF: control chars, braces, unicode, newlines."""
    out: list[str] = []
    for ch in text:
        if ch == "\\":
            out.append("\\\\")
        elif ch == "{":
            out.append("\\{")
        elif ch == "}":
            out.append("\\}")
        elif ch == "\n":
            out.append("\\par\n")
        elif (cp := ord(ch)) > 127:  # RTF \\u takes signed 16-bit
            out.append(f"\\u{cp - 65536 if cp > 32767 else cp}?")
        else:
            out.append(ch)
    return "".join(out)


def build_rtf(widget: tk.Text) -> str:
    """Serialize selected text to RTF, preserving foreground colors.

    Scans all tag boundaries so each text segment carries a single
    ``\\cfN`` color reference. Colors are collected into a \\colortbl
    from the full widget, then only the selected range is emitted.
    Falls back to full text if no selection.
    """
    end = widget.index("end-1c")
    # Determine slice: use selection if present; coerce Tcl_Obj → str (hackable set element)
    sel = widget.tag_ranges("sel")
    start, end = (widget.index(sel[0]), widget.index(sel[1])) if sel else ("1.0", end)
    palette = {
        t: tk_color_to_rgb(widget, widget.tag_cget(t, "foreground"))
        for t in widget.tag_names()
        if t != "sel"
        and widget.tag_cget(t, "foreground")
        and widget.tag_ranges(t)  # only tags actually covering text
    }
    colors = list(dict.fromkeys(palette.values()))

    # Slice the text at every tag boundary so each segment has one tag set
    cuts: set[str] = {"1.0", end}
    for t in palette:
        cuts.update(map(widget.index, widget.tag_ranges(t)))
    cuts_sorted = sorted(cuts, key=lambda i: tuple(map(int, i.split("."))))

    body: list[str] = []
    for a, b in itertools.pairwise(cuts_sorted):
        if widget.compare(a, ">=", b):
            continue
        seg_a = a if widget.compare(a, ">=", start) else start
        seg_b = b if widget.compare(b, "<=", end) else end
        if not widget.compare(seg_a, "<", seg_b):
            continue
        fg = next(
            (palette[t] for t in reversed(widget.tag_names(seg_a)) if t in palette),
            None,
        )
        seg = _esc(widget.get(seg_a, seg_b))
        if fg is not None:
            body.append(f"{{\\cf{colors.index(fg) + 1} {seg}}}")
        else:
            body.append(seg)

    table = "".join(f"\\red{r}\\green{g}\\blue{b};" for r, g, b in colors)
    font = tk_font_family(widget)
    # \fonttbl is mandatory for Word to honour \cfN runs; \deff0 references \f0.
    header = (
        r"{\rtf1\ansi\deff0"
        rf"{{\fonttbl{{\f0\fmodern\fcharset0 {font};}}}}"
        rf"{{\colortbl;{table}}}"
    )
    return f"{header}{''.join(body)}}}"


def copy_rich(widget: tk.Text) -> None:
    """Plain text + RTF onto the clipboard; falls back to plain text elsewhere.

    Hardened against two real-world clipboard failures observed in the field:

    1. ``build_rtf`` raising *before* any clipboard op would otherwise empty
       the OS clipboard but place nothing → user loses the prior clipboard
       content.  Build the RTF payload first; only on success open +
       ``EmptyClipboard`` + write.
    2. ``win32clipboard.OpenClipboard`` raising ``pywintypes.error``
       ("Отказано в доступе"/Access denied) when another viewer holds the
       clipboard momentarily.  Retry ``OpenClipboard`` briefly, then fall back
       to ``widget.clipboard_clear()`` + ``widget.clipboard_append`` so the
       user still gets plain text — never an unhandled exception from a
       ``<Control-c>`` binding.
    """
    sel = widget.tag_ranges("sel")
    plain = (
        widget.get(widget.index(sel[0]), widget.index(sel[1]))
        if sel
        else widget.get("1.0", "end-1c")
    )
    try:
        import win32clipboard as wcb  # pywin32
    except ImportError:
        widget.clipboard_clear()
        widget.clipboard_append(plain)
        return

    # Compute RTF payload BEFORE touching the OS clipboard — a failed build
    # leaves the user's prior clipboard intact rather than emptying it.
    try:
        CF_RTF = wcb.RegisterClipboardFormat("Rich Text Format")
        rtf_bytes = build_rtf(widget).encode("ascii")
    except Exception:  # noqa: BLE001 — never crash <Control-c>; fall back.
        lf.warning("RTF build failed; copying plain text only: {}", exc_info=True)
        widget.clipboard_clear()
        widget.clipboard_append(plain)
        return

    # OpenClipboard can raise pywintypes.error ("Access denied") when another
    # viewer holds the clipboard; retry briefly before degrading to Tk plain.
    import time

    for attempt in range(20):
        try:
            wcb.OpenClipboard()
            break
        except Exception:  # noqa: BLE001 — pywintypes.error "Access denied".
            time.sleep(0.05)
    else:
        lf.warning("OpenClipboard stayed locked after retries; copying plain text only")
        widget.clipboard_clear()
        widget.clipboard_append(plain)
        return

    try:
        wcb.EmptyClipboard()
        wcb.SetClipboardData(wcb.CF_UNICODETEXT, plain)  # fallback target
        wcb.SetClipboardData(CF_RTF, rtf_bytes)
    finally:
        wcb.CloseClipboard()