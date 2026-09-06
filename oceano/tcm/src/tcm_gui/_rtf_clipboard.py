"""Rich-text clipboard: ScrolledText → RTF + HTML + plain text (Ctrl+C with colors).

Places RTF, HTML Format, and plain text on the clipboard so Word, Outlook,
CopyQ, and other apps preserve foreground colors and hyperlinks.  Plain-text
targets degrade automatically.  Falls back to plain-text-only copy if
``pywin32`` is absent or the OS clipboard is momentarily locked
(:func:`copy_rich`).
"""

from __future__ import annotations

import itertools
import tkinter as tk
from collections.abc import Iterator
from typing import TypeAlias

from utils import log_init

from tcm_gui.const import tk_font_family
from tcm_gui.theme import tk_color_to_rgb

lf = log_init.LoggingStyleAdapter(__name__)

RGB: TypeAlias = tuple[int, int, int]
# (start, end, foreground RGB | None, link URL | None) — one styled span
Segment: TypeAlias = tuple[str, str, RGB | None, str | None]


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


def _palette(widget: tk.Text) -> dict[str, RGB]:
    """tag → RGB for tags with a foreground actually covering text (``sel`` excluded)."""
    return {
        t: tk_color_to_rgb(widget, widget.tag_cget(t, "foreground"))
        for t in widget.tag_names()
        if t != "sel" and widget.tag_cget(t, "foreground") and widget.tag_ranges(t)
    }


def _segments(widget: tk.Text) -> Iterator[Segment]:
    """Yield ``(start, end, fg, url)`` per single-style span of the selection (full text if none).

    Slices at every ``_palette`` tag boundary so each span carries one tag set;
    ``url`` resolves through the widget's optional duck-typed ``link_url_at``
    hook (:meth:`tcm_gui.md_label.MarkdownLabel.link_url_at`,
    :meth:`tcm_gui.log_bridge.LogText.link_url_at`) — widgets without it
    simply yield ``url=None``.
    """
    end = widget.index("end-1c")
    # Determine slice: use selection if present; coerce Tcl_Obj → str (hackable set element)
    sel = widget.tag_ranges("sel")
    start, end = (widget.index(sel[0]), widget.index(sel[1])) if sel else ("1.0", end)
    palette = _palette(widget)
    link_url_at = getattr(widget, "link_url_at", None)

    # Slice the text at every tag boundary so each segment has one tag set
    cuts: set[str] = {"1.0", end}
    for t in palette:
        cuts.update(map(widget.index, widget.tag_ranges(t)))

    for a, b in itertools.pairwise(sorted(cuts, key=lambda i: tuple(map(int, i.split("."))))):
        if widget.compare(a, ">=", b):
            continue
        seg_a = a if widget.compare(a, ">=", start) else start
        seg_b = b if widget.compare(b, "<=", end) else end
        if not widget.compare(seg_a, "<", seg_b):
            continue
        fg = next((palette[t] for t in reversed(widget.tag_names(seg_a)) if t in palette), None)
        yield seg_a, seg_b, fg, link_url_at(seg_a) if link_url_at else None


def build_rtf(widget: tk.Text) -> str:
    """Serialize selected text to RTF, preserving foreground colors and hyperlinks.

    Each :func:`_segments` span becomes one ``\\cfN`` run; link spans wrap in a
    ``\\field{\\*\\fldinst HYPERLINK "url"}`` so Word keeps them clickable
    (``\\ul`` underlined).  Falls back to full text if no selection.
    """
    colors: list[RGB] = []
    body: list[str] = []
    for seg_a, seg_b, fg, url in _segments(widget):
        if fg is not None and fg not in colors:
            colors.append(fg)
        seg = _esc(widget.get(seg_a, seg_b))
        fmt = (f"\\cf{colors.index(fg) + 1}" if fg is not None else "") + ("\\ul" if url else "")
        run = f"{{{fmt} {seg}}}" if fmt else seg
        if url is not None:  # \* marks fldinst skippable for non-field-aware readers
            run = f'{{\\field{{\\*\\fldinst{{HYPERLINK "{_esc(url)}"}}}}{{\\fldrslt{run}}}}}'
        body.append(run)

    table = "".join(f"\\red{r}\\green{g}\\blue{b};" for r, g, b in colors)
    font = tk_font_family(widget)
    # \fonttbl is mandatory for Word to honour \cfN runs; \deff0 references \f0.
    header = (
        r"{\rtf1\ansi\deff0"
        rf"{{\fonttbl{{\f0\fmodern\fcharset0 {font};}}}}"
        rf"{{\colortbl;{table}}}"
    )
    return f"{header}{''.join(body)}}}"


def _esc_html(text: str) -> str:
    """Escape for HTML: &, <, >, double quotes (attribute-safe)."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def build_html(widget: tk.Text) -> bytes:
    """Build Windows ``CF_HTML`` clipboard content with color spans and links.

    The Windows HTML clipboard format requires a header with byte offsets::

        Version:0.9
        StartHTML:0000000100
        EndHTML:0000000200
        StartFragment:0000000120
        EndFragment:0000000180
        <html><body><!--StartFragment-->..colored spans..<!--EndFragment--></body></html>

    Offsets are zero-padded 10-digit ASCII decimal byte positions from the
    start of the entire payload.  Returns ``bytes`` (UTF-8 for the HTML body,
    ASCII for the header) with correct byte offsets.
    """
    spans: list[str] = []
    for seg_a, seg_b, fg, url in _segments(widget):
        text = _esc_html(widget.get(seg_a, seg_b)).replace("\n", "<br>")
        if fg is not None:
            r, g, b_ = fg
            text = f'<span style="color:#{r:02x}{g:02x}{b_:02x}">{text}</span>'
        if url is not None:
            text = f'<a href="{_esc_html(url)}">{text}</a>'
        spans.append(text)

    fragment = "".join(spans)
    body = f"<html><body><!--StartFragment-->{fragment}<!--EndFragment--></body></html>"
    body_bytes = body.encode("utf-8")

    # Build header with placeholder offsets to measure its length.
    hdr_tpl = "Version:0.9\r\nStartHTML:{:010d}\r\nEndHTML:{:010d}\r\nStartFragment:{:010d}\r\nEndFragment:{:010d}\r\n"
    hdr_len = len(hdr_tpl.format(0, 0, 0, 0))
    frag_start = hdr_len + body_bytes.index(b"<!--StartFragment-->") + len(b"<!--StartFragment-->")
    frag_end = hdr_len + body_bytes.index(b"<!--EndFragment-->")
    header = hdr_tpl.format(hdr_len, hdr_len + len(body_bytes), frag_start, frag_end)
    return header.encode("ascii") + body_bytes


def copy_rich(widget: tk.Text) -> None:
    """Plain text + RTF + HTML onto the clipboard; falls back to plain text.

    Hardened against two real-world clipboard failures observed in the field:

    1. ``build_rtf`` raising *before* any clipboard op would otherwise empty
       the OS clipboard but place nothing → user loses the prior clipboard
       content.  Build payloads first; only on success open +
       ``EmptyClipboard`` + write.
    2. ``win32clipboard.OpenClipboard`` raising ``pywintypes.error``
       ("Отказано в доступе"/Access denied) when another viewer holds the
       clipboard momentarily.  Retry ``OpenClipboard`` briefly, then fall back
       to ``widget.clipboard_clear()`` + ``widget.clipboard_append`` so the
       user still gets plain text — never an unhandled exception from Ctrl+C.
    """
    sel = widget.tag_ranges("sel")
    plain = widget.get(widget.index(sel[0]), widget.index(sel[1])) if sel else widget.get("1.0", "end-1c")
    try:
        import win32clipboard as wcb  # pywin32
    except ImportError:
        widget.clipboard_clear()
        widget.clipboard_append(plain)
        return

    # Build payloads BEFORE touching the OS clipboard — a failed build
    # leaves the user's prior clipboard intact rather than emptying it.
    try:
        CF_RTF = wcb.RegisterClipboardFormat("Rich Text Format")
        CF_HTML = wcb.RegisterClipboardFormat("HTML Format")
        rtf_bytes = build_rtf(widget).encode("ascii")
        html_bytes = build_html(widget)
    except Exception:  # noqa: BLE001 — never crash Ctrl+C; fall back.
        lf.warning("RTF/HTML build failed; copying plain text only", exc_info=True)
        widget.clipboard_clear()
        widget.clipboard_append(plain)
        return

    # OpenClipboard can raise pywintypes.error ("Access denied") when another
    # viewer holds the clipboard; retry briefly before degrading to Tk plain.
    import time

    widget.update()
    for attempt in range(40):
        try:
            wcb.OpenClipboard()
            break
        except Exception:  # noqa: BLE001 — pywintypes.error "Access denied".
            widget.update()
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
        wcb.SetClipboardData(CF_HTML, html_bytes)
    finally:
        wcb.CloseClipboard()
