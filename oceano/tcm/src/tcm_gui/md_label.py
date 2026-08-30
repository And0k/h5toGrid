"""Markdown-rendering Tk Text widget for help/status messages.

Renders a small Markdown subset directly into tagged Tk Text ranges.
The parser lives in :mod:`tcm._md_parse`; this module provides only
the Tk renderer (:class:`MarkdownLabel`).

Tables are aligned by measured tab stops.
"""

from __future__ import annotations

import logging
import os
import re
import tkinter as tk
import tkinter.font as tkfont
from collections.abc import Callable
from itertools import accumulate, zip_longest
from pathlib import Path
from typing import TypeAlias

from tcm._md_parse import Block, CodeBlock, Heading, Inline, List, Paragraph, Table, parse_markdown

from . import theme

FontSpec: TypeAlias = tkfont.Font | str | tuple[str | int, ...] | None

_l = logging.getLogger(__name__)

# The parser's ``{#name}`` color grammar (see ``_md_parse.py``) — a tag shaped
# like a color name is a color even when unmapped; anything else is a link URL.
_COLOR_NAME = re.compile(r"[a-z_]+\Z")

# ── Tk widget ────────────────────────────────────────────────────────────────


class MarkdownLabel(tk.Text):
    """Read-only label-like Tk Text widget rendering a Markdown subset.

    Inline ``[text](url)`` links parse to a span whose tag IS the target URL
    (:mod:`tcm._md_parse`); this renderer styles them (``link`` tag: link
    color + underline), switches the cursor to ``hand2`` on hover, and
    forwards clicks to the ``on_link`` callback with the URL and the ``base``
    set via :meth:`set_text` (for relative-link resolution).
    """

    _CELL_PAD = 8

    def __init__(
        self,
        master: tk.Misc | None = None,
        *,
        font: FontSpec = None,
        background: str | None = None,
        foreground: str | None = None,
        autoheight: bool = True,
        colors: dict[str, str] | None = None,
        on_link: Callable[[str, str | None], None] | None = None,
        **kwargs,
    ):
        super().__init__(
            master,
            **{
                "borderwidth": 0,
                "highlightthickness": 0,
                "relief": "flat",
                "wrap": "none",
                "cursor": "",
                "width": 0,
                "height": 0,
                "padx": 2,
                "pady": 0,
                **kwargs,
                "takefocus": 0,
                "state": "disabled",
            },
        )

        self._autoheight = autoheight
        self._wrap = kwargs.get("wrap", "none")  # caller-requested wrap; _render restores it
        self._autosizing = False
        self._sizing_width = False
        self._font_scaled = False
        self._current: tuple[Block, ...] | None = None
        self._table_uid = 0
        self._fitted_h: int | None = None  # cached height from _fit_height
        self._fitted_w: int = 0  # cached width from _fit_height
        self._colors: dict[str, str] = colors or {}  # {#name} → hex foreground
        self._on_link = on_link
        self._base: Path | None = None  # source-doc dir for relative links
        self._links: list[tuple[str, str, str]] = []  # (start, end, url) spans

        self._fonts = self._build_fonts(font)
        self._configure_tags()
        self._raise_span_tags()
        self.tag_bind("link", "<Button-1>", self._open_link)
        self.tag_bind("link", "<Enter>", lambda _e: self.configure(cursor="hand2"))
        self.tag_bind("link", "<Leave>", lambda _e: self.configure(cursor=""))

        if background is None and master is not None:
            try:
                background = master.cget("background")
            except tk.TclError:
                background = None

        if background is not None:
            self.configure(background=background)
        if foreground is not None:
            self.configure(foreground=foreground)

        # Re-compute height when widget resizes (window resize, etc.)
        self.bind("<Configure>", self._on_configure, add="+")

    # ── public API ───────────────────────────────────────────────────────

    def set_text(self, text: str, /, *, raw: bool = False, base: str | os.PathLike | None = None) -> bool:
        """Render *text* as Markdown (default) or as a literal single span (``raw=True``).

        ``raw`` bypasses :func:`parse_markdown` — for strings that interpolate
        untrusted content (e.g. filesystem paths in ``STR["tab.status"]``),
        preventing ``_``/``*``/``\\``/`` ` `` in the substitution from being
        reinterpreted as inline markup.  No-op if parsed blocks are unchanged.

        ``base`` is the source document directory for relative link targets
        (e.g. ``docs/reference/`` when *text* came from ``config_reference.md``);
        ``raw=True`` text contains no links, so *base* is ignored.
        """
        self._base = Path(base) if base else None
        if not text:
            blocks: tuple[Block, ...] = ()
        elif raw:
            blocks = (Paragraph(((text, "plain"),)),)
        else:
            blocks = parse_markdown(text)
        if blocks == self._current:
            return False
        self._current = blocks
        self._render(blocks)
        return True

    def rerender(self) -> bool:
        """Re-run _render on cached blocks (after font scale). No reparse."""
        if self._current is None:
            return False
        blocks, self._current = self._current, None  # break guard for force-rerun
        self._render(blocks)
        return True

    def clear(self) -> None:
        self.set_text("")

    def mark_font_ready(self) -> None:
        """Enable auto-sizing without changing font sizes.

        Call after the font is finalized (e.g. by ``UIScale.font()``) so
        that ``_fit_width`` / ``_fit_height`` / ``_on_configure`` can run.
        Separate from :meth:`fit_to_height` which *also* rescales fonts.
        """
        self._font_scaled = True

    def fit_to_height(self, max_px: int) -> None:
        """Scale fonts so the tallest single display line fits within *max_px*.

        Uses the maximum linespace across **all** font variants (plain,
        bold, heading, code) — heading is ``size+1`` and would otherwise
        overflow *max_px* when only ``plain`` is checked.
        """
        if max_px <= 4:
            return
        max_ls = max(f.metrics("linespace") for f in self._fonts.values())
        if max_ls <= max_px:
            self._font_scaled = True
            return
        ratio = max_px / max_ls
        for f in self._fonts.values():
            cur = abs(int(f.cget("size")))
            new = max(6, int(cur * ratio))
            if new != cur:
                f.configure(size=new)
        self._font_scaled = True

    # ── rendering ────────────────────────────────────────────────────────

    def _render(self, blocks: tuple[Block, ...]) -> None:
        # wrap="none" during insert (stable positions); restored below —
        # tables stay un-wrapped via per-tag ``wrap="none"``.
        self.configure(state="normal", wrap="none")
        self.delete("1.0", "end")
        self._table_uid = 0
        self._links = []

        for block in blocks:
            match block:
                case Heading(text=text):
                    self._insert_inline(text, ("heading",))
                    self.insert("end", "\n")

                case Paragraph(text=text):
                    self._insert_inline(text, ("normal",))
                    self.insert("end", "\n")

                case CodeBlock(text=text):
                    self.insert("end", text, "codeblock")
                    self.insert("end", "\n")

                case List(items=items):
                    for item in items:
                        self.insert("end", "• ", ("normal",))
                        self._insert_inline(item, ("normal",))
                        self.insert("end", "\n")

                case Table() as table:
                    self._render_table(table)

        if self.compare("end-1c", ">", "1.0") and self.get("end-2c", "end-1c") == "\n":
            self.delete("end-2c", "end-1c")

        self.configure(state="disabled", wrap=self._wrap)
        self.configure(cursor="")  # content swapped → stale hand2 from a link hover
        self._fitted_h = None  # content changed → re-measure height
        self.after_idle(self._fit_width)
        self.after_idle(self._fit_height)

    def _render_table(self, table: Table) -> None:
        self._table_uid += 1
        tag = f"tbl{self._table_uid}"

        widths = self._table_widths(table)
        tabs = tuple(accumulate(widths[:-1])) if len(widths) > 1 else ()

        self.tag_configure(
            tag,
            tabs=tabs,
            wrap="none",
            lmargin1=0,
            lmargin2=0,
            spacing1=2,
            spacing3=2,
        )

        self._render_table_row(tag, table.header, len(widths), header=True)
        for row in table.rows:
            self._render_table_row(tag, row, len(widths))

    def _render_table_row(
        self,
        tag: str,
        row: tuple[Inline, ...],
        ncols: int,
        *,
        header: bool = False,
    ) -> None:
        base = (tag, "table_header" if header else "table_cell")

        for j in range(ncols):
            if j:
                self.insert("end", "\t", tag)

            cell = row[j] if j < len(row) else ()
            self._insert_inline(cell, base)

        self.insert("end", "\n", tag)

    def _insert_inline(self, inline: Inline, base_tags: tuple[str, ...]) -> None:
        """Insert inline spans with correct tag application.

        A span tag that is not ``plain``/a color name/one of this widget's
        font variants is a link target URL (see :mod:`tcm._md_parse`): the
        span renders under the ``link`` tag and its range is recorded in
        ``_links`` for click resolution.
        """
        for text, tag in inline:
            if tag == "plain":
                tags = base_tags
            elif tag in self._colors:
                self.tag_configure(tag, foreground=self._colors[tag])
                tags = (*base_tags, tag)
            elif tag in self._fonts:
                tags = (*base_tags, tag)  # bold / italic / code style span
            elif _COLOR_NAME.fullmatch(tag):
                tags = (*base_tags, tag)  # unmapped {#name} color — tag passthrough
            else:
                tags = (*base_tags, "link")  # tag = link target URL
                self.insert("end", text, tags)
                end = self.index("insert")  # insert cursor: just after inserted text
                self._links.append((f"{end}-{len(text)}c", end, tag))
                continue
            self.insert("end", text, tags)

    def link_url_at(self, index: str) -> str | None:
        """Target URL of the link span containing *index* (``None`` outside links).

        Duck-typed hook for :mod:`tcm_gui._rtf_clipboard` — any Text widget
        exposing it exports clickable links (RTF ``HYPERLINK`` / HTML ``<a>``).
        """
        return next(
            (
                url
                for start, end, url in self._links
                if self.compare(start, "<=", index) and self.compare(index, "<", end)
            ),
            None,
        )

    def link_at(self, x: int, y: int) -> str | None:
        """Target URL of the link span at widget coordinates ``(x, y)``."""
        return self.link_url_at(self.index(f"@{x},{y}"))

    def _open_link(self, event: tk.Event) -> None:
        """Forward a link click (``url``, ``base``) to the ``on_link`` callback."""
        if self._on_link is not None and (url := self.link_at(event.x, event.y)):
            self._on_link(url, self._base)

    # ── table metrics ────────────────────────────────────────────────────

    def _table_widths(self, table: Table) -> tuple[int, ...]:
        def width(cell: Inline) -> int:
            return self._CELL_PAD + sum(
                self._fonts.get(tag, self._fonts["plain"]).measure(text) for text, tag in cell
            )

        return tuple(
            max(map(width, column), default=0)
            for column in zip_longest(table.header, *table.rows, fillvalue=())
        )

    # ── fonts / tags ─────────────────────────────────────────────────────

    def _build_fonts(self, spec: FontSpec) -> dict[str, tkfont.Font]:
        base = self._font_from(spec)
        family = base.cget("family")
        size = int(base.cget("size"))
        heading_size = size + 1 if size > 0 else size - 1

        return {
            "plain": base,
            "bold": tkfont.Font(family=family, size=size, weight="bold"),
            "italic": tkfont.Font(family=family, size=size, slant="italic"),
            "heading": tkfont.Font(family=family, size=heading_size, weight="bold"),
            "code": tkfont.Font(family="Consolas", size=size),
        }

    @staticmethod
    def _font_from(spec: FontSpec) -> tkfont.Font:
        if spec is None:
            return tkfont.nametofont("TkDefaultFont").copy()

        if isinstance(spec, tkfont.Font):
            return spec.copy()

        if isinstance(spec, str):
            try:
                return tkfont.nametofont(spec).copy()
            except tk.TclError:
                return tkfont.Font(family=spec)

        return tkfont.Font(font=spec)

    def _configure_tags(self) -> None:
        self.tag_configure("normal", font=self._fonts["plain"])
        self.tag_configure("bold", font=self._fonts["bold"])
        self.tag_configure("italic", font=self._fonts["italic"])
        self.tag_configure(
            "code",
            font=self._fonts["code"],
            foreground=theme.CODE_FG,
            background=theme.CODE_BG,
            selectforeground=theme.CODE_SEL_FG,
            selectbackground=theme.CODE_SEL_BG,
        )

        self.tag_configure(
            "heading",
            font=self._fonts["heading"],
            spacing1=4,
            spacing3=2,
        )

        self.tag_configure(
            "codeblock",
            font=self._fonts["code"],
            lmargin1=8,
            lmargin2=8,
            spacing1=2,
            spacing3=2,
        )

        self.tag_configure("table_header", font=self._fonts["bold"])
        self.tag_configure("table_cell", font=self._fonts["plain"])
        self.tag_configure("link", foreground=theme.LINK_FG, selectforeground=theme.LINK_SEL_FG, underline=True)

    def _raise_span_tags(self) -> None:
        for tag in ("code", "bold", "italic", "link"):
            self.tag_raise(tag)

    # ── size management ──────────────────────────────────────────────────

    def _on_configure(self, _event: tk.Event) -> None:
        """Widget resized — recompute width and height (only after font scaled)."""
        if self._font_scaled and not self._autosizing and not self._sizing_width:
            self.after_idle(self._fit_width)
            self.after_idle(self._fit_height)

    def _fit_width(self) -> None:
        """Measure natural content width from AST + per-span font metrics.

        Walks the parsed blocks (not raw widget text) so each inline
        span is measured with its **actual** font — code (Consolas) is
        wider than bold (Segoe UI) for the same text, so using
        ``bold.measure()`` for everything underestimates.

        If the natural width fits the window, keeps ``wrap="none"``
        (tight contraction).  Otherwise switches to ``wrap="word"`` and
        caps to window width.  Once ``wrap="word"`` is set, does not
        switch back — prevents oscillation on ``<Configure>`` cycles.
        """
        if not self._font_scaled or self._sizing_width:
            return

        if str(self.cget("wrap")) == "word":
            return

        self._sizing_width = True
        try:
            blocks = self._current
            if not blocks:
                self.place_configure(width=1)
                return
            px_pad = int(self.cget("padx")) * 2
            natural = self._natural_width(blocks, px_pad)
            if natural == 0:
                self.place_configure(width=1)
                return

            win_w = self.winfo_toplevel().winfo_width() - 16
            if natural <= win_w:
                self.place_configure(width=natural)
            else:
                self.configure(wrap="word")
                self.place_configure(width=win_w)
        finally:
            self._sizing_width = False

    def _natural_width(self, blocks: tuple[Block, ...], px_pad: int) -> int:
        """Widest display-line pixel width from AST + font metrics."""
        mx = 0
        for block in blocks:
            match block:
                case Heading(text=inline):
                    mx = max(mx, self._inline_width(inline, "heading") + px_pad)
                case Paragraph(text=inline):
                    mx = max(mx, self._inline_width(inline, "plain") + px_pad)
                case CodeBlock(text=text):
                    code = self._fonts["code"]
                    for line in text.split("\n"):
                        w = code.measure(line or " ") + px_pad + 16  # lmargin
                        mx = max(mx, w)
                case List(items=items):
                    for item in items:
                        mx = max(mx, self._inline_width(item, "plain") + px_pad + 12)  # "• " prefix
                case Table() as table:
                    mx = max(mx, sum(self._table_widths(table)) + px_pad)
        return mx

    def _inline_width(self, inline: Inline, fallback: str) -> int:
        """Pixel width of one inline span sequence using per-span fonts."""
        return sum(self._fonts.get(tag, self._fonts[fallback]).measure(text) for text, tag in inline)

    def _fit_height(self) -> None:
        """Set height so all display lines (incl. tag spacing) are visible.

        ``dlineinfo`` clips reported height to the widget's visible area,
        so we expand to a generous height first.  The widget is moved
        off-screen during measurement so the expand→shrink cycle is
        invisible (no flicker).

        ``_fitted_h``/``_fitted_w`` cache prevents the
        ``<Configure>`` re-entry loop.
        """
        if not self._autoheight or self._autosizing or not self._font_scaled:
            return

        if self.winfo_width() <= 10:
            self.after(50, self._fit_height)
            return

        cur_w = self.winfo_width()
        if self._fitted_h is not None and self._fitted_w == cur_w:
            if abs(self.winfo_height() - self._fitted_h) <= 1:
                return

        self._autosizing = True
        try:
            text = self.get("1.0", "end-1c")
            if not text:
                self.place_configure(height="")
                self.configure(height=1)
                self._fitted_h = 1
                self._fitted_w = cur_w
                return

            # Move off-screen + expand so dlineinfo reports true geometry.
            saved_x = self.place_info().get("x", "") or "0"
            self.place_configure(x=-10000, height=2000)
            self.update_idletasks()

            # Walk display lines → actual required pixel height.
            idx = "1.0"
            max_bottom = 0
            for _ in range(200):
                dl = self.dlineinfo(idx)
                if dl is None:
                    break
                max_bottom = max(max_bottom, dl[1] + dl[3])
                nxt = self.index(f"{idx} + 1 displayline")
                if self.compare(nxt, "==", idx):
                    break
                idx = nxt

            # Restore position + set exact pixel height.
            self._fitted_h = max(1, max_bottom)
            self._fitted_w = cur_w
            self.place_configure(x=saved_x, height=self._fitted_h)
            self.update_idletasks()
        finally:
            self._autosizing = False
