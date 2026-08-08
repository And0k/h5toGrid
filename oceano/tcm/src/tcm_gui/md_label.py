"""Markdown-rendering Tk Text widget for help/status messages.

Renders a small Markdown subset directly into tagged Tk Text ranges.
The parser lives in :mod:`tcm._md_parse`; this module provides only
the Tk renderer (:class:`MarkdownLabel`).

Tables are aligned by measured tab stops.
"""

from __future__ import annotations

import logging
import tkinter as tk
import tkinter.font as tkfont
from itertools import accumulate, zip_longest
from typing import TypeAlias

from tcm._md_parse import Block, CodeBlock, Heading, Inline, Paragraph, Table, parse_markdown

FontSpec: TypeAlias = tkfont.Font | str | tuple[str | int, ...] | None

_l = logging.getLogger(__name__)

# ── Tk widget ────────────────────────────────────────────────────────────────


class MarkdownLabel(tk.Text):
    """Read-only label-like Tk Text widget rendering a Markdown subset."""

    _CELL_PAD = 8

    def __init__(
        self,
        master: tk.Misc | None = None,
        *,
        font: FontSpec = None,
        background: str | None = None,
        foreground: str | None = None,
        autoheight: bool = True,
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
        self._autosizing = False
        self._sizing_width = False
        self._font_scaled = False
        self._current: tuple[str, str] | None = None
        self._table_uid = 0

        self._fonts = self._build_fonts(font)
        self._configure_tags()
        self._raise_span_tags()

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

    def set_markdown(self, src: str, /) -> bool:
        """Render Markdown source. No-op if content is unchanged."""
        src = src or ""
        key = ("md", src)

        if key == self._current:
            return False

        self._current = key
        self._render(parse_markdown(src))
        return True

    def set_plain(self, text: str, /) -> bool:
        """Render plain text without Markdown interpretation."""
        text = text or ""
        key = ("plain", text)

        if key == self._current:
            return False

        self._current = key
        self._render_plain(text)
        return True

    def set_text(self, text: str, /, *, markdown: bool = False) -> bool:
        """Convenience dispatcher for existing status-publishing code."""
        return self.set_markdown(text) if markdown else self.set_plain(text)

    def clear(self) -> None:
        self.set_plain("")

    def fit_to_height(self, max_px: int) -> None:
        """Scale fonts down so a single line fits within *max_px* pixels."""
        if max_px <= 4:
            return
        plain = self._fonts["plain"]
        linespace = plain.metrics("linespace")
        if linespace <= max_px:
            self._font_scaled = True
            return
        ratio = max_px / linespace
        for f in self._fonts.values():
            cur = abs(int(f.cget("size")))
            new = max(6, int(cur * ratio))
            if new != cur:
                f.configure(size=new)
        self._font_scaled = True

    # ── rendering ────────────────────────────────────────────────────────

    def _render(self, blocks: tuple[Block, ...]) -> None:
        self.configure(state="normal", wrap="none")
        self.delete("1.0", "end")
        self._table_uid = 0

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

                case Table() as table:
                    self._render_table(table)

        if self.compare("end-1c", ">", "1.0") and self.get("end-2c", "end-1c") == "\n":
            self.delete("end-2c", "end-1c")

        self.configure(state="disabled")
        self.after_idle(self._fit_width)
        self.after_idle(self._fit_height)

    def _render_plain(self, text: str) -> None:
        self.configure(state="normal", wrap="none")
        self.delete("1.0", "end")

        if text:
            self.insert("end", text, "normal")

        self.configure(state="disabled")
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
        """Insert inline spans with correct tag application."""
        for text, tag in inline:
            tags = base_tags if tag == "plain" else (*base_tags, tag)
            self.insert("end", text, tags)

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
        self.tag_configure("code", font=self._fonts["code"], foreground="#C7254E", background="#F9F2F4")

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

    def _raise_span_tags(self) -> None:
        for tag in ("code", "bold", "italic"):
            self.tag_raise(tag)

    # ── size management ──────────────────────────────────────────────────

    def _on_configure(self, _event: tk.Event) -> None:
        """Widget resized — recompute width and height (only after font scaled)."""
        if self._font_scaled and not self._autosizing and not self._sizing_width:
            self.after_idle(self._fit_width)
            self.after_idle(self._fit_height)

    def _fit_width(self) -> None:
        """Measure natural content width and set ``place`` width accordingly.

        Starts with ``wrap="none"`` (set in ``__init__`` and reset on each
        content change).  Measures via font metrics (independent of current
        widget width).  If it fits within the window, keeps ``wrap="none"``
        at that width (tight contraction).  Otherwise switches to
        ``wrap="word"`` and caps to window width.

        Once ``wrap="word"`` is set, does not switch back — prevents
        oscillation between wrap modes on ``<Configure>`` cycles.
        """
        if not self._font_scaled or self._sizing_width:
            return

        # Already wrapping — don't re-measure (would see wrapped widths).
        if str(self.cget("wrap")) == "word":
            return

        self._sizing_width = True
        try:
            # Measure via font metrics — works regardless of current widget
            # width (dlineinfo fails when widget is collapsed to 1px).
            text = self.get("1.0", "end-1c")
            if not text:
                self.place_configure(width=1)
                return
            plain = self._fonts["plain"]
            px_pad = int(self.cget("padx")) * 2  # left + right internal padding
            natural = max(plain.measure(line) + px_pad for line in text.split("\n"))
            if natural == 0:
                self.place_configure(width=1)
                return

            win_w = self.winfo_toplevel().winfo_width() - 16  # margin
            if natural <= win_w:
                self.place_configure(width=natural)
            else:
                self.configure(wrap="word")
                self.place_configure(width=win_w)
        finally:
            self._sizing_width = False

    def _fit_height(self) -> None:
        """Set height so content is fully visible at current width.

        Deferred via ``after_idle`` / ``<Configure>``.  Retries if width
        not yet settled.  Only changes ``height`` — never width.
        Only runs after fonts have been scaled by ``fit_to_height``.
        """
        if not self._autoheight or self._autosizing or not self._font_scaled:
            return

        if self.winfo_width() <= 10:
            self.after(50, self._fit_height)
            return

        self._autosizing = True
        try:
            self.update_idletasks()
            count = self.count("1.0", "end-1c", "displaylines")
            lines = max(1, int(count[0]) if count else 1)

            if int(self.cget("height")) != lines:
                self.configure(height=lines)
                self.update_idletasks()

            last = self.index("end-1c")
            while not self.bbox(last) and lines < 50:
                lines += 1
                self.configure(height=lines)
                self.update_idletasks()
        finally:
            self._autosizing = False
