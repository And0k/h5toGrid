"""Tests for ``tcm._md_parse``: pure Markdown parser (zero Tk dependency).

Cover ``split_table_row``, ``parse_inline``, and ``parse_markdown``:
  * table row splitting with escaped pipes
  * inline spans: bold, italic, code, links, escape sequences
  * block parsing: headings, paragraphs, code fences, tables
  * dedup via ``@lru_cache`` (same input → same object)
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tcm._md_parse import (
    CodeBlock,
    Heading,
    Inline,
    Paragraph,
    Table,
    parse_inline,
    parse_markdown,
    split_table_row,
)


def _s(text: str, *tags: str):
    """Build one inline span: ``(text, frozenset(tags))`` — empty = plain."""
    return (text, frozenset(tags))


# ── split_table_row ──────────────────────────────────────────────────────────


class TestSplitTableRow:
    """Table row splitting with edge pipe handling and escaped literals."""

    @pytest.mark.parametrize(
        ("row", "expected"),
        [
            pytest.param("| a | b | c |", ("a", "b", "c"), id="simple"),
            pytest.param(" a | b | c ", ("a", "b", "c"), id="no-edge-pipes"),
            pytest.param("| `field` | type | desc |", ("`field`", "type", "desc"), id="backtick-cell"),
            pytest.param(
                r"| \|col\| | b |",
                ("|col|", "b"),
                id="escaped-pipes-restored",
            ),
            pytest.param("| single |", ("single",), id="single-cell"),
            pytest.param("| | empty |", ("", "empty"), id="empty-leading-cell"),
        ],
    )
    def test_split(self, row: str, expected: tuple[str, ...]):
        actual = split_table_row(row)
        assert actual == expected, f"split_table_row({row!r}): {expected=!r}, {actual=!r}"

    def test_returns_tuple(self):
        """Return type is tuple, not list — immutable."""
        result = split_table_row("| a | b |")
        assert isinstance(result, tuple), f"expected tuple, got {type(result).__name__}"


# ── parse_inline ─────────────────────────────────────────────────────────────


class TestParseInline:
    """Inline Markdown span parsing."""

    def test_plain_text(self):
        assert parse_inline("hello world") == (_s("hello world"),)

    def test_bold_asterisks(self):
        result = parse_inline("**bold**")
        assert result == (_s("bold", "bold"),), f"bold asterisks: {result!r}"

    def test_bold_underscores(self):
        result = parse_inline("__bold__")
        assert result == (_s("bold", "bold"),), f"bold underscores: {result!r}"

    def test_italic_asterisks(self):
        result = parse_inline("*italic*")
        assert result == (_s("italic", "italic"),), f"italic asterisks: {result!r}"

    def test_italic_underscores(self):
        result = parse_inline("_italic_")
        assert result == (_s("italic", "italic"),), f"italic underscores: {result!r}"

    def test_inline_code(self):
        result = parse_inline("`code`")
        assert result == (_s("code", "code"),), f"inline code: {result!r}"

    def test_link_text_only(self):
        """Links keep their text; the span tag is the target URL."""
        result = parse_inline("[click here](https://example.com)")
        assert result == (_s("click here", "https://example.com"),), f"link: {result!r}"

    def test_link_with_anchor(self):
        """A relative doc link with a heading anchor keeps text + target."""
        result = parse_inline("([подробнее](io_formats.md#directory-layout))")
        assert result == (
            _s("("),
            _s("подробнее", "io_formats.md#directory-layout"),
            _s(")"),
        ), f"anchored link: {result!r}"

    def test_link_adjacent(self):
        """Adjacent links with different targets stay separate spans."""
        result = parse_inline("[a](x)[b](y)")
        assert result == (_s("a", "x"), _s("b", "y")), f"adjacent links: {result!r}"

    def test_link_same_target_merged(self):
        """Adjacent links with the same target merge like any same-tag spans."""
        result = parse_inline("[a](x)[b](x)")
        assert result == (_s("ab", "x"),), f"merged link: {result!r}"

    def test_link_empty_url(self):
        """Brackets without a target keep the text as a (dead) link span."""
        result = parse_inline("[x]()")
        assert result == (_s("x", ""),), f"empty url: {result!r}"

    def test_link_empty_text(self):
        result = parse_inline("[](https://example.com)")
        assert result == (), f"empty link text dropped: {result!r}"

    def test_escape_sequence(self):
        result = parse_inline(r"\*not italic\*")
        assert result == (_s("*not italic*"),), f"escape: {result!r}"

    def test_mixed_bold_and_code(self):
        result = parse_inline("**bold** and `code`")
        assert result == (
            _s("bold", "bold"),
            _s(" and "),
            _s("code", "code"),
        ), f"mixed: {result!r}"

    def test_adjacent_same_tag_merged(self):
        """Adjacent spans with the same tag are coalesced."""
        result = parse_inline("**a****b**")
        assert len(result) == 1, f"expected 1 merged span, got {len(result)}"
        assert result[0] == _s("ab", "bold"), f"merged bold: {result!r}"

    def test_plain_leading_and_trailing(self):
        result = parse_inline("before **mid** after")
        assert result == (
            _s("before "),
            _s("mid", "bold"),
            _s(" after"),
        ), f"leading/trailing: {result!r}"

    def test_empty_string(self):
        assert parse_inline("") == (), f"empty: {parse_inline('')!r}"

    def test_color_tag_simple(self):
        result = parse_inline("{#error}err{/}")
        assert result == (_s("err", "error"),), f"color tag: {result!r}"

    def test_color_tag_in_sentence(self):
        result = parse_inline("before {#error}err{/} after")
        assert result == (
            _s("before "),
            _s("err", "error"),
            _s(" after"),
        ), f"color in sentence: {result!r}"

    def test_color_tag_adjacent(self):
        result = parse_inline("{#debug}d{/}{#error}e{/}")
        assert result == (
            _s("d", "debug"),
            _s("e", "error"),
        ), f"adjacent colors: {result!r}"

    def test_color_tag_with_bold(self):
        """Color tags and bold coexist as separate spans."""
        result = parse_inline("{#error}err{/} **bold**")
        assert result == (
            _s("err", "error"),
            _s(" "),
            _s("bold", "bold"),
        ), f"color + bold: {result!r}"

    def test_cache_returns_same_object(self):
        """Same input → same cached object (identity check)."""
        a = parse_inline("hello **world**")
        b = parse_inline("hello **world**")
        assert a is b, "parse_inline should be cached (lru_cache)"


class TestNestedFormatting:
    """Bold/italic wrapping inline code — generalizations of ``**`.yaml`**``."""

    def test_bold_wrapping_code(self):
        result = parse_inline("**`.yaml`**")
        assert result == (_s(".yaml", "bold", "code"),), f"bold+code: {result!r}"

    def test_bold_code_simple(self):
        result = parse_inline("**`code`**")
        assert result == (_s("code", "bold", "code"),), f"bold+code simple: {result!r}"

    def test_italic_wrapping_code(self):
        result = parse_inline("*`code`*")
        assert result == (_s("code", "italic", "code"),), f"italic+code: {result!r}"

    def test_bold_before_code_after(self):
        result = parse_inline("**before `code` after**")
        assert result == (
            _s("before ", "bold"),
            _s("code", "bold", "code"),
            _s(" after", "bold"),
        ), f"bold with code inside: {result!r}"

    def test_bold_italic_nested(self):
        result = parse_inline("**bold *italic* text**")
        assert result == (
            _s("bold ", "bold"),
            _s("italic", "bold", "italic"),
            _s(" text", "bold"),
        ), f"bold+italic: {result!r}"

    def test_code_literal_no_nesting(self):
        """Code spans are literal — `` `**bold**` `` must not recurse."""
        result = parse_inline("`**bold**`")
        assert result == (_s("**bold**", "code"),), f"code literal: {result!r}"

    def test_triple_nested(self):
        """``**_`nested`_**`` → bold+italic+code triple."""
        result = parse_inline("**_`nested`_**")
        assert result == (_s("nested", "bold", "italic", "code"),), f"triple: {result!r}"

    def test_conversely_bold_inside_code_not_parsed(self):
        """Inside `` `...` `` bold markers stay literal (already covered)."""
        result = parse_inline("`outer **bold**`")
        assert result == (_s("outer **bold**", "code"),), f"code outer: {result!r}"


# ── parse_markdown ───────────────────────────────────────────────────────────


class TestParseMarkdown:
    """Block-level Markdown parsing."""

    def test_heading(self):
        blocks = parse_markdown("# Title")
        assert len(blocks) == 1, f"expected 1 block, got {len(blocks)}"
        assert isinstance(blocks[0], Heading), f"expected Heading, got {type(blocks[0]).__name__}"
        assert blocks[0].level == 1, f"heading level: {blocks[0].level}"
        assert blocks[0].text == (_s("Title"),), f"heading text: {blocks[0].text!r}"

    def test_heading_levels(self):
        for level in range(1, 7):
            prefix = "#" * level
            blocks = parse_markdown(f"{prefix} H{level}")
            assert blocks[0].level == level, f"level {level}: got {blocks[0].level}"

    def test_paragraph(self):
        blocks = parse_markdown("some text")
        assert len(blocks) == 1
        assert isinstance(blocks[0], Paragraph)
        assert blocks[0].text == (_s("some text"),)

    def test_paragraph_multiline(self):
        """Consecutive non-blank lines merge into one paragraph."""
        blocks = parse_markdown("line one\nline two")
        assert len(blocks) == 1
        assert isinstance(blocks[0], Paragraph)
        # Joined by space
        assert "line one" in blocks[0].text[0][0] and "line two" in blocks[0].text[0][0]

    def test_code_block(self):
        src = textwrap.dedent("""\
            ```python
            x = 1
            y = 2
            ```
        """)
        blocks = parse_markdown(src)
        assert len(blocks) == 1
        assert isinstance(blocks[0], CodeBlock)
        assert "x = 1" in blocks[0].text and "y = 2" in blocks[0].text

    def test_code_block_tilde(self):
        src = "~~~\ncode here\n~~~"
        blocks = parse_markdown(src)
        assert len(blocks) == 1
        assert isinstance(blocks[0], CodeBlock)
        assert blocks[0].text == "code here"

    def test_table_basic(self):
        src = textwrap.dedent("""\
            | H1 | H2 |
            |----|----|
            | a  | b  |
            | c  | d  |
        """)
        blocks = parse_markdown(src)
        assert len(blocks) == 1
        assert isinstance(blocks[0], Table)
        t = blocks[0]
        assert len(t.header) == 2, f"header cells: {len(t.header)}"
        assert len(t.rows) == 2, f"data rows: {len(t.rows)}"

    def test_table_header_inline(self):
        """Bold in table header is parsed as inline spans."""
        src = "| **Name** | Value |\n|---|---|\n| x | 1 |"
        blocks = parse_markdown(src)
        t = blocks[0]
        assert isinstance(t, Table)
        # First header cell should have bold span
        assert t.header[0] == (_s("Name", "bold"),), f"header bold: {t.header[0]!r}"

    def test_empty_table_cell(self):
        src = "| A | |\n|---|---|\n| 1 | |"
        blocks = parse_markdown(src)
        t = blocks[0]
        assert isinstance(t, Table)
        assert t.rows[0][1] == (), f"empty cell → empty Inline: {t.rows[0][1]!r}"

    def test_blank_lines_separate_paragraphs(self):
        blocks = parse_markdown("first\n\nsecond")
        assert len(blocks) == 2
        assert all(isinstance(b, Paragraph) for b in blocks)

    def test_heading_then_paragraph(self):
        blocks = parse_markdown("# Title\n\nBody text")
        assert len(blocks) == 2
        assert isinstance(blocks[0], Heading)
        assert isinstance(blocks[1], Paragraph)

    def test_fenced_code_not_parsed_as_heading(self):
        """Lines starting with ``#`` inside fences are code, not headings."""
        src = "```\n# not a heading\n```"
        blocks = parse_markdown(src)
        assert len(blocks) == 1
        assert isinstance(blocks[0], CodeBlock)
        assert "# not a heading" in blocks[0].text

    def test_empty_input(self):
        assert parse_markdown("") == (), f"empty: {parse_markdown('')!r}"

    def test_whitespace_only(self):
        assert parse_markdown("   \n  \n") == (), f"whitespace-only: {parse_markdown('   \n  \n')!r}"

    def test_cache_returns_same_object(self):
        a = parse_markdown("# Hello\n\nWorld")
        b = parse_markdown("# Hello\n\nWorld")
        assert a is b, "parse_markdown should be cached (lru_cache)"

    def test_mixed_blocks(self):
        """Heading + paragraph + table + code fence in one document."""
        src = textwrap.dedent("""\
            # Intro

            Some text.

            | A | B |
            |---|---|
            | 1 | 2 |

            ```
            block
            ```
        """)
        blocks = parse_markdown(src)
        types = [type(b).__name__ for b in blocks]
        assert types == ["Heading", "Paragraph", "Table", "CodeBlock"], f"block types: {types}"

    def test_separator_row_not_emitted_as_table_data(self):
        """The ``|---|---|`` row is consumed as separator, not as data."""
        src = "| A |\n|---|\n| 1 |\n| 2 |"
        blocks = parse_markdown(src)
        t = blocks[0]
        assert isinstance(t, Table)
        assert len(t.rows) == 2, f"expected 2 data rows, got {len(t.rows)}"


# ── integration: _help uses split_table_row ──────────────────────────────────


class TestHelpUsesSharedSplitter:
    """Verify ``_help.parse_reference`` uses ``split_table_row`` from ``_md_parse``."""

    def test_escaped_pipe_in_help_short(self):
        """``\\|col\\|`` in a config_reference row is restored to literal ``|``."""
        from tcm_gui._help import parse_reference

        sample = textwrap.dedent("""\
            ## `input` — Section

            | Field | Type | Default | Purpose |
            |-------|------|---------|---------|
            | `mode` | `str` | `True` | See `\\|col\\|` shorthand. |
        """)
        entries = parse_reference(sample)
        assert "`|col|`" in entries["input.mode"].short, (
            f"escaped pipe not restored: {entries['input.mode'].short!r}"
        )


# ── MarkdownLabel rendering (requires Tk) ────────────────────────────────────


class TestMarkdownLabelRendering:
    """Verify tags are applied correctly and text is not duplicated."""

    @pytest.fixture(autouse=True, scope="class")
    @classmethod
    def _tk_root(cls, request):
        """Single Tk root for the entire class — skips if Tcl broken."""
        import tkinter as tk

        try:
            root = tk.Tk()
            root.withdraw()
        except tk.TclError:
            pytest.skip("Tk not available in this environment")
            return
        request.cls._root = root
        yield
        root.destroy()

    def _make_label(self, **kw):
        from tcm_gui.md_label import MarkdownLabel

        w = MarkdownLabel(self.__class__._root, font=("TkDefaultFont", 9), width=80, **kw)
        w.pack(fill="x")
        return w

    def test_no_text_duplication(self):
        """``**bold**`` renders as 'bold', not 'boldbold' (Tk insert tag bug)."""
        lbl = self._make_label()
        lbl.set_text("**bold** and *italic* and `code`")
        content = lbl.get("1.0", "end-1c")
        assert content == "bold and italic and code", f"duplication detected: {content!r}"

    def test_bold_tag_applied(self):
        lbl = self._make_label()
        lbl.set_text("**bold text**")
        pos = lbl.get("1.0", "end-1c").index("bold")
        assert "bold" in lbl.tag_names(f"1.{pos}"), (
            f"bold tag missing at 'bold' span: {lbl.tag_names(f'1.{pos}')!r}"
        )

    def test_italic_tag_applied(self):
        lbl = self._make_label()
        lbl.set_text("*italic text*")
        pos = lbl.get("1.0", "end-1c").index("italic")
        assert "italic" in lbl.tag_names(f"1.{pos}"), (
            f"italic tag missing at 'italic' span: {lbl.tag_names(f'1.{pos}')!r}"
        )

    def test_code_tag_applied(self):
        lbl = self._make_label()
        lbl.set_text("`code`")
        pos = lbl.get("1.0", "end-1c").index("code")
        assert "code" in lbl.tag_names(f"1.{pos}"), (
            f"code tag missing at 'code' span: {lbl.tag_names(f'1.{pos}')!r}"
        )

    def test_link_tag_applied(self):
        """``[text](url)`` renders text only, tagged ``link``, URL recorded."""
        lbl = self._make_label()
        lbl.set_text("[click here](io_formats.md#directory-layout)")
        content = lbl.get("1.0", "end-1c")
        assert content == "click here", f"url must not render: {content!r}"
        pos = content.index("click here")
        assert "link" in lbl.tag_names(f"1.{pos}"), f"link tag missing: {lbl.tag_names(f'1.{pos}')!r}"
        assert len(lbl._links) == 1, f"expected 1 recorded link: {lbl._links!r}"
        start, end, url = lbl._links[0]
        assert url == "io_formats.md#directory-layout", f"recorded url: {url!r}"
        assert lbl.get(start, end) == "click here", f"range text: {lbl.get(start, end)!r}"

    def test_link_at_returns_url(self):
        """``link_at`` resolves the URL under widget coordinates."""
        lbl = self._make_label()
        lbl.set_text("[click here](https://example.com)")
        lbl.update_idletasks()
        bb = lbl.bbox("1.0")
        if bb is None:
            pytest.skip("no geometry in headless env")
        assert lbl.link_at(bb[0] + 2, bb[1] + bb[3] // 2) == "https://example.com"

    def test_links_reset_on_rerender(self):
        lbl = self._make_label()
        lbl.set_text("[a](x)")
        lbl.set_text("plain text")
        assert lbl._links == [], f"stale links after rerender: {lbl._links!r}"

    def test_raw_never_links(self):
        """``raw=True`` (untrusted content) must never produce link spans."""
        lbl = self._make_label()
        lbl.set_text("[a](x)", raw=True)
        assert "link" not in lbl.tag_names("1.0"), f"raw leaked link tag: {lbl.tag_names('1.0')!r}"
        assert lbl._links == []

    def test_color_tag_rendered(self):
        """{#name}text{/} applies color map foreground."""
        colors = {"error": "#CC0000", "debug": "#808080"}
        lbl = self._make_label(colors=colors)
        lbl.set_text("{#error}err{/} plain {#debug}dbg{/}")
        content = lbl.get("1.0", "end-1c")

        pos_err = content.index("err")
        tags_err = lbl.tag_names(f"1.{pos_err}")
        assert "error" in tags_err, f"'error' tag missing at colored span: {tags_err!r}"

        pos_plain = content.index("plain")
        tags_plain = lbl.tag_names(f"1.{pos_plain}")
        assert "error" not in tags_plain, f"'error' tag leaked to plain span: {tags_plain!r}"

        pos_dbg = content.index("dbg")
        tags_dbg = lbl.tag_names(f"1.{pos_dbg}")
        assert "debug" in tags_dbg, f"'debug' tag missing at colored span: {tags_dbg!r}"

    def test_color_tag_without_map(self):
        """Without colors dict, {#name} tags are applied but not configured."""
        lbl = self._make_label()
        lbl.set_text("{#error}err{/}")
        content = lbl.get("1.0", "end-1c")
        pos = content.index("err")
        tags = lbl.tag_names(f"1.{pos}")
        assert "error" in tags, f"'error' tag should still be applied: {tags!r}"

    def test_color_tag_foreground_value(self):
        """Color map value is set as the tag's foreground."""
        colors = {"error": "#CC0000"}
        lbl = self._make_label(colors=colors)
        lbl.set_text("{#error}err{/}")
        fg = lbl.tag_cget("error", "foreground")
        assert fg == "#CC0000", f"error tag foreground: expected #CC0000, got {fg!r}"

    def test_fonts_visually_distinct(self):
        """Bold/italic/code fonts have distinct properties."""
        lbl = self._make_label()
        assert lbl._fonts["bold"].cget("weight") == "bold", "bold font weight"
        assert lbl._fonts["italic"].cget("slant") == "italic", "italic font slant"
        assert lbl._fonts["code"].cget("family") == "Consolas", "code font family"

    def test_plain_text_no_tags(self):
        """Plain text has only 'normal' tag, no inline markup."""
        lbl = self._make_label()
        lbl.set_text("hello world", raw=True)
        tags = lbl.tag_names("1.0")
        assert "normal" in tags, f"normal tag missing: {tags!r}"
        assert "bold" not in tags, f"unexpected bold tag: {tags!r}"

    def test_default_parses_markdown(self):
        """Default path parses ``**bold**`` — regression for STR chrome status (app.py:248).

        ``set_text`` no longer takes ``markdown=False``: it parses by
        default, so chrome hover status from ``STR["{role}.status"]`` honors
        ``**bold**`` without callers passing any flag.
        """
        lbl = self._make_label()
        lbl.set_text("Changing path **resets tabs**")
        content = lbl.get("1.0", "end-1c")
        assert content == "Changing path resets tabs", f"default parse should strip ** markers: {content!r}"
        pos = content.index("resets tabs")
        assert "bold" in lbl.tag_names(f"1.{pos}"), (
            f"bold tag missing in default-parse mode: {lbl.tag_names(f'1.{pos}')!r}"
        )

    def test_raw_bypasses_markdown_parsing(self):
        """``raw=True`` inserts literal text — ``**`` survives uninterpreted.

        Used at sites that interpolate untrusted content (filesystem paths in
        ``tab.status`` after ``.format(path=...)``); prevents ``_``/``*``/``\\``
        in the substitution from being reinterpreted as inline markup.
        """
        lbl = self._make_label()
        lbl.set_text("path/with_underscøre **not bold**", raw=True)
        content = lbl.get("1.0", "end-1c")
        assert content == "path/with_underscøre **not bold**", f"raw should preserve literal **: {content!r}"
        assert "bold" not in lbl.tag_names("1.0"), f"raw must not apply bold tag: {lbl.tag_names('1.0')!r}"

    def test_single_line_height(self):
        """Single-line text stays at height=1."""
        lbl = self._make_label()
        lbl.set_text("Ready", raw=True)
        assert int(lbl.cget("height")) == 1, f"single-line height: {lbl.cget('height')}"

    def test_multi_line_height(self):
        """Multi-paragraph markdown gets height > 1 when widget has real width."""
        import tkinter as tk

        root = self.__class__._root
        # Give the frame a real width so the label gets a real layout.
        root.geometry("600x400")
        f = tk.Frame(root, width=600, height=30)
        f.pack(fill="x")
        f.pack_propagate(False)

        from tcm_gui.md_label import MarkdownLabel

        lbl = MarkdownLabel(f)
        lbl.place(relx=0, rely=1.0, anchor="sw", relwidth=0.667)
        lbl.set_text("line1\n\nline2\n\nline3")
        # Process after_idle (_fit_height) + Configure events.
        for _ in range(5):
            root.update()
        h = int(lbl.cget("height"))
        # In headless env the widget may not get real width — skip if so.
        if lbl.winfo_width() <= 10:
            pytest.skip("no real widget width in headless env")
        assert h > 1, f"multi-line height: {h}"


class TestLinkEvents:
    """Coordinate/event link behavior — needs a real mapped root.

    ``link_at``/click/cursor resolve widget coordinates via ``bbox``, which
    is degenerate on a withdrawn root — this class maps its own root.
    """

    @pytest.fixture(autouse=True)
    def _mapped_root(self):
        import tkinter as tk

        try:
            root = tk.Tk()
        except tk.TclError:
            pytest.skip("Tk not available")
            return
        root.geometry("800x200")
        root.update_idletasks()
        yield root
        root.destroy()

    def _make_label(self, root, **kw):
        from tcm_gui.md_label import MarkdownLabel

        w = MarkdownLabel(root, font=("TkDefaultFont", 9), width=80, **kw)
        w.pack(fill="x")
        return w

    @staticmethod
    def _link_coords(lbl):
        """Pointer coords inside the first link text char, or skip headless."""
        lbl.update_idletasks()
        bb = lbl.bbox("1.0")
        if bb is None:
            pytest.skip("no geometry in headless env")
        return bb[0] + 2, bb[1] + bb[3] // 2

    def test_link_at_returns_url(self, _mapped_root):
        lbl = self._make_label(_mapped_root)
        lbl.set_text("[click here](https://example.com)")
        x, y = self._link_coords(lbl)
        assert lbl.link_at(x, y) == "https://example.com"
        last = lbl.index("end-1c")
        lbl.update_idletasks()
        bb = lbl.bbox(last)
        if bb is not None:  # past the text end → outside every link range
            assert lbl.link_at(bb[0] + bb[2] + 4, bb[1] + bb[3] // 2) is None

    def test_link_on_link_callback(self, _mapped_root):
        """``_open_link`` (bound to the tag's ``<Button-1>``) forwards ``(url, base)``.

        Text tag button bindings do not fire from ``event_generate`` (Tk
        processes real input-manager events only), so the handler is invoked
        directly with a synthetic event — the same call the tag binding makes.
        """
        from types import SimpleNamespace

        calls = []
        lbl = self._make_label(_mapped_root, on_link=lambda url, base: calls.append((url, base)))
        lbl.set_text("[a](b.md#c)", base="D:/docs/reference")
        x, y = self._link_coords(lbl)
        lbl._open_link(SimpleNamespace(x=x, y=y))
        lbl.update()
        assert calls == [("b.md#c", Path("D:/docs/reference"))], f"on_link calls: {calls!r}"

    def test_link_hover_cursor(self, _mapped_root):
        """Hovering a link switches the widget cursor to ``hand2``."""
        lbl = self._make_label(_mapped_root)
        lbl.set_text("[a](b.md)")
        x, y = self._link_coords(lbl)
        lbl.event_generate("<Motion>", x=x, y=y)
        lbl.update()
        assert str(lbl.cget("cursor")) == "hand2", f"cursor over link: {lbl.cget('cursor')!r}"


class TestHeightSufficient:
    """Verify _fit_height sets widget height so last display line is fully visible.

    Uses a real mapped window (not withdrawn) so dlineinfo and winfo_height
    return accurate pixel values.  Tests at multiple fit_to_height values
    to cover different UI_SCALE scenarios.
    """

    @pytest.fixture()
    def _mapped_root(self):
        import tkinter as tk

        try:
            root = tk.Tk()
        except tk.TclError:
            pytest.skip("Tk not available")
        root.geometry("800x300")
        root.update_idletasks()
        yield root
        root.destroy()

    @staticmethod
    def _assert_last_line_visible(lbl, root, label):
        """Assert the last display line is fully within the widget."""
        # Exhaust pending after_idle / Configure / after(50) callbacks.
        for _ in range(10):
            root.update_idletasks()
            root.update()

        last = lbl.index("end-1c")
        dl = lbl.dlineinfo(last)
        wh = lbl.winfo_height()
        ww = lbl.winfo_width()
        if ww <= 10:
            pytest.skip("no real widget width in headless env")
        assert dl, (
            f"{label}: dlineinfo(last) is empty — last char not rendered. winfo_height={wh}, winfo_width={ww}"
        )
        dl_bottom = dl[1] + dl[3]
        assert dl_bottom <= wh, (
            f"{label}: last display line clipped — "
            f"dl_bottom={dl_bottom} > winfo_height={wh}. "
            f"dlineinfo={dl}, width={ww}"
        )

    @pytest.mark.parametrize(
        "bar_h, text, raw",
        [
            pytest.param(22, "Ready", True, id="single-line-bar22"),
            pytest.param(16, "Ready", True, id="single-line-bar16"),
            pytest.param(14, "Ready", True, id="single-line-bar14"),
            pytest.param(
                22,
                "**Source of truth** for time window. **Auto-populated** from data.",
                False,
                id="inline-bold-bar22",
            ),
            pytest.param(
                16,
                "**Source of truth** for time window. **Auto-populated** from data.",
                False,
                id="inline-bold-bar16",
            ),
            pytest.param(
                14,
                "**Source of truth** for time window. **Auto-populated** from data.",
                False,
                id="inline-bold-bar14",
            ),
            pytest.param(22, "line one\nline two\nline three", False, id="3lines-bar22"),
            pytest.param(16, "line one\nline two\nline three", False, id="3lines-bar16"),
            pytest.param(14, "line one\nline two\nline three", False, id="3lines-bar14"),
            pytest.param(
                16,
                "# Heading\n\nParagraph text here.",
                False,
                id="heading-para-bar16",
            ),
            pytest.param(
                14,
                "# Heading\n\nParagraph text here.",
                False,
                id="heading-para-bar14",
            ),
        ],
    )
    def test_last_line_visible(self, _mapped_root, bar_h, text, raw):
        import tkinter as tk

        from tcm_gui.md_label import MarkdownLabel

        root = _mapped_root
        f = tk.Frame(root, width=800, height=30)
        f.pack(fill="x")
        f.pack_propagate(False)

        lbl = MarkdownLabel(f)
        lbl.place(relx=0, rely=1.0, anchor="sw", relwidth=1.0)
        lbl.fit_to_height(bar_h)
        lbl.set_text(text, raw=raw)
        self._assert_last_line_visible(lbl, root, f"bar_h={bar_h}")

    def test_wrapped_2row_height(self, _mapped_root):
        """Wrapped text that spans 2 display lines must be fully visible.

        Uses a narrow widget (180px) so _fit_width switches to wrap='word'
        and the text wraps into multiple display lines.
        """
        import tkinter as tk

        from tcm_gui.md_label import MarkdownLabel

        root = _mapped_root
        # Narrow frame — forces wrapping for medium-length text.
        f = tk.Frame(root, width=180, height=200)
        f.pack(fill="x")
        f.pack_propagate(False)

        lbl = MarkdownLabel(f)
        lbl.place(relx=0, rely=1.0, anchor="sw", x=0, y=0)
        lbl.fit_to_height(18)
        lbl.set_text("Changing data path rescans and **resets all config tabs below**")
        self._assert_all_display_lines_visible(lbl, root)

    def test_wrapped_heading_and_paragraph(self, _mapped_root):
        """Heading + wrapped paragraph at narrow width must be fully visible."""
        import tkinter as tk

        from tcm_gui.md_label import MarkdownLabel

        root = _mapped_root
        f = tk.Frame(root, width=160, height=200)
        f.pack(fill="x")
        f.pack_propagate(False)

        lbl = MarkdownLabel(f)
        lbl.place(relx=0, rely=1.0, anchor="sw", x=0, y=0)
        lbl.fit_to_height(18)
        lbl.set_text("# Status\n\nSource of truth for time window. Auto-populated from data.")
        self._assert_all_display_lines_visible(lbl, root)

    @staticmethod
    def _assert_all_display_lines_visible(lbl, root):
        """Assert every display line is within the widget bounds."""
        for _ in range(10):
            root.update_idletasks()
            root.update()

        wh = lbl.winfo_height()
        ww = lbl.winfo_width()
        if ww <= 10:
            pytest.skip("no real widget width in headless env")

        idx = "1.0"
        for _ in range(200):
            dl = lbl.dlineinfo(idx)
            if dl is None:
                break
            bottom = dl[1] + dl[3]
            assert bottom <= wh + 1, (
                f"display line at {idx} clipped — "
                f"bottom={bottom} > winfo_height={wh}. "
                f"dlineinfo={dl}, width={ww}"
            )
            nxt = lbl.index(f"{idx} + 1 displayline")
            if lbl.compare(nxt, "==", idx):
                break
            idx = nxt


class TestFitWidth:
    """Verify _fit_width measures inline spans with their actual fonts."""

    @pytest.fixture()
    def _mapped_root(self):
        import tkinter as tk

        try:
            root = tk.Tk()
        except tk.TclError:
            pytest.skip("Tk not available")
        root.geometry("1200x300")
        root.update_idletasks()
        yield root
        root.destroy()

    def test_code_span_not_clipped(self, _mapped_root):
        """Inline code (Consolas) is wider than bold — width must account for it."""
        import tkinter as tk

        from tcm_gui.md_label import MarkdownLabel

        root = _mapped_root
        f = tk.Frame(root, width=1200, height=30)
        f.pack(fill="x")
        f.pack_propagate(False)

        lbl = MarkdownLabel(f)
        lbl.place(relx=0, rely=1.0, anchor="sw")
        lbl.fit_to_height(18)
        lbl.set_text("Polynomial: `Vabs(inclination)`")

        # Exhaust pending callbacks.
        for _ in range(10):
            root.update_idletasks()
            root.update()

        ww = lbl.winfo_width()
        if ww <= 10:
            pytest.skip("no real widget width in headless env")

        # The last character must be within the widget bounds.
        last = lbl.index("end-1c")
        bb = lbl.bbox(last)
        assert bb is not None, f"last char invisible — widget too narrow: w={ww}"
        # bbox x+width must fit within widget width.
        assert bb[0] + bb[2] <= ww + 1, (
            f"last char clipped: bbox right={bb[0] + bb[2]} > widget_w={ww}. "
            f"code font (Consolas) wider than bold — _fit_width underestimates."
        )

    def test_bold_inline_sufficient(self, _mapped_root):
        """Bold-only inline text — width must be sufficient."""
        import tkinter as tk

        from tcm_gui.md_label import MarkdownLabel

        root = _mapped_root
        f = tk.Frame(root, width=1200, height=30)
        f.pack(fill="x")
        f.pack_propagate(False)

        lbl = MarkdownLabel(f)
        lbl.place(relx=0, rely=1.0, anchor="sw")
        lbl.fit_to_height(18)
        lbl.set_text("**Source of truth** for time window")

        for _ in range(10):
            root.update_idletasks()
            root.update()

        ww = lbl.winfo_width()
        if ww <= 10:
            pytest.skip("no real widget width in headless env")

        last = lbl.index("end-1c")
        bb = lbl.bbox(last)
        assert bb is not None, f"last char invisible — w={ww}"
        assert bb[0] + bb[2] <= ww + 1, f"last char clipped: bbox right={bb[0] + bb[2]} > widget_w={ww}"
