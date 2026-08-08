"""Tests for ``tcm._md_parse``: pure Markdown parser (zero Tk dependency).

Cover ``split_table_row``, ``parse_inline``, and ``parse_markdown``:
  * table row splitting with escaped pipes
  * inline spans: bold, italic, code, links, escape sequences
  * block parsing: headings, paragraphs, code fences, tables
  * dedup via ``@lru_cache`` (same input → same object)
"""

from __future__ import annotations

import textwrap

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
        assert parse_inline("hello world") == (("hello world", "plain"),)

    def test_bold_asterisks(self):
        result = parse_inline("**bold**")
        assert result == (("bold", "bold"),), f"bold asterisks: {result!r}"

    def test_bold_underscores(self):
        result = parse_inline("__bold__")
        assert result == (("bold", "bold"),), f"bold underscores: {result!r}"

    def test_italic_asterisks(self):
        result = parse_inline("*italic*")
        assert result == (("italic", "italic"),), f"italic asterisks: {result!r}"

    def test_italic_underscores(self):
        result = parse_inline("_italic_")
        assert result == (("italic", "italic"),), f"italic underscores: {result!r}"

    def test_inline_code(self):
        result = parse_inline("`code`")
        assert result == (("code", "code"),), f"inline code: {result!r}"

    def test_link_text_only(self):
        """Links render as plain text — no click handling."""
        result = parse_inline("[click here](https://example.com)")
        assert result == (("click here", "plain"),), f"link: {result!r}"

    def test_link_empty_text(self):
        result = parse_inline("[](https://example.com)")
        assert result == (), f"empty link text dropped: {result!r}"

    def test_escape_sequence(self):
        result = parse_inline(r"\*not italic\*")
        assert result == (("*not italic*", "plain"),), f"escape: {result!r}"

    def test_mixed_bold_and_code(self):
        result = parse_inline("**bold** and `code`")
        assert result == (
            ("bold", "bold"),
            (" and ", "plain"),
            ("code", "code"),
        ), f"mixed: {result!r}"

    def test_adjacent_same_tag_merged(self):
        """Adjacent spans with the same tag are coalesced."""
        result = parse_inline("**a****b**")
        assert len(result) == 1, f"expected 1 merged span, got {len(result)}"
        assert result[0] == ("ab", "bold"), f"merged bold: {result!r}"

    def test_plain_leading_and_trailing(self):
        result = parse_inline("before **mid** after")
        assert result == (
            ("before ", "plain"),
            ("mid", "bold"),
            (" after", "plain"),
        ), f"leading/trailing: {result!r}"

    def test_empty_string(self):
        assert parse_inline("") == (), f"empty: {parse_inline('')!r}"

    def test_cache_returns_same_object(self):
        """Same input → same cached object (identity check)."""
        a = parse_inline("hello **world**")
        b = parse_inline("hello **world**")
        assert a is b, "parse_inline should be cached (lru_cache)"


# ── parse_markdown ───────────────────────────────────────────────────────────


class TestParseMarkdown:
    """Block-level Markdown parsing."""

    def test_heading(self):
        blocks = parse_markdown("# Title")
        assert len(blocks) == 1, f"expected 1 block, got {len(blocks)}"
        assert isinstance(blocks[0], Heading), f"expected Heading, got {type(blocks[0]).__name__}"
        assert blocks[0].level == 1, f"heading level: {blocks[0].level}"
        assert blocks[0].text == (("Title", "plain"),), f"heading text: {blocks[0].text!r}"

    def test_heading_levels(self):
        for level in range(1, 7):
            prefix = "#" * level
            blocks = parse_markdown(f"{prefix} H{level}")
            assert blocks[0].level == level, f"level {level}: got {blocks[0].level}"

    def test_paragraph(self):
        blocks = parse_markdown("some text")
        assert len(blocks) == 1
        assert isinstance(blocks[0], Paragraph)
        assert blocks[0].text == (("some text", "plain"),)

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
        assert t.header[0] == (("Name", "bold"),), f"header bold: {t.header[0]!r}"

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
    def _tk_root(self, request):
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
        lbl.set_markdown("**bold** and *italic* and `code`")
        content = lbl.get("1.0", "end-1c")
        assert content == "bold and italic and code", f"duplication detected: {content!r}"

    def test_bold_tag_applied(self):
        lbl = self._make_label()
        lbl.set_markdown("**bold text**")
        pos = lbl.get("1.0", "end-1c").index("bold")
        assert "bold" in lbl.tag_names(f"1.{pos}"), (
            f"bold tag missing at 'bold' span: {lbl.tag_names(f'1.{pos}')!r}"
        )

    def test_italic_tag_applied(self):
        lbl = self._make_label()
        lbl.set_markdown("*italic text*")
        pos = lbl.get("1.0", "end-1c").index("italic")
        assert "italic" in lbl.tag_names(f"1.{pos}"), (
            f"italic tag missing at 'italic' span: {lbl.tag_names(f'1.{pos}')!r}"
        )

    def test_code_tag_applied(self):
        lbl = self._make_label()
        lbl.set_markdown("`code`")
        pos = lbl.get("1.0", "end-1c").index("code")
        assert "code" in lbl.tag_names(f"1.{pos}"), (
            f"code tag missing at 'code' span: {lbl.tag_names(f'1.{pos}')!r}"
        )

    def test_fonts_visually_distinct(self):
        """Bold/italic/code fonts have distinct properties."""
        lbl = self._make_label()
        assert lbl._fonts["bold"].cget("weight") == "bold", "bold font weight"
        assert lbl._fonts["italic"].cget("slant") == "italic", "italic font slant"
        assert lbl._fonts["code"].cget("family") == "Consolas", "code font family"

    def test_plain_text_no_tags(self):
        """Plain text has only 'normal' tag, no inline markup."""
        lbl = self._make_label()
        lbl.set_plain("hello world")
        tags = lbl.tag_names("1.0")
        assert "normal" in tags, f"normal tag missing: {tags!r}"
        assert "bold" not in tags, f"unexpected bold tag: {tags!r}"

    def test_single_line_height(self):
        """Single-line text stays at height=1."""
        lbl = self._make_label()
        lbl.set_plain("Ready")
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
        lbl.set_markdown("line1\n\nline2\n\nline3")
        # Process after_idle (_fit_height) + Configure events.
        for _ in range(5):
            root.update()
        h = int(lbl.cget("height"))
        # In headless env the widget may not get real width — skip if so.
        if lbl.winfo_width() <= 10:
            pytest.skip("no real widget width in headless env")
        assert h > 1, f"multi-line height: {h}"
