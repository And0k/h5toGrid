"""Pure Markdown parser — zero Tk dependency.

Parses a Markdown subset into a small block AST consumed by
:mod:`tcm_gui.md_label` (Tk renderer) and :mod:`tcm_gui._help` (table row
splitting).  Supported constructs:

* paragraphs, ``#``–``######`` headings
* ``**bold**``, ``__bold__``, ``*italic*``, ``_italic_``
* `` `inline code` ``
* ``{#name}colored text{/}`` — color tag (name resolved by renderer's color map)
* ``` ``` fenced code blocks ```
* Markdown tables (pipe-delimited)
* ``- `` unordered list items (flat; indented continuation lines fold into
  the preceding item — nested lists are not supported)
* ``[text](url)`` links → plain text (no click handling)

No HTML, no images, no blockquotes, no ordered (``1.``) lists.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import TypeAlias

# ── type aliases ─────────────────────────────────────────────────────────────

Tag = str
Span: TypeAlias = tuple[str, Tag]
Inline: TypeAlias = tuple[Span, ...]

# ── AST ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Heading:
    level: int
    text: Inline


@dataclass(frozen=True, slots=True)
class Paragraph:
    text: Inline


@dataclass(frozen=True, slots=True)
class CodeBlock:
    text: str


@dataclass(frozen=True, slots=True)
class Table:
    header: tuple[Inline, ...]
    rows: tuple[tuple[Inline, ...], ...]


@dataclass(frozen=True, slots=True)
class List:
    items: tuple[Inline, ...]


Block = Heading | Paragraph | CodeBlock | Table | List

# ── regex primitives ─────────────────────────────────────────────────────────

# Cell separator: a ``|`` not preceded by ``\`` (markdown table escape).
# ``\|`` inside a cell is a literal pipe — kept intact after splitting.
_CELL_SEP = re.compile(r"(?<!\\)\|")

# Table separator row: ``| --- | :---: | ---: |`` etc.
_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)*\|?\s*$")

# Heading: ``# text`` through ``###### text``, optional trailing ``#``.
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")

# Code fence toggle: ``` or ~~~.
_FENCE = re.compile(r"^\s*(```|~~~)")

# Unordered list item: ``- text`` at column 0 (column-0 only → nested indented
# markers don't match; their lines fold into the parent item as continuation).
_LIST_ITEM = re.compile(r"^-\s+(.+?)\s*$")

# Inline patterns: escape sequences, color tags, inline code, links, bold, italic.
_INLINE = re.compile(
    r"(?P<esc>\\[\\`*_{}\[\]()#+\-.!>~|])"
    r"|(?P<color>\{#(?P<color_name>[a-z_]+)\}(?P<color_text>.*?)\{/\})"
    r"|(?P<code>`[^`]+`)"
    r"|(?P<link>\[(?P<link_text>[^\]]*)\]\([^)]*\))"
    r"|(?P<bold>\*\*(?P<bold_ast>.+?)\*\*|(?<!\w)__(?P<bold_und>.+?)__(?!\w))"
    r"|(?P<italic>\*(?P<italic_ast>.+?)\*|(?<!\w)_(?P<italic_und>.+?)_(?!\w))"
)


# ── table row splitting ──────────────────────────────────────────────────────


def split_table_row(row: str) -> tuple[str, ...]:
    """Split one Markdown table row; escaped pipes become literal pipes."""
    inner = row.strip()

    if inner.startswith("|"):
        inner = inner[1:]
    if inner.endswith("|") and not inner.endswith(r"\|"):
        inner = inner[:-1]

    return tuple(p.replace(r"\|", "|").strip() for p in _CELL_SEP.split(inner))


def _is_table_row(line: str) -> bool:
    return "|" in line and not _TABLE_SEPARATOR.match(line) and bool(split_table_row(line))


# ── inline parsing ───────────────────────────────────────────────────────────


def _merge_spans(spans: list[Span]) -> Inline:
    """Coalesce adjacent spans with the same tag into one."""
    merged: list[Span] = []
    for text, tag in spans:
        if not text:
            continue
        if merged and merged[-1][1] == tag:
            merged[-1] = (merged[-1][0] + text, tag)
        else:
            merged.append((text, tag))
    return tuple(merged)


@lru_cache(maxsize=512)
def parse_inline(text: str) -> Inline:
    """Parse inline Markdown spans into ``(text, tag)`` tuples."""
    out: list[Span] = []
    pos = 0

    for m in _INLINE.finditer(text):
        if m.start() > pos:
            out.append((text[pos : m.start()], "plain"))

        if esc := m.group("esc"):
            out.append((esc[1], "plain"))

        elif m.group("color"):
            out.append((m.group("color_text"), m.group("color_name")))

        elif code := m.group("code"):
            out.append((code[1:-1], "code"))

        elif m.group("link"):
            # Simplified: display link text only; no click handling.
            out.append((m.group("link_text") or "", "plain"))

        elif m.group("bold"):
            out.append((m.group("bold_ast") or m.group("bold_und") or "", "bold"))

        elif m.group("italic"):
            out.append((m.group("italic_ast") or m.group("italic_und") or "", "italic"))

        pos = m.end()

    if pos < len(text):
        out.append((text[pos:], "plain"))

    return _merge_spans(out)


# ── block parsing ────────────────────────────────────────────────────────────


@lru_cache(maxsize=128)
def parse_markdown(src: str) -> tuple[Block, ...]:
    """Parse Markdown source into a small block AST."""
    lines = src.splitlines()
    blocks: list[Block] = []
    para: list[str] = []
    code: list[str] = []
    fence: str | None = None
    i = 0
    n = len(lines)

    def flush_para() -> None:
        nonlocal para
        if text := " ".join(para).strip():
            blocks.append(Paragraph(parse_inline(text)))
        para = []

    while i < n:
        line = lines[i]

        if fence:
            if line.strip().startswith(fence):
                blocks.append(CodeBlock("\n".join(code)))
                fence, code = None, []
            else:
                code.append(line)
            i += 1
            continue

        if m := _FENCE.match(line):
            flush_para()
            fence = m.group(1)
            i += 1
            continue

        if not line.strip():
            flush_para()
            i += 1
            continue

        if i + 1 < n and _is_table_row(line) and _TABLE_SEPARATOR.match(lines[i + 1]):
            flush_para()
            header = tuple(parse_inline(cell) for cell in split_table_row(line))
            i += 2

            rows: list[tuple[Inline, ...]] = []
            while i < n and _is_table_row(lines[i]):
                rows.append(tuple(parse_inline(cell) for cell in split_table_row(lines[i])))
                i += 1

            blocks.append(Table(header, tuple(rows)))
            continue

        if m := _HEADING.match(line):
            flush_para()
            blocks.append(Heading(len(m.group(1)), parse_inline(m.group(2))))
            i += 1
            continue

        # Unordered list: ``- `` marker at column 0; indented continuation
        # lines (≥1 leading space, not a marker) fold into the current item.
        if m := _LIST_ITEM.match(line):
            flush_para()
            items: list[Inline] = []
            cur: list[str] = [m.group(1)]
            i += 1
            while i < n:
                ln = lines[i]
                if not ln.strip() or _FENCE.match(ln) or _HEADING.match(ln):
                    break
                if cm := _LIST_ITEM.match(ln):
                    items.append(parse_inline(" ".join(s.strip() for s in cur)))
                    cur = [cm.group(1)]
                    i += 1
                    continue
                if ln.startswith(" "):  # indented continuation
                    cur.append(ln)
                    i += 1
                    continue
                break  # non-indented non-marker → list ends
            items.append(parse_inline(" ".join(s.strip() for s in cur)))
            blocks.append(List(tuple(items)))
            continue

        para.append(line.strip())
        i += 1

    if fence:
        blocks.append(CodeBlock("\n".join(code)))

    flush_para()
    return tuple(blocks)
