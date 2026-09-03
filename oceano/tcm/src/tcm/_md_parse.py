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
* ``[text](url)`` links → span ``(text, url)`` (the URL is the span tag;
  click handling lives in the renderer)

No HTML, no images, no blockquotes, no ordered (``1.``) lists.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import TypeAlias

# ── type aliases ─────────────────────────────────────────────────────────────

Tag = str
Tags = frozenset[Tag]
Span: TypeAlias = tuple[str, Tags]
Inline: TypeAlias = tuple[Span, ...]

# Style tags that affect font composition — single source of truth for
# :mod:`tcm._md_parse` and :mod:`tcm_gui.md_label` (imported there).
STYLE_TAGS: frozenset[Tag] = frozenset({"bold", "italic", "code"})

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

# Inline — built from named pieces for maintainability (order matters:
# escape → color → code → link → bold → italic; earlier wins at same pos).
_ESC = r"(?P<esc>\\[\\`*_{}\[\]()#+\-.!>~|])"
_COLOR = r"(?P<color>\{#(?P<color_name>[a-z_]+)\}(?P<color_text>.*?)\{/\})"
_CODE = r"(?P<code>`[^`]+`)"
_LINK = r"(?P<link>\[(?P<link_text>[^\]]*)\]\((?P<link_url>[^)]*)\))"
_BOLD = r"(?P<bold>\*\*(?P<bold_ast>.+?)\*\*|(?<!\w)__(?P<bold_und>.+?)__(?!\w))"
_ITALIC = r"(?P<italic>\*(?P<italic_ast>.+?)\*|(?<!\w)_(?P<italic_und>.+?)_(?!\w))"
_INLINE = re.compile("|".join((_ESC, _COLOR, _CODE, _LINK, _BOLD, _ITALIC)))


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
    """Coalesce adjacent spans with the same tag set into one."""
    merged: list[Span] = []
    for text, tags in spans:
        if not text:
            continue
        if merged and merged[-1][1] == tags:
            merged[-1] = (merged[-1][0] + text, tags)
        else:
            merged.append((text, tags))
    return tuple(merged)


def _nest(tag: Tag, inner: str) -> list[Span]:
    """Recursively parse *inner* and add *tag* to each resulting span's tags.

    DRY helper for bold/italic nesting: ``**`.yaml``` → ``.yaml`` tagged
    ``{bold,code}``.  Empty *inner* yields no spans.
    """
    return [(txt, tags | frozenset({tag})) for txt, tags in parse_inline(inner)] if inner else []


@lru_cache(maxsize=512)
def parse_inline(text: str) -> Inline:
    """Parse inline Markdown spans into ``(text, tags)`` tuples.

    Bold/italic content is recursively parsed, so nested markup like
    ``**`.yaml``` yields a span tagged *both* ``bold`` and ``code``.  Each
    span carries a frozenset of tags (style name / color name / link URL);
    plain text has the empty set.  Adjacent spans with identical tag sets
    are coalesced by :func:`_merge_spans`.
    """
    out: list[Span] = []
    pos = 0

    for m in _INLINE.finditer(text):
        if m.start() > pos:
            out.append((text[pos : m.start()], frozenset()))

        if esc := m.group("esc"):
            out.append((esc[1], frozenset()))

        elif m.group("color"):
            out.append((m.group("color_text"), frozenset({m.group("color_name")})))

        elif code := m.group("code"):
            out.append((code[1:-1], frozenset({"code"})))

        elif m.group("link"):
            out.append((m.group("link_text") or "", frozenset({m.group("link_url")})))

        elif m.group("bold"):
            out.extend(_nest("bold", m.group("bold_ast") or m.group("bold_und") or ""))

        elif m.group("italic"):
            out.extend(_nest("italic", m.group("italic_ast") or m.group("italic_und") or ""))

        pos = m.end()

    if pos < len(text):
        out.append((text[pos:], frozenset()))

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
