"""Resolve Hydra config paths to help text auto-extracted from ``config_reference.md``.

Single source of truth for non-chrome (config-cell) tooltips: the existing
field tables in ``docs/tcm_cli/config_reference.md``.

Table-driven sections
---------------------
For each field section heading of the form:
    ## `input`
    ## `input.coefs`

the parser scans markdown table rows whose first cell is a backticked field
identifier:

    | `field_name` | ... | Purpose / Physical meaning |

and emits one :class:`HelpEntry` keyed by ``{section}.{field}``.  The last
cell becomes :attr:`HelpEntry.short`.

Mode-tagged detail sections
---------------------------
When a field's meaning depends on context, detailed documentation is written
as mode-tagged ``###`` subsections:

    ### `input.path` <mode>probe</mode>
    ### `input.path` <mode>search</mode>

These are stored in :attr:`HelpEntry.body` keyed by mode.

Detail sub-blocks
-----------------
Inside a mode section, ``####`` headings define named detail blocks:

    #### Detailed

Lines before the first ``####`` are stored as the mode's short body.
Named ``####`` blocks are stored in :attr:`ModeBody.details`.


Arrays are resolved at the field level: ``Ag[0]`` / ``Ag[1][2]`` are stripped
to ``Ag`` before lookup.

Localization uses ``config_reference_{lang}.md`` when present, with fallback
to ``config_reference.md``.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

from tcm import _constants
from tcm._md_parse import split_table_row

from ._i18n import resolve_lang

_l = logging.getLogger(__name__)

# Sections whose ``## ``{section}```` heading opens table-driven field scanning.
# ``filter/calib`` and decision-table sections are excluded — those fields are
# either unused by the GUI Hydra tree (calib entry point) or carry
# non-identifier first columns (``Stage``, ``Column``, …).
_FIELD_SECTIONS: frozenset[str] = frozenset({"input", "input.coefs", "out", "filter", "program"})

# ``## ``input.coefs`` — subtitle``.
_RE_SECTION_HEAD = re.compile(r"^##\s+`(?P<section>[A-Za-z_]\w*(?:\.\w+)*)`\s*(?:—\s*(?P<subtitle>.+))?\s*$")
# ``### `input.path` <mode>probe</mode>``; ``</>`` shorthand is accepted.
_RE_FIELD_MODE_HEAD = re.compile(
    r"^###\s+`(?P<path>[A-Za-z_]\w*(?:\.\w+)*)`\s+<mode>(?P<mode>[a-z_]+)</(?:mode)?>"
)

# ``#### Detailed`` or any other named detail block.
_RE_DETAIL_HEAD = re.compile(r"^####\s+(?P<tag>.+?)\s*$")
# Any markdown heading.  Checked only after mode/detail headings so that
# ``###`` mode headers and ``####`` detail headers do not close their parent.
_RE_ANY_HEADING = re.compile(r"^#{1,6}\s")

# Code-fence toggle.  ``#`` inside a fence is not a heading.
_RE_FENCE = re.compile(r"^\s*(```|~~~)")

# Table row whose first cell is a bare backtick-quoted identifier:
# ``| `field_name` | … | description |``.
_RE_FIELD_ROW = re.compile(r"^\|\s*`(?P<field>[A-Za-z_]\w*)`\s*\|")

# Array indices at lookup time: ``Ag[0]`` / ``Ag[1][2]`` → ``Ag``.
_RE_ARR_INDEX = re.compile(r"\[\d+\]")


@dataclass(frozen=True, slots=True)
class ModeBody:
    """Parsed body for one mode-tagged ``###`` section.

    Attributes:
        short: Content before the first ``####`` sub-block.  This is the
            short mode body used for hover/status text.
        details: Named ``####`` sub-blocks: ``tag → content``.  The canonical
            tag is ``"Detailed"``.
    """

    short: str
    details: Mapping[str, str] = field(default_factory=dict)


# One mode value in a full HelpEntry.body mapping.
#
# - ``str`` when the mode section has no ``####`` detail blocks.
# - ``ModeBody`` when the mode section has named ``####`` detail blocks.
ModeValue = str | ModeBody

# Full body mapping for an entry before mode/detail reduction.
HelpBody = Mapping[str, ModeValue]


@dataclass(frozen=True, slots=True)
class HelpEntry:
    """Help entry for one config path.

    Attributes:
        path: Dotted Hydra path, without array indices.
        short: Short tooltip extracted from the last table cell.
        body: Full mode mapping when no mode is selected; a reduced string
            when :func:`help_for_path` is called with ``mode`` or ``detail``.
    """

    path: str
    short: str
    body: HelpBody | str = field(default_factory=dict)


@dataclass(slots=True)
class _State:
    """Mutable one-pass parser state."""

    section: str | None = None
    in_field_section: bool = False
    fence: bool = False

    mode_path: str | None = None
    mode_tag: str | None = None
    short_lines: list[str] = field(default_factory=list)

    detail_tag: str | None = None
    detail_lines: list[str] = field(default_factory=list)
    details: dict[str, str] = field(default_factory=dict)

    @property
    def in_mode(self) -> bool:
        """True while inside a ``### `field` <mode>…</mode>`` section."""
        return self.mode_path is not None and self.mode_tag is not None

    def clear_mode(self) -> None:
        """Reset mode/detail accumulation buffers."""
        self.mode_path = self.mode_tag = self.detail_tag = None
        self.short_lines.clear()
        self.detail_lines.clear()
        self.details.clear()


def parse_reference(text: str) -> dict[str, HelpEntry]:
    """Parse ``config_reference.md`` content into path → :class:`HelpEntry`.

    Args:
        text: Full markdown source text.

    Returns:
        Mutable mapping ``{dotted.path: HelpEntry}``.

    Behavior:
        * ``## ``section`` headings define field sections.
        * Table rows inside those sections create short help entries.
        * ``### ``field.path`` <mode>…</mode>`` headings create mode bodies.
        * ``#### <tag>`` headings inside a mode create named detail blocks.
        * Code fences are ignored for heading detection, but fence markers
          and fenced content are preserved inside open mode bodies.
    """
    entries: dict[str, HelpEntry] = {}
    bodies: defaultdict[str, dict[str, ModeValue]] = defaultdict(dict)
    st = _State()

    def flush_detail() -> None:
        """Freeze the active ``####`` block into ``st.details``."""
        if st.detail_tag is not None:
            st.details[st.detail_tag] = "\n".join(st.detail_lines).strip()
        st.detail_tag = None
        st.detail_lines.clear()

    def close_mode() -> None:
        """Freeze the active mode section into ``bodies``."""
        flush_detail()

        if (path := st.mode_path) and (tag := st.mode_tag) and path in entries:
            short = "\n".join(st.short_lines).strip()
            bodies[path][tag] = short if not st.details else ModeBody(short=short, details=dict(st.details))

        st.clear_mode()

    for line in text.splitlines():
        if _RE_FENCE.match(line):
            # Preserve fence markers inside mode bodies so downstream markdown
            # rendering can still recognize fenced code blocks.
            if st.in_mode:
                target = st.detail_lines if st.detail_tag is not None else st.short_lines
                target.append(line)

            st.fence = not st.fence
            continue

        if not st.fence and (m := _RE_SECTION_HEAD.match(line)):
            close_mode()

            section = m["section"]
            st.section = section
            st.in_field_section = section in _FIELD_SECTIONS

            if st.in_field_section:
                subtitle = (m["subtitle"] or "").strip()
                entries[section] = HelpEntry(
                    path=section,
                    short=subtitle or section,
                )

            continue

        # Must be checked before generic heading handling.
        if not st.fence and (m := _RE_FIELD_MODE_HEAD.match(line)):
            close_mode()
            st.mode_path, st.mode_tag = m["path"], m["mode"]
            continue

        # Meaningful only inside an open mode; otherwise it is a heading.
        if not st.fence and st.in_mode and (m := _RE_DETAIL_HEAD.match(line)):
            flush_detail()
            st.detail_tag = m["tag"].strip()
            st.detail_lines.clear()
            continue

        if not st.fence and st.section is not None and _RE_ANY_HEADING.match(line):
            close_mode()
            st.section, st.in_field_section = None, False
            continue

        if st.in_mode:
            target = st.detail_lines if st.detail_tag is not None else st.short_lines
            target.append(line)
            continue

        if st.fence:
            continue

        if st.in_field_section and (section := st.section) and (m := _RE_FIELD_ROW.match(line)):
            path = f"{section}.{m['field']}"
            cells = split_table_row(line)
            entries[path] = HelpEntry(
                path=path,
                short=cells[-1].strip() if cells else "",
            )

    close_mode()

    for path, modes in bodies.items():
        if path in entries:
            entries[path] = replace(entries[path], body=modes)

    return entries


# ── loader & resolver ─────────────────────────────────────────────────────────


def _doc_path(lang: str | None = None) -> Path:
    """Localized ``config_reference`` path; English fallback."""
    if lang and (p := _constants.DOC_DIR / f"config_reference_{lang}.md").is_file():
        return p
    return _constants.DOC_DIR / "config_reference.md"


_CACHE: dict[str, dict[str, HelpEntry]] = {}
"""Per-language parsed entries.

Key = resolved two-letter language code.  Populated lazily by :func:`_load`
and cleared by :func:`reload_cache`.
"""


def _load(lang: str | None = None) -> Mapping[str, HelpEntry]:
    """Load and cache help entries for *lang*.

    Args:
        lang: Two-letter language code.  ``None`` resolves the current
            application language via :func:`resolve_lang`.

    Returns:
        Mapping ``{dotted.path: HelpEntry}``.  On missing file or parse
        failure, returns an empty mapping so GUI hover degrades gracefully.
    """
    if lang is None:
        lang = resolve_lang()

    if lang in _CACHE:
        return _CACHE[lang]

    path = _doc_path(lang)

    try:
        entries = parse_reference(path.read_text(encoding="utf-8"))
    except OSError:
        _l.error("config_reference not found at %s — hover disabled", path)
        entries = {}
    except Exception:  # noqa: BLE001 — parse failure is non-fatal for tooltips
        _l.error("config_reference parse error at %s — hover disabled", path, exc_info=True)
        entries = {}
    _l.debug("Loaded %d config help entries from %s (lang=%s)", len(entries), path, lang)
    _CACHE[lang] = entries
    return entries


def reload_cache(lang: str | None = None) -> Mapping[str, HelpEntry]:
    """Clear cached parsed entries and reload. Helpd testing

    Args:
        lang: Language code to evict/reload.  ``None`` clears all cached
            languages and reloads the currently resolved language.

    Returns:
        Fresh mapping ``{dotted.path: HelpEntry}``.
    """
    if lang is None:
        _CACHE.clear()
    else:
        _CACHE.pop(lang, None)

    return _load(lang)


def help_for_path(
    path: str,
    *,
    mode: str | None = None,
    detail: str | None = None,
) -> HelpEntry | None:
    """Resolve a dotted Hydra path to a :class:`HelpEntry`.

    Array indices are stripped before lookup:

        ``Ag[0]``      → ``Ag``
        ``Ag[1][2]``   → ``Ag``

    Args:
        path: Dotted Hydra path, e.g. ``input.coefs.Ag[0]``.
        mode: Optional mode selector, e.g. ``"probe"`` or ``"search"``.
        detail: Optional ``####`` detail tag inside *mode*, e.g.
            ``"Detailed"``.

    Returns:
        * ``None`` if the path is undocumented.
        * If ``mode is None``: the full entry.  ``entry.body`` is a mapping
          ``mode → str | ModeBody``.
        * If ``mode`` is given and ``detail is None``: a reduced entry whose
          ``body`` is the mode's short text.
        * If both ``mode`` and ``detail`` are given: a reduced entry whose
          ``body`` is the named detail block, or ``""`` if absent.
    """
    entry = _load().get(_RE_ARR_INDEX.sub("", path))

    if entry is None:
        return None

    if mode is None:
        return entry

    raw = entry.body.get(mode) if isinstance(entry.body, Mapping) else None

    if raw is None:
        return replace(entry, body="")

    if detail is None:
        body = raw.short if isinstance(raw, ModeBody) else raw
    else:
        body = raw.details.get(detail, "") if isinstance(raw, ModeBody) else ""

    return replace(entry, body=body)
