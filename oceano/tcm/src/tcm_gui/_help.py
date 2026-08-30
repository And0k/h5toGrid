"""Resolve Hydra config paths to help text auto-extracted from ``config_reference.md``.

Single source of truth for non-chrome (config-cell) tooltips: the existing
field tables in ``docs/reference/config_reference.md``.

Table-driven sections
---------------------
For each field section heading of the form:
    ## `input`
    ## `input.coefs`

Backticks are optional — a plain heading registers iff its name is in
:data:`_FIELD_SECTIONS` (general prose sections stay non-registered).

The parser scans markdown table rows whose first cell is a backticked field
identifier:

    | `field_name` = default | ... | Purpose / Physical meaning |

and emits one :class:`HelpEntry` keyed by ``{section}.{field}``.  The last
cell (the description column) becomes :attr:`HelpEntry.short`.

Mode-tagged detail sections
---------------------------
Detailed documentation lives in ``###`` subsections of a field.  The
``<mode>`` tag is optional — use it only when the field's meaning depends on
the consumer context (one section per context); a modeless ``### `field` ``
section is the single-context default and is stored under :data:`_NO_MODE`:

    ### `input.path`
    ### `path_field` <mode>dirs</mode>
    ### `input.coefs.P_t`

These are stored in :attr:`HelpEntry.body` keyed by mode.

Detail sub-blocks
-----------------
Inside a mode section, ``####`` headings define named detail blocks:

    #### Detailed

Lines before the first ``####`` are stored as the mode's short body.
Named ``####`` blocks are stored in :attr:`ModeBody.details`.

Field-level detail blocks
-------------------------
``####`` headings may also appear directly under a ``## ``section``` heading
(without an intervening ``###`` mode tag).  These are stored under the
sentinel mode key ``_`` in :attr:`HelpEntry.body`, making them reachable
via :func:`help_for_path` without specifying a mode.  This is the primary
mechanism for config-cell dwell tooltips on fields that do not have
mode-dependent meanings.


Arrays are resolved at the field level: ``Ag[0]`` / ``Ag[1][2]`` are stripped
to ``Ag`` before lookup.

Localization uses ``config_reference_{lang}.md`` when present, with fallback
to ``config_reference.md``.
"""

from __future__ import annotations

import dataclasses
import logging
import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import get_type_hints

from tcm import _constants, schema
from tcm._md_parse import split_table_row

from ._cell_spec import _unwrap
from ._i18n import resolve_lang

_l = logging.getLogger(__name__)

# Sections whose ``## ``{section}```` heading opens table-driven field scanning,
# inferred from the structured schema: top-level ``Config`` groups plus nested
# dataclass groups of ``input`` (``input.coefs``, ``input.calib``).  ``proc`` is
# not a ``Config`` field and decision-table sections carry non-identifier first
# columns (``Stage``, ``Column``, …) — both stay excluded by construction.
# ``_unwrap`` resolves the ``X | None`` union around nested group hints.
# ``metadata`` and ``path_field`` are non-schema sections: per-probe deployment
# journal and the GUI search path (its ``<mode>`` bodies feed PathField statuses).
_FIELD_SECTIONS: frozenset[str] = frozenset(
    {n for n, t in get_type_hints(schema.Config).items() if dataclasses.is_dataclass(_unwrap(t))}
    | {
        f"input.{n}"
        for n, t in get_type_hints(schema.ConfigIn_InclProc).items()
        if dataclasses.is_dataclass(_unwrap(t))
    }
    | {"metadata", "path_field"}
)

# ``## ``input.coefs`` — subtitle``.  Backticks optional around the section
# name and the subtitle — a plain heading registers iff its name is in
# :data:`_FIELD_SECTIONS` (non-schema sections carry plain titles).
_RE_SECTION_HEAD = re.compile(
    r"^##\s+`?(?P<section>[A-Za-z_]\w*(?:\.\w+)*)`?\s*(?:[—-]\s*(?P<subtitle>.+))?\s*$"
)
# ``### `input.path` <mode>probe</mode>`` — the <mode> tag is optional (a
# modeless ``### `field` `` section is the single-context default); ``</>``
# shorthand is accepted.
_RE_FIELD_MODE_HEAD = re.compile(
    r"^###\s+`(?P<path>[A-Za-z_]\w*(?:\.\w+)*)`(?:\s+<mode>(?P<mode>[a-z_]+)</(?:mode)?>)?"
)

# ``#### <mode>dirs</mode>`` — mode tag under a ``### `field` `` heading.  Inherits
# the parent field path, equivalent to a separate ``### `field` <mode>dirs</mode>``
# heading but nests the mode detail under the general field description.
# Its child detail blocks use ``#####`` (one level deeper) to avoid ambiguity
# with ``####`` siblings of the parent ``###`` section.
_RE_MODE_IN_DETAIL = re.compile(r"^####\s+<mode>(?P<mode>[a-z_]+)</(?:mode)?>")

# ``#### Detailed`` or any other named detail block (child of a ``###`` mode).
_RE_DETAIL_HEAD = re.compile(r"^####\s+(?P<tag>.+?)\s*$")
# ``##### Detailed`` — child of a ``#### <mode>`` section.
_RE_DETAIL_HEAD5 = re.compile(r"^#####\s+(?P<tag>.+?)\s*$")
# Any markdown heading.  Checked only after mode/detail headings so that
# ``###`` mode headers and ``####`` detail headers do not close their parent.
_RE_ANY_HEADING = re.compile(r"^#{1,6}\s")

# Code-fence toggle.  ``#`` inside a fence is not a heading.
_RE_FENCE = re.compile(r"^\s*(```|~~~)")

# Table row whose first cell is a backtick-quoted identifier optionally followed
# by `` = default`` — the joined ``Field = Default`` column:
# ``| `field_name` = default | … | description |``.
_RE_FIELD_ROW = re.compile(r"^\|\s*`(?P<field>[A-Za-z_]\w*)`[^|]*\|")

# Array indices at lookup time: ``Ag[0]`` / ``Ag[1][2]`` → ``Ag``.
_RE_ARR_INDEX = re.compile(r"\[\d+\]")

# Sentinel mode key for field-level ``#### Detail`` blocks (no ``###`` mode tag).
_FIELD_DETAIL = "_"

# Implicit body key for modeless ``### `field` `` sections — single-context
# detail without a probe/search split.  Any name from ``[a-z_]+`` works; this
# one is reserved so a user tag cannot collide with it.
_NO_MODE = "detail"

# ``{#explicit-id}`` suffix on a heading (doc browser honors it as the anchor).
_RE_ANCHOR_ID = re.compile(r"\{#([^{}]+)\}\s*$")
# GitHub-style slug drops: everything but word chars (unicode letters/digits/_),
# whitespace and hyphens — mirrors ``browser/web/viewer.js::slugify``.
_RE_SLUG_DROP = re.compile(r"[^\w\s-]", re.UNICODE)


def slugify(text: str) -> str:
    """Heading anchor — explicit ``{#id}`` wins, else the GitHub-style slug.

    Mirrors ``browser/web/viewer.js::slugify``: lowercase, drop punctuation,
    each space → one dash (so ``a — b`` → ``a--b``).  Public alias for the
    parser's internal ``_slug`` — use it to derive table-row anchors that
    match the viewer's heading ids (dots are dropped: ``input.path`` → ``inputpath``).
    """
    return _slug(text)


def _slug(text: str) -> str:
    """Heading anchor — explicit ``{#id}`` wins, else the GitHub-style slug.

    Mirrors ``browser/web/viewer.js::slugify``: lowercase, drop punctuation,
    each space → one dash (so ``a — b`` → ``a--b``).
    """
    if m := _RE_ANCHOR_ID.search(text):
        return m.group(1)
    return _RE_SLUG_DROP.sub("", text.lower()).strip().replace(" ", "-")


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
        anchor: Heading anchor in the source doc (browser F1-jump target);
            field rows inherit their section heading's anchor.
    """

    path: str
    short: str
    body: HelpBody | str = field(default_factory=dict)
    anchor: str = ""


@dataclass(slots=True)
class _State:
    """Mutable one-pass parser state."""

    section: str | None = None
    in_field_section: bool = False
    fence: bool = False

    # Paragraph text between a ``## `` heading and its table (the section's
    # short description). Captured until a table row or the next heading.
    capture_post_heading: bool = False
    post_heading_para: list[str] = field(default_factory=list)

    mode_path: str | None = None
    mode_tag: str | None = None
    mode_level: int | None = None  # 3 for ``### `field` <mode>``, 4 for ``#### <mode>``
    short_lines: list[str] = field(default_factory=list)

    detail_tag: str | None = None
    detail_lines: list[str] = field(default_factory=list)
    details: dict[str, str] = field(default_factory=dict)

    # Field-level detail blocks (#### outside any ### mode).
    # Keyed by the full dotted field path (e.g. "input.time_ranges").
    field_details: dict[str, dict[str, str]] = field(default_factory=dict)
    last_field_path: str | None = None
    section_anchor: str = ""

    @property
    def in_mode(self) -> bool:
        """True while inside a ``### `field` <mode>…</mode>`` section."""
        return self.mode_path is not None and self.mode_tag is not None

    def clear_mode(self) -> None:
        """Reset mode/detail accumulation buffers."""
        self.mode_path = self.mode_tag = self.mode_level = self.detail_tag = None
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
            if tag == "Detailed":
                # Bare ``### Detailed`` has no nested ``#### `` block — store its
                # content as the mode body's short so _resolve_detail finds it.
                bodies[path][tag] = ModeBody(short=short, details=dict(st.details))
            else:
                bodies[path][tag] = short if not st.details else ModeBody(short=short, details=dict(st.details))
            # Entries created from a ``### `` subsection without a table row (e.g.
            # ``metadata.path``) have no short of their own — populate it from the
            # subsection's lead-in text so the status bar shows it.
            if not entries[path].short and short:
                entries[path] = replace(entries[path], short=short)

        st.clear_mode()

    def flush_any_detail() -> None:
        """Freeze the active ``####`` buffer — mode detail inside a mode, field-level otherwise."""
        (flush_detail if st.in_mode else flush_section_detail)()

    def _finalize_post_heading() -> None:
        """Set the current section's short from its post-heading paragraph (if any)."""
        if not st.capture_post_heading or st.section not in entries:
            return
        if st.post_heading_para:
            para = "\n".join(st.post_heading_para).strip()
            if para:
                entries[st.section] = replace(entries[st.section], short=para)
        st.capture_post_heading = False
        st.post_heading_para.clear()

    def flush_section_detail() -> None:
        """Freeze the active field-level ``####`` block into ``st.field_details``."""
        if st.detail_tag is not None and st.last_field_path:
            st.field_details.setdefault(st.last_field_path, {})[st.detail_tag] = "\n".join(
                st.detail_lines
            ).strip()
        st.detail_tag = None
        st.detail_lines.clear()

    for line in text.splitlines():
        if _RE_FENCE.match(line):
            # Preserve fence markers inside mode bodies so downstream markdown
            # rendering can still recognize fenced code blocks.
            if st.in_mode:
                target = st.detail_lines if st.detail_tag is not None else st.short_lines
                target.append(line)

            st.fence = not st.fence
            continue

        # Citation blockquote — cut the section before it: finalize whatever is
        # being accumulated (post-heading paragraph or mode body) and ignore the
        # citation line. Subsequent headings still start new sections.
        if not st.fence and line.lstrip().startswith(">"):
            _finalize_post_heading()
            close_mode()
            continue

        if not st.fence and (m := _RE_SECTION_HEAD.match(line)):
            flush_any_detail()
            close_mode()

            section = m["section"]
            st.section = section
            st.in_field_section = section in _FIELD_SECTIONS
            st.last_field_path = None

            if st.in_field_section:
                subtitle = _RE_ANCHOR_ID.sub("", (m["subtitle"] or "")).strip()
                st.section_anchor = _slug(line)
                entries[section] = HelpEntry(
                    path=section,
                    short=subtitle or section,
                    anchor=st.section_anchor,
                )
                # Capture the paragraph between this heading and its table to use
                # as the section's short description (falls back to subtitle if empty).
                st.capture_post_heading = True
                st.post_heading_para.clear()
            else:
                st.section_anchor = ""

            continue

        # Must be checked before generic heading handling.
        if not st.fence and (m := _RE_FIELD_MODE_HEAD.match(line)):
            flush_any_detail()
            close_mode()
            path = m["path"]
            # Bare ``### `path_field` `` opens its own section when the doc
            # carries no ``## `path_field` `` heading — the section registers
            # iff its name is in :data:`_FIELD_SECTIONS`.
            if st.section != path and path in _FIELD_SECTIONS:
                st.section, st.in_field_section = path, True
                st.last_field_path = None
                # Anchor = heading minus the ``<mode>`` tail (the viewer's
                # slugify would otherwise bake the tag into the anchor).
                st.section_anchor = _slug(re.sub(r"\s*<mode>.*$", "", line))
                if path not in entries:
                    entries[path] = HelpEntry(path=path, short="", anchor=st.section_anchor)
            elif st.section is not None and st.section != path and path.startswith(st.section + "."):
                # Child subsection without its own table row (e.g. ``metadata.path``
                # after it was moved from the table into a ``### `` block). Register
                # an entry so its short body + ``#### Detailed`` are preserved.
                if path not in entries:
                    anchor = _slug(re.sub(r"\s*<mode>.*$", "", line))
                    entries[path] = HelpEntry(path=path, short="", anchor=anchor)
            st.mode_path, st.mode_tag, st.mode_level = path, m["mode"] or _NO_MODE, 3
            continue

        # Bare `### Detailed` heading: tooltip for the current parent section.
        # Does not close the section — subsequent ``### `` path blocks still work.
        if not st.fence and st.in_field_section and line.strip() == "### Detailed":
            flush_any_detail()
            close_mode()
            st.mode_path, st.mode_tag, st.mode_level = st.section, "Detailed", 3
            continue

        # ``#### <mode>dirs</mode>`` under ``### `field` `` — inherits parent field
        # path.  Equivalent to a separate ``### `field` <mode>dirs</mode>`` heading
        # but nests the mode detail under the general field description.
        # Its child details use ``#####`` (one level deeper).
        if not st.fence and st.mode_path is not None and st.mode_path in entries and (m := _RE_MODE_IN_DETAIL.match(line)):
            flush_any_detail()
            # Save current mode content (general description under _NO_MODE).
            if st.mode_tag is not None:
                short = "\n".join(st.short_lines).strip()
                bodies[st.mode_path][st.mode_tag] = (
                    short if not st.details else ModeBody(short=short, details=dict(st.details))
                )
            # Reset detail buffers but keep mode_path for the new mode.
            st.detail_tag = None
            st.detail_lines.clear()
            st.details.clear()
            st.short_lines.clear()
            st.mode_tag, st.mode_level = m["mode"], 4
            continue

        # Meaningful only inside an open mode; otherwise it is a heading.
        # ``####`` details belong to ``###`` modes (level 3), ``#####`` to ``#### <mode>`` modes (level 4).
        if not st.fence and st.in_mode and st.mode_level == 3 and (m := _RE_DETAIL_HEAD.match(line)):
            flush_detail()
            st.detail_tag = m["tag"].strip()
            st.detail_lines.clear()
            continue
        if not st.fence and st.in_mode and st.mode_level == 4 and (m := _RE_DETAIL_HEAD5.match(line)):
            flush_detail()
            st.detail_tag = m["tag"].strip()
            st.detail_lines.clear()
            continue

        # Field-level #### detail block (no ### mode tag active).
        if not st.fence and not st.in_mode and st.in_field_section and (m := _RE_DETAIL_HEAD.match(line)):
            flush_section_detail()
            st.detail_tag = m["tag"].strip()
            st.detail_lines.clear()
            continue

        if not st.fence and st.section is not None and _RE_ANY_HEADING.match(line):
            flush_any_detail()
            close_mode()
            _finalize_post_heading()
            st.section, st.in_field_section = None, False
            st.last_field_path = None
            st.section_anchor = ""
            continue

        # Capture the paragraph between a ## heading and its table. Finalize
        # when the table (or any heading) starts.
        if st.capture_post_heading:
            if _RE_FIELD_ROW.match(line) or line.lstrip().startswith("|"):
                _finalize_post_heading()
            else:
                st.post_heading_para.append(line.strip())
                continue

        if st.in_mode:
            target = st.detail_lines if st.detail_tag is not None else st.short_lines
            target.append(line)
            continue

        # Field-level detail content (#### outside ### mode, inside ## section).
        if not st.in_mode and st.in_field_section and st.detail_tag is not None:
            st.detail_lines.append(line)
            continue

        if st.fence:
            continue

        if st.in_field_section and (section := st.section) and (m := _RE_FIELD_ROW.match(line)):
            field_name = m["field"]
            path = f"{section}.{field_name}"
            st.last_field_path = path
            cells = split_table_row(line)
            entries[path] = HelpEntry(
                path=path,
                short=cells[-1].strip() if cells else "",
                anchor=st.section_anchor,
            )

    flush_any_detail()
    close_mode()
    # Store field-level details under each field's own path.
    for field_path, details in st.field_details.items():
        if field_path in entries:
            bodies[field_path][_FIELD_DETAIL] = ModeBody(short="", details=dict(details))

    for path, modes in bodies.items():
        if path in entries:
            entries[path] = replace(entries[path], body=modes)

    return entries


# ── loader & resolver ─────────────────────────────────────────────────────────


def doc_path(lang: str | None = None) -> Path:
    """Localized ``config_reference`` path; English fallback.

    The source document for :func:`help_for_path` entries — also the base
    directory for relative markdown links inside their bodies (tooltips).
    """
    if lang and (p := _constants.DOC_DIR / "reference" / f"config_reference_{lang}.md").is_file():
        return p
    return _constants.DOC_DIR / "reference" / "config_reference.md"


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

    path = doc_path(lang)

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
        mode: Optional mode selector, e.g. ``"files"`` or ``"dirs"``.
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

    # Modeless ``### `field` `` section is the fallback for any requested mode
    # without a tagged section (keeps statuses armed when only one context is
    # documented).
    if raw is None and isinstance(entry.body, Mapping):
        raw = entry.body.get(_NO_MODE)

    if raw is None:
        return replace(entry, body="")

    if detail is None:
        body = raw.short if isinstance(raw, ModeBody) else raw
    else:
        body = raw.details.get(detail, "") if isinstance(raw, ModeBody) else ""

    return replace(entry, body=body)


def help_general_for_path(path: str) -> str:
    """Return the general (modeless) description for a config path.

    This is the ``### `field` `` body — the text before any
    ``#### <mode>`` or ``### `field` <mode>mode</mode>`` section.  For
    ``path_field`` the ``#### Important`` sub-block (if present) is the
    content shown on field-associated errors (e.g. ``FileNotFoundError`` on a
    failed data/config search); otherwise the short pre-``####`` body is used.
    Mode-specific bodies are irrelevant for this call.

    Array indices are stripped before lookup, mirroring :func:`help_for_path`.
    """
    entry = _load().get(_RE_ARR_INDEX.sub("", path))
    if entry is None:
        return ""
    if isinstance(entry.body, Mapping):
        raw = entry.body.get(_NO_MODE, "")
        if isinstance(raw, ModeBody):
            # ``path_field`` uses ``#### Important`` as the error hint.
            if isinstance(raw.details, Mapping) and "Important" in raw.details:
                return raw.details["Important"]
            return raw.short
        return raw if isinstance(raw, str) else ""
    return ""
