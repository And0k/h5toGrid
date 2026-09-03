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
# Level-agnostic: ``##``/``###``/``####`` all accepted, L derived from hashes.
_RE_FIELD_MODE_HEAD = re.compile(
    r"^(?P<hashes>#{2,6})\s+`(?P<path>[A-Za-z_]\w*(?:\.\w+)*)`(?:\s+<mode>(?P<mode>[a-z_]+)</(?:mode)?>)?"
)

# ``#### <mode>dirs</mode>`` — mode tag under a ``### `field` `` heading.  Inherits
# the parent field path, equivalent to a separate ``### `field` <mode>dirs</mode>``
# heading but nests the mode detail under the general field description.
# Its child detail blocks use ``#####`` (one level deeper) to avoid ambiguity
# with ``####`` siblings of the parent ``###`` section.
# Level-agnostic: capture hashes to derive L.
_RE_MODE_IN_DETAIL = re.compile(r"^(?P<hashes>#{2,6})\s+<mode>(?P<mode>[a-z_]+)</(?:mode)?>")

# ``#### Detailed`` or any other named detail block (child of a ``###`` mode).
_RE_DETAIL_HEAD = re.compile(r"^####\s+(?P<tag>.+?)\s*$")
# ``##### Detailed`` — child of a ``#### <mode>`` section.
_RE_DETAIL_HEAD5 = re.compile(r"^#####\s+(?P<tag>.+?)\s*$")
# Any markdown heading.  Checked only after mode/detail headings so that
# ``###`` mode headers and ``####`` detail headers do not close their parent.
_RE_ANY_HEADING = re.compile(r"^#{1,6}\s")
# Generic heading level — level-agnostic extraction: derive L from parent heading
# and match detail levels as L+1 / L+2 rather than hard-coding 3/4/5.
_RE_HEADING = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<body>.*)$")

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
# Table-row "↓" anchor link with no useful display text — stripped from GUI status.
_RE_DOWN_LINK = re.compile(r"\[↓\]\([^)]+\)")


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
        sub_details: Nested ``#####`` sub-blocks under a ``####`` detail:
            ``detail_tag → {sub_tag → content}``.  Used to capture
            ``Important`` under ``Detailed`` without breaking the parent
            detail at a new heading.
    """

    short: str
    details: Mapping[str, str] = field(default_factory=dict)
    sub_details: Mapping[str, Mapping[str, str]] = field(default_factory=dict)


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
    section_level: int | None = None
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

    # Nested ``#####`` blocks under a ``####`` detail (e.g. Important under Detailed).
    sub_detail_tag: str | None = None
    sub_detail_lines: list[str] = field(default_factory=list)
    sub_details: dict[str, dict[str, str]] = field(default_factory=dict)

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
        self.sub_detail_tag = None
        self.sub_detail_lines.clear()
        self.sub_details.clear()


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

    def flush_sub_detail() -> None:
        """Freeze the active ``#####`` sub-block into ``st.sub_details``."""
        if st.sub_detail_tag is not None and st.detail_tag is not None:
            (inner := st.sub_details.setdefault(st.detail_tag, {}))
            txt = _clean_text("\n".join(st.sub_detail_lines)).strip()
            # Concatenate duplicate tags with newline (supports multiple Important blocks).
            if st.sub_detail_tag in inner and inner[st.sub_detail_tag] and txt:
                inner[st.sub_detail_tag] = f"{inner[st.sub_detail_tag]}\n{txt}"
            elif txt or st.sub_detail_tag not in inner:
                inner[st.sub_detail_tag] = txt
        st.sub_detail_tag = None
        st.sub_detail_lines.clear()

    def flush_detail() -> None:
        """Freeze the active ``####`` block into ``st.details`` (flushing sub-detail first)."""
        flush_sub_detail()
        if st.detail_tag is not None:
            txt = _clean_text("\n".join(st.detail_lines)).strip()
            if st.detail_tag in st.details and st.details[st.detail_tag] and txt:
                st.details[st.detail_tag] = f"{st.details[st.detail_tag]}\n{txt}"
            elif txt or st.detail_tag not in st.details:
                st.details[st.detail_tag] = txt
        st.detail_tag = None
        st.detail_lines.clear()

    def close_mode() -> None:
        """Freeze the active mode section into ``bodies``."""
        flush_detail()

        if (path := st.mode_path) and (tag := st.mode_tag) and path in entries:
            short = _clean_text("\n".join(st.short_lines)).strip()
            # Materialize sub_details as plain dicts for ModeBody.
            subs = {k: dict(v) for k, v in st.sub_details.items()}
            has_subs = any(bool(v) for v in subs.values())
            has_details = bool(st.details) or has_subs
            if tag == "Detailed":
                # Bare ``### Detailed`` has no nested ``#### `` block — store its
                # content as the mode body's short so _resolve_detail finds it.
                bodies[path][tag] = ModeBody(short=short, details=dict(st.details), sub_details=subs)
            else:
                bodies[path][tag] = (
                    ModeBody(short=short, details=dict(st.details), sub_details=subs)
                    if has_details
                    else short
                )
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
            para = _clean_text("\n".join(st.post_heading_para)).strip()
            if para:
                entries[st.section] = replace(entries[st.section], short=para)
        st.capture_post_heading = False
        st.post_heading_para.clear()

    def flush_section_detail() -> None:
        """Freeze the active field-level ``####`` block into ``st.field_details``."""
        if st.detail_tag is not None and st.last_field_path:
            st.field_details.setdefault(st.last_field_path, {})[st.detail_tag] = _clean_text(
                "\n".join(st.detail_lines)
            ).strip()
        st.detail_tag = None
        st.detail_lines.clear()

    for line in text.splitlines():
        if _RE_FENCE.match(line):
            # Preserve fence markers inside mode bodies so downstream markdown
            # rendering can still recognize fenced code blocks.
            if st.in_mode:
                if st.sub_detail_tag is not None:
                    target = st.sub_detail_lines
                elif st.detail_tag is not None:
                    target = st.detail_lines
                else:
                    target = st.short_lines
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
            st.section_level = 2
            st.in_field_section = section in _FIELD_SECTIONS
            st.last_field_path = None

            if st.in_field_section:
                subtitle = _clean_text(_RE_ANCHOR_ID.sub("", (m["subtitle"] or "")).strip())
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
                # Heading level of ``### `path_field` `` is 3; derive generically if needed.
                if (hm2 := _RE_HEADING.match(line)):
                    st.section_level = len(hm2.group("hashes"))
                else:
                    st.section_level = 3
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
            # Level-agnostic: derive L from heading hashes, not hard-coded 3.
            lvl = len(_RE_HEADING.match(line).group("hashes")) if _RE_HEADING.match(line) else 3
            st.mode_path, st.mode_tag, st.mode_level = path, m["mode"] or _NO_MODE, lvl
            continue

        # Bare `### Detailed` heading: tooltip for the current parent section.
        # Does not close the section — subsequent ``### `` path blocks still work.
        # Level-agnostic: any heading whose body is exactly ``Detailed`` at L = section_level+1,
        # only when not already inside a mode (otherwise it's a detail of that mode).
        if not st.fence and st.in_field_section and not st.in_mode and (hm_d := _RE_HEADING.match(line)):
            body_d = _RE_ANCHOR_ID.sub("", hm_d.group("body")).strip()
            lvl_d = len(hm_d.group("hashes"))
            if body_d == "Detailed" and st.section_level is not None and lvl_d == st.section_level + 1:
                flush_any_detail()
                close_mode()
                st.mode_path, st.mode_tag, st.mode_level = st.section, "Detailed", lvl_d
                continue

        # ``#### <mode>dirs</mode>`` under ``### `field` `` — inherits parent field
        # path.  Equivalent to a separate ``### `field` <mode>dirs</mode>`` heading
        # but nests the mode detail under the general field description.
        # Its child details use ``#####`` (one level deeper).
        if not st.fence and st.mode_path is not None and st.mode_path in entries and (m := _RE_MODE_IN_DETAIL.match(line)):
            flush_any_detail()
            # Save current mode content (general description under _NO_MODE).
            if st.mode_tag is not None:
                short = _clean_text("\n".join(st.short_lines)).strip()
                subs = {k: dict(v) for k, v in st.sub_details.items()}
                has_subs = any(bool(v) for v in subs.values())
                has_details = bool(st.details) or has_subs
                bodies[st.mode_path][st.mode_tag] = (
                    ModeBody(short=short, details=dict(st.details), sub_details=subs)
                    if has_details
                    else short
                )
            # Reset detail buffers but keep mode_path for the new mode.
            st.detail_tag = None
            st.detail_lines.clear()
            st.details.clear()
            st.sub_detail_tag = None
            st.sub_detail_lines.clear()
            st.sub_details.clear()
            st.short_lines.clear()
            lvl_nested = len(m.group("hashes")) if "hashes" in m.groupdict() else 4
            st.mode_tag, st.mode_level = m["mode"], lvl_nested
            continue

        # ── level-agnostic detail / sub-detail (derive L from parent) ──────
        # L = mode_level, L+1 = detail (####), L+2 = sub-detail (##### Important under Detailed).
        # This replaces hard-coded 3/4/5 checks — any root heading level works (##, ###, ####).
        if not st.fence and st.in_mode and st.mode_level is not None and (hm := _RE_HEADING.match(line)):
            lvl = len(hm.group("hashes"))
            body = hm.group("body").strip()
            is_mode_heading = body.lstrip().startswith("<mode>") or body.lstrip().startswith("`")
            if not is_mode_heading:
                if lvl == st.mode_level + 1:
                    flush_sub_detail()
                    flush_detail()
                    tag = _RE_ANCHOR_ID.sub("", body).strip()
                    if tag:
                        st.detail_tag = tag
                        st.detail_lines.clear()
                        st.sub_detail_tag = None
                        st.sub_detail_lines.clear()
                        continue
                elif lvl == st.mode_level + 2 and st.detail_tag is not None:
                    flush_sub_detail()
                    tag = _RE_ANCHOR_ID.sub("", body).strip()
                    if tag:
                        st.sub_detail_tag = tag
                        st.sub_detail_lines.clear()
                        continue

        # Field-level #### detail block (no ### mode tag active).
        # Only after at least one field row — a leading #### before the table
        # is section prose, not a field detail (prevents swallowing the table).
        if (
            not st.fence
            and not st.in_mode
            and st.in_field_section
            and st.last_field_path is not None
            and (m := _RE_DETAIL_HEAD.match(line))
        ):
            flush_section_detail()
            st.detail_tag = m["tag"].strip()
            st.detail_lines.clear()
            continue

        if not st.fence and st.section is not None and (hm3 := _RE_HEADING.match(line)):
            lvl = len(hm3.group("hashes"))
            # Level-agnostic close: only headings at or above the section's level close it.
            # Deeper headings (L+1, L+2) are details/modes handled above; a leading
            # ``#### Detailed`` before the table (level 4 > 2) must NOT close the
            # ``##`` section — it would swallow the subsequent table.
            if st.section_level is not None and lvl <= st.section_level:
                flush_any_detail()
                close_mode()
                _finalize_post_heading()
                st.section, st.in_field_section = None, False
                st.section_level = None
                st.last_field_path = None
                st.section_anchor = ""
                continue
            if st.capture_post_heading and lvl > (st.section_level or 0):
                # Leading detail heading before table (e.g. ``#### Detailed`` under
                # ``## `input.coefs```) — keep section open and capture heading body
                # as part of post_heading paragraph rather than closing.
                body = hm3.group("body").strip()
                # Strip anchor id if any.
                body = _RE_ANCHOR_ID.sub("", body).strip()
                if body:
                    st.post_heading_para.append(body)
                continue
            # Deeper heading not handled as field/mode detail but still deeper than
            # section — don't close; let it fall through to content handlers.
            if lvl > (st.section_level or 0):
                # If in_mode already, it would have been handled as detail/sub-detail;
                # if not, treat as plain prose inside section (not closing).
                # For field-level detail case already handled, this is a no-op.
                # Preserve heading text as post_heading if still capturing?
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
            if st.sub_detail_tag is not None:
                target = st.sub_detail_lines
            elif st.detail_tag is not None:
                target = st.detail_lines
            else:
                target = st.short_lines
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
                short=_clean_text(cells[-1]).strip() if cells else "",
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
        if isinstance(raw, ModeBody):
            # Case-insensitive lookup for base detail
            base = raw.details.get(detail, "")
            if base == "":
                for k, v in raw.details.items():
                    if k.lower() == detail.lower():
                        base = v
                        break
            # Include nested ``Important`` under ``Detailed`` without breaking
            # (previously ``##### Important`` closed the ``#### Detailed`` block)
            subs = getattr(raw, "sub_details", {})
            # Find Detailed key case-insensitively
            sub_map = None
            for k, v in subs.items():
                if k.lower() == detail.lower():
                    sub_map = v
                    break
            if sub_map:
                important_parts = [
                    sv.strip()
                    for sk, sv in sub_map.items()
                    if sk.lower() == "important" and sv.strip()
                ]
                if important_parts:
                    body = (base.strip() + "\n" + "\n".join(important_parts)) if base.strip() else "\n".join(
                        important_parts
                    )
                else:
                    body = base
            else:
                body = base
        else:
            body = ""

    return replace(entry, body=body)


def section_body_short(entry: HelpEntry) -> str:
    """Return the best short status text for a help entry.

    Prefers the ``###`` section lead-in text (stored under ``_NO_MODE``) —
    the "section status below the table" — over the table row's last cell
    (``entry.short``).  The section text is typically more descriptive, so
    the status bar shows it when both exist; otherwise falls back to the
    table row short.
    """
    if isinstance(entry.body, Mapping):
        raw = entry.body.get(_NO_MODE)
        if isinstance(raw, ModeBody):
            if raw.short:
                return raw.short
        elif isinstance(raw, str) and raw:
            return raw
    return entry.short


def _filter_h5_lines(text: str) -> str:
    """Drop lines mentioning HDF5/NetCDF when H5 is unavailable."""
    if _constants.H5_AVAILABLE:
        return text
    return "\n".join(
        line for line in text.split("\n") if not re.search(r"HDF5|NetCDF", line, re.IGNORECASE)
    )


def _clean_text(text: str) -> str:
    """Clean doc-extracted text for GUI status/tooltip display.

    - Strip ``[↓](#anchor)`` links — arrow-only, no useful display text.
    - When H5 is unavailable, drop lines mentioning HDF5/NetCDF.
    """
    return _filter_h5_lines(_RE_DOWN_LINK.sub("", text))


def help_general_for_path(path: str) -> str:
    """Return the general (modeless) description for a config path.

    This is the ``### `field` `` body — the text before any
    ``#### <mode>`` or ``### `field` <mode>mode</mode>`` section.  For
    ``path_field`` (and any field) the ``Important`` hint is the
    concatenation (``\\n``-joined) of:

    * all ``Important`` (case-insensitive) first-child sections of the parent
      (``L+1`` where ``L`` is the parent heading level, e.g. ``#### Important``
      under ``### `path_field```), and
    * all ``Important`` subsections under every ``Detailed`` child
      (``L+2``, e.g. ``##### Important`` under ``#### Detailed``).

    Previously only a literal ``#### Important`` was returned and
    ``##### Important`` under ``Detailed`` was lost because the parser broke
    the tooltip at any new heading.  The parser is now level-agnostic
    (``L+1``/``L+2`` derived from the parent) and preserves the nested
    ``Important`` via :attr:`ModeBody.sub_details`; this function aggregates
    both sources.  If no ``Important`` block exists, falls back to the short
    pre-``####`` body.  Mode-specific bodies are irrelevant for this call.

    Array indices are stripped before lookup, mirroring :func:`help_for_path`.
    """
    entry = _load().get(_RE_ARR_INDEX.sub("", path))
    if entry is None:
        return ""
    if isinstance(entry.body, Mapping):
        raw = entry.body.get(_NO_MODE, "")
        if isinstance(raw, ModeBody):
            # Collect Important at L+1 and Important under Detailed at L+2.
            parts: list[str] = []
            if isinstance(raw.details, Mapping):
                for tag, content in raw.details.items():
                    if tag.strip().lower() == "important" and content.strip():
                        parts.append(content.strip())
            subs = getattr(raw, "sub_details", {})
            if isinstance(subs, Mapping):
                for dtag, inner in subs.items():
                    if dtag.strip().lower() == "detailed" and isinstance(inner, Mapping):
                        for stag, scontent in inner.items():
                            if stag.strip().lower() == "important" and scontent.strip():
                                parts.append(scontent.strip())
            if parts:
                return "\n".join(parts)
            return raw.short
        return raw if isinstance(raw, str) else ""
    return ""
