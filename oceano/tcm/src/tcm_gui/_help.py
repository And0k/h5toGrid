"""Resolve Hydra config paths to help text auto-extracted from ``config_reference.md``.

Single source of truth for non-chrome (config-cell) tooltips: the existing
field tables in ``docs/tcm_cli/config_reference.md``.  No dotted-path headings
need to be authored — the parser walks the table rows under each
``## ``{section}```` heading (``input``, ``input.coefs``, ``out``, ``filter``,
``program``) and emits one ``HelpEntry`` per row, keyed by
``{section}.{field}``.  The last cell of each row (``Purpose`` or
``Physical meaning``) is the short tooltip / hover-status.

**Mode-tagged sections** — When a field's meaning depends on context (per-probe
processing vs. input specification), detailed documentation goes into ``###``
subsections tagged with a mode: ``### `input.path` <mode>probe</mode>``.
The parser extracts these into ``HelpEntry.body`` as a ``dict[str, str]``
(mode → content).  Consumers select by mode:

    help_for_path("input.path", mode="probe")   # per-probe body only
    help_for_path("input.path", mode="search")  # input patterns only
    help_for_path("input.path")                 # full dict {"probe": ..., "search": ...}

**Detail sub-blocks** — A mode section may carry ``####`` sub-blocks; the
canonical one is ``#### Detailed`` (the long-form requirements / tooltip body).
Lines before the first ``####`` = short status lines; the ``#### <Tag>`` block
= the detail string addressed by ``detail="<Tag>"``:

    help_for_path("input.path", mode="search", detail="Detailed")  # long-form body
    help_for_path("input.path", mode="search")                     # short lines only

Cell hover resolution does not call ``set_widget_meta`` — ``coef_sheet`` looks
up ``_meta[iid]["path"]``, strips array indices (``Ag[0]`` → ``Ag``) and
calls :func:`help_for_path`.  i18n = swap ``config_reference_{lang}.md``
(resolved via :func:`tcm_gui.const.resolve_lang`, fallback to English).

Arrays documented at the field level (``input.coefs.Ag``) — children (``Ag[0]``,
``Ag[1][2]``) reuse the parent entry via index stripping.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from tcm import _constants
from tcm._md_parse import split_table_row

from ._i18n import resolve_lang

_l = logging.getLogger(__name__)


def _doc_path(lang: str|None = None) -> Path:
    """Localized config_reference path; English fallback."""
    if lang and (p := _constants.DOC_DIR / f"config_reference_{lang}.md").is_file():
        return p
    return _constants.DOC_DIR / "config_reference.md"


# Sections whose ``## ``{section}```` heading opens table-driven field scanning.
# ``filter/calib`` and decision-table sections are excluded — those fields are
# either unused by the GUI Hydra tree (calib entry point) or carry
# non-identifier first columns (``Stage``, ``Column``, …).
_FIELD_SECTIONS: frozenset[str] = frozenset({"input", "input.coefs", "out", "filter", "program"})

# ``## ``input.coefs```` — dotted identifier inside backticks at level 2.
# Group 1 = section identifier, group 2 = subtitle text after the separator
# (``—``, ``:``).  Accepts CamelCase field names (``Ag``, ``Cg``, ``Rz``, ``P``)
# — Hydra sections convention is lowercase, but coefficient names use
# Cyrillic-stemed abbrevs.
_RE_SECTION_HEAD = re.compile(r"^##\s+`([A-Za-z_]\w*(?:\.\w+)*)`\s*(?:—\s*(.+))?\s*$")
# ``### \`field.path\` <mode>value</mode>`` — mode-tagged field detail section.
# Group 1 = dotted field path, group 2 = mode value (e.g. ``probe``, ``search``).
# ``</>`` shorthand also accepted as closing tag.
_RE_FIELD_MODE_HEAD = re.compile(r"^###\s+`([A-Za-z_]\w*(?:\.\w+)*)`\s+<mode>([a-z_]+)</(?:mode)?>")
# ``#### <Tag>`` — sub-block INSIDE the active ``###`` mode (does NOT close it).
# Group 1 = detail tag slug (e.g. ``Detailed``).  Only recognized while a mode
# is open; outside a mode a ``####`` is plain prose.
_RE_DETAIL_HEAD = re.compile(r"^####\s+(.+?)\s*$")
# Any markdown heading — closes the current section's field scan.
# NOTE: used only AFTER _FIELD_MODE_HEAD and _DETAIL_HEAD are ruled out, so
# ``###`` mode headers and ``####`` detail headers don't close the parent.
_RE_ANY_HEADING = re.compile(r"^#{1,6}\s")
# Code-fence toggle (``` or ~~~) — ``#`` inside is not a heading.
_RE_FENCE = re.compile(r"^\s*(```|~~~)")
# Table row whose first cell is a bare backtick-quoted identifier:
# ``| `field_name` | … | description |``.
_RE_FIELD_ROW = re.compile(r"^\|\s*`([A-Za-z_]\w*)`\s*\|")
# Array indices at lookup: ``Ag[0]`` / ``Ag[1][2]`` → ``Ag``.
_RE_ARR_INDEX = re.compile(r"\[\d+\]")


@dataclass(frozen=True, slots=True)
class HelpEntry:
    """One field's help.

    ``short``  — last cell of the table row (tooltip / hover-status).
    ``body``   — mode-tagged detail content: ``{"probe": ..., "search": ...}``.
                 Empty dict when no ``###`` subsections exist for this field.
                 When :func:`help_for_path` is called with *mode*, ``body`` is
                 reduced to the single mode's content: a ``str`` (short lines)
                 when *detail* is ``None``, or the named detail block string
                 when *detail* is set.
    ``path``   — dotted Hydra path (``input.coefs.Ag``).
    """

    path: str
    short: str
    body: str | dict[str, str]


@dataclass(slots=True)
class _ModeBody:
    """Mode content split into short lines + named ``####`` detail blocks.

    ``short`` = lines accumulated before the first ``####`` in the mode section
    (== the whole mode body when no ``####`` is present).
    ``details`` = ``{tag: text}`` for each ``#### <Tag>`` sub-block.
    """

    short: str = ""
    details: dict[str, str] = field(default_factory=dict)

    @property
    def has_details(self) -> bool:
        return bool(self.details)


def parse_reference(text: str) -> dict[str, HelpEntry]:
    """Parse ``config_reference.md`` content → ``{path: HelpEntry}``.

    One pass over the lines while tracking the current ``## ``{section}````
    block (skipping code fences).  For each field section, markdown table rows
    emit entries with ``short`` from the last cell; ``###`` mode-tagged
    subheaders (``### `field.path` <mode>value</mode>``) accumulate detail
    content into ``body[mode]``.  Inside an open mode, ``#### <Tag>`` headings
    start named detail sub-blocks (the canonical one is ``Detailed``) — they
    do NOT close the parent mode; only a subsequent ``###`` or ``##`` does.
    """
    entries: dict[str, HelpEntry] = {}

    def _scan(lines: list[str]) -> None:
        section: str | None = None
        in_field_section = False
        fence = False
        # Mode-tagged section accumulation state.
        mode_path: str | None = None  # field path from ### heading
        mode_tag: str | None = None  # mode value (e.g. "probe")
        mode_short_lines: list[str] = []  # short-status lines (pre-first ####)
        mode_detail_tag: str | None = None  # active #### tag (None until first ####)
        mode_detail_lines: list[str] = []  # lines for current #### block
        mode_details: dict[str, str] = {}  # tag → joined detail text

        def _flush_detail() -> None:
            """Freeze accumulated #### block lines into mode_details."""
            nonlocal mode_detail_tag
            if mode_detail_tag is not None:
                content = "\n".join(mode_detail_lines).strip()
                mode_details[mode_detail_tag] = content
            mode_detail_tag = None
            mode_detail_lines.clear()

        def _close_mode() -> None:
            """Freeze accumulated mode content into the entry's body dict."""
            nonlocal mode_path, mode_tag, mode_detail_tag
            _flush_detail()
            if mode_path and mode_tag and mode_path in entries:
                short = "\n".join(mode_short_lines).strip()
                e = entries[mode_path]
                new_body = dict(e.body) if isinstance(e.body, dict) else {}
                new_body[mode_tag] = (
                    short if not mode_details else _ModeBody(short=short, details=dict(mode_details))
                )
                entries[mode_path] = HelpEntry(e.path, e.short, new_body)
            mode_path, mode_tag, mode_detail_tag = None, None, None
            mode_short_lines.clear()
            mode_detail_lines.clear()
            mode_details.clear()

        for line in lines:
            if _RE_FENCE.match(line):
                # Preserve fence markers in the mode body so downstream
                # parse_markdown still recognises fenced code blocks —
                # otherwise the rows inside ```…``` collapse into one
                # Paragraph (flush_para joins them with spaces).  The
                # `fence` state still guards heading detection below.
                if mode_path and mode_tag:
                    tgt = mode_detail_lines if mode_detail_tag is not None else mode_short_lines
                    tgt.append(line)
                fence = not fence
                continue

            if not fence and (m := _RE_SECTION_HEAD.match(line)):
                _close_mode()
                section = m.group(1)
                in_field_section = section in _FIELD_SECTIONS
                # Emit section-level entry: ``input`` → "Data source & parameters",
                # ``input.coefs`` → "Calibration coefficients", etc.
                if in_field_section:
                    subtitle = (m.group(2) or "").strip()
                    entries[section] = HelpEntry(section, subtitle or section, {})
                continue

            # ### mode-tagged subheader — MUST be checked BEFORE _ANY_HEADING
            # so that ### doesn't close the current ## section.
            if not fence and (fm := _RE_FIELD_MODE_HEAD.match(line)):
                _close_mode()
                mode_path, mode_tag = fm.group(1), fm.group(2)
                mode_short_lines.clear()
                mode_details.clear()
                mode_detail_tag = None
                continue

            # #### detail subheader — only meaningful INSIDE a mode; otherwise
            # treated as plain prose (falls through to accumulation / row scan).
            # MUST be checked BEFORE _ANY_HEADING so #### doesn't close the mode.
            if (
                not fence
                and mode_path is not None
                and mode_tag is not None
                and (dm := _RE_DETAIL_HEAD.match(line))
            ):
                _flush_detail()
                mode_detail_tag = dm.group(1).strip()
                mode_detail_lines.clear()
                continue

            if (not fence) and section is not None and _RE_ANY_HEADING.match(line):
                _close_mode()
                section, in_field_section = None, False
                continue

            if mode_path and mode_tag:
                # Inside a mode: route lines to short-status or active detail.
                if mode_detail_tag is not None:
                    mode_detail_lines.append(line)
                else:
                    mode_short_lines.append(line)
                continue

            if section is None:
                continue

            if not in_field_section:
                continue

            if (fm := _RE_FIELD_ROW.match(line)) and (field := fm.group(1)):
                path = f"{section}.{field}"
                cells = split_table_row(line)
                raw = cells[-1].strip() if cells else ""
                entries[path] = HelpEntry(path, raw, {})

        _close_mode()

    _scan(text.splitlines())
    return entries


# ── loader & resolver ─────────────────────────────────────────────────────────


_CACHE: dict[str, dict[str, HelpEntry]] = {}
"""Per-language parsed entries.  Key = resolved two-letter lang code (e.g.
``"en"``, ``"ru"``).  Populated lazily by :func:`_load`; cleared by
:func:`reload_cache` (tests monkeypatch ``_constants.DOC_DIR`` then call it)."""


def _load(lang: str) -> dict[str, HelpEntry]:
    """Read & parse the reference once per language; log a one-line summary.

    Memoized in the module-level :data:`_CACHE` (not ``functools.lru_cache`` —
    tests call :func:`reload_cache` to bypass it after monkeypatching
    ``_constants.DOC_DIR``).  On any read/parse error logs at INFO and caches ``{}``
    for that lang (graceful degradation: no tooltips, no crash).
    """
    if lang in _CACHE:
        return _CACHE[lang]
    path = _doc_path(lang)
    try:
        text = path.read_text(encoding="utf-8")
        entries = parse_reference(text)
    except OSError:
        _l.error("config_reference not found at %s — config-cell hover disabled", path)
        entries = {}
    except Exception:  # noqa: BLE001 — any parse failure is non-fatal here
        _l.error("config_reference parse error at %s — hover disabled", path, exc_info=True)
        entries = {}
    _l.debug("Loaded %d config help entries from %s (lang=%s)", len(entries), path, lang)
    _CACHE[lang] = entries
    return entries


def reload_cache(lang: str | None = None) -> dict[str, HelpEntry]:
    """Force a fresh parse; ``lang=None`` clears ALL cached languages.

    Tests that monkeypatch ``_constants.DOC_DIR`` call ``reload_cache()`` (no-arg,
    preserves the existing contract) to drop stale entries; tests targeting a
    specific lang pass it explicitly to evict just that slot.
    """
    if lang is None:
        _CACHE.clear()
    else:
        _CACHE.pop(lang, None)
    return _load(lang if lang is not None else resolve_lang())


def help_for_path(path: str, *, mode: str | None = None, detail: str | None = None):
    """Dotted Hydra path → entry; array indices stripped (``Ag[0]`` → ``Ag``).

    Modes:
    - ``mode=None, detail=None`` → entry as-is (``body`` is a ``dict`` of
      per-mode content; each value is a ``str`` (no ``####``) or a
      :class:`_ModeBody` (mode carries ``####`` detail blocks)).
    - ``mode=..., detail=None`` → new :class:`HelpEntry` with ``body`` reduced
      to the mode's short lines (``str``).  For modes without ``####`` this is
      the whole mode body (backward-compatible); for modes with ``####`` it is
      the text before the first ``####``.
    - ``mode=..., detail=<tag>`` → new :class:`HelpEntry` with ``body`` reduced
      to that detail block's text (``str``).  Returns ``body=""`` when the
      mode has no such ``####`` block (caller no-ops).
    """
    if (entry := _load(resolve_lang()).get(_RE_ARR_INDEX.sub("", path))) is None:
        return None
    if mode is None:
        return entry
    raw = entry.body.get(mode, "") if isinstance(entry.body, dict) else ""
    if detail is None:
        body = raw.short if isinstance(raw, _ModeBody) else raw
        return HelpEntry(entry.path, entry.short, body)
    body = raw.details.get(detail, "") if isinstance(raw, _ModeBody) else ""
    return HelpEntry(entry.path, entry.short, body)
