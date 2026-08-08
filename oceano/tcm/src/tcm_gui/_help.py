"""Resolve Hydra config paths to help text auto-extracted from ``config_reference.md``.

Single source of truth for non-chrome (config-cell) tooltips: the existing
field tables in ``docs/tcm_clc/config_reference.md``.  No dotted-path headings
need to be authored — the parser walks the table rows under each
``## ``{section}```` heading (``input``, ``input.coefs``, ``out``, ``filter``,
``program``) and emits one ``HelpEntry`` per row, keyed by
``{section}.{field}``.  The last cell of each row (``Purpose`` or
``Physical meaning``) is the short tooltip / hover-status; the entire section
prose (headings-bound) is the long popup body.

Cell hover resolution does not call ``set_widget_meta`` — ``coef_sheet`` looks
up ``_meta[iid]["path"]``, strips array indices (``Ag[0]`` → ``Ag``) and
calls :func:`help_for_path`.  i18n = swap ``config_reference_<lang>.md``.

Arrays documented at the field level (``input.coefs.Ag``) — children (``Ag[0]``,
``Ag[1][2]``) reuse the parent entry via index stripping.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from tcm import _constants
from tcm._md_parse import split_table_row

_l = logging.getLogger(__name__)

# Resolved once at import: ``tcm/__file__`` → ``src/tcm``; two parents up is
# the project root in dev and ``_internal`` in pyinstaller ``--onedir`` dist,
# both of which carry ``docs/tcm_clc/config_reference.md`` as a sibling.
_DOC_PATH: Path = _constants.PROJECT_ROOT.parent.parent / "docs" / "tcm_clc" / "config_reference.md"

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
_SECTION_HEAD = re.compile(r"^##\s+`([A-Za-z_]\w*(?:\.\w+)*)`\s*(?:—\s*(.+))?\s*$")
# Any markdown heading — closes the current section's field scan.
_ANY_HEADING = re.compile(r"^#{1,6}\s")
# Code-fence toggle (``` or ~~~) — ``#`` inside is not a heading.
_FENCE = re.compile(r"^\s*(```|~~~)")
# Table row whose first cell is a bare backtick-quoted identifier:
# ``| `field_name` | … | description |``.
_FIELD_ROW = re.compile(r"^\|\s*`([A-Za-z_]\w*)`\s*\|")
# Array indices at lookup: ``Ag[0]`` / ``Ag[1][2]`` → ``Ag``.
_ARR_INDEX = re.compile(r"\[\d+\]")


@dataclass(frozen=True, slots=True)
class HelpEntry:
    """One field's help.

    ``short``  — last cell of the table row (tooltip / hover-status).
    ``body``   — full prose of the owning ``## ``{section}```` block (popup text),
                 shared across all fields of the section.
    ``path``   — dotted Hydra path (``input.coefs.Ag``).
    """

    path: str
    short: str
    body: str


# ── parser ────────────────────────────────────────────────────────────────────


def parse_reference(text: str) -> dict[str, HelpEntry]:
    """Parse ``config_reference.md`` content → ``{path: HelpEntry}``.

    One pass over the lines while tracking the current ``## ``{section}````
    block (skipping code fences).  For each ``## ``input```` / ``## ``out````
    / ``## ``filter```` / ``## ``program```` / ``## ``input.coefs```` section,
    every markdown table row with a bare-identifier first cell emits an entry
    keyed by ``{section}.{field}``; the section prose accumulated between the
    heading and the next heading is shared as ``body``.
    """
    entries: dict[str, HelpEntry] = {}

    def _scan(lines: list[str]) -> None:
        section: str | None = None
        in_field_section = False
        fence = False
        body_lines: list[str] = []
        pending: list[str] = []  # paths awaiting body fill

        def _close() -> None:
            """Freeze section prose as ``body`` on all pending entries; reset."""
            body = "\n".join(body_lines).strip()
            for p in pending:
                e = entries[p]
                entries[p] = HelpEntry(p, e.short, body)
            pending.clear()
            body_lines.clear()

        for line in lines:
            if _FENCE.match(line):
                fence = not fence
                if section is not None:
                    body_lines.append(line)
                continue

            if not fence and (m := _SECTION_HEAD.match(line)):
                if section is not None:
                    _close()
                section = m.group(1)
                in_field_section = section in _FIELD_SECTIONS
                # Emit section-level entry: ``input`` → "Data source & parameters",
                # ``input.coefs`` → "Calibration coefficients", etc.
                # The subtitle comes from the heading line after the ``—`` separator.
                if in_field_section:
                    subtitle = (m.group(2) or "").strip()
                    entries[section] = HelpEntry(section, subtitle or section, "")
                    pending.append(section)
                body_lines.append(line)
                continue

            if (not fence) and section is not None and _ANY_HEADING.match(line):
                _close()
                section, in_field_section = None, False
                continue

            if section is None:
                continue

            body_lines.append(line)
            if not in_field_section:
                continue

            if (fm := _FIELD_ROW.match(line)) and (field := fm.group(1)):
                path = f"{section}.{field}"
                cells = split_table_row(line)
                short = cells[-1].strip() if cells else ""
                entries[path] = HelpEntry(path, short, "")
                pending.append(path)

        if section is not None:
            _close()

    _scan(text.splitlines())
    return entries


# ── loader & resolver ─────────────────────────────────────────────────────────


def _load() -> dict[str, HelpEntry]:
    """Read & parse the reference once; log a one-line summary at INFO.

    Memoized via :func:`functools.lru_cache`; tests call :func:`reload_cache`
    to bypass it after monkeypatching ``_DOC_PATH``.
    """
    return _load_impl()


@lru_cache(maxsize=1)
def _load_impl() -> dict[str, HelpEntry]:
    try:
        text = _DOC_PATH.read_text(encoding="utf-8")
    except OSError:
        _l.info("config_reference.md not found at %s — config-cell hover disabled", _DOC_PATH)
        return {}
    entries = parse_reference(text)
    _l.debug("Loaded %d config help entries from %s", len(entries), _DOC_PATH)
    return entries


def reload_cache() -> dict[str, HelpEntry]:
    """Force a fresh parse — tests use this after changing ``_DOC_PATH``."""
    _load_impl.cache_clear()
    return _load()


def help_for_path(path: str) -> HelpEntry | None:
    """dotted Hydra path → entry; array indices stripped (``Ag[0]`` → ``Ag``)."""
    return _load().get(_ARR_INDEX.sub("", path))
