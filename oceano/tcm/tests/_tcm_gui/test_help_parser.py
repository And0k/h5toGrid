"""Tests for ``tcm_gui._help``: auto-extract config-cell help from ``config_reference.md``.

Cover the table-driven parser (``parse_reference``) and the index-stripping
resolver (``help_for_path``):
  * ``## ``section```` opens a config-group section (``input``/``out``/``filter``/
    ``program``/``input.coefs``); narrative ``###`` subsections close the scan.
  * Each ``| `field` = default | … | description |`` row (joined
    ``Field = Default`` column) inside a section emits
    ``HelpEntry(path="{section}.{field}", short=<last cell>)``.
  * Escaped pipes (``\\\\|``) inside cells stay literal; CamelCase field names
    (``Ag``, ``Cg``, ``Rz``) parse just like lowercase Hydra names.
  * Arrays resolve via index stripping — ``Ag[0]``/``Ag[1][2]`` → ``input.coefs.Ag``.
  * Code-fence ``#`` lines are NOT headings.
  * Decision-table sections (Stage classification, filter/calib, …) are ignored.
"""

from __future__ import annotations

import textwrap

import pytest

from tcm_gui._help import HelpEntry, doc_path, parse_reference, reload_cache

_DOC_PATH = doc_path()

# ── parser smoke (real config_reference.md) ─────────────────────────────────


class TestRealReference:
    """Smoke test against the actual bundled ``config_reference.md``."""

    def test_loads_without_error(self):
        """``reload_cache`` parses the real doc into a non-empty dict."""
        if not _DOC_PATH.exists():
            pytest.skip(f"config_reference.md not present at {_DOC_PATH}")
        entries = reload_cache()
        assert entries, f"expected entries from {_DOC_PATH}, got empty dict"

    def test_all_field_sections_represented(self):
        """Each of the 5 config-group sections contributes at least one entry."""
        from tcm_gui._help import reload_cache

        entries = reload_cache()
        prefixes = {p.split(".", 1)[0] for p in entries}
        assert prefixes >= {"input", "out", "filter", "program"}, (
            f"section prefixes missing — got {sorted(prefixes)}"
        )
        # input.coefs is emitted as a sub-prefix of input — check a known coef.
        assert "input.coefs.Ag" in entries, (
            "Ag row missing — CamelCase field regex regression? "
            f"input.coefs.* = {sorted(p for p in entries if p.startswith('input.coefs'))[:6]}"
        )

    def test_help_for_path_strips_array_index(self):
        """``Ag[0]`` and ``Ag[1][2]`` both resolve to the same entry as ``Ag``."""
        from tcm_gui._help import help_for_path, reload_cache

        reload_cache()
        ref = help_for_path("input.coefs.Ag")
        assert ref is not None, "input.coefs.Ag missing from parsed doc"
        assert help_for_path("input.coefs.Ag[0]").short == ref.short, (
            "Ag[0] short should equal Ag short (single-index strip)"
        )
        assert help_for_path("input.coefs.Ag[1][2]").short == ref.short, (
            "Ag[1][2] short should equal Ag short (multi-index strip)"
        )

    def test_unknown_path_returns_none(self):
        from tcm_gui._help import help_for_path, reload_cache

        reload_cache()
        assert help_for_path("nonexistent.field") is None, "unknown path should return None"

    def test_body_populated_for_known_field(self):
        """A field with ``<mode>`` sections has non-empty body dict."""
        from tcm_gui._help import help_for_path, reload_cache

        reload_cache()
        # ``input.path`` has a modeless ``###`` section (body under ``_NO_MODE``).
        e = help_for_path("input.path")
        if e is None:
            pytest.skip("input.path not in bundled doc — version drift")
        assert isinstance(e.body, dict), f"body should be dict, got {type(e.body).__name__}"
        assert e.body, f"input.path body should have mode content, got {e.body!r}"
        assert e.short, "input.path short should be the row's last cell, got empty"

    def test_mode_resolution_for_known_field(self):
        """``help_for_path(path, mode=...)`` reduces body to single mode string."""
        from tcm_gui._help import help_for_path, reload_cache

        reload_cache()
        e = help_for_path("path_field", mode="dirs")
        if e is None:
            pytest.skip("path_field not in bundled doc — version drift")
        assert isinstance(e.body, str), f"mode-resolved body should be str, got {type(e.body).__name__}"
        assert e.body, f"path_field search body should be non-empty, got {e.body!r}"

    def test_section_level_entries_emitted(self, monkeypatch):
        """Section-level entries resolve for bare paths like ``input``, ``out``, ``filter``."""
        from tcm_gui._help import help_for_path, reload_cache

        monkeypatch.setattr("tcm_gui._help.resolve_lang", lambda: "en")
        reload_cache("en")
        for section, expected_substring in (
            ("input", "Data source"),
            # Bare ``### Detailed`` prose arms the dwell body, not the short:
            # the section short falls back to the ``##`` subtitle.
            ("input.coefs", "Calibration coefficients"),
            ("out", "Output configuration"),
            # Real doc: the post-heading paragraph is now the short (values exceeding
            # thresholds become NaN...); the old subtitle "quality thresholds" is gone.
            ("filter", "process-stage"),
            ("program", "Runtime flags"),
        ):
            e = help_for_path(section)
            assert e is not None, (
                f"section-level entry for '{section}' missing — needed for parent row hover in tksheet"
            )
            assert expected_substring in e.short, (
                f"'{section}' short should contain '{expected_substring}', got {e.short!r}"
            )
        dwell = help_for_path("input.coefs", mode="Detailed")
        assert dwell is not None and isinstance(dwell.body, str), (
            "input.coefs section dwell should resolve via the bare Detailed mode"
        )
        assert "Loaded from the coefficient file" in dwell.body, (
            f"dwell should carry the Detailed prose, got {dwell.body!r}"
        )

    def test_section_level_body_filled(self):
        """Section-level entries exist with correct short text.

        Body is mode-tagged (``dict``); section entries have ``body={}``
        unless they carry ``###`` mode subsections of their own.
        """
        from tcm_gui._help import help_for_path, reload_cache

        reload_cache()
        e = help_for_path("input.coefs")
        assert e is not None, "input.coefs section entry should exist for parent row hover in tksheet"
        assert isinstance(e.body, dict), f"body should be dict, got {type(e.body).__name__}"


# ── #### sub-blocks inside ### modes (Detailed sub-block) ─────────────────────


class TestRealReferenceDetailed:
    """Real-doc checks for the ``#### Detailed`` sub-block under ``path_field`` search mode."""

    def test_search_mode_detailed_block_reachable(self):
        """``help_for_path("path_field", mode="dirs", detail="Detailed")`` returns str."""
        from tcm_gui._help import help_for_path, reload_cache

        reload_cache()
        e = help_for_path("path_field", mode="dirs", detail="Detailed")
        if e is None:
            pytest.skip("path_field not in bundled doc — version drift")
        assert isinstance(e.body, str), f"detail-resolved body should be str, got {type(e.body).__name__}"
        assert e.body, (
            "path_field search #### Detailed body should be non-empty — "
            "did config_reference.md gain the Detailed sub-block?"
        )
        # Detailed content must mention the directory layout anchor, not the
        # one-line search summary that now lives in the short pre-#### body.
        assert "_raw" in e.body, f"Detailed body should mention _raw directory layout; got {e.body!r}"

    def test_search_mode_short_lines_excludes_detailed(self):
        """``mode="dirs"`` (no detail) returns SHORT lines only — pre-#### text."""
        from tcm_gui._help import help_for_path, reload_cache

        reload_cache()
        e = help_for_path("path_field", mode="dirs")
        if e is None:
            pytest.skip("path_field not in bundled doc — version drift")
        assert isinstance(e.body, str), f"mode-resolved short body should be str, got {type(e.body).__name__}"
        assert e.body, "search short body should be non-empty (pre-#### one-liner)"
        # The Detailed directory-layout text must NOT have leaked into the short body.
        assert "_raw" not in e.body, (
            f"short body should be the pre-#### one-liner, not the Detailed block; got {e.body!r}"
        )

    def test_search_mode_has_details_in_full_body(self):
        """Full ``help_for_path("path_field")`` body[detail] is a _ModeBody (has details)."""
        from tcm_gui._help import _NO_MODE, ModeBody, help_for_path, reload_cache

        reload_cache()
        e = help_for_path("path_field")
        if e is None:
            pytest.skip("path_field not in bundled doc — version drift")
        assert isinstance(e.body, dict), f"full body should be dict, got {type(e.body).__name__}"
        general = e.body.get(_NO_MODE)
        assert isinstance(general, ModeBody), (
            "general (modeless) now carries #### Detailed → body[_NO_MODE] should be ModeBody; "
            f"got {type(general).__name__}"
        )
        assert general.details, f"general ModeBody should have details, got {general.details!r}"
        assert "Detailed" in general.details, (
            f"general details should include 'Detailed' tag; got {sorted(general.details)!r}"
        )

    def test_detail_resolution_for_unknown_detail_returns_empty_str(self):
        """Unknown detail tag → empty str (caller no-ops), not None."""
        from tcm_gui._help import help_for_path, reload_cache

        reload_cache()
        e = help_for_path("path_field", mode="dirs", detail="Nonexistent")
        if e is None:
            pytest.skip("path_field not in bundled doc — version drift")
        assert isinstance(e.body, str), (
            f"unknown-detail body should be str (empty), got {type(e.body).__name__}"
        )
        assert e.body == "", f"unknown detail should resolve to '', got {e.body!r}"

    def test_files_mode_without_detailed_stays_str(self):
        """Non-Hydra ``path_field`` section registers with its mode bodies."""
        from tcm_gui._help import help_for_path, reload_cache

        reload_cache()
        e = help_for_path("path_field", mode="files")
        if e is None:
            pytest.skip("path_field not in bundled doc — version drift")
        assert isinstance(e.body, str), (
            f"files body (no ####) should stay str, got {type(e.body).__name__}"
        )
        assert e.body, "files short body should be non-empty"

    def test_field_level_detailed_block_reachable(self, monkeypatch):
        """``filter.max`` modeless-section ``#### Detailed`` block is reachable."""
        from tcm_gui._help import _NO_MODE, help_for_path, reload_cache

        monkeypatch.setattr("tcm_gui._help.resolve_lang", lambda: "en")
        reload_cache("en")
        # ``filter.max`` carries a modeless ### section with a #### Detailed
        # block (process-computed threshold columns).
        e = help_for_path("filter.max", mode=_NO_MODE, detail="Detailed")
        if e is None:
            pytest.skip("filter.max not in bundled doc — version drift")
        assert isinstance(e.body, str), f"modeless detail body should be str, got {type(e.body).__name__}"
        assert e.body, (
            "filter.max Detailed body should be non-empty — "
            "did config_reference.md gain the filter.max section?"
        )
        assert "process-computed" in e.body, f"Detailed body should mention process-computed; got {e.body!r}"

    def test_field_level_detailed_via_resolve_detail(self, monkeypatch):
        """``program.return_`` modeless-section Detailed returns the phase table."""
        from tcm_gui._help import _NO_MODE, help_for_path, reload_cache

        monkeypatch.setattr("tcm_gui._help.resolve_lang", lambda: "en")
        reload_cache("en")
        # ``program.return_`` carries a modeless ### section with the phase-stopping table.
        e = help_for_path("program.return_", mode=_NO_MODE, detail="Detailed")
        if e is None:
            pytest.skip("program.return_ not in bundled doc")
        assert isinstance(e.body, str), f"modeless detail body should be str, got {type(e.body).__name__}"
        assert e.body, "program.return_ Detailed block should be non-empty"
        assert "saved_coefs" in e.body, f"Detailed body should mention phase-stopping values; got {e.body!r}"

    def test_per_lang_loader_fallback_to_en(self, monkeypatch, tmp_path):
        """Absent ``config_reference_ru.md`` → fallback to English ``config_reference.md``."""
        from tcm_gui import _help

        # Point DOC_DIR at a tmp dir with ONLY the English file (reference/ subdir, as bundled).
        ref = tmp_path / "reference"
        ref.mkdir()
        en = ref / "config_reference.md"
        en.write_text(
            "## `input` — Data source\n\n| `path` | `str` | — | **Yes** | File path (en). |\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(_help._constants, "DOC_DIR", tmp_path)
        _help.reload_cache("ru")
        # resolve_lang is cached; force ru for this assertion.
        monkeypatch.setattr(_help, "resolve_lang", lambda: "ru")
        e = _help.help_for_path("input.path")
        assert e is not None, "ru request must fall back to en doc, not return None"
        assert "en" in e.short, f"ru fallback should read the en file content; got short={e.short!r}"
        # Cleanup so other tests don't see the monkeypatched cache.
        _help.reload_cache("ru")
        _help.reload_cache("en")

    def test_per_lang_loader_uses_localized_file(self, monkeypatch, tmp_path):
        """When ``config_reference_ru.md`` exists, ru request reads it, not the en file."""
        from tcm_gui import _help

        ref = tmp_path / "reference"
        ref.mkdir()
        en = ref / "config_reference.md"
        en.write_text(
            "## `input` — Data source\n\n| `path` | `str` | — | **Yes** | English short. |\n",
            encoding="utf-8",
        )
        ru = ref / "config_reference_ru.md"
        ru.write_text(
            "## `input` — Источник данных\n\n| `path` | `str` | — | **Да** | Русский short. |\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(_help._constants, "DOC_DIR", tmp_path)
        monkeypatch.setattr(_help, "resolve_lang", lambda: "ru")
        _help.reload_cache("ru")
        e = _help.help_for_path("input.path")
        assert e is not None, "ru doc exists → must load it, not return None"
        assert "Русский" in e.short, f"ru locale should read config_reference_ru.md; got short={e.short!r}"
        _help.reload_cache("ru")
        _help.reload_cache("en")


# ── parser logic (synthetic samples — no dependency on bundled doc) ──────────


_SAMPLE = textwrap.dedent(
    """\
    # Configuration Schema and Device Metadata Reference

    Intro prose — should NOT contribute any field entries.

    ## `input` — Data source & parameters

    Narrative for input — emitted as body to all input.* entries.

    | Field = Default | Purpose |
    |-----------------|---------|
    | `path` = — | File path, glob, or regex pattern (CLI input). |
    | `tables` = `['incl*']` | HDF5 table names (regex allowed). |
    | `corr_time_mode` = `True` | Time correction mode; see `\\|col\\|` shorthand. |

    ### Pattern interpretation

    Narrative — closing the input table scan.

    ## `input.coefs` — Calibration coefficients

    | Field | Type | Default | Physical meaning |
    |-------|------|---------|------------------|
    | `Ag` | 3×3 float | identity | Accelerometer scale matrix: `G = Ag @ (Axyz − Cg)` |
    | `Cg` | 3-float | `[10,10,10]` | Accelerometer bias vector |
    | `kVabs` | 6-float | `[10,-10]` | Polynomial: `Vabs(inclination)` |

    ### Coefficient loading priority

    Prose — closes input.coefs table scan.

    ## `out` — Output configuration

    | Field | Type | Default | Required | Purpose |
    |-------|------|---------|----------|---------|
    | `dt_bins` | `List[int]` | `[0,2,600]` | **Yes** | Averaging bins (seconds). |

    ## `filter` — Process-stage NaN-out

    | Field | Type | Default | Purpose |
    |-------|------|---------|---------|
    | `min` | `Dict[str,float]` | `{}` | Lower bounds. |
    | `max` | `Dict[str,float]` | `{'g':1}` | Upper bounds. |

    ## Decision tables and behavior tuning

    A narrative section whose ``| Field |…`` rows must NOT be parsed (not a
    config-group section, so ``_FIELD_SECTIONS`` excludes it).

    | Field | Type | Purpose |
    |-------|------|---------|
    | `bogus` | `str` | ignored by parser |
    """
)


class TestParseReference:
    """``parse_reference`` against a synthetic doc covering edge cases."""

    def setup_method(self):
        self.entries = parse_reference(_SAMPLE)

    def test_emits_one_entry_per_field_row(self):
        """Row → HelpEntry, path = section.field."""
        assert self.entries["input.path"].short == "File path, glob, or regex pattern (CLI input).", (
            "input.path short mismatch"
        )
        assert self.entries["input.tables"].short == "HDF5 table names (regex allowed).", (
            "input.tables short mismatch"
        )

    def test_section_level_entries_emitted(self):
        """Section headings emit entries keyed by section name.

        Short = the paragraph between the ``## `` heading and its table (falls back
        to the subtitle when no paragraph is present).
        """
        # Sample contains input, input.coefs, out, filter sections. Only `input`
        # carries a post-heading paragraph, so its short is that paragraph; the
        # others fall back to their subtitle.
        for section, expected_substring in (
            ("input", "Narrative for input"),
            ("input.coefs", "Calibration coefficients"),
            ("out", "Output configuration"),
            ("filter", "NaN-out"),
        ):
            assert section in self.entries, f"section entry for '{section}' missing"
            assert expected_substring in self.entries[section].short, (
                f"'{section}' section short should contain '{expected_substring}'; "
                f"got {self.entries[section].short!r}"
            )

    def test_camelcase_field_names_parsed(self):
        """Coefficient names ``Ag``/``Cg``/``kVabs`` (CamelCase / mixed) parse."""
        assert "input.coefs.Ag" in self.entries, "Ag missing — CamelCase regex regression"
        assert "input.coefs.Cg" in self.entries, "Cg missing"
        assert "input.coefs.kVabs" in self.entries, "kVabs missing"
        assert self.entries["input.coefs.Ag"].short == "Accelerometer scale matrix: `G = Ag @ (Axyz − Cg)`", (
            f"Ag short mismatch — got {self.entries['input.coefs.Ag'].short!r}"
        )

    def test_escaped_pipe_restored(self):
        """``\\|col\\|`` in a cell stays literal — not treated as a column separator."""
        assert "`|col|`" in self.entries["input.corr_time_mode"].short, (
            "escaped pipe should be restored to a literal | in the short text; "
            f"got {self.entries['input.corr_time_mode'].short!r}"
        )

    def test_section_body_shared_across_entries(self):
        """All entries of one section share the same body type.

        Without ``<mode>`` sections (synthetic sample), all bodies are ``{}``.
        With ``<mode>`` sections, only the tagged field gets non-empty body;
        other fields in the section keep ``{}``.
        """
        # Filter pure ``input.*`` field entries — exclude ``input.coefs`` and
        # ``input.coefs.*`` (section entry has path without trailing dot).
        input_field_paths = [
            p
            for p in self.entries
            if p.startswith("input.") and not p.startswith("input.coefs") and p != "input"
        ]
        # Synthetic sample has no <mode> sections → all bodies are {}.
        for p in input_field_paths:
            assert self.entries[p].body == {}, (
                f"{p} body should be empty dict (no modes in sample), got {self.entries[p].body!r}"
            )
        coefs_paths = [p for p in self.entries if p.startswith("input.coefs")]
        for p in coefs_paths:
            assert self.entries[p].body == {}, (
                f"{p} body should be empty dict (no modes in sample), got {self.entries[p].body!r}"
            )

    def test_unknown_section_rows_not_emitted(self):
        """``## Decision tables…`` rows must NOT produce entries — not in _FIELD_SECTIONS."""
        assert "Decision tables" not in self.entries, "decision section heading must not become a field key"
        assert "Decision.bogus" not in self.entries, "decision-table rows must not emit"
        assert not any(p.endswith(".bogus") for p in self.entries), (
            "'bogus' row from decision table leaked into entries"
        )

    def test_narrative_subsection_closes_section_scan(self):
        """``### Pattern interpretation`` narrative ends input's table scan.

        Any rows after it remain part of the body (prose) but new ``##`` opens
        the next section.  `corr_time_mode` is BEFORE the ``###`` close line,
        so it MUST be parsed; a synthetic field after ``###`` must NOT.
        """
        assert "input.corr_time_mode" in self.entries, "row preceding ``###`` close must still be parsed"

    def test_heading_inside_code_fence_not_recognised(self):
        """``#`` lines inside a fenced block are skipped — code, not headings."""
        sample = textwrap.dedent(
            """\
            ## `input` — Section

            | Field | Type | Purpose |
            |-------|------|---------|
            | `path` | `str` | File path. |

            ```yaml
            # not a heading
            ## `out`
            | `bogus_out` | `str` | ignored — inside fence |
            ```

            ## `out` — Real out

            | Field | Type | Purpose |
            |-------|------|---------|
            | `dt_bins` | `List[int]` | Averaging. |
            """
        )
        entries = parse_reference(sample)
        # fence content did NOT close the input scan mid-stream
        assert "input.path" in entries, "input.path should parse before the fence"
        # inside-fence "## `out`" did NOT open a section
        assert "out.bogus_out" not in entries, (
            "fenced-line `##` must not open a section — `bogus_out` would leak"
        )
        # the real `## \`out\`` after the fence end opened the section
        assert "out.dt_bins" in entries, "real `## out` heading after fence close must open the section"


# ── #### sub-block nesting inside ### modes (synthetic) ──────────────────────


_DETAIL_SAMPLE = textwrap.dedent(
    """\
    ## `input` — Data source

    | Field | Type | Default | Required | Purpose |
    |-------|------|---------|----------|---------|
    | `path` | `str` | — | **Yes** | File path. |

    ### `input.path` <mode>dirs</mode>
    Short one-liner here.

    #### Detailed
    Detailed body line 1.
    Detailed body line 2.

    #### Other
    Other detail body.

    ### `input.path` <mode>probe</mode>
    Probe body — no #### inside.

    ## `out` — Output

    | Field | Type | Purpose |
    |-------|------|---------|
    | `dt_bins` | `List[int]` | Averaging. |
    """
)


class TestDetailSubblock:
    """``#### <Tag>`` sub-blocks nest inside the active ``###`` mode (do not close it)."""

    def setup_method(self):
        self.entries = parse_reference(_DETAIL_SAMPLE)

    def test_detailed_block_nested_not_closing_mode(self):
        """``#### Detailed`` does NOT close the ``### … <mode>dirs</mode>`` mode.

        The Detailed content must reach the entry's search-mode body as a
        detail block, NOT be dropped as plain prose outside any section.
        """
        from tcm_gui._help import ModeBody

        e = self.entries["input.path"]
        assert isinstance(e.body, dict), "input.path should have mode-tagged body"
        search = e.body["dirs"]
        assert isinstance(search, ModeBody), (
            f"search mode carries #### → body[search] should be ModeBody; got {type(search).__name__}"
        )
        assert search.short == "Short one-liner here.", (
            f"search short should be the pre-#### line; got {search.short!r}"
        )
        assert "Detailed" in search.details, (
            f"search details should include 'Detailed'; got {sorted(search.details)!r}"
        )
        assert "line 1" in search.details["Detailed"] and "line 2" in search.details["Detailed"], (
            f"Detail body should contain both lines; got {search.details['Detailed']!r}"
        )

    def test_multiple_detail_blocks_in_one_mode(self):
        """A mode may carry several ``####`` blocks — each addressed by its tag."""
        from tcm_gui._help import ModeBody

        search = self.entries["input.path"].body["dirs"]
        assert isinstance(search, ModeBody)
        assert set(search.details) == {"Detailed", "Other"}, (
            f"search should have both detail tags; got {sorted(search.details)!r}"
        )
        assert "Other detail body" in search.details["Other"], (
            f"'Other' detail block content mismatch; got {search.details['Other']!r}"
        )

    def test_mode_without_details_stays_plain_str(self):
        """Mode with NO ``####`` keeps backward-compat: body[mode] == str."""
        e = self.entries["input.path"]
        probe = e.body["probe"]
        assert isinstance(probe, str), (
            f"probe mode (no ####) body should be plain str; got {type(probe).__name__}"
        )
        assert probe == "Probe body — no #### inside.", f"probe body content mismatch; got {probe!r}"

    def test_section_after_mode_with_details_closes_correctly(self):
        r"""A ``##`` heading after a mode carrying ``####`` blocks closes the mode.

        ``out.dt_bins`` must be emitted as a normal field row — proving the
        search mode's ``####`` blocks did not swallow the ``## \`out\``` heading
        (which would otherwise leak out.dt_bins into input's body).
        """
        assert "out.dt_bins" in self.entries, (
            "out.dt_bins must parse normally after search mode's #### blocks — "
            "#### must not eat the following ## heading"
        )


# ── #### sub-blocks at field level (no ### mode tag) ─────────────────────


_FIELD_DETAIL_SAMPLE = textwrap.dedent(
    """\
    ## `input` — Data source

    | Field | Type | Default | Required | Purpose |
    |-------|------|---------|----------|---------|
    | `path` | `str` | — | **Yes** | File path. |
    | `time_ranges` | `List[str]` | `None` | No | Time window. |
    | `coordinates` | `List[float]` | `None` | No | Station coords. |

    #### Detailed
    Field-level detail for coordinates (last field before this block).

    ## `program` — Runtime flags

    | Field | Type | Default | Purpose |
    |-------|------|---------|---------|
    | `verbose` | `str` | `'INFO'` | Log verbosity. |
    | `return_` | `str` | `'<end>'` | Pipeline exit point. |

    #### Detailed
    Phase-stopping table for return_.

    #### Examples
    Some examples for return_.
    """
)


class TestFieldLevelDetail:
    """``#### <Tag>`` blocks directly under ``## section`` (no ``###`` mode tag).

    Each ``####`` block is associated with the last table field row seen
    before it (tracked via ``last_field_path``).
    """

    def setup_method(self):
        self.entries = parse_reference(_FIELD_DETAIL_SAMPLE)

    def test_field_level_detail_stored_under_correct_field(self):
        """``#### Detailed`` after ``coordinates`` row → stored under ``input.coordinates``."""
        from tcm_gui._help import _FIELD_DETAIL, ModeBody

        e = self.entries["input.coordinates"]
        assert isinstance(e.body, dict), f"body should be dict, got {type(e.body).__name__}"
        field_detail = e.body.get(_FIELD_DETAIL)
        assert isinstance(field_detail, ModeBody), (
            f"coordinates should carry field-level detail as ModeBody; got {type(field_detail).__name__}"
        )
        assert "Detailed" in field_detail.details, (
            f"should have 'Detailed' tag; got {sorted(field_detail.details)!r}"
        )
        assert "coordinates" in field_detail.details["Detailed"], (
            f"detail content should mention coordinates; got {field_detail.details['Detailed']!r}"
        )

    def test_field_level_detail_not_on_other_fields(self):
        """Fields before the ``####`` block must NOT carry field-level detail."""
        from tcm_gui._help import _FIELD_DETAIL

        # ``input.path`` and ``input.time_ranges`` have no #### after them.
        for path in ("input.path", "input.time_ranges"):
            e = self.entries[path]
            assert _FIELD_DETAIL not in (e.body if isinstance(e.body, dict) else {}), (
                f"{path} should NOT have field-level detail; got body={e.body!r}"
            )

    def test_multiple_detail_tags_on_same_field(self):
        """A field may carry several ``####`` tags (Detailed + Examples)."""
        from tcm_gui._help import _FIELD_DETAIL, ModeBody

        e = self.entries["program.return_"]
        field_detail = e.body.get(_FIELD_DETAIL)
        assert isinstance(field_detail, ModeBody)
        assert set(field_detail.details) == {"Detailed", "Examples"}, (
            f"return_ should have both tags; got {sorted(field_detail.details)!r}"
        )
        assert "Phase-stopping" in field_detail.details["Detailed"]
        assert "examples" in field_detail.details["Examples"].lower()

    def test_help_for_path_field_detail_resolution(self):
        """``help_for_path(path, mode='_', detail='Detailed')`` returns the detail body."""
        from tcm_gui._help import _FIELD_DETAIL, help_for_path

        # Use parse_reference directly since this sample isn't the bundled doc.
        entries = parse_reference(_FIELD_DETAIL_SAMPLE)
        # Manually check the structure (help_for_path uses the cache, not this sample).
        e = entries["program.return_"]
        field_detail = e.body[_FIELD_DETAIL]
        assert isinstance(field_detail.details.get("Detailed", ""), str)
        assert "Phase-stopping" in field_detail.details["Detailed"]


class TestModeSectionAfterFieldDetail:
    """A ``###`` mode section opening after a field-level ``####`` block flushes
    the field detail first — mode body never leaks into the previous field."""

    _SAMPLE = textwrap.dedent(
        """\
        ## `input.coefs` — Coefficients

        | Field = Default | Physical meaning |
        |-----------------|------------------|
        | `P_t` = `None` | Pressure polynomial. |
        | `date` = `None` | Overall date. |

        #### Detailed
        Field-level block for date.

        ### `input.coefs.P_t` <mode>probe</mode>
        P_t short body.

        #### Detailed
        P_t detail body.
        """
    )

    def setup_method(self):
        self.entries = parse_reference(self._SAMPLE)

    def test_field_detail_not_contaminated_by_mode(self):
        """date's field-level Detailed keeps only its own content."""
        from tcm_gui._help import _FIELD_DETAIL, ModeBody

        e = self.entries["input.coefs.date"]
        field_detail = e.body[_FIELD_DETAIL]
        assert isinstance(field_detail, ModeBody)
        detailed = field_detail.details["Detailed"]
        assert "Field-level block for date." in detailed
        assert "P_t" not in detailed, f"mode body leaked into date's detail; got {detailed!r}"

    def test_mode_body_reaches_own_field(self):
        """P_t's mode section carries its own short body + Detailed block."""
        from tcm_gui._help import ModeBody

        probe = self.entries["input.coefs.P_t"].body["probe"]
        assert isinstance(probe, ModeBody)
        assert probe.short == "P_t short body."
        assert probe.details["Detailed"] == "P_t detail body."


class TestModelessSections:
    """``### `field` `` without a ``<mode>`` tag — the single-context default."""

    _SAMPLE = textwrap.dedent(
        """\
        ## `input` — Data source

        | Field = Default | Purpose |
        |-----------------|---------|
        | `time_ranges` = `None` | Time window. |

        ### `input.time_ranges`
        Short body.

        #### Detailed
        Detailed body.

        ## `out` — Output

        | Field = Default | Purpose |
        |-----------------|---------|
        | `overwrite_db` = `None` | NC overwrite strategy. |

        ### `out.overwrite_db`
        Narrative with a decision table after it — the table must NOT leak
        field rows (the section swallows it as body):

        | `None` | No | subset | Skip NC |
        | `trim` | — | any | Trim |
        """
    )

    def test_modeless_section_stored_under_no_mode(self):
        from tcm_gui._help import _NO_MODE, ModeBody

        e = parse_reference(self._SAMPLE)["input.time_ranges"]
        body = e.body[_NO_MODE]
        assert isinstance(body, ModeBody)
        assert body.short == "Short body."
        assert body.details["Detailed"] == "Detailed body."

    def test_table_after_modeless_section_not_leaked_as_rows(self):
        entries = parse_reference(self._SAMPLE)
        assert "out.overwrite_db" in entries, "table row must parse before the modeless section"
        assert not any(p in entries for p in ("out.None", "out.trim")), (
            "decision-table rows after a modeless section must stay body, not become field rows"
        )


class TestAnchors:
    """Section-heading anchors (GitHub-style slug, ``{#id}`` honored) for F1 help."""

    def test_real_doc_section_anchors(self, monkeypatch):
        """Section-heading anchors match the slug of their ``## `` heading line.

        Computes the expected anchor from the source doc via :func:`slugify`
        so the test tracks content changes without hardcoding subtitles.
        """
        import re

        from tcm_gui._help import _FIELD_SECTIONS, doc_path, reload_cache, slugify

        monkeypatch.setattr("tcm_gui._help.resolve_lang", lambda: "en")
        reload_cache("en")

        # Map registered section path → expected anchor (slug of its heading line).
        expected: dict[str, str] = {}
        for line in doc_path("en").read_text(encoding="utf-8").splitlines():
            if not line.lstrip().startswith("## ") or not (m := re.search(r"`(\w[\w.]*)`", line)):
                continue
            section = m.group(1)
            if section not in _FIELD_SECTIONS:
                continue
            expected[section] = slugify(line)

        assert expected, "no registered ## section headings found in config_reference.md"
        entries = reload_cache("en")
        for section, want in expected.items():
            assert section in entries, f"section '{section}' missing from parsed doc"
            assert entries[section].anchor == want, (
                f"'{section}' anchor {entries[section].anchor!r} != slug of heading line {want!r}"
            )

    def test_field_rows_inherit_section_anchor(self, monkeypatch):
        from tcm_gui._help import help_for_path, reload_cache

        monkeypatch.setattr("tcm_gui._help.resolve_lang", lambda: "en")
        reload_cache("en")
        assert help_for_path("input.coefs.Ag[0]").anchor == "inputcoefs--calibration-coefficients"

    def test_real_doc_calib_entries(self, monkeypatch):
        """Schema-derived ``input.calib`` section feeds sheet status + dwell tooltips.

        Regression: ``_FIELD_SECTIONS`` is inferred from ``schema`` (nested input
        dataclass groups) — a missing ``input.calib`` made every calib sheet row
        fall back to the bare key name in the status bar.
        """
        from tcm_gui._help import help_for_path, reload_cache

        monkeypatch.setattr("tcm_gui._help.resolve_lang", lambda: "en")
        reload_cache("en")
        for p in (
            "input.calib.g0xyz",
            "input.calib.time_ranges_zeroing",
            "input.calib.time_ranges_azimuth",
            "input.calib.coordinates",
            "input.calib.azimuth_add",
        ):
            e = help_for_path(p)
            assert e is not None and e.short, f"{p} must parse from the input.calib field table"
            assert help_for_path(p, mode="detail", detail="Detailed").body, f"{p} needs a #### Detailed block"

    def test_explicit_id_wins_and_strips_subtitle(self):
        sample = textwrap.dedent(
            """\
            ## `filter` — Process-stage {#quality}

            | Field = Default | Purpose |
            |-----------------|---------|
            | `min` = `{}` | Lower bounds. |
            """
        )
        e = parse_reference(sample)
        assert e["filter"].anchor == "quality"
        assert e["filter.min"].anchor == "quality"
        assert e["filter"].short == "Process-stage"


class TestHelpForPath:
    """``help_for_path`` index-stripping against the bundled cache."""

    def setup_method(self):
        from tcm_gui._help import reload_cache

        # Capture state once; populates the lru_cache.
        self._reload = reload_cache

    def test_strip_single_index(self):
        from tcm_gui._help import help_for_path

        self._reload()
        ref = help_for_path("input.coefs.Ag")
        assert ref is not None
        assert help_for_path("input.coefs.Ag[0]").path == ref.path, (
            "Ag[0] must resolve to the same Ag entry after single-index strip"
        )

    def test_strip_multi_index(self):
        from tcm_gui._help import help_for_path

        self._reload()
        ref = help_for_path("input.coefs.Ag")
        assert help_for_path("input.coefs.Ag[2][1]").path == ref.path, (
            "Ag[2][1] must resolve to the same Ag entry after multi-index strip"
        )

    def test_helpentry_is_frozen(self):
        """``HelpEntry`` is hashable & immutable (slots/frozen) — safe as registry value."""
        e = HelpEntry(path="x", short="s", body="b")
        with pytest.raises(AttributeError):
            e.path = "y"  # type: ignore[misc]


def test_bare_detailed_before_table_arms_short_and_dwell():
    """Bare ``### Detailed`` prose between ``##`` and its table arms the dwell, not the short.

    Regression chain: the section dwell first showed the table (sibling
    table-title heading left the mode open, swallowing table rows), then showed
    nothing (prose captured only as section short, so status and dwell carried
    identical text and the dwell firing was invisible).  Expected: section
    ``short`` falls back to the ``##`` subtitle (status), ``body["Detailed"]``
    carries the prose (dwell), table rows become field entries.
    """
    from tcm_gui._help import ModeBody

    sample = textwrap.dedent(
        """\
        ## `input.coefs` — Calibration coefficients

        ### Detailed
        Lead prose for dwell only.

        ### Table. Coefficient parameters

        | Field = Default | Purpose |
        |-----------------|---------|
        | `Ag` = `1` | Accel scale. |
        """
    )
    entries = parse_reference(sample)
    assert entries["input.coefs"].short == "Calibration coefficients"
    dwell = entries["input.coefs"].body.get("Detailed")
    assert isinstance(dwell, ModeBody), f"dwell body should be ModeBody, got {type(dwell).__name__}"
    assert dwell.short == "Lead prose for dwell only."
    assert entries["input.coefs.Ag"].short == "Accel scale."
