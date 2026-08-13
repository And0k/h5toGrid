"""Tests for ``tcm_gui._help``: auto-extract config-cell help from ``config_reference.md``.

Cover the table-driven parser (``parse_reference``) and the index-stripping
resolver (``help_for_path``):
  * ``## ``section```` opens a config-group section (``input``/``out``/``filter``/
    ``program``/``input.coefs``); narrative ``###`` subsections close the scan.
  * Each ``| `field` | … | description |`` row inside a section emits
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

from tcm_gui._help import HelpEntry, _doc_path, parse_reference, reload_cache

_DOC_PATH = _doc_path()

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
        # ``input.path`` has ``<mode>probe</mode>`` and ``<mode>search</mode>`` sections.
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
        e = help_for_path("input.path", mode="probe")
        if e is None:
            pytest.skip("input.path not in bundled doc — version drift")
        assert isinstance(e.body, str), f"mode-resolved body should be str, got {type(e.body).__name__}"
        assert e.body, f"input.path probe body should be non-empty, got {e.body!r}"

    def test_section_level_entries_emitted(self, monkeypatch):
        """Section-level entries resolve for bare paths like ``input``, ``out``, ``filter``."""
        from tcm_gui._help import help_for_path, reload_cache

        monkeypatch.setattr("tcm_gui._help.resolve_lang", lambda: "en")
        reload_cache("en")
        for section, expected_substring in (
            ("input", "Data source"),
            ("input.coefs", "Calibration coefficients"),
            ("out", "Output configuration"),
            # Real doc subtitle: "Process-stage quality thresholds" — values exceeding
            # become NaN (body explains); subtitle was reworded from older "NaN-out".
            ("filter", "quality thresholds"),
            ("program", "Runtime flags"),
        ):
            e = help_for_path(section)
            assert e is not None, (
                f"section-level entry for '{section}' missing — needed for parent row hover in tksheet"
            )
            assert expected_substring in e.short, (
                f"'{section}' short should contain '{expected_substring}', got {e.short!r}"
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
    """Real-doc checks for the ``#### Detailed`` sub-block under ``input.path`` search mode."""

    def test_search_mode_detailed_block_reachable(self):
        """``help_for_path("input.path", mode="search", detail="Detailed")`` returns str."""
        from tcm_gui._help import help_for_path, reload_cache

        reload_cache()
        e = help_for_path("input.path", mode="search", detail="Detailed")
        if e is None:
            pytest.skip("input.path not in bundled doc — version drift")
        assert isinstance(e.body, str), f"detail-resolved body should be str, got {type(e.body).__name__}"
        assert e.body, (
            "input.path search #### Detailed body should be non-empty — "
            "did config_reference.md gain the Detailed sub-block?"
        )
        # Detailed content must mention the directory layout anchor, not the
        # one-line search summary that now lives in the short pre-#### body.
        assert "_raw" in e.body, f"Detailed body should mention _raw directory layout; got {e.body!r}"

    def test_search_mode_short_lines_excludes_detailed(self):
        """``mode="search"`` (no detail) returns SHORT lines only — pre-#### text."""
        from tcm_gui._help import help_for_path, reload_cache

        reload_cache()
        e = help_for_path("input.path", mode="search")
        if e is None:
            pytest.skip("input.path not in bundled doc — version drift")
        assert isinstance(e.body, str), f"mode-resolved short body should be str, got {type(e.body).__name__}"
        assert e.body, "search short body should be non-empty (pre-#### one-liner)"
        # The Detailed directory-layout text must NOT have leaked into the short body.
        assert "_raw" not in e.body, (
            f"short body should be the pre-#### one-liner, not the Detailed block; got {e.body!r}"
        )

    def test_search_mode_has_details_in_full_body(self):
        """Full ``help_for_path("input.path")`` body[search] is a _ModeBody (has details)."""
        from tcm_gui._help import _ModeBody, help_for_path, reload_cache

        reload_cache()
        e = help_for_path("input.path")
        if e is None:
            pytest.skip("input.path not in bundled doc — version drift")
        assert isinstance(e.body, dict), f"full body should be dict, got {type(e.body).__name__}"
        search = e.body.get("search")
        assert isinstance(search, _ModeBody), (
            "search mode now carries #### Detailed → body[search] should be _ModeBody; "
            f"got {type(search).__name__}"
        )
        assert search.has_details, f"search _ModeBody should have details, got {search.details!r}"
        assert "Detailed" in search.details, (
            f"search details should include 'Detailed' tag; got {sorted(search.details)!r}"
        )

    def test_detail_resolution_for_unknown_detail_returns_empty_str(self):
        """Unknown detail tag → empty str (caller no-ops), not None."""
        from tcm_gui._help import help_for_path, reload_cache

        reload_cache()
        e = help_for_path("input.path", mode="search", detail="Nonexistent")
        if e is None:
            pytest.skip("input.path not in bundled doc — version drift")
        assert isinstance(e.body, str), (
            f"unknown-detail body should be str (empty), got {type(e.body).__name__}"
        )
        assert e.body == "", f"unknown detail should resolve to '', got {e.body!r}"

    def test_probe_mode_without_detailed_stays_str(self):
        """Modes without ``####`` (probe) keep backward-compat: body == whole str."""
        from tcm_gui._help import _ModeBody, help_for_path, reload_cache

        reload_cache()
        e = help_for_path("input.path", mode="probe")
        if e is None:
            pytest.skip("input.path not in bundled doc — version drift")
        assert isinstance(e.body, str), f"probe body (no ####) should stay str, got {type(e.body).__name__}"
        # Sanity: _ModeBody must NOT be returned for modes without #### sub-blocks.
        assert not isinstance(e.body, _ModeBody), (
            "probe mode has no #### sub-blocks — body must be a plain str, not _ModeBody"
        )

    def test_per_lang_loader_fallback_to_en(self, monkeypatch, tmp_path):
        """Absent ``config_reference_ru.md`` → fallback to English ``config_reference.md``."""
        from tcm_gui import _help

        # Point DOC_DIR at a tmp dir with ONLY the English file.
        en = tmp_path / "config_reference.md"
        en.write_text(
            "## `input` — Data source\n\n| `path` | `str` | — | **Yes** | File path (en). |\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(_help, "DOC_DIR", tmp_path)
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

        en = tmp_path / "config_reference.md"
        en.write_text(
            "## `input` — Data source\n\n| `path` | `str` | — | **Yes** | English short. |\n",
            encoding="utf-8",
        )
        ru = tmp_path / "config_reference_ru.md"
        ru.write_text(
            "## `input` — Источник данных\n\n| `path` | `str` | — | **Да** | Русский short. |\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(_help, "DOC_DIR", tmp_path)
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
    # Config YAML Field Reference

    Intro prose — should NOT contribute any field entries.

    ## `input` — Data source & parameters

    Narrative for input — emitted as body to all input.* entries.

    | Field | Type | Default | Required | Purpose |
    |-------|------|---------|----------|---------|
    | `path` | `str` | — | **Yes** | File path, glob, or regex pattern (CLI input). |
    | `tables` | `List[str]` | `['incl*']` | No | HDF5 table names (regex allowed). |
    | `corr_time_mode` | `[bool, str, None]` | `True` | No | Time correction mode; see `\\|col\\|` shorthand. |

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
        """Section headings emit entries keyed by section name, subtitle as short."""
        # Sample contains input, input.coefs, out, filter sections.
        for section, expected_substring in (
            ("input", "Data source"),
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

    ### `input.path` <mode>search</mode>
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
        """``#### Detailed`` does NOT close the ``### … <mode>search</mode>`` mode.

        The Detailed content must reach the entry's search-mode body as a
        detail block, NOT be dropped as plain prose outside any section.
        """
        from tcm_gui._help import _ModeBody

        e = self.entries["input.path"]
        assert isinstance(e.body, dict), "input.path should have mode-tagged body"
        search = e.body["search"]
        assert isinstance(search, _ModeBody), (
            f"search mode carries #### → body[search] should be _ModeBody; got {type(search).__name__}"
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
        from tcm_gui._help import _ModeBody

        search = self.entries["input.path"].body["search"]
        assert isinstance(search, _ModeBody)
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
