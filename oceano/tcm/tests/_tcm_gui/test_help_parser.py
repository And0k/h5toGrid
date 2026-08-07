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

from tcm_gui._help import HelpEntry, parse_reference

# ── parser smoke (real config_reference.md) ─────────────────────────────────


class TestRealReference:
    """Smoke test against the actual bundled ``config_reference.md``."""

    def test_loads_without_error(self):
        """``reload_cache`` parses the real doc into a non-empty dict."""
        from tcm_gui._help import _DOC_PATH, reload_cache

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
        """Each entry's body carries the section's prose — non-empty for a documented field."""
        from tcm_gui._help import help_for_path, reload_cache

        reload_cache()
        e = help_for_path("input.corr_time_mode")
        if e is None:
            pytest.skip("input.corr_time_mode not in bundled doc — version drift")
        assert e.body, "corr_time_mode body should be section prose, got empty"
        assert e.short, "corr_time_mode short should be the row's last cell, got empty"

    def test_section_level_entries_emitted(self):
        """Section-level entries resolve for bare paths like ``input``, ``out``, ``filter``."""
        from tcm_gui._help import help_for_path, reload_cache

        reload_cache()
        for section, expected_substring in (
            ("input", "Data source"),
            ("input.coefs", "Calibration coefficients"),
            ("out", "Output configuration"),
            ("filter", "NaN-out"),  # "Process-stage NaN-out thresholds"
            ("program", "Runtime flags"),
        ):
            e = help_for_path(section)
            assert e is not None, (
                f"section-level entry for '{section}' missing — "
                f"needed for parent row hover in tksheet"
            )
            assert expected_substring in e.short, (
                f"'{section}' short should contain '{expected_substring}', got {e.short!r}"
            )

    def test_section_level_body_filled(self):
        """Section-level entries have non-empty body (section prose)."""
        from tcm_gui._help import help_for_path, reload_cache

        reload_cache()
        e = help_for_path("input.coefs")
        assert e is not None and e.body, (
            f"input.coefs section entry should have body (section prose), "
            f"got {e!r}"
        )


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
        """All entries of one section share a single body string (section prose).

        Different sections (``input`` vs ``input.coefs``) have distinct bodies —
        the parser captures one body per ``## ``section```` block, including a
        section-level entry (e.g. ``input.coefs`` itself).
        """
        # Filter pure ``input.*`` field entries — exclude ``input.coefs`` and
        # ``input.coefs.*`` (section entry has path without trailing dot).
        input_field_paths = {
            p for p in self.entries
            if p.startswith("input.") and not p.startswith("input.coefs") and p != "input"
        }
        input_bodies = {self.entries[p].body for p in input_field_paths}
        assert len(input_bodies) == 1, (
            f"all input.* field entries (not input.coefs.*) should share one body; "
            f"got {len(input_bodies)} distinct"
        )
        coefs_paths = {p for p in self.entries if p.startswith("input.coefs")}
        coefs_bodies = {self.entries[p].body for p in coefs_paths}
        assert len(coefs_bodies) == 1, (
            f"all input.coefs* entries (incl section-level) should share one body; "
            f"got {len(coefs_bodies)} distinct"
        )
        # Two distinct sections → two distinct bodies (each section has its own prose)
        assert next(iter(input_bodies)) != next(iter(coefs_bodies)), (
            "input.* and input.coefs.* are different sections — bodies must differ"
        )
        # Body includes the section heading line (captured from the ``## `` `` `` `` `` open)
        assert "## `input`" in next(iter(input_bodies)), (
            "input section body should include the `## `input` heading line"
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
