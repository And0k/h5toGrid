"""Headless regression for :mod:`tcm_gui._sheet_patch` — dataclass-typed generic patch.

No Tk: ``build_patch`` takes ``meta`` + ``cell_str`` directly, so int-strictness
(``out.dt_bins`` stays ``int``), calib write-back (``input.calib.*``) and
skip-registry (``input.path``/``input.coefs``/containers) pin here without a
Sheet. Tk-backed round-trips stay in ``test_full_mode.py``.
"""

from __future__ import annotations

from tcm import schema
from tcm_gui import _sheet_patch as sp

_CELLS = {
    "dt": ["0", "600"],
    "dt_bad": ["600.0"],
    "g0": ["1.5", "2.5", "3.5"],
    "az": ["5"],
    "trz": ["2026-01-01T00:00:00", "2026-01-02T00:00:00"],
    "tp": ["custom_out"],
    "bl": ["true"],
}
_META = {
    "dt": {"path": "out.dt_bins", "max_col": 6},
    "dt_bad": {"path": "out.dt_bins", "max_col": 1},
    "g0": {"path": "input.calib.g0xyz", "max_col": 3},
    "az": {"path": "input.calib.azimuth_add", "max_col": 1},
    "trz": {"path": "input.calib.time_ranges_zeroing", "max_col": 6, "is_string": True},
    "tp": {"path": "out.text_path", "max_col": 1, "is_string": True},
    "bl": {"path": "out.b_split_by_time_ranges", "max_col": 1},
    "skip_coefs": {"path": "input.coefs.Ag[0]", "max_col": 3},
    "skip_path": {"path": "input.path", "max_col": 1},
    "skip_cont": {"path": "filter", "max_col": 0},
}


def _rd(iid, j):
    row = _CELLS.get(iid, [])
    return row[j] if j < len(row) else ""


class TestSheetPatch:
    def test_int_list_stays_int(self):
        """``out.dt_bins: list[int]`` edits round-trip as ``int``, never ``float``."""
        patch = sp.build_patch({"dt": _META["dt"]}, _rd, 6, schema.Config, None, ("out",))
        assert patch["out"]["dt_bins"] == [0, 600], patch
        assert all(type(x) is int for x in patch["out"]["dt_bins"]), patch

    def test_strict_int_rejects_float_spelling(self):
        """``"600.0"`` for an int field drops the row (no silent ``600.0`` in YAML)."""
        assert sp.build_patch({"dt_bad": _META["dt_bad"]}, _rd, 6, schema.Config, None, None) == {}

    def test_calib_generic(self):
        """``input.calib`` vectors/scalar/date-lists go through the generic reader."""
        patch = sp.build_patch(
            {"g0": _META["g0"], "az": _META["az"], "trz": _META["trz"]}, _rd, 6, schema.Config, None, None
        )
        assert patch == {
            "input": {
                "calib": {
                    "g0xyz": [1.5, 2.5, 3.5],
                    "azimuth_add": 5.0,
                    "time_ranges_zeroing": ["2026-01-01T00:00:00", "2026-01-02T00:00:00"],
                }
            }
        }, patch

    def test_bool_scalar(self):
        patch = sp.build_patch({"bl": _META["bl"]}, _rd, 6, schema.Config, None, None)
        assert patch == {"out": {"b_split_by_time_ranges": True}}, patch

    def test_skips_dedicated_and_containers(self):
        """``input.path``/``input.coefs``/dict containers never enter the generic patch."""
        patch = sp.build_patch(
            {
                "skip_coefs": _META["skip_coefs"],
                "skip_path": _META["skip_path"],
                "skip_cont": _META["skip_cont"],
            },
            _rd,
            6,
            schema.Config,
            None,
            None,
        )
        assert patch == {}, patch

    def test_legacy_sections_filter_kept(self):
        """Default ``get_edited_full()`` sections still exclude ``input.*`` for old callers."""
        patch = sp.build_patch(_META, _rd, 6, schema.Config, None, ("out", "filter", "proc", "program"))
        assert "input" not in patch, patch
        assert patch["out"]["dt_bins"] == [0, 600], patch

    def test_parse_int_strict(self):
        assert sp.parse_int_strict("600") == 600
        assert sp.parse_int_strict(" -3 ") == -3
        assert sp.parse_int_strict("600.0") is None
        assert sp.parse_int_strict("1e3") is None
        assert sp.parse_int_strict("") is None

    def test_at_default_omitted(self):
        """Rows matching dataclass defaults stay out (minimal-override contract)."""
        dflt_cells = {"tp": ["text_output"]}
        rd = lambda iid, j: dflt_cells.get(iid, [""])[j] if j < len(dflt_cells.get(iid, [])) else ""
        assert sp.build_patch({"tp": _META["tp"]}, rd, 6, schema.Config, None, None) == {}


class TestDateRows:
    """``check: "sorted"`` rows validate format + write canonical ISO-T (``iso_secs``)."""

    def test_canonical_iso_t_written(self):
        """Space / dd.mm.yyyy spellings normalize to ISO-T; empty mid-row stays an open bound."""
        cells = {
            "tr": ["2026-01-02 03:04:05", "02.01.2026"],
            "tr2": ["2026-01-01T00:00:00", "", "2026-01-03T00:00:00"],
        }
        meta = {
            "tr": {"path": "input.time_ranges", "max_col": 6, "check": "sorted"},
            "tr2": {"path": "input.calib.time_ranges_azimuth", "max_col": 6, "check": "sorted"},
        }
        rd = lambda iid, j: cells.get(iid, [""])[j] if j < len(cells.get(iid, [])) else ""
        patch = sp.build_patch(meta, rd, 6, schema.Config, None, None)
        assert patch == {
            "input": {
                "time_ranges": ["2026-01-02T03:04:05", "2026-01-02T00:00:00"],
                "calib": {"time_ranges_azimuth": ["2026-01-01T00:00:00", "", "2026-01-03T00:00:00"]},
            }
        }, patch

    def test_unparseable_row_skipped(self):
        """A row holding an unparseable cell is dropped — the stored YAML value survives."""
        meta = {"tr": {"path": "input.time_ranges", "max_col": 6, "check": "sorted"}}
        rd = lambda iid, j: ["2026-01-02T00:00:00", "oops"][j] if j < 2 else ""
        assert sp.build_patch(meta, rd, 6, schema.Config, None, None) == {}
