"""Tests for _xr/filters.py — centralised load/process filtering and sugar expansion.

Covers: filter_global_minmax (nested dict read), filter_local (nested dict read),
expand_m_shorthand, apply_load_time_ranges, warn_on_holes.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tcm._xr import filters
import tcm.cli

# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #

def _ds(n: int = 10, *, start: str = "2024-01-01") -> xr.Dataset:
    """Build minimal Dataset for filter tests."""
    time = pd.date_range(start, periods=n, freq="s")
    rng = np.random.default_rng(42)
    return xr.Dataset(
        {"Ax": ("time", rng.normal(0, 1, n)), "Ay": ("time", np.ones(n)),
         "Mx": ("time", np.full(n, 0.5)), "My": ("time", np.full(n, 3.0)),
         "Mz": ("time", np.full(n, 0.8))},
        coords={"time": time},
    )


# --------------------------------------------------------------------------- #
# filter_global_minmax — reads nested cfg['min']/cfg['max'] dicts
# --------------------------------------------------------------------------- #

@pytest.mark.xr
class TestFilterGlobalMinMax:
    def test_basic_drop_rows_outside_bounds(self):
        """filter_global_minmax drops rows where column value exceeds bounds."""
        ds = _ds(10)
        # All Mx=0.5 → drop rows where Mx > 0.3 should drop all rows
        cfg = {"max": {"Mx": 0.3}}
        result = filters.filter_global_minmax(ds, cfg)
        assert result.sizes["time"] == 0, (
            f"Expected 0 rows (all Mx=0.5 > max 0.3), got {result.sizes['time']}"
        )

    def test_noop_when_empty(self):
        """filter_global_minmax returns ds unchanged when min/max empty."""
        ds = _ds(5)
        result = filters.filter_global_minmax(ds, {"min": {}, "max": {}})
        assert result.sizes["time"] == 5, "No-op should preserve all rows"

    def test_none_cfg_returns_unchanged(self):
        """filter_global_minmax returns ds unchanged when cfg_in is None."""
        ds = _ds(5)
        result = filters.filter_global_minmax(ds, None)
        assert result.sizes["time"] == 5


# --------------------------------------------------------------------------- #
# filter_local — reads nested cfg_filter['min']/cfg_filter['max'] dicts
# --------------------------------------------------------------------------- #

@pytest.mark.xr
class TestFilterLocal:
    def test_nan_out_on_threshold(self):
        """filter_local NaN-out values exceeding threshold in nested dict."""
        ds = _ds(10)
        # All Mx=0.5, My=3.0 → NaN-out My where |My| > 2.0
        cfg = {"max": {"My": 2.0}}
        result = filters.filter_local(ds, cfg)
        assert result["My"].isnull().all(), (
            "My=3.0 should be NaN-out (|3.0| > 2.0 threshold)"
        )
        assert not result["Mx"].isnull().any(), "Mx=0.5 should NOT be NaN-out (< 2.0)"

    def test_noop_when_empty(self):
        """filter_local returns ds unchanged when cfg_filter min/max empty."""
        ds = _ds(5)
        result = filters.filter_local(ds, {"min": {}, "max": {}})
        for col in result.data_vars:
            assert not result[col].isnull().any(), f"{col} should have no NaN from no-op"

    def test_ignore_absent(self):
        """filter_local skips keys in ignore_absent when column is missing."""
        ds = _ds(5)
        cfg = {"max": {"g_minus_1": 0.5, "h_minus_1": 5.0}}
        # g_minus_1 and h_minus_1 are absent from ds — should be skipped silently
        result = filters.filter_local(ds, cfg, ignore_absent={"g_minus_1", "h_minus_1"})
        for col in result.data_vars:
            assert not result[col].isnull().any(), f"{col} should not be affected by absent keys"


# --------------------------------------------------------------------------- #
# expand_m_shorthand
# --------------------------------------------------------------------------- #

@pytest.mark.xr
class TestExpandMShorthand:
    def test_expands_m_to_xyz(self):
        """expand_m_shorthand expands 'M' to 'Mx','My','Mz' in-place."""
        cfg = {"min": {"M": 0.1}, "max": {"M": 5.0, "Ax": 10.0}}
        tcm.cli.sugar_expand_m(cfg)
        # 'M' should be replaced by Mx/My/Mz
        assert "M" not in cfg["min"], "Shorthand 'M' should be removed from min"
        assert cfg["min"]["Mx"] == 0.1, "Mx should inherit M value 0.1"
        assert cfg["min"]["My"] == 0.1
        assert cfg["min"]["Mz"] == 0.1
        assert "M" not in cfg["max"], "Shorthand 'M' should be removed from max"
        assert cfg["max"]["Mx"] == 5.0, "Mx should inherit M value 5.0"
        assert cfg["max"]["Ax"] == 10.0, "Ax should be preserved"

    def test_no_override_existing_keys(self):
        """expand_m_shorthand does not override existing Mx/My/Mz keys."""
        cfg = {"max": {"M": 5.0, "Mx": 3.0}}
        tcm.cli.sugar_expand_m(cfg)
        assert cfg["max"]["Mx"] == 3.0, "Existing Mx=3.0 should NOT be overridden by M=5.0"
        assert cfg["max"]["My"] == 5.0, "My absent → should inherit M=5.0"
        assert cfg["max"]["Mz"] == 5.0, "Mz absent → should inherit M=5.0"


# --------------------------------------------------------------------------- #
# apply_load_time_ranges
# --------------------------------------------------------------------------- #

@pytest.mark.xr
class TestApplyLoadTimeRanges:
    def test_slice_window(self):
        """apply_load_time_ranges drops rows outside [start, end]."""
        ds = _ds(10, start="2024-01-01")  # 10 seconds from 00:00:00
        # Window: 00:00:03 to 00:00:07 (4 rows: idx 3,4,5,6)
        result = filters.apply_load_time_ranges(ds, ["2024-01-01T00:00:03", "2024-01-01T00:00:07"])
        assert result.sizes["time"] == 5, (
            f"Expected 5 rows inside [03s, 07s], got {result.sizes['time']}"
        )

    def test_noop_when_none(self):
        """apply_load_time_ranges is no-op when time_ranges is None."""
        ds = _ds(5)
        result = filters.apply_load_time_ranges(ds, None)
        assert result.sizes["time"] == 5


# --------------------------------------------------------------------------- #
# warn_on_holes
# --------------------------------------------------------------------------- #

@pytest.mark.xr
class TestWarnOnHoles:
    @pytest.mark.parametrize(
        ("warning", "description"),
        [
            pytest.param(600, "int seconds", id="int"),
            pytest.param(timedelta(seconds=600), "timedelta (post-main_init)", id="timedelta"),
            pytest.param(600.0, "float seconds", id="float"),
        ],
    )
    def test_no_warning_when_no_holes(self, warning, description):
        """warn_on_holes accepts int, float, and timedelta threshold."""
        ds = _ds(10, start="2024-01-01")  # 1-second intervals
        filters.warn_on_holes(ds, dt_hole_warning=warning)

    def test_warn_on_large_gap(self):
        """warn_on_holes completes without error on large gap (warns internally)."""
        # Create ds with 601-second gap: 9 rows at 1s, then a 10th at 601s after 8th
        time = pd.date_range("2024-01-01", periods=9, freq="s").append(
            pd.DatetimeIndex(["2024-01-01 00:10:01"])
        )
        ds = xr.Dataset({"Ax": ("time", np.ones(10))}, coords={"time": time})
        # Should not raise — emits a warning via lf.warning (tested via smoke)
        filters.warn_on_holes(ds, dt_hole_warning=timedelta(seconds=600))

    @pytest.mark.parametrize(
        ("warning", "description"),
        [
            pytest.param(None, "None disables", id="none"),
            pytest.param(0, "zero disables (int)", id="zero-int"),
            pytest.param(-1, "negative disables (int)", id="neg-int"),
            pytest.param(timedelta(0), "zero timedelta disables", id="zero-td"),
            pytest.param(timedelta(seconds=-1), "negative timedelta disables", id="neg-td"),
        ],
    )
    def test_noop_when_disabled(self, warning, description):
        """warn_on_holes is no-op when dt_hole_warning is None, 0, or negative."""
        ds = _ds(5)
        filters.warn_on_holes(ds, dt_hole_warning=warning)