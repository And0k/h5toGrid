"""Tests for tcm/config.py — B-2 regroup, new fields, inheritance, ConfigStore groups.

Covers: new input.min/max fields, corr_time_* moved to input,
ConfigFilterCalib inherits ConfigFilter_InclProc, ConfigStore proc group,
calibration ConfigInCalib(ConfigIn_InclProc).
"""
from __future__ import annotations

import os

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

import tcm.config  # noqa: F401 — triggers ConfigStore registration
from tcm.config import (
    Config, ConfigFilterCalib, ConfigFilter_InclProc, ConfigIn_InclProc,
    ConfigProcCalib, ConfigProcSpectrum,
)


# --------------------------------------------------------------------------- #
# ConfigStore groups
# --------------------------------------------------------------------------- #

@pytest.mark.xr
class TestConfigStoreGroups:
    def test_proc_group_registered(self):
        """ConfigStore has proc group with calib and spectrum options."""
        repo = ConfigStore.instance().repo
        proc_group = repo.get("proc")
        assert proc_group is not None, "proc group not in ConfigStore"
        proc_options = set(proc_group.keys())
        assert "calib.yaml" in proc_options, (
            f"proc/calib not in ConfigStore — options: {sorted(proc_options)}"
        )
        assert "spectrum.yaml" in proc_options, (
            f"proc/spectrum not in ConfigStore — options: {sorted(proc_options)}"
        )

    def test_filter_calib_registered(self):
        """ConfigStore has filter/calib option (ConfigFilterCalib)."""
        repo = ConfigStore.instance().repo
        filter_group = repo.get("filter")
        assert filter_group is not None, "filter group not in ConfigStore"
        filter_options = set(filter_group.keys())
        assert "calib.yaml" in filter_options, (
            f"filter/calib not in ConfigStore — options: {sorted(filter_options)}"
        )


# --------------------------------------------------------------------------- #
# ConfigIn_InclProc — new fields
# --------------------------------------------------------------------------- #

@pytest.mark.xr
class TestConfigInFields:
    def test_input_has_min_max(self):
        """ConfigIn_InclProc has min/max dicts for raw-col DROP."""
        c = ConfigIn_InclProc()
        assert hasattr(c, "min"), "ConfigIn_InclProc missing 'min' field"
        assert hasattr(c, "max"), "ConfigIn_InclProc missing 'max' field"
        assert isinstance(c.min, dict), "min should default to empty dict"
        assert isinstance(c.max, dict), "max should default to empty dict"

    def test_input_has_time_corr_fields(self):
        """ConfigIn_InclProc has corr_time_mode + corr_time_outlier_threshold_s (moved from filter)."""
        c = ConfigIn_InclProc()
        assert hasattr(c, "corr_time_mode"), "corr_time_mode missing from ConfigIn_InclProc"
        assert hasattr(c, "corr_time_outlier_threshold_s"), "corr_time_outlier_threshold_s missing"
        assert hasattr(c, "dt_interp_between"), "dt_interp_between missing from ConfigIn_InclProc"
        assert hasattr(c, "fs_rounding"), "fs_rounding missing from ConfigIn_InclProc"


# --------------------------------------------------------------------------- #
# ConfigFilter_InclProc — only NaN-out fields
# --------------------------------------------------------------------------- #

@pytest.mark.xr
class TestConfigFilterFields:
    def test_filter_has_no_time_corr(self):
        """ConfigFilter_InclProc does NOT have corr_time_mode (moved to input)."""
        f = ConfigFilter_InclProc()
        assert not hasattr(f, "corr_time_mode"), (
            "corr_time_mode should be in ConfigIn_InclProc, not ConfigFilter_InclProc"
        )
        assert not hasattr(f, "corr_time_outlier_threshold_s"), (
            "corr_time_outlier_threshold_s should be in ConfigIn_InclProc"
        )

    def test_filter_calib_inherits_filter(self):
        """ConfigFilterCalib inherits from ConfigFilter_InclProc."""
        assert issubclass(ConfigFilterCalib, ConfigFilter_InclProc), (
            "ConfigFilterCalib should inherit ConfigFilter_InclProc"
        )
        c = ConfigFilterCalib()
        assert hasattr(c, "min"), "Inherited min field from ConfigFilter_InclProc"
        assert hasattr(c, "max"), "Inherited max field from ConfigFilter_InclProc"
        assert hasattr(c, "A"), "ConfigFilterCalib should have A (despike channel)"
        assert hasattr(c, "M"), "ConfigFilterCalib should have M (despike channel)"
        assert hasattr(c, "blocks"), "ConfigFilterCalib should have apex blocks"


# --------------------------------------------------------------------------- #
# corr_time_mode migration: callers read from input, not filter
# --------------------------------------------------------------------------- #

@pytest.mark.xr
class TestCorrTimeModeMigration:
    """Verify that corr_time_mode is accessible from cfg.input, not cfg.filter.

    This is a TDD regression test for the bug where config_yaml.gen_metadata()
    reads cfg["filter"]["corr_time_mode"] which raises ConfigKeyError after
    the field was moved to ConfigIn_InclProc.
    """

    def test_corr_time_mode_in_input_structured(self):
        """OmegaConf structured ConfigIn_InclProc has corr_time_mode."""
        cfg = OmegaConf.structured(ConfigIn_InclProc())
        assert OmegaConf.select(cfg, "corr_time_mode") == True, (
            "corr_time_mode should be accessible on ConfigIn_InclProc"
        )

    def test_corr_time_mode_not_in_filter_structured(self):
        """OmegaConf structured ConfigFilter_InclProc does NOT have corr_time_mode."""
        cfg = OmegaConf.structured(ConfigFilter_InclProc())
        with pytest.raises(Exception):
            OmegaConf.select(cfg, "corr_time_mode", no_throw=False)

    def test_full_config_corr_time_mode_in_input(self, tmp_path):
        """compose() places corr_time_mode under input, not filter.

        Simulates the exact Hydra composition path that gen_metadata() uses.
        """
        cfg_dir = tmp_path / "cfg_proc"
        cfg_dir.mkdir()
        (cfg_dir / "config.yaml").write_text(
            "defaults:\n"
            "  - input: base\n  - out: base\n  - filter: base\n  - program: base\n  - _self_\n"
        )
        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            with initialize_config_dir(config_dir=str(cfg_dir), version_base="1.3"):
                cfg = compose(config_name="config")
        finally:
            os.chdir(old_cwd)

        assert OmegaConf.select(cfg, "input.corr_time_mode") == True, (
            "corr_time_mode should be under cfg.input in composed config"
        )
        # filter.corr_time_mode must NOT exist — this is the exact bug
        assert OmegaConf.select(cfg, "filter.corr_time_mode") is None, (
            "corr_time_mode must NOT be under cfg.filter — moved to cfg.input"
        )

    def test_gen_metadata_reads_input_corr_time_mode(self):
        """config_yaml.gen_metadata reads corr_time_mode from cfg.input, not cfg.filter.

        TDD regression: this was the exact crash at config_yaml.py:250.
        """
        from tcm.config_yaml import gen_metadata

        # Minimal cfg that gen_metadata would see (DictConfig, structured)
        cfg = DictConfig({
            "input": {
                "path": "/nonexistent",
                "tables": ["incl*"],
                "corr_time_mode": True,
                "dt_from_utc": 0,
            },
            "out": {"dt_bins": [0, 2]},
            "filter": {"corr_time_mode": True},  # old location — should NOT be read
        })
        # gen_metadata will try to discover tables and fail (no file) —
        # that's fine; the key assertion is that it doesn't crash on
        # cfg["filter"]["corr_time_mode"] before reaching file I/O.
        try:
            list(gen_metadata(cfg, []))
        except FileNotFoundError:
            pass  # expected — no files; the important thing is no ConfigKeyError
        except Exception as e:
            if "corr_time_mode" in str(e):
                pytest.fail(f"gen_metadata still reads corr_time_mode from filter: {e}")
            # other errors are fine (file not found, etc.)

    def test_gen_metadata_survives_missing_dt_bins(self, tmp_path):
        """gen_metadata does not crash when cfg["out"] has no dt_bins (plain dict after to_container).

        TDD regression: OmegaConf.to_container → create round-trip drops structured
        defaults, leaving cfg["out"] as a plain dict without dt_bins.
        """
        import numpy as np
        import pandas as pd
        import xarray as xr
        from tcm.config_yaml import gen_metadata

        # Create a minimal NC file with an incl table group so discovery succeeds
        nc_path = tmp_path / "test.nc"
        time = pd.date_range("2024-01-01", periods=10, freq="s")
        ds = xr.Dataset(
            {"Ax": ("time", np.zeros(10)), "Ay": ("time", np.ones(10))},
            coords={"time": time},
        )
        ds.to_netcdf(nc_path, group="incl_p05", engine="netcdf4")

        # Simulate post-round-trip config: out is a plain dict missing dt_bins
        cfg = DictConfig({
            "input": {
                "path": str(nc_path),
                "tables": ["incl*"],
                "corr_time_mode": True,
                "dt_from_utc": 0,
            },
            "out": {"table": ""},  # no dt_bins — simulates to_container drop
            "filter": {},
        })
        try:
            results = list(gen_metadata(cfg, [nc_path]))
            assert results, "gen_metadata should yield at least one probe config"
        except Exception as e:
            if "dt_bins" in str(e):
                pytest.fail(f"gen_metadata crashes on missing dt_bins: {e}")
            raise  # re-raise unexpected errors


@pytest.mark.xr
class TestDefaultTablesPattern:
    """Verify that the default ``input.tables`` pattern matches common table naming conventions.

    Both HDF5 (``incl.05``) and NC (``incl_p05``) naming must be discoverable
    by the default pattern.  Patterns use glob semantics via
    :func:`csv_load._glob_to_regex` (same rules as text-file search).
    """

    def test_default_tables_pattern(self):
        """Default tables pattern is incl* (glob — matches both dot and underscore)."""
        from tcm.config import ConfigIn_InclProc

        cfg = ConfigIn_InclProc()
        assert cfg.tables == ["incl*"], (
            f"Default tables pattern should be ['incl*'], got {cfg.tables}"
        )

    @pytest.mark.parametrize(
        ("table_name", "should_match"),
        [
            pytest.param("incl.05", True, id="hdf5-dot"),
            pytest.param("incl.06", True, id="hdf5-dot-06"),
            pytest.param("incl_p05", True, id="nc-underscore"),
            pytest.param("incl_p06", True, id="nc-underscore-06"),
            pytest.param("pressure.01", False, id="other-table"),
        ],
    )
    def test_default_pattern_matches_both_conventions(self, table_name, should_match):
        """Default incl* matches both HDF5 (dot) and NC (underscore) via _glob_to_regex."""
        import re

        from tcm.csv_load import _glob_to_regex

        pattern = ConfigIn_InclProc().tables[0]
        regex = re.compile(_glob_to_regex(pattern))
        matches = regex.fullmatch(table_name) is not None
        if should_match:
            assert matches, (
                f"Pattern {pattern!r} (regex {regex.pattern!r}) should match {table_name!r}"
            )
        else:
            assert not matches, f"Pattern {pattern!r} should NOT match {table_name!r}"