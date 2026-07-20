"""Tests for _xr/physical.py (binning, process), _xr/io.py, and _xr/dataset.py."""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from tcm._xr.calc import binning
import pytest
import xarray as xr

from tcm._xr import io as xr_io
from tcm._xr.coefs import save_coefs_to_nc
from tcm._xr.dataset import merge_probes, open_nc, open_csv
from tcm._xr.physical import process

_VELOCITY_COLS = ("Vabs", "Vdir", "v", "u", "inclination")
_RAW_COLS = ("Ax", "Ay", "Az", "Mx", "My", "Mz")


# --------------------------------------------------------------------------- #
# Binning
# --------------------------------------------------------------------------- #

@pytest.mark.xr
class TestBinning:
    def _make_ds(self, n=100, freq="s"):
        time = pd.date_range("2024-01-01", periods=n, freq=freq)
        rng = np.random.default_rng(42)
        return xr.Dataset(
            {"Vabs": ("time", rng.uniform(0, 1, n)), "Temp": ("time", 20 + rng.normal(0, 0.1, n))},
            coords={"time": time},
        )

    def test_no_binning(self):
        """dt_bin=0 returns dataset unchanged."""
        ds = self._make_ds()
        assert binning(ds, timedelta(0)) is ds

    def test_binning_reduces_size(self):
        """10-second binning of 100s data → ~10 bins."""
        result = binning(self._make_ds(n=100, freq="s"), timedelta(seconds=10))
        assert result is not None
        assert 9 <= result.sizes["time"] <= 11

    def test_binning_drops_sparse_bins(self):
        """Bins with too few valid samples are dropped."""
        ds = self._make_ds(n=100, freq="s")
        ds["Vabs"] = ds["Vabs"].where(ds["Vabs"] > 0.9)  # 90% → NaN
        result = binning(ds, timedelta(seconds=10), min_valid_fraction=0.1)
        assert result is not None and result.sizes["time"] < 10

    def test_binning_all_nan_returns_none(self):
        """All NaN → None."""
        ds = self._make_ds(n=10, freq="s")
        ds["Vabs"] = xr.DataArray(np.full(10, np.nan), dims="time")
        assert binning(ds, timedelta(seconds=5)) is None

    def test_binning_preserves_variables(self):
        """All data variables survive binning."""
        result = binning(self._make_ds(), timedelta(seconds=10))
        assert set(result.data_vars) == {"Vabs", "Temp"}


# --------------------------------------------------------------------------- #
# process
# --------------------------------------------------------------------------- #

@pytest.mark.xr
class TestToPhysical:
    def test_returns_list(self, sensor_ds, identity_coefs):
        """process returns a list of datasets."""
        result = process(sensor_ds, coefs=identity_coefs, dt_bins=[timedelta(0)])
        assert isinstance(result, list) and len(result) == 1
        assert result[0] is not None

    def test_no_avg_contains_velocity(self, sensor_ds, identity_coefs):
        """No-avg result contains velocity columns."""
        ds = process(sensor_ds, coefs=identity_coefs, dt_bins=[timedelta(0)])[0]
        assert set(_VELOCITY_COLS) <= ds.data_vars.keys()

    def test_no_avg_raw_columns_removed(self, sensor_ds, identity_coefs):
        """No-avg result has raw columns removed."""
        ds = process(sensor_ds, coefs=identity_coefs, dt_bins=[timedelta(0)])[0]
        assert not (set(_RAW_COLS) & ds.data_vars.keys())

    def test_binned_output(self, sensor_ds, identity_coefs):
        """Binning with dt > dt_min produces binned output."""
        result = process(
            sensor_ds, coefs=identity_coefs,
            dt_bins=[timedelta(seconds=5)], dt_min_binning_proc=timedelta(seconds=2),
        )
        assert len(result) == 1 and result[0] is not None
        assert result[0].sizes["time"] <= 10

    def test_no_avg_plus_binned(self, sensor_ds, identity_coefs):
        """Combined: no-avg + binned → 2 entries."""
        result = process(sensor_ds, coefs=identity_coefs, dt_bins=[timedelta(0), timedelta(seconds=5)])
        assert len(result) == 2
        assert result[0] is not None and result[1] is not None

    def test_pressure_in_output(self, sensor_ds_with_pressure, identity_coefs):
        """Pressure computed when P_t provided."""
        coefs = {**identity_coefs, "P_t": np.array([[2.0, 0.0], [0.0, 0.0]])}
        ds = process(sensor_ds_with_pressure, coefs=coefs, dt_bins=[timedelta(0)])[0]
        assert "Pressure" in ds


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #

@pytest.mark.xr
class TestIO:
    def test_csv_roundtrip(self, sensor_ds, tmp_path):
        """Dataset → CSV → Dataset roundtrip preserves data (within float format precision)."""
        path = tmp_path / "test.csv"
        xr_io.ds_to_csv(sensor_ds, path)
        result = xr_io.load_csv_as_ds(path)
        for var in sensor_ds.data_vars:
            if var == "axis":
                continue  # non-numeric, not roundtrippable
            assert var in result
            np.testing.assert_allclose(result[var].values, sensor_ds[var].values, rtol=1e-4, atol=1e-10)

    def test_netcdf_roundtrip(self, sensor_ds, tmp_path):
        """Dataset → netCDF → Dataset roundtrip preserves data."""
        path = tmp_path / "test.nc"
        xr_io.save_netcdf(sensor_ds, path)
        assert path.exists()
        result = xr_io.open_netcdf(path)
        for var in sensor_ds.data_vars:
            assert var in result
            np.testing.assert_allclose(result[var].values, sensor_ds[var].values, atol=1e-10)

    def test_csv_split_period(self, tmp_path):
        """split_period creates multiple files."""
        ds = xr.Dataset(
            {"Vabs": ("time", np.ones(30))},
            coords={"time": pd.date_range("2024-01-01", periods=30, freq="h")},
        )
        written = xr_io.ds_to_csv(ds, tmp_path / "test.csv", split_period="D")
        assert len(written) >= 2 and all(f.exists() for f in written)

    def test_csv_simple(self, sensor_ds, tmp_path):
        """Simple CSV write (no split)."""
        ds = sensor_ds.drop_vars("axis", errors="ignore")
        written = xr_io.ds_to_csv(ds, tmp_path / "test.csv")
        assert len(written) == 1 and written[0].exists()


# --------------------------------------------------------------------------- #
# merge_probes
# --------------------------------------------------------------------------- #

@pytest.mark.xr
class TestMergeProbes:
    def test_single_probe(self):
        """Single probe → probe dim of size 1."""
        time = pd.date_range("2024-01-01", periods=5, freq="s")
        ds = xr.Dataset({"Vabs": ("time", np.ones(5))}, coords={"time": time})
        result = merge_probes({"p01": ds})
        assert "probe" in result.dims and result.sizes["probe"] == 1
        np.testing.assert_array_equal(result.probe.values, ["p01"])

    def test_multiple_probes_outer_join(self):
        """Different-length time axes → outer join (NaN-padded)."""
        t1, t2 = pd.date_range("2024-01-01", periods=3, freq="s"), pd.date_range("2024-01-01", periods=5, freq="s")
        ds1 = xr.Dataset({"Vabs": ("time", [1.0, 2.0, 3.0])}, coords={"time": t1})
        ds2 = xr.Dataset({"Vabs": ("time", [10.0, 20.0, 30.0, 40.0, 50.0])}, coords={"time": t2})
        result = merge_probes({"a": ds1, "b": ds2})
        assert result.sizes["probe"] == 2
        assert result.sizes["time"] == 5
        assert np.isnan(result.Vabs.sel(probe="a").values[3])

    def test_preserves_variables(self):
        """All data variables survive merge."""
        time = pd.date_range("2024-01-01", periods=3, freq="s")
        ds = xr.Dataset(
            {"Vabs": ("time", [1.0, 2.0, 3.0]), "Vdir": ("time", [0.0, 90.0, 180.0])},
            coords={"time": time},
        )
        assert set(merge_probes({"p1": ds, "p2": ds}).data_vars) == {"Vabs", "Vdir"}

    def test_empty_raises(self):
        """Empty dict → ValueError."""
        with pytest.raises(ValueError, match="at least one"):
            merge_probes({})


# --------------------------------------------------------------------------- #
# open_csv
# --------------------------------------------------------------------------- #

@pytest.mark.xr
class TestOpenCsv:
    def test_calls_load_from_csv_gen(self, sensor_df, monkeypatch):
        """open_csv delegates to csv_load.load_from_csv_gen and converts to Dataset."""
        import tcm.csv_load as csv_load_mod
        meta = (1, "i_p01", Path("dummy.csv"))
        monkeypatch.setattr(csv_load_mod, "search_csv_files", lambda path: {("i", 1): [Path("dummy.csv")]})
        monkeypatch.setattr(csv_load_mod, "load_from_csv_gen", lambda **kw: iter([(sensor_df, meta)]))
        ds = open_csv(Path("_raw/*i*.txt"), text_type="i")
        assert isinstance(ds, xr.Dataset) and "time" in ds.coords
        assert ds.sizes["time"] == len(sensor_df)
        assert set(sensor_df.columns) <= ds.data_vars.keys()

    def test_concatenates_chunks(self, monkeypatch):
        """Multiple chunks from generator are concatenated."""
        import tcm.csv_load as csv_load_mod
        t1 = pd.date_range("2024-01-01", periods=3, freq="s", name="Time")
        t2 = pd.date_range("2024-01-01 00:00:03", periods=3, freq="s", name="Time")
        df1, df2 = pd.DataFrame({"Ax": [1.0, 2.0, 3.0]}, index=t1), pd.DataFrame({"Ax": [4.0, 5.0, 6.0]}, index=t2)
        meta = (1, "i_p01", Path("dummy.csv"))
        monkeypatch.setattr(csv_load_mod, "search_csv_files", lambda path: {("i", 1): [Path("dummy.csv")]})
        monkeypatch.setattr(csv_load_mod, "load_from_csv_gen", lambda **kw: iter([(df1, meta), (df2, meta)]))
        ds = open_csv(Path("_raw/*i*.txt"), text_type="i")
        assert ds.sizes["time"] == 6
        np.testing.assert_array_equal(ds.Ax.values, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    def test_empty_returns_empty_dataset(self, monkeypatch):
        """Generator yielding no data → empty Dataset."""
        import tcm.csv_load as csv_load_mod
        monkeypatch.setattr(csv_load_mod, "search_csv_files", lambda path: {("i", 1): [Path("dummy.csv")]})
        monkeypatch.setattr(csv_load_mod, "load_from_csv_gen", lambda **kw: iter([]))
        ds = open_csv(Path("_raw/*i*.txt"), text_type="i")
        assert isinstance(ds, xr.Dataset) and len(ds.data_vars) == 0


# --------------------------------------------------------------------------- #
# open_nc (from dataset_nc)
# --------------------------------------------------------------------------- #

@pytest.mark.xr
class TestOpenNc:
    """open_nc loads data and coefs from NetCDF4 files."""

    @pytest.fixture()
    def nc_data_file(self, tmp_path):
        """Create a .nc file with data in /incl_01/ group."""
        nc_path = tmp_path / "test.raw.nc"
        ds = xr.Dataset(
            {"Ax": ("time", np.random.default_rng(42).normal(0, 1, 100)),
             "Ay": ("time", np.random.default_rng(43).normal(0, 1, 100)),
             "Az": ("time", np.random.default_rng(44).normal(0, 1, 100))},
            coords={"time": pd.date_range("2024-01-01", periods=100, freq="s")},
        )
        ds.to_netcdf(nc_path, group="incl_01", engine="netcdf4")
        return nc_path

    @pytest.fixture()
    def nc_with_data_and_coefs(self, tmp_path):
        """Create a .nc file with data + coefs in /incl_01/ group."""
        nc_path = tmp_path / "test.raw.nc"
        ds = xr.Dataset(
            {"Ax": ("time", np.random.default_rng(42).normal(0, 1, 50)),
             "Mx": ("time", np.random.default_rng(43).normal(0, 1, 50))},
            coords={"time": pd.date_range("2024-01-01", periods=50, freq="s")},
        )
        ds.to_netcdf(nc_path, group="incl_01", engine="netcdf4")
        save_coefs_to_nc(nc_path, "incl_01", {"Ag": np.eye(3), "Cg": np.array([0.1, 0.2, 0.3]), "date": "2024-01-01T00:00:00"})
        return nc_path

    def test_loads_data_from_table_group(self, nc_data_file):
        """open_nc reads data variables from /{tbl}/ group."""
        ds, coefs = open_nc(nc_data_file, tbl="incl_01")
        assert isinstance(ds, xr.Dataset)
        assert {"Ax", "Ay", "Az"} <= ds.data_vars.keys()
        assert ds.sizes["time"] == 100

    def test_extracts_coefs_when_present(self, nc_with_data_and_coefs):
        """open_nc extracts coefs from /{tbl}/coef/ group."""
        _, coefs = open_nc(nc_with_data_and_coefs, tbl="incl_01")
        assert coefs is not None
        np.testing.assert_array_equal(coefs["Ag"], np.eye(3))
        np.testing.assert_array_equal(coefs["Cg"], [0.1, 0.2, 0.3])

    def test_returns_none_coefs_when_absent(self, nc_data_file):
        """open_nc returns None coefs when no /{tbl}/coef/ group."""
        assert open_nc(nc_data_file, tbl="incl_01")[1] is None

    def test_time_is_coordinate(self, nc_data_file):
        """Loaded Dataset has 'time' as a coordinate."""
        assert "time" in open_nc(nc_data_file, tbl="incl_01")[0].coords

    def test_chunk_time(self, nc_data_file):
        """chunk_time parameter chunks the time dimension."""
        ds, _ = open_nc(nc_data_file, tbl="incl_01", chunk_time=50)
        assert ds.chunks is not None and ds.chunks["time"] == (50, 50)
