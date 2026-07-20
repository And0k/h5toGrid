"""Roundtrip tests — write with _write_dataset_to_nc_group / store_processed,
read back with xr.open_dataset, verify values are identical.

Covers the h5py-only writer and xarray's to_netcdf writer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tcm._xr.storage import store_processed, _write_dataset_to_nc_group


def _make_ds(n: int = 50, freq: str = "100ms") -> xr.Dataset:
    """Build a synthetic Dataset with known time, 10 Hz by default."""
    time = pd.date_range("2024-06-15 12:00:00", periods=n, freq=freq).as_unit("ns")
    rng = np.random.default_rng(42)
    return xr.Dataset(
        {
            "Vabs": ("time", np.linspace(0.1, 1.0, n)),
            "Vdir": ("time", np.linspace(10.0, 350.0, n)),
            "inclination": ("time", rng.normal(10.0, 0.5, n)),
        },
        coords={"time": time},
    )


# ---------------------------------------------------------------------------
# _write_dataset_to_nc_group (h5py-only writer)
# ---------------------------------------------------------------------------


@pytest.mark.xr
class TestH5pyWriterRoundtrip:
    """Verify _write_dataset_to_nc_group preserves all values exactly."""

    def test_time_roundtrip(self, tmp_path):
        """Time values survive h5py write → xr.open_dataset read."""
        ds = _make_ds()
        nc_path = tmp_path / "test.nc"
        _write_dataset_to_nc_group(ds, nc_path, "tbl")

        with xr.open_dataset(nc_path, group="tbl", engine="netcdf4") as ds2:
            np.testing.assert_array_equal(
                ds2.time.values, ds.time.values,
                err_msg="Time values differ after h5py roundtrip",
            )
            assert ds2.time.dtype == ds.time.dtype, (
                f"Time dtype changed: {ds.time.dtype} -> {ds2.time.dtype}"
            )

    def test_data_roundtrip(self, tmp_path):
        """All data variables survive h5py write → xr.open_dataset read."""
        ds = _make_ds()
        nc_path = tmp_path / "test.nc"
        _write_dataset_to_nc_group(ds, nc_path, "tbl")

        with xr.open_dataset(nc_path, group="tbl", engine="netcdf4") as ds2:
            for name in ds.data_vars:
                np.testing.assert_array_almost_equal(
                    ds2[name].values, ds[name].values,
                    err_msg=f"Variable '{name}' differs after h5py roundtrip",
                )

    def test_10hz_frequency_preserved(self, tmp_path):
        """100ms intervals survive roundtrip (10 Hz data)."""
        ds = _make_ds(n=50, freq="100ms")
        nc_path = tmp_path / "test.nc"
        _write_dataset_to_nc_group(ds, nc_path, "tbl")

        with xr.open_dataset(nc_path, group="tbl", engine="netcdf4") as ds2:
            diffs = np.diff(ds2.time.values.astype(np.int64))
            expected_ns = 100_000_000  # 100ms in nanoseconds
            np.testing.assert_array_equal(
                diffs, np.full(len(diffs), expected_ns),
                err_msg="10Hz frequency not preserved after roundtrip",
            )

    def test_multiple_groups(self, tmp_path):
        """Multiple groups in same file are independent."""
        ds1 = _make_ds(n=10)
        ds2 = _make_ds(n=20)
        nc_path = tmp_path / "test.nc"
        _write_dataset_to_nc_group(ds1, nc_path, "g1")
        _write_dataset_to_nc_group(ds2, nc_path, "g2")

        with xr.open_dataset(nc_path, group="g1", engine="netcdf4") as r1:
            assert r1.sizes["time"] == 10
        with xr.open_dataset(nc_path, group="g2", engine="netcdf4") as r2:
            assert r2.sizes["time"] == 20

    def test_overwrite_existing_group(self, tmp_path):
        """Writing to same group twice overwrites cleanly."""
        ds1 = _make_ds(n=10)
        ds2 = _make_ds(n=30)
        nc_path = tmp_path / "test.nc"
        _write_dataset_to_nc_group(ds1, nc_path, "tbl")
        _write_dataset_to_nc_group(ds2, nc_path, "tbl")

        with xr.open_dataset(nc_path, group="tbl", engine="netcdf4") as r:
            assert r.sizes["time"] == 30


# ---------------------------------------------------------------------------
# store_processed (xarray's to_netcdf writer)
# ---------------------------------------------------------------------------


@pytest.mark.xr
class TestStoreProcessedRoundtrip:
    """Verify store_processed preserves all values exactly."""

    def test_time_roundtrip(self, tmp_path):
        """Time values survive xr.to_netcdf → xr.open_dataset."""
        ds = _make_ds()
        nc_path = tmp_path / "test.nc"
        store_processed(ds, nc_path, group="i01", mode="a")

        with xr.open_dataset(nc_path, group="i01", engine="netcdf4") as ds2:
            np.testing.assert_array_equal(
                ds2.time.values, ds.time.values,
                err_msg="Time values differ after store_processed roundtrip",
            )

    def test_10hz_frequency_preserved(self, tmp_path):
        """100ms intervals survive store_processed roundtrip."""
        ds = _make_ds(n=50, freq="100ms")
        nc_path = tmp_path / "test.nc"
        store_processed(ds, nc_path, group="i01", mode="a")

        with xr.open_dataset(nc_path, group="i01", engine="netcdf4") as ds2:
            diffs = np.diff(ds2.time.values.astype(np.int64))
            expected_ns = 100_000_000
            np.testing.assert_array_equal(
                diffs, np.full(len(diffs), expected_ns),
                err_msg="10Hz frequency not preserved after store_processed roundtrip",
            )

    def test_shared_file_multi_group(self, tmp_path):
        """Multiple probes write to same file, each with correct time."""
        ds1 = _make_ds(n=10, freq="100ms")
        ds2 = _make_ds(n=20, freq="1s")
        nc_path = tmp_path / "shared.nc"

        store_processed(ds1, nc_path, group="i01", mode="a")
        store_processed(ds2, nc_path, group="i02", mode="a")

        with xr.open_dataset(nc_path, group="i01", engine="netcdf4") as r1:
            assert r1.sizes["time"] == 10
            np.testing.assert_array_equal(r1.time.values, ds1.time.values)
        with xr.open_dataset(nc_path, group="i02", engine="netcdf4") as r2:
            assert r2.sizes["time"] == 20
            np.testing.assert_array_equal(r2.time.values, ds2.time.values)
