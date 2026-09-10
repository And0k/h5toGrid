"""Tests for _xr/storage.py — netCDF persistence, groups, and incremental skip."""

from __future__ import annotations

import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
from tcm._xr.nc_utils import strip_tz_datetime
import xarray as xr

from tcm import _constants
from tcm._constants import _h5py
from tcm._xr.storage import (
    incremental_skip,
    open_processed_grouped,
    store_processed,
    store_raw,
)
from tcm._xr.coefs import prepare_coefs, coef_zeroing_rotation_from_data, coef_azimuth_from_data
from tcm.config_yaml import _discover_tables


def _sample_ds(n: int = 10, *, start: str = "2024-01-01") -> xr.Dataset:
    """Build a minimal Dataset for storage tests."""
    time = pd.date_range(start, periods=n, freq="s")
    return xr.Dataset(
        {"Ax": ("time", np.arange(n, dtype=float)), "Ay": ("time", np.ones(n))},
        coords={"time": time},
    )


@pytest.mark.xr
class TestStoreRaw:
    def test_creates_file(self, tmp_path):
        """store_raw writes a .nc file."""
        path = tmp_path / "raw.nc"
        assert store_raw(_sample_ds(), path) == path
        assert path.exists()

    def test_roundtrip_preserves_data(self, tmp_path):
        """Written data matches original."""
        ds = _sample_ds(20)
        path = tmp_path / "raw.nc"
        store_raw(ds, path)
        with xr.open_dataset(path) as loaded:
            for var in ds.data_vars:
                np.testing.assert_array_equal(loaded[var].values, ds[var].values)

    def test_attrs_written(self, tmp_path):
        """Global attributes persist in netCDF."""
        ds = _sample_ds()
        path = tmp_path / "raw.nc"
        store_raw(ds, path, attrs={"source": "test", "version": 1})
        with xr.open_dataset(path) as loaded:
            assert loaded.attrs["source"] == "test"
            assert loaded.attrs["version"] == 1


@pytest.mark.xr
class TestStoreProcessed:
    def test_write_and_append_mode(self, tmp_path):
        """mode='w' creates file; mode='a' appends variables."""
        path = tmp_path / "proc.nc"
        store_processed(_sample_ds().drop_vars("Ay"), path, mode="w")
        assert path.exists()
        store_processed(_sample_ds().drop_vars("Ax"), path, mode="a")
        with xr.open_dataset(path) as loaded:
            assert "Ax" in loaded and "Ay" in loaded

    def test_append_to_group_same_size(self, tmp_path):
        """mode='a' to same group with matching time dim size should succeed.

        Reprods: store_processed(..., group='i_p05', mode='a') called multiple times
        when multiple stems map to the same pcid (same time length assumed).
        """
        path = tmp_path / "proc.nc"
        ds = _sample_ds(50)
        store_processed(ds, path, group="i_p05", mode="a")
        store_processed(ds, path, group="i_p05", mode="a")
        with xr.open_dataset(path, group="i_p05") as loaded:
            assert loaded.sizes["time"] == 50

    def test_append_to_group_different_size_merges(self, tmp_path):
        """mode='a' to same group with different (longer) time dim should append, not crash.

        Reprods user log: 'NC write failed ... Unable to update size for existing
        dimension 'time' (24000 != 12000) — skipped ... proc_noAvg.nc'.

        Expected behavior: second call extends the existing group along 'time'
        (or fails gracefully with a clear error if extension is not supported),
        rather than silently skipping.  Uses non-overlapping time ranges so the
        position-aware ``append_to_nc`` writes all new rows.
        """
        path = tmp_path / "proc.nc"
        ds_small = _sample_ds(10)
        store_processed(ds_small, path, group="i_p05", mode="a")
        ds_large = _sample_ds(20, start="2024-01-01 00:00:10")
        store_processed(ds_large, path, group="i_p05", mode="a")
        with xr.open_dataset(path, group="i_p05") as loaded:
            assert loaded.sizes["time"] == 30  # 10 + 20 concatenated along time

    def test_aux_vars_dropped(self, tmp_path):
        """Battery/TempP live in raw.nc only — excluded from processed NC.

        Synthetic inclinometer/pressure probe dataset has aux fields present in
        the in-memory Dataset; processed output must drop them before persist.
        """
        time = pd.date_range("2024-01-01", periods=5, freq="s")
        ds = xr.Dataset(
            {"v": ("time", np.arange(5.0)),
             "TempP": ("time", np.full(5, 25.0)),
             "Battery": ("time", np.full(5, 9.0)),
             "Temp": ("time", np.full(5, 20.0)),
             "P_counts": ("time", np.arange(5.0)),
             "Pressure": ("time", np.full(5, 10.0)),
             "inclination": ("time", np.full(5, 12.0))},
            coords={"time": time},
        )
        raw_path = tmp_path / "probe.raw.nc"
        proc_path = tmp_path / "probe.proc.nc"

        store_raw(ds, raw_path)
        store_processed(ds, proc_path, group="i01", mode="w")

        with xr.open_dataset(raw_path) as raw:
            assert "TempP" in raw.data_vars
            assert "Battery" in raw.data_vars
        with xr.open_dataset(proc_path, group="i01") as proc:
            assert "TempP" not in proc.data_vars
            assert "Battery" not in proc.data_vars
            assert "v" in proc.data_vars


@pytest.mark.xr
class TestStoreProcessedH5pyFallback:
    """After h5py resize fallback, netCDF4 reads/writes must still work."""

    def test_incremental_after_h5py_rebuild(self, tmp_path):
        """store_processed_incremental must work after h5py rebuild.

        Reprods user crash: _rebuild_and_append writes with h5py →
        store_processed_incremental opens with netCDF4 →
        AttributeError: 'NoneType' object has no attribute 'dimensions'.
        """
        from tcm._xr.storage import store_processed_incremental

        path = tmp_path / "proc.nc"
        ds1 = _sample_ds(10)
        store_processed(ds1, path, group="i_p05bin2s", mode="a")
        # Force h5py rebuild: write different-size data to same group
        ds2 = _sample_ds(20, start="2024-01-01 00:01:00")
        store_processed(ds2, path, group="i_p05bin2s", mode="a")
        # Now store_processed_incremental must NOT crash on netCDF4 read
        ds3 = _sample_ds(5, start="2024-01-01 00:02:00")
        result = store_processed_incremental(ds3, path, group="i_p05bin2s")
        assert result == path
        with xr.open_dataset(path, group="i_p05bin2s") as loaded:
            assert loaded.sizes["time"] == 35  # 10 + 20 + 5

    def test_write_after_h5py_rebuild(self, tmp_path):
        """store_processed must not fail with OSError after h5py rebuild.

        Reprods user crash: _rebuild_and_append → subsequent xr.to_netcdf(mode='a')
        → OSError: [Errno -103] NetCDF: Can't write file.

        Uses non-overlapping time ranges so ``append_to_nc`` writes all rows
        (position-aware append only trims overlapping portions).
        """
        path = tmp_path / "proc.nc"
        ds1 = _sample_ds(10)
        store_processed(ds1, path, group="i_p05", mode="a")
        # Force h5py rebuild — different size, non-overlapping time
        ds2 = _sample_ds(20, start="2024-01-01 00:00:10")
        store_processed(ds2, path, group="i_p05", mode="a")
        # Third write must not crash with OSError
        ds3 = _sample_ds(30, start="2024-01-01 00:01:00")
        store_processed(ds3, path, group="i_p05", mode="a")
        with xr.open_dataset(path, group="i_p05") as loaded:
            assert loaded.sizes["time"] == 60  # 10 + 20 + 30

    def test_extend_preserves_dim_scales(self, tmp_path):
        """_h5py_extend_group must re-attach dimension scales after resize.

        Regression: h5py resize stripped DIMENSION_LIST metadata that netCDF4
        requires, causing ``AttributeError: 'NoneType' has no attribute
        'dimensions'`` on the next ``xr.open_dataset(engine=_constants.nc_engine)``.
        """
        from tcm._xr.storage import _h5py_extend_group, dt_ns_to_cf

        path = tmp_path / "proc.nc"
        ds1 = _sample_ds(10)
        store_processed(ds1, path, group="g", mode="a")
        # Force h5py rebuild → creates extendable datasets with dim scales
        ds2 = _sample_ds(20)
        store_processed(ds2, path, group="g", mode="a")
        with xr.open_dataset(path, group="g") as d:
            n_before = d.sizes["time"]
        # Simulate _h5py_extend_group (the resize-only path, no rebuild)
        extra = _sample_ds(3, start="2024-01-02")
        time_cf = dt_ns_to_cf(extra["time"].values.astype("datetime64[ns]"))
        var_arrays = {"time": time_cf, "Ax": extra["Ax"].values, "Ay": extra["Ay"].values}
        _h5py_extend_group(path, "g", 3, var_arrays)
        # netCDF4 must still be able to open the file
        with xr.open_dataset(path, group="g") as loaded:
            assert loaded.sizes["time"] == n_before + 3


@pytest.mark.xr
class TestEnsureDimScales:
    """ensure_dim_scales is idempotent and safe on valid files."""

    def test_ensure_dim_scales_idempotent(self, tmp_path):
        """Calling ensure_dim_scales on a valid file preserves data integrity."""
        from tcm._xr.storage import ensure_dim_scales

        path = tmp_path / "proc.nc"
        ds1 = _sample_ds(10)
        store_processed(ds1, path, group="g", mode="a")
        ds2 = _sample_ds(20, start="2024-01-01 00:00:10")
        store_processed(ds2, path, group="g", mode="a")

        # Repair on already-valid file — must not corrupt anything
        ensure_dim_scales(path)

        with xr.open_dataset(path, group="g") as loaded:
            assert loaded.sizes["time"] == 30
            assert set(loaded.data_vars) == {"Ax", "Ay"}

    def test_open_without_chunks(self, tmp_path):
        """Basic open returns Dataset with expected vars."""
        path = tmp_path / "proc.nc"
        store_processed(_sample_ds(), path)
        with xr.open_dataset(path) as loaded:
            assert isinstance(loaded, xr.Dataset)
            assert set(loaded.data_vars) == {"Ax", "Ay"}

    def test_open_with_chunks(self, tmp_path):
        """chunks parameter enables dask backing."""
        pytest.importorskip("dask")
        path = tmp_path / "proc.nc"
        store_processed(_sample_ds(100), path)
        # Chunk size 10 ≠ stored chunk along time; suppress xarray performance warning
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The specified chunks separate", category=UserWarning)
            with xr.open_dataset(path, chunks=10) as loaded:
                assert loaded.chunks is not None


@pytest.mark.xr
class TestIncrementalSkip:
    @pytest.mark.parametrize(
        ("offset", "expected"),
        [
            pytest.param(-10, True, id="output-newer"),
            pytest.param(10, False, id="output-older"),
        ],
    )
    def test_mtime_comparison(self, tmp_path, offset, expected):
        """Output {newer,older} than input → {skip,don't skip}."""
        path = tmp_path / "out.nc"
        path.write_text("dummy")
        input_mtime = path.stat().st_mtime + offset
        assert incremental_skip(path, input_mtime) is expected

    def test_no_output_returns_false(self):
        """Missing output → don't skip."""
        assert incremental_skip(Path("/nonexistent/file.nc"), 0.0) is False


@pytest.mark.xr
class TestStoreProcessedGrouped:
    """store_processed with group= writes per-probe groups."""

    def test_write_to_group_and_multiple_groups(self, tmp_path):
        """Writing with group= stores data in NC4 group; two probes can share one file."""
        path = tmp_path / "proc.nc"
        store_processed(_sample_ds(10), path, group="i_01", mode="w")
        store_processed(_sample_ds(20), path, group="i_02", mode="a")
        with (
            xr.open_dataset(path, group="i_01", engine=_constants.nc_engine) as g1,
            xr.open_dataset(path, group="i_02", engine=_constants.nc_engine) as g2,
        ):
            assert set(g1.data_vars) == {"Ax", "Ay"}
            assert g1.sizes["time"] == 10
            assert g2.sizes["time"] == 20


@pytest.mark.xr
class TestOpenProcessedGrouped:
    """open_processed_grouped reads per-probe groups from shared file."""

    def test_reads_multiple_groups(self, tmp_path):
        """Returns dict of group name → Dataset."""
        path = tmp_path / "proc.nc"
        store_processed(_sample_ds(10), path, group="i_01", mode="w")
        store_processed(_sample_ds(20), path, group="i_02", mode="a")
        result = open_processed_grouped(path)
        assert set(result.keys()) == {"i_01", "i_02"}
        assert result["i_01"].sizes["time"] == 10
        assert result["i_02"].sizes["time"] == 20
        for ds in result.values():
            ds.close()

    def test_returns_empty_for_no_groups(self, tmp_path):
        """File with only root variables → empty dict."""
        path = tmp_path / "proc.nc"
        _sample_ds().to_netcdf(path, engine=_constants.nc_engine)
        assert open_processed_grouped(path) == {}


@pytest.mark.xr
class TestPrepareCoefs:
    """prepare_coefs applies azimuth correction and zeroing rotation."""

    @pytest.mark.parametrize(
        ("azimuth_add", "expected_deg"),
        [
            pytest.param(5.0, 15.0, id="add-5"),
            pytest.param(None, 10.0, id="no-add"),
        ],
    )
    def test_azimuth_correction(self, azimuth_add, expected_deg):
        """azimuth_add updates (or leaves) azimuth_shift_deg in coefs."""
        coefs = {"azimuth_shift_deg": 10.0, "dates": {}}
        result, _, _, _ = prepare_coefs(coefs, _sample_ds(), azimuth_add=azimuth_add)
        assert result["azimuth_shift_deg"] == pytest.approx(expected_deg)

    def test_returns_coef_zeroing_matrix(self):
        """Returns coef_zeroing_matrix from get_coef_zeroing_matrix."""
        coefs = {"Rz": np.eye(3), "dates": {}}
        _, matrix, _, _ = prepare_coefs(coefs, _sample_ds())
        # Rz == eye(3) → matrix is None (no rotation needed)
        assert matrix is None

    def test_zeroing_returns_none_when_no_data_in_range(self):
        """time_ranges outside data range → no rotation applied."""
        time = pd.date_range("2024-01-01", periods=10, freq="s", tz="UTC")
        ds = xr.Dataset(
            {
                "Ax": ("time", np.ones(10)),
                "Ay": ("time", np.ones(10)),
                "Az": ("time", np.ones(10) * 9.8),
                "Mx": ("time", np.ones(10)),
                "My": ("time", np.zeros(10)),
                "Mz": ("time", np.zeros(10)),
            },
            coords={"time": time},
        )
        result = coef_zeroing_rotation_from_data(
            ds,
            time_ranges=["2099-01-01", "2099-01-02"],
            Ag=np.eye(3),
            Cg=np.zeros(3),
        )
        assert result is None

    def test_prepare_coefs_g0xyz_produces_coef_zeroing_matrix(self):
        """prepare_coefs with g0xyz kwarg (input.calib.g0xyz) → non-None coef_zeroing_matrix."""
        coefs = {
            "Ag": np.eye(3) * 0.00173,
            "Cg": np.array([10.0, 10.0, 10.0]),
            "dates": {},
        }
        _, matrix, _, msg = prepare_coefs(coefs, _sample_ds(), g0xyz=np.array([0.1, 0.2, 9.8]))
        assert matrix is not None and matrix.shape == (3, 3), (
            f"g0xyz should produce coef_zeroing_matrix, got {matrix}"
        )
        assert "g0xyz" in msg

    def test_prepare_coefs_azimuth_from_data(self):
        """time_ranges_azimuth computes azimuth_shift_deg from mag+accel data."""
        n = 50
        tilt = np.radians(15.0)
        heading = np.radians(45.0)
        time = pd.date_range("2024-01-01", periods=n, freq="s")
        ds = xr.Dataset(
            {
                "Ax": ("time", np.sin(tilt) * np.ones(n)),
                "Ay": ("time", np.zeros(n)),
                "Az": ("time", np.cos(tilt) * np.ones(n)),
                "Mx": ("time", np.cos(heading) * np.cos(tilt) * np.ones(n)),
                "My": ("time", np.sin(heading) * np.ones(n)),
                "Mz": ("time", -np.cos(heading) * np.sin(tilt) * np.ones(n)),
            },
            coords={"time": time},
        )
        coefs = {
            "Ag": np.eye(3),
            "Cg": np.zeros(3),
            "Ah": np.eye(3),
            "Ch": np.zeros(3),
            "azimuth_shift_deg": 180.0,
            "dates": {},
        }
        result, _, _, msg = prepare_coefs(
            coefs,
            ds,
            time_ranges_azimuth=["2024-01-01", "2024-01-02"],
        )
        # azimuth_shift_deg should be overwritten by data-computed value (~45°)
        assert result["azimuth_shift_deg"] != 180.0, (
            f"azimuth_shift_deg should be updated from data, still {result['azimuth_shift_deg']}"
        )
        assert abs(result["azimuth_shift_deg"] - 45.0) < 10.0, (
            f"Expected azimuth ~45° for heading=45°, got {result['azimuth_shift_deg']:.1f}°"
        )


@pytest.mark.xr
class TestCoefAzimuthFromData:
    """coef_azimuth_from_data computes azimuth from mag+accel unit vectors."""

    def test_returns_none_when_no_time_ranges(self):
        """No time_ranges → None."""
        assert coef_azimuth_from_data(_sample_ds(), time_ranges=None) is None

    def test_returns_none_when_no_data_in_range(self):
        """Time range outside data → None."""
        ds = _sample_ds()
        assert (
            coef_azimuth_from_data(
                ds,
                time_ranges=["2099-01-01", "2099-01-02"],
                Ah=np.eye(3),
                Ch=np.zeros(3),
                Ag=np.eye(3),
                Cg=np.zeros(3),
            )
            is None
        )

    def test_returns_finite_degrees(self):
        """Valid data → finite float (degrees)."""
        n = 50
        time = pd.date_range("2024-01-01", periods=n, freq="s")
        ds = xr.Dataset(
            {
                "Ax": ("time", np.zeros(n)),
                "Ay": ("time", np.zeros(n)),
                "Az": ("time", np.ones(n)),
                "Mx": ("time", np.ones(n)),
                "My": ("time", np.zeros(n)),
                "Mz": ("time", np.zeros(n)),
            },
            coords={"time": time},
        )
        result = coef_azimuth_from_data(
            ds,
            time_ranges=["2024-01-01", "2024-01-02"],
            Ah=np.eye(3),
            Ch=np.zeros(3),
            Ag=np.eye(3),
            Cg=np.zeros(3),
        )
        assert isinstance(result, float)
        assert np.isfinite(result)


@pytest.mark.xr
class TestDiscoverTables:
    """_discover_tables finds groups in HDF5/NC files."""

    @pytest.mark.parametrize(
        ("pattern", "expected"),
        [
            pytest.param("incl*", {"incl_01", "incl_02"}, id="match-incl"),
            pytest.param("pressure*", set(), id="no-match"),
        ],
    )
    def test_discovers_nc_groups(self, tmp_path, pattern, expected):
        """Finds groups matching pattern in NC file."""
        nc_path = tmp_path / "test.nc"
        ds = _sample_ds()
        ds.to_netcdf(nc_path, group="incl_01", engine=_constants.nc_engine)
        ds.to_netcdf(nc_path, group="incl_02", engine=_constants.nc_engine, mode="a")
        assert set(_discover_tables(nc_path, pattern)) == expected

    @pytest.mark.parametrize(
        ("pattern", "tables", "expected"),
        [
            # incl* glob → incl.*? regex: matches both dot and underscore
            pytest.param(
                "incl*",
                ["incl.05", "incl.06", "incl_p05", "incl_p06", "pressure.01"],
                {"incl.05", "incl.06", "incl_p05", "incl_p06"},
                id="incl-star-glob-matches-both",
            ),
            # incl.* glob → incl\..*? regex: literal dot required (HDF5 only)
            pytest.param(
                "incl.*",
                ["incl.05", "incl_p05", "pressure.01"],
                {"incl.05"},
                id="incl-dot-star-glob-dot-only",
            ),
            # pressure* glob → matches both dot and underscore
            pytest.param(
                "pressure*",
                ["incl.05", "pressure.01", "pressure_p01"],
                {"pressure.01", "pressure_p01"},
                id="glob-pressure-star",
            ),
        ],
    )
    def test_pattern_matching_semantics(self, pattern, tables, expected):
        """Verify glob→regex semantics for table patterns."""
        import re

        from tcm.csv_load import _glob_to_regex

        regex = re.compile(_glob_to_regex(pattern))
        matched = {t for t in tables if regex.fullmatch(t)}
        assert matched == expected, (
            f"Pattern {pattern!r} (regex {regex.pattern!r}) against {tables}: "
            f"expected {expected}, got {matched}"
        )


def _sample_ds_tz_utc(n: int = 10) -> xr.Dataset:
    """Build a minimal Dataset with tz-aware (UTC) datetime64 time coord."""
    time = pd.date_range("2024-01-01", periods=n, freq="s", tz="UTC")
    return xr.Dataset(
        {"Ax": ("time", np.arange(n, dtype=float)), "Ay": ("time", np.ones(n))},
        coords={"time": time},
    )


@pytest.mark.xr
class TestStripTzDatetime:
    """_strip_tz_datetime converts tz-aware datetime64 to naive."""

    def test_strips_utc_from_time_coord(self):
        """UTC timezone is removed from time coordinate."""
        ds = _sample_ds_tz_utc()
        result = strip_tz_datetime(ds)
        # After strip: numpy DateTime64DType has no .tz attr
        assert not hasattr(result.coords["time"].dtype, "tz") or result.coords["time"].dtype.tz is None

    def test_preserves_naive_datetime(self):
        """Already-tz-naive coords pass through unchanged."""
        ds = _sample_ds()
        result = strip_tz_datetime(ds)
        np.testing.assert_array_equal(result.coords["time"].values, ds.coords["time"].values)

    def test_values_identical_after_strip(self):
        """Stripping tz preserves nanosecond timestamps."""
        ds = _sample_ds_tz_utc()
        result = strip_tz_datetime(ds)
        # Compare via int64 view (both should be same ns-since-epoch)
        expected_ns = ds.coords["time"].values.astype("datetime64[ns]").astype(np.int64)
        result_ns = result.coords["time"].values.astype("datetime64[ns]").astype(np.int64)
        np.testing.assert_array_equal(result_ns, expected_ns)


@pytest.mark.xr
class TestStoreWithTzAwareDatetime:
    """store_raw / store_processed handle tz-aware datetime64[ns, UTC]."""

    def test_store_raw_tz_utc(self, tmp_path):
        """store_raw succeeds with UTC-aware time and produces valid NC."""
        ds = _sample_ds_tz_utc()
        path = tmp_path / "raw_tz.nc"
        store_raw(ds, path)
        assert path.exists()
        with xr.open_dataset(path) as loaded:
            np.testing.assert_array_equal(loaded["Ax"].values, ds["Ax"].values)

    def test_store_processed_tz_utc(self, tmp_path):
        """store_processed + group= succeed with UTC-aware time."""
        ds_root = _sample_ds_tz_utc()
        ds_grp = _sample_ds_tz_utc()
        path_root = tmp_path / "proc_tz.nc"
        path_grp = tmp_path / "proc_grp_tz.nc"
        store_processed(ds_root, path_root)
        store_processed(ds_grp, path_grp, group="i_01", mode="w")
        assert path_root.exists() and path_grp.exists()
        with xr.open_dataset(path_root) as loaded:
            np.testing.assert_array_equal(loaded["Ax"].values, ds_root["Ax"].values)
        with xr.open_dataset(path_grp, group="i_01", engine=_constants.nc_engine) as loaded:
            np.testing.assert_array_equal(loaded["Ax"].values, ds_grp["Ax"].values)

    def test_roundtrip_tz_utc_preserves_data(self, tmp_path):
        """Write-read roundtrip with tz-aware input preserves variable values."""
        ds = _sample_ds_tz_utc(20)
        path = tmp_path / "roundtrip_tz.nc"
        store_processed(ds, path)
        with xr.open_dataset(path) as loaded:
            for var in ds.data_vars:
                np.testing.assert_array_equal(loaded[var].values, ds[var].values)


# ---------------------------------------------------------------------------
# store_processed_incremental: _run_params attribute
# ---------------------------------------------------------------------------


@pytest.mark.xr
class TestRunParamsAttr:
    """store_processed_incremental stores param_spans on NC file."""

    def test_run_params_written(self, tmp_path):
        """param_spans written to /param_spans/{group}."""
        from tcm._xr.store_params import get_latest_params, read_param_spans
        from tcm._xr.storage import store_processed_incremental

        path = tmp_path / "proc.nc"
        ds = _sample_ds(50)
        store_processed_incremental(ds, path, group="i_p01", filter_params="filter.max.g=1.0")
        params = read_param_spans(path, "i_p01")
        assert get_latest_params(params) == "filter.max.g=1.0", (
            f"param_spans latest mismatch: expected 'filter.max.g=1.0', got {get_latest_params(params)!r}"
        )

    def test_run_params_warn_on_diff(self, tmp_path):
        """When stored settings differ (ignoring time_ranges) and force_reprocess=False, raise ValueError."""
        from tcm._xr.storage import store_processed_incremental

        path = tmp_path / "proc.nc"
        ds = _sample_ds(50)
        # First write with one set of params
        store_processed_incremental(
            ds,
            path,
            group="i_p01",
            filter_params="filter.max.g=1.0\ncoef.Ax=1.0",
        )
        # Second write with different coef — data is covered, should error
        with pytest.raises(ValueError, match="Coefficients/params changed"):
            store_processed_incremental(
                ds,
                path,
                group="i_p01",
                filter_params="filter.max.g=1.0\ncoef.Ax=2.0",
            )

    def test_run_params_no_warn_when_same(self, tmp_path, caplog):
        """Same settings on re-run → no warning (skip is silent)."""
        import logging

        from tcm._xr.storage import store_processed_incremental

        path = tmp_path / "proc.nc"
        ds = _sample_ds(50)
        params = "filter.max.g=1.0\ncoef.Ag=[1]"
        store_processed_incremental(ds, path, group="i_p01", filter_params=params)
        with caplog.at_level(logging.WARNING, logger="tcm._xr.storage"):
            store_processed_incremental(ds, path, group="i_p01", filter_params=params)
        assert not any("Run params changed" in r.message for r in caplog.records), (
            f"Unexpected warning with same params: {[r.message for r in caplog.records]}"
        )

    def test_run_params_history_preserved(self, tmp_path):
        """Each write appends a new interval — previous intervals are preserved."""
        from tcm._xr.store_params import get_latest_params, read_param_spans
        from tcm._xr.storage import store_processed_incremental

        path = tmp_path / "proc.nc"
        ds_a = _sample_ds(50)
        ds_b = _sample_ds(50, start="2024-02-01")
        params_a = "filter.max.g=1.0\ncoef.Ax=1.0"
        params_b = "filter.max.g=1.0\ncoef.Ax=2.0"

        store_processed_incremental(ds_a, path, group="i_p01", filter_params=params_a)
        store_processed_incremental(ds_b, path, group="i_p01", filter_params=params_b, force_reprocess=True)

        params = read_param_spans(path, "i_p01")
        assert params.sizes.get("interval", 0) == 2, (
            f"Expected 2 param_spans intervals, got {params.sizes.get('interval', 0)}"
        )
        assert get_latest_params(params) == params_b
        assert params_a in params["params"].values
        assert params_b in params["params"].values

    def test_run_params_no_duplicate_when_unchanged(self, tmp_path):
        """Same params on force-reprocess → no duplicate interval."""
        from tcm._xr.store_params import get_latest_params, read_param_spans
        from tcm._xr.storage import store_processed_incremental

        path = tmp_path / "proc.nc"
        ds_a = _sample_ds(50)
        ds_b = _sample_ds(50, start="2024-02-01")
        params = "filter.max.g=1.0\ncoef.Ax=1.0"

        store_processed_incremental(ds_a, path, group="i_p01", filter_params=params)
        store_processed_incremental(ds_b, path, group="i_p01", filter_params=params, force_reprocess=True)

        stored = read_param_spans(path, "i_p01")
        assert stored.sizes.get("interval", 0) == 1, (
            f"Expected 1 interval (dedup), got {stored.sizes.get('interval', 0)}"
        )
        assert get_latest_params(stored) == params


# ---------------------------------------------------------------------------
# ensure_dim_scales: OSError handling
# ---------------------------------------------------------------------------


@pytest.mark.xr
class TestEnsureDimScalesOSError:
    """ensure_dim_scales catches OSError (Windows file lock) gracefully."""

    def test_oserror_caught_and_warned(self, tmp_path, caplog, mocker):
        """OSError from h5py.File is caught and logged as warning (not re-raised)."""
        import logging

        from tcm._xr.storage import ensure_dim_scales

        path = tmp_path / "proc.nc"
        ds = _sample_ds(10)
        store_processed(ds, path, group="g", mode="a")

        # Patch h5py.File to raise OSError (simulates Windows file lock)
        def _raise_os(*a, **kw):
            raise OSError("Unable to synchronously open file (Win32 lock)")

        mocker.patch.object(_h5py, "File", side_effect=_raise_os)
        with caplog.at_level(logging.WARNING, logger="tcm._xr.storage"):
            ensure_dim_scales(path)  # must NOT raise
        assert any("ensure_dim_scales failed" in r.message for r in caplog.records), (
            f"Expected warning about failed ensure_dim_scales, got: {[r.message for r in caplog.records]}"
        )
