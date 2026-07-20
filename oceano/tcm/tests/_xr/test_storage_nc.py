"""Tests for tcm/_xr/storage.py — NC log table, incremental append, tz stripping."""
from __future__ import annotations

from datetime import datetime

import h5py
import numpy as np
import pytest
import xarray as xr

from tcm._xr.storage import (
    _read_nc_group_as_dataset,
    _read_nc_group_h5py,
    _strip_tz_datetime,
    _write_dataset_to_nc_group,
    _LogDecision,
    append_to_nc,
    check_file_vs_log,
    keep_recorded_nc,
    nc_incremental_update,
    read_nc_log,
    store_processed_incremental,
    write_nc_log,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dt_ns(*strings: str) -> np.ndarray:
    """Convenience: ISO date strings → datetime64[ns] array."""
    return np.array(strings, dtype="datetime64[ns]")


def _make_ds(n: int = 50, start: str = "2024-01-01", seed: int = 42) -> xr.Dataset:
    """Small time-series Dataset for testing."""
    rng = np.random.default_rng(seed)
    return xr.Dataset(
        {"Ax": ("time", rng.normal(0, 1, n)),
         "Ay": ("time", rng.normal(0, 1, n))},
        coords={"time": xr.date_range(start, periods=n, freq="s")},
    )


# ---------------------------------------------------------------------------
# Log table I/O
# ---------------------------------------------------------------------------

@pytest.mark.xr
class TestNcLogTable:
    """read_nc_log / write_nc_log round-trip."""

    @pytest.fixture()
    def sample_log(self) -> xr.Dataset:
        """Sample log with all columns."""
        return xr.Dataset(
            {
                "fileName": ("Date0", ["raw/@i_01", "raw/@i_02"]),
                "fileChangeTime": ("Date0", _dt_ns("2024-01-01", "2024-01-02")),
                "DateEnd": ("Date0", _dt_ns("2024-01-01T12:00:00", "2024-01-02T12:00:00")),
                "DateProc": ("Date0", _dt_ns("2024-01-01", "2024-01-02")),
            },
            coords={"Date0": _dt_ns("2024-01-01", "2024-01-02")},
        )

    @pytest.fixture()
    def empty_nc(self, tmp_path):
        nc_path = tmp_path / "test.raw.nc"
        with h5py.File(nc_path, "w"):
            pass
        return nc_path

    def test_round_trip_all_columns(self, sample_log, empty_nc):
        """All columns survive write → read."""
        write_nc_log(empty_nc, "incl_01", sample_log)
        result = read_nc_log(empty_nc, "incl_01")
        assert result.sizes["Date0"] == 2
        np.testing.assert_array_equal(result["Date0"].values, sample_log["Date0"].values)
        np.testing.assert_array_equal(result["fileName"].values, sample_log["fileName"].values)
        for col in ("fileChangeTime", "DateEnd", "DateProc"):
            np.testing.assert_array_equal(result[col].values, sample_log[col].values), (
                f"Round-trip: {col} mismatch"
            )

    def test_round_trip_partial_columns(self, empty_nc):
        """Only fileName + Date0 (no optional datetime columns)."""
        log = xr.Dataset(
            {"fileName": ("Date0", ["a/b"])},
            coords={"Date0": _dt_ns("2024-06-01")},
        )
        write_nc_log(empty_nc, "incl_01", log)
        result = read_nc_log(empty_nc, "incl_01")
        assert result.sizes["Date0"] == 1
        assert "fileChangeTime" not in result.data_vars
        assert result["fileName"].values[0] == "a/b"

    def test_returns_empty_when_missing_file(self, tmp_path):
        """Missing file → empty Dataset."""
        result = read_nc_log(tmp_path / "nonexistent.nc", "incl_01")
        assert result.sizes.get("Date0", 0) == 0

    def test_returns_empty_when_missing_group(self, empty_nc):
        """File exists but group doesn't → empty."""
        result = read_nc_log(empty_nc, "nonexistent_tbl")
        assert result.sizes.get("Date0", 0) == 0

    def test_write_empty_log(self, empty_nc):
        """Writing empty log doesn't crash; reading back is empty."""
        empty = xr.Dataset(coords={"Date0": _dt_ns()})
        write_nc_log(empty_nc, "incl_01", empty)
        assert read_nc_log(empty_nc, "incl_01").sizes.get("Date0", 0) == 0

    def test_overwrite(self, sample_log, empty_nc):
        """Writing twice overwrites cleanly (no duplicate rows)."""
        write_nc_log(empty_nc, "incl_01", sample_log)
        write_nc_log(empty_nc, "incl_01", sample_log.isel(Date0=slice(0, 1)))
        assert read_nc_log(empty_nc, "incl_01").sizes["Date0"] == 1

    def test_multiple_tables(self, sample_log, empty_nc):
        """Different tbl names in same file are independent."""
        write_nc_log(empty_nc, "incl_01", sample_log)
        write_nc_log(empty_nc, "incl_02", sample_log.isel(Date0=slice(0, 1)))
        assert read_nc_log(empty_nc, "incl_01").sizes["Date0"] == 2
        assert read_nc_log(empty_nc, "incl_02").sizes["Date0"] == 1

    def test_dtype_datetime64_ns(self, sample_log, empty_nc):
        """All datetime columns are datetime64[ns] after read."""
        write_nc_log(empty_nc, "incl_01", sample_log)
        result = read_nc_log(empty_nc, "incl_01")
        assert result["Date0"].dtype == np.dtype("datetime64[ns]")
        for col in ("fileChangeTime", "DateEnd", "DateProc"):
            assert result[col].dtype == np.dtype("datetime64[ns]"), f"{col} dtype"


# ---------------------------------------------------------------------------
# keep_recorded_nc
# ---------------------------------------------------------------------------

@pytest.mark.xr
class TestKeepRecordedNc:

    @pytest.fixture()
    def log(self, tmp_path) -> xr.Dataset:
        nc_path = tmp_path / "log.nc"
        with h5py.File(nc_path, "w"):
            pass
        log = xr.Dataset(
            {
                "fileName": ("Date0", ["raw/@i_01", "raw/@i_02"]),
                "fileChangeTime": ("Date0", _dt_ns("2024-01-01", "2024-01-02")),
            },
            coords={"Date0": _dt_ns("2024-01-01", "2024-01-02")},
        )
        write_nc_log(nc_path, "incl_01", log)
        return read_nc_log(nc_path, "incl_01")

    def test_same_file_same_time_skip(self, log):
        """Same fileName + same fileChangeTime → skip."""
        cur = {"fileName": "raw/@i_01", "fileChangeTime": np.datetime64("2024-01-01", "ns")}
        assert keep_recorded_nc(cur, log) is True

    def test_same_file_newer_time_no_skip(self, log):
        """Same fileName + newer fileChangeTime → don't skip (default keep_newer=True)."""
        cur = {"fileName": "raw/@i_01", "fileChangeTime": np.datetime64("2024-01-03", "ns")}
        assert keep_recorded_nc(cur, log) is False

    def test_different_file_no_skip(self, log):
        """Different fileName → don't skip."""
        cur = {"fileName": "raw/@other", "fileChangeTime": np.datetime64("2024-01-01", "ns")}
        assert keep_recorded_nc(cur, log) is False

    def test_exact_match_mode(self, log):
        """keep_newer=False: skip only on exact fileChangeTime match."""
        cur = {"fileName": "raw/@i_01", "fileChangeTime": np.datetime64("2024-01-01", "ns")}
        assert keep_recorded_nc(cur, log, keep_newer=False) is True
        # Same file, same time as existing record → skip
        cur2 = {"fileName": "raw/@i_01", "fileChangeTime": np.datetime64("2024-01-01", "ns")}
        assert keep_recorded_nc(cur2, log, keep_newer=False) is True

    def test_empty_log(self):
        """Empty log → never skip."""
        empty = xr.Dataset(coords={"Date0": _dt_ns()})
        cur = {"fileName": "x", "fileChangeTime": np.datetime64("2024-01-01", "ns")}
        assert keep_recorded_nc(cur, empty) is False

    def test_accepts_datetime_input(self, log):
        """fileChangeTime as Python datetime (from file_name_and_time_to_record)."""
        cur = {"fileName": "raw/@i_01", "fileChangeTime": datetime(2024, 1, 1)}
        assert keep_recorded_nc(cur, log) is True


# ---------------------------------------------------------------------------
# check_file_vs_log — new 3-way decision logic
# ---------------------------------------------------------------------------

@pytest.mark.xr
class TestCheckFileVsLog:

    @pytest.fixture()
    def log(self) -> xr.Dataset:
        return xr.Dataset(
            {
                "fileName": ("Date0", ["raw/@i_01", "raw/@i_02"]),
                "fileChangeTime": ("Date0", _dt_ns("2024-01-01", "2024-01-02")),
            },
            coords={"Date0": _dt_ns("2024-01-01", "2024-01-02")},
        )

    def test_same_file_older_time_skip(self, log):
        cur = {"fileName": "raw/@i_01", "fileChangeTime": np.datetime64("2023-12-01", "ns")}
        assert check_file_vs_log(cur, log) == _LogDecision.SKIP

    def test_same_file_same_time_skip(self, log):
        cur = {"fileName": "raw/@i_01", "fileChangeTime": np.datetime64("2024-01-01", "ns")}
        assert check_file_vs_log(cur, log) == _LogDecision.SKIP

    def test_same_file_newer_time_resume(self, log):
        cur = {"fileName": "raw/@i_01", "fileChangeTime": np.datetime64("2024-06-01", "ns")}
        assert check_file_vs_log(cur, log) == _LogDecision.RESUME

    def test_different_file_new_file(self, log):
        cur = {"fileName": "raw/@other", "fileChangeTime": np.datetime64("2024-01-01", "ns")}
        assert check_file_vs_log(cur, log) == _LogDecision.NEW_FILE

    def test_empty_log_new_file(self):
        empty = xr.Dataset(coords={"Date0": _dt_ns()})
        cur = {"fileName": "x", "fileChangeTime": np.datetime64("2024-01-01", "ns")}
        assert check_file_vs_log(cur, empty) == _LogDecision.NEW_FILE


# ---------------------------------------------------------------------------
# _time_range_overlap — position classification
# ---------------------------------------------------------------------------

# todo: implement

# ---------------------------------------------------------------------------
# Incremental append
# ---------------------------------------------------------------------------

@pytest.mark.xr
class TestNcIncrementalAppend:

    def test_append_creates_file(self, tmp_path):
        """append_to_nc creates NC file when absent."""
        ds = _make_ds(50)
        nc_path = tmp_path / "test.raw.nc"
        append_to_nc(ds, nc_path, "incl_01")
        with xr.open_dataset(nc_path, group="incl_01", engine="netcdf4") as result:
            assert result.sizes["time"] == 50

    def test_append_after_fast_path(self, tmp_path):
        """append_to_nc AFTER: sequential data → resize (no re-read)."""
        ds1 = _make_ds(50, start="2024-01-01", seed=42)
        ds2 = _make_ds(30, start="2024-01-01 00:00:50", seed=43)  # starts at t[49]+1s
        nc_path = tmp_path / "test.raw.nc"
        append_to_nc(ds1, nc_path, "incl_01")
        append_to_nc(ds2, nc_path, "incl_01")
        with h5py.File(nc_path, "r") as f:
            assert f["incl_01"]["time"].shape[0] == 80
            # Verify order: first 50 times from ds1, then 30 from ds2 (never re-sorted)
            times = f["incl_01"]["time"][:]
            assert times[0] < times[-1]  # monotonically increasing

    def test_append_containment_skip(self, tmp_path):
        """append_to_nc skips when new data fully contained in existing."""
        ds = _make_ds(50, start="2024-01-01")
        nc_path = tmp_path / "test.raw.nc"
        append_to_nc(ds, nc_path, "incl_01")
        # Sub-range of existing data — should skip
        ds_sub = ds.isel(time=slice(5, 45))
        append_to_nc(ds_sub, nc_path, "incl_01")
        with xr.open_dataset(nc_path, group="incl_01", engine="netcdf4") as result:
            assert result.sizes["time"] == 50

    def test_append_overlap_tail_trims_and_appends(self, tmp_path):
        """OVERLAP_TAIL: overlapping portion of new data is trimmed, then appended."""
        ds1 = _make_ds(50, start="2024-01-01", seed=42)
        ds2 = _make_ds(60, start="2024-01-01 00:00:30", seed=43)  # overlap last 20s of ds1
        nc_path = tmp_path / "test.raw.nc"
        append_to_nc(ds1, nc_path, "incl_01")
        append_to_nc(ds2, nc_path, "incl_01")
        with h5py.File(nc_path, "r") as f:
            n = f["incl_01"]["time"].shape[0]
            # 50 original + (60 - 20 overlapping) = 90
            assert n == 90

    def test_append_prepend_before(self, tmp_path):
        """BEFORE: new data earlier than existing gets prepended."""
        ds_early = _make_ds(20, start="2023-12-31", seed=42)
        ds_late = _make_ds(30, start="2024-01-01", seed=43)
        nc_path = tmp_path / "test.raw.nc"
        # Write late data first
        append_to_nc(ds_late, nc_path, "incl_01")
        # Then prepend early data
        append_to_nc(ds_early, nc_path, "incl_01")
        with h5py.File(nc_path, "r") as f:
            n = f["incl_01"]["time"].shape[0]
            assert n == 50  # 20 + 30, no overlap

    def test_append_prepend_preserves_data(self, tmp_path):
        """BEFORE prepend must preserve all data values, not just row count.

        Reprods bug: _prepend_nc_group chunked copy loop only copies the last
        element when n_old < chunk (default 50000). With n_old=12000 and
        chunk=50000, range(11999, -1, -50000) = [11999] → only index 11999
        is shifted; indices 0..11998 become garbage zeros.
        """
        ds_early = _make_ds(20, start="2023-12-31", seed=42)
        ds_late = _make_ds(30, start="2024-01-01", seed=43)
        nc_path = tmp_path / "test.raw.nc"
        append_to_nc(ds_late, nc_path, "incl_01")
        append_to_nc(ds_early, nc_path, "incl_01")
        result = _read_nc_group_h5py(nc_path, "incl_01")
        # Early data (prepended) should be at the start
        np.testing.assert_array_equal(result["time"][:20], ds_early["time"].values)
        # Late data should follow — all 30 values intact
        np.testing.assert_array_equal(result["time"][20:], ds_late["time"].values)
        # Data vars also preserved
        np.testing.assert_almost_equal(result["Ax"][:20], ds_early["Ax"].values, decimal=7)
        np.testing.assert_almost_equal(result["Ax"][20:], ds_late["Ax"].values, decimal=7)

    def test_append_overlap_head_trims_and_prepends(self, tmp_path):
        """OVERLAP_HEAD: overlapping new data head trimmed, then prepended."""
        ds_late = _make_ds(30, start="2024-01-01 00:00:20", seed=43)
        ds_early = _make_ds(50, start="2024-01-01", seed=42)  # overlaps first 20s of ds_late
        nc_path = tmp_path / "test.raw.nc"
        append_to_nc(ds_late, nc_path, "incl_01")
        append_to_nc(ds_early, nc_path, "incl_01")
        with h5py.File(nc_path, "r") as f:
            n = f["incl_01"]["time"].shape[0]
            # 20 non-overlapping from ds_early + 30 from ds_late = 50
            assert n == 50

    def test_append_tz_utc_datetime(self, tmp_path):
        """append_to_nc succeeds with tz-aware (UTC) datetime64 time."""
        import pandas as pd

        ds_tz = xr.Dataset(
            {"Ax": ("time", np.ones(10)), "Ay": ("time", np.zeros(10))},
            coords={"time": pd.date_range("2024-01-01", periods=10, freq="s", tz="UTC")},
        )
        nc_path = tmp_path / "test_tz.raw.nc"
        append_to_nc(ds_tz, nc_path, "incl_01")
        with xr.open_dataset(nc_path, group="incl_01", engine="netcdf4") as ds:
            assert ds.sizes["time"] == 10

    def test_never_re_sorts(self, tmp_path):
        """append_to_nc preserves input order — never re-sorts concatenated data."""
        ds1 = _make_ds(50, start="2024-01-01", seed=42)
        ds2 = _make_ds(30, start="2024-01-01 00:00:50", seed=43)
        nc_path = tmp_path / "test.raw.nc"
        append_to_nc(ds1, nc_path, "incl_01")
        append_to_nc(ds2, nc_path, "incl_01")
        # Verify via h5py: last value of ds1 matches first value before ds2 region
        result = _read_nc_group_h5py(nc_path, "incl_01")
        # ds1 values: time[0]..time[49], ds2 values: time[50]..time[79]
        t1_last = ds1["time"].values[-1]
        t2_first = ds2["time"].values[0]
        assert result["time"].values[49] == t1_last
        assert result["time"].values[50] == t2_first

    def test_incremental_update_appends(self, tmp_path):
        """nc_incremental_update appends new data and updates log."""
        ds1 = _make_ds(50, seed=42)
        ds2 = _make_ds(30, start="2024-01-01 00:00:50", seed=43)
        nc_path = tmp_path / "test.raw.nc"
        meta1 = {"fileName": "raw/@i_01", "fileChangeTime": np.datetime64("2024-01-01", "ns")}
        assert nc_incremental_update(ds1, nc_path, "incl_01", meta1) is True
        meta2 = {"fileName": "raw/@i_02", "fileChangeTime": np.datetime64("2024-01-02", "ns")}
        assert nc_incremental_update(ds2, nc_path, "incl_01", meta2) is True
        log = read_nc_log(nc_path, "incl_01")
        assert log.sizes["Date0"] == 2

    def test_incremental_skip_duplicate(self, tmp_path):
        """nc_incremental_update skips duplicate file."""
        ds = _make_ds(50)
        nc_path = tmp_path / "test.raw.nc"
        meta = {"fileName": "raw/@i_01", "fileChangeTime": np.datetime64("2024-01-01", "ns")}
        assert nc_incremental_update(ds, nc_path, "incl_01", meta) is True
        assert nc_incremental_update(ds, nc_path, "incl_01", meta) is False

    def test_incremental_skip_older_file(self, tmp_path):
        """nc_incremental_update skips same fileName with older fileChangeTime."""
        ds = _make_ds(50)
        nc_path = tmp_path / "test.raw.nc"
        meta_new = {"fileName": "raw/@i_01", "fileChangeTime": np.datetime64("2024-06-01", "ns")}
        assert nc_incremental_update(ds, nc_path, "incl_01", meta_new) is True
        meta_old = {"fileName": "raw/@i_01", "fileChangeTime": np.datetime64("2024-01-01", "ns")}
        assert nc_incremental_update(ds, nc_path, "incl_01", meta_old) is False

    def test_incremental_resume_same_file_newer(self, tmp_path):
        """RESUME: same fileName but newer fileChangeTime → only tail appended."""
        ds1 = _make_ds(50, start="2024-01-01", seed=42)      # t=0..49s
        ds2 = _make_ds(80, start="2024-01-01", seed=99)       # t=0..79s (same file, longer)
        nc_path = tmp_path / "test.raw.nc"
        meta1 = {"fileName": "raw/@i_01", "fileChangeTime": np.datetime64("2024-01-01", "ns")}
        nc_incremental_update(ds1, nc_path, "incl_01", meta1)
        # Same file, newer mtime → resume: append only t=50s..79s
        meta2 = {"fileName": "raw/@i_01", "fileChangeTime": np.datetime64("2024-06-01", "ns")}
        result = nc_incremental_update(ds2, nc_path, "incl_01", meta2)
        assert result is True
        # Should have 80 total (50 original + 30 tail appended)
        with h5py.File(nc_path, "r") as f:
            assert f["incl_01"]["time"].shape[0] == 80

    def test_incremental_resume_log_has_two_rows(self, tmp_path):
        """RESUME: log gets two rows (original start + tail end) for updated file."""
        ds1 = _make_ds(50, start="2024-01-01", seed=42)
        ds2 = _make_ds(80, start="2024-01-01", seed=99)
        nc_path = tmp_path / "test.raw.nc"
        meta1 = {"fileName": "raw/@i_01", "fileChangeTime": np.datetime64("2024-01-01", "ns")}
        nc_incremental_update(ds1, nc_path, "incl_01", meta1)
        log_after1 = read_nc_log(nc_path, "incl_01")
        assert log_after1.sizes["Date0"] == 1
        meta2 = {"fileName": "raw/@i_01", "fileChangeTime": np.datetime64("2024-06-01", "ns")}
        nc_incremental_update(ds2, nc_path, "incl_01", meta2)
        log_after2 = read_nc_log(nc_path, "incl_01")
        # Two rows: original + tail end
        assert log_after2.sizes["Date0"] == 2

    def test_incremental_update_with_datetime_meta(self, tmp_path):
        """fileChangeTime as Python datetime (from file_name_and_time_to_record)."""
        ds = _make_ds(20)
        nc_path = tmp_path / "test.raw.nc"
        meta = {"fileName": "raw/@i_01", "fileChangeTime": datetime(2024, 6, 15, 12, 0, 0)}
        assert nc_incremental_update(ds, nc_path, "incl_01", meta) is True
        # Duplicate with same datetime → skip
        assert nc_incremental_update(ds, nc_path, "incl_01", meta) is False

    def test_incremental_overlap_emits_warning(self, tmp_path, caplog):
        """Overlapping new file triggers warning and trim."""
        import logging
        ds1 = _make_ds(50, start="2024-01-01", seed=42)
        ds2 = _make_ds(60, start="2024-01-01 00:00:30", seed=43)  # overlap
        nc_path = tmp_path / "test.raw.nc"
        meta1 = {"fileName": "raw/@i_01", "fileChangeTime": np.datetime64("2024-01-01", "ns")}
        nc_incremental_update(ds1, nc_path, "incl_01", meta1)
        with caplog.at_level(logging.WARNING, logger="tcm._xr.storage"):
            meta2 = {"fileName": "raw/@i_02", "fileChangeTime": np.datetime64("2024-01-02", "ns")}
            nc_incremental_update(ds2, nc_path, "incl_01", meta2)
        assert any("overlap" in r.message.lower() for r in caplog.records)

    def test_incremental_log_preserves_all_columns(self, tmp_path):
        """Log written by nc_incremental_update has all expected columns."""
        ds = _make_ds(20)
        nc_path = tmp_path / "test.raw.nc"
        meta = {"fileName": "raw/@i_01", "fileChangeTime": np.datetime64("2024-01-01", "ns")}
        nc_incremental_update(ds, nc_path, "incl_01", meta)
        log = read_nc_log(nc_path, "incl_01")
        assert "fileName" in log.data_vars
        assert "fileChangeTime" in log.data_vars
        assert "DateEnd" in log.data_vars
        assert "DateProc" in log.data_vars
        assert log["fileName"].values[0] == "raw/@i_01"

    def test_write_to_already_read_on_xarray_fails_with_os_error(self, tmp_path):
        """h5py write to a file that xarray holds a read handle on raises OSError.

        When ``xr.open_dataset`` keeps a netCDF4 backing handle open,
        subsequent ``nc_incremental_update`` (which opens the file via
        h5py) must fail with ``OSError: Unable to synchronously open
        file`` on Windows — HDF5 mandatory locking prevents re-opening
        a file that already has a read-only handle.

        After the reader closes, the h5py write succeeds (proving the
        lock was from the xarray handle, not from corrupt on-disk data).
        """
        # Create seed file with first probe
        ds1 = _make_ds(20, start="2024-01-01")
        nc_path = tmp_path / "locked.raw.nc"
        meta1 = {"fileName": "probeA.txt", "fileChangeTime": np.datetime64("2024-01-01", "ns")}
        nc_incremental_update(ds1, nc_path, "incl_A", meta1)

        # Open the file with xarray (simulates load_raw keeping a handle)
        with xr.open_dataset(nc_path, group="incl_A", engine="netcdf4"):
            ds2 = _make_ds(30, start="2024-02-01")
            meta2 = {"fileName": "probeB.txt", "fileChangeTime": np.datetime64("2024-02-01", "ns")}
            with pytest.raises(OSError, match="already open"):
                nc_incremental_update(ds2, nc_path, "incl_B", meta2)

        # After closing the xarray reader, the write must succeed
        ds2 = _make_ds(30, start="2024-02-01")
        meta2 = {"fileName": "probeB.txt", "fileChangeTime": np.datetime64("2024-02-01", "ns")}
        nc_incremental_update(ds2, nc_path, "incl_B", meta2)


# ---------------------------------------------------------------------------
# store_processed_incremental
# ---------------------------------------------------------------------------

@pytest.mark.xr
class TestStoreProcessedIncremental:

    def test_skip_when_data_covered(self, tmp_path):
        """Skip write when existing group already covers the time range."""
        ds = _make_ds(50)
        path = tmp_path / "test.proc.nc"
        # First write
        store_processed_incremental(ds, path, group="i_p01")
        # Same data again — should skip (return existing path)
        result = store_processed_incremental(ds, path, group="i_p01")
        assert result == path
        # Verify only one group's worth of data
        with xr.open_dataset(path, group="i_p01", engine="netcdf4") as existing:
            assert existing.sizes["time"] == 50

    def test_write_when_new_group(self, tmp_path):
        """Write when group doesn't exist yet."""
        ds = _make_ds(30)
        path = tmp_path / "test.proc.nc"
        store_processed_incremental(ds, path, group="i_p01")
        with xr.open_dataset(path, group="i_p01", engine="netcdf4") as existing:
            assert existing.sizes["time"] == 30

    def test_write_when_data_extends(self, tmp_path):
        """Write when new data extends beyond existing range."""
        ds1 = _make_ds(50, start="2024-01-01")
        ds2 = _make_ds(50, start="2024-01-02")
        path = tmp_path / "test.proc.nc"
        store_processed_incremental(ds1, path, group="i_p01")
        store_processed_incremental(ds2, path, group="i_p01", mode="w")
        with xr.open_dataset(path, group="i_p01", engine="netcdf4") as existing:
            assert existing.sizes["time"] == 50

    def test_handles_int64_time_from_legacy_write(self, tmp_path):
        """Handles int64 time from files written by _write_dataset_to_nc_group."""
        ds = _make_ds(50)
        path = tmp_path / "test.proc.nc"
        # Write using h5py directly (simulates legacy format with int64 ns time)
        _write_dataset_to_nc_group(ds, path, "i_p01")
        # Now store_processed_incremental should handle int64 time without crash
        ds_new = _make_ds(50, start="2024-01-02")
        store_processed_incremental(ds_new, path, group="i_p01", mode="w")

    def test_different_groups_independent(self, tmp_path):
        """Different groups in same file are independent."""
        ds1 = _make_ds(30, seed=42)
        ds2 = _make_ds(40, seed=43)
        path = tmp_path / "test.proc.nc"
        store_processed_incremental(ds1, path, group="i_p01")
        store_processed_incremental(ds2, path, group="i_p02")
        with xr.open_dataset(path, group="i_p01", engine="netcdf4") as r1:
            assert r1.sizes["time"] == 30
        with xr.open_dataset(path, group="i_p02", engine="netcdf4") as r2:
            assert r2.sizes["time"] == 40


# ---------------------------------------------------------------------------
# h5py read/write round-trip (low-level)
# ---------------------------------------------------------------------------

@pytest.mark.xr
class TestNcGroupRoundTrip:
    """_write_dataset_to_nc_group / _read_nc_group_as_dataset round-trip."""

    def test_basic_roundtrip(self, tmp_path):
        """Write then read back via h5py preserves data."""
        ds = _make_ds(50)
        nc_path = tmp_path / "test.nc"
        _write_dataset_to_nc_group(ds, nc_path, "test_grp")
        with h5py.File(nc_path, "r") as f:
            result = _read_nc_group_as_dataset(f, "test_grp")
        assert result.sizes["time"] == 50
        np.testing.assert_allclose(result["Ax"].values, ds["Ax"].values)
        np.testing.assert_array_equal(result["time"].values, ds["time"].values)

    def test_time_is_datetime64_after_read(self, tmp_path):
        """Time coordinate read back is datetime64[ns] (not int64)."""
        ds = _make_ds(10)
        nc_path = tmp_path / "test.nc"
        _write_dataset_to_nc_group(ds, nc_path, "g")
        with h5py.File(nc_path, "r") as f:
            result = _read_nc_group_as_dataset(f, "g")
        assert result["time"].dtype == np.dtype("datetime64[ns]")

    def test_string_vars(self, tmp_path):
        """String data variables survive round-trip."""
        ds = xr.Dataset(
            {"label": ("time", ["a", "b", "c"])},
            coords={"time": xr.date_range("2024-01-01", periods=3, freq="s")},
        )
        nc_path = tmp_path / "test.nc"
        _write_dataset_to_nc_group(ds, nc_path, "g")
        with h5py.File(nc_path, "r") as f:
            result = _read_nc_group_as_dataset(f, "g")
        assert list(result["label"].values) == [b"a", b"b", b"c"]


# ---------------------------------------------------------------------------
# _strip_tz_datetime
# ---------------------------------------------------------------------------

@pytest.mark.xr
class TestStripTzDatetime:

    def test_strips_tz_from_coord(self):
        """tz-aware time coordinate → tz-naive datetime64[ns]."""
        import pandas as pd

        ds = xr.Dataset(
            {"x": ("time", [1.0, 2.0])},
            coords={"time": pd.date_range("2024-01-01", periods=2, tz="UTC")},
        )
        assert ds["time"].dtype.tz is not None
        result = _strip_tz_datetime(ds)
        assert result["time"].dtype == np.dtype("datetime64[ns]")

    def test_naive_unchanged(self):
        """tz-naive data passes through unchanged."""
        ds = xr.Dataset(
            {"x": ("time", [1.0, 2.0])},
            coords={"time": xr.date_range("2024-01-01", periods=2, freq="s")},
        )
        result = _strip_tz_datetime(ds)
        assert result["time"].dtype == np.dtype("datetime64[ns]")
        np.testing.assert_array_equal(result["time"].values, ds["time"].values)

    def test_no_time_coord(self):
        """Dataset without time coord passes through unchanged."""
        ds = xr.Dataset({"x": ("dim", [1, 2, 3])})
        result = _strip_tz_datetime(ds)
        assert result.equals(ds)

    def test_preserves_data_values(self):
        """Stripping tz doesn't change the actual time values."""
        import pandas as pd

        times_utc = pd.date_range("2024-06-15 10:00", periods=5, freq="h", tz="UTC")
        ds = xr.Dataset(
            {"val": ("time", np.arange(5, dtype=float))},
            coords={"time": times_utc},
        )
        result = _strip_tz_datetime(ds)
        # Values should be the same epoch nanoseconds — use pd.DatetimeIndex
        # to get int64 from tz-aware source (numpy .astype(int64) fails on
        # tz-aware object arrays)
        expected_ns = pd.DatetimeIndex(ds["time"].values).to_numpy(dtype="datetime64[ns]").astype(np.int64)
        np.testing.assert_array_equal(
            result["time"].values.astype(np.int64),
            expected_ns,
        )
