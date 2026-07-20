"""E2E tests against real CSV inclinometer data — no mocking load_raw.

Exercises the full pipeline: CSV load → physical.process → store_processed
→ NC output + TSV export.  Uses the real file format with integer-second
timestamps at ~10 Hz.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest
import xarray as xr
from omegaconf import DictConfig

from tcm._constants import RAW_DIR_NAME
from tcm._xr import io as xr_io
from tcm.config import Return
from tcm.processing import run_processing

# ---------------------------------------------------------------------------
# Real data fixture
# ---------------------------------------------------------------------------

_REAL_FILE = Path(r"D:\WorkData\experiment\inclinometer\260604_test_format\_raw\@i_p1.TXT")

_COEFS = {
    "Ag": [[0.00173, 0, 0], [0, 0.00173, 0], [0, 0, 0.00173]],
    "Cg": [0, 0, 0],
    "Ah": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "Ch": [0, 0, 0],
    "kVabs": [10, -10, -10, -3, 3, 70],
    "azimuth_shift_deg": 180,
}


@pytest.fixture()
def real_env(tmp_path):
    """Copy real CSV into a tmp project layout, build cfg."""
    if not _REAL_FILE.exists():
        pytest.skip(f"Real data file not found: {_REAL_FILE}")

    proc_dir = tmp_path / "260604_test_format"
    raw_dir = proc_dir / RAW_DIR_NAME
    raw_dir.mkdir(parents=True)
    csv_file = raw_dir / _REAL_FILE.name
    shutil.copy2(_REAL_FILE, csv_file)

    cfg = DictConfig({
        "input": {
            "path": str(csv_file),
            "coefs_path": None,
            "coefs": _COEFS,
            "tables": ["incl*"],
            "text_type": None,
            "text_line_regex": None,
            "prefix": None,
            "dt_from_utc": 0,
        },
        "out": {"dt_bins": [0, 2, 600], "dir": str(proc_dir)},
        "filter": {},  # corr_time_mode=True is now in input (default) → snap-to-grid preserves 10Hz
        "program": {"return_": Return.END, "verbose": "INFO"},
    })

    return dict(
        proc_dir=proc_dir, raw_dir=raw_dir, csv_file=csv_file, cfg=cfg,
        noavg_path=proc_dir / "260604.proc_noAvg.nc",
        avg_path=proc_dir / "260604.proc.nc",
        text_path=proc_dir / "text_output",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.xr
class TestE2ERealCSV:
    """Full pipeline with real CSV data — no mocked load_raw."""

    def test_load_raw_returns_datetime64_time(self, real_env):
        """xr_io.load_raw must produce datetime64 time (may be tz-aware)."""
        cfg_in = real_env["cfg"].input
        ds, _coefs = xr_io.load_raw(
            cfg_in["path"], tbl="incl.*", cfg_in=cfg_in,
        )
        assert ds is not None, "load_raw returned None"
        assert hasattr(ds.time.dtype, "kind") and ds.time.dtype.kind == "M", (
            f"time must be datetime64, got {ds.time.dtype}"
        )

    def test_proc_noavg_time_is_datetime64(self, real_env):
        """proc_noAvg.nc time must decode as datetime64, not raw integers."""

        run_processing(real_env["cfg"])

        noavg = real_env["noavg_path"]
        assert noavg.exists(), (
            f"proc_noAvg.nc not created; dir: {list(real_env['proc_dir'].iterdir())}"
        )

        with xr.open_dataset(noavg, group="i_p01", engine="netcdf4") as ds:
            assert hasattr(ds.time.dtype, "kind") and ds.time.dtype.kind == "M", (
                f"proc_noAvg.nc time must be datetime64, got {ds.time.dtype}; "
                f"time[:5]={ds.time.values[:5]}"
            )

    def test_proc_nc_has_binned_groups(self, real_env):
        """proc.nc must contain binned groups (bin2s, bin600s)."""

        run_processing(real_env["cfg"])

        avg = real_env["avg_path"]
        assert avg.exists(), (
            f"proc.nc not created; dir: {list(real_env['proc_dir'].iterdir())}"
        )
        with h5py.File(str(avg), "r") as f:
            groups = list(f.keys())
            assert any("bin2s" in g for g in groups), (
                f"bin2s group missing from proc.nc: {groups}"
            )

    def test_tsv_files_created(self, real_env):
        """TSV files must appear for bins >= dt_bins_min_save_text."""

        run_processing(real_env["cfg"])

        text_dir = Path(real_env["cfg"].out.get("text_path", real_env["proc_dir"]))
        tsv_files = list(text_dir.glob("*.tsv"))
        assert tsv_files, (
            f"No TSV files in {text_dir}; proc_dir: {list(real_env['proc_dir'].iterdir())}"
        )
        assert any("bin2s" in f.name for f in tsv_files), (
            f"bin2s TSV missing: {[f.name for f in tsv_files]}"
        )

    def test_10hz_time_preserved(self, real_env):
        """10Hz integer-second data must be snapped to ~100ms grid, not collapsed to 1Hz."""

        run_processing(real_env["cfg"])

        noavg = real_env["noavg_path"]
        assert noavg.exists(), "proc_noAvg.nc not created"

        with xr.open_dataset(noavg, group="i_p01", engine="netcdf4") as ds:
            diffs_ns = np.diff(ds.time.values.astype(np.int64))
            median_ms = np.median(diffs_ns) / 1e6
            assert median_ms < 500, (
                f"Time collapsed to {median_ms:.0f}ms steps — expected ~100ms for 10Hz. "
                f"corr_time_mode snap-to-grid not wired."
            )
