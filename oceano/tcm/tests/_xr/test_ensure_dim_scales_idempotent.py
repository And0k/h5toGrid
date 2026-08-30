"""Test ensure_dim_scales is truly idempotent (no file growth on re-call)."""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tcm._xr.storage import ensure_dim_scales


@pytest.mark.xr
class TestEnsureDimScalesIdempotent:
    def test_no_growth_on_repeated_calls(self, tmp_path):
        """Calling ensure_dim_scales twice must not increase file size."""
        p = tmp_path / "test.nc"
        ds = xr.Dataset(
            {"u": ("time", np.arange(10, dtype=float))},
            coords={"time": pd.date_range("2024-01-01", periods=10, freq="s")},
        )
        ds.to_netcdf(p, engine="h5netcdf")
        ensure_dim_scales(p)  # first call — attach scales

        size_after_first = p.stat().st_size
        ensure_dim_scales(p)  # second call — must be no-op
        size_after_second = p.stat().st_size

        assert size_after_second == size_after_first, (
            f"File grew on second ensure_dim_scales call: "
            f"{size_after_first} -> {size_after_second} (+{size_after_second - size_after_first})"
        )

    def test_no_growth_on_real_file(self):
        """Verify on the actual run1 proc_Avg.nc (if available)."""
        src = Path(r"B:\Cruises\BalticSea\260711_Pionerskiy@i\run1\260711.proc_Avg.nc")
        if not src.exists():
            pytest.skip("Real data not available")
        tmp = Path(r"C:\Users\User\AppData\Local\Temp\kilo") / "test_real.nc"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, tmp)

        size_before = tmp.stat().st_size
        ensure_dim_scales(tmp)
        size_after = tmp.stat().st_size

        assert size_after == size_before, (
            f"ensure_dim_scales grew real file: {size_before} -> {size_after} (+{size_after - size_before})"
        )
