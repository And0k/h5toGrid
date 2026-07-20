"""End-to-end test: synthetic sensor Dataset → physical processing → netCDF roundtrip."""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import pytest
import xarray as xr

from tcm.processing import process_inmemory as run

_VELOCITY_COLS = ("Vabs", "Vdir", "v", "u", "inclination")


def _make_sensor_data(n: int = 10, tilt_deg: float = 10.0, heading_deg: float = 45.0):
    """Build deterministic Ax,Ay,Az,Mx,My,Mz for a constant-tilt constant-heading instrument."""
    from numpy import radians
    tilt, heading = radians(tilt_deg), radians(heading_deg)
    Ax = np.sin(tilt) * np.ones(n)
    Ay = np.zeros(n)
    Az = np.cos(tilt) * np.ones(n)
    Mx = np.cos(heading) * np.cos(tilt) * np.ones(n)
    My = np.sin(heading) * np.ones(n)
    Mz = -np.cos(heading) * np.sin(tilt) * np.ones(n)
    return Ax, Ay, Az, Mx, My, Mz


@pytest.mark.xr
class TestEndToEnd:
    def test_full_sensor_data(self, sensor_ds, identity_coefs, tmp_path):
        """sensor_ds → run(dt=0) → netCDF roundtrip."""
        tmpdir = Path(mkdtemp())
        nc_path = tmpdir / "output.nc"

        results = run(
            sensor_ds,
            coefs=identity_coefs,
            dt_bins=[timedelta(0)],
            out_path=nc_path,
        )

        assert len(results) == 1
        ds = results[0]
        assert ds is not None

        # Contains velocity columns
        for col in _VELOCITY_COLS:
            assert col in ds, f"Missing '{col}' in processed output"

        # netCDF file exists and roundtrips
        assert nc_path.exists()
        loaded = xr.open_dataset(nc_path)
        for var in ds.data_vars:
            np.testing.assert_allclose(
                loaded[var].values, ds[var].values, atol=1e-10,
                err_msg=f"Roundtrip mismatch for '{var}'",
            )
        loaded.close()
