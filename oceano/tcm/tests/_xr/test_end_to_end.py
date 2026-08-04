"""End-to-end test: synthetic sensor Dataset → physical processing → netCDF roundtrip."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest
import xarray as xr

from tcm._xr.physical import add_vabs_vdir
from tcm.processing import process_inmemory as run

# calc_velocity persists v, u, inclination; Vabs/Vdir computed on-the-fly for TSV only
_PERSISTED_VELOCITY_COLS = ("v", "u", "inclination")


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
        """sensor_ds → run(dt=0) → netCDF roundtrip.

        Vabs/Vdir are intentionally absent from NC and in-memory results;
        they are computed on-the-fly only for per-probe TSV via add_vabs_vdir.
        """
        nc_path = tmp_path / "output.nc"

        results = run(
            sensor_ds,
            coefs=identity_coefs,
            dt_bins=[timedelta(0)],
            out_path=nc_path,
        )

        assert len(results) == 1
        ds = results[0]
        assert ds is not None

        # Contains persisted velocity columns (Vabs/Vdir intentionally absent)
        for col in _PERSISTED_VELOCITY_COLS:
            assert col in ds, f"Missing '{col}' in processed output"
        for col in ("Vabs", "Vdir"):
            assert col not in ds, f"'{col}' must not persist in NC/in-memory output"

        # Vabs/Vdir recoverable on-the-fly from v/u
        ds_with_vabs = add_vabs_vdir(ds)
        assert "Vabs" in ds_with_vabs, "add_vabs_vdir should produce Vabs"
        assert "Vdir" in ds_with_vabs, "add_vabs_vdir should produce Vdir"

        # netCDF file exists and roundtrips (without Vabs/Vdir)
        assert nc_path.exists()
        with xr.open_dataset(nc_path) as loaded:
            for var in ds.data_vars:
                np.testing.assert_allclose(
                    loaded[var].values,
                    ds[var].values,
                    atol=1e-10,
                    err_msg=f"Roundtrip mismatch for '{var}'",
                )
