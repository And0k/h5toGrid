"""TDD comparison tests — numpy reference vs _xr pipeline.

The numpy reference pipeline is the ground truth; _xr must match it.
"""
from __future__ import annotations

import numpy as np
import pytest
from numpy.polynomial.polynomial import polyval2d

from tcm._xr.physical import calc_pressure, calc_velocity
from tcm.calibration.orientation import tilt_from_vertical
from tcm.incl_calc.calc import (
    fG,
    polar2dekart,
    v_abs_from_incl,
)

_VELOCITY_COLS = ("Vabs", "Vdir", "v", "u", "inclination")


def _reference_velocity_pipeline(
    Ax, Ay, Az, Mx, My, Mz, *, Ag, Cg, Ah, Ch, kVabs, azimuth_shift_deg=0.0,
    calc_version="trigonometric(incl)",
):
    """Pure-numpy reference — mirrors _dask_legacy without dask/despike/recovery."""
    Axyz = np.vstack([Ax, Ay, Az]).astype(float)
    Mxyz = np.vstack([Mx, My, Mz]).astype(float)
    Gxyz = fG(Axyz, Ag, Cg)
    Hxyz = fG(Mxyz, Ah, Ch)
    incl = tilt_from_vertical(Gxyz)
    GsumMinus1 = np.linalg.norm(Gxyz, axis=0) - 1
    Vabs = v_abs_from_incl(incl, kVabs, calc_version=calc_version)
    # Vdir formula from _dask_legacy (with GsumMinus1+1 correction)
    Vdir = azimuth_shift_deg - np.degrees(np.arctan2(
        (Gxyz[0, :] * Hxyz[1, :] - Gxyz[1, :] * Hxyz[0, :]) * (GsumMinus1 + 1),
        Hxyz[2, :] * (Gxyz[0, :] ** 2 + Gxyz[1, :] ** 2)
        - Gxyz[2, :] * (Gxyz[0, :] * Hxyz[0, :] + Gxyz[1, :] * Hxyz[1, :])
    ))
    v, u = polar2dekart(Vabs, Vdir)
    return {"Vabs": Vabs, "Vdir": Vdir, "v": v, "u": u, "inclination": np.degrees(incl)}


def _assert_velocity_matches(result, ref, *, atol=1e-10, cols=_VELOCITY_COLS):
    """Assert xr calc_velocity output matches numpy reference for each column."""
    for col in cols:
        assert col in result, f"Missing column '{col}'"
        np.testing.assert_allclose(result[col].values, ref[col], atol=atol, err_msg=f"Mismatch in '{col}'")


# --------------------------------------------------------------------------- #
# _xr vs numpy reference
# --------------------------------------------------------------------------- #

@pytest.mark.xr
@pytest.mark.comparison
class TestVelocityComparison:
    """_xr/physical.py::calc_velocity must match numpy reference."""

    def test_identity_calibration(self, sensor_ds, identity_coefs):
        """Clean data, identity calibration → _xr matches reference."""
        ref = _reference_velocity_pipeline(
            sensor_ds.Ax.values, sensor_ds.Ay.values, sensor_ds.Az.values,
            sensor_ds.Mx.values, sensor_ds.My.values, sensor_ds.Mz.values,
            **identity_coefs,
        )
        _assert_velocity_matches(calc_velocity(sensor_ds, **identity_coefs), ref)

    def test_simple_calibration(self, sensor_ds, simple_coefs):
        """Non-trivial calibration → _xr matches reference."""
        ref = _reference_velocity_pipeline(
            sensor_ds.Ax.values, sensor_ds.Ay.values, sensor_ds.Az.values,
            sensor_ds.Mx.values, sensor_ds.My.values, sensor_ds.Mz.values,
            **simple_coefs,
        )
        _assert_velocity_matches(calc_velocity(sensor_ds, **simple_coefs), ref)

    def test_raw_columns_removed(self, sensor_ds, identity_coefs):
        """After velocity calc, raw Ax..Mz columns should be dropped."""
        result = calc_velocity(sensor_ds, **identity_coefs)
        for col in ("Ax", "Ay", "Az", "Mx", "My", "Mz"):
            assert col not in result, f"Raw column '{col}' not removed"

    def test_with_azimuth_shift(self, sensor_ds, identity_coefs):
        """Azimuth shift adds offset to Vdir."""
        coefs = {**identity_coefs, "azimuth_shift_deg": 30.0}
        result = calc_velocity(sensor_ds, **coefs)
        ref = _reference_velocity_pipeline(
            sensor_ds.Ax.values, sensor_ds.Ay.values, sensor_ds.Az.values,
            sensor_ds.Mx.values, sensor_ds.My.values, sensor_ds.Mz.values,
            **coefs,
        )
        np.testing.assert_allclose(result.Vdir.values, ref["Vdir"], atol=1e-10)

    def test_zero_kVabs_no_velocity(self, sensor_ds, identity_coefs):
        """kVabs=None → no Vabs/Vdir/v/u computed."""
        result = calc_velocity(sensor_ds, **{**identity_coefs, "kVabs": None})
        for col in ("Vabs", "Vdir", "v", "u"):
            assert col not in result, f"'{col}' should not exist when kVabs=None"


# --------------------------------------------------------------------------- #
# Pressure comparison
# --------------------------------------------------------------------------- #

@pytest.mark.xr
@pytest.mark.comparison
class TestPressureComparison:
    """_xr/physical.py::calc_pressure vs numpy polyval2d reference."""

    def test_pressure_identity(self, sensor_ds_with_pressure):
        """P_t=None → pressure column unchanged."""
        assert "P_counts" in calc_pressure(sensor_ds_with_pressure, P_t=None)

    def test_pressure_simple_poly(self, sensor_ds_with_pressure):
        """Simple polynomial conversion matches numpy reference."""
        ds = sensor_ds_with_pressure
        P_t = np.array([[2.0, 0.0], [0.0, 0.0]])
        result = calc_pressure(ds, P_t=P_t)
        assert "Pressure" in result
        assert "P_counts" not in result
        expected = polyval2d(ds.P_counts.values.astype(float), ds.Temp.values, P_t)
        np.testing.assert_allclose(result.Pressure.values, expected, atol=1e-10)
