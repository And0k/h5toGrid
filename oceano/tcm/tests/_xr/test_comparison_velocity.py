"""TDD comparison tests — numpy reference vs _xr pipeline.

The numpy reference pipeline is the ground truth; _xr must match it.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.polynomial.polynomial import polyval2d

from tcm._xr.physical import add_vabs_vdir, calc_pressure, calc_velocity
from tcm.calibration.orientation import tilt_from_vertical
from tcm.incl_calc.calc import (
    fG,
    polar2dekart,
    v_abs_from_incl,
)

# calc_velocity returns v, u, inclination; Vabs/Vdir are computed on-the-fly for TSV only
_PERSISTED_COLS = ("v", "u", "inclination")


def _reference_velocity_pipeline(
    Ax,
    Ay,
    Az,
    Mx,
    My,
    Mz,
    *,
    Ag,
    Cg,
    Ah,
    Ch,
    kVabs,
    azimuth_shift_deg=0.0,
    calc_version="trigonometric(incl)",
    kVabs_switch_to_linear=None,
):
    """Pure-numpy reference — mirrors old pipeline without dask/despike/recovery."""
    Axyz = np.vstack([Ax, Ay, Az]).astype(float)
    Mxyz = np.vstack([Mx, My, Mz]).astype(float)
    Gxyz = fG(Axyz, Ag, Cg)
    Hxyz = fG(Mxyz, Ah, Ch)
    incl = tilt_from_vertical(Gxyz)
    GsumMinus1 = np.linalg.norm(Gxyz, axis=0) - 1
    Vabs = v_abs_from_incl(incl, kVabs, calc_version=calc_version, kVabs_switch_to_linear=kVabs_switch_to_linear)
    # Vdir formula from old pipeline (with GsumMinus1+1 correction)
    Vdir = azimuth_shift_deg - np.degrees(
        np.arctan2(
            (Gxyz[0, :] * Hxyz[1, :] - Gxyz[1, :] * Hxyz[0, :]) * (GsumMinus1 + 1),
            Hxyz[2, :] * (Gxyz[0, :] ** 2 + Gxyz[1, :] ** 2)
            - Gxyz[2, :] * (Gxyz[0, :] * Hxyz[0, :] + Gxyz[1, :] * Hxyz[1, :]),
        )
    )
    v, u = polar2dekart(Vabs, Vdir)
    return {"Vabs": Vabs, "Vdir": Vdir, "v": v, "u": u, "inclination": np.degrees(incl)}


def _assert_velocity_matches(result, ref, *, atol=1e-10, cols=_PERSISTED_COLS):
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
            sensor_ds.Ax.values,
            sensor_ds.Ay.values,
            sensor_ds.Az.values,
            sensor_ds.Mx.values,
            sensor_ds.My.values,
            sensor_ds.Mz.values,
            **identity_coefs,
        )
        _assert_velocity_matches(calc_velocity(sensor_ds, **identity_coefs), ref)

    def test_simple_calibration(self, sensor_ds, simple_coefs):
        """Non-trivial calibration → _xr matches reference."""
        ref = _reference_velocity_pipeline(
            sensor_ds.Ax.values,
            sensor_ds.Ay.values,
            sensor_ds.Az.values,
            sensor_ds.Mx.values,
            sensor_ds.My.values,
            sensor_ds.Mz.values,
            **simple_coefs,
        )
        _assert_velocity_matches(calc_velocity(sensor_ds, **simple_coefs), ref)

    def test_raw_columns_removed(self, sensor_ds, identity_coefs):
        """After velocity calc, raw Ax..Mz columns should be dropped."""
        result = calc_velocity(sensor_ds, **identity_coefs)
        for col in ("Ax", "Ay", "Az", "Mx", "My", "Mz"):
            assert col not in result, f"Raw column '{col}' not removed"

    def test_with_azimuth_shift(self, sensor_ds, identity_coefs):
        """Azimuth shift adds offset to Vdir (verified via add_vabs_vdir on-the-fly)."""
        coefs = {**identity_coefs, "azimuth_shift_deg": 30.0}
        result = calc_velocity(sensor_ds, **coefs)
        ref = _reference_velocity_pipeline(
            sensor_ds.Ax.values,
            sensor_ds.Ay.values,
            sensor_ds.Az.values,
            sensor_ds.Mx.values,
            sensor_ds.My.values,
            sensor_ds.Mz.values,
            **coefs,
        )
        assert "Vdir" not in result, "Vabs/Vdir should not be in calc_velocity output"
        result_with_vabs = add_vabs_vdir(result)
        np.testing.assert_allclose(result_with_vabs.Vdir.values, ref["Vdir"], atol=1e-10)

    def test_zero_kVabs_no_velocity(self, sensor_ds, identity_coefs):
        """kVabs=None → no velocity columns computed at all."""
        result = calc_velocity(sensor_ds, **{**identity_coefs, "kVabs": None})
        for col in ("Vabs", "Vdir", "v", "u"):
            assert col not in result, f"'{col}' should not exist when kVabs=None"

    def test_Vabs_Vdir_not_persisted(self, sensor_ds, identity_coefs):
        """Vabs/Vdir are NOT in calc_velocity output (computed on-the-fly for TSV only)."""
        result = calc_velocity(sensor_ds, **identity_coefs)
        assert "Vabs" not in result, "Vabs should not be in calc_velocity output"
        assert "Vdir" not in result, "Vdir should not be in calc_velocity output"
        assert "v" in result, "v must be persisted"
        assert "u" in result, "u must be persisted"

    def test_kVabs_switch_to_linear_passthrough(self):
        """calc_velocity forwards kVabs_switch_to_linear to v_abs_from_incl (50° tilt > 45° tangent point)."""
        import pandas as pd
        import xarray as xr

        n, tilt = 10, np.radians(50.0)
        time = pd.date_range("2024-01-01", periods=n, freq="s")
        ds = xr.Dataset(
            {
                "Ax": ("time", np.full(n, np.sin(tilt))),
                "Ay": ("time", np.zeros(n)),
                "Az": ("time", np.full(n, np.cos(tilt))),
                "Mx": ("time", np.full(n, np.cos(tilt))),
                "My": ("time", np.zeros(n)),
                "Mz": ("time", np.full(n, -np.sin(tilt))),
            },
            coords={"time": time},
        )
        coefs = {
            "Ag": np.eye(3),
            "Cg": np.zeros((3, 1)),
            "Ah": np.eye(3),
            "Ch": np.zeros((3, 1)),
            "kVabs": np.array([1.0, 0.5, 0.3, 0.1, 0.05]),
            "azimuth_shift_deg": 0.0,
            "kVabs_switch_to_linear": 45.0,
        }
        ref = _reference_velocity_pipeline(
            ds.Ax.values, ds.Ay.values, ds.Az.values, ds.Mx.values, ds.My.values, ds.Mz.values, **coefs
        )
        _assert_velocity_matches(calc_velocity(ds, **coefs), ref)
        defaulted = calc_velocity(ds, **{**coefs, "kVabs_switch_to_linear": 60.0})
        assert not np.allclose(defaulted["v"].values, ref["v"]), "override must move the tangent point"


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
