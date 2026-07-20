"""
Tests for :func:`tcm.calibration.orientation.azimuth_shift` — magnetic North
azimuth detection from calibrated unit vectors.

Replaces the former ``calibration.zeroing.find_azimuth_shift`` which used
velocity/magnitude calculation (``kVabs``).
"""
from __future__ import annotations

import numpy as np
import pytest

from tcm.calibration.calibrate import SensorCalibration
from tcm.calibration.orientation import azimuth_shift


IDENTITY_CAL = SensorCalibration(bias=np.zeros((3, 1)), a2d=np.eye(3))


# --------------------------------------------------------------------------- #
# Helpers — synthetic instrument at known heading
# --------------------------------------------------------------------------- #

def _make_heading_samples(
    heading_deg: float,
    tilt_deg: float = 10.0,
    n: int = 500,
    noise: float = 0.01,
    rng_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(mag_raw, accel_raw)`` as (3, N) arrays for an instrument at *heading_deg*."""
    rng = np.random.default_rng(rng_seed)
    tilt = np.radians(tilt_deg)
    heading = np.radians(heading_deg)

    # Gravity in instrument frame (tilted around Y axis)
    Ax = np.sin(tilt) * np.ones(n)
    Ay = np.zeros(n)
    Az = np.cos(tilt) * np.ones(n)

    # Magnetic field in instrument frame (horizontal component along heading)
    Mx = np.cos(heading) * np.cos(tilt) * np.ones(n)
    My = np.sin(heading) * np.ones(n)
    Mz = -np.cos(heading) * np.sin(tilt) * np.ones(n)

    accel = np.vstack([Ax, Ay, Az]) + rng.normal(scale=noise, size=(3, n))
    mag = np.vstack([Mx, My, Mz]) + rng.normal(scale=noise, size=(3, n))
    return mag, accel


# --------------------------------------------------------------------------- #
# azimuth_shift — core algorithm
# --------------------------------------------------------------------------- #

@pytest.mark.calibration
class TestAzimuthShift:
    """``azimuth_shift(mag, cal, accel, cal)`` returns azimuth in degrees."""

    def test_north_heading(self):
        """Instrument pointing North (heading=0°) → azimuth shift ≈ 0°."""
        mag, accel = _make_heading_samples(heading_deg=0.0)
        shift = azimuth_shift(mag, IDENTITY_CAL, accel, IDENTITY_CAL)
        assert np.isfinite(shift)
        assert abs(shift) < 5.0, f"North heading shift should be ~0°, got {shift:.1f}°"

    def test_east_heading(self):
        """Instrument pointing East (heading=90°) → azimuth shift ≈ +90°."""
        mag, accel = _make_heading_samples(heading_deg=90.0)
        shift = azimuth_shift(mag, IDENTITY_CAL, accel, IDENTITY_CAL)
        assert np.isfinite(shift)
        assert abs(shift - 90.0) < 5.0, f"East heading shift should be ~90°, got {shift:.1f}°"

    def test_returns_float(self):
        """Return type is float."""
        mag, accel = _make_heading_samples(0.0, n=100)
        result = azimuth_shift(mag, IDENTITY_CAL, accel, IDENTITY_CAL)
        assert isinstance(result, float)

    def test_consistency(self):
        """Same input → same output (deterministic)."""
        mag1, accel1 = _make_heading_samples(45.0, n=200, rng_seed=123)
        mag2, accel2 = _make_heading_samples(45.0, n=200, rng_seed=123)
        s1 = azimuth_shift(mag1, IDENTITY_CAL, accel1, IDENTITY_CAL)
        s2 = azimuth_shift(mag2, IDENTITY_CAL, accel2, IDENTITY_CAL)
        assert s1 == pytest.approx(s2)

    def test_varying_heading(self):
        """Different headings produce different azimuth shifts."""
        mag_n, accel_n = _make_heading_samples(0.0, n=200)
        mag_e, accel_e = _make_heading_samples(90.0, n=200)
        s_n = azimuth_shift(mag_n, IDENTITY_CAL, accel_n, IDENTITY_CAL)
        s_e = azimuth_shift(mag_e, IDENTITY_CAL, accel_e, IDENTITY_CAL)
        assert s_n != pytest.approx(s_e, abs=1.0)

    @pytest.mark.parametrize("heading", [0, 30, 90, 180, 270])
    def test_heading_roundtrip(self, heading: int):
        """azimuth_shift recovers the heading within noise tolerance."""
        mag, accel = _make_heading_samples(heading_deg=heading, noise=0.001, n=1000)
        shift = azimuth_shift(mag, IDENTITY_CAL, accel, IDENTITY_CAL)
        # heading + shift should point to magnetic North (~0° or ~360°)
        # shift itself IS the bearing of North from forward axis

        # Normalized angular difference
        diff = (shift - heading) % 360
        angular_diff = diff if diff <= 180 else diff - 360

        assert angular_diff == pytest.approx(0, abs=3.0), (
            f"Heading {heading}°: expected shift ≈ {heading}°, got {shift:.1f}°"
        )


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #

@pytest.mark.calibration
class TestAzimuthShiftEdgeCases:

    def test_vertical_instrument(self):
        """Instrument pointing straight up — horizontal projection degenerate, but no crash."""
        n = 100
        accel = np.tile([[0.0], [0.0], [1.0]], (1, n))
        mag = np.tile([[1.0], [0.0], [0.0]], (1, n))
        # May produce NaN for degenerate geometry — just check no crash
        result = azimuth_shift(mag, IDENTITY_CAL, accel, IDENTITY_CAL)
        assert isinstance(result, float)

    def test_small_dataset(self):
        """Very small dataset — no crash."""
        mag, accel = _make_heading_samples(0.0, n=5)
        result = azimuth_shift(mag, IDENTITY_CAL, accel, IDENTITY_CAL)
        assert np.isfinite(result) or np.isnan(result)
