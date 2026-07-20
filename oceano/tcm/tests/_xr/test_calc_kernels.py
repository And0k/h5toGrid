"""
Deterministic tests for low-level math kernels (tcm.incl_calc.calc).

Known inputs → known outputs, no mocking.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tcm.calibration.orientation import tilt_from_vertical
from tcm.incl_calc.calc import (
    fG,
    i_bursts_starts,
    norm_field,
    polar2dekart,
    v_abs_from_incl,
)


# --------------------------------------------------------------------------- #
# fG — calibration matrix application
# --------------------------------------------------------------------------- #

class Test_fG:
    """fG(Axyz, Ag, Cg) = Ag @ (Axyz - Cg)."""

    @pytest.mark.parametrize(
        ("A", "Ag_diag", "Cg", "expected"),
        [
            pytest.param(
                [[1, 2], [3, 4], [5, 6]], [1, 1, 1], [0, 0, 0],
                [[1, 2], [3, 4], [5, 6]], id="identity",
            ),
            pytest.param(
                [[10], [20], [30]], [1, 1, 1], [1, 2, 3],
                [[9], [18], [27]], id="offset-only",
            ),
            pytest.param(
                [[1], [1], [1]], [2, 3, 4], [0, 0, 0],
                [[2], [3], [4]], id="gain-only",
            ),
            pytest.param(
                [[5], [5], [5]], [2, 2, 2], [1, 1, 1],
                [[8], [8], [8]], id="combined",
            ),
        ],
    )
    def test_fG(self, A, Ag_diag, Cg, expected):
        """fG applies gain and offset correctly."""
        result = fG(np.array(A), np.diag(Ag_diag), np.atleast_2d(Cg).T if Cg else np.zeros((3, 1)))
        np.testing.assert_allclose(result, expected)

    def test_1d_Cg(self):
        """Cg as 1-D array is reshaped to (3,1)."""
        result = fG(np.array([[1.0], [2.0], [3.0]]), np.eye(3), np.array([0.1, 0.2, 0.3]))
        np.testing.assert_allclose(result, [[0.9], [1.8], [2.7]])


# --------------------------------------------------------------------------- #
# fInclination — arctan2(||Gxy||, Gz)
# --------------------------------------------------------------------------- #

class Test_fInclination:
    @pytest.mark.parametrize(
        ("Gxyz", "expected"),
        [
            pytest.param([[0], [0], [1]], 0.0, id="vertical"),
            pytest.param([[1], [0], [0]], np.pi / 2, id="horizontal"),
        ],
    )
    def test_known_angles(self, Gxyz, expected):
        result = tilt_from_vertical(np.array(Gxyz, dtype=float)).item()
        assert pytest.approx(result, abs=1e-12) == expected

    def test_45_deg(self):
        Gxyz = np.array([[1.0], [0.0], [1.0]])
        assert pytest.approx(tilt_from_vertical(Gxyz).item(), abs=1e-12) == np.arctan2(1.0, 1.0)


class Test_polar2dekart:
    @pytest.mark.parametrize(
        ("Vabs", "Vdir", "expected_v", "expected_u"),
        [
            pytest.param([0], [0], [0.0], [0.0], id="zero"),
            pytest.param([5], [0], [5.0], [0.0], id="north"),
            pytest.param([5], [90], [0.0], [5.0], id="east"),
        ],
    )
    def test_directions(self, Vabs, Vdir, expected_v, expected_u):
        v, u = polar2dekart(np.array(Vabs, float), np.array(Vdir, float))
        np.testing.assert_allclose(v, expected_v, atol=1e-12)
        np.testing.assert_allclose(u, expected_u, atol=1e-12)


# --------------------------------------------------------------------------- #
# v_abs_from_incl
# --------------------------------------------------------------------------- #

_TRIG_COEFS = np.array([1.0, 0.5, 0.3, 0.1, 0.05, 60.0])  # last = max_incl_of_fit_deg


class Test_v_abs_from_incl:
    def test_empty(self):
        assert len(v_abs_from_incl(np.array([]), [1.0, 0.0], calc_version="polynom(force)")) == 0

    @pytest.mark.parametrize(
        ("incls_deg", "expected_zero"),
        [
            pytest.param([0.0], True, id="zero-incl→zero-vel"),
        ],
    )
    def test_zero_inclination(self, incls_deg, expected_zero):
        result = v_abs_from_incl(np.radians(incls_deg), _TRIG_COEFS, calc_version="trigonometric(incl)")
        assert (pytest.approx(result.item(), abs=1e-12) == 0.0) == expected_zero

    def test_monotonic(self):
        """Velocity should increase with inclination."""
        result = v_abs_from_incl(np.radians([5.0, 10.0, 20.0]), _TRIG_COEFS, calc_version="trigonometric(incl)")
        assert result[0] < result[1] < result[2]


# --------------------------------------------------------------------------- #
# norm_field
# --------------------------------------------------------------------------- #

class Test_norm_field:
    def test_identity_calibration(self):
        raw = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        np.testing.assert_allclose(norm_field(raw, np.eye(3), np.zeros((3, 1))), raw)

    def test_unit_normalization(self):
        """After normalization, |Gxyz| ≈ 1 for well-calibrated data."""
        raw = np.array([[0.0], [0.0], [1000.0]])
        result = norm_field(raw, np.diag([0.001, 0.001, 0.001]), np.zeros((3, 1)))
        np.testing.assert_allclose(np.linalg.norm(result, axis=0), [1.0], atol=1e-12)


# --------------------------------------------------------------------------- #
# i_bursts_starts — burst boundary detection
# --------------------------------------------------------------------------- #

class Test_i_bursts_starts:
    """i_bursts_starts(tim, dt_between_blocks) → (i_bursts, mean_size, max_hole)."""

    def test_uniform_time_no_bursts(self):
        """Uniform 1-s spacing with auto threshold → no gaps, single burst."""
        tim = pd.date_range("2024-01-01", periods=20, freq="s")
        i_b, ms, mh = i_bursts_starts(tim)
        np.testing.assert_array_equal(i_b, [0])
        assert ms == 20
        assert mh == np.timedelta64(0)

    @pytest.mark.parametrize(
        ("dt_blocks", "expected_i_bursts", "expected_mean_size", "expected_max_hole_s"),
        [
            pytest.param(np.timedelta64(5, "s"), [0, 3], 7, 10, id="single-gap"),
            pytest.param(np.inf, [0], 100, 0, id="inf-threshold"),
        ],
    )
    def test_gap_detection(self, dt_blocks, expected_i_bursts, expected_mean_size, expected_max_hole_s):
        """Gap detection with various thresholds."""
        if expected_mean_size == 100:
            # Inf threshold: two widely separated blocks merged
            base = pd.date_range("2024-01-01", periods=50, freq="s")
            tim = base.append(pd.date_range("2025-01-01", periods=50, freq="s"))
        else:
            base = pd.date_range("2024-01-01", periods=3, freq="s")
            tim = base.append(pd.date_range(base[-1] + pd.Timedelta("10s"), periods=7, freq="s"))
        i_b, ms, mh = i_bursts_starts(tim, dt_between_blocks=dt_blocks)
        np.testing.assert_array_equal(i_b, expected_i_bursts)
        assert ms == expected_mean_size
        assert mh == np.timedelta64(expected_max_hole_s, "s")

    def test_empty_input(self):
        """Empty DatetimeIndex → empty array, zero stats."""
        i_b, ms, mh = i_bursts_starts(pd.DatetimeIndex([]))
        np.testing.assert_array_equal(i_b, np.int32([]))
        assert ms == 0
        assert mh == np.timedelta64(0)
