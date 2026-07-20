"""
Tests for _xr/calc.py — xr.apply_ufunc wrappers.

Verifies that xarray wrappers produce identical results to the numpy kernels.
"""
import numpy as np
import pytest
import xarray as xr

from tcm._xr import calc as xr_calc
from tcm.calibration import orientation as np_calc

@pytest.mark.xr
class Test_xr_axis_first_reduce:
    def test_matches_numpy(self):
        Gxyz = np.array([[0.1, 0.0, 0.5], [0.0, 0.2, 0.0], [0.99, 0.98, 0.87]])
        da = xr.DataArray(Gxyz, dims=["axis", "time"])
        result = xr_calc._axis_first_reduce(np_calc.tilt_from_vertical)(da)
        expected = np_calc.tilt_from_vertical(Gxyz)
        np.testing.assert_allclose(result.values, expected, atol=1e-12)
