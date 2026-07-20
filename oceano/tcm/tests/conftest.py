"""Shared fixtures for tcm tests — coefficients, configs, and DataFrames."""
from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Coefficient fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def coefs_basic() -> Dict[str, Any]:
    """Minimal coefficient dict returned by incl_calc.coef_prepare."""
    return {
        "date": pd.Timestamp("2024-06-01"),
        "kx": 1.0, "ky": 1.0, "kz": 1.0,
    }


@pytest.fixture
def coef_zeroing_matrix_basic():
    """Minimal zeroing matrix (identity-like)."""
    return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


# ---------------------------------------------------------------------------
# Calibration coefficients (used by _xr and _calibration)
# ---------------------------------------------------------------------------

@pytest.fixture
def identity_coefs() -> Dict[str, Any]:
    """Identity calibration (no rotation, no offset)."""
    return {
        "Ag": np.eye(3),
        "Cg": np.zeros((3, 1)),
        "Ah": np.eye(3),
        "Ch": np.zeros((3, 1)),
        "kVabs": np.array([1.0, 0.0, 0.5]),
        "azimuth_shift_deg": 0.0,
    }


@pytest.fixture
def simple_coefs() -> Dict[str, Any]:
    """Non-trivial but simple calibration for testing arithmetic."""
    return {
        "Ag": np.diag([1.1, 1.1, 1.0]),
        "Cg": np.array([[0.01], [0.02], [0.0]]),
        "Ah": np.eye(3),
        "Ch": np.zeros((3, 1)),
        "kVabs": np.array([1.0, 0.0, 0.5]),
        "azimuth_shift_deg": 0.0,
    }


# ---------------------------------------------------------------------------
# Config fragments
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg_in_minimal() -> Dict[str, Any]:
    """Minimal ``cfg["in"]`` fragment required by :func:`cur_cfg`."""
    return {
        "path": None,
        "tables": ["incl*"],
        "coefs_path": None,
        "coefs": {},
        "dt_min_binning_proc": pd.Timedelta("2s"),
        "max_incl_of_fit_deg": 5.0,
        "calc_version": 1,
    }


@pytest.fixture
def cfg_out_minimal() -> Dict[str, Any]:
    """Minimal ``cfg["out"]`` fragment."""
    return {
        "dt_bins": [timedelta(0), timedelta(seconds=60)],
        "table": "",
        "db_path": None,
        "split_period": None,
    }


@pytest.fixture
def cfg_filter_minimal() -> Dict[str, Any]:
    """Minimal ``cfg["filter"]`` fragment."""
    return {"min": {"M": 0}, "max": {"M": 100}}


@pytest.fixture
def cfg_full(cfg_in_minimal, cfg_out_minimal, cfg_filter_minimal) -> Dict[str, Any]:
    """Top-level config dict combining in/out/filter."""
    return {
        "in": cfg_in_minimal,
        "out": cfg_out_minimal,
        "filter": cfg_filter_minimal,
        "program": {"return_after_saving_to_raw_db": False, "dask_scheduler": None},
    }


# ---------------------------------------------------------------------------
# DataFrame fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def df_simple() -> pd.DataFrame:
    """Small DataFrame with datetime index for testing."""
    return pd.DataFrame(
        {"v": [1.0, 2.0, 3.0], "u": [4.0, 5.0, 6.0]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )


@pytest.fixture
def df_two_probes() -> Dict[str, pd.DataFrame]:
    """Two DataFrames simulating different probes, keyed by pcid."""
    idx = pd.to_datetime(["2024-01-01", "2024-01-02"])
    return {
        "i_01": pd.DataFrame({"v": [1.0, 2.0], "u": [3.0, 4.0]}, index=idx),
        "i_02": pd.DataFrame({"v": [10.0, 20.0], "u": [30.0, 40.0]}, index=idx),
    }
