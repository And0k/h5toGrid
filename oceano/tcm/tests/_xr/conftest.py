"""Shared fixtures for _xr tests — synthetic sensor data, calibration,
and pipeline integration infrastructure.
"""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from omegaconf import DictConfig

from tcm._constants import RAW_DIR_NAME
from tcm.config import Return


# ---------------------------------------------------------------------------
# Auto-configure use_h5 for all tests (mirrors processing._resolve_use_h5)
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _auto_use_h5_set():
    """Set use_h5 based on H5_AVAILABLE before each test.

    Resets to the default after the test, preventing state leakage from tests
    that call ``_constants.use_h5_set(value)`` explicitly.
    """
    from tcm import _constants
    _constants.use_h5_set(True if _constants.H5_AVAILABLE else None)
    yield
    _constants.use_h5_set(None)  # reset to unresolved


# --------------------------------------------------------------------------- #
# Synthetic sensor data (10 samples, known geometry)
# --------------------------------------------------------------------------- #

def _make_sensor_data(n: int = 10, tilt_deg: float = 10.0, heading_deg: float = 45.0):
    """Build deterministic Ax,Ay,Az,Mx,My,Mz for a constant-tilt constant-heading instrument."""
    tilt, heading = np.radians(tilt_deg), np.radians(heading_deg)
    Ax = np.sin(tilt) * np.ones(n)
    Ay = np.zeros(n)
    Az = np.cos(tilt) * np.ones(n)
    Mx = np.cos(heading) * np.cos(tilt) * np.ones(n)
    My = np.sin(heading) * np.ones(n)
    Mz = -np.cos(heading) * np.sin(tilt) * np.ones(n)
    return Ax, Ay, Az, Mx, My, Mz


@pytest.fixture
def sensor_df() -> pd.DataFrame:
    """pandas DataFrame with synthetic sensor data (legacy format)."""
    Ax, Ay, Az, Mx, My, Mz = _make_sensor_data()
    idx = pd.date_range("2024-01-01", periods=len(Ax), freq="s", name="time")
    return pd.DataFrame({"Ax": Ax, "Ay": Ay, "Az": Az, "Mx": Mx, "My": My, "Mz": Mz}, index=idx)


@pytest.fixture
def sensor_ds() -> xr.Dataset:
    """xarray Dataset with same synthetic sensor data (_xr format)."""
    Ax, Ay, Az, Mx, My, Mz = _make_sensor_data()
    time = pd.date_range("2024-01-01", periods=len(Ax), freq="s")
    return xr.Dataset(
        {"Ax": ("time", Ax), "Ay": ("time", Ay), "Az": ("time", Az),
         "Mx": ("time", Mx), "My": ("time", My), "Mz": ("time", Mz)},
        coords={"time": time},
    )


@pytest.fixture
def sensor_df_with_pressure() -> pd.DataFrame:
    """pandas DataFrame with sensor + pressure data."""
    Ax, Ay, Az, Mx, My, Mz = _make_sensor_data()
    n = len(Ax)
    idx = pd.date_range("2024-01-01", periods=n, freq="s", name="time")
    return pd.DataFrame(
        {"Ax": Ax, "Ay": Ay, "Az": Az, "Mx": Mx, "My": My, "Mz": Mz,
         "P_counts": np.arange(n, dtype=float),
         "Temp": 20.0 + np.random.default_rng(42).normal(0, 0.1, n)},
        index=idx,
    )


@pytest.fixture
def sensor_ds_with_pressure() -> xr.Dataset:
    """xarray Dataset with sensor + pressure data."""
    Ax, Ay, Az, Mx, My, Mz = _make_sensor_data()
    n = len(Ax)
    time = pd.date_range("2024-01-01", periods=n, freq="s")
    rng = np.random.default_rng(42)
    return xr.Dataset(
        {"Ax": ("time", Ax), "Ay": ("time", Ay), "Az": ("time", Az),
         "Mx": ("time", Mx), "My": ("time", My), "Mz": ("time", Mz),
         "P_counts": ("time", np.arange(n, dtype=float)),
         "Temp": ("time", 20.0 + rng.normal(0, 0.1, n))},
        coords={"time": time},
    )


# --------------------------------------------------------------------------- #
# Pipeline integration infrastructure
# --------------------------------------------------------------------------- #

def _mock_pipeline(cfg, env, mocker, pcid="i_01"):
    """Mock main_init + xr_io.load_raw for pipeline integration tests.

    After the refactor, ``run_processing`` calls ``cli.main_init(cfg)``
    (which converts DictConfig → plain dict) and then ``xr_io.load_raw``
    (replacing the removed ``_load_single``).  This helper mocks both,
    returning a pre-converted dict from ``main_init`` and synthetic data
    from ``load_raw``.

    Also applies PathLayout to the original DictConfig (``cfg.out``) so
    that tests checking ``cfg.out.raw_db_path`` etc. still pass.
    """
    from tcm.paths import PathLayout

    # Resolve output paths on the original DictConfig (mimics main_init behaviour)
    try:
        layout = PathLayout.from_cfg(cfg.input, cfg.out)
        layout.apply_to_cfg(cfg.out)
    except (ValueError, OSError):
        pass  # some test configs lack _raw/ ancestor

    # Build a plain dict that main_init would return (post-conversion form)
    cfg_t = {
        "input": {
            "path": Path(cfg.input.path) if cfg.input.path else None,
            "tables": list(cfg.input.tables) if cfg.input.tables else ["incl*"],
            "coefs_path": cfg.input.get("coefs_path"),
            "coefs": dict(cfg.input.get("coefs") or {}),
            "dt_from_utc": timedelta(seconds=int(cfg.input.get("dt_from_utc") or 0)),
            "corr_time_mode": cfg.input.get("corr_time_mode"),
            "time_ranges": list(cfg.input.get("time_ranges") or []),
            "min": dict(cfg.input.get("min") or {}),
            "max": dict(cfg.input.get("max") or {}),
        },
        "out": {},
        "filter": dict(cfg.get("filter") or {}),
        "program": dict(cfg.get("program") or {"return_": Return.END}),
    }
    # Carry forward all input/out keys (including PathLayout-resolved paths)
    for k, v in cfg.input.items():
        if k not in cfg_t["input"]:
            cfg_t["input"][k] = v
    # out: convert PathLayout-resolved paths from the DictConfig
    for k in list(vars(cfg.out)) if hasattr(cfg.out, '__dict__') else []:
        pass  # handled below via OmegaConf iteration
    for k in ("raw_db_path", "not_joined_db_path", "db_path", "text_path"):
        v = cfg.out.get(k)
        cfg_t["out"][k] = Path(v) if v is not None else None
    for k, v in cfg.out.items():
        if k not in cfg_t["out"]:
            cfg_t["out"][k] = v
    # dt_bins: normalize to timedelta
    cfg_t["out"]["dt_bins"] = [
        timedelta(seconds=int(b)) for b in (cfg_t["out"].get("dt_bins") or [0])
    ]

    mocker.patch("tcm.processing.cli.main_init", return_value=cfg_t)
    mocker.patch(
        "tcm.processing.xr_io.load_raw",
        return_value=(env.synthetic_ds, None),
    )
    mocker.patch("tcm.processing.get_coefs_from_cfg", return_value=env.coefs)


@pytest.fixture()
def mock_pipeline():
    """Callable fixture: patches main_init + xr_io.load_raw + get_coefs_from_cfg."""
    return _mock_pipeline


@pytest.fixture()
def pipeline_env(tmp_path):
    """Standard project layout for pipeline integration tests.

    Directory: ``{tmp_path}/240730_inclinometer/{RAW_DIR_NAME}/@i_01.txt``
    PathLayout resolves::

        raw_db_path        → raw_dir / "240730.raw.nc"
        not_joined_db_path → proc_dir / "240730.proc_noAvg.nc"
        db_path            → proc_dir / "240730.proc.nc"
        text_path          → proc_dir / "text_output"

    Returns :class:`SimpleNamespace` with all paths, synthetic data, coefs,
    and a pre-built :class:`DictConfig` for :func:`run_processing`.
    """
    n = 100
    proc_dir = tmp_path / "240730_inclinometer"
    raw_dir = proc_dir / RAW_DIR_NAME
    raw_dir.mkdir(parents=True)

    # Synthetic CSV file (minimal inclinometer format)
    csv_file = raw_dir / "@i_01.txt"
    Ax, Ay, Az, Mx, My, Mz = _make_sensor_data(n=n)
    time = pd.date_range("2024-01-01", periods=n, freq="s")
    lines = ["yyyy,mm,dd,HH,MM,SS,Ax,Ay,Az,Mx,My,Mz,Battery,Temp"]
    for i in range(n):
        lines.append(
            f"2024,01,01,00,{i // 60:02d},{i % 60:02d},"
            f"{Ax[i]:.6f},{Ay[i]:.6f},{Az[i]:.6f},"
            f"{Mx[i]:.6f},{My[i]:.6f},{Mz[i]:.6f},12.5,25.0"
        )
    csv_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Synthetic Dataset (bypasses CSV parser in tests)
    synthetic_ds = xr.Dataset(
        {"Ax": ("time", Ax), "Ay": ("time", Ay), "Az": ("time", Az),
         "Mx": ("time", Mx), "My": ("time", My), "Mz": ("time", Mz),
         "Battery": ("time", np.full(n, 12.5)),
         "Temp": ("time", np.full(n, 25.0))},
        coords={"time": time},
    )

    # Known coefs (constant-tilt constant-heading geometry)
    coefs = {
        "Ag": np.eye(3) * 0.00173, "Cg": np.zeros(3),
        "Ah": np.eye(3), "Ch": np.zeros(3),
        "kVabs": np.array([1.0, 0.0, 0.5]),
        "azimuth_shift_deg": 180.0,
    }

    return SimpleNamespace(
        proc_dir=proc_dir, raw_dir=raw_dir, csv_file=csv_file,
        synthetic_ds=synthetic_ds, coefs=coefs, cfg=DictConfig({
            "input": {
                "path": str(csv_file), "coefs_path": None, "coefs": {},
                "tables": ["incl*"], "text_type": None, "text_line_regex": None,
                "prefix": None, "dt_from_utc": 0,
                "corr_time_mode": None,  # moved from filter
            },
            "out": {"dt_bins": [0, 2, 600]},
            "filter": {},
            "program": {"return_": Return.END},
        }),
        raw_db_path=raw_dir / "240730.raw.nc",
        noavg_path=proc_dir / "240730.proc_noAvg.nc",
        avg_path=proc_dir / "240730.proc.nc",
        text_path=proc_dir / "text_output",
    )
