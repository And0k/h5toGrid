"""Shared NC utilities — time encoding

Extracted from :mod:`storage` so that both :mod:`storage` and :mod:`store_params`
can import them without circular dependencies.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from tcm._constants import _h5py

# CF-standard time encoding — single source of truth for all NC I/O.
EPOCH_NS: int = np.datetime64((_ := "1970-01-01"), "ns").astype(np.int64)
CF_TIME_UNITS = f"seconds since {_}"
CF_CALENDAR = "proleptic_gregorian"


def dt_ns_to_cf(arr: np.ndarray) -> np.ndarray:
    """datetime64[ns] → float64 seconds since epoch (for h5py write)."""
    return (arr.astype("datetime64[ns]").astype(np.int64) - EPOCH_NS) / 1e9


def cf_to_dt_ns(raw: np.ndarray, units: str | bytes = "") -> np.ndarray:
    """float64 seconds OR legacy int64 ns → datetime64[ns] (for h5py read)."""
    if isinstance(units, bytes):
        units = units.decode()
    if raw.dtype.kind == "f":
        epoch_ns = np.datetime64(units.removeprefix("seconds since "), "ns").astype(np.int64) if units else 0
        return ((raw * 1e9).astype(np.int64) + epoch_ns).astype("datetime64[ns]")
    return raw.astype(np.int64).astype("datetime64[ns]")


def write_time_ds(grp: _h5py.Group, name: str, data: np.ndarray) -> None:
    """Create a CF-standard time dataset (float64, ``"seconds since ..."``)."""
    ds = grp.create_dataset(name, data=data, dtype="f8")
    ds.attrs["units"] = CF_TIME_UNITS
    ds.attrs["calendar"] = CF_CALENDAR


# ---------------------------------------------------------------------------
# Encoding helpers — compression, float32 downcast, epoch forcing.
# ---------------------------------------------------------------------------

_ZLIB_CFG: dict[str, Any] = {"zlib": True, "complevel": 9, "shuffle": True, "fletcher32": True}


def force_epoch(ds: xr.Dataset) -> dict[str, Any]:
    """Return xarray encoding dict forcing CF-standard epoch on ``time`` coord."""
    if "time" not in ds.coords:
        return {}
    return {"time": {"units": CF_TIME_UNITS, "calendar": CF_CALENDAR, "dtype": "f8"}}


def downcast_float32(ds: xr.Dataset) -> xr.Dataset:
    """Convert all float64 data variables to float32 in-place."""
    for name, da in ds.data_vars.items():
        if da.dtype == np.float64:
            ds[name] = da.astype(np.float32)
    return ds


def strip_tz_datetime(ds: xr.Dataset) -> xr.Dataset:
    """Convert tz-aware datetime64 coordinates/vars to naive.

    ``datetime64[ns, UTC]`` is a pandas extension dtype that numpy cannot
    interpret (TypeError in ``np.issubdtype``).  xarray's netCDF writer
    hits this during CF encoding, so we strip timezone before persist.

    Uses ``pd.DatetimeIndex.tz_localize(None)`` to drop timezone info,
    then converts to ``datetime64[ns]`` for netCDF compatibility.
    """
    for coord_name in list(ds.coords):
        coord = ds.coords[coord_name]
        try:
            if coord.dtype.tz is None:
                continue
        except (TypeError, AttributeError):
            continue
        naive = pd.DatetimeIndex(coord.values).tz_localize(None).to_numpy(dtype="datetime64[ns]")
        ds = ds.assign_coords({coord_name: naive})

    # Fix tz-aware data variables (uncommon but defensive)
    for var_name in list(ds.data_vars):
        var = ds[var_name]
        try:
            if var.dtype.tz is None:
                continue
        except (TypeError, AttributeError):
            continue
        naive = pd.DatetimeIndex(var.values).tz_localize(None).to_numpy(dtype="datetime64[ns]")
        ds[var_name] = naive

    return ds


def compression_encoding(ds: xr.Dataset) -> dict[str, dict[str, Any]]:
    """Return xarray encoding dict with zlib compression for each data variable."""
    return {
        name: {**_ZLIB_CFG, "dtype": "float32"}
        for name, da in ds.data_vars.items()
        if da.dtype.kind in ("f", "i", "u")
    }
