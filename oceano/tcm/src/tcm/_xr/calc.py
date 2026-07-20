"""
- Binning
- Helper that wraps low-level math kernels with ``xr.apply_ufunc``.
"""

from datetime import timedelta
from functools import wraps
from typing import Callable, Optional

import numpy as np
import pandas as pd
import xarray as xr
# from tcm._xr.storage import _time_values_to_int64

def _time_values_to_int64(time_coord: xr.DataArray) -> np.ndarray:
    """Extract int64 nanoseconds from time coordinate (tz-aware safe).

    Routes through ``pd.DatetimeIndex`` for tz-aware extension dtypes;
    fast numpy path for naive ``datetime64``.
    """
    try:
        if time_coord.dtype.tz is not None:
            return pd.DatetimeIndex(time_coord.values).to_numpy(dtype="datetime64[ns]").astype(np.int64)
    except (TypeError, AttributeError):
        pass
    return time_coord.values.astype("datetime64[ns]").astype(np.int64)


def _resample_unify(
    ds: xr.Dataset,
    dt_ns: int,
) -> tuple[xr.Dataset, np.ndarray]:
    """
    Per-bin ``(sum, count)`` → deferred mean. Boundary-safe across chunks.

    defers the final ``sum / count`` division so that chunk-boundary-spanning
    bins are aggregated correctly (no premature averaging of partial means).

    Parameters
    ----------
    ds
        Input dataset with ``time`` coordinate (must be sorted).
    dt_ns
        Bin width in integer nanoseconds.

    Returns
    -------
    result
        Binned dataset (NaN where count == 0).
    n_good
        Valid-sample count per bin (first data variable).
    """
    import dask.array as da

    first_var: str = next(iter(ds.data_vars))
    is_dask: bool = isinstance(ds[first_var].data, da.Array)
    b_lib = da if is_dask else np
    b_count = da.bincount if is_dask else np.bincount

    # ── Grid-aligned topology ──────────────────────────────────────────
    t_ns: np.ndarray = _time_values_to_int64(ds.time)
    t0_ns: int = (int(t_ns[0]) // dt_ns) * dt_ns  # ← epoch snap
    idx: np.ndarray = ((t_ns - t0_ns) // dt_ns).astype(np.intp)
    n_bins: int = int(idx.max()) + 1

    if is_dask:
        idx = da.from_array(idx, chunks=ds[first_var].chunks)

    # ── Parallel (sum, count) aggregation ──────────────────────────────
    def _sum_cnt(v: str) -> tuple:
        arr = ds[v].data
        valid = ~b_lib.isnan(arr)
        return (
            b_count(idx, weights=b_lib.where(valid, arr, 0.0), minlength=n_bins),
            b_count(idx, weights=valid.astype(np.float64), minlength=n_bins),
        )

    sums, cnts = zip(*(_sum_cnt(v) for v in ds.data_vars))

    # ── Deferred division (boundary-safe) ──────────────────────────────
    data_vars: dict[str, tuple] = {
        v: (
            "time",
            b_lib.where(c > 0, s / c, np.nan)
            if is_dask
            else np.divide(s, c, out=np.full(n_bins, np.nan), where=c > 0),
        )
        for v, s, c in zip(ds.data_vars, sums, cnts)
    }

    bin_times: np.ndarray = (t0_ns + np.arange(n_bins) * dt_ns).astype("datetime64[ns]")
    n_good: np.ndarray = cnts[0].compute().astype(np.intp) if is_dask else cnts[0].astype(np.intp)
    return xr.Dataset(data_vars, coords={"time": bin_times}), n_good


def binning(
    ds: xr.Dataset,
    dt_bin: timedelta,
    min_valid_fraction: float = 0.1,
    *,
    progress: bool = False,  # API compat; numpy path needs no bar
) -> Optional[xr.Dataset]:
    """O(n) time-resampling with NaN-fraction threshold.

    Discards bins where valid count < ``min_valid_fraction × mean(nonempty)``.
    Bins are epoch-grid-aligned (e.g. 2 s → …:00, …:02, …:04).

    Uses ``np.bincount`` / ``da.bincount`` for O(n) binning — avoids
    ``xarray.resample`` which has quadratic graph construction time on dask
    arrays in xarray ≥2026.7.  Deferred division guarantees correct means
    for bins spanning chunk boundaries.

    Parameters
    ----------
    ds
        Input dataset with ``time`` coordinate (must be sorted).
    dt_bin
        Binning interval (e.g. ``timedelta(seconds=600)``).
    min_valid_fraction
        Minimum fraction of the mean valid-sample count per bin.
    progress
        Accepted for API compatibility — numpy binning is fast enough
        that a progress bar adds no value.

    Returns
    -------
    xr.Dataset or None
        Binned dataset, or ``None`` if all data is discarded.
    """
    if not dt_bin or dt_bin <= timedelta(0):
        return ds

    dt_ns: int = int(pd.Timedelta(dt_bin).value)  # exact ns
    result, n_good = _resample_unify(ds, dt_ns)

    # Mask bins with too few valid samples
    if not (nonempty := n_good > 0).any():
        return None
    if not (keep := n_good > float(n_good[nonempty].mean()) * min_valid_fraction).any():
        return None

    return result.isel(time=keep) or None


def _axis_first_reduce(fn: Callable[..., np.ndarray]) -> Callable[[xr.DataArray], xr.DataArray]:
    """
    Wrap numpy kernel expecting axis-first (3, N) layout for xarray apply_ufunc:
    - transposes input from axis-last to axis-first before invocation;
    - binds standard ufunc application (core-dim/exclude-dim/dask configuration).

    """

    @wraps(fn)
    def _transposed(arr: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Adapt xarray's axis-last layout to numpy kernels"""
        return fn(np.moveaxis(arr, -1, 0), *args, **kwargs)

    def wrapper(gxyz: xr.DataArray) -> xr.DataArray:
        return xr.apply_ufunc(
            _transposed,
            gxyz,
            input_core_dims=[["axis"]],
            exclude_dims={"axis"},
            dask="parallelized",
        )  # puts the core dim last: (N, 3).

    return wrapper
