"""
xarray-native filtering for Dataset time-series — load-stage DROP + process-stage NaN-out.

Replaces ``tcm._dask_legacy.utils_dask.filter_global_minmax`` and
``filter_local`` with fixed ``ds.where()`` equivalents that read nested
``cfg['min']`` / ``cfg['max']`` dicts (legacy parity).
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from tcm._xr.calc import _time_values_to_int64
from tcm import utils2init, utils_time_corr

lf = utils2init.LoggingStyleAdapter(__name__)


# --------------------------------------------------------------------------- #
# Load-stage DROP: rows removed when raw-column values exceed bounds
# --------------------------------------------------------------------------- #

def filter_global_minmax(
    ds: xr.Dataset,
    cfg_in: Optional[Mapping[str, Any]] = None,
) -> xr.Dataset:
    """Drop rows where any raw-column value exceeds configured min/max bounds.

    Reads ``cfg_in['min']`` / ``cfg_in['max']`` **nested dicts** (legacy parity).
    Keys are column names — either bare (``Mx``) or shorthand ``M`` (expanded
    to ``Mx``/``My``/``Mz`` by :func:`expand_m_shorthand` before calling).

    Parameters
    ----------
    ds
        Input dataset with raw-column data variables.
    cfg_in
        Input configuration dict.  Reads ``cfg_in.min`` / ``cfg_in.max``
        nested dicts where each entry maps ``column_name → threshold``.
        ``None`` or empty dicts are silently skipped.

    Returns
    -------
    xr.Dataset
        Filtered dataset — rows outside any bound are **dropped**.
    """
    if cfg_in is None:
        return ds

    mask = None
    for lim_key, lim_dict in (
        ("min", cfg_in.get("min")),
        ("max", cfg_in.get("max")),
    ):
        if not lim_dict or not isinstance(lim_dict, dict):
            continue
        for col_key, val in lim_dict.items():
            if val is None:
                continue

            col = next((c for c in ds.data_vars if c.lower() == col_key.lower()), None)
            if col is None:
                lf.warning("filter_global_minmax: no column '{}'!", col_key)
                continue

            cond = ds[col] >= val if lim_key == "min" else ds[col] <= val
            mask = cond if mask is None else (mask & cond)

    if mask is not None:
        ds = ds.where(mask, drop=True)
    return ds


# --------------------------------------------------------------------------- #
# Process-stage NaN-out: values exceeding thresholds set to NaN (shape preserved)
# --------------------------------------------------------------------------- #

def filter_local(
    ds: xr.Dataset,
    cfg_filter: Optional[Mapping[str, Any]] = None,
    ignore_absent: Optional[set[str]] = None,
) -> xr.Dataset:
    """Set values to NaN where computed-column values exceed configured thresholds.

    Reads ``cfg_filter['min']`` / ``cfg_filter['max']`` **nested dicts**
    (legacy parity — fixes broken top-level iteration).  Each entry maps
    ``column_name → threshold``.  Values whose absolute value exceeds the
    threshold are set to NaN; rows are NOT dropped.

    Parameters
    ----------
    ds
        Input dataset — typically post-velocity where ``g_minus_1`` /
        ``h_minus_1`` have been computed.
    cfg_filter
        Filter configuration dict.  Reads ``cfg_filter.min`` / ``cfg_filter.max``
        nested dicts.  ``None`` or empty dicts are silently skipped.
    ignore_absent
        Column keys to silently skip when absent from the dataset
        (e.g. ``{"g_minus_1", "h_minus_1"}`` for datasets without G/H).

    Returns
    -------
    xr.Dataset
        Dataset of same shape — filtered values replaced with NaN.
    """
    if cfg_filter is None:
        return ds
    if ignore_absent is None:
        ignore_absent = set()

    for lim_key, lim_dict in (
        ("min", cfg_filter.get("min")),
        ("max", cfg_filter.get("max")),
    ):
        if not lim_dict or not isinstance(lim_dict, dict):
            continue
        for col_key, val in lim_dict.items():
            if val is None:
                continue

            col = next((c for c in ds.data_vars if c.lower() == col_key.lower()), None)
            if col is None:
                if col_key in ignore_absent:
                    continue
                lf.debug("filter_local: skipping unknown key '{}'", col_key)
                continue

            # NaN-out values exceeding threshold (abs for limit-sign symmetry)
            ds[col] = ds[col].where(np.abs(ds[col]) <= val)
            lf.debug("filter_local: NaN-out {}|{}| > {:g}", lim_key, col_key, val)

    return ds


# --------------------------------------------------------------------------- #
# Load-stage time windowing: drop rows outside configured time_ranges
# --------------------------------------------------------------------------- #

# def apply_load_time_ranges(
#     ds: xr.Dataset,
#     time_ranges: Optional[list[str]] = None,
# ) -> xr.Dataset:
#     """Drop rows whose ``time`` falls outside the configured window(s).

#     For NC/HDF5 sources, time_ranges drop is the load-stage equivalent of
#     CSV's ``time_corr._make_range_mask`` → ``b_ok`` row drop.  CSV already
#     applies time_ranges via time_corr during parsing; this function is
#     called for NC/HDF5 only.

#     Parameters
#     ----------
#     ds
#         Input dataset with ``time`` coordinate.
#     time_ranges
#         List of 2 ISO-8601 strings ``[start, end]`` — rows outside
#         ``[start, end]`` are dropped.  ``None`` or empty = no-op.

#     Returns
#     -------
#     xr.Dataset
#         Filtered dataset (rows outside window removed).
#     """
#     if not time_ranges or len(time_ranges) < 2:
#         return ds
#     if None in time_ranges:
#         return ds

#     if np.datetime64("NaT") in (
#         (start := np.datetime64(time_ranges[0])),
#         (end := np.datetime64(time_ranges[-1])),
#     ):
#         lf.warning("apply_load_time_ranges: invalid time_ranges {!r} — skipping", time_ranges)
#     elif start <= (t_min := ds["time"].values.min()) and (t_max := ds["time"].values.max()) <= end:
#         # skip if dataset is already within window
#         lf.debug("apply_load_time_ranges: data already inside [{!s}, {!s}] — no-op", start, end)
#     else:
#         ds = ds.sel(time=slice(time_ranges[0], time_ranges[1]))
#         if (n_before := ds.sizes.get("time", 0)) != (n_after := ds.sizes.get("time", 0)):
#             lf.info(
#                 "apply_load_time_ranges: dropped {} rows outside [{!s}, {!s}] ({}..{})",
#                 n_before - n_after, start, end, n_before, n_after,
#             )
#     return ds


def apply_load_time_ranges(
    ds: xr.Dataset,
    time_ranges: Sequence[str | None] | None = None,
) -> xr.Dataset:
    """Filter dataset rows via unified mask logic"""
    if not time_ranges:
        return ds

    # Coerce time coord to int64 ns view for alignment with _make_range_mask
    t_val = ds["time"].values
    t_ns = (
        t_val.astype("M8[ns]", copy=False).view(np.int64)
        if np.issubdtype(t_val.dtype, np.datetime64)
        else pd.to_datetime(t_val, utc=True).as_unit("ns").view(np.int64)
    )

    if (mask := utils_time_corr.make_range_mask(t_ns, time_ranges)).all():
        lf.debug("subset already valid -> no-op")
        return ds

    n_pre = ds.sizes.get("time", 0)
    ds_out = ds.isel(time=mask)

    if (drop_cnt := n_pre - (sel_cnt := ds_out.sizes.get("time", 0))) > 0:
        lf.info("dropped {} rows via range mask", drop_cnt) if drop_cnt < sel_cnt else lf.info(
            "selected {} rows via range mask", sel_cnt
        )

    return ds_out


# --------------------------------------------------------------------------- #
# Post-load gap warning: log if maximum data hole exceeds threshold
# --------------------------------------------------------------------------- #

def warn_on_holes(
    ds: xr.Dataset,
    dt_hole_warning: "Optional[int | float | timedelta]" = None,
) -> None:
    """Log a warning if the maximum data gap exceeds *dt_hole_warning* seconds.

    Uses ``np.diff`` on the ``time`` coordinate to compute gaps.  Only warns
    when *dt_hole_warning* is positive; ``None`` or 0 disables the check.

    After :func:`tcm.cli.main_init`, the ``dt_hole_warning`` config key is
    converted to ``timedelta`` by :func:`utils2init.type_fix` (``dt_*`` prefix).
    This function accepts both ``timedelta`` and numeric (int/float) values.

    Parameters
    ----------
    ds
        Input dataset with ``time`` coordinate.
    dt_hole_warning
        Maximum tolerated gap in **seconds** (``int``/``float``) or as a
        ``timedelta``.  A warning is emitted when any gap exceeds this value.
        ``None`` or ≤0 → no-op.
    """
    if dt_hole_warning is None:
        return
    warning_s = (
        dt_hole_warning.total_seconds()
        if isinstance(dt_hole_warning, timedelta)
        else float(dt_hole_warning)
    )
    if warning_s <= 0:
        return
    if ds.sizes.get("time", 0) < 2:
        return

    t_ns = _time_values_to_int64(ds["time"])
    diffs = np.diff(t_ns)
    max_hole_ns = diffs.max()
    max_hole_s = max_hole_ns / 1e9

    if max_hole_s > warning_s:
        lf.warning(
            "Data gap {:.1f}s exceeds dt_hole_warning={}s at index {} (max gap position)",
            max_hole_s, warning_s, diffs.argmax(),
        )
    else:
        lf.debug("Max data gap {:.3f}s (OK, ≤ {}s)", max_hole_s, warning_s)