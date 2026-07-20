"""
Physical-parameter calculations (:class:`xarray.Dataset` backend).

Replaces ``tcm._dask_legacy.incl_calc.physical`` with pure
``xarray.Dataset`` / ``dask.array`` implementations.
"""
import time as _time
from datetime import timedelta
from typing import Any, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import xarray as xr
from numpy.polynomial.polynomial import polyval2d
from tqdm.auto import tqdm

from tcm import utils2init
import tcm.calibration.orientation
from tcm.incl_calc import calc
from tcm._xr import calc as calc_xr
from tcm._xr.filters import filter_local

lf = utils2init.LoggingStyleAdapter(__name__)


# --------------------------------------------------------------------------- #
# Velocity
# --------------------------------------------------------------------------- #

def calc_velocity(
    ds: xr.Dataset,
    *,
    Ag: np.ndarray,
    Cg: np.ndarray,
    Ah: np.ndarray,
    Ch: np.ndarray,
    kVabs: Optional[Sequence] = None,
    azimuth_shift_deg: float = 0,
    calc_version: str = "trigonometric(incl)",
    filt_max: Optional[Mapping[str, float]] = None,
    **kwargs,
) -> xr.Dataset:
    """
    Calculates velocity from raw accelerometer/magnetometer data.

    Parameters
    ----------
    ds
        Dataset containing ``Ax, Ay, Az, Mx, My, Mz`` data variables.
    Ag, Cg, Ah, Ch
        Calibration matrices.
    kVabs
        Velocity-conversion coefficients.
    azimuth_shift_deg
        Azimuth offset in degrees.
    calc_version
        Method passed to ``v_abs_from_incl``.
    filt_max
        Process-stage NaN-out thresholds from ``cfg_filter['max']``.
        ``g_minus_1`` → NaN inclination where |‖Gxyz‖ − 1| > threshold.
        ``h_minus_1`` → NaN Vdir where |‖Hxyz‖ − 1| > threshold.
        Detailed despike/recover port deferred (see ``dtdr-todo.md`` D).

    Returns
    -------
    xr.Dataset
        Same dataset with ``Vabs, Vdir, v, u, inclination`` added;
        raw ``Ax..Mz`` columns removed.
    """
    Axyz = ds[["Ax", "Ay", "Az"]].to_array(dim="axis")
    Mxyz = ds[["Mx", "My", "Mz"]].to_array(dim="axis")

    # Calibration: gain matrix + offset
    Gxyz = xr.apply_ufunc(calc.fG, Axyz, Ag, Cg, dask="parallelized")
    Hxyz = xr.apply_ufunc(calc.fG, Mxyz, Ah, Ch, dask="parallelized")
    incl = calc_xr._axis_first_reduce(tcm.calibration.orientation.tilt_from_vertical)(Gxyz)

    # GsumMinus1 = ||Gxyz|| - 1  (reduces 'axis' dim)
    GsumMinus1 = np.sqrt((Gxyz ** 2).sum("axis")) - 1

    # Process-stage NaN-out: |GsumMinus1| > threshold → null inclination
    try:
        bad_g = np.abs(GsumMinus1) > (g_minus_1_max := filt_max["g_minus_1"])
    except (KeyError, TypeError):
        pass  # no threshold configured
    else:
        if (n_bad := int(bad_g.sum())):
            lf.warning(
                "Acceleration |‖G‖−1| > {} in {:d} points ({:.1f}%) => inclination nulled",
                g_minus_1_max,
                n_bad,
                100 * n_bad / GsumMinus1.size,
            )
            # Apply g_minus_1 NaN-out to inclination (legacy: incl_rad[bad] = nan)
            incl = incl.where(~bad_g)

    if kVabs is not None:
        Vabs = xr.apply_ufunc(
            calc.v_abs_from_incl,
            incl,
            kwargs=dict(coefs=kVabs, calc_version=calc_version),
            dask="parallelized",
        )

        # Vdir: heading from gravity × magnetic cross-product.
        # Factor (GsumMinus1 + 1) corrects for non-unit gravity magnitude.
        Gx = Gxyz.isel(axis=0)
        Gy = Gxyz.isel(axis=1)
        Gz = Gxyz.isel(axis=2)
        Hx = Hxyz.isel(axis=0)
        Hy = Hxyz.isel(axis=1)
        Hz = Hxyz.isel(axis=2)

        Vdir = azimuth_shift_deg - np.degrees(np.arctan2(
            (Gx * Hy - Gy * Hx) * (GsumMinus1 + 1),
            Hz * (Gx ** 2 + Gy ** 2) - Gz * (Gx * Hx + Gy * Hy),
        ))

        # Process-stage NaN-out: |HsumMinus1| > threshold → null Vdir
        if filt_max and "h_minus_1" in filt_max and filt_max["h_minus_1"] is not None:
            HsumMinus1 = np.sqrt((Hxyz ** 2).sum("axis")) - 1
            bad_h = np.abs(HsumMinus1) > filt_max["h_minus_1"]
            if (n_bad_h := int(bad_h.sum())):
                lf.warning(
                    "Magnetometer |‖H‖−1| > {} in {:d} points ({:.1f}%) => Vdir nulled",
                    filt_max["h_minus_1"],
                    n_bad_h,
                    100 * n_bad_h / HsumMinus1.size,
                )
            Vdir = Vdir.where(~bad_h)

        v, u = calc.polar2dekart(Vabs, Vdir)
        ds = ds.assign(
            Vabs=Vabs,
            Vdir=Vdir,
            v=v,
            u=u,
            inclination=np.degrees(incl),
        )
        ds = ds.drop_vars(["Ax", "Ay", "Az", "Mx", "My", "Mz"])

    return ds


# --------------------------------------------------------------------------- #
# Pressure
# --------------------------------------------------------------------------- #

def calc_pressure(
    ds: xr.Dataset,
    P_t: Optional[np.ndarray] = None,
    bad_p_at_bursts_starts_period: str = "",
    **kwargs,
) -> xr.Dataset:
    """
    Convert raw ``P`` / ``P_counts`` to physical pressure.

    Parameters
    ----------
    ds
        Dataset with ``P`` or ``P_counts`` and ``Temp`` variables.
    P_t
        2-D polynomial coefficients for temperature compensation.
    bad_p_at_bursts_starts_period
        Pandas offset alias (e.g. ``'1h'``, ``'30min'``) for pressure burst
        NaN-out — nulls the first 2 points of each burst period.  Empty string
        disables.  Ported from legacy ``incl_calc/physical.py:440-449``.
    """
    if P_t is None:
        return ds
    col = "P" if "P" in ds else "P_counts" if "P_counts" in ds else None
    if col is None:
        return ds

    arr = xr.apply_ufunc(
        polyval2d,
        ds[col].astype(float),
        ds["Temp"],
        P_t,
        dask="parallelized",
    )
    ds = ds.drop_vars(col)
    ds["Pressure"] = arr

    # Port: bad_p_at_bursts_starts_period — null first 2 of each burst period
    if bad_p_at_bursts_starts_period and "Pressure" in ds:
        ds = _null_first2_per_period(ds, "Pressure", bad_p_at_bursts_starts_period)
    return ds


def _null_first2_per_period(
    ds: xr.Dataset,
    var: str,
    period: str,
) -> xr.Dataset:
    """Null the first 2 points of each resampled period for *var*.

    Mirrors legacy ``calc_and_rem2first`` (``physical.py:443-447``) which
    repartitions by *period* and sets ``pressure[:2] = np.nan`` per partition.
    Uses xr.resample + a per-group mask to propagate to the original index.
    """
    pd_bin = pd.Timedelta(period) if not isinstance(period, str) or period.isdigit() else pd.Timedelta(period)
    # Group by period, identify global first-2 indices of each group, null them
    groups = ds[var].resample(time=pd_bin).groups
    mask = xr.ones_like(ds[var], dtype=bool)
    for _label, idx_slice in groups.items():
        # idx_slice is a slice into the original time axis
        start = idx_slice.start if idx_slice.start is not None else 0
        stop = min(start + 2, idx_slice.stop if idx_slice.stop is not None else len(ds[var]))
        if stop > start:
            # Null the first 2 of this group
            time_vals = ds["time"].values[start:stop]
            mask = mask.where(~ds["time"].isin(time_vals), other=False)
    ds[var] = ds[var].where(mask)
    return ds


# --------------------------------------------------------------------------- #
# Full pipeline
# --------------------------------------------------------------------------- #

def process(
    ds_raw: xr.Dataset,
    *,
    coefs: Mapping[str, Any],
    coef_zeroing_matrix: Optional[np.ndarray] = None,
    cfg_filter: Optional[Mapping[str, Any]] = None,
    dt_bins: Sequence[timedelta] = (timedelta(0),),
    dt_min_binning_proc: timedelta = timedelta(seconds=2),
    pcid: str = "",
) -> List[Optional[xr.Dataset]]:
    """
    Full pipeline: filter → velocity → pressure → binning.

    Parameters
    ----------
    ds_raw
        Raw sensor dataset (Ax..Mz, optionally P_counts, Temp).
    coefs
        Calibration coefficients dict (Ag, Cg, Ah, Ch, kVabs, P_t, …).
    coef_zeroing_matrix
        Optional rotation matrix applied to Ag, Ah before velocity calc.
    cfg_filter
        Filter configuration (passed to :func:`filter_local`).
    dt_bins
        Binning intervals.  ``timedelta(0)`` means no-averaging output.
    dt_min_binning_proc
        Bins with ``dt ≤ dt_min_binning_proc`` are computed on raw data
        (bin-then-calc); larger bins compute on processed data (calc-then-bin).
    pcid
        Probe ID for logging.

    Returns
    -------
    list
        ``[no_avg_result] + [binned_result, ...]``  — one entry per dt_bin.
        ``None`` entries mean no data survived that binning step.
    """
    if not dt_bins:
        return [None]

    # 1. Apply rotation if needed
    if coef_zeroing_matrix is not None:
        coefs = {
            **coefs,
            **{c: coef_zeroing_matrix @ coefs[c] for c in ("Ag", "Ah")},
        }

    # 2. Filter
    ds = filter_local(ds_raw, cfg_filter, ignore_absent={"h_minus_1", "g_minus_1"})
    bad_p_period = cfg_filter.get("bad_p_at_bursts_starts_period", "") if cfg_filter else ""

    need_d_out = dt_bins[0] == timedelta(0)
    dt_bins_remaining = list(dt_bins)

    d = None  # the "calc-then-bin" source (physical domain)
    d_avgs: List[Optional[xr.Dataset]] = []

    if need_d_out:
        dt_bins_remaining = dt_bins_remaining[1:]

    # 3. Compute velocity on raw (or binned-raw for small bins)
    velocity_computed = False
    n_raw = ds.sizes.get("time", 0)
    # Suppress tqdm for small data — avoids visual blink when each bin < ~1s
    show_progress = n_raw >= 100_000
    pbar = tqdm(
        dt_bins_remaining, desc=f"[{pcid}] bins", unit="bin", leave=False,
        disable=not show_progress,
    )
    for dt_bin in pbar:
        bin_label = f"{int(dt_bin.total_seconds())}s"
        bin_before_physical = dt_bin <= dt_min_binning_proc
        t0 = _time.monotonic()

        if bin_before_physical:
            pbar.set_postfix_str(f"raw {bin_label}-binning ")
            ds_binned = calc_xr.binning(ds, dt_bin, progress=show_progress)
            if ds_binned is None:
                lf.error("No {} data after {}-binning of raw => Skipping", pcid, dt_bin)
                d_avgs.append(None)
                pbar.set_postfix_str(f"{bin_label}: no data after binning")
                continue
            pbar.set_postfix_str(f"velocity {bin_label} ({ds_binned.sizes.get('time', 0):} rows)")
            d_avg = calc_velocity(ds_binned, filt_max=cfg_filter.get("max") if cfg_filter else None, **coefs)
            d_avg = calc_pressure(d_avg, bad_p_at_bursts_starts_period=bad_p_period, **coefs)
            d_avgs.append(d_avg)
        else:
            # Compute velocity once on raw (or first time we need it)
            if not velocity_computed:
                n_raw = ds.sizes.get("time", 0)
                pbar.set_postfix_str(f"velocity raw ({n_raw:} rows)")
                lf.debug("Computing velocity on raw data ({} rows)", n_raw)
                d = calc_velocity(ds, filt_max=cfg_filter.get("max") if cfg_filter else None, **coefs)
                d = calc_pressure(d, bad_p_at_bursts_starts_period=bad_p_period, **coefs)
                velocity_computed = True

            pbar.set_postfix_str(f"{bin_label}-binning")
            if (d_avg := calc_xr.binning(d, dt_bin, progress=show_progress)) is None:
                lf.error("No {} data after {}-binning of processed => Skipping", pcid, dt_bin)
            d_avgs.append(d_avg)

        dt_s = _time.monotonic() - t0
        pbar.set_postfix_str(f"{bin_label} done ({dt_s:.1f}s)")

    # 4. Assemble: noAvg (if requested) + binned results
    result: List[Optional[xr.Dataset]] = []
    if need_d_out:
        if d is None and not velocity_computed:
            t0 = _time.monotonic()
            n_raw = ds.sizes.get("time", 0)
            lf.debug("Computing velocity for noAvg ({} rows)", n_raw)
            d = calc_velocity(ds, filt_max=cfg_filter.get("max") if cfg_filter else None, **coefs)
            d = calc_pressure(d, bad_p_at_bursts_starts_period=bad_p_period, **coefs)
            lf.debug("noAvg velocity done ({:.1f}s)", _time.monotonic() - t0)
        result.append(d)
    result.extend(d_avgs)
    pbar.close()

    # 5. Reorder columns: v, u, inclination, Vabs, Vdir first (legacy cols_out_h5 order)
    _cols_first_names = ("v", "u", "inclination", "Vabs", "Vdir")
    for i, r in enumerate(result):
        if r is not None:
            first = [c for c in _cols_first_names if c in r.data_vars]
            remaining = [c for c in r.data_vars if c not in first]
            if first:
                result[i] = r[first + remaining]

    return result
