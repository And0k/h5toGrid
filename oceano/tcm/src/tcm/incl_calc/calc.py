"""
Low-level math kernels for physical calculations.
Pure numpy implementations — no dask dependencies.
_xr/calc.py wraps these with xr.apply_ufunc for dask.array support.
"""
from datetime import timedelta
from typing import (
    List,
    Sequence,
    TypeVar,
    TYPE_CHECKING,
    Union,
)
import numpy as np
import pandas as pd

from tcm.calibration.calibrate import SensorCalibration, to_unit_vector

if TYPE_CHECKING:
    import dask.array as da
    import xarray as xr

from tcm import utils2init

lf = utils2init.LoggingStyleAdapter(__name__)


# @allow_dask
def fIncl_rad2force(incl_rad: np.ndarray):
    """Theoretical force from inclination"""
    return np.sqrt(np.tan(incl_rad) / np.cos(incl_rad))


# @allow_dask
def fIncl_deg2force(incl_deg):
    return fIncl_rad2force(np.radians(incl_deg))


# @allow_dask
def fG(Axyz: Union[np.ndarray, 'da.Array'],
       Ag: Union[np.ndarray, 'da.Array'],
       Cg: Union[np.ndarray, 'da.Array']) -> Union[np.ndarray, 'da.Array']:
    """Apply linear coef to data matrix. Allows use of transposed Cg"""
    assert Ag.any(), 'Ag coefficients all zeros!!!'
    if Cg.ndim < 2:
        Cg = Cg.reshape(-1, 1)
    return Ag @ (Axyz - Cg)


# @allow_dask
def polar2dekart(
    Vabs: Union[np.ndarray, "da.Array", "xr.DataArray"], Vdir: Union[np.ndarray, "da.Array", "xr.DataArray"]
) -> List[Union[np.ndarray, "da.Array", "xr.DataArray"]]:
    """
    Polar → cartesian (v, u)
    Vabs, Vdir polar components: module and angle (degrees)
    return: [v, u] (north, east) components
    """
    rad = np.radians(Vdir)
    return [Vabs * np.cos(rad), Vabs * np.sin(rad)]


def fVabsMax0(x_range, y0max, coefs):
    x0 = x_range[np.flatnonzero(np.polyval(coefs, x_range) > y0max)[0]]
    return (x0, np.polyval(coefs, x0))


def fVabs_from_force(force, coefs, vabs_good_max=0.5):
    is_nans = np.isnan(force)
    force[is_nans] = 0
    x0, v0 = fVabsMax0(np.arange(1, 4, 0.01), vabs_good_max, coefs)

    def v_normal(x):
        v = np.polyval(coefs, x)
        return np.where(x < x0, v, v0 + (x - x0) * (v0 - np.polyval(coefs, x0 - 0.1)) / 0.1)

    incl_good_min = 0.0623
    incl_range0 = np.linspace(0, incl_good_min, 15)
    force_range0 = fIncl_rad2force(incl_range0)

    def v_linear(x):
        return np.interp(x, force_range0, incl_range0) * np.polyval(coefs, 0.25) / incl_good_min

    force = np.where(force > 0.25, v_normal(force), v_linear(force))
    force[is_nans] = np.nan
    return force


def trigonometric_series_sum(r, coefs):
    out = np.empty_like(r)
    out[:] = coefs[0]
    for n in range(1, (len(coefs) + 1) // 2):
        a = coefs[n * 2 - 1]
        b = coefs[n * 2]
        nr = n * r
        out += (a * np.cos(nr) + b * np.sin(nr))
    return out


def rep_if_bad(checkit, replacement):
    return checkit if (checkit and np.isfinite(checkit)) else replacement


def f_linear_k(x0, g, g_coefs):
    replacement = np.float64(10)
    return min(rep_if_bad(np.diff(g(x0 - np.float64([0.01, 0]), g_coefs)).item() / 0.01, replacement), replacement)


def f_linear_end(g, x, x0, g_coefs):
    g0 = g(x0, g_coefs)
    return np.where(x < x0, g(x, g_coefs), g0 + (x - x0) * f_linear_k(x0, g, g_coefs))


def v_trig(r, coefs):
    squared = np.sin(r) / trigonometric_series_sum(r, coefs)
    return np.where(squared > 0, np.sqrt(squared), 0)


def v_abs_from_incl(
    incl_rad: np.ndarray, coefs: Sequence, calc_version="trigonometric(incl)", max_incl_of_fit_deg=None
) -> np.ndarray:
    if len(coefs) <= 4 or calc_version == 'polynom(force)':
        if not len(incl_rad):
            return incl_rad
        return fVabs_from_force(fIncl_rad2force(incl_rad), coefs)
    elif calc_version == 'trigonometric(incl)':
        if max_incl_of_fit_deg:
            max_incl_of_fit = np.radians(max_incl_of_fit_deg)
        else:
            max_incl_of_fit = np.radians(coefs[-1])
            coefs = coefs[:-1]
        with np.errstate(invalid='ignore'):
            return f_linear_end(
                g=v_trig, x=incl_rad, x0=np.atleast_1d(max_incl_of_fit),
                g_coefs=np.float64(coefs)
            )
    else:
        raise NotImplementedError(f'Bad calc method {calc_version}')


def dekart2polar_df_uv(df, **kwargs):
    if 'u' in df.columns:
        kdegrees = 180 / np.pi
        return df.eval(f"""
Vabs = hypot(u, v)
Vdir = arctan2(u, v)*{kdegrees:.20}
""", **kwargs)
    else:
        return df


def norm_field(raw3d, coef_a2d, coef_c, raw3d_helps_recover=None):
    """
    Apply calibration coefficients to raw 3D data.
    Handles both numpy and dask.array (via hasattr checks).
    """
    if coef_c.ndim < 2:
        coef_c = coef_c.reshape(-1, 1)
    # Apply coefs
    if hasattr(raw3d, 'map_blocks'):  # dask.array
        s = raw3d.map_blocks(
            lambda a3d: coef_a2d @ (a3d - coef_c),
            dtype=np.float64,
            meta=np.float64([]),
        )
    else:
        s = coef_a2d @ (raw3d - coef_c)

    # If gain for some channel is zero
    if (n_bad := (i_ch_bad := np.flatnonzero(coef_a2d.diagonal() == 0)).size):
        lf.warning(
            "Zero gain ({} values) for channel(s) {} -> recovering from other channels",
            n_bad, " ".join("xyz"[i] for i in i_ch_bad),
        )
        i_ch_ok = [i for i in range(3) if i != i_ch_bad]
        if hasattr(s, 'compute'):
            s = s.compute()

        s[i_ch_bad] = np.square(1 - (s[i_ch_ok] ** 2).sum(axis=0))
        if (s[i_ch_bad].imag != 0).any():
            s[i_ch_bad] = s[i_ch_bad].real

        if raw3d_helps_recover is not None:
            _ = np.sign(raw3d_helps_recover[i_ch_bad])
            if hasattr(raw3d_helps_recover, 'compute'):
                _ = _.compute()
            s[i_ch_bad] *= _
            return s

        s_dif = np.ediff1d(s[i_ch_bad], to_begin=0)
        b_reversed = s_dif < 0
        b_reversed &= np.append(~b_reversed[1:], False)
        _ = s[i_ch_bad, b_reversed] + s[i_ch_bad, np.roll(b_reversed, 1)]
        s_dif_prev = s_dif[np.roll(b_reversed, -1)]
        b_reversed[b_reversed] = np.abs(s_dif_prev - _) < np.abs(s_dif_prev - s_dif[b_reversed])

        s_rev_sign = np.zeros_like(s_dif)
        s_rev_sign[0] = 1
        n_reversed = sum(b_reversed)
        s_rev_sign[b_reversed] = np.tile([-2, 2], int(np.ceil(n_reversed / 2)))[:n_reversed]
        s_rev_sign = np.cumsum(s_rev_sign)
        s[i_ch_bad] = s_rev_sign * s[i_ch_bad]
    return s


out_velocity_cols = ('Vabs', 'Vdir', 'v', 'u', 'inclination')


# --------------------------------------------------------------------------- #
# Burst detection
# --------------------------------------------------------------------------- #

def _to_timedelta64(val) -> np.timedelta64:
    """Coerce *val* to ``np.timedelta64`` — accepts ``timedelta``, ``pd.Timedelta``, or ``np.timedelta64``."""
    if isinstance(val, np.timedelta64):
        return val
    if isinstance(val, pd.Timedelta):
        return val.to_timedelta64()
    if isinstance(val, timedelta):
        return np.timedelta64(int(val.total_seconds()), 's')
    raise TypeError(f"Cannot coerce {type(val).__name__} to np.timedelta64")


def i_bursts_starts(
    tim,
    dt_between_blocks=None,
) -> tuple[np.ndarray, int | float, np.timedelta64]:
    """Detect burst boundaries in a datetime index.

    Pure numpy — no dask dependency.  Both ``_dask_legacy`` and ``_xr``
    pipelines use this as the canonical burst-detection kernel.

    Parameters
    ----------
    tim
        Datetime-like array (:class:`pandas.DatetimeIndex`,
        :class:`numpy.ndarray` of ``datetime64``, or anything with a
        ``.values`` attribute returning one).
    dt_between_blocks
        Minimum gap that separates two bursts.  Accepted types:

        * ``None`` — auto-detect: ``min(first two diffs) + 1 s``.
        * ``np.timedelta64`` / ``datetime.timedelta`` / ``pd.Timedelta``.
        * ``np.inf`` (or any float ``> max timedelta64``) — treat the
          entire *tim* as a single burst.

    Returns
    -------
    i_bursts : np.ndarray[int32]
        Indices of burst starts; always begins with ``0``.
    mean_burst_size : int | float
        Mean number of samples between consecutive burst starts.
        Equals ``len(tim)`` when no gaps are found.
    max_hole : np.timedelta64
        Largest inter-sample gap among detected boundaries.
        Zero when no gaps are found.
    """
    dt_zero = np.timedelta64(0)
    max_hole = dt_zero

    # Unwrap DatetimeIndex to numpy array
    if hasattr(tim, 'values'):
        tim = tim.values
    if not len(tim):
        return np.int32([]), 0, max_hole

    dtime = np.diff(tim)

    # Warn on non-monotonic input (log once with all stats)
    if (non_mono := np.flatnonzero(dtime <= dt_zero)).size:
        lf.warning(
            "Non-monotonic time: {:d} decreasing + {:d} equal, first at index {:d}",
            np.sum(dtime < dt_zero), np.sum(dtime == dt_zero), non_mono[0],
        )

    # Normalize dt_between_blocks
    if dt_between_blocks is None:
        dt_between_blocks = dtime[:2].min() + np.timedelta64(1, 's')
    elif isinstance(dt_between_blocks, (int, float)):
        # np.inf path — entire series is one burst
        return np.int32([0]), len(tim), max_hole
    else:
        dt_between_blocks = _to_timedelta64(dt_between_blocks)

    # Guard against overflow when dt_between_blocks exceeds max timedelta64
    max_delta_ns = np.timedelta64((1 << 63) - 1, 'ns')
    if dt_between_blocks > max_delta_ns.astype('m8[s]'):
        return np.int32([0]), len(tim), max_hole

    i_burst = np.flatnonzero(dtime > dt_between_blocks)

    if i_burst.size:
        if i_burst.size > 1:
            mean_burst_size = np.mean(np.diff(i_burst))
        else:  # exactly one gap
            i_burst_st = i_burst[0] + 1
            mean_burst_size = max(i_burst_st, len(tim) - i_burst_st)
        max_hole = dtime[i_burst].max()
    else:
        mean_burst_size = len(tim)

    return np.append(0, i_burst + 1).astype(np.int32), mean_burst_size, max_hole
