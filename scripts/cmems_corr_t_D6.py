"""
Pipeline B: bias-field correction for regular 2D data (time × depth)
===================================================================

Input assumptions
-----------------
reg_tz : np.ndarray[float], shape (T, Z)
    Regular grid (daily × depth).
ref_time : np.ndarray[int or float], shape (K,)
    Times (in same units as reg grid) of irregular profiles.
ref_z : np.ndarray[float], shape (K, Zi)
    Each row = 1 profile along depth.
reg_depth : np.ndarray[float], shape (Z,)
    Depth axis of regular grid.
ref_depth : np.ndarray[float], shape (Zi,)
    Depth axis for profile measurements.

All arrays already aligned in memory. Time axis of reg_tz is 0..T-1.
No external description required — full documentation inside.
"""

# %%

from datetime import datetime
from functools import partial
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from operator import sub
from pathlib import Path
import re
from scipy.interpolate import (
    # RBFInterpolator,
    # NearestNDInterpolator,
    # LinearNDInterpolator,
    # CloughTocher2DInterpolator,
    RectBivariateSpline,
    PchipInterpolator,
    # FloaterHormannInterpolator,
    interp1d,
)
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree, Delaunay
from sklearn.isotonic import IsotonicRegression

from statsmodels.tsa.seasonal import STL

from typing import Any, Callable, Dict, Optional, Sequence
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# import warnings
# warnings.filterwarnings('ignore')

from get_datasets import utils as ds_utils
from hdf5_pandas import h5
from vsz_loader import veusz_load_hdf5_ctd_profile
from utils.logging_config import setup_logging
from veusz_helpers.common import func_vsz as fv

# Import module from current dir
import sys, os

logger = setup_logging(__name__)  # , console_format_args={"name": False, "datefmt": "%H:%M:%S"})

# os.path.abspath("") not works as we not in current dir which is a project root dir
# nb_dir = os.path.dirname(os.path.abspath(__vsc_ipynb_file__))
# logger.info(f"Adding {nb_dir} to sys.path")
# if nb_dir not in sys.path:
#     sys.path.insert(0, nb_dir)
# import o2_from_t_chain_and_cmems


# %%

def interp_profile_to_reg_depth(
    ref_depth: np.ndarray,
    ref_z: np.ndarray,
    reg_depth: np.ndarray
) -> np.ndarray:
    """
    Interpolate each profile's depth dimension to match the regular grid.

    Parameters
    ----------
    ref_depth : 1D Zi-array of profile depths.
    ref_z     : 2D (K, Zi) array of measured profiles.
    reg_depth  : 1D Z-array of regular depth grid.

    Returns
    -------
    ref_z_i : 2D (K, Z) interpolated profiles.
    """
    f = interp1d(
        ref_depth,
        ref_z,
        axis=1,
        kind="cubic",
        fill_value="extrapolate",
        bounds_error=False,
    )
    return f(reg_depth)


def build_bias_field(
    reg_time: np.ndarray,
    reg_tz: np.ndarray,
    ref_time: np.ndarray,
    ref_z_i: np.ndarray
) -> np.ndarray:
    """
    Compute raw bias at profile times:
        bias(t*, z) = ref_z_i(t*, z) - reg_tz(t*, z)

    ref_time — время профилей в реальных единицах

    Returns
    -------
    bias_raw : 2D array (K, Z)
    """
    idx = np.searchsorted(reg_time, ref_time)
    return ref_z_i - reg_tz[idx]


def interp_over_time_simple(
    ref_time: np.ndarray,
    ref_z: np.ndarray,
    out_time: np.ndarray,
) -> np.ndarray:
    """
    Interpolate bias in time for each depth independently using PCHIP.

    PCHIP avoids overshoot and handles large gaps well.

    Parameters
    ----------
    ref_time : times of profiles.
    reg_time : regular time
    bias_raw : (n_x_in, n_y) raw bias at profile times.


    Returns
    -------
    bias_tz : (T, n_y) full bias field.
    """
    n_x_in, n_y = ref_z.shape
    n_x_out = len(out_time)
    out_z = np.empty((n_x_out, n_y), float)

    for iy in range(n_y):
        z = ref_z[:, iy]
        f = PchipInterpolator(ref_time, z, extrapolate=True)
        out_z[:, iy] = f(out_time)
    return out_z


def clamp_trend(t: np.ndarray, t_base, slope, lim):
    """Clamp linear trend extension outside data window."""
    delta = t - t_base
    return slope * np.where(abs(delta) <= lim, delta, lim * np.sign(t - t_base))


def interp_over_time(
    ref_time: np.ndarray,
    ref_z: np.ndarray,
    out_time: np.ndarray,
    y,
    *,
    stl_period: int | None = None,
    win_smooth: int = 5,
    poly_smooth: int = 2,
    ext_limit: float = 0.5,
    b_plot: bool = True
):
    """
    Compute a stable bias(t) field using seasonal decomposition +
    smoothing of residuals + controlled extrapolation of trend.

    Parameters
    ----------
    t_prof : array
        Irregular times of profile data.
    b_prof : array
        Bias = P(t_i)-R(t_i) at profile timestamps.
    t_reg : array
        Regular grid of times.
    stl_period : int, optional
        Seasonality period (samples) if known (e.g. 365 for daily yearly).
        If None → seasonality skipped.
    win_smooth : int
        Savitzky–Golay window for smoothing residuals (odd).
    poly_smooth : int
        Polynomial order for SG smoothing.
    ext_limit : float
        Fraction of extrapolation range allowed for trend extension.
        Prevents explosive extrapolation far from data.

    Returns
    -------
    out_b : array
        Bias interpolated to t_reg.
    figs : dict[str, matplotlib.figure.Figure]
        Diagnostic figures.
    """

    # --- prepare and clean ---
    n_x_in, n_y = ref_z.shape
    n_x_out = len(out_time)
    out_z = np.empty((n_x_out, n_y), float)
    z_out_raw = np.empty((n_x_out, n_y), float)
    z_in_sm = np.empty((n_x_in, n_y), float) + np.nan

    m = np.isfinite(ref_z)
    for iy in range(n_y):
        m_iy = m[:, iy]
        ref_t, z_in_iy = ref_time[m_iy], ref_z[m_iy, iy]


        # --- 1) seasonal decomposition (optional) ---
        if stl_period:
            stl = STL(z_in_iy, period=stl_period, robust=True).fit()
            trend = stl.trend
            seas = stl.seasonal
            resid = stl.resid
        else:
            trend = savgol_filter(z_in_iy, win_smooth, poly_smooth)
            seas = np.zeros_like(z_in_iy)
            resid = z_in_iy - trend

        # --- 2) smooth residuals robustly ---
        # guarantee window odd, > poly
        if win_smooth % 2 == 0:
            win_smooth += 1
        if win_smooth <= poly_smooth:
            win_smooth = poly_smooth + 3

        resid_s = savgol_filter(resid, win_smooth, poly_smooth)

        # --- 3) smoothed recombined input ---
        z_in_sm[:, iy] = trend + seas + resid_s

        # --- 4) controlled-trend extrapolation ---
        # Build PCHIP but damp the trend slope outside data

        pchip = PchipInterpolator(ref_t, z_in_sm[:, iy], extrapolate=True)
        z_out_raw[:, iy] = pchip(out_time)

        # how far extrapolation allowed
        t_in_min, t_in_max = ref_t.min(), ref_t.max()
        dt = t_in_max - t_in_min
        dt = 1e-9 if dt == 0 else dt
        lim = ext_limit * dt

        # slope at edges
        edge_slope_st = sub(*z_in_sm[1::-1, iy]) / (sub(*ref_t[1::-1]) + 1e-12),
        edge_slope_en = sub(*z_in_sm[:-3:-1, iy]) / (sub(*ref_time[:-3:-1]) + 1e-12)

        out_z[:, iy] = z_out_raw[:, iy].copy()

        b_extrap = out_time < t_in_min
        out_z[b_extrap, iy] = z_in_sm[0, iy] + clamp_trend(out_time[b_extrap], t_in_min, edge_slope_st, lim)
        b_extrap = out_time > t_in_max
        out_z[b_extrap, iy] = z_in_sm[-1, iy] + clamp_trend(out_time[b_extrap], t_in_max, edge_slope_en, lim)
    axes = None
    if b_plot:

        cmap = cm.get_cmap("jet", n_y)
        colors = cmap(range(n_y))  # list of N colors

        for iy in range(n_y):
            m_iy = m[:, iy]
            ref_t, z_in_iy = ref_time[m_iy], ref_z[m_iy, iy]
            axes = plot_lines(
                out_time,
                ref_t,
                z_in_iy,
                z_in_sm[:, iy],
                z_out_raw[:, iy],
                out_z[:, iy],
                axes,
                f"z={y[iy]}",
                color=colors[iy],
            )
    return out_z, axes


def smooth_bias_2d(
    reg_t,
    reg_y,
    reg_z,
    ref_t,
    ref_y,
    ref_z,
    stl_period=365,
    min_points_pchip=3,
    win_z=7,
    poly_z=2,
    gp_sigma_reg=None,
    gp_sigma_prof=None,
):
    """
    Smooth 2D bias field between irregular profiles and regular data.

    Parameters
    ----------
    reg_t : 1D ndarray
        Regular time grid.
    reg_y : 1D ndarray
        Regular depth grid.
    reg_z : 2D ndarray [t,z]
        Regular field R(t,z).
    ref_t : 1D ndarray
        Irregular profile times.
    ref_y : 1D ndarray
        Depths of profiles (must match reg_z or contain NaNs to interpolate per profile).
    ref_z : 2D ndarray [t_i,z]
        Profile measurements P(t_i,z).
    stl_period : int
        Seasonal period (in index units). E.g. 365 for daily data.
    min_points_pchip : int
        Minimum number of neighbors required for pchip interpolation.
    win_z, poly_z : int
        Savitzky–Golay filter parameters for vertical smoothing.
    gp_sigma_reg, gp_sigma_prof : optional 2D ndarrays [t,z]
        Uncertainty estimates for weighting.

    Returns
    -------
    out_bias : 2D ndarray [t,z]
        Smoothed bias interpolated to regular grid.
    """
    n_x_in, n_y = ref_z.shape
    n_x_out = len(reg_t)

    bias_layered = np.zeros((n_x_out, n_y))

    # --- 1) Compute bias at profile times and each depth
    #      b(t_i, z) = P - R(t_i)
    #      where R(t_i) interpolated by pchip over time for each z
    for iy in range(n_y):
        # Extract regular and profile data for depth z
        r_z = reg_z[:, iy]
        p_z = ref_z[:, iy]

        # interpolate R to profile times (avoid extrapolation blow-up)
        if len(reg_t) >= min_points_pchip:
            r_interp = PchipInterpolator(reg_t, r_z, extrapolate=False)
            R_prof = r_interp(ref_t)
        else:
            R_prof = np.full_like(ref_t, np.nan)

        bias_profile = p_z - R_prof

        # --- 2) Seasonal decomposition + residual smoothing (STL)
        # Build a temporary daily series with NaN gap filling using local pchip
        if np.sum(~np.isnan(bias_profile)) >= min_points_pchip:
            stl_series = np.interp(
                reg_t,
                ref_t[~np.isnan(bias_profile)],
                bias_profile[~np.isnan(bias_profile)],
                left=np.nan,
                right=np.nan,
            )
        else:
            stl_series = np.zeros_like(reg_t) * np.nan

        # safe NaN filling for STL (with conservative smoothing)
        nan_mask = np.isnan(stl_series)
        if np.any(~nan_mask):
            stl_series[nan_mask] = np.interp(reg_t[nan_mask], reg_t[~nan_mask], stl_series[~nan_mask])
        else:
            stl_series[:] = 0.0

        stl = STL(stl_series, period=stl_period, seasonal=7, trend=31, robust=True)
        res = stl.fit()
        seasonal = res.seasonal
        resid = res.resid

        # residual smoothing with minimum width ~3 profile points
        # use UnivariateSpline-like effect via Savitzky–Golay by shrinking window
        w = max(5, 2 * min_points_pchip + 1)
        resid_smooth = savgol_filter(resid, window_length=w, polyorder=1)

        # result at this depth
        bias_layered[:, iy] = seasonal + resid_smooth

    # --- 3) vertical smoothing (stabilize between depth levels)
    for ti in range(n_x_out):
        bias_layered[ti, :] = savgol_filter(bias_layered[ti, :], window_length=win_z, polyorder=poly_z)

    # --- 4) compute weighting by uncertainties (if provided)
    if gp_sigma_reg is not None and gp_sigma_prof is not None:
        w = 1.0 / (1.0 + (gp_sigma_reg / gp_sigma_prof) ** 2)
        w = np.clip(w, 0.05, 1.0)
        bias_layered *= w

    # --- 5) 2D spline to ensure smoothness & bounded extrapolation
    spline = RectBivariateSpline(reg_t, reg_y, bias_layered, kx=2, ky=2)
    out_bias = spline(reg_t, reg_y)

    return out_bias


def plot_lines(t_reg, t, z, b_s, b_raw, b_adj, axes=None, leg_add="", **kwargs):
    """Visualization"""

    # switch on date labels
    t = t.astype("M8[D]")
    t_reg = t_reg.astype("M8[D]")

    if axes is None:
        fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
        axes[0].set_title("bias smoothing")
        axes[1].set_title("controlled extrapolation")

        # correct date labels
        loc = mdates.AutoDateLocator()
        fmt = mdates.ConciseDateFormatter(loc)
        for i_ax in [0, 1]:
            axes[0].xaxis.set_major_locator(loc)
            axes[0].xaxis.set_major_formatter(fmt)
            axes[0].xaxis.set_major_locator(loc)
            axes[0].xaxis.set_major_formatter(fmt)

            axes[1].grid(True, alpha=0.3)

    axes[0].plot(t, z, ".", label=f"raw bias {leg_add}", **kwargs)
    axes[0].plot(t, b_s, "-", label=f"smoothed bias {leg_add}", **kwargs)

    axes[1].plot(t_reg, b_raw, ".", label=f"raw extrapolated {leg_add}", **kwargs)
    axes[1].plot(t_reg, b_adj, label=f"clamped {leg_add}", **kwargs)
    # for i_ax in [0, 1]:
    #     axes[i_ax].legend()

    # plt.tight_layout()
    # plt.show()

    return axes


def apply_bias(
    reg_tz: np.ndarray,
    bias_tz: np.ndarray
) -> np.ndarray:
    """
    Apply interpolated bias to regular grid.

    Returns
    -------
    corrected : (T, Z)
    """
    return reg_tz + bias_tz


def correct_regular_by_profiles(
    reg_time,
    reg_depth: np.ndarray,
    reg_tz: np.ndarray,
    ref_time: np.ndarray,
    ref_depth: np.ndarray,
    ref_z: np.ndarray,
    dir_save_nc=None,
    save_suffixes_for_reg_bias_cor_prof=["CMEMS", "bias", "CMEMS_cor", "CTD"],
    meta={},
) -> np.ndarray:
    """
    Full correction pipeline of reg_tz to ref_z.
    reg_tz: 2D data
    ref_z: 2D data
    Steps
    -----
    1) depth-interpolate profiles;
    2) compute bias at profile times;
    3) PCHIP temporal interpolation of bias;
    4) add bias to regular grid.

    Returns
    -------
    reg_corr : (T, Z) corrected field.
    """
    # 1) depth interpolation
    if np.unique(np.diff(ref_depth)).size == 1 and len(ref_depth) > len(reg_depth):
        # interpolation of 2D "regular" data to depth of profiles because it is worse (and even may be not regular)
        ref_z_i = ref_z
        reg_tz = interp_profile_to_reg_depth(reg_depth, reg_tz, ref_depth)
        reg_depth = ref_depth
    else:
        # 2D profiles data interpolation to the depth of regular data
        ref_z_i = interp_profile_to_reg_depth(ref_depth, ref_z, reg_depth)

    # 2) bias at measurement times
    bias_raw = build_bias_field(reg_time, reg_tz, ref_time, ref_z_i)

    # 3) bias interpolation over time
    bias_tz, axes = interp_over_time(ref_time, bias_raw, reg_time, reg_depth, stl_period=365)

    # 4) apply correction
    reg_tz_cor = apply_bias(reg_tz, bias_tz)

    # Prepare to save 2D data that can be obtained only from profiles without regular data
    ref_tz_save = interp_over_time_simple(ref_time, ref_z_i, reg_time)

    # Save
    prefix = "for_srf"
    b_save_to_netcdf = dir_save_nc and not any(dir_save_nc.glob(f"{prefix}*{save_suffixes_for_reg_bias_cor_prof[0]}.nc"))
    if b_save_to_netcdf:
        time = reg_time.astype("M8[s]")

        ds_utils.save_nc_for_surfer(
            time=time,
            y=-reg_depth,  # grid_tc["p_dbar"],
            # puts y 1st for Surfer, as required by save_nc_for_surfer():
            out={
                f"T_{sfx}": val.T
                for sfx, val in zip(
                    save_suffixes_for_reg_bias_cor_prof, [reg_tz, bias_tz, reg_tz_cor, ref_tz_save]
                )
            },
            path_base=dir_save_nc / prefix,
            dt=sub(*time[1::-1]),
            not_interp_keys={"T"},
            stem_sfx="",
            attrs={
                f"T_{sfx}": {
                    "standard_name": "sea_water_potential_temperature",
                    "units": "degree_Celsius",
                    "name": "temperature",
                }
                for sfx in save_suffixes_for_reg_bias_cor_prof
            },
            **{k: meta[k] for k in ["lat", "lon"] if k in meta},
        )

        path_save_fig = dir_save_nc / "bias_smoothing(depth).png"

        axes[1].set_xlim(-5, 5)
        axes[0].figure.savefig(
            path_save_fig,
            format="png",
            dpi=300,
            transparent=False,
        )   # , bbox_inches="tight"
        logger.info(f"Plot saved to {path_save_fig}")

    else:
        logger.warning(f'Skipping saving to existed "{prefix}*@{save_suffixes_for_reg_bias_cor_prof[0]}.nc"')
    return reg_tz_cor


def fill_nan_1d_local(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Fill NaNs in a 1D profile y(x) using PCHIP on existing finite points.
    If profile fully NaN → return unchanged.
    """
    mask = np.isfinite(y)
    if mask.sum() < 2:
        return y
    f = PchipInterpolator(x[mask], y[mask], extrapolate=False)
    y_filled = y.copy()
    y_filled[~mask] = f(x[~mask])
    return y_filled


# %%

if __name__ == "__main__":
    # Input paths
    path_ctd = Path(
        r"B:\WorkData\CMEMS\1201..2511_D6_CTD\CTD_D6_worked_sorted_filtered_collection@Krechik.csv"
    )
    path_cmems_point_time_sections = Path(
        r"B:\WorkData\CMEMS\1201..2511_D6_CTD\@cmems_mod_bal"
    )
    dir_save_nc = path_cmems_point_time_sections.parent / "srf"
    # %% Load CTD collection data of format:
    # Date,Depth_st,Temperature,Salinity,Station,Latitude,Longitude
    # 18/01/2012,1,5.06,7.2,D6,55.3275,20.5755
    df = pd.read_csv(
        path_ctd,
        parse_dates=['Date'],  # Парсинг даты из столбца 'Date'
        dayfirst=True,         # Указание формата день/месяц/год
        dtype={                # Явное указание типов для столбцов
            'Depth_st': np.float32,
            'Temperature': np.float32,
            'Salinity': np.float32,
            'Station': str,    # Сохранение текстового формата
            'Latitude': np.float32,
            'Longitude': np.float32
        },
        skipinitialspace=True  # Удаление пробелов после разделителя
    )
    df.Depth_st = -df.Depth_st + 0.5  # source depth is negative
    df = df.rename(columns={"Depth_st": "Depth"})
    df.Date = df.Date + pd.Timedelta("9:00:00")  # Shift + to UTC

    # Создаем сводную таблицу
    df_tmp = df.pivot_table(
        index='Date',           # строки - даты
        columns='Depth',        # столбцы - глубины
        values='Temperature',   # значения - температура
        aggfunc='first'         # берем первое значение если есть дубликаты
    )
    assert len(np.unique(np.diff(df_tmp.columns.values))) == 1
    ref_time = df_tmp.index.astype("M8[s]").to_numpy(int)
    ref_depth = df_tmp.columns.values
    print(f"Depth levels: {ref_depth}")
    ref_z = df_tmp.values

    # Clean each profile from NaN
    ref_z = np.vstack([
        fill_nan_1d_local(ref_depth, row)
        for row in ref_z
    ])


    # %% Load CMEMS
    path_sections = list(path_cmems_point_time_sections.glob("phy_*.nc"))
    if not path_sections:
        raise (
            ValueError(
                f"Not a dir: {path_cmems_point_time_sections}!"
                if not path_cmems_point_time_sections.is_dir()
                else f"Matched files not found in {path_cmems_point_time_sections}"
            )
        )
    assert len(path_sections) == 2, f"The number of files {path_sections} is wrong: Check"

    # Load reanalysis first, then forecast
    if path_sections[0].stem.startswith("phy_anfc_P1D"):
        path_sections = path_sections[::-1]

    variables = ["thetao"]  # if many vars, you must put variable with max dimensions first
    out = {}
    meta = {}
    for i1ds, path_section in enumerate(path_sections, start=1):
        print(path_section.name)
        # Open CMEMS netCDF4 file
        time_nc, y_nc, z_nc, meta[i1ds] = ds_utils.nc_load(path_section, variables)
        time_nc = time_nc.astype("M8[s]").astype(int)
        if i1ds == 1:
            reg_time = time_nc
            reg_depth = y_nc
            reg_tz = z_nc
        else:
            b_need = time_nc > reg_time[-1]
            reg_time = np.append(reg_time, time_nc[b_need])
            assert (reg_depth == y_nc).all()
            reg_tz = np.append(reg_tz, z_nc[b_need, :], axis=0)

    # Check that sufficient CMEMS data time range is loaded
    assert reg_time[0] <= ref_time[0]
    assert reg_time[-1] >= ref_time[-1]
    # remove bad depth layers
    b_bad = np.isnan(reg_tz)
    b_need = ~b_bad.all(axis=0)
    if any(~b_need):
        reg_depth = reg_depth[b_need]
        reg_tz = reg_tz[:, b_need]
        b_bad = b_bad[:, b_need]
    assert (~b_bad).all()

    # %% Process loaded data
    reg_corr = correct_regular_by_profiles(
        reg_time, reg_depth, reg_tz, ref_time, ref_depth, ref_z, dir_save_nc=dir_save_nc, meta=meta[1]
    )

# %%
