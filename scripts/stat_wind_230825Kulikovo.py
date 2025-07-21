import types
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import h5py  # same result as netCDF4 loading function
import netCDF4
from numpy import sin, cos, radians, cumsum, append, absolute, degrees, arctan2
from datetime import datetime, timedelta
from itertools import combinations
# from scipy.stats import skew, kurtosis  - pandas methods used
from scipy.stats import pearsonr, circmean, circstd, linregress
from scipy.spatial.distance import euclidean
from scipy.signal import coherence, welch, csd

try:
    from astropy.stats import circstats  # , circcorrcoef  #  to use circstats.circcorrcoef
except ImportError as e:
    print(e, "- circstats.circcorrcoef() result will be None")
    circstats = lambda *x: None
    circstats.circcorrcoef = circstats
import statsmodels.api as sm


import func_vsz as fv
# import sys
# from importlib import import_module
import stat_wind


def load_netcdf(
    filepath, time_range=None, time_shift_s=0, slices=None, vars=["eastward_wind", "northward_wind"]
):
    """
    load datasets named '/time', '/eastward_wind', '/northward_wind' converting from cm/s to m/s
    :param: slices: None - other is not checked/implemented
    """
    # Open the netCDF file
    with xr.open_dataset(filepath) as ds:
        try:
            time = ds["time"][slice(slices)]
        except KeyError:
            time = ds["valid_time"][slice(slices)]
        # + np.timedelta64(time_shift_s + 631152000, 's') + 1970-01-01T00:00:00Z
        # Get indices for analysis
        if time_range is not None:
            iu = np.searchsorted(time, time_range)
            if slices:
                slices_vars = slices[0] + slice(*iu)
            else:
                slices_vars = slice(*iu)
        else:
            iu = None
            slices_vars = slice(slices)

        var_values = [ds[var][slices_vars].to_numpy().flatten() for var in vars]
        lat_lon = []
        for name_options in [('lat', 'latitude'), ('lon', 'longitude')]:
            for name_option in name_options:
                try:
                    lat_lon.append(ds[name_option].to_numpy().item())
                    break
                except KeyError:
                    continue
        print(
            f"Loaded {var_values[0].size} {vars} for ({', '.join(f'{c:.6g}' for c in lat_lon)})",
            f"from {'/'.join(filepath.parts[-2:])}")
    return (np.array(time[slice(*iu)], 'M8[s]') , *var_values, *lat_lon)


if __name__ == "__main__":
    # Calculate statistics for GMX500, CMEMS, ECMWF, NCEP, D6, Rybnoe

    # Recreate the variables from my Veusz file.
    # - Constants

    path_meteo = Path(r"B:\WorkData\BalticSea\230825_Kulikovo@ADCP,ADV,i,tr\meteo")
    files = {
        "CMEMS": path_meteo / (r"CMEMS\cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H_multi-vars-to_20.2997E_54.9896N_2023-08-20-2023-09-20.nc"),
        "ECMWF": path_meteo / r"ECMWF\area(54.75-55.0N,20.25-20.5E)\data_stream-oper_stepType-instant-to_20.3E_54.99N.nc",
        "NCEP": path_meteo / r"NCEP_CFSv2\area(55.09-54.89N,20.25-20.45E)(U,V)-to_20.3E_54.99N.nc",
        "meteo": path_meteo / r"_proc_to10m(GMX500),CMEMS,D6,Rybnoe\230825_1200@GMX500,D6,Rybnoe.tsv",
    }

    USEtime_Wind = np.array(
        ["2023-08-30T00:00:00", "2023-09-10T00:00:00"], "M8[s]",
    ) + np.array([0, 1], "m8[s]")  # because function loading from NetCDF clips before last edge  # UTC


    WIND_time_shift = 7200   # [s] add to data source (UTC) time. USEtime_Wind now is on this time
    out_file_add_str = (
        f"{(USEtime_Wind[0]).item():%y%m%d_%H%M}-{(USEtime_Wind[-1]).item():%y%m%d_%H%M}"
        # + np.timedelta64(WIND_time_shift, 's')  #UTC
    ).replace('_0000', '')

    WIND_bin_average_s = 3600

    device_to_column = {
        "GMX": "Буй GMX500",
        "CM": "CMEMS",
        "EC": "ECMWF",
        "NC": "NCEP",
        "D6": "Д-6",
        "Ryb": "Рыбное",
    }
    v = "W"  # our field letter
    ve_vn = {}
    # %% Load CMEMS data
    time_CM, *ve_vn["CM"], latitude_CM, longitude_CM = (
        stat_wind.load_NetCDF_data(files["CMEMS"], USEtime_Wind, time_shift_s=WIND_time_shift)
    )  # fs = 1/np.int32(np.diff(time_CM[1:3])).item()  # data frequency

    # %% Load ECMWF data
    time_EC, *ve_vn["EC"], latitude_EC, longitude_EC = load_netcdf(
        files["ECMWF"],
        USEtime_Wind,
        time_shift_s=WIND_time_shift,
        vars=["u10", "v10"],
    )

    # %% Load NCEP data
    time_NC, *ve_vn["NC"], latitude_NC, longitude_NC = load_netcdf(
        files["NCEP"],
        USEtime_Wind,
        time_shift_s=WIND_time_shift,
        vars=["U_GRD_L103", "V_GRD_L103"],
    )

    # %% Load buoy and stations data
    data = stat_wind.load_CSV_data(files["meteo"], USEtime_Wind, time_shift_s=WIND_time_shift)
    # Find columns that contain any NaN values
    cols_with_nans = data.columns[data.isna().any()].tolist()
    if cols_with_nans:
        _ = {k: d for k, d in data.isna().sum().items() if d > 0}
        print("interpolating columns with NaNs:", _)

        # Interpolate all missing data based on the time index
        data_interpolated = data.interpolate(method='time')

    # If there are columns that should not be interpolated (non-numeric or categorical columns),
    # you can exclude them from interpolation
    columns_to_interpolate = data.select_dtypes(include=[np.number]).columns
    data[columns_to_interpolate] = data[columns_to_interpolate].interpolate(method='time')

    # append columns of |W|_CM and Wdir_CM to gather all input data in one dataframe
    for device in ["CM", "EC", "NC"]:
        complex_vector = ve_vn[device][0] + 1j * ve_vn[device][1]
        data[f"|W|_{device}"] = np.absolute(complex_vector)
        # with mat to geo angle conversion (same as degrees(arctan2(eastward_wind_CM, northward_wind_CM)))
        data[f"Wdir_{device}"] = (90 - np.angle(complex_vector, deg=True)) % 360

    # Calculate the u and v components for each device
    for device in device_to_column:
        data[f"u_{device}"] = data[f"|W|_{device}"] * np.sin(
            np.radians(data[f"Wdir_{device}"])
        )
        data[f"v_{device}"] = data[f"|W|_{device}"] * np.cos(
            np.radians(data[f"Wdir_{device}"])
        )

    stats_out_df, corr_stats_df = stat_wind.get_stat(data, device_to_column, v="W")

    # Save "blow from" directions:
    stats_out_df.loc[["<W>dir", "<Wdir>"], :] = (stats_out_df.loc[["<W>dir", "<Wdir>"], :] + 180) % 360

    stat_wind.save_stat(
        stats_out_df,
        corr_stats_df,
        path_base=files["meteo"],
        device_to_column=device_to_column,
        out_file_add_str=out_file_add_str,
    )

    # simpler method to get correlations:

    corr_abs = data.filter(regex=fr"^\|{v}\|", axis=1).corr(min_periods=10)
    corr_dir = (
        (data.filter(regex=f"^{v}dir", axis=1))
        .apply(np.radians)
        .corr(min_periods=10, method=stat_wind.circstats.circcorrcoef)
    )

    corr_abs_top_dir_bot = corr_abs.rename(
        **{arg: lambda x: x.removeprefix(f"|{v}|_") for arg in ("index", "columns")},
    ).mask(
        np.tril(np.ones(corr_abs.shape, dtype=bool), k=-1),
        corr_dir.rename(
            **{arg: lambda x: x.removeprefix(f"{v}dir_") for arg in ("index", "columns")}
        )
    )

    diff_shift_std = stat_wind.pairwise_shift_and_std(data)
    diff_shift_std.round(2).to_csv(
        Path(files["meteo"]).with_suffix(f".{out_file_add_str}stats_diff_shift&std.csv"), sep="\t"
    )
    corr_stats_df.round(2).to_csv(
        Path(files["meteo"]).with_suffix(f".{out_file_add_str}stats_corr_tol2.csv"), sep="\t"
    )


    # DISPdevices_info = {
    #     'GMX': ['буй\\\\GMX500', 1, 1, '⯯', 54.953351, 20.44482],
    #     'CM': [fv.c1('{CMEMS}'.format_map(fv.I)), 1, 0, '⯯', latitude_CM, longitude_CM],
    #     'D6': [fv.c1('{Д6}'.format_map(fv.I)), 1, 0, '⯯', 54.953351, 20.44482],
    #     'Ryb': [fv.c1('{Рыбное}'.format_map(fv.I)), 1, 0, '⯯', 54.953351, 20.44482]
    #     }


    # # Calculate stats_Wcorr_abs_dir
    # def calculate_correlation(data, devices):
    #     correlations = []
    #     for dev1 in devices:
    #         for dev2 in devices:
    #             if dev1 != dev2:
    #                 corr = np.corrcoef(data[f'|W|_{dev1}'], data[f'|W|_{dev2}'])[0, 1]
    #                 correlations.append(corr)
    #     return np.array(correlations)

    # stats_Wcorr_abs_dir = calculate_correlation(data, device_to_column)
    # stats_Wcorr_abs_dir = pd.DataFrame({'corr_abs_dir': stats_Wcorr_abs_dir})




    # Compute rolling window statistics for Wabs2D
    # todo
    RollWindow = 24 * 3  # This is the rolling window size
    stats_Wrollcorr = np.array([
        np.corrcoef(
            Wabs2D[:, i:i + RollWindow].flatten(),
            Wdir2D[:, i:i + RollWindow].flatten()
        )[0, 1]
        for i in range(Wabs2D.shape[1] - RollWindow + 1)
    ])

if False:  # uncomment if need following vars
    dt_CM = np.min(np.diff(time_CM[:3]))

    # Cumulative sums for 'bin2_u_cum_CM' and 'bin2_v_cum_CM'

    # Function definitions from Veusz custom definitions
    f = lambda fun, *args, **kwargs: fun(*args, **kwargs)
    sl = lambda x, y: slice(x, y)

    bin2_u_cum_CM = append(0, cumsum(f(np.nanmean, bin2_u_CM[sl(iu_CM[0], iu_CM[-1])]) * dt_CM) * Wind_to_current_coef)
    bin2_v_cum_CM = append(0, cumsum(f(np.nanmean, bin2_v_CM[sl(iu_CM[0], iu_CM[-1])]) * dt_CM) * Wind_to_current_coef)





# you might need a different approach based on your definition of angular correlation
# def angular_correlation(angle1, angle2):
#     """Function for angular correlation"""
#     # Convert angles to radians for circular statistics
#     angle1_rad = np.radians(angle1)
#     angle2_rad = np.radians(angle2)
#     # Compute the mean of the cosine of the angle differences
#     mean_cos = np.mean(np.cos(angle1_rad - angle2_rad))
#     return mean_cos

# Other corr. statistics you can include:
# Linear least-squares regression for two sets of wind speeds measurements
# - slope : float
#     Slope of the regression line.
# - intercept : float
#     Intercept of the regression line.
# - r_value : float
#     The Pearson correlation coefficient. The square of ``r_value``
#     is equal to the coefficient of determination.
# - p_value : float
#     The p-value for a hypothesis test whose null hypothesis is
#     that the slope is zero, using Wald Test with t-distribution of
#     the test statistic. See `alternative` above for alternative
#     hypotheses.
# - stderr : float
#     Standard error of the estimated slope (gradient), under the
#     assumption of residual normality.
# - intercept_stderr : float
#     Standard error of the estimated intercept, under the assumption
#     of residual normality.
# stats_cur_dev = dict(
#     zip(
#         "slope intercept r_value p_value std_err".split(),
#         linregress(mag1, mag2),
#     )
# )

# # Euclidean distance between wind speed vectors
# stats_cur_dev["dist"] = euclidean(mag1, mag2)

# # Cross-correlation function calculations using statsmodels
# # .. [1] Brockwell and Davis, 2016. Introduction to Time Series and
# # Forecasting, 3rd edition, p. 242.
# ccf = sm.tsa.stattools.ccf(mag1, mag2, adjusted=False)
# # Max cross-correlation value and its lag
# imax = np.argmax(ccf)
# stats_cur_dev["max_cross_corr"] = ccf[imax]
# stats_cur_dev["lag_at_max_cross_corr"] = (
#     imax - (len(ccf) - 1) / 2
# )  # Assuming zero-centered lag
#
# # Coherence between the magnitudes of the wind speed
# f, Cxy = coherence(mag1, mag2, fs=fs)
#
# # Cross spectral density between series
# # f, Pxx = welch(mag1, fs=fs)  # Power spectral density
# f, Pxy = csd(mag1, mag2, fs=fs)
#
# # Phase spectrum = phase differences
# phase_diff = np.angle(Pxy)
# # Calculate the frequency at which the phase difference is maximum
# max_phase_diff_index = np.argmax(np.abs(phase_diff))
# freq_at_max_phase_diff = f[max_phase_diff_index]
#
# # Calculate the frequency at which the phase difference is minimum
# min_phase_diff_index = np.argmin(np.abs(phase_diff))
# freq_at_min_phase_diff = f[min_phase_diff_index]
#
# # Compute the average phase difference across all frequencies
# stats_cur_dev["Pxy_mean"] = np.mean(phase_diff)  # avg_phase_diff
#
# # Identify frequencies where the phase relationship changes significantly
# # Define a threshold for significant change
# phase_diff_threshold = np.pi / 4  # Example threshold of 45 degrees, adjust as needed
# significant_phase_shift_indices = np.where(np.abs(np.diff(phase_diff)) > phase_diff_threshold)[0]
# significant_phase_shift_freqs = f[significant_phase_shift_indices]
#
#
# # Maximum coherence and corresponding frequency
# imax = np.argmax(Cxy)
# stats_cur_dev["Cxy_max"] = Cxy[imax]     # max coherence
# stats_cur_dev["freq_Cxy_max"] = f[imax]  # frequency at max coherence
#