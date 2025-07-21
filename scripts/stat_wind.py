import types
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
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
# # Absolute path to the directory containing the module.
# module_dir = r"C:\Work\Python\AB_SIO_RAS\Veusz_plugins"
# module_name = 'func_vsz'
# sys.path.append(module_dir)     # Add the module's directory to sys.path.
# fv = import_module(module_name)  # Import the module using its name.
# sys.path.remove(module_dir)     # to avoid conflicts with future imports.

# # from runpy import run_path
# # fv = type("Namespace", (object,), run_path(
# #     r"C:\Work\Python\AB_SIO_RAS\Veusz_plugins\func_vsz.py"
# #     ))()

pd.pandas.set_option("display.max_columns", None)  # for better debug display

# functions to load data from CSV and HDF5 files

def load_HDF5_data(filepath, time_range=None, time_shift_s=0, slices=None):
    """
    load datasets named '/time', '/eastward_wind', '/northward_wind' converting from cm/s to m/s
    """
    with h5py.File(filepath, "r") as f:
        eastward_wind_CM = f["/eastward_wind"][slice(slices)].flatten() * 0.01
        northward_wind_CM = f["/northward_wind"][slice(slices)].flatten() * 0.01
        time_CM = f["/time"][slice(slices)].flatten()
        lat = f['latitude']
        lon = f['longitude']

    time_CM = np.array(time_CM, 'M8[s]') + np.timedelta64(
        time_shift_s + 631152000, 's'  # + 1970-01-01T00:00:00Z
    )

    # Get indices for analysis
    if time_range is not None:
        iu_CM = slice(*np.searchsorted(time_CM, time_range))
    else:
        iu_CM = slice(None)
    return (time_CM[iu_CM], eastward_wind_CM[iu_CM], northward_wind_CM[iu_CM], lat, lon)


def load_NetCDF_data(
    filepath, time_range=None, time_shift_s=0, slices=None, vars=["eastward_wind", "northward_wind"]
):
    """
    load datasets named '/time', '/eastward_wind', '/northward_wind' converting from cm/s to m/s
    :param: slices: None - other is not checked/implemented
    """
    # Open the netCDF file
    with netCDF4.Dataset(filepath) as f:
        # Access a specific group and variable
        group = f #.groups['/']
        time_CM = group["/time"][slice(slices)].flatten()
        time_CM = np.array(time_CM, 'M8[s]') + np.timedelta64(
            time_shift_s + 631152000, 's'
        )  # + 1970-01-01T00:00:00Z
        # Get indices for analysis
        if time_range is not None:
            iu_CM = np.searchsorted(time_CM, time_range)
            if slices:
                slices_vars = slices[0] + slice(*iu_CM)
            else:
                slices_vars = slice(*iu_CM)
        else:
            iu_CM = None
            slices_vars = slice(slices)

        var_values = [group[f"/{var}"][slices_vars].flatten() for var in vars]

        lat = group["latitude"]
        lon = group["longitude"]

    return (time_CM[slice(*iu_CM)], *var_values, lat, lon)


def load_CSV_data(filepath, time_range=None, time_shift_s=0):
    data = pd.read_csv(
        filepath,
        delimiter="\t",
        encoding="ascii",
        skip_blank_lines=True,  # Veusz 'blanksaredata' is True, but pandas treats blank lines as NaN by default
        skipinitialspace=True,
        # comment='\t',
        header=0,
        parse_dates=["Time_UTC"],
        date_format="ISO8601",
        index_col=False  ## setting "Time_UTC" as index not works for all files!
    ).dropna(how="all", axis=1)  # delete empty columns
    # Find the last valid index (row) that doesn't contain only NaNs
    last_valid_index = data.dropna(how='all').index[-1]
    # keep only the rows up to and including the last valid index
    data = data.loc[:last_valid_index]

    # Convert 'stimeUTC_Wind' to datetime if it's a string
    if data['Time_UTC'].dtype == 'object':
        data['Time_UTC'] = pd.to_datetime(data['Time_UTC'])
        # ValueError: day is out of range for month

    dt_shift = pd.Timedelta(seconds=time_shift_s)
    if time_range is not None:
        b_Wind = data["Time_UTC"].between(*[t - dt_shift for t in time_range])
    else:
        b_Wind = slice(None)
    data.loc[b_Wind, "Time_UTC"] += dt_shift

    # set 'Time_UTC' as the index (need if we will need interpolation)
    data = data.loc[b_Wind, :].set_index("Time_UTC")

    dir_cols = [col for col in data.columns if col.startswith('Vdir')]
    assert len(dir_cols) == 3
    data.loc[:, dir_cols] = (data.loc[:, dir_cols] - 180)%360  # convert to "blow to" direction

    return data.rename(
        columns=lambda col: f"W{col[1:]}"
        if col.startswith("V")
        else f"|W{col[2:]}"
        if col.startswith("|V")
        else col
    )  # {col: "V" + col[1:] if col.startswith("W") else col for col in data.columns}


############################################################################################

def devices_from_cols(cols, prefix):
    i_sfx_st = len(prefix)
    return [col[i_sfx_st:] for col in cols if col.startswith(prefix)]


def pairwise_shift_and_std(df: pd.DataFrame, devices: Iterable = None, v="W"):
    """
    Для каждой пары колонок вычисляет:
    - mean_shift[i, j] = среднее (col_i - col_j)
    - std_diff[i, j]   = СКО разностей (col_i - col_j)

    Матрица mean_shift будет антисимметричной (mean_shift[j, i] = –mean_shift[i, j]),
    а std_diff — симметричной (std_diff[j, i] = std_diff[i, j]).
    На диагонали обеих матриц стоят нули.
    """
    cols = df.columns
    var_name_prefix = f"|{v}|_"
    if devices is None:
        devices = devices_from_cols(data.columns, prefix=var_name_prefix)


    # Перебираем только уникальные пары (col1, col2), где col1 != col2
    diff_shift_std = {}
    var_name_prefix_out = var_name_prefix.removesuffix('_')
    for device1, device2 in combinations(devices, 2):
        devs_key = f"{device1}-{device2}"
        col1 = f"{var_name_prefix}{device1}"
        col2 = f"{var_name_prefix}{device2}"
        s1 = df[col1]
        s2 = df[col2]

        mask = s1.notna() & s2.notna()
        diff = s1[mask] - s2[mask]

        diff_shift_std[devs_key] = {
            f"{var_name_prefix_out}diff_mean": diff.mean(), f"{var_name_prefix_out}diff_std": diff.std()
        }
    return pd.DataFrame.from_dict(diff_shift_std)


def get_stat(data, devices: Iterable = None, v="W"):
    """Calculate statistical parameters for wind measurements from multiple devices.
    This function computes various statistical parameters for wind measurements, including:
    - Mean vector wind (<v>)
    - Circular mean and standard deviation of wind direction
    - Basic statistics (min, max, mean, etc.) of wind speed
    - Wind resistance (ratio of vector mean to scalar mean speed)
    - Correlations between device pairs for both wind speed and direction
    Parameters
    ----------
    data : pandas.DataFrame - Input data with datetime index, containing wind measurements
    with columns formatted as:
        - "|{v}|_{device}" : Wind speed magnitude for each device
        - "{v}dir_{device}" : Wind direction for each device
        - "time_{device}": optional columns to get time statistics (Start, End, dT_days range) on this columns
        instead `index` if such columns exists.
    devices : list
        List of device identifiers present in the data
        if None then find all devices by finding all suffixes of all "|{W}|_*" data columns
    v : str, optional
        Variable prefix used in column names (default "W" for wind)
    Returns
    -------
    tuple
        (stats_out_df, corr_stats_df)
        - stats_out_df : pandas.DataFrame
            Statistics for each device including mean vector, circular statistics,
            and basic statistical measures
        - corr_stats_df : pandas.DataFrame
            Correlation statistics between pairs of devices including:
            - Pearson correlation for wind speeds
            - Angular correlation for wind directions
    Notes
    -----
    Wind directions are assumed to be in meteorological convention (0-360°),
    and are converted to mathematical angles for calculations.
    Circular statistics are computed using scipy.stats.circmean and circstd.
    References
    ----------
    [1] Mardia, K. V. (1972). Statistics of Directional Data.
        Academic Press. doi:10.1016/C2013-0-07425-7 (pp. 18-24)
    """

    stats_functions = {
        f"|{v}|min": np.nanmin,
        f"|{v}|max": np.nanmax,
        f"<|{v}|>": np.nanmean,
        f"|{v}|var": np.nanvar,
        f"|{v}|std": np.nanstd,
        f"|{v}|skewness": pd.Series.skew,
        f"|{v}|kurtosis": pd.Series.kurt,
        f"|{v}|std/<|{v}|>": lambda x: np.nanstd(x) / np.nanmean(x),  # coef_var
        f"<|{v}|-<|{v}|>>": lambda x: np.nanmean(
            np.absolute(x - np.nanmean(x))
        ),  # mean_abs_dev
        "N": lambda x: np.sum(~np.isnan(x)),  # length
        # we use special variable name index to determine we want to substitute it with time:
        "Start": lambda index: index[0].strftime("%Y-%m-%d %H:%M"),
        "End": lambda index: index[-1].strftime("%Y-%m-%d %H:%M"),
        "dT_days": lambda index: (index[-1] - index[0]).total_seconds() / (3600 * 24),
    }

    if devices is None:
        devices = devices_from_cols(data.columns, prefix=f"|{v}|_")

    # Statistics calculations

    complex_vectors = {
        device: data[f"|{v}|_{device}"]
        * np.exp(  # with geo to mat angle conversion
            1j * np.radians(90 - data[f"{v}dir_{device}"])
        )
        for device in devices
    }

    stats_dict, b_nans = {}, {}
    for device, vec in complex_vectors.items():
        # Mean wind vector (<v>)
        complex_vector_mean = vec.mean()  # pandas removes NaNs
        # Simple statistics from <v>
        nan_dir = np.isnan(data[f"{v}dir_{device}"].to_numpy())
        try:
            i_nans = np.flatnonzero(nan_dir)[[0, -1]].tolist()
            b_nans[device] = True
            if i_nans[0] == i_nans[-1]:
                del i_nans[-1]
            for i, rep_word in [(0, "start"), (nan_dir.size - 1, "end")]:
                try:
                    i_nans[i_nans.index(i)] = rep_word
                    if nan_dir.all():
                        i_nans = ["all!"]
                        break
                except (ValueError, IndexError):
                    pass
            print(
                f"NaN in {v}dir_{device}: {nan_dir.sum()}/{nan_dir.size}.",
                "1st & last NaN:", ", ".join(str(i_nan) for i_nan in i_nans)
            )
        except IndexError:
            b_nans[device] = False

        stats_cur_dev = {
            # absolute value and direction
            f"|<{v}>|": np.abs(complex_vector_mean),
            f"<{v}>dir": (90 - np.angle(complex_vector_mean, deg=True)) % 360,
            # Circular mean and std deviation for wind directions
            f"<{v}dir>": circmean(data[f"{v}dir_{device}"][~nan_dir], high=360),
            f"{v}dir_std": circstd(data[f"{v}dir_{device}"][~nan_dir], high=360),
        }
        # Calculate simple statistics from |v| or time
        for stat_name, stat_func in stats_functions.items():
            try:  # If function uses 'index' in argument
                if 'index' in stat_func.__code__.co_varnames:
                    stats_cur_dev[stat_name] = stat_func(
                        (
                            pd.to_datetime(data[f"time_{device}"][[0, -1]])
                            if f"time_{device}" in data.columns
                            else data.index
                        )[(~nan_dir) if b_nans[device] else slice(None)]
                    )
                else:
                    raise AttributeError
            except AttributeError:
                stats_cur_dev[stat_name] = stat_func(data[f"|{v}|_{device}"])

        stats_cur_dev[f"|{v}|max-min"] = stats_cur_dev[f"|{v}|max"] - stats_cur_dev[f"|{v}|min"]
        # Wind resistance (simple statistics from |v| and <v>)
        stats_cur_dev[f"|<{v}>|/<|{v}|>"] = stats_cur_dev[f"|<{v}>|"] / stats_cur_dev[f"<|{v}|>"]

        stats_dict[device] = stats_cur_dev

    stats_out_df = pd.DataFrame.from_dict(stats_dict)

    # Calculate statistics for each pair of devices

    corr_stats_dict = {}
    stats_cur_dev = {}  # clear from previous cycle usage
    for (device1, vec1), (device2, vec2) in combinations(complex_vectors.items(), 2):
        devs_key = f"{device1}-{device2}"
        if b_nans[device1] or b_nans[device2]:
            ok_mask = (
                np.isfinite(data[f"{v}dir_{device1}"].to_numpy())
                & np.isfinite(data[f"{v}dir_{device2}"].to_numpy())
            )
            n_ok = ok_mask.sum().item()
            print(f"Ok data points for both {device1} & {device2}: {n_ok}")
        else:
            ok_mask = slice(None)
        # Gather absolute wind speed and wind directions for the pair
        mag1, mag2, dir1, dir2 = data.loc[
            ok_mask,
            [
                f"|{v}|_{device1}",
                f"|{v}|_{device2}",
                f"{v}dir_{device1}",
                f"{v}dir_{device2}",
            ],
        ].values.T

        # Calculate Pearson correlation for absolute wind speeds
        # and p-value for testing that the correlation is nonzero
        stats_cur_dev[f"|{v}|xy_corr"], stats_cur_dev["corr_abs_p-value"] = pearsonr(
            mag1, mag2
        )
        # Angular correlation for wind directions
        stats_cur_dev[f"{v}dir_xy_corr"] = circstats.circcorrcoef(
            np.radians(dir1), np.radians(dir2)
        )  # angular_correlation(dir1, dir2)

        # Store the statistics for the device pair
        corr_stats_dict[devs_key] = stats_cur_dev.copy()

    # Convert the correlations dictionary to a DataFrame
    corr_stats_df = pd.DataFrame.from_dict(corr_stats_dict)
    return stats_out_df, corr_stats_df


# Save files

def save_stat(
    stats_out_df, corr_stats_df, path_base, device_to_column=None, out_file_add_str="", v="W"
):

    notation_to_parameter = {
        f"|{v}|min": "Минимум модуля",
        f"|{v}|max": "Максимум модуля",
        f"<|{v}|>": "Средний модуль",
        f"|{v}|max-min": "Размах модуля",
        f"|<{v}>|": "Модуль среднего вектора",
        f"<{v}>dir": "Угол среднего вектора",
        f"|<{v}>|/<|{v}|>": f"Устойчивость {'ветра' if v=='W' else 'вектора'}",
        f"<|{v}|-<|{v}|>>": "Ср. абсолют. откл. модуля",
        f"|{v}|var": "Дисперсия модуля",
        f"|{v}|std": "Ср. кв. откл. модуля",
        f"|{v}|skewness": "Асимметрия модуля",
        f"|{v}|kurtosis": "Эксцесс модуля",
        f"|{v}|std/<|{v}|>": "Коэф. вариации модуля",
        f"<{v}dir>": "Ср. круговое углов",
        f"{v}dir_std": "Ср. кв. откл. углов",
        "N": "Кол-во отсчетов",
        "Start": "Начало",
        "End": "Конец",
        "dT_days": "Длительность",
    }

    notation_to_parameter_corr = {
        # "slope": "Наклон регрессии",
        # "intercept": "Пересечение регрессии",
        # "r_value": "Коэффициент Пирсона",
        # "p_value": "P-значение",
        # "std_err": "Станд. ошибка наклона",
        # "intercept_stderr": "Станд. ошибка пересечения",
        f"|{v}|xy_corr": "Корреляция модулей",
        # "corr_abs_p-value": "P-значение корреляции модулей",
        f"{v}dir_xy_corr": "Круговой коэф. корр. направлений",  # Корреляция направлений
        # "dist": "Евклидово расстояние",
        # "max_cross_corr": "Макс. кросс-корреляция",
        # "lag_at_max_cross_corr": "Лаг при макс. кросс-корреляции",
        # 'Cxy_max': 'Макс. когерентность',
        # 'freq_Cxy_max': 'Частота при макс. ког',
        # "phase_diff": "Разность фаз",
    }

    if device_to_column is None:
        device_to_column = {}

    df = (stats_out_df  # FutureWarning: DataFrame.applymap has been deprecated. Use DataFrame.map instead
        .map(fv.fmt_3_digits_after_dot).reset_index()  # applymap
        .rename(columns={"index": fv.c1(fv.I["notation"]), **device_to_column})
        .set_index(stats_out_df.index.map(notation_to_parameter))
        .loc[list(notation_to_parameter.values())]
    )
    df.index.name = fv.c1(fv.I['parameter'])
    df.to_csv(Path(path_base).with_suffix(f".{out_file_add_str}stats.csv"), sep="\t")

    # Save correlation parameters

    df = (
        corr_stats_df.map(fv.fmt_3_digits_after_dot)  # applymap
        .reset_index()
        .rename(
            columns={
                "index": fv.c1(fv.I["notation"]),
                **{
                    f"{dev1}-{dev2}": f"{col1}-{col2}"
                    for (dev1, col1), (dev2, col2) in combinations(
                        device_to_column.items(), 2
                    )
                },
            }
        )
        .set_index(corr_stats_df.index.map(notation_to_parameter_corr))
        .loc[list(notation_to_parameter_corr.values())]
    )
    df.index.name = fv.c1(fv.I["parameter"])
    df.to_csv(
        Path(path_base).with_suffix(f".{out_file_add_str}stats_corr.csv"), sep="\t"
    )


if __name__ == "__main__":
    # Calculate statistics for GMX500, CMEMS, D6, Rybnoe

    # Recreate the variables from my Veusz file.
    # - Constants
    b_2023_data = True
    if b_2023_data:
        cmems_netcdf_file = r"d:\WorkData\BalticSea\230825_Kulikovo@ADCP,ADV,i,tr\meteo\CMEMS\cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H_multi-vars_20.31E_54.94N_2023-08-20-2023-09-20.nc"
        meteo_csv_file = r"d:\WorkData\BalticSea\230825_Kulikovo@ADCP,ADV,i,tr\meteo\_proc_to10m(GMX500),CMEMS,D6,Rybnoe\230825_1200@GMX500,D6,Rybnoe.tsv"
        USEtime_Wind = np.array(  # UTC
            [
                # "2023-08-29T22:00:00"),
                # "2023-08-25T14:00:00", "2023-09-10T21:00:01",
                # "2023-09-01T00:00:00", "2023-09-01T15:00:00",
                "2023-08-30T00:00:00", "2023-09-10T00:00:00",
            ],
            "M8[s]",
        ) + np.array([0, 1], "m8[s]")  # because function loading from NetCDF clips before last edge
    else:
        cmems_netcdf_file = r'd:/WorkData/BalticSea/220505_D6/meteo/CMEMS/cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H_multi-vars_20.56E_55.31N_2022-05-01-2022-05-31.nc'
        meteo_csv_file = r'd:\WorkData\BalticSea\230825_Kulikovo@ADCP,ADV,i,tr\meteo\_proc_to10m(GMX500),CMEMS,D6,Rybnoe\220505_1600@GMX500,D6,Rybnoe.tsv'
        USEtime_Wind = np.array(  # UTC
            [
                "2022-05-05T18:00:00", "2022-05-17T10:00:01"

            ],
            "M8[s]",
        )

    WIND_time_shift = 7200   # [s] add to data source (UTC) time. USEtime_Wind now is on this time
    out_file_add_str = (  # ''
        f"{(USEtime_Wind[0]).item():%y%m%d_%H%M}-{(USEtime_Wind[-1]).item():%y%m%d_%H%M}"  # + np.timedelta64(WIND_time_shift, 's')  #UTC
    ).replace('_0000', '')

    WIND_bin_average_s = 3600

    devices = ['GMX', 'CM', 'D6', 'Ryb']



    # Load the data
    time_CM, eastward_wind_CM, northward_wind_CM, latitude_CM, longitude_CM = (
        load_NetCDF_data(
            cmems_netcdf_file, USEtime_Wind, time_shift_s=WIND_time_shift, slices=None
        )
    )
    # fs = 1/np.int32(np.diff(time_CM[1:3])).item()  # data frequency

    data = load_CSV_data(meteo_csv_file, USEtime_Wind, time_shift_s=WIND_time_shift)
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
    complex_vector_CM = eastward_wind_CM + 1j * northward_wind_CM
    data["|W|_CM"] = np.absolute(complex_vector_CM)
    # with mat to geo angle conversion (same as degrees(arctan2(eastward_wind_CM, northward_wind_CM)))
    data["Wdir_CM"] = (90 - np.angle(complex_vector_CM, deg=True)) % 360

    # Calculate the u and v components for each device
    for device in devices:
        data[f"u_{device}"] = data[f"|W|_{device}"] * np.sin(
            np.radians(data[f"Wdir_{device}"])
        )
        data[f"v_{device}"] = data[f"|W|_{device}"] * np.cos(
            np.radians(data[f"Wdir_{device}"])
        )

    stats_out_df, corr_stats_df = get_stat(data, devices, v="W")

    device_to_column = {
        'GMX':  'Буй GMX500',
        'CM':   'CMEMS',
        'D6':   'Д6',
        'Ryb':  'Рыбное'
    }
    save_stat(
        stats_out_df,
        corr_stats_df,
        path_base=meteo_csv_file,
        device_to_column=device_to_column,
        out_file_add_str=out_file_add_str,
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

    # stats_Wcorr_abs_dir = calculate_correlation(data, devices)
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