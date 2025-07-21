import json
from datetime import datetime, timedelta
from itertools import dropwhile, groupby
import numpy as np
import pandas as pd
import scipy.signal as sp
from pathlib import Path
from typing import Mapping, Optional
import re
import statsmodels.api as sm  # .stats.api as sms

# Custom functions
from scripts import stat_wind
import plot
import func_vsz as fv

def time_cols_to_dt64(_yyyy__, _mm__, _dd__, _HH__, _MM__, _SS__, time_add_s=0):
    date0 = np.datetime64(datetime(year=_yyyy__[0], month=_mm__[0], day=_dd__[0]), "s")
    return (
        date0
        + 3600 * (24 * np.cumsum(np.ediff1d(_HH__, to_begin=0) < 0) + _HH__)
        + 60 * _MM__
        + _SS__
        + time_add_s
    )


# Load data from file


def load_sontek_adv(
    file_path, date_format="%Y-%m-%d", time_add_s=0, excludelist=["std_P"]
):
    """
    Load Sontek ADV data from a .dat file, converting time to datetime64 and cm/s to m/s
    """

    # Define the data format
    data_format = (
        "yyyy,mm,dd,HH,MM,SS,u,v,Vz,std_u,std_v,std_Vz,"
        "SNR1,SNR2,SNR3,Int1,Int2,Int3,Noise1,Noise2,Noise3,GOOD,"
        "Heading,Pitch,Roll,std_Heading,std_Pitch,std_Roll,Temp,P,std_P,"
        "Power,Vabs,Vdir"
    )

    cols = data_format.split(",")

    i_not_time = 6
    # dtype of the resulting array
    dtype = np.dtype(
        [(col, int) for col in cols[:i_not_time]]
        + [(col, float) for col in cols[i_not_time:]]
    )

    # Load the data

    data = np.genfromtxt(
        file_path,
        dtype=dtype,  # delimiter=" " - not considers consecutive spaces as one
        skip_header=1,
        # excludelist=excludelist,  # not works?
        unpack=True,
    )
    cols = list(dtype.names)

    # Extract time variables
    i_time = 0
    data[i_time] = time_cols_to_dt64(*data[:i_not_time], time_add_s)
    cols[i_time] = "time"
    del data[i_time + 1 : i_not_time]
    del cols[i_time + 1 : i_not_time]

    try:
        fs = 1 / np.diff(data[i_time][:2]).item().total_seconds()
        data_len = data[i_time].size
        fs_mean = (data_len - 1) / np.diff(data[i_time][[0, -1]]).item().total_seconds()
        # if abs(fs - fs[tbl]) > 0.001 else "")
        print(
            file_path.name, f"{data_len}rows, fs0 = {fs}Hz,", f"fs_mean = {fs_mean}Hz"
        )
    except IndexError:
        print(file_path.name, "- No data!", flush=True)

    # Convert the structured array to a dictionary
    data_dict = dict(zip(cols, data))

    # delete not needed fields
    for col in excludelist:
        del data_dict[col]

    # cm to m/s
    cols_in_cm = ["u", "v", "Vz", "std_u", "std_v", "std_Vz", "Vabs"]
    for col in cols_in_cm:
        data_dict[col] *= 0.01

    return data_dict, fs


def process_adv(data, config):
    """
    Process ADV data according to the configuration.
    config:
    - "USEtime"
    - "dt_bin": optional, binning interval to compute binned data and statistics based on it, else statistics will be computed on original data.
    returns out, bin, stat
    out data:
    - "|V|" : speed magnitude for device (replacing original name for this column ("Vabs"))
    - "Vdir" : direction for device
    """

    disp_time_range = np.array(config["USEtime"], "M8[s]")

    # Apply user-defined time range
    mask = (disp_time_range[0][0] <= data["time"]) & (
        data["time"] <= disp_time_range[0][1]
    )
    out = {
        ("|V|" if col == "Vabs" else col): data[col][mask]
        for col in ["time", "u", "v", "Vabs", "Vdir"]
    }

    # Compute bin averages
    if config.get("dt_bin"):
        bin_edges = fv.i_whole_time_intervals(data["time"][mask], config["dt_bin"])
        bin = {
            "u": fv.bin_avg(out["u"], bin_edges),
            "v": fv.bin_avg(out["v"], bin_edges),
        }
        bin["|V|"] = np.hypot(bin["u"], bin["v"])
        bin["Vdir"] = np.degrees(np.arctan2(bin["u"], bin["v"]))
        # fv.bin_avg(direction, bin_edges) # fv.bin_avg(speed, bin_edges)
    else:
        bin = out

    # Compute statistics
    stat = {"mean_u": np.nanmean(bin["u"]), "mean_v": np.nanmean(bin["v"])}
    stat["mean_speed"] = np.hypot(
        stat["mean_u"], stat["mean_v"]
    )  # np.sqrt(mean_u**2 + mean_v**2)
    stat["mean_direction"] = fv.wrap_dir(
        np.degrees(np.arctan2(stat["mean_u"], stat["mean_v"]))
    )
    # Additional
    stat["|V|max"] = np.nanmax(bin["|V|"])  # np.round(np.nanmin(bin_Vabs), 3)
    stat["|V|min"] = np.nanmin(bin["|V|"])
    stat["|V|mid"] = (stat["|V|min"] + stat["|V|max"]) / 2

    return out, bin, stat


def to_polar(df: pd.DataFrame):
    prefix = "u_"
    i_sfx_st = len(prefix)
    # find devices
    devices = sorted({col[i_sfx_st:] for col in df.columns if col.startswith(prefix)})
    # calc / del
    for device in devices:
        df[f"|V|_{device}"] = np.hypot(
            df[f"u_{device}"].values, df[f"v_{device}"].values
        )
        df[f"Vdir_{device}"] = fv.wrap_dir(
            np.degrees(np.arctan2(df[f"u_{device}"], df[f"v_{device}"]))
        )
        del df[f"u_{device}"]
        del df[f"v_{device}"]


def to_polar_dfs(dfs: Mapping[str, pd.DataFrame], b_device_in_col_suffix=False):
    """
    Compute derived variables |V| and Vdir in each dataframe
    removes "u", "v"
    """
    if isinstance(dfs, Mapping):  # dfs is dict of DataFrames
        for k, df in dfs.items():
            if b_device_in_col_suffix:
                to_polar(df)
            else:
                dfs[k]["|V|"] = np.hypot(df["u"], df["v"])
                dfs[k]["Vdir"] = fv.wrap_dir(np.degrees(np.arctan2(df["u"], df["v"])))

                # # Checked Ok
                # complex_vector = df["u"] + 1j * df["v"]
                # assert np.allclose(dfs_tcm[device]["|V|"], np.absolute(complex_vector))
                # # with mat to geo angle conversion (same as degrees(arctan2(eastward_wind, northward_wind)))
                # assert np.allclose(
                #     dfs_tcm[device]["Vdir"], (90 - np.angle(complex_vector, deg=True)) % 360
                # )
                del df["u"]
                del df["v"]


def load_incl(cfg):
    dfs, fs, fs_mean = {}, {}, {}
    with pd.HDFStore(cfg["in"]["path"], mode="r") as storeIn:
        for tbl in cfg["in"]["table"]:
            if cfg["in"]["USEtime"]:
                time_range = np.ravel(np.array(cfg["in"]["USEtime"], "M8[s]"))
                qstr_trange_pattern = "index>='{}' & index<='{}'"
                qstr = qstr_trange_pattern.format(*time_range)
            else:
                qstr = None
            dfs[tbl] = storeIn.select(tbl, where=qstr, columns=cfg["in"]["cols"])
            df_len = dfs[tbl].shape[0]
            try:
                fs[tbl] = 1 / np.diff(dfs[tbl].index[:2]).item().total_seconds()
                fs_mean[tbl] = (df_len - 1) / np.diff(
                    dfs[tbl].index[[0, -1]]
                ).item().total_seconds()
                print(
                    f"{tbl} {df_len}rows, fs0 = {fs[tbl]}Hz,",
                    f"fs_mean = {fs_mean[tbl]}Hz",
                )
                # if abs(fs - fs[tbl]) > 0.001 else "")
            except IndexError:
                print(f"No {tbl} data!", flush=True)
                continue
        return dfs, fs, fs_mean


def clip_to_whole_period(df, dt="10min"):  # "H"
    """
    Clip the DataFrame to the nearest dt boundaries relative to day's start
    to ensure whole timestamps
    :param df:
    :param dt:
    :raises ValueError:
    :return: clipped_df, rounded start time, rounded end time
    """
    # Ensure the index is a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")

    # Find the nearest rounded start time
    start_time = df.index[0].ceil(dt)
    end_time = df.index[-1].floor(dt)

    return df.loc[start_time:end_time], start_time, end_time


def fill_cols_reflecting(arr, b_bad=None):
    arr_filled = np.copy(arr)
    rows, cols = arr.shape

    # Create a boolean mask for NaNs
    if b_bad is None:
        b_bad = np.isnan(arr)

    # Find the indices of non-NaN elements in each row
    non_nan_indices = np.where(~b_bad)

    # Separate row and column indices
    row_indices, col_indices = non_nan_indices

    # Process each col separately
    nan_edges = np.zeros((2, cols))
    for c in range(cols):
        # Get all column indices for the current row that are not NaN
        i_ok = row_indices[col_indices == c]
        if len(i_ok) == 0:
            continue
        st, en = nan_edges[:, c] = i_ok[[0, -1]]

        # Fill leading NaNs by reflecting from the first non-NaN value
        if st > 0:
            lead_nans = np.arange(st - 1, -1, -1)
            reflected_indices = 2 * st - lead_nans  # <=> st - distances
            arr_filled[lead_nans, c] = arr[reflected_indices, c]

        # Fill trailing NaNs by reflecting from the last non-NaN value
        if en < rows - 1:
            trail_nans = np.arange(en + 1, rows)
            reflected_indices = 2 * en - trail_nans  # <=> en - distances
            arr_filled[trail_nans, c] = arr[reflected_indices, c]

    return arr_filled, nan_edges


def resample_to_freq(x, y, freq, time_st=None, time_end=None):
    print(f"Make constant frequency {freq} Hz...")
    out_index = np.arange(
        time_st.to_numpy() if time_st else x[0],
        time_end.to_numpy()
        if time_end
        else x[-1] + np.timedelta64(int((0.5 / freq) * 1e9), "ns"),
        np.timedelta64(int((1 / freq) * 1e9), "ns"),
    )
    arr_interp = np.empty((out_index.size, y.shape[1]))
    for c, arr_col in enumerate(y.T):
        arr_interp[:, c] = np.interp(
            out_index.astype(int),
            x.astype("M8[ns]").astype(int),
            arr_col,
        )
    return arr_interp


def decimate_df(
    df, original_freq, interp_freq=None, target_freq=None, target_period=None
):
    """
    Decimate the DataFrame using scipy's decimate function
    with first make constant frequency by interpolating to `interp_freq` frequency if set (optimally, I
    think, is to use mean freq)
    :param df:
    :param original_freq: Original sampling rate in Hz
    :param interp_freq: interpolating frequency if interpolation needed (for example if data has not
    constant frequency, which violates decimation function input requirements)
    :param target_freq: Target sampling rate in Hz
    :return:
    """
    # Calculate the decimation factor
    if not target_freq:
        target_freq = 1 / target_period
    if not target_period:
        target_period = int(1 / target_freq)

    # Clip data to ensure whole timestamps

    df, time_whole_st, time_whole_end = clip_to_whole_period(df, dt=f"{target_period}s")
    n_not_equal_dt = (np.diff(df.index.values.astype(int), 2).astype(bool)).sum()
    if n_not_equal_dt:
        print(f"data of not constant frequency found ({n_not_equal_dt})")

    # Apply decimation to each column
    arr = df.to_numpy()
    b_bad = np.isnan(arr)
    # Find the indices of non-NaN elements in each row
    arr, nan_edges = fill_cols_reflecting(arr, b_bad)
    n_nan_remains = np.isnan(arr).sum(axis=0)
    if n_nan_remains.any():
        print("Number of NaNs remains before decimation:", n_nan_remains)
    if (
        interp_freq
        and original_freq != interp_freq
        or n_nan_remains.any()
        or bool(n_not_equal_dt)
    ):
        # first make constant frequency by interpolating to interp_freq frequency
        arr_interp = resample_to_freq(
            df.index.values, arr, interp_freq, time_whole_st, time_whole_end
        )
        k_dec = int(interp_freq / target_freq)
        arr = arr_interp
    else:
        k_dec = int(original_freq / target_freq)
    print(f"decimating with factor {k_dec}...", end=" ")
    arr_dec = sp.decimate(arr, k_dec, ftype="fir", zero_phase=True, axis=0)

    # fill nan back to decimated data
    nan_edges = nan_edges / k_dec
    for c, (st, en) in enumerate(
        zip(np.int32(np.ceil(nan_edges[0, :])), np.int32(np.floor(nan_edges[1, :])))
    ):
        arr_dec[:st, c] = np.nan
        arr_dec[en:, c] = np.nan

    # Generate new timestamps for the desired frequency
    df_decimated_len = arr_dec.shape[0]
    df_decimated = pd.DataFrame(
        arr_dec,
        index=pd.date_range(
            time_whole_st, periods=df_decimated_len, freq=f"{target_period}s"
        ),
        columns=df.columns,
    )
    print("empty!" if df_decimated.empty else f"out size = {df_decimated_len}")
    return df_decimated


def fit_angular_regression(df, col_x=None, col_y=None):
    """
    Fits an angular regression model where both predictor (x) and response (y) are angles.
    Transforms both into sine and cosine components, fits separate OLS models,
    and reconstructs the predicted angles.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.
    col_x (str): The name of the column in df representing the predictor variable (angle).
    col_y (str): The name of the column in df representing the response variable (angle).

    Returns:
    np.ndarray: An array of predicted angles based on the fitted model.

    Example usage:
    Assuming 'df' is your DataFrame with appropriate 'angle_x' and 'angle_y' columns
    predicted_angles = fit_angular_regression(df, 'angle_x', 'angle_y')
    print(predicted_angles)
    """
    if col_x:
        # Ensure the specified columns exist in the DataFrame
        if col_x not in df.columns or col_y not in df.columns:
            raise ValueError("Specified column names must exist in the DataFrame")
    else:
        col_x, col_y = list(df.columns)

    # Transform both x & y angles into sine and cosine components
    # todo: use new var _ = {}
    for fun_str in ("sin", "cos"):
        for col in (col_x, col_y):
            df[f"{fun_str}_{col}"] = getattr(np, fun_str)(np.radians(df[col]))
    # Define predictors including the transformed cyclic variables for x
    X = sm.add_constant(df[[f"sin_{col_x}", f"cos_{col_x}"]])

    # Fit separate models for sine and cosine components of y
    predicted = []
    for fun_str in ("sin", "cos"):
        model = sm.OLS(df[f"{fun_str}_{col_y}"], X).fit()
        # Predicted values using the fitted models
        predicted.append(model.fittedvalues)  # = model.predict(X)

    # Combine predicted sine and cosine back into an angle
    predicted_angle = np.degrees(np.arctan2(*predicted)) % 360
    return predicted_angle


def fit_regression(df):
    x, y = df.values.T
    model_result = sm.OLS(y, sm.add_constant(x), missing="drop").fit()
    print("Fitting", " to ".join([f"{k} = {v}" for k,v in zip("xy", df.columns)]))
    print(model_result.summary())
    y_predicted = model_result.fittedvalues
    return y_predicted


def psd_welch(df, fs, nfft=4096):
    """
    Compute the Power Spectral Density using Welch's method on data which
    possibly of not constant frequncy and with NaNs
    :param df: data datafreame
    :param fs:
    :param nfft:
    :return: DataFrame
    """
    # Interpolating to constant freq index excluding NaNs from input
    b_ok = ~df.isna().any(axis=1)
    a = resample_to_freq(df.index[b_ok].values, df.values[b_ok], fs)
    psd = {}
    for i, col_prm_x in enumerate(df.columns):
        freq, psd[col_prm_x] = sp.welch(a[:, i], fs=fs, nperseg=nfft)
    df = pd.DataFrame(psd, index=freq)
    df.index.name = "freq"
    return df


def df_tcm_col_for_dev(col, st, device):
    """
    True if column col refers to data at station `st` and also for `device` if at `st`
    there were many devices
    """
    st_, *tcm_d = col.split("_")[1:]
    if st_.endswith(st):
        if len(tcm_d) > 1:
            return tcm_d[0] == device
        else:
            return True
    return False


#############################################################################################
if __name__ == "__main__":
    lang = "Ru"
    # 1. Configuration
    ## ADV configuration

    path_raw_adv = Path(
        r"B:\WorkData\Cruises(older)\_KaraSea\220906_AMK89-1\ADV_Sontek\_raw,to_txt,vsz"
    )
    file_paths = list(path_raw_adv.glob("**/*.dat"))
    # r"B:\WorkData\Cruises(older)\_KaraSea\220906_AMK89-1\ADV_Sontek\_raw,to_txt,vsz\7440A#D805,P=70m\7440A#D805.dat"

    with Path(path_raw_adv.with_name("info_devices.json")).open(encoding="utf8") as f:
        device_info_adv = json.load(f)
    depth = {
        device: v[2] if v[3] == ">" else v[1] - v[2]
        for device, v in device_info_adv.items()
    }

    config = {
        # "USEtime": []  # - instead we use intervals loaded from info_devices.json
        "USE_timeShift_s": 3600 * 3,
        "dt_bin": 600,  # 10 minutes / 3600,  # 1 hour
    }

    ## TCM configuration
    cfg_tcm = {
        "in": {
            "USEtime": [["2022-09-10T10:26", "2022-09-14T07:57"]],
            "path": Path(
                r"B:\WorkData\Cruises(older)\_KaraSea\220906_AMK89-1\inclinometer\220910.proc_noAvg.h5"
            ),
            "table": ["i_b21", "i_b22"] + ["i37", "i38"],  #
            "cols": ["u", "v"],
        }
    }
    with Path(cfg_tcm["in"]["path"].with_name("info_devices.json")).open(
        encoding="utf8"
    ) as f:
        device_info_tcm = json.load(f)
    for device, v in device_info_tcm.items():
        depth[device] = v[1] - v[2]

    adv_str = "ADV"
    tcm_str = "TCM"
    stations = sorted({file_path.stem.split("#", 1)[0] for file_path in file_paths})
    stations_tcm = {
        v[0]
        for device, v in device_info_tcm.items()
        if device in cfg_tcm["in"]["table"]
    }

    ## Renaming TCM settings to associate with station where one TCM on station only
    tcm2st = {
        device: v[0]
        for device, v in device_info_tcm.items()
        if device in cfg_tcm["in"]["table"]
    }
    # Keep device suffix where TCM to station mapping is not unique
    if len(tcm2st.values()) > len(stations_tcm):
        n_devs_st = {st: 0 for st in stations_tcm}
        for st in tcm2st.values():
            n_devs_st[st] += 1
        tcm2st = {
            device: f"{st}_{device}" if n_devs_st[st] > 1 else st
            for device, st in tcm2st.items()
        }

    ## Renaming TCM on stations settings
    # to associate with depth where one TCM with such depth on station
    # keeping device suffix where TCM to depth mapping is not unique
    n_devs_st = {
        (v[0], depth[device]): 0
        for device, v in device_info_tcm.items()
        if device in cfg_tcm["in"]["table"]
    }
    for device, v in device_info_tcm.items():
        if device in cfg_tcm["in"]["table"]:
            n_devs_st[v[0], depth[device]] += 1
    tcm2depth = {
        device: (
            f"{depth[device]}m_{device}"
            if n_devs_st[v[0], depth[device]] > 1
            else f"{depth[device]}m"
        )
        for device, v in device_info_tcm.items()
        if device in cfg_tcm["in"]["table"]
    }

    # 2. Loading data
    # If exist, we use saved data to faster load next time for plot or further processing

    # Load data processed here earlier
    path_db = path_raw_adv.with_name(f"st{','.join(stations)}@{adv_str}+{tcm_str}.h5")

    dfs = {}  # stations dict with values of all devices data merged into one dataframe
    psds_tcm_orig = {}
    psds_at_st = {
        st: {} for st in stations
    }  # All PSD with fs of ADV nested in stations dict

    if not path_db.is_file():  # remove `path_db` file if need original TCM data
        ## ADV

        print(f"Loading {len(file_paths)} {adv_str} data files...")
        bin, stat, dfs_adv, dfs_adv_uv, fs_adv = {}, {}, {}, {}, {}
        for file_path in file_paths:
            # file_path = Path(file_path)
            device = file_path.stem.split("#", 1)[-1]
            # Load data
            df_adv_orig, fs_adv[device] = load_sontek_adv(
                file_path, time_add_s=-config["USE_timeShift_s"]
            )
            # Process data
            config["USEtime"] = [device_info_adv[device][6:8]]
            out, bin[file_path.stem], stat[file_path.stem] = process_adv(
                df_adv_orig, config
            )

            dfs_adv[file_path.stem] = pd.DataFrame(
                {
                    f"{k}_{adv_str}{depth[device]}m": v
                    for k, v in out.items()
                    if k in ("|V|", "Vdir", "u", "v")
                },
                index=out["time"],
            )

            # For PSD
            dfs_adv_uv[file_path.stem] = pd.DataFrame.from_records(
                {
                    f"{k}_{adv_str}{depth[device]}m": v
                    for k, v in out.items()
                    if k in ("u", "v")
                },
                index=out["time"],
            )

        ## TCM

        print(
            f"Loading {len(cfg_tcm['in']['table'])} {tcm_str} tables from {cfg_tcm['in']['path'].name}..."
        )
        dfs_tcm, fs0_tcm, fs_mean_tcm = load_incl(cfg_tcm)

        # assuming all ADV original datasets sampled at same fs_adv frequency
        fs0_adv = next(iter(fs_adv.values()))

        # 3. Multi-scale statistics analysis

        for dt in [10 * 60]:  # , 3600 averaging scales
            # t=10 (corresponded to target_freq=fs0_adv= 1/10 = 0.1Hz) have been already processed

            # Interpolate all to single index with this frequency before get statistics / correlation
            target_freq = 1 / dt

            ## 3.1 TCM Decimation (reduction) from original freq (~5Hz) to `target_freq` on u, v
            # and concatenation (only after decimation to set same overlapped indexes, because else, if
            # indexes of devices not matches or there is abscence of data pd.concat() will change
            # frequency of result)
            df_tcm = pd.concat(
                [
                    decimate_df(
                        dfs_tcm[device].rename(
                            {
                                col: f"{col}_{tcm_str}{tcm2st[device]}_{depth[device]}m"
                                for col in ["u", "v"]
                            },
                            axis="columns",
                        ),
                        original_freq=fs_mean_tcm[device],
                        interp_freq=fs0_tcm[device],
                        target_freq=target_freq,
                    )
                    for device in dfs_tcm.keys()
                ],
                axis=1,
            )

            if target_freq == fs0_adv:
                ## 3.2 Do following only once
                ### All PSD with time resolution of ADV nested in dict by stations

                psds_at_st = {st: {} for st in stations_tcm}

                #### PSD of decimated TCM (to check only, as a full resolution PSD will be calculated also)
                for device in (
                    dfs_tcm.keys()
                ):  # use original loaded data only to get TCM devices names
                    st = tcm2st[device].split("_")[0]
                    params_st = [
                        c for c in df_tcm.columns if df_tcm_col_for_dev(c, st, device)
                    ]
                    psds_at_st[st][f"{tcm_str}{tcm2depth[device]}"] = psd_welch(
                        df_tcm[params_st], target_freq, nfft=4048
                    ).rename(lambda c: c.split("_")[0], axis="columns")
                    # not need `st` in name of columns again

                #### PSD of ADV
                for st in stations:
                    for st_device, df in dfs_adv_uv.items():
                        _, device = st_device.split("#")
                        if st == _:  # st_device.startswith(st)
                            psds_at_st[st][f"{adv_str}{depth[device]}m"] = psd_welch(
                                df, target_freq, nfft=4048
                            ).rename(lambda c: c.split("_")[0], axis="columns")

                ### Statistics of and between inclinometers (on all loaded stations) wirh original freq

                fs0_tcm = next(iter(fs0_tcm.values()))
                if True:
                    # Interpolating to constant freq index excluding NaNs from input
                    dfs_tcm_interp = []
                    for i, (device, df) in enumerate(dfs_tcm.items()):
                        b_ok = ~df.isna().any(axis=1)
                        # Find the nearest rounded start time so different dataframes indexes will mostly
                        # match here only for rounding the start of target index
                        target_period = int(1 / target_freq)
                        time_whole_st = df.index[0].ceil(f"{target_period}s")
                        a_interp = resample_to_freq(
                            df.index[b_ok].values,
                            df.values[b_ok],
                            fs0_tcm,
                            time_st=time_whole_st,
                        )
                        dfs_tcm_interp.append(
                            pd.DataFrame(
                                a_interp,
                                index=pd.date_range(
                                    time_whole_st,
                                    periods=a_interp.shape[0],
                                    freq=f"{1 / fs0_tcm}s",
                                ),
                                columns=[
                                    f"{col}_{tcm_str}{tcm2st[device]}"
                                    for col in df.columns
                                ],
                            )  # with appending ["u", "v"] column names with station info
                        )
                    df_tcm_interp = pd.concat(dfs_tcm_interp, axis=1)

                    to_polar(df_tcm_interp)  # makes ["|V|", "Vdir"]

                    # - calc
                    df_stats, df_stats_corr = stat_wind.get_stat(df_tcm_interp, v="V")
                    # - save
                    stat_wind.save_stat(
                        df_stats,
                        df_stats_corr,
                        path_base=path_raw_adv.with_name(
                            "{}_{}Hz_st{}".format(
                                tcm_str,
                                # prevent creating file suffixes which will be replaced:
                                f"{fs0_tcm:g}".replace(".", "p"),
                                ",".join(tcm2st.values()),
                            )
                        ),
                        v="V",
                    )

                dfs_adv_dec = dfs_adv
            else:
                ## 3.2a ADV Decimation (reduction) from original freq (0.1Hz) to `target_freq` on u, v

                dfs_adv_dec = {
                    k: decimate_df(
                        df.filter(regex="^[uv]_"),
                        original_freq=fs0_adv,
                        interp_freq=fs0_adv,
                        target_freq=target_freq,
                    )
                    for k, df in dfs_adv.items()
                }
                to_polar_dfs(dfs_adv_dec, b_device_in_col_suffix=True)

            ## 3.3 Statistics analysis for current time resolution scale

            # |V| & Vdir of ADV + TCM, decimated to ADV freq, in single dataframe per each station

            to_polar(df_tcm)  # get |V| & Vdir for statistics (removes u, v)
            # Convert tz-aware index to tz-naive to be compatible with ADV index
            df_tcm.index = df_tcm.index.tz_localize(None)

            # Dataframes with all devices data
            for st in stations:
                dfs[st] = pd.concat(
                    [v for k, v in dfs_adv_dec.items() if k.startswith(st)]
                    + [
                        df_tcm.filter(like=st).rename(
                            lambda col: col.replace(f"{st}_", ""),
                            axis="columns",
                            # not need `st` in name as one TCM on station
                        )
                    ],
                    axis=1,
                )

            # All devices data statistics
            for st in stations:
                df_stats, df_stats_corr = stat_wind.get_stat(dfs[st], v="V")
                stat_wind.save_stat(
                    df_stats,
                    df_stats_corr,
                    path_base=path_raw_adv.with_name(f"st{st}_avg-dec={int(dt)}s"),
                    v="V",
                )

        # 4. PSD of TCM from its original frequency resolution data

        psds_tcm_orig = {}
        for device, df in dfs_tcm.items():
            psds_tcm_orig[device] = psd_welch(df, fs0_tcm[device])

        # 5. Save to HDF to faster load next time for plot or further processing
        print(f"Saving to {path_db}...", end="")
        with pd.HDFStore(path_db, mode="a") as store:
            # dfs that is needed for regression plots
            for st, df in dfs.items():
                df.rename(
                    lambda col: col.replace("|V|", "Vabs"), axis="columns"
                ).to_hdf(
                    store,
                    key=f"st{st}",
                    append=True,
                    data_columns=True,
                    format="table",
                    index=False,
                    # dropna=
                )

            # PSDs that may be needed for visualisation with better quality i.e. in Veusz
            for device, df in psds_tcm_orig.items():
                df.to_hdf(
                    store,
                    key=f"PSD_{device}_orig",
                    append=True,
                    data_columns=True,
                    format="table",
                    index=False,
                    # dropna=
                )

            for st, psds in psds_at_st.items():
                for device, df in psds.items():
                    df.to_hdf(
                        store,
                        key=f"st{st}/PSD_{device}",
                        append=True,
                        data_columns=True,
                        format="table",
                        index=False,
                        # dropna=
                    )
        print("ok.")
    else:
        # 2. Load from DB
        print(f"Loading from {path_db}...", end="")
        with pd.HDFStore(path_db, mode="r") as store:
            # dfs that is needed for regression plots
            for st in stations:
                dfs[st] = store[f"st{st}"].rename(
                    lambda col: col.replace("Vabs", "|V|"), axis="columns"
                )

            # PSDs that may be needed for visualisation with better quality i.e. in Veusz
            devices_TCM = device_info_tcm.keys()
            for device in devices_TCM:
                try:
                    psds_tcm_orig[device] = store[f"PSD_{device}_orig"]
                except KeyError:
                    continue  # not all TCMs# are needed (not saved intentionally)

            devices_ADV = [file_path.stem.split("#", 1)[-1] for file_path in file_paths]
            for st in stations:
                for dev_str, devices, dev2dep in (
                    (
                        adv_str,
                        devices_ADV,
                        {device: f"{d}m" for device, d in depth.items()},
                    ),
                    (tcm_str, list(devices_TCM), tcm2depth),
                ):
                    for device in devices:
                        try:
                            psds_at_st[st][device] = store[
                                f"st{st}/PSD_{dev_str}{dev2dep[device]}"
                            ]
                        except KeyError:
                            continue  # not all TCMs# are needed (not saved intentionally)
        print("ok.")

    # 3. Plotting
    print("Plotting")
    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from matplotlib.ticker import LogLocator, AutoMinorLocator, MultipleLocator
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    matplotlib.rcParams["axes.linewidth"] = 1.5
    matplotlib.rcParams["figure.figsize"] = (16, 7)
    matplotlib.rcParams["font.size"] = 12
    matplotlib.rcParams["axes.xmargin"] = 0.001  # (default: 0.05)
    matplotlib.rcParams["axes.ymargin"] = 0.01

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from mpl_toolkits.axes_grid1.axes_size import Fixed, Scaled

    try:
        matplotlib.use(
            "Qt5Agg"
        )  # must be before importing plt (raises error after although documentation said no effect)
    except ImportError:
        pass
    matplotlib.interactive(True)
    plt.ion()
    # import seaborn as sns
    # sns.set_theme(style="whitegrid")

    str_abs, str_dir = "|V|", "Vdir"
    n_params = 2  # number of (above) parameters to plot
    b_plot_psd = False
    if b_plot_psd:
        ## 3.2 Plot PSD
        # Plot original TCM data u&v spectrums
        nrows = len(psds_tcm_orig)
        for b_log_x in [False, True]:
            for b_gray in [False, True]:
                fig, axes = plt.subplots(nrows=nrows, sharex=True, sharey=True)
                for i, ((device, df), ax) in enumerate(
                    zip(psds_tcm_orig.items(), axes)
                ):
                    plot.vs_freq(
                        df,
                        ax=ax,
                        legend_title=re.sub("^[i_]+", "", device),
                        b_log_x=b_log_x,
                        b_gray=b_gray,
                        path_dir=path_raw_adv.parent
                        if i == nrows - 1
                        else None,  # save after draw on last ax
                    )

        # Plot original TCM |V| spectrum
        df = next(iter(psds_tcm_orig.values()))  # to get index
        Vabs = pd.DataFrame(
            {
                re.sub("^[i_]+", "", device): np.hypot(*df.values.T)
                for device, df in psds_tcm_orig.items()
            },
            index=df.index,
        )
        for b_log_x in [False, True]:
            for b_gray in [False, True]:
                fig_psd, axes = plt.subplots()
                plot.vs_freq(
                    Vabs,
                    axes,
                    legend_title=("TCM {#}").format_map(fv.I),
                    b_log_x=b_log_x,
                    b_gray=b_gray,
                    path_dir=path_raw_adv.parent,
                    ylabel_prefix="PSD(|V|)",
                    save_name_parts=(
                        "gray_subdir?",
                        "ylabel_prefix",
                        "@TCMs",
                        "log(x)?",
                        ".png",
                    ),
                    colors=["red", "green", "blue", "black"],
                )

        # Plot ADV data spectrum and TCM decimated to same freq for comparison and allow to check decimation
        for b_log_x in [True, False]:
            for i, (st, psds) in enumerate(psds_at_st.items()):
                nrows = len(psds)
                if i == 0:
                    fig_height = plot.figure_height_for_same_axes_height(fig_psd, nrows)
                fig_psd = plt.figure(figsize=(fig_psd.get_size_inches()[0], fig_height))
                axes = fig_psd.subplots(nrows=nrows, sharex=True, sharey=True)
                for i, ((device, df), ax) in enumerate(zip(psds.items(), axes)):
                    plot.vs_freq(
                        df,
                        ax,
                        legend_title=device,
                        b_log_x=b_log_x,
                        path_dir=path_raw_adv.parent
                        if i == nrows - 1
                        else None,  # save after draw on last ax
                        ylabel_prefix=f"PSD(u,v)_st{st}",
                    )

    ## 3.3 Plot regressions
    dt = (
        np.diff(next(iter(dfs.values())).index[:2].to_numpy("M8[s]")).astype(int).item()
    )
    print("Plot regressions...", f"Curremt data is decimated to dt={int(dt)}s")
    # Plot regressions
    nrows, ncols = 2, 2  # stations, parameters
    wspace, hspace, cbar_space, cbar_height = 0.1, 0.1, 0.03, 0.8

    # Figure per station
    for st, df in dfs.items():
        # Convert dates to days for color points and label colorbar
        days_start = df.index[0].floor("d")
        df["days"] = (df.index - days_start).total_seconds() / 86400

        # fig, axes = plot.create_equal_subplots(
        #     ncols,
        #     nrows_per_col=[nrows, nrows],
        #     figsize=(12, 12),
        #     constrained_layout=True
        # )
        fig, axes = plot.create_uniform_subplots(
            ncols, nrows, figsize=(12, 12), constrained_layout=True
        )

        # figsize=plot.fig_size(ncols, nrows, subplot_width=5)----
        # fig = plt.figure(figsize=figsize)
        # # Use GridSpec to create a 2x2 grid with extra columns for colorbars
        # gs = GridSpec(
        #     nrows, 2*ncols, figure=fig, wspace=wspace, hspace=hspace,
        #     width_ratios=[1, cbar_space] * ncols
        # )

        axs, cbars = [], []
        # Axes column per parameter
        for i_prm, (str_prm, str_unit) in enumerate([(str_abs, "m/s"), (str_dir, "°")]):
            # Debug:
            # st, df = next(iter(dfs.items()))
            # i_prm, str_prm = 0, str_abs
            cols_prm = [c for c in df.columns if c.startswith(str_prm)]
            len_str_prm_prefix = len(str_prm) + 1
            devs = [c[len_str_prm_prefix:] for c in cols_prm]
            print(st, devs)
            icol_x = devs.index([dev for dev in devs if dev.startswith(tcm_str)][0])
            col_prm_y = cols_prm[icol_x]
            plot.regression(
                df[cols_prm + ["days"]],
                col_prm_y=col_prm_y,
                predict_fun=(
                    fit_angular_regression
                    if col_prm_y.startswith("Vdir")
                    else fit_regression
                ),
                axes=axes[i_prm],
                str_unit=str_unit,
                days_start=days_start,
                lang=lang,
            )
            # integer ticks
            # cbar.set_ticks(
            #     range(
            #         int(df["days"].min()),
            #         int(df["days"].max()) + 1,
            #     )
            # )
            # axs.append(ax)
            # cbars.append(cbar)

            # for i, ax in enumerate(axs):
            #     pos = ax.get_position()

            #     # To ensure all graphs have the same height, we need to manually adjust the
            #     # position of each subplot
            #     new_pos = [pos.x0, pos.y0, pos.width, 1 / nrows]
            #     ax.set_position(new_pos)

            #     # Ensure colorbars are positioned right after the plot axes
            #     cbar_ax = cbars[i].ax
            #     cbar_pos = cbar_ax.get_position()

            #     # Set the left edge of the colorbar to the right edge of the plot
            #     cbar_ax.set_position(
            #         [pos.x1, cbar_pos.y0, cbar_pos.width, cbar_height / nrows - hspace]
            #     )

            # plt.subplots_adjust(wspace=0.1, hspace=0.1)  # Reduce the space between subplots
            # plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        # Automatically adjust data limits to fill axes
        for ax_col in axes:
            min_x0, max_x1 = 1e9, -1e9
            for ax, axc in ax_col:
                ax_pos = ax.get_position()
                if ax_pos.x0 < min_x0:
                    min_x0 = ax_pos.x0
                if ax_pos.x1 > max_x1:
                    max_x1 = ax_pos.x1

            for ax, axc in ax_col:
                ax.autoscale(enable=True, tight=True)
                ax.apply_aspect()
                plot.force_equal_ticks(ax, select_fun=min)

        try:  # Stop here to manually extend width of figure if graphs not fit!
            file_fig = path_raw_adv.with_name(
                f"Reg_{st}@{','.join(devs)}_avg-dec={int(dt)}s.png"
            )
            fig.savefig(
                file_fig,
                dpi=300,
                bbox_inches="tight",
            )
            print(file_fig.name, "saved")
        except Exception as e:
            print(f"Can not save fig to {file_fig}: {e}")

            # sns.lmplot(x="FlyAsh", y="Strength", hue="AirEntrain", data=con);

        # plt.show()

        # plt.figure(figsize=(10, 6))
        # plt.plot(out["time"], out["speed"], label="Speed")
        # plt.xlabel("Time (s)")
        # plt.ylabel("Speed (m/s)")
        # plt.title("ADV Data")
        # plt.legend()
        # plt.show()
        # print("ok>")
