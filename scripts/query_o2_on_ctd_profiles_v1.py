# query_o2_on_ctd_profiles(time,depth,up_or_down).py
# %% imports
from datetime import datetime
from functools import partial
import pandas as pd
import numpy as np

# import matplotlib.pyplot as plt
# from matplotlib import cm

from operator import sub
from pathlib import Path
import re
from typing import Any, Callable, Dict, Optional, Sequence
from contextlib import nullcontext

import gsw

from hdf5_pandas import h5
from vsz_loader import veusz_load_hdf5_ctd_profile
from utils.logging_config import setup_logging
from veusz_helpers.common import func_vsz as fv
from get_datasets import d_utils as ds_utils

l = setup_logging(__name__, console_format_args={"name": False, "funcName": False, "datefmt": "%H:%M:%S"})

# %% Define constants for column names
col_O = "O2ppm"
col_T = "Temp"
col_P = "Pres"

col_show = {
    col_O: {
        "letter": "O",
        "units": "mg/l",
        "long name": "estimated mass concentration of oxygen in sea water",
    },
    col_T: {"letter": "t", "units": "°C", "long name": "sea water temperature"},
}

fmt_time_print = "%y-%m-%d %H:%M"
fmt_time_filename = "%y%m%d_%H%M"

# %% Paths
path_cruise = Path(r"D:\Cruises\BalticSea\251201_ABP64")

# Path to device data DB and device table inside
path_db = (path_cruise / path_cruise.name.split("@", 1)[0]).with_suffix(".not_cor_O2.h5")
device = "CTD_SST_48Mc#1253"

path_save = Path(r"D:\WorkData\experiment\SST48Mc_calibr(O2)-in_ABP64")

#
def get_table(h, i_ranges, grp_d, time_range_raw):
    tbl_path = grp_d["table"]
    i_range = i_ranges["table"]
    data = h[tbl_path][:][slice(*i_range)]
    return pd.DataFrame.from_records(
        data, exclude=["index"], index=pd.DatetimeIndex(data["index"], dtype="M8[ns]")
    )


# %% Load query parameters (time and depth) from Excel xlsx file
dtype = (
    {
        "Date_Time": str,  # "M8[s]" not works
        "St": str,
        "Lat": float,
        "Lon": float,
        "Depth_ref": float,
        f"{col_O}_ref": float,
    }
)
with pd.ExcelFile(path_save / "Кислород_АБП64_ЮВБ.xlsx") as f_xls:
    xls = pd.read_excel(
        f_xls,
        # sheet_name='Лист1',
        usecols="A,B,H,I,J,K,L",
        header=None,
        names=["Date", "Time"] + list(dtype.keys())[1:],
        skiprows=3,
        parse_dates=[[0, 1]],  # only combines columns, not converts to dates!
        # index_col="Date_Time",  # forward filling not works!
        # Not works(!):
        # date_format={"Date": "%d/%m/%Y", "Time": "%H:%M:%S"},  # 02/12/2025	02:54:32
        dtype=dtype,
        na_values=[" ", "nan nan"],  # allows to load required types
    )

xls.Date_Time[xls.Date_Time.str.startswith("nan")] = np.nan
xls.dropna(how="all", inplace=True)

date_time = pd.Series([pd.NaT]*xls.Date_Time.size)
for date_fmt in ("%d/%m/%Y", "%Y-%m-%d"):
    date_re = date_fmt.replace("%d", r"\d{2}").replace("%m", r"\d{2}").replace("%Y", r"\d{4}")
    b_date_ok = xls.Date_Time.str.match(date_re).fillna(False)
    date_time[b_date_ok] = pd.to_datetime(xls.Date_Time[b_date_ok].str.replace("00:00:00 ", ""), format=f"{date_fmt} %H:%M:%S")
xls.Date_Time = date_time.ffill()
# nav_df = xls['Lat/Lng'].str.extract(r'(?P<Lat>[^,]*), (?P<Lon>[^,]*)')
xls.dtypes
# %% Load CTD profiles
profiles_starts = xls.Date_Time[xls.Date_Time.diff() != pd.Timedelta(0)]
l.info(
    f"Loading profiles for {profiles_starts.size} reference data dates"
)
data_prof_dict = {}
max_time_diff = pd.Timedelta(seconds=600)
i_del = []
for i, time_st in enumerate(profiles_starts):
    data_prof_dict[time_st] = veusz_load_hdf5_ctd_profile(
        path_db,
        [time_st, np.nan],
        device=device,
        fun_custom=get_table,
        time_shift_s=1800,  # shifts query time back because function searches profile after
    ).dropna(how="all")
    time_diff = time_st - data_prof_dict[time_st].index[0]
    l.info(f"{i}. {data_prof_dict[time_st].index[0]} for {time_st}: diff = {time_diff}")

    if abs(time_diff) > max_time_diff:
        i_del.append(i)
# %%
if len(i_del):
    l.info(
        f"Deleting {len(i_del)} profiles and corresponding reference data where time diff to found data > {max_time_diff} (#{i_del})"
    )
    for time_st in profiles_starts.iloc[i_del]:
        del data_prof_dict[time_st]
        xls = xls[xls.Date_Time!=time_st]
    xls.reset_index(drop=True, inplace=True)

# %% Select required data for each profile
dfs_up = []
dfs_down = []
for time_st, df_run in data_prof_dict.items():
    query = xls[xls.Date_Time == time_st]

    # Search in down and up profiles separately
    idx_P_max = df_run[col_P].argmax()
    l.info(f"Profile {time_st} depths")
    for b_up, df in enumerate([df_run.iloc[:idx_P_max], df_run.iloc[idx_P_max:]]):
        depth_q = query["Depth_ref"]
        if b_up: # to find next value along movement (better accounts for probe delay)
            df[col_P] = -df[col_P]
            depth_q = -depth_q

        df = df.sort_values(by=col_P)
        idx = df[col_P].searchsorted(depth_q)

        if b_up:  # recover correct values back
            df[col_P] = -df[col_P]
            depth_q = -depth_q

        df = df.iloc[np.clip(idx, 0, df.shape[0]-1), :]
        l.info(
            f"{df[col_P].tolist()} for {query['Depth_ref'].tolist()} depths on run {'up' if b_up else 'down'}"
        )
        (dfs_up if b_up else dfs_down).append(df)

df_up = pd.concat(dfs_up)
df_down = pd.concat(dfs_down)

# %%  Save combined CTD and reference data (if no output DB file exist)
db_path = path_save / f"{device}_o2_data_found.h5"
b_save = not db_path.is_file()
with pd.HDFStore(db_path) if b_save else nullcontext() as db:
    for sfx, df in [("up", df_up), ("down", df_down)]:
        df = pd.concat(
            (df[[col_P, col_O]].reset_index().rename(columns={"index": "Time"}), xls[["Date_Time", "Depth_ref", f"{col_O}_ref", "St"]]), axis=1
        )  # not works without rename()!
        tbl = f"runs_{sfx}"
        if b_save:
            l.info(f"writing to {db_path}/{tbl}")
            df.to_hdf(
                db,
                key=f"runs_{sfx}",
                append=True,
                data_columns=True,
                min_itemsize=32,  # for text St column
                format="table",
                index=False,
                # dropna=True,
            )
        else:
            l.info("Skipping saving: output DB exist")

# %% Not tested (instead data was fitted in Veisz)
profiles_starts = xls.Date_Time[xls.Date_Time.diff() != pd.Timedelta(0)]
l.info(f"processing {profiles_starts.size} profiles")
for i, time_st in enumerate(profiles_starts):
    coef_order = [1, 2]

coefs = {i: [] for i in coef_order}
fits = {i: [] for i in coef_order}
for i, param in enumerate(params):
    u = data[:, i_col_params_st + i]  # probe data
    ref = p if i == 0 else t
    for order in coef_order:  # each order fit
        coef = np.polyfit(u, ref, order)
        coefs[order].append(coef)
        fits[order].append(np.polyval(coef, u))
        resid = ref - fits[order][i]
        print(f"{i}.order={order}. {param}:", resid, f"={ref}-{fits[order][i]}")

# %% Calc Saturation
# from seawater import satO2
# Solubility = satO2(df.S.values, df["T"].values)  # salinity [psu (PSS-78)], temperature [℃ (ITS-68)],

def mmol2mg(x):
    """
    'mmol m-3' to 'mg l-1'
    """
    return x*0.0319988

def DO(O2ppm, Sal, Temp, Pres, Lat=55.3, Lon=19.7):
    # default coords Lat=55.3, Lon=19.7 are mean for calibration data
    SA = gsw.SA_from_SP(Sal, Pres, lat=Lat, lon=Lon)  # Absolute Salinity  [g/kg]
    conservative_temperature = gsw.conversions.CT_from_t(SA, Temp, Pres)
    pt = gsw.pt_from_CT(SA, conservative_temperature)
    Solubility = mmol2mg(gsw.O2sol_SP_pt(Sal, pt))
    DO = O2ppm * 100 / Solubility
    return DO


col_S = "Sal"
col_DO = "O2"  # raw
col_DO_cor = "O2%"  # from raw O2ppm (should be same)
for sfx, df in [("up", df_up), ("down", df_down)]:
    df = pd.concat(
        (
            df[[col_O, col_P, col_T, col_S, col_DO]].reset_index().rename(columns={"index": "Time"}),
            xls,
        ),
        axis=1,
    )  # not works without rename()!
    # df[col_DO_cor] = DO(*(df[[col_O, col_S, col_T, col_P, "Lat", "Lon"]].values.T))


    SA = gsw.SA_from_SP(df[col_S], df[col_P], lat=df["Lat"], lon=df["Lon"])  # Absolute Salinity  [g/kg]
    conservative_temperature = gsw.conversions.CT_from_t(SA, df[col_T], df[col_P])
    df["pt"] = gsw.pt_from_CT(SA, conservative_temperature)
    Solubility = mmol2mg(gsw.O2sol_SP_pt(df[col_S], df["pt"]))

    # bad:
    # Solubility2 = mmol2mg(1e-3*fv.oxygen_solubility_scor(df[col_T], df[col_S], df[col_P]))
    # Solubility3 = fv.oxygen_solubility(df[col_T], df[col_S])

    df[col_DO_cor] = df["O2ppm"] * 100 / Solubility
    df["dDO"] = df[col_DO_cor] - df[col_DO]
# %%
