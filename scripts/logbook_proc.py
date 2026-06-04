"""
# Logbook format
to able be processed by GPSBabel program
(so it has format configuration defined columns in corresponding GPSBabel *.style file:
D:/WorkData/~pattern~/txt2gpx/logbook.style) but hardcode the loading here
```
DateTime	Lat	Lon	Speed	Depth	Remarks
2025-12-01 14:36Z	054° 38.1660' N	019° 45.6768' E	0.37 kts	 17.3 м 	001. Начало работ Глубина по эхолоту 19.1 м
```

In `Remarks` column the Station is mostly at beginning with removed `st_pefix`
To do this replace using regex:
re.sub(
   "(?:([^\\t]*)(?:(?:(?:Начало (?:работы? на)? ?ст(?:анции|\\.)?|[cС]т\\.?)+? ?АБП\\d\\d)([\\d_]+).?) *([^\\r\\n\\t]*))|([^\\t]+)[. ]*$",
  "$4(?{2}$2. $1$3)"
)
and then all "\\t(.*)(АБП\\d\\d)([\\d_]+)(.*)" with r"\\t$3. $1$4"


"""
# %% imports
from datetime import datetime
from functools import partial
import pandas as pd
import numpy as np
import re
# import matplotlib.pyplot as plt
# from matplotlib import cm

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence
from contextlib import nullcontext

from hdf5_pandas.csv_specific_proc import deg_min_float_as_text2deg
from utils.logging_config import setup_logging

from utils.filters import inearestsorted
from utils.time import datetime_fun, check_time_diff
from veusz_helpers.common.vsz_loader import get_path_in_parents

l = setup_logging(__name__, console_format_args={"name": False, "funcName": False, "datefmt": "%H:%M:%S"})

# %% Load logbook for Time and Stations names

# Paths

path_logbook = Path(
    r"D:\Cruises\BalticSea\260421_ABP65\navigation\260421-0505_logbook-fmt.tsv"
    # r"D:\Cruises\BalticSea\251201_ABP64\navigation\251201-20_logbook-fmt.tsv"
)

path_cruise = get_path_in_parents(path_logbook.parent, 'navigation', target_is_dir=True).parent
cruise = re.match(r"(?P<year>\d\d)\d+_*(?P<vessel>\D+)(?P<num>\d+)", path_cruise.stem).groupdict()
st_prefix = "{vessel}{num}".format_map(cruise)  # "АБП64"

# %%

re_lat = re.compile(r"0*(\d+)° (\d+\.\d+)' [NS]")  # "054° 38.1660' N"
re_lon = re.compile(r"0*(\d+)° (\d+\.\d+)' [EW]")  # "019° 45.6768' E"

def str_deg_min_to_deg(str_d, str_m):
    try:
        return float(str_d) + float(str_m)/60
    except AttributeError:  # 'NoneType' object has no attribute 'groups'
        return np.nan

def lat_cnv(lat):
    try:
        return str_deg_min_to_deg(*re_lat.match(lat).groups())
    except AttributeError:  # 'NoneType' object has no attribute 'groups'
        return np.nan

def lon_cnv(lon):
    return str_deg_min_to_deg(*re_lon.match(lon).groups())

def float_from_1st_word(spd):
    try:
        return float(spd.split(" ", 1)[0])
    except ValueError: # could not convert string to float: ''
        return np.nan

dtype = {
    "Date": None,  # can not set type as it specified by `parse_dates`
    "Lat": float,
    "Lon": float,
    "Speed": float,
    "Depth": str,
    "Remarks": str,
    # "St": str,  # Station содержится в "Remarks"
}
df_lb = pd.read_csv(
    path_logbook,
    parse_dates=["Date"],
    # dtype=dtype,
    # skiprows=1,  # if uncomment deletes row under header
    header=0,  # required to set custom column names:
    names=list(dtype.keys()),
    index_col=["Date"],
    date_format="ISO8601",
    converters={"Lat": lat_cnv, "Lon": lon_cnv, "Speed": float_from_1st_word, "Depth": float_from_1st_word},
    sep="\t",
    skipinitialspace=True,
)
# Station is mostly at beginning with removed `st_pefix` because of hoq logbook was formatted
df_lb["St"] = df_lb["Remarks"].str.extract(r"^(\d{3,}[_\d]*)")
b_na = df_lb["St"].isna()
# find more records about stations in rows were have been not found
df_lb.loc[b_na, "St"] = df_lb.loc[b_na, "Remarks"].str.extract(rf"{st_prefix}(\d\d\d[_\d]*)")
df_lb["St"] = df_lb["St"].ffill()

# Check that stations are in increasing order
st_parts = df_lb["St"].str.split("_", expand=True, n=2)
st_parts.loc[st_parts[1].isnull(), 1] = "0"
n_digits = int(st_parts[1].str.len().max())
st_parts[1] = st_parts[1].str.rjust(n_digits, fillchar='0')
st_float = (st_parts[0] + "." + st_parts[1]).astype(float)  # or df['First'].str.cat(df['Last'], sep=' ', na_values='')
if not st_float.is_monotonic_increasing:
    i_dec = np.flatnonzero(st_float.diff() < 0)
    l.warning(f"Found decreasing station numbering at indexes {i_dec} - St: {df_lb['St'].values[i_dec]}")  # may be this is ok
    df_lb = df_lb.sort_index()
else:
    l.info("Checked - ok: all St are in increasing order")


# %%

# %% Rename files that have not station in name by adding suffix "st{station_number}"
# need update only files that has such stem:
glob_name_update = "[0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9]*"
paths_to_rename = [
    path_cruise / f
    for f in [
        r"CTD\SST48\Exported\*.csv",
        r"CTD\SST48\RAW\*.SRD",
        r"CTD\SAIV\Exported\*.txt",
        r"CTD\MWS12\Exported\*.txt"
    ]
]

for p in paths_to_rename:
    files = list(p.parent.glob(f"{glob_name_update}{p.suffix}"))
    l.info(f"Renaming {p} {len(files)} found files...")
    stems = []
    stem_suffixes = []
    for f in files:
        m = re.match(r"(?P<date>\d{6}_\d{4})(?P<st>?:st[\d-_]+[\d])?(?P<sfx>.*)", f.stem)
        if m:
            stems.append(m.group("date"))
            stem_suffixes.append(m.group("sfx"))
        else:
            stems.append("990101_0000")
            stem_suffixes.append("")
            l.warning(f"file name is not date: {f.name}")

    try:  # Excluding technical file from renaming
        i = stems.index("000000_0000")
    except ValueError:  # '000000_0000' is not in list
        pass
    else:
        del stems[i]
        del files[i]
    try:
        file_dates = pd.to_datetime(stems, format="%y%m%d_%H%M")
    except ValueError as e:  #  unconverted data remains when parsing with format "%y%m%d_%H%M": "3", at position 12.
        file_dates = []
        for stem in stems:
            try:
                file_dates.append(pd.to_datetime(stem, format="%y%m%d_%H%M"))
            except ValueError:
                l.error(f"Can not convert file stem {stem} to date")
        raise e
    # closest Station date indexes to each `file_dates`:
    i2file = datetime_fun(
        inearestsorted,
        df_lb.index.tz_localize(None),
        file_dates,
        type_of_operation="<M8[ms]",
        type_of_result="i8",
    )
    file_names_new = pd.Series(stems, index=df_lb.index[i2file])
    file_names_new = file_names_new.str.cat(df_lb.iloc[i2file, df_lb.columns.get_indexer(["St"])], sep="st")
    b = check_time_diff(
        file_names_new.index.tz_localize(None),
        file_dates.values,
        dt_warn=pd.Timedelta(minutes=10),  # pd.Timedelta(minutes=2)
        msg="Time difference [{units}] in {n} points exceeds {dt_warn} after :",
        max_msg_rows=20,
    )

    for i1, (path_in, stem_out, stem_sfx) in enumerate(zip(files, file_names_new.values, stem_suffixes), start=1):
        i_add = 1
        stem_out0 = stem_out = f"{stem_out}{stem_sfx}"
        try:
            while True:
                try:
                    path_in.rename((path_in.parent / stem_out).with_suffix(path_in.suffix))
                    break
                except FileExistsError:
                    stem_out = f"{stem_out0}~{i_add}"
                    i_add += 1
                    continue
        except FileNotFoundError:
            continue
        print(
            i1,
            path_in.stem,
            f"-> {stem_out}{path_in.suffix}",
        )
# %%

# Path to device data DB and device table inside
# path_cruise = Path(r"D:\Cruises\BalticSea\251201_ABP64")
path_db = (path_cruise / path_cruise.name.split("@", 1)[0]).with_suffix(".h5")
device = "CTD_SST_48Mc#1253"
