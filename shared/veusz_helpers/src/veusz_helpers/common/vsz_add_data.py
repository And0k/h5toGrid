"""
Globals used:
- parent: Path of executed file
- basename: vsz file base name
parent.name dir string (or basename with bigger priority) is used to get interval [s] of data to load.
Before digits and after units may be any characters that will not be used here (see re_dt)
Date units: years 'Y', months 'M', weeks 'W', and days 'D',
time units: hours 'h', minutes 'm', seconds 's'
"""

import logging
# from itertools import groupby, dropwhile
# import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Tuple, Sequence, Union
# from time import strptime
# from datetime import datetime
import numpy as np
import re
import h5py

import metadata
import func_vsz as fv

l = logging.getLogger(__name__)

NaT = np.datetime64("NaT")

class DumbMapping(dict):
    def __getitem__(self, key):
        return self.get(key, key)


def bool2ranges(b_ok, min_range, min_range_bad=None):
    """
    Get changing edges ignoring short intervals between
    :param b_ok:
    :param min_range:     min number of elements in intervals where b_ok is True
    :param min_range_bad: min number of elements in intervals where b_ok is False
    :return:
    """
    d_ok = np.diff(b_ok, prepend=False, append=False)
    edges = np.flatnonzero(d_ok != 0)
    n_rows = np.diff(edges)
    b_del_bad_interval = n_rows[1::2] < (min_range_bad or min_range)
    if b_del_bad_interval.any():  # delete starts and ends of too short no data intervals
        edges = edges[np.hstack((True, ~np.repeat(b_del_bad_interval, 2), True))]
        n_rows = np.diff(edges)
    b_del_good_interval = n_rows[::2] < min_range
    if b_del_good_interval.any():  # delete starts and ends of too short data intervals
        edges = edges[~np.repeat(b_del_good_interval, 2)]
    return edges


def search_time_range_indexes(index, time_range, raw_time_shift_s, raw_time_units="ns"):
    """
    :param index:
    :param time_range: [s] or convertible
    :param time_shift_s:
    :param raw_time_units: , defaults to "ns"
    :return: list
    """
    if any(np.isfinite(time_range)):
        have_timedelta = isinstance(time_range[0], np.timedelta64)
        if have_timedelta:
            raw_search_add = index[-1]
        else:
            if raw_time_shift_s:
                to_raw_time_units = 1 / np.timedelta64(1, raw_time_units).astype("m8[s]").astype(int).item()
                # ns?
                raw_search_add = -to_raw_time_units * raw_time_shift_s
            else:
                raw_search_add = 0
        if np.isfinite(time_range[-1]):
            time_range_raw_search = np.array(time_range, f"M8[{raw_time_units}]").astype(np.int64)
            i_range = np.fmin(
                np.searchsorted(index, time_range_raw_search + raw_search_add),
                len(index) - 1,
            ).tolist()
        else:
            time_range_raw_search = np.array(time_range[0]).astype(np.int64)
            i_range = [np.searchsorted(index, time_range_raw_search + raw_search_add).item(), len(index) - 1]
    else:
        i_range = [0, -1]
    return i_range


def pcid_from_parts(type: str = "i", model: str = None, number: str | int = None, b_raw=False, **kwargs):
    """
    Get Probe Column ID name (PCID)
    :param type: probe type ('i' for inclinometers)
    :param b_raw: return PCID for raw data tables
    :return: pcid string:
    if type=model="i", then set model="", then place "_" between type and model only if both not in ("i", "")
    """
    if type == "i":
        if model:
            if model == "i":
                model = ""  # not repeat "i"
                _ = ""
            else:
                _ = "_"
        else:
            _ = ""
    elif type == "" and model == "i":
        type, model = model, type
        _ = ""
    else:
        _ = "_"
    if b_raw and type == "i":
        type = "incl"
    return f"{type}{_}{model}{number:0>2}"


def _config_text_header_dtype(text_type, file_path=None) -> dict[str, Any]:
    try:
        from tcm import csv_load
        return csv_load.config_text_header_dtype(text_type, file_path)
    except ImportError as e:
        l.warning(f"{e} -> Using freezed version of TCM column definition. May be not up to date")

    if text_type is None:
        return {}
    known = ("i", "p", "b", "d", "w")
    if text_type.lower() not in known:
        raise TypeError(f"Probe model {text_type} not recognized! Known: {known}")
    b_default_type = text_type in ("i", "", "b")
    cfg_for_type = {
        "header": "yyyy(text),mm(text),dd(text),HH(text),MM(text),SS(text),Ax,Ay,Az,Mx,My,Mz"
        + (",Battery,Temp" if b_default_type else ",P_counts,Temp,Battery"),
        "dtype": "|S4 |S2 |S2 |S2 |S2 |S2 i2 i2 i2 i2 i2 i2 f8 f8".split()
        + ([] if b_default_type else ["f8"]),
    }
    return cfg_for_type


def load_tcm_config(path_in: Path) -> dict:
    if path_in.suffix != ".yaml":
        path_in = path_in.with_suffix(".yaml")
        # raise ValueError(f"The configuration must be *.yaml file, got {path_in}")

    with path_in.open(encoding="utf8") as f:

        from yaml import safe_load

        content = safe_load(f.read())
        return content

        {
                device_id: [
                    # Nested dict structure with station_id keys -> convert to single list compatible with
                    # single interval device
                    # todo: use dicts as from _meta_array_to_dict() instead of arrays
                    # instantly, to not loss data as in code below:
                    seq[0]
                    if all(s == seq[0] for s in seq)
                    else min(seq)
                    if col == "time_st"
                    else max(seq)
                    if col == "time_en"
                    else np.mean(seq)
                    if col in ["lat", "lon"]
                    else ",".join(str(s) for s in seq)
                    for col, seq in zip(
                        ["p", "b", "bd", "s", "lat", "lon", "time_st", "time_en", "burst_dt", "bursts_t"],
                        zip(*meta.values()),
                    )
                ]
                if isinstance(meta, dict)
                # Single interval device (list/tuple) - convert to nested dict with key "0"
                else meta
                for device_id, meta in content.items()
            }


def _cf_to_dt_ns(raw: np.ndarray, units: Union[str, bytes] = "seconds since 1970-01-01", int=False) -> np.ndarray:
    """float64 seconds OR legacy int64 ns → datetime64[ns] (for h5py read).

    Backward-compatible: inspects dtype to handle both CF-float64 and
    legacy-int64 formats.  For float64, parses the epoch date from *units*
    (e.g. ``"seconds since 1970-01-01"``).  When *units* is empty the
    legacy-1970 path is used.  Accepts ``bytes`` (from h5py attrs).
    """
    if isinstance(units, bytes):
        units = units.decode()
    if raw.dtype.kind == "f":  # Float path — seconds since 1970 (NetCDF format)
        epoch_ns = np.datetime64(units.removeprefix("seconds since "), "ns").astype(np.int64) if units else 0
        out_int = (((raw[:] if any(raw.shape) else raw) * 1e9).astype(np.int64) + epoch_ns)
    else:  # Legacy int64 path — nanoseconds since 1970 (h5py format)
        out_int = raw.astype(np.int64)
    return out_int if int else out_int.astype("datetime64[ns]")

def veusz_load_hdf5(
    file,
    device_ids: Sequence[str],
    grp_d: Optional[Mapping[str, str]] = None,
    cols_namemap: Optional[Mapping[str, Union[Mapping[str, str], Iterable[str]]]] = None,
    grp_d_rename_funs: Mapping[str, Callable[[str, str], str]] = None,
    time_range=tuple(),
    time_shift_s: int = 0,
    b_load_to_veusz=True,
    decimation: Optional[int] = None,
    prefix: Optional[str] = "",
    suffix: Optional[str] = "",
) -> Tuple[Mapping[str, Any], Tuple[np.timedelta64, np.timedelta64], Mapping[str, list | np.ndarray]]:
    """
    Finds integer index slice from specified time range and loads hdf5 data to Veusz.
    Intended to be executed in caller's vsz workspace

    :param file: db path, i.e. r'../200819incl.h5'
    :param device_ids: [id1, id2, ... id#, ...] top level groups in hdf5
    :param grp_d: {grp: val} dict to form hdf5 subgroups for each id#: "/{id#}{grp_d.value}". If None, then
    one default group will be loaded: {'table': '/table/'} useful for simple pandas tables
    :param cols_namemap: maps hdf5 vars to output cols (or list of vars if not need rename) for each of grp
    :param grp_d_rename_funs: dict of functions(col, device_id) for each of grp_d group to further rename
    output cols (useful to add prefixes / suffixes)
    :param time_range: either:
    - any 2-element sequence convertible to datetime64 array. For example ['2020-08-19T21:59:22', '2020-08-20T06:05:04']
    - timedelta instance: then it is dt_to_last: time interval to last db data to find t_start (using index[-1] from 1st table)
    :param time_shift_s: shift in seconds You'll add to loaded time, here used to set input range correctly
    :param b_load_to_veusz:
    :param decimation:
    :param prefix: "",
    :param suffix: ""
    :return: (existed_devs, timerange, i_ranges):
    - existed_devs: dict with fields [device_id][grp]
    - timerange: time range [start, end] with account to time_shift_s that you'll add to loaded time in Veusz
    - i_ranges:
    Usage: insert next rows into top of vsz (and delete old ImportFileHDF5...):
    # Next 8 rows will be replaced by the hard to edit code if You save the file in Veusz's GUI!
    ### Loading part of hdf5 data using time range ###
    # (method can also be used to load any vsz inside this)
    >> from pathlib import Path; from sys import argv
    >> time_range_selector_file = Path(argv[1]).parent / 'time_range_selector.vsz'
    >> exec(compile(time_range_selector_file.read_text(), time_range_selector_file, 'exec'))
    >> veusz_load_hdf5(r'../200819_AI56.h5', ['CTD_Idronaut_OS316#494'], ['2020-08-19T21:59:22', '2020-08-20T05:04:00'])
    # ('201202.raw.h5', ['/incl04], ['/coef', '/table']
    ImportFileHDF5('201202.raw.h5', ['/incl04/coef', '/incl04/table'], linked=True, namemap={'/incl04/coef/G/A': 'Ag0', '/incl04/coef/G/C': 'Cg', '/incl04/coef/H/A': 'Ah0', '/incl04/coef/H/C': 'Ch', '/incl04/coef/H/azimuth_shift_deg': 'azimuth_shift_deg', '/incl04/table/Ax': 'countsAx', '/incl04/table/Ay': 'countsAy', '/incl04/table/Az': 'countsAz', '/incl04/table/Mx': 'countsMx', '/incl04/table/My': 'countsMy', '/incl04/table/Mz': 'countsMz', '/incl04/table/Temp': 'sT'}, renames={'Vabs0': 'kVabs'})

    """
    # Determine whether `time_range` should be timedelta or time range and convert it to corresponding array
    b_ok = np.isfinite(time_range)
    if any(b_ok):
        print("Loading interval", np.datetime_as_string(time_range).tolist(), "from", file)
        have_timedelta = isinstance(time_range[0], np.timedelta64)
        time_range = np.array(time_range, f"{'m' if have_timedelta else 'M'}8[ns]")
        if not have_timedelta:
            # allow find maximum if NaT using np.searchsorted() correctly (because NaT is negative number)
            if np.isnat(time_range[-1]):
                time_range[-1] = np.datetime64(np.iinfo(np.int64).max, "ns")
    else:
        time_range = []
    time_raw_min = np.iinfo(np.int64).max  # not NaT because all our *_raw* data are int64
    time_raw_max = 0
    if file.suffix == ".h5":
        if grp_d is None:
            grp_d = {"table": "/table/"}
        time_var_name = "index"
    else:
        if grp_d is None:
            grp_d = {"table": "/"}
        time_var_name = "time"

    # set default mappings where skipped
    if cols_namemap is None:
        cols_namemap = {grp: DumbMapping() for grp in grp_d}
    else:
        cols_namemap = {
            grp: {s: s for s in map_or_seq} if not isinstance(map_or_seq, Mapping) else map_or_seq
            for grp, map_or_seq in cols_namemap.items()
        }
    if grp_d_rename_funs is None:
        grp_d_rename_funs = {grp: lambda x, device_id: x for grp in grp_d}

    with h5py.File(file, "r") as h:
        # Existed parameters in DB
        existed_devs = {}
        grp_dev = {}
        i_ranges = {}
        for i, device_id in enumerate(device_ids, start=1):
            existed_devs[device_id] = {}
            grp_dev[device_id] = {}
            print(f"{i}. " if len(device_ids) > 1 else "", device_id, end=": ")
            try:
                for grp, t in grp_d.items():
                    cur_gr = f"/{device_id}{t}"
                    print(t, end=" ")
                    # updating grp_d
                    grp_dev[device_id][grp] = cur_gr

                    try:
                        if isinstance(h[cur_gr], h5py.Group):
                            existed_devs[device_id][grp] = set()
                            for k, val in h[cur_gr].items():
                                if isinstance(val, h5py.Group):
                                    for k1, val1 in val.items():
                                        existed_devs[device_id][grp].add(f"{k}/{k1}")
                                else:
                                    existed_devs[device_id][grp].add(f"{k}")
                            existed_devs[device_id][grp] &= set(cols_namemap[grp])
                        else:
                            existed_devs[device_id][grp] = set(h[cur_gr].dtype.fields.keys())

                            no_cols = set(cols_namemap[grp]).difference(existed_devs[device_id][grp])
                            if no_cols:
                                print(f"No columns: {no_cols}, existed_devs: {existed_devs[device_id][grp]}")
                                # for k in no_cols:
                                #     # del existed_devs[device_id][grp][k]
                                #     del cols_namemap[grp][k]
                    except KeyError:
                        existed_devs[device_id][grp] = None
                        continue

                print(end=", " if i < len(device_ids) else ". ")
                index = _cf_to_dt_ns(h[grp_dev[device_id]["table"]][time_var_name], int=True)
                # todo: check this to replace code below:
                # i_ranges[device_id] = search_time_range_indexes(index, time_range, time_shift_s)
                if any(np.isfinite(time_range)):
                    time_range_raw_cur = np.int64(time_range) + (
                        index[-1] if have_timedelta else -1e9 * time_shift_s
                    )

                    i_ranges[device_id] = np.searchsorted(index, time_range_raw_cur).tolist()
                    try:
                        time_range_raw = index[i_ranges[device_id]]
                    except (IndexError, TypeError):  # `TypeError: Indexing elements must be in increasing
                    # order` in addition and before IndexError if both indexes not increasing and above limit
                        _ = min(i_ranges[device_id][-1], len(index) - 1)
                        if _ <= i_ranges[device_id][0]:
                            raise IndexError(
                                "Required time range {} is after the data range {}".format(
                                    time_range, _cf_to_dt_ns(index[[0, -1]])
                                )
                            )
                        try:
                            time_range_raw = index[[i_ranges[device_id][0], _]]
                        except IndexError:
                            raise IndexError(
                                "Required time range: {}, data range: {}".format(
                                    time_range, _cf_to_dt_ns(index[[0, -1]])
                                )
                            )
                else:
                    time_range_raw = index[[0, -1]]
                    i_ranges[device_id] = [0, len(index)]

                time_raw_min = np.fmin(time_range_raw[0], time_raw_min)
                time_raw_max = np.fmax(time_range_raw[-1], time_raw_max)
            except Exception:
                l.exception(f'Working with HDF5 file "{file}", {grp_dev[device_id]}')
                raise
    time_range_raw = _cf_to_dt_ns(np.array([time_raw_min, time_raw_max]))
    # if time_range was specified this will be for last device_id:
    time_range_out = time_range_raw.astype("M8[s]") + time_shift_s if any(time_range_raw) else []
    if any(time_range_out):
        print(
            "Max of",
            "selected" if any(np.isfinite(time_range)) else "DB",
            f"time ranges for all used devices: {time_range_out}.",  # ?
            "Corresponding data indexes found:",
            i_ranges,
        )
    else:
        print("No time ranges found")

    # Load to Veusz

    if b_load_to_veusz:

        def cols_spec_or_existed(device_id, grp):
            if not cols_namemap[grp]:
                return existed_devs[device_id][grp]
            else:
                return set(existed_devs[device_id][grp]).intersection(cols_namemap[grp])

        slices = {
            f"{grp_dev[device_id][grp]}{col}": (i_range + [decimation],)
            for col in cols_spec_or_existed(device_id, grp)
            for device_id, i_range in i_ranges.items()
        }  # print('slices:', slices)
        namemap = {
            f"{t}{name}": grp_d_rename_funs[grp](cols_namemap[grp][name], device_id)
            for device_id in device_ids
            for grp, t in grp_dev[device_id].items()
            for name in cols_spec_or_existed(device_id, grp)
        }
        ImportFileHDF5(  # noqa: F821
            file,
            (list(namemap) or [_ for device_id in device_ids for _ in grp_dev[device_id].values()]),
            linked=True,
            namemap=namemap,
            slices=slices,
            **{k: v for k, v in [("prefix", prefix), ("suffix", suffix)] if v},
        )
        # Add tag "loaded"
        TagDatasets(
            "loaded",
            sorted([
                grp_d_rename_funs["table"](cols_namemap["table"][col], device_id)
                for device_id in device_ids
                for col in cols_spec_or_existed(device_id, "table")
            ]),
        )
    return existed_devs, time_range_out, i_ranges


def veusz_load_hdf5_tcm_raw(
    file, devices, time_range, time_shift_s: int = 0, dev_dir=None, decimation: Optional[int] = None
):
    """
    :param devices: dict with fields equal to pids. Each value is a dict of parts of probe name, here used
    to construct pandas data table name '{type}{_}{pid}'
    :param file: data file
    :param time_range: any 2-element sequence convertible to datetime64 array in displaying zone. For example ['2020-08-19T21:59:22', '2020-08-20T06:05:04']
    :param time_shift_s: shift in seconds You'll add to loaded time to display data, here used to set input range correctly
    :param dev_dir: use filtered db from this dir where coef. already applied
    :return:
    """
    # take 1st probe
    _ = iter(devices.items())
    pid, probe = next(_)
    _ = [(pid, probe)] + list(_)
    p_type = probe["type"]
    device_ids = [
        f"incl{pid[1:]}"
        if pid[0] == "i"
        else "_".join(
            (["incl"] if p_type == "i" else [p_type] if p_type and p_type[0] != pid[0] else []) + [pid]
        )
        for pid, probe in _
    ]

    grp_d = {
        "coef": "/coef/",
        "table": ("/table/" if file.suffix == ".h5" else "/"),
    }
    table_cols_to_slice_namemap_common = {  # common parameters which time slice is need to load
        **({"index": "t_ns"} if file.suffix == ".h5" else {"time": "t_s"}),
        "P_counts": "_P_counts",
        "P": "_P_counts",
        "Temp": "Temp",
        "Battery": "Battery",
    }
    if p_type == "w":
        cols_namemap = {
            "coef": {
                p: (f"coef_{p}__" if p[0].islower() else f"coef{p}__")
                for p in (
                    "P",
                    "PTemp",
                    "Battery_ok_min",
                    "PBattery",
                    "PBattery_min",
                    "pid",
                )
            },
            "table": table_cols_to_slice_namemap_common,
        }

        if dev_dir:
            file = dev_dir / f"{dev_dir.stem}.proc_noAvg.h5"
            if not Path(file).is_file():  # try & except not works here (OSError)
                file = dev_dir.stem / f"{dev_dir.stem}@w.proc_noAvg.h5"

            del grp_d["coef"]
            SetDataExpression("kP", "[1, 0]", linked=True)  # 1:1
            # SetDataExpression('i', '1', linked=True)

    else:
        cols_namemap = {
            "coef": {
                "G/A": "Ag0",
                "G/C": "Cg",
                "H/A": "Ah0",
                "H/C": "Ch",
                "H/azimuth_shift_deg": "azimuth_shift_deg",
                "Rz": "Rz",
                "Vabs0": "coefs_Vabs",
                "i": "pid_",
            },
            "table": {
                **{k: f"{k}_counts" for k in "Ax Ay Az Mx My Mz".split()},
                **table_cols_to_slice_namemap_common,
            },
        }
        if "p" in probe["model"]:
            cols_namemap["coef"]["P_t"] = "P_t"
            cols_namemap["table"]["TempP"] = "TempP"

    # prefixes and suffixes for columns of each grp_d group
    def f_table_cols_fmt(col, device_id):
        return col if col.endswith("counts") else f"_{col}__"

    grp_d_rename_funs = {"coef": "{}".format, "table": f_table_cols_fmt}

    existed_devs, time_range, i_ranges = veusz_load_hdf5(
        file,
        device_ids,
        grp_d,
        cols_namemap,
        grp_d_rename_funs,
        time_range,
        time_shift_s=time_shift_s,
        decimation=decimation,
    )

    # Dummy coef if not existed in DB:
    for device_id, grp_d in existed_devs.items():
        if "coef" not in grp_d.keys():
            print(f'No "coef" in DB table {device_id} => assinging dummy in Custom Definitions.')
            AddCustom(
                "definition",
                "Ag0",
                "[[0.00173, 0, 0], [0, 0.00173, 0], [0, 0, 0.00173]]  # coef. before rotate with Rz",
            )
            AddCustom("definition", "Cg", "[[10], [10], [10]]")

            AddCustom(
                "definition",
                "Ah0",
                "[[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # coef. before rotate with Rz",
            )
            AddCustom("definition", "Ch", "[[10],[10],[10]]")
            AddCustom("definition", "Rz", "[[1, 0, 0], [0, 1, 0], [0, 0, 1]]")
            AddCustom("definition", "azimuth_shift_deg", "180")
            AddCustom("definition", "coefs_Vabs", "[10, -10, -10, -3, 3, 70]")
            AddCustom("definition", "pid", "0")
            # del grp_d['coef']
            # del cols_namemap['coef']
            # del grp_d_rename_funs[0]

        if "coef" in grp_d:
            try:
                TagDatasets("coefficient", [cols_namemap["coef"][k] for k in grp_d["coef"]])
            except KeyError as e:  # not all possible coef must exists
                print("KeyError:", e, "Skipping TagDatasets and continue...")

    # Add time which should be used by all functions used for ~drawer@i.vsz as veusz_load_csv_tcm_raw give it
    if file.suffix != ".h5":  # .nc
        SetDataExpression("_t_ns__", "1E9*_t_s__", linked=True)
    SetDataExpression("time__", "v.dt64s2vsz(1E-9*_t_ns__[v.sl(iu__)]) + USE_timeShift_s", linked=True)

    return (
        existed_devs,
        time_range,
        file,
        cols_namemap,
        grp_d_rename_funs,
        f_table_cols_fmt,
    )  #


def veusz_load_csv_tcm_raw(
    file,
    db,
    time_range,
    time_shift_s=0,
    probe_info: Optional[dict] = None,
    fun_get_time_ranges: Optional[Callable[[Tuple[np.ndarray]], np.ndarray]] = None,
    # ImportFileCSV kwargs:
    rowsignore=3,  # same as default `skiprows` in tcm.csv_load
    encoding="ascii",
    headermode="none",
    skipwhitespace=True,
    **kwargs,  # other
):
    """
    Loads TCM data from text *.csv file (example below is for p-model, but may be other models):
    Inkl P01
    Y,M,D,H,M,S,Ax,Ay,Az,Mx,My,Mz,ADC,T,Bat
    2023,4,25,12,24,11,160,-2576,-13808,-111,62,547,1849841,21.63,5.42
    :param db: path to hdf5 data with coefs
    :param time_range: Not implemented: any 2-element sequence convertible to datetime64 array in displaying zone. For example ['2020-08-19T21:59:22', '2020-08-20T06:05:04']
    :param file: text file to load data
    :param time_shift_s: Not implemented: shift in seconds You'll add to loaded time to display data, here used to set input range correctly

    :return: (time_range_raw, time, a1d, icol):
    - time_range_raw: time range of raw data
    """

    if any(np.isfinite(time_range)) or fun_get_time_ranges:
        # Need limiting of data loading to Veusz
        if fun_get_time_ranges:
            raise NotImplementedError("Limiting of data loading to Veusz")  # todo
            # a1d = np.genfromtxt(file, usecols=list(range(n_start_cols)) + i_cols, **kwargs)
            # time = np.datetime64("%02.0f-%02.0f-%02.0f" % tuple(a1d[0, icols_date_reorder]), "s") + np.array(
            #     (np.append(0, np.cumsum(np.diff(a1d[:, idd]) != 0) * 24) + a1d[:, iHH]) * 3600
            #     + a1d[:, iMM] * 60
            #     + a1d[:, iSS],
            #     "m8[s]",
            # )
            # if any(np.isfinite(time_range)):
            #     iu = np.searchsorted(time, time_range).tolist()
            #     time = time[slice(*iu)]
            #     a1d = a1d[slice(*iu), n_start_cols:]
            # else:
            #     a1d = a1d[:, n_start_cols:]

            # b_ok = fun_get_time_ranges(
            #     *a1d[:, [icol[col] for col in fun_get_time_ranges.__code__.co_varnames]].T
            # )
            # edges = bool2ranges(b_ok, min_range, min_range_del)
            # try:
            #     time_ranges = time[edges]
            # except IndexError:
            #     time_ranges = np.append(time[edges[edges < time.size]], time[-1] + np.timedelta64(1, "s"))

            # print(
            #     f"Time ranges of data found:",
            #     ", ".join([f"{st} - {en}" for st, en in zip(time_ranges[::2], time_ranges[1::2])]),
            # )
            # time_range_raw = time_ranges[[0, -1]]
        else:
            time_ranges = []
    else:
        time_ranges = []

    # Determine probe type and get configuration

    if probe_info:
        probe_type = probe_info.get("type", "i")
        probe_model = probe_info.get("model", probe_type)
        # raw data historical naming
        probe_id = pcid_from_parts(**probe_info, b_raw=True)
    else:
        # Default probe type if not specified
        probe_model = "i"
        probe_id = "i00"

    ## Get text header and dtype configuration based on probe type
    config = _config_text_header_dtype(probe_model[0] if probe_model else probe_type, file)

    ## Determine column mappings based on probe type
    cols = config.get("header", "").split(",")

    # Create renames mapping based on the header configuration

    ## Map date/time columns (keep one letter and add prefix to get ['_y', '_m', '_d', '_H', '_M', '_S'])
    len_date_cols = 6
    renames = {
        f"col{i}": f"_{date_col[0]}".removesuffix("(text)")
        for i, date_col in enumerate(cols[:len_date_cols], start=1)
    }

    ## Map data columns
    data_col_mapping = {
        "Ax": "Ax_counts",
        "Ay": "Ay_counts",
        "Az": "Az_counts",
        "Mx": "Mx_counts",
        "My": "My_counts",
        "Mz": "Mz_counts",
        # 'Battery': 'Battery',
        "Temp": "_Temp__",
        # 'P_counts': 'P_counts'
    }

    ## Process remaining columns based on header
    renames.update({
        f"col{i}": data_col_mapping.get(field, field)
        for i, field in enumerate(cols[len_date_cols:], start=len_date_cols + 1)
    })

    # Get data to Veusz allowing to save vsz without copying data
    ImportFileCSV(
        file,  # f"@i_{probe_n:>03}.TXT"
        renames=renames,
        rowsignore=rowsignore,
        encoding=encoding,
        headermode=headermode,
        skipwhitespace=skipwhitespace,
        linked=True,
        **kwargs,
    )

    # Convertion variables to that used with hdf5 drawers
    SetDataExpression(
        "time__",
        "(lambda x: v.rep2mean(x, ediff1d(x, to_end=0)!=0))(fdate([_y, _m, _d]) + 3600*_H + 60*_M + _S) + "
        "USE_timeShift_s",
        linked=True,
    )
    SetDataExpression("_t_ns__", "(time__ + 1230768000)*1E9", linked=True)

    if db and Path(db).is_file():
        ImportFileHDF5(
            db,
            [f"/{probe_id}/coef"],
            linked=True,
            namemap={
                f"/{probe_id}/coef/G/A": "Ag0",
                f"/{probe_id}/coef/G/C": "Cg",
                f"/{probe_id}/coef/H/A": "Ah0",
                f"/{probe_id}/coef/H/C": "Ch",
                f"/{probe_id}/coef/H/azimuth_shift_deg": "azimuth_shift_deg",
                f"/{probe_id}/coef/Vabs0": "coefs_Vabs0",
            },
        )
    else:
        # Load coefficients from hydra yaml configuration
        dir_raw = metadata.get_path_in_parents(file, "_raw", target_is_dir=True)
        dir_cfgs = dir_raw / "cfg_proc" / "defaults"
        cfgs_existed, cfgs = metadata.select_cfgs(
            dir_cfgs=dir_cfgs, incl_type_nums={pcid_from_parts(**probe_info, b_raw=False).lower()}
        )
        if cfgs:
            print(f"=== Using {len(cfgs)}/{len(cfgs_existed)} user configurations in {dir_cfgs}")
            content = load_tcm_config(dir_cfgs / next(iter(cfgs.values())))
            renames = {"Ag": "Ag0", "Ah": "Ah0", "kVabs": "coefs_Vabs(numeric)"}
            if not "azimuth_shift_deg" in content["input"]["coefs"]:
                content["input"]["coefs"]["azimuth_shift_deg(numeric)"] = [180]
            for k, v in content["input"]["coefs"].items():
                try:
                    k = renames[k]
                except KeyError:
                    pass
                if isinstance(v, list):
                    if isinstance(v[0], list):
                        ImportString2D(
                            k,
                            "xrange 0.0 3.0\nyrange 0.0 3.0\n" +
                            "\n".join(" ".join(str(item) for item in v0) for v0 in v)
                        )
                    else:
                        ImportString(f'{k}(numeric)', "\n".join(str(item) for item in v))


    return (time_ranges,)


def veusz_load_hdf5_ctd_profile(
    file,
    time_range,
    device=None,
    time_shift_s: int = 0,
    n_runs=1,
    params_d=None,
    renames_no_prefix={
        "table_log": {"index": "DateSt", "rows": "downcast_len"},
        "table": {"O2": "O2sat", "Turb": "Turb_nof", "Fluor": "ChlA_nof"},  # 'Pres': 'Pres_NoSep',
    },
    prefix_d={
        "table_log": "_log_",
        "table": "_",
    },
    fun_custom=None,
) -> None:
    """
    Find integer index slice from specified time range and loads data needed to draw "Zabor".
    Intended to be executed in caller's vsz workspace
    :param file: hdf5 data file with data table and log table
    :param time_range: any 2-element sequence convertible to datetime64 array in displaying zone. If 2nd
    element is not finite, then searches row >= 1st time in table_log and get time_en at n_runs-th row after
    found row. After we have 2 finite time values, the program searches indexes of rows in data table.
    :param time_shift_s: shift in seconds You'll add to loaded time to display data, here used to set input range correctly
    :param params_d: columns to load from each table. For example {
        "table_log": ["index", "DateEnd", "rows"]
        + [f"{p}_{sfx}" for p in "Lat Lon DepEcho".split() for sfx in ["st", "en"]],
        "table": "index Pres Temp Cond Sal O2 O2ppm Turb Fluor".split(),
    }
    :param renames_no_prefix:
    :param prefix_d:
    :param fun_custom: in not None, then return result of executing this function instead exec Veusz commands.
    Function receives:
    - h: opened file handle,
    - i_ranges = {
        "table_log": [log_i_st, log_i_st + n_runs],
        "table": [i_time_st, i_time_st + len_last_run]
    },
    - time_range_raw

    Usage: insert next rows into top of vsz (and delete old ImportFileHDF5...):
    # Next 8 rows will be replaced by the hard to edit code if You save the file in Veusz's GUI!
    ### Loading part of hdf5 data using time range ###
    # (method can also be used to load any vsz inside this)
    >> from pathlib import Path; from sys import argv
    >> time_range_selector_file = Path(argv[1]).parent / 'time_range_selector.vsz'
    >> exec(compile(time_range_selector_file.read_text(), time_range_selector_file, 'exec'))
    >> import_hdf5(['2020-08-19T21:59:22', '2020-08-20T05:04:00'], r'../200819_AI56.h5', 'CTD_Idronaut_OS316#494')
    """
    # or '{type}_{model}#{id}'.format_map(probe).replace(' ', '_')
    grp_d = {"table": f"/{device}/table/", "table_log": f"/{device}/logRuns/table/"}
    if params_d is None:
        params_d = {tbl: [] for tbl in grp_d}

    with h5py.File(file, "r") as h:
        tbl_cols_exist = {tbl: h[grp_d[tbl]].dtype.fields.keys() for tbl in params_d}
        log_index = h[grp_d["table_log"]]["index"]
        if not np.isfinite(time_range[-1]):  # loading n runs
            time_range_raw0 = np.int64(np.array(time_range[0], "M8[ns]") - np.timedelta64(time_shift_s, "s"))
            # Better time accuracy is in log file:
            log_i_st = np.searchsorted(log_index, time_range_raw0)

            if log_i_st < log_index.size:
                log_i_en = log_i_st + n_runs - 1
                if log_i_en >= log_index.size:
                    log_i_en = log_index.size - 1
                    print("len(table_log) exceeded => get slice to last element")
            else:
                print("len(table_log) exceeded => get last element")
                log_i_en = log_i_st = log_index.size - 1
            i_ranges = {"table_log": [log_i_st, log_i_en + 1]}
            time_range_raw = log_index[[log_i_st, log_i_en]]
            try:
                # last time may be too far, so not used (to do: remove it):

                # print('Start to next start (raw time):', np.array(time_range_raw, 'M8[ns]'))
                # i_ranges['table'] = np.searchsorted(h[grp_d['table']]['index'], time_range_raw).tolist()

                # Start row of 1st run (and last if n_runs > 1):
                i_time_st = np.searchsorted(
                    h[grp_d["table"]]["index"], time_range_raw[0] if log_i_st == log_i_en else time_range_raw
                )
                # length of last run
                len_last_run = (
                    h[grp_d["table_log"]]["rows"][log_i_en] + h[grp_d["table_log"]]["rows_filtered"][log_i_en]
                )
                # add it to start row of last run (to next start or data end):
                i_ranges["table"] = (i_time_st + [0, len_last_run]).tolist()

            except IndexError:
                print("len(table) exceeded => set slice to last element")
                time_range_raw = np.int64([log_index[log_i_st], h[grp_d["table"]]["index"][-1]])
                i_ranges["table"] = [
                    np.searchsorted(h[grp_d["table"]]["index"], time_range_raw[0]),
                    None,
                ]
            time_range = time_range_raw + np.timedelta64(time_shift_s, "s")  # output (displaying) time zone
        else:  # loading specified interval
            time_range_raw = np.int64(np.array(time_range, "M8[ns]") - np.timedelta64(time_shift_s, "s"))
            # if not limit:
            i_ranges = {
                grp: np.searchsorted(h[tbl_path]["index"], time_range_raw).tolist()
                for grp, tbl_path in grp_d.items()
            }

        if fun_custom:
            return fun_custom(h, i_ranges, grp_d, time_range_raw)
        else:
            for tbl in grp_d:
                _ = params_d[tbl]
                params_d[tbl] = list(tbl_cols_exist[tbl] if not _ else set(_) & tbl_cols_exist[tbl])
            SetData("_log_Lats_st", h[grp_d["table_log"]]["Lat_st"])
            SetData("_log_Lons_st", h[grp_d["table_log"]]["Lon_st"])
            SetData(
                "_log_t64s_st",
                np.int64(np.array(log_index, "M8[ns]").astype("M8[s]")) + time_shift_s,
            )
    print(f"corresponding data indices in {file}:", i_ranges)
    if not fun_custom:
        ImportFileHDF5(
            str(file),
            [f"{grp_d[t]}{p}" for t, pp in params_d.items() for p in pp] + ["/map"],
            linked=True,
            namemap={f"{grp_d[t]}{p}": f"{prefix_d[t]}{p}" for t, pp in params_d.items() for p in pp},
            slices={f"{grp_d[t]}{p}": (i_ranges[t] + [None],) for t, pp in params_d.items() for p in pp},
            renames={
                f"{prefix_d[t]}{p}": f"{prefix_d[t]}{p_new}"
                for t, remap in renames_no_prefix.items()
                for p, p_new in remap.items()
            },
        )

        # Add tag "loaded"
        for tbl, remap in renames_no_prefix.items():
            cols_set = set(tbl_cols_exist[tbl])
            remap_checked = remap.copy()
            for p in remap.keys():
                if p in cols_set:
                    params_d[tbl].remove(p)
                else:
                    del remap_checked[p]
            params_d[tbl].extend(remap_checked.values())
        TagDatasets(
            "loaded",
            [f"{prefix_d[t]}{p}" for t, pp in params_d.items() for p in pp]
            + ["_log_Lats_st", "_log_Lons_st", "_log_t64s_st", "_log_downcast_len"],
        )
    return time_range






def veusz_load_hdf5_cmems(file, time_range, time_shift_s: int = 0, load_map=False, **kwargs):
    """
    :param file: like "../CMEMS/cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H_multi-vars_20.31E_54.94N_2023-08-20-2023-09-20.nc"
    :param time_range: any 2-element sequence convertible to datetime64 array in displaying zone. For example ['2020-08-19T21:59:22', '2020-08-20T06:05:04']
    :param time_shift_s: shift in seconds You'll add to loaded time to display data, here used to set input range correctly
    :return loaded time range in displaying zone, wind_mean_uv
    """

    # required_name_part = "_(P1D-m|PT1H)[-_]multi-vars"
    # m = re.match(
    #     fr"(?P<file_name_prefix>[^P]+){required_name_part}(?P<file_name_suffix>.*\.nc)", file.name
    # )
    # prefix = m.group("file_name_prefix")
    # file_name_suffix = m.group("file_name_suffix")
    # files = [
    #     f for f in file.parent.glob(
    #         re.sub(  # regex to glob pattern:
    #             r"\([^)]+\)",
    #             "*",
    #             f"{prefix}{required_name_part}{file_name_suffix}"
    #         )
    #     )
    # ]

    index_name = "time"
    # files_params = {
    #     f"{prefix}_obs-wind_glo_phy_nrt_l4": [
    #         "eastward_wind",
    #         "northward_wind",
    #         "air_density",
    #         "eastward_wind_sdd",
    #         "northward_wind_sdd"
    #         index_name,
    #     ],
    #     f"{prefix}_mod_bal_phy_anfc": [
    #         "bottomT", "so", "sob", "thetao", "uo", "vo", "wo",
    #     ]
    # }

    grp = ""  # root
    cols_namemap = {
        grp: {
            index_name: "timeUnix",
            "eastward_wind": "u_cm",
            "northward_wind": "v_cm",
            "air_density": "air_density",
            "eastward_wind_sdd": "u_cm_sdd",
            "northward_wind_sdd": "v_cm_sdd",
        },
    }
    suffix = "_Wind"
    scalers = {}
    with h5py.File(file, "r") as h:
        for col_in, col in cols_namemap[grp].items():
            # Get the add_offset and scale_factor values
            try:
                attributes = h[col_in].attrs
                scalers[col] = [attributes[k].item() for k in ["scale_factor", "add_offset"]]
            except KeyError:
                continue
        t_shift_to_unix = 599616000 - 1230768000
        index = h[index_name]
        if any(np.isfinite(time_range)):
            have_timedelta = isinstance(time_range[0], np.timedelta64)
            time_range_raw = np.array(time_range).astype(np.int64) + (
                index[-1] if have_timedelta else (t_shift_to_unix - time_shift_s)
            )
            i_range = np.searchsorted(index, time_range_raw).tolist()
        else:
            time_range_raw = index[[0, -1]]
            i_range = [0, len(index)]
            time_range = np.int64(time_range_raw - t_shift_to_unix + time_shift_s)

        if np.diff(i_range) <= 2:
            # For only 1 or 2 vectors we will not draw in separate graphs, but display message
            wind_mean_uv = complex(*[  # cdouble(real=0, imag=0) not works
                np.polyval(
                    scalers[cols_namemap[grp][col]],
                    np.mean(h[col][slice(*i_range), :, :]),
                )
                for col in ["eastward_wind", "northward_wind"]
            ])
        else:
            wind_mean_uv = None

    ImportFileHDF5(
        file,
        [grp],
        linked=True,
        namemap={"/".join([grp, col_in]): col for col_in, col in cols_namemap[grp].items()},
        slices={
            f"/{col}": (((i_range + [None]), 0, 0) if col != "time" else (i_range + [None],))
            for col in cols_namemap[grp]
        },
        suffix="_Wind",  # "_CM"
    )
    # Select time range and convert time to Veusz format
    SetData2DExpression(
        "iu_Wind",
        (
            "atleast_2d(v.i_positive(v.i_use(timeUnix_Wind, USEtime_Wind, 1230768000 - 599616000 + "
            "Wind_timeShift_s, t_units='s'), timeUnix_Wind.size))"
        ),
        linked=True,
    )
    SetDataExpression(
        "time_Wind",
        "timeUnix_Wind[sl_(iu_Wind if diff(iu_Wind) > 1 else iu_Wind + [0, 1])] "
        "- 599616000 + Wind_timeShift_s",
        linked=True,
    )
    # Convert NetCDF stored values to physical
    for col in scalers:
        col_scaled = col.replace("_cm", "")
        if col_scaled != col:
            SetDataExpression(
                f"{col_scaled}{suffix}",
                f"polyval({scalers[col]}, {col}{suffix})",
                linked=True,
            )

    return (time_range, wind_mean_uv)


def veusz_load_hdf5_ecmwf(file, time_range, time_shift_s: int = 0, load_map=False, **kwargs):
    """
    :param file: file with mask `data_stream-(oper|wave)_stepType-(instant|max|accum)(.*).nc`
    like "../ECMWF/data_stream-oper_stepType-instant_20.31E_54.94N_2023-08-20-2023-09-20.nc"
    (.*) part should be constant for each of required files
    :param time_range: any 2-element sequence convertible to datetime64 array in displaying zone. For example ['2020-08-19T21:59:22', '2020-08-20T06:05:04']
    :param time_shift_s: shift in seconds You'll add to loaded time to display data, here used to set input range correctly (usually you must set it equal to `cus.Wind_timeShift_s`)
    :return loaded time range in displaying zone, wind_mean_uv
    """
    if file.is_dir():
        file = next(file.glob("*-oper_stepType-instant.nc"))

    required_name_part = "-(oper|wave)_stepType-(instant|max|accum)"
    m = re.match(rf"(?P<file_name_prefix>[^-]+){required_name_part}(?P<file_name_suffix>.*\.nc)", file.name)

    try:
        prefix = m.group("file_name_prefix")
        file_name_suffix = m.group("file_name_suffix")
        files = [
            f
            for f in file.parent.glob(
                re.sub(  # regex to glob pattern:
                    r"\([^)]+\)", "*", f"{prefix}{required_name_part}{file_name_suffix}"
                )
            )
        ]
        index_name = "valid_time"
    except AttributeError:  # 'NoneType' object has no attribute 'group'
        l.warning("trying to load all from one file (old style input)")
        prefix = ""
        files = [file]
        file_name_prefix = file.name
        file_name_suffix = ""
        index_name = "time"

    files_params = {
        f"{prefix}-oper_stepType-instant": [
            "u10",
            "v10",
            "sp",
            "sst",
            index_name,
        ],  # 10 metre U&V wind component, Surface pressure, Sea surface temperature
        f"{prefix}-oper_stepType-max": ["fg10"],  # Maximum 10 metre wind gust since previous post-processing
        f"{prefix}-wave_stepType-instant": [
            "mwd",
            "mwp",
            "pp1d",
            "swh",
        ],  # Mean wave direction, Mean wave period, Peak wave period, Significant height of combined wind
        # waves and swell
        f"{prefix}-oper_stepType-accum": ["tp"],  # Total precipitation
    }
    # everywhere: valid_time  # seconds since 1970-01-01
    # longitude, latitude metadata

    var_suffix = "_Wind"
    var_prefix = "_"
    var_time = "time_s"

    # ImportFileHDF5() parameters
    grp = ""  # root
    cols_namemap = {
        grp: {
            **{p: p for params in files_params.values() for p in params},
            index_name: var_time,
            "u10": "u",
            "v10": "v",
            # "eastward_wind_sdd": "u_cm_sdd",
            # "northward_wind_sdd": "v_cm_sdd",
        },
    }

    scalers = {}
    with h5py.File(file, "r") as h:
        for col_in, col in cols_namemap[grp].items():
            # Get the add_offset and scale_factor values
            try:
                attributes = h[col_in].attrs
                scalers[col] = [attributes[k].item() for k in ["scale_factor", "add_offset"]]
            except KeyError:
                continue

        index = h[index_name]
        raw_time_units = index.attrs["units"].decode()
        try:
            time_scaler = scalers.pop(index_name)
        except KeyError:
            time_scaler = [1, 0]  # assume seconds from 1900-01-01

        t_shift_to_unix = -np.datetime64(raw_time_units.split("since ", 1)[-1], "s").astype(int)
        # old: 599616000 - 1230768000
        time_scaler[1] += time_shift_s - t_shift_to_unix

        if raw_time_units[0] == "h":
            time_scaler[0] *= 3600
            # time_scaler[1] *= 3600
        # raw_time_units = "s"

        if False:
            if any(np.isfinite(time_range)):
                # Check below is equal to:
                have_timedelta = isinstance(time_range[0], np.timedelta64)
                time_range_raw = np.array(time_range).astype(np.int64) + (
                    index[-1] if have_timedelta else (t_shift_to_unix - time_shift_s)
                )
                i_range = np.searchsorted(index, time_range_raw).tolist()
            else:
                i_range = [0, -1]

        i_range = search_time_range_indexes(
            index,
            time_range,
            time_scaler[1],
            raw_time_units=raw_time_units[0],  # "s" or "h"
        )
        if np.diff(i_range) == 0:
            i_range[1] += 1
            if i_range[1] == index.size:
                i_range[0] -= 1
                i_range[1] -= 1

        time_range = np.array(np.polyval(time_scaler, index[i_range]), "M8[s]")

        # For number of vector records <= 2 we will not draw in separate graphs, but display message?
        # if np.diff(i_range) <= 2:
        try:
            wind_mean_uv = complex(*[  # cdouble(real=0, imag=0) not works
                np.polyval(
                    scalers.get(cols_namemap[grp][col], [1, 0]),
                    np.mean(h[col][slice(*i_range), :, :]),
                )
                for col in ["u10", "v10"]  # ["eastward_wind", "northward_wind"]
            ])
            print("wind_mean_uv =", wind_mean_uv)
        except KeyError:
            wind_mean_uv = None
            print('"wind_mean_uv" not calculated as not found "u10" or "v10" in', file.name)

    print(f"loading params from {len(files)} files: ", end="")
    for file in files:
        if file_name_suffix:
            file_name_prefix = file.name.removesuffix(file_name_suffix)
            try:
                params = files_params[file_name_prefix]
                print(str(params), end=", ")
            except KeyError as e:
                print(f'Not known file prefix/suffix in "{file_name_prefix}", skipping file...')
                continue
        else:
            params = [p for params in files_params.values() for p in params]
        ImportFileHDF5(
            file,
            [grp],
            linked=True,
            namemap={
                "/".join([grp, col_in]): col for col_in, col in cols_namemap[grp].items() if col_in in params
            },
            slices={
                f"/{col_in}": (((i_range + [None]), 0, 0) if col_in != index_name else (i_range + [None],))
                for col_in, col in cols_namemap[grp].items()
                if col_in in params
            },
            suffix=var_suffix,  # "_EC"
            prefix=var_prefix,
        )

    # Select time range and convert time to Veusz format with displaying time zone

    SetData2DExpression(
        "iu_Wind",
        (
            f"atleast_2d(v.i_positive(v.i_use({var_prefix}{var_time}{var_suffix}, "
            f"USEtime_Wind, Wind_timeShift_s, t_units='s'), {var_prefix}{var_time}{var_suffix}.size))"
        ),
        linked=True,
    )
    SetDataExpression(
        "time_Wind",
        "".join(
            ["v.dt64s2vsz("]
            + ([f"{t_shift_to_unix} + "] if t_shift_to_unix != 0 else [])
            + ([f"{time_scaler[0]}*"] if time_scaler[0] != 1 else [])
            + [var_prefix, var_time, var_suffix]
            + ["[sl_(iu_Wind if diff(iu_Wind) > 1 else iu_Wind + [0, 1])]) + Wind_timeShift_s"]
        ),
        linked=True,
    )
    # Convert NetCDF stored values to physical
    if any(scalers):
        for col in scalers:
            col_scaled = col.replace("_cm", "")
            if col_scaled != col:
                SetDataExpression(
                    f"{col_scaled}{var_suffix}",
                    f"polyval({scalers[col]}, {var_prefix}{col}{var_suffix})",
                    linked=True,
                )
        print("NetCDF scaler attributes applied:", scalers)
    return (time_range, wind_mean_uv)


# def veusz_load_hdf5_nav(file, time_range, time_shift_s: int = 0, load_map=False, **kwargs):
#     """
#     :param time_range: any 2-element sequence convertible to datetime64 array in displaying zone. For example ['2020-08-19T21:59:22', '2020-08-20T06:05:04']
#     :param time_shift_s: shift in seconds You'll add to loaded time to display data, here used to set input range correctly
#     """
#     device_ids = ["navigation"]
#     # grp_d = None
#     if device_dir:  # use filtered db where coef. already applied
#         file = device_dir / f"{(file or device_dir).stem}.h5"

#     # cols_namemap = {
#     #     'table': {
#     #         'index': 't_ns',
#     #     }
#     # }

#     existed_devs, time_range, i_ranges = veusz_load_hdf5(
#         file,
#         device_ids,
#         grp_d_rename_funs={"table": lambda col, device_id: f"/nav1D/{col}"},
#         time_range=time_range,
#         **kwargs,
#     )


def veusz_load_csv_gmx500(
    file,
    time_range,
    time_shift_s=0,
    fun_get_time_ranges: Optional[Callable[[Tuple[np.ndarray]], np.ndarray]] = None,
    db=None,
    **kwargs,
):
    """
    Loads GMX 500 data from text *.csv file (example below, speed units: m/s, dir: degrees, blow from):
    ,Vdir-,Vabs,Vdir,Pa,Humidity,Temp,Dewpoint,time,               ,Voltage,,,Y,m,d,H,M,S,
    Q,161,000.04,027,1017.3,021,+022.6,-000.2,2023-11-21T10:00:21.8,+08.1,0000,33,2000,3,29,16,41,59
    Q,161,000.06,027,1017.3,021,+022.6,-000.3,2023-11-21T10:00:22.8,+08.1,0000,33,2000,3,29,16,42,0
    :param file: data file
    :param time_range: Not implemented: any 2-element sequence convertible to datetime64 array in displaying zone. For example ['2020-08-19T21:59:22', '2020-08-20T06:05:04']
    :param time_shift_s: Not implemented: shift in seconds You'll add to loaded time to display data, here used to set input range correctly
    :param db: Not implemented, path to hdf5 data with 'navigation' and CTD data tables
    :return: (time_range_raw, time, a1d, icol):
    - time_range_raw: loaded time range in displaying zone
    """
    if any(np.isfinite(time_range)) or fun_get_time_ranges:
        # Need limiting of data loading to Veusz
        if fun_get_time_ranges:
            raise NotImplementedError("Limiting of data loading to Veusz")  # todo
            # b_ok = fun_get_time_ranges(
            #     *a1d[:, [icol[col] for col in fun_get_time_ranges.__code__.co_varnames]].T
            # )
            # edges = bool2ranges(b_ok, min_range, min_range_del)
            # try:
            #     time_ranges = time[edges]
            # except IndexError:
            #     time_ranges = np.append(time[edges[edges < time.size]], time[-1] + np.timedelta64(1, "s"))

            # print(
            #     "Time ranges of data found:",
            #     ", ".join([f"{st} - {en}" for st, en in zip(time_ranges[::2], time_ranges[1::2])]),
            # )
            # time_range_raw = time_ranges[[0, -1]]
        else:
            time_ranges = []
    else:
        time_ranges = []
    # Get data to Veusz allowing to save vsz without copying data
    ImportFileCSV(
        file,
        blanksaredata=True,
        encoding="ascii",
        headermode="none",
        linked=True,
        dsprefix="_",
        renames={
            "_col10__": "_Voltage__",
            "_col13__": "_Y__",
            "_col14__": "_m__",
            "_col15__": "_d__",
            "_col16__": "_H__",
            "_col17__": "_M__",
            "_col18__": "_S__",
            "_col2__": "_Vdir__-",
            "_col3__": "_Vabs__",
            "_col4__": "_Vdir__",
            "_col5__": "_Pa__",
            "_col6__": "_Humidity__",
            "_col7__": "_Temp__",
            "_col8__": "_Dew_point__",
            "_col9__": "_time__",
        },
        skipwhitespace=True,
        dssuffix="__",
        textdelimiter="\x03",
    )

    # # Load navigation
    # veusz_load_hdf5_nav(db, time_range_raw, time_shift_s=time_shift_s, **kwargs)

    # # Load SBE CTD to correct ADV bad pressure (partly missed because of bad calibration)
    # existed_devs, time_range, i_ranges = veusz_load_hdf5(
    # db, ['CTD_SBE_911plus@Rozeta'], cols_namemap={'table': {'index', 'Pres', 'AltM'}},
    # grp_d_rename_funs={'table': lambda col, device_id: f'_{col}_ctd'},
    # time_range=time_range_raw, **kwargs
    # )

    return (time_ranges,)


def veusz_load_csv_ecmwf(
    file,
    time_range,
    time_shift_s=0,
    fun_get_time_ranges: Optional[Callable[[Tuple[np.ndarray]], np.ndarray]] = None,
    db=None,
    **kwargs,
):
    """
    Loads GMX 500 data from text *.csv file (example below, speed units: m/s, dir: degrees, blow from):
    Time	u10	v10	mwd	mwp	pp1d	swh	sp
    2023-11-01 00:00	8.1045	4.625	253.39	5.1031	5.8709	1.5198	99660
    :param file: data file
    :param time_range: Not implemented: any 2-element sequence convertible to datetime64 array in displaying zone. For example ['2020-08-19T21:59:22', '2020-08-20T06:05:04']
    :param time_shift_s: Not implemented: shift in seconds You'll add to loaded time to display data, here used to set input range correctly
    :param db: Not implemented, path to hdf5 data with 'navigation' and CTD data tables
    :return: (time_range_raw, time, a1d, icol):
    - time_range_raw: time range of raw data
    """
    if any(np.isfinite(time_range)) or fun_get_time_ranges:
        # Need limiting of data loading to Veusz
        if fun_get_time_ranges:
            raise NotImplementedError("Limiting of data loading to Veusz")  # todo
            # a1d = np.genfromtxt(file, usecols=list(range(n_start_cols)) + i_cols, **kwargs)
            # time = np.datetime64("%02.0f-%02.0f-%02.0f" % tuple(a1d[0, icols_date_reorder]), "s") + np.array(
            #     (np.append(0, np.cumsum(np.diff(a1d[:, idd]) != 0) * 24) + a1d[:, iHH]) * 3600
            #     + a1d[:, iMM] * 60
            #     + a1d[:, iSS],
            #     "m8[s]",
            # )
            # if any(np.isfinite(time_range)):
            #     iu = np.searchsorted(time, time_range).tolist()
            #     time = time[slice(*iu)]
            #     a1d = a1d[slice(*iu), n_start_cols:]
            # else:
            #     a1d = a1d[:, n_start_cols:]

            # b_ok = fun_get_time_ranges(
            #     *a1d[:, [icol[col] for col in fun_get_time_ranges.__code__.co_varnames]].T
            # )
            # edges = bool2ranges(b_ok, min_range, min_range_del)
            # try:
            #     time_ranges = time[edges]
            # except IndexError:
            #     time_ranges = np.append(time[edges[edges < time.size]], time[-1] + np.timedelta64(1, "s"))

            # print(
            #     f"Time ranges of data found:",
            #     ", ".join([f"{st} - {en}" for st, en in zip(time_ranges[::2], time_ranges[1::2])]),
            # )
            # time_range_raw = time_ranges[[0, -1]]
        else:
            time_ranges = []
    else:
        time_ranges = []

    # Get data to Veusz allowing to save vsz without copying data
    ImportFileCSV(
        file,
        blanksaredata=True,
        dateformat="YYYY-MM-DD hh:mm",
        delimiter="\t",
        encoding="cp1251",
        headermode="1st",
        linked=True,
        renames={"Time": "_timeUTC__", "u10": "_u__", "v10": "_v__"},  # UTC_Wind
    )
    return (time_ranges,)


def veusz_load_meteo(
    time_range, coords=None, time_shift_s=0, dev_wind=None, folder_name=None, dir_parent=None, msg_for=""
):
    """
    Load meteo DB data (netcdf data downloaded from CMEMS/ECMWF servers) which are in hdf5 files named by
    coordinates.
    :param folder_name: if not None loads data directly from `dir_parent / folder_name` else searches first
    folder (among ECMWF, CMEMS) with data in `dir_parent`
    :param coordinates: try to find nearest DB file
    :return: dev_wind, time_range_raw_wind, wind_mean_uv
    """
    file_wind = None
    coord_patterns = [
        r"E(?P<lon>\d+.?\d*)[,_]N(?P<lat>\d+.?\d*)",
        r"N(?P<lat>\d+.?\d*)[,_]E(?P<lon>\d+.?\d*)",
        r"(?P<lon>\d+.?\d*)E[,_](?P<lat>\d+.?\d*)N",
        r"(?P<lat>\d+.?\d*)N[,_](?P<lon>\d+.?\d*)E",
    ]

    choices = {}  # folder, (dev_wind, fun_load)
    if (dev_wind and "CMEMS" not in dev_wind) and folder_name != "CMEMS":
        choices["ECMWF"] = ("'ECMWF ERA5'", veusz_load_hdf5_ecmwf)
    if (dev_wind and not dev_wind.startswith("ECMWF")) and folder_name != "ECMWF":
        choices["CMEMS"] = ("ESCAPE('CMEMS WIND_GLO_PHY_L4_NRT_012_004')", veusz_load_hdf5_cmems)

    time_range_raw_wind = [NaT, NaT]
    wind_mean_uv = None
    for suffix in [".nc", ".tsv"]:
        for folder, (dev_wind, fun_load) in choices.items():
            _ = dir_parent / (folder_name if folder_name else folder)

            # Find coordinates from file names to select file nearest to probe
            coords_to_path = {}
            coords = None

            files = list(_.glob(f"*{suffix}"))
            if not files:
                try:
                    files = [d for d in _.iterdir() if d.is_dir() and d.name.startswith("area(")]
                except FileNotFoundError:
                    if not _.is_dir():
                        l.warning(f"Can not load meteo ({_} is not a dir) required for {[dev_wind]} {msg_for}")

            for file in files:
                for coord_pattern in coord_patterns:
                    try:
                        coord_wind = re.search(coord_pattern, file.name).groupdict()
                        coords_to_path[(float(coord_wind["lat"]), float(coord_wind["lon"]))] = file
                    except AttributeError:
                        continue

            if files:  # Wind file have been found
                if coords_to_path and coords:
                    # Check the distance to probe
                    coords_wind = list(coords_to_path.keys())
                    (dx, dy, dist_m, bearing) = fv.dx_dy_dist_bearing(
                        *coords[::-1],
                        *np.fliplr(coords_wind).T,
                    ).T
                    if len(coords_to_path) > 1:
                        print("Selecting file with coordinates nearest to coordinates of 1st probe")
                        i_coord = np.argmin(dist_m)
                        file_wind = coords_to_path[coords_wind[i_coord]]
                    else:
                        file_wind = next(iter(coords_to_path.values()))
                        i_coord = 0
                    print(
                        f"Found nearest {dev_wind} data point with distance to 1st probe point: "
                        f"{(dist_m[i_coord] if isinstance(dist_m, np.ndarray) else dist_m) / 1e3:g}km"
                    )
                else:
                    file_wind = (
                        next(f for f in files if "wind" in f.name) or files[0]
                    )  # prefer "wind" in name
                    if not coords_to_path:
                        print(
                            f"No coords found for {dev_wind} data. Selecting 1st file. Consider "
                            f"to include coordinates into file name ({coord_patterns[0]}) to enable "
                            "assess data distance"
                        )
                    if not coords:
                        print(
                            f"No coords found for {msg_for} data. Selecting 1st file."
                            "Consider to include coordinates into device_info.json to enable assess "
                            "data distance"
                        )

                print(f"Loading wind from {file_wind}...")
                if suffix == ".nc":
                    time_range_raw_wind, wind_mean_uv = fun_load(
                        file_wind,
                        time_range,
                        time_shift_s=time_shift_s,
                    )
                elif folder == "CMEMS":
                    print("loading *.tcv from CMEMS not implemented")
                    file_wind = None
                    continue
                break  # to load tcv from ECMWF
            else:
                continue
        if file_wind:
            break

    if file_wind:
        if file_wind.suffix == ".tcv" and file_wind.parent == "ECMWF":
            # '../../meteo/220601wind@ECMWF-ERA5(N55.875,E19.116).tsv'
            print("loading wind from", file_wind)
            ImportFileCSV(
                file_wind,
                blanksaredata=True,
                dateformat="YYYY-MM-DD hh:mm",
                delimiter="\t",
                encoding="cp1251",
                headermode="1st",
                linked=True,
                renames={
                    "Time": "stimeUTC_Wind",
                    "u10": "u_Wind",
                    "v10": "v_Wind",
                },
            )
            # Select time range and convert time to Veusz
            SetData2DExpression(
                "iu_Wind",
                (
                    "atleast_2d(v.i_positive(v.i_use(stimeUTC_Wind, USEtime_Wind, 1230768000 +"
                    " Wind_timeShift_s, t_units='s'), stimeUTC_Wind.size))"
                ),
                linked=True,
            )
            SetDataExpression(
                "time_Wind",
                "stimeUTC_Wind[sl_(iu_Wind)] + Wind_timeShift_s",
                linked=True,
            )
            dev_wind = "'ECMWF ERA5'"  # for "info_wind" Custom Definition
    else:
        print("No wind data found!")

    if dev_wind:
        pid = "_Wind"
        SetDataExpression(f"dt{pid}", f"min(diff(time{pid}[1:4]))", linked=True)
        bin = "bin2_"
        SetDataExpression(
            f"{bin}i0st{pid}",
            f"v.i_whole_time_intervals(time{pid}, WIND_bin_average_s)",
            linked=True,
        )
        SetDataExpression(f"{bin}t0st{pid}", f"time{pid}[int32({bin}i0st{pid})]", linked=True)
        SetDataExpression(
            f"{bin}u{pid}",
            f"v.bin_avg(u{pid}[sl_(iu{pid})], {bin}i0st{pid})",
            linked=True,
        )
        SetDataExpression(
            f"{bin}v{pid}",
            f"v.bin_avg(v{pid}[sl_(iu{pid})], {bin}i0st{pid})",
            linked=True,
        )
        SetDataExpression(f"{bin}iu{pid}", "None", linked=True)  # to plot all wind?

        SetData2DExpression(
            f"{bin}iu_cmn{pid}",
            f"[searchsorted({bin}t0st{pid}, time_span_i_common) + int32([0, -1])]",
            linked=True,
        )

    return dev_wind, time_range_raw_wind,wind_mean_uv
