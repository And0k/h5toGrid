"""
Globals used:
- parent: Path of executed file
- basename: vsz file base name
parent.name dir string (or basename with bigger priority) is used to get interval [s] of data to load.
Before digits and after units may be any characters that will not be used here (see re_dt)
Date units: years 'Y', months 'M', weeks 'W', and days 'D',
time units: hours 'h', minutes 'm', seconds 's'
"""

import json
from logging import info, warning, exception
from itertools import groupby, dropwhile
import sys
from pathlib import Path
from runpy import run_module
from typing import Any, Callable, Iterable, Mapping, Optional, Tuple, Sequence, Union
from time import strptime
from datetime import datetime
from calendar import monthrange
import numpy as np
import re
import h5py

import func_vsz as fv
re_dt = r"(?P<dt>\d+\.?\d*)(?P<dt_u>[YMWDhms])(?:in)?"

NaT = np.datetime64("NaT")

class DumbMapping(dict):
    def __getitem__(self, key):
        return self.get(key, key)

class Custom:
    """
    Set Veusz `Custom Definition` through str representation and assign original value to instance attribute.
    `add_later(key=value, ...)` used to accumulate attributes without adding them to Veusz, and be able to
    use them through `postponed` attribute. Then use `add_postponed()` to add them to Veusz at once.
    """
    def __init__(self):
        super().__setattr__("postponed", {})

    def __setattr__(self, key, value, comment=""):
        super().__setattr__(key, value)
        if key != "postponed":
            AddCustom("definition", key, str(value) if comment is None else f"{value}  # {comment}")
        else:
            raise NameError(
                "`postponed` used internally, do not edit it directly. Use `add_later()` instead"
            )

    def add(self, comments=None, **kwargs):
        """
        Set Veusz `Custom Definition` through `str(value)` and assign `value` to
        instance attribute for each item of `kwargs`
        :param kwargs: {key: value} dict
        """
        n_kwargs = len(kwargs)

        if comments:
            comments = [comments] if isinstance(comments, str) else comments
            n_comments = len(comments) if comments else 0
            if n_kwargs != n_comments:
                raise ValueError(
                    f"Number of comments ({n_comments}) must be equal to number of kwargs ({n_kwargs})"
                )
            for (k, v), comment in zip(kwargs.items(), comments):
                self.__setattr__(k, v, comment)
        else:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def add_later(self, **kwargs):
        self.postponed.update(kwargs)

    def add_postponed(self, **kwargs):
        self.add(**self.postponed)
        super().__setattr__("postponed", {})  # clear postponed attribute

cus = Custom()


def add_months(t: np.timedelta64, months: int):
    """Add months to a numpy.datetime64[s] object."""
    # Convert numpy datetime64 to a datetime object
    ts = t.item()
    year, month, day, *time_part, _, _, _ = ts.timetuple()
    # Add months handling year overflow
    year, month = divmod(year * 12 + month + months - 1, 12)
    month += 1  # Month is 1-based
    day = min(day, monthrange(year, month)[1])  # Handle month day overflow

    # Construct new datetime with the new year, month, day
    new_dt = datetime(year, month, day, *time_part)

    # Convert back to numpy datetime64
    return np.datetime64(new_dt.strftime("%Y-%m-%dT%H:%M:%S"))


def get_info_from_filename(basename) -> Tuple[Optional[Tuple[Any]], Mapping[str, Any]]:
    """
    :param basename: vsz file base name - string to get interval [s] of data to load.
    After units may be any characters that will not be used here
    {t_start or dt_to_last}{dt}[@{type}{model}{number}], where:
    - t_start: start date and time,
    - dt_to_last: time interval to last db data to find t_start.
    - dt: time interval to overwrite ``dt`` argument,
    - {type}{model}{number} - optional device info for allowed devices
    (for example, "wind" is not allowed: You need remove it before call)
    Date units are years ('Y'), months ('M'), weeks ('W'), and days ('D'), while the time units are hours ('h'), minutes ('m'), seconds ('s')
    :return: (time_range, out_info) where out_info dict of info extracted from basename, without info about
    time_range
    Prints '"{basename}" -> {time_range}, {out_info}'
    Example
    If dt = '5min' then converts:
    - If dt = '5min':  200917_2000i03.vsz -> (['2020-09-17T20:00:00', '2020-09-17T20:05:00'], 'incl03')
    - If last data in db was at '2020-09-17T20:20:00'
    3h_to_last_tr0 -> (['2020-09-17T20:15:00', '2020-09-17T20:20:00'], 'tr0')
    """
    print(f'"{basename}"', end=" ")
    max_devices_idx = 9  # allow comma separated list of this + 1 number of devices
    re_time = r"(?P<yy>\d\d)(?P<mm>\d\d)(?P<dd>\d\d)[_T]?(?P<HH>\d\d)?(?P<MM>\d\d)?(?P<SS>\d\d)?"
    d = r"\d"  # digits
    custom_device_types = ('i', 'w', 'tr')  # our devices which pids will have zero prefix if `number` < 2
    def re_model_and_number(i: int):
        """
        Regex to get `model` and `number`
        Must ends with number except for last device where allowed any [a-zA-Z_-] chars
        optionally ending with numbers (for `number`)
        Regex field `is_type_mod` for last (used only if one) device just denotes that user wants the type and model be used togather
        """
        return (
            f"(?P<model{i}>"
            f"[DBdbp]?)(?P<number{i}>{d}[{d}-]*{d}|{d})"
            if i < max_devices_idx
            else f"(?P<is_type_mod>-?)(?P<model{i}>[a-zA-Z_-]*)(?P<number{i}>{d}[{d}-]*{d}|{d}*)"
        )

    re_exp = (
        r"(?:"
        r"(?:{time}|(?P<dt_to_last>\d+(?:h|s))_to_last)?"
        r"(?:-{time_end})?(?:[_,]?(?:dt=)?{dt})?"
        r")?(?:[,_]?d(?P<decimation>\d+))?(?:[,_ -]*(?P<descr>[^@\d][^@]*))?(?:@{pids})?\.vsz"
    ).format(
        time=re_time,  # end time have same parts but optional and with new names:
        time_end=re_time.replace(">", "e>").replace(")(", ")?(").replace(")[", ")?["),
        dt=re_dt,
        pids="".join(  # some allowed types&models are switches under "Load text file(s) in Veusz" below
            rf'?:{"?," if i else ""}?((?P<type{i}>i(ncl)?|w|tr|ADV|ECMWF|CMEMS|)_?{re_model_and_number(i)})'
            for i in range(max_devices_idx + 1)
        ),
    )
    re_parts = re.match(re_exp, basename)
    if re_parts is not None:
        re_parts = re_parts.groupdict()
    if re_parts is None:
        raise (NameError(f'File name: "{basename}" not matches regex "{re_exp}"'))
    elif re_parts["dt_to_last"]:
        t_start = np.timedelta64(-1, re_parts["dt_to_last"])
    elif re_parts["dd"]:
        for re_time_part in ["SS", "MM", "HH"]:
            if not re_parts[re_time_part]:
                re_parts[re_time_part] = "00"
        t_start = np.datetime64("20{yy}-{mm}-{dd}T{HH}:{MM}:{SS}".format_map(re_parts))
    else:
        t_start = None

    if t_start:
        if re_parts["dt"]:
            # Overwrite time_range
            if "." in re_parts["dt"]:
                re_parts["dt"], _ = re_parts["dt"].split(".", 1)
                dt = np.timedelta64(_, re_parts["dt_u"]).astype("m8[s]") / 10 ** len(_)
            else:
                dt = 0
            dt += np.timedelta64(re_parts["dt"], re_parts["dt_u"])
            time_range = [
                t_start,
                add_months(t_start, dt.astype("m8[M]").astype(int))
                if dt.dtype.str[-1] in ["Y", "M", "W"]
                else t_start + dt.astype("m8[s]"),
            ]
        elif re_parts["yye"]:
            # Parse end date
            n_add_months = 0
            if re_parts["dde"] is None:
                # Some date parts skipped in basename and regex returned them shifted to variables of bigger
                # time scale. => Shift back, replacing skipped with corresponding parts of start date
                if (
                    re_parts["mme"] is None
                ):  # shift on 2 vars: (yye) mme dde -> yye mme (dde)
                    re_parts["mme"], re_parts["dde"] = re_parts["mm"], re_parts["yye"]
                    if re_parts["dde"] < re_parts["dd"]:  # need to increase mme by 1
                        n_add_months = 1
                else:  # shift on 1 var: (yye) [mme] dde -> yye (mme) [dde]
                    re_parts["mme"], re_parts["dde"] = re_parts["yye"], re_parts["mme"]
                    if re_parts["mme"] < re_parts["mm"]:  # need to increase yye by 1
                        n_add_months = 12
                re_parts["yye"] = re_parts["yy"]
            # replace skipped end time parts in basename with '00'
            if re_parts["SSe"] is None:
                re_parts["SSe"] = "00"
            if re_parts["MMe"] is None:
                re_parts["MMe"] = "00"
            if re_parts["HHe"] is None:
                re_parts["HHe"] = "00"
            t_end = np.datetime64(
                "20{yye}-{mme}-{dde}T{HHe}:{MMe}:{SSe}".format_map(re_parts)
            )
            if n_add_months:
                t_end = add_months(t_end, n_add_months)
            time_range = [t_start, t_end]
        else:
            time_range = [t_start, t_start]
    else:
        time_range = []

    devices = {}
    model = None
    device_type = None
    for i in range(max_devices_idx + 1):
        number = re_parts.pop(f"number{i}")
        if not number:
            device_type_cur = re_parts.pop(f"type{i}")
            model = re_parts.pop(f"model{i}")
            continue
        if "-" in number:
            st, en = number.split("-")
            numbers = range(int(st), int(en) + 1)
        else:
            numbers = [number]
        for number in numbers:
            number = int(number)
            try:
                device_type_cur = re_parts.pop(f"type{i}")
            except KeyError:
                pass
            else:
                if device_type_cur:
                    device_type = device_type_cur.replace("incl", "i", 1)
            try:
                # model if is not specified: if device type is specified then set to default 'i' else previous
                model = re_parts.pop(f"model{i}") or ("i" if device_type_cur else model)
            except KeyError:
                pass
            pid = f"{model or device_type}{number:{'02d' if device_type in custom_device_types else 'd'}}"
            devices[pid] = {"type": device_type, "model": model, "number": number}
    out_info = {
        "devices": devices or {
            device_type_cur: {
                "type": device_type_cur,
                "model": model or device_type_cur,
                "number": number,
            }
        },
        "descr": re_parts["descr"],
        "is_type_mod": bool(re_parts["is_type_mod"])
    }
    if re_parts["decimation"]:
        out_info["decimation"] = int(re_parts["decimation"])
    print("-> ", time_range, out_info, end="")
    return time_range, out_info


def normalize_device_id(device_id: str) -> str:
    """
    Normalize a device ID by removing all underscores before numeric part and leading zeros from the numeric
    part.

    Examples:
    - i_03 -> i3
    - i_b27 -> ib27
    - i_p06 -> ip6
    - ip06 -> ip6
    - w01 -> w1
    """
    match = re.match(r"^(?P<letters>[a-z_]*)0*(?P<number>\d*)", device_id)
    if match:
        groups = match.groupdict()
        groups["letters"] = groups["letters"].replace("_", "")
        return "{letters}{number}".format_map(groups)


def search_time_range_indexes(index, time_range, raw_time_shift_s, raw_time_units="ns"):
    """
    :param index:
    :param time_range:
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
                to_raw_time_units = np.timedelta64(1, raw_time_units).astype("m8[ns]").astype(int).item()
                raw_search_add = -to_raw_time_units*raw_time_shift_s
            else:
                raw_search_add = 0
        if np.isfinite(time_range[-1]):
            time_range_raw_search = np.array(time_range).astype(np.int64)
            i_range = np.fmin(
                np.searchsorted(index, time_range_raw_search + raw_search_add), len(index) - 1
            ).tolist()
        else:
            time_range_raw_search = np.array(time_range[0]).astype(np.int64)
            i_range = [np.searchsorted(index, time_range_raw_search + raw_search_add).item(), len(index) - 1]
    else:
        i_range = [0, -1]
    return i_range



def veusz_load_hdf5(
    file,
    device_ids: Sequence[str],
    grp_d: Optional[Mapping[str, str]] = None,
    cols_namemap: Optional[
        Mapping[str, Union[Mapping[str, str], Iterable[str]]]
    ] = None,
    grp_d_rename_funs: Mapping[str, Callable[[str, str], str]] = None,
    time_range=tuple(),
    time_shift_s: int = 0,
    b_load_to_veusz = True,
    decimation: Optional[int] = None,
) -> Tuple[Mapping[str, Any], Tuple[np.timedelta64, np.timedelta64], Mapping[str, list|np.ndarray]]:
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
        max_time_span_s_strings = [f"{t}" for t in time_range]
        print("Loading interval", max_time_span_s_strings, "from", file)
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
    if grp_d is None:
        grp_d = {"table": "/table/"}

    # set default mappings where skipped
    if cols_namemap is None:
        cols_namemap = {grp: DumbMapping() for grp in grp_d}
    else:
        cols_namemap = {
            grp: {s: s for s in map_or_seq}
            if not isinstance(map_or_seq, Mapping)
            else map_or_seq
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

                            no_cols = set(cols_namemap[grp]).difference(
                                existed_devs[device_id][grp]
                            )
                            if no_cols:
                                print(
                                    f"No columns: {no_cols}, existed_devs: {existed_devs[device_id][grp]}"
                                )
                                # for k in no_cols:
                                #     # del existed_devs[device_id][grp][k]
                                #     del cols_namemap[grp][k]
                    except KeyError:
                        existed_devs[device_id][grp] = None
                        continue

                print(end=", " if i < len(device_ids) else ". ")
                index = h[grp_dev[device_id]["table"]]["index"]
                # todo: check this to replace code below:
                # i_ranges[device_id] = search_time_range_indexes(index, time_range, time_shift_s)
                if any(np.isfinite(time_range)):
                    time_range_raw_cur = np.int64(time_range) + (
                        index[-1] if have_timedelta else -1e9 * time_shift_s
                    )
                    i_ranges[device_id] = np.searchsorted(
                        index, time_range_raw_cur
                    ).tolist()
                    try:
                        time_range_raw = index[i_ranges[device_id]]
                    except IndexError:
                        _ = min(i_ranges[device_id][-1], len(index) - 1)
                        if _ <= i_ranges[device_id][0]:
                            raise IndexError(
                                "Required time range {} is after the data range {}".format(
                                    time_range,
                                    np.array(index[[0, -1]], "M8[ns]").astype("M8[s]")
                            ))
                        try:
                            time_range_raw = index[[i_ranges[device_id][0], _]]
                        except IndexError:
                            raise IndexError(
                                "Required time range: {}, data range: {}".format(
                                    time_range,
                                    np.array(index[[0, -1]], "M8[ns]").astype("M8[s]")
                            )
                        )
                else:
                    time_range_raw = index[[0, -1]]
                    i_ranges[device_id] = [0, len(index)]

                time_raw_min = np.fmin(time_range_raw[0], time_raw_min)
                time_raw_max = np.fmax(time_range_raw[-1], time_raw_max)
            except Exception:
                exception(f'Working with HDF5 file "{file}", {grp_dev[device_id]}')
                raise
    time_range_raw = np.array([time_raw_min, time_raw_max], "M8[ns]")
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
            (
                list(namemap)
                or [_ for device_id in device_ids for _ in grp_dev[device_id].values()]
            ),
            linked=True,
            namemap=namemap,
            slices=slices,
        )
        # Add tag "loaded"
        TagDatasets(
            "loaded",
            sorted(
                [
                    grp_d_rename_funs["table"](cols_namemap["table"][col], device_id)
                    for device_id in device_ids
                    for col in cols_spec_or_existed(device_id, "table")
                ]
            ),
        )
    return existed_devs, time_range_out, i_ranges


def veusz_load_hdf5_tcm_raw(
    file,
    devices,
    time_range,
    time_shift_s: int = 0,
    cruise_dir=None,
    decimation: Optional[int] = None
):
    """
    :param devices: dict with fields equal to pids. Each value is a dict of parts of probe name, here used
    to construct pandas data table name '{type}{_}{pid}'
    :param file: data file
    :param time_range: any 2-element sequence convertible to datetime64 array in displaying zone. For example ['2020-08-19T21:59:22', '2020-08-20T06:05:04']
    :param time_shift_s: shift in seconds You'll add to loaded time to display data, here used to set input range correctly
    :param cruise_dir:
    :return:
    """
    # take 1st probe
    _ = iter(devices.items())
    pid, probe = next(_)
    _ = [(pid, probe)] + list(_)

    device_ids = [
        f"incl{pid[1:]}" if pid[0] == "i"
        else f"incl_{pid}" if probe["type"] == "i"
        else f"{probe['type']}_{pid}"
        for pid, probe in _
    ]

    grp_d = {
        "coef": "/coef/",
        "table": "/table/",
    }
    table_cols_to_slice_namemap_common = {  # common parameters which time slice is need to load
        "index": "t_ns",
        "P_counts": "P_counts",
        "Temp": "Temp",
        "Battery": "Battery",
    }
    if probe["type"] == "w":
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
            "talble": table_cols_to_slice_namemap_common,
        }

        if cruise_dir:  # use filtered db where coef. already applied
            file = cruise_dir / f"{cruise_dir.stem}.proc_noAvg.h5"
            if not Path(file).is_file():  # try & except not works here (OSError)
                file = cruise_dir.stem / f"{cruise_dir.stem}@w.proc_noAvg.h5"

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

    # prefixes and suffixes for columns of each grp_d group
    def f_table_cols_fmt(col, device_id):
        return f"{col}" if col.endswith("counts") else f"_{col}__"

    grp_d_rename_funs = {"coef": "{}".format, "table": f_table_cols_fmt}

    existed_devs, time_range, i_ranges = veusz_load_hdf5(
        file,
        device_ids,
        grp_d,
        cols_namemap,
        grp_d_rename_funs,
        time_range,
        time_shift_s=time_shift_s,
        decimation=decimation
    )

    # Dummy coef if not existed in DB:
    for device_id, grp_d in existed_devs.items():
        if "coef" not in grp_d.keys():
            print(
                f'No "coef" in DB table {device_id} => assinging dummy in Custom Definitions.'
            )
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
                TagDatasets(
                    "coefficient", [cols_namemap["coef"][k] for k in grp_d["coef"]]
                )
            except KeyError as e:  # not all possible coef must exists
                print("KeyError:", e, "Skipping TagDatasets and continue...")

    return (
        existed_devs,
        time_range,
        file,
        cols_namemap,
        grp_d_rename_funs,
        f_table_cols_fmt,
    )  #


def veusz_load_hdf5_ctd_profile(
    file,
    time_range,
    device=None,
    time_shift_s: int = 0,
    n_runs=1,
    params_d=None,
    renames_no_prefix={
        "table_log": {"index": "DateSt", "rows": "downcast_len"},
        "table": {"Turb": "Turb_nof", "Fluor": "ChlA_nof"},  # 'Pres': 'Pres_NoSep',
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
    grp_d = {
        "table": f"/{device}/table/",
        "table_log": f"/{device}/logRuns/table/"
    }
    if params_d is None:
        params_d={tbl: [] for tbl in grp_d}

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
                    h[grp_d["table"]]["index"],
                    time_range_raw[0] if log_i_st == log_i_en else time_range_raw
                )
                # length of last run
                len_last_run = (
                    h[grp_d["table_log"]]["rows"][log_i_en]
                    + h[grp_d["table_log"]]["rows_filtered"][log_i_en]
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
                np.int64(np.array(log_index, "M8[ns]").astype("M8[s]"))
                + time_shift_s,
            )
    print(f"corresponding data indices in {file}:", i_ranges)
    if not fun_custom:
        ImportFileHDF5(
            str(file),
            [f"{grp_d[t]}{p}" for t, pp in params_d.items() for p in pp] + ["/map"],
            linked=True,
            namemap={
                f"{grp_d[t]}{p}": f"{prefix_d[t]}{p}"
                for t, pp in params_d.items()
                for p in pp
            },
            slices={
                f"{grp_d[t]}{p}": (i_ranges[t] + [None],)
                for t, pp in params_d.items()
                for p in pp
            },
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


def veusz_load_hdf5_nav(
    file, time_range, time_shift_s: int = 0, load_map=False, **kwargs
):
    """
    :param time_range: any 2-element sequence convertible to datetime64 array in displaying zone. For example ['2020-08-19T21:59:22', '2020-08-20T06:05:04']
    :param time_shift_s: shift in seconds You'll add to loaded time to display data, here used to set input range correctly
    """
    device_ids = ["navigation"]
    # grp_d = None
    if cruise_dir:  # use filtered db where coef. already applied
        file = cruise_dir / f"{(file or cruise_dir).stem}.h5"

    # cols_namemap = {
    #     'table': {
    #         'index': 't_ns',
    #     }
    # }

    existed_devs, time_range, i_ranges = veusz_load_hdf5(
        file,
        device_ids,
        grp_d_rename_funs={"table": lambda col, device_id: f"/nav1D/{col}"},
        time_range=time_range,
        **kwargs,
    )

wind_mean_uv = None

def veusz_load_hdf5_cmems(
    file, time_range, time_shift_s: int = 0, load_map=False, **kwargs
):
    """
    :param file: like "../CMEMS/cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H_multi-vars_20.31E_54.94N_2023-08-20-2023-09-20.nc"
    :param time_range: any 2-element sequence convertible to datetime64 array in displaying zone. For example ['2020-08-19T21:59:22', '2020-08-20T06:05:04']
    :param time_shift_s: shift in seconds You'll add to loaded time to display data, here used to set input range correctly
    :return loaded time range in displaying zone
    """
    global wind_mean_uv
    grp = ""  # root
    cols_namemap = {
        grp: {
            "time": "timeUnix",
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
        index = h["time"]
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

    ImportFileHDF5(
        file,
        [grp],
        linked=True,
        namemap={"/".join([grp, col_in]): col for col_in, col in cols_namemap[grp].items()},
        slices={
            f"/{col}": (
                ((i_range + [None]), 0, 0) if col != "time" else (i_range + [None],)
            )
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

    return (time_range,)


def veusz_load_hdf5_ecmwf(
    file, time_range, time_shift_s: int = 0, load_map=False, **kwargs
):
    """
    :param file: file with mask `data_stream-(oper|wave)_stepType-(instant|max|accum)(.*).nc`
    like "../ECMWF/data_stream-oper_stepType-instant_20.31E_54.94N_2023-08-20-2023-09-20.nc"
    (.*) part should be constant for each of required files
    :param time_range: any 2-element sequence convertible to datetime64 array in displaying zone. For example ['2020-08-19T21:59:22', '2020-08-20T06:05:04']
    :param time_shift_s: shift in seconds You'll add to loaded time to display data, here used to set input range correctly
    :return loaded time range in displaying zone
    Updates global `wind_mean_uv`
    """
    global wind_mean_uv
    if file.is_dir():
        file = next(file.glob("*-oper_stepType-instant.nc"))

    required_name_part = "-(oper|wave)_stepType-(instant|max|accum)"
    m = re.match(
        fr"(?P<file_name_prefix>[^-]+){required_name_part}(?P<file_name_suffix>.*\.nc)", file.name
    )
    prefix = m.group("file_name_prefix")
    file_name_suffix = m.group("file_name_suffix")
    files = [
        f for f in file.parent.glob(
            re.sub(  # regex to glob pattern:
                r"\([^)]+\)",
                "*",
                f"{prefix}{required_name_part}{file_name_suffix}"
            )
        )
    ]

    index_name = "valid_time"
    files_params = {
        f"{prefix}-oper_stepType-instant": [
            "u10",
            "v10",
            "sp",
            "sst",
            index_name,
        ],  # 10 metre U&V wind component, Surface pressure, Sea surface temperature
        f"{prefix}-oper_stepType-max": [
            "fg10"
        ],  # Maximum 10 metre wind gust since previous post-processing
        f"{prefix}-wave_stepType-instant": [
            "mwd",
            "mwp",
            "pp1d",
            "swh",
        ],  # Mean wave direction, Mean wave period, Peak wave period, Significant height of combined wind
        # waves and swell
        f"{prefix}-oper_stepType-accum": [
            "tp"
        ],  # Total precipitation
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
    t_shift_to_unix = 0  # 599616000 - 1230768000
    time_raw_to_input = time_shift_s - t_shift_to_unix

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
            index, time_range, time_raw_to_input, raw_time_units="s"
        )
        time_range = np.array(index[i_range] + time_raw_to_input, "M8[s]")


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
            print('wind_mean_uv =' , wind_mean_uv)
        except KeyError:
            print('"wind_mean_uv" not calculated as not found "u10" or "v10" in' , file.name)
    print(f"loading params from {len(files)} files: ", end='')
    for file in files:
        file_name_prefix = file.name.removesuffix(file_name_suffix)
        try:
            params = files_params[file_name_prefix]
            print(str(params), end=', ')
        except KeyError as e:
            print(f'Not known file prefix/suffix in "{file_name_prefix}", skipping file...')
            continue
        ImportFileHDF5(
            file,
            [grp],
            linked=True,
            namemap={
                "/".join([grp, col_in]): col
                for col_in, col in cols_namemap[grp].items()
                if col_in in params
            },
            slices={
                f"/{col_in}": (
                    ((i_range + [None]), 0, 0)
                    if col_in != index_name
                    else (i_range + [None],)
                )
                for col_in, col in cols_namemap[grp].items()
                if col_in in params
            },
            suffix=var_suffix,  # "_EC"
            prefix=var_prefix
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
        f"v.dt64s2vsz({var_prefix}{var_time}{var_suffix}"
        "[sl_(iu_Wind if diff(iu_Wind) > 1 else iu_Wind + [0, 1])]) + Wind_timeShift_s",
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
    return (time_range,)



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
            b_ok = fun_get_time_ranges(
                *a1d[
                    :, [icol[col] for col in fun_get_time_ranges.__code__.co_varnames]
                ].T
            )
            edges = bool2ranges(b_ok, min_range, min_range_del)
            try:
                time_ranges = time[edges]
            except IndexError:
                time_ranges = np.append(
                    time[edges[edges < time.size]], time[-1] + np.timedelta64(1, "s")
                )

            print(
                "Time ranges of data found:",
                ", ".join(
                    [
                        f"{st} - {en}"
                        for st, en in zip(time_ranges[::2], time_ranges[1::2])
                    ]
                ),
            )
            time_range_raw = time_ranges[[0, -1]]
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
            a1d = np.genfromtxt(
                file, usecols=list(range(n_start_cols)) + i_cols, **kwargs
            )
            time = np.datetime64(
                "%02.0f-%02.0f-%02.0f" % tuple(a1d[0, icols_date_reorder]), "s"
            ) + np.array(
                (np.append(0, np.cumsum(np.diff(a1d[:, idd]) != 0) * 24) + a1d[:, iHH])
                * 3600
                + a1d[:, iMM] * 60
                + a1d[:, iSS],
                "m8[s]",
            )
            if any(np.isfinite(time_range)):
                iu = np.searchsorted(time, time_range).tolist()
                time = time[slice(*iu)]
                a1d = a1d[slice(*iu), n_start_cols:]
            else:
                a1d = a1d[:, n_start_cols:]

            b_ok = fun_get_time_ranges(
                *a1d[
                    :, [icol[col] for col in fun_get_time_ranges.__code__.co_varnames]
                ].T
            )
            edges = bool2ranges(b_ok, min_range, min_range_del)
            try:
                time_ranges = time[edges]
            except IndexError:
                time_ranges = np.append(
                    time[edges[edges < time.size]], time[-1] + np.timedelta64(1, "s")
                )

            print(
                f"Time ranges of data found:",
                ", ".join(
                    [
                        f"{st} - {en}"
                        for st, en in zip(time_ranges[::2], time_ranges[1::2])
                    ]
                ),
            )
            time_range_raw = time_ranges[[0, -1]]
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


def _config_text_header_dtype(text_type) -> dict[str, Any]:
    if text_type is None:
        return {}
    if text_type not in ("i", "p", "b", "d", "w"):
        raise TypeError("Probe model not recognized!")
    b_default_type = text_type in ("i", "", "b")
    cfg_for_type = {
        "header": "yyyy(text),mm(text),dd(text),HH(text),MM(text),SS(text),Ax,Ay,Az,Mx,My,Mz"
        + (",Battery,Temp" if b_default_type else ",P_counts,Temp,Battery"),
        "dtype": "|S4 |S2 |S2 |S2 |S2 |S2 i2 i2 i2 i2 i2 i2 f8 f8".split()
        + ([] if b_default_type else ["f8"]),
    }
    return cfg_for_type


def veusz_load_csv_tcm_raw(
    db,
    time_range,
    file,
    time_shift_s=0,
    probe_info: Optional[dict] = None,
    fun_get_time_ranges: Optional[Callable[[Tuple[np.ndarray]], np.ndarray]] = None,
    **kwargs,
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
            a1d = np.genfromtxt(file, usecols=list(range(n_start_cols)) + i_cols, **kwargs)
            time = np.datetime64("%02.0f-%02.0f-%02.0f" % tuple(a1d[0, icols_date_reorder]), "s") + np.array(
                (np.append(0, np.cumsum(np.diff(a1d[:, idd]) != 0) * 24) + a1d[:, iHH]) * 3600
                + a1d[:, iMM] * 60
                + a1d[:, iSS],
                "m8[s]",
            )
            if any(np.isfinite(time_range)):
                iu = np.searchsorted(time, time_range).tolist()
                time = time[slice(*iu)]
                a1d = a1d[slice(*iu), n_start_cols:]
            else:
                a1d = a1d[:, n_start_cols:]

            b_ok = fun_get_time_ranges(
                *a1d[:, [icol[col] for col in fun_get_time_ranges.__code__.co_varnames]].T
            )
            edges = bool2ranges(b_ok, min_range, min_range_del)
            try:
                time_ranges = time[edges]
            except IndexError:
                time_ranges = np.append(time[edges[edges < time.size]], time[-1] + np.timedelta64(1, "s"))

            print(
                f"Time ranges of data found:",
                ", ".join([f"{st} - {en}" for st, en in zip(time_ranges[::2], time_ranges[1::2])]),
            )
            time_range_raw = time_ranges[[0, -1]]
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
    config = _config_text_header_dtype(probe_model[0] if probe_model else None)

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
        'Ax': 'Ax_counts',
        'Ay': 'Ay_counts',
        'Az': 'Az_counts',
        'Mx': 'Mx_counts',
        'My': 'My_counts',
        'Mz': 'Mz_counts',
        # 'Battery': 'Battery',
        'Temp': '_Temp__',
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
        encoding="ascii",
        headermode="none",
        linked=True,
        renames=renames,
        rowsignore=2,
        skipwhitespace=True,
    )

    # Convertion variables to that used with hdf5 drawers
    SetDataExpression(
        "time__",
        "(lambda x: v.rep2mean(x, ediff1d(x, to_end=0)!=0))(fdate([_y, _m, _d]) + 3600*_H+ 60*_M + _S) + "
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


    return (time_ranges,)



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
    if (
        b_del_bad_interval.any()
    ):  # delete starts and ends of too short no data intervals
        edges = edges[np.hstack((True, ~np.repeat(b_del_bad_interval, 2), True))]
        n_rows = np.diff(edges)
    b_del_good_interval = n_rows[::2] < min_range
    if b_del_good_interval.any():  # delete starts and ends of too short data intervals
        edges = edges[~np.repeat(b_del_good_interval, 2)]
    return edges


def zone_to_seconds_offset(zone: str):
    hours = zone.removeprefix("UTC")
    if not hours:
        return 0
    return strptime(f'{hours[0]}{f"{hours[1:]:>02s}":<04s}', "%z").tm_gmtoff


def _info_json_item_array_to_dict(
    p, b, bd, s, lat=None, lon=None, time_st="", time_en="", burst_dt=None, bursts_t=None
):
    return dict(
        zip(
            "pbdscrtT",
            [
                p.format_map(fv.I),
                b,
                None if None in (b, bd) else round(b - bd, 1),
                s,
            ]
            + ([(lat, lon)] if lat else [])
            + ([(time_st, time_en)] if time_st else [])
            + ([burst_dt, bursts_t] if bursts_t else []),
        )
    )


def load_info_json(probes: dict, path_in: str | Path) -> dict:
    """
    Load device information from a JSON file and map it to the provided probes.
    :param probes: A dictionary containing probe information, including a "devices" key with a list of device IDs.
    :param device_dir: The directory where the info_devices.json file is located.
    :return: A dictionary mapping device IDs to their respective information.
    """
    device_info = {}
    print(
            "Loading info_devices.json data for devices ",
            list(probes["devices"]),
            end=": ",
        )
    with path_in.open(encoding="utf8") as f:
        device_info_loaded = json.load(f)
    pid_info = None
    for pid_cur in probes["devices"]:
        try:
            pid_info = device_info_loaded[pid_cur]
        except KeyError:
            if not pid_cur or pid_cur[0] == "i":
                continue
            try:
                pid_info = device_info_loaded[f"i{pid_cur}"]
                    # pid_cur = piid if piid[1].isdigit() else piid[1:]
            except KeyError:
                continue
        device_info[pid_cur] = _info_json_item_array_to_dict(*pid_info)
    return device_info


def get_path_in_parents(dir: Path, file_name) -> Path:
    """
    Determine the device directory where the `file_name` file is located searching in parent dirs
    :param dir: starting child directory path
    :param file_name: searching file name
    :return: file_name path
    raises FileNotFoundError if not found
    """
    while True:
        file = dir / file_name
        if file.is_file():
            return file
        dir_parent = dir.parent
        if dir != dir_parent:
            dir = dir_parent
        else:
            raise FileNotFoundError(file_name)


def get_fun_load_end_ext(probe, db, parent=None, time_range=tuple(), time_shift_s=None, data_file_ext=None):
    """
    Loading parameters and fuction: veusz_load_hdf5_ctd_profile, veusz_load_csv_gmx500, veusz_load_csv_ecmwf
    :param probe: controlls selection process
    """
    b_allow_many_sources = False
    if probe["model"] == "GMX500":
        data_file_ext = ".csv"
        fun_load = lambda file: veusz_load_csv_gmx500(file, time_range, db=db)
    elif probe["type"] in ("ECMWF", "CMEMS") and data_file_ext and data_file_ext != ".nc":
        data_file_ext = ".tsv"
        fun_load = lambda file: veusz_load_csv_ecmwf(file, time_range, db=db)
    elif probe["model"] in ("ECMWF", "CMEMS"):
        data_file_ext = ".nc"
        fun_load = lambda file: globals()["veusz_load_hdf5_{model}".format_map(probe).lower()](
            file, time_range, db=db)
        b_allow_many_sources = True
    elif probe["model"] == "Nortek_AquadoppDW":  # and probe['type'] == 'ADV'
        data_file_ext = ".dat"
        fun_load = lambda file: load_adv_sontek(file, time_range, db=db)
    elif probe["type"] == "CTD":  # and probe['type'] == 'ADV'
        data_file_ext = ".txt"
        # re_n_runs = '(?P<n_runs>\d*)(?:run)s?'

        # hdf5 data group name must be equal to dir name of current file?
        time_range_raw = veusz_load_hdf5_ctd_profile(
            db,
            time_range,
            device="{type}_{model}{id}".format_map(probe),
            time_shift_s=time_shift_s,
            n_runs=1,
        )

        probe_data = {k: v.replace("_", " ") for k, v in probe.items()}
        probe_data["id_expr"] = "'{}'.format_map(I)".format(probe_data["id"].replace("#", "{#}"))
        if parent.name == "profiles_vsz":
            fun_load = None
            probe_data['st_expr'] = "'АБП64{}'.format(DATA('_log_fileName_st')[0].split('st')[-1].replace('_', r'\\underline{ }'))"
        else:
            fun_load = (lambda x: [time_range_raw])
            probe_data["st_expr"] = ""


        # AddCustom('import', 'importlib', 'util')
        # AddCustom('import', 'itertools', 'dropwhile')
        AddCustom(
            "definition",
            "DISPtime",
            "[['{:%Y-%m-%d %H:%M}', '{:%Y-%m-%d %H:%M}']]  # graph auto range, UTC".format(
                *np.array(time_range_raw, "M8[ns]").astype("M8[s]").tolist()
            ),
        )
        AddCustom("definition", "f", "lambda fun, *args: fun(*args)")
        AddCustom(
            "definition",
            "argv1",
            "argv[1] if argv[1]!='--embed-remote' else ENVIRON.get('VSZ_PATH', FILENAME())",
        )
        # AddCustom('definition', 'import_file(path, module_name)',
        #     "( lambda spec: (lambda mod: ( spec.loader.exec_module(mod), mod, warning(f'loading {mod}'))[1] )(util.module_from_spec(spec)) )(util.spec_from_file_location(module_name, (lambda fpy: fpy(next(dropwhile(lambda p: not fpy(p).is_file(), path.parents))))(lambda p: (p / module_name).with_suffix('.py'))))")
        # AddCustom('definition', 'v', "import_file(Path(argv1), 'func_vsz')")
        # AddCustom('definition', 'I',
        #     "type('ClassI', (dict,), {'__getitem__': lambda self, key: self.get(key, key)})({n: LANG({'default': n, 'ru': u}) for n, u in v.en2ru.items()})")
        AddCustom(
            "definition",
            "DISPinfo__",
            "{{'id': {id_expr}, 'type': I['{type}'], 'model': '{model}', 'zone': 'UTC', 'st': {st_expr}}}".format_map(probe_data)
        )

        AddCustom(
            "definition",
            "fDisp_date_u(ax, t_span_var, **kwargs)",
            "v.str_date_unit_with_suffix([f(lambda l: l if l!='Auto' else t, SETTING(f'{ax}/{lim:s}')) for lim, t in zip(('min', 'max'), DATA(f'{t_span_var}'))], str_zone=DISPinfo__['zone'], lang=LANG({'default': 'en', 'ru': 'ru'}), **kwargs)",
        )
        """ Old code
                device_model = device.replace('_', ' ')
                if '#' in device_model:
                    device_model, _id = device_model.rsplit('#', 1)
                    _id = f' #{_id}'
                else:
                    _id = ''
                _type, _model = device_model.split(' ', 1)
                AddCustom('definition', 'DISPinfo__',
                    f"{{'id': '{_id}', 'type': I['{_type}'], 'model': '{_model}'}}")
                """
    elif probe["type"] == "i":
        def data_file_ext(probe, name_prefix="", name_suffix=""):
            return f"{name_prefix}@{{type}}_{{model}}{{number:0>2}}{name_suffix}.txt".format_map(probe)
        fun_load = lambda file: veusz_load_csv_tcm_raw(db, time_range, file, time_shift_s, probe)
    else:  # not known file type or inclinometer from db
        fun_load = None
    return fun_load, data_file_ext, b_allow_many_sources


def pcid_from_parts(type: str = "i", model: str = None, number: str|int = None, b_raw=False, **kwargs):
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
    if b_raw and type=="i":
        type = "incl"
    return f"{type}{_}{model}{number:0>2}"


def prepare_draw_tcm(
    probes,
    device_wind,
    time_range,
    time_range_raw,
    device_info,
    device_dir,
    db_stem,
    cus,
    use_bins = {"": 2, "bin_": 600, "bin2_": 3600},
    b_old_format_in_h5: bool = True,
):
    """
    Load processed TCM data to draw with default drawer `vsz_drawer.py`

    Set needed inclinometers IDs for types of devices:
    - cus.USE_bursts: inclinometers working in burst mode
    - ids_i: all inclinometers (burst and continuous),
    - ids_w: wave gauges
    Note: to easier distinguish probe in veusz object names (and replace probe names later) we use in
    probe names "_"-prefix and probe number suffix formatted as 2 digits filled left with 0
    f'_{n}' for n in 'i03,i04,i19,i37,i38,i_p06,i_p05'.split(',')
    :param probes:
    :param time_range:
    :param device_info:
    :param device_dir:
    :param db_stem:
    :param cus:
    :param use_bins: bin average intervals and corresponding data names prefixes.
    defaults to {"": 2, "bin_": 600, "bin2_": 3600} [s].
    You can exclude high resolution data to load/edit faster
    :param b_old_format_in_h5: , defaults to True
    :retrun: tuple:
        ids_i,
        ids_ip,
        ids_p,
        ids_w,
        ids_order,
        device_wind,
        time_range_raw_wind,
        max_time_span_s_strings,
        use_bins,
        use_bins_w,
        bin0name,
        bin_burst_name,
        b_one_table,
    """
    format_model_part = lambda model: "" if model == "i" else f"_{model}"
    if b_old_format_in_h5:
        format_model_part_old = lambda model: "" if model == "i" else model
        ids_i_old = [  # f'_i{i}' for n in ''.split(',') if n '5,19,11,15'
                "_{type}{_model}{number:02d}".format(**probe, _model=format_model_part_old(probe["model"]))
                for pid, probe in probes["devices"].items() if pid
            ]
    ids_i = [  # f'_i{i}' for n in ''.split(',') if n '5,19,11,15'
            "_{type}{_model}{number:02d}".format(**probe, _model=format_model_part(probe["model"]))
            for pid, probe in probes["devices"].items() if pid
        ]
    _ = set(f"_{k}" for k, v in device_info.items() if hasattr(v, "get") and v.get("w"))
    if _:
        print(f"Devices in burst mode found from metadata: {_}")
        cus.add_later(USE_bursts=_ & set(ids_i))  # Set which of ids_i is in burst mode  | {"_i04"}
    else:
        cus.add_later(USE_bursts=set())
        # todo: autoget and use pulse ratio
    ids_w = [pid for pid in ids_i if pid.startswith("_w")]
    if ids_w:
        ids_i = [pid for pid in ids_i if not pid.startswith("_w")]
        # f'_w{int(n):02d}' for n in ''.split(',') if n]

    print("Prepare processed data for inclinometers & wavegauges:", ids_w, ids_i)
    # Load all devices info {id: 'p','b','bd','s'} - p: point, bd, s: device's depth above bottom and type symbol
    print("loading", (device_dir / "info_devices.json").absolute().resolve())
    with open(device_dir / "info_devices.json", encoding="utf8") as j:
        meta_devices_all = json.load(j)

    # Get probes from `info_devices.json` config to draw in that order, after loading from `hdf5`
    ids_order = {}  # temporary dict is for uniqueness: will be converted to list
    ids = ids_i + ids_w
    meta_devices = {}
    for pid, val in meta_devices_all.items():
        if True:  # p.startswith('E'):
            for id_to_load in ids:
                id_to_load1 = id_to_load[1:]
                if pid in id_to_load1:
                    # assigning to any value - it will not be used
                    ids_order[id_to_load] = None

                    # Also adding items for stacked devices in meta_devices by combining existed items (and copy other used items)
                    idls = [s for s in re.split(r"([^\d]+\d+)_", id_to_load1) if s]
                    if len(idls) > 1:
                        # print([meta_devices[f'{id_to_load1[0]}{idl[-2:]}'] for idl in id_to_load1.split('_')])
                        meta_devices[
                                id_to_load1
                            ] = [  # combine str items, use 1st item for numbers
                                ",".join(vv)
                                if isinstance(vv[0], str) and iparam != 0
                                else vv[0]
                                for iparam, vv in enumerate(
                                    zip(
                                        *[
                                            meta_devices_all[
                                                f"{id_to_load[0]}{idl[-2:]}"
                                            ]
                                            for idl in idls
                                        ]
                                    )
                                )
                            ]  # ((p,p),(b,b),(bd,bd),(s,s))
                    else:
                        meta_devices[id_to_load1] = val  # (p,b,bd,s,*lat_lon)

                    # # helps to my translation to russian:
                    # if meta_devices[id_to_load1][0] in ('seaward point', 'shoreward point'):
                    #     meta_devices[id_to_load1][0] = '_{}'.format(meta_devices[id_to_load1][0])

    print("Info loaded from info_devices.json:", meta_devices)
    cus.add(DISPdevices_info=meta_devices, comments="p, b, bd, s, c, t, burst_w")

    ids_ip = [n for n in ids_i if n.startswith("_i_p")]
    # names different from ids_i to combine with
    ids_p = [n.replace("i_p", "p") for n in ids_ip]
    # For inclinometers with pressure sensor put pressure graphs before velocities
    ids_order = [
        i
        for n in ids_order
        for i in ([ids_p[ids_ip.index(n)], n] if n in ids_ip else [n])
    ]
    # ids_order = [i for n in ids_order for i in ([n, ids_p[ids_ip.index(n)]] if n in ids_ip else [n])]
    print("Devices draw order:", ids_order)

    # Loading

    ## Define bin averaged data to load

    # Use raw data sampling frequency if time range is small
    if any(np.isfinite(time_range)) and len(time_range) >= 2 and time_range[-1] is not NaT:
        dtime_range_s = np.diff(time_range).astype("m8[s]").astype(int).item()
        # <=> np.subtract(*time_range[::-1]).astype(int).item()

        # Exclude too big bins for our time range
        use_bins = {
                n: dt
                for n, dt in use_bins.items()
                if dt < dtime_range_s / 2
            }
        if dtime_range_s <= 3600 or not use_bins:  # < 1H  # may be need < 10min
            use_bins = {"": 0}  # , 'bin_': 2  # 0 means raw data sampling frequency
    else:
        dtime_range_s = None
    # todo: make needed averaging in Veusz if not found in db-data

    use_bins_w = use_bins if any(ids_w) or any(ids_p) else {}

    if cus.postponed["USE_bursts"]:
        use_bins["binB_"] = 1800
        if dtime_range_s and use_bins["binB_"] and dtime_range_s < use_bins["binB_"]:
            cus.add_later(USE_bursts = set())
    cus.add_postponed()
    _ = len(use_bins)
    # Define variables containing names of minimum and burst or (else) max bin for incl/wavegauges
    if _ > 1:
        bin0name, bin_burst_name = list(use_bins)[::(_ - 1)]  # 'bin2_'
    else:
        bin0name = bin_burst_name = list(use_bins)[0]

    # - inclinometers

    # load db with one table for all devices if binned data will be needed
    b_one_table = db_stem.endswith('.proc')  # still may load other db with separate table for each device

    b_load_all_data = False
    if b_load_all_data:
        try:
            ImportFileHDF5(
                db,
                [f"/i_bin{dt_s}s/table"],
                linked=True,
                namemap={
                    f"/i_bin{dt_s}s/table/index": "t_ns",
                    **{
                        f"/i_bin{dt_s}s/table/Pressure{idci}": f"P{idc}"
                        for idci, idc in zip(ids_ip, ids_p)
                    },
                    **{
                        f"/i_bin{dt_s}s/table/{prm}{idci}": f"{prm}{idci}"
                        for idci in ids_i
                        for prm in ("u", "v", "Temp")
                    },
                },
                prefix=bin,
            )
        except Exception as e:
            print("Can not load", f"/i_bin{dt_s}s/table", "from", db)
            raise e

        # - wave gauges
        if any(ids_w):
            # use_bins_w = {'': 2, 'bin_': 300, 'bin2_': 3600}
            print("Loading", ids_w, "from", db)
            namemap = {
                f"/w_bin{dt_s}s/table/{v}{'' if v == 'index' else idc}": "".join(
                    bin,
                    'P' if v == 'Pressure' else 't_ns' if v == 'index' else v,
                    '_w' if v == 'index' else idc
                )
                for idc in ids_w
                for v in ("Pressure", "Temp", "index")
                for bin, dt_s in use_bins_w.items()
            }
            ImportFileHDF5(
                db,
                [f"/w_bin{dt_s}s/table" for dt_s in use_bins_w.values()],
                linked=True,
                namemap=namemap,
            )
            print(namemap)

        try:
            TagDatasets(
                "loaded",
                [
                    f"{bin}{param}{pid}"
                    for pid in ids_i
                    for param in ("u", "v", "Temp")
                    for bin in use_bins
                ]  # inclinometers
                + [
                    f"{bin}{param}{pid}"
                    for pid in ids_w
                    for param in (["P"] if pid in ids_p else ("P", "Temp"))
                    for bin in use_bins_w
                ]  # wave gauges
                + [
                    f"{bin}t_ns{sfx_w}"
                    for sfx_w, bins in (("", use_bins), ("_w", use_bins_w))
                    for bin in bins
                    if pid not in ids_p and use_bins[bin]
                ],  # indexes
            )  # todo: add all f't_ns{pid}' when not use_bins[bin]
        except Exception as e:
            print("TagDatasets Error: ", e)

    elif ids:
        params = ["u", "v", "Temp"]
        for is_binning, ub in groupby(use_bins.items(), lambda x: x[1] > 0):
            if is_binning:  # bin > 0
                cols_name_map = {"index": "t_ns"}
                if b_one_table:
                    h5_group_to_bin_prefix = {
                        f"{'i' if ids!=ids_w else 'w'}_bin{dt_s}s": bin for bin, dt_s in ub
                    }
                    top_groups = list(h5_group_to_bin_prefix.keys())

                    # for bin, dt_s in ub.items():  # ?
                    if b_old_format_in_h5:
                        for idci, idci_old in zip(ids_i, ids_i_old):
                            for prm in params:
                                cols_name_map[f"{prm}{idci_old}"] = f"{prm}{idci}"
                    else:
                        for idci in ids_i:
                            for prm in params:
                                cols_name_map[f"{prm}{idci}"] = f"{prm}{idci}"  # need?
                    for id_p, idc in zip(ids_ip, ids_p):
                        cols_name_map[f"Pressure{id_p}"] = f"P{idc}"
                    # if any(ids_w) or any(ids_p):
                    #     # - wave gauges
                else:
                    h5_group_to_bin_prefix = {
                        f"{pid.removeprefix('_')}bin{dt_s}s": (bin, pid)
                        for bin, dt_s in ub
                        for pid in ids_i
                    }
                    top_groups = list(h5_group_to_bin_prefix.keys())
                    cols_name_map.update(
                        {
                            f"{prm}": f"{prm}"
                            for prm in params
                            # if len(probes["devices"]) > 1 else params
                        }
                    )
                    for id_p in ids_p:
                        cols_name_map[f"Pressure{id_p}"] = f"P{id_p}"
                if any(ids_w):
                    top_groups += [f"w_bin{dt_s}s" for bin, dt_s in ub]

                def f_table_cols_fmt(col, device_id):
                    """Add prefixes / suffixes for columns of each grp_d group"""
                    if b_one_table:
                        bin = h5_group_to_bin_prefix[device_id]
                        if device_id in ids_w:
                            if col == "index":  # 't_ns'?
                                prefix = ""
                                sfx = "_w"
                            else:
                                prefix = "w_"
                                sfx = ""
                        else:
                            prefix = ""
                            sfx = ""
                    else:
                        prefix = ""
                        bin, device_id = h5_group_to_bin_prefix[device_id]
                        sfx = device_id

                    return f"{prefix}{bin}{col}{sfx}"
                db_for_bin = db
            else:
                cols_name_map = {"index": "t_ns", **dict(zip(params, params))}
                if ids_p:
                    cols_name_map["Pressure"] = "P"
                top_groups = {pid.removeprefix("_") for pid in ids_i}

                def f_table_cols_fmt(col, device_id):
                    """Add prefixes / suffixes for columns of each grp_d group"""
                    # old: col_out = col.replace("ip", "i_p").replace("ib", "i_b")
                    if col == 'P':
                        device_id = device_id.replace('i_', '')
                    return f"{col}_{device_id}"

                db_for_bin = db.with_name(
                    db.name.replace(
                        ".proc." if len(probes["devices"]) > 1  else ".proc_Avg.",
                        ".proc_noAvg.",
                        1,
                    )
                )
            existed_devs, time_range_raw, i_ranges = veusz_load_hdf5(
                db_for_bin,
                top_groups,
                grp_d={"table": "/table/"},
                cols_namemap={"table": cols_name_map},
                grp_d_rename_funs={"table": f_table_cols_fmt},
                time_range=time_range,
                time_shift_s=cus.USE_timeShift_s,
                decimation=probes.get('decimation')
            )

    # Set not defined `time_range` elements from raw time range
    if not any(np.isfinite(time_range)):
        time_range = time_range_raw
    elif len(time_range_raw) == 2:
        time_range = np.where(np.isnat(time_range), time_range_raw)

    # Dataset that contains wind or useful for wave gauges P_a data
    if device_wind or ids_w or ids_p:
        file_wind = None
        coord_patterns = [
            r"E(?P<lon>\d+.?\d*)[,_]N(?P<lat>\d+.?\d*)",
            r"N(?P<lat>\d+.?\d*)[,_]E(?P<lon>\d+.?\d*)",
            r"(?P<lon>\d+.?\d*)E[,_](?P<lat>\d+.?\d*)N",
            r"(?P<lat>\d+.?\d*)N[,_](?P<lon>\d+.?\d*)E",
        ]
        for suffix in [".nc", ".tsv"]:
            for folder, device_wind, fun_load in (  # device_wind for "info_wind"
                (
                    [("ECMWF", "'ECMWF ERA5'", veusz_load_hdf5_ecmwf)]
                    if device_wind != "CMEMS"
                    else []
                ) +
                (
                    [
                        (
                            "CMEMS",
                            "ESCAPE('CMEMS  WIND_GLO_PHY_L4_NRT_012_004')",
                            veusz_load_hdf5_cmems,
                        )
                    ]
                    if device_wind != "ECMWF"
                    else []
                )
            ):
                _ = cruise_dir / "meteo" / folder

                # Find coordinates from file names to select file nearest to probe
                coords_to_path = {}
                coords = None

                files = list(_.glob(f"*{suffix}"))
                if not files:
                    files = [d for d in _.iterdir() if d.is_dir() and d.name.startswith("area(")]
                for file in files:
                    for coord_pattern in coord_patterns:
                        try:
                            coords = re.search(coord_pattern, file.name).groupdict()
                            coords_to_path[
                                (float(coords["lat"]), float(coords["lon"]))
                            ] = file
                        except AttributeError:
                            continue

                if files:  # Wind file have been found
                    try:
                        t = ids[0][1:]
                    except IndexError:
                        # meta_devices, pid_info?
                        coords_probe = None
                        t = "[not specified probe]"
                    else:
                        coords_probe = device_info[t].get("c")  # Lat,Lon
                    if coords_to_path and coords_probe:
                        # Check the distance to probe

                        coords_wind = list(coords_to_path.keys())
                        (dx, dy, dist_m, bearing) = fv.dx_dy_dist_bearing(
                            *coords_probe[::-1],
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
                            f"Found nearest {device_wind} data point with distance to 1st probe point: "
                            f"{(dist_m[i_coord] if isinstance(dist_m, np.ndarray) else dist_m)/1e3:g}km"
                        )
                    else:
                        file_wind = files[0]
                        if not coords_to_path:
                            print(
                                f"No coords found for {device_wind} data. Selecting 1st file. Consider "
                                f"to include coordinates into file name ({coord_patterns[0]}) to enable "
                                "assess data distance"
                            )
                        if not coords_probe:
                            print(
                                f"No coords found for {t} data. Selecting 1st file."
                                "Consider to include coordinates into device_info.json to enable assess "
                                "data distance"
                            )

                    print(f"Loading wind from {file_wind}...")
                    if suffix == ".nc":
                        time_range_raw_wind = fun_load(
                            file_wind,
                            time_range,
                            time_shift_s=zone_to_seconds_offset(device_info["zone"]),
                        )
                    elif folder == "CMEMS":
                        print("loading *.tcv from CMEMS not implemented")
                        file_wind = None
                        continue
                    break         # to load tcv from ECMWF
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
                device_wind = "'ECMWF ERA5'"  # for "info_wind" Custom Definition
        else:
            print("No wind data found!")

        if use_bins_w:
            bin0name_w = list(use_bins_w)[0]  # name of minimum bin

    # Calculation
    # ###########
    if any(ids_p):
        ids_w += ids_p
    if ids_w:
        bin_max_w = list(use_bins_w)[-1]

    for devs, sfx_w, ub, bin0 in [
        (ids_i, "", use_bins, bin0name),  # inclinometers
        (ids_w, "w", *((use_bins_w, bin0name_w) if ids_w else ([], None))),  # wave gauges
    ]:
        # Pid time suffix needed if loaded data is not of one device per type (w/i) or data is not combined
        b_t_sfx_is_pid = not b_one_table  # not (b_one_table and (bin0 or (ub and ub[bin0])))?
        t_sfx = (devs[0] if devs else None) if b_t_sfx_is_pid else ("_w" if sfx_w else "")
        for i_bin, bin in enumerate(ub):
            if i_bin == 0:
                ## Minimum/no bin
                if t_sfx is not None:
                    SetDataExpression(
                        "dt" if not bin else f"{bin[:-1]}{sfx_w}",
                        f"1E-9*min(diff({bin}t_ns{t_sfx}[:3]))",
                        linked=True,
                    )
                # Intervals for each device
                for i_pid, pid in enumerate(devs):
                    if b_t_sfx_is_pid:
                        t_sfx = pid
                    if sfx_w:
                        _w = "_w"
                        try:
                            ip = ids_p.index(pid)
                        except ValueError:  # pid is not in list
                            SetData2DExpression(
                                f"iUseAuto{pid}",
                                f"[flatnonzero(isfinite({bin0}P{pid}))[[0,-1]]]",
                                linked=True,
                            )
                            SetData2DExpression(
                                f"iu{pid}",
                                f"v.min_range_2d(atleast_2d(v.i_positive(v.i_use(t_ns_w, USEtime{pid}, "
                                "t_shift_s=USE_timeShift_s), t_ns_w.size)), iUseAuto{pid})",
                                linked=True,
                            )
                    else:
                        SetData2DExpression(
                            f"iUseAuto{pid}",
                            f"[flatnonzero(isfinite({bin}u{pid}))[[0,-1]]]",
                            linked=True,
                        )
                        SetData2DExpression(
                            f"iu{pid}",
                            f"v.min_range_2d(atleast_2d(v.i_positive(v.i_use(t_ns{t_sfx}, USEtime{pid}, "
                            f"t_shift_s=USE_timeShift_s), t_ns{t_sfx}.size)), iUseAuto{pid})",
                            linked=True,
                        )
                        # SetData2DExpression(
                        #     f"iu{pid}",
                        #     f"searchsorted({bin}t_ns{t_sfx}, (float64(array(DISPtime, 'datetime64[ns]'))) - "
                        #     f"USE_timeShift_s*1E9) if len(DISPtime)>0 else [[0, {bin}t_ns{sfx_w}.size]]",
                        #     linked=True
                        # )
                        SetDataExpression(
                            f"time_span{pid}",
                            f"around(v.dt64s2vsz(1E-9*{bin}t_ns{t_sfx}[sl_(iu{pid})][[0, -1]] + "
                            "USE_timeShift_s) / 60) * 60",
                            linked=True
                        )
            elif devs:
                ## Binning
                for pid in devs:
                    # - other bins data:
                    t_sfx = "" if b_one_table else pid

                    if sfx_w:
                        try:
                            ip = ids_p.index(pid)
                            _w = ""
                        except ValueError:  # pid is not in list
                            _w = "_w"
                        SetDataExpression(
                            f"mean_P{pid}",
                            f"nanmean({bin_max_w}P{pid}[sl_({bin_max_w}iu{pid if _w else ids_ip[ip]})])",
                            linked=True,
                        )
                    else:
                        _w = ""
                    # Suffix is used if we've loaded not combined bin0name data:
                    t0sfx = pid if b_t_sfx_is_pid else _w

                    SetData2DExpression(
                        f"{bin}iu{pid}",
                        (
                            f"[searchsorted({bin}t_ns{t_sfx}, {bin0}t_ns{t0sfx}[int32(clip("
                            f"iu{pid}[0], 0, {bin0}t_ns{t0sfx}.size - 1))]) + int32([0, -1])]"
                        ),
                        linked=True,
                    )
                    # Mininum interval of all devs
                    SetData2DExpression(
                        f"{bin}iu_cmn{pid}",
                        (
                            f"[searchsorted({bin}t_ns{t_sfx}, "
                            "v.vsz2dt64s(time_span_i_common).astype(int)*1E9) + int32([0, -1])]"
                        ),
                        linked=True,
                    )


                    # Time for individual devices
                    SetDataExpression(
                        f"{bin}t0st{pid}",
                        f"v.dt64s2vsz(1E-9*{bin}t_ns{t_sfx}[sl_({bin}iu{pid})]) + USE_timeShift_s",
                        linked=True,
                    )

                    SetDataExpression(
                        f"{bin}Vabs{pid}",
                        f"absolute({bin}u{pid}+1j*{bin}v{pid})[sl_({bin}iu{pid})]",
                        linked=True,
                    )
                    SetDataExpression(
                        f"{bin}Vdir{pid}",
                        f"v.wrap_dir(degrees(arctan2({bin}u{pid}, {bin}v{pid})[sl_({bin}iu{pid})]), disp_central_dir)",
                        linked=True,
                    )
                SetDataExpression(
                    f"{bin[:-1]}{sfx_w}",
                    f"1E-9*min(diff({bin}t_ns{t_sfx}[:3]))",
                    linked=True,
                )
    SetDataExpression(
        f"time_span_i",
        "(lambda time_st, time_en: [min(time_st), max(time_en)])(*column_stack(({})))".format(
            "".join(f"time_span{pid}, " for pid in ids_i)
        ),
        linked=True,
    )
    SetDataExpression(
        f"time_span_i_common",
        "(lambda time_st, time_en: [max(time_st), min(time_en)])(*column_stack(({})))".format(
            "".join(f"time_span{pid}, " for pid in ids_i)
        ),
        linked=True,
    )
    SetDataExpression(
        f"disp_time_span",
        "v.dt64s2vsz(array(DISPtime[0], 'M8[s]')) if len(DISPtime)>0 else time_span_i",
        linked=True,
    )
    # Burst data
    for pid in cus.USE_bursts:
        # Suffix is needed for bins which corresponding loaded data is not combined
        t0sfx = "" if b_one_table and use_bins[bin0name] else pid
        t_sfx = "" if b_one_table else pid
        SetData2DExpression(
            f"binB_iu{pid}",
            f"[searchsorted(binB_t_ns{t_sfx}, {bin0name}t_ns{t0sfx}[int32("
            f"clip(iu{pid}[0], 0, {bin0name}t_ns{t0sfx}.size - 1))]) + int32([0, -1])]",
            linked=True,
        )
        SetDataExpression(
            f"binB_t0st{pid}",
            f"v.dt64s2vsz(1E-9*binB_t_ns{t_sfx}[sl_(binB_iu{pid})] + USE_timeShift_s)",
            linked=True,
        )
        SetDataExpression(
            f"disp_ones_burst_st{pid}",
            f"ones_like({bin_burst_name}t0st{pid}) if '{pid}' in USE_bursts else []",
            linked=True,
        )

    if device_wind:
        pid = "_Wind"
        SetDataExpression(f"dt{pid}", f"min(diff(time{pid}[1:4]))", linked=True)
        bin = "bin2_"
        SetDataExpression(
            f"{bin}i0st{pid}",
            f"v.i_whole_time_intervals(time{pid}, WIND_bin_average_s)",
            linked=True,
        )
        SetDataExpression(
            f"{bin}t0st{pid}", f"time{pid}[int32({bin}i0st{pid})]", linked=True
        )
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
    SetDataExpression(
        "disp_central_dir",
        f"0  # f(lambda result: result + (-180 if result > 0 else 180), f(lambda angle, n: floor_divide(360,n)*(floor_divide(n*(angle + 180) + 180, 360)%n), mean_bin_Vdir{ids_i[0] if ids_i else 0}, 4))",
        linked=True,
    )

    # Propose default time range for vsz_drawer:
    #  time_range, for time_range > 1D rounded to hours + 1h to edges
    if any(np.isfinite(time_range)):
        t = np.array(time_range, "M8[s]")
        t_float = t.astype(np.float64)
        t += np.array(
            (
                [0 if r == 0 else (a - r) for a, r in zip([-3600, 3600], t_float % 3600)]
                if np.diff(t_float) > 24 * 3600  # time range > np.timedelta64(1, "D")
                else 0
            ),
            "m8[s]",
        )
        max_time_span_s_strings = [
            str(_) if np.isfinite(_) else NaT if isinstance(_, np.datetime64) else _ for _ in t
        ]
    else:
        max_time_span_s_strings = []


    return (
        ids_i,
        ids_ip,
        ids_p,
        ids_w,
        ids_order,
        device_wind,
        time_range_raw_wind,
        max_time_span_s_strings,
        use_bins,
        use_bins_w,
        bin0name,
        bin_burst_name,
        b_one_table,
    )



##############################################################################################################
##############################################################################################################
##############################################################################################################
# Run

if __name__ in ("__main__", "builtins"):
    if __name__ == "__main__":
        import os
        from utils.veuszPropagate import load_vsz_closure
        parent, basename = (lambda p: (p.parent, p.name))(Path(sys.argv[1]))
        ## This should be done in runner to search "drawer" locally or in PYTHONPATH environment var settings
        sys.path[:0] = [str(parent)] + os.environ.get("PYTHONPATH", "").split(os.pathsep)
        load_vsz = load_vsz_closure(
            # cfg['program']['veusz_path'],
            # cfg['program']['load_timeout_s'],
            # cfg['program']['b_execute_vsz'],
            # cfg['program']['hidden']
        )
        vsze, ctd_dict = load_vsz()  # vsz_path, vsze, prefix=vsz_vars_prefixes
        globals().update(vsze.__dict__)

    AddCustom("import", "logging", "warning")  # type: ignore
    AddCustom("import", "pathlib", "Path")  # type: ignore
    AddCustom("import", "sys", "argv")  # type: ignore
    AddCustom("import", "v", "*") # type: ignore
    AddCustom("import", "v", "I") # type: ignore


    # Common info in Custom Definitions
    ###################################

    # 1. `DISPdevice_info` (will be added to Custom Definition from updated `device_info` later)
    # Displaying time zone:
    device_info = {"zone": "UTC"}    # "UTC+2"

    # 2. `USE_timeShift_s`: Time shift to add to loading data
    if False:  # Set True to manual control time shift of source not UTC data to displaying time zone
        # Loading data zone is not UTC
        cus.USE_timeShift_s = 0
    else:
        # Loading data zone is UTC, automatically determine time shift
        try:
            cus.USE_timeShift_s = zone_to_seconds_offset(device_info["zone"])
        except IndexError:
            # no shift to zone was set => draw in same zone as input data (local)
            cus.USE_timeShift_s = 0
        except ValueError:
            raise NotImplementedError(
                'Can not parse time shift to UTC from zone "{zone}"'.format_map(device_info)
            )


    # Default parameters from folder name (get if possible)
    dt_0 = np.timedelta64(0)
    if re.search(r"[=@\(]", parent.name):  # may be additional info in folder
        print("Extracting from folder name", end=" ")

        # Make folder match regex (?P<yy>\d\d)(?P<mm>\d\d)(?P<dd>\d\d)[_T](?P<HH>\d\d)(?P<MM>\d\d)
        # for func. that works with such file names
        fake_in_stem = (
            parent.name.lstrip(".")
            .removeprefix("vsz(")
            .removeprefix("vsz")
            .replace("range", "dt", 1)
            .replace(")", "")
            .replace("__", "_")
        )
        ich_start_device = fake_in_stem.find("@")
        b_dir_have_device = ich_start_device != -1
        b_dir_have_date = fake_in_stem[:6].isdigit()
        time_range_0, probes_0 = get_info_from_filename(
            "{}{}{}.vsz".format(  # Add date to folder name if not present to match regex fields next to date
                "" if b_dir_have_date else np.datetime64("now").item().strftime("%y%m%d"),
                (
                    fake_in_stem[:ich_start_device].replace(",", "_").removesuffix("_")
                    + fake_in_stem[ich_start_device:]
                )
                if b_dir_have_device
                else fake_in_stem,
                "" if b_dir_have_device else "@i0",
            )
        )
        if not b_dir_have_device:
            del probes_0["devices"]
        if all(np.isfinite(time_range_0)):
            dt_0 = np.diff(time_range_0)[0]
            print(f" (duration: {dt_0.item()})")
            if not b_dir_have_date:
                time_range_0 = []
        else:
            print()
    else:
        time_range_0, probes_0 = [], {}


    # Add wind to inclinometers proc.data

    device_wind = None  # will store the replaced part
    def replace_and_capture(match):
        """
        Replace matching groups (starting from 1st) in string argument of re.sub to ""
        and capture 1st not None group to `device_wind`
        """
        global device_wind
        device_wind = next(dropwhile(lambda x: x is None, match.groups()), None)
        return ""  # Replacement string

    # Parameters from file name
    pattern = r"[@,](?:(wind)|(?:(?:wind)?(ECMWF|ERA5|CMEMS)))\b"  # see re.search(pattern, basename).groups()
    basename = re.sub(pattern, replace_and_capture, basename, count=1)

    if len(basename) <= 5:  # ".vsz" or @.vsz: no more info in basename => need draw wind only
        time_range, probes = time_range_0, {"devices": {}}
    else:
        print("Extracting time range, device from file name", end=" ")
        time_range, probes = get_info_from_filename(basename=basename)
        if not any(np.isfinite(time_range)):  # set to value from folder / default
            # warning(f'File name: "{basename}" not matches regex "{re_exp}"')
            time_range = time_range_0  # if any(time_range_0) else [time_start, time_start+np.timedelta64(3600 * 3, 's')]
        else:
            dt = np.diff(time_range)
            if (dt > np.timedelta64(0)).item():  # time_range obtained now
                print(
                    f": [{', '.join(f'{t.item()}' for t in time_range)}] (duration: {dt.item()})"
                )
            elif (dt_0 > np.timedelta64(0)).item():  # start obtained now, duration earlier
                dt = dt_0
                print(
                    f": [{', '.join(f'{t.item()}' for t in time_range)} + duration from folder name]"
                )
                time_range = [time_range[0], time_range[0] + dt]
            else:
                time_range = [time_range[0], NaT]
    probes = {**probes_0, **probes}

    b_use_db_raw = "_raw" in parent.parent.parts
    if "txt" not in parent.name:
        if "txt" in parent.parent.name:
            b_use_db_raw = False
    else:
        b_use_db_raw = False

    # If `_raw` in parent path parts then start search from it, else if `vsz` or `txt` in parent then start
    # search from its parent
    device_dir = parent
    try:
        device_dir = Path(*parent.parts[: parent.parts.index("_raw")])
    except ValueError:
        if "txt" in parent.name or "vsz" in parent.name:
            device_dir = parent.parent
    try:
        device_dir = get_path_in_parents(device_dir, file_name="info_devices.json").parent
        print(f"from device_dir ({device_dir}) as info_devices.json found")
        b_info_devices_json_found = True
    except FileNotFoundError:
        print(f"from device_dir ({device_dir}) as no info_devices.json found:")
        b_info_devices_json_found = False

    cruise_dir = device_dir if device_dir.name[0].isdigit() else device_dir.parent

    # if any(probes["devices"]):
    #    # Probes have been determined from file name or parent dir - add common info to each probe?
    #    for pid, probe in probes["devices"].items():
    #        probe.update(device_info)

    if not any(probes["devices"]):  # probes have not been determined from file name or parent dir
        # One probe name equal to `device_dir` folder name
        # Set its drawing info in probes["devices"]
        probe = re.match(
            r"(?:(?P<type>[^_#@\d\.]+)?)_(?P<model>[\w]+)?([#@](?P<id>[\w]+))?",
            device_dir.name,
        )  # ?
        probe = (
            probe.groupdict()
            if probe
            else {
                "model": device_dir.name,
                "type": "" if device_dir.name[0].isdigit() else device_dir.parent.name,
                "id": "",
            }
        )
        if probe["id"] is None:
            # correct to exact model and id we use to can construct exact table name from type+model+id later
            if probe["model"] == "SST48":
                probe["model"] = "SST_48Mc"
                probe["id"] = "#1253"
            else:
                probe["id"] = ""
        probes["devices"] = {probe["id"]: probe}

    # DB file
    try:
        db_stem = re.match(".*,db_stem=([^,)]+)", parent.name).group(1)
        print("db from dir name: {db_stem}")
    except Exception:
        db_stem = (device_dir if device_dir.name[0].isdigit() else device_dir.parent).name.split("@")[0]

    # get 1st pid & probe
    pid, probe = next(iter(probes["devices"].items()))
    b_device_is_tcm = (
        any(p for p in device_dir.parts if p.startswith("inclinometer"))
        or probe["type"] == "i"
    )
    if b_device_is_tcm:
        # DB stem should be like '220128'
        db_stem = db_stem.split('_', maxsplit=1)[0]
        if b_use_db_raw:
            db_stem = f'{db_stem}.raw'
            db = device_dir / "_raw" / f"{db_stem}.h5"
            b_db_ok = db.is_file()
        elif "_raw" not in parent.parts:
            # Try load best db for our case first
            dbs = []
            for sfx in [".proc", ".proc_Avg"] if len(probes["devices"]) > 1 else [".proc_Avg", ".proc"]:
                _ = list(device_dir.glob(f"*{sfx}.h5"))
                if not any(_):
                    continue
                dbs += _
                _ = [f.name for f in _]
                try:
                    db = dbs[_.index(f"{db_stem}{sfx}.h5")]
                    b_db_ok = True
                    break
                except ValueError:  # exact file name with `db_stem` not in index, but with `sfx` suffix exist
                    continue
            else:
                _ = len(dbs)
                b_db_ok = _ > 0
                if b_db_ok:
                    if _ > 1:
                        dbs_check_1st = [f for f in dbs if f.name.endswith(f"{probe['type']}{sfx}.h5")]
                        if dbs_check_1st:
                            dbs = dbs_check_1st
                            _ = len(dbs)
                        if _ > 1:
                            warning(
                                f"No {db_stem}.proc.h5 or {db_stem}.proc_Avg.h5 found, other variants number "
                                f"with this suffixes > 1 ({_}): {dbs}. Selecting 1st"
                            )
                    db = dbs[0]
                else:
                    db = ".proc.h5 or .proc_Avg.h5"
                # todo: fail back to other db if current db loading will be failed
            if b_db_ok:
                db_stem = db.stem
        else:
            db = None
            b_db_ok = False
    else:
        # DB stem should be equal to the name of parent folder and DB file will be under _raw dir or
        # cruise_dir
        db = (device_dir / "_raw" if b_use_db_raw else cruise_dir) / f"{db_stem}.h5"
        b_db_ok = db.is_file()

    time_range_raw = []  # time range from raw data
    if b_use_db_raw:
        if not b_db_ok:
            raise (FileNotFoundError(f"{db} not found!"))
        print("Raw data DB found:", db)
        existed_devs = {}  # dict with existed data in db (fields [device_id][grp])
        if b_device_is_tcm:
            # Load raw TCM data in Veusz
            existed_devs, time_range_raw, file, cols_namemap, grp_d_rename_funs, f_table_cols_fmt = (
                veusz_load_hdf5_tcm_raw(
                    db,
                    probes["devices"],
                    time_range,
                    time_shift_s=cus.USE_timeShift_s,
                    decimation=probes.get("decimation"),
                )
            )
    else:
        # Parent folder encodes folder where get data `data_dir` relative to current dir `parent` if has "vsz"
        # - by ".." before "vsz" in its name: means the number of parent levels relative to vsz folder
        # - by "vsz({dir})" to point on sibling folder {dir}. "vsz" alone means that data in parent folder
        if (
            "vsz" in parent.name
        ):  # get name with '.' that can be used in dir name to encode data_dir location
            data_dir = parent.parent
            # 1. Check/use "..vsz" parent dir encoding
            _ = parent.name.split("..")
            for p in _:
                if p:  # no more ".."
                    break  # need?
                data_dir = data_dir.parent
                print("<", end="")
            if data_dir.name == "vsz":
                print("<", end="")
                data_dir = data_dir.parent
            else:  # 2. Check/use vsz({dir}) relative dir encoding
                # vsz(single argument without "=" or named argument `dir`)
                m = re.match(r"_?vsz\(([^)=]+)\)", _[-1]) or re.match(r"_?vsz\(dir=([^,)=]+)", _[-1])
                if m:
                    data_dir = data_dir.with_name(m.group(1))  # if m else device_dir
        else:
            data_dir = device_dir

        ## Load known file types into Veusz if any in `probes["devices"]` (except TCM data, which proc. later)
        b_allow_many_sources = False  # unique 1 source file allowed
        for pid, probe in probes["devices"].items():
            fun_load, data_file_ext, b_allow_many_sources = get_fun_load_end_ext(
                probe, db, parent=parent, time_range=time_range, time_shift_s=cus.USE_timeShift_s
            )
            if fun_load:
                # Search data files
                # 1. Try exact match
                if isinstance(data_file_ext, str):
                    data_file = (data_dir / basename).with_suffix(data_file_ext)
                else:
                    data_file = data_dir / data_file_ext(probe)
                if not data_file.is_file():
                    if "@" in basename:
                        # _type should be in dir or in data file name
                        _type = re.match(
                            "([^+&.-]+).*", basename.split("@", 1)[1]
                        ).group(1)
                        _type = (
                            "" if data_dir.name.startswith(_type) else f"@{_type}*"
                        )  # glob pattern part
                    else:  # _type not needed in data file name
                        _type = ""
                    # Data files named as starting part of vsz
                    # (vsz files names can contain extra text after part equal to data file name beginning with +- or &)
                    data_file = ""
                    name_splitted = [basename, ""]
                    while len(name_splitted) > 1:
                        name_prefix = re.match('([^+&.-]+)*', name_splitted[0]).group(1)
                        data_file = list(
                            data_dir.glob(
                                f"{name_prefix}*{_type}{data_file_ext}"
                                if isinstance(data_file_ext, str)
                                else data_file_ext(probe, name_prefix=f"{name_prefix}*")
                            )
                        )
                        if any(data_file):
                            if (not b_allow_many_sources) and len(data_file) > 1:
                                raise FileExistsError(
                                    "Not possible to find unique data from vsz path: found %s files",
                                    data_file,
                                )
                            data_file = data_file[0]
                            break
                        name_splitted = name_splitted[0].rsplit("_", 1)
                    else:
                        # 1 data file with name started from digits and that includes needed type + ext
                        def search_file(data_dir, _type, data_file_ext):
                            # 1 data file with name that includes needed type + ext in priority order
                            data_globs = (
                                f"[0-9][0-9][0-9][0-9][0-9][0-9]*{_type}",  # started from digits
                                f"@{_type}",  # checked/corrected for syntax errors
                                f"{_type}",  # any other
                            )
                            data_files = []
                            if isinstance(data_file_ext, str):
                                for name_prefix in data_globs:
                                    for p in data_dir.glob(f"{name_prefix}{data_file_ext}"):
                                        data_files.append(p)
                            else:  # (probe, name_prefix="", name_suffix="")
                                pure_data_file_name = Path(data_file_ext(probe))
                                name_suffix = "*"  # todo: get exact string form vsz name
                                for name_prefix in data_globs:
                                    for p in data_dir.glob(
                                        f"{name_prefix}{pure_data_file_name.stem}"
                                        f"{name_suffix}{pure_data_file_name.suffix}"
                                    ):
                                        data_files.append(p)
                            return data_files
                        data_file = None
                        for _type in [_type, ""]:
                            data_files = search_file(data_dir, _type, data_file_ext)
                            max_len = 0
                            for _ in data_files:
                                if len(_.stem) > max_len:  # if basename > data_file.stem:  # ?
                                    data_file = _
                                    max_len = len(data_file.stem)
                            if data_file:
                                break
                        else:
                            raise FileNotFoundError(
                                f"No files match {data_dir}/[0-9][0-9][0-9][0-9][0-9][0-9]*{_type}{data_file_ext} found: {data_files}" if data_dir.is_dir()
                                else f"{data_dir} is not a directory!"
                            )
                time_range_raw = fun_load(data_file)[0]


    # Load device info `id: (p d s)` to `DISPdevice_info` from "info_devices.json":
    # p: point
    # d, s: device's depth and type symbol
    if b_info_devices_json_found:
        _ = load_info_json(probes, device_dir / "info_devices.json")
        if not _:
            print('No device info in "info_devices.json" file found')
            b_info_devices_json_found = False
            point_name = ""
        else:
            device_info.update(_)
            cus.DISPdevice_info = device_info
            print(device_info)
    else:
        if b_db_ok:
            print('No "info_devices.json" file')
            try:
                point_name = (
                    re.match(r"[\d_]*([^_@]*)", device_dir.stem).group(1)
                    if b_use_db_raw
                    else re.sub(r"([^_]*)(\w)_(\d)(.*)", r"\1\2\3\4", data_file.stem)
                )
            except NameError:  # if data_file not defined
                point_name = ""
        else:
            point_name = ""

    if not b_info_devices_json_found:
        # Add common info `probes` once (to last used `pid`)
        cus.DISPdevice_info = {
            **device_info,
            "p": point_name,
            "b": None,
            "d": None,
            "s": "",
            **probes["devices"][pid],
            "model": probes["devices"][pid]["model"].replace("_", " "),
        }
    AddCustom("definition", "pid", f"'{pid}'")

    # Load text file with time intervals if exist: with pid suffix (more priority) or without
    logs_txt_prefix = "intervals_selected"
    log_txt = None
    for p in parent.glob(f"{logs_txt_prefix}*.txt"):
        sfx = p.stem[len(logs_txt_prefix) + 1 :]
        if not sfx:
            log_txt = p
            # continue search sfx
        elif pid == sfx or "{type}{model}{number}".format_map(probe) == normalize_device_id(sfx):
            log_txt = p
            break
    if log_txt:
        ImportFileCSV(
            log_txt.name,
            blanksaredata=True,
            delimiter="\t",
            encoding="ascii",
            headermode="1st",
            linked=True,
            dsprefix="area_",
        )


    #%% Prepare TCM drawer or load other device specific drawer
    models = set()  # to define drawers `~drawer@{model}.vsz`
    if b_device_is_tcm and b_db_ok and not b_use_db_raw:
        # Load processed TCM data to draw with default drawer `vsz_drawer.py`
        (
            ids_i,
            ids_ip,
            ids_p,
            ids_w,
            ids_order,
            device_wind,
            time_range_raw_wind,
            max_time_span_s_strings,
            use_bins,
            use_bins_w,
            bin0name,
            bin_burst_name,
            b_one_table
        ) = prepare_draw_tcm(
            probes,
            device_wind,
            time_range,
            time_range_raw,
            device_info,
            device_dir,
            db_stem,
            cus,
            # use_bins = {'': 600, 'bin_': 3600, 'bin2_': 7200}  # 7200  # must be sorted
        )

    else:
        # Gather used probe models. They are corresponds to specific raw drawer file names we will call.
        for pid, probe in probes["devices"].items():
            # i: inclinometer drawer, # p: pressure drawerg
            if probe["model"] != "i" and probe["type"] == "i":
                models.add("i")
            models.add(probe["model"])

        if models:
            models = list(models)
            models.sort()
            if probes["is_type_mod"]:
                models[0]= '-'.join([probe["type"], models[0]])


            # Custom Definitions of time ranges
            if device_wind and len(models) == 1:
                _ = "Wind"
                cus.Wind_timeShift_s = cus.USE_timeShift_s
                # cus.USE_timeShift_s is no more need
            else:
                _ = "_"
            for range_name, t in [("DISPtime", time_range), (f"USEtime_{_}", time_range_raw)]:
                max_time_span_s_strings = (
                    [str(_) if isinstance(_, np.datetime64) else _ for _ in t] if t is not None and len(t) else []
                )
                setattr(cus, range_name, f"[{max_time_span_s_strings}]" if max_time_span_s_strings else [])



    if True:  # __name__ == "__main__":
        if models:
            print(f"Running {__name__} drawer for models {models}...")
            # Execute local drawers for model part parameters
            def next_executable():
                for model in models:  # i: inclinometer drawer, # p: pressure drawer
                    print("Drawer", f'"{model}"')
                    yield f"~drawer@{model}.vsz"

            for _ in next_executable():
                if not _:
                    break
                exec(compile((parent / _).read_text(encoding="utf-8"), _, "exec"))

        else:
            # run default py-drawer (where same Custom Definitions as for models above are specified)

            # Execute drawer (currently defined only for inclinometers)
            globals_ = globals()
            parts = Path(sys.argv[1]).stem.rsplit("@")
            suffix_search_first = [
                f"{'@' if len(parts) > 1 else ''}{parts[-1]}",
                "",
            ]
            print(f"Search vsz_drawer_cfg{{sfx}} in {sys.path} where sfx={suffix_search_first}")
            for suffix in suffix_search_first:
                try:
                    print(
                        f"Trying vsz_drawer_cfg{suffix}",
                        end="... ",
                    )
                    globals_ = run_module(f"vsz_drawer_cfg{suffix}", globals_, run_name=__name__)
                    # for k in globals_.keys():
                    #    print(k)
                    break
                except ImportError:
                    print("not found")
                    pass
            print(f"Running  {__name__}.vsz_drawer", end="... ")
            run_module("vsz_drawer", globals_, run_name=__name__)
