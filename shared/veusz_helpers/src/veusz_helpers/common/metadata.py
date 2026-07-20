from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple, Dict, Set, Optional

import numpy as np
import re

from calendar import monthrange
from datetime import datetime

import func_vsz as fv


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


re_dt = r"(?P<dt>\d+\.?\d*)(?P<dt_u>[YMWDhms])(?:in)?"


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
        # Regex field `is_type_mod` for last (used only if one) device just denotes that user wants the type and model be used together
        """
        return (
            f"(?P<model{i}>[DBPdbp]?)(?P<number{i}>{d}[{d}-]*{d}|{d})"
            if i < max_devices_idx
            else f"(?P<model{i}>[^-][a-zA-Z_-]*)(?P<number{i}>{d}[{d}-]*{d}|{d}*)"
        )

    re_exp = (
        r"(?:"
        r"(?:{time}|(?P<dt_to_last>\d+(?:h|s))_to_last)?"
        r"(?:(?:\.\.|-){time_end})?(?:[_,]? ?(?:dt=)?{dt})?"
        r"(?:[,_]?d(?P<decimation>\d+))?"
        r")?(?:@{pids})?"
        r"(?:[-,_] ?(?P<descr>[^@\d][^@]*))?\.vsz"
    ).format(
        time=re_time,  # end time have same parts but optional and with new names:
        time_end=re_time.replace(">", "e>").replace(")(", ")?(").replace(")[", ")?["),
        dt=re_dt,
        pids="".join(  # some allowed types&models are switches under "Load text file(s) in Veusz" below
            rf"?:{'?,' if i else ''}?((?P<type{i}>i(ncl)?|INKL|w|tr|ADV|ECMWF|CMEMS|)_?"
            f"{re_model_and_number(i)})"
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
                    device_type = device_type_cur.replace("incl", "i", 1).replace("INKL", "i", 1)
            try:
                # model if is not specified: if device type is specified then set to default 'i' else previous
                model = re_parts.pop(f"model{i}")  # or device_type_cur or model or "i"
            except KeyError:
                pass
            pid = f"{model or device_type}{number:{'02d' if device_type in custom_device_types else 'd'}}"
            devices[pid] = {"type": device_type, "model": model, "number": number}
    out_info = {
        "devices": devices
        or {
            device_type_cur: {
                "type": device_type_cur,
                "model": model or device_type_cur,
                "number": number,
            }
        },
        "descr": re_parts["descr"],
        "is_type_mod": ("-" in model),  # re_parts[f"model{max_devices_idx}"] (?P<is_type_mod>-?)
    }
    if re_parts["decimation"]:
        out_info["decimation"] = int(re_parts["decimation"])
    print("-> ", time_range, out_info, end="")
    return time_range, out_info


def load_file_meta(path_in: Path) -> dict:
    with path_in.open(encoding="utf8") as f:
        if path_in.suffix == ".yaml":
            from yaml import safe_load
            content = safe_load(f.read())
            return {
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
        else:
            import json
            return json.load(f)


def select_cfgs(dir_cfgs: Path, incl_type_nums: Set[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Search existed hydra configs of tcm package and filter them by incl_type_nums
    parsing config name is nearly as cfg_name2pcid = lambda name: name.rsplit("_", 2)[0]
    """
    cfgs_existed = {re.match(r"(^.+?\d)_", f.stem).group(1): f.stem for f in (dir_cfgs).glob("*.yaml")}
    #     next(iter(get_info_from_filename(f.stem)[1]["devices"]))["number"]: f.stem
    #     # int(csv_specific_proc.parse_name(f.stem)["number"]): f.stem
    #     for f in dir_cfgs.glob("*.yaml")
    # }
    if not incl_type_nums:
        cfgs = {}
    elif incl_type_nums in ({"*"}, "*"):
        cfgs = cfgs_existed
    else:  # incl_type_nums without config
        missing = incl_type_nums.difference(cfgs_existed)
        if missing:
            raise FileNotFoundError(f"No config files for {len(missing)} probes {missing}!")
        cfgs = {num: stem for num, stem in cfgs_existed.items() if num in incl_type_nums}
    return cfgs_existed, cfgs


def _meta_array_to_dict(
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


def get_path_in_parents(dir: Path, *file_names, target_is_dir=False) -> Path:
    """
    Determine the device directory where the `file_name` file is located searching in parent dirs
    :param dir: starting child directory path
    :param file_names: list of file names to search and return 1st found file
    :return: path of existing file
    raises FileNotFoundError if not found
    """
    while True:
        for file_name in file_names:
            file = dir / file_name
            if file.is_dir() if target_is_dir else file.is_file():
                return file
        dir_parent = dir.parent
        if dir != dir_parent:
            dir = dir_parent
        else:  # root dir
            raise FileNotFoundError(str(file_names))


def extract_devices_info(meta: dict, devices: Sequence[str]) -> dict:
    """
    Load device information from a JSON file and map it to the provided probes.
    :param probes: A dictionary containing probe information, including a "devices" key with a list of device IDs.
    :param device_dir: The directory where the info_devices.json file is located.
    :return: A dictionary mapping device IDs to their respective information.
    """
    device_info = {}
    pid_array = None
    for pid_cur in devices:
        try:
            pid_array = meta[pid_cur]
        except KeyError:
            if not pid_cur or pid_cur[0] == "i":
                continue
            try:
                pid_array = meta[f"i{pid_cur}"]
                    # pid_cur = piid if piid[1].isdigit() else piid[1:]
            except KeyError:
                continue
        device_info[pid_cur] = _meta_array_to_dict(*pid_array)
    return device_info