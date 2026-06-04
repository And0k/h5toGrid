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
from itertools import dropwhile
from functools import partial
import sys
from pathlib import Path

from typing import Any, Callable, Iterable, Mapping, Optional, Tuple, Sequence, Union
from time import strptime
from datetime import datetime
from calendar import monthrange
import numpy as np
import re
import h5py
import importlib.util
import vsz_add_data  # namespace will be updated in runtime by definitions of Veusz functions
import func_vsz as fv


re_dt = r"(?P<dt>\d+\.?\d*)(?P<dt_u>[YMWDhms])(?:in)?"

NaT = np.datetime64("NaT")

cruise_dir = None

wind_mean_uv = None  # todo: collect all statistics into one var

def exec_module_into_globals(name: str, g: dict) -> None:
    """
    Load with globals

    Effect of exec_module_into_globals("vsz_add_data", globals()) equivalent to
    ```
    globals_ = run_module("vsz_add_data", globals(), run_name=__name__)
    ```
        :param name: _description_
        :param g: _description_
        :raises ImportError: _description_
    """
    spec = importlib.util.find_spec(name)
    if spec is None or spec.loader is None:
        raise ImportError(name)

    # minimal module context required for proper imports
    g.setdefault("__name__", name)
    g.setdefault("__package__", spec.parent)
    g.setdefault("__loader__", spec.loader)
    g.setdefault("__spec__", spec)

    code = spec.loader.get_code(name)
    print(f"Running  {__name__}->{name}", end="... ")
    exec(code, g)


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
        r"(?:(?:\.\.|-){time_end})?(?:[_,]?(?:dt=)?{dt})?"
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
                model = re_parts.pop(f"model{i}")  # or device_type_cur or model or "i"
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



def zone_to_seconds_offset(zone: str):
    hours = zone.removeprefix("UTC")
    if not hours:
        return 0
    return strptime(f'{hours[0]}{f"{hours[1:]:>02s}":<04s}', "%z").tm_gmtoff


def load_file_meta(path_in: Path) -> dict:
    with path_in.open(encoding="utf8") as f:
        if path_in.suffix == ".yaml":
            from yaml import safe_load
            content = safe_load(f.read())
            return {
                device_id: [
                    # Nested dict structure with station_id keys -> convert to single list compatible with single interval device
                    # todo: use dicts as from _meta_array_to_dict() instead of arrays instantly, to not loss data as in code below:
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
            return json.load(f)


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
        else:
            raise FileNotFoundError(str(file_names))


def get_fun_load_end_ext(probe, db, parent=None, time_range=tuple(), time_shift_s=None, data_file_ext=None):
    """
    Loading parameters and function: veusz_load_hdf5_ctd_profile, veusz_load_csv_gmx500, veusz_load_csv_ecmwf
    :param probe: controls selection process
    """
    b_allow_many_sources = False
    if probe["model"] == "GMX500":
        data_file_ext = ".csv"
        fun_load = partial(vsz_add_data.veusz_load_csv_gmx500, time_range=time_range, db=db)
    elif probe["type"] in ("ECMWF", "CMEMS") and data_file_ext and data_file_ext != ".nc":
        data_file_ext = ".tsv"
        fun_load = partial(vsz_add_data.veusz_load_csv_ecmwf, time_range=time_range, db=db)
    elif probe["model"] in ("ECMWF", "CMEMS"):
        data_file_ext = ".nc"
        fun_load = partial(  # or globals()["veusz_load_hdf5_{model}".format_map(probe).lower()]
            getattr(sys.modules[__name__], "veusz_load_hdf5_{model}".format_map(probe).lower()),
            time_range=time_range,
            db=db,
        )
        b_allow_many_sources = True
    elif probe["model"] == "Nortek_AquadoppDW":  # and probe['type'] == 'ADV'
        data_file_ext = ".dat"
        fun_load = partial(vsz_add_data.load_adv_sontek, time_range=time_range, db=db)
    elif probe["type"] == "CTD":  # and probe['type'] == 'ADV'
        data_file_ext = ".txt"
        # re_n_runs = '(?P<n_runs>\d*)(?:run)s?'

        # hdf5 data group name must be equal to dir name of current file?
        time_range_raw = vsz_add_data.veusz_load_hdf5_ctd_profile(
            db,
            time_range,
            device="{type}_{model}{id}".format_map(probe),
            time_shift_s=time_shift_s,
            n_runs=1,
        )
        # Set dumb loading function (data already loaded) and prepare DISPinfo__ metadata
        probe_data = {k: v.replace("_", " ") for k, v in probe.items()}
        probe_data["id_expr"] = "'{}'.format_map(I)".format(probe_data["id"].replace("#", "{#}"))
        if parent.name == "profiles_vsz":
            fun_load = None
            # todo: extract info from folder name
            try:
                cruise = re.match(
                    r"(?P<year>\d\d)\d+_*(?P<vessel>\D+)(?P<num>\d+)", cruise_dir.stem
                ).groupdict()
            except:
                cruise = {"vessel": "", "num": ""}
            probe_data["st_expr"] = "".join([
                "'{}{}'.format(",
                "LANG({{'ru': '{}', 'default': '{}'}}), ".format(
                    fv.translit_en_ru(cruise["vessel"]), cruise["vessel"]
                ),
                "(lambda st: st if len(st) > 4 and st.startswith('{num}') else f'{num}{{st}}')".format_map(
                    cruise
                ),
                "(DATA('_log_fileName_st')[0].split('/')[-1].split('st')[-1]"
                ").replace('_', r'\\underline{ }'))",
            ])
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
        fun_load = partial(
            vsz_add_data.veusz_load_csv_tcm_raw,
            db=db,
            time_range=time_range,
            time_shift_s=time_shift_s,
            probe=probe,
        )
    else:  # not known file type or inclinometer from db
        fun_load = None
    return fun_load, data_file_ext, b_allow_many_sources


device_wind = None

def remove_meteo_device(basename):
    """Replace wind device part and store it the global variable"""

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
    return basename


# max_time_span_s_strings = [
#     np.datetime_as_string(_) if np.isfinite(_) else NaT if isinstance(_, np.datetime64) else _
#     for _ in t
# ]

##############################################################################################################
# Run
##############################################################################################################

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

        #  exec_module_into_globals("vsz_add_data", globals())
    vsz_add_data.AddCustom = AddCustom
    vsz_add_data.SetData = SetData
    vsz_add_data.SetDataExpression = SetDataExpression
    vsz_add_data.SetData2DExpression = SetData2DExpression
    vsz_add_data.ImportFileCSV = ImportFileCSV
    vsz_add_data.ImportFileHDF5 = ImportFileHDF5
    vsz_add_data.TagDatasets = TagDatasets

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

    basename = remove_meteo_device(basename)  # Assigns wind device to global variable for further processing
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

    # If `_raw` in parent path parts then start search from it, else if `vsz` or `txt` in parent then start
    # search from its parent
    device_dir = parent
    try:
        device_dir = Path(*parent.parts[: parent.parts.index("_raw")])
    except ValueError:
        while "txt" in device_dir.name or "vsz" in device_dir.name:
            device_dir = device_dir.parent
    files_meta_possible = ["info_devices.yaml", "info_devices.json"]
    try:
        file_meta = get_path_in_parents(device_dir, *files_meta_possible)
        device_dir = file_meta.parent
        print(f"from device_dir ({device_dir}) as {file_meta.name} found")
    except FileNotFoundError:
        print(f"from device_dir ({device_dir}) as no {files_meta_possible} found:")
        file_meta = None

    # dir where search this device and other devices/meteo dirs
    cruise_dir = (
        device_dir.parent
        if not device_dir.name[0].isdigit()
        else device_dir.parent.parent
        if device_dir.parent.name.startswith(
            ("inclinometer", "CTD", "meteo")  # and so on (add all used devices to exclude)
        )
        else device_dir
    )

    # if any(probes["devices"]):
    #    # Probes have been determined from file name or parent dir - add common info to each probe?
    #    for pid, probe in probes["devices"].items():
    #        probe.update(device_info)

    if not any(probes["devices"]):  # Probes have not been determined from file name or parent dir
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

    # get 1st pid & probe
    pid, probe = next(iter(probes["devices"].items()))

    #################################################################
    # Check whether default DB file (*.h5) exist to load data from it
    #################################################################
    try:  # DB specified in dir name?
        db_stem = re.match(".*,db_stem=([^,)]+)", parent.name).group(1)
        print(f"DB from dir name: {db_stem}")
    except Exception:
        db_stem = (device_dir if device_dir.name[0].isdigit() else device_dir.parent).name.split("@")[0]

    # Does it must be raw DB? (search "*.raw.h5")
    b_use_db_raw = "_raw" in parent.parent.parts
    if "txt" not in parent.name:
        if "txt" in parent.parent.name:
            b_use_db_raw = False
    else:
        b_use_db_raw = False

    b_device_is_tcm = (
        any(p for p in device_dir.parts if p.startswith("inclinometer"))
        or probe["type"] == "i"
    )
    if b_device_is_tcm:
        # DB stem should not contain "_" (but dir can)
        db_stem = db_stem.split('_', maxsplit=1)[0]
        if b_use_db_raw:
            db_stem = f'{db_stem}.raw'
            db = device_dir / "_raw" / f"{db_stem}.h5"
            b_db_ok = db.is_file()
            if not b_db_ok:
                raise (FileNotFoundError(f"{db} not found!"))
            # Load raw TCM data in Veusz
            existed_devs, time_range_raw, file, cols_namemap, grp_d_rename_funs, f_table_cols_fmt = (
                vsz_add_data.veusz_load_hdf5_tcm_raw(
                    db,
                    probes["devices"],
                    time_range,
                    time_shift_s=cus.USE_timeShift_s,
                    decimation=probes.get("decimation"),
                )
            )
        elif "_raw" not in parent.parts:  #?
            # Whether we need raw data sampling frequency? (i.e. set use_bins = {"": 0} to use .proc_noAvg.h5)
            if any(np.isfinite(time_range)) and len(time_range) >= 2 and time_range[-1] is not NaT:
                dtime_range_s = np.diff(time_range).astype("m8[s]").astype(int).item()
                use_bins = {"": 0} if dtime_range_s <= 3600 else None  # < 1H  # may be need < 10min
            else:
                use_bins = None  # will be set to default in prepare_draw_tcm()

            # Search in best DB order for our case
            dbs = []
            for sfx in (
                [".proc_noAvg"]
                if use_bins
                else [".proc", ".proc_Avg"]
                if len(probes["devices"]) > 1
                else [".proc_Avg", ".proc"]
            ):
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
        # device_dir
        if parent.name == "profiles_vsz":
            # Special folder name to draw profiles in device dir taking data from common cruise DB
            db = cruise_dir / f"{db_stem}.h5"
        else:
            db = device_dir / ("_raw" if b_use_db_raw else "") / f"{db_stem}.h5"

        b_db_ok = db.is_file()
        if b_db_ok:
            print("Raw data DB found:", db)
            existed_devs = {}  # dict with existed data in db (fields [device_id][grp])
            time_range_raw = []  # time range from raw data
        elif not device_wind:
            raise (FileNotFoundError(f"{db} not found!"))
        # else we will search for DBs specilly for device_wind

    if not b_use_db_raw:  # still need to load data
        # Parent folder encodes folder where get data `data_dir` relative to current dir `parent` if has "vsz"
        # - by ".." before "vsz" in its name: means the number of parent levels relative to vsz folder
        # - by "vsz({dir})" to point on sibling folder {dir}. "vsz" alone means that data in parent folder
        if "vsz" in parent.name:
            # 1. Check/use "..vsz" parent dir encoding: '..' used to encode data_dir location
            _ = parent.name.rsplit("..", 1)[-1]
            m = parent.name.count("..")
            data_dir = parent.parents[m]
            m = re.match(r"_?vsz\(([^)=]+)\)", _[-1]) or re.match(r"_?vsz\(dir=([^,)=]+)", _[-1])
            if m:
                data_dir = data_dir.with_name(m.group(1))  # if m else device_dir
        else:
            data_dir = device_dir

        ## Load known file types into Veusz if any in `probes["devices"]` (except TCM data, which proc. later)

        b_allow_many_sources = False  # unique 1 source file allowed
        for pid, probe in probes["devices"].items():
            # [] if b_device_is_tcm and b_db_ok and not b_use_db_raw else?
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
                            if not device_wind:
                                raise FileNotFoundError(
                                    f"No files match {data_dir}/[0-9][0-9][0-9][0-9][0-9][0-9]*{_type}"
                                    f"{data_file_ext} found: {data_files}"
                                    if data_dir.is_dir()
                                    else f"{data_dir} is not a directory!"
                                )
                            else:
                                break  # we will search for DBs specilly for device_wind
                time_range_raw, *stats = fun_load(data_file)
                if stats:
                    wind_mean_uv = stats[0]  # check me


    # Load device info `id: (p d s)` to `DISPdevice_info` from "info_devices.json":
    # p: point
    # d, s: device's depth and type symbol
    if file_meta:
        print(
            f"Loading {file_meta.name} data for devices ",
            list(probes["devices"]),
            end=": ",
        )
        meta_arrays = load_file_meta(file_meta)
        _ = extract_devices_info(meta_arrays, probes["devices"])
        if _:
            device_info.update(_)
            cus.DISPdevice_info = device_info
            print(device_info)
        else:
            print(f'No device info in "{file_meta.name}" file found')
            file_meta = None
            point_name = ""
    else:
        meta_arrays = None
        if b_db_ok:
            print(f'No "{file_meta.name}" file')
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

    if not file_meta:
        # Single (from last used `pid` from `probes`) device info with placeholders
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

        exec_module_into_globals("vsz_draw_init_tcm", globals())

        # from vsz_draw_init_tcm import vsz_draw_init_tcm
        vsz_draw_init_tcm.vsz_add_data = vsz_add_data  # works
        # vsz_draw_init_tcm.SetDataExpression = SetDataExpression  # not works
        # vsz_draw_init_tcm.SetData = SetData

        (
            ids_i,
            ids_ip,
            ids_p,
            ids_w,
            ids_order,
            disp_time_range,
            use_bins,
            use_bins_w,
            bin0name,
            bin_burst_name,
            b_one_table
        ) = vsz_draw_init_tcm(
            probes,
            meta_arrays,
            db,
            time_range,
            cus,
            use_bins=use_bins,  # {'': 600, 'bin_': 3600, 'bin2_': 7200}  # 7200  # sorted
            # b_old_format_in_h5 = True
        )

        # Load data that contains wind or useful for wave gauges P_a data
        if device_wind or ids_w or ids_p:
            try:
                t = (ids_w + ids_p)[0][1:]
            except IndexError:
                # meta_devices, pid_info?
                coords_probe = None
                t = "[not specified probe]"
            else:
                coords_probe = device_info[t].get("c")  # Lat, Lon
            device_wind, time_range_raw_wind, wind_mean_uv = (
                vsz_add_data.veusz_load_meteo(
                    time_range,
                    coords_probe,
                    zone_to_seconds_offset(device_info["zone"]),
                    dev_wind=device_wind,
                    dir_parent=cruise_dir / "meteo",
                    msg_for=ids_w + ids_p,
                )
            )
    else:
        # Gather used probe models. They are corresponds to specific raw drawer file names we will call.
        for pid, probe in probes["devices"].items():
            # i: inclinometer drawer, # p: pressure drawer
            m = probe["model"]
            if m != "i" and probe["type"] == "i":
                models.add("i")
            models.add(m or probe["type"])

        if models:
            models = list(models)
            models.sort()
            if probes["is_type_mod"]:
                models[0]= '-'.join([probe["type"], models[0]])


            # Custom Definitions of time ranges
            if device_wind and len(models) == 1:
                device_wind, time_range_raw, wind_mean_uv = vsz_add_data.veusz_load_meteo(
                    time_range,
                    None,
                    zone_to_seconds_offset(device_info["zone"]),
                    dev_wind=device_wind,
                    dir_parent=parent.parent,
                    folder_name=parent.name,
                    msg_for="",
                )

                _ = "Wind"
                cus.Wind_timeShift_s = cus.USE_timeShift_s
                # cus.USE_timeShift_s is no more need
            else:
                _ = "_"
            for range_name, t in [("DISPtime", time_range), (f"USEtime_{_}", time_range_raw)]:
                # max_time_span_s_strings = (
                #     [str(_) if isinstance(_, np.datetime64) else _ for _ in t]
                #     if t is not None and len(t)
                #     else []
                # )
                setattr(
                    cus,
                    range_name,
                    f"[{np.datetime_as_string(t).tolist()}]" if t is not None else [],
                )


    # Draw loaded data
    if True:  # __name__ == "__main__":
        if models:
            # Run pid drawers instead of model drawer if exist
            for pid, probe in probes["devices"].items():
                _ = parent / f"~drawer@{pid}.vsz"
                if _.is_file():
                    print(f"Running {__name__} drawer specific to current pid {pid}...")
                    m = probe["model"] or probe["type"]
                    models.remove(m)
                    exec(compile(_.read_text(encoding="utf-8"), _.name, "exec"))

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
            # globals_ = globals()
            parts = Path(sys.argv[1]).stem.rsplit("@")
            suffix_search_first = [
                f"{'@' if len(parts) > 1 else ''}{parts[-1]}",
                "",
            ]
            print(f"Trying run vsz_drawer_cfg{{sfx}} in {sys.path} where sfx={suffix_search_first}", end=" ")
            for suffix in suffix_search_first:
                try:
                    exec_module_into_globals(f"vsz_drawer_cfg{suffix}", globals())
                    # globals_ = run_module(f"vsz_drawer_cfg{suffix}", globals_, run_name=__name__)

                    # for k in globals_.keys():
                    #    print(k)
                    break
                except ImportError:
                    print(f"vsz_drawer_cfg{suffix} - not found", end="... ")
                    pass
            exec_module_into_globals("vsz_drawer", globals())
            # run_module("vsz_drawer", globals_, run_name=__name__)
