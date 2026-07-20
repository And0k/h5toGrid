"""
Format-specific CSV post-load processors — pure pandas/numpy, no dask.

Extracted from ``_dask_legacy.csv_specific_proc`` to make raw CSV loading
available on Layer 0 (the default environment).  All functions here are
self-contained: they depend only on numpy/pandas/re and on
``tcm.utils2init`` / ``tcm.utils_time_corr`` — never on ``dask.dataframe``.

Public API (re-exported by :mod:`tcm.csv_load`):
- :func:`loaded_tcm`  — TCM inclinometer date/time + magnetometer inversion
- :func:`loaded_corr` — oneliner ``csv_specific_param`` corrections
- :func:`correct_txt`  — raw-file regex replacement → corrected file
- :func:`mod_name`     — filename normalization (translit → latin, @-prefix)
"""
from __future__ import annotations

import io
import re
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path, PurePath
from typing import (
    Any,
    AnyStr,
    BinaryIO,
    Callable,
    Dict,
    Iterable,
    Mapping,
    Match,
    Optional,
    Sequence,
    TextIO,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import pandas as pd

from tcm.utils2init import (
    LoggingStyleAdapter,
    dir_create_if_need,
    set_field_if_no,
    standard_error_info,
    FakeContextIfOpen,
)
from tcm.utils_time_corr import save_time_corr_diagnostics

lf = LoggingStyleAdapter(__name__)

# Shared constant — century prefix for 2-digit year bytes (mirrors tcm.format.century)
century = b"20"


# =========================================================================== #
# Low-level helpers
# =========================================================================== #


def chars_array_to_datetimeindex(
    date: Union[np.ndarray, pd.Series], dtype: np.dtype, format: str = "%Y-%m-%dT%H:%M:%S"
) -> pd.DatetimeIndex:
    """Error-corrected conversion of byte/string date arrays to DatetimeIndex.

    Bad or unparseable entries are forward-filled with the previous good value.
    """
    try:
        is_series = isinstance(date, pd.Series)
        if isinstance(date.iat[0] if is_series else date[0], bytes):
            date = (date.str.decode if is_series else (lambda **kw: np.char.decode(date, **kw)))(
                encoding="utf-8", errors="replace"
            )
    except Exception:
        pass

    try:
        date = date.astype(dtype)
    except TypeError as e:
        print(f"Date strings converting to {dtype} error: ", standard_error_info(e))
    except ValueError as e:
        print("Bad date: ", standard_error_info(e))

    try:
        date = pd.to_datetime(date, format=format, errors="coerce")
        t_is_bad = date.isna()
        if (t_is_bad_sum := t_is_bad.sum()):
            lf.warning("replacing {:d} bad strings with previous", t_is_bad_sum)
            date.ffill(inplace=True)
        if date.dtype != dtype:
            date = date.astype(dtype)
    except Exception as e:
        print("to_datetime not works", standard_error_info(e))
        raise
    return pd.DatetimeIndex(date)


def fill0(arr: np.ndarray, width: int) -> np.ndarray:
    """Right-justify byte array with ``b'0'``."""
    return np.char.rjust(arr.astype(f"|S{width}"), width, fillchar=b"0")


def out_fields(
    types: Mapping[str, type],
    keys_del: Optional[Iterable[str]] = (),
    add_before: Optional[Mapping[str, type]] = None,
    add_after: Optional[Mapping[str, type]] = None,
) -> Dict[str, type]:
    """Remove ``keys_del`` from ``types`` and add ``add_before`` / ``add_after``."""
    if add_before is None:
        add_before = {}
    if add_after is None:
        add_after = {}
    return {**add_before, **{k: v for k, v in types if k not in keys_del}, **add_after}


def log_csv_specific_param_operation(
    key_logged: str, functions_str: Sequence[str], cfg_in
) -> None:
    """Log csv_specific_param operations once per cfg_in session."""
    key_logged_full = f"csv_specific_param_logged{'-' if key_logged else ''}{key_logged}"
    if not cfg_in.get(key_logged_full):
        cfg_in[key_logged_full] = True
        lf.info(f"csv_specific_param {list(functions_str)} modifications applied")


def param_funs_closure(
    csv_specific_param: Mapping[str, Union[Callable[[str], Any], float]],
    cfg_in: Mapping[str, Any],
) -> Mapping[str, Callable[[str], Any]]:
    """Convert ``csv_specific_param`` dict to per-column assignment functions.

    Suffix ``_fun`` → apply callable; suffix ``_add`` → add constant.
    Used by :func:`loaded_corr`.
    """
    params_funs: dict[str, Callable] = {}
    for k, fun_or_const in csv_specific_param.items():
        param, fun_id = k.rsplit("_", 1)
        if fun_id == "fun":

            def fun(prm, f):
                param_closure = prm
                v_closure = f
                params_closure = f.__code__.co_varnames

                if len(params_closure) <= 1:
                    def fun_closure(x):
                        return v_closure(x[param_closure])
                else:
                    params_closure = list(params_closure)
                    def fun_closure(x):
                        return v_closure(*x[params_closure].T.values)
                return fun_closure

        elif fun_id == "add":

            def fun(prm, const):  # noqa: F811
                param_closure = prm
                v_closure = const
                def fun_closure(x):
                    return x[param_closure] + v_closure
                return fun_closure
        else:
            continue
        params_funs[param] = fun(param, fun_or_const)
    if params_funs:
        log_csv_specific_param_operation("", params_funs.keys(), cfg_in)
    return params_funs


# =========================================================================== #
# day_jumps_correction — correct day-boundary jumps in corrupted time series
# =========================================================================== #


def day_jumps_correction(
    cfg_in: Mapping[str, Any],
    t: Union[np.ndarray, pd.DatetimeIndex],
    path_save_image: str = "day_jumps_corr",
):
    """Correct day jumps (up/down) caused by unsynchronised date+time sources."""
    dT_day_jump = np.timedelta64(1, "D")
    set_field_if_no(cfg_in, "time_last", t[0])

    dT = np.diff(
        np.insert(
            t if isinstance(t, np.ndarray) else t.to_numpy(),
            0,
            0 if cfg_in["time_last"] is pd.NaT else cfg_in["time_last"],
        )
    )
    dT_resolution = max(np.timedelta64(1, "s"), np.median(dT)) * 10

    def _b_day_jump(dT_arr, dT_day=dT_day_jump, dT_res=dT_resolution, *, up: bool):
        return np.logical_and(
            (dT_day - dT_res < dT_arr) if up else (-dT_day + dT_res > dT_arr),
            (dT_arr <= dT_day + dT_res) if up else (dT_arr >= -dT_day - dT_res),
        )

    jumpU = np.flatnonzero(_b_day_jump(dT, up=True))
    jumpD = np.flatnonzero(_b_day_jump(dT, up=False))

    lU, lD = len(jumpU), len(jumpD)
    if lU or lD:
        jumps = np.hstack((jumpU, jumpD))
        ijumps = np.argsort(jumps)
        jumps = np.append(jumps[ijumps], len(t))
        bjumpU = np.append(np.ones(lU, np.bool_), np.zeros(lD, np.bool_))[ijumps]
        t_orig = t
        for bjU, jSt, jEn in zip(bjumpU[::2], jumps[:-1:2], jumps[1::2]):
            t_datetime = (
                datetime.fromtimestamp(t[jSt].astype(datetime) * 1e-9, timezone.utc)
                if isinstance(t, np.ndarray) else t[jSt]
            )
            if bjU:
                t[jSt:jEn] -= dT_day_jump
                print(
                    "Date correction to {:%d.%m.%y}UTC: day jumps up was detected in [{}:{}] rows".format(
                        t_datetime, jSt, jEn
                    )
                )
            else:
                t[jSt:jEn] += dT_day_jump
                print(
                    "Date correction to {:%d.%m.%y}UTC: day jumps down was detected in [{}:{}] rows".format(
                        t_datetime, jSt, jEn
                    )
                )
        if t_orig is not None and path_save_image:
            try:
                tim_out = pd.to_datetime(t, unit="ns", utc=True)
                save_time_corr_diagnostics(t_orig, tim_out, np.ones_like(t, np.bool_), cfg_in, "day_jumps_corr")
            except Exception:
                lf.debug("failed to save diagnostics", exc_info=True)
    return t


# =========================================================================== #
# concat_to_iso8601 — byte-string date/time columns → ISO 8601 series
# =========================================================================== #


def concat_to_iso8601(a: pd.DataFrame) -> pd.Series:
    """Concatenate ``yyyy,mm,dd,HH,MM,SS`` byte-string columns → ISO 8601 series."""
    d = a["yyyy"].str.decode("utf-8")
    d = d.str.cat([a[c].str.decode("utf-8").str.zfill(2) for c in ["mm", "dd"]], sep="-")
    t = a["HH"].str.decode("utf-8").str.zfill(2)
    t = t.str.cat([a[c].str.decode("utf-8").str.zfill(2) for c in ["MM", "SS"]], sep=":")
    return d.str.cat(t, sep="T")


# =========================================================================== #
# Public API — loaded_tcm
# =========================================================================== #


# =========================================================================== #
# Public API — loaded_tcm
# =========================================================================== #


def loaded_tcm(
    a: pd.DataFrame,
    cfg_in: Mapping[str, Any] = None,
    csv_specific_param: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    """TCM inclinometer post-load processing.

    - Converts ``yyyy..SS`` byte columns → ``datetime64[ns]`` ``Time``.
    - Applies ``time_shift`` bad-time correction (linear or affine).
    - Inverts magnetometer channels if ``invert_magnetometer`` is set.
    """
    try:
        date = concat_to_iso8601(a)
    except Exception as e:
        lf.exception("Can not convert date in {}", a)
        raise e

    tim_index = chars_array_to_datetimeindex(date, "datetime64[ns]")

    # time_shift correction
    if csv_specific_param is not None:
        key = "time_shift"
        try:
            cfg_dt = csv_specific_param[key]
        except KeyError:
            cfg_dt = None
        if cfg_dt:
            dt0 = pd.Timedelta(cfg_dt.get("dt0", "0s"))
            time_raw_st = cfg_in.get("time_raw_st")
            if not time_raw_st:
                time_raw_st = tim_index[0]
                cfg_in["time_raw_st"] = time_raw_st

            time_st = time_raw_st + dt0
            time_en = pd.Timestamp(cfg_dt["time_en"])
            dt_full = (time_en - time_st).as_unit("ns")

            linear_len = cfg_dt.get("linear_len")
            if linear_len:
                dt = np.arange(cfg_in["n_rows"], cfg_in["n_rows"] + len(tim_index)) * (dt_full / linear_len)
                tim_index = pd.DatetimeIndex(time_st + dt)
                lf.info(
                    "{} applied: replacing time by series of {:g} Hz from {:%y-%m-%d %H:%M:%S} + {}",
                    key, linear_len / dt_full.total_seconds(), time_st, dt0 or "0s",
                )
            else:
                dt_end = pd.Timedelta(cfg_dt.get("dt_end", pd.NaT))
                if pd.isna(dt_end):
                    time_raw_en = pd.Timestamp(cfg_dt["time_raw_en"])
                    try:
                        dt_end = time_en - time_raw_en
                    except KeyError:
                        lf.info(
                            "Can not apply {} to data: (time_en and time_raw_en) or dt_end must be specified. "
                            "time_raw_st={:%y-%m-%d %H:%M:%S}, dt0={}, linear_len={}",
                            key, time_raw_st, dt0 or "0s", linear_len,
                        )
                else:
                    time_raw_en = time_en - dt_end
                    dt_raw_full = (time_raw_en - time_raw_st).as_unit("ns")

                dt_end_ns = dt_end.as_unit("ns")
                dt = (tim_index - time_raw_st) * (dt_end_ns / dt_raw_full)
                tim_index += dt0 + dt
                lf.info(
                    "{} applied. Parameters: time_st={:%y-%m-%d %H:%M:%S} + {}, dt_end={}",
                    key, time_raw_st, dt0 or "0s", dt_end or "0s",
                )

        try:
            msg_time_rng = "{:%y-%m-%d %H:%M} – {:%m-%d %H:%M} ".format(*tim_index[[0, -1]].to_pydatetime())
        except KeyError:
            lf.warning("bad edge time")
            msg_time_rng = ""
        lf.info("Time {}({} values) converted", msg_time_rng, tim_index.size)

        # invert_magnetometer
        key = "invert_magnetometer"
        try:
            invert_flag = csv_specific_param[key]
        except KeyError:
            invert_flag = False
        if invert_flag and "Mx" in a.columns:
            magnetometer_channels = ["Mx", "My", "Mz"]
            lf.debug("'{}' applied", key)
            a.loc[:, magnetometer_channels] = -a.loc[:, magnetometer_channels].values
            a = a.copy()
        elif invert_flag:
            lf.debug("'{}' skipped — no magnetometer columns", key)

    return a.assign(Time=tim_index)


# same but no magnetometer columns → no inversion
loaded_wavegauge = loaded_tcm


# =========================================================================== #
# Public API — loaded_corr
# =========================================================================== #


def loaded_corr(
    a: Union[pd.DataFrame, np.ndarray],
    cfg_in: Mapping[str, Any],
    csv_specific_param: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    """Apply oneliner ``csv_specific_param`` corrections (``_fun`` / ``_add`` suffixes)."""
    if csv_specific_param is not None:
        params_funs = param_funs_closure(csv_specific_param, cfg_in)
        if params_funs:
            return a.assign(**params_funs)
    return a


# =========================================================================== #
# f_repl_by_dict + rep_in_file — regex-based file correction helpers
# =========================================================================== #


def f_repl_by_dict(
    replist: Iterable[AnyStr], binary_str: bool = True
) -> Callable[[Match[AnyStr]], AnyStr]:
    """Build a regex-substitution function from a list of alternative patterns.

    The returned ``fsub(line)`` keeps the last named capture group and
    deletes everything else.  Unmatched lines become empty.
    """
    regex = re.compile((b"|" if binary_str else "|").join(x for x in replist))

    def replfunc(match):
        if match.lastgroup:
            return match.group(match.lastgroup)
        return b"" if binary_str else ""

    def fsub(line):
        return regex.sub(replfunc, line)

    return fsub


def rep_in_file(
    file_in: Union[str, Path, BinaryIO, TextIO],
    file_out,
    f_replace: Union[Callable[[bytes], bytes], Callable[[str], str]],
    header_rows=0,
    block_size=None,
    min_out_length=2,
    f_replace_in_header: Optional[Callable[[bytes], bytes]] = None,
    binary_mode=True,
) -> int:
    """Replace text in file via *f_replace*, keeping *header_rows* intact."""
    try:
        file_in_path = Path(file_in)
        if not file_in_path.is_file():
            print(f"{file_in_path} not found")
            return None
    except TypeError:
        file_in_path = Path(file_in.name)

    lf.warning(
        "preliminary correcting csv file {:s} by removing irregular rows, writing to {:s}.",
        file_in_path.name,
        Path(file_out).name,
    )
    file_out = Path(file_out)
    if file_in_path == file_out:
        file_out, file_out_original = file_out.with_suffix(f"{file_out.suffix}.bak"), file_out
    else:
        file_out_original = ""

    sum_deleted = 0
    with (
        FakeContextIfOpen(
            lambda x: open(x, **({"mode": "rb"} if binary_mode else {"mode": "r", "errors": "ignore"})),
            file_in,
        ) as fin,
        open(file_out, "wb" if binary_mode else "w") as fout,
    ):
        if isinstance(header_rows, range):
            for row in range(header_rows[0]):
                fin.readline()
            for row in header_rows:
                fout.write(fin.readline())
        else:
            for row in range(header_rows):
                fout.write(fin.readline())

        if block_size is None:
            for line in fin:
                newline = f_replace(line)
                if len(newline) > min_out_length:
                    fout.write(newline)
                else:
                    sum_deleted += 1
        else:
            the_end = b"" if binary_mode else ""
            for block in iter(lambda: fin.read(block_size), the_end):
                block = f_replace(block)
                if block != the_end:
                    fout.write(block)

    if file_out_original:
        file_out.replace(file_out_original)
    return sum_deleted


# =========================================================================== #
# Public API — correct_txt
# =========================================================================== #


def correct_txt(
    file_in: Union[str, Path, BinaryIO, TextIO],
    file_out: Optional[Path] = None,
    dir_out: Optional[PurePath] = None,
    mod_file_name: Callable[[PurePath], PurePath] = lambda n: Path(
        n.name.replace(".", "_clean.")
    ),
    sub_str_list: Sequence[bytes] = None,
    **kwargs,
) -> Path:
    """Replace bad strings in a CSV file and write the corrected version.

    Uses :func:`f_repl_by_dict` to build a regex filter from *sub_str_list*,
    then streams through :func:`rep_in_file`.
    """
    is_opened = isinstance(file_in, (io.TextIOBase, io.RawIOBase))
    msg_file_in = file_in

    if file_out:
        pass
    elif dir_out:
        msg_file_in = (Path(file_in) if isinstance(file_in, str) else file_in).name
        name_maybe_with_sub_dir = mod_file_name(Path(msg_file_in))
        file_out = dir_out / Path(name_maybe_with_sub_dir).name
    else:
        if is_opened:
            inf = getattr(file_in, "_inf", None)
            if inf:
                file_in_path = Path(inf.volume_file)
                file_out = mod_file_name(file_in_path.parent / file_in.name)
                file_in_path /= file_in.name
            else:
                file_in_path = Path(file_in.name)
                file_out = mod_file_name(file_in_path)
            msg_file_in = file_in_path.name
        elif not file_out:
            file_in_path = Path(file_in)
            file_out = file_in_path.with_name(str(mod_file_name(Path(file_in_path.name))))

    out_dir = file_out.parent
    out_dir = out_dir.with_name(out_dir.name.replace(".", "-"))
    dir_create_if_need(out_dir)
    file_out = out_dir / file_out.name

    if file_out.is_file() and file_out.stat().st_size > 100:
        if is_opened:
            from tcm._dask_legacy.csv_specific_proc import correct_old_zip_name_ru
            msg_file_in = correct_old_zip_name_ru(msg_file_in)
        lf.warning(f"skipping of pre-correcting csv file {msg_file_in} to {file_out.name}: destination exist")
        return file_out

    binary_mode = isinstance(file_in, io.RawIOBase)
    if sub_str_list:
        fsub = f_repl_by_dict(
            [x if binary_mode else bytes.decode(x) for x in sub_str_list],
            binary_str=binary_mode,
        )
        sum_deleted = rep_in_file(file_in, file_out, fsub, **{"binary_mode": binary_mode, **kwargs})
        if sum_deleted:
            lf.warning("{} bad lines deleted", sum_deleted)
    else:
        lf.warning(f"skipping of pre-correcting csv file {msg_file_in} to {file_out.name}: just extracting to output dir")
        block_size = 1000000
        the_end = b"" if binary_mode else ""
        with open(file_out, "w") as fout:
            for block in iter(lambda: file_in.read(block_size), the_end):
                if block != the_end:
                    fout.write(block)

    return file_out


# =========================================================================== #
# Public API — mod_name
# =========================================================================== #


def _parse_name(name: str):
    """Extract logical parts of inclinometer / wave gauge name from source raw csv file name.

    All chars before 1st ``i``/``w``/``*``/``[`` are ignored.
    ``*`` and ``[`` (if present) indicate a glob pattern (regex 2).
    The ``comment`` group captures any trailing suffix after the probe number,
    to be preserved as ``-{comment}`` in the corrected filename
    (see :func:`mod_name`).
    """
    name = name.lower()

    # 1. Regular file stem — captures full suffix as 'comment'
    m = re.match(
        r"[^iw]*(?P<type>[iw])(?P<chars1>((?:nkl|ncl|))_?)"
        r"(?P<model>[bdp]|[0bdp]{1,4}]|)(?P<chars2>_?0*)(?P<number>\d{1,4})(?P<comment>.*)",
        name,
    )
    if m:
        m = m.groupdict()
        m["chars0"] = ""
        return m

    # 2. Glob of file stems — chars3 for glob reconstruction (no comment)
    m = re.match(
        r"[^iw\*\[]*(?P<chars0>\*?\[?)(?P<type>[iw])(?P<chars1>((?:nkl|ncl|))_?\[?)"
        r"(?P<model>[bdpw]{0,4})(?P<chars2>\]?\*?0*)(?P<number>\d{0,4})(?P<chars3>\D*)",
        name,
    )
    if m:
        return m.groupdict()

    # 3. Unusual i/w (e.g. voln_v)
    m = re.match(
        r"@?(?P<type>voln_v)(?P<chars2>\D*0*)(?P<number>\d\d)(?P<comment>.*)", name
    )
    if m:
        m = m.groupdict()
        m["chars0"] = m["chars1"] = ""
        m["type"] = "w"
        m["model"] = ""
        return m

    return None


def mod_name(
    file_in: Union[str, PurePath], add_prefix: str = "", parse: bool = True
) -> Tuple[str, Path]:
    """Normalize inclinometer filename → ``{add_prefix}{pcid}-{comment}.{ext}``.

    Extracts ``(type, model, number, comment)`` probe identity via :func:`_parse_name`,
    then reconstructs the canonical name ``{type}_{model}{number}-{comment}``.
    Leading zeros in the probe number are stripped (legacy: ``i_056`` → ``i_56``).
    Trailing suffix after the number is preserved as a ``-`` delimited comment
    (e.g. ``INKL_P05_0_v_trube`` → ``@i_p5-0_v_trube``).

    :param parse: if ``False``, skip name parsing — just prepend *add_prefix*.
    :param file_in: full path / name / glob of the source raw csv file.
    :param add_prefix: prefix to insert before the name (e.g. ``"@"``).
    :return: ``(model, file_out)`` where *model* is ``'i'``/``'b'``/``'p'``/… or ``None``.
    """
    file_in = PurePath(file_in)
    name = file_in.stem.lower().replace("inkl", "incl")
    if parse:
        b_pattern = ("*" in name or "?" in name)
        if (m := _parse_name(name)):
            if not (model := m["model"]):
                model = m["type"]
            if b_pattern:
                # Glob pattern: reconstruct with original chars (no comment)
                name = "{chars0}{type}{chars1}{model}{chars2}{number}{chars3}".format_map(m)
            else:
                # Regular stem: pcid + optional -comment
                name = "{type}_{model}{number}".format_map(m)
                if (comment := m.get("comment", "").lstrip("_-")):
                    name = f"{name}-{comment}"
            if not b_pattern and not m["number"]:
                print(f"Bad probe name {file_in}: probe number not detected")
        else:
            model = None
            if not b_pattern:
                print(f"Not known probe name: {file_in}")

        if add_prefix:
            def prefix_target(matchobj):
                return f"{add_prefix}{matchobj.group(0)}"
            name = re.sub(r"^\*?([^*.]*)", prefix_target, name)
    else:
        model = None
        name = f"{add_prefix}{name.lstrip(add_prefix)}"

    file_out = file_in.with_name(name).with_suffix(file_in.suffix)
    return model, file_out
