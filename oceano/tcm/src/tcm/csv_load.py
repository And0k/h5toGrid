import io
import re
import sys
from collections import defaultdict, deque, namedtuple
from datetime import datetime, timezone
from functools import update_wrapper
from itertools import chain, dropwhile, groupby, islice
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd

# My
from . import format, utils_time_corr
from .utils2init import ExitStatus, LoggingStyleAdapter, set_field_if_no, update_cfg_time_ranges

lf = LoggingStyleAdapter(__name__)

from tcm.format_loaded import (  # noqa: E402
    correct_txt,
    loaded_corr,
    loaded_tcm,
    mod_name,
)


def _compress_times(times: Iterable[Any], limit: int = 20, fmt="%Y-%m-%d %H:%M:%S") -> str:
    """RLE consecutive duplicates, cap output groups, suffix tz once."""
    _NS = 1_000_000_000
    ts_iter = (datetime.fromtimestamp(int(t) // _NS, tz=timezone.utc) for t in times)
    try:
        first = next(ts_iter)
    except StopIteration:
        return ""

    def _fmt(v):
        return v.strftime(fmt)  # v.replace(tzinfo=None).isoformat(**args_isoformat)

    parts = []
    for i, (s, g) in enumerate(groupby(chain([first], ts_iter), key=_fmt), start=1):
        if i == limit:
            total = f"… (first {limit} distinct)"
            break
        parts.append(f"{s}(×{n})" if (n := sum(1 for _ in g)) > 1 else s)
    else:
        total = f"(total {i} distinct)"

    rle = ", ".join(parts)
    show = f"{rle} [{tz}] {total}" if (tz := first.tzinfo) else f"{rle} {total}"
    return show


def _glob_to_regex(s: str) -> str:
    """Convert glob pattern to regex: * → .*?, ? → ., escape literal dots.

    Literal dots in the input pattern (e.g. file extension) are escaped to ``\\.``.
    ``?`` is replaced before ``*`` to avoid clobbering the ``?`` in ``.*?``.
    """
    # Escape dots that are literal in the input (e.g. file extension separator)
    s = s.replace('.', r'\.')
    s = re.sub(r'\?', '.', s)       # ? → . first
    s = re.sub(r'\*', '.*?', s)     # then * → .*?
    return s


# Default regex when input.path is a directory — matches inclinometer files (see _pattern_to_regex)
_DIR_DEFAULT_REGEX = r'i.*\.txt'


def _pattern_to_regex(name: str) -> str:
    """Interpret *name* as glob or regex, return a compiled regex string.

    Rules (see docs/tcm_clc/how_it_works.md → "Path pattern interpretation"):

    - **Glob** if either: (a) *name* is not a valid regex, or (b) it is a
      valid regex but the dot before the file extension is **not** escaped
      (no ``\\`` immediately before that dot).
    - **Regex** otherwise (valid regex + extension dot escaped).

    Glob conversion delegates to :func:`_glob_to_regex`.  The extension dot
    is the **last** dot in the name that is followed by a non-empty suffix
    containing no dots — e.g. in ``file?.txt`` the dot before ``txt``.
    """
    # Try compiling as regex
    try:
        re.compile(name)
    except re.error:
        return _glob_to_regex(name)  # Not valid regex → glob

    # Valid regex — check if the extension dot is escaped
    # Find last dot followed by extension (no further dots)
    dot_pos = name.rfind('.')
    if dot_pos > 0 and name[dot_pos - 1] != '\\':
        # Extension dot unescaped → treat as glob
        return _glob_to_regex(name)

    # Valid regex with escaped extension dot → use as-is
    return name


def init_input_cols(cfg_in: Optional[MutableMapping[str, Any]]=None):
    """
    Append/modify `cfg_in` for parameters of dask / pandas `load_csv()` and `numpy.loadtxt()` functions
    :param cfg_in: dictionary, may has fields:
    - header (required if no 'cols') - comma/space separated string. Column names in source file data header
    (as in Veusz standard input dialog), used to find cfg_in['cols'] if last is not cpecified
    - dtype - numpy type of data in column (as in `loadtxt()`)
    - converters - dict (see "converters" in `loadtxt()`) or function(cfg_in) to make dict here
    - cols_load - list of used column names
a list from the header by splitting it and removing format specifiers.
    :return: modified cfg_in dictionary. Will have fields:
    - cols: a list from the header by splitting it and removing format specifiers: '(text)', '(float)', '(time)'
    - cols_load: list[int], indexes of ``cols`` in needed to save order
    - coltime/coldate: assigned to index of 'Time'/'Date' column
    - dtype: numpy.dtype of data after using loading function but before filtering/calculating fields
            numpy.float64 - default and for '(float)' format specifier
            numpy string with length cfg_in['max_text_width'] - for '(text)'
            datetime64[ns] - for coldate column (or coltime if no coldate) and for '(time)'
    - col_index_name: index name for saving Pandas frame. Will be set to name of cfg_in['coltime'] column
        if not exist already used in main() default time postload proc only (if no specific loader which
        calculates and returns time column for index) cols_loaded_save_b - columns mask of cols_load to save
        (some columns needed only before save to calulate of others). Default: excluded (text) columns and
        index and coldate (because index saved in other variable and coldate may only used to create it)

    Example
    -------
    header= u'`Ensemble #`,txtYY_M_D_h_m_s_f(text),,,Top,`Average Heading (degrees)`,`Average Pitch (degrees)`,stdPitch,`Average Roll (degrees)`,stdRoll,`Average Temp (degrees C)`,txtu_none(text) txtv_none(text) txtVup(text) txtErrVhor(text) txtInt1(text) txtInt2(text) txtInt3(text) txtInt4(text) txtCor1(text) txtCor2(text) txtCor3(text) txtCor4(text),,,SpeedE_BT SpeedN_BT SpeedUp ErrSpeed DepthReading `Bin Size (m)` `Bin 1 Distance(m;>0=up;<0=down)` absorption IntScale'.strip()
    """

    if cfg_in is None:
        cfg_in = dict()
    set_field_if_no(cfg_in, 'max_text_width', 2000)

    dtype_text_max = '|S{:.0f}'.format(cfg_in['max_text_width'])  # '2000 #np.str

    if cfg_in.get('header'):  # if header specified
        re_sep = ' *(?:(?:\n)|[\n, ]) *'  # process ",," right
        cfg_in['cols'] = re.split(re_sep, cfg_in['header'])
        # re_fast = re.compile(u"(?:[ \n,]+[ \n]*|^)(`[^`]+`|[^`,\n ]*)", re.VERBOSE)
        # cfg_in['cols']= re_fast.findall(cfg_in['header'])
    elif 'cols' not in cfg_in:  # cols is from header, is specified or is default
        raise KeyError('Neither "cols" nor "header" specified in config.in — cannot determine column layout')

    # Default parameters dependent from ['cols']
    cols_load_b = np.ones(len(cfg_in['cols']), np.bool_)
    set_field_if_no(cfg_in, 'comments', '"')

    # assign data type of input columns
    b_was_no_dtype = 'dtype' not in cfg_in
    if b_was_no_dtype:
        cfg_in['dtype'] = np.array([np.float64] * len(cfg_in['cols']))
        # 32 gets truncation errors after 6th sign (=> shows long numbers after dot)
    elif isinstance(cfg_in['dtype'], str):
        cfg_in['dtype'] = np.array([np.dtype(cfg_in['dtype'])] * len(cfg_in['cols']))
    elif isinstance(cfg_in['dtype'], list):
        # prevent numpy array(list) guess minimal dtype because dtype of dtype_text_max may be greater
        numpy_cur_dtype = np.min_scalar_type(cfg_in['dtype'])
        numpy_cur_dtype_len = numpy_cur_dtype.itemsize / np.dtype((numpy_cur_dtype.kind, 1)).itemsize
        cfg_in['dtype'] = np.array(cfg_in['dtype'], '|S{:.0f}'.format(
            max(len(dtype_text_max), numpy_cur_dtype_len)))

    for col, col_name in (['coltime', 'Time'], ['coldate', 'Date']):
        if col not in cfg_in:
            # if cfg['col(time/date)'] is not provided try find 'Time'/'Date' column name
            if col_name not in cfg_in['cols']:
                col_name = col_name + '(text)'
            if col_name not in cfg_in['cols']:
                continue
            cfg_in[col] = cfg_in['cols'].index(col_name)  # 'Time'/'Date' csv column index
        elif isinstance(cfg_in[col], str):
            cfg_in[col] = cfg_in['cols'].index(cfg_in[col])

    if 'converters' not in cfg_in:
        cfg_in['converters'] = None
    elif cfg_in['converters']:
        if not isinstance(cfg_in['converters'], dict):
            # suspended evaluation required
            cfg_in['converters'] = cfg_in['converters'](cfg_in)
        if b_was_no_dtype:
            # converters produce datetime64[ns] for coldate column (or coltime if no coldate):
            cfg_in['dtype'][cfg_in['coldate' if 'coldate' in cfg_in else 'coltime']] = 'datetime64[ns]'

    # process format specifiers: '(text)','(float)','(time)' and remove it from ['cols'],
    # also find not used cols specified by skipping name between commas like in 'col1,,,col4'
    for i, s in enumerate(cfg_in['cols']):
        if len(s) == 0:
            cols_load_b[i] = 0
            cfg_in['cols'][i] = f'NotUsed{i}'
        else:
            b_i_not_in_converters = (i not in cfg_in['converters'].keys()) \
                if cfg_in['converters'] else True
            i_suffix = s.rfind('(text)')
            if i_suffix > 0:  # text
                cfg_in['cols'][i] = s[:i_suffix]
                if (cfg_in['dtype'][
                        i] == np.float64) and b_i_not_in_converters:  # reassign from default float64 to text
                    cfg_in['dtype'][i] = dtype_text_max
            else:
                i_suffix = s.rfind('(float)')
                if i_suffix > 0:  # float
                    cfg_in['cols'][i] = s[:i_suffix]
                    if b_i_not_in_converters:
                        # assign to default. Already done?
                        assert cfg_in['dtype'][i] == np.float64
                else:
                    i_suffix = s.rfind('(time)')
                    if i_suffix > 0:
                        cfg_in['cols'][i] = s[:i_suffix]
                        if (cfg_in['dtype'][i] == np.float64) and b_i_not_in_converters:
                            cfg_in['dtype'][i] = 'datetime64[ns]'  # np.str

    if any(cfg_in.get('cols_load', [])):
        cols_load_b &= np.isin(cfg_in['cols'], cfg_in['cols_load'])
    else:
        cfg_in['cols_load'] = np.array(cfg_in['cols'])[cols_load_b]

    col_names_out = cfg_in['cols_load'].copy()
    # Convert ``cols_load`` to index (to be compatible both with readcsv() and numpy loadtxt())
    cfg_in["cols_load"] = np.int32([
        cfg_in["cols"].index(c) for c in cfg_in["cols_load"] if c in cfg_in["cols"]
    ])
    # not_cols_load = np.array([n in cfg_in['cols_not_save'] for n in cfg_in['cols']], np.bool_)
    # cfg_in['cols_load']= np.logical_and(~not_cols_load, cfg_in['cols_load'])
    # cfg_in['cols']= np.array(cfg_in['cols'])[cfg_in['cols_load']]
    # cfg_in['dtype']=  cfg_in['dtype'][cfg_in['cols_load']]
    # cfg_in['cols_load']= np.flatnonzero(cfg_in['cols_load'])
    # cfg_in['dtype']= np.dtype({'names': cfg_in['cols'].tolist(), 'formats': cfg_in['dtype'].tolist()})

    cfg_in['cols'] = np.array(cfg_in['cols'])
    cfg_in["dtype_raw"] = np.dtype({"names": cfg_in["cols"], "formats": cfg_in["dtype"].tolist()})
    cfg_in['dtype'] = np.dtype({
        'names': cfg_in['cols'][cfg_in['cols_load']],
        'formats': cfg_in['dtype'][cfg_in['cols_load']].tolist()
        })

    # Get index name for saving Pandas frame
    b_index_exist = cfg_in.get('coltime') is not None
    if b_index_exist:
        set_field_if_no(cfg_in, 'col_index_name', cfg_in['cols'][cfg_in['coltime']])

    # Mask of only needed output columns

    # Output columns mask
    if 'cols_loaded_save_b' in cfg_in:  # list to array
        cfg_in['cols_loaded_save_b'] = np.bool_(cfg_in['cols_loaded_save_b'])
    else:
        cfg_in["cols_loaded_save_b"] = np.logical_not(
            np.array([cfg_in["dtype"].fields[n][0].char == "S" for n in cfg_in["dtype"].names])
        )  # a.dtype will = cfg_in['dtype']

        if 'coldate' in cfg_in:
            cfg_in['cols_loaded_save_b'][
                cfg_in['dtype'].names.index(
                    cfg_in['cols'][cfg_in['coldate']])] = False

    # Exclude index from cols_loaded_save_b
    if b_index_exist and cfg_in['col_index_name']:
        cfg_in['cols_loaded_save_b'][cfg_in['dtype'].names.index(
            cfg_in['col_index_name'])] = False  # (must index be used separately?)

    if 'cols_not_save' in cfg_in:
        b_cols_load_in_used = np.isin(
            cfg_in['dtype'].names, cfg_in['cols_not_save'], invert=True)
        if not np.all(b_cols_load_in_used):
            cfg_in['cols_loaded_save_b'] &= b_cols_load_in_used

    # Output columns dtype
    col_names_out = np.array(col_names_out)[cfg_in["cols_loaded_save_b"]].tolist() + cfg_in.get(
        "cols_save", []
    )
    cfg_in["dtype_out"] = np.dtype({
        "formats": [
            cfg_in["dtype"].fields[n][0] if n in cfg_in["dtype"].names else np.dtype(np.float64)
            for n in col_names_out
        ],
        "names": col_names_out,
    })

    return cfg_in


def csv_process(
    df: pd.DataFrame, cfg_in: Mapping[str, Any], t_prev=None, *, first_last_row: bool = False
) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """
    Execute `cfg_in['fun_proc_loaded']` on DataFrame and filter its `Time` column
    prepended with `t_prev` by `time_cor()`
    :param df: DataFrame
    :param cfg_in: corr_time_mode
    :param t_prev: will be prepended to df.Time before time filtering and removed after
    :param first_last_row: edge-row mode (config generation) — skip _correct_time
    :return: new (df, t_prev):
    - df: input df after filtering and concatenation
    - t_prev: last part of filtered df.Time that can be used to prepend next call df.Time
    """
    n_overlap = 2 * int(np.ceil(cfg_in['fs'])) if cfg_in.get('fs') else 50
    b_overlap = t_prev is not None
    # Convert df columns (at least get date)
    df_cnv = cfg_in['fun_proc_loaded'](df, cfg_in)
    try:
        date = df_cnv.Time  # fun_proc_loaded() returned modified DataFrame
        df = df_cnv
        lf.debug("converted csv columns with time correction")
    except AttributeError:  # fun_proc_loaded() returned only Time column
        lf.debug("time correction (Time column only)")
        date = df_cnv

    if b_overlap:
        date = pd.concat([t_prev, date])
    time_cor, b_ok = utils_time_corr.time_corr(date, cfg_in, process=not first_last_row)  # may be long
    if b_overlap:
        _ = t_prev.shape[0]
        time_cor = time_cor[_:]
        b_ok = b_ok[_:]

    if nbad_time := b_ok.size - b_ok.sum().item():
        bad_times = time_cor[~(b_ok | np.isnat(time_cor))]
        lf.info(
            "Values ({:d}) outside range: {}",
            nbad_time,
            _compress_times(bad_times, limit=10)
        )


    t_prev = df.loc[df.index[-n_overlap:], 'Time']
    # Removing rows with bad time
    df_filt = df.loc[b_ok, list(cfg_in['dtype_out'].names)]
    time_filt = time_cor[b_ok]
    n_dup = time_filt.size - np.unique(time_filt).size
    if n_dup:
        raise ValueError(
            f"csv_process: {n_dup} duplicate time(s) after b_ok filter. "
            f"First dup at index {np.flatnonzero(np.diff(time_filt.view(np.int64)) == 0)[0]}"
        )
    df_filt = df_filt.set_index(time_filt).rename_axis('Time')

    if False and __debug__:
        len_out = len(time_filt)
        print('out data length:', len_out)
        # print('index_limits:', df_time_ok.divisions[0::df_time_ok.npartitions])
        sum_na_out, df_time_ok_time_min, df_time_ok_time_max = (
            time_filt.notnull().sum(), time_filt.min(), time_filt.max())
        print('out data len, nontna, min, max:', sum_na_out, df_time_ok_time_min, df_time_ok_time_max)

    return df_filt, t_prev


def range_message(range_source, n_ok_time, n_all_rows):
    t = range_source() if isinstance(range_source, Callable) else range_source
    lf.info(
        "loaded source range: {:s}",
        f"{t['min']:%Y-%m-%d %H:%M:%S} - {t['max']:%Y-%m-%d %H:%M:%S %Z}, {n_ok_time:d}/{n_all_rows:d} rows"
    )


def extract_csv_edges(
    path: Path, mult: int = 5, skiprows: Optional[int] = 0, chunksize: Optional[int] = None, **read_csv_args
) -> pd.DataFrame:
    """Extract the first and last non-blank rows of a delimited file as a <=2-row frame.

    Assumes >=2 non-blank lines fall within `mult * len(first line)` bytes of EOF, and that
    b"\\n" is a safe, unambiguous line terminator for `encoding` (ASCII/UTF-8/Latin-1; not
    UTF-16/32). `buf` never contains a header line, so `read_csv_args` must label columns without one
    (`header=None` + `names=`), matching how the rest of the pipeline already reads this file.

    Args:
        skiprows: leading lines to discard before the header search (clears read_csv_args as we'll skip rows other way).
	chunksize: mirrors read_csv, intercepted here to clear read_csv_args, since they'd corrupt a 1-2 line parse
        mult: byte-lookback multiplier for the tail scan, in units of the first line's length;
            also caps how many leading blank lines the header search tolerates before giving up.
        **read_csv_args: forwarded to `pd.read_csv` (index_col=False unless overridden).

    Returns:
        2-row frame (1 row if first == last), or an empty frame if no non-blank line is found.
    """

    read_csv_args.setdefault("index_col", False)
    enc = read_csv_args.get("encoding", "utf-8")

    with path.open("rb") as f:
        deque(islice(f, skiprows), maxlen=0)  # consume+discard `skiprows` lines, EAFP-safe past EOF

        # Head: first non-blank line within `mult` lines of the (post-skip) start
        first = next(dropwhile(lambda ln: not ln.strip(), islice(f, mult)), b"")

        # Tail: seek back `mult` first-line-lengths from EOF, scan backward for last non-blank;
        # reversed() naturally skips the (near-certainly truncated) leading fragment of the window
        size = f.seek(0, io.SEEK_END)
        f.seek(max(0, size - (first_size:=len(first)) * mult))
        last = next(dropwhile(lambda ln: len(ln) < first_size/2, reversed(f.readlines())), b"")

    lf.debug(
        "{}: size={:d}B skip={:d} mult={:d} first={:d}B last={:d}B",
        path.name,
        size,
        skiprows,
        mult,
        first_size,
        len(last),
    )

    buf = first if first == last else b"\n".join([first, last])   # guard duplicating a lone row
    if not buf:
        return pd.DataFrame()
    try:
        return pd.read_csv(io.StringIO(buf.decode(enc)), **read_csv_args)
    except Exception:
        lf.exception("{}: failed decoding/parsing extremes (skip={:d} mult={:d})", path.name, skiprows, mult)
        raise



def csv_read_gen(
    files: Sequence[Union[str, Path]], first_last_row: bool = False, **cfg_in: Mapping[str, Any]
) -> Iterator[Tuple[int, Path, Optional[pd.DataFrame]]]:
    """
    Reads csv to pandas DataFrame in chunks
    Calls `cfg_in['fun_proc_loaded']()` (if specified)
    Calls `time_corr()`: corrects/checks Time (with arguments defined in cfg_in fields)
    Sets Time as output dataframe index
    :param files: list of file names for a single probe
    :param first_last_row: returned `df` will have only edge csv rows and other rows will not be read from csv
    :param cfg_in: contains fields for arguments of `pandas.read_csv()` correspondence:
    - names=cfg_in['cols'][cfg_in['cols_load']]
    - usecols=cfg_in['cols_load']
    - on_bad_lines=cfg_in['on_bad_lines']
    - comment=cfg_in['comment']
    - chunksize=cfg_in['blocksize']
    Other arguments corresponds to fields with same name:
    - dtype=cfg_in['dtype']
    - delimiter=cfg_in['delimiter']
    - converters=cfg_in['converters']
    - skiprows=cfg_in['skiprows']

    Also cfg_in has fields:
    - dtype_out: numpy.dtype, which "names" field used to determine output columns
    - fun_proc_loaded: None or Callable[
    [Union[pd.DataFrame, np.array], Mapping[str, Any], Optional[Mapping[str, Any]]],
    Union[pd.DataFrame, pd.DatetimeIndex]]

    See also `time_corr()` for used fields


    :yield: tuple (i1_path, i_chunk, path, df_filt) where
    - i1_path: 1-based counter of yielded data,
    - i1_chunk: 1-based counter of chunks,
    - path: file name,
    - df_filt: dataframe with time index and only columns listed in `cfg_in['dtype_out']`.names
    """

    read_csv_args_to_cfg_in = {
        'dtype': 'dtype_raw',
        'names': 'cols',
        'on_bad_lines': 'on_bad_lines',
        'comment': 'None',
        'delimiter': 'delimiter',
        'converters': 'converters',
        'skiprows': 'skiprows',
        'chunksize': 'blocksize',  # load in chunks
        'encoding': 'encoding'
        }
    read_csv_args = {arg: cfg_in[key] for arg, key in read_csv_args_to_cfg_in.items() if key in cfg_in}
    read_csv_args.update({
        'skipinitialspace': True,
        'usecols': cfg_in['dtype'].names,
        'header': None
    })
    # Removing "ParserWarning: Both a converter and dtype were specified for column k..."
    if read_csv_args['converters']:
        read_csv_args['dtype'] = {
            k: v[0] for i, (k, v) in enumerate(read_csv_args['dtype'].fields.items())
                if i not in read_csv_args['converters']
            }

    t_prev = None  #  not corrected part of previous time chunk for time_corr() filtering in csv_process()
    for i1_path, path in enumerate(files, start=1):
        # Save params that are needed below to extract date in `csv_process()`
        cfg_in['file_cur'] = Path(path)
        cfg_in['file_stem'] = cfg_in['file_cur'].stem
        cfg_in['n_rows'] = 0  # number of file rows loaded
        cfg_in['time_raw_st'] = None  # can be set after time is calculated (in csv_specific_proc)

        for i_retry in [False, True]:  # try again on ParserError with tuned loading parameters
            try:
                if first_last_row:
                    # Yield only 1st and last rows (metadata)
                    df = extract_csv_edges(path, mult=5, **read_csv_args)

                    df_filt, t_prev = csv_process(df, cfg_in, t_prev, first_last_row=True)
                    cfg_in['n_rows'] = float('nan')  # we don't know number of rows available
                    yield i1_path, 1, path, df_filt  # i1_chunk = 1 (caller expect 1st chunk)

                else:

                    # Normal loading

                    for i1_chunk, df in enumerate(
                        pd.read_csv(path, **read_csv_args, index_col=False), start=1
                    ):
                        df_filt, t_prev = csv_process(df, cfg_in, t_prev)
                        cfg_in['n_rows'] += len(df)
                        if df_filt is None:
                            continue
                        yield i1_path, i1_chunk, path, df_filt
                break
            except pd.errors.ParserError:  # for example NotImplementedError if bad file
                if (read_csv_args['on_bad_lines'] not in ('warn', 'skip')
                    or read_csv_args.get('engine') != 'python'
                    ):
                    lf.exception('Trying set [in].on_bad_lines = "warn" and retry\n')
                    read_csv_args['on_bad_lines'] = 'warn'
                    read_csv_args['engine'] = 'python'
                else:
                    lf.exception('Failed loading file')

            # lf.exception('If file "{}" have no data try to delete it?', paths)

format_parts_all = {  # columns formats in order to join (some may be skipped) for any input type
    "yyyy(text),mm(text),dd(text),HH(text),MM(text),SS(text)": {
            "regex": format.century + rb"\d{2}(,\d{1,2}){5}",
        "dtype": "|S4 |S2 |S2 |S2 |S2 |S2",
    },
    "Ax,Ay,Az,Mx,My,Mz": {
        "regex": rb"(,\-?\d{1,6}){6}",
        "dtype": "i2 i2 i2 i2 i2 i2",
    },
    "P_counts": {
        "regex": rb"(,\-?\d{1,8})",
        "dtype": "i4",
    },
    "Temp": {
        "regex": rb",\-?\d{1,3}(\.\d{1,2})?",
        "dtype": "f8",
    },
    "TempP": {
        "regex": rb",\-?\d{1,3}(\.\d{1,2})?",
        "dtype": "f8",
    },
    "Battery": {
        "regex": rb",\d{1,2}(\.\d{1,2})?",
        "dtype": "f8",
    },
}

# 1. Groups aliases
aliases = {
    "Y,M,D,H,M,S": "yyyy(text),mm(text),dd(text),HH(text),MM(text),SS(text)",
    "ADC": "P_counts",
    "T": "Temp",
    "TP": "TempP",
    "Tempr": "Temp",
    "TemprPres": "TempP",
    "Bat": "Battery"
}

# Предварительно собираем ВСЕ возможные строки для сопоставления в один список:
# (шаблон_для_сравнения, целевой_ключ_из_format_parts_all, длина_группы_в_колонках)
key_lengths = {fk: len(fk.split(",")) for fk in format_parts_all}
valid_patterns = []

for full_key in format_parts_all:
    L = key_lengths[full_key]
    # 1. Сам полный ключ (с типами)
    valid_patterns.append((full_key, full_key, L))
    # 2. Базовые имена колонок (без типов в скобках)
    base_cols = [p.split("(")[0].strip() for p in full_key.split(",")]
    valid_patterns.append((",".join(base_cols), full_key, L))

for alias, target in aliases.items():
    if target in key_lengths:
        L = key_lengths[target]
        # 3. Все алиасы
        valid_patterns.append((alias, target, L))


def longest_common_prefix_len(s1, s2):
    length = 0
    for c1, c2 in zip(s1.lower(), s2.lower()):
        if c1 == c2:
            length += 1
        else:
            break
    return length


def format_parts_select_raw(file_path: Path, max_bad_top_rows=100, min_good_cols=12, sep=",") -> Tuple[List[str], int, str | None]:
    """
    Parses a CSV header by reading up to 10 lines and matching columns to format groups.

    Scans the first 10 lines for the first numeric row after header row (where each column starts with letter
    with spaces/comma between). The line immediately preceding
    it is treated as the header (ignoring any garbage or duplicate headers before it).
    Columns are mapped to `format_parts_all` keys via exact `aliases` or longest
    common prefix matching, which prioritizes longer, more specific keys to resolve conflicts.

    If no numeric row is found within 10 lines, defaults to a sequence of format
    groups based on the maximum column count detected, preserving 'Battery' last.

    Args:
        file_path: Path to the CSV file.
        min_good_cols: default 12 (6 date cols + 3 accelerometers + 3 magnetometers cols)
    Returns:
        : (matched_format_keys, skiprows, header_line).
        header_line is None if defaulting.

    """
    # Values Pattern: must match at starts of strings
    data_line_pattern = re.compile(rf"^\s*\d+({sep}-?\d+(\.\d+)?){{{min_good_cols},}}")
    # Паттерн заголовка: CSV-строка, где КАЖДАЯ колонка начинается с буквы
    header_line_pattern = re.compile(rf"^\s*[a-zA-Z]\w*\s*({sep}\s*[a-zA-Z]\w*\s*){{{min_good_cols},}}$")

    header_line = None
    max_cols = 0
    data_cols = 0  # column count from first data line (authoritative)
    # Scan first max_bad_top_rows lines for header and first data row
    with file_path.open("r", encoding="utf-8-sig") as f:
        for skiprows, line in enumerate(islice(f, max_bad_top_rows)):
            if not (line := line.strip()):
                continue
            cols = line.split(sep)
            if len(cols) > max_cols:
                max_cols = len(cols)
            if data_line_pattern.match(line):
                if skiprows > 0:
                    data_cols = len(cols)
                    break  # Found data — header is the line(s) before it
                else:
                    continue  # Data on first line → no header yet
            if header_line_pattern.match(line):
                header_line = line
        else:
            lf.warning(
                "{}: no good data found in {} rows (row with {} max columns found)",
                file_path.name,
                max_bad_top_rows,
                max_cols,
            )
            header_line = None
        if max_cols < min_good_cols:
            lf.warning(
                "{}: no good data found in {} rows (row with {} max columns found, it is less than min useful: {})",
                file_path.name,
                max_bad_top_rows,
                max_cols,
                min_good_cols
            )
            return [], None, max_bad_top_rows
        # Если заголовок не найден
        if not header_line:
            # Формируем порядок групп по умолчанию
            default_keys = []
            acc = 0
            bat_cols = key_lengths.get("Battery", 1)  # Резервируем колонки под Battery

            for full_key, L in key_lengths.items():
                if full_key == "Battery":
                    continue  # Battery добавим в самом конце

                # Добавляем группу, если она вместе с зарезервированной Battery помещается в max_cols
                if acc + L + bat_cols <= max_cols:
                    default_keys.append(full_key)
                    acc += L

            # Добавляем Battery в конец, если общее количество колонок сходится
            if acc + bat_cols == max_cols:
                default_keys.append("Battery")

            return [], None, max_bad_top_rows

    # Header found — match columns to format groups
    header_cols = [col.strip() for col in header_line.split(sep)]

    # Guard: if data has fewer columns than header (e.g. previously botched
    # correction stripped trailing columns), trust the data column count.
    if data_cols and len(header_cols) > data_cols:
        lf.warning(
            "{}: header has {} columns but data has {} — truncating header to match data",
            file_path.name, len(header_cols), data_cols,
        )
        header_cols = header_cols[:data_cols]

    result_keys = []
    i_col = 0
    while i_col < len(header_cols):
        best_match_key = None
        best_score = -1
        best_L = 0
        for pattern, target_key, L in valid_patterns:
            if i_col + L > len(header_cols):
                continue

            header_chunk = ",".join(header_cols[i_col : i_col + L])
            if (p_len := longest_common_prefix_len(header_chunk, pattern)) == 0:
                continue

            # Scoring weights for header-to-pattern matching
            # 1) Primary: matched prefix length
            score = p_len * 1000000

            # 2) Full header-chunk match (entire header chunk consumed)
            if p_len == len(header_chunk):
                score += 100000

            # 3) Full pattern match — only when the pattern also covers the
            #    entire header chunk.  Without this guard a 1-char alias like
            #    "T" beats "TempP" on the "TP" header column because p_len==1
            #    trivially equals len("T"), even though only half of "TP" matched.
            if p_len == len(pattern) == len(header_chunk):
                score += 10000

            # 4) Specificity: longer patterns break remaining ties
            score += len(pattern)

            if score > best_score:
                best_score = score
                best_match_key = target_key
                best_L = L

        if best_match_key:
            result_keys.append(best_match_key)
            i_col += best_L
        else:
            i_col += 1

    return result_keys, header_line, skiprows


def format_parts_select(text_type):
    """Return format_parts_all keys for the given probe model.

    Order matches the actual column layout in raw files:
    date → sensors → P_counts → Temp → TempP → Battery.
    """
    b_default_type = text_type in ('i', '', 'b')
    return ["yyyy(text),mm(text),dd(text),HH(text),MM(text),SS(text)"] + (
        (["Ax,Ay,Az,Mx,My,Mz"] if text_type != "w" else [])
        + (
            ["Battery", "Temp"] if b_default_type
            else ["P_counts", "Temp", "TempP", "Battery"]  # text_type == 'p'
        )
    )


def _text_line_regex_from_parts(parts: list[str]) -> bytes:
    """Build ``^(?P<use>...) .*`` regex from format_parts_all keys."""
    return b"".join(
        [b"^(?P<use>"] + [p["regex"] for k, p in format_parts_all.items() if k in parts] + [b").*"]
    )


# Number of header rows to preserve in corrected file (from cfg_default['in']['skiprows'])
_N_KEEP_HEADER_ROWS: int = 3


def config_text_params(text_type, file_path=None) -> dict[str, Any]:
    """
    All parameters for csv correction and loading

    When ``file_path`` is given, parses file header via ``format_parts_select_raw()``
    to extract real ``skiprows`` / column layout. Otherwise falls back to defaults
    from ``text_type``. Reads the file only once.
    :return: dict with keys ``text_line_regex``, ``header``, ``dtype``, ``skiprows``.
        Empty dict if ``text_type`` is falsy and no ``file_path``.
    """
    skiprows = None
    if file_path:
        try:
            parts, header_line, skiprows = format_parts_select_raw(file_path)
        except FileNotFoundError:
            raise
        except Exception:
            parts = None
            lf.exception(
                'Failed to find known columns in raw file "{:s}"! => '
                'Trying to use defaults for this text type ({:s})',
                str(file_path), text_type,
            )
        else:
            lf.debug("Found columns in file under header {} after {} rows", header_line, skiprows)
    else:
        parts = None

    if not parts:  # None or [] — autodetection failed or found nothing
        if text_type:
            known = ("i", "p", "b", "d", "w")
            if text_type.lower() not in known:
                raise TypeError(f"Probe model {text_type} not recognized! Known: {known}")
            parts = format_parts_select(text_type)
            lf.warning("Falling back to default columns for text_type={}", text_type)
        else:
            return {}

    if skiprows is None:
        skiprows = cfg_default["in"]["skiprows"]

    # Deduplicate parts (preserve order) — prevents header having more names
    # than dtype entries when autodetection maps two header columns to the same key.
    parts = list(dict.fromkeys(parts))

    return {
        "text_line_regex": _text_line_regex_from_parts(parts),
        "header": ",".join(parts),
        "dtype": [t for k, p in format_parts_all.items() if k in parts for t in p["dtype"].split()],
        "skiprows": skiprows,
    }


def search_csv_files(
    path_in: Path,
) -> Dict[Tuple[str, int], list[Path]]:
    """Discover probe files by scanning directory matching *path_in* pattern.

    ``path_in.name`` is interpreted as glob or regex per :func:`_pattern_to_regex`.
    If *path_in* is a directory (no glob/regex in name), defaults to the
    regex ``i.*\\.txt`` (case-insensitive) — matching any inclinometer file
    with a ``.txt`` extension; corrected ``@``-prefixed files are always found
    independently (matched with or without ``@?`` prefix in pattern).

    If both ``@``-prefixed (corrected) and raw versions exist for the same
    probe identity, **only** the corrected version is included in the result.

    :param path_in: glob or regex pattern. ``path_in.name`` → pattern;
        ``path_in.parent`` → directory to scan.  If *path_in* is a
        directory, uses default regex ``i.*\\.txt`` (case-insensitive).
    :return: ``{(model, number): [paths]}`` — files grouped by probe identity.
        Only corrected files are returned when duplicates exist.
    :raises FileNotFoundError: when no probe files match.
    """
    # Directory input → default regex; otherwise interpret name as glob/regex
    if path_in.is_dir():
        parent = path_in
        ptn = re.compile(_DIR_DEFAULT_REGEX, re.IGNORECASE)
        lf.debug("input.path is directory => default regex: {}", _DIR_DEFAULT_REGEX)
    else:
        parent = path_in.parent
        regex_str = _pattern_to_regex(path_in.name)
        ptn = re.compile(regex_str, re.IGNORECASE)
        lf.debug("Pattern '{}' => regex /{}/i", path_in.name, regex_str)

    # Collect (identity, path, is_corrected) triples
    raw_files: list[Tuple[Tuple[str, int], Path, bool]] = []
    for f in sorted(parent.iterdir()):
        if not f.is_file():
            continue
        is_corrected = f.stem.startswith('@')
        match_name = f.name[1:] if is_corrected else f.name
        if not (ptn.match(match_name) or ptn.match(f.name)):
            continue
        identity = format.probe_from_name(f.stem.lower())
        if identity:
            raw_files.append((identity, f, is_corrected))

    if not raw_files:
        raise FileNotFoundError(f"No input files found matching {path_in}")

    # Group by identity; per-file pairing via canonical stem (mod_name normalization)
    by_identity: Dict[Tuple[str, int], list[Path]] = defaultdict(list)

    # Build set of canonical stems for corrected files → used to suppress only matched raw files
    corr_stems: set[str] = set()
    for _, f, is_corr in raw_files:
        if is_corr:
            _, p = mod_name(f.name, add_prefix="")
            corr_stems.add(p.stem.lower())

    for identity, f, is_corr in raw_files:
        if is_corr:
            by_identity[identity].append(f)
        else:
            # Suppress raw only if its corrected counterpart (same canonical stem) exists
            _, p = mod_name(f.name, add_prefix="")
            if p.stem.lower() not in corr_stems:
                by_identity[identity].append(f)

    n_suppressed = sum(1 for _, f, is_corr in raw_files if not is_corr) - sum(
        1 for ff in by_identity.values() for f in ff if not f.stem.startswith("@")
    )
    lf.info(
        "Files for {:d} probe{:s} found ({:d} raw suppressed by corrected counterparts):\n{:s}",
        len(by_identity),
        "" if len(by_identity) == 1 else "s",
        n_suppressed,
        "\n".join(
            "{}{}: {}".format(m, n, ", ".join(f.name for f in ff))
            for (m, n), ff in by_identity.items()
        ),
    )
    return dict(by_identity)


def correct_raw_files(
    csv_files: list[Path],
    text_type: str,
    text_line_regex: Optional[bytes] = None,
    skiprows: Optional[int] = None,
    pid: str = "",
) -> tuple[list[Path], dict]:
    """Correct raw (non-``@``-prefixed) files and return loading params.

    Already-corrected (``@``-prefixed) files pass through unchanged.
    ``search_csv_files`` already filters out originals when corrected versions
    exist, so this function simply corrects each raw file it receives.

    When all files are ``@``-prefixed, no detection happens and *params* is
    empty — the caller should then detect from the first corrected file
    via :func:`config_text_params`.

    :param csv_files: file list for a single probe (from ``search_csv_files``).
    :param text_type: probe model letter (``'i'``, ``'p'``, etc.).
    :param text_line_regex: regex bytes for ``correct_txt`` sub_str_list.
        If ``None``, derived per-file from :func:`config_text_params`.
    :param skiprows: total header rows in raw file.  The corrected file keeps
        only the last ``_N_KEEP_HEADER_ROWS`` of them.
    :param pid: probe column ID (e.g. ``"i67"``) for log messages.
    :return: ``(corrected_paths, params)`` where *params* contains
        ``header``, ``dtype``, ``skiprows``, ``text_line_regex`` from the first
        successful detection.  Empty dict when no raw files were corrected.
    """
    parent = csv_files[0].parent if csv_files else Path('.')
    paths_corrected: list[Path] = []
    params: dict = {}
    n_raw_corrected = 0

    for f in csv_files:
        if f.stem.startswith('@'):
            paths_corrected.append(f)
            continue

        n_raw_corrected += 1

        # Derive regex per-file (each file may have different column layout)
        if text_line_regex is None:
            file_params = config_text_params(text_type, f)
            file_regex = file_params["text_line_regex"]
            file_skiprows = file_params.get("skiprows", cfg_default["in"]["skiprows"])
        else:
            file_regex = text_line_regex
            file_skiprows = skiprows if skiprows is not None else cfg_default["in"]["skiprows"]
            file_params = {"text_line_regex": file_regex, "skiprows": file_skiprows}

        # Capture first detection for caller (avoids re-detection)
        if not params:
            params = file_params

        # Keep only last _N_KEEP_HEADER_ROWS header rows in corrected file
        n_keep = _N_KEEP_HEADER_ROWS
        header_rows = range(max(0, file_skiprows - n_keep), file_skiprows) if file_skiprows else 0

        _, file_out = mod_name(f.name, parse=True, add_prefix="@")
        corrected = correct_txt(
            f,
            dir_out=parent,
            mod_file_name=lambda _: file_out,
            sub_str_list=[file_regex, b"^.+"],
            binary_mode=False,
            header_rows=header_rows,
        )
        paths_corrected.append(corrected)

    n_corrected = n_raw_corrected
    # Safety-net: deduplicate (e.g. when search_csv_files counterpart detection
    # was bypassed or failed — prevents loading the same data twice)
    paths_corrected = list(dict.fromkeys(paths_corrected))
    probe_label = f" for {pid}" if pid else f" for text_type={text_type}"
    if n_corrected:
        lf.info("Corrected {:d} file{:s}{:s}", n_corrected,
                "" if n_corrected == 1 else "s", probe_label)
    else:
        lf.info("No raw files to correct{:s} (all already @-prefixed)", probe_label)
    return paths_corrected, params


#############################################################################################################
# Inclinometer file format and other parameters
cfg_default = {
    'in':      {
        'delimiter':          ',',  # \t not specify if need "None" useful for fixed length format
        'skiprows':           3,  # ignore this number of top rows both in preliminary correction and read_csv
        'on_bad_lines':       'warn',  #'error',
        # '--min_date', '07.10.2017 11:04:00',  # not output data < min_date
        # '--max_date', '29.12.2018 00:00:00',  # UTC, not output data > max_date
        'blocksize':          5_000_000,  # 1_000_000  # 15_000_000 hangs my comp
        'b_interact':         '0',
        'csv_specific_param': {
            'invert_magnetometer': True,
        # Bad time correction
        #     'time_shift': {
        #         'dt0': '0s',
        #         'time_st': None,                    # needed start time or it will be taken from existed bad time
        #         'time_en': '2023-07-24T13:10:00',   # required
        #         'time_raw_en': '2023-07-19T23:31:10',    # required if not linear_len and not time_raw_en
        #         'dt_end': None,  # can specify this (time_en - time_raw_en) interval instead time_en & bad
        #
        #         'linear_len': 16404000  # replace time using linear increased values of this length
        #         # (instead of linear transformation of existed values)
        #     }

        },
        'dt_max_interp_err': pd.Timedelta('15s'),   # 11s = (1.5s)*(time_en - time_st)/(time_end_bad - time_st)

        'dt_interp_between': pd.Timedelta('1.5s'),  # default
        'encoding':           'CP1251',  # read_csv() encoding parameter
        'max_text_width':     1000,  # used here to get sample for dask that should be > max possible row length
        # '--dt_from_utc_seconds', str(cfg['in']['dt_from_utc'][probe].total_seconds()),
        # '--fs_float', str(p_type[cfg['in']['probes_prefix']]['fs']),  # f'{fs(probe, file_in.stem)}',
        # 'corr_time_mode': True,  # to make sorted index: required to can process loaded data as timeseries by dask
        'text_type': None,
        'text_line_regex': None,
        'dt_from_utc': 0,
        'fun_proc_loaded': loaded_tcm
    },
    # 'out':     {},
    # 'filter':  {},
    'program': {
        'log_file_name':     'tcm_csv.log',
        'verbose': 'INFO',
        # 'dask_scheduler': 'synchronous'
    }
    # Warning! If True and b_incremental_update= True then not replace temporary file with result before proc.
    # '--log', 'log/csv2h5_inclin_Kondrashov.log'  # log operations
}


def load_from_csv_gen(
    csv_files_dict,
    cfg_in: Mapping[str, Any],
    cfg_in_probe: Optional[Mapping[str, Any]] = None,
    skip_for_meta: Optional[Callable[[Any], bool]] = None,
    return_=None,
) -> Iterator[Tuple[pd.DataFrame, Tuple[int, Any, Path]]]:
    """
    Generate (DataFrame, metadata) tuples from corrected CSV files.

    Iterates probes in insertion order of ``csv_files_dict``, processing each CSV
    after optional raw-file correction via ``correct_raw_files()``.  Yields one
    tuple per CSV chunk (or per file when ``blocksize`` is not set).

    :param csv_files_dict: ``{(text_type, number): [Path, ...]}`` mapping from
        ``search_csv_files()`` or ``discover_probes()``.
    :param cfg_in: input configuration merged over ``cfg_default["in"]``.  Key fields:
        ``path`` (str/Path, required), ``fun_proc_loaded`` (callable for post-load
        column processing), ``text_type`` (str, optional — selects pre-defined
        ``header``/``dtype``/``text_line_regex`` via ``config_text_params()``),
        ``blocksize`` (int, optional — enables chunked reading),
        ``csv_specific_param`` (dict, optional — oneliner corrections),
        ``fun_date_from_filename`` (str expression or callable, optional),
        ``min_date``/``max_date``/``date_to_from`` (time-range controls).
        ``corr_time_mode`` - see corresponding parameter in :func:`time_corr`.
    :param cfg_in_probe: optional ``{pid: overrides}`` dict; per-probe overrides
        merged into ``cfg_in`` for each pid during iteration.
    :param skip_for_meta: optional ``Callable[(i1_pid, pid, path)] -> bool``;
        returns ``True`` to skip a file (yields ``(None, meta)`` for it).
    :param return_: controls output mode:
        - ``None`` — yield all data normally.
        - ``"files_list"`` — yield ``(None, meta)`` for every file without loading.
        - ``"first_last_row"`` — yield only first & last row of each CSV.
    :yields: ``(df, (i1_pid, pid, path_csv))`` where:
        - ``df``: ``DataFrame`` (or ``None`` in short-circuit modes).
        - ``i1_pid``: 1-based probe index in ``csv_files_dict`` iteration order.
        - ``pid``:  probe identifier string (e.g. ``"i_R_01"``).
        - ``path_csv``: ``Path`` of the loaded (corrected) CSV file.
    """
    if (n_probes := len(csv_files_dict)) == 0:
        lf.warning("No raw files {:s} found!", str(cfg_in["path"]))
        sys.exit(ExitStatus.failure)

    # Optional return file list without any processing or configure to return only edge rows
    if return_:
        if 'files_list' in return_:
            lf.info('{:d} probes has raw files to process...', n_probes)
            for i1_pid, ((text_type, number), paths_csv) in enumerate(csv_files_dict.items(), start=1):
                for path_csv in paths_csv:
                    yield (
                        None,
                        (
                            i1_pid,
                            format.pcid_from_parts(model=text_type, number=number),
                            path_csv,
                        ),
                    )
            return
        first_last_row = "first_last_row" in return_
    else:
        first_last_row = False

    cfg_in = {**cfg_default["in"], **cfg_in}

    # Converting loaded columns configuration

    # Function for getting main fields after dataframe loaded from csv. Can be appended so overwritten below
    # - extract date from file name if needed
    if cfg_in.get("fun_date_from_filename") and isinstance(cfg_in["fun_date_from_filename"], str):
        cfg_in["fun_date_from_filename"] = eval(
            compile("lambda file_stem, century=None: {}".format(cfg_in["fun_date_from_filename"]), "", "eval")
        )
    # - additional calculation in read_csv() if needed
    if cfg_in.get("fun_proc_loaded") is None:
        # Default time processing after loading by dask/pandas.read_csv()
        if "coldate" not in cfg_in:  # if Time includes Date then we will just return it
            cfg_in["fun_proc_loaded"] = lambda a, cfg_in, dummy=None: a[cfg_in["col_index_name"]]
        else:  # else will return Time + Date
            cfg_in["fun_proc_loaded"] = lambda a, cfg_in, dummy=None: a["Date"] + np.array(
                np.int32(1000 * a[cfg_in["col_index_name"]]), dtype="m8[ms]"
            )

    if cfg_in["csv_specific_param"]:
        # Split 'csv_specific_param' fields into two parts for :
        # 1. loaded_corr() - oneliner operations ('fun', 'add'), and rest embed into
        # 2. cfg_in['fun_proc_loaded']()
        arg_loaded_corr = {}
        arg_fun_proc_loaded = {}
        for k, v in cfg_in["csv_specific_param"].items():
            (arg_loaded_corr if k.rsplit("_", 1)[-1] in ("fun", "add") else arg_fun_proc_loaded)[k] = v
        arg_fun_proc_loaded = {"csv_specific_param": arg_fun_proc_loaded} if arg_fun_proc_loaded else {}
        arg_loaded_corr = {"csv_specific_param": arg_loaded_corr} if arg_loaded_corr else {}

        # Update `cfg_in['fun_proc_loaded']` incorporating these two types of operations in our `read_csv()
        fun_proc_loaded = cfg_in["fun_proc_loaded"]

        def fun_loaded_and_loaded_corr(a, cfg_in):
            result = fun_proc_loaded(a, cfg_in=cfg_in, **arg_fun_proc_loaded)
            b = loaded_corr(result, cfg_in, **arg_loaded_corr)
            return b

        # Preserve the attributes of fun_proc_loaded
        update_wrapper(fun_loaded_and_loaded_corr, fun_proc_loaded)
        cfg_in["fun_proc_loaded"] = fun_loaded_and_loaded_corr

    # Loading files of corrected format and processing their data
    lf.debug(
        "{:d} probe{:s}{}...",
        n_probes,
        "" if n_probes == 1 else "s",
        f" by chunks of {b} counts" if (b := cfg_in.get("blocksize")) else "",
    )
    for i1_pid, ((text_type, number), paths_csv) in enumerate(csv_files_dict.items(), start=1):

        pid = format.pcid_from_parts(model=text_type, number=number)  # probe_type = "i"
        if skip_for_meta:
            # Skip specific files
            paths_csv_orig, paths_csv, paths_csv_old = paths_csv, [], []
            for path in paths_csv_orig:
                if skip_for_meta((i1_pid, pid, path)):
                    paths_csv_old.append(path)
                else:
                    paths_csv.append(path)
            if paths_csv_old:
                skipped_count = len(paths_csv_old)
                lf.warning('Skipped loading {:d} CSV files for present "{:s}" data', skipped_count, pid)
                yield None, (i1_pid, pid, paths_csv_old)
                if not paths_csv:
                    continue

        cfg_in_cur = {**cfg_in, "files": paths_csv}
        if cfg_in_probe and pid in cfg_in_probe:
            cfg_in_cur.update(cfg_in_probe[pid])
        update_cfg_time_ranges(cfg_in_cur, cfg_in_cur.get("min_date"), cfg_in_cur.get("max_date"))
        if cfg_in_cur.get("date_to_from"):
            t_to, t_from = cfg_in_cur["date_to_from"][:2]
            cfg_in_cur["dt_from_utc"] = t_from - t_to
            lf.warning(
                "Time shift to {} from {} will be performed ({} hours)",
                *cfg_in_cur["date_to_from"],
                -cfg_in_cur["dt_from_utc"],
            )

        # Correct raw files — already-corrected @-files pass through unchanged.
        paths_csv, _ = correct_raw_files(
            paths_csv,
            text_type=text_type,
            text_line_regex=cfg_in.get("text_line_regex"),
            pid=pid,
        )
        cfg_in_cur["files"] = paths_csv

        # Build base loading config (without column layout — that's per-file)
        cfg_in_base = {**cfg_in_cur}
        cfg_in_base.pop("header", None)
        cfg_in_base.pop("dtype", None)
        cfg_in_base.pop("skiprows", None)

        # Load each file with its own column detection.
        # Different files in the same probe group may have different column
        # counts (e.g. older 15-col vs newer 16-col pressure probes).
        n_paths = len(paths_csv)
        for i1_path, path_csv in enumerate(paths_csv, start=1):
            file_params = config_text_params(text_type, path_csv)
            cfg_text_specific = init_input_cols({
                **cfg_in_base,
                **{k: v for k, v in file_params.items() if k in ("header", "dtype", "skiprows")},
            })
            cfg_in_file = {**cfg_in_base, **cfg_text_specific, "files": [path_csv]}

            for _, i1_chunk, chunk_path, df in csv_read_gen(**cfg_in_file, first_last_row=first_last_row):
                if not first_last_row:
                    lf.debug(
                        "{: >2}.{:d} {:s} {:s}loading...",
                        i1_pid,
                        i1_chunk if cfg_in_file.get("blocksize") else "",
                        path_csv.name,
                        f" {i1_path}/{n_paths} " if n_paths > 1 else "",
                    )
                if df is None:
                    if i1_chunk == 1:
                        lf.warning('Not processed (empty) {}', path_csv.name)
                    continue
                yield df, (i1_pid, pid, path_csv)



# Per-pcid discovered file pair: raw (original) and corrected (@-prefixed).
# Either field may be None — never both (that would mean no file at all).
ProbeFiles = namedtuple("ProbeFiles", "raw corrected")



if __name__ == '__main__':

    def main():
        filenames_default = '*.txt'
        if len(sys.argv) > 1:
            dir_in, raw_re_ptn_file = sys.argv[1].split('*', 1)
            dir_in = Path(dir_in)
            raw_re_ptn_file = f'*{raw_re_ptn_file}' if raw_re_ptn_file else filenames_default
            lf.info(
                "Searching config file and input files in {:s} (default mask: {:s})", dir_in, raw_re_ptn_file
            )
        else:
            dir_in = Path.cwd().resolve()
            raw_re_ptn_file = filenames_default
            lf.info(
                "No command line arguments given => searching for {:s} input files and config in current dir",
                raw_re_ptn_file,
            )

        cfg_in = {'path': Path(dir_in) / raw_re_ptn_file}
        for i1_pid, pid, paths_csv, d in load_from_csv_gen(cfg_in):
            print(d.compute())
            # todo


    main()
