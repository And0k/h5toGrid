#!/usr/bin/env python3
# coding:utf-8
"""
  Author:  Andrey Korzh <ao.korzh@gmail.com>
  Purpose: Save synchronised averaged data to hdf5 tables
  Created: 01.09.2021

Load raw data and coefficients from hdf5 table (or group of tables)
Calculate new data (averaging by specified interval)
Combine this data to one new table
"""
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
import gc
import logging
from pathlib import Path
import re
import sys

from time import sleep
from typing import (Annotated, Any, Callable, Dict, Iterator, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple, List,
                    Union, TypeVar)
from numba import njit, prange
import numpy as np
import pandas as pd
import dask.array as da
import dask.dataframe as dd
from omegaconf import open_dict
import hydra

# my:

# # allows to run on both my Linux and Windows systems:
# scripts_path = Path(f"{'D:' if sys.platform == 'win32' else '/mnt/D'}/Work/_Python3/And0K/h5toGrid/scripts")
# sys.path.append(str(scripts_path.parent.resolve()))
# sys.path.append( str(Path(__file__).parent.parent.resolve()) ) # os.getcwd()
# from utils2init import ini2dict
# from scripts.incl_calibr import calibrate, calibrate_plot, coef2str
# from other_filters import despike, rep2mean
import cfg_dataclasses as cfg_d
from utils2init import Ex_nothing_done, call_with_valid_kwargs, set_field_if_no, this_prog_basename, LoggingStyleAdapter
from to_pandas_hdf5.h5toh5 import h5.out_init, h5.find_tables, h5.move_tables, h5.coords, h5.close, \
    h5.dispenser_and_names_gen
from to_pandas_hdf5.h5_dask_pandas import h5_load_range_by_coord, h5.append_to, i_bursts_starts, \
    filter_global_minmax, filter_local, cull_empty_partitions, dd_to_csv
from filters import rep2mean
from inclinometer.h5inclinometer_coef import rot_matrix_x, rotate_y

lf = LoggingStyleAdapter(logging.getLogger(__name__))
prog = 'incl_h5clc'
VERSION = '1.0.1'
hydra.output_subdir = 'cfg'

# ConfigIn_InclProc config parameters which are dicts with probes ID keys allows select different config value to
# process each probe data table. For this dict ID keys will be compared to regex group <n> returned by match this
# expression to table name (also matches tables with names having numbers joined by "_" of combined data):
re_probe_number = r'(?:(?:w|incl|i)_?(?P<n>[A-z]*\d+)_?)+'

@dataclass
class ConfigIn_InclProc(cfg_d.ConfigInHdf5_Simple):
    """Parameters of input files
    Constructor arguments:
    db_paths: path to pytables hdf5 store to load data. May use patterns in Unix shell style (usually *.h5)
    tables: list of tables names or regex patterns to search. Note: regex search is switched on only if you include "*"
    symbol somewhere in the pattern.

    Note: str for time values is used because datetime not supported by Hydra
    """
    tables: List[str] = field(default_factory=lambda: ['incl.*'])  # table names in hdf5 store to get data. Uses regexp if only one table name
    db_paths: Optional[List[str]] = None

    min_date: Optional[str] = None  # imput data time range minimum
    max_date: Optional[str] = None  # imput data time range maximum
    time_range: Optional[List[str]] = None
    # cruise directories to search in in.db_path to set path of out.db_path under it if out.db_path is not absolute:
    raw_dir_words_list: Optional[List[str]] = field(
        default_factory=lambda: cfg_d.ConfigInCsv().raw_dir_words)
    # fields that are dicts of existed fields - they specifies different values for each probe ID:
    dates_min: Dict[str, Any] = field(default_factory=dict)  # Any is for List[str] but hydra not supported
    dates_max: Dict[str, Any] = field(default_factory=dict)  # - // -
    time_ranges: Optional[Dict[str, Any]] = field(default_factory=dict)
    time_ranges_zeroing: Optional[Dict[str, Any]] = None
    bad_p_at_bursts_starts_periods: Optional[Dict[str, Any]] = None

@dataclass
class ConfigOut_InclProc(cfg_d.ConfigOutSimple):
    """
    "out": parameters of output files

    :param not_joined_db_path: If set then save processed velocity for each probe individually to this path. If not set then still write not averaged data separately for each probe to out.db_path with suffix ".proc_noAvg.h5". Table names will be the same as input data.
    :param table: table name in hdf5 store to write combined/averaged data. If not specified then will be generated on base of path of input files. Note: "*" is used to write blocks in autonumbered locations (see dask to_hdf())
    :param split_period: string (pandas offset: D, H, 2S, ...) to process and output in separate blocks. If saving csv for data with original sampling period is set then that csv will be splitted with by this length (for data with no bin averaging  only)
    :param aggregate_period: s, pandas offset strings  (5D, H, ...) to bin average data. This can greatly reduce sizes of output hdf5/text files. Frequenly used: None, 2s, 600s, 7200s
    :param text_path: path to save text files with processed velocity (each probe individually). No file if empty, "text_output" if ``out.aggregate_period`` is set. Relative paths are from out.db_path
    :param text_date_format: default '%Y-%m-%d %H:%M:%S.%f', format of date column in output text files. (.%f will be removed from end when aggregate_period > 1s). Can use float or string representations
    :param text_columns_list: if not empty then saved text files will contain only columns here specified
    :param b_all_to_one_col: concatenate all data in same columns in out db. both separated and joined text files will be written
    :param b_del_temp_db: default='False', temporary h5 file will be automatically deleted after operation. If false it will be remained. Delete it manually. Note: it may contain useful data for recovery

    """
    not_joined_db_path: Any = None
    table: str = ''
    aggregate_period: Optional[str] = ''
    split_period: str = ''
    text_path: Any = None
    text_date_format: str = '%Y-%m-%d %H:%M:%S.%f'
    text_columns: List[str] = field(default_factory=list)
    b_split_by_time_ranges: bool = False  # split averaged data by cfg['in']['time_ranges'] only if this set
    b_all_to_one_col: bool = False
    b_del_temp_db: bool = False
    b_overwrite: bool = True  # default is not to add new data to previous


@dataclass
class ConfigFilter_InclProc:
    """
    "filter": excludes some data:
    :param min_dict, max_dict: Filter out (set to NaN) data of ``key`` columns if it is below/above ``value``
    Possible keys are: input parameters: ... and intermediate parameters:
    - g_minus_1: sets Vabs to NaN if module of acceleration is greater
    - h_minus_1: sets Vdir to zero if module of magnetic field is greater,

    :param dates_min_dict: List with items in "key:value" format. Start of time range for each probe: (used instead common for each probe min_dict["Time"]) ')
    :param dates_max_dict: List with items in "key:value" format. End of time range for each probe: (used instead common for each probe max_dict["Time"]) ')

    :param bad_p_at_bursts_starts_period: pandas offset string. If set then marks each 2 samples of Pressure at start of burst as bad')

    """
    min: Optional[Dict[str, float]] = field(default_factory=dict)
    max: Optional[Dict[str, float]] = field(default_factory=lambda: {'g_minus_1': 1, 'h_minus_1': 8})
    bad_p_at_bursts_starts_period: str = ''


@dataclass
class ConfigProcess_InclProc:
    """
    Processing parameters
    :param time_range_zeroing_list: if specified then rotate data in this interval such that it will have min mean pitch and roll, display "info" warning about')
    :param time_range_zeroing_dict: {table: [start, end]}, rotate data in this interval only for specified probe number(s) data such that it will have min mean pitch and roll, the about "info" warning will be displayed. Probe number is int number consisted of digits in table name')
    :param azimuth_add_float: degrees, adds this value to velocity direction (will sum with _azimuth_shift_deg_ coef)')
    :param calc_version: string, default=, variant of processing Vabs(inclination):',
               choices=['trigonometric(incl)', 'polynom(force)'])
    :param max_incl_of_fit_deg_float', Finds point where g(x) = Vabs(inclination) became bend down and replaces after g with line so after max_incl_of_fit_deg {\\Delta}^{2}y ≥ 0 for x > max_incl_of_fit_deg') used if not provided in coef.
    """
    time_range_zeroing: Optional[List[str]] = field(default_factory=list)
    time_range_zeroing_dict: Optional[Dict[str, str]] = field(default_factory=dict)
    azimuth_add: float = 0
    calc_version: str = 'trigonometric(incl)'
    max_incl_of_fit_deg: Optional[float] = None


@dataclass
class ConfigProgram_InclProc(cfg_d.ConfigProgram):
    """
    return_ may have values: "<cfg_before_cycle>" to return config before loading data, or other any nonempty
    threads - good for numeric code that releases the GIL (like NumPy, Pandas, Scikit-Learn, Numba, …)
    """
    dask_scheduler: str = ''  # variants: synchronous, threads, processes, single-threaded, distributed


# @dataclass
# class ConfigProbes_InclProc:
#     filter: ConfigFilter_InclProc
#     process: ConfigProcess_InclProc


cs_store_name = Path(__file__).stem
cs, ConfigType = cfg_d.hydra_cfg_store(
    cs_store_name, {
        'input': [ConfigIn_InclProc],  # Load the config "in_hdf5" from the config group "input"
        'out': [ConfigOut_InclProc],  # Set as MISSING to require the user to specify a value on the command line.
        'filter': [ConfigFilter_InclProc],
        'process': [ConfigProcess_InclProc],
        'program': [ConfigProgram_InclProc]
        # 'probes': ['probes__incl_proc'],
        # 'search_path': 'empty.yml' not works
    },
    module=sys.modules[__name__]
    )


RT = TypeVar('RT')  # return type


def allow_dask(wrapped: Callable[..., RT]) -> Callable[..., RT]:
    """
    Use dask.Array functions instead of numpy if first argument is dask.Array
    :param wrapped: function that use methods of numpy each of that existed in dask.Array
    :return:
    """

    @wraps(wrapped)
    def _func(*args):
        if isinstance(args[0], (da.Array, dd.DataFrame)):
            np = da
        return wrapped(*args)

    return _func


@allow_dask
def f_pitch(Gxyz):
    """
    Pitch calculating
    :param Gxyz: shape = (3,len) Accelerometer data
    :return: angle, radians, shape = (len,)
    """
    return -np.arctan(Gxyz[0, :] / np.linalg.norm(Gxyz[1:, :], axis=0))
    # =arctan2(Gxyz[0,:], sqrt(square(Gxyz[1,:])+square(Gxyz[2,:])) )')


@allow_dask
def f_roll(Gxyz):
    """
    Roll calculating
    :param Gxyz: shape = (3,len) Accelerometer data
    :return: angle, radians, shape = (len,)
    """
    return np.arctan2(Gxyz[1, :], Gxyz[2, :])


# @njit - removed because numba gets long bytecode dump without messages/errors
def fIncl_rad2force(incl_rad: np.ndarray):
    """
    Theoretical force from inclination
    :param incl_rad:
    :return:
    """
    return np.sqrt(np.tan(incl_rad) / np.cos(incl_rad))


@allow_dask
def fIncl_deg2force(incl_deg):
    return fIncl_rad2force(np.radians(incl_deg))


# no @jit - polyval not supported
def fVabsMax0(x_range, y0max, coefs):
    """End point of good interpolation"""
    x0 = x_range[np.flatnonzero(np.polyval(coefs, x_range) > y0max)[0]]
    return (x0, np.polyval(coefs, x0))


# no @jit - polyval not supported
def fVabs_from_force(force, coefs, vabs_good_max=0.5):
    """

    :param force:
    :param coefs: polynom coef ([d, c, b, a]) to calc Vabs(force)
    :param vabs_good_max: m/s, last (max) value of polynom to calc Vabs(force) next use line function
    :return:
    """

    # preventing "RuntimeWarning: invalid value encountered in less/greater"
    is_nans = np.isnan(force)
    force[is_nans] = 0

    # last force of good polynom fitting
    x0, v0 = fVabsMax0(np.arange(1, 4, 0.01), vabs_good_max, coefs)

    def v_normal(x):
        """
            After end point of good polinom fitting use linear function
        :param x:
        :return:
        """
        v = np.polyval(coefs, x)
        return np.where(x < x0, v, v0 + (x - x0) * (v0 - np.polyval(coefs, x0 - 0.1)) / 0.1)

    # Using fact that 0.25 + eps = fIncl_rad2force(0.0623) where eps > 0 is small value
    incl_good_min = 0.0623
    incl_range0 = np.linspace(0, incl_good_min, 15)
    force_range0 = fIncl_rad2force(incl_range0)

    def v_linear(x):
        """
        Linear(incl) function crossed (0,0) and 1st good polinom fitting force = 0.25
        where force = fIncl_rad2force(incl)
        :param x:
        :return: array with length of x

        """
        return np.interp(x, force_range0, incl_range0) * np.polyval(coefs, 0.25) / incl_good_min

    force = np.where(force > 0.25, v_normal(force), v_linear(force))
    force[is_nans] = np.nan
    return force

    """
    f(lambda x, d, c, b, a: where(x > 0.25,
                              f(lambda x, v, x0, v0: where(x < x0, v, v0 + (x - x0)*(v0 - polyval([d,c,b,a], x0 - 0.1))/0.1), x, polyval([d,c,b,a], x), *fVabsMax0(0.5, a, b, c, d)),
                              f(lambda incl, y: interp(y, fIncl_deg2force(incl), incl), linspace(0,3.58627992,15), x)*polyval([d,c,b,a], 0.25)/3.58627992), force, *kVabs)
    """


@njit(fastmath=True)
def trigonometric_series_sum(r, coefs):
    """

    :param r:
    :param coefs: array of even length
    :return:
    Not jitted version was:
    def trigonometric_series_sum(r, coefs):
        return coefs[0] + np.nansum(
            [(a * np.cos(nr) + b * np.sin(nr)) for (a, b, nr) in zip(
                coefs[1::2], coefs[2::2], np.arange(1, len(coefs) / 2)[:, None] * r)],
            axis=0)
    """
    out = np.empty_like(r)
    out[:] = coefs[0]
    for n in prange(1, (len(coefs)+1) // 2):
        a = coefs[n * 2 - 1]
        b = coefs[n * 2]
        nr = n * r
        out += (a * np.cos(nr) + b * np.sin(nr))
    return out


@njit(fastmath=True)
def rep_if_bad(checkit, replacement):
    return checkit if (checkit and np.isfinite(checkit)) else replacement


@njit(fastmath=True)
def f_linear_k(x0, g, g_coefs):
    replacement = np.float64(10)
    return min(rep_if_bad(np.diff(g(x0 - np.float64([0.01, 0]), g_coefs)).item() / 0.01, replacement),
               replacement)


@njit(fastmath=True)
def f_linear_end(g, x, x0, g_coefs):
    """
    :param g, x, g_coefs: function and its arguments to calc g(x, *g_coefs)
    :param x0: argument where g(...) replace with linear function if x > x0
    :return:
    """
    g0 = g(x0, g_coefs)
    return np.where(x < x0, g(x, g_coefs), g0 + (x - x0) * f_linear_k(x0, g, g_coefs))


@njit(fastmath=True)
def v_trig(r, coefs):
    squared = np.sin(r) / trigonometric_series_sum(r, coefs)
    # with np.errstate(invalid='ignore'):  # removes warning of comparison with NaN
    # return np.sqrt(squared, where=squared > 0, out=np.zeros_like(squared)) to can use njit replaced with:
    return np.where(squared > 0, np.sqrt(squared), 0)


def v_abs_from_incl(incl_rad: np.ndarray, coefs: Sequence, calc_version='trigonometric(incl)', max_incl_of_fit_deg=None) -> np.ndarray:
    """
    Vabs = np.polyval(coefs, Gxyz)

    :param incl_rad:
    :param coefs: coefficients.
    Note: for 'trigonometric(incl)' option if not max_incl_of_fit_deg then it is in last coefs element
    :param calc_version: 'polynom(force)' if this str or len(coefs)<=4 else if 'trigonometric(incl)' uses trigonometric_series_sum()
    :param max_incl_of_fit_deg:
    :return:
    """
    if len(coefs)<=4 or calc_version == 'polynom(force)':
        lf.warning('Old coefs method "polynom(force)"')
        if not len(incl_rad):   # need for numba njit error because of dask calling with empty arg to check type if no meta?
            return incl_rad     # empty of same type as input
        force = fIncl_rad2force(incl_rad)
        return fVabs_from_force(force, coefs)

    elif calc_version == 'trigonometric(incl)':
        if max_incl_of_fit_deg:
            max_incl_of_fit = np.radians(max_incl_of_fit_deg)
        else:
            max_incl_of_fit = np.radians(coefs[-1])
            coefs = coefs[:-1]

        with np.errstate(invalid='ignore'):                         # removes warning of comparison with NaN
            return f_linear_end(g=v_trig, x=incl_rad,
                                x0=np.atleast_1d(max_incl_of_fit),
                                g_coefs=np.float64(coefs))          # atleast_1d, float64 is to can use njit
    else:
        raise NotImplementedError(f'Bad calc method {calc_version}', )


# @overload(np.linalg.norm)
# def jit_linalg_norm(x, ord=None, axis=None, keepdims=False):
#     # replace notsupported numba argument
#     if axis is not None and (ord is None or ord == 2):
#         s = (x.conj() * x).real
#         # original implementation: return sqrt(add.reduce(s, axis=axis, keepdims=keepdims))
#         return np.sqrt(s.sum(axis=axis, keepdims=keepdims))


@allow_dask
def fInclination(Gxyz: np.ndarray):
    return np.arctan2(np.linalg.norm(Gxyz[:-1, :], axis=0), Gxyz[2, :])


# @allow_dask not need because explicit np/da lib references are not used
#@njit("f8[:,:](f8[:,:], f8[:,:], f8[:,:])") - failed for dask array
def fG(Axyz: Union[np.ndarray, da.Array],
       Ag: Union[np.ndarray, da.Array],
       Cg: Union[np.ndarray, da.Array]) -> Union[np.ndarray, da.Array]:
    """
    Allows use of transposed Cg
    :param Axyz:
    :param Ag: scaling coefficient
    :param Cg: shift coefficient
    :return:
    """
    assert Ag.any(), 'Ag coefficients all zeros!!!'
    c = np.atleast_2d(Cg)
    return Ag @ (Axyz - (c if c.shape[0] == Ag.shape[0] else c.T))


# @overload(fG)
# def jit_fG(Axyz: np.ndarray, Ag: np.ndarray, Cg: np.ndarray):
#     # replace notsupported numba int argument
#     if isinstance(Axyz.dtype, types.Integer):
#         return Ag @ (Axyz.astype('f8') - (Cg if Cg.shape[0] == Ag.shape[0] else Cg.T))


# def polar2dekart_complex(Vabs, Vdir):
#     return Vabs * (da.cos(da.radians(Vdir)) + 1j * da.sin(da.radians(Vdir)))
@allow_dask
def polar2dekart(Vabs, Vdir) -> List[Union[da.Array, np.ndarray]]:
    """

    :param Vabs:
    :param Vdir:
    :return: list [v, u] of (north, east) components. List (not tuple) is used because it is more convenient to concatenate it with other data
    """
    return [Vabs * np.cos(np.radians(Vdir)), Vabs * np.sin(np.radians(Vdir))]


# @allow_dask
# def dekart2polar(v_en):
#     """
#     Not Tested
#     :param u:
#     :param v:
#     :return: [Vabs, Vdir]
#     """
#     return np.linalg.norm(v_en, axis=0), np.degrees(np.arctan2(*v_en))

def dekart2polar_df_uv(df, **kwargs):
    """

    :param d: if no columns u and v remains unchanged
    :**kwargs :'inplace' not supported in dask. dumn it!
    :return: [Vabs, Vdir] series
    """

    # why da.linalg.norm(df.loc[:, ['u','v']].values, axis=1) gives numpy (not dask) array?
    # da.degrees(df.eval('arctan2(u, v)')))

    if 'u' in df.columns:

        # kdegrees = 180 / np.pi
        # *{kdegrees:.20}

        return df.eval(f"""
        Vabs = hypot(u, v)
        Vdir = degrees(arctan2(u, v))
        """, **kwargs)
    else:
        return df


def incl_calc_velocity_nodask(
        a: pd.DataFrame,
        Ag, Cg, Ah, Ch,
        kVabs: Sequence = (1, 0),
        azimuth_shift_deg: float = 0,
        cfg_filter: Mapping[str, Any] = None,
        cfg_proc=None):
    """

    :param a:
    :param Ag:
    :param Cg:
    :param Ah:
    :param Ch:
    :param azimuth_shift_deg:
    :param cfg_filter: dict. cfg_filter['max_g_minus_1'] useed to check module of Gxyz, cfg_filter['max_h_minus_1'] to set Vdir=0
    :param cfg_proc: 'calc_version', 'max_incl_of_fit_deg'
    :return: dataframe withcolumns ['Vabs', 'Vdir', col, 'inclination'] where col is additional column in _a_, or may be absent
    """
    da = np
    # da.from_array = lambda x, *args, **kwargs: x
    lf.info('calculating V')
    if kVabs == (1, 0):
        lf.warning('kVabs == (1, 0)! => V = sqrt(sin(inclination))')
    #
    # old coefs need transposing: da.dot(Ag.T, (Axyz - Cg[0, :]).T)
    # fG = lambda Axyz, Ag, Cg: da.dot(Ag, (Axyz - Cg))
    # fInclination = lambda Gxyz: np.arctan2(np.sqrt(np.sum(np.square(Gxyz[:-1, :]), 0)), Gxyz[2, :])

    try:
        Gxyz = fG(a.loc[:, ('Ax', 'Ay', 'Az')].to_numpy().T, Ag, Cg)  # lengths=True gets MemoryError   #.to_dask_array()?, dd.from_pandas?
        # .rechunk((1800, 3))
        # filter
        GsumMinus1 = da.linalg.norm(Gxyz, axis=0) - 1  # should be close to zero
        incl_rad = fInclination(Gxyz)

        if cfg_filter and ('max_g_minus_1' in cfg_filter):
            bad_g = np.fabs(GsumMinus1.compute()) > cfg_filter['max_g_minus_1']
            bad_g_sum = bad_g.sum(axis=0)
            if bad_g_sum > 0.1 * len(GsumMinus1):
                print('Acceleration is too high (>{}) in {}% points!'.format(
                    cfg_filter['max_g_minus_1'], 100 * bad_g_sum / len(GsumMinus1))
                )
            incl_rad[bad_g] = np.nan

        Vabs = v_abs_from_incl(incl_rad, kVabs, cfg_proc['calc_version'], cfg_proc['max_incl_of_fit_deg'])

        Hxyz = fG(a.loc[:, ('Mx', 'My', 'Mz')].to_numpy().T, Ah, Ch)
        Vdir = azimuth_shift_deg - da.degrees(da.arctan2(
            (Gxyz[0, :] * Hxyz[1, :] - Gxyz[1, :] * Hxyz[0, :]) * (GsumMinus1 + 1),
            Hxyz[2, :] * da.square(Gxyz[:-1, :]).sum(axis=0) - Gxyz[2, :] * (
                (Gxyz[:-1, :] * Hxyz[:-1, :]).sum(axis=0))
            ))

        col = 'Pressure' if ('Pressure' in a.columns) else 'Temp' if ('Temp' in a.columns) else []
        columns = ['Vabs', 'Vdir', 'v', 'u', 'inclination']
        arrays_list = [Vabs, Vdir] + polar2dekart(Vabs, Vdir) + [da.degrees(incl_rad)]
        a = a.assign(**{c: ar for c, ar in zip(columns, arrays_list)})  # a[c] = ar

        # df = pd.DataFrame.from_records(dict(zip(columns, [Vabs, Vdir, np.degrees(incl_rad)])), columns=columns, index=a.index)  # no sach method in dask
        return a[columns + [col]]
    except Exception as e:
        lf.exception('Error in incl_calc_velocity():')
        raise


@njit
def recover_x__sympy_lambdify(y, z, Ah, Ch, mean_Hsum):
    """
    After sympy added abs() under sqrt() to exclude comlex values
    :param y:
    :param z:
    :param Ah:
    :param Ch:
    :param mean_Hsum:
    :return:
    """

    [a00, a01, a02] = Ah[0]
    [a10, a11, a12] = Ah[1]
    [a20, a21, a22] = Ah[2]
    [c00, c10, c20] = np.ravel(Ch)
    return (
        a00 ** 2 * c00 + a00 * a01 * c10 - a00 * a01 * y + a00 * a02 * c20 - a00 * a02 * z + a10 ** 2 * c00 +
        a10 * a11 * c10 - a10 * a11 * y + a10 * a12 * c20 - a10 * a12 * z + a20 ** 2 * c00 + a20 * a21 * c10 -
        a20 * a21 * y + a20 * a22 * c20 - a20 * a22 * z - np.sqrt(np.abs(
               -a00 ** 2 * a11 ** 2 * c10 ** 2 + 2 * a00 ** 2 * a11 ** 2 * c10 * y - a00 ** 2 * a11 ** 2 * y ** 2 - 2 * a00 ** 2 * a11 * a12 * c10 * c20 + 2 * a00 ** 2 * a11 * a12 * c10 * z + 2 * a00 ** 2 * a11 * a12 * c20 * y - 2 * a00 ** 2 * a11 * a12 * y * z - a00 ** 2 * a12 ** 2 * c20 ** 2 + 2 * a00 ** 2 * a12 ** 2 * c20 * z - a00 ** 2 * a12 ** 2 * z ** 2 - a00 ** 2 * a21 ** 2 * c10 ** 2 + 2 * a00 ** 2 * a21 ** 2 * c10 * y - a00 ** 2 * a21 ** 2 * y ** 2 - 2 * a00 ** 2 * a21 * a22 * c10 * c20 + 2 * a00 ** 2 * a21 * a22 * c10 * z + 2 * a00 ** 2 * a21 * a22 * c20 * y - 2 * a00 ** 2 * a21 * a22 * y * z - a00 ** 2 * a22 ** 2 * c20 ** 2 + 2 * a00 ** 2 * a22 ** 2 * c20 * z - a00 ** 2 * a22 ** 2 * z ** 2 + a00 ** 2 * mean_Hsum ** 2 + 2 * a00 * a01 * a10 * a11 * c10 ** 2 - 4 * a00 * a01 * a10 * a11 * c10 * y + 2 * a00 * a01 * a10 * a11 * y ** 2 + 2 * a00 * a01 * a10 * a12 * c10 * c20 - 2 * a00 * a01 * a10 * a12 * c10 * z - 2 * a00 * a01 * a10 * a12 * c20 * y + 2 * a00 * a01 * a10 * a12 * y * z + 2 * a00 * a01 * a20 * a21 * c10 ** 2 - 4 * a00 * a01 * a20 * a21 * c10 * y + 2 * a00 * a01 * a20 * a21 * y ** 2 + 2 * a00 * a01 * a20 * a22 * c10 * c20 - 2 * a00 * a01 * a20 * a22 * c10 * z - 2 * a00 * a01 * a20 * a22 * c20 * y + 2 * a00 * a01 * a20 * a22 * y * z + 2 * a00 * a02 * a10 * a11 * c10 * c20 - 2 * a00 * a02 * a10 * a11 * c10 * z - 2 * a00 * a02 * a10 * a11 * c20 * y + 2 * a00 * a02 * a10 * a11 * y * z + 2 * a00 * a02 * a10 * a12 * c20 ** 2 - 4 * a00 * a02 * a10 * a12 * c20 * z + 2 * a00 * a02 * a10 * a12 * z ** 2 + 2 * a00 * a02 * a20 * a21 * c10 * c20 - 2 * a00 * a02 * a20 * a21 * c10 * z - 2 * a00 * a02 * a20 * a21 * c20 * y + 2 * a00 * a02 * a20 * a21 * y * z + 2 * a00 * a02 * a20 * a22 * c20 ** 2 - 4 * a00 * a02 * a20 * a22 * c20 * z + 2 * a00 * a02 * a20 * a22 * z ** 2 - a01 ** 2 * a10 ** 2 * c10 ** 2 + 2 * a01 ** 2 * a10 ** 2 * c10 * y - a01 ** 2 * a10 ** 2 * y ** 2 - a01 ** 2 * a20 ** 2 * c10 ** 2 + 2 * a01 ** 2 * a20 ** 2 * c10 * y - a01 ** 2 * a20 ** 2 * y ** 2 - 2 * a01 * a02 * a10 ** 2 * c10 * c20 + 2 * a01 * a02 * a10 ** 2 * c10 * z + 2 * a01 * a02 * a10 ** 2 * c20 * y - 2 * a01 * a02 * a10 ** 2 * y * z - 2 * a01 * a02 * a20 ** 2 * c10 * c20 + 2 * a01 * a02 * a20 ** 2 * c10 * z + 2 * a01 * a02 * a20 ** 2 * c20 * y - 2 * a01 * a02 * a20 ** 2 * y * z - a02 ** 2 * a10 ** 2 * c20 ** 2 + 2 * a02 ** 2 * a10 ** 2 * c20 * z - a02 ** 2 * a10 ** 2 * z ** 2 - a02 ** 2 * a20 ** 2 * c20 ** 2 + 2 * a02 ** 2 * a20 ** 2 * c20 * z - a02 ** 2 * a20 ** 2 * z ** 2 - a10 ** 2 * a21 ** 2 * c10 ** 2 + 2 * a10 ** 2 * a21 ** 2 * c10 * y - a10 ** 2 * a21 ** 2 * y ** 2 - 2 * a10 ** 2 * a21 * a22 * c10 * c20 + 2 * a10 ** 2 * a21 * a22 * c10 * z + 2 * a10 ** 2 * a21 * a22 * c20 * y - 2 * a10 ** 2 * a21 * a22 * y * z - a10 ** 2 * a22 ** 2 * c20 ** 2 + 2 * a10 ** 2 * a22 ** 2 * c20 * z - a10 ** 2 * a22 ** 2 * z ** 2 + a10 ** 2 * mean_Hsum ** 2 + 2 * a10 * a11 * a20 * a21 * c10 ** 2 - 4 * a10 * a11 * a20 * a21 * c10 * y + 2 * a10 * a11 * a20 * a21 * y ** 2 + 2 * a10 * a11 * a20 * a22 * c10 * c20 - 2 * a10 * a11 * a20 * a22 * c10 * z - 2 * a10 * a11 * a20 * a22 * c20 * y + 2 * a10 * a11 * a20 * a22 * y * z + 2 * a10 * a12 * a20 * a21 * c10 * c20 - 2 * a10 * a12 * a20 * a21 * c10 * z - 2 * a10 * a12 * a20 * a21 * c20 * y + 2 * a10 * a12 * a20 * a21 * y * z + 2 * a10 * a12 * a20 * a22 * c20 ** 2 - 4 * a10 * a12 * a20 * a22 * c20 * z + 2 * a10 * a12 * a20 * a22 * z ** 2 - a11 ** 2 * a20 ** 2 * c10 ** 2 + 2 * a11 ** 2 * a20 ** 2 * c10 * y - a11 ** 2 * a20 ** 2 * y ** 2 - 2 * a11 * a12 * a20 ** 2 * c10 * c20 + 2 * a11 * a12 * a20 ** 2 * c10 * z + 2 * a11 * a12 * a20 ** 2 * c20 * y - 2 * a11 * a12 * a20 ** 2 * y * z - a12 ** 2 * a20 ** 2 * c20 ** 2 + 2 * a12 ** 2 * a20 ** 2 * c20 * z - a12 ** 2 * a20 ** 2 * z ** 2 + a20 ** 2 * mean_Hsum ** 2))
           ) / (a00 ** 2 + a10 ** 2 + a20 ** 2)


def recover_magnetometer_x(Mcnts, Ah, Ch, max_h_minus_1, len_data):
    Hxyz = fG(Mcnts, Ah, Ch)  # x.rechunk({0: -1, 1: 'auto'}, block_size_limit=1e8)
    HsumMinus1 = da.linalg.norm(Hxyz, axis=0) - 1  # should be close to zero

    # Channel x recovering
    bad = da.isnan(Mcnts[0, :])
    need_recover_mask = da.isfinite(Mcnts[1:, :]).any(axis=0)  # where other channels ok
    #sleep(cfg_filter['sleep_s'])
    can_recover = need_recover_mask.sum(axis=0).compute()
    if can_recover:
        Mcnts_list = [[], [], []]
        need_recover_mask &= bad  # only where x is bad
        need_recover = need_recover_mask.sum(axis=0).compute()
        if need_recover:  # have points where recover is needed and is possible
            lf.info(
                'Magnetometer x channel recovering where {:d} it is bad and y&z is ok (y&z ok in {:d}/{:d})',
                need_recover, can_recover, len_data
            )
            # Try to recover mean_Hsum (should be close to 1)
            mean_HsumMinus1 = np.nanmedian((HsumMinus1[HsumMinus1 < max_h_minus_1]).compute())
            if np.isnan(mean_HsumMinus1) or (np.fabs(mean_HsumMinus1) > 0.5 and need_recover / len_data > 0.95):
                # need recover all x points because too small points with good HsumMinus1
                lf.warning('mean_Hsum is mostly bad (mean={:g}), most of data need to be recovered ({:g}%) so no trust it'
                          ' at all. Recovering all x-ch.data with setting mean_Hsum = 1',
                          mean_HsumMinus1, 100 * need_recover / len_data)
                bad = da.ones_like(HsumMinus1, dtype=np.bool_)
                mean_HsumMinus1 = 0
            else:
                lf.warning('calculated mean_Hsum - 1 is good (close to 0): mean={:s}', mean_HsumMinus1)

            # Recover magnetometer's x channel

            # todo: not use this channel but recover dir instantly
            # da.apply_along_axis(lambda x: da.where(x < 0, 0, da.sqrt(abs(x))),
            #                               0,
            #                  mean_Hsum - da.square(Ah[2, 2] * (rep2mean_da(Mcnts[2,:], Mcnts[2,:] > 0) - Ch[2])) -
            #                               da.square(Ah[1, 1] * (rep2mean_da(Mcnts[1,:], Mcnts[1,:] > 0) - Ch[1]))
            #                               )

            #Mcnts_x_recover = recover_x__sympy_lambdify(Mcnts[1, :], Mcnts[2, :], Ah, Ch, mean_Hsum=mean_HsumMinus1 + 1)
            Mcnts_x_recover = da.map_blocks(  # replaced to this to use numba:
                recover_x__sympy_lambdify, Mcnts[1, :], Mcnts[2, :],
                Ah=Ah, Ch=Ch, mean_Hsum=mean_HsumMinus1 + 1, dtype=np.float64, meta=np.float64([])
            )
            Mcnts_list[0] = da.where(need_recover_mask, Mcnts_x_recover, Mcnts[0, :])
            bad &= ~need_recover_mask

            # other points recover by interp
            Mcnts_list[0] = da.from_array(rep2mean_da2np(Mcnts_list[0], ~bad), chunks=Mcnts_list[0].chunks,
                                          name='Mcnts_list[0]')
        else:
            Mcnts_list[0] = Mcnts[0, :]

        lf.debug('interpolating magnetometer data using neighbor points separately for each channel...')
        need_recover_mask = da.ones_like(HsumMinus1, dtype=np.bool_)  # here save where Vdir can not recover
        for ch, i in [('x', 0), ('y', 1), ('z', 2)]:  # in ([('y', 1), ('z', 2)] if need_recover else
            print(ch, end=' ')
            if (ch != 'x') or not need_recover:
                Mcnts_list[i] = Mcnts[i, :]
            bad = da.isnan(Mcnts_list[i])
            n_bad = bad.sum(axis=0).compute()  # exits with "Process finished with exit code -1073741819 (0xC0000005)"!
            if n_bad:
                n_good = HsumMinus1.shape[0] - n_bad
                if n_good / n_bad > 0.01:
                    lf.info(f'channel {ch}: bad points: {n_bad} - recovering using nearest good points ({n_good})')
                    Mcnts_list[i] = da.from_array(rep2mean_da2np(Mcnts_list[i], ~bad), chunks=Mcnts_list[0].chunks,
                                                  name=f'Mcnts_list[{ch}]-all_is_finite')
                else:
                    lf.warning(
                        f'channel {ch}: bad points: {n_bad} - will not recover because too small good points ({n_good})'
                    )
                    Mcnts_list[i] = np.nan + da.empty_like(HsumMinus1)
                    need_recover_mask[bad] = False

        Mcnts = da.vstack(Mcnts_list)
        Hxyz = fG(Mcnts, Ah, Ch)  # #x.rechunk({0: -1, 1: 'auto'}, block_size_limit=1e8)

    else:
        lf.info('Magnetometer can not be recovered')
        need_recover_mask = None

    return Hxyz, need_recover_mask


def rep2mean_da(y: da.Array, bOk=None, x=None, overlap_depth=None) -> da.Array:
    """
    Interpolates bad values (inverce of bOk) in each dask block.
    Note: can leave NaNs if no good data in block
    :param y:
    :param bOk:
    :param x:
    :return: dask array of np.float64 values

    g = da.overlap.overlap(x, depth={0: 2, 1: 2}, boundary={0: 'periodic', 1: 'periodic'})
    todo:
    g2 = g.map_blocks(myfunc)
    result = da.overlap.trim_internal(g2, {0: 2, 1: 2})
    """
    if x is None:  # dask requires "All variadic arguments must be arrays"
        return da.map_overlap(rep2mean, y, bOk, depth=overlap_depth, dtype=np.float64, meta=np.float64([]))
    else:
        return da.map_overlap(rep2mean, y, bOk, x, depth=overlap_depth, dtype=np.float64, meta=np.float64([]))
    #y.map_blocks(rep2mean, bOk, x, dtype=np.float64, meta=np.float64([]))


def rep2mean_da2np(y: da.Array, bOk=None, x=None) -> np.ndarray:
    """
    same as rep2mean_da but also replaces bad block values with constant (rep2mean_da can not replace bad if no good)
    :param y:
    :param bOk:
    :param x: None, other not implemented
    :return: numpy array of np.float64 values

    """

    y_last = None
    y_out_list = []
    for y_bl, b_ok in zip(y.blocks, bOk.blocks):
        y_bl, ok_bl = da.compute(y_bl, b_ok)
        y_bl = rep2mean(y_bl, ok_bl)
        ok_in_replaced = np.isfinite(y_bl[~ok_bl])
        if not ok_in_replaced.all():            # have bad
            assert not ok_in_replaced.any()     # all is bad
            if y_last:                          # This only useful to keep infinite/nan values considered ok:
                y_bl[~ok_bl] = y_last          # keeps y[b_ok] unchanged - but no really good b_ok if we here
            print('continue search good data...')
        else:
            y_last = y_bl[-1]                  # no more y_last can be None
        y_out_list.append(y_bl)
    n_bad_from_start = y.numblocks[0] - len(y_out_list)  # should be 0 if any good in first block
    for k in range(n_bad_from_start):
        y_out_list[k][:] = y_out_list[n_bad_from_start+1][0]
    return np.hstack(y_out_list)

    # # This little simpler version not keeps infinite values considered ok:
    # y_rep = rep2mean_da(y, bOk, x)
    # y_last = None
    # y_out_list = []
    # for y_bl in y_rep.blocks:
    #     y_bl = y_bl.compute()
    #     ok_bl = np.isfinite(y_bl)
    #     if not ok_bl.all():
    #         if y_last:
    #             y_bl[~ok_bl] = y_last
    #     else:
    #         y_last = y_bl[-1]                  # no more y_last can be None
    #     y_out_list.append(y_bl)
    # n_bad_from_start = len(y.blocks) - len(y_out_list)
    # for k in range(n_bad_from_start):
    #     y_out_list[k][:] = y_out_list[n_bad_from_start+1][0]
    # return np.hstack(y_out_list)

# Next function calculates this columns:
incl_calc_velocity_cols = ('Vabs', 'Vdir', 'v', 'u', 'inclination')

def incl_calc_velocity(a: dd.DataFrame,
                       filt_max: Optional[Mapping[str, float]] = None,
                       cfg_proc: Optional[Mapping[str, Any]] = None,
                       Ag: Optional[np.ndarray] = None, Cg: Optional[np.ndarray] = None,
                       Ah: Optional[np.ndarray] = None, Ch: Optional[np.ndarray] = None,
                       kVabs: Optional[np.ndarray] = None, azimuth_shift_deg: Optional[float] = 0,
                       cols_prepend: Optional[Sequence[str]] = incl_calc_velocity_cols,
                       **kwargs
                       ) -> dd.DataFrame:
    """
    Calculates velocity from raw accelerometer/magnetometer data. Replaces raw columns with velocity vector parameters.
    :param a: dask dataframe with columns (at least):
    - 'Ax','Ay','Az': accelerometer channels
    - 'Mx','My','Mz': magnetometer channels
    Coefficients:
    :param Ag: coef
    :param Cg:
    :param Ah:
    :param Ch:
    :param kVabs: if None then will not try to calc velocity
    :param azimuth_shift_deg:
    :param filt_max: dict with fields:
    - g_minus_1: mark bad points where |Gxyz| is greater, if any then its number will be logged,
    - h_minus_1: to set Vdir=0 and...
    :param cfg_proc: 'calc_version', 'max_incl_of_fit_deg'
    :param cols_prepend: new parameters that will be prepended to a columns (only columns if a have only raw
    accelerometer and magnetometer fields). If None then ('Vabs', 'Vdir', 'v', 'u', 'inclination') will be used. Use
     any from these elements in needed order.
    :param kwargs: not affects calculation
    :return: dataframe with prepended columns ``cols_prepend`` and removed accelerometer and
    magnetometer raw data
    """

    # old coefs need transposing: da.dot(Ag.T, (Axyz - Cg[0, :]).T)
    # fG = lambda Axyz, Ag, Cg: da.dot(Ag, (Axyz - Cg))  # gets Memory error
    # def fInclination_da(Gxyz):  # gets Memory error
    #     return da.arctan2(da.linalg.norm(Gxyz[:-1, :], axis=0), Gxyz[2, :])

    # f_pitch = lambda Gxyz: -np.arctan2(Gxyz[0, :], np.sqrt(np.sum(np.square(Gxyz[1:, :]), 0)))
    # f_roll = lambda Gxyz: np.arctan2(Gxyz[1, :], Gxyz[2, :])
    # fHeading = lambda H, p, r: np.arctan2(H[2, :] * np.sin(r) - H[1, :] * np.cos(r), H[0, :] * np.cos(p) + (
    #        H[1, :] * np.sin(r) + H[2, :] * np.cos(r)) * np.sin(p))

    # Gxyz = a.loc[:,('Ax','Ay','Az')].map_partitions(lambda x,A,C: fG(x.values,A,C).T, Ag, Cg, meta=('Gxyz', float64))
    # Gxyz = da.from_array(fG(a.loc[:,('Ax','Ay','Az')].values, Ag, Cg), chunks = (3, 50000), name='Gxyz')
    # Hxyz = da.from_array(fG(a.loc[:,('Mx','My','Mz')].values, Ah, Ch), chunks = (3, 50000), name='Hxyz')

    lengths = tuple(a.map_partitions(len).compute())  # or True to autocalculate it
    len_data = sum(lengths)

    if kVabs is not None and 'Ax' in a.columns:
        lf.info('calculating V')
        try:    # lengths=True gets MemoryError   #.to_dask_array()?, dd.from_pandas?
            Gxyz = fG(a.loc[:, ('Ax', 'Ay', 'Az')].to_dask_array(lengths=lengths).T, Ag, Cg)

            # .rechunk((1800, 3))
            # filter
            GsumMinus1 = da.linalg.norm(Gxyz, axis=0) - 1  # should be close to zero
            incl_rad = fInclination(Gxyz)  # .compute()

            if 'g_minus_1' in filt_max:
                bad = np.fabs(GsumMinus1) > filt_max['g_minus_1']  # .compute()
                bad_g_sum = bad.sum(axis=0).compute()
                if bad_g_sum:
                    if bad_g_sum > 0.1 * len(GsumMinus1):  # do not message for few points
                        lf.warning(
                            'Acceleration is too high (>{}) in {:g}% points (data removed)!',
                            filt_max['g_minus_1'], 100 * bad_g_sum / len(GsumMinus1)
                        )
                    incl_rad[bad] = np.nan
            # else:
            #     bad = da.zeros_like(GsumMinus1, np.bool_)

            # lf.debug('{:.1g}Mb of data accumulated in memory '.format(dfs_all.memory_usage().sum() / (1024 * 1024)))

            # sPitch = f_pitch(Gxyz)
            # sRoll = f_roll(Gxyz)
            # Vdir = np.degrees(np.arctan2(np.tan(sRoll), np.tan(sPitch)) + fHeading(Hxyz, sPitch, sRoll))

            # Velocity absolute value

            #Vabs = da.map_blocks(incl_rad, kVabs, )  # , chunks=GsumMinus1.chunks
            Vabs = incl_rad.map_blocks(v_abs_from_incl,
                                       coefs=kVabs,
                                       calc_version=cfg_proc['calc_version'],
                                       max_incl_of_fit_deg=cfg_proc['max_incl_of_fit_deg'],
                                       dtype=np.float64, meta=np.float64([]))
            # Vabs = np.polyval(kVabs, np.where(bad, np.nan, Gxyz))
            # v = Vabs * np.cos(np.radians(Vdir))
            # u = Vabs * np.sin(np.radians(Vdir))

            Hxyz, need_recover_mask = recover_magnetometer_x(
                a.loc[:, ('Mx', 'My', 'Mz')].to_dask_array(lengths=lengths).T, Ah, Ch, filt_max['h_minus_1'], len_data)
            if need_recover_mask is not None:
                HsumMinus1 = da.linalg.norm(Hxyz, axis=0) - 1  # should be close to zero
                Vdir = 0  # default value
                # bad = ~da.any(da.isnan(Mcnts), axis=0)
                Vdir = da.where(da.logical_or(need_recover_mask, HsumMinus1 < filt_max['h_minus_1']),
                                azimuth_shift_deg - da.degrees(da.arctan2(
                                    (Gxyz[0, :] * Hxyz[1, :] - Gxyz[1, :] * Hxyz[0, :]) * (GsumMinus1 + 1),
                                    Hxyz[2, :] * da.square(Gxyz[:-1, :]).sum(axis=0) - Gxyz[2, :] * (
                                            Gxyz[:-1, :] * Hxyz[:-1, :]).sum(axis=0)
                                    )),
                                Vdir  # default value
                                )
            else:  # Set magnetometer data as a function of accelerometer data - worst case: relative direction recovery
                lf.warning(
                    'Bad magnetometer data => Assign direction inversely proportional to toolface angle (~ relative angle if no rotations around device axis)')
                Vdir = azimuth_shift_deg - da.degrees(da.arctan2(Gxyz[0, :], Gxyz[1, :]))
            Vdir = Vdir.flatten()

            # Combine calculated data with existed in ``a`` except raw accelerometer and magnetometer columns.
            a = a.drop(['Ax', 'Ay', 'Az', 'Mx', 'My', 'Mz'], axis='columns')
            cols_remains = a.columns.to_list()
            # Placing columns at first, defined in ``cols_prepend`` in their order, and keep remained in previous order
            _ = zip(incl_calc_velocity_cols, [Vabs, Vdir] + polar2dekart(Vabs, Vdir) + [da.degrees(incl_rad)])
            a = a.assign(
                **{c: (
                    ar if isinstance(ar, da.Array) else
                    da.from_array(ar, chunks=GsumMinus1.chunks)
                    ).to_dask_dataframe(index=a.index) for c, ar in _ if c in cols_prepend
                   }
                )[list(cols_prepend) + cols_remains]  # reindex(, axis='columns')  # a[c] = ar
        except Exception as e:
            lf.exception('Error in incl_calc_velocity():')
            raise

    return a


def calc_pressure(a: dd.DataFrame,
                  bad_p_at_bursts_starts_period: Optional[str] = None,
                  P=None,
                  PTemp=None,
                  PBattery=None,
                  PBattery_min=None,
                  Battery_ok_min=None,
                  **kwargs
                  ) -> dd.DataFrame:
    """
    replaces P column by Pressure applying polyval() and temperature and battery corrections if such coefs exists to P.
    :param a:
    :param bad_p_at_bursts_starts_period:
    :param P: conversion of raw Pressure to physical units polynom coefficents
    :param PTemp: polynom coefficients for temperature compensation that will be added to raw Pressure
    :param PBattery: raw P battery compensation coefficients
    :param PBattery_min: if is not None then:
      - where Battery > PBattery_min only add constant polyval(PBattery, PBattery_min) to Pressure
      - where Battery < PBattery_min add polyval(PBattery, Battery)
      else adds polyval(PBattery, Battery) everywhere
    :param Battery_ok_min:
    :param kwargs:
    :return:
    """
    if P is None:
        return a
    for col_p in ['P', 'P_counts']:
        if col_p in a.columns:
            break
    else:
        return a

    meta = ('Pressure', 'f8')
    lengths = tuple(a.map_partitions(len, enforce_metadata=False).compute())
    len_data = sum(lengths)
    a = a.rename(columns={col_p: 'Pressure'})
    a['Pressure'] = a['Pressure'].astype(float)  # pressure is in integer counts and we will add float correction

    # Compensate for Temperature
    if PTemp is not None:
        a, lengths = cull_empty_partitions(a, lengths)  # removing empty partitions need for no empty chunks for rep2mean_da

        arr = a.Temp.to_dask_array(lengths=lengths)
        # Interpolate Temp jaggies

        # where arr changes:
        bc = (da.ediff1d(arr, to_begin=1000) != 0).rechunk(chunks=arr.chunks)  # diff get many 1-sized chunks

        def f_st_en(x):
            b = np.flatnonzero(x)
            if b.size:
                return b[[0, -1]]
            else:
                return np.int64([-1, -1])

        st_en_use = bc.map_blocks(f_st_en, dtype=np.int64).compute()
        i_ok = np.append(
            (np.append(0, np.cumsum(bc.chunks[0][:-1])).repeat(2) + st_en_use)[st_en_use >= 0],
            len_data
            )
        i_ok = i_ok[np.ediff1d(i_ok, to_begin=i_ok[0]) > 100]
        d_ok = np.ediff1d(i_ok, to_begin=i_ok[0])
        assert d_ok.sum() == len_data
        d_ok = (tuple(d_ok),)
        # interpolate between change points:
        arr_smooth = rep2mean_da(arr.rechunk(chunks=d_ok), bOk=bc.rechunk(chunks=d_ok), overlap_depth=1)

        a_add = arr_smooth.rechunk(chunks=arr.chunks).map_blocks(
            lambda x: np.polyval(PTemp, x), dtype=np.float64, meta=np.float64([])
            ).to_dask_dataframe(index=a.index)
        a.Pressure += a_add  #

    # Compensate for Battery
    if PBattery is not None:
        arr = a.Battery.to_dask_array(lengths=lengths)

        # Interpolate Battery bad region (near the end where Battery is small and not changes)
        if Battery_ok_min is not None:
            i_0, i_1, i_st_interp = da.searchsorted(  # 2 points before bad region start and itself
                -arr, -da.from_array(Battery_ok_min + [0.08, 0.02, 0.001]), side='right'
                ).compute()
            # if have bad region => the number of source and target points is sufficient:
            if i_0 != i_1 and i_st_interp < len_data:
                arr[i_st_interp:] = da.arange(len_data - i_st_interp) * \
                             ((arr[i_1] - arr[i_0]) / (i_1 - i_0)) + arr[i_st_interp]

        # Compensation on Battery after Battery < PBattery_min, before add constant polyval(PBattery, PBattery_min)
        if PBattery_min is not None:
            i_st_compensate = da.searchsorted(-arr, da.from_array(-PBattery_min)).compute().item()
            arr[:i_st_compensate] = np.polyval(PBattery, PBattery_min)
            arr[i_st_compensate:] = arr[i_st_compensate:].map_blocks(
                lambda x: np.polyval(PBattery, x), dtype=np.float64, meta=np.float64([]))
        else:
            arr = arr.map_blocks(lambda x: np.polyval(PBattery, x), dtype=np.float64, meta=np.float64([]))

        a.Pressure += arr.to_dask_dataframe(index=a.index)

    # Calculate pressure using P polynom
    if bad_p_at_bursts_starts_period:   # '1h'
        # with marking bad P data in first samples of bursts (works right only if bursts is at hours starts!)
        p_bursts = a.Pressure.repartition(freq=bad_p_at_bursts_starts_period)

        def calc_and_rem2first(p: pd.Series) -> pd.Series:
            """ mark bad data in first samples of burst"""
            # df.iloc[0:1, df.columns.get_loc('P')]=0  # not works!
            pressure = np.polyval(P, p.values)
            pressure[:2] = np.nan
            p[:] = pressure
            return p

        a.Pressure = p_bursts.map_partitions(calc_and_rem2first, meta=meta)
    else:
        a.Pressure = a.Pressure.map_partitions(lambda x: np.polyval(P, x), meta=meta)

    return a


def coef_zeroing_rotation(g_xyz_0, Ag_old, Cg) -> np.ndarray:
    """
    Zeroing rotation matrix Rz to correct old rotation matrices Ag_old, Ah_old (Ag = Rz @ Ag_old, Ah = Rz @ Ah_old)
    based on accelerometer values when inclination is zero.
    :param g_xyz_0: 3x1 values of columns 'Ax','Ay','Az' (it is practical to provide mean values of some time range)
    :param Ag_old, Cg: numpy.arrays, rotation matrix and shift for accelerometer
    :return Rz: 3x3 numpy.array , corrected rotation matrices

    Method of calculation of ``g_xyz_0`` from dask dataframe data:
     g_xyz_0 = da.atleast_2d(da.from_delayed(
          a_zeroing.loc[:, ('Ax', 'Ay', 'Az')].mean(
             ).values.to_delayed()[0], shape=(3,), dtype=np.float64, name='mean_G0'))
     """

    if not len(g_xyz_0):
        print(f'zeroing(): no data {g_xyz_0}, no rotation is done')
        return 1
    if g_xyz_0.shape[0] != 3:
        raise ValueError('Bad g_xyz_0 shape')

    old_g_xyz_0 = fG(g_xyz_0, Ag_old, Cg)
    old_pitch = f_pitch(old_g_xyz_0)
    old_roll = f_roll(old_g_xyz_0)
    lf.info('Zeroing pitch = {}°, roll = {}°', *[
        np.format_float_positional(x, precision=5) for x in np.rad2deg([old_pitch[0], old_roll[0]])
    ])
    return rotate_y(rot_matrix_x(np.cos(old_roll), np.sin(old_roll)), angle_rad=old_pitch)


def coef_rotate(*A, Z):
    return [Z @ a for a in A]


def coef_zeroing_rotation_from_data(
        df: dd.DataFrame | pd.DataFrame,
        Ag, Cg,
        time_range=None,
        **kwargs) -> np.ndarray:
    """
    Zeroing rotation matrix Rz to correct old rotation matrices Ag_old, Ah_old (Ag = Rz @ Ag_old, Ah = Rz @ Ah_old)
    based on accelerometer raw data from time range
    :param df: pandas or dask DataFrame with columns 'Ax', 'Ay', 'Az' - raw accelerometer data
    :param time_range: optional data index time range to select data
    Old coefficients:
    :param Ag:
    :param Cg:
    :param kwargs: not used
    :return: Ag, Ah - updated coefficients
    """
    df = df.loc[slice(*pd.to_datetime(time_range, utc=True)) if time_range else None, ('Ax', 'Ay', 'Az')]
    lf.info('Zeroing data: average {:d} points in interval {:s} - {:s}', len(df),
            *getattr(df, 'divisions' if (b_dd := isinstance(df, dd.DataFrame)) else 'index')[[0, -1]])
    g_xyz_0 = np.atleast_2d(df.mean().values.compute() if b_dd else df.mean().values).T
    return coef_zeroing_rotation(g_xyz_0, Ag, Cg)


def coef_zeroing(g_xyz_0, Ag_old, Cg, Ah_old) -> Tuple[np.ndarray, np.ndarray]:
    """
    Zeroing: correct Ag_old, Ah_old
    :param g_xyz_0: 1x3 values of columns 'Ax','Ay','Az' (it is practical to provide mean values of some time range)
    :param Ag_old, Cg: numpy.arrays, rotation matrix and shift for accelerometer
    :param Ah_old: numpy.array 3x3, rotation matrix for magnetometer
    return (Ag, Ah): numpy.arrays (3x3, 3x3), corrected rotation matrices

    Calculation method of ``g_xyz_0`` from dask dataframe data can be:
     g_xyz_0 = da.atleast_2d(da.from_delayed(
          a_zeroing.loc[:, ('Ax', 'Ay', 'Az')].mean(
             ).values.to_delayed()[0], shape=(3,), dtype=np.float64, name='mean_G0'))
     """
    lf.warning('coef_zeroing() depreciated. Use coef_zeroing_rotation() instead and save rotation matrix for future use')
    Z = coef_zeroing_rotation(g_xyz_0, Ag_old, Cg, Ah_old)
    Ag = Z @ Ag_old
    Ah = Z @ Ah_old
    lf.debug('calibrated Ag = {:s},\n Ah = {:s}', Ag, Ah)
    # # test: should be close to zero:
    # Gxyz0 = fG(g_xyz_0, Ag, Cg)
    # #? Gxyz0mean = np.transpose([np.nanmean(Gxyz0, 1)])

    return Ag, Ah

    # filter temperature
    # if 'Temp' in a.columns:
    # x = a['Temp'].map_partitions(np.asarray)
    # blocks = np.diff(np.append(i_starts, len(x)))
    # chunks = (tuple(blocks.tolist()),)
    # y = da.from_array(x, chunks=chunks, name='tfilt')
    #
    # def interp_after_median3(x, b):
    #     return np.interp(
    #         da.arange(len(b_ok), chunks=cfg_out['chunksize']),
    #         da.flatnonzero(b_ok), median3(x[b]), da.nan, da.nan)
    #
    # b = da.from_array(b_ok, chunks=chunks, meta=('Tfilt', 'f8'))
    # with ProgressBar():
    #     Tfilt = da.map_blocks(interp_after_median3(x, b), y, b).compute()

    # hangs:
    # Tfilt = dd.map_partitions(interp_after_median3, a['Temp'], da.from_array(b_ok, chunks=cfg_out['chunksize']), meta=('Tfilt', 'f8')).compute()

    # Tfilt = np.interp(da.arange(len(b_ok)), da.flatnonzero(b_ok), median3(a['Temp'][b_ok]), da.nan,da.nan)
    # @+node:korzh.20180524213634.8: *3* main
    # @+others
    # @-others


def filt_data_dd(a, dt_between_bursts=None, dt_hole_warning: Optional[np.timedelta64] = None, cfg_filter=None
                 ) -> Tuple[dd.DataFrame, np.array]:
    """
    Filter and get burst starts (i.e. finds gaps in data)
    :param a:
    :param dt_between_bursts: minimum time interval between blocks to detect them and get its starts (i_burst),
    also repartition on found blocks if can i.e. if known_divisions. If None then set as defined in i_bursts_starts():
    greater than min of two first intervals + 1s
    :param dt_hole_warning: numpy.timedelta64
    :param cfg_filter: if set then filter by removing rows
    :return: (a, i_burst) where:
     - a: filtered,
     - i_burst: array with 1st elem 0 and other - starts of data after big time holes

    """
    if True:  # try:
        # filter

        # this is will be done by filter_global_minmax() below
        # if 'P' in a.columns and min_p:
        #     print('restricting time range by good Pressure')
        #     # interp(NaNs) - removes warning 'invalid value encountered in less':
        #     a['P'] = filt_blocks_da(a['P'].values, i_burst, i_end=len(a)).to_dask_dataframe(['P'], index=tim)
        #     # todo: make filt_blocks_dd and replace filt_blocks_da: use a['P'] = a['P'].repartition(chunks=(tuple(np.diff(i_starts).tolist()),))...?

        # decrease interval based on ini date settings and filtering and recalculate bursts
        a = filter_global_minmax(a, cfg_filter=cfg_filter)
        tim = a.index.compute()  # History: MemoryError((6, 10868966), dtype('float64'))
        i_burst, mean_burst_size, max_hole = i_bursts_starts(tim, dt_between_blocks=dt_between_bursts)
        # or use this and check dup? shift?:
        # i_good = np.search_sorted(tim, a.index.compute())
        # i_burst = np.search_sorted(i_good, i_burst)

        if not a.known_divisions:  # this is usually required for next op
            divisions = tuple(tim[np.append(i_burst, len(tim) - 1)])
            a.set_index(a.index, sorted=True).repartition(divisions=divisions)
            # a = a.set_index(a.index, divisions=divisions, sorted=True)  # repartition? (or reset_index)

        if max_hole and dt_hole_warning and max_hole > dt_hole_warning:
            lf.warning(f'max time hole: {max_hole.astype(datetime)*1e-9}s')
        return a, i_burst


def probes_gen(cfg_in: Mapping[str, Any], cfg_out: None = None
                 ) -> Iterator[Tuple[str, Tuple[Any, ...]]]:
    """
    Generate table names with associated coefficients which are loaded from '{tbl}/coef' node of hdf5 file.
    :param cfg_in: dict with fields:
    - tables: tables names search pattern or sequence of table names
    - db_path: hdf5 file with tables which have coef group nodes.
    :param cfg_out: not used but kept for the requirement of h5.dispenser_and_names_gen() argument
    :return: iterator that returns (table_name, coefficients).
    - table_name is same as input tables names except that "incl" replaced with "i"
    - coefficients are None if db_path ends with 'proc_noAvg'.
     "Vabs0" coef. name are replaced with "kVabs"
    Updates cfg_in['tables'] - sets to list of found tables in store
    """

    with pd.HDFStore(cfg_in['db_path'], mode='r') as store:
        if len(cfg_in['tables']) == 1:
            cfg_in['tables'] = h5.find_tables(store, cfg_in['tables'][0])
        for tbl in cfg_in['tables']:
            if '.proc_noAvg' in cfg_in['db_path'].suffixes[-2:-1]:
                # Loading already processed data
                tbl = tbl.replace('incl', 'i')  # can be useful only if need loading from processed tbl
                coefs_dict = None
            else:
                coefs_dict = {}
                # Finds up to 2 levels of coefficients, naming rule gives coefs this names (but accepts any paths):
                # coefs: ['coef/G/A', 'coef/G/C', 'coef/H/A', 'coef/H/C', 'coef/H/azimuth_shift_deg', 'coef/Vabs0'])
                # names: ['Ag', 'Cg', 'Ah', 'Ch', 'azimuth_shift_deg', 'kVabs'],

                node_coef = store.get_node(f'{tbl}/coef')
                if node_coef is None:
                    lf.warning('Skipping this table "{:s}" - not found coefs!', tbl)
                    continue
                for node_name in node_coef.__members__:
                    node_coef_l2 = node_coef[node_name]
                    if getattr(node_coef_l2, '__members__', False):  # node_coef_l2 is group
                        for node_name_l2 in node_coef_l2.__members__:
                            name = f'{node_name_l2}{node_name.lower() if node_name_l2[-1].isupper() else ""}'
                            coefs_dict[name] = node_coef_l2[node_name_l2].read()
                    else:  # node_coef_l2 is value
                        coefs_dict[node_name if node_name != 'Vabs0' else 'kVabs'] = node_coef_l2.read()
            yield tbl, coefs_dict

    return


def gen_subconfigs(
        cfg: MutableMapping[str, Any],
        fun_gen=probes_gen,
        db_paths=None,
        tables=None,
        dates_min=None,
        dates_max=None,
        time_ranges=None,
        time_ranges_zeroing=None,
        bad_p_at_bursts_starts_periods=None,
        **cfg_in_common) -> Iterator[Tuple[dd.DataFrame, Dict[str, np.array], str, int]]:
    """
    Wraps h5.dispenser_and_names_gen() to deal with many db_paths, tables, dates_min and dates_max

    h5.dispenser_and_names_gen() parameters:
    :param time_ranges:
    :param time_ranges_zeroing:
    :param bad_p_at_bursts_starts_periods:
    :param cfg: dict
    :param fun_gen: generator of (tbl, coefs)

    Optional plural named parameters - same meaning as keys of cfg['in'/'process'/'filter'] but for many probes:
    :param db_paths,
    :param tables,
    :param dates_min,
    :param dates_max
    :param time_ranges: optional - generate dataframes in this parts, one part equivalent to (date_min, date_max)
    :param time_ranges_zeroing: list of [st1 en1 st2 en2...] - like cfg['process']['time_range_zeroing'] for many probes
    :param bad_p_at_bursts_starts_periods: True - like cfg['filter']['bad_p_at_bursts_starts_period']

    Other fields originated from cfg['in']:
    :param: cfg_in_common: single valued fields which will be replaced by earlier described fields:
    - db_path: Union[str, Path], str/path of real path or multiple paths joined by '|' to be splitted in list cfg['db_paths'],
    - table:
    - min_date:
    - max_date:
    - dt_between_bursts, dt_hole_warning - optional - arguments
    :return: Iterator(d, coefs.copy(), tbl, probe_number), where:
    - d - data
    - coefs, tbl - fun_gen output
    - probe_number - probe number - digits from tbl

    """

    # cfg_in = locals()

    # dict to load from many sources:
    cfg_in_many = {}
    for k, v in [  # single_name_to_many_vals
            ('db_path', db_paths),
            ('table', tables),
            ]:
        try:
            cfg_in_many[k] = v or cfg_in_common[k]
        except KeyError:
            continue

    # dict to process with different parameters:
    cfg_many = {}
    for k, v in [  # single_name_to_many_vals
            ('min_date', dates_min),
            ('max_date', dates_max),
            ('time_range', time_ranges),  # time_range can include many intervals: [start1, end1, start2, end2, ...]
            ('time_range_zeroing', time_ranges_zeroing),
            ('bad_p_at_bursts_starts_period', bad_p_at_bursts_starts_periods)
        ]:
        try:
            cfg_many[k] = v or cfg_in_common[k]
        except KeyError:
            continue

    def delistify(vals):  # here only to get table item from 'tables' parameter (that is of type list with len=1)
        return vals[0] if isinstance(vals, Iterable) and len(vals) == 1 else vals

    def group_dict_vals(param_dicts, probes=None) -> Iterator[Dict[str, str]]:
        """
        Generate dicts {parameter: value} from dict {parameter: {probe: value}} for each probe
        :param param_dicts: dict {parameter: {probe: value}} or {parameter: value} to use this value for all probes
        :param probes: if value is a list and has length = len(probes) then assign each item to each probe. Not
        used with our configs as cfg['in']['probe'] config parameter even not exist.
        :return: dict with singular named keys having str values
        """
        probes_dict = {}
        params_common = {}
        for param, probe_vals in param_dicts.items():
            # if probe_vals is None:
            #     continue
            if isinstance(probe_vals, list) and isinstance(probes, list) and len(probe_vals) == len(probes):
                probe_vals = dict(zip(probes, probe_vals))
            if isinstance(probe_vals, Mapping):
                for probe, val in probe_vals.items():
                    if probe in probes_dict:
                        probes_dict[probe][param] = val
                    else:
                        probes_dict[probe] = {param: val}
            else:
                params_common[param] = delistify(probe_vals)

        if probes_dict:
            # copy params_common to each probe
            return {probe: {**params_common, **probe_vals} for probe, probe_vals in probes_dict.items()}
        else:
            return {'*': {param: delistify(probe_vals) for param, probe_vals in param_dicts.items()}}

    probes_dict = group_dict_vals(cfg_many)

    # copy to exclude the possibility of next cycles be depended on changes in previous
    fields_can_change_all = [k for k in ('filter', 'process') if k in cfg]
    cfg_copy = {k: cfg[k].copy() for k in (fields_can_change_all + ['out'])}
    for group_in, cfg_in_cur in group_dict_vals(cfg_in_many, cfg['in'].get('probes', None)).items():
        cfg_in_cur['tables'] = [cfg_in_cur['table']]  # for h5.dispenser_and_names_gen()

        n_yields = 1
        for itbl, (tbl, coefs) in h5.dispenser_and_names_gen(
                cfg_in_cur, cfg['out'],
                fun_gen=fun_gen,
                b_close_at_end=False
                ):  # gets cfg['out']['db'] to write

            # recover initial cfg (not ['out']['db'/'b_remove_duplicates'] that h5.dispenser_and_names_gen() updates)
            for k in fields_can_change_all:
                cfg[k] = cfg_copy[k]
            cfg_in_copy = cfg_in_common.copy()
            cfg_in_copy['table'] = tbl
            if aggr := cfg['out'].get('aggregate_period'):  # if not any(re.findall('\d', cfg['out']['table'])):
                # output table name for averaged data:
                # - prefix is based on search pattern without digits and special characters used by regex
                # (we not using re.sub(r'[0-9]', '', tbl) as it will always separate devices)
                # - suffix "_bin{aggr}" indicates averaging
                cfg['out']['table'] = '{}_bin{}'.format(
                    cfg_copy['out']['table'] or re.sub(r'[\[\]\|\(\)./\\*+\d\-]', '',
                                                       re.sub(r'\\d', '', cfg_in_cur['table'])
                                                       ),
                    aggr
                    )
            # Model and number excluding type (w - wave gauges / i or incl - inclinometers)
            probe_number_str = re.search(re_probe_number, tbl).group('n')  # combined data are named by numbers joined by "_"
            if probe_number_str in probes_dict:
                cfg_cur = probes_dict[probe_number_str].copy()
                # copy() is need because this destructive for cfg_cur cycle can be run for other table of same probe

                # update not cfg['in'] fields that was set by plural-named fields - to use after the yielding
                for k1, k2 in (
                        ('filter', 'bad_p_at_bursts_starts_period'),
                        ('process', 'time_ranges_zeroing')
                        ):
                    try:
                        cfg[k1][k2] = cfg_cur.pop(k2)
                    except KeyError:
                        pass

                cfg_in_copy.update(cfg_cur)

            try:
                # Warning: Getting more/overwrite probe specific settings from "probes/{table}.yaml" files by Hydra
                # Be careful especially if update 'out' config
                cfg_hy = hydra.compose(overrides=[f"+probes={tbl}"])
                if cfg_hy:
                    with open_dict(cfg_hy):
                        cfg_hy.pop('probes')
                    if cfg_hy:
                        lf.info('Loaded YAML Hydra configuration for data table "{:s}" {}', tbl, cfg_hy)
                        for k, v in cfg_hy.items():
                            cfg_copy[k].update(v)
            except hydra.errors.MissingConfigException:
                pass

            # ``time_range`` must be specified before loading so filling with dumb values where no info.
            if cfg_in_copy['time_range'] is None and (cfg_in_copy['min_date'] or cfg_in_copy['max_date']):
                cfg_in_copy['time_range'] = [
                    cfg_in_copy['min_date'] or '2000-01-01',
                    cfg_in_copy['max_date'] or pd.Timestamp.now()]

            def yielding(d, msg=':'):
                """
                Yield with message
                :param d:
                :param msg:
                :return:
                changes the name of the output table from incl to i to reflect the fact that the data is processed there
                """
                lf.warning(' {:s}.{:d}-{:s}{:s}', group_in, itbl, tbl, msg)  # itbl
                # cfg_in_copy['tables'] = cfg_in_copy['table']  # recover (seems not need now but may be for future use)

                # copy to not cumulate the coefs corrections in several cycles of generator consumer for same probe:
                coefs_copy = coefs.copy() if coefs else None
                return d, coefs_copy, tbl.replace('incl', 'i'), probe_number_str

            with pd.HDFStore(cfg_in_copy['db_path'], mode='r') as store:
                if '.proc_noAvg' in cfg['in']['db_path'].suffixes[-2:-1] and not cfg['out']['b_split_by_time_ranges']:
                    # assume not need global filtering/get intervals for such input database
                    d = h5_load_range_by_coord(**cfg_in_copy, range_coordinates=None)
                    yield yielding(d)
                else:
                    # query and process several independent intervals
                    # Get index only and find indexes of data
                    try:
                        index_range, i0range, iq_edges = h5.coords(store, tbl, cfg_in_copy['time_range'])
                    except TypeError:  # skip empty nodes
                        # TypeError: cannot create a storer if the object is not existing nor a value are passed
                        lf.warning('Skipping {} without data table found', tbl)
                        continue
                    b_many_parts = len(iq_edges) > 2
                    for i_part, iq_edges_cur in enumerate(zip(iq_edges[::2], iq_edges[1::2])):
                        n_rows_to_load = -np.subtract(*iq_edges_cur)
                        if n_rows_to_load == 0:  # empty interval - will get errors in main circle
                            continue  # It is normal when used config with many intervals for different source databases
                        ddpart = h5_load_range_by_coord(**cfg_in_copy, range_coordinates=iq_edges_cur)
                        d, iburst = filt_data_dd(
                            ddpart, cfg_in_copy.get('dt_between_bursts'), cfg_in_copy.get('dt_hole_warning'),
                            cfg_in_copy
                            )
                        if index_range is not None:
                            yield yielding(d, msg='{}: {:%Y-%m-%d %H:%M:%S} – {:%m-%d %H:%M:%S}'.format(
                                f'.{i_part}' if b_many_parts else '', *index_range[iq_edges - i0range]
                                ))
                        else:
                            yield yielding(d)
                        n_yields += 1



# ---------------------------------------------------------------------------------------------------------------------

@hydra.main(config_name=cs_store_name, config_path='cfg', version_base='1.3')  # adds config store data/structure to :param config
def main(config: ConfigType) -> None:
    """
    Load data from hdf5 table (or group of tables)
    Calculate new data or average by specified interval
    Combine this data to new table
    :param config:
    :return:
    """
    # input:
    global cfg
    cfg = cfg_d.main_init(config, cs_store_name, __file__=None)
    cfg = cfg_d.main_init_input_file(cfg, cs_store_name, msg_action='Loading data from')
    lf.info('Begin {:s}(aggregate_period={:s})', this_prog_basename(__file__),
            cfg['out']['aggregate_period'] or 'None'
            )
    aggregate_period_timedelta = pd.Timedelta(pd.tseries.frequencies.to_offset(a)) if (
        a := cfg['out']['aggregate_period']) else None

    # minimum time between blocks, required in filt_data_dd() for data quality control messages:
    cfg['in']['dt_between_bursts'] = np.inf  # inf to not use bursts, None to autofind and repartition
    cfg['in']['dt_hole_warning'] = np.timedelta64(10, 'm')

    _db_path_proc_no_h5suffix = (
        cfg['in']['db_path'].with_suffix('') if '.proc_noAvg' in cfg['in']['db_path'].suffixes[-2:-1]
        # by default put proc DBs (with and without "_noAvg") 1 level higher than raw DB with raw's name with "proc"
        else cfg['in']['db_path'].parent.with_name(cfg['in']['db_path'].stem)  # path with '.raw' or '.proc_noAvg' suffix
        )
    if not cfg['out']['db_path']:
        cfg['out']['db_path'] = _db_path_proc_no_h5suffix.with_suffix(f'.proc.h5')

    h5.out_init(cfg['in'], cfg['out'])
    # cfg_out_table = cfg['out']['table']  # need? save because will need to change for h5.append()

    # will search / use this db:
    db_path_proc_noAvg = _db_path_proc_no_h5suffix.with_suffix('.proc_noAvg.h5')
    # If 'split_period' not set use custom splitting to be in memory limits
    if aggregate_period_timedelta:
        # Restricting number of counts=100000 in dask partition to not memory overflow
        split_for_memory = cfg['out']['split_period'] or next(iter(  # timedelta value of biggest unit component + 1
            [f'{t + 1}{c}' for t, c in zip(tuple((100000 * aggregate_period_timedelta).components), 'Dhms') if t > 0]
            ))

        if cfg['out']['text_date_format'].endswith('.%f') and not np.any(aggregate_period_timedelta.components[-3:]):
            # milliseconds=0, microseconds=0, nanoseconds=0
            cfg['out']['text_date_format'] = cfg['out']['text_date_format'][:-len('.%f')]

        if '.proc_noAvg' not in cfg['in']['db_path'].suffixes[-2:-1]:
            if db_path_proc_noAvg.is_file():
                # Changing input db and tables
                cfg['in']['tables'] = [tbl.replace('incl', 'i') for tbl in cfg['in']['tables']]  # can be useful only if need loading from processed tbl
                lf.info('Using found {}/{} db and its {} tables as source for averaging', *db_path_proc_noAvg.parts[-2:],
                        cfg['in']['tables'])
                cfg['in']['db_path'] = db_path_proc_noAvg

        if cfg['out']['text_path'] is None:
            cfg['out']['text_path'] = Path('text_output')
        cols_out_h5 = ['v', 'u', 'Pressure', 'Temp']                                # absent here cols will be ignored
    else:
        # Restricting dask partition size by time period
        split_for_memory = cfg['out']['split_period'] or pd.Timedelta(1, 'D')
        cfg['out']['aggregate_period'] = None  # 0 to None

        if not cfg['out']['not_joined_db_path']:
            cfg['out']['not_joined_db_path'] = db_path_proc_noAvg
        cols_out_h5 = ['v', 'u', 'Pressure', 'Temp', 'inclination']                 # absent here cols will be ignored

    if cfg['out']['text_path'] is not None and not cfg['out']['text_path'].is_absolute():
        cfg['out']['text_path'] = cfg['out']['db_path'].parent / cfg['out']['text_path']

    # Set columns from incl_calc_velocity() we need to prepend: Vabs/dir columns needed only to save them in txt
    if cfg['out']['text_path']:
        incl_calc_kwargs = {}  # all its cols
    else:
        _ = cols_out_h5.copy(); _.remove('Pressure'); _.remove('Temp')  # ('v', 'u', 'inclination')
        incl_calc_kwargs = {'cols_prepend': _}

    def map_to_suffixed(names, tbl, probe_number_str):
        """ Adds tbl suffix to output columns before accumulate in cycle for different tables"""
        suffix = f'{tbl[0]}{probe_number_str}'  #:02
        return {col: f'{col}_{suffix}' for col in names}

    # Filter [min/max][M] can be was specified with just key M - it is to set same value for keys Mx My Mz
    for lim in ['min', 'max']:
        if 'M' in cfg['filter'][lim]:
            for ch in ('x', 'y', 'z'):
                set_field_if_no(cfg['filter'][lim], f'M{ch}', cfg['filter'][lim]['M'])

    cfg['filter']['sleep_s'] = 0.5  # to run slower, helps for system memory management?

    log = {}
    dfs_all_list = []
    tbls = []
    dfs_all: Optional[pd.DataFrame] = None
    cfg['out']['tables_written'] = set()
    tables_written_not_joined = set()

    if cfg['program']['return_'] == '<cfg_before_cycle>':  # to help testing
        return cfg
    if cfg['program']['dask_scheduler'] == 'distributed':
        from dask.distributed import Client, progress
        # cluster = dask.distributed.LocalCluster(n_workers=2, threads_per_worker=1, memory_limit="5.5Gb")
        client = Client(processes=False)
        # navigate to http://localhost:8787/status to see the diagnostic dashboard if you have Bokeh installed
        # processes=False: avoide inter-worker communication for computations releases the GIL (numpy, da.array)  # without is error
    else:
        if cfg['program']['dask_scheduler'] == 'synchronous':
            lf.warning('using "synchronous" scheduler for debugging')
        import dask
        dask.config.set(scheduler=cfg['program']['dask_scheduler'])
        progress = None
        client = None

    tbl_prev = probe_number_str_prev = None
    for d, coefs, tbl, probe_number_str in gen_subconfigs(
            cfg,
            fun_gen=probes_gen,
            **cfg['in']
            ):
        d = filter_local(d, cfg['filter'], ignore_absent={'h_minus_1', 'g_minus_1'})  # d[['Mx','My','Mz']] = d[['Mx','My','Mz']].mask(lambda x: x>=4096)
        if '.raw' in cfg['in']['db_path'].suffixes[-2:-1]:
            # Zeroing
            if _ := cfg['process']['time_range_zeroing']:
                coefs['Rz'] = coef_zeroing_rotation_from_data(d, _, **coefs)
            # Azimuth correction
            if cfg['process']['azimuth_add']:
                # individual or the same correction for each table:
                coefs['azimuth_shift_deg'] += cfg['process']['azimuth_add']
                lf.warning('Azimuth correction updated ({}° added) to {}°',
                           coefs['azimuth_shift_deg'], cfg['process']['azimuth_add']
                           )

        if aggregate_period_timedelta:
            # Binning
            if '.proc_noAvg' not in cfg['in']['db_path'].suffixes[-2:-1]:
                lf.warning(str_warning := 'Raw data averaging before processing! Consider calculate physical parameters'
                           f" to {cfg['in']['db_path'].stem.removesuffix('.raw')}.proc_noAvg.h5 before averaging!")
                raise Ex_nothing_done(str_warning)  # comment to proceed

            if aggregate_period_timedelta > np.diff(d.divisions[0::(len(d.divisions)-1)]):  # ~ period > data time span
                # convert to one row dask dataframe with index = mean index
                df = d.mean().compute().to_frame().transpose()
                df.index = [d.index.compute().mean()]
                d = dd.from_pandas(df, npartitions=1)
            else:
                counts = (~d.iloc[:, 0].isna()).resample(aggregate_period_timedelta).count()
                d = d.resample(aggregate_period_timedelta,
                               # closed='right' if 'Pres' in cfg['in']['db_path'].stem else 'left'
                               # 'right' for burst mode because the last value of interval used in wavegauges is round - not true?
                               ).mean()
                # detect number of existed values in each resampled period (using 1st column)
                d = d.where(counts > counts[counts > 0].mean()/10
                            ).dropna(how='all')  # axis=0 is default
                try:  # persist speedups calc_velocity greatly but may require too many memory
                    lf.info('Persisting data aggregated by {:s}', cfg['out']['aggregate_period'])
                    d.persist()  # excludes missed values?
                except MemoryError:
                    lf.info('Persisting failed (not enough memory). Continue...')

            # Recalculating aggregated polar coordinates and angles that are invalid after the direct aggregating
            d = dekart2polar_df_uv(d)
            if cfg['out']['split_period']:  # for csv splitting only
                d = d.repartition(freq=cfg['out']['split_period'])
        else:                       # no averaging
            # Velocity calculation
            # --------------------
            if 'Rz' in coefs:
                for c in ['Ag', 'Ah']:
                    coefs[c] = coefs['Rz'] @ coefs[c]
            else:
                lf.info('No rotation coefficient (Rz) loaded. Suppose rotations already applied')
            d = incl_calc_velocity(  # with repartition for ascii splitting (also helps to prevent MemoryError)
                d.repartition(freq=split_for_memory), cfg_proc=cfg['process'],
                filt_max=cfg['filter']['max'], **coefs, **incl_calc_kwargs
            )
            d = calc_pressure(d.loc[:, [c for c in cols_out_h5 if c in d.columns]] if cfg['out']['text_path'] else d,
                              # no redundant cols if not kept to save to txt
                              **{(pb := 'bad_p_at_bursts_starts_period'): cfg['filter'][pb]},
                              **coefs
                              )
            # Write velocity to h5 - for each probe in separated table
            if cfg['out']['not_joined_db_path']:
                log = {
                    'Date0': d.divisions[0],
                    'DateEnd': d.divisions[-1],
                    'fileName': f"{cfg['in']['db_path'].parent.name}/{cfg['in']['db_path'].stem}"[
                        -cfg['out']['logfield_fileName_len']:],
                    'fileChangeTime': datetime.fromtimestamp(cfg['in']['db_path'].stat().st_mtime),
                    'rows': len(d)
                    }
                tables_written_not_joined |= (
                    h5.append_to(d, tbl, cfg['out'], log, msg=f'saving {tbl} to temporary store')
                )

        probe_continues = (tbl == tbl_prev and probe_number_str == probe_number_str_prev)
        tbl_prev = tbl; probe_number_str_prev = probe_number_str
        dd_to_csv(d, cfg['out']['text_path'], cfg['out']['text_date_format'], cfg['out']['text_columns'],
                  cfg['out']['aggregate_period'], suffix=f'@{tbl}', b_single_file=not cfg['out']['split_period'],
                  progress=progress, client=client)  #, mode='a' if probe_continues else 'r'

        # Combine data columns if we aggregate (in such case all data have index of equal period)
        if aggregate_period_timedelta:
            try:
                cols_save = [c for c in cols_out_h5 if c in d.columns]
                sleep(cfg['filter']['sleep_s'])
                Vne = d[cols_save].compute()  # MemoryError((1, 12400642), dtype('float64'))

                if not cfg['out']['b_all_to_one_col']:
                    Vne.rename(columns=map_to_suffixed(cols_save, tbl, probe_number_str), inplace=True)
                dfs_all_list.append(Vne)
                tbls.append(tbl)
            except Exception as e:
                lf.exception('Can not cumulate result! ')
                raise
                # todo: if low memory do it separately loading from temporary tables in chanks

        gc.collect()  # frees many memory. Helps to not crash

    # Combined data to hdf5
    #######################

    if aggregate_period_timedelta:
        # Concatenate several columns by one of method:
        # - consequently, like 1-probe data or
        # - parallel (add columns): dfs_all without any changes
        dfs_all = pd.concat(
            dfs_all_list, sort=True, axis=(0 if cfg['out']['b_all_to_one_col'] or probe_continues else 1)
            )
        dfs_all_log = pd.DataFrame(
            [df.index[[0, -1]].to_list() for df in dfs_all_list], columns=['Date0', 'DateEnd']
            ).assign(table_name=tbls) #.set_index('table_name')\
            #.sort_index() #\
        for after_remove_dup_index in [False, True]:
            try:
                cfg['out']['tables_written'] |= (
                    h5.append_to(dfs_all, cfg['out']['table'], cfg['out'], log=dfs_all_log,
                                 msg='Saving accumulated data'
                                 )
                    )
                break
            except ValueError as e:
                # ValueError: cannot reindex from a duplicate axis
                if dfs_all.index.is_unique and dfs_all_log.index.is_unique:
                    lf.exception('Can not understand problem. Skipping saving to hdf5!')
                    break
                lf.error('Removing duplicates in index')
                dfs_all = dfs_all.loc[~dfs_all.index.duplicated(keep='last')]
    h5.close(cfg['out'])  # close temporary output store
    if tables_written_not_joined:
        try:
            failed_storages = h5.move_tables(
                {**cfg['out'], 'db_path': cfg['out']['not_joined_db_path'], 'b_del_temp_db': False},
                tables_written_not_joined
                )
        except Ex_nothing_done as e:
            lf.warning('Tables {} of separate data not moved', tables_written_not_joined)
    if cfg['out']['tables_written']:
        try:
            failed_storages = h5.move_tables(cfg['out'], cfg['out']['tables_written'])
        except Ex_nothing_done as e:
            lf.warning('Tables {} of combined data not moved', cfg['out']['tables_written'])

        # Write concatenated dataframe to ascii (? with resample if b_all_to_one_col)
        if dfs_all is not None and len(dfs_all_list) > 1:
            call_with_valid_kwargs(
                dd_to_csv,
                (lambda x:
                    x.resample(rule=aggregate_period_timedelta)
                    .first() if cfg['out']['b_all_to_one_col'] else x
                 )(dd.from_pandas(dfs_all, chunksize=500000)),  # .fillna(0): absent values filling with 0  ???
                **cfg['out'],
                suffix=f"@{','.join(cfg['in']['tables'])}",
                progress=progress, client=client
                )

    print('Ok.', end=' ')

    # h5.index_sort(cfg['out'], out_storage_name=f"{cfg['out']['db_path'].stem}-resorted.h5", in_storages= failed_storages)
    # dd_out = dd.multi.concat(dfs_list, axis=1)

if __name__ == '__main__':
    main()



r"""
# old coefs uses:
da.degrees(da.arctan2(
(Gxyz[0, :] * Hxyz[1, :] - Gxyz[1, :] * Hxyz[0, :]) * (GsumMinus1 + 1),
Hxyz[2, :] * da.square(Gxyz[:-1, :]).sum(axis=0) - Gxyz[2, :] * ((Gxyz[:-1, :] * Hxyz[:-1, :]).sum(axis=0))))

else:
    Vdir = da.zeros_like(HsumMinus1)
    lf.warning('Bad Vdir: set all to 0 degrees')

a.drop(set(a.columns).difference(columns + [col]), axis=1)
for c, ar in zip(columns, arrays_list):
    # print(c, end=' ')
    if isinstance(ar, da.Array):
        a[c] = ar.to_dask_dataframe(index=a.index)
    else:
        a[c] = da.from_array(ar, chunks=GsumMinus1.chunks).to_dask_dataframe(index=a.index)
        #dd.from_array(ar, chunksize=int(np.ravel(GsumMinus1.chunksize)), columns=[c]).set_index(a.index) ...??? - not works

df = dd.from_dask_array(arrays, columns=columns, index=a.index)  # a.assign(dict(zip(columns, arrays?)))    #
if ('Pressure' in a.columns) or ('Temp' in a.columns):
    df.assign = df.join(a[[col]])

# Adding column of other (complex) type separatly
# why not works?: V = df['Vabs'] * da.cos(da.radians(df['Vdir'])) + 1j*da.sin(da.radians(df['Vdir']))  ().to_frame() # v + j*u # below is same in more steps
V = polar2dekart_complex(Vabs, Vdir)
V_dd = dd.from_dask_array(V, columns=['V'], index=a.index)
df = df.join(V_dd)

df = pd.DataFrame.from_records(dict(zip(columns, [Vabs, Vdir, np.degrees(incl_rad)])), columns=columns, index=tim)  # no sach method in dask



    dfs_all = pd.merge_asof(dfs_all, Vne, left_index=True, right_index=True,
                  tolerance=pd.Timedelta(cfg['out']['aggregate_period'] or '1ms'),
                            suffixes=('', ''), direction='nearest')
    dfs_all = pd.concat((Vne, how='outer')  #, rsuffix=tbl[-2:] join not works on dask
    V = df['V'].to_frame(name='V' + tbl[-2:]).compute()
if dfs_all is computed it is in memory:
mem = dfs_all.memory_usage().sum() / (1024 ** 2)
if mem > 50:
    lf.debug('{:.1g}Mb of data accumulated in memory '.format(mem))

df_to_csv(df, cfg_out, add_subdir='V,P_txt')
? h5_append_cols()
df_all = dd.merge(indiv, cm.reset_index(), on='cmte_id')


old cfg

    cfg = {  # how to load:
        'in': {
            'db_path': '/mnt/D/workData/BalticSea/181116inclinometer_Schuka/181116incl.h5', #r'd:\WorkData\BalticSea\181116inclinometer_Schuka\181116incl.h5',
            'tables': ['incl.*'],
            'chunksize': 1000000, # 'chunksize_percent': 10,  # we'll repace this with burst size if it suit
            'min_date': datetime.strptime('2018-11-16T15:19:00', '%Y-%m-%dT%H:%M:%S'),
            'max_date': datetime.strptime('2018-12-14T14:35:00', '%Y-%m-%dT%H:%M:%S')
            'split_period': '999D',  # pandas offset string (999D, H, ...) ['D' ]
            'aggregate_period': '2H',  # pandas offset string (D, 5D, H, ...)
            #'max_g_minus_1' used only to replace bad with NaN
        },
        'out': {
            'db_path': '181116incl.proc.h5',
            'table': 'V_incl',

    },
        'program': {
            'log': str(scripts_path / 'log/incl_h5clc.log'),
            'verbose': 'DEBUG'
        }
    }

    # optional external coef source:
    # cfg['out']['db_coef_path']           # defaut is same as 'db_path'
    # cfg['out']['table_coef'] = 'incl10'  # defaut is same as 'table'
"""
