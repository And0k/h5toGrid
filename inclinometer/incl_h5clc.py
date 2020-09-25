#!/usr/bin/env python3
# coding:utf-8
"""
  Author:  Andrey Korzh <ao.korzh@gmail.com>
  Purpose: Save syncronised averaged data to hdf5 tables
  Created: 01.03.2019

Load data from hdf5 table (or group of tables)
Calculate new data (averaging by specified interval)
Combine this data to one new table
"""

import gc
import logging
import re
import sys
from functools import wraps
from pathlib import Path
from time import sleep
from typing import Any, Callable, Dict, Iterator, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple, List, Union, TypeVar

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
from numba import njit, prange
from numba.extending import overload
from dask.diagnostics import ProgressBar

# my:
# allows to run on both my Linux and Windows systems:
scripts_path = Path(f"{'D:' if sys.platform == 'win32' else '/mnt/D'}/Work/_Python3/And0K/h5toGrid/scripts")
sys.path.append(str(scripts_path.parent.resolve()))
# sys.path.append( str(Path(__file__).parent.parent.resolve()) ) # os.getcwd()
# from utils2init import ini2dict
# from scripts.incl_calibr import calibrate, calibrate_plot, coef2str
# from other_filters import despike, rep2mean
from utils2init import Ex_nothing_done, set_field_if_no, init_logging, cfg_from_args, init_file_names, \
    my_argparser_common_part, this_prog_basename, dir_create_if_need
from utils_time import intervals_from_period, pd_period_to_timedelta
from to_pandas_hdf5.h5toh5 import h5init, h5find_tables, h5remove_table, h5move_tables
from to_pandas_hdf5.h5_dask_pandas import h5_append, h5q_intervals_indexes_gen, h5_load_range_by_coord, i_bursts_starts, \
    filt_blocks_da, filter_global_minmax, filter_local
from to_pandas_hdf5.csv2h5 import h5_dispenser_and_names_gen, h5_close
from other_filters import rep2mean
from inclinometer.h5inclinometer_coef import rot_matrix_x, rotate_y

if __name__ == '__main__':  # True:
    from dask.distributed import Client

    client = Client(
        processes=False)  # navigate to http://localhost:8787/status to see the diagnostic dashboard if you have Bokeh installed
    # processes=False: avoide inter-worker communication for computations releases the GIL (numpy, da.array)  # without is error
    from dask.distributed import progress  # or distributed.progress when using the distributed scheduler
else:
    progress = None

if __name__ == '__main__':
    l = None  # see main(): l = init_logging(logging, None, cfg['program']['log'], cfg['program']['verbose'])
else:
    l = logging.getLogger(__name__)
    # level_console = 'INFO'
    # level_file = None
    # # set up logging to console
    # console = logging.StreamHandler()
    # console.setLevel(level_console if level_console else 'INFO' if level_file != 'DEBUG' else 'DEBUG')  # logging.WARN
    # # set a format which is simpler for console use
    # formatter = logging.Formatter('%(message)s')  # %(name)-12s: %(levelname)-8s ...
    # console.setFormatter(formatter)
    # l.addHandler(console)

prog = 'incl_h5clc'
version = '0.1.1'


def my_argparser(varargs=None):
    """

    :return p: configargparse object of parameters
    """
    if not varargs:
        varargs = {}

    varargs.setdefault('description', f'{prog} version {version}' + """
    ---------------------------------
    Load data from hdf5 table (or group of tables)
    Calculate new data (averaging by specified interval)
    Combine this data to new specified table
    ---------------------------------
    """)
    p = my_argparser_common_part(varargs, version)

    # Fill configuration sections
    # All argumets of type str (default for add_argument...), because of
    # custom postprocing based of my_argparser names in ini2dict

    s = p.add_argument_group('in',
                             'Parameters of input files')
    s.add('--db_path', default='*.h5',
             help='path to pytables hdf5 store to load data. May use patterns in Unix shell style')
    s.add('--tables_list', default='incl.*',
             help='table names in hdf5 store to get data. Uses regexp if only one table name')
    s.add('--chunksize_int', help='limit loading data in memory', default='50000')
    s.add('--date_min', help='time range min to use')
    s.add('--date_max', help='time range max to use')
    s.add('--raw_dir_words_list',
               help='cruise directories to search in in.db_path to set path of out.db_path under it if out.db_path is not absolute')

    s = p.add_argument_group('filter',
                             'Filter all data based on min/max of parameters')
    s.add('--max_g_minus_1_float', default='1',
              help='sets Vabs to NaN if module of acceleration is greater')
    s.add('--max_h_minus_1_float', default='8',
              help='sets Vdir to zero if module of magnetic field is greater')
    s.add('--min_dict',
              help='List with items in  "key:value" format. Filter out (set to NaN) data of ``key`` columns if it is below ``value``')
    s.add('--max_dict',
              help='List with items in  "key:value" format. Filter out data of ``key`` columns if it is above ``value``')
    s.add('--bad_p_at_bursts_starts_peroiod',
               help='pandas offset string. If set then marks each 2 samples of Pressure at start of burst as bad')


    s = p.add_argument_group('out',
                             'Parameters of output files')
    s.add('--out.db_path', help='hdf5 store file path')
    s.add('--not_joined_h5_path',
              help='If something then saving proced velocity for each probe individually to out.db_path. Todo: use this settings to can save in other path')
    s.add('--table',
              help='table name in hdf5 store to write data. If not specified then will be generated on base of path of input files. Note: "*" is used to write blocks in autonumbered locations (see dask to_hdf())')
    s.add('--split_period',
             help='pandas offset string (5D, H, ...) to proc and output in separate blocks. Use big values to not split. If saves to csv then writes in parts of this length, but if no aggregate_period_s only')
    s.add('--aggregate_period',
             help='pandas offset string (D, H, 2S, ...) to bin data and thus reduce output size')
    s.add('--text_path',
              help='path to save text files with processed velocity (each probe individually)')
    s.add('--text_date_format', default='%Y-%m-%d %H:%M:%S.%f',
              help='Format of date column in output text files. Can use float or string representations')
    s.add('--text_columns_list',
              help='if not empty then saved text files will contain only specified here columns')
    s.add('--b_all_to_one_col',
              help='concatenate all data in same columns in out db, both separated and joined text files will be written')
    s.add('--b_del_temp_db', default='False', help='temporary h5 file will be deleted after operation')


    s = p.add_argument_group('proc',
                             'Processing parameters')
    s.add('--timerange_zeroing_list',
             help='if specified then rotate data in this interval such that it will have min mean pitch and roll, display "info" warning about')
    s.add('--timerange_zeroing_dict',
             help='{table: [start, end]}, rotate data in this interval only for specified probe number(s) data such that it will have min mean pitch and roll, the about "info" warning will be displayed. Probe number is int number consisted of digits in table name')
    s.add('--azimuth_add_float', help='degrees, adds this value to velocity direction (will sum with _azimuth_shift_deg_ coef)')
    s.add('--calc_version', default='trigonometric(incl)',
               help='string: variant of processing Vabs(inclination):',
               choices=['trigonometric(incl)', 'polynom(force)'])
    s.add('--max_incl_of_fit_deg_float',
               help='Finds point where g(x) = Vabs(inclination) became bend down and replaces after g with line so after max_incl_of_fit_deg {\\Delta}^{2}y ≥ 0 for x > max_incl_of_fit_deg')

    s = p.add_argument_group('program',
                             'Program behaviour')
    s.add_argument('--return', default='<end>', choices=['<return_cfg>', '<return_cfg_with_options>'],
                           help='executes part of code and returns parameters after skipping of some code')
    s.add_argument('--b_interact', default='False',
                           help='ask showing source files names before process them')

    return (p)


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
    return -np.arctan(Gxyz[0, :] / np.linalg.norm(Gxyz[1:, :]))
    # =arctan2(Gxyz[0,:], sqrt(square(Gxyz[1,:])+square(Gxyz[2,:])) )')


@allow_dask
def f_roll(Gxyz):
    """
    Roll calculating
    :param Gxyz: shape = (3,len) Accelerometer data
    :return: angle, radians, shape = (len,)
    """
    return np.arctan2(Gxyz[1, :], Gxyz[2, :])


#@njit - removed because numba gets long bytecode dump without messages/errors
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
    force[is_nans] = np.NaN
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
        l.warning('Old coefs method "polynom(force)"')
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


# @allow_dask not need because not used explicit np/da lib references
#@njit("f8[:,:](f8[:,:], f8[:,:], f8[:,:])") - failed for dask array
def fG(Axyz: Union[np.ndarray, da.Array],
       Ag: Union[np.ndarray, da.Array],
       Cg: Union[np.ndarray, da.Array]) -> Union[np.ndarray, da.Array]:
    """
    Allows use of transposed Cg
    :param Axyz:
    :param Ag: scaling coef
    :param Cg: shift coef
    :return:
    """
    assert Ag.any(), 'Ag coefficients all zeros!!!'
    return Ag @ (Axyz - (Cg if Cg.shape[0] == Ag.shape[0] else Cg.T))


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
    :return: list [Vn, Ve] - list (not tuple) is used because it is need to concatenate with other data
    """
    return [Vabs * np.cos(np.radians(Vdir)), Vabs * np.sin(np.radians(Vdir))]


# @allow_dask
# def dekart2polar(v_en):
#     """
#     Not Tested
#     :param Ve:
#     :param Vn:
#     :return: [Vabs, Vdir]
#     """
#     return np.linalg.norm(v_en, axis=0), np.degrees(np.arctan2(*v_en))

def dekart2polar_df_v_en(df, **kwargs):
    """

    :param d: if no columns Ve and Vn remains unchanged
    :**kwargs :'inplace' not supported in dask. dumn it!
    :return: [Vabs, Vdir] series
    """

    # why da.linalg.norm(df.loc[:, ['Ve','Vn']].values, axis=1) gives numpy (not dask) array?
    # da.degrees(df.eval('arctan2(Ve, Vn)')))

    if 'Ve' in df.columns:

        kdegrees = 180 / np.pi

        return df.eval(f"""
        Vabs = sqrt(Ve**2 + Vn**2)
        Vdir = arctan2(Ve, Vn)*{kdegrees:.20}
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
    l.info('calculating V')
    if kVabs == (1, 0):
        l.warning('kVabs == (1, 0)! => V = sqrt(sin(inclination))')
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
                print('Acceleration is bad in {}% points!'.format(100 * bad_g_sum / len(GsumMinus1)))
            incl_rad[bad_g] = np.NaN

        Vabs = v_abs_from_incl(incl_rad, kVabs, cfg_proc['calc_version'], cfg_proc['max_incl_of_fit_deg'])

        Hxyz = fG(a.loc[:, ('Mx', 'My', 'Mz')].to_numpy().T, Ah, Ch)
        Vdir = azimuth_shift_deg - da.degrees(da.arctan2(
            (Gxyz[0, :] * Hxyz[1, :] - Gxyz[1, :] * Hxyz[0, :]) * (GsumMinus1 + 1),
            Hxyz[2, :] * da.square(Gxyz[:-1, :]).sum(axis=0) - Gxyz[2, :] * (
                (Gxyz[:-1, :] * Hxyz[:-1, :]).sum(axis=0))
            ))

        col = 'Pressure' if ('Pressure' in a.columns) else 'Temp' if ('Temp' in a.columns) else []
        columns = ['Vabs', 'Vdir', 'Vn', 'Ve', 'inclination']
        arrays_list = [Vabs, Vdir] + polar2dekart(Vabs, Vdir) + [da.degrees(incl_rad)]
        a = a.assign(**{c: ar for c, ar in zip(columns, arrays_list)})  # a[c] = ar

        # df = pd.DataFrame.from_records(dict(zip(columns, [Vabs, Vdir, np.degrees(incl_rad)])), columns=columns, index=a.index)  # no sach method in dask
        return a[columns + [col]]
    except Exception as e:
        l.exception('Error in incl_calc_velocity():')
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


def recover_magnetometer_x(Mcnts, Ah, Ch, cfg_filter, len_data):
    Hxyz = fG(Mcnts, Ah, Ch)  # #x.rechunk({0: -1, 1: 'auto'}, block_size_limit=1e8)
    HsumMinus1 = da.linalg.norm(Hxyz, axis=0) - 1  # should be close to zero

    # Channel x recovering
    bad = da.isnan(Mcnts[0, :])
    need_recover_mask = da.isfinite(Mcnts[1:, :]).any(axis=0)  # where other channels ok
    sleep(cfg_filter['sleep_s'])
    can_recover = need_recover_mask.sum(axis=0).compute()
    if can_recover:
        Mcnts_list = [[], [], []]
        need_recover_mask &= bad  # only where x is bad
        sleep(cfg_filter['sleep_s'])
        need_recover = need_recover_mask.sum(axis=0).compute()
        l.info('Magnetometer x channel %s: %d bad where y&z is ok. y&z ok in %d/%d',
               'recovering' if need_recover else 'checked - ok', need_recover, can_recover, len_data)
        if need_recover:  # have poins where recover is needed and is posible

            # Try to recover mean_Hsum (should be close to 1)
            mean_HsumMinus1 = np.nanmedian(
                (HsumMinus1[HsumMinus1 < cfg_filter['max_h_minus_1']]).compute()
                )

            if np.isnan(mean_HsumMinus1) or (np.fabs(mean_HsumMinus1) > 0.5 and need_recover / len_data > 0.95):
                l.warning('mean_Hsum is mostly bad (mean=%g), most of data need to be recovered (%g%%) so no trust it'
                          ' at all. Recovering all x-ch.data with setting mean_Hsum = 1',
                          mean_HsumMinus1, 100 * need_recover / len_data)
                bad = da.ones_like(HsumMinus1,
                                   dtype=np.bool8)  # need recover all x points because too small points with good HsumMinus1
                mean_HsumMinus1 = 0
            else:
                l.warning('calculated mean_Hsum - 1 is good (close to 0): mean=%s', mean_HsumMinus1)

            # Recover magnetometer's x channel

            # todo: not use this channel but recover dir instantly
            # da.apply_along_axis(lambda x: da.where(x < 0, 0, da.sqrt(abs(x))),
            #                               0,
            #                  mean_Hsum - da.square(Ah[2, 2] * (rep2mean_da(Mcnts[2,:], Mcnts[2,:] > 0) - Ch[2])) -
            #                               da.square(Ah[1, 1] * (rep2mean_da(Mcnts[1,:], Mcnts[1,:] > 0) - Ch[1]))
            #                               )


            #Mcnts_x_recover = recover_x__sympy_lambdify(Mcnts[1, :], Mcnts[2, :], Ah, Ch, mean_Hsum=mean_HsumMinus1 + 1)
            # replaced to this to use numba:
            Mcnts_x_recover = da.map_blocks(recover_x__sympy_lambdify, Mcnts[1, :], Mcnts[2, :],
                                            Ah=Ah, Ch=Ch, mean_Hsum=mean_HsumMinus1 + 1, dtype=np.float64, meta=np.float64([]))

            Mcnts_list[0] = da.where(need_recover_mask, Mcnts_x_recover, Mcnts[0, :])
            bad &= ~need_recover_mask

            # other points recover by interp
            Mcnts_list[0] = da.from_array(rep2mean_da2np(Mcnts_list[0], ~bad), chunks=Mcnts_list[0].chunks,
                                          name='Mcnts_list[0]')
        else:
            Mcnts_list[0] = Mcnts[0, :]

        l.debug('interpolating magnetometer data using neighbor points separately for each channel...')
        need_recover_mask = da.ones_like(HsumMinus1)  # here save where Vdir can not recover
        for ch, i in [('x', 0), ('y', 1), ('z', 2)]:  # in ([('y', 1), ('z', 2)] if need_recover else
            print(ch, end=' ')
            if (ch != 'x') or not need_recover:
                Mcnts_list[i] = Mcnts[i, :]
            bad = da.isnan(Mcnts_list[i])
            #sleep(cfg_filter['sleep_s'])
            n_bad = bad.sum(axis=0).compute()  # exits with "Process finished with exit code -1073741819 (0xC0000005)"!
            if n_bad:
                n_good = HsumMinus1.shape[0] - n_bad
                if n_good / n_bad > 0.01:
                    l.info(f'channel {ch}: bad points: {n_bad} - recovering using nearest good points ({n_good})')
                    Mcnts_list[i] = da.from_array(rep2mean_da2np(Mcnts_list[i], ~bad), chunks=Mcnts_list[0].chunks,
                                                  name=f'Mcnts_list[{ch}]-all_is_finite')
                else:
                    l.warning(f'channel {ch}: bad points: {n_bad} - will not recover because too small good points ({n_good})')
                    Mcnts_list[i] = np.NaN + da.empty_like(HsumMinus1)
                    need_recover_mask[bad] = False

        Mcnts = da.vstack(Mcnts_list)
        Hxyz = fG(Mcnts, Ah, Ch)  # #x.rechunk({0: -1, 1: 'auto'}, block_size_limit=1e8)

    else:
        l.info('Magnetometer can not be recovered')
        need_recover_mask = None

    return Hxyz, need_recover_mask


def rep2mean_da(y: da.Array, bOk=None, x=None) -> da.Array:
    """
    Interpolates bad values (inverce of bOk) in each dask block.
    Note: can leave NaNs if no good data in block
    :param y:
    :param bOk:
    :param x:
    :return: dask array of np.float64 values

    g = da.overlap.overlap(x, depth={0: 2, 1: 2},
... boundary={0: 'periodic', 1: 'periodic'})
>>> g2 = g.map_blocks(myfunc)
>>> result = da.overlap.trim_internal(g2, {0: 2, 1: 2})     # todo it
    """
    return y.map_blocks(rep2mean, bOk, x, dtype=np.float64, meta=np.float64([]))


def rep2mean_da2np(y: da.Array, bOk=None, x=None) -> np.ndarray:
    """
    same as rep2mean_da but also replaces bad block values with constant (rep2mean_da can not repldace bad if no good)
    :param y:
    :param bOk:
    :param x: None, other not implemented
    :return: numpy array of np.float64 values

    """

    y_last = None
    y_out_list = []
    for y_bl, b_ok in zip(y.blocks, (da.isfinite(y) & bOk).blocks):
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


def incl_calc_velocity(a: dd.DataFrame,
                       Ag: Optional[np.ndarray] = None, Cg: Optional[np.ndarray] = None,
                       Ah: Optional[np.ndarray] = None, Ch: Optional[np.ndarray] = None,
                       kVabs: Optional[np.ndarray] = None, azimuth_shift_deg: Optional[float] = None,
                       P: Optional[np.ndarray] = None,
                       cfg_filter: Optional[Mapping[str, Any]] = None,
                       cfg_proc: Optional[Mapping[str, Any]] = None,
                       **kwargs) -> dd.DataFrame:
    """
    Calculates dataframe with velocity vector module and direction. Also replaces P column by Pressure applying polyval() to P if it exists.
    :param a: dask dataframe
    Coefficients:
    :param Ag: coef
    :param Cg:
    :param Ah:
    :param Ch:
    :param kVabs: if None then will not try to calc velocity
    :param azimuth_shift_deg:
    :param P:
    :param cfg_filter: dict. with fields:
        max_g_minus_1: useed to check module of Gxyz,
        max_h_minus_1: to set Vdir=0 and...
    :param cfg_proc: 'calc_version', 'max_incl_of_fit_deg'
    :param kwargs: other arguments not affects calculation
    :return: dataframe withcolumns ['Vabs', 'Vdir', col, 'inclination'] where col is additional column in _a_, or may be absent
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

    if kVabs is not None:
        l.info('calculating V')
        try:    # lengths=True gets MemoryError   #.to_dask_array()?, dd.from_pandas?
            Gxyz = fG(a.loc[:, ('Ax', 'Ay', 'Az')].to_dask_array(lengths=lengths).T, Ag, Cg)

            # .rechunk((1800, 3))
            # filter
            GsumMinus1 = da.linalg.norm(Gxyz, axis=0) - 1  # should be close to zero
            incl_rad = fInclination(Gxyz)  # .compute()

            if 'max_g_minus_1' in cfg_filter:
                bad = np.fabs(GsumMinus1) > cfg_filter['max_g_minus_1']  # .compute()
                bad_g_sum = bad.sum(axis=0).compute()
                if bad_g_sum:
                    if bad_g_sum > 0.1 * len(GsumMinus1):  # do not message for few points
                        l.warning('Acceleration is bad in %g%% points!', 100 * bad_g_sum / len(GsumMinus1))
                    incl_rad[bad] = np.NaN
            # else:
            #     bad = da.zeros_like(GsumMinus1, np.bool8)

            # l.debug('{:.1g}Mb of data accumulated in memory '.format(dfs_all.memory_usage().sum() / (1024 * 1024)))

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
            # Vabs = np.polyval(kVabs, np.where(bad, np.NaN, Gxyz))
            # Vn = Vabs * np.cos(np.radians(Vdir))
            # Ve = Vabs * np.sin(np.radians(Vdir))

            Hxyz, need_recover_mask = recover_magnetometer_x(
                a.loc[:, ('Mx', 'My', 'Mz')].to_dask_array(lengths=lengths).T, Ah, Ch, cfg_filter, len_data)
            if need_recover_mask is not None:
                HsumMinus1 = da.linalg.norm(Hxyz, axis=0) - 1  # should be close to zero
                Vdir = 0  # default value
                # bad = ~da.any(da.isnan(Mcnts), axis=0)
                Vdir = da.where(da.logical_or(need_recover_mask, HsumMinus1 < cfg_filter['max_h_minus_1']),
                                azimuth_shift_deg - da.degrees(da.arctan2(
                                    (Gxyz[0, :] * Hxyz[1, :] - Gxyz[1, :] * Hxyz[0, :]) * (GsumMinus1 + 1),
                                    Hxyz[2, :] * da.square(Gxyz[:-1, :]).sum(axis=0) - Gxyz[2, :] * (
                                            Gxyz[:-1, :] * Hxyz[:-1, :]).sum(axis=0)
                                    )),
                                Vdir  # default value
                                )
            else:  # Set magnetometer data to be function of accelerometer data - allows calc waves parameters
                l.warning(
                    'Bad magnetometer data => Assign direction inversely proportional to toolface angle (~ relative angle if no rotations around device axis)')
                Vdir = azimuth_shift_deg - da.degrees(da.arctan2(Gxyz[0, :], Gxyz[1, :]))
            Vdir = Vdir.flatten()

            columns = ['Vabs', 'Vdir', 'Vn', 'Ve', 'inclination']
            arrays_list = [Vabs, Vdir] + polar2dekart(Vabs, Vdir) + [da.degrees(incl_rad)]  # da.stack(, axis=1)

            a = a.assign(**{c: (ar
                                if isinstance(ar, da.Array) else
                                da.from_array(ar, chunks=GsumMinus1.chunks)
                                ).to_dask_dataframe(index=a.index) for c, ar in zip(columns, arrays_list)})  # a[c] = ar
        except Exception as e:
            l.exception('Error in incl_calc_velocity():')
            raise
    else:
        columns = []

    if 'Temp' in a.columns:
        columns.append('Temp')

    if (P is not None) and 'P' in a.columns:
        # Calculate pressure using P polynom
        meta = ('Pressure', 'f8')

        if cfg_filter.get['bad_p_at_bursts_starts_peroiod']:   # '1h'
            p_bursts = a.P.repartition(freq=cfg_filter['bad_p_at_bursts_starts_peroiod'])  # bursts must starts at beginnings of hours

            def calc_and_rem2first(p: pd.Series) -> pd.Series:
                """ mark bad data in first samples of burst"""
                # df.iloc[0:1, df.columns.get_loc('P')]=0  # not works!
                pressure = np.polyval(P, p.values)
                pressure[:2] = np.NaN
                p[:] = pressure
                return p

            a = a.assign(**{'Pressure': p_bursts.map_partitions(calc_and_rem2first, meta=meta)})
        else:
            a = a.assign(**{'Pressure': a.P.map_partitions(lambda x: np.polyval(P, x), meta=meta)})

    if 'Pressure' in a.columns:
        columns.append('Pressure')

        # old coefs uses: da.degrees(da.arctan2(
        # (Gxyz[0, :] * Hxyz[1, :] - Gxyz[1, :] * Hxyz[0, :]) * (GsumMinus1 + 1),
        # Hxyz[2, :] * da.square(Gxyz[:-1, :]).sum(axis=0) - Gxyz[2, :] * ((Gxyz[:-1, :] * Hxyz[:-1, :]).sum(axis=0))))
        #
        # else:
        #     Vdir = da.zeros_like(HsumMinus1)
        #     l.warning('Bad Vdir: set all to 0 degrees')

        # a.drop(set(a.columns).difference(columns + [col]), axis=1)
        # for c, ar in zip(columns, arrays_list):
        #     # print(c, end=' ')
        #     if isinstance(ar, da.Array):
        #         a[c] = ar.to_dask_dataframe(index=a.index)
        #     else:
        #         a[c] = da.from_array(ar, chunks=GsumMinus1.chunks).to_dask_dataframe(index=a.index)
        #         #dd.from_array(ar, chunksize=int(np.ravel(GsumMinus1.chunksize)), columns=[c]).set_index(a.index) ...??? - not works

        # df = dd.from_dask_array(arrays, columns=columns, index=a.index)  # a.assign(dict(zip(columns, arrays?)))    #
        # if ('Pressure' in a.columns) or ('Temp' in a.columns):
        #     df.assign = df.join(a[[col]])

        # # Adding column of other (complex) type separatly
        # # why not works?: V = df['Vabs'] * da.cos(da.radians(df['Vdir'])) + 1j*da.sin(da.radians(df['Vdir']))  ().to_frame() # Vn + j*Ve # below is same in more steps
        # V = polar2dekart_complex(Vabs, Vdir)
        # V_dd = dd.from_dask_array(V, columns=['V'], index=a.index)
        # df = df.join(V_dd)

        # df = pd.DataFrame.from_records(dict(zip(columns, [Vabs, Vdir, np.degrees(incl_rad)])), columns=columns, index=tim)  # no sach method in dask
    return a[columns]


def coef_zeroing(mean_countsG0, Ag_old, Cg, Ah_old):
    """
    Zeroing: correct Ag_old, Ah_old
    :param mean_countsG0: 1x3 values of columns 'Ax','Ay','Az' (it is practical to provide mean values of some time range)
    :param Ag_old, Cg: numpy.arrays, rotation matrix and shift for accelerometer
    :param Ah_old: numpy.array 3x3, rotation matrix for magnetometer
    return (Ag, Ah): numpy.arrays (3x3, 3x3), corrected rotation matrices

    Methond of calculation of ``mean_countsG0`` from dask dataframe data:
     mean_countsG0 = da.atleast_2d(da.from_delayed(
          a_zeroing.loc[:, ('Ax', 'Ay', 'Az')].mean(
             ).values.to_delayed()[0], shape=(3,), dtype=np.float64, name='mean_G0'))
     """

    if not len(mean_countsG0):
        print(f'zeroing(): no data {mean_countsG0}, returning same coef')
        return Ag_old, Ah_old
    if mean_countsG0.shape[0] != 3:
        raise ValueError('Bad mean_countsG0 shape')

    Gxyz0old = fG(mean_countsG0, Ag_old, Cg)
    old1pitch = f_pitch(Gxyz0old)
    old1roll = f_roll(Gxyz0old)
    l.info('zeroing pitch = %s, roll = %s degrees', *np.rad2deg([old1pitch[0], old1roll[0]]))
    Rcor = rotate_y(
        rot_matrix_x(np.cos(old1roll), np.sin(old1roll)),
        angle_rad=old1pitch)

    Ag = Rcor @ Ag_old
    Ah = Rcor @ Ah_old
    l.debug('calibrated Ag = %s,\n Ah = %s', Ag, Ah)
    # # test: should be close to zero:
    # Gxyz0 = fG(mean_countsG0, Ag, Cg)
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
    #         da.flatnonzero(b_ok), median3(x[b]), da.NaN, da.NaN)
    #
    # b = da.from_array(b_ok, chunks=chunks, meta=('Tfilt', 'f8'))
    # with ProgressBar():
    #     Tfilt = da.map_blocks(interp_after_median3(x, b), y, b).compute()

    # hangs:
    # Tfilt = dd.map_partitions(interp_after_median3, a['Temp'], da.from_array(b_ok, chunks=cfg_out['chunksize']), meta=('Tfilt', 'f8')).compute()

    # Tfilt = np.interp(da.arange(len(b_ok)), da.flatnonzero(b_ok), median3(a['Temp'][b_ok]), da.NaN,da.NaN)
    # @+node:korzh.20180524213634.8: *3* main
    # @+others
    # @-others


def filt_data_dd(a, cfg_in: Mapping[str, Any]) -> Tuple[dd.DataFrame, np.array]:
    """
    Also gets burst starts (i.e. finds gaps in data)
    :param a:
    :param cfg_in: dict with fields:
     - burst_min
     - min_p, optional:
    :return: (a, i_burst) where:
     - a: filtered,
     - i_burst: array with 1st elem 0 and other - starts of data after big time holes

    """
    if True:  # try:
        # determine indexes of bursts starts
        tim = a.index.compute()  # History: MemoryError((6, 12275998), dtype('float64'))
        i_burst, mean_burst_size = i_bursts_starts(tim, dt_between_blocks=cfg_in['burst_min'])

        # filter
        if 'P' in a.columns and cfg_in.get('min_p'):
            print('restricting time range by good Pressure')
            # interp(NaNs) - removes warning 'invalid value encountered in less':
            a['P'] = filt_blocks_da(a['P'].values, i_burst, i_end=len(a)).to_dask_dataframe(['P'], index=tim)
            # todo: make filt_blocks_dd and replace filt_blocks_da: use a['P'] = a['P'].repartition(chunks=(tuple(np.diff(i_starts).tolist()),))...?
        # decrease interval based on ini date settings and filtering and recalc bursts
        a = filter_global_minmax(a, cfg_filter=cfg_in)
        tim = a.index.compute()  # History: MemoryError((6, 10868966), dtype('float64'))
        i_burst, mean_burst_size = i_bursts_starts(tim, dt_between_blocks=cfg_in['burst_min'])
        # or use this and check dup? shift?:
        # i_good = np.search_sorted(tim, a.index.compute())
        # i_burst = np.search_sorted(i_good, i_burst)

        if not a.known_divisions:  # this is usually required for next op
            divisions = tuple(tim[np.append(i_burst, len(tim) - 1)])
            a.set_index(a.index, sorted=True).repartition(divisions=divisions)
            1
            # a = a.set_index(a.index, divisions=divisions, sorted=True)  # repartition? (or reset_index)
        return a, i_burst


def gen_data_on_intervals(cfg_in: Mapping[str, Any], split_period=None) -> Iterator[Tuple[dd.DataFrame, np.array]]:
    """
    Loading data of specified timerange
    :param cfg_in: must have fiels:
        fields required by intervals_from_period() except _split_period_ which is defined by 2nd input arg:
        split_period:
        other fields required by h5_load_range_by_coord():
            db_path, ...
    :return:
    """
    t_prev_interval_start, t_intervals_start = intervals_from_period(**cfg_in, period=split_period)
    for start_end in h5q_intervals_indexes_gen(cfg_in, t_prev_interval_start, t_intervals_start):
        a = h5_load_range_by_coord(cfg_in, start_end)
        yield filt_data_dd(a, cfg_in)


def h5_names_gen(cfg_in: Mapping[str, Any], cfg_out: None = None
                 ) -> Iterator[Tuple[str, Tuple[Any, ...]]]:
    """
    Generate table names with associated coefficients
    :param cfg_in: dict with fields:
      - tables: pattern to find names
      - db_path:
    :param cfg_out: not used but kept for the requirement of h5_dispenser_and_names_gen() argument
    :return: iterator that returns (table name, coefficients)
    updates cfg_in['tables'] - sets to list of found tables in store
    """
    # dfLogOld = h5temp_open(cfg_out), cfg_out['db'].close()
    with pd.HDFStore(cfg_in['db_path'], mode='r') as store:
        if len(cfg_in['tables']) == 1:
            cfg_in['tables'] = h5find_tables(store, cfg_in['tables'][0])

        if cfg_in['db_path'].stem.endswith('proc_noAvg'):
            # Loading already processed data
            for tbl in cfg_in['tables']:
                yield (tbl, None)
        else:
            for tbl in cfg_in['tables']:
                # if int(tbl[-2:]) in {5,9,10,11,14,20}:
                coefs_dict = {}
                # Finds up to 2 levels of coefficients, naming rule gives coefs this names (but accepts any paths):
                # coefs: ['coef/G/A', 'coef/G/C', 'coef/H/A', 'coef/H/C', 'coef/H/azimuth_shift_deg', 'coef/Vabs0'])
                # names: ['Ag', 'Cg', 'Ah', 'Ch', 'azimuth_shift_deg', 'kVabs'],

                node_coef = store.get_node(f'{tbl}/coef')
                if node_coef is None:
                    l.warning('Skipping this table "%s" - not found coefs!', tbl)
                    continue
                for node_name in node_coef.__members__:
                    node_coef_l2 = node_coef[node_name]
                    if getattr(node_coef_l2, '__members__', False):  # node_coef_l2 is group
                        for node_name_l2 in node_coef_l2.__members__:
                            name = f'{node_name_l2}{node_name.lower() if node_name_l2[-1].isupper() else ""}'
                            coefs_dict[name] = node_coef_l2[node_name_l2].read()
                    else:  # node_coef_l2 is value
                        coefs_dict[node_name if node_name != 'Vabs0' else 'kVabs'] = node_coef_l2.read()
                yield (tbl, coefs_dict)
    return


def h5_append_to(dfs: Union[pd.DataFrame, dd.DataFrame],
                 tbl: str,
                 cfg_out: Mapping[str, Any],
                 log: Optional[Mapping[str, Any]] = None,
                 msg: Optional[str] = None,
                 print_ok: bool = None):
    """ Trying to remove table then append data - useful for tables that are wrote in one short"""
    if dfs is not None:
        if msg: l.info(msg)
        tables_dict = {'table': tbl, 'table_log': f'{tbl}/logFiles'}
        try:
            h5remove_table(cfg_out, tbl)
        except Exception as e:  # no such table?
            pass
        h5_append({**cfg_out, **tables_dict}, dfs,
                  {} if log is None else log)  # , cfg_out['log'], log_dt_from_utc=cfg_in['dt_from_utc'], 'tables': None, 'tables_log': None
        # dfs_all.to_hdf(cfg_out['db_path'], tbl, append=True, format='table', compute=True)
        if print_ok: print(print_ok, end=' ')
        return tables_dict.values()
    else:
        print('No data.', end=' ')


def gen_variables(cfg_in: MutableMapping[str, Any], cfg_out: MutableMapping[str, Any], fun_gen=h5_names_gen
            ) -> Iterator[Tuple[dd.DataFrame, np.array]]:
    """
    Wraps h5_dispenser_and_names_gen() to deal with many db_paths, tables, dates_min and dates_max
    :param cfg_in: dict with fields:
      - db_path: str/path of real path or multiple paths joined by '|' to be splitted in list cfg['db_paths'],
      - tables,
      - dates_min,
      - dates_max
      - split_period, optional - generate dataframes in parts with this period
    :param cfg_out: dict with fields:

    :param fun_gen:
    :return: dict with plural named keys having list values
    """

    names_many_to_one = {
        'db_paths': 'db_path',
        'tables': 'table',
        'dates_min': 'date_min',
        'dates_max': 'date_max'}

    def gen_sources_dict(cfg_in: MutableMapping[str, Union[str, list]]) -> Iterator[Dict[str, str]]:
        """
        Generate dicts with values of one str from dict having values of lists of str
        :param cfg_in: dict with plural named keys having list values and may be singular named keys to fall back to
        :return: dict with singular named keys and str values
        """
        if '|' in str(cfg_in['db_path']):
            cfg_in['db_paths'] = [Path(p) for p in str(cfg_in['db_path']).split('|')]

        cfg_in_lists = {}
        for k, v in names_many_to_one.items():
            if k in cfg_in:
                cfg_in_lists[v] = cfg_in[k] if (not isinstance(cfg_in[k], str)
                                                and isinstance(cfg_in[k], Iterable)) else [cfg_in[k]]
        # dict of lists
        for v in zip(*cfg_in_lists.values()):
            yield dict(zip(cfg_in_lists, v))

    cfg_in_copy = cfg_in.copy()
    n = 1
    for d_source in gen_sources_dict(cfg_in_copy):
        cfg_in.update(d_source)
        cfg_in['tables'] = [cfg_in['table']]                 # for h5_dispenser_and_names_gen()
        for itbl, (tbl, coefs) in h5_dispenser_and_names_gen(cfg_in, cfg_out, fun_gen=fun_gen, b_close_at_end=False):
            l.info('%s. %s: ', n, tbl)  # itbl
            cfg_in['table'] = tbl                            # for gen_data_on_intervals()
            for d, i_burst in gen_data_on_intervals(cfg_in, cfg_in.get('split_period')):
                assert i_burst == 0                             # this is not a cycle
                cfg_in['tables'] = cfg_in_copy['tables']     # recover (seems not need now but may be for future use)
                coefs_copy = coefs.copy() if coefs else None # copy to not cumulate corrections in several cycles of generator consumer for same probe coefs
                yield n, tbl, coefs_copy, d
                n += 1


def dd_to_csv(d: dd.DataFrame, cfg_out: Mapping[str, Any], suffix='', b_single_file=True):
    """
    Save to ascii if cfg_out['text_path'] is not None
    :param d:
    :param cfg_out: dict with fields:
      - text_path: None or directory path
      - text_date_format: If callable then create "Date" column by calling it (dd.index), retain index only if "Time" in 'text_columns'. If string use it as format for index (Time) column
      - text_columns': optional
      - aggregate_period: str or with class with repr() that gives good file name part - to add suffix to files
    :param suffix:
    :param b_single_file: save all to one file or each partition individually
    :return:
    """
    if cfg_out['text_path'] is None:
        return


    tab = '\t'
    sep = tab
    ext = '.tsv' if sep==tab else '.csv'
    l.info('Saving *%s: %s', ext, '1 file' if b_single_file else f'{d.npartitions} files')
    try:
        dir_create_if_need(cfg_out['text_path'])
        def combpath(dir_or_prefix, s):
            return str(dir_or_prefix / s)
    except:
        l.exception('Dir not created!')
        def combpath(dir_or_prefix, s):
            return f'{dir_or_prefix}{s}'

    def name_that_replaces_asterisk(i_partition):
        return f'{d.divisions[i_partition]:%y%m%d_%H%M}'
        # too long variant: '{:%y%m%d_%H%M}-{:%H%M}'.format(*d.partitions[i_partition].index.compute()[[0,-1]])

    filename = combpath(
        cfg_out['text_path'],
        f"{name_that_replaces_asterisk(0) if b_single_file else '*'}{{}}_{suffix.replace('incl', 'i')}{ext}".format(
        f"bin{cfg_out['aggregate_period'].lower()}" if cfg_out['aggregate_period'] else '',  # lower seconds: S -> s
        ))

    with ProgressBar():
        d_out = d.round({'Vdir': 4, 'inclination': 4, 'Pressure': 3})
        # if not cfg_out.get('b_all_to_one_col'):
        #     d_out.rename(columns=map_to_suffixed(d.columns, suffix))
        if callable(cfg_out['text_date_format']):
            arg_out = {'index': 'Time' in cfg_out.get('text_columns', []),
                       'columns': cfg_out.get('text_columns', d_out.columns.insert(0, 'Date'))
                       }
            d_out['Date'] = d_out.map_partitions(lambda df: cfg_out['text_date_format'](df.index))
        else:
            arg_out = {'date_format': cfg_out['text_date_format'],
                       'columns': cfg_out.get('text_columns') or None  # write all columns if list is empty
                       }

        if progress is not None:
            progress(d_out)
        d_out.to_csv(filename=filename,
                     single_file=b_single_file,
                     name_function=None if b_single_file else name_that_replaces_asterisk,  # 'epoch' not works
                     float_format='%.5g',
                     sep=sep,
                     encoding="ascii",
                     #compression='zip',
                     **arg_out
                     )


# ---------------------------------------------------------------------------------------------------------------------
def main(new_arg=None, **kwargs):
    global l
    # input:
    cfg = cfg_from_args(my_argparser(), new_arg, **kwargs)
    if not cfg or not cfg['program']['return']:
        print('Can not initialise')
        return cfg
    elif cfg['program']['return'] == '<cfg_from_args>':  # to help testing
        return cfg
    l = init_logging(logging, None, cfg['program']['log'], cfg['program']['verbose'])
    l.info('Started %s(aggregete_period=%s)', this_prog_basename(__file__), cfg['out']['aggregate_period'] or 'None')
    # l = logging.getLogger(prog)
    try:
        pass
        #cfg['in'] = init_file_names(cfg['in'], cfg['program']['b_interact'], path_field='db_path')
    except Ex_nothing_done as e:
        print(e.message)
        return ()

    cfg['in']['i_range'] = [0, -1]  # USEi
    cfg['in']['burst_min'] = np.inf  # inf to not use bursts, None to autofind
    cfg['out']['chunksize'] = cfg['in']['chunksize']

    # If 'split_period' not set use custom splitting to be in memory limits
    if cfg['out']['aggregate_period']:
        aggregate_period_timedelta = pd_period_to_timedelta(cfg['out']['aggregate_period'])
        # Restricting number of counts=100000 in dask partition to not memory overflow
        split_for_memory = cfg['out']['split_period'] or 100000 * aggregate_period_timedelta
    else:
        aggregate_period_timedelta = None
        # Restricting number of counts in dask partition by time period
        split_for_memory = cfg['out']['split_period'] or pd.Timedelta(1, 'D')
    # Also is possible to set cfg['in']['split_period'] to cycle in parts but not need because dask takes control of parts if cfg_out['split_period'] is set

    h5init(cfg['in'], cfg['out'])
    cfg_out_table = cfg['out']['table']  # need? save beacause will need to change for h5_append()
    cols_out_allow = ['Vn', 'Ve', 'Pressure', 'Temp']  # ubsent cols will be ignored
    # cfg_out['data_columns'] = []  # can not index hdf5 complex column (see pandas to_hdf "data_columns" argument)
    # if len(cfg['in']['tables']) == 1 and '*' in cfg['in']['tables'][0]:  # pattern specified
    set_field_if_no(cfg['out'], 'not_joined_h5_path', not aggregate_period_timedelta)

    def map_to_suffixed(names, tbl, probe_number):
        suffix = f'{tbl[0]}{probe_number:02}'
        return {col: f'{col}_{suffix}' for col in names}

    for lim in ['min', 'max']:
        if 'M' in cfg['filter'][lim]:
            for ch in ('x', 'y', 'z'):
                set_field_if_no(cfg['filter'][lim], f'M{ch}', cfg['filter'][lim]['M'])

    cfg['filter']['sleep_s'] = 0.5  # to run slower, helps for system memory management?

    log = {}
    dfs_all_list = []
    dfs_all_log = []
    dfs_all: Optional[pd.DataFrame] = None
    cfg['out']['tables_have_wrote'] = []

    # for itbl, (tbl, coefs) in h5_dispenser_and_names_gen(cfg['in'], cfg['out'], fun_gen=h5_names_gen):
    #     l.info('{}. {}: '.format(itbl, tbl))
    #     cfg['in']['table'] = tbl  # to get data by gen_intervals()
    #     for d, i_burst in (gen_data_on_intervals if False else gen_data_on_intervals_from_many_sources)(cfg):
    #         assert i_burst == 0  # this is not a cycle
    for itbl, tbl, coefs, d in gen_variables(cfg['in'], cfg['out'], fun_gen=h5_names_gen):
        d = filter_local(d, cfg['filter'])  # d[['Mx','My','Mz']] = d[['Mx','My','Mz']].mask(lambda x: x>=4096)
        probe_number = int(re.findall('\d+', tbl)[0])

        if not cfg['in']['db_path'].stem.endswith('proc_noAvg'):
            # Zeroing
            if cfg['in']['timerange_zeroing']:
                # individual or same interval for each table:
                timerange_zeroing = cfg['in']['timerange_zeroing'].get(probe_number) if isinstance(
                    cfg['in']['timerange_zeroing'], Mapping) else cfg['in']['timerange_zeroing']

                if timerange_zeroing:
                    d_zeroing = d.loc[slice(*pd.to_datetime(timerange_zeroing, utc=True)), ('Ax', 'Ay', 'Az')]
                    l.info('Zeroing data: average %d points in interval %s - %s', len(d_zeroing),
                           d_zeroing.divisions[0], d_zeroing.divisions[-1])
                    mean_countsG0 = np.atleast_2d(d_zeroing.mean().values.compute()).T
                    coefs['Ag'], coefs['Ah'] = coef_zeroing(mean_countsG0, coefs['Ag'], coefs['Cg'], coefs['Ah'])
            # Azimuth correction
            if cfg['in']['azimuth_add']:
                # individual or the same correction for each table:
                coefs['azimuth_shift_deg'] += (cfg['in']['azimuth_add'].get(probe_number, 0) if isinstance(
                        cfg['in']['azimuth_add'], Mapping) else cfg['in']['azimuth_add'])

        if aggregate_period_timedelta:
            if not cfg['in']['db_path'].stem.endswith('proc_noAvg'):
                l.warning('Not standard db name! Ensure that you do not average already avereged data')
            d = d.resample(aggregate_period_timedelta,
                           closed='right' if 'Pres' in cfg['in']['db_path'].stem else 'left'
                           # 'right' for burst mode because the last value of interval used in wavegauges is round
                           ).mean()
            try:  # persist speedups calc_velocity greatly but may require too many memory
                l.info('Persisting data aggregated by %s', cfg['out']['aggregate_period'])
                d.persist()  # excludes missed values?
            except MemoryError:
                l.debug('- Failed (not enough memory for persisting). Continue...')

        if cfg['in']['db_path'].stem.endswith('proc_noAvg'):
            # recalculate aggregated values of polar coordinates and angles that is invalid after aggregated directly
            d = dekart2polar_df_v_en(d)
            if cfg['out']['split_period']:  # for csv splitting only
                d = d.repartition(freq=cfg['out']['split_period'])
        else:
            # Velocity calculation

            # with repartition for split ascii and/or remove MemoryError
            d = incl_calc_velocity(d.repartition(freq=split_for_memory), **coefs,
                                   cfg_filter=cfg['filter'],
                                   cfg_proc=cfg['proc'])

            # Separated (not joined) probes velocity to h5
            if cfg['out']['not_joined_h5_path']:
                log['Date0'], log['DateEnd'] = d.divisions[:-1], d.divisions[1:]
                tables_wrote_now = h5_append_to(d, tbl, cfg['out'], log, msg=f'saving {tbl} (separately)',
                                                print_ok=None)
                if tables_wrote_now:
                    cfg['out']['tables_have_wrote'].append(tables_wrote_now)

        dd_to_csv(d, cfg['out'], tbl, b_single_file=not cfg['out']['split_period'])

        if aggregate_period_timedelta:  # data have index of same period
            # Combine data
            try:
                cols_save = [c for c in cols_out_allow if c in d.columns]
                sleep(cfg['filter']['sleep_s'])
                Vne = d[cols_save].compute()  # MemoryError((1, 12400642), dtype('float64'))
                if not cfg['out']['b_all_to_one_col']:
                    Vne.rename(columns=map_to_suffixed(cols_save, tbl, probe_number), inplace=True)
                dfs_all_list.append(Vne)
                dfs_all_log.append(tbl)
            except Exception as e:
                l.exception('Can not cumulate result! ')
                raise
                # todo: if low memory do it separately loading from temporary tables in chanks

        gc.collect()  # frees many memory. Helps to not crash


    # Combined data to hdf5
    if aggregate_period_timedelta:
        dfs_all = pd.concat(dfs_all_list, sort=True, axis=(0 if cfg['out']['b_all_to_one_col'] else 1))
        dfs_all_log = pd.DataFrame([df.index[[0,-1]].to_list() for df in dfs_all_list],
                                   columns=['Date0', 'DateEnd'])\
            .set_index('Date0')\
            .assign(table_name=dfs_all_log)\
            .sort_index()
        tables_wrote_now = h5_append_to(dfs_all, cfg_out_table, cfg['out'], log=dfs_all_log, msg='Saving accumulated data',
                                        print_ok='.')
        if tables_wrote_now:
            cfg['out']['tables_have_wrote'].append(tables_wrote_now)


    # close temporary output store
    h5_close(cfg['out'])

    try:
        failed_storages = h5move_tables(cfg['out'], cfg['out']['tables_have_wrote'])
    except Ex_nothing_done as e:
        l.warning('Tables not moved')


    if dfs_all is not None and cfg['out']['b_all_to_one_col']:
        # Save to 1 combined ascii with regular time interval filling absent values with 0
        dd_to_csv(
            dd.from_pandas(dfs_all, chunksize=500000)
                .resample(rule=aggregate_period_timedelta)
                .first()
                .fillna(0),  # inserts zeros in holes
            cfg['out'], suffix=','.join(cfg['in']['tables'])
            )

    print('Ok.', end=' ')
    # h5index_sort(cfg['out'], out_storage_name= cfg['out']['db_base']+'-resorted.h5', in_storages= failed_storages)
    # dd_out = dd.multi.concat(dfs_list, axis=1)


if __name__ == '__main__':
    main()

r"""

    dfs_all = pd.merge_asof(dfs_all, Vne, left_index=True, right_index=True,
                  tolerance=pd.Timedelta(cfg['out']['aggregate_period'] or '1ms'),
                            suffixes=('', ''), direction='nearest')
    dfs_all = pd.concat((Vne, how='outer')  #, rsuffix=tbl[-2:] join not works on dask
    V = df['V'].to_frame(name='V' + tbl[-2:]).compute()
if dfs_all is computed it is in memory:
mem = dfs_all.memory_usage().sum() / (1024 ** 2)
if mem > 50:
    l.debug('{:.1g}Mb of data accumulated in memory '.format(mem))

export_df_to_csv(df, cfg_out, add_subdir='V,P_txt')
? h5_append_cols()
df_all = dd.merge(indiv, cm.reset_index(), on='cmte_id')


old cfg

    cfg = {  # how to load:
        'in': {
            'db_path': '/mnt/D/workData/BalticSea/181116inclinometer_Schuka/181116incl.h5', #r'd:\WorkData\BalticSea\181116inclinometer_Schuka\181116incl.h5',
            'tables': ['incl.*'],
            'chunksize': 1000000, # 'chunksize_percent': 10,  # we'll repace this with burst size if it suit
            'date_min': datetime.strptime('2018-11-16T15:19:00', '%Y-%m-%dT%H:%M:%S'),
            'date_max': datetime.strptime('2018-12-14T14:35:00', '%Y-%m-%dT%H:%M:%S')
            'split_period': '999D',  # pandas offset string (999D, H, ...) ['D' ]
            'aggregate_period': '2H',  # pandas offset string (D, 5D, H, ...)
            #'max_g_minus_1' used only to replace bad with NaN
        },
        'out': {
            'db_path': '181116incl_proc.h5',
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
