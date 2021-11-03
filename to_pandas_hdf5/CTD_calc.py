# !/usr/bin/env python
# coding:utf-8

# from to_pandas_hdf5.csv_specific_proc import deg_min_float_as_text2deg

"""
  Author:  Andrey Korzh <korzh@n_extmail.ru>
  Purpose: load CTD data from hdf5 PyTables store, calc Sal / find runs
  Created: 18.10.2016
  Updated: 15:07.2019
"""

import logging
from collections import OrderedDict
from datetime import datetime, timedelta
from os import path as os_path
from sys import stdout as sys_stdout
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union
from pathlib import PurePath, Path
import numpy as np
import pandas as pd

if __debug__:
    from matplotlib import pyplot as plt
import gsw

# my
from utils2init import my_argparser_common_part, cfg_from_args, this_prog_basename, init_file_names, init_logging, \
    Ex_nothing_done, set_field_if_no, dir_create_if_need, FakeContextIfOpen
from utils_time import timzone_view
from other_filters import rep2mean, inearestsorted
from to_pandas_hdf5.csv2h5 import set_filterGlobal_minmax
from to_pandas_hdf5.h5toh5 import h5temp_open, h5move_tables, h5init, h5del_obsolete, h5index_sort, query_time_range, \
    h5remove_duplicates, h5select
from to_pandas_hdf5.h5_dask_pandas import h5_append

date_format_ISO9115 = '%Y-%m-%dT%H:%M:%S'  # for Obninsk

if __name__ == '__main__':
    l = None  # see main(): l = init_logging(logging, None, cfg['program']['log'], cfg['program']['verbose'])
else:
    l = logging.getLogger(__name__)
version = '0.0.1'


# noinspection PyUnresolvedReferences
def my_argparser():
    """
    Configuration parser
    - add here common options for different inputs
    - add help strings for them
    :return p: configargparse object of parameters
    All p argumets are of type str (default for add_argument...), because of
    custom postprocessing based of args names in ini2dict
    """

    p = my_argparser_common_part({'description': 'csv2h5 version {}'.format(version) + """
----------------------------
Extract data from Pandas HDF5 store*.h5 files, 
process it and save HDF5/CSV
----------------------------"""}, version)

    # , 'default_config_files': [os_path.join(os_path.dirname(__file__), name) for name in
    #                            ('CTD_calc.ini', 'CTD_calc.json')]

    # Configuration sections
    s = p.add_argument_group('in',
                             'all about input files')
    s.add('--db_path', default='.',  # nargs=?,
             help='path to pytables hdf5 store to load data. May use patterns in Unix shell style')
    s.add('--tables_list',
             help='table name in hdf5 store to read data. If not specified then will be generated on base of path of input files')
    s.add('--tables_log',
             help='table name in hdf5 store to read data intervals. If not specified then will be "{}/logFiles" where {} will be replaced by current data table name')
    s.add('--table_nav', default='navigation',
             help='table name in hdf5 store to add data from it to log table when in "find runs" mode. Use empty strng to not add')
    s.add('--dt_from_utc_hours', default='0',
             help='add this correction to loading datetime data. Can use other suffixes instead of "hours"')
    s.add('--b_skip_if_up_to_date', default='True',
             help='exclude processing of files with same name and which time change is not bigger than recorded in database (only prints ">" if detected). If finds updated version of same file then deletes all data which corresponds old file and after it brfore procesing of next files')
    s.add('--b_temp_on_its90', default='True',
             help='When calc CTD parameters treat Temp have red on ITS-90 scale. (i.e. same as "temp90")')
    s.add('--path_coef',
             help='path to file with coefficients. Used for processing of Neil Brown CTD data')
    s.add('--lat_float', help='Latitude used to calc SA if no such data column')
    s.add('--lon_float', help='Longitude used to calc SA if no such data column')

    s = p.add_argument_group('out',
                             'all about output files')
    info_default_path = '[in] path from *.ini'
    s.add('--out.db_path', help='hdf5 store file path')
    s.add('--out.tables_list',
              help='table name in hdf5 store to write data. If not specified then it is same as input tables_list (only new subtable will created here), else it will be generated on base of path of input files')
    s.add('--out.tables_log_list',
             help='table name in hdf5 store to save intervals. If contains "logRuns" then runs will be found first')
    s.add('--path_csv',
              help='path to output directory of csv file(s)')
    s.add('--data_columns_list',
              help='list of columns names used in output csv file(s)')
    s.add('--b_insert_separator', default='True',
              help='insert NaNs row in table after each file data end')
    s.add('--b_remove_duplicates', default='False', help='Set True if you see warnings about')
    s.add('--text_date_format', default='%Y-%m-%d %H:%M:%S.%f',
              help='Format of date column in csv files. Can use float or string representations')

    s = p.add_argument_group('extract_runs',
                             'program behaviour')
    s.add('--cols_list', default='Pres',
              help='column for extract_runs (other common variant besides default is "Depth")')
    s.add('--dt_between_min_minutes', default='1', help='')
    s.add('--min_dp', default='20', help='')
    s.add('--min_samples', default='200', help='100 use small value (10) for binned (averaged) samples')
    s.add('--b_keep_minmax_of_bad_files', default='False',
              help='keep 1 min before max and max of separated parts of data where movements insufficient to be runs')
    s.add('--b_save_images', default='True', help='to review split result')

    s = p.add_argument_group('filter',
                             'filter all data based on min/max of parameters')
    s.add('--min_dict',
              help='List with items in  "key:value" format. Sets to NaN data of ``key`` columns if it is below ``value``')
    s.add('--max_dict',
              help='List with items in  "key:value" format. Sets to NaN data of ``key`` columns if it is above ``value``')

    s = p.add_argument_group('program',
                             'program behaviour')
    s.add('--return', default='<end>',  # nargs=1,
               choices=['<cfg_from_args>', '<gen_names_and_log>', '<end>'],
               help='<cfg_from_args>: returns cfg based on input args only and exit, <gen_names_and_log>: execute init_input_cols() and also returns fun_proc_loaded function... - see main()')
    return (p)


def extractRuns(P: Sequence, cfg_extract_runs: Mapping[str, Any]) -> Tuple[List[int], List[int]]:
    '''
    Extract runs based on length and amplitude of intervals with sign of gradient(P)
    :param P: z coordinate - negative is below surface
    :param cfg_extract_runs: dict with fields:
        min_samples
        min_dp
    :return: [imin,imax]
    todo: replace with one based on famous line simplification algorithm
#                          min_samples
#surface -----------------|-|----+---------------
#          x     x         x     |                
#           x   x x       x x    |min_dp
#            x x   x     x   x   |
#             x     x    x    x  |
#                    x   x     x-+
#                     x  x      x x        
#                      xx        x          
#result separation:
#          0>>>>>>>>>>>><<<>>>>>0%%%%%%%%%%%%%%%%+
#positions: |imin(1)  |imax(1)
#                          |imin(2)
#                               |imax(2)
    '''
    min_samples = np.array(cfg_extract_runs['min_samples']) if (
            'min_samples' in cfg_extract_runs and cfg_extract_runs['min_samples']) else 0
    min_dp = np.array(cfg_extract_runs['min_dp']) if (
            'min_dp' in cfg_extract_runs and cfg_extract_runs['min_dp']) else 0

    dP = np.ediff1d(P, to_end=-np.diff(P[-2:]))  # add opposite extremum to end
    bok = dP < 0  # direction is "down" (if negative pressure)

    # Extremums + edges
    bex = np.ediff1d(bok.view(np.int8), to_begin=True).view(np.bool8)
    pex = P[bex]
    iex = np.flatnonzero(bex)
    # n_ex= len(iex)
    bt = bok[bex]  # True if it is a top extremum
    bl = ~bt  # bot (low) extremums

    # if __debug__ and cfg_extract_runs['path_image'].endswith('155551.png'):
    #     print('stop')

    # Removing bad extremums
    n_ok = len(iex)
    if __debug__:
        fig = None
        b_plot_started = False
    while True:
        # bbad - mask of candidates to remove from extremums
        if min_samples:
            # length samples of each up/down interval:
            s = np.ediff1d(iex, to_end=0)
            # find intervals with insufficient number of samples:
            if np.size(min_samples) > 1:
                bok = s > min_samples[np.int64(bl)]
            else:
                bok = s > min_samples
        else:
            bok = np.zeros_like(bt)
        if min_dp:
            # height of each up/down interval:
            s = np.abs(np.ediff1d(pex, to_end=0))
            # find intervals with insufficient height:
            if np.size(min_dp) > 1:
                bok |= (s > min_dp[np.int64(bl)])  # down
            else:
                bok |= (s > min_dp)

        bok2 = np.zeros_like(bt)

        # Terminal extremums:
        bok2[bt] = np.ediff1d(pex[bt],
                              to_end=1) < 0  # next highland lower, last is not terminal (not save low highland near end)
        bok2[bl] = np.ediff1d(pex[bl], to_end=1) > 0  # next lowland higher, last is terminal

        # Deleting extremums only if not terminal:
        bok |= bok2  # print(repr(np.vstack((bt, bok, bok2, iex, np.int64(pex))).T))
        bbad = ~bok
        # Deleting not terminal opposite extremums near deleted extremums:
        # near:
        b_near = np.logical_or(np.hstack((False, bbad[:-1])), np.hstack((bbad[1:], False)))
        bbad |= np.logical_and(b_near, ~bok)

        bok = ~bbad
        bt = bt[bok]
        iex = iex[bok]
        if bt.size <= 1:
            print('no good extremums in run' + (
                    ': ' + cfg_extract_runs['path_image']) if 'path_image' in cfg_extract_runs else '')
            bl = ~bt
        else:
            pex = pex[bok]

            # Deleting smaller adjasent extremums of one type
            bbad = np.ediff1d(bt.view(np.int8), to_begin=-1, to_end=-1) == 0  # False ... False
            isten = np.flatnonzero(np.ediff1d(bbad.view(np.int8))).reshape((-1, 2))
            isten[:, 1] += 1
            ist = isten[:, 0]
            maxmin = lambda a, bmax: np.argmax(a) if bmax else np.argmin(a)
            iok = [st + maxmin(pex[slice(*sten)], b) for st, sten, b in zip(ist, isten, bt[ist])]
            bbad = bbad[:-1] | bbad[1:]
            bbad[iok] = False  # print(repr(np.vstack((bt, bbad, pex)).T))

            assert np.all(~bl[bok][~bbad] == bt[~bbad])
            bok = ~bbad
            bt = bt[bok]
            bl = ~bt
            pex = pex[bok]
            iex = iex[bok]

        n_ok_cur = bok.size
        if n_ok_cur == n_ok:  # must decrease to continue
            if __debug__:  # result should be logged:
                if fig == None:
                    fig = plt.figure(111)
                    b_plot_started = True
                plt.cla()  # plt.hold(False) get AttributeError on TkAgg backend
                plt.plot(P, color='c', alpha=0.5)  # '.',
                # plt.hold(True)
                # plt.plot(iex[bt&~bbad], P[iex[bt&~bbad]], 'og', alpha=0.5)
                # plt.plot(iex[bl&~bbad], P[iex[bl&~bbad]], '+r', alpha=0.5)
                plt.plot(iex[bt], P[iex[bt]], 'og', alpha=0.5)
                plt.plot(iex[bl], P[iex[bl]], '+r', alpha=0.5)
                if 'path_image' in cfg_extract_runs:
                    plt.savefig(cfg_extract_runs['path_image'])
                else:
                    plt.show()

            break
        else:
            n_ok = n_ok_cur  # save

    return (iex[bt].tolist(), iex[bl].tolist())  # imin, imax


def CTDrunsExtract(P: np.ndarray,
                   dnT: np.ndarray,
                   cfg_extract_runs: Dict[str, Any]) -> np.ndarray:
    '''
    find profiles ("Mainas"). Uses extractRuns()
    :param P: Pressure/Depth
    :param dnT: Time
    :param cfg_extract_runs: settings dict with fields:
      - dt_between_min
      - min_dp
      - min_samples
      - dt_hole_max - split runs where dt between adjasent samples bigger. If not
      specified it is set equal to 'dt_between_min' automatically
      - b_do - if it is set to False intepret all data as one run
      - b_keep_minmax_of_bad_files, optional - keep 1 min before max and max of separated parts of data where movements insufficient to be runs
    :return: iminmax: 2D numpy array np.int64([[minimums],[maximums]])
    '''

    if ('do' not in cfg_extract_runs) or cfg_extract_runs['b_do']:  # not do only if b_do is set to False
        P = np.abs(rep2mean(P))
        if not 'dt_hole_max' in cfg_extract_runs:
            cfg_extract_runs['dt_hole_max'] = cfg_extract_runs['dt_between_min']
        dt64_hole_max = np.timedelta64(cfg_extract_runs['dt_hole_max'], 'ns')
        # time_holes= np.flatnonzero(np.ediff1d(dnT, dt64_hole_max, dt64_hole_max) >= dt64_hole_max) #bug in numpy
        time_holes = np.hstack((0, np.flatnonzero(np.diff(dnT) >= dt64_hole_max), len(dnT)))
        imin = []
        imax = []
        i_keep_bad_runs = []  #
        for ist, ien in zip(time_holes[:-1], time_holes[1:]):
            islice = slice(ist, ien)
            if (ien - ist) < cfg_extract_runs['min_samples']:
                continue

            if (P[islice].max() - P[islice].min()) < cfg_extract_runs['min_dp']:
                if cfg_extract_runs.get('b_keep_minmax_of_bad_files'):
                    i_keep_bad_runs.append(len(imax))
                    imax.append(P[islice].argmax())
                    imin.append(P[ist:imax[-1]].argmin())
            else:
                if 'path_images' in cfg_extract_runs:
                    cfg_extract_runs['path_image'] = os_path.join(cfg_extract_runs['path_images'],
                                                                  'extract_runs{:%y%m%d_%H%M%S}'.format(
                                                                      np.datetime64(dnT[ist], 's').astype(
                                                                          datetime))) + '.png'
                [it, il] = extractRuns(-P[islice], cfg_extract_runs)
                # Correct extractRuns func (mins and maxs must alternates):
                # make 1st min be less than 1st max
                if it and il:
                    if il[0] < it[0]:
                        del il[0]
                # make length of min and max be equal
                if len(it) > len(il):
                    del it[-1]
                    il.append(ien - ist - 1)
                elif len(it) < len(il):
                    if it and it[0] > il[0]:
                        del il[0]
                    else:
                        it.append(ien - ist - 1)
                imin.extend([i + ist for i in it])
                imax.extend([i + ist for i in il])
        # Filter run down intervals:
        if len(imin):
            iminmax = np.vstack((imin, imax))
            bok = np.logical_and(np.diff(iminmax, 1, 0) >= cfg_extract_runs['min_samples'],
                                 np.diff(P[iminmax], 1, 0) >= cfg_extract_runs['min_dp']).flatten()
            bok[i_keep_bad_runs] = True
            if ~np.all(bok):
                iminmax = iminmax[:, bok]
        else:
            l.warning('no runs!')
            return np.int64([[],[]])
        # N= min(len(imax), len(imin))
        # iminMax = [imin, imax]
    else:
        # N= 0
        iminmax = np.int64([[0], [len(P)]])

    # # make mask with ends set to -1
    # b_maina = np.zeros(len(P), 'int8')
    # for k in range(N):
    #     b_maina[imin[k]:imax[k]] = 1
    # b_maina[imax] = -1

    # Runs.PMax= P(imax)
    return iminmax  # , b_maina


## Functions for prepare sycle ###################################################
# - can assign data to cfg['for']
def load_coef(cfg):
    set_field_if_no(cfg, 'for', {})
    cfg['for']['k_names'], cfg['for']['kk'] = \
        np.loadtxt(cfg['in']['path_coef'],
                   dtype=[('name', 'S10'), ('k', '4<f4')],
                   skiprows=1, unpack=True)  # k0	k1*x      	k2*x^2     	k3*x^3	CKO
    cfg['for']['kk'] = np.fliplr(cfg['for']['kk'])


## Functions for execute in sycle ################################################
# - output will be saved
def process_brown(df_raw, cfg: Mapping[str, Any]):
    '''
    Calc physical values from codes
    :param df_raw:
    :param cfg:
    :return: pandas dataframe
    # todo: use signs. For now our data haven't negative values and it is noise if signs!=0. Check: df_raw.signs[df_raw.signs!=0]
    '''
    Val = {}
    if b'Pres' in cfg['for']['k_names'] and not 'Pres' in df_raw.columns:
        df_raw = df_raw.rename(columns={'P': 'Pres'})

    for nameb in np.intersect1d(np.array(df_raw.columns, 'S10'), cfg['for']['k_names']):
        name = nameb.decode('ascii')
        Val[name] = np.polyval(cfg['for']['kk'][nameb == cfg['for']['k_names']].flat, df_raw[name])

    # Practical Salinity PSS-78
    Val['Sal'] = gsw.SP_from_C(Val['Cond'], Val['Temp'], Val['Pres'])
    df = pd.DataFrame(Val, columns=cfg['out']['data_columns'], index=df_raw.index)

    # import seawater as sw
    # T90conv = lambda t68: t68/1.00024
    # Val['sigma0'] sw.pden(s, T90conv(t), p=0) - 1000
    return df


def log_runs(df_raw: pd.DataFrame,
             cfg: Mapping[str, Any],
             log: Optional[MutableMapping[str, Any]] = None) -> pd.DataFrame:
    """
    Changes log
    :param df_raw: DataFrame of parameters (data)
    :param cfg: dict with fields:
      - extract_runs:
      - in
      -
    :param log: here result is saved in fields:
     'Date0', 'DateEnd', 'rows', 'rows_filtered', 'fileName', 'fileChangeTime', and df_raw column names
    :return: empty DataFrame (for compatibility)
    """

    if log is None:
        print('not updating log')
        log = {'fileName': None,
               'fileChangeTime': None}
    else:
        log.clear()
    imin, imax = CTDrunsExtract(
        df_raw[cfg['extract_runs']['cols'][0]].values,
        df_raw.index.values,
        cfg['extract_runs'])
    if not len(imin):
        return pd.DataFrame(data=None, columns=df_raw.columns, index=df_raw.index[[]])  # empty dataframe

    log.update(  # pd.DataFrame(, index=log_update['_st'].index).rename_axis('Date0')
        {'rows': imax - imin,
        'rows_filtered': np.append(imin[1:], len(df_raw)) - imax,  # rows between runs down. old: imin - np.append(0, imax[:-1])
        'fileName': [os_path.basename(cfg['in']['file_stem'])] * len(imin),
        'fileChangeTime': [cfg['in']['fileChangeTime']] * len(imin),
        })

    log.update(
        get_runs_parameters(df_raw, df_raw.index[imin], df_raw.index[imax],
                            cols_good_data=cfg['extract_runs']['cols'],
                            dt_search_nav_tolerance=cfg['out'].get('dt_search_nav_tolerance', timedelta(minutes=2)),
                            **{k: cfg['in'].get(k) for k in ['dt_from_utc', 'db', 'db_path', 'table_nav']}
                ))
    l.info('updating log with %d row%s...', imin.size, 's' if imin.size > 1 else '')

    # list means save only log but not data
    print('runs lengths, initial filtered counts: {rows}, {rows_filtered}'.format_map(log))

    # isinstance(cfg_out['log']['rows'], int) and
    # log.update(**dict([(i[0], i[1].values) for st_en in zip(
    #     *(dfNpoints.iloc[sl].add_suffix(sfx).items() for sfx, sl in (
    #         ('_st', slice(0, len(log_update['_st']))),
    #         ('_en', slice(len(log_update['_st']), len(dfNpoints)))
    #     )
    # )) for i in st_en]))

    # log.update(
    #     Date0=timzone_view(df_raw.index[imin], cfg['in']['dt_from_utc']),
    #     DateEnd=timzone_view(df_raw.index[imax], cfg['in']['dt_from_utc']),
    #     fileName=[os_path.basename(cfg['in']['file_stem'])]*len(imin),
    #     fileChangeTime=[cfg['in']['fileChangeTime']]*len(imin),
    #     rows=imax - imin,                            #rows down
    #     rows_filtered=imin - np.append(0, imax[:-1]) #rows up from previous down (or start)
    # )
    # adding nothing to main table:
    return pd.DataFrame(data=None, columns=df_raw.columns, index=df_raw.index[[]])    # empty dataframe
    # df_raw.ix[[]] gets random error - bug in pandas


def get_runs_parameters(df_raw, times_min, times_max, cols_good_data: Union[str, Sequence[str], None],
                        dt_from_utc: timedelta = timedelta(0), db=None, db_path=None,
                        table_nav=None,
                        table_nav_cols=('Lat', 'Lon', 'DepEcho', 'Speed', 'Course'),
                        dt_search_nav_tolerance=timedelta(minutes=2)):
    """

    :param df_raw:
    :param times_min:
    :param times_max:
    :param cols_good_data: cols of essential data that must be good (depth)
    :param dt_from_utc:
    :param db:
    :param db_path:
    :param table_nav: 'navigation' table to find data absent in df_raw. Note: tries to find only positive vals
    :param table_nav_cols:
    :param dt_search_nav_tolerance:
    :return:
    """

    log = {}
    log_update = {}  # {_st: DataFrame, _en: DataFrame} - dataframes of parameters for imin and imax
    for times_lim, suffix, log_time_col, i_search in ((times_min, '_st', 'Date0',    0),
                                                      (times_max, '_en', 'DateEnd', -1)
                                                      ):
        log_update[suffix] = df_raw.asof(times_lim, subset=cols_good_data)  # rows of last good data
        log[log_time_col] = timzone_view(log_update[suffix].index, dt_from_utc)

        # Search for nearest good values if have bad parameter p
        for (p, *isnan) in log_update[suffix].isna().T.itertuples(name=None):
            if i_search == -1:
                log_update[suffix].loc[isnan, p] = df_raw[p].asof(times_max[isnan])
            else:
                # "asof()"-alternative for 1st notna: take 1st good element in each interval
                for time_nan, time_min, time_max in zip(times_lim[isnan], times_min[isnan], times_max[isnan]):
                    s_search = df_raw.loc[time_min:time_max, p]

                    try:
                        log_update[suffix].at[time_nan, p] = s_search[s_search.notna()].iat[0]  # same as .at[s_search.first_valid_index()]
                    except IndexError:
                        l.warning('no good values for parameter "%s" in run started %s', p, time_nan)
                        continue
        log_update[suffix] = log_update[suffix].add_suffix(suffix)
    log.update(  # pd.DataFrame(, index=log_update['_st'].index).rename_axis('Date0')
        {**dict(
            [(k, v.values) for st_en in zip(log_update['_st'].items(), log_update['_en'].items()) for k, v in st_en]),
         # flatten pares
         })

    if table_nav:
        time_points = log_update['_st'].index.append(log_update['_en'].index)
        with FakeContextIfOpen(lambda f: pd.HDFStore(f, mode='r'), db_path, db) as store:
            df_nav, dt = h5select(  # all starts then all ends in row
                store, table_nav,
                columns=table_nav_cols,
                time_points=time_points,
                dt_check_tolerance=dt_search_nav_tolerance
                )

        # {:0.0f}s'.format(cfg['out']['dt_search_nav_tolerance'].total_seconds())
        # todo: allow filter for individual columns. solution: use multiple calls for columns that need filtering with appropriate query_range_pattern argument of h5select()
        isnan = df_nav.isna()
        for col in df_nav.columns[isnan.any(axis=0)]:

            # not works:
            # df_nav_col, dt_col = h5select(  # for current parameter's name
            #         cfg['in']['db'], cfg['in']['table_nav'],
            #         columns=[col],
            #         query_range_lims=time_points[[0,-1]],
            #         time_points=time_points[isnan[col]],
            #         query_range_pattern = f"index>=Timestamp('{{}}') & index<=Timestamp('{{}}') & {col} > 0 ",
            #         dt_check_tolerance=cfg['out']['dt_search_nav_tolerance']
            #         )

            # Note: tries to find only positive vals:
            df_nav_col = store.select(
                table_nav,
                where="index>=Timestamp('{}') & index<=Timestamp('{}') & {} > 0".format(
                    *(time_points[[0, -1]] + np.array(dt_search_nav_tolerance, 'm8[s]') * [-1, 1]), col),
                columns=[col])
            try:
                vals = df_nav_col[col].values
                vals = vals[inearestsorted(df_nav_col.index, time_points[isnan[col]])]
            except IndexError:
                continue  # not found
            if vals.any():
                df_nav.loc[isnan[col], col] = vals

        # df_nav['nearestNav'] = dt.astype('m8[s]').view(np.int64)
        df_edges_items_list = [df_edge.add_suffix(suffix).items() for suffix, df_edge in (
            ('_st', df_nav.iloc[:len(log_update['_st'])]),
            ('_en', df_nav.iloc[len(log_update['_st']):len(df_nav)]))]

        for st_en in zip(*df_edges_items_list):
            for name, series in st_en:
                # If have from data table already => update needed elements only
                if name in log:
                    b_need = np.isnan(log.get(name))
                    if b_need.any():
                        b_have = np.isfinite(series.values)
                        # from loaded nav in points
                        b_use = b_need & b_have
                        if b_use.any():
                            log[name][b_use] = series.values[b_use]
                        # # from all nav (not loaded)
                        # b_need &= ~b_have
                        #
                        # if b_need.any():
                        #     # load range to search nearest good val. for specified fields and tolerance
                        #     df = cfg['in']['db'].select(cfg['in']['table_nav'], where=query_range_pattern.format(st_en.index), columns=name)

                        # df_nav = h5select(  # for current parameter's name
                        #     cfg['in']['db'], cfg['in']['table_nav'],
                        #     columns=name,
                        #     query_range_lims=st_en
                        #     time_points=log_update['_st'].index.append(log_update['_en'].index),
                        #     dt_check_tolerance=cfg['out']['dt_search_nav_tolerance']
                        #     )
                    continue
                # else:
                #     b_need = np.isnan(series.values)
                #     for

                # Else update all elements at once
                log[name] = series.values
    return log


def coord_data_col_ensure(df: pd.DataFrame, log_row):
    for coord in ['Lat', 'Lon']:
        if coord in df.columns:
            ok = df[coord].notna()
            if any(ok):
                df[coord].interpolate(method='time', inplace=True)
                continue
        # if existed data in column not sufficient then copy from log_row's start coordinate
        df[coord] = getattr(log_row, f'{coord}_st')


def add_ctd_params(df_in: MutableMapping[str, Sequence], cfg: Mapping[str, Any], lon=16.7, lat=55.2):
    """
    Calculate all parameters from 'sigma0', 'depth', 'soundV', 'SA' that is specified in cfg['out']['data_columns']
    :param df_in: DataFrame with columns:
     'Pres', 'Temp90' or 'Temp', and may be others:
     'Lat', 'Lon': to use instead cfg['in']['lat'] and lat and -//- lon
    :param cfg: dict with fields:
        ['out']['data_columns'] - list of columns in output dataframe
        ['in'].['b_temp_on_its90'] - optional
    :param lon:  # 54.8707   # least priority values
    :param lon:  # 19.3212
    :return: DataFrame with only columns specified in cfg['out']['data_columns']
    """
    ctd = df_in
    params_to_calc = set(cfg['out']['data_columns']).difference(ctd.columns)
    params_coord_needed_for = params_to_calc.intersection(('depth', 'sigma0', 'SA', 'soundV'))  # need for all cols?
    if any(params_coord_needed_for):
        # todo: load from nav:
        if np.any(ctd.get('Lat')):
            lat = ctd['Lat']
            lon = ctd['Lon']
        else:
            if 'lat' in cfg['in']:
                lat = cfg['in']['lat']
                lon = cfg['in']['lon']
            else:
                print('Calc', '/'.join(params_coord_needed_for), f'using MANUAL INPUTTED coordinates: lat={lat}, lon={lon}')

    pd_chained_assignment, pd.options.mode.chained_assignment = pd.options.mode.chained_assignment, None
    if 'Temp90' not in ctd.columns:  # if isinstance(ctd, pd.DataFrame)
        if cfg['in'].get('b_temp_on_its90'):
            ctd['Temp90'] = ctd['Temp']
        else:
            ctd['Temp90'] = gsw.conversions.t90_from_t68(df_in['Temp'])

    ctd['SA'] = gsw.SA_from_SP(ctd['Sal'], ctd['Pres'], lat=lat, lon=lon)  # or Sstar_from_SP() for Baltic where SA=S*
    # Val['Sal'] = gsw.SP_from_C(Val['Cond'], Val['Temp'], Val['P'])
    if 'soundV' in params_to_calc:
        ctd['soundV'] = gsw.sound_speed_t_exact(ctd['SA'], ctd['Temp90'], ctd['Pres'])

    if 'depth' in params_to_calc:
        ctd['depth'] = np.abs(gsw.z_from_p(np.abs(ctd['Pres'].to_numpy()), lat))  # to_numpy() works against NotImplementedError: Cannot apply ufunc <ufunc 'z_from_p'> to mixed DataFrame and Series inputs.
    if 'sigma0' in params_to_calc:
        CT = gsw.CT_from_t(ctd['SA'], ctd['Temp90'], ctd['Pres'])
        ctd['sigma0'] = gsw.sigma0(ctd['SA'], CT)
        # ctd = pd.DataFrame(ctd, columns=cfg['out']['data_columns'], index=df_in.index)
    if 'Lat' in params_to_calc and not 'Lat' in ctd.columns:
        ctd['Lat'] = lat
        ctd['Lon'] = lon

    pd.options.mode.chained_assignment = pd_chained_assignment
    return ctd[cfg['out']['data_columns']]


##################################################################################
def main(new_arg=None):
    """

    :param new_arg: returns cfg if new_arg=='<cfg_from_args>' but it will be None if argument
     argv[1:] == '-h' or '-v' passed to this code
    argv[1] is cfgFile. It was used with cfg files:
        'csv2h5_nav_supervisor.ini'
        'csv2h5_IdrRedas.ini'
        'csv2h5_Idronaut.ini'
    :return:
    """

    global l
    cfg = cfg_from_args(my_argparser(), new_arg)
    if not cfg or not cfg['program'].get('return'):
        print('Can not initialise')
        return cfg
    elif cfg['program']['return'] == '<cfg_from_args>':  # to help testing
        return cfg

    l = init_logging(logging, None, cfg['program']['log'], cfg['program']['verbose'])
    print('\n', this_prog_basename(__file__), end=' started. ')
    try:
        cfg['in']['paths'], cfg['in']['nfiles'], cfg['in']['path'] = init_file_names(
            **{**cfg['in'], 'path': cfg['in']['db_path']}, b_interact=cfg['program']['b_interact'])
        set_field_if_no(cfg['in'], 'tables_log', '{}/logFiles')  # will be filled by each table from cfg['in']['tables']
        cfg['in']['query'] = query_time_range(**cfg['in'])
        set_field_if_no(cfg['out'], 'db_path', cfg['in']['db_path'])
        # cfg['out'] = init_file_names(cfg['out'], , path_field='db_path')
    except Ex_nothing_done as e:
        print(e.message)
        return ()

    # args = parser.parse_args()
    # args.verbose= args.verbose[0]
    # try:
    #     cfg= ini2dict(args.cfgFile)
    #     cfg['in']['cfgFile']= args.cfgFile
    # except IOError as e:
    #     print('\n==> '.join([a for a in e.args if isinstance(a,str)])) #e.message
    #     raise(e)
    # Open text log
    if 'log' in cfg['program'].keys():
        dir_create_if_need(os_path.dirname(cfg['program']['log']))
        flog = open(cfg['program']['log'], 'a+', encoding='cp1251')

    cfg['out']['log'] = {'fileName': None, 'fileChangeTime': None}

    # Prepare saving to csv
    if 'file_names_add_fun' in cfg['out']:
        file_names_add = eval(compile(cfg['out']['file_names_add_fun'], '', 'eval'))
    else:
        file_names_add = lambda i: '.csv'  # f'_{i}.csv'


    # Prepare data for output store and open it
    if cfg['out']['tables'] == ['None']:
        # will not write new data table and its log
        cfg['out']['tables'] = None
        # cfg['out']['tables_log'] = None  # for _runs cfg will be redefined (this only None case that have sense?)

    h5init(cfg['in'], cfg['out'])
    # store, df_log_old = h5temp_open(**cfg['out'])

    cfg_fileN = os_path.splitext(cfg['in']['cfgFile'])[0]
    out_tables_log = cfg['out'].get('tables_log')
    if cfg_fileN.endswith('_runs') or (bool(out_tables_log) and 'logRuns' in out_tables_log[0]):

        # Will calculate only after filter  # todo: calculate derived parameters before were they are bad (or replace all of them if any bad?)
        func_before_cycle = lambda x: None
        func_before_filter = lambda df, log_row, cfg: df
        func_after_filter = lambda df, cfg: log_runs(df, cfg, cfg['out']['log'])

        # this table will be added:
        cfg['out']['tables_log'] = [cfg['out']['tables'][0] + '/logRuns']
        cfg['out']['b_log_ready'] = True  # to not update time range in h5_append() from data having only metadata

        # Settings to not affect main data table and switch off not compatible options:
        cfg['out']['tables'] = []
        cfg['out']['b_skip_if_up_to_date'] = False  # todo: If False check it: need delete all previous result of CTD_calc() or set min_time > its last log time. True not implemented?
        cfg['program']['b_log_display'] = False  # can not display multiple rows log
        if 'b_save_images' in cfg['extract_runs']:
            cfg['extract_runs']['path_images'] = cfg['out']['db_path'].with_name('_subproduct')
            dir_create_if_need(cfg['extract_runs']['path_images'])
    else:
        if 'brown' in cfg_fileN.lower():
            func_before_cycle = load_coef
            if 'Lat' in cfg['in']:
                func_before_filter = lambda *args, **kwargs: add_ctd_params(process_brown(*args, **kwargs), kwargs['cfg'])
            else:
                func_before_filter = process_brown
        else:
            func_before_cycle = lambda x: None

            def ctd_coord_and_params(df: pd.DataFrame, log_row, cfg):
                coord_data_col_ensure(df, log_row)
                return add_ctd_params(df, cfg)


            func_before_filter = ctd_coord_and_params
        func_after_filter = lambda df, cfg: df  # nothing after filter

    func_before_cycle(cfg)  # prepare: usually assign data to cfg['for']
    if cfg['out'].get('path_csv'):
        dir_create_if_need(cfg['out']['path_csv'])
    # Load data Main circle #########################################
    # Open input store and cicle through input table log records
    qstr_trange_pattern = "index>=Timestamp('{}') & index<=Timestamp('{}')"
    iSt = 1

    df_log_old, cfg['out']['db'], cfg['out']['b_skip_if_up_to_date'] = h5temp_open(**cfg['out'])
    b_out_db_is_different = cfg['out']['db'] is not None and cfg['out']['db_path_temp'] != cfg['in']['db_path']
    # Cycle for each table, for each row in log:
    # for path_csv in gen_names_and_log(cfg['out'], df_log_old):
    with FakeContextIfOpen(lambda f: pd.HDFStore(f, mode='r'),
                           cfg['in']['db_path'],
                           None if b_out_db_is_different else cfg['out']['db']
                           ) as cfg['in']['db']:  # not opens ['in']['db'] if already opened to write

        for tbl in cfg['in']['tables']:
            if False:  # Show table info
                nodes = sorted(cfg['out']['db'].root.__members__)  # , key=number_key
                print(nodes)
            print(tbl, end='. ')

            df_log = cfg['in']['db'].select(cfg['in']['tables_log'].format(tbl) or tbl,
                where=cfg['in']['query']
                )
            if True:  # try:
                if 'log' in cfg['program'].keys():
                    nRows = df_log.rows.size
                    flog.writelines(datetime.now().strftime('\n\n%d.%m.%Y %H:%M:%S> processed ') + f'{nRows} row' + (
                        's:' if nRows > 1 else ':'))

                for ifile, r in enumerate(df_log.itertuples(), start=iSt):  # name=None
                    print('.', end='')
                    sys_stdout.flush()

                    path_raw = PurePath(r.fileName)
                    cfg['out']['log'].update(fileName=path_raw.name, fileChangeTime=r.fileChangeTime)
                    # save current state
                    cfg['in']['file_stem'] = cfg['out']['log']['fileName']  # for exmple to can extract date in subprogram
                    cfg['in']['fileChangeTime'] = cfg['out']['log']['fileChangeTime']

                    if cfg['in']['b_skip_if_up_to_date']:
                        b_stored_newer, b_stored_dups = h5del_obsolete(cfg['out'], cfg['out']['log'], df_log_old)
                        if b_stored_newer:
                            continue
                        if b_stored_dups:
                            cfg['out']['b_remove_duplicates'] = True
                    print('{}. {}'.format(ifile, path_raw.name), end=': ')

                    # Load data
                    qstr = qstr_trange_pattern.format(r.Index, r.DateEnd)
                    df_raw = cfg['in']['db'].select(tbl, qstr)
                    cols = df_raw.columns.tolist()


                    # cfg['in']['lat'] and ['lon'] may be need in add_ctd_params() if Lat not in df_raw
                    if 'Lat_en' in df_log.columns and 'Lat' not in cols:
                        cfg['in']['lat'] = np.nanmean((r.Lat_st, r.Lat_en))
                        cfg['in']['lon'] = np.nanmean((r.Lon_st, r.Lon_en))

                    df = func_before_filter(df_raw, log_row=r, cfg=cfg)

                    if df.size:  # size is zero means save only log but not data
                        # filter, updates cfg['out']['log']['rows']
                        df, _ = set_filterGlobal_minmax(df, cfg['filter'], cfg['out']['log'])
                    if 'rows' not in cfg['out']['log']:
                        l.warning('no data!')
                        continue
                    elif isinstance(cfg['out']['log']['rows'], int):
                        print('filtered out {rows_filtered}, remains {rows}'.format_map(cfg['out']['log']))
                        if cfg['out']['log']['rows']:
                            print('.', end='')
                        else:
                            l.warning('no data!')
                            continue

                    df = func_after_filter(df, cfg=cfg)

                    # Append to Store
                    h5_append(cfg['out'], df, cfg['out']['log'], log_dt_from_utc=cfg['in']['dt_from_utc'])

                    # Copy to csv
                    if cfg['out'].get('path_csv'):
                        fname = '{:%y%m%d_%H%M}-{:%d_%H%M}'.format(r.Index, r.DateEnd) + file_names_add(ifile)
                        if not 'data_columns' in cfg['out']:
                            cfg['out']['data_columns'] = slice(0, -1)  # all cols
                        df.to_csv(  # [cfg['out']['data_columns']]
                            cfg['out']['path_csv'] / fname, date_format=cfg['out']['text_date_format'],
                            float_format='%5.6g', index_label='Time')  # to_string, line_terminator='\r\n'

                    # Log to screen (if not prohibited explicitly)
                    if cfg['out']['log'].get('Date0') is not None and (
                            ('b_log_display' not in cfg['program']) or cfg['program']['b_log_display']):
                        str_log = '{fileName}:\t{Date0:%d.%m.%Y %H:%M:%S}-' \
                                  '{DateEnd:%d.%m %H:%M:%S%z}\t{rows}rows'.format_map(
                            cfg['out']['log'])  # \t{Lat}\t{Lon}\t{strOldVal}->\t{mag}
                        l.info(str_log)
                    else:
                        str_log = str(cfg['out']['log'].get('rows', '0'))
                    # Log to logfile
                    if 'log' in cfg['program'].keys():
                        flog.writelines('\n' + str_log)

    if b_out_db_is_different:
        try:
            if cfg['out']['tables'] is not None:
                print('')
                if cfg['out']['b_remove_duplicates']:
                    h5remove_duplicates(cfg['out'], cfg_table_keys=('tables', 'tables_log'))
                # Create full indexes. Must be done because of using ptprepack in h5move_tables() below
                l.debug('Create index')
                for tblName in (cfg['out']['tables'] + cfg['out']['tables_log']):
                    try:
                        cfg['out']['db'].create_table_index(tblName, columns=['index'], kind='full')
                    except Exception as e:
                        l.warning(': table {}. Index not created - error'.format(tblName), '\n==> '.join(
                            [s for s in e.args if isinstance(s, str)]))
        except Exception as e:
            l.exception('The end. There are error ')

            import traceback, code
            from sys import exc_info as sys_exc_info
            tb = sys_exc_info()[2]  # type, value,
            traceback.print_exc()
            last_frame = lambda tb=tb: last_frame(tb.tb_next) if tb.tb_next else tb
            frame = last_frame().tb_frame
            ns = dict(frame.f_globals)
            ns.update(frame.f_locals)
            code.interact(local=ns)
        finally:

            cfg['out']['db'].close()
            if cfg['program']['log']:
                flog.close()
            if cfg['out']['db'].is_open:
                print('Wait store is closing...')
                sleep(2)

            failed_storages = h5move_tables(cfg['out'])
            print('Finishing...' if failed_storages else 'Ok.', end=' ')
            h5index_sort(cfg['out'], out_storage_name=f"{cfg['out']['db_path'].stem}-resorted.h5",
                         in_storages=failed_storages)


if __name__ == '__main__':
    main()

""" trash ##############################################

timeit('max(P[islice])', number= 1000, globals=globals())
timeit('P[islice].max()', number= 1000, globals=globals())
            
iex = np.flatnonzero(np.diff(bbad))  # will be edges + extremums indices
if iex[0] != 0:
    iex = np.insert(iex, 0, 0)  # start edge added

if iex[-1] != N:
    bt = bbad[iex]  # top extremums (which starts to go "Down")
    iex = np.append(iex, N)  # end edge added
else:
    bt = bbad[iex[:-1]]  # top extremums

bl = ~bt  # bot extremums

# Removing bad extremums
Nremoved = 0
while True:
    # bbad - mask of candidates to remove from extremums
    if 'min_samples' in cfg_extract_runs and cfg_extract_runs['min_samples']:
        # length samples of each up/down interval:
        s = np.diff(iex)
        # find intervals with insufficient number of samples:
        if np.size(cfg_extract_runs['min_samples']) > 1:
            bbad = np.zeros_like(bt)
            bbad[bt] = s[bt] < cfg_extract_runs['min_samples'][0]  # down
            bbad[bl] = s[bl] < cfg_extract_runs['min_samples'][1]  # up
        else:
            bbad = s < cfg_extract_runs['min_samples']
    else:
        bbad = np.zeros_like(bt)

    if 'min_dp' in cfg_extract_runs and cfg_extract_runs['min_dp']:
        # height of each up/down interval:
        s = np.abs(np.diff(P[iex]))
        # find intervals with insufficient height:
        if np.size(cfg_extract_runs['min_dp']) > 1:
            bbad2 = np.zeros_like(bt)
            bbad2[bt] = s[bt] < cfg_extract_runs['min_dp'][0]  # down
            bbad2[bl] = s[bl] < cfg_extract_runs['min_dp'][1]  # up
            bbad = bbad | bbad2
        else:
            bbad = bbad | (s < cfg_extract_runs['min_dp'])

    n_bad_cur = np.sum(bbad)
    imin = iex[bt]
    imax = iex[bl]
    if Nremoved == n_bad_cur:  # nothing changed
        if __debug__:  # result should be logged:
            plt.figure(111), plt.hold(False)
            plt.plot(P, '.', color='c', alpha=0.5)  # plt.show()
            plt.hold(True)
            plt.plot(iex[bt], P(iex[bt]), '.g')
            plt.plot(iex[bl], P(iex[bl]), '.r')
        break
    else:
        Nremoved = np.sum(bbad)
        #     iminIn= find(bt)
        #     imaxIn= find(bl)
        #     bbad2(iminIn([false, (diff(P(imin))<0)]))= false
        #     bbad2(imaxIn([false, (diff(P(imax))>0)]))= false
        #     bbad2(~bbad)= true
        # Delete extremums only if not bok:
        bok = np.zeros_like(bt)
        bok[bt] = np.ediff1d(P[imin], to_end=-1) < 0  # continues up
        bok[bl] = np.ediff1d(P[imax], to_end=1) > 0  # continues down
        bok |= ~bbad
        bok[0] = True
        # Deleting
        iex = iex[np.append(bok, True)]
        bt = bt[bok]
        bl = bl[bok]

        # bok= np.insert(((np.diff(bt)!=0)&(np.diff(bl)!=0)), True, 0)
        ---
        # bok= np.logical_and(np.ediff1d(bt, to_begin= True)!=0,
        #                     np.ediff1d(bl, to_begin= True)!=0)        
        
        # iex= iex[np.append(bok, True)]
        # bt= bt[bok]
        # bl= bl[bok]
        
        
        
                    #if bbad[ 0]: bbad[ 1]= True
            #if bbad[-1]: bbad[-2]= True

            bbadt = np.logical_and(bt, bbad)[1:-1]
            bbadl = np.logical_and(bl, bbad)[1:-1]

            # del higest lowlands near deleted highlands:
            bbad= bbad|~bok

            b_after = np.hstack((False,False, bbadt))
            b_befor = np.hstack((bbadt, False,False))
            bbad[np.where(pex[b_after] > pex[b_befor],
                          np.flatnonzero(b_after), np.flatnonzero(b_befor))]= True

            # del lowest highlands near deleted lowlands:
            b_after = np.hstack((False,False, bbadl))
            b_befor = np.hstack((bbadl, False,False))
            bbad[np.where(pex[b_after] < pex[b_befor],
                          np.flatnonzero(b_after), np.flatnonzero(b_befor))]= True
        
"""
