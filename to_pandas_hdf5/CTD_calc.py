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
from datetime import datetime
from os import path as os_path
from sys import stdout as sys_stdout
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

if __debug__:
    from matplotlib import pyplot as plt
import gsw

# my
from utils2init import my_argparser_common_part, cfg_from_args, this_prog_basename, init_file_names, init_logging, \
    Ex_nothing_done, set_field_if_no, dir_create_if_need, FakeContextIfOpen
from utils_time import timzone_view
from other_filters import rep2mean
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
    p_in = p.add_argument_group('in', 'all about input files')
    p_in.add('--db_path', default='.',  # nargs=?,
             help='path to pytables hdf5 store to load data. May use patterns in Unix shell style')
    p_in.add('--tables_list',
             help='table name in hdf5 store to read data. If not specified then will be generated on base of path of input files')
    p_in.add('--tables_log',
             help='table name in hdf5 store to read data intervals. If not specified then will be "{}/logFiles" where {} will be replaced by current data table name')
    p_in.add('--dt_from_utc_hours', default='0',
             help='add this correction to loading datetime data. Can use other suffixes instead of "hours"')
    p_in.add('--b_skip_if_up_to_date', default='True',
             help='exclude processing of files with same name and wich time change is not bigger than recorded in database (only prints ">" if detected). If finds updated version of same file then deletes all data which corresponds old file and after it brfore procesing of next files')
    p_in.add('--b_temp_on_its90', default='True',
             help='When calc CTD parameters treat Temp have red on ITS-90 scale. (i.e. same as "temp90")')
    p_in.add('--path_coef',
             help='path to file with coefficients. Used for processing of Neil Brown CTD data')

    p_out = p.add_argument_group('output_files', 'all about output files')
    info_default_path = '[in] path from *.ini'
    p_out.add('--output_files.db_path', help='hdf5 store file path')
    p_out.add('--output_files.tables_list',
              help='table name in hdf5 store to write data. If not specified then it is same as input tables_list (only new subtable will created here), else it will be generated on base of path of input files')
    # p_out.add('--tables_list',
    #     #           help='tables names in hdf5 store to write data (comma separated)')
    p_out.add('--path_csv',
              help='path to output directory of csv file(s)')
    p_out.add('--data_columns_list',
              help='list of columns names used in output csv file(s)')
    p_out.add('--b_insert_separator', default='True',
              help='insert NaNs row in table after each file data end')
    p_out.add('--b_remove_duplicates', default='False', help='Set True if you see warnings about')
    p_out.add('--csv_date_format', default='%Y-%m-%d %H:%M:%S.%f',
              help='Format of date column in csv files. Can use float or string representations')

    p_run = p.add_argument_group('extract_runs', 'program behaviour')
    p_run.add('--cols_list', default='Pres',
              help='column for extract_runs (other common variant is "Depth")')
    p_run.add('--dt_between_min_minutes', default='1', help='')
    p_run.add('--min_dp', default='20', help='')
    p_run.add('--min_samples', default='200', help='100 use small value (10) for binned (averaged) samples')
    p_run.add('--b_keep_minmax_of_bad_files', default='False',
              help='keep 1 min before max and max of separated parts of data where movements insufficient to be runs')
    p_run.add('--b_save_images', default='True', help='to review split result')

    p_prog = p.add_argument_group('program', 'program behaviour')
    p_prog.add('--return', default='<end>',  # nargs=1,
               choices=['<cfg_from_args>', '<gen_names_and_log>', '<end>'],
               help='<cfg_from_args>: returns cfg based on input args only and exit, <gen_names_and_log>: execute init_input_cols() and also returns fun_proc_loaded function... - see main()')
    return (p)


def extractRuns(P: Sequence, cfg_extract_runs: Mapping[str, Any]) -> Tuple[List[int], List[int]]:
    '''
    Extract runs
    :param P: z coordinate - negative is below
    :param cfg_extract_runs: dict with fields:
        min_samples
        min_dp
    :return: [imin,imax]
    todo: replace with famouse line simplification algorithm
            or repeat function with result replacing finding of dP
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
        # Deleting not terminal opposit extremums near deleted extremums:
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
    :param cfg_extract_runs - settings dict with fields:
        'dt_between_min'
        'min_dp'
        'min_samples'
        ['dt_hole_max'] - split runs where dt between adjasent samples bigger. If not
        specified it is set equal to 'dt_between_min' automatically
        ['b_do'] - if it is set to False intepret all data as one run
        'b_keep_minmax_of_bad_files', optional - keep 1 min before max and max of separated parts of data where movements insufficient to be runs
    :return: iminmax: 2D numpy array np.int64([[minimums],[maximums]])
    '''

    if ('do' not in cfg_extract_runs) or cfg_extract_runs['b_do']:  # not do only if b_do is set to False
        P = np.abs(rep2mean(P))
        # if issubclass(x, DatetimeIndex)     # not works ??????????????
        # dnT= rep2mean(dnT, pd.notnull(dnT)) #           ??????????????
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

            # timeit('max(P[islice])', number= 1000, globals=globals())
            # timeit('P[islice].max()', number= 1000, globals=globals())

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

    # Practical Salinty PSS-78
    Val['Sal'] = gsw.SP_from_C(Val['Cond'], Val['Temp'], Val['Pres'])
    df = pd.DataFrame(Val, columns=cfg['output_files']['data_columns'], index=df_raw.index)

    # import seawater as sw
    # T90conv = lambda t68: t68/1.00024
    # Val['sigma0'] sw.pden(s, T90conv(t), p=0) - 1000
    return df


def log_runs(df_raw: pd.DataFrame,
             cfg: Mapping[str, Any],
             log: Optional[Mapping[str, Any]] = None) -> pd.DataFrame:
    """
    Changes log
    :param df_raw:
    :param cfg:
    :param log:
    :return: empty DataFrame (for compability)
    """

    if log is None:
        log = {'fileName': None,
               'fileChangeTime': None}
    else:
        log.clear()
    imin, imax = CTDrunsExtract(
        df_raw[cfg['extract_runs']['cols'][0]],
        df_raw.index.values,
        cfg['extract_runs'])
    df_log_st = df_raw.asof(df_raw.index[imin], subset=cfg['extract_runs']['cols']).add_suffix('_st')
    df_log_en = df_raw.asof(df_raw.index[imax], subset=cfg['extract_runs']['cols']).add_suffix('_en')
    log.update(  # pd.DataFrame(, index=df_log_st.index).rename_axis('Date0')
        {'Date0': timzone_view(df_log_st.index, cfg['in']['dt_from_utc']),
         'DateEnd': timzone_view(df_log_en.index, cfg['in']['dt_from_utc']),
         **dict([(i[0], i[1].values) for st_en in zip(df_log_st.items(), df_log_en.items()) for i in st_en]),
         'rows': imax - imin,
         'rows_filtered': imin - np.append(0, imax[:-1]),
         'fileName': [os_path.basename(cfg['in']['file_stem'])] * len(imin),
         'fileChangeTime': [cfg['in']['fileChangeTime']] * len(imin),
         })

    set_field_if_no(cfg['in'], 'table_nav', 'navigation')
    set_field_if_no(cfg['output_files'], 'dt_search_nav_tolerance', pd.Timedelta(minutes=2))

    dfNpoints = h5select(  # all starts then all ends in row
        cfg['in']['db'], cfg['in']['table_nav'], columns=['Lat', 'Lon', 'DepEcho'],
        time_points=df_log_st.index.append(df_log_en.index),
        dt_check_tolerance=cfg['output_files']['dt_search_nav_tolerance']
        )
    # todo: allow filter for individual columns. solution: use multiple calls for columns that need filtering with appropriate query_range_pattern argument of h5select()

    df_edges_items_list = [df_edge.add_suffix(sfx).items() for sfx, df_edge in (
        ('_st', dfNpoints.iloc[:len(df_log_st)]),
        ('_en', dfNpoints.iloc[len(df_log_st):len(dfNpoints)]))]
    log_update = {}
    for st_en in zip(*df_edges_items_list):
        for name, series in st_en:
            log_update[name] = series.values
    log.keys()
    log.update(log_update)

    # log.update(**dict([(i[0], i[1].values) for st_en in zip(
    #     *(dfNpoints.iloc[sl].add_suffix(sfx).items() for sfx, sl in (
    #         ('_st', slice(0, len(df_log_st))),
    #         ('_en', slice(len(df_log_st), len(dfNpoints)))
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
    return pd.DataFrame(data=None, columns=df_raw.columns, index=df_raw.index[[]])
    # df_raw.ix[[]] gets random error - bug in pandas


def add_ctd_params(df_in: Mapping[str, Sequence], cfg: Mapping[str, Any]):
    """
    Calculate all parameters from 'sigma0', 'depth', 'soundV', 'SA' that is specified in cfg['output_files']['data_columns']
    :param df_in: DataFrame with columns 'Pres', 'Temp'
    :param cfg: dict with fields:
        ['output_files']['data_columns'] - list of columns in output dataframe
        ['in'].['b_temp_on_its90'] - optional

    :return: DataFrame with only columns specified in cfg['output_files']['data_columns']
    """
    ctd = df_in
    params_to_calc = set(cfg['output_files']['data_columns']).difference(ctd.columns)
    params_coord_needed_for = params_to_calc.intersection(('depth', 'sigma0'))
    if any(params_coord_needed_for):
        # todo: load from nav:
        if not 'Lat' in ctd.columns:
            if 'lat' in cfg['in']:
                lat = cfg['in']['lat']
                lon = cfg['in']['lon']
            else:
                lat = 55.2  # 54.8707
                lon = 16.7  # 19.3212
                print('Calc {} using MANUAL INPUTTED coordinates: lat={lat}, lon={lon}'.format(
                    '/'.join(params_coord_needed_for), lon=lon, lat=lat))

    if 'Temp90' not in ctd.columns:
        if cfg['in'].get('b_temp_on_its90'):
            ctd['Temp90'] = ctd['Temp']
        else:
            ctd['Temp90'] = gsw.conversions.t90_from_t68(df_in['Temp'])

    ctd['SA'] = gsw.SA_from_SP(ctd['Sal'], ctd['Pres'], lat=ctd.get('Lat', lat),
                               lon=ctd.get('Lon', lon))  # or Sstar_from_SP() for Baltic where SA=S*
    # Val['Sal'] = gsw.SP_from_C(Val['Cond'], Val['Temp'], Val['P'])
    if 'soundV' in params_to_calc:
        ctd['soundV'] = gsw.sound_speed_t_exact(ctd['SA'], ctd['Temp90'], ctd['Pres'])

    if 'depth' in params_to_calc:
        ctd['depth'] = np.abs(gsw.z_from_p(np.abs(ctd['Pres']), ctd.get('Lat', lat)))
    if 'sigma0' in params_to_calc:
        CT = gsw.CT_from_t(ctd['SA'], ctd['Temp90'], ctd['Pres'])
        ctd['sigma0'] = gsw.sigma0(ctd['SA'], CT)
        # ctd = pd.DataFrame(ctd, columns=cfg['output_files']['data_columns'], index=df_in.index)
    return ctd[cfg['output_files']['data_columns']]


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
    print('\n' + this_prog_basename(__file__), end=' started. ')
    try:
        cfg['in'] = init_file_names(cfg['in'], cfg['program']['b_interact'], path_field='db_path')
        set_field_if_no(cfg['in'], 'tables_log', '{}/logFiles')
        cfg['in']['query'] = query_time_range(cfg['in'])
        set_field_if_no(cfg['output_files'], 'db_path', cfg['in']['db_path'])
        # cfg['output_files'] = init_file_names(cfg['output_files'], , path_field='db_path')
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
    set_field_if_no(cfg, 'filter', {})
    # Open text log
    if 'log' in cfg['program'].keys():
        dir_create_if_need(os_path.dirname(cfg['program']['log']))
        flog = open(cfg['program']['log'], 'a+', encoding='cp1251')
    str_log = ''
    cfg['output_files']['log'] = OrderedDict({'fileName': None,
                                              'fileChangeTime': None})

    # Prepare save to csv
    if 'file_names_add_fun' in cfg['output_files']:
        file_names_add = eval(compile(cfg['output_files']['file_names_add_fun'], '', 'eval'))
    else:
        file_names_add = lambda i: '.csv'  # f'_{i}.csv'

    cfg_out = cfg['output_files']
    if True:  # try:
        # Prepare data for output store and open it
        if cfg_out['tables'] == ['None']:
            cfg_out['tables'] = None
            cfg_out['tables_log'] = None  # also initialise to do nothing
        if cfg_out['tables'] is not None and not len(cfg_out['tables']):
            cfg_out['tables'] = cfg['in']['tables']
        h5init(cfg['in'], cfg_out)
        # store, dfLogOld = h5temp_open(cfg_out)

    cfg_fileN = os_path.splitext(cfg['in']['cfgFile'])[0]
    if cfg_fileN.endswith('_runs'):
        func_before_cycle = lambda x: None
        func_in_cycle = lambda df_raw, cfg: log_runs(df_raw, cfg, cfg_out['log'])
        cfg['filter'] = None
        # this table will be added:
        cfg_out['tables_log'] = [cfg_out['tables'][0] + '/logRuns']
        cfg_out['b_log_ready'] = True  # to not apdate time range in h5_append()
        # not affect main data table and switch off not compatible options:
        cfg_out['tables'] = []
        cfg_out['b_skip_if_up_to_date'] = False
        cfg['program']['b_log_display'] = False  # can not display multiple rows log
        # cfg_out['log']['Date0'] = timzone_view(df_raw.index[0], log_dt_from_utc)
        if 'b_save_images' in cfg['extract_runs']:
            cfg_out['dir'] = os_path.join(os_path.dirname(cfg_out['db_path']), '_subproduct')
            dir_create_if_need(cfg_out['dir'])
            cfg['extract_runs']['path_images'] = cfg_out['dir']
    else:
        if 'brown' in cfg_fileN.lower():
            func_before_cycle = load_coef
            if 'Lat' in cfg['in']:
                func_in_cycle = lambda *args, **kwargs: add_ctd_params(process_brown(*args, **kwargs), kwargs['cfg'])
            else:
                func_in_cycle = process_brown
        else:
            func_before_cycle = lambda x: None
            func_in_cycle = add_ctd_params

    func_before_cycle(cfg)  # prepare: usually assign data to cfg['for']
    if cfg_out.get('path_csv'):
        dir_create_if_need(cfg_out['path_csv'])
    # Load data Main circle #########################################
    # Open input store and cicle through input table log records
    qstr_trange_pattern = "index>=Timestamp('{}') & index<=Timestamp('{}')"
    iSt = 1

    dfLogOld = h5temp_open(cfg_out)
    b_out_db_is_different = cfg_out.get('db') is not None and cfg_out['db_path'] != cfg['in']['db_path']  # .is_open
    # for path_csv in gen_names_and_log(cfg_out, dfLogOld):
    with FakeContextIfOpen(lambda f: pd.HDFStore(f, mode='r'),
                           cfg_out['db'] if cfg_out.get('db') and not b_out_db_is_different else cfg['in']['db_path']
                           ) as cfg['in']['db']:  # not opens ['in']['db'] if already opened to write
        for tbl in cfg['in']['tables']:
            if False:  # Show table info
                cfg_out['db'].get_storer(tbl).table  # ?
                nodes = sorted(cfg_out['db'].root.__members__)  # , key=number_key
                print(nodes)
                # cfg_out['db'].get_node('CTD_Idronaut(Redas)').logFiles        # n_ext level nodes
            print(tbl, end='. ')
            # Process for each row in log and write multiple rows at once
            df_log = cfg['in']['db'].select(cfg['in']['tables_log'].format(tbl), where=cfg['in']['query'])
            if True:  # try:
                if 'log' in cfg['program'].keys():
                    nRows = df_log.rows.size
                    flog.writelines(datetime.now().strftime('\n\n%d.%m.%Y %H:%M:%S> processed ') + f'{nRows} row' + (
                        's:' if nRows > 1 else ':'))
                nfiles = 0
                for ifile, r in enumerate(df_log.itertuples(), start=iSt):  # name=None
                    print('.', end='')
                    sys_stdout.flush()
                    # str_time_short= '{:%d %H:%M}'.format(r.Index.to_datetime())
                    # timeUTC= r.Index.tz_convert(None).to_datetime()

                    nameFE = os_path.basename(r.fileName)
                    cfg_out['log'].update(fileName=os_path.basename(r.fileName), fileChangeTime=r.fileChangeTime)
                    # save current state
                    cfg['in']['file_stem'] = cfg_out['log']['fileName']  # for exmple to can extract date in subprogram
                    cfg['in']['fileChangeTime'] = cfg_out['log']['fileChangeTime']

                    if cfg['in']['b_skip_if_up_to_date']:
                        bExistOk, bExistDup = h5del_obsolete(cfg_out, cfg_out['log'], dfLogOld)
                        if bExistOk:
                            continue
                        if bExistDup:
                            cfg_out['b_remove_duplicates'] = True
                    print('{}. {}'.format(ifile, nameFE), end=': ')

                    # Get station data
                    qstr = qstr_trange_pattern.format(r.Index, r.DateEnd)
                    df_raw = cfg['in']['db'].select(tbl, qstr)
                    cols = df_raw.columns.tolist()
                    if 'Lat_en' in df_log.columns:
                        # cfg['in']['lat'] will not be used (overrided by df_raw['Lat']) if Lat in df_raw. Same for Lon
                        cfg['in']['lat'] = np.nanmean((r.Lat_st, r.Lat_en))
                        cfg['in']['lon'] = np.nanmean((r.Lon_st, r.Lon_en))
                    df = func_in_cycle(df_raw, cfg=cfg)

                    # filter
                    df, _ = set_filterGlobal_minmax(df, cfg['filter'], cfg_out['log'])
                    if not isinstance(cfg_out['log']['rows'], int):
                        print(
                            'runs lengths, initial fitered counts: {rows}, {rows_filtered}'.format_map(cfg_out['log']))
                    elif cfg_out['log']['rows']:  # isinstance(cfg_out['log']['rows'], int) and
                        print('filtered out {rows_filtered}, remains {rows}'.format_map(cfg_out['log']))
                        if cfg_out['log']['rows']:
                            print('.', end='')
                        else:
                            l.warning('no data!')
                            continue

                    # Append to Store
                    h5_append(cfg_out, df, cfg_out['log'], log_dt_from_utc=cfg['in']['dt_from_utc'])

                    # Copy to csv
                    if cfg_out.get('path_csv'):
                        fname = '{:%y%m%d_%H%M}-{:%d_%H%M}'.format(r.Index, r.DateEnd) + file_names_add(ifile)
                        if not 'data_columns' in cfg_out:
                            cfg_out['data_columns'] = slice(0, len(cols))
                        df.to_csv(  # [cfg_out['data_columns']]
                            cfg_out['path_csv'] / fname, date_format=cfg_out['csv_date_format'],
                            float_format='%5.6g', index_label='Time')  # to_string, line_terminator='\r\n'

                    # Log to screen (if not prohibited explicitly)
                    if cfg_out['log'].get('Date0') is not None and (
                            ('b_log_display' not in cfg['program']) or cfg['program']['b_log_display']):
                        str_log = '{fileName}:\t{Date0:%d.%m.%Y %H:%M:%S}-' \
                                  '{DateEnd:%d. %H:%M:%S%z}\t{rows}rows'.format_map(
                            cfg_out['log'])  # \t{Lat}\t{Lon}\t{strOldVal}->\t{mag}
                        print(str_log)
                    else:
                        str_log = str(cfg_out['log']['rows'])
                        # Log to logfile
                    if 'log' in cfg['program'].keys():
                        flog.writelines('\n' + str_log)

    if b_out_db_is_different:
        try:
            if cfg_out['tables'] is not None:
                print('')
                if cfg_out['b_remove_duplicates']:
                    h5remove_duplicates(cfg_out, cfg_table_keys=('tables', 'tables_log'))
                # Create full indexes. Must be done because of using ptprepack in h5move_tables() below
                l.debug('Create index')
                for tblName in (cfg_out['tables'] + cfg_out['tables_log']):
                    try:
                        cfg_out['db'].create_table_index(tblName, columns=['index'], kind='full')
                    except Exception as e:
                        l.warning(': table {}. Index not created - error'.format(tblName), '\n==> '.join(
                            [s for s in e.args if isinstance(s, str)]))
        except Exception as e:
            l.error('The end. There are error ' + str(e.__class__) + ':\n==> '.join(
                [s for s in e.args if isinstance(s, str)]))

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

            cfg_out['db'].close()
            if cfg['program']['log']:
                flog.close()
            if cfg_out['db'].is_open:
                print('Wait store is closing...')
                sleep(2)

            new_storage_names = h5move_tables(cfg_out)
            print('Ok.', end=' ')
        h5index_sort(cfg_out, out_storage_name=cfg_out['db_base'] + '-resorted.h5', in_storages=new_storage_names)


if __name__ == '__main__':
    main()

""" trash ##############################################
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
