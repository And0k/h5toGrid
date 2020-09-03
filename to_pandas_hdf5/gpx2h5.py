#!/usr/bin/env python
# coding:utf-8
"""
  Author:  Andrey Korzh --<ao.korzh@gmail.com>
  Purpose: Convert gpx files to PyTables hdf5 file
  Created: 27.05.2015
"""
#from __future__ import print_function

import logging
from codecs import open
from pathlib import PurePath
from sys import stdout as sys_stdout
from typing import Any, Dict, Mapping, Union

import gpxpy
import numpy as np
import pandas as pd
from gpxpy.gpx import GPX
# my
from utils2init import cfg_from_args, my_argparser_common_part, init_file_names, Ex_nothing_done, set_field_if_no, \
    this_prog_basename, init_logging, standard_error_info
from to_pandas_hdf5.csv2h5 import h5_dispenser_and_names_gen
from to_pandas_hdf5.h5_dask_pandas import multiindex_timeindex, multiindex_replace, h5_append, \
    filterGlobal_minmax  # filter_global_minmax
from to_pandas_hdf5.h5toh5 import h5move_tables, h5index_sort, h5init
from utils_time_corr import time_corr

dt64_1s = np.int64(1e9)
df_names = ['waypoints', 'tracks', 'segments', 'routes']

if __name__ == '__main__':
    l = None  # see main(): l = init_logging(logging, None, cfg['program']['log'], cfg['program']['verbose'])
else:
    l = logging.getLogger(__name__)


def my_argparser():
    p = my_argparser_common_part(
        {'description': 'Add data from *.gpx to *.h5'})

    p_in = p.add_argument_group('in', 'data')
    p_in.add('--path',
             help='path/mask to GPX file(s) to parse')
    p_in.add('--b_search_in_subdirs', default='False',
             help='used if mask or only dir in path (not full path) to search in subdirectories')
    p_in.add('--ext', default='gpx',
             help='used if only dir in path - extension of gpx files')
    p_in.add('--dt_from_utc_hours', default='0',
             help='add this correction to loading datetime data. May to use other suffixes instead of "hours"')
    p_in.add('--waypoints_cols_list', default='time, latitude, longitude, name, symbol, description',
             help='column names (comma separated) of gpxpy fields of gpx waypoints to load (symbol=sym,  description=cmt), first will be index. Its number and order must match output_files.waypoints_cols_list')
    p_in.add('--routes_cols_list', default='time, latitude, longitude, name, symbol, description',
             help='same as waypoints_cols_list but for routes')
    p_in.add('--tracks_cols_list', default='time, latitude, longitude',
             help='same as waypoints_cols_list but for tracks')
    p_in.add('--segments_cols_list', default='time, latitude, longitude',
             help='same as waypoints_cols_list but for segments')
    p_in.add('--b_skip_if_up_to_date', default='True',
             help='exclude processing of files with same name and which time change is not bigger than recorded in database (only prints ">" if detected). If finds updated version of same file then deletes all data which corresponds old file and after it brfore procesing of next files')
    p_in.add('--b_make_time_inc', default='False',  # 'correct', 'sort_rows'
             help='if time not sorted then coorect it trying affecting small number of values. Used here for tracks/segments only. This is different from sorting rows which is performed at last step after the checking table in database')

    p_out = p.add_argument_group('output_files', 'all about output files')
    p_out.add('--db_path', help='hdf5 store file path')
    p_out.add('--table_prefix',
              help='prepend tables names to save data with this string (Note: _waypoints or _routes or ... suffix will be added automaticaly)')
    p_out.add('--tables_list', default='waypoints, tracks, tracks/segments, routes',
              help='tables names (comma separated) in hdf5 store to write data'
                   'keep them in logical order: [waypoints, tracks, tracks sections, routes]')
    p_out.add('--output_files.waypoints_cols_list', default='time, Lat, Lon, name, sym, cmt',
              help='column names (comma separated) in hdf5 table to write data, '
                   'its number and order must match in.waypoints_cols_list')
    p_out.add('--output_files.tracks_cols_list', default='time, Lat, Lon',
              help='same as waypoints_cols_list but for tracks')
    p_out.add('--output_files.segments_cols_list', default='time, Lat, Lon',
              help='same as waypoints_cols_list but for segments')

    p_out.add('--b_insert_separator', default='False',
              help='insert NaNs row in table after each file data end')
    p_out.add('--b_use_old_temporary_tables', default='False',
              help='Warning! Set True only if temporary storage already have good data!'
                   'if True and b_skip_if_up_to_date= True then not replace temporary storage with current storage before adding data to the temporary storage')

    # candidates to move out to common part
    p_in.add('--exclude_dirs_ends_with_list', default='-, bad, test, TEST, toDel-',
             help='exclude dirs which ends with this srings. This and next option especially useful when search recursively in many dirs')
    p_in.add('--exclude_files_ends_with_list', default='coef.txt, -.txt, test.txt',
             help='exclude files which ends with this srings')

    p_filt = p.add_argument_group('filter', 'filter all data based on min/max of parameters')
    p_filt.add('--date_min', help='minimum time')
    p_filt.add('--date_max', help='maximum time')

    p_prog = p.add_argument_group('program', 'program behaviour')
    p_prog.add('--return', default='<end>',  # nargs=1,
               choices=['<cfg_from_args>', '<gen_names_and_log>', '<end>'],
               help='<cfg_from_args>: returns cfg based on input args only and exit, <gen_names_and_log>: execute init_input_cols() and also returns fun_proc_loaded function... - see main()')
    return p


# ----------------------------------------------------------------------
def gpxConvert(cfg: Mapping[str, Any],
               fileInF: Union[GPX, PurePath, str]
               ) -> Dict[str, pd.DataFrame]:
    """
    Fill dataframes 'tracks', 'segments' and 'waypoints' with corresponding gpx objects data
    :param cfg: dict with keys
    ['in']['*_cols'], where * is name of mentioned above gpx object with needed gpxpy properties to store
    ['output_files']['*_cols'] columns names of dataframes to rename gpxpy properties
    :param fileInF: gpx file path or gpxpy gpx class instance
    :return: dict of dataframes with keys 'tracks', 'segments' and 'waypoints'
    """
    if isinstance(fileInF, (str, PurePath)):
        # fileInF is not gpx class => init this class
        print('read', end=', ');
        sys_stdout.flush()
        with open(fileInF, 'r', encoding='utf-8') as gpx_file:
            gpx = gpxpy.parse(gpx_file)
    else:  # isinstance(fileInF, gpxpy.GPX)
        gpx = fileInF
    Ncols_t = 1
    dfs = {}
    print('convert', end=', ');
    sys_stdout.flush()
    for dfname in df_names:
        c = dfname + '_cols'
        dfs[dfname] = pd.DataFrame(columns=cfg['in'][c][Ncols_t:])

    # check gpx is compatible to cfg segments
    try:
        set_atr_tr_seg = set(dir(gpx.tracks[0].segments[0].points[0]))
    except IndexError:
        set_atr_tr_seg = []
    if set_atr_tr_seg and not set_atr_tr_seg.issuperset(cfg['in']['segments_cols']):  # (not set_atr_tr_seg or
        cols_list = []
        for a in cfg['in']['segments_cols']:
            if a not in set_atr_tr_seg:
                print('no such segment data col: "{}"'.format(a))
            else:
                cols_list.append(a)
    else:
        cols_list = cfg['in']['segments_cols']

    # too tired to check compability to cfg tracks:
    tr_cols = cfg['in']['tracks_cols']
    for track in gpx.tracks:
        for segment in track.segments:
            dfs['segments'] = pd.concat([dfs['segments'],
                                         pd.DataFrame.from_records([[getattr(segment.points[0], c) for c in cols_list]],
                                                                   columns=cols_list,
                                                                   index=cols_list[0])])
            dfs['tracks'] = pd.concat([dfs['tracks'], pd.DataFrame.from_records(
                [[getattr(point, c) for c in tr_cols] for point in segment.points],
                columns=tr_cols,
                index=tr_cols[0])])  # .__dict__ -> dir()

    # dfs['waypoints']= pd.DataFrame(columns= cfg['in']['waypoints_cols'][Ncols_t:])
    # pd.concat([dfs['waypoints'],
    dfs['waypoints'] = pd.DataFrame.from_records(
        [[getattr(waypoint, attr) for attr in cfg['in']['waypoints_cols']] for waypoint in gpx.waypoints],
        columns=cfg['in']['waypoints_cols'],
        index=cfg['in']['waypoints_cols'][0])  # Waypoint changed time not affected by editing?
    dfs['waypoints'].index = pd.DatetimeIndex(dfs['waypoints'].index)
    # for waypoint in gpx.waypoints:
    # print('waypoint {0} -&gt; ({1},{2})'.format(waypoint.name, waypoint.latitude, waypoint.longitude))
    # for route in gpx.routes:
    # print 'Route:'
    # for point in route.points:
    # print 'Point at ({0},{1}) -&gt; {2}'.format(point.latitude, point.longitude, point.elevation)
    # df= pd.DataFrame(a[:,Ncols_t:].view(dtype=np.uint16),
    # columns= cfg['Header'][Ncols_t:],
    # dtype= np.uint16,
    # index= gpx.time + cfg['TimeAdd'])

    if gpx.routes:
        dfs['routes'] = pd.concat([pd.DataFrame.from_records([[
            getattr(waypoint, attr) for attr in cfg['in']['routes_cols']] for waypoint in route.points],
            columns=cfg['in']['routes_cols'],
            index=cfg['in']['routes_cols'][0]) for route in gpx.routes],
            copy=False,
            keys=[r.name for r in gpx.routes])

    def df_rename_cols(df, col_in, col_out, Ncols_t=1):
        if df.empty:
            return
        df.index.name = col_out[0]  # ?df = df.reindex(df.index.rename(['Date']))
        df.rename(columns=dict(zip(col_in[Ncols_t:], col_out[Ncols_t:])), inplace=True)

    for dfname in df_names:
        c = dfname + '_cols'
        df_rename_cols(dfs[dfname], cfg['in'][c], cfg['output_files'][c])

    return dfs


def df_filter_and_save_to_h5(cfg_out, cfg, df, key):
    df_t_index, itm = multiindex_timeindex(df.index)
    # sorting will break multiindex?
    df_t_index, b_ok = time_corr(df_t_index, cfg['in'], b_make_time_inc=key not in {'waypoints',
                                                                                    'routes'})  # need sort in tracks/segments only
    df.index = multiindex_replace(df.index, df_t_index, itm)

    if 'filter' in cfg:
        rows_in = len(df)
        bGood = filterGlobal_minmax(df, df.index, cfg['filter'])
        df = df[bGood]
        cfg_out['log']['rows'] = len(df)
        print('filtered out {} from {}.'.format(rows_in - cfg_out['log']['rows'], rows_in))
    else:
        cfg_out['log']['rows'] = len(df)
    if df.empty:
        print('No data => skip file')
        return 'continue'

    # # Log statistic
    # cfg_out['log']['Date0'  ]= timzone_view(df_t_index[ 0], cfg['in']['dt_from_utc'])
    # cfg_out['log']['DateEnd']= timzone_view(df_t_index[-1], cfg['in']['dt_from_utc'])
    # # Add separatiion row of NaN and save to store
    # if cfg_out['b_insert_separator'] and itm is None:
    #     # 0 (can not use np.nan in int) [tim[-1].to_datetime() + timedelta(seconds = 0.5/cfg['fs'])]
    #     df_dummy.index= (df.index[-1] + (df.index[-1] - df.index[-2])/2,)
    #     df= df.append(df_dummy)

    # store.append(tables[key], df, data_columns= True, index= False)
    # # Log to store #or , index=False?
    # dfLog= pd.DataFrame.from_records(cfg_out['log'], exclude= ['Date0'], index= [cfg_out['log']['Date0']]) #
    # #dfLog= pd.DataFrame.from_dict(cfg_out['log']) #, index= 'Date0'
    # store.append(tables_log[key], dfLog, data_columns= True, expectedrows= cfg['in']['nfiles'], index=False) #append

    h5_append(cfg_out, df, cfg_out['log'], cfg['in']['dt_from_utc'], tim=df_t_index)
    cfg_out['tables_have_wrote'].add((cfg_out['table'], cfg_out['table_log']))
    return 0


# ##############################################################################
def main(new_arg=None):
    cfg = cfg_from_args(my_argparser(), new_arg)
    if not cfg or not cfg['program'].get('return'):
        print('Can not initialise')
        return cfg
    elif cfg['program']['return'] == '<cfg_from_args>':  # to help testing
        return cfg

    l = init_logging(logging, None, cfg['program']['log'], cfg['program']['verbose'])
    print('\n' + this_prog_basename(__file__), end=' started. ')

    try:
        cfg['in'] = init_file_names(cfg['in'], cfg['program']['b_interact'])
        cfg_out = cfg['output_files']
        h5init(cfg['in'], cfg_out)
    except Ex_nothing_done as e:
        print(e.message)
        exit()

    df_dummy = pd.DataFrame(
        np.full(1, np.NaN, dtype=np.dtype({
            'formats': ['float64', 'float64'],
            'names': cfg_out['tracks_cols'][1:]})),
        index=(pd.NaT,))  # used for insert separator lines

    if 'routes_cols' not in cfg['in']:
        cfg['in']['routes_cols'] = cfg['in']['waypoints_cols']
    if 'routes_cols' not in cfg_out:
        cfg_out['routes_cols'] = cfg_out['waypoints_cols']  # cfg['in']['routes_cols']  #
    # Writing
    if True:  # try:
        l.warning('processing ' + str(cfg['in']['nfiles']) + ' file' + 's:' if cfg['in']['nfiles'] > 1 else ':')
        cfg_out['log'] = {}
        set_field_if_no(cfg_out, 'table_prefix', PurePath(cfg['in']['filemask']).stem)
        cfg_out['table_prefix'] = cfg_out['table_prefix'].replace('-', '')
        if len([t for t in cfg_out['tables'] if len(t)]) > 1:
            cfg_out['tables'] = \
                [cfg_out['table_prefix'] + '_' + s for s in cfg_out['tables']]
            cfg_out['tables_log'] = \
                [cfg_out['table_prefix'] + '_' + s for s in cfg_out['tables_log']]

        tables = dict(zip(df_names, cfg_out['tables']))
        tables_log = dict(zip(df_names, cfg_out['tables_log']))
        cfg_out['tables_have_wrote'] = set()
        # Can not save path to DB (useless?) so set  for this max file name length:
        set_field_if_no(cfg_out, 'logfield_fileName_len', 50)
        cfg_out['index_level2_cols'] = cfg['in']['routes_cols'][0]

        # ###############################################################
        # ## Cumulate all data in cfg_out['path_temp'] ##################
        ## Main circle ############################################################
        for i1_file, path_gpx in h5_dispenser_and_names_gen(cfg, cfg_out):
            l.info('{}. {}: '.format(i1_file, path_gpx.name))

            # Loading data
            dfs = gpxConvert(cfg, path_gpx)
            # Add time shift specified in configuration .ini
            print('write', end=': ');
            sys_stdout.flush()
            for key, df in dfs.items():
                if (not tables.get(key)) or df.empty:
                    continue
                elif key == 'tracks':
                    # Save last time to can filter next file
                    cfg['in']['time_last'] = df.index[-1]

                # monkey patching
                if 'tracker' in tables[key]:
                    # Also {} must be in tables[key]. todo: better key+'_fun_tracker' in cfg_out?
                    # Trackers processing
                    trackers_numbers = {
                        '0-3106432': '1',
                        '0-2575092': '2',
                        '0-3124620': '3',
                        '0-3125300': '4',
                        '0-3125411': '5',
                        '0-3126104': '6'}
                    tables_pattern = tables[key]
                    tables_log_pattern = tables_log[key]

                    df['comment'] = df['comment'].str.split(" @", n=1, expand=True)[0]
                    # split data and save to multipe tables
                    df_all = df.set_index(['comment', df.index])
                    for sn, n in trackers_numbers.items():  # set(df_all.index.get_level_values(0))
                        try:
                            df = df_all.loc[sn]
                        except KeyError:
                            continue
                        # redefine saving parameters
                        cfg_out['table'] = tables_pattern.format(trackers_numbers[sn])
                        cfg_out['table_log'] = tables_log_pattern.format(trackers_numbers[sn])
                        df_filter_and_save_to_h5(cfg_out, cfg, df, key)
                else:
                    cfg_out['table'] = tables[key]
                    cfg_out['table_log'] = tables_log[key]
                    df_filter_and_save_to_h5(cfg_out, cfg, df, key)

    # try:
    # if cfg_out['b_remove_duplicates']:
    #     for tbls in cfg_out['tables_have_wrote']:
    #         for tblName in tbls:
    #             cfg_out['db'][tblName].drop_duplicates(keep='last', inplace= True)
    # print('Create index', end=', ')

    # create_table_index calls create_table which docs sais "cannot index Time64Col() or ComplexCol"
    # so load it, index, then save
    # level2_index = None
    # df = cfg_out['db'][tblName] # last commented
    # df.set_index([navp_all_index, level2_index])
    # df.sort_index()

    # cfg_out['db'][tblName].sort_index(inplace=True)

    # if df is not None:  # resave
    #     df_log = cfg_out['db'][tblName]
    #     cfg_out['db'].remove(tbls[0])
    #     cfg_out['db'][tbls[0]] = df
    #     cfg_out['db'][tbls[1]] = df_log

    try:
        pass
    except Exception as e:
        print('The end. There are error ', standard_error_info(e))

    #     import traceback, code
    #     from sys import exc_info as sys_exc_info
    #
    #     tb = sys_exc_info()[2]  # type, value,
    #     traceback.print_exc()
    #     last_frame = lambda tb=tb: last_frame(tb.tb_next) if tb.tb_next else tb
    #     frame = last_frame().tb_frame
    #     ns = dict(frame.f_globals)
    #     ns.update(frame.f_locals)
    #     code.interact(local=ns)
    # finally:
    #     cfg_out['db'].close()
    #     failed_storages= h5move_tables(cfg_out, cfg_out['tables_have_wrote'])


    failed_storages = h5move_tables(cfg_out, tbl_names=cfg_out['tables_have_wrote'])
    print('Finishing...' if failed_storages else 'Ok.', end=' ')
    if cfg['in'].get('time_last'):
        # if have any processed data that need to be sorted (not the case for the routes and waypoints), also needed because ``ptprepack`` not closes hdf5 source if it not finds data
        cfg_out['b_remove_duplicates'] = True
        h5index_sort(cfg_out, out_storage_name=cfg_out['db_base'] + '-resorted.h5', in_storages=failed_storages,
                     tables=cfg_out['tables_have_wrote'])


if __name__ == '__main__':
    main()

# trash
"""
with pd.HDFStore('d:\\WorkData\\BalticSea\\171003_ANS36\\171003Strahov_not_sorted.h5', mode='r') as store:
    print(repr(store.get_storer('/navigation/sectionsBaklan_d100_routes').table))
    print(repr(store['/navigation/sectionsBaklan_d100_routes'].index))


                store.append(cfg_out['strProbe'], chunk[chunk.columns].astype('float32'), data_columns= True, index= False)
                store.append(cfg_out['strProbe'], chunk[chunk.columns].astype('float32'), data_columns= True, index= False)

                #try#chunk.columns
                strLog= '{Date0:%d.%m.%Y %H:%M:%S} - {DateEnd:%d.%m.%Y %H:%M:%S}'.format(**log) #\t{Lat}\t{Lon}\t{strOldVal}->\t{mag}
                print(strLog)
                if 'log' in cfg['program'].keys():
                    f.writelines( '\n' + nameFE + '\t' + strLog)
        try:
            print('ok')
        except Exception as e:
            print('The end. There are errors: ', e.message)
        finally:
            if 'log' in cfg['program'].keys():
                f.close()

        #Remove duplicates and add index
        with pd.get_store(cfg_out['path_temp']) as store:
            s= store.get(cfg_out['strProbe'])
            s= s.groupby(level=0).first()
            store.append(cfg_out['strProbe'], s, append=False,
                         data_columns=True, expectedrows= s.shape[0])
            store.create_table_index(cfg_out['strProbe'],
                                     columns=['index'], kind='full')
        #Save result in h5NameOut
        h5sort_pack(cfg_out['path_temp'], h5NameOut, cfg_out['strProbe'])
        print('ok')
"""
