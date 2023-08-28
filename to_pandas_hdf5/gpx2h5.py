#!/usr/bin/env python
# coding:utf-8
"""
  Author:  Andrey Korzh --<ao.korzh@gmail.com>
  Purpose: Convert gpx files to PyTables hdf5 file
  Created: 27.05.2015
"""
#

import logging
from codecs import open
from pathlib import PurePath
from sys import stdout as sys_stdout
from typing import Any, Dict, Mapping, MutableMapping, Union

import gpxpy
import numpy as np
import pandas as pd
from gpxpy.gpx import GPX
# my
from utils2init import cfg_from_args, my_argparser_common_part, init_file_names, Ex_nothing_done, set_field_if_no, \
    this_prog_basename, init_logging, standard_error_info, call_with_valid_kwargs
from to_pandas_hdf5.h5_dask_pandas import multiindex_timeindex, multiindex_replace, h5_append, \
    filterGlobal_minmax  # filter_global_minmax
from to_pandas_hdf5.h5toh5 import h5move_tables, h5index_sort, h5out_init, h5_dispenser_and_names_gen
from utils_time_corr import time_corr

dt64_1s = np.int64(1e9)
df_names = ['waypoints', 'tracks', 'segments', 'routes']

if __name__ == '__main__':
    l = None  # see main(): l = init_logging('', cfg['program']['log'], cfg['program']['verbose'])
else:
    l = logging.getLogger(__name__)


def my_argparser():
    p = my_argparser_common_part(
        {'description': 'Add data from *.gpx to *.h5'})

    s = p.add_argument_group('in', 'data')
    s.add('--path',
             help='path/mask to GPX file(s) to parse')
    s.add('--b_search_in_subdirs', default='False',
             help='used if mask or only dir in path (not full path) to search in subdirectories')
    s.add('--ext', default='gpx',
             help='used if only dir in path - extension of gpx files')
    s.add('--dt_from_utc_hours', default='0',
             help='add this correction to loading datetime data. May to use other suffixes instead of "hours"')
    s.add('--b_incremental_update', default='True',
             help='exclude processing of files with same name and which time change is not bigger than recorded in database (only prints ">" if detected). If finds updated version of same file then deletes all data which corresponds old file and after it brfore procesing of next files')
    s.add('--corr_time_mode', default='False',  # 'correct', 'sort_rows'
             help='if time not sorted then coorect it trying affecting minimum number of values. Used here for tracks/segments only. This is different from sorting rows which is performed at last step after the checking table in database')

    # Parameters specific to gpx
    s.add('--waypoints_cols_list', default='time, latitude, longitude, name, symbol, description',
             help='column names (comma separated) of gpxpy fields of gpx waypoints to load (symbol=sym,  description=cmt), first will be index. Its number and order must match out.waypoints_cols_list')
    s.add('--routes_cols_list', default='time, latitude, longitude, name, symbol, description',
             help='same as waypoints_cols_list but for routes')
    s.add('--tracks_cols_list', default='time, latitude, longitude',
             help='same as waypoints_cols_list but for tracks')
    s.add('--segments_cols_list', default='time, latitude, longitude',
             help='same as waypoints_cols_list but for segments')


    s = p.add_argument_group('out',
                             'all about output files')
    s.add('--db_path', help='hdf5 store file path')
    s.add('--table_prefix',
              help='prepend tables names to save data with this string (Note: _waypoints or _routes or ... suffix will be added automaticaly)')
    s.add('--tables_list', default='waypoints, tracks, tracks/segments, routes',
              help='tables names (comma separated) in hdf5 store to write data'
                   'keep them in logical order: [waypoints, tracks, tracks sections, routes]')
    s.add('--out.waypoints_cols_list', default='time, Lat, Lon, name, sym, cmt',
              help='column names (comma separated) in hdf5 table to write data, '
                   'its number and order must match in.waypoints_cols_list')
    s.add('--out.tracks_cols_list', default='time, Lat, Lon',
              help='same as waypoints_cols_list but for tracks')
    s.add('--out.segments_cols_list', default='time, Lat, Lon',
              help='same as waypoints_cols_list but for segments')
    s.add('--b_sort', default=True,
              help='may not needed for manually constructed sections. But it is difficult to use not sorted data')
    s.add('--b_insert_separator', default='False',
              help='insert NaNs row in table after each file data end')
    s.add('--b_reuse_temporary_tables', default='False',
              help='Warning! Set True only if temporary storage already have good data!'
                   'if True and b_incremental_update= True then not replace temporary storage with current storage before adding data to the temporary storage')

    # candidates to move out to common part
    s.add('--exclude_dirs_endswith_list', default='-, bad, test, TEST, toDel-',
             help='exclude dirs which ends with this srings. This and next option especially useful when search recursively in many dirs')
    s.add('--exclude_files_endswith_list', default='coef.txt, -.txt, test.txt',
             help='exclude files which ends with this srings')

    s = p.add_argument_group('filter',
                             'filter all data based on min/max of parameters')
    s.add('--min_date', help='minimum time')
    s.add('--max_date', help='maximum time')
    s.add('--min_DepEcho', help='minimum DepEcho (if it in out.{}_cols_list). Data rows will be deleted where it is below')
    s.add('--max_DepEcho', help='maximum DepEcho (if it in out.{}_cols_list). Data rows will be deleted where it is above')
    s.add('--fun_DepEcho', help='Numpy function name (fun) to delete rows where numpy.fun(DepEcho) is False')
    s.add('--min_dict', help='List with items in "key:value" format. Global filtering as for min_{param}. Todo: set to NaN data of ``key`` columns if it is below ``value``')
    s.add('--max_dict', help='List with items in "key:value" format. Global filtering as for min_{param}. Todo: set  to NaN data of ``key`` columns if it is above ``value``')
    s.add('--fun_dict', help='List with items in "key:fun" format. Global filtering as for min_{param}. Todo: set  to NaN data of ``key`` columns where numpy.fun(data) is False')
    s = p.add_argument_group('program', 'program behaviour')
    s.add('--return', default='<end>',  # nargs=1,
               choices=['<cfg_from_args>', '<gen_names_and_log>', '<end>'],
               help='<cfg_from_args>: returns cfg based on input args only and exit, <gen_names_and_log>: execute init_input_cols() and also returns fun_proc_loaded function... - see main()')
    return p


# ----------------------------------------------------------------------
def df_rename_cols(df, col_in, col_out):
    if df.empty:
        return
    df.index.name = col_out[0]  # ?df = df.reindex(df.index.rename(['Date']))
    df.rename(columns=dict(zip(col_in[1:], col_out[1:])), inplace=True)


def gpxConvert(cfg: Mapping[str, Any],
               fileInF: Union[GPX, PurePath, str]
               ) -> Dict[str, pd.DataFrame]:
    """
    Fill dataframes 'tracks', 'segments' and 'waypoints' with corresponding gpx objects data
    :param cfg: dict with keys
    - ['in']['*_cols'], where * is name of mentioned above gpx object with needed gpxpy properties to store.
    If absent then this object data will be ignored
    - ['out']['*_cols'] columns names of dataframes to rename gpxpy properties
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
        if c in cfg['in']:
            dfs[dfname] = pd.DataFrame(columns=cfg['in'][c][Ncols_t:])

    # check gpx is compatible to cfg segments
    try:
        set_atr_tr_seg = set(dir(gpx.tracks[0].segments[0].points[0]))
    except IndexError:
        set_atr_tr_seg = []

    tr_cols = cfg['in'].get('tracks_cols')
    if (sg_cols := cfg['in'].get('segments_cols')) is not None:
        if set_atr_tr_seg and not set_atr_tr_seg.issuperset(cfg['in']['segments_cols']):  # (not set_atr_tr_seg or
            cols_list = []
            for a in sg_cols:
                if a not in set_atr_tr_seg:
                    print('no such segment data col: "{}"'.format(a))
                else:
                    cols_list.append(a)
        else:
            cols_list = sg_cols
    else:
        cols_list = tr_cols

    if cols_list is not None:
        for track in gpx.tracks:
            for segment in track.segments:
                if sg_cols is not None:
                    dfs['segments'] = pd.concat([
                        dfs['segments'],
                        pd.DataFrame.from_records(
                            [[getattr(segment.points[0], c) for c in cols_list]],
                            columns=cols_list,
                            index=cols_list[0])
                        ])
                if tr_cols is not None:    # too tired to check compatibility to cfg tracks:
                    dfs['tracks'] = pd.concat([dfs['tracks'], pd.DataFrame.from_records(
                        [[getattr(point, c) for c in tr_cols] for point in segment.points],
                        columns=tr_cols,
                        index=tr_cols[0])])  # .__dict__ -> dir()

    # dfs['waypoints']= pd.DataFrame(columns= cfg['in']['waypoints_cols'][Ncols_t:])
    # pd.concat([dfs['waypoints'],
    if (wp_cols := cfg['in'].get('waypoints_cols')) is not None:
        dfs['waypoints'] = pd.DataFrame.from_records(
            [[getattr(waypoint, attr) for attr in wp_cols] for waypoint in gpx.waypoints],
            columns=wp_cols,
            index=wp_cols[0]
            )  # Waypoint changed time not affected by editing?
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


    for dfname, df in dfs.items():  # df_names
        c = dfname + '_cols'
        df_rename_cols(df, cfg['in'][c], cfg['out'][c])

    return dfs


def h5_sort_filt_append(
        df, input, out: MutableMapping,
        filter=None,
        process: Union[str, bool, None] = None
        ) -> Union[str, int]:
    """
    If specified then sorts index of df and filters by filterGlobal_minmax(),
    then appends to hdf5 store:
    - df to out['table'] in opened out['db'], and
    - corresponded row to out['table_log'] table
    :param df:
    :param input: cfg dict with fields... dt_from_utc: timedelta (optional), to correct out['log'] time
    :param out: out cfg, must have fields:
      - log
      - db
    :param filter (optional)
    :param process: how to deal with not sorted / duplicated index: see time_corr().
    :return: 'continue' if no data else df
    :updates out:
      - adds field 'tables_written': Set[Tuple[str, str]]
      - out['log']: fields 'Date0', 'DateEnd' if not cfg_out.get('b_log_ready')

    See also: filterGlobal_minmax
    """
    df_t_index, itm = multiindex_timeindex(df.index)
    # sorting will break multiindex?
    df_t_index, b_ok = time_corr(df_t_index, input, process)  # need sort in tracks/segments only
    df.index = multiindex_replace(df.index, df_t_index, itm)

    if filter:
        rows_in = len(df)
        bGood = filterGlobal_minmax(df, df.index, filter)
        df = df[bGood & b_ok & df.notna().any(axis=1)]
        out['log']['rows'] = len(df)
        print('filtered out {} from {}.'.format(rows_in - out['log']['rows'], rows_in))
    else:
        df = df[b_ok]
        out['log']['rows'] = len(df)
    if df.empty:
        print('No data => skip file')
        return df

    # # Log statistic
    # out['log']['Date0'  ]= timzone_view(df_t_index[ 0], input['dt_from_utc'])
    # out['log']['DateEnd']= timzone_view(df_t_index[-1], input['dt_from_utc'])
    # # Add separation row of NaN and save to store
    # if out['b_insert_separator'] and itm is None:
    #     # 0 (can not use np.nan in int) [tim[-1].to_datetime() + timedelta(seconds = 0.5/cfg['fs'])]
    #     df_dummy.index= (df.index[-1] + (df.index[-1] - df.index[-2])/2,)
    #     df= df.append(df_dummy)

    # store.append(tables[key], df, data_columns= True, index= False)
    # # Log to store #or , index=False?
    # dfLog= pd.DataFrame.from_records(out['log'], exclude= ['Date0'], index= [out['log']['Date0']]) #
    # #dfLog= pd.DataFrame.from_dict(out['log']) #, index= 'Date0'
    # store.append(tables_log[key], dfLog, data_columns= True, expectedrows= input['nfiles'], index=False) #append

    log_dt_from_utc = {'log_dt_from_utc': dt_from_utc} if (dt_from_utc := input.get('dt_from_utc')) else {}
    h5_append(out, df, out['log'], tim=df_t_index, **log_dt_from_utc)
    return df


# ##############################################################################
def main(new_arg=None):
    cfg = cfg_from_args(my_argparser(), new_arg)
    if not cfg or not cfg['program'].get('return'):
        print('Can not initialise')
        return cfg
    elif cfg['program']['return'] == '<cfg_from_args>':  # to help testing
        return cfg

    l = init_logging('', cfg['program']['log'], cfg['program']['verbose'])
    print('\n', this_prog_basename(__file__), end=' started. ')

    try:
        cfg['in']['paths'], cfg['in']['nfiles'], cfg['in']['path'] = init_file_names(
            **cfg['in'], b_interact=cfg['program']['b_interact'], cfg_search_parent=cfg['out'])
        h5out_init(cfg['in'], cfg['out'])
    except Ex_nothing_done as e:
        print(e.message)
        exit()

    if 'routes_cols' not in cfg['in']:
        cfg['in']['routes_cols'] = cfg['in']['waypoints_cols']
    if 'routes_cols' not in cfg['out']:
        cfg['out']['routes_cols'] = cfg['out']['waypoints_cols']  # cfg['in']['routes_cols']  #
    # Writing
    if True:  # try:
        l.warning('processing %d file%s', cfg['in']['nfiles'], 's:' if cfg['in']['nfiles'] > 1 else ':')
        cfg['out']['log'] = {}
        set_field_if_no(cfg['out'], 'table_prefix', PurePath(cfg['in']['path']).stem)
        cfg['out']['table_prefix'] = cfg['out']['table_prefix'].replace('-', '')
        if len([t for t in cfg['out']['tables'] if len(t)]) > 1:
            cfg['out']['tables'] = \
                [cfg['out']['table_prefix'] + '_' + s for s in cfg['out']['tables']]
            cfg['out']['tables_log'] = \
                [cfg['out']['table_prefix'] + '_' + s for s in cfg['out']['tables_log']]

        tables = dict(zip(df_names, cfg['out']['tables']))
        tables_log = dict(zip(df_names, cfg['out']['tables_log']))
        # Can not save path to DB (useless?) so set  for this max file name length:
        set_field_if_no(cfg['out'], 'logfield_fileName_len', 50)
        cfg['out']['index_level2_cols'] = cfg['in']['routes_cols'][0]

        # ###############################################################
        # ## Cumulate all data in cfg['out']['path_temp'] ##################
        ## Main circle ############################################################
        for i1_file, path_gpx in h5_dispenser_and_names_gen(cfg['in'], cfg['out']):
            l.info('{}. {}: '.format(i1_file, path_gpx.name))
            # Loading data
            dfs = gpxConvert(cfg, path_gpx)
            print('write', end=': '); sys_stdout.flush()
            for key, df in dfs.items():
                if (not tables.get(key)) or df.empty:
                    continue
                elif key == 'tracks':
                    # Save last time to can filter next file
                    cfg['in']['time_last'] = df.index[-1]

                sort_time = False if key in {'waypoints', 'routes'} else None

                # patching
                if 'tracker' in tables[key]:
                    # Also {} must be in tables[key]. todo: better key+'_fun_tracker' in cfg['out']?
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
                        cfg['out']['table'] = tables_pattern.format(trackers_numbers[sn])
                        cfg['out']['table_log'] = tables_log_pattern.format(trackers_numbers[sn])
                        call_with_valid_kwargs(h5_sort_filt_append, df ** cfg, input=cfg['in'], procss=sort_time)
                else:
                    cfg['out']['table'] = tables[key]
                    cfg['out']['table_log'] = tables_log[key]
                    call_with_valid_kwargs(h5_sort_filt_append, df, **cfg, input=cfg['in'], procss=sort_time)

    # try:
    # if cfg['out']['b_remove_duplicates']:
    #     for tbls in cfg['out']['tables_written']:
    #         for tblName in tbls:
    #             cfg['out']['db'][tblName].drop_duplicates(keep='last', inplace= True)
    # print('Create index', end=', ')

    # create_table_index calls create_table which docs sais "cannot index Time64Col() or ComplexCol"
    # so load it, index, then save
    # level2_index = None
    # df = cfg['out']['db'][tblName] # last commented
    # df.set_index([navp_all_index, level2_index])
    # df.sort_index()

    # cfg['out']['db'][tblName].sort_index(inplace=True)

    # if df is not None:  # resave
    #     df_log = cfg['out']['db'][tblName]
    #     cfg['out']['db'].remove(tbls[0])
    #     cfg['out']['db'][tbls[0]] = df
    #     cfg['out']['db'][tbls[1]] = df_log

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
    #     cfg['out']['db'].close()
    #     failed_storages= h5move_tables(cfg['out'], cfg['out']['tables_written'])

    try:
        failed_storages = h5move_tables(cfg['out'], tbl_names=cfg['out'].get('tables_written', set()))
        print('Finishing...' if failed_storages else 'Ok.', end=' ')
        # Sort if you have any processed data that needs it (not the case for the routes and waypoints), else don't because ``ptprepack`` not closes hdf5 source if it not finds data
        if cfg['in'].get('time_last') and cfg['out']['b_sort']:
            cfg['out']['b_remove_duplicates'] = True
            # do not tauch sections points order
            h5index_sort(cfg['out'], out_storage_name=f"{cfg['out']['db_path'].stem}-resorted.h5", in_storages=failed_storages,
                         tables=cfg['out'].get('tables_written', set()))
    except Ex_nothing_done:
        print('ok')

if __name__ == '__main__':
    main()

# trash
"""
with pd.HDFStore('d:\\WorkData\\BalticSea\\171003_ANS36\\171003Strahov_not_sorted.h5', mode='r') as store:
    print(repr(store.get_storer('/navigation/sectionsBaklan_d100_routes').table))
    print(repr(store['/navigation/sectionsBaklan_d100_routes'].index))


                store.append(cfg['out']['strProbe'], chunk[chunk.columns].astype('float32'), data_columns= True, index= False)
                store.append(cfg['out']['strProbe'], chunk[chunk.columns].astype('float32'), data_columns= True, index= False)

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
        with pd.get_store(cfg['out']['path_temp']) as store:
            s= store.get(cfg['out']['strProbe'])
            s= s.groupby(level=0).first()
            store.append(cfg['out']['strProbe'], s, append=False,
                         data_columns=True, expectedrows= s.shape[0])
            store.create_table_index(cfg['out']['strProbe'],
                                     columns=['index'], kind='full')
        #Save result in h5NameOut
        h5sort_pack(cfg['out']['path_temp'], h5NameOut, cfg['out']['strProbe'])
        print('ok')
"""
