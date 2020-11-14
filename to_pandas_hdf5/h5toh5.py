#!/usr/bin/env python
# coding:utf-8
"""
  Author:  Andrey Korzh --<ao.korzh@gmail.com>

"""
from __future__ import print_function, division

import logging
from pathlib import Path, PurePath
import re
import sys  # from sys import argv
import warnings
from os import path as os_path, getcwd as os_getcwd, chdir as os_chdir, remove as os_remove
from time import sleep
from typing import Any, Dict, Iterable, Iterator, Mapping, MutableMapping, Optional, Sequence, Tuple, Union, List, Set

import numpy as np
import pandas as pd
from tables import NaturalNameWarning
from tables.exceptions import HDF5ExtError, ClosedFileError
from tables.scripts.ptrepack import main as ptrepack

if __debug__:
    from matplotlib import pyplot as plt
warnings.catch_warnings()
warnings.simplefilter("ignore", category=NaturalNameWarning)
# warnings.filterwarnings('ignore', category=tables.NaturalNameWarning)
# my
from other_filters import inearestsorted, inearestsorted_around, check_time_diff
from utils2init import set_field_if_no, pathAndMask, dir_create_if_need, getDirBaseOut, FakeContextIfOpen, \
    Ex_nothing_done, standard_error_info
from utils_time import multiindex_timeindex

pd.set_option('io.hdf.default_format', 'table')
if __name__ != '__main__':
    l = logging.getLogger(__name__)


def main():
    """
    Purpose: execute query from command line and returns data to stdout
    (or wrtes to new hdf5 - to do) from PyTables hdf5 file
    Created: 30.07.2013

    """
    import argparse
    parser = argparse.ArgumentParser(description='Save part of source file.')
    parser.add_argument('Input', nargs=2, type=str,
                        help='Source file full path, Node name')
    parser.add_argument('-columns', nargs='*', type=str)
    parser.add_argument('-where', nargs='*', type=str)
    parser.add_argument('-chunkDays', type=float)
    # to do:
    parser.add_argument('-saveTo', type=str,
                        help='Save result (default: not save, index of first )')

    args = parser.parse_args()

    def proc_h5toh5(args):
        '''
        args: argparse.Namespace, must have attributes:
            Input[0] - name of hdf5 PyTables file
            Input[1] - name of table in this file
            where    - query for data in this table
            args may have fields:
            columns  - name of columns to return
        :return: numpy array of int, indexes of data satisfied query
        '''
        fileInF = args.Input[0]
        strProbe = args.Input[1]
        str_where = args.where
        with pd.get_store(fileInF, mode='r') as store:  # error if open with fileInF
            # store= pd.HDFStore(fileInF, mode='r')
            try:
                if not args.chunkDays: args.chunkDays = 1
                if str_where:  # s=str_where[0]
                    Term = []
                    bWate = False
                    for s in str_where:
                        if bWate:
                            if s[-1] == '"':
                                bWate = False
                                Term[-1] += f' {s[:-1]}'
                            else:
                                Term[-1] += f' {s}'
                        elif s[0] == '"':
                            bWate = True
                            Term.append(s[1:])
                        else:
                            Term.append(s)
                    # Term= [pd.Term(s[1:-1]) if s[-1]==s[0]=='"' else pd.Term(s) for s in str_where]
                    # Term= pd.Term(str_where)
                    if args.columns:
                        df = store.select(strProbe, Term, columns=args.columns)
                    else:
                        df = store.select(strProbe, Term)
                    df = df.index
                    # start=0,  stop=10)
                    coord = store.select_as_coordinates(strProbe, Term).values[[0, -1]]
                else:
                    df = store.select_column(strProbe, 'index')
                    coord = [0, df.shape[0]]
            except:

                if str_where:
                    #  this  is  in-memory  version  of  this  type  of  selection
                    df = store.select(strProbe)
                    coord = [0, df.shape[0]]
                    df = df[eval(str_where)]
                    df = df.index
                else:
                    df = store.select_column(strProbe, 'index')
                    coord = [0, df.shape[0]]
        # df= store.get(strProbe)
        # store.close()
        if df.shape[0] > 0:
            tGrid = np.arange(df[0].date(), df[df.shape[0] - 1].date() +
                              pd.Timedelta(days=1), pd.Timedelta(days=args.chunkDays),
                              dtype='datetime64[D]').astype('datetime64[ns]')
            iOut = np.hstack([coord[0] + np.searchsorted(df.values, tGrid), coord[1]])
            if coord[0] == 0 and iOut[0] != 0:
                iOut = np.hstack([0, iOut])

        else:
            iOut = 0
        return iOut

    return proc_h5toh5(args)


if __name__ == '__main__':
    # sys.stdout.write('hallo\n')
    sys.stdout.write(str(main()))


def unzip_if_need(lst_of_lsts: Iterable[Union[Iterable[str], str]]) -> Iterator[str]:
    for lsts in lst_of_lsts:
        if isinstance(lsts, str):
            yield lsts
        else:
            yield from lsts


def unzip_if_need_enumerated(lst_of_lsts: Iterable[Union[Iterable[str], str]]) -> Iterator[Tuple[int, str]]:
    for lsts in lst_of_lsts:
        if isinstance(lsts, str):
            yield (0, lsts)
        else:
            yield from enumerate(lsts)


def getstore_and_print_table(fname, strProbe):
    import pprint
    with FakeContextIfOpen(lambda f: pd.HDFStore(f, mode='r'), fname) as store:
        # if isinstance(fname, str):
        #     store = pd.HDFStore(fname, mode='r')
        # elif isinstance(fname, pd.HDFStore):
        #     store = fname
        try:
            pprint.pprint(store.get_storer(strProbe).group.table)
        except AttributeError as e:
            print('Error', standard_error_info(e))
            print('Checking all root members:')
            nodes = store.root.__members__
            for n in nodes:
                print('  ', n)
                try:
                    pprint.pprint(store.get_storer(n))
                except Exception as e:
                    print(n, 'error!', standard_error_info(e))
    # return store


def h5find_tables(store: pd.HDFStore, pattern_tables: str, parent_name=None) -> List[str]:
    """
    Get list of tables in hdf5 store
    :param store: pandas hdf5 store
    :param pattern_tables: str, substring to search paths or regex if with '*'
    :param parent_name: str, substring to search parent paths or regex if with '*'
    :return: list of paths
    """

    if parent_name is None:
        if '/' in pattern_tables:
            parent_name, pattern_tables = pattern_tables.rsplit('/', 1)
            if not parent_name:
                parent_name = '/'
        else:
            parent_name = '/'

    if '*' in parent_name:
        regex_parent = re.compile(parent_name)
        tg = {tblD for tblD in store.root}
        parent_names = {tblD for tblD in store.root.__members__ if regex_parent.match(tblD)}
    else:
        parent_names = [parent_name]  # store.get_storer(parent_name)

        # parent_names = store.root.__members__

    if '*' in pattern_tables:
        regex_parent = re.compile(pattern_tables)
        regex_tables = lambda tblD: regex_parent.match(tblD)
    else:
        regex_tables = lambda tblD: pattern_tables in tblD
    tables = []

    for parent_name in parent_names:
        node = store.get_node(parent_name)
        for tblD in [g for g in node.__members__ if
                     (g != 'table') and (g != '_i_table')]:  # (store.get_storer(n).pathname for n in nodes):
            if regex_tables(tblD):
                tables.append(f'{parent_name}/{tblD}' if parent_name != '/' else tblD)
    l.info('{} tables found'.format(len(tables)))
    tables.sort()
    return tables


def h5sort_pack(h5source_fullpath, h5out_name, table_node, arguments=None, addargs=None,
                b_remove: Optional[bool]=False,
                col_sort: Optional[str]='index'):
    """
    # Compress and save table (with sorting by index) from h5_source_fullpath to h5_cumulative using ``ptprepack`` utility
    
    :param h5source_fullpath: - full file name
    :param h5out_name: base file name + ext of cumulative hdf5 store only
                         (other than that in h5_source_fullpath)
    :param table_node: node name in h5_source_fullpath file
    :param arguments: list, 'fast' or None. None is equal to ['--chunkshape=auto', '--propindexes',
        '--complevel=9', '--complib=zlib',f'--sortby={col_sort}', '--overwrite-nodes']
    :param addargs: list, extend arguments with more parameters
    :param b_remove: file h5_source_fullpath will be deleted after operation!
    :return: full path of cumulative hdf5 store
    Note: ``ptprepack`` not closes hdf5 source if not finds data!
    """

    h5dir, h5source = os_path.split(h5source_fullpath)
    if not table_node:
        return os_path.join(h5dir, h5out_name)
    print(f"sort&pack({table_node}) to {h5out_name}")
    path_prev = os_getcwd()
    argv_prev = sys.argv
    os_chdir(h5dir)  # os.getcwd()
    try:
        if arguments == 'fast':  # bRemove= False, bFast= False,
            arguments = ['--chunkshape=auto', f'--sortby={col_sort}', '--overwrite-nodes']
            if addargs:
                arguments.extend(addargs)
        else:
            if arguments is None:
                arguments = ['--chunkshape=auto', '--propindexes',
                             '--complevel=9', '--complib=zlib', f'--sortby={col_sort}', '--overwrite-nodes']
            if addargs:
                arguments.extend(addargs)
        # arguments + [sourcefile:sourcegroup, destfile:destgroup]
        sys.argv[1:] = arguments + [f'{h5source}:/{table_node}', f'{h5out_name}:/{table_node}']
        # --complib=blosc --checkCSI=True

        ptrepack()


    except Exception as e:
        tbl_cur = 'ptrepack failed!'
        try:
            if f'--sortby={col_sort}' in arguments:
                # check that requirement of fool index is recursively satisfied
                with pd.HDFStore(h5source_fullpath) as store:
                    print('Trying again:\n\t1. Creating index for all childs not having one...')
                    nodes = store.get_node(table_node).__members__
                    for n in nodes:
                        tbl_cur = table_node if n == 'table' else  f'{table_node}/{n}'
                        try:
                            store_tbl = store.get_storer(tbl_cur)
                        except AttributeError:
                            raise HDF5ExtError(f'Table {tbl_cur} error!')
                        if 'index' not in store_tbl.group.table.colindexes:
                            print(tbl_cur, end=' - was no indexes, ')
                            store.create_table_index(tbl_cur, columns=['index'], kind='full')
                            store.flush()
                        else:
                            print(store.get_storer(tbl_cur).group.table.colindexes, end=' - was index. ')
                    # store.get_storer('df').table.reindex_dirty() ?
                print('\n\t2. Restart...')
                ptrepack()
                print('\n\t- Ok')
        except HDF5ExtError:
            raise
        except Exception as ee:  # store.get_storer(tbl_cur).group AttributeError: 'UnImplemented' object has no attribute 'description'
            print('Error:', standard_error_info(ee), f'- creating index on table "{tbl_cur}" is not success.')
            # try without --propindexes yet?
            raise e
        except:
            print('some error')
    finally:
        os_chdir(path_prev)
        sys.argv = argv_prev

    if b_remove:
        try:
            os_remove(h5source_fullpath)
        except:
            print(f'can\'t remove temporary file "{h5source_fullpath}"')
    return os_path.join(h5dir, h5out_name)


def h5sel_index_and_istart(store: pd.HDFStore,
                           tbl_name: str,
                           query_range_lims: Optional[Iterable[Any]] = None,
                           query_range_pattern: str = "index>=Timestamp('{}') & index<=Timestamp('{}')",
                           to_edge: Optional[Any] = None) -> Tuple[pd.Index, int]:
    """
    Get data index by executing query and find queried start index in stored table
    :param store:
    :param tbl_name:
    :param query_range_lims: values to print in query_range_pattern
    :param query_range_pattern:
    :param to_edge:
    :return:
    """
    if query_range_lims is None:  # select all
        df0 = store.select(tbl_name, columns=[])
        i_start = 0
    else:  # select redused range
        if to_edge:
            query_range_lims = list(query_range_lims)
            query_range_lims[0] -= to_edge
            query_range_lims[-1] += to_edge
        qstr = query_range_pattern.format(*query_range_lims)
        l.info(f'query:\n{qstr}... ')
        df0 = store.select(tbl_name, where=qstr, columns=[])
        i_start = store.select_as_coordinates(tbl_name, qstr)[0]
    return df0, i_start


def h5sel_interpolate(i_queried, store, tbl_name, columns=None, time_points=None, method='linear'):
    """

    :param i_queried:
    :param store:
    :param tbl_name:
    :param columns:
    :param time_points:
    :param method: see pandas interpolate but most likely only 'linear' is relevant for 2 closest points
    :return: pandas Dataframe with out_cols columns
    """
    l.info('time interpolating...')
    df = store.select(tbl_name, where=i_queried, columns=columns)
    if not (isinstance(time_points, pd.DatetimeIndex) or isinstance(time_points, pd.Timestamp)):
        t = pd.DatetimeIndex(time_points, tz=df.index.tz)  # to_datetime(t).tz_localize(tzinfo)
    else:
        t = time_points.tz_localize(df.index.tzinfo)
    # try:
    new_index = df.index | t  # pd.Index(timzone_view(time_points, dt_from_utc=df.index.tzinfo._utcoffset))
    # except TypeError as e:  # if Cannot join tz-naive with tz-aware DatetimeIndex
    #     new_index = timzone_view(df.index, dt_from_utc=0) | pd.Index(timzone_view(time_points, dt_from_utc=0))

    df_interp_s = df.reindex(new_index).interpolate(method=method, )  # why not works fill_value=new_index[[0,-1]]?
    df_interp = df_interp_s.loc[t]
    return df_interp


def h5select(store: pd.HDFStore,
             tbl_name: str,
             columns: Optional[Sequence[Union[str, int]]] = None,
             time_points: Optional[Union[np.ndarray, pd.Series, Sequence[int]]] = None,
             dt_check_tolerance=pd.Timedelta(seconds=1),
             query_range_lims: Optional[Union[np.ndarray, pd.Series, Sequence[int]]] = None,
             query_range_pattern="index>=Timestamp('{}') & index<=Timestamp('{}')",
             time_ranges=None,
             interpolate='time') -> Union[pd.DataFrame, Tuple[pd.DataFrame, np.ndarray]]:
    """
    Get hdf5 data with index near the time points or between time ranges, or/and in specified query range
    :param store: pandas hdf5 store
    :param tbl_name: table having sorted index   # tbl_name in store.root.members__
    :param columns: a list of columns that if not None, will limit the return columns
    :param time_points: numpy.array(dtype='datetime64[ns]') to return rows with closest index. If None then uses time_ranges
    :param time_ranges: array of values to return rows with index between. Used if time_points is None
    :param dt_check_tolerance: pd.Timedelta, display warning if found index far from requested values
    :param query_range_lims: initial data range limits to query data, 10 minutes margin will be added
    Note: useful to reduce size of intermediate loading index even if time_points/time_ranges is used
    :param query_range_pattern: format pattern for query with query_range_lims
    :param interpolate: str: "method" arg of pandas.Series.interpolate. If not interpolate, then return closest points
    :return: (df, bbad): df - table of found points, bbad - boolean array returned by other_filters.check_time_diff() or
              df - dataframe of query_range_lims if no time_ranges nor time_points

    Note: query_range_pattern will be used only if query_range_lims specified
    """
    # h5select(store, cfg['in']['table_nav'], ['Lat', 'Lon', 'DepEcho'], dfL.index, query_range_lims=(dfL.index[0], dfL['DateEnd'][-1]), query_range_pattern=cfg['process']['dt_search_nav_tolerance'])
    q_time = time_ranges if time_points is None else time_points
    if (q_time is not None) and len(q_time) and any(q_time):
        try:
            if not np.issubdtype(q_time.dtype, np.datetime64):  # '<M8[ns]'
                q_time = q_time.values
        except TypeError:  # not numpy 'datetime64[ns]'
            q_time = q_time.values
    else:
        df = store.select(
            tbl_name,
            where=query_range_pattern.format(*query_range_lims) if query_range_pattern else None,
            columns=columns)
        return df

    # Get index only and find indexes of data
    df0, i_start = h5sel_index_and_istart(
        store, tbl_name, query_range_lims, query_range_pattern, to_edge=pd.Timedelta(minutes=10))
    i_queried = inearestsorted(df0.index.values, q_time)
    if time_ranges:  # fill indexes inside intervals
        i_queried = np.hstack(np.arange(*se) for se in zip(i_queried[:-1], i_queried[1:] + 1))

    bbad, dt = check_time_diff(t_queried=q_time, t_found=df0.index[i_queried].values, dt_warn=dt_check_tolerance, return_diffs=True)

    if any(bbad) and interpolate:
        i_queried = inearestsorted_around(df0.index.values, q_time)
        i_queried += i_start
        df = h5sel_interpolate(i_queried, store, tbl_name, columns=columns, time_points=q_time, method=interpolate)
    else:
        i_queried += i_start
        df = store.select(tbl_name, where=i_queried, columns=columns)

    return df, dt
    # df_list = []
    # for t_interval in (lambda x=iter(t_intervals): zip(x, x))():
    #     df_list.append(h5select(
    #         store, table, query_range_lims=t_interval,
    #         interpolate=None, query_range_pattern=query_range_pattern
    #         )[0])
    # df = pd.concat(df_list, copy=False)

    # with pd.HDFStore(cfg['in']['db_path'], mode='r') as storeIn:
    #     try:  # Sections
    #         df = storeIn[cfg['in']['table_sections']]  # .sort()
    #     except KeyError as e:
    #         l.error('Sections not found in {}!'.format(cfg['in']['db_path']))
    #         raise e


def h5temp_open(
        db_path: Path,
        db_path_temp: Path,

        tables: Optional[List[str]],
        tables_log: Optional[List[str]],
        db=None,
        b_skip_if_up_to_date: bool = False,
        b_use_old_temporary_tables: bool = False,
        b_overwrite: bool = False,
        **kwargs
        ) -> Tuple[Optional[pd.DataFrame], Optional[pd.HDFStore], bool]:
    """
    Checks and generates some names used for saving data to pandas *.h5 files with log table.
    Opens temporary HDF5 store (db_path_temp), copies previous store (db_path) data to it.

    Temporary HDF5 store needed because of using ptprepack to create index and sort all data at last step
    is faster than support indexing during data appending.

    Parameters are fields that is set when called h5init(cfg_in, cfg_out):
    :param: db_path,
    :param: db_path_temp
    :param: tables, tables_log - if tables is None return (do nothing), else opens HDF5 store and tries to work with ``tables_log``
    :param: b_skip_if_up_to_date:
    :param: b_use_old_temporary_tables, bool, defult False - not copy tables from dest to temp
    :param: b_overwrite: remove all existed data in tables where going to write
    :param: kwargs: not used, for use convenience
    :return: (df_log, db, b_skip_if_up_to_date)
        - df_log: dataframe of log from store if cfg_in['b_skip_if_up_to_date']==True else None.
        - db: pandas HDF5 store - opened db_path_temp
        - b_skip_if_up_to_date:
    Modifies (creates): db - handle of opened pandas HDF5 store
    """
    df_log = None
    if db:
        l.warning('DB is already opened: handle detected!')

    if tables is None:
        return None  # skipping open, may be need if not need write
    else:
        print('saving to', db_path_temp / ','.join(tables), end=':\n')

    try:
        try:  # open temporary output file
            if db_path_temp.is_file():
                db = pd.HDFStore(db_path_temp)
                if not b_use_old_temporary_tables:
                    h5remove_tables(db, tables, tables_log)
        except IOError as e:
            print(e)

        if not b_overwrite:
            if not b_use_old_temporary_tables:
                # Copying previous store data to temporary one
                l.info('Copying previous store data to temporary one:')
                tbl = 'is absent'
                try:
                    with pd.HDFStore(db_path) as storeOut:
                        for tbl in (tables + tables_log):
                            if not tbl:
                                continue
                            try:  # Check output store
                                if tbl in storeOut:  # avoid harmful sortAndPack errors
                                    h5sort_pack(db_path, db_path_temp.name, tbl)
                                else:
                                    raise HDF5ExtError(f'Table {tbl} does not exist')
                            except HDF5ExtError as e:
                                if tbl in storeOut.root.__members__:
                                    print('Node exist but store is not conforms Pandas')
                                    getstore_and_print_table(storeOut, tbl)
                                raise e  # exclude next processing
                            except RuntimeError as e:
                                l.error('Failed copy from output store (RuntimeError). '
                                        'May be need first to add full index to original store? Trying: ')
                                nodes = storeOut.get_node(tbl).__members__  # sorted(, key=number_key)
                                for n in nodes:
                                    tbl_cur = tbl if n == 'table' else f'{tbl}/{n}'
                                    l.info(tbl_cur, end=', ')
                                    storeOut.create_table_index(tbl_cur, columns=['index'], kind='full')
                                # storeOut.flush()
                                l.error('Trying again')
                                if (db is not None) and db.is_open:
                                    db.close()
                                    db = None
                                h5sort_pack(db_path, db_path_temp.name, tbl)

                except HDF5ExtError as e:
                    l.warning(e.args[0])   # print('processing all source data... - no table with previous data')
                    b_skip_if_up_to_date = False
                except Exception as e:
                    print('processing all source data... - no previous data (output table {}): {}'.format(
                        tbl, '\n==> '.join([s for s in e.args if isinstance(s, str)])))
                    b_skip_if_up_to_date = False
                else:
                    if b_skip_if_up_to_date: l.info('Will append data only from new files.')
        if (db is None) or not db.is_open:
            # todo: create group if table that path directs to
            # Open temporary output file to return
            for attempt in range(2):
                try:
                    db = pd.HDFStore(db_path_temp)
                    break
                except IOError as e:
                    print(e)
                except HDF5ExtError as e:  #
                    print('can not use old temporary output file. Deleting it...')
                    os_remove(db_path_temp)
                    # raise(e)

        if b_overwrite:
            df_log = None
            if not b_use_old_temporary_tables:  # Remove existed tables to write
                h5remove_tables(db, tables, tables_log)
        else:
            df_log = [db[tbl_log] for tbl_log in tables_log if tbl_log and (tbl_log in db)]
            df_log = df_log[0] if len(df_log) >= 1 else None

    except HDF5ExtError as e:
        if db:
            db.close()
            db = None
        print('Can not use old temporary output file. Deleting it...')
        for k in range(10):
            try:
                os_remove(db_path_temp)
            except PermissionError:
                print(end='.')
                sleep(1)
            except FileNotFoundError:
                print(end=' - was not exist')
        if os_path.exists(db_path_temp):
            p_name, p_ext = os_path.splitext(db_path_temp)
            db_path_temp = f'{p_name}-{p_ext}'
            print('Can not remove temporary db! => Use another temporary db: "{}"'.format(db_path_temp))
        sleep(1)
        for k in range(10):
            try:
                db = pd.HDFStore(db_path_temp)
            except HDF5ExtError:
                print(end='.')
                sleep(1)
        b_skip_if_up_to_date = False
    except Exception as e:
        l.exception('Can not open temporary hdf5 store')
    return df_log, db, b_skip_if_up_to_date


def df_data_append_fun(df, tbl_name, cfg_out, **kwargs):
    df.to_hdf(cfg_out['db'], tbl_name, append=True, data_columns=cfg_out.get('data_columns', True),
              format='table', dropna=not cfg_out.get('b_insert_separator'), index=False, **kwargs
              )


def df_log_append_fun(df, tbl_name, cfg_out):
    cfg_out['db'].append(tbl_name, df, data_columns=True, expectedrows=cfg_out['nfiles'], index=False,
                         min_itemsize={'values': cfg_out['logfield_fileName_len']})


def h5remove_table(db: pd.HDFStore, node: Optional[str] = None):
    """
    Removes table, skips if not(node) or no such node in currently open db.
    :param db: pandas hdf5 store
    :param node: str, table name
    :modifies db: if reopening here (dew to pandas bug?)
    Note: Raises HDF5ExtError on KeyError if no such node in db.filename and it is have not opened
    """
    try:
        was = node and (node in db)
        if was:
            db.remove(node)
    except KeyError:
        l.info('Trable when removing {}. Solve pandas bug by reopen store.'.format(node))
        sleep(1)
        db.close()
        #db_filename = db.filename
        #db = None
        sleep(1)
        db.open(mode='r+')  # =pd.HDFStore(db_filename, )  # db_path_temp
        try:
            db.remove(node)
            return True
        except KeyError:
            raise HDF5ExtError('Can not remove table "{}"'.format(node))
    return was


def h5remove_tables(db: pd.HDFStore, tables: Iterable[str], tables_log: Iterable[str], db_path_temp=None):
    """
    Removes tables and tables_log from db with retrying if error. Flashes operation
    :param db: pandas hdf5 store
    tables names:
    :param tables,
    :param tables_log:

    :param db_path_temp
    :return:
    """
    name_prev = ''  # used to filter already deleted children (how speed up?)
    for tbl in sorted(tables + tables_log):
        if len(name_prev) < len(tbl) and tbl.startswith(name_prev) and tbl[len(name_prev)] == '/':  # filter
            continue  # parent of this nested have deleted on previous iteration
        for i in range(1, 4):
            try:
                h5remove_table(db, tbl)
                name_prev = tbl
                break
            except ClosedFileError as e:  # file is not open
                l.error('wating %s (/3) because of error: %s', i, str(e))
                sleep(i)
            # except HDF5ExtError as e:
            #     break  # nothing to remove
        else:
            l.error('not successed => Reopening...')
            if db_path_temp:
                db = pd.HDFStore(db_path_temp)
            else:
                db.open(mode='r+')
            h5remove_table(db, tbl)
    db.flush()
    return db

# ----------------------------------------------------------------------
class ReplaceTableKeepingChilds:
    """
        Saves childs (before You delete tbl_parent)
        #for h5find_tables(store, '', parent_name=tbl_parent)

        cfg_out must have field: 'db' - handle of opened store
    """

    def __init__(self, dfs, tbl_parent, cfg_out, write_fun=None):
        self.cfg_out = cfg_out
        self.tbl_parent = tbl_parent
        self.dfs = [dfs] if isinstance(dfs, pd.DataFrame) else dfs
        self.write_fun = write_fun

    def __enter__(self):

        self.childs = []
        try:
            parent_group = self.cfg_out['db'].get_storer(self.tbl_parent).group
            nodes = parent_group.__members__
            self.childs = [f'/{self.tbl_parent}/{g}' for g in nodes if (g != 'table') and (g != '_i_table')]
            if self.childs:
                l.info('found {} childs of {}. Copying...'.format(len(nodes), self.tbl_parent))
                # [parent_group[n].pathname for n in nodes]
                # [parent_group[n] for n in nodes if n!='table']

                for tblD in self.childs:
                    self.cfg_out['db']._handle.move_node(tblD, newparent='/to_copy_back', createparents=True,
                                                         overwrite=True)
                self.cfg_out['db'].flush()  # .flush(fsync=True

        except AttributeError:
            pass  # print(tbl_parent + ' has no childs')
        # Make index to be UTC

        # remove parent table that must be writed back in "with" block
        try:
            h5remove_table(self.cfg_out['db'], self.tbl_parent)
        except KeyError:
            print('was removed?')

        return self.childs

    def __exit__(self, exc_type, ex_value, ex_traceback):

        if self.write_fun is None:
            def write_fun(df, tbl, cfg):
                return df.to_hdf(
                    cfg['db'], tbl, format='table', data_columns=True, append=False, index=False)

            self.write_fun = write_fun

        # write parent table
        for df in self.dfs:
            self.write_fun(df, self.tbl_parent, self.cfg_out)
        self.cfg_out['db'].create_table_index(self.tbl_parent, columns=['index'], kind='full')

        # write childs back
        self.cfg_out['db'].flush()
        if exc_type is None:
            for tblD in self.childs:
                self.cfg_out['db']._handle.move_node(tblD.replace(self.tbl_parent, 'to_copy_back'),
                                                     newparent=f'/{self.tbl_parent}', createparents=True,
                                                     overwrite=True)
        # cfg_out['db'].move('/'.join(tblD.replace(tbl_parent,'to_copy_back'), tblD))
        # cfg_out['db'][tblD] = df # need to_hdf(format=table)
        return False


# ----------------------------------------------------------------------
def h5remove_duplicates(cfg, cfg_table_keys: Iterable[Union[Iterable[str], str]]) -> Set[str]:
    """
    Remove duplicates inplace
    :param cfg: dict with keys specified in cfg_table_keys
    :param cfg_table_keys: list, in which 'tables_log' means that cfg['tables_log'] is a log table. Alternatively group tables in subsequences such that log tables names is after data table in each subsequence (cfg[cfg_table_keys[group]])
    :return tbl_dups: tables that still have duplicates
    """

    # load data frames from store to memory removing duplicates
    dfs = {}
    tbl_dups = set()  # will remove tables if will found duplicates
    for cfgListName in cfg_table_keys:
        for tbl in unzip_if_need(cfg[cfgListName]):
            if tbl in cfg['db']:
                ind_series = cfg['db'].select_column(tbl, 'index')
                # dfs[tbl].index.is_monotonic_increasing? .is_unique()?
                b_dup = ind_series.duplicated(keep='last')
                if b_dup.any():
                    i_dup = b_dup[b_dup].index
                    l.info('deleting {} duplicates in {} (first at {}){}'.format(
                        len(i_dup), tbl, ind_series[i_dup[0]],
                        '' if i_dup.size < 50 or Path(cfg['db'].filename).stem.endswith('not_sorted') else
                        '. Note: store size will not shrinked!'))  # if it is in temp db to copy from then it is ok
                    try:
                        cfg['db'].remove(tbl, where=i_dup)  # may be very long.
                        # todo: if many to delete try h5remove_duplicates_by_loading()
                    except:
                        l.exception('can not delete duplicates')
                        tbl_dups.add(tbl)
    return tbl_dups


def h5remove_duplicates_by_loading(cfg, cfg_table_keys: Iterable[Union[Iterable[str], str]]) -> Set[str]:
    """
    Remove duplicates by coping tables to memory
    :param cfg: dict with keys:
        keys specified by cfg_table_keys
        chunksize - for data table
        logfield_fileName_len, nfiles - for log table
    :param cfg_table_keys: list, in which 'tables_log' means that cfg['tables_log'] is a log table. Alternatively group tables in subsequences such that log tables names is after data table in each subsequence (cfg[cfg_table_keys[group]])
    :return tbl_dups: tables that still have duplicates
    """

    # load data frames from store to memory removing duplicates
    dfs = {}
    tbl_dups = set()  # will remove tables if will found duplicates
    for cfgListName in cfg_table_keys:
        for tbl in unzip_if_need(cfg[cfgListName]):
            if tbl in cfg['db']:
                dfs[tbl] = cfg['db'][tbl]
                # dfs[tbl].index.is_monotonic_increasing? .is_unique()?
                b_dup = dfs[tbl].index.duplicated(keep='last')
                if np.any(b_dup):
                    tbl_dups.add(tbl)
                    l.info('{} duplicates in {} (first at {})'.format(
                        sum(b_dup), tbl, dfs[tbl].index[np.flatnonzero(b_dup)[0]]))
                    dfs[tbl] = dfs[tbl][~b_dup]

    # update data frames in store
    if len(tbl_dups):
        l.info('Remove duplicates. ')
        for cfgListName in cfg_table_keys:
            for i_in_group, tbl in unzip_if_need_enumerated(cfg[cfgListName]):
                if tbl in tbl_dups:
                    try:
                        with ReplaceTableKeepingChilds([dfs[tbl]], tbl, cfg, df_log_append_fun \
                            if (cfgListName == 'tables_log' or i_in_group > 0) else df_data_append_fun):
                            pass
                            # cfg['db'].append(tbl, dfs[tbl], data_columns=True, index=False, **(
                            #     {'expectedrows': cfg['nfiles'],
                            #      'min_itemsize': {'values': cfg['logfield_fileName_len']}
                            #      } if (cfgListName == 'tables_log' or i_in_group > 0) else
                            #     {'chunksize': cfg['chunksize']
                            #      }
                            #     ))

                            tbl_dups.discard(tbl)
                    except Exception as e:
                        l.exception('Table %s not recorded because of error when removing duplicates', tbl)
                        # cfg['db'][tbl].drop_duplicates(keep='last', inplace=True) #returns None
    else:
        l.info('Not need remove duplicates. ')
    return tbl_dups


def create_indexes(cfg_out, cfg_table_keys):
    """
    Create full indexes. That is mandatory before using ptprepack in h5move_tables() below
    :param cfg_out: must hav fields
    - 'db': handle of opened HDF5Store
    - fields specified in :param cfg_table_keys where values are table names that need index. Special field name:
        - 'tables_log': means that cfg_out['tables_log'] is a log table
    - 'index_level2_cols': second level for Multiindex (only 2 level supported, 1st is always named 'index')
    :param cfg_table_keys: list of cfg_out field names having tables names that need index. Instead of using
    'tables_log' for log tables the list can contain subsequences where log tables names fields will be after data table in each subsequence
    :return:
    """
    l.debug('Creating index')
    for cfgListName in cfg_table_keys:
        for i_in_group, tbl in unzip_if_need_enumerated(cfg_out[cfgListName]):
            if not tbl:
                continue
            try:
                if i_in_group == 0 or cfgListName != 'tables_log':  # not nested (i.e. log) table
                    navp_all_index, level2_index = multiindex_timeindex(cfg_out['db'][tbl].index)
                else:
                    level2_index = None
                columns = ['index'] if level2_index is None else ['index', cfg_out['index_level2_cols']]
                cfg_out['db'].create_table_index(tbl, columns=columns, kind='full')
            # except KeyError:
            #     pass  # 'No object named ... in the file'
            except Exception as e:
                l.warning('Index in table "{}" not created - error: {}'.format(tbl, standard_error_info(e)))
            # except TypeError:
            #     print('can not create index for table "{}"'.format(tbl))


def h5move_tables(cfg_out, tbl_names: Union[Sequence[str], Sequence[Sequence[str]], None]=None, **kwargs
                  ) -> Dict[str, str]:
    """
    Copy tables tbl_names from one store to another using ptrepack. If fail to store
    in specified location then creates new store and tries to save there.
    :param cfg_out: dict - must have fields:
      - db_path_temp: source of not sorted tables
      - db_path: full path name (extension ".h5" will be added if absent) of hdf store to put
      - tables, tables_log: Sequence[str], if tbl_names not specified
    :param tbl_names: list of strings or list of lists (or tuples) of strings. List of lists is useful to keep order of operation: put nested tables last.
    Note ``ptprepack`` not closes hdf5 source if it not finds data!

        Strings are names of hdf5 tables to copy
    :return: None if all success else if have errors - dict of locations of last tried savings for each table
    """
    if tbl_names is None:  # copy all cfg_out tables
        tbl_names = cfg_out['tables'] + cfg_out['tables_log']

    tables = list(unzip_if_need(tbl_names))
    if tables:
        tbl = tables[0]
        l.info('moving tables %s to %s:', ', '.join(tables), cfg_out['db_path'].name)
    else:
        raise Ex_nothing_done('no tables to move')

    # h5sort_pack can not remove/update dest table so we do:
    try:
        with pd.HDFStore(cfg_out['db_path']) as store:
            h5remove_table(store, tbl)
    except HDF5ExtError:
        file_bad = Path(cfg_out['db_path'])
        file_bad_keeping = file_bad.with_suffix('.bad.h5')
        l.exception('Bad output file - can not use!!! Renaming to "%s". Delete it if not useful', file_bad_keeping)
        file_bad.rename(file_bad_keeping)
        l.warning('Renamed: old data (if any) will not be in %s!!! Writing current data...', file_bad)

    with pd.HDFStore(cfg_out['db_path_temp']) as store_in:  #pd.HDFStore(cfg_out['db_path']) as store,
        for tbl in tables:
            if 'index' not in store_in.get_storer(tbl).group.table.colindexes:
                print(tbl, end=' - was no indexes, creating.')
                store_in.create_table_index(tbl, columns=['index'], kind='full')
            elif cfg_out.get('recreate_index') and tbl in cfg_out['recreate_index']:
                print(tbl, end=' - was indexes, but recreating by loading, saving with no index then add infdex')

                df = store_in[tbl]  #.sort_index()
                cfg = {'db': store_in}
                with ReplaceTableKeepingChilds([df], tbl, cfg):
                    pass

                    # cfg['db'].append(
                    #     tbl, df, data_columns=True, index=False, **(
                    #         {'expectedrows': cfg['nfiles'],
                    #          'min_itemsize': {'values': cfg['logfield_fileName_len']}} if (
                    #                 cfgListName == 'tables_log' or i_in_group > 0) else
                    #         {'chunksize': cfg['chunksize']})
                    #     )
    i_when_remove_store = len(tables) if cfg_out.get('b_del_temp_db') else -1  # remove store when last table has copied
    for i, tbl in enumerate(tables, start=1):
        try:
            h5sort_pack(cfg_out['db_path_temp'],
                        cfg_out['db_path'].with_suffix('.h5').name,
                        tbl,
                        addargs=cfg_out.get('addargs'),
                        b_remove=(i==i_when_remove_store),
                        **kwargs)
            sleep(2)
        except Exception as e:
            l.error('Error: "{}"\nwhen write table "{}" from {} to {}'.format(
                e, tbl, cfg_out['db_path_temp'], cfg_out['db_path']))
    return
    # storage_basenames = {}
    #         if False:  # not helps?
    #             storage_basename = os_path.splitext(cfg_out['db_base'])[0] + "-" + tbl.replace('/', '-') + '.h5'
    #             l.info('so start write to {}'.format(storage_basename))
    #             try:
    #                 h5sort_pack(cfg_out['db_path_temp'], storage_basename, tbl, addargs=cfg_out.get('addargs'), **kwargs)
    #                 sleep(4)
    #             except Exception as e:
    #                 storage_basename = cfg_out['db_base'] + '-other_place.h5'
    #                 l.error('Error: "{}"\nwhen write {} to original place! So start write to {}'.format(e, tbl,
    #                                                                                                    storage_basename))
    #                 try:
    #                     h5sort_pack(cfg_out['db_path_temp'], storage_basename, tbl, addargs=cfg_out.get('addargs'), **kwargs)
    #                     sleep(8)
    #                 except:
    #                     l.error(tbl + ': no success')
    #             storage_basenames[tbl] = storage_basename
    # if storage_basenames == {}:
    #     storage_basenames = None


def h5index_sort(cfg_out,
                 out_storage_name=None,
                 in_storages: Optional[Mapping[str, str]]=None,
                 tables: Optional[Iterable[Union[str, Tuple[str]]]] = None) -> None:
    """
    Checks if tables in store have sorted index and if not then sort it by loading, sorting and saving data
    Not tries to sort nonmonotonic data
    :param cfg_out: dict - must have fields:
        'db_path': store where tables will be checked
        'db_path_temp': source of not sorted tables for h5move_tables() if index is not monotonic
        'base': base name (extension ".h5" will be added if absent) of hdf store to put
        'tables' and 'tables_log': tables to check monotonousness and if they are sorted, used if :param tables: not specified only
        'dt_from_utc'
        'b_remove_duplicates': if True then deletes duplicates by loading data in memory
    :param out_storage_name:
    :param in_storages: to use its values istead cfg_out['db_path']
    :param tables: iterable of table names
    :return:
    """
    l.info('Checking that indexes are sorted:')
    if out_storage_name is None:
        out_storage_name = cfg_out['storage']
    set_field_if_no(cfg_out, 'dt_from_utc', 0)

    if in_storages is None:
        in_storages = cfg_out['db_path']
    else:
        in_storages = [v for v in in_storages.values()]

        if len(in_storages) > 1:
            l.warning('Not implemented for result stored in multiple locations. Check only first')

        in_storages = os_path.join(os_path.dirname(cfg_out['db_path']), in_storages[0])

    if tables is None:
        tables = cfg_out['tables'] + cfg_out['tables_log']
    with pd.HDFStore(in_storages) as store:
        # store= pd.HDFStore(cfg_out['db_path'])
        # b_need_save = False
        tbl_dups = set()
        tbl_nonm = set()
        for tbl in unzip_if_need(tables):
            if tbl not in store:
                l.warning('{} not in {}'.format(tbl, in_storages))
                continue
            try:
                df = store[tbl]
                if df is None:
                    l.warning('None table {} in {}'.format(tbl, store.filename))
                    continue
            except TypeError as e:
                l.exception('Can not access table %s', tbl)
                continue
            # store.close()
            if df.index.is_monotonic:
                if df.index.is_unique:
                    l.info(f'{tbl} - sorted')
                else:
                    l.warning(f'{tbl} - sorted, but have duplicates')

                    # experimental
                    if cfg_out['b_remove_duplicates']:
                        l.warning(f'{tbl} - removing duplicates - experimental!')
                        tbl_dups.update(
                            h5remove_duplicates(
                                {**cfg_out, 'db': store,
                                'tables': [t for t in cfg_out['tables'] if t],
                                'tables_log': [t for t in cfg_out['tables_log'] if t],
                                },
                                cfg_table_keys=['tables', 'tables_log']
                                )
                            )
                    else:
                        tbl_dups.add(tbl)
                continue
            else: # only printing messages about what the problem with sorting by trying it
                tbl_nonm.add(tbl) # b_need_save = True

                l.warning(f'{tbl} - not sorted!')
                print(repr(store.get_storer(tbl).group.table))

                df_index, itm = multiindex_timeindex(df.index)
                if __debug__:
                    plt.figure(f'Not sorted index that we are sorting {"on" if itm is None else "before"} saving...')
                    plt.plot(df_index)  # np.diff(df.index)
                    plt.show()

                if not itm is None:
                    l.warning('sorting multiindex...')
                    df = df.sort_index()  # inplace=True
                    if df.index.is_monotonic:
                        if df.index.is_unique:
                            l.warning('Ok')
                        else:
                            tbl_dups.add(tbl)
                            l.warning('Ok, but have duplicates')
                        continue
                    else:
                        print('Failure!')
                # else:
                #     pass  # will sorting by prepack in h5move_tables #l.warning('skipped of sorting ')
        if tbl_dups:
            l.warning(f'To drop duplicates from {tbl_dups} restart with [out][b_remove_duplicates] = True')
            tbl_nonm -= tbl_dups
        else:
            l.info('no duplicates...')
        if tbl_nonm:
            l.warning(f'{tbl_nonm} have no duplicates but nonmonotonic. Forcing update index before move and sort...')
            if tbl_nonm:
                # as this fun is intended to check h5move_tables stranges, repeat it with forcing update index
                h5move_tables({**cfg_out, 'recreate_index': tbl_nonm}, tbl_names=tables)
        else:
            l.info(f'{"other" if tbl_dups else "all"} tables monotonic.{"" if tbl_dups else " Ok>"}')

        # if b_need_save:
        #     # out to store
        #     cfg_out['db_path'], cfg_out['db_path_temp'] = cfg_out['db_path_temp'], cfg_out['db_path_temp']
        #     h5move_tables(cfg_out, tbl_names=tables)
        #     cfg_out['db_path'], cfg_out['db_path_temp'] = cfg_out['db_path_temp'], cfg_out['db_path_temp']
        #     h5move_tables(cfg_out, tbl_names=tables)

            # store = pd.HDFStore(cfg_out['db_path_temp'])
            # store.create_table_index(tbl, columns=['index'], kind='full')
            # store.create_table_index(cfg_out['tables_log'][0], columns=['index'], kind='full') #tbl+r'/logFiles'
            # h5_append(store, df, log, cfg_out, cfg_out['dt_from_utc'])
            # store.close()
            # h5sort_pack(cfg_out['db_path_temp'], out_storage_name, tbl) #, ['--overwrite-nodes=true']


def h5del_obsolete(cfg_out: MutableMapping[str, Any],
                   log: Mapping[str, Any],
                   df_log: pd.DataFrame) -> Tuple[bool, bool]:
    """
    Check that current file has been processed and it is up to date
    Removes all data from the store table and log table where time >= time of new data!
    New data determined by comparing df_log records to current file (log) values

    Also removes duplicates in the table if found duplicate records in the log
    :param cfg_out: dict, must have field
        'db' - handle of opened store
        'b_use_old_temporary_tables' - for message
        'tables_log', 'tables' - to able check and deleting
    :param log: dict, with info about current data, must have fields for compare:
        'fileName' - in format as in log table to able find duplicates
        'fileChangeTime', datetime - to able find outdate data
    :param df_log: dataframe, log table loaded from store before updating
    :return: (have_older_data, have_duplicates):
        - have_duplicates: Duplicate entries in df_log detected
        - have_older_data: Have recorded data that was changed after current file was changed (log['fileChangeTime'])
    """
    if cfg_out['tables'] is None or df_log is None:
        return (False, False)

    df_log_cur = df_log[df_log['fileName'] == log['fileName']]
    n_log_rows_for_file = len(df_log_cur)
    have_duplicates = False  # not detected yet
    have_older_data = False  # not detected yet
    if n_log_rows_for_file:
        if n_log_rows_for_file > 1:
            have_duplicates = True
            print('Duplicate entries in log => will be removed from tables! (detected "{}")'.format(log['fileName']))
            cfg_out['b_remove_duplicates'] = True
            if cfg_out['b_use_old_temporary_tables']:
                print('Consider set [out].b_use_old_temporary_tables=0,[in].b_skip_if_up_to_date=0')
            print('Continuing...')
            imax = df_log_cur['fileChangeTime'].argmax()  # np.argmax([r.to_pydatetime() for r in ])
        else:
            imax = 0
        last_fileChangeTime = df_log_cur['fileChangeTime'].iat[imax].to_pydatetime()
        if last_fileChangeTime >= log['fileChangeTime']:
            have_older_data = True
            print('>', end='')
            df_log_cur = df_log_cur[np.arange(len(df_log_cur)) != imax]  # last row record
        if not df_log_cur.empty:  # delete other records
            print('removing obsolete stored data rows:', end=' ')
            qstr = "index>=Timestamp('{}')".format(df_log_cur.index[0])
            qstrL = "fileName=='{}'".format(df_log_cur['fileName'][0])
            try:
                tbl = ''; tbl_log = ''
                for tbl, tbl_log in zip(cfg_out['tables'], cfg_out['tables_log']):
                    if tbl:
                        n_rows = cfg_out['db'].remove(tbl, where=qstr)
                        print(f'{n_rows} in table', end=', ')
                    if tbl_log:
                        n_rows = cfg_out['db'].remove(tbl_log, where=qstrL)
                        print(f'{n_rows} in log.')
            except NotImplementedError as e:
                l.exception('Can not delete obsolete rows, so removing full tables %s & %s and filling with all currently found data', tbl, tbl_log)
                if tbl_log: cfg_out['db'].remove(tbl_log)  # useful if it  is not a child
                if tbl: cfg_out['db'].remove(tbl)
                have_older_data = False
                have_duplicates = False
    return (have_older_data, have_duplicates)


def h5init(cfg_in: Mapping[str, Any], cfg_out: MutableMapping[str, Any]) -> None:
    """
    Init cfg_out database (hdf5 data store) information in cfg_out _if it is not exist_
    :param: cfg_in - configuration dicts, with fields:
    - path: if no 'db_path' in cfg_out
    - cfgFile - if no cfg_out['b_insert_separator'] defined or determine the table name is failed - to extract from cfgFile name name
    - raw_dir_words: (optional), default: ['source', 'WorkData', 'workData'] - see getDirBaseOut()
    - nfiles: (optional)
    - b_skip_if_up_to_date: (optional)
    :param: cfg_out - configuration dict, where all fields are optional. Do nothing if cfg_out['tables'] is None

    Sets or updates fields of cfg_out:
    % paths %
    - db_path: absolute path of hdf5 store with suffix ".h5"
    - tables, tables_log: tables names of data and log (metadata) - based on cfg_in and cfg_in['raw_dir_words']
    - db_path_temp: temporary h5 file name
    % other %
    - nfiles: default 1, copied from cfg_in - to set store.append() 'expectedrows' argument
    - b_skip_if_up_to_date: default False, copied from cfg_in
    - chunksize: default None
    - logfield_fileName_len: default 255
    - b_remove_duplicates: default False
    - b_use_old_temporary_tables: default False

    :return: None
    """
    if 'tables' in cfg_out and cfg_out['tables'] is None:
        return
    set_field_if_no(cfg_out, 'logfield_fileName_len', 255)
    set_field_if_no(cfg_out, 'chunksize')
    set_field_if_no(cfg_out, 'b_skip_if_up_to_date', cfg_in['b_skip_if_up_to_date' \
        ] if 'b_skip_if_up_to_date' in cfg_in else False)
    set_field_if_no(cfg_out, 'b_remove_duplicates', False)
    set_field_if_no(cfg_out, 'b_use_old_temporary_tables', True)
    if cfg_out.get('b_insert_separator') is None:
        cfg_file = PurePath(cfg_in['cfgFile']).stem
        cfg_out['b_insert_separator'] = '_ctd_' in cfg_file.lower()

    # Automatic db file and tables names
    if not (cfg_out.get('db_path') and cfg_out['db_path'].is_absolute()):
        path_in = Path(cfg_in.get('path' if 'path' in cfg_in else 'db_path')).parent
        cfg_out['db_path'] = path_in / (f'{path_in.stem}_out' if not cfg_out.get('db_path') else cfg_out['db_path'])
    dir_create_if_need(cfg_out['db_path'].parent)
    cfg_out['db_path'] = cfg_out['db_path'].with_suffix('.h5')

    # Will save to temporary file initially
    set_field_if_no(cfg_out, 'db_path_temp', cfg_out['db_path'].with_name(f"{cfg_out['db_path'].stem}_not_sorted.h5"))
    set_field_if_no(cfg_out, 'nfiles', cfg_in.get('nfiles', 1))

    if 'tables' in cfg_out and cfg_out['tables']:
        set_field_if_no(cfg_out, 'tables_log', [((f'{tab}/logFiles') if tab else '') for tab in cfg_out['tables']])
    elif 'table' in cfg_out and cfg_out['table']:
        cfg_out['tables'] = [cfg_out['table']]
        set_field_if_no(cfg_out, 'tables_log', [f"{cfg_out['table']}/logFiles"])
    else:
        table_auto = cfg_in.get('table')
        if not table_auto:
            _, _, table_auto = getDirBaseOut(
                cfg_out['db_path'],
                cfg_in.get('raw_dir_words') or ['raw', '_raw', 'source', '_source', 'WorkData', 'workData']
                )
        if not table_auto:
            table_auto = Path(cfg_in['cfgFile']).stem
            l.warning('Can not dertermine table_name from file structure. '
                      'Set [tables] in ini! Now use table_name "%s"', table_auto)
        cfg_out['tables'] = [table_auto]
        set_field_if_no(cfg_out, 'tables_log', [f'{table_auto}/logFiles'])


# Functions to iterate rows of db log instead of files in dir

def query_time_range(min_time=None, max_time=None, **kwargs) -> str:
    """
    Query Time for pandas.Dataframe
    :param min_time:
    :param max_time:
    :return:
    """
    if min_time:
        query_range = (f"index>=Timestamp('{min_time}') & index<=Timestamp('{max_time}')" if max_time else
                       f"index>=Timestamp('{min_time}')")
    elif max_time:
        query_range = f"index<=Timestamp('{max_time}')"
    else:
        query_range = None
    return query_range


def h5log_rows_gen(
        db_path,
        table_log: str,
        min_time=None, max_time=None,
        query_range=None,
        db: Optional[pd.HDFStore]=None, **kwargs) -> Iterator[Dict[str, Any]]:
    """
    Dicts from each h5 log row
    :param db_path, str: name of hdf5 pandas store where is log table, used only for message if it is set and db is set
    :param db: handle of already open pandas hdf5 store
    :param table_log, str: name of log table - table with columns for intervals:
      - index - starts, pd.DatetimeIndex
      - DateEnd - ends, pd.Datetime
    :param min_time, max_time: datetime, optional, allows limit the range of table_log rows, not used if query_range is set
    :param query_range: query str to limit the range of table_log rows to load
        Example table_log name: cfg_in['table_log'] ='/CTD_SST_48M/logRuns'
    :param kwargs: not used
    Yeilds dicts where keys: col names, values: current row values of tbl_intervals = cfg_in['table_log']
    """
    if query_range is None:
        query_range = query_time_range(min_time, max_time)
    with FakeContextIfOpen(lambda f: pd.HDFStore(f, mode='r'), file=db_path, opened_file_object=db) as store:
        # with pd.HDFStore(db_path, mode='r') as store:
        if db_path:
            print("loading from {db_path}: ".format(db_path), end='')
        for n, rp in enumerate(store.select(table_log, where=query_range).itertuples()):
            r = dict(zip(rp._fields, rp))
            yield (r)  # r.Index, r.DateEnd


def h5log_names_gen(cfg_in, f_row_to_name=lambda r: '{Index:%y%m%d_%H%M}-{DateEnd:%H%M}'.format_map(r)):
    """
    Genereates outputs of f_row_to_name function which receves dicts from each h5 log row (see h5log_rows_gen)
    :param cfg_in: dict with field 'log_row' where h5log_rows_gen(cfg_in) item will be saved
    :param f_row_to_name: function(dict) where dict have fields from h5 log row.
        By default returns string suitable to name files by start-end date/time
    :return:
    :modifies cfg: adds/replaces field 'log_row': dict from h5 log row. This allows use this dict also
    Replasing for veuszPropagate.ge_names() to use tables instead files
    """
    for row in h5log_rows_gen(**cfg_in):
        cfg_in['log_row'] = row
        yield f_row_to_name(row)
