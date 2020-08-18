#!/usr/bin/env python
# coding:utf-8
"""
  Author:  Andrey Korzh --<ao.korzh@gmail.com>

"""
from __future__ import print_function, division

import logging
import re
import sys  # from sys import argv
import warnings
from os import path as os_path, getcwd as os_getcwd, chdir as os_chdir, remove as os_remove
from time import sleep
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union, List

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
                                Term[-1] += (' ' + s[:-1])
                            else:
                                Term[-1] += (' ' + s)
                        elif s[0] == '"':
                            bWate = True;
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


def unzip_if_need(lst_of_lsts):
    for lsts in lst_of_lsts:
        if isinstance(lsts, str):
            yield lsts
        else:
            yield from lsts


def unzip_if_need_enumerated(lst_of_lsts):
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


def h5sort_pack(h5_source_fullpath, h5_cumulative, table_node, arguments=None, addargs=None, b_remove=False):
    """
    # Compress and save table (with sorting by index) from h5_source_fullpath to h5_cumulative
    
    :param h5_source_fullpath: - full file name
    :param h5_cumulative: base file name + ext of cumulative hdf5 store only
                         (other than that in h5_source_fullpath)
    :param table_node: node name in h5_source_fullpath file
    :param arguments: list, 'fast' or None. None is equal to ['--chunkshape=auto', '--propindexes',
        '--complevel=9', '--complib=zlib','--sortby=index', '--overwrite-nodes']
    :param addargs: list, extend arguments with more parameters
    :param b_remove: file h5_source_fullpath will be deleted after operation!
    :return: full path of cumulative hdf5 store 
    """

    h5Dir, h5Source = os_path.split(h5_source_fullpath)
    if not table_node:
        return os_path.join(h5Dir, h5_cumulative)
    print("sort&pack({}) to {}".format(table_node, h5_cumulative))
    path_prev = os_getcwd()
    argv_prev = sys.argv
    os_chdir(h5Dir)  # os.getcwd()
    try:
        if arguments == 'fast':  # bRemove= False, bFast= False,
            arguments = ['--chunkshape=auto', '--sortby=index', '--overwrite-nodes']
            if addargs:
                arguments.extend(addargs)
        else:
            if arguments is None:
                arguments = ['--chunkshape=auto', '--propindexes',
                             '--complevel=9', '--complib=zlib', '--sortby=index', '--overwrite-nodes']
            if addargs:
                arguments.extend(addargs)
        # arguments + [sourcefile:sourcegroup, destfile:destgroup]
        sys.argv[1:] = arguments + [h5Source + ':/' + table_node, h5_cumulative + ':/' + table_node]
        # --complib=blosc --checkCSI=True

        ptrepack()
    except Exception as e:
        try:
            if '--sortby=index' in arguments:
                # check requriment of fool index is recursive sutisfied
                with pd.HDFStore(h5_source_fullpath) as store:
                    print('trying again with creating index ...')
                    nodes = store.get_node(table_node).__members__
                    for n in nodes:
                        tbl_cur = table_node if n == 'table' else table_node + '/' + n
                        if 'index' not in store.get_storer(tbl_cur).group.table.colindexes:
                            print(tbl_cur, end=' - was no indexes, ')
                            store.create_table_index(tbl_cur, columns=['index'], kind='full')
                        else:
                            print(store.get_storer(tbl_cur).group.table.colindexes, end=' - was index')
                    # store.get_storer('df').table.reindex_dirty() ?
                print('restart: ')
                ptrepack()
        except Exception as ee:
            print('Error ', standard_error_info(ee))
            print('- creating index not success.')
            # try without --propindexes yet?
            raise e
    finally:
        os_chdir(path_prev)
        sys.argv = argv_prev

    if b_remove:
        try:
            os_remove(h5_source_fullpath)
        except:
            print('can\'t remove temporary file "' + h5_source_fullpath + '"')
    return os_path.join(h5Dir, h5_cumulative)


def h5sel_index_and_istart(store: pd.HDFStore,
                           tbl_name: str,
                           query_range_lims: Optional[Iterable[Any]] = None,
                           query_range_pattern: str = "index>=Timestamp('{}') & index<=Timestamp('{}')",
                           to_edge: Optional[Any] = None) -> Tuple[pd.Index, int]:
    # Get data index extcuting query and find queried start index in stored table
    if query_range_lims is None:  # select all
        df0 = store.select(tbl_name, columns=[])
        i_start = 0
    else:  # select redused range
        if to_edge:
            query_range_lims = list(query_range_lims)
            query_range_lims[0] -= to_edge
            query_range_lims[-1] += to_edge
        qstr = query_range_pattern.format(*query_range_lims)
        l.info('query:\n' + qstr + '... ')
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
        t = time_points = pd.DatetimeIndex(time_points, tz=df.index.tz)  # to_datetime(t).tz_localize(tzinfo)
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
             interpolate='linear') -> Union[pd.DataFrame, Tuple[pd.DataFrame, np.ndarray]]:
    """
    Get hdf5 data with index near time points or between time ranges, optionly in specified query range
    for less memory loading
    :param store: pandas hdf5 store
    :param tbl_name: table having sorted index   # tbl_name in store.root.members__
    :param columns: a list of columns that if not None, will limit the return columns
    :param time_points: array of values numpy.dtype('datetime64[ns]') to return rows with closest index. If None then uses time_ranges
    :param time_ranges: array of values to return rows with index between. Used if time_points is None
    :param dt_check_tolerance: pd.Timedelta, display warning if found index far from requested values
    :param query_range_lims: data range if no time_ranges nor time_points, for time_points can be used to reduce size of intermediate loading index
    :param query_range_pattern: optional format pattern for query_range_lims
    :param interpolate: str: see pandas interpolate. If not interpolate, then return closest points
    :return: (df, bbad): df - table of found points, bbad - boolean array returned by other_filters.check_time_diff() or
              df - dataframe of query_range_lims if no time_ranges nor time_points
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
        df = store.select(tbl_name, where=query_range_pattern.format(*query_range_lims) if query_range_pattern else None, columns=columns)
        return df

    # Get index only and find indexes of data
    df0, i_start = h5sel_index_and_istart(
        store, tbl_name, query_range_lims, query_range_pattern, to_edge=pd.Timedelta(minutes=10))
    i_queried = inearestsorted(df0.index.values, q_time)
    if time_ranges:  # fill indexes inside itervals
        i_queried = np.hstack(np.arange(*se) for se in zip(i_queried[:-1], i_queried[1:] + 1))

    bbad, dt = check_time_diff(t_queried=q_time, t_found=df0.index[i_queried].values, dt_warn=dt_check_tolerance, return_diffs=True)

    if any(bbad) and interpolate:
        i_queried = inearestsorted_around(df0.index.values, q_time)
        i_queried += i_start
        df = h5sel_interpolate(i_queried, store, tbl_name, columns=columns, time_points=q_time, method='linear')
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
                tables.append(parent_name + '/' + tblD if parent_name != '/' else tblD)
    l.info('{} tables found'.format(len(tables)))
    tables.sort()
    return tables


def h5temp_open(cfg_out: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """
    Checks and generates some names used for saving to my *.h5 files. Opens HDF5 store,
    copies previous store data to this if 'b_skip_if_up_to_date'*

    :param: cfg_out, dict, requires fields (that is set when called h5init(cfg_in, cfg_out)):
          db_path, db_path_temp
          tables, tables_log - if tables is None return (do nothing), else opens HDF5 store and tries to work with ``tables_log``
          b_skip_if_up_to_date
          b_use_old_temporary_tables, bool, defult False - not copy tables from dest to temp
    :return: df_log, dataframe of log from store if cfg_in['b_skip_if_up_to_date']==True else None.
    Modifies (creates): cfg_out['db'] - handle of opened pandas HDF5 store
    """

    if cfg_out.get('db'):
        l.warning('DB is already opened: handle detected!')
    else:
        cfg_out['db'] = None

    if cfg_out['tables'] is None:
        return None  # skipping open, may be need if not need write
    else:
        print('saving to', '/'.join([cfg_out['db_path_temp'], ','.join(cfg_out['tables'])]) + ':')

    try:
        try:  # open temporary output file
            if os_path.isfile(cfg_out['db_path_temp']):
                cfg_out['db'] = pd.HDFStore(cfg_out['db_path_temp'])
                if not cfg_out['b_use_old_temporary_tables']:
                    h5remove_tables(cfg_out)
        except IOError as e:
            print(e)

        if cfg_out['b_skip_if_up_to_date']:
            if not cfg_out['b_use_old_temporary_tables']:
                # Copying previous store data to temporary one
                l.info('Copying previous store data to temporary one:')
                tbl = 'is absent'
                try:
                    with pd.HDFStore(cfg_out['db_path']) as storeOut:
                        for tbl in (cfg_out['tables'] + cfg_out['tables_log']):
                            if not tbl:
                                continue
                            try:  # Check output store
                                if tbl in storeOut:  # avoid harmful sortAndPack errors
                                    h5sort_pack(cfg_out['db_path'], os_path.basename(
                                        cfg_out['db_path_temp']), tbl)
                                else:
                                    raise HDF5ExtError(f'Table {tbl} does not exist')
                            except HDF5ExtError as e:
                                if tbl in storeOut.root.__members__:
                                    print('Node exist but store is not conforms Pandas')
                                    getstore_and_print_table(storeOut, tbl)
                                raise e  # exclude next processing
                            except RuntimeError as e:
                                l.error(
                                    'failed check on copy. May be need first to add full index to original store? Trying: ')
                                nodes = storeOut.get_node(tbl).__members__  # sorted(, key=number_key)
                                for n in nodes:
                                    tbl_cur = tbl if n == 'table' else tbl + '/' + n
                                    l.info(tbl_cur, end=', ')
                                    storeOut.create_table_index(tbl_cur, columns=['index'],
                                                                kind='full')  # storeOut[tbl_cur]
                                # storeOut.flush()
                                l.error('Trying again')
                                if (cfg_out['db'] is not None) and cfg_out['db'].is_open:
                                    cfg_out['db'].close()
                                    cfg_out['db'] = None
                                h5sort_pack(cfg_out['db_path'], os_path.basename(
                                    cfg_out['db_path_temp']), tbl)

                except HDF5ExtError as e:
                    l.warning(e.args[0])   # print('processing all source data... - no table with previous data')
                    cfg_out['b_skip_if_up_to_date'] = False
                except Exception as e:
                    print('processing all source data... - no previous data (output table {}): {}'.format(
                        tbl, '\n==> '.join([s for s in e.args if isinstance(s, str)])))
                    cfg_out['b_skip_if_up_to_date'] = False
                else:
                    l.info('Will append data only from new files.')
        if (cfg_out['db'] is None) or not cfg_out['db'].is_open:
            # todo: create group if table that path directs to
            # Open temporary output file to return
            for attempt in range(2):
                try:
                    cfg_out['db'] = pd.HDFStore(cfg_out['db_path_temp'])
                    break
                except IOError as e:
                    print(e)
                except HDF5ExtError as e:  #
                    print('can not use old temporary output file. Deleting it...')
                    os_remove(cfg_out['db_path_temp'])
                    # raise(e)

        if cfg_out['b_skip_if_up_to_date']:
            df_log = [cfg_out['db'][tbl_log] for tbl_log in cfg_out['tables_log'] if
                      tbl_log and (tbl_log in cfg_out['db'])]
            if len(df_log) >= 1:
                df_log = df_log[0]
            else:  # if len(df_log) == 0:
                df_log = None
        else:
            df_log = None
            if not cfg_out['b_use_old_temporary_tables']:  # Remove existed tables to write
                h5remove_tables(cfg_out)

    except HDF5ExtError as e:
        if cfg_out['db']:
            cfg_out['db'].close()
            cfg_out['db'] = None
        else:
            df_log = None
        print('can not use old temporary output file. Deleting it...')
        for k in range(10):
            try:
                os_remove(cfg_out['db_path_temp'])
            except PermissionError:
                print(end='.')
                sleep(1)
            except FileNotFoundError:
                print(end=' - was not exist')
        if os_path.exists(cfg_out['db_path_temp']):
            p_name, p_ext = os_path.splitext(cfg_out['db_path_temp'])
            cfg_out['db_path_temp'] = p_name + '-' + p_ext
            print('Can not remove temporary db! => Use another temporary db: "{}"'.format(cfg_out['db_path_temp']))
        sleep(1)
        for k in range(10):
            try:
                cfg_out['db'] = pd.HDFStore(cfg_out['db_path_temp'])
            except HDF5ExtError:
                print(end='.')
                sleep(1)
        cfg_out['b_skip_if_up_to_date'] = False
    except Exception as e:
        l.exception('Can not open temporery hdf5 store')
    return df_log


def h5remove_table(cfg_out, node: Optional[str] = None):
    """
    Removes table, skips if not(node) or no such node in currently open db.
    :param cfg_out: dict with fields: 'db' - hdf5 store
    :param node: str, table name
    :modifies cfg_out['db']: because of changing due to possible reopening here
    Note: Raises HDF5ExtError on KeyError if no such node in cfg_out['db'].filename and it is have not opened
    """
    try:
        if node and (node in cfg_out['db']):
            cfg_out['db'].remove(node)
    except KeyError:
        l.info('Trable when removing {}. Solve Pandas bug by reopen store.'.format(node))
        sleep(1)
        cfg_out['db'].close()
        db_filename = cfg_out['db'].filename
        cfg_out['db'] = None
        sleep(1)
        cfg_out['db'] = pd.HDFStore(db_filename, mode='r+')  # cfg_out['db_path_temp']
        try:
            cfg_out['db'].remove(node)
        except KeyError:
            raise HDF5ExtError('Can not remove table "{}"'.format(node))


def h5remove_tables(cfg_out):
    """
    Removes tables and tables_log from db with retrying if error. Flashes operations
    :param cfg_out: with fields:
    - tables, tables_log: tables names
    - db: hdf5 store
    :return:
    """
    name_prev = ''  # used to filter already deleted children (how speed up?)
    for tbl in sorted(cfg_out['tables'] + cfg_out['tables_log']):
        if len(name_prev) < len(tbl) and tbl.startswith(name_prev) and tbl[len(name_prev)] == '/':  # filter
            continue  # parent of this nested have deleted on previous iteration
        for i in range(1, 4):
            try:
                h5remove_table(cfg_out, tbl)
                name_prev = tbl
                break
            except ClosedFileError as e:  # file is not open
                l.error('wating %s (/3) because of error: %s', i, str(e))
                sleep(i)
            # except HDF5ExtError as e:
            #     break  # nothing to remove
        else:
            l.error('not successed => Reopening...')
            cfg_out['db'] = pd.HDFStore(cfg_out['db_path_temp'])
            h5remove_table(cfg_out, tbl)
    cfg_out['db'].flush()

# ----------------------------------------------------------------------
class TemporaryMoveChilds:
    """
        Save childs before delete tbl_parent
        #for h5find_tables(store, '', parent_name=tbl_parent)
    """

    def __init__(self, cfg_out, tbl_parent):
        self.cfg_out = cfg_out
        self.tbl_parent = tbl_parent

    def __enter__(self):

        self.childs = []
        try:
            parent_group = self.cfg_out['db'].get_storer(self.tbl_parent).group
            nodes = parent_group.__members__
            self.childs = ['/' + self.tbl_parent + '/' + g for g in nodes if (g != 'table') and (g != '_i_table')]
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

        return self.childs

    def __exit__(self, exc_type, ex_value, ex_traceback):
        if exc_type is None:
            for tblD in self.childs:
                self.cfg_out['db']._handle.move_node(tblD.replace(self.tbl_parent, 'to_copy_back'),
                                                     newparent='/' + self.tbl_parent, createparents=True,
                                                     overwrite=True)
        self.cfg_out['db'].flush()
        # cfg_out['db'].move('/'.join(tblD.replace(tbl_parent,'to_copy_back'), tblD))
        # cfg_out['db'][tblD] = df # need to_hdf(format=table)
        return False


# ----------------------------------------------------------------------

def h5remove_duplicates(cfg, cfg_table_keys):
    """
    Remove duplicates by coping tables to memory
    :param cfg: dict with keys:
        keys specified by cfg_table_keys
        chunksize - for data table
        logfield_fileName_len, nfiles - for log table
    :param cfg_table_keys: list, in which 'tables_log' means that cfg['tables_log'] is a log table. Alternatively group tables in subsequences such that log tables names is after data table in each subsequence (cfg[cfg_table_keys[group]])
    :return:
    """

    # load data frames from store to memory removing duplicates
    dfs = {}
    b_need_remove = {}  # will remove tables if will found duplicates
    for cfgListName in cfg_table_keys:
        for tbl in unzip_if_need(cfg[cfgListName]):
            if tbl in cfg['db']:
                dfs[tbl] = cfg['db'][tbl]
                # dfs[tbl].index.is_monotonic_increasing? .is_unique()?
                b_dup = dfs[tbl].index.duplicated(keep='last')
                b_need_remove[tbl] = np.any(b_dup)
                if b_need_remove[tbl]:
                    l.info('{} duplicates in {} (first at {})'.format(
                        sum(b_dup), tbl, dfs[tbl].index[np.flatnonzero(b_dup)[0]]))
                    dfs[tbl] = dfs[tbl][~b_dup]

    # update data frames in store
    if any(list(b_need_remove.values())):
        l.info('Remove duplicates. ')
        for cfgListName in cfg_table_keys:
            for i_in_group, tbl in unzip_if_need_enumerated(cfg[cfgListName]):
                if b_need_remove[tbl]:
                    try:
                        with TemporaryMoveChilds(cfg, tbl):
                            h5remove_table(cfg, tbl)
                            if cfgListName == 'tables_log' or i_in_group > 0:
                                cfg['db'].append(tbl, dfs[tbl], data_columns=True, index=False,
                                                 expectedrows=cfg['nfiles'],
                                                 min_itemsize={'values': cfg['logfield_fileName_len']})
                            else:
                                cfg['db'].append(tbl, dfs[tbl], data_columns=True, index=False,
                                                 chunksize=cfg['chunksize'])
                    except Exception as e:
                        l.exception('Table %s not recorded because of error when removing duplicates', tbl)
                        # cfg['db'][tbl].drop_duplicates(keep='last', inplace=True) #returns None
    else:
        l.info('Not need remove duplicates. ')


def create_indexes(cfg_out, cfg_table_keys):
    # Create full indexes. Must be done because of using ptprepack in h5move_tables() below
    l.debug('Creating index')
    for cfgListName in cfg_table_keys:
        for i_in_group, tbl in unzip_if_need_enumerated(cfg_out[cfgListName]):
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
                l.warning('Index in table "{}" not created - error: '.format(tbl) +
                          '\n==> '.join([s for s in e.args if isinstance(s, str)]))
            # except TypeError:
            #     print('can not create index for table "{}"'.format(tbl))


def h5move_tables(cfg_out, tbl_names=None):
    """
    Copy tables tbl_names from one store to another using ptrepack. If fail to store
    in specified location then creates new store and tries to save there.
    :param cfg_out: dict - must have fields:
        'db_path_temp': source of not sorted tables
        'db_path', 'db_base': full and base name (extension ".h5" will be added if absent) of hdf store to put
        'tables' and 'tables_log' - if tbl_names not specified
    :param tbl_names: list of strings or list of lists (or tuples) of strings.
    List of lists is useful to keep order of operation: put nested tables last.

        Strings are names of hdf5 tables to copy
    :return: None if all success else if have errors - dict of locations of last tried savings for each table
    """
    l.info('move tables:')
    if tbl_names is None:  # copy all cfg_out tables
        tbl_names = cfg_out['tables'] + cfg_out['tables_log']
    storage_basename = os_path.splitext(cfg_out['db_base'])[0]
    storage_basenames = {}

    tables = list(unzip_if_need(tbl_names))
    if tables:
        tbl = tables[0]
    else:
        raise Ex_nothing_done('no tables to move')

    # Check if table to write is ok
    with pd.HDFStore(cfg_out['db_path']) as store, pd.HDFStore(cfg_out['db_path_temp']) as store_in:
        try:
            out_is_bad = (not ((tbl in store) and (tbl in store_in))) or store[tbl] is None
        except TypeError:
            out_is_bad = True

        if out_is_bad:
            l.warning('Removing %s table %s because it is not pandas', cfg_out['tables'][0], cfg_out['db_path'])
        elif set(store[tbl].columns) != set(store_in[tbl].columns):
            # Out table have different columns (why? - h5sort_pack not removed columns metaddata?)
            out_is_bad = True
            l.warning(
                'removing %s table from %s having different columns. Note: checking implemented only for 1st table!',
                cfg_out['tables'][0], cfg_out['db_path'])
        if out_is_bad:
            cfg_out['db'] = store
            h5remove_table(cfg_out, cfg_out['tables'][0])

    for tbl in tables:
        try:
            h5sort_pack(cfg_out['db_path_temp'], storage_basename + '.h5', tbl)  # (fileOutF, FileCum, strProbe)
            sleep(2)
        except Exception as e:
            l.error('Error: "{}"\nwhen write {} to {}'.format(e, tbl, cfg_out['db_path_temp']))

            if False:
                storage_basename = os_path.splitext(cfg_out['db_base'])[0] + "-" + tbl.replace('/', '-') + '.h5'
                l.info('so start write to {}'.format(storage_basename))
                try:
                    h5sort_pack(cfg_out['db_path_temp'], storage_basename, tbl)
                    sleep(4)
                except Exception as e:
                    storage_basename = cfg_out['db_base'] + '-other_place.h5'
                    l.error('Error: "{}"\nwhen write {} to original place so start write to {}'.format(e, tbl,
                                                                                                       storage_basename))
                    try:
                        h5sort_pack(cfg_out['db_path_temp'], storage_basename, tbl)
                        sleep(8)
                    except:
                        l.error(tbl + ': no success')
                storage_basenames[tbl] = storage_basename
    if storage_basenames == {}:
        storage_basenames = None
    return storage_basenames


def h5index_sort(cfg_out,
                 out_storage_name=None,
                 in_storages=None,
                 tables: Optional[Iterable[Union[str, Tuple[str]]]] = None) -> None:
    """
    Checks if tables in store have sorted index
     and if not then sort it by loading, sorting and saving data
    :param cfg_out: dict - must have fields:
        'path': tables to check monotonous and sort/
        'db_path_temp': source of not sorted tables
        'base': base name (extension ".h5" will be added if absent) of hdf store to put
        'tables' and 'tables_log' - if tables not specified

        'dt_from_utc'
    :param tables: iterable of table names, may be grouped in tuples
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
        # if True:
        # store= pd.HDFStore(cfg_out['db_path'])
        b_need_save = False
        b_have_duplicates = False
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
                    l.info(tbl + ' - sorted')
                else:
                    b_have_duplicates = True
                    l.warning(tbl + ' - sorted, but have duplicates')

                    # experimental
                    if cfg_out['b_remove_duplicates']:
                        l.warning(tbl + ' - removing duplicates - experimental!')
                        h5remove_duplicates({
                            **cfg_out,
                            'tables': [t for t in cfg_out['tables'] if t],
                            'tables_log': [t for t in cfg_out['tables_log'] if t],
                            'db': store,
                            }, cfg_table_keys=['tables', 'tables_log'])
                continue
            else:
                b_need_save = True
                l.warning(tbl + ' - not sorted!')
                print(repr(store.get_storer(tbl).group.table))

                df_index, itm = multiindex_timeindex(df.index)
                if __debug__:
                    plt.figure('index is not sorted')
                    plt.plot(df_index)  # np.diff(df.index)
                    plt.show()

                if not itm is None:
                    l.warning('sorting multiindex...')
                    df = df.sort_index()  # inplace=True
                    if df.index.is_monotonic:
                        if df.index.is_unique:
                            l.warning('Ok')
                        else:
                            b_have_duplicates = True
                            l.warning('Ok, but have duplicates')
                        continue
                    else:
                        print('Failure!')
                else:
                    pass  # will sorting by prepack in h5move_tables #l.warning('skipped of sorting ')
        if b_have_duplicates:
            l.warning('To drop duplicates restart with [output_files][b_remove_duplicates] = True')
        else:
            l.info('Ok, no duplicates')
        if b_need_save:
            # out to store
            h5move_tables(cfg_out, tbl_names=tables)

            # store = pd.HDFStore(cfg_out['db_path_temp'])
            # store.create_table_index(tbl, columns=['index'], kind='full')
            # store.create_table_index(cfg_out['tables_log'][0], columns=['index'], kind='full') #tbl+r'/logFiles'
            # h5_append(store, df, log, cfg_out, cfg_out['dt_from_utc'])
            # store.close()
            # h5sort_pack(cfg_out['db_path_temp'], out_storage_name, tbl) #, ['--overwrite-nodes=true']


def h5del_obsolete(cfg_out: Mapping[str, Any],
                   log: Mapping[str, Any],
                   df_log: pd.DataFrame) -> Tuple[bool, bool]:
    """
    Check that current file has been processed and it is up to date
    Removes all data from the store table and logtable which time >= time of data
    in log record of current file if it is changed!

    Also removes duplicates in the table if found duplicate records in the log
    :param cfg_out: dict, must have field
        'db' - handle of opened store
        'b_use_old_temporary_tables' - for message
        'tables_log', 'tables' - to able check and deleting
    :param log: dict, with info about current data, must have fields for compare:
        'fileName' - in format as in log table to able find duplicates
        'fileChangeTime', datetime - to able find outdate data
    :param df_log: dataframe, log table loaded from store before updating
    :return: (bExistOk, bExistDup)
    """
    if cfg_out['tables'] is None or df_log is None:
        return (False, False)

    rows_for_file = df_log[df_log['fileName'] == log['fileName']]
    L = len(rows_for_file)
    bExistDup = False  # not detected yet
    bExistOk = False  # not detected yet
    if L:
        if L > 1:
            bExistDup = True
            print('Duplicate entries in log => will be removed from tables! (detected "{}")'.format(log['fileName']))
            cfg_out['b_remove_duplicates'] = True
            if cfg_out['b_use_old_temporary_tables']:
                print('Consider set [output_files].b_use_old_temporary_tables=0,[in].b_skip_if_up_to_date=0')
            print('Continuing...')
            imax = np.argmax([r.to_pydatetime() for r in rows_for_file['fileChangeTime']])
        else:
            imax = 0
        last_fileChangeTime = rows_for_file['fileChangeTime'][imax].to_pydatetime()
        if last_fileChangeTime >= log['fileChangeTime']:
            bExistOk = True
            print('>', end='')
            rows_for_file = rows_for_file[np.arange(len(rows_for_file)) != imax]  # keep up to date record
        if not rows_for_file.empty:  # delete other records
            print('removing obsolete stored data rows:', end=' ')
            qstr = "index>=Timestamp('{}')".format(rows_for_file.index[0])
            qstrL = "fileName=='{}'".format(rows_for_file['fileName'][0])
            try:
                tbl=''
                for tbl, tbl_log in zip(cfg_out['tables'], cfg_out['tables_log']):
                    Ln = cfg_out['db'].remove(tbl_log, where=qstrL)  # useful if it is not a child
                    L = cfg_out['db'].remove(tbl, where=qstr)
                    print('{} in table/{} in log'.format(L, Ln))
            except NotImplementedError as e:
                l.exception('Can not delete obsolete rows, so removing full tables %s & %s', tbl, tbl_log)
                cfg_out['db'].remove(tbl_log)
                cfg_out['db'].remove(tbl)
                bExistOk = False
                bExistDup = False
    return (bExistOk, bExistDup)


def h5init(cfg_in: Mapping[str, Any], cfg_out: Dict[str, Any]):
    """
    Init cfg_out database (hdf5 data store) information in cfg_out _if it is not exist_
    :param: cfg_in - configuration dicts, with fields:
            path if no 'db_path' in cfg_out
            source_dir_words (optional), default: ['source', 'WorkData', 'workData'] - see getDirBaseOut()
            nfiles (optional)
            b_skip_if_up_to_date (optional)
    :param: cfg_out - configuration dict, where all fields are optional. Do nothing if cfg_out['tables'] is None

    Sets fields of cfg_out _if not exist_. Updated fields are:
        % paths %:
    tables, tables_log: tables names of data and log (metadata)
    db_dir, db_base: parts of db (hdf5 store) path - based on cfg_in and cfg_in['source_dir_words']
    db_path: db_dir + "/" + db_base
    db_path_temp: temporary h5 file name
        % other %:
    nfiles: default 1, copied from cfg_in - to set store.append() 'expectedrows' argument
    b_skip_if_up_to_date: default False, copied from cfg_in
    chunksize: default None
    logfield_fileName_len: default 255
    b_remove_duplicates: default False
    b_use_old_temporary_tables: default False

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

    # automatic names
    cfg_source_dir_words = cfg_in['source_dir_words'] if 'source_dir_words' in cfg_in else ['raw', 'source', 'WorkData',
                                                                                            'workData']
    auto = {'db_ext': 'h5'}
    if ('db_path' in cfg_out) and cfg_out['db_path']:
        # print(cfg_out)
        # print('checking db_path "{}" is absolute'.format(cfg_out['db_path']))
        if os_path.isabs(cfg_out['db_path']):
            auto['db_path'] = cfg_out['db_path']
        else:
            cfg_out['db_base'] = cfg_out['db_path']
            cfg_out['db_path'] = ''
            auto['db_path'] = os_path.split(cfg_in.get('path' if 'path' in cfg_in else 'db_path'))[0]
            auto['db_base'] = cfg_out['db_base']  # os_path.join(auto['db_base'], cfg_out['db_path'])

    else:
        auto['db_path'] = os_path.split(cfg_in.get('path'))[0]
    auto['db_path'], auto['db_base'], auto['table'] = getDirBaseOut(auto['db_path'], cfg_source_dir_words)
    auto['db_base'] = os_path.splitext(auto['db_base'])[0]  # extension is specified in db_ext

    cfg_out['db_dir'], cfg_out['db_base'] = pathAndMask(
        *[cfg_out[spec] if (spec in cfg_out and cfg_out[spec]) else auto[spec] for spec in
          ['db_path', 'db_base', 'db_ext']])
    dir_create_if_need(cfg_out['db_dir'])
    cfg_out['db_path'] = os_path.join(cfg_out['db_dir'], cfg_out['db_base'])

    # set_field_if_no(cfg_out, 'db_base', auto['db_base'] + ('.h5' if not auto['db_base'].endswith('.h5') else ''))
    # set_field_if_no(cfg_out, 'db_path', os_path.join(auto['db_path'], cfg_out['db_base']+ ('.h5' if not cfg_out['db_base'].endswith('.h5') else '')))

    # Will save to temporary file initially
    set_field_if_no(cfg_out, 'db_path_temp', cfg_out['db_path'][:-3] + '_not_sorted.h5')

    set_field_if_no(cfg_out, 'nfiles', cfg_in.get('nfiles') if 'nfiles' in cfg_in else 1)

    if 'tables' in cfg_out and cfg_out['tables']:
        set_field_if_no(cfg_out, 'tables_log', [((tab + '/logFiles') if tab else '') for tab in cfg_out['tables']])
    elif 'table' in cfg_out and cfg_out['table']:
        cfg_out['tables'] = [cfg_out['table']]
        set_field_if_no(cfg_out, 'tables_log', [cfg_out['table'] + '/logFiles'])
    else:
        if auto['table'] == '':
            auto['table'] = os_path.basename(cfg_in['cfgFile'])
            l.warning('Can not dertermine table_name from file structure. '
                      'Set [tables] in ini! Now use table_name "{}"'.format(auto['table']))
        cfg_out['tables'] = [auto['table']]
        set_field_if_no(cfg_out, 'tables_log', [auto['table'] + '/logFiles'])


# functions to iterate db log instead of files in dir
def query_time_range(cfg_in):
    if cfg_in.get('min_time'):
        if cfg_in.get('max_time'):
            query_range = "index>=Timestamp('{min_time}') & index<=Timestamp('{max_time}')".format_map(cfg_in)
        else:
            query_range = "index>=Timestamp('{min_time}')".format_map(cfg_in)
    elif cfg_in.get('max_time'):
        query_range = "index<=Timestamp('{max_time}')".format_map(cfg_in)
    else:
        query_range = None
    return query_range


def h5log_rows_gen(cfg_in, tbl_intervals=None):
    """
    Dicts from each h5 log row
    :param cfg_in: dict, with fields:
        db_path, str: name of hdf5 pandas store where is log table
        min_time, max_time: datetime, optional, allows range table_log rows
        table_log, str: name of log table - table with intervals:
            index - pd.DatetimeIndex for starts of intervals
            DateEnd - pd.Datetime col for ends of intervals
        Example name: cfg_in['table_log'] ='/CTD_SST_48M/logRuns'
    :param tbl_intervals: name of log table to use instead cfg_in['table_log']
    """
    if tbl_intervals is None:
        tbl_intervals = cfg_in['table_log']
    query_range = query_time_range(cfg_in)

    with pd.HDFStore(cfg_in['db_path'], mode='r') as store:
        print("loading from {db_path}: ".format_map(cfg_in), end='')
        for n, rp in enumerate(store.select(tbl_intervals, where=query_range).itertuples()):
            r = dict(zip(rp._fields, rp))
            yield (r)  # r.Index, r.DateEnd


def h5log_names_gen(cfg_in, f_row_to_name=lambda r: '{Index:%y%m%d_%H%M}-{DateEnd:%H%M}'.format_map(r)):
    """
    Genereates outputs of f_row_to_name function which receves dicts from each h5 log row (see h5log_rows_gen)
    :param cfg_in:
    :param f_row_to_name: function(dict) where dict have fields from h5 log row.
        By default returns string suitable to name files by start-end date/time
    :return:
    :modifies cfg: adds/replaces field 'log_row': dict from h5 log row. This allows use this dict also
    Replasing for veuszPropagate.ge_names() to use tables instead files
    """
    for row in h5log_rows_gen(cfg_in):
        cfg_in['log_row'] = row
        yield f_row_to_name(row)
