#!/usr/bin/env python3
# coding:utf-8
"""
  Author:  Andrey Korzh <ao.korzh@gmail.com>
  Purpose: Convert (multiple) csv and alike text files to pandas hdf5 store with addition of log table
  Created: 26.02.2016
  Modified: 29.08.2020
"""
import logging
import sys
import re
import warnings
from codecs import open
from collections import OrderedDict
from datetime import datetime
from functools import partial
from pathlib import Path, PurePath
from time import sleep
from typing import Any, Callable, Iterator, Mapping, MutableMapping, Optional, Sequence, Tuple, Union
from tables.exceptions import HDF5ExtError

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask import delayed, compute, persist
# my:
from utils2init import init_file_names, Ex_nothing_done, set_field_if_no, cfg_from_args, my_argparser_common_part, \
    this_prog_basename, init_logging, standard_error_info, ExitStatus
from to_pandas_hdf5.h5_dask_pandas import h5_append, filter_global_minmax, filter_local
from to_pandas_hdf5.h5toh5 import h5temp_open, h5remove_duplicates, h5remove_duplicates_by_loading, h5move_tables, \
    h5index_sort, h5init, h5del_obsolete, create_indexes
import to_pandas_hdf5.csv_specific_proc
import utils_time_corr

# import dask; dask.config.set(scheduler='synchronous')  # !!! for test

if __name__ == '__main__':
    if False:  # True:  temporary for debug
        from dask.distributed import Client

        client = Client(
            processes=False)  # navigate to http://localhost:8787/status to see the diagnostic dashboard if you have Bokeh installed
        # processes=False: avoide inter-worker communication for computations releases the GIL (numpy, da.array)  # without is error
    else:
        pass


    l = None  # see main(): l = init_logging(logging, None, cfg['program']['log'], cfg['program']['verbose'])
else:
    l = logging.getLogger(__name__)
version = '0.2.0'


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
Add data from CSV-like files
to Pandas HDF5 store*.h5
----------------------------"""}, version)
    # Configuration sections
    s = p.add_argument_group('in', 'all about input files')
    s.add('--path', default='.',  # nargs=?,
             help='path to source file(s) to parse. Use patterns in Unix shell style')
    s.add('--b_search_in_subdirs', default='False',
             help='search in subdirectories, used if mask or only dir in path (not full path)')
    s.add('--exclude_dirs_endswith_list', default='toDel, -, bad, test, TEST',
             help='exclude dirs which ends with this srings. This and next option especially useful when search recursively in many dirs')
    s.add('--exclude_files_endswith_list', default='coef.txt, -.txt, test.txt',
             help='exclude files which ends with this srings')
    s.add('--b_skip_if_up_to_date', default='True',
             help='exclude processing of files with same name and which time change is not bigger than recorded in database (only prints ">" if detected). If finds updated version of same file then deletes all data which corresponds old file and after it before procesing of next files: 1. Program copyes all data to temporary storage and 2. deletes old data there if found. 3. New data appended. 4. Data tables copyed back with deleting original data')
    s.add('--dt_from_utc_seconds', default='0',
             help='source datetime data shift. This constant will be substructed just after the loading to convert to UTC. Can use other suffixes instead of "seconds"')
    s.add('--dt_from_utc_hours', default='0',
             help='source datetime data shift. This constant will be substructed just after the loading to convert to UTC. Can use other suffixes instead of "hours"')
    s.add('--fs_float',
             help='sampling frequency, uses this value to calculate intermediate time values between time changed values (if same time is assined to consecutive data)')
    s.add('--fs_old_method_float',
             help='sampling frequency, same as ``fs_float``, but courses the program to use other method. If smaller than mean data frequency then part of data can be deleted!(?)')
    s.add('--header',
             help='comma separated list matched to input data columns to name variables. To autoadd exclude column from loading use ",,". Can contain type suffix i.e.'
             '- (float): (default),'
             '- (text): required if read_csv() can not convert to float/time, and if specific converter used, '
             '- (time): for ISO 8601 format (only?)')
    s.add('--cols_load_list',
             help='comma separated list of names from header to be loaded from csv. Do not use "/" char, or type suffixes like in ``header`` for them. Defaut - all columns')
    s.add('--cols_not_use_list',
             help='comma separated list of names from header to not be saved in hdf5 store.')
    s.add('--cols_use_list',
             help='comma separated list of names from header to be saved in hdf5 store. Because of autodeleting converted (text) columns include them here if want to keep. New columns that created in csv_specific_proc() after read_csv() must also be here to get')

    s.add('--skiprows_integer', default='1',
             help='skip rows from top. Use 1 to skip one line of header')
    s.add('--b_raise_on_err', default='True',
             help='if false then not rise error on rows which can not be loaded (only shows warning). Try set "comments" argument to skip them without warning')
    s.add('--delimiter_chars',
             help='parameter of dask.read_csv(). Default None is useful for fixed length format')
    s.add('--max_text_width', default='1000',
             help='maximum length of text fields (specified by "(text)" in header) for dtype in numpy.loadtxt')
    s.add('--chunksize_percent_float',
             help='percent of 1st file length to set up hdf5 store tabe chunk size')
    s.add('--blocksize_int', default='20000000',
             help='bytes, chunk size for loading and processing csv')
    s.add('--sort', default='True',
             help='if time not sorted then modify time values trying to affect small number of values. This is different from sorting rows which is performed at last step after the checking table in database')
    s.add('--fun_date_from_filename',
             help='function(file_stem: str, century: Optional[str]=None) -> Any[compartible to input of pandas.to_datetime()]: to get date from filename to time column in it.')
    s.add('--fun_proc_loaded',
             help='function(df: Dataframe, cfg_in: Optional[Mapping[str, Any]] = None) -> Dataframe/DateTimeIndex: to update/calculate new parameters from loaded data  before filtering. If output is Dataframe then function should have meta_out attribute which is Callable[[np.dtype, Iterable[str], Mapping[str, dtype]], Dict[str, np.dtype]]')
    s.add('--csv_specific_param_dict',
             help='not default parameters for function in csv_specific_proc.py used to load data')

    s = p.add_argument_group('out',
                             'all about output files')
    s.add('--db_path', help='hdf5 store file path')
    s.add('--table',
              help='table name in hdf5 store to write data. If not specified then will be generated on base of path of input files. Note: "*" is used to write blocks in autonumbered locations (see dask to_hdf())')
    # s.add('--tables_list',
    #           help='tables names in hdf5 store to write data (comma separated)')
    s.add('--b_insert_separator',
              help='insert NaNs row in table after each file data end')
    s.add('--b_reuse_temporary_tables', default='False',
              help='Warning! Set True only if temporary storage already have good data!'
                   'if True and b_skip_if_up_to_date= True then program will not replace temporary storage with current storage before adding data to the temporary storage')
    s.add('--b_remove_duplicates', default='False', help='Set True if you see warnings about')
    s.add('--b_del_temp_db', default='False', help='temporary h5 file will be deleted after operation')

    s = p.add_argument_group('filter',
                             'filter all data based on min/max of parameters')
    s.add('--min_date', help='minimum time')
    s.add('--max_date', help='maximum time')
    s.add('--min_dict', help='List with items in  "key:value" format. Sets to NaN data of ``key`` columns if it is below ``value``')
    s.add('--max_dict', help='List with items in  "key:value" format. Sets to NaN data of ``key`` columns if it is above ``value``')
    s.add('--b_bad_cols_in_file_name', default='True',
              help='find string "<Separator>no_<col1>[,<col2>]..." in file name and set all values of col1[, col2] to NaN. Here <Separator> is one of -_()[, ')

    s = p.add_argument_group('program',
                             'program behaviour')
    s.add('--return', default='<end>',  # nargs=1,
               choices=['<cfg_from_args>', '<gen_names_and_log>', '<end>'],
               help='<cfg_from_args>: returns cfg based on input args only and exit, <gen_names_and_log>: execute init_input_cols() and also returns fun_proc_loaded function... - see main()')
    return (p)


def init_input_cols(cfg_in=None):
    """
        Append/modify dictionary cfg_in for parameters of dask/pandas load_csv() function and of save to hdf5.
    :param cfg_in: dictionary, may has fields:
        header (required if no 'cols') - comma/space separated string. Column names in source file data header
        (as in Veusz standard input dialog), used to find cfg_in['cols'] if last is not cpecified
        dtype - type of data in column (as in Numpy loadtxt)
        converters - dict (see "converters" in Numpy loadtxt) or function(cfg_in) to make dict here
        cols_load - list of used column names

    :return: modified cfg_in dictionary. Will have fields:
        cols - list constructed from header by spit and remove format cpecifiers: '(text)', '(float)', '(time)'
        cols_load - list[int], indexes of ``cols`` in needed to save order
        coltime/coldate - assigned to index of 'Time'/'Date' column
        dtype: numpy.dtype of data after using loading function but before filtering/calculating fields
            numpy.float64 - default and for '(float)' format specifier
            numpy string with length cfg_in['max_text_width'] - for '(text)'
            datetime64[ns] - for coldate column (or coltime if no coldate) and for '(time)'
        col_index_name - index name for saving Pandas frame. Will be set to name of cfg_in['coltime'] column if not exist already
        used in main() default time postload proc only (if no specific loader which calculates and returns time column for index)
        cols_loaded_save_b - columns mask of cols_load to save (some columns needed only before save to calulate
        of others). Default: excluded (text) columns and index and coldate
        (because index saved in other variable and coldate may only used to create it)

    Example
    -------
    header= u'`Ensemble #`,txtYY_M_D_h_m_s_f(text),,,Top,`Average Heading (degrees)`,`Average Pitch (degrees)`,stdPitch,`Average Roll (degrees)`,stdRoll,`Average Temp (degrees C)`,txtVe_none(text) txtVn_none(text) txtVup(text) txtErrVhor(text) txtInt1(text) txtInt2(text) txtInt3(text) txtInt4(text) txtCor1(text) txtCor2(text) txtCor3(text) txtCor4(text),,,SpeedE_BT SpeedN_BT SpeedUp ErrSpeed DepthReading `Bin Size (m)` `Bin 1 Distance(m;>0=up;<0=down)` absorption IntScale'.strip()
    """

    if cfg_in is None: cfg_in = dict()
    set_field_if_no(cfg_in, 'max_text_width', 2000)
    set_field_if_no(cfg_in, 'dt_from_utc', 0)
    dtype_text_max = '|S{:.0f}'.format(cfg_in['max_text_width'])  # '2000 #np.str

    if cfg_in.get('header'):  # if header specified
        re_sep = ' *(?:(?:,\n)|[\n,]) *'  # not isolate "`" but process ",," right
        cfg_in['cols'] = re.split(re_sep, cfg_in['header'])
        # re_fast = re.compile(u"(?:[ \n,]+[ \n]*|^)(`[^`]+`|[^`,\n ]*)", re.VERBOSE)
        # cfg_in['cols']= re_fast.findall(cfg_in['header'])
    elif not 'cols' in cfg_in:  # cols is from header, is specified or is default
        warnings.warn("default 'cols' is deprecated, use init_input_cols({header: "
                      "'stime, latitude, longitude'}) instead", DeprecationWarning, 2)
        cfg_in['cols'] = ('stime', 'latitude', 'longitude')

    # default parameters dependent from ['cols']
    cols_load_b = np.ones(len(cfg_in['cols']), np.bool8)
    set_field_if_no(cfg_in, 'comments', '"')

    # assign data type of input columns
    b_was_no_dtype = not 'dtype' in cfg_in
    if b_was_no_dtype:
        cfg_in['dtype'] = np.array([np.float64] * len(cfg_in['cols']))
        # 32 gets trunkation errors after 6th sign (=> shows long numbers after dot)
    elif isinstance(cfg_in['dtype'], str):
        cfg_in['dtype'] = np.array([np.dtype(cfg_in['dtype'])] * len(cfg_in['cols']))
    elif isinstance(cfg_in['dtype'], list):
        # prevent numpy array(list) guess minimal dtype because dtype for represent dtype of dtype_text_max may be greater
        numpy_cur_dtype = np.min_scalar_type(cfg_in['dtype'])
        numpy_cur_dtype_len = numpy_cur_dtype.itemsize / np.dtype((numpy_cur_dtype.kind, 1)).itemsize
        cfg_in['dtype'] = np.array(cfg_in['dtype'], '|S{:.0f}'.format(
            max(len(dtype_text_max), numpy_cur_dtype_len)))

    for sCol, sDefault in (['coltime', 'Time'], ['coldate', 'Date']):
        if (sCol not in cfg_in):
            # if cfg['col(time/date)'] is not provided try find 'Time'/'Date' column name
            if not (sDefault in cfg_in['cols']):
                sDefault = sDefault + '(text)'
            if not (sDefault in cfg_in['cols']):
                continue
            cfg_in[sCol] = cfg_in['cols'].index(sDefault)  # assign 'Time'/'Date' column index to cfg['col(time/date)']
        elif isinstance(cfg_in[sCol], str):
            cfg_in[sCol] = cfg_in['cols'].index(cfg_in[sCol])

    if not 'converters' in cfg_in:
        cfg_in['converters'] = None
    else:
        if not isinstance(cfg_in['converters'], dict):
            # suspended evaluation required
            cfg_in['converters'] = cfg_in['converters'](cfg_in)
        if b_was_no_dtype:
            # converters produce datetime64[ns] for coldate column (or coltime if no coldate):
            cfg_in['dtype'][cfg_in['coldate' if 'coldate' in cfg_in
            else 'coltime']] = 'datetime64[ns]'

    # process format cpecifiers: '(text)','(float)','(time)' and remove it from ['cols'],
    # also find not used cols cpecified by skipping name between commas like in 'col1,,,col4'
    for i, s in enumerate(cfg_in['cols']):
        if len(s) == 0:
            cols_load_b[i] = 0
            cfg_in['cols'][i] = f'NotUsed{i}'
        else:
            b_i_not_in_converters = (not (i in cfg_in['converters'].keys())) \
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

    if cfg_in.get('cols_load'):
        cols_load_b &= np.isin(cfg_in['cols'], cfg_in['cols_load'])
    else:
        cfg_in['cols_load'] = np.array(cfg_in['cols'])[cols_load_b]
    # apply settings that more narrows used cols
    if 'cols_not_use' in cfg_in:
        cols_load_in_used_b = np.isin(cfg_in['cols_load'], cfg_in['cols_not_use'], invert=True)
        if not np.all(cols_load_in_used_b):
            cfg_in['cols_load'] = cfg_in['cols_load'][cols_load_in_used_b]
            cols_load_b = np.isin(cfg_in['cols'], cfg_in['cols_load'])

    col_names_out = cfg_in['cols_load'].copy()
    # Convert ``cols_load`` to index (to be compatible with numpy loadtxt()), names will be in cfg_in['dtype'].names
    cfg_in['cols_load'] = np.int32([cfg_in['cols'].index(c) for c in cfg_in['cols_load'] if c in cfg_in['cols']])
    # not_cols_load = np.array([n in cfg_in['cols_not_use'] for n in cfg_in['cols']], np.bool)
    # cfg_in['cols_load']= np.logical_and(~not_cols_load, cfg_in['cols_load'])
    # cfg_in['cols']= np.array(cfg_in['cols'])[cfg_in['cols_load']]
    # cfg_in['dtype']=  cfg_in['dtype'][cfg_in['cols_load']]
    # cfg_in['cols_load']= np.flatnonzero(cfg_in['cols_load'])
    # cfg_in['dtype']= np.dtype({'names': cfg_in['cols'].tolist(), 'formats': cfg_in['dtype'].tolist()})


    cfg_in['cols'] = np.array(cfg_in['cols'])
    cfg_in['dtype_raw'] = np.dtype({'names': cfg_in['cols'],
                                    'formats': cfg_in['dtype'].tolist()})
    cfg_in['dtype'] = np.dtype({'names': cfg_in['cols'][cfg_in['cols_load']],
                                'formats': cfg_in['dtype'][cfg_in['cols_load']].tolist()})

    # Get index name for saving Pandas frame
    b_index_exist = cfg_in.get('coltime') is not None
    if b_index_exist:
        set_field_if_no(cfg_in, 'col_index_name', cfg_in['cols'][cfg_in['coltime']])

    # Output columns mask
    if not 'cols_loaded_save_b' in cfg_in:
        # Mask of only needed output columns (text columns are not more needed after load)
        cfg_in['cols_loaded_save_b'] = np.logical_not(np.array(
            [cfg_in['dtype'].fields[n][0].char == 'S' for n in
             cfg_in['dtype'].names]))  # a.dtype will = cfg_in['dtype']

        if 'coldate' in cfg_in:
            cfg_in['cols_loaded_save_b'][
                cfg_in['dtype'].names.index(
                    cfg_in['cols'][cfg_in['coldate']])] = False
    else:  # list to array
        cfg_in['cols_loaded_save_b'] = np.bool8(cfg_in['cols_loaded_save_b'])

    # Exclude index from cols_loaded_save_b
    if b_index_exist and cfg_in['col_index_name']:
        cfg_in['cols_loaded_save_b'][cfg_in['dtype'].names.index(
            cfg_in['col_index_name'])] = False  # (must index be used separately?)

    # Output columns dtype
    col_names_out = np.array(col_names_out)[cfg_in['cols_loaded_save_b']].tolist() + cfg_in.get('cols_use', [])
    cfg_in['dtype_out'] = np.dtype({
        'formats': [cfg_in['dtype'].fields[n][0] if n in cfg_in['dtype'].names else
                    np.dtype(np.float64) for n in col_names_out],
        'names': col_names_out})

    return cfg_in


def set_filterGlobal_minmax(a: Union[pd.DataFrame, dd.DataFrame],
                            cfg_filter: Optional[Mapping[str, Any]] = None,
                            log: Optional[MutableMapping[str, Any]] = None,
                            dict_to_save_last_time: Optional[MutableMapping[str, Any]] = None
                            ) -> Tuple[Union[pd.DataFrame, dd.DataFrame], pd.DatetimeIndex]:
    """
    Finds bad with filterGlobal_minmax and removes it from a,tim
    Adds remaining 'rows' and 'rows_filtered' to log

    :param a:
    :param cfg_filter: filtering settings, do nothing if None, can has dict 'delayedfunc' if :param:a is a dask dataframe, that have fields:
      - args: delayed values to be args for to dask.compute(), last is len of a. Output will be used as args of function func:
      - func: function to execute on computed args

    :param log: changes inplace - adds ['rows_filtered'] and ['rows'] - number of remaining rows
    :param dict_to_save_last_time: dict where 'time_last' field will be updated
    :return: dataframe with remaining rows

    """

    if log is None:
        log = dict_to_save_last_time or {}

    log['rows_filtered'] = 0

    if cfg_filter is not None:
        meta_time = pd.Series([], name='Time', dtype=np.bool8)  # pd.Series([], name='Time',dtype='M8[ns]')

        # Applying filterGlobal_minmax(a, tim, cfg_filter) to dask or pandas dataframe
        if isinstance(a, dd.DataFrame):  # may be dask or not dask array
            out = filter_global_minmax(a, cfg_filter)  #a.map_partitions(filterGlobal_minmax, None, cfg_filter, meta=meta_time)

            # # i_starts = np.diff(np.append(tim.searchsorted(a.divisions), len(tim))).tolist()
            # i_starts = [len(p) for p in bGood.partitions]
            # # b_ok_da = da.from_array(b_ok, chunk=(tuple(i_starts),)).to_dask_dataframe(index=bGood.index)
            # # dd.from_pandas(pd.Series(b_ok, index=bGood.index.compute().tz_convert('UTC')),
            # #                            npartitions=bGood.npartitions)
            # b_ok = (bGood if b_ok_ds is True else bGood.mask(~b_ok_ds, False))  #.persist()
            # a = a.loc[bGood]

            try:
                log['rows'] = out.shape[0].compute()
                # execute at once
                if 'delayedfunc' in cfg_filter:
                    sum_good, tim, args = compute(
                        out.shape[0],
                        out.index,
                        cfg_filter['delayedfunc']['args']
                        )
                    cfg_filter['delayedfunc']['func'](*args)  # delayedfunc should show messages
                    log['rows'] = args[-1]  # a.shape[0]
                else:
                    sum_good, tim, log['rows'] = compute(
                        out.shape[0],
                        out.index,
                        a.shape[0]
                        )
            except Exception as e:
                l.exception('Can not filter data:')
                sum_good = np.NaN
                log['rows'] = np.NaN
                tim = a.index.compute()
                out = a
        else:
            log['rows'] = len(a)  # shape[0]
            out = filter_global_minmax(a, cfg_filter)  # b_ok_ds.values.compute()?
            tim = out.index
            sum_good = len(tim)

        if sum_good < log['rows']:
                 # and not np.isscalar(b_ok) <=> b_ok.any() and b_ok is not scalar True (True is if not need filter)
            # tim = tim[b_ok]
            #     #a = a.merge(pd.DataFrame(index=np.flatnonzero(b_ok)))  # a = a.compute(); a[b_ok]
            #
            # a = a.join(pd.DataFrame(index=tim), how='right')
            #     tim= tim[b_ok.values] #.iloc

            log['rows_filtered'] = log['rows'] - sum_good
            log['rows'] = sum_good
    else:
        tim = a.index.compute() if isinstance(a, dd.DataFrame) else a.index
        out = a

    # Save last time to can filter next file
    if dict_to_save_last_time:
        try:
            dict_to_save_last_time['time_last'] = tim[-1]
        except IndexError:
            l.warning('no data!')
    return out, tim


def filter_local_with_file_name_settings(d: Union[pd.DataFrame, dd.DataFrame],
                                         cfg: Mapping[str, Any],
                                         path_csv: PurePath) -> Union[pd.DataFrame, dd.DataFrame]:
    """
    Set all data in columns to NaN if file name has string "{separator}no_{Name1[, Name2...]}"
    where:
        separator is one of "-_,;([" sybmols
        names Name1... matched to data column names. Except "Ox" - this abbreviaton mean "O2, O2ppm"

    Only if cfg['filter']['b_bad_cols_in_file_name'] is True
    :param d: DataFrame
    :param cfg: must have field 'filter'
    :param path_csv: file path name
    :return: filtered d

    """

    # general filtering
    d = filter_local(d, cfg['filter'])

    # filtering based on file name
    if cfg['filter'].get('b_bad_cols_in_file_name'):
        splitted_str = re.split('[-_,;([]no_', path_csv.stem)
        if len(splitted_str) < 2:
            return d
        bad_col_str = splitted_str[-1].split(';)]')[0]
        bad_col_list = bad_col_str.split(',')
        if 'Ox' in bad_col_list:
            bad_col_list.remove('Ox')
            bad_col_list.extend(['O2', 'O2ppm'])
        bad_col_list_checked = set(d.columns).intersection(bad_col_list)
        if len(bad_col_list_checked) < len(bad_col_list):
            l.warning('not found columns to set bad: %s', bad_col_list_checked.symmetric_difference(bad_col_list))
            if not bad_col_list_checked:
                return d

        if isinstance(d, pd.DataFrame):
            d.loc[:, bad_col_list_checked] = np.NaN
        else:
            mode_of_chained_assignment = pd.get_option('mode.chained_assignment')

            def set_nan(df):
                pd.set_option('mode.chained_assignment', None)
                df.loc[:, bad_col_list_checked] = np.NaN
                pd.set_option('mode.chained_assignment', mode_of_chained_assignment)
                return df

            d = d.map_partitions(set_nan, meta=d)

            # d.assign(bad_col_list_checked=np.NaN)
        l.info('bad column%s set: %s',
               's are' if len(bad_col_list_checked) > 1 else ' is',
               bad_col_list_checked)

    return d


# ----------------------------------------------------------------------

def read_csv(paths: Sequence[Union[str, Path]],
             **cfg_in: Mapping[str, Any]
             ) -> Tuple[Union[pd.DataFrame, dd.DataFrame, None], Optional[dd.Series]]:
    """
    Reads csv in dask DataFrame
    Calls cfg_in['fun_proc_loaded'] (if specified)
    Calls time_corr: corrects/checks Time (with arguments defined in cfg_in fields)
    Sets Time as index
    :param paths: list of file names
    :param cfg_in: contains fields for arguments of dask.read_csv correspondence:
        
        names=cfg_in['cols'][cfg_in['cols_load']]
        usecols=cfg_in['cols_load']
        error_bad_lines=cfg_in['b_raise_on_err']
        comment=cfg_in['comments']
        
        Other arguments corresponds to fields with same name:
        dtype=cfg_in['dtype']
        delimiter=cfg_in['delimiter']
        converters=cfg_in['converters']
        skiprows=cfg_in['skiprows']
        blocksize=cfg_in['blocksize']
        
        Also cfg_in has filds:
            dtype_out: numpy.dtype, which "names" field used to detrmine output columns
            fun_proc_loaded: None or Callable[
            [Union[pd.DataFrame, np.array], Mapping[str, Any], Optional[Mapping[str, Any]]],
             Union[pd.DataFrame, pd.DatetimeIndex]]
            If it returns pd.DataFrame then it also must has attribute:
                meta_out: Callable[[np.dtype, Iterable[str], Mapping[str, dtype]], Dict[str, np.dtype]]

            See also time_corr() for used fields
    
    
    
    :return: tuple (a, b_ok) where
        a:      dask dataframe with time index and only columns listed in cfg_in['dtype_out'].names
        b_ok:   time correction rezult boolean array
    """
    read_csv_args_to_cfg_in = {
        'dtype': 'dtype_raw',
        'names': 'cols',
        'error_bad_lines': 'b_raise_on_err',
        'comment': 'comments',
        'delimiter': 'delimiter',
        'converters': 'converters',
        'skiprows': 'skiprows',
        'blocksize': 'blocksize'
        }
    read_csv_args = {arg: cfg_in[key] for arg, key in read_csv_args_to_cfg_in.items()}
    read_csv_args.update({
        'skipinitialspace': True,
        'usecols': cfg_in['dtype'].names,
        'header': None})
    # removing "ParserWarning: Both a converter and dtype were specified for column k - only the converter will be used"
    if read_csv_args['converters']:
        read_csv_args['dtype'] = {k: v[0] for i, (k, v) in enumerate(read_csv_args['dtype'].fields.items()) if i not in read_csv_args['converters']}
    try:
        try:
            # raise ValueError('Temporary')
            ddf = dd.read_csv(paths, **read_csv_args)
            # , engine='python' - may help load bad file

            # index_col=False  # force pandas to _not_ use the first column as the index (row names) - no in dask
            # names=None, squeeze=False, prefix=None, mangle_dupe_cols=True,
            # engine=None, true_values=None, false_values=None, skipinitialspace=False,
            #     nrows=None, na_values=None, keep_default_na=True, na_filter=True, verbose=False,
            #     skip_blank_lines=True, parse_dates=False, infer_datetime_format=False, keep_date_col=False,
            #     date_parser=None, dayfirst=False, iterator=False, chunksize=None, compression='infer',
            #     thousands=None, decimal=b'.', lineterminator=None, quotechar='"', quoting=0,
            #     escapechar=None, encoding=None, dialect=None, tupleize_cols=None,
            #      warn_bad_lines=True, skipfooter=0, skip_footer=0, doublequote=True,
            #     delim_whitespace=False, as_recarray=None, compact_ints=None, use_unsigned=None,
            #     low_memory=True, buffer_lines=None, memory_map=False, float_precision=None)
        except ValueError as e:
            l.exception('dask lib can not load data. Trying pandas lib...')
            del read_csv_args['blocksize']  # because pandas.read_csv has no such arg
            for i, nf in enumerate(paths):
                df = pd.read_csv(nf, **read_csv_args, index_col=False)  # chunksize=cfg_in['blocksize']
                if i > 0:
                    raise NotImplementedError('list of files => need concatenate data')
            ddf = dd.from_pandas(df, chunksize=cfg_in['blocksize'])  #
        except NotImplementedError as e:
            l.exception('If file "%s" have no data try to delete it', paths)
            return None, None
    except Exception as e:  # for example NotImplementedError if bad file
        msg = 'Bad file. skip!'
        ddf = None
        if cfg_in['b_raise_on_err']:
            l.exception('%s\n Try set [in].b_raise_on_err = False\n', msg)
            raise (e)
        else:
            l.exception(msg)
    if __debug__:
        l.debug('read_csv initialised')
    if ddf is None:
        return None, None

    meta_time = pd.Series([], name='Time', dtype='datetime64[ns, UTC]')  # np.dtype('datetime64[ns]')
    meta_time_index = pd.DatetimeIndex([], dtype='datetime64[ns, UTC]', name='Time')
    meta_df_with_time_col = cfg_in['cols_load']
    meta_time_and_mask = {'Time': 'datetime64[ns, utc]', 'b_ok': np.bool8}
    # meta_time_and_mask.time = meta_time_and_mask.time.astype('M8[ns]')
    # meta_time_and_mask.b_ok = meta_time_and_mask.b_ok.astype(np.bool8)


    utils_time_corr.tim_min_save = pd.Timestamp('now', tz='UTC')  # initialisation for time_corr_df()
    utils_time_corr.tim_max_save = pd.Timestamp(0, tz='UTC')

    n_overlap = 2 * int(np.ceil(cfg_in['fs'])) if cfg_in.get('fs') else 50

    # Process ddf and get date in ISO string or numpy standard format
    cfg_in['file_stem'] = Path(paths[0]).stem  # may be need in func below to extract date
    date = None
    meta_out = getattr(cfg_in['fun_proc_loaded'], 'meta_out', None)
    try:
        try:
            if meta_out is not None:
                # fun_proc_loaded() will return not only date column but full data DataFrame. Go to exception handler
                # todo: find better condition
                raise TypeError('fun_proc_loaded() will return full data dataframe')

            date = ddf.map_partitions(lambda *args, **kwargs: pd.Series(
                cfg_in['fun_proc_loaded'](*args, **kwargs)), cfg_in, meta=meta_time)  # meta_time_index
            # date = date.to_series()

            l.info(*('time correction in %s blocks...', date.npartitions) if date.npartitions > 1 else
            ('time correction...',))

            def time_corr_df(t, cfg_in):
                """ Convert tuple returned by time_corr() to dataframe
                """
                return pd.DataFrame.from_dict(OrderedDict(zip(
                    meta_time_and_mask.keys(), utils_time_corr.time_corr(t, cfg_in))))
                # return pd.DataFrame.from_items(zip(meta_time_and_mask.keys(), time_corr(t, cfg_in)))
                # pd.Series()

            df_time_ok = date.map_overlap(
                time_corr_df, before=n_overlap, after=n_overlap, cfg_in=cfg_in, meta=meta_time_and_mask)
            # try:
            #     df_time_ok = df_time_ok.persist()  # triggers all csv_specific_proc computations
            # except Exception as e:
            #     l.exception(
            #         'Can not speed up by persist, doing something that can trigger error to help it identificate...')
            #     df_time_ok = time_corr_df(date.compute(), cfg_in=cfg_in)

            if cfg_in.get('csv_specific_param'):
                # need run this:
                # ddf = to_pandas_hdf5.csv_specific_proc.proc_loaded_corr(
                #     ddf, cfg_in, cfg_in['csv_specific_param'])
                ddf = ddf.map_partitions(to_pandas_hdf5.csv_specific_proc.proc_loaded_corr, cfg_in, cfg_in['csv_specific_param'])  #, meta=meta_out

        except (TypeError, Exception) as e:
            # fun_proc_loaded() will return full data DataFrame (having Time col)

            meta_out = meta_out(cfg_in['dtype']) if callable(meta_out) else None

            l.info('processing csv data with time correction%s...' %
                   f' in {ddf.npartitions} blocks' if ddf.npartitions > 1 else '')

            # initialisation for utils_time_corr.time_corr():

            def fun_proc_loaded_and_time_corr_df(df, cfg_in):
                """fun_proc_loaded() then time_corr()
                """
                df_out = cfg_in['fun_proc_loaded'](df, cfg_in)
                return df_out.assign(**dict(zip(meta_time_and_mask.keys(),
                                                utils_time_corr.time_corr(df_out.Time, cfg_in))))

            #ddf = ddf.map_partitions(cfg_in['fun_proc_loaded'], cfg_in, meta=meta_out)
            ddf = ddf.map_overlap(fun_proc_loaded_and_time_corr_df, before=n_overlap, after=n_overlap,
                            cfg_in=cfg_in, meta={**meta_out, **meta_time_and_mask})
            df_time_ok = ddf[['Time', 'b_ok']]

    except IndexError:
        print('no data?')
        return None, None

    meta_out = cfg_in.get('dtype_out', meta_out)
    if meta_out:
        if cfg_in.get('meta_out_df') is None:
            # construct meta (in format of dataframe to able set index name)
            dict_dummy = {k: np.zeros(1, dtype=v[0]) for k, v in meta_out.fields.items()}
            meta_out_df = pd.DataFrame(dict_dummy, index=pd.DatetimeIndex([], name='Time', tz='UTC'))
        else:
            meta_out_df = cfg_in['meta_out_df']
    else:
        meta_out_df = None

    if isinstance(df_time_ok, dd.DataFrame):
        nbad_time = True

    # df_time_ok.compute(scheduler='single-threaded')
        # todo: return for msg something like bad_time_ind = df_time_ok['b_ok'].fillna(False).ne(True).to_numpy().nonzero()  # or query('') because below commented as compute() is long
        # nbad_time = len(df_time_ok['b_ok']) - df_time_ok['b_ok'].sum().compute()
        # if nbad_time:
        #     nonzero = []
        #     for b_ok in df_time_ok['b_ok'].ne(True).partitions:  # fillna(0).
        #         b = b_ok.compute()
        #         if b.any():
        #             nonzero.extend(b.to_numpy().nonzero()[0])
        #     l.info('Bad time values (%d): %s%s', nbad_time, nonzero[:20], ' (shows first 20)' if nbad_time > 20 else '')

        # try:  # catch exception not works
        #     # if not interpolates (my condition) use simpler method:
        #     df_time_ok.Time = df_time_ok.Time.map_overlap(pd.Series.interpolate, before=n_overlap, after=n_overlap,
        #                                                   inplace=True, meta=meta_time)  # method='linear', - default
        # except ValueError:
        # df_time_ok.Time = df_time_ok.Time.where(df_time_ok['b_ok']).map_overlap(
        #     pd.Series.fillna, before=n_overlap, after=n_overlap, method='ffill', inplace=False, meta=meta_time)
    else:
        nbad_time = len(df_time_ok['b_ok']) - df_time_ok['b_ok'].sum()
        l.info('Bad time values (%d): %s%s', nbad_time,
               df_time_ok['b_ok'].fillna(False).ne(True).to_numpy().nonzero()[0][:20],
               ' (shows first 20)' if nbad_time > 20 else '')

    # Define range_message()' delayed args
    n_ok_time = df_time_ok['b_ok'].sum()
    n_all_rows = df_time_ok.shape[0]

    if cfg_in.get('min_date'):  # condition for calc. tim_min_save/tim_max_save in utils_time_corr() and to set
        # index=cfg_in['min_date'] or index=cfg_in['max_date'] where it is out of config range

        @delayed(pure=True)
        def range_source():
            # Only works as delayed (except of in debug inspecting mode) because utils_time_corr.time_corr(Time) must be processed to set 'tim_min_save', 'tim_max_save'
            return {k: getattr(utils_time_corr, attr) for k, attr in (('min', 'tim_min_save'), ('max', 'tim_max_save'))}

        # Filter data that is out of config range (after range_message()' args defined as both uses df_time_ok['b_ok'])
        df_time_ok['b_ok'] = df_time_ok['b_ok'].mask(
            (df_time_ok['Time'] < pd.Timestamp(cfg_in['min_date'], tz='UTC')) |
            (df_time_ok['Time'] > pd.Timestamp(cfg_in['max_date'], tz='UTC')), False)
        #nbad_time = True  # need filter by 'b_ok': df_time_ok['b_ok'].any().compute()  # compute() takes too long
    else:
        # to be computed before filtering
        range_source = df_time_ok['Time'].reduction(
            chunk=lambda x: pd.Series([x.min(), x.max()]),
            # combine=chunk_fun, not works - gets None in args
            aggregate=lambda x: pd.Series([x.iloc[0].min(), x.iloc[-1].max()], index=['min', 'max']),  # skipna=True is default
            meta=meta_time)
        # df_time_ok['Time'].apply([min, max], meta=meta_time) - not works for dask (works for pandas)
        # = df_time_ok.divisions[0::df_time_ok.npartitions] if (isinstance(df_time_ok, dd.DataFrame) and df_time_ok.known_divisions)

    range_message_args = [range_source, n_ok_time, n_all_rows]
    def range_message(range_source, n_ok_time, n_all_rows):
        t = range_source() if isinstance(range_source, Callable) else range_source
        l.info(f'loaded source range: {t["min"]:%Y-%m-%d %H:%M:%S} - {t["max"]:%Y-%m-%d %H:%M:%S %Z}, {n_ok_time:d}/{n_all_rows:d} rows')

        #delayed_range_message = range_message  # range_source

        # df_time_ok.loc[df_time_ok['b_ok'], 'Time'] = pd.NaT
        # try:  # interpolate that then helps use Time as index:
        #     df_time_ok.Time = df_time_ok.Time.interpolate(inplace=False)                 # inplace=True - not works, method='linear', - default
        # except ValueError:  # if not interpolates (my condition) use simpler method:
        #     df_time_ok.Time = df_time_ok.Time.fillna(method='ffill', inplace=True)

        # # dask get IndexingError: Unalignable boolean Series provided as indexer (index of the boolean Series and of the indexed object do not match):
        # ddf_out = ddf.loc[df_time_ok['b_ok'], list(cfg_in['dtype_out'].names)].set_index(
        #    df_time_ok.loc[df_time_ok['b_ok'], 'Time'], sorted=True)

    out_cols = list(cfg_in['dtype_out'].names)

    if not ddf.known_divisions:  # meta_out is None  # always

        @delayed(pure=True)
        def time_index_ok(new_df, new_time_ok):
            #if new_time_ok['b_ok'].any():
            #     need values below (i.e. drop index) because of due to map_overlap() the time.index is shifted relative to df.index
            new_df_filt = new_df.loc[new_time_ok['b_ok'].values, out_cols]
            new_time_filt = new_time_ok.loc[new_time_ok['b_ok'].values, 'Time']
            return new_df_filt.set_index(new_time_filt)   # new_df_filt is pandas: no 'sorted' arg
            #else:
            #    return results_list

        ddf_out_list = []
        for dl_f, dl_time_ok in zip(ddf.to_delayed(), df_time_ok.to_delayed()):
            ddf_out_list.append(time_index_ok(dl_f, dl_time_ok))
            # ddf_out_list.append(dl_f[dl_time_ok['b_ok']].set_index(df_time_ok.loc[df_time_ok['b_ok'], 'Time'], sorted=True))


        ddf_out = dd.from_delayed(ddf_out_list, divisions='sorted', meta=meta_out_df)
    else:
        #ddf, df_time_ok = compute(ddf, df_time_ok)  # for testing
        # Removing rows with bad time
        ddf_out = ddf.loc[df_time_ok['b_ok'], out_cols]

        if False and __debug__:
            len_out = len(df_time_ok)
            print('out data length before del unused blocks:', len_out)

        # Removing rows with bad time (continue)
        df_time_ok = df_time_ok[df_time_ok['b_ok']]

        if False and __debug__:
            len_out = len(df_time_ok)
            print('out data length:', len_out)
            print('index_limits:', df_time_ok.divisions[0::df_time_ok.npartitions])
            sum_na_out, df_time_ok_time_min, df_time_ok_time_max = compute(
                df_time_ok['Time'].notnull().sum(), df_time_ok['Time'].min(),  df_time_ok['Time'].max())
            print('out data len, nontna, min, max:', sum_na_out, df_time_ok_time_min, df_time_ok_time_max)

        ddf_out = ddf_out.set_index(df_time_ok['Time'], sorted=True)  #

    # try:
    #     ddf_out = ddf_out.persist()  # triggers all csv_specific_proc computations
    # except Exception as e:
    #     l.exception('Can not speed up by persist')


    # print('data loaded shape: {}'.format(ddf.compute(scheduler='single-threaded').shape))  # debug only
    # if nbad_time: #and cfg_in.get('keep_input_nans'):
    #     df_time_ok = df_time_ok.set_index('Time', sorted=True)
    #     # ??? after I set index: ValueError: Not all divisions are known, can't align partitions. Please use `set_index` to set the index.
    #     ddf_out = ddf_out.loc[df_time_ok['b_ok'], :].repartition(freq='1D')

    # if isinstance(df_time_ok, dd.DataFrame) else df_time_ok['Time'].compute()
    # **({'sorted': True} if a_is_dask_df else {}
    # [cfg_in['cols_load']]
    # else:
    #     col_temp = ddf.columns[0]
    #     b = ddf[col_temp]
    #     b[col_temp] = b[col_temp].map_partitions(lambda s, t: t[s.index], tim, meta=meta)
    #     ddf = ddf.reset_index().set_index('index').set_index(b[col_temp], sorted=True).loc[:, list(cfg_in['dtype_out'].names)]

    # date = pd.Series(tim, index=ddf.index.compute())  # dd.from_dask_array(da.from_array(tim.values(),chunks=ddf.divisions), 'Time', index=ddf.index)
    # date = dd.from_pandas(date, npartitions=npartitions)
    # ddf = ddf.loc[:, list(cfg_in['dtype_out'].names)].set_index(date, sorted=True)

    # ddf = ddf.loc[:, list(cfg_in['dtype_out'].names)].compute()
    # ddf.set_index(tim, inplace=True)
    # ddf = dd.from_pandas(ddf, npartitions=npartitions)

    logger = logging.getLogger("dask")
    logger.addFilter(lambda s: s.getMessage() != "Partition indices have overlap.")
    # b_ok = df_time_ok['b_ok'].to_dask_array().compute() if isinstance(
    #     df_time_ok, dd.DataFrame) else df_time_ok['b_ok'].to_numpy()
    # ddf_out.index.name = 'Time' not works
    # b_ok_ds= df_time_ok.set_index('Time')['b_ok']
    return ddf_out, {'func': range_message, 'args': range_message_args}  # , b_ok_ds


# @delayed
# def loadtxt(cfg_in):
#     if not cfg_in['b_raise_on_err']:
#         try:
#             a= np.genfromtxt(nameFull, dtype= cfg_in['dtype'],
#                 delimiter= cfg_in['delimiter'],
#                 usecols= cfg_in['cols_load'],
#                 converters= cfg_in['converters'],
#                 skip_header= cfg_in['skiprows'],
#                 comments= cfg_in['comments'],
#                 invalid_raise= False) #,autostrip= True
#             #warnings.warn("Mean of empty slice.", RuntimeWarning)
#         except Exception as e:
#             print(standard_error_info(e), '- Bad file. skip!\n')
#             a = None
#     else:
#         try:
#             a= np.loadtxt(nameFull, dtype= cfg_in['dtype'],
#                           delimiter= cfg_in['delimiter'],
#                           usecols= cfg_in['cols_load'],
#                           converters= cfg_in['converters'],
#                           skiprows= cfg_in['skiprows'])
#         except Exception as e:
#             print('{}\n Try set [in].b_raise_on_err= False'.format(e))
#             raise(e)
#     return a

class open_if_can:
    def __init__(self, path_log):
        self.path_log = path_log

    def __enter__(self):
        try:
            self.flog = open(self.path_log, 'a+', encoding='cp1251')
        except FileNotFoundError as e:
            print(standard_error_info(e), '- skip logging operations!\n')
            self.flog = None
        except Exception as e:
            l.exception('saving log of operations')
        return self.flog

    def __exit__(self, exc_type, exc_value, traceback):
        if self.flog:
            self.flog.close()

        if exc_type:
            l.error("Aborted %s", self, exc_info=(exc_type, exc_value, traceback))

        return False


def h5_names_gen(cfg_in, cfg_out: Mapping[str, Any], **kwargs) -> Iterator[Path]:
    """
    Yields Paths from cfg_in['paths'] items
    :updates: cfg_out['log'] fields 'fileName' and 'fileChangeTime'

    :param cfg_in: dict, must have fields:
        - paths: iterator - returns full file names

    :param cfg_out: dict, with fields needed for h5_dispenser_and_names_gen() and print info:
        - Date0, DateEnd, rows: must have (should be updated) after yield
        - log: (will be created if absent) current file info - else prints "file not processed"
    """
    set_field_if_no(cfg_out, 'log', {})
    for name_full in cfg_in['paths']:
        pname = Path(name_full)

        cfg_out['log']['fileName'] = pname.name[-cfg_out['logfield_fileName_len']:-4]
        cfg_out['log']['fileChangeTime'] = datetime.fromtimestamp(pname.stat().st_mtime)

        try:
            yield pname     # Traceback error line pointing here is wrong
        except GeneratorExit:
            print('Something wrong?')
            return

        # Log to logfile
        if cfg_out['log'].get('Date0'):
            strLog = '{fileName}:\t{Date0:%d.%m.%Y %H:%M:%S}-{DateEnd:%d.%m %H:%M:%S%z}\t{rows}rows'.format(
                **cfg_out['log'])  # \t{Lat}\t{Lon}\t{strOldVal}->\t{mag}
        else:
            strLog = "file not processed"
        l.info(strLog)



def h5_close(cfg_out: Mapping[str, Any]) -> None:
    """
    Closes cfg_out['db'] store, removes duplicates (if need) and creates indexes
    :param cfg_out: dict, to remove duplicates it must have 'b_remove_duplicates': True
    :return: None
    """
    try:
        print('')
        cfg_table_keys = ['tables_have_wrote'] if ('tables_have_wrote' in cfg_out) else ('tables', 'tables_log')
        if cfg_out['b_remove_duplicates']:
            tbl_dups = h5remove_duplicates_by_loading(cfg_out, cfg_table_keys=cfg_table_keys)
            # or h5remove_duplicates() but it can take very long time
        create_indexes(cfg_out, cfg_table_keys)
    except Exception as e:
        l.exception('\nError of adding data to temporary store: ')

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
        try:
            cfg_out['db'].close()
        except HDF5ExtError:
            l.exception(f"Error closing: {cfg_out['db']}")
        if cfg_out['db'].is_open:
            print('Wait store closing...')
            sleep(2)
        cfg_out['db'] = None
        return


def h5_dispenser_and_names_gen(
        cfg_in: Mapping[str, Any],
        cfg_out: Optional[MutableMapping[str, Any]] = None,
        fun_gen: Callable[[Mapping[str, Any], Mapping[str, Any]], Iterator[Any]] = h5_names_gen,
        b_close_at_end: Optional[bool] = True, **kwargs) -> Iterator[Tuple[int, Any]]:
    """
    Prepares HDF5 store to insert/update data and yields fun_gen(...) outputs:
        - Opens DB (see h5temp_open() requirements)
        - Finds data labels by fun_gen(): default are file names and their modification date
        - Removes outdated data
        - Generates file names which data is absent in DB (to upload new/updated data)
        - Tide up DB: creats index, closes DB.
    This function supports storing data in HDF5 used in h5toGrid: dataframe's child 'table' node always contain adjasent
    "log" node. "log" dataframe labels parent dataframe's data segments and allows to check it for existance and
    relevance.

    :param cfg_in: dict, must have fields:
        - fields used in your fun_gen(cfg_in, cfg_out)
    :param cfg_out: dict, must have fields
        - log: dict, with info about current data, must have fields for compare:
            - 'fileName' - in format as in log table to able find duplicates
            - 'fileChangeTime', datetime - to able find outdate data
        - b_skip_if_up_to_date: if True then not yields previously processed files. But if file was changed 1. removes stored data and 2. yields fun_gen(...) result
        - tables_have_wrote: sequence of table names where to create index
    :param fun_gen: function with arguments (cfg_in, cfg_out, **kwargs), that
        - generates data labels, default are file's ``Path``s,
        - updates cfg_out['log'] fields 'fileName' (by current label) and 'fileChangeTime' needed to store and find
        data. They named historically, in principle, you can use any unique identificator composed of this two fields.

    :return: Iterator that returns (i1, pname):
        - i1: index (starting with 1) of fun_gen generated data label (may be file)
        - pname: fun_gen output (may be path name)
        Skips (i1, pname) for existed labels that also has same stored data label (file) modification date
    :updates:
        - cfg_out['db'],
        - cfg_out['b_remove_duplicates'] and
        - that what fun_gen() do
    """
    # copy data to temporary HDF5 store and open it or work with source data if
    df_log_old, cfg_out['db'], cfg_out['b_skip_if_up_to_date'] = h5temp_open(**cfg_out)
    try:
        for i1, gen_out in enumerate(fun_gen(cfg_in, cfg_out, **kwargs), start=1):
            # if current file it is newer than its stored data then remove data and yield its info to process again
            if cfg_out['b_skip_if_up_to_date']:
                b_stored_newer, b_stored_dups = h5del_obsolete(
                    cfg_out, cfg_out['log'], df_log_old, cfg_out.get('field_to_del_older_records')
                    )
                if b_stored_newer:
                    continue  # not need process: current file already loaded
                if b_stored_dups:
                    cfg_out['b_remove_duplicates'] = True  # normally no duplicates but we set if detect

            yield i1, gen_out

    except Exception as e:
        l.exception('\nError preparing data:')
        sys.exit(ExitStatus.failure)
    finally:
        if b_close_at_end:
            h5_close(cfg_out)


def get_fun_proc_loaded_converters(cfg_in: MutableMapping[str, Any]
                                   ) -> Callable[[pd.DataFrame, Mapping[str, Any], Mapping[str, Any]], Iterator[Any]]:
    """
    Assign castom prep&proc and modify cfg_in['converters']
    in dependance to cfg_in['cfgFile'] name
    :param cfg_in:
    :return fun_proc_loaded: Callable if cfgFile name match found or cfg_in['fun_proc_loaded'] specified explicitly else None
    Modifies: cfg_in['converters'] if cfg_file ends with 'IdrRedas' or 'csv_iso_time'
    """

    cfg_file = Path(cfg_in['cfgFile']).stem

    if cfg_file.endswith('IdrRedas'):
        # cfg_in['converters'] = {cfg_in['coltime']: lambda txtD_M_YYYY_hhmmssf:
        # np.datetime64(b'%(2)b-%(1)b-%(0)bT%(3)b' % dict(
        #     zip([b'0', b'1', b'2', b'3'], (txtD_M_YYYY_hhmmssf[:19].replace(b' ', b'/').split(b'/')))))}
        def reformat_date(txtD_M_YYYY_hhmmssf):
            d, m, yyyy_hhmmss = txtD_M_YYYY_hhmmssf[:19].split('/')
            yyyy, hhmmss = yyyy_hhmmss.split(' ')
            return np.datetime64('%s-%s-%sT%s' % (yyyy, m, d, hhmmss))

        cfg_in['converters'] = {cfg_in['coltime']: reformat_date}
        # b'{2}-{1}-{0}T{3}' % (txtD_M_YYYY_hhmmssf[:19].replace(b' ',b'/').split(b'/')))} #len=19 because bug of bad milliseconds
        # fun_proc_loaded= proc_loaded_IdrRedas
    elif cfg_file.endswith('csv_iso_time'):
        # more prepare for time in standard ISO 8601 format
        cfg_in['converters'] = {cfg_in['coltime']: lambda txtYY_M_D_h_m_s_f: np.array(
            '20{0:02.0f}-{1:02.0f}-{2:02.0f}T{3:02.0f}:{4:02.0f}:{5:02.0f}.{6:02.0f}0'.format(
                *np.array(np.fromstring(txtYY_M_D_h_m_s_f, dtype=np.uint8, sep=','), dtype=np.uint8)),
            dtype='datetime64[ns]')}  # - np.datetime64('2009-01-01T00:00:00', dtype='datetime64[ns]')

    fun_proc_loaded = cfg_in.get('fun_proc_loaded')
    if fun_proc_loaded:
        return fun_proc_loaded
    try:
        fun_suffix = re.findall('csv_(.*)', cfg_file)[0]
    except IndexError:
        return None  # fun_proc_loaded is not needed
    try:
        fun_suffix = fun_suffix.replace('&', '_and_')  # sea_and_sun
        suffix_st = len('proc_loaded_')
        fun_names = [f for f in dir(to_pandas_hdf5.csv_specific_proc) if f.startswith('proc_loaded_') and fun_suffix.endswith(f[suffix_st:])]
        if len(fun_names) == 1:
            fun_proc_loaded = getattr(to_pandas_hdf5.csv_specific_proc, fun_names[0])
            return fun_proc_loaded
        else:
            raise AttributeError('found %d functions in to_pandas_hdf5.csv_specific_proc for loading %s',
                                 len(fun_names), cfg_file)
    except AttributeError:
        l.debug('No fun_proc_loaded')  # fun_proc_loaded is not needed probably

    # if cfg_file.endswith('Sea&Sun'):
    #     fun_proc_loaded = to_pandas_hdf5.csv_specific_proc.proc_loaded_sea_and_sun
    # elif cfg_file.endswith('Idronaut'):
    #     fun_proc_loaded = to_pandas_hdf5.csv_specific_proc.proc_loaded_Idronaut
    # elif cfg_file.endswith('nav_supervisor') or cfg_file.endswith('meteo'):
    #     fun_proc_loaded = to_pandas_hdf5.csv_specific_proc.proc_loaded_nav_supervisor
    # elif cfg_file.endswith('ctd_Schuka'):
    #     fun_proc_loaded = to_pandas_hdf5.csv_specific_proc.proc_loaded_ctd_Schuka
    # elif cfg_file.endswith('ctd_Schuka_HHMM'):
    #     fun_proc_loaded = to_pandas_hdf5.csv_specific_proc.proc_loaded_ctd_Schuka_HHMM
    # elif cfg_file.endswith('csv_log'):
    #     fun_proc_loaded = to_pandas_hdf5.csv_specific_proc.proc_loaded_csv_log
    # elif cfg_file.endswith('chain_Baranov') or cfg_file.endswith('inclin_Baranov'):
    #     fun_proc_loaded = to_pandas_hdf5.csv_specific_proc.proc_loaded_chain_Baranov
    # elif cfg_file.endswith('csv_Baklan'):
    #     fun_proc_loaded = to_pandas_hdf5.csv_specific_proc.proc_loaded_Baklan
    # elif cfg_file.endswith('inclin_Kondrashov'):
    #     fun_proc_loaded = to_pandas_hdf5.csv_specific_proc.proc_loaded_inclin_Kondrashov
    # elif cfg_file.endswith('nav_HYPACK'):
    #     fun_proc_loaded = to_pandas_hdf5.csv_specific_proc.proc_loaded_nav_HYPACK
    # return fun_proc_loaded


def main(new_arg=None, **kwargs):
    """

    :param new_arg: list of strings, command line arguments
    :kwargs: dicts of dictcts (for each ini section): specified values overwrites ini values
    Note: if new_arg=='<cfg_from_args>' returns cfg but it will be None if argument
     argv[1:] == '-h' or '-v' passed to this code
    argv[1] is cfgFile. It was used with cfg files:
        'csv2h5_nav_supervisor.ini'
        'csv2h5_IdrRedas.ini'
        'csv2h5_Idronaut.ini'

    :return:
    todo: add freq attribute to data index in store
    """

    global l

    cfg = cfg_from_args(my_argparser(), new_arg, **kwargs)
    if not cfg or not cfg['program'].get('return'):
        print('Can not initialise')
        return cfg
    elif cfg['program']['return'] == '<cfg_from_args>':  # to help testing
        return cfg

    l = init_logging(logging, None, cfg['program']['log'], cfg['program']['verbose'])
    print('\n', this_prog_basename(__file__), end=' started. ')
    try:
        cfg['in']['paths'], cfg['in']['nfiles'], cfg['in']['path'] = init_file_names(
            **cfg['in'], b_interact=cfg['program']['b_interact'])
    except Ex_nothing_done as e:
        print(e.message)
        return ()

    # Prepare loading and writing specific to format
    # prepare if need extract date from file name
    if cfg['in'].get('fun_date_from_filename') and isinstance(cfg['in']['fun_date_from_filename'], str):
        cfg['in']['fun_date_from_filename'] = eval(
            compile("lambda file_stem, century=None: {}".format(cfg['in']['fun_date_from_filename']), '', 'eval'))
    cfg['in']['fun_proc_loaded'] = get_fun_proc_loaded_converters(cfg['in'])
    cfg['in'] = init_input_cols(cfg['in'])
    # cfg['out']['dtype'] = cfg['in']['dtype_out']
    cfg_out = cfg['out']
    h5init(cfg['in'], cfg_out)

    if cfg['in']['fun_proc_loaded'] is None:
        # Default time processing after loading by dask/pandas.read_csv()
        if 'coldate' not in cfg['in']:  # if Time includes Date then we will just return it
            cfg['in']['fun_proc_loaded'] = lambda a, cfg_in, dummy=None: a[cfg_in['col_index_name']]
        else:                           # else will return Time + Date
            cfg['in']['fun_proc_loaded'] = lambda a, cfg_in, dummy=None: a['Date'] + np.array(
                np.int32(1000 * a[cfg_in['col_index_name']]), dtype='m8[ms]')

    if cfg['in'].get('csv_specific_param'):
        # Add additional configured argument to fun_proc_loaded()
        # and append it with proc_loaded_corr()
        t = getattr(cfg['in']['fun_proc_loaded'], 'meta_out', None)  # save before wrapping
        if t is not None:  # if attribute then it lost during partil wrapping so add it back
            fun_proc_loaded = partial(cfg['in']['fun_proc_loaded'], csv_specific_param=cfg['in']['csv_specific_param'])

            def fun_proc_loaded_folowed_proc_loaded_corr(a, cfg_in):
                a = fun_proc_loaded(a, cfg_in)
                a = to_pandas_hdf5.csv_specific_proc.proc_loaded_corr(a, cfg_in, cfg['in']['csv_specific_param'])
                return a

            cfg['in']['fun_proc_loaded'] = fun_proc_loaded_folowed_proc_loaded_corr
            cfg['in']['fun_proc_loaded'].meta_out = t
        # else: fun_proc_loaded returns only time. We need run proc_loaded_corr() separately (see read_csv())

    if cfg['program']['return'] == '<return_cfg_step_fun_proc_loaded>':  # to help testing
        return cfg
    if cfg['program']['return'] == '<gen_names_and_log>':  # to help testing
        cfg['in']['gen_names_and_log'] = h5_dispenser_and_names_gen
        cfg['out'] = cfg_out
        return cfg

    cfg_out['log'] = {'fileName': None, 'fileChangeTime': None}
    # for lim in ['min', 'max']:
    #     cfg['filter'][f'{lim}_date'] = cfg['in'][f'date_{lim}']
    if True:  # try:   # Writing
        ## Main circle ############################################################
        for i1_file, path_csv in h5_dispenser_and_names_gen(cfg['in'], cfg_out):
            if cfg['in']['nfiles'] > 1:
                l.info('%s. %s: ', i1_file, path_csv.name)
            # Loading and processing data
            d, cfg['filter']['delayedfunc'] = read_csv(
                **{**cfg['in'], 'paths': [path_csv]},
                **{k: cfg['filter'].get(k) for k in ['min_date', 'max_date']}
                )  # , b_ok_ds

            if d is None:
                l.warning('not processing')
                continue
            try:
                # filter
                d, tim = set_filterGlobal_minmax(
                    d, cfg_filter=cfg['filter'], log=cfg_out['log'], dict_to_save_last_time=cfg['in'])
            except TypeError:  # "TypeError: Cannot compare type NaTType with type str_" if len(d) = 0
                l.exception('can not process: no data?')  # warning
                continue

            if cfg_out['log']['rows_filtered']:
                print('filtered out {}, remains {}'.format(cfg_out['log']['rows_filtered'], cfg_out['log']['rows']))
                if not cfg_out['log']['rows']:
                    l.warning('no data! => skip file')
                    continue
            elif cfg_out['log']['rows']:
                print('.', end='')  # , divisions=d.divisions), divisions=pd.date_range(tim[0], tim[-1], freq='1D')
            else:
                l.warning('no data! => skip file')
                continue
            d = filter_local_with_file_name_settings(d, cfg, path_csv)

            h5_append(cfg_out, d, cfg_out['log'], tim=tim)  # , log_dt_from_utc=cfg['in']['dt_from_utc']
    # Sort if have any processed data else don't because ``ptprepack`` not closes hdf5 source if it not finds data
    if cfg['in'].get('time_last'):
        failed_storages = h5move_tables(cfg_out)
        print('Ok.', end=' ')
        h5index_sort(cfg_out, out_storage_name=f"{cfg_out['db_path'].stem}-resorted.h5", in_storages=failed_storages)


if __name__ == '__main__':
    main()

""" trash ##############################################
        

    # ddf_len = len(ddf)
    # counts_divisions = list(range(1, int(ddf_len / cfg_in.get('decimate_rate', 1)), cfg_in['blocksize']))
    # counts_divisions.append(ddf_len)
    #
    # date_delayed = delayed(cfg_in['fun_proc_loaded'], nout=1)(ddf, cfg_in)
    # date = dd.from_delayed(date_delayed, meta=meta_time_index, divisions=ddf.index.divisions)
    # date = dd.from_dask_array(date.values, index=ddf.index)

    # .to_series()
    # if __debug__:
    #     c = df_time_ok.compute()
    # tim = date.compute().values()
    # tim, b_ok = time_corr(tim, cfg_in)

    # return None, None
    # if len(ddf) == 1:  # size
    #     ddf = ddf[np.newaxis]

    # npartitions = ddf.npartitions
    # ddf = ddf.reset_index().set_index('index')
    # col_temp = set(ddf.columns).difference(cfg_in['dtype_out'].names).pop()

    # ddf.index is not unique!
    # if col_temp:
    #      # ddf[col_temp].compute().is_unique # Index.is_monotonic_increasing()
    #     # ddf[col_temp] = ddf[col_temp].map_partitions(lambda s, t: t[s.index], tim, meta=meta)

    # # fun_proc_loaded retuns tuple (date, a)
    # changing_size = False  # ? True  # ?
    # if changing_size:
    #     date_delayed, a = delayed(cfg_in['fun_proc_loaded'], nout=2)(ddf, cfg_in)
    #     ddf_len = len(ddf)
    #     counts_divisions = list(range(1, int(ddf_len / cfg_in.get('decimate_rate', 1)), cfg_in['blocksize']))
    #     counts_divisions.append(ddf_len)
    #     ddf = dd.from_delayed(a, divisions=(0, counts_divisions))
    #     date = dd.from_delayed(date_delayed, meta=meta_time_index, divisions=counts_divisions)
    #     date = dd.from_dask_array(date.values, index=ddf.index)
    #     # date = dd.from_pandas(date.to_series(index=), chunksize=cfg_in['blocksize'], )
    #     # _pandas(date, chunksize=cfg_in['blocksize'], name='Time')
    # else:

    # date.rename('time').to_series().reset_index().compute()
    # date.to_series().repartition(divisions=ddf.divisions[1])


    def time_corr_ar(t, cfg_in):
        # convert tuple returned by time_corr() to dataframe
        return np.array(time_corr(t, cfg_in))
        #return pd.DataFrame.from_items(zip(meta_time_and_mask.keys(), time_corr(t, cfg_in)))
        # pd.Series()
    da.overlap.map_overlap(date.values, time_corr_ar, depth=n_overlap)


# dask can not set index:
try:
    #?! df = d.set_index(tim, sorted=True) # pd.DataFrame(d[list(cfg_out['dtype'].names)], index= tim) #index= False?
    #?! d.set_index(dd.from_pandas(tim.to_series().reset_index(drop=True), npartitions=d.npartitions), sorted=True)
    #d = d.join(tim.to_frame(False).rename(lambda x: 'tim', axis='columns'), npartitions=d.npartitions)
    #divisions = pd.date_range(tim[0], tim[-1] + pd.Timedelta(1, 'D'), freq='1D', normalize=True)
    #d = d.repartition(np.searchsorted(tim,divisions)) #divisions.tolist())
    df = d.join(pd.DataFrame(index=tim), how='right')
        #d.join(dd.from_pandas(pd.DataFrame(tim, columns=['tim']), npartitions=d.npartitions), how='right')
    #df = d.set_index('tim', sorted=True, inplace=True)
except ValueError as e:
    l.warning(': Index not created by Dask, using Pandas... Error:' + '\n==> '.join(
          [s for s in e.args if isinstance(s, str)]))
    df = d.compute()
    df.set_index('tim' if 'tim' in df.columns else tim, inplace=True)
except NotImplementedError as e:
    l.warning(': Index not created by Dask, using Pandas... Error:' + '\n==> '.join(
        [s for s in e.args if isinstance(s, str)]))
    df = d.set_index(tim, compute=True)  #.compute()
            


# http://stackoverflow.com/questions/1111317/how-do-i-print-a-python-datetime-in-the-local-timezone
#import time
#def datetime_to_local_timezone(dt):
    #epoch = dt.timestamp()          # Get POSIX timestamp of the specified datetime.
    #st_time = time.localtime(epoch) # Get struct_time for the timestamp. This will be created using the system's locale and it's time zone information.
    #tz = datetime.timezone(pd.Timedelta(seconds = st_time.tm_gmtoff)) # Create a timezone object with the computed offset in the struct_time.
    #dt.astimezone(tz=None)
    #return dt.astimezone(tz) # Move the datetime instance to the new time zone.

import time, datetime
def nowString():
    # we want something like '2007-10-18 14:00+0100'
    mytz="%+4.4d" % (time.timezone / -(60*60) * 100) # time.timezone counts westwards!
    #returns the local timezone, albeit with opposite sign compared to UTC. So it says "-3600" to express UTC+1.
    #ignores Daylight Saving Time (DST): time.timezone contain the offset to UTC that is used when DST is not in effect

    dt  = datetime.datetime.now()
    dts = dt.strftime('%Y-%m-%d %H:%M')  # %Z (timezone) would be empty
    nowstring="%s%s" % (dts,mytz)
    return nowstring

"""
# np.column_stack((date['YYYY'].astype(np.object), date['-MM-'].astype(np.object)) +
# date['DD'].astype(np.object)+ 'T' + a['txtT'].astype(np.object)
# a['Time']= (date['YYYY'].astype(np.object) + date['-MM-'].astype(np.object) +
# date['DD'].astype(np.object)+ 'T' + a['txtT'].astype(np.object)).astype('datetime64[ns]')
