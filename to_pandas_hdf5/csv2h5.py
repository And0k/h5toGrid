#!/usr/bin/env python3
# coding:utf-8
"""
  Author:  Andrey Korzh <ao.korzh@gmail.com>
  Purpose: Convert (multiple) csv and alike text files to pandas hdf5 store with
           addition of log table
  Created: 26.02.2016
  Modified: 20.12.2019
"""
import logging
import re
# from builtins import input
# from debug import __debug___print
# from  pandas.tseries.offsets import DateOffset
import warnings
from codecs import open
from collections import OrderedDict
from datetime import datetime
from functools import partial
from pathlib import Path, PurePath
from time import sleep
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence, Tuple, Union

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask import delayed

from to_pandas_hdf5.h5_dask_pandas import h5_append, filterGlobal_minmax, filter_local
from to_pandas_hdf5.h5toh5 import h5temp_open, h5remove_duplicates, h5move_tables, h5index_sort, h5init, h5del_obsolete, \
    create_indexes
# my:
from utils2init import init_file_names, Ex_nothing_done, set_field_if_no, cfg_from_args, my_argparser_common_part, \
    this_prog_basename, init_logging
# pathAndMask, dir_walker, bGood_dir, bGood_file
from utils_time import time_corr

if __name__ == '__main__':
    if False:  # True:  temporary for debug
        from dask.distributed import Client

        client = Client(
            processes=False)  # navigate to http://localhost:8787/status to see the diagnostic dashboard if you have Bokeh installed
        # processes=False: avoide inter-worker communication for computations releases the GIL (numpy, da.array)  # without is error

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
Add data from CSV-like files
to Pandas HDF5 store*.h5
----------------------------"""}, version)
    # Configuration sections
    p_in = p.add_argument_group('in', 'all about input files')
    p_in.add('--path', default='.',  # nargs=?,
             help='path to source file(s) to parse. Use patterns in Unix shell style')
    p_in.add('--b_search_in_subdirs', default='False',
             help='search in subdirectories, used if mask or only dir in path (not full path)')
    p_in.add('--exclude_dirs_ends_with_list', default='toDel, -, bad, test, TEST',
             help='exclude dirs wich ends with this srings. This and next option especially useful when search recursively in many dirs')
    p_in.add('--exclude_files_ends_with_list', default='coef.txt, -.txt, test.txt',
             help='exclude files wich ends with this srings')
    p_in.add('--b_skip_if_up_to_date', default='True',
             help='exclude processing of files with same name and wich time change is not bigger than recorded in database (only prints ">" if detected). If finds updated version of same file then deletes all data which corresponds old file and after it brfore procesing of next files')
    p_in.add('--dt_from_utc_seconds', default='0',
             help='add this correction to loading datetime data. Can use other suffixes instead of "seconds"')
    p_in.add('--dt_from_utc_hours', default='0',
             help='add this correction to loading datetime data. Can use other suffixes instead of "hours"')
    p_in.add('--fs_float',
             help='sampling frequency, uses this value to calculate intermediate time values between time changed values (if same time is assined to consecutive data)')
    p_in.add('--fs_old_method_float',
             help='sampling frequency, same as ``fs_float``, but courses the program to use other method. If smaller than mean data frequency then part of data can be deleted!(?)')
    p_in.add('--header',
             help='comma separated list mached to input data columns to name variables. Can contain type suffix i.e. (float) - which is default, (text) - also to convert by specific converter, or (time) - for ISO format only. If it will')
    p_in.add('--cols_load_list',
             help='comma separated list of names from header to be saved in hdf5 store. Do not use "/" char, or type suffixes like in ``header`` for them. Defaut - all columns')
    p_in.add('--cols_not_use_list',
             help='comma separated list of names from header to not be saved in hdf5 store.')
    p_in.add('--skiprows_integer', default='1',
             help='skip rows from top. Use 1 to skip one line of header')
    p_in.add('--b_raise_on_err', default='True',
             help='if false then not rise error on rows which can not be loaded (only shows warning). Try set "comments" argument to skip them without warning')
    p_in.add('--delimiter_chars',
             help='parameter of dask.read_csv()')
    p_in.add('--max_text_width', default='1000',
             help='maximum length of text fields (specified by "(text)" in header) for dtype in numpy loadtxt')
    p_in.add('--chunksize_percent_float',
             help='percent of 1st file length to set up hdf5 store tabe chunk size')
    p_in.add('--blocksize_int', default='20000000',
             help='bytes, chunk size for loading and processing csv')
    p_in.add('--b_make_time_inc', default='True',
             help='if time not sorted then modify time values trying to affect small number of values. This is different from sorting rows which is performed at last step after the checking table in database')
    p_in.add('--fun_date_from_filename',
             help='function(file_stem: str, century: Optional[str]=None) -> Any[compartible to input of pandas.to_datetime()]: to get date from filename to time column in it.')
    p_in.add('--fun_proc_loaded',
             help='function(df: Dataframe, cfg_in: Optional[Mapping[str, Any]] = None) -> Dataframe/DateTimeIndex: to update/calculate new parameters from loaded data  before filtering. If output is Dataframe then function should have meta_out attribute which is Callable[[np.dtype, Iterable[str], Mapping[str, dtype]], Dict[str, np.dtype]]')
    p_in.add('--csv_specific_param_dict',
             help='not default parameters for function in csv_specific_proc.py used to load data')
    p_out = p.add_argument_group('output_files', 'all about output files')
    p_out.add('--db_path', help='hdf5 store file path')
    p_out.add('--table',
              help='table name in hdf5 store to write data. If not specified then will be generated on base of path of input files. Note: "*" is used to write blocks in autonumbered locations (see dask to_hdf())')
    # p_out.add('--tables_list',
    #           help='tables names in hdf5 store to write data (comma separated)')
    p_out.add('--b_insert_separator', default='True',
              help='insert NaNs row in table after each file data end')
    p_out.add('--b_use_old_temporary_tables', default='False',
              help='Warning! Set True only if temporary storage already have good data!'
                   'if True and b_skip_if_up_to_date= True then not replace temporary storage with current storage before adding data to the temporary storage')
    p_out.add('--b_remove_duplicates', default='False', help='Set True if you see warnings about')

    p_flt = p.add_argument_group('filter', 'filter all data based on min/max of parameters')
    p_flt.add('--date_min', help='minimum time')  # todo: set to filt_min.key and filt_min.value
    p_flt.add('--date_max', help='maximum time')  # todo: set to filt_max.key and filt_max.value
    p_flt.add('--min_dict',
              help='List with items in  "key:value" format. Sets to NaN data of ``key`` columns if it is below ``value``')
    p_flt.add('--max_dict',
              help='List with items in  "key:value" format. Sets to NaN data of ``key`` columns if it is above ``value``')
    p_flt.add('--b_bad_cols_in_file_name', default='True',
              help='find string "<Separator>no_<col1>[,<col2>]..." in file name. Here <Separator> is one of -_()[, and set all values of col1[, col2] to NaN')

    p_prog = p.add_argument_group('program', 'program behaviour')
    p_prog.add('--return', default='<end>',  # nargs=1,
               choices=['<cfg_from_args>', '<gen_names_and_log>', '<end>'],
               help='<cfg_from_args>: returns cfg based on input args only and exit, <gen_names_and_log>: execute init_input_cols() and also returns fun_proc_loaded function... - see main()')
    return (p)


def init_input_cols(cfg_in=None):
    """
        Append/modify dictionary cfg_in for parameters of dask.load_csv() (or pandas.load_csv() , previously numpy.loadtxt) function and later pandas save to hdf5.
    :param cfg_in: dictionary, may has fields:
        header (required) - comma/space separated string. Column names in source file data header
        (as in Veusz standard imput dialog), used to find cfg_in['cols'] if last is not cpecified
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
        of others). Default: exluded (text) columns and index and coldate
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
    b_wNo_dtype = not 'dtype' in cfg_in
    if b_wNo_dtype:
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
        if b_wNo_dtype:
            # converters produce datetime64[ns] for coldate column (or coltime if no coldate):
            cfg_in['dtype'][cfg_in['coldate' if 'coldate' in cfg_in
            else 'coltime']] = 'datetime64[ns]'

    # process format cpecifiers: '(text)','(float)','(time)' and remove it from ['cols']
    # also find not used cols if cols assigned such as 'col1,,,col4'
    for i, s in enumerate(cfg_in['cols']):
        if len(s) == 0:
            cols_load_b[i] = 0
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
    b_index_exist = 'coltime' in cfg_in
    if b_index_exist:
        set_field_if_no(cfg_in, 'col_index_name', cfg_in['cols'][cfg_in['coltime']])
    if not 'cols_loaded_save_b' in cfg_in:
        # Automatic detection of not needed output columns
        # (text columns are not more needed after load)
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

    col_names_out = np.array(col_names_out)[cfg_in['cols_loaded_save_b']]
    cfg_in['dtype_out'] = np.dtype({
        'formats': [cfg_in['dtype'].fields[n][0] if n in cfg_in['dtype'].names else
                    np.dtype(np.float64) for n in col_names_out],
        'names': col_names_out})

    return cfg_in


def set_filterGlobal_minmax(a, cfg_filter=None, log=None, b_ok_ds=True, dict_to_save_last_time=None):
    """
    Finds bad with filterGlobal_minmax and removes it from a,tim
    Adds 'rows' remained and 'rows_filtered' to log

    :param a:
    :param cfg_filter: filtering settings, do nothing if None
    :param log: changes inplacce - adds ['rows_filtered'] and ['rows'] - number of rows remained
    :param b_ok_ds: initial mask or True that means "not filtered yet"
    :param dict_to_save_last_time: dict where 'time_last' field will be updated
    :return: number of rows remained

    """

    tim = a.index
    if isinstance(a, dd.DataFrame):
        tim = tim.compute()
    log['rows_filtered'] = 0
    # try:
    log['rows'] = len(tim)  # shape[0]
    # except IndexError:  # when?
    #     a = []
    #     return 0, None

    if cfg_filter is not None:
        meta_time = pd.Series([], name='Time', dtype=np.bool8)  # pd.Series([], name='Time',dtype='M8[ns]')

        # filterGlobal_minmax(a, tim, cfg_filter)
        if isinstance(a, dd.DataFrame):  # may be dask or not dask array
            bGood = a.map_partitions(filterGlobal_minmax, None, cfg_filter, meta=meta_time)
            range_source = a.divisions[0::a.npartitions]
            print('filtering... source range: {:%Y-%m-%d %H:%M:%S} - {:%Y-%m-%d %H:%M:%S %Z}'.format(*range_source))
            tim = a.index.compute()
            # i_starts = np.diff(np.append(tim.searchsorted(a.divisions), len(tim))).tolist()
            i_starts = [len(p) for p in bGood.partitions]
            # b_ok_da = da.from_array(b_ok, chunk=(tuple(i_starts),)).to_dask_dataframe(index=bGood.index)
            # dd.from_pandas(pd.Series(b_ok, index=bGood.index.compute().tz_convert('UTC')),
            #                            npartitions=bGood.npartitions)

            b_ok = (bGood if b_ok_ds is True else bGood.mask(~b_ok_ds, False)).persist()
            a = a.loc[b_ok]
            b_ok = b_ok.values.compute()
            tim = tim[b_ok]  # a.index.compute()
            sum_good = b_ok.sum()  # b.sum().compute()
        else:
            b_ok = filterGlobal_minmax(a, tim, cfg_filter, b_ok_ds)  # b_ok_ds.values.compute()?
            sum_good = np.sum(b_ok)

        if sum_good < log['rows'] and not np.isscalar(
                b_ok):  # <=> b_ok.any() and b_ok is not scalar True (True is if not need filter)
            # tim = tim[b_ok]
            #     #a = a.merge(pd.DataFrame(index=np.flatnonzero(b_ok)))  # a = a.compute(); a[b_ok]
            #
            # a = a.join(pd.DataFrame(index=tim), how='right')
            #     tim= tim[b_ok.values] #.iloc

            log['rows_filtered'] = log['rows'] - sum_good
            log['rows'] = sum_good

    # Save last time to can filter next file
    if dict_to_save_last_time:
        try:
            dict_to_save_last_time['time_last'] = tim[-1]
        except IndexError:
            l.warning('no data!')
    return a, tim


def filter_local_with_file_name_settings(d: Union[pd.DataFrame, dd.DataFrame],
                                         cfg: Mapping[str, Any],
                                         path_csv: PurePath) -> Union[pd.DataFrame, dd.DataFrame]:
    """
    Set all data in columns to NaN if file name has string "{separator}no_{Name1[, Name2...]}"
    where:
        separator is one of "-_,;([" sybmols
        names Name1... mached to data column names. Except "Ox" - this abbreviaton mean "O2, O2ppm"

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

def read_csv(nameFull: Sequence[Union[str, Path]],
             cfg_in: Mapping[str, Any]) -> Tuple[Union[pd.DataFrame, dd.DataFrame], dd.Series]:
    """
    Reads csv in dask DataFrame
    Calls cfg_in['fun_proc_loaded'] (if specified)
    Calls time_corr: corrects/checks Time (with arguments defined in cfg_in fields)
    Sets Time as index
    :param nameFull: list of file names
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
        b_ok:   time correction reszult bulean array
    """
    try:
        try:
            # raise ValueError('Temporary')
            ddf = dd.read_csv(
                nameFull, dtype=cfg_in['dtype_raw'], names=cfg_in['cols'],
                delimiter=cfg_in['delimiter'], skipinitialspace=True, usecols=cfg_in['dtype'].names,
                # cfg_in['cols_load'],
                converters=cfg_in['converters'], skiprows=cfg_in['skiprows'],
                error_bad_lines=cfg_in['b_raise_on_err'], comment=cfg_in['comments'],
                header=None, blocksize=cfg_in['blocksize'])  # not infer

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
            l.error('dask lib can not load data {}: {}. Trying pandas lib...'.format(e.__class__, '\n==> '.join(
                [m for m in e.args if isinstance(m, str)])))
            for i, nf in enumerate(nameFull):
                df = pd.read_csv(
                    nf, dtype=cfg_in['dtype_raw'], names=cfg_in['cols'], usecols=cfg_in['dtype'].names,
                    # cfg_in['cols_load'],
                    delimiter=cfg_in['delimiter'], skipinitialspace=True, index_col=False,
                    converters=cfg_in['converters'], skiprows=cfg_in['skiprows'],
                    error_bad_lines=cfg_in['b_raise_on_err'], comment=cfg_in['comments'],
                    header=None)
                if i > 0:
                    raise NotImplementedError('list of files => need concatenate data')
            ddf = dd.from_pandas(df, chunksize=cfg_in['blocksize'])  #
    except Exception as e:  # for example NotImplementedError if bad file
        msg = '{}: {} - Bad file. skip!\n'.format(e.__class__, '\n==> '.join([
            m for m in e.args if isinstance(m, str)]))
        ddf = None
        if cfg_in['b_raise_on_err']:
            l.error(msg + '%s\n Try set [in].b_raise_on_err= False\n', e)
            raise (e)
        else:
            l.exception(msg)
    if __debug__:
        l.debug('read_csv initialised')
    if ddf is None:
        return None, None

    meta_time = pd.Series([], name='Time', dtype='M8[ns]')  # np.dtype('datetime64[ns]')
    meta_time_index = pd.DatetimeIndex([], dtype='datetime64[ns]', name='Time')
    meta_df_with_time_col = cfg_in['cols_load']

    # Process ddf and get date in ISO string or numpy standard format
    cfg_in['file_stem'] = Path(nameFull[0]).stem  # may be need in func below to extract date
    try:
        date_delayed = None
        try:
            if not getattr(cfg_in['fun_proc_loaded'], 'meta_out', None) is None:
                # fun_proc_loaded will return modified data. Go to catch it
                # todo: find better condition
                raise TypeError

            # ddf_len = len(ddf)
            # counts_divisions = list(range(1, int(ddf_len / cfg_in.get('decimate_rate', 1)), cfg_in['blocksize']))
            # counts_divisions.append(ddf_len)
            #
            # date_delayed = delayed(cfg_in['fun_proc_loaded'], nout=1)(ddf, cfg_in)
            # date = dd.from_delayed(date_delayed, meta=meta_time_index, divisions=ddf.index.divisions)
            # date = dd.from_dask_array(date.values, index=ddf.index)

            date = ddf.map_partitions(lambda *args, **kwargs: pd.Series(
                cfg_in['fun_proc_loaded'](*args, **kwargs)), cfg_in, meta=meta_time)  # meta_time_index
            # date = date.to_series()
        except (TypeError, Exception) as e:
            # fun_proc_loaded retuns tuple (date, a)
            changing_size = False  # ? True  # ?
            if changing_size:
                date_delayed, a = delayed(cfg_in['fun_proc_loaded'], nout=2)(ddf, cfg_in)
                # if isinstance(date, tuple):
                #     date, a = date
                # if isinstance(a, pd.DataFrame):
                #     a_is_dask_df = False
                # else:chunksize=cfg_in['blocksize']
                ddf_len = len(ddf)
                counts_divisions = list(range(1, int(ddf_len / cfg_in.get('decimate_rate', 1)), cfg_in['blocksize']))
                counts_divisions.append(ddf_len)
                ddf = dd.from_delayed(a, divisions=(0, counts_divisions))
                date = dd.from_delayed(date_delayed, meta=meta_time_index, divisions=counts_divisions)
                date = dd.from_dask_array(date.values, index=ddf.index)
                # date = dd.from_pandas(date.to_series(index=), chunksize=cfg_in['blocksize'], )
                # _pandas(date, chunksize=cfg_in['blocksize'], name='Time')
            else:  # getting df with time col
                meta_out = cfg_in['fun_proc_loaded'].meta_out(cfg_in['dtype']) if callable(
                    cfg_in['fun_proc_loaded'].meta_out) else None
                ddf = ddf.map_partitions(cfg_in['fun_proc_loaded'], cfg_in, meta=meta_out)
                date = ddf.Time
    except IndexError:
        print('no data?')
        return None, None
        # add time shift specified in configuration .ini

    n_overlap = 2 * int(np.ceil(cfg_in['fs'])) if cfg_in.get('fs') else 50
    # reset_index().set_index('index').
    meta2 = {'Time': 'M8[ns]', 'b_ok': np.bool8}

    #     pd.DataFrame(columns=('Time', 'b_ok'))
    # meta2.time = meta2.time.astype('M8[ns]')
    # meta2.b_ok = meta2.b_ok.astype(np.bool8)

    def time_corr_df(t, cfg_in):
        """convert tuple returned by time_corr() to dataframe"""
        return pd.DataFrame.from_dict(OrderedDict(zip(meta2.keys(), time_corr(t, cfg_in))))
        # return pd.DataFrame.from_items(zip(meta2.keys(), time_corr(t, cfg_in)))
        # pd.Series()

    # date.rename('time').to_series().reset_index().compute()
    # date.to_series().repartition(divisions=ddf.divisions[1])

    '''
    def time_corr_ar(t, cfg_in):
        """convert tuple returned by time_corr() to dataframe"""
        return np.array(time_corr(t, cfg_in))
        #return pd.DataFrame.from_items(zip(meta2.keys(), time_corr(t, cfg_in)))
        # pd.Series()
    da.overlap.map_overlap(date.values, time_corr_ar, depth=n_overlap)
    '''

    l.info('time correction in %s blocks...', date.npartitions)
    df_time_ok = date.map_overlap(time_corr_df, before=n_overlap, after=n_overlap, cfg_in=cfg_in, meta=meta2)
    # .to_series()
    # if __debug__:
    #     c = df_time_ok.compute()
    # tim = date.compute().get_values()
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
    try:
        df_time_ok = df_time_ok.persist()

    except Exception as e:
        l.exception('Can not speed up by persist')
        # # something that can trigger error to help it identificate ???
        # date = date.persist()
        # df_time_ok = df_time_ok.compute()
        df_time_ok = time_corr_df(
            (date_delayed if date_delayed is not None else date).compute(), cfg_in=cfg_in)
        # raise pass

    # df_time_ok.compute(scheduler='single-threaded')
    if isinstance(df_time_ok, dd.DataFrame):
        nbad_time = len(df_time_ok['b_ok']) - df_time_ok['b_ok'].sum().compute()
        if nbad_time:
            nonzero = []
            for b_ok in df_time_ok['b_ok'].ne(True).partitions:  # fillna(0).
                b = b_ok.compute()
                if b.any():
                    nonzero.extend(b.to_numpy().nonzero()[0])
        l.info('Bad time values (%d): %s%s', nbad_time, nonzero[:20], ' (shows first 20)' if nbad_time > 20 else '')

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
               df_time_ok['b_ok'].fillna(0).ne(True).to_numpy().nonzero()[0][:20],
               ' (shows first 20)' if nbad_time > 20 else '')

        # df_time_ok.loc[df_time_ok['b_ok'], 'Time'] = pd.NaT
        # try:
        #     df_time_ok.Time = df_time_ok.Time.interpolate(inplace=False)                 # inplace=True - not works, method='linear', - default
        # except ValueError:  # if not interpolates (my condition) use simpler method:
        #     df_time_ok.Time = df_time_ok.Time.fillna(method='ffill', inplace=True)

        # # dask get IndexingError: Unalignable boolean Series provided as indexer (index of the boolean Series and of the indexed object do not match):
        # ddf_out = ddf.loc[df_time_ok['b_ok'], list(cfg_in['dtype_out'].names)].set_index(
        #    df_time_ok.loc[df_time_ok['b_ok'], 'Time'], sorted=True)
    # so we have done interpolate that helps this:
    ddf_out = ddf.loc[:, list(cfg_in['dtype_out'].names)].set_index(df_time_ok['Time'], sorted=True)  #
    # print('data loaded shape: {}'.format(ddf.compute(scheduler='single-threaded').shape))  # debug only

    if nbad_time and cfg_in.get('keep_input_nans'):
        df_time_ok = df_time_ok.set_index('Time', sorted=True)
        # ??? after I set index: ValueError: Not all divisions are known, can't align partitions. Please use `set_index` to set the index.
        ddf_out = ddf_out.loc[df_time_ok['b_ok'], :]

    # if isinstance(df_time_ok, dd.DataFrame) else df_time_ok['Time'].compute()
    # **({'sorted': True} if a_is_dask_df else {}
    # [cfg_in['cols_load']]
    # else:
    #     col_temp = ddf.columns[0]
    #     b = ddf[col_temp]
    #     b[col_temp] = b[col_temp].map_partitions(lambda s, t: t[s.index], tim, meta=meta)
    #     ddf = ddf.reset_index().set_index('index').set_index(b[col_temp], sorted=True).loc[:, list(cfg_in['dtype_out'].names)]

    # date = pd.Series(tim, index=ddf.index.compute())  # dd.from_dask_array(da.from_array(tim.get_values(),chunks=ddf.divisions), 'Time', index=ddf.index)
    # date = dd.from_pandas(date, npartitions=npartitions)
    # ddf = ddf.loc[:, list(cfg_in['dtype_out'].names)].set_index(date, sorted=True)

    # ddf = ddf.loc[:, list(cfg_in['dtype_out'].names)].compute()
    # ddf.set_index(tim, inplace=True)
    # ddf = dd.from_pandas(ddf, npartitions=npartitions)

    logger = logging.getLogger("dask")
    logger.addFilter(lambda s: s.getMessage() != "Partition indices have overlap.")
    # b_ok = df_time_ok['b_ok'].to_dask_array().compute() if isinstance(
    #     df_time_ok, dd.DataFrame) else df_time_ok['b_ok'].to_numpy()

    # b_ok_ds= df_time_ok.set_index('Time')['b_ok']
    return ddf_out  # , b_ok_ds


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
#             print('{}: {} - Bad file. skip!\n'.format(e.__class__, '\n==> '.join([
#                 a for a in e.args if isinstance(a, str)])))
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
            print('{}: {} - skip logging operations!\n'.format(
                e.__class__, '\n==> '.join([a for a in e.args if isinstance(a, str)])))
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


def h5_names_gen(cfg: Mapping[str, Any], cfg_out: Mapping[str, Any]) -> Iterator[Path]:
    """
    Yields Paths from cfg['in']['namesFull'] items
    :updates: cfg_out['log'] fields 'fileName' and 'fileChangeTime'
    If can open cfg['program']['log'] then writes to it:
        - header: current date, cfg['in']['nfiles']
        - status: cfg_out's 'fileName', 'Date0', 'DateEnd', 'rows'

    :param cfg: dict of dicts:
        in: dict, must have fields:
            'namesFull', iterator - returns full file names
            'nfiles', int - number of file names will be generated
        program: dict, must have fields:
            'log', str file name of additional metadata log file about loaded data
    :param cfg_out: dict, need have field 'log', dict with fields needed for h5_dispenser_and_names_gen() and print info:
        'Date0', 'DateEnd', 'rows' - must have (should be updated) after yield
        'log': current file info - else prints "file not processed"
    """
    set_field_if_no(cfg_out, 'log', {})
    with open_if_can(cfg['program']['log']) as flog:
        if flog:
            flog.writelines(datetime.now().strftime(
                '\n\n%d.%m.%Y %H:%M:%S> processed ' + str(cfg['in']['nfiles']) + ' file' + 's:' if cfg['in'][
                                                                                                       'nfiles'] > 1 else ':'))

        for name_full in cfg['in']['namesFull']:
            pname = Path(name_full)

            cfg_out['log']['fileName'] = pname.name[-cfg_out['logfield_fileName_len']:-4]
            cfg_out['log']['fileChangeTime'] = datetime.fromtimestamp(pname.stat().st_mtime)
            # os_path.getmtime(
            yield pname

            # Log to logfile
            if cfg_out['log'].get('Date0'):
                strLog = '{fileName}:\t{Date0:%d.%m.%Y %H:%M:%S}-{DateEnd:%d. %H:%M:%S%z}\t{rows}rows'.format(
                    **cfg_out['log'])  # \t{Lat}\t{Lon}\t{strOldVal}->\t{mag}
            else:
                strLog = "file not processed"
            print(strLog)
            if flog:
                flog.writelines('\n' + strLog)  # + nameFE + '\t' +


def h5_dispenser_and_names_gen(cfg: Mapping[str, Any],
                               cfg_out: Optional[Mapping] = None,
                               fun_gen: Callable[[Mapping[str, Any], Mapping[str, Any]], Iterator[Any]] = h5_names_gen
                               ) -> Iterator[Tuple[int, Any]]:
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

    :param cfg: dict, must have fields:
        - program: dict, must have field ``log``, str: file name of additional metadata log file about loaded data
        - fields used in your fun_gen(cfg, cfg_out)
    :param cfg_out: dict, must have fields
        - log: current file info
        - b_skip_if_up_to_date: if True then skips procesed files. But if file was changed removes stored data
        and nevetherless yields fun_gen(...) result
        - tables_have_wrote: sequence of table names where to create index
    :param fun_gen: function with arguments (cfg, cfg_out), that
        - generates data labels, default are file's ``Path``s,
        - updates cfg_out['log'] fields 'fileName' (by current label) and 'fileChangeTime' needed to store and find
        data. They named historically, in priciple, you can use any unique idetnificator composed of this two fields.
    :return: Iterator that returns (i1, pname):
        - i1: index (starting with 1) of fun_gen generated data label (may be file)
        - pname: fun_gen output (may be path name)
        Skips (i1, pname) for existed labels that also has same stored data label (file) modification date
    :updates:
        - cfg_out['db'],
        - cfg_out['b_remove_duplicates'] and
        - that what fun_gen() do
    """

    if cfg_out is None:
        cfg_out = cfg['output_files']

    dfLogOld = h5temp_open(cfg_out)
    try:
        for i1, gen_out in enumerate(fun_gen(cfg, cfg_out), start=1):
            # remove stored data and process file if it was changed
            if cfg_out['b_skip_if_up_to_date']:
                bExistOk, bExistDup = h5del_obsolete(cfg_out, cfg_out['log'], dfLogOld)
                if bExistOk:
                    continue
                if bExistDup:
                    cfg_out['b_remove_duplicates'] = True  # normally no duplicates but will if detect

            yield i1, gen_out

    except Exception as e:
        l.exception('\nError preparing data: ' + str(e.__class__) + ':\n==> '.join(
            [s for s in e.args if isinstance(s, str)]))
    finally:
        try:
            print('')
            cfg_table_keys = ['tables_have_wrote'] if ('tables_have_wrote' in cfg_out) else ('tables', 'tables_log')
            if cfg_out['b_remove_duplicates']:
                h5remove_duplicates(cfg_out, cfg_table_keys=cfg_table_keys)
            create_indexes(cfg_out, cfg_table_keys)
        except Exception as e:
            l.error('\nError of adding data to temporary store: ' + str(e.__class__) + ':\n==> '.join(
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
            if cfg_out['db'].is_open:
                print('Wait store closing...')
                sleep(2)
            cfg_out['db'] = None
            return


def get_fun_proc_loaded_converters(cfg_in: Mapping[str, Any]):
    """
    Assign castom prep&proc based on args.cfgFile name
    :param cfg_in:
    :return fun_proc_loaded: Callable if cfgFile name match found
        None if cfg_in['fun_proc_loaded'] not specifed
        cfg_in['fun_proc_loaded'] if
    """
    import to_pandas_hdf5.csv_specific_proc

    fun_proc_loaded = cfg_in.get('fun_proc_loaded')
    if fun_proc_loaded:
        return fun_proc_loaded
    cfg_file = Path(cfg_in['cfgFile']).stem
    if cfg_file.endswith('Sea&Sun'):
        fun_proc_loaded = to_pandas_hdf5.csv_specific_proc.proc_loaded_sea_and_sun
    elif cfg_file.endswith('Idronaut'):
        fun_proc_loaded = to_pandas_hdf5.csv_specific_proc.proc_loaded_Idronaut
    elif cfg_file.endswith('IdrRedas'):
        cfg_in['converters'] = {cfg_in['coltime']: lambda txtD_M_YYYY_hhmmssf:
        np.datetime64(b'%(2)b-%(1)b-%(0)bT%(3)b' % dict(
            zip([b'0', b'1', b'2', b'3'], (txtD_M_YYYY_hhmmssf[:19].replace(b' ', b'/').split(b'/')))))}
        # b'{2}-{1}-{0}T{3}' % (txtD_M_YYYY_hhmmssf[:19].replace(b' ',b'/').split(b'/')))} #len=19 because bug of bad milliseconds
        # fun_proc_loaded= proc_loaded_IdrRedas
    elif cfg_file.endswith('nav_supervisor') or cfg_file.endswith('meteo'):
        fun_proc_loaded = to_pandas_hdf5.csv_specific_proc.proc_loaded_nav_supervisor
    elif cfg_file.endswith('CTD_Schuka'):
        fun_proc_loaded = to_pandas_hdf5.csv_specific_proc.proc_loaded_CTD_Schuka
    elif cfg_file.endswith('CTD_Schuka_HHMM'):
        fun_proc_loaded = to_pandas_hdf5.csv_specific_proc.proc_loaded_CTD_Schuka_HHMM
    elif cfg_file.endswith('csv_log'):
        fun_proc_loaded = to_pandas_hdf5.csv_specific_proc.proc_loaded_csv_log
    elif cfg_file.endswith('ISO_time'):
        # more prepare for time in standard ISO 8601 format
        cfg_in['converters'] = {cfg_in['coltime']: lambda txtYY_M_D_h_m_s_f: np.array(
            '20{0:02.0f}-{1:02.0f}-{2:02.0f}T{3:02.0f}:{4:02.0f}:{5:02.0f}.{6:02.0f}0'.format(
                *np.array(np.fromstring(txtYY_M_D_h_m_s_f, dtype=np.uint8, sep=','), dtype=np.uint8)),
            dtype='datetime64[ns]')}  # - np.datetime64('2009-01-01T00:00:00', dtype='datetime64[ns]')
    elif cfg_file.endswith('Baranov_chain') or cfg_file.endswith('Baranov_inclin'):
        fun_proc_loaded = to_pandas_hdf5.csv_specific_proc.proc_loaded_Baranov_chain
    elif cfg_file.endswith('csv_Baklan'):
        fun_proc_loaded = to_pandas_hdf5.csv_specific_proc.proc_loaded_Baklan
    elif cfg_file.endswith('Kondrashov_inclin'):
        fun_proc_loaded = to_pandas_hdf5.csv_specific_proc.proc_loaded_Kondrashov_inclin
    elif cfg_file.endswith('nav_HYPACK'):
        fun_proc_loaded = to_pandas_hdf5.csv_specific_proc.proc_loaded_nav_HYPACK
    return fun_proc_loaded


def main(new_arg=None, **kwargs):
    """

    :param new_arg: list of strings, command line arguments
    :kwargs: dicts for each section: to overwrite values in them (overwrites even high priority values, other values remains)
    Note: if new_arg=='<cfg_from_args>' returns cfg but it will be None if argument
     argv[1:] == '-h' or '-v' passed to this code
    argv[1] is cfgFile. It was used with cfg files:
        'csv2h5_nav_supervisor.ini'
        'csv2h5_IdrRedas.ini'
        'csv2h5_Idronaut.ini'

    :return:
    """

    global l

    cfg = cfg_from_args(my_argparser(), new_arg, **kwargs)
    if not cfg or not cfg['program'].get('return'):
        print('Can not initialise')
        return cfg
    elif cfg['program']['return'] == '<cfg_from_args>':  # to help testing
        return cfg

    l = init_logging(logging, None, cfg['program']['log'], cfg['program']['verbose'])
    print('\n' + this_prog_basename(__file__), end=' started. ')
    try:
        cfg['in'] = init_file_names(cfg['in'], cfg['program']['b_interact'])
    except Ex_nothing_done as e:
        print(e.message)
        return ()

    # Prepare loading and writing cpecific to format
    cfg['in']['fun_proc_loaded'] = get_fun_proc_loaded_converters(cfg['in'])
    cfg['in'] = init_input_cols(cfg['in'])
    # cfg['output_files']['dtype'] = cfg['in']['dtype_out']
    cfg_out = cfg['output_files'];
    h5init(cfg['in'], cfg_out)
    # Default time postload proc
    if cfg['in']['fun_proc_loaded'] is None:
        if 'coldate' not in cfg['in']:  # Time includes Date
            cfg['in']['fun_proc_loaded'] = delayed(lambda a, cfg_in: a[cfg_in['col_index_name']])
        else:  # Time + Date
            cfg['in']['fun_proc_loaded'] = delayed(lambda a, cfg_in: a['Date'] + np.array(
                np.int32(1000 * a[cfg_in['col_index_name']]), dtype='m8[ms]'))

    if 'csv_specific_param' in cfg['in']:
        t = getattr(cfg['in']['fun_proc_loaded'], 'meta_out', None)
        cfg['in']['fun_proc_loaded'] = partial(cfg['in']['fun_proc_loaded'],
                                               csv_specific_param=cfg['in']['csv_specific_param'])
        if t is not None:
            cfg['in']['fun_proc_loaded'].meta_out = t

    if cfg['program']['return'] == '<return_cfg_step_fun_proc_loaded>':  # to help testing
        return cfg

    # to insert separator lines:
    df_dummy = pd.DataFrame(np.full(
        1, np.NaN, dtype=cfg['in']['dtype_out']), index=(pd.NaT,))
    cfg_out['log'] = {'fileName': None, 'fileChangeTime': None}
    # log= np.array(([], [],'0',0), dtype= [('Date0', 'O'), ('DateEnd', 'O'),
    # ('fileName', 'S255'), ('rows', '<u4')])

    if cfg['program']['return'] == '<gen_names_and_log>':  # to help testing
        cfg['in']['gen_names_and_log'] = h5_dispenser_and_names_gen
        cfg['output_files'] = cfg_out
        return cfg

    # Writing
    if True:  # try:

        ## Main circle ############################################################
        for i1_file, path_csv in h5_dispenser_and_names_gen(cfg, cfg_out):
            l.info('{}. {}: '.format(i1_file, path_csv.name))
            # Loading and processing data
            d = read_csv([path_csv], cfg['in'])  # , b_ok_ds
            if d is None:
                l.warning('not processing')
                continue

            # filter
            d, tim = set_filterGlobal_minmax(d, cfg_filter=cfg['filter'], log=cfg_out['log'],
                                             dict_to_save_last_time=cfg['in'])  # b_ok_ds=b_ok_ds,

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

    if cfg['in'].get(
            'time_last'):  # if have any processed data (needed because ``ptprepack`` not closses hdf5 source if it not finds data)
        new_storage_names = h5move_tables(cfg_out)
        print('Ok.', end=' ')
        h5index_sort(cfg_out, out_storage_name=cfg_out['db_base'] + '-resorted.h5', in_storages=new_storage_names)


if __name__ == '__main__':
    main()

""" trash ##############################################

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