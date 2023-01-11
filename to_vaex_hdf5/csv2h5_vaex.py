#!/usr/bin/env python3
# coding:utf-8
"""
  Author:  Andrey Korzh <ao.korzh@gmail.com>
  Purpose: Convert (multiple) csv and alike text files to vaex hdf5 store with
           addition of log table
  Created: 26.02.2016
  Modified: 20.12.2019
"""
import logging
import re
import warnings
from codecs import open
from collections import OrderedDict
from datetime import datetime
from functools import partial
from pathlib import Path, PurePath
from time import sleep
from typing import Any, Callable, Iterator, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import vaex

from utils2init import init_file_names, Ex_nothing_done, set_field_if_no, cfg_from_args, my_argparser_common_part, \
    this_prog_basename, init_logging, standard_error_info
import utils_time_corr

if __name__ == '__main__':
    l = None  # see main(): l = init_logging(logging, None, cfg['program']['log'], cfg['program']['verbose'])
else:
    l = logging.getLogger(__name__)

VERSION = '0.0.1'

def cmdline_help_mod(version, info):
    'csv2h5 version {}'.format(version) + info

def argparser_files(
        # "in" default parameters
        path='.', *,
        b_search_in_subdirs=False,
        exclude_dirs_endswith_list='toDel, -, bad, test, TEST',
        exclude_files_endswith_list='coef.txt, -.txt, test.txt',
        b_incremental_update=True,
        dt_from_utc_seconds=0,
        dt_from_utc_hours=0,
        skiprows_integer=1,
        on_bad_lines='error',
        max_text_width=1000,
        blocksize_int=20000000,
        sort=True,
        # "out"
        b_insert_separator=True,
        b_reuse_temporary_tables=False,
        b_remove_duplicates=False,
        # "filter" default parameters
        b_bad_cols_in_file_name=False,
        **kwargs
        ):
    """
    ----------------------------
    Add data from CSV-like files
    to Pandas HDF5 store*.h5
    ----------------------------

    "in": all about input files:
    
    :param path: path to source file(s) to parse. Use patterns in Unix shell style
    :param b_search_in_subdirs: search in subdirectories, used if mask or only dir in path (not full path)
    :param exclude_dirs_endswith_list: exclude dirs which ends with this srings. This and next option especially useful when search recursively in many dirs
    :param exclude_files_endswith_list: exclude files which ends with this srings
    :param b_incremental_update: exclude processing of files with same name and which time change is not bigger than recorded in database (only prints ">" if detected). If finds updated version of same file then deletes all data which corresponds old file and after it brfore procesing of next files
    :param dt_from_utc_seconds: add this correction to loading datetime data. Can use other suffixes instead of "seconds"
    :param dt_from_utc_hours: add this correction to loading datetime data. Can use other suffixes instead of "hours"
    :param fs_float: sampling frequency, uses this value to calculate intermediate time values between time changed values (if same time is assigned to consecutive data)
    :param fs_old_method_float: sampling frequency, same as ``fs_float``, but courses the program to use other method. If smaller than mean data frequency then part of data can be deleted!(?)
    :param header: comma separated list matched to input data columns to name variables. Can contain type suffix i.e.
     (float) - which is default, (text) - also to convert by specific converter, or (time) - for ISO format only
    :param cols_load_list: comma separated list of names from header to be saved in hdf5 store. Do not use "/" char, or type suffixes like in ``header`` for them. Defaut - all columns
    :param cols_not_save_list: comma separated list of names from header to not be saved in hdf5 store
    :param skiprows_integer: skip rows from top. Use 1 to skip one line of header
    :param on_bad_lines: choices=['error', 'warn', 'skip'],
        "warn" print a warning when a bad line is encountered and skip that line. See also "comment" argument to skip bad line without warning
    :param delimiter_chars: parameter of pandas.read_csv()
    :param max_text_width: maximum length of text fields (specified by "(text)" in header) for dtype in numpy loadtxt
    :param chunksize_percent_float: percent of 1st file length to set up hdf5 store tabe chunk size
    :param blocksize_int: bytes, chunk size for loading and processing csv
    :param sort: if time not sorted then modify time values trying to affect small number of values. This is different from sorting rows which is performed at last step after the checking table in database
    :param fun_date_from_filename: function(file_stem: str, century: Optional[str]=None) -> Any[compartible to input of pandas.to_datetime()]: to get date from filename to time column in it.

    :param csv_specific_param_dict: not default parameters for function in csv_specific_proc.py used to load data


    "out": all about output files:
    
    :param db_path: hdf5 store file path
    :param table: table name in hdf5 store to write data. If not specified then will be generated on base of path of input files. Note: "*" is used to write blocks in autonumbered locations (see dask to_hdf())
    :param b_insert_separator: insert NaNs row in table after each file data end
    :param b_reuse_temporary_tables: Warning! Set True only if temporary storage already have good data! If True and b_incremental_update= True then not replace temporary storage with current storage before adding data to the temporary storage
    :param b_remove_duplicates: Set True if you see warnings about

    "filter": filter all data based on min/max of parameters:

    :param min_dict: List with items in  "key:value" format. Sets to NaN data of ``key`` columns if it is below ``value``'). To filter time use ``date`` key
    :param max_dict: List with items in  "key:value" format. Sets to NaN data of ``key`` columns if it is above ``value``'). To filter time use ``date`` key
    :param b_bad_cols_in_file_name: find string "<Separator>no_<col1>[,<col2>]..." in file name. Here <Separator> is one of -_()[, and set all values of col1[, col2] to NaN

    "program": program behaviour:

    :param return_: choices=[],
        <cfg_from_args>: returns cfg based on input args only and exit,
        <gen_names_and_log>: execute init_input_cols() and also returns fun_proc_loaded function... - see main()
    """
    return


def init_input_cols(*, header=None, dtype, converters=None, cols_load, max_text_width=2000, dt_from_utc=0, comment='"',
                    cols_loaded_save_b=None):
    """ Append/modify dictionary cfg_in for parameters of dask/pandas load_csv() function and of save to hdf5.
    :param header (required if no 'cols'): comma/space separated string, column names in source file data header. Used to find cfg_in['cols']
         if last is not cpecified. May have format cpecifiers: '(text)','(float)','(time)', and also not used cols
         cpecified by skipping name between commas like in 'col1,,,col4' as in Veusz standard input dialog.
    :param dtype: type of data in column (as in Numpy loadtxt)
    :param converters: dict (see "converters" in Numpy loadtxt) or function(cfg_in) to make dict here
    :param cols_load: list of used column names

    :return: modified cfg_in dictionary. Will have fields:
        cols - list constructed from header by spit and remove format cpecifiers: '(text)', '(float)', '(time)'
        cols_load - list[int], indexes of ``cols`` in needed to save order
        coltime/coldate - index of 'Time'/'Date' column
        dtype: numpy.dtype of data after using loading function but before filtering/calculating fields
            numpy.float64 - default and for '(float)' format specifier
            numpy string with length cfg_in['max_text_width'] - for '(text)'
            datetime64[ns] - for coldate column (or coltime if no coldate) and for '(time)'
        col_index_name - index name for saving Pandas frame. Will be set to name of cfg_in['coltime'] column if not exist already
        used in main() default time postload proc only (if no specific loader which calculates and returns time column for index)
        cols_loaded_save_b - columns mask of cols_load to save (some columns needed only before save to calculate
        of others). Default: exluded (text) columns and index and coldate
        (because index saved in other variable and coldate may only used to create it)

    Example
    -------
    header= u'`Ensemble #`,txtYY_M_D_h_m_s_f(text),,,Top,`Average Heading (degrees)`,`Average Pitch (degrees)`,stdPitch,`Average Roll (degrees)`,stdRoll,`Average Temp (degrees C)`,txtu_none(text) txtv_none(text) txtVup(text) txtErrVhor(text) txtInt1(text) txtInt2(text) txtInt3(text) txtInt4(text) txtCor1(text) txtCor2(text) txtCor3(text) txtCor4(text),,,SpeedE_BT SpeedN_BT SpeedUp ErrSpeed DepthReading `Bin Size (m)` `Bin 1 Distance(m;>0=up;<0=down)` absorption IntScale'.strip()
    """
    cfg_in = locals()   # must be 1st row in function to be dict of input args
    dtype_text_max = '|S{:.0f}'.format(max_text_width)  # np.str

    if header:  # if header specified
        re_sep = ' *(?:(?:,\n)|[\n,]) *'  # not isolate "`" but process ",," right
        cfg_in['cols'] = re.split(re_sep, header)
        # re_fast = re.compile(u"(?:[ \n,]+[ \n]*|^)(`[^`]+`|[^`,\n ]*)", re.VERBOSE)
        # cfg_in['cols']= re_fast.findall(cfg_in['header'])
    elif not 'cols' in cfg_in:  # cols is from header, is specified or is default
        warnings.warn("default 'cols' is deprecated, use init_input_cols({header: "
                      "'stime, latitude, longitude'}) instead", DeprecationWarning, 2)
        cfg_in['cols'] = ('stime', 'latitude', 'longitude')

    # default parameters dependent from ['cols']
    cols_load_b = np.ones(len(cfg_in['cols']), np.bool8)

    # assign data type of input columns
    b_was_no_dtype = not 'dtype' in cfg_in
    if b_was_no_dtype:
        cfg_in['dtype'] = np.array([np.float64] * len(cfg_in['cols']))
        # 32 gets trunkation errors after 6th sign (=> shows long numbers after dot)
    elif isinstance(cfg_in['dtype'], str):
        cfg_in['dtype'] = np.array([np.dtype(cfg_in['dtype'])] * len(cfg_in['cols']))
    elif isinstance(cfg_in['dtype'], list):
        # prevent numpy array(list) guess minimal dtype because otherwise dtype will take maximum memory of length dtype_text_max
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

    if cfg_in['converters']:
        if not isinstance(cfg_in['converters'], dict):
            # suspended evaluation required
            cfg_in['converters'] = cfg_in['converters'](cfg_in)
        if b_was_no_dtype:
            # converters produce datetime64[ns] for coldate column (or coltime if no coldate):
            cfg_in['dtype'][cfg_in.get('coldate', cfg_in['coltime'])] = 'datetime64[ns]'

    # process format cpecifiers: '(text)','(float)','(time)' and remove it from ['cols'],
    # also find not used cols cpecified by skipping name between commas like in 'col1,,,col4'
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
    if 'cols_not_save' in cfg_in:
        cols_load_in_used_b = np.isin(cfg_in['cols_load'], cfg_in['cols_not_save'], invert=True)
        if not np.all(cols_load_in_used_b):
            cfg_in['cols_load'] = cfg_in['cols_load'][cols_load_in_used_b]
            cols_load_b = np.isin(cfg_in['cols'], cfg_in['cols_load'])

    col_names_out = cfg_in['cols_load'].copy()
    # Convert ``cols_load`` to index (to be compatible with numpy loadtxt()), names will be in cfg_in['dtype'].names
    cfg_in['cols_load'] = np.int32([cfg_in['cols'].index(c) for c in cfg_in['cols_load'] if c in cfg_in['cols']])
    # not_cols_load = np.array([n in cfg_in['cols_not_save'] for n in cfg_in['cols']], np.bool)
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


    # Output columns mask
    if cfg_in['cols_loaded_save_b'] is None:
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
    col_names_out = np.array(col_names_out)[cfg_in['cols_loaded_save_b']].tolist() + cfg_in['cols_save']
    cfg_in['dtype_out'] = np.dtype({
        'formats': [cfg_in['dtype'].fields[n][0] if n in cfg_in['dtype'].names else
                    np.dtype(np.float64) for n in col_names_out],
        'names': col_names_out})

    return cfg_in



def read_csv(paths: Sequence[Union[str, Path]], cfg_in: Mapping[str, Any]) -> Union[pd.DataFrame, vaex.dataframe.DataFrame]:
    """
    Reads csv in dask DataFrame
    Calls cfg_in['fun_proc_loaded'] (if specified)
    Calls time_corr: corrects/checks Time (with arguments defined in cfg_in fields)
    Sets Time as index
    :param paths: list of file names
    :param cfg_in: contains fields for arguments of dask.read_csv correspondence:

        names=cfg_in['cols'][cfg_in['cols_load']]
        usecols=cfg_in['cols_load']
        on_bad_lines=cfg_in['on_bad_lines']
        comment=cfg_in['comment']

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
            # for ichunk, chunk in enumerate(pd.read_csv(paths, chunksize=1000, delimiter='\t')):
            df = pd.read_csv(
                paths,
                dtype=cfg_in['dtype_raw'],
                names=cfg_in['cols'],
                delimiter=cfg_in['delimiter'],
                skipinitialspace=True,
                usecols=cfg_in['dtype'].names,
                # cfg_in['cols_load'],
                converters=cfg_in['converters'],
                skiprows=cfg_in['skiprows'],
                on_bad_lines=cfg_in['on_bad_lines'],
                comment=cfg_in['comment'],
                header=None,
                blocksize=cfg_in['blocksize'])  # not infer

            # , engine='python' - may help load bad file

            # index_col=False  # force pandas to _not_ use the first column as the index (row names) - no in dask
            # names=None, squeeze=False, prefix=None, mangle_dupe_cols=True,
            # engine=None, true_values=None, false_values=None, skipinitialspace=False,
            #     nrows=None, na_values=None, keep_default_na=True, na_filter=True, verbose=False,
            #     skip_blank_lines=True, parse_dates=False, infer_datetime_format=False, keep_date_col=False,
            #     date_parser=None, dayfirst=False, iterator=False, chunksize=1000000, compression='infer',
            #     thousands=None, decimal=b'.', lineterminator=None, quotechar='"', quoting=0,
            #     escapechar=None, encoding=None, dialect=None, tupleize_cols=None,
            #      warn_bad_lines=True, skipfooter=0, skip_footer=0, doublequote=True,
            #     delim_whitespace=False, as_recarray=None, compact_ints=None, use_unsigned=None,
            #     low_memory=True, buffer_lines=None, memory_map=False, float_precision=None)
        except ValueError as e:
            l.exception('dask lib can not load data. Trying pandas lib...')
            for i, nf in enumerate(paths):
                df = pd.read_csv(
                    nf, dtype=cfg_in['dtype_raw'], names=cfg_in['cols'], usecols=cfg_in['dtype'].names,
                    # cfg_in['cols_load'],
                    delimiter=cfg_in['delimiter'], skipinitialspace=True, index_col=False,
                    converters=cfg_in['converters'], skiprows=cfg_in['skiprows'],
                    on_bad_lines=cfg_in['on_bad_lines'], comment=cfg_in['comment'],
                    header=None)
                if i > 0:
                    raise NotImplementedError('list of files => need concatenate data')
            ddf = vaex.from_pandas(df, chunksize=cfg_in['blocksize'])  #
    except Exception as e:  # for example NotImplementedError if bad file
        msg = '- Bad file. skip!'
        ddf = None
        if cfg_in['on_bad_lines'] == 'error':
            l.exception('%s\n Try set [in].on_bad_lines = warn\n', msg)
            raise
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
    cfg_in['file_stem'] = Path(paths[0]).stem  # may be need in func below to extract date
    try:
        date_delayed = None
        try:
            if not getattr(cfg_in['fun_proc_loaded'], 'meta_out', None) is None:
                # fun_proc_loaded will return modified data. Go to catch it
                # todo: find better condition
                raise TypeError

            date = ddf.map_partitions(lambda *args, **kwargs: pd.Series(
                cfg_in['fun_proc_loaded'](*args, **kwargs)), cfg_in, meta=meta_time)  # meta_time_index
            # date = date.to_series()
        except (TypeError, Exception) as e:
            # fun_proc_loaded retuns tuple (date, a)
            changing_size = False  # ? True  # ?
            if changing_size:

                @vaex.delayed
                def run_fun_proc_loaded():
                    """
                    delayed(, nout=2)(ddf, cfg_in)
                    :return:
                    """
                    return cfg_in['fun_proc_loaded']()


                date_delayed, a = run_fun_proc_loaded()
                ddf_len = len(ddf)
                counts_divisions = list(range(1, int(ddf_len / cfg_in.get('decimate_rate', 1)), cfg_in['blocksize']))
                counts_divisions.append(ddf_len)
                ddf = vaex.from_delayed(a, divisions=(0, counts_divisions))

                #date, meta = meta_time_index, divisions = counts_divisions); from_dask_array(date.values, index=ddf.index)
                date = date_delayed.get()

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
        return pd.DataFrame.from_dict(OrderedDict(zip(meta2.keys(), utils_time_corr.time_corr(t, cfg_in))))
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
    try:
        df_time_ok = df_time_ok.persist()

    except Exception as e:
        l.exception('Can not speed up by persist')
        # # something that can trigger error to help it identificate ???
        # date = date.persist()
        # df_time_ok = df_time_ok.compute()
        df_time_ok = time_corr_df(
            (date_delayed if date_delayed is not None else date).compute(), cfg_in=cfg_in)

    nbad_time = len(df_time_ok['b_ok']) - df_time_ok['b_ok'].sum()
    l.info('Removing %d bad time values: %s%s', nbad_time,
           df_time_ok['b_ok'].fillna(0).ne(True).to_numpy().nonzero()[0][:20],
           ' (shows first 20)' if nbad_time > 20 else '')

    df_time_ok.loc[df_time_ok['b_ok'], 'Time'] = pd.NaT
    try:
        df_time_ok.Time = df_time_ok.Time.interpolate(
            inplace=False)  # inplace=True - not works, method='linear', - default
    except ValueError:  # if not interpolates (my condition) use simpler method:
        df_time_ok.Time = df_time_ok.Time.fillna(method='ffill', inplace=True)

    if nbad_time:
        # # dask get IndexingError: Unalignable boolean Series provided as indexer (index of the boolean Series and of the indexed object do not match):
        # ddf_out = ddf.loc[df_time_ok['b_ok'], list(cfg_in['dtype_out'].names)].set_index(
        #    df_time_ok.loc[df_time_ok['b_ok'], 'Time'], sorted=True)

        # so we have done interpolate that helps this:
        ddf_out = ddf.loc[:, list(cfg_in['dtype_out'].names)].set_index(df_time_ok['Time'])  # , sorted=True
        ddf_out = ddf_out.loc[df_time_ok['b_ok'], :]
    else:
        # print('data loaded shape: {}'.format(ddf.compute(scheduler='single-threaded').shape))  # debug only
        ddf_out = ddf.loc[:, list(cfg_in['dtype_out'].names)].set_index(df_time_ok['Time'], sorted=True)

    logger = logging.getLogger("dask")
    logger.addFilter(lambda s: s.getMessage() != "Partition indices have overlap.")
    return ddf_out


def get_fun_proc_loaded_converters(
        fun_proc_loaded: Optional[Callable[[Any], Any]],
        converters: Optional[Mapping[str, Callable[[Any], Any]]],
        cfgFile: Union[str, PurePath],
        coltime):
    """
    Assign castom prep&proc based on args.cfgFile name
    :param fun_proc_loaded:
    :param converters:
    :param cfgFile: configuration file name, used for tring to determine fun_proc_loaded by its stem's last part
    :param coltime: only to get time converters for cfgFile ending with 'IdrRedas' or 'csv_iso_time'

    :return fun_proc_loaded: Callable if cfgFile name match found

        None if cfg_in['fun_proc_loaded'] not specifed
        cfg_in['fun_proc_loaded'] if
    """
    import to_pandas_hdf5.csv_specific_proc

    if fun_proc_loaded:
        return fun_proc_loaded
    cfg_file = Path(cfgFile).stem
    matched_fun_suffix = re.search('_((ctd|nav|inclin|chain|csv)_.*)', cfg_file)
    if not matched_fun_suffix:
        if cfg_file.endswith('IdrRedas'):
            converters = {coltime: lambda txtD_M_YYYY_hhmmssf:
            np.datetime64(b'%(2)b-%(1)b-%(0)bT%(3)b' % dict(
                zip([b'0', b'1', b'2', b'3'], (txtD_M_YYYY_hhmmssf[:19].replace(b' ', b'/').split(b'/')))))}
        elif cfg_file.endswith('csv_iso_time'):
            # more prepare for time in standard ISO 8601 format
            converters = {coltime: lambda txtYY_M_D_h_m_s_f: np.array(
                '20{0:02.0f}-{1:02.0f}-{2:02.0f}T{3:02.0f}:{4:02.0f}:{5:02.0f}.{6:02.0f}0'.format(
                    *np.array(np.fromstring(txtYY_M_D_h_m_s_f, dtype=np.uint8, sep=','), dtype=np.uint8)),
                dtype='datetime64[ns]')}  # - np.datetime64('2009-01-01T00:00:00', dtype='datetime64[ns]'):

        return None
    else:
        fun_proc_loaded = getattr(to_pandas_hdf5.csv_specific_proc, f'proc_loaded_{matched_fun_suffix.group(0)}')
        return fun_proc_loaded



import hydra
from omegaconf import DictConfig

try:
    from clize import run, converters, parameters
    from sigtools.wrappers import decorator

    @decorator
    def with_prog_config(
            wrapped,
            config_path='to_pandas_hdf5/csv2h5_ini/csv2h5_vaex.yml',
            return_: parameters.one_of('<cfg_from_args>', '<gen_names_and_log>', '<end>')='<end>',
            # b_interact=False,
            # verbose=:
            **kwargs):
        """
        overwrite command line arguments with program and in
        :param wrapped:
        :param return_: parameters.one_of('<cfg_from_args>', '<gen_names_and_log>', '<end>')='<end>',
        :param in_:
        :return:
        """
        global l

        @hydra.main(config_path=config_path)
        def main_cfg(cfg: DictConfig):
            global l

            # cfg = cfg_from_args(argparser_files(), **kwargs)
            if not cfg or not cfg['program'].get('return'):
                print('Can not initialise')
                return cfg
            elif cfg['program']['return'] == '<cfg_from_args>':  # to help testing
                return cfg

            l = init_logging(logging, None, cfg['program']['log'], cfg['program']['verbose'])
            print('\n' + this_prog_basename(__file__), end=' started. ')
            try:
                cfg['in']['paths'], cfg['in']['nfiles'], cfg['in']['path'] = init_file_names(
                    **cfg['in'], b_interact=cfg['program']['b_interact'])

            except Ex_nothing_done as e:
                print(e.message)
                return ()

            return cfg

        def wrap(**kwargs):
            return wrapped(**main_cfg(kwargs))

        return wrap


    @decorator
    def with_in_config(
            wrapped,
            fun_proc_loaded=None,
            **kwargs):
        """
            :param fun_proc_loaded: function(df: Dataframe, cfg_in: Optional[Mapping[str, Any]] = None) -> Dataframe/DateTimeIndex: to update/calculate new parameters from loaded data  before filtering. If output is Dataframe then function should have meta_out attribute which is Callable[[np.dtype, Iterable[str], Mapping[str, dtype]], Dict[str, np.dtype]]
        """

        def wrap(**kwargs):
            # Prepare loading and writing specific to format
            kwargs['in']['fun_proc_loaded'] = get_fun_proc_loaded_converters(**kwargs['in'])
            kwargs['in'] = init_input_cols(**kwargs['in'])

            return wrapped(**kwargs)

        return wrap


    @with_in_config
    def main(cfg):
        """
        :param new_arg: list of strings, command line arguments

        Note: if new_arg=='<cfg_from_args>' returns cfg but it will be None if argument
         argv[1:] == '-h' or '-v' passed to this code
        argv[1] is cfgFile. It was used with cfg files:
            'csv2h5_nav_supervisor.ini'
            'csv2h5_IdrRedas.ini'
            'csv2h5_Idronaut.ini'

        :return:
        """


    def version():
        """Show the version"""
        return 'version {0}'.format(VERSION)


    if __name__ == '__main__':
        run(main, alt=version)

except ModuleNotFoundError as e:
    print(standard_error_info(e))