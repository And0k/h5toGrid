#!/usr/bin/env python
# coding:utf-8
"""
  Author:  Andrey Korzh <ao.korzh@gmail.com>
  Purpose: Convert (multiple) csv and alike text files to pandas hdf5 store with
           addition of log table
  Created: 26.02.2016
"""
from __future__ import print_function, division

import logging
import re
from codecs import open
from datetime import datetime
from os import remove as os_remove, path as os_path  # , environ as os_environ
from time import sleep

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask import delayed

if __debug__:
    from matplotlib import pyplot as plt
    from sys import stdout as sys_stdout
# from future.moves.itertools import zip_longest
# from builtins import input
# from debug import __debug___print
# from  pandas.tseries.offsets import DateOffset
from dateutil.tz import tzoffset

import warnings
from tables.exceptions import HDF5ExtError
from tables import NaturalNameWarning
from pathlib import Path

# my:
from utils2init import init_file_names, getDirBaseOut, dir_create_if_need, \
    Ex_nothing_done, set_field_if_no, cfg_from_args, my_argparser_common_part, \
    this_prog_basename, init_logging, pathAndMask
# pathAndMask, dir_walker, bGood_dir, bGood_file
from old.loadBaranovText2h5 import find_sampling_frequency  # _fromMat.
from to_pandas_hdf5.h5toh5 import h5sort_pack, getstore_and_print_table
from other_filters import make_linear, longest_increasing_subsequence_i, check_time_diff, repeated2increased, rep2mean

warnings.catch_warnings()
warnings.simplefilter("ignore", category=NaturalNameWarning)
# warnings.filterwarnings('ignore', category=tables.NaturalNameWarning)

if __name__ == '__main__':
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
    """

    p = my_argparser_common_part({'description': 'csv2h5 version {}'.format(version) + """
----------------------------
Add data from CSV-like files
to Pandas HDF5 store*.h5
----------------------------""", 'default_config_files': [os_path.join(os_path.dirname(__file__), name) for name in
                                                          ('csv2h5.ini', 'csv2h5.json')]}, version)

    # Configuration sections

    # All argumets of type str (default for add_argument...), because of
    # custom postprocessing based of args names in ini2dict
    p_in = p.add_argument_group('in', 'all about input files')
    p_in.add('--path', default='.',  # nargs=?,
             help='path to source file(s) to parse. Use patterns in Unix shell style')
    p_in.add('--b_search_in_subdirs', default='True',
             help='search in subdirectories, used if mask or only dir in path (not full path)')
    p_in.add('--exclude_dirs_ends_with_list', default='toDel, -, bad, test, TEST',
             help='exclude dirs which ends with this srings. This and next option especially useful when search recursively in many dirs')
    p_in.add('--exclude_files_ends_with_list', default='coef.txt, -.txt, test.txt',
             help='exclude files which ends with this srings')
    p_in.add('--b_skip_if_up_to_date', default='True',
             help='exclude processing of files with same name and which time change is not bigger than recorded in database (only prints ">" if detected). If finds updated version of same file then deletes all data which corresponds old file and after it brfore procesing of next files')
    p_in.add('--dt_from_utc_hours', default='0',
             help='add this correction to loading datetime data. Can use other suffixes instead of "hours"')
    p_in.add('--header',
             help='comma separated list mached to input data columns to name variables. Can contain type suffix i.e. (float) - which is default, (text) - also to convert by specific converter, or (time) - for ISO format only. If it will')
    p_in.add('--cols_load_list',
             help='comma separated list of names from header to be saved in hdf5 store. not use "/" char for them')
    p_in.add('--skiprows_integer', default='1',
             help='skip rows from top. Use 1 to skip one line of header')
    p_in.add('--b_raise_on_err', default='True',
             help='if false then not rise error on rows which can not be loaded (only shows warning). Try set "comments" argument to skip them without warning')

    p_in.add('--max_text_width', default='1000',
             help='maximum length of text fields (specified by "(text)" in header) for dtype in numpy loadtxt')
    p_in.add('--chunksize_percent_float',
             help='percent of 1st file length to set up hdf5 store tabe chunk size')
    p_in.add('--blocksize_int', default='20000000',
             help='bytes, chunk size for loading and processing csv')
    p_in.add('--return',
             help='<return_cfg>: returns cfg only and exit, <return_cfg_step_gen_names_and_log>: execute init_input_cols() and also returns fun_proc_loaded function ... - see code')
    p_in.add('--b_interact', default='True',
             help='ask with showing files to process them')
    p_out = p.add_argument_group('output_files', 'all about output files')
    p_out.add('--db_path', help='hdf5 store file path')
    p_out.add('--table',
              help='table name in hdf5 store to write data. If not specified then will be generated on base of path of input files')
    # p_out.add('--tables_list',
    #           help='tables names in hdf5 store to write data (comma separated)')
    p_out.add('--b_insert_separator', default='True',
              help='insert NaNs row in table after each file data end')
    p_out.add('--b_use_old_temporary_tables', default='False',
              help='Warning! Set True only if temporary storage already have good data!'
                   'if True and b_skip_if_up_to_date= True then not replace temporary storage with current storage before adding data to the temporary storage')
    p_out.add('--b_remove_duplicates', default='False', help='Set True if you see warnings about')

    return (p)


dt64_1s = np.int64(1e9)
tzUTC = tzoffset('UTC', 0)


def timzone_view(t, dt_from_utc=0):
    """
    
    :param t: Pandas Timestamp time
    :param dt_from_utc: pd.Timedelta - timezone offset
    :return: t with applied timezone dt_from_utc
      Assume that if time zone of tz-naive Timestamp is naive then it is UTC
    """
    if dt_from_utc == 0:
        # dt_from_utc = pd.Timedelta(0)
        tzinfo = tzUTC
    elif dt_from_utc == pd.Timedelta(0):
        tzinfo = tzUTC
    else:
        tzinfo = tzoffset(None, pd.to_timedelta(dt_from_utc).total_seconds())

    if isinstance(t, pd.DatetimeIndex) or isinstance(t, pd.Timestamp):
        if t.tz is None:
            # think if time zone of tz-naive Timestamp is naive then it is UTC
            t = t.tz_localize('UTC')
        return t.tz_convert(tzinfo)
    else:
        l.error('Bad time format {}: {} - it is not subclass of pd.Timestamp/DatetimeIndex!'.format(type(t), t))
        t = pd.to_datetime(t).tz_localize(tzinfo)

    # t.to_datetime().replace(tzinfo= tzinfo) + dt_from_utc
    # t.astype(datetime).replace(


# ----------------------------------------------------------------------
def init_input_cols(cfg_in=None):
    """
        Append/modify dictionary cfg_in for parameters of numpy.loadtxt function and later pandas save to hdf5.
    :param cfg_in: dictionary, may has fields:
        header (required) - comma/space separated string. Column names in source file data header
        (as in Veusz standard input dialog), used to find cfg_in['cols'] if last is not cpecified
        dtype - type of data in column (as in Numpy loadtxt)
        converters - dict (see "converters" in Numpy loadtxt) or function(cfg_in) to make dict here
        cols_load - list of used column names

    :return: modified cfg_in dictionary. Will have fields:
        cols - list constructed from header by spit an remove format cpecifiers: '(text)','(float)','(time)'
        cols_load - list of used column numbers in needed order to save later
        coltime/coldate - assigned to index of 'Time'/'Date' column
        dtype:
            numpy.float64 - default and for '(float)' format specifier
            numpy string with length cfg_in['max_text_width'] - for '(text)'
            datetime64[ns] - for coldate column (or coltime if no coldate) and for '(time)'
        col_index_name - index name for saving Pandas frame. Will be set to name of cfg_in['coltime'] column if not exist already
        used in main() default time postload proc only (if no specific loader which calculates and returns time column for index)
        cols_loaded_save_b - columns mask of cols_load to save (some columns needed only before save to alculate
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

    if 'header' in cfg_in:  # if header specified
        compile_obj = re.compile(u"(?:[ \n,]+[ \n]*|^)(`[^`]+`|[^`,\n ]*)", re.VERBOSE)
        cfg_in['cols'] = compile_obj.findall(cfg_in['header'])
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

    if cfg_in['cols_load']:
        cols_load_b &= np.isin(cfg_in['cols'], cfg_in['cols_load'])
    else:
        cfg_in['cols_load'] = np.array(cfg_in['cols'])[cols_load_b]

    if 'cols_not_use' in cfg_in:  # may more narrow used cols
        cols_load_in_used_b = np.isin(cfg_in['cols_load'], cfg_in['cols_not_use'], invert=True)
        if not np.all(cols_load_in_used_b):
            cfg_in['cols_load'] = cfg_in['cols_load'][cols_load_in_used_b]
            cols_load_b = np.isin(cfg_in['cols'], cfg_in['cols_load'])

    col_names_out = cfg_in['cols_load'].copy()
    # convert to index to be compatible with numpy loadtxt() (not only genfromtxt())
    cfg_in['cols_load'] = np.int32([cfg_in['cols'].index(c) for c in cfg_in['cols_load'] if c in cfg_in['cols']])
    # not_cols_load = np.array([n in cfg_in['cols_not_use'] for n in cfg_in['cols']], np.bool)
    # cfg_in['cols_load']= np.logical_and(~not_cols_load, cfg_in['cols_load'])
    # cfg_in['cols']= np.array(cfg_in['cols'])[cfg_in['cols_load']]
    # cfg_in['dtype']=  cfg_in['dtype'][cfg_in['cols_load']]
    # cfg_in['cols_load']= np.flatnonzero(cfg_in['cols_load'])
    # cfg_in['dtype']= np.dtype({'names': cfg_in['cols'].tolist(), 'formats': cfg_in['dtype'].tolist()})

    cfg_in['cols'] = np.array(cfg_in['cols'])
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

    # to be shure that needed parameter for call loadtxt(...) exist
    set_field_if_no(cfg_in, 'delimiter')

    return cfg_in


# ----------------------------------------------------------------------
def time_corr(date, cfg_in, b_make_time_inc=False):
    """
    #for time in text format, propely formatted for conv.
    #convert to UTC    
    :param date: numpy datetime64 or array text in ISO 8601 format
    :param cfg_in:
    :param b_make_time_inc: check time resolution and increase if needed to avoid duplicates
    :return:
        tim, same size as date input
        hours_from_utc_f,
        b_ok - mask of not decreasing elements
    """
    if __debug__:
        l.debug('time_corr (time correction) started')
    if 'dt_from_utc' in cfg_in and cfg_in['dt_from_utc']:
        hours_from_utc_f = cfg_in['dt_from_utc'].total_seconds() / 3600
        Hours_from_UTC = int(hours_from_utc_f)
        hours_from_utc_f -= Hours_from_UTC
        if isinstance(date[0], str):
            if abs(hours_from_utc_f) > 0.0001:
                print('added only fixed number of hours. Implement time correction!')
            tim = pd.to_datetime((date.astype(np.object) + '{:+03d}'.format(Hours_from_UTC)).astype('datetime64[ns]'),
                                 utc=True)
        elif isinstance(date, pd.Index):
            tim = date
            if Hours_from_UTC == 0:
                tim = tim.tz_localize('UTC')
            else:
                tim.tz = tzoffset(None, -Hours_from_UTC * 3600)  # invert localize
                tim = tim.tz_localize(None).tz_localize('UTC')  # correct
        else:
            tim = pd.to_datetime(date.astype('datetime64[ns]') - np.timedelta64(
                pd.Timedelta(hours=Hours_from_UTC)), utc=True)
        # tim+= np.timedelta64(pd.Timedelta(hours=hours_from_utc_f)) #?
    else:
        if (not isinstance(date, pd.Series)) and (not isinstance(date, np.datetime64)):
            date = date.astype('datetime64[ns]')
        tim = pd.to_datetime(date, utc=True)  # .tz_localize('UTC')tz_convert(None)
        hours_from_utc_f = 0

    b_ok = np.ones_like(tim, np.bool8)
    if b_make_time_inc:
        # Check time resolution and increase if needed to avoid duplicates
        t = tim.values.view(np.int64)  # 'datetime64[ns]'
        freq, n_same, nDecrease, b_same_prev = find_sampling_frequency(t, precision=6, b_show=False)
        if nDecrease > 0:
            # Excude elements

            # if True:
            #     # try fast method
            #     b_bad_new = True
            #     k = 10
            #     while np.any(b_bad_new):
            #         k -= 1
            #         if k > 0:
            #             b_bad_new = bSpike1point(t[b_ok], max_spyke=2 * np.int64(dt64_1s / freq))
            #             b_ok[np.flatnonzero(b_ok)[b_bad_new]] = False
            #             print('step {}: {} spykes found, deleted {}'.format(k, np.sum(b_bad_new),
            #                                                                 np.sum(np.logical_not(b_ok))))
            #             pass
            #         else:
            #             break
            # if k > 0:  # success?
            #     t = rep2mean(t, bOk=b_ok)
            #     freq, n_same, nDecrease, b_same_prev = find_sampling_frequency(t, precision=6, b_show=False)
            #     # print(np.flatnonzero(b_bad))
            # else:
            #     t = tim.values.view(np.int64)
            # if nDecrease > 0:  # fast method is not success
            # take time:i
            # l.warning(Fast method is not success)

            # excluding repeated values
            i_different = np.flatnonzero(~b_same_prev)
            b_ok = np.zeros_like(tim, np.bool8)
            b_ok[i_different[longest_increasing_subsequence_i(t[~b_same_prev])]] = True
            # b_ok= nondecreasing_b(t, )
            # t = t[b_ok]
            t = rep2mean(t, bOk=b_ok)
            idel = np.flatnonzero(np.logical_not(b_ok))
            msg = 'Filtered time: {} values interpolated'.format(len(idel))

            l.warning('decreased time ({} times) is detected! {}'.format(nDecrease, msg))
            plt.figure('Decreasing time corr');
            plt.title(msg)
            plt.plot(np.arange(tim.size), tim, color='r')
            plt.plot(np.flatnonzero(b_ok), pd.to_datetime(t[b_ok], utc=True), color='g')
            plt.show()

        if 'fs' in cfg_in and n_same > 0:
            t = repeated2increased(t, cfg_in['fs'], b_same_prev)
            tim = pd.to_datetime(t, utc=True)
        elif n_same > 0 or nDecrease > 0:  # >0.1?
            # Increase time resolution by recalculating all values using constant frequency
            l.info('Increase time resolution using constant freq = ' + str(freq) + 'Hz')

            # # Check if we can simply use linear increasing values
            # freq2= np.float64(dt64_1s)*t.size/(t[-1]-t[0])
            # freqErr= abs(freq - freq2)/freq
            # if freqErr < 0.1: #?
            #     t = np.linspace(t[0], t[-1], t.size)
            #     #print(str(round(100*freqErr,1)) + '% error in frequency estimation. May be data gaps. Use unstable algoritm to correct')
            #     #t= time_res1s_inc(t, freq)
            # else: # if can not than use special algorithm:
            tim_before = pd.to_datetime(t, utc=True)
            make_linear(t, freq)  # will change t and tim
            tim = pd.to_datetime(t, utc=True)  # not need allways?
            check_time_diff(tim_before, tim.values, dt_warn=pd.Timedelta(minutes=2),
                            mesage='Big time diff after corr: difference [min]:')

        # check all is ok
        b_same_prev = np.diff(t) < 0
        nDecrease = np.sum(b_same_prev)
        if nDecrease > 0:
            l.warning('decreased time ({} times) remains!'.format(nDecrease))

        b_same_prev = np.diff(t) == 0
        n_same = np.sum(b_same_prev)
        if n_same > 0:
            l.warning('nonincreased time ({} times) is detected! - interp {}'.format(n_same))
            tim = pd.to_datetime(rep2mean(t, bOk=np.logical_not(b_same_prev)), utc=True)
            # t = repeated2increased(t, freq, b_same_prev)

    return tim, hours_from_utc_f, b_ok


# dfs['track'].index+= cfg['TimeAdd']
# dfs['track'].index.tz= tzoffset('UTC', 0)
# dfs['waypoints'].index+= cfg['TimeAdd']
# dfs['waypoints'].index.tz= tzoffset('UTC', 0)

# ----------------------------------------------------------------------
def filterGlobal_minmax(a, tim=None, cfg_filter=None):
    '''
    Filter min/max limits
    :param a:           numpy record array or Dataframe
    :param tim:         time array (convrtable to pandas Datimeinex) or None then use a.index instead
    :param cfg_filter:  dict with keys max_'field', min_'field', where 'field' must be
     in a or 'date' (case insensetive)
    :return:            dask bool array of good rows (or array if tim is not dask and only tim is filtered)
    '''
    """
    """
    bGood = True  # np.ones_like(tim, dtype=bool)
    for fkey, fval in cfg_filter.items():  # between(left, right, inclusive=True)
        fkey, flim = fkey.rsplit('_', 1)
        # swap if need:
        if fkey == 'min':
            fkey = flim
            flim = 'min'
        elif fkey == 'max':
            fkey = flim
            flim = 'max'
        # fkey may be lowercase(field) when parsed from *.ini so need find field yet:
        field = [field for field in (a.dtype.names if isinstance(a, np.ndarray
                                                                 ) else a.columns.get_values()) if
                 field.lower() == fkey.lower()]
        if field:
            field = field[0]
            if flim == 'min':
                bGood = da.logical_and(bGood, a[field] > fval)
            elif flim == 'max':
                bGood = da.logical_and(bGood, a[field] < fval)
        elif fkey == 'date':  # 'index':
            if tim is None:
                tim = a.index
            # fval= pd.to_datetime(fval, utc=True)
            fval = pd.Timestamp(fval, tz='UTC')
            if flim == 'min':
                bGood = da.logical_and(bGood, tim > fval)
            elif flim == 'max':
                bGood = da.logical_and(bGood, tim < fval)
        else:
            l.warning('filter worning: no field "{}"!'.format(fkey))
    return bGood


def set_filterGlobal_minmax(a, log, cfg_filter=None):
    '''
    Finds bad with filterGlobal_minmax and removes it from a,tim
    Adds 'rows' remained and 'rows_filtered' to log
    :param a:
    :param log: changes inplacce - adds ['rows_filtered'] and ['rows'] - number of rows remained
    :param cfg_filter: filtering settings, do nothing if None
    :return: number of rows remained
    '''
    tim = a.index.compute()
    if cfg_filter is None:  # can not do anything
        return a, tim

    log['rows_filtered'] = 0
    try:
        log['rows'] = len(a)  # shape[0]
    except IndexError:
        a = []
        return 0, None
    if cfg_filter:  # 'filter' in cfg:

        bGood = filterGlobal_minmax(a, tim, cfg_filter)
        try:  # may be dask or not dask array
            bGood = bGood.compute()
        except AttributeError:
            pass
        sum_good = np.sum(bGood)
        if sum_good < log['rows']:  # <=> bGood.any()

            try:

                # try:
                #     a = a.compute().iloc[bGood.values]
                #     #a.reindex(a.index[bGood], copy=False)
                #     #a.ix[bGood] #  or a.iloc[bGood.values]
                # except:
                # a.merge(pd.DataFrame(index=np.flatnonzero(bGood)))

                tim = tim[bGood]
                # a = a.merge(pd.DataFrame(index=np.flatnonzero(bGood)))  # a = a.compute(); a[bGood]

                a = a.join(pd.DataFrame(index=tim), how='right')

                # try:
                #     tim= tim[bGood.values] #.iloc
                # except:
                # ValueError: Item wrong length 2898492 instead of 2896739.
            except IndexError:
                a = []
                return 0, None
            log['rows_filtered'] = log['rows'] - sum_good
            log['rows'] = sum_good

    return a, tim


def multiindex_timeindex(df_index):
    """

    :param df_index: pandas index
    :return: DatetimeIndex if in df_index else []
    """
    b_MultiIndex = isinstance(df_index, pd.MultiIndex)
    if b_MultiIndex:
        itm = [isinstance(L, pd.DatetimeIndex) for L in df_index.levels].index(True)
        df_t_index = df_index.get_level_values(itm)  # not use df_index.levels[itm] which returns sorted
    else:
        df_t_index = df_index
        itm = None
    return df_t_index, itm


def multiindex_replace(pd_index, new1index, itm):
    """
    replace timeindex_even if_multiindex
    :param pd_index:
    :param new1index: replacement for pandas 1D index in pd_index
    :param itm: index of dimention in pandas MultiIndex which is need to replace by new1index. Use None if not MultiIndex
    :return: modified MultiIndex if itm is not None else new1index
    """
    if not itm is None:
        pd_index.set_levels(new1index, level=itm, verify_integrity=False)
        # pd_index = pd.MultiIndex.from_arrays([[new1index if i == itm else L] for i, L in enumerate(pd_index.values)], names= pd_index.names)
        # pd.MultiIndex([new1index if i == itm else L for i, L in enumerate(pd_index.levels)], labels=pd_index.labels, verify_integrity=False)
        # pd.MultiIndex.from_tuples([ind_new.values])
    else:
        pd_index = new1index
    return pd_index


# ----------------------------------------------------------------------
def h5_remove_table(store, tblName):
    if tblName in store:
        try:
            store.remove(tblName)
        except KeyError:
            # solve Pandas bug by reopen:
            store.close()
            store = pd.HDFStore(store.filename)  # cfg_out['db_path_temp']
            try:
                store.remove(tblName)
            except KeyError:
                raise HDF5ExtError('Can not remove table "{}"'.format(tblName))


# ----------------------------------------------------------------------
def h5del_obsolete(store, cfg_out, log, dfL):
    """
    Check that current file has been processed and it is up to date
    Removes all data from the store table and logtable which time >= time of data
    in log record of current file if it is changed!

    Also removes duplicates in the table if found duplicate records in the log
    """
    # dfL - log table loaded from store before updating
    # log - dict with info about current data, must have fields for compare:
    # 'fileName' - in format as in log table
    # 'fileChangeTime' - datetime
    # cfg_out - must have fields:
    # 'b_use_old_temporary_tables' - for message
    # 'tables', 'tables_log' - for deleting

    rows_for_file = dfL[dfL['fileName'] == log['fileName']]
    L = len(rows_for_file)
    bExistDup = False  # no detected yet
    bExistOk = False  # no detected yet
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
                for tblName, tblNameL in zip(cfg_out['tables'], cfg_out['tables_log']):
                    Ln = store.remove(tblNameL, where=qstrL)  # useful if it is not a child
                    L = store.remove(tblName, where=qstr)
                    print('{} in table/{} in log'.format(L, Ln))
            except NotImplementedError as e:
                print('Can not delete: ', e.__class__, ':', '\n==> '.join(
                    [s for s in e.args if isinstance(s, str)]))
                print('So removing full tables {} & {}'.format(tblName, tblNameL))
                store.remove(tblNameL)
                store.remove(tblName)
                bExistOk = False
                bExistDup = False
    return (bExistOk, bExistDup)


def h5_append_dummy_row(df, freq=None, tim=None):
    """
    Add row of NaN with index value that will between one of last data and one of next data start
    :param df: dataframe
    :param freq: frequency to calc index. If logically equal to False, then will be calculated using
     time of 2 previous rows
    :return: appended dataframe
    """
    if tim is not None:
        dindex = pd.Timedelta(seconds=0.5 / freq) if freq else np.abs(tim[-1] - tim[-2]) / 2
        ind_new = [tim[-1] + dindex]
    else:
        df_index, itm = multiindex_timeindex(df.index)
        dindex = pd.Timedelta(seconds=0.5 / freq) if freq else np.abs(df_index[-1] - df_index[-2]) / 2
        ind_new = multiindex_replace(df.index[-1:], df_index[-1:] + dindex, itm)

    df_dummy = pd.DataFrame(
        {name: field.type(0) if np.issubdtype(field.type, np.integer) else np.NaN if np.issubdtype(
            field.type, np.floating) else '' for name, field in df.dtypes.iteritems()},
        columns=df.columns.values, index=ind_new)
    if isinstance(df, dd.DataFrame):
        return dd.concat([df, df_dummy], axis=0, interleave_partitions=True)  # buggish dask not always can append
    else:
        return df.append(df_dummy)

    # np.array([np.int32(0) if np.issubdtype(field.type, int) else
    #           np.NaN if np.issubdtype(field.type, float) else
    #           [] for field in df.dtypes.values]).view(
    #     dtype=np.dtype({'names': df.columns.values, 'formats': df.dtypes.values})))

    # insert separator # 0 (can not use np.nan in int) [tim[-1].to_pydatetime() + pd.Timedelta(seconds = 0.5/cfg['in']['fs'])]
    #   df_dummy= pd.DataFrame(0, columns=cfg_out['names'], index= (pd.NaT,))
    #   df_dummy= pd.DataFrame(np.full(1, np.NaN, dtype= df.dtype), index= (pd.NaT,))
    # used for insert separator lines


def h5init(cfg_in, cfg_out):
    """
    Init cfg_out db (hdf5 data store) information in cfg_out _if not exist_
    :param: cfg_in, cfg_out - configuration dicts, with fields:
        cfg_in:
            path if no 'db_path' in cfg_out
            source_dir_words (optional), default: ['source', 'WorkData', 'workData'] - see getDirBaseOut()
            nfiles (optional)
            b_skip_if_up_to_date (optional)
        cfg_out: all fields are optional

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
        
    Returns: None
    """
    set_field_if_no(cfg_out, 'logfield_fileName_len', 255)
    set_field_if_no(cfg_out, 'chunksize')
    set_field_if_no(cfg_out, 'b_skip_if_up_to_date', cfg_in['b_skip_if_up_to_date' \
        ] if 'b_skip_if_up_to_date' in cfg_in else False)
    set_field_if_no(cfg_out, 'b_remove_duplicates', False)
    set_field_if_no(cfg_out, 'b_use_old_temporary_tables', True)

    # automatic names
    cfg_source_dir_words = cfg_in['source_dir_words'] if 'source_dir_words' in cfg_in else \
        ['source', 'WorkData', 'workData']
    auto = {'db_ext': 'h5'}
    if ('db_path' in cfg_out) and cfg_out['db_path']:
        # print(cfg_out)
        # print('checking db_path "{}" is absolute'.format(cfg_out['db_path']))
        if os_path.isabs(cfg_out['db_path']):
            auto['db_path'] = cfg_out['db_path']
        else:
            auto['db_base'] = os_path.join(os_path.split(cfg_in.get('path'))[0], cfg_out['db_path'])
            cfg_out['db_path'] = ''
    else:
        auto['db_path'] = os_path.split(cfg_in.get('path'))[0]
    auto['db_path'], auto['db_base'], auto['table'] = getDirBaseOut(auto['db_path'], cfg_source_dir_words)
    auto['db_base'] = os_path.splitext(auto['db_base'])[0]  # extension is specified in db_ext

    cfg_out['db_dir'], cfg_out['db_base'] = pathAndMask(*[cfg_out[spec] if (spec in cfg_out and cfg_out[spec]) else
                                                          auto[spec] for spec in ['db_path', 'db_base', 'db_ext']])
    dir_create_if_need(cfg_out['db_dir'])
    cfg_out['db_path'] = os_path.join(cfg_out['db_dir'], cfg_out['db_base'])

    # set_field_if_no(cfg_out, 'db_base', auto['db_base'] + ('.h5' if not auto['db_base'].endswith('.h5') else ''))
    # set_field_if_no(cfg_out, 'db_path', os_path.join(auto['db_path'], cfg_out['db_base']+ ('.h5' if not cfg_out['db_base'].endswith('.h5') else '')))

    # Will save to temporary file initially
    set_field_if_no(cfg_out, 'db_path_temp', cfg_out['db_path'][:-3] + '_not_sorted.h5')

    set_field_if_no(cfg_out, 'nfiles', cfg_in.get('nfiles') if 'nfiles' in cfg_in else 1)

    if 'tables' in cfg_out and cfg_out['tables']:
        set_field_if_no(cfg_out, 'tables_log', [tab + '/logFiles' for tab in cfg_out['tables']])
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


def h5temp_open(cfg_out):
    """
    Checks and generates some names used for saving to my *.h5 files. Opens HDF5 store,
    copies previous store data to this if 'b_skip_if_up_to_date'*  
    
    :param: cfg_out, dict
    Note:
      h5init(cfg_in, cfg_out) or similar must be called before to setup needed fields:
          db_path, db_path_temp
          tables, tables_log
          b_skip_if_up_to_date
          b_use_old_temporary_tables, bool, defult False - not copy tables from dest to temp
          
          
    Returns: (store, dfL)
    store   - pandas HDF5 store
    dfL     - is dataframe of log from store if cfg_in['b_skip_if_up_to_date']==True else None.
    """

    print('saving to', '/'.join([cfg_out['db_path_temp'], ','.join(
        cfg_out['tables'])]) + ':')
    store = None
    try:
        try:  # open temporary output file
            if os_path.isfile(cfg_out['db_path_temp']):
                store = pd.HDFStore(cfg_out['db_path_temp'])

                if not cfg_out['b_use_old_temporary_tables']:
                    for tblName in (cfg_out['tables'] + cfg_out['tables_log']):
                        if store:
                            h5_remove_table(store, tblName)
        except IOError as e:
            print(e)

        if cfg_out['b_skip_if_up_to_date']:
            if not cfg_out['b_use_old_temporary_tables']:
                # Copying previous store data to temporary one
                l.info('Copying previous store data to temporary one:')
                tblName = 'begin'
                try:
                    with pd.HDFStore(cfg_out['db_path']) as storeOut:
                        for tblName in (cfg_out['tables'] + cfg_out['tables_log']):
                            try:  # Check output store
                                if tblName in storeOut:  # avoid harmful sortAndPack errors
                                    h5sort_pack(cfg_out['db_path'], os_path.basename(
                                        cfg_out['db_path_temp']), tblName)
                                else:
                                    raise HDF5ExtError('Table {} not exist'.format(tblName))
                            except HDF5ExtError as e:
                                if tblName in storeOut.root.__members__:
                                    print('Node exist but store is not conforms Pandas')
                                    getstore_and_print_table(storeOut, tblName)
                                raise e  # exclude next processing
                            except RuntimeError as e:
                                l.error(
                                    'failed check on copy. May be need first to add full index to original store? Trying: ')
                                nodes = storeOut.get_node(tblName).__members__  # sorted(, key=number_key)
                                for n in nodes:
                                    tblNameCur = tblName if n == 'table' else tblName + '/' + n
                                    l.info(tblNameCur, end=', ')
                                    storeOut.create_table_index(tblNameCur, columns=['index'],
                                                                kind='full')  # storeOut[tblNameCur]
                                # storeOut.flush()
                                l.error('Trying again')
                                if (store is None) and store.is_open:
                                    store.close()
                                h5sort_pack(cfg_out['db_path'], os_path.basename(
                                    cfg_out['db_path_temp']), tblName)
                    l.info('Will append data only if find new files.')
                except Exception as e:
                    print('- failed option (when coping {}: {}). '.format(
                        tblName, '\n==> '.join([s for s in e.args if isinstance(s, str)])))
                    print('Will process all source data')
                    cfg_out['b_skip_if_up_to_date'] = False

        if (store is None) or not store.is_open:
            # Open temporary output file to return
            for attempt in range(2):
                try:
                    store = pd.HDFStore(cfg_out['db_path_temp'])
                    break
                except IOError as e:
                    print(e)
                except HDF5ExtError as e:  #
                    print('can not use old temporary output file. Deleting it...')
                    os_remove(cfg_out['db_path_temp'])
                    # raise(e)

        # Copy one log dataframe
        if cfg_out['b_skip_if_up_to_date']:
            if cfg_out['tables_log'][0] in store:
                dfL = store[cfg_out['tables_log'][0]]
        else:
            dfL = None
            # Remove existed tables to write if allowed:
            if not cfg_out['b_use_old_temporary_tables']:
                with pd.HDFStore(cfg_out['db_path']) as storeOut:
                    for tblName in (cfg_out['tables'] + cfg_out['tables_log']):
                        try:
                            h5_remove_table(storeOut, tblName)
                        except HDF5ExtError as e:
                            print('can not remove table {} in source store! Delete it manually!'.format(tblName))
                            exit()
                for tblName in (cfg_out['tables'] + cfg_out['tables_log']):
                    h5_remove_table(store, tblName)
    except HDF5ExtError as e:
        store.close()
        print('can not use old temporary output file. Deleting it...')
        os_remove(cfg_out['db_path_temp'])
        sleep(1)
        store = pd.HDFStore(cfg_out['db_path_temp'])
        cfg_out['b_skip_if_up_to_date'] = False

    return (store, dfL)


# ----------------------------------------------------------------------

def h5_append(store, df, log, cfg_out, log_dt_from_utc=pd.Timedelta(0), tim=None):
    '''
    Append to Store dataframe df to table and
    append chield table with 1 row metadata including 'index' and 'DateEnd'
    is calculated as first and last elements of df.index
            
    :param store: hdf5 file where to append
    :param df: pandas or dask datarame to append. If dask then log_dt_from_utc must be None (not assign log metadata here)
    :param log: dict which will be appended to chield table, cfg_out['table_log']
    :param cfg_out: dict with fields:
        table: name of table to update (or tables: list, then used only 1st element)
        table_log: name of chield table (or tables_log: list, then used only 1st element)
        b_insert_separator (optional), freq (optional)
        chunksize: may be None but then must be chunksize_percent to calcW ake Up:
            chunksize = len(df) * chunksize_percent / 100
        
    :param log_dt_from_utc: 0 or pd.Timedelta - to correct start and end time: index and DateEnd.
        if None then start and end time: 'Date0' and 'DateEnd' fields of log must be filled right already
    :return: None
    :updates:
        log:
            'Date0' and 'DateEnd'
        cfg_out: only if not defined already:
            cfg_out['table_log'] = cfg_out['tables_log'][0]
            table_log
    '''
    df_len = len(df) if tim is None else len(
        tim)  # not very fast for dask.dataframes so use computed values if possible
    msg_func = 'h5_append({}rows)'.format(df_len)
    if __debug__:
        l.info(msg_func + ' is going... ')
    if df_len:  # dask.dataframe.empty is not implemented
        # check/set tables names
        if 'tables' in cfg_out: set_field_if_no(cfg_out, 'table', cfg_out['tables'][0])
        if 'tables_log' in cfg_out: set_field_if_no(cfg_out, 'table_log', cfg_out['tables_log'][0])
        set_field_if_no(cfg_out, 'table_log', cfg_out['table'] + '/log')
        set_field_if_no(cfg_out, 'logfield_fileName_len', 255)
        set_field_if_no(cfg_out, 'nfiles', 1)
        # Add separatiion row of NaN and save to store
        if ('b_insert_separator' in cfg_out) and cfg_out['b_insert_separator']:
            msg_func = 'h5_append({}rows+1dummy)'.format(df_len)
            cfg_out.setdefault('fs')
            df = h5_append_dummy_row(df, cfg_out['fs'], tim)
            df_len += 1
        if (cfg_out['chunksize'] is None) and ('chunksize_percent' in cfg_out):  # based on first file
            cfg_out['chunksize'] = int(df_len * cfg_out['chunksize_percent'] / 1000) * 10
            if cfg_out['chunksize'] < 100: cfg_out['chunksize'] = None
        try:

            df.to_hdf(store, cfg_out['table'], append=True,
                      data_columns=True, format='table')  # , compute=False
            # store.append(cfg_out['table'], df, data_columns=True, index=False,
            #              chunksize=cfg_out['chunksize'])

        except ValueError as e:
            error_info_list = [s for s in e.args if isinstance(s, str)]
            msg = msg_func + ' Error:'.format(e.__class__) + '\n==> '.join(error_info_list)
            try:
                if error_info_list and error_info_list[0] == 'Not consistent index':
                    msg += 'Not consistent index detected'
                elif error_info_list and error_info_list[0] == 'cannot match existing table structure':
                    pass
                else:  # Can only append to Tables
                    msg += ' - Can not handle this error!'
                l.error(msg + 'Not consistent index? Changing index to standard UTC')
                df_cor = store[cfg_out['table']]
                if cfg_out['table_log'] in store:
                    dfLog = store[cfg_out['table_log']]  # copy before delete
                df_cor.index = pd.to_datetime(store[cfg_out['table']].index, utc=True)
                store.remove(cfg_out['table'])
                store.append(cfg_out['table'],
                             df_cor.append(df, verify_integrity=True),
                             data_columns=True, index=False,
                             chunksize=cfg_out['chunksize'])
                if cfg_out['table_log'] in store:
                    store.remove(cfg_out['table_log'])  # have removed only if it is a child
                    store.append(cfg_out['table_log'], dfLog, data_columns=True, expectedrows=cfg_out['nfiles'],
                                 index=False, min_itemsize={'values': cfg_out['logfield_fileName_len']})  # append
            except Exception as e:
                l.error(msg_func + ' Can not write to store. Error:'.format(e.__class__) + '\n==> '.join(
                    [s for s in e.args if isinstance(s, str)]))
        except TypeError as e:
            if isinstance(df, dd.DataFrame):
                l.error(msg_func + ': dask not writes separator? skip {}: '.format(e.__class__) + '\n==> '.join(
                    [s for s in e.args if isinstance(s, str)]))
            else:
                l.error(msg_func + ': Can not write to store. {}: '.format(e.__class__) + '\n==> '.join(
                    [s for s in e.args if isinstance(s, str)]))
                raise (e)
        except Exception as e:
            l.error(msg_func + ': Can not write to store. {}: '.format(e.__class__) + '\n==> '.join(
                [s for s in e.args if isinstance(s, str)]))
            raise (e)

        # Log to store
        log['Date0'], log['DateEnd'] = timzone_view((tim if tim is not None else df.index)[[0, -1]], log_dt_from_utc)
        # dfLog = pd.DataFrame.from_dict(log, np.dtype(np.unicode_, cfg_out['logfield_fileName_len']))
        dfLog = pd.DataFrame.from_records(log, exclude=['Date0'],
                                          index=[log['Date0']])  # index='Date0' not work dor dict
        store.append(cfg_out['table_log'], dfLog, data_columns=True, expectedrows=cfg_out['nfiles'],
                     index=False, min_itemsize={'values': cfg_out['logfield_fileName_len']})


# ----------------------------------------------------------------------
def h5_remove_duplicates(store, cfg, cfg_table_keys):
    """
    Remove duplicates by coping tables to memory
    :param store: 
    :param cfg: dict with keys:
        keys specified by cfg_table_keys
        chunksize - for data table
        logfield_fileName_len, nfiles - for log table
    :param cfg_table_keys: list, in which 'tables_log' means that cfg['tables_log'] is a log table
    :return: 
    """

    # load data frames from store to memory removing duplicates
    dfs = {}
    b_need_remove = {}
    for cfgListName in cfg_table_keys:
        for tblName in cfg[cfgListName]:
            if tblName in store:
                dfs[tblName] = store[tblName]
                b_dup = dfs[tblName].index.duplicated(keep='last')
                b_need_remove[tblName] = np.any(b_dup)
                if b_need_remove[tblName]:
                    print('{} duplicates in {} (first at {})'.format(
                        sum(b_dup), tblName, dfs[tblName].index[np.flatnonzero(b_dup)[0]]), end=' ')
                    dfs[tblName] = dfs[tblName][~b_dup]

    # update data frames in store
    if ~np.any([b for b in b_need_remove.values()]):
        print(end='Not need remove duplicates. ')
    else:
        print(end='Remove duplicates. ')
        for cfgListName in cfg_table_keys:
            for tblName in cfg[cfgListName]:
                if b_need_remove[tblName]:
                    try:
                        h5_remove_table(store, tblName)
                        # def h5_append(store, cfg, cfg_table_keys, cfg_table_log_key):
                        if cfgListName == 'tables_log':
                            store.append(tblName, dfs[tblName], data_columns=True, index=False,
                                         expectedrows=cfg['nfiles'],
                                         min_itemsize={'values': cfg['logfield_fileName_len']})
                        else:
                            store.append(tblName, dfs[tblName], data_columns=True, index=False,
                                         chunksize=cfg['chunksize'])
                    except Exception as e:
                        print(': table {} not recorded because of error when removing duplicates'.format(tblName))
                        print('{}: '.format(e.__class__), '\n==> '.join([s for s in e.args if isinstance(s, str)]))
                        # store[tblName].drop_duplicates(keep='last', inplace=True) #returns None


# ----------------------------------------------------------------------
def h5move_tables(cfg_out, tbl_names=None):
    """
    Copy tables tbl_names from one store to another using ptrepack. If fail to store
    in specified location then creates new store and tries to save there.
    :param cfg_out: dict - must have fields:
        'db_path_temp': source of not sorted tables
        'base': base name (extension ".h5" will be added if absent) of hdf store to put
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

    def unzip_if_need(lst_of_lsts):
        for lsts in lst_of_lsts:
            if isinstance(lsts, str):
                yield lsts
            else:
                yield from lsts

    for tblName in unzip_if_need(tbl_names):
        try:
            h5sort_pack(cfg_out['db_path_temp'], storage_basename + '.h5', tblName)  # (fileOutF, FileCum, strProbe)
            sleep(2)
        except Exception as e:
            l.error('Error: "{}"\nwhen write {} to {}'.format(e, tblName, cfg_out['db_path_temp']))

            if False:
                storage_basename = os_path.splitext(cfg_out['db_base'])[0] + "-" + tblName.replace('/', '-') + '.h5'
                l.info('so start write to {}'.format(storage_basename))
                try:
                    h5sort_pack(cfg_out['db_path_temp'], storage_basename, tblName)
                    sleep(4)
                except Exception as e:
                    storage_basename = cfg_out['db_base'] + '-other_place.h5'
                    l.error('Error: "{}"\nwhen write {} to original place so start write to {}'.format(e, tblName,
                                                                                                       storage_basename))
                    try:
                        h5sort_pack(cfg_out['db_path_temp'], storage_basename, tblName)
                        sleep(8)
                    except:
                        l.error(tblName + ': no success')
                storage_basenames[tblName] = storage_basename
    if storage_basenames == {}:
        storage_basenames = None
    return storage_basenames


#
def h5index_sort(cfg_out, out_storage_name=None, in_storages=None, tables=None):
    """
    Checks if tables in store have sorted index
     and if not then sort it by loading, sorting and saving data
    :param cfg_out: dict - must have fields:
        'path': tables to check monotonous and sort/
        'db_path_temp': source of not sorted tables
        'base': base name (extension ".h5" will be added if absent) of hdf store to put
        'tables' and 'tables_log' - if tables not specified
    
        'dt_from_utc'
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
        for tblName in tables:
            if tblName not in store:
                l.warning('{} not in {}'.format(tblName, in_storages))
                continue
            try:
                df = store[tblName]  # cfg_out['tables'][0]
                if df is None:
                    l.warning('None table {} in {}'.format(tblName, store.filename))
                    continue
            except TypeError as e:
                l.warning('Can not access table {}. {}:'.format(tblName, e.__class__) + '\n==> '.join(
                    [s for s in e.args if isinstance(s, str)]))
                continue
            # store.close()
            if df.index.is_monotonic:
                if df.index.is_unique:
                    l.info(tblName + ' - sorted')
                else:
                    b_have_duplicates = True
                    l.warning(tblName + ' - sorted, but have duplicates')
                continue
            else:
                b_need_save = True
                l.warning(tblName + ' - not sorted!')
                print(repr(store.get_storer(cfg_out['tables'][0]).group.table))

                df_index, itm = multiindex_timeindex(df.index)
                if __debug__:
                    plt.figure('index is not sorted')
                    plt.plot(df_index)  # np.diff(df.index)
                    plt.show()

                if not itm is None:
                    l.warning('sorting...')
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
                    l.warning('skipped of sorting multiindex')
        if b_have_duplicates:
            l.warning('To drop duplicates restart with [output_files][b_remove_duplicates] = True')
        else:
            l.info('Ok, no duplicates')
        if b_need_save:
            # out to store
            h5move_tables(cfg_out, tbl_names=tables)

            # store = pd.HDFStore(cfg_out['db_path_temp'])
            # store.create_table_index(tblName, columns=['index'], kind='full')
            # store.create_table_index(cfg_out['tables_log'][0], columns=['index'], kind='full') #tblName+r'/logFiles'
            # h5_append(store, df, log, cfg_out, cfg_out['dt_from_utc'])
            # store.close()
            # h5sort_pack(cfg_out['db_path_temp'], out_storage_name, tblName) #, ['--overwrite-nodes=true']


# ##############################################################################
# ___________________________________________________________________________

def read_csv(nameFull, **cfg_in):
    """
    Read csv in dask dataframe and then time correction with arguments defined in cfg_in fields
    :param nameFull:
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
        
        Also cfg_in has fild
            'dtype_out' which "names" field used to detrmine output columns
        
        See also time_corr() for used fields
    
    
    
    :return: tuple (a, b_ok) where
        a:      dask dataframe with time index and only columns listed in cfg_in['dtype_out'].names
        b_ok:   time correction result bulean array
    """
    try:
        # df
        a = dd.read_csv(
            nameFull, dtype=cfg_in['dtype'], names=cfg_in['cols'][cfg_in['cols_load']],
            delimiter=cfg_in['delimiter'], usecols=cfg_in['cols_load'],
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
    except Exception as e:
        msg = '{}: {} - Bad file. skip!\n'.format(e.__class__, '\n==> '.join([
            m for m in e.args if isinstance(m, str)]))
        a = None
        if cfg_in['b_raise_on_err']:
            l.error(msg + '{}\n Try set [in].b_raise_on_err= False'.format(e))
            raise (e)
        else:
            l.error(msg)
    if __debug__:
        l.debug('read_csv initialised')
    if a is None:
        return None, None

    try:
        # Process a and get date date in ISO string or numpy standard format
        date = cfg_in['fun_proc_loaded'](a, cfg_in).compute()
        if isinstance(date, tuple):  # date is actually (date, a) tuple
            date, a = date
    except IndexError:
        print('no data?')
        return None, None
        # add time shift specified in configuration .ini
    tim, hours_from_utc_f, b_ok = time_corr(date, cfg_in, b_make_time_inc=True)
    # if len(a) == 1:  # size
    #     a = a[np.newaxis]
    npartitions = a.npartitions
    a = a.loc[:, list(cfg_in['dtype_out'].names)].compute()
    a.set_index(tim, inplace=True)
    a = dd.from_pandas(a, npartitions=npartitions)

    return a, b_ok


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

def main(new_arg=None):
    """

    :param new_arg: returns cfg if new_arg=='<return_cfg>' but it will be None if argument
     argv[1:] == '-h' or '-v' passed to this code
    argv[1] is cfgFile. It was used with cfg files:
        'csv2h5_nav_supervisor.ini'
        'csv2h5_IdrRedas.ini'
        'csv2h5_Idronaut.ini'
    :return:
    """

    # global cfg, l
    cfg = cfg_from_args(my_argparser(), new_arg)
    if not cfg:
        return
    if cfg['in']['return'] == '<return_cfg>':  # to help testing
        return cfg

    l = init_logging(logging, None, cfg['program']['log'], cfg['program']['verbose'])
    print('\n' + this_prog_basename(__file__), end=' started. ')
    try:
        cfg['in'] = init_file_names(cfg['in'], ('b_interact' not in cfg['in']) or cfg['in']['b_interact'])
    except Ex_nothing_done as e:
        print(e.message)
        return ()

    ### Assign castom prep&proc based on args.cfgFile name ###
    cfg['in']['fun_proc_loaded'] = None  # Assign default proc below column assinment
    import to_pandas_hdf5.csv_specific_proc
    if cfg['in']['cfgFile'].endswith('Sea&Sun'):
        cfg['in']['fun_proc_loaded'] = delayed(to_pandas_hdf5.csv_specific_proc.proc_loaded_sea_and_sun)
    elif cfg['in']['cfgFile'].endswith('Idronaut'):
        cfg['in']['fun_proc_loaded'] = delayed(to_pandas_hdf5.csv_specific_proc.proc_loaded_Idronaut)
    elif cfg['in']['cfgFile'].endswith('IdrRedas'):
        cfg['in']['converters'] = {cfg['in']['coltime']: lambda txtD_M_YYYY_hhmmssf:
        np.datetime64(b'%(2)b-%(1)b-%(0)bT%(3)b' % dict(
            zip([b'0', b'1', b'2', b'3'], (txtD_M_YYYY_hhmmssf[:19].replace(b' ', b'/').split(b'/')))))}
        # b'{2}-{1}-{0}T{3}' % (txtD_M_YYYY_hhmmssf[:19].replace(b' ',b'/').split(b'/')))} #len=19 because bug of bad milliseconds
        # cfg['in']['fun_proc_loaded']= proc_loaded_IdrRedas
    elif cfg['in']['cfgFile'].endswith('nav_supervisor') or cfg['in']['cfgFile'].endswith('meteo'):
        cfg['in']['fun_proc_loaded'] = delayed(to_pandas_hdf5.csv_specific_proc.proc_loaded_nav_supervisor)
    elif cfg['in']['cfgFile'].endswith('ctd_Schuka'):
        cfg['in']['fun_proc_loaded'] = delayed(to_pandas_hdf5.csv_specific_proc.proc_loaded_ctd_Schuka)
    elif cfg['in']['cfgFile'].endswith('ctd_Schuka_HHMM'):
        cfg['in']['fun_proc_loaded'] = delayed(to_pandas_hdf5.csv_specific_proc.proc_loaded_ctd_Schuka_HHMM)
    elif cfg['in']['cfgFile'].endswith('csv_log'):
        cfg['in']['fun_proc_loaded'] = delayed(to_pandas_hdf5.csv_specific_procproc_loaded_csv_log)
    elif cfg['in']['cfgFile'].endswith('csv_iso_time'):
        # more prepare for time in standard ISO 8601 format
        cfg['in']['converters'] = {cfg['in']['coltime']: lambda txtYY_M_D_h_m_s_f: np.array(
            '20{0:02.0f}-{1:02.0f}-{2:02.0f}T{3:02.0f}:{4:02.0f}:{5:02.0f}.{6:02.0f}0'.format(
                *np.array(np.fromstring(txtYY_M_D_h_m_s_f, dtype=np.uint8, sep=','), dtype=np.uint8)),
            dtype='datetime64[ns]')}  # - np.datetime64('2009-01-01T00:00:00', dtype='datetime64[ns]')
    elif cfg['in']['cfgFile'].endswith('chain_Baranov') or cfg['in']['cfgFile'].endswith('inclin_Baranov'):
        cfg['in']['fun_proc_loaded'] = delayed(to_pandas_hdf5.csv_specific_proc.proc_loaded_chain_Baranov)
    elif cfg['in']['cfgFile'].endswith('csv_Baklan'):
        cfg['in']['fun_proc_loaded'] = delayed(to_pandas_hdf5.csv_specific_proc.proc_loaded_Baklan)
    elif cfg['in']['cfgFile'].endswith('inclin_Kondrashov'):
        cfg['in']['fun_proc_loaded'] = delayed(to_pandas_hdf5.csv_specific_proc.proc_loaded_inclin_Kondrashov)

    # Prepare cpecific format loading and writing
    cfg['in'] = init_input_cols(cfg['in'])
    # cfg['output_files']['dtype'] = cfg['in']['dtype_out']
    cfg_out = cfg['output_files'];
    h5init(cfg['in'], cfg_out)
    # Default time postload proc
    if cfg['in']['fun_proc_loaded'] is None:
        if 'coldate' not in cfg['in']:  # Time includes Date
            # name=''
            cfg['in']['fun_proc_loaded'] = delayed(lambda a, cfg_in: a[cfg_in['col_index_name']])
        else:  # Time + Date
            cfg['in']['fun_proc_loaded'] = delayed(lambda a, cfg_in: a['Date'] + np.array(
                np.int32(1000 * a[cfg_in['col_index_name']]), dtype='m8[ms]'))

    if cfg['in']['return'] == '<return_cfg_step_fun_proc_loaded>':  # to help testing
        return cfg

    # to insert separator lines:
    df_dummy = pd.DataFrame(np.full(
        1, np.NaN, dtype=cfg['in']['dtype_out']), index=(pd.NaT,))
    cfg_out['log'] = {'fileName': None,
                      'fileChangeTime': None}  # log= np.array(([], [],'0',0), dtype= [('Date0', 'O'), ('DateEnd', 'O'),

    # ('fileName', 'S255'), ('rows', '<u4')])

    def gen_names_and_log(cfg_out, store=None, dfLogOld=None):
        """
        Generates file names to load.
        If cfg_out['b_skip_if_up_to_date'] skips procesed files. But if file was changed
        removes stored data and nevetherless returns file name
        :param cfg_out: have fields
            log: current file info
        :param store, dfLogOld: as returned by h5temp_open()
        :return:
        :updates: cfg_out['b_remove_duplicates'], cfg_out['log']['fileName'], cfg_out['log']['fileChangeTime']
        """
        # todo: if cfg_in is None:, cfg_in=None

        for ifile, nameFull in enumerate(cfg['in']['namesFull'], start=1):
            # info to identify record (used to update db incrementally)
            nameFE = os_path.basename(nameFull)
            cfg['in']['file_stem'] = nameFE[:-4]  # to extract date if need
            cfg_out['log']['fileName'] = nameFE[-cfg_out['logfield_fileName_len']:-4]
            cfg_out['log']['fileChangeTime'] = datetime.fromtimestamp(os_path.getmtime(nameFull))

            # remove stored data and process file if it is changed file
            if cfg_out['b_skip_if_up_to_date']:
                bExistOk, bExistDup = h5del_obsolete(store, cfg_out, cfg_out['log'], dfLogOld)
                if bExistOk:
                    continue
                if bExistDup:
                    cfg_out['b_remove_duplicates'] = True  # normally no duplicates but will if detect
            print('{}. {}'.format(ifile, nameFE), end=': ')
            if __debug__:
                sys_stdout.flush()
            yield Path(nameFull)

    if cfg['in']['return'] == '<return_cfg_step_gen_names_and_log>':  # to help testing
        cfg['in']['gen_names_and_log'] = gen_names_and_log
        cfg['output_files'] = cfg_out
        return cfg

    # Writing
    if True:  # try:
        if cfg['program']['log']:
            try:
                flog = open(cfg['program']['log'], 'a+', encoding='cp1251')
                flog.writelines(datetime.now().strftime('\n\n%d.%m.%Y %H:%M:%S> processed '
                                                        + str(cfg['in']['nfiles']) + ' file' + 's:' if cfg['in'][
                                                                                                           'nfiles'] > 1 else ':'))
            except FileNotFoundError as e:
                print('{}: {} - skip logging operations!\n'.format(
                    e.__class__, '\n==> '.join([a for a in e.args if isinstance(a, str)])))
                cfg['program']['log'] = None

        # ## Main circle ############################################################
        store, dfLogOld = h5temp_open(cfg_out)
        for path_csv in gen_names_and_log(cfg_out, store, dfLogOld):
            # Loading and processing data
            d, b_ok = read_csv([path_csv], **cfg['in'])

            # filter
            if not np.all(b_ok):
                # todo: filter - check if d is bad where not b_ok
                # or dd.from_pandas(d.loc[:,list(cfg_in['dtype_out'].names)].compute()[b_ok], chunksize=cfg_in['blocksize']), tim
                pass
            d, tim = set_filterGlobal_minmax(d, cfg_out['log'], cfg['filter'])
            if cfg_out['log']['rows_filtered']:
                print('filtered out {}, remains {}'.format(cfg_out['log']['rows_filtered'], cfg_out['log']['rows']))
            elif cfg_out['log']['rows']:
                print('.', end='')  # , divisions=d.divisions), divisions=pd.date_range(tim[0], tim[-1], freq='1D')
            else:
                l.warning('no data!')
                continue

            # Save last time to can filter next file
            cfg['in']['time_last'] = tim[-1]  # date[-1]

            # if d.empty.compute(): #log['rows']==0
            #     print('No data => skip file')
            #     continue

            # Append to Store
            """
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
            """
            h5_append(store, d, cfg_out['log'], cfg_out, log_dt_from_utc=cfg['in']['dt_from_utc'], tim=tim)

            # Log to logfile
            strLog = '{fileName}:\t{Date0:%d.%m.%Y %H:%M:%S}-{DateEnd:%d. %H:%M:%S%z}\t{rows}rows'.format(
                **cfg_out['log'])  # \t{Lat}\t{Lon}\t{strOldVal}->\t{mag}
            print(strLog)
            if cfg['program']['log']:
                flog.writelines('\n' + strLog)  # + nameFE + '\t' +
    try:
        print('')
        if cfg_out['b_remove_duplicates']:
            h5_remove_duplicates(store, cfg_out, cfg_table_keys=('tables', 'tables_log'))
        # Create full indexes. Must be done because of using ptprepack in h5move_tables() below
        l.debug('Create index')
        for tblName in (cfg_out['tables'] + cfg_out['tables_log']):
            try:
                store.create_table_index(tblName, columns=['index'], kind='full')
            except Exception as e:
                l.warning(': table {}. Index not created - error'.format(tblName), '\n==> '.join(
                    [s for s in e.args if isinstance(s, str)]))
    except Exception as e:
        l.error('The end. There are error ', e.__class__, ':', '\n==> '.join(
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
        store.close()
        if cfg['program']['log']:
            flog.close()
        if store.is_open:
            print('Wait store is closing...')
            sleep(2)

        failed_storages = h5move_tables(cfg_out)
        print('Ok.', end=' ')
    h5index_sort(cfg_out, out_storage_name=cfg_out['db_base'] + '-resorted.h5', in_storages=failed_storages)


if __name__ == '__main__':
    main()

# cfg_out['db_base'] = os_path.basename(os_path.dirname(cfg_out['db_path']))


### Trash ###
"""
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
