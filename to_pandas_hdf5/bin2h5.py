#!/usr/bin/env python
# coding:utf-8
"""
  Author:  Andrey Korzh --<ao.korzh@gmail.com>
  Purpose: Convert binary source files of some AB SIO RAS probes to text
  Created: 22.10.2014
"""
import logging
import re
from codecs import open, decode
from contextlib import closing
from datetime import datetime, timedelta
from mmap import ACCESS_READ, mmap
from os import path as os_path
from pathlib import PurePath
from sys import argv as sys_argv
from typing import Any, Mapping

import numpy as np
import pandas as pd

from to_pandas_hdf5.csv2h5 import init_input_cols, set_filterGlobal_minmax, h5_dispenser_and_names_gen
from to_pandas_hdf5.h5_dask_pandas import h5_append
from to_pandas_hdf5.h5toh5 import h5temp_open, h5move_tables, h5init, h5index_sort
from utils2init import my_argparser_common_part, cfg_from_args, init_logging, init_file_names, Ex_nothing_done, \
    set_field_if_no, this_prog_basename

if __name__ == '__main__':
    l = None  # see main(): l = init_logging(logging, None, cfg['program']['log'], cfg['program']['verbose'])
else:
    l = logging.getLogger(__name__)


def my_argparser():
    """
    Configuration parser
    - add here common options for different inputs
    - add help strings for them
    :return p: configargparse object of parameters
    All p argumets are of type str (default for add_argument...), because of
    custom postprocessing based of args names in ini2dict
    """
    version = '0.0.1'
    p = my_argparser_common_part({'description': 'csv2h5 version {}'.format(version) + """
----------------------------
Add data from bin files
to Pandas HDF5 store*.h5
----------------------------"""}, version=version)
    # Configuration sections
    s = p.add_argument_group('in', 'all about input files')
    s.add('--path', default='.',  # nargs=?,
             help='path to source file(s) to parse. Use patterns in Unix shell style')
    s.add('--data_word_len_integer', default='2',  # nargs=?,
             help='[bytes] => data type is int16')
    s.add('--filename2timestart_format', default='%Y%m%d_%H%M',
             help='Time from file name. For example for RealTerm v3+ writes names formatted %Y-%m-%d_%H%M')

    s = p.add_argument_group('out', 'all about output files')
    s.add('--db_path', help='hdf5 store file path')
    s.add('--table',
              help='table name in hdf5 store to write data. If not specified then will be generated on base of path of input files. Note: "*" is used to write blocks in autonumbered locations (see dask to_hdf())')
    # s.add('--tables_list',
    #           help='tables names in hdf5 store to write data (comma separated)')
    s.add('--b_insert_separator', default='True',
              help='insert NaNs row in table after each file data end')
    s.add('--b_use_old_temporary_tables', default='False',
              help='Warning! Set True only if temporary storage already have good data!'
                   'if True and b_skip_if_up_to_date= True then not replace temporary storage with current storage before adding data to the temporary storage')
    s.add('--b_remove_duplicates', default='False', help='Set True if you see warnings about')
    # 'logfield_filename_len': 255,

    s = p.add_argument_group('filter', 'filter all data based on min/max of parameters')
    s.add('--min_date', help='minimum time')  # todo: set to filt_min.key and filt_min.value
    s.add('--max_date', help='maximum time')  # todo: set to filt_max.key and filt_max.value
    s.add('--min_dict',
              help='List with items in  "key:value" format. Filter out (set to NaN) data of ``key`` columns if it is below ``value``')
    s.add('--max_dict',
              help='List with items in  "key:value" format. Filter out data of ``key`` columns if it is above ``value``')

    s = p.add_argument_group('program', 'program behaviour')
    s.add('--return', default='<end>',  # nargs=1,
               choices=['<cfg_from_args>', '<gen_names_and_log>', '<end>'],
               help='<cfg_from_args>: returns cfg based on input args only and exit, <gen_names_and_log>: execute init_input_cols() and also returns fun_proc_loaded function... - see main()')

    return (p)


# import unittest
def readFF00FF(path_in, cfg: Mapping[str, Any]):
    '''
     Read binary file with 4byte data blocks each prefixed with "0x0000FFFF"
    to uint16 matrix "out"

     path_in - mask of directory with bin files, or single file name, or
     its fid (then it must be already open).
     cfg - struct with indexes of source columns, it's names and output
    order
    '''
    sinchro_byte_st = np.uint8(0xFF)
    sinchro_byte_en = np.uint8(0)
    with open(path_in, 'rb') as f, closing(mmap(f.fileno(), 0, access=ACCESS_READ)) as mm:
        # mm length is equal to the current file size
        # Table with starts of variables (b) alined to frames
        # Frames:           FFFF  DDDD  0000  FFFF
        # Mark 1            bFFFF       b0000 bFFFF
        # Mark 2                        bE= (b0000 and next bFFFF)
        # Mark 3 (bS - not used)              bS= (bFFFF and next bFFFF)
        # Data:                   V= Data at positions from bE shifted back
        c = np.fromstring(mm, dtype=np.uint8)

        bFF = c == sinchro_byte_st
        bFFFF = np.logical_and(bFF[:-1], bFF[1:])
        b00 = c == sinchro_byte_en
        b0000 = np.logical_and(b00[:-1], b00[1:])
        bE = np.logical_and(b0000[:-2:2],
                            bFFFF[2::2])  # devide frequency x2 (can because bFFFF can not start before b0000 end)
        ind = 2 * np.nonzero(bE)[0]  # multiply frequency x2 back
        V = np.uint16(c[ind - 2]) + np.uint16(c[ind - 1]) * np.uint16(256)
        return V

    #


def readBinFramed(path_in, cfg_in: Mapping[str, Any], sinchro_words=None):
    """
    Read binary file with 4byte data blocks (#0x000000E6) each prefixed with sinchro word "sinchro_word"
    :param path_in: mask of directory with bin files, or single file name, or its fid (then it must be already open).
    :param cfg_in: struct with indexes of source columns, it's names and output order:
        delimiter_hex_list
        data_words    #3
        data_word_len #2 [bytes] => data type is int16
            Read all 16-bit Words in matrix, deletes unused columns
            adds HighWord, finished in cfg_in['']Name with '_HB', to previous Word
            to produce uint32 output
    :param sinchro_words: list of sinchro_words. sinchro_word can contain "??" instead of bytes which take place but value isn't care
    :return:
    """
    sinchro_words_n_skip = 0  # count of "?" in syncro_word before meaningful digits
    if not sinchro_words is None:
        cfg_in['delimiter_hex'] = sinchro_words
    if 'delimiter_hex' in cfg_in:
        sinchro_words = [decode(d, 'hex') for d in cfg_in['delimiter_hex']]
    else:
        sinchro_words_b = np.fromstring(cfg_in['delimiter_hex'], 'S1')
        sinchro_words_skip = sinchro_words_b == b'?'
        sinchro_words_n_skip = int(sum(sinchro_words_skip) / 2)  # each byte = 2 hex digits
        if sinchro_words_n_skip:
            sinchro_words = [decode(sinchro_words_b[~sinchro_words_skip], 'hex')]
        else:
            sinchro_words = [decode(cfg_in['delimiter_hex'], 'hex')]
    if not 'data_words' in cfg_in:
        cfg_in['data_words'] = len(cfg_in['cols'])
    data_len_bytes = cfg_in['data_words'] * cfg_in['data_word_len']
    len_sinchro_word = np.max([len(sinchro_w) for sinchro_w in sinchro_words]) + sinchro_words_n_skip
    len_frame = (data_len_bytes + len_sinchro_word)
    with open(path_in, 'rb') as f, closing(mmap(f.fileno(), 0, access=ACCESS_READ)) as mm:
        # mm length is equal to the current file size
        # Table with starts of variables (b) alined to frames
        # Frames:           FFFF  DDDD  0000  FFFF
        # Mark 1            bFFFF       b0000 bFFFF
        # Mark 2                        bE= (b0000 and next bFFFF)
        # Mark 3 (bS - not used)              bS= (bFFFF and next bFFFF)
        # Data:                   V= Data at positions from bE shifted back
        if 'remove_bad_strings' in cfg_in:
            # cfg_in['remove_bad_strings']= re.compile(b"""^\d{6}\.\d+,""") #(?<=[\r\n]|^)
            c = np.frombuffer(re.sub(bytes(cfg_in['remove_bad_strings'], 'ascii'),
                                     b'', mm), dtype=np.uint8)
        else:
            c = np.frombuffer(mm, dtype=np.uint8)
        bFound = np.zeros_like(c, np.bool)
        # np.flatnonzero(np.bitwise_xor(*[np.frombuffer(s, 'uint8') for s in sinchro_words]))
        for sinchro_w in sinchro_words:
            # Find 1st sinchro:
            char = sinchro_w[0]
            bFound_w = (c == char)
            for i, char in enumerate(sinchro_w[1:], start=1):
                bFound_w[:-i] &= (c[i:] == char)
            bFound |= bFound_w
        iEn = np.flatnonzero(bFound) - sinchro_words_n_skip  # data ends
        iSt = iEn - data_len_bytes  # data starts

        # Check ends
        if iSt[0] < 0:
            iSt = iSt[1:]
            # iEn = iEn[1:]
        iEnLast = iSt[-1] + data_len_bytes
        if (iEnLast + 1) > len(c):
            d = np.diff(iSt)
            iSt = iSt[:-1]
        else:
            d = np.diff(np.append(iSt, iEnLast + len_sinchro_word))
        #   iEn = np.append(iEn, values=len(c))

        # Check frames length
        bStartOk = d == len_frame

        # Detect frames containing data equal to sinchro_words
        # Check adjasent bad intervals which less than data frame:
        bAdj_inBad = d < len_frame
        bAdj_inBad[:-1] &= bAdj_inBad[1:]
        bAdj_inBad[-1] = False
        if np.any(bAdj_inBad):
            bAdj_inBad[:-1] |= bAdj_inBad[1:]
            bAdj_inBad = np.append(False, bAdj_inBad[:-1])
            iAdj_inBad = np.flatnonzero(bAdj_inBad)
            kPrev = iAdj_inBad[0] - 1
            s = 0
            for k, d_cur in zip(iAdj_inBad, d[bAdj_inBad]):
                kPrev += 1
                if kPrev != k:  # start cumullate sum of next group of adjasent bad frames
                    kPrev = k
                    s = 0
                    continue
                if s == 0:
                    s = d_cur
                    iFrameSt = k
                    continue
                s += d_cur
                if s > len_frame:
                    if bAdj_inBad[k + 1]:
                        # shift frame start wile sum is bigger than len_frame
                        while s > len_frame:
                            s -= d[iFrameSt]
                            iFrameSt += 1
                if s == len_frame:
                    bStartOk[iFrameSt] = True
                    s = 0

        nFrames = np.sum(bStartOk)
        if np.any(~bStartOk):
            temp = np.sum(~bStartOk)
            indOk = 100 * temp / (temp + nFrames)
            l.warning(', {:d} bad frames ({:2.2f}% of {:d} good)! - continue... '.format(temp, indOk, nFrames))

        iSt = iSt[bStartOk]  # use only frames enclosed with sinchro_words
        V = np.zeros(shape=(len(iSt),), dtype=cfg_in['dtype'])  # , cfg_in['data_words'] np.uint16
        for i, iShift in enumerate(range(0, data_len_bytes, cfg_in['data_word_len'])):
            # V[:, i] = np.uint16(c[iSt + iShift]) + np.uint16(c[iSt + 1 + iShift]) * np.uint16(256)  # Big Endian
            iWord_St = iSt + iShift
            bytes_in_word_range = range(*(cfg_in['data_word_len'],) if cfg_in['b_byte_order_is_big_endian'] else (
                cfg_in['data_word_len'] - 1, -1, -1))
            ibLast = bytes_in_word_range[-1]
            if cfg_in['b_baklan']:
                for ib in bytes_in_word_range:
                    V[cfg_in['cols'][i]] += np.bitwise_and(127, c[iWord_St + ib])
                    if ib == ibLast:
                        break
                    V[cfg_in['cols'][i]] = np.left_shift(V[cfg_in['cols'][i]], 7)
            else:
                for ib in bytes_in_word_range:
                    # V[cfg_in['cols'][i]]+=(c[iWord_St + ib]*2**(8*ib)) #.astype(cfg_in['dtype'][i]) #.view(cfg_in['dtype'])
                    V[cfg_in['cols'][i]] += c[iWord_St + ib]
                    if ib == ibLast:
                        break
                    V[cfg_in['cols'][i]] = np.left_shift(V[cfg_in['cols'][i]], 8)

        return V, iSt / len_frame


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
        cfg['in']['paths'], cfg['in']['nfiles'], cfg['in']['path'] = init_file_names(
            **cfg['in'], b_interact= cfg['program']['b_interact'])
    except Ex_nothing_done as e:
        print(e.message)
        return ()

    bOld_FF00FF = False
    # if 'TermGrunt' in sys.argv[1] FF00FF' in str(cfg['in']['path']):  # 'TermGrunt.h5'  ? args.path.endswith ('bin'):
    #     bOld_FF00FF = True
    #     cfg['in'].update({
    #     'header': 'TERM',
    #     'dt_from_utc': timedelta(hours=-1),
    #     'fs': 1, 'b_time_fromtimestamp': True,
    #     'b_time_fromtimestamp_source': False})
    # else:  # 'Katran.h5'
    #     cfg['in'].update({
    #     'delimiter_hex': '000000E6',
    #     'header': 'P, Temp, Cond',
    #     'dt_from_utc': timedelta(hours=0),
    #     'fs': 10, 'b_time_fromtimestamp': False,
    #     'b_time_fromtimestamp_source': False})

    set_field_if_no(cfg['in'], 'dtype', 'uint{:d}'.format(
        2 ** (3 + np.searchsorted(2 ** np.array([3, 4, 5, 6, 7]) >
                                  np.array(8 * (cfg['in']['data_word_len'] - 1)), 1))))

    # Prepare cpecific format loading and writing
    set_field_if_no(cfg['in'], 'coltime', [])
    cfg['in'] = init_input_cols(cfg['in'])
    cfg['out']['names'] = np.array(cfg['in']['dtype'].names)[ \
        cfg['in']['cols_loaded_save_b']]
    cfg['out']['formats'] = [cfg['in']['dtype'].fields[n][0]
                                      for n in cfg['out']['names']]
    cfg['out']['dtype'] = np.dtype({
        'formats': cfg['out']['formats'],
        'names': cfg['out']['names']})
    h5init(cfg['in'], cfg['out'])

    # cfg['Period'] = 1.0 / cfg['in']['fs']  # instead Second can use Milli / Micro / Nano:
    # cfg['pdPeriod'] = pd.to_timedelta(cfg['Period'], 's')
    # #pd.datetools.Second(cfg['Period'])\
    #     if 1 % cfg['in']['fs'] == 0 else\
    #     pd.datetools.Nano(cfg['Period'] * 1e9)

    # log table of loaded files. columns: Start time, file name, and its index in array off all loaded data:
    log_item = cfg['out']['log'] = {}  # fields will have: 'fileName': None, 'fileChangeTime': None, 'rows': 0

    strLog = ''
    # from collections import namedtuple
    # type_log_files = namedtuple('type_log_files', ['label','iStart'])
    # log.sort(axis=0, order='log_item['Date0']')#sort files by time

    df_log_old, cfg['out']['db'], cfg['out']['b_skip_if_up_to_date'] = h5temp_open(**cfg['out'])
    if 'log' in cfg['program'].keys():
        f = open(PurePath(sys_argv[0]).parent / cfg['program']['log'], 'a', encoding='cp1251')
        f.writelines(datetime.now().strftime('\n\n%d.%m.%Y %H:%M:%S> processed '
                                             + str(cfg['in']['nfiles']) + ' file' + 's:' if cfg['in'][
                                                                                                'nfiles'] > 1 else ':'))
    b_remove_duplicates = False  # normally no duplicates but will if detect
    # Config specially for readBinFramed
    set_field_if_no(cfg['in'], 'b_byte_order_is_big_endian', True)
    set_field_if_no(cfg['in'], 'b_baklan', False)
    set_field_if_no(cfg['in'], 'b_time_fromtimestamp_source', False)
    cfg['out']['fs'] = cfg['in']['fs']
    if True:
        ## Main circle ############################################################
        for i1_file, path_in in h5_dispenser_and_names_gen(cfg['in'], cfg['out']):
            l.info('{}. {}: '.format(i1_file, path_in.name))

            # Loading data
            if bOld_FF00FF:
                V = readFF00FF(path_in, cfg)
                iFrame = np.arange(len(V))
            else:
                V, iFrame = readBinFramed(path_in, cfg['in'])
            if ('b_time_fromtimestamp' in cfg['in'] and cfg['in']['b_time_fromtimestamp']) or \
                    ('b_time_fromtimestamp_source' in cfg['in'] and cfg['in']['b_time_fromtimestamp_source']):
                path_in_rec = os_path.join(
                    'd:\\workData\\_source\\BalticSea\\151021_T1Grunt_Pregol\\_source\\not_corrected',
                    os_path.basename(path_in)[:-3] + 'txt') if cfg['in']['b_time_fromtimestamp_source'
                ] else path_in
                log_item['Date0'] = datetime.fromtimestamp(os_path.getmtime(path_in_rec))  # getctime is bad
                log_item['Date0'] -= iFrame[-1] * timedelta(
                    seconds=1 / cfg['in']['fs'])  # use for computer filestamp at end of recording
            else:
                log_item['Date0'] = datetime.strptime(path_in.stem, cfg['in']['filename2timestart_format'])
            log_item['Date0'] += cfg['in']['dt_from_utc']
            tim = log_item['Date0'] + iFrame * timedelta(seconds=1 / cfg['in'][
                'fs'])  # tim = pd.date_range(log_item['Date0'], periods=np.size(V, 0), freq=cfg['pdPeriod'])
            df = pd.DataFrame(V.view(dtype=cfg['out']['dtype']),  # np.uint16
                              columns=cfg['out']['names'],
                              index=tim)
            # pd.DataFrame(V, columns=cfg['out']['names'], dtype=cfg['out']['formats'], index=tim)
            if df.empty:  # log['rows']==0
                print('No data => skip file')
                continue

            df, tim = set_filterGlobal_minmax(df, cfg_filter=cfg['filter'], log=log_item,
                                              dict_to_save_last_time=cfg['in'])
            if log_item['rows_filtered']:
                print('filtered out {}, remains {}'.format(log_item['rows_filtered'], log_item['rows']))
            if not log_item['rows']:
                l.warning('no data! => skip file')
                continue
            elif log_item['rows']:
                print('.', end='')  # , divisions=d.divisions), divisions=pd.date_range(tim[0], tim[-1], freq='1D')
            else:
                l.warning('no data! => skip file')
                continue

            # Append to Store
            h5_append(cfg['out'], df.astype('int32'), log_item)

            if 'txt' in cfg['program'].keys():  # can be saved as text too
                np.savetxt(cfg['program']['txt'], V, delimiter='\t', newline='\n',
                           header=cfg['in']['header'] + log_item['fileName'], fmt='%d',
                           comments='')

    try:
        if b_remove_duplicates:
            for tblName in (cfg['out']['table'] + cfg['out']['tableLog_names']):
                cfg['out']['db'][tblName].drop_duplicates(keep='last', inplace=True)  # subset='fileName',?
        if len(strLog):
            print('Create index', end=', ')
            for tblName in (cfg['out']['table'] + cfg['out']['tableLog_names']):
                cfg['out']['db'].create_table_index(tblName, columns=['index'], kind='full')
        else:
            print('done nothing')
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
    # sort index if have any processed data (needed because ``ptprepack`` not closses hdf5 source if it not finds data)
    if cfg['in'].get('time_last'):
        failed_storages = h5move_tables(cfg['out'])
        print('Ok.', end=' ')
        h5index_sort(cfg['out'], out_storage_name=f"{cfg['out']['db_path'].stem}-resorted.h5", in_storages=failed_storages)


if __name__ == '__main__':
    main()

"""



                if cfg_out['b_insert_separator']:
                    # insert separator # 0 (can not use np.nan in int) [tim[-1].to_datetime() + timedelta(seconds = 0.5/cfg['in']['fs'])]
                    df = df.append(pd.DataFrame(0, columns=cfg_out['names'], index=[
                        tim[-1] + timedelta(seconds=0.5 / cfg['in']['fs'])])) #.to_datetime()



    dfLog = pd.DataFrame.from_records(log, exclude=['Date0'], index=['Date0'])
    store.append(cfg_out['tableLog_names'][0], dfLog, data_columns=True,
                 expectedrows=cfg['in']['nfiles'] + 1)  # append
    store.create_table_index(cfg_out['table'], columns=['index'], kind='full')
else:



                store.append(cfg_out['table'], df.astype('int32'), data_columns=True, index=False)
                # [df.columns] df.columns #ignore_index=True, chunksize= cfg['dtRec']/dt64i,
                log.append(log_item.copy())



#log_dtype = [('Date0', 'O'), ('fileName', '<U255'), ('iStart', '<u4'), ('fileChangeTime', 'O')]
# log_dtype[1]= ('fileName', '<U' + str(cfg_out['logfield_filename_len']))
# log= np.array(log, dtype=log_dtype)  # np.array(os_path.getmtime(path_in)).astype('datetime64[s]')


    if False:
        # Find 1st sinchro:
        d1st_sinch = mm.find(sinchro_words)
        c = np.frombuffer(mm, dtype=np.uint8, offset= d1st_sinch) #fromstring
        char = sinchro_words[0]  # np.fromstring(, dtype=np.uint8) #ord(sinchro_words[0])
        bFound = c == char
        for i, char in enumerate(sinchro_words[1:], start=1):  # np.fromstring(, dtype=np.uint8)
            bFound = np.logical_and(bFound[:-1], c[i:] == char)



    a1 = np.fromstring(mm[16:30])


indStart= [1; uint32(find(bStart))];
bStartOk= diff(indStart)==100;
nFrames= sum(bStartOk);
if any(~bStartOk):
    temp= sum(~bStartOk);  indOk= 100*temp/(temp+nFrames);
    fprintf(', %d bad frames (%2.2f% of %d good)! - skip... ', temp, indOk, nFrames);

"""
"""
bHB= [];
if nargin<2:
    iOutCols= 1
elif isstruct(cfg):
    if isfield(cfg, 'i')
        iOutCols= cfg['i'];

    if 'Name' in cfg:
        bHBout= cellfun(@(x) strncmp(fliplr(x),'BH_', 3), cfg['Name']);
        k= find(bHBout);
        bHB= false(size(cfg['']iperm));
        for k= k; bHB= bHB|(cfg['']iperm==k); end


iuseBytes= uint32(repmat(int32([-2;-1;-0]),1, numel(iOutCols))+repmat(iOutCols*3, 3, 1));
indOk= repmat(indStart(bStartOk),1,numel(iuseBytes))+repmat(iuseBytes(:)', nFrames,1);
% add2end= [];
% if indOk(end)>numel(mm) %dStart(end)>1 %~bStartOk(end)&&
%     add2end= [mm(indOk(end,1):end); NaN(size((numel(mm)+1):indOk(end)))'];
%     indOk(end,:)= [];
% end
% mm= [; add2end']; %dec2bin(mm(1:3))

%Use maskData  = np.uint8([  3; 127; 127]);
mm= bitand(mm(indOk), np.uint8(127)); %mm(:,1:3:end)= bitand(mm(:,1:3:end), 3);
clear('indOk', 'iuseBytes', 'bStart','indStart','bStartOk')
mm= reshape(mm', 3, numel(mm)/3);
out= bitand(mm(2,:), np.uint8(1));
mm(2,:)= bitshift(mm(2,:), -1);
out= bitshift(out, 7);
mm(3,:)= bitor(mm(3,:), out);
out= bitshift(uint16(mm(1,:)), 14);
out= bitor(out, bitshift(uint16(mm(2,:)), 8));
out= bitor(out, uint16(mm(3,:)));
out= uint32(reshape(out, numel(iOutCols), nFrames)');
if any(bHB)
  str= fieldnames(cfg['']iName);
  bStart= cfg['']iperm~=find(strcmpi(str,'counts_HB')); %'
  nFrames= find(bHB&bStart);
  for iOutCols= nFrames; %del "0" in HB and LB (iOutCols = temporary var.)
    bBad= out(:, iOutCols)==0;
    if any(bBad)
      %interp1(find(~bBad), single(out(~bBad, iOutCols)), find(bBad),
      %'nearest', 'extrap');
      if any(~bBad)
        %correct HB(HB==0):
        if(~strcmpi(str{cfg['']iperm(iOutCols)},'C_HB'))
          out(:, iOutCols)= uint32(rep2previous(single(out(:,iOutCols)), bBad));
        end
        %correct LB(LB==0 & old HB==0):
        bBad= bBad&(out(:, iOutCols-1)==0);
        out(:, iOutCols-1)= uint32(rep2previous(single(out(:,iOutCols-1)), bBad));
      else
        fprintf(1, '\nAll zeros in %s! - skip... ', ...
          cfg['']Name{cfg['']iperm(iOutCols)});
      end
    end
    %
  end
  mm=   bitshift(out(:, bHB), 16);
  out(:,bHB(2:end))= out(:,bHB(2:end)) + mm; %add HighWord to previous Word
  out(:,bHB)= [];
  cfg['']iperm(bHB)= [];
  out(:, cfg['']iperm)= out;
  str(bHBout)= [];
  cfg['']iName= cell2struct(num2cell(1:numel(str)), str, 2);
  cfg['']Name(bHBout)= [];
"""
