#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Purpose:  detect date of first data in each file
@author:   Andrey Korzh <ao.korzh@gmail.com>
Created:  03.11.2016
"""

from datetime import datetime, timedelta
from io import BytesIO
from os import path as os_path
from shutil import move

import numpy as np

from to_pandas_hdf5.csv2h5 import init_input_cols
from utils2init import ini2dict, init_file_names, Ex_nothing_done, standard_error_info
from utils_time import timzone_view
from utils_time_corr import time_corr

def proc_loaded_ADCP_WH(a, cfg_in):
    pass
    # ADCP_WH specified proc
    # Time calc
    # gets string for time in current zone


# ##############################################################################
# ___________________________________________________________________________
if __name__ == '__main__':
    #    unittest.main()
    import argparse

    parser = argparse.ArgumentParser(description='Rename CSV-like files '
                                                 'according to date it contains',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     epilog='If use special characters then insert arguments in quotes',
                                     )
    parser.add_argument('--version', action='version', version='%(prog)s '
                                                               'version 0.0.1 - (c) 2016 Andrey Korzh <ao.korzh@gmail.com>')  # sourceConvert

    parser.add_argument('cfgFile', nargs='?', type=str, default='fileNameTime.ini',
                        help='Path to confiuration *.ini file with all parameters. '
                             'Next parameters here overwrites them')
    info_default_path = '[in] path from *.ini if specified'
    parser.add_argument('-path', nargs='?', type=str, default=info_default_path,
                        help='Path to source file(s) to parse')
    parser.add_argument('-verbose', nargs=1, type=str, default=['INFO'],
                        choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'],
                        help='Verbosity of messages in log file')

    args = parser.parse_args()
    args.verbose = args.verbose[0]
    # args.cfgFile= 'csv2h5_nav_supervisor.ini'
    # args.cfgFile= 'csv2h5_IdrRedas.ini'
    # args.cfgFile= 'csv2h5_Idronaut.ini'
    try:
        cfg = ini2dict(args.cfgFile)
    except FileNotFoundError:
        print('no config found')
        cfg = {'in': {'path': args.path}, 'output_files': {}, 'program': {'log': 'fileNameTime.log'}}
        if 'ADCP_WH' in cfg['in']['path']:
            cfg['in']['cfgFile'] = 'ADCP_WH'
            cfg['in']['delimiter'] = '\t'
            cfg['in']['header'] = (
                '`Ensemble #`,txtYY_M_D_h_m_s_f(text),Lat,Lon,Top,`Average Heading (degrees)`,`Average Pitch (degrees)`,'
                'stdPitch,`Average Roll (degrees)`,stdRoll,`Average Temp (degrees C)`,txtVe_none(text),txtVn_none(text),'
                'txtVup(text),txtErrVhor(text),txtInt1(text),txtInt2(text),txtInt3(text),txtInt4(text),txtCor1(text),'
                'txtCor2(text),txtCor3(text),txtCor4(text),GOOD,SpeedE_GGA,SpeedN_GGA,SpeedE_BT,SpeedN_BT,SpeedUp,ErrSpeed_BT,'
                'DepthReading,`Bin Size (m)`,`Bin 1 Distance(m;>0=up;<0=down)`,absorption,IntScale')  # .split(',')

            cfg['in']['converters'] = {1: lambda txtYY_M_D_h_m_s_f: np.datetime64(
                '20{0:02.0f}-{1:02.0f}-{2:02.0f}T{3:02.0f}:{4:02.0f}:{5:02.0f}'.format(
                    *np.fromstring(txtYY_M_D_h_m_s_f, dtype=np.uint8, count=6, sep=',')))}
            cfg['in']['dt_from_utc'] = timedelta(0)
            cfg['in']['skiprows'] = 0
            # cfg['in']['comments'],
            cfg['in']['coltime'] = 1
            cfg['in']['b_raise_on_err'] = True
    except IOError as e:
        print('\n==> '.join([s for s in e.args if isinstance(s, str)]))  # e.message
        raise (e)

    try:
        cfg['in'] = init_file_names(cfg['in'])
    except Ex_nothing_done as e:
        print(e.message)
        exit()
    # Assign castom prep&proc based on args.cfgFile name #######################
    fun_proc_loaded = None  # Assign default proc below column assinment
    # if cfg['in']['cfgFile'].endswith('ADCP_WH'):
    #    fun_proc_loaded = proc_loaded_ADCP_WH
    # Default time postload proc
    if fun_proc_loaded is None:
        if 'coldate' not in cfg['in']:  # Time includes Date
            fun_proc_loaded = lambda a, cfg_in: \
                a[cfg['in']['col_index_name']]
        else:  # Time + Date
            fun_proc_loaded = lambda a, cfg_in: a['Date'] + np.array(
                np.int32(1000 * a[cfg['in']['col_index_name']]), dtype='m8[ms]')

    # Prepare cpecific format loading and writing
    cfg['in'] = init_input_cols(cfg['in'])
    cfg['output_files']['names'] = np.array(
        cfg['in']['dtype'].names)[cfg['in']['cols_loaded_save_b']]
    cfg['output_files']['formats'] = [
        cfg['in']['dtype'].fields[n][0] for n in cfg['output_files']['names']]
    cfg['output_files']['dtype'] = np.dtype({
        'formats': cfg['output_files']['formats'],
        'names': cfg['output_files']['names']})
    cfg['output_files']['logfield_filename_len'] = 100
    # Default time postload proc
    if fun_proc_loaded is None:
        if 'coldate' not in cfg['in']:  # Time includes Date
            fun_proc_loaded = lambda a, cfg_in: \
                a[cfg['in']['col_index_name']]
        else:  # Time + Date
            fun_proc_loaded = lambda a, cfg_in: a['Date'] + np.array(
                np.int32(1000 * a[cfg['in']['col_index_name']]), dtype='m8[ms]')
    log = []
    log_item = {}
    if 'log' in cfg['program'].keys():
        f = open(cfg['program']['log'], 'a+', encoding='cp1251')
        f.writelines(datetime.now().strftime('\n\n%d.%m.%Y %H:%M:%S> processed '
                                             + str(nFiles) + ' file' + 's:' if nFiles > 1 else ':'))

    # ## Main circle ############################################################
    for ifile, nameFull in enumerate(cfg['in']['namesFull'], start=1):
        nameFE = os_path.basename(nameFull)
        log_item['fileName'] = nameFE[-cfg['output_files']['logfield_filename_len']:-4]
        log_item['fileChangeTime'] = datetime.fromtimestamp(os_path.getmtime(nameFull))
        print('{}. {}'.format(ifile, nameFE), end=': ')
        # Loading data
        with open(nameFull, 'r') as fdata:
            str = fdata.readline()
        fdata = BytesIO(str.encode())

        if 'b_raise_on_err' in cfg['in'] and not cfg['in']['b_raise_on_err']:
            try:
                a = np.genfromtxt(fdata, dtype=cfg['in']['dtype'],
                                  delimiter=cfg['in']['delimiter'],
                                  usecols=cfg['in']['cols_load'],
                                  converters=cfg['in']['converters'],
                                  skip_header=cfg['in']['skiprows'],
                                  comments=cfg['in']['comments'],
                                  invalid_raise=False)  # ,autostrip= True
                # warnings.warn("Mean of empty slice.", RuntimeWarning)
            except Exception as e:
                print(*standard_error_info(e), ' - Bad file. skip!\n')
                continue
        else:
            try:
                a = np.loadtxt(fdata, dtype=cfg['in']['dtype'],
                               delimiter=cfg['in']['delimiter'],
                               usecols=cfg['in']['cols_load'],
                               converters=cfg['in']['converters'],
                               skiprows=cfg['in']['skiprows'])
            except Exception as e:
                print('{}\n Try set [in].b_raise_on_err= False'.format(e))
                raise (e)
        # Process a and get date date in ISO standard format
        try:
            date = fun_proc_loaded(a, cfg['in'])
        except IndexError:
            print('no data!')
            continue
        # add time shift specified in configuration .ini
        date = np.atleast_1d(date)
        tim, b_ok = time_corr(date, cfg['in'], b_make_time_inc=True)
        # Save last time to can filter next file
        cfg['in']['time_last'] = date[-1]
        log_item['rows'] = 1
        log_item['Date0'] = timzone_view(tim[0], cfg['in']['dt_from_utc'])
        log_item['DateEnd'] = datetime.now()  # can not paste np.NaN
        log_item['fileNameNew'] = '{Date0:%y%m%d_%H%M}'.format(**log_item)
        log.append(log_item.copy())
        strLog = '{fileName}:\t{Date0:%d.%m.%Y %H:%M:%S}->\t{fileNameNew}.txt'.format(
            **log_item)  # \t{Lat}\t{Lon}\t{strOldVal}->\t{mag}
        print(strLog)
        if 'log' in cfg['program'].keys():  # Log to logfile
            f.writelines('\n' + strLog)
    else:
        if len(log):
            s = input('\n{} txt files. Rename _ASC.TXT, .TXT, r.000, r.000.nc? Y/n: '.format(nFiles))
            if 'n' in s or 'N' in s:
                print('nothing done')
                nFiles = 0
            else:
                print('wait... ', end='')
                for ifile, log_item in enumerate(log, start=1):
                    for str_en in ('.TXT', '_ASC.TXT', 'r.000', 'r.000.nc'):
                        if str_en != '.TXT':
                            f_in_PNE = os_path.join(cfg['in']['path'], '_RDI' + log_item['fileName'][1:] + str_en)
                        else:
                            f_in_PNE = os_path.join(cfg['in']['path'], log_item['fileName'] + str_en)
                        f_out_PNE = os_path.join(cfg['in']['path'], log_item['fileNameNew'] + str_en)
                        if os_path.isfile(f_in_PNE):
                            if os_path.isfile(f_out_PNE):
                                print('!', end='')
                            else:
                                move(f_in_PNE, f_out_PNE)
                                print('+', end='')
                        else:
                            if os_path.isfile(f_out_PNE):
                                print('_', end='')  # already renamed
                            else:
                                print('-', end='')  # no sach file

        else:
            print('"done nothing"')
    print('OK>')
    try:
        pass
    except Exception as e:
        print('The end. There are error ', standard_error_info(e))

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
        if 'log' in cfg['program'].keys():
            f.close()
        print('Ok')
