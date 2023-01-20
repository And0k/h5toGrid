#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Purpose:  detect date of first data in each file
@author:   Andrey Korzh <ao.korzh@gmail.com>
Created:  03.11.2016
"""

from datetime import datetime, timedelta
from io import BytesIO
import os
from shutil import move

import numpy as np

from to_pandas_hdf5.csv2h5 import init_input_cols
from utils2init import ini2dict, init_file_names, Ex_nothing_done, standard_error_info, my_argparser_common_part, cfg_from_args
from utils_time import timzone_view
from utils_time_corr import time_corr

def proc_loaded_ADCP_WH(a, cfg_in):
    pass
    # ADCP_WH specified proc
    # Time calc
    # gets string for time in current zone


# ##############################################################################
# ___________________________________________________________________________

version = '0.1.0'


def my_argparser():
    """
    Configuration parser
    - add here common options for different inputs
    - add help strings for them
    :return p: configargparse object of parameters
    All p arguments are of type str (default for add_argument...), because of
    custom postprocessing based of args names in ini2dict
    """

    p = my_argparser_common_part({'description': 'csv2h5 version {}'.format(version) + """
----------------------------
Rename CSV-like files 
according to date it contains
----------------------------"""}, version)
    # Configuration sections
    s = p.add_argument_group('in', 'all about input files')
    s.add('--path', default='.',  # nargs=?,
          help='path to source file(s) to parse. Use patterns in Unix shell style')

    s = p.add_argument_group('out',
                             'all about output files')

    s = p.add_argument_group('program',
                             'program behaviour')
    s.add('--return', default='<end>',  # nargs=1,
               choices=['<cfg_from_args>', '<gen_names_and_log>', '<end>'],
            help='<cfg_from_args>: returns cfg based on input args only and exit, <gen_names_and_log>: execute init_input_cols() and also returns fun_proc_loaded function... - see main()')

    return p


def main(new_arg=None, **kwargs):


    # args.cfgFile= 'csv2h5_nav_supervisor.ini'
    # args.cfgFile= 'csv2h5_IdrRedas.ini'
    # args.cfgFile= 'csv2h5_Idronaut.ini'
    try:
        cfg = cfg_from_args(my_argparser(), new_arg, **kwargs)
    except FileNotFoundError:
        print('no config found')
        cfg = {'in': {'path': args.path}, 'out': {}, 'program': {'log': 'fileNameTime.log'}}
        if 'ADCP_WH' in cfg['in']['path']:
            cfg['in']['cfgFile'] = 'ADCP_WH'
            cfg['in']['delimiter'] = '\t'
            cfg['in']['header'] = (
                '`Ensemble #`,txtYY_M_D_h_m_s_f(text),Lat,Lon,Top,`Average Heading (degrees)`,`Average Pitch (degrees)`,'
                'stdPitch,`Average Roll (degrees)`,stdRoll,`Average Temp (degrees C)`,txtu_none(text),txtv_none(text),'
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
            cfg['in']['on_bad_lines'] = 'error'
            cfg['in']['comments'] = None
    except IOError as e:
        print('\n==> '.join([s for s in e.args if isinstance(s, str)]))  # e.message
        raise(e)

    if not cfg or not cfg['program'].get('return'):
        print('Can not initialise')
        return cfg
    elif cfg['program']['return'] == '<cfg_from_args>':  # to help testing
        return cfg

    try:
        cfg['in']['paths'], cfg['in']['nfiles'], cfg['in']['path'] = init_file_names(
            **cfg['in'], b_interact=cfg['program']['b_interact'])
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
    cfg['out']['names'] = np.array(
        cfg['in']['dtype'].names)[cfg['in']['cols_loaded_save_b']]
    cfg['out']['formats'] = [
        cfg['in']['dtype'].fields[n][0] for n in cfg['out']['names']]
    cfg['out']['dtype'] = np.dtype({
        'formats': cfg['out']['formats'],
        'names': cfg['out']['names']})
    cfg['out']['logfield_filename_len'] = 100
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
        f.writelines(datetime.now().strftime('\n\n%d.%m.%Y %H:%M:%S> processed ' + f"{cfg['in']['nfiles']} file" +
                                             's:' if cfg['in']['nfiles'] > 1 else ':'))

    # ## Main circle ############################################################
    dt_sum = 0
    for ifile, nameFull in enumerate(cfg['in']['paths'], start=1):
        nameFE = os.path.basename(nameFull)
        log_item['fileName'] = nameFE[-cfg['out']['logfield_filename_len']:-4]
        log_item['fileChangeTime'] = datetime.fromtimestamp(os.path.getmtime(nameFull))
        print('{}. {}'.format(ifile, nameFE), end=': ')
        # Loading data
        with nameFull.open('rb') as fdata:  # nameFull.open('rb')
            lines = fdata.readline()
            found_content = False
            try:  # catch OSError in case of a one line file
                fdata.seek(-2, os.SEEK_END)
                while True:
                    c = fdata.read(1)
                    if not c.isspace():
                        found_content = True
                    if found_content and c == b'\n':
                        if found_content:
                            break
                    fdata.seek(-2, os.SEEK_CUR)
                lines += fdata.readline()
            except OSError:
                if not found_content:
                    print('one line!')


        fdata = BytesIO(lines)
        #lines = lines.decode()
        if 'on_bad_lines' in cfg['in'] and cfg['in']['on_bad_lines'] != 'error':
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
                print('{}\n Try set [in].on_bad_lines = "warn"'.format(e))
                raise (e)
        # Process a and get date date in ISO standard format
        try:
            date = fun_proc_loaded(a, cfg['in'])
        except IndexError:
            print('no data!')
            continue

        # add time shift specified in configuration .ini
        date = np.atleast_1d(date)

        dt = np.subtract(*date[[-1,0]])
        dt_sum += dt

        tim, b_ok = time_corr(date, cfg['in'], process=True)
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
            s = input('\n{} txt files. Rename _ASC.TXT, .TXT, r.000, r.000.nc? Y/n: '.format(cfg['in']['nfiles']))
            if 'n' in s or 'N' in s:
                print('nothing done')
                nFiles = 0
            else:
                print('wait... ', end='')
                for ifile, log_item in enumerate(log, start=1):
                    for str_en in ('.TXT', '_ASC.TXT', 'r.000', 'r.000.nc'):
                        if str_en != '.TXT':
                            f_in_PNE = os.path.join(cfg['in']['path'], '_RDI' + log_item['fileName'][1:] + str_en)
                        else:
                            f_in_PNE = os.path.join(cfg['in']['path'], log_item['fileName'] + str_en)
                        f_out_PNE = os.path.join(cfg['in']['path'], log_item['fileNameNew'] + str_en)
                        if os.path.isfile(f_in_PNE):
                            if os.path.isfile(f_out_PNE):
                                print('!', end='')
                            else:
                                move(f_in_PNE, f_out_PNE)
                                print('+', end='')
                        else:
                            if os.path.isfile(f_out_PNE):
                                print('_', end='')  # already renamed
                            else:
                                print('-', end='')  # no sach file

        else:
            print('"done nothing"')
    print(f'sum(dt)={dt_sum} OK>')
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


if __name__ == '__main__':
    main()