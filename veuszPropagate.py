#!/usr/bin/env python
# coding:utf-8
# from __future__ import print_function, division
"""
  Author:  Andrey Korzh <ao.korzh@gmail.com>
  Purpose: update Veusz pattern by means of Veusz commands iterating over
  list of data files. Saves changed vsz files and export images.
  
  todo: Use hdf5 store to save dates of last modification of processed files 
  to have the ability not process twice same data on next calls.
  
  Created: 02.09.2016
"""
import logging
import re
from datetime import datetime
from os import chdir as os_chdir, getcwd as os_getcwd, environ as os_environ
from pathlib import Path, PurePath
from sys import platform as sys_platform, stdout as sys_stdout
from time import sleep
from typing import Any, Callable, Dict, Iterator, Optional, Tuple, Union

import pandas as pd
from dateutil.tz import tzlocal, tzoffset

from to_pandas_hdf5.h5toh5 import h5log_names_gen, h5find_tables
# my
from utils2init import my_argparser_common_part, cfg_from_args, this_prog_basename, init_file_names, dir_from_cfg, \
    init_logging, Ex_nothing_done, import_file

# Globals
to_mytz_offset = tzoffset(None, -tzlocal()._dst_offset.total_seconds())
# veusz = None  # 'veusz' variable must be corrected later
load_vsz = None  # must be corrected later

if __name__ != '__main__':
    l = logging.getLogger(__name__)
else:
    l = None  # will set in main()


def my_argparser():
    """
    Configuration parser
    - add here common options for different inputs
    - add help strings for them
    :return p: configargparse object of parameters
    """
    version = '0.1.0'
    p = my_argparser_common_part({'description': 'veuszPropagate version {}'.format(version) + """
----------------------------
Create vsz file for each source
file based on vsz pattern
----------------------------"""}, version)

    p_in = p.add_argument_group('in', 'data')
    p_in.add('--path',
             help='path to source file(s) to generate list of their names (usually *.csv or *.txt) or hdf5 store')
    p_in.add('--pattern_path',
             help='path to ".vsz" file to use as pattern')  # '*.h5'
    p_in.add('--import_method',
             help='Veusz method to imort data in ".vsz" pattern')  # todo: read it from pattern
    p_in.add('--start_file_index', default="0",
             help='indexes begins from 0')
    p_in.add('--add_custom_list',
             help='custom definitions names for evaluation of expressions defined in add_custom_expressions_list')
    p_in.add('--add_custom_expressions_list',
             help='custom_expressions_list to add by Veusz AddCustom() function')
    p_in.add('--eval_list',
             help='string represented Veusz.Embed function call to eval')
    p_in.add('--data_yield_prefix',
             help='used to get data from Vieusz which names started from this')
    p_in.add('--tables_list',
             help='path to tables in db to find instead files')
    p_in.add('--table_log',
             help='name of log table - path to hdf5 table having intervals ("index" of type pd.DatetimeIndex and "DateEnd" of type pd.Datetime)')
    p_in.add('--min_time', help='%%Y-%%m-%%dT%%H:%%M:%%S, optional, allows range table_log rows')
    p_in.add('--max_time', help='%%Y-%%m-%%dT%%H:%%M:%%S, optional, allows range table_log rows')

    p_out = p.add_argument_group('output_files', 'all about output files')
    p_out.add('--export_pages_int_list', default='0',
              help='pages numbers to export, comma separated (1 is first), 0= all')
    p_out.add('--b_images_only', default='False',
              help='export only. If true then all output vsz must exist, they will be loaded and vsz not be updated')
    p_out.add('--b_update_existed', default='False',
              help='replace all existed vsz files else skip existed files')
    p_out.add('--export_dir', default='images(vsz)',
              help='subdir relative to input path or absolute path to export images')
    p_out.add('--export_format', default='jpg',
              help='extention of images to export which defines format')
    p_out.add('--export_dpi_int_list', default='300',
              help='resolution (dpi) of images to export for all pages, defined in `export_pages_int_list`')
    p_out.add('--filename_fun', default='lambda tbl: tbl',
              help='function to modify output file name. Argument is input table name in hdf5')
    p_out.add('--add_to_filename', default='',
              help='string will be appended to output filenames. If input is from hdf5 table then filename is name of table, this will be added to it')

    # candidates to move out to common part
    p_in.add('--exclude_dirs_ends_with_list', default='-, bad, test, TEST, toDel-',
             help='exclude dirs wich ends with this srings. This and next option especially useful when search recursively in many dirs')
    p_in.add('--exclude_files_ends_with_list', default='coef.txt, -.txt, test.txt',
             help='exclude files wich ends with this srings')

    p_prog = p.add_argument_group('program', 'program behaviour')
    p_prog.add('--export_timeout_s_float', default='0',
               help='export asyncroniously with this timeout, s (tryed 600s?)')
    p_prog.add('--veusz_path',
               default=u'C:\\Program Files (x86)\\Veusz' if sys_platform == 'win32' else '/home/korzh/.local/lib/python3.6/site-packages/veusz',
               # '/usr/lib64/python3.6/site-packages/veusz-2.1.1-py3.6-linux-x86_64.egg/veusz', # os_environ['PATH']
               help='directory of Veusz')
    p_prog.add('--before_next_list', default=',',
               help=''' "Close()" - each time reopens pattern,
    "restore_config" - saves and restores initial configuration (may be changed in data_yield mode: see data_yield_prefix argument)''')
    p_prog.add('--f_custom_in_cycle',
               help='''function evaluated in cycle: not implemented over command line''')  # todo: implement
    p_prog.add('--return', default='<end>',  # nargs=1,
               choices=['<cfg_from_args>', '<gen_names_and_log>', '<embedded_object>', '<end>'],
               help='<cfg_from_args>: returns cfg based on input args only and exit, <gen_names_and_log>: execute init_input_cols() and also returns fun_proc_loaded function... - see main()')

    return p


# ----------------------------------------------------------------------

def veusz_data(veusze, prefix: str, suffix_prior: str = '') -> Dict[str, Any]:
    """
    Get data, loaded into the Veusz document filtered by prefix and suffix_prior
    :param veusze: Veusz embedded object
    :param prefix: string, include in output only datasets which name starts with this
    :param suffix_prior: string, if several datasets filtered by prefix diff only with this suffix, out only thouse which name ends with this, but rename keys to exclude (to be without) this suffix
    Returns: dict with found dataset wich keys are its names excluding prefix and suffix
    """
    names = veusze.GetDatasets()
    prefixlen = len(prefix)

    names_filt = dict()
    names_to_check_on_step2 = dict()

    # remove versions and keep which ends with suffix_prior
    # step1: separate max priority fields (names_filt) and others (names_to_check_on_step2)
    for name in names:
        if name.startswith(prefix):
            name_out = name if (len(name) == prefixlen) else name[prefixlen:].lstrip('_').split("_")[0]
            if name.endswith(suffix_prior):
                names_filt[name_out] = name
            else:
                names_to_check_on_step2[name_out] = name
    # step2: append other fields to names_filt if not exist
    msg_names_skip = []
    if len(names_to_check_on_step2.keys()):
        for name_out, name in names_to_check_on_step2.items():
            if not name_out in names_filt:
                names_filt[name_out] = name
            else:
                msg_names_skip.append(name)
    if msg_names_skip:
        msg_names_skip = 'skip {} fields: '.format(prefix) + ','.join(msg_names_skip)
    else:
        msg_names_skip = ''
    l.debug('\n'.join([msg_names_skip, ' load fields: {}'.format(names_filt)]))

    vsz_data = dict([(name_out, veusze.GetData(name)[0]) for name_out, name in names_filt.items()])
    if ('time' in vsz_data) and len(vsz_data['time']):
        vsz_data['time'] = ((vsz_data['time'] + (1230768000 - 7200)) * 1E+9).astype('datetime64[ns]')
        # Convert time back to UTC:
        # Veusz imports only naive times. Python think it is local, but really it is UTC.
        vsz_data['time'] = pd.to_datetime(vsz_data['time']).tz_localize(to_mytz_offset).tz_convert('UTC')

        vsz_data['starts'] = vsz_data['starts'].astype('int32')
        vsz_data['ends'] = vsz_data['ends'].astype('int32')
    return vsz_data


def load_vsz_closure(veusz_path: PurePath) -> Callable[
    [Union[str, PurePath], Optional[str], Optional[str], Optional[str]], Tuple[Any, Optional[Dict[str, Any]]]]:
    """
    See load_vsz inside
    :param veusz_path: pathlib Path to directory of embed.py
    """

    # def import_veusz(veusz_path=u'C:\\Program Files (x86)\\Veusz'):
    #     if not os_path.isdir(veusz_path):
    #         return  # 'veusz' variable must be coorected later
    #     veusz_parent_dir, veusz_dir_name = os_path.split(veusz_path)
    #     sys_path.append(os_path.dirname(veusz_parent_dir))
    #     # import Veusz.embed as veusz  #
    #     import_file(full_name, path):
    #     #importlib.import_module('.embed', package=veusz_dir_name)
    #     sys_path.pop()
    #
    #     sep = ';' if sys_platform == 'win32' else ':'
    #     os_environ['PATH'] += sep + veusz_path
    #     return
    # not works:
    # sys_path.append(cfg['program']['veusz_path'])
    # sys_path.append(os_path.dirname(cfg['program']['veusz_path']))

    sep = ';' if sys_platform == 'win32' else ':'
    # to find Veusz executable (Windows only):
    os_environ['PATH'] += f'{sep}{veusz_path}'

    # for Linux set in ./bash_profile if not in Python path yet:
    # PYTHONPATH=$PYTHONPATH:$HOME/Python/other_sources/veusz
    # export PYTHONPATH
    # if you compile Veusz also may be you add there
    # VEUSZ_RESOURCE_DIR=/usr/share/veusz
    # export VEUSZ_RESOURCE_DIR

    veusz = import_file(veusz_path, 'embed')

    # sys_path.append(os_path.dirname(cfg['program']['veusz_path']))

    def load_vsz(vsz: Union[str, PurePath, None] = None,
                 veusze: Optional[str] = None,
                 prefix: Optional[str] = None,
                 suffix_prior: Optional[str] = '_fbot') -> Tuple[veusz.Embedded, Optional[Dict[str, Any]]]:
        """
        Load (create) specifid data from '*.vsz' files
        :param vsz: full name of vsz or None. If not None and such file not found then create it
        :param veusze: veusz.Embedded object or None - will be created if None else reused
        :param prefix: only data started with this prefix will be loaded
        :param suffix_prior: high priority names suffix, removes other version of data if starts same but with no such suffix (see veusz_data())
        :return: (vsz_data, veusze) if prefix is not none else only veusze
                  vsz_data - data loaded,
                  veusze - veusz.Embedded object
        """
        if vsz is None:
            file_exists = False
            if veusze is None:
                title = 'empty'
                l.debug('new embedded window')
            else:
                l.debug('keep same embedded window')
        else:  # isinstance(vsz, (str, PurePath)):
            file_exists = Path(vsz).is_file()
            if file_exists:
                l.debug(f'loading found vsz: {vsz}')
                title = f'{vsz} - was found'
            else:
                l.debug(f'creatig vsz: {vsz}')
                title = f'{vsz} - was created'

        if veusze is None:  # construct a Veusz embedded window
            veusze = veusz.Embedded(title)
            # veusze.EnableToolbar()
            # veusze.Zoom('page')

        if file_exists:
            veusze.Load(str(vsz))
            sleep(1)

        if prefix is None:
            return veusze, None
        return veusze, veusz_data(veusze, prefix, suffix_prior)

    return load_vsz


def export_images(veusze, cfg_out, suffix, b_skip_if_exists=False):
    """

    :param veusze: Veusz embedded object
    :param cfg_out: must have fields 'export_pages' and 'export_dir': see command line arguments help
    :param suffix:
    :return:

    export_images(veusze, cfg['output_files'], log['out_name'])
    """
    # Export images
    if not cfg_out['export_pages']:
        return
    l.debug('exporting %s %s images:', len(cfg_out['export_pages']), cfg_out['export_format'])
    dpi_list = cfg_out.get('export_dpi', [300])
    for i, key in enumerate(veusze.GetChildren(where='/'), start=1):
        if cfg_out['export_pages'][0] == 0 or (  # = b_export_all_pages
                i in cfg_out['export_pages']):
            file_name = Path(cfg_out['export_dir']) / (key + suffix + '.' + cfg_out['export_format'])  # '.png'
            if veusze.Get(key + '/hide') or (b_skip_if_exists and file_name.is_file()):
                continue
            # Veusz can not export too big images - reduce
            dpi = dpi_list[min(i, len(dpi_list) - 1)]
            if int(re.sub('[^\d]', '', veusze.Get(key + '/width'))) > 400 and dpi > 200:
                dpi = 200

            try:
                veusze.Export(str(file_name), page=i - 1, dpi=dpi)
            except Exception as e:
                l.error('Exporting error %s: %s', e.__class__, '\n==> '.join([a for a in e.args if isinstance(a, str)]))
            l.debug('%s,', i)


try:
    import async_timeout
    import asyncio


    def force_async(fn):
        '''
        turns a sync function to async function using threads
        :param sync function:
        :return async funciton:

        run a sync task in a thread (returning a future object, coroutine.Future) and then turn this future object into a asyncio.Future object. Obivously, this future object will be resolve when the sync function is done, and also is awaitable.
        '''
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor()

        def wrapper(*args, **kwargs):
            future = pool.submit(fn, *args, **kwargs)
            return asyncio.wrap_future(future)  # make it awaitable

        return wrapper


    @force_async
    def export_images_a(veusze, cfg_out, suffix, b_skip_if_exists=False):
        export_images(veusze, cfg_out, suffix, b_skip_if_exists)


    async def export_images_timed(veusze, cfg, suffix, b_skip_if_exists=False) -> bool:
        """
        Asyncronous export_images(...)
        :param veusze:
        :param cfg: must have ['async']['export_timeout_s'] to skip on timeout, dict 'output_files' to call export_images(...)
        :param suffix:
        :param b_skip_if_exists:
        :return:
        """
        async with async_timeout.timeout(cfg['async']['export_timeout_s']) as cm:
            export_images_a(veusze, cfg['output_files'], suffix, b_skip_if_exists)
        if cm.expired:
            l.warning('timeout on exporting. going to next file')
        return cm.expired

except Exception as e:
    export_images_timed = None


def veusze_commands(veusze, cfg_in, file_name_r):
    """
    Execute Vieusz commands specified by special cfg_in fields
    :param veusze:
    :param cfg_in: dict, modify veusze if following fields specified:
        'add_custom' and 'add_custom_expressions' - lists of equal size
        'import_method', string - one of: 'ImportFile', 'ImportFileCSV' to import file_name_r file
        'eval', list of strings - patterns modified by format_map(cfg_in). file_name_r can be referred as cfg_in['nameRFE']
    :param file_name_r: relative name of data file
    :return:
    """
    if cfg_in is None:
        return
    if 'import_method' in cfg_in:
        if cfg_in['import_method'] == 'ImportFile':
            # veusze.call('ImportFile')
            veusze.ImportFile(f'{file_name_r}', cfg_in['c'], encoding='ascii', ignoretext=True, linked=True,
                              prefix=u'_')
            # veusze.ImportFile(f'{file_name_r}', u'`date`,Tim(time),Pres(float),Temp(float),Cond(float),'+
            # 'Sal(float),O2(float),O2ppm(floatadd_custom_expressions),pH(float),Eh(float),Turb(float)',
            #            encoding='ascii', ignoretext=True, linked=True, prefix=u'_')
        elif cfg_in['import_method'] == 'ImportFileCSV':
            veusze.ImportFileCSV(f'{file_name_r}', blanksaredata=True, dateformat=u'DD/MM/YYYY hh:mm:s',
                                 delimiter=b'\t', encoding='ascii', headermode='1st', linked=True, dsprefix=u'_',
                                 skipwhitespace=True)
    if ('add_custom_expressions' in cfg_in) and cfg_in['add_custom_expressions']:
        for name, expr in zip(cfg_in['add_custom'], cfg_in['add_custom_expressions']):
            veusze.AddCustom('definition', name, expr.format_map(cfg_in).strip(), mode='replace')

    if ('eval' in cfg_in) and cfg_in['eval']:
        cfg_in['nameRFE'] = file_name_r
        for ev in cfg_in['eval']:
            eval_str = ev.format_map(cfg_in).strip()
            if __debug__:
                l.debug('eval: ' + eval_str)
            try:
                eval("veusze." + eval_str)  # compile(, '', 'eval') or [], 'eval')
            except Exception as e:
                l.error('error to eval "{}" - {}: {}'.format(eval_str, e.__class__,
                                                             '\n==> '.join([a for a in e.args if isinstance(a, str)])))
    # veusze.AddCustom('constant', u'fileDataSource', f"u'{file_name_r}'", mode='replace')


def load_to_veusz(in_fulls, cfg, veusze=None):
    """
    Generate Veusz embedded instances by opening vsz-file(s) and modify it by executing commands specified in cfg
    :param in_full: full name of source data file to load in veusz pattern (usually csv)
    :param cfg: dict with keys:
        output_files:
            dir, namesFull - pattern path
            b_images_only, b_update_existed - command line arguments - see my_argparser()
        in:
            before_next - modify Veusz pattern data by execute Veusz commands if have fields:
                'Close()' - reopen same pattern each cycle

            import_method, header if import_method == 'ImportFile',
            add_custom, add_custom_expressions - Veusz Castom Definitions
            eval - any Veusz command
    :param veusze: Veusz embedded object. If it is None creates new Veusz embedded object loading pattern path
    :yields (veusze, log)
        veusze: Veusz embedded object.
        log: dict, {'out_name': inF, 'out_vsz_full': out_vsz_full}

    Note 1: Uses global load_vsz function if veusze = None or cfg['output_files']['b_images_only']
    which is defined by call load_vsz_closure()
    Note 2: If 'restore_config' in cfg['program']['before_next'] then sets cfg['in']= cfg['in_saved']
    """

    filename_fun = eval(compile(cfg['output_files']['filename_fun'], '', 'eval'))
    ifile = 0
    for in_full in in_fulls:
        ifile += 1
        in_full = Path(in_full)

        in_ext = in_full.suffix.lower()
        out_name = filename_fun(in_full.stem) + cfg['output_files']['add_to_filename']
        out_vsz_full = (Path(cfg['output_files']['dir']) / out_name).with_suffix('.vsz')

        # if ifile < cfg['in']['start_file']:
        #     continue

        # skip existed vsz if need
        if (not cfg['output_files']['b_images_only'] and
                (not cfg['output_files']['b_update_existed']) and
                out_vsz_full.is_file()):
            # yield (None, None)
            continue
        if in_full.stem != out_vsz_full.stem:
            l.info('%d. %s -> %s, ', ifile, in_full.name, out_vsz_full.name)
        else:
            l.info('%d. %s, ', ifile, in_full.name)
        sys_stdout.flush()
        log = {'out_name': out_name, 'out_vsz_full': out_vsz_full}

        if veusze:
            try:
                b_closed = veusze.IsClosed()
            except Exception as e:
                l.error('IsClosed() error {}: {}'.format(
                    e.__class__, '\n==> '.join([a for a in e.args if isinstance(a, str)])))
                b_closed = True
            if b_closed:
                veusze = None
            elif 'Close()' in cfg['program']['before_next'] or in_ext == '.vsz':
                veusze.Close()
                veusze = None

        # load same filePattern (last in list) if data file not "vsz"
        vsz_load = in_full if in_ext == '.vsz' else cfg['output_files']['namesFull'][-1]

        if cfg['output_files']['b_images_only']:
            veusze = load_vsz(vsz_load, veusze)[0]  # veusze.Load(in_full.with_suffix('.vsz'))
        else:
            if 'restore_config' in cfg['program']['before_next']:
                cfg['in'] = cfg['in_saved'].copy()  # useful if need restore add_custom_expressions?
            if not veusze:
                veusze = load_vsz(vsz_load)[0]  # , veusze=veusze
                if cfg['program']['verbose'] == 'DEBUG':
                    veusze.SetVerbose()  # nothing changes
            # Relative path from new vsz to data, such as u'txt/160813_0010.txt'
            try:
                file_name_r = in_full.relative_to(cfg['output_files']['dir'])
            except ValueError as e:
                # l.exception('path not related to pattern')
                file_name_r = in_full
            veusze_commands(veusze, cfg['in'], file_name_r)
        yield (veusze, log)


# -----------------------------------------------------------------------

def ge_names(cfg, f_mod_name=lambda x: x):
    """
    Yield all full file names
    :param cfg: dict with field ['in']['namesFull'], - list of parameters for f_mod_name
    :param f_mod_name: function(filename:str) returns new full file name: str
    :yields: f_mod_name(cfg['in']['namesFull'] elements)
    """
    for name in cfg['in']['namesFull']:
        yield f_mod_name(name)


def ge_names_from_hdf5_paths(cfg, f_file_name=lambda x: x):
    """
    Replasing for veuszPropagate.ge_names() to use tables instead files
    :param cfg: dict with field ['in']['tables'], - list of tables or list with regular expression path to find tables
    :return:
    """
    with pd.HDFStore(cfg['in']['path'], mode='r') as store:
        if len(cfg['in']['tables']) == 1:
            cfg['in']['tables'] = h5find_tables(store, cfg['in']['tables'][0])
    for tbl in cfg['in']['tables']:
        yield f_file_name(tbl)


def co_savings(cfg: Dict[str, Any]) -> Iterator[None]:
    """
    Saves vsz, exports images and saves hdf5 log
    Corutine must receive:
        veusze: Veusz embedded object
        log: dict with parameters: 'out_name' - log's index, 'out_vsz_full' - vsz file name to save

    log parameters will be saved to pandas dataframe end then to hdf5 log cfg['program']['log'])[0]+'.h5'
    """
    with pd.HDFStore(Path(cfg['program']['log']).with_suffix('.h5'), mode='a') as storeLog:
        veusze = None
        path_prev = os_getcwd()
        os_chdir(cfg['output_files']['dir'])
        print('Saving to {}'.format(Path(cfg['output_files']['dir']).absolute()))
        try:
            while True:
                veusze, log = yield ()
                if not cfg['output_files']['b_images_only']:
                    veusze.Save(str(log['out_vsz_full']))
                    # Save vsz modification date
                    log['fileChangeTime'] = datetime.fromtimestamp(Path(
                        log['out_vsz_full']).stat().st_mtime),
                    dfLog = pd.DataFrame.from_records(log, exclude=['out_name', 'out_vsz_full'],
                                                      index=[log['out_name']])
                    storeLog.append(Path(cfg['output_files']['path']).name, dfLog, data_columns=True,
                                    expectedrows=cfg['in']['nfiles'], index=False, min_itemsize={'index': 30})
                if cfg['async']['loop']:
                    try:  # yield from     asyncio.ensure_future(
                        # asyncio.wait_for(, cfg['async']['export_timeout_s'], loop=cfg['async']['loop'])
                        b = cfg['async']['loop'].run_until_complete(
                            export_images_timed(veusze, cfg, '#' + log['out_name']))
                    except asyncio.TimeoutError:
                        l.warning('can not export in time')
                else:
                    export_images(veusze, cfg['output_files'], '#' + log['out_name'])
        except GeneratorExit:
            print('Ok>')
        finally:
            os_chdir(path_prev)
            if veusze and cfg['program']['return'] != '<embedded_object>':
                veusze.Close()
                l.info('closing Veusz embedded object')
            # veusze.WaitForClose()


def co_send_data(gen_veusz_and_logs, cfg, cor_savings):
    """
    - sends loaded Veusz data to caller and recives parameter cfg_in for next step
    - executes Veusz commands by veusze_commands(veusze, cfg_in, ...)
    - sends (veusze, log) to cor_savings()

    :param gen_veusz_and_logs: load_to_veusz(ge_names(cfg), cfg)
    :param cfg: configuration dict. Must contain ['in']['data_yield_prefix'] field to specify which data to send
    :param cor_savings: corutine which receives (veusze, log) for example co_savings corutine

    Usage:
    cfgin_update = None
    while True:
        try:
            d, log = gen_data.send(cfgin_update)
        except (GeneratorExit, StopIteration):
            print('Ok>')
            break
        except Exception as e:
            print('There are error ', e.__class__, '!',
                  '\n==> '.join([a for a in e.args if isinstance(a, str)]), sep='')
            continue

        # ... do some calcuations to prepare custom_expressions ...

        # if add_custom the same then update only add_custom_expressions:
        cfgin_update = {'add_custom_expressions': [expression1, expression2, ...]}

    Note: Courutine fubricated by this function is returned by veuszPropogate if it is called with --data_yield_prefix argument.
    """
    for veusze, log in gen_veusz_and_logs:
        if not veusze:
            continue
        vsz_data = veusz_data(veusze, cfg['in']['data_yield_prefix'])
        # caller do some processing of data and gives new cfg:
        cfgin_update = yield (vsz_data, log)  # to test here run veusze.Save('-.vsz')
        # cfg['in'].update(cfgin_update)  # only update of cfg.in.add_custom_expressions is tested

        file_name_r = Path(log['out_vsz_full']).relative_to(cfg['output_files']['dir'])
        veusze_commands(veusze, cfgin_update, file_name_r)
        cor_savings.send((veusze, log))


def main(new_arg=None, veusze=None):
    """
    Initialise configuration and runs or returns routines
    cfg:
        ['program']['log'],
        'output_files'
        'in'
        'async'
    globals:
        load_vsz
        l

    :param new_arg:
    :param veusze: used to reuze veusz embedded object (thus to not leak memory)
    :return:
    """
    global l, load_vsz
    cfg = cfg_from_args(my_argparser(), new_arg)
    if not cfg or not cfg['program'].get('return'):
        print('Can not initialise')
        return cfg
    elif cfg['program']['return'] == '<cfg_from_args>':  # to help testing
        return cfg

    l = init_logging(logging, None, cfg['program']['log'], cfg['program']['verbose'])
    cfg['program']['log'] = l.root.handlers[0].baseFilename  # sinchronize obtained absolute file name

    print('\n' + this_prog_basename(__file__), 'started', end=' ')
    if cfg['output_files']['b_images_only']:
        print('in images only mode. Output pattern: ')
    else:
        print('. Output pattern and Data: ')
    try:
        # Using cfg['output_files'] to store pattern information
        if not Path(cfg['in']['pattern_path']).is_absolute():
            cfg['in']['pattern_path'] = Path(cfg['in']['path']).with_name(cfg['in']['pattern_path'])
        cfg['output_files']['path'] = cfg['in']['pattern_path']
        cfg['output_files'] = init_file_names(cfg['output_files'], b_interact=False)  # find it
    except Ex_nothing_done as e:
        if not cfg['output_files']['b_images_only']:
            l.warning(e.message, ' - no pattern!')
            return  # or raise FileNotFoundError?

    if (cfg['output_files']['b_images_only'] and cfg['output_files']['namesFull']):
        cfg['in']['namesFull'] = cfg['output_files']['namesFull']  # have all we need to export
    else:
        try:
            cfg['in'] = init_file_names(cfg['in'], cfg['program']['b_interact'])
        except Ex_nothing_done as e:
            print(e.message)
            return  # or raise FileNotFoundError?

    dir_from_cfg(cfg['output_files'], 'export_dir')

    if 'restore_config' in cfg['program']['before_next']:
        cfg['in_saved'] = cfg['in'].copy()
    if cfg['output_files']['b_images_only'] and not 'Close()' in cfg['program']['before_next']:
        cfg['program']['before_next'].append(
            'Close()')  # usually we need to load new file for export (not only modify previous file)

    if cfg['program']['export_timeout_s'] and export_images_timed:
        cfg['async'] = {'loop': asyncio.get_event_loop(),
                        'export_timeout_s': cfg['program']['export_timeout_s']
                        }
    else:
        cfg['async'] = {'loop': None}

    load_vsz = load_vsz_closure(cfg['program']['veusz_path'])
    cfg['load_vsz'] = load_vsz
    cfg['co'] = {}
    if cfg['in']['table_log'] and cfg['in']['path'].suffix == '.h5' and not (
            cfg['output_files']['b_images_only'] and len(cfg['in']['namesFull']) > 1):
        # load data by ranges from table log rows
        cfg['in']['db_path'] = cfg['in']['path']
        in_fulls = h5log_names_gen(cfg['in'])
    elif cfg['in']['tables']:
        # tables instead files
        in_fulls = ge_names_from_hdf5_paths(cfg)
    else:  # switch to use found vsz as source if need only export images (even with database source)
        in_fulls = ge_names(cfg)

    cor_savings = co_savings(cfg)
    cor_savings.send(None)
    nfiles = 0
    if True:
        if cfg['program']['return'] == '<corutines_in_cfg>':
            cfg['co']['savings'] = cor_savings
            cfg['co']['gen_veusz_and_logs'] = load_to_veusz(in_fulls, cfg)
            cfg['co']['send_data'] = co_send_data(load_to_veusz, cfg, cor_savings)
            return cfg  # return with link to generator function
        elif cfg['in'].get('data_yield_prefix'):
            # cycle with obtaining Veusz data
            cfgin_update = None
            while True:  # for vsz_data, log in cor_send_data.send(cfgin_update):
                try:
                    vsz_data, log = co_send_data.send(cfgin_update)
                    nfiles += 1
                except (GeneratorExit, StopIteration, Ex_nothing_done):
                    break
                if 'f_custom_in_cycle' in cfg['program']:
                    cfgin_update = cfg['program']['f_custom_in_cycle'](vsz_data, log)
        else:
            # cycle without obtaining Veusz data (or implemented by user's cfg['program']['f_custom_in_cycle'])
            for veusze, log in load_to_veusz(in_fulls, cfg, veusze):
                file_name_r = Path(log['out_vsz_full']).relative_to(cfg['output_files']['dir'])
                if cfg['program'].get('f_custom_in_cycle'):
                    cfgin_update = cfg['program']['f_custom_in_cycle'](veusze, log)
                    veusze_commands(veusze, cfgin_update, file_name_r)
                cor_savings.send((veusze, log))
                nfiles += 1
            cor_savings.close()
            if cfg['program']['return'] != '<embedded_object>':
                veusze = None  # to note that it is closed in cor_savings.close()
        print(f'{nfiles} processed. ok>')
    try:
        pass
    except Exception as e:
        print('The end. There are error ', e.__class__, ':', '\n==> '.join(
            [a for a in e.args if isinstance(a, str)]))
        return  # or raise FileNotFoundError?
    finally:
        if cfg['async']['loop']:
            cfg['async']['loop'].close()
        # os_chdir(path_prev)
        if veusze and cfg['program']['return'] == '<end>':
            veusze.Close()
            veusze.WaitForClose()
            veusze = None
        elif cfg['program']['return'] == '<embedded_object>':
            cfg['veusze'] = veusze
            return cfg


if __name__ == '__main__':
    main()