"""
Runs steps:
1. csv2h5(): save inclinometer data from Kondrashov format to DB
2. Calculate velocity and average
3. Calculate spectrograms.
4. veuszPropagate(): draw using Veusz pattern

See readme.rst

"""
import logging
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from functools import partial
import itertools
from pathlib import Path
import glob
from typing import Optional

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

# import my scripts
from to_pandas_hdf5.csv2h5 import main as csv2h5
from to_pandas_hdf5.csv_specific_proc import mod_incl_name, rep_in_file, correct_txt
from to_pandas_hdf5.h5_dask_pandas import h5q_interval2coord
from inclinometer.h5inclinometer_coef import h5copy_coef
from inclinometer.incl_calibr import dict_matrices_for_h5

import inclinometer.incl_h5clc as incl_h5clc
import inclinometer.incl_h5spectrum as incl_h5spectrum
import veuszPropagate
from utils_time import intervals_from_period # pd_period_to_timedelta
from utils2init import init_logging, open_csv_or_archive_of_them, st, cfg_from_args, my_argparser_common_part
from magneticDec import mag_dec

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

    p = my_argparser_common_part({'description': 'incl_load version {}'.format(version) + """
---------------------------------------------------
Processing raw inclinometer data,
Saving result to indexed Pandas HDF5 store (*.h5) and *.csv

saves loading log in inclinometer\scripts\log\csv2h5_inclin_Kondrashov.log
-----------------------------------------------------------"""}, version)
    # Configuration sections
    s = p.add_argument_group('in',
                             'All about input files')
    s.add('--path_cruise', default='.',  # nargs=?,
          help='Directory where inclinometer data will be stored, must have subdir '
          '"_raw" with raw file(s) may be in archive/subdirectory (raw_subdir)')
    s.add('--raw_subdir', default='',
          help='Optional zip/rar archive name (data will be unpacked) or subdir in "path_cruise/_raw". For multiple '
          'files symbols * and ? not supported but `prefix` and `number` format strings are avalable as in `raw_pattern`')
    s.add('--raw_pattern', default='*{prefix:}{number:0>3}*.[tT][xX][tT]',
          help='Pattern to find raw files: Python "format" command pattern to format prefix and probe number.'
          'where "prefix" is a --prefix arg. that is in UPPER case and INCL replaced with INKL. To find files in archive'
               'subdirs can be prepended with "/" but files will be unpacked flatten in dir with archive name')
    s.add('--probes_int_list',
          help='Note: Not affects steps 2, 3, set empty list to load all')
    s.add('--prefix', default='incl',
          help='''Table name prefix in DB (and in raw files with modification described in --raw_pattern help).
                  I have used "incl" for inclinometers, "w" for wavegauges. Note (at step 1): only if prefix
                  starts with "incl" or "voln" is in, then raw data must be in Kondrashov format else Baranov's format.
                  For "voln" we replace "voln_v" with "w" when saving corrected raw files and use it to name tables so 
                  only "w" in outputs and we replace "voln" with "w" to search tables''')
    s.add('--db_coefs', default=r'd:\WorkData\~configuration~\inclinometr\190710incl.h5',
          help='coefs will be copied from this hdf5 store to output hdf5 store')
    s.add('--time_range_zeroing_list', help='See incl_h5clc')
    s.add('--time_range_zeroing_dict', help='See incl_h5clc. Example: incl14: [2020-07-10T21:31:00, 2020-07-10T21:39:00]')
    s.add('--dt_from_utc_seconds_dict',
          help='add this correction to loading datetime data. Can use other suffixes instead of "seconds"')
    s.add('--dt_from_utc_hours_dict',
          help='add this correction to loading datetime data. Can use other suffixes instead of "hours"')
    s.add('--dt_from_utc_days_dict',
          help='add this correction to loading datetime data. Can use other suffixes instead of "days"')
    s.add('--time_start_utc_dict', help='Start time of probes started without time setting: when raw date start is 2000-01-01T00:00')
    s.add('--azimuth_add_dict', help='degrees, adds this value to velocity direction (will sum with _azimuth_shift_deg_ coef)')
    s.add('--db_path', help='input hdf5 file path, to load in step 2 and after the data from other database than created on step 1.')

    s = p.add_argument_group('filter',
                             'Filter all data based on min/max of parameters')
    s.add('--min_date_dict', default='0: 2020-01-01T00:00',
          help='minimum time for each probe, use probe number 0 to set default value. For step 2 only number 0 is used')
    s.add('--max_date_dict', default='0: now',
          help='maximum time for each probe, use probe number 0 to set default value')

    s = p.add_argument_group('out',
                             'All about output files')
    s.add('--db_name', help='output hdf5 file name, if not set then dir name will be used. As next steps use DB saved on previous steps do not change between steps or you will need rename source DB accordingly')
    s.add(
        '--aggregate_period_s_int_list', default='',
        help='bin average data in this intervals [s]. Default [None, 300, 600, 3600] if "w" in [in][prefix]'
        ' else last in list is replaced to 7200. None means do not average. Output with result data for'
        ' None will be placed in hdf5 store with suffix "proc_noAvg" in separate sections for each probe. For other'
        ' values in list result will be placed in hdf5 store with suffix "proc" in tables named "bin{average_value}"'
        ' in columns named by parameter suffixed by probe number. Also result will be saved to text files with'
        ' names having date and suffixes for each probe and average value')
    s.add('--aggregate_period_s_not_to_text_int_list', default='None', help='do not save text files for this aggregate periods')
    s.add('--split_period', default='1D',
          help='pandas offset string (5D, H, ...) to proc and output in separate blocks. If saves to csv then writes in parts of this length, but if no bin averaging (aggregate_period) only')

    s = p.add_argument_group('program',
                             'Program behaviour')
    s.add('--step_start_int', default='1', choices=[str(i) for i in  [1, 2, 3, 4, 40, 50]],
          help='step to start')
    s.add('--step_end_int', default='2', choices=['1', '2', '3', '4', '40'],
          help='step to end (inclusive, or if less than start then will run one start step only)')
    s.add('--dask_scheduler',
          help='can be "synchronous" (for help debugging) or "distributed"')
    s.add('--load_timeout_s_float', default='180',
          help='For step 4: export asynchronously with this timeout, s (tried 600s?)')
    return p


def constant_factory(val):
    def default_val():
        return val
    return default_val


#  @hydra.main(config_path="ini", config_name=Path(__file__).stem)
def main(new_arg=None, **kwargs):
    """
    
    :param new_arg: list of strings, command line arguments
    :kwargs: dicts of dicts (for each ini section): specified values overwrites ini values
    """

    # global l
    cfg = cfg_from_args(my_argparser(), new_arg, **kwargs)
    if not cfg['program']:
        return  # usually error of unrecognized arguments displayed
    l = init_logging(logging, None, None, 'INFO')
    # l = init_logging(logging, None, cfg['program']['log'], cfg['program']['verbose'])

    cfg['in']['db_coefs'] = Path(cfg['in']['db_coefs'])
    for path_field in ['db_coefs', 'path_cruise']:
        if not cfg['in'][path_field].is_absolute():
            cfg['in'][path_field] = (cfg['in']['cfgFile'].parent / cfg['in'][path_field]).resolve().absolute()  # cfg['in']['cfgFile'].parent /


    for lim_str, lim_default in (('min_date', np.datetime64('2000-01-01', 'ns')),
                                 ('max_date', np.datetime64('now', 'ns'))):
        # convert cfg['filter'][{lim}] keys to int to be comparable with probes_int_list (for command line arguments keys are allways strings, in yaml you can set string or int)
        _ = {int(k): v for k, v in cfg['filter'][lim_str].items()}
        cfg['filter'][lim_str] = defaultdict(constant_factory(_.get(0, lim_default)), _)

    if True:  # False. Experimental speedup but takes memory
        from dask.cache import Cache
        cache = Cache(2e9)  # Leverage two gigabytes of memory
        cache.register()    # Turn cache on globally

    if cfg['program']['dask_scheduler']:
        if cfg['program']['dask_scheduler'] == 'distributed':
            from dask.distributed import Client
            # cluster = dask.distributed.LocalCluster(n_workers=2, threads_per_worker=1, memory_limit="5.5Gb")
            client = Client(processes=False)
            # navigate to http://localhost:8787/status to see the diagnostic dashboard if you have Bokeh installed
            # processes=False: avoid inter-worker communication for computations releases the GIL (numpy, da.array)  # without is error
        else:
            if cfg['program']['dask_scheduler'] == 'synchronous':
                l.warning('using "synchronous" scheduler for debugging')
            import dask
            dask.config.set(scheduler=cfg['program']['dask_scheduler'])

    # Run steps :
    st.start = cfg['program']['step_start']
    st.end = cfg['program']['step_end']
    st.go = True

    if not cfg['out']['db_name']:
        # set db_name by 'path_cruise' name's digits or its parent's if it starts with digits
        for p in (lambda p: [p, p.parent])(cfg['in']['path_cruise']):
            if m := re.match(r'(^[\d_]*).*', p.name):
                cfg['out']['db_name'] = f"{m.group(1).strip('_')}.raw.h5"
                break
    #
    dir_incl = next((d for d in cfg['in']['path_cruise'].glob('*inclinometer*') if d.is_dir()), cfg['in']['path_cruise'])
    db_path = dir_incl / '_raw' / cfg['out']['db_name']
    # ---------------------------------------------------------------------------------------------

    def datetime64_str(time_str: Optional[str] = None) -> np.ndarray:
        """
        Reformat time_str to ISO 8601 or to 'NaT'. Used here for input in funcs that converts str to numpy.datetime64
        :param time_str: May be 'NaT'
        :return: ndarray of strings (tested for 1 element only) formatted by numpy.
        """
        return np.datetime_as_string(np.datetime64(time_str, 's'))

    probes = cfg['in']['probes'] or range(1, 41)  # sets default range, specify your values before line ---
    raw_prefix, probe_is_incl = re.subn('INCL_?', 'INKL_', cfg['in']['prefix'].upper())
    if not probe_is_incl:
        if probe_is_incl := cfg['in']['prefix'].upper().startswith(raw_prefix_check := 'I'):
            raw_prefix = raw_prefix_check

    # some parameters that depend on probe type (indicated by prefix)
    p_type = defaultdict(
        constant_factory(
            {
                'correct_fun': partial(
                    correct_txt,
                    mod_file_name=mod_incl_name,
                    sub_str_list=[                                                 # \.\d{2})(,\-?\d{1,3}\.\d{2}))
                        b'^(?P<use>20\d{2}(,\d{1,2}){5}(,\-?\d{1,6}){6},\d{1,2}(\.\d{1,2})?,\-?\d{1,3}(\.\d{1,2})?).*',
                        b'^.+'
                        ]),
                'fs': 5,
                'format': 'Kondrashov',
             }),
            {
                'voln': {  # w?
                    'correct_fun': partial(
                        correct_txt,
                        mod_file_name=mod_incl_name,
                        sub_str_list=[b'^(?P<use>20\d{2}(,\d{1,2}){5}(,\-?\d{1,8})(,\-?\d{1,2}\.\d{2}){2}).*', b'^.+']),
                    'fs': 5,
                    # 'tbl_prefix': 'w',
                    'format': 'Kondrashov',
                    },
                'INKLBA': {  # baranov's format
                    'correct_fun': partial(
                        correct_txt,
                        mod_file_name=mod_incl_name,
                        sub_str_list=[b'^\r?(?P<use>20\d{2}(\t\d{1,2}){5}(\t\d{5}){8}).*', b'^.+']),
                    'fs': 10,
                    'format': 'Baranov',
                    }
             }
        )

    # def fs(probe, name):
    #     if 'w' in name.lower():  # Baranov's wavegauge electronic
    #         return 10  # 5
    #     return 5
    # if probe < 20 or probe in [23, 29, 30, 32, 33]:  # 30 [4, 11, 5, 12] + [1, 7, 13, 30]
    #     return 5
    # if probe in [21, 25, 26] + list(range(28, 35)):
    #     return 8.2
    # return 4.8


    if st(1, 'Save inclinometer or wavegauge data from ASCII to HDF5'):
        # Note: Can not find additional not corrected files for same probe if already have any corrected in search path (move them out if need)

        probes_found = []  # collect probes for which files found
        i_raw_found = 0  # counter of processed files
        # patten to identify only _probe_'s raw data files that need to correct '*INKL*{:0>2}*.[tT][xX][tT]':

        raw_parent = dir_incl / '_raw'  # raw_parent /=
        if cfg['in']['raw_subdir'] is None:
            cfg['in']['raw_subdir'] = ''

        # Output dir name that correct_fun() can make (1 level only) if input is archive - replacing multilevel subdirs:
        dir_csv_cor = raw_parent / re.sub(
            r'[.\\/ *?]', '_', cfg['in']['raw_subdir'].format(prefix=raw_prefix, number=0)
            )

        def dt_from_utc_2000(probe):
            """ Correct time of probes started without time setting. Its start date in raw file must be 2000-01-01T00:00
            """
            return (datetime(year=2000, month=1, day=1) - cfg['in']['time_start_utc'][probe]
                    ) if cfg['in']['time_start_utc'].get(probe) else timedelta(0)

        # Convert cfg['in']['dt_from_utc'] keys to int
        cfg['in']['dt_from_utc'] = {int(p): v for p, v in cfg['in']['dt_from_utc'].items()}
        # Convert cfg['in']['t_start_utc'] to cfg['in']['dt_from_utc'] and keys to int
        cfg['in']['dt_from_utc'].update(    # overwriting the 'time_start_utc' where already exist
            {int(p): dt_from_utc_2000(p) for p, v in cfg['in']['time_start_utc'].items()}
            )
        # Make cfg['in']['dt_from_utc'][0] be the default value
        cfg['in']['dt_from_utc'] = defaultdict(constant_factory(cfg['in']['dt_from_utc'].pop(0, timedelta(0))),
                                                                                     cfg['in']['dt_from_utc'])

        for probe in probes:
            raw_found = []
            raw_subdir_for_probe = cfg['in']['raw_subdir'].format(prefix=raw_prefix, number=probe)
            raw_pattern_file = str(Path(glob.escape(raw_subdir_for_probe)) /
                                   cfg['in']['raw_pattern'].format(prefix=raw_prefix, number=probe))
            correct_fun = p_type[cfg['in']['prefix']]['correct_fun']
            tbl = f"{cfg['in']['prefix']}{probe:0>2}"
            # add str will be added only if corrected output name matches input:
            correct_fun.keywords['mod_file_name'] = partial(
                correct_fun.keywords['mod_file_name'], add_str=(add_str := f'@{tbl}')
                )
            if not (raw_subdir := (raw_parent / raw_subdir_for_probe)).is_dir():
                raw_subdir = ''
            # if not archive:
            if (not re.match(r'.*\.(zip|rar)$', raw_subdir_for_probe, re.IGNORECASE)) and raw_subdir:  # why dir_incl was used before?
                raw_found = list(file_in for file_in in raw_parent.glob(raw_pattern_file) if not file_in.stem.endswith(add_str))
            if not raw_found:
                # Check if already have corrected files for probe generated by correct_txt(). If so then just use them
                raw_found = list((raw_subdir or dir_csv_cor).glob(
                    str(correct_fun.keywords['mod_file_name'](raw_pattern_file)))  # f"{tbl}.txt"
                    )
                if not raw_found:
                    raw_found = list((raw_parent / cfg['in']['raw_subdir']).glob(f"{tbl}.txt"))
                if raw_found:
                    print('corrected csv file', [r.name for r in raw_found], 'found')

                    def correct_fun(x, dir_out):
                        return x
                elif not raw_subdir_for_probe:
                    continue
            i_file = 0  # will be > 0 if found files
            for file_in in (
                raw_found or open_csv_or_archive_of_them(raw_parent, binary_mode=False, pattern=raw_pattern_file)
                    ):
                file_in = correct_fun(file_in, dir_out=dir_csv_cor, binary_mode=False)
                i_raw_found += 1
                i_file += 1
            else:
                if i_file == 0:
                    print('no', raw_pattern_file, end=', ')
                    continue
            probes_found.append(probe)
        print('probes:', probes_found, ', (', i_raw_found, 'raw files was found)')
        i_probe = None
        for i_probe, probe in enumerate(probes_found, start=1):
            tbl = f"{cfg['in']['prefix']}{probe:0>2}"  # file_in.stem
            # Already have corrected files for probe generated by correct_txt() so just use them
            raw_subdir_for_probe = cfg['in']['raw_subdir'].format(prefix=raw_prefix, number=probe)
            if not (raw_subdir := (raw_parent / raw_subdir_for_probe)).is_dir():
                raw_subdir = ''
            raw_pattern_file = cfg['in']['raw_pattern'].format(prefix=raw_prefix, number=probe).split('/')[-1]
            # for file_in in raw_found:
            #     if not file_in:
            #         continue
            # tbl = re.sub('^((?P<i>inkl)|w)_0', lambda m: 'incl' if m.group('i') else 'w',  # correct name
            #              re.sub('^[\d_]*|\*', '', file_in.stem).lower()),  # remove date-prefix if in name
            csv2h5([
                str(Path(__file__).parent / 'cfg' / f"csv_{'inclin' if probe_is_incl else 'wavegauge'}"
                                                    f"_{p_type[cfg['in']['prefix']]['format']}.ini"),
                '--path', str(
                    (raw_subdir or dir_csv_cor) / str(correct_fun.keywords['mod_file_name'](raw_pattern_file))
                    ),
                '--blocksize_int', '50_000_000',  # 50Mbt
                '--table', tbl,
                '--db_path', str(db_path),
                # '--log', str(scripts_path / 'log/csv2h5_inclin_Kondrashov.log'),
                # '--on_bad_lines', 'warn',
                '--b_interact', '0',
                '--fs_float', str(p_type[cfg['in']['prefix']]['fs']),  #f'{fs(probe, file_in.stem)}',
                '--dt_from_utc_seconds', str(cfg['in']['dt_from_utc'][probe].total_seconds()),
                '--b_del_temp_db', '1',
                #'--b_remove_duplicates', '1',
                ] +
               (
               ['--csv_specific_param_dict', 'invert_magnitometr: True'
                ] if probe_is_incl else []
               ),
                **{
                'filter': {
                     'min_date': cfg['filter']['min_date'][probe],
                     'max_date': cfg['filter']['max_date'][probe],  # simple 'now' works in sinchronious mode
                    }
                }
            )

            # Get coefs:
            l.info(f"Adding coefficients to {db_path}/{tbl} from {cfg['in']['db_coefs']}")
            try:
                h5copy_coef(cfg['in']['db_coefs'], db_path, tbl)
            except KeyError as e:  # Unable to open object (component not found)
                l.warning('No coefs to copy?')  # write some dummy coefficients to can load Veusz patterns:
                h5copy_coef(None, db_path, tbl, dict_matrices=dict_matrices_for_h5(tbl=tbl))
            except OSError as e:
                l.warning('Not found DB with coefs?')  # write some dummy coefficients to can load Veusz patterns:
                h5copy_coef(None, db_path, tbl, dict_matrices=dict_matrices_for_h5(tbl=tbl))
        if i_probe is not None:
            print('Ok:', i_probe, 'probes processed.')



    # 'incl.*|w\d*'  inclinometers or wavegauges w\d\d # 'incl09':
    tables_list_regex = f"{cfg['in']['prefix'].replace('voln', 'w')}.*"
    if cfg['in']['probes']:
        tables_list_regex += "(?:{})".format('|'.join('{:0>2}'.format(p) for p in cfg['in']['probes']))

    if st(2, 'Calculate physical parameters and average'):
        kwarg = {'in': {
            'min_date': cfg['filter']['min_date'][0],
            'max_date': cfg['filter']['max_date'][0],
            'time_range_zeroing': cfg['in']['time_range_zeroing']
            }, 'proc': {}
            }
        # if aggregate_period_s is None then not average and write to *.proc_noAvg.h5 else loading from that h5 and writing to _proc.h5
        if not cfg['out']['aggregate_period_s']:
            cfg['out']['aggregate_period_s'] = [None, 2, 600, 7200 if probe_is_incl else 3600]

        if cfg['in']['azimuth_add']:
            msgs = []
            if 'Lat' in cfg['in']['azimuth_add']:
                # add magnetic declination,° for used coordinates
                # todo: get time
                kwarg['proc']['azimuth_add'] = mag_dec(
                    cfg['in']['azimuth_add']['Lat'], cfg['in']['azimuth_add']['Lon'],
                    datetime(2020, 9, 10), depth=-1
                    )
                msgs.append("magnetic declination: {kwarg['proc']['azimuth_add']}")
            else:
                kwarg['proc']['azimuth_add'] = 0
            if cfg['in']['azimuth_add'].get('constant'):
                # and add constant. For example, subtruct declination at the calibration place if it was applied
                kwarg['proc']['azimuth_add'] += cfg['in']['azimuth_add']['constant']  # add -6.656 to account for calibration in Kaliningrad (mag deg = 6.656°)
                msgs.append("constant: {cfg['in']['azimuth_add']['constant']}")
            if kwarg['proc']['azimuth_add']:
                print('azimuth correction: ', kwarg['proc']['azimuth_add'], ',°:', 'plus '.join(msgs))

        for aggregate_period_s in cfg['out']['aggregate_period_s']:
            if aggregate_period_s is None:
                db_path_in = cfg['in']['db_path'] or db_path  # allows user's db_path to load data in counts for step 2
                db_path_out = dir_incl / f'{db_path.stem}.proc_noAvg.h5'
            else:
                db_path_in = dir_incl / f'{db_path.stem}.proc_noAvg.h5'
                db_path_out = dir_incl / f'{db_path.stem}_proc.h5'  # or separately: '_proc{aggregate_period_s}.h5'


            args = [
                    '../../empty.yml',  # all settings are here, so to not print 'using default configuration' we use some existed empty file
                    '--db_path', str(db_path_in),
                    '--tables_list', tables_list_regex,
                    '--aggregate_period', f'{aggregate_period_s}S' if aggregate_period_s else '',
                    '--out.db_path', str(db_path_out),
                    '--table', f'V_incl_bin{aggregate_period_s}' if aggregate_period_s else 'V_incl',
                    '--verbose', 'INFO',  #'DEBUG' get many numba messages
                    '--b_del_temp_db', '1',
                    # '--calc_version', 'polynom(force)',  # depreshiated
                    # '--chunksize', '20000',
                    # '--not_joined_db_path', f'{db_path.stem}_proc.h5',
                    ]

            if aggregate_period_s is None:  # proc. parameters (if we have saved proc. data then when aggregating we are not processing)
                # Note: for Baranov's prog 4096 is not suited:
                args += (
                    ['--max_dict', 'M[xyz]:4096',
                     # '--time_range_zeroing_dict', "incl19: '2019-11-10T13:00:00', '2019-11-10T14:00:00'\n,"  # not works - use kwarg
                     # '--time_range_zeroing_list', '2019-08-26T04:00:00, 2019-08-26T05:00:00'
                     '--split_period', cfg['out']['split_period']
                    ] if probe_is_incl else
                    [# '--bad_p_at_bursts_starts_peroiod', '1H',
                    ])
                    # csv splitted by 1day (default for no avg) else csv is monolith
            if aggregate_period_s not in cfg['out']['aggregate_period_s_not_to_text']:  # , 300, 600]:
                args += ['--text_path', str(dir_incl / 'text_output')]
            # If need all data to be combined one after one:
            # set_field_if_no(kwarg, 'in', {})
            # kwarg['in'].update({
            #
            #         'tables': [f'incl{i:0>2}' for i in min_date.keys() if i!=0],
            #         'dates_min': min_date.values(),  # in table list order
            #         'dates_max': max_date.values(),  #
            #         })
            # set_field_if_no(kwarg, 'out', {})
            # kwarg['out'].update({'b_all_to_one_col': 'True'})


            incl_h5clc.main(args, **kwarg)


    if st(3, 'Calculate spectrograms'):  # Can be done at any time after step 1
        min_Pressure = 0.5

        # add dict dates_min like {probe: parameter} of incl_clc to can specify param to each probe
        def raise_ni():
            raise NotImplementedError('Can not proc probes having different fs in one run: you need to do it separately')

        args = [
            Path(incl_h5clc.__file__).with_name(f'incl_h5spectrum{db_path.stem}.yaml'),
            # if no such file all settings are here
            '--db_path', str(dir_incl / f'{db_path.stem}.proc_noAvg.h5'),
            '--tables_list', tables_list_regex,
            # '--aggregate_period', f'{aggregate_period_s}S' if aggregate_period_s else '',

            '--min_date', datetime64_str((lambda t: t.replace(second=0, microsecond=0, minute=0) +
                                         timedelta(hours=t.minute//30))(cfg['filter']['min_date'][0])
                                         ),  # round start to hours mainly for data with bursts that starts with hours
            '--max_date', datetime64_str(cfg['filter']['max_date'][0]),  # '2019-09-09T16:31:00',  #17:00:00
            '--min_Pressure', f'{min_Pressure}',
            # '--max_dict', 'M[xyz]:4096',  # use if db_path is not ends with .proc_noAvg.h5 i.e. need calc velocity
            '--out.db_path', f"{db_path.stem.replace('incl', cfg['in']['prefix'])}_proc_psd.h5",
            # '--table', f'psd{aggregate_period_s}' if aggregate_period_s else 'psd',
            '--fs_float', str(p_type[cfg['in']['prefix']]['fs']),  # f"{fs(probes[0], cfg['in']['prefix'])}",
            # (lambda x: x == x[0])(np.vectorize(fs)(probes, prefix))).all() else raise_ni()
            #
            # '--time_range_zeroing_list', '2019-08-26T04:00:00, 2019-08-26T05:00:00'
            # '--verbose', 'DEBUG',
            # '--chunksize', '20000',
            '--b_interact', '0',
            ]
        if probe_is_incl:
            args += ['--split_period', '2H',
                     '--fmin', '0.0004',  #0.0004
                     '--fmax', '1.05'
                     ]
        else:
            args += ['--split_period', '1H',
                     '--dt_interval_minutes', '15',  # set this if burst mode to the burst interval
                     '--fmin', '0.0001',
                     '--fmax', '4',
                     #'--min_Pressure', '-1e15',  # to not load NaNs
                     ]


        incl_h5spectrum.main(args)


    if st(4, 'Draw in Veusz'):
        pattern_path = dir_incl / r'processed_h5,vsz/201202-210326incl_proc#28.vsz'
        # r'\201202_1445incl_proc#03_pattern.vsz'  #'
        # db_path.parent / r'vsz_5min\191119_0000_5m_incl19.vsz'  # r'vsz_5min\191126_0000_5m_w02.vsz'

        b_images_only = False
        # importing in vsz index slices replacing:
        pattern_str_slice_old = None

        # Length of not adjacent intervals, s (set None to not allow)
        # pandas interval in string or tuple representation '1D' of period between intervals and interval to draw
        period_str = '0s'  # '1D'  #  dt
        dt_str = '0s'  # '5m'
        file_intervals = None

        period = to_offset(period_str).delta
        dt = to_offset(dt_str).delta  # timedelta(0)  #  60 * 5

        if file_intervals and period and dt:

            # Load starts and assign ends
            t_intervals_start = pd.read_csv(
                cfg['in']['path_cruise'] / r'vsz+h5_proc\intervals_selected.txt',
                converters={'time_start': lambda x: np.datetime64(x, 'ns')}, index_col=0).index
            edges = (
                pd.DatetimeIndex(t_intervals_start), pd.DatetimeIndex(t_intervals_start + dt_custom_s))  # np.zeros_like()
        elif period and dt:
            # Generate periodic intervals
            t_interval_start, t_intervals_end = intervals_from_period(datetime_range=np.array(
                [cfg['filter']['min_date']['0'], cfg['filter']['max_date']['0']],
                # ['2018-08-11T18:00:00', '2018-09-06T00:00:00'],
                # ['2019-02-11T13:05:00', '2019-03-07T11:30:00'],
                # ['2018-11-16T15:19', '2018-12-14T14:35'],
                # ['2018-10-22T12:30', '2018-10-27T06:30:00'],
                'datetime64[s]'), period=period)
            edges = (pd.DatetimeIndex([t_interval_start]).append(t_intervals_end[:-1]),
                     pd.DatetimeIndex(t_intervals_end))
        else:  # [min, max] edges for each probe
            edges_dict = {pr: [cfg['filter']['min_date'][pr], cfg['filter']['max_date'][pr]] for pr in probes}

        cfg_vp = {'veusze': None}
        for i, probe in enumerate(probes):
            # cfg_vp = {'veusze': None}
            if edges_dict:  # custom edges for each probe
                edges = [pd.DatetimeIndex([t]) for t in edges_dict[probe]]

            # substr in file to rerplace probe_name_in_pattern (see below).
            probe_name = f"_{cfg['in']['prefix'].replace('incl', 'i')}{probe:02}"
            tbl = None  # f"/{cfg['in']['prefix']}{probe:02}"  # to check probe data exist in db else will not check
            l.info('Draw %s in Veusz: %d intervals...', probe_name, edges[0].size)
            # for i_interval, (t_interval_start, t_interval_end) in enumerate(zip(pd.DatetimeIndex([t_interval_start]).append(t_intervals_end[:-1]), t_intervals_end), start=1):


            for i_interval, (t_interval_start, t_interval_end) in enumerate(zip(*edges), start=1):

                # if i_interval < 23: #<= 0:  # TEMPORARY Skip this number of intervals
                #     continue
                if period  and period != dt:
                    t_interval_start = t_interval_end - pd.Timedelta(dt_custom_s, 's')

                if tbl:
                    try:  # skipping absent probes
                        start_end = h5q_interval2coord(
                            db_path=str(db_path),
                            table=tbl,
                            t_interval=(t_interval_start, t_interval_end))
                        if not len(start_end):
                            break  # no data
                    except KeyError:
                        break  # device name not in specified range, go to next name


                pattern_path_new = pattern_path.with_name(
                    ''.join([f'{t_interval_start:%y%m%d_%H%M}', f'_{dt_str}' if dt else '', f'{probe_name}.vsz']))

                # Modify pattern file
                if not b_images_only:
                    pattern_type, pattern_number = re.match(r'.*(incl|w)_proc?#?(\d*).*', pattern_path.name).groups()
                    probe_name_in_pattern = f"_{pattern_type.replace('incl', 'i')}{pattern_number}"

                    def f_replace(line):
                        """
                        Replace in file
                        1. probe name
                        2. slice
                        """
                        # if i_interval == 1:
                        line, ok = re.subn(probe_name_in_pattern, probe_name, line)
                        if ok and pattern_str_slice_old:  # can be only in same line
                            str_slice = '(({:d}, {:d}, None),)'.format(
                                *(start_end + np.int32([-1, 1])))  # bytes(, 'ascii')
                            line = re.sub(pattern_str_slice_old, str_slice, line)
                        return line


                    if not rep_in_file(pattern_path, pattern_path_new, f_replace=f_replace, binary_mode=False):
                        l.warning('Veusz pattern not changed!')  # may be ok if we need draw pattern
                        # break
                    elif cfg_vp['veusze']:
                        cfg_vp['veusze'].Load(str(pattern_path_new))
                elif cfg_vp['veusze']:
                    cfg_vp['veusze'].Load(str(pattern_path_new))

                txt_time_range = \
                    """
                    "[['{:%Y-%m-%dT%H:%M}', '{:%Y-%m-%dT%H:%M}']]" \
                    """.format(t_interval_start, t_interval_end)
                print(f'{i_interval}. {txt_time_range}', end=' ')

                cfg_vp = veuszPropagate.main(
                    [Path(veuszPropagate.__file__).parent.with_name('veuszPropagate.ini'),
                     # '--data_yield_prefix', '-',

                     # '--path', str(db_path),  # if custom loading from db and some source is required
                     '--tables_list', '', # switches to search vsz-files only # f'/{probe_name}',  # 181022inclinometers/ \d*

                     '--pattern_path', str(pattern_path_new),
                     # fr'd:\workData\BalticSea\190801inclinometer_Schuka\{probe_name}_190807_1D.vsz',
                     # str(dir_incl / f'{probe_name}_190211.vsz'), #warning: create file with small name
                     # '--before_next', 'restore_config',
                     # '--add_to_filename', f"_{t_interval_start:%y%m%d_%H%M}_{dt}",
                     '--filename_fun', f'lambda tbl: "{pattern_path_new.name}"',
                     '--add_custom_list',
                     f'USEtime__',  # f'USEtime{probe_name}', nAveragePrefer',
                     '--add_custom_expressions_list',
                     txt_time_range,
                     # + """
                     # ", 5"
                     # """,
                     '--b_update_existed', 'True',
                     '--export_pages_int_list', '0',  # 0 for all '6, 7, 8',  #'1, 2, 3'
                     # '--export_dpi_int', '200',
                     '--export_format', 'jpg', #'emf',
                     '--b_interact', '0',
                     '--b_images_only', f'{b_images_only}',
                     '--return', '<embedded_object>',  # reuse to not bloat memory
                     '--b_execute_vsz', 'True', '--before_next', 'Close()'  # Close() need if b_execute_vsz many files
                     ],
                    veusze=cfg_vp['veusze'])


    if st(40, f'Draw in Veusz by loader-drawer.vsz method'):
        # save all vsz files that uses separate code

        from os import chdir as os_chdir
        dt_s = 300
        cfg['in']['pattern_path'] = db_path.parent / f'vsz_{dt_s:d}s' / '~pattern~.vsz'

        time_starts = pd.read_csv(
            db_path.parent / r'processed_h5,vsz' / 'intervals_selected.txt',
            index_col=0, parse_dates=True, date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%dT%H:%M:%S')
            ).index

        pattern_code = cfg['in']['pattern_path'].read_bytes()  # encoding='utf-8'
        path_vsz_all = []
        for i, probe in enumerate(probes):
            probe_name = f"{cfg['in']['prefix']}{probe:02}"  # table name in db
            l.info('Draw %s in Veusz: %d intervals...', probe_name, time_starts.size)
            for i_interval, time_start in enumerate(time_starts, start=1):
                path_vsz = cfg['in']['pattern_path'].with_name(
                    f"{time_start:%y%m%d_%H%M}_{probe_name.replace('incl','i')}.vsz"
                    )
                # copy file to path_vsz
                path_vsz.write_bytes(pattern_code)  # replaces 1st row
                path_vsz_all.append(path_vsz)

        os_chdir(cfg['in']['pattern_path'].parent)
        veuszPropagate.main(['cfg/veuszPropagate.ini',
                             '--path', str(cfg['in']['pattern_path'].with_name('??????_????_*.vsz')),  # db_path),
                             '--pattern_path', f"{cfg['in']['pattern_path']}_",
                             # here used to auto get export dir only. may not be _not existed file path_ if ['out']['paths'] is provided
                             # '--table_log', f'/{device}/logRuns',
                             # '--add_custom_list', f'{device_veusz_prefix}USE_time_search_runs',  # 'i3_USE_timeRange',
                             # '--add_custom_expressions',
                             # """'[["{log_row[Index]:%Y-%m-%dT%H:%M:%S}", "{log_row[DateEnd]:%Y-%m-%dT%H:%M:%S}"]]'""",
                             # '--export_pages_int_list', '1', #'--b_images_only', 'True'
                             '--b_interact', '0',
                             '--b_update_existed', 'True',  # todo: delete_overlapped
                             '--b_images_only', 'True',
                             '--load_timeout_s_float', str(cfg['program']['load_timeout_s'])
                             # '--min_time', '2020-07-08T03:35:00',

                             ],
                            **{'out': {'paths': path_vsz_all}})

    if st(50, 'Export from existed Veusz files in dir'):
        pattern_parent = db_path.parent  # r'vsz_5min\191126_0000_5m_w02.vsz''
        pattern_path = str(pattern_parent / r'processed_h5,vsz' / '??????incl_proc#[1-9][0-9].vsz')  # [0-2,6-9]
        veuszPropagate.main([
            'cfg/veuszPropagate.ini',
             '--path', pattern_path,
             '--pattern_path', pattern_path,
             # '--export_pages_int_list', '1', #'--b_images_only', 'True'
             '--b_interact', '0',
             '--b_update_existed', 'True',  # todo: delete_overlapped
             '--b_images_only', 'True',
             '--load_timeout_s_float', str(cfg['program']['load_timeout_s']),
             '--b_execute_vsz', 'True', '--before_next', 'Close()'  # Close() need if b_execute_vsz many files
            ])



if __name__ == '__main__':
    main()