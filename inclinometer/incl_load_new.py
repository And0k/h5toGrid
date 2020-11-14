"""
Runs steps:
1. csv2h5(): save inclinometer data from Kondrashov format to DB
2. Calculate velocity and average
3. Calculate spectrograms.
4. veuszPropagate(): draw using Veusz pattern

Allows to specify some of args:
    - control execution parameters:
        :start: float, start step. Begins from step >= start
        Raw file names should be matched by regex: "[\d_]*(W|INKL)_\d\d*.txt"

    - data processing parameters:

        :path_cruise:
        :dir_incl: or :file_in:
        :path_db:
        :probes:
        ...

"""
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional
from functools import partial
import numpy as np
import pandas as pd
import hydra

# import my scripts
from to_pandas_hdf5.csv2h5 import main as csv2h5
from to_pandas_hdf5.csv_specific_proc import correct_kondrashov_txt, rep_in_file, correct_baranov_txt
from to_pandas_hdf5.h5_dask_pandas import h5q_interval2coord
from inclinometer.h5inclinometer_coef import h5copy_coef
from inclinometer.incl_calibr import dict_matrices_for_h5

import inclinometer.incl_h5clc as incl_h5clc
import inclinometer.incl_h5spectrum as incl_h5spectrum
import veuszPropagate
from utils_time import pd_period_to_timedelta
from utils2init import path_on_drive_d, init_logging, open_csv_or_archive_of_them, st, cfg_from_args, my_argparser_common_part
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
-----------------------------------------------------------"""}, version)
    # Configuration sections
    s = p.add_argument_group('in',
                             'All about input files')
    s.add('--path_cruise', default='.',  # nargs=?,
             help='Directory where inclinometer data will be stored, subdirs:'
                  '"_raw": required, with raw file(s)')
    s.add('--raw_subdir', default='',
             help='Optional zip/rar arhive name (data will be unpacked) or subdir in "path_cruise/_raw"')
    s.add('--raw_pattern', default="*{prefix:}{number:0>3}*.[tT][xX][tT]",
             help='Pattern to find raw files: Python "format" command pattern to format prefix and probe number.'
                  '"prefix" is a --probes_prefix arg that is in UPPER case and INCL replaced with INKL.')
    s.add('--probes_int_list',
             help='Note: Not affects steps 2, 3, set empty list to load all')
    s.add('--probes_prefix', default='incl',
             help='''Table name prefix in DB (and in raw files with modification described in --raw_pattern help).
                  I have used "incl" for inclinometers, "w" for wavegauges. Note: only if "incl" in probes_prefix the 
                  Kondrashov format is used else it must be in Baranov's format''')

    s.add('--db_coefs', default=r'd:\WorkData\~configuration~\inclinometr\190710incl.h5',
             help='coefs will be copied from this hdf5 store to output hdf5 store')
    s.add('--timerange_zeroing_list', help='See incl_h5clc')
    s.add('--timerange_zeroing_dict', help='See incl_h5clc. Example: incl14: [2020-07-10T21:31:00, 2020-07-10T21:39:00]')
    s.add('--dt_from_utc_seconds', default='0',
             help='add this correction to loading datetime data. Can use other suffixes instead of "seconds"')
    s.add('--dt_from_utc_hours', default='0',
             help='add this correction to loading datetime data. Can use other suffixes instead of "hours"')
    s.add('--azimuth_add_float_dict', help='degrees, adds this value to velocity direction (will sum with _azimuth_shift_deg_ coef)')

    s = p.add_argument_group('filter',
                             'Filter all data based on min/max of parameters')
    s.add('--min_date_dict', default='0: 2020-06-28T18:00',
             help='minimum time for each probe, use probe number 0 to set default value')
    s.add('--max_date_dict', default='0: now',
             help='maximum time for each probe, use probe number 0 to set default value')

    s = p.add_argument_group('out',
                             'All about output files')
    s.add('--db_name', help='output hdf5 file name, do not set for auto using dir name')
    s.add('--aggregate_period_s_int_list', default='',
              help='bin average data in this intervals will be placed in separate section in output DB and csv for '
                   '[None, 300, 600], default: None, 2, 600, 3600 if "w" in [in][probes_prefix] else 7200')
    s.add('--aggregate_period_s_not_to_text_int_list', default='None', help='do not save text files for this aggregate periods')

    s = p.add_argument_group('program',
                             'Program behaviour')
    s.add('--step_start_int', default='1', choices=['1', '2', '3', '4'],
               help='step to start')
    s.add('--step_end_int', default='2', choices=['1', '2', '3', '4'],
               help='step to end (inclusive, or if less than start then will run one start step only)')
    s.add('--dask_scheduler',
               help='can be "synchronous" (for help debugging) or "distributed"')
    return (p)


@hydra.main(config_path="ini", config_name=Path(__file__).stem)
def main(new_arg=None, **kwargs):
    """

    :param new_arg: list of strings, command line arguments
    :kwargs: dicts of dictcts (for each ini section): specified values overwrites ini values
    """

    # global l
    cfg = cfg_from_args(my_argparser(), new_arg, **kwargs)
    cfg['in']['db_coefs'] = Path(cfg['in']['db_coefs'])
    for path_field in ['db_coefs', 'path_cruise']:
        if not cfg['in'][path_field].is_absolute():
            cfg['in'][path_field] = (cfg['in']['cfgFile'].parent / cfg['in'][path_field]).resolve().absolute()  # cfg['in']['cfgFile'].parent /

    def constant_factory(val):
        def default_val():
            return val
        return default_val

    for lim in ('min_date', 'max_date'):
        cfg['filter'][lim] = defaultdict(constant_factory(cfg['filter'][lim].get('0', cfg['filter'][lim].get(0))),
                                         cfg['filter'][lim])

    l = init_logging(logging, None, None, 'INFO')
    #l = init_logging(logging, None, cfg['program']['log'], cfg['program']['verbose'])


    if True:  # False. Experimental speedup but takes memory
        from dask.cache import Cache
        cache = Cache(2e9)  # Leverage two gigabytes of memory
        cache.register()    # Turn cache on globally
    if cfg['program']['dask_scheduler']:
        if cfg['program']['dask_scheduler'] == 'distributed':
            from dask.distributed import Client
            client = Client(
                processes=False)  # navigate to http://localhost:8787/status to see the diagnostic dashboard if you have Bokeh installed
            # processes=False: avoide inter-worker communication for computations releases the GIL (numpy, da.array)  # without is error
        else:
            if cfg['program']['dask_scheduler'] == 'synchronous':
                l.warning('using "synchronous" scheduler for debugging')
            import dask
            dask.config.set(scheduler=cfg['program']['dask_scheduler'])

    # Run steps :
    st.start = cfg['program']['step_start']
    st.end = cfg['program']['step_end']
    st.go = True

    if not cfg['out']['db_name']:  # set name by 'path_cruise' name or parent if it has digits at start. priority for name  is  "*inclinometer*"
        for p in (lambda p: [p, p.parent])(cfg['in']['path_cruise']):
            m = re.match('(^[\d_]*).*', p.name)
            if m:
                break
        cfg['out']['db_name'] = f"{m.group(1).strip('_')}incl.h5"
    cfg['in']['path_cruise'].glob('*inclinometer*')
    dir_incl = next((d for d in cfg['in']['path_cruise'].glob('*inclinometer*') if d.is_dir()), cfg['in']['path_cruise'])
    db_path = dir_incl / cfg['out']['db_name']
    # ---------------------------------------------------------------------------------------------
    def fs(probe, name):
        return 5
        # if 'w' in name.lower():  # Baranov's wavegauge electronic
        #     return 5  # 10
        # if probe < 20 or probe in [23, 29, 30, 32, 33]:  # 30 [4, 11, 5, 12] + [1, 7, 13, 30]
        #     return 5
        # if probe in [21, 25, 26] + list(range(28, 35)):
        #     return 8.2
        # return 4.8


    def datetime64_str(time_str: Optional[str] = None) -> np.ndarray:
        """
        Reformat time_str to ISO 8601 or to 'NaT'. Used here for input in funcs that converts str to numpy.datetime64
        :param time_str: May be 'NaT'
        :return: ndarray of strings (tested for 1 element only) formatted by numpy.
        """
        return np.datetime_as_string(np.datetime64(time_str, 's'))


    probes = cfg['in']['probes'] or range(1, 41)  # sets default range, specify your values before line ---
    raw_root, subs_made = re.subn('INCL_?', 'INKL_', cfg['in']['probes_prefix'].upper())
    if st(1):  # Can not find additional not corrected files for same probe if already have any corrected in search path (move them out if need)
        i_proc_probe = 0  # counter of processed probes
        i_proc_file = 0  # counter of processed files
        # patten to identify only _probe_'s raw data files that need to correct '*INKL*{:0>2}*.[tT][xX][tT]':

        raw_parent = dir_incl / '_raw'
        dir_out = raw_parent / re.sub(r'[.\\/ ]', '_', cfg['in']['raw_subdir'])  # sub replaces multilevel subdirs to 1 level that correct_fun() can only make
        raw_parent /= cfg['in']['raw_subdir']
        for probe in probes:
            raw_found = []
            raw_pattern_file = cfg['in']['raw_pattern'].format(prefix=raw_root, number=probe)
            correct_fun = partial(correct_kondrashov_txt if subs_made else correct_baranov_txt, dir_out=dir_out)
            # if not archive:
            if (not '.zip' in cfg['in']['raw_subdir'].lower() and not '.rar' in cfg['in']['raw_subdir'].lower()) or raw_parent.is_dir():
                raw_found = list(raw_parent.glob(raw_pattern_file))
            if not raw_found:
                # Check if already have corrected files for probe generated by correct_kondrashov_txt(). If so then just use them
                raw_found = list(raw_parent.glob(f"{cfg['in']['probes_prefix']}{probe:0>2}.txt"))
                if raw_found:
                    print('corrected csv file', [r.name for r in raw_found], 'found')
                    correct_fun = lambda x: x
                elif not cfg['in']['raw_subdir']:
                    continue

            for file_in in (raw_found or open_csv_or_archive_of_them(raw_parent, binary_mode=False, pattern=raw_pattern_file)):
                file_in = correct_fun(file_in)
                if not file_in:
                    continue
                tbl = f"{cfg['in']['probes_prefix']}{probe:0>2}"
                # tbl = re.sub('^((?P<i>inkl)|w)_0', lambda m: 'incl' if m.group('i') else 'w',  # correct name
                #              re.sub('^[\d_]*|\*', '', file_in.stem).lower()),  # remove date-prefix if in name
                csv2h5(
                    [str(Path(__file__).parent / 'ini' / f"csv_inclin_{'Kondrashov' if subs_made else 'Baranov'}.ini"),
                    '--path', str(file_in),
                    '--blocksize_int', '50_000_000',  # 50Mbt
                    '--table', tbl,
                    '--db_path', str(db_path),
                    # '--log', str(scripts_path / 'log/csv2h5_inclin_Kondrashov.log'),
                    # '--b_raise_on_err', '0',  # ?
                    '--b_interact', '0',
                    '--fs_float', f'{fs(probe, file_in.stem)}',
                    '--dt_from_utc_seconds', str(cfg['in']['dt_from_utc'].total_seconds()),
                    '--b_del_temp_db', '1',
                    ] +
                   (
                   ['--csv_specific_param_dict', 'invert_magnitometr: True'
                    ] if subs_made else
                   ['--cols_load_list', "yyyy,mm,dd,HH,MM,SS,P,U"
                    ]
                   ),
                    **{'filter': {
                         'min_date': cfg['filter']['min_date'][probe],
                         'max_date': cfg['filter']['max_date'][probe],
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
                i_proc_file += 1
            else:
                print('no', raw_pattern_file, end=', ')
            i_proc_probe += 1
        print('Ok:', i_proc_probe, 'probes,', i_proc_file, 'files processed.')

    # Calculate velocity and average
    if st(2):
        # if aggregate_period_s is None then not average and write to *_proc_noAvg.h5 else loading from that h5 and writing to _proc.h5
        if not cfg['out']['aggregate_period_s']:
            cfg['out']['aggregate_period_s'] = [None, 2, 600, 3600 if 'w' in cfg['in']['probes_prefix'] else 7200]

        if cfg['in']['azimuth_add']:
            if 'Lat' in cfg['in']['azimuth_add']:
                from datetime import datetime
                # add magnetic declination,° for used coordinates
                # todo: get time
                azimuth_add = mag_dec(cfg['in']['azimuth_add']['Lat'], cfg['in']['azimuth_add']['Lon'], datetime(2020, 9, 10), depth=-1)
            else:
                azimuth_add = 0
            if 'constant' in cfg['in']['azimuth_add']:
                # and add constant. For example, subtruct declination at the calibration place if it was applied
                azimuth_add += cfg['in']['azimuth_add']['constant']  # add -6.65644183° to account for calibration in Kaliningrad
        for aggregate_period_s in cfg['out']['aggregate_period_s']:
            if aggregate_period_s is None:
                db_path_in = db_path
                db_path_out = db_path.with_name(f'{db_path.stem}_proc_noAvg.h5')
            else:
                db_path_in = db_path.with_name(f'{db_path.stem}_proc_noAvg.h5')
                db_path_out = f'{db_path.stem}_proc.h5'  # or separately: '_proc{aggregate_period_s}.h5'

            args = [Path(incl_h5clc.__file__).with_name(f'incl_h5clc_{db_path.stem}.yaml'),
                    # if no such file all settings are here
                    '--db_path', str(db_path_in),
                    # !   'incl.*|w\d*'  inclinometers or wavegauges w\d\d # 'incl09':
                    '--tables_list', 'incl.*' if not cfg['in']['probes'] else f"incl.*(?:{'|'.join('{:0>2}'.format(p) for p in cfg['in']['probes'])})",
                    '--aggregate_period', f'{aggregate_period_s}S' if aggregate_period_s else '',
                    '--out.db_path', str(db_path_out),
                    '--table', f'V_incl_bin{aggregate_period_s}' if aggregate_period_s else 'V_incl',
                    '--verbose', 'INFO',  #'DEBUG' get many numba messages
                    '--b_del_temp_db', '1',
                    # '--calc_version', 'polynom(force)',  # depreshiated
                    # '--chunksize', '20000',
                    # '--not_joined_h5_path', f'{db_path.stem}_proc.h5',
                    ]
            # if aggregate_period_s <= 5:   # [s], do not need split csv for big average interval
            #     args += (['--split_period', '1D'])
            if aggregate_period_s is None:  # proc. parameters (if we have saved proc. data then when aggregating we are not processing)
                args += (
                    ['--max_dict', 'M[xyz]:4096',
                     # Note: for Baranov's prog 4096 is not suited
                     # '--timerange_zeroing_dict', "incl19: '2019-11-10T13:00:00', '2019-11-10T14:00:00'\n,"  # not works - use kwarg
                     # '--timerange_zeroing_list', '2019-08-26T04:00:00, 2019-08-26T05:00:00'
                     '--split_period', '1D'
                    ] if subs_made else
                    ['--bad_p_at_bursts_starts_peroiod', '1H',
                    ])
            # csv splitted by 1day (default for no avg) and monolith csv if aggregate_period_s==600
            if aggregate_period_s not in cfg['out']['aggregate_period_s_not_to_text']:  # , 300, 600]:
                args += ['--text_path', str(db_path.parent / 'text_output')]
            kwarg = {'in': {
                'min_date': cfg['filter']['min_date'][0],
                'max_date': cfg['filter']['max_date'][0],
                'timerange_zeroing':  cfg['in']['timerange_zeroing'],
                'azimuth_add': azimuth_add
                }
                }
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


    # Calculate spectrograms.
    if st(3):  # Can be done at any time after step 1
        def raise_ni():
            raise NotImplementedError('Can not proc probes having different fs in one run: you need to do it separately')

        args = [
            Path(incl_h5clc.__file__).with_name(f'incl_h5spectrum{db_path.stem}.yaml'),
            # if no such file all settings are here
            '--db_path', str(db_path.with_name(f'{db_path.stem}_proc_noAvg.h5')),
            '--tables_list', f"{cfg['in']['probes_prefix']}.*",  # inclinometers or wavegauges w\d\d  ## 'w02', 'incl.*',
            # '--aggregate_period', f'{aggregate_period_s}S' if aggregate_period_s else '',

            '--min_date', datetime64_str(cfg['filter']['min_date'][0]),
            '--max_date', datetime64_str(cfg['filter']['max_date'][0]),  # '2019-09-09T16:31:00',  #17:00:00
            # '--max_dict', 'M[xyz]:4096',  # use if db_path is not ends with _proc_noAvg.h5 i.e. need calc velocity
            '--out.db_path', f"{db_path.stem.replace('incl', cfg['in']['probes_prefix'])}_proc_psd.h5",
            # '--table', f'psd{aggregate_period_s}' if aggregate_period_s else 'psd',
            '--fs_float', f"{fs(probes[0], cfg['in']['probes_prefix'])}",
            # (lambda x: x == x[0])(np.vectorize(fs)(probes, prefix))).all() else raise_ni()
            #
            # '--timerange_zeroing_list', '2019-08-26T04:00:00, 2019-08-26T05:00:00'
            # '--verbose', 'DEBUG',
            # '--chunksize', '20000',
            '--b_interact', '0',
            ]
        if 'w' in cfg['in']['probes_prefix']:
            args += ['--split_period', '1H',
                     '--dt_interval_minutes', '10',  # burst mode
                     '--fmin', '0.0001',
                     '--fmax', '4'
                     ]
        else:
            args += ['--split_period', '2H',
                     '--fmin', '0.0004',  #0.0004
                     '--fmax', '1.05'
                     ]

        incl_h5spectrum.main(args)

    # Draw in Veusz
    if st(4):
        b_images_only = True  # False
        pattern_path = db_path.parent / r'vsz_5min\191119_0000_5m_incl19.vsz'  # r'vsz_5min\191126_0000_5m_w02.vsz'
        if not b_images_only:
            pattern_bytes_slice_old = re.escape(b'((5828756, 5830223, None),)')

        # Length of not adjacent intervals, s (set None to not allow)
        period = '1D'
        length = '5m'  # period  # '1D'

        dt_custom_s = pd_period_to_timedelta(length) if length != period else None  # None  #  60 * 5

        if True:
            # Load starts and assign ends
            t_intervals_start = pd.read_csv(
                cfg['in']['path_cruise'] / r'vsz+h5_proc\intervals_selected.txt',
                converters={'time_start': lambda x: np.datetime64(x, 'ns')}, index_col=0).index
            edges = (
                pd.DatetimeIndex(t_intervals_start), pd.DatetimeIndex(t_intervals_start + dt_custom_s))  # np.zeros_like()
        else:
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

        for i, probe in enumerate(probes):
            probe_name = f"{cfg['in']['probes_prefix']}{probe:02}"  # table name in db
            l.info('Draw %s in Veusz: %d intervals...', probe_name, edges[0].size)
            # for i_interval, (t_interval_start, t_interval_end) in enumerate(zip(pd.DatetimeIndex([t_interval_start]).append(t_intervals_end[:-1]), t_intervals_end), start=1):

            cfg_vp = {'veusze': None}
            for i_interval, (t_interval_start, t_interval_end) in enumerate(zip(*edges), start=1):

                # if i_interval < 23: #<= 0:  # TEMPORARY Skip this number of intervals
                #     continue
                if period != length:
                    t_interval_start = t_interval_end - pd.Timedelta(dt_custom_s, 's')

                try:  # skipping absent probes
                    start_end = h5q_interval2coord(
                        db_path=str(db_path),
                        table=f'/{probe_name}',
                        t_interval=(t_interval_start, t_interval_end))
                    if not len(start_end):
                        break  # no data
                except KeyError:
                    break  # device name not in specified range, go to next name

                pattern_path_new = pattern_path.with_name(f"{t_interval_start:%y%m%d_%H%M}_{length}_{probe_name}.vsz")

                # Modify pattern file
                if not b_images_only:
                    probe_name_old = re.match('.*((?:incl|w)\d*).*', pattern_path.name).groups()[0]
                    bytes_slice = bytes('(({:d}, {:d}, None),)'.format(*(start_end + np.int32([-1, 1]))), 'ascii')


                    def f_replace(line):
                        """
                        Replace in file
                        1. probe name
                        2. slice
                        """
                        # if i_interval == 1:
                        line, ok = re.subn(bytes(probe_name_old, 'ascii'), bytes(probe_name, 'ascii'), line)
                        if ok:  # can be only in same line
                            line = re.sub(pattern_bytes_slice_old, bytes_slice, line)
                        return line


                    if not rep_in_file(pattern_path, pattern_path_new, f_replace=f_replace):
                        l.warning('Veusz pattern not changed!')
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
                     '--path', str(db_path),  # use for custom loading from db and some source is required
                     '--tables_list', f'/{probe_name}',  # 181022inclinometers/ \d*
                     '--pattern_path', str(pattern_path_new),
                     # fr'd:\workData\BalticSea\190801inclinometer_Schuka\{probe_name}_190807_1D.vsz',
                     # str(db_path.parent / dir_incl / f'{probe_name}_190211.vsz'), #warning: create file with small name
                     # '--before_next', 'restore_config',
                     # '--add_to_filename', f"_{t_interval_start:%y%m%d_%H%M}_{length}",
                     '--filename_fun', f'lambda tbl: "{pattern_path_new.name}"',
                     '--add_custom_list',
                     'USEtime',  # nAveragePrefer',
                     '--add_custom_expressions_list',
                     txt_time_range,
                     # + """
                     # ", 5"
                     # """,
                     '--b_update_existed', 'True',
                     '--export_pages_int_list', '1, 2',  # 0 for all '6, 7, 8',  #'1, 2, 3'
                     # '--export_dpi_int', '200',
                     '--export_format', 'emf',
                     '--b_interact', '0',
                     '--b_images_only', f'{b_images_only}',
                     '--return', '<embedded_object>',  # reuse to not bloat memory
                     ],
                    veusze=cfg_vp['veusze'])


if __name__ == '__main__':
    main()