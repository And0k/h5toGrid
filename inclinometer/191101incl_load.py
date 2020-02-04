"""
Runs steps:
1. csv2h5(): save inclinometer data from Kondrashov format to DB
2. veuszPropagate(): draw using Veusz pattern

Specify:
    - control execution parameters:

        :start: float, start step. Begins from step >= start
        :go: bool. Control of execution next steps. Insert ``go = True`` or ``go = False`` where need

    - data processing parameters:

        :path_cruise:
        :dir_incl: or :in_file:
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

import numpy as np
import pandas as pd

# import my scripts
drive_d = Path('D:/' if sys.platform == 'win32' else '/mnt/D')  # allows to run on both my Linux and Windows systems:
scripts_path = drive_d.joinpath('Work/_Python3/And0K/h5toGrid/scripts')
sys.path.append(str(Path(scripts_path).parent.resolve()))  # os.getcwd()
from to_pandas_hdf5.csv2h5 import main as csv2h5
from to_pandas_hdf5.csv_specific_proc import correct_kondrashov_txt, rep_in_file
from to_pandas_hdf5.h5_dask_pandas import h5q_interval2coord
from inclinometer.h5inclinometer_coef import h5copy_coef
import inclinometer.incl_h5clc as incl_h5clc
import inclinometer.incl_h5spectrum as incl_h5spectrum
import veuszPropagate
from utils_time import pd_period_to_timedelta
from utils2init import path_on_drive_d, init_logging

# l = logging.getLogger(__name__)
l = init_logging(logging, None, None, 'INFO')
# Directory where inclinometer data will be stored
path_cruise = path_on_drive_d(r'd:\WorkData\BalticSea\191210_Pregolya,Lagoon-inclinometer'
                              )

r"""
d:\WorkData\_experiment\_2019\inclinometer\200117_tank[23,30,32]
d:\workData\BalticSea\191119_Filino\inclinometer
d:\workData\BalticSea\191108_Filino\inclinometer
d:\WorkData\BlackSea\191029_Katsiveli\inclinometer

d:\WorkData\_experiment\_2019\inclinometer\191106_tank_ex1[1,13,14,16]
d:\WorkData\BalticSea\190806_Yantarniy\inclinometer
d:\workData\BalticSea\190801inclinometer_Schuka
d:\workData\BalticSea\190713_ABP45\inclinometer
d:\workData\BalticSea\190817_ANS42\inclinometer
d:\workData\BalticSea\180418_Svetlogorsk\inclinometer
d:\WorkData\_experiment\_2018\inclinometer\180406_tank[9]

d:\WorkData\_experiment\_2019\inclinometer\190711_tank[1,4,5,7,11,12]
d:\WorkData\_experiment\_2018\inclinometer\181004_tank[1-20]
d:\workData\BalticSea\190801inclinometer_Schuka
d:\WorkData\_experiment\_2019\inclinometer\190902_compas_calibr-byMe
d:\WorkData\_experiment\_2019\inclinometer\190917_intertest
d:\workData\BalticSea\190806_Yantarniy\inclinometer
d:\workData\BalticSea\190817_ANS42\inclinometer

dt_from_utc = defaultdict(
    lambda: '19-07-09T17:48:00',
)
    lambda: '19-07-03T15:15:00', #'19-07-03T12:00:00',
    {
        12: '19-07-03T15:00:00',
        25: '19-07-03T14:05:00',
        26: '19-07-03T14:05:00',
    }
)
"""

# Note: Not affects steps 2, 3:
probes = [7, 23, 30,
          32]  # [3, 13] [7][12,19,14,15,7,11,4,9]  [11, 9, 10, 5, 13, 16] range(1, 20)  #[2]   #  [29, 30, 33][3, 16, 19]   # [4, 14]  #  [9] #[1,4,5,7,11,12]  # [17, 18] # range(30, 35) #  # [10, 14] [21, 23] #range(1, 40)   [25,26]  #, [17, 18]
# set None for auto:
db_name = '191210incl.h5'  # None # 191224incl 191108incl.h5 190806incl#29,30,33.h5 190716incl 190806incl 180418inclPres

dir_incl = '' if 'inclinometer' in str(path_cruise) else 'inclinometer/'
if not db_name:
    db_name = re.match('(^[\d_]*).*', (path_cruise if dir_incl else path_cruise.parent).name).group(1).strip(
        '_') + 'incl.h5'  # group(0) if db name == cruise dir name
db_path = path_cruise / db_name  # _z  '190210incl.h5' 'ABP44.h5'

# dafault and specific to probe limits
date_min = defaultdict(
    lambda: '2019-12-10T13:00',
    # '2020-01-17T13:00''2019-08-07T14:10:00''2019-11-19T16:00''2019-11-19T12:30''2019-11-08T12:20' 2019-11-08T12:00 '2019-11-06T12:35''2019-11-06T10:50''2019-07-16T17:00:00' 2019-06-20T14:30:00  '2019-07-21T20:00:00', #None,
    {  # 0: #'2019-08-07T16:00:00' '2019-08-17T18:00', 0: '2018-04-18T07:15',
        # 1: '2018-05-10T15:00',
        })
date_max = defaultdict(
    lambda: '2019-12-26T16:00',
    # '2020-01-17T17:00''2019-09-09T17:00:00', #'2019-12-04T00:00', # 20T14:00 # '2019-11-19T14:30',  '2019-11-06T13:34','2019-11-06T12:20' '2019-08-31T16:38:00' '2019-07-19T17:00:00', # '2019-08-18T01:45:00', # None,
    {  # 0:  # '2019-09-09T17:00:00' '2019-09-06T04:00:00' '2019-08-27T02:00', 0: '2018-05-07T11:55',
        # 1: '2018-05-30T10:30',
        })


def datetime64_str(time_str: Optional[str]) -> str:
    """ return: string formatted right for input parameters. May be 'NaT'"""
    return np.datetime_as_string(np.datetime64(time_str, 's'))


start = 2
end = 3  # inclusive
# ---------------------------------------------------------------------------------------------
go = True  # False #


def st(current: int) -> bool:
    """
    :param current: step
    :return: True if start <= current <= max(start, end)): allows one step if end <= start
    """
    if (start <= current <= max(start, end)) and go:
        print(f'step {current}')
        return True
    return False


# ---------------------------------------------------------------------------------------------
probe_type = 'incl'  # table name prefix in db: 'incl' - inclinometer, 'w' - wavegauge


def fs(probe, name):
    if 'w' in name.lower():  # Baranov's wavegauge electronic
        return 10
    if probe < 20 or probe in [23, 29, 30, 32, 33]:  # 30 [4, 11, 5, 12] + [1, 7, 13, 30]
        return 5
    if probe in [21, 25, 26] + list(range(28, 35)):
        return 8.2
    return 4.8


if st(1):
    i_proc_probe = 0  # counter of processed probes
    i_proc_file = 0  # counter of processed files
    for probe in probes:
        source_pattern = f'*INKL*{probe:0>2}*.[tT][xX][tT]'  # remove some * if load other probes!
        find_in_dir = (path_cruise / (dir_incl + '_raw')).glob
        source_found = list(find_in_dir(source_pattern))
        if not source_found:  # if have only output files of correct_kondrashov_txt() then just use them
            source_found = find_in_dir(f'incl{probe:0>2}.txt')
            correct_fun = lambda x: x
        else:
            correct_fun = correct_kondrashov_txt
        for in_file in source_found:
            # '_source/incl_txt/180510_INKL10.txt' # r'_source/180418_INKL09.txt'
            in_file = correct_fun(in_file)
            if not in_file:
                continue

            csv2h5([scripts_path / 'ini/csv_Kondrashov_inclin.ini',
                    '--path', str(in_file),
                    '--blocksize_int', '10000000',  # 10Mbt
                    '--table', re.sub('^inkl_0', 'incl',
                                      re.sub('^[\d_]*|\*', '', in_file.stem).lower()),
                    '--date_min', datetime64_str(date_min[i_proc_file]),
                    '--date_max', datetime64_str(date_max[i_proc_file]),
                    '--db_path', str(db_path),
                    '--log', str(scripts_path / 'log/csv2h5_Kondrashov_inclin.log'),
                    # '--b_raise_on_err', '0',  # ?!
                    '--b_interact', '0',
                    '--fs_float', f'{fs(probe, in_file.stem)}',
                    # '--dt_from_utc_seconds', "{}".format(int((np.datetime64('00', 'Y') - np.datetime64(dt_from_utc[probe]
                    #  #   ['19-06-24T10:19:00', '19-06-24T10:21:30'][i_proc_probe]
                    #     ))/np.timedelta64(1,'s')))
                    '--csv_specific_param_dict', 'invert_magnitometr: True',
                    ])

            # Get coefs:
            db_coefs = r'd:\WorkData\_experiment\_2019\inclinometer\190710_compas_calibr-byMe\190710incl.h5'
            try:
                tbl = f'incl{probe:0>2}'
                l.info(f"Adding coefficients to {db_path}/{tbl} from {db_coefs}")
                h5copy_coef(db_coefs, db_path, tbl)
            except KeyError as e:  # Unable to open object (component not found)
                l.warning('Coef is not copied!')
                # todo write some dummy coefficients to can load Veusz patterns
            i_proc_file += 1
        i_proc_probe += 1
    print(f'Ok! ({i_proc_probe} probes, {i_proc_file} files processed)')

# Calculate velocity and average
if st(2):
    # if aggregate_period_s is None then not average and write to *_proc_noAvg.h5 else loading from that h5 and writing to _proc.h5
    for aggregate_period_s in [300]:  # [None, 2, 600, 7200]  [None], [2, 600, 7200], [3600]
        if aggregate_period_s is None:
            db_path_in = db_path
            db_path_out = db_path.with_name(f'{db_path.stem}_proc_noAvg.h5')
        else:
            db_path_in = db_path.with_name(f'{db_path.stem}_proc_noAvg.h5')
            db_path_out = f'{db_path.stem}_proc300.h5'  # !

        args = [Path(incl_h5clc.__file__).with_name(f'incl_h5clc_{db_path.stem}.yaml'),
                # if no such file all settings are here
                '--db_path', str(db_path_in),
                '--tables_list', 'incl.*|w02',  # inclinometers or wavegauges w\d\d # 'incl09',
                '--aggregate_period', f'{aggregate_period_s}S' if aggregate_period_s else '',
                '--date_min', datetime64_str(date_min[0]),  # '2019-08-18T06:00:00',
                '--date_max', datetime64_str(date_max[0]),  # '2019-09-09T16:31:00',  #17:00:00
                '--output_files.db_path', str(db_path_out),
                '--table', f'V_incl_bin{aggregate_period_s}' if aggregate_period_s else 'V_incl',
                '--verbose', 'DEBUG',
                # '--calc_version', 'polynom(force)',  # depreshiated!
                # '--chunksize', '20000',
                # '--not_joined_h5_path', f'{db_path.stem}_proc.h5',
                ]
        if not '_proc_' in db_path_in.stem:

            args += [
                '--max_dict', 'M[xyz]:4096',  # Note: for Baranov's prog 4096 is not suited!
                # '--timerange_zeroing_dict', "incl19: '2019-11-10T13:00:00', '2019-11-10T14:00:00'\n,"  # not works - use kwarg
                # '--timerange_zeroing_list', '2019-08-26T04:00:00, 2019-08-26T05:00:00'
                ]
            kwarg = {}  # 'in': {'timerange_zeroing': {'incl19': ['2019-11-14T06:30:00', '2019-11-14T06:50:00']}}
        else:
            kwarg = {}
        # csv splitted by 1day (default for no avg) and monolit csv if aggregate_period_s==600
        if aggregate_period_s in [None, 300, 600]:
            args += ['--not_joined_csv_path', str(db_path.parent / 'csv')]

        incl_h5clc.main(args, **kwarg)

# Calculate spectrograms.
if st(3):  # Can be done at any time after step 1
    def raise_ni():
        raise NotImplementedError('Different fs - Calculate separately!')


    args = [
        Path(incl_h5clc.__file__).with_name(f'incl_h5spectrum{db_path.stem}.yaml'),
        # if no such file all settings are here
        '--db_path', str(db_path.with_name(f'{db_path.stem}_proc_noAvg.h5')),
        '--tables_list', 'incl.*|w02',  # inclinometers or wavegauges w\d\d  ## 'w02', 'incl.*',
        # '--aggregate_period', f'{aggregate_period_s}S' if aggregate_period_s else '',

        '--date_min', datetime64_str(date_min[0]),
        '--date_max', datetime64_str(date_max[0]),  # '2019-09-09T16:31:00',  #17:00:00
        # '--max_dict', 'M[xyz]:4096',  # use if db_path is not ends with _proc_noAvg.h5 i.e. need calc velocity
        '--output_files.db_path', f'{db_path.stem}_proc_psd.h5',
        # '--table', f'psd{aggregate_period_s}' if aggregate_period_s else 'psd',
        '--fs_float', (f'{fs(probes[0], probe_type)}' if (
            (lambda x: x == x[0])(np.vectorize(fs)(probes, probe_type))).all() else raise_ni()),
        #
        # '--timerange_zeroing_list', '2019-08-26T04:00:00, 2019-08-26T05:00:00'
        # '--verbose', 'DEBUG',
        # '--chunksize', '20000',
        '--b_interact', '0',
        ]
    if 'w' in probe_type:
        args += ['--split_period', '1H',
                 '--dt_interval_minutes', '10',
                 '--fmin', '0.0001',
                 '--fmax', '4'
                 ]
    else:
        args += ['--split_period', '2H',
                 '--fmin', '0.0004',
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
            path_cruise / r'vsz+h5_proc\intervals_selected.txt',
            converters={'time_start': lambda x: np.datetime64(x, 'ns')}, index_col=0).index
        edges = (
            pd.DatetimeIndex(t_intervals_start), pd.DatetimeIndex(t_intervals_start + dt_custom_s))  # np.zeros_like()
    else:
        # Generate periodic intervals
        t_interval_start, t_intervals_end = intervals_from_period(datetime_range=np.array(
            [date_min[0], date_max[0]],
            # ['2018-08-11T18:00:00', '2018-09-06T00:00:00'],
            # ['2019-02-11T13:05:00', '2019-03-07T11:30:00'],
            # ['2018-11-16T15:19', '2018-12-14T14:35'],
            # ['2018-10-22T12:30', '2018-10-27T06:30:00'],
            'datetime64[s]'), period=period)
        edges = (pd.DatetimeIndex([t_interval_start]).append(t_intervals_end[:-1]),
                 pd.DatetimeIndex(t_intervals_end))

    for i, probe in enumerate(probes):
        probe_name = f'{probe_type}{probe:02}'  # table name in db
        l.info('Draw %s in Veusz: %d intervals...', probe_name, edges[0].size)
        # for i_interval, (t_interval_start, t_interval_end) in enumerate(zip(pd.DatetimeIndex([t_interval_start]).append(t_intervals_end[:-1]), t_intervals_end), start=1):

        cfg_vp = {'veusze': None}
        for i_interval, (t_interval_start, t_interval_end) in enumerate(zip(*edges), start=1):

            # if i_interval < 23: #<= 0:  # TEMPORARY! Skip this number of intervals
            #     continue
            if period != length:
                t_interval_start = t_interval_end - pd.Timedelta(dt_custom_s, 's')

            try:
                start_end = h5q_interval2coord(
                    {'db_path': str(db_path), 'table': f'/{probe_name}'}, (t_interval_start, t_interval_end))
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
                 # str(db_path.parent.joinpath(dir_incl + f'{probe_name}_190211.vsz')), #warning: create file with small name
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

    # # Irregular intervals
    # # t_intervals_end = pd.DatetimeIndex(['2018-11-19T00:00', '2018-11-30T00:00', '2018-12-05T00:00'])',
    # # t_intervals_end = pd.DatetimeIndex(['2018-10-22T12:30', '2018-10-27T06:30:00'])
    # t_intervals_end = pd.to_datetime(
    #     np.array(['2019-02-16T08:00', '2019-02-17T04:00', '2019-02-18T00:00', '2019-02-28T00:00',
    #               '2019-02-14T12:00', '2019-02-15T12:00', '2019-02-16T23:50',
    #               '2019-02-20T00:00', '2019-02-20T22:00', '2019-02-22T06:00', '2019-02-23T03:00',
    #               '2019-02-13T11:00', '2019-02-14T13:00', '2019-02-16T23:00', '2019-02-18T12:00',
    #               '2019-02-19T00:00', '2019-02-19T16:00', '2019-02-21T00:00', '2019-02-22T02:00', '2019-02-23T00:00',
    #               '2019-02-26T06:00', '2019-02-26T16:00', '2019-02-28T06:00'
    #               ], dtype='datetime64[s]') + np.timedelta64('5', 'm'))
    #    #  np.datetime64(['2019-02-13T07:33:11', '2019-02-14T09:34:09',
    #    # '2019-02-16T19:41:47', '2019-02-18T07:14:54',
    #    # '2019-02-18T11:51:35', '2019-02-18T23:01:40',
    #    # '2019-02-18T22:48:51', '2019-02-19T11:55:59',
    #    # '2019-02-20T22:13:44', '2019-02-22T21:52:33'],
    # t_interval_start = t_intervals_end[0] - np.timedelta64('5', 'm')
