"""
Runs steps:
1. csv2h5(): save inclinometer data from Kondrashov format to DB
2. Calculate velocity and average
3. Calculate spectrograms.
4. veuszPropagate(): draw using Veusz pattern

Specify:
    - control execution parameters:
        :start: float, start step. Begins from step >= start
        Raw file names should be matched by regex: "[\d_]*(W|INKL)_\d\d*.txt"

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
from functools import partial
import numpy as np
import pandas as pd

# import my scripts
drive_d = Path('D:/' if sys.platform == 'win32' else '/mnt/D')  # allows to run on both my Linux and Windows systems:
scripts_path = drive_d.joinpath('Work/_Python3/And0K/h5toGrid/scripts')
sys.path.append(str(Path(scripts_path).parent.resolve()))  # os.getcwd()
from to_pandas_hdf5.csv2h5 import main as csv2h5
from to_pandas_hdf5.csv_specific_proc import correct_kondrashov_txt, rep_in_file, correct_baranov_txt
from to_pandas_hdf5.h5_dask_pandas import h5q_interval2coord
from inclinometer.h5inclinometer_coef import h5copy_coef
import inclinometer.incl_h5clc as incl_h5clc
import inclinometer.incl_h5spectrum as incl_h5spectrum
import veuszPropagate
from utils_time import pd_period_to_timedelta
from utils2init import path_on_drive_d, init_logging, open_csv_or_archive_of_them, st

# l = logging.getLogger(__name__)
l = init_logging(logging, None, None, 'INFO')

if True:  # False. Experimental speedup but takes memory
    from dask.cache import Cache
    cache = Cache(2e9)  # Leverage two gigabytes of memory
    cache.register()    # Turn cache on globally
if False:  #  True:  # False:  #
    l.warning('using "synchronous" scheduler for debugging')
    import dask
    dask.config.set(scheduler='synchronous')

# Directory where inclinometer data will be stored
path_cruise = path_on_drive_d(r'd:\WorkData\BalticSea\200628_Pregolya,Lagoon-inclinometer'
                              )

r"""
d:\WorkData\BalticSea\200630_AI55\inclinometer
d:\WorkData\_experiment\inclinometer\200610_tank_ex[4,5,7,9,10,11][3,12,13,14,15,16,19]
d:\WorkData\BalticSea\200514_Pregolya,Lagoon-inclinometer
d:\WorkData\BalticSea\200317_Pregolya,Lagoon-inclinometer
d:\WorkData\BalticSea\191210_Pregolya,Lagoon-inclinometer
d:\WorkData\_experiment\inclinometer\200117_tank[23,30,32]
d:\workData\BalticSea\191119_Filino\inclinometer
d:\workData\BalticSea\191108_Filino\inclinometer
d:\WorkData\BlackSea\191029_Katsiveli\inclinometer

d:\WorkData\_experiment\inclinometer\191106_tank_ex1[1,13,14,16]
d:\WorkData\BalticSea\190806_Yantarniy\inclinometer
d:\workData\BalticSea\190801inclinometer_Schuka
d:\workData\BalticSea\190713_ABP45\inclinometer
d:\workData\BalticSea\190817_ANS42\inclinometer
d:\workData\BalticSea\180418_Svetlogorsk\inclinometer
d:\WorkData\_experiment\_2018\inclinometer\180406_tank[9]

d:\WorkData\_experiment\inclinometer\190711_tank[1,4,5,7,11,12]
d:\WorkData\_experiment\_2018\inclinometer\181004_tank[1-20]
d:\workData\BalticSea\190801inclinometer_Schuka
d:\WorkData\_experiment\inclinometer\190902_compas_calibr-byMe
d:\WorkData\_experiment\inclinometer\190917_intertest
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
# Pattern modifier to search row data in archives under _row subdir. Set to '' if data unpacked.
raw_archive_name = 'INKL_009.rar'
# 'Преголя и залив №4.rar', 'INKL_014.ZIP', 'экс2кти100620.zip' 'Преголя и залив №3.rar' 'Преголя и залив №2.rar'

# Note: Not affects steps 2, 3, set empty list to load all:
probes = []  # 14, 7, 23, 30, 32  [3, 13] [7][12,19,14,15,7,11,4,9]  [11, 9, 10, 5, 13, 16] range(1, 20)  #[2]   #  [29, 30, 33][3, 16, 19]   # [4, 14]  #  [9] #[1,4,5,7,11,12]  # [17, 18] # range(30, 35) #  # [10, 14] [21, 23] #   [25,26]  #, [17, 18]
# set None for auto:
db_name = None  # '191210incl.h5'  # 191224incl 191108incl.h5 190806incl#29,30,33.h5 190716incl 190806incl 180418inclPres

# dafault and specific to probe limits (use "i_proc_file" index instead "probe" if many files for same probe)
min_date = defaultdict(
    lambda: '2020-06-28T18:00', {
#'2020-06-28T15:30','2020-06-30T22:00','2020-05-14T13:00'
# 13: '2020-05-14T11:49:00',
# 3: '2020-05-14T12:02:50',
# 7: '2020-05-14T12:15:40',
# 9: '2020-05-14T12:24:40',
# 5: '2020-05-14T12:33:30'
})
#  '2020-03-17T13:00','2019-12-10T13:00','2020-01-17T13:00''2019-08-07T14:10:00''2019-11-19T16:00''2019-11-19T12:30''2019-11-08T12:20' 2019-11-08T12:00 '2019-11-06T12:35''2019-11-06T10:50''2019-07-16T17:00:00' 2019-06-20T14:30:00  '2019-07-21T20:00:00', #None,
# 0: #'2019-08-07T16:00:00' '2019-08-17T18:00', 0: '2018-04-18T07:15',# 1: '2018-05-10T15:00',

max_date = defaultdict(lambda: '2020-07-27T09:00', {  # 'now'
# '2020-06-28T16:30','2020-07-13T14:10'
# 13: '2020-05-14T12:02',
# 3: '2020-05-14T12:15',
# 7: '2020-05-14T12:24',
# 9: '2020-05-14T12:33',
# 5: '2020-05-14T12:42',
})
# '2019-12-26T16:00','2020-01-17T17:00''2019-09-09T17:00:00', #'2019-12-04T00:00', # 20T14:00 # '2019-11-19T14:30',  '2019-11-06T13:34','2019-11-06T12:20' '2019-08-31T16:38:00' '2019-07-19T17:00:00', # '2019-08-18T01:45:00', # None,
# 0:  # '2019-09-09T17:00:00' '2019-09-06T04:00:00' '2019-08-27T02:00', 0: '2018-05-07T11:55', # 1: '2018-05-30T10:30',

# Run steps (inclusive):
st.start = 2 # 1
st.end = 1 # 3
st.go = True  # False #

prefix = 'incl'  # 'incl' or 'w'  # table name prefix in db and in raw files (to find raw fales name case will be UPPER anyway): 'incl' - inclinometer, 'w' - wavegauge

dir_incl = '' if 'inclinometer' in str(path_cruise) else 'inclinometer'
if not db_name:  # then name by cruise dir:
    db_name = re.match('(^[\d_]*).*', (path_cruise.parent if dir_incl or path_cruise.name.startswith('inclinometer') else path_cruise).name
                       ).group(1).strip('_') + 'incl.h5'  # group(0) if db name == cruise dir name
db_path = path_cruise / db_name  # _z  '190210incl.h5' 'ABP44.h5', / '200514incl_experiment.h5'
# ---------------------------------------------------------------------------------------------
def fs(probe, name):
    if 'w' in name.lower():  # Baranov's wavegauge electronic
        return 5  # 10
    if probe < 20 or probe in [23, 29, 30, 32, 33]:  # 30 [4, 11, 5, 12] + [1, 7, 13, 30]
        return 5
    if probe in [21, 25, 26] + list(range(28, 35)):
        return 8.2
    return 4.8


def datetime64_str(time_str: Optional[str] = None) -> np.ndarray:
    """
    Reformat time_str to ISO 8601 or to 'NaT'. Used here for input in funcs that converts str to numpy.datetime64
    :param time_str: May be 'NaT'
    :return: ndarray of strings (tested for 1 element only) formatted by numpy.
    """
    return np.datetime_as_string(np.datetime64(time_str, 's'))


probes = probes or range(9, 40)  # sets default range, specify your values before line ---
if st(1):  # Can not find additional not corrected files for same probe if already have any corrected in search path (move them out if need)
    i_proc_probe = 0  # counter of processed probes
    i_proc_file = 0  # counter of processed files
    # patten to identify only _probe_'s raw data files that need to correct '*INKL*{:0>2}*.[tT][xX][tT]':
    raw_pattern = f'*{prefix.replace("incl","inkl").upper()}_{{:0>3}}*.[tT][xX][tT]'
    raw_parent = path_cruise / dir_incl / '_raw'
    for probe in probes:
        correct_fun = partial(correct_kondrashov_txt if prefix == 'incl' else correct_baranov_txt, dir_out=raw_parent)
        raw_found = []
        if not raw_archive_name:
            raw_found = list(raw_parent.glob(raw_pattern.format(probe)))
        if not raw_found:
            # Check if already have corrected files for probe generated by correct_kondrashov_txt(). If so then just use them
            raw_found = list(raw_parent.glob(f'{prefix}{probe:0>2}.txt'))
            if raw_found:
                print('corrected csv file', [r.name for r in raw_found], 'found')
                correct_fun = lambda x: x
            elif not raw_archive_name:
                continue

        for in_file in (raw_found or open_csv_or_archive_of_them(raw_parent / raw_archive_name,
                                                                 binary_mode=False, pattern=raw_pattern.format(probe))):
            in_file = correct_fun(in_file)
            if not in_file:
                continue

            csv2h5(
                [scripts_path / f"cfg/csv_inclin_{'Kondrashov' if prefix == 'incl' else 'Baranov'}.ini",
                '--path', str(in_file),
                '--blocksize_int', '50_000_000',  # 50Mbt
                '--table', re.sub('^((?P<i>inkl)|w)_0', lambda m: 'incl' if m.group('i') else 'w',  # correct name
                                  re.sub('^[\d_]*|\*', '', in_file.stem).lower()),  # remove date-prefix if in name
                '--min_date', datetime64_str(min_date[probe]),  # use i_proc_file instead probe if many files for same probe
                '--max_date', datetime64_str(max_date[probe]),
                '--db_path', str(db_path),
                # '--log', str(scripts_path / 'log/csv2h5_inclin_Kondrashov.log'),
                # '--on_bad_lines', 'warn',  # ?
                '--b_interact', '0',
                '--fs_float', f'{fs(probe, in_file.stem)}',
                # '--dt_from_utc_seconds', "{}".format(int((np.datetime64('00', 'Y') - np.datetime64(dt_from_utc[probe]
                #  #   ['19-06-24T10:19:00', '19-06-24T10:21:30'][i_proc_probe]
                #     ))/np.timedelta64(1,'s')))
                ] +
               (
               ['--csv_specific_param_dict', 'invert_magnitometr: True'
                ] if prefix == 'incl' else
               ['--cols_load_list', "yyyy,mm,dd,HH,MM,SS,P,U"
                ]
               )
                 )

            # Get coefs:
            db_coefs = r'd:\WorkData\~configuration~\inclinometr\190710incl.h5'
            try:
                tbl = f'{prefix}{probe:0>2}'
                l.info(f"Adding coefficients to {db_path}/{tbl} from {db_coefs}")
                h5copy_coef(db_coefs, db_path, tbl)
            except KeyError as e:  # Unable to open object (component not found)
                l.warning('Coef is not copied!')
                # todo write some dummy coefficients to can load Veusz patterns
            i_proc_file += 1
        else:
            print(probe, end=': no, ')
        i_proc_probe += 1
    print('Ok:', i_proc_probe, 'probes,', i_proc_file, 'files processed.')

# Calculate velocity and average
if st(2):
    # if aggregate_period_s is None then not average and write to *_proc_noAvg.h5 else loading from that h5 and writing to _proc.h5
    for aggregate_period_s in [None, 2, 600, 3600 if 'w' in prefix else 7200]:  # 2,, 7200  # 300, 600,  [None], [None, 2, 600, 3600 if 'w' in prefix else 7200], [3600]
        if aggregate_period_s is None:
            db_path_in = db_path
            db_path_out = db_path.with_name(f'{db_path.stem}_proc_noAvg.h5')
        else:
            db_path_in = db_path.with_name(f'{db_path.stem}_proc_noAvg.h5')
            db_path_out = f'{db_path.stem}_proc.h5'  # or separately: '_proc{aggregate_period_s}.h5'

        args = [Path(incl_h5clc.__file__).with_name(f'incl_h5clc_{db_path.stem}.yaml'),
                # if no such file all settings are here
                '--db_path', str(db_path_in),
                '--tables_list', 'incl.*',  #!   'incl.*|w\d*'  inclinometers or wavegauges w\d\d # 'incl09',
                '--aggregate_period', f'{aggregate_period_s}S' if aggregate_period_s else '',
                '--min_date', datetime64_str(min_date[0]),  # '2019-08-18T06:00:00',
                '--max_date', datetime64_str(max_date[0]),  # '2019-09-09T16:31:00',  #17:00:00
                '--out.db_path', str(db_path_out),
                '--table', f'V_incl_bin{aggregate_period_s}' if aggregate_period_s else 'V_incl',
                '--verbose', 'INFO',  #'DEBUG' get many numba messages
                # '--calc_version', 'polynom(force)',  # depreshiated
                # '--chunksize', '20000',
                # '--not_joined_db_path', f'{db_path.stem}_proc.h5',
                ]
        if aggregate_period_s is None:  # proc. parameters (if we have saved proc. data then when aggregating we are not processing)

            args += (
                ['--max_dict', 'M[xyz]:4096',
                 # Note: for Baranov's prog 4096 is not suited
                 # '--time_range_zeroing_dict', "incl19: '2019-11-10T13:00:00', '2019-11-10T14:00:00'\n,"  # not works - use kwarg
                 # '--time_range_zeroing_list', '2019-08-26T04:00:00, 2019-08-26T05:00:00'
                ] if prefix == 'incl' else
                ['--bad_p_at_bursts_starts_peroiod', '1H',
                ])
            kwarg = {'in': {'time_range_zeroing': {'incl14': ['2020-07-10T21:31:00', '2020-07-10T21:39:00']}}}  #{} {'incl14': ['2019-11-14T06:30:00', '2019-11-14T06:50:00']}}}
        else:
            kwarg = {}
        # csv splitted by 1day (default for no avg) and monolit csv if aggregate_period_s==600
        if aggregate_period_s in [None, 300, 600]:
            args += ['--text_path', str(db_path.parent / 'text_output')]


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
        '--tables_list', f'{prefix}.*',  # inclinometers or wavegauges w\d\d  ## 'w02', 'incl.*',
        # '--aggregate_period', f'{aggregate_period_s}S' if aggregate_period_s else '',

        '--min_date', datetime64_str(min_date[0]),
        '--max_date', datetime64_str(max_date[0]),  # '2019-09-09T16:31:00',  #17:00:00
        # '--max_dict', 'M[xyz]:4096',  # use if db_path is not ends with _proc_noAvg.h5 i.e. need calc velocity
        '--out.db_path', f'{db_path.stem.replace("incl", prefix)}_proc_psd.h5',
        # '--table', f'psd{aggregate_period_s}' if aggregate_period_s else 'psd',
        '--fs_float', f'{fs(probes[0], prefix)}',
        # (lambda x: x == x[0])(np.vectorize(fs)(probes, prefix))).all() else raise_ni()
        #
        # '--time_range_zeroing_list', '2019-08-26T04:00:00, 2019-08-26T05:00:00'
        # '--verbose', 'DEBUG',
        # '--chunksize', '20000',
        '--b_interact', '0',
        ]
    if 'w' in prefix:
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
            path_cruise / r'vsz+h5_proc\intervals_selected.txt',
            converters={'time_start': lambda x: np.datetime64(x, 'ns')}, index_col=0).index
        edges = (
            pd.DatetimeIndex(t_intervals_start), pd.DatetimeIndex(t_intervals_start + dt_custom_s))  # np.zeros_like()
    else:
        # Generate periodic intervals
        t_interval_start, t_intervals_end = intervals_from_period(datetime_range=np.array(
            [min_date[0], max_date[0]],
            # ['2018-08-11T18:00:00', '2018-09-06T00:00:00'],
            # ['2019-02-11T13:05:00', '2019-03-07T11:30:00'],
            # ['2018-11-16T15:19', '2018-12-14T14:35'],
            # ['2018-10-22T12:30', '2018-10-27T06:30:00'],
            'datetime64[s]'), period=period)
        edges = (pd.DatetimeIndex([t_interval_start]).append(t_intervals_end[:-1]),
                 pd.DatetimeIndex(t_intervals_end))

    for i, probe in enumerate(probes):
        probe_name = f'{prefix}{probe:02}'  # table name in db
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
