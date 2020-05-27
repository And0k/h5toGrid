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
        :db_path:
        :probes:
        ...

"""
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# import my scripts
drive_d = Path('D:/' if sys.platform == 'win32' else '/mnt/D')  # allows to run on both my Linux and Windows systems:
scripts_path = drive_d.joinpath('Work/_Python3/And0K/h5toGrid/scripts')
sys.path.append(str(Path(scripts_path).parent.resolve()))  # os.getcwd()
from to_pandas_hdf5.csv2h5 import main as csv2h5
from to_pandas_hdf5.csv_specific_proc import correct_kondrashov_txt
from inclinometer.h5inclinometer_coef import h5copy_coef
import veuszPropagate
from utils_time import intervals_from_period
from utils2init import path_on_drive_d

path_cruise = path_on_drive_d(
    r'd:\WorkData\_experiment\_2019\inclinometer\200117_tank[23,30,32]'
    )
r"""
d:\workData\BalticSea\191119_Filino
d:\WorkData\_experiment\_2019\inclinometer\190710_compas_calibr-byMe
d:\WorkData\_experiment\_2019\inclinometer\190710_compas_calibr-byMe\190902[3,14,15,16,19]
d:\WorkData\_experiment\_2019\inclinometer\190711_tank[1,4,5,7,11,12]
d:\WorkData\_experiment\_2019\inclinometer\190710_compas_calibr-byMe

d:\WorkData\_experiment\_2019\inclinometer\190710_compas_calibr-byMe\191009incl15_directions_test
d:\WorkData\BalticSea\190806_Yantarniy\inclinometer
d:\WorkData\_experiment\_2019\inclinometer\190704_tank_ex2[12,22,27,28,30,31,35]
d:\WorkData\_experiment\_2019\inclinometer\190704_tank_ex1[21,23,24,25,26,29,32,34]
d:\WorkData\_experiment\_2019\inclinometer\190710_compas_calibr-byMe
d:\WorkData\_experiment\_2019\inclinometer\190704


t_start_utc = {
    lambda: '2019-10-09T12:00:00',  #'2019-07-03T15:15:00', # '2019-07-03T12:00:00', '19-07-09T17:48:00'
    {
        12: '2019-07-03T15:00:00',
        25: '2019-07-03T14:05:00',
        26: '2019-07-03T14:05:00',
    }
                          )
(

        30: '2019-07-09T17:48',
        34: '2019-07-10T15:08',
        31: '2019-07-10T16:55',
        23: '2019-07-10T17:24',
        21: '2019-07-10T18:10',
        32: '2019-07-10T18:38',
        28: '2019-07-10T19:13',
        29: '2019-07-10T19:14',
        26: '2019-07-11T13:32',
       925: '2019-07-11T19:31',
        25: '2019-07-12T12:00',
        33: '2019-07-12T12:28',

         3: '2019-09-03T19:34:00',
        16: '2019-09-04T19:00',
        19: '2019-09-03T18:50',

        13: '2019-10-15T16:37',
    }
    

date_min_NOT_USED = {   # not output data < date_min, UTC
        30: '2019-07-09T17:54',
        34: '2019-07-10T15:10',
        31: '2019-07-10T16:59',
        23: '2019-07-10T17:28',
        21: '2019-07-10T18:17',
        32: '2019-07-10T18:43',
        28: '2019-07-10T19:34',
        29: '2019-07-10T19:18',
        26: '2019-07-11T13:38',
       925: '2019-07-11T19:41',
        25: '2019-07-12T12:01',
        33: '2019-07-12T12:29',

         7: '2019-07-11T16:52',
         4: '2019-07-11T17:21',
        11: '2019-07-11T17:41',
        12: '2019-07-11T18:04',
         5: '2019-07-11T18:26',
         1: '2019-07-11T18:48',

        13: '2019-10-15T16:40',

    }    

date_max = defaultdict(lambda: None, {
    7:  '2019-07-11T17:21',
    4:  '2019-07-11T17:41',
    11: '2019-07-11T18:04',
    12: '2019-07-11T18:26',
    5:  '2019-07-11T18:48',
    16: '2019-09-03T19:34',
    19: '2019-09-03T19:00',
    })

date_min = defaultdict(lambda x: '2019-11-06T10:50', 0)
date_max = defaultdict(lambda x: '2019-11-06T12:20', 0)
"""
# dafault and specific to probe limits
date_min = defaultdict(lambda: None, {9: '2019-12-23T17:00'})  # '2019-11-06T12:35''2019-11-06T13:34'
date_max = defaultdict(lambda: None, {9: '2019-12-23T17:35'})

date_min = defaultdict(
    lambda: '2020-01-17T13:00',
    # '2019-11-19T12:30',  # '2019-11-08T12:20', 2019-11-08T12:00 '2019-11-06T12:35', # '2019-11-06T10:50' '2019-07-16T17:00:00' 2019-06-20T14:30:00  '2019-07-21T20:00:00', #None,
    {  # 0: #'2019-08-07T16:00:00' '2019-08-17T18:00', 0: '2018-04-18T07:15',
        # 1: '2018-05-10T15:00',
        })
date_max = defaultdict(
    lambda: '2020-01-17T15:00',
    # '2019-11-06T13:34','2019-11-06T12:20' '2019-08-31T16:38:00' '2019-07-19T17:00:00', # '2019-08-18T01:45:00', # None,
    {  # 0:  # '2019-09-09T17:00:00' '2019-09-06T04:00:00' '2019-08-27T02:00', 0: '2018-05-07T11:55',
        # 1: '2018-05-30T10:30',
        })

t_start_utc = defaultdict(lambda: None, {9: '2019-11-19T13:28:53',  # '2019-12-20T16:16',
                                         # 10: '2000-01-01T04:32',
                                         11: '2019-11-19T14:00:50 ',
                                         })


def dt_from_utc_2000(probe):
    """ Correct time of probes started without time setting"""
    return int((np.datetime64('2000', 'Y') - np.datetime64(t_start_utc[probe]
                                                           )) / np.timedelta64(1, 's')) if t_start_utc.get(probe) else 0


def f_next_date_min(key):
    """one by one work"""
    d_search = {**t_start_utc, **date_min}
    try:
        val = d_search[key]
    except KeyError:  # key is not in date_min/t_start_utc. Nothing to compare
        return None

    for k, v in sorted(d_search.items(), key=lambda x: x[1]):
        if v <= val:
            continue
        return v


probes = [23, 30,
          32]  # , 3, 13range(1, 20)  # [9, 11]  # range(1, 20) #[3,14,15,16,19]  #[4, 11, 5, 12]  # 29, 30 range(12, 13)  # [1, 4, 7, 11, ] 5, 12  # sorted(t_start_utc.keys())  # [1, 4, 5, 7, 11, 12, 14]  #[21, 23]  #range(1, 40)  # [25,26]  #, [17, 18]
# '190727incl.h5'
db_path = r'd:\workData\BalticSea\191119_Filino\inclinometer\191119incl.h5'  # r'd:\WorkData\_experiment\_2019\inclinometer\190710_compas_calibr-byMe\190710incl.h5'  # path_cruise / (re.match('(^[\d_]*).*', path_cruise.name).groups()[0].strip('_') + 'incl.h5')
# db_path = Path(r'd:\workData\BalticSea\190713_ABP45\inclinometer\190721incl.h5')  #!
dir_incl = '' if 'inclinometer' in str(path_cruise) else 'inclinometer/'
start = 1


def fs(probe):
    # if probe < 20 or probe in [30]:  #[4, 11, 5, 12] + [1, 7, 13, 30]
    #     return 5
    # if probe in [21, 25, 26] + list(range(28, 35)):
    #     return 8.2
    return 5


# ---------------------------------------------------------------------------------------------
if start == 1:
    i_proc = 0  # counter of processed probes
    for probe in probes:
        in_file = path_cruise / (dir_incl + '_raw') \
                  / f'INKL_{probe:03d}.TXT'  # '_source/incl_txt/180510_INKL10.txt' # r'_source/180418_INKL09.txt'
        in_file = correct_kondrashov_txt(in_file)
        if not in_file:
            continue

        csv2h5([scripts_path / 'ini/csv_inclin_Kondrashov.ini',
                '--path', str(in_file),
                '--blocksize_int', '10000000',  # 10Mbt
                '--table', re.sub('^inkl_0', 'incl',
                                  re.sub('^[\d_]*|\*', '', in_file.stem).lower()),
                '--date_min', np.datetime_as_string(np.datetime64(date_min[probe], 's')),  # + np.timedelta64(0, 'm')
                '--date_max', np.datetime_as_string(np.datetime64(date_max[probe], 's')),
                # f_next_date_min(probe) '2019-07-04T21:00:00',
                '--db_path', str(db_path),
                '--log', str(scripts_path / 'log/csv2h5_inclin_Kondrashov.log'),
                # '--b_raise_on_err', '0',  # ?! need fo this file only?
                '--b_interact', '0',
                '--fs_float', f'{fs(probe)}',
                '--dt_from_utc_seconds', f'{dt_from_utc_2000(probe)}',
                '--csv_specific_param_dict', 'invert_magnitometr: True'
                ])

        for db_coefs in [
            r'd:\WorkData\_experiment\_2019\inclinometer\190704_tank_ex2[12,22,27,28,30,31,35]\190704incl_not_-M.h5',
            r'd:\WorkData\_experiment\_2018\inclinometer\181004_tank[1-20]\181004_KTIz.h5'
            # # old DB with inverted M like new (coef not copied?)

            ]:
            r"""
              d:\WorkData\_experiment\_2018\inclinometer\181004_tank[1-20]\181004_KTI.h5
            d:\WorkData\_experiment\_2019\inclinometer\190704_tank_ex1[21,23,24,25,26,29,32,34]\190704incl_not_-M.h5
            d:\WorkData\_experiment\_2019\inclinometer\190704_tank_ex1[21,23,24,25,26,29,32,34]\190704incl.h5
            d:\WorkData\_experiment\_2019\inclinometer\190704_tank_ex2[12,22,27,28,30,31,35]\190704incl.h5
            """
            db_path_coefs = Path(db_coefs)
            try:
                tbl = f'incl{probe:0>2}'
                print(f"Adding coefficients to {db_path}/{tbl} from {db_path_coefs}")
                h5copy_coef(db_path_coefs, db_path, tbl)
                break
            except KeyError as e:  # Unable to open object (component not found)
                pass
        else:
            print('Coef is not copied!')
            # todo write some dummy coefficients to can load Veusz patterns
        i_proc += 1
    print(f'Ok! ({i_proc} probes processed)')

# Draw in Veusz
if start == 2:
    # Not adjacent intervals with this length, s (set None to not allow)
    dt_custom_s = None  # 60 * 5  # None
    # Periodic intervals
    t_interval_start, t_intervals_end = intervals_from_period(datetime_range=np.array(
        ['2019-08-11T18:00:00', '2019-09-06T00:00:00'],
        # ['2019-07-28T00:00:00', '2019-09-11T00:00:00'],  #08-14 07-28T00:00:00
        'datetime64[s]'), period='1D')
    # t_interval_start = t_intervals_end[0] - np.timedelta64('5', 'm')

    for i, probe in enumerate(probes):
        probe_name = f'incl{probe:02}'
        print('processing {} intervals...'.format(len(t_intervals_end)))
        # for i_interval, (t_interval_start, t_interval_end) in enumerate(zip(pd.DatetimeIndex([t_interval_start]).append(t_intervals_end[:-1]), t_intervals_end), start=1):
        cfg_vp = {'veusze': None}
        for i_interval, (t_interval_end_prev, t_interval_end) in enumerate(zip(
                pd.DatetimeIndex([t_interval_start]).append(t_intervals_end[:-1]), pd.DatetimeIndex(t_intervals_end)),
                start=1):  # list()[::-1]

            # if i_interval <= 2: #<= 0:  # TEMPORARY! Skip this number of intervals
            #     continue

            if dt_custom_s:
                t_interval_start = t_interval_end - pd.Timedelta(dt_custom_s, 's')  # t_interval_start
            else:
                t_interval_start = t_interval_end_prev
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
                 '--pattern_path', fr'd:\WorkData\BalticSea\190806_Yantarniy\inclinometer\{probe_name}_190727_1D.vsz',
                 # str(db_path.parent.joinpath(dir_incl + f'{probe_name}_190211.vsz')), #warning: create file with small name
                 # '--before_next', 'restore_config',
                 '--add_to_filename', "_{:%y%m%d_%H%M}_1D".format(t_interval_start),
                 '--add_custom_list', 'USEtime',  # nAveragePrefer',
                 '--add_custom_expressions_list',
                 txt_time_range,
                 # + """
                 # ", 5"
                 # """,
                 '--b_update_existed', 'True',
                 '--export_pages_int_list', '6, 7, 8',  # '0',  #'1, 2, 3'
                 # '--export_dpi_int', '200',
                 '--b_interact', '0',
                 # '--b_images_only', 'True'
                 '--return', '<embedded_object>',  # reuse to not bloat memory
                 ],
                veusze=cfg_vp['veusze'])
