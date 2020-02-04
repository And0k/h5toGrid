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
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# import my scripts
drive_d = Path('D:/' if sys.platform == 'win32' else '/mnt/D')  # allows to run on both my Linux and Windows systems:
scripts_path = drive_d.joinpath('Work/_Python3/And0K/h5toGrid/scripts')
sys.path.append(str(Path(scripts_path).parent.resolve()))  # os.getcwd()
import veuszPropagate
from utils_time import intervals_from_period

path_cruise = drive_d.joinpath(
    # r'd:\WorkData\_experiment\_2019\inclinometer\190711_tank'
    r'd:\WorkData\_experiment\_2019\inclinometer\190710_compas_calibr-byMe'
    )
"""
r'd:\WorkData\_experiment\_2019\inclinometer\190704'
"""

# path_db = path_cruise / (re.match('(^[\d_]*).*', path_cruise.name).groups()[0].strip('_') + 'incl.h5')
path_db = Path(r'd:\workData\BalticSea\190713_ABP45\inclinometer\190721incl.h5')
dir_incl = '' if 'inclinometer' in path_db.parent.parts else 'inclinometer/'
probes = [5,
          12]  # [1, 4, 5, 7, 11, 12] #sorted(dt_from_utc_for_probe.keys())  # [1, 4, 5, 7, 11, 12, 14] #[21, 23] #range(1, 40)  # [25,26]  #, [17, 18]
start = 2
go = True

# Draw in Veusz
if start <= 2 and go:  # False: #

    dt_custom_s = None  # 60 * 5  #None  #  allow not adjacent intervals with this length, s
    t_interval_start, t_intervals_end = intervals_from_period(datetime_range=np.array(
        ['2019-07-21T20:00:00', '2019-08-18T01:45:00'],
        'datetime64[s]'), period='1D')
    # t_interval_start = t_intervals_end[0] - np.timedelta64('5', 'm')

    for i, probe in enumerate(probes):
        probe_name = f'incl{probe:02}'
        print('processing {} intervals...'.format(len(t_intervals_end)))
        # for i_interval, (t_interval_start, t_interval_end) in enumerate(zip(pd.DatetimeIndex([t_interval_start]).append(t_intervals_end[:-1]), t_intervals_end), start=1):
        for i_interval, (t_interval_end_prev, t_interval_end) in list(enumerate(zip(
                pd.DatetimeIndex([t_interval_start]).append(t_intervals_end[:-1]), pd.DatetimeIndex(t_intervals_end)),
                start=1))[::-1]:

            # if i_interval < 23: #<= 0:  # TEMPORARY! Skip this number of intervals
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
            veuszPropagate.main([
                Path(veuszPropagate.__file__).parent.with_name('veuszPropagate.ini'),
                # '--data_yield_prefix', '-',
                '--path', str(path_db),  # use for custom loading from db and some source is required
                '--tables_list', f'/{probe_name}',  # 181022inclinometers/ \d*
                '--pattern_path', fr'd:\workData\BalticSea\190713_ABP45\inclinometer\190721_1D_{probe_name}.vsz',
                # str(path_db.parent.joinpath(dir_incl + f'{probe_name}_190211.vsz')), #warning: create file with small name
                '--before_next', 'restore_config',

                '--add_to_filename', "_{:%y%m%d_%H%M}_1D".format(t_interval_start),
                '--add_custom_list',
                'USEtime',  # nAveragePrefer',
                '--add_custom_expressions_list',
                txt_time_range,
                # + """
                # ", 5"
                # """,
                '--b_update_existed', 'True',
                '--export_pages_int_list', '0',  # '1, 2, 3'
                # '--export_dpi_int', '200',
                '--b_interact', '0',
                # '--b_images_only', 'True'
                ])
