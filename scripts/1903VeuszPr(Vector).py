"""
veuszPropagate(): draw using Veusz pattern

Specify:
    - control execution parameters:

        :start: float, start step. Begins from step >= start
        :go: bool. Control of execution next steps. Insert ``go = True`` or ``go = False`` where need

    - data processing parameters:

        :path_cruise:
        :dir_probe: or :in_file:
        :path:
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
    'workData/BalticSea/190210/')

dir_probe = 'ADV_Nortek-Vector/'

start = 1
go = True
# ---------------------------------------------------------------------------------------------
if start <= 1 and go:  # False: #

    dt_custom_s = None  # 60 * 5  # allow not adjacent intervals with this length, s
    t_interval_start, t_intervals_end = intervals_from_period(datetime_range=np.array(
        ['2019-02-13T00:00:00', '2019-02-25T00:00:00'],
        # ['2018-11-16T15:19', '2018-12-14T14:35'],
        # ['2018-10-22T12:30', '2018-10-27T06:30:00'],
        'datetime64[s]'), period='1D')
    # t_intervals_end = pd.DatetimeIndex(['2018-11-19T00:00', '2018-11-30T00:00', '2018-12-05T00:00'])
    # t_interval_start = t_intervals_end[0]
    # t_intervals_end = t_intervals_end[1:]
    # '2018-12-09T03:10', '2018-12-06T21:30', '2018-12-11T11:40',
    # t_intervals_end = pd.DatetimeIndex(['2018-10-22T12:30', '2018-10-27T06:30:00'])

    probe_name = 'Vector'
    file_pattern = f'1D_{probe_name}_190212.vsz' * 70

    print('processing {} intervals...'.format(len(t_intervals_end)))
    for i_interval, (t_interval_start, t_interval_end) in enumerate(
            zip(pd.DatetimeIndex([t_interval_start]).append(t_intervals_end[:-1]), t_intervals_end), start=1):

        # TEMPORARY! Skip this number of intervals:
        if i_interval <= 0:
            continue

        if dt_custom_s:
            t_interval_end = t_interval_start + pd.Timedelta(dt_custom_s, 's')  # t_interval_start

        txt_time_range = \
            """
            "[['{:%Y-%m-%dT%H:%M}', '{:%Y-%m-%dT%H:%M}']]" \
            """.format(t_interval_start, t_interval_end)
        print(f'{i_interval}: {txt_time_range}', end=' ')

        veuszPropagate.main([
            Path(veuszPropagate.__file__).parent.with_name('veuszPropagate.ini'),
            # '--data_yield_prefix', '-',
            '--path', str(path_cruise / dir_probe / file_pattern),
            '--pattern_path', str(path_cruise / dir_probe / file_pattern),
            # warning: create file with small name
            # '--before_next', 'restore_config',
            '--add_to_filename', "_{:%y%m%d}".format(t_interval_start),  # _%H%M}_300s  , t_interval_start
            '--add_custom_list',
            'USEtime',  # nAveragePrefer',
            '--add_custom_expressions_list',
            txt_time_range,
            '--b_update_existed', 'True',
            '--export_pages_int_list', '1',  # 1, 2 '0'
            '--export_dpi_int_list', '300, 200',
            '--b_interact', '0',
            # '--b_images_only', 'True'
            ])
