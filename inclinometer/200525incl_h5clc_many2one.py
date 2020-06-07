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
import inclinometer.incl_h5clc as incl_h5clc
import inclinometer.incl_h5spectrum as incl_h5spectrum
import veuszPropagate
from utils2init import path_on_drive_d, init_logging, open_csv_or_archive_of_them, st

# l = logging.getLogger(__name__)
l = init_logging(logging, None, None, 'INFO')

import dask
dask.config.set(scheduler='synchronous')
from dask.cache import Cache
cache = Cache(2e9)  # Leverage two gigabytes of memory
cache.register()    # Turn cache on globally

# Directory where inclinometer data will be stored

# using data from 190716И1#i14+190721i05+190818i04_proc.vsz in path:
path_ref = Path(r'd:\workData\BalticSea\190713_ABP45\inclinometer\proc')
paths_cruise = [
    '../proc/190716incl_proc.h5',
    '../proc/190721incl_proc.h5',
    '../../../190817_ANS42/inclinometer/190818incl_proc/190818incl_proc.h5'
    ]
timeranges = [
    [['2019-07-16T17:00:00', '2019-07-19T16:30:00']],
    [['2019-07-21T20:00:00', '2019-08-18T01:45:00']],
    [['2019-08-18T06:30:00', '2019-08-26T15:50:00']],
    ]


#Path(p).relative_to(path_ref)
paths_db_in = [(path_ref / p).resolve().parent.with_name(Path(p).name.replace('_proc','')) for p in paths_cruise]  #path_on_drive_d() f'{db_path.stem}_proc_noAvg.h5'

probes = [14, 5, 4]

prefix = 'incl'  # 'incl' or 'w'  # table name prefix in db and in raw files (to find raw fales name case will be UPPER anyway): 'incl' - inclinometer, 'w' - wavegauge

# dir_incl = '' if 'inclinometer' in str(path_cruise) else 'inclinometer'
# if not db_name:  # then name by cruise dir:
#     db_name = re.match('(^[\d_]*).*', (path_cruise.parent if dir_incl else path_cruise).name
#                        ).group(1).strip('_') + 'incl.h5'
db_path_out = paths_db_in[0].with_name('many2one_out.h5')





# Run steps (inclusive):
st.start = 2  # 1
st.end = 2
m_TimeStart_csv = pd.Timestamp('2019-07-08T00:00:00Z')

# Calculate velocity and average
if st(2):
    # if aggregate_period_s is None then not average and write to *_proc_noAvg.h5 else loading from that h5 and writing to _proc.h5
    for aggregate_period_s in [720]:   # [None, 2, 300, 600, 3600 if 'w' in prefix else 7200]

        args = [Path(incl_h5clc.__file__).with_name(f'incl_h5clc_many2one.yaml'),
                # if no such file all settings are here
                '--db_path', '|'.join(str(p) for p in paths_db_in),
                '--tables_list', ','.join(f'incl{i:0>2}' for i in probes),  #incl.*| !  'incl.*|w\d*'  inclinometers or wavegauges w\d\d # 'incl09',
                '--aggregate_period', f'{aggregate_period_s}S' if aggregate_period_s else '',
                # '--date_min', datetime64_str(date_min[0]),  # '2019-08-18T06:00:00',
                # '--date_max', datetime64_str(date_max[0]),  # '2019-09-09T16:31:00',  #17:00:00
                '--output_files.db_path', str(db_path_out),
                '--table', f'V_incl_bin{aggregate_period_s}' if aggregate_period_s else 'V_incl',
                '--verbose', 'DEBUG',
                # '--calc_version', 'polynom(force)',  # depreshiated
                # '--chunksize', '20000',
                # '--not_joined_h5_path', f'{db_path.stem}_proc.h5',
                # '--csv_date_format', '%g'
                ]
        if aggregate_period_s is None:  # proc. parameters (if we have saved proc. data then when aggregating we are not processing)

            args += (
                ['--max_dict', 'M[xyz]:4096',
                 # Note: for Baranov's prog 4096 is not suited!
                 # '--timerange_zeroing_dict', "incl19: '2019-11-10T13:00:00', '2019-11-10T14:00:00'\n,"  # not works - use kwarg
                 # '--timerange_zeroing_list', '2019-08-26T04:00:00, 2019-08-26T05:00:00'
                ] if prefix == 'incl' else
                ['--bad_p_at_bursts_starts_peroiod', '1H',
                ])
            kwarg = {}  # 'in': {'timerange_zeroing': {'incl19': ['2019-11-14T06:30:00', '2019-11-14T06:50:00']}}
        else:
            kwarg = {'in': {
                'dates_min': [timeranges[iprobe][0][0] for iprobe in range(len(probes))],  # '2019-08-18T06:00:00',
                'dates_max': [timeranges[iprobe][0][1] for iprobe in range(len(probes))],  # '2019-09-09T16:31:00',  #17:00:00
                }, 'output_files': {
                'b_all_to_one_col': True,
                'csv_date_format': lambda t: (t - m_TimeStart_csv) / np.timedelta64(1, 'h'),
                'csv_columns': ['Date', 'Ve', 'Vn']
                }}
        # csv splitted by 1day (default for no avg) and monolit csv if aggregate_period_s==600
        if aggregate_period_s is not None:
            args += ['--csv_path', str(db_path_out.parent / 'csv')]

        incl_h5clc.main(args, **kwarg)