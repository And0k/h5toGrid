from datetime import timedelta
from pathlib import Path
import logging
import json
import pandas as pd
import numpy as np
import to_vaex_hdf5.cfg_dataclasses

# Set hydra.searchpath to cruise specific config dir with incl_h5clc_hy.yaml in inclinometer/cfg/incl_h5clc_hy.yaml
# else it will be not used and hydra will only warn!

from inclinometer.incl_h5clc_hy import *

# raw data db - data that was converted from csv
path_db_raw = Path(
    r'd:\workData\BalticSea\201202_BalticSpit_inclinometer\211008E15m@i11,36,37,38,w2\_raw\211008.raw.h5'
    # r'd:\workData\BalticSea\201202_BalticSpit_inclinometer\210618P7.5m@i9,14,15,19w1,2,4\_raw\210726.raw.h5'
    # r'd:\workData\BalticSea\201202_BalticSpit_inclinometer\210618P7.5m@i9,14,15,19w1,2,4\210618@i14,15+19+09,w2+1,4.proc_noAvg.h5'
    # r'd:\workData\BalticSea\201202_BalticSpit_inclinometer\210618P7.5m@i9,14,15,19w1,2,4\210618@i14,15+19,w2+1,4.proc_noAvg.h5'
    # r'd:\workData\BalticSea\201202_BalticSpit_inclinometer\210726P10,15,E14.5m@i4,5,9,11,36,37,38,w1,2,5,6\210726P10,15m@i5+14,9+15,w5+1,2.proc_noAvg.h5'
    # r'd:\workData\BalticSea\201202_BalticSpit_inclinometer\210726@P7.5m,P15m-i9,14,19w1,4\_raw\210726.raw.h5'
    # r'd:\workData\BalticSea\201202_BalticSpit_inclinometer\210827@P10m,P15m-i14,15,w1,4\_raw\210827.raw.h5'
    # r'd:\workData\BalticSea\201202_BalticSpit_inclinometer\210618@P7.5m-i15,w2(cal)\_raw\210618.raw.h5'
    # r'd:\workData\BalticSea\201202_BalticSpit_inclinometer\210922@E15m-i19,36,37,38,w2\_raw\210922.raw.h5'
    # r'd:\workData\BalticSea\201202_BalticSpit_inclinometer\201202@i3,5,9,10,11,15,19,23,28,30,32,33,w1-6\_raw\201202.raw.h5'
    # r'd:\workData\BalticSea\201202_BalticSpit_inclinometer\210726@i4,5,11,36,37,38,w2,5,6\_raw\210726.raw.h5'
    ).absolute()

db_in = '"{}"'.format(path_db_raw).replace('\\', '/')
db_out = None  # '"{}"'.format((path_db_raw.parent.parent / f'{path_db_raw.stem}_proc23,32;30.h5').replace('\\', '/'))

aggregate_period_s = {  # will be joined for multirun
    'inclinometers': [0, 2, 600, 7200],   # [0, 2, 600, 7200]  [0, 2, 300, 600, 1800, 7200]: [300, 1800] is for burst mode.  #[0, ]
    'wavegauges': [2, 300, 3600],      # [0, 2, 300, 3600]   #[0],
    }

# Change current dir. to raw data dir.: config dir will be relative to this dir. and hydra output dir. will be here
sys_argv_save = sys.argv
#sys.argv = ["c:/temp/cfg_proc"]  #[str(path_db_raw.parent / 'cfg_proc')]  # path of yaml config for hydra (main_call() uses sys.argv[0] to add it)

for probes in 'wavegauges'.split():  # 'inclinometers    # inclinometers wavegauges' , 'inclinometers_tau600T1800'
    df = to_vaex_hdf5.cfg_dataclasses.main_call([
        f'input.db_path={db_in}',
        # '++filter.time_bad_intervals=[2021-06-02T13:49, now]', # todo
        # 'input.tables=["incl.*"]', # was set in config probes directory
        f'out.db_path={db_out}',
        # f'out.table=V_incl_bin{aggregate_period_s}s',
        'out.b_del_temp_db=True',
        # f'out.text_path=text_output',
        'program.verbose=INFO',
        'program.dask_scheduler=threads',
        f'+probes={probes}',  # see config probes directory
        f"out.aggregate_period={','.join(f'{a}s' for a in aggregate_period_s[probes])}",
        # '--config-path=cfg_proc',  # Primary config module 'inclinometer.cfg_proc' not found.
        # '--config-dir=cfg_proc'  # additional cfg dir
        ] + (
            ["input.tables=['i.*']"] if probes == 'inclinometers' else  # ['incl(23|30|32).*']  # ['i.*']
            ["input.tables=['w.*']"]                                    # ['w0[2-6].*']         # ['w.*']
            ) +
        ['--multirun'],
        fun=main)

sys.argv = sys_argv_save