from datetime import timedelta
from pathlib import Path
import logging
import json
import pandas as pd
import numpy as np
import to_vaex_hdf5.cfg_dataclasses

from inclinometer.incl_h5clc_hy import *

# raw data db - data that was converted from csv
path_db = Path(
    r'd:\workData\BalticSea\201202_BalticSpit_inclinometer\210726@i4,5,11,36,37,38,w2,5,6\_raw\210726incl.h5'
    ).absolute()

db_in = str(path_db).replace('\\', '/')
db_out = str(path_db.parent.parent / f'{path_db.stem}_proc.h5').replace('\\', '/')

aggregate_period_s = {  # will be joined for multirun
    'inclinometers': [0, 2, 600, 7200],  #[0, ]
    'wavegages': [0, 2, 300, 3600]
    }

# Change current dir. to raw data dir.: config dir will be relative to this dir. and hydra output dir. will be here
sys_argv_save = sys.argv
#sys.argv = ["c:/temp/cfg_proc"]  #[str(path_db.parent / 'cfg_proc')]  # path of yaml config for hydra (main_call() uses sys.argv[0] to add it)

for probes in ('inclinometers', 'wavegages'):  #
    df = to_vaex_hdf5.cfg_dataclasses.main_call([
        f'input.db_path="{db_in}"',
        # '++filter.time_bad_intervals=[2021-06-02T13:49, now]', # todo
        # 'input.tables=["incl.*"]', # was set in config probes directory
        f'out.db_path="{db_out}"',
        # f'out.table=V_incl_bin{aggregate_period_s}s',
        'out.b_del_temp_db=True',
        # f'out.text_path=text_output',
        'program.verbose=INFO',
        'program.dask_scheduler=synchronous',
        f'+probes={probes}',  # see config probes directory
        f"out.aggregate_period={','.join(f'{a}s' for a in aggregate_period_s[probes])}",
        '--multirun',
        # '--config-path=cfg_proc',  # Primary config module 'inclinometer.cfg_proc' not found.
        # '--config-dir=cfg_proc'  # additional cfg dir
        ], fun=main)

sys.argv = sys_argv_save