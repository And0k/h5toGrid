from datetime import timedelta
from pathlib import Path
import logging
import json
import pandas as pd
import numpy as np
import to_vaex_hdf5.cfg_dataclasses
from yaml import safe_dump as yaml_safe_dump
# raw data db - data that was converted from csv
path_db_raw = Path(
    r'd:\workData\BalticSea\_Pregolya,Lagoon\211111-inclinometer\_raw\211111.raw.h5'
    # r'd:\workData\BalticSea\201202_BalticSpit_inclinometer\211008E15m@i11,36,37,38,w2\_raw\211008.raw.h5'
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


# Set hydra.searchpath to cruise specific config dir with incl_h5clc_hy.yaml in inclinometer/cfg/incl_h5clc_hy.yaml
# else it will be not used and hydra will only warn
with open(r'D:\Work\_Python3\And0K\h5toGrid\inclinometer\cfg\incl_h5clc_hy.yaml', 'w') as f:
    yaml_safe_dump({
        'defaults': ['base_incl_h5clc_hy', '_self_'],
        'hydra': {'searchpath': [f'file://{path_db_raw.parent}/cfg_proc'.replace('\\', '/')]}
        }, f)
"""
defaults:
  - base_incl_h5clc_hy
# - probes: wavegauges
# if installed
#  - override hydra/job_logging: colorlog
#  - override hydra/hydra_logging: colorlog
  - _self_  # just to prevent warning
hydra:
  searchpath:  # add path to cfg_proc dir
    - file://d:/workData/BalticSea/201202_BalticSpit_inclinometer/211008E15m@i11,36,37,38,w2/_raw/cfg_proc
#    - file://d:/workData/BalticSea/201202_BalticSpit_inclinometer/210618P7.5m@i9,14,15,19w1,2,4/_raw/cfg_proc
#    d:/workData/BalticSea/201202_BalticSpit_inclinometer/210827@P10m,P15m-i14,15,w1,4/_raw/cfg_proc
#    - file://d:/workData/BalticSea/201202_BalticSpit_inclinometer/210618@P7.5m-i15,w2(cal)/_raw/cfg_proc
#    - file://d:/workData/BalticSea/201202_BalticSpit_inclinometer/210922@E15m-i19,36,37,38,w2/_raw/cfg_proc/
#    - file://c:/temp/cfg_proc
#    - file://d:/workData/BalticSea/201202_BalticSpit_inclinometer/201202@i3,5,9,10,11,15,19,23,28,30,32,33,w1-6/_raw/cfg_proc
"""

from inclinometer.incl_h5clc_hy import *



db_out = None  # '"{}"'.format((path_db_raw.parent.parent / f'{path_db_raw.stem}_proc23,32;30.h5').replace('\\', '/'))

aggregate_period_s = {  # will be joined for multirun
    'inclinometers': [0, 2, 600, 7200],   # [0, 2, 600, 7200]  [0, 2, 300, 600, 1800, 7200]: [300, 1800] is for burst mode.  #[0, ]
    'wavegauges': [2, 300, 3600],      # [0, 2, 300, 3600]   #[0],
    }

# Change current dir. to raw data dir.: config dir will be relative to this dir. and hydra output dir. will be here
sys_argv_save = sys.argv
#sys.argv = ["c:/temp/cfg_proc"]  #[str(path_db_raw.parent / 'cfg_proc')]  # path of yaml config for hydra (main_call() uses sys.argv[0] to add it)
db_in = str(path_db_raw).replace('\\', '/')

for probes in 'inclinometers'.split():  # wavegauges    # 'inclinometers wavegauges' , 'inclinometers_tau600T1800'
    # only after common .proc_noAvg.h5 saved (with aggregate_period_s=0)
    for st_en in [
        '10:03-11:22',
        '13:03-14:03',
        '14:22-15:17',
        '16:10-17:13',
        '17:48-18:47']:
        st, en = st_en.split('-')
        out_db = str(path_db_raw.parent.with_name(
            '_'.join([
                path_db_raw.parent.parent.name.replace('-inclinometer', ''),
                st.replace(":", "")
                ])
            ).with_suffix(".proc.h5")).replace('\\', '/')
        to_vaex_hdf5.cfg_dataclasses.main_call([
            f'in.min_date=2021-11-11T{st}:00',
            f'in.max_date=2021-11-11T{en}:00',
            f'in.db_path="{db_in}"',
            f'out.db_path="{out_db}"',
            'out.b_del_temp_db=True',
            'program.verbose=INFO',
            'program.dask_scheduler=threads',
            f'+probes={probes}',  # see config probes directory
            f"out.aggregate_period={','.join(f'{a}s' for a in aggregate_period_s[probes] if a !=0)}",
            ] + (
                ["in.tables=['i.*']"] if probes == 'inclinometers' else  # ['incl(23|30|32).*']  # ['i.*']
                ["in.tables=['w.*']"]                                    # ['w0[2-6].*']         # ['w.*']
                ) +
            ['--multirun'],
            fun=main)
    continue


    df = to_vaex_hdf5.cfg_dataclasses.main_call([
        f'in.db_path="{db_in}"',
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
            ["in.tables=['i.*']"] if probes == 'inclinometers' else  # ['incl(23|30|32).*']  # ['i.*']
            ["in.tables=['w.*']"]                                    # ['w0[2-6].*']         # ['w.*']
            ) +
        ['--multirun'],
        fun=main)

sys.argv = sys_argv_save