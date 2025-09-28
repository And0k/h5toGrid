"""
Loading two trackers data in one DB, calc distance between, recalc all

Note: To convert in commandline args replace { and } with %, put " around lists
"""
# import pytest
from datetime import timedelta
from pathlib import Path
import logging
import json
import pandas as pd
import numpy as np
from gps_tracker.autofon_coord import *  # autofon_df_from_dict
from cfg_dataclasses import main_call

# %%
# def call_example_sp5ref6__230825():

# sys_argv_save = sys.argv.copy()
# if __name__ != '__main__':
#     sys.argv = [__file__]

path_device = Path(
    r'd:\WorkData\_experiment\tracker\240315_Devau'.replace('\\', '/')
    # r'd:\WorkData\BalticSea\230825_Kulikovo@ADCP,ADV,i,tr\tracker_SPOT'.replace('\\', '/')
)


if False:
    # Old configs
    cfg = {
        'dir_device': str(path_device),
        'DEVICE': 'sp4',
        'ANCHOR_DEVICE_NUM': 1,
        'ANCHOR_DEVICE_TYPE': 'sp'
    }

    cfg = {
        'dir_device': str(path_device),
        'file_raw_local': str(
            path_device / 'raw' / 'ActivityReport.xlsx'
        ).replace('\\', '/'),  #.replace(':', r'\:'),
        'file_raw_local1': str(
            path_device / 'raw' / 'ActivityReport1.xlsx'
        ).replace('\\', '/'),
        'DEVICE': 'sp4',
        'ANCHOR_DEVICE_NUM': 1,
        'ANCHOR_DEVICE_TYPE': 'sp'
    }    
    args = [  # cycle removes (possible) remaining line breaks
        a.strip() for a in """
        input.time_interval=[2023-08-25T13:10, now] ^
        input.dt_from_utc_hours=2 ^
        +input.alias={{{ANCHOR_DEVICE_TYPE}{ANCHOR_DEVICE_NUM}:LECO}} ^
        +input.path_raw_local={{{DEVICE}:"{file_raw_local}", {ANCHOR_DEVICE_TYPE}{ANCHOR_DEVICE_NUM}:"{file_raw_local1}"}} ^
        process.anchor_coord_default=[54.9896, 20.299717] ^
        process.anchor_tracker=[{ANCHOR_DEVICE_TYPE}{ANCHOR_DEVICE_NUM}] ^
        process.anchor_depth=20 ^
        +process.max_dr={{{DEVICE}:200, {DEVICE}_ref_{ANCHOR_DEVICE_TYPE}{ANCHOR_DEVICE_NUM}:100}} ^
        out.db_path='{dir_device}/{file_stem}.h5' ^
        out.tables=["{ANCHOR_DEVICE_TYPE}{ANCHOR_DEVICE_NUM}","{DEVICE}"] ^
        process.period_tracks=1D""".format_map(cfg).split(' ^\n')
    ]



cfg = {
    'dir_device': str(path_device),
    'file_raw_local': str(
        path_device / 'raw' / 'ActivityReport.xlsx'
    ).replace('\\', '/'),
    'date_prefix': path_device.stem.split('_')[0],
    'date_start': '2024-03-15T11:00',  # UTC
    'DEVICE0': 'sp0',
    'DEVICE1': 'sp1',
    'DEVICE2': 'sp2',
    'DEVICE3': 'tr2',    
    'ANCHOR_DEVICE_NUM': 1,    
    'ANCHOR_DEVICE_TYPE': 'tr'    
}
# dependant fields
cfg.update({
    # 'TYPE@DEVICE': 'current@{DEVICE}ref{ANCHOR_DEVICE_NUM}'.format_map(cfg),
    'file_stem': '{date_prefix}@sp0-2,tr1-2'.format_map(cfg)
})

args = [  # cycle removes (possible) remaining line breaks
    a.strip() for a in """
input.time_interval=[{date_start}, now] ^
+input.dt_from_utc_hours={{sp.*:2}} ^
+input.alias={{sp0:LECO, sp1:LOO1, sp2:LOO3}} ^
+input.path_raw_local={{{DEVICE3}:None, {ANCHOR_DEVICE_TYPE}{ANCHOR_DEVICE_NUM}:None}} ^
process.anchor_tracker=[{ANCHOR_DEVICE_TYPE}{ANCHOR_DEVICE_NUM}] ^
process.anchor_depth=-0.5 ^
out.db_path='{dir_device}/{file_stem}.h5' ^
out.tables=["{ANCHOR_DEVICE_TYPE}{ANCHOR_DEVICE_NUM}","{DEVICE0}","{DEVICE1}","{DEVICE2}","{DEVICE3}"] ^
process.period_tracks=1D ^
process.max_dr_default=9999999 ^
out.dt_bins=[] ^
out.dt_rollings=[]

""".format_map(cfg).split(' ^\n')
]
# 'out.tables=[{}]'.format(','.join([f'"{d}"' for d in device])),
# +input.path_raw_local={{{ANCHOR_DEVICE_TYPE}{ANCHOR_DEVICE_NUM}:"{file_raw_local}"}} ^

main_call(args, fun=main)
# sys.argv = sys_argv_save
# %%
