"""
Loading two trackers data in one DB, calc distance between, recalc all
To convert in commandline args replace { and } with %, put " around lists
"""
# import pytest
from datetime import timedelta
from pathlib import Path
import logging
import json
import pandas as pd
import numpy as np
from to_vaex_hdf5.autofon_coord import *  # autofon_df_from_dict
from cfg_dataclasses import main_call

# %%
# def call_example_sp5ref6__230825():

# sys_argv_save = sys.argv.copy()
# if __name__ != '__main__':
#     sys.argv = [__file__]

path_db = Path(
    r'd:\WorkData\BalticSea\230825_Kulikovo@ADCP,ADV,i,tr\tracker_SPOT'.replace('\\', '/')
)


if False:
    cfg = {
        'dir_device': str(path_db),
        'DEVICE': 'sp5',
        'ANCHOR_DEVICE_NUM': 6,
        'ANCHOR_DEVICE_TYPE': 'sp'
    }
else:
    cfg = {
        'dir_device': str(path_db),
        'file_raw_local': str(
            path_db / 'raw' / 'ActivityReport.xlsx'
        ).replace('\\', '/'),  #.replace(':', r'\:'),
        'DEVICE': 'sp4',
        'ANCHOR_DEVICE_NUM': 1,
        'ANCHOR_DEVICE_TYPE': 'sp'
    }


cfg.update({
    'TYPE@DEVICE': 'current@{DEVICE}ref{ANCHOR_DEVICE_NUM}'.format_map(cfg),
    'file_stem': '230825@{DEVICE}ref{ANCHOR_DEVICE_NUM}'.format_map(cfg)
})

args = [  # cycle removes (possible) remaining line breaks
    a.strip() for a in """
input.time_interval=[2023-08-25T11:20, now] ^
input.dt_from_utc_hours=2 ^
+input.alias={{{ANCHOR_DEVICE_TYPE}{ANCHOR_DEVICE_NUM}:LECO}} ^
process.anchor_coord_default=[54.989683, 20.301067] ^
process.anchor_tracker=[{ANCHOR_DEVICE_TYPE}{ANCHOR_DEVICE_NUM}] ^
process.anchor_depth=20 ^
+process.max_dr={{{DEVICE}:200, {DEVICE}_ref_{ANCHOR_DEVICE_TYPE}{ANCHOR_DEVICE_NUM}:100}} ^
out.db_path='{dir_device}/{file_stem}.h5' ^
out.tables=["{ANCHOR_DEVICE_TYPE}{ANCHOR_DEVICE_NUM}","{DEVICE}"] ^
process.period_tracks=1D""".format_map(cfg).split(' ^\n')
]
#
# +input.path_raw_local={{{ANCHOR_DEVICE_TYPE}{ANCHOR_DEVICE_NUM}:"{file_raw_local}"}} ^

main_call(args, fun=main)
# sys.argv = sys_argv_save
# %%
