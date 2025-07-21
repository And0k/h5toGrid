import logging
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from functools import partial
import itertools
from pathlib import Path
import glob
from typing import Optional

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

# import my scripts
from to_pandas_hdf5.csv2h5 import main as csv2h5
from to_pandas_hdf5.csv_specific_proc import mod_incl_name, rep_in_file, correct_txt, loaded_tcm
from utils2init import open_csv_or_archive_of_them, ExitStatus

if len(sys.argv) > 1:
    dir_in, raw_pattern_file = sys.argv[1].split('*', 1)
    dir_in = Path(dir_in)
    raw_pattern_file = f'*{Path(sys.argv[1])}' if raw_pattern_file else '*.txt'
    print(f'Search config file and input files in {dir_in} (default mask: {raw_pattern_file})')
else:
    dir_in = Path.cwd().resolve()
    raw_pattern_file = '*.csv'
    print(f'No input arguments => using current dir to search config file and input files (default mask: {raw_pattern_file})')


file_ini = dir_in / 'tcm.ini'

dir_csv_cor = None  # dir for cleaned (intermediate) raw files, None => same as dir_in
file_out = dir_in / 'test.h5'
tbl_out = 'test'


# inclinometer file format parameters that can be provided by config file
cfg_csv_tcm = [
    '--delimiter_chars', ',',  # \t not specify if need "None" useful for fixed length format
    '--skiprows_integer', '3',  # 0 skip header
    '--header', 'yyyy(text),mm(text),dd(text),HH(text),MM(text),SS(text),Ax,Ay,Az,Mx,My,Mz,Battery,Temp',
    '--dtype_list', '|S4, |S2, |S2, |S2, |S2, |S2, i2, i2, i2, i2, i2, i2, f8, f8',
    '--on_bad_lines', 'error',
    '--b_incremental_update', 'True',
    # '--min_date', '07.10.2017 11:04:00',  # not output data < min_date
    # '--max_date', '29.12.2018 00:00:00',  # UTC, not output data > max_date
    '--b_insert_separator', 'False',  # insert NaNs between files
    '--b_reuse_temporary_tables', 'False',  # Warning! If True and b_incremental_update= True then not replace temporary file with result before proc.
    # '--log', 'log/csv2h5_inclin_Kondrashov.log'  # log operations
    ]


fun_correct_name = partial(mod_incl_name, add_prefix='@')
raw_corrected = set(dir_in.glob(str(fun_correct_name(raw_pattern_file))))  # returns, may be not all, but only corrected file names
raw_corrected = {fun_correct_name(file) for file in (set(dir_in.glob(raw_pattern_file)) - raw_corrected)}  # returns all corrected files + may be some other (cor of cor) file names
raw_found = set(dir_in.glob(raw_pattern_file)) - raw_corrected  # excluding corrected and other not needed files
correct_fun = partial(correct_txt,
                    dir_out=dir_csv_cor, binary_mode=False, mod_file_name=fun_correct_name,
                    sub_str_list=[
                        b'^(?P<use>20\d{2}(,\d{1,2}){5}(,\-?\d{1,6}){6},\d{1,2}(\.\d{1,2})?,\-?\d{1,3}(\.\d{1,2})?).*',
                        b'^.+'
                        ])
# Correct non-regularity in source (primary) csv raw files
if n_raw_found := len(raw_found):
    print('Cleaning', n_raw_found, f'{dir_in / raw_pattern_file}', 'found files...')
n_raw_cor_found = 0
for file_in in raw_found:
    file_in = correct_fun(file_in)
    raw_corrected.add(file_in)

if (n_raw_cor_found := len(raw_corrected)) == 0:
    print('No', raw_pattern_file, end='raw files found')
    sys.exit(ExitStatus.failure)
else:
    print(f"Loading {n_raw_cor_found}{'' if n_raw_found else ' previously'} corrected files")
    # prints ' previously' if all source (primary) row files where deleted



for file_in in raw_corrected:
    csv2h5(
        ([str(file_ini)] if file_ini.is_file() else cfg_csv_tcm) + [
            '--path', str(file_in),
            '--blocksize_int', '50_000_000',  # 50Mbt
            '--table', tbl_out,
            '--db_path', str(file_out),
            # '--log', str(scripts_path / 'log/csv2h5_inclin_Kondrashov.log'),
            # '--on_bad_lines', 'warn',  # ?
            '--b_interact', '0',
            # '--fs_float', str(p_type[cfg['in']['probes_prefix']]['fs']),  # f'{fs(probe, file_in.stem)}',
            # '--dt_from_utc_seconds', str(cfg['in']['dt_from_utc'][probe].total_seconds()),
            '--b_del_temp_db', '1',
            '--csv_specific_param_dict', 'invert_magnetometer: True'
        ],
        **{'in': {'fun_proc_loaded': loaded_tcm}}  # not need if cfg file with corresponded name used
        )

    # +(
    #     [
    #      ] if probe_is_incl else []
    # ),
    # **{
    #     'filter': {
    #         'min_date': cfg['filter']['min_date'].get(probe, np.datetime64(0, 'ns')),
    #         'max_date': cfg['filter']['max_date'].get(probe, np.datetime64('now', 'ns')),
    #         # simple 'now' works in synchronous mode
    #         }
    #     }
