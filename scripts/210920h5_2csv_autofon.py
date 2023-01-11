import sys
from pathlib import Path
import re
import numpy as np
import pandas as pd

drive_d = 'D:' if sys.platform == 'win32' else '/mnt/D'  # to run on my Linux/Windows systems both
scripts_path = Path(drive_d + '/Work/_Python3/And0K/h5toGrid/scripts')
sys.path.append(str(Path(scripts_path).parent.resolve()))
# my funcs
from to_vaex_hdf5.h5tocsv import main_call as h5tocsv
from utils2init import st
import veuszPropagate
from to_pandas_hdf5.csv2h5 import main as csv2h5
from to_pandas_hdf5.gpx2h5 import main as gpx2h5
from to_pandas_hdf5.CTD_calc import main as CTD_calc
from to_pandas_hdf5.csv_specific_proc import proc_loaded_corr
from h5toGpx import main as h5toGpx
from grid2d_vsz import main as grid2d_vsz

devices = ['tr2', 'sp4', 'sp5']

for device in devices:
    path_db = Path(fr'D:\workData\BalticSea\210515_tracker\210726@sp4,5,tr2\210726_1200{device}.h5')

    h5tocsv([
        f'input.db_path="{path_db}"',
        f'input.tables=["{device}"]',
        f'input.tables_log=[""]',
        fr"out.text_path='{path_db.parent / 'text_output'}'",
        f'out.text_date_format="%Y-%m-%dT%H:%M:%S"',
        f'out.text_float_format="%.7g"',
        # f'out.file_name_fun="f\'{text_file_name_add}{{i+1:0>2}}.csv\'"',
        # f'out.file_name_fun_log="\'{text_file_name_add}POS.csv\'"',
        # rec_num;identific;station;Pres;Temp;cond;Sal;O2%;O2ppm;sigmaT;soundV
        # f'+out.cols_log={{rec_num: "i + 1", identific: "i + 1", station: "{i_cruise * 1000 + 1} + i" , LONG: Lon_st, LAT: Lat_st, DATE: index}}',
        # ''.join([
        #     f'+out.cols={{rec_num: "i + 1", identific: "i_log_row + 1", station: "{i_cruise * 1000 + 1} + i_log_row", ',
        #     ', '.join([p if ':' in p else f'{p}: {p}' for p in
        #                'Pres;Temp:Temp90;Cond;Sal;O2;O2ppm'.split(';')]),  # ;SigmaT;SoundVel
        #     '}'
        #     ]),
        f'+out.cols={{Time: index, *:*}}',
        'out.sep="\t"'
        ])