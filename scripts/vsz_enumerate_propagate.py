import sys
from os import chdir as os_chdir
from pathlib import Path
import re
import numpy as np

drive_d = 'D:' if sys.platform == 'win32' else '/mnt/D'  # to run on my Linux/Windows systems both
scripts_path = Path(drive_d + '/Work/_Python3/And0K/h5toGrid/scripts')
sys.path.append(str(Path(scripts_path).parent.resolve()))
from itertools import takewhile
# my funcs
import veuszPropagate

# from to_pandas_hdf5.csv_specific_proc import proc_loaded_corr

from to_pandas_hdf5.h5toh5 import h5log_names_gen


path_cruise = Path(r'd:\WorkData\BalticSea\_other_data\_model\Copernicus\GoF\Veusz')
path_db = path_cruise / path_cruise.with_suffix('.h5').name  # same name as dir

cfg_in = {
    'pattern_path': path_cruise / '000000_0000.vsz',
    # 'min_time': np.datetime64('2022-12-21T10:02:00'),
    # 'max_time': '2020-12-30T22:37:00',
    }
f_row2name = lambda r: '{:%y%m%d_%H%M%S}.vsz'.format(r['Index'])
# It is possible to add an exact interval to the filename but the time after probe is back on surface can be determined
# only from next row, so we rely on ~pattern_loader.vsz to do it. Even freq=16Hz to determine last time not helps:
# '_{}s.vsz'.format(round(max(r['rows']/16, (r['DateEnd'] - r['Index'] + pd.Timedelta(300, "s")).total_seconds()))
pattern_code = cfg_in['pattern_path'].read_bytes()  # encoding='utf-8'
filename_st = None
os_chdir(cfg_in['pattern_path'].parent)
for file_index in range(806):
    filename = f'{file_index:03d}.vsz'
    path_vsz = cfg_in['pattern_path'].with_name(filename)
    path_vsz.write_bytes(pattern_code)  # re.sub(rb'^([^\n]+)', str_expr, pattern_code, count=1)
    # Get filename_st (do once)
    if filename_st is None:
        filename_st = filename
# cfg_in['min_time'] not works on filenames, so we convert it to 'start_file_index'
if 'min_time' in cfg_in:
    del cfg_in['min_time']  # del to count fro 0:
    start_file_index = len(list(takewhile(lambda x: x < filename_st, h5log_names_gen(cfg_in, f_row2name))))
else:
    start_file_index = 0
veuszPropagate.main([
    'cfg/veuszPropagate.ini',
    '--path', str(cfg_in['pattern_path'].with_name('???.vsz')),  # _*s path_db),
    '--pattern_path', f"{cfg_in['pattern_path']}_",
    # here used to auto get export dir only. must not be not existed file path
    # '--table_log', f'/{device}/logRuns',
    # '--add_custom_list', f'{device_veusz_prefix}USE_time_search_runs',  # 'i3_USE_timeRange',
    # '--add_custom_expressions',
    # """'[["{log_row[Index]:%Y-%m-%dT%H:%M:%S}", "{log_row[DateEnd]:%Y-%m-%dT%H:%M:%S}"]]'""",
    # '--export_pages_int_list', '7', # 0  '--b_images_only', 'True'
    '--b_update_existed', 'True',  # False is default todo: allow "delete_overlapped" time named files
    '--b_interact', '0',
    '--b_images_only', 'True',  # mandatory
    '--b_execute_vsz', 'True',
    '--start_file_index', str(start_file_index),
    '--export_format', 'png'
    # '--min_time', cfg_in['min_time'].item().isoformat(),  # not works on filenames (no time data)
    # '--max_time', cfg_in['max_time'].item().isoformat(),
    ])ч
