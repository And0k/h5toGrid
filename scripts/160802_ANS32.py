import sys
from pathlib import Path

drive_d = 'D:' if sys.platform == 'win32' else '/mnt/D'  # to run on my Linux/Windows systems both
scripts_path = Path(drive_d + '/Work/_Python3/And0K/h5toGrid/scripts')
sys.path.append(str(Path(scripts_path).parent.resolve()))
# my funcs
from to_pandas_hdf5.csv2h5 import main as csv2h5
from to_pandas_hdf5.gpx2h5 import main as gpx2h5
from to_pandas_hdf5.CTD_calc import main as CTD_calc
from h5toGpx import main as h5toGpx

device = 'CTD_Idronaut#387'
path_cruise = Path(r'd:\workData\BalticSea\160802_ANS32')
path_db = path_cruise / path_cruise.with_suffix('.h5').name  # name by dir
go = True  # False #
start = 6
# ---------------------------------------------------------------------------------------------

if st(5):  # False: #
    # Extract CTD runs (if files are not splitted on runs):
    CTD_calc([  # 'ctd_calc-find_runs.ini',
        '--db_path', str(path_db),
        '--tables_list', f'{device}',
        # '--min_samples', '99',  # fs*depth/speed = 200: if fs = 10Hz for depth 20m
        # '--min_dp', '9',
        # '--b_keep_minmax_of_bad_files', 'True',
        '--path_csv', str(path_cruise / device / 'txt_processed'),
        '--data_columns_list', 'Pres, Temp90, Cond, Sal, O2, O2ppm, pH, Eh, SA, sigma0',
        '--b_incremental_update', 'True',
        # todo: check it. If False need delete all previous result of ctd_calc() or set min_time > its last log time
        '--out.tables_list', 'None',
        ])

# BAKLAN Matlab txt ouput
device = 'Baklan'
decimate_rate = 100
device_tbl = f'{device}_d{decimate_rate}'
if go:  # True: #
    csv2h5([
        'cfg/csv_Baklan.ini', '--db_path', str(path_db),
        '--path', str(path_cruise / device / '20*p1.txt'),
        '--delimiter_chars', '\\t',
        '--table', device_tbl])
if go:  # True: #
    h5toGpx([f'cfg/h5toGpx_{device}.ini', '--db_path', str(path_db),
             '--select_from_tablelog_ranges_index', '0'])
# also to get gpx is possible to execute ../scripts/filetime_to_gpx.py without csv2h5

# Now change starts of sections and excluded stations with specified symbol using MapSource

if go:  # True: #
    gpx2h5([
        '', '--db_path', str(path_db),
        '--path', str(path_cruise / 'navigation' / f'{device_tbl}_sections.gpx'),
        '--table_prefix', f'navigation/sections{device_tbl}'])

go = False
########################################################################################

# extract all navigation tracks
if False:  # True: #
    # sys.argv[0]= argv0   os_path.join(os_path.dirname(file_h5toGpx)
    h5toGpx(['cfg/h5toGpx_nav_all.ini',
             '--path_cruise', str(path_cruise),
             '--tables_list', 'navigation',
             '--simplify_tracks_error_m_float', '10',
             '--select_from_tablelog_ranges_index', None])
