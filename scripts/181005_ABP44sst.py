import sys
from os import path as os_path
from pathlib import Path

# my funcs
drive_d = 'D:' if sys.platform == 'win32' else '/mnt/D'  # allows to run on both my Linux and Windows systems:
scripts_path = Path(drive_d + '/Work/_Python3/And0K/h5toGrid/scripts')
sys.path.append(str(Path(scripts_path).parent.resolve()))
import veuszPropagate
from to_pandas_hdf5.csv2h5 import main as csv2h5
from to_pandas_hdf5.gpx2h5 import main as gpx2h5
from h5toGpx import main as h5toGpx
from grid2d_vsz import main as grid2d_vsz

path_cruise = r'd:\workData\BalticSea\181005_ABP44'  # d:\WorkData\181005_ABP44
path_in_SST_48Mc = os_path.join(path_cruise, 'CTD_SST_48Mc#1253/Time_in_TOB=UTC+2h/ABP44*[0-9].TOB')
path_db = os_path.join(path_cruise, '181005_ABP44.h5')
go = True
start = 20
# ---------------------------------------------------------------------------------------------

if st(1):  # False: #
    # Draw SST 48M data profiles
    veuszPropagate.main(['cfg/veuszPropagate.ini',
                         '--path', path_in_SST_48Mc,
                         '--pattern_path', os_path.join(path_cruise, r'CTD_SST_48Mc#1253\ABP44001.vsz'),
                         '--before_next', 'restore_config',
                         '--eval_list',
                         """
                         "ImportFileCSV(u'{nameRFE}', dateformat=u'hh:mm:ss', delimiter=u' ', encoding=u'ascii', headermode=u'none', linked=True, dsprefix=u'_', renames={{u'_col1': u'ss_#', u'_col10': u'ss_Cond', u'_col11': u'ss_VBat', u'_col2': u'ss_date_txt', u'_col3': u'ss_time_part', u'_col4': u'ss_Pres', u'_col5': u'ss_Temp', u'_col6': u'ss_Sal', u'_col7': u'ss_O2', u'_col8': u'ss_O2ppm', u'_col9': u'ss_SigmaT'}}, rowsignore=30, skipwhitespace=True)",
                         "TagDatasets(u'loaded', [u'ss_#', u'ss_Cond', u'ss_VBat', u'ss_date_txt', u'ss_time_part', u'ss_Pres', u'ss_Temp', u'ss_Sal', u'ss_O2', u'ss_O2ppm', u'ss_SigmaT'])"
                         """,
                         # '--export_pages_int_list', '1', #'--b_images_only', 'True'
                         '--b_interact', '0',
                         # '--b_update_existed', '1'
                         ])

if st(2):  # False: #
    # Save CTD_SST_48Mc to DB
    csv2h5([
        'cfg/csv_CTD_Sea&Sun.ini',
        '--path', path_in_SST_48Mc,
        '--dt_from_utc_hours', '2',
        '--header', 'Number,Date(text),Time(text),Pres,Temp,Sal,O2,O2ppm,SIGMA,Cond,Vbatt,SVel',
        '--cols_not_use_list', 'Number,SIGMA,Vbatt,SVel',
        '--delimiter_chars', '\\ \\',  # ''\s+',
        '--table', 'CTD_SST_48Mc',
        '--b_interact', '0',
        # '--b_raise_on_err', '0'
        ])

if st(4):  # False: #
    # Save navigation to DB
    gpx2h5(['',
            '--db_path', path_db,
            '--path', os_path.join(path_cruise, r'navigation\source_OpenCPN\tracks\*.gpx'),
            '--tables_list', ',navigation,',  # skip waypoints
            '--table_prefix', r'',
            # '--sort', 'False',
            ])

if st(5):  # False: #
    # Save depth to DB (saved gpx data is sparse and coinsedence of time samples is seldom, but need to check and delete duplicates)
    csv2h5(['cfg/csv_nav_HYPACK.ini',
            '--db_path', path_db,
            '--path',
            os_path.join(path_cruise, r'd:\workData\BalticSea\181005_ABP44\navigation\bathymetry_HYPACK\*.txt'),
            '--table', 'navigation',
            '--sort', 'False'  # '--fs_float', '4'
            ])

# go = False
if st(6):  # False: #
    # Extract navigation data at time station starts to GPX waypoints
    h5toGpx(['cfg/h5toGpx_CTDs.ini',
             '--db_path', path_db,
             '--tables_list', 'CTD_SST_48Mc',
             '--gpx_names_fun_format', 's{:02d}',
             '--select_from_tablelog_ranges_index', '0'
             ])

if st(7):  # False: #
    # Save waypoints/routes from _manually_ prepared gpx to hdf5
    gpx2h5(['', '--path', os_path.join(path_cruise, r'navigation\CTD-sections=routes.gpx'),
            '--table_prefix', r'navigation/sectionsCTD'])

if st(8):  # False: #
    # Gridding
    grid2d_vsz(['cfg/grid2d_vsz.ini', '--db_path', path_db,
                '--table_sections', r'navigation/sectionsCTD_routes',
                '--subdir', 'CTD-sections',
                '--data_columns_list', 'Temp, Sal, SigmaTh, O2, O2ppm'  # 'N^2'
                ])

# go = False
########################################################################################

if st(10):  # False: #
    # Save gpx from treckers to DB
    gpx2h5(['',
            '--db_path', path_db,  # str(Path().with_name('trackers_temp')),
            '--path', os_path.join(path_cruise, r'navigation\*spot*.gpx'),
            '--table_prefix', r'navigation/',
            '--segments_cols_list', "time, latitude, longitude, comment",
            '--out.segments_cols_list', 'Time, Lat, Lon, comment',
            '--tables_list', ',,tracker{}', ])
# go = True
if st(11):  # False: #
    # Export treckers tracks to GPX tracks
    h5toGpx(['cfg/h5toGpx_nav_all.ini',
             '--db_path', path_db,
             '--tables_list', 'tracker{}',
             '--select_from_tablelog_ranges_index', None,  # Export tracks
             '--gpx_names_fun_format', '{1:%y%m%d}_{0:}',  # track name of format(timeLocal, tblD_safe)
             '--gpx_names_funs_list', '"i, row.Index"',
             '--gpx_names_funs_cobined', ''
             ])

if st(20):  # False: #
    # extract all navigation tracks
    h5toGpx(['cfg/h5toGpx_nav_all.ini',
             '--db_path', path_db,
             '--tables_list', 'navigation',
             # '--period_segments', 'D',
             '--period_files', 'D',
             '--tables_log_list', '""'
             # '--simplify_tracks_error_m_float', '10',
             # '--select_from_tablelog_ranges_index', None
             ])
