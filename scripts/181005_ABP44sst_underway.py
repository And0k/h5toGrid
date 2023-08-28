import sys
from pathlib import Path

drive_d = 'D:' if sys.platform == 'win32' else '/mnt/D'  # allows to run on both my Linux and Windows systems:
scripts_path = Path(drive_d + '/Work/_Python3/And0K/h5toGrid/scripts')
sys.path.append(str(Path(scripts_path).parent.resolve()))
from os import path as os_path
# my funcs
import veuszPropagate
from to_pandas_hdf5.csv2h5 import main as csv2h5
from to_pandas_hdf5.gpx2h5 import main as gpx2h5
from to_pandas_hdf5.CTD_calc import main as CTD_calc
from h5toGpx import main as h5toGpx
from grid2d_vsz import main as grid2d_vsz

path_cruise = r'd:\workData\BalticSea\181005_ABP44'  # d:\WorkData\181005_ABP44
path_db = os_path.join(path_cruise, '181005_ABP44.h5')
go = False
start = 13
# ---------------------------------------------------------------------------------------------

if st(1):  # False: #
    # Save CTD_SST_48Mc Underway to DB
    csv2h5([
        'cfg/csv_CTD_SST.ini',
        '--path', os_path.join(path_cruise, 'CTD_SST_48Mc#1253\Time_in_TOB=UTC\18*[0-9].TOB'),
        '--dt_from_utc_hours', '0',
        '--header', 'Number,Date(text),Time(text),Pres,Temp,Sal,O2,O2ppm,SIGMA,Cond,Vbatt,SVel',
        '--cols_not_save_list', 'Number,SIGMA,Vbatt,SVel',
        '--delimiter_chars', '\\ \\',  # ''\s+',
        '--table', 'CTD_SST_48Mc',
        '--b_interact', '0',
        # '--on_bad_lines', 'warn'
        ])

if start <= 3 and False:  # go: #  # already done in scripts/181005_ABP44sst.py
    # Save navigation to DB
    gpx2h5(['',
            '--db_path', path_db,
            '--path', os_path.join(path_cruise, r'navigation\source_OpenCPN\*.gpx'),
            '--tables_list', ',navigation,',  # skip waypoints
            '--table_prefix', r''])

if st(5):  # False: #
    # Extract CTD runs (if files are not splitted on runs):
    CTD_calc(['ctd_calc-find_runs.ini',
              '--db_path', path_db,
              '--tables_list', 'CTD_SST_48Mc',
              '--min_samples', '100',  # '100' is for 20m if Speed=1m/s
              '--min_dp', '20',
              '--b_incremental_update', 'True',
              # todo: check it. If False need delete all previous result of ctd_calc() or set min_time > its last log time
              # '--out.tables_list', '',
              ])

if st(7):  # False: #
    # Draw SST 48M data profiles
    veuszPropagate.main(['cfg/veuszPropagate.ini',
                         '--db_path', path_db,
                         '--pattern_path',
                         os_path.join(path_cruise, r'CTD_SST_48Mc#1253\profiles\181005_1810-1813.vsz'),
                         '--table_log', '/CTD_SST_48Mc/logRuns',
                         '--add_custom_list', 'USE_timeRange',
                         '--add_custom_expressions',
                         """'[["{log_row[Index]:%Y-%m-%dT%H:%M:%S}", "{log_row[DateEnd]:%Y-%m-%dT%H:%M:%S}"]]'""",
                         # '--export_pages_int_list', '1', #'--b_images_only', 'True'
                         '--b_interact', '0',
                         # '--b_update_existed', 'True'
                         ])

if st(8):  #: # may not comment always because can not delete same time more than once
    # Deletng bad runs from DB:
    import pandas as pd

    # find bad runs that have time:
    time_in_bad_run_any = ['2018-10-16T19:35:41+00:00']
    tbl = '/CTD_SST_48Mc'
    tbl_log = tbl + '/logRuns'
    print('Deletng bad runs from DB: tables: {}, {} run with time {}'.format(tbl, tbl_log, time_in_bad_run_any))
    with pd.HDFStore(path_db) as store:
        for t in time_in_bad_run_any:
            query_log = "index<='{}' & DateEnd>='{}'".format(t, t)
            df_log_bad_range = store.select(tbl_log, where=query_log)
            if len(df_log_bad_range) == 1:
                store.remove(tbl_log, where=query_log)
                store.remove(tbl, "index>='{}' & index<='{}'".format(
                    *[t for t in df_log_bad_range.DateEnd.items()][0]))
            else:
                print('Not found run with time {}'.format(t))

if st(9):  # False: #
    # Extract navigation data at time station starts to GPX waypoints
    h5toGpx(['cfg/h5toGpx_CTDs.ini',
             '--db_path', path_db,
             '--tables_list', 'CTD_SST_48Mc',
             '--tables_log_list', 'logRuns',
             '--gpx_names_funs_list', """'*(("s",i+1) if i<35 else ("",i-34))'""",
             '--gpx_names_fun_format', '{:s}{:03d}',
             '--select_from_tablelog_ranges_index', '0'
             ])
    go = False  # Hey! Prepare gpx tracks _manually_ before continue.

if st(11):
    # Extract navigation data at runs/starts to GPX tracks
    h5toGpx(['cfg/h5toGpx_CTDs.ini',
             '--db_path', path_db,
             '--tables_list', 'CTD_SST_48Mc',
             '--tables_log_list', 'logRuns',
             '--select_from_tablelog_ranges_index', None,  # Export tracks
             '--gpx_names_fun_format', '{1:%y%m%d}_{0:}',  # track name of format(timeLocal, tblD_safe)
             '--gpx_names_funs_list', '"i, row.Index"',
             '--gpx_names_funs_cobined', ''
             ])

# go=False
if st(13):  # False: #
    # Save waypoints/routes from _manually_ prepared gpx to hdf5
    gpx2h5(['', '--path', os_path.join(path_cruise, r'navigation\CTD-sections=routes.gpx'),
            '--table_prefix', r'navigation/sectionsCTD'])
go = True
if st(15):  # False: #
    # Gridding
    grid2d_vsz(['cfg/grid2d_vsz.ini', '--db_path', path_db,
                '--table_sections', r'navigation/sectionsCTD_routes',
                '--subdir', 'CTD-sections',
                '--begin_from_section_int', '1',

                '--data_columns_list', 'Temp, Sal, SigmaTh, O2, O2ppm',  # 'N^2'
                '--filter_depth_wavelet_level_int', '7',  # 6, (7 for 2)
                '--depecho_add_float', '0',
                ])

go = False
########################################################################################

if st(10):  # False: #
    # Save gpx from trackers to DB
    gpx2h5(['',
            '--db_path', path_db,  # str(Path().with_name('trackers_temp')),
            '--path', os_path.join(path_cruise, r'navigation\*spot*.gpx'),
            '--table_prefix', r'navigation/',
            '--segments_cols_list', "time, latitude, longitude, comment",
            '--out.segments_cols_list', 'Time, Lat, Lon, comment',
            '--tables_list', ',,tracker{}', ])
# go = True
if st(11):  # False: #
    # Export trackers tracks to GPX tracks
    h5toGpx(['cfg/h5toGpx_nav_all.ini',
             '--db_path', path_db,
             '--tables_list', 'tracker{}',
             '--select_from_tablelog_ranges_index', None,  # Export tracks
             '--gpx_names_fun_format', '{1:%y%m%d}_{0:}',  # track name of format(timeLocal, tblD_safe)
             '--gpx_names_funs_list', '"i, row.Index"',
             '--gpx_names_funs_cobined', ''
             ])
# extract all navigation tracks
if False:  # True: #
    # sys.argv[0]= argv0   os_path.join(os_path.dirname(file_h5toGpx)
    h5toGpx(['cfg/h5toGpx_nav_all.ini',
             '--path_cruise', path_cruise,
             '--tables_list', 'navigation',
             '--simplify_tracks_error_m_float', '10',
             '--select_from_tablelog_ranges_index', None])
