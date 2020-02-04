import sys
from pathlib import Path

drive_d = 'D:' if sys.platform == 'win32' else '/mnt/D'  # to run on my Linux/Windows systems both
scripts_path = Path(drive_d + '/Work/_Python3/And0K/h5toGrid/scripts')
sys.path.append(str(Path(scripts_path).parent.resolve()))
# my funcs
import veuszPropagate
from to_pandas_hdf5.csv2h5 import main as csv2h5
from to_pandas_hdf5.gpx2h5 import main as gpx2h5
from to_pandas_hdf5.CTD_calc import main as CTD_calc
from h5toGpx import main as h5toGpx
from grid2d_vsz import main as grid2d_vsz

device = 'CTD_Idronaut_OS310'
path_cruise = Path(r'd:\workData\BalticSea\190817_ANS42')
path_db = path_cruise / path_cruise.with_suffix('.h5').name  # name by dir
go = True  # False #
start = 15
# ---------------------------------------------------------------------------------------------
# navigation already added in scripts/190817_ANS42i316_underway.py

if st(1):  # False: #
    # Save {device} data to DB
    csv2h5([
        'ini/csv_CTD_Idronaut.ini',
        '--path', str(path_cruise / device / '_raw_txt' / '[19|45]*.txt'),
        '--db_path', str(path_db),
        '--table', f'{device}',
        '--dt_from_utc_hours', '0',
        '--header',
        'date(text),txtT(text),Pres(float),Temp(float),Cond(float),Sal(float),O2(float),O2ppm(float),pH(float),Eh(float)',
        '--delimiter_chars', '\\ \\',  # ''\s+',
        '--b_interact', '0',
        # '--b_raise_on_err', '0'
        ])

if st(5):  # False: #
    # Extract CTD runs (if files are not splitted on runs).
    # Note: Saves extended log needed by pattern used in next step with veuszPropagate
    # todo: be able provide log with (Lat,Lon) separately
    CTD_calc(['CTD_calc-find_runs.ini',
              '--db_path', str(path_db),
              '--tables_list', f'{device}',
              '--min_samples', '99',  # fs*depth/speed = 200: if fs = 10Hz for depth 20m
              '--min_dp', '9',
              '--b_keep_minmax_of_bad_files', 'True',
              '--b_skip_if_up_to_date', 'True',
              # todo: check it. If False need delete all previous result of CTD_calc() or set min_time > its last log time
              # '--output_files.tables_list', '',
              ])

if st(7):  # False: #
    # Draw {device} data profiles
    veuszPropagate.main(['ini/veuszPropagate.ini',
                         '--path', str(path_db),
                         '--pattern_path', str(path_cruise / device / '190713_2131.vsz'),
                         '--table_log', f'/{device}/logRuns',
                         '--add_custom_list', 'i0_USE_time_search_runs',  # 'i3_USE_timeRange',
                         '--add_custom_expressions',
                         """'[["{log_row[Index]:%Y-%m-%dT%H:%M:%S}", "{log_row[DateEnd]:%Y-%m-%dT%H:%M:%S}"]]'""",
                         # '--export_pages_int_list', '1', #'--b_images_only', 'True'
                         '--b_interact', '0',
                         '--b_update_existed', 'True'
                         ])

if start <= 8 and False:  #: # may not comment always because can not delete same time more than once
    # Deletng bad runs from DB:
    import pandas as pd

    # find bad runs that have time:
    time_in_bad_run_any = ['2018-10-16T19:35:41+00:00']
    tbl = f'/{device}'
    tbl_log = tbl + '/logRuns'
    print('Deletng bad runs from DB: tables: {}, {} run with time {}'.format(tbl, tbl_log, time_in_bad_run_any))
    with pd.HDFStore(path_db) as store:
        for t in time_in_bad_run_any:
            query_log = "index<=Timestamp('{}') and DateEnd>=Timestamp('{}')".format(t, t)
            df_log_bad_range = store.select(tbl_log, where=query_log)
            if len(df_log_bad_range) == 1:
                store.remove(tbl_log, where=query_log)
                store.remove(tbl, "index>=Timestamp('{}') and index<=Timestamp('{}')".format(
                    *[t for t in df_log_bad_range.DateEnd.items()][0]))
            else:
                print('Not found run with time {}'.format(t))

if st(9):  # False: #
    # Extract navigation data at time station starts to GPX waypoints
    h5toGpx(['ini/h5toGpx_CTDs.ini',
             '--db_path', str(path_db),
             '--tables_list', f'CTD_Idronaut_OS316, {device}',
             '--tables_log_list', 'logRuns',
             '--gpx_names_funs_list', """i+1""",
             '--gpx_names_fun_format', '{:03d}',
             '--select_from_tablelog_ranges_index', '0'
             ])
    go = False  # Hey! Prepare gpx tracks _manually_ before continue!

go = True
# if st(11):
#     # Extract navigation data at runs/starts to GPX tracks. Useful to indicate where no nav?
#     h5toGpx(['ini/h5toGpx_CTDs.ini',
#              '--db_path', str(path_db),
#              '--tables_list', f'{device}',
#              '--tables_log_list', 'logRuns',
#              '--select_from_tablelog_ranges_index', None,  # Export tracks
#              '--gpx_names_fun_format', '{1:%y%m%d}_{0:}',      # track name of format(timeLocal, tblD_safe)
#              '--gpx_names_funs_list', '"i, row.Index"',
#              '--gpx_names_funs_cobined', ''
#              ])

# go=False
if st(13):  # False: #
    # Save waypoints/routes from _manually_ prepared gpx to hdf5
    gpx2h5(['', '--path', str(path_cruise / r'navigation\CTD-sections=routes.gpx'),
            '--table_prefix', r'navigation/sectionsCTD'])

if st(15):  # False: #
    # Gridding
    grid2d_vsz(['ini/grid2d_vsz.ini', '--db_path', str(path_db),
                '--table_sections', r'navigation/sectionsCTD_routes',
                '--subdir', 'CTD-sections',
                '--begin_from_section_int', '2',  # 0,1 means no skip
                '--data_columns_list', 'Temp, Sal, SigmaTh, O2, O2ppm, Turb',  # todo: N^2 - need calc before
                '--filter_depth_wavelet_level_int', '11',  # 6, (7 for 2)
                '--min_depth', '35',
                '--max_depth', '110',
                # '--depecho_add_float', '0',
                ])

go = False
########################################################################################

# extract all navigation tracks
if False:  # True: #
    # sys.argv[0]= argv0   os_path.join(os_path.dirname(file_h5toGpx)
    h5toGpx(['ini/h5toGpx_nav_all.ini',
             '--db_path', str(path_db),
             '--tables_list', 'navigation',
             '--simplify_tracks_error_m_float', '10',
             '--period_files', 'D',
             '--tables_log_list', '""'
             # '--select_from_tablelog_ranges_index', None - defaut
             ])
