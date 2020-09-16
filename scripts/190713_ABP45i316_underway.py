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

device = 'CTD_Idronaut_OS316'
path_cruise = Path(r'd:\workData\BalticSea\190713_ABP45')
path_db = path_cruise / path_cruise.with_suffix('.h5').name  # name by dir
go = True  # False #
start = 7
# ---------------------------------------------------------------------------------------------

if st(1):  # False: #
    # Save {device} data to DB
    csv2h5([
        'ini/csv_CTD_Idronaut.ini',
        '--path', str(path_cruise / device / '_raw_txt' / '[19|45]*[0-9].txt'),
        '--db_path', str(path_db),
        '--table', f'{device}',
        '--dt_from_utc_hours', '0',
        '--header',
        'date(text),txtT(text),Pres(float),Temp(float),Cond(float),Sal(float),O2(float),O2ppm(float),pH(float),Eh(float),ChlA(float),Turb(float)',
        '--cols_not_use_list', 'pH,Eh,ChlA',
        '--delimiter_chars', '\\ \\',  # ''\s+',
        # todo  '--f_set_nan_list', 'Turb, x < 0',
        '--b_interact', '0',
        # '--b_raise_on_err', '0'
        ])

if st(3):  # False: #
    # Save depth to DB (saved gpx data is sparse and coinsedence of time samples is seldom, but need to check and delete duplicates)
    csv2h5([
        'ini/csv_nav_HYPACK.ini',
        '--path', str(path_cruise / r'navigation\echosounder_EA400\*.txt'),
        '--db_path', str(path_db),
        '--table', 'navigation',
        '--header', 'N, Time(text), Lat, Lon, , DepEcho',
        '--coltime_integer', '1',
        '--fun_date_from_filename', "pd.to_datetime(file_stem[:6], format='%y%m%d')",
        '--b_make_time_inc', 'False',
        '--b_interact', '0',
        # '--fs_float', '4'
        ])

if st(4):  # nav with depth is in next section
    # Save navigation to DB
    gpx2h5(['',
            '--db_path', str(path_db),
            '--path', str(path_cruise / r'navigation\source_OpenCPN\*.gpx'),
            '--tables_list', ',navigation,',  # skip waypoints
            '--table_prefix', r'',
            # '--date_min', '2019-07-17T14:00:00',
            ])

if st(5):  # False: #
    # Extract CTD runs (if files are not splitted on runs):
    CTD_calc(['CTD_calc-find_runs.ini',
              '--db_path', str(path_db),
              '--tables_list', f'{device}',
              '--min_samples', '99',  # fs*depth/speed = 200: if fs = 10Hz for depth 20m
              '--min_dp', '9',
              '--b_keep_minmax_of_bad_files', 'True',
              '--b_skip_if_up_to_date', 'True',
              # todo: check it. If False need delete all previous result of CTD_calc() or set min_time > its last log time
              # '--out.tables_list', '',
              ])

if st(7):  # False: #
    # Draw {device} data profiles
    veuszPropagate.main(['ini/veuszPropagate.ini',
                         '--path', str(path_db),
                         '--pattern_path', str(path_cruise / device / '190714_0757.vsz'),
                         '--table_log', f'/{device}/logRuns',
                         '--add_custom_list', 'i3_USE_time_search_runs',  # 'i3_USE_timeRange',
                         '--add_custom_expressions',
                         """'[["{log_row[Index]:%Y-%m-%dT%H:%M:%S}", "{log_row[DateEnd]:%Y-%m-%dT%H:%M:%S}"]]'""",
                         # '--export_pages_int_list', '1', #'--b_images_only', 'True'
                         '--b_interact', '0',
                         '--b_update_existed', 'True'
                         ])

go = False
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
             '--tables_list', f'{device}',
             '--tables_log_list', 'logRuns',
             '--gpx_names_funs_list', """'*(("s",i+1) if i<35 else ("",i-34))'""",
             '--gpx_names_fun_format', '{:s}{:03d}',
             '--select_from_tablelog_ranges_index', '0'
             ])
    go = False  # Hey! Prepare gpx tracks _manually_ before continue.

if st(11):
    # Extract navigation data at runs/starts to GPX tracks
    h5toGpx(['ini/h5toGpx_CTDs.ini',
             '--db_path', str(path_db),
             '--tables_list', f'{device}',
             '--tables_log_list', 'logRuns',
             '--select_from_tablelog_ranges_index', None,  # Export tracks
             '--gpx_names_fun_format', '{1:%y%m%d}_{0:}',  # track name of format(timeLocal, tblD_safe)
             '--gpx_names_funs_list', '"i, row.Index"',
             '--gpx_names_funs_cobined', ''
             ])

# go=False
if st(13):  # False: #
    # Save waypoints/routes from _manually_ prepared gpx to hdf5
    gpx2h5(['', '--path', str(path_cruise / r'navigation\CTD-sections=routes.gpx'),
            '--table_prefix', r'navigation/sectionsCTD'])
# go = True
if st(15):  # False: #
    # Gridding
    grid2d_vsz(['ini/grid2d_vsz.ini', '--db_path', str(path_db),
                '--table_sections', r'navigation/sectionsCTD_routes',
                '--subdir', 'CTD-sections',
                '--begin_from_section_int', '1',

                '--data_columns_list', 'Temp, Sal, SigmaTh, O2, O2ppm',  # N^2,
                '--filter_depth_wavelet_level_int', '7',  # 6, (7 for 2)
                '--depecho_add_float', '0',
                ])

go = False
########################################################################################

# extract all navigation tracks
if False:  # True: #
    # sys.argv[0]= argv0   os_path.join(os_path.dirname(file_h5toGpx)
    h5toGpx(['ini/h5toGpx_nav_all.ini',
             '--path_cruise', str(path_cruise),
             '--tables_list', 'navigation',
             '--simplify_tracks_error_m_float', '10',
             '--select_from_tablelog_ranges_index', None])
