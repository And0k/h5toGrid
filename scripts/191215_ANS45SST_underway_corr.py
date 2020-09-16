import sys
from functools import partial
from pathlib import Path

drive_d = 'D:' if sys.platform == 'win32' else '/mnt/D'  # to run on my Linux/Windows systems both
scripts_path = Path(drive_d + '/Work/_Python3/And0K/h5toGrid/scripts')
sys.path.append(str(Path(scripts_path).parent.resolve()))
# my funcs
from utils2init import st as st_full
import veuszPropagate
from to_pandas_hdf5.csv2h5 import main as csv2h5
from to_pandas_hdf5.gpx2h5 import main as gpx2h5
from to_pandas_hdf5.CTD_calc import main as CTD_calc
from h5toGpx import main as h5toGpx
from grid2d_vsz import main as grid2d_vsz

# ---------------------------------------------------------------------------------------------
device = 'CTD_SST_48Mc#1253'
path_cruise = Path(r'd:\workData\BalticSea\191215_ANS46')
path_db = path_cruise / path_cruise.with_suffix('.h5').name  # same name as dir
device_veusz_prefix = 'ss_'
go = True  # False #
start = 130  # 5 70 110
end = 120  # 60 80 120
# Usually need stop before steps that need manual preparings (70): so stop at 60, and last step for device.
# Gridding (last step) needs debugging if interactive filtering is needed
# ---------------------------------------------------------------------------------------------
st = partial(st_full, start=start, end=end)

if st(1) and False:  # nav with depth is in next section
    # Save navigation to DB
    gpx2h5(['',
            '--db_path', str(path_db),
            '--path', str(path_cruise / r'navigation\_raw\*.gpx'),
            '--tables_list', ',navigation,',  # skip waypoints
            '--table_prefix', r'',
            # '--date_min', '2019-07-17T14:00:00',
            ])

if st(5):
    csv2h5(['ini/csv_nav_supervisor.ini',
            '--db_path', str(path_db),
            '--path', str(path_cruise / r'navigation\bridge\??????.txt'),
            '--table', 'navigation',  # skip waypoints
            '--b_remove_duplicates', 'True',
            '--csv_specific_param_dict', 'DepEcho_add:4.5',
            '--min_dict', 'DepEcho:10',
            ])

if st(10):  # False: #
    # Save CTD_SST_48Mc Underway to DB
    from to_pandas_hdf5.csv_specific_proc import proc_loaded_sea_and_sun

    csv2h5([
        'ini/csv_CTD_Sea&Sun.ini',
        '--path', str(path_cruise / device / '_raw' / '19*[0-9].TOB'),
        '--db_path', str(path_db),
        '--dt_from_utc_hours', '0',
        '--header', 'Number,Date(text),Time(text),Pres,Temp,Sal,O2,O2ppm,SIGMA,Cond,Vbatt,SVel',
        '--cols_not_use_list', 'Number,SIGMA,Vbatt,SVel',
        '--delimiter_chars', '\\ \\',  # ''\s+',
        '--table', f'{device}',
        '--b_interact', '0'
        # '--b_raise_on_err', '0',
        ],
        **{'in': {
            'fun_proc_loaded': proc_loaded_sea_and_sun,
            'csv_specific_param': {'Temp_fun': lambda x: (x + 0.254) / 1.00024,
                                   # 'Temp_add': 0.254, And convert to ITS90
                                   'Sal_fun': lambda x: (1 + 0.032204423446495364) * x + 0.045516504802752523,
                                   'Cond_fun': lambda x: -0.000098593 * x ** 2 + 1.040626 * x + 0.01386
                                   }
            }}
        )

if st(20):  # False: #
    # Extract CTD runs (if files are not splitted on runs).
    # Note: Saves extended log needed by pattern used in next step with veuszPropagate
    # todo: be able provide log with (Lat,Lon) separately
    CTD_calc(['ini/CTD_calc-find_runs.ini',
              '--db_path', str(path_db),
              '--tables_list', f'{device}',
              '--min_samples', '50',  # fs*depth/speed = 200: if fs = 10Hz for depth 20m
              '--min_dp', '15',
              # '--b_keep_minmax_of_bad_files', 'True',
              '--b_skip_if_up_to_date', 'True',
              # todo: check it. If False need delete all previous result of CTD_calc() or set min_time > its last log time
              # '--out.tables_list', '',
              ])

if st(30):  # False: #
    # Draw {device} data profiles
    veuszPropagate.main(['ini/veuszPropagate.ini',
                         '--path', str(path_db),
                         '--pattern_path', str(path_cruise / device / '~pattern~.vsz'),
                         '--table_log', f'/{device}/logRuns',
                         '--add_custom_list', f'{device_veusz_prefix}USE_time_search_runs',  # 'i3_USE_timeRange',
                         '--add_custom_expressions',
                         """'[["{log_row[Index]:%Y-%m-%dT%H:%M:%S}", "{log_row[DateEnd]:%Y-%m-%dT%H:%M:%S}"]]'""",
                         # '--export_pages_int_list', '1', #'--b_images_only', 'True'
                         '--b_interact', '0',
                         # '--b_update_existed', 'True',
                         # '--b_images_only', 'True'
                         ])

if start <= 40 and False:  #: # may not comment always because can not delete same time more than once
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

if st(50):  # False: #
    # Extract navigation data at time station starts to GPX waypoints
    h5toGpx(['ini/h5toGpx_CTDs.ini',
             '--db_path', str(path_db),
             '--tables_list', f'{device}',  # CTD_Idronaut_OS316',
             '--tables_log_list', 'logRuns',
             '--gpx_names_funs_list', """i+1""",
             '--gpx_names_fun_format', '{:02d}',
             '--select_from_tablelog_ranges_index', '0'
             ])
    go = False  # Hey! Prepare gpx tracks _manually_ before continue!

go = True
if start <= 60 and False:
    # Extract navigation data at runs/starts to GPX tracks. Useful to indicate where no nav?
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
if st(70):  # False: #
    # Save waypoints/routes from _manually_ prepared "CTD-sections=routes.gpx" to hdf5
    gpx2h5(['', '--path', str(path_cruise / r'navigation\CTD-sections=routes.gpx'),
            '--table_prefix',
            r'navigation/sectionsCTD'])  # need copy reult from {path_db}_not_sorted manually, todo: auto copy

if st(80):  # False: #
    # Gridding
    # Note: Prepare veusz "zabor" pattern before
    grid2d_vsz(['ini/grid2d_vsz.ini', '--db_path', str(path_db),
                '--table_sections', r'navigation/sectionsCTD_routes',
                '--subdir', 'CTD-sections',
                '--begin_from_section_int', '1',  # values <= 1 means no skip
                '--data_columns_list', "Temp, Sal, SigmaTh, O2, O2ppm, soundV",
                # 'Eh, pH',  todo: N^2 - need calc before
                '--max_depth', '110',
                '--filter_depth_wavelet_level_int', '7',  # 4, 2 for section 3
                '--filter_ctd_bottom_edge_float', 'True',
                # '--x_resolution', '0.2',
                # '--y_resolution', '5',
                '--dt_search_nav_tolerance_seconds', '120',
                '--symbols_in_veusz_ctd_order_list',
                "'Triangle, Green', 'Diamond, Blue', 'Triangle, Red', 'Square, Green'",
                '--b_temp_on_its90', 'True',  # modern probes
                '--blank_level_under_bot', '-150',
                # '--interact', 'False',
                # '--b_reexport_images', 'True'
                ])

    # todo: bug: bad top and bottom edges

# Export csv with some new calculated paremeters
if st(110):  # False: #
    # Extract CTD runs (if files are not splitted on runs):
    CTD_calc([  # 'CTD_calc-find_runs.ini',
        '--db_path', str(path_db),
        '--tables_list', f'{device}',
        '--tables_log', '{}/logRuns',
        # '--min_samples', '99',  # fs*depth/speed = 200: if fs = 10Hz for depth 20m
        # '--min_dp', '9',
        # '--b_keep_minmax_of_bad_files', 'True',
        '--path_csv', str(path_cruise / device / 'txt_processed'),
        '--data_columns_list', 'Pres, Temp90, Cond, Sal, O2, O2ppm, SA, sigma0, depth, soundV',  # pH, Eh,
        '--b_skip_if_up_to_date', 'True',
        # todo: check it. If False need delete all previous result of CTD_calc() or set min_time > its last log time
        '--out.tables_list', 'None',
        ])

# Meteo
if st(120):  # True: #
    csv2h5([
        'ini/csv_meteo.ini', '--path',  # to_pandas_hdf5/
        str(path_cruise / r"meteo\ship's_meteo_st_source\*.mxt"), '--header',
        'date(text),Time(text),t_air,Vabs_m__s,Vdir,dew_point,Patm,humidity,t_w,precipitation',
        '--coldate_integer', '0', '--coltime_integer', '1',
        '--cols_not_use_list', 't_w,precipitation',  # bad constant data
        '--delimiter_chars', ',', '--max_text_width', '12',
        '--b_raise_on_err', 'False', '--b_insert_separator', 'False',
        '--chunksize_percent_float', '500',
        '--fs_float', '60',
        '--skiprows', '0'
        ])

# extract all navigation tracks
if st(130):  # True: #
    # sys.argv[0]= argv0   os_path.join(os_path.dirname(file_h5toGpx)
    h5toGpx(['ini/h5toGpx_nav_all.ini',
             '--db_path', str(path_db),
             '--tables_list', 'navigation',
             '--simplify_tracks_error_m_float', '10',
             '--period_files', 'D',
             '--tables_log_list', '""'
             # '--select_from_tablelog_ranges_index', None - defaut
             ])
########################################################################################
