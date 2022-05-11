import sys
from pathlib import Path

drive_d = 'D:' if sys.platform == 'win32' else '/mnt/D'  # to run on my Linux/Windows systems both
scripts_path = Path(drive_d + '/Work/_Python3/And0K/h5toGrid/scripts')
sys.path.append(str(Path(scripts_path).parent.resolve()))
# my funcs
import veuszPropagate
from to_pandas_hdf5.bin2h5 import main as bin2h5
from to_pandas_hdf5.CTD_calc import main as CTD_calc

# ---------------------------------------------------------------------------------------------
device = 'CTD_NeilBrown_Mark3'
path_cruise = Path(r'd:\WorkData\AtlanticOcean\161113_ANS33')  #d:\workData\AtlanticOcean\191000
path_raw = path_cruise / device / '_raw/bin/Realterm'
path_db_raw = path_raw.parent.parent / path_cruise.with_suffix('.h5').name
path_db = path_cruise / path_cruise.with_suffix('.h5').name  # same name as dir
device_veusz_prefix = 'm3_'
go = True  # False #

start = 10
end = 20  # 10000


# ---------------------------------------------------------------------------------------------
def st(current):
    if (start <= current <= end) and go:
        print(f'step {current}')
        return True
    return False


if st(10):  # False: #
    # Save CTD_SST_48Mc Underway to DB
    bin2h5(['cfg/bin_Brown.ini',
            '--path', str(path_raw / '[0-9]*.bin'),  #'2019*[0-9].bin'
            # \CTD_NeilBrown_Mark3\_raw\20191013_170300.bin
            '--db_path', str(path_db_raw),
            # '--dt_from_utc_hours', '0',
            # '--header', 'Number,Date(text),Time(text),Pres,Temp,Sal,O2,O2ppm,SIGMA,Cond,Vbatt,SVel',
            # '--cols_not_use_list', 'Number,SIGMA,Vbatt,SVel',
            # '--delimiter_chars', '\\ \\', #''\s+',
            '--table', f'{device}',
            '--b_interact', '0',
            # '--on_bad_lines', 'warn'
            ])

if st(20):  # False: #
    # Extract CTD runs (if files are not splitted on runs).
    # Note: Saves extended log needed by pattern used in next step with veuszPropagate
    # todo: be able provide log with (Lat,Lon) separately
    CTD_calc(['cfg/CTD_calc_Brown.ini',
              '--db_path', str(path_db_raw),
              '--tables_list', f'{device}',
              '--out.db_path', str(path_db),
              '--min_samples', '50',  # fs*depth/speed = 200: if fs = 10Hz for depth 20m
              '--min_dp', '5',
              '--b_keep_minmax_of_bad_files', 'True',
              '--path_csv', str(path_cruise / device / 'txt_processed'),
              '--data_columns_list', 'Pres, Temp, Temp90, Cond, Sal',  # , sigma0, Temp90  SA,depth, soundV
              '--b_incremental_update', 'True',
              # todo: check it. If False need delete all previous result of CTD_calc() or set min_time > its last log time
              # '--out.tables_list', '',
              '--path_coef', r'd:\Work\_Python3\And0K\h5toGrid\scripts\ini\coef#Brawn_190918.txt'
              ])

if st(30):  # False: #
    # Draw {device} data profiles
    veuszPropagate.main(['cfg/veuszPropagate.ini',
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
    h5toGpx(['cfg/h5toGpx_CTDs.ini',
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
    h5toGpx(['cfg/h5toGpx_CTDs.ini',
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
    grid2d_vsz(['cfg/grid2d_vsz.ini', '--db_path', str(path_db),
                '--table_sections', r'navigation/sectionsCTD_routes',
                '--subdir', 'CTD-sections',
                '--begin_from_section_int', '1',  # values <= 1 means no skip
                '--data_columns_list', "Temp, Sal, SigmaTh, O2, O2ppm, soundV",
                # 'Eh, pH',  todo: N^2 - need calc before
                '--filter_depth_wavelet_level_int', '2',  # 4, 2 for section 3
                '--x_resolution', '0.2',
                # '--y_resolution', '5',
                '--dt_search_nav_tolerance_seconds', '120',
                '--symbols_in_veusz_ctd_order_list',
                "'Triangle, Green', 'Diamond, Blue', 'Triangle, Red', 'Square, Green'",
                '--b_temp_on_its90', 'True',  # modern probes
                '--blank_level_under_bot', '-600',
                '--interact', 'False',
                ])

########################################################################################
go = False
# Export csv with new parameters
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
        '--data_columns_list', 'Pres, Temp90, Cond, Sal, O2, O2ppm, pH, Eh, SA, sigma0, depth, soundV',
        '--b_incremental_update', 'True',
        # todo: check it. If False need delete all previous result of CTD_calc() or set min_time > its last log time
        '--out.tables_list', 'None',
        ])

# # ################################################################################################################
go = True
device_prev, device = device, 'CTD_Idronaut_OS310'
device_veusz_prefix = 'i0_'

if st(200):  # False: #

    # Save {device} data to DB
    # Save {device} data to DB
    csv2h5([
        'cfg/csv_CTD_Idronaut.ini',
        '--path', str(path_cruise / device / '_raw' / '19*.txt'),
        '--db_path', str(path_db),
        '--table', f'{device}',
        '--dt_from_utc_hours', '0',
        '--header',
        'date(text),txtT(text),Pres(float),Temp(float),Cond(float),Sal(float),O2(float),O2ppm(float),pH(float),Eh(float)',
        '--delimiter_chars', '\\ \\',  # ''\s+',
        '--b_interact', '0',
        # '--on_bad_lines', 'warn'
        ])

if st(220):  # False: #
    # Extract CTD runs (if files are not splitted on runs).
    # Note: Saves extended log needed by pattern used in next step with veuszPropagate
    # todo: be able provide log with (Lat,Lon) separately
    CTD_calc(['CTD_calc-find_runs.ini',
              '--db_path', str(path_db),
              '--tables_list', f'{device}',
              '--min_samples', '99',  # fs*depth/speed = 200: if fs = 10Hz for depth 20m
              '--min_dp', '9',
              '--b_keep_minmax_of_bad_files', 'True',
              '--b_incremental_update', 'True',
              # todo: check it. If False need delete all previous result of CTD_calc() or set min_time > its last log time
              # '--out.tables_list', '',
              ])

if st(230):  # False: #
    # Draw {device} data profiles
    veuszPropagate.main(['cfg/veuszPropagate.ini',
                         '--path', str(path_db),
                         '--pattern_path', str(path_cruise / device / '~pattern~.vsz'),
                         '--table_log', f'/{device}/logRuns',
                         '--add_custom_list', f'{device_veusz_prefix}USE_time_search_runs',  # 'i3_USE_timeRange',
                         '--add_custom_expressions',
                         """'[["{log_row[Index]:%Y-%m-%dT%H:%M:%S}", "{log_row[DateEnd]:%Y-%m-%dT%H:%M:%S}"]]'""",
                         # '--export_pages_int_list', '4', #'--b_images_only', 'True'
                         '--b_interact', '0',
                         '--b_update_existed', 'True',
                         # '--b_images_only', 'True'
                         ])

if st(250):  # False: #
    # Extract navigation data at time station starts to GPX waypoints
    h5toGpx(['cfg/h5toGpx_CTDs.ini',
             '--db_path', str(path_db),
             '--tables_list', f'{device_prev}, {device}',
             '--tables_log_list', 'logRuns',
             '--gpx_names_funs_list', """i+1""",
             '--gpx_names_fun_format', '{:02d}',
             '--select_from_tablelog_ranges_index', '0'
             ])
    go = False  # Hey! Prepare gpx tracks _manually_ before continue!

go = True
# go=False
if st(270):  # False: #
    # Save waypoints/routes from _manually_ prepared "CTD-sections=routes.gpx" to hdf5
    gpx2h5(['', '--path', str(path_cruise / r'navigation\CTD-sections=routes.gpx'),
            '--table_prefix',
            r'navigation/sectionsCTD'])  # need copy reult from {path_db}_not_sorted manually, todo: auto copy

device = 'tracker'
path_db_device = path_db.with_name(f'~{device}s.h5')
if st(300):  # False: #
    # Save gpx from treckers to DB
    gpx2h5(['',
            '--db_path', str(path_db_device),
            '--path', str(path_cruise / rf'navigation/{device}s/*spot*messages.gpx'),
            '--table_prefix', r'navigation/',
            '--segments_cols_list', "time, latitude, longitude, comment",
            '--out.segments_cols_list', 'Time, Lat, Lon, comment',
            '--tables_list', ',,tracker{}', ])
# go = True
if st(310):  # False: #
    # Export treckers tracks to GPX tracks
    h5toGpx(['cfg/h5toGpx_nav_all.ini',
             '--db_path', str(path_db_device.with_name(
            path_db_device.stem + '_not_sorted.h5')),
             '--tables_list', f'{device}.*',
             '--table_nav', '',
             '--select_from_tablelog_ranges_index', None,  # Export tracks
             '--gpx_names_fun_format', '{1:%y%m%d}_{0:}',  # track name of format(timeLocal, tblD_safe)
             '--gpx_names_funs_list', '"i, row.Index"',
             '--gpx_names_funs_cobined', ''
             ])
