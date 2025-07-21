import sys
from pathlib import Path
import re

drive_d = 'D:' if sys.platform == 'win32' else '/mnt/D'  # to run on my Linux/Windows systems both
scripts_path = Path(drive_d + '/Work/_Python3/And0K/h5toGrid/scripts')
sys.path.append(str(Path(scripts_path).parent.resolve()))
# my funcs
from utils2init import st
import veuszPropagate
import cfg_dataclasses as cfg_d
from to_vaex_hdf5.nmea2h5 import main as nmea2h5
from to_pandas_hdf5.csv2h5 import main as csv2h5
from to_pandas_hdf5.gpx2h5 import main as gpx2h5
from to_pandas_hdf5.CTD_calc import main as CTD_calc
from h5toGpx import main as h5toGpx
from grid2d_vsz import main as grid2d_vsz

st.go = True   # False #
st.start = 70   # 1 5 30 70 80 115
st.end = 50    # 60 80 120

path_cruise = Path(r'd:\WorkData\BlackSea\220620')
path_db = (path_cruise / path_cruise.name).with_suffix('.h5')  # same name as dir

# Stop before steps that need a manual prepare (70) i.e. set end < 70 at first
# Gridding (last step) needs debugging if interactive filtering is needed
# ---------------------------------------------------------------------------------------------

min_coord = 'Lat:44.3, Lon:37.7'
max_coord = 'Lat:44.5964, Lon:38.3498'  # have slightly bigger spikes

if st(1, 'Save GPX navigation to DB'):
    gpx2h5(['',
            '--db_path', str(path_db),
            '--path', str(path_cruise / '_raw/navigation/*.gpx'),
            '--tables_list', ',navigation,',  # skip waypoints
            '--table_prefix', r'',
            # '--min_date', '2019-07-17T14:00:00',
            '--min_dict', f'{min_coord}',  # use at least -32768 to replace it by NaN
            '--max_dict', f'{max_coord}',
            '--corr_time_mode', 'False',  # 'delete_inversions',
            # '--b_incremental_update', '0',  # '1' coerce to delete data loaded in same table in previous steps
            '--b_interact', '0',
            ])
if st(5, 'Save NMEA navigation to DB'):
    # Save navigation to DB
    path_raw_pattern = str(path_cruise / r'_raw\ADCP_600kHz\nav_NMEA\*000n.0*')

    cfg_d.main_call([
        'input.time_interval=[2022-06-20T00:00, 2022-06-28T00:00]',
        'input.dt_from_utc_hours=0',
        # 'process.period_tracks=1D',
        'input.path="{}"'.format(path_raw_pattern.replace('\\', '/')),
        'out.db_path="{}"'.format(str(path_db).replace('\\', '/')),
        'out.tables=[navigation]',
        'out.b_remove_duplicates=True',
        'out.b_insert_separator=False',
        '+filter.min={{{}, DepEcho:3}}'.format(min_coord),
        '+filter.max={{{}}}'.format(max_coord),
        ], nmea2h5)

if st(6, 'Extract all navigation tracks to check manually: usually bad GPS data near V.V.Putin`s residence as hi blocks signal'):
    # sys.argv[0]= argv0   os_path.join(os_path.dirname(file_h5toGpx)
    h5toGpx(['cfg/h5toGpx_nav_all.ini',
             '--db_path', str(path_db),
             '--tables_list', 'navigation',
             # '--simplify_tracks_error_m_float', '10',
             '--period_files', 'D',
             '--tables_log_list', '""'
             # '--select_from_tablelog_ranges_index', None - default
             ])


device = 'CTD_Idronaut_OS316'
device_veusz_prefix = 'i3_'
common_ctd_params_list = [
    '--db_path', str(path_db),
    '--table', f'{device}',
    '--min_dict', f'Cond:0.5, Sal:0.2',  # O2:-2, O2ppm:-2',  # deletes zeros & strange big negative values  # SigmaT:2,
    # '--max_dict', f'O2:200, O2ppm:20',  #, {max_coord} for REDAS-like data
    ]

device_params_dict = {
    'in': {
        #'fun_proc_loaded': loaded_corr,
        'csv_specific_param': {
        'Temp90_fun': lambda t68: t68 / 1.00024,
           # -1.925036627169023e-06, 6.577767835930226e-05, 1.000754132707556, -0.014076681292841897], x/1.00024),
        # 'Cond_fun': lambda x: np.polyval([
        #     -5.17193900715981e-6, 0.00052017169295646, 0.99678538638325, 0.089677845676474], x),
        # 'Sal_fun': lambda Cond, Temp90, Pres: gsw.SP_from_C(Cond, Temp90, Pres),
        }
    }
}

if st(10, f'Save {device} data to DB'):
    csv2h5([
        'cfg/csv_CTD_Idronaut.ini',
        '--path', str(path_cruise / fr'_raw' / device / 'CAST*[0-9].txt'),
        '--table', f'{device}',
        '--dt_from_utc_hours', '3',  # '2' '0'
        '--dt_from_utc_seconds', '-550',  # '850'
        '--header',
        # Date	Time	Pres	Temp	Cond	Sal	SoundV
        # [D-M-Y]	[h:m:s]	[dbar]	[°C]	[mS/cm]	[PSU]	[m/s]
        # 20-10-2021 12:32:53.90     2.09   15.044    0.004    0.009 1466.1282
        'date(text),txtT(text),Pres(float),Temp90(float),Cond(float),Sal(float),SoundV',  #,SigmaT(float)
        '--delimiter_chars', '\\ \\',  # ''\s+',
        '--b_interact', '0',
        '--skiprows_integer', '1',
        # '--on_bad_lines', 'warn'
        ] + common_ctd_params_list,
          **device_params_dict
        )


if st(20, 'Extract CTD runs to "logRuns" table, filling it with CTD & nav params'):  # False: #
    # Extracts CTD runs (needed if files are not splitted on runs).
    # Note: Saves extended log needed by pattern used in next step with veuszPropagate()
    # todo: be able provide log with (Lat,Lon) separately
    st.go = () != CTD_calc(['cfg/ctd_calc-find_runs.ini',
        '--db_path', str(path_db),
        '--tables_list', f'{device}',
        #'--table_nav', '',       # uncomment if nav data only in CTD data file
        '--min_samples', '40',  # fs*depth/speed = 200: if fs = 10Hz for depth 20m
        '--min_dp', '15',
        # Followig Not Helped!
        '--dt_between_min_minutes', '5',  # default 1s lead to split when commnication with sonde lost
        # '--b_keep_minmax_of_bad_files', 'True',
        # '--b_incremental_update', 'True', - not works. Delete previous table manually, and from ~not_sorted!

        # '--out.tables_list', '',
        '--b_interact', '0'
        ])


# Note: "map" node with table in hdf5 store of shore polygon is needed to can draw *.vsz
if st(30, f'Draw {device} data profiles'):  # False: #
    from to_pandas_hdf5.h5toh5 import h5.log_names_gen
    import re
    from os import chdir as os_chdir

    cfg_in = {
        'log_row': {},
        'db_path': str(path_db),  # name of hdf5 pandas store where is log table
        #min_time, max_time: datetime, optional, allows range table_log rows
        'table_log': f'/{device}/logRuns', # str: name of log table - table with intervals:

        'pattern_path': path_cruise / device / 'profiles_vsz/000000_0000-0000.vsz'
        }
    f_row2name = lambda r: '{:%y%m%d_%H%M%S}.vsz'.format(r['Index'])
    # It is possible to add exact interval to filename but time after probe is back on surface can be determined only
    # from next row, so we rely on ~pattern_loader.vsz to do it. Even freq=16Hz to determine last time not helps:
    # '_{}s.vsz'.format(round(max(r['rows']/16, (r['DateEnd'] - r['Index'] + pd.Timedelta(300, "s")).total_seconds()))
    pattern_code = cfg_in['pattern_path'].read_bytes()  #encoding='utf-8'

    os_chdir(cfg_in['pattern_path'].parent)
    for filename in h5.log_names_gen(cfg_in, f_row2name):
        path_vsz = cfg_in['pattern_path'].with_name(filename)
        path_vsz.write_bytes(pattern_code)  # re.sub(rb'^([^\n]+)', str_expr, pattern_code, count=1)

    veuszPropagate.main(['cfg/veuszPropagate.ini',
                         '--path', str(cfg_in['pattern_path'].with_name('2'+cfg_in['pattern_path'].name.replace('0', '?')[1:])),  #_*s path_db),
                         '--pattern_path', f"{cfg_in['pattern_path']}_",  # here used to auto get export dir only. must not be not existed file path
                         #'--table_log', f'/{device}/logRuns',
                         #'--add_custom_list', f'{device_veusz_prefix}USE_time_search_runs',  # 'i3_USE_timeRange',
                         # '--add_custom_expressions',
                         # """'[["{log_row[Index]:%Y-%m-%dT%H:%M:%S}", "{log_row[DateEnd]:%Y-%m-%dT%H:%M:%S}"]]'""",
                         # '--export_pages_int_list', '7', # 0  '--b_images_only', 'True'
                         '--b_update_existed', 'True',  # False is default todo: allow "delete_overlapped" time named files
                         '--b_interact', '0',
                         '--b_images_only', 'True',      # mandatory
                         '--b_execute_vsz', 'True'
                         #'--min_time', '2020-07-08T03:35:00',
                         #'--max_time', '2020-06-30T22:37:00',
                         ])

if False:
    # Merge each needed runs
    import pandas as pd
    from to_pandas_hdf5.h5toh5 import h5.move_tables, h5.merge_two_runs  #, h5.index_sort, h5.out_init

    tbl = f'/{device}'
    tbl_log = f'{tbl}/logRuns'
    with pd.HDFStore(path_db) as store:
        #     store = pd.HDFStore(path_db)
        df_log = store[tbl_log]

    # repeat if need:
    irow_to = 130  # 85
    h5.merge_two_runs(df_log, irow_to, irow_from=None)

    # write back
    with pd.HDFStore(path_db.with_name('_not_sorted.h5')) as store_tmp:
        try:
            del store_tmp[tbl_log]
        except KeyError:
            pass
        df_log.to_hdf(store_tmp, tbl_log, append=True, data_columns=True,
                      format='table', dropna=True, index=False)
    h5.move_tables({
        'temp_db_path': path_db.with_name('_not_sorted.h5'),
        'db_path': path_db,
        'tables': [tbl_log],
        'tables_log': [],
        'addargs': ['--checkCSI', '--verbose']
        })

    # Now run step 30 with veuszPropagate seting: '--b_update_existed', 'False' to save only modified vsz/images. After that delete old vsz and its images


if False:  # st(40)  # may not comment always because can not delete same time more than once
    # Deletng bad runs from DB:
    import pandas as pd

    # find bad runs that have time:
    time_in_bad_run_any = ['2018-10-16T19:35:41+00:00']
    tbl = f'/{device}'
    tbl_log = f'{tbl}/logRuns'
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

if st(50, 'Extract navigation data at time station starts to GPX waypoints'):
    h5toGpx([
        'cfg/h5toGpx_CTDs.ini',
        '--db_path', str(path_db),
        '--tables_list', f'{device}',
        '--tables_log_list', 'logRuns',
        '--gpx_names_funs_list', """i+1""",
        '--gpx_names_fun_format', '{:02d}',  # '{:03d}'
        '--select_from_tablelog_ranges_index', '0',
        '--dt_search_nav_tolerance_minutes', '1'  # to trigger interpolate
        ])
    print('Hey! Save _manually_ prepared routes navigation\CTD-sections=routes.gpx before continue from st.start = 70!')
    st.go = False

if False:  # st(60, 'Extract navigation data at runs/starts to GPX tracks.'):    # Useful to indicate where no nav?
    h5toGpx([
        'cfg/h5toGpx_CTDs.ini',
        '--db_path', str(path_db),
        '--tables_list', f'{device}',
        '--tables_log_list', 'logRuns',
        '--select_from_tablelog_ranges_index', None,  # Export tracks
        '--gpx_names_fun_format', '{1:%y%m%d}_{0:}',  # track name of format(timeLocal, tblD_safe)
        '--gpx_names_funs_list', '"i, row.Index"',
        '--gpx_names_funs_cobined', ''
        ])
    print('Hey! Save _manually_ prepared routes navigation\CTD-sections=routes.gpx before continue from st.start = 70!')
    st.go = False

if st(70, 'Save waypoints/routes from _manually_ prepared gpx to hdf5'):  # False: #
    gpx2h5(['', '--path', str(path_cruise / r'navigation\CTD-sections=routes.gpx'),
            '--table_prefix', r'navigation/sectionsCTD'])  # need copy result from navigation\{path_db}_not_sorted manually, todo: auto copy

if st(80, 'Gridding Zabor data'):  # and False: #
    # Note: Prepare veusz "zabor" pattern before
    grid2d_vsz(['cfg/grid2d_vsz.ini', '--db_path', str(path_db),
                '--table_sections', r'navigation/sectionsCTD_routes',
                '--subdir', 'CTD-sections',
                '--begin_from_section_int', '1', #'1',  # values <= 1 means no skip
                '--data_columns_list', f"Temp, Sal, SigmaTh, soundV",  # , ChlA only for section < 5
                # 'Eh, pH',  todo: N^2 - need calc before
                '--max_depth', '3000',
                '--filter_depth_wavelet_level_int', '2',  # 4, 5, 6
                '--convexing_ctd_bot_edge_max', '40',  # set < bottom because it is harder to recover than delete
                # '--x_resolution', '0.2',
                # '--y_resolution', '5',
                '--dt_search_nav_tolerance_seconds', '120',
                '--symbols_in_veusz_ctd_order_list',
                "'Triangle, Green', 'Diamond, Blue', 'Triangle, Red', 'Square, Green'",
                '--b_temp_on_its90', 'True',  # modern probes
                '--blank_level_under_bot', '-500',
                # '--interact', 'False',
                '--b_reexport_images', 'True'
                ])

    # todo: bug: bad top and bottom edges

if st(110, 'Export csv with some new calculated parameters'):  # False: #
    CTD_calc([  # 'ctd_calc-find_runs.ini',
        '--db_path', str(path_db),
        '--tables_list', f'{device}',
        '--tables_log', '{}/logRuns',
        # '--min_samples', '99',  # fs*depth/speed = 200: if fs = 10Hz for depth 20m
        # '--min_dp', '9',
        # '--b_keep_minmax_of_bad_files', 'True',
        '--path_csv', str(path_cruise / device / 'txt_processed'),
        '--data_columns_list', 'Pres, Temp90, Cond, Sal, O2, O2ppm, SA, sigma0, depth, soundV',  #, pH, Eh  , Lat, Lon
        '--b_incremental_update', 'True',
        # todo: check it. If False need delete all previous result of ctd_calc() or set min_time > its last log time
        '--out.tables_list', 'None',
        ])

if st(115, 'Export csv for Obninsk'):
    m = re.match(r'[\d_]*(?P<abbr_cruise>[^\d]*)(?P<i_cruise>.*)', path_cruise.name)
    i_cruise = int(m.group('i_cruise'))
    text_file_name_add = f"E090005O2_{m.group('abbr_cruise')}_{i_cruise}_H10_"

    from to_vaex_hdf5.h5tocsv import main_call as h5tocsv
    h5tocsv([
        f'input.db_path="{path_db}"',
        f'input.tables=["{device}"]',
        f'input.tables_log=["{device}/logRuns"]',
        fr"out.text_path='{path_cruise / device / 'txt_for_Obninsk'}'",
        f'out.text_date_format="%Y-%m-%dT%H:%M:%S"',
        f'out.text_float_format="%.6g"',
        f'out.file_name_fun="f\'{text_file_name_add}{{i+1:0>2}}.csv\'"',
        f'out.file_name_fun_log="\'{text_file_name_add}POS.csv\'"',
        # rec_num;identific;station;Pres;Temp;cond;Sal;O2%;O2ppm;sigmaT;soundV
        f'+out.cols_log={{rec_num: "i + 1", identific: "i + 1", station: "{i_cruise * 1000 + 1} + i" , LONG: Lon_st, LAT: Lat_st, DATE: index}}',
        ''.join([
            f'+out.cols={{rec_num: "i + 1", identific: "@i_log + 1", station: "{i_cruise * 1000 + 1} + @i_log", ',
            ', '.join([p if ':' in p else f'{p}: {p}' for p in
                       'Pres;Temp:Temp90;Cond;Sal;O2;O2ppm'.split(';')]), #;SigmaT;SoundVel
            '}'
            ]),
        'out.sep=";"'
        ])


if st(120, 'Meteo'):
    csv2h5([
        'cfg/csv_meteo.ini', '--path',  # to_pandas_hdf5/
        str(path_cruise / r"meteo\ship's_meteo_st_source\*.mxt"), '--header',
        'date(text),Time(text),t_air,Vabs_m__s,Vdir,dew_point,Patm,humidity,t_w,precipitation',
        '--coldate_integer', '0', '--coltime_integer', '1',
        '--cols_not_save_list', 't_w,precipitation',  # bad constant data
        '--delimiter_chars', ',', '--max_text_width', '12',
        '--on_bad_lines', 'warn', '--b_insert_separator', 'False',
        '--chunksize_percent_float', '500',
        '--fs_float', '60',
        '--skiprows', '0'
        ])

if st(130, 'extract all navigation tracks'):
    # sys.argv[0]= argv0   os_path.join(os_path.dirname(file_h5toGpx)
    h5toGpx(['cfg/h5toGpx_nav_all.ini',
             '--db_path', str(path_db),
             '--tables_list', 'navigation',
             '--simplify_tracks_error_m_float', '10',
             '--period_files', 'D',
             '--tables_log_list', '""'
             # '--select_from_tablelog_ranges_index', None - defaut
             ])

device_prev = device
device = 'CTD_SST_CTD90'
device_veusz_prefix = 'ss_'

common_ctd_params_list = [
    '--db_path', str(path_db),
    '--min_dict', f'Sal:0.2',
    ]

if st(210, f'Save {device} data to DB'):
    # IntD        IntT      Press     Temp    SALIN    SIGMA     Turb    SOUND
    from to_pandas_hdf5.csv_specific_proc import loaded_sst

    csv2h5([
        'cfg/csv_CTD_SST.ini',
        # '--skiprows_integer', '34', # default
        '--path', str(path_cruise / device / '_raw_csv' / 'АБП*[0-9].CSV'),
        # '--dt_from_utc_hours', '0',
        '--header', 'Date(text),Time(text),Pres,Temp90,Sal,SIGMA,Turb,SVel',
        '--cols_not_save_list', 'SIGMA,SVel',
        '--delimiter_chars', ',',  # ''\s+',
        '--table', f'{device}',
        '--b_interact', '0'
        # '--on_bad_lines', 'warn',
        ] + common_ctd_params_list,
        **{'in': {
            'fun_proc_loaded': loaded_sst,
            # 'csv_specific_param': {'Temp_fun': lambda x: (x + 0.254) / 1.00024,
            #                        # 'Temp_add': 0.254, And convert to ITS90
            #                        'Sal_fun': lambda x: (1 + 0.032204423446495364) * x + 0.045516504802752523,
            #                        'Cond_fun': lambda x: -0.000098593 * x ** 2 + 1.040626 * x + 0.01386
            #                        }
            }}
        )

if st(220, 'Extract CTD runs to "logRuns" table, filling it with CTD & nav params'):  # False: # (if files are not splitted on runs).
    # Note: extended logRuns fields needed in Veusz in next step
    # todo: be able provide log with (Lat,Lon) separately, improve start message if calc runs, check interpolation
    st.go = () != CTD_calc(['cfg/ctd_calc-find_runs.ini',
              '--db_path', str(path_db),
              '--tables_list', f'{device}',
              '--min_samples', '100',  # fs*depth/speed = 200: if fs = 10Hz for depth 20m
              '--min_dp', '20',  # todo: <=25
              '--dt_between_min_minutes', '5',
              # '--b_keep_minmax_of_bad_files', 'True',
              # '--b_incremental_update', 'True', - not works. Delete previous table manually, and from ~not_sorted!

              # '--out.tables_list', '',
              '--b_interact', '0'
              ])

if st(230, f'Draw {device} data profiles'):  # False: #
    from to_pandas_hdf5.h5toh5 import h5.log_names_gen
    import re
    from os import chdir as os_chdir

    cfg_in = {
        'log_row': {},
        'db_path': str(path_db), # name of hdf5 pandas store where is log table
        #min_time, max_time: datetime, optional, allows range table_log rows
        'table_log': f'/{device}/logRuns', # str: name of log table - table with intervals:

        'pattern_path': path_cruise / device / '000000_0000-0000.vsz'
        }
    f_row2name = lambda r: '{:%y%m%d_%H%M%S}.vsz'.format(r['Index'])
    # It is possible to add exact interval to filename but time after probe is back on surface can be determined only
    # from next row, so we rely on ~pattern_loader.vsz to do it. Even freq=16Hz to determine last time not helps:
    # '_{}s.vsz'.format(round(max(r['rows']/16, (r['DateEnd'] - r['Index'] + pd.Timedelta(300, "s")).total_seconds()))
    pattern_code = cfg_in['pattern_path'].read_bytes()  #encoding='utf-8'

    os_chdir(cfg_in['pattern_path'].parent)
    for filename in h5.log_names_gen(cfg_in, f_row2name):
        path_vsz = cfg_in['pattern_path'].with_name(filename)
        path_vsz.write_bytes(pattern_code)  # re.sub(rb'^([^\n]+)', str_expr, pattern_code, count=1)

    veuszPropagate.main(['cfg/veuszPropagate.ini',
                         '--path', str(cfg_in['pattern_path'].with_name('??????_??????.vsz')),  #_*s path_db),
                         '--pattern_path', f"{cfg_in['pattern_path']}_",  # here used to auto get export dir only. must not be not existed file path
                         #'--table_log', f'/{device}/logRuns',
                         #'--add_custom_list', f'{device_veusz_prefix}USE_time_search_runs',  # 'i3_USE_timeRange',
                         # '--add_custom_expressions',
                         # """'[["{log_row[Index]:%Y-%m-%dT%H:%M:%S}", "{log_row[DateEnd]:%Y-%m-%dT%H:%M:%S}"]]'""",
                         # '--export_pages_int_list', '7', # 0  '--b_images_only', 'True'
                         # '--b_update_existed', 'True',  # False is default todo: allow "delete_overlapped" time named files
                         '--b_interact', '0',
                         '--b_images_only', 'True',      # mandatory
                         '--b_execute_vsz', 'True'
                         #'--min_time', '2020-07-08T03:35:00',
                         #'--max_time', '2020-06-30T22:37:00',
                         ])

if st(250, 'Extract navigation data at time station starts to GPX waypoints'):  # False: #
    h5toGpx([
    'cfg/h5toGpx_CTDs.ini',
    '--db_path', str(path_db),
    '--tables_list', f'{device_prev}, {device}',
    '--gpx_symbols_list', "'Diamond, Blue', 'Triangle, Red'",
    '--tables_log_list', 'logRuns',
    '--gpx_names_funs_list', """i+1""",
    '--gpx_names_fun_format', '{:02d}',
    '--select_from_tablelog_ranges_index', '0',
    '--dt_search_nav_tolerance_minutes', '1'  # to trigger interpolate
    ])

if st(270, 'Save waypoints/routes from _manually_ prepared gpx to hdf5'):  # False: #
    gpx2h5(['', '--path', str(path_cruise / fr'navigation\CTD-sections=routes_{device}.gpx'),
            '--table_prefix', fr'navigation/sectionsCTD_{device}'])  # need copy result from navigation\{path_db}_not_sorted manually, todo: auto copy

if st(280, 'Gridding'):  # and False: #
    # Note: Prepare veusz "zabor" pattern before
    grid2d_vsz(['cfg/grid2d_vsz.ini', '--db_path', str(path_db),
                '--table_sections', fr'navigation/sections_{device}_routes',
                '--subdir', 'CTD-sections',
                '--begin_from_section_int', '1', #'1',  # values <= 1 means no skip
                '--data_columns_list', "Turb, Temp, Sal, SigmaTh, soundV", #O2, O2ppm,
                # 'Eh, pH',  todo: N^2 - need calc before
                '--max_depth', '250', #'250',
                '--filter_depth_wavelet_level_int', '5',  # 4, 5, 5, 4, 6, 4, 4, 5
                '--convexing_ctd_bot_edge_max', '40',  # set < bottom because it is harder to recover than delete
                # '--x_resolution', '0.2',
                # '--y_resolution', '5',
                '--dt_search_nav_tolerance_seconds', '120',
                # '--symbols_in_veusz_ctd_order_list',
                # "'Triangle, Green', 'Diamond, Blue', 'Triangle, Red', 'Square, Green'",
                '--b_temp_on_its90', 'True',  # modern probes
                '--blank_level_under_bot', '-220',
                '--symbols_in_veusz_ctd_order_list', "'Triangle, Red', "
                # '--interact', 'False',
                #'--b_reexport_images', 'True'
                ])

if st(290, 'Export csv with some new calculated parameters'):  # False: #
    # Extract CTD runs (if files are not splitted on runs):
    CTD_calc([  # 'ctd_calc-find_runs.ini',
        '--db_path', str(path_db),
        '--tables_list', f'{device}',
        '--tables_log', '{}/logRuns',
        # '--min_samples', '99',  # fs*depth/speed = 200: if fs = 10Hz for depth 20m
        # '--min_dp', '9',
        # '--b_keep_minmax_of_bad_files', 'True',
        '--path_csv', str(path_cruise / device / 'txt_processed'),
        '--data_columns_list', 'Pres, Temp90, Cond, Sal, O2, O2ppm, Lat, Lon, SA, sigma0, depth, soundV',  #
        '--b_incremental_update', 'True',
        # todo: check it. If False need delete all previous result of ctd_calc() or set min_time > its last log time
        '--out.tables_list', 'None',
        ])

if st(315, 'Export csv for Obninsk'):
    m = re.match(r'[\d_]*(?P<abbr_cruise>[^\d]*)(?P<i_cruise>.*)', path_cruise.name)
    i_cruise = int(m.group('i_cruise'))
    text_file_name_add = f"E090005O2_{m.group('abbr_cruise')}_{i_cruise}_H10_"

    from to_vaex_hdf5.h5tocsv import main_call as h5tocsv
    h5tocsv([
        f'input.db_path="{path_db}"',
        f'input.tables=["{device}"]',
        f'input.tables_log=["{device}/logRuns"]',
        fr"out.text_path='{path_cruise / device / 'txt_for_Obninsk'}'",
        f'out.text_date_format="%Y-%m-%dT%H:%M:%S"',
        f'out.text_float_format="%.6g"',
        f'out.file_name_fun="f\'{text_file_name_add}{{i+1:0>2}}.csv\'"',
        f'out.file_name_fun_log="\'{text_file_name_add}POS.csv\'"',
        # rec_num;identific;station;Pres;Temp;cond;Sal;O2%;O2ppm;sigmaT;soundV
        f'+out.cols_log={{rec_num: "i + 1", identific: "i + 1", station: "{i_cruise * 1000 + 1} + i" , LONG: Lon_st, LAT: Lat_st, DATE: index}}',
        ''.join([
            f'+out.cols={{rec_num: "i + 1", identific: "@i_log + 1", station: "{i_cruise * 1000 + 1} + @i_log", ',
            ', '.join([p if ':' in p else f'{p}: {p}' for p in
                       'Pres;Temp:Temp90;Sal;Turb'.split(';')]), #;SigmaT;SoundVel;O2;O2ppm
            '}'
            ]),
        'out.sep=";"'
        ])
