import sys
from os import chdir as os_chdir
from pathlib import Path
import re
import numpy as np
import pandas as pd
import gsw
drive_d = 'D:' if sys.platform == 'win32' else '/mnt/D'  # to run on my Linux/Windows systems both
scripts_path = Path(drive_d + '/Work/_Python3/And0K/h5toGrid/scripts')
sys.path.append(str(Path(scripts_path).parent.resolve()))
from itertools import takewhile
# my funcs
from utils2init import st, pairwise
import veuszPropagate
from to_pandas_hdf5.csv2h5 import main as csv2h5
from to_pandas_hdf5.gpx2h5 import main as gpx2h5
from to_pandas_hdf5.CTD_calc import main as CTD_calc
# from to_pandas_hdf5.csv_specific_proc import loaded_corr
from h5toGpx import main as h5toGpx
from grid2d_vsz import main as grid2d_vsz
from to_pandas_hdf5.h5toh5 import h5.log_names_gen

st.go = True   # False #
st.start = 80   # 1 5 30 70 80 115
st.end = 115   # 60 80 120

path_cruise = Path(r'd:\WorkData\BalticSea\221116_AMK91')
path_db = path_cruise / path_cruise.with_suffix('.h5').name  # same name as dir

# Stop before steps that need a manual prepare (70) i.e. set end < 70 at first
# Gridding (last step) needs debugging if interactive filtering is needed
# ---------------------------------------------------------------------------------------------

min_coord = 'Lat:53, Lon:18.6'  # 10
max_coord = 'Lat:60.55, Lon:30.3'  # includes Gulf Of Finland
if st(1, 'Save gpx navigation to DB'):
    # Save navigation to DB
    for folder in (['_raw_gpx/_combined_used']):  # notebook_MSI, 'OpenCPN_Trimble(aft)', 'OpenCPN_Trimble(foredeck)']):
        gpx2h5(['',
            '--db_path', str(path_db),
            '--path', str(path_cruise / 'navigation' / folder / '*22*.gpx'),
            '--tables_list', ',navigation,',  # skip waypoints
            '--table_prefix', r'',
            # '--b_search_in_subdirs', if set True (to get rid of this loop) then will be problems with overlapped data files
            # '--min_date', '2019-07-17T14:00:00',
            '--min_dict', f'{min_coord}',  # use at least -32768 to replace it by NaN
            '--max_dict', f'{max_coord}',
            '--corr_time_mode', 'False', #'delete_inversions?',
            # '--b_incremental_update', '0',  # '1' coerce to delete data loaded in same table in previous steps (only if previous same log file detected?)
            '--b_interact', '0',
            # '--b_remove_duplicates', '1',  # not allowed and not need: does always
            ])


if st(2, "Save bathymetry to DB from *.xyz export of EA600 echosounder data"):
    # (saved gpx data is sparse and coincidence of time samples is seldom, but need to check and delete duplicates)
    csv2h5([
        'cfg/csv_nav_HYPACK_EA600.yml',  # We think now that it is not HYPACK specific format, but we not change name
        '--db_path', str(path_db),
        '--path', str(path_cruise / r'navigation/EA600-12kHz' / 'L*.xyz'),
        # '--dt_from_utc_hours', '2',
        # '--dt_from_utc_seconds', '838',        # will be subtracted: 13:28:05(UTC) - 13:42:03
        '--b_incremental_update', '0',          # needed to keep overlapped data from other data sources
        '--min_dict', f'{min_coord}, DepEcho:24',  # 3 # use at least -32768 to replace it by NaN
        '--max_dict', f'{max_coord}, DepEcho:125',  # 300
        '--min_date', '2022-11-01',  # use some value to filter NaTs
        '--corr_time_mode', 'False',  # 'delete_inversions?',
        # '--b_incremental_update', '0',  # '1' coerce to delete data loaded in same table in previous steps (only if previous same log file detected?)
        '--b_interact', '0'],
         **{'in': {
        'csv_specific_param': {
        #     'DepEcho_fun': lambda x: np.polyval(         # approximation for (Sound speed?) correction
        #         [0.0004000000018551386, 0.9362000000550506, 0.600399992957307], x
        #         )
        #     }  # '--fun_DepEcho', f'isfinite',
            'DepEcho_add': -3,
        }}}
        )


device = 'CTD_SST_48Mc#1253'
# device_veusz_prefix = 'ss_'
common_ctd_params_list = [
    '--db_path', str(path_db),
    '--table', f'{device}',
    '--min_dict', f'Cond:0.5, Sal:0.2, O2:-2, O2ppm:-2',  # deletes zeros & strange big negative values  # SigmaT:2,
    '--max_dict', f'O2:200, O2ppm:20',  #, {max_coord} for REDAS-like data
]
device_params_dict = \
    {
     }

if st(10, f'Save {device} data to DB'):
    from to_pandas_hdf5.csv_specific_proc import loaded_sst
    csv2h5([
        'cfg/csv_CTD_SST.ini',
        '--path', str(path_cruise / device / '_raw' / '22[01][012][0123][0-9]_*.TOB'),
        '--table', f'{device}',
        #'--dt_from_utc_hours', '0', #'2'
        '--header',
        'Number,Date(text),Time(text),Pres,Temp,Sal,O2,O2ppm,SIGMA,Cond,Vbatt,SVel',
        '--cols_not_save_list', 'Number,SIGMA,Vbatt,SVel',
        '--delimiter_chars', '\\ \\',  # ''\s+',
        '--b_interact', '0',
        #'--cols_not_save_list', 'N',
        # '--on_bad_lines', 'warn'
        #'--min_dict', 'O2:0, O2ppm:0',  # replace strange values
        ] + common_ctd_params_list,
        **{  # **device_params_dict,
            'in': {
                'fun_proc_loaded': loaded_sst,
                'csv_specific_param': {
                    'Temp_fun': lambda x: np.polyval([
                        -1.2268433823676e-6, 4.91808363884267e-5, 0.99938539313664, 0.0074516860934196], x),
                    'Cond_fun': lambda x: np.polyval([
                        -7.57195308190065e-7, 3.80941696689889e-5, 1.0018619893672, -0.034845948296864], x),
                    'Sal_fun': lambda Cond, Temp, Pres: gsw.SP_from_C(Cond, Temp, Pres),
                    'O2_fun': lambda x: (x - 3.04733)*100/106.63753,           # _add': -3.477,
                    'O2ppm_fun': lambda x: (x - 0.272)*10.2745973/10.3573,     # _add': -0.39
                    }
                }
        }
    )


if st(20, 'Extract CTD runs to "logRuns" table, filling it with CTD & nav params'):  # False: #
    # Extracts CTD runs (needed if files are not splitted on runs).
    # Note: Saves extended log needed by pattern used in next step with veuszPropagate()
    # todo: be able provide log with (Lat,Lon) separately
    st.go = () != CTD_calc(['cfg/ctd_calc-find_runs.ini',
        '--db_path', str(path_db),
        '--tables_list', f'{device}',
        #'--table_nav', '',       # uncomment if nav data only in CTD data file
        '--min_samples', '100',  # fs*depth/speed = 200: if fs = 10Hz for depth 20m
        '--min_dp', '10',
        # Following Not Helped!
        '--dt_between_min_minutes', '5',  # default 1s lead to split when communication with sonde lost
        '--b_keep_minmax_of_bad_files', 'True',
        # '--b_incremental_update', 'True', - not works. Delete previous table manually, and from ~not_sorted!

        # '--out.tables_list', '',
        '--b_interact', '0'
        ])


# Usually not needed step! #
b_update = True  # False:  #  # if False may not skip because can not delete same rows more than once
if False and st(25, 'Values correction (updating DB)' if b_update else f'Deleting bad runs from DB'):
    from h5_cor import main as h5_cor_main
    h5_cor_main(path_db, device)
    if False:  # for debug:
        from h5_cor import h5cor
        h5cor(time_ranges, edges_sources, b_update=True,
                  cfg_out=cfg_out, coef_for_interval=coef_for_interval
                  )
    st.go = (False, 'Hey! logRuns table removed for correction! Recalculate bot/top parameters: set st.start = 20, comment this step and run!')

if st(30, f'Draw {device} data profiles'):  # False: #
    # Note: if vsz pattern uses map from *.h5, then be sure that it exists
    cfg_in = {
        'log_row': {},
        'db_path': str(path_db), # name of hdf5 pandas store where is log table
        'table_log': f'/{device}/logRuns', # str: name of log table - table with intervals:
        'pattern_path': path_cruise / device / 'profiles_vsz' / '000000_000000.vsz',
        'min_time': np.datetime64('2022-12-21T10:02:00'),
        # 'max_time': '2020-12-30T22:37:00',
        }
    f_row2name = lambda r: '{:%y%m%d_%H%M%S}.vsz'.format(r['Index'])
    # It is possible to add exact interval to filename but time after probe is back on surface can be determined only
    # from next row, so we rely on ~pattern_loader.vsz to do it. Even freq=16Hz to determine last time not helps:
    # '_{}s.vsz'.format(round(max(r['rows']/16, (r['DateEnd'] - r['Index'] + pd.Timedelta(300, "s")).total_seconds()))
    pattern_code = cfg_in['pattern_path'].read_bytes()  # encoding='utf-8'
    filename_st = None
    os_chdir(cfg_in['pattern_path'].parent)
    for filename in h5.log_names_gen(cfg_in, f_row2name):
        path_vsz = cfg_in['pattern_path'].with_name(filename)
        path_vsz.write_bytes(pattern_code)  # re.sub(rb'^([^\n]+)', str_expr, pattern_code, count=1)
        # Get filename_st (do once)
        if filename_st is None:
            filename_st = filename
    # cfg_in['min_time'] not works on filenames, so we convert it to 'start_file_index'
    if 'min_time' in cfg_in:
        del cfg_in['min_time']  # del to count fro 0:
        start_file_index = len(list(takewhile(lambda x: x < filename_st, h5.log_names_gen(cfg_in, f_row2name))))
    else:
        start_file_index = 0
    veuszPropagate.main([
        'cfg/veuszPropagate.ini',
        '--path', str(cfg_in['pattern_path'].with_name('??????_??????.vsz')),  #_*s path_db),
        '--pattern_path', f"{cfg_in['pattern_path']}_",  # here used to auto get export dir only. must not be not existed file path
        #'--table_log', f'/{device}/logRuns',
        #'--add_custom_list', f'{device_veusz_prefix}USE_time_search_runs',  # 'i3_USE_timeRange',
        # '--add_custom_expressions',
        # """'[["{log_row[Index]:%Y-%m-%dT%H:%M:%S}", "{log_row[DateEnd]:%Y-%m-%dT%H:%M:%S}"]]'""",
        # '--export_pages_int_list', '7', # 0  '--b_images_only', 'True'
        '--b_update_existed', 'True',  # False is default todo: allow "delete_overlapped" time named files
        '--b_interact', '0',
        '--b_images_only', 'True',      # mandatory
        '--b_execute_vsz', 'True',
        '--start_file', str(start_file_index),
        #'--min_time', cfg_in['min_time'].item().isoformat(),  # not works on filenames (no time data)
        #'--max_time', cfg_in['max_time'].item().isoformat(),
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

file_tracks = 'CTD-sections=routes.gpx'
if st(50, 'Extract navigation data at time station starts to GPX waypoints'):  # False: #
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
    st.go = (False, f'Hey! Prepare gpx tracks ({file_tracks}) from waypoints _manually_ before continue and rerun from st.start = 70!')

if False:  # st(60, 'Extract navigation data at runs/starts to GPX tracks.'):    # Extract     # Useful to indicate where no nav?
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
    st.go = (False, 'Hey! Prepare gpx tracks _manually_ before continue (rerun from st.start = 70)!')


if st(70, f'Save waypoints/routes from _manually_ prepared {file_tracks} to hdf5'):  # False: #
    gpx2h5(['', '--path', str(path_cruise / rf'navigation\{file_tracks}'),
            '--table_prefix', r'navigation/sectionsCTD',
            '--b_sort', 'False'])  # need copy result from navigation\{path_db} manually, todo: auto copy
    st.go = (False, f'Hey! copy result from navigation/{path_db} _manually_ before continue (rerun from st.start = 80)!')

if st(80, 'Gridding'):  # and False: #
    # Note: Prepare veusz "zabor" pattern before
    grid2d_vsz(['cfg/grid2d_vsz.ini', '--db_path', str(path_db),
                '--table_sections', r'navigation/sectionsCTD_routes',
                '--subdir', 'CTD-sections',

                '--begin_from_section_int', '4', #'1',  # values <= 1 means no skip
                '--data_columns_list', "Temp, Sal, SigmaTh, O2, O2ppm, soundV",
                # 'Eh, pH',  todo: N^2 - need calc before
                '--max_depth', '150',  # '250',
                '--filter_depth_wavelet_level_int', '7',  # 4, 5, 5, 4, 6, 4, 4, 5
                '--convexing_ctd_bot_edge_max', '25',  # 40 set < bottom because it is harder to recover than delete
                # '--x_resolution', '0.2',
                # '--y_resolution', '5',
                '--depecho_add_float', '-1.5',
                '--dt_search_nav_tolerance_seconds', '120',
                '--symbols_in_veusz_ctd_order_list',
                "'Triangle, Green', 'Diamond, Blue', 'Triangle, Red', 'Square, Green'",
                '--b_temp_on_its90', 'True',  # modern probes
                '--blank_level_under_bot', '-220',
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
        '--data_columns_list', 'Pres, Temp, Cond, Sal, O2, O2ppm, SA, sigma0, depth, soundV',  #, pH, Eh  , Lat, Lon
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
        f'+out.cols_log={{rec_num: "@i + 1", identific: "@i + 1", station: "{i_cruise * 1000 + 1} + @i" , LONG: Lon_st, LAT: Lat_st, DATE: index}}',
        ''.join([
            f'+out.cols={{rec_num: "@i + 1", identific: "@i_log + 1", station: "{i_cruise * 1000 + 1} + @i_log", ',
            ', '.join([p if ':' in p else f'{p}: {p}' for p in
                       'Pres;Temp;Cond;Sal;O2;O2ppm'.split(';')]), #Temp:Temp90;SigmaT;SoundVel
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
device = 'CTD_SST_MWS#3613'
device_folder = 'CTD_SST_MWS#3613(Rozeta)'
device_veusz_prefix = 'ss_'

common_ctd_params_list = [
    '--db_path', str(path_db),
    '--min_dict', f'Sal:0.2',
    ]

if st(210, f'Save {device} data to DB'):
    # Time [hh:mm:ss]	Bottle []	Pressure [dbar]	Temperature [°C]	Conductivity [mS/cm]	Salinity [PSU]
    #                                               Sound Vel. [m/s]	Density [kg/mі]	Spec. Cond. [mS/cm]	Comments
    # 08:35:22	0	-0.2	9.158	-0.006	0.000	1443.90	999.77	-0.008
    # 08:35:25	0	-0.2	9.814	6.578	5.205	1452.98	1003.79	9.447
    from to_pandas_hdf5.csv_specific_proc import loaded_sst_mws

    csv2h5([
        #'cfg/csv_CTD_SST.ini',
        '--skiprows_integer', '40',
        '--path', str(path_cruise / device_folder / '_raw_txt' / '54*.txt'),
        '--dt_from_utc_hours', '2',
        '--header', 'Time(text),Bottle,Pres,Temp,Cond,Sal,SVel,Dens,SpCond,Comments',
        '--cols_not_save_list', 'Bottle,SVel,Dens,SpCond,Comments',
        '--delimiter_chars', r'\t',  # ''\s+',
        '--table', f'{device}',
        '--b_interact', '0'
        # '--on_bad_lines', 'warn',
        ] + common_ctd_params_list,
        **{'in': {
            # 'csv_specific_param': {'Temp_fun': lambda x: (x + 0.254) / 1.00024,
            #                        # 'Temp_add': 0.254, And convert to ITS90
            #                        'Sal_fun': lambda x: (1 + 0.032204423446495364) * x + 0.045516504802752523,
            #                        'Cond_fun': lambda x: -0.000098593 * x ** 2 + 1.040626 * x + 0.01386
            #                        }
            'fun_proc_loaded': loaded_sst_mws,
            'csv_specific_param': {
                'Temp_fun': lambda x: np.polyval([
                    - 1.49640674355499e-8, 2.73759658836018e-6, -8.36587113499398e-5, 1.0006301100888,
                    0.00089533857713988], x),
                'Cond_fun': lambda x: np.polyval([
                    -3.78808059923396e-6, 0.00025187456004893, 1.0054667814625, 0.0064369578275656], x),
                'Sal_fun': lambda Cond, Temp, Pres: gsw.SP_from_C(Cond, Temp, Pres),
                }
            }}
        )

if st(220, 'Extract CTD runs to "logRuns" table, filling it with CTD & nav params'):  # False: # (if files are not splitted on runs).
    # Note: extended logRuns fields needed in Veusz in next step
    # todo: be able provide log with (Lat,Lon) separately, improve start message if calc runs, check interpolation
    st.go = () != CTD_calc(['cfg/ctd_calc-find_runs.ini',
              '--db_path', str(path_db),
              '--tables_list', f'{device}',
              '--min_samples', '20',  # fs*depth/speed = 200: if fs = 10Hz for depth 20m
              '--min_dp', '5',  # todo: <=25
              '--dt_between_min_minutes', '5',
              '--b_keep_minmax_of_bad_files', 'True',
              # '--b_incremental_update', 'True', - not works. Delete previous table manually, and from ~not_sorted!

              # '--out.tables_list', '',
              '--b_interact', '0'
              ])

if st(230, f'Draw {device} data profiles'):  # False: #
    # Note: if vsz pattern uses map from *.h5, then be sure that it exists
    cfg_in = {
        'log_row': {},
        'db_path': str(path_db), # name of hdf5 pandas store where is log table
        'table_log': f'/{device}/logRuns', # str: name of log table - table with intervals:
        'pattern_path': path_cruise / device_folder / 'profiles_vsz' / '000000_000000.vsz',
        # 'min_time': np.datetime64('2022-11-04T22:00:00'),
        # 'max_time': '2020-06-30T22:37:00',
        }
    f_row2name = lambda r: '{:%y%m%d_%H%M%S}.vsz'.format(r['Index'])
    # It is possible to add exact interval to filename but time after probe is back on surface can be determined only
    # from next row, so we rely on ~pattern_loader.vsz to do it. Even freq=16Hz to determine last time not helps:
    # '_{}s.vsz'.format(round(max(r['rows']/16, (r['DateEnd'] - r['Index'] + pd.Timedelta(300, "s")).total_seconds()))
    pattern_code = cfg_in['pattern_path'].read_bytes()  # encoding='utf-8'
    filename_st = None
    os_chdir(cfg_in['pattern_path'].parent)
    for filename in h5.log_names_gen(cfg_in, f_row2name):
        path_vsz = cfg_in['pattern_path'].with_name(filename)
        path_vsz.write_bytes(pattern_code)  # re.sub(rb'^([^\n]+)', str_expr, pattern_code, count=1)
        # Get filename_st (do once)
        if filename_st is None:
            filename_st = filename

    veuszPropagate.main([
        'cfg/veuszPropagate.ini',
        '--path', str(cfg_in['pattern_path'].with_name('??????_??????.vsz')),  #_*s path_db),
        '--pattern_path', f"{cfg_in['pattern_path']}_",  # here used to auto get export dir only. must not be not existed file path
        # '--table_log', f'/{device}/logRuns',
        # '--add_custom_list', f'{device_veusz_prefix}USE_time_search_runs',  # 'i3_USE_timeRange',
        # '--add_custom_expressions',
        # """'[["{log_row[Index]:%Y-%m-%dT%H:%M:%S}", "{log_row[DateEnd]:%Y-%m-%dT%H:%M:%S}"]]'""",
        # '--export_pages_int_list', '7', # 0  '--b_images_only', 'True'
        '--b_update_existed', 'True',  # False is default todo: allow "delete_overlapped" time named files
        '--b_interact', '0',
        '--b_images_only', 'True',      # mandatory
        '--b_execute_vsz', 'True',
        '--start_file', str(
            len(list(takewhile(lambda x: x != filename_st, h5.log_names_gen(cfg_in, f_row2name))))
            ),
        #'--min_time', cfg_in['min_time'].item().isoformat(),  # not works on filenames (no time data)
        #'--max_time', cfg_in['max_time'].item().isoformat(),
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
    st.go = (False, f'Hey! Prepare gpx tracks ({file_tracks}) from waypoints _manually_ before continue and rerun from st.start > {st.current}!')

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
        '--path_csv', str(path_cruise / device_folder / 'txt_processed'),
        '--data_columns_list', 'Pres, Temp, Cond, Sal, Lat, Lon, SA, sigma0, depth, soundV',  # O2, O2ppm,
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
        fr"out.text_path='{path_cruise / device_folder / 'txt_for_Obninsk'}'",
        f'out.text_date_format="%Y-%m-%dT%H:%M:%S"',
        f'out.text_float_format="%.6g"',
        f'out.file_name_fun="f\'{text_file_name_add}{{i+1:0>2}}.csv\'"',
        f'out.file_name_fun_log="\'{text_file_name_add}POS.csv\'"',
        # rec_num;identific;station;Pres;Temp;cond;Sal;O2%;O2ppm;sigmaT;soundV
        f'+out.cols_log={{rec_num: "@i + 1", identific: "@i + 1", station: "{i_cruise * 1000 + 1} + @i" , LONG: Lon_st, LAT: Lat_st, DATE: index}}',
        ''.join([
            f'+out.cols={{rec_num: "@i + 1", identific: "@i_log + 1", station: "{i_cruise * 1000 + 1} + @i_log", ',
            ', '.join([p if ':' in p else f'{p}: {p}' for p in
                       'Pres;Temp;Sal'.split(';')]), #;SigmaT;SoundVel;O2;O2ppm
            '}'
            ]),
        'out.sep=";"'
        ])
