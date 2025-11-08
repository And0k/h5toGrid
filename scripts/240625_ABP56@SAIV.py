# Note:
# Stops before steps that need a manual prepare (70) i.e. you need set start = 70 to continue
# Gridding step needs debugging if interactive filtering is needed
# ---------------------------------------------------------------------------------------------
# import sys
from os import chdir as os_chdir
from pathlib import Path
import re
import numpy as np
import pandas as pd
import gsw
from itertools import takewhile
# my funcs
from utils.init import st, pairwise
from utils import veuszPropagate
from hdf5_pandas.csv2h5 import main as csv2h5
from hdf5_pandas.gpx2h5 import main as gpx2h5
from hdf5_pandas.ctd_calc import main as ctd_calc
# from hdf5_pandas.csv_specific_proc import loaded_corr
from utils.h5_to_gpx import main as h5_to_gpx
from utils.grid2d_vsz import main as grid2d_vsz
from hdf5_pandas import h5
from hdf5_alt.h5tocsv import main_call as h5tocsv

st.go = True   # False #
st.start = 250   # 1 5 30 70 80 115 | 210 315
st.end = 500   # 60 80 120

path_cruise = Path(r'D:\Cruises\BalticSea\240616_ABP56@i,t-chain')
path_db = (path_cruise / path_cruise / path_cruise.name.split("@", 1)[0]).with_suffix(".h5")

min_coord = 'Lat:53, Lon:18.6'  # 10
max_coord = 'Lat:60.55, Lon:30.3'  # includes Gulf Of Finland

match = re.match(r"(?P<year>\d\d)\d+\D+(?P<st_prefix>\d+)", path_db.stem)
year = match.group("year")  # or now().year  # =23
st_prefix = match.group("st_prefix")  # cruise number digits

devices = {}


##############################################################################################################
if st(1, 'Save gpx navigation to DB'):
    # Save navigation to DB
    for folder in (['_raw']):
        gpx2h5([  # '',
            '--db_path', str(path_db),
            '--path', str(path_cruise / 'navigation' / folder / f'{year}*.gpx'),
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


##############################################################################################################
device = 'CTD_SST_MWS#3613'
devices[device] = {'abbr': 'sm', 'folder': 'CTD_SST_MWS#3613', 'gpx_symbol': 'Triangle, Green'}
##############################################################################################################

common_ctd_params_list = [
    '--db_path', str(path_db),
    '--min_dict', f'Sal:0.2',
    ]

if st(10, f'Save {device} data to DB'):
    # Time [hh:mm:ss]	Bottle []	Pressure [dbar]	Temperature [░C]	Conductivity [mS/cm]	Salinity [PSU]	Sound Vel. [m/s]	Density [kg/m│]	Spec. Cond. [mS/cm]	Latitude	Longitude	UTC	Comments  [Index]
    # 14:22:48	0	-0.4	-1.870	-0.002	0.000	1392.76	999.68	-0.004	54░32'24.02'' N	19░39'7.75'' E	2023-12-09 14:22:47
    from hdf5_pandas.csv_specific_proc import loaded_sst_mws_with_coord
    #  todo: check why incremental_update not works
    #  todo: remove "Error" lines from files before loading
    csv2h5([
        #'cfg/csv_CTD_SST.ini',
        '--skiprows_integer', '40',
        '--path', str(path_cruise / devices[device].get('folder', device) / '_raw_txt' / f'{st_prefix}*.txt'),
        # '--dt_from_utc_hours', '2',
        '--header', 'NoDate(text),Bottle,Pres,Temp,Cond,Sal,SVel,Dens,SpCond,Lat(text),Lon(text),Time(text),Comments(text)',
        '--cols_not_save_list', 'NoDate,Bottle,SVel,Dens,SpCond,Comments',
        '--cols_save_list', 'Pres,Temp,Cond,Sal,Lat,Lon',
        '--delimiter_chars', r'\t',  # ''\s+',
        '--table', f'{device}',
        '--b_interact', '0'
        # '--on_bad_lines', 'warn',
        ] + common_ctd_params_list,
        **{'in': {
            'fun_proc_loaded': loaded_sst_mws_with_coord,
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

if st(20, 'Extract CTD runs to "logRuns" table, filling it with CTD & nav params'):  # False: # (if files are not splitted on runs).
    # Note: extended logRuns fields needed in Veusz in next step
    # todo: be able provide log with (Lat,Lon) separately, improve start message if calc runs, check interpolation
    st.go = () != ctd_calc(['cfg/ctd_calc-find_runs.ini',
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

if st(30, f'Draw {device} data profiles'):  # False: #
    # Note: if vsz pattern uses map from *.h5, then be sure that it exists
    cfg_in = {
        'log_row': {},
        'db_path': str(path_db), # name of hdf5 pandas store where is log table
        'table_log': f'/{device}/logRuns', # str: name of log table - table with intervals:
        'pattern_path': path_cruise / devices[device].get('folder', device) / 'profiles_vsz' / '000000_000000.vsz',
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
        # '--add_custom_list', f"{devices[device]['abbr']}_USE_time_search_runs",  # 'i3_USE_timeRange',
        # '--add_custom_expressions',
        # """'[["{log_row[Index]:%Y-%m-%dT%H:%M:%S}", "{log_row[DateEnd]:%Y-%m-%dT%H:%M:%S}"]]'""",
        # '--export_pages_int_list', '7', # 0  '--b_images_only', 'True'
        '--b_update_existed', 'True',  # False is default todo: allow "delete_overlapped" time named files
        '--b_interact', '0',
        '--b_images_only', 'True',      # mandatory
        '--b_execute_vsz', 'True',
        '--start_file', '231211_140752.vsz',
        # str(
        #     len(list(takewhile(lambda x: x != filename_st, h5.log_names_gen(cfg_in, f_row2name))))
        #     ),
        #'--min_time', cfg_in['min_time'].item().isoformat(),  # not works on filenames (no time data)
        #'--max_time', cfg_in['max_time'].item().isoformat(),
    ])

if st(50, 'Extract navigation data at time station starts to GPX waypoints'):  # False: #
    h5_to_gpx([
        'cfg/h5_to_gpx_CTDs.ini',
        '--db_path', str(path_db),
        '--tables_list', f"{','.join(devices)}",
        '--gpx_symbols_list', ','.join("'{gpx_symbol}'".format_map(d) for d in devices.values()),
        '--tables_log_list', 'logRuns',
        '--gpx_names_funs_list', gpx_names_funs_list,  # """i+1""",
        '--gpx_names_fun_format', '{:s}m',  # gpx_names_fun_format,  # print variable
        '--select_from_tablelog_ranges_index', '0',
        '--dt_search_nav_tolerance_minutes', '1'  # to trigger interpolate
    ])
    st.go = (False, f'Hey! Prepare gpx tracks ({file_tracks}) from waypoints _manually_ before continue and rerun from st.start > {st.current}!')

if st(70, 'Save waypoints/routes from _manually_ prepared gpx to hdf5'):  # False: #
    gpx2h5(['', '--path', str(path_cruise / fr'navigation\CTD-sections=routes_{device}.gpx'),
            '--table_prefix', fr'navigation/sectionsCTD_{device}'])  # need copy result from navigation\{path_db}_not_sorted manually, todo: auto copy

if st(80, 'Gridding'):  # and False: #
    # Note: Prepare veusz "zabor" pattern before
    grid2d_vsz([
        'cfg/grid2d_vsz.ini', '--db_path', str(path_db),
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

if st(90, 'Export csv with some new calculated parameters'):  # False: #
    # Extract CTD runs (if files are not splitted on runs):
    ctd_calc([  # 'ctd_calc-find_runs.ini',
        '--db_path', str(path_db),
        '--tables_list', f'{device}',
        '--tables_log', '{}/logRuns',
        # '--min_samples', '99',  # fs*depth/speed = 200: if fs = 10Hz for depth 20m
        # '--min_dp', '9',
        # '--b_keep_minmax_of_bad_files', 'True',
        '--path_csv', str(path_cruise / devices[device].get('folder', device) / 'txt_processed'),
        '--data_columns_list', 'Pres, Temp, Cond, Sal, Lat, Lon, SA, sigma0, depth, soundV',  # O2, O2ppm,
        '--b_incremental_update', 'True',
        # todo: check it. If False need delete all previous result of ctd_calc() or set min_time > its last log time
        '--out.tables_list', 'None',
    ])

if st(115, 'Export csv for Obninsk'):
    m = re.match(r'[\d_]*(?P<abbr_cruise>\D*)(?P<i_cruise>.*)', path_cruise.name)
    i_cruise = int(m.group('i_cruise'))
    text_file_name_add = f"E090005O2_{m.group('abbr_cruise')}_{i_cruise}_H10_"
    gpx_names_fun_str = '"df.fileName_st.str.split(\'/\', n=1).str[-1]"'

    h5tocsv([
        f'input.db_path="{path_db}"',
        f'input.tables=["{device}"]',
        f'input.tables_log=["{device}/logRuns"]',
        fr"out.text_path='{path_cruise / device / 'txt_for_Obninsk'}'",
        f'out.text_date_format="%Y-%m-%dT%H:%M:%S"',
        f'out.text_float_format="%.6g"',
        f'out.file_name_fun="f\'{text_file_name_add}{{i+1:0>2}}.csv\'"',
        f'out.file_name_fun_log="\'{text_file_name_add}POS.csv\'"',
        # f'out.station_fun={gpx_names_fun_str}',
        f'+out.cols_log={{rec_num: "@i + 1", identific: "@i + 1", station: {gpx_names_fun_str}, '
        f'LONG: Lon_st, LAT: Lat_st, DATE: index}}',  # station: "[out_col_station_fun(ii) for ii in i]"
        ''.join([  # add 'Time': 'index' if need Time
            '+out.cols={rec_num: "@i + 1", identific: "@i_log + 1", station: "@df_log.station.iat[@i_log]", ',
            ', '.join([p if ':' in p else f'{p}: {p}' for p in
                'Pres;Temp;Cond;Sal;sigma0;soundV'.split(';')]),  # Temp:Temp90;SigmaT;SoundVel
            '}'
        ]),
        'out.sep=";"'
    ])  # , out_col_station_fun=gpx_names_fun


##############################################################################################################
device = 'CTD_SAIV'
devices[device] = {'abbr': 'sa', 'folder': 'CTD_SAIV', 'gpx_symbol': 'Diamond, Blue'}
##############################################################################################################

common_ctd_params_list = [
    '--db_path', str(path_db),
    '--min_dict', 'Sal:0.2',
]

if st(210, f'Save {device} data to DB'):
    # Ser	Meas	Sal.	Cond.	Temp	Ox %	mg/l	F (µg/l)	T (FTU)	Density	S. vel.	Depth(u)	Date	Time
    # 1	108	7.669	8.333	5.167	103.88	12.50	0.76	0.46	6.058	1436.77	0.3075	09.12.2023	14:23:09
    # Ser	Meas	Sal.	Cond.	Temp	                                    Density	S. vel.	Depth(u)	Date	Time
    from hdf5_pandas.csv_specific_proc import loaded_sst
    sub_dir_in = "Exported"  # 'txt'
    csv2h5([
        # 'cfg/csv_CTD_SST.ini',
        '--skiprows_integer', '4',
        '--path', str(path_cruise / devices[device].get('folder', device) / sub_dir_in / f'{st_prefix}*.txt'),
        # '--dt_from_utc_hours', '2',  # (text),Pres,Temp,Cond,Sal,SIGMA,DO_ml,SVel,Vbatt
        '--header',  # 'Date(text),Time(text),Pres,Temp,Cond,Sal,SIGMA,O2,O2ppm,DO_ml,SVel,Vbatt',
        'No,Meas,Sal,Cond,Temp,O2,O2ppm,Fluor,Turb,Dens,SVel,Pres,Date(text),Time(text)',
        # 'No,Meas,Sal,Cond,Temp,Dens,SVel,Pres,Date(text),Time(text)',  # if some probes switched off
        '--cols_not_save_list', 'No,Meas',  # ,SVel,Dens
        '--delimiter_chars', r'\t',  # ''\s+',
        '--table', f'{device}',
        '--b_interact', '0'
        # '--on_bad_lines', 'warn',
        ] + common_ctd_params_list,
        **{'in': {
            'fun_proc_loaded': loaded_sst,
        }}
    )
if st(220,
        'Extract CTD runs to "logRuns" table, filling it with CTD & nav params'):  # False: # (if files are not splitted on runs).
    # Note: extended logRuns fields needed in Veusz in next step
    st.go = () != ctd_calc(['cfg/ctd_calc-find_runs.ini',
        '--db_path', str(path_db),
        '--tables_list', f'{device}',
        '--min_samples', '20',  # fs*depth/speed = 200: if fs = 10Hz for depth 20m
        '--min_dp', '5',  # bug: result 2nd run from same file has Pres_en = 1.26: bad Pres_en?
        '--dt_between_min_minutes', '5',
        '--b_keep_minmax_of_bad_files', 'True',
        # '--b_incremental_update', 'True', - not works. Delete previous table manually, and from ~not_sorted!
        # '--out.tables_list', '',
        '--b_interact', '0'
    ])


if st(230, f"Draw {device} data profiles"):  # False: #
    # Note: if vsz pattern uses map from *.h5, then be sure that it exists
    cfg_in = {
        "log_row": {},
        "db_path": str(path_db),  # name of hdf5 pandas store where is log table
        "table_log": f"/{device}/logRuns",  # str: name of log table - table with intervals:
        "pattern_path": path_cruise / device / "profiles_vsz" / "000000_000000.vsz",
        # "min_time": np.datetime64("2023-05-22T00:00:00"),
        # 'max_time': '2020-12-30T22:37:00',
    }
    f_row2name = lambda r: "{:%y%m%d_%H%M%S}.vsz".format(r["Index"])
    # It is possible to add exact interval to filename but time after probe is back on surface can be determined only
    # from next row, so we rely on ~pattern_loader.vsz to do it. Even freq=16Hz to determine last time not helps:
    # '_{}s.vsz'.format(round(max(r['rows']/16, (r['DateEnd'] - r['Index'] + pd.Timedelta(300, "s")).total_seconds()))

    # Copy files
    pattern_code = cfg_in["pattern_path"].read_bytes()  # encoding='utf-8'
    filename_st = None
    os_chdir(cfg_in["pattern_path"].parent)
    for filename in h5.log_names_gen(cfg_in, f_row2name):
        path_vsz = cfg_in["pattern_path"].with_name(filename)
        path_vsz.write_bytes(pattern_code)  # re.sub(rb'^([^\n]+)', str_expr, pattern_code, count=1)
        # Get filename_st (do once)
        if filename_st is None:
            filename_st = filename
    # cfg_in['min_time'] not works on filenames, so we convert it to 'start_file_index'
    if "min_time" in cfg_in:
        del cfg_in["min_time"]  # del to count fro 0:
        start_file_index = len(
            list(takewhile(lambda x: x < filename_st, h5.log_names_gen(cfg_in, f_row2name)))
        )
    else:
        start_file_index = 0
    veuszPropagate.main([
        "cfg/veuszPropagate.ini",
        "--path", str(cfg_in["pattern_path"].with_name("??????_??????.vsz")),  # _*s path_db),
        "--pattern_path", f"{cfg_in['pattern_path']}_",  # here used to auto get export dir only. must not be not existed file path
        #'--table_log', f'/{device}/logRuns',
        #'--add_custom_list', f'{device_veusz_prefix}USE_time_search_runs',  # 'i3_USE_timeRange',
        # '--add_custom_expressions',
        # """'[["{log_row[Index]:%Y-%m-%dT%H:%M:%S}", "{log_row[DateEnd]:%Y-%m-%dT%H:%M:%S}"]]'""",
        # '--export_pages_int_list', '7', # 0  '--b_images_only', 'True'
        "--b_update_existed", "True",  # False is default todo: allow "delete_overlapped" time named files
        "--b_interact", "0",
        "--b_images_only", "True",  # mandatory
        "--b_execute_vsz", "True",
        "--start_file", str(start_file_index),
        #'--min_time', cfg_in['min_time'].item().isoformat(),  # not works on filenames (no time data)
        #'--max_time', cfg_in['max_time'].item().isoformat(),
    ])

file_tracks = "CTD-sections=routes.gpx"
gpx_names_funs_list = """
    f'{row.fileName_st.split(chr(47))[-1]}'
    """  # variable  # Note: can not use "," inside one fun
#     i+1 if i <= 3 else i+2 if i <= 5 else i+3 if i < 25 else f"ctd{i-24:02d}" if i<41 else i-13 if i<=41
#    else i+9 if i<=56  # 42 -> 64
#    else i+ 15
# gpx_names_fun_format = """f'{{:{"s" if 25 <= i < 41 else "02d"}}}'"""
if st(250, 'Extract navigation data at time station starts to GPX waypoints'):  # False: #
    h5_to_gpx([
        'cfg/h5_to_gpx_CTDs.ini',
        '--db_path', str(path_db),
        '--tables_list', f"{','.join(devices)}",
        '--gpx_symbols_list', ','.join("'{gpx_symbol}'".format_map(d) for d in devices.values()),
        '--tables_log_list', 'logRuns',
        '--gpx_names_funs_list', gpx_names_funs_list,  # """i+1""",
        '--gpx_names_fun_format', '{{:s}}{}'.format(devices[device]['abbr'][0]),  # gpx_names_fun_format,  # print variable
        '--select_from_tablelog_ranges_index', '0',
        '--dt_search_nav_tolerance_minutes', '1'  # to trigger interpolate
    ])
    st.go = (
        False, f'Hey! Prepare gpx tracks ({file_tracks}) from waypoints _manually_ '
        f'before continue and rerun from st.start > {st.current}!')

if st(270, 'Save waypoints/routes from _manually_ prepared gpx to hdf5'):  # False: #
    gpx2h5(['', '--path', str(path_cruise / fr'navigation\CTD-sections=routes_{device}.gpx'),
        '--table_prefix',
        fr'navigation/sectionsCTD_{device}'])  # need copy result from navigation\{path_db}_not_sorted manually, todo: auto copy

if st(280, 'Gridding'):  # and False: #
    # Note: Prepare veusz "zabor" pattern before
    grid2d_vsz(['cfg/grid2d_vsz.ini', '--db_path', str(path_db),
        '--table_sections', fr'{device}/sectionsCTD_routes',
        '--subdir', 'CTD-sections',
        '--begin_from_section_int', '3',  # '1',  # values <= 1 means no skip
        '--data_columns_list', "Temp, Sal, SigmaTh, O2, O2ppm, Turb, ChlA",  # soundV
        # 'Eh, pH',  todo: N^2 - need calc before
        '--max_depth', '250',  # '250',
        '--filter_depth_wavelet_level_int', '5',  # 4, 5, 5, 4, 6, 4, 4, 5
        '--convexing_ctd_bot_edge_max', '40',  # set < bottom because it is harder to recover than delete
        # '--x_resolution', '0.2',
        # '--y_resolution', '5',
        '--dt_search_nav_tolerance_seconds', '120',
        # '--symbols_in_veusz_ctd_order_list',
        # "'Triangle, Green', 'Diamond, Blue', 'Triangle, Red', 'Square, Green'",
        '--b_temp_on_its90', 'True',  # modern probes
        '--blank_level_under_bot', '-220',
        '--symbols_in_veusz_ctd_order_list', "'Triangle, Red', ",
        # '--interact', 'False',
        '--b_reexport_images', 'True'
    ])

if st(290, 'Export csv with some new calculated parameters'):  # False: #
    # Extract CTD runs (if files are not splitted on runs):
    ctd_calc([  # 'ctd_calc-find_runs.ini',
        '--db_path', str(path_db),
        '--tables_list', f'{device}',
        '--tables_log', '{}/logRuns',
        # '--min_samples', '99',  # fs*depth/speed = 200: if fs = 10Hz for depth 20m
        # '--min_dp', '9',
        # '--b_keep_minmax_of_bad_files', 'True',
        '--path_csv', str(path_cruise / devices[device].get('folder', device) / 'txt_processed'),
        '--data_columns_list', 'Pres, Temp, Cond, Sal, Lat, Lon, SA, sigma0, depth, soundV',  # O2, O2ppm,
        '--b_incremental_update', 'True',
        # todo: check it. If False need delete all previous result of ctd_calc() or set min_time > its last log time
        '--out.tables_list', 'None',
    ])

if st(315, 'Export csv for Obninsk'):
    m = re.match(r'[\d_]*(?P<abbr_cruise>\D*)(?P<i_cruise>.*)', path_cruise.name)
    i_cruise = int(m.group('i_cruise'))
    text_file_name_add = f"E090005O2_{m.group('abbr_cruise')}_{i_cruise}_H10_"
    gpx_names_fun_str = '"df.fileName_st.str.split(\'/\', n=1).str[-1]"'

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
        # f'out.station_fun={gpx_names_fun_str}',
        f'+out.cols_log={{rec_num: "@i + 1", identific: "@i + 1", station: {gpx_names_fun_str}, '
        f'LONG: Lon_st, LAT: Lat_st, DATE: index}}',  # station: "[out_col_station_fun(ii) for ii in i]"
        ''.join([
            '+out.cols={rec_num: "@i + 1", identific: "@i_log + 1", station: "@df_log.station.iat[@i_log]", ',
            ', '.join([p if ':' in p else f'{p}: {p}' for p in
                'Pres;Temp;Sal'.split(';')]),  # Temp:Temp90;SigmaT;SoundVel
            '}'
        ]),
        'out.sep=";"'
    ])  # , out_col_station_fun=gpx_names_fun
