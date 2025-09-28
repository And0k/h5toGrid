# Note:
# Stops before steps that need a manual prepare (70) i.e. you need set start = 70 to continue
# Gridding step needs debugging if interactive filtering is needed
# ---------------------------------------------------------------------------------------------
from os import chdir as os_chdir
from pathlib import Path
import re
from typing import Dict
import numpy as np
import pandas as pd
import gsw
from itertools import takewhile, chain
# my funcs
from utils.init import st
from utils import veuszPropagate
from hdf5_pandas.csv2h5 import main as csv2h5
from hdf5_pandas.gpx2h5 import main as gpx2h5
from hdf5_pandas.ctd_calc import main as ctd_calc
# from hdf5_pandas.csv_specific_proc import loaded_corr
from utils.h5_to_gpx import main as h5_to_gpx
from utils.grid2d_vsz import main as grid2d_vsz
from hdf5_pandas import h5

st.go = True   # False #
st.start = 20  # 20   # 1 5 30 70 80 115
st.end = 115   # 60 80 120

path_cruise = Path(r'd:\WorkData\BalticSea\221116_AMK91')
path_db = path_cruise / path_cruise.with_suffix('.h5').name  # same name as dir


# ---------------------------------------------------------------------------------------------

min_coord = 'Lat:53, Lon:18.6'  # 10
max_coord = 'Lat:60.55, Lon:30.3'  # includes Gulf Of Finland

def common_ctd_params(device):
    return {
    'db_path': str(path_db),
    'table': device,
    # deletes zeros & strange big negative values  # SigmaT:2,
    'min_dict': 'Cond:0.5, Sal:0.2'  #, O2:-2, O2ppm:-2',
    #    '--max_dict', f'O2:200, O2ppm:20',  # , {max_coord} for REDAS-like data
    }

def join_dicts_simple(dict_orig: Dict[str, str], dict_add: Dict[str, str]) -> Dict[str, str]:
    """
    Joins two dictionaries by combining values for common keys.
    All combined values are joined using ",".join() where dict_orig contains "," in value
    """
    result = dict_orig.copy()
    for k, v in dict_add.items():
        if k in result and ',' in v:
            # join values by comma
            result[k] = ', '.join(result[k].strip(", "), v.strip(", "))
        else:
            # assign/overwrite
            result[k] = v
    return result


def appended_arg_params_list(arg_dict, common_dict):
    arg_dict = join_dicts_simple(arg_dict, common_dict)
    arg_list = list(chain.from_iterable((f"--{k}", v) for k, v in arg_dict.items()))
    return arg_list


devices = {}

##############################################################################################################
device = 'CTD_SST_CTD90'
devices[device] = {'abbr': 's⁹', 'gpx_symbol': 'Triangle, Red'}
##############################################################################################################

if st(10, f'Save {device} data to DB'):
    from hdf5_pandas.csv_specific_proc import loaded_sst
    csv2h5(
        ["cfg/csv_CTD_SST.ini"]
        + appended_arg_params_list(
            common_ctd_params(device),
            {
                "path": str(path_cruise / device / "_raw" / "22[01][012][0123][0-9]_*.TOB"),
                #'dt_from_utc_hours': '0', #'2'
                "header": "Number,Date(text),Time(text),Pres,Temp,Sal,SIGMA,Turb,Trans,Cond,SVel,Vbatt",
                "cols_not_save_list": "Number,SIGMA,Vbatt,SVel",
                "delimiter_chars": "\\ \\",  # ''\s+',
                "b_interact": "0",
                # 'cols_not_save_list': 'N',
                # 'on_bad_lines': 'warn',
                #'min_dict': #'O2:0, O2ppm:0,   # replace strange values , not 'Turb:0' because we need 0 not NaN,
                # 'min_dict': 'Cond:0.5, Sal:0.2, Trans:40',
                # 'max_dict': 'Turb:10',
            },
        ),
        **{
            "in": {
                "fun_proc_loaded": loaded_sst,
                "csv_specific_param": {
                    "Temp_fun": lambda x: np.polyval([-1.102460295e-05, 1.00018, 0.037725], x),
                    "Cond_fun": lambda x: np.polyval([-0.000666294, 1.0279, -0.140743], x),
                    "Sal_fun": lambda Cond, Temp, Pres: gsw.SP_from_C(Cond, Temp, Pres),
                    "Turb_fun": lambda x: np.where(x < 0, 0, x),
                },
            }
        },
    )


if st(20, 'Extract CTD runs to "logRuns" table, filling it with CTD & nav params'):  # False: #
    # Note: this "logRuns" table needed by pattern used in next step with veuszPropagate()
    # todo: add station name column
    st.go = () != ctd_calc(['cfg/ctd_calc-find_runs.ini',
        '--db_path', str(path_db),
        '--tables_list', f'{device}',
        #'--table_nav', '',       # uncomment if nav data only in CTD data file
        '--min_samples', '100',  # fs*depth/speed = 200: if fs = 10Hz for depth 20m
        '--min_dp', '10',
        '--dt_between_min_minutes', '5',  # default 1s lead to split when communication with sonde lost
        '--b_keep_minmax_of_bad_files', 'True',  # (True helps get small runs if files was splitted on runs)
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
        'pattern_path': path_cruise / device / 'profiles_vsz' / '000000_000000.vsz',
        # 'min_time': np.datetime64('2022-12-16T00:00:00'),
        # 'max_time': '2020-12-30T22:37:00',
    }
    f_row2name = lambda r: '{:%y%m%d_%H%M%S}.vsz'.format(r['Index'])
    # It is possible to add exact interval to filename but time after probe is back on surface can be determined only
    # from next row, so we rely on ~pattern_loader.vsz to do it. Even freq=16Hz to determine last time not helps:
    # '_{}s.vsz'.format(round(max(r['rows']/16, (r['DateEnd'] - r['Index'] + pd.Timedelta(300, "s")).total_seconds()))

    # Copy files
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


file_tracks = f'CTD-sections=routes_{device}.gpx'
gpx_names_funs_list = """
    f'{row.fileName_st.split(chr(47))[-1]}'
    """  # variable  # Note: can not use "," inside one fun
#     i+1 if i <= 3 else i+2 if i <= 5 else i+3 if i < 25 else f"ctd{i-24:02d}" if i<41 else i-13 if i<=41
#    else i+9 if i<=56  # 42 -> 64
#    else i+ 15
# gpx_names_fun_format = """f'{{:{"s" if 25 <= i < 41 else "02d"}}}'"""
gpx_names_fun_format = '{:02d}⁹'
if st(50, 'Extract navigation data at time station starts to GPX waypoints'):  # False: #
    h5_to_gpx([
        'cfg/h5_to_gpx_CTDs.ini',
        '--db_path', str(path_db),
        '--tables_list', f"{','.join(devices)}",
        '--tables_log_list', 'logRuns',
        '--gpx_names_funs_list', gpx_names_funs_list,
        '--gpx_names_fun_format', gpx_names_fun_format,  # print variable
        '--select_from_tablelog_ranges_index', '0',
        '--dt_search_nav_tolerance_minutes', '1'  # to trigger interpolate
        ])
    st.go = (False, f'Hey! Prepare gpx tracks ({file_tracks}) from waypoints _manually_ before continue '
            'and rerun from st.start = 70!')

if st(70, f'Save waypoints/routes from _manually_ prepared {file_tracks} to hdf5'):  # False: #
    gpx2h5(['', '--path', str(path_cruise / rf'navigation\{file_tracks}'),
            '--table_prefix', r'navigation/sectionsCTD',
            '--b_sort', 'False'])  # need copy result from navigation\{path_db} manually, todo: auto copy
    st.go = (False, f'Hey! copy result from navigation/{path_db} _manually_ before continue (rerun from st.start = 80)!')

if st(80, 'Gridding'):  # and False: #
    # Note: Prepare veusz "zabor" pattern before
    grid2d_vsz(['cfg/grid2d_vsz.ini', '--db_path', str(path_db),
        '--table_sections', f'{device}/sectionsCTD_routes',  # navigation/
        '--subdir', 'CTD-sections',

        '--begin_from_section_int', '1', #'1',  # values <= 1 means no skip
        '--data_columns_list', "Turb, Trans, Temp, Sal, SigmaTh, soundV", #O2, O2ppm,
        # 'Eh, pH',  todo: N^2 - need calc before
        '--max_depth', '150',  # '250',
        '--filter_depth_wavelet_level_int', '7',  # 4, 5, 5, 4, 6, 4, 4, 5, 11
        '--convexing_ctd_bot_edge_max', '25',
        # Depth where we may not reach bot (40 set < bottom because it is harder to recover than delete?)
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
    ctd_calc([  # 'ctd_calc-find_runs.ini',
        '--db_path', str(path_db),
        '--tables_list', f'{device}',
        '--tables_log', '{}/logRuns',
        # '--min_samples', '99',  # fs*depth/speed = 200: if fs = 10Hz for depth 20m
        # '--min_dp', '9',
        # '--b_keep_minmax_of_bad_files', 'True',
        '--path_csv', str(path_cruise / device / 'txt_processed'),
        '--data_columns_list', 'Pres, Temp, Cond, Sal, Turb, Trans, Lat, Lon, SA, sigma0, depth, soundV',  #
        '--b_incremental_update', 'True',
        # todo: check it. If False need delete all previous result of ctd_calc() or set min_time > its last log time
        '--out.tables_list', 'None',
    ])

if st(115, 'Export csv for Obninsk'):
    try:
        m = re.match(r'[\d_]*(?P<abbr_cruise>[^\d]*)(?P<i_cruise>.*)', (path_cruise if path_cruise.stem[0].isdigit() else path_db).stem)
        i_cruise = int(m.group('i_cruise'))
        text_file_name_add = f"E090005O2_{m.group('abbr_cruise')}_{i_cruise}_H10_"
    except Exception as e:
        m = re.match(r'[\d_]*(?P<abbr_cruise>[^\d].*)', (path_cruise if path_cruise.stem[0].isdigit() else path_db).stem)
        i_cruise = 0
        text_file_name_add = f"E090005O2_{m.group('abbr_cruise')}_{i_cruise}_H10_"

    gpx_names_fun_str = '"df.Station"'  #fileName.str.split(\'/\', n=1).str[-1] or '"df.fileName_st.str.slice(5)"'
    from hdf5_alt.h5tocsv import main_call as h5tocsv
    h5tocsv([
        f'input.db_path="{path_db}"',
        f'input.tables=["{device}"]',
        f'input.tables_log=["{device}/logRuns"]',
        rf"out.text_path='{path_cruise / device / 'txt_for_Obninsk'}'",
        f'out.text_date_format="%Y-%m-%dT%H:%M:%S"',
        f'out.text_float_format="%.6g"',
        f"out.file_name_fun=\"f'{text_file_name_add}{{i+1:0>2}}.csv'\"",
        f"out.file_name_fun_log=\"'{text_file_name_add}POS.csv'\"",
        # rec_num;identific;station;Pres;Temp;cond;Sal;O2%;O2ppm;sigmaT;soundV
        # f'out.station_fun={gpx_names_fun_str}',
        f'+out.cols_log={{rec_num: "@i + 1", identific: "@i + 1", station: {gpx_names_fun_str}, '
        f"LONG: Lon_st, LAT: Lat_st, DATE: index}}",  # station: "[out_col_station_fun(ii) for ii in i]"
        "".join([
            '+out.cols={rec_num: "@i + 1", identific: "@i_log + 1", station: "@df_log.station.iat[@i_log]", ',
            ", ".join([
                p if ":" in p else f"{p}: {p}" for p in "Pres;Temp;Cond;Sal".split(";")
            ]),  # Temp:Temp90;SigmaT;SoundVel;O2;O2ppm
            "}",
        ]),
        'out.sep=";"',
    ])  #, out_col_station_fun=gpx_names_fun


if st(315, 'Export csv for Obninsk'):
    m = re.match(r'[\d_]*(?P<abbr_cruise>[^\d]*)(?P<i_cruise>.*)', path_cruise.name)
    i_cruise = int(m.group('i_cruise'))
    text_file_name_add = f"E090005O2_{m.group('abbr_cruise')}_{i_cruise}_H10_"

    # eval same as in h5_to_gpx.py
    gpx_names_fun_str = "lambda i, row, t=0: {}.format({})".format(
        (f"'{gpx_names_fun_format}'" if not gpx_names_fun_format.startswith("f'") else gpx_names_fun_format),
        gpx_names_funs_list
    )
    gpx_names_fun_ = eval(compile(gpx_names_fun_str, '', 'eval'))

    def gpx_names_fun(i):
        j = gpx_names_fun_(i, None).replace('⁹', '')
        return f"{i_cruise:02d}" + (f"{int(j):03d}" if j.isdigit() else f"{j:s}")
        # j = i+1 if i < 5 else i+2 if i < 8 else i+3 if i < 25 else f"ctd{i-24:02d}" if i < 41 else i-13
        # return f"{i_cruise:02d}" + (f"{j:s}" if isinstance(j, str) else f"{j:03d}")

    from hdf5_alt.h5tocsv import main_call as h5tocsv
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
                       'Pres;Temp;Cond;Sal;SigmaTh:sigma0;Turb;Trans'.split(';')]), #Temp:Temp90;SigmaT;SoundVel
            '}'
        ]),
        'out.sep=";"'
    ])  # , out_col_station_fun=gpx_names_fun
