import sys
from functools import partial
from pathlib import Path
import numpy as np
import pandas as pd

drive_d = 'D:' if sys.platform == 'win32' else '/mnt/D'  # to run on my Linux/Windows systems both
scripts_path = Path(drive_d + '/Work/_Python3/And0K/h5toGrid/scripts')
sys.path.append(str(Path(scripts_path).parent.resolve()))
# my funcs
from utils2init import st
import veuszPropagate
from to_pandas_hdf5.csv2h5 import main as csv2h5
from to_pandas_hdf5.gpx2h5 import main as gpx2h5
from to_pandas_hdf5.CTD_calc import main as CTD_calc
from to_pandas_hdf5.csv_specific_proc import proc_loaded_corr
from h5toGpx import main as h5toGpx
from grid2d_vsz import main as grid2d_vsz
from to_vaex_hdf5.nmea2h5 import main as nmea2h5


st.go = True   # False #
st.start = 110   # 5 30 70 80
st.end = 230   # 60 80 120

path_cruise = Path(r'd:\WorkData\BlackSea\200909_Ashamba')
path_db = path_cruise / path_cruise.with_suffix('.h5').name  # same name as dir

# Stop before steps that need manual preparings (70) i.e. set end < 70 at first
# Gridding (last step) needs debugging if interactive filtering is needed
# ---------------------------------------------------------------------------------------------

min_coord = 'Lat:10, Lon:10'
max_coord = 'Lat:50, Lon:50'
if st(1, 'Save gpx navigation to DB'):
    # Save navigation to DB
    for folder in ('OpenCPN_my', 'MapSource_Baranov'):
        gpx2h5(['',
                '--db_path', str(path_db),
                '--path', str(path_cruise / 'navigation' / folder / '*.gpx'),
                '--tables_list', ',navigation,',  # skip waypoints
                '--table_prefix', r'',
                #'--b_search_in_subdirs', if set True to get rid of this loop then will be problems with overlapped data files
                # '--min_date', '2019-07-17T14:00:00',
                '--b_interact', '0',
                '--b_skip_if_up_to_date', '0',  # '1' coerce to delete data loaded in same table in previous steps
                '--min_dict', f'{min_coord}',  # use at least -32768 to replace it by NaN
                '--max_dict', f'{max_coord}',
                ])


if st(2, 'Save NMEA navigation to DB'):
    argv_safe = sys.argv
    path_input = path_cruise / r'ADCP_600kHz\_raw\*000n.000'
    sys.argv = [__file__,
        f'input.path={path_input}',
        'input.b_search_in_subdirs=1',
        f'out.db_path={path_db}',
        f'out.table=navigation',  # skip waypoints
        'out.b_skip_if_up_to_date=0',  # '1' not implemented but would coerce to delete data loaded in same table in previous steps # todo implement
        # '--table_prefix', r'',
        # '--min_date', '2019-07-17T14:00:00',
        'program.b_interact=0',
        f'+filter.min={{{min_coord}, DepEcho:4}}',
        f'+filter.max={{{max_coord}, DepEcho:1500}}',
        ]
    nmea2h5()
    sys.argv = argv_safe


if st(3, 'Save Depth and navigation from ADCP data, exported by WinRiver II with at.ttf settings to DB'):
    # Save {device} data to DB. DepthReading is used for DepEcho column
    csv2h5(['cfg/csv_nav_ADCP_WinRiver2_at.ini',
            '--db_path', str(path_db),
            '--path', str(path_cruise / r'ADCP_600kHz\_raw\ASCII\*_at_ASC.TXT'),
            '--dt_from_utc_hours', '3',
            '--header', 'N, Time(text), Lat, Lon, Top, Heading, Pitch, stdPitch, Roll, stdRoll, Temper, Ve_none , Vn_none , Vup , ErrVhor , Int1, Int2, Int3, Int4, Cor1, Cor2, Cor3, Cor4, GOOD_bad, SpeedE_GGA, SpeedN_GGA, SpeedE_BT, SpeedN_BT, SpeedUp, ErrSpeed, DepEcho, CellSize, Cell1range, absorption, IntScale, ExtDepth, ExtHeading',
            '--cols_load_list', 'Time, Lat, Lon, DepEcho',
            '--delimiter_chars', r'\t',
            '--skiprows_integer', '0',
            '--table', 'navigation',
            '--b_remove_duplicates', 'True',
            '--b_skip_if_up_to_date', '0',  # '1' coerce to delete data loaded in same table in previous steps
            '--b_interact', '0',
            # '--csv_specific_param_dict', 'DepEcho_add:4.5',
            #'--min_dict', 'DepEcho:3',
            '--min_dict', f'{min_coord}', # use at least -32768 to replace it by NaN
            '--max_dict', f'{max_coord}',
            ],
           **   {'filter': {'min_DepEcho': 4}}  # removes rows

           # {'in': {
           #     'fun_proc_loaded': proc_loaded_corr}}
           )


device = 'CTD_Idronaut_OS310'
device_veusz_prefix = 'i0_'
common_ctd_params_list = [
    '--db_path', str(path_db),
    '--table', f'{device}',
    '--min_dict', f'Cond:10, Sal:5, SigmaT:3, O2:-2, O2ppm:-2, pH:7.77, {min_coord}, SoundVel:1465',  # del zeros & strange big negative values
    '--max_dict', 'O2:170, O2ppm:14.5',
    ]
common_ctd_params_dict = {'in': {
    'fun_proc_loaded': proc_loaded_corr,
    'csv_specific_param': {'O2_add': -0.357,
                          'O2ppm_add': -0.0332,
                          # 'Temp_add': 0.254, And convert to ITS90
                          'Sal_add': -0.003,
                          }
    }}

if st(10, f'Save {device} data to DB recorded by REDAS software'):
    # Save {device} data to DB
    csv2h5(['cfg/csv_CTD_IdrRedas.ini',
        '--path', str(path_cruise / device / '_raw_txt' / 'Ash[0-9]*.txt'), # str(path_cruise / device / r'_raw_txt\Ioffe*.txt') '[20|42]*.txt'
        #'--dt_from_utc_hours', '0', #'2'
        '--header',
        'Time(text),Pres(float),Temp90(float),Cond(float),Sal(float),SigmaT(float),O2(float),O2ppm(float),pH(float),'
        'Eh(float),SoundVel,Lat,Lon',
        '--delimiter_chars', r'\t',  # ''\s+',
        '--b_interact', '0',
        '--cols_not_use_list', 'N',
        # '--b_raise_on_err', '0'
        ] + common_ctd_params_list,
        **common_ctd_params_dict
        )

del common_ctd_params_dict['in']['fun_proc_loaded']
if st(11, f'Save {device} data to DB recorded in autonomous mode'):
    csv2h5([
        'cfg/csv_CTD_Idronaut.ini',
        '--path', str(path_cruise / device / '_raw_txt' / '20*[0-9].txt'),
        #'--dt_from_utc_hours', '0', #'2'
        '--header',
        'date(text),txtT(text),Pres(float),Temp90(float),Cond(float),Sal(float),O2(float),O2ppm(float),pH(float),Eh(float)',
        '--delimiter_chars', '\\ \\',  # ''\s+',
        '--b_interact', '0',
        #'--cols_not_use_list', 'N',
        # '--b_raise_on_err', '0'
        #'--min_dict', 'O2:0, O2ppm:0',  # replace strange values
        ] + common_ctd_params_list,
        **common_ctd_params_dict
        )

if st(12, f'Save {device} data to DB recorded in terminal mode'):
    if False: # done
        from to_pandas_hdf5.csv_specific_proc import correct_idronaut_terminal_txt
        from utils2init import open_csv_or_archive_of_them

        raw_parent = path_cruise / device / '_raw_txt'
        raw_pattern_file = '20*[0-9]t.txt'
        raw_found = list(raw_parent.glob(raw_pattern_file))
        dir_out = raw_parent / 'terminal_cleaned'  # sub replaces multilevel subdirs to 1 level that correct_fun() can only make
        for file_in in (raw_found or open_csv_or_archive_of_them(raw_parent, binary_mode=False, pattern=raw_pattern_file)):
            file_in = correct_idronaut_terminal_txt(file_in, dir_out=dir_out)
    # After manually convert formats (from, to):
    #           Press Temp Cond Sal O_O2% O_O2ppm pH Eh Time&Memory
    # Date Time Pres Temp Cond Sal OPT-O2%OPT-O2ppm pH Eh
    csv2h5([
        'cfg/csv_CTD_Idronaut.ini',
        '--path', str(path_cruise / device / '_raw_txt' / '20*[0-9]t.txt'),
        #'--dt_from_utc_hours', '0', #'2'
        '--header',
        'date(text),txtT(text),Pres(float),Temp90(float),Cond(float),Sal(float),O2(float),O2ppm(float),pH(float),Eh(float)',
        '--delimiter_chars', '\\ \\',  # ''\s+',
        '--b_interact', '0',
        #'--cols_not_use_list', 'N',
        # '--b_raise_on_err', '0'
        #'--min_dict', 'O2:0, O2ppm:0',  # replace strange values
        ] + common_ctd_params_list,
        **common_ctd_params_dict
        )




if st(20, 'Extract CTD runs to "logRuns" table, filling it with CTD & nav params'):  # False: #
    # Extract CTD runs (if files are not splitted on runs).
    # Note: Saves extended log needed by pattern used in next step with veuszPropagate
    # todo: be able provide log with (Lat,Lon) separately
    st.go = () != CTD_calc(['cfg/CTD_calc-find_runs.ini',
              '--db_path', str(path_db),
              '--tables_list', f'{device}',
              #'--table_nav', '',       # uncomment if nav data only in CTD data file
              '--min_samples', '95',  # fs*depth/speed = 200: if fs = 10Hz for depth 20m
              '--min_dp', '6',
              # Followig Not Helped!
              '--dt_between_min_minutes', '5',  # default 1s lead to split when commnication with sonde lost
              # '--b_keep_minmax_of_bad_files', 'True',
              # '--b_skip_if_up_to_date', 'True', - not works. Delete previous table manually, and from ~not_sorted!

              # '--out.tables_list', '',
              '--b_interact', '0'
              ])

if st(30, f'Draw {device} data profiles'):  # False: #
    from to_pandas_hdf5.h5toh5 import h5log_names_gen
    import re

    cfg_in = {
        'log_row': {},
        'db_path': str(path_db), # name of hdf5 pandas store where is log table
        #min_time, max_time: datetime, optional, allows range table_log rows
        'table_log': f'/{device}/logRuns', # str: name of log table - table with intervals:

        'pattern_path': path_cruise / device / '~pattern~.vsz'
        }
    f_row = lambda r: [
        '{Index:%y%m%d_%H%M}-{DateEnd:%H%M}.vsz'.format_map(r),
        bytes("time_range = ['{:%Y-%m-%dT%H:%M:%S}', '{:%Y-%m-%dT%H:%M:%S}']".format(r['Index'], r['DateEnd'] + pd.Timedelta(300, "s")), 'utf-8')]
    pattern_code = cfg_in['pattern_path'].read_bytes()  #encoding='utf-8'

    from os import getcwd as os_getcwd, chdir as os_chdir
    path_prev = os_getcwd()
    argv_prev = sys.argv

    os_chdir(cfg_in['pattern_path'].parent)
    for filename, str_expr in h5log_names_gen(cfg_in, f_row):
        path_vsz = cfg_in['pattern_path'].with_name(filename)
        path_vsz.write_bytes(re.sub(rb'^([^\n]+)', str_expr, pattern_code, count=1))

    veuszPropagate.main(['cfg/veuszPropagate.ini',
                         '--path', str(cfg_in['pattern_path'].with_name('??????_????-????.vsz')),  #path_db),
                         '--pattern_path', f"{cfg_in['pattern_path']}_",  # here used to auto get export dir only. must not be not existed file path
                         #'--table_log', f'/{device}/logRuns',
                         #'--add_custom_list', f'{device_veusz_prefix}USE_time_search_runs',  # 'i3_USE_timeRange',
                         # '--add_custom_expressions',
                         # """'[["{log_row[Index]:%Y-%m-%dT%H:%M:%S}", "{log_row[DateEnd]:%Y-%m-%dT%H:%M:%S}"]]'""",
                         #'--export_pages_int_list', '4', #'--b_images_only', 'True'
                         '--b_update_existed', 'False',  # todo: allow "delete_overlapped" time named files
                         '--b_interact', '0',
                         '--b_images_only', 'True'      # mandatory
                         #'--min_time', '2020-07-08T03:35:00',
                         #'--max_time', '2020-06-30T22:37:00',
                         ])



def merge_two_runs(df_log, irow_to, irow_from=None):
    """
    Merge 2 runs: copy ends data to row to keep from log's next row and then delete it
    :param df_log:
    :param irow_to:
    :param irow_from:
    :return:
    """
    if irow_from is None:
        irow_from = irow_to + 1
    df_merging = df_log.iloc[[irow_to, irow_from], :]
    k = input(f'{df_merging} rows selected (from, to). merge ? [y/n]:\n')
    if k.lower()!='y':
        print('done nothing')
        return
    cols_en = ['DateEnd'] + [col for col in df_log.columns if col.endswith('en')]
    ind_to, ind_from = df_merging.index
    df_log.loc[ind_to, cols_en] = df_log.loc[ind_from, cols_en]
    cols_sum = ['rows', 'rows_filtered']
    df_log.loc[ind_to, cols_sum] += df_log.loc[ind_from, cols_sum]
    df_log.drop(ind_from, inplace=True)
    print('ok, 10 nearest rows became:', df_log.iloc[(irow_from-5):(irow_to+5), :])


if False:
    # Merge each needed runs
    import pandas as pd
    from to_pandas_hdf5.h5toh5 import h5move_tables  #, h5index_sort, h5init

    tbl = f'/{device}'
    tbl_log = tbl + '/logRuns'
    with pd.HDFStore(path_db) as store:
        #     store = pd.HDFStore(path_db)
        df_log = store[tbl_log]

    # repeat if need:
    merge_two_runs(df_log, irow_to, irow_from=None)

    # write back
    with pd.HDFStore(path_db.with_name('_not_sorted.h5')) as store_tmp:
        try:
            del store_tmp[tbl_log]
        except KeyError:
            pass
        df_log.to_hdf(store_tmp, tbl_log, append=True, data_columns=True,
                      format='table', dropna=True, index=False)
    h5move_tables({
        'db_path_temp': path_db.with_name('_not_sorted.h5'),
        'db_path': path_db,
        'tables': [tbl_log],
        'tables_log': [],
        'addargs': ['--checkCSI', '--verbose']
        })

    # Now run step 30 with veuszPropagate seting: '--b_update_existed', 'False' to save only modified vsz/images. After that delete old vsz and its images


if False: #st(40)  # may not comment always because can not delete same time more than once
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

if st(50, 'Extract navigation data at time station starts to GPX waypoints'):  # False: #
    h5toGpx([
        'cfg/h5toGpx_CTDs.ini',
         '--db_path', str(path_db),
         '--tables_list', f'{device}',
         '--tables_log_list', 'logRuns',
         '--gpx_names_funs_list', """i+1""",
         '--gpx_names_fun_format', '{:02d}',
         '--select_from_tablelog_ranges_index', '0',
         '--dt_search_nav_tolerance_minutes', '1'  # to trigger interpolate
         ])

if False: # st(60, 'Extract navigation data at runs/starts to GPX tracks.'):    # Extract     # Useful to indicate where no nav?
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
    st.go = False  # Hey! Prepare gpx tracks _manually_ before continue and rerun from st.start = 70!

if st(70, 'Save waypoints/routes from _manually_ prepared gpx to hdf5'):  # False: #
    gpx2h5(['', '--path', str(path_cruise / r'navigation\CTD-sections=routes.gpx'),
            '--table_prefix',
            r'navigation/sectionsCTD'])  # need copy reult from {path_db}_not_sorted manually, todo: auto copy

if st(80, 'Gridding'):  # and False: #
    # Note: Prepare veusz "zabor" pattern before
    grid2d_vsz(['cfg/grid2d_vsz.ini', '--db_path', str(path_db),
                '--table_sections', r'navigation/sectionsCTD_routes',
                '--subdir', 'CTD-sections',
                '--begin_from_section_int', '8', #'1',  # values <= 1 means no skip
                '--data_columns_list', "Temp, Sal, SigmaTh, O2, O2ppm, Eh, pH, soundV",
                # 'Eh, pH',  todo: N^2 - need calc before
                '--max_depth', '1000', #'250',
                '--filter_depth_wavelet_level_int', '0',  # too little data to filter
                '--convexing_ctd_bot_edge_max', '50',
                # '--x_resolution', '0.2',
                # '--y_resolution', '5',
                '--dt_search_nav_tolerance_seconds', '120',
                '--symbols_in_veusz_ctd_order_list',
                "'Triangle, Green', 'Diamond, Blue', 'Triangle, Red', 'Square, Green'",
                '--b_temp_on_its90', 'True',  # modern probes
                '--blank_level_under_bot', '-500',
                # '--interact', 'False',
                #'--b_reexport_images', 'True'
                ])

    # todo: bug: bad top and bottom edges

if st(110, 'Export csv with some new calculated paremeters'):  # False: #
    # Extract CTD runs (if files are not splitted on runs):
    CTD_calc([  # 'CTD_calc-find_runs.ini',
        '--db_path', str(path_db),
        '--tables_list', f'{device}',
        '--tables_log', '{}/logRuns',
        # '--min_samples', '99',  # fs*depth/speed = 200: if fs = 10Hz for depth 20m
        # '--min_dp', '9',
        # '--b_keep_minmax_of_bad_files', 'True',
        '--path_csv', str(path_cruise / device / 'txt_processed'),
        '--data_columns_list', 'Pres, Temp90, Cond, Sal, O2, O2ppm, pH, Eh, Lat, Lon, SA, sigma0, depth, soundV',  #
        '--b_skip_if_up_to_date', 'True',
        # todo: check it. If False need delete all previous result of CTD_calc() or set min_time > its last log time
        '--out.tables_list', 'None',
        ])

if st(120, 'Meteo'):
    csv2h5([
        'cfg/csv_meteo.ini', '--path',  # to_pandas_hdf5/
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
device = 'CTD_Idronaut_OS316#494'
device_veusz_prefix = 'i3_'

if st(210, f'Save {device} data to DB'):  # False: #
    csv2h5([
        'cfg/csv_CTD_Idronaut.ini',
        '--path', str(path_cruise / device / '_raw_txt' / '20*[0-9].txt'),
        '--db_path', str(path_db),
        '--table', f'{device}',
        #'--dt_from_utc_hours', '0', #'2'
        '--header',
        'date(text),txtT(text),Pres(float),Temp90(float),Cond(float),Sal(float),O2(float),O2ppm(float),SigmaT(float)',
        '--delimiter_chars', '\\ \\',  # ''\s+',
        '--b_interact', '0',
        #'--cols_not_use_list', 'N',
        # '--b_raise_on_err', '0'
        #'--min_dict', 'O2:0, O2ppm:0',  # replace strange values
        ],
        **{'in': {
           #'fun_proc_loaded': proc_loaded_corr,
           'csv_specific_param': {
            'Temp90_fun': lambda x: np.polyval([-1.925036627169023e-06, 6.577767835930226e-05, 1.000754132707556, -0.014076681292841897], x/1.00024),
            'Sal_add': -0.01,
                                  }
           }}
        )  # todo: correct message on bad ['in']['csv_specific_param'] fun

if st(220, 'Extract CTD runs to "logRuns" table, filling it with CTD & nav params'):  # False: # (if files are not splitted on runs).
    # Note: extended logRuns fields needed in Veusz in next step
    # todo: be able provide log with (Lat,Lon) separately, improve start message if calc runs, check interpolation
    st.go = () != CTD_calc(['cfg/CTD_calc-find_runs.ini',
              '--db_path', str(path_db),
              '--tables_list', f'{device}',
              '--min_samples', '400',  # fs*depth/speed = 200: if fs = 10Hz for depth 20m
              '--min_dp', '40',  # todo: <=25
              '--b_keep_minmax_of_bad_files', 'True',
              # '--b_skip_if_up_to_date', 'True', - not works. Delete previous table manually, and from ~not_sorted!

              # '--out.tables_list', '',
              ])

if st(230, f'Draw {device} data profiles'):  # False: #
    # save all vsz files that uses separate code
    from to_pandas_hdf5.h5toh5 import h5log_names_gen
    import re

    cfg_in = {
        'log_row': {},
        'db_path': str(path_db), # name of hdf5 pandas store where is log table
        #min_time, max_time: datetime, optional, allows range table_log rows
        'table_log': f'/{device}/logRuns', # str: name of log table - table with intervals:

        'pattern_path': path_cruise / device / '~pattern~.vsz',
        # '--min_time', '2020-07-08T03:35:00',
        'min_time': pd.to_datetime('2020-07-06T17:23:00')
        }
    f_row = lambda r: [
        '{Index:%y%m%d_%H%M}-{DateEnd:%H%M}.vsz'.format_map(r),
        bytes("time_range = ['{:%Y-%m-%dT%H:%M:%S}', '{:%Y-%m-%dT%H:%M:%S}']".format(r['Index'], r['DateEnd'] + pd.Timedelta(300, "s")), 'utf-8')]
    pattern_code = cfg_in['pattern_path'].read_bytes()  #encoding='utf-8'

    from os import getcwd as os_getcwd, chdir as os_chdir
    path_prev = os_getcwd()
    argv_prev = sys.argv

    os_chdir(cfg_in['pattern_path'].parent)
    path_vsz_all = []
    for filename, str_expr in h5log_names_gen(cfg_in, f_row):

        path_vsz = cfg_in['pattern_path'].with_name(filename)
        path_vsz.write_bytes(re.sub(rb'^([^\n]+)', str_expr, pattern_code, count=1))  # replaces 1st row
        path_vsz_all.append(path_vsz)
        # try:
        #     with Popen(rf'"C:\Program Files (x86)\Veusz\veusz.exe" {filename} --unsafe-mode',
        #                stdout=PIPE, stderr=STDOUT) as proc:
        #         pass
        #
        # try:
        #     remote = Popen(
        #         [rf'"Program Files (x86)/Veusz/veusz.exe" {path_vsz.name} --export=export_0.jpg --export-option=page=[0] --export-option=dpi=300'])
        # except Exception as e:
        #     print(e)
        # r'"Program Files (x86)/Veusz/veusz.exe" {path_vsz} --export=export_%n.jpg --export-option=page=[5] --export-option=dpi=200'

        # [veusze.remote.args[0], str(vsz), '--unsafe-mode', '--embed-remote'],
        # shell=False, bufsize=0,
        # close_fds=False,
        # stdin=subprocess.PIPE,
        # stdout=subprocess.PIPE) #


    veuszPropagate.main(['cfg/veuszPropagate.ini',
                         '--path', str(cfg_in['pattern_path'].with_name('??????_????-????.vsz')),  #path_db),
                         '--pattern_path', f"{cfg_in['pattern_path']}_",  # here used to auto get export dir only. may not be _not existed file path_ if ['out']['paths'] is provided
                         #'--table_log', f'/{device}/logRuns',
                         #'--add_custom_list', f'{device_veusz_prefix}USE_time_search_runs',  # 'i3_USE_timeRange',
                         # '--add_custom_expressions',
                         # """'[["{log_row[Index]:%Y-%m-%dT%H:%M:%S}", "{log_row[DateEnd]:%Y-%m-%dT%H:%M:%S}"]]'""",
                         # '--export_pages_int_list', '1', #'--b_images_only', 'True'
                         '--b_interact', '0',
                         '--b_update_existed', 'True', # todo: delete_overlapped
                         '--b_images_only', 'True'
                         #'--min_time', '2020-07-08T03:35:00',

                         ],
        **{'out': {'paths': path_vsz_all}})
########################################################################################

if st(250, 'Extract navigation data at time station starts to GPX waypoints'):  # False: #
    h5toGpx([
    'cfg/h5toGpx_CTDs.ini',
    '--db_path', str(path_db),
    '--tables_list', f'{device_prev}, {device}',
    '--tables_log_list', 'logRuns',
    '--gpx_names_funs_list', """i+1""",
    '--gpx_names_fun_format', '{:03d}',
    '--select_from_tablelog_ranges_index', '0',
    '--dt_search_nav_tolerance_minutes', '1'  # to trigger interpolate
    ])