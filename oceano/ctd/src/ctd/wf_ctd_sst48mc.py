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

from ctd import wf_cfg  # path_cruise, path_db, min_coord, max_coord


max_coord = 'Lat:60.55, Lon:30.3'  # includes Gulf Of Finland

# separate cruise number digits
cruise = re.match(r"(?P<year>\d\d)\d+_*(?P<vessel>\D+)(?P<num>\d+)",wf_cfg.path_cruise.stem).groupdict()

# %% Save device data to DB
device = "CTD_SST_48Mc#1253"
devices = {device: {"abbr": "ss", "folder": "CTD_SST48", "gpx_symbol": "Triangle, Red"}}
##########################################################################################################
sub_dir_in = "Exported"  # 'txt'


def proc(common_ctd_params_list, o2_fun=None, o2ppm_fun=None, st_base=10):

    st_prefix = cruise["year"]
    if st(st_base, f"Save {device} data to DB. Searching {st_prefix}*.csv files"):
        from hdf5_pandas.csv_specific_proc import loaded_sst

        csv2h5(
            [
                "cfg/csv_CTD_SST.ini",
                "--path", str(wf_cfg.path_cruise / devices[device]["folder"] / sub_dir_in / f"{st_prefix}*.csv"),
                "--table", f"{device}",
                #'--dt_from_utc_hours', '0', #'2'
                "--header", "Date(text),Time(text),Pres,Temp,Cond,Sal,SIGMA,O2,O2ppm,SVel,Vbatt",
                # IntD,IntD,Press,Temp,Cond,Salin,Sigma,sat,DO_mg,Sound,Vbatt ,
                "--cols_not_save_list", "SIGMA,Vbatt,SVel",
                # '--delimiter_chars', '\\ \\',  # ''\s+',
                "--b_interact", "0",
                #'--cols_not_save_list', 'N',
                # '--on_bad_lines', 'warn'
                #'--min_dict', 'O2:0, O2ppm:0',  # replace strange values
            ]
            + common_ctd_params_list,
            **{  # device_params_dict
                "in": {
                    "fun_proc_loaded": loaded_sst,
                    "csv_specific_param": {
                        "Temp_fun": lambda x: np.polyval(  # 2025-11-21
                            [-0.00010627, 1.003, -0.0099154],
                            x,
                        ),
                        # "Cond_fun": lambda x: np.polyval(  # 2025-11-28 - not used (may be bad)
                        #     [7.61387725e-06, -0.00061253472, 1.01855, -0.11019353], x
                        # ),
                        "Cond_fun": lambda x: np.polyval(
                            [-7.57195308190065e-7, 3.80941696689889e-5, 1.0018619893672, -0.034845948296864], x
                        ),  # old
                        "Sal_fun": lambda Cond, Temp, Pres: gsw.SP_from_C(Cond, Temp, Pres),
                        # coef from ABP64 Winkler data before or in the GoF
                        "O2_fun": o2_fun,
                        "O2ppm_fun": o2ppm_fun,
                        # coef from ABP64 all Winkler data together - not to use
                        # "O2_fun": lambda O2ppm, Sal, Temp, Pres: DO(0.44499 + 1.1976 * O2ppm, Sal, Temp, Pres),
                        # "O2ppm_fun": lambda x: 0.44499 + 1.1976 * x,  # 2025-12-19 ABP64 intercal. to Winkler
                        # 0.43915 + 1.1301 * x,  # 2025-12-08 ABP64 intercal. to Winkler
                    },
                }
            },
        )

    if st(st_base + 10, 'Extract CTD runs to "logRuns" table, filling it with CTD & nav params'):
        # Note: this "logRuns" table needed by pattern used in next step with veuszPropagate()

        st.go = () != ctd_calc([
            "cfg/ctd_calc-find_runs.ini",
            "--db_path", str(wf_cfg.path_db),
            "--tables_list", f"{device}",
            #'--table_nav', '',       # uncomment if nav data only in CTD data file
            "--min_samples", "500",  # 50 fs*depth/speed = 200: if fs = 10Hz for depth 20m
            "--min_dp", "10",  # 5
            "--dt_between_min_minutes", "5",  # default 1s lead to split when communication with sonde lost
            "--b_keep_minmax_of_bad_files", "True",  # (True helps get small runs if files was splitted on runs)
            # '--b_incremental_update', 'True', - not works. Delete previous table manually, and from ~not_sorted!
            # '--out.tables_list', '',
            "--b_interact", "0",
        ])


    # Usually not needed step! #
    b_update = True  # False:  #  # if False may not skip because can not delete same rows more than once
    if False and st(
        st_base + 15, "Values correction (updating DB)" if b_update else f"Deleting bad runs from DB"
    ):
        from h5_cor import main as h5_cor_main

        h5_cor_main(wf_cfg.path_db, device)
        if False:  # for debug:
            from h5_cor import h5cor

            h5cor(time_ranges, edges_sources, b_update=True, cfg_out=cfg_out, coef_for_interval=coef_for_interval)
        st.go = (
            False,
            "Hey! logRuns table removed for correction! Recalculate bot/top parameters: set st.start = 20, comment this step and run!",
        )

    if st(st_base + 20, f"Draw {device} data profiles"):  # False: #
        # Note: if vsz pattern uses map from *.h5, then be sure that it exists
        cfg_in = {
            "log_row": {},
            "db_path": str(wf_cfg.path_db),  # name of hdf5 pandas store where is log table
            "table_log": f"/{device}/logRuns",  # str: name of log table - table with intervals:
            "pattern_path": wf_cfg.path_cruise / devices[device]["folder"] / "profiles_vsz" / "000000_000000.vsz",
            # 'min_time': np.datetime64('2022-11-04T22:00:00'),
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
            if not path_vsz.is_file():  # skip is useful if user edited this vsz file
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
            "--path", str(cfg_in["pattern_path"].with_name("??????_??????.vsz")),  # _*s wf_cfg.path_db),
            "--pattern_path", f"{cfg_in['pattern_path']}_",  # here used to auto get export dir only. must not be not existed file path
            #'--table_log', f'/{device}/logRuns',
            #'--add_custom_list', f"{devices[device]['abbr']}_USE_time_search_runs",  # 'i3_USE_timeRange',
            # '--add_custom_expressions',
            # """'[["{log_row[Index]:%Y-%m-%dT%H:%M:%S}", "{log_row[DateEnd]:%Y-%m-%dT%H:%M:%S}"]]'""",
            # '--export_pages_int_list', '2,3', # 0  '--b_images_only', 'True'
            "--export_format", "png",
            "--b_update_existed", "True",  # False is default todo: allow "delete_overlapped" time named files
            "--b_interact", "0",
            "--b_images_only", "True",  # mandatory
            "--b_execute_vsz", "True",
            "--start_file", str(start_file_index),
            #'--min_time', cfg_in['min_time'].item().isoformat(),  # not works on filenames (no time data)
            #'--max_time', cfg_in['max_time'].item().isoformat(),
        ])

    if False:
        tbl = f"/{device}"
        tbl_log = f"{tbl}/logRuns"
        with pd.HDFStore(wf_cfg.path_db) as store:
            #     store = pd.HDFStore(wf_cfg.path_db)
            df_log = store[tbl_log]

        # repeat if need:
        irow_to = 130  # 85
        h5.merge_two_runs(df_log, irow_to, irow_from=None)

        # write back
        with pd.HDFStore(wf_cfg.path_db.with_name("_not_sorted.h5")) as store_tmp:
            try:
                del store_tmp[tbl_log]
            except KeyError:
                pass
            df_log.to_hdf(
                store_tmp, tbl_log, append=True, data_columns=True, format="table", dropna=True, index=False
            )
        h5.move_tables({
            "temp_db_path": wf_cfg.path_db.with_name("_not_sorted.h5"),
            "db_path": wf_cfg.path_db,
            "tables": [tbl_log],
            "tables_log": [],
            "addargs": ["--checkCSI", "--verbose"],
        })
        # Now run step 30 with veuszPropagate setting: '--b_update_existed', 'False' to save only modified vsz/images. After that delete old vsz and its images

    file_tracks = "CTD-sections=routes.gpx"
    gpx_names_funs_list = """
        f'{row.fileName_st.split(chr(47))[-1]}'
        """  # variable  # Note: can not use "," inside one fun
    gpx_names_fun_format = "{:s}"
    if st(st_base + 40, "Extract navigation data at time station starts to GPX waypoints"):  # False: #
        h5_to_gpx([
            "cfg/h5_to_gpx_CTDs.ini",
            "--db_path", str(wf_cfg.path_db),
            "--tables_list", f"{','.join(devices)}",
            "--tables_log_list", "logRuns",
            "--gpx_names_funs_list", gpx_names_funs_list,
            "--gpx_names_fun_format", gpx_names_fun_format,  # print variable
            "--select_from_tablelog_ranges_index", "0",
            "--dt_search_nav_tolerance_minutes", "1",  # to trigger interpolate
        ])
        st.g = (
            False,
            f"Hey! Prepare gpx tracks ({file_tracks}) from waypoints _manually_ before continue and rerun frm st.start = 70!",
        )

    if False:  # st(st_base + 50, 'Extract navigation data at runs/starts to GPX tracks.') # to indicate where no nav?
        h5_to_gpx([
            "cfg/h5_to_gpx_CTDs.ini",
            "--db_path", str(wf_cfg.path_db),
            "--tables_list", f"{','.join(devices)}",
            "--tables_log_list", "logRuns",
            "--select_from_tablelog_ranges_index", None,  # Export tracks
            "--gpx_names_fun_format", "{1:%y%m%d}_{0:}",  # track name of format(timeLocal, tblD_safe)
            "--gpx_names_funs_list", '"i, row.Index"',
            "--gpx_names_funs_cobined", "",
        ])
        st.go = (False, "Hey! Prepare gpx tracks _manually_ before continue (rerun from st.start = 70)!")

    if st(st_base + 60, f"Save waypoints/routes from _manually_ prepared {file_tracks} to hdf5"):  # False: #
        gpx2h5([
            "",
            "--path", str(wf_cfg.path_cruise / rf"navigation\{file_tracks}"),
            "--table_prefix", r"navigation/sectionsCTD",
            "--b_sort", "False",
        ])  # need copy result from navigation\{wf_cfg.path_db} manually, todo: auto copy
        st.go = (
            False,
            f"Hey! copy result from navigation/{wf_cfg.path_db} _manually_ before continue (rerun from st.start = {st_base + 70})!",
        )

    if st(st_base + 70, "Gridding"):  # and False: #
        # Note: Prepare veusz "zabor" pattern before
        grid2d_vsz([
            "cfg/grid2d_vsz.ini",
            "--db_path", str(wf_cfg.path_db),
            "--table_sections", f"{device}/sectionsCTD_routes",  # navigation/
            "--subdir", "CTD-sections",
            "--begin_from_section_int", "4",  # '1',  # values <= 1 means no skip
            "--data_columns_list", "Temp, Sal, SigmaTh, O2, O2ppm, soundV",
            # 'Eh, pH',  todo: N^2 - need calc before
            "--max_depth", "150",  # '250',
            "--filter_depth_wavelet_level_int", "4",  # 4, 5, 5, 4, 6, 4, 4, 5
            "--convexing_ctd_bot_edge_max", "95",  # Depth where we may not reach bot (40 set < bottom because it is harder to recover than delete?)
            # '--x_resolution', '0.2',
            # '--y_resolution', '5',
            "--depecho_add_float", "0",
            "--dt_search_nav_tolerance_seconds", "120",
            "--symbols_in_veusz_ctd_order_list",
            "'Triangle, Green', 'Diamond, Blue', 'Triangle, Red', 'Square, Green'",
            "--b_temp_on_its90", "True",  # modern probes
            "--blank_level_under_bot", "-220",
            "--b_reexport_images", "True"
        ])

        # todo: bug: bad top and bottom edges

    if st(st_base + 100, "Export csv with some new calculated parameters"):  # False: #
        ctd_calc([  # 'ctd_calc-find_runs.ini',
            "--db_path", str(wf_cfg.path_db),
            "--tables_list", f"{device}",
            "--tables_log", "{}/logRuns",
            # '--min_samples', '99',  # fs*depth/speed = 200: if fs = 10Hz for depth 20m
            # '--min_dp', '9',
            # '--b_keep_minmax_of_bad_files', 'True',
            "--path_csv", str(wf_cfg.path_cruise / device / "txt_processed"),
            "--data_columns_list", "Pres, Temp, Cond, Sal, O2, O2ppm, SA, sigma0, depth, soundV",  # , pH, Eh, Lat, Lon
            "--b_incremental_update", "True",
            # todo: check it. If False need delete all previous result of ctd_calc() or set min_time > its last log time
            "--out.tables_list", "None",
        ])

    if st(st_base + 105, "Export csv for Obninsk"):
        m = re.match(r"[\d_]*(?P<abbr_cruise>[^\d]*)(?P<i_cruise>.*)", wf_cfg.path_cruise.name)
        i_cruise = int(m.group("i_cruise"))
        text_file_name_add = f"E090005O2_{m.group('abbr_cruise')}_{i_cruise}_H10_"
        gpx_names_fun_str = "\"df.fileName_st.str.rsplit('st', n=1).str[-1]\""  # .cat(prefix, how='left')
        # gpx_names_fun_str ='"df.fileName_st.str.split(\'/\', n=1).str[-1]"'
        h5tocsv([
            f'input.db_path="{wf_cfg.path_db}"',
            f'input.tables=["{device}"]',
            f'input.tables_log=["{device}/logRuns"]',
            rf"out.text_path='{wf_cfg.path_cruise / device / 'txt_for_Obninsk'}'",
            'out.text_date_format="%Y-%m-%dT%H:%M:%S"',
            'out.text_float_format="%.6g"',
            f"out.file_name_fun=\"f'{text_file_name_add}{{i+1:0>2}}.csv'\"",
            f"out.file_name_fun_log=\"'{text_file_name_add}POS.csv'\"",
            # rec_num;identific;station;Pres;Temp;cond;Sal;O2%;O2ppm;sigmaT;soundV
            # f'out.station_fun={gpx_names_fun_str}',
            f'+out.cols_log={{rec_num: "@i + 1", identific: "@i + 1", station: {gpx_names_fun_str}, '
            f"LONG: Lon_st, LAT: Lat_st, DATE: index}}",  # station: "[out_col_station_fun(ii) for ii in i]"
            "".join([
                '+out.cols={rec_num: "@i + 1", identific: "@i_log + 1", station: "@df_log.station.iat[@i_log]", ',
                ", ".join([
                    p if ":" in p else f"{p}: {p}" for p in "Pres;Temp;Cond;Sal;O2;O2ppm;sigma0;soundV".split(";")
                ]),  # Temp:Temp90;SigmaT;SoundVel
                "}",
            ]),
            'out.sep=";"',
        ])  # , out_col_station_fun=gpx_names_fun
