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

# my funcs
from utils.init import st
from hdf5_pandas.csv2h5 import main as csv2h5
from hdf5_pandas.gpx2h5 import main as gpx2h5
from ctd import wf_cfg, wf_ctd_sst48mc, wf_ctd_saiv

st_base = 200             # 10: SST, 200: SAIV
st.start = st_base + 105  # 10: Extract runs; 20: Draw profiles; 100, 105: Export csv
st.end = st_base + 105   # 300  # st.start
st.go = True  # False?

wf_cfg.path_cruise = path_cruise = Path(r"D:\Cruises\BalticSea\251201_ABP64")
wf_cfg.path_db = (path_cruise / path_cruise / path_cruise.name.split("@", 1)[0]).with_suffix(".h5")

wf_cfg.min_coord = 'Lat:53, Lon:18.6'  # 10
wf_cfg.max_coord = 'Lat:60.55, Lon:30.3'  # includes Gulf Of Finland

# separate cruise number digits
wf_cfg.cruise = re.match(r"(?P<year>\d\d)\d+_*(?P<vessel>\D+)(?P<num>\d+)", path_cruise.stem).groupdict()
wf_cfg.devices = {}


# %% Save navigation to DB
if st(1, "Save gpx navigation to DB"):
    for folder in ["_raw"]:
        gpx2h5([  # '',
            "--db_path",
            str(wf_cfg.path_db),
            "--path",
            str(path_cruise / "navigation" / folder / f"{wf_cfg.cruise['year']}*.gpx"),
            "--tables_list",
            ",navigation,",  # skip waypoints
            "--table_prefix",
            r"",
            # '--b_search_in_subdirs', if set True (to get rid of this loop) then will be problems with overlapped data files
            # '--min_date', '2019-07-17T14:00:00',
            "--min_dict",
            f"{wf_cfg.min_coord}",  # use at least -32768 to replace it by NaN
            "--max_dict",
            f"{wf_cfg.max_coord}",
            "--corr_time_mode",
            "False",  #'delete_inversions?',
            # '--b_incremental_update', '0',  # '1' coerce to delete data loaded in same table in previous steps (only if previous same log file detected?)
            "--b_interact",
            "0",
            # '--b_remove_duplicates', '1',  # not allowed and not need: does always
        ])



# %% Save device data to DB
##############################################################################################################

def mmol2mg(x):
    """'mmol m-3' to 'mg l-1'"""
    return x * 0.0319988


def DO(O2ppm, Sal, Temp, Pres, Lat=55.3, Lon=19.7):
    # default coords Lat=55.3, Lon=19.7 are mean for calibration data
    SA = gsw.SA_from_SP(Sal, Pres, lat=Lat, lon=Lon)  # Absolute Salinity  [g/kg]
    conservative_temperature = gsw.conversions.CT_from_t(SA, Temp, Pres)
    pt = gsw.pt_from_CT(SA, conservative_temperature)
    Solubility = mmol2mg(gsw.O2sol_SP_pt(Sal, pt))
    DO = O2ppm * 100 / Solubility
    return DO


def b_between(t, t_min, t_max):
    t = t[0]
    return (t_min <= t)&(t < t_max)

st_base = 10
if any(st(s, 'SST48') for s in list(range(st_base, st_base+190, 5))):

    def do_polyval_time_sst48mc(x, t):
        # Coef from ABP64 Winkler data obtained from and for:
        # - data before GoF, use also for data after GoF
        poly_noGoF = [1.1133, 0.51236]
        # - data in GoF (2025-12-11T00:00:00 - 2025-12-17T00:00:00)
        poly_GoF = [1.2154, 1.1213]

        b_GoF = b_between(t, *np.array(["2025-12-11T00:00:00", "2025-12-17T00:00:00"], "M8[s]"))

        return np.polyval(poly_GoF if b_GoF else poly_noGoF, x)


    common_ctd_params_list = [
        "--db_path",
        str(wf_cfg.path_db),
        "--min_dict",
        f"Cond:0.5, Sal:0.2, O2:-2, O2ppm:-2",  # deletes zeros & strange big negative values  # SigmaT:2,
        "--max_dict",
        f"O2:200, O2ppm:20",  # , {max_coord} for REDAS-like data
    ]

    wf_ctd_sst48mc.proc(
        common_ctd_params_list,
        o2_fun=lambda O2ppm, Sal, Temp, Pres, Time: DO(do_polyval_time_sst48mc(O2ppm, Time), Sal, Temp, Pres),
        o2ppm_fun=lambda O2ppm, Time: do_polyval_time_sst48mc(O2ppm, Time),
        st_base=st_base
    )

st_base = 200
if any(st(s, 'SAIV') for s in list(range(st_base, st_base+200, 5))):
    common_ctd_params_list = [
        "--db_path", str(wf_cfg.path_db),
        "--min_dict", "Sal:0.1",
    ]

    # def do_polyval_time_saiv(x, t):
    #     # Coef from ABP64 Winkler data obtained from and for:
    #     # - data before GoF, use also for data after GoF
    #     poly_noGoF = [1.1133, 0.51236]
    #     # - data in GoF (2025-12-11T00:00:00 - 2025-12-17T00:00:00)
    #     poly_GoF = [1.2154, 1.1213]

    #     b_GoF = b_between(t, *np.array(["2025-12-11T00:00:00", "2025-12-17T00:00:00"], "M8[s]"))

    #     return np.polyval(poly_GoF if b_GoF else poly_noGoF, x)

    wf_ctd_saiv.proc(
        common_ctd_params_list,
        # o2_fun=lambda O2ppm, Sal, Temp, Pres, Time: DO(do_polyval_time_saiv(O2ppm, Time), Sal, Temp, Pres),
        # o2ppm_fun=lambda O2ppm, Time: do_polyval_time_saiv(O2ppm, Time),
        st_base=st_base
    )

##############################################################################################################

if st(120, 'Meteo'):
    csv2h5([
        'cfg/csv_meteo.ini', '--path',  # hdf5_pandas/
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
    # sys.argv[0]= argv0   os_path.join(os_path.dirname(file_h5_to_gpx)
    h5_to_gpx([
        "cfg/h5_to_gpx_nav_all.ini",
        "--db_path", str(wf_cfg.path_db),
        "--tables_list", "navigation",
        "--simplify_tracks_error_m_float", "10",
        "--period_files", "D",
        "--tables_log_list",
        '""',
        # '--select_from_tablelog_ranges_index', None - defaut
    ])
