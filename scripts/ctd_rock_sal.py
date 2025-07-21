# Andrey Korzh, 22.11.2023
import sys
from pathlib import Path
# my funcs
from utils2init import LoggingStyleAdapter
from to_vaex_hdf5.h5tocsv import ctd2csv
sys.path.append(r'C:\Work\Python\AB_SIO_RAS\tcm')
from tcm.csv_load import load_from_csv_gen
from tcm.csv_specific_proc import loaded_rock, century
from tcm.filters import longest_increasing_subsequence_i
lf = LoggingStyleAdapter(__name__)

# def sigma_th0(Sal, Temp90, Pres):
#     """
#     :param Temp90: temperature ITS90
#     :param Sal: practical salinity
#     :param Pres:
#     :return: array-like, kg/m^3
#         potential density anomaly with
#         respect to a reference pressure of 0 dbar,
#         that is, this potential density - 1000 kg/m^3.
#     """
#     lat = 77.30  # 7730.07885328,N,06634.94560096,E
#     lon = 66.35
#     SA = gsw.SA_from_SP(Sal, Pres, lat=lat, lon=lon)
#     CT = gsw.CT_from_t(SA, Temp90, Pres)
#     sigma0 = gsw.sigma0(SA, CT)
#     return sigma0

def ctd_rock_sal(cfg_in, cfg_out, lat, lon):
    """
    Load ROCK CTD text data of table format: "-0.0013 3.8218 0.1531 2023-11-21 11:53:22".
    Adds Sal, SigmaTh0.
    Saves to tab separated values files named by start and end data time.
    :return:
    """
    
    paths_csv_prev = None
    for itbl, pid, paths_csv, df_raw in load_from_csv_gen(cfg_in):
        if paths_csv_prev != paths_csv:
            paths_csv_prev = paths_csv
            csv_part = 0
        else:
            csv_part += 1  # next part of same csv
        # todo: append previous file if next part is of same source file (if csv_part > 0)
        
        n_rows = len(df_raw)
        ind = longest_increasing_subsequence_i(df_raw.Pres.to_numpy())
        df_raw = df_raw.iloc[ind, :]
        lf.info(
            f'Removed {n_rows - len(df_raw)} rows of non-increased Pressure '
            f'in possible shortest intervals. Remains {len(df_raw)} rows'
        )
        
        ctd2csv(
            df_raw,
            cfg_out,
            df_log_csv=None,
            lat=lat, lon=lon,
            log_row=None, i_log_row=None, tbl=None
        )


if __name__ == '__main__':
    path_cruise = Path(r'd:\WorkData\KaraSea\231110_AMK93')
    device = 'CTD_ROCK'
    cfg_in = {  # ROCK_AMK_[0-9][0-9][0-9][0-9]_*
        'path':               path_cruise / device / r'_raw\[0-9]*.TXT',
        'fun_proc_loaded':    loaded_rock,
        'csv_specific_param': {
            # 'Pres_fun': lambda x: np.polyval([100, 0], x),
            # 'Sal_fun': lambda Cond, Temp90, Pres: gsw.SP_from_C(Cond, Temp90, Pres),  # not adds col!
            # 'SigmaTh_fun': lambda Sal, Temp90, Pres: sigma_th0(Sal, Temp90, Pres)     # not adds col!
        },
        'text_line_regex':    b'^(?P<use>' + b'(\-?\d{1,4}\.\d{1,4} ){3}' + century + br'\d{2}-\d{1,2}-\d{1,2} \d{1,2}:\d{1,2}:\d{1,2}).*',
        'header':             'Pres Temp90 Cond Date(text) Time(text)',
        'delimiter':          ' '
        # 'dtype':
    }
    # CTD data collected near region with nav info: 7730.07885328,N,06634.94560096,E
    lat = 77.30
    lon = 66.35
    
    # min_coord = 'Lat:53, Lon:18.6'  # 10
    # max_coord = 'Lat:80.55, Lon:30.3'
    
    cfg_out = {
        'cols':              ['Pres', 'Temp90', 'Cond', 'Sal', 'sigma0'],
        'text_path':         path_cruise / device,
        'file_name_fun':     (lambda i_log, t_st, t_en, tbl: f'{t_st:%y%m%d_%H%M}-{t_en:%H%M}.tsv'),
        'text_date_format':  "%Y-%m-%d %H:%M:%S",
        'text_float_format': "%.4f",
        'sep':               '\t'
    }
    
    ctd_rock_sal(cfg_in, cfg_out, lat, lon)
