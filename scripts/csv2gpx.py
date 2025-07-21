from pathlib import Path
from typing import Any, Dict, Mapping, Union
import numpy as np
import pandas as pd
from pyproj import Proj

import gpxpy
from pandas.tseries.frequencies import to_offset
from gpxpy.gpx import GPX

from h5toGpx import save_to_gpx

from to_pandas_hdf5.gpx2h5 import df_rename_cols  # gpxConvert
from to_pandas_hdf5.h5_dask_pandas import h5_append, df_to_csv
from re import search

# Input csv data
path_csv = Path(
    r'd:\WorkData\_experiment\tracker\240306_Devau\_raw\Topcon_GR-5.txt'
)
device = 'Topcon GR-5'

# Automatic output file name
add_sfx = f"@{device.replace(' ', '_')}" if device else ''
cfg = {
    'out': {
        'path': (  # time_st will be replaced to datastart
            path_csv.parent if path_csv.parent.stem == '_raw' else path_csv
            ).with_name(f'{{time_st:%y%m%d}}{add_sfx}').with_suffix('.gpx')  
    }}


def waypoint_symbf(row) -> str:
    return 'Diamond, Red' if row.Name[0].isdigit() else 'Triangle, Blue'

def gpx_obj_namef(i_row, row, it) -> str:
    return row.Name

# see also loaded_nav_Dudkov_HydroProfiles()
def utm34to_latlon(
        nav_df: Union[pd.DataFrame, np.ndarray],
        cfg: Mapping[str, Any]=None,
        gpx_obj_namef=gpx_obj_namef, waypoint_symbf=waypoint_symbf, gpx=None
        ) -> pd.DatetimeIndex:
    """
    Specified prep&proc of Dudkov\tr_ABP049_000-078_1sec_processed_HydroProfiles:
    - Time calc: time is already UTC in ISO format
    - X, Y WGS84_UTM34N to Lat, Lon degrees conversion

    :param a: numpy record array. Will be modified inplace.
    :param cfg: dict
    :return: numpy 'datetime64[ns]' array


    Example input:
    a = {
    'date': "02:13:12.30", #'Time'
    'Lat': 55.94522129,
    'Lon': 18.70426069,
    'Depth': 43.01}
    """

    p = Proj(proj='utm', zone=34, ellps='WGS84', preserve_units=False)
    nav_df['Lon'], nav_df['Lat'] = p(*nav_df[['Lon', 'Lat']].values.T, inverse=True)  # inverse transform
    if 'Time' in nav_df.columns:
        nav_df['Time'] = np.array(nav_df['Time'].values, 'M8[ns]')
    else:
        # time_st = pd.to_datetime(search('\d+', path_csv.stem).group(0), dayfirst=False, yearfirst=True, format='%y%m%d', utc=True)  # %H%M
        # nav_df['Time'] = time_st + pd.to_timedelta(nav_df.index, 's')
        time_st = pd.to_datetime('2024-03-06T10:20:00', format="ISO8601", utc=True)
        time_rel = nav_df.Name.where(nav_df.Name.str.isdigit(), np.nan).astype(float)
        period = 2  # sec
        nav_df['Time'] = time_st + pd.Series(np.array(period * (time_rel - time_rel.min()), 'm8[s]'))
        # pd.date_range(time_st, periods=len(nav_df), freq='2s')  # if equal intervals and no index
    nav_df.set_index('Time', inplace=True)
    # date = pd.to_datetime(a.loc[uniq, 'Time'].str.decode('utf-8', errors='replace'), format='%d.%m.%YT%H:%M:%S.%F')
    # 'DepEcho'
    
    for wp in [False, True]:
        if not wp:
            # track rows
            b = nav_df['Name'].str.isdigit()
            if not b.any():
                continue
            # not save gpx but return track to save on next iteration
            wp_args = {'fileOutPN': None, 'gpx_obj_namef': device}  
            print(f'saving {b.sum()} track rows', end=', ')
        else:
            # waypoints rows
            b = ~b
            if b.any():
                
                # replace file name time_st pattern with value
                p_out = cfg['out']['path']
                p_out = p_out.with_name(p_out.name.format(time_st=nav_df.index.min()))

                wp_args = {
                    'gpx_obj_namef': gpx_obj_namef,
                    'waypoint_symbf': waypoint_symbf,
                    'fileOutPN': p_out
                }
                print(f'saving {b.sum()} waypoints')
            else:
                print()
        gpx = save_to_gpx(
            nav_df.loc[b, :].dropna(subset=['Lat', 'Lon']),  # *.gpx will be not compatible to GPX if it will have NaN values
            cfg_proc=cfg.get('proc', None), gpx=gpx,
            **wp_args
        )


if __name__ == '__main__':
    print(pd.Timestamp.now(), '> Starting', Path(__file__).stem)
    nav_df = pd.read_csv(path_csv, sep='\t')  # , parse_dates=['Time'], index_col='Time'

    utm34to_latlon(nav_df, gpx_obj_namef=gpx_obj_namef, waypoint_symbf=waypoint_symbf, cfg=cfg)  #.rename(columns=dict(zip(col_in, col_out)))
    print('Ok >')