#!/usr/bin/env python
"""
Download ERA5 data from Climate Data Store to NetCDF file
See also netcdf2csv.py to convert result to csv
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple, List, Union, TypeVar
from datetime import datetime
import pandas as pd

from scripts.netcdf2csv import main as netcdf2csv

#masha: 54.625    19.875
#need: N54.61106 E19.84847
# lat_st, lon_st = 54.9689, 20.2446  # Pionersky
lat_st, lon_st = 54.615, 19.841  # Baltic_spit

dGrid = 0  # 0.25
lat_en = lat_st + dGrid
lon_en = lon_st + dGrid
dir_save = r'd:\workData\BalticSea\201202_BalticSpit_inclinometer'
file_nc_name = f'wind@ECMWF-ERA5_area({lat_st},{lon_st},{lat_en},{lon_en}).nc'

l = logging.getLogger(__name__)
l.info(f'Downloading {file_nc_name}: to {dir_save}...')


use_date_range_str = ['2020-09-01', '2021-09-16']  # ['2020-12-01', '2021-01-31']  # ['2018-12-01', '2018-12-31']
use_date_range = [datetime.strptime(t, '%Y-%m-%d') for t in use_date_range_str]  # T%H:%M:%S
file_date_prefix = '{:%y%m%d}-{:%m%d}'.format(*use_date_range)

# common = {
#     'class': 'ei',
#     'dataset': 'interim',
#     'date': '{}/to/{}'.format(*use_date_range_str),
#     'expver': '1',
#     'grid': '0.75/0.75',
#     'area': '{}/{}/{}/{}'.format(lat_st, lon_st, lat_en, lon_en),  # SWSE
#     'levtype': 'sfc',
#     'param': '165.128/166.128',
#     'stream': 'oper',
#     'format': 'netcdf'
#     }

if True:
    # European Centre for Medium-Range Weather Forecasts (ECMWF)
    # retrieve of ERA5 data from Climate Data Store
    # https://github.com/ecmwf/cdsapi
    #
    # Code requires {UID}:{API Key} is in my C:\Users\{USER}\.cdsapirc

    import cdsapi
    import urllib3

    c = cdsapi.Client()
    urllib3.disable_warnings()  # prevent InsecureRequestWarning... Adding certificate verification is strongly advised.
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'time': [f'{i:02d}:00' for i in range(24)],
            'date': [f'{d}' for d in pd.period_range(*use_date_range_str, freq='D')],
            'variable': [
                '10m_u_component_of_wind',
                '10m_v_component_of_wind',
                '10m_wind_gust_since_previous_post_processing',
                'instantaneous_10m_wind_gust',
                'mean_wave_direction',
                'mean_wave_period',
                'peak_wave_period',
                'significant_height_of_combined_wind_waves_and_swell'
                ],
            'area': [lat_st, lon_st, lat_en, lon_en],
            # 'year': '2021',
            # 'month': ['01', '02'],
            # 'day' : [f'{i:02d}' for i in range(1, 32)],
        },
        (path_nc:=(Path(dir_save) / file_nc_name))
    )

    netcdf2csv(path_nc)


if False: # old
    from ecmwfapi import ECMWFDataServer
    server = ECMWFDataServer()
    l.info('part 1')
    server.retrieve({**common, **{
        'step': '0',
        'time': '00:00/06:00/12:00/18:00',
        'type': 'an',
        'target': file_date_prefix + 'analysis.nc',
        }})
    l.info('part 2')
    server.retrieve({**common, **{
        'step': '3/9',  # '3/6/9/12'
        'time': '00:00/12:00',
        'type': 'fc',
        'target': file_date_prefix + 'forecast.nc',
        }})

print('ok')
