#!/usr/bin/env python
"""
Download ECMWF ERA5 data from Climate Data Store to NetCDF file
See also
 https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation
 - netcdf2csv.py to convert result to csv
 - download_copernicus.py
"""
from io import IOBase
from os import SEEK_END, SEEK_SET
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple, List, Union, TypeVar
from datetime import datetime
import pandas as pd

from scripts.netcdf2csv import main as netcdf2csv

# Set:
# dir_save
# lat_st, lon_st
# use_date_range - Set [] to load from last loaded data to now
dir_save, lat_st, lon_st, use_date_range = r'd:\WorkData\KaraSea\220906_AMK89-1\meteo', 72.33385, 63.53786, ['2022-09-01', '2022-09-15']


# r'e:\WorkData\BalticSea\181005_ABP44\meteo', 55.88333, 19.13139, ['2018-10-01', '2019-01-01']
# r'd:\WorkData\BalticSea\220601_ABP49\meteo'
# r'd:\workData\BalticSea\201202_BalticSpit_inclinometer', 54.615, 19.841  ## Pionersky: 54.9689, 20.2446
# 55.32659, 20.57875  # 55.874845000, 19.116386667  # masha: 54.625, 19.875 #need: N54.61106 E19.84847
# ['2022-05-01', '2022-05-20']
# ['2022-06-01', '2022-06-23']  # ['2020-12-01', '2021-01-31']  # ['2018-12-01', '2018-12-31'], ['2020-09-01', '2021-09-16']

dGrid = 0  # 0.25
lat_en = lat_st + dGrid
lon_en = lon_st + dGrid



file_nc_name = f'wind@ECMWF-ERA5_area({lat_st},{lon_st},{lat_en},{lon_en}).nc'

l = logging.getLogger(__name__)
l.info(f'Downloading {file_nc_name}: to {dir_save}...')

# Get interval from last data timestamp we have to now

class ReverseFile(IOBase):
    """
    Edited source from https://stackoverflow.com/a/51750850/2028147
    An example
    rev = ReverseFile(filename)
    for i, line in enumerate(rev):
            print("{0}: {1}".format(i, line.strip()))
    """
    def __init__ (self, filename, headers=0, **kwargs):
        """

        :param filename:
        :param headers:
        :param kwargs: args to call open(filename, **kwargs)
        """
        self.fp = open(filename, **kwargs)
        self.headers = headers
        self.reverse = self.reversed_lines()
        self.end_position = -1
        self.current_position = -1

    def readline(self, size=-1):
        if self.headers > 0:
            self.headers -= 1
            raw = self.fp.readline(size)
            self.end_position = self.fp.tell()
            return raw

        raw = next(self.reverse)
        if self.current_position > self.end_position:
            return raw

        raise StopIteration

    def reversed_lines(self):
        """Generate the lines of file in reverse order.
        """
        part = ''
        for block in self.reversed_blocks():
            block = block + part
            block = block.split('\n')
            block.reverse()
            part = block.pop()
            if block[0] == '':
                block.pop(0)

            for line in block:
                yield line + '\n'

        if part:
            yield part

    def reversed_blocks(self, blocksize=0xFFFF):
        """Generate blocks of file's contents in reverse order.
        """
        file = self.fp
        file.seek(0, SEEK_END)
        here = file.tell()
        while 0 < here:
            delta = min(blocksize, here)
            here -= delta
            file.seek(here, SEEK_SET)
            self.current_position = file.tell()
            yield file.read(delta)


if not use_date_range:
    with ReverseFile(r'd:\workData\BalticSea\201202_BalticSpit_inclinometer\wind\200901wind@ECMWF-ERA5(N54.615,E19.841).tsv',
              encoding='ascii') as f_prev:
        last_line = next(f_prev)
    print('last ECMWF data found:', last_line)
    use_date_range = [last_line.split()[0], f'{datetime.utcnow():%Y-%m-%d}']
elif len(use_date_range) <= 1 or not use_date_range[1]:
    use_date_range = [use_date_range[1] or '2020-09-01', f'{datetime.utcnow():%Y-%m-%d}']

date_range = [datetime.strptime(t, '%Y-%m-%d') for t in use_date_range]  # T%H:%M:%S
file_date_prefix = '{:%y%m%d}-{:%m%d}'.format(*date_range)

l.info('Downloading interval {} - {}...', *date_range)
# common = {
#     'class': 'ei',
#     'dataset': 'interim',
#     'date': '{}/to/{}'.format(*use_date_range),
#     'expver': '1',
#     'grid': '0.75/0.75',
#     'area': '{}/{}/{}/{}'.format(lat_st, lon_st, lat_en, lon_en),  # SWSE
#     'levtype': 'sfc',
#     'param': '165.128/166.128',
#     'stream': 'oper',
#     'format': 'netcdf'
#     }

try:
    # European Centre for Medium-Range Weather Forecasts (ECMWF)
    # retrieve of ERA5 data from Climate Data Store
    # https://github.com/ecmwf/cdsapi
    #
    # Code requires {UID}:{API Key} is in my C:\Users\{USER}\.cdsapirc

    import cdsapi
    import urllib3
    import requests

    # manager = PoolManager(num_pools=2)

    proxies = {
        'http': 'http://127.0.0.1:28080',
        'https': 'http://127.0.0.1:28080',
        'ftp': 'http://127.0.0.1:28080'
        }

    # InsecureRequestWarning: Unverified HTTPS request is being made to host 'cds.climate.copernicus.eu'
    with requests.Session() as session:
        # session.proxies.update(proxies)  # not works

        c = cdsapi.Client(session=session)  # warning_callback=
        urllib3.disable_warnings()  # prevent InsecureRequestWarning... Adding certificate verification is strongly advised.  # better use warning_callback=?
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'time': [f'{i:02d}:00' for i in range(24)],
                'date': [f'{d}' for d in pd.period_range(*use_date_range, freq='D')],
                'variable': [
                    '10m_u_component_of_wind',
                    '10m_v_component_of_wind',
                    '10m_wind_gust_since_previous_post_processing',
                    'instantaneous_10m_wind_gust',
                    'mean_wave_direction',
                    'mean_wave_period',
                    'peak_wave_period',
                    'significant_height_of_combined_wind_waves_and_swell',
                    'surface_pressure'
                    ],
                'area': [lat_st, lon_st, lat_en, lon_en],
                # 'year': '2021',
                # 'month': ['01', '02'],
                # 'day' : [f'{i:02d}' for i in range(1, 32)],
            },
            (path_nc := (Path(dir_save) / file_nc_name))
        )

    netcdf2csv(path_nc)
except Exception as e:
    raise(e)

if False:  # old
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
