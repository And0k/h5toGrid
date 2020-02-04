#!/usr/bin/env python
"""

Your registered email: 	korzh@nextmail.ru
Your API key: 	cfb5a1249d5f4ba65823331dd93dfe95 (valid until March 13, 2020, 5:01 p.m.)
Content of $HOME/.ecmwfapirc

{
    "url"   : "https://api.ecmwf.int/v1",
    "key"   : "cfb5a1249d5f4ba65823331dd93dfe95",
    "email" : "korzh@nextmail.ru"
}
"""

import logging
from datetime import datetime

from ecmwfapi import ECMWFDataServer

l = logging.getLogger(__name__)

server = ECMWFDataServer()

lon = 19.75
lat = 54.75
use_date_range_str = ['2018-11-16', '2018-12-31']  # ['2018-12-01', '2018-12-31']
use_date_range = [datetime.strptime(t, '%Y-%m-%d') for t in use_date_range_str]  # T%H:%M:%S
file_date_prefix = '{:%y%m%d}-{:%m%d}'.format(*use_date_range)

common = {
    'class': 'ei',
    'dataset': 'interim',
    'date': '{}/to/{}'.format(*use_date_range_str),
    'expver': '1',
    'grid': '0.75/0.75',
    'area': '{}/{}/{}/{}'.format(lat, lon, lat, lon),  # SWSE
    'levtype': 'sfc',
    'param': '165.128/166.128',
    'stream': 'oper',
    'format': 'netcdf'
    }
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
