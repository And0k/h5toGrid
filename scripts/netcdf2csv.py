#!/usr/bin/python3.7+
"""
Convert NetCDF file of ERA5 data that was downloaded by ecmwf_api_client.py to csv
u10   10 metre U wind component
v10   10 metre V wind component
fg10  10 metre wind gust since previous post-processing
i10fg Instantaneous 10 metre wind gust
mwd   Mean wave direction
mwp   Mean wave period (Tm-1)
pp1d  Peak wave period
swh   Significant height of combined wind waves and swell (Hs)
"""
import netCDF4
from netCDF4 import num2date
import numpy as np
from pathlib import Path
import pandas as pd
import re
from itertools import tee
import difflib

# netCDF4 file
file_path = \
r'd:\workData\BalticSea\201202_BalticSpit\inclinometer\processed_h5,vsz\wind@ECMWF-ERA5_area(54.615,19.841,54.615,19.841).nc'
# d:\workData\BalticSea\201202_BalticSpit\inclinometer\processed_h5,vsz/wind@ECMWF-ERA5_area(54.9689,20.2446,54.9689,20.2446).nc'

file_path = Path(file_path)
output_dir = file_path.parent
method = 'file_for_each_coord'  # {'file_for_each_time', 'one_file', 'file_for_each_coord'}
variables = ['u10', 'v10', 'i10fg', 'fg10', 'mwd', 'mwp', 'pp1d', 'swh']

var_short_names = variables
# {
# 'u10': 'u10',
# 'v10': 'v10',
# 'fg10': 'fg10',
# 'i10fg': 'i10fg',
# 'mwd': 'mwd',
# 'mwp': 'mwp',
# 'swh': 'Hs'
#     }
# variables_long = ['10m_u_component_of_wind', '10m_v_component_of_wind']
#
# def pairwise(iterable):
#     """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
#     a, b = tee(iterable)
#     next(b, None)
#     return zip(a, b)
#
#
# var_short_names = []
# for v_prev, v_next in pairwise(variables):
#     var_short_names.extend([li[2:] for li in difflib.ndiff(v_prev, v_next) if li[0] != ' '])

# Open netCDF4 file
f = netCDF4.Dataset(file_path)
# Extract variable
t2m = f.variables[variables[0]]

# Get dimensions assuming 3D: time, latitude, longitude
dims_list = t2m.get_dims()
dims = {d.name: d.size for d in t2m.get_dims()}
time_name, *expver_name, lat_name, lon_name = dims.keys()
time_var = f.variables[time_name]
times = num2date(time_var[:], time_var.units, only_use_cftime_datetimes=False, only_use_python_datetimes=True)
latitudes = f.variables[lat_name][:]
longitudes = f.variables[lon_name][:]


if method == 'file_for_each_time':      # METHOD 1
    # Extract each time as a 2D pandas DataFrame and write it to CSV
    Path.mkdir(output_dir, parents=True, exist_ok=True)
    for v in variables:
        for i, t in enumerate(times):
            filename = output_dir / f'{t.isoformat()}{v}.csv'
            print(f'Writing time {t} to {filename}')
            df = pd.DataFrame(f.variables[v][i, :, :], index=latitudes, columns=longitudes)
            df.to_csv(filename)
elif method == 'one_file':              # METHOD 2
    # Write data as a table with 4+ columns: time, value0, value1... for each (latitude, longitude)
    filename = output_dir / f'{times[0]:%y%m%d_%H%M}{file_path.stem}.csv'
    print(f'Writing data in tabular form to {filename} (this may take some time)...')
    times_grid, latitudes_grid, longitudes_grid = [
        x.flatten() for x in np.meshgrid(times, latitudes, longitudes, indexing='ij')]
    df = pd.DataFrame({
        'Time': times_grid,
        'Lat': latitudes_grid,
        'Lon': longitudes_grid,
        **{n: f.variables[v][:].flatten() for n, v in zip(var_short_names, variables)}
        })
    df.to_csv(filename, index=False)
elif method == 'file_for_each_coord':
    # We will construct filename from date, source name excluding source area coordinates info, and point's coordinates
    filename_part_time = f'{times[0]:%y%m%d_%H%M}'.replace('_0000', '')
    filename_part_source= re.sub('_?area\([^)]*\)', '', file_path.stem)
    for ilat, lat in enumerate(latitudes):
        for ilon, lon in enumerate(longitudes):
            var_dict = {}
            for n, v in zip(var_short_names, variables):
                if expver_name:
                    vv = f.variables[v]
                    if not (vv[:, :, ilat, ilon].mask.astype(np.int8).sum(axis=1) == 1).all():
                        raise NotImplementedError
                    var_dict[n] = vv[:, :, ilat, ilon].compressed()
                else:
                    var_dict[n] = f.variables[v][:, ilat, ilon]

            df = pd.DataFrame({'Time': times, **var_dict})

            filename = output_dir / f'{filename_part_time}{filename_part_source}(N{lat:.5g},E{lon:.5g}).tsv'
            df.to_csv(filename, index=False, date_format='%Y-%m-%d %H:%M', float_format='%.5g',
                     sep='\t', encoding="ascii")

print('saved to', filename)