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
sp    surface_pressure
"""

import netCDF4
from netCDF4 import num2date
import numpy as np
from pathlib import Path
import pandas as pd
from typing import Any, Callable, Dict, Iterator, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple, List, Union, TypeVar
import re
# from itertools import islice
# import difflib


def main(file_path: Union[Path, str],
         output_dir: Optional[Path]=None,
         method: str='file_for_each_coord',
         variables: Optional[Sequence[str]]=None,
         var_short_names: Optional[Sequence[str]]=None):
    """

    :param file_path:
    :param output_dir:
    :param method: 'file_for_each_coord'  # {'file_for_each_time', 'one_file', 'file_for_each_coord'}
    :param variables:
    :param var_short_names:
    :return:
    """

    file_path = Path(file_path)
    if output_dir is None:
        output_dir = file_path.parent

    # Open netCDF4 file
    f = netCDF4.Dataset(file_path)

    if variables is None:
        itim = list(f.variables.keys()).index('time')
        variables = list(f.variables.keys())[(itim+1):]  # not in first 'longitude', 'latitude', 'expver', 'time',
        # ['u10', 'v10', 'i10fg', 'fg10', 'mwd', 'mwp', 'pp1d', 'swh', 'sp']
        del variables[variables.index('fg10')]
        del variables[variables.index('i10fg')]

    if var_short_names is None:
        var_short_names = variables

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
        # To construct filename from date, source name excluding source area coordinates info, and point's coordinates:
        filename_part_time = f'{times[0]:%y%m%d_%H%M}'.replace('_0000', '')
        filename_part_source= re.sub('_?area\([^)]*\)', '', file_path.stem)
        nvars_ok_cum = np.ones(dims[time_name], dtype=bool)
        for ilat, lat in enumerate(latitudes):
            for ilon, lon in enumerate(longitudes):
                var_dict = {}
                for n, v in zip(var_short_names, variables):
                    if expver_name:
                        vv = f.variables[v]
                        nvars_ok = np.ma.count_masked(vv[:, :, ilat, ilon], axis=1) == 1
                        if not nvars_ok.all():
                            mask = ~nvars_ok
                            nvars_ok_cum &= nvars_ok
                            print(f'{mask.sum()} data is bad (first 3 indexes: {np.flatnonzero(mask)[0:3]}...) - ignored')
                            # unmasking to keep shape on next op. (and delete later):
                            vv_no_bad_mask = vv[:, :, ilat, ilon]
                            vv_no_bad_mask[mask, :] = np.ma.masked_array([np.ma.masked]*(vv.shape[1] -1) + [np.NaN])
                            var_dict[n] = vv_no_bad_mask.compressed()
                    else:
                        var_dict[n] = f.variables[v][:, ilat, ilon]
                if not nvars_ok_cum.all():
                    var_dict = {k: v[nvars_ok_cum] for k, v in var_dict.items()}
                    df = pd.DataFrame({'Time': times[nvars_ok_cum], **var_dict})
                else:
                    df = pd.DataFrame({'Time': times, **var_dict})

                filename = output_dir / f'{filename_part_time}{filename_part_source}(N{lat:.5g},E{lon:.5g}).tsv'
                df.to_csv(filename, index=False, date_format='%Y-%m-%d %H:%M', float_format='%.5g',
                         sep='\t', encoding="ascii")
    else:
        print(f'Unknown method: {method}.', 'Set one of:',
              ', '.join('file_for_each_coord', 'file_for_each_time', 'one_file', 'file_for_each_coord')
              )

    print('saved to', filename)


if __name__ == '__main__':
    #
    main(file_path=r'd:\workData\BalticSea\201202_BalticSpit_inclinometer\wind@ECMWF-ERA5_area(54.615,19.841,54.615,19.841).nc'
         )         # d:\workData\BalticSea\201202_BalticSpit_inclinometer\wind@ECMWF-ERA5_area(54.615,19.841,54.615,19.841).nc
#
# r'd:\workData\BalticSea\201202_BalticSpit\inclinometer\processed_h5,vsz\wind@ECMWF-ERA5_area(54.615,19.841,54.615,19.841).nc'
# d:\workData\BalticSea\201202_BalticSpit\inclinometer\processed_h5,vsz/wind@ECMWF-ERA5_area(54.9689,20.2446,54.9689,20.2446).nc'

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