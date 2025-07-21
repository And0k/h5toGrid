#!/usr/bin/python3.7+
"""
Convert NetCDF file of ERA5 data that was downloaded by download_ecmwf.py to csv
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
    if isinstance(file_path, (str, Path)):  # file_path.suffix.lower() != ".zip"
        file_path = Path(file_path)
        if output_dir is None:
            output_dir = file_path.parent
        # Open netCDF4 file
        f = netCDF4.Dataset(file_path)

        if variables is None:
            itim = list(f.variables.keys()).index('time')
            variables = list(f.variables.keys())[(itim+1):]  # not in first 'longitude', 'latitude', 'expver', 'time',
            # ['u10', 'v10', 'i10fg', 'fg10', 'mwd', 'mwp', 'pp1d', 'swh', 'sp']

            # del variables[variables.index('fg10')]
            # del variables[variables.index('i10fg')]

        if var_short_names is None:
            var_short_names = variables

        # Extract variable
        t2m = f.variables[variables[0]]
    else:  # len(file_path) != 1
        # file_path = file_path / "data_stream-oper_stepType-instant.nc"  # todo
        # data_stream-wave_stepType-instant.nc, ...
        vars = []
        for file_p in file_path:
            try:
                f = netCDF4.Dataset(file_path)
            except OSError as e:
                raise (NotImplementedError("multiple hdf5 files not supported"))
            if variables is None:
                itim = list(f.variables.keys()).index("time")
                vars += list(f.variables.keys())[(itim + 1) :]  # not in first 'longitude', 'latitude',
        if variables is None:
            variables = vars
        raise(NotImplementedError("multiple files not supported"))

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
        filename_part_source = re.sub(r'_?area\([^)]*\)', '', file_path.stem)
        for ilat, lat in enumerate(latitudes):
            for ilon, lon in enumerate(longitudes):
                var_dict = {}
                for n, v in zip(var_short_names, variables):
                    if expver_name:
                        vv = f.variables[v]
                        nvars_ok = np.ma.count_masked(vv[:, :, ilat, ilon], axis=1) == 1
                        if not nvars_ok.all():
                            mask = ~nvars_ok
                            print(f'{mask.sum()} data is bad (first 3 indexes: {np.flatnonzero(mask)[0:3]}...) - ignored')
                            # unmasking to keep shape on next op. (and delete later):
                            vv_no_bad_mask = vv[:, :, ilat, ilon]
                            vv_no_bad_mask[mask, :] = np.ma.masked_array([np.ma.masked]*(vv.shape[1] - 1) + [np.nan])
                            var_dict[n] = vv_no_bad_mask.compressed()
                        else:
                            var_dict[n] = vv[:, :, ilat, ilon].compressed()  # vv[:, 0, ilat, ilon]
                    else:
                        var_dict[n] = f.variables[v][:, ilat, ilon]  # old worked: f.variables[v][:, ilat, ilon]
                df = pd.DataFrame({'Time': times, **var_dict})

                filename = output_dir / f'{filename_part_time}{filename_part_source}(N{lat:.5g},E{lon:.5g}).tsv'
                df.to_csv(
                    filename, index=False, date_format='%Y-%m-%d %H:%M', float_format='%.5g', sep='\t', encoding="ascii"
                )
    else:
        print(f'Unknown method: {method}.', 'Set one of:',
              ', '.join('file_for_each_coord', 'file_for_each_time', 'one_file', 'file_for_each_coord')
              )

    print('saved to', filename)


if __name__ == '__main__':
    #
    main(
        file_path=
        r"C:\Work\Veusz\meteo\ECMWF\wind@ECMWF-ERA5_area(54.744417,19.5799,54.744417,19.5799).zip",
        variables= [
            '10m_u_component_of_wind', '10m_v_component_of_wind', 'surface_pressure', 'sea_surface_temperature', 'total_precipitation', '10m_wind_gust_since_previous_post_processing', 'mean_wave_direction', 'mean_wave_period', 'peak_wave_period', 'significant_height_of_combined_wind_waves_and_swell']
    )

    # r'd:/WorkData/BalticSea/220505_D6/meteo/ECMWF/wind@ECMWF-ERA5_area(55.3266,20.5789,55.3266,20.5789).nc'
    # r'd:\WorkData\BalticSea\_Pregolya,Lagoon\231208@i19,ip5,6\meteo\wind@ECMWF-ERA5_area(54.64485,21.07382,54.64485,21.07382).nc'
    # r'd:\WorkData\BalticSea\230507_ABP53\meteo\wind@ECMWF-ERA5_area(55.922656,19.018713,55.922656,19.018713).nc'
    # r'd:\workData\BalticSea\201202_BalticSpit_inclinometer\wind@ECMWF-ERA5_area(54.615,19.841,54.615,19.841).nc'
    # d:\workData\BalticSea\201202_BalticSpit_inclinometer\wind@ECMWF-ERA5_area(54.615,19.841,54.615,19.841).nc
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
