#!/usr/bin/python3.7+
"""
Convert NetCDF data that was downloaded by download_copernicus.py to csv or grd
Input file can be, for example, loaded from dataset-bal-analysis-forecast-phy-monthlymeans dataset and have variables:
uo,vo,mlotst,so,sob

NAME    STANDARD NAME       UNITS
bottomT sea water potential temperature at sea floor, degrees C
mlotst  ocean mixed layer thickness defined by sigma theta, m
siconc  sea ice area fraction   1
sithick sea ice thickness, m
sla     sea surface height above sea level, m
so      sea water salinity * 0.001
sob     sea water salinity at sea floor * 0.001
thetao  sea water potential temperature, degree Celsius
uo      eastward sea water velocity, m/s
vo      northward sea water velocity, m/s
"""
import netCDF4
from netCDF4 import num2date
import numpy as np
from pathlib import Path
import pandas as pd
import re
from collections import defaultdict
from itertools import tee
import difflib


def constant_factory(val):
    def default_val():
        return val
    return default_val

#######################################################################################################################
# netCDF4 file
file_path = \
r'd:\WorkData\BalticSea\_other_data\_model\Copernicus\Sal_bin=1month\201016-bal-analysis-forecast-phy-monthlymeans.nc'

file_path = Path(file_path)
output_dir = file_path.parent
method = 'file_for_each_time'  # {'file_for_each_time', 'one_file', 'file_for_each_coord'}
# put variable with max dimensions first (they will be determined from it)
variables = ['so', 'sob']

# make "x: lambda x" be default value
variables_apply_coef = defaultdict(constant_factory(lambda x: x),
    {'depth': lambda x: -x})  # 'so': lambda x: x*1e3, 'sob': lambda x: x, *1e3

var_short_names = variables
z_isosurface = {'so': [7, 7.5, 8, 8.5]}



# Open netCDF4 file
f = netCDF4.Dataset(file_path)

# Extract dimensions from 1st variable
t2m = f.variables[variables[0]]

# Get dimensions assuming 3D: time, latitude, longitude
dims_list = t2m.get_dims()
dims = {d.name: d.size for d in t2m.get_dims()}
time_name, *expver_name, lat_name, lon_name = dims.keys()
time_var = f.variables[time_name]
times = num2date(time_var[:], time_var.units, only_use_cftime_datetimes=False, only_use_python_datetimes=True)
latitudes = f.variables[lat_name][:]
longitudes = f.variables[lon_name][:]
b_2d_to_txt = False
iso_surface_window = 1  # weighted mean between 2 nearest points


def calc_iso_surface(v3d, v_isosurface, zi, interp_order=1, weight_power=1, dv_max=20) -> np.array:
    """
    weighted average to compute the iso-surface
    :param v3d:
    :param v_isosurface:
    :param zi: 1d array of z values corresponded to its indexes
    :param interp_order: number of points to one side of requested
    :param weight_power:  1 for linear
    :return z2d: 2D array of shape v3d.shape[:-1] of nearest z values
    """

    # Add top & bot edges
    dv = v3d - v_isosurface
    dv_filled = dv.filled(np.nan)
    shape_collect = list(v3d.shape[:-1]) + [interp_order * 2]
    # collect nearest values with positive dv and negative separately to:
    arg_nearest = np.zeros(shape_collect, dtype=int)
    # collect its weights and mask to:
    z_weight = np.ma.zeros(shape_collect, dtype=float)

    for proc_dv_sign, sl_collect in [(-1, slice(0, interp_order)), (1, slice(interp_order, None))]:
        # mask outside current (positive/negative) side of dv
        dv_ma = np.ma.masked_outside(proc_dv_sign * dv_filled, 0, dv_max)
        # find nearest elements of current side
        arg = dv_ma.argsort(axis=-1, kind='mergesort')[..., :interp_order]   # use stable algorithm
        dv_sorted = np.take_along_axis(dv_ma, arg, axis=-1)
        # keep elements of current side only by mask with weights
        z_weight[..., sl_collect] = 1 / dv_sorted ** weight_power
        arg_nearest[..., sl_collect] = arg

    # check if closest points on each side is not adjacent points (also deletes where some side have no points?)
    b_ambiguous = abs(np.diff(arg_nearest[..., (interp_order - 1):(interp_order + 1)], axis=-1))[..., 0] != 1
    # # set output to NaN if from some side have no points:
    # b_outside = ~arg_nearest[..., (interp_order - 1):(interp_order + 1)].all(axis=-1)
    z_nearest = zi[arg_nearest]
    z_nearest[b_ambiguous, :] = np.nan
    # average
    return np.ma.average(z_nearest, weights=abs(z_weight), axis=-1)
    # np.ma.masked_array(z_nearest, z_weight.mask)


if b_2d_to_txt:
    def save2d(v2d, t, v_name):
        file_name = f'{t:%y%m%d_%H%M}{v_name}.csv'
        file_path = output_dir / file_name
        print(f'Writing to {file_name}')
        df = pd.DataFrame(v2d, index=latitudes, columns=longitudes)
        df.to_csv(file_path)

else:
    from gs_surfer import save_grd

    x_min, x_max = longitudes[[0, -1]]
    y_min, y_max = latitudes[[0, -1]]
    x_resolution = np.diff(longitudes[:2]).item()
    y_resolution = np.diff(latitudes[:2]).item()
    # check grid is ok
    np.testing.assert_almost_equal(x_resolution, (x_max - x_min) / (longitudes.size - 1), decimal=5)
    np.testing.assert_almost_equal(y_resolution, (y_max - y_min) / (latitudes.size - 1), decimal=5)  # 6 works too not 7

    def save2d(v2d, t, v_name):
        file_name = f'{t:%y%m%d_%H%M}{v_name}.grd'
        file_path = output_dir / file_name
        print(f'Writing to {file_name}')
        save_grd(np.flipud(v2d.filled(np.nan)), x_min, y_max, x_resolution, y_resolution, file_path)


if method == 'file_for_each_time':      # METHOD 1
    # Write 2D each time
    Path.mkdir(output_dir, parents=True, exist_ok=True)
    for v_name in variables:
        v = f.variables[v_name]                                                             # netcdf var
        v = variables_apply_coef[v_name](v[:, :, :, :] if v.ndim > 3 else v[:, :, :])    # numpy var
        if v.ndim > 3:
            # Extract data v_name as a 3D numpy array, calculate and write 2D isosurfaces
            z_values = variables_apply_coef['depth'](f.variables[expver_name[0]][:])  # depth values
            for i, t in enumerate(times):
                v3d = np.moveaxis(v[i, ...], 0, -1)
                for z_iso in z_isosurface[v_name]:
                    v2d = calc_iso_surface(v3d, v_isosurface=z_iso, zi=z_values, interp_order=iso_surface_window)
                    save2d(v2d, t, f'z({v_name}={z_iso})')
        else:
            # Extract data v_name as a 2D pandas DataFrame and write it to CSV
            for i, t in enumerate(times):
                v2d = v[i, ...]
                save2d(v2d, t, v_name)

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
        **{n: variables_apply_coef[v_name](f.variables[v_name].flatten()) for
           n, v_name in zip(var_short_names, variables)
           }
        })
    df.to_csv(filename, index=False)
    print('saved to', filename)
elif method == 'file_for_each_coord':
    # We will construct filename from date, source name excluding source area coordinates info, and point's coordinates
    filename_part_time = f'{times[0]:%y%m%d_%H%M}'.replace('_0000', '')
    filename_part_source= re.sub('_?area\([^)]*\)', '', file_path.stem)

    for ilat, lat in enumerate(latitudes):
        for ilon, lon in enumerate(longitudes):
            var_dict = {}
            for n, v_name in zip(var_short_names, variables):
                if expver_name:
                    if not (f.variables[v_name][:, :, ilat, ilon].mask.astype(np.int8).sum(axis=1) == 1).all():
                        raise NotImplementedError
                    v1d = f.variables[v_name][:, :, ilat, ilon].compressed()
                else:
                    v1d = f.variables[v_name][:, ilat, ilon]
                var_dict[n] = variables_apply_coef[v_name](v1d)

            df = pd.DataFrame({'Time': times, **var_dict})

            filename = output_dir / f'{filename_part_time}{filename_part_source}(N{lat:.5g},E{lon:.5g}).tsv'
            df.to_csv(filename, index=False, date_format='%Y-%m-%d %H:%M', float_format='%.5g',
                     sep='\t', encoding="ascii")

    print('saved to', filename)
print(f"{datetime.now():%Y-%m-%d %H:%M:%S} Ok>")