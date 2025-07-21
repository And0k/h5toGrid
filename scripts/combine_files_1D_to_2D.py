#!/usr/bin/env python
"""
Save NetCDF (*.nc) and Surfer binary (*.grd) 2D grids from many files of same text column format
"""
from pathlib import Path
import numpy as np
import netCDF4
# My functions:
from gs_surfer import save_2d_to_grd


def save2d2netcdf(path, xyz, table=None, format='NETCDF4'):
    """
    Saves 2D variable `z` with its coordinates `x` and `y` defined in xyz to NetCDF file
    :param path:
    :param xyz: items with
     - keys are x,y,z NetCDF variables names. `x` and `y` dimension names will be the same.
     - values are corresponded  NetCDF variables values: `x`, `y`, `z` of size `n`, `m` and `n` x `m` correspondingly
    :param table: if not None put grid in this NetCDF group (not compatible to Surfer if not None)
    :param format: for some types may need 'NETCDF4_CLASSIC' to use CLASSIC format for Views compatibility
    :return:
    """
    nv = {}     # variables to be used as NetCDF variables
    nc = None   # group where to save NetCDF variables

    def cr_create_dim(name, val):
        nonlocal nc, nv  # to not take them from global scope if same vars would exist there

        def create_dim(n):
            """
            Create NetCDF dimension of name `name` and assign values `val`. These variables are closures.
            :param n: dimension size
            :return: None
            """
            nc.createDimension(name, n)
            nv[name] = nc.createVariable(name, 'f4', (name,), zlib=False)
            nv[name][:] = val

        return create_dim

    do = []  # saves pended operations
    with netCDF4.Dataset(path.with_suffix('.nc'), 'w', format=format) as nc_root:
        nc = nc_root.createGroup(table) if table else nc_root
        for i, (name, val) in enumerate(xyz.items()):
            if i < 2:
                # remember operations to wait when we reach dimension size
                do.append(cr_create_dim(name, val))
            else:
                for k in [0, 1]:
                    do[k](val.shape[k])
                nc.createVariable(name, 'f4', tuple(nv.keys()), zlib=True)
                nc[name][:] = val


def combine_1d_files_to_grd(path, x_min, x_resolution, cols, delimiter, file_dim_name='time', transpose=False):
    """
    Saves NetCDF (*.nc) and Surfer binary (*.grd) grid for data of each `z` cols column
    with `x` is changed along with order of found files in `path`.
    :param path:
    :param x_min: `x[0]` that is value corresponded to 1st file.
    :param x_resolution: increment value of `x` with next file.
    :param cols: dict. 1st item has name of `y`-column and value is column number, this col will be loaded only from 1st
     file (should be the same for all files).
     - keys are param names, keys (except 1st) will be part of output file_grd name.
     - values are numpy.loadtxt() parameter.
    :param delimiter: numpy.loadtxt() parameter
    :param file_dim_name: files dimension name
    :param transpose: bool, transpose output

    Issues:
    1. `transpose` no effect on NetCDF grids for use in Surfer
    2. *.grd grids data is shifted on 1/2 cell
    """
    path = Path(path)
    files = list(path.parent.glob(path.name))
    n_files = len(files)
    print(f'Loading {n_files} {path} files...')
    if n_files <= 1:
        raise NotImplementedError(f'Too little files: can not convert {n_files} to 2D grid')

    usecols = list(cols.values())
    z2d = []
    for i, file in enumerate(files):
        m = np.loadtxt(
            file,
            usecols=usecols,
            skiprows=0,
            delimiter=delimiter)
        if not z2d:
            # save and exclude y
            usecols = usecols[1:]
            y = m[:, 0]
            m = m[:, 1] if len(usecols) == 1 else m[:, 1:]
        z2d.append(m)

    xy_params = {
        'x_min': (x_min.astype('M8[s]')).astype(float) / (24 * 3600) + 1 - np.datetime64('1900-01-01').astype(float),
        'y_max': y.max().item(),
        'x_resolution': x_resolution,
        'y_resolution': np.ediff1d(y).mean()  # np.ediff1d(y[:2]).item()
        }
    if transpose:
        xy_params = {
            'x_min': xy_params['y_max'] - xy_params['y_resolution'] * (len(y) - 1),
            'y_max': xy_params['x_min'] + xy_params['x_resolution'] * (n_files - 1),
            'x_resolution': xy_params['y_resolution'],
            'y_resolution': xy_params['x_resolution']
            }
    y_name, *z_names = cols.keys()
    for icol, out_name in enumerate(z_names):
        file_grd = path.with_name(f'{x_min.item():%y%m%d}{out_name}')
        z = np.vstack(z2d if len(usecols) == 1 else [z2[:, icol] for z2 in z2d])

        # Saving NetCDF
        xyz = {
            file_dim_name: np.arange(xy_params['y_max'] - xy_params['y_resolution'] * z.shape[0],
                                     xy_params['y_max'] - xy_params['y_resolution'] * 0.5, xy_params['y_resolution']),
            # "Reverse axis" in GUI (Surfer) is only option to reverse `y`
            y_name: y,  # exact. If need strictly equal spaced, use:
                # np.arange(xy_params['x_min'], xy_params['x_min'] + xy_params['x_resolution'] * (z.shape[1] - 0.5),
                #           xy_params['x_resolution']),
            out_name: z
            }
        if transpose:
            xyz = list(xyz.items())
            xyz[1], xyz[0] = xyz[0], xyz[1]
            xyz[2] = (out_name, z.T)
            xyz = {k: v for k, v in xyz}
        print(f'Saving {z.shape} with {xy_params} grid to {file_grd}...')
        save2d2netcdf(
            path.with_name(f'{x_min.item():%y%m%d}{out_name}.nc'),
            xyz
            )

        # Saving Surfer binary grid
        if not transpose:
            z = z.T
        save_2d_to_grd(np.flipud(z), **xy_params, file_grd=file_grd)


if __name__ == '__main__':
    # Using example

    x_min = np.datetime64('2020-10-03')
    x_resolution = 1  # days
    delimiter = None  # or '\t'
    transpose = True  # or False

    params = [
        {'path': r'd:\WorkData\BalticSea\_other_data\_model\Copernicus\GoF\Aver&fluct\*.txt',
            'cols': {'lon': 1, 's_diff': 4}
         },
        {
            'path': r'd:\WorkData\BalticSea\_other_data\_model\Copernicus\GoF\GoF_sm+Wind\*_wn.txt',
            'cols': {'lon': 0, 'wind_abs': 4, 'wind_dir': 5}
         }
        ]

    for param in params:
        combine_1d_files_to_grd(
            x_min=x_min, x_resolution=x_resolution, delimiter=delimiter, transpose=transpose, **param
            )

    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} Ok>")
