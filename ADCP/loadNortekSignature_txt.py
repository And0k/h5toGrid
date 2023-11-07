from pathlib import Path
from typing import Tuple, Optional, Union
from datetime import timedelta
import numpy as np
import dask.array as da  # import h5py
# import pandas as pd
import vaex
import xarray as xr

import sys

drive_d = 'D:' if sys.platform == 'win32' else '/mnt/D'  # to run on my Linux/Windows systems both
for path in ['/Work/_Python3/And0K/Veusz_plugins', '/Work/_Python3/And0K/tcm']:
    scripts_path = Path(drive_d + path)
    sys.path.append(str(Path(scripts_path).parent.resolve()))
from Veusz_plugins import func_vsz as v
from tcm.incl_h5clc_hy import polar2dekart


def create_dar(data=None, coords=None,
        interp_dt: Tuple[None, np.timedelta64, timedelta]=None,
        bin_dt=None, bin_dz=None
):
    """
    Helper function to create NetCDF dataset
    :param data: if None then will be zeros for coords, which must be defined
    :param coords: if None then coordinates will be data integer indexes of each dimension
    :param interp_dt:
    :return:
    """

    if data is None:
        data = np.zeros([len(v) for v in coords.values()])
    else:
        if coords is None:
            coords = {c: np.arange(0, n) for c, n in zip('xyz', data.shape)}

    dar = xr.DataArray(
        data,
        dims=coords.keys(),
        coords=coords
    )
    
    if interp_dt:
        if isinstance(interp_dt, np.timedelta64):  # to timedelta:
            interp_dt = interp_dt.astype('m8[s]').item()
        dar = dar.resample(time=interp_dt).interpolate('linear')
    if bin_dt or bin_dz:
        if bin_dt:
            if isinstance(bin_dt, np.timedelta64):  # to timedelta:
                bin_dt = bin_dt.astype('m8[s]').item()
            dar_out = dar.resample(time=bin_dt).mean()  # dim=['z', 'time'] .groupby_bins(bins, group='time', precision=15)
        else:
            dar_out = dar
        if bin_dz:
            dz = np.diff(coords['z'][:2]).item()
            dz_bin = dz * (bin_dz + 1)
            n_bins = len(coords['z']) // bin_dz
            bins = coords['z'][0] + np.cumsum([dz_bin] * n_bins) - dz_bin - dz / 2
            # bins = coords['z'][::n_bins]
            # bins = len(coords['z'])//bin_dz
            # np.arange(coords['z'][0], coords['z'][-1] + bin_dz/2, bin_dz)
            dar_out = dar_out.groupby_bins(bins=bins, group='z', restore_coord_dims=True).mean()
            coord_z_new = np.float64([bin.mid for bin in dar_out['z_bins'].values])
            # transpose back (why above have transposed data?)  # restore_coord_dims not works?
            if dar_out.coords.keys() != dar.coords.keys():
                # dar_out = dar_out.transpose()
                dar_out = xr.DataArray(
                    dar_out.data,
                    dims=coords.keys(),
                    coords={key: coord_z_new if key == 'z' else dar_out.coords[key] for key in coords.keys()}
                )
    else:
        dar_out = dar
    return dar_out


def load_adp_adcp_nortek_signature(path_base, cfg=None):
    if cfg is None:
        cfg = {}
    if not cfg.get('cell_cize'):
        cfg['cell_cize'] = 1       # m
    if not cfg.get('blank_dist'):
        cfg['blank_dist'] = -0.5   # >0=up, <0=down - reflected in b_up
    if not cfg.get('min_p'):
        cfg['min_p'] = 75          # dBar, filter above

    b_up = cfg['blank_dist'] < 0

    df = vaex.read_csv(path_base, **{'delimiter': ';', 'parse_dates': [0]}, copy_index=False)
    df.rename('DateTime', 'Time')   # df.drop(['DateTime'], inplace=True, check=True) - hide check not works: Time not accessible!

    print('all data: {} - {}'.format(
        *[df.Time[slice(st, en)].to_numpy().astype('M8[s]').item() for st, en in [(None, 1), (-1, None)]]))

    df = df[df.Pressure > cfg['min_p']]
    print('where P > {}dBar: {} - {}'.format(
        cfg['min_p'],
        *[df.Time[slice(st, en)].to_numpy().astype('M8[s]').item() for st, en in [(None, 1), (-1, None)]])
        )

    # Dict of {hdf5 key: dask data array}
    cols_1d = [
         'Battery',
         'Heading',
         'Pitch',
         'Roll',
         'Pressure',
         'Temperature',
        ]
    
    # Convert Time because can not store datetime64 in hdf5 by h5py
    out = {'Time': df.Time.astype('f8').to_dask_array()}
    # Not assigning vaex astype('f8') result to df column as it had no good result
    out.update({f'{p}': df[p].to_dask_array() for p in cols_1d})
    # Save only 1D source data
    da.to_hdf5(path_base.with_suffix('.1D.hdf5'), out)
    
    # convert to dask arrays with dimensions (z, time) for GS Surfer
    for p, p_col_start in [
        ('u', 'East#'), ('v', 'North#'), ('up1', 'Up1#'), ('up2', 'Up2#'),
        ]:  # 'Vu'
        cols = [k for k in df.columns.keys() if k.startswith(p_col_start)]
        if b_up:
            cols.reverse()
        out[p] = df[cols].to_dask_array().T

    for p, p_col_start in [('intensity', 'Amp{}#'), ('corr', 'Corr{}#')]:
        for i_beam in range(1, 5):
            cols = [k for k in df.columns.keys() if k.startswith(p_col_start.format(i_beam))]
            if b_up:
                cols.reverse()
            out[p] = df[cols].to_dask_array().T
    
    # find interval for most of the data
    dt_all = df.diff(periods=1, column='Time').Time
    dt_value_counts = dt_all.value_counts()  # if 1st is always NaN, so if size > 2 then time is irregular (used below)
    dt = dt_value_counts.index[np.argmax(dt_value_counts)]  # ns

    # Save 1D and 2D source data
    # da.to_hdf5(path_base.with_suffix('.hdf5'), out)

    df.p_mean = df.Pressure.mean().item()
    df.z_from_device = df.p_mean + cfg['blank_dist'] + cfg['cell_cize'] * (
        np.arange(*((-np.shape(out['u'])[0], 0) if b_up else (1, np.shape(out['u'])[0] + 1)))  # reversed as cols if b_up
        )
    
    save_2d_for_surfer(
        time=df.Time.to_numpy(), z=-df.z_from_device, out=out, path_base=path_base,
        dt=(dt if dt_value_counts.size > 2 else None)
    )


def save_2d_for_surfer(time, z, out, path_base, dt: Union[None, list, np.timedelta64], dz=None):
    print(
        f'Export 2D datasets to {path_base.parent.name}/{path_base.stem}_{{param}}.nc '
        'files as NetCdf grids for param = ', end=''
    )
    b_have_u = 'u' in out
    da_np = da if isinstance(out.get('u' if b_have_u else 'Vabs'), da.Array) else np
    if not b_have_u:
        out['v'], out['u'] = polar2dekart(out['Vabs'], out['Vdir'])
        del out['Vabs'], out['Vdir']
        out['Vabs'] = out['Vdir'] = None  # to the end of ``out``

    # Time for Surfer
    # (will be calculated before saving of 1st 2d dataset and then assigned to all next datasets)
    dt = dt if isinstance(dt, list) else [dt]
    dz = dz if isinstance(dz, list) else [dz]*len(dt)
    for i_used_dt, (dt, dz) in enumerate(zip(dt, dz)):
        ds_saved = {}
        time_coord_converted = None  # we set value after creating of ds because dt and time should have same units
        str_dt = None
        coords = {'z': z, 'time': time}
        for name, data in out.items():
            if name == 'Vabs':
                data = da_np.sqrt(ds_saved['u'] ** 2 + ds_saved['v'] ** 2)
                dt_cur = dz_cur = None                # do not interp/bin 2nd time
                coords = ds_saved['u'].coords   # use interp/binned coord
            elif name == 'Vdir':  # must run after 'Vabs' in this cycle
                data = da_np.degrees(da_np.arctan2(ds_saved['u'], ds_saved['v'])) % 360
                # leave dt_cur and coords same as for Vabs
            else:
                dt_cur = dt
                dz_cur = dz
            if len(data.shape) != 2:
                continue
            print(name, end=', ')
            dar = create_dar(
                data, coords=coords,
                **{('bin_dt' if i_used_dt > 0 else 'interp_dt'): dt_cur},
                bin_dz=dz_cur
            )
            # .chunk({"x": 100, "y": 100, "time": -1})
            
            # Save Vabs and Vdir only after interp/bining of u and v in create_xrds(). Skip saving u and v
            if name in ('u', 'v'):
                ds_saved[name] = dar
                continue
                
            if time_coord_converted is None:
                # to Excel time
                time_coord_converted = (dar.coords['time'] - np.datetime64('1899-12-30T00:00:00')
                                        ).astype('M8[ns]').values.astype('f8') / (24 * 3600E9)
                str_dt = v.str_dt(dt.astype('m8[s]') if isinstance(dt, np.timedelta64) else np.timedelta64(dt, 's'))
            dar['time'] = time_coord_converted  # changes dar.coords['time']
            xr.Dataset({name: dar}).to_netcdf(
                path_base.parent / f'{path_base.stem}_{name}_dt={str_dt}{f",dz={dz}" if dz else ""}.nc',
                mode='w', format='NETCDF4_CLASSIC'
            )


def main():
    load_adp_adcp_nortek_signature(Path(
        r'd:\WorkData\BalticSea\181005_ABP44\ADCP_Nortek1MHz\_raw\S100452A014_TestA0_avgd.csv'
    ))


if __name__ == '__main__':
    main()
