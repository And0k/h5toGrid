from pathlib import Path
from typing import Optional
import numpy as np
import dask.array as da  # import h5py
import pandas as pd
import vaex
import xarray as xr



def create_xrds(name, data, coords, interp_dt: Optional[np.timedelta64]=None):
    """ helper function to create dataset"""

    if data is not None:
        if coords is None:
            coords = {c: np.arange(0, n) for c, n in zip('xyz', data.shape)}
    else:
        data = da.zeros([len(v) for v in coords.values()])

    ds = xr.Dataset({
        name: xr.DataArray(
            data,
            dims=coords.keys(),
            coords=coords)
        })
    if interp_dt:
        return ds.resample(time=interp_dt.astype('m8[s]').item()).interpolate('linear')

    return ds


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
    out = {'/Time': df.Time.astype('f8').to_dask_array()  # convert because can not store datetime64 in hdf5 by h5py
           }                          # not assigning vaex astype('f8') result to df column as it had no good result
    cols_1d = [
         'Battery',
         'Heading',
         'Pitch',
         'Roll',
         'Pressure',
         'Temperature',
        ]
    out.update({f'/{p}': df[p].to_dask_array() for p in cols_1d})

    # convert to dask arrays with dimensions (z, time) for GS Surfer
    for p, p_col_start in [
        ('/u', 'East#'), ('/v', 'North#'), ('/up1', 'Up1#'), ('/up2', 'Up2#'),
        ]:  # 'Vu'
        cols = [k for k in df.columns.keys() if k.startswith(p_col_start)]
        if b_up:
            cols.reverse()
        out[p] = df[cols].to_dask_array().T

    for p, p_col_start in [('/intensity', 'Amp{}#'), ('/corr', 'Corr{}#')]:
        for i_beam in range(1, 5):
            cols = [k for k in df.columns.keys() if k.startswith(p_col_start.format(i_beam))]
            if b_up:
                cols.reverse()
            out[p] = df[cols].to_dask_array().T

    dt_all = df.diff(periods=1, column='Time').Time
    dt_value_counts = dt_all.value_counts()  # if 1st is always NaN, so if size > 2 then time is irregular (used below)
    dt = dt_value_counts.index[np.argmax(dt_value_counts)]  # ns

    # Save 1D and 2D source data
    # da.to_hdf5(path_base.with_suffix('.hdf5'), out)

    df.p_mean = df.Pressure.mean().item()
    df.z_from_device = df.p_mean + cfg['blank_dist'] + cfg['cell_cize'] * (
        np.arange(*((-np.shape(out['/u'])[0], 0) if b_up else (1, np.shape(out['/u'])[0] + 1)))  # reversed as cols if b_up
        )

    print('Export 2D datasets as NetCdf grids')
    out['/Vabs'] = da.sqrt(out['/u']**2 + out['/v']**2)
    out['/Vdir'] = da.degrees(da.arctan2(out['/u'], out['/v']))
    del out['/u']
    del out['/v']

    time_coord_converted = None  # will be calculated before saving of 1st 2d dataset and then assigned to all next datasets
    coords = {'z': -df.z_from_device, 'time': df.Time.to_numpy()}  #(out['/Time']/(24 * 3600E9)).compute()
    for name_p, data in out.items():
        if len(data.shape) != 2:
            continue
        name = name_p[1:]
        print(name, end=', ')
        ds = create_xrds(name, data, coords, interp_dt=np.timedelta64(dt, 'ns') if dt_value_counts.size > 2 else None)
        # .chunk({"x": 100, "y": 100, "time": -1})
        if time_coord_converted is None:
            time_coord_converted = (ds.coords['time'] - np.datetime64('1899-12-30T00:00:00')  # to Excel time
                                    ).values.astype('f8') / (24 * 3600E9)
        ds['time'] = time_coord_converted
        ds.to_netcdf(path_base.parent / f'{path_base.stem}_{name}.nc', mode='w', format='NETCDF4_CLASSIC')
        
        
def main():
    load_adp_adcp_nortek_signature(
        Path(r'd:\workData\BalticSea\ADCP_Nortek1MHz\S100452A014_TestA0_avgd.csv'
        )