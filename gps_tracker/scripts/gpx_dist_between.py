import logging
from codecs import open
from pathlib import Path, PurePath
from sys import stdout as sys_stdout
from typing import Any, Dict, Mapping, Union
from itertools import combinations

import gpxpy
import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from gpxpy.gpx import GPX
from gps_tracker.autofon_coord import dx_dy_dist_bearing, resample_df

from to_pandas_hdf5.gpx2h5 import df_rename_cols  # gpxConvert
from to_pandas_hdf5.h5_dask_pandas import h5_append, df_to_csv

path_gpx = PurePath(
    r'd:\WorkData\_experiment\tracker\240315_Devau\240315_1100sp0,sp1,sp2,tr1,tr2.raw.gpx'
    # r'd:\WorkData\_experiment\tracker\240306_Devau\240306@Topcon_GR-5.gpx'
    # r'd:\WorkData\_experiment\tracker\240306\all_filtered.gpx'
    # r'd:\WorkData\_experiment\tracker\220318_1633@sp2&3.gpx'
    # r'd:\WorkData\_experiment\tracker\240229\240229_1150@v1-3.gpx'
)
add_sfx = '_abs'    # add suffix to autput files names
tbl_prefix = ''    # 'sp'
cfg_out = {
    'db_path': path_gpx.with_name(f'{path_gpx.stem}{add_sfx}').with_suffix('.h5'),
    }


# Default deletion logic
## Delete previous data if out file exists
b_del_prev = Path(cfg_out['db_path']).is_file()
## Save table with raw data to hdf5 file if it not exists (i.e. delete file to save raw)
b_save_raw = not b_del_prev

# Overwrite default deletion logic
b_del_prev = True  # False
b_save_raw = True

# dfs_raw = gpxConvert(
#     {'in': {'tracks_cols': },
#      'out': {'tracks_cols': ['time', 'Lat', 'Lon']}
#      },
#     path_gpx)  # concatenates all tracks but this is not that we want

with open(path_gpx, 'r', encoding='utf-8') as gpx_file:
    gpx = gpxpy.parse(gpx_file)

tr_cols_in = ['time', 'latitude', 'longitude']
tr_cols_out = ['time', 'Lat', 'Lon']
dfs_raw = {}
for i, track in enumerate(gpx.tracks):
    df_segs = pd.concat([
        pd.DataFrame.from_records(
            [[getattr(point, c) for c in tr_cols_in] for point in segment.points],
            columns=tr_cols_in,
            index=tr_cols_in[0]) for segment in track.segments])

    n_points = len(df_segs)

    track.name = track.name.replace(' ', '_')
    print(i + 1, f'{track.name}: {n_points}', end='')
    if len(df_segs) <= 2:
        print(' - skipping')
        continue
    print()
    df_rename_cols(df_segs, tr_cols_in[1:], *tr_cols_out)
    dfs_raw[track.name] = df_segs

if not dfs_raw:
    print(f'No tracks in {path_gpx}')
# d = dx_dy_dist_bearing(*dfs[0].loc[:, ['Lon', 'Lat']].to_numpy().T, *dfs[1].loc[:, ['Lon', 'Lat']].to_numpy().T)


def map_to_suffixed(names, tbl_prefix, probe_number_str):
    """ Adds tbl suffix to output columns before accumulate in cycle for different tables
    mod of map_to_suffixed used in main() of inclinometer.incl_h5clc_hy
    """
    suffix = f'{tbl_prefix}{probe_number_str}'  #:02
    return {col: f'{col}_{suffix}' for col in names}


def combine_words(word1, word2):
    if word1 == word2:
        return word1
    i = 0
    for i, (l1, l2) in enumerate(zip(word1, word2)):

        if l1.isalpha():
            if l1 == l2:
                # skip common alpha
                continue
            # no common alpha-only prefix
            sep = '_to_'
            i = 0
            break
        if l2.isalpha():
            # l1 is digit => no common alpha-only prefix
            sep = '_to_'
            i = 0
            break

        # common alpha prefix
        sep = 'to'
        break

    # while i < len(word1) and i < len(word2) and word1[i] == word2[i] and :
    #     i += 1
    i_common_prefix_end = i
    return sep.join((word1, word2[i_common_prefix_end:]))


# Raw data to HDF5
if b_save_raw:
    print('Saving raw data to', cfg_out['db_path'])
    with pd.HDFStore(cfg_out['db_path'], mode='w') as store:
        for name, df_raw in dfs_raw.items():  # track names as table name for row data
            h5_append({
                **cfg_out,
                'table': name,
                'db': store,
                },
                df_raw,
                log={}
            )
    b_del_prev = False

# Resample
index_st = np.min([df.index[0] for df in dfs_raw.values()])
index_en = np.max([df.index[-1] for df in dfs_raw.values()])
# 1st interval selected to get good 1st point for b_relative_to_first_points mode
# 1st point for each tracker:
first_points = {}
mean_period = {}
for ibin, bin in enumerate(['5min', '20min', '1h']):  # , '30s'
    dfs = {}

    shift_to_mid = to_offset(bin) / 2
    for name, df_raw in dfs_raw.items():
        if ibin == 0:
            # Collect mean period
            mean_period[name] = np.mean(np.diff(df_raw.index))  # np.diff(df_raw.index[[0,-1]]) / len(df_raw)
            print(name, f'mean period: {mean_period[name]}')
        # Interpolate if mean period less than bin else bin average
        if shift_to_mid < mean_period[name]:  # interpolate
            df, _ = resample_df(
                df_raw, bin, index_st=index_st, index_en=index_en,
                # limit=process['interp_limit']
            )
        else:                                 # bin average
            df = df_raw.resample(bin, offset=-shift_to_mid).mean()
            df.index += shift_to_mid
        dfs[name] = df.rename(columns=map_to_suffixed(df.columns, tbl_prefix, str(name)))

    dfs_all = pd.concat(dfs.values(), sort=False, axis=1)

    b_relative_to_first_points = True   # set False for relative positions calculation
    if b_relative_to_first_points:
        if ibin == 0:
            first_points = {key_a: dfs_all.loc[dfs_all[f'Lon_{tbl_prefix}{key_a}'].first_valid_index(), [
                    f'Lon_{tbl_prefix}{key_a}',
                    f'Lat_{tbl_prefix}{key_a}'
                ]].values for key_a in dfs}
            print('Displacements will be found relative to 1st points of each tracker:', first_points)
        pairs = zip(dfs.keys(), dfs.keys())
    else:
        pairs = list(combinations(dfs.keys(), 2))
    for key_a, key_b in pairs:
        pair_sfx = f'_{combine_words(key_a, key_b)}' if len(dfs) > 1 else ''  #   f'{key_a}{key_b}'
        dfs_all.loc[:, [f'dx{pair_sfx}', f'dy{pair_sfx}', f'dr{pair_sfx}', f'Vdir{pair_sfx}']] = (
            dx_dy_dist_bearing(
            *first_points[key_a],
            *dfs_all.loc[:, [
                f'Lon_{tbl_prefix}{key_a}',
                f'Lat_{tbl_prefix}{key_a}'
            ]].to_numpy().T) if b_relative_to_first_points else
            dx_dy_dist_bearing(*dfs_all.loc[:, [
                f'Lon_{tbl_prefix}{key_a}',
                f'Lat_{tbl_prefix}{key_a}',
                f'Lon_{tbl_prefix}{key_b}',
                f'Lat_{tbl_prefix}{key_b}'
            ]].to_numpy().T)
        )
    # df = pd.DataFrame.from_records(d, columns=['dx', 'dy', 'dr', 'Vdir'], index=dfs[0].index)

    # Resampled data to HDF5
    cfg_out['table'] = f'{",".join(dfs.keys())}avg{bin}'
    with pd.HDFStore(cfg_out['db_path'] , mode='w' if ibin == 0 and b_del_prev else 'a') as store:  #
        h5_append({
            **cfg_out,
            'db': store
            },
            dfs_all,
            log={}
        )

    df_to_csv(dfs_all, cfg_out, add_suffix=add_sfx)  #, 'dir_export':

    print(f'saved {len(df)} rows. ok>')


if __name__ == '__main__':
    print(pd.Timestamp.now(), '> Starting', Path(__file__).stem)