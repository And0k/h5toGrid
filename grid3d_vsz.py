#!/usr/bin/env python
# coding:utf-8
from __future__ import print_function, division

"""
  Author:  Andrey Korzh <ao.korzh@gmail.com>
  Purpose: load CTD data from Veusz sections (there they are filtered),
           add nav. from hdf5 store (which is also source of data for Veusz sections),
           create 2D grids for depth layers (to make 3D plot)
  Created: 02.09.2016
"""
from sys import stdout as sys_stdout
from fnmatch import fnmatch
from datetime import timedelta
from pathlib import Path
from typing import Dict
from itertools import product
import numpy as np
#from numpy.core.records import fromarrays
import pandas as pd
from scipy import interpolate

if __debug__:
    from matplotlib import pyplot as plt
from other_filters import inearestsorted
from to_pandas_hdf5.h5_dask_pandas import filter_local, filter_local_arr
from veuszPropagate import load_vsz_closure
from grid2d_vsz import ge_sections, write_grd_fun
from gs_surfer import griddata_by_surfer  #sec_edges
# idata_from_tpoints, runs_ilast_good, track_b_invert
from utils_time import timzone_view
from utils2init import standard_error_info

# ##############################################################################
startSectionN = 1  # 1 Debug: skipped sections!!!!
db_path = r'd:\WorkData\BlackSea\200909_Ashamba\200909_Ashamba.h5'
params = [u'Temp', u'Sal', u'SigmaTh', u'O2', u'pH', u'Eh']  # u'ChlA', , u'Turb'
pst = 5
pen = 80
pstep = 5
pwidth = 5  # m

ranges = [  # of centers. First must be depth (to select it in interp in cycle)
    np.arange(pst, pen, pstep),
    pd.date_range(start='2020-09-10T12:00', periods=3)
    ]
margins = [pwidth, pd.Timedelta(0.5, 'D')]
patterns = [
    "Pres >= {} & Pres < {}",  # and Lon >= 16.15 and Lon < 17.00
    "index>=Timestamp('{}') & index<=Timestamp('{}')"
    ]

print_patern='{Pres:g}m. {index:%Y-%m-%d}'
file_patern='({index:%y%m%d},{Pres:}m)'
file_patern_no_time='{Pres:}m'  # file name pattern that will be prefixed with actual time range


def gen_queries(ranges, patterns, margins, param_names=None, print_patern='_'.join(['{}']*len(ranges))):
    """

    :param ranges:
    :param patterns:
    :param margins:
    :param print_patern:
    :return: Iterable[centers, query_str]
    """
    if param_names is None:
        param_names = [p.split(None,1)[0].split('>', 1)[0] for p in patterns]
    for centers in product(*(list(s) for s in ranges)):
        qstr_list = []
        for c, margin, qstr_pattern in zip(centers, margins, patterns):
            qstr_list.append(qstr_pattern.format(c - margin, c + margin))
        centers_dict = dict(zip(param_names, centers))
        print(f'\n{print_patern}'.format(**centers_dict), end='... ')
        yield (centers_dict, ' and '.join(qstr_list))


y_resolution = 0.5  # km


#r'd:\WorkData\BalticSea\160802_ANS32\160802_Strahov.h5'
tblN = 'navigation'  # '/navigation/table' after saving without no data_columns= True
tbl_sec_points = f'{tblN}/sectionsCTD_waypoints'  # '1608_CTDsections_waypoints'

cfg = {'gpx': {}, 'vsz_files': {}, 'db_path': Path(db_path),
       'filter': {'min': {'Lat': 10, 'Lon': 10}, 'max': {'Lat': 70, 'Lon': 70}}}
cfg['vsz_files'] = {'path': cfg['db_path'].parent / 'CTD-sections', 'filemask': '??????_????Z*.vsz'}
cfg['program'] = {'log': cfg['vsz_files']['path'] / 'CTDprofilesEnds.txt'}
# cfg['program']['logs']= 'CTDprofilesEnds.txt'
cfg['program']['veusz_path'] = u'C:\\Program Files (x86)\\Veusz'  # directory of Veusz
load_vsz = load_vsz_closure(Path(cfg['program']['veusz_path']))
search_nav_tolerance = timedelta(minutes=1)
t_our_zone = timedelta(hours=2)

cfg['gpx']['symbol_break_keeping_point'] = 'Circle, Red'  # will not keeping to if big dist:
cfg['gpx']['symbol_break_notkeeping_dist'] = 20  # km
cfg['gpx']['symbol_excude_point'] = 'Circle with X'
b_filter_time = False
# dt_point2run_max= timedelta(minutes=15)

path_dir_temp = cfg['db_path'].with_name('_subproduct')
# ----------------------------------------------------------------------
# dir_walker
vszFs = list(cfg['vsz_files']['path'].glob(cfg['vsz_files']['filemask']))  #cfg['vsz_files']['filemask'])]
print('Process {} sections'.format(len(vszFs)))
# Load data #################################################################
bFirst = True
timeEnd_Last = np.datetime64('0')
vsze = None
ctd_list = []
db_path_temp = path_dir_temp / 'CTD+Nav.h5'
try:
    if not db_path_temp.is_file():
        # __vsz data to hdf5__
        with pd.HDFStore(cfg['db_path'], mode='r') as storeIn:
            cols_nav = ['Lat', 'Lon']  #, 'Dist'
            # colCTD = [u'Pres', u'Temp', u'Sal', u'SigmaTh', u'O2', u'ChlA', u'Turb', u'pH', u'Eh']  # u'ChlA', , u'Turb'
            for isec, vszF in enumerate(vszFs, start=1):
                # Load filtered down runs data
                if isec < startSectionN:
                    continue
                print('\n{}. {}'.format(isec, vszF))
                vsze, ctd = load_vsz(vszF, veusze=vsze, prefix='CTD')
                if 'ends' not in ctd:
                    print('data error!')

                # Add full resolution nav to section from navigation['Lat', 'Lon']
                qstr_trange_pattern = "index>=Timestamp('{}') & index<=Timestamp('{}')"
                qstr = qstr_trange_pattern.format(*ctd['time'][[0, -1]])

                df_nav = storeIn.select(tblN, qstr, columns=cols_nav)  # DepEcho
                # qstr = qstr_trange_pattern.format(ctd['time'][0], df_nav.index[0])
                # Nind_st = storeIn.select_as_coordinates(tblN, "index=={}".format(df_nav.index[0]))[0]
                df_nav = filter_local(df_nav, cfg['filter'])
                ctd = filter_local_arr(ctd, {k: v for k, v in cfg['filter'].items() if k in ctd})
                for col in df_nav.columns:
                    b_ok = df_nav[col].notna()
                    n_ind = inearestsorted(
                        df_nav.index[b_ok],
                        ctd['time'][np.isnan(ctd[col])] if col in ctd else ctd['time']
                        )
                    # Check time difference between nav found and time of requested points
                    dt = df_nav.index[n_ind] - ctd['time']
                    if np.any(abs(dt) > search_nav_tolerance * 5):
                        print('bad nav data coverage')
                    ctd[col] = np.interp(ctd['time'].view(np.int64),
                                         df_nav.index[b_ok].values.view(np.int64),
                                         df_nav.loc[b_ok, col])
                cols = list(ctd.keys()).copy()
                cols.remove('time')
                cols = [col for col in cols if ctd[col].size == ctd['time'].size]

                # with pd.HDFStore(cfg['db_path'], mode='w') as storeIn:
                ctd_list.append(np.core.records.fromarrays(
                    [ctd['time']] + [v for k, v in ctd.items() if k in cols],
                    names=['time'] + cols, formats=['M8[ns]'] + ['f8'] * len(cols)
                    ))
                # this creates strange ncols x len(array) instead of size (len(array),)
                # np.array([ctd['time']] + [v.ravel() for k, v in ctd.items() if k != 'time'],
                #          dtype={'names': ['time'] + cols,
                #                 'formats': ['M8[ns]'] + ['f8'] * len(cols)})
                # with pd.HDFStore(cfg['db_path'], mode='w') as storeIn:
        df_ctd = pd.DataFrame.from_records(np.hstack(ctd_list), index='time')
        with pd.HDFStore(db_path_temp, mode='w') as storeOut:
            storeOut.put('CTD', df_ctd, format='table', data_columns=['Pres', 'Lat', 'Lon'])
        print('source data, combined with nav. saved to storage ', db_path_temp)
    else:
        # Gridding


        maxDiff = {'Lat': 0.1, 'Lon': 0.1}

        # 1.standard 2D

        # with pd.HDFStore(cfg['db_path'], mode='r') as storeIn:
        #     navs = [(navp, navp_d) for navp, navp_d in ge_sections(storeIn[tbl_sec_points], cfg)]

        b1st_hor = True
        print('Gridding on horisonts:', end=' ')
        y_resolution /= (1.852 * 60)  # %km2deg(1)
        for centers, qstr in gen_queries(ranges, patterns, margins, print_patern=print_patern):
            file_stem = file_patern.format(**centers)
            file_stem_no_time = file_patern_no_time.format(**centers)
            # p, p_st, p_en in np.arange(pst, pen, pstep)[:, np.newaxis] + np.array(
            # [[0, -pwidth, pwidth]]) / 2:  # [:, np.]
            # print('\n{:g}m.'.format(p), end=' ')
            # qstr = qstr_pattern.format(p_st, p_en)
            FCTD = pd.read_hdf(db_path_temp, 'CTD', where=qstr)
            if FCTD.empty:
                print('- empty', end='')
                continue
            time_st_local, time_en_local = [timzone_view(x, t_our_zone) for x in FCTD.index[[0, -1]]]
            fileN_time =\
                f'{time_st_local:%y%m%d_%H%M}-'\
                f'{{:{"%d_" if time_st_local.day!=time_en_local.day else ""}%H%M}}'.format(time_en_local)

            # Get data for each run
            # It is possible to get it by aggeregation (df_points = FCTD.groupby(['Lat', 'Lon']))
            # but here we use runs info which is icapsulated in _shift_. Runs were found in Veusz
            iruns = np.flatnonzero(np.diff(FCTD['shift']) != 0)
            ctd = np.empty((iruns.size + 1,), {'names': params + ['Lat', 'Lon'],
                                               'formats': ['f8'] * (len(params) + 2)})
            ctd.fill(np.NaN)
            for param in ctd.dtype.names:
                if param not in maxDiff: maxDiff[param] = np.NaN
                for irun, isten in enumerate(np.column_stack((np.append(1, iruns),
                                                              np.append(iruns, len(FCTD))))):
                    data_check = FCTD.iloc[slice(*isten), FCTD.columns.get_indexer(('Pres', param))]
                    data_check.dropna(axis=0, inplace=True)

                    # Outlines:
                    bBad = np.abs(np.diff(data_check[param])) > maxDiff[param]
                    if np.any(bBad):
                        print(f'\nrun {irun}: {param} have {np.sum(bBad)} bad diff between vals', end=', ')
                        plt.scatter(data_check.index, data_check[param])
                        plt.scatter(data_check.index[:-1][bBad], data_check[:-1].loc[bBad, param], marker='+', c='r', s=15,
                                    zorder=10)
                        # delete outlines:
                        bBad = abs(data_check[param] - np.nanmean(data_check[param])) > maxDiff[param]
                        print('delete {}/{} outlines'.format(bBad.sum(), bBad.size()))
                        data_check.loc[bBad, param] = np.NaN
                    # interpolate to center depth:  # need smooth?
                    if data_check.empty:
                        print(f'\nrun{irun}, {param} - nans only, ', end='')
                        continue
                    ctd[param][irun] = np.interp(centers['Pres'], data_check['Pres'], data_check[param])                   # in previous code the mean was used:
                    # ctd[param][irun] = np.nanmean(data_check[param])

            # Save run track path
            # print('Saving CTD run track path')
            np.savetxt(path_dir_temp / f'{file_stem}points.csv',
                       np.vstack((ctd['Lat'], ctd['Lon'])).T,
                       fmt='%g,%g', header='Lon,Lat', delimiter=',', comments='')

            x_min = np.nanmin(ctd['Lon'])
            x_max = np.nanmax(ctd['Lon'])
            y_min = np.nanmin(ctd['Lat'])
            y_max = np.nanmax(ctd['Lat'])
            Lat_mean = (y_min + y_max) / 2
            x_resolution = y_resolution * np.cos(np.deg2rad(Lat_mean))
            gdal_geotransform = (x_min, x_resolution, 0, y_max, 0, -y_resolution)
            write_grd_this_geotransform = write_grd_fun(gdal_geotransform)
            x = np.arange(x_min, x_max, x_resolution)
            y = np.arange(y_min, y_max, y_resolution)
            # todo: Interpolate path in grid coordinates for blanking: use filter of border to not delete too much
            y_edge_path = []
            xm, ym = np.meshgrid(x, y)
            # bot_edge_add = y_resolution  # m
            # ym_blank = ym > bot_edge_add - y_edge_path
            print('{}x{} - '.format(*xm.shape), end='')

            griddata_by_surfer(
                ctd, path_stem_pattern=(path_dir_temp / 'surfer' / f'{file_stem}{{}}'), margins=True,
                xCol='Lon', yCol='Lat', xMin=x_min, xMax=x_max, yMin=y_min, yMax=y_max,
                SearchAngle=45, AnisotropyAngle = 45, AnisotropyRatio=0.8, SearchNumSectors=8
                )


            bFirst = True
            for iparam, param in enumerate(params):  # param= u'Temp'
                label_param = param  # .lstrip('_').split("_")[0] #remove techcnical comments in names
                if bFirst:
                    print(label_param, end='')
                else:
                    print(', ', label_param, end='')
                if __debug__:
                    sys_stdout.flush()
                    i = 0
                bGood = np.isfinite(ctd[param])  # np.logical_and(, bGoodP)

                interp_method_dir = []  # [['linear', ''], ['cubic', 'cubic\\']] - not use because worse
                for interp_method, interp_method_subdir in interp_method_dir:
                    z = interpolate.griddata((np.append(ctd['Lon'], [])[bGood],
                                              np.append(ctd['Lat'], [])[bGood]),
                                             np.append(ctd[param], [])[bGood], (xm, ym),
                                             method=interp_method)  # , rescale= True'cubic','linear','nearest'
                    # Blank z below bottom (for compressibility)
                    # z[ym_blank] = np.NaN

                    path_stem = path_dir_temp / interp_method_subdir / f'{file_stem}{label_param}' #f'{fileN_time}{label_param},{file_stem_no_time}'
                    if False:  #__debug__:
                        try:
                            plt.figure((iparam + 1) * 10 + i);
                            i += 1
                            plt.title(label_param)
                            im = plt.imshow(z, extent=[x[0], x[-1], y_max, y_min])  # , extent=(0,1,0,1), origin=’lower’
                            CS = plt.contour(xm, ym, z, 6, colors='k')
                            plt.clabel(CS, fmt='%g', fontsize=9, inline=1)
                            # if Dep_filt is not None:
                            #     plt.plot(nav_dist, -nav_dep, color='m', alpha=0.5, label='Depth')  # bot
                            if bFirst:
                                plt.plot(ctd['Lon'], ctd['Lat'], color='m', alpha=0.5, label='run path')
                            CBI = plt.colorbar(im, shrink=0.8)
                            # CB = plt.colorbar(CS, shrink=0.8, extend='both')
                            # plt.gcf().set_size_inches(9, 3)
                            plt.savefig(path_stem.with_suffix('.png'))  # , dpi= 200
                            # plt.show()
                            plt.close()
                            pass
                        except Exception as e:
                            print('\nCan no draw contour! ', standard_error_info(e))
                    # gdal_drv_grid.Register()

                    write_grd_this_geotransform(path_stem.with_suffix('.grd'), z)
                if bFirst:
                    bFirst = False

except Exception as e:
    print('\nError! ', standard_error_info(e))
    raise e
