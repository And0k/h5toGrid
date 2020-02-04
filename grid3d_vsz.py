#!/usr/bin/env python
# coding:utf-8
from __future__ import print_function, division

"""
  Author:  Andrey Korzh <ao.korzh@gmail.com>
  Purpose: load filtered CTD data from Veusz sections,
            add nav. from hdf5 store (which is also source of data for Veusz sections),
            create 2D grids for depth layers (to make 3D plot)
  Created: 02.09.2016
"""
from os import listdir as os_listdir, path as os_path
from fnmatch import fnmatch
from datetime import timedelta
from pathlib import Path
from typing import Dict

import numpy as np
from numpy.core.records import fromarrays
import pandas as pd
from scipy import interpolate

if __debug__:
    from matplotlib import pyplot as plt
from other_filters import inearestsorted
from veuszPropagate import load_vsz_closure
from grid2d_vsz import sec_edges, write_grd_fun, griddata_by_surfer
# idata_from_tpoints, runs_ilast_good, track_b_invert
from utils_time import timzone_view

# ##############################################################################
startSectionN = 1  # 1 Debug: skipped sections!!!!
fileInFH5 = r'd:\WorkData\BalticSea\160802_ANS32\160802_Strahov.h5'
cfg = {'gpx': {}, 'vsz_files': {}}
cfg['vsz_files'] = {'path': Path(fileInFH5).parent()}
cfg['program'] = {'log': os_path.join(cfg['vsz_files']['path'], 'CTDprofilesEnds.txt')}
# cfg['program']['logs']= 'CTDprofilesEnds.txt'
cfg['program']['veusz_path'] = u'C:\\Program Files (x86)\\Veusz'  # directory of Veusz
load_vsz = load_vsz_closure(Path(cfg['program']['veusz_path']))
tbl_sec_points = '1608_CTDsections_waypoints'
tblN = 'navigation'  # '/navigation/table' after saving without no data_columns= True
search_nav_tolerance = timedelta(minutes=1)
t_our_zone = timedelta(hours=2)

cfg['vsz_files']['filemask'] = '*.vsz'

cfg['gpx']['symbol_break_keeping_point'] = 'Circle, Red'  # will not keeping to if big dist:
cfg['gpx']['symbol_break_notkeeping_dist'] = 20  # km
cfg['gpx']['symbol_excude_point'] = 'Circle with X'
b_filter_time = False
# dt_point2run_max= timedelta(minutes=15)

fileOutP = os_path.split(fileInFH5)[0]
fileOutTempP = os_path.join(fileOutP, 'subproduct')
# ----------------------------------------------------------------------
# dir_walker
vszFs = [os_path.join(cfg['vsz_files']['path'], f) for f in os_listdir(
    cfg['vsz_files']['path']) if fnmatch(f, cfg['vsz_files']['filemask'])]
print('Process {} sections'.format(len(vszFs)))
# Load data #################################################################
bFirst = True
timeEnd_Last = np.datetime64('0')
f = None;
g = None;
LCTD = []
h5storageTemp = os_path.join(fileOutTempP, 'CTD+Nav.h5')
try:
    if not os_path.isfile(h5storageTemp):
        # __vsz data to hdf5__
        with pd.HDFStore(fileInFH5, mode='r') as storeIn:
            try:  # Sections
                df_points = storeIn[tbl_sec_points]  # .sort()
            except KeyError as e:
                print('Sections not found in DataBase')
                raise e
            bExclude = df_points.sym == cfg['gpx']['symbol_excude_point']
            dfpExclude = df_points[bExclude]
            df_points = df_points[~bExclude]
            iStEn, b_navp_all_exclude, ddist_all = sec_edges(df_points, cfg)

            colNav = ['Lat', 'Lon', 'Dist']
            # colCTD = [u'Pres', u'Temp', u'Sal', u'SigmaTh', u'O2', u'ChlA', u'Turb', u'pH', u'Eh']  # u'ChlA', , u'Turb'
            for isec, vszF in enumerate(vszFs, start=1):
                # Load filtered down runs data
                if isec < startSectionN:
                    continue
                print('\n{}. {}'.format(isec, vszF))
                g, CTD = load_vsz(vszF, veusze=g, prefix='CTD')
                if 'ends' not in CTD:
                    print('data error!')

                # Add full resolution nav to section from navigation['Lat', 'Lon']
                qstr_trange_pattern = "index>=Timestamp('{}') & index<=Timestamp('{}')"
                qstr = qstr_trange_pattern.format(*CTD['time'][[0, -1]])
                Nav = storeIn.select(tblN, qstr, columns=['Lat', 'Lon'])  # DepEcho
                qstr = qstr_trange_pattern.format(CTD['time'][0], Nav.index[0])
                # Nind_st = storeIn.select_as_coordinates(tblN, "index=={}".format(Nav.index[0]))[0]
                Nind = inearestsorted(Nav.index, CTD['time'])
                # Check time difference between nav found and time of requested points
                dT = Nav.index[Nind] - CTD['time']
                if np.any(abs(dT) > search_nav_tolerance * 5):
                    print('bad nav data coverage')
                for col in Nav.columns:
                    CTD[col] = np.interp(CTD['time'].view(np.int64),
                                         Nav.index.values.view(np.int64),
                                         Nav[col])
                cols = list(CTD.keys()).copy()
                cols.remove('time')
                cols = [col for col in cols if CTD[col].size == CTD['time'].size]

                # with pd.HDFStore(fileInFH5, mode='w') as storeIn:
                LCTD.append(fromarrays([CTD['time']] + [v for k, v in CTD.items() if k in cols],
                                       names=['time'] + cols, formats=['M8[ns]'] + ['f8'] * len(cols)))
                # this creates strange ncols x len(array) instead of size (len(array),)
                # np.array([CTD['time']] + [v.ravel() for k, v in CTD.items() if k != 'time'],
                #          dtype={'names': ['time'] + cols,
                #                 'formats': ['M8[ns]'] + ['f8'] * len(cols)})
                # with pd.HDFStore(fileInFH5, mode='w') as storeIn:
        FCTD = pd.DataFrame.from_records(np.hstack(LCTD), index='time')
        with pd.HDFStore(h5storageTemp, mode='w') as storeOut:
            storeOut.put('CTD', FCTD, format='table', data_columns=['Pres', 'Lat', 'Lon'])
        print('source data, combined with nav. saved to storage ', h5storageTemp)
    else:
        # Gridding

        pst = 10;
        pen = 80;
        pstep = 5;
        pwidth = 5  # m

        y_resolution = 0.5  # km
        qstr_pattern = "Lon >= 16.15 and Lon < 17.00 and Pres >= {} and Pres < {}"
        maxDiff = {'Lat': 0.1, 'Lon': 0.1}

        # 1.standard 2D
        params = [u'Temp', u'Sal', u'SigmaTh', u'O2', u'ChlA', u'Turb', u'pH', u'Eh']  # u'ChlA', , u'Turb'
        b1st_hor = True
        print('Gridding on horisonts:', end=' ')
        y_resolution /= (1.852 * 60)  # %km2deg(1)
        for p, p_st, p_en in np.arange(pst, pen, pstep)[:, np.newaxis] + np.array(
                [[0, -pwidth, pwidth]]) / 2:  # [:, np.]
            print('\n{:g}m.'.format(p), end=' ')
            qstr = qstr_pattern.format(p_st, p_en)
            FCTD = pd.read_hdf(h5storageTemp, 'CTD', where=qstr)
            if b1st_hor:
                b1st_hor = False
                time_st_local, time_en_local = [timzone_view(x, t_our_zone) for x in FCTD.index[[0, -1]]]
                fileN_time = '{:%y%m%d_%H%M}-{:%d_%H%M}'.format(time_st_local, time_en_local)
            iruns = np.flatnonzero(np.diff(FCTD['shift']) != 0)
            CTD = np.empty((iruns.size + 1,), {'names': params + ['Lat', 'Lon'],
                                               'formats': ['f8'] * (len(params) + 2)})
            # Average data for each run
            for param in CTD.dtype.names:
                if param not in maxDiff: maxDiff[param] = np.NaN
                for irun, isten in enumerate(np.column_stack((np.append(1, iruns),
                                                              np.append(iruns, len(FCTD))))):
                    data_check = FCTD[param][slice(*isten)]
                    bBad = np.abs(np.diff(data_check)) > maxDiff[param]
                    if np.any(bBad):
                        print('bad data ({}) in {}'.format(np.sum(bBad), param), end=', ')
                        plt.scatter(data_check.index, data_check)
                        plt.scatter(data_check.index[:-1][bBad], data_check[:-1][bBad], marker='+', c='r', s=15,
                                    zorder=10)
                        # delete outlines:
                        bBad = abs(data_check - np.nanmean(data_check)) > maxDiff[param]
                        data_check[bBad] = np.NaN
                    CTD[param][irun] = np.nanmean(data_check)

            # Save run track path
            # print('Saving CTD run track path')
            np.savetxt(os_path.join(fileOutTempP, fileN_time + ',{:g}m_points.csv'.format(p)),
                       np.vstack((CTD['Lat'], CTD['Lon'])).T,
                       fmt='%g,%g', header='Lon,Lat', delimiter=',', comments='')

            x_min = np.min(CTD['Lon'])
            x_max = np.max(CTD['Lon'])
            y_min = np.min(CTD['Lat'])
            y_max = np.max(CTD['Lat'])
            Lat_mean = (y_min + y_max) / 2
            x_resolution = y_resolution * np.cos(np.deg2rad(Lat_mean))
            gdal_geotransform = (x_min, x_resolution, 0, -y_min, 0, -y_resolution)
            write_grd_this_geotransform = write_grd_fun(gdal_geotransform)
            x = np.arange(x_min, x_max, x_resolution)
            y = np.arange(y_min, y_max, y_resolution)
            # todo: Interpolate path in grid coordinates for blanking: use filter of border to not delete too much
            y_edge_path = []
            xm, ym = np.meshgrid(x, y)
            # bot_edge_add = y_resolution  # m
            # ym_blank = ym > bot_edge_add - y_edge_path

            print('{}x{} - '.format(*xm.shape), end='')

            # [['surfer', 'surfer']]

            griddata_by_surfer(CTD, outFnoE_pattern=os_path.join(fileOutTempP,
                                                                 'surfer', '{}' + ',{:g}m'.format(p)), xCol='Lon',
                               yCol='Lat',
                               zCols=params, NumCols=y.size, NumRows=x.size,
                               xMin=x_min, xMax=x_max, yMin=y_min, yMax=y_max)
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
                bGood = np.isfinite(CTD[param])  # np.logical_and(, bGoodP)

                interp_method_dir = []  # [['linear', ''], ['cubic', 'cubic\\']]
                for interp_method, interp_method_subdir in interp_method_dir:
                    z = interpolate.griddata((np.append(CTD['Lon'], [])[bGood],
                                              np.append(CTD['Lat'], [])[bGood]),
                                             np.append(CTD[param], [])[bGood], (xm, ym),
                                             method=interp_method)  # , rescale= True'cubic','linear','nearest'
                    # Blank z below bottom (for compressibility)
                    # z[ym_blank] = np.NaN

                    fileFN = os_path.join(fileOutTempP, interp_method_subdir,
                                          fileN_time + label_param + ',{:g}m'.format(p))
                    if __debug__:
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
                                plt.plot(CTD['Lon'], CTD['Lat'], color='m', alpha=0.5, label='run path')
                            CBI = plt.colorbar(im, shrink=0.8)
                            # CB = plt.colorbar(CS, shrink=0.8, extend='both')
                            # plt.gcf().set_size_inches(9, 3)
                            plt.savefig(fileFN + '.png')  # , dpi= 200
                            # plt.show()
                            plt.close()
                            pass
                        except Exception as e:
                            print('\nCan no draw contour! ', e.__class__, ':', '\n==> '.join(
                                [a for a in e.args if isinstance(a, str)]))
                    # gdal_drv_grid.Register()

                    write_grd_this_geotransform(fileFN + '.grd', z)
                if bFirst:
                    bFirst = False

except Exception as e:
    print('\nError! ', e.__class__, ':', '\n==> '.join(
        [a for a in e.args if isinstance(a, str)]))
    raise e
