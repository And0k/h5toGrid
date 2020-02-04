#!/usr/bin/env python
# coding:utf-8
"""
  Author:  Andrey Korzh <ao.korzh@gmail.com>
  Purpose: load filtered CTD data from Veusz sections, add nav. from Veusz data source store
  Created: 16.08.2016
"""

from pathlib import Path
import numpy as np
import pandas as pd

from os import listdir as os_listdir, path as os_path
from fnmatch import fnmatch

from datetime import datetime, timedelta
from gsw import distance
import pyproj

from veuszPropagate import load_vsz_closure
from grid2d_vsz import runs_ilast_good, sec_edges, track_b_invert
from other_filters import inearestsorted

# idata_from_tpoints

# ##############################################################################
cfg = {'input_h5store': {}, 'process': {}, 'gpx': {}, 'vsz_files': {}, 'output_files': {}}

cfg['input_h5store']['path'] = r'd:\WorkData\BalticSea\160802_ANS32\160802_Strahov.h5'
cfg['input_h5store']['tbl_sec_points'] = '1609_CTDsections_waypoints'
cfg['input_h5store']['tbl_nav'] = 'navigation'  # '/navigation/table' after saving without no data_columns= True

cfg['vsz_files']['subdir'] = 'CTD-zabor'
cfg['vsz_files']['path'] = os_path.join(os_path.dirname(cfg['input_h5store']['path']), cfg['vsz_files']['subdir'])
cfg['vsz_files']['filemask'] = '*.vsz'

cfg['gpx']['symbol_break_keeping_point'] = 'Circle, Red'  # will not keeping to if big dist:
cfg['gpx']['symbol_break_notkeeping_dist'] = 20  # km
cfg['gpx']['symbol_excude_point'] = 'Circle with X'
cfg['process']['dt_search_nav_tolerance'] = timedelta(minutes=1)
cfg['process']['invert_prior_sn_angle'] = 30

cfg['output_files']['skip_to_section'] = 11  # 1 - no skip. > 1 - skipped sections
cfg['output_files']['path'] = os_path.join(os_path.dirname(cfg['input_h5store']['path']), 'subproduct')
cfg['output_files']['dt_from_utc'] = timedelta(hours=2)
cfg['program'] = {'log': os_path.join(cfg['output_files']['path'], 'S&S_CTDprofilesEnds.txt')}  # common data out
cfg['program']['logs'] = 'CTDprofilesEnds.txt'  # separate file for each section suffix
cfg['program']['veusz_path'] = u'C:\\Program Files (x86)\\Veusz'  # directory of Veusz
load_vsz = load_vsz_closure(Path(cfg['program']['veusz_path']))

b_filter_time = False
# dt_point2run_max= timedelta(minutes=15)

if not os_path.isdir(cfg['output_files']['path']):
    raise (FileNotFoundError('output dir not exist: ' + cfg['output_files']['path']))
# ----------------------------------------------------------------------
# dir_walker
vszFs = [os_path.join(cfg['vsz_files']['path'], f) for f in os_listdir(
    cfg['vsz_files']['path']) if fnmatch(f, cfg['vsz_files']['filemask'])]
print('Process {} sections'.format(len(vszFs)))
# Load data #################################################################
bFirst = True
timeEnd_Last = np.datetime64('0')
f = None
g = None
try:

    with pd.HDFStore(cfg['input_h5store']['path'], mode='r') as storeIn:
        try:  # Sections
            df_points = storeIn[cfg['input_h5store']['tbl_sec_points']]  # .sort()
        except KeyError as e:
            print('Sections not found in DataBase')
            raise e
        iStEn, b_navp_all_exclude, ddist_all = sec_edges(df_points, cfg['gpx'])
        dfpExclude = df_points[b_navp_all_exclude]
        df_points = df_points[~b_navp_all_exclude]

    colNav = ['Lat', 'Lon', 'Dist']
    colCTD = [u'Pres', u'Temp', u'Sal', u'SigmaTh', u'O2', u'ChlA', u'Turb', u'pH', u'Eh']  # u'ChlA', , u'Turb'
    for isec, vszF in enumerate(vszFs, start=1):
        # Load filtered down runs data
        if isec < cfg['output_files']['skip_to_section']:
            continue
        print('\n{}. {}'.format(isec, vszF))
        g, CTD = load_vsz(vszF, veusze=g, prefix='CTD')
        if 'ends' not in CTD:
            print('data error!')
            g = None

        # Output structure
        Ncols_noTime = len(colNav) + len(colCTD) + 1
        CTDout = np.empty_like(CTD['ends'], dtype={'names': ['timeEnd'] + colNav + colCTD + ['O2_min'],
                                                   'formats': ['|S20'] + ['f8'] * Ncols_noTime})
        # Fill it
        temp = CTD['time'][CTD['ends']] + np.timedelta64(cfg['output_files']['dt_from_utc'])
        CTDout['timeEnd'][:] = ['{:%d:%m:%y %H:%M:%S}'.format(t) for t in temp.astype(
            'M8[s]').astype(datetime)]
        # 1. Load existed bottom edge parameters:
        temp = load_vsz(veusze=g, prefix='botO2')[1]  # min O2 after delay near bottom
        CTDout['O2_min'][:] = temp['min']
        # 2. Calc bottom values
        for param in colCTD:  # param= u'Temp'
            CTDout[param][:] = np.NaN

            print(param, end='')
            if param not in CTD:
                print(' not loaded')
                continue
            if __debug__:
                sys_stdout.flush()

            bGood = np.logical_and(np.isfinite(CTD[param]), np.isfinite(CTD['Pres']))
            i_add, brun_add = runs_ilast_good(bGood, CTD['starts'], CTD['ends'], CTD['Pres'])
            CTDout[param][brun_add] = CTD[param][i_add]
            print(', ', end='')
        # 3. Add navigation data for points used in secion
        # lrunsUsed, bUsed = idata_from_tpoints(df_points.index.values, CTD['time'],
        #                                       CTD['starts'], dt_point2run_max)
        ipoints = inearestsorted(df_points.index, CTD['time'][CTD['starts']])
        CTDout['Lat'][:] = df_points.Lat[ipoints].values
        CTDout['Lon'][:] = df_points.Lon[ipoints].values

        # Not use nav where time difference between nav found and time of requested points is big
        dT = df_points.index[ipoints] - CTD['time'][CTD['starts']]
        bBad = abs(dT) > cfg['process']['dt_search_nav_tolerance'] * 5
        if np.any(bBad):
            print('Bad nav. data coverage: difference to nearest point in time [min]:')
            print('\n'.join(['{}. {}:\t{}{:.1f}'.format(i, tdat, m, dt.seconds / 60) for i, tdat, m, dt in
                             zip(np.flatnonzero(bBad), CTD['time'][CTD['starts'][bBad]],
                                 np.where(dT[bBad].astype(np.int64)
                                          < 0, '-', '+'), np.abs(dT[bBad]))]))
            CTDout['Lat'][bBad] = np.NaN
            CTDout['Lon'][bBad] = np.NaN

        # Calculate distance #
        # Exclude points removed from tracks manually
        if dfpExclude.size:
            ipoints_del = inearestsorted(dfpExclude.index, CTD['time'][CTD['starts']])
            dT = dfpExclude.index[ipoints_del] - CTD['time'][CTD['starts']]
            bUseDist = abs(dT) > cfg['process']['dt_search_nav_tolerance'] * 5
            bUseDist = bUseDist & np.logical_not(bBad)
        else:
            bUseDist = np.logical_not(bBad)
        # Calculate
        ddist = distance(CTDout['Lon'][bUseDist], CTDout['Lat'][bUseDist]) / 1e3  # km
        print('Distances between CTD runs:', ddist)
        if np.any(np.isnan(ddist)):
            print('(!) Found NaN coordinates in nav at indexes: ',
                  np.flatnonzero(np.isnan(ddist)))  # dfNpoints['Lat'][]
        dist = np.cumsum(np.append(0, ddist))
        # Invert if need
        ist, ien = np.flatnonzero(bUseDist)[[0, -1]]
        # course = geog.course((CTDout['Lon'][ist], CTDout['Lat'][ist]),
        #                      (CTDout['Lon'][ien], CTDout['Lat'][ien]),
        #                      bearing=True)
        geod = pyproj.Geod(ellps='WGS84')
        course, azimuth2, distance = geod.inv(
            CTDout['Lon'][ist], CTDout['Lat'][ist],
            CTDout['Lon'][ien], CTDout['Lat'][ien])

        b_invert, strInvertReason = track_b_invert(course, cfg['process']['invert_prior_sn_angle'])
        if b_invert: dist = dist[-1] - dist

        CTDout['Dist'][:] = np.NaN
        CTDout['Dist'][bUseDist] = dist

        # Save to files #
        format_str = '{:s}' + '\t{:f}' * Ncols_noTime + '\n'

        if 'logs' in cfg['program'].keys():
            fileN_time_st = os_path.splitext(os_path.basename(vszF))[0]  # '{:%y%m%d_%H%M}'.format(CTDout['Dist'])
            with open(os_path.join(cfg['output_files']['path'], fileN_time_st + cfg['program']['logs']), 'w') as fs:
                fs.writelines('\t'.join(CTDout.dtype.names) + '\n')
                for row in CTDout:
                    fs.writelines(format_str.format(*row)[1:])

        if b_filter_time:
            # filter time
            if temp[0] < timeEnd_Last:
                print('time decreased!')
            elif temp[0] == timeEnd_Last:
                CTDout = CTDout[1:]
            timeEnd_Last = temp[-1]

        if 'log' in cfg['program'].keys():
            if bFirst:
                bFirst = False
                f = open(cfg['program']['log'], 'w')
                f.writelines('\t'.join(CTDout.dtype.names) + '\n')
            for row in CTDout:
                f.writelines(format_str.format(*row)[1:])  # , '%s\t%s %s'
    #              if __debug__:
    #                 f.flush()
    if 'log' in cfg['program'].keys():
        f.close()
except Exception as e:
    raise e
finally:
    if f:
        f.close()
