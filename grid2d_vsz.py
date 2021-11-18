#!/usr/bin/env python
# coding:utf-8
# from __future__ import print_function, division
"""
  Author:  Andrey Korzh <ao.korzh@gmail.com>
  Purpose:
load sections divisions from pandas hdf5 storage,
use vsz pattern to filter and visualise source data,
grid CTD sections
  Created: 16.08.2016
  Modified: 20.08.2021
"""

import logging
from pathlib import Path
from sys import stdout as sys_stdout, platform
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union
from collections import namedtuple

import numpy as np
import pandas as pd

from osgeo import gdal, ogr  # I have to add OSError to try-except in module's __init__ to ignore "." in sys.path
# except ModuleNotFoundError as e:
#     print(e.args[0], ' - may be not needed. Continue!')
import pyproj  # import geog
from third_party.descartes.patch import PolygonPatch  # !Check!
from gsw import distance as gsw_distance  # from gsw.gibbs.earth  import distance
# from scipy.ndimage import gaussian_filter1d
from scipy import interpolate
from scipy.signal import medfilt
from scipy.ndimage.filters import gaussian_filter1d
from shapely.geometry import MultiPolygon, asPolygon, Polygon

from graphics import make_figure, interactive_deleter
from other_filters import rep2mean, is_works, too_frequent_values, waveletSmooth, despike, i_move2good, \
    inearestsorted, closest_node
from to_pandas_hdf5.CTD_calc import add_ctd_params
from to_pandas_hdf5.h5toh5 import h5select
# my
from utils2init import init_logging, Ex_nothing_done, standard_error_info
from utils_time import datetime_fun, timzone_view, multiindex_timeindex, check_time_diff
from veuszPropagate import load_vsz_closure, export_images  # , veusz_data

Axes2d = namedtuple('axes2d', ('x', 'y'))
MinMax = namedtuple('MinMax', ('min', 'max'))
ax2col = Axes2d('dist', 'depth')

# graphics/interactivity
if True:  # __debug__:
    import matplotlib

    matplotlib.rcParams['axes.linewidth'] = 1.5
    matplotlib.rcParams['figure.figsize'] = (16, 7)
    try:
        matplotlib.use('Qt5Agg')  # must be before importing plt (raises error after although docs said no effect)
    except ImportError:
        pass
    from matplotlib import pyplot as plt

    matplotlib.interactive(True)
    plt.style.use('bmh')

if __name__ == '__main__':
    l = None
else:
    l = logging.getLogger(__name__)
load_vsz = None


def my_argparser():
    """
    Configuration parser options and its description
    :return p: configargparse object of parameters
    """
    from utils2init import my_argparser_common_part

    p = my_argparser_common_part({'description':
                                      'Grid data from Pandas HDF5, VSZ files '
                                      'and Pandas HDF5 store*.h5'})

    s = p.add_argument_group('in', 'data from hdf5 store')
    s.add('--db_path', help='hdf5 store file path')  # '*.h5'
    s.add('--table_sections', help='table name with sections waypoints data')
    s.add('--table_nav', default='navigation',  # '/navigation/table' after saving without no data_columns= True
             help='table name with sections waypoints data')
    s.add('--b_temp_on_its90', default='True',
             help='When calc CTD parameters treat Temp have red on ITS-90 scale. (i.e. same as "temp90")')

    s = p.add_argument_group('vsz_files', 'data from hdf5 store')
    s.add('--subdir', default='CTD-sections', help='Path to source file(s) to parse')
    s.add('--filemask', default='[0-9]*.vsz',
              help='path mask to Veusz pattern file(s). If any files found has names that starts with current section '
                   'time formatted "%y%m%d_%H%M" then use 1st of it without modification. Else use last name that '
                   'conforms to filemask as pattern')
    s.add('--export_pages_int_list', default='0',
              help='pages numbers to export, comma separated (1 is first), 0= all')
    s.add('--export_dir', default='images(vsz)',
              help='subdir relative to input path or absolute path to export images')
    s.add('--export_format', default='png',
              help='extention of images to export which defines format')

    s = p.add_argument_group('gpx', 'symbols names')
    s.add('--symbol_break_keeping_point', default='Circle, Red', help='to break all data to sections')
    s.add('--symbol_break_notkeeping_dist_float', default='20', help='km, will not keeping to if big dist')
    s.add('--symbol_excude_point', default='Circle with X', help='to break all data to sections')
    s.add('--symbols_in_veusz_ctd_order_list',
              help="GPX symbols of section in order of Veusz joins tables of CTD data (use if section has data from several tables, see CTDitable variable in Veusz file). Used only to exclude CTD runs if its number bigger than number of section's points.")  # todo: use names of tables which Veusz loads
    s = p.add_argument_group('out',
                                 'Output files: paths, formats... - not calculation intensive affecting parameters')
    s.add('--subdir_out', default='subproduct', help='path relative to in.db_path')
    s.add('--dt_from_utc_hours', default='0')
    s.add('--x_resolution_float', default='0.5',
              help='Dist, km. Default is 0.5km, but if dd = length/(nuber of profiles) is less then decrease to dd/2')
    s.add('--y_resolution_float', default='1.0', help='Depth, m')
    s.add('--blank_level_under_bot_float', default='-300',
              help='Depth, m, that higher than maximum of plot y axis to not see it and to create polygon without self intersections')
    s.add('--data_columns_list',
              help='Comma separated string with data column names (of hdf5 table) to use. Not existed will skipped')
    s.add('--b_reexport_images',
              help='Export images of loaded .vsz files (if .vsz creared then export ever)')

    s = p.add_argument_group('process', 'process')
    s.add('--begin_from_section_int', default='0', help='0 - no skip. > 0 - skipped sections')
    s.add('--interact', default='editable_figures',
               help='if not "False" then display figures where user can delete data and required to press Q to continue')
    s.add('--dt_search_nav_tolerance_seconds', default='1',
               help='start interpolate navigation when not found exact data time')
    s.add('--invert_prior_sn_angle_float', default='30',
               help='[0-90] degrees: from S-N to W-E, 45 - no priority')
    s.add('--depecho_add_float', default='0', help='add value to echosounder depth data')
    s.add('--convexing_ctd_bot_edge_max_float', default='0',  # filter_ctd_bottom_edge_float, min_ctd_end_as_bot_edge
               help='filter ctd_bottom_edge line closer to bottom (to be convex) where its depth is lesser')
    s.add('--min_depth', default='4', help='filter out smaller depths')
    s.add('--max_depth', default='1E5', help='filter out deeper depths')
    s.add('--filter_depth_wavelet_level_int', default='4', help='level of wavelet filtering of depth')
    # s.add(
    #     '--dt_point2run_max_minutes', #default=None,
    #     help='time interval to sinchronize points on map and table data (to search data marked as excluded on map i.e. runs which start time is in (t_start_good, t_start_good+dt_point2run_max). If None then select data in the range from current to the next point')

    s = p.add_argument_group('program', 'program behaviour')
    s.add('--veusz_path',
               default=u'C:\\Program Files (x86)\\Veusz' if platform == 'win32' else u'/home/korzh/.virtualenvs/veusz_experiments/lib/python3.6/site-packages/veusz-2.2.2-py3.6-linux-x86_64.egg/veusz',
               help='directory of Veusz')

    return p


# ------------------------------------------------------------------------
def check_time_dif(tstart, t64st_data, dt_max=0, data_name=''):
    """
    Check time difference between time of data found and time of requested points
    :param tstart: datetimeindex of requested points
    :param t64st_data: time of data found (numpy.datetime64)
    :param dt_max:
    :param data_name:
    :return: datetimeindex of found difference
    Shows differences
    """
    diffValsFound = tstart.difference(t64st_data)
    diffValsFound = [d for d in diffValsFound.values if d not in t64st_data]  # ?
    if len(diffValsFound):
        print('Found {} to {} data differences: '.format(len(diffValsFound), data_name))
        print('All points time was | found time'.format(tstart, t64st_data))
        [print('{}|{}'.format(t, nt)) for t, nt in zip(tstart, t64st_data)]
    return diffValsFound


def write_grd_fun(gdal_geotransform):
    gdal_drv_grid = gdal.GetDriverByName('GS7BG')

    def write_grd1(file_grd, z):
        nonlocal gdal_drv_grid, gdal_geotransform
        gdal_raster = gdal_drv_grid.Create(str(file_grd), z.shape[1], z.shape[0], 1, gdal.GDT_Float32)  #
        if gdal_raster is None:
            l.error('Could not create %s', file_grd)
            # continue #sys.exit(1)
        # georeference the image and set the projection
        gdal_raster.SetGeoTransform(gdal_geotransform)
        # gdal_raster.SetProjection(inDs.GetProjection())
        # , src_ds.RasterXSize, src_ds.RasterYSize, out_bands, gdal.GDT_Byte
        outBand = gdal_raster.GetRasterBand(1)

        z = np.float32(z)
        GSSurfer_no_data_value = np.float32(1.70141E+038)
        z = np.where(np.isnan(z), GSSurfer_no_data_value, z)  # 1.7014100091878E+038
        outBand.WriteArray(z)  # .WriteRaster(z)
        # flush data to disk, set the NoData value and calculate stats
        # outBand.SetNoDataValue(1.70141e+38)
        outBand.FlushCache()

    return write_grd1


def save_shape(target_path, shapelyGeometries, driverName='ESRI Shapefile'):
    target_path = Path(target_path)
    if target_path.suffix:
        pass
    elif driverName == 'BNA':
        target_path = target_path.with_suffix('.bna')
    elif driverName == 'ESRI Shapefile':
        target_path = target_path.with_suffix('.shp')
    """
    Save shapelyGeometries using the given proj4 and fields
    """
    # As of ver 3.3 BNA is not supported by GDAL (still avalable at https://github.com/OSGeo/gdal-extra-drivers)
    # there is my adhoc method to write BNA. Only exterior polygons in "Polygon" or "MultiPolygon" will be saved
    if driverName == 'BNA':
        def write_bna(f, geom):
            bln = np.asarray(geom.exterior.coords)
            header = f'"","",{bln.shape[0]}'
            np.savetxt(f, bln, fmt='%g', delimiter=',', header=header,
                       comments='', encoding='ascii')

        with target_path.open('w') as f:
            if shapelyGeometries.geom_type == "Polygon":
                write_bna(f, shapelyGeometries)
            elif shapelyGeometries.geom_type == "MultiPolygon":
                for geom in shapelyGeometries.geoms:
                    write_bna(f, geom)

        return target_path

    ogr_drv = ogr.GetDriverByName(driverName)
    if not ogr_drv:
        raise GeometryError('Could not load driver: {}'.format(driverName))

    # Make dataSource
    if target_path.exists():
        target_path.unlink()
    dataSource = ogr_drv.CreateDataSource(str(target_path))

    # Add one attribute
    fieldDefinitions = [('id', ogr.OFTInteger)]

    # Make layer
    spatialReference = None  # get_spatialReference(targetProj4 or sourceProj4)
    geometryType = ogr.wkbPolygon  # get_geometryType(shapelyGeometries)
    layer = dataSource.CreateLayer(target_path.stem, spatialReference, geometryType)
    # Make fieldDefinitions in featureDefinition
    for fieldName, fieldType in fieldDefinitions:
        layer.CreateField(ogr.FieldDefn(fieldName, fieldType))
    # Save features
    feature = ogr.Feature(layer.GetLayerDefn())
    feature.SetGeometry(ogr.CreateGeometryFromWkb(shapelyGeometries.wkb))
    layer.CreateFeature(feature)
    feature.Destroy()  # Clean up
    # Return
    return target_path


class GeometryError(Exception):
    """Exception raised when there is an error loading or saving geometries"""
    pass


def plot_polygon(polygon, color='#999999'):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    margin = .5
    x_min, y_min, x_max, y_max = polygon.bounds
    ax.set_xlim([x_min - margin, x_max + margin])
    ax.set_ylim([y_min - margin, y_max + margin])
    patch = PolygonPatch(polygon, fc=color,
                         ec='#000000', fill=True,
                         zorder=-1)
    ax.add_patch(patch)
    return ax


def to_polygon(x: Sequence, y: Sequence, y_add_at_edges: Union[int, Sequence]) -> Polygon:
    """
    Create Shapely polygon from not closed line
    :param x: line x coordinates
    :param y: line y coordinates
    :param y_add_at_edges: (end, start) - values to add before_close line
    :return:
    """
    temp_DistDep = np.vstack((x, y))
    try:
        if not len(y_add_at_edges) == 2:
            y_add_at_edges = y_add_at_edges[:2]
    except:
        y_add_at_edges = [y_add_at_edges] * 2
    temp_DistDep = np.hstack((temp_DistDep, [temp_DistDep[0, [-1, 0]], y_add_at_edges]))
    # np.pad(, ((0, 0), (0, 1)), 'wrap')
    #
    p = asPolygon(temp_DistDep.T)
    if not p.is_valid:
        p = p.buffer(0)
    if False:
        ax = plot_polygon(p)
        ax.plot(*temp_DistDep, 'o', color='y')
    return p


distance = lambda lon, lat: gsw_distance(lon, lat, p=np.float(0))  # fix gsw bug of require input of numpy type for "p"


def dist_clc(nav, ctd_time, cfg):
    """
    Calculate distance
    selects good points for calc. distance
    :param nav:
    :return:
    """
    useLineDist = cfg['process'].get('useLineDist', 0.05)
    pass  # Not implemented


def dist_ctd(time_ctd,
             time_points_st,
             time_points_en,
             lonlat_points_st,
             lonlat_points_en
             ):
    """
    Calculate ctd distance along its path
    :param time_ctd: time where distance calculation needed, may be not sorted
    :param time_points_st: int64, time array corresponded to lonlat_points_st
    :param time_points_en: int64, time array corresponded to lonlat_points_en
    :param lonlat_points_st: 2xNpoints array, coordinates (lon, lat) of CTD runs starting points
    :param lonlat_points_en: 2xNpoints array, coordinates (lon, lat) of CTD runs ending points
    :return: distance, run_time_topbot, run_dist_topbot

    Note: This function is replacement of this old simpler code that not accounts for movement during lowering:
    _ = np.hstack([np.full(shape=sh, fill_value=v) for sh, v in zip(
        np.diff(np.append(ctd_prm['starts'], ctd.time.size)), run_dist)])
    ctd['dist'] = _[:ctd.depth.size] if _.size > ctd.depth.size else _

    """
    # df_points.iloc[:ctd_prm['starts'].size, df_points.columns.get_indexer(('Lon', 'Lat'))].values.T

    ddist_points_st = distance(*lonlat_points_st) * 1e-3  # km, distance between starting points
    l.info('Distances between profiles: {}'.format(np.round(ddist_points_st, int(2 - np.log10(np.mean(ddist_points_st))))))

    if np.any(np.isnan(ddist_points_st)):
        l.error('(!) Found NaN coordinates in navigation at indexes: ',
                np.flatnonzero(np.isnan(ddist_points_st)))

    # Project end points on lines between start points

    # next starts relative to starts (last point is extrapolated by adding same distance in same direction as previous)
    ab = (lambda x: np.append(x, x[:, -1:], axis=1))(np.diff(lonlat_points_st, axis=1))
    # CTD ends relative to starts
    ap = lonlat_points_en - lonlat_points_st
    lonlat_ends_proj = lonlat_points_st + np.einsum('ij,ij->j', ap, ab) / np.sum(ab ** 2, axis=0) * ab
    if False:  # plot
        f, ax = plt.subplots()
        ax.plot(*lonlat_points_st, ':xg', label='starts')
        ax.plot(*lonlat_ends_proj, '.r', label='proj')
        ax.plot(*lonlat_points_en, 'k+', label='ends')
        ax.legend(prop={'size': 10}, loc='lower left')
    # insert ends between starts to calculate distances between them all in time order
    lonlat_topbot = np.empty((2, lonlat_points_st.shape[1]*2), dtype=lonlat_points_st.dtype)
    lonlat_topbot[:, ::2] = lonlat_points_st  # starts
    lonlat_topbot[:, 1::2] = lonlat_ends_proj  # projected ends
    ddist_topbot = distance(*lonlat_topbot) * 1e-3
    # Recover projection sign
    # If distance of (start to end) + (next start to end) > (start to next start) then end is not between.
    # If it is also nearer to start then it is in opposite dir.
    # Last end-point is in back direction if it is between prev start and start or else nearer to prev start
    ddist_last2st_prev = distance(*lonlat_topbot[:, [-4, -1]]) * 1e-3  # -4 is index of prev start, -1 of last end-point
    err = 1e-5  # (1cm) - max numeric error, km  (on check it was < 10^-8 km)
    bback = np.append(
        (ddist_topbot[:-1:2] + ddist_topbot[1::2] > ddist_points_st + err) &
        (ddist_topbot[:-1:2] < ddist_topbot[1::2]),
        (ddist_topbot[-1:] + ddist_last2st_prev < ddist_points_st[-1] + err) | (ddist_last2st_prev < ddist_topbot[-1:])
        )
    ddist_topbot[::2][bback] = -ddist_topbot[::2][bback]
    run_dist_topbot = np.cumsum(np.append(0, ddist_topbot))

    # shift start to zero (used .min() instead of [0] because movement back allowed)
    run_dist_topbot = run_dist_topbot - run_dist_topbot.min()
    run_time_topbot = np.empty_like(run_dist_topbot)
    run_time_topbot[::2] = time_points_st
    run_time_topbot[1::2] = time_points_en
    i_run_time_topbot = run_time_topbot.argsort()
    run_time_topbot[:] = run_time_topbot
    run_dist_topbot[:] = run_dist_topbot
    return (
        np.interp(time_ctd, run_time_topbot[i_run_time_topbot], run_dist_topbot[i_run_time_topbot]),
        run_time_topbot,
        run_dist_topbot
        )


def closest_point_on_line(a, b, p):
    """
    Project a point P onto a line AB
    Algorithm: projects vector AP onto vector AB, then add the resulting vector to point A
    :param a:
    :param b:
    :param p:
    :return:
    """
    ap = p - a
    ab = b - a
    result = a + np.dot(ap, ab) / np.dot(ab, ab) * ab
    return result


def track_b_invert(Course, angle_use_SN_max=10):
    """
    function [bInvert bNS]= directionChanged(Course, bLonsEntoSt, angle_use_SN_max)
    Is it need to invert track direction for calculate distance?
    Outputs:
     bInvert = true if need to invert
    Inputs:
     Course - course of track;
     angle_use_SN_max= [0-90 degrees] N->S priority aganest E->W.
    90 - max N->S priority, 0 - max W->E priority. At 45 are no priorities.
    if angle_use_SN_max<0 direction is always changed
    """
    # bNS= True
    if angle_use_SN_max is None:
        angle_use_SN_max = 10
    if isinstance(Course, float):
        C = Course
    else:
        try:
            C = Course[0].copy()
        except:
            C = Course.copy()
    C %= 360
    if abs(C - 180) < angle_use_SN_max:
        bInvert = False;
        msg_inv_reason = 'NS'  # N->S
    else:
        if angle_use_SN_max < 0:
            bInvert = True;
            msg_inv_reason = '!-!'
        elif (C < angle_use_SN_max) or ((360 - C) < angle_use_SN_max):
            bInvert = True;
            msg_inv_reason = 'SN!'  # S->N
        else:
            bInvert = C > 180  # E->W
            msg_inv_reason = 'EW!' if bInvert else 'WE'
            # bNS= False
    return bInvert, msg_inv_reason  # , bNS


def sec_edges(navp_in: pd.DataFrame, cfg_gpx: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get all start points and section start points from navigation log dataframe
    :param navp_in: navigation log dataframe
    :param cfg_gpx: if waypoits are used for split waypoints to sections then dict that must have fields:
      - symbol_excude_point:
      - symbol_break_keeping_point:
    :return: ranges, b_navp_exclude, ddist or sec_names
      - b_navp_exclude: numpy 2D array and bool mask of excluded points: (A, b_navp_exclude)
        Array A allows to get starts/ends for section with index i: A[0/1, i]
    """
    b_navp_exclude = navp_in.sym.values == cfg_gpx.get('symbol_excude_point')
    df = navp_in[~b_navp_exclude]
    # [0,: -1]
    ddist = distance(df.Lon.values, df.Lat.values) / 1e3  # km
    df_timeind, itm = multiindex_timeindex(df.index)

    if itm is None:  # isinstance(df_timeind, pd.DatetimeIndex):
        l.warning('Special waypoints are used for split them to sections. Better use routes!')
        b_break_condition = df.sym == cfg_gpx.get('symbol_break_keeping_point')
        # Remove boundary points from sections where distance to it is greater cfg_gpx['symbol_break_notkeeping_dist']
        isec_break = np.append(np.flatnonzero(b_break_condition), df.shape[0])
        i0between_sec = isec_break[1:-1]  # if isec_break[-1]+1 == df.shape[0] else isec_break[1:
        # st_with_prev = np.append(False, abs(ddist[i0between_sec]) < cfg_gpx['symbol_break_notkeeping_dist'])
        en_with_next = np.append(abs(ddist[i0between_sec - 1]) < cfg_gpx['symbol_break_notkeeping_dist'], True)
        ranges = np.vstack((isec_break[:-1], isec_break[1:] - 1 + np.int8(en_with_next)))
        # - st_with_prev
        # remove empty sections:
        ranges = ranges[:, np.diff(ranges, axis=0).flat > 1]
        return ranges, b_navp_exclude, ddist
    else:
        sec_names = df.index.levels[df.index.nlevels - 1 - itm]
        sect_st_time = [(i, df.loc[sec_name].index[0]) for i, sec_name in enumerate(sec_names)]
        sect_st_time.sort(key=lambda k: k[1])
        isort = [i for i, t in sect_st_time]

        n_points = [df.loc[sec_name].shape[0] for sec_name in sec_names]
        isec_break = np.cumsum(np.int32(n_points))
        isec_break[-1] += 1
        ranges = np.vstack((np.append(0, isec_break[:-1]), isec_break - 1))[:, isort]
        sec_names = sec_names[isort]
        # same effect:
        # ranges = np.transpose([df.index.get_indexer([(sec_name, df.loc[sec_name].index[0]), (sec_name, df.loc[sec_name].index[-1])]) for sec_name in sec_names])
        return ranges, b_navp_exclude, sec_names



def ge_sections(navp_all: pd.DataFrame,
                cfg: MutableMapping[str, Any],
                isec_min=0,
                isec_max=np.inf) -> Iterable[Tuple[pd.DataFrame, Dict[str, Any]]]:
    """

    :param navp_all: navigation log dataframe
    :param cfg:
    - route_time_level: will be assigned to time axis index of navp_all MultiIndex axes or None if no MultiIndex
    - gpx:
    - process: dict with field 'invert_prior_sn_angle'
    - out:  dict with field 'dt_from_utc'
    :param isec_min:
    :param isec_max:
    :return: navp, navp_d - dataframe and dict of some data associated with:
    - sec_#
    - sec_name
    - exclude
    - isort
    - indexs
    - time_poss_max
    - b_invert
    - time_msg_min
    - time_msg_max
    - stem_time_st
    - msg

    """

    ranges, b_navp_all_exclude, sec_names = sec_edges(navp_all, cfg['gpx'])
    navp_all_index, cfg['route_time_level'] = multiindex_timeindex(navp_all.index)
    navp_all_indexs = navp_all_index.sort_values()
    for isec, (st, en) in enumerate(ranges.T, start=1):
        if isec < isec_min:
            continue
        if isec > isec_max:
            break
        navp_d = {'sec_#': isec}

        if cfg['route_time_level'] is None:
            navp = navp_all[st:(en + 1)]
        else:
            navp_d['sec_name'] = sec_names[isec - 1]
            assert navp_d['sec_name'] == navp_all.index[st][navp_all.index.nlevels - 1 - cfg['route_time_level']]
            navp = navp_all.loc[navp_d['sec_name']]

        navp_d['exclude'] = navp[b_navp_all_exclude[st:(en + 1)]]
        navp = navp[~b_navp_all_exclude[st:(en + 1)]]
        navp_index = navp.index
        navp_d['isort'] = navp_index.argsort()
        navp_d['indexs'] = navp_index[navp_d['isort']]
        try:
            navp_d['time_poss_max'] = navp_all_indexs[navp_all_indexs.searchsorted(
                navp_d['indexs'][-1] + pd.Timedelta(seconds=1))]
        except IndexError:  # handle last array lament
            navp_d['time_poss_max'] = navp_all_indexs[-1] + pd.Timedelta(hours=1)

        # Autoinvert flag (if use route guess order from time of edges)
        # course = geog.course(*navp.loc[navp_d['indexs'][[0, -1]], ['Lon', 'Lat']].values, bearing=True)
        #                    became: (navp['Lon'][0], navp['Lat'][0]), (navp['Lon'][en], navp['Lat'][en])
        geod = pyproj.Geod(ellps='WGS84')
        course, azimuth2, distance = geod.inv(
            *navp.loc[navp_d['indexs'][[0, -1]], ['Lon', 'Lat']].values.flat)  # lon0, lat0, lon1, lat1

        navp_d['b_invert'], msg_inv_reason = track_b_invert(course, cfg['process']['invert_prior_sn_angle'])
        if cfg['route_time_level'] is None:
            msg_invert = ''
        else:
            # retain msg about calculated direction, but use invertion from loaded section
            navp_d['b_invert'] = np.diff(navp_index[[0, -1]])[0].to_numpy() < np.timedelta64(0)
            msg_invert = 'Veusz section will {}need to be inverted to approximate this route '.format(
                '' if navp_d['b_invert'] else 'not ')

        navp_d['time_msg_min'], navp_d['time_msg_max'] = [timzone_view(x, cfg['out'][
            'dt_from_utc']) for x in navp_d['indexs'][[0, -1]]]
        navp_d['stem_time_st'] = '{:%y%m%d_%H%M}'.format(navp_d['time_msg_min'])
        navp_d['msg'] = '\n{} "{}". Section of {}runs {} ({:.0f}\xb0=>{}). '.format(
            isec, navp_d.get('sec_name', ''), len(navp),
            '{time_msg_min:%m.%d %H:%M} - {time_msg_max:%d %H:%M}'.format_map(navp_d),
            course, msg_inv_reason) + msg_invert

        yield navp, navp_d


def load_cur_veusz_section(cfg: Mapping[str, Any],
                           navp_d: Mapping[str, Any],
                           vsze=None) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]], Any]:
    """
    Loads processed CTD data using Veusz:
    - searches existed Veusz file named by start datetime of current section and must contain "Inv" only if
    navp_d['b_invert']
    - creates and saves its copy with modified setting ``USE_timeRange`` in Custom dafinitions for new sections
    - opens and gets data from it
    return: Tuple:
        ctd: pd.DataFrame, loaded from Veusz CTD data,
        ctd_prm: dict of parameters of other length than ctd,
        - starts: runs starts
        - ends: runs ends
        vsze: Veusz embedded object
    """

    global load_vsz
    if load_vsz is None:
        load_vsz = load_vsz_closure(cfg['program']['veusz_path'])
    def good_name(pathname):
        return pathname.startswith(navp_d['stem_time_st']) and navp_d['b_invert'] ^ ('Inv' not in pathname)
    vsz_names = [Path(v).name for v in cfg['vsz_files']['paths'] if good_name(Path(v).name)]
    if vsz_names:  # Load data from Veusz vsz
        l.warning('%s\nOpening matched %s as source...', navp_d['msg'], vsz_names[0])
        vsz_path = cfg['vsz_files']['path'].with_name(vsz_names[0])
        vsze, ctd_dict = load_vsz(vsz_path, vsze, prefix='CTD')
        # vsz_path = vsz_path.with_suffix('')  # for comparbility with result of 'else' part below
        b_new_vsz = False
    else:  # Modify Veusz pattern and Load data from it, save it
        l.warning(navp_d['msg'])  # , end=' '
        if vsze:
            print('Use currently opened pattern...', end=' ')
        else:
            print('Opening last file {} as pattern...'.format(cfg['vsz_files']['paths'][-1]), end=' ')
            vsze, ctd_dict = load_vsz(cfg['vsz_files']['paths'][-1], prefix='CTD')
            if 'time' not in ctd_dict:
                l.error('vsz data not processed!')
                return None, None, None
        print('Load our section...', end='')
        vsze.AddCustom('constant', u'USE_runRange', u'[[0, -1]]', mode='replace')  # u
        vsze.AddCustom('constant', u'USE_timeRange', '[[{0}, {0}]]'.format(
            "'{:%Y-%m-%dT%H:%M:%S}'").format(navp_d['time_msg_min'], timzone_view(  # iso
            navp_d['time_poss_max'], cfg['out']['dt_from_utc'])), mode='replace')
        vsze.AddCustom('constant', u'Shifting_multiplier', u'-1' if navp_d['b_invert'] else u'1', mode='replace')

        # If pattern has suffix (excluding 'Inv') then add it to our name (why? - adds 'Z')
        stem_no_inv = Path(cfg['vsz_files']['paths'][0]).stem
        if stem_no_inv.endswith('Inv'): stem_no_inv = stem_no_inv[:-3]
        len_stem_time_st = len(navp_d['stem_time_st'])
        vsz_path = cfg['vsz_files']['path'].with_name(''.join([
            navp_d['stem_time_st'],
            stem_no_inv[len_stem_time_st:] if len_stem_time_st < len(stem_no_inv) else 'Z',
            'Inv' if navp_d['b_invert'] else '']))
        vsze.Save(str(vsz_path.with_suffix('.vsz')))
        b_new_vsz = True
        vsze, ctd_dict = load_vsz(veusze=vsze, prefix='CTD')

    if 'time' not in ctd_dict:
        l.error('vsz data is bad!')
        return None, None, None

    # Group all data columns in datframe
    ctd_prm = {k: v for k, v in ctd_dict.items() if v.size != ctd_dict['time'].size}
    for k in ctd_prm.keys():
        del ctd_dict[k]
    ctd_prm['stem'] = vsz_path.stem
    ctd = pd.DataFrame.from_dict(ctd_dict)

    if b_new_vsz or cfg['out']['b_reexport_images'] != False:
        export_images(vsze, cfg['vsz_files'], vsz_path.stem,
                      b_skip_if_exists=cfg['out']['b_reexport_images'] is None)

    l.info('- %s CTD runs loaded', ctd_prm['starts'].size)
    return ctd, ctd_prm, vsze


def idata_from_tpoints(tst_approx, tdata, idata_st, dt_point2run_max=None):
    """
    Select data runs which start time is in (tst_approx, tst_approx+dt_point2run_max)
    :param tst_approx: approx run start times
    :param tdata: numpy.datetime64 array of data's time values
    :param idata_st: data indexes of runs starts, must be sorted, may be list or numpy array
    :param dt_point2run_max: pd.Timedelta, interval to select. If None then select to the next point
    Now don't undestand why search only after tst_approx if dt_point2run_max is defined. If error in tst_approx is positive we will take next point instead best closest. So:
    todo: delete dt_point2run_max parameter and logic

    :return: sel_run - list of selected run indices (0 means run started at tdata[idata_st[0]])
            sel_data - mask of selected data
    :note: If exist some tst_approx < tdata[idata_st[0]] then it couses of appiar 0 in sel_run
    """
    sel_run = np.int32([])
    sel_data = np.zeros_like(tdata, bool)
    if not len(tst_approx):
        return sel_run, sel_data

    sel_run = []
    tdata_st = tdata[idata_st]
    idata_st = np.array(idata_st)

    # tst_approx_max_err= np.min(np.diff(tdata_st))/2 #found this delta
    # if tst_approx_max_err < np.timedelta64(1, 's'):
    #     print('Small time difference between starts (mimum is {}) detected!'.format(tst_approx_max_err))
    #     tst_approx_max_err = max(tst_approx_max_err, np.timedelta64(1, 'ms'))
    #
    # if not dt_point2run_max:
    #     iruns_found = np.atleast_1d(np.searchsorted(tdata_st, tst_approx - np.timedelta64(tst_approx_max_err)))
    if not dt_point2run_max:
        # because of not supported time span of numpy '<M8[ns]' for (real years)*2 use sufficient '<M8[s]'
        iruns_found = datetime_fun(inearestsorted, tdata_st, tst_approx, type_of_operation='<M8[s]',
                                   type_of_result='i8')
        if not len(iruns_found):
            return sel_run, sel_data
        iruns_found[iruns_found >= len(idata_st)] = iruns_found[0]  # to remove by unique:
        iruns_found = np.unique(iruns_found)
        if not np.any(iruns_found < len(idata_st)):
            return sel_run, sel_data
        # Find intervals
        idata_to_found = idata_st[iruns_found]

        # intervals ends
        iruns_next_to_found = iruns_found + 1
        try:
            idata_next_to_found = idata_st[iruns_next_to_found]
            tdata_next_to_found = tdata[idata_next_to_found]
        except IndexError:  # handle last array lament
            idata_next_to_found = np.empty_like(iruns_found, 'int')
            tdata_next_to_found = np.empty_like(iruns_found, 'datetime64[ns]')
            idata_next_to_found[:-1] = idata_st[iruns_next_to_found[:-1]]
            tdata_next_to_found[:-1] = tdata[idata_next_to_found[:-1]]
            # replase last time of interval with value just after next interval
            tdata_next_to_found[-1] = tdata[idata_st[iruns_next_to_found[-1]]] if \
                iruns_next_to_found[-1] < len(idata_st) else \
                (np.diff(tdata[-2:])[0] + tdata[-1])  # np.diff(tdata[-2:]) instead tst_approx_max_err
        # mark data in intervals
        for i_st, i_en in zip(idata_to_found, idata_next_to_found):
            sel_data[i_st:i_en] = True
            # sel_data |= np.logical_and(t_st <= tdata, tdata <t_en)  #tdata[]

        return iruns_found, sel_data

    # If dt_point2run_max is defined:
    for t_st, t_en in zip(tst_approx, tst_approx + np.timedelta64(dt_point2run_max)):
        bBetween = np.logical_and(t_st <= tdata_st, tdata_st < t_en)

        if np.any(bBetween):
            irun = np.flatnonzero(bBetween)
            if len(irun) > 1:
                print('{} runs found after {} in {}!'.format(len(irun), t_st, dt_point2run_max))
                sel_run.append(irun)

            sel_data[slice(idata_st[irun[0]],
                           idata_st[irun[-1] + 1])] = True

    return sel_run, sel_data


def runs_ilast_good(bctd_ok: Sequence[bool], starts: Sequence[int], ends: Sequence[int], ctd_depth: Optional[Sequence],
                    max_altitude: Optional[float] = None, bottom_at_ends: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds last not NaN value in each run
    :param bctd_ok: mask where profile values are good
    :param starts:  start indexes of each profile (of runs)
    :param ends:    end indexes of each profiles

    Filtering too short runs parameters:
    :param ctd_depth: CTD depth data, same size as :param bctd_ok. Set NaN or small if not need the filtering
    :param max_altitude: max good distance to bottom, set NaN or big to not filter
    :param bottom_at_ends: set NaN or big (>0) to not filter
    :return ends_cor, ok_runs:
        - ends_cor: index of last value have found in each run skipping ones having bad range_to_bot
        - ok_runs: bool mask size of starts: 1 for runs having good range_to_bot, else 0.
    """
    ok_runs = np.ones_like(starts, np.bool)
    ends_cor = np.empty_like(starts)
    # for each profile:
    for k, (st, en) in enumerate(zip(starts, ends)):
        try:
            ends_cor[k] = np.flatnonzero(bctd_ok[st:en])[-1] + st
        except IndexError:                      # no good values in profile
            ok_runs[k] = False
            continue
    # filter runs that ends too far from bottom
    if any(arg is None for arg in (ctd_depth, bottom_at_ends, max_altitude)):
        d = bottom_at_ends[ok_runs] - ctd_depth[ends_cor[ok_runs]]
        ok_runs[ok_runs] = d <= max_altitude

    return ends_cor[ok_runs], ok_runs


def b_add_ctd_depth(dist, depth, add_dist, add_depth, max_dist=0.5, max_bed_gradient=50, max_bed_add=5):
    """
    Mask useful elements from CTD profile to add them to echosounder depth profile
    where it has no data near.
    Modifies (!) depth: sets depth smaller than add_depth near same dist to NaN.
    :param add_dist: dist of profile (CTD) to append
    :param add_depth: depth of profile (CTD) to append
    :param dist:
    :param depth:
    :param max_dist: 0.5km - Check existance of bottom depth data in this range near data points
    :param max_bed_gradient: 50m/km - Will check that bottom is below CTD assuming max depth gradient
    :param max_bed_add: 5m uncertainty
    :return: mask of edge_path elements to use
    """
    # Set bottom depth by bottom edge of CTD path (keep lowest) where no bottom profile data near the run's bot edge
    # todo: also allow filter depth far from CTD bot using this method
    b_add = np.ones_like(add_dist, np.bool8)
    b_any_r = True
    for i, (dSt, dEn) in enumerate(add_dist[:, np.newaxis] + np.array([[-1, 1]]) * max_dist):
        b_near_l = np.logical_and(dSt < dist, dist <= add_dist[i])
        b_near_r = np.logical_and(add_dist[i] < dist, dist <= dEn)

        if not np.any(b_near_l | b_near_r):
            # no depth data near measuring point
            b_any_r = False
        else:  # Check that depth below CTD
            # If bottom below inverted triangle zone, then use it
            # dSt   dEn
            # |     |
            # \    /
            #  \  /
            # --\\/_add_depth
            #    \____/\__- depth
            #

            b_near_l = depth[b_near_l] < add_depth[i] + max_bed_add + max_bed_gradient * np.abs(
                add_dist[i] - dist[b_near_l])
            b_any_l = any(b_near_l)
            if not (b_any_r | b_any_l):
                # remove Dep data (think it bad) in all preivious interval:
                b_near_l = np.logical_and(add_dist[i - 1] < dist, dist <= add_dist[i])
                depth[b_near_l] = np.NaN
                b_add[int(i - 1)] = True  # add data from CTD if was not added

            b_near_r = depth[b_near_r] < add_depth[i] + max_bed_add + max_bed_gradient * np.abs(
                add_dist[i] - dist[b_near_r])
            b_any_r = any(b_near_r)
            if (b_any_r | b_any_l):
                b_add[i] = False  # good depth exist => not need add CTD depth data
    return b_add  # , depth


def data_sort_to_nav(navp: pd.DataFrame,
                     navp_exclude: pd.Series,
                     b_invert: np.ndarray,
                     ctd: pd.DataFrame,
                     ctd_prm: Dict[str, Any],
                     cfg: Mapping[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any], np.ndarray]:
    """
    Finds correspondence of CTD runs to nav points using its times and device markers.
    Excludes CTD runs and data where corresponding navigation points marked by `symbol_exclude_point`
    :param navp: pandas DataFrame. Navigation points table (with datetimeindex and obligatory columns Lat, Lon, name,
    sym). Rows order must be in desired direction.
    :param navp_exclude: navigation points to exclude
    :param b_invert:
    :param ctd:
    :param ctd_prm: dict with fields:
        'starts', 'ends' - edge indexes of good run downs,
        'itable' - optional, need if data and nav points is by different CTDs
    :param cfg: fields:
        route_time_level: not none if points aranged right
    :return: tuple:
        - ctd - sorted in order of nav having: index - integer range, column 'time' - original index
        - ctd_prm - dict with replaced fields: 'starts', 'ends'
        - navp_ictd - order in accordance to nav


    navp            0   1   2   3   4  ...
    navp_isort      3   4   2   1   0  ... - time order
    navp_isym       1   0   0   0   2  ...
    ctd_inavp_i
    """

    # ctd_prm['starts'] = np.append(0, ctd_prm['starts'])
    # ctd_prm['ends'] = np.append(ctd_prm['starts'], ctd['time'].size))

    # Find CTD runs where time is closest to time of nav points

    navp_index = navp.index.values
    ctd_isort = ctd.time.iloc[ctd_prm['starts']].values.argsort()
    ctd_sts = ctd.time.iloc[ctd_prm['starts'][ctd_isort]].values
    # closest CTD indexes to each nav point:
    navp_ictd = datetime_fun(inearestsorted, ctd_sts, navp_index, type_of_operation='<M8[ms]', type_of_result='i8')

    # Check/correct one to one correspondence
    navp_ictd_isort = navp_ictd.argsort()
    navp_ictd_isort_diff = np.diff(navp_ictd[navp_ictd_isort])
    inav_jumped = None
    to_del_navs = []
    for inav, ictd in zip(
            (navp_ictd_ibads := np.flatnonzero(navp_ictd_isort_diff != 1)),
            navp_ictd[navp_ictd_isort[navp_ictd_ibads]]
            ):
        #  Can correct if have pairs of nav points assigned to one CTD
        #  n1   n2  n3    n4    - nav time not sorted
        #  |     \ /       \    - navp_ictd
        #  C1    C2   C3   C4   - CTD time sorted

        if navp_ictd_isort_diff[inav] == 0:
            # Same CTD assigned to this nav point and next
            # - because current or next nav point is bad and skipped/no previous or next CTD
            try:
                if inav_jumped is None:  # have no not assigned nav point(s)
                    # check that next CTD is skipped (diff > 1)
                    inav_jumped = inav + 1
                    if navp_ictd_isort_diff[inav_jumped] > 1:
                        # the guess just made was correct
                        ictd_skipped = ictd + 1
                    else:
                        inav_jumped = None  # not correct
                        l.warning('Not implemented condition 1! Reassign nav to CTD manually')
                        if ctd_isort < navp_ictd.size:  # if number of CTD < number of nav
                            # insert 1 dummy ctd point but mark this nav for deletion (experimental)
                            to_del_navs.append(inav)
                            navp_ictd.insert(inav, inav)  # increase length to decrease chance of this block repetition
                        raise NotImplementedError()
                elif inav == inav_jumped:
                    inav += 1
                else:
                    l.warning('Not implemented condition 2! Reassign nav to CTD manually')
                    raise NotImplementedError()
            except NotImplementedError:
                l.warning('Not implemented condition')
            else:
                # variants: (C2n2, C3n3), (C3n2, C2n3) - here n2,n3 - current&next nav, C2,C3 - current&skipped CTD time
                # reassign skipped CTD to nav by minimise sum(C - n). So check 2 assignment variants:
                # 1. direct:
                sum1 = abs(ctd_sts[[ictd, ictd_skipped]] - navp_index[navp_ictd_isort[[inav, inav_jumped]]]).sum()
                # 2. reverse:
                sum2 = abs(ctd_sts[[ictd, ictd_skipped]] - navp_index[navp_ictd_isort[[inav_jumped, inav]]]).sum()
                if sum1 < sum2:
                    navp_ictd[navp_ictd_isort[[inav, inav_jumped]]] = [ictd, ictd_skipped]
                else:
                    navp_ictd[navp_ictd_isort[[inav_jumped, inav]]] = [ictd, ictd_skipped]
        else:  # next CTD point was skipped (diff > 1)
            inav_jumped = inav + 1
            ictd_skipped = ictd + 1
            continue

    # ctd to points order indexer:
    navp_ictd = ctd_isort[navp_ictd]  # convert index of ctd sorted to ctd

    b_far = check_time_diff(
        ctd.time.iloc[ctd_prm['starts'][navp_ictd]], navp_index,
        dt_warn=pd.Timedelta(cfg['process']['dt_search_nav_tolerance']),
        mesage='CTD runs which start time is far from nearest time of closest navigation point # [min]:\n'
        )
    # Checking that indexer is correct.
    if len(cfg['gpx']['symbols_in_veusz_ctd_order']) and 'itable' in ctd_prm:
        # many CTD data in same time. Right ctd will be assigned
        # navp.sym to index
        navp_isym = -np.ones(navp.shape[0], np.int32)
        for i, sym in enumerate(cfg['gpx']['symbols_in_veusz_ctd_order']):
            navp_isym[navp.sym == sym] = i
        b_wrong_ctd = ctd_prm['itable'][navp_ictd] != navp_isym
        if any(b_wrong_ctd):
            if len(navp_isym) - len(ctd_prm['itable']) == sum(b_wrong_ctd):
                l.warning('it looks like Veusz data have not all navigation points. Excluding')
                # navp = navp[~b_wrong_ctd]
                navp_ictd = navp_ictd[~b_wrong_ctd]
                navp_index = navp_index[~b_wrong_ctd]
                b_far = b_far[~b_wrong_ctd]
                b_wrong_ctd = [False]
            b_far |= b_wrong_ctd
    else:
        b_wrong_ctd = [False]

    if np.any(b_far):
        # Try to find indexer error and suggest correction
        while any(b_wrong_ctd):
            try:  # CTD table order correction
                for p in np.flatnonzero(b_far):
                    if b_wrong_ctd[p]:  # suggest nearest good ctd run
                        b_ctd_goods = ctd_prm['itable'][ctd_isort] == navp_isym[p]
                        igood = datetime_fun(inearestsorted, ctd_sts[b_ctd_goods], navp_index[p],
                                             type_of_operation='<M8[s]', type_of_result='i8')
                        igood = (ctd_isort[b_ctd_goods])[igood]
                        if igood in navp_ictd:  # need remove previous assignment
                            iprev = np.flatnonzero(navp_ictd == igood)
                            if not b_far[iprev]:
                                print('dbstop here: replacing previous assignment which was good!')
                            b_wrong_ctd[iprev] = True  # becouse used in more than 1 point
                        navp_ictd[p] = igood
                        b_wrong_ctd[p] = False

                        # if b_far[igood]:
            except ValueError as e:
                l.error('can not correct CTD table order. May be need change gpx configuration.'
                        ' Current gpx.symbols_in_veusz_ctd_order_list = {}',
                        cfg['gpx']['symbols_in_veusz_ctd_order_list'])
                raise e  # ValueError
        b_far = check_time_diff(ctd.time.iloc[ctd_prm['starts'][navp_ictd]].values, navp_index, pd.Timedelta(minutes=1),
                                mesage='CTD runs (after correction) far from nearest time of closest navigation point # [min]:\n')
        ctd_not_in_navp = np.setdiff1d(np.arange(ctd_prm['starts'].size), navp_ictd)
        if ctd_not_in_navp:
            msg = 'Excluding runs # (based on sym index)... '
            ctd_far_time = ctd.time.values[ctd_prm['starts']][ctd_not_in_navp]
            l.info(msg + '\n'.join(['{}:{} "{}"'.format(i, t, s) for i, t, s in
                                    zip(ctd_not_in_navp, ctd_far_time,
                                        np.array(cfg['gpx']['symbols_in_veusz_ctd_order'])[
                                            np.int32(ctd_prm['itable'][ctd_not_in_navp])])]))
        temp = False  # old code block
        if temp:
            navp_isort = navp_index.argsort()
            navp_indexs = navp_index[navp_isort]  # sorted by time
            navp_isort_back = np.empty_like(navp_isort, dtype=np.int32)
            navp_isort_back[navp_isort] = np.arange(navp_isort.size, dtype=np.int32)
            # closest(sorted time) points indexes to CTD time starts
            ctd_inavp = datetime_fun(inearestsorted, navp_indexs, ctd.time.values[ctd_prm['starts']],
                                     type_of_operation='<M8[s]', type_of_result='i8')
            ctd_inavp = navp_isort[ctd_inavp]  # convert index of points sorted to point

        # Find CTD runs for which ctd_prm['itable'] not in navp.sym index
        if temp and 'itable' in ctd_prm:
            b_ctd_exclude = navp_isym[ctd_inavp] != ctd_prm['itable']
            if np.any(b_ctd_exclude):
                # Incorrect indexer
                # find navpoints near to ctd_prm runs having same point

                inavp_with_many_runs = np.flatnonzero(np.bincount(ctd_inavp) > 1)
                b_ctd_same_point = np.zeros_like(ctd_inavp, dtype=bool)
                for p in inavp_with_many_runs:
                    b_ctd_same_point[:] = np.logical_and(ctd_inavp == p, b_ctd_exclude)
                    b_far[np.int32(ctd_prm['itable'])[b_ctd_same_point] != navp_isym[p]] = True
                    # wrong run detected exactly
                if ctd_inavp.size - sum(b_far) == navp_index.size:  # autodetected ok
                    b_ctd_exclude = b_far
                    msg = 'Excluding runs (based on sym index):\n'
                else:
                    b_ctd_exclude |= b_far
                    msg = 'Need to exclude some of this runs (based on sym index):\n'
                ctd_far_time = ctd.time.values[ctd_prm['starts']][b_ctd_exclude]
                l.info(msg + '\n'.join(['{}:{} "{}"'.format(i, t, s) for i, t, s in zip(
                    np.flatnonzero(b_ctd_exclude), ctd_far_time,
                    np.array(cfg['gpx']['symbols_in_veusz_ctd_order'])[
                        np.int32(ctd_prm['itable'][b_ctd_exclude])])]))

                if ctd_inavp.size - sum(b_far) != navp_index.size:  # need manual corr
                    if np.any(b_ctd_same_point & b_ctd_exclude):  # check: b_ctd_same_point overwrites in a loop!
                        if ctd_inavp.size - sum(b_ctd_exclude) == navp_index.size:
                            l.info(
                                'Excluding {} runs not found in section points (far CTD runs or from other table)'.format(
                                    sum(b_ctd_exclude)))
                        else:
                            print("dbstop here: set manualy what to exclude")
                            # b_ctd_exclude[0]= False  # don't want to delete run index 0
                    ctd_far_time = ctd.time.values[ctd_prm['starts']][b_ctd_exclude]
    else:
        ctd_not_in_navp = []

    # CTD runs and data which excluded in navigation points by symbol_excude_point
    # ctd_exclude_time = np.append(navp_exclude.index.values.astype('datetime64[ns]'), ctd_far_time)
    ctd_idel, bDel = idata_from_tpoints(
        tst_approx=navp_exclude.index.values.astype('datetime64[ns]'),
        tdata=ctd.time.values, idata_st=ctd_prm['starts'],
        dt_point2run_max=cfg['process']['dt_point2run_max'])

    # CTD runs and data which was not found in points:
    if ctd_not_in_navp:
        ctd_st_ext = np.append(ctd_prm['starts'], ctd.time.size)
        for i in ctd_not_in_navp:
            bDel[slice(*ctd_st_ext[[i, i + 1]])] = True
        ctd_idel = np.union1d(ctd_idel, ctd_not_in_navp)

    # Remove not needed CTD data
    # remove runs
    if len(ctd_idel):
        l.info('deleting %d runs # which not in section: %s', len(ctd_idel), ctd_idel)
        ctd_prm['starts'] = np.delete(ctd_prm['starts'], ctd_idel)
        ctd_prm['ends'] = np.delete(ctd_prm['ends'], ctd_idel)
        ctd_bdel = np.zeros_like(ctd_prm['starts'], bool)
        ctd_bdel[np.int32(ctd_idel)] = True
        navp_ictd = i_move2good(navp_ictd, ctd_bdel)
        if np.flatnonzero(np.bincount(navp_ictd) > 1):  # inavp_with_many_runs =
            print("dbstop here: set manualy what to exclude")
    # remove rows
    bDel |= ctd.Pres.isna().values
    if any(bDel):
        ctd = ctd[~bDel]
        ctd_prm['starts'] = i_move2good(ctd_prm['starts'], bDel)
        ctd_prm['ends'] = i_move2good(ctd_prm['ends'], bDel, 'right')

    # Sort CTD data in order of points.

    # CTD order to points order:
    if cfg['route_time_level'] is None:  # arrange runs according to b_invert
        navp_ictd = np.arange(
            navp_index.size - 1, -1, -1, dtype=np.int32) if b_invert else np.arange(
            navp_index.size, dtype=np.int32)
    # else:  # runs will exactly follow to route
    #     navp_ictd = np.empty_like(navp_isort, dtype=np.int32)
    #     navp_ictd[navp_isort] = np.arange(navp_isort.size, dtype=np.int32)

    if np.any(navp_ictd != np.arange(navp_ictd.size, dtype=np.int32)):
        # arrange run starts and ends in points order
        # 1. from existed intervals (sorted by time like starts from Veusz)
        run_endup_from = np.append(ctd_prm['starts'][1:], ctd.time.size)
        run_edges_from = np.column_stack((ctd_prm['starts'], run_endup_from))  #
        if len(run_edges_from) != navp_ictd.size:
            l.error('Number of loaded nav. points {} != {} ctd points', navp_ictd.size, len(run_edges_from))
        run_edges_from = run_edges_from[navp_ictd, :]  # rearrange ``from`` intervals

        # 2. to indexes
        run_dst = np.diff(run_edges_from).flatten()
        run_next_to = np.cumsum(run_dst)
        ctd_prm['starts'] = np.append(0, run_next_to[:-1])
        run_edges_to = np.column_stack((ctd_prm['starts'], run_next_to))  # ?

        # sort CTD data
        ind_by_points = np.empty(ctd.time.size, dtype=np.int32)
        for se_to, se_from in zip(run_edges_to, run_edges_from):
            ind_by_points[slice(*se_from)] = np.arange(*se_to)
        ctd.index = ind_by_points
        ctd = ctd.sort_index()  # inplace=True???


        # 'ends' will be needed later
        _junks_to = (run_endup_from - ctd_prm['ends'])[navp_ictd]
        ctd_prm['ends'] = run_edges_to[:, 1] - _junks_to

        # temp = np.empty(ctd['time'].size)
        # ctd.reindex
        # for key in ctd.columns():
        #         temp[:] = ctd[key]
        #         for ise, ises in zip(run_edges_to, run_edges_to):
        #             ctd[key][slice(*ise)] = temp[slice(*ises)]

    return ctd, ctd_prm, navp_ictd


"""
    # old
    b_far = check_time_diff(ctd['time'][ctd_prm['starts']], navp_index[ctd_inavp], pd.Timedelta(minutes=1), mesage='CTD runs which start time is far from nearest time of navigation point [min]:')

    # find navpoints near to each other which can cause ambiguity (only if have many runs in same point)
    inavp_with_many_runs = np.flatnonzero(np.bincount(ctd_inavp) > 1)
    for t in navp_index[inavp_with_many_runs]:
        b_navp_near = abs(t - navp_index) <= pd.Timedelta(minutes=10)
        if sum(b_navp_near) > 1:
            print(
                "dbstop here: CTD runs to navigation points ambiguity! Correct CTD runs (Veusz data) indexes of corresponded points")


    ctd_far_time = np.array([], dtype='datetime64[ns]')
    if ctd_inavp.size > navp_index.size or :
        if ctd_inavp.size - sum(b_far) == navp_index.size:
            print('Excluding {} far CTD runs'.format(sum(b_far)))
        else:
            print('Loaded number of CTD runs is bigger on {} than points. Found {} far CTD runs ({}).'.format(
                ctd_inavp.size - navp_index.size, sum(b_far), np.flatnonzero(b_far)))
        ctd_far_time = ctd['time'][ctd_prm['starts']][b_far]  # set b_far[ind] = True

        # Find CTD runs for which ctd_prm['itable'] not in navp.sym index
        if 'itable' in CTD:
            # navp.sym to index
            navp_isym = -np.ones(navp.shape[0], np.int32)
            for i, sym in enumerate(cfg['gpx']['symbols_in_veusz_ctd_order']):
                navp_isym[navp.sym == sym] = i

            # Closest symbol from navigation should be equal to number of table:
            # but if >1 runs in one time ctd_inavp will shows same index
            # chek it manually!
            b_ctd_same_point = np.ediff1d(ctd_inavp, to_end=0) == 0
            b_ctd_same_point[-1] = b_ctd_same_point[-2]
"""


def filt_depth(s_ndepth: pd.Series, **cfg_proc: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray, matplotlib.axes.Axes]:
    """
    Filter depth
    :param s_ndepth: pandas time series of source depth
    :param cfg_proc: dict with fields (float):
        - min_depth
        - max_depth
        - depecho_add
        - filter_depth_wavelet_level
    :return: (depth_out, bGood, ax) - data reduced in size depth, corresponding mask of original s_ndepth, axis
    """
    # Filter DepEcho
    print('Bottom profile filtering ', end='')
    bed = np.abs(s_ndepth.values.flatten()) + cfg_proc['depecho_add']
    ok = is_works(bed)
    ok &= ~np.isnan(bed)
    if cfg_proc['min_depth']:
        ok[ok] &= (bed[ok] > cfg_proc['min_depth'])
    if cfg_proc['max_depth']:
        ok[ok] &= (bed[ok] < cfg_proc['max_depth'])
    ok[ok] &= ~too_frequent_values(bed[ok], max_portion_allowed=5)  # critical parameter!!! try 1

    """
    ok&= bed<74; 109; 105; ok&= bed>15 37 10;
    b_ind = (np.arange(bed.size)>25000)&(np.arange(bed.size)<61088)
    b_ind = (np.arange(bed.size)<43200) &(np.arange(bed.size)<61088)
    ok[b_ind]&= ~too_frequent_values(bed[b_ind], max_portion_allowed = 0.5)
    ok&= np.where(b_ind, bed<70, True)     #  65.1 =106.3  #48145
    ax.plot(np.flatnonzero(ok), bed[ok], color='b', alpha=0.7, label='Works')
    """
    if __debug__:
        ax, lines = make_figure(y_kwrgs=({'data': bed, 'label': 'bed sourse0', 'color': 'r', 'alpha': 0.1},),
                                mask_kwrgs={'data': ok, 'label': 'Works', 'color': 'r', 'alpha': 0.7},
                                ax_title='Bottom profile filtering', ax_invert=True)
    else:
        ax = None
    # Despike
    depth_filt = bed
    depth_out = bed[ok]
    wdesp = min(100, sum(ok))
    if wdesp > 50:
        coef_std_offsets = (15, 7)  # filters too many if set some < 3
        # back and forward:
        depth_out = despike(depth_out[::-1], coef_std_offsets, wdesp)[0]
        depth_out = despike(depth_out[::-1], coef_std_offsets, wdesp)[0]
        ok[ok] &= ~np.isnan(depth_out)
        depth_filt = rep2mean(depth_filt, ok,
                              s_ndepth.index.view(np.int64))  # need to execute waveletSmooth on full length

        ax.plot(np.flatnonzero(ok), bed[ok], color='g', alpha=0.9, label='despike')

        # Smooth some big spikes and noise              # - filter_depth_wavelet_level']=11
        depth_filt, ax = waveletSmooth(depth_filt, 'db4', cfg_proc['filter_depth_wavelet_level'], ax,
                                       label='Depth')

        # Smooth small high frequency noise (do not use if big spikes exist!)
        sGood = ok.sum()  # ok[:] = False  # to use max of data as bed
        # to make depth follow lowest data execute: depth_filt[:] = 0
        n_smooth = 5  # 30
        if sGood > 100 and (np.abs(np.diff(depth_filt)) < 30).all():
            depth_filt_smooth = gaussian_filter1d(depth_filt, n_smooth)
            if ax: ax.plot(depth_filt_smooth, color='y', alpha=0.7, label='smooth')
            # to cansel gaussian filter: dbstop, set depth_filt_smooth = depth_filt
            depth_out = depth_filt_smooth[ok]
            depth_filt.fill(np.NaN)
            depth_filt[ok] = depth_out
    else:
        sGood = True

    if sGood and __debug__:
        interactive_deleter(y_kwrgs=({'data': bed, 'label': 'Depth', 'color': 'k', 'alpha': 0.6},),
                            mask_kwrgs={'data': ok, 'label': 'initial'},
                            ax=ax, ax_title='Nav. bottom profile smoothing(index)', ax_invert=True,
                            lines=lines, stop=cfg_proc['interact'])

        # while True:  # draw
        #     if ax is None or f is None or not plt.fignum_exists(f.number):  # or get_fignums().
        #         ax, lines = make_figure(bed, ok, position=(10, 0))  # only if closed
        #         plot_prepare_input(ax)
        #     else:    # update
        #         pass
        #         # plt.draw()
        #
        #     plt.show()  # ? (block=True - hangs 2nd time) allows select bad regions (pycharm: not stops if dbsops before)
        #     # left -> right : delete
        #     # right -> left : recover
        #     # dbstop
        #     if not np.any(plt_selected_x_range_arr):
        #         break
        #     else:
        #         ok[slice(*np.int64(plt_selected_x_range_arr))] = np.diff(plt_selected_x_range_arr) > 0

        # depth_filt = rep2mean(depth_filt, ok)
        # ind = 87650; depth_filt[ind:] = bed[ind:]
    ok &= np.isfinite(depth_filt)
    depth_out = depth_filt[ok]
    if (sGood and __debug__) and not plt.fignum_exists(ax.figure.number):  # and np.diff(plt_selected_x_range_arr) > 0:
        ax, lines = make_figure(y_kwrgs=({'data': depth_filt, 'label': 'smoothed', 'color': 'r', 'alpha': 0.1},),
                                mask_kwrgs={'data': ok, 'label': 'masked manually', 'color': 'b', 'alpha': 0.7},
                                ax_title='Bottom profile filtering (continue)', ax_invert=True)

        # _, ax = plt.subplots()
        #         # ax.plot(np.flatnonzero(ok), depth_out, color='m', label='Result')
        #         # plt.show()

    return depth_out, ok, ax


# def extract_keeping_row0_edges(a, b_ok):
#     out = a[:, b_ok]
#     out[0, [0, -1]] = a[0, [0, -1]]
#     return out


def extract_repeat_at_bad_edges(a: np.ndarray, b_ok: np.ndarray) -> np.ndarray:
    """
    Returns a[:, b_ok] except where `b_ok`` at edge is False, always keeps edges of 1st row of ``a``.
    :param a: 2d array, 1st row
    :param b_ok: 1d, boolean
    :return: a[:, b_ok] with edge columns replaced by nearest ``a`` column where ``b_ok`` is True,
    and where the 1st row edges are always the edges of ``a``
    """

    if b_ok[[0, -1]].all():
        return a[:, b_ok]

    edges_i_ok_near = np.flatnonzero(b_ok)[[0, -1]]
    edges_b_ok = b_ok.copy()
    edges_b_ok[[0, -1]] = True
    out = a[:, edges_b_ok]
    out[0, [0, -1]] = a[0, [0, -1]]
    out[1:, [0, -1]] = a[1:, edges_i_ok_near]
    return out


def add_data_at_edges(
        ctd_dist, ctd_depth, ctd_z, ctd_prm, edge_depth, edge_dist,
        ok_ctd, ok_ends: np.ndarray, cfg, x_limits
        ) -> pd.DataFrame:  # ctd_add_bt, ctd_add_lr,
    """
    Adding repeated/interpolated data at edges to help gridding

    :param ctd_dist: CTD data
    :param ctd_depth: CTD data
    :param ctd_z: CTD data
    :param ctd_prm: dict with fields: "starts", "ends" - CTD run's top and end edges
    :param edge_depth:
    :param edge_dist:
    :param ok_ctd:
    :param ok_ends: mask runs from which take no bottom edge values
    :param cfg: fields: 'x_resolution_use', 'y_resolution_use'
    :param x_limits:
    :return: ctd_with_adds (nCTD + nBot*2 + nL + nR) x 3
    """

    # Synthetic data that helps gridding
    # ----------------------------------
    # Additional data along bottom edge

    # Additional data to increase grid length at left/right edges upon resolution/2

    ## Add data to left and right from existed data
    # todo: if bad length profile at edge then use its data with interpolated nearest good length profile data

    #  - good length profiles at edges:
    edges_ok = [0, -1] if ok_ends[[0, -1]].all() else np.flatnonzero(ok_ends)[[0, -1]]
    #  - slices of 1st & last run
    # - index slices of 1st & last run of :param ctd: from which data will be copied to ctd_with_adds
    edges_sl_in = [slice(ctd_prm['starts'][e], ctd_prm['ends'][e]) for e in edges_ok]
    # - index slices of :return: ctd_with_adds where data from ctd will be copied
    edges_sl_out = (lambda d: [slice(0, d[0]), slice(*np.cumsum(d))])(
        [edge.stop - edge.start for edge in edges_sl_in])

    ctd_add_lr = np.empty((edges_sl_out[1].stop, 3), np.float64)
    #  - copy y and update x from ctd edge profiles data here, for z see below
    for lim_val, sl_out in zip(x_limits, edges_sl_out):
        ctd_add_lr[sl_out, 0] = lim_val
    for sl_in, sl_out in zip(edges_sl_in, edges_sl_out):
        ctd_add_lr[sl_out, 1] = ctd_depth[sl_in]
        ctd_add_lr[sl_out, 2] = np.NaN
        # pd.DataFrame({
        # 'x': np.empty(edges_sl_out[1].stop, np.float64),
        # 'y': np.empty(edges_sl_out[1].stop, np.float64),
        # 'z': np.empty(edges_sl_out[1].stop, np.float64)
        # })

    # Interpolate data along bottom edge and add them as source to griddata():
    # todo: also add values to top edge
    # - find last not NaN value in each run
    ctd_ok = np.isfinite(ctd_z) & ok_ctd
    ctd_ends_f, b_run_to_edge = runs_ilast_good(
        bctd_ok=ctd_ok, starts=ctd_prm['starts'][ok_ends], ends=ctd_prm['ends'][ok_ends],
        ctd_depth=ctd_depth, max_altitude=cfg['y_resolution_use'] * 5, bottom_at_ends=edge_depth[ok_ends]
        )
    ctd_add_bt_x = np.arange(*x_limits, cfg['x_resolution_use'] / 10)
    ctd_add_bt = np.column_stack([
        # I think minimum addition would be 1 point for per cell to avoid strange grid calculations.
        # Increasing number of points here in 10 times (if more data added then it will have more priority):
        ctd_add_bt_x,
        np.interp(ctd_add_bt_x, ctd_dist[ctd_ends_f], ctd_depth[ctd_ends_f]),  # 'y'
        np.interp(ctd_add_bt_x, ctd_dist[ctd_ends_f], ctd_z[ctd_ends_f])  # default values, will be replaced gradually
        # np.empty(len(add_bot_x)) + np.NaN # 'z': updating this is a main function task solved below
        ])
    # intervals indexes (0 - before the second, last - after the last but one)
    i_add = ctd_dist[ctd_ends_f[1:-1]].searchsorted(ctd_add_bt_x)

    # Interpolate z along nearest vertical profile (by y) separately between profiles that reach the edge

    # Old code:
    # ctd_add_bt.z= np.interp(ctd_add_bt.x, ctd.dist.iloc[ctd_ends_f].values,
    #                  ctd.iloc[ctd_ends_f, icol_z].values)    # , ctd.columns.get_indexer(ax2col[:])].values.T
    # It works, but not account for nonlinear z below y_s. Instead:

    def lin_squeeze(x_old: np.ndarray, x0keep: float, x1old: float, x1new: float) -> np.ndarray:
        """
        Linear squeeze x by moving x0 while keeping the x1 the same
        :param x_old: array to be scaled
        :param x0keep: value of array that must remain the same after scaling
        :param x1old: x point that need to shift to ``x1new``
        :param x1new: required value of scaled array
        :return: scaled array
        """
        given_old_range = x0keep - x1old
        if given_old_range:
            k = (x0keep - x1new) / given_old_range
            return x1new + k * (x_old - x1old)
        else:
            return x_old

    print('Add z-values on bot. edge from scaled bot. part of longer profiles. Found z shifts [m]:')
    for i0, (ddist, st0, st1, en0, en1) in enumerate(zip(
            np.diff(edge_dist),
            *[s for se in (ctd_prm['starts'][ok_ends][b_run_to_edge], ctd_ends_f) for s in (se[:-1], se[1:])]
            )):
        print(f'{i0}: ', end='')

        # names numeric suffixes: 0 - current, 1 - next profile
        #                         s - shorter (shallow), l - longer (deeper) profile
        """
        Figure. Trying the interpolation to be more hydrophysic
        - left: scaling the bottom part of longer profile (l-profile) to the depth where its field value z is equal  
        to the last z of shorter one.
        - right: projection of PE to SE, where P is found such that z(P) nealy equal to z(S) (with restrictions)

                                l     s                                 l     s
        ~st_l+i_y_l_prior       _|...|-en_s, y_s, z_s
                                 | ~/                                   P   S new y end
        st_z_l_prior, y_found, ~z_s -|~/                    ->          ~z_s-|~/ 
        en_l                    _|/                                     _|/
                                                                          E   y_max
                                     [y_s, y_e] to [st_z_l_prior, y_e]
        Then interp z from l-profile bottom part 
        """

        y0, y1 = ctd_depth[[en0, en1]]
        if y0 > y1:  # go up
            st_s, st_l = st1, st0
            en_s, en_l = en1, en0
            y_s, y_e = y1, y0
        else:
            st_s, st_l = st0, st1
            en_s, en_l = en0, en1
            y_s, y_e = y0, y1
        z_s = ctd_z[en_s]  # todo: filter bad z. How?
        dz2en = ctd_z[st_l:en_l] - z_s  # z differences of longer profile to z_s
        # index on l-profile with ~ same depth as y_s
        i_y_l_prior = min(ctd_depth[st_l:en_l].searchsorted(y_s), dz2en.size - 1)

        # Search y on l-profile that have ~ same value as z_s = z[en_s]:

        # search depth range restriction (down, up) from y_s tuple:
        # z search limits
        dz_min, dz_max = (0, dz2en[-1]) if z_s < ctd_z[en_l] else (dz2en[-1], 0)

        max_ddepth = (y_e - y_s,  # if bottom level changes may be not need this restriction
                      -(y_s - y_s / (1 + ddist))
                      # in opposite direction: if ddist -> oo,|max_ddepth|-> y_s
                      )

        # Search best z on l-profile that nealy equal to the last z on shallower profile
        # - method 1: search same z with account of nearest depth - where its diff. to z_s changes sign.
        # Method fails if no dz2en inversion and not robust if noise but restrictions indexes will be found (sl_stop).
        # If it fails next search method will be applied.
        dy2en_found = np.array([np.inf, np.inf])
        st_found = [0, 0]
        sl_stop = [i_y_l_prior, i_y_l_prior]   # relative interval satisfying restrictions stub
        for up, sl in ((0, slice(i_y_l_prior + 1, dz2en.size, 1)),
                       (1, slice(i_y_l_prior, 0, -1))):  # search down: up = 0, search up: up = 1
            # Update searching slice basing on z restriction
            b_bad = (dz2en[sl] < dz_min) | (dz2en[sl] > dz_max)
            try:
                sl_stop[up] = sl.start + np.flatnonzero(b_bad)[0]
            except IndexError:  # no values over z limits
                sl_stop[up] = sl.start + sl.step * b_bad.size
            sl = slice(sl.start, sl_stop[up], sl.step)

            # Update searching slice basing on depth restriction
            sl_stop[up] = sl.start + sl.step * (
                (sl.step * ctd_depth[st_l:en_l][sl]).searchsorted(
                    sl.step * (y_s + max_ddepth[up]))
                )
            # search index of best z in slice:
            sl = slice(sl.start, sl_stop[up], sl.step)
            try:
                i_sl = sl.step * np.flatnonzero(dz2en[sl] > 0
                                                if dz2en[i_y_l_prior] < 0 else
                                                dz2en[sl] < 0)[0]
            except IndexError:  # no dz2en inversion
                continue
            st_found[up] = i_y_l_prior + st_l + i_sl    # absolute index
            y_found = ctd_depth[st_found[up]]           # current best y
            dy2en_found[up] = y_found - y_s             # the shift from y_s


        def dy_check(st_z_l_best):
            """
            Prints y[st_z_l_prior] - y_s
            :param st_z_l_best: absolute index of best dz
            :return: (depth at best dz, accept)
            """
            y_l_best = ctd_depth[st_z_l_best]
            y_l2s = y_l_best - y_s

            print(f'{y_l2s:.1f}', end='')

            dy_to_feat = -5  # upper margin accounts for profile variability and lower separation between features, m
            if y_l2s < dy_to_feat:  # check if too high
                # check that found point is not associated to the local field feature of that depth i.e.
                # discard if have same value on shorter profile above. Better will be to find profiles similarity end
                st_s_at_y_l_best, st_s_feat = st_s + ctd_depth[st_s:en_s].searchsorted([y_l_best + dy_to_feat, y_s + dy_to_feat])
                b_fail = min(abs(ctd_z[st_s_at_y_l_best:st_s_feat] - z_s)) < abs(dz2en[i_y_l_prior])
                # or abs(ctd_z[st_s_at_y_l_best] - ctd_z[st_z_l_best]) < abs(dz2en[i_y_l_prior])
                if b_fail:
                    print(f'- discard similar z (dz = {dz2en[st_z_l_best - st_l]:g}) => scale {dz2en[i_y_l_prior]:g})')
            else:
                b_fail = not (max_ddepth[1] < y_l2s < max_ddepth[0])
                if b_fail:
                    print(f'- discard big => scale dz = {dz2en[st_z_l_best - st_l]:g} -> {dz2en[i_y_l_prior]:g})')

            if b_fail:
                y_l_best = y_s  # or try # 2 if here after # 1
            else:
                print(end=', ')
            return y_l_best, b_fail

        if np.subtract(*sl_stop) > 0:  # have some points satisfying restrictions
            b_fail = not np.isfinite(dy2en_found).any()
            if not b_fail:  # check method 1 result
                # index on l-profile with ~ same value as z_s
                st_z_l_prior = st_found[np.argmin(np.abs(dy2en_found))]
                y_found, b_fail = dy_check(st_z_l_prior)
            if b_fail:
                # - method 2. search nearest everywhere using argmin() if method 1 is failed
                # is search with punish for big abs(dy) needed?
                print(f'->', end='')
                st_z_l_prior = st_l + sl_stop[1] + np.argmin(np.abs(dz2en[slice(*reversed(sl_stop))]))
                y_found, b_fail = dy_check(st_z_l_prior)
        else:
            print(end='~')  # use default
            b_fail = True
            y_found = y_s

        if not b_fail:  # else keep simple interpolation between 2 points
            b_add_cur = i0 == i_add
            # 1. scale z so that z(P) = z(S) exactly
            # 2. project (scale) y
            # 3. find (interp) scaled z on projected y
            ctd_add_bt[b_add_cur, 2] = np.interp(
                lin_squeeze(ctd_add_bt[b_add_cur, 1], y_e, y_s, y_found),  # scale bottom edge height
                ctd_depth[st_z_l_prior:en_l],
                lin_squeeze(ctd_z[st_z_l_prior:en_l], ctd_z[en_l], ctd_z[st_z_l_prior], z_s)
                )
            pass

    for sl_in, sl_out in zip(edges_sl_in, edges_sl_out):
        ctd_add_lr[sl_out, 2] = ctd_z[sl_in]
    ctd_add_bt_f = ctd_add_bt[np.isfinite(ctd_add_bt[:, 1:]).any(axis=1), :]
    ctd_add_bt_f_under_bot = ctd_add_bt_f; ctd_add_bt_f_under_bot[:, 1] += (5 * cfg['y_resolution_use'])
    #  todo: y= ctd_add_bt_f.y + 5*cfg['y_resolution_use'] - not sufficient but if multiply > 20
    #  then fill better but this significant affect curvature so: shift x such that new data will below previous only.
    #  ctd_add_bt_f.x + np.where(ctd_add_bt_f['y'].diff() > 0, 1, -1) * cfg['y_resolution_use']
    #  cols = list(ax2col) + [col_z]
    ctd_with_adds = np.vstack([
        np.column_stack([ctd_dist, ctd_depth, ctd_z])[ctd_ok, :],
        ctd_add_bt_f,
        ctd_add_bt_f_under_bot,
        ctd_add_lr[np.hstack([ctd_ok[sl_in] for sl_in in edges_sl_in])],
        ])
    return ctd_with_adds


# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
def main(new_arg=None):
    """
    Veusz files must have CTD data dict with fields:
        'starts' - start indexes of each profile
        'ends' - end of profile with maximum depth
        ...
    :param new_arg:
    :return:
    """
    global l

    from utils2init import init_file_names, cfg_from_args, \
        dir_create_if_need, this_prog_basename

    cfg = cfg_from_args(my_argparser(), new_arg)
    if not cfg:
        return
    if new_arg == '<return_cfg>':  # to help testing
        return cfg

    cfg['in']['db_path'] = Path(cfg['in']['db_path'])
    cfg['out']['path'] = cfg['in']['db_path'].with_name(cfg['out']['subdir_out'])
    cfg['vsz_files']['path'] = cfg['in']['db_path'].with_name(cfg['vsz_files']['subdir'])

    dir_create_if_need(cfg['out']['path'])
    if not Path(cfg['vsz_files']['export_dir']).is_absolute():
        cfg['vsz_files']['export_dir'] = cfg['vsz_files']['path'] / cfg['vsz_files']['export_dir']
    dir_create_if_need(cfg['vsz_files']['export_dir'])

    cfg['process'].setdefault('dt_point2run_max')
    # Logging
    l = init_logging(logging, None, cfg['program']['log'], cfg['program']['verbose'])
    print('\n' + this_prog_basename(__file__), end=' started. ')
    if cfg['process']['interact'].lower() == 'false':
        cfg['process']['interact'] = False
    ax = None
    load_vsz = load_vsz_closure(cfg['program']['veusz_path'])
    try:
        # dir_walker
        cfg['vsz_files']['paths'], cfg['vsz_files']['nfiles'], cfg['vsz_files']['path'] = init_file_names(
            **cfg['vsz_files'], b_interact=False)

        # Load data #################################################################
        with pd.HDFStore(cfg['in']['db_path'], mode='r') as cfg['in']['db']:
            try:  # Sections
                navp_all = cfg['in']['db'][cfg['in']['table_sections']]  # .sort()
            except KeyError as e:
                l.error('Sections not found in %s!', cfg['in']['db_path'])
                raise e
            navp_all = navp_all[np.isfinite(navp_all.Lat)]  # remove nans

            info_all_sec_list = ['{} "{}" {:%d.%m.%y %H:%M} - {:%d.%m.%y %H:%M}UTC'.format(
                navp_d['sec_#'],
                navp_d.get('sec_name', navp_d['stem_time_st']),
                *navp_d['indexs'][[0, -1]].to_pydatetime()
                ) for navp, navp_d in ge_sections(navp_all, cfg)]
            l.warning('Found %s sections:\n%s. %s', len(info_all_sec_list), '\n'.join(info_all_sec_list),
                      'Begin from section {}!'.format(cfg['process']['begin_from_section']) if
                      cfg['process']['begin_from_section'] > 1 else 'Processing...'
                      )
            vsze = None
            for navp, navp_d in ge_sections(navp_all, cfg, isec_min=cfg['process']['begin_from_section']):
                if __debug__:
                    plt.close('all')
                    # to flip track run:
                    # navp, navp_d = next(ge_sections(pd.concat({navp_d['sec_name']: navp})[::-1], cfg))
                print(end='\n...')
                stem_time = navp_d['stem_time_st'] + '{:-%d_%H%M}'.format(navp_d['time_msg_max'])  # name section files

                # Load processed CTD data from Veusz vsz
                ctd, ctd_prm, vsze = load_cur_veusz_section(cfg, navp_d, vsze)

                ctd, ctd_prm, navp_d['ictd'] = data_sort_to_nav(navp, navp_d['exclude'], navp_d['b_invert'], ctd,
                                                                ctd_prm, cfg)
                # Calculate CTD depth and other output fields

                try:
                    # for interp must (np.diff(navp_d['indexs'].values.view(np.int64)) > 0).all()
                    for coord in ['Lat', 'Lon']:
                        ctd.loc[:, coord] = rep2mean(np.interp(
                            ctd.time.to_numpy(np.int64),
                            navp_d['indexs'].view(np.int64),
                            navp.iloc[navp_d['isort'], navp.columns.get_loc(coord)].values),
                            x=ctd.time.to_numpy(np.int64))
                    ctd = add_ctd_params(ctd, {**cfg, 'out': {'data_columns': set(
                        ctd.columns[~ctd.columns.str.startswith('shift')]).union(
                        cfg['out']['data_columns'] + ['depth'])
                        }})
                except Exception as e:
                    l.exception('\nCTD depth calculation error - assigning it to "Pres" insted! %s')
                    ctd['depth'] = ctd.Pres.abs()

                # Add full resolution bottom profile to section from navigation['DepEcho']
                qstr = "index>=Timestamp('{}') & index<=Timestamp('{}')".format(
                    navp_d['indexs'][0], navp_d['time_poss_max'])
                nav = cfg['in']['db'].select(cfg['in']['table_nav'], qstr, columns=['DepEcho', 'Lat', 'Lon'])
                have_bt = 'DepEcho' in nav.columns and any(nav['DepEcho'])
                if have_bt:
                    bt, bbed, ax = filt_depth(nav['DepEcho'], **cfg['process'])
                    bt = pd.DataFrame({'DepEcho': bt, 'time': nav.index[bbed]})  # .view(np.int64)
                    have_bt = 'DepEcho' in nav.columns and not bt.empty
                    nav.drop(columns='DepEcho')
                else:
                    l.warning('No depth (DepEcho column data) in navigation!')

                # Distance calculation
                # 1. load navigation at CTD run starts and ends
                # todo: dist_clc(nav, ctd_time, cfg): calc full dist
                # - need direction to next route point and projection on it?
                df_points = h5select(
                    cfg['in']['db'], cfg['in']['table_nav'], ['Lat', 'Lon', 'DepEcho'],
                    time_points=ctd.time.iloc[np.append(ctd_prm['starts'], ctd_prm['ends'])].values,
                    dt_check_tolerance=cfg['process']['dt_search_nav_tolerance'],
                    query_range_lims=[navp_d['indexs'][0], navp_d['time_poss_max']],
                    # query_range_pattern=qstr  - commented to add to_edge
                    )[0]
                # Try get non NaN from dfL if it has needed columns (we used to write there edges' data with _st/_en suffixes)
                # if nav have nans search in other places
                df_na = df_points.isna()
                if df_na.any().any():
                    # ctd is an already loaded data, so try search here first
                    cols_interp = [c for c in df_na.columns[df_na.any()] if c in ctd.columns]
                    ctd_sort = ctd[cols_interp].set_index(ctd.time).sort_index()
                    #dfL = cfg['in']['db'][cfg['in']['table_runs']] - other variant
                    for col in cols_interp:
                        try:
                            ctd_sort_no_na = ctd_sort[col].dropna()
                            vals = ctd_sort_no_na.iloc[
                                inearestsorted(ctd_sort_no_na.index.values, df_points.index[df_na[col]].values)
                                # better inearestsorted_around with next interp
                            ].values
                        except IndexError:
                            continue  # not found
                        # vals = df_nav_col[col]
                        if vals.any():
                            df_points.loc[df_na[col], col] = vals

                    # dfL_col_suffix = 'st' if cfg['out']['select_from_tablelog_ranges'] == 0 else 'en'
                    # for col in cols_nav:
                    #     col_dat = f'{col}_{dfL_col_suffix}'
                    #     if isna[col].any() and  col_dat in dfL.columns:
                    #         b_use = isna[col].values & dfL[col_dat].notna().values
                    #         nav2add.loc[b_use, col] = dfL.loc[b_use, col_dat].values

                n_profiles = ctd_prm['starts'].size

                # 3. distances for each CTD data
                lonlat = df_points[['Lon', 'Lat']].values.T
                ctd['dist'], run_time_topbot, run_dist_topbot = dist_ctd(
                    time_ctd=ctd.time.to_numpy(np.int64),
                    time_points_st=df_points.index[:n_profiles].to_numpy(np.int64),
                    time_points_en=df_points.index[n_profiles:].to_numpy(np.int64),
                    lonlat_points_st=lonlat[:, :n_profiles],
                    lonlat_points_en=lonlat[:, n_profiles:],
                    )
                run_dist = run_dist_topbot[::2]

                # Location info to insert in Surfer plot (if need)
                imax_run_dist_topbot = run_dist_topbot.argmax()
                msg = '\n'.join([
                    '{:g}km – {:%d.%m.%y %H:%M}UTC, {:.6f}N, {:.6f}E'.format(
                        round(dist, 1), np.array(run_time_topbot[ich], 'M8[ns]').astype('M8[s]').item(),
                        df_points.Lat.iloc[i_points], df_points.Lon.iloc[i_points]
                        ) for ich, i_points, dist in zip(
                        [0, imax_run_dist_topbot],                                      # index for interchanged top bot
                        [0, (imax_run_dist_topbot % 2) * n_profiles + (imax_run_dist_topbot // 2)],     # sequential top bot
                        [0, run_dist_topbot[imax_run_dist_topbot]],
                        )
                    ])
                l.warning(msg)

                # Adjust (fine) x gridding resolution if too many profiles per distance:
                dd = np.subtract(*run_dist_topbot[[-1, 0]]) / n_profiles
                cfg['x_resolution_use'] = min(cfg['out']['x_resolution'], dd / 2)
                cfg['y_resolution_use'] = cfg['out']['y_resolution']

                # Bottom edge of CTD path
                # -----------------------
                # todo: def edge_of_CTD_path(CTD, bt['DepEcho']):
                print('Bottom edge of CTD path. 1. filtering...', end=' ')
                edge_depth = ctd.depth.iloc[ctd_prm['ends']].values
                edge_dist = ctd.dist.iloc[ctd_prm['ends']].values

                if have_bt:
                    # - get dist for depth profile by extrapolate CTD run_dist
                    ctd_isort = ctd.time.iloc[ctd_prm['starts']].values.argsort()
                    # nav_dist_isort = nav_dist.argsort()

                    # if not enough points then use max of it and CTD
                    if 0 < bt.size <= 2:
                        bt.loc[:, 'DepEcho'] = np.append(bt, ctd.depth.iloc[ctd_prm['ends']].values).max()

                    # Change index from time to dist
                    bt.index = np.interp(
                        bt.time.view(np.int64),
                        ctd.time.iloc[ctd_prm['starts'][ctd_isort]].to_numpy(np.int64),
                        run_dist[ctd_isort])
                    bt.index.name = 'dist'  # bt._AXIS_ALIASES['dist'] = 'index'
                    bt = bt.sort_index()
                    # bt.drop_duplicates([],keep='last',inplace=True)
                    bt = bt[~bt.index.duplicated()]

                    # navp_d['indexs'].values bt.index[bbed].values.view(np.int64),ctd['time'][ctd_prm['starts']].view(np.int64), run_dist)
                    # nav_dist = np.interp(nav_dists, run_dist, run_dist[])

                    ## Keep CTD points only near the bottom ##
                    # bottom at each path point:
                    edge_bed = np.interp(run_dist, bt.index, bt.DepEcho)
                    # ctd['time'][ctd_prm['ends']].view(np.int64),
                    # bt.index[bbed].view(np.int64), bt.DepEcho)

                    # Correct echo data if data below bottom: adding constant
                    echo_to_depths = edge_bed - edge_depth
                    bBad = echo_to_depths < 0
                    if np.any(bBad):
                        if sum(bBad) / len(bBad) > 0.4:  # > ~half of data need corrction
                            bed_add_calc = -np.mean(echo_to_depths[bBad]) + 0.5
                            # added constant because of not considered errors for bottom will below CTD data
                            l.warning(
                                '{:.1f}% of runs ends is below bottom echo profile. '
                                'Adding mean of those shift + some delta = {:.1f}'.format(
                                    100 * sum(bBad) / len(bBad), bed_add_calc))
                            # move all echo data down
                            bt.DepEcho += bed_add_calc  # bed_add_calc = 0
                            edge_bed += bed_add_calc
                            echo_to_depths = edge_bed - edge_depth
                            bBad = echo_to_depths < 0
                        # replace remained depths below CTD with CTD depth
                        edge_bed[bBad] = edge_depth[bBad]
                else:  # will save/display depth as constant = max(Pres at CTD ends)
                    edge_bed = np.max(edge_depth) * np.ones_like(edge_depth)

                # Go along bottom and collect nearest CTD bottom edge path points
                ok_edge = np.zeros_like(edge_dist, [('hard', np.bool8), ('soft', np.bool8)])
                ok_edge['soft'][:] = abs(edge_depth) > cfg['process']['convexing_ctd_bot_edge_max']
                k = 2  # vert to hor. scale for filter, m/km
                edge_path_scaled = np.vstack((edge_dist * k, edge_depth))
                for bed_point in np.vstack((edge_path_scaled[0, :], edge_bed)).T:
                    i = closest_node(bed_point[:, np.newaxis], edge_path_scaled)
                    ok_edge['soft'][i] = True
                ok_edge['soft'][[0, -1]] = True  # not filter edges anyway

                # Filter bottom edge of CTD path manually
                if __debug__:
                    interactive_deleter(x=edge_dist,
                                        y_kwrgs=(
                                            {'data': edge_bed, 'label': 'depth'},
                                            {'data': edge_depth, 'label': 'source'}
                                            ),
                                        mask_kwrgs={'data': ok_edge['soft'], 'label': 'closest to bottom',
                                                    'marker': "o", 'fillstyle': 'none'},
                                        ax=ax,
                                        ax_title='Bottom edge of CTD path filtering',
                                        ax_invert=True, clear=True,
                                        stop=cfg['process']['interact'])
                ok_edge['hard'] = ok_edge['soft']

                """



                # filter it
                max_spike_variants= [1, 5]; colors='gb'  #soft and hard filtering
                if __debug__:
                    plt.style.use('bmh')
                    f, ax = plt.subplots()
                    ax.plot(edge_dist, edge_depth, color='r',
                            alpha=0.5, label='source')
                    ax.plot(edge_dist, edge_bed, color='c', label='depth')


                for max_spike, key, clr in zip(max_spike_variants, ok_edge.dtype.fields.keys(), colors):
                    ok_edge[key]= ~b1spike_up(edge_depth, max_spike)
                    ok_edge[key][[0, -1]] = True #not filter edges
                    if __debug__:

                        ax.plot(edge_dist[ok_edge[key]], edge_depth[ok_edge[key]],
                                color=clr, label= '{} spike height= {}'.format(key, max_spike))

                """
                # Creating bottom edge of CTD path polygon.

                # Extend polygon edges according to grid extending (see below)
                lim = Axes2d(*[MinMax(ctd[col].min(), ctd[col].max()) for col in ax2col[:]])
                edge_dist[[0, -1]] = [lim.x.min - 0.1, lim.x.max + 0.1]  # +- cfg['x_resolution_use']
                # old +=(np.array([-1, 1]) * (cfg['out']['x_resolution'] / 2 + 0.01))
                polygon_edge_path = to_polygon(
                    *extract_repeat_at_bad_edges(np.vstack([edge_dist, -edge_depth]), ok_edge['soft']),
                    cfg['out']['blank_level_under_bot']
                    )  # use soft filter because deleting is simpler than adding in Surfer

                # Bottom depth line
                # -----------------
                # you can also set depth to be:
                # - constant = max(Pres at CTD ends) if not have_bt: see else with l.info() (edge_bed already assigned)
                # - lowest CTD data: if bt is empty or set (have_bt=False; edge_bed=edge_depth+1): see next else with l.info()

                if have_bt:
                    # add CTD end's depth data to DepEcho
                    b_add = b_add_ctd_depth(bt.index, bt.DepEcho.values, edge_dist, edge_depth,
                                            max_dist=0.5, max_bed_gradient=50, max_bed_add=5)
                    min_df_bt = np.nanmin(bt.DepEcho.values)
                    b_add &= edge_depth > max(cfg['process']['min_depth'] or 10, min_df_bt)  # add only where CTD end's depth > min(depth) and > 10m

                    # f, ax = plt.subplots(); ax.invert_yaxis()
                    # ax.plot(bt.index, bt.DepEcho.values, alpha=0.5, color='c', label='depth')
                    # plt.show()
                    nav_dist = np.append(bt.index, edge_dist[b_add])
                    isort = np.argsort(nav_dist)
                    nav_dist = nav_dist[isort]
                    nav_dep = np.append(bt.DepEcho.values, edge_bed[b_add])[isort]
                    # edge_depth
                else:
                    l.info('set bot line as constant = max(Pres at CTD ends)')
                    nav_dep = edge_bed[ok_edge['soft']]  # constant if not have_bt
                    nav_dist = edge_dist[ok_edge['soft']]

                # filter depth
                if have_bt:
                    # import statsmodels.api as sm
                    # depth_lowess = sm.nonparametric.lowess(bt, nav_dist, frac=0.1, is_sorted= True, return_sorted= False) # not works for frac=0.01
                    try:
                        plt.close()
                    except Exception as e:
                        pass

                    b_good = np.ones_like(nav_dist, dtype=np.bool8)
                    depth_lowess = gaussian_filter1d(nav_dep, 10)  # 300, mode='nearest'

                    if ax is None or not plt.fignum_exists(ax.figure.number):
                        f, ax = plt.subplots()
                        ax.invert_yaxis()
                    else:
                        ax.clear()
                    ax.plot(edge_dist, edge_bed, alpha=0.3, color='c', label='source under CTD')
                    interactive_deleter(x=nav_dist,
                                        y_kwrgs=({'data': depth_lowess, 'label': 'lowess'},
                                                 {'data': nav_dep, 'label': 'source'}),
                                        mask_kwrgs={'data': b_good},  # , 'label': 'closest to bottom'
                                        ax=ax, ax_title='Bottom depth smoothing',
                                        ax_invert=True, position=(15, -4), stop=cfg['process']['interact'])
                    # Deletion with keeping left & right edge values:
                    nav_dist, nav_dep, depth_lowess = extract_repeat_at_bad_edges(np.vstack(
                        [nav_dist, nav_dep, depth_lowess]), b_good)
                    if not plt.fignum_exists(ax.figure.number):
                        _, ax = plt.subplots()
                        ax.yaxis.set_inverted(True)
                    ax.plot(nav_dist, nav_dep, alpha=0.5, color='k', label='source')
                    plt.show()

                    if np.max(depth_lowess) - np.min(depth_lowess) > 1:  # m, checks if it became ~constant
                        bok_depth_lowess = False  # Not applying lowess filtering, but it may be useful for awful data:
                        if bok_depth_lowess:      # dbstop if want to apply lowess, set: bok_depth_lowess = True
                            l.info('set bot line to Gaussian "lowess" filtered depth')
                            nav_dep = depth_lowess  # if whant lowess some regions use as: sl = nav_dist >18.7; nav_dep[sl]= depth_lowess[sl]
                    else:
                        l.info('set bot line to DepEcho data at runs ends')
                        nav_dep = edge_bed[ok_edge['soft']]
                        nav_dist = edge_dist[ok_edge['soft']]

                # Save polygons

                nav_dist[[0, -1]] = edge_dist[[0, -1]]  # Extend polygon edges according to grid extending (see below)
                # old: += (np.array([-1, 1]) * (cfg['out']['x_resolution'] / 2 + 0.01))
                polygon_depth = to_polygon(nav_dist, -nav_dep, cfg['out']['blank_level_under_bot'])
                polygon_depth = polygon_depth.simplify(0.05, preserve_topology=False)  #

                # # Filter bottom depth by bottom edge of CTD path (keep lowest)
                # polygon_depth= polygon_depth.intersection(polygon_edge_path)
                # polygon_depth= polygon_depth.buffer(0)

                save_shape(cfg['out']['path'] / f'{stem_time}Depth', polygon_depth, 'BNA')

                # CTD blank polygons for top and bottom areas
                # -------------------------------------------
                print('Saving CTD blank polygons. ', end='')
                top_edge_dist = ctd.dist.values[ctd_prm['starts']]
                top_edge_dist[[0, -1]] = edge_dist[[0, -1]]
                # += (np.array([-1, 1]) * (cfg['out']['x_resolution'] / 2 + 0.01))  # To hide polygon edges on image ~*k_inv
                polygon_top_edge_path = to_polygon(
                    top_edge_dist,
                    -(lambda a: np.fmin(medfilt(a), a))(ctd.depth.iloc[ctd_prm['starts']].values),
                    0)
                save_shape(cfg['out']['path'] / (stem_time + 'P'), MultiPolygon(
                    [polygon_top_edge_path, polygon_edge_path]), 'BNA')

                # Runs down tracks
                # ----------------
                print('Saving CTD pressure(depth) runs in "*P.txt" and data with coord. for gridding in "*params.txt"')
                ok_ctd = ctd.depth.notna()  # find NaNs in Pres
                ctd.depth = -ctd.depth  # temporary
                np.savetxt(cfg['out']['path'] / (stem_time + 'P.txt'),
                           ctd.loc[ok_ctd, ['dist', 'depth']].values,
                           fmt='%g\t%g', header='Dist_km\tDepth_m', delimiter='\t', comments='')
                cols_nav = ['dist', 'depth', 'Lat', 'Lon']
                cols_nav_suffix = ['km', 'm', '°', '°']
                cols = ['_'.join(c).capitalize() for c in zip(cols_nav, cols_nav_suffix)] + cfg['out']['data_columns']
                np.savetxt(cfg['out']['path'] / (stem_time + 'params.txt'),
                           ctd.loc[ok_ctd, cols_nav + cfg['out']['data_columns']].values,
                           fmt='\t'.join(['%g'] * len(cols)), delimiter='\t', header='\t'.join(cols), comments='')
                ctd.depth = -ctd.depth  # back to positive

                # Gridding ######################################################################

                for i_col, (col, ax, lim_ax) in enumerate(zip(ax2col[:], ax2col._fields, lim[:])):
                    shift = cfg.get(f'{ax}_resolution_use', cfg['out'][f'{ax}_resolution'])
                    lim = lim._replace(**{ax: lim_ax._replace(min=lim_ax.min - shift, max=lim_ax.max + shift)})

                # Resulting grid coordinates

                x = np.arange(lim.x.min, lim.x.max, cfg['x_resolution_use'])
                y = np.arange(lim.y.min, lim.y.max, cfg['y_resolution_use'])

                gdal_geotransform = (lim.x.min, cfg['x_resolution_use'], 0,
                                     -lim.y.min, 0, -cfg['y_resolution_use'])
                # [0] координата x верхнего левого угла
                # [1] ширина пиксела
                # [2] поворот, 0, если изображение ориентировано на север
                # [3] координата y верхнего левого угла
                # [4] поворот, 0, если изображение ориентировано на север
                # [5] высота пиксела

                # Grid blanking bot. edge: use hard filter and blur by 1 run to keep more of our beautiful contours
                d_blur = (lambda d: np.where(d[1:] > 0, d[1:], np.where(d[:-1] < 0, -d[:-1], cfg['y_resolution_use']))
                          )(np.pad(np.diff(edge_depth[ok_edge['hard']]), 1, 'edge'))     # to less del. data
                y_edge_path = np.interp(x, edge_dist[ok_edge['hard']], edge_depth[ok_edge['hard']] + d_blur)
                xm, ym = np.meshgrid(x, y)
                # to prevent stepping edge of blanking that reach useful data at high slope regions:
                ym_blank = ym > y_edge_path + cfg['y_resolution_use']

                l.info('Gridding to {}x{} points'.format(*xm.shape))
                b_1st = True
                for iparam, col_z in enumerate(cfg['out']['data_columns']):  # col_z= u'Temp'
                    label_param = col_z  # .lstrip('_').split("_")[0]  # remove technical comments in names
                    icol_z = ctd.columns.get_loc(col_z)
                    print(label_param, end=': ')
                    if col_z not in ctd:
                        print(' not loaded')
                        continue
                    if __debug__:
                        sys_stdout.flush()
                        i = 0

                    if False:  # col_z=='O2': #. Debug!!!
                        # remove bad runs
                        iruns_del = np.int32([18, 19])  # insert right run indexes of current section
                        l.warning('Removing ', col_z, ' runs:')
                        for st, en in zip(ctd_prm['starts'][iruns_del], ctd_prm['ends'][iruns_del]):
                            l.warning('{:%d %H:%M}'.format(timzone_view(pd.to_datetime(
                                ctd.time.iloc[st]), cfg['out']['dt_from_utc'])))
                            ctd.iloc[st:en, icol_z] = np.NaN

                    ctd_with_adds = add_data_at_edges(
                        *ctd[['dist', 'depth', col_z]].values.T, ctd_prm, edge_depth, edge_dist,
                        ok_ctd.values, ok_edge['soft'], cfg, lim.x[:]
                        )
                    """
                    griddata_by_surfer(ctd, path_stem_pattern=os_path.join(
                                cfg['out']['path'], 'surfer', f'{stem_time}{{}}'),  ),
                                       xCol='Dist', yCol='Pres',
                                       zCols=col_z, NumCols=y.size, NumRows=x.size,
                                       xMin=lim.x.min, xMax=lim.x.max, yMin=lim.y.min, yMax=lim.y.max)
                    """
                    write_grd_this_geotransform = write_grd_fun(gdal_geotransform)
                    for interp_method, interp_method_subdir in [['linear', ''], ['cubic', 'cubic\\']]:
                        dir_interp_method = cfg['out']['path'] / interp_method_subdir
                        dir_create_if_need(dir_interp_method)
                        # may be very long! : try extent=>map_extent?
                        z = interpolate.griddata(points=ctd_with_adds[:, :-1],
                                                 values=ctd_with_adds[:, -1],
                                                 xi=(xm, ym),
                                                 method=interp_method)  # , rescale= True'cubic','linear','nearest'
                        # Blank z below bottom (for compressibility)
                        z[ym_blank] = np.NaN
                        # if navp_d['b_invert']:
                        #    z= np.fliplr(z)

                        if __debug__:
                            try:
                                # f= plt.figure((iparam+1)*10 + i); i+=1
                                # if (not plt.fignum_exists(ax.figure.number)):
                                f, ax = plt.subplots(num=(iparam + 1) * 10 + i, sharex=True, sharey=True)
                                i += 1

                                ax.set_title(label_param)
                                im = plt.imshow(z, extent=[x[0], x[-1], lim.y.min, lim.y.max],
                                                origin='lower')  # , extent=(0,1,0,1)
                                ax.invert_yaxis()
                                # np.seterr(divide='ignore', invalid='ignore')
                                # contour.py:370: RuntimeWarning: invalid value encountered in true_divide
                                CS = plt.contour(xm, ym, z, 6, colors='k')
                                plt.clabel(CS, fmt='%g', fontsize=9, inline=1)
                                if have_bt:
                                    plt.plot(nav_dist, nav_dep, color='m', alpha=0.5, label='Depth')  # bot
                                if b_1st:
                                    plt.plot(ctd.dist.values, ctd.depth.values, 'm.', markersize=1, alpha=0.5,
                                             label='run path')
                                CBI = plt.colorbar(im, shrink=0.8)
                                # CB = plt.colorbar(CS, shrink=0.8, extend='both')
                                # plt.gcf().set_size_inches(9, 3)
                                plt.savefig(dir_interp_method / ('grid' + stem_time + label_param + '.png'))
                                plt.show(block=False)
                                # , dpi= 200
                                pass
                            except Exception as e:
                                l.error('\nCan not draw contour! ', exc_info=1)
                        # gdal_drv_grid.Register()
                        file_grd = cfg['out']['path'] / interp_method_subdir / f'{stem_time}{label_param}.grd'
                        write_grd_this_geotransform(file_grd, z)
                        if b_1st:
                            b_1st = False

        print('\nOk.')
        if __debug__:
            plt.close('all')
        vsze.Close()
    except Ex_nothing_done as e:
        print(e.message)
    except Exception as e:
        l.exception('Not good:')  # l.error((e.msg if hasattr(e,'msg') else ''), exc_info=1) # + msg_option
    finally:
        # f.close()
        # l.handlers[0].flush()
        logging.shutdown()


if __name__ == '__main__':
    main()

""" Trash

        # add delta time to ensure that found data indexes would after tst_approx
        # np.atleast_1d(np.searchsorted(  # big enough at end
        # datetime_mean(tdata_st, np.append(tdata_st[1:], np.array('9999-12-01', '<M8[s]'))), tst_approx))

        for key in ctd.keys():
            if len(ctd[key]) == ctd['time']:
                ctd[key]= ctd[key][bGood]
        ctd_prm['starts']= ctd_prm['starts'] - np.searchsorted(ctd_prm['starts'] - np.flatnonzero(~bGood))



    #scipy.ndimage.interpolation.map_coordinates or scipy.interpolate.RegularGridInterpolator
    #

    # or use
    # matplotlib.tri.Triangulation and a matplotlib.tri.TriInterpolator
    # old grid
    x, y = np.mgrid[0:1:201j, 0:1:513j]
    z = np.sin(x*20) * (1j + np.cos(y*3))**2   # some data

    # new grid
    x2, y2 = np.mgrid[0.1:0.9:201j, 0.1:0.9:513j]

    # interpolate onto the new grid
    z2 = scipy.interpolate.griddata((x.ravel(), y.ravel()), z.ravel(), (x2, y2), method='cubic')

    #If your data is on a grid (i.e., the coordinates corresponding to value z[i,j] are (x[i], y[j])), you can get more speed by using scipy.interpolate.RectBivariateSpline
    z3 = (scipy.interpolate.RectBivariateSpline(x[:,0], y[0,:], z.real)(x2[:,0], y2[0,:])
          + 1j*scipy.interpolate.RectBivariateSpline(x[:,0], y[0,:], z.imag)(x2[:,0], y2[0,:]))


    #2048x2048 mesh of irregular data zi = f(xi, yi) which are essentially three independent sets of 2048 real values. I need to smoothly interpolate (perhaps bicubic spline) that into a regular mesh of wi = f(ui, vi) where ui and vi are integer values from 0 to 2047.
    y,x=indices([2048,2048],dtype='float64')
    z = randn(2048,2048)
    yr = y + randn(2048,2048)
    xr = x + randn(2048,2048)
    zn = griddata(xr.ravel(),yr.ravel(),z.ravel(),x,y)
    zl = griddata(xr.ravel(),yr.ravel(),z.ravel(),x,y,interp='linear')
    
    
# Nind_st = storeIn.select_as_coordinates(cfg['in']['table_nav'], qstr)[0]
#
# Nind= bt.index.searchsorted(ctd['time'][ctd_prm['starts']])
# Nind[Nind >= bt.index.size]= bt.index.size - 1 #???????
# # Check time difference between navigation found and time of requested points
# check_time_diff(ctd['time'][ctd_prm['starts']], bt.index[Nind],
#                 cfg['process']['dt_search_nav_tolerance']*5)
# Nind+= Nind_st
# df_points= storeIn.select(cfg['in']['table_nav'], where= Nind, columns=['Lat', 'Lon', 'DepEcho'])
"""
