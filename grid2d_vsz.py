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
"""

import logging
from pathlib import Path
from sys import stdout as sys_stdout, platform
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import gdal
import numpy as np
import ogr
import pandas as pd
import pyproj  # import geog
from third_party.descartes.patch import PolygonPatch  # !Check!
from gsw import distance as gsw_distance  # from gsw.gibbs.earth  import distance
# from scipy.ndimage import gaussian_filter1d
from scipy import interpolate, diff
from scipy.ndimage.filters import gaussian_filter1d
from shapely.geometry import MultiPolygon, asPolygon, Polygon

from graphics import make_figure, interactive_deleter
from other_filters import rep2mean, is_works, too_frequent_values, waveletSmooth, despike, check_time_diff, move2GoodI, \
    inearestsorted, closest_node
from to_pandas_hdf5.CTD_calc import add_ctd_params
from to_pandas_hdf5.h5toh5 import h5select
# my
from utils2init import init_logging, Ex_nothing_done, standard_error_info
from utils_time import datetime_fun, timzone_view, multiindex_timeindex
from veuszPropagate import load_vsz_closure, export_images  # , veusz_data

# graphics/interactivity
if True:  # __debug__:
    import matplotlib

    matplotlib.rcParams['axes.linewidth'] = 1.5
    matplotlib.rcParams['figure.figsize'] = (16, 7)
    try:
        matplotlib.use(
            'Qt5Agg')  # must be before importing plt (rases error after although documentation sed no effect)
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

    p_in = p.add_argument_group('in', 'data from hdf5 store')
    p_in.add('--db_path', help='hdf5 store file path')  # '*.h5'
    p_in.add('--table_sections', help='table name with sections waypoints data')
    p_in.add('--table_nav', default='navigation',  # '/navigation/table' after saving without no data_columns= True
             help='table name with sections waypoints data')
    p_in.add('--b_temp_on_its90', default='True',
             help='When calc CTD parameters treat Temp have red on ITS-90 scale. (i.e. same as "temp90")')

    p_vsz = p.add_argument_group('vsz_files', 'data from hdf5 store')
    p_vsz.add('--subdir', default='CTD-sections', help='Path to source file(s) to parse')
    p_vsz.add('--filemask', default='[0-9]*.vsz',
              help='path mask to Veusz pattern file(s). If any files found has names that starts with current section '
                   'time formatted "%y%m%d_%H%M" then use 1st of it without modification. Else use last name that '
                   'conforms to filemask as pattern')
    p_vsz.add('--export_pages_int_list', default='0',
              help='pages numbers to export, comma separated (1 is first), 0= all')
    p_vsz.add('--export_dir', default='images(vsz)',
              help='subdir relative to input path or absolute path to export images')
    p_vsz.add('--export_format', default='png',
              help='extention of images to export which defines format')

    p_gpx = p.add_argument_group('gpx', 'symbols names')
    p_gpx.add('--symbol_break_keeping_point', default='Circle, Red', help='to break all data to sections')
    p_gpx.add('--symbol_break_notkeeping_dist_float', default='20', help='km, will not keeping to if big dist')
    p_gpx.add('--symbol_excude_point', default='Circle with X', help='to break all data to sections')
    p_gpx.add('--symbols_in_veusz_ctd_order_list',
              help="GPX symbols of section in order of Veusz joins tables of CTD data (use if section has data from several tables, see CTDitable variable in Veusz file). Used only to exclude CTD runs if its number bigger than number of section's points.")  # todo: use names of tables which Veusz loads
    p_out = p.add_argument_group('output_files',
                                 'Output files: paths, formats... - not calculation intensive affecting parameters')
    p_out.add('--subdir_out', default='subproduct', help='path relative to in.db_path')
    p_out.add('--dt_from_utc_hours', default='0')
    p_out.add('--x_resolution_float', default='0.5',
              help='Dist, km. Default is 0.5km, but if dd = length/(nuber of profiles) is less then decrease to dd/2')
    p_out.add('--y_resolution_float', default='1.0', help='Depth, m')
    p_out.add('--blank_level_under_bot_float', default='-300',
              help='Depth, m, that higher than maximum of plot y axis to not see it and to create polygon without self intersections')
    p_out.add('--data_columns_list',
              help='Comma separated string with data column names (of hdf5 table) to use. Not existed will skipped')
    p_out.add('--b_reexport_images',
              help='Export images of loaded .vsz files (if .vsz creared then export ever)')

    p_proc = p.add_argument_group('process', 'process')
    p_proc.add('--begin_from_section_int', default='0', help='0 - no skip. > 0 - skipped sections')
    p_proc.add('--interact', default='editable_figures',
               help='if not "False" then display figures where user can delete data and required to press Q to continue')
    p_proc.add('--dt_search_nav_tolerance_seconds', default='1',
               help='start interpolte navigation when not found exact data time')
    p_proc.add('--invert_prior_sn_angle_float', default='30',
               help='[0-90] degrees: from S-N to W-E, 45 - no priority')
    p_proc.add('--depecho_add_float', default='0', help='add value to echosounder depth data')
    p_proc.add('--filter_ctd_bottom_edge_bool',
               help='filter ctd_bottom_edge line closer to bottom (be convex). Try negative values: if data will found below profile then it will shift bottom profile down')
    p_proc.add('--min_depth', default='4', help='filter out smaller depths')
    p_proc.add('--max_depth', default='1E5', help='filter out deeper depths')
    p_proc.add('--filter_depth_wavelet_level_int', default='4', help='level of wavelet filtering of depth')
    # p_proc.add(
    #     '--dt_point2run_max_minutes', #default=None,
    #     help='time interval to sinchronize points on map and table data (to search data marked as excluded on map i.e. runs which start time is in (t_start_good, t_start_good+dt_point2run_max). If None then select data in the range from current to the next point')

    p_prog = p.add_argument_group('program', 'program behaviour')
    p_prog.add('--veusz_path',
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

    def write_grd1(fileOut_grd, z):
        nonlocal gdal_drv_grid, gdal_geotransform
        gdal_raster = gdal_drv_grid.Create(str(fileOut_grd), z.shape[1], z.shape[0], 1, gdal.GDT_Float32)  #
        if gdal_raster is None:
            l.error('Could not create %s', fileOut_grd)
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
    # Add one attribute
    fieldDefinitions = [('id', ogr.OFTInteger)]
    # Make dataSource
    if target_path.exists():
        target_path.unlink()
    ogr_drv = ogr.GetDriverByName(driverName)
    if not ogr_drv:
        raise GeometryError('Could not load driver: {}'.format(driverName))
    dataSource = ogr_drv.CreateDataSource(str(target_path))
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


distance = lambda lon, lat: gsw_distance(lon, lat, p=np.float(0))  # fix gsw bug of requre input of numpy type for "p"


def dist_clc(nav, ctd_time, cfg):
    """
    Calculate distanse
    selects good points for calc. distance
    :param nav:
    :return:
    """
    useLineDist = cfg['process'].get('useLineDist', 0.05)
    pass  # Not implemented


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
    Get all start points and section start points
    :param navp_in:
    :param cfg_gpx: dict whitch must have keys:
        'symbol_excude_point'
        'symbol_break_keeping_point'
    :return: numpy 2D array and bool mask of excluded points: (A, b_navp_exclude)
        Array A allows to get starts/ends for section with index i: A[0/1, i]
    """
    b_navp_exclude = navp_in.sym.values == cfg_gpx['symbol_excude_point']
    df = navp_in[~b_navp_exclude]
    # [0,: -1]
    ddist = distance(df.Lon, df.Lat) / 1e3  # km
    df_timeind, itm = multiindex_timeindex(df.index)

    if itm is None:  # isinstance(df_timeind, pd.DatetimeIndex):
        b_break_condition = df.sym == cfg_gpx['symbol_break_keeping_point']
        # Remove boundary points from sections where distance to it is greater cfg_gpx['symbol_break_notkeeping_dist']
        isec_break = np.append(np.flatnonzero(b_break_condition), df.shape[0])
        i0between_sec = isec_break[1:-1]  # if isec_break[-1]+1 == df.shape[0] else isec_break[1:
        # st_with_prev = np.append(False, abs(ddist[i0between_sec]) < cfg_gpx['symbol_break_notkeeping_dist'])
        en_with_next = np.append(abs(ddist[i0between_sec - 1]) < cfg_gpx['symbol_break_notkeeping_dist'], True)
        ranges = np.vstack((isec_break[:-1], isec_break[1:] - 1 + np.int8(en_with_next)))
        # - st_with_prev
        # remove empty sections:
        ranges = ranges[:, diff(ranges, axis=0).flat > 1]
    else:
        df_index_sect_names = df.index.levels[len(df.index.levels) - 1 - itm]
        isec_break = np.cumsum(np.int32([df.loc[sec_name].shape[0] for sec_name in df_index_sect_names]))
        isec_break[-1] += 1
        ranges = np.vstack((np.append(0, isec_break[:-1]), isec_break - 1))

    return ranges, b_navp_exclude, ddist


def ge_sections(navp_all: pd.DataFrame,
                cfg: Mapping[str, Any],
                isec_min=0,
                isec_max=np.inf) -> Iterable[Tuple[pd.DataFrame, Dict[str, Any]]]:
    """

    :param navp_all:
    :param cfg:
    :param isec_min:
    :param isec_max:
    :return: navp, navp_d - dataframe and dict of some data associated with:

    """

    ranges, b_navp_all_exclude, ddist_all = sec_edges(navp_all, cfg['gpx'])
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
            navp_d['sec_name'] = navp_all.index[st][len(navp_all.index.levels) - 1 - cfg['route_time_level']]
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
            navp_d['b_invert'] = np.diff(navp_index[[0, -1]]) < np.timedelta64(0)
            msg_invert = 'Veusz section will {}need to be inverted to approximate this route '.format(
                '' if navp_d['b_invert'] else 'not ')

        navp_d['time_msg_min'], navp_d['time_msg_max'] = [timzone_view(x, cfg['output_files'][
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
    - searches existed Veusz file named by start datetime of current section
    - creates and saves its copy with modified setting ``USE_timeRange`` in Custom dafinitions for new sections
    - opens and gets data from it
    return: Tuple:
        ctd: pd.DataFrame, loaded from Veusz CTD data,
        ctd_prm: dict of parameters of other length than ctd,
        vsze: Veusz embedded object
    """

    global load_vsz
    if load_vsz is None:
        load_vsz = load_vsz_closure(cfg['program']['veusz_path'])

    vsz_names = [Path(v).name for v in cfg['vsz_files']['namesFull'] if Path(v).name.startswith(navp_d['stem_time_st'])]
    if vsz_names:  # Load data from Veusz vsz
        l.warning('%s\nOpening matched %s as source...', navp_d['msg'], vsz_names[0])
        vsz_path = cfg['vsz_files']['path'] / vsz_names[0]
        vsze, ctd_dict = load_vsz(vsz_path, vsze, prefix='CTD')
        # vsz_path = vsz_path.with_suffix('')  # for comparbility with result of 'else' part below
        b_new_vsz = False
    else:  # Modify Veusz pattern and Load data from it, save it
        l.warning(navp_d['msg'])  # , end=' '
        if vsze:
            print('Use currently opened pattern...', end=' ')
        else:
            print('Opening last file {} as pattern...'.format(cfg['vsz_files']['namesFull'][-1]), end=' ')
            vsze, ctd_dict = load_vsz(cfg['vsz_files']['namesFull'][-1], prefix='CTD')
            if 'time' not in ctd_dict:
                l.error('vsz data not processed!')
                return None, None, None
        print('Load our section...', end='')
        vsze.AddCustom('constant', u'USE_runRange', u'[[0, -1]]', mode='replace')  # u
        vsze.AddCustom('constant', u'USE_timeRange', '[[{0}, {0}]]'.format(
            "'{:%Y-%m-%dT%H:%M:%S}'").format(navp_d['time_msg_min'], timzone_view(  # iso
            navp_d['time_poss_max'], cfg['output_files']['dt_from_utc'])), mode='replace')
        vsze.AddCustom('constant', u'Shifting_common', u'-1' if navp_d['b_invert'] else u'1', mode='replace')

        # If pattern has suffix (excluding 'Inv') then add it to our name (why? - adds 'Z')
        stem_no_inv = Path(cfg['vsz_files']['namesFull'][0]).stem
        if stem_no_inv.endswith('Inv'): stem_no_inv = stem_no_inv[:-3]
        len_stem_time_st = len(navp_d['stem_time_st'])
        vsz_path = cfg['vsz_files']['path'] / (navp_d['stem_time_st'] + (
            stem_no_inv[len_stem_time_st:] if len_stem_time_st < len(stem_no_inv) else 'Z'
        ) + ('Inv' if navp_d['b_invert'] else ''))
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

    if b_new_vsz or cfg['output_files']['b_reexport_images'] != False:
        export_images(vsze, cfg['vsz_files'], vsz_path.stem,
                      b_skip_if_exists=cfg['output_files']['b_reexport_images'] is None)

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


def runs_ilast_good(bctd_ok: Sequence,
                    starts: Sequence,
                    ends: Sequence,
                    ctd_z: Optional[Sequence],
                    range_to_bot: Optional[float] = None,
                    bottom_at_ends: Optional[Sequence] = None) -> Tuple[List, np.ndarray]:
    """
    Find last not nan value in each run
    :param bctd_ok: mask where source is good
    :param starts:  start of profiles indexes
    :param ends:    end of profiles indexes


    Filtering too short runs parametrs

    :param ctd_z:     Increase to bottom (usually positive). set NaN or small if not care
    :param range_to_bot: set NaN or big if not care
    :param bottom_at_ends:     set NaN or big (>0) if not care
    :return ends_cor, ok_runs:
     ends_cor: last value have found in each run skipping ones having bad range_to_bot
     ok_runs: bool mask size of starts: 1 for runs having good range_to_bot, else 0.
    """
    if (ctd_z is None) or (bottom_at_ends is None) or (range_to_bot is None):
        bottom_at_ends = np.empty_like(starts)  # not care
        range_to_bot = None  # not care
        ctd_z = None  # not care
    ok_runs = np.ones_like(starts, np.bool)
    ends_cor = []
    # for each profile:
    for k, (st, en, Pen) in enumerate(zip(starts, ends, bottom_at_ends)):
        try:
            i_last = np.flatnonzero(bctd_ok[st:en])[-1]
        except IndexError:  # no good in profile
            ok_runs[k] = False
            continue
        # if (np.flatnonzero(bctd_ok[st:en])).size > 0:
        i_last += st
        if ctd_z is not None:
            t = ctd_z[i_last]
            if (t - Pen) > range_to_bot:  # too far from bottom
                ok_runs[k] = False
                continue
        ends_cor.append(i_last)

    return ends_cor, ok_runs


def b_add_ctd_depth(dist, depth, add_dist, add_depth, max_dist=0.5, max_bed_gradient=50, max_bed_add=5):
    """
    Mask useful elements from CTD profile to add them to echosounder depth profile
    where it has no data near.
    Modifies (!) depth: Sets depth smaller than add_depth near same dist to NaN.
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
    # todo: also filter depth far from CTD bot using this method
    bAdd = np.ones_like(add_dist, np.bool8)
    bAnyR = True
    for i, (dSt, dEn) in enumerate(add_dist[:, np.newaxis] + np.array([[-1, 1]]) * max_dist):
        bNearLeft = np.logical_and(dSt < dist, dist <= add_dist[i])
        bNearRight = np.logical_and(add_dist[i] < dist, dist <= dEn)

        if not np.any(bNearLeft | bNearRight):
            # no depth data near measuring point
            bAnyR = False
        else:  # Check that depth below CTD
            # If bottom below inverted triangle zone, then use it
            # dSt   dEn
            # |     |
            # \    /
            #  \  /
            # --\\/_add_depth
            #    \____/\__- depth
            #

            bNearLeft = depth[bNearLeft] < add_depth[i] + max_bed_add + max_bed_gradient * np.abs(
                add_dist[i] - dist[bNearLeft])
            bAnyL = any(bNearLeft)
            if not (bAnyR | bAnyL):
                # remove Dep data (think it bad) in all preivious interval:
                bNearLeft = np.logical_and(add_dist[i - 1] < dist, dist <= add_dist[i])
                depth[bNearLeft] = np.NaN
                bAdd[int(i - 1)] = True  # add data from CTD if was not added

            bNearRight = depth[bNearRight] < add_depth[i] + max_bed_add + max_bed_gradient * np.abs(
                add_dist[i] - dist[bNearRight])
            bAnyR = any(bNearRight)
            if (bAnyR | bAnyL):
                bAdd[i] = False  # good depth exist => not need add CTD depth data
    return bAdd  # , depth


try:  # try get griddata_by_surfer() function reqwirements
    from win32com.client import constants, Dispatch  # , CastTo
    # python "c:\Programs\_coding\WinPython3\python-3.5.2.amd64\Lib\site-packages\win32com\client\makepy.py" -i "c:\Program Files\Golden Software\S
    # urfer 13\Surfer.exe"
    # Use these commands in Python code to auto generate .py support
    from win32com.client import gencache

    gencache.EnsureModule('{54C3F9A2-980B-1068-83F9-0000C02A351C}', 0, 1, 4)
    Surfer = Dispatch("Surfer.Application")


    def griddata_by_surfer(ctd, outFnoE_pattern=r'%TEMP%\xyz{}', xCol='Lon', yCol='Lat', zCols=None,
                           NumCols=None, NumRows=None, xMin=None, xMax=None, yMin=None, yMax=None):
        """
        Grid by Surfer
        :param ctd:
        :param outFnoE_pattern:
        :param xCol:
        :param yCol:
        :param zCols:
        :param NumCols:
        :param NumRows:
        :param xMin:
        :param xMax:
        :param yMin:
        :param yMax:
        :return:
        """
        tmpF = outFnoE_pattern.format('_temp') + '.csv'
        xCol = ctd.dtype.names.index(xCol) + 1
        yCol = ctd.dtype.names.index(yCol) + 1
        izCols = [ctd.dtype.names.index(zCol) + 1 for zCol in zCols]
        np.savetxt(tmpF, ctd, header=','.join(ctd.dtype.names), delimiter=',', comments='')
        dist_etrap = (yMax - yMin) / 10
        # const={'srfDupAvg': 15, 'srfGridFmtS7': 3}
        # gdal_geotransform = (x_min, cfg['output_files']['x_resolution'], 0, -y_min, 0, -cfg['y_resolution_use'])
        for i, izCol in enumerate(izCols):
            outGrd = outFnoE_pattern.format(zCols[i]) + '.grd'
            Surfer.GridData3(DataFile=tmpF, xCol=xCol, yCol=yCol, zCol=izCol, NumCols=NumCols, NumRows=NumRows,
                             xMin=xMin,
                             xMax=xMax, yMin=yMin, yMax=yMax, SearchEnable=True, SearchRad1=dist_etrap * 2,
                             ShowReport=False, DupMethod=constants.srfDupAvg, OutGrid=outGrd,
                             OutFmt=constants.srfGridFmtS7,
                             BlankOutsideHull=True, InflateHull=dist_etrap)
except Exception as e:
    l.error('\nCan not initialiase Surfer.Application! %s', standard_error_info(e))


    def griddata_by_surfer(ctd, outFnoE_pattern=r'%TEMP%\xyz{}', xCol='Lon', yCol='Lat', zCols=2,
                           NumCols=None, NumRows=None, xMin=None, xMax=None, yMin=None, yMax=None):
        pass


def data_sort_to_nav(navp,
                     navp_exclude,
                     b_invert: np.ndarray,
                     ctd: pd.DataFrame,
                     ctd_prm: Dict[str, Any],
                     cfg: Mapping[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any], np.ndarray]:
    """
    :param navp: pandas DataFrame. Navigation points table (with datemindex and obligatory columns Lat, Lon, name, sym)
    :param navp_exclude:
    :param b_invert:
    :param ctd:
    :param ctd_prm:
    :param cfg: fields:
        route_time_level: not none if points aranged right
    :return: tuple:
        - ctd,
        - ctd_prm - dict with added/replaced fields: starts, ends
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
    ctd_isort_back = np.empty_like(ctd_isort, dtype=np.int32)
    ctd_isort_back[ctd_isort] = np.arange(ctd_isort.size, dtype=np.int32)
    # closest CTD indexes to each nav point:
    navp_ictd = datetime_fun(inearestsorted, ctd_sts, navp_index, type_of_operation='<M8[ms]', type_of_result='i8')
    # check/correct one to one correspondance
    navp_ictd_isort = navp_ictd.argsort()
    navp_ictd_isort_diff = np.diff(navp_ictd[navp_ictd_isort])
    navp_ictd_ibads = np.flatnonzero(navp_ictd_isort_diff != 1)
    inav_jumped = None
    for inav, ictd in zip(navp_ictd_ibads, navp_ictd[navp_ictd_isort][navp_ictd_ibads]):
        #  Can coorect if have pairs of nav points assigned to one CTD
        #  n1   n2  n3    n4    - nav time not sorted
        #  |     \ /       \    - navp_ictd
        #  C1    C2   C3   C4   - CTD time sorted
        if navp_ictd_isort_diff[inav] == 0:
            # Same CTD assigned to this nav point and next
            # - because current or next nav point is bad and skipped previous or next CTD
            ictd = navp_ictd[navp_ictd_isort][inav]
            if inav_jumped is None:
                # check that next CTD is skipped (diff > 1)
                inav_jumped = inav + 1
                if navp_ictd_isort_diff[inav_jumped] > 1:
                    ictd_skipped = ictd + 1
                else:
                    l.warning('Not implemented condition1! Reassign nav to CTD manually')
            elif inav == inav_jumped:
                inav += 1
            else:
                l.warning('Not implemented condition2! Reassign nav to CTD manually')

            # variants: (C2n2, C3n3), (C3n2, C2n3) - here n2,n3 - current&next nav, C2,C3 - current&skipped CTD time
            # reassign skipped CTD to nav by minimise sum(C - n). So check 2 assignment variants:
            # 1. direct :
            sum1 = abs(ctd_sts[[ictd, ictd_skipped]] - navp_index[navp_ictd_isort[[inav, inav_jumped]]]).sum()
            # 2. reverce:
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

    b_far = check_time_diff(ctd.time.iloc[ctd_prm['starts'][navp_ictd]], navp_index,
                            dt_warn=pd.Timedelta(cfg['process']['dt_search_nav_tolerance']),
                            mesage='CTD runs which start time is far from nearest time of closest navigation point # [min]:\n')

    # Checking that indexer is correct.
    if len(cfg['gpx'][
               'symbols_in_veusz_ctd_order']) and 'itable' in ctd_prm:  # if many CTD data in same time right ctd will be assigned
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
                l.error('can not correct CTD table order. Check [gpx] configuration of symbols_in_veusz_ctd_order_list')
                raise (e)  # ValueError
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
        ctd_bdel = np.zeros_like(ctd_prm['starts'], bool);
        ctd_bdel[np.int32(ctd_idel)] = True
        navp_ictd = move2GoodI(navp_ictd, ctd_bdel)
        if np.flatnonzero(np.bincount(navp_ictd) > 1):  # inavp_with_many_runs =
            print("dbstop here: set manualy what to exclude")
    # remove rows
    bDel |= ctd.Pres.isna().values
    if any(bDel):
        ctd = ctd[~bDel]
        ctd_prm['starts'] = move2GoodI(ctd_prm['starts'], bDel)
        ctd_prm['ends'] = move2GoodI(ctd_prm['ends'], bDel, 'right')

    # Sort CTD data in order of points.

    # CTD order to points order:
    if cfg['route_time_level'] is None:  # arange runs according to b_invert
        navp_ictd = np.arange(
            navp_index.size - 1, -1, -1, dtype=np.int32) if b_invert else np.arange(
            navp_index.size, dtype=np.int32)
    # else:  # runs will exactly follow to route
    #     navp_ictd = np.empty_like(navp_isort, dtype=np.int32)
    #     navp_ictd[navp_isort] = np.arange(navp_isort.size, dtype=np.int32)

    if np.any(navp_ictd != np.arange(navp_ictd.size, dtype=np.int32)):
        # arrange run starts and ends in points order
        # 1. from existed intervals (sorted by time like starts from Veusz)
        run_next_from = np.append(ctd_prm['starts'][1:], ctd.time.size)
        run_edges_from = np.column_stack((ctd_prm['starts'], run_next_from))  #
        run_edges_from = run_edges_from[navp_ictd, :]  # rearrage ``from`` interals

        # 2. to indexes
        run_dst = diff(run_edges_from).flatten()
        run_next_to = np.cumsum(run_dst)
        ctd_prm['starts'] = np.append(0, run_next_to[:-1])
        run_edges_to = np.column_stack((ctd_prm['starts'], run_next_to))  # ?

        # 'ends' will be needed later
        _junks_to = (run_next_from - ctd_prm['ends'])[navp_ictd]
        ctd_prm['ends'] = run_next_to - _junks_to

        # sort CTD data
        ind_by_points = np.empty(ctd.time.size, dtype=np.int32)
        for se_to, se_from in zip(run_edges_to, run_edges_from):
            ind_by_points[slice(*se_from)] = np.arange(*se_to)
        ctd.index = ind_by_points
        ctd = ctd.sort_index()  # inplace=True???

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


def filt_depth(cfg, s_ndepth):
    """
    Filter depth
    :param cfg:
        cfg['process']['depecho_add'],
        cfg['process']['filter_depth_wavelet_level']
    :param s_ndepth: pandas time series of source depth
    :return: (depth_out, bGood, ax) - data reduced in size depth, corresponding mask of original s_ndepth, axis
    """
    # Filter DepEcho
    print('Bottom profile filtering ', end='')
    bed = np.abs(s_ndepth.values.flatten()) + cfg['process']['depecho_add']
    ok = is_works(bed)
    ok &= ~np.isnan(bed)
    if 'min_depth' in cfg['process']:
        ok[ok] &= (bed[ok] > cfg['process']['min_depth'])
    if 'max_depth' in cfg['process']:
        ok[ok] &= (bed[ok] < cfg['process']['max_depth'])
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
    if wdesp:
        coef_std_offsets = (15, 7)  # filters too many if set some < 3
        # back and forward:
        depth_out = despike(depth_out[::-1], coef_std_offsets, wdesp)[0]
        depth_out = despike(depth_out[::-1], coef_std_offsets, wdesp)[0]
        ok[ok] &= ~np.isnan(depth_out)
        depth_filt = rep2mean(depth_filt, ok,
                              s_ndepth.index.view(np.int64))  # need to execute waveletSmooth on full length

        ax.plot(np.flatnonzero(ok), bed[ok], color='g', alpha=0.9, label='despike')

        # Smooth some big spykes and noise              # cfg['process']['filter_depth_wavelet_level']=11
        depth_filt, ax = waveletSmooth(depth_filt, 'db4', cfg['process']['filter_depth_wavelet_level'], ax,
                                       label='Depth')

        # Smooth small high frequency noise (do not use if big spykes exist!)
        sGood = sum(ok)  # ok[:] = False  # to use max of data as bed
        # to make depth follow lowest data execute: depth_filt[:] = 0
        n_smooth = 5  # 30
        if sGood > 100 and np.all(np.abs(np.diff(depth_filt)) < 30):
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
                            ax=ax, ax_title='Bottom profile smoothing', ax_invert=True,
                            lines=lines, stop=cfg['process']['interact'])

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
    cfg['output_files']['path'] = cfg['in']['db_path'].with_name(cfg['output_files']['subdir_out'])
    cfg['vsz_files']['path'] = cfg['in']['db_path'].with_name(cfg['vsz_files']['subdir'])

    dir_create_if_need(cfg['output_files']['path'])
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
        cfg['vsz_files'] = init_file_names(cfg['vsz_files'], b_interact=False)

        # Load data #################################################################
        with pd.HDFStore(cfg['in']['db_path'], mode='r') as cfg['in']['db']:
            try:  # Sections
                navp_all = cfg['in']['db'][cfg['in']['table_sections']]  # .sort()
            except KeyError as e:
                l.error('Sections not found in %s!', cfg['in']['db_path'])
                raise e
            navp_all = navp_all[np.isfinite(navp_all.Lat)]  # remove nans

            info_all_sec_list = ['{} "{}"'.format(navp_d['sec_#'], navp_d.get('sec_name', navp_d['stem_time_st'])) for
                                 navp, navp_d in ge_sections(navp_all, cfg)]
            l.warning('Found %s sections:\n%s. %s', len(info_all_sec_list), '\n'.join(info_all_sec_list),
                      'Begin from section {}!'.format(cfg['process']['begin_from_section']) if cfg['process'][
                                                                                                   'begin_from_section'] > 1 else 'Processing...')

            vsze = None
            for navp, navp_d in ge_sections(navp_all, cfg, isec_min=cfg['process']['begin_from_section']):
                if __debug__:
                    plt.close('all')
                print(end='\n...')
                # Load processed CTD data from Veusz vsz
                ctd, ctd_prm, vsze = load_cur_veusz_section(cfg, navp_d, vsze)

                ctd, ctd_prm, navp_d['ictd'] = data_sort_to_nav(navp, navp_d['exclude'], navp_d['b_invert'], ctd,
                                                                ctd_prm, cfg)

                # Calculte CTD depth and other output fields
                try:
                    # for interp must np.all(np.diff(navp_d['indexs'].values.view(np.int64)) > 0)
                    for coord in ['Lat', 'Lon']:
                        ctd[coord] = rep2mean(np.interp(
                            ctd.time.to_numpy(np.int64),
                            navp_d['indexs'].view(np.int64),
                            navp.iloc[navp_d['isort'], navp.columns.get_loc(coord)].values),
                            x=ctd.time.to_numpy(np.int64))
                    ctd = add_ctd_params(ctd, {**cfg, 'output_files': {'data_columns': set(
                        ctd.columns[~ctd.columns.str.startswith('shift')]).union(
                        cfg['output_files']['data_columns'] + ['depth'])
                        }})
                except Exception as e:
                    l.exception('\nCTD depth calculation error - assigning it to "Pres" insted! %s')
                    ctd['depth'] = ctd.Pres.abs()

                # Add full resolution bottom profile to section from navigation['DepEcho']
                qstr = "index>=Timestamp('{}') & index<=Timestamp('{}')".format(
                    navp_d['indexs'][0], navp_d['time_poss_max'])
                nav = cfg['in']['db'].select(cfg['in']['table_nav'], qstr, columns=['DepEcho', 'Lat', 'Lon'])
                have_bed = 'DepEcho' in nav.columns and any(nav['DepEcho'])
                if have_bed:
                    bt, bbed, ax = filt_depth(cfg, nav['DepEcho'])
                    bt = pd.DataFrame({'DepEcho': bt, 'time': nav.index[bbed]})  # .view(np.int64)
                    have_bed = 'DepEcho' in nav.columns and not bt.empty
                    nav.drop(columns='DepEcho')
                else:
                    l.warning('No depth (DepEcho column data) in navigation!')
                # Get navigation at run starts points (to calc dist)
                # todo: dist_clc(nav, ctd_time, cfg): calc full dist
                # - need direction to next route point and projection on it?
                dfNpoints = h5select(
                    cfg['in']['db'], cfg['in']['table_nav'], ['Lat', 'Lon', 'DepEcho'],
                    ctd.time.iloc[np.append(ctd_prm['starts'], ctd_prm['ends'])],
                    dt_check_tolerance=cfg['process']['dt_search_nav_tolerance'],
                    query_range_pattern=qstr
                    )
                # use distance between start points
                ddist = np.append(0, distance(
                    *dfNpoints.iloc[:ctd_prm['starts'].size,
                     dfNpoints.columns.get_indexer(('Lon', 'Lat'))
                     ].values.T
                    ) * 1e-3)  # km

                run_dist = np.cumsum(ddist)
                l.info('Distances between CTD runs: {}'.format(np.round(ddist[1:], int(2 - np.log10(np.mean(ddist))))))
                if np.any(np.isnan(ddist)):
                    l.error('(!) Found NaN coordinates in navigation at indexes: ',
                            np.flatnonzero(np.isnan(ddist)))

                _ = np.hstack([np.full(shape=l, fill_value=v) for l, v in zip(
                    np.diff(np.append(ctd_prm['starts'], ctd.time.size)), run_dist)])  # run_dst
                ctd['dist'] = _[:ctd.depth.size] if _.size > ctd.depth.size else _
                stem_time = navp_d['stem_time_st'] + '{:-%d_%H%M}'.format(navp_d['time_msg_max'])

                # Adjust (fine) x resolution if too many profiles per distance:
                dd = ctd_prm['starts'].size / np.diff(run_dist[[0, ctd_prm['starts'].size - 1]])
                cfg['x_resolution_use'] = min(cfg['output_files']['x_resolution'], dd / 2)
                cfg['y_resolution_use'] = cfg['output_files']['y_resolution']
                strLog = '\n'.join(
                    ['{:g}km – {:%d.%m.%y %H:%M}UTC, {:.6f}N, {:.6f}E'.format(
                        round(run_dist[k], 1), pd.to_datetime(ctd.time.iloc[ctd_prm['starts'][k]]),
                        dfNpoints.Lat.iloc[k], dfNpoints.Lon.iloc[k]) for k in [0, ctd_prm['starts'].size - 1]])
                l.warning(strLog)

                # Bottom edge of CTD path
                # -----------------------
                # def edge_of_CTD_path(CTD, bt['DepEcho']):
                print('Bottom edge of CTD path. 1. filtering...', end=' ')
                edge_depth = ctd.depth.iloc[ctd_prm['ends']].values
                edge_dist = ctd.dist.iloc[ctd_prm['ends']].values

                if have_bed:
                    # - get dist for depth profile by extrapolate CTD run_dist
                    ctd_isort = ctd.time.iloc[ctd_prm['starts']].values.argsort()
                    # nav_dist_isort = nav_dist.argsort()

                    # if not enough points then use max of it and CTD
                    if (0 < bt.size <= 2):
                        bt.loc[:, 'DepEcho'] = np.append(bt, ctd.depth.iloc[ctd_prm['ends']].values).max()

                    # Change ndex from time to dist
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

                    # Correct echo data if data below bottom
                    echo_to_depths = edge_bed - edge_depth
                    bBad = echo_to_depths < 0
                    if np.any(bBad):
                        if sum(bBad) / len(bBad) > 0.4:
                            bed_add_calc = -np.mean(echo_to_depths[
                                                        bBad]) + 0.5  # last is some delta because of not considered errors for bottom will below CTD data
                            l.warning(
                                '{:.1f}% of runs ends is below bottom echo profile. '
                                'Adding mean of those shift + some delta = {:.1f}'.format(
                                    100 * sum(bBad) / len(bBad), bed_add_calc))
                            # move all echo data down
                            bt.DepEcho += bed_add_calc  # bed_add_calc = 0
                            edge_bed += bed_add_calc
                            echo_to_depths = edge_bed - edge_depth
                            bBad = echo_to_depths < 0
                        # correct only bad points
                        edge_bed[bBad] = edge_depth[bBad]
                else:  # will save/display depth as constant = max(Pres at CTD ends)
                    edge_bed = np.max(edge_depth) * np.ones_like(edge_depth)

                # Go along bottom and collect nearest CTD bottom edge path points
                ok_edge = np.zeros_like(edge_dist, [('hard', np.bool8), ('soft', np.bool8)])
                if cfg['process']['filter_ctd_bottom_edge']:
                    k = 2  # vert to hor. scale for filter, m/km
                    edge_path_scaled = np.vstack((edge_dist * k, edge_depth))
                    for bed_point in np.vstack((edge_path_scaled[0, :], edge_bed)).T:
                        ok_edge['soft'][closest_node(bed_point[:, np.newaxis], edge_path_scaled)] = True
                    ok_edge['soft'][[0, -1]] = True  # not filter edges anyway
                else:
                    ok_edge['soft'][:] = True

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
                max_spyke_variants= [1, 5]; colors='gb'  #soft and hard filtering
                if __debug__:
                    plt.style.use('bmh')
                    f, ax = plt.subplots()
                    ax.plot(edge_dist, edge_depth, color='r',
                            alpha=0.5, label='source')
                    ax.plot(edge_dist, edge_bed, color='c', label='depth')


                for max_spyke, key, clr in zip(max_spyke_variants, ok_edge.dtype.fields.keys(), colors):
                    ok_edge[key]= ~bSpike1pointUp(edge_depth, max_spyke)
                    ok_edge[key][[0, -1]] = True #not filter edges
                    if __debug__:

                        ax.plot(edge_dist[ok_edge[key]], edge_depth[ok_edge[key]],
                                color=clr, label= '{} spyke height= {}'.format(key, max_spyke))

                """
                # Creating bottom edge of CTD path polygon.
                # increasing x limits to hide polygon edges on image:
                edge_dist[[0, -1]] += (np.array([-1, 1]) * (cfg['output_files']['x_resolution'] / 2 + 0.01))
                polygon_edge_path = to_polygon(
                    edge_dist[ok_edge['soft']],  # use soft filter because deleting is simpler than adding in Surfer
                    -edge_depth[ok_edge['soft']],
                    cfg['output_files']['blank_level_under_bot'])

                # Bottom depth line
                # -----------------
                # you can also set depth to be
                # - lowest data                      (see else)
                # - constant = max(Pres at CTD ends) (see next else)

                if have_bed:
                    bAdd = b_add_ctd_depth(bt.index, bt.DepEcho.values, edge_dist, edge_depth,
                                           max_dist=0.5, max_bed_gradient=50, max_bed_add=5)
                    min_df_bt = np.nanmin(bt.DepEcho.values)
                    bAdd &= edge_depth > max(10, min_df_bt)  # add only if > min(depth) and > 10m

                    # f, ax = plt.subplots(); ax.invert_yaxis()
                    # ax.plot(bt.index, bt.DepEcho.values, alpha=0.5, color='c', label='depth')
                    # plt.show()
                    nav_dist = np.append(bt.index, edge_dist[bAdd])
                    isort = np.argsort(nav_dist)
                    nav_dist = nav_dist[isort]
                    nav_dep = np.append(bt.DepEcho.values, edge_bed[bAdd])[isort]
                    # edge_depth
                else:  # will save/display depth as constant = max(Pres at CTD ends)
                    nav_dep = edge_bed
                    nav_dist = edge_dist

                # filter depth
                if have_bed:
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
                    # Deletion
                    depth_lowess = depth_lowess[b_good]
                    nav_dep = nav_dep[b_good]  # retain alternative variant
                    nav_dist = nav_dist[b_good]
                    if (not plt.fignum_exists(ax.figure.number)):
                        _, ax = plt.subplots()
                        ax.yaxis.set_inverted(True)
                    ax.plot(nav_dist, nav_dep, alpha=0.5, color='k', label='source')
                    plt.show()
                    # try:
                    #     ax.plot(nav_dist, nav_dep, alpha=0.5, color='b', label='corrected')
                    # except Exception as e:
                    #     pass

                    if np.max(depth_lowess) - np.min(depth_lowess) > 1:  # m, checks if it became ~constant
                        bok_depth_lowess = False  # Not applying lowess filtering, but it may be useful for awful data:
                        if bok_depth_lowess:  # dbstop if want to apply lowess, set: bok_depth_lowess = True
                            nav_dep = depth_lowess  # if whant lowess some regions use as: sl = nav_dist >18.7; nav_dep[sl]= depth_lowess[sl]
                    else:  # set depth to be lowest data
                        nav_dep = edge_bed
                        nav_dist = edge_dist

                # Save polygons

                nav_dist[[0, -1]] += (np.array([-1, 1]) * (
                        cfg['output_files']['x_resolution'] / 2 + 0.01))  # To hide polygon edges on image
                polygon_depth = to_polygon(nav_dist, -nav_dep, cfg['output_files']['blank_level_under_bot'])
                polygon_depth = polygon_depth.simplify(0.05, preserve_topology=False)  #

                # # Filter bottom depth by bottom edge of CTD path (keep lowest)
                # polygon_depth= polygon_depth.intersection(polygon_edge_path)
                # polygon_depth= polygon_depth.buffer(0)

                save_shape(cfg['output_files']['path'] / (stem_time + 'Depth'), polygon_depth, 'BNA')

                # CTD blank polygons for top and bottom areas
                # -------------------------------------------
                print('Saving CTD blank polygons. ', end='')
                # todo: separate top blank polygon
                top_edge_dist = ctd.dist.values[ctd_prm['starts']]
                top_edge_dist[[0, -1]] += (np.array([-1, 1]) * (
                        cfg['output_files']['x_resolution'] / 2 + 0.01))  # To hide polygon edges on image ~*k_inv
                polygon_top_edge_path = to_polygon(top_edge_dist, -ctd.depth.iloc[ctd_prm['starts']].values, 0)
                save_shape(cfg['output_files']['path'] / (stem_time + 'P'), MultiPolygon(
                    [polygon_top_edge_path, polygon_edge_path]), 'BNA')

                # Runs tracks path
                # ----------------
                print('Saving CTD runs tracks.')
                ok_ctd = ctd.depth.notna()  # find NaNs in Pres
                ctd.depth = -ctd.depth
                np.savetxt(cfg['output_files']['path'] / (stem_time + 'P.txt'),
                           ctd.loc[ok_ctd, ['dist', 'depth']].values,
                           fmt='%g\t%g', header='Dist_km\tDepth_m', delimiter='\t', comments='')
                np.savetxt(
                    cfg['output_files']['path'] / (stem_time + 'params.txt'),
                    ctd.loc[ok_ctd, ['dist', 'depth', 'Lat', 'Lon'] + cfg['output_files']['data_columns']].values,
                    fmt='\t'.join(['%g'] * (4 + len(cfg['output_files']['data_columns']))), delimiter='\t',
                    header='\t'.join(['Dist_km', 'Depth_m', 'Lat_°', 'Lon_°'] + cfg['output_files']['data_columns']),
                    comments='')
                ctd.depth = -ctd.depth  # back to positive

                ### Gridding ######################################################################

                col2ax = {'dist': 'x', 'depth': 'y'}
                lim = {ax: {'min': ctd[col].min(),
                            'max': ctd[col].max()
                            } for col, ax in col2ax.items()
                       }

                ### Syntetic data that helps gridding ###
                # ---------------------------------------
                # Additional data along bottom edge
                # I think add_bot_x = x will be the minimum addition to avoid big stranges. Increasing here in 10 times
                # (if more data added then it will have more prority):
                add_bot_x = np.arange(lim['x']['min'], lim['x']['max'], cfg['x_resolution_use'] / 10)
                ctd_add_bt = pd.DataFrame({'x': add_bot_x,
                                           'y': np.empty(len(add_bot_x)),
                                           'z': np.empty(len(add_bot_x)),
                                           })
                # Additional data to increase grid length at left/right edges upon resolution/2
                # copy from ctd data here:
                sl_run_0_in = slice(ctd_prm['starts'][0], ctd_prm['ends'][0])  # 1st run
                sl_run_e_in = slice(ctd_prm['starts'][-1], ctd_prm['ends'][-1])  # last run
                # to dataframe ctd_add_lr here:
                sl_run_0 = slice(0, sl_run_0_in.stop - sl_run_0_in.start)
                sl_run_e = slice(sl_run_0.stop, sl_run_e_in.stop - sl_run_e_in.start + sl_run_0.stop)
                ctd_add_lr = pd.DataFrame({
                    'x': np.empty(sl_run_e.stop, np.float64),
                    'y': np.empty(sl_run_e.stop, np.float64),
                    'z': np.empty(sl_run_e.stop, np.float64), })
                for lim_name, sl_in, sl_out, sign in (
                        ('min', sl_run_0_in, sl_run_0, -1),
                        ('max', sl_run_e_in, sl_run_e, 1)):
                    for i_col, (col, ax) in enumerate(col2ax.items()):
                        shift = sign * cfg['output_files'][ax + '_resolution']
                        # update limits
                        lim[ax][lim_name] += shift
                        ctd_add_lr.iloc[sl_out, i_col] = (
                            ctd.iloc[sl_in, ctd.columns.get_loc(col)].values if ax != 'x' else lim[ax][lim_name]
                        )
                # Rezulting grid coordinates
                x = np.arange(lim['x']['min'], lim['x']['max'], cfg['x_resolution_use'])
                y = np.arange(lim['y']['min'], lim['y']['max'], cfg['y_resolution_use'])

                gdal_geotransform = (lim['x']['min'], cfg['x_resolution_use'], 0,
                                     -lim['y']['min'], 0, -cfg['y_resolution_use'])
                # [0] координата x верхнего левого угла
                # [1] ширина пиксела
                # [2] поворот, 0, если изображение ориентировано на север
                # [3] координата y верхнего левого угла
                # [4] поворот, 0, если изображение ориентировано на север
                # [5] высота пиксела

                # Grid blanking bot. edge: use hard filter to keep more of our beautiful contours
                y_edge_path = np.interp(x,
                                        edge_dist[ok_edge['hard']],
                                        edge_depth[ok_edge['hard']])
                xm, ym = np.meshgrid(x, y)
                edge_add = 3 * cfg[
                    'y_resolution_use']  # [m]. todo: not helps if no data. and better use polygon or result grid modification method
                ym_blank = ym > y_edge_path + edge_add  # to prevent stepping edge of blanking that reach useful data at high slope regions

                l.info('Gridding to {}x{} points'.format(*xm.shape))
                b_1st = True
                for iparam, z_col in enumerate(cfg['output_files']['data_columns']):  # z_col= u'Temp'
                    label_param = z_col  # .lstrip('_').split("_")[0]  # remove technical comments in names
                    i_z_col = ctd.columns.get_loc(z_col)
                    print(label_param, end=': ')
                    if z_col not in ctd:
                        print(' not loaded')
                        continue
                    if __debug__:
                        sys_stdout.flush()
                        i = 0

                    if False:  # z_col=='O2': #. Debug!!!
                        # remove bad runs
                        iruns_del = np.int32([18, 19])  # insert right run indexes of current section
                        l.warning('Removing ', z_col, ' runs:')
                        for st, en in zip(ctd_prm['starts'][iruns_del], ctd_prm['ends'][iruns_del]):
                            l.warning('{:%d %H:%M}'.format(timzone_view(pd.to_datetime(
                                ctd.time.iloc[st]), cfg['output_files']['dt_from_utc'])))
                            ctd.iloc[st:en, i_z_col] = np.NaN

                    # Adding repeated/interpolated data at edges to help gridding
                    # -----------------------------------------------------------
                    # Interpolate data along bottom edge and add them as source to griddata():
                    # - find last not nan value in each run
                    bGood = ctd[z_col].notna() & ok_ctd
                    ends_param, b_run_to_edge = runs_ilast_good(
                        bGood.values,
                        ctd_prm['starts'][ok_edge['soft']],
                        ctd_prm['ends'][ok_edge['soft']], ctd.depth.values,
                        cfg['y_resolution_use'] * 5,
                        edge_depth[ok_edge['soft']])
                    ctd_add_bt.loc[:, 'y'] = np.interp(ctd_add_bt.x, *ctd.iloc[
                        ends_param, ctd.columns.get_indexer(['dist', 'depth'])].values.T)
                    # Old code:
                    # ctd_add_bt.z= np.interp(ctd_add_bt.x, ctd.dist.iloc[ends_param].values,
                    #                  ctd.iloc[ends_param, i_z_col].values)
                    # It works, but not account for nonlinear z below y_s. Instead:
                    # Interpolate z along nearest vertical profile (by y) separately between profiles that reach the edge
                    ctd_add_bt.loc[:, 'z'] = np.NaN
                    # intervals indexes (0 - before the second, last - after the last but one)
                    i_add = ctd.dist.iloc[ends_param[1:-1]].searchsorted(ctd_add_bt.x)

                    def lin_squeeze(x_old, x0keep, x1old, x1new):
                        """
                        Linear squeeze x by moving x0 while keepeng the x1 the same
                        :param x_old: sequence to be scaled
                        :param x0keep: value of sequence that must remain the same after scaling
                        :param x1old: x point that need to shift to ``x1new``
                        :param x1new: required value of scaled sequence
                        :return: scaled sequence
                        """

                        given_old_range = x0keep - x1old
                        if given_old_range:
                            k = (x0keep - x1new) / given_old_range
                            return x1new + k * (x_old - x1old)
                        else:
                            return x_old

                    print('Add z-values on bot. edge from scaled bot. part of longer profiles. Found z shifts [m]:')
                    st_ends01 = (lambda se: np.column_stack((se[:-1, :], se[1:, :])))(
                        np.column_stack((ctd_prm['starts'][ok_edge['soft']][b_run_to_edge], ends_param)))
                    for i0, (ddist, (st0, en0, st1, en1)) in enumerate(zip(np.diff(edge_dist), st_ends01)):
                        # names numeric suffixes: 0 - current, 1 - next profile
                        #                         s - shorter (shallow), l - longer (deeper) profile
                        """
                        Trying the interpolation to be more hydrophysic: scaling the bottom part of longer profile 
                        (l-profile) to the depth where field value z equal to the last z of shorter one (figure):
                        projection of PE to SE, where P is found such that z(P) nealy equal to z(S) (with restrictions)
                                                       
                                                l     s                                 l     s
                        ~st_l+i_prior_y_l       _|...|-en_s, y_s, z_s
                                                 | ~/                                   P   S new
                        st_prior_z_l, y_l, ~z_s -|~/                    ->          ~z_s-|~/  min
                        en_l                    _|/                                     _|/
                                                                                          E
                                                     [y_s, y_e] to [st_prior_z_l, y_e]
                                                     
                        Then interp z from l-profile bottom part 
                        """
                        #
                        #
                        y0, y1 = ctd.depth.iloc[[en0, en1]].values
                        if y0 > y1:
                            st_l = st0
                            en_s, en_l = en1, en0
                            y_s, y_e = y1, y0
                        else:
                            st_l = st1
                            en_s, en_l = en0, en1
                            y_s, y_e = y0, y1

                        # Search y on l-profie that have same value as z_s = z[en_s]:
                        # restrict search by depth range from y_s (down, up) tuple:
                        max_ddepth = (y_e - y_s,  # if bottom level changes may be not need this restriction
                                      -(y_s - y_s / (1 + ddist))
                                      # in opposite direction: if ddist -> oo,|max_ddepth|-> y_s
                                      )
                        z_s = ctd[z_col].iat[en_s]  # todo: filter bad z. How?
                        z_l2s = ctd.iloc[st_l:en_l, i_z_col].values - z_s  # z diffeerences of longer profile to z_s
                        # index on l-profile with ~ same depth as y_s
                        i_prior_y_l = min(ctd.depth.iloc[st_l:en_l].searchsorted(y_s), z_l2s.size - 1)

                        # Search z on l-profile that nealy equal to the last z on shallower profile
                        # - method 1. search with account in nearest depth of same z - where its diff. to z_s change sign.
                        # Method fails if no z_l2s inversion and not robust if noise. If it fails next metod applied.
                        y_l2s01 = np.array([np.inf, np.inf])
                        st_prior_z_l01 = [0, 0]
                        sl_stop = [np.NaN, np.NaN]
                        for up, sl in ((0, slice(i_prior_y_l, z_l2s.size - 1)),
                                       (1, slice(i_prior_y_l, 0, -1))):  # search down: up = 0, search up: up = 1
                            opposite = -1 if up else 1
                            sl_stop[up] = sl.start + opposite * (
                                (opposite * ctd.depth.iloc[st_l:en_l][sl]).searchsorted(
                                    opposite * (y_s + max_ddepth[up])))
                            # if sl_stop[up] >= sl.start + len(z_l2s):
                            #     sl_stop[up] = sl.start + len(z_l2s) - 1

                            # search index of best z in slice:
                            sl = slice(sl.start, sl_stop[up], sl.step)
                            try:
                                i_sl = opposite * np.flatnonzero(z_l2s[sl] > 0
                                                                 if z_l2s[i_prior_y_l] < 0 else
                                                                 z_l2s[sl] < 0)[0]
                            except IndexError:
                                continue
                            st_prior_z_l01[up] = i_prior_y_l + st_l + i_sl  # absolute index
                            y_l = ctd.iloc[st_prior_z_l01[up], ctd.columns.get_loc('depth')]  # current best y
                            y_l2s01[up] = y_l - y_s  # shifts from y_s

                        def dy_check(st_prior_z_l):
                            """

                            :param st_prior_z_l: absolute index of best dz
                            :return: (depth at , accept)
                            """
                            y_l = ctd.depth.iat[st_prior_z_l]
                            y_l2s = y_l - y_s
                            print(f'{y_l2s:.1f} ', end='')
                            b_fail = not (max_ddepth[1] < y_l2s < max_ddepth[0])
                            if b_fail:
                                y_l = y_s  # or try # 2 if here after # 1
                                print(
                                    f'- not use so big. dz = {z_l2s[st_prior_z_l - st_l]:g} -> {z_l2s[i_prior_y_l]:g})')
                            else:
                                print(end=', ')
                            return y_l, b_fail

                        b_fail = not np.any(np.isfinite(y_l2s01))
                        if not b_fail:
                            print(f'm1: ', end='')
                            # index on l-profile with ~ same value as z_s
                            st_prior_z_l = st_prior_z_l01[np.argmin(np.abs(y_l2s01))]
                            y_l, b_fail = dy_check(st_prior_z_l)
                        if b_fail:
                            # - method 2. search everywhere using argmin() if method 1 is failed
                            # - need sort by depth instead to find nearest
                            print(f'm2: ', end='')
                            st_prior_z_l = st_l + sl_stop[1] + np.argmin(np.abs(z_l2s[slice(*reversed(sl_stop))]))
                            y_l, b_fail = dy_check(st_prior_z_l)

                        b_add_cur = i0 == i_add
                        # 1. scale z so that z(P) = z(S) exactly
                        # 2. project (scale) y
                        # 3. find (interp) scaled z on projected y
                        ctd_add_bt.z[b_add_cur] = np.interp(
                            lin_squeeze(ctd_add_bt.y[b_add_cur].values, y_e, y_s, y_l),  # scale bottom edge height
                            ctd.iloc[st_prior_z_l:en_l, ctd.columns.get_loc('depth')].values,
                            lin_squeeze(ctd.iloc[st_prior_z_l:en_l, i_z_col].values, ctd.iloc[en_l, i_z_col],
                                        ctd.iloc[st_prior_z_l, i_z_col], z_s)
                            )
                        pass
                    for sl_in, sl_out in ((sl_run_0_in, sl_run_0),
                                          (sl_run_e_in, sl_run_e)):
                        ctd_add_lr.iloc[sl_out, ctd_add_lr.columns.get_loc('z')] = ctd.iloc[sl_in, i_z_col].values

                    ctd_add1 = ctd_add_bt[ctd_add_bt.y.notna() & ctd_add_bt.z.notna()]
                    #  todo: y= ctd_add1.y + 5*cfg['output_files']['y_resolution'] - not sufficient but if multiply > 20
                    #  then fill better but this significant affect covature so: shift x such that new data will
                    #  below previous only.
                    #  ctd_add1.x + np.where(ctd_add1['y'].diff() > 0, 1, -1) * cfg['output_files']['y_resolution']
                    ctd_with_adds = pd.concat([
                        ctd.loc[bGood, ['dist', 'depth', z_col]].rename(
                            {**col2ax, z_col: 'z'}, axis='columns', copy=False),
                        ctd_add1,
                        ctd_add1.assign(y=ctd_add1.y + 5 * cfg['output_files']['y_resolution']),
                        ctd_add_lr[pd.concat([bGood[sl_run_0_in], bGood[sl_run_e_in]], ignore_index=True, copy=False)]
                        ], ignore_index=True, copy=False)
                    """
                    griddata_by_surfer(ctd, outFnoE_pattern=os_path.join(
                                cfg['output_files']['path'], 'surfer', stem_time + '{}'),  ),
                                       xCol='Dist', yCol='Pres',
                                       zCols=z_col, NumCols=y.size, NumRows=x.size,
                                       xMin=lim['x']['min'], xMax=lim['x']['max'], yMin=lim['y']['min'], yMax=lim['y']['max'])
                    """
                    write_grd_this_geotransform = write_grd_fun(gdal_geotransform)
                    for interp_method, interp_method_subdir in [['linear', ''], ['cubic', 'cubic\\']]:
                        dir_interp_method = cfg['output_files']['path'] / interp_method_subdir
                        dir_create_if_need(dir_interp_method)
                        # may be very long! : try extent=>map_extent?
                        z = interpolate.griddata(points=ctd_with_adds[['x', 'y']].values,
                                                 values=ctd_with_adds.z.values,
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
                                im = plt.imshow(z, extent=[x[0], x[-1], lim['y']['min'], lim['y']['max']],
                                                origin='lower')  # , extent=(0,1,0,1)
                                ax.invert_yaxis()
                                # np.seterr(divide='ignore', invalid='ignore')
                                # contour.py:370: RuntimeWarning: invalid value encountered in true_divide
                                CS = plt.contour(xm, ym, z, 6, colors='k')
                                plt.clabel(CS, fmt='%g', fontsize=9, inline=1)
                                if have_bed:
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
                        fileOut_grd = cfg['output_files']['path'] / interp_method_subdir / (
                                stem_time + label_param + '.grd')
                        write_grd_this_geotransform(fileOut_grd, z)
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
# dfNpoints= storeIn.select(cfg['in']['table_nav'], where= Nind, columns=['Lat', 'Lon', 'DepEcho'])
"""
