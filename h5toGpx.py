"""
Extract data from Pandas HDF5 store*.h5 files to GPX
todo: calc preferred section directions and save updated gpx file with this indication
"""


import logging
import sys

from pathlib import Path
from gpxpy.geo import simplify_polyline as gpxpy_simplify_polyline
from gpxpy.gpx import GPX, GPXTrack, GPXTrackPoint, GPXTrackSegment, GPXWaypoint  # xml.etree.ElementTree as ET
import numpy as np
import pandas as pd


from to_pandas_hdf5.h5_dask_pandas import filterGlobal_minmax
from to_pandas_hdf5.h5toh5 import h5load_points, h5find_tables
# my
from utils2init import cfg_from_args, my_argparser_common_part, this_prog_basename, init_logging, Ex_nothing_done
from utils_time import timzone_view, pd_period_to_timedelta

if __name__ != '__main__':
    l = logging.getLogger(__name__)


def my_argparser():
    """
    Configuration parser options and its description
    """

    p = my_argparser_common_part({'description': 'Extract data from Pandas HDF5 '
                                                 'store*.h5 files to GPX'})  # 'gpx2h5.ini'

    s = p.add_argument_group('in', 'data from hdf5 store')
    s.add('--db_path', help='hdf5 store file path')  # '*.h5'
    s.add('--tables_log_list', help='hdf5 log files name', default='logFiles')
    s.add('--table_nav', default='navigation',
          help='table where to search coordinates. If empty then use data tables')

    s = p.add_argument_group('out', 'Output files: paths, formats... '
                             ' - not calculation intensive affecting parameters')
    s.add('--select_from_tablelog_ranges_index',
          help='if set to 0 (or -1) then use only 1 data point per log row and retrieve navigation at data points only (to calc dist) else if None then use all data for ranges specified in log rows and saves tracks (not points)')
    s.add('--gpx_names_funs_list', default='i+1',
          help='list of functions to name of tracks/waypoints, each item for each table. Use arguments of current indexes: i - waypoint. will be converted to string, to duplicates will be added letter in alphabet order. Functions arguments are i: row index, row: pandas series with fields: Index (datetime), Lat, Lon')
    s.add('--gpx_names_funs_cobined', default='gpx_names_funs[row.itbl](i)',
          help='tracks/waypoints names of combined gpx. Possibilites are the same as for gpx_names_funs_list item. Default function will keep combined values same as individual')
    s.add('--gpx_names_fun_format', default='{}',
          help='name\'s format to display gpx_names_funs_(list/combined) result')
    s.add('--gpx_symbols_list', default="'Diamond, Red','Triangle, Blue'",
          help='list of symbols supported by your gpx display program. Each item is for correspondig input table')

    s.add('--path', default='',
          help='directory to place output files')

    s = p.add_argument_group('process', 'calculation parameters')
    s.add_argument('--b_missed_coord_to_zeros',
                        help='out all points even if no coordinates, but replace them to zeros')
    s.add_argument('--simplify_tracks_error_m_float',
                        help='does Ramer-Douglas-Peucker algorithm for simplification of tracks if set')
    s.add('--dt_search_nav_tolerance_seconds', default='1',
               help='start interpolte nav when not found exact data time')
    s.add('--period_files', default='',
               help='pandas offset strings as D, 5D, H, ... (most useful: D), export data in intervals')
    s.add('--period_segments', default='',
               help='pandas offset strings as D, 5D, H, ... to divide track on segments')

    s = p.add_argument_group('program', 'Program behaviour')
    s.add_argument('--log', help='write log if path to existed file is specified')

    return p


def write_file(fileOutPN, xml, mode='w'):
    p = Path(fileOutPN).with_suffix('.gpx')
    try:
        h_file = p.open(mode)
    except FileNotFoundError as e:
        p.parent.mkdir(parents=False)
        h_file = p.open(mode)
    try:
        h_file.write(xml)
    except Exception as e:
        # l.error((e.msg if hasattr(e,'msg') else '') + msg_option)
        print(e.msg if hasattr(e, 'msg') else e)
    finally:
        h_file.close()


def gpx_track_create(gpx, gpx_obj_namef):
    '''
    Create tracks in our GPX
    :param gpx:
    :param gpx_obj_namef:
    :return:
    '''
    gpx_track = {}
    print('track name: ', gpx_obj_namef)
    gpx_track[gpx_obj_namef] = GPXTrack(name=gpx_obj_namef)
    gpx_track[gpx_obj_namef].name = gpx_obj_namef  # noticed that constructor not apply it
    gpx.tracks.append(gpx_track[gpx_obj_namef])
    # Create segment in this GPX track:
    return gpx_track


def gpx_save(gpx, gpx_obj_namef, cfg_proc, path_stem):
    if cfg_proc['b_missed_coord_to_zeros']:
        for p in gpx.walk(only_points=True):
            if p.latitude is None or p.longitude is None:
                p.latitude = '0'  # float('NaN') #0
                p.longitude = '0'  # float('NaN') #0
            # if bFilterCoord:
            # #if p_prev==p:
            # p.delete
            # p_prev= p

    if isinstance(gpx_obj_namef, str):
        gpx.description = gpx_obj_namef
    gpx.author_email = 'andrey.korzh@atlantic.ocean.ru'
    write_file(path_stem, gpx.to_xml())
    print(Path(path_stem).stem + '.gpx saved')


gpx_names_funs = None  # need when eval gpx_obj_namef()?


def save_to_gpx(nav_df: pd.DataFrame, fileOutPN, gpx_obj_namef=None, waypoint_symbf=None, cfg_proc=None, gpx=None):  #
    """
    Save navigation from dataframe to *.gpx file. track or waypoints.
    Generate waypoints names and selects symbols from cfg['out']['gpx_symbols'] based on current row in nav_df
    :param nav_df: DataFrame with fields:
        - Lat, Lon: if not empty
        - DepEcho: to add its data as elevation
        - itbl: if ``waypoint_symbf``
    :param fileOutPN:       *.gpx file full name without extension. Set None to not write (useful if gpx only needed)
    :param gpx_obj_namef:   str or fun(waypoint number). If None then we set it to fileOutPN.stem
    :param waypoint_symbf:  str or fun(nav_df record = row). If None saves track
    :param cfg_proc:
        'simplify_tracks_error_m'
        'dt_per_file'
        'b_missed_coord_to_zeros'
        period_segments or period_tracks: to split track by this in one file
    :param gpx: gpx object to update. If None (default) then will be created here, updated and saved
    :return: None
    """

    if nav_df.empty:
        l.warning('no data')
        return
    if gpx_obj_namef is None:
        gpx_obj_namef = Path(fileOutPN).stem
    if cfg_proc is None:
        cfg_proc = {'dt_per_file': None}
    elif 'dt_per_file' not in cfg_proc:
        cfg_proc['dt_per_file'] = None
    if gpx is None:
        gpx = GPX()

    if waypoint_symbf:
        # , fun_symbol= 'Waypoint', fun_name= str
        if isinstance(waypoint_symbf, str):
            s = waypoint_symbf
            waypoint_symbf = lambda x: s
        b_useDepEcho = 'DepEcho' in nav_df.columns and any(nav_df['DepEcho'])

        w_names = set()
        # w_name = None # same perpose for not all conditions but faster
        # nav_dft= nav_df.reset_index().set_index('itbl', drop=False, append=True) #, inplace=True
        # for t in range(nav_dft.itbl.min(), nav_dft.itbl.max()+1):  #.ptp() = -
        for t, nav_dft in nav_df.groupby('itbl'):  # .reset_index()
            for i, r in enumerate(nav_dft.itertuples()):  # .loc[t] name=None
                str_time_short = '{:%d %H:%M}'.format(r.Index.round('s').to_pydatetime())
                timeUTC = r.Index.round('s').tz_convert(None).to_pydatetime()
                str_time_long = '{:%d.%m.%y %H:%M:%S}'.format(timeUTC)
                name = gpx_obj_namef if isinstance(gpx_obj_namef, str) else gpx_obj_namef(i, r, t)

                # remove duplicates by add letter
                name_test_dup = name
                i_dup = 0
                while name_test_dup in w_names:  # name== w_name or :
                    name_test_dup = name + chr(97 + i_dup)  # chr(97) = 'a'
                    i_dup += 1
                else:
                    name = name_test_dup
                w_names.add(name)

                gpx_waypoint = GPXWaypoint(
                    latitude=r.Lat,
                    longitude=r.Lon,
                    time=timeUTC,
                    description=str_time_long,
                    comment=str_time_short,
                    name=name,
                    symbol=waypoint_symbf(r),
                    elevation=-r.DepEcho if b_useDepEcho and np.isfinite(
                        r.DepEcho) else None)  # , description=, type=, comment=
                # if not i_dup:
                #     w_name = name  # to check duplicates on next cycle

                gpx.waypoints.append(gpx_waypoint)
        if isinstance(gpx_obj_namef, str):
            gpx.description = gpx_obj_namef
        if fileOutPN:
            gpx.author_email = 'andrey.korzh@atlantic.ocean.ru'
            write_file(fileOutPN, gpx.to_xml())
    else:  # tracks

        # loc= np.zeros_like(nav_df.index, dtype= int)
        # Lat= np.zeros_like(nav_df.index, dtype= np.float64)
        # Lon= np.zeros_like(nav_df.index, dtype= np.float64)
        # T= np.zeros_like(nav_df.index, dtype= pd.Timedelta)

        b_have_depth = ('DepEcho' in nav_df.columns)
        #b_have_speed = ('Speed' in nav_df.columns)
        period_split = cfg_proc.get('period_segments') or cfg_proc.get('period_tracks')
        if period_split:
            period_split = pd_period_to_timedelta(period_split)
            t_intervals_start = pd.date_range(
                start=nav_df.index[0].normalize(),
                end=max(nav_df.index[-1],
                        nav_df.index[-1].normalize() + period_split),
                freq=period_split)[1:]  # make last t_interval_start >= all_data[-1]
            #format_time =
        else:
            t_intervals_start = nav_df.index[-1:]  # series with 1 last value
        t_interval_end = nav_df.index[0]
        n_intervals_without_data = 0
        part = 0
        nav_df = nav_df.tz_convert('utc', copy=False)
        Tprev = nav_df.index[0].to_pydatetime()
        Tcur = Tprev
        if not cfg_proc.get('period_tracks'):
            gpx_track = gpx_track_create(gpx, gpx_obj_namef)
        for t_interval_start in t_intervals_start:
            t_interval = slice(t_interval_end, t_interval_start)  # from previous last
            # USEtime = [[t_interval_end.isoformat(), t_interval_start.isoformat()]]
            nav_df_cur = nav_df.truncate(t_interval_end, t_interval_start, copy=True)
            t_interval_end = t_interval_start
            # load_interval
            if not len(nav_df_cur):
                print('empty interval')
                n_intervals_without_data += 1
                if n_intervals_without_data > 30:
                    print('30 intervals without data => think it is the end')
                    break
                continue
            gpx_segment = GPXTrackSegment()
            if cfg_proc.get('period_tracks'):
                fmt = '%y-%m-%d' if t_interval_start.second == 0 and t_interval_start.hour == 0 else '%y-%m-%d %H:%M'
                track_name = f'{gpx_obj_namef} {t_interval_start:{fmt}}'
                gpx_track = gpx_track_create(gpx, track_name)
                gpx_track[track_name].segments.append(gpx_segment)
            else:
                gpx_track[gpx_obj_namef].segments.append(gpx_segment)

            for i, r in enumerate(nav_df_cur.itertuples()):
                Tcur = r.Index.to_pydatetime()
                gpx_point = GPXTrackPoint(
                    latitude=r.Lat, longitude=r.Lon,
                    elevation=r.DepEcho if b_have_depth and not np.isnan(r.DepEcho) else None,
                    time=Tcur)  # , speed= speed_b, comment= Comment
                gpx_segment.points.append(gpx_point)
                # if i==1:
                # gpx.description= gpx_obj_namef
                # gpx.author_email= 'andrey.korzh@atlantic.ocean.ru'
                # gpxxml= gpx.to_xml()
                # tree = ET.parse(gpxxml)
                # root = tree.getroot()

            if cfg_proc.get('simplify_tracks_error_m'):
                try:
                    gpx_segment.points = gpxpy_simplify_polyline(gpx_segment.points,
                                                                 cfg_proc['simplify_tracks_error_m'])
                except RecursionError as e:
                    recursion_limit = sys.getrecursionlimit()
                    l.error('Check time in data! Why increasing old recursion limit (%s) is needed? Trying: x10...',
                            recursion_limit)
                    try:
                        sys.setrecursionlimit(recursion_limit * 10)
                        gpx_segment.points = gpxpy_simplify_polyline(gpx_segment.points,
                                                                     cfg_proc['simplify_tracks_error_m'])
                        l.warning('now track simplified successfuly')
                    except Exception as e:
                        l.exception('not succes. skip simplifying tracks', recursion_limit)

            if cfg_proc['dt_per_file'] and Tcur - Tprev > cfg_proc['dt_per_file']:  # save to next file
                part += 1
                if fileOutPN:
                    gpx_save(gpx, gpx_obj_namef, cfg_proc, f'{fileOutPN}part{part}')
                gpx_track = gpx_track_create(gpx, gpx_obj_namef)
                Tprev = Tcur
        if fileOutPN:
            gpx_save(gpx, gpx_obj_namef, cfg_proc, fileOutPN)

    return gpx
# ___________________________________________________________________________


def str_deg_minut_from_deg(Coord, strFormat, LetterPlus='', LetterMinus='-'):
    """
    Convert array of coordinates in (degrees) to (degrees and minutes) and
    format it. Returns formatted string
    Coord     - array of coordinates
    strFormat - format string, can be 'N{0:02.0f} {1:02.6f}' or 'E{0:02.0f} {1:02.6f}'
    (thear is bug for print numpy arrays - it is not fill preceding zeros)
    """
    deg = np.trunc(Coord)
    minut = np.abs(Coord - deg) * 60
    return np.array([strFormat.format(d, m, L) for L, d, m in zip(
        np.where(deg < 0, LetterMinus, LetterPlus), np.abs(deg), minut)])


def save_to_csv(nav, datetimeindex, filepath):
    """
    Save to tab delimited
    :param nav: shou have fields Lat, Lon
    :param datetimeindex
    :param filepath:
    :return: None
    """

    # Format coordinates for Mapsource GUI of waypoint inserting
    str_lat = str_deg_minut_from_deg(nav['Lat'], '{2:}{0:02.0f} {1:02.6f}', 'N', 'S')
    str_lon = str_deg_minut_from_deg(nav['Lon'], '{2:}{0:02.0f} {1:02.6f}', 'E', 'W')
    # print(str_lat.astype('O') + ', ' + str_lon.astype('O'))
    # strLatLon4mapsource= 'N{} {} E{} {}'.format(Lat_deg, Lat_minut, Lon_deg, Lon_minut)

    # - combine results to record array

    rnav_dt = np.dtype({'names': ['DateTime', 'strLat', 'strLon'], 'formats': ['M8[us]', '|S30', '|S30']})  # 'f8'
    datetimeindex = datetimeindex.round(pd.Timedelta(seconds=1)).tz_localize(None)  # corrupt but in file look better
    # rnav = fromarrays([datetimeindex, str_lat, str_lon], dtype=rnav_dt)  # .astype('M8[us]'

    rnav = pd.DataFrame(np.array([str_lat, str_lon]).T, columns=['strLat', 'strLon'],
                        index=datetimeindex)
    rnav.index.name = 'Date        Time        '
    rnav.to_csv(filepath, sep='\t')

    # if True:
    #     f = open(cfg['out']['path'].with_name(f'{tblD}.txt', 'w+');
    #     #np.savetxt(f, np.atleast_2d(rnav_dt.names), '%s', delimiter='\t') - not works
    #     header= rnav_dt.names
    #     rnav.view()
    #     np.savetxt(f, rnav, '%s\t%s %s')
    #     f.close()


def init_gpx_symbols_fun(cfg_out):
    """

    :param cfg_out: 'gpx_symbols_fun', 'gpx_symbols'
    :return:
    """
    # compile functions if defined in cfg or assign default
    if 'gpx_symbols_fun' in cfg_out:
        sym_index_fun = eval(compile("lambda row: {}".format(cfg_out['gpx_symbols_fun']), '', 'eval'))
        gpx_symbols = lambda row: cfg_out['gpx_symbols'][sym_index_fun(row)]
    else:
        gpx_symbols = lambda row: cfg_out['gpx_symbols'][row.itbl]
        sym_index_fun = None
    return gpx_symbols


def file_from_tblname(tblD: str, tbl_log: str) -> str:
    """

    :param tblD: data table name
    :param tbl_log: interval table name
    :return:
    """
    tblD_safe = tblD if tbl_log == 'logFiles' else tblD + '_' + tbl_log.replace('log', '')
    return tblD_safe.replace('/', '').replace('\\', '')


# ##############################################################################
def main(new_arg=None):
    cfg = cfg_from_args(my_argparser(), new_arg)
    if not cfg:
        return
    if new_arg == '<return_cfg>':  # to help testing
        return cfg
    l = init_logging('', cfg['program']['log'], cfg['program']['verbose'])
    if not cfg['out']['path'].is_absolute():
        # set path relative to cfg['in']['db_path']
        cfg['out']['path'] = cfg['in']['db_path'].with_name(str(cfg['out']['path']))

    l.warning('\n {}({}) is gonna save gpx to ..{} dir. '.format(
        this_prog_basename(__file__), cfg['in']['db_path'], cfg['out']['path'].parent))

    if cfg['out']['select_from_tablelog_ranges'] is None:
        gpx_symbols = None
    else:
        gpx_symbols = init_gpx_symbols_fun(cfg['out'])

    global gpx_names_funs  # Shortcat for cfg['out']['gpx_names_funs']

    # Load data #################################################################
    qstr_trange_pattern = "index>='{}' & index<='{}'"
    with pd.HDFStore(cfg['in']['db_path'], mode='r') as store:
        # Find tables by pattern
        if '*' in cfg['in']['tables'][0]:
            # if 'table_prefix' in cfg['in']
            pattern_tables = cfg['in']['tables'][0]
            cfg['in']['tables'] = h5find_tables(store, pattern_tables)
            len_tables = len(cfg['in']['tables'])
            msg = 'Found {} tables with pattern {}'.format(len_tables, pattern_tables)
            if len_tables:
                l.info(msg)
            else:
                raise Ex_nothing_done(msg + '!')

            gpx_names_funs = []
            for itbl in range(len(cfg['in']['tables'])):  # same fo each table
                gpx_names_funs.append(cfg['out']['gpx_names_funs'][0])
        else:  # fixed number of tables
            # initialise with defaults if need:
            gpx_names_funs = cfg['out']['gpx_names_funs']
            for itbl in range(len(gpx_names_funs), len(cfg['in']['tables'])):
                gpx_names_funs.append('i+1')
        dfs_rnav = []
        nav2add_cur = None
        tbl_names_all_shortened = []
        for itbl, tblD in enumerate(cfg['in']['tables']):
            print(itbl, '. ', tblD, end=': ', sep='')
            if cfg['in']['tables_log'][0]:
                tblL = tblD + '/' + cfg['in']['tables_log'][0]
                try:
                    dfL = store[tblL]
                except KeyError as e:
                    l.warning(' '.join([s for s in e.args if isinstance(s, str)]))
                    continue
            else:  # only for tables without log (usually no such tables)
                l.warning('configuration specifies to get data without use of "log..." tables')
                st_en = store[tblD].index[[0, -1]]
                if cfg['process']['period_files']:
                    t_intervals_start = pd.date_range(
                        start=st_en[0].normalize(),
                        end=max(st_en[-1], st_en[-1].normalize() + pd_period_to_timedelta(
                            cfg['process']['period_files'])),
                        freq=cfg['process']['period_files'])[1:]  # makes last t_interval_start >= all_data[-1]
                    dfL = pd.DataFrame.from_records({'DateEnd': t_intervals_start, 'fileName': tblD},
                                                    index=st_en[:1].append(t_intervals_start[:-1]))
                else:
                    dfL = pd.DataFrame.from_records({'DateEnd': st_en[-1], 'fileName': tblD}, index=st_en[:1])

            gpx_names_fun_str = "lambda i, row, t=0: {}.format({})".format(
                (
                    f"'{cfg['out']['gpx_names_fun_format']}'" if not cfg['out']['gpx_names_fun_format'].startswith("f'")
                    else cfg['out']['gpx_names_fun_format']
                ),
                gpx_names_funs[itbl])
            gpx_names_fun = eval(compile(gpx_names_fun_str, '', 'eval'))
            if cfg['out']['select_from_tablelog_ranges'] is None:
                # Use all data for ranges specified in log rows and saves tracks (not points)

                for irow, r in enumerate(dfL.itertuples()):  # iterrows()
                    qstr = qstr_trange_pattern.format(r.Index, r.DateEnd)
                    print(qstr, end='… ')
                    try:
                        dfD = store.select(cfg['in']['table_nav'
                                           ] if cfg['in']['table_nav'] else tblD, qstr,
                                           columns=['Lat', 'Lon', 'DepEcho'])
                    except Exception as e:
                        l.exception('Error when query:  {}. '.format(qstr))
                        # '\n==> '.join([s for s in e.args if isinstance(s, str)])))
                        continue
                    # Keep data with period = 1s only
                    dfD = dfD[~dfD.index.round(pd.Timedelta(seconds=1)).duplicated()]
                    # dfD.drop_duplicates(['Lat', 'Lon', 'index'])'

                    bGood = filterGlobal_minmax(dfD, dfD.index, cfg['filter'])
                    dfD = dfD[bGood]
                    # Add UTC time and table name to output file name
                    # Local time and table name to gpx object name
                    str_time_long = '{:%y%m%d_%H%M}'.format(r.Index)
                    r = r._replace(Index=timzone_view(r.Index, cfg['out']['dt_from_utc_in_comments']))
                    tblD_safe = file_from_tblname(tblD, cfg['in']['tables_log'][0])
                    try:
                        gpx_names_fun_result = gpx_names_fun(tblD_safe, r)  # '{:%y%m%d}'.format(timeLocal)
                    except TypeError as e:
                        raise TypeError('Can not evalute gpx_names_fun "{}"'.format(gpx_names_fun_str)).with_traceback(
                            e.__traceback__)

                    save_to_gpx(
                        dfD, cfg['out']['path'].with_name(f'{str_time_long}{tblD_safe}'),
                        gpx_obj_namef=gpx_names_fun_result, cfg_proc=cfg['process'])

                    if len(cfg['in']['tables']) > 1:
                        nav2add_cur = dfD if irow == 0 else nav2add_cur.append(dfD)
                if len(cfg['in']['tables']) > 1:
                    nav2add_cur = dfD.assign(itbl=itbl)

            else:
                # Use only 1 data point per log row

                if cfg['out']['select_from_tablelog_ranges'] != 0:
                    print('selecting from {} row index of log table'.format(
                        cfg['out']['select_from_tablelog_ranges']))

                try:
                    dfL.index = dfL.index.tz_convert('UTC')
                except TypeError as e:
                    print((e.msg if hasattr(e, 'msg') else str(e)) + '!\n- continue presume on UTC log index...')
                print(end='all log data ')
                time_points = (dfL.index if cfg['out']['select_from_tablelog_ranges'] == 0 else
                               dfL['DateEnd'] if cfg['out']['select_from_tablelog_ranges'] == -1 else
                               None)
                if time_points is None:
                    raise (ValueError("cfg['out']['select_from_tablelog_ranges'] must be 0 or -1"))
                cols_nav = ['Lat', 'Lon', 'DepEcho']
                nav2add = h5load_points(store, cfg['in']['table_nav'], cols_nav, time_points=time_points,
                                        dt_check_tolerance=cfg['process']['dt_search_nav_tolerance'],
                                        query_range_lims=(time_points[0], dfL['DateEnd'][-1]))[0]
                cols_nav = nav2add.columns  # not all columns may be loaded
                # Try get non NaN from dfL if it has needed columns (we used to write there edges' data with _st/_en suffixes)
                isna = nav2add.isna()
                dfL_col_suffix = 'st' if cfg['out']['select_from_tablelog_ranges'] == 0 else 'en'
                for col in cols_nav:
                    col_dat = f'{col}_{dfL_col_suffix}'
                    if isna[col].any() and  col_dat in dfL.columns:
                        b_use = isna[col].values & dfL[col_dat].notna().values
                        nav2add.loc[b_use, col] = dfL.loc[b_use, col_dat].values

                nav2add.index = timzone_view(nav2add.index, dt_from_utc=cfg['out']['dt_from_utc_in_comments'])
                # tz_local= tzoffset(None, cfg['out']['dt_from_utc_in_comments'].total_seconds())
                # if nav2add.index.tz is None:
                #     # think if time zone of tz-naive Timestamp is naive then it is UTC
                #     nav2add.index = nav2add.index.tz_localize('UTC')
                # nav2add.tz_convert(tz_local, copy= False)

                # Save to gpx waypoints
                nav2add_cur = nav2add.assign(itbl=itbl)

                # if 'gpx_names_funs' in cfg['out'] and \
                #     len(cfg['out']['gpx_names_funs'])>itbl:
                #
                #     gpx_names = eval(compile('lambda i: str({})'.format(
                #         cfg['out']['gpx_names_funs'][itbl]), [], 'eval'))
                #
                save_to_gpx(nav2add_cur,
                            cfg['out']['path'] / f"stations_{file_from_tblname(tblD, cfg['in']['tables_log'][0])}",
                            gpx_obj_namef=gpx_names_fun, waypoint_symbf=gpx_symbols,
                            cfg_proc=cfg['process']
                            )
                # save_to_csv(nav2add, dfL.index, cfg['out']['path'].with_name(f'nav{tblD}.txt'))
                if False:  # Show table info
                    store.get_storer(tblD).table

                    nodes = sorted(store.root.__members__)  # , key=number_key
                    print(nodes)
                    # store.get_node('CTD_Idronaut(Redas)').logFiles        # next level nodes

            # prepare saving of combined gpx
            if tbl_names_all_shortened:
                i_new = 0
                for c_prev, c_new in zip(tbl_names_all_shortened[-1], tblD):
                    if c_new == c_prev:
                        i_new += 1
                    else:
                        break
                tbl_names_all_shortened.append(tblD[i_new:])
            else:
                tbl_names_all_shortened.append(tblD)
            dfs_rnav.append(nav2add_cur)

        if len(cfg['in']['tables']) > 1 and cfg['out']['gpx_names_funs_cobined']:
            print('combined: ', end='')  # Save combined data to gpx
            df_rnav_combined = pd.concat(dfs_rnav)
            df_rnav_combined.sort_index(inplace=True)
            # Save to gpx waypoints
            if 'gpx_names_funs' in cfg['out']['gpx_names_funs_cobined']:
                gpx_names_funs = [  # row not used, it is here only for compatibility with tracks
                    eval(compile("lambda i: " + f, '', 'eval')) for f in gpx_names_funs]
            gpx_names_fun = eval(compile(
                "lambda i,row,t: '{gpx_names_fun_format}'.format({gpx_names_funs_cobined})".format_map(
                    cfg['out']), '', 'eval'))

            # gpx_symbols = lambda row: cfg['out']['gpx_symbols'][sym_index_fun(row)]

            # gpx_names = eval(compile("lambda i,row: '{gpx_names_fun_format}'.format({gpx_names_funs_cobined})".format_map(cfg['out']), '', 'eval'))
            # gpx_names = lambda i: str(i + 1)

            save_to_gpx(
                df_rnav_combined,
                cfg['out']['path'].with_name(
                    'all_' + file_from_tblname(','.join(tbl_names_all_shortened), cfg['in']['tables_log'][0])),
                gpx_obj_namef=gpx_names_fun, waypoint_symbf=gpx_symbols, cfg_proc=cfg['process'])
    print('Ok')


if __name__ == '__main__':
    main()

# trash
"""
                dfN = store[cfg['in']['table_nav']]

                # Check/correct table for next pandas processing commands requirements
                if not dfN.index.is_unique:
                    print('{} non unique index elements removing'.format(
                        len(dfN.index) - dfN.index.nunique()))  # dfN.index.get_duplicates()
                    dfN = dfN[~dfN.index.duplicated(keep='last')]

                dfN['timeNav'] = dfN.index #temorary to check reindex
                                if cfg['out']['select_from_tablelog_ranges']==0:
                    nav2add = dfN.reindex(dfL.index.tz_convert('UTC'), method='nearest',
                                          tolerance= cfg['process']['dt_search_nav_tolerance'])
                else:
                    raise NotImplementedError('only 0 carrently sapported')
                nav2add.index = nav2add['timeNav'].index.tz_convert(dfL.index.tz)

                #Nind = inearestsorted(Nav.index, CTD['time']) #other method

                # Check nearest search quality
                bBad = np.isnan(nav2add['Lat'])  # abs(dT) > pd.Timedelta(minutes=10)
                if np.any(bBad):
                    nBad = np.sum(bBad)
                    print('absent data near {} elements'.format(str(np.flatnonzero(bBad))))

                    # Check time difference between nav found and time of requested points
                    dT = nav2add.index - nav2add['timeNav']
                    print(pd.DataFrame({'dT': dT[bBad], 'File': dfL.fileName[bBad]}, columns=['dT', 'File'],
                                       index=dfL.index[bBad]))
                    #next, interpolate?
                del nav2add['timeNav'] #remove temoraries


"""
