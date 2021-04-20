# !/usr/bin/env python3
# coding:utf-8
"""
  Author:  Andrey Korzh <ao.korzh@gmail.com>
  Purpose: Load coordinates from autofon (http://www.autofon.ru/autofon/item/seplus) GPS trackers to GPX files and HDF5 pandas store
  Created: 08.04.2021
  Modified: 09.04.2021
"""
import sys
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, List, Sequence, Tuple, Union
from datetime import datetime, timedelta, timezone
import hydra
import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
import requests

import pyproj   # from geopy import Point, distance
from h5toGpx import save_to_gpx  # gpx_track_create
from to_pandas_hdf5.h5_dask_pandas import h5_append, filter_global_minmax, filter_local
from to_pandas_hdf5.h5toh5 import unzip_if_need

import to_vaex_hdf5.cfg_dataclasses
from utils2init import LoggingStyleAdapter, dir_create_if_need, FakeContextIfOpen, set_field_if_no, Ex_nothing_done, call_with_valid_kwargs

# from csv2h5_vaex import argparser_files, with_prog_config
from to_pandas_hdf5.csv2h5 import h5_dispenser_and_names_gen
from to_pandas_hdf5.h5toh5 import h5move_tables, h5index_sort, h5init
from to_pandas_hdf5.gpx2h5 import df_rename_cols, df_filter_and_save_to_h5

lf = LoggingStyleAdapter(logging.getLogger(__name__))

tables2mid = {
    'tr0': 221910,
    'tr2': 221912
    }
mid2tables = {v: k for k,v in tables2mid.items()}

cfg = {
    'in': {
        'tables': ['tr0'],  # '*',   # what to process
        'time_intervals': {
            'tr0': [pd.Timestamp('2021-04-08T12:00:00'), pd.Timestamp('2021-04-08T12:00:00')]},
        'dt_from_utc': timedelta(hours=3),
        # use already loaded coordinates instead of request:
        'path_local_xlsx': {
            'tr0': Path('d:\Work') /
                    r' - координаты - адреса выходов на связь - 09-04-2021 16-46-52 - 09-04-2021 16-46-52.xlsx'
            },
        'anchor': '[44.56905, 37.97308]',
        },
    'out': {
        'path': Path(r'd:\WorkData\BlackSea\210408_trackers\210408trackers.h5'),
        },
    'process':
        {
            'simplify_tracks_error_m': 0,
            'dt_per_file': timedelta(days=356),
            'b_missed_coord_to_zeros': False
            }
    }
# # Default (max) interval to load new data
# cur_time = datetime.now()
# time_interval_default = {
#     'start': int((cur_time - timedelta(days=100)).timestamp()),
#     # 1st data to load, will be corrected if already have some
#     'end': int(cur_time.timestamp())
#     }

def save2gpx(nav_df, track_name, tbl_name=None, path=None, process=None, gpx=None, dt_from_utc=None):
    """

    :param nav_df:
    :param track_name:
    :param tbl_name:
    :return: updated gpx
    """
    nav_df.index.name = 'Time'
    str_time_long = f'{nav_df.index[0]:%y%m%d_%H%M}'
    gpx = save_to_gpx(
        nav_df if dt_from_utc is None else nav_df.tz_convert(timezone(dt_from_utc)),
        None,
        gpx_obj_namef=track_name, cfg_proc=process, gpx=gpx)

    return save_to_gpx(
        pd.DataFrame({
            'Lat': process['anchor_coord'][0],
            'Lon': process['anchor_coord'][1],
            'DepEcho': process['anchor_depth'],
            'itbl': 0
            }, index=nav_df.index[[0]]),
        path.with_name(f"{str_time_long}{track_name}"),
        waypoint_symbf='Anchor',
        gpx_obj_namef='Anchor', cfg_proc=process, gpx=gpx)


def prepare_loading_xlsx_links_by_pandas():
    """xlsx hyperlink support to pandas, modified original https://github.com/pandas-dev/pandas/issues/13439"""

    if prepare_loading_xlsx_links_by_pandas.is_done:
        return()

    from pandas.io.excel._openpyxl import OpenpyxlReader
    from pandas._typing import FilePathOrBuffer, Scalar
    from openpyxl.cell.cell import TYPE_BOOL, TYPE_ERROR, TYPE_NUMERIC, TIME_TYPES

    def _convert_cell(self, cell, convert_float: bool) -> Scalar:

        # here we adding this hyperlink support:
        if cell.hyperlink and cell.hyperlink.target:
            return cell.hyperlink.target
            # just for example, you able to return both value and hyperlink,
            # comment return above and uncomment return below
            # btw this may hurt you on parsing values, if symbols "|||" in value or hyperlink.
            # return f'{cell.value}|||{cell.hyperlink.target}'
        # here starts original code, except for "if" became "elif"
        else:
            cell_type = cell.data_type
            if cell_type in TIME_TYPES:
                return cell.value
            elif cell_type == TYPE_ERROR:
                return np.nan
            elif cell_type == TYPE_BOOL:
                return bool(cell.value)
            elif cell.value is None:
                return ""  # compat with xlrd
            elif cell_type == TYPE_NUMERIC:
                # GH5394
                if convert_float:
                    val = int(cell.value)
                    if val == cell.value:
                        return val
                else:
                    return float(cell.value)

        return cell.value

    def load_workbook(self, filepath_or_buffer: FilePathOrBuffer):
        from openpyxl import load_workbook
        # had to change read_only to False:
        return load_workbook(filepath_or_buffer, read_only=False, data_only=True, keep_links=False)

    OpenpyxlReader._convert_cell = _convert_cell
    OpenpyxlReader.load_workbook = load_workbook
    prepare_loading_xlsx_links_by_pandas.is_done = True

prepare_loading_xlsx_links_by_pandas.is_done = False

    # --->

def loading(table, path_local_xlsx, time_interval, dt_from_utc, out, process):
    """
    input config:
    :param table: str
    :param path_local_xlsx:
    :param time_interval:
    :param dt_from_utc:

    :param out: dict, output config
    :param process: dict, processing config
    :return:
    """
    # if probes == '*':
    #     probes = mid2tables.keys()

        # h5_append(out, nav_df)

    nav_dfs = {}
    if path_local_xlsx:
        prepare_loading_xlsx_links_by_pandas()
        xls = pd.read_excel(path_local_xlsx, usecols='C:D', skiprows=4, index_col='Дата')
        nav_df = xls['Адрес / Координаты'].str.extract(r'[^=]+\=(?P<Lat>[^,]*),(?P<Lon>[^,]*)')
        nav_df.set_index((nav_df.index - dt_from_utc).tz_localize('utc'), inplace=True)
        for coord in ['Lat', 'Lon']:
            nav_df[coord] = pd.to_numeric(nav_df[coord], downcast='float', errors='coerce')

        b_good = nav_df.index > (time_interval[0] - dt_from_utc).tz_localize('utc')
        nav_df = nav_df[b_good]
    else:
        url = 'http://176.9.114.139:9002/jsonapi'
        key_pwd = 'key=d7f1c7a5e53f48a5b1cb0cf2247d93b6&pwd=ao.korzh@yandex.ru'
        mid = tables2mid[table]

        # Request and display last info
        r = requests.post(f'{url}/laststates/?{key_pwd}')
        if r.status_code != 200:
            print(r)
            exit()
        for d in r.json():
            if d['id'] != mid:
                continue
            tim_last_coord = pd.Timestamp(datetime.fromtimestamp(d['tscrd']), tz=timezone(dt_from_utc))
            lf.info(f'Last coordinates on server for #{{id}}: {tim_last_coord}, (N{{lat}}, E{{lng}})'.format_map(d))

        # Request and save new data

        if False: # this will obtain filtered data useful for display only
            r = requests.post(f'{url}/?{key_pwd}',
                              json=[{'mid': str(k), **time_interval} for k in tables2mid.keys()]
                              )
            if r.status_code != 200:
                print(r)
                exit()
            nav_df = pd.DataFrame.from_records(d['points'], index='ts') \
                .rename(columns={'lat': 'Lat', 'lng': 'Lon', 'v': 'Speed'})

        r = requests.post(
            f"{url}/messages/{mid}?{key_pwd}&fromdate={{}}&todate={{}}&minlevel=6".format(
                *[int(t.timestamp()) for t in time_interval]))  # &lang=en not works
        if r.status_code != 200:
            print(r)
            exit()
        g = r.json()[0]['items']
        gg = []
        for gi in g:
            try:
                if gi['s'] == 2155872256:
                    gg.append(gi['p'])
            except:
                print(g)
                continue
        # no error if have resent data already
        if len(gg)==0 and time_interval[1] - tim_last_coord.tz_convert('utc') < timedelta(minutes=1):
            return ()  # no new data
        assert len(gg)==1

        p = pd.DataFrame.from_records(gg[0], index='r', exclude='s')['p']
        nav_df = pd.DataFrame(
            np.float32(p.tolist()), #to_numpy(dtype=), #.
            index=pd.to_datetime(p.index, unit='s', utc=True, origin=dt_from_utc.total_seconds()),
            columns=['Lat', 'Lon', 'Speed']
            )
            # .convert_dtypes(
            # infer_objects=False,
            # convert_string=False,
            # convert_integer=True,
            # convert_boolean=False,
            # convert_floating=True
            # )
        nav_df.index.name = 'Time'
        # display last point with local time
        lf.info(f"Downloaded {len(nav_df)} points for #{mid}, last - "
                f"{nav_df.index[-1].tz_convert(timezone(dt_from_utc))}: "
                "{Lat:0.6f}N, {Lon:0.6f}E".format_map(nav_df.iloc[-1]))
    return nav_df

def saving(nav_dfs, path, process):
    for tbl, nav_df in nav_dfs.items():
        save2gpx(nav_df, tbl, path=path, process=process)

        # bin average data
        bins = ['1H', '5min']
        for bin in bins:
            save2gpx(nav_df.resample(bin).mean(), f'{tbl}_avg({bin})', path=path, process=process)

def proc(cfg=cfg):
    cur_time = datetime.now()
    time_interval = {
        'start': int((cur_time - timedelta(days=100)).timestamp()),
        # 1st data to load, will be corrected if already have some
        'end': int(cur_time.timestamp())
        }

    nav_dfs = []
    for tbl in cfg['in']['tables']:
        nav_dfs[tbl] = call_with_valid_kwargs(
            loading, time_interval=time_interval, out=cfg['out'], process=cfg['process'], **cfg['in']
            )
    saving(nav_dfs,
           path=cfg['out']['path'],
           process=cfg['process']
           )



class OpenHDF5(FakeContextIfOpen):
    """
    Context manager that do nothing if file is not str/PurePath or custom open function is None/False
    useful if instead file want use already opened file object
    Returns table names and opened store

    :param tables: tables names search pattern or sequence of table names
    :param tables_log: tables names for metadata of data in `tables`
    :param db_path:
    :param cfg_out: not used but kept for the requirement of h5_dispenser_and_names_gen() argument
    :return: iterator that returns (table name, coefficients)
    updates cfg_in['tables'] - sets to list of found tables in store
    """
    # will be filled by each table from cfg['in']['tables']

    def __init__(self, db_path, tables, tables_log, db=None):
        self.tables = tables
        self.tables_log = tables_log
        self.handle = super(OpenHDF5, self).__init__(
            lambda f: pd.HDFStore(f, mode='r'),
            file=db_path,
            opened_file_object=db)

    def __enter__(self):
        """
        :return: opened handle or :param file: from __init__ if not need open
        """
        self.handle = super(OpenHDF5, self).__enter__()
        return (
            self.handle,
            self.tables,
            self.tables_log or [f'{t}/log' for t in self.tables]
            )

hydra.output_subdir = 'cfg'
# hydra.conf.HydraConf.output_subdir = 'cfg'
# hydra.conf.HydraConf.run.dir = './outputs/${now:%Y-%m-%d}_${now:%H-%M-%S}'

cs_store_name = Path(__file__).stem
cs, ConfigType = to_vaex_hdf5.cfg_dataclasses.hydra_cfg_store(cs_store_name, {
    'input': ['in_autofon'],  # Load the config "in_hdf5" from the config group "input"
    'out': ['out'],  # Set as MISSING to require the user to specify a value on the command line.
    #'filter': ['filter'],
    'process': ['process'],
    'program': ['program'],
    # 'search_path': 'empty.yml' not works
    })
#cfg = {}

def dx_dy_dist_bearing(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    R = 6371000  # radius of the earth in m
    klon = np.cos((lat2+lat1)/2)
    dx  = R * klon*dlon
    dy = R * dlat

    d =  np.sqrt(dy**2 + dx**2)
    angle = np.arctan2(dlat, dlon)
    return np.column_stack((dx, dy, d, np.degrees(angle)))

    #haversine_
    # x = sin(dlon) * cos(lat2)
    # y = cos(lat1) * sin(lat2) - (sin(lat1) * cos(lat2) * cos(diffLong))
    #
    #
    # m = 6367000 * 2 * np.arcsin(np.sqrt(a))
    #
    # dx = np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)
    # dy =np.sin(dlon / 2.0)
    # a = np.sin(dlat / 2.0)


def proc_and_h5save(df, tbl, cfg_in, out, process, bin: Optional[str] = None, rolling_dt: Optional[str] = None):
    """
    Calculates displacement, bin average and saves to HDF5 and GPX
    :param process:
    :param df:
    :param tbl:
    :param cfg_in:
    :param out:
    :param bin:
    :param rolling_dt:
    :return:
    """

    if not any(df):
        return ()
    if bin is None:
        out['table'] = tbl   # for df_filter_and_save_to_h5()
    elif rolling_dt:
        out['table'] = f'{tbl}_avg({bin},mov={rolling_dt})'
        window_not_even = int(to_offset(bin) / to_offset(rolling_dt))
        if not (window_not_even % 2):
            window_not_even += 1

        df = df.resample(rolling_dt, origin='epoch').mean().rolling(
            window_not_even, center=True, win_type='gaussian', min_periods=1).mean(std=3)
    else:  # bin average data
        # need reaverage combined data to account for partial averaging in last saved / loaded 1st intervals
        out['table'] = f'{tbl}_avg({bin})'
        shift_to_mid = to_offset(bin) / 2
        df = df.resample(bin, offset=-shift_to_mid).mean()
        df.index += shift_to_mid


    out['log']['fileChangeTime'] = pd.Timestamp.now()  # :%y%m%d_%H%M%S
    out['log']['fileName'] = '{:%y%m%d_%H%M}-{:%m%d_%H%M}'.format(*df.index[[0, -1]])
    if bin:
        out['log']['fileName'] += f'bin{bin}'

    # compute forward and back azimuths, plus distance
    # course, azimuth2, distance = geod.inv(
    #     *df.loc[navp_d['indexs'][[0, -1]], ['Lon', 'Lat']].values.flat)  # lon0, lat0, lon1, lat1
    df.loc[:, ['dx', 'dy', 'dr', 'Vdir']] = dx_dy_dist_bearing(
        *process['anchor_coord'][::-1],
        *df[['Lon', 'Lat']].values.T
        )

    # Saving to HDF5
    df_filter_and_save_to_h5(df, input={**cfg_in, 'dt_from_utc': timedelta(0)}, out=out)

    return df


def h5_names_gen(cfg_in, cfg_out: Mapping[str, Any], **kwargs) -> Iterator[Path]:
    """
    """
    set_field_if_no(cfg_out, 'log', {})
    for tbl in cfg_out['tables']:
        cfg_out['log']['fileName'], cfg_out['log']['fileChangeTime'] = cfg_in['time_interval']
        try:
            yield tbl     # Traceback error line pointing here is wrong
        except GeneratorExit:
            print('Something wrong?')
            return


def h5move_and_sort(out):
    """
    Moves from temporary storage and sorts `tables_have_wrote` tables and clears this list
    :param out: fields:
        tables_have_wrote: set of tuples of str, table names
    :return:

    """
    failed_storages = h5move_tables(out, tbl_names=out['tables_have_wrote'])
    print('Finishing...' if failed_storages else 'Ok.', end=' ')
    # Sort if have any processed data, else don't because ``ptprepack`` not closes hdf5 source if it not finds data
    out['b_remove_duplicates'] = True
    h5index_sort(out, out_storage_name=f"{out['db_path'].stem}-resorted.h5",
                 in_storages=failed_storages, tables=out['tables_have_wrote'])
    out['tables_have_wrote'] = set()


@hydra.main(config_name=cs_store_name)  # adds config store cs_store_name data/structure to :param config
def main(config: ConfigType) -> None:
    """
    ----------------------------
    Save data tp GPX files and Pandas HDF5 store*.h5
    ----------------------------
    The store contains tables for each device and each device table contains log with metadata of recording sessions

    :param config: with fields:
    - in - mapping with fields:
      - tables_log: - log table name or pattern str for it: in pattern '{}' will be replaced by data table name
      - cols_good_data: -
      ['dt_from_utc', 'db', 'db_path', 'table_nav']
    - out - mapping with fields:
      - cols: can use i - data row number and i_log_row - log row number that is used to load data range
      - cols_log: can use i - log row number
      - text_date_format
      - file_name_fun, file_name_fun_log - {fun} part of "lambda rec_num, t_st, t_en: {fun}" string to compile function
      for name of data and log text files
      - sep

    """
    global cfg
    cfg = to_vaex_hdf5.cfg_dataclasses.main_init(config, cs_store_name)
    cfg_in = cfg.pop('input')
    cfg_in['cfgFile'] = cs_store_name
    cfg['in'] = cfg_in
    # try:
    #     cfg = to_vaex_hdf5.cfg_dataclasses.main_init_input_file(cfg, cs_store_name, )
    # except Ex_nothing_done:
    #     pass  # existed db is not mandatory

    out = cfg['out']
    h5init(cfg['in'], out)

    # geod = pyproj.Geod(ellps='WGS84')
    # dir_create_if_need(out['text_path'])
    time_interval_default = [pd.Timestamp(t, tz='utc') for t in cfg_in['time_interval']]
    ## Main circle ############################################################
    for i1_tbl, tbl in h5_dispenser_and_names_gen(cfg_in, out, fun_gen=h5_names_gen):
        lf.info('{}. {}: ', i1_tbl, tbl)
        # Check existed data
        time_interval = time_interval_default
        try:
            df_log = out['db'][out['tables_log'][0].format(tbl)]
            t_max_exist = df_log['DateEnd'].max()
            # do not get existed data
            if t_max_exist > time_interval_default[0]:
                time_interval[0] = t_max_exist
                lf.info('Continue downloading {} {:%y-%m-%d %H:%M:%S%Z} \u2013 {:%m-%d %H:%M:%S%Z} (from last saved)', tbl, *time_interval)
                time_interval[0] += timedelta(seconds=1)  # else 1st downloaded point will be same as the last we have
            elif t_max_exist:
                lf.warning('Continue downloading {} {:%y-%m-%d %H:%M:%S%Z} \u2013 {:%m-%d %H:%M:%S%Z}: Gap {} from last saved!', tbl, *time_interval, time_interval_default[0] - t_max_exist)
        except KeyError:  # No object named tr0/log in the file
            lf.info('Downloading {} {}', tbl, time_interval)  # may be was no data

        # Loading data from internet
        df_loaded = call_with_valid_kwargs(
            loading, table=tbl, **{**cfg_in, 'time_interval': time_interval},
            out=out, process=cfg['process']
            )

        # Write
        proc_and_h5save(df_loaded, tbl, cfg_in, out, cfg['process'])

    if any(unzip_if_need(out['tables_have_wrote'])):  # cfg_in.get('time_last'):
        out['b_del_temp_db'] = True  # to remove temp db: else we will need remove old tables
        h5move_and_sort(out)

        df_loaded = None
        for i1_tbl, tbl in h5_dispenser_and_names_gen(cfg_in, out, fun_gen=h5_names_gen):
            if df_loaded is None:  # do 1 time here:
                df_loaded = out['db'][tbl]  # now it contains concatenated current and previous data
                # Saving GPX
                save2gpx(df_loaded, out['table'], path=out['db_path'], process=cfg['process'],
                         dt_from_utc=cfg_in['dt_from_utc'])

            # Calculate averages and saving other tables to HDF5 and GPX
            bins = ['1H', '5min']
            for bin in bins:
                df = proc_and_h5save(df_loaded, tbl, cfg_in, out, cfg['process'], bin)
                save2gpx(df, out['table'], path=out['db_path'], process=cfg['process'],
                         dt_from_utc=cfg_in['dt_from_utc'])

            df = proc_and_h5save(df_loaded, tbl, cfg_in, out, cfg['process'], bin='1H', rolling_dt='10min')
            save2gpx(df, out['table'], path=out['db_path'], process=cfg['process'],
                     dt_from_utc=cfg_in['dt_from_utc'])
        if any(unzip_if_need(out['tables_have_wrote'])):  # cfg_in.get('time_last')
            # with pd.HDFStore(db_path_temp) as out['db']:
            #     for tbl in unzip_if_need(out['tables_have_wrote']):
            #         out['db'].remove(tbl)
            h5move_and_sort(out)

    print('Ok>', end=' ')


def main_call(cmd_line_list: Optional[List[str]] = None, fun: Callable[[Any], Any] = main) -> Dict:
    """
    Convert arguments to command line args with that calls fun. Then restores command line args
    :param cmd_line_list: command line args of hydra commands or config options selecting/overwriting

    :return: global cfg
    """

    sys_argv_save = sys.argv
    if cmd_line_list is not None:
        sys.argv += cmd_line_list

    # hydra.conf.HydraConf.run.dir = './outputs/${now:%Y-%m-%d}_${now:%H-%M-%S}'
    fun()
    sys.argv = sys_argv_save
    return cfg


if __name__ == '__main__':
    main()  # [f'--config-dir={Path(__file__).parent}'])


def pairwise(iterable):
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)


def call_example():
    """
    to run from IDE or from bat-file with parameters
    --- bat file ---
    call conda.bat activate py3.7x64h5togrid
    D: && cd D:\Work\_Python3\And0K\h5toGrid
    python -c "from to_vaex_hdf5.autofon_coord import call_example; call_example()"
    ----------------
    # python -m to_vaex_hdf5.autofon_coord.call_example() not works
    :return:
    """
    # from to_vaex_hdf5.h5tocsv import main_call as h5tocsv
    path_db = Path(r'd:\WorkData\BlackSea\210408_trackers\tr0\210408trackers.h5')
    device = ['tr0']  # 221912
    main_call([  # '='.join(k,v) for k,v in pairwise([   # ["2021-04-08T08:35:00", "2021-04-14T11:45:00"]'
        'input.time_interval=["2021-04-08T09:00:00", "now"]',   # UTC, max (will be loaded and updated what is absent)
        'input.dt_from_utc_hours=3',
        'process.anchor_coord=[44.56905, 37.97309]',
        'process.anchor_depth=20',
        'process.period_tracks="1D"',
        # 'process.period_segments="2H"', todo: not implemented to work if period_tracks is set
        f'out.db_path="{path_db}"',
        'out.tables=[{}]'.format(','.join([f'"{d}"' for d in device])),
        'out.tables_log=["{}/log"]',
        'out.b_insert_separator=False',
        # 'input.path_local_xlsx="{}"'.format({  # use already loaded coordinates instead of request:
        #     221910: Path('d:\Work') /
        #             r' - координаты - адреса выходов на связь - 09-04-2021 08-03-13 - 09-04-2021 08-03-13.xlsx'
        #     }),
        # 'input.time_intervals={221910: "2021-04-08T12:00:00"}',

        ])

    # conda activate py3.7x64h5togrid && D: && cd D:\Work\_Python3\And0K\h5toGrid && python -m to_vaex_hdf5.autofon_coord.call_example()

# r = requests.post(
#     'http://176.9.114.139/jsonapi/?key=5cee5887daa94dff88e51dc9bd6d16a2&pwd=0887219675',  # &a0ghzybxtr
#     json={'mid': 'ao', **time_interval})

# print()
# print(r.json())
#
# with open(r'd:\Downloads\py.html','wb') as f:
#     f.write(r.content)
#
# gpx = GPX.GPX()
# gpx_segment = GPX.GPXTrackSegment()
# gpx_track = gpx_track_create(gpx, gpx_obj_namef)
# gpx_track[gpx_obj_namef].segments.append(gpx_segment)
# gpx.description = contact_name_d
# gpx.author_email = 'andrey.korzh@atlantic.ocean.ru'
# return gpx
"""
    i_log_row_st = 0
    with OpenHDF5(cfg['in']['db_path'], cfg['in']['tables'], cfg['in']['tables_log']) as (store, tables, tables_log):
        # get last times from log or assign specified interval back from now

        # Loading
        time_intervals_dict = {}
        for tbl in cfg['in']['tables']:
            if tbl in tables:
                df_log = store.select(tables_log[tables.index(tbl)])
                _ = time_interval_default.copy()
                _['start'] = int(df_log[df_log.index[-1], 'DateEnd']).timestamp()
                time_intervals_dict[tbl] = _
            else:
                time_intervals_dict[tbl] = time_interval_default.copy()


            # mid = mid2tables[int()]
            # json = [{'mid': str(mid), **time_interval} for mid, time_interval in time_intervals_dict]
        cfg['in']['time_intervals'] = time_intervals_dict
        tbl_navs = loading(out=cfg['out'], process=cfg['process'], **cfg['in'])

        # saving(tbl_navs):

        for tbl, tbl_log in zip(tables, tables_log):
            # get last time from log
            if tbl_log:
                df_log = store.select(tbl_log,
                                      where=cfg['in']['query']
                                      )
                lf.info('Saving {} data files of ranges listed in {}', df_log.shape[0], tbl_log)

                df_log_csv = None  # order_cols(df_log, out['cols_log'])

                # df_log_csv = interp_vals(
                #     df_log_csv,
                #     df_search=None,
                #     #cols_good_data = None,
                #     db = store,
                #     dt_search_nav_tolerance = timedelta(minutes=2)
                #     )

                df_log_csv.to_csv(
                    out['text_path'] / out['file_name_fun_log'](i_log_row_st, df_log.index[0],
                                                                              df_log.DateEnd[-1]),
                    date_format=out['text_date_format'],
                    float_format=out['text_float_format'],
                    sep=out['sep']
                    )
            else:
                lf.info('{}: ', tbl)

        for i_log_row, log_row in enumerate(df_log.itertuples(),
                                            start=i_log_row_st):  # h5log_rows_gen(table_log=tbl_log, db=store, ):
            # Load data chunk that log_row describes
            print('.', end='')
            qstr = qstr_trange_pattern.format(log_row.Index, log_row.DateEnd)
            df_raw = store.select(tbl, qstr)
            df_raw['i_log_row'] = i_log_row
            df_csv = None  # order_cols(df_raw, out['cols'])
            # Save data
            df_csv.to_csv(
                out['text_path'] / out['file_name_fun'](i_log_row, df_raw.index[0], df_raw.index[-1]),
                date_format=out['text_date_format'],
                float_format=out['text_float_format'],
                sep=out['sep']
                )

        i_log_row_st += df_log.shape[0]
"""