# !/usr/bin/env python3
# coding:utf-8
"""
  Author:  Andrey Korzh <ao.korzh@gmail.com>
  Purpose: Trying using Hydra
  Created: 15.09.2020
  Modified: 20.09.2020, not implemented!
"""

import sys
import logging
from pathlib import Path

from datetime import time, datetime, timedelta
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, List, Sequence, Tuple, Union
from dataclasses import dataclass, field
from omegaconf import OmegaConf, DictConfig, MISSING
import hydra
import numpy as np
import pandas as pd
# import vaex
import pynmea2

import to_vaex_hdf5.cfg_dataclasses

from utils2init import init_file_names, Ex_nothing_done, this_prog_basename, standard_error_info, LoggingStyleAdapter, \
    call_with_valid_kwargs

# from csv2h5_vaex import argparser_files, with_prog_config
from to_pandas_hdf5.h5toh5 import h5move_tables, h5index_sort, h5init, h5_dispenser_and_names_gen, h5_close
from to_pandas_hdf5.gpx2h5 import df_rename_cols, h5_sort_filt_append
from filters import b1spike
from to_pandas_hdf5.h5_dask_pandas import filter_local

lf = LoggingStyleAdapter(logging.getLogger(__name__))
enc = 'cp1251'  # conversion that will be used for not unicode friendly console logging
cfg = None


def load_nmea(file: Path, in_cols: Optional[Sequence[str]] = None, time_prev: Optional[datetime] = None
              ) -> pd.DataFrame:
    """
    NMEA to DataFrame convertor
    :param file: NMEA file Path
    :param in_cols: attributes to search in pynmea2 NMEASentences (corresponded to needed data in NMEA strings), 1st is
     index
    :param time_prev: time used only to recover bad date while no new date loaded
    :return: pandas DataFrame with in_cols columns (except columns for which no data found) with 'datetime' index
    """
    max_no_time_to_same_row = 15
    if in_cols is None:
        search_fields = ('latitude', 'longitude', 'spd_over_grnd', 'true_course', 'depth_meters', 'depth')
        # not includes index. Also Magnetic Variation ('mag_variation') may be useful for ADCP proc
        index = 'datetime'  # todo: also try 'timestamp' if 'datetime' not found, will requre to add date
    else:
        index, *search_fields = in_cols
    rows = []
    d = {}

    no_time_sentences = 0
    time_last = time_prev
    time_with_date_1st_parsed = None

    def do_if_no_time(no_time_sentences: int, t_prev: Union[datetime, None], d: Mapping, nmea_sentence=''):
        """
        What to do if no good timestamp in sentence
        :param no_time_sentences: counter of no good timestamp sentences
        :param t_prev: previous timestamp
        :param d: sentence data dict
        :return:
        """
        no_time_sentences += 1
        if no_time_sentences > max_no_time_to_same_row:     # too long no timestamp after previous
            if len(d) > 1 and index in d:                   # have timestamp and other data =>
                rows.append(d)                              # save previous loaded data
            if t_prev is not None:
                t_prev += timedelta(seconds=1)  # increase time. Мay be better delete, or better interpolate?
                d = {index: t_prev}             # begin new time row
                lf.warning(
                    '{:s} <- {}. New row timestamp = previous + 1s because too long ({:d} sentences) no new time',
                    str(nmea_sentence).encode(enc, errors='replace').decode(), t_prev, max_no_time_to_same_row
                    )
                no_time_sentences = 0
            else:
                d = {}
                lf.warning(
                    '{:s}. New row begin because too long ({:d} sentences) no any timestamp yet',
                    str(nmea_sentence).encode(enc, errors='replace').decode(), max_no_time_to_same_row
                    )
        else:
            lf.debug('{:s} - no time in sentence', str(nmea_sentence).encode(enc, errors='replace').decode())
        return no_time_sentences, t_prev, d


    with pynmea2.NMEAFile(file.open(mode='r', encoding='ascii', errors='replace')) as _h_nmea:
        while True:  # to continue on error
            try:
                for nmea_sentence in _h_nmea:  # best method because with _h_nmea.readline() need to check for EOF, nmea_sentences = _h_nmea.read() fails when tries to read all at once

                    # Begin row with new timestamp index (or append to existed if still no)
                    try:
                        time_last = getattr(nmea_sentence, index)
                        b_time_parsed = True
                    except (AttributeError, KeyError, TypeError):
                        # AttributeError/KeyError/TypeError if bad timestamp or
                        try:
                            b_time_parsed = isinstance(nmea_sentence.timestamp, time)  # good time
                            # bad date:
                            try:
                                bad_date = isinstance(nmea_sentence.datestamp, str) and not \
                                    getattr(nmea_sentence, 'datestamp', None)
                            except AttributeError:
                                bad_date = True

                            if b_time_parsed:
                                if bad_date:
                                    if time_last is None:  # need other date source
                                        date_in_str = input(f'Input date (in ISO format like 2022-12-20) for file {file}:')  # todo recover from file name
                                        try:
                                            time_last = datetime.combine(datetime.fromisoformat(date_in_str),
                                                                         nmea_sentence.timestamp)
                                            lf.info('{:s} <- {} - using input date', str(nmea_sentence).encode(
                                                enc, errors='replace').decode(), time_last.date())
                                        except ValueError as e:
                                            b_time_parsed = False
                                            lf.info('Error with input date: {}', standard_error_info(e))

                                    else:
                                        # Can recover datestamp using previous date
                                        time_cur = datetime.combine(time_last.date(), nmea_sentence.timestamp)
                                        if time_cur - time_last > timedelta(hours=1):  # message only if interval is big
                                            lf.info('{:s} <- {} - using previous date', str(nmea_sentence).encode(
                                                enc, errors='replace').decode(), time_last.date())
                                        time_last = time_cur
                                    if b_time_parsed:
                                        no_time_sentences = 0

                                else:  # worse than only bad date?
                                    b_time_parsed = False
                        except AttributeError:
                            b_time_parsed = False

                    if b_time_parsed:  # time_last parsed Ok
                        try:
                            if time_last != d[index]:
                                # # new time => new row
                                if time_last < d[index]:
                                    lf.warning('{:s} new time < previous: {} < {}', str(nmea_sentence).encode(
                                        enc, errors='replace').decode(), time_last.date(), d[index])
                                if len(d) > 1:          # accumulated data have not only timestamp =>
                                    rows.append(d)      # save loaded data
                                d = {index: time_last}  # begin new time row
                            else:
                                pass   # already saved  # todo: average data for same time or increase time resolution
                        except KeyError:  # 1st time (no 'datestamp' in d) =>
                            # continue/begin row and will try to append other fields before start next row
                            d[index] = time_last

                        if time_with_date_1st_parsed is None:
                            time_with_date_1st_parsed = time_last
                        no_time_sentences = 0
                    else:
                        # Can not recover datestamp
                        no_time_sentences, time_last, d = do_if_no_time(
                            no_time_sentences, time_last, d, nmea_sentence)


                    # Append other fields to row except index
                    for f in search_fields:
                        try:
                            d[f] = getattr(nmea_sentence, f)
                            pass
                        except AttributeError:
                            continue

            except pynmea2.ParseError:
                continue
            break
    # add last row to ``rows`` if not yet added
    try:
        if rows[-1][index] != d[index]:
            rows.append(d)
    except Exception:
        pass

    if not rows:  # from_records() gets strange output on empty input so we force returning empty dataframe
        return pd.DataFrame()
    df = pd.DataFrame.from_records(rows, index=[index], coerce_float=True)  # not converts strings to floats, but eliminates objects that we can not write to HDF5
    for col in df.columns[df.dtypes == 'object']:                           # convert strings to floats
        df[col] = pd.to_numeric(df[col], errors='coerce')

    if time_prev and time_with_date_1st_parsed:
        # If time_prev is far from this file data (> 1D todo: better condition) then use date of time_with_date_1st_parsed instead

        # check bad date in first df rows because of bad time_prev
        #df.index[df.index - pd.DatetimeIndex(time_prev) >]

        idate_1st = np.flatnonzero(df.index == time_with_date_1st_parsed)
        if idate_1st:
            idate_1st = idate_1st[0]
        else:
            return df
        if idate_1st > 0 and any(np.ediff1d(df.index[(idate_1st - 1):(idate_1st + 1)]) > np.timedelta64(1, 'D')):  # one value check
            lf.warning(
                'Replacing dates of 1st {} rows to {} from next rows: previous time {} provided for file is wrong',
                    idate_1st, time_prev.date(), time_with_date_1st_parsed.date()
                )
            df.index[:idate_1st] = df.index[:idate_1st].map(
                lambda t: datetime.combine(time_with_date_1st_parsed.date(), t.time())
                )
    return df


VERSION = '0.0.1'

# def cmdline_help_mod(version, info):
#     'nmea2h5 version {}'.format(version) + info
#
# def version():
#     """Show the version"""
#     return 'version {0}'.format(VERSION)


# @dataclass hydra_conf(hydra.conf.HydraConf):
#     run: field(default_factory=lambda: defaults)dir
#hydra.conf.HydraConf.run.dir = './outputs/${now:%Y-%m-%d}_${now:%H-%M-%S}'
hydra.output_subdir = 'cfg'
# hydra.conf.HydraConf.output_subdir = 'cfg'
# hydra.conf.HydraConf.run.dir = './outputs/${now:%Y-%m-%d}_${now:%H-%M-%S}'
# hydra.conf.HydraConf.hydra_logging = 'colorlog'
# hydra.conf.HydraConf.job_logging = 'colorlog'


@dataclass
class ConfigInNmeaFiles:
    path: str
    time_interval: List[str] = field(default_factory=lambda: ['2021-01-01T00:00:00', 'now'])  # UTC
    dt_from_utc_hours: int = 0

ConfigOut = to_vaex_hdf5.cfg_dataclasses.ConfigOut
# ConfigOut.b_insert_separator will be forced to True where time of new file start > 1D relative to previous time

ConfigFilterNav = to_vaex_hdf5.cfg_dataclasses.ConfigFilterNav
ConfigProgram = to_vaex_hdf5.cfg_dataclasses.ConfigProgram



cs_store_name = Path(__file__).stem  # 'nmea2h5'
cs, ConfigType = to_vaex_hdf5.cfg_dataclasses.hydra_cfg_store(f'base_{cs_store_name}', {
    'input': ['in_nmea_files'],  # Load the config "in_autofon" from the config group "input"
    'out': ['out'],  # Set as MISSING to require the user to specify a value on the command line.
    #'filter': ['filter'],
    'filter': ['filter_nav'],  # may be move fields to input?
    'program': ['program'],
    # 'search_path': 'empty.yml' not works
    },
    module=sys.modules[__name__]
    )



@hydra.main(config_name=cs_store_name, config_path="cfg")  # adds config store data/structure to :param config
def main(config: ConfigType):
    """
    ----------------------------
    Add data from CSV-like files
    to Pandas HDF5 store*.h5
    ----------------------------
    :param cfg: is a hydra required arg, not use when call
    :return:
    """
    global cfg
    cfg = to_vaex_hdf5.cfg_dataclasses.main_init(config, cs_store_name, __file__=None)
    cfg = to_vaex_hdf5.cfg_dataclasses.main_init_input_file(cfg, cs_store_name, in_file_field='path')
    do(cfg)


# def main_init(cfg: DictConfig) -> DictConfig:
#     """
#     Common startup initializer
#     :param cfg:
#     :return:
#     """
#     #global lf
#     # if cfg.search_path is not None:
#     #     override_path = hydra.utils.to_absolute_path(cfg.search_path)
#     #     override_conf = OmegaConf.load(override_path)
#     #     cfg = OmegaConf.merge(cfg, override_conf)
#
#     print("Working directory : {}".format(os.getcwd()))
#     print(OmegaConf.to_yaml(cfg))
#
#     # cfg = cfg_from_args(argparser_files(), **kwargs)
#     if not cfg.program.return_:
#         print('Can not initialise')
#         return cfg
#     elif cfg.program.return_ == '<cfg_from_args>':  # to help testing
#         return cfg
#
#     hydra.verbose = 1 if cfg.program.verbose == 'DEBUG' else 0   # made compatible to my old cfg
#
#     print('\n' + this_prog_basename(__file__), end=' started. ')
#     try:
#         cfg['in']['paths'], cfg['in']['nfiles'], cfg['in']['path'] = init_file_names(
#             **cfg.input, b_interact=cfg.program.b_interact)
#     except Ex_nothing_done as e:
#         print(e.message)
#         return ()
#
#     return cfg


def do(cfg: Mapping):
    """
    :param cfg: OmegaConf configuration
    :return:
    """
    h5init(cfg['in'], cfg['out'])
    # OmegaConf.update(cfg, "in", cfg.input, merge=False)  # error
    # to allow non primitive types (cfg.out['db']) and special words field names ('in'):
    #cfg = OmegaConf.to_container(cfg)

    cfg['filter']['min'] = OmegaConf.to_container(cfg['filter']['min'])
    cfg['filter']['max'] = OmegaConf.to_container(cfg['filter']['max'])
    cfg['filter']['min']['depth'] = cfg['filter']['min']['DepEcho']
    cfg['filter']['max']['depth'] = cfg['filter']['max']['DepEcho']
    cfg['filter']['min'] = {k: v for k, v in cfg['filter']['min'].items() if v is not MISSING}
    cfg['filter']['max'] = {k: v for k, v in cfg['filter']['max'].items() if v is not MISSING}

    cfg['filter']['min_date'], cfg['filter']['max_date'] = cfg['in']['time_interval']

    cfg['out']['table_log'] = f"{cfg['out']['table']}/logFiles"
    b_insert_separator_original = cfg['out']['b_insert_separator']
    in_cols = ('datetime', 'latitude', 'longitude', 'depth_meters', 'depth')
    out_cols = ('Time', 'Lat', 'Lon', 'DepEcho')
    out_dtypes = {'Lat': np.float64, 'Lon': np.float64, 'DepEcho': np.float16}
    time_prev = None
    df_list = []

    ## Main circle ############################################################
    try:
        for i1_file, file in h5_dispenser_and_names_gen(cfg['in'], cfg['out'],
                                                        b_close_at_end=False,
                                                        check_have_new_data=False
                                                        ):
            lf.info('{}. {}: ', i1_file, file.name)
            ## Loading data #############
            df = load_nmea(file, in_cols, time_prev)
            if df.empty:
                continue
            time_prev = df.index[-1]

            df_rename_cols(df, in_cols, out_cols)

            df = filter_local(df, cfg['filter'])

            if 'DepEcho' in df.columns and 'depth' in df.columns:
                df['DepEcho'] = df['DepEcho'].where(df['DepEcho'].isna(), df['depth'])
                df.drop(columns='depth', inplace=True)
            elif 'depth' in df.columns:
                df.rename(columns={'depth': 'DepEcho'}, inplace=True)

            msg_parts = ['Spikes deleted ']
            for col in ['Lat', 'Lon']:
                bad = b1spike(df[col].values, max_spike=0.1)
                n_bad = bad.sum()
                if n_bad:
                    df.loc[bad, col] = np.NaN
                    msg_parts += [f'{col}: {n_bad:d},']
            if len(msg_parts) > 1:
                msg_parts[-1] = msg_parts[-1][:-1]  # del last comma
                lf.info(' '.join(msg_parts))

            cols_absent = [col for col in out_cols[1:] if col not in df.columns]
            if cols_absent:
                lf.info('Empty cols: {} - filling with NaNs', cols_absent)
                for col in cols_absent:
                    df[col] = np.array(np.NaN, out_dtypes[col])
                if any(df.columns != out_cols[1:]):
                    df = df[list(out_cols[1:])]

            # make dataframes uniform and append
            df_list.append(df.astype(out_dtypes))

        lf.info('Filter (delete bad rows) and write')
        for i_next, df in enumerate(df_list, start=1):
            if i_next < len(df_list):
                time_next = df_list[i_next].index[0]
                cfg['out']['b_insert_separator'] = time_next - df.index[-1] > np.timedelta64(1, 'D')
            else:
                cfg['out']['b_insert_separator'] = b_insert_separator_original

            call_with_valid_kwargs(h5_sort_filt_append, df, **cfg, input=cfg['in'])
    finally:
        h5_close(cfg['out'])
    failed_storages = h5move_tables(cfg['out'], tbl_names=cfg['out']['tables_written'])
    print('Finishing...' if failed_storages else 'Ok.', end=' ')

    # Sort if have any processed data, else don't because ``ptprepack`` not closes hdf5 source if it not finds data
    if cfg['in'].get('time_prev'):

        cfg['out']['b_remove_duplicates'] = True
        h5index_sort(
            cfg['out'],
            out_storage_name=f"{cfg['out']['db_path'].stem}-resorted.h5",
            in_storages=failed_storages,
            tables=cfg['out']['tables_written']
            )

if __name__ == '__main__':
    main()


