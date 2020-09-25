# !/usr/bin/env python3
# coding:utf-8
"""
  Author:  Andrey Korzh <ao.korzh@gmail.com>
  Purpose: Trying using Hydra
  Created: 15.09.2020
  Modified: 20.09.2020
"""
import os
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, List, Sequence, Tuple, Union
from omegaconf import OmegaConf, DictConfig

import pandas as pd
# import vaex
import pynmea2

from to_vaex_hdf5.cfg_structured import *
from utils2init import init_file_names, Ex_nothing_done, this_prog_basename, standard_error_info, LoggingStyleAdapter

# from csv2h5_vaex import argparser_files, with_prog_config
from to_pandas_hdf5.csv2h5 import h5_dispenser_and_names_gen
from to_pandas_hdf5.h5toh5 import h5move_tables, h5index_sort, h5init
from to_pandas_hdf5.gpx2h5 import df_rename_cols, df_filter_and_save_to_h5

lf = LoggingStyleAdapter(logging.getLogger(__name__))


def load_nmea(file: Path, in_cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """
    NMEA to DataFrame convertor
    :param file: NMEA file Path
    :param in_cols: attributes to search in pynmea2 NMEASentences (corresponded to needed data in NMEA strings)
    :return: pandas DataFrame with in_cols columns (except columns for which no data found) with 'datetime' index
    """
    if in_cols is None:
        search_fields = ('latitude', 'longitude', 'spd_over_grnd', 'true_course','depth_meters')  # not includes index. Also Magnetic Variation ('mag_variation') may be useful for ADCP proc
        index = 'datetime'  # todo: also try 'timestamp' if 'datetime' not found, will requre to add date
    else:
        index, *search_fields = in_cols
    rows = []
    d = {}
    b_have_time = False
    with pynmea2.NMEAFile(file.open(mode='r')) as _f:
        while True:  # to continue on error
            try:
                for nmea_sentence in _f:  # best method because with _f.readline() need to check for EOF, nmea_sentences = _f.read() fails when tries to read all at once
                    try:
                        t = getattr(nmea_sentence, index)
                        if b_have_time:
                            rows.append(d)
                            d = {index: t}  # new time row begin
                        else:
                            d[index] = t
                            b_have_time = True
                    except AttributeError:
                        pass

                    for f in search_fields:  # all except index
                        try:
                            d[f] = getattr(nmea_sentence, f)
                        except AttributeError:
                            continue

            except pynmea2.ParseError:
                continue
            break
    if not rows:  # from_records() gets strange output on empty input so we force returning empty dataframe:
        return pd.DataFrame()
    df = pd.DataFrame.from_records(rows, index=[index], coerce_float=True)
    # arg columns=search_fields not works, coerce_float is needed because eliminates objects that we can not write to HDF5
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

cs = ConfigStore.instance()
cs.store(group='input', name='nmea_files', node=ConfigIn)
cs.store(group='out', name='hdf5_vaex_files', node=ConfigOut)
cs.store(group='filter', name='filter', node=ConfigFilter)
cs.store(group='program', name='program', node=ConfigProgram)
#cs.store(group='hydra', name='hydra', node=ConfigProgram)
# Registering the Config class with the name 'config'.
cs.store(name='cfg', node=Config)

#config_path = 'ini/nmea2h5.yml'
@hydra.main(config_name="cfg")
def main(cfg: DictConfig):
    """
    ----------------------------
    Add data from CSV-like files
    to Pandas HDF5 store*.h5
    ----------------------------
    :param cfg: is a hydra required arg, not use when call
    :return:
    """
    do(main_init(cfg))


def main_init(cfg: DictConfig) -> DictConfig:
    """
    Common startup initializer
    :param cfg:
    :return:
    """
    #global lf
    # if cfg.search_path is not None:
    #     override_path = hydra.utils.to_absolute_path(cfg.search_path)
    #     override_conf = OmegaConf.load(override_path)
    #     cfg = OmegaConf.merge(cfg, override_conf)

    print("Working directory : {}".format(os.getcwd()))
    print(OmegaConf.to_yaml(cfg))

    # cfg = cfg_from_args(argparser_files(), **kwargs)
    if not cfg.program.return_:
        print('Can not initialise')
        return cfg
    elif cfg.program.return_ == '<cfg_from_args>':  # to help testing
        return cfg

    hydra.verbose = 1 if cfg.program.verbose == 'DEBUG' else 0   # made compatible to my old cfg

    print('\n' + this_prog_basename(__file__), end=' started. ')
    try:
        init_file_names(cfg.input, cfg.program.b_interact)  # changes cfg.input
    except Ex_nothing_done as e:
        print(e.message)
        return ()

    return cfg


def do(cfg):
    """
    :param new_arg: list of strings, command line arguments

    Note: if new_arg=='<cfg_from_args>' returns cfg but it will be None if argument
     argv[1:] == '-h' or '-v' passed to this code
    argv[1] is cfgFile. It was used with cfg files:
        'csv2h5_nav_supervisor.ini'
        'csv2h5_IdrRedas.ini'
        'csv2h5_Idronaut.ini'

    :return:
    """
    h5init(cfg.input, cfg.out)
    # OmegaConf.update(cfg, "in", cfg.input, merge=False)  # error
    # to allow non primitive types (cfg.out['db']) and special words field names ('in'):
    cfg = OmegaConf.to_container(cfg)
    cfg['in'] = cfg.pop('input')

    cfg['in']['dt_from_utc'] = 0
    cfg['out']['table_log'] = f"{cfg['out']['table']}/logFiles"
    cfg['out']['b_insert_separator'] = False
    in_cols = ('datetime', 'latitude', 'longitude', 'depth_meters')
    out_cols = ('Time', 'Lat', 'Lon', 'DepEcho')
    ## Main circle ############################################################
    for i1_file, file in h5_dispenser_and_names_gen(cfg['in'], cfg['out']):
        lf.info('{}. {}: ', i1_file, file.name)
        # Loading data
        df = load_nmea(file, in_cols)
        df_rename_cols(df, in_cols, out_cols)
        lf.info('write: ')
        df_filter_and_save_to_h5(cfg['out'], cfg, df)
    failed_storages = h5move_tables(cfg['out'], tbl_names=cfg['out']['tables_have_wrote'])
    print('Finishing...' if failed_storages else 'Ok.', end=' ')
    if cfg['in'].get('time_last'):
        # if have any processed data that need to be sorted (not the case for the routes and waypoints), also needed because ``ptprepack`` not closes hdf5 source if it not finds data
        cfg['out']['b_remove_duplicates'] = True
        h5index_sort(cfg['out'], out_storage_name=cfg['out']['db_base'] + '-resorted.h5', in_storages=failed_storages,
                     tables=cfg['out']['tables_have_wrote'])

if __name__ == '__main__':
    main()


