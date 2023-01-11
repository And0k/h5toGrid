# !/usr/bin/env python3
# coding:utf-8
"""
  Author:  Andrey Korzh <ao.korzh@gmail.com>
  Purpose: Convert (multiple) csv and alike text files to vaex hdf5 store with
           addition of log table
  Created: 26.02.2016
  Modified: 20.12.2019
"""
import logging
import re
import warnings
from codecs import open
from collections import OrderedDict
from datetime import datetime
from functools import partial
from pathlib import Path, PurePath
from time import sleep
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import vaex

from dataclasses import dataclass
from omegaconf import OmegaConf, DictConfig
import hydra
from hydra.core.config_store import ConfigStore

from clize import run, converters, parameters, ArgumentError
from sigtools.wrappers import decorator

import pynmea2

from utils2init import init_file_names, Ex_nothing_done, set_field_if_no, cfg_from_args, my_argparser_common_part, \
    this_prog_basename, init_logging, standard_error_info, LoggingStyleAdapter
import utils_time_corr

#from csv2h5_vaex import argparser_files , with_prog_config
from to_pandas_hdf5.h5toh5 import h5_dispenser_and_names_gen

if __name__ == '__main__':
    lf = None  # see main(): lf = init_logging(logging, None, cfg['program']['log'], cfg['program']['verbose'])
else:
    lf = LoggingStyleAdapter(logging.getLogger(__name__))

VERSION = '0.0.1'

def cmdline_help_mod(version, info):
    'nmea2h5 version {}'.format(version) + info





# @dataclass
# class MySQLConfig:
#     host: str = "localhost"
#     port: int = 3306


@decorator
def config_in(
        wrapped, *args,  # keep for clize
        # "in" default parameters
        path='.',
        b_search_in_subdirs=False,
        exclude_dirs_endswith_list='toDel, -, bad, test, TEST',
        exclude_files_endswith_list='coef.txt, -.txt, test.txt',
        b_incremental_update=True,
        dt_from_utc_seconds=0,
        dt_from_utc_hours=0,
        skiprows_integer=1,
        on_bad_lines='error',
        max_text_width=1000,
        blocksize_int=20000000,
        sort=True,
        **kwargs        # keep for clize
        ):
    """
    "in": all about input files:

    :param path: path to source file(s) to parse. Use patterns in Unix shell style
    :param b_search_in_subdirs: search in subdirectories, used if mask or only dir in path (not full path)
    :param exclude_dirs_endswith_list: exclude dirs which ends with this srings. This and next option especially useful when search recursively in many dirs
    :param exclude_files_endswith_list: exclude files which ends with this srings
    :param b_incremental_update: exclude processing of files with same name and which time change is not bigger than recorded in database (only prints ">" if detected). If finds updated version of same file then deletes all data which corresponds old file and after it brfore procesing of next files
    :param dt_from_utc_seconds: add this correction to loading datetime data. Can use other suffixes instead of "seconds"
    :param dt_from_utc_hours: add this correction to loading datetime data. Can use other suffixes instead of "hours"
    :param fs_float: sampling frequency, uses this value to calculate intermediate time values between time changed values (if same time is assigned to consecutive data)
    :param fs_old_method_float: sampling frequency, same as ``fs_float``, but courses the program to use other method. If smaller than mean data frequency then part of data can be deleted!(?)
    :param header: comma separated list matched to input data columns to name variables. Can contain type suffix i.e.
     (float) - which is default, (text) - also to convert by specific converter, or (time) - for ISO format only
    :param cols_load_list: comma separated list of names from header to be saved in hdf5 store. Do not use "/" char, or type suffixes like in ``header`` for them. Defaut - all columns
    :param cols_not_save_list: comma separated list of names from header to not be saved in hdf5 store
    :param skiprows_integer: skip rows from top. Use 1 to skip one line of header
    :param b_raise_on_err: if False then not rise error on rows which can not be loaded (only shows warning). Try set "comment" argument to skip them without warning
    :param delimiter_chars: parameter of pandas.read_csv()
    :param max_text_width: maximum length of text fields (specified by "(text)" in header) for dtype in numpy loadtxt
    :param chunksize_percent_float: percent of 1st file length to set up hdf5 store tabe chunk size
    :param blocksize_int: bytes, chunk size for loading and processing csv
    :param sort: if time not sorted then modify time values trying to affect small number of values. This is different from sorting rows which is performed at last step after the checking table in database
    :param fun_date_from_filename: function(file_stem: str, century: Optional[str]=None) -> Any[compartible to input of pandas.to_datetime()]: to get date from filename to time column in it.

    :param csv_specific_param_dict: not default parameters for function in csv_specific_proc.py used to load data
    """
    kw_args = locals()
    if __debug__:
        del kw_args['kw_args']  # exist if debug
    return wrapped(*args, cfg_in=kw_args, **kwargs)


#def with_my_config(config_path='to_pandas_hdf5/csv2h5_ini/csv2h5_vaex.yml'):
@decorator
def config_out(
        wrapped, *args,  # keep for clize
        b_insert_separator=True,
        b_reuse_temporary_tables=False,
        b_remove_duplicates=False,
        **kwargs        # keep for clize
        ):
    """
    "out": all about output files:

    :param db_path: hdf5 store file path
    :param table: table name in hdf5 store to write data. If not specified then will be generated on base of path of input files. Note: "*" is used to write blocks in autonumbered locations (see dask to_hdf())
    :param b_insert_separator: insert NaNs row in table after each file data end
    :param b_reuse_temporary_tables: Warning! Set True only if temporary storage already have good data! If True and b_incremental_update= True then not replace temporary storage with current storage before adding data to the temporary storage
    :param b_remove_duplicates: Set True if you see warnings about
    """
    kw_args = locals()
    if __debug__:
        del kw_args['kw_args']  # exist if debug
    return wrapped(*args, cfg_out=kw_args, **kwargs)


@decorator
def config_filter(
        wrapped, *args,  # keep for clize
        min_dict=None,
        max_dict=None,
        b_bad_cols_in_file_name=False,
        **kwargs        # keep for clize
        ):
    """
    ----------------------------
    Add data from CSV-like files
    to Pandas HDF5 store*.h5
    ----------------------------

    "filter": filter all data based on min/max of parameters:

    :param min_dict: List with items in  "key:value" format. Sets to NaN data of ``key`` columns if it is below ``value``'). To filter time use ``date`` key
    :param max_dict: List with items in  "key:value" format. Sets to NaN data of ``key`` columns if it is above ``value``'). To filter time use ``date`` key
    :param b_bad_cols_in_file_name: find string "<Separator>no_<col1>[,<col2>]..." in file name. Here <Separator> is one of -_()[, and set all values of col1[, col2] to NaN
    """
    kw_args = locals()
    if __debug__:
        del kw_args['kw_args']  # exist if debug
    return wrapped(*args, cfg_filter=kw_args, **kwargs)


@decorator
def config_program(
        wrapped, *args,  # keep for clize
        return_: parameters.one_of('<cfg_from_args>', '<gen_names_and_log>', '<end>') = '<end>',
        b_interact=False,
        log='',
        verbose: parameters.one_of('CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET') = 'INFO',
        **kwargs        # keep for clize
        ):
    """
    ----------------------------
    Add data from CSV-like files
    to Pandas HDF5 store*.h5
    ----------------------------

    "program": program behaviour:

    :param return_: choices=[],
        <cfg_from_args>: returns cfg based on input args only and exit,
        <gen_names_and_log>: execute init_input_cols() and also returns fun_proc_loaded function... - see main()
    :param log_,
    :param verbose_,
    """
    kw_args = locals()
    if __debug__:
        del kw_args['kw_args']  # exist if debug
    return wrapped(*args, cfg_program=kw_args, **kwargs)



cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.



@config_in
@config_out
@config_filter
@config_program
def main(cfg_in, cfg_out, cfg_filter, cfg_program):  #

    cfg = {'in': cfg_in, 'out': cfg_out, 'filter': cfg_filter, 'program': cfg_program}
    try:
        # delete garbage added by locals()
        for k, v in cfg.items():
            del v['kw_args']
            del v['kwargs']
            del v['args']
            del v['wrapped']
    except KeyError:
        pass
    cs.store(name="cfg", node=cfg)

    hydra.conf.HydraConf.output_subdir = 'cfg'
    hydra.conf.HydraConf.run.dir = './outputs/${now:%Y-%m-%d}_${now:%H-%M-%S}'
    hydra.conf.HydraConf.hydra_logging = 'colorlog'
    hydra.conf.HydraConf.job_logging = 'colorlog'
    # - hydra.job_logging: colorlog  # if installed ("pip install hydra_colorlog --upgrade")
    # - hydra.hydra_logging: colorlog

    config_path = 'cfg/nmea2h5.yml'

    @hydra.main(config_name="cfg")
    def main_cfg(cfg: DictConfig):  # hydra required arg, not use when call
        """
        ----------------------------
        Add data from CSV-like files
        to Pandas HDF5 store*.h5
        ----------------------------
        """

        #print(OmegaConf.to_yaml(cfg))
        global lf
        # cfg = cfg_from_args(argparser_files(), **kwargs)
        if not cfg.program.return_:
            print('Can not initialise')
            return cfg
        elif cfg.program.return_ == '<cfg_from_args>':  # to help testing
            return cfg

        lf = LoggingStyleAdapter(init_logging(logging, None, cfg.program.log, cfg.rogram.verbose))
        print('\n' + this_prog_basename(__file__), end=' started. ')
        try:
            cfg['in']['paths'], cfg['in']['nfiles'], cfg['in']['path'] = init_file_names(
                **cfg['in'], b_interact=cfg['program']['b_interact'])
        except Ex_nothing_done as e:
            print(e.message)
            return ()

        return cfg


    do(main_cfg())


# @decorator
# def with_in_config(
#         wrapped,
#         fun_proc_loaded=None,
#         **kwargs):
#     """
#         :param fun_proc_loaded: function(df: Dataframe, cfg_in: Optional[Mapping[str, Any]] = None) -> Dataframe/DateTimeIndex: to update/calculate new parameters from loaded data  before filtering. If output is Dataframe then function should have meta_out attribute which is Callable[[np.dtype, Iterable[str], Mapping[str, dtype]], Dict[str, np.dtype]]
#     """
#
#     def wrap(**kwargs):
#         # Prepare loading and writing specific to format
#         kwargs['in']['fun_proc_loaded'] = get_fun_proc_loaded_converters(**kwargs['in'])
#         kwargs['in'] = init_input_cols(**kwargs['in'])
#
#         return wrapped(**kwargs)
#
#     return wrap


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
    ## Main circle ############################################################
    for i1_file, file in h5_dispenser_and_names_gen(cfg['in'], cfg['out']):
        lf.info('{}. {}: '.format(i1_file, file.name))
        # Loading data

        dfs = load_nmea(file, cfg)
        lf.info('write: ')



def load_nmea(file, cfg):
    with pynmea2.NMEAFile(file) as _f:
        nmea_sentences = _f.read()
        #nmea_strings = [_f.readline() for i in range(10)]
        #msg = pynmea2.ptarse('$SDDBT,665.4,f,202.8,M,110.9,F*06')
    return nmea_sentences



def version():
    """Show the version"""
    return 'version {0}'.format(VERSION)


if __name__ == '__main__':
    run(main, alt=version)


