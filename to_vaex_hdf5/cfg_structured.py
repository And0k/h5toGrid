# !/usr/bin/env python3
# coding:utf-8
"""
  Author:  Andrey Korzh <ao.korzh@gmail.com>
  Purpose:
  Created: 14.09.2020
  Modified: 14.09.2020
"""

from typing import Any, Callable, Optional, Dict, List, Tuple, Union

from dataclasses import dataclass, field
from omegaconf import MISSING  # Do not confuse with dataclass.MISSING
import hydra
from hydra.core.config_store import ConfigStore




@dataclass
class ConfigIn:
    """
    "in": all about input files:

    :param path: path to source file(s) to parse. Use patterns in Unix shell style
    :param b_search_in_subdirs: search in subdirectories, used if mask or only dir in path (not full path)
    :param exclude_dirs_ends_with_list: exclude dirs which ends with this srings. This and next option especially useful when search recursively in many dirs
    :param exclude_files_ends_with_list: exclude files which ends with this srings
    :param b_skip_if_up_to_date: exclude processing of files with same name and which time change is not bigger than recorded in database (only prints ">" if detected). If finds updated version of same file then deletes all data which corresponds old file and after it brfore procesing of next files
    :param dt_from_utc_seconds: add this correction to loading datetime data. Can use other suffixes instead of "seconds"
    :param dt_from_utc_hours: add this correction to loading datetime data. Can use other suffixes instead of "hours"
    :param fs_float: sampling frequency, uses this value to calculate intermediate time values between time changed values (if same time is assined to consecutive data)
    :param fs_old_method_float: sampling frequency, same as ``fs_float``, but courses the program to use other method. If smaller than mean data frequency then part of data can be deleted!(?)
    :param header: comma separated list matched to input data columns to name variables. Can contain type suffix i.e.
     (float) - which is default, (text) - also to convert by specific converter, or (time) - for ISO format only
    :param cols_load_list: comma separated list of names from header to be saved in hdf5 store. Do not use "/" char, or type suffixes like in ``header`` for them. Defaut - all columns
    :param cols_not_use_list: comma separated list of names from header to not be saved in hdf5 store
    :param skiprows_integer: skip rows from top. Use 1 to skip one line of header
    :param b_raise_on_err: if False then not rise error on rows which can not be loaded (only shows warning). Try set "comments" argument to skip them without warning
    :param delimiter_chars: parameter of pandas.read_csv()
    :param max_text_width: maximum length of text fields (specified by "(text)" in header) for dtype in numpy loadtxt
    :param chunksize_percent_float: percent of 1st file length to set up hdf5 store tabe chunk size
    :param blocksize_int: bytes, chunk size for loading and processing csv
    :param b_make_time_inc: if time not sorted then modify time values trying to affect small number of values. This is different from sorting rows which is performed at last step after the checking table in database
    :param fun_date_from_filename: function(file_stem: str, century: Optional[str]=None) -> Any[compartible to input of pandas.to_datetime()]: to get date from filename to time column in it.

    :param csv_specific_param_dict: not default parameters for function in csv_specific_proc.py used to load data
    """
    path: Any = '.'
    b_search_in_subdirs: bool = False
    exclude_dirs_ends_with: Tuple[str] = ('toDel', '-', 'bad', 'test', 'TEST')
    exclude_files_ends_with: Tuple[str] = ('coef.txt', '-.txt', 'test.txt')
    b_skip_if_up_to_date: bool = True
    dt_from_utc = 0
    skiprows = 1
    b_raise_on_err = True
    max_text_width = 1000
    blocksize_int = 20000000
    b_make_time_inc = True
    dir: Optional[str] = MISSING
    ext: Optional[str] = MISSING
    filemask: Optional[str] = MISSING
    paths: Optional[List[Any]] = field(default_factory= list)
    nfiles: Optional[int] = 0  # field(default=MISSING, init=False)
    raw_dir_words: Optional[List[str]] = field(default_factory= lambda: ['raw', 'source', 'WorkData', 'workData'])


@dataclass
class ConfigOut:
    """
    "out": all about output files:

    :param db_path: hdf5 store file path
    :param table: table name in hdf5 store to write data. If not specified then will be generated on base of path of input files. Note: "*" is used to write blocks in autonumbered locations (see dask to_hdf())
    :param b_insert_separator: insert NaNs row in table after each file data end
    :param b_use_old_temporary_tables: Warning! Set True only if temporary storage already have good data! If True and b_skip_if_up_to_date= True then not replace temporary storage with current storage before adding data to the temporary storage
    :param b_remove_duplicates: Set True if you see warnings about
    """
    db_path: Any = ''
    table: str = 'navigation'
    tables: List[str] = field(default_factory= list)
    tables_log: List[str] = field(default_factory=list)
    b_insert_separator: bool = True
    b_use_old_temporary_tables: bool = False
    b_remove_duplicates: bool = False
    b_skip_if_up_to_date: bool = True  # todo: link to ConfigIn
    db_path_temp: Any = MISSING
    b_overwrite: Optional[bool] = False
    db: Optional[Any] = False
    logfield_fileName_len: Optional[int] = 255
    chunksize: Optional[int] = MISSING
    db_dir: Optional[str] = MISSING  # todo remove from config
    db_base: Optional[str] = MISSING
    db_ext: Optional[str] = MISSING
    nfiles: Optional[int] = MISSING

@dataclass
class ConfigFilter:
    """
    "filter": filter all data based on min/max of parameters:

    :param min_dict: List with items in  "key:value" format. Sets to NaN data of ``key`` columns if it is below ``value``'). To filter time use ``date`` key
    :param max_dict: List with items in  "key:value" format. Sets to NaN data of ``key`` columns if it is above ``value``'). To filter time use ``date`` key
    :param b_bad_cols_in_file_name: find string "<Separator>no_<col1>[,<col2>]..." in file name. Here <Separator> is one of -_()[, and set all values of col1[, col2] to NaN
    """
    min: Dict[str, float] = MISSING  #field(default_factory= dict)
    max: Dict[str, float] = MISSING  #field(default_factory= dict)
    b_bad_cols_in_file_name: bool = False


@dataclass
class ConfigProgram:
    """

    "program": program behaviour:

    :param return_: one_of('<cfg_from_args>', '<gen_names_and_log>', '<end>')
        <cfg_from_args>: returns cfg based on input args only and exit,
        <gen_names_and_log>: execute init_input_cols() and also returns fun_proc_loaded function... - see main()
    :param log_,
    :param verbose_: one_of('CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'),
    """

    return_: str = '<end>'
    b_interact: bool = False
    log: str =''
    verbose: str = 'INFO'


defaults = [dict([item]) for item in {
    'input': 'nmea_files',  # Load the config "nmea_files" from the config group "input"
    'out': 'hdf5_vaex_files',  # Set as MISSING to require the user to specify a value on the command line.
    'filter': 'filter',
    'program': 'program',
    #'search_path': 'empty.yml' not works
     }.items()]


@dataclass
class Config:

    # this is unfortunately verbose due to @dataclass limitations
    defaults: List[Any] = field(default_factory=lambda: defaults)

    # Hydra will populate this field based on the defaults list
    input: Any = MISSING
    out: Any = MISSING
    filter: Any = MISSING
    program: Any = MISSING

    # Note the lack of defaults list here.
    # In this example it comes from config.yaml
    # search_path: str = MISSING  # not helps without defaults

# cs = ConfigStore.instance()
# cs.store(group='in', name='nmea_files', node=ConfigIn)
# cs.store(group='out', name='hdf5_vaex_files', node=ConfigOut)
# cs.store(group='filter', name='filter', node=ConfigFilter)
# cs.store(group='program', name='program', node=ConfigProgram)
# # Registering the Config class with the name 'config'.
# cs.store(name='cfg', node=Config)


# hydra.conf.HydraConf.hydra_logging = 'colorlog'  # if installed ("pip install hydra_colorlog --upgrade")
# hydra.conf.HydraConf.job_logging = 'colorlog'

