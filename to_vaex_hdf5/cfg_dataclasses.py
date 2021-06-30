# !/usr/bin/env python3
# coding:utf-8
"""
  Author:  Andrey Korzh <ao.korzh@gmail.com>
  Purpose:
  Created: 14.09.2020
  Modified: 19.12.2020
"""

import os, sys
from typing import Any, Callable, Optional, Dict, List, Mapping, Sequence, Tuple, Union
from dataclasses import dataclass, field, make_dataclass
from omegaconf import OmegaConf, MISSING, MissingMandatoryValue  # Do not confuse with dataclass.MISSING
import hydra
from hydra.core.config_store import ConfigStore

from utils2init import this_prog_basename, ini2dict, Ex_nothing_done, init_file_names, standard_error_info, LoggingStyleAdapter

lf = LoggingStyleAdapter(__name__)

@dataclass
class ConfigInput:
    """
    "in": all about input files:

    :param path: path to source file(s) to parse. Use patterns in Unix shell style
    :param b_search_in_subdirs: search in subdirectories, used if mask or only dir in path (not full path)
    :param exclude_dirs_endswith_list: exclude dirs which ends with this srings. This and next option especially useful when search recursively in many dirs
    :param exclude_files_endswith_list: exclude files which ends with this srings
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
    :param sort: if time not sorted then modify time values trying to affect small number of values. This is different from sorting rows which is performed at last step after the checking table in database
    :param fun_date_from_filename: function(file_stem: str, century: Optional[str]=None) -> Any[compartible to input of pandas.to_datetime()]: to get date from filename to time column in it.

    :param csv_specific_param_dict: not default parameters for function in csv_specific_proc.py used to load data
    """
    path: Any = '.'
    b_search_in_subdirs: bool = False
    exclude_dirs_endswith: Tuple[str] = ('toDel', '-', 'bad', 'test', 'TEST')
    exclude_files_endswith: Tuple[str] = ('coef.txt', '-.txt', 'test.txt')
    b_skip_if_up_to_date: bool = True
    dt_from_utc = 0
    skiprows = 1
    b_raise_on_err = True
    max_text_width = 1000
    blocksize_int = 20000000
    sort = True
    dir: Optional[str] = MISSING
    ext: Optional[str] = MISSING
    filemask: Optional[str] = MISSING
    paths: Optional[List[Any]] = field(default_factory=list)
    nfiles: Optional[int] = 0  # field(default=MISSING, init=False)
    raw_dir_words: Optional[List[str]] = field(default_factory= lambda: ['raw', 'source', 'WorkData', 'workData'])



@dataclass
class ConfigInHdf5_Simple:
    """
    "in": all about input files:
    :param db_path: path to pytables hdf5 store to load data. May use patterns in Unix shell style
             default='.'
    :param tables_list: table name in hdf5 store to read data. If not specified then will be generated on base of path of input files
    :param tables_log: table name in hdf5 store to read data intervals. If not specified then will be "{}/logFiles" where {} will be replaced by current data table name
    :param dt_from_utc_hours: add this correction to loading datetime data. Can use other suffixes instead of "hours",
            default='0'
    :param b_skip_if_up_to_date: exclude processing of files with same name and which time change is not bigger than recorded in database (only prints ">" if detected). If finds updated version of same file then deletes all data which corresponds old file and after it brfore procesing of next files.
            default='True'
    """
    db_path: str = MISSING
    tables: List[str] = field(default_factory=lambda: ['.*'])  # field(default_factory=list)
    tables_log: List[str] = field(default_factory=list)
    dt_from_utc_hours = 0
    b_skip_if_up_to_date: bool = True


@dataclass
class ConfigInHdf5(ConfigInHdf5_Simple):
    """
    Same as ConfigInHdf5_Simple + specific (CTD and navigation) data properties:
    :param table_nav: table name in hdf5 store to add data from it to log table when in "find runs" mode. Use empty strng to not add
            default='navigation'
    :param b_temp_on_its90: When calc CTD parameters treat Temp have red on ITS-90 scale. (i.e. same as "temp90"),
            default='True'

    """

    query: Optional[str] = None
    table_nav: Optional[str] = 'navigation'
    b_temp_on_its90: bool = True

    # path_coef: Optional[str] = MISSING  # path to file with coefficients for processing of Neil Brown CTD data


@dataclass
class ConfigOutSimple:
    """
    "out": all about output files:

    :param db_path: hdf5 store file path
    :param table: table name in hdf5 store to write data. If not specified then will be generated on base of path of input files. Note: "*" is used to write blocks in autonumbered locations (see dask to_hdf())
    :param b_insert_separator: insert NaNs row in table after each file data end
    :param b_use_old_temporary_tables: Warning! Set True only if temporary storage already have good data! If True and b_skip_if_up_to_date= True then not replace temporary storage with current storage before adding data to the temporary storage
    :param b_remove_duplicates: Set True if you see warnings about
    """
    db_path: Any = ''
    tables: List[str] = field(default_factory=list)
    tables_log: List[str] = field(default_factory=lambda: ['{}/log'])
    b_use_old_temporary_tables: bool = False
    b_remove_duplicates: bool = False
    b_skip_if_up_to_date: bool = True  # todo: link to ConfigIn
    db_path_temp: Any = None
    b_overwrite: Optional[bool] = False
    db: Optional[Any] = None  # False?
    logfield_fileName_len: Optional[int] = 255
    chunksize: Optional[int] = None
    nfiles: Optional[int] = None


@dataclass
class ConfigOut(ConfigOutSimple):
    """
    "out": all about output files:

    :param table: table name in hdf5 store to write data. If not specified then will be generated on base of path of input files. Note: "*" is used to write blocks in autonumbered locations (see dask to_hdf())
    :param b_insert_separator: insert NaNs row in table after each file data end
    """
    table: str = 'navigation'
    tables_log: List[str] = field(default_factory=list)  # overwrited parent
    b_insert_separator: bool = True


@dataclass
class ConfigOutCsv:
    cols: Optional[Dict[str, str]] = field(default_factory=dict)
    cols_log: Optional[Dict[str, str]] = field(default_factory=dict)  # Dict[str, str] =
    text_path: Optional[str] = None
    text_date_format: Optional[str] = None
    text_float_format: Optional[str] = None
    file_name_fun: str = ''
    file_name_fun_log: str = ''
    sep: str = '\t'

ParamsCTDandNav = make_dataclass('ParamsCTDandNav', [  # 'Pres, Temp90, Cond, Sal, O2, O2ppm, Lat, Lon, SA, sigma0, depth, soundV'
    (p, Optional[float], MISSING) for p in 'Pres Temp Cond Sal SigmaTh O2sat O2ppm soundV'.split()
    ])


@dataclass
class ConfigFilter:
    """
    "filter": filter all data based on min/max of parameters:

    :param min_dict: List with items in  "key:value" format. Sets to NaN data of ``key`` columns if it is below ``value``'). To filter time use ``date`` key
    :param max_dict: List with items in  "key:value" format. Sets to NaN data of ``key`` columns if it is above ``value``'). To filter time use ``date`` key
    :param b_bad_cols_in_file_name: find string "<Separator>no_<col1>[,<col2>]..." in file name. Here <Separator> is one of -_()[, and set all values of col1[, col2] to NaN
    """
    #Optional[Dict[str, float]] = field(default_factory= dict) leads to .ConfigAttributeError/ConfigKeyError: Key 'Sal' is not in struct
    min: Optional[ParamsCTDandNav] = ParamsCTDandNav()
    max: Optional[ParamsCTDandNav] = ParamsCTDandNav()
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


def hydra_cfg_store(
        cs_store_name: str,
        cs_store_group_options: Mapping[str, Sequence[str]],
        module=sys.modules[__name__]  # to_vaex_hdf5.cfg_dataclasses
        ) -> Tuple[ConfigStore, object]:
    """
    Registering Structured config with defaults specified by dataclasses in ConfigStore
    :param cs_store_name: config name
    :param cs_store_group_options:
        - keys: config group names
        - values: list of str, - group option names used for:
          - Yaml config files stems
          - Dataclasses to use, - finds names constructed nealy like `Config{capwords(name)}` - must exist in `module`.
          - setting 1st values item as default option for a group
    :param module: module where to search Dataclasses names, default: current module
    :return: (cs, Config)
    cs: ConfigStore
    Config: configuration dataclass
    """

    # Config class (type of result configuration) with assigning defaults to 1st cs_store_group_options value
    defaults_list = [{group: names[0]} for group, names in cs_store_group_options.items()]
    Config = make_dataclass(
        'Config',
        [('defaults', List[Any], field(default_factory=lambda: defaults_list))] +
        [(group, Any, MISSING) for group in cs_store_group_options.keys()]
        )

    cs = ConfigStore.instance()

    # Registering groups schemas
    for group, names in cs_store_group_options.items():
        for name in names:
            class_name = ''.join(['Config'] + [s.title() for s in name.split('_')])
            last_char = name[-1]
            if last_char == '_':
                class_name += last_char
            try:
                cl = getattr(module, class_name)
                cs.store(name=name, node=cl, group=group)
            except Exception as err:
                raise TypeError(f'Error init "{name}" group option of class {class_name}') from err
    # Registering all groups
    cs.store(name=cs_store_name, node=Config)

    return cs, Config


def main_init_input_file(cfg_t, cs_store_name, in_file_field='db_path'):
    """
    - finds input files paths
    - renames cfg['input'] to cfg['in'] and fills its field 'cfgFile' to cs_store_name
    :param cfg_t:
    :param cs_store_name:
    :param in_file_field:
    :return:
    """
    cfg_in = cfg_t.pop('input')
    cfg_in['cfgFile'] = cs_store_name
    try:
        # with omegaconf.open_dict(cfg_in):
        cfg_in['paths'], cfg_in['nfiles'], cfg_in['path'] = init_file_names(
            **{**cfg_in, 'path': cfg_in[in_file_field]},
            b_interact=cfg_t['program']['b_interact']
            )
    except Ex_nothing_done as e:
        print(e.message)
        cfg_t['in'] = cfg_in
        return cfg_t
    except FileNotFoundError as e:  #
        print('Initialisation error:', e.message, 'Calling arguments:', sys.argv)
        raise

    cfg_t['in'] = cfg_in
    return cfg_t


def main_init(cfg, cs_store_name, __file__=None, ):
    """
    - prints parameters
    - prints message that program (__file__ or cs_store_name) started
    - converts cfg parameters to types according to its prefixes/suffixes names (see ini2dict())

    :param cfg:
    :param cs_store_name:
    :param __file__:
    :return:
    """

    # global lf
    # if cfg.search_path is not None:
    #     override_path = hydra.utils.to_absolute_path(cfg.search_path)
    #     override_conf = OmegaConf.load(override_path)
    #     cfg = OmegaConf.merge(cfg, override_conf)

    print("Working directory : {}".format(os.getcwd()))

    # print not empty / not False values # todo: print only if config changed instead
    print(OmegaConf.to_yaml({k0: {k1: v1 for k1, v1 in v0.items() if v1} for k0, v0 in cfg.items()}))

    # cfg = cfg_from_args(argparser_files(), **kwargs)
    if not cfg.program.return_:
        print('Can not initialise')
        return cfg
    elif cfg.program.return_ == '<cfg_from_args>':  # to help testing
        return cfg

    hydra.verbose = 1 if cfg.program.verbose == 'DEBUG' else 0  # made compatible to my old cfg

    print('\n' + this_prog_basename(__file__) if __file__ else cs_store_name, end=' started. ')
    try:
        cfg_t = ini2dict(cfg)  # fields named with type pre/suffixes are converted
    except MissingMandatoryValue as e:
        lf.error(standard_error_info(e))
        raise Ex_nothing_done()
    except Exception:
        lf.exception('startup error')

    # OmegaConf.update(cfg, "in", cfg.input, merge=False)  # error
    # to allow non primitive types (cfg.out['db']) and special words field names ('in'):
    # cfg = omegaconf.OmegaConf.to_container(cfg)
    return cfg_t


def main_call(
        cmd_line_list: Optional[List[str]],
        fun: Callable[[], Any]
        ) -> Dict:
    """
    Adds command line args, calls fun, then restores command line args. Replaces shortcut "in." in them to "input."
    :param cmd_line_list: command line args of hydra commands or config options selecting/overwriting.
    :param fun: function that uses command line args, usually called ``main``
    :return: ``main()``
    """

    sys_argv_save = sys.argv
    if cmd_line_list is not None:
        cmd_line_list_upd = sys.argv
        len_in_rep = len('in.')
        for c in cmd_line_list:
            if c.startswith('in.'):
                cmd_line_list_upd.append(f'input.{c[len_in_rep:]}')
            else:
                cmd_line_list_upd.append(c)
        sys.argv = cmd_line_list_upd

    # hydra.conf.HydraConf.run.dir = './outputs/${now:%Y-%m-%d}_${now:%H-%M-%S}'
    out = fun()
    sys.argv = sys_argv_save
    return out



# defaults = [dict([item]) for item in {
#     'input': 'nmea_files',  # Load the config "nmea_files" from the config group "input"
#     'out': 'hdf5_vaex_files',  # Set as MISSING to require the user to specify a value on the command line.
#     'filter': 'filter',
#     'program': 'program',
#     #'search_path': 'empty.yml' not works
#      }.items()]


# @dataclass
# class Config:
#
#     # this is unfortunately verbose due to @dataclass limitations
#     defaults: List[Any] = field(default_factory=lambda: defaults)
#
#     # Hydra will populate this field based on the defaults list
#     input: Any = MISSING
#     out: Any = MISSING
#     filter: Any = MISSING
#     program: Any = MISSING
#
#     # Note the lack of defaults list here.
#     # In this example it comes from config.yaml
#     # search_path: str = MISSING  # not helps without defaults

# cs = ConfigStore.instance()
# cs.store(group='in', name='nmea_files', node=ConfigIn)
# cs.store(group='out', name='hdf5_vaex_files', node=ConfigOut)
# cs.store(group='filter', name='filter', node=ConfigFilter)
# cs.store(group='program', name='program', node=ConfigProgram)
# # Registering the Config class with the name 'config'.
# cs.store(name='cfg', node=Config)


# hydra.conf.HydraConf.hydra_logging = 'colorlog'  # if installed ("pip install hydra_colorlog --upgrade")
# hydra.conf.HydraConf.job_logging = 'colorlog'

