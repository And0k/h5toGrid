# !/usr/bin/env python3
# coding:utf-8
"""
  Author:  Andrey Korzh <ao.korzh@gmail.com>
  Purpose: Export pandas hdf5 tables data to csv files (hole or using intervals specified in table/log table)
  Created: 15.09.2020
  See main() docstring
"""
import sys
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Iterable, Mapping, Optional, List, Tuple
from datetime import timedelta
from itertools import zip_longest

import omegaconf  #, OmegaConf DictConfig, MISSING, open_dict OmegaConf, DictConfig, MISSING, open_dict
import hydra
# from hydra.core.config_store import ConfigStore
import numpy as np
import pandas as pd
# import vaex

import cfg_dataclasses

from utils2init import LoggingStyleAdapter, dir_create_if_need, FakeContextIfOpen, set_field_if_no

# from csv2h5_vaex import argparser_files, with_prog_config

from to_pandas_hdf5 import h5
from to_pandas_hdf5.CTD_calc import get_runs_parameters, add_ctd_params

lf = LoggingStyleAdapter(logging.getLogger(__name__))
VERSION = '0.0.1'

# def cmdline_help_mod(version, info):
#     'nmea2h5 version {}'.format(version) + info
#
# def version():
#     """Show the version"""
#     return 'version {0}'.format(VERSION)


def rep_comma_sep_items(names: Iterable[str]) -> list[str]:
    """
    Replaces names that contain ',' to one word from them splitting them by ','. Repeated names with ',' are replaced
    with corresponded to name order word from that name. At that, treats word with length less than first (or previous
    returned) word as abbreviation containing only last characters, so appends them from left with chars equal to that
    in previous returned word.
    :param names: iterable of words
    :return:
    >>>  rep_comma_sep_items([
    ... '49001,15,20', '49001,15,20', '49050,65', '50000', '50000,3', '49024', '49001,15,20', '50000,3'])
    ['49001', '49015', '49050', '50000', '50000', '49024', '49020', '50003']
    """
    # Save state
    prevs = {}

    def replace_comma_separated(name_or_names: str) -> str:
        """
        Replaces comma separated items.
        :param name_or_names: string that returns unmodified if not contains ',' else it is treated as many names
        separates by ','. Which of them to return and whether to modify it is defined by nonlocal ``prevs`` dict having:
        - i: index of word in previous same input ``name_or_names``
        - name: previous name
        :return:
        >>>  prevs = {}
        >>>  replace_comma_separated('49001,15,20')
        '49001'
        >>>  prevs = {'49001,15,20': {'name': '49001', 'i': 0}}
        >>>  replace_comma_separated('49001,15,20')
        '49015'
        >>>  prevs = {'49001,15,20': {'name': '49015', 'i': 1}}
        >>>  replace_comma_separated('49001,15,20')
        '49020'
        # If i-field is too big then just return saved name-field:
        >>>  prevs = {'49001,15,20': {'name': '49001', 'i': 2}}
        >>>  replace_comma_separated('49001,15,20')
        '49001'
        """
        nonlocal prevs
        if ',' in name_or_names:
            names_list = name_or_names.split(',')
            try:
                prev = prevs[name_or_names]
                lp = len(prev_name:=prev['name'])
                prev_i = prev['i']
                prev_i += 1
                try:
                    name = names_list[prev_i]
                except IndexError:
                    # remaining on previous saved name and last prev_i
                    name = prev_name
                    prev_i = len(names_list) - 1
                else:
                    ln = len(name)
                    if ln < lp:  # add skipped chars from left of name:
                        name = f'{prev_name[:-ln]}{name}'
                prevs[name_or_names] = {'name': name, 'i': prev_i}
            except KeyError:
                prevs[name_or_names] = {'name': (name:=names_list[0]), 'i': 0}
            return name
        else:  # name_or_names actually is a name
            return name_or_names

    return [replace_comma_separated(n) for n in names]


def dd_to_csv(
        d: pd.DataFrame,
        text_path=None,
        text_date_format: Optional[str] = None,
        text_columns=None,
        aggregate_period=None,
        suffix='',
        b_single_file=True
        ):
    """
    Depreciated! - see to_pandas_hdf5.dd_to_csv()
    Save to ascii if _text_path_ is not None
    :param d:
    :param text_path: None or directory path
    :param text_date_format: If callable then create "Date" column by calling it (dd.index), retain index only if "Time" in text_columns. If string use it as format for index (Time) column
    :param text_columns: optional
    :param aggregate_period: str or class with repr() to add "bin{}" suffix to files names
    :param suffix:
    :param b_single_file: save all to one file or each partition individually
    :return:
    """
    if text_path is None:
        return


    tab = '\t'
    sep = tab
    ext = '.tsv' if sep==tab else '.csv'
    lf.info('Saving *{}: {}', ext, '1 file' if b_single_file else f'{d.npartitions} files')
    try:
        dir_create_if_need(text_path)
        def combpath(dir_or_prefix, s):
            return str(dir_or_prefix / s)
    except:
        lf.exception('Dir not created!')
        def combpath(dir_or_prefix, s):
            return f'{dir_or_prefix}{s}'

    def name_that_replaces_asterisk(i_partition):
        return f'{d.divisions[i_partition]:%y%m%d_%H%M}'
        # too long variant: '{:%y%m%d_%H%M}-{:%H%M}'.format(*d.partitions[i_partition].index.compute()[[0,-1]])

    filename = combpath(
        text_path,
        f"{name_that_replaces_asterisk(0) if b_single_file else '*'}{{}}_{suffix.replace('incl', 'i')}{ext}".format(
        f'bin{aggregate_period.lower()}' if aggregate_period else '',  # lower seconds: S -> s
        ))

    if True:  # with ProgressBar():
        d_out = d.round({'Vdir': 4, 'inclination': 4, 'Pressure': 3})
        # if not cfg_out.get('b_all_to_one_col'):
        #     d_out.rename(columns=map_to_suffixed(d.columns, suffix))
        if callable(text_date_format):
            arg_out = {'index': bool(text_columns) and 'Time' in text_columns,
                       'columns': bool(text_columns) or d_out.columns.insert(0, 'Date')
                       }
            d_out['Date'] = text_date_format(d_out.index)
        else:
            arg_out = {'date_format': text_date_format,
                       'columns': text_columns or None  # for write all columns if empty (replaces to None)
                       }
        #
        # if progress is not None:
        #     progress(d_out)
        d_out.to_csv(filename=filename,
                     single_file=b_single_file,
                     name_function=None if b_single_file else name_that_replaces_asterisk,  # 'epoch' not works
                     float_format='%.5g',
                     sep=sep,
                     encoding="ascii",
                     #compression='zip',
                     **arg_out
                     )


def h5_tables_gen(db_path, tables, tables_log, db=None) -> Iterator[Tuple[str, pd.HDFStore]]:
    """
    Generate table names with associated coefficients
    :param tables: tables names search pattern or sequence of table names
    :param tables_log: tables names for metadata of data in `tables`
    :param db_path:
    :param cfg_out: not used but kept for the requirement of h5.dispenser_and_names_gen() argument
    :return: iterator that returns (table name, coefficients)
    updates cfg_in['tables'] - sets to list of found tables in store
    """
    # will be filled by each table from cfg['in']['tables']
    try:
        tbl_log_pattern = tables_log[0]
    except omegaconf.errors.ConfigIndexError:
        tbl_log_pattern = ''
        tables_log = ['']
    with FakeContextIfOpen(lambda f: pd.HDFStore(f, mode='r'), file=db_path, opened_file_object=db) as store:
        if len(tables) == 1:
            tables = h5.find_tables(store, tables[0])
        for tbl, tbl_log in zip_longest(tables, tables_log, fillvalue=tbl_log_pattern):
            yield tbl, tbl_log.format(tbl), store


def order_cols(
        df: pd.DataFrame,
        cols: Mapping[str, Any] = None,
        i_log: Optional[int] = None,
        df_log: Optional[int] = None
        ) -> pd.DataFrame:
    """

    :param df:
    :param cols: one of:
    - mapping out column names to expressions (which includes input col. names) for pd.DataFrame.eval().
    1st item will be the index
    - can be also simply input column names
    :param i_log: log row index to can eval expressions referring it as '@i_log'
    :param df_log: log DataFrame to can eval expressions referring it as '@df_log'
    :return:
    """
    if not cols:
        return df

    df = df.copy()


    # def i_term_is_used() -> bool:
    #     for in_col in cols.values():
    #         if 'i' in in_col.split():
    #             return True
    #     return False
    #
    # if i_term_is_used():

    # Add row index to local so in df.eval(expression) we can use it referring as '@i'
    i = np.arange(df.shape[0])  # pd.RangeIndex( , name='rec_num') same effect

    df_out = pd.DataFrame(index=df.index)
    #cols_save = omegaconf.OmegaConf.to_container(cols)  # make editable copy
    # if cols_save.pop('rec_num', None):  # 'rec_num' in df_out
    #     df_out['rec_num'] = df['rec_num']

    dict_rename = {}
    for icol, (out_col, in_col) in enumerate(cols.items()):
        if isinstance(in_col, str) and in_col.isidentifier() and in_col not in dict_rename:
            if in_col not in df.columns:
                if icol == 0 and in_col == 'index':
                    # just change index name
                    df_out.index.name = out_col
                    continue
                else:
                    # add column without data
                    df[in_col] = None
            dict_rename[in_col] = out_col
        elif out_col == '*':  #
            out_cols = [c for c in df.columns if c not in cols.values()]
            df_out[out_cols] = df[out_cols]
            try:
                cols = omegaconf.OmegaConf.to_container(cols)
            except ValueError:  # Input cfg is not an OmegaConf config object
                pass
            del cols['*']
            cols = {**cols, **dict(zip(df_out.columns, df.columns))}
            break
        else:
            if isinstance(in_col, str) and ('@' in in_col or 'df.' in in_col) and not any(s for s in [':', 'f\''] if s in in_col):
                df_out[out_col] = df.eval(in_col.replace('df.', ''))
            elif callable(in_col):  # in_col(df) if callable(in_col) else
                df_out[out_col] = [in_col(ii) for ii in i]  # .apply(
            else:
                df_out[out_col] = eval(in_col)

                    # in_col = in_col.strip().replace('\\', '')
                    # try:
                    #     df_out[out_col] = eval(f'[out_cols_{out_col}_fun(ii) for ii in i]')
                    # except NameError as e:
                    #     if in_col.startswith('def '):
                    #         gpx_names_fun = exec(in_col)  # gpx_names_fun_str
                    #         df_out[out_col] = eval('[gpx_names_fun(ii) for ii in i]')
                    #     else:


    df_to_rename = df[dict_rename.keys()]
    # removing index if exists because df.rename() renames only columns and add it as column
    col_index = dict_rename.pop('index', None)  # index will be placed in this column
    if col_index:
        df_out[col_index] = df_out.index
    df_out = df_out.join(
        df_to_rename.rename(columns=dict_rename, copy=False)
        )

    cols_iter = iter(cols.items())
    index_name, in_1st_col = next(cols_iter)
    if 'index' not in in_1st_col:  # original index is not at 1st column so need to be replaced
        df_out.set_index(index_name, inplace=True)
    return df_out[[k for k, v in cols_iter]]
    # df_out['DATE'] = df_out['DATE'].dt.tz_convert(None)
    # return df_out[cols.keys()]  # seems was not worked ever


if False:
    def interp_vals(
            df: pd.DataFrame, cols: Mapping[str, str] = None,
            i_log_row_st=0,
            times_min=None,
            times_max=None,
            df_search=None,
            cols_good_data=('P', 'Depth'),
            db=None,
            db_path=None,
            table_nav='navigation',
            dt_search_nav_tolerance=timedelta(minutes=2)):
        """

        :param df:
        :param cols: mapping out col names to expressions for pd.DataFrame.eval() (using input col names) or just input col names
        # :param times_min:
        # :param times_max:
        # :param i_log_row_st:
        :param cols_good_data:
        :param db:
        :param db_path:
        :param table_nav:
        :param dt_search_nav_tolerance:
        :return:
        """
        # replace NaNs where it can be found in other tables
        df_out = get_runs_parameters(
            df_search, times_min=df.index, times_max=df.index,
            cols_good_data=cols_good_data,
            dt_from_utc=None, db=db, db_path=db_path,
            add_tables_cols={table_nav: ('Lat', 'Lon', 'DepEcho', 'Speed', 'Course')},
            dt_search_tolerances=dt_search_nav_tolerance
        )

# @dataclass hydra_conf(hydra.conf.HydraConf):
#     run: field(default_factory=lambda: defaults)dir

hydra.output_subdir = 'cfg'
# hydra.conf.HydraConf.output_subdir = 'cfg'
# hydra.conf.HydraConf.run.dir = './outputs/${now:%Y-%m-%d}_${now:%H-%M-%S}'

cs_store_name = Path(__file__).stem
cs, ConfigType = cfg_dataclasses.hydra_cfg_store(
    cs_store_name, {
    'input': ['in_hdf5'],  # Load the config "in_hdf5" from the config group "input"
    'out': ['out_csv'],  # Set as MISSING to require the user to specify a value on the command line.
    'filter': ['filter_CTD'],
    'program': ['program'],
    # 'search_path': 'empty.yml' not works
    })


cfg = {}

@hydra.main(config_name=cs_store_name, config_path="cfg", version_base='1.3')  # adds config store cs_store_name data/structure to :param config data/structure to :param config
def cfg_by_hydra(config: ConfigType):
    """
    Exports Pandas HDF5 store tables data *.h5 to CSV-like files
    ----------------------------
    Saves to CSV specified columns of the HDF5 store data table and its meta table (that contains starts and
    ends of records to split data in many files). Applies data modification functions and renaming of columns.
    Also:
    - calculates CTD columns, accepted by add_ctd_params() that are specified in config['out']['cols'] values
    :param config: with fields:
    - in - mapping with fields:
      - tables_log: - log table name. Can have placeholder '{}' that will be replaced by data table name
      - cols_good_data: -
      ['dt_from_utc', 'db', 'db_path', 'table_nav']
    - out - mapping with fields:
      - cols: mapping of output data csv column names to input data table column names. Values can be data table
      column names and functions with them to eval. Also, these predefined variables can be used:
        - "@i": data row number,
        - "@i_log": log row number that was used to load data range.
        - "@df_log": @i_log-th log row object

      - cols_log: Same as cols, but maps output header csv column names to input log table column names.
        Note: predefined variable of log row number here is "@i".
      - text_date_format
      - file_name_fun, file_name_fun_log - {fun} part of "lambda i_log, t_st, t_en: {fun}" string to compile function
      for name of data and log text files
      - sep

    """
    global cfg

    cfg = cfg_dataclasses.main_init(config, cs_store_name)
    cfg = cfg_dataclasses.main_init_input_file(cfg, cs_store_name)
    #h5.out_init(cfg['in'], cfg['out'])
    #cfg['out']['dt_from_utc'] = 0
    return


def main(**kwargs) -> None:
    global cfg
    cfg_by_hydra()

    if _ := kwargs.get('out_col_station_fun'):
        # assign to global variable
        cfg['out']['col_station_fun'] = _
        # replace call to of gl
        cfg['out']['cols'] = dict(cfg['out']['cols'])
        cfg['out']['cols']['station'] = cfg['out']['cols']['station'].replace(
            'out_col_station_fun', "cfg['out']['col_station_fun']"
        )
        cfg['out']['cols_log'] = dict(cfg['out']['cols_log'])
        cfg['out']['cols_log']['station'] = cfg['out']['cols_log']['station'].replace(
            'out_col_station_fun', "cfg['out']['col_station_fun']"
        )
    qstr_trange_pattern = "index>='{}' & index<='{}'"

    # Prepare saving to csv
    # file name for files and log list:
    for fun in ['file_name_fun', 'file_name_fun_log']:
        cfg['out'][fun] = (
            eval(compile(f"lambda i, t_st, t_en, tbl: {cfg['out'][fun]}", '', 'eval')) if cfg['out'][fun] else
                (
                    (lambda i_log, t_st, t_en, tbl: f'log@{tbl}.csv') if fun.endswith('log') else
                    (lambda i_log, t_st, t_en, tbl: f'{t_st:%y%m%d_%H%M}-{t_en:%H%M}@{tbl}.csv')
                )  # f'_{i}.csv'
            ) if fun not in kwargs else kwargs[fun]
    set_field_if_no(cfg['out'], 'text_path', cfg['in']['db_path'].parent)
    dir_create_if_need(cfg['out']['text_path'])

    # Main circle #############################################################
    i_log_row_st = 0
    for tbl, tbl_log, store in h5_tables_gen(cfg['in']['db_path'], cfg['in']['tables'], cfg['in']['tables_log']):
        # save log list
        if tbl_log:
            df_log = store.select(tbl_log,
                    where=cfg['in']['query']
                    )
            lf.info('Saving {} data files of ranges listed in {}', df_log.shape[0], tbl_log)

            df_log_csv = order_cols(df_log, cfg['out']['cols_log'])

            # df_log_csv = interp_vals(
            #     df_log_csv,
            #     df_search=None,
            #     #cols_good_data = None,
            #     db = store,
            #     dt_search_nav_tolerance = timedelta(minutes=2)
            #     )

            df_log_csv.to_csv(
                cfg['out']['text_path'] / cfg['out']['file_name_fun_log'](
                    i_log_row_st, df_log.index[0], df_log.DateEnd.iat[-1], tbl
                    ),
                date_format=cfg['out']['text_date_format'],
                float_format=cfg['out']['text_float_format'],
                sep=cfg['out']['sep']
                )
        elif tbl:
            lf.info('No log tables found. So exporting all data from {}: ', tbl)
            # set interval bigger than possible to load and export all data in one short
            df_log = pd.DataFrame({'DateEnd': [np.datetime64('now')]}, index=[np.datetime64('1970', 'ns')])
            df_log_csv = None
        else:
            raise(KeyError(f'Table {tbl} not found.'))

        for i_log_row, log_row in enumerate(df_log.itertuples(), start=i_log_row_st):  #  h5.log_rows_gen(table_log=tbl_log, db=store, ):
            # Load data chunk that log_row describes
            print('.', end='')
            qstr = qstr_trange_pattern.format(log_row.Index, log_row.DateEnd)
            df_raw = store.select(tbl, qstr)

            # Calculate CTD columns that are specified in cfg['out']['cols'] values
            ctd2csv(
                df_raw, cfg_in=cfg['in'], cfg_out=cfg['out'], df_log_csv=df_log_csv,
                lat=log_row.Lat_st, lon=log_row.Lon_st, i_log_row=i_log_row, tbl=tbl
            )

        i_log_row_st += df_log.shape[0]

    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} Ok>", end=' ')


def ctd2csv(
        df_raw, cfg_out: Mapping[str, Any], cfg_in: Optional[Mapping[str, Any]] = None,
        df_log_csv=None, lat=None, lon=None, log_row=None, i_log_row=None, tbl=None
):
    """
    Calculates specified additional params of CTD and add data related to current record using order_cols dict.
    Saves result to CSV.
    :param cfg_out:
    - cols: list of columns in output dataframe (you may add new columns to calculate)
    - text_path
    - file_name_fun: callable(i_log_row, t_st, t_en, tbl) where t_st, t_en will be df_raw.index[0], df_raw.index[-1]
    - text_date_format
    - text_float_format
    - sep
    :param df_raw:
    :param df_log_csv: can be used for cfg_out['cals'] items to eval
    :param i_log_row:
    :param log_row:
    :param tbl:
    :return:
    """
    # Calculate CTD parameters of cols if they are not in df_raw
    if isinstance(cfg_out['cols'], Mapping):
        ctd_cols = [c for c in cfg_out['cols'].values() if isinstance(c, str) and c.isalnum()]
    else:
        ctd_cols = cfg_out['cols']
        cfg_out['cols'] = {'Time': 'index', **{c: c for c in ctd_cols}}
    df_raw = add_ctd_params(
        df_raw, {
            'in': {} if cfg_in is None else cfg_in,
            'out': {
                **cfg_out,
                'data_columns': ctd_cols
            }
        },
        lon=lon, lat=lat
    )

    # Calculate user custom parameters
    df_csv = order_cols(
        df_raw, cfg_out['cols'], i_log=i_log_row, df_log=df_log_csv
    )

    # Save data
    df_csv.to_csv(
        cfg_out['text_path'] / cfg_out['file_name_fun'](
            i_log_row, df_raw.index[0], df_raw.index[-1], tbl
        ),
        date_format=cfg_out['text_date_format'],
        float_format=cfg_out['text_float_format'],
        sep=cfg_out['sep']
    )


def main_call(
        cmd_line_list: Optional[List[str]] = None,
        fun: Callable[[Any], Any] = main,
        **kwargs
        ) -> Dict:
    """
    Adds command line args, calls fun, then restores command line args
    :param cmd_line_list: command line args of hydra commands or config options selecting/overwriting

    :return: global cfg
    """

    sys_argv_save = sys.argv.copy()
    if cmd_line_list is not None:
        sys.argv += cmd_line_list

    # hydra.conf.HydraConf.run.dir = './outputs/${now:%Y-%m-%d}_${now:%H-%M-%S}'
    fun(**kwargs)
    sys.argv = sys_argv_save
    return cfg


if __name__ == '__main__':
    main()  #[f'--config-dir={Path(__file__).parent}'])
