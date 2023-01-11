import pandas as pd
from to_pandas_hdf5.h5toh5 import h5log_names_gen, h5init, h5_dispenser_and_names_gen, h5move_tables, h5index_sort, h5remove  # ReplaceTableKeepingChilds
from to_pandas_hdf5.h5_dask_pandas import h5_append
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence, Tuple, Union
from pathlib import PurePath
from time import sleep

def h5cor(
        time_ranges: Sequence[Union[str, Tuple[str, str]]],
        edges_sources: str,
        b_update: bool,
        cfg_out: Mapping[str, Any],
        **kwargs
        ):
    """

    :param time_ranges: list of:
    - time in log record, to get time range of data from it
    - time range of log records, to get time range of data from 1st and last of them
    - time range in data table
    :param edges_sources: word (list of letters) mapped here to log table name, determines
    whether range is improved by getting edges from:
    - 'f': logFiles,
    - 'r': logRuns,
    - any else: query timerange of data directly
    :param b_update: if True Values correction (updating DB) else Deleting bad runs from DB (latter not tested)
    :param cfg_out:
    :param kwargs:
    :return:
    """

    def check_len(db, df_len_prev=None):
        df = db[cfg_out['table']]
        df_len = df.shape[0]
        df_is_sorted = df.index.is_monotonic
        b_strange = df_len_prev and df_len_prev != df_len
        if b_strange or df_len_prev is None or not df_is_sorted:
            print(
                f'{PurePath(db.filename).name}/{cfg_out["table"]} shape: {df.shape}.',
                'It is', 'sorted.' if df_is_sorted else 'not sorted!'
                )
        if b_strange:
            print(f'Data shape change: {df_len - df_len_prev} rows!')
            pass
        return df_len

    def yield_intervals(*args) -> Iterator:
        """
        :param args: for compatibility with intended usage as fun_gen argument of h5_dispenser_and_names_gen()
        globals: time_ranges
        yields: (edges_source, pattern)
        """
        for t_range, edges_source in zip(time_ranges, edges_sources):
            b_st_en = (len(t_range) == 2)
            str_q = (
                "index>Timestamp('{}') and DateEnd<=Timestamp('{}')".format(*t_range) if b_st_en else
                "index<=Timestamp('{}') and DateEnd>Timestamp('{}')".format(t_range, t_range)
                ) if edges_source in 'fr' else \
                "index>=Timestamp('{}+00:00') and index<=Timestamp('{}+00:00')".format(*t_range)
            yield edges_source, str_q

    cfg_out = {
        **cfg_out,
        'b_insert_separator': False,  # should be already inserted
        'dropna': False  # Not change previously inserted separators
        }
    h5init({}, cfg_out)
    # cfg_out['tables_written'] = set()
    with pd.HDFStore(cfg_out['db_path'], 'r') as store:
        if __debug__:
            out_len = out_len_tmp = check_len(store)
        for i, (edges_source, query) in h5_dispenser_and_names_gen(
                cfg_in={'db': store}, cfg_out=cfg_out,
                fun_gen=yield_intervals,
                ):
            if __debug__:
                out_len_tmp = check_len(cfg_out['db'], out_len_tmp)

            if edges_source in 'fr':
                log = 'logFiles' if edges_source == 'f' else 'logRuns'
                table_log = f"/{cfg_out['table']}/{log}"
                df_log = store.select(table_log, query)
                df_log_len = len(df_log)
                print(f' {i}.', df_log_len, log, 'rows loaded', end=', ' if df_log_len else '!!!')
                if not df_log_len:
                    print('query', query, 'not returns', table_log, 'records.\nCheck input range!')
                query_df = "index>=Timestamp('{}') and index<=Timestamp('{}')".format(
                   *[df_log.index[0], df_log.DateEnd[-1]]
                   )  # *[t for t in df_log_bad_range.DateEnd.items()][0]
                print('pointing to data:\n', query_df)
            else:
                query_df = store.select(cfg_out['table'], where=query)
                print(f' {i}.', 'query', cfg_out['table'], 'directly')
            if b_update:
                df = store.select(cfg_out['table'], query_df)
                print('Calculations on the loaded', df_len := df.shape[0], end=' rows, ')
                if not df_len:
                    print('query', query, 'not returns', cfg_out['table'], 'records.\nCheck input range!')

                df[['O2', 'O2ppm']] *= kwargs['coef_for_interval'][i - 1]
                # df[['O2', 'O2ppm']] = df[['O2', 'O2ppm']] * coef_for_interval[i - 1]
            else:
                print('Removing log rows from temp store(', PurePath(cfg_out['db'].filename).name, end='), ')
                # cfg_out['db'].remove(table_log, where=query)
                h5remove(cfg_out['db'], table_log, query)

            print('Removing rows from temp store (', PurePath(cfg_out['db'].filename).name, end=')')
            h5remove(cfg_out['db'], cfg_out['table'], query_df)

            if b_update:
                print('appending corrected rows to temp store...')
                # df.to_hdf(cfg_out['db'], cfg_out['table'], append=True, data_columns=True,
                #            format='table', dropna=True, index=False)

                # cfg_out['tables_written'] = {cfg_out['table']}
                try:
                    h5_append(cfg_out={**cfg_out, 'b_log_ready': True}, df=df, log={})
                    cfg_out['db'].flush()  # not helps: 2304900 # 1679834
                    # flush() not works, so alternative:
                    cfg_out['db'].close(); sleep(1); cfg_out['db'].open('r+');
                    if __debug__:
                       # why gets pandas.io.pytables.ClosedFileError on next cycle?
                        sleep(1)
                        out_len_tmp = check_len(cfg_out['db'], out_len_tmp)
                    pass
                except Exception as e:
                    print(e)
        # h5remove(store, cfg_out['table'])
    #if cfg['in'].get('time_last'):
    # Replace old store with temp store
    if b_update:  # optional speedup, log tables will be copied with parent anyway
        # del cfg_out['table_log']
        del cfg_out['tables_log']
    failed_storages = h5move_tables({
        **cfg_out,
        'addargs': ['--overwrite-nodes', '--checkCSI', '--verbose']
        })
    print('Ok.', end=' ')

    if __debug__:
        with pd.HDFStore(cfg_out['db_path']) as store:
            out_len = check_len(store, out_len)

        with pd.HDFStore(cfg_out['db_path_temp']) as store:
            out_len_tmp = check_len(store, out_len_tmp)

    h5index_sort(cfg_out,
                 out_storage_name=f"{cfg_out['db_path'].stem}-resorted.h5",
                 in_storages=failed_storages)

    if __debug__:
        with pd.HDFStore(cfg_out['db_path']) as store:
            out_len = check_len(store, out_len)
            pass

    with pd.HDFStore(cfg_out['db_path']) as store:
        h5remove(store, f"/{cfg_out['table']}/logRuns")


def main(path_db, device):
    """

    :param path_db:
    :param device:
    :return:

    # path_db = r'd:\WorkData\BalticSea\220601_ABP49\220601_ABP49.h5'
    # device = 'CTD_Idronaut_OS316#494'
    """

    # Find bad runs that
    # - starts before and ends after this values i.e. 1 time value to find run to delete:
    time_ranges = [  # time_in_bad_run
        '2022-06-02 19:15+00:00',
        '2022-06-02 19:22+00:00'
        ]
    # - starts after and ends before (smaller) intervals (2 time values to find interval of runs or source files to delete):
    time_ranges += [  # time_ranges_of_runs
        ('2022-06-02 19:31+00:00', '2022-06-02 20:05+00:00'),  # two runs from 19:31 to 20:04
        ('2022-06-02 20:02+00:00', '2022-06-04 22:28+00:00'),  # file start in 20:02
        ('2022-06-05 04:25+00:00', '2022-06-05T06:00+00:00'),
        ('2022-06-12 12:27+00:00', '2022-06-13 07:11+00:00'),
        ]
    coef_for_interval = [0.923215, 0.94403, 0.95151, 0.95990, 1.087, 0.721426]

    edges_sources = 'rrrfff'

    cfg_out = {
        'table': f'/{device}',
        'tables_log': [f'/{device}/logFiles', f'{device}/logRuns'],  # to prepare to move/sort on close db
        'db_path': path_db,
        }
    # print('Tables: {table}, {tables_log}'.format_map(cfg_out))

    h5cor(time_ranges, edges_sources, b_update=True,
          cfg_out=cfg_out, coef_for_interval=coef_for_interval
          )







    # with pd.HDFStore(path_db) as store:
    #     for i, query_log in enumerate(yield_intervals()):
    #         df_log_bad_range = store.select(tbl_log, where=query_log)
    #         query_df = "index>=Timestamp('{}') and index<=Timestamp('{}')".format(
    #             *[df_log_bad_range[0].index, df_log_bad_range[-1].DateEnd]
    #             )  # *[t for t in df_log_bad_range.DateEnd.items()][0]
    #         if b_update:
    #             # calculate
    #             df = store.select(tbl, query_df)
    #             df['O2'] *= coef_for_interval[i]
    #             df['O2ppm'] *= coef_for_interval[i]
    #
    #         # remove rows
    #         if True:  #len(df_log_bad_range) == 1:
    #             if not b_update:
    #                 store.remove(tbl_log, where=query_log)
    #             store.remove(tbl, query_df)
    #         else:
    #             print('Not found run with time {}'.format(t))
    #
    #         if b_update:
    #             store.select(tbl, query_df)