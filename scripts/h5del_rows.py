import pandas as pd
from to_pandas_hdf5.h5toh5 import ReplaceTableKeepingChilds

def h5del_rows(db_path, tbl, tbl_log=None, keep_where: str = ''):
    with pd.HDFStore(db_path) as store_in:
        df = store_in.select(tbl)
        len_before = len(df)
        print('Store', db_path, 'has', len_before, 'rows in table', tbl)
        df = store_in.select(tbl, where=keep_where)
        len_after = len(df)
        print('Loading and saving', len_after, 'rows only')
        if len_after >= len_before:
            print('Nothing to do')
            return
        # df_log = store_in.select(tbl_log, where=keep_where)
        with ReplaceTableKeepingChilds(df, tbl, {'db': store_in}):
            pass


db_path = r'd:\WorkData\BalticSea\210515_tracker\210618_1440_wind@tr2\210618_1440tr2.h5'
tbl = 'tr2'
tbl_log = f'{tbl}/log'
q_exclude = "index<=Timestamp('{}') | index>=Timestamp('{}')"  # UTC
intervals_to_remove = [
    #('2021-06-21T12:30', '2021-06-21T12:50'),
    ('2021-06-21T17:20', '2021-06-21T19:40'),
    ('2021-06-25T09:50', '2021-06-25T12:50'),
    ]

keep_where = '&'.join(f'({q_exclude.format(*t_st_en)})' for t_st_en in intervals_to_remove)
h5del_rows(db_path, tbl, tbl_log, keep_where)