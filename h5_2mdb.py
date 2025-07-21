#!/usr/bin/env python
# coding:utf-8

from datetime import timedelta
from pandas import HDFStore, isna
import pyodbc

from to_pandas_hdf5.h5toh5 import h5.load_ranges
"""
  Author:  Andrey Korzh --<ao.korzh@gmail.com>
  Purpose: Insert PyTables timeseries data from table of hdf5 file to existed table in Microsoft Access database.
  Created: 27.05.2015
"""


def insert(in_path, out_path, in_tbl=None, out_tbl=None, time_add=None,
           odbc_conn_str='DRIVER=Microsoft Access Driver (*.mdb, *.accdb);DBQ={out_path};',
           cols=None, time_ranges=None):
    """

    :param in_path:
    :param out_path:
    :param in_tbl:
    :param out_tbl:
    :param time_add:
    :param odbc_conn_str: ODBC connection string
    :param cols: for example:
        ['Lat', 'Lon'], or
        {'Lat': 'Lat', 'Lon': 'Lon', '': 'DepEcho': 'Depth'}
    :return:


    out_path = r'd:\WorkData\_subproduct\BalticSea\160310_DrLubecki\BaS1603.mdb'
    in_tbl = 'navigation'
    """

    if in_tbl is None:
        in_path, in_tbl = in_path.rsplit('/', 1)

    if out_tbl is None:
        out_tbl = in_tbl


    # Make a direct connection to a database and create a cursor.
    cnxn = pyodbc.connect(odbc_conn_str.format(out_path=out_path.replace('\\', '/')))
    cursor = cnxn.cursor()

    with HDFStore(in_path) as store:
        df = h5.load_ranges(store, table=in_tbl, t_intervals=time_ranges)
        # df = store[in_tbl]
        df.index += time_add
        print(f'Loaded data ({df.columns}) of range:', df.index[[0, -1]])
        # from sqlalchemy import create_engine
        # engine = create_engine('access+pyodbc:///' + DB['name'])
        # df.to_sql(in_tbl, engine, if_exists='append')
        if cols is None:
            cols = cols_out = df.columns
        elif isinstance(cols, list):
            cols_out = cols
            df = df[cols]
        else:  # cols is Mapping
            cols_out = cols.values()
            cols = list(cols.keys())
            df = df[cols]

        if 'DepEcho' in cols:
            print('col. DepEcho inverting and shifting!')
            df['DepEcho'] = -df['DepEcho'] - 1.5

        str_insert = f'insert into {out_tbl} (iTime, {",".join(cols_out)}) values ( ' \
                     f'{",".join(["?"]*(len(cols) + 1))})'
        formats = ['%d-%m-%Y %H:%M:%S'] + ['.15'] * (len(cols))

        # cursor.executemany(str_insert, df.itertuples())
        block_size = df.shape[0] // 10
        block1size = df.shape[0] % block_size  # 1st block equal to the reminder
        print(f'Saving {df.shape[0]} rows in 10 blocks... ', end='')
        kLast = 0

        iError = 0
        iCheck = 0  # to del
        if block1size == 0:
            block1size = block_size
        for k in range(block1size, df.shape[0] + 1, block_size):
            for r in df.iloc[kLast:k].itertuples():
                try:
                    cursor.execute(str_insert,
                                   *[None if isna(c) else f'{c:{fmt}}' for c, fmt in zip(r, formats)],
                                   # r[0].strftime("%d-%m-%Y %H:%M:%S"), format(r[1], '6.15'), format(r[2], '6.15')
                                   )  # ("%a %b %d %H:%M:%S UTC %Y")'Sat Oct 04 13:00:36 UTC 2014'
                except pyodbc.IntegrityError:
                    iError += 1
                    pass
                iCheck += 1
            print('.', end='')
            cnxn.commit()
            kLast = k
        if iError > 0:
            print(f'Error occurred {iError} times')
        if iCheck != df.shape[0]:
            print('program logic error: Data size (' + format(df.shape[0], '2') +
                  ') not equal ' + format(iCheck, '2') + ' lines processed'
                  )
        cursor.close()
        cnxn.close()
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} Ok>")


if __name__ == '__main__':

    # Default full config
    cfg = {
        'in': {
            'tbl': 'navigation',
            'time_shifted': timedelta(hours=0),  # shift of source data relative to needed output
            'time_ranges': None,                 # ranges of source data (before we'll remove the shift)
            'path': r'',
            'cols': ['Lat', 'Lon']  # , 'DepEcho']

        },
        'out': {
            'path': r'',
            'tbl': None,
            'cols': ['Lat', 'Lon']  #, 'Depth']
        }
       }

    # Overwrite default config:
    cfg['in']['path'] = r'd:\WorkData\KaraSea\231110_AMK93\231110_AMK93.h5'
        # r'BlackSea\220920\220920.nav.h5'  # r'160310_DrLubecki\ADCP_WH600\nav_NMEA\converted\160319,21WinRiver.h5',
    cfg['out']['path'] = r'd:\WorkData\KaraSea\231110_AMK93\ADCP_75kHz\ADCP_75kHz.mdb'
    #  cfg['in']['time_shifted'] = timedelta(hours=-3)
    cfg['in']['time_ranges'] = ['2000-01-01', '2023-11-10T17:05:35']
    # Run
    insert(
        cfg['in']['path'], cfg['out']['path'],
        cfg['in']['tbl'], cfg['out']['tbl'],
        time_add=-cfg['in']['time_shifted'],
        cols=dict(zip(cfg['in']['cols'], cfg['out']['cols'])),
        time_ranges=cfg['in']['time_ranges']
    )


# d:\WorkData\_source\BalticSea\160310_DrLubecki\navigation\Garmin_GPSmap_62stc\Garmin_GPSmap_62stc.h5'
# r'd:\WorkData\_source\BalticSea\160310_DrLubecki\ADCP_WH600\nav_NMEA\converted\10-12ADCP.h5'
#
# r'd:\workData\_source\KaraSea\150816_Kartesh-river_Ob\navigation\1508-09all.h5'
# r'd:\WorkData\Cruises\_BalticSea\150429_Oceania(NoMe)\150429_Oceania.h5'

# """
# DB= {'name': r'd:\workData\_source\KaraSea\150816_Kartesh-river_Ob\navigation\1505r_Ob.mdb',
#      'tbl_navigation': in_tbl
# }
# """

# access_database_file = 'C:\\Users\\davisr\\My Documents\\TEMP\\Comp_Model_Db_Testing.mdb'
# ODBC_CONN_STR = 'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=%s;' %access_database_file
