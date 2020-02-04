#!/usr/bin/env python
# coding:utf-8
from __future__ import print_function

from datetime import timedelta

"""
  Author:  Andrey Korzh --<ao.korzh@gmail.com>
  Purpose: Convert source text files of AB SIO RAS probes such as "Termochain",
  "Inclinometr" to PyTables hdf5 file
  Created: 27.05.2015
"""

cfg = {"strProbe": 'navigation',
       "TimeAdd": timedelta(hours=0)  # UTC?
       }
h5NameF = r'd:\WorkData\_source\BalticSea\160310_DrLubecki\ADCP_WH600\nav_NMEA\converted\160319,21WinRiver.h5'
# d:\WorkData\_source\BalticSea\160310_DrLubecki\navigation\Garmin_GPSmap_62stc\Garmin_GPSmap_62stc.h5'
# r'd:\WorkData\_source\BalticSea\160310_DrLubecki\ADCP_WH600\nav_NMEA\converted\10-12ADCP.h5'
#
# r'd:\workData\_source\KaraSea\150816_Kartesh-river_Ob\navigation\1508-09all.h5'
# r'd:\WorkData\Cruises\_BalticSea\150429_Oceania(NoMe)\150429_Oceania.h5'
DB = {'name': r'd:\WorkData\_subproduct\BalticSea\160310_DrLubecki\BaS1603.mdb',
      'tbl_navigation': cfg["strProbe"]
      }
"""
DB= {'name': r'd:\workData\_source\KaraSea\150816_Kartesh-river_Ob\navigation\1505r_Ob.mdb',
     'tbl_navigation': cfg["strProbe"]
}
"""
from pandas import get_store as pd_get_store
import pyodbc

# Make a direct connection to a database and create a cursor.
cnxn = pyodbc.connect(r"DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=" + DB['name'] + ";")
cursor = cnxn.cursor()

with pd_get_store(h5NameF) as store:
    df = store.get(cfg["strProbe"])
    df.index += cfg['TimeAdd']
    # from sqlalchemy import create_engine
    # engine = create_engine('access+pyodbc:///' + DB['name'])
    # df.to_sql(cfg["strProbe"], engine, if_exists='append')

    strInsert = "insert into " + cfg["strProbe"] + "(iTime, Lat, Lon) values (?,?,?)"
    # cursor.executemany(strInsert, df.itertuples())
    nBlock = df.shape[0] / 10

    print('save ' + format(df.shape[0], '2') + ' rows in 10 blocks', end='')
    kLast = 0
    kBreak1 = df.shape[0] % nBlock
    iError = 0
    iCheck = 0  # to del
    if kBreak1 == 0: kBreak1 = nBlock
    for k in xrange(kBreak1, df.shape[0] + 1, nBlock):
        for r in df[kLast:k].itertuples():
            try:
                cursor.execute(strInsert, r[0].strftime("%d-%m-%Y %H:%M:%S"),
                               format(r[1], '6.15'), format(r[2],
                                                            '6.15'))  # ("%a %b %d %H:%M:%S UTC %Y")'Sat Oct 04 13:00:36 UTC 2014'
            except pyodbc.IntegrityError:
                iError += 1
                pass
            iCheck += 1
        print('.', end='')
        cnxn.commit()
        kLast = k
    if iError > 0: print('There error occured ' + format(iError, '2') + ' times')
    if iCheck != df.shape[0]: print('program logic error: Data size (' +
                                    format(df.shape[0], '2') + ') not equal ' + format(iCheck,
                                                                                       '2') + ' lines prosessed')
    cursor.close()
    cnxn.close()
print('ok')
# access_database_file = 'C:\\Users\\davisr\\My Documents\\TEMP\\Comp_Model_Db_Testing.mdb'
# ODBC_CONN_STR = 'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=%s;' %access_database_file
