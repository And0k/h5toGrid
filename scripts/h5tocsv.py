#!/usr/bin/env python
# coding:utf-8
from __future__ import print_function, division

"""
  Author:  Andrey Korzh <ao.korzh@gmail.com>
  Purpose: load CTD data from hdf5 db (from Veusz data source store) and save it to csv,
            with adding nav.
  Created: 18.10.2016
"""

from os import path as os_path
from sys import stdout as sys_stdout
import numpy as np
import pandas as pd

from utils2init import ini2dict
from grid2d_vsz import inearestsorted

date_format_ISO9115 = '%Y-%m-%dT%H:%M:%S'  # for Obninsk
# ##############################################################################
# ___________________________________________________________________________
if __name__ == '__main__':
    #    unittest.main()
    import argparse

    parser = argparse.ArgumentParser(description='Extract data from Pandas HDF5 '
                                                 'store*.h5 files to CSV including log',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     epilog='If use special characters then insert arguments in quotes',
                                     )
    parser.add_argument('--version', action='version', version='%(prog)s '
                                                               'version 0.0.1 - (c) 2016 Andrey Korzh <ao.korzh@gmail.com>')  # sourceConvert

    parser.add_argument('cfgFile', nargs='?', type=str, default='h5toObninsk.ini',
                        help='Path to confiuration *.ini file with all parameters. '
                             'Next parameters here overwrites them')
    info_default_path = '[in] path from *.ini'
    parser.add_argument('path', nargs='?', type=str, default=info_default_path,
                        help='Path to source file(s) to parse')
    parser.add_argument('out', nargs='?', type=str, default='./<filename>.h5/gpx',
                        help='''Output .h5/table path.
If "<filename>" found it will be substituted with [1st file name]+, if "<dir>" -
with last ancestor directory name. "<filename>" string
will be sabstituted with correspondng input file names.
''')
    parser.add_argument('-verbose', nargs=1, type=str, default=['INFO'],
                        choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'],
                        help='Verbosity of messages in log file')

    args = parser.parse_args()
    args.verbose = args.verbose[0]
    # args.cfgFile= 'csv2h5_nav_supervisor.ini'
    # args.cfgFile= 'csv2h5_IdrRedas.ini'
    # args.cfgFile= 'csv2h5_Idronaut.ini'
    try:
        cfg = ini2dict(args.cfgFile)
        cfg['in']['cfgFile'] = args.cfgFile
    except IOError as e:
        print('\n==> '.join([a for a in e.args if isinstance(a, str)]))  # e.message
        raise (e)
    nadd = len(cfg['in']['table']) - len(cfg['output_files']['file_names'])
    if nadd != 0:  # same file names for all tables
        cfg['output_files']['file_names'] += [cfg['output_files']['file_names'][-1]] * nadd
    # Load data #################################################################
    fileOutP = cfg['output_files']['path'] if 'path' in cfg['output_files'] else os_path.dirname(
        cfg['in']['path'])
    qstr_trange_pattern = "index>=Timestamp('{}') & index<=Timestamp('{}')"
    if 'file_names_add_fun' in cfg['output_files']:
        file_names_add = eval(compile(cfg['output_files']['file_names_add_fun'], [], 'eval'))
    else:
        file_names_add = lambda i: str(i) + '.csv'
    iSt = 1
    with pd.HDFStore(cfg['in']['path'], mode='r') as storeIn:
        for tblD, fileOutN in zip(cfg['in']['table'],
                                  cfg['output_files']['file_names']):
            if False:  # Show table info
                storeIn.get_storer(tblD).table  # ?
                nodes = sorted(storeIn.root.__members__)  # , key=number_key
                print(nodes)
                # storeIn.get_node('CTD_Idronaut(Redas)').logFiles        # next level nodes
            print(tblD, end='. ')
            bFirst_data_from_table = True
            tblL = tblD + '/logFiles'
            dfL = storeIn[tblL]

            # Get navigation at data points
            qstr = qstr_trange_pattern.format(dfL.index[0], dfL['DateEnd'][-1])
            print('loading all needed nav: ', qstr, end='... ')
            Nav = storeIn.select(cfg['in']['table_nav'], qstr, columns=[])
            Nind_st = storeIn.select_as_coordinates(cfg['in']['table_nav'], qstr)[0]
            dfL.index = dfL.index.tz_convert('UTC')
            Nind = inearestsorted(Nav.index.values, dfL.index.values)

            # strLog= '{0:%d.%m.%Y %H:%M:%S} - {1:%d. %H:%M:%S%z}\t{2}rows'\t{2}rows'.format(
            #     dfL.index[0], dfL.index[-1], dfL.size) #\t{Lat}\t{Lon}\t{strOldVal}->\t{mag}
            # print(strLog)

            # Check time difference between nav found and time of requested points
            dT = np.array(Nav.index[Nind].values - dfL.index.values, 'timedelta64[ns]')
            bBad = np.any(abs(dT) > np.timedelta64(cfg['process']['dt_search_nav_tolerance']))
            if np.any(bBad):
                print('Bad nav. data coverage: difference to nearest point in time [min]:')
                print('\n'.join(['{}. {}:\t{}{:.1f}'.format(i, tdat, m, dt.seconds / 60) for i, tdat, m, dt in
                                 zip(np.flatnonzero(bBad), dfL.index[bBad],
                                     np.where(dT[bBad].astype(np.int64)
                                              < 0, '-', '+'), np.abs(dT[bBad]))]))
            Nind += Nind_st
            nav2add = storeIn.select(cfg['in']['table_nav'], where=Nind,
                                     columns=cfg['output_files']['nav_cols'])
            print("{} rows loaded".format(dfL.shape[0]))

            # Save waypoints to csv
            fname = fileOutN + '_POS.csv'
            fileOutPN = os_path.join(fileOutP, fname)
            nav2add = nav2add.assign(Date=dfL.index, Identific=np.arange(start=iSt, stop=iSt + dfL.shape[0])
                                     ).set_index(np.arange(start=iSt, stop=iSt + dfL.shape[0]), drop=False)
            nav2add[['Identific', 'Date', 'Lat', 'Lon']].to_csv(
                fileOutPN, mode='a', header=not os_path.isfile(fileOutPN),
                date_format=date_format_ISO9115, float_format='%2.8f', index_label='Rec_num')
            if True:
                for i, r in enumerate(dfL.itertuples(), start=iSt):  # name=None
                    print('.', end='')
                    sys_stdout.flush()
                    # str_time_short= '{:%d %H:%M}'.format(r.Index.to_datetime())
                    # timeUTC= r.Index.tz_convert(None).to_datetime()
                    # str_time_long= ('{:'+date_format_ISO9115+'}').format(timeUTC)
                    fname = fileOutN + file_names_add(i)

                    # Save station data
                    qstr = qstr_trange_pattern.format(r.Index, r.DateEnd)
                    Dat = storeIn.select(tblD, qstr)
                    Dat = Dat.assign(Date=Dat.index.tz_convert(None), Identific=i).set_index(
                        np.arange(1, Dat.shape[0] + 1), drop=False)
                    cols = Dat.columns.tolist()
                    if bFirst_data_from_table:
                        # check existanse of needed cols
                        bFirst_data_from_table = False
                        add_empty_col = [c for c in cfg['output_files']['data_columns'] if c not in cols]
                    for c in add_empty_col:
                        Dat[c] = None

                    # cols = list(df.columns.values)
                    # pd.merge(ind, Dat) #Dat.join
                    Dat[cfg['output_files']['data_columns']].to_csv(os_path.join(fileOutP, fname),
                                                                    date_format=date_format_ISO9115,
                                                                    float_format='%4.4g',
                                                                    index_label='Rec_num')  # to_string, line_terminator='\r\n'

            iSt += nav2add.shape[0]
            print('')
        if len(cfg['in']['table']) > 1:
            # Save combined data to gpx
            print('.')
    print('Ok')