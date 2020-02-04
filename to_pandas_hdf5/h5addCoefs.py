# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 21:02:08 2013

@author: Korzh
"""
import os.path
import sys
from datetime import datetime  #

# HDF5 library imports
import h5py
import numpy as np
import pandas as pd
# from sys import path
# path.append(r'd:\Work\_Python\_other')
from _other.mat73_to_pickle import recursive_dict, matlab2datetime64

all_coef_path = r'd:\Work\MatlabHistory\mat\coef_Electronics.mat'


def stopHere(strmessage, Nprobe=0):
    print(strmessage)


def coefLoad(Nprobe=0, dataTime0=0, probeType='i'):
    # probeType 'Inclinometer' or use its alias 'i','#'

    # Example:
    # SCoef= coefLoad(datenum('140617','yymmdd'),'Inclinometr',6);
    # coefDisp(SCoef) #  Show loaded coeficients

    if type(dataTime0) == str:
        dataTime0 = np.datetime64(datetime.strptime(dataTime0, '%Y\t%m\t%d\t%H\t%M\t%S'),
                                  'ns').astype(np.int64)  # dataTime= "2013	08	07	17	00	00"
        # datetime64('1931-01-01','D') + array(index/1000000000.0, 'timedelta64[s]')
    SCoef = {}
    if Nprobe > 0 and os.path.isfile(all_coef_path):
        if probeType.startswith(('i', 'I', '#')): probeType = 'Inclinometer'
        f = h5py.File(all_coef_path, 'r')
        SCoef = recursive_dict(f[probeType])
        SCoef['TimeRange'] = matlab2datetime64(SCoef['TimeRange'])
        # not (probeType in SCoef and
        if not 'i' in SCoef: stopHere('Bad coefficients file!')
        bInd = SCoef['i'] == Nprobe
        k = np.sum(bInd)
        while k > 1:
            if k > 1:
                k = np.nonzero(bInd)[0]
                for t in k:
                    bInd[t] = not ((dataTime0 < SCoef['TimeRange'][t, 0]) or
                                   (dataTime0 >= SCoef['TimeRange'][t, 1]))  # inversion for right NaN proc
                #        if sum(bInd)==1:
                #          k= k(bInd); bInd= false(size(SCoef)); bInd(k)= true
                if not np.any(bInd):
                    p = str(input(format((Nprobe, datetime.ctime(dataTime0), len(k)),
                                         'No coef for probe#%d for data %s.\n' +
                                         'If use some of %d existed for this probe then input it sequence number:')))
                    if 0 < p and p <= len(k):
                        bInd[:] = False;
                        bInd[k[p]] = True;
                        k = 1
                    else:
                        k = 0
                else:
                    k = sum(bInd)

            if k == 0:
                p = float(input(format((Nprobe, datetime.ctime(dataTime0)), 'No coef for probe#%d for data %s.\n' +
                                       'If use other # input it:')))
                bInd = SCoef['i'] == p
                k = sum(bInd)
            elif k > 1:
                p = float(input('More than 1 coef (%d) for probe#%d.\n' +
                                'Input used sequence #:', k, Nprobe))
                if 0 < p and p <= k: k = np.nonzero(bInd); bInd[:] = False; bInd[k[p]] = True; k = 1
            else:
                k = 0
        if k == 1:
            print('\nUse coefficients for probe %s#%d obtained %s ',
                  probeType, Nprobe, datetime.ctime(SCoef['TimeProcessed'][bInd]))  # , 'dd.mm.yyyy HH:MM'
            SCoef = SCoef[bInd]
            return SCoef
        else:
            print('\nNo coef for probe#%d', Nprobe)


def h5addCoefs(args):
    SCoef = coefLoad(Nprobe=1, dataTime0='2015	01	01	00	00	00', probeType='i')

    FileInF = args.source[0]
    bColsSpecified = len(args.source > 1) & args.source[1] != '*'
    with pd.HDFStore(FileInF, mode='r') as storeIn:
        for strProbe in storeIn:
            if bColsSpecified & strProbe != args.source[1]: continue
            try:
                df = storeIn.select_column(strProbe, 'index')
            except:
                print('no index column!')
                continue

        # df= storeIn.get(strProbe)
        # storeIn.close()
        if df.shape[0] > 0:
            if not args.chunkDays: args.chunkDays = 1
            tGrid = np.arange(df[0].date(), df[df.shape[0] - 1].date() +
                              pd.Timedelta(days=1), pd.Timedelta(days=args.chunkDays),
                              dtype='datetime64[D]').astype('datetime64[ns]')
            # iOut= np.hstack([coord[0] + np.searchsorted(df.values, tGrid), coord[1]])
            # if coord[0]==0 and iOut[0]!=0:
            #     iOut= np.hstack([0, iOut])

        else:
            iOut = 0

    #    iOut= np.zeros(np.size(tGrid), np.int32)
    #    for i, tSt in enumerate(tGrid):
    #        iOut[i]= np.argmin(df.values<tSt)

    # iOut= [np.argmin(df.values<tSt) for tSt in tGrid]
    sys.stdout.write(str(iOut))


#    df[(df.index  >  0)  &  (df_dc.C  >  0)
#    storeIn.select(strProbe, where= [str_where])


#    n= FileInF.rfind("\\")
#    store= pd.HDFStore(FileInF[:(n+1)]+'out.hdf')
#    store.append('df',  df)
#    store.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Save part of source file.')
    parser.add_argument('source', nargs=1, type=str,
                        help='Source HDF5 file full path, Node name or *')
    parser.add_argument('-coef', nargs='*', type=str, help='Non standard coef file full path')
    parser.add_argument('-saveTo', type=str,
                        help='Save result (default: overwrite source)')

    args = parser.parse_args()
    h5addCoefs(args)


if __name__ == '__main__':
    main()  # sys.stdout.write('hallo\n')

"""
import getopt, sys
#def main():
if __name__ == '__main__':
    usage= "pd.read_hdf(FileInF,  'table',  where  =  ['index>2'])"
    try:
        opts, pargs = getopt.getopt(sys.argv[1:], 'FileInF:',
                                    ['where=',
                                     ])
     # default options
        where = None
        # Get the options
        for option in opts:
            if option[0] == '-h':
                sys.stderr.write(usage)
                sys.exit(0)
            elif option[0] == '--where':
                where = option[1]
        # if we pass a number of files different from 1, abort
        if len(pargs) != 1:
            print "You need to pass source filename!."
            sys.stderr.write(usage)
            sys.exit(0)

    except:
        (type, value, traceback) = sys.exc_info()
        print "Error parsing the options. The error was:", value
        sys.stderr.write(usage)
        sys.exit(0)

    # default options
"""
