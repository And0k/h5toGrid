#!/usr/bin/env python
# coding:utf-8
from __future__ import print_function, division

from to_pandas_hdf5.h5toh5 import h5sort_pack

"""
  Author:  Andrey Korzh --<ao.korzh@gmail.com>
  Purpose: Convert source text files of AB SIO RAS probes such as "Termochain",
  "Inclinometr" to PyTables hdf5 file
  Created: 26.02.2016
"""

import numpy as np
import pandas as pd
# import os
from os import path as os_path
# from future.moves.itertools import zip_longest
from datetime import datetime, timedelta
from codecs import open
import re

from other_filters import find_sampling_frequency

# import debug
# from debug import debug_print
# from  pandas.tseries.offsets import DateOffset

dt64_1s = np.int64(1e9)


def nameProbe(fileInF, reProbeInFileName=None):
    fileInP, fileInN = os_path.split(fileInF)
    if reProbeInFileName:
        strFound = reProbeInFileName.findall(fileInN)
        if len(strFound) > 0:  # use part of base name
            strProbe = strFound[0]
        else:  # use part of path name
            strFound = reProbeInFileName.findall(fileInP)
            if len(strFound) > 0:
                strProbe = strFound[0]
            else:
                return '?' + fileInN  # undefined
    else:
        # Probe name should be after "#" in file name or anywere in path otherwise
        iSt = fileInN.rfind('#')
        if iSt >= 0:  # use part of base name after '#'
            iSt += 1
            m = re.match(r'[^_\.]+', fileInN[iSt:])
            if (not m is None) and (not m.group(1) is None):
                strProbe = m.group(1)
            else:
                strProbe = '#' + fileInN  # undefined
        else:  # find number in path
            iSt = fileInP.rfind('#')
            if iSt >= 0:  # ok -> use part of path name after '#'
                iSt += 1
                m = re.match(r'[^_\.\\]+', fileInP[iSt:])
                if (not m is None) and (not m.group(1) is None):
                    strProbe = m.group(1)
                else:
                    strProbe = '#' + fileInN  # undefined
            else:  # use starting part of base name
                m = re.match(r'[^_\.]+', fileInN)
                if (not m is None) and (not m.group(1) is None):
                    strProbe = m.group(1)
                else:
                    strProbe = fileInN  # undefined

    # Format name.
    # First characters must be letters - if not then add i ("inclinometer")
    m = re.match('[a-zA-Z]+', strProbe)
    if m is None:
        strProbeSt = 'i'
        m = re.search(r'\d+', strProbe)
        numProbe = int(m.group(0))
    else:
        strProbeSt = m.group(0)
        m = re.search(r'\d+', strProbe[m.end(0):])
        numProbe = int(m.group(0))
    return (strProbeSt.lower() + "{:02d}".format(numProbe), numProbe)


def nameFileOut(fileInF, strProbe='', fileOutF='src.h5'):
    if not '\\' in fileOutF:
        fileOutF = os_path.join(os_path.split(fileInF)[0], strProbe + fileOutF)
    elif os_path.isdir(fileOutF):
        fileOutF = os_path.join(fileOutF, 'src.h5')
    return fileOutF


# Detect 1st element of pattern:
bSpikeShift = lambda bBadU, bBadD, Shift: np.hstack((False, np.logical_or(
    np.logical_and(bBadD[Shift:], bBadU[:-Shift]),  # spyke below
    np.logical_and(bBadU[Shift:], bBadD[:-Shift])),  # spyke above
                                                     np.zeros(Shift, dtype=np.bool)))


def bSpikeXpoints(a, max_spyke, max_points=1, points=None):
    # a: array to filter
    # max_spyke: int or array_like (only first 2 elements will used)
    # max_points: int -
    # returns bool arrray of detected spyke regions
    #
    # Picture. Example for Xpoints= 2
    # detect pattern: __--__   --__--
    # diff(pattern):   _-_ _    _ _-_
    #                     -      -
    # bBadD            ___-_    _-___
    # bBadU            _-___    ___-_
    #                   012      012
    diffX = np.diff(a)
    if np.size(max_spyke) > 1:
        bBadD = diffX < -max_spyke[0]
        bBadU = diffX > max_spyke[1]
    else:
        bBadD = diffX < -max_spyke
        bBadU = diffX > max_spyke

    bBad = np.zeros_like(a, dtype=np.bool)
    if max_points > 1:
        for Shift in np.arange(max_points, 0, step=-1):
            if Shift < max_points:  # cumilate long spykes regions by stretch them forward
                bBad[1:] = np.logical_or(bBad[1:], bBad[:-1])
            # Add point of pattern of spyke width = Shift at start of each pattern
            bBad[1:-Shift] = np.logical_or(bBad[1:-Shift], np.logical_or(
                np.logical_and(bBadU[:-Shift], bBadD[Shift:]),  # spyke above
                np.logical_and(bBadD[:-Shift], bBadU[Shift:])))  # spyke below
    if points > 1:
        bBadD = bSpikeShift(diffX < -max_spyke, diffX > max_spyke, points)
        # Mark the rest spyke elements
        for i in np.arange(points - 1):
            bBadU = np.hstack((False, bBadD[:-1]))
            bBadD = np.logical_or(bBadD, bBadU)
        bBad = np.logical_or(bBad, bBadD)
    return bBad


def bSpike1point(a, max_spyke):
    diffX = np.diff(a)

    bSingleSpike_1 = lambda bBadU, bBadD: np.logical_or(
        np.logical_and(np.append(bBadD, True), np.hstack((True, bBadU))),  # spyke to down
        np.logical_and(np.append(bBadU, True), np.hstack((True, bBadD))))  # spyke up

    return bSingleSpike_1(diffX < -max_spyke, diffX > max_spyke)


def rem2mean(x, bOk):
    return np.interp(np.arange(len(x)), np.flatnonzero(bOk), x[bOk], np.NaN, np.NaN)


def rem2mean_bBad(x, bBad):
    bOk = ~bBad
    return np.interp(np.flatnonzero(bBad), np.flatnonzero(bOk), x[bOk], np.NaN, np.NaN)


# ----------------------------------------------------------------------
def time_res1s_inc(tim, freq, tGoodSt=None):
    """
    Increase 1s time resolution with checking spykes and repeated time blocks
    :param: tim - time with resolution 1s
    :param: freq - fequency
    :param: tGoodSt - correct first time value
    :return: correct time

    # global variables:
    """
    global dt64_1s  # np.int64(1e9)
    dt64i = np.int64(dt64_1s / freq)  # int64, delta time of 1 count
    freqInt = np.int64(freq)
    if tGoodSt is None:
        tGoodSt = tim[0]

    d = np.ediff1d(tim, to_begin=0, to_end=dt64_1s)  # how value chenged
    iDiff = np.append(0, np.flatnonzero(d))  # where it changed
    dd = np.diff(iDiff)  # lengths of intervals with same values

    # for normal intervals:
    bGood = dd == freqInt
    ind = iDiff[bGood] + freqInt  # ends of good intervals
    bGood[bGood] = d[ind] >= dt64_1s  # check: next interval => more than 1s
    ind = iDiff[bGood]  # starts of good intervals
    for dt in np.arange(dt64i, dt64_1s, dt64i, dtype=np.int64):
        ind += 1
        if ind[-1] >= tim.size:
            ind[-1] = ind[-2]  # think faster than remove
        tim[ind] += dt

    # for other intervals:
    bGood = ~bGood
    ind = iDiff[bGood]  # starts of intervals with length~=freq_int
    if ind[0] == 0: d[0] = tim[0] - tGoodSt  # fist interval first diff
    for iSt, iEn, dCur in zip(ind, ind + dd[bGood], dd[bGood]):
        # spyke checking of tim[iSt]
        tPrev = (tim[iSt - 1] if iSt else tGoodSt)
        if dCur == 1 or d[iSt] <= 0:  # tim[iSt] or next val is spyke or it from next second
            dt = d[iSt] + d[iEn]
            if 0 < dt <= dt64_1s:  # next not spyke/other period
                tSt = tim[iEn] - dCur * dt64i
            elif dt == 0:  # next is same interval
                tSt = tim[iEn]
            else:  # spyke => use prev
                tSt = tPrev + dt64i
                # if tSt==0 can not use tim[iSt-1], can use tGood[-1]
            if tSt < tPrev: tSt = tPrev + dt64i  # check assignment
        else:  # have repeated information
            tSt = tim[iSt]
            if tSt < tPrev: tSt = tPrev + dt64i  # check assignment
            if dCur > freqInt:  # too long interval
                dt = tSt - tPrev  # (not used d[iSt] because not updated)
                # discard values after iSt+freqInt
                tim[(iSt + freqInt):iEn] = tSt + (freqInt - 1) * dt64i
                # recover i values if can shift interval time back
                if dt64_1s >= dt > dt64i:  # not new period and have time
                    i = min(int(dt / dt64i) - 1, dCur - freqInt)  # shift counts
                    dCur = freqInt + i
                    tSt = tSt - dt64i * i
                else:
                    dCur = freqInt
                iEn = iSt + dCur
        tim[iSt:iEn] = np.arange(tSt, tSt + dCur * dt64i, dt64i, np.int64)
    return tim


def time_ideal(tim, dt64i, tGoodSt, tRange=None, dt_period=None, dtBigHole=None):
    """
    Create ideal time vector tGood which exclude big holes by shift tRange
    :param tim:         time, may be not regular
    :param dt64i:       period between counts
    :param tGoodSt:     correct first time value
    :param tRange:      equally spaced vaues of current full work time interval
    :param dt_period:
    :param dtBigHole:   min interval considered as big (holes bigger dtBigHole will exludeded)
    :return:  (ideal time, next tRange)
    """

    d = np.diff(tim)  # how time chenged
    idiff = np.nonzero(d >= dtBigHole)[0]  # where data fin before hole
    d = tim[np.hstack((idiff - 1, -1))]  # time data fin
    if tim[0] - tGoodSt > dtBigHole:  # add 1st diff if just after hole
        d = np.hstack((tGoodSt, d))
        idiff = np.hstack((-1, idiff))
    tGoodNext = tGoodSt + dt64i
    tGood = np.array([], np.int64)  # ideal time init for this chunk
    for tEn, tSt in zip(d, np.hstack((tim[idiff + 1], 0))):  # Fill tGood
        # tEn - time data finish before hole
        # tSt - time data start after hole
        if tEn > tRange[-1]:
            # use prev peace of tRange:
            tGood = np.hstack((tGood, np.arange(tGoodNext, \
                                                tRange[-1] + dt64i, dt64i, dtype=np.int64)))
            tRange += dt_period  # make always tRange[0] < tGood[-1] <= tRange[-1]
            while tRange[-1] < tEn:  # to the last readed value
                tGood = np.hstack((tGood, tRange))
                tRange += dt_period
            else:  # create last peace:
                tGood = np.hstack((tGood, np.arange(tRange[0], \
                                                    tEn + dt64i, dt64i, dtype=np.int64)))
        else:  # add peace < tRange
            tGood = np.hstack((tGood, np.arange(tGoodNext,
                                                tEn + dt64i, dt64i, dtype=np.int64)))

        if tSt > 0:
            # move tRange to cross start after hole
            tRange += dt_period * np.int64((tSt - tRange[0]) / dt_period)
        tGoodNext = tRange[0]
    return tGood, tRange


# ## Main circle ############################################################
def sourceConvert(config, fileInF=None, fileOutF='src.h5'):
    if 'metadata' in config:
        cfg = config['metadata']
    if fileInF is None:
        fileInF = config['in']['path']
    if 'output_files' in config:
        if 'path' in config['output_files']:
            fileOutF = config['output_files']['path']
    if not (('in' in config) and ('re_probename' in config['in'])):
        config['in']['re_probename'] = None
    strProbe, numProbe = nameProbe(fileInF, config['in']['re_probename'])
    fileOutF = nameFileOut(fileInF, strProbe, fileOutF)

    config['columns']['header_in'] = config['columns']['header_in'].strip().split()
    config['columns']['header_out'] = config['columns']['header_out'].strip().split()

    if not 'b_filter_time' in cfg:
        cfg['b_filter_time'] = False
    if not 'time_Sincro' in cfg:
        cfg['time_Sincro'] = 0  # start time, if 0 - will not correct
    if not 'freq_hz' in cfg:
        cfg['b_filter_time'] = False
        freq = 1.0 / 30  # Hz
    else:
        cfg['b_filter_time'] = True
        if cfg['freq_hz'] == 'constant':
            freq = np.NaN
        else:
            freq = cfg['freq_hz']
    if not 'dt_rec' in cfg:
        cfg['b_make_time_regular'] = False
        cfg['dt_rec'] = 60 * (60 * dt64_1s)  # min*(), measure and record operation
        cfg['dt_period'] = 60 * (60 * dt64_1s)  # min*(), work cycle
    elif not ('b_make_time_regular' in cfg and not cfg['b_make_time_regular']):
        cfg['b_make_time_regular'] = True
    if not 'TimeAdd' in cfg:
        cfg['TimeAdd'] = timedelta(0)

        # cfg['dt_rec'].
        # Converions:
        cfg['dt_rec'] = np.int64(cfg['dt_rec'].total_seconds()) * dt64_1s  # int64, delta time of operation
        cfg['dt_period'] = np.int64(cfg['dt_period'].total_seconds()) * dt64_1s  # int64, delta time of Period
    if 'dt_big_hole' in cfg:
        dtBigHole = np.int64(cfg['dt_big_hole'].total_seconds()) * dt64_1s
    else:
        dtBigHole = cfg['dt_period'] * 2
    cfg['TimeAdd'] = np.int64(config['TimeAdd'].total_seconds()) * dt64_1s  # convert to int64

    if not np.isnan(freq):
        chunksize = int((cfg['dt_rec'] * freq * 1e3) / (cfg['dt_period'] * dt64_1s))
    else:
        chunksize = 60000
    if chunksize < 50000:    chunksize *= int(60000 / chunksize)

    # store= pd.HDFStore(fileOutF)
    with pd.get_store(fileOutF) as store, open(fileInF, 'r') as f:
        # try:
        if strProbe in store: store.remove(strProbe)
        #    pass
        # except:
        #    pass

        # number of columns and Time columns
        Ncols = len(config['columns']['header_in'])
        Ncols_t = np.sum([colName[:4] in ('Time', 'Date') for colName in config['columns']['header_in']])
        b1stChunk = True
        maxMonth = min(datetime.now().month, 12)
        iChunk = 0
        n_processed = 0
        for strChunk in zip_longest(*[f] * chunksize):
            iChunk += 1
            print('.', end='')

            iDel = []
            a = np.zeros([chunksize, Ncols], np.int16)
            for i, s in enumerate(strChunk):
                if s is None:
                    if strChunk.count(None) + i == chunksize:  # tipically s is None at end of file
                        a = a[:i, :]
                        break
                    continue
                try:
                    a[i] = np.fromstring(s, dtype=np.uint16, sep="\t")
                except TypeError:
                    iDel.append(i)
                    # strChunk= strChunk[:i] + filter(None, strChunk[i+1:])
                except ValueError:  # IndexError:
                    n = s.count('\t')
                    if n >= Ncols_t - 1:
                        if n >= Ncols: n = Ncols
                        try:
                            a[i][:n] = np.fromstring(s, dtype=np.uint16,
                                                     sep="\t")[:n]
                        except ValueError:
                            iDel.append(i)
                    else:
                        iDel.append(i)
                except:
                    raise ('unexpected error')
            # good year
            bBadY = np.logical_or(a[:, 0] < 2000, a[:, 0] > datetime.now().year)
            # bad month
            bBad = np.logical_or(a[:, 1] < 1, a[:, 1] > maxMonth)
            for i in (0, 1):
                if i == 0:
                    # correct bad year where good month:
                    bGood = np.logical_and(~bBad, bBadY)  # actually bad
                else:
                    # correct bad month where good year:
                    bGood = np.logical_and(bBad, ~bBadY)  # actually bad
                row = np.nonzero(bGood)[0] - 1  # use previous indexes
                while True:
                    b = row < 0  # check that we after start
                    row[b] = bGood.argmin()  # corret to first idex of bGood==False
                    b = bGood[row]  # check bad values
                    if ~any(b): break  # no bad :)
                    row[b] -= 1  # corret to previous
                a[bGood, i] = a[row, i]

            # additionally delete rows with both bad year and month
            iDel = iDel + list(np.nonzero(np.logical_and(bBadY, bBad))[0])
            if iDel:
                a = np.delete(a, iDel, axis=0)
                print('N(del.rows)=', len(iDel), 'first row #', (iChunk - 1) * chunksize + iDel[0])
                if s:
                    try:
                        print(s[iDel[0]])
                    except:
                        raise ('can not print bad string')
            n = a.shape[0]
            tim = np.array(np.ones(n), datetime)  # np.empty((r,c),dtype=np.object)
            for i, row in enumerate(a[:, :Ncols_t]):
                try:
                    tim[i] = datetime(*row.tolist())
                except ValueError:
                    if i > 0:
                        tim[i] = tim[i - 1]
                    else:  # seldom
                        while i < n:
                            i += 1
                            try:
                                tim[i] = datetime(*a[i, :Ncols_t].tolist())
                                break
                            except ValueError:
                                print('T!')
                                pass
            chunk = pd.DataFrame(a[:, Ncols_t:].view(dtype=np.uint16),
                                 columns=config['columns']['header_in'][Ncols_t:],
                                 dtype=np.uint16,
                                 index=tim + config['TimeAdd'])
            chunk = chunk.reindex(columns=config['columns']['header_out'])

            ## Filter time
            if cfg['b_filter_time']:
                # # Correct repiated time values (may be many if freq>1 because of reslution is 1s)
                tim = chunk.index.values.view(dtype=np.int64)
                # - filter time:
                bBad = bSpikeXpoints(tim, max_spyke=1,
                                     max_points=3)  # max_spyke should be bigger than intrval to prevent find spykes to bottom?
                if np.any(bBad):
                    nDecrease = np.sum(bBad)
                    print(str(nDecrease) + ' time spykes detected in rows#: ' +
                          str(n_processed + np.flatnonzero(bBad)) + ' => interpolating...')
                    tim[bBad] = rem2mean_bBad(tim, bBad)

                if b1stChunk:
                    if np.isnan(freq): freq = find_sampling_frequency(tim)[0]
                    # Converions:
                    dt64i = np.int64(dt64_1s / freq)  # int64, delta time of 1 count

                    if cfg['time_Sincro'] > 0:
                        tGoodNext = cfg['time_Sincro'] + cfg['dt_period'] * np.ceil(
                            (tim[0] - cfg['time_Sincro']) / cfg['dt_period']).astype(np.int64)
                        if tGoodNext != tim[0]:
                            print('otputut sinhronised to be started from {0}',
                                  str(tGoodNext.astype('<M8[ns]')))
                    else:
                        tGoodNext = tim[0]
                    tGood = np.array([tGoodNext - dt64i])  # ideal time previous init
                    #
                    tRange = np.arange(tGoodNext, tGoodNext + cfg['dt_rec'], dt64i, dtype=np.int64)

                tim = time_res1s_inc(tim, freq, tGoodSt=tGood[-1])
                #
                chunk.index = pd.DatetimeIndex(tim, copy=False)  # need update if tim became a copy
                chunk = chunk.groupby(level=0).last()

                if cfg['b_make_time_regular']:
                    tGood, tRange = time_ideal(tim, dt64i, tGoodSt=tGood[-1],
                                               tRange=tRange, dt_period=cfg['dt_period'], dtBigHole=dtBigHole)
                    chunk = chunk.reindex(index=pd.DatetimeIndex(tGood, copy=False),
                                          fill_value=0, copy=False)  # , tolerance= dt64_1s

            # end of filter time --------------------------------------------------------
            if b1stChunk:
                log['Date0'] = chunk.index[0]
                b1stChunk = False
            else:
                n_processed += n
            store.append(strProbe, chunk[chunk.columns].astype('int32'),
                         data_columns=True, index=False)  # chunk.columns
            # ignore_index=True, chunksize= cfg['dt_rec']/dt64i,
        store.create_table_index(strProbe, columns=['index'], kind='full')
        # df2.duplicated([’a’,’b’])
        # df2.drop_duplicates([’a’,’b’], take_last=True)
        log['DateEnd'] = chunk.index[-1]
    # end of with store ( so no store.close() )

    if True:  # / False
        FileCum = os_path.basename(os_path.dirname(fileInF)) + '.h5'
        h5sort_pack(fileOutF, FileCum, strProbe)
        # store.remove(strProbe)
        # getstore_and_print_table(FileCum, strProbe)

    print("Display time graph of last chank")
    from matplotlib import pyplot as plt
    plt.plot(tim)
    plt.show()
    return (log)


if __name__ == '__main__':
    #    unittest.main()
    from utils2init import ini2dict, pathAndMask, dir_walker, readable, bGood_dir, bGood_file

    cfg = ini2dict()  # r'd:\Work\_Python\_fromMat\loadInclinometer.ini'
    cfg['in']['path'], cfg['in']['filemask'] = pathAndMask(*[cfg['in'][spec]
                                                             if spec in cfg['in'] else None for spec in
                                                             ['path', 'filemask', 'ext']])
    # Filter unused directories and files
    filt_dirCur = lambda f: readable(f) and bGood_dir(f, namesBadAtEdge=(r'bad', r'test'))
    filt_fileCur = lambda f, mask: bGood_file(f, mask, namesBadAtEdge=(r'coef'))

    print('found: ', end='')
    namesFull = [f for f in dir_walker(
        cfg['in']['path'],
        cfg['in']['filemask'],
        bGoodFile=filt_fileCur,
        bGoodDir=filt_dirCur)]
    nFiles = len(namesFull)
    if nFiles == 0:
        print('(0 files) => nothing done')
        # exit?
    else:
        s = raw_input('\n(' + str(nFiles) + r' files). Process it? Y/n: ')
        if 'n' in s or 'N' in s:
            print('nothing done')
        else:
            print('wait... ', end='')
            log = {'nameNavFull': None,
                   'nav1D': None}
            if True:  # try:
                if 'log' in cfg['program'].keys():
                    f = open(cfg['program']['log'], 'a+', encoding='cp1251')
                    f.writelines(datetime.now().strftime('\n\n%d.%m.%Y %H:%M:%S> processed '
                                                         + str(nFiles) + ' file' + 's:' if nFiles > 1 else ':'))
                for nameFull in namesFull:
                    nameFE = os_path.basename(nameFull)
                    print(nameFE, end=': ')
                    # result= repInFile(nameFull, cfg, result)
                    log = sourceConvert(cfg, nameFull)
                    strLog = '{Date0:%d.%m.%Y %H:%M:%S}-{DateEnd:%H:%M:%S%z}'.format(
                        **log)  # \t{Lat}\t{Lon}\t{strOldVal}->\t{mag}
                    print(strLog)
                    if 'log' in cfg['program'].keys():
                        f.writelines('\n' + nameFE + '\t' + strLog)
            try:
                print('ok')
            except Exception as e:
                print('The end. There are errors: ', e.message)
            finally:
                if 'log' in cfg['program'].keys():
                    f.close()
