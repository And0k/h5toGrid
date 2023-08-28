#!/usr/bin/env python
# coding:utf-8


"""
  Author:  Andrey Korzh --<ao.korzh@gmail.com>
  Purpose: add magnetic declination constant to specified files
  Created: 08.09.2015
"""
# For each *.vsz file:
# 1. Get date from file name (defined by input data mask) to get nav(time) and mag_dec(..., time)
# 2. Get name of measuring device data in *.vsz code:
# 2a. if need name of source bin measuring device data file, create it from name of (usually txt) source for veusz
# 3. Get nav file name in *.vsz code
# 4. Find coordinates for date(step1-2a) in nav file (step3)
# 5. Get position of Magnetic Declination value in *.vsz code:
# 6. Get and replace Magnetic Declination
### Configuration for this script and comments may be found in magneticDec.ini

#####################################################################################################
import sys
from os import path as os_path
from datetime import datetime, timedelta
from codecs import open
import warnings

from typing import Iterable
import datetime
import wmm2020 as wmm


def year_fraction(date: datetime.datetime) -> float:
    """
    datetime dates to decimal years
    https://stackoverflow.com/a/36949905/2028147
    :param date:
    :return:
    Note: calculates the fraction based on the start of the day, so December 31 will be 0.997, not 1.0.

    >>> print year_fraction(datetime.datetime.today())
    2016.32513661
    """
    start = datetime.date(date.year, 1, 1).toordinal()
    year_length = datetime.date(date.year+1, 1, 1).toordinal() - start
    return date.year + float(date.toordinal() - start) / year_length


def mag_dec(lat, lon, time: datetime.datetime, depth: float = 0):
    """
        Returns magnetic declination using wmm2020 library
        
    :param lat, lon: coordinates in degrees WGS84
    :param time: # ='2020,09,20'
    :param depth: in meters (negative below sea surface)
    """

    yeardec = year_fraction(time)
    mag = wmm.wmm(lat, lon, depth/1000, yeardec)
    # .item()
    return mag.decl if (isinstance(lat, Iterable) or isinstance(lon, Iterable)) else mag.decl.item(0)


# Next writes value to file but it must be slow to read for 1 value compared to previous method,
# but it is documented:
# 'f' switch: converting file with multiple locations.
# The first five output columns repeat the input coordinates.
# Then follows D, I, H, X, Y, Z, and F.
# Finally the SV: dD, dI, dH, dX, dY, dZ,  and dF
# The units are the same as when the program is
# run in command line or interactive mode.

# from tempfile import NamedTemporaryFile
# fIn=   NamedTemporaryFile(delete=False) #([mode='w+b'[, bufsize=-1[, suffix=''[, prefix='tmp'[, dir=None[, delete=True]]]]]])
# fIn.write("2009.5 D K100  70.3  30.8\r\n")
# fIn.close()
# fOut=  NamedTemporaryFile(delete=False)
# fOut.close()
# txt= subprocess_call(' '.join((cfg['Geomag']['path'], cfg['Geomag']['pathmodel'], 'f', fIn.name, fOut.name)))
# with open(fOut,'r') as f:
# for line in fOut: #import os; fOut.flush(); os.fsync(fOut.fileno())
# print(line)


def magDecFile(geomagInputFile, geomagOutputFile, stdOutRedirection='>NUL', strCmdPattern=None):
    warnings.warn('not implemented')
    # Decode many
    # geomag is limited to path length of 92 characters
    if length(geomagOutputFile) > 92:
        error('magneticDeclinationPP cannot be applied on ' + geomagInputFile + \
              '. Change your toolbox location so that ' + geomagOutputFile + \
              ' is shorter than 92 characters long (Geomag limitation).')
    # end

    # we run the geomag program and read its output
    # stdOutRedirection = ''
    geomagCmd = sprintf('%s %s f %s %s %s', cfg['Geomag']['path'], cfg['Geomag']['pathmodel'], geomagInputFile,
                        geomagOutputFile, stdOutRedirection)
    system(geomagCmd)

    # 2. Create combined file with the name of moved 1st file
    if not os_path.isfile(nameFull0):  # Check that file actually have moved
        runString = strCmdPattern.format(nameFullDst,
                                         nameFull0)  # os_path.join(nameD, os_path.splitext(nameFE0)[0] + '.0000')
        subprocess.call(runString)
    else:
        s = raw_input('File "' + nameFull0 + '" will be overwrited! Continue?')
        if 'n' in s or 'N' in s:
            print('not all joined')
            exit
        else:
            print('continue...')

    geomagFormat = '%s D M%f %f %f %fd %fm \
%*s %*s %*f %*f %*f %*f %*f %*f %*f %*f %*f %*f %*f %*f' + endOfLine
    outputId = fopen(geomagOutputFile, 'r')
    geomagOutputData = textscan(outputId, geomagFormat, \
                                'HeaderLines', 1, \
                                'Delimiter', ' ', \
                                'MultipleDelimsAsOne', true, \
                                'EndOfLine', endOfLine)
    fclose(outputId)

    geomagDate = datenum(geomagOutputData[1], 'yyyy,mm,dd')
    geomagDepth = -geomagOutputData[2]
    geomagLat = geomagOutputData[3]
    geomagLon = geomagOutputData[4]
    signDeclin = sign(geomagOutputData[5])
    if signDeclin >= 0: signDeclin = 1  # end
    geomagDeclin = geomagOutputData[5] + signDeclin * geomagOutputData[6] / 60

    nMagDataSet = length(iMagDataSet)
    for i in range(nMagDataSet):
        isMagDecApplied = false
        magneticDeclinationComment = 'magneticDeclinationPP: data initially referring to magnetic North has \
    been modified so that it now refers to true North, applying a computed magnetic \
    declination of ' + str(geomagDeclin(i)) + 'degrees. NOAA''s Geomag v7.0 software + IGRF11 \
    model have been used to compute this value at a latitude=' + str(geomagLat(i)) + 'degrees \
    North, longitude=' + str(geomagLon(i)) + 'degrees East, depth=' + str(geomagDepth(i)) + 'm \
    (instrument nominal depth) and date=' + datestr(geomagDate(i), 'yyyy/mm/dd') + \
                                     ' (date in the middle of time_coverage_start and time_coverage_end).'
    data_mag = sample_data[iMagDataSet(i)].variables[j].data
    data = data_mag + geomagDeclin(i)


def indOf1stDiff(path_source, nameD):
    for i, d, s in zip(range(len(path_source)), nameD.lower(), path_source.lower()):
        if d != s:
            return (i)


# [None if d==s else i for i,d,s in zip(range(len()), .lower(), path_source.lower())]

# isertDeclination
def repInFile(nameFull, cfg, result):  # result is previous or with ['nameNavFull']= None, ['nav1D']= None
    # replace text in file nameFull
    # cfg['re mask'] is compiled patterns of re modeule to search:
    #  ['sourceDate'] - date from nameFull
    #  
    nameD, nameFE = os_path.split(nameFull)

    # 1. Get date from file name:
    m = cfg['re mask']['date'].match(nameFE)
    bTimeInName = False
    if (m is not None) and (m.group(1) is not None):  # and len(m.groups())>0:
        bFoundDate = True
        if m.group(2) is None:
            strDate = m.group(1) + '_0000'
        else:
            bTimeInName = True
            strDate = '_'.join(m.group(1, 2))
        DateFromName = datetime.strptime(strDate[:11], '%y%m%d_%H%M%S') + cfg['TimeAdd']
    else:
        if 'Date' in result.keys():
            warnings.warn('NOT FOUND DATE IN FILE NAME! USE TIME OF PREVIOUS FILE!')
        else:
            warnings.warn('NOT FOUND DATE IN FILE NAME! USE CURRENT TIME!')
            result['Date'] = datetime.now()
    with open(nameFull, 'r+b', encoding='utf-8') as f:  # , open(os_path.join(nameD, nameFull[]), 'r') as f:
        file_content = f.read()
        if 'fixed_manetic_declination' in cfg['Geomag'].keys():
            result['mag'] = cfg['Geomag']['fixed_manetic_declination']
        else:
            if 'fixed_lon' in cfg['Geomag'].keys():
                result['Lat'] = cfg['Geomag']['fixed_lat']
                result['Lon'] = cfg['Geomag']['fixed_lon']
            else:
                if not bTimeInName:
                    # 1.1 Get more accurate time from source file for veusz if exist
                    # a) Get source file name
                    m = cfg['re mask']['data'].search(file_content)
                    if (m is not None) and (m.group(1) is not None):
                        nameDatFE = m.group(1)
                        nameDatFull = os_path.join(nameD, nameDatFE)
                        # b) Get date from source binary file modification stamp:
                        if 'path' in cfg['bin files'].keys():
                            nameBinFull = os_path.join(cfg['bin files']['path'], nameDatFE)
                        elif 'path_source' in cfg['bin files'].keys():
                            ix = indOf1stDiff(cfg['bin files']['path_source'], nameD)
                            nameBinFull = os_path.join(cfg['bin files']['path_source'] + nameD[ix:], nameDatFE)
                        else:
                            nameBinFull = os_path.join(nameD, nameDatFE)
                        nameBinFull = nameBinFull[:-len(cfg['dat files']['ext'])] + cfg['bin files']['ext']
                        # nameBinFull, bOk= re.subn(cfg['dat files']['ext'] + '$',
                        # cfg['bin files']['ext'], nameDatFull)
                        timeStart = None
                        if readable(nameBinFull):
                            timeStart = datetime.fromtimestamp(os_path.getmtime(nameBinFull)) + cfg['TimeAddBin']
                            if abs(timeStart - DateFromName) < timedelta(days=1):
                                result['Date'] = timeStart
                            else:
                                warnings.warn(' source bin time further than 1 day! ')
                        else:
                            warnings.warn(' no source bin! ')

                    if timeStart is None:
                        if 'Date' in result.keys():
                            warnings.warn('NOT FOUND DATE from bin time stamp! USE TIME OF PREVIOUS FILE!')
                        else:
                            warnings.warn('NOT FOUND DATE from bin time stamp! USE CURRENT TIME!')
                            result['Date'] = datetime.now()
                else:
                    result['Date'] = DateFromName
                # 2. Search name of nav data file:
                m = cfg['re mask']['nav'].search(file_content)
                nameNavFE = m.group(1)
                nameNavFull = os_path.join(nameD, nameNavFE)

                # 3. Find coordinates for Date(step1) in nameNavFull(step2)
                if result['nameNavFull'] != nameNavFull:
                    result['nameNavFull'] = nameNavFull
                    result['nav1D'] = read_mat73('nav1D', result['nameNavFull'])
                ix = np_searchsorted(result['nav1D']['Time'], datetime2matlab(result['Date']))
                if ix >= len(result['nav1D']['Time']):
                    warnings.warn('last time in nav < data time')
                elif ix <= 0:
                    warnings.warn('first time in nav > data time')
                else:
                    result['Lat'] = result['nav1D']['Lat'][ix]
                    result['Lon'] = result['nav1D']['Lon'][ix]

            # 4. Get MagneticDeclination value
            result['mag'] = round(
                mag_dec(round(result['Lat'], 10), round(result['Lon'], 10), result['Date'].strftime("%Y,%m,%d")), 3)

        # 5. Replace MagneticDeclination:
        m = cfg['re mask']['mag'].search(file_content)  # m = cfg['re mask']['mag'].split(file_content,  maxsplit=1)
        if m is None:
            warnings.warn('not found position of replace pattern')
            result['strOldVal'] = 'None'
        else:
            result['strOldVal'] = m.group(1)
            f.seek(m.start(1), 0)
            f.write(str(result['mag']))
            f.write(file_content[m.end(1):])
            f.truncate()
            # nameNavFE= m.group(1)
        # new_content, ok= cfg['re mask']['mag'].subn(str(result['mag']), file_content, count=1)
        # if ok:
        # f.seek(offset, from_what)
        # f.write(new_content)
        # else:
        # warnings.warn('not found position of replace pattern')
    return (result)  # Lat, Lon, Date, strOldVal, mag, nameNavFull, nav1D)


def repInFiles(files, pattern, fgetVals):
    pass


if __name__ == '__main__':
    from utils2init import ini2dict, dir_walker, readable, bGood_dir, bGood_file
    from _other.mat73_to_pickle import read_mat73, datetime2matlab
    from numpy import searchsorted as np_searchsorted

    cfg = ini2dict()
    strCmdPattern = ' '.join((cfg['Geomag']['path'], cfg['Geomag']['pathmodel'], '{0} D M{1} {2} {3}'))

    sys.path.append(r'd:\Work\_Python\_fromMat')
    ''' Filter unused directories and files '''
    filt_dirCur = lambda f: readable(f) and bGood_dir(f, namesBadAtEdge=(r'bad', r'test'))
    filt_fileCur = lambda f, mask: bGood_file(f, mask, namesBadAtEdge=(r'coef'))

    print('found: ', end='')
    paths = [f for f in dir_walker(
        cfg['Veusz files']['dir'],
        cfg['Veusz files']['filemask'],
        bGoodFile=filt_fileCur,
        bGoodDir=filt_dirCur)]
    nFiles = len(paths)
    if nFiles == 0:
        print('(0 files) => nothing done')
        # exit?
    else:
        s = raw_input('\n(' + str(nFiles) + r' files). Process it? Y/n: ')
        if 'n' in s or 'N' in s:
            print('nothing done')
        else:
            print('wait... ', end='')
            # sys.stdout.write('hallo\n')
            print('wait')
            result = {'nameNavFull': None,
                      'nav1D': None}
            if True:  # try:
                if 'log' in cfg['program'].keys():
                    if not ('\\' in cfg['program']['log'] or \
                            r'/' in cfg['program']['log']):
                        cfg['program']['log'] = os_path.join(os_path.dirname(sys.argv[0]),
                                                             cfg['program']['log'])
                    f = open(cfg['program']['log'], 'a+', encoding='cp1251')
                    f.writelines(datetime.now().strftime('\n\n%d.%m.%Y %H:%M:%S> processed '
                                                         + str(nFiles) + ' file' + 's:' if nFiles > 1 else ':'))
                for nameFull in paths:
                    nameFE = os_path.basename(nameFull)
                    print(nameFE, end=': ')
                    #
                    result = repInFile(nameFull, cfg, result)
                    strLog = '\t{Lat}\t{Lon}\t{Date:%d.%m.%Y %H:%M:%S}\t{strOldVal}->\t{mag}'.format(**result)
                    print(strLog)
                    if 'log' in cfg['program'].keys():
                        f.writelines('\n' + nameFE + '\t' + strLog)
            try:
                print('ok')
            except Exception as e:
                print('The end. There are errors: ', e.message)
            finally:
                f.close()
"""    
    #Test1:
    geoLat= 44; geoLon= 55
    out= mag_dec(geoLat, geoLon)
    if out:
      sys.stdout.write(str(out))
      
"""
