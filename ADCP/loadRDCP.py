#!/usr/bin/env python
# coding:utf-8
"""
  Author:  Andrey Korzh --<ao.korzh@gmail.com>
  Purpose: add magnetic declination constant to specified files
  Created: 08.11.2015
"""
# For exported files from RDCP Studio:
### Configuration for this script and comments may be found in loadRDCP.ini

#####################################################################################################
import sys
from os import path as os_path
from datetime import datetime, timedelta
from codecs import open
import warnings

from magneticDec import magDec
from ..utils2init import ini2dict

loadRDCP_INI = os_path.join(os_path.dirname(sys.argv[0]), 'loadRDCP.ini')
# Path to this file may be overwrited here (only one line after first "\" will be used):
# loadRDCP_INI = r'\
# d:\workData\_source\KaraSea\150816_Kartesh-river_Ob\ADVS_RDI#17937 — копия\magneticDec.ini
# \\\
# '
# loadRDCP_INI = first_of_paths_text(loadRDCP_INI) #Get only first path from strDir
#
cfg = ini2dict(loadRDCP_INI)
strCmdPattern = ' '.join((cfg['Geomag']['path'], cfg['Geomag']['pathmodel'], '{0} D M{1} {2} {3}'))


# isertDeclination
def repInFile(nameFull, cfg, result):  # result is previous or with ['nameNavFull']= None, ['nav1D']= None
    # replace text in file nameFull
    # cfg['re mask'] is compiled patterns of re modeule to search:
    #  ['sourceDate'] - date from nameFull
    #  
    nameD, nameFE = os_path.split(nameFull)

    # 1. Get date from file name:
    m = cfg['re mask']['date'].match(nameFE);
    bTimeInName = False
    if (not m is None) and (not m.group(1) is None):  # and len(m.groups())>0:
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
                    if (not m is None) and (not m.group(1) is None):
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
                magDec(round(result['Lat'], 10), round(result['Lon'], 10), result['Date'].strftime("%Y,%m,%d")), 3)

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


if __name__ == '__main__':
    from utils2init import dir_walker, readable, bGood_dir, bGood_file
    from _other.mat73_to_pickle import read_mat73, datetime2matlab
    from numpy import searchsorted as np_searchsorted

    sys.path.append(r'd:\Work\_Python\_fromMat')
    ''' Filter used directories and files '''
    filt_dirCur = lambda f: readable(f) and bGood_dir(f, namesBadAtEdge=(r'bad', r'test'))
    filt_fileCur = lambda f, mask: bGood_file(f, mask, namesBadAtEdge=(r'coef'))

    print('found: ', end='')
    namesFull = [f for f in dir_walker(
        cfg['Veusz files']['dir'],
        cfg['Veusz files']['filemask'],
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
                for nameFull in namesFull:
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
    out= magDec(geoLat, geoLon)
    if out:
      sys.stdout.write(str(out))
      
"""
