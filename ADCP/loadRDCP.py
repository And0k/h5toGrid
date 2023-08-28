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

from magneticDec import mag_dec
from utils2init import ini2dict, dir_walker, readable, bGood_dir, bGood_file


from pathlib import Path
import numpy as np

from grid2d_vsz import save_shape, to_polygon


def main():

    a1d = load_rdcp_aux(
        r'd:\WorkData\BalticSea\_Pregolya,Lagoon\210519-inclin+RDCP\RDCP\txt\TxtExportAuxSensors.txt'
        )



    print(a1d)


def load_rdcp_aux(file_in, file_out=None, delimiter='\t'):
    """
    loads RDCP Aux Sensors txt file of format:
    Date - Time Battery Heading Pitch Roll Reference Temperature Conductivity Oxygen 3835 4017 Turbidity 3612 Depth Salinity Speed of sound
    2021-05-19 10:19:51 11.99 62.69 -35.179 -26.823 692.000 17.225 0.308 70.320 199.219 6.539 2012.194 0.170 1507.2

    :param file_bln_in:
    :param file_bln_out:
    :param delimiter: if not ',' useful to invert 1st column of text files of other types
    :return:
    """
    if not file_out:
        p_in = Path(file_in)
        file_out = p_in.with_name(f'{p_in.stem}_out').with_suffix(p_in.suffix)
    with open(file_in, 'rb') as f:
        header = f.readline()
        # gsw_z_from_p()
        # a1D.P_dBar = a1D.a4017 / 10;
        # a1D.O2 = a1D('Oxygen 3835');
        col_names = 'Time Battery Heading Pitch Roll Reference Temperature Conductivity O2 P_dBar Turbidity Depth Salinity SoundV'.split()
        n_float_cols = len(col_names) - 1
        formats = ['M8[s]'] + ['f4'] * n_float_cols
        dtype = {
            'names': col_names,
            'formats': formats
        }
        a1d = np.loadtxt(f, dtype=dtype, skiprows=0, delimiter=delimiter)
    a1d['P_dBar'] = a1d['P_dBar'] / 10 - 11.1

    p_min = 0
    p_max = 15
    b_good = (p_min < a1d['P_dBar']) & (a1d['P_dBar'] < p_max)

    b_excel_time = True
    if b_excel_time:
        excel_dates_offset_s  = np.int64(np.datetime64(datetime(1899, 12, 30), 's'))
        dtype['formats'][0] = 'f8'
        a1d_float_time = (np.int64(a1d['Time'].astype('M8[s]')) - excel_dates_offset_s) / (24 * 3600)  # days, Excel time
        a1d = a1d.astype(dtype)
        a1d['Time'] = a1d_float_time
        np.savetxt(file_out, a1d,
                   fmt='\t'.join(['%.10f'] + ['%g']*n_float_cols),
                   delimiter=delimiter,
                   header='\t'.join(col_names),
                   comments='',
                   encoding='ascii'
                   )

        p_max_show = 20
        save_shape(
            file_out.with_name(f'{p_in.stem}_P'),
            to_polygon(a1d_float_time[b_good], a1d['P_dBar'][b_good], p_max_show),
            'BNA'
            )

        np.savetxt(file_out.with_name(f'{p_in.stem}_P').with_suffix('.txt'), a1d[['Time', 'P_dBar']][b_good],
                   fmt='\t'.join(['%.10f'] + ['%g']),
                   delimiter=delimiter,
                   header='\t'.join(col_names),
                   comments='',
                   encoding='ascii'
                   )
    else:
        np.savetxt(file_out, a1d,
                   fmt='\t'.join(['%s'] + ['%g']*n_float_cols),
                   delimiter=delimiter,
                   header='\t'.join(col_names),
                   comments='',
                   encoding='ascii'
                   )


def repInFile(nameFull, cfg, result):  # result is previous or with ['nameNavFull']= None, ['nav1D']= None
    """
    replaces (inserts?) declination text in file nameFull

    :param nameFull:
    :param cfg:
        - 're mask': compiled patterns of re modeule to search ['sourceDate'] - date from nameFull
    :param result:
    :return:
    """

    sys.path.append(r'd:\Work\_Python\_fromMat')
    from _other.mat73_to_pickle import read_mat73, datetime2matlab

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
                ix = np.searchsorted(result['nav1D']['Time'], datetime2matlab(result['Date']))
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

def old_repInFile_for_vszs():
    """add magnetic declination constant to specified files
    """

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

    # Filter used directories and files
    filt_dirCur = lambda f: readable(f) and bGood_dir(f, namesBadAtEdge=(r'bad', r'test'))
    filt_fileCur = lambda f, mask: bGood_file(f, mask, namesBadAtEdge=(r'coef'))

    print('found: ', end='')
    paths = [f for f in dir_walker(
        cfg['Veusz files']['path'].parent,
        cfg['Veusz files']['path'].name,
        bGoodFile=filt_fileCur,
        bGoodDir=filt_dirCur)]
    nFiles = len(paths)
    if nFiles == 0:
        print('(0 files) => nothing done')
        # exit?
    else:
        s = input('\n(' + str(nFiles) + r' files). Process it? Y/n: ')
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


if __name__ == '__main__':
    main()

"""    
    #Test1:
    geoLat= 44; geoLon= 55
    out= mag_dec(geoLat, geoLon)
    if out:
      sys.stdout.write(str(out))
      
"""
