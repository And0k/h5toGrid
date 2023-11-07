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
import pandas as pd

from ADCP.loadNortekSignature_txt import save_2d_for_surfer

try:
    from grid2d_vsz import save_shape, to_polygon
except ModuleNotFoundError as e:
    def get_dummies(e):
        return (
            lambda *args: print(f'Module to save shape was not imported ({e})'),
            lambda *args: print(f'Module to save polygon was not imported ({e})')
        )
    save_shape, to_polygon = get_dummies(e)
    
def main():
    path = r'd:\WorkData\BalticSea\230825_Kulikovo@ADCP,ADV,i,tr\RDCP\_txt(rds)\TxtExportAuxSensors.txt'
    # r'd:\WorkData\BalticSea\_Pregolya,Lagoon\210519-inclin+RDCP\RDCP\txt\TxtExportAuxSensors.txt'
    path_2d = Path(path).with_name('TxtExportCol1.txt')
    path_out = path_2d.parent.parent / '_srf,vsz' / '_'
    a2d = load_rdcp_profile(path_2d, path_out=path_out)
    a1d = load_rdcp_aux(path, path_out=path_out)
    print(a1d, a2d)


def load_rdcp_aux(file_in, path_out=None, delimiter='\t', modifier=None, filter=None):
    """
    loads RDCP Aux Sensors txt file of format:
    Date - Time Battery Heading Pitch Roll Reference Temperature Conductivity Oxygen 3835 4017 Turbidity 3612 Depth Salinity Speed of sound
    2021-05-19 10:19:51 11.99 62.69 -35.179 -26.823 692.000 17.225 0.308 70.320 199.219 6.539 2012.194 0.170 1507.2

    :param file_bln_in:
    :param file_bln_out:
    :param delimiter: if not ',' useful to invert 1st column of text files of other types
    :return:
    """
    file_in = Path(file_in)
    if not path_out:
        path_out = file_in.with_name(f'{file_in.stem}_out').with_suffix(file_in.suffix)
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
    
    if modifier:
        a1d['P_dBar'] = a1d['P_dBar'] / 10 - 11.1
    if filter:
        p_min = 0
        p_max = 15
        b_good = (p_min < a1d['P_dBar']) & (a1d['P_dBar'] < p_max)
    else:
        b_good = slice(None)

    b_excel_time = True
    if b_excel_time:
        excel_dates_offset_s = np.int64(np.datetime64(datetime(1899, 12, 30), 's'))
        dtype['formats'][0] = 'f8'
        a1d_float_time = (np.int64(a1d['Time'].astype('M8[s]')) - excel_dates_offset_s) / (24 * 3600)  # days, Excel time
        a1d = a1d.astype(dtype)
        a1d['Time'] = a1d_float_time
    
    np.savetxt(path_out, a1d,
               fmt='\t'.join(['%.10f'] + ['%g']*n_float_cols),
               delimiter=delimiter,
               header='\t'.join(col_names),
               comments='',
               encoding='ascii'
               )
        
    # Save lines for Surfer
    p_max_show = 20
    save_shape(
        path_out.with_name(f'{file_in.stem}_P'),
        to_polygon(a1d_float_time[b_good], a1d['P_dBar'][b_good], p_max_show),
        'BNA'
        )
    np.savetxt(path_out.with_name(f'{file_in.stem}_P').with_suffix('.txt'), a1d[['Time', 'P_dBar']][b_good],
               fmt='\t'.join(['%.10f'] + ['%g']),
               delimiter=delimiter,
               header='\t'.join(col_names),
               comments='',
               encoding='ascii'
               )
    
    
    # else:
    #     np.savetxt(file_out, a1d,
    #                fmt='\t'.join(['%s'] + ['%g']*n_float_cols),
    #                delimiter=delimiter,
    #                header='\t'.join(col_names),
    #                comments='',
    #                encoding='ascii'
    #                )


def load_rdcp_profile(file_in, path_out=None, delimiter='\t', modifier=None, filter=None):
    
    with open(file_in) as f:
        # Find the distance from the instrument to the center of the cells from line 1 in file.
        line1 = f.readline()
        z = np.float64(line1.split('\t')[1:])
    
        # Read column headers and determine the number of columns from line 2 in file.
        line2 = f.readline()
        column_headers = line2.split('\t')[1:]  # skip 1s that is Time
        n_columns = len(column_headers)
    
        # Read the rest of the file into a pandas DataFrame.
        data = pd.read_csv(f, sep='\t', header=None)
    
    # Organize the data in a pandas DataFrame
    # Account of selectd cells to export (modified 19.10.2017)
    cell_number_in_str_st = len('Direction_')
    icells = np.int32([c[cell_number_in_str_st:] for i, c in enumerate(column_headers) if c.startswith('Direction_')])
    z = z[icells - 1]
    n_cells = icells.size
    

    # Organize the data in structure
    
    # Time Matlab to numpy conversion (to seconds and adding matlab epoch start: datetime64('-001-12-31T00:00:00'))
    
    if isinstance(data.loc[0, 0], str) and ':' in data.loc[0, 0]:
        if data.loc[0, 0][3] == '-':
            dd=str2num(time[:,1:2])
            mm=str2num(time[:,4:5])
            yyyy=2000+str2num(time[:,7:8])
            HH=str2num(time[:,10:11])
            MM=str2num(time[:,13:14])
            out.yyyy=yyyy
            out.mm=mm
            out.dd=dd
            out.HH=HH
            out.MM=MM
            out.matTime=datenum(yyyy,mm,dd,HH,MM,0)
        else:
            out.matTime=datenum(time, 'yyyy-mm-dd HH:MM:SS')
    else:
        time = (data.loc[:, 0] * 3600 * 24 - 62167305600).to_numpy('M8[s]').astype('M8[ns]')
    
    dt_all = np.diff(time)
    dt = np.median(dt_all)
    if all(dt_all == dt):
        dt = None  # not need interp
    
    
    n_parameters = n_columns // len(z)
    data = np.reshape(data.values[:, 1:], (data.shape[0], -1, n_parameters)).T
    out = {}
    rename = {  # colnames before "_"
        'Horizontal': 'Vabs',
        'Direction': 'Vdir',
        'Vertical': 'Vz',
        'Beam1': 'Vbeam1',
        'Beam2': 'Vbeam2',
        'Beam3': 'Vbeam3',
        'Beam4': 'Vbeam4',
        'SP Std.': 'Vstd',      # cm/s Single ping standard deviation
        'Strength': 'Sv'        # dB Signal strength
    }
    for i in range(n_parameters):
        col_name = column_headers[i].split('_')[0]
        try:
            col_name = rename[col_name]
        except KeyError:
            print(f'Not known column name: {col_name}')
        out[col_name] = data[i, :, :]
    
    out['Vabs'] = out['Vabs']/100
    #     name = column_headers[i][:-2].replace(' ', '_').replace('.', '').replace('1', 'a1')
    #     out[name] = data.iloc[:, (i + 1)::n_parameters].values.flatten()
    
    # col_names = out.keys()
    # n_float_cols = len(col_names) - 1
    save_2d_for_surfer(
        time=time,
        z=z,
        out=out,
        path_base=(path_out if path_out else file_in).with_name('RDCP_2d'),
        dt=[dt] + np.array([30, 120, 360], 'm8[m]').tolist(),  # optimal minimum, and more averaged grids
        dz=[None]*3 + [2]
    )
    return out


def matlab_datenum_to_python_datetime(matlab_datenum):
    python_datetime = (
        datetime.datetime.fromordinal(int(matlab_datenum)) +
        datetime.timedelta(days=matlab_datenum%1) - datetime.timedelta(days=366)
    )
    return python_datetime


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
