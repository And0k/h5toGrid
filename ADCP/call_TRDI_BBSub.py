#!/usr/bin/env python
# coding:utf-8
from __future__ import print_function

"""
  Author:  Andrey Korzh --<ao.korzh@gmail.com>
  Purpose: Calls TRDI 'pathRDI_Tools'\BBSub.exe on 1st files which has neighbours with same name and increased
  extension
   Result placed instead this 1st file and source files archived into 'nameSourceArchive'.zip before replacing
  Created: 21.03.2015
"""

''' Input data path (only first line after "strDir = r'\" will be used): '''
strDir = r'\
d:\WorkData\_source\BalticSea\150921Nord3\ADCP_RDI600kHz\0924\
d:\WorkData\Cruises\_BalticSea\150317_Oceania\_source\ADCP\notebookMSI(+1hour_in_names_only)\
d:\WorkData\Cruises\_Schuka\_source\1409PSh129\
d:\WorkData\Cruises\_Schuka\1409PSh128\vmd\VMD_ENX_renamed\
d:\WorkData\Cruises\_Schuka\1409PSh128\vmd\VMD_ENS_renamed\
d:\WorkData\Cruises\_Schuka\1409PSh128\1\
d:\WorkData\Cruises\_Schuka\1409PSh128\
\\\
'
''' Input data mask  '''
strMask = r'*r.0??'

BBSub_filter = 'Redo'  # 'OnlyDown','OnlyUp','ForceDown','OS','Redo','Resample,n' where "n" is number of ensembles
""" BBSub_filter description from WorkHorse Operation Manual \ RDI TOOLS SOFTWARE USER’S GUIDE - February 2013
''           - Do not Filter Data – No filtering is done.
'OnlyDown'   - Extract only Down-Looking Data – Only ensembles with orientation sensor pointing down will be copied.
'OnlyUp'     - Extract only Up-Looking Data – Only ensembles with orientation sensor pointing up will be copied. Page 24
'ForceDown'  - Force Sensor to be Down-Looking – All the ensembles will have the orientation sensor pointing down.
'OS'         - Extract Secondary OS Data – If both the BroadBand and NarrowBand data are recorded (Ocean Surveyor)
this option extracts the NarrowBand data. This makes the data compatible with TRDI’s standard software.
'Redo'       - Redo Ensemble Numbers – Ensembles will be renumbered starting with ensemble num-ber 1.
'Resample,n' - Resample Data – Every n-th ensemble can be extracted to a new binary data file.
•            - Force 3 beams to 4 – Changes the number of beams listed in fixed leader of pd0 from 3 to 4.
This is for horizontal data that has 3 beams listed in the fixed leader and cannot be repro-cessed by other programs (see FSB-188).
"""

nameSourceArchive = '_source-Splitted'  # name of created directory to put result and archive of replaced files
maxFilesSize = 32 * 1024 * 1024  # bytes
pathRDI_Tools = r'c:\Program Files\RD Instruments\RDI Tools'  # path will be corrected for x64 windows automatically

from utils2init import dir_walker, readable, bGood_dir, bGood_file, first_of_paths_text
from os import mkdir as os_mkdir, rmdir as os_rmdir, remove as os_remove, path as os_path
from shutil import make_archive, move
import subprocess

''' Filter used directories and files '''
filt_dirCur = lambda f: readable(f) and bGood_dir(f, namesBadAtEdge=(r'bad', r'test'))
filt_fileCur = lambda f, mask: bGood_file(f, mask, namesBadAtEdge=(r'coef'))

strPath_BBSub = os_path.join(pathRDI_Tools, r'BBSub.exe')
if not os_path.isfile(strPath_BBSub):
    pathRDI_Tools = r'c:\Program Files (x86)\RD Instruments\RDI Tools'
    strPath_BBSub = os_path.join(pathRDI_Tools, r'BBSub.exe')
    if not os_path.isfile(strPath_BBSub):
        raise IOError('BBSub.exe not found')


# Returns indexes of first and last files with same name and increased extension
def range_of_files_with_inc_ext(namesFull):
    nameDW = ''  # directory name was took before current cycle
    nameFW = ''  # file name without ext was took before current cycle
    # nameEW= '9999' #             extension was took before current cycle
    bSavedStart = False
    # inameEW= 9999
    for iname, nameFull in enumerate(namesFull):
        nameD, nameFE = os_path.split(nameFull)
        nameF, nameE = os_path.splitext(nameFE)
        if nameDW == nameD and nameFW == nameF:
            if not nameE[1:].isdigit():
                s = raw_input('Extension is not digits in "' + nameFull + '". Continue?')
                if 'n' in s or 'N' in s:
                    print('nothing done')
                    exit
                else:
                    print('wait')

            # if nameEW < nameE and
            if not bSavedStart:  # exist next file with the same name but next numeric extension
                bSavedStart = True
                inameSt = iname - 1  # remember start
                # nameEW= '9999'
            # else:
            # nameEW= nameE
            # inameW= iname
        else:
            if bSavedStart:
                bSavedStart = False
                yield (inameSt, iname)
            nameDW = nameD
            nameFW = nameF
            # nameEW= nameE
    if bSavedStart: yield (inameSt, iname + 1)


class Tconsole:
    def write(self, arg):
        print(arg)

    def show(self):
        pass


console = Tconsole()
from datetime import datetime

console.write("%s > " % datetime.now())
console.show()

if __name__ == '__main__':
    strDir = first_of_paths_text(strDir)  # Get only first path from strDir
    print('found: ', end='')
    namesFull = [f for f in dir_walker(strDir, strMask, bGoodFile=filt_fileCur, bGoodDir=filt_dirCur)]
    nFiles = len(namesFull)
    if nFiles == 0:
        print('(0 files) => nothing done')
        exit
    else:
        s = raw_input('\n(' + str(nFiles) + r' files). Process it? Y/n: ')
        if 'n' in s or 'N' in s:
            print('nothing done')
            exit
        else:
            print('wait... ', end='')

            strPrint = strPath_BBSub + ' -in:{0} -out:{1}' + \
                       ((' -filter:' + BBSub_filter) if BBSub_filter else '')

            for inameSt, inameEn in range_of_files_with_inc_ext(
                    namesFull):  # indexes files with same name and increased extension
                # 1. move files which will modify
                while inameSt + 1 < inameEn:  # returns here while get FilesSizeSum > maxFilesSize
                    nameFull0 = namesFull[inameSt]
                    nameD, nameFE0 = os_path.split(nameFull0)
                    dst = os_path.join(nameD, nameSourceArchive)
                    if not os_path.isdir(dst): os_mkdir(dst)
                    nameFullDst = os_path.join(dst, nameFE0)
                    move(nameFull0, nameFullDst)  # move 1st file
                    FilesSizeSum = os_path.getsize(nameFullDst)
                    for inameSt in range(inameSt + 1, inameEn):  # move next files
                        nameFull = namesFull[inameSt]
                        nameFE = os_path.basename(nameFull)
                        FilesSizeSum += os_path.getsize(nameFull)
                        if FilesSizeSum < maxFilesSize:
                            move(nameFull, dst)  # dst
                        else:
                            break
                    # 2. Create combined file with the name of moved 1st file
                    if not os_path.isfile(nameFull0):  # Check that file actually have moved
                        runString = strPrint.format(nameFullDst,
                                                    nameFull0)  # os_path.join(nameD, os_path.splitext(nameFE0)[0] + '.0000')
                        subprocess.call(runString)
                    else:
                        s = raw_input('File "' + nameFull0 + '" will be overwrited! Continue?')
                        if 'n' in s or 'N' in s:
                            print('not all joined')
                            exit
                        else:
                            print('continue...')

            print('making archive of replaced files...')
            nameFull0 = make_archive(dst, 'zip', dst)
            if nameFull0 == dst + '.zip':  # think that all archived Ok, so delete source
                for inameSt, inameEn in range_of_files_with_inc_ext(namesFull):
                    for nameFull in namesFull[inameSt:inameEn]:
                        os_remove(os_path.join(dst, os_path.basename(nameFull)))
                try:
                    os_rmdir(dst)
                except WindowsError:
                    print('can not remove directory ' + nameSourceArchive)
            print('ok')  # end

##remove unused characters
# compile_obj = re.compile(rawstr,  re.MULTILINE| re.VERBOSE)
# with open(fileNames, 'r') as f:  matchstr = f.read()
# console.write(compile_obj.sub(repstr, matchstr))


# matchstr= compile_obj.findall(matchstr)
