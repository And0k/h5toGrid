#!/usr/bin/env python
# coding:utf-8
""" NOT WORKED!
  Author:  Andrey Korzh --<ao.korzh@gmail.com>
  Purpose: load binary ADCP files to...
  Created: 25.03.2015
"""
from __future__ import print_function

import os.path
from sys import path as sys_path

from utils2init import dir_walker, readable, bGood_dir, bGood_file, get1stString

sys_path.append(r'd:\Work\_Python\_other\GliderGroup')  # trdi_adcp_readers\pd0
from trdi_adcp_readers.pd0.pd0_parser import parse_pd0_bytearray

''' Input data path (only first line after "strDir = r'\" will be used): '''
strDir = get1stString(r'\
d:\WorkData\Cruises\_Schuka\1409PSh128\vmd\VMD_ENS_renamed\
\\\
')  # Get only first path from strDir

''' Input data mask  '''
strMask = r'*r.???'

''' Filter used directories and files '''
filt_dirCur = lambda f: readable(f) and bGood_dir(f, namesBadAtEdge=(r'bad', r'test'))
filt_fileCur = lambda f, mask: bGood_file(f, mask, namesBadAtEdge=(r'coef'))


# Standard starting dialog
def found_dialog(namesFull):
    nFiles = len(namesFull)
    if nFiles == 0:
        print('(0 files) => nothing done')
        # sys.exit(?)
    else:
        s = raw_input('\n(' + str(nFiles) + r' files). Process it? Y/n: ')
        if 'n' in s or 'N' in s:
            print('nothing done')
            nFiles = 0
        else:
            print('wait... ', end='')
    return nFiles


def file_walker(namesFull):
    nameDW = ''  # directory name was took before current cycle
    nameFW = ''  # file name without ext was took before current cycle
    bSavedStart = False

    for iname, nameFull in enumerate(namesFull):
        nameD, nameFE = os.path.split(nameFull)
        nameF, nameE = os.path.splitext(nameFE)
        pd0 = ''
        with open(nameFull, 'rb') as f:
            pd0 = f.read()
        pd0_bytes = bytearray(pd0)
        data = parse_pd0_bytearray(pd0_bytes)
        data.variable_leader


if __name__ == '__main__':
    # Find files
    print('found: ', end='')
    namesFull = [f for f in dir_walker(strDir, strMask, bGoodFile=filt_fileCur, bGoodDir=filt_dirCur)]
    if not found_dialog(namesFull): exit  # sys.exit()?
    file_walker(namesFull)

    # unittest.main()
# import sys
