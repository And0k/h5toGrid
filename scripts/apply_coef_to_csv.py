# -*- coding: utf-8 -*-
"""
  Author:  Andrey Korzh --<ao.korzh@gmail.com>
  Purpose: 1. load polinom coeffcients from text file,
           2. load all text data files provided by mask 
           3. apply coef. and save result text files to parent dir
  Created: 23.10.2014
"""
from __future__ import print_function

fileCoef = \
    r'd:\WorkData\Experiment\TermoChain\141021\coef#t_TINA_141017.txt'
fileMaskIn = \
    r'd:\WorkData\Experiment\TermoChain\141021\TERM_GRUNT\*.txt'
from utils2init import dir_walker, readable
import os.path
import numpy as np

# 1. load polinom coeffcients from text file
cfg = {}
cfg['NamesOut'], cfg['k'] = np.loadtxt(fileCoef, dtype={'names': ('kProbe', 'k'),
                                                        'formats': ('S10', '4f4')}, skiprows=1, unpack=True)
cfg['k'] = np.fliplr(np.atleast_2d(cfg['k'])).T
Nprobes = np.shape(cfg['k'])[1]
cfg['NamesOut'] = str(cfg['NamesOut'])  # default names

# 2. load all text data files provided by mask
cfg['dtype'] = np.uint16
dirname, strMask = os.path.split(fileMaskIn)
saveDir = os.path.abspath(os.path.join(dirname, os.path.pardir))
fileInF = None
print('saving to', saveDir + ':')
for fileInF in dir_walker(dirname, strMask, readable):
    data = np.loadtxt(fileInF, dtype=cfg['dtype'], skiprows=1, unpack=True)  # np.atleast_2d()
    # apply coef.
    V = np.polyval(cfg['k'], np.reshape(data, (np.size(data, 0), Nprobes)))
    # save
    fileInF = os.path.basename(fileInF)
    print(fileInF)
    fileInF = fileInF[:-4]
    np.savetxt(os.path.join(saveDir, fileInF + '.txt'), V, delimiter='\t', newline='\n',
               header=cfg['NamesOut'] if Nprobes > 1 else fileInF, fmt='%2.8f', comments='')
else:
    if not fileInF: print('"nothing"')
print('OK>')
