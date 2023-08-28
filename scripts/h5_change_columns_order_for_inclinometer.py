# -*- coding: utf-8 -*-
"""
Created on Sat Aug 03 16:50:22 2013
Purpose: change columns order of PyTables hdf5 file
@author: Korzh
"""


import re

# import numpy as np
import pandas as pd

fileInF = \
    r'd:\WorkData\Cruises\_BalticSea\130510-\_source\Konek,Volnomer\130609\130609.h5'
fileOutF = r'c:\Temp\out.h5'
# = pd.HDFStore()
chunksize = 60000
bBegin = False  # True #


def number_key(name):
    # get list of words with digits converted to int (?)
    parts = re.findall('[^0-9]+|[0-9]+', name)
    L = []
    for part in parts:
        try:
            L.append(int(part))
        except ValueError:
            L.append(part)
    return L


# sorted(nodes, key=number_key)
# cmp1= lambda x,y: cmp(x.lower(), y.lower())

if __name__ == '__main__':
    with pd.HDFStore(fileInF, mode='r') as storeIn:
        nodes = sorted(storeIn.root.__members__, key=number_key)
        print(nodes)
        for m in nodes:  # k in storeIn.root:
            if (not bBegin):
                if m == 'i8':
                    bBegin = True
                else:
                    continue
            print(m, end=' ')
            df = storeIn[m]
            df.rename(columns={'U': 'P', 'P': 'U', 'Gy': 'Gz', 'Gz': 'Gy'},
                      inplace=True)
            df = df.reindex(columns='P Gx Gy Gz Hx Hy Hz U'.split())
            with pd.get_store(fileOutF) as store:  # error if open with fileInF
                store.remove(m)
                store.append(m, df, data_columns=True, chunksize=chunksize,
                             complib='zlib', complevel=9)
                store.flush()
