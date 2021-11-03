"""
Warning: script not works correctly!
"""

from os import chdir as os_chdir, getcwd as os_getcwd
from pathlib import Path
import h5py
import numpy as np

path_cruise = Path(r'd:\workData\BalticSea\201202_BalticSpit_inclinometer\210726@P7.5m,P15m-i9,14,19w1,4')
db_name_out = r"210726vds_noAvg.h5"

db_names = ['210726_noAvg.h5', r'210827.proc_noAvg.proc_noAvg.h5']
tables = ['incl09', 'incl15']
table_out = f"/{','.join(tables)}/table"

path_prev = os_getcwd()
os_chdir(path_cruise)
# param = 'Vabs'
sources = []
total_length = 0
for i, (filename, table) in enumerate(zip(db_names, tables)):
    with h5py.File(filename, 'r') as h:
        # shape = h[f'/{table}/table'].shape
        # dtype = h[f'/{table}/table'].dtype
        vsource = h5py.VirtualSource(h[f'/{table}/table'])  # pandas dataset path standard
        total_length += vsource.shape[0]
        sources.append(vsource)

print('Source dtype:', vsource.dtype, end=',\n')
layout = h5py.VirtualLayout(shape=(total_length,),
                            dtype=vsource.dtype, # 'f8',  #only float ends without error but results in bad file because we ned structured type
                            )  # ilename=db_name_out
print('sources lengths:')
offset = 0
for vsource in sources:
    length = vsource.shape[0]
    layout[offset: offset + length] = vsource
    offset += length
    print(length)

with h5py.File(db_name_out, 'w', libver='latest') as f:
    f.create_virtual_dataset(table_out, layout)  #, fillvalue=0
print('stacked ok.')

os_chdir(path_prev)          # recover