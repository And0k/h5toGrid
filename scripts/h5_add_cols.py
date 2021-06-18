# -*- coding: utf-8 -*-
"""
Created on 20.04.2021
Purpose: change columns order of PyTables hdf5 file
@author: Korzh
"""
import sys
from pathlib import Path
import pandas as pd
from time import sleep

drive_d = 'D:' if sys.platform == 'win32' else '/mnt/D'  # to run on my Linux/Windows systems both
scripts_path = Path(drive_d + '/Work/_Python3/And0K/h5toGrid/scripts')
sys.path.append(str(Path(scripts_path).parent.resolve()))

from to_pandas_hdf5.h5toh5 import h5move_tables
from utils2init import Ex_nothing_done

store_in = Path(r'd:\workData\BalticSea\201202_BalticSpit\inclinometer\201202incl_proc.h5')
store_out = Path(r'd:\workData\BalticSea\201202_BalticSpit\inclinometer\processed_h5,vsz\201202incl_proc.h5')
tables = [f'V_incl_bin{bin}' for bin in (2, 600, 1800, 7200)]
b_childs_to_log_rows = True

# save to temp store
tables_have_wrote = set()
store_out_temp = store_out.with_suffix('.noindex.h5')
with pd.HDFStore(store_in, 'r') as sr, pd.HDFStore(store_out, 'r') as sw, pd.HDFStore(store_out_temp, 'w') as st:
    for tbl in tables:
        print(f'{tbl}:')
        # index=False is mandatory because it will not be CSI and ptrepack used in h5move_tables raises error
        same_columns = set(sw[tbl].columns).intersection(sr[tbl].columns)
        if same_columns:
            print(f'{store_out} already have columns {same_columns}')
        sw[tbl].join(sr[tbl], how="outer").to_hdf(
            st, tbl, format='table', data_columns=True, append=False, index=False
            )

        # copy children
        tables_have_wrote_cur = [tbl]
        nodes_cr = sr.get_storer(tbl).group.__members__
        cr = [f'/{tbl}/{g}' for g in nodes_cr if (g != 'table') and (g != '_i_table')]
        if cr:
            if b_childs_to_log_rows:
                nodes_cw = sw.get_storer(tbl).group.__members__
                cw = [f'/{tbl}/{g}' for g in nodes_cw if (g != 'table') and (g != '_i_table')]
                for child_w, child_r in zip(cr, cw):
                    try:
                        sw[child_w].append(sr[child_r]).to_hdf(
                            st, child_w, format='table', data_columns=True, append=False, index=False
                            )
                    except AttributeError:  # 'NoneType' object has no attribute 'startswith'
                        print('may be', sw, child_w, 'corrupted. Trying overwrite with', store_in, 'only values')
                        sw[child_w] = sr[child_r]
                    tables_have_wrote_cur.append(child_w)
            else:
                print('found {} cr of {}. Copying...'.format(len(nodes_cr), tbl))
                for child_r in cr:
                    st._handle.copy_node(child_r, newparent=sw.get_storer(tbl).group, recursive=True, overwrite=True)
        tables_have_wrote.add(tuple(tables_have_wrote_cur))
    st.flush()  # .flush(fsync=True
sleep(8)

# write tables with sorted index
try:
    failed_storages = h5move_tables({
        'db_path_temp': store_out_temp,
        'db_path': store_out,
        'addargs': ['--overwrite']  # '--checkCSI'
        },
        tbl_names=tables_have_wrote,
        # col_sort='Time'  # must exist
        )  # 'navigation/logFiles' will be copied as child
except Ex_nothing_done as e:
    print('Tables not moved')
except RuntimeError:  # not captured
    raise

if False:
    store_out = str(Path(store_out).with_name('sort_man.h5'))
    with pd.HDFStore(store_in, 'r') as sr, pd.HDFStore(store_out, 'w') as sw:
        df = sr[tbl]
        df.sort_index().to_hdf(sw, tbl, format='table', data_columns=True, append=False, index=False)
        sw.create_table_index(tbl, columns=['index'], kind='full')

    store_in, store_out = store_out, str(Path(store_out).with_name('sort_man_ptp.h5'))
    h5move_tables({
        'db_path_temp': store_in,
        'db_path': store_out,
        'tables': [tbl],
        'tables_log': [],
        'addargs': ['--checkCSI', '--verbose']
        })