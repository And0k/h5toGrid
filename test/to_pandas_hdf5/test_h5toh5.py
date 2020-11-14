import sys
from pathlib import Path
import pandas as pd
from time import sleep

drive_d = 'D:' if sys.platform == 'win32' else '/mnt/D'  # to run on my Linux/Windows systems both
scripts_path = Path(drive_d + '/Work/_Python3/And0K/h5toGrid/scripts')
sys.path.append(str(Path(scripts_path).parent.resolve()))

from to_pandas_hdf5.h5toh5 import h5move_tables


store_in = r'd:\workData\BalticSea\200819_AI56\200819_AI56.h5'
store_out = r'd:\workData\BalticSea\200819_AI56\200819_AI56_nav.h5'
tbl = 'navigation'
# create temp test table from existed
with pd.HDFStore(store_in, 'r') as sr, pd.HDFStore(store_out, 'w') as sw:
    df = sr[tbl]
    df['DepEcho'] = df['DepEcho'].abs()
    #sw.put(tbl, df, format='table', data_columns=True)

    # index=False is mandatory because it will not be CSI and ptrepack used in h5move_tables raises error
    df.to_hdf(sw, tbl, format='table', data_columns=True, append=False, index=False) # dropna=not cfg_out.get('b_insert_separator'

    # copy childs
    parent_group = sr.get_storer(tbl).group
    nodes = parent_group.__members__
    childs = [f'/{tbl}/{g}' for g in nodes if (g != 'table') and (g != '_i_table')]
    if childs:
        print('found {} childs of {}. Copying...'.format(len(nodes), tbl))
        for child in childs:
            sr._handle.copy_node(child, newparent=sw.get_storer(tbl).group, recursive=True, overwrite=True)
        sw.flush()  # .flush(fsync=True
sleep(8)

# write tables back with sorted index
store_in, store_out = store_out, store_in
h5move_tables({
    'db_path_temp': store_in,
    'db_path': store_out,
    'tables': [tbl],
    'tables_log': [],
    'addargs': ['--checkCSI']
    },
    col_sort='Time'
    )  #'navigation/logFiles' will be copied as child


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

