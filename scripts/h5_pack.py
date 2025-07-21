from pathlib import Path, PurePath
import pandas as pd
from to_pandas_hdf5.h5toh5 import h5.find_tables, h5.move_tables

db_path = r'd:\workData\BalticSea\201202_BalticSpit_inclinometer\201202@i3,5,9,10,11,15,19,23,28,30,32,33,w1-6\201202.proc_noAvg.h5'
temp_db_path = r'd:\workData\BalticSea\201202_BalticSpit_inclinometer\201202@i3,5,9,10,11,15,19,23,28,30,32,33,w1-6\201202proc_not_sorted.h5'
with pd.HDFStore(temp_db_path, mode='r') as store:
    tables = h5.find_tables(store, '.*')

tables

failed_storages = h5.move_tables({
        'temp_db_path': PurePath(temp_db_path),
        'db_path': PurePath(db_path),
        'addargs': ['--checkCSI', '--verbose'],  # --overwrite-nodes
        'b_del_temp_db': False},
    tables
    )