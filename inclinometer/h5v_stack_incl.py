from datetime import datetime
from pathlib import Path
from re import match
from itertools import zip_longest

db_paths = [
    r'e:\WorkData\BalticSea\181005_ABP44\inclinometer\181017.proc_noAvg.h5',
    r'e:\WorkData\BalticSea\181005_ABP44\inclinometer\181022.proc_noAvg.h5'

    # r'd:\workData\BalticSea\201202_BalticSpit_inclinometer\210618P7.5m@i9,14,15,19w1,2,4\210618@i15,w02.proc_noAvg.h5',
    # r'd:\workData\BalticSea\201202_BalticSpit_inclinometer\210618P7.5m@i9,14,15,19w1,2,4\210726@w1.proc_noAvg.h5',

    # r'd:\workData\BalticSea\201202_BalticSpit_inclinometer\210618P7.5m@i9,14,15,19w1,2,4\210618@i14,15+19,w2+1,4.proc_noAvg.h5',
    # r'd:\workData\BalticSea\201202_BalticSpit_inclinometer\210618P7.5m@i9,14,15,19w1,2,4\210618@i09.proc_noAvg.h5'

    # r'd:\workData\BalticSea\201202_BalticSpit_inclinometer\210618P7.5m@i15,w2;210709@i14,19w1,4\210618@w02+01.proc_noAvg.h5',
    # r'd:\workData\BalticSea\201202_BalticSpit_inclinometer\210618P7.5m@i15,w2;210709@i14,19w1,4\210709,26.proc_noAvg.h5',

    # r'd:\workData\BalticSea\201202_BalticSpit_inclinometer\210618P7.5m@i15,w2;210709@i14,19w1,4\210618.proc_noAvg.h5',
    # r'd:\workData\BalticSea\201202_BalticSpit_inclinometer\210618P7.5m@i15,w2;210709@i14,19w1,4\210709,26.proc_noAvg.h5',

    # r'd:\workData\BalticSea\201202_BalticSpit_inclinometer\210618P7.5m@i15,w2(cal),210709@i14,19w1,4\210618.proc_noAvg.h5',
    # r'd:\workData\BalticSea\201202_BalticSpit_inclinometer\210618P7.5m@i15,w2(cal),210709@i14,19w1,4\210709,26.proc_noAvg.h5',

    # r'd:\workData\BalticSea\201202_BalticSpit_inclinometer\210726@P10,E14.5,P15m-i4,5,11,36,37,38,w2,5,6\210726.proc_noAvg.h5',
    # r'd:\workData\BalticSea\201202_BalticSpit_inclinometer\210827@P10m,P15m-i14,15,w1,4\210827.proc_noAvg.h5'

    # r'd:\workData\BalticSea\201202_BalticSpit_inclinometer\210726@P10,E14.5,P15m-i4,5,11,36,37,38,w2,5,6\210726.proc_noAvg.h5',
    # r'd:\workData\BalticSea\201202_BalticSpit_inclinometer\210827@P10m,P15m-i14,15,w1,4\210827.proc_noAvg.h5'

    # r'd:\workData\BalticSea\201202_BalticSpit_inclinometer\210726@P7.5m,P15m-i9,14,19w1,4\210726.proc_noAvg.h5',
    # r'd:\workData\BalticSea\201202_BalticSpit_inclinometer\210827@P10m,P15m-i14,15,w1,4\210827.proc_noAvg.h5'
    ]
tables = [['i03'],['i09'],['i10']]
    # ['i16', 'i19']
    # ['w02', 'w01']
    # ['i15_19', 'incl09']
# ['incl15', 'incl19']
# ['w04', 'w04']
# ['incl05', 'incl14']
# ['w05', 'w01']
# ['incl09', 'incl15']
cfg_out = {
    'logfield_fileName_len': 255,
    'b_insert_separator': True,
    'fs': 5  # Hz
    }

table_groups = [tables] if isinstance(tables[0], str) else tables
device_type = match(r'([^\d]+).*', table_groups[0][0]).group(1)

device_numbers = [match(r'[^\d]+(.*)', table).group(1) for table in table_groups[0]]
ids_str = f"{device_type[0]}{'_'.join(device_numbers)}"
path_db_out = Path(db_paths[0])
path_db_out_stem = f"{path_db_out.with_suffix('').stem}@{ids_str}".replace('..', '.')  # stem without .proc_noAvg
path_db_out = path_db_out.with_name(path_db_out_stem).with_suffix(''.join(path_db_out.suffixes))
table_out = ids_str
cols_out_h5 = ['v', 'u', 'inclination', 'Temp'] if device_type[0] == 'i' else ['Pressure', 'Temp']

to_vaex = False
if to_vaex:

    db_path_out = r"d:\workData\BalticSea\201202_BalticSpit_inclinometer\210726@P7.5m,P15m-i9,14,19w1,4\incl09,15.vaex.hdf5"
    from to_vaex_hdf5.csv_to_h5vaex import h5pandas_to_vaex, h5pandas_to_vaex_combine  #, h5pandas_to_vaex_file_names

    chunk_start = 0
    for i, (db_path, table) in enumerate(zip(db_paths, tables)):
        chunk_start = h5pandas_to_vaex(db_path, table, chunk_start=chunk_start, merge=False)

    # copy files in db_path_out now

    h5pandas_to_vaex_combine(
        '*_000[0-9][0-9][0-9][0-9][0-9][0-9].vaex.hdf5',
        db_path_out,
        check_files_number=chunk_start,
        del_found_tmp_files=True
        )

else:
    import numpy as np
    import pandas as pd
    from to_pandas_hdf5.h5_dask_pandas import h5_load_range_by_coord, h5_append
    from to_pandas_hdf5.h5toh5 import h5move_tables


    path_db_out_temp = path_db_out.with_suffix('.temp.h5')
    if True:  # set to False if want reuse temporary table from previous run
        with pd.HDFStore(path_db_out_temp, mode='w') as cfg_out['db']:
            for tables in table_groups:
                tbl_prev = ''
                tables_for_out_name = [(tbl_prev := tbl) for tbl in tables if tbl_prev != tbl]  # skip same consequent names

                device_type = match(r'([^\d]+).*', tables_for_out_name[0]).group(1)
                device_numbers = [match(r'[^\d]+(.*)', table).group(1) for table in tables_for_out_name]
                table_out = f"{device_type[0]}{'_'.join(device_numbers)}"

                if len(tables) == 1:
                    tables = tables * 2
                for i, (db_path, table) in enumerate(zip(db_paths, tables if len(tables) > 1 else tables*len(db_paths))):
                    path_db = Path(db_path)
                    d = h5_load_range_by_coord(
                        path_db,
                        table,
                        # range_coordinates=None,
                        columns=cols_out_h5)
                    # d['Pressure'] = d['Pressure'].astype(np.float32)
                    d_mod = d.astype(np.float32)  # to decrease file size

                    log = {
                        'Date0': d.divisions[0],
                        'DateEnd': d.divisions[-1],
                        'fileName': f'{path_db.stem}/{table}'[-cfg_out['logfield_fileName_len']:],  # path_db.parent.name/
                        'fileChangeTime': datetime.fromtimestamp(path_db.stat().st_mtime),
                        'rows': len(d)
                        }
                    print(log)
                    cfg_out['table'] = table_out
                    h5_append(cfg_out,
                        d_mod,
                        log=log,
                        tim=d.index.compute()  # need if b_insert_separator=True
                        )
                    # todo: insert row of NaNs


                # # if want sort by index (that will be done also if call with h5move_tables with arguments=None or ='fast')
                # # then add index first:
                # cfg_out['db'].create_table_index(table, columns=['index'], kind='full')

    failed_storages = h5move_tables({
        'db_path_temp': path_db_out_temp,
        'db_path': path_db_out
        #'tables_log': []
        },
        tbl_names=cfg_out['tables_written'],
        # do not sort (suppose data already sorted) - if we not set ``arguments`` to default None or "fast":
        arguments=[
            '--chunkshape=auto', '--propindexes', '--checkCSI',
            '--verbose', '--complevel=9',
            '--overwrite-nodes']  #, '--complib=blosc' - lib not installed
    )