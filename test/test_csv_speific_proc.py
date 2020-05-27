# datafile = '/mnt/D/Work/_Python3/And0K/h5toGrid/test/csv2h5/data/inclin_Kondrashov.txt'
import os
import sys
import unittest

from to_pandas_hdf5.csv2h5 import *  # main as csv2h5, __file__ as file_csv2h5, read_csv
from to_pandas_hdf5.h5_dask_pandas import filterGlobal_minmax

# import imp; imp.reload(csv2h5)

path_cruise = '/mnt/D/Work/_Python3/And0K/h5toGrid/test/csv2h5/data'  # r'd:/WorkData/BalticSea/180418_Svetlogorsk/inclinometer'
scripts_path = '/mnt/D/Work/_Python3/And0K/h5toGrid/scripts'  # to find ini
test_path = Path('/mnt/D/Work/_Python3/And0K/h5toGrid/test')
sys.path.append(str(Path(test_path).parent.resolve()))


# g.es(sys.argv[0])
# sys.argv[0] = scripts_path                                    # to can write log to ".\log" directory
# os.chdir(os.path.dirname(scripts_path))


class test_1(unittest.TestCase):
    def test_rep_in_file(self):
        in_file = test_path / 'csv2h5/data/INKL_008_Kondrashov_raw.txt'

        fsub = f_repl_by_dict([b'(?P<use>^20\d{2}(,\d{1,2}){5}(,\-?\d{1,6}){6}(,\d{1,2}\.\d{2})(,\-?\d{1,2}\.\d{2})).*',
                               b'^.+'])  # $ not works without \r\n so it is useless
        # '^Inklinometr, S/N 008, ABIORAS, Kondrashov A.A.': '',
        # '^Start datalog': '',
        # '^Year,Month,Day,Hour,Minute,Second,Ax,Ay,Az,Mx,My,Mz,Battery,Temp':

        out_file = in_file.with_name(re.sub('^inkl_0', 'incl', in_file.name.lower()))

        rep_in_file(in_file, out_file, fsub, header_rows=1)

        with open(out_file, 'rb') as fout:
            for irow in range(3):
                line = fout.readline()
                assert not b'RS' in line


if True:  # g.unitTesting:  #
    # import project_root_path
    # g.cls()
    from to_pandas_hdf5.csv_specific_proc import *
    from to_pandas_hdf5.h5toh5 import h5init, h5del_obsolete
    from to_pandas_hdf5.h5_dask_pandas import h5_append_dummy_row

# cfg
cfg = main([
    os.path.join(scripts_path, 'ini/csv_inclin_Kondrashov.ini'),
    '--path', os.path.join(path_cruise, 'inclin_Kondrashov_180430.txt'),
    '--b_interact', 'False',
    '--return', '<gen_names_and_log>',
    '--log', os.path.join(scripts_path, 'log/csv2h5_inclin_Kondrashov.log'),
    '--b_skip_if_up_to_date', 'False',  # to not use store
    '--date_min', '30.04.2018 23:59:51',  # ; UTC, not output data < date_min
    '--date_max', '01.05.2018 00:00:05',  # ; UTC, not output data > date_max
    ])
print(cfg)
cfg_out = cfg['output_files']
# cfg['in']['fun_proc_loaded'].visualize()
for nameFull in cfg['in']['gen_names_and_log'](cfg):
    d = read_csv(nameFull, **cfg['in'])  # , b_ok
    tim = d.index.compute()
    # assert isinstance(tim, pd.Series)
    assert isinstance(tim, pd.DatetimeIndex)  # , 'tim class'
    assert isinstance(d, dd.DataFrame)

    if d is None:
        continue

    # test filterGlobal_minmax()
    bGood = filterGlobal_minmax(d, None, cfg['filter'])
    # test set_filterGlobal_minmax()
    d, tim = set_filterGlobal_minmax(d, cfg['filter'], cfg_out['log'])

    if cfg_out['log']['rows_filtered']:
        print('filtered out {}, remains {}'.format(cfg_out['log']['rows_filtered'], cfg_out['log']['rows']))
    elif cfg_out['log']['rows']:
        print('.', end='')
    else:
        print('no data!')
        continue

    cfg['in']['time_last'] = tim[-1]  # save last time to can filter next file
    print(f'Filtering success, data range: {tim[0]} - {tim[-1]}')

    # test h5_append_dummy_row()
    cfg_out.setdefault('fs')
    d1 = h5_append_dummy_row(d, cfg_out['fs'], tim)

    df = d1.compute()  # [list(cfg_out['dtype'].names)].set_index(tim)
    assert isinstance(df, pd.DataFrame)

    # csv2h5(['ini/csv_inclin_Kondrashov.ini',
#        '--path', os_path.join(path_cruise, r'inclin_Kondrashov_180430.txt'),
#        ])
