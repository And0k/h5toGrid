# datafile = '/mnt/D/Work/_Python3/And0K/h5toGrid/test/csv2h5/data/inclin_Kondrashov.txt'
import os
import sys
import unittest, pytest
from functools import partial

from to_pandas_hdf5.csv2h5 import *  # main as csv2h5, __file__ as file_csv2h5, read_csv
from to_pandas_hdf5.csv_specific_proc import *
from to_pandas_hdf5.h5toh5 import h5init, h5del_obsolete

from to_pandas_hdf5.h5_dask_pandas import filterGlobal_minmax, h5_append_dummy_row
from utils2init import path_on_drive_d
# import imp; imp.reload(csv2h5)

r'd:/WorkData/BalticSea/180418_Svetlogorsk/inclinometer'
scripts_path = path_on_drive_d('/mnt/D/Work/_Python3/And0K/h5toGrid/scripts')  # to find ini
test_path = path_on_drive_d('/mnt/D/Work/_Python3/And0K/h5toGrid/test')
sys.path.append(str(Path(test_path).parent.resolve()))


# g.es(sys.argv[0])
# sys.argv[0] = scripts_path                                    # to can write log to ".\log" directory
# os.chdir(os.path.dirname(scripts_path))


def test_proc_loaded_nav_HYPACK_SES2000(a: Union[pd.DataFrame, np.ndarray], cfg_in: Mapping[str, Any]) -> pd.DatetimeIndex:
    """
    Specified prep&proc of SES2000 data from program "HYPACK":
    - Time calc: gets string for time in current zone
    - Lat, Lon to degrees conversion

    :param a: numpy record array. Will be modified inplace.
    :param cfg_in: dict
    :return: numpy 'datetime64[ns]' array

    Example input:
    a = {
    'date': "02:13:12.30", #'Time'
    'Lat': 55.94522129,
    'Lon': 18.70426069,
    'Depth': 43.01}
    """

    proc_loaded_nav_HYPACK_SES2000(a, cfg_in)


    # extract date from file name
    if not cfg_in.get('fun_date_from_filename'):
        def date_from_filename(file_stem, century=None):
            return century + '-'.join(file_stem[(slice(k, k + 2))] for k in (6, 3, 0))

        cfg_in['fun_date_from_filename'] = date_from_filename
    elif isinstance(cfg_in['fun_date_from_filename'], str):
        cfg_in['fun_date_from_filename'] = eval(
            compile("lambda file_stem, century=None: {}".format(cfg_in['fun_date_from_filename']), '', 'eval'))

    str_date = cfg_in['fun_date_from_filename'](cfg_in['file_stem'], century.decode())
    t = pd.to_datetime(str_date) + \
        pd.to_timedelta(a['Time'].str.decode('utf-8', errors='replace'))
    # t = day_jumps_correction(cfg_in, t.values)
    return t



def test_rep_in_file():
    file_in = test_path / 'csv2h5/data/INKL_008_Kondrashov_raw.txt'

    fsub = f_repl_by_dict([b'(?P<use>^20\d{2}(,\d{1,2}){5}(,\-?\d{1,6}){6}(,\d{1,2}\.\d{2})(,\-?\d{1,2}\.\d{2})).*',
                           b'^.+'])  # $ not works without \r\n so it is useless
    # '^Inklinometr, S/N 008, ABIORAS, Kondrashov A.A.': '',
    # '^Start datalog': '',
    # '^Year,Month,Day,Hour,Minute,Second,Ax,Ay,Az,Mx,My,Mz,Battery,Temp':

    file_out = file_in.with_name(re.sub('^inkl_0', 'incl', file_in.name.lower()))

    rep_in_file(file_in, file_out, fsub, header_rows=1)

    with open(file_out, 'rb') as fout:
        for irow in range(3):
            line = fout.readline()
            assert not b'RS' in line




@pytest.fixture()
def cfg():
    path_cruise = path_on_drive_d('/mnt/D/Work/_Python3/And0K/h5toGrid/test/csv2h5/data')  #
    get_cfg = partial(main, [
        os.path.join(scripts_path, 'cfg/csv_inclin_Kondrashov.ini'),
        '--path', os.path.join(path_cruise, 'inclin_Kondrashov_180430.txt'),
        '--b_interact', 'False',
        '--return', '<gen_names_and_log>',
        '--log', os.path.join(scripts_path, 'log/csv2h5_inclin_Kondrashov.log'),
        '--b_incremental_update', 'False',  # to not use store
        '--min_date', '30.04.2018 23:59:51',  # ; UTC, not output data < min_date
        '--max_date', '01.05.2018 00:00:05',  # ; UTC, not output data > max_date
        ])
    return get_cfg


def test_csv2h5(cfg):
    print(cfg)
    cfg_out = cfg['out']
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

    # csv2h5(['cfg/csv_inclin_Kondrashov.ini',
#        '--path', os_path.join(path_cruise, r'inclin_Kondrashov_180430.txt'),
#        ])
