import pytest
from datetime import timedelta
from pathlib import Path
import logging
import json
import pandas as pd
import numpy as np

from to_vaex_hdf5.autofon_coord import *  # autofon_df_from_dict

@pytest.fixture()
def autofon_dict():
    """

    3179282432 - будильники
    :return:
    """
    with Path(r'data\tracker\autofon_coord_200msgs.json').open('r') as fp:
        return json.load(fp)


#@pytest.mark.parametrize('autofon_dict', )
def test_autofon_df_from_dict(autofon_dict):
    df = autofon_df_from_dict(autofon_dict,timedelta(0))
    assert ['Lat',
            'Lon',
            'Speed',
            'LGSM',
            'HDOP',
            'n_GPS',
            'Temp',
            'Course',
            #'Height',
            #'Acceleration'
            ] == df.columns.to_list()
    dtype = np.dtype
    assert [dtype('float32'),
            dtype('float32'),
            dtype('float32'),
            dtype('int8'),
            dtype('float16'),
            dtype('int8'),
            dtype('int8'),
            dtype('int8'),
            #dtype('int8'),
            #dtype('int8')
            ] == df.dtypes.to_list()

    assert len(df) == 200


@pytest.mark.parametrize(
    'file_raw_local', [
        r'data\tracker\AB_SIO_RAS_tracker',
        r'data\tracker\SPOT_ActivityReport.xlsx'
        ])
def test_file_raw_local(file_raw_local):
    path_raw_local = Path(file_raw_local)
    cfg_in = {'time_interval': ['2021-06-02T13:49', '2021-06-04T20:00']}  # UTC
    time_interval = [pd.Timestamp(t, tz='utc') for t in cfg_in['time_interval']]

    df = loading(
        table='sp4',
        path_raw_local=path_raw_local,
        time_interval=time_interval,
        dt_from_utc=timedelta(hours=2)
        )
    assert all(df.columns == ['Lat', 'Lon'])
    assert df.shape[0] == 3


def test_multiple_with_anchor_as_mean():
    """Loading 5 trackers data in one DB, calc mean positions and displacements relative to them"""
    # todo: check why tr2 anchor so North shifted (maybe data filtered after anchor pos. determined?)
    path_db = Path(
        r'd:\Work\_Python3\And0K\h5toGrid\test\data\tracker\220810@sp2,4,5,tr1,2.h5'.replace('\\', '/')
        )
    file_raw_local = str(path_db.with_suffix('.raw.h5')).replace('\\', '/')
    device = ['sp2', 'sp4', 'sp5', 'tr1', 'tr2']
    sys_argv_save = sys.argv.copy()
    sys.argv = [__file__]
    main_call([
        'input.path_raw_local="{}"'.format(file_raw_local),  # not load from internet
        # f'+input.path_raw_local_dict={{sp:"{file_raw_local}"}}',
        'input.time_interval=[2022-08-10T09:16, 2022-08-10T13:22]',
        'input.dt_from_utc_hours=2',
        'out.db_path="{}"'.format(str(path_db).replace('\\', '/')),
        'out.tables=[{}]'.format(','.join([f'"{d}"' for d in device])),
        'process.anchor_coord="mean"',
        # 'process.b_reprocess=True',
        'process.max_dr=200',
        'process.anchor_depth=0',
        # 'process.period_tracks=1D'
        ])
    sys.argv = sys_argv_save



def test_multiple_with_anchor_as_mean():
    """Loading 5 trackers data in one DB, calc mean positions and displacements relative to them"""
    # Same as previous for other data
    path_db_in = Path(
        r'd:\Work\_Python3\And0K\h5toGrid\test\data\tracker\220831@sp2,4,5,6,tr1,2.raw.h5'.replace('\\', '/')
        )
    path_db_out = path_db_in.parent / '220831_1340@sp2,4,5,6,tr1,2.h5'

    device = ['sp2', 'sp4', 'sp5', 'sp5', 'tr1', 'tr2']
    sys_argv_save = sys.argv.copy()
    sys.argv = [__file__]
    main_call([
        'input.path_raw_local="{}"'.format(path_db_in),  # not load from internet
        # f'+input.path_raw_local_dict={{sp:"{file_raw_local}"}}',
        'input.time_interval=[2022-08-31T13:40, 2022-08-31T17:33]',
        'input.dt_from_utc_hours=2',
        'out.db_path="{}"'.format(str(path_db_out).replace('\\', '/')),
        'out.tables=[{}]'.format(','.join([f'"{d}"' for d in device])),
        'process.anchor_coord="mean"',
        # 'process.b_reprocess=True',
        'process.max_dr=300',
        'process.dt_max_hole="20min"',
        'process.anchor_depth=0',
        # 'process.period_tracks=1D'
        ])
    sys.argv = sys_argv_save



def test_call_example_sp2to3():
    """Loading two trackers data in one DB, calc distance between, recalc all"""
    path_db = Path(
        r'd:\WorkData\BalticSea\220505_D6\tracker\220505@sp2ref3_test'.replace('\\', '/')
        )  # r'd:\WorkData\_experiment\tracker\220318_1633@sp2&3.h5'.replace('\\', '/')
    file_raw_local = str(path_db.parent / 'raw' / 'ActivityReport.xlsx').replace('\\', '/')
    device = ['sp3', 'sp2']
    sys_argv_save = sys.argv.copy()
    if __name__ != '__main__':
        sys.argv = [__file__]
    main_call([
        f'input.path_raw_local="{file_raw_local}"',
        'input.time_interval=[2022-05-05T12:45, now]',
        'input.dt_from_utc_hours=2',
        'out.db_path="{}"'.format(str(path_db).replace('\\', '/')),
        'out.tables=[{}]'.format(','.join([f'"{d}"' for d in device])),

        'process.b_reprocess=True',
        'process.anchor_coord=[sp3]',
        # 'process.max_dr=50',
        'process.anchor_depth=40',
        'process.period_tracks=1D'
        ])
    sys.argv = sys_argv_save


# Loading 3 trackers data in one DB, calc distance between. Different data sources
def test_call_example_tr2_sp5to2():

    path_db = Path(
        r'd:\WorkData\_experiment\tracker\220715\220715.h5'.replace('\\', '/')
        )
    file_raw_local = str(path_db.with_suffix('.raw.h5')).replace('\\', '/')
    device = ['sp2', 'sp5', 'tr2']
    sys_argv_save = sys.argv.copy()
    sys.argv = [__file__]
    main_call([
        'input.path_raw_local="{}"'.format(file_raw_local),  # not load from internet
        # f'+input.path_raw_local_dict={{sp:"{file_raw_local}"}}',
        'input.time_interval=[2022-07-15T10:50, 2022-07-15T18:50]',
        'input.dt_from_utc_hours=2',
        'out.db_path="{}"'.format(str(path_db).replace('\\', '/')),
        'out.tables=[{}]'.format(','.join([f'"{d}"' for d in device])),

        # 'process.b_reprocess=True',
        'process.anchor_tracker=[sp2]',
        # 'process.max_dr=50',
        'process.anchor_depth=5',
        'process.period_tracks=1D'
        ])
    sys.argv = sys_argv_save


def test_call_example_sp6ref5():
    """Loading two trackers data in one DB, calc distance between, recalc all"""
    path_db = Path(
        r'd:\WorkData\BalticSea\220505_D6\tracker\220505@sp6ref5_test'.replace('\\', '/')
        )
    file_raw_local = str(path_db.parent / 'raw' / 'ActivityReport_test.xlsx').replace('\\', '/')
    device = ['sp5', 'sp6']
    sys_argv_save = sys.argv.copy()
    if __name__ != '__main__':
        sys.argv = [__file__]
    main_call([
        f'input.path_raw_local="{file_raw_local}"',
        'input.time_interval=[2022-05-05T12:45, now]',
        'input.dt_from_utc_hours=2',
        'out.db_path="{}"'.format(str(path_db).replace('\\', '/')),
        'out.tables=[{}]'.format(','.join([f'"{d}"' for d in device])),

        'process.b_reprocess=True',
        'process.anchor_coord=[55.32659, 20.57875]',
        'process.anchor_tracker=[sp5]',
        '+process.max_dr_dict={sp6:200, sp6_ref_sp5: 100}',
        'process.anchor_depth=40',
        'process.period_tracks=1D'
        ])
    sys.argv = sys_argv_save


#%%
def test_call_example_sp5ref6__230825():
    """Loading two trackers data in one DB, calc distance between, recalc all"""

    path_db = Path(
        r'd:\WorkData\BalticSea\230825_Kulikovo@ADCP,ADV,i,tr\tracker_SPOT'.replace('\\', '/')
    )
    file_raw_local = str(path_db.parent / 'raw' / 'ActivityReport_test.xlsx').replace('\\', '/')
    device = ['sp6', 'sp5']

    cfg = {
        'DEVICE': 'sp5',
        'ANCHOR_DEVICE_NUM': 6,
        'ANCHOR_DEVICE_TYPE': 'sp'
    }
    cfg.update({
        'TYPE@DEVICE': 'current@{DEVICE}ref{ANCHOR_DEVICE_NUM}'.format_map(cfg),
        'file_stem': '230825@{DEVICE}ref{ANCHOR_DEVICE_NUM}'.format_map(cfg)
    })
    
    sys_argv_save = sys.argv.copy()
    if __name__ != '__main__':
        sys.argv = [__file__]
    args = """
    input.path_raw_local=None ^
    input.time_interval="[2023-08-25T11:20, now]" ^
    input.dt_from_utc_hours=2 ^
    process.anchor_coord="[54.989683, 20.301067]" ^
    process.anchor_tracker="[{ANCHOR_DEVICE_TYPE}{ANCHOR_DEVICE_NUM}]" ^
    process.anchor_depth=20 ^
    +process.max_dr_dict="{{{DEVICE}:200, {DEVICE}_ref_{ANCHOR_DEVICE_TYPE}{ANCHOR_DEVICE_NUM}:100}}" ^
    out.db_path='{dir_device}/{file_stem}.h5' ^
    out.tables=["{ANCHOR_DEVICE_TYPE}{ANCHOR_DEVICE_NUM}","{DEVICE}"] ^
    process.period_tracks=1D""".format_map(cfg).split(r' ^\n')
    main_call(args)
    sys.argv = sys_argv_save
#%%



path_db = Path(
    r'data/tracker/out/tracker.h5'
    # r'd:/workData/BalticSea/210515_tracker/current@sp2/210611_1300sp2.h5' #.replace('@', '\@')
    # r'd:\WorkData\BlackSea\210408_trackers\tr0\210408trackers.h5'
    ).absolute()


@pytest.mark.parametrize(
    'file_raw_local', [
        None,  # load from GMail
        # "c:/Users/and0k/AppData/Roaming/Thunderbird/Profiles/qwd33vkh.default-release/Mail/Local Folders/AB_SIO_RAS_tracker",
        # '"{}"'.format(str(path_db.with_name('ActivityReport.xlsx')).replace('\\', '/'))
        str(path_db.parent.with_name('SPOT_ActivityReport_sp4.xlsx')).replace('\\', '/'),
        str(path_db.parent.with_name('SPOT_ActivityReport_sp4_updated.xlsx')).replace('\\', '/')  # has 3 new rows for updating checking
    ])
def test_call_example_sp4(file_raw_local):
    device = ['sp4']  # 221912
    sys_argv_save = sys.argv.copy()
    sys.argv = [__file__]
    main_call([
        f'input.path_raw_local="{file_raw_local}"',
        'input.time_interval=[2021-06-02T13:49, now]',
        'input.dt_from_utc_hours=2',
        'process.anchor_coord=[54.616175, 19.84136]', #tr0: 54.616175, 19.84134166
        "++process.anchor_coord_time_dict={"
        r"2021-06-02T13\:50:[54.62355, 19.82249],"
        r"2021-06-02T14\:00:[54.62039, 19.83019],"
        r"2021-06-02T14\:10:[54.61916, 19.83516]"
        "}",
        'process.b_reprocess=True',
        'process.anchor_depth=15',
        'process.period_tracks=1D',
        'out.db_path="{}"'.format(str(path_db).replace('\\', '/')),
        'out.tables=[{}]'.format(','.join([f'"{d}"' for d in device]))
        ])
    sys.argv = sys_argv_save


def test_call_example_sp2(file_raw_local=None):
    device = ['sp2']
    sys_argv_save = sys.argv.copy()
    sys.argv = [__file__]
    main_call([
        f'input.path_raw_local="{file_raw_local}"',
        'input.time_interval=[2021-06-25T15:35:00, now]',
        'input.dt_from_utc_hours=2',
        'process.anchor_coord=[54.62457, 19.82311]', #tr0: 54.616175, 19.84134166
        'process.b_reprocess=True',
        'process.anchor_depth=15',
        'process.period_tracks=1D',
        'out.db_path="{}"'.format(str(path_db.with_name('tracker_sp2.h5')).replace('\\', '/')),
        'out.tables=[{}]'.format(','.join([f'"{d}"' for d in device]))
        ])
    sys.argv = sys_argv_save


def test_call_example_tr2(db_path=r'd:\Work\_Python3\And0K\h5toGrid\test\data\tracker\210726_1000tr2.h5'):
    device = ['tr2']
    sys_argv_save = sys.argv.copy()
    sys.argv = [__file__]
    main_call([
        'input.time_interval=[2021-07-26T10:00, now]',
        'input.dt_from_utc_hours=2',
        'process.anchor_coord=[54.62505, 19.82292]', #tr0: 54.616175, 19.84134166
        'process.anchor_depth=15',
        'process.period_tracks=1D',
        'out.db_path="{}"'.format(str(db_path).replace('\\', '/')),
        'out.tables=[{}]'.format(','.join([f'"{d}"' for d in device]))
        ])
    sys.argv = sys_argv_save


def test_call_example_tr2(db_path=r'd:\workData\BalticSea\210515_tracker\map-setup\test_accuracy\211024tr2.h5'):
    device = ['tr2']
    sys_argv_save = sys.argv.copy()
    sys.argv = [__file__]
    main_call([
        'input.time_interval=[2021-10-24T14:00, 2021-10-24T23:00]',
        'input.dt_from_utc_hours=2',
        'process.anchor_coord=[54.62505, 19.82292]', #tr0: 54.616175, 19.84134166
        'process.anchor_depth=15',
        'process.period_tracks=1D',
        'out.db_path="{}"'.format(str(db_path).replace('\\', '/')),
        'out.tables=[{}]'.format(','.join([f'"{d}"' for d in device]))
        ])
    sys.argv = sys_argv_save


# if __name__=='__main__':  # not works
#     sys.path.extend([
#         r'd:\Work\_Python3\And0K\h5toGrid',
#         r'd:\Work\_Python3\And0K\h5toGrid\to_pandas_hdf5',
#         r'd:\Work\_Python3\And0K\h5toGrid\to_pandas_hdf5\h5_dask_pandas'
#         ])
#     test_call_example_sp2to3()
