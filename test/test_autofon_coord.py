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


path_db = Path(
    r'd:/workData/BalticSea/210515_tracker/current@sp2/210611_1300sp2.h5' #.replace('@', '\@')
    # r'd:\WorkData\BlackSea\210408_trackers\tr0\210408trackers.h5'
    )

@pytest.mark.parametrize(
    'file_raw_local', [
     # "c:/Users/and0k/AppData/Roaming/Thunderbird/Profiles/qwd33vkh.default-release/Mail/Local Folders/AB_SIO_RAS_tracker",
     '"{}"'.format(str(path_db.with_name('ActivityReport.xlsx')).replace('\\', '/'))
    ])
def test_call_example_sp4(file_raw_local):

    device = ['sp4']  # 221912

    sys_argv_save = sys.argv
    sys.argv = [__file__]

    main_call([
        f'input.path_raw_local="{file_raw_local}"',
        'input.time_interval=[2021-06-02T13:49, now]',
        'input.dt_from_utc_hours=2',
        'process.anchor_coord=[54.616175, 19.84136]', #tr0: 54.616175, 19.84134166
        'process.anchor_depth=15',
        'process.period_tracks=1D',
        r'out.db_path="{}"'.format(str(path_db).replace('\\', '/')),
        'out.tables=[{}]'.format(','.join([f'"{d}"' for d in device]))
        ])

    sys.argv = sys_argv_save