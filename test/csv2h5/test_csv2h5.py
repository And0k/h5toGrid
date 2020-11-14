#!/usr/bin/env python3
# coding:utf-8
"""
  Author:  Andrey Korzh <ao.korzh@gmail.com>
  Modified: 29.10.2020
  Passed: 29.10.2020
"""
import sys
import pytest
import unittest
from pathlib import Path
from datetime import datetime, timedelta
import dask.dataframe as dd
import numpy as np
import pandas as pd

test_path = Path('/mnt/D/Work/_Python3/And0K/h5toGrid/test')
sys.path.append(str(Path(test_path).parent.resolve()))

if not __debug__:
    from unittest import TestCase
    #from nose import main

# my
# from to_pandas_hdf5.csv2h5 import init_input_cols
from to_pandas_hdf5.csv2h5 import *


def test_init_input_cols():
    cfg_in = {
        'header':
            r"counts, Pres(float), Cond(float), Sh3i, Sh3, Sh1, Sh2i, Sh2, T1LPF71, Temp("
            r"float), Gx_m\s^2, Gy_m\s^2, Gz_m\s^2, TP, Time(float), dC_mS\cm_\m, dT1_°C\m, Sal, SigmaTh("
            r"float), N^2, eps3varSh_W\kg, Speed",
        'coltime': 14}
    cfg_in = init_input_cols(cfg_in)
    assert cfg_in['coltime'] == 14

    #     l_Del, b_Del = idata_from_tpoints(tst_approx, tdata, ist)
    #     assert l_Del == answer_l_Del
    #     np.testing.assert_array_equal(b_Del, answer_b_Del)
    #
    # for t in [1]:
    #     check_init_input_cols()



# TestIdata_from_tpoints.test_idata_from_tpoints()
# self.fail('Finish the test!')
def test_h5init():
    cfg_in = {'path': 'data/*.txt',
              'cfgFile': '200901incl_load.yml'
    }
    cfg_out = {}
    h5init(cfg_in, cfg_out)
    assert cfg_out == {
        'b_insert_separator': False,
        'logfield_fileName_len': 255,
        'chunksize': None,
        'b_skip_if_up_to_date': False,
        'b_remove_duplicates': False,
        'b_use_old_temporary_tables': True,
        'nfiles': 1,
        'tables': ['200901incl_load'],
        'tables_log': ['200901incl_load/logFiles'],
        'db_path': Path('data/data_out.h5'),
        'db_path_temp': Path('data/data_out_not_sorted.h5')
        }




cfg_filter = {'min_date': datetime.fromisoformat('2020-10-10'),
              'max_date': datetime.fromisoformat('2020-10-11')}
dict_log = {}

len_df = 20
df = pd.DataFrame(
    np.linspace(0, 10, num=len_df), columns=["Sample"],
    index=pd.date_range(cfg_filter['min_date'] - timedelta(hours=10), freq='5H', periods=len_df, tz='UTC'))

@pytest.fixture
def dask_df():
    return dd.from_pandas(df, npartitions=2)

@pytest.mark.parametrize('cfg_filter, dict_to_save_last_time', [(cfg_filter, dict_log)])
def test_set_filterGlobal_minmax(dask_df, cfg_filter, dict_to_save_last_time, log=None):

    out_dd, tim = set_filterGlobal_minmax(dask_df, cfg_filter, log=dict_log,
                            dict_to_save_last_time=dict_log)
    out = out_dd.compute()
    assert all(out.index == tim)
    assert min(out.index) > pd.Timestamp(cfg_filter['min_date'], tz='UTC')
    assert max(out.index) < pd.Timestamp(cfg_filter['max_date'], tz='UTC')
    assert dict_log['rows'] == 4
    assert dict_log['rows_filtered'] == 16
    assert dict_to_save_last_time['time_last'] == pd.Timestamp('2020-10-10 20:00:00+0000', tz='UTC', freq='5H')



if __name__ == '__main__':
    if __debug__:
        test_csv2h5()
    else:
        main()
