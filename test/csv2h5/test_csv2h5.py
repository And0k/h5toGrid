import sys
import unittest
from pathlib import Path

test_path = Path('/mnt/D/Work/_Python3/And0K/h5toGrid/test')
sys.path.append(str(Path(test_path).parent.resolve()))

if not __debug__:
    from unittest import TestCase
    #from nose import main

# my
# from to_pandas_hdf5.csv2h5 import init_input_cols
from to_pandas_hdf5.csv2h5 import *


class test_csv2h5(unittest.TestCase):
    """
    Select indexes i of ist_data for which tst_approx is between tst_data[i] and tst_data[i]+dt_point2run_max
    :param tst_approx: approx run start times
    :param tdata:    array of data's time values
    :param ist: data indexes of runs starts
    :param dt_point2run_max: timedelta, interval to select. If None then select to the next point
    :return: sel_run - list of selected run indices (0 means run started at tdata[ist_data[0]])
            sel_data - mask of selected data
    """

    def check_init_input_cols(self):
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


# class TestIdata_from_tpoints(TestCase):


# TestIdata_from_tpoints.test_idata_from_tpoints()
# self.fail('Finish the test!')
def test_h5init():
    cfg_in = {'path': '/mnt/D/workData/_experiment/_2018/inclinometr/180605Pcalibr/inclPres11/*.txt'}
    cfg_out = {}
    h5init(cfg_in, cfg_out)
    assert (cfg_out == \
            {'logfield_fileName_len': 255,
             'chunksize': None,
             'b_skip_if_up_to_date': False,
             'b_remove_duplicates': False,
             'b_use_old_temporary_tables': True,
             'nfiles': 1,
             'tables': ['/mnt/D/workData/_experiment/_2018/inclinometr/180605Pcalibr/inclPres1'],
             'tables_log': ['/mnt/D/workData/_experiment/_2018/inclinometr/180605Pcalibr/inclPres1/logFiles'],
             'db_dir': '/mnt/D/workData/_experiment/_2018/inclinometr/180605Pcalibr/inclPres1',
             'db_base': '/mnt/D/workData/_experiment/_2018/inclinometr/180605Pcalibr/inclPres1.h5',
             'db_path': '/mnt/D/workData/_experiment/_2018/inclinometr/180605Pcalibr/inclPres1.h5',
             'db_path_temp': '/mnt/D/workData/_experiment/_2018/inclinometr/180605Pcalibr/inclPres1_not_sorted.h5'})


if __name__ == '__main__':
    if __debug__:
        test_csv2h5()
    else:
        main()
