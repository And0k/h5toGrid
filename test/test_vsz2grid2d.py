import numpy as np

from grid2d_vsz import idata_from_tpoints


def test_idata_from_tpoints():
    """
    Select indexes i of ist_data for which tst_approx is between tst_data[i] and tst_data[i]+dt_point2run_max
    :param tst_approx: approx run start times
    :param tdata:    array of data's time values
    :param ist: data indexes of runs starts
    :param dt_point2run_max: timedelta, interval to select. If None then select to the next point
    :return: sel_run - list of selected run indices (0 means run started at tdata[ist_data[0]])
            sel_data - mask of selected data
    """
    tdata = np.array(
        ['2017-06-25T00:27', '2017-06-25T00:28', '2017-06-25T00:29', '2017-06-25T00:30', '2017-06-25T00:31',
         '2017-06-25T00:32', '2017-06-25T00:33', '2017-06-25T00:34'], dtype='datetime64[ns]')
    ist = [0, 2, 5]

    tst_in_out = [{'approx': tdata[ist][0], 'sel_run': [0], 'sel_data': [1, 1, 0, 0, 0, 0, 0, 0]},
                  {'approx': tdata[ist][1], 'sel_run': [1], 'sel_data': [0, 0, 1, 1, 1, 0, 0, 0]},
                  {'approx': [tdata[ist][0], tdata[5]], 'sel_run': [0, 2], 'sel_data': [1, 1, 0, 0, 0, 1, 1, 1]}]

    def check_idata_from_tpoints(tst_approx, tdata, ist, answer_l_Del, answer_b_Del):
        l_Del, b_Del = idata_from_tpoints(tst_approx, tdata, ist)
        assert l_Del == answer_l_Del
        np.testing.assert_array_equal(b_Del, answer_b_Del)

    for t in tst_in_out:
        yield check_idata_from_tpoints, t['approx'], tdata, ist, t['sel_run'], np.array(t['sel_data'], bool)


# class TestIdata_from_tpoints(TestCase):


# TestIdata_from_tpoints.test_idata_from_tpoints()
# self.fail('Finish the test!')
