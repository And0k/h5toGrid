import numpy as np

from grid2d_vsz import *


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


def test_add_data_at_edges():
    """
    Adding repeated/interpolated data at edges to help gridding
    """
    cfg={'x_resolution_use': 0.1, 'y_resolution_use': 0.1}  # {m, km}
    ctd_dist = np.arange(0, 5, cfg['x_resolution_use'])  # km
    edge_depth = [20, 30, 40, 30, 20]

    ctd_prm = {
        'starts': (st := np.arange(0, ctd_dist.size, ctd_dist.size // len(edge_depth))),
        'ends': st - st[-1] + ctd_dist.size - 1,
        }
    ctd_depth = np.hstack([np.linspace(0, p_max, n_points) for p_max, n_points in zip(
        edge_depth, ctd_prm['ends'] - ctd_prm['starts'] + 1
        )])
    # period is 1.5 times smaller than run size:
    ctd_z = np.sin(ctd_dist*ctd_prm['starts'].size*1.5*np.pi / ctd_dist.max())
    ctd_with_adds = add_data_at_edges(
        ctd_dist=ctd_dist, ctd_depth=ctd_depth, ctd_z=ctd_z,
        ctd_prm=ctd_prm, edge_depth=ctd_depth[ctd_prm['ends']], edge_dist=ctd_dist[ctd_prm['ends']],
        ok_ctd=np.ones_like(ctd_dist, dtype=bool), ok_ends=np.ones_like(ctd_prm['ends'], dtype=bool),
        cfg=cfg, x_limits=ctd_dist[[0,-1]]
        )
    ctd_with_adds
