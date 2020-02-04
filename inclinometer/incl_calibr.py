"""
autocalibration of all found probes. export coefficients to hdf5
do not aignn (rotata) coefficient matrix to Nord / Graviy.
Need check if rotation exist?:
                # Q, L = A.diagonalize() # sympy
                # E_w, E_v = np.linalg.eig(E)
Probes data table contents: index,Ax,Ay,Az,Mx,My,Mz
"""
import logging
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Tuple, Dict

import numpy as np
import pandas as pd
from scipy import linalg

if __debug__:
    import matplotlib

    matplotlib.rcParams['axes.linewidth'] = 1.5
    matplotlib.rcParams['figure.figsize'] = (16, 7)
    try:
        matplotlib.use(
            'Qt5Agg')  # must be before importing plt (rases error after although documentation sed no effect)
    except ImportError:
        pass
    from matplotlib import pyplot as plt

    matplotlib.interactive(True)
    plt.style.use('bmh')

# import my functions:
try:
    scripts_path = str(Path(__file__).parent)
except Exception as e:  # if __file__ is wrong
    drive_d = Path(
        'D:/' if sys.platform == 'win32' else '/mnt/D')  # allows to run on both my Linux and Windows systems:
    scripts_path = drive_d.joinpath('Work/_Python3/And0K/h5toGrid/scripts')
sys.path.append(str(Path(scripts_path).parent.resolve()))
# from inclinometer.incl_calibr import calibrate, calibrate_plot, coef2str
from inclinometer.h5inclinometer_coef import h5copy_coef
from inclinometer.incl_h5clc import incl_calc_velocity_nodask
from utils2init import cfg_from_args, this_prog_basename, init_logging
from to_pandas_hdf5.h5toh5 import h5select, h5find_tables
from other_filters import despike
from graphics import make_figure

if __name__ != '__main__':
    l = logging.getLogger(__name__)


def my_argparser():
    """
    Configuration parser options and its description
    :return p: configargparse object of parameters
    """
    from utils2init import my_argparser_common_part

    p = my_argparser_common_part({'description':
                                      'Grid data from Pandas HDF5, VSZ files '
                                      'and Pandas HDF5 store*.h5'})

    p_in = p.add_argument_group('in', 'data from hdf5 store')
    p_in.add('--db_path', help='hdf5 store file path where to load source data and write resulting coef')  # '*.h5'
    p_in.add('--tables_list', help='tables names list or pattern to find tables to load data')
    p_in.add('--channels_list',
             help='channel can be "magnetometer" or "M" for magnetometer and any else for accelerometer',
             default='M, A')
    p_in.add('--chunksize_int', help='limit loading data in memory', default='50000')
    p_in.add('--timerange_list', help='time range to use')
    p_in.add('--timerange_nord_list', help='time range to zeroing nord. Not zeroing Nord if not used')

    # p_filter = p.add_argument_group('filter', 'limits amount of data loading')
    p_out = p.add_argument_group('output_files', 'where write resulting coef (additionally)')
    p_out.add('--output_files.db_path',
              help='hdf5 store file path where to write resulting coef. Writes to tables that names configured for input data (cfg[in].tables) in this file')

    p_prog = p.add_argument_group('program', 'program behaviour')
    p_prog.add('--return', default='<end>',  # nargs=1,
               choices=['<cfg_from_args>', '<gen_names_and_log>', '<end>'],
               help='<cfg_from_args>: returns cfg based on input args only and exit, <gen_names_and_log>: execute init_input_cols() and returns... - see main()')
    return (p)


def load_hdf5_data(store, cfg_in: Mapping[str, Any], t_intervals=None, table=None,
                   query_range_pattern="index>=Timestamp('{}') & index<=Timestamp('{}')"):
    """
    Load data
    :param cfg_in:
    :param t_intervals: even sequence of strings convertable to index type values. Each pair defines edjes of data that will be concatenated. 1st and last must be min and max values in sequence.
    :return:
    """
    # from h5_dask_pandas import h5q_interval2coord, h5_load_range_by_coord
    #
    # if t_interval is None:
    #
    # start_end = h5q_interval2coord(cfg['in'], t_interval[0])
    # a = h5_load_range_by_coord(cfg['in'], start_end)
    if t_intervals is None:
        t_intervals = cfg_in['timerange']
    if table is None:
        table = cfg_in['table']
    df_list = []
    n = len(t_intervals)
    if n > 2:
        query_range_pattern = '|'.join(f'({query_range_pattern.format(*query_range_lims)})' for query_range_lims in (
            (lambda x=iter(t_intervals): zip(x, x))())
                                       )
    df = h5select(store, table, query_range_lims=t_intervals[0::(n - 1)],
                  interpolate=None, query_range_pattern=query_range_pattern
                  )
    # for t_interval in (lambda x=iter(t_intervals): zip(x, x))():
    #     df_list.append(h5select(
    #         store, table, query_range_lims=t_interval,
    #         interpolate=None, query_range_pattern=query_range_pattern
    #         ))
    # df = pd.concat(df_list, copy=False)

    # with pd.HDFStore(cfg['in']['db_path'], mode='r') as storeIn:
    #     try:  # Sections
    #         df = storeIn[cfg['in']['table_sections']]  # .sort()
    #     except KeyError as e:
    #         l.error('Sections not found in {}!'.format(cfg['in']['db_path']))
    #         raise e

    return df


def channel_cols(channel: str) -> Tuple[str, str]:
    """
    Data columns names (col_str M/A) and coef letters (coef_str H/G) from parameter name (or its abbreviation)
    :param channel:
    :return: (col_str, coef_str)
    """
    if (channel == 'magnetometer' or channel == 'M'):
        col_str = 'M'
        coef_str = 'H'
    else:
        col_str = 'A'
        coef_str = 'G'
    return (col_str, coef_str)


def fG(Axyz, Ag, Cg):
    return Ag @ (Axyz - Cg)


# fG = lambda countsAx, countsAy, countsAz, Ag, Cg: np.dot(
#     Ag, np.float64((countsAx, countsAy, countsAz)) - Cg)
#
# fGi = lambda countsAx, countsAy, countsAz, Ag, Cg, i: np.dot(
#     Ag, np.float64((countsAx, countsAy, countsAz))[slice(*i)] - Cg)

# fGi = lambda Ax, Ay, Az, Ag, Cg, i: np.dot(
#     (np.column_stack((Ax, Ay, Az))[slice(*i)] - Cg[0, :]), Ag).T


def filter_channes(a3d: np.ndarray, a_time=None, fig=None, fig_save_prefix=None
                   ) -> Tuple[np.ndarray, np.ndarray, matplotlib.figure.Figure]:
    """
    despike a3d - 3 channels of data and plot data and overlayed results
    :param a3d: shape = (3,len)
    :param a_time:
    :param fig:
    :param fig_save_prefix: save figure to this path + 'despike({ch}).png' suffix
    :return: a3d[ :,b_ok], b_ok
    """
    # dim_channel = 0
    dim_length = 1

    blocks = np.minimum((21, 7), a3d.shape[dim_length])
    offsets = (1.5, 2)  # filters too many if set some < 3
    std_smooth_sigma = 4
    b_ok = np.ones((a3d.shape[dim_length],), np.bool8)
    if fig:
        fig.axes[0].clear()
        ax = fig.axes[0]
    else:
        ax = None
    for i, (ch, a) in enumerate(zip(('x', 'y', 'z'), a3d)):
        ax_title = f'despike({ch})'
        ax, lines = make_figure(y_kwrgs=((
            {'data': a, 'label': 'source', 'color': 'r', 'alpha': 1},
            )), ax_title=ax_title, ax=ax, lines='clear')
        # , mask_kwrgs={'data': b_ok, 'label': 'filtered', 'color': 'g', 'alpha': 0.7}
        b_nan = np.isnan(a)
        n_nans_before = b_nan.sum()
        b_ok &= ~b_nan

        # back and forward:
        a_f = np.float64(a[b_ok][::-1])
        a_f, _ = despike(a_f, offsets, blocks, std_smooth_sigma=std_smooth_sigma)
        a_f, _ = despike(a_f[::-1], offsets, blocks, ax, label=ch,
                         std_smooth_sigma=std_smooth_sigma, x_plot=np.flatnonzero(b_ok))
        b_nan[b_ok] = np.isnan(a_f)
        n_nans_after = b_nan.sum()
        b_ok &= ~b_nan

        # ax, lines = make_figure(y_kwrgs=((
        #     {'data': a, 'label': 'source', 'color': 'r', 'alpha': 1},
        # )), mask_kwrgs={'data': b_ok, 'label': 'filtered', 'color': 'g', 'alpha': 0.7}, ax=ax,
        #     ax_title=f'despike({ch})', lines='clear')

        ax.legend(prop={'size': 10}, loc='upper right')
        l.info('despike(%s, offsets=%s, blocks=%s) deleted %s',
               ch, offsets, blocks, n_nans_after - n_nans_before)
        plt.show()
        if fig_save_prefix:  # dbstop
            ax.figure.savefig(fig_save_prefix + (ax_title + '.png'), dpi=300, bbox_inches="tight")
        # Dep_filt = rep2mean(a_f, b_ok, a_time)  # need to execute waveletSmooth on full length

    # ax.plot(np.flatnonzero(b_ok), Depth[b_ok], color='g', alpha=0.9, label=ch)
    return a3d[:, b_ok], b_ok, ax.figure


def calc_vel_flat_coef(coef_nested: Mapping[str, Mapping[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """ Convert coef_nested in format of incl_calc_velocity() args"""
    arg = {}
    for ch, coefs in coef_nested.items():
        sfx = channel_cols(ch)[1].lower()
        for key, val in coefs.items():
            arg[('C' if key == 'b' else 'A') + sfx] = val
    return arg


def str_range(ranges, ind):
    return "'{}'".format(', '.join(f"'{t}'" for t in ranges[ind])) if ind in ranges else ''


# for copy/paste ##########################################################
def plotting(a):
    """
    plot source
    :param a:
    :return:
    """

    plt.plot(a['Mx'])  # , a['My'], a['Mz'])

    if False:
        msg = 'Loaded ({})!'.format(a.shape)
        fig = plt.figure(msg)
        ax1 = fig.add_subplot(112)

        plt.title(msg)
        plt.plot(a['Hx'].values, color='b')
        plt.plot(a['Hy'].values, color='g')
        plt.plot(a['Hz'].values, color='r')


def axes_connect_on_move(ax, ax2):
    canvas = ax.figure.canvas

    def on_move(event):
        if event.inaxes == ax:
            ax2.view_init(elev=ax.elev, azim=ax.azim)
        elif event.inaxes == ax2:
            ax.view_init(elev=ax2.elev, azim=ax2.azim)
        else:
            return
        canvas.draw_idle()

    c1 = canvas.mpl_connect('motion_notify_event', on_move)
    return c1


def plotEllipsoid(center, radii, rotation, ax=None, plotAxes=False, cageColor='b', cageAlpha=0.2):
    """Plot an ellipsoid"""
    make_ax = ax == None
    if make_ax:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)

    # cartesian coordinates that correspond to the spherical angles:
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    # rotate accordingly
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], rotation) + center

    if plotAxes:
        # make some purdy axes
        axes = np.array([[radii[0], 0.0, 0.0],
                         [0.0, radii[1], 0.0],
                         [0.0, 0.0, radii[2]]])
        # rotate accordingly
        for i in range(len(axes)):
            axes[i] = np.dot(axes[i], rotation)

        # plot axes
        for p in axes:
            X3 = np.linspace(-p[0], p[0], 100) + center[0]
            Y3 = np.linspace(-p[1], p[1], 100) + center[1]
            Z3 = np.linspace(-p[2], p[2], 100) + center[2]
            ax.plot(X3, Y3, Z3, color=cageColor)

    # plot ellipsoid
    ax.plot_wireframe(x, y, z, rstride=4, cstride=4, color=cageColor, alpha=cageAlpha)

    if make_ax:
        plt.show()
        plt.close(fig)
        del fig


def fit_quadric_form(s):
    '''
     Estimate quadric form parameters from a set of points.
    :param s: array_like
          The samples (M,N) where M=3 (x,y,z) and N=number of samples.
    :return: M, n, d : array_like, array_like, float
          The quadric form parameters in : h.T*M*h + h.T*n + d = 0

        References
        ----------
        .. [1] Qingde Li; Griffiths, J.G., "Least squares ellipsoid specific
           fitting," in Geometric Modeling and Processing, 2004.
           Proceedings, vol., no., pp.335-340, 2004
    '''

    # D (samples)
    D = np.array(
        [s[0] ** 2., s[1] ** 2., s[2] ** 2.,
         2. * s[1] * s[2], 2. * s[0] * s[2], 2. * s[0] * s[1],
         2. * s[0], 2. * s[1], 2. * s[2],
         np.ones_like(s[0])])

    # S, S_11, S_12, S_21, S_22 (eq. 11)
    S = np.dot(D, D.T)
    S_11 = S[:6, :6]
    S_12 = S[:6, 6:]
    S_21 = S[6:, :6]
    S_22 = S[6:, 6:]

    # inv(C) (Eq. 8, k=4)
    Cinv = np.array(  # C = np.array(
        [[0, 0.5, 0.5, 0, 0, 0],  # [[-1, 1, 1, 0, 0, 0],
         [0.5, 0, 0.5, 0, 0, 0],  # [1, -1, 1, 0, 0, 0],
         [0.5, 0.5, 0, 0, 0, 0],  # [1, 1, -1, 0, 0, 0],
         [0, 0, 0, -0.25, 0, 0],  # [0, 0, 0, -4, 0, 0],
         [0, 0, 0, 0, -0.25, 0],  # [0, 0, 0, 0, -4, 0],
         [0, 0, 0, 0, 0, -0.25]])  # [0, 0, 0, 0, 0, -4]])

    # v_1 (eq. 15, solution)
    E = np.dot(Cinv, S_11 - np.dot(S_12, np.dot(linalg.inv(S_22), S_21)))

    E_w, E_v = np.linalg.eig(E)

    v_1 = E_v[:, np.argmax(E_w)]
    if v_1[0] < 0: v_1 = -v_1

    # v_2 (eq. 13, solution)
    v_2 = np.dot(np.dot(-np.linalg.inv(S_22), S_21), v_1)

    # quadric-form parameters
    M = v_1[np.array([[0, 3, 4],
                      [3, 1, 5],
                      [4, 5, 2]], np.int8)]
    n = v_2[:-1, np.newaxis]
    d = v_2[3]

    return M, n, d


def calibrate(raw3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """

    :param raw3d:
    :return: combined scale factors and combined bias
    """

    # initialize values
    F = np.float64(1.0)  # Expected earth magnetic field intensity, default=1.
    # b = np.zeros([3, 1])
    # A_1 = np.eye(3)

    # Ellipsoid fit
    meanHxyz = np.mean(raw3d, 1)  # dfcum[['Hx', 'Hy', 'Hz']].mean()
    s = np.array(raw3d - meanHxyz[:, np.newaxis])  # dfcum[['Hx', 'Hy', 'Hz']] - meanHxyz).T
    M, n, d = fit_quadric_form(s)  # M= A.T*A.inv, n= 2*M*b, d= b.T*n/2 - F**2

    # Calibration parameters

    M_inv = linalg.inv(M)
    # combined bias:
    b = -np.dot(M_inv, n) + meanHxyz[:, np.newaxis]  # np.array()
    # scale factors, soft iron, and misalignments:
    # note: some implementations of sqrtm return complex type, taking real
    a2d = np.real(F / np.sqrt(np.dot(n.T, np.dot(M_inv, n)) - d) * linalg.sqrtm(M))

    return a2d, b


def coef2str(a2d: np.ndarray, b: np.ndarray) -> Tuple[str, str]:
    """
    Numpy text representation of matrix a2d and vector b
    :param a2d:
    :param b:
    :return:
    """
    A1e4 = np.round(np.float64(a2d) * 1e4, 1)
    A_str = 'float64([{}])*1e-4'.format(
        ',\n'.join(
            ['[{}]'.format(','.join(str(A1e4[i, j]) for j in range(a2d.shape[1]))) for i in range(a2d.shape[0])]))
    b_str = 'float64([[{}]])'.format(','.join(str(bi) for bi in b.flat))
    return A_str, b_str


def calibrate_plot(raw3d: np.ndarray, a2d: np.ndarray, b, fig=None, window_title=None, clear=True):
    """

    :param raw3d:
    :param a2d:
    :param b:
    :param fig:
    :param window_title:
    :param clear:
    :return:
    """
    make_fig = fig is None
    if make_fig:
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, subplot_kw={'projection': '3d'})
    else:
        ax1, ax2 = fig.axes
        if clear:
            ax1.clear()
            ax2.clear()
    if window_title:
        man = plt.get_current_fig_manager()
        man.canvas.set_window_title(window_title)
    # output data:
    s = np.dot(a2d, raw3d - b)  # s[:,:]

    # Calibrated magnetic ﬁeld measurements plotted on the
    # sphere manifold whose radius equals 1
    # the norm of the local Earth’s magnetic ﬁeld

    # ax = axes3d(fig)
    # ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title('source')
    ax1.scatter(raw3d[0, :], raw3d[1, :], raw3d[2, :], color='k', marker='.', s=0.2)
    # , alpha=0.1) # dfcum['Hx'], dfcum['Hy'], dfcum['Hz']
    # plot sphere
    # find the rotation matrix and radii of the axes
    U, c, rotation = linalg.svd(linalg.inv(a2d))
    radii = c  # np.reciprocal()
    plotEllipsoid(b.flatten(), radii, rotation, ax=ax1, plotAxes=True, cageColor='r', cageAlpha=0.1)

    # ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title('calibrated')
    # plot points
    ax2.scatter(s[0, :], s[1, :], s[2, :], color='g', marker='.', s=0.2)  # , alpha=0.2  # s is markersize,
    axes_connect_on_move(ax1, ax2)
    # plot unit sphere
    center = np.zeros(3, float)
    rotation = np.diag(np.ones(3, float))
    radii = np.ones(3, float)
    plotEllipsoid(center, radii, rotation, ax=ax2, plotAxes=True)

    # if make_fig:
    #     plt.show()
    #     plt.close(fig); del fig
    #     return None
    return fig


def zeroing_azimuth(store, tbl, coefs, cfg_in):
    """
    azimuth_shift_deg by calculating velocity (Ve, Vn) in cfg_in['timerange_nord'] interval of tbl data:
     taking median, calculating direction, multipling by -1
    :param store:
    :param tbl:
    :param coefs: dict with fields having values of array type with sizes:
    'Ag': (3, 3), 'Cg': (3, 1), 'Ah': (3, 3), 'Ch': array(3, 1), 'azimuth_shift_deg': (1,), 'kVabs': (n,)
    :param cfg_in: dict with fields:
        - timerange_nord
        - other, needed in load_hdf5_data() and optionally in incl_calc_velocity_nodask()
    :return: azimuth_shift_deg
    """
    l.debug('Zeroing Nord direction')
    df = load_hdf5_data(store, cfg_in, t_intervals=cfg_in['timerange_nord'], table=tbl)
    if df.empty:
        l.info('Zero calibration range out of data scope')
        return
    dfv = incl_calc_velocity_nodask(df, **coefs, cfg_filter=cfg_in, cfg_proc=
    {'calc_version': 'trigonometric(incl)', 'max_incl_of_fit_deg': 70})
    dfv.query('10 < inclination & inclination < 170', inplace=True)
    dfv_mean = dfv.loc[:, ['Ve', 'Vn']].median()
    # or df.apply(lambda x: [np.mean(x)], result_type='expand', raw=True)
    # df = incl_calc_velocity_nodask(dfv_mean, **calc_vel_flat_coef(coefs), cfg_in=cfg_in)

    # coefs['M']['A'] = rotate_z(coefs['M']['A'], dfv_mean.Vdir[0])
    azimuth_shift_deg = -np.degrees(np.arctan2(*dfv_mean.to_numpy()))
    l.info('Nord azimuth shifting coef. found: %s degrees', azimuth_shift_deg)
    return azimuth_shift_deg


# ###################################################################################
def main(new_arg=None):
    """
    1. Obtains command line arguments (for description see my_argparser()) that can be passed from new_arg and ini.file
    also.
    2. Calibrates configured by cfg['in']['channels'] channels ('accelerometer' and/or 'magnetometer') for axes coeff.,
    not 90deg axes / soft iron
    3. Wrong implementation - not use! todo: Rotates compas cfg['in']['timerange_nord']
    :param new_arg: returns cfg if new_arg=='<cfg_from_args>' but it will be None if argument
     argv[1:] == '-h' or '-v' passed to this code
    argv[1] is cfgFile. It was used with cfg files:

    :return:
    """

    global l

    cfg = cfg_from_args(my_argparser(), new_arg)
    if not cfg:
        return
    if cfg['program']['return'] == '<cfg_from_args>':  # to help testing
        return cfg

    l = init_logging(logging, None, cfg['program']['log'], cfg['program']['verbose'])
    l.info("%s(%s) channels: %s started. ",
           this_prog_basename(__file__), cfg['in']['tables'], cfg['in']['channels'])
    fig = None
    fig_filt = None
    channel = 'accelerometer'  # 'magnetometer'
    fig_save_dir_path = cfg['in']['db_path'].parent
    with pd.HDFStore(cfg['in']['db_path'], mode='r') as store:
        if len(cfg['in']['tables']) == 1:
            cfg['in']['tables'] = h5find_tables(store, cfg['in']['tables'][0])
        coefs = {}
        for itbl, tbl in enumerate(cfg['in']['tables'], start=1):

            l.info(f'{itbl}. {tbl}: ')
            a = load_hdf5_data(store, cfg['in'], table=tbl)
            # iUseTime = np.searchsorted(stime, [np.array(s, 'datetime64[s]') for s in np.array(strTimeUse)])
            coefs[tbl] = {}
            for channel in cfg['in']['channels']:
                print(f' channel "{channel}"', end=' ')
                (col_str, coef_str) = channel_cols(channel)
                vec3d = np.column_stack(
                    (a[col_str + 'x'], a[col_str + 'y'], a[col_str + 'z'])).T  # [slice(*iUseTime.flat)]
                if True:  # col_str == 'A'?
                    vec3d, b_ok, fig_filt = filter_channes(
                        vec3d, a.index, fig_filt, fig_save_prefix=f"{fig_save_dir_path / tbl}-'{channel}'")
                A, b = calibrate(vec3d)
                window_title = f"{tbl} '{channel}' channel ellipse"
                fig = calibrate_plot(vec3d, A, b, fig, window_title=window_title)
                fig.savefig(fig_save_dir_path / (window_title + '.png'), dpi=300, bbox_inches="tight")
                A_str, b_str = coef2str(A, b)
                l.info('Calibration coefficients calculated: \nA = \n%s\nb = \n%s', A_str, b_str)
                coefs[tbl][channel] = {'A': A, 'b': b}

            # Zeroing Nord direction
            if len(cfg['in']['timerange_nord']):
                coefs[tbl]['M']['azimuth_shift_deg'] = zeroing_azimuth(store, tbl, calc_vel_flat_coef(coefs[tbl]),
                                                                       cfg['in'])

    for cfg_output in (['in', 'output_files'] if cfg['output_files'].get('db_path') else ['in']):
        l.info(f"Write to {cfg[cfg_output]['db_path']}")
        for itbl, tbl in enumerate(cfg['in']['tables'], start=1):
            i_search = re.search('\d*$', tbl)
            for channel in cfg['in']['channels']:
                (col_str, coef_str) = channel_cols(channel)
                dict_matrices = {f'//coef//{coef_str}//A': coefs[tbl][channel]['A'],
                                 f'//coef//{coef_str}//C': coefs[tbl][channel]['b'],
                                 }
                if channel == 'M':
                    if coefs[tbl]['M'].get('azimuth_shift_deg'):
                        dict_matrices[f'//coef//{coef_str}//azimuth_shift_deg'] = coefs[tbl]['M']['azimuth_shift_deg']
                    # Coping probe number to coefficient to can manually check when copy manually
                    if i_search:
                        try:
                            dict_matrices['//coef//i'] = int(i_search.group(0))
                        except Exception as e:
                            pass

                h5copy_coef(None, cfg[cfg_output]['db_path'], tbl, dict_matrices=dict_matrices)


if __name__ == '__main__':
    # Calculation example
    timeranges = {
        30: ['2019-07-09T18:51:00', '2019-07-09T19:20:00'],
        12: ['2019-07-11T18:07:50', '2019-07-11T18:24:22'],
        5: ['2019-07-11T18:30:11', '2019-07-11T18:46:28'],
        4: ['2019-07-11T17:25:30', '2019-07-11T17:39:30'],
        }

    timeranges_nord = {
        # 30: ['2019-07-09T17:54:50', '2019-07-09T17:55:22'],
        # 12: ['2019-07-11T18:04:46', '2019-07-11T18:05:36'],
        }

    i = 14

    # multiple timeranges not supported so calculate one by one probe?
    probes = [i]
    main(['', '--db_path',
          r'd:\WorkData\_experiment\_2019\inclinometer\190710_compas_calibr-byMe\190710incl.h5',
          # r'd:\WorkData\_experiment\_2019\inclinometer\190320\190320incl.h5',
          # r'd:\WorkData\_experiment\_2018\inclinometr\181003_compas\181003compas.h5',
          '--channels_list', 'M,A',  # 'M,', Note: empty element cause calc of accelerometer coef.
          '--tables_list', ', '.join(f'incl{i:0>2}' for i in probes),
          #    'incl02', 'incl03','incl04','incl05','incl06','incl07','incl08','incl09','incl10','incl11','incl12','incl13','incl14','incl15','incl17','incl19','incl20','incl16','incl18',
          '--timerange_list', str_range(timeranges, i),
          '--timerange_nord_list', str_range(timeranges_nord, i),
          # '--timerange_list', "'2019-03-20T11:53:35', '2019-03-20T11:57:20'",
          # '--timerange_list', "'2019-03-20T11:49:10', '2019-03-20T11:53:00'",

          # "'2018-10-03T18:18:30', '2018-10-03T18:20:00'",
          # "'2018-10-03T17:18:00', '2018-10-03T17:48:00'",
          # "'2018-10-03T18:10:19', '2018-10-03T18:16:20'",
          # "'2018-10-03T17:13:09', '2018-10-03T18:20:00'",
          # "'2018-10-03T17:30:30', '2018-10-03T17:34:10'",
          # "'2018-10-03T17:27:00', '2018-10-03T17:50:00'",
          # "'2018-10-03T17:23:00', '2018-10-03T18:05:00'",?
          # "'2018-10-03T17:23:00', '2018-10-03T18:05:00'",
          # "'2018-10-03T17:23:00', '2018-10-03T18:07:00'",
          # "'2018-10-03T17:23:00', '2018-10-03T18:10:00'",
          # "'2018-10-03T17:45:30', '2018-10-03T18:09:00'",
          # "'2018-10-03T17:59:00', '2018-10-03T18:03:00'",
          # "'2018-10-03T18:23:00', '2018-10-03T18:28:20'",
          # "'2018-10-03T17:23:00', '2018-10-03T18:30:00'",
          # "'2018-10-03T17:23:00', '2018-10-03T18:03:00'",
          # "'2018-10-03T17:52:32', '2018-10-03T17:59:00'",
          # "'2018-10-03T18:21:00', '2018-10-03T18:24:00'",
          # "'2018-10-03T18:05:39', '2018-10-03T18:46:35'",
          # "'2018-10-03T17:23:40', '2018-10-03T17:24:55'",
          # "'2018-10-03T16:13:00', '2018-10-03T17:14:30'",

          ])

    # # Old variant not using argparser
    #
    # cfg = {'in': {
    #     'db_path': r'd:\workData\BalticSea\171003_ANS36\inclinometr\171015_intercal_on_board\#*.TXT',
    #     'use_timerange_list': ['2017-10-15T15:37:00', '2017-10-15T19:53:00'],
    #     'delimiter': ',',
    #     'skiprows': 13}}
    #
    # import pandas as pd
    # from utils2init import ini2dict
    # from inclinometer.h5inclinometer_coef import h5copy_coef, h5_rotate_coef
    # cfg = ini2dict(r'D:\Work\_Python3\_projects\PyCharm\h5toGrid\to_pandas_hdf5\csv_Baranov_inclin.ini')
    # tbl = cfg['output_files']['table']
    # tblL = tbl + '/log'
    # dt_add= np.timedelta64(60, 's')
    # dt_interval = np.timedelta64(60, 's')
    # fileInF= r'd:\WorkData\_experiment\_2017\inclinometr\1704calibration.h5'
    # with pd.HDFStore(fileInF, mode='r') as storeIn:
    #     # Query table tbl by intervals from table tblL. Join them
    #     dfL = storeIn[tblL]
    #     qstr_range_pattern = "index>=Timestamp('{}') & index<=Timestamp('{}')"
    #     dfL.index= dfL.index + dt_add
    #     dfcum= pd.DataFrame()
    #     for n, r in enumerate(dfL.itertuples()): # if n == 3][0]  # dfL.iloc[3], r['Index']= dfL.index[3]
    #         qstr = qstr_range_pattern.format(r.Index, r.Index + dt_interval) #r.DateEnd
    #         Dat = storeIn.select(tbl, qstr)
    #         dfcum= dfcum.append(Dat)
    #
    # if False:
    #     # plot source
    #     msg = 'Loaded ({})!'.format(dfcum.shape)
    #     fig= plt.figure(msg)
    #     ax1 = fig.add_subplot(112)
    #
    #     plt.title(msg)
    #     plt.plot(dfcum['Hx'].values, color='b')
    #     plt.plot(dfcum['Hy'].values, color='g')
    #     plt.plot(dfcum['Hz'].values, color='r')
    #
    #     # ax = dfcum[['Hx', 'Hy', 'Hz']].plot()
    #     # dfL['ones']= 1
    #     # plt.plot(dfL['ones'], color='r')
    #     # ax.set_ylabel('Magnetomer')
    #     # plt.show()
    #
    # raw3d = np.array(dfcum[['Hx', 'Hy', 'Hz']]).T
    # a2d, b = calibrate(raw3d)
    # calibrate_plot(raw3d, a2d, b)
    # h5copy_coef(r'd:\WorkData\_experiment\_2017\inclinometr\inclinometr.h5', fileInF, tbl, dict_matrices={'//coef//H//A': a2d, '//coef//H//C': b.T})
    # h5_rotate_coef(h5file_source, h5file_dest, tbl)  # rotate G/A
    # pass

# Example of 3D sphere(?) plot
# n_angles = 36
# n_radii = 8
#
# radii = np.linspace(0.125, 1.0, n_radii)
# angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
# angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)
#
# x = np.append(0, (radii*np.cos(angles)).flatten())
# y = np.append(0, (radii*np.sin(angles)).flatten())
# z = np.sin(-x*y)
#
# fig = plt.figure( figsize=(13,6))
# fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0)
# ax = fig.add_subplot(1, 2, 1, projection='3d')
# ax2 = fig.add_subplot(1, 2, 2, projection='3d')
#
# ax.plot_trisurf(x, y, z, cmap=matplotlib.cm.jet, linewidth=0.2)
# ax2.plot_trisurf(x, y, z, cmap=matplotlib.cm.viridis, linewidth=0.5)
