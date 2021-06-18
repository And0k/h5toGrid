"""
autocalibration of all found probes. export coefficients to hdf5.
Does not align (rotate) coefficient matrix to North / Gravity.
Need check if rotation exist?:
                # Q, L = A.diagonalize() # sympy
                # E_w, E_v = np.linalg.eig(E)
Probes data table contents: index,Ax,Ay,Az,Mx,My,Mz
"""
import logging
import re
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Tuple, Dict, List, Union
from omegaconf import MISSING
import hydra

import numpy as np
import pandas as pd
from scipy import linalg
from dataclasses import dataclass, field

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
from utils2init import this_prog_basename, standard_error_info, LoggingStyleAdapter, set_field_if_no
# init_file_names, Ex_nothing_done, this_prog_basename, standard_error_info, dir_create_if_need, ini2dict, FakeContextIfOpen, set_field_if_no
from to_pandas_hdf5.h5toh5 import h5select, h5find_tables
from other_filters import is_works, despike
from graphics import make_figure

from to_vaex_hdf5.cfg_dataclasses import hydra_cfg_store, ConfigInHdf5_Simple, ConfigProgram, main_init, main_call, main_init_input_file  #  ConfigProgram is used indirectly
#from to_vaex_hdf5.h5tocsv import main_call


lf = LoggingStyleAdapter(logging.getLogger(__name__))
VERSION = '0.0.1'


# @dataclass hydra_conf(hydra.conf.HydraConf):
#     run: field(default_factory=lambda: defaults)dir
hydra.output_subdir = 'cfg'
# hydra.conf.HydraConf.output_subdir = 'cfg'
# hydra.conf.HydraConf.run.dir = './outputs/${now:%Y-%m-%d}_${now:%H-%M-%S}'


@dataclass
class ConfigFilter:
    """
    "filter": excludes some data:

    no_works_noise: is_works() noise argument for each channel: excludes data if too small changes
    blocks: List[int] ='21, 7' despike() argument
    offsets: List[float] despike() argument
    std_smooth_sigma: float = 4, help='despike() argument
    """
    #Optional[Dict[str, float]] = field(default_factory= dict) leads to .ConfigAttributeError/ConfigKeyError: Key 'Sal' is not in struct
    min_date: Optional[Dict[str, str]] = field(default_factory=dict)
    max_date: Optional[Dict[str, str]] = field(default_factory=dict)
    no_works_noise: Dict[str, float] = field(default_factory=lambda: {'M': 10, 'A': 100})
    blocks: List[int] = field(default_factory=lambda: [21, 7])
    offsets: List[float] = field(default_factory=lambda: [1.5, 2])
    std_smooth_sigma: float = 4


@dataclass
class ConfigOut:
    """
    "out": all about output files:

    db_path: hdf5 store file path where to write resulting coef. Writes to tables that names configured for input data (cfg[in].tables) in this file
    """
    db_paths: Optional[List[str]] = field(default_factory=list)
    aggregate_period_s: List[Any] = field(default_factory=lambda: [None, 2, 600, 7200])


# @dataclass
# class DictTimeRange(ConfigInHdf5_Simple):
#     03: List[str] = field(default_factory=list)


@dataclass
class ConfigInHdf5InclCalibr(ConfigInHdf5_Simple):
    """
    Same as ConfigInHdf5_Simple + specific (inclinometr calibration) data properties:
    channels: List: (, channel can be "magnetometer" or "M" for magnetometer and any else for accelerometer',
    chunksize: limit loading data in memory (default='50000')
    time_range_list: time range to use
    time_range_dict: time range to use for each inclinometer number (consisted of digits in table name)')
    time_range_nord_list: time range to zeroing north. Not zeroing Nord if not used')
    time_range_nord_dict: time range to zeroing north for each inclinometer number (consisted of digits in table name)')

    """
    db_path: Optional[str] = MISSING
    probes: List[str] = field(default_factory=list)
    probes_prefix: str = MISSING
    raw_pattern: str = "*{prefix:}{number:0>2}*.[tT][xX][tT]"

    channels: List[str] = field(default_factory=lambda: ['M', 'A'])
    chunksize: int = 50000
    # time_range: List[str] = field(default_factory=list)
    time_range: Dict[int, Any] = field(default_factory=dict)  # Dict[int, str] not supported in omegaconf
    time_range_nord: List[str] = field(default_factory=list)
    time_range_nord_dict: Dict[str, str] = field(default_factory=dict)  # Dict[int, str] not supported in omegaconf
    # Dict[Optional[str], Optional[List[str]]]

@dataclass
class ConfigProgramSt(ConfigProgram):
    step_start: int = 1
    step_end: int = 1


cs_store_name = Path(__file__).stem
cs, ConfigType = hydra_cfg_store(
    f'base_{cs_store_name}',
    {
        'input': ['in_hdf5__incl_calibr'],  # Load the config ConfigInHdf5_InclCalibr from the config group "input"
        'out': ['out'],  # Set as MISSING to require the user to specify a value on the command line.
        'filter': ['filter'],
        'program': ['program_st'],
    },
    module=sys.modules[__name__]
    )


def load_hdf5_data(store, table=None, t_intervals=None,
                   query_range_pattern="index>=Timestamp('{}') & index<=Timestamp('{}')"):
    """
    Load data
    :param t_intervals: even sequence of datetimes or strings convertable to index type values. Each pair defines edges of data that will be concatenated. 1st and last must be min and max values in sequence.
    :param table:
    :return:
    """

    n = len(t_intervals) if t_intervals is not None else 0
    if n > 2:
        query_range_pattern = '|'.join(f'({query_range_pattern.format(*query_range_lims)})' for query_range_lims in (
            (lambda x=iter(t_intervals): zip(x, x))())
                                   )
    elif n < 2:
        query_range_pattern = None
        t_intervals = []
    df = h5select(store, table, query_range_lims=t_intervals[0::(n - 1)],
                  interpolate=None, query_range_pattern=query_range_pattern)
    return df


def fG(Axyz, Ag, Cg):
    return Ag @ (Axyz - Cg)


# fG = lambda countsAx, countsAy, countsAz, Ag, Cg: np.dot(
#     Ag, np.float64((countsAx, countsAy, countsAz)) - Cg)
#
# fGi = lambda countsAx, countsAy, countsAz, Ag, Cg, i: np.dot(
#     Ag, np.float64((countsAx, countsAy, countsAz))[slice(*i)] - Cg)

# fGi = lambda Ax, Ay, Az, Ag, Cg, i: np.dot(
#     (np.column_stack((Ax, Ay, Az))[slice(*i)] - Cg[0, :]), Ag).T


def filter_channes(a3d: np.ndarray, a_time=None, fig=None, fig_save_prefix=None,
                   blocks=(21, 7), offsets=(1.5, 2), std_smooth_sigma=4
                   ) -> Tuple[np.ndarray, np.ndarray, matplotlib.figure.Figure]:
    """
    Filter back and forward each column of a3d by despike()
    despike a3d - 3 channels of data and plot data and overlayed results
    :param a3d: shape = (3,len)
    :param a_time:
    :param fig:
    :param fig_save_prefix: save figure to this path + 'despike({ch}).png' suffix
    :param blocks: filter window width - see despike()
    :param offsets: offsets to std - see despike(). Note: filters too many if set some < 3
    :param std_smooth_sigma - see despike()
    :return: a3d[ :,b_ok], b_ok
    """

    dim_length = 1   # dim_channel = 0
    blocks = np.minimum(blocks, a3d.shape[dim_length])
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

        if len(offsets):
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
            lf.info('despike({:s}, offsets={}, blocks={}) deleted {:d}',
                    ch, offsets, blocks, n_nans_after - n_nans_before)
        plt.show()
        if fig_save_prefix:  # dbstop
            try:
                ax.figure.savefig(fig_save_prefix + (ax_title + '.png'), dpi=300, bbox_inches="tight")
            except Exception as e:
                lf.warning(f'Can not save fig: {standard_error_info(e)}')
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
    marker_size = 5  # 0.2
    ax1.set_title('source')
    ax1.scatter(raw3d[0, :], raw3d[1, :], raw3d[2, :], color='k', marker='.', s=marker_size)
    # , alpha=0.1) # dfcum['Hx'], dfcum['Hy'], dfcum['Hz']
    # plot sphere
    # find the rotation matrix and radii of the axes
    U, c, rotation = linalg.svd(linalg.inv(a2d))
    radii = c  # np.reciprocal()
    plotEllipsoid(b.flatten(), radii, rotation, ax=ax1, plotAxes=True, cageColor='r', cageAlpha=0.1)

    # ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title('calibrated')
    # plot points
    ax2.scatter(s[0, :], s[1, :], s[2, :], color='k', marker='.', s=marker_size)  # , alpha=0.2  # s is markersize,
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


def zeroing_azimuth(store, tbl, time_range_nord, coefs=None, cfg_in=None):
    """
    azimuth_shift_deg by calculating velocity (Ve, Vn) in cfg_in['time_range_nord'] interval of tbl data:
     taking median, calculating direction, multipling by -1
    :param time_range_nord:
    :param store:
    :param tbl:
    :param coefs: dict with fields having values of array type with sizes:
    'Ag': (3, 3), 'Cg': (3, 1), 'Ah': (3, 3), 'Ch': array(3, 1), 'azimuth_shift_deg': (1,), 'kVabs': (n,)
    :param cfg_in: dict with fields:
        - time_range_nord
        - other, needed in load_hdf5_data() and optionally in incl_calc_velocity_nodask()
    :return: azimuth_shift_deg
    """
    lf.debug('Zeroing Nord direction')
    df = load_hdf5_data(store, table=tbl, t_intervals=time_range_nord)
    if df.empty:
        lf.info('Zero calibration range out of data scope')
        return
    dfv = incl_calc_velocity_nodask(df, **coefs, cfg_filter=cfg_in, cfg_proc=
    {'calc_version': 'trigonometric(incl)', 'max_incl_of_fit_deg': 70})
    dfv.query('10 < inclination & inclination < 170', inplace=True)
    dfv_mean = dfv.loc[:, ['Ve', 'Vn']].median()
    # or df.apply(lambda x: [np.mean(x)], result_type='expand', raw=True)
    # df = incl_calc_velocity_nodask(dfv_mean, **calc_vel_flat_coef(coefs), cfg_in=cfg_in)

    # coefs['M']['A'] = rotate_z(coefs['M']['A'], dfv_mean.Vdir[0])
    azimuth_shift_deg = -np.degrees(np.arctan2(*dfv_mean.to_numpy()))
    lf.info('Nord azimuth shifting coef. found: {:f} degrees', azimuth_shift_deg)
    return azimuth_shift_deg


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


def dict_matrices_for_h5(coefs=None, tbl=None, channels=None):
    """

    :param coefs: some fields from: {'M': {'A', 'b', 'azimuth_shift_deg'} 'A': {'A', 'b', 'azimuth_shift_deg'}, 'Vabs0'}
    :param tbl: should include probe number in name. Example: "incl_b01"
    :param channels: some from default ['M', 'A']: magnetometer, accelerometer
    :return: dict_matrices
    """
    if channels is None:
        channels = ['M', 'A']

    # Fill coefs where not specified
    dummies = []
    b_have_coefs = coefs is not None
    if not b_have_coefs:
        coefs = {}  # dict(zip(channels,[None]*len(channels)))

    for channel in channels:
        if not coefs.get(channel):
            coefs[channel] = ({'A': np.identity(3) * 0.00173, 'b': np.zeros((3, 1))} if channel == 'M' else
                              {'A': np.identity(3) * 6.103E-5, 'b': np.zeros((3, 1))}
                              )
            if b_have_coefs:
                dummies.append(channel)
        if channel == 'M':
            if not coefs['M'].get('azimuth_shift_deg'):
                coefs['M']['azimuth_shift_deg'] = 180
                if b_have_coefs and not 'M' in dummies:
                    dummies.append('azimuth_shift_deg')  # only 'azimuth_shift_deg' for M channel is dummy

    if coefs.get('Vabs0') is None:
        coefs['Vabs0'] = np.float64([10, -10, -10, -3, 3, 70])
        if b_have_coefs:
            dummies.append('Vabs0')

    if dummies or not b_have_coefs:
        lf.warning('Coping coefficients, {:s} - dummy!', ','.join(dummies) if b_have_coefs else 'all')
    else:
        lf.info('Coping coefficients')

    # Fill dict_matrices with coefs values
    dict_matrices = {}
    if tbl:
        # Coping probe number to coefficient to can manually check when copy manually
        i_search = re.search('\d*$', tbl)
        if i_search:
            try:
                dict_matrices['//coef//i'] = int(i_search.group(0))
            except Exception as e:
                pass
        dict_matrices['//coef//Vabs0'] = coefs['Vabs0']


    for channel in channels:
        (col_str, coef_str) = channel_cols(channel)
        dict_matrices.update(
            {f'//coef//{coef_str}//A': coefs[channel]['A'],
             f'//coef//{coef_str}//C': coefs[channel]['b'],
             })
        if channel == 'M':
            dict_matrices[f'//coef//{coef_str}//azimuth_shift_deg'] = coefs['M']['azimuth_shift_deg']

    return dict_matrices


cfg = {}

@hydra.main(config_name=cs_store_name, config_path="cfg")  # adds config store cs_store_name data/structure to :param config
def main(config: ConfigType) -> None:
    """
    ----------------------------
    Calculates coefficients from
    data of Pandas HDF5 store*.h5
    and saves them back
    ----------------------------
    1. Obtains command line arguments (for description see my_argparser()) that can be passed from new_arg and ini.file
    also.
    2. Loads device data of calibration in laboratory from hdf5 database (cfg['in']['db_path'])
    2. Calibrates configured by cfg['in']['channels'] channels ('accelerometer' and/or 'magnetometer'): soft iron
    3. Wrong implementation - not use cfg['in']['time_range_nord']! todo: Rotate compass using cfg['in']['time_range_nord']
    :param config: returns cfg if new_arg=='<cfg_from_args>' but it will be None if argument
     argv[1:] == '-h' or '-v' passed to this code
    argv[1] is cfgFile. It was used with cfg files:


    """
    global cfg, l
    cfg = main_init(config, cs_store_name, __file__=None)
    cfg = main_init_input_file(cfg, cs_store_name)
    # input data tables may be defined by 'probes_prefix' and 'probes' fields of cfg['in']
    if cfg['in']['probes'] or not len(cfg['in']['tables']):
        if cfg['in']['probes']:
           cfg['in']['tables'] = [f"{cfg['in']['probes_prefix']}{probe:0>2}" for probe in cfg['in']['probes']]
        elif cfg['in']['probes_prefix']:
            cfg['in']['tables'] = [f"{cfg['in']['probes_prefix']}.*"]
        # else:  # default config
        #     cfg['in']['tables'] = ['.*']

    #h5init(cfg['in'], cfg['out'])
    #cfg['out']['dt_from_utc'] = 0
    # cfg = cfg_from_args(my_argparser(), new_arg)

    lf.info(
        "{:s}({:s}) for channels: {} started. ",
        this_prog_basename(__file__), ', '.join(cfg['in']['tables']), cfg['in']['channels']
        )
    fig = None
    fig_filt = None

    fig_save_dir_path = cfg['in']['db_path'].parent
    with pd.HDFStore(cfg['in']['db_path'], mode='r') as store:
        if len(cfg['in']['tables']) == 1:
            cfg['in']['tables'] = h5find_tables(store, cfg['in']['tables'][0])
        coefs = {}
        for itbl, tbl in enumerate(cfg['in']['tables'], start=1):
            probe_number = int(re.findall('\d+', tbl)[0])
            lf.info(f'{itbl}. {tbl}: ')
            if isinstance(cfg['in']['time_range'], Mapping):  # individual interval for each table
                if probe_number in cfg['in']['time_range']:
                    time_range = cfg['in']['time_range'][probe_number]
                else:
                    time_range = None
            else:
                time_range = cfg['in']['time_range']  # same interval for each table
            a = load_hdf5_data(store, table=tbl, t_intervals=time_range)
            # iUseTime = np.searchsorted(stime, [np.array(s, 'datetime64[s]') for s in np.array(strTimeUse)])

            # Calibrate channels of 'accelerometer' or/and 'magnetometer'
            coefs[tbl] = {}
            for channel in cfg['in']['channels']:
                print(f' channel "{channel}"', end=' ')
                (col_str, coef_str) = channel_cols(channel)

                # filtering # col_str == 'A'?
                if True:
                    b_ok = np.zeros(a.shape[0], bool)
                    for component in ['x', 'y', 'z']:
                        b_ok |= is_works(a[col_str + component], noise=cfg['filter']['no_works_noise'][channel])
                    lf.info('Filtered not working area: {:2.1f}%', (b_ok.size - b_ok.sum())*100/b_ok.size)
                    # vec3d = np.column_stack(
                    #     (a[col_str + 'x'], a[col_str + 'y'], a[col_str + 'z']))[:, b_ok].T  # [slice(*iUseTime.flat)]
                    vec3d = a.loc[b_ok, [col_str + 'x', col_str + 'y', col_str + 'z']].to_numpy(float).T
                    index = a.index[b_ok]

                    vec3d, b_ok, fig_filt = filter_channes(
                        vec3d, index, fig_filt, fig_save_prefix=f"{fig_save_dir_path / tbl}-'{channel}'",
                        blocks=cfg['filter']['blocks'],
                        offsets=cfg['filter']['offsets'],
                        std_smooth_sigma=cfg['filter']['std_smooth_sigma'])

                A, b = calibrate(vec3d)
                window_title = f"{tbl} '{channel}' channel ellipse"
                fig = calibrate_plot(vec3d, A, b, fig, window_title=window_title)
                fig.savefig(fig_save_dir_path / (window_title + '.png'), dpi=300, bbox_inches="tight")
                A_str, b_str = coef2str(A, b)
                lf.info('Calibration coefficients calculated: \nA = \n{:s}\nb = \n{:s}', A_str, b_str)
                coefs[tbl][channel] = {'A': A, 'b': b}

            # Zeroing Nord direction
            time_range_nord = cfg['in']['time_range_nord']
            if isinstance(time_range_nord, Mapping):
                time_range_nord = time_range_nord.get(probe_number)
            if time_range_nord:
                coefs[tbl]['M']['azimuth_shift_deg'] = zeroing_azimuth(
                    store, tbl, time_range_nord, calc_vel_flat_coef(coefs[tbl]), cfg['in'])
            else:
                lf.info('not zeroing North')
    # Write coefs to each of output tables named same as input
    for db_path in cfg['out']['db_paths']:
        db_path = Path(db_path)
        lf.info('Writing to {}', db_path)
        for itbl, tbl in enumerate(cfg['in']['tables'], start=1):
            # i_search = re.search('\d*$', tbl)
            # for channel in cfg['in']['channels']:
            #     (col_str, coef_str) = channel_cols(channel)
            #     dict_matrices = {f'//coef//{coef_str}//A': coefs[tbl][channel]['A'],
            #                      f'//coef//{coef_str}//C': coefs[tbl][channel]['b'],
            #                      }
            #     if channel == 'M':
            #         if coefs[tbl]['M'].get('azimuth_shift_deg'):
            #             dict_matrices[f'//coef//{coef_str}//azimuth_shift_deg'] = coefs[tbl]['M']['azimuth_shift_deg']
            #         # Coping probe number to coefficient to can manually check when copy manually
            #         if i_search:
            #             try:
            #                 dict_matrices['//coef//i'] = int(i_search.group(0))
            #             except Exception as e:
            #                 pass
            dict_matrices = dict_matrices_for_h5(coefs[tbl], tbl, cfg['in']['channels'])
            h5copy_coef(None, db_path, tbl, dict_matrices=dict_matrices)

    print('Ok>', end=' ')



# def main_call(
#         cmd_line_list: Optional[List[str]] = None,
#         fun: Callable[[], Any] = main
#         ) -> Dict:
#     """
#     Adds command line args, calls fun, then restores command line args
#     :param cmd_line_list: command line args of hydra commands or config options selecting/overwriting
#
#     :return: global cfg
#     """
#
#     sys_argv_save = sys.argv
#     if cmd_line_list is not None:
#         sys.argv += cmd_line_list
#
#     # hydra.conf.HydraConf.run.dir = './outputs/${now:%Y-%m-%d}_${now:%H-%M-%S}'
#     fun()
#     sys.argv = sys_argv_save
#     return cfg



if __name__ == '__main__':
    main()

    # Calculation examples:
    # inclinometer/190901incl_calibr.py
    # or here:
    if False:
        time_ranges = {
            30: ['2019-07-09T18:51:00', '2019-07-09T19:20:00'],
            12: ['2019-07-11T18:07:50', '2019-07-11T18:24:22'],
            5: ['2019-07-11T18:30:11', '2019-07-11T18:46:28'],
            4: ['2019-07-11T17:25:30', '2019-07-11T17:39:30'],
            }

        time_ranges_nord = {
            # 30: ['2019-07-09T17:54:50', '2019-07-09T17:55:22'],
            # 12: ['2019-07-11T18:04:46', '2019-07-11T18:05:36'],
            }

        i = 14

        # multiple time_ranges not supported so calculate one by one probe?
        probes = [i]
        main(['', '--db_path',
              r'd:\WorkData\_experiment\inclinometer\190710_compas_calibr-byMe\190710incl.h5',
              # r'd:\WorkData\_experiment\_2019\inclinometer\190320\190320incl.h5',
              # r'd:\WorkData\_experiment\_2018\inclinometr\181003_compas\181003compas.h5',
              '--channels_list', 'M,A',  # 'M,', Note: empty element cause calc of accelerometer coef.
              '--tables_list', ', '.join(f'incl{i:0>2}' for i in probes),
              #    'incl02', 'incl03','incl04','incl05','incl06','incl07','incl08','incl09','incl10','incl11','incl12','incl13','incl14','incl15','incl17','incl19','incl20','incl16','incl18',
              '--time_range_list', str_range(time_ranges, i),
              '--time_range_nord_list', str_range(time_ranges_nord, i),
              # '--time_range_list', "'2019-03-20T11:53:35', '2019-03-20T11:57:20'",
              # '--time_range_list', "'2019-03-20T11:49:10', '2019-03-20T11:53:00'",

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
    #     'use_time_range_list': ['2017-10-15T15:37:00', '2017-10-15T19:53:00'],
    #     'delimiter': ',',
    #     'skiprows': 13}}
    #
    # import pandas as pd
    # from utils2init import ini2dict
    # from inclinometer.h5inclinometer_coef import h5copy_coef, h5_rotate_coef
    # cfg = ini2dict(r'D:\Work\_Python3\_projects\PyCharm\h5toGrid\to_pandas_hdf5\csv_inclin_Baranov.ini')
    # tbl = cfg['out']['table']
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
