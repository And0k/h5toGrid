# @+leo-ver=5-thin
# @+node:korzh.20180610155307.4: * @file /mnt/D/Work/_Python3/And0K/h5toGrid/inclinometer/incl_unVeuszed-dask.py
"""
Zeroing inclinometr and write coef to hdf5 db
Load data by specified interval and
    Calcuate angles
    Calcuate wave parameters: spectrum and its statistics - if Pressure data exists
    Write to csv
"""
# @+<<declarations>>
# @+node:korzh.20180520131532.2: ** <<declarations>>
# import dask.dataframe as dd
# from dask.distributed import Client
# client = Client(processes=False)  # navigate to http://localhost:8787/status to see the diagnostic dashboard if you have Bokeh installed
# # processes=False: avoide inter-worker communication for computations releases the GIL (numpy, da.array)  # without is error
# @+<<declarations>>
# @+node:korzh.20180520131532.2: ** <<declarations>>
import sys
from pathlib import Path

# import dask.dataframe as dd
import dask.array as da
# from dask.distributed import Client
# client = Client(processes=False)  # navigate to http://localhost:8787/status to see the diagnostic dashboard if you have Bokeh installed
# # processes=False: avoide inter-worker communication for computations releases the GIL (numpy, da.array)  # without is error
import h5py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, rcParams

rcParams['axes.linewidth'] = 1.5
rcParams['figure.figsize'] = (19, 7)

# my:
drive_d = 'D:' if sys.platform == 'win32' else '/mnt/D'  # allows to run on both my Linux and Windows systems:
scripts_path = Path(drive_d + '/Work/_Python3/And0K/h5toGrid/scripts')
sys.path.append(str(Path(scripts_path).parent.resolve()))
# sys.path.append( str(Path(__file__).parent.parent.resolve()) ) # os.getcwd()

# from utils2init import ini2dict
# from scripts.incl_calibr import calibrate, calibrate_plot, coef2str
# from other_filters import despike, rep2mean

cfg = {  # output configuration after loading csv:
    'output_files': {
        'table': 'inclPres11',  # 'inclPres14',
        'db_path': '/mnt/D/workData/_source/BalticSea/180418/_source/180418inclPres.h5',
        'chunksize': None, 'chunksize_percent': 10  # we'll repace this with burst size if it suit
        }, 'in': {}}


# cfg = ini2dict(r'D:\Work\_Python3\_projects\PyCharm\h5toGrid\to_pandas_hdf5\csv_inclin_Baranov.ini')
# tblD = cfg['output_files']['table']
# tblL = tblD + '/log'
# dtAdd= np.timedelta64(60,'s')
# dtInterval = np.timedelta64(60, 's')

# @+others
# @+node:korzh.20180521124530.1: *3* functions
# @+others
# @+node:korzh.20180520182928.1: *4* load_data_intervals

# @+node:korzh.20180604165309.1: *4* interpolate
def interpolate(x):
    """
    Interpolate (as pandas interpolate() do)
    """
    bOk = np.isfinite(x)
    if np.all(bOk):
        return x
    return np.interp(np.arange(len(x)), np.flatnonzero(bOk), x[bOk])


# @+node:korzh.20180520173058.1: *4* plot2vert
def plot2vert(x, y=None, y_new=None, title='', b_show_diff=True, ylabel='P, dBar'):
    """
    :param y: points
    :param y_new: line
    example:
        plot2vert(dfcum.index, dfcum.P, y_filt, 'Loaded {} points'.format(dfcum.shape[0]), b_show_diff=False)
    """
    ax1 = plt.subplot(211 if b_show_diff else 111)
    ax1.set_title(title)
    if y is None:
        b_show_diff = False
    else:
        plt.plot(x, y, '.b')  # ,  label='data'
    if y_new is None:
        b_show_diff = False
    else:
        plt.plot(x, y_new, 'g--')  # , label='fit'
    ax1.grid(True)
    plt.ylabel(ylabel)
    plt.legend()

    if b_show_diff:
        plt.setp(ax1.get_xticklabels(), visible=False)

        ax2 = plt.subplot(212, sharex=ax1)  # share x only
        plt.plot(x, y_new - y, '.r', label='fit')
        plt.setp(ax2.get_xticklabels(), fontsize=6)
        ax2.grid(True)
        plt.ylabel('error, dBar')
        plt.xlabel('codes')

    plt.show()


# @+node:korzh.20180604164658.1: *4* poly_load_apply
def poly_load_apply(x, pattern_path_of_poly, pattern_format_arg):
    """
    Load and apply calibraion (polynominal coefficients)
    poly = '/mnt/D/workData/_experiment/_2018/inclinometr/180416Pcalibr/fitting_result(P#' + '11' + 'calibr_log).txt'
    >>> dfcum.P = poly_load_apply(y_filt, pattern_path_of_poly = '/mnt/D/workData/_experiment/_2018/inclinometr/180605Pcalibr/fitting_result(inclPres{}).txt', pattern_format_arg=cfg['output_files']['table'][-2:])
    """

    # pattern_path_of_poly ='/mnt/D/workData/_experiment/_2018/inclinometr/180416Pcalibr/fitting_result(P#{}calibr_log).txt')

    path_poly = Path(pattern_path_of_poly.format(cfg['output_files']['table'][-2:]))

    if not path_poly.exists():
        raise (FileNotFoundError(f'can not load coef from {path_poly}'))
    poly = np.loadtxt(path_poly, delimiter=',', dtype='f8')
    # [6.456200646941875e-14, -7.696552837506894e-09, 0.0010251691717479085, -2.900615915446312]
    # [3.67255551e-16,  1.93432957e-10,  1.20038165e-03, -1.66400137e+00]
    print(f'applying polynomial coef {poly}')
    return np.polyval(poly, x)  # pd.Series(, index=dfcum.index)


# @-others
# @-others
# @-<<declarations>>
# @afterref
# cloned
# @+<<more declarations>>
# @+node:korzh.20180602082319.1: ** <<more declarations>>
from incinometer.h5inclinometer_coef import h5_save_coef
from utils2init import Ex_nothing_done, standard_error_info

# input:
cfg['output_files']['table'] = 'incl10'  # 'inkl09' # 'inclPres11' #4
cfg['output_files']['db_path'] = '/mnt/D/workData/_source/BalticSea/180418/_source/180418inclPres.h5'

# optional external coef source:
# cfg['output_files']['db_coef_path']           # defaut is same as 'db_path'
# cfg['output_files']['table_coef'] = 'incl10'  # defaut is same as 'table'

# '/mnt/D/workData/_source/BalticSea/180418/_source/180418incl,P(cal0605).h5'  # 180418incl,P.h5
cfg['output_files']['chunksize'] = 50000

# @+others
# @+node:korzh.20180603070720.1: *3* functions
# @+others
# @+node:korzh.20180602125115.1: *4* h5q_interval2coord
from to_pandas_hdf5.h5_dask_pandas import h5q_interval2coord, h5_load_range_by_coord, export_df_to_csv


#   Veusz inline version of this (viv):
# dstime = np.diff(stime)
# i_burst = nonzero(dstime>(dstime[0] if dstime[1]>dstime[0] else dstime[1])*2)[0]+1  
# mean_burst_size = i_burst[1]-i_burst[0] if len(i_burst)>0 else np.diff(USEi[0,:])
# @+node:korzh.20180603091204.1: *4* zeroing
def zeroing(a_zeroing, Ag_old, Cg, Ah_old):
    """
    Zeroing: correct Ag_old, Ah_old
    Depreciated: copied to incl_h5ckc to remove dependance of wafo but modified there
    :param a_zeroing: dask.dataframe with columns 'Ax','Ay','Az'
    :param Ag_old, Cg: numpy.arrays, rotation matrix and shift for accelerometer
    :param Ah_old: numpy.array 3x3, rotation matrix for magnetometer
    return (Ag, Ah): numpy.arrays (3x3, 3x3), corrected rotation matrices
    """

    if not len(a_zeroing):
        print(f'zeroing(): no data {a_zeroing}, returning same coef')
        return Ag_old, Ah_old

    mean_countsG0 = da.atleast_2d(da.from_delayed(
        a_zeroing.loc[:, ('Ax', 'Ay', 'Az')].mean(
            ).values.to_delayed()[0], shape=(3,), dtype=np.float64, name='mean_G0'))  #
    Gxyz0old = fG(mean_countsG0, Ag_old, Cg)  # .compute()

    # Gxyz0old = delayed(fG, pure=True)(a.loc[:, ('Ax','Ay','Az')].mean().values, Ag_old, Cg).compute().compute()

    # Gxyz0old = [[np.NaN, np.NaN, np.NaN]] if np.nansum(isnan(iCalibr0V))>0 else \
    # fG(np.column_stack((np.nanmean(a['Ax'][slice(*iCalibr0V[0,:])]),
    # np.nanmean(a['Ay'][slice(*iCalibr0V[0,:])]),
    # np.nanmean(a['Az'][slice(*iCalibr0V[0,:])]) )), Ag_old, Cg)

    # fPitch = lambda Gxyz: -da.arctan2(Gxyz[0,:], da.sqrt(da.sum(da.square(Gxyz[1:,:]), 0)))
    # fRoll = lambda Gxyz: da.arctan2(Gxyz[1,:], Gxyz[2,:]) #da.arctan2(Gxyz[1,:], da.sqrt(da.sum(da.square(Gxyz[(0,2),:]), 0)))
    old1pitch = -fPitch(Gxyz0old).compute()
    old1roll = -fRoll(Gxyz0old).compute()
    print(
        f'zeroing pitch = {np.rad2deg(old1pitch[0])}, roll = {np.rad2deg(old1roll[0])} degrees ({old1pitch[0]}, {old1roll[0]} radians)')

    # 'coeficient'
    def rotate(A, old1pitch, old1roll):
        return np.transpose(np.dot(
            np.dot([[np.cos(old1pitch), 0, -np.sin(old1pitch)],
                    [0, 1, 0],
                    [np.sin(old1pitch), 0, np.cos(old1pitch)]],

                   [[1, 0, 0],
                    [0, np.cos(old1roll), np.sin(old1roll)],
                    [0, -np.sin(old1roll), np.cos(old1roll)]]),

            np.transpose(A)))

    Ag = rotate(Ag_old, old1pitch, old1roll)
    Ah = rotate(Ah_old, old1pitch, old1roll)

    # # test: should be close to zero:
    # Gxyz0 = fG(mean_countsG0, Ag, Cg)
    # #? Gxyz0mean = np.transpose([np.nanmean(Gxyz0, 1)])

    return Ag, Ah

    # a['P'].join() #columns=['time', 'inclination', 'direction', 'pressure'])


# @+node:korzh.20180608002823.1: *4*

# @+node:korzh.20180608095643.1: *4* waves_proc
def waves_proc(df, i_burst, len_avg=600, i_burst_good_exactly=1):
    if not 'Pressure' in a.columns:
        return

    import wafo.objects as wo
    import wafo.misc as wm

    print('waves_proc')

    def df2ts(series):
        return wo.mat2timeseries(
            pd.concat(
                [(series.index.to_series() - series.index.values[0]).dt.total_seconds(),
                 series
                 ], axis=1).values)

    # i_burst, mean_burst_size = i_bursts_starts(df.index, dt_between_blocks = pd.Timedelta(minutes=10))

    if len(df) - i_burst[-1] < len_avg:
        last_burst = i_burst[-1]
        i_burst = i_burst[:-1]
    else:
        last_burst = len(df)

    # use freqencies as calculated for normal burst:
    wfreq = df2ts(df['Pressure'][slice(*i_burst[i_burst_good_exactly:(i_burst_good_exactly + 2)])]).tospecdata(
        L=600).args
    Sp = np.empty((i_burst.size, wfreq.size)) + np.NaN
    df['P_detrend'] = df['Pressure'] + np.NaN
    calc_spectrum_characteristics = ['Hm0', 'Tm_10', 'Tp', 'Ss', 'Rs']
    ar_spectrum_characteristics = np.empty((i_burst.size, len(calc_spectrum_characteristics))) + np.NaN
    # Hm0 = 4*sqrt(m0)                          - Significant wave height
    # Tm_10 = 2*pi*m_1/m0                       - Energy period
    # Tp = 2*pi*int S(w)^4 dw / int w*S(w)^4 dw - Peak Period
    # Ss = 2*pi*Hm0/(g*Tm02^2)                  - Significant wave steepness
    # Rs = Quality control parameter
    i_st = 0
    for i, i_en in enumerate(np.append(i_burst[1:], last_burst)):
        sl = slice(i_st + 1, i_en)  # we have bad 1st data in each burst. removing it
        # ?! df['Pressure'][sl].apply(lambda x: wm.detrendma(x, sl.stop - sl.start)).values returns all [0] objects
        try:
            ps_detr = df['Pressure'][sl]
            ps_detr -= ps_detr.rolling(600, min_periods=1, center=True, win_type='blackman').mean()
            ps_detr.interpolate('nearest', inplace=True)
            df['P_detrend'][sl] = ps_detr

            ts = df2ts(ps_detr)
            Sest = ts.tospecdata(L=600)

            if np.allclose(wfreq, Sest.args):
                Sp[i, :] = Sest.data
                ar_spectrum_characteristics[i, :] = \
                    Sest.characteristic(fact=calc_spectrum_characteristics, T=diff(ts.args[[0, -1]]))[0]
            else:
                print(f'burst #{i} of different length ({i_en - i_st}) skipped')
                # Sest.plot('-')

                # # Probability distribution of wave trough period
                # dt = Sest.to_t_pdf(pdef='Tt', paramt=(0, 10, 51), nit=3)
                # T, index = ts.wave_periods(vh=0, pdef='d2u')
                # Tcrcr, ix = ts.wave_periods(vh=0, pdef='c2c', wdef='tw', rate=8)
                # Tc, ixc =   ts.wave_periods(vh=0, pdef='u2d', wdef='tw', rate=8)
                # bins = wm.good_bins(T, num_bins=25, odd=True)
                # wm.plot_histgrm(T, bins=bins, normed=True)
                # wave_parameters = ts.wave_parameters()
                # if wave_parameters['Ac']==[]:
                #    wave_parameters =
        except Exception as e:
            print(f'Error in burst #{i}: {e}')
        i_st = i_en
    ###

    # save calculated parameters
    export_df_to_csv(
        pd.DataFrame(ar_spectrum_characteristics, index=df.index[i_burst], columns=calc_spectrum_characteristics),
        cfg['output_files'],
        add_subdir='V,P_txt',
        add_suffix='spectrum_characteristics')

    export_df_to_csv(
        pd.DataFrame(Sp, index=df.index[i_burst], columns=wfreq),
        cfg['output_files'],
        add_subdir='V,P_txt',
        add_suffix='spectrum')

    # envelope ("ogibayushchaya") 
    ts = df2ts(df['P_detrend'].fillna(method='backfill'))
    ct = wm.findtc(ts.data, 0)[0]  # , kind='cw','tw'  # 0 is mean, 0 becouse it is derended
    ind_crest = ct[::2]
    ind_trough = ct[1::2]

    if False:
        plot(ps_detr, 'y')
        plot(ps_detr[ct[::2]], '.b')
        plot(ps_detr[ct[1::2]], '.r')
        ct += sl.start

        Tcrcr, ix = ts.wave_periods(vh=0, pdef='c2c', wdef='tw', rate=8)
        Tc, ixc = ts.wave_periods(vh=0, pdef='u2d', wdef='tw', rate=8)

        # @+others
        # @+node:korzh.20180608193514.1: *5* save plot
        # plt.setp(plt.gca().spines.values(), linewidth=1)
        fig = plt.figure();
        plt.gcf().set_size_inches(19, 7)
        ax = fig.add_axes([0.05, 0.05, 0.93, 0.93])
        ax.plot(df.index, df.P_detrend, '-k', linewidth=0.1)
        ax.plot(df.index[ind_crest], df.P_detrend[ind_crest], '-b', linewidth=1)
        ax.plot(df.index[ind_trough], df.P_detrend[ind_trough], '-r', linewidth=1)
        ax.set_ylim(-2, 2)
        ax.grid(True)
        # plt.savefig(Path(cfg['output_files']['db_path']).with_name(cfg['output_files']['table'] + '_P_detrend (600dpi)_18_1930-2300').with_suffix('.png'), dpi=600)
        # @-others

    return df, ind_crest, ind_trough


# @-others
# @-others
# @-<<more declarations>>
# @+<<Castom defenitions>>
# @+node:korzh.20180525085719.1: ** <<Castom defenitions>>
USEi = np.int64([[0, -1]])
USEtime = [['2018-04-18T10:00', '2019-04-19T09:00']]  # [['2018-04-17T20:00', '2018-04-28T00:00']]
USEcalibr0V_i = []
USEcalibr0V_time = [['2018-04-17T17:28', '2018-04-17T17:38']]  # [['2018-04-17T19:10', '2018-04-17T19:20']]
nAveragePrefer = 128  # 1200

# Filter
min_P = 0.5

# Displey
vecScale = 1
DISP_TimeZoom = [['2018-04-17T20:00', '2018-04-18T00:00']]

maxGsumMinus1 = 0.3  # 0.1
TimeShiftedFromUTC_s = 0

# Angles in radians:
fPitch = lambda Gxyz: -da.arctan2(Gxyz[0, :], da.sqrt(da.sum(da.square(Gxyz[1:, :]), 0)))
fRoll = lambda Gxyz: da.arctan2(Gxyz[1, :], Gxyz[2,
                                            :])  # da.arctan2(Gxyz[1,:], da.sqrt(da.sum(da.square(Gxyz[(0,2),:]), 0)))  #=da.arctan2(Gxyz[1,:], da.sqrt(da.square(Gxyz[0,:])+da.square(Gxyz[2,:])) )
fInclination = lambda Gxyz: da.arctan2(da.sqrt(da.sum(da.square(Gxyz[:-1, :]), 0)), Gxyz[2, :])
fHeading = lambda H, p, r: da.arctan2(H[2, :] * da.sin(r) - H[1, :] * da.cos(r),
                                      H[0, :] * da.cos(p) + (H[1, :] * da.sin(r) + H[2, :] * da.cos(r)) * da.sin(p))

fG = lambda Axyz, Ag, Cg: da.dot(Ag.T, (Axyz - Cg[0, :]).T)
fGi = lambda Ax, Ay, Az, Ag, Cg, i: da.dot(Ag.T, (da.column_stack((Ax, Ay, Az))[slice(*i)] - Cg[0, :]).T)

fbinningClip = lambda x, bin2_iStEn, bin1_nAverage: da.mean(da.reshape(x[slice(*bin2_iStEn)], (-1, bin1_nAverage)), 1)
fbinning = lambda x, bin1_nAverage: da.mean(da.reshape(x, (-1, bin1_nAverage)), 1)
repeat3shift1 = lambda A2: [A2[t:(len(A2) - 2 + t)] for t in range(3)]
median3cols = lambda a, b, c: da.where(a < b, da.where(c < a, a, da.where(b < c, b, c)),
                                       da.where(a < c, a, da.where(c < b, b, c)))
median3 = lambda x: da.hstack((np.NaN, median3cols(*repeat3shift1(x)), np.NaN))
# not convertable to dask easily:
fVabs_old = lambda Gxyz, kVabs: np.polyval(kVabs.flat, np.sqrt(np.tan(fInclination(Gxyz))))
rep2mean = lambda x, bOk: np.interp(np.arange(len(x)), np.flatnonzero(bOk), x[bOk], np.NaN, np.NaN)
fForce2Vabs_fitted = lambda x: da.where(x > 2, 2, da.where(x < 1, 0.25 * x, 0.25 * x + 0.3 * (x - 1) ** 4))
fIncl2Force = lambda incl: da.sqrt(da.tan(incl))
fVabs = lambda Gxyz, kVabs: fForce2Vabs_fitted(fIncl2Force(fInclination(Gxyz)))
f = lambda fun, *args: fun(*args)
positiveInd = lambda i, L: np.int32(da.where(i < 0, L - i, i))
minInterval = lambda iLims1, iLims2, L: f(
    lambda iL1, iL2: da.transpose([max(iL1[:, 0], iL2[:, 0]), min(iL1[:, -1], iL2[:, -1])]), positiveInd(iLims1, L),
    positiveInd(iLims2, L))
fStEn2bool = lambda iStEn, length: da.hstack(
    [(da.ones(iEn2iSt, dtype=np.bool8) if b else da.zeros(iEn2iSt, dtype=np.bool8)) for iEn2iSt, b in da.vstack((
        da.diff(
            da.hstack(
                (
                    0,
                    iStEn.flat,
                    length))),
        da.hstack(
            (
                da.repeat(
                    [
                        (
                            False,
                            True)],
                    np.size(
                        iStEn,
                        0),
                    0).flat,
                False)))).T])
TimeShift_Log_sec = 60

kVabs = np.float64([[0.361570991503], [0]])
# @-<<Castom defenitions>>
# @+<<loading>>
# @+node:korzh.20180525121734.1: ** <<loading>>
# @+others
# @+node:korzh.20180526160931.1: *3* coef
"""
Load or set default
Ag_old, Cg: scaling coefficients for inclinometer
Ah_old, Ch: scaling coefficients for magnitometer
"""
try:

    with h5py.File(cfg['output_files']['db_path_coef' if 'db_path_coef' in cfg['output_files'] else 'db_path']
            , "r") as h5source:
        tblD = cfg['output_files']['table_coef' if 'table_coef' in cfg['output_files'] else 'table']
        print(f'loading coefficient from {h5source.file.name}/{tblD}')
        Ag_old = h5source[tblD + '//coef//G//A'].value
        Cg = h5source[tblD + '//coef//G//C'].value
        Ah_old = h5source[tblD + '//coef//H//A'].value
        Ch = h5source[tblD + '//coef//H//C'].value
except Exception as e:
    print(standard_error_info(e), '- Can not load coef. Using default!\n')
    Ah_old = np.float64([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # /500.0
    Ag_old = np.float64([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) / 16384.0
    Cg = np.float64([[0, 0, 0]])
    Ch = np.float64([[0, 0, 0]])
# @-others

# @-<<loading>>
# @+others
# @+node:korzh.20180603123200.1: ** zeroing
try:

    if True:
        # zeroing
        start_end = h5q_interval2coord(cfg['output_files'], USEcalibr0V_time[0])
        a = h5_load_range_by_coord(cfg['output_files'], start_end)
        Ag, Ah = zeroing(a, Ag_old, Cg, Ah_old)
        if np.allclose(Ag, Ag_old):
            raise Ex_nothing_done('zeroing coefficients are not changed')
    else:
        print('redefine coef')
        Cg = np.float64([[0, 0, 0]])
        Ag = np.float64([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) / 16384.0
        Ch = np.float64([[24.8310570566, -112.332977452, -633.595234596]])
        Ah = np.float64([[53.1, -8.3, -1.6], [-8.3, 67.0, -1.0], [-1.6, -1.0, 85.9]]) * 1e-4

    b_update = True
    print('have new zeroing coefficients')
except Ex_nothing_done as ex:  # not need update except need copy
    if ('db_path_coef' in cfg['output_files']) or ('table_coef' in cfg['output_files']):
        print('copying loaded coef replacing Ag, Ah to specified Ag_old, Ah_old')
        Ag, Ah = Ag_old, Ah_old
        b_update = True
    else:
        print('not need update zeroing coefficients')
        b_update = False
if b_update:
    h5_save_coef(cfg['output_files'].get('db_path_coef'),
                 cfg['output_files']['db_path'],
                 tblD_source=cfg['output_files'].get('table_coef'),
                 tblD_dest=cfg['output_files']['table'],
                 dict_matrices={'//coef//H//A': Ah,
                                '//coef//G//A': Ag})
# @+node:korzh.20180610154356.1: ** |||break|||
raise (UserWarning('my break of calculation'))
# @+node:korzh.20180604143456.1: ** circle
all_data = pd.to_datetime(['2018-04-18T10:00', '2018-06-06T00:00'])  # use something like [-inf, inf] for all data
cfg['output_files']['period'] = 'Y'  # 'D'   # pandas. Offset strings (as D, 5D, H, ... )
t_interval_end = all_data[0]
t_intervals_start = pd.date_range(start=t_interval_end.normalize(), end=max(all_data[-1],
                                                                            t_interval_end.normalize() + pd_period_to_timedelta(
                                                                                cfg['output_files']['period'])),
                                  freq=cfg['output_files']['period'])  # make last t_interval_start >= all_data[-1]
n_intervals_without_data = 0

for t_interval_start in t_intervals_start:
    USEtime = [[t_interval_end.isoformat(), t_interval_start.isoformat()]]
    t_interval_end = t_interval_start
    try:
        # @+others
        # @+node:korzh.20180526160951.1: *3* load_interval
        start_end = h5_data_interval(cfg['output_files'], USEtime[0])
        a = h5_load_interval_dd(cfg['output_files'], start_end)
        # determine indexes of bursts starts

        tim = a.index.compute()
        i_burst, mean_burst_size = i_bursts_starts(tim, dt_between_blocks=None)

        if 'P' in a.columns:
            print('restricting time range by good Pressure')
            Pfilt = filt_blocks_da(a['P'].values, i_burst,
                                   i_end=len(a))  # removes warning 'invalid value encountered in less
            # with ProgressBar():
            Pfilt = Pfilt.compute()
            # a['P'].values = Pfilt
            # df.mask(lambda x: abs(x)>2, inplace=True) 
            bP = ~(Pfilt < min_P)  # a['P'].values
        else:
            bP = np.ones_like(a.index, np.bool8)

        iStEn_auto = np.flatnonzero(bP)[
            [0, -1]]  # ? np.atleast_2d((np.searchsorted(bP, 1), len(bP) - np.searchsorted(np.flipud(bP), 1)))

        iUseTime = np.int32(USEi)

        positiveInd = lambda i, L: np.int32(np.where(i < 0, L - i, i))
        minInterval = lambda iLims1, iLims2, L: f(
            lambda iL1, iL2: np.transpose([max(iL1[:, 0], iL2[:, 0]), min(iL1[:, -1], iL2[:, -1])]),
            positiveInd(iLims1, L), positiveInd(iLims2, L))

        iUseC = minInterval(np.atleast_2d(iStEn_auto), minInterval(np.int32(USEi), iUseTime, len(a)), len(a))[0]

        if np.any(iUseC - np.int32([0, len(a)])):
            print('processing interval {}: dropping ({}, {}) samples at edges'.format(tim[iUseC[[0, -1]]], iUseC[0],
                                                                                      len(a) - iUseC[-1]))
            tim = tim[iUseC[0]:iUseC[-1]]
            # ?! a.loc[tim,:]
            if 'P' in a.columns:
                a = a.drop(columns='P').join(pd.DataFrame(Pfilt[iUseC[0]:iUseC[-1]], index=tim, columns=['Pressure']),
                                             how='right')
            else:
                a = a.join(pd.DataFrame(index=tim), how='right')
            i_burst = i_burst[(iUseC[0] < i_burst) & (i_burst < iUseC[-1])] - iUseC[0]
        # @+node:korzh.20180525113836.1: *3* filter temperature
        if 'Temp' in a.columns:
            x = a['Temp'].map_partitions(np.asarray)
            blocks = np.diff(np.append(i_starts, len(x)))
            chunks = (tuple(blocks.tolist()),)
            y = da.from_array(x, chunks=chunks, name='tfilt')


            def interp_after_median3(x, b):
                return np.interp(
                    da.arange(len(bP), chunks=cfg_out['chunksize']),
                    da.flatnonzero(bP), median3(x[b]), da.NaN, da.NaN)


            b = da.from_array(bP, chunks=chunks, meta=('Tfilt', 'f8'))
            with ProgressBar():
                Tfilt = da.map_blocks(interp_after_median3(x, b), y, b).compute()

            # hangs:
            # Tfilt = dd.map_partitions(interp_after_median3, a['Temp'], da.from_array(bP, chunks=cfg_out['chunksize']), meta=('Tfilt', 'f8')).compute()   

            # Tfilt = np.interp(da.arange(len(bP)), da.flatnonzero(bP), median3(a['Temp'][bP]), da.NaN,da.NaN)
        # @+node:korzh.20180524213634.8: *3* main
        # @+others
        # @-others

        # 'mainparam'
        fG = lambda Axyz, Ag, Cg: np.dot(Ag.T, (Axyz - Cg[0, :]).T)
        fPitch = lambda Gxyz: -np.arctan2(Gxyz[0, :], np.sqrt(np.sum(np.square(Gxyz[1:, :]), 0)))
        fRoll = lambda Gxyz: np.arctan2(Gxyz[1, :], Gxyz[2, :])
        fInclination = lambda Gxyz: np.arctan2(np.sqrt(np.sum(np.square(Gxyz[:-1, :]), 0)), Gxyz[2, :])
        fHeading = lambda H, p, r: np.arctan2(H[2, :] * np.sin(r) - H[1, :] * np.cos(r), H[0, :] * np.cos(p) + (
                H[1, :] * np.sin(r) + H[2, :] * np.cos(r)) * np.sin(p))

        Gxyz = fG(a.loc[:, ('Ax', 'Ay', 'Az')].values.compute(), Ag, Cg)
        Hxyz = fG(a.loc[:, ('Mx', 'My', 'Mz')].values.compute(), Ah, Ch)

        """
        def out_calc(dfA, ):
            # a.loc[:,('Ax','Ay','Az')]
            Gxyz = fG(dfA.values, Ag, Cg))
            
            Hxyz = fG(a.loc[:,('Mx','My','Mz')].values, Ah, Ch))
        def out_calc(df):     
        """
        # Gxyz = a.loc[:,('Ax','Ay','Az')].map_partitions(lambda x,A,C: fG(x.values,A,C).T, Ag, Cg, meta=('Gxyz', float64))
        # Gxyz = da.from_array(fG(a.loc[:,('Ax','Ay','Az')].values, Ag, Cg), chunks = (3, 50000), name='Gxyz')
        # Hxyz = da.from_array(fG(a.loc[:,('Mx','My','Mz')].values, Ah, Ch), chunks = (3, 50000), name='Hxyz')
        # test: should be close to zero:
        GsumMinus1 = np.sqrt(np.sum(np.square(Gxyz), 0)) - 1
        HsumMinus1 = np.sqrt(np.sum(np.square(Hxyz), 0)) - 1

        # filter
        bad_g = np.abs(GsumMinus1) > maxGsumMinus1
        bad_g_sum = sum(bad_g)
        if bad_g_sum > 0.1 * len(GsumMinus1):
            print('Acceleration is bad in {}% cases!'.format(100 * bad_g_sum / len(GsumMinus1)))

        if False:
            # Velocity
            Vabs = fVabs_old(np.where(bad_g, np.NaN, Gxyz), kVabs)
            # DatasetPlugin('PolarToCartesian', {'x_out': Vn, 'units': np.degrees, 'r_in': Vabs, 'y_out': Ve, 'theta_in': Vdir})
            Vn = Vabs * np.cos(np.radians(Vdir))
            Ve = Vabs * np.sin(np.radians(Vdir))
            # np.degrees(np.arctan2(x1, x2[, out]))

        sPitch = fPitch(Gxyz)
        sRoll = fRoll(Gxyz)
        if ('Pressure' in a.columns) or ('Temp' in a.columns):
            if ('Temp' in a.columns) and not ('Pressure' in a.columns):
                col_may_replace = 'Temp'
            else:
                col_may_replace = 'Pressure'
            df = pd.DataFrame(
                [a[col_may_replace].compute(),
                 np.degrees(fInclination(Gxyz)),
                 np.degrees(np.arctan2(np.tan(sRoll), np.tan(sPitch)) + fHeading(Hxyz, sPitch, sRoll))]
                , columns=[col_may_replace, 'inclination', 'Vdir'],
                index=tim)  # ?.set_names('DateTime_UTC', inplace=True)
        else:
            df = pd.DataFrame(
                [
                    np.degrees(fInclination(Gxyz)),
                    np.degrees(np.arctan2(np.tan(sRoll), np.tan(sPitch)) + fHeading(Hxyz, sPitch, sRoll))]
                , columns=['inclination', 'Vdir'], index=tim)

        export_df_to_csv(df, cfg['output_files'], add_subdir='V,P_txt')

        df, ind_crest, ind_trough = waves_proc(df, i_burst, len_avg=600, i_burst_good_exactly=1)
        # @+node:korzh.20180525161640.1: *3* open_veusz_pages
        # @+<<declarations>>
        # @+node:korzh.20180527111005.1: *4* <<declarations>>
        if 'in' not in cfg: cfg['in'] = {}
        cfg['in']['pattern_path'] = '/mnt/D/workData/_source/BalticSea/180418/_source/180418#P_nodata.vsz'

        # path to /home/korzh/Python/PycharmProjects/veusz_experiments/veusz/veusz/embed.py
        if 'program' not in cfg: cfg['program'] = {}
        cfg['program']['veusz_path'] = '/home/korzh/.local/lib/python3.6/site-packages/veusz'
        # ? '/home/korzh/Python/PycharmProjects/veusz_experiments/veusz/veusz'
        #  '/home/korzh/.virtualenvs/veusz_experiments/lib/python3.6/site-packages/veusz-2.2.2-py3.6-linux-x86_64.egg/veusz'

        # @+others
        # @+node:korzh.20180608215123.1: *5* import veusz_embed
        try:
            from veusz import embed as veusz_embed
            # import embed
        except (ImportError, ModuleNotFoundError) as e:
            print(standard_error_info(e), '- Can not load module "embed". Trying add to sys.path first...')
            # python3 will require this
            if cfg['program']['veusz_path'] not in sys.path:
                sys.path = [cfg['program']['veusz_path']] + sys.path
            from veusz import embed as veusz_embed
        # @-others
        # @-<<declarations>>

        # veusz_embed = import_file(cfg['program']['veusz_path'], 'embed')

        # sys.path = [os.path.dirname(cfg['program']['veusz_path'])] + sys.path
        cfg['in']['pattern_path'] = '/mnt/D/workData/_source/BalticSea/180418/_source/180510_1653Pdetrend&envelop14.vsz'
        path_vsz = Path(cfg['in']['pattern_path'])
        print(f'opening {path_vsz}')
        veusze = veusz_embed.Embedded(path_vsz.name + ' - opened' if path_vsz.is_file() else ' - was created')
        veusze.Load(cfg['in']['pattern_path'])
        # run_module_main('veusz', fun='run')

        veusze.EnableToolbar()

        veusze.SetData('time', int64(df.index.values - datetime64('2009-01-01T00:00:00')) * 1E-9 + TimeShiftedFromUTC_s)
        veusze.SetData('P', df.P_detrend.values)
        veusze.SetData('ind_crest', ind_crest)
        veusze.SetData('ind_trough', ind_trough)
        veusze.SetData('i_burst', i_burst)
        path_vsz_save = Path(cfg['output_files']['db_path']).with_name(
            cfg['output_files']['table'] + '_P_detrend').with_suffix('.vszh5')
        print(f'saving {path_vsz_save.name} ...', end='')

        veusze.Save(str(path_vsz_save), mode='hdf5')  # veusze.Save(str(path_vsz_save)) saves time with bad resolution
        print(f'Ok')
        # @others
        # @-others
    except Ex_nothing_done as e:
        print(e.message)
        n_intervals_without_data += 1
    if n_intervals_without_data > 30:
        print('30 intervals without data => think it is the end')
        break
print('Ok')
# @-others

# @@language python
# @@tabwidth -4
# @-leo
