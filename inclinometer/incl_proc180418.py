# @+leo-ver=5-thin
# @+node:korzh.20180610155307.5: * @file /mnt/D/Work/_Python3/And0K/h5toGrid/inclinometer/incl_proc180418.py
"""
Apply coef to P in hdf5 *inclPres.h5 and filter stored data.
Save hdf5 to *incl,P.h5
todo: export results to Veusz
"""
# @+<<declarations>>
# @+node:korzh.20180520131532.2: ** <<declarations>>
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, rcParams

rcParams['axes.linewidth'] = 1.5
rcParams['figure.figsize'] = (19, 7)

# import dask.dataframe as dd
import dask.array as da
from dask.diagnostics import ProgressBar  # or distributed.progress when using the distributed scheduler
# from dask.distributed import Client
# client = Client(processes=False)  # navigate to http://localhost:8787/status to see the diagnostic dashboard if you have Bokeh installed
# # processes=False: avoide inter-worker communication for computations releases the GIL (numpy, da.array)  # without is error
import sys
from pathlib import Path
from typing import Any, Sequence

# my:
__file__ = '/mnt/D/Work/_Python3/And0K/h5toGrid/scripts/incl_proc180418.py'
sys.path.append(str(Path(__file__).parent.parent.resolve()))  # os.getcwd()
from to_pandas_hdf5.csv2h5 import h5out_init, h5temp_open
from to_pandas_hdf5.h5toh5 import h5move_tables, h5index_sort
from to_pandas_hdf5.h5_dask_pandas import h5_append
from filters import rep2mean
from filters_scipy import despike

cfg = {  # output configuration after loading csv:
    'out': {
        'table': 'inclPres11',  # 'inclPres14',
        'db_path': '/mnt/D/workData/_source/BalticSea/180418/_source/180418inclPres.h5',
        'chunksize': None, 'chunksize_percent': 10  # we'll repace this with burst size if it suit
        }, 'in': {}}


# cfg = ini2dict(r'D:\Work\_Python3\_projects\PyCharm\h5toGrid\to_pandas_hdf5\csv_inclin_Baranov.ini')
# tblD = cfg['out']['table']
# tblL = tblD + '/log'
# dtAdd= np.timedelta64(60,'s')
# dtInterval = np.timedelta64(60, 's')

# @+others
# @+node:korzh.20180521124530.1: *3* functions
# @+others
# @+node:korzh.20180520182928.1: *4* load_data_intervals
def load_data_intervals(df_intervals, cfg_out):
    """
    Deprishiated! Use to_pandas_hdf5/h5_dask_pandas.h5q_ranges_gen instead
    :param df_intervals: dataframe, with:
        index - pd.DatetimeIndex for starts of intervals
        DateEnd - pd.Datetime col for ends of intervals
    :param cfg_out: dict, with fields:
        db_path, str
        table, str
        
    >>> df_intervals = pd.DataFrame({'DateEnd': np.max([t_edges[1], t_edges_Calibr[1]])},
                                 index=[np.min([t_edges[0], t_edges_Calibr[0]])])    
    ... a = load_data_intervals(df_intervals, cfg['out'])
    """
    qstr_range_pattern = "index>='{}' & index<='{}'"
    with pd.HDFStore(cfg_out['db_path'], mode='r') as storeIn:
        print("loading from {db_path}: ".format_map(cfg_out), end='')
        # Query table tblD by intervals from table tblL
        # dfL = storeIn[tblL]
        # dfL.index= dfL.index + dtAdd
        dfcum = pd.DataFrame()
        for n, r in enumerate(df_intervals.itertuples()):  # if n == 3][0]  # dfL.iloc[3], r['Index']= dfL.index[3]
            qstr = qstr_range_pattern.format(r.Index, r.DateEnd)  #
            dfcum = storeIn.select(cfg_out['table'], qstr)  # or dd.query?
            print(qstr)
    return dfcum


# @+node:korzh.20180520212556.1: *4* i_bursts_starts
def i_bursts_starts(tim, dt_between_blocks=None):
    """ Determine starts of burst in datafreame's index and mean burst size
    :param: tim, pd.datetimeIndex
    :param: dt_between_blocks, pd.Timedelta or None - minimum time between blocks.
            Must be greater than delta time within block
            If None then auto find: greater than min of two first intervals + 1s       
    return: (i_burst, mean_burst_size)
         i_burst - indexes of starts of bursts
         mean_burst_size - mean burst size
    
    >>> tim = pd.date_range('2018-04-17T19:00', '2018-04-17T20:10', freq='2ms').to_series()
    ... di_burst = 200000  # start of burst in tim i.e. burst period = period between samples in tim * period (period is a freq argument) 
    ... burst_len = 100
    ... ix = np.arange(1, len(tim) - di_burst, di_burst) + np.int32([[0], [burst_len]])
    ... tim = pd.concat((tim[st:en] for st,en in ix.T)).index
    ... i_bursts_starts(tim)
    (array([  0, 100, 200, 300, 400, 500, 600, 700, 800, 900]), 100.0)
    # same from i_bursts_starts(tim, dt_between_blocks=pd.Timedelta(minutes=2))
    """
    if isinstance(tim, pd.DatetimeIndex):
        tim = tim.values
    dtime = np.diff(tim)
    # ? else:
    # dtime = np.diff(tim.base)
    if dt_between_blocks is None:
        # Auto find it: greater interval than min of two first + constant.
        # Some intervals may be zero (in case of bad time resolution) so adding constant enshures that intervals between blocks we'll find is bigger than constant)
        dt_between_blocks = (dtime[0] if dtime[0] < dtime[1] else dtime[1]) + np.timedelta64(1, 's')
    elif isinstance(dt_between_blocks, pd.Timedelta):
        dt_between_blocks = dt_between_blocks.to_timedelta64()

    # indexes of burst starts
    i_burst = np.append(0, np.flatnonzero(dtime > dt_between_blocks) + 1)

    # calculate mean_block_size
    if len(i_burst) > 1:
        if len(i_burst) > 2:  # amount of data is sufficient to not include edge (likely part of burst) in statistics
            mean_burst_size = np.mean(np.diff(i_burst[1:]))
        if len(i_burst) == 2:  # select biggest of two burst parts we only have
            mean_burst_size = max(i_burst[1], len(tim) - i_burst[1])
    else:
        mean_burst_size = len(tim)

    # dtime_between_bursts = dtime[i_burst-1]     # time of hole  '00:39:59.771684'
    return i_burst, mean_burst_size


#   Veusz inline version of this (viv):
# dstime = np.diff(stime)
# burst_i = nonzero(dstime>(dstime[0] if dstime[1]>dstime[0] else dstime[1])*2)[0]+1  
# mean_burst_size = burst_i[1]-burst_i[0] if len(burst_i)>0 else np.diff(USEi[0,:])
# @+node:korzh.20180520185242.1: *4* filt_blocks_array
def filt_blocks_array(x, i_starts, func=None):
    """
    Filter each block of numpy array separate using provided function.
    :param x: numpy array, to filter
    :param i_starts: numpy array, indexes of starts of bocks
    :param func: other_filters.despike() used if None
    returns: numpy array of same size as x with bad values replased with NaNs
    
    """
    if func is None:
        func = lambda x: despike(x, offsets=(20, 5), blocks=len(x), ax=None, label=None)[0]

    y = da.from_array(x, chunks=(tuple(np.diff(np.append(i_starts, len(x))).tolist()),), name='filt')
    with ProgressBar():
        y_out = y.map_blocks(func, dtype=np.float64, name='blocks_arr').compute()
    return y_out

    # for ist_en in np.c_[i_starts[:-1], i_starts[1:]]:
    # sl = slice(*ist_en)
    # y[sl], _ = despike(x[sl], offsets=(200, 50), block=block, ax=None, label=None)
    # return y


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
def poly_load_apply(x: Sequence, pattern_path_of_poly: str, pattern_format_arg: Any = None) -> np.ndarray:
    """
    Load and apply calibraion (polynominal coefficients)
    :param x: input Sequence
    :param pattern_path_of_poly: path to numpy.loadtxt() formatted with next arg
    :param pattern_format_arg:
    :return: np.ndarray of same size as ``x``
    poly = '/mnt/D/workData/_experiment/_2018/inclinometer/180416Pcalibr/fitting_result(P#' + '11' + 'calibr_log).txt'
    >>> dfcum.P = poly_load_apply(y_filt, pattern_path_of_poly = '/mnt/D/workData/_experiment/_2018/inclinometer/180605Pcalibr/fitting_result(inclPres{}).txt', pattern_format_arg=cfg['out']['table'][-2:])

    """

    # pattern_path_of_poly ='/mnt/D/workData/_experiment/_2018/inclinometer/180416Pcalibr/fitting_result(P#{}calibr_log).txt')

    path_poly = Path(pattern_path_of_poly.format(pattern_format_arg))

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
# @+others
# @+node:korzh.20180520131532.3: ** process
# Load Dataframe: index(DateTimeIndex), P, U, Gx, Gz, Gy, Hx, Hy, Hz  # todo change to Ax,Ay,Az,Mx,My,Mz,P,Battery

t_start = pd.to_datetime('2018-04-17T00:00')  # 18 09:14
t_interval = pd.Timedelta(hours=10000)  # 40

df_intervals = pd.DataFrame({'DateEnd': [t_start + t_interval]}, index=[t_start])  # only 1 interval now
dfcum = load_data_intervals(df_intervals, cfg['out'])
i_burst, mean_burst_size = i_bursts_starts(dfcum.index, dt_between_blocks=pd.Timedelta(minutes=10))

### Pressure ###
# Filter
y_filt = filt_blocks_array(dfcum.P.values, i_burst)  # 6129
y_filt = filt_blocks_array(y_filt, i_burst, func=interpolate)
# plot2vert(dfcum.index, dfcum.P, y_filt, 'Loaded {} points'.format(dfcum.shape[0]), b_show_diff=False)

# plot2vert(dfcum.index, dfcum.P, y_filt, 'Presure: {} points'.format(dfcum.shape[0]), b_show_diff=False,  ylabel='P, counts')

dfcum.P = poly_load_apply(y_filt,
                          pattern_path_of_poly='/mnt/D/workData/_experiment/_2018/inclinometer/180605Pcalibr/fitting_result(inclPres{}).txt',
                          pattern_format_arg=cfg['out']['table'][-2:])
# @+node:korzh.20180608193421.1: ** save plot
plt.plot(dfcum.index, dfcum.P, '-k', linewidth=0.01)
plt.savefig(Path(cfg['out']['db_path']).with_name(cfg['out']['table'] + '600dpi').with_suffix('.png'),
            dpi=600)
# @+node:korzh.20180524090703.1: ** right columns order and dtypes
# they all are float64
cols_names_right_order = ('P', 'Ax', 'Ay', 'Az', 'Mx', 'My', 'Mz', 'Battery')
cols_int16 = set(cols_names_right_order[1:])

df = dfcum.rename(columns={"Gx": "Ax", "Gz": "Az", "Gy": "Ay",
                           "Hx": "Mx", "Hz": "Mz", "Hy": "My",
                           "U": "Battery"})[list(cols_names_right_order)]  # , inplace=True

cfg['out']['dtype'] = np.dtype(
    {'names': df.dtypes.keys(),
     'formats': [np.uint16 if k in cols_int16 else v for k, v in df.dtypes.items()]})
# astype() need to convert to simpler dict (df= pd.DataFrame(df[...], dtype=...) also not works):
cfg['out']['dtype'] = {k: v[0] for k, v in cfg['out']['dtype'].fields.items()}

try:
    df = df.astype(cfg['out']['dtype'], copy=False)
except ValueError as e:
    print("replacing bad values of integer columns by interpolated mean")


    def bad_and_message(data, fun_bad, msg_bad):
        b_bad = fun_bad(data)
        yes_bad = b_bad.any()
        if yes_bad:
            # b_bad = np.logical_not(b_finite)
            s_bad = sum(b_bad)
            print('{} {} in {} at {}'.format(s_bad, msg_bad, col, np.flatnonzero(b_bad)[:10]))
            # end = ' '
            return b_bad
        else:
            return None


    for col in cols_int16:
        b_bad = bad_and_message(data=df[col], fun_bad=np.isnan, msg_bad='nan vaues')
        b_bad2 = bad_and_message(data=df[col], fun_bad=np.isinf, msg_bad='inf vaues')
        if b_bad is None:
            if b_bad2 is None:
                continue
            b_bad = b_bad2
        elif b_bad2 is not None:
            b_bad |= b_bad2
        df[col] = rep2mean(df[col], np.logical_not(b_bad), df.index.astype('i8').astype('f8'))

df = df.astype(cfg['out']['dtype'], copy=False)


# @+node:korzh.20180521171338.1: ** save
def change_db_path(cfg, str_old='Pres.h5', str_new=',P(cal0605).h5'):
    if not cfg['db_path'].endswith(str_new):
        cfg['db_path'] = cfg['db_path'][:-len(str_old)] + str_new


change_db_path(cfg['out'])
log = {}
try:  # set chanks to mean data interval between holes
    cfg['out']['chunksize'] = int(mean_burst_size)  # np.median(np.diff(i_burst[:-1]))
except ValueError:  # some default value if no holes
    cfg['out']['chunksize'] = 100000

h5out_init(cfg['in'], cfg['out'])  # cfg['in'] = {}
try:
    cfg['out']['b_incremental_update'] = False  # not copy prev data: True not implemented
    df_log_old, store, cfg['out']['b_incremental_update'] = h5temp_open(**cfg['out'])
    # with pd.HDFStore(fileOut, mode='w') as store:
    # Append to Store
    if df.empty:  # log['rows']==0
        print('No data => skip file')
    h5_append(cfg['out'], df, log)
    b_appended = True
except Exception as e:
    b_appended = False
finally:
    store.close()

if b_appended:
    if store.is_open:
        print('Wait store is closing...')
        # from time import sleep
        # sleep(2)
    failed_storages = h5move_tables(cfg['out'])
    print('Ok.', end=' ')
    h5index_sort(cfg['out'], out_storage_name=f"{cfg['out']['db_path'].stem}-resorted.h5",
                 in_storages=failed_storages)

# @+node:korzh.20180520131532.4: ** garbage
# def main_gabage():
#     print('\n' + this_prog_basename(__file__), end=' started. ')
#     try:
#         cfg['in']= init_file_names(cfg['in'])
#     except Ex_nothing_done as e:
#         print(e.message)
#         return()
#
#     fGi = lambda Ax,Ay,Az,Ag,Cg,i: np.dot(Ag.T, (np.column_stack((Ax, Ay, Az))[
#                                                      slice(*i)] - Cg[0,:]).T)
#     strTimeUse =  [['2017-10-15T15:37:00', '2017-10-15T19:53:00']]
#     for ifile, nameFull in enumerate(cfg['in']['paths'], start=1):
#         nameFE = os_path.basename(nameFull)
#         print('{}. {}'.format(ifile, nameFE), end=': ')
#
#         a, stime = ImportFileCSV(nameFull)
#
#         iUseTime = np.searchsorted(stime, [np.array(s, 'datetime64[s]') for s in np.array(strTimeUse)])
#         # Ah_old = np.float64([[1,0,0],[0,1,0],[0,0,1]])
#         # Ch = np.float64([[0, 0, 0]])
#         # Hxyz = fGi(a['Mx'], a['My'], a['Mz'], Ah_old, Ch, iUseTime.flat)
#         Hxyz = np.column_stack((a['Mx'], a['My'], a['Mz']))[slice(*iUseTime.flat)].T
#         A, b = calibrate(Hxyz)
#         calibrate_plot(Hxyz, A, b)
#         A_str, b_str = coef2str(A, b)
#         print('calibration coefficients calculated:', '\nA = \n', A_str,
#             '\nb = \n', b_str )
#         return A_str, b_str

# if __name__ == '__main__':
# main()
# freq =5  # Hz
# interval = freq * 10 * 60  # 10 min
# start = interval
# df = dask.​dataframe.read_hdf(cfg['out']['db_path'], cfg['out']['table'], start=start, stop=start+interval, columns='P', sorted_index=True, lock=False, mode='r')  # chunksize=1000000,
# @-others

# @@language python
# @@tabwidth -4
# @-leo
