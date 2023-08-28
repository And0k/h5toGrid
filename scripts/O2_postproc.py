#!/usr/bin/env python
# coding: utf-8

import logging
import pandas as pd, numpy as np
from typing import Any, Callable, Dict, Iterator, Iterable, Mapping, MutableMapping, Optional, List, Sequence, Tuple, Union
# import vaex
# import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.ndimage.filters import gaussian_filter1d
# import gsw
from pathlib import Path
from collections import namedtuple
from utils2init import LoggingStyleAdapter
from to_pandas_hdf5.h5toh5 import ReplaceTableKeepingChilds

lf = LoggingStyleAdapter(logging.getLogger(__name__))

device = 'CTD_Idronaut_OS316#494'
path_db = Path(r'd:\WorkData\BalticSea\210701_ASV\210701_ASV.h5')


# Coefficients of dynamic correction formulas are based on expected Time constants: tau, tau0_lpf

def coefs_gain(tau: float) -> Tuple[float, float]:
    """ Coefficients (gain of differentiator)
    """
    phi = np.exp(-ts / tau)
    pci = 1 / (1 - phi)
    return pci, phi

# correction functions

def f_cor_exp(out_p: float, in_p: float, in_i: float, pci: float, phi: float) -> float:
    """ Exponetiation
    suffixes: _p - previous, _i - carrent
    """
    out_i = pci * (in_i - phi * in_p)
    return out_i


def f_cor_lpf(out_p: float, in_p: float, in_i: float, a: float, b: float) -> float:
    """ Smoothing
    suffixes: _p - previous, _i - carrent
    """
    out_i = a * (in_i + in_p) + b * out_p
    return out_i


def f_cor_exp_lpf(out_p: float, in_p: float, in_i: float, pci: float, phi: float, a: float, b: float) -> float:
    return f_cor_lpf(out_p, in_p, f_cor_exp(out_p, in_p, in_i, pci, phi), a, b)


def run_f_cor(in_arr: Sequence, fun, *args) -> List[float]:
    """
    Runs sequentially fun(out_p, in_p, in_i, *args)
    args: other fun arguments
    """
    out_p = in_arr[0]
    out_list = [out_p]
    for in_p, in_i in zip(in_arr[:-1], in_arr[1:]):
        out_p = fun(out_p, in_p, in_i, *args)
        out_list.append(out_p)
    return out_list


def run_f_cor_speed(in_arr: Sequence,
                    fun: Callable[[float, float, float, Iterable], float],
                    fun_tau: Callable[[float], Iterable], speed: float
                    ) -> List[float]:
    """
    Runs sequentially fun(out_p, in_p, in_i, *args)
    fun_tau: fun_tau(speed) calculates fun arguments *args depending on speed
    """
    out_p = in_arr[0]
    out_list = [out_p]
    for in_p, in_i, speed_i in zip(in_arr[:-1], in_arr[1:], speed[:-1]):
        out_p = fun(out_p, in_p, in_i, *fun_tau(speed_i))
        out_list.append(out_p)
    return out_list


def get_time_params(r: np.recarray, run: namedtuple) -> Tuple[int, int, float]:
    """

    :param r:
    :param run:
    :return: imax: index of run down end, iend: index of run up (back) point, ts: mean time sampling interval, s
    """
    tmin = r.Time[0]  # == df_log_sel.index[0]
    tmax = run.DateEnd.to_numpy()
    imin = np.searchsorted(r.Time, tmin)
    imax = np.searchsorted(r.Time, tmax)
    pmax = run.Pres_en
    # find ending upper point because of rows_filtered is wrong
    icross = imax + np.flatnonzero(np.diff(np.int8(r.Pres[imax:] > pmax / 3)) != 0)
    iend = icross[0] + r.Pres[slice(*icross[:2])].argmin()
    pmin = r.Pres[iend]
    i_o2min = imax + r.O2[imax:iend].argmin()
    o2min = r.O2[i_o2min]

    print('crossings of the mean depth:', icross)
    print(f'run start [{imin}]:\t', r.Time[0], ', Pres = ', run.Pres_st,
          f'\nrun max [{imax}]:\t', r.Time[imax], ', Pres = ', pmax, ', O2 = ', r.O2[imax],
          f'\nrun end [{iend}]:\t', r.Time[iend], ', Pres = ', pmin, ', O2_min = ', o2min, f'[{i_o2min}]',
          sep='')
    ts = ((tmax - tmin) / (imax - imin)).astype(float)/1E9  #.total_seconds()
    print('sampling interval:', ts)
    speed = gaussian_filter1d(np.ediff1d(r.Pres, 0) / np.ediff1d(r.Time.astype(np.int64) * 1E-9, 1), 5)
    return imax, iend, ts


def plot_time_series(r: np.recarray, imax: int, iend: int, speed=None):
    fig, ax = plt.subplots(figsize=(20, 5))
    # plt.tight_layout()
    if speed:
        ax.plot(r.Time[:iend], 100 * speed[:iend], **{'color': 'k', 'alpha': 0.5, 'ls': '--', 'label': 'speed'})
    ax.plot(r.Time[:imax], r.Pres[:imax], **{'color': 'k', 'label': 'Pres'})
    ax.plot(r.Time[:imax], r.O2[:imax], **{'color': 'k', 'label': 'O2'})
    ax.plot(r.Time[imax:iend], r.Pres[imax:iend], **{'color': 'k', 'label': 'Pres_up'})
    ax.plot(r.Time[imax:iend], r.O2[imax:iend], **{'color': 'k', 'label': 'O2_up'})


def filter_and_show_profile(r: np.recarray, imax: int, iend: int, ts: float, tau0_lpf: float=3,
                            taus: Optional[Sequence[float]]=None, t_delay: float=5,
                            f_cor_lpf: Callable[[Any], float]=f_cor_lpf,
                            f_cor_exp: Callable[[Any], float]=f_cor_exp) -> Tuple[List[np.ndarray], Figure]:
    """

    :param r: data numpy.recordarray
    :param imax: end of run down
    :param iend: end of run up
    :param ts:
    :param tau0_lpf: time constant, s: bigger -> more smoothing
    :param taus:
    :param t_delay:
    :param f_cor_lpf:
    :param f_cor_exp:
    :return: o2_v, fig
    """
    # Coefficients of LPF:
    a0 = 1 / (1 + 2 * tau0_lpf / ts)
    b0 = 1 - 2 * a0
    o2_lpf = np.float32(run_f_cor(r.O2[:iend], f_cor_lpf, a0, b0))

    o2_v = []  # variants
    if taus is None:
        taus = [13, 15]

    t_delay_counts = round(t_delay / ts)  # 35
    for tau in taus:
        o2_exp = np.float32(run_f_cor(o2_lpf, f_cor_exp, *coefs_gain(tau)))
        o2_exp = np.hstack([o2_exp[t_delay_counts:], np.empty(t_delay_counts) + np.NaN])
        #     o2_exp = np.float32(run_f_cor_speed(
        #         o2_lpf, f_cor_exp, lambda sp: coefs_gain(tau - tau_speed_coef*abs(sp)), speed
        #     ))
        o2_v.append(gaussian_filter1d(o2_exp, 2))

    fig, ax = plt.subplots(figsize=(20, 12))
    ax.yaxis.set_inverted(True)

    lines = []
    lines.append(ax.plot(o2_v[0][:imax], r.Pres[:imax], **{'color': 'k', 'label': f'down_tau={taus[0]}'}))
    lines.append(ax.plot(o2_v[0][imax:iend], r.Pres[imax:iend], **{'color': 'c', 'alpha': 0.5, 'label': 'up'}))

    lines.append(ax.plot(o2_v[1][:imax], r.Pres[:imax], **{'color': 'g', 'ls': '--', 'label': f'down_tau={taus[1]}'}))
    lines.append(ax.plot(o2_v[1][imax:iend], r.Pres[imax:iend], **{'color': 'b', 'alpha': 0.5, 'ls': '--', 'label': 'up'}))
    # plot inputed data too:
    lines.append(ax.plot(r.O2[:imax], r.Pres[:imax], **{'color': 'r', 'ls': '--', 'label': 'down_source'}))
    lines.append(ax.plot(r.O2[imax:iend], r.Pres[imax:iend], **{'color': 'm', 'alpha': 0.5, 'ls': '--', 'label': 'up'}))
    plt.xlim(-10, 160)
    ax.legend(prop={'size': 10}, loc='lower right')
    plt.show()
    return o2_v, fig


def save_fig(fig: Figure, r: np.recarray, taus: Sequence[float], t_delay: float):
    figname = f"{r.Time[0].astype('M8[s]').item():%y%m%d_%H%M}O2cor-tau={','.join(str(t) for t in taus)};dalay={t_delay:.2g}s.png"
    fig.savefig(path_db.with_name('_subproduct') / figname, format='png', dpi=300, transparent=False)
    print(f'figure {figname} saved')


def h5replace_cols(db_path: Union[Path, str], tbl: str, df_new: Union[pd.DataFrame, pd.Series]):
    with pd.HDFStore(db_path) as store:
        df = store.select(tbl)
        len_after = len(df_new)
        len_before = len(df)
        print(f'Replacing ({len_after}) rows of columns ', ','.join(df_new.columns),
              f'of dataframe ({len_before}) loaded from', db_path)
        df.loc[df_new.index, df_new.columns] = df_new
        len_after = len(df)
        print('Loading and saving...')
        if len_after > len_before:
            print(f'new {len_after - len_before} rows added')

        with ReplaceTableKeepingChilds(df, tbl, {'db': store}):
            pass



# For each run
tbl = f'/{device}'
tbl_log = f'{tbl}/logRuns'
qfmt = "index>='{}' & index<='{}'"
with pd.HDFStore(path_db, mode='r') as store:
    df_log = store[tbl_log]
    o2_ser_l = []  # accumulates result from each run
    for irow, run in enumerate(df_log.itertuples()):
        qstr = qfmt.format(run.Index, run.DateEnd)
        print(irow, run.rows, run.rows_filtered, qstr, end='… ')
        ind_st = store.select_as_coordinates(tbl, qstr)[0]
        try:
            # this would be load only run down but we need run up too
            # df = store.select(tbl, qstr, columns=['Pres', 'Temp90', 'Sal', 'O2', 'O2ppm'])
            df = store.select(
                tbl, start=ind_st,
                stop=ind_st + run.rows + run.rows_filtered,
                columns=['Pres', 'Temp90', 'Sal', 'O2', 'O2ppm']
            )
            print('Selected run loaded:', df)  # of length len(df)
            r = df.to_records()
            imax, iend, ts = get_time_params(r, run)
            plot_time_series(r, imax, iend, speed=None)

            t_delay = 5
            taus = (13, 15)
            o2_v, fig = filter_and_show_profile(r, imax, iend, ts, tau0_lpf=3, taus=taus, t_delay=t_delay)
            save_fig(fig, r, taus, t_delay)
            i_variant = 0
            o2_ser_l.append(pd.Series(o2_v[i_variant], index=df.index))

        except Exception as e:
            lf.exception('Error when query:  {}. ', qstr)
            continue


o2_ser = pd.concat(o2_ser_l)
h5replace_cols(path_db, tbl, o2_ser)
print('Ok>')


# with pd.HDFStore(path_db) as store:
#     qfmt = "index>=Timestamp('{}') and index<=Timestamp('{}')"
#     qstr = qfmt.format(*list(df_log_sel.DateEnd.items())[0])
#     t_start = df_log_sel.index.tz_localize(None)[0] #.to_pydatetime() df_log_sel.index[0].tz_localize(None)
#     ind_all = store.select_as_coordinates(tbl, f'index==Timestamp("{t_start}")')
# ind_all