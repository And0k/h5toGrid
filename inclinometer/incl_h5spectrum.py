#!/usr/bin/env python3
# coding:utf-8
"""
  Author:  Andrey Korzh <ao.korzh@gmail.com>
  Purpose: Calculate spectrum at specified time intervals
  Created: 25.06.2019

"""

import logging
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence, Tuple, Union

import netCDF4
import numpy as np
import pandas as pd
import xarray as xr
# from mne.time_frequency.multitaper import _compute_mt_params
from matplotlib import pyplot as plt
from scipy import signal
from numba import jit

# pd.set_option('io.hdf.default_format','table')
# from matplotlib import pyplot as plt, rcParams
# rcParams['axes.linewidth'] = 1.5
# rcParams['figure.figsize'] = (19, 7)

# my:
drive_d = 'D:' if sys.platform == 'win32' else '/mnt/D'  # allows to run on both my Linux and Windows systems:
scripts_path = Path(drive_d + '/Work/_Python3/And0K/h5toGrid/scripts')
sys.path.append(str(scripts_path.parent.resolve()))

# sys.path.append( str(Path(__file__).parent.parent.resolve()) ) # os.getcwd()

# from utils2init import ini2dict
# from scripts.incl_calibr import calibrate, calibrate_plot, coef2str
# from other_filters import despike, rep2mean

from utils2init import Ex_nothing_done, init_logging, cfg_from_args, init_file_names, my_argparser_common_part, call_with_valid_kwargs
from utils_time import intervals_from_period, pd_period_to_timedelta
from to_pandas_hdf5.h5toh5 import h5init, h5select
from to_pandas_hdf5.h5_dask_pandas import h5_load_range_by_coord, filter_local
# h5q_intervals_indexes_gen
from inclinometer.incl_h5clc import incl_calc_velocity_nodask, my_argparser, h5_names_gen, filt_data_dd

path_mne = Path(r'd:\Work\_Python3\And0K\h5toGrid\other\mne')
sys.path.append(str(path_mne)) #.parent.resolve()
# sep = ';' if sys.platform == 'win32' else ':'
# os_environ['PATH'] += f'{sep}{path_mne}'
# multitaper = import_file(path_mne / 'time_frequency', 'multitaper')


from third_party.mne.time_frequency import multitaper

prog = 'incl_h5spectrum'  # this_prog_basename(__file__)
version = '0.1.1'
if __name__ == '__main__':
    l = None  # see main()
else:
    l = logging.getLogger(__name__)


def my_argparser(varargs=None):
    """
    todo: implement
    :return p: configargparse object of parameters
    """
    if not varargs:
        varargs = {}

    varargs.setdefault('description', '{} version {}'.format(prog, version) + """
    ---------------------------------
    Load data from hdf5 table (or group of tables)
    Calculate new data (averaging by specified interval)
    Combine this data to new specified table
    ---------------------------------
    """)
    p = my_argparser_common_part(varargs, version)

    # Fill configuration sections
    # All argumets of type str (default for add_argument...), because of
    # custom postprocessing based of my_argparser names in ini2dict

    s = p.add_argument_group('in',
                             'Parameters of input files')
    s.add('--db_path', default='*.h5',  # nargs=?,
                             help='path to pytables hdf5 store to load data. May use patterns in Unix shell style')
    s.add('--tables_list',   help='table names in hdf5 store to get data. Uses regexp')
    s.add('--chunksize_int', help='limit loading data in memory', default='50000')
    s.add('--date_min',      help='time range min to use', default='2019-01-01T00:00:00')
    s.add('--date_max',      help='time range max to use')
    s.add('--fs_float',      help='sampling frequency of input data, Hz')

    s = p.add_argument_group('filter',
                             'Filter all data based on min/max of parameters')
    s.add('--min_dict',
        help='List with items in  "key:value" format. Filter out (set to NaN) data of ``key`` columns if it is below ``value``')
    s.add('--max_dict',
        help='List with items in  "key:value" format. Filter out data of ``key`` columns if it is above ``value``')

    s = p.add_argument_group('out',
                             'Parameters of output files')
    s.add('--out.db_path', help='hdf5 store file path')
    s.add('--table', default='psd',
        help='table name in hdf5 store to write data. If not specified then will be generated on base of path of input files. Note: "*" is used to write blocks in autonumbered locations (see dask to_hdf())')
    s.add('--split_period',
        help='pandas offset string (5D, H, ...) to process and output in separate blocks. Number of spectrums is split_period/overlap_float. Use big values to not split',
        default='100Y')

    s = p.add_argument_group('proc',
                             'Processing parameters')
    s.add('--overlap_float',
        help='period overlap ratio [0, 1): 0 - no overlap. 0.5 for default dt_interval')
    s.add('--time_intervals_center_list',
        help='list of intervals centers that need to process. Used only if if period is not used')
    s.add('--dt_interval_hours',
        help='time range of each interval. By default will be set to the split_period in units of suffix (hours+minutes)')
    s.add('--dt_interval_minutes')
    s.add('--fmin_float',  # todo: separate limits for different parameters
        help='min output frequency to calc')
    s.add('--fmax_float',
        help='max output frequency to calc')
    s.add('--calc_version', default='trigonometric(incl)',
        help='string: variant of processing Vabs(inclination):',
        choices=['trigonometric(incl)', 'polynom(force)'])
    s.add('--max_incl_of_fit_deg_float',
        help='Overwrites last coefficient of trigonometric version of g: Vabs = g(Inclingation). It corresponds to point where g(x) = Vabs(inclination) became bend down. To prevent this g after this point is replaced with line, so after max_incl_of_fit_deg {\Delta}^{2}y ≥ 0 for x > max_incl_of_fit_deg')

    s = p.add_argument_group('program',
                             'Program behaviour')
    s.add('--return', default='<end>', choices=['<return_cfg>', '<return_cfg_with_options>'],
        help='executes part of code and returns parameters after skipping of some code')

    return (p)

#@jit failed for n_signals, n_tapers, n_freqs = x_mt.shape and not defined weights
def _psd_from_mt_adaptive(x_mt: np.ndarray, eigvals, freq_mask, max_iter=150,
                          return_weights=False):
    r"""Use iterative procedure to compute the PSD from tapered spectra.

    .. note:: Modified from NiTime.

    Parameters
    ----------
    x_mt : array, shape=(n_signals, n_tapers, n_freqs)
       The DFTs of the tapered sequences (only positive frequencies)
    eigvals : array, length n_tapers
       The eigenvalues of the DPSS tapers
    freq_mask : array
        Frequency indices to keep
    max_iter : int
       Maximum number of iterations for weight computation
    return_weights : bool
       Also return the weights

    Returns
    -------
    psd : array, shape=(n_signals, np.sum(freq_mask))
        The computed PSDs
    weights : array shape=(n_signals, n_tapers, np.sum(freq_mask))
        The weights used to combine the tapered spectra

    Notes
    -----
    The weights to use for making the multitaper estimate, such that
    :math:`S_{mt} = \sum_{k} |w_k|^2S_k^{mt} / \sum_{k} |w_k|^2`
    """
    n_signals, n_tapers, n_freqs = x_mt.shape

    if len(eigvals) != n_tapers:
        raise ValueError('Need one eigenvalue for each taper')

    if n_tapers < 3:
        raise ValueError('Not enough tapers to compute adaptive weights.')

    rt_eig = np.sqrt(eigvals)

    # estimate the variance from an estimate with fixed weights
    psd_est = _psd_from_mt(x_mt, rt_eig[np.newaxis, :, np.newaxis])
    x_var = np.trapz(psd_est, dx=np.pi / n_freqs) / (2 * np.pi)
    del psd_est

    # allocate space for output
    psd = np.empty((n_signals, np.sum(freq_mask)))

    # only keep the frequencies of interest
    x_mt = x_mt[:, :, freq_mask]

    if return_weights:
        weights = np.empty((n_signals, n_tapers, psd.shape[1]))

    for i, (xk, var) in enumerate(zip(x_mt, x_var)):
        # combine the SDFs in the traditional way in order to estimate
        # the variance of the timeseries

        # The process is to iteratively switch solving for the following
        # two expressions:
        # (1) Adaptive Multitaper SDF:
        # S^{mt}(f) = [ sum |d_k(f)|^2 S_k(f) ]/ sum |d_k(f)|^2
        #
        # (2) Weights
        # d_k(f) = [sqrt(lam_k) S^{mt}(f)] / [lam_k S^{mt}(f) + E{B_k(f)}]
        #
        # Where lam_k are the eigenvalues corresponding to the DPSS tapers,
        # and the expected value of the broadband bias function
        # E{B_k(f)} is replaced by its full-band integration
        # (1/2pi) int_{-pi}^{pi} E{B_k(f)} = sig^2(1-lam_k)

        # start with an estimate from incomplete data--the first 2 tapers
        psd_iter = _psd_from_mt(xk[:2, :], rt_eig[:2, np.newaxis])

        err = np.zeros_like(xk)
        for n in range(max_iter):
            d_k = (psd_iter / (eigvals[:, np.newaxis] * psd_iter +
                               (1 - eigvals[:, np.newaxis]) * var))
            d_k *= rt_eig[:, np.newaxis]
            # Test for convergence -- this is overly conservative, since
            # iteration only stops when all frequencies have converged.
            # A better approach is to iterate separately for each freq, but
            # that is a nonvectorized algorithm.
            # Take the RMS difference in weights from the previous iterate
            # across frequencies. If the maximum RMS error across freqs is
            # less than 1e-10, then we're converged
            err -= d_k
            if np.max(np.mean(err ** 2, axis=0)) < 1e-10:
                break

            # update the iterative estimate with this d_k
            psd_iter = _psd_from_mt(xk, d_k)
            err = d_k

        if n == max_iter - 1:
            l.warning('Iterative multi-taper PSD computation did not converge.')

        psd[i, :] = psd_iter

        if return_weights:
            weights[i, :, :] = d_k

    if return_weights:
        return psd, weights
    else:
        return psd

@jit
def _psd_from_mt(x_mt, weights):
    """Compute PSD from tapered spectra.

    Parameters
    ----------
    x_mt : array
        Tapered spectra
    weights : array
        Weights used to combine the tapered spectra

    Returns
    -------
    psd : array
        The computed PSD
    """
    psd = weights * x_mt
    psd *= psd.conj()
    psd = psd.real.sum(axis=-2)
    psd *= 2 / (weights * weights.conj()).real.sum(axis=-2)
    return psd


def gen_intervals(starts_time: Union[np.ndarray, pd.Series], dt_interval: Any) -> Iterator[Tuple[Any, Any]]:
    for t_start_end in zip(starts_time, starts_time + dt_interval):
        yield t_start_end


def h5q_starts2coord(
        cfg_in: Mapping[str, Any],
        starts_time: Optional[Union[np.ndarray, pd.Series, list]] = None,
        dt_interval: Optional[timedelta] = None
        ) -> pd.Index:
    """
    Edge coordinates of index range query
    As it is nealy part of h5toh5.h5select() may be depreshiated? See Note
    :param starts_time: array or list with strings convertable to pandas.Timestamp
    :param dt_interval: pd.TimeDelta
    :param: cfg_in, dict with fields:
        db_path, str
        table, str
        time_intervals_start, to use instead _starts_time_ if it is None
    :return: ``qstr_range_pattern`` edge coordinates
    Note: can use instead:
    >>> from to_pandas_hdf5.h5toh5 import h5select
    ... with pd.HDFStore(cfg_in['db_path'], mode='r') as store:
    ...     df, bbad = h5select(store, cfg_in['table'], columns=None, query_range_lims=cfg_in['timerange'])

    """
    # qstr_range_pattern = f"index>=st[{i}] & index<=en[{i}]"
    if starts_time is None:
        starts_time = cfg_in['time_intervals_start']
    if not (isinstance(starts_time, list) and isinstance(starts_time[0], str)):
        starts_time = np.array(starts_time).ravel()
    if dt_interval is None:
        dt_interval = cfg_in['dt_interval']

    print(f"loading {len(starts_time)} ranges from {cfg_in['db_path']}/{cfg_in['table']}: ")
    ind_st_last = 0
    with pd.HDFStore(cfg_in['db_path'], mode='r') as store:
        table_pytables = store.get_storer(cfg_in['table']).table
        to_end = table_pytables.nrows
        for i, (st, en) in enumerate(gen_intervals(starts_time, dt_interval)):
            qstr = "index>=st & index<=en"
            ind_all = store.select_as_coordinates(cfg_in['table'], qstr, start=ind_st_last)
            # ind_lim = table_pytables.get_where_list("(index>=st) & (index<=en)", condvars={
            # "st": np.int64(lim), "en": np.int64(lim + dt_interval)}, start=st_last, step=table_pytables.nrows)
            nrows = len(ind_all)
            if nrows:
                l.debug('%d. [%s, %s] - %drows', i + 1, st, en, nrows)
                ind_st_en = ind_all[[0, -1]]  # .values
                ind_lim_last = ind_st_en[-1]
            else:  # no data
                # l.debug('%d. [%s, %s] - no data', i + 1, st, en)
                continue
            yield ind_st_en

    # [t_prev_interval_start.isoformat(), t_interval_start.isoformat()])


def h5_velocity_by_intervals_gen(cfg: Mapping[str, Any], cfg_out: Mapping[str, Any]
                                 ) -> Iterator[Tuple[str, Tuple[Any, ...]]]:
    """
    Loads data and calculates velocity: many intervals from many of hdf5 tables sequentially.
    :param cfg: dict with fields:
        ['proc']['dt_interval'] - numpy.timedelta64 time interval of loading data
        one group of fields:
            1.  'split_period', pandas interval str, as required by intervals_from_period() to cover all data by it
                'overlap'

            2.  'time_intervals_start' - manually specified starts of intercals

    :param cfg_out: fields must be provided:
        - see h5_names_gen(cfg_in, cfg_out) requirements
    :return:
    """
    # Prepare cycle
    if cfg_out.get('split_period'):
        # variant 1. genereate ragular intervals (may be with overlap)
        def gen_loaded(tbl):
            cfg['in']['table'] = tbl
            # To obtain ``t_intervals_start`` used in query inside gen_data_on_intervals(cfg_out, cfg)
            # we copy its content here:
            t_prev_interval_start, t_intervals_start = intervals_from_period(
                **cfg['in'], period=cfg_out['split_period'])
            if cfg['proc']['overlap']:
                dt_shifts = np.arange(0, 1, (1 - cfg['proc']['overlap'])) * pd_period_to_timedelta(
                    cfg_out['split_period'])
                t_intervals_start = (
                        t_intervals_start.to_numpy(dtype="datetime64[ns]")[np.newaxis].T + dt_shifts).flatten()
                if cfg['in']['date_max']:
                    idel = t_intervals_start.searchsorted(
                        np.datetime64(cfg['in']['date_max'] - pd_period_to_timedelta(cfg_out['split_period'])))
                    t_intervals_start = t_intervals_start[:idel]
                cfg['in']['time_intervals_start'] = t_intervals_start  # to save queried time - see main()
            for start_end in h5q_starts2coord(cfg['in'], t_intervals_start, dt_interval=cfg['proc']['dt_interval']):
                a = h5_load_range_by_coord(cfg['in'], start_end)
                d, i_burst = filt_data_dd(a, cfg['in'])
                n_bursts = len(i_burst)
                if n_bursts > 1:  # 1st is always 0
                    l.info('gaps found: (%s)! at %s', n_bursts - 1, i_burst[1:] - 1)
                df0 = d.compute()
                if not len(df0):
                    continue
                start_end = df0.index[[0, -1]].values
                yield df0, start_end

    else:
        # variant 2. genereate intervals at specified start values with same width cfg['proc']['dt_interval']
        query_range_pattern = "index>=Timestamp('{}') & index<=Timestamp('{}')"

        def gen_loaded(tbl):
            for start_end in zip(cfg['in']['time_intervals_start'],
                                 cfg['in']['time_intervals_start'] + cfg['proc']['dt_interval']):
                query_range_lims = pd.to_datetime(start_end)
                qstr = query_range_pattern.format(*query_range_lims)
                l.info(f'query:\n%s... ', qstr)
                df0 = store.select(tbl, where=qstr, columns=None)
                yield df0, start_end

    dt_interval_in_its_units = cfg['proc']['dt_interval'].astype(int)
    dt_interval_units = np.datetime_data(cfg['proc']['dt_interval'])[0]
    data_name_suffix = f'{dt_interval_in_its_units}{dt_interval_units}'

    # Cycle
    with pd.HDFStore(cfg['in']['db_path'], mode='r') as store:
        for (tbl, coefs) in h5_names_gen(cfg['in'], cfg_out):
            # Get data in ranges
            for df0, start_end in gen_loaded(tbl):
                if cfg['in']['db_path'].stem.endswith('proc_noAvg'):
                    df = df0
                else:  # loading source data needed to be processed to calc velocity
                    df0 = filter_local(df0, cfg['filter'])
                    df = incl_calc_velocity_nodask(df0, **coefs, cfg_filter=cfg['in'], cfg_proc=cfg['proc'])

                data_name = f'{tbl}/PSD_{start_end[0]}{data_name_suffix}'
                yield (df, tbl, data_name)


def psd_mt_params(length, bandwidth, low_bias, adaptive, n_times=0, dt=None, fs=None, fmin=0, fmax=None, **kwargs):
    """
    Dpss calculation and default parameters.

    :param bandwidth, The bandwidth of the multi taper windowing function in Hz.
    :param low_bias, Only use tapers with more than 90% spectral concentration within bandwidth.
    :param length:
    :param dt: if provided assign to prm['dt'] and assign prm['fs'] = 1/prm['dt']
    :param fs: The sampling frequency. if provided assign to prm['fs'] and assign prm['dt'] = 1/prm['fs']
    :param fmin: float, The lower frequency of interest.
    :param fmax: float, The upper frequency of interest.

    :return: prm, dict with fields:
    - length
    - n_fft
    - dt
    - fs
    - dpss
    - eigvals
    - adaptive
    - weights
    - freqs
    - freq_mask
    """

    prm = {}
    prm['length'] = length
    if not kwargs.get('n_fft'):
        prm['n_fft'] = max(256, 2 ** np.ceil(np.log2(prm['length'])).astype('int'))
    # if kwargs.get('dpss') is None:
    # prm['n_fft'] = length

    if dt is not None:
        prm['dt'] = dt
        prm['fs'] = 1 / prm['dt']
    elif fs is not None:
        prm['fs'] = fs
        prm['dt'] = 1 / prm['fs']
    elif kwargs.get('dt') is not None:
        prm['fs'] = 1 / kwargs['dt']
    else:
        prm['dt'] = 1 / kwargs['fs']

    if fmax is not None:
        prm['fmax'] = fmax
    elif kwargs['in'].get('fmax') is None:
        prm['fmax'] = np.inf

    prm['freqs'] = np.fft.rfftfreq(prm['n_fft'], prm['dt'])  # only keep positive frequencies
    prm['freq_mask'] = (fmin <= prm['freqs']) & (prm['freqs'] <= prm['fmax'])
    prm['freqs'] = prm['freqs'][prm['freq_mask']]

    prm['dpss'], prm['eigvals'], prm['adaptive_if_can'] = multitaper._compute_mt_params(
        prm['length'], prm['fs'], bandwidth, low_bias, adaptive)  # normalization='length'
    prm['weights'] = np.sqrt(prm['eigvals'])[np.newaxis, :, np.newaxis]

    return prm

#@jit filed with even for x.any() and np.atleast_2d(x)
def psd_mt(x, dpss, weights, dt, n_fft, freq_mask, adaptive_if_can=None, eigvals=None):
    """
    Compute power spectral density (PSD) using a multi-taper method.
    :param x: array, shape=(..., n_times). The data to compute PSD from.
    :param dpss:
    :param weights:
    :param dt:
    :param n_fft:
    :param freq_mask: bool array
    :param adaptive_if_can: bool,
    :param eigvals:  - required if ``adaptive_if_can``
    :return psd: ndarray, shape (..., n_freqs). The PSDs. All dimensions up to the last will be the same as ``x`` input.

    See Also
    --------
    psd_mt_params - parameters for this calculation
    mne.psd_array_multitaper - base code. Here psd is separated from dpss calculation unlike in mne
    mne._mt_spectra(x, dpss, fs)[0]
    """

    if not x.any():  # (x!=0).sum() > N?   fast return for if no data
        return np.full((1, freq_mask.sum()), np.nan)
    x = np.atleast_2d(x)
    x = x.reshape(-1, x.shape[-1])

    n_tapers = dpss.shape[0] if dpss.ndim > 1 else 1
    x_mt = np.zeros(x.shape[:-1] + (n_tapers, freq_mask.sum()), dtype=np.complex128)

    # The following is equivalent to this, but uses less memory:
    # x_mt = fftpack.fft(x[:, np.newaxis, :] * dpss, n=n_fft)
    for idx, sig in enumerate(x - np.mean(x, axis=-1, keepdims=True)):  # remove mean
        x_mt[idx] = np.fft.rfft(sig[..., np.newaxis, :] * dpss, n=n_fft)[..., freq_mask]

    # Adjust DC and maybe Nyquist, depending on one-sided transform
    if freq_mask[0]:
        x_mt[:, :, 0] /= np.sqrt(2.)
    if freq_mask[-1] and x.shape[1] % 2 == 0:
        x_mt[:, :, -1] /= np.sqrt(2.)
    if not adaptive_if_can:
        psds = weights * x_mt
        psds *= psds.conj()  # same to abs(psd)**2
        psd = psds.real.sum(axis=-2)
        psd *= 2 / (weights * weights.conj()).real.sum(axis=-2)
    else:
        # # from mne.parallel import parallel_func
        # # from mne.time_frequency.multitaper import _psd_from_mt_adaptive
        # psds = list(
        #     _psd_from_mt_adaptive(x, eigvals, np.ones((sum(freq_mask),), dtype=np.bool))
        #      # x already masked so we put all ok mask
        #            for x in np.array_split(x_mt, 1)
        #            )
        # psd = np.concatenate(psds)
        psd = _psd_from_mt_adaptive(x_mt, eigvals, np.ones((freq_mask.sum(),), dtype=np.bool))

        # make output units V^2/Hz:  (like mne.mne.time_frequency.psd_array_multitaper option normalization = 'full')
    psd *= dt
    return psd


def psd_calc(df, fs, freqs, adaptive=None, b_plot=False, **kwargs):
    """
    Compute Power Spectral Densities (PSDs) of df.Ve/Vn using multitaper method
    :param df: dataframe with Ve and Vn
    :param b_plot:
    :param kwargs: psd_mt kwargs
    :return:
    """

    psdm_Ve = psd_mt(df.Ve.to_numpy(), **kwargs)[0, :]
    psdm_Vn = psd_mt(df.Vn.to_numpy(), **kwargs)[0, :]

    if False:
        ## high level mne functions recalcs windows each time
        from third_party.mne.time_frequency import psd_array_multitaper
        multitaper.warn = l.warning
        psdm_Ve, freq = psd_array_multitaper(df.Ve, sfreq=fs, adaptive=adaptive,
                                             normalization='length')  # fmin=0, fmax=0.5,
        psdm_Vn, freq = psd_array_multitaper(df.Vn, sfreq=fs, adaptive=adaptive, normalization='length')  #

    if b_plot:
        # plot all

        plt.figure(figsize=(5, 4))
        # multitaper
        plt.semilogx(freqs, psdm_Ve)
        plt.semilogx(freqs, psdm_Vn)
        # # Welch
        # plt.semilogx(freqs, psd_Ve)
        # plt.semilogx(freqs, psd_Vn)
        # # Spectrum module result
        # plt.semilogx(kwargs['freqs'], sk_Ve)
        # plt.semilogx(kwargs['freqs'], sk_Vn)

        plt.title('PSD: power spectral density')
        plt.xlabel('Frequency')
        plt.ylabel('Power')
        plt.tight_layout()
        plt.show()
        pass

    ds = xr.Dataset(
        {'PSD_Ve': (('time', 'freq'), psdm_Ve[np.newaxis]),
         'PSD_Vn': (('time', 'freq'), psdm_Vn[np.newaxis]),
         'time': df.index[:1].values,
         'freq': freqs})

    return ds


def main(new_arg=None, **kwargs):
    """
    Accumulats results of differen source tables in 2D netcdf matrices of each result parameter.
    :param new_arg:
    :return:
    Spectrum parameters used (taken from nitime/algorithems/spectral.py):
        NW : float, by default set to 4: that corresponds to bandwidth of 4 times the fundamental frequency
        The normalized half-bandwidth of the data tapers, indicating a
        multiple of the fundamental frequency of the DFT (Fs/N).
        Common choices are n/2, for n >= 4. This parameter is unitless
        and more MATLAB compatible. As an alternative, set the BW
        parameter in Hz. See Notes on bandwidth.

        BW : float
        The sampling-relative bandwidth of the data tapers, in Hz.

        adaptive : {True/False}
           Use an adaptive weighting routine to combine the PSD estimates of
           different tapers.
        low_bias : {True/False}
           Rather than use 2NW tapers, only use the tapers that have better than
           90% spectral concentration within the bandwidth (still using
           a maximum of 2NW tapers)
    Notes
    -----

    The bandwidth of the windowing function will determine the number
    tapers to use. This parameters represents trade-off between frequency
    resolution (lower main lobe BW for the taper) and variance reduction
    (higher BW and number of averaged estimates). Typically, the number of
    tapers is calculated as 2x the bandwidth-to-fundamental-frequency
    ratio, as these eigenfunctions have the best energy concentration.

    Result file is nc format that is Veusz compatible hdf5 format. If file exists it will be overwited

    todo: best may be is use DBMT: Dynamic Bayesian Multitaper (matlab code downloaded from git)
    """
    global l

    cfg = cfg_from_args(my_argparser(), new_arg)
    if not cfg or not cfg['program'].get('return'):
        print('Can not initialise')
        return cfg
    elif cfg['program']['return'] == '<cfg_from_args>':  # to help testing
        return cfg

    init_logging(logging, None, cfg['program']['log'], cfg['program']['verbose'])
    l = logging.getLogger(prog)

    multitaper.warn = l.warning  # module is not installed but copied. so it can not import this dependace

    try:
        cfg['in'] = init_file_names(cfg['in'], cfg['program']['b_interact'], path_field='db_path')
    except Ex_nothing_done as e:
        print(e.message)
        return ()
    print('\n' + prog, end=' started. ')

    cfg_out = cfg['out']
    if 'split_period' in cfg['out']:
        cfg['proc']['dt_interval'] = np.timedelta64(cfg['proc']['dt_interval'] if cfg['proc']['dt_interval'] else
                                                    pd_period_to_timedelta(cfg['out']['split_period']))
        if (not cfg['proc']['overlap']) and \
                (cfg['proc']['dt_interval'] == np.timedelta64(pd_period_to_timedelta(cfg['out']['split_period']))):
            cfg['proc']['overlap'] = 0.5
    else:
        cfg['proc']['dt_interval'] = np.timedelta64(cfg['proc']['dt_interval'])
        # cfg['proc']['dt_interval'] = np.timedelta64('5', 'm') * 24
        cfg['proc']['time_intervals_start'] = np.array(cfg['proc']['time_intervals_center'], np.datetime64) - \
                                              cfg['proc']['dt_interval'] / 2
    # minimum time between blocks, required in filt_data_dd().
    cfg['in']['burst_min'] = None  # If None report any interval bigger then min(1st, 2nd)

    cfg_out['chunksize'] = cfg['in']['chunksize']
    h5init(cfg['in'], cfg_out)
    # cfg_out_table = cfg_out['table']  need? save beacause will need to change
    cfg_out['save_proc_tables'] = True  # False

    # cfg['proc'] = {}
    prm = cfg['proc']
    prm['adaptive'] = True  # pmtm spectrum param

    prm['fs'] = cfg['in']['fs']
    prm['bandwidth'] = 8 / cfg['proc']['dt_interval'].astype('timedelta64[s]').astype(
        'float')  # 8 * 2 * prm['fs']/34000  # 4 * 2 * 5/34000 ~= 4 * 2 * fs / N
    prm['low_bias'] = True

    nc_root = netCDF4.Dataset(Path(cfg_out['db_path']).with_suffix('.nc'), 'w',
                              format='NETCDF4')  # (for some types may need 'NETCDF4_CLASSIC' to use CLASSIC format for Views compability)
    nc_psd = nc_root.createGroup(cfg_out['table'])
    nc_psd.createDimension('time', None)
    nc_psd.createDimension('value', 1)
    nc_psd.createVariable('time_good_min', 'f8', ('value',))
    nc_psd.createVariable('time_good_max', 'f8', ('value',))
    nc_psd.createVariable('time_interval', 'f4', ('value',))
    if cfg['out'].get('split_period'):
        # nv_time_interval = nc_psd.createVariable('time_interval', 'f8', ('time',), zlib=False)
        nc_psd.variables['time_interval'][:] = pd_period_to_timedelta(cfg['out']['split_period']).delta
    else:
        nc_psd.variables['time_interval'][:] = cfg['proc']['dt_interval']
    # Dataframe of accumulating results: adding result columns in cycle with appending source table name to column names
    dfs_all = None
    time_good_min = pd.Timestamp.max
    time_good_max = pd.Timestamp.min
    prm['length'] = None
    nv_vars_for_tbl = {}
    tbl_prev = ''
    itbl = 0
    for df, tbl_in, dataname in h5_velocity_by_intervals_gen(cfg, cfg_out):
        tbl = tbl_in.replace('incl', '_i')
        # _, (df, tbl, dataname) in h5_dispenser_and_names_gen(cfg['in'], cfg_out, fun_gen=h5_velocity_by_intervals_gen):
        len_data_cur = df.shape[0]
        if tbl_prev != tbl:
            itbl += 1
            l.info('%s: len=%s', dataname, len_data_cur)
        l.info('    %s. Writing to "%s"', itbl, tbl)

        # Prepare
        if prm['length'] is None:
            # 1st time
            prm['length'] = len_data_cur
            prm.update(psd_mt_params(**prm, dt=float(np.median(np.diff(df.index.values))) / 1e9))
            nc_psd.createDimension('freq', len(prm['freqs']))
            # nv_... - variables to be used as ``NetCDF variables``
            nv_freq = nc_psd.createVariable('freq', 'f4', ('freq',), zlib=True)
            nv_freq[:] = prm['freqs']
            check_fs = 1e9/np.median(np.diff(df.index.values)).item()
            if prm.get('fs'):
                np.testing.assert_almost_equal(prm['fs'], check_fs, decimal=7, err_msg='', verbose=True)
            else:
                prm['fs'] = check_fs
        elif prm['length'] != len_data_cur:
            prm['length'] = len_data_cur
            try:
                prm['dpss'], prm['eigvals'], prm['adaptive_if_can'] = \
                    multitaper._compute_mt_params(prm['length'], prm['fs'], prm['bandwidth'],
                                                  prm['low_bias'], prm['adaptive'])
            except (ModuleNotFoundError, ValueError) as e:
                # l.error() already reported as multitaper.warn is reassignred to l.warning()
                prm['eigvals'] = np.int32([0])
            prm['weights'] = np.sqrt(prm['eigvals'])[np.newaxis, :, np.newaxis]
            # l.warning('new length (%s) is different to last (%s)', len_data_cur, prm['length'])

        if not tbl in nc_psd.groups:
            nc_tbl = nc_psd.createGroup(tbl)
            cols = set()
            if 'Pressure' in df.columns:
                cols.add('Pressure')
                nc_tbl.createVariable('Pressure', 'f4', ('time', 'freq',), zlib=True)
            if 'Ve' in df.columns:
                cols.update(['Ve', 'Vn'])
                nc_tbl.createVariable('Ve', 'f4', ('time', 'freq',), zlib=True)
                nc_tbl.createVariable('Vn', 'f4', ('time', 'freq',), zlib=True)
            nc_tbl.createVariable('time_start', 'f8', ('time',), zlib=True)
            nc_tbl.createVariable('time_end', 'f8', ('time',), zlib=True)
            out_row = 0
        nc_tbl.variables['time_start'][out_row], nc_tbl.variables['time_end'][out_row] = df.index[[0, -1]].values

        # Calculate
        if prm['eigvals'].any():
            for var_name in cols:
                nc_tbl.variables[var_name][out_row, :] = call_with_valid_kwargs(psd_mt, df[var_name].to_numpy(), **prm)[0, :]
            if np.datetime64(time_good_min, 'ns') > df.index[
                0].to_numpy():  # use values to not deal with tz-naive/aware timestamps
                time_good_min = df.index[0]
            if np.datetime64(time_good_max.value, 'ns') < df.index[-1].to_numpy():
                time_good_max = df.index[-1]
        else:
            for var_name in cols:
                nc_tbl.variables[var_name][out_row, :] = np.NaN

        out_row += 1

        # if cfg_out['save_proc_tables']:
        #     # ds_psd.to_netcdf('d:\\WorkData\\BlackSea\\190210\\190210incl_proc-psd_test.nc', format='NETCDF4_CLASSIC')
        #     #f.to_hdf('d:\\WorkData\\BlackSea\\190210\\190210incl_proc-psd_test.h5', 'psd', format='fixed')
        #     # tables_have_write.append(tbl)
        #     try:
        #         h5_append_to(df_psd, tbl, cfg_out, msg='save (temporary)', print_ok=None)
        #     except HDF5ExtError:
        #         cfg_out['save_proc_tables'] = False
        #         l.warning('too very many colums for "table" format but "fixed" is not updateble so store result in memory 1st')
        #
        #
        #
        # df_cur = df_psd[['PSD_Vn', 'PSD_Ve']].rename(
        #     columns={'PSD_Ve': 'PSD_Ve' + tbl[-2:], 'PSD_Vn': 'PSD_Vn' + tbl[-2:]}).compute()
        # if dfs_all is None:
        #     dfs_all = df_cur
        # else:
        #     dfs_all = dfs_all.join(df_cur, how='outer')  # , rsuffix=tbl[-2:] join not works on dask

        # if itbl == len(cfg['in']['tables']):  # after last cycle. Need incide because of actions when exit generator
        #     h5_append_to(dfs_all, cfg_out_table, cfg_out, msg='save accumulated data', print_ok='Ok.')

    # nv_time_start_query = nc_psd.createVariable('time_start_query', 'f8', ('time',), zlib=True)
    # nv_time_start_query[:] = cfg['in']['time_intervals_start'].to_numpy(dtype="datetime64[ns]") \
    #     if isinstance(cfg['in']['time_intervals_start'], pd.DatetimeIndex) else cfg['in']['time_intervals_start']

    nc_psd.variables['time_good_min'][:] = np.array(time_good_min.value, 'M8[ns]')
    nc_psd.variables['time_good_max'][:] = np.array(time_good_max.value, 'M8[ns]')
    # failed_storages = h5move_tables(cfg_out)
    print('Ok.', end=' ')
    nc_root.close()


if __name__ == '__main__':
    main()


# tried but not used code ##############################################################################################

def psd_calc_other_methods(df, prm: Mapping[str, Any]):
    ## scipy
    windows = signal.windows.dpss(180000, 2.5, Kmax=9, norm=2)
    signal.spectrogram

    ## Welch
    nperseg = 1024 * 8
    freqs, psd_Ve = signal.welch(df.Ve, prm['fs'], nperseg=nperseg)
    freqs, psd_Vn = signal.welch(df.Vn, prm['fs'], nperseg=nperseg)

    ## use Spectrum module
    from spectrum import dpss

    def pmtm(x, eigenvalues, tapers, n_fft=None, method='adapt'):
        """Multitapering spectral estimation

        :param array x: the data
        :param eigenvalues: the window concentrations (eigenvalues)
        :param tapers: the matrix containing the tapering windows
        :param str method: set how the eigenvalues are used. Must be
            in ['unity', 'adapt', 'eigen']
        :return: Sk (each complex), weights, eigenvalues

        Usually in spectral estimation the mean to reduce bias is to use tapering
        window. In order to reduce variance we need to average different spectrum.
        The problem is that we have only one set of data. Thus we need to
        decompose a set into several segments. Such method are well-known: simple
        daniell's periodogram, Welch's method and so on. The drawback of such
        methods is a loss of resolution since the segments used to compute the
        spectrum are smaller than the data set.
        The interest of multitapering method is to keep a good resolution while
        reducing bias and variance.

        How does it work? First we compute different simple periodogram with the
        whole data set (to keep good resolution) but each periodgram is computed
        with a differenttapering windows. Then, we average all these spectrum.
        To avoid redundancy and bias due to the tapers mtm use special tapers.

        from spectrum import data_cosine, dpss, pmtm
        data = data_cosine(N=2048, A=0.1, sampling=1024, freq=200)
        [tapers, eigen] = dpss(2048, 2.5, 4)
        res = pmtm(data, eigenvalues=eigen, tapers=tapers, show=False)

        .. versionchanged:: 0.6.2a
        The most of spectrum.pmtm original code is to calc PSD but it is not returns so here we return it
        + Removed redandand functionality (calling semilogy plot and that what included in spectrum.dpss)
        """
        assert method in ['adapt', 'eigen', 'unity']

        N = len(x)
        if eigenvalues is not None and tapers is not None:
            eig = eigenvalues[:]
            tapers = tapers[:]
        else:
            raise ValueError("if eigenvalues provided, v must be provided as well and viceversa.")
        nwin = len(eig)  # length of the eigen values vector to be used later

        if n_fft is None:
            n_fft = max(256, 2 ** np.ceil(np.log2(N)).astype('int'))

        Sk_complex = np.fft.fft(tapers.transpose() * x, n_fft)

        # if nfft < N, cut otherwise add zero.
        Sk = (Sk_complex * Sk_complex.conj()).real  # abs() ** 2
        if method in ['eigen', 'unity']:
            if method == 'unity':
                weights = np.ones((nwin, 1))
            elif method == 'eigen':
                # The S_k spectrum can be weighted by the eigenvalues, as in Park et al.
                weights = np.array([_x / float(i + 1) for i, _x in enumerate(eig)])
                weights = weights.reshape(nwin, 1)
                Sk = np.mean(Sk * weights, axis=0)
        elif method == 'adapt':
            # This version uses the equations from [2] (P&W pp 368-370).
            Sk = Sk.transpose()
            S = Sk[:, :2].mean()  # Initial spectrum estimate

            # Set tolerance for acceptance of spectral estimate:
            sig2 = np.dot(x, x) / float(N)
            tol = 0.0005 * sig2 / float(n_fft)
            a = sig2 * (1 - eig)

            # Wrap the data modulo nfft if N > nfft
            S = S.reshape(n_fft, 1)
            for i in range(100):  # converges very quickly but for safety; set i<100
                # calculate weights
                b1 = np.multiply(S, np.ones((1, nwin)))
                b2 = np.multiply(S, eig.transpose()) + np.ones((n_fft, 1)) * a.transpose()
                b = b1 / b2

                # calculate new spectral estimate
                weights = (b ** 2) * (np.ones((n_fft, 1)) * eig.transpose())
                S1 = ((weights * Sk).sum(axis=1, keepdims=True) / weights.sum(axis=1, keepdims=True))
                S, S1 = S1, S
                if np.abs(S - S1).sum() / n_fft < tol:
                    break
            Sk = (weights * Sk).mean(axis=1)

        if np.isrealobj(x):  # Double to account for the energy in the negative frequencies
            if prm['n_fft'] % 2 == 0:
                Sk = 2 * Sk[:int(prm['n_fft'] / 2 + 1)]
            else:
                Sk = 2 * Sk[:int((prm['n_fft'] + 1) / 2)]

        return Sk_complex, Sk, weights

    prm['dpss_sp'], prm['eigvals_sp'] = dpss(prm['n_fft'], 3.5)
    sk_complex_Ve, sk_Ve_, weights_Ve = pmtm(df.Ve.values, prm['eigvals_sp'], prm['dpss_sp'])  # n_fft=prm['n_fft']
    sk_complex_Vn, sk_Vn_, weights_Vn = pmtm(df.Vn.values, prm['eigvals_sp'], prm['dpss_sp'])
    # Convert Power Spectrum to Power Spectral Density
    record_time_length = prm['length'] * prm['dt']
    sk_Ve = sk_Ve_ / record_time_length
    sk_Vn = sk_Vn_ / record_time_length


"""old cfg


    cfg = {  # output configuration after loading csv:
        'in': {
            'db_path': r'd:\WorkData\BalticSea\181116inclinometer_Schuka\181116incl.h5',
                # r'd:\WorkData\BlackSea\190210\inclinometer_ABSIORAS\190210incl.h5',
            #
            'tables': ['incl.*'],
            'split_period': '2H',  # pandas offset string (D, 5D, H, ...)
            'overlap': 0.5,
            'date_min': datetime.strptime('2018-11-16T15:30:00', '%Y-%m-%dT%H:%M:%S'),
            'date_max': datetime.strptime('2018-12-14T14:35:00', '%Y-%m-%dT%H:%M:%S'),
            # 'date_min': datetime.strptime('2019-02-11T14:00:00', '%Y-%m-%dT%H:%M:%S'),
            # 'date_max': datetime.strptime('2019-02-28T14:00:00', '%Y-%m-%dT%H:%M:%S'),              #.replace(tzinfo=timezone.utc), gets dask error
            'chunksize': 1000000,  # 'chunksize_percent': 10,  # we'll repace this with burst size if it suit
            # 'max_g_minus_1' used only to replace bad with NaN
        },
        'out': {
            'db_path': 'incl_proc.h5',
            'table': 'V_incl',

            #'aggregate_period': '2H',  # pandas offset string (Y, D, 5D, H, ...)
        },
        'program': {
            'log': str(scripts_path / 'log/incl_h5clc.log'),
            'verbose': 'DEBUG'
        }
    }
    
    # not used if cfg['out']['split_period'] is specified:
    cfg['proc']['time_intervals_center'] = pd.to_datetime(np.sort(np.array(
        ['2019-02-16T08:00', '2019-02-17T04:00', '2019-02-18T00:00', '2019-02-28T00:00',
        '2019-02-14T12:00', '2019-02-15T12:00', '2019-02-16T23:50',
        '2019-02-20T00:00', '2019-02-20T22:00', '2019-02-22T06:00', '2019-02-23T03:00',
        '2019-02-13T11:00', '2019-02-14T13:00', '2019-02-16T23:00', '2019-02-18T12:00',
        '2019-02-19T00:00', '2019-02-19T16:00', '2019-02-21T00:00', '2019-02-22T02:00', '2019-02-23T00:00',
        '2019-02-26T06:00', '2019-02-26T16:00', '2019-02-28T06:00'
        ], dtype='datetime64[s]'))
        )
"""
