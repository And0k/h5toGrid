#!/usr/bin/env python3
# coding:utf-8
"""
Author:  Andrey Korzh <ao.korzh@gmail.com>
Purpose: Calculate spectrum at specified time intervals
Updated: 17.03.2026
"""

import logging
import re
from datetime import timedelta
from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path, PurePath
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple, Union

import netCDF4
import numpy as np
import pandas as pd
import xarray as xr

from matplotlib import pyplot as plt
from scipy import interpolate
from numba import jit

from utils.init import Ex_nothing_done, init_logging, cfg_from_args, init_file_names, my_argparser_common_part, call_with_valid_kwargs
from hdf5_pandas import h5
from hdf5_pandas.h5_dask_pandas import filter_local, filter_global_minmax, i_bursts_starts
from tcm.incl_h5clc import incl_calc_velocity_nodask, my_argparser, h5_names_gen
from veusz_helpers.common import func_vsz as fv

from mne.time_frequency import multitaper  # third_party


version = '0.1.1'
if __name__ == '__main__':
    l = None  # see main()
    prog = 'spectr_clc'  # this_prog_basename(__file__)
else:
    l = logging.getLogger(__name__)
    prog = __name__


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
    # All arguments of type str (default for add_argument...), because of
    # custom postprocessing based of my_argparser names in ini2dict

    s = p.add_argument_group("in", "Parameters of input files")
    s.add('--db_path', default='*.h5',  # nargs=?,
                             help='path to pytables hdf5 store to load data. May use patterns in Unix shell style')
    s.add('--tables_list',   help='table names in hdf5 store to get data. Uses regexp')
    s.add('--chunksize_int', help='limit loading data in memory', default='50000')
    s.add('--min_date',      help='time range min to use', default='2019-01-01T00:00:00')
    s.add('--max_date',      help='time range max to use')
    s.add('--fs_float',      help='sampling frequency of input data, Hz')
    s.add("--dt_hole_warning", help="warning when bigger time hole will be found in data")

    s = p.add_argument_group("filter", "Filter all data based on min/max of parameters")
    s.add('--min_dict',
        help='List with items in  "key:value" format. Filter out (not load), data of ``key`` columns if it is below ``value``')
    s.add('--max_dict',
        help='List with items in  "key:value" format. Filter out data of ``key`` columns if it is above ``value``')
    s.add('--min_Pressure',      help='min value of Pressure range use. Note: to filer {parameter} NaNs - use some value in min_{parameter} or max_{parameter}. Example of filtering out only nans of Pressure using very big min_dict value: "--min_Pressure -1e15"', default='-1e15')
    s.add('--max_Pressure',      help='max value of Pressure range to use')

    s = p.add_argument_group("out", "Parameters of output files")
    s.add('--out.db_path', help='hdf5 store file path')
    s.add('--table', default='psd',
        help='table name in hdf5 store to write data. If not specified then will be generated on base of path of input files. Note: "*" is used to write blocks in autonumbered locations (see dask to_hdf())')

    s = p.add_argument_group("proc", "Processing parameters")
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
        help=r'Overwrites last coefficient of trigonometric version of g: Vabs = g(Inclingation). It corresponds to point where g(x) = Vabs(inclination) became bend down. To prevent this g after this point is replaced with line, so after max_incl_of_fit_deg {\Delta}^{2}y ≥ 0 for x > max_incl_of_fit_deg')

    s = p.add_argument_group("program", "Program behavior")
    s.add('--return', default='<end>', choices=['<return_cfg>', '<return_cfg_with_options>'],
        help='executes part of code and returns parameters after skipping of some code')

    return (p)


def df_interp(df: pd.DataFrame, fs=1, cols=[], method=None) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Interpolate and extrapolate dataframe columns to constant frequency index using numpy / scipy
    This is should be the same as following, but not give all identical values (!) as pandas sometimes give:
    df = df.resample(timedelta(seconds=1 / prm['fs'])).interpolate()
    :param df:
    :param fs: frequency to get output regular grid time delta
    :param cols: output columns, return all if empty
    :param method: if 'pchip' then use scipy.interpolate.PchipInterpolator
    :return: (df_out, bads):
    - dataframe with fixed frequency index,
    - bool array of bad values of source dataframe
    """

    # linear timedelta index with the desired frequency
    index = pd.date_range(*df.index[[0, -1]].to_list(), freq=timedelta(seconds=1 / fs))
    bads = {}
    values = {}
    for col in cols if any(cols) else df.columns:
        ser_col_ok = df[col]
        bads[col] = df[col].isna().values
        ser_col_ok = ser_col_ok[~bads[col]]
        try:
            if method == 'pchip':
                interp_obj = interpolate.PchipInterpolator(
                    ser_col_ok.index, ser_col_ok.values, extrapolate=False
                )  # we not use extrapolation here as it can end at very big values
                values[col] = interp_obj(index)
                # Handle extrapolation: constant value
                values[col][index < ser_col_ok.index[0]] = ser_col_ok.values[0]
                values[col][index > ser_col_ok.index[-1]] = ser_col_ok.values[-1]
            else:
                values[col] = np.interp(
                    index, ser_col_ok.index, ser_col_ok.values
                )
        except ValueError:  # array of sample points is empty
            values[col] = np.nan
    return pd.DataFrame.from_records(values, index=index), bads


#@jit failed for n_signals, n_tapers, n_freqs = x_mt.shape and not defined weights
def _psd_from_mt_adaptive(
        x_mt: np.ndarray, eigvals, freq_mask, max_iter=150,
        return_weights=False
    ):
    r"""
    Use iterative procedure to compute the PSD from tapered spectra.
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

        b_zero = psd_iter == 0
        if any(b_zero):
            if all(b_zero):
                l.warning('No data for PSD computation')
                psd[i, :] = np.nan
                if return_weights:
                    weights[i, :, :] = np.nan
                continue
            # todo: check problems if any b_zero
            pass

        err = np.zeros_like(xk)
        for n in range(max_iter):
            d_k = psd_iter / (
                eigvals[:, np.newaxis] * psd_iter + (1 - eigvals[:, np.newaxis]) * var
            )
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


def gen_interval_starts(
        min_date: pd.Timestamp, max_date: pd.Timestamp,
        dt_interval: np.timedelta64, overlap: float,
        msg_log="Loading {total_intervals} ranges"
    ):
    """Generator that yields interval start times

    Generates base intervals from min_date to max_date with dt_interval period,
    then applies overlap shifts to create additional overlapping intervals.
    Yields intervals in chronological order.

    Args:
        min_date: Starting point for interval generation
        max_date: End point for interval generation
        dt_interval: Base time interval between non-overlapping intervals
        overlap: Overlap ratio [0, 1) for generating shifted intervals

    Yields:
        datetime64[ns]: Interval start times
    """
    # Calculate overlap shifts for interval generation
    dt_interval_seconds = dt_interval.astype('m8[s]').astype(float)
    # For overlap=0.5, this creates start times at 0, 0.5*dt_interval, 1.0*dt_interval, etc.
    dt_shifts_seconds = np.arange(0, 1, (1 - overlap)) * dt_interval_seconds
    dt_shifts = dt_shifts_seconds.astype('m8[s]')

    min_date = pd.Timestamp(min_date).tz_localize(None)
    max_date = pd.Timestamp(max_date).tz_localize(None)

    # Calculate total intervals for logging before first yield
    # Calculate number of base intervals
    total_duration = max_date - min_date
    base_intervals_count = int(np.ceil(total_duration / pd.Timedelta(dt_interval)))
    total_intervals = base_intervals_count * len(dt_shifts)
    l.info(msg_log.format(total_intervals=total_intervals))

    for current_start in pd.date_range(
        start=min_date,
        end=max_date,
        freq=pd.Timedelta(dt_interval),
        inclusive='left'
    ):
        for shift in dt_shifts:
            # Calculate and yield interval start time with shift applied
            interval_start = current_start + shift
            # Only yield if within max_date constraint
            if interval_start < max_date:
                yield interval_start


def gen_intervals(starts_time: Union[np.ndarray, pd.Series, Iterator], dt_interval: Any) -> Iterator[Tuple[Any, Any]]:
    """Generate (start, end) pairs from start times.

    When starts_time is a generator/iterator, yields (start, start + dt_interval) pairs.
    When starts_time is an array/list, yields (start, start + dt_interval) pairs.

    Args:
        starts_time: Array, list, or iterator of start times
        dt_interval: Time interval to add to each start time

    Yields:
        Tuple of (start, end) for each interval
    """
    # Check if starts_time is a generator/iterator (not supporting + operator)
    if hasattr(starts_time, '__iter__') and not hasattr(starts_time, '__add__'):
        # It's a generator/iterator - yield pairs one at a time
        for start in starts_time:
            yield (start, start + dt_interval)
    else:
        # It's an array/list - use vectorized operation
        for t_start_end in zip(starts_time, starts_time + dt_interval):
            yield t_start_end


def h5q_starts2coord(
        file_or_handle,
        table,
        gen_intervals: Any = gen_intervals,
        **kwargs
        ) -> Iterator[Tuple[pd.Index, int]]:
    """
    Edge coordinates of index range query with tracking of intervals with no data.
    Supports both arrays/lists and generators for lazy evaluation.

    When starts_time is a generator, gen_intervals(starts_time, dt_interval) is called
    to create (start, end) pairs from the generator's yielded start times.

    When starts_time is an array/list, gen_intervals(starts_time, dt_interval) is called
    to create (start, end) pairs by adding dt_interval to each start time.

    :param file_or_handle: HDF5 file path or handle
    :param table: Table name in HDF5 store
    :param gen_intervals: Generator function that creates (start, end) pairs from start times
    :param kwargs: for default gen_intervals it must be:
    - starts_time: array, list, or iterator with strings convertable to pandas.Timestamp
    - dt_interval: pd.TimeDelta (required when starts_time is array/list)
    :return: Iterator yielding tuples of (ind_st_en, intervals_with_no_data_count)
    where ind_st_en are edge coordinates, and intervals_with_no_data_count
    is the cumulative count of intervals with no data in HDF5
    """
    ind_st_last = 0
    intervals_with_no_data_count = 0

    with (
        pd.HDFStore(file_or_handle, mode="r") if isinstance(file_or_handle, (str, PurePath)) else
        nullcontext(file_or_handle)
    ) as store:
        qstr = "index>=st & index<=en"
        for i, (st, en) in enumerate(gen_intervals(**kwargs)):
            ind_all = store.select_as_coordinates(table, qstr, start=ind_st_last)
            if (nrows := len(ind_all)):
                l.debug('%d. [%s, %s] - %drows', i + 1, st, en, nrows)
                ind_st_en = ind_all[[0, -1]]  # .values
                # Record coordinate which before next query
                # (It is an additional condition that may help to faster search on next query,
                # we use 1st and not use last because intervals may overlap)
                ind_st_last = ind_st_en[0]
                yield (ind_st_en, intervals_with_no_data_count)
            elif ind_st_last:  # no data after some data
                # l.debug('%d. [%s, %s] - no data', i + 1, st, en)
                intervals_with_no_data_count += 1
                try:  # Check that will no more data
                    nd_all = store.select_as_coordinates(table, "index>=st", start=ind_st_last)
                    if nd_all.empty:
                        break
                except MemoryError:
                    ind_st_last = 0  # only to not check more
                    print('many data ahead...')  # it was just check for speed up


def align_to_grid(
        grid_origin: pd.Timestamp,
        min_date_candidate: pd.Timestamp,
        dt_interval: np.timedelta64,
        overlap: Optional[float] = None
    ) -> pd.Timestamp:
    """
    Align min_date to the grid defined by dt_interval starting from grid_origin.

    When overlap is specified, uses a finer grid step to ensure that both
    base intervals and overlap-shifted intervals are aligned to the grid.

    Args:
        grid_origin: The origin point of the time grid (e.g., configured min_date or start of day)
        min_date_candidate: The candidate min_date to search result after
        dt_interval: The base time interval step for the grid
        overlap: Overlap ratio [0, 1). If specified, uses a finer grid step of
                 (1 - overlap) * dt_interval to ensure overlap-shifted intervals
                 are also on the grid.

    Returns:
        The first grid point that is >= min_date_candidate
    """
    # Convert inputs to pd.Timestamp to ensure consistent arithmetic behavior
    grid_origin = pd.Timestamp(grid_origin)
    min_date_candidate = pd.Timestamp(min_date_candidate)

    # Calculate effective grid step: use finer grid when overlap is specified
    if overlap is not None and overlap > 0:
        # For overlap=0.5, effective step is 0.5 * dt_interval
        # This ensures both base intervals and shifted intervals are on grid
        # Convert to seconds, multiply by ratio, then convert back to timedelta64
        dt_interval_seconds = dt_interval.astype('m8[s]').astype(float)
        effective_step_ratio = 1 - overlap
        dt_interval_effective = np.timedelta64(int(dt_interval_seconds * effective_step_ratio), 's')
    else:
        dt_interval_effective = dt_interval

    dt_interval_timedelta = pd.Timedelta(dt_interval_effective)
    # Calculate number of intervals from grid origin to min_date_candidate
    intervals_diff = (min_date_candidate - grid_origin) / dt_interval_timedelta
    # Round up to next integer interval
    intervals_needed = int(np.ceil(intervals_diff))
    # Calculate the actual min_date on the grid
    return grid_origin + intervals_needed * dt_interval_timedelta


def h5_velocity_by_intervals_gen(
        cfg: Mapping[str, Any], cfg_out: Mapping[str, Any], tables_time_range_prev=None
    ) -> Iterator[Tuple[str, Tuple[Any, ...]]]:
    """
    - Load data: many intervals from many of hdf5 tables sequentially.
    - Filter and calculate velocity for inclinometers raw data (if "incl" in cfg["in"]["db_path"].stem or
    its suffix[-1] not starts with ".proc")
    :param cfg: dict with fields:
    - ['proc']['dt_interval'] - numpy.timedelta64 time interval of loading data
    - ['proc']['overlap'] - overlap ratio [0, 1) for generating additional intervals.
        If overlap is not None, generates regular intervals with overlap using dt_interval as the base period.
        If overlap is None uses manually specified intervals.
    - ['in']['time_intervals_start'] - manually specified intervals starts (if not using regular intervals)
    - ['in']['min_date'] or tables_time_range_prev[-1], datetime - required to not guess (to not generate empty intervals internally). Provide together with ['in']['max_date'] for faster execution and clear logging message.
    - ['in']['max_date'], datetime - required to not guess (to not generate empty intervals internally till `now`)
    :param cfg_out: dict with fields:
        - see h5.names_gen(cfg_in, cfg_out) requirements
    :param tables_time_range_prev: dict with table names as keys and tuples (time_start, time_end) as values
        representing existing time ranges to exclude from processing
    :return:
        Yields tuples of (df, tbl, data_name) for each interval with data,
        and (None, tbl, intervals_with_no_data) sentinel at end of each table processing
    """

    # Prepare cycle
    try:
        dt_interval = cfg["proc"]["dt_interval"]

        # Convert datetime.timedelta to np.timedelta64 if needed
        if not isinstance(dt_interval, np.timedelta64):
            dt_interval = np.timedelta64(dt_interval)
        if dt_interval <= np.timedelta64(0, 'ns'):
            raise ValueError(
                f"dt_interval must be positive, got: {dt_interval}. "
                f"This will result in no data being processed."
            )

        # dt_interval_in_its_units = dt_interval.astype(int)
        # dt_interval_units = np.datetime_data(dt_interval)[0]
        # data_name_suffix = f'dt={dt_interval_in_its_units}{dt_interval_units}'
        data_name_suffix = fv.str_dt(dt_interval.astype("m8[s]").astype(int), lang=None)
    except KeyError:  # no cfg["proc"]["dt_interval"]
        dt_interval = None
        data_name_suffix = "all"

    intervals_generated = 0

    # Use overlap parameter as trigger: if overlap > 0, generate regular intervals with overlap
    # Otherwise, use manually specified time_intervals_start
    if cfg['proc'].get('overlap') is not None:

        # Variant 1: Generate regular intervals (may be with overlap)
        # Track intervals with no data for statistics (nonlocal to be accessible in outer scope)

        def gen_loaded(tbl):
            """
            Variant 1. Generate regular intervals (may be with overlap)
            :param tbl:
            :return:
            """
            nonlocal intervals_generated

            # Exclude existing/not needed beginning part of data for current table
            tbl_normalized = tbl.replace('incl', '_i')
            t_range_del = None
            if tables_time_range_prev and tbl_normalized in tables_time_range_prev:
                t_range_del = tables_time_range_prev[tbl_normalized]
                l.debug(f"Excluding {tbl} data in range from {t_range_del[0]} to {t_range_del[1]}")

            # Generate interval start times using dt_interval as base period
            # Determine grid origin: use cfg['in']['min_date'] if defined, otherwise start of day
            grid_origin = cfg['in'].get('min_date')
            if grid_origin is None and t_range_del is not None:
                # If no min_date configured, use start of day of t_range_del[1] as grid origin
                grid_origin = t_range_del[1].floor('D')

            # Calculate min_date: find first grid point after max(grid_origin, t_range_del[1])
            # Use finer grid when overlap is specified to ensure overlap-shifted intervals are aligned
            if t_range_del is not None:
                min_date_candidate = max(grid_origin, t_range_del[1]) if grid_origin is not None else t_range_del[1]
                # Align to grid by finding the first grid point >= min_date_candidate
                # Pass overlap parameter to use finer grid step if overlap is specified
                if grid_origin is not None:
                    min_date = align_to_grid(
                        grid_origin, min_date_candidate, dt_interval, cfg['proc'].get('overlap')
                    )
                else:
                    # No grid origin, just ceil to dt_interval
                    min_date = min_date_candidate.ceil(pd.Timedelta(dt_interval))
            else:
                min_date = grid_origin

            with pd.HDFStore(cfg["in"]["db_path"], mode="r") as store:

                # Correct configured max_date with table-based last_date as their minimum

                # Get last index value from table
                table_storer = store.get_storer(tbl)
                max_date = cfg['in'].get('max_date')
                if hasattr(table_storer, "table") and table_storer.table.nrows > 0:
                    last_date = store.select(
                        tbl,
                        columns=[],
                        start=table_storer.table.nrows - 1,
                        stop=table_storer.table.nrows,
                    ).index.item().tz_localize(None)
                    if max_date:
                        if last_date:
                            max_date = min(last_date, max_date)
                    else:
                        max_date = last_date

                # Regenerate intervals for actual processing
                t_intervals_start_gen = gen_interval_starts(
                    min_date=min_date,
                    max_date=max_date,
                    dt_interval=dt_interval,
                    overlap=cfg['proc'].get('overlap'),
                    msg_log="loading {total_intervals} ranges from "
                    f"{cfg['in']['db_path'].name}/{tbl}: "
                )
                cfg_filter = None
                qstr = "index>=st & index<=en"
                for intervals_generated, (st, en) in enumerate(
                    gen_intervals(t_intervals_start_gen, dt_interval)
                ):
                    df0 = store.select(
                        tbl, columns=cfg["in"]["columns"],
                        where=qstr,

                    )
                    if not len(df0):
                        continue
                    if cfg_filter is None:  # only 1 time
                        detect_filt = f"m(ax|in)_({'|'.join(df0.columns)})"
                        cfg_filter = {k: v for k, v in cfg['filter'].items() if re.match(detect_filt, k)}
                    df0 = filter_global_minmax(df0, cfg_filter=cfg_filter)
                    i_burst, mean_burst_size, max_hole = i_bursts_starts(
                        df0.index,
                        dt_detect_bursts=cfg["in"].get("dt_between_bursts"),  # if None will autodetect
                    )
                    n_bursts = len(i_burst)
                    if n_bursts > 1 and max_hole:  # 1st is always 0
                        dt_max_hole = max_hole.astype("m8[s]").item()
                        if cfg["in"]["dt_hole_warning"] and dt_max_hole > cfg["in"]["dt_hole_warning"]:
                            l.warning(
                                "max time hole: %s",
                                fv.str_dt(dt_max_hole.total_seconds(), lang=None),
                            )
                            i_burst = i_burst[1:] - 1  # actual bursts indices
                            l.info(
                                "gaps (%d) found at %s: %s!",
                                n_bursts - 1,
                                i_burst,
                                ", ".join(
                                    fv.str_dt(dt.total_seconds(), lang=None)
                                    for dt in df0.index[i_burst + 1] - df0.index[i_burst]
                                ),
                            )
                    if not len(df0):
                        continue
                    start_end = df0.index[[0, -1]].values
                    yield df0, start_end
    else:
        # Variant 2: Generate intervals at specified start values cfg['in']['time_intervals_start']
        store = None
        query_range_pattern = "index>='{}' & index<='{}'"
        if dt_interval:

            # Track intervals with no data for statistics (nonlocal to be accessible in outer scope)

            def gen_loaded(tbl):
                """
                Variant 2. Generate intervals at specified start values cfg['in']['time_intervals_start']
                with same width dt_interval
                :param tbl:
                :return:
                """
                nonlocal intervals_generated
                # Filter out intervals that fall within excluding time range for this table
                tbl_normalized = tbl.replace('incl', '_i')
                t_range_del = None
                if tables_time_range_prev and tbl_normalized in tables_time_range_prev:
                    t_range_del = tables_time_range_prev[tbl_normalized]
                    l.debug(f"Excluding {tbl} data in range from {t_range_del[0]} to {t_range_del[1]}")

                # Generate interval start and end pairs once
                t_intervals_start = cfg["in"]["time_intervals_start"]
                t_intervals_end = t_intervals_start + dt_interval

                # Filter intervals that fall within excluding time range
                intervals_to_process = []
                for start, end in zip(t_intervals_start, t_intervals_end):
                    interval_start = pd.Timestamp(start)
                    interval_end = pd.Timestamp(end)

                    # Skip if entire interval is within excluding time range
                    if t_range_del is not None and (
                        interval_start >= t_range_del[0] and interval_end <= t_range_del[1]
                    ):
                        l.info(
                            f"Skipping interval {interval_start} to {interval_end} "
                            f"as it's within excluding time range {t_range_del[0]} to {t_range_del[1]}"
                        )
                        continue
                    intervals_to_process.append((start, end))

                # Process filtered intervals
                for intervals_generated, start_end in enumerate(intervals_to_process):
                    query_range_lims = pd.to_datetime(start_end)
                    qstr = query_range_pattern.format(*query_range_lims)
                    l.info('query:\n%s... ', qstr)
                    df0 = store.select(tbl, where=qstr, columns=None)
                    if not len(df0):
                        continue
                    yield df0, np.array(start_end)
        else:
            # Track intervals with no data for statistics (nonlocal to be accessible in outer scope)

            def gen_loaded(tbl):
                """
                Variant 3. load all at once
                """
                nonlocal intervals_generated
                intervals_generated = 1
                df0 = store.select(tbl)
                yield df0, ['', '']

    # Cycle
    with pd.HDFStore(cfg['in']['db_path'], mode='r') as store:
        for (tbl, coefs) in h5_names_gen(cfg_out, cfg['in']['tables'], cfg['in']['db_path']):
            intervals_with_data = 0

            # Get data in ranges
            for intervals_with_data, (df0, start_end) in enumerate(gen_loaded(tbl)):
                db_suffixes = cfg["in"]["db_path"].suffixes[:-1]
                if (
                    (len(db_suffixes) and not db_suffixes[-1].startswith(".proc"))
                    or "incl" not in cfg["in"]["db_path"].stem
                ):  # have processed data (not averaged)
                    df = df0
                else:  # loading source data and calculate velocity
                    df0 = filter_local(df0, cfg['filter'])
                    df = incl_calc_velocity_nodask(df0, **coefs, cfg_filter=cfg['in'], cfg_proc=cfg['proc'])
                data_str_range = fv.str_time_range(
                    *start_end.astype("M8[s]").tolist(), sep="_", sep_interval="-"
                )
                data_name = f"{tbl}/PSD_{data_str_range}dt={data_name_suffix}"
                yield (df, tbl, data_name)

            # Yield sentinel value at end of table processing to signal end and collect
            # total intervals with no data
            intervals_with_no_data = intervals_generated - intervals_with_data
            yield (None, tbl, intervals_with_no_data)

@lru_cache()  # (maxsize=None)
def psd_mt_params(
    length: int,
    bandwidth: float,
    low_bias: bool,
    adaptive: bool,
    dt: Optional[float] = None,
    fs: Optional[float] = None,
    n_fft: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Dpss calculation and default parameters.

    Args:
        length: Data length (n_times)
        bandwidth: The bandwidth of the multi taper windowing function in Hz
        low_bias: Only use tapers with more than 90% spectral concentration within bandwidth
        adaptive: Use adaptive weighting for multitaper PSD estimation
        dt: Time step in seconds (float, not timedelta64). If provided, fs = 1/dt
        fs: Sampling frequency in Hz. If provided, dt = 1/fs
        n_fft: FFT size (power of 2). If not provided, calculated from length

    Returns:
        prm: Dict with fields:
            - length
            - n_fft
            - dt (float, time step in seconds)
            - fs (float, sampling frequency in Hz)
            - dpss
            - eigvals
            - adaptive_if_can
            - weights

    Notes:
        The multitaper DPSS (Discrete Prolate Spheroidal Sequences) method requires bandwidth (NW) to be less
        than length/2: NW < M/2 where M is data length or, mathematically the same:
        Kmax (number of tapers = 2*NW) to be less than M (data length)
    """

    prm = {"length": length}
    if dt is not None:
        prm['dt'] = float(dt)
        prm['fs'] = 1.0 / prm['dt']
    elif fs is not None:
        prm['fs'] = float(fs)
        prm['dt'] = 1.0 / prm['fs']
    # elif kwargs.get('dt') is not None:
    #     prm['fs'] = 1.0 / float(kwargs['dt'])
    # else:
    #     prm['dt'] = 1.0 / float(kwargs['fs'])

    # Debug logging to help diagnose multitaper issues
    l.debug(
        f"psd_mt_params: length={length}, "  #, n_fft={prm['n_fft']},
        f"fs={prm['fs']}, bandwidth={bandwidth}, low_bias={low_bias}, adaptive={adaptive}"
    )
    l.debug(
        f"  Multitaper constraint check: 2*bandwidth={2*bandwidth} < length={length} ? "
        f"{2*bandwidth < length}"
    )
    l.debug(f"  Expected Kmax (number of tapers) = 2*bandwidth = {2*bandwidth}")

    prm['dpss'], prm['eigvals'], prm['adaptive_if_can'] = multitaper._compute_mt_params(
        prm['length'], prm['fs'], bandwidth, low_bias, adaptive)  # normalization='length'
    l.debug(f"  Computed {len(prm['eigvals'])} tapers with eigvals: {prm['eigvals']}")
    prm['weights'] = np.sqrt(prm['eigvals'])[np.newaxis, :, np.newaxis]
    return prm


def next_power_of_2(length):
    """Returns same as int(2 ** np.ceil(np.log2(length))))"""
    if length <= 1:
        return 1
    # Check if already a power of 2
    if length & (length - 1) == 0:
        return length
    # Find position of highest set bit and shift
    return 1 << (length - 1).bit_length()


def _get_n_fft(length: int) -> int:
    """
    Calculate n_fft (power of 2) from data length.

    Args:
        length: Data length (number of samples)

    Returns:
        n_fft: FFT size (power of 2, at least 256)
    """
    return max(256, next_power_of_2(length))



def _get_freqs(
    n_fft: int,
    dt: float,
    fmin: float,
    fmax: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate frequency array and freq_mask from n_fft and dt.

    Args:
        n_fft: FFT size (number of FFT points)
        dt: Time step in seconds (float, not timedelta64)
        fmin: Minimum frequency in Hz
        fmax: Maximum frequency in Hz

    Returns:
        freqs: Frequency array (filtered by fmin and fmax), shape (n_freqs,)
        freq_mask: Boolean mask for filtering full frequency array inside `psd_mt(dt, n_fft)`,
                 shape (n_fft // 2 + 1,)
    """
    full_freqs = np.fft.rfftfreq(n_fft, dt)  # only keep positive frequencies
    freq_mask = (fmin <= full_freqs) & (full_freqs <= fmax)
    return full_freqs[freq_mask], freq_mask


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
    if adaptive_if_can:
        # # from mne.parallel import parallel_func
        # # from mne.time_frequency.multitaper import _psd_from_mt_adaptive
        # psds = list(
        #     _psd_from_mt_adaptive(x, eigvals, np.ones((sum(freq_mask),), dtype=np.bool_))
        #      # x already masked so we put all ok mask
        #            for x in np.array_split(x_mt, 1)
        #            )
        # psd = np.concatenate(psds)
        psd = _psd_from_mt_adaptive(x_mt, eigvals, np.ones((freq_mask.sum(),), dtype=np.bool_))

        # make output units V^2/Hz:  (like mne.mne.time_frequency.psd_array_multitaper option normalization = 'full')
    else:
        psds = weights * x_mt
        psds *= psds.conj()  # same to abs(psd)**2
        psd = psds.real.sum(axis=-2)
        psd *= 2 / (weights * weights.conj()).real.sum(axis=-2)

    psd *= dt
    return psd


def psd_calc(df, fs, freqs, adaptive=None, b_plot=False, **kwargs):
    """
    Compute Power Spectral Densities (PSDs) of df.u/v using multitaper method
    :param df: dataframe with u and v
    :param b_plot:
    :param kwargs: psd_mt kwargs
    :return:
    """

    psdm_Ve = psd_mt(df.u.to_numpy(), **kwargs)[0, :]
    psdm_Vn = psd_mt(df.v.to_numpy(), **kwargs)[0, :]

    if False:
        ## high level mne functions recalcs windows each time
        from mne.time_frequency import psd_array_multitaper  # third_party
        multitaper.warn = l.warning
        psdm_Ve, freq = psd_array_multitaper(
            df.u, sfreq=fs, adaptive=adaptive,
            normalization='length')  # fmin=0, fmax=0.5,
        psdm_Vn, freq = psd_array_multitaper(df.v, sfreq=fs, adaptive=adaptive, normalization='length')  #

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

    ds = xr.Dataset({
        'PSD_Ve': (('time', 'freq'), psdm_Ve[np.newaxis]),
        'PSD_Vn': (('time', 'freq'), psdm_Vn[np.newaxis]),
        'time': df.index[:1].values,
        'freq': freqs
    })
    return ds


def get_sampling_frequency(db_path: Path, table: str) -> float:
    """
    Determine sampling frequency from HDF5 table.

    Args:
        db_path: Path to HDF5 store
        table: Name of table to check

    Returns:
        Sampling frequency in Hz
    """
    with pd.HDFStore(str(db_path), mode="r") as store:
        df = store[table]
        if len(df) < 2:
            raise ValueError(
                f"Table {table} has insufficient data to determine sampling frequency"
            )
        # Calculate time difference between consecutive samples
        dt = (df.index[1] - df.index[0]).total_seconds()
        fs = 1.0 / dt
    return fs


def _find_available_filename(nc_file):
    """
    Find available filename by adding numbered suffix if original exists or is locked.

    :param nc_file: original file path
    :return: available file path
    """
    for i in range(1, 100):
        nc_file_new = nc_file.with_name(f"{nc_file.stem}_{i}{nc_file.suffix}")
        if not nc_file_new.exists():
            return nc_file_new
    raise RuntimeError("Could not find available filename after trying 99 suffixes")


def init_psd_nc_file(
    db_path, n_time=None, tables_in=None, db_path_in=None,  # good_len: int = 100
):
    """
    Creates unlimited size "time" and 1 element size "value" dimensions.
    Creates temporary file for first pass processing.
    If file exists and contains the table group, opens it in append mode.
    If file is locked (permission denied), tries adding number suffix to filename.
    :param db_path: path to output file (without .nc extension)
    :param table: name of the table/group to create or open
    :param n_time: number of time dimensions (None for unlimited)
    :param db_path_in=cfg["in"]["db_path"]: input configuration dictionary
    :param tables_in=cfg["in"]["tables"]: output configuration dictionary
    :return: nc_root, nc_psd, tables_time_range_prev
    nc_root: Openes NetCDF file handle you'll need to close
    tables_time_range_prev: dict with table names as keys and tuples (time_start, time_end) of initial deleting time range as values
    """

    def _create_nc_psd_structure(nc_root, n_time):
        """Create common NetCDF file structure with CF-compliant coordinates"""
        nc_psd = nc_root.createGroup("psd")
        # Note: Dimensions are now created per-table, not at root or psd level
        # This allows each table to have independent time dimension sizes
        return nc_psd

    nc_file = Path(db_path).with_suffix(".nc")
    nc_file_temp = nc_file.with_suffix(".tmp.nc")
    tables_time_range_prev = {}

    # Determine source table time ranges first (for tables that don't exist in NetCDF)
    if db_path_in and tables_in:
        # Get source tables using same function that we use internally in main processing
        if len(tables_in) == 1 and "*" in tables_in[0]:
            with pd.HDFStore(db_path_in, mode="r") as store:
                tables_in = h5.find_tables(store, tables_in[0])
        else:
            tables_in = tables_in

        # Determine time range for each source table
        for tbl_in in tables_in:
            tbl = tbl_in.replace("incl", "_i")  # normalized output NetCDF dataset name
            try:
                with pd.HDFStore(db_path_in, mode="r") as store:
                    if tbl_in in store:
                        df_in = store.select(tbl_in, columns=None, stop=1)
                        if len(df_in) > 0:
                            # Time range starts at datetime.min and ends at first source table time
                            # This allows excluding time ranges from generation that before source data
                            tables_time_range_prev[tbl] = (pd.Timestamp.min, df_in.index[0].tz_localize(None))
                            l.info(f"Source table {tbl} time range start: {df_in.index[0]}")
            except Exception as e:
                l.warning(f"Could not determine time range for source table {tbl}: {e}")

    # Check if file exists and try to read existing tables
    existing_tables = []
    if nc_file.exists():
        try:
            # Open existing file without context manager to keep it open for caller
            # The caller (main()) is responsible for closing the file
            nc_root = netCDF4.Dataset(nc_file, "a")
            # Check if the psd group exists and get list of tables (subgroups) within it
            if "psd" in nc_root.groups:
                nc_psd = nc_root.groups["psd"]
                existing_tables = list(nc_psd.groups.keys())
                l.info(f"Found existing file with tables: {existing_tables}")

                # todo: Remove tables that need to be overwritten (has < good_len records in NetCDF)
                # Only check tables that are in both existing_tables and source tables
                n_datasets = 0
                for tbl in existing_tables.copy():
                    if tbl in tables_time_range_prev:
                        nc_tbl = nc_psd.groups[tbl]
                        # Check the 'time' dimension in the table (dimensions are now per-table)
                        if 'time' in nc_tbl.dimensions:
                            # Get time size from the table's dimension
                            time_size = nc_tbl.dimensions['time'].size

                            # Table is good (>= good_len records), get its time range
                            if 'time_start' in nc_tbl.variables and 'time_end' in nc_tbl.variables:
                                # Replace source table time ranges for tables that exist in NetCDF
                                # This allows excluding time ranges from generation that we already have
                                try:
                                    time_start = pd.Timestamp(nc_tbl.variables["time_start"][0].item())
                                    time_end = pd.Timestamp(nc_tbl.variables['time_end'][-1].item())
                                except (IndexError, ValueError, TypeError):
                                    l.exception("bad time_start or time_end")
                                else:
                                    # Warn if source data is before previous table's end time
                                    if (
                                        (tables_time_range_prev[tbl][0] < time_start)
                                        and tables_time_range_prev[tbl][0] != pd.Timestamp.min
                                    ): # or (tables_time_range_prev[tbl][1] < time_start):
                                        l.error(
                                            f"Source data starts at {tables_time_range_prev[tbl][-1]} which is "
                                            f"before existing output time start {time_end[0] or time_start[-1]}"
                                        )
                                        raise ValueError("Not going to write data: will be not sorted by time")
                                    tables_time_range_prev[tbl] = (time_start, time_end)
                                    l.info(
                                        f"Table {tbl} has {time_size} records, "  # >= {good_len}
                                        f"time range: {time_start} to {time_end}"
                                    )
                                    n_datasets += 1
                                    continue
                        # Table needs to be overwritten
                        existing_tables.remove(tbl)
                        time_size = "no"
                        l.warning(
                            f"NetCDF dataset {tbl} has {time_size} time records => "
                            "will be updated. Keeping"  # todo: Delete if < {good_len}...
                        )
                        # del nc_psd.groups[tbl]  # really does nothing useful: deleting is not supported
                if n_datasets:
                    l.info(
                        f"For {n_datasets} existed NetCDF datasets, "  #  number of records > {good_len}
                        "we will use existing time ranges to skip duplicate intervals"
                    )

            if existing_tables:
                nc_root.close()
                # Rename existing file to temporary file for first pass
                nc_file.rename(nc_file_temp)
                l.info(f"Renamed existing file to temporary: {nc_file_temp}")
                nc_root = netCDF4.Dataset(nc_file_temp, "w", format="NETCDF4")
                return nc_root, nc_psd, tables_time_range_prev
            else:
                l.info(f"Existing {nc_file} file with no known/good data will be recreated")
                nc_root.close()
                nc_file.unlink()
        except PermissionError:
            l.warning(f"File {nc_file.name} is locked, trying with number suffix")
            nc_file = _find_available_filename(nc_file)
        except (OSError, RuntimeError) as e:
            # File is corrupted or has unknown format - try to delete and recreate
            file_size = nc_file.stat().st_size
            if file_size < 300:
                l.warning(f"File has too small size ({file_size}B), deleting and recreating: {nc_file.name}")
            else:
                l.warning(f"File is corrupted or has unknown format, deleting: {nc_file.name}: {e}")
            try:
                nc_file.unlink()
            except PermissionError:
                # File is locked, try adding number suffix
                l.warning("Cannot delete locked file, trying with number suffix")
                nc_file = _find_available_filename(nc_file)
            # Fall through to create new file below

    # Create new temporary file for first pass
    if nc_file_temp.is_file():
        l.warning("Overwritting previous temporary file...")
    nc_root = netCDF4.Dataset(nc_file_temp, "w", format="NETCDF4")
    nc_psd = _create_nc_psd_structure(nc_root, n_time)

    return nc_root, nc_psd, tables_time_range_prev



def _validate_cfg_parameters(cfg: Mapping[str, Any]) -> None:
    """
    Validate critical configuration parameters before processing.

    Args:
        cfg: Configuration dictionary containing proc and in sections

    Raises:
        ValueError: If any critical parameter is invalid
        TypeError: If parameter has wrong type
    """
    # Validate dt_interval
    dt_interval = cfg['proc'].get('dt_interval')
    if dt_interval is not None:
        # Check if timedelta64 units are properly defined (only for timedelta64)
        if isinstance(dt_interval, np.timedelta64):
            unit, step = np.datetime_data(dt_interval.dtype)
            if unit == 'generic':
                raise ValueError(
                    f"dt_interval has undefined units (generic). "
                    f"Please specify units explicitly (e.g., np.timedelta64(3600, 's') "
                    f"or timedelta(hours=1)). Got: {dt_interval}"
                )
        # Convert to timedelta64 for validation
        if not isinstance(dt_interval, np.timedelta64):
            try:
                if isinstance(dt_interval, timedelta):
                    # Convert Python timedelta to timedelta64 with explicit units
                    dt_interval = np.timedelta64(int(dt_interval.total_seconds() * 1e9), 'ns')
                elif isinstance(dt_interval, (int, np.integer)):
                    # For int, we cannot assume units - raise error
                    raise ValueError(
                        f"dt_interval is an integer without units. "
                        f"Please specify units explicitly (e.g., np.timedelta64(3600, 's') "
                        f"or timedelta(hours=1)). Got: {dt_interval}"
                    )
                else:
                    raise ValueError(
                        f"Invalid dt_interval type: {type(dt_interval)}. "
                        f"Expected timedelta or np.timedelta64 with explicit units."
                    )
            except (TypeError, AttributeError) as e:
                raise ValueError(
                    f"Invalid dt_interval type: {type(dt_interval)}. "
                    f"Expected timedelta or convertible to timedelta. Error: {e}"
                ) from e

        # Check if dt_interval is zero or negative
        if dt_interval <= np.timedelta64(1, 'ms'):
            raise ValueError(
                f"dt_interval must be positive, got: {dt_interval}. "
                f"This will result in no data being processed."
            )

    # Validate overlap parameter
    overlap = cfg['proc'].get('overlap')
    if overlap is not None:
        if not isinstance(overlap, (int, float)):
            raise TypeError(f"overlap must be numeric, got: {type(overlap)}")
        if not (0 <= overlap < 1):
            raise ValueError(f"overlap must be in range [0, 1), got: {overlap}")

    # Validate frequency parameters
    fs = cfg['in'].get('fs')
    if fs is not None:
        if not isinstance(fs, (int, float)) or fs <= 0:
            raise ValueError(f"fs (sampling frequency) must be positive, got: {fs}")

    fmin = cfg['proc'].get('fmin')
    if fmin is not None:
        if not isinstance(fmin, (int, float)) or fmin < 0:
            raise ValueError(f"fmin must be non-negative, got: {fmin}")
    fmax = cfg['proc'].get('fmax')
    if fmax is not None:
        if not isinstance(fmax, (int, float)) or fmax <= 0:
            raise ValueError(f"fmax must be positive, got: {fmax}")

    # Validate fmin < fmax if both specified
    if fmin is not None and fmax is not None and fmin >= fmax:
        raise ValueError(f"fmin ({fmin}) must be less than fmax ({fmax})")

    # Validate time range parameters
    min_date = cfg['in'].get('min_date')
    max_date = cfg['in'].get('max_date')

    if min_date is not None and max_date is not None:
        try:
            min_dt = pd.to_datetime(min_date)
            max_dt = pd.to_datetime(max_date)
            if min_dt >= max_dt:
                raise ValueError(
                    f"min_date ({min_date}) must be before max_date ({max_date})"
                )
        except Exception as e:
            raise ValueError(f"Invalid date format in min_date or max_date: {e}") from e

    # Validate tables list
    tables = cfg['in'].get('tables')
    if not tables or not isinstance(tables, (list, tuple)) or len(tables) == 0:
        raise ValueError("tables list must be non-empty")


def _create_freq_variable(
    nc_group: netCDF4.Group,
    freq_dim_name: str,
    f_masked: np.ndarray
) -> None:
    """
    Create frequency dimension and variable in a NetCDF group with CF-compliant attributes.
    If dimension already exists, only creates/overwrites the variable.

    Args:
        nc_group: NetCDF group (nc_psd or table group)
        freq_dim_name: Name of the frequency dimension ('freq')
        f_masked: Frequency array values
    """
    # Only create dimension if it doesn't exist (dimensions cannot be deleted in NetCDF4)
    if freq_dim_name not in nc_group.dimensions:
        nc_group.createDimension(freq_dim_name, f_masked.size)
    # Create or overwrite variable
    nv_freq = nc_group.createVariable(freq_dim_name, 'f4', (freq_dim_name,), zlib=True)
    nv_freq.standard_name = 'frequency'
    nv_freq.axis = 'X'
    nv_freq.units = 'Hz'
    nv_freq[:] = f_masked


def _get_or_create_freq_dim(
    nc_tbl: netCDF4.Group,
    n_fft: int,
    dt: float,
    fmin: float,
    fmax: float,
    tbl: str,
    nc_psd: netCDF4.Group
) -> Tuple[str, np.ndarray]:
    """
    Get or create frequency dimension in table group.

    Creates global frequency dimension in nc_psd group on first call.
    Only creates table-specific frequencies when they differ from global frequencies
    (i.e., when n_fft changes across tables). Otherwise, references global dimension.

    Args:
        nc_tbl: NetCDF table group
        n_fft: Current n_fft value (FFT size)
        dt: Time step in seconds (float, not timedelta64)
        fmin: Minimum frequency in Hz
        fmax: Maximum frequency in Hz
        tbl: Current table name
        nc_psd: NetCDF psd group (for creating global frequency dimension)

    Returns:
        Tuple of (freq_dim_name, freq_mask):
            - freq_dim_name: Name of the frequency dimension to use ('freq')
            - freq_mask: Boolean mask for filtering full frequency array inside `psd_mt(dt, n_fft)`,
                         shape (n_fft // 2 + 1,)
    """
    f_masked, freq_mask = _get_freqs(n_fft, dt, fmin, fmax)
    freq_dim_name = 'freq'

    # Get or create global frequency dimension in nc_psd group
    try:
        # Validate that global frequency dimension matches current fmin/fmax constraints
        # If global frequencies don't match current constraints, we need table-specific dimension
        global_freqs = nc_psd.variables['freq'][:]
        # Check if global frequencies are within current fmin/fmax range
        if global_freqs.size == freq_mask.size:
            global_fmin, global_fmax = global_freqs[0], global_freqs[-1]
            # Allow small tolerance for floating point comparison
            if (abs(global_fmin - fmin) < 1e-6 and abs(global_fmax - fmax) < 1e-6):
                return freq_mask
    except KeyError:
        _create_freq_variable(nc_psd, freq_dim_name, f_masked)
        l.info(
            f"Created global freq dimension in nc_psd with n_fft={n_fft}, "
            f"{f_masked.size} frequencies in range [{f_masked[0]:.4f}, {f_masked[-1]:.4f}] Hz"
        )
        return freq_mask  # not need to create table-specific frequency

    # Create or reuse table-specific frequency dimension
    if freq_dim_name not in nc_tbl.dimensions:
        _create_freq_variable(nc_tbl, freq_dim_name, f_masked)
        l.info(
            f"Created table-specific freq dimension in '{tbl}' with n_fft={n_fft}, "
            f"{f_masked.size} frequencies in range [{f_masked[0]:.4f}, {f_masked[-1]:.4f}] Hz"
        )
    else:
        # Check if existing freq dimension size matches current n_fft
        if nc_tbl.dimensions[freq_dim_name].size != f_masked.size:
            # Size differs, need to overwrite frequency variable (dimension cannot be deleted)
            if freq_dim_name in nc_tbl.variables:
                try:
                    del nc_tbl.variables[freq_dim_name]
                    _create_freq_variable(nc_tbl, freq_dim_name, f_masked)
                    l.info(
                        f"Overwrote freq variable in table '{tbl}' from "
                        f"{nc_tbl.dimensions[freq_dim_name].size} to {f_masked.size} frequencies "
                        f"in range [{f_masked[0]:.4f}, {f_masked[-1]:.4f}] Hz (n_fft={n_fft})"
                    )
                except Exception:
                    l.exception(f"Cannot recreate freq dimension in table '{tbl}'")
            else:
                l.warning(
                    f"Freq dimension exists in table '{tbl}' but variable missing. "
                    f"Dimension size {nc_tbl.dimensions[freq_dim_name].size} differs from required {f_masked.size}. "
                    f"Cannot resize NetCDF dimension - data may be truncated."
                )
    return freq_mask


def _create_final_nc_file(
    db_path: str,
    nc_psd_temp: netCDF4.Group,
    tables_time_range_prev: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]],
    tables_to_recalc: List[str],
) -> Tuple[netCDF4.Dataset, netCDF4.Group, Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]]:
    """
    Create final NetCDF file with correct frequency dimensions.
    Copies data from tables that don't need recalculation.
    If all tables have correct n_fft, just renames temporary file.

    Args:
        db_path: Path to output file (without .nc extension)
        nc_psd_temp: Temporary NetCDF psd group from first pass (must be from open file)
        tables_time_range_prev: Dictionary of existing time ranges to exclude
        tables_to_recalc: List of tables that need to be recalculated

    Returns:
        Tuple of (nc_root, nc_psd, tables_time_range_prev):
            - nc_root: Opened NetCDF file handle (caller must close)
            - nc_psd: NetCDF psd group in final file
            - tables_time_range_prev: Updated dictionary excluding copied tables
    """
    nc_file = Path(db_path).with_suffix(".nc")
    nc_file_temp = nc_file.with_suffix(".tmp.nc")

    if not tables_to_recalc:
        # All tables have correct n_fft, just rename temporary file
        l.info("All tables have correct n_fft, renaming temporary {nc_file_temp.name} to final {nc_file.name}")
        nc_psd_temp.parent.close()  # Close the temporary root
        nc_file_temp.rename(nc_file)
        # Reopen the final file
        nc_root = netCDF4.Dataset(nc_file, "a")
        nc_psd = nc_root.groups["psd"]
        return nc_root, nc_psd, tables_time_range_prev

    # Some tables need recalculation - create new file and copy unchanged tables
    l.info(f"Creating final file and copying all except {tables_to_recalc} tables needing to recalcutate")

    # Create new final file
    try:
        nc_root = netCDF4.Dataset(nc_file, "w", format="NETCDF4")
    except PermissionError:
        nc_file = _find_available_filename(nc_file)
        l.warning(
            f"File {nc_file.name} is locked ⇒ use available {nc_file.name} "
            "name instead"
        )
        nc_root = netCDF4.Dataset(nc_file, "w", format="NETCDF4")
    nc_psd = nc_root.createGroup("psd")

    # Copy global attributes and frequency dimension from temp file
    for attr_name in nc_psd_temp.parent.ncattrs():
        nc_root.setncattr(attr_name, nc_psd_temp.parent.getncattr(attr_name))

    # Copy global frequency dimension if it exists
    if 'freq' in nc_psd_temp.dimensions and 'freq' in nc_psd_temp.variables:
        f_masked = nc_psd_temp.variables['freq'][:]
        _create_freq_variable(nc_psd, 'freq', f_masked)
        l.info(f"Copied global freq dimension with {f_masked.size} frequencies")

    # Copy tables that don't need recalculation
    tables_to_copy = [tbl for tbl in nc_psd_temp.groups if tbl not in tables_to_recalc]
    for tbl in tables_to_copy:
        l.info(f"Copying table '{tbl}' to final file (no recalculation needed)")
        nc_tbl_temp = nc_psd_temp.groups[tbl]
        nc_tbl = nc_psd.createGroup(tbl)

        # Copy dimensions
        for dim_name in nc_tbl_temp.dimensions:
            dim = nc_tbl_temp.dimensions[dim_name]
            nc_tbl.createDimension(dim_name, dim.size if dim.isunlimited() else None)

        # Copy variables
        for var_name in nc_tbl_temp.variables:
            var_temp = nc_tbl_temp.variables[var_name]
            nc_var = nc_tbl.createVariable(
                var_name,
                var_temp.dtype,
                var_temp.dimensions,
                zlib=var_temp.filters() is not None
            )
            # Copy attributes
            for attr_name in var_temp.ncattrs():
                nc_var.setncattr(attr_name, var_temp.getncattr(attr_name))
            # Copy data
            nc_var[:] = var_temp[:]

        # Update tables_time_range_prev to exclude copied tables
        if tbl in tables_time_range_prev:
            del tables_time_range_prev[tbl]

    # Close and delete temporary file
    nc_psd_temp.parent.close()
    nc_file_temp.unlink()
    l.info(f"Deleted temporary file: {nc_file_temp}")

    return nc_root, nc_psd, tables_time_range_prev


def _log_interval_stats(table_name: str, interval_stats: Dict[str, Any]) -> None:
    """
    Log interval processing statistics for a table.

    Args:
        table_name: Name of table
        interval_stats: Dictionary with structure:
            - 'total_intervals': Total intervals with time coordinates written to NetCDF
            - 'skip_no_data': Intervals skipped due to no data in HDF5
            - 'skip_no_eigvals': Intervals written with NaN PSD values (invalid eigvals)
            - 'skip_data_len': Intervals skipped due to insufficient data length
            - 'skip_nfft_changed': Intervals skipped due to n_fft mismatch
            - 'cols': Dict with column names as keys and stats dicts as values
              Each stats dict contains 'adaptive', 'non_adaptive', and
              skip reason counts: 'skip_no_variation'
    """
    l.info("=" * 70)
    l.info(f"Interval Processing Statistics for table '{table_name}':")
    l.info(
        f"  Intervals with time coordinates written to NetCDF: "
        f"{interval_stats.get('total_intervals', 0)}"
    )
    if (s := interval_stats.get("skip_no_eigvals")):
        l.info(f"  Intervals written with NaN PSD (invalid eigvals): {s}")
    if (s := interval_stats.get("skip_data_len")):
        l.info(f"  Intervals skipped (insufficient data): {s}")
    if (s := interval_stats.get("skip_nfft_changed")):
        l.info(f"  Intervals skipped (n_fft mismatch): {s}")
    l.info(f"  Intervals with no data in HDF5: {interval_stats.get('skip_no_data', 0)}")
    l.info("=" * 70)
    for col_name, stats in interval_stats.get('cols', {}).items():
        adaptive = stats.get('adaptive', 0)
        non_adaptive = stats.get('non_adaptive', 0)
        skip_no_variation = stats.get('skip_no_variation', 0)

        total_skipped = skip_no_variation
        total_processed = adaptive + non_adaptive + total_skipped
        if total_processed > 0:
            adaptive_pct = (adaptive / total_processed) * 100
            non_adaptive_pct = (non_adaptive / total_processed) * 100
            skipped_pct = (total_skipped / total_processed) * 100

            l.info(
                f"  Column '{col_name}': "
                f"Adaptive: {adaptive} ({adaptive_pct:.1f}%), "
                f"Non-adaptive: {non_adaptive} ({non_adaptive_pct:.1f}%), "
                f"Skipped: {total_skipped} ({skipped_pct:.1f}%) "
                f"[Total processed: {total_processed}]"
            )
            if total_skipped > 0:
                l.info(f"    Skip reasons: no_variation: {skip_no_variation}")
    l.info("=" * 70)


def _process_intervals_common(
    cfg: Mapping[str, Any],
    cfg_out: Mapping[str, Any],
    tables_time_range_prev: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]],
    nc_psd: netCDF4.Group,
    prm: Dict[str, Any],
    table_time_ranges: Dict[str, Dict[str, pd.Timestamp]],
    interval_stats: Dict[str, Any],
    n_fft_override: Optional[callable] = None,
    n_fft_callback: Optional[callable] = None,
) -> None:
    """
    Common processing logic for both first and second pass.

    Args:
        cfg: Configuration dictionary
        cfg_out: Output configuration dictionary
        tables_time_range_prev: Dictionary of existing time ranges to exclude
        nc_psd: NetCDF psd group
        prm: Parameters dictionary for PSD calculation, prm[length] must be not set to any possible value
        table_time_ranges: Dictionary to store time ranges per table
        interval_stats: Dictionary to track interval processing statistics with structure:
                       {'total_intervals': int, 'skip_no_data': int,
                        'skip_no_eigvals': int, 'skip_data_len': int,
                        'skip_nfft_changed': int,
                        'cols': {column_name: {'adaptive': int, 'non_adaptive': int,
                                               'skip_no_variation': int}}}
        n_fft_override: Optional callable(tbl) that returns n_fft override value or None
        n_fft_callback: Optional callback function(tbl, n_fft) called when n_fft changes (see `n_fft_tracker` that is used as n_fft_callback)
    """
    tbl_update_stat = ""  # also used as trigger to create dimensions if not None (even if '')

    itbl = 0
    out_row: int = 0
    cols: List[str] = []
    nc_tbl: netCDF4.Dataset

    # Initialize interval statistics for current table
    interval_stats = {
        "total_intervals": 0,
        "skip_no_data": 0,
        "skip_no_eigvals": 0,
        "skip_data_len": 0,
        "skip_nfft_changed": 0,
        "cols": {},
    }

    for df, tbl_in, dataname in h5_velocity_by_intervals_gen(cfg, cfg_out, tables_time_range_prev):
        tbl = tbl_in.replace("incl", "_i")

        # Check for sentinel value (df is None) indicating end of table processing
        if df is None:
            tbl_update_stat = tbl  # other table processing required later
            # Update interval statistics for completed table
            interval_stats["skip_no_data"] = dataname
            itbl += 1
            cols = []
            prm["length"] = -1  # initialise with any inpossible value to trigger prm update

            # Log statistics for completed table
            interval_stats["total_intervals"] = out_row
            _log_interval_stats(tbl, interval_stats)
            continue

        # interpolate to regular grid
        df, bads = df_interp(df, fs=prm["fs"], cols=cols)
        del bads  # todo: use
        len_data_cur = df.shape[0]

        if not cols:
            for col in ["Pressure", "u", "v"]:
                if col in df.columns:
                    cols.append(col)
            if not any(cols):
                # Not inclinometer / wavegage => use all columns
                cols = df.columns

            # Initialize column-specific statistics for new table
            interval_stats["cols"] = {
                col: {"adaptive": 0, "non_adaptive": 0, "skip_no_variation": 0} for col in cols
            }

        # Requirement froms _compute_mt_params() in mne.time_frequency.multitaper
        # `bandwidth * len_data_cur / (2 * fs) >= 0.5` =>
        # Skip processing this interval as data is insufficient
        if len_data_cur < prm["fs"] / prm["bandwidth"]:
            l.warning(
                " %d. %s: len=%s < required %.0f ⇒ skipping",
                out_row,
                dataname,
                len_data_cur,
                prm["fs"] / prm["bandwidth"],
            )
            interval_stats["skip_data_len"] += 1

            continue

        dt = np.median(np.diff(df.index.values)).item() / 1e9

        # Validate and update sampling frequency
        check_fs = 1 / dt
        if prm.get("fs"):
            np.testing.assert_almost_equal(prm["fs"], check_fs, decimal=3, err_msg="", verbose=True)
        else:
            prm["fs"] = check_fs

        if prm["length"] != len_data_cur:
            # Recalculate psd_mt_params with new length
            prm["n_fft"] = _get_n_fft(len_data_cur) if n_fft_override is None else n_fft_override(tbl)

        # Call n_fft callback if provided (for first pass prm['n_fft'] tracking)
        if n_fft_callback is not None:
            _ = n_fft_callback(tbl, prm["n_fft"], tbl_update_stat)
            if _:
                # n_fft changed for current tbl, but we can save data only for same n_fft per tbl
                prm["n_fft"] = _  # forse to use previous n_fft
                # interval_stats['skip_nfft_changed'] += 1

        if prm["length"] != len_data_cur:
            # Recalculate other params if length changed
            prm.update(
                psd_mt_params(
                    length=len_data_cur,
                    bandwidth=prm["bandwidth"],
                    low_bias=prm["low_bias"],
                    adaptive=prm["adaptive"],
                    dt=dt,
                )
            )

            # Recalculate other params 2: dpss, eigvals, weights for current length

            # Track if adaptive method was requested but not available
            adaptive_requested = prm["adaptive"]

            # Suppress MNE warnings about adaptive combination - statistics tracking is sufficient
            # Save original warn function and replace with filter that suppresses adaptive warnings
            original_warn = multitaper.warn

            def filtered_warn(msg, *args, **kwargs):
                """Filter out adaptive combination warnings - statistics track these"""
                msg_str = str(msg)
                # Only suppress warnings about adaptive combination (which we track in stats)
                if "adaptively combining" not in msg_str.lower() and "low_bias" not in msg_str.lower():
                    original_warn(msg, *args, **kwargs)

            multitaper.warn = filtered_warn

            try:
                (prm["dpss"], prm["eigvals"], prm["adaptive_if_can"]) = multitaper._compute_mt_params(
                    prm["length"], prm["fs"], prm["bandwidth"], prm["low_bias"], prm["adaptive"]
                )
                # Track non-adaptive intervals when adaptive was requested but not available
                if adaptive_requested and not prm["adaptive_if_can"]:
                    for col in cols:
                        interval_stats["cols"][col]["non_adaptive"] += 1
            except (ModuleNotFoundError, ValueError):
                # l.error() already reported as multitaper.warn is reassignred to l.warning()
                prm["eigvals"] = np.int32([0])
                if adaptive_requested:
                    for col in cols:
                        interval_stats["cols"][col]["non_adaptive"] += 1
            finally:
                # Restore original warn function
                multitaper.warn = original_warn

            prm["weights"] = np.sqrt(prm["eigvals"])[np.newaxis, :, np.newaxis]

            if tbl_update_stat is not None:
                tbl_update_stat = None
                try:
                    nc_tbl = nc_psd.groups[tbl]
                    out_row = nc_tbl.variables[col[0]].size
                    l.info('    %d. Updating "%s"', itbl, tbl)
                except KeyError:
                    nc_tbl = nc_psd.createGroup(tbl)
                    # Create dimensions for this table (per-table dimensions for independent sizes)
                    nc_tbl.createDimension("time", None)  # unlimited
                    nc_tbl.createDimension("value", 1)
                    # Create CF-compliant time coordinate variable in this table
                    nc_time = nc_tbl.createVariable("time", "f8", ("time",), zlib=True)
                    nc_time.standard_name = "time"
                    nc_time.axis = "T"
                    nc_time.units = "seconds since 1970-01-01 00:00:00"
                    nc_time.calendar = "gregorian"

                    # Create frequency dimension for this table using current n_fft, dt, fmin, fmax
                    # This ensures freq dimension exists before creating variables that reference it
                    prm["freq_mask"] = _get_or_create_freq_dim(
                        nc_tbl,
                        prm["n_fft"],
                        prm["dt"],
                        prm["fmin"],
                        prm["fmax"],
                        tbl=tbl,
                        nc_psd=nc_psd,
                    )
                    out_row = 0
                    l.info('    %d. Created "%s"', itbl, tbl)

                for col in cols:
                    if col not in nc_tbl.variables.keys():
                        nc_tbl.createVariable(
                            col,
                            "f4",
                            (
                                "time",
                                "freq",
                            ),
                            zlib=True,
                        )
                for col in ["time_start", "time_end"]:
                    if col not in nc_tbl.variables.keys():
                        nc_var = nc_tbl.createVariable(col, "f8", ("time",), zlib=True)
                        # Add CF-compliant units attribute for time variables
                        # These variables store datetime64[ns] values converted to seconds since epoch
                        nc_var.units = "seconds since 1970-01-01 00:00:00"
                        nc_var.standard_name = "time"
                        nc_var.calendar = "gregorian"
                # Create time_good_range array variable with dimension 'range' of size 2
                # This replaces the separate time_good_min and time_good_max scalar variables
                # to make them compatible with xarray (which expects arrays, not scalars)
                if "time_good_range" not in nc_tbl.variables.keys():
                    nc_tbl.createDimension("range", 2)
                    nc_var = nc_tbl.createVariable("time_good_range", "f8", ("range",))
                    # Add CF-compliant units attribute for time variable
                    # This variable stores datetime64[ns] values converted to seconds since epoch
                    # Index 0 = time_good_min, Index 1 = time_good_max
                    nc_var.units = "seconds since 1970-01-01 00:00:00"
                    nc_var.standard_name = "time"
                    nc_var.calendar = "gregorian"
                    nc_var.long_name = "time range [min, max] of good data"
                if cfg["proc"]["dt_interval"]:
                    col = "time_interval"
                    if col not in nc_tbl.variables.keys():
                        nc_tbl.createVariable(col, "f4", ("value",))
                    nc_tbl.variables["time_interval"][:] = (
                        cfg["proc"]["dt_interval"].astype("m8[s]").item().total_seconds()
                    )

        l.info(" %d. %s: len=%s", out_row, dataname, len_data_cur)
        # Write time_start, time_end, and time coordinate for this interval
        # Convert datetime64[ns] to seconds since epoch for CF-compliant storage
        time_start_end = df.index[[0, -1]].to_numpy("M8[ns]")
        (
            nc_tbl.variables["time_start"][out_row],
            nc_tbl.variables["time_end"][out_row],
            nc_tbl.variables["time"][out_row],
        ) = (
            *time_start_end.astype("datetime64[s]").astype(int),
            (time_start_end.astype(int).mean() / 1e9).astype(int),
        )
        # the later item result above also can be obtained like this:
        # date2num((time_start_end.astype("M8[ns]").astype(int).mean()/1e9).astype("M8[s]").item(),
        #     "seconds since 1970-01-01 00:00:00")

        # Calculate PSD with current n_fft
        if prm["eigvals"].any():
            b_ok_cols = np.diff(df[cols].values, axis=0).any(axis=0)
            for var_name, b_ok_col in zip(cols, b_ok_cols):
                if not b_ok_col:
                    interval_stats["cols"][var_name]["skip_no_variation"] += 1
                    nc_tbl.variables[var_name][out_row, :] = np.nan
                    continue

                if prm.get("adaptive_if_can", False):
                    interval_stats["cols"][var_name]["adaptive"] += 1

                nc_tbl.variables[var_name][out_row, :] = call_with_valid_kwargs(psd_mt, df[var_name], **prm)[
                    0, :
                ]

            # Update time range for current table
            st, en = df.index[[0, -1]].to_numpy("M8[s]")
            if tbl not in table_time_ranges:
                # Initialize time_good_range array with [min, max]
                table_time_ranges[tbl] = np.array([st, en], dtype="M8[s]")
            else:
                # Update min and max values in the array
                if table_time_ranges[tbl][0] > st:
                    table_time_ranges[tbl][0] = st
                if table_time_ranges[tbl][1] < en:
                    table_time_ranges[tbl][1] = en
        else:
            interval_stats["skip_no_eigvals"] += 1
            for var_name in cols:
                nc_tbl.variables[var_name][out_row, :] = np.nan
        out_row += 1

    # Write time_good_range for processed tables
    # Convert datetime64[ns] values to seconds since epoch for CF-compliant storage
    for tbl_name, time_range in table_time_ranges.items():
        if tbl_name in nc_psd.groups:
            nc_tbl = nc_psd.groups[tbl_name]
            if "time_good_range" in nc_tbl.variables:
                nc_tbl.variables["time_good_range"][:] = time_range.astype("M8[s]").astype(int)

    # Statistics for last table processed
    if n_fft_callback is not None:
        n_fft_callback(tbl, prm["n_fft"], tbl)
    # Log statistics with total intervals processed and intervals with no data
    if interval_stats:
        interval_stats["total_intervals"] = out_row
        _log_interval_stats(tbl, interval_stats)


def _process_first_pass(
    cfg: Mapping[str, Any],
    cfg_out: Mapping[str, Any],
    tables_time_range_prev: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]],
    nc_psd: netCDF4.Group,
    prm: Dict[str, Any],
    table_time_ranges: Dict[str, Dict[str, pd.Timestamp]],
    tables_n_fft_m: Dict[str, Optional[int]],
    tables_n_fft_prev: Dict[str, Optional[int]],
    interval_stats: Dict[str, Any],
) -> None:
    """
    First pass: Process all intervals and track n_fft statistics per table.
    Creates global frequency dimension in nc_psd group for first table.

    Args:
        cfg: Configuration dictionary
        cfg_out: Output configuration dictionary
        tables_time_range_prev: Dictionary of existing time ranges to exclude
        nc_psd: NetCDF psd group
        prm: Parameters dictionary for PSD calculation
        table_time_ranges: Dictionary to store time ranges per table
        tables_n_fft_m: Dictionary to store most common n_fft per table (modified in place)
        tables_n_fft_prev: Dictionary to store n_fft actually used per table at 1st pass (modified in place)
        interval_stats: Dictionary to track interval processing statistics
    """
    n_fft_table: Optional[int] = None  # Track n_fft actually used (first one encountered)
    n_fft_counts: Dict[int, int] = {}  # {n_fft: count}

    def n_fft_tracker(tbl: str, n_fft: int, tbl_to_update: str | None) -> None | int:
        """
        Callback to track `n_fft` statistics per table returns previous n_fft if changes else None,
        assigns `n_fft_table` to first good `n_fft` for current table

        Args:
            tbl: Current table name
            n_fft: Current n_fft value
            tbl_to_update: If Falsy track statistics else the name to store them: triggers calculation
        """

        nonlocal n_fft_table, n_fft_counts

        # Calculate most common n_fft for previous table when switching tables
        # (ignore if 2nd time called for same table: possible if tbl_to_update is not fully updated while
        # started process next table)
        if tbl_to_update and tbl_to_update not in tables_n_fft_m:
            if n_fft_counts:
                n_fft_m = max(n_fft_counts, key=n_fft_counts.get)
                tables_n_fft_m[tbl_to_update] = n_fft_m
                l.info(
                    f"Table '{tbl_to_update}': Most common n_fft = {n_fft_m} (counts: {n_fft_counts}), used n_fft = {n_fft_table}"
                )

                # Record n_fft actually used
                tables_n_fft_prev[tbl] = n_fft_table

                # Begin record n_fft_counts for new table
                n_fft_counts = {n_fft: 1}
            n_fft_table = n_fft
        else:
            # Track n_fft statistics
            try:
                n_fft_counts[n_fft] += 1
            except KeyError:
                n_fft_counts[n_fft] = 1
            l.debug(f"n_fft={n_fft}, count={n_fft_counts[n_fft]}")

            if n_fft_table:
                if n_fft != n_fft_table:
                    return n_fft_table
            else:
                tables_n_fft_prev[tbl] = n_fft_table = n_fft  # assign for 1st table
        return None

    # Process all intervals with n_fft tracking callback
    _process_intervals_common(
        cfg,
        cfg_out,
        tables_time_range_prev,
        nc_psd,
        prm,
        table_time_ranges,
        interval_stats,
        n_fft_override=None,
        n_fft_callback=n_fft_tracker,
    )


def _process_second_pass(
    cfg: Mapping[str, Any],
    cfg_out: Mapping[str, Any],
    tables_time_range_prev: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]],
    nc_psd: netCDF4.Group,
    prm: Dict[str, Any],
    tables_n_fft_m: Dict[str, Optional[int]],
    tables_n_fft_prev: Dict[str, Optional[int]],
    table_time_ranges: Dict[str, Dict[str, pd.Timestamp]],
    interval_stats: Dict[str, Any],
) -> List[str]:
    """
    Second pass: Recalculate spectrum for tables where initial n_fft guess was wrong.
    Overwrites frequency dimensions with correct n_fft values.

    Args:
        cfg: Configuration dictionary
        cfg_out: Output configuration dictionary
        tables_time_range_prev: Dictionary of existing time ranges to exclude
        nc_psd: NetCDF psd group
        prm: Parameters dictionary for PSD calculation
        tables_n_fft_m: Dictionary of most common n_fft per table from first pass
        table_time_ranges: Dictionary to store time ranges per table
        interval_stats: Dictionary to track interval processing statistics

    Returns:
        List of table names that were recalculated
    """
    # Get list of tables that need recalculation
    tables_to_recalc = [tbl for tbl, n_fft_m in tables_n_fft_m.items() if n_fft_m != tables_n_fft_prev[tbl]]
    if not tables_to_recalc:
        l.info("No tables need recalculation (all tables have correct n_fft)")
        return []

    l.info(f"Recalculating {len(tables_to_recalc)} table(s): {tables_to_recalc}")

    # Filter cfg['in']['tables'] to only include tables that need recalculation
    original_tables = cfg["in"]["tables"]
    cfg["in"]["tables"] = tables_to_recalc

    # Process only tables that need recalculation with n_fft override
    _process_intervals_common(
        cfg,
        cfg_out,
        tables_time_range_prev,
        nc_psd,
        prm,
        table_time_ranges,
        interval_stats,
        n_fft_override=lambda tbl: tables_n_fft_m.get(tbl),
    )

    # Restore original tables list
    cfg["in"]["tables"] = original_tables

    return tables_to_recalc


def main(new_arg=None, **kwargs):
    """
    Accumulats results of different source tables in 2D NetCDF matrices of each result parameter.
    :param new_arg:
    :return:
    Spectrum parameters used (taken from nitime/algorithems/spectral.py):
    - NW: float, by default set to 4: that corresponds to bandwidth of 4 times the fundamental frequency
        The normalized half-bandwidth of the data tapers, indicating a
        multiple of the fundamental frequency of the DFT (Fs/N).
        Common choices are n/2, for n >= 4. This parameter is unitless
        and more MATLAB compatible. As an alternative, set the BW
        parameter in Hz. See Notes on bandwidth.

    - BW: float
        The sampling-relative bandwidth of the data tapers, in Hz.

    - adaptive: {True/False}
    Use an adaptive weighting routine to combine the PSD estimates of
    different tapers.
    - low_bias: {True/False}
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

    cfg = cfg_from_args(my_argparser(), new_arg, **kwargs)
    if not cfg or not cfg['program'].get('return'):
        print('Can not initialise')
        return cfg
    elif cfg['program']['return'] == '<cfg_from_args>':  # to help testing
        return cfg

    init_logging(l, cfg['program']['log'], cfg['program']['verbose'])

    multitaper.warn = l.warning  # module is not installed but copied. so it can not import this dependace

    try:
        cfg['in']['paths'], cfg['in']['nfiles'], cfg['in']['path'] = init_file_names(
            **{**cfg['in'], 'path': cfg['in']['db_path']}, b_interact=cfg['program']['b_interact'])
    except Ex_nothing_done as e:
        print(e.message)
        return ()
    print('\n' + prog, end=' started. ')

    # Validate configuration parameters before processing
    try:
        _validate_cfg_parameters(cfg)
    except (ValueError, TypeError) as e:
        l.error(f"Configuration validation failed: {e}")
        raise

    cfg['in']['columns'] = ['u', 'v', 'Pressure']
    # minimum time between blocks, required in filt_data_dd() for data quality control messages:
    cfg['in']['dt_between_bursts'] = None  # If None report any interval bigger then min(1st, 2nd)
    cfg['in'].setdefault('dt_hole_warning', np.timedelta64(2, 's'))

    cfg_out = cfg['out']
    # Set dt_interval from configuration
    if cfg['proc']['dt_interval']:
        cfg['proc']['dt_interval'] = np.timedelta64(cfg['proc']['dt_interval'])
    # Default overlap to 0.5 if not specified
    if cfg['proc'].get('overlap') is None:
        cfg['proc']['overlap'] = 0.5
    cfg_out['chunksize'] = cfg['in']['chunksize']
    h5.out_init(cfg['in'], cfg_out)
    # cfg_out_table = cfg_out['table']  need? save because will need to change
    cfg_out['save_proc_tables'] = True  # False

    # cfg['proc'] = {}
    prm = cfg['proc']
    prm['adaptive'] = True  # pmtm spectrum param (will warning if can't)

    # Determine sampling frequency if not provided
    if (b_get_f := bool(not cfg["in"]["fs"])):
        # Use first table to determine sampling frequency
        first_table = cfg['in']['tables'][0]
        prm['fs'] = get_sampling_frequency(
            Path(cfg['in']['db_path']), first_table
        )
    else:
        prm['fs'] = cfg['in']['fs']

    prm['low_bias'] = True
    if prm["fmin"] is None:  # 0.0001
        prm["fmin"] = 1.1 / (prm["dt_interval"].astype("m8[s]").item().total_seconds())
    if prm["fmax"] is None:
        prm["fmax"] = prm['fs'] / 2  # 4
    if cfg['proc']['dt_interval']:
        k_burst = 4  # decrease coefficient in burst mode to get real data length relative to dt_interval
        prm["bandwidth"] = (16 / k_burst) / cfg["proc"]["dt_interval"].astype("m8[s]").astype("float")
        # 8 / fs will get 1 tapers for fs=5, dt = 1h. Adaptive requires 3 minimum, so use 32 / fs
        # 8 * 2 * prm['fs']/34000  # 4 * 2 * 5/34000 ~= 4 * 2 * fs / N
    else:
        prm['bandwidth'] = None
    prm['length'] = None
    l.info(
        "Initial spectrum parameters: "
        "fs={fs}Hz%s, fmin={fmin}, fmax={fmax}, bandwidth={bandwidth:g} (from intervals length %s)"
        "".format_map(prm),
        " (determined)" if b_get_f else " (configured)",
        str(cfg["proc"]["dt_interval"]),
    )
    nc_root, nc_psd, tables_time_range_prev = init_psd_nc_file(
        db_path=cfg_out['db_path'],
        n_time=None,
        db_path_in=cfg["in"]["db_path"],
        tables_in=cfg["in"]["tables"]
    )

    # Initializing variables to search data time range of calculated per table
    table_time_ranges: Dict[str, Dict[str, pd.Timestamp]] = {}

    # Track per-table n_fft statistics for two-pass processing
    tables_n_fft_m: Dict[str, Optional[int]] = {}  # {tbl: n_fft_m (most common)}
    tables_n_fft_prev: Dict[str, Optional[int]] = {}  # {tbl: n_fft actually used at 1st pass}

    # Track statistics for interval processing for current table
    interval_stats: Dict[str, Any] = {}


    # First pass: Process all intervals, track n_fft statistics (writes to temporary file)
    _process_first_pass(
        cfg, cfg_out, tables_time_range_prev, nc_psd, prm,
        table_time_ranges, tables_n_fft_m, tables_n_fft_prev, interval_stats
    )

    # Check if any tables need recalculation
    tables_to_recalc = [
        tbl for tbl, n_fft_m in tables_n_fft_m.items() if n_fft_m != tables_n_fft_prev.get(tbl, -1)
    ]

    # Create final file with correct dimensions, copy data
    # from tables that don't need recalculation into it,
    # close the temporary file and open final file instead
    nc_root, nc_psd, tables_time_range_prev = _create_final_nc_file(
        cfg_out["db_path"], nc_psd, tables_time_range_prev, tables_to_recalc
    )

    # Second pass: Recalculate tables where initial n_fft guess was wrong
    tables_recalculated = _process_second_pass(
        cfg, cfg_out, tables_time_range_prev, nc_psd, prm,
        tables_n_fft_m, tables_n_fft_prev, table_time_ranges, interval_stats
    )

    # failed_storages = h5.move_tables(cfg_out)
    print('Ok.', end=' ')
    nc_root.close()

if __name__ == '__main__':
    main()