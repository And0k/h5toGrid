#!/usr/bin/env python
# coding:utf-8
"""
  Author:  Andrey Korzh <ao.korzh@gmail.com>
  Purpose: pandas time convert utils
  Created: 26.02.2016
"""
import logging
import re
from pathlib import PurePath
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

if __debug__:
    from matplotlib import pyplot as plt
    # datetime converter for a matplotlib plotting method
    from pandas.plotting import register_matplotlib_converters

    register_matplotlib_converters()
# from future.moves.itertools import zip_longest
# from builtins import input
# from debug import __debug___print
# from  pandas.tseries.offsets import DateOffset
from dateutil.tz import tzoffset
# my:
from other_filters import make_linear, longest_increasing_subsequence_i, check_time_diff, repeated2increased, rep2mean, \
    find_sampling_frequency, rep2mean_with_const_freq_ends
from utils2init import dir_create_if_need

if __name__ == '__main__':
    l = None  # see main(): l = init_logging(logging, None, cfg['program']['log'], cfg['program']['verbose'])
else:
    l = logging.getLogger(__name__)

dt64_1s = np.int64(1e9)
tzUTC = tzoffset('UTC', 0)


# def numpy_to_datetime(arr):
#     return np.apply_along_axis(np.ndarray.item, 0, np.array([arr], 'datetime64[s]'))


def datetime_fun(fun, *args, type_of_operation='<M8[s]', type_of_result='<M8[s]'):
    """
    :param x: array
    :param fun: function to apply on x
    :param type_of_operation: type to convert x before apply fun to not overflow
    :return: fun result of type type_of_operation

    >>> import pandas as pd; df_index = pd.DatetimeIndex(['2017-10-20 12:36:32', '2017-10-20 12:41:32'], dtype='datetime64[ns]', name='time', freq=None)
    >>> datetime_fun(lambda x: -np.subtract(*x)/2 .view('<m8[s]'), df_index[-2:].values)
    # 150s
    """
    return np.int64(fun(*[x.astype(type_of_operation).view('i8') for x in args])).view(type_of_result)


def datetime_mean(x: np.ndarray, y: np.ndarray, type_of_operation='<M8[s]'):
    """
    Compute mean vector of two time vectors
    :param x: numpy datetime64 vector
    :param y: numpy datetime64 vector
    :param type_of_operation: numpy type to convert x and y before average to not overflow
    :return: numpy datetime64 vector of type_of_operation
    """
    result = datetime_fun(lambda x2d: np.mean(x2d, 1), np.column_stack((x, y)), type_of_operation=type_of_operation)
    return result


# def datetime_mean(x, y):
#     return np.int64((x.astype('<M8[s]').view('i8') + y.astype('<M8[s]').view('i8') / 2)).view('<M8[s]')


def multiindex_timeindex(df_index):
    """
    Extract DatetimeIndex even it is in MultiIndex
    :param df_index: pandas index
    :return: df_t_index, DatetimeIndex,
        itm - next MultiIndex level if exist else None
    """
    b_MultiIndex = isinstance(df_index, pd.MultiIndex)
    if b_MultiIndex:
        itm = [isinstance(L, pd.DatetimeIndex) for L in df_index.levels].index(True)
        df_t_index = df_index.get_level_values(itm)  # not use df_index.levels[itm] which returns sorted
    else:
        df_t_index = df_index
        itm = None
    return df_t_index, itm


def multiindex_replace(pd_index, new1index, itm):
    """
    replace timeindex_even if_multiindex
    :param pd_index:
    :param new1index: replacement for pandas 1D index in pd_index
    :param itm: index of dimention in pandas MultiIndex which is need to replace by new1index. Use None if not MultiIndex
    :return: modified MultiIndex if itm is not None else new1index
    """
    if not itm is None:
        pd_index.set_levels(new1index, level=itm, verify_integrity=False)
        # pd_index = pd.MultiIndex.from_arrays([[new1index if i == itm else L] for i, L in enumerate(pd_index.values)], names= pd_index.names)
        # pd.MultiIndex([new1index if i == itm else L for i, L in enumerate(pd_index.levels)], labels=pd_index.labels, verify_integrity=False)
        # pd.MultiIndex.from_tuples([ind_new.values])
    else:
        pd_index = new1index
    return pd_index


def timzone_view(t, dt_from_utc=0):
    """
    
    :param t: Pandas Timestamp time
    :param dt_from_utc: pd.Timedelta - timezone offset
    :return: t with applied timezone dt_from_utc
      Assume that if time zone of tz-naive Timestamp is naive then it is UTC
    """
    if dt_from_utc == 0:
        # dt_from_utc = pd.Timedelta(0)
        tzinfo = tzUTC
    elif dt_from_utc == pd.Timedelta(0):
        tzinfo = tzUTC
    else:
        tzinfo = tzoffset(None, pd.to_timedelta(dt_from_utc).total_seconds())  # better pd.datetime.timezone?

    if isinstance(t, pd.DatetimeIndex) or isinstance(t, pd.Timestamp):
        if t.tz is None:
            # think if time zone of tz-naive Timestamp is naive then it is UTC
            t = t.tz_localize('UTC')
        return t.tz_convert(tzinfo)
    else:
        l.error(
            'Bad time format {}: {} - it is not subclass of pd.Timestamp/DatetimeIndex => Converting...'.format(type(t),
                                                                                                                t))
        t = pd.to_datetime(t).tz_localize(tzinfo)
        return t
        # t.to_datetime().replace(tzinfo= tzinfo) + dt_from_utc
    # t.astype(datetime).replace(


# ----------------------------------------------------------------------
def time_corr(date, cfg_in: Mapping[str, Any], b_make_time_inc: Optional[bool] = True, path_save_image='time_corr'):
    """
    :param date: numpy datetime64 or array text in ISO 8601 format
    :param cfg_in: dict with fields:
    - dt_from_utc: correct time by adding this constant
    - fs: sampling frequency
    - b_make_time_inc
    - keep_input_nans: nans in date remains unchenged
    - path: where save images of bad time corrected
    :param b_make_time_inc: check time resolution and increase if needed to avoid duplicates
    :return:
        tim, pandas time series, same size as date input
        b_ok - mask of not decreasing elements
    Note: convert to UTC time if ``date`` in text format, propely formatted for conv.
    todo: use Kalman filter?
    """
    if __debug__:
        l.debug('time_corr (time correction) started')
    if cfg_in.get('dt_from_utc'):
        if isinstance(date[0], str):
            hours_from_utc_f = cfg_in['dt_from_utc'].total_seconds() / 3600
            Hours_from_UTC = int(hours_from_utc_f)
            hours_from_utc_f -= Hours_from_UTC

            if abs(hours_from_utc_f) > 0.0001:
                print('For string data can add only fixed number of hours! Adding', Hours_from_UTC / 3600, 'Hours')
            tim = pd.to_datetime((date.astype(np.object) + '{:+03d}'.format(Hours_from_UTC)).astype('datetime64[ns]'),
                                 utc=True)
        elif isinstance(date, pd.Index):
            tim = date
            tim += cfg_in['dt_from_utc']
            tim = tim.tz_localize('UTC')
            # if Hours_from_UTC==0:
            #
            # else:
            # tim.tz= tzoffset(None, -Hours_from_UTC*3600)   #invert localize
            # tim= tim.tz_localize(None).tz_localize('UTC')  #correct

        else:
            try:
                if isinstance(date, pd.Series):
                    tim = date - np.timedelta64(cfg_in['dt_from_utc'])
                else:
                    tim = pd.to_datetime(date.astype('datetime64[ns]') - np.timedelta64(
                        pd.Timedelta(cfg_in['dt_from_utc'])), utc=True)  # hours=Hours_from_UTC
            except OverflowError:  # still need??
                tim = pd.to_datetime(
                    datetime_fun(np.subtract, tim.values,
                                 np.timedelta64(cfg_in['dt_from_utc'])), type_of_operation='<M8[ms]')
        # tim+= np.timedelta64(pd.Timedelta(hours=hours_from_utc_f)) #?
    else:
        if (not isinstance(date, pd.Series)) and (not isinstance(date, np.datetime64)):
            date = date.astype('datetime64[ns]')
        tim = pd.to_datetime(date, utc=True)  # .tz_localize('UTC')tz_convert(None)
        hours_from_utc_f = 0

    b_ok_in = date.notna()
    n_bad_in = (~b_ok_in).sum()
    if n_bad_in and cfg_in.get('keep_input_nans'):
        tim = tim[b_ok_in]
        try:
            b_ok_in = b_ok_in.values
        except AttributeError:  # numpy.ndarray' object has no attribute 'values'
            pass  # if date is not Timeseries but DatetimeIndex we already have array

    if b_make_time_inc or ((b_make_time_inc is None) and cfg_in.get('b_make_time_inc')) and tim.size > 1:
        # Check time resolution and increase if needed to avoid duplicates
        t = tim.values.view(np.int64)  # 'datetime64[ns]'
        freq, n_same, nDecrease, b_ok = find_sampling_frequency(t, precision=6, b_show=False)

        # # show linearity of time # plt.plot(date)
        # fig, axes = plt.subplots(1, 1, figsize=(18, 12))
        # t = date.values.view(np.int64)
        # t_lin = (t - np.linspace(t[0], t[-1], len(t)))
        # axes.plot(date, / dt64_1s)
        # fig.savefig(os_path.join(cfg_in['dir'], cfg_in['file_stem'] + 'time-time_linear,s' + '.png'))
        # plt.close(fig)

        if nDecrease > 0:
            # Excude elements

            # if True:
            #     # try fast method
            #     b_bad_new = True
            #     k = 10
            #     while np.any(b_bad_new):
            #         k -= 1
            #         if k > 0:
            #             b_bad_new = bSpike1point(t[b_ok], max_spyke=2 * np.int64(dt64_1s / freq))
            #             b_ok[np.flatnonzero(b_ok)[b_bad_new]] = False
            #             print('step {}: {} spykes found, deleted {}'.format(k, np.sum(b_bad_new),
            #                                                                 np.sum(np.logical_not(b_ok))))
            #             pass
            #         else:
            #             break
            # if k > 0:  # success?
            #     t = rep2mean(t, bOk=b_ok)
            #     freq, n_same, nDecrease, b_same_prev = find_sampling_frequency(t, precision=6, b_show=False)
            #     # print(np.flatnonzero(b_bad))
            # else:
            #     t = tim.values.view(np.int64)
            # if nDecrease > 0:  # fast method is not success
            # take time:i
            # l.warning(Fast method is not success)

            # excluding repeated values
            i_different = np.flatnonzero(b_ok)
            # trusting to repeating values, keepeng them to not interp near holes (else use np.zeros):
            b_ok = np.ediff1d(t, to_end=True) == 0
            b_ok[i_different[longest_increasing_subsequence_i(t[i_different])]] = True
            # b_ok= nondecreasing_b(t, )
            # t = t[b_ok]

            t = np.int64(rep2mean(t, bOk=b_ok))

            idel = np.flatnonzero(np.logical_not(b_ok))  # to show what is done
            b_ok = np.ediff1d(t, to_end=True) > 0  # updation for next step

            msg = f'Filtered time: {len(idel)} values interpolated'
            l.warning('decreased time (%d times) is detected! %s', nDecrease, msg)
            # plt can hang.
            plt.figure('Decreasing time corr');
            plt.title(msg)
            plt.plot(idel, tim.iloc[idel], '.m')
            plt.plot(np.arange(t.size), tim, 'r')
            plt.plot(np.flatnonzero(b_ok), pd.to_datetime(t[b_ok], utc=True), color='g', alpha=0.8)
            if 'path' in cfg_in and path_save_image:
                if not PurePath(path_save_image).is_absolute():
                    path_save_image = PurePath(cfg_in['path']).with_name(path_save_image)
                    dir_create_if_need(path_save_image)
                fig_name = path_save_image / '{:%y%m%d_%H%M}-{:%H%M}.png'.format(*tim.iloc[[0, -1]])
                plt.savefig(fig_name)
                l.info(' - figure saved to %s', fig_name)
            try:
                pass
                # plt.show(block=True)
            except RuntimeError as e:
                l.exception('can not show plot in dask?')

        if n_same > 0 and cfg_in.get('fs') and not cfg_in.get('fs_old_method'):
            # This is most simple operation that showld be done usually for CTD
            t = repeated2increased(t, cfg_in['fs'], b_ok if nDecrease else None)
            tim = pd.to_datetime(t, utc=True)
        elif n_same > 0 or nDecrease > 0:
            # Increase time resolution by recalculating all values using constant frequency
            if cfg_in.get('fs_old_method'):
                l.warning('Linearize time interval using povided freq = %fHz (determined: %f)',
                          cfg_in.get('fs_old_method'), freq)
                freq = cfg_in.get('fs_old_method')
            else:  # constant freq = filtered mean
                l.warning('Linearize time interval using freq = %fHz determined', freq)

            # # Check if we can simply use linear increasing values
            # freq2= np.float64(dt64_1s)*t.size/(t[-1]-t[0])
            # freqErr= abs(freq - freq2)/freq
            # if freqErr < 0.1: #?
            #     t = np.linspace(t[0], t[-1], t.size)
            #     #print(str(round(100*freqErr,1)) + '% error in frequency estimation. May be data gaps. Use unstable algoritm to correct')
            #     #t= time_res1s_inc(t, freq)
            # else: # if can not than use special algorithm:
            tim_before = pd.to_datetime(t, utc=True)
            make_linear(t, freq)  # will change t and tim
            tim = pd.to_datetime(t, utc=True)  # not need allways?
            bbad = check_time_diff(tim_before, tim.values, dt_warn=pd.Timedelta(minutes=2),
                                   mesage='Big time diff after corr: difference [min]:')
            if np.any(bbad):
                pass

        dt = np.ediff1d(t, to_begin=1)
        b_ok = dt > 0
        # check all is ok
        # tim.is_unique , len(np.flatnonzero(tim.duplicated()))
        b_decrease = dt < 0  # with set of first element as increasing
        nDecrease = b_decrease.sum()
        if nDecrease > 0:
            l.warning('decreased time remains - {} masked!'.format(nDecrease))
            b_ok &= ~b_decrease

        b_same_prev = np.ediff1d(t, to_begin=1) == 0  # with set of first element as changing
        n_same = b_same_prev.sum()

        if cfg_in.get('keep_input_nans'):
            if n_same > 0:
                l.warning('nonincreased time (%d times) is detected! - interp ', n_same)
        else:
            # prepare to interp all nonincreased (including NaNs)
            if n_bad_in:
                b_same_prev &= ~b_ok_in

            msg = ', '.join(
                f'{fault} time ({n} times)' for (n, fault) in ((n_same, 'nonincreased'), (n_bad_in, 'NaN')) if n > 0
                )
            if msg:
                l.warning('%s is detected! - interp ', msg)

        if n_same > 0 or nDecrease > 0:
            # tim = pd.to_datetime(
            #     rep2mean(t, bOk=np.logical_not(b_same_prev if nDecrease==0 else (b_same_prev | b_decrease))), utc=True)
            b_bad = b_same_prev if nDecrease == 0 else (b_same_prev | b_decrease)
            tim = pd.to_datetime(rep2mean_with_const_freq_ends(t, ~b_bad, freq), utc=True)

    else:
        l.debug('time not nessesary to be sorted')
        b_ok = np.ones(tim.size, np.bool8)

    if n_bad_in and cfg_in.get('keep_input_nans'):
        # place initially bad elements back
        t = np.NaN + np.empty_like(b_ok_in)
        t[b_ok_in] = tim
        tim = pd.to_datetime(t)
        b_ok_in[b_ok_in] = b_ok
        b_ok = b_ok_in

    return tim, b_ok


def pd_period_to_timedelta(period: str) -> pd.Timedelta:
    """
    Converts str to pd.Timedelta
    :param period: str, in format of pandas offset string 'D' (Y, D, 5D, H, ...)
    :return:
    """
    number_and_units = re.search('(^\d*)(.*)', period).groups()
    if not number_and_units[0]:
        number_and_units = (1, number_and_units[1])
    else:
        number_and_units = (int(number_and_units[0]), number_and_units[1])
    return pd.Timedelta(*number_and_units)


def intervals_from_period(
        datetime_range: Optional[np.ndarray] = None,
        date_min: Optional[pd.Timestamp] = None,
        date_max: Optional[pd.Timestamp] = None,
        period: Optional[str] = '999D',
        **kwargs) -> (pd.Timestamp, pd.DatetimeIndex):
    """
    Start times from intervals defined by period, limied by cfg
    :param period: pandas offset string 'D' (Y, D, 5D, H, ...) if None such field must be in cfg_in
    :param datetime_range: list of 2 elements, use something like np.array(['0', '9999'], 'datetime64[s]') for all data. If not provided 'date_min' and 'date_max' will be used
    :param date_min, date_max:
    :return: (t_prev_interval_start, t_intervals_start) (Timestamp, fixed frequency DatetimeIndex)
    """

    # divide datetime_range on intervals of period
    if datetime_range is not None:
        t_prev_interval_start = pd.Timestamp(datetime_range[0])  # (temporarely) end of previous interval
    else:
        t_prev_interval_start = pd.to_datetime(date_min) if date_min else pd.Timestamp(year=2000, month=1, day=1)
        if date_max is not None:
            t_interval_last = pd.to_datetime(date_max)  # last
        else:
            t_interval_last = pd.datetime.now()  # i.e. big value
        datetime_range = [t_prev_interval_start, t_interval_last]

    period_timedelta = pd_period_to_timedelta(period)

    # set the first split on the end of day if interval is bigger than day
    if period_timedelta >= pd.Timedelta(1, 'D'):
        t_1st_split = t_prev_interval_start.normalize()
        if t_1st_split <= t_prev_interval_start:
            t_1st_split += period_timedelta
    else:
        t_1st_split = t_prev_interval_start

    if t_1st_split > datetime_range[-1]:
        t_intervals_start = pd.DatetimeIndex(datetime_range[-1:])
    else:
        t_intervals_start = pd.date_range(
            start=t_1st_split,
            end=max(datetime_range[-1], t_1st_split + period_timedelta),
            freq=period)
        # make last t_interval_start bigger than datetime_range[-1]
        if t_intervals_start[-1] < datetime_range[-1]:
            t_intervals_start = t_intervals_start.append(pd.DatetimeIndex(datetime_range[-1:]))
    return t_prev_interval_start, t_intervals_start


def positiveInd(i, L):
    """
    Positive index
    :param i: int, index
    :param L: int, length of indexing array
    :return: int, index i if i>0 else if i is negative then its positive python equivalent
    """
    ia = np.int64(i)
    return np.where(ia < 0, L - ia, ia)  #


def minInterval(iLims1, iLims2, L):
    """
    Intersect of two ranges
    :param iLims1: range1: min and max indexes
    :param iLims2: range2: min and max indexes
    :param L: int, length of indexing array
    :return: tuple: (max of first iLims elements, min of last iLims elements)
    """

    def maxmin(iL1, iL2):
        return np.transpose([max(iL1[:, 0], iL2[:, 0]), min(iL1[:, -1], iL2[:, -1])])

    return maxmin(positiveInd(iLims1, L), positiveInd(iLims2, L))
