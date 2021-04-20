#!/usr/bin/env python
# coding:utf-8
"""
  Author:  Andrey Korzh <ao.korzh@gmail.com>
  Purpose: pandas time convert utils
  Created: 26.02.2016
"""
import logging
import re
from typing import Optional
from datetime import datetime
import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from utils2init import LoggingStyleAdapter

if __debug__:
    # datetime converter for a matplotlib plotting method
    from pandas.plotting import register_matplotlib_converters

    register_matplotlib_converters()
# from future.moves.itertools import zip_longest
# from builtins import input
# from debug import __debug___print
# from  pandas.tseries.offsets import DateOffset
from dateutil.tz import tzoffset
# my:

lf = LoggingStyleAdapter(logging.getLogger(__name__))

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
        lf.error(
            'Bad time format {}: {} - it is not subclass of pd.Timestamp/DatetimeIndex => Converting...', type(t), t)
        t = pd.to_datetime(t).tz_localize(tzinfo)
        return t
        # t.to_datetime().replace(tzinfo= tzinfo) + dt_from_utc
    # t.astype(datetime).replace(


# ----------------------------------------------------------------------
def pd_period_to_timedelta(period: str) -> pd.Timedelta:
    """
    Converts str to pd.Timedelta. May be better to use pd.Timedelta(*to_offset(period))
    :param period: str, in format of pandas offset string 'D' (Y, D, 5D, H, ...)
    :return:
    """
    number_and_units = re.search(r'(^\d*)(.*)', period).groups()
    if not number_and_units[0]:
        number_and_units = (1, number_and_units[1])
    else:
        number_and_units = (int(number_and_units[0]), number_and_units[1])
    return pd.Timedelta(*number_and_units)


def intervals_from_period(
        datetime_range: Optional[np.ndarray] = None,
        min_date: Optional[pd.Timestamp] = None,
        max_date: Optional[pd.Timestamp] = None,
        period: Optional[str] = '999D',
        **kwargs) -> (pd.Timestamp, pd.DatetimeIndex):
    """
    Divide datetime_range on intervals of period, normalizes starts[1:] if period>1D and returns them in tuple's 2nd element
    :param period: pandas offset string 'D' (Y, D, 5D, H, ...) if None such field must be in cfg_in
    :param datetime_range: list of 2 elements, use something like np.array(['0', '9999'], 'datetime64[s]') for all data.
    If not provided 'min_date' and 'max_date' will be used
    :param min_date, max_date: used if datetime_range is None. If neither provided then use range from 2000/01/01 to now
    :return (start, ends): (Timestamp, fixed frequency DatetimeIndex)
    """

    # Set _datetime_range_ if need and its _start_
    if datetime_range is not None:
        start = pd.Timestamp(datetime_range[0])  # (temporarely) end of previous interval
    else:
        start = pd.to_datetime(min_date) if min_date else pd.Timestamp(year=2000, month=1, day=1)
        if max_date is not None:
            t_interval_last = pd.to_datetime(max_date)  # last
        else:
            t_interval_last = datetime.now()  # i.e. big value
        datetime_range = [start, t_interval_last]

    if period:
        period_timedelta = to_offset(period)  # pd_period_to_timedelta(

        # Set next start on the end of day if interval is bigger than day
        if period_timedelta >= pd.Timedelta(1, 'D'):
            start_next = start.normalize()
            if start_next <= start:
                start_next += period_timedelta
        else:
            start_next = start

        if start_next > datetime_range[-1]:
            ends = pd.DatetimeIndex(datetime_range[-1:])
        else:
            ends = pd.date_range(
                start=start_next,
                end=max(datetime_range[-1], start_next + period_timedelta),
                freq=period)
            # make last start bigger than datetime_range[-1]
            if ends[-1] < datetime_range[-1]:
                ends = ends.append(pd.DatetimeIndex(datetime_range[-1:]))
    else:
        ends = pd.DatetimeIndex(datetime_range[-1:])

    return start, ends


def positiveInd(i: int, l: int) -> int:
    """
    Positive index
    :param i: index
    :param l: length of indexing array
    :return: index i if i>0 else if i is negative then its positive python equivalent
    """
    ia = np.int64(i)
    return np.where(ia < 0, l - ia, ia)


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



# str_time_short= '{:%d %H:%M}'.format(r.Index.to_datetime())
# timeUTC= r.Index.tz_convert(None).to_datetime()