from pathlib import PurePath
from typing import Mapping, Any, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from other_filters import find_sampling_frequency, longest_increasing_subsequence_i, rep2mean, repeated2increased, \
    make_linear, check_time_diff, rep2mean_with_const_freq_ends
from utils2init import dir_create_if_need
from utils_time import l, datetime_fun
#from to_pandas_hdf5.h5_dask_pandas import filter_global_minmax

tim_min_save: pd.Timestamp     # can only decrease in time_corr(), set to pd.Timestamp('now', tz='UTC') before call
tim_max_save: pd.Timestamp     # can only increase in time_corr(), set to pd.Timestamp(0, tz='UTC') before call
def time_corr(date, cfg_in: Mapping[str, Any], b_make_time_inc: Optional[bool] = True, path_save_image='time_corr'):
    """
    :param date: numpy datetime64 or array text in ISO 8601 format
    :param cfg_in: dict with fields:
    - dt_from_utc: correct time by adding this constant
    - fs: sampling frequency
    - b_make_time_inc
    - keep_input_nans: nans in date remains unchanged
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

    if cfg_in.get('date_min'):
        # Skip processing if data out of filtering range
        global tim_min_save, tim_max_save
        tim_min = tim.min(skipna=True)
        tim_max = tim.max(skipna=True)
        # also collect statistics of min&max for messages:
        tim_min_save = min(tim_min_save, tim_min)
        tim_max_save = max(tim_max_save, tim_max)

        if (tim_min > pd.Timestamp(cfg_in['date_max'], tz='UTC') or
            tim_max < pd.Timestamp(cfg_in['date_min'], tz='UTC')):  # tim.iat[0], iat[-1] is less safier
            tim[:]= pd.NA
            return tim, np.ones_like(tim, dtype=bool)   # mark out of range as good values

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
                l.warning('Linearize time interval using medean* freq = %fHz determined', freq)

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