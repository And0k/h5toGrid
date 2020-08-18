from pathlib import PurePath
from typing import Mapping, Any, Optional, Union

import numpy as np
import pandas as pd

import matplotlib
try:
    matplotlib.use(
        'Qt5Agg')  # must be before importing plt (rases error after although documentation sed no effect)
    matplotlib.interactive(False)
except ImportError:
    print('matplotlib can not import Qt5Agg backend - may be errors in potting')
    pass
from matplotlib import pyplot as plt

from other_filters import find_sampling_frequency, longest_increasing_subsequence_i, rep2mean, repeated2increased, \
    make_linear, check_time_diff, rep2mean_with_const_freq_ends
from utils2init import dir_create_if_need
from utils_time import l, datetime_fun
#from to_pandas_hdf5.h5_dask_pandas import filter_global_minmax



tim_min_save: pd.Timestamp     # can only decrease in time_corr(), set to pd.Timestamp('now', tz='UTC') before call
tim_max_save: pd.Timestamp     # can only increase in time_corr(), set to pd.Timestamp(0, tz='UTC') before call
def time_corr(date: Union[pd.Series, pd.Index, np.ndarray],
              cfg_in: Mapping[str, Any],
              b_make_time_inc: Optional[bool] = True,
              path_save_image='time_corr'):
    """
    :param date: numpy np.ndarray elements may be datetime64 or text in ISO 8601 format
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
    if not date.size:
        return pd.DatetimeIndex([]), np.bool_([])
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
                    tim = pd.to_datetime(date - np.timedelta64(cfg_in['dt_from_utc']), utc=True)

                else:
                    tim = pd.to_datetime(date.astype('datetime64[ns]') - np.timedelta64(
                        pd.Timedelta(cfg_in['dt_from_utc'])), utc=True)  # hours=Hours_from_UTC
            except OverflowError:  # still need??
                tim = pd.to_datetime(
                    datetime_fun(np.subtract, tim.values,
                                 np.timedelta64(cfg_in['dt_from_utc'])), type_of_operation='<M8[ms]', utc=True)
            # tim+= np.timedelta64(pd.Timedelta(hours=hours_from_utc_f)) #?
    else:
        if (not isinstance(date, pd.Series)) and (not isinstance(date, np.datetime64)):
            date = date.astype('datetime64[ns]')
        tim = pd.to_datetime(date, utc=True)  # .tz_localize('UTC')tz_convert(None)
        #hours_from_utc_f = 0


    if cfg_in.get('date_min'):
        # Skip processing if data out of filtering range
        global tim_min_save, tim_max_save
        tim_min = tim.min(skipna=True)
        tim_max = tim.max(skipna=True)
        # also collect statistics of min&max for messages:
        tim_min_save = min(tim_min_save, tim_min)
        tim_max_save = max(tim_max_save, tim_max)

        # set time to special values keeping it sorted for dask and mark out of range as good values
        if tim_max < pd.Timestamp(cfg_in['date_min'], tz='UTC'):
            tim[:] = pd.Timestamp(cfg_in['date_min'], tz='UTC') - np.timedelta64(1, 'ns')  # pd.NaT                      # ns-resolution maximum year
            return tim, np.ones_like(tim, dtype=bool)
        elif cfg_in.get('date_max') and tim_min > pd.Timestamp(cfg_in['date_max'], tz='UTC'):
            tim[:] = pd.Timestamp(cfg_in['date_max'], tz='UTC') + np.timedelta64(1, 'ns')  # pd.Timestamp('2262-01-01')  # ns-resolution maximum year
            return tim, np.ones_like(tim, dtype=bool)
        else:
            it_se = np.flatnonzero(
                (tim >= pd.Timestamp(cfg_in['date_min'], tz='UTC')) &
                (tim <= pd.Timestamp(cfg_in.get('date_max'), tz='UTC')) )[[0,-1]]
            it_se[1] += 1
            tim = tim[slice(*it_se)]
            # tim.clip(lower=pd.Timestamp(cfg_in['date_min'], tz='UTC') - np.timedelta64(1, 'ns'),
            #          upper=pd.Timestamp(cfg_in['date_max'], tz='UTC') + np.timedelta64(1, 'ns'),
            #          inplace=True)


    b_ok_in = tim.notna()
    n_bad_in = b_ok_in.size - b_ok_in.sum()
    if n_bad_in:
        if cfg_in.get('keep_input_nans'):
            tim = tim[b_ok_in]
        try:
            b_ok_in = b_ok_in.to_numpy()
        except AttributeError:
            pass  # we already have numpy array


    t = tim.to_numpy(np.int64)
    if b_make_time_inc or ((b_make_time_inc is None) and cfg_in.get('b_make_time_inc')) and tim.size > 1:
        # Check time resolution and increase if needed to avoid duplicates
        if n_bad_in and not cfg_in.get('keep_input_nans'):
            t = np.int64(rep2mean(t, bOk=b_ok_in))
        freq, n_same, n_decrease, b_ok = find_sampling_frequency(t, precision=6, b_show=False)
        if freq:
            cfg_in['fs_last'] = freq  # fallback freq to get value for next files on fail
        elif cfg_in['fs_last']:
            l.warning('Using fallback (last) sampling frequency fs = %s', cfg_in['fs_last'])
            freq = cfg_in['fs_last']
        elif cfg_in.get('fs'):
            l.warning('Ready to use specified sampling frequency fs = %s', cfg_in['fs'])
            freq = cfg_in['fs']
        elif cfg_in.get('fs_old_method'):
            l.warning('Ready to use specified sampling frequency fs_old_method = %s', cfg_in['fs_old_method'])
            freq = cfg_in['fs_old_method']
        else:
            l.warning('Ready to set sampling frequency to default value: fs = 1Hz')
            freq = 1

        # # show linearity of time # plt.plot(date)
        # fig, axes = plt.subplots(1, 1, figsize=(18, 12))
        # t = date.values.view(np.int64)
        # t_lin = (t - np.linspace(t[0], t[-1], len(t)))
        # axes.plot(date, / dt64_1s)
        # fig.savefig(os_path.join(cfg_in['dir'], cfg_in['file_stem'] + 'time-time_linear,s' + '.png'))
        # plt.close(fig)

        if n_decrease > 0:
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
            #     freq, n_same, n_decrease, b_same_prev = find_sampling_frequency(t, precision=6, b_show=False)
            #     # print(np.flatnonzero(b_bad))
            # else:
            #     t = tim.values.view(np.int64)
            # if n_decrease > 0:  # fast method is not success
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
            b_ok = np.ediff1d(t, to_end=True) > 0  # adaption for next step

            msg = f'Filtered time: {len(idel)} values interpolated'
            l.warning('decreased time (%d times) is detected! %s', n_decrease, msg)
            # plt can hang.
            plt.figure('Decreasing time corr')
            plt.title(msg)
            plt.plot(idel, tim.iloc[idel], '.m')
            plt.plot(np.arange(t.size), tim, 'r')
            plt.plot(np.flatnonzero(b_ok), pd.to_datetime(t[b_ok], utc=True), color='g', alpha=0.8)
            if 'path' in cfg_in and path_save_image:
                if not PurePath(path_save_image).is_absolute():
                    path_save_image = PurePath(cfg_in['path']).with_name(path_save_image)
                    dir_create_if_need(path_save_image)
                fig_name = path_save_image / '{:%y%m%d_%H%M}-{:%H%M}.png'.format(*(tim_min, tim_max) if cfg_in.get('date_min') else tim.iloc[[0, -1]])
                plt.savefig(fig_name)
                l.info(' - figure saved to %s', fig_name)
            try:
                pass
                # plt.show(block=True)
            except RuntimeError as e:
                l.exception('can not show plot in dask?')

        if n_same > 0 and cfg_in.get('fs') and not cfg_in.get('fs_old_method'):
            # This is most simple operation that should be done usually for CTD
            t = repeated2increased(t, cfg_in['fs'], b_ok if n_decrease else None)
            tim = pd.to_datetime(t, utc=True)
        elif n_same > 0 or n_decrease > 0:
            # Increase time resolution by recalculating all values using constant frequency
            if cfg_in.get('fs_old_method'):
                l.warning('Linearize time interval using povided freq = %fHz (determined: %f)',
                          cfg_in.get('fs_old_method'), freq)
                freq = cfg_in.get('fs_old_method')
            else:  # constant freq = filtered mean
                l.warning('Linearize time interval using medean* freq = %fHz determined', freq)

            if freq <= 1:
                l.warning('Typically data resolution is sufficient for this frequency. Not linearizing')
            else:
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
                bbad = check_time_diff(tim_before, t.view('M8[ns]'), dt_warn=pd.Timedelta(minutes=2),
                                       mesage='Big time diff after corr: difference [min]:')
                if np.any(bbad):
                    pass

        dt = np.ediff1d(t, to_begin=1)
        b_ok = dt > 0
        # check all is ok
        # tim.is_unique , len(np.flatnonzero(tim.duplicated()))
        b_decrease = dt < 0  # with set of first element as increasing
        n_decrease = b_decrease.sum()
        if n_decrease > 0:
            l.warning(
                'Decreased time remained (%d) are masked!%s%s',
                n_decrease,
                '\n'.join('->'.join('{:%y.%m.%d %H:%M:%S.%f%z}'.format(_) for _ in tim[se].to_numpy()) for se in
                         np.flatnonzero(b_decrease)[:3, None] + np.int32([-1, 0])),
                '...' if n_decrease > 3 else ''
                )

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

        if n_same > 0 or n_decrease > 0:
            # rep2mean(t, bOk=np.logical_not(b_same_prev if n_decrease==0 else (b_same_prev | b_decrease)))
            b_bad = b_same_prev if n_decrease == 0 else (b_same_prev | b_decrease)
            t = rep2mean_with_const_freq_ends(t, ~b_bad, freq)

    else:
        l.debug('time not nessesary to be sorted')
        b_ok = np.ones(tim.size, np.bool8)
    # make initial shape: paste back NaNs
    if n_bad_in and cfg_in.get('keep_input_nans'):
        # place initially bad elements back
        t, t_in = (np.NaN + np.empty_like(b_ok_in)), t
        t[b_ok_in] = t_in
        b_ok_in[b_ok_in] = b_ok
        b_ok = b_ok_in
    # make initial shape: pad with constants of config. limits where data was removed because input is beyond this limits
    if cfg_in.get('date_min') and np.any(it_se != np.int64([0, date.size])):
        pad_width = (it_se[0], date.size - it_se[1])
        t = np.pad(t, pad_width, constant_values=np.array((cfg_in['date_min'], cfg_in['date_max']), 'M8[ns]'))
        b_ok = np.pad(b_ok, pad_width, constant_values=True)

    return pd.to_datetime(t, utc=True), b_ok