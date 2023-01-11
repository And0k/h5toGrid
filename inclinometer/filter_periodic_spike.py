import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union
from itertools import tee

from dataclasses import dataclass, field
import hydra
from hydra.core.config_store import ConfigStore

import numpy as np
import pandas as pd

from filters import b1spike_up
from to_pandas_hdf5.h5_dask_pandas import i_bursts_starts, h5_append
from to_pandas_hdf5.h5toh5 import h5move_tables, h5_dispenser_and_names_gen

# to save through temporary store (with log for each saving data part - here it is burst):
from to_pandas_hdf5.h5toh5 import h5init, h5remove, h5index_sort
from utils2init import this_prog_basename, standard_error_info, LoggingStyleAdapter

@dataclass
class ConfigType:
    # table name and column name: standard values for my hdf5 store
    device:  str = 'w01'
    col_out: str = 'Pressure'
    cols_order: List[str] = field(default_factory=lambda: ['Pressure', 'Temp', 'Battery'])
    db_path: str = '201202pres_proc_noAvg'

cs_store_name = Path(__file__).stem
cs = ConfigStore.instance()
cs.store(name=cs_store_name, node=ConfigType)  # Registering the Config class with the name 'config'.

# cs, ConfigType = hydra_cfg_store(
#     cs_store_name,
#     {
#         'input': ['in_hdf5__incl_calibr'],  # Load the config ConfigInHdf5_InclCalibr from the config group "input"
#         'out': ['out'],  # Set as MISSING to require the user to specify a value on the command line.
#         'filter': ['filter'],
#         'program': ['program'],
#     },
#     module=sys.modules[__name__]
#     )

cfg_in = {
    'db_path': Path(
        # r'd:\workData\BalticSea\201202_BalticSpit_inclinometer\201202@i3,5,9,10,11,15,19,23,28,30,32,33,w1-6\201202..proc_noAvg.h5'
        # r'd:\workData\BalticSea\201202_BalticSpit\inclinometer\201202.raw.h5'
        ),
    'device': ConfigType.device,
    'table': f'/{ConfigType.device}',
    'col': 'P',
    'b_show': False,  # True

    'min_date': '2020-12-02T10:00',
    'max_date': '2021-03-02T01:15:05',  # 'now' '2021-01-08T12:20',
    }


lf = LoggingStyleAdapter(logging.getLogger(__name__))
pattern_log_dt = '{:%y-%m-%d %H:%M:%S} \u2013 {:%m-%d %H:%M:%S}'

def start_sinch(v, cur, max_shift, std_fraq_noise=0.2):
    """
    Find period index of out of sinch, best sinch correction and for how much periods it helps
    :param v: array which values may be spikes
    :param cur: indices of supposed spike positions
    :param max_shift: shifts to check
    :param std_fraq_noise: reduce probability of adding shifts
    :return: spike positions: cur_out = cur + shifts[ishifts_best[...]]
    """

    shifts = np.empty(max_shift * 2 + 1, dtype=np.int8)
    shifts[1::2] = np.arange(1,  max_shift + 1)
    shifts[ ::2] = np.arange(0, -max_shift - 1, -1)
    # where dv bigger the spike probability is greater
    dv = -v[np.clip(cur + np.transpose([shifts]), a_min=None, a_max=v.size - 1)]  # , a_min = 0 not need because we excluding 1st elements

    # Last term punishes for amount of shifts:
    dv -= (dv.std() * std_fraq_noise * np.abs([shifts]).T)
    ishifts_best = dv.argmax(0)

    cur_out = cur.copy()
    if ishifts_best.any():
        # Filter noise: single values (except edges)
        # start indexes of different shifts (except 1st if it not 0)
        idiff = np.flatnonzero(np.ediff1d(ishifts_best, to_begin=0))  # ... to filter edges too
        if idiff.any():
            # removes changes that in touch then convert to list for faster appending later
            b_ok = np.ediff1d(idiff, to_end=0) != 1
            idiff_single = idiff[~b_ok]
            b_ok[~b_ok] = dv[ishifts_best[idiff_single], idiff_single] >= dv.mean()  # do not remove if diff big
            idiff_list = idiff[b_ok].tolist()

            # add 1st if was not for shift==0 (we not added it before to not filter in case it is single)
            if ishifts_best[0]:
                idiff_list.insert(0, 0)  # <=> idiff_list = [0] + idiff_list

            idiff_list.append(None)  # None effect equal to cur.size or any very big value
            for st, en in pairwise(idiff_list):
                cur_out[st:en] += shifts[ishifts_best[st]]
        else:  # everywhere shift is same
            cur_out += shifts[ishifts_best[0]]
    return cur_out

    # vcur = v[cur]
    # n_helped_max = 0
    # out_param = [0, 0]
    # side_range = np.arange(1, max_shift + 1)
    # shifts = np.column_stack((side_range, -side_range)).flatten()
    # dv_max = 0
    #
    # # where dv bigger the spike probability is greater. Last term punishes for amount of shifts
    # dv = vcur - v[cur + np.transpose([shifts])] - dv_noise*np.abs([shifts]).T
    # b_ok = dv > 0
    #
    # # First possible spike for each shift (ish0)
    # ish, jsh = np.nonzero(b_ok)
    # if ish.size:
    #     # where starts indexes for each shift in jsh
    #     jsh_st = np.flatnonzero(np.ediff1d(ish, to_begin=1).astype(np.bool8))
    #     # jsh_en = jsh_st[1:]  # last jsh index for each shift
    #     ish0 = jsh[jsh_st]
    #     ishifts = ish[jsh_st]
    #     dv_max_all = np.zeros_like(cur)
    #     for i, ish_st in np.vstack((ishifts, ish0)).T:
    #
    #         # corrected dv_sh_mean (dv_sh_mean = dv[i, st:en].mean()) for supposed en
    #         ish_en = (lambda no: np.flatnonzero(no)[0] if no.any() else no.size)(~b_ok[i, ish_st:])
    #         # if slice is too small may be ignore:
    #         if ish_en < 1:
    #             continue
    #             # ish_en = 1
    #             # dv_sh_mean = dv[i, ish_st]
    #
    #         ish_en += ish_st
    #         dv_sh_mean = dv[i, ish_st:ish_en].mean()
    #         # analise each slice save separate:
    #         dv_max_all_slice = dv_max_all[ish_st:ish_en]
    #         dv_max_all_slice[dv_max_all_slice < dv_max] = dv_sh_mean
    #
    #         if dv_sh_mean > dv_max:
    #             dv_max = dv_sh_mean
    #             ishift = i
    #             # number of consequent good spike values from 1st better value found
    # if dv_max:
    #     ish_st = ish0[ishifts == ishift].item()
    #     #n_shifted = (lambda no: np.flatnonzero(no)[0] if no.any() else no.size)(~b_ok[ishift, ish_st:])
    #
    #     return shifts[ishift], ish_st
    # else:  # nowhere shift needed
    #     return 0, cur.size


    # for shift in shifts:
    #     # add shift to phase, check if it a better spike value:
    #     bo_add = vcur > v[cur + shift]
    #     if bo_add.any():
    #         # spike positions from 1st better value position
    #         icur_st = np.flatnonzero(bo_add)[0]
    #         # number of consequent better spike values from 1st better value found
    #         n_helped = (lambda no: np.flatnonzero(no)[0] if no.any() else no.size)(~bo_add[icur_st:])
    #         if n_helped > n_helped_max:
    #             n_helped_max = n_helped
    #             out_param = [shift, icur_st]
    # return out_param


def filter_periodic_spike(
        ser, dp_down_min=0.002,
        di_period_min=13, di_period_max=16, di_period=None,
        n_bad_start=6, n_bad_start_possible=20,
        ax=None
        ):
    """

    :param ser: pandas series
    :param dp_down_min: abs. value of down spike to find maximum of real spikes and minimum not spikes to synchro filter
    :param di_period_min: min period
    :param di_period_max: di_period_min <= period < di_period_max
    :param di_period: if None then it will be found for each burst separately. If you know that it is same for entire
     sequence you can calc. and provide it: (lambda di: np.mean(di[(di_period_min <= di) & (di <= di_period_max)]))(
        np.diff(np.flatnonzero(b1spike_up(-ser.values, max_spike=dp_down_min))))
    :param ax: None to not plot or matplotlib axis to plot results
    :return:
    """
    b = b1spike_up(-ser.values, max_spike=dp_down_min)
    i_spike = np.flatnonzero(b)
    max_shift = 5

    # Filter small intervals
    di = np.diff(i_spike)
    ok_di = di >= di_period_min
    # point is good only if diff to prev and after is ok (not detecting here cur point or next is really bad)
    b = (lambda small_d: np.append(small_d, True) & np.append(True, small_d))(ok_di)
    try:
        i = i_spike[b]
    except IndexError as e:
        return ser.index
    # save not used values in our filter to filter them later (if our filter will not removes them):
    i_spike = i_spike[~b]

    # More reliable points: excluding big consecutive intervals too
    di = np.diff(i)
    ok_di = di < di_period_max
    # keep both points between which difference is good - think they are reliable (so we can reference them further):
    i_f = i[(lambda small_d: np.append(small_d, True) | np.append(True, small_d))(ok_di)]

    # intervals starts where some periods was not found reliable
    di_f = np.diff(i_f)
    bi_fill = di_f > di_period_max

    # mean spike freq:
    if di_period is None:
        di_period = np.mean(di_f[~bi_fill])
        if di_period > di_period_max:
            lf.warning('mean spike freq in interval {:%y-%m-%d %H:%M:%S%Z} \u2013 {:%m-%d %H:%M:%S%Z} bigger than limit {} > {}! Using previous value {}',
                       *ser.index[[0, -1]], di_period, di_period_max, filter_periodic_spike.di_period)
            di_period = filter_periodic_spike.di_period
        elif di_period < di_period_min:
            lf.warning('mean spike freq in interval {:%y-%m-%d %H:%M:%S%Z} \u2013 {:%m-%d %H:%M:%S%Z} less than limit {} < {}! Using previous value {}',
                       *ser.index[[0, -1]], di_period, di_period_max, filter_periodic_spike.di_period)
            di_period = filter_periodic_spike.di_period
        elif np.isnan(di_period):
            di_period = filter_periodic_spike.di_period
        else:
            filter_periodic_spike.di_period = di_period
    # Save indexes of reliable spike data
    i_f_fill = []

    ## save 1st `n_bad_start` values of burst
    if n_bad_start_possible:  # filter 1st lower then max values
        ibad_last = n_bad_start + np.argmax(ser.values[n_bad_start:n_bad_start_possible])
        ser.values[n_bad_start + ibad_last - 1]
    else:
        ibad_last = n_bad_start
    i_f_fill.append(np.arange(0, ibad_last))

    ## find 1st possible spike after ibad_last by substracting hole number of periods back to start from 1st reliable:
    bi_fill &= (i_f > ibad_last)[:-1]
    ii_fill = np.flatnonzero(bi_fill)
    if ii_fill.size:
        i_f0 = i_f[ii_fill[0]]
        n_periods_to_st = np.int32((i_f0 - ibad_last)/di_period)
        # add it to indexes of reliable spikes to fill from by start_sinch()
        if n_periods_to_st:
            i_f = np.append(np.int32(i_f0 - n_periods_to_st * di_period), i_f)
            ii_fill = np.append(0, ii_fill+1)
        #i_f_fill.append(np.int32(np.arange(i_f[0] - di_period, n_bad_start, -di_period))[::-1])

        ## fill and save perionds separately in long intervals between and after last reliable spikes:
        for st, en in zip(  # appending interval from last to end
                i_f[np.append(ii_fill, -1)] + di_period,
                np.append(i_f[ii_fill + 1] - di_period + max_shift, len(ser))
                ):
            try:
                cur = np.int32(np.arange(st, en, di_period))
            except ValueError as e:
                pass
            if not cur.size:
                continue
            #print(f'{cur[0]} ({ser.index[cur[0]]}): ~{cur.size} spikes shifting')
            cur_sinch = start_sinch(ser.values, cur, max_shift=max_shift)
            i_f_fill.append(cur_sinch)

            # # cumulative sinch error
            # # last_diff = di_period - (en - cur[-1])
            #
            # # correct if we out of sinch
            # while True:
            #     shift, icur_en = start_sinch(ser.values, cur, max_shift=max_shift)
            #     print(f'{cur[0]} ({ser.index[cur[0]]}): {icur_en} spikes shifted on {shift}')
            #     if not shift:
            #         i_f_fill.append(cur)
            #         break
            #
            #     if icur_en:
            #         i_f_fill.append(cur[:icur_en])
            #     else:
            #         icur_en = 1
            #
            #     cur = np.int32(np.arange(cur[icur_en] + shift, en, di_period))
            #     if not cur.size:
            #         break
            #
            #
            # #  = i[cur[:icur_add_st]] > i[cur_add]
            # # if bo_add.any():
            # #     icur_add_st = np.flatnonzero(bo_add)[0]
            # # else:
            # #     i_f_fill.append(cur[:icur_add_st])
            # #     i_f_fill.append(cur_add)
            #
            #
            # # bo_sub = i[cur] > i[cur-1]
            # # if bo_sub.any():
            # #     icur_sub_st = np.flatnonzero(bo_sub)[0]
            # #     cur_sub = np.int32(np.arange(cur[icur_sub_st]-1, en, di_period))



    # ## fill from last spike to end:
    # i_f_fill.append(np.int32(np.arange(i_f[-1] + di_period, b.size, di_period)))
    i_f_fill.append(i_f)



    # We have indexes found by our experimental periodic spike filter, but it is not ideal so
    # we will also remove spikes we found before (not all used with this filter):
    i_f_fill.append(i_spike)


    index_to_drop = ser.index[np.hstack(i_f_fill)]

    if ax:
        ax.clear()
        ser.loc[index_to_drop].plot(color='r', marker='.', label='bad: all', axes=ax)
        ser.iloc[i_f].plot(color='k', marker='*', label='bad: reliable', axes=ax)

    # Remove bad data
    ser.drop(index_to_drop, inplace=True)

    if ax:
        ser.plot(color='g', label='remains', axes=ax)
        ax.legend(prop={'size': 10}, loc='upper right')
        ax.grid(True, alpha=.95, linestyle='-')
        ax.set_xlim(ser.index[[0, -1]])
        ax.set_ylim(min(ser.values), max(ser.values))

    return ser.index


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


@hydra.main(config_name=cs_store_name, config_path="cfg")  # adds config store cs_store_name data/structure to param:config
def main(config: ConfigType) -> None:  #
    raw = 'raw' in cfg_in['db_path'].stem  # else "proc_noAvg" is in
    with pd.HDFStore(cfg_in['db_path'], mode='r') as store:
        df = store[cfg_in['table']][cfg_in['min_date']:cfg_in['max_date']]
        if raw:
            k = store.get_node(f"{cfg_in['table']}/coef")[cfg_in['col']].read()

    n_rows_before = df.shape[0]
    lf.info('Loaded data {0[0]} - {0[1]}: {1} rows. Filtering {2[col_out]}...',
            df.index[[0, -1]], n_rows_before, config
            )

    # too many messages if working on bursts  # todo: write through buffer
    logger = logging.getLogger('to_pandas_hdf5.h5_dask_pandas')
    logger.setLevel(logging.ERROR)

    # print(f"Loaded data {df.index[0]} - {df.index[-1]}: {n_rows_before} rows. Filtering {cfg_in['col']}...")
    p_name = config['col_out']
    if raw:
        df[p_name] = np.polyval(k, df[cfg_in['col']])

        # Battery compensation
        kBat = [1.7314032932363, -11.9301097967443]
        df[p_name] -= np.polyval(kBat, df['Battery'])
    MIN_P = 6  # P filtered below: to not delete spikes that may be used to find other spikes using ~constant period

    if config['cols_order']:
        df = df.loc[:, config['cols_order']]
    else:
        df.drop(cfg_in['col'], axis='columns', inplace=True)

    i_burst, mean_burst_size, max_hole = i_bursts_starts(df.index)

    i_col = df.columns.get_loc(p_name)

    if cfg_in['b_show']:
        from matplotlib import pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.grid(True, alpha=.85, color='white', axis='y', linestyle='-')
        fig.subplots_adjust(top=.89)

        fig.show()
    else:
        ax = None

    # 'db': store
    cfg_out = {
        'table': cfg_in['table'],
        'table_log': f"{cfg_in['table']}/logFiles", 'log': {},
        'db_path': Path(config['db_path']) if 'db_path' in config else (
           cfg_in['db_path'].with_name(f"{cfg_in['db_path'].stem}_filt_s.h5")
        )}

    def h5_names_gen(cfg_in, cfg_out: Mapping[str, Any], **kwargs) -> Iterator[None]:
        #cfg_out['log']['fileName'] = pname.name[-cfg_out['logfield_fileName_len']:-4]
        cfg_out['log']['fileChangeTime'] = datetime.fromtimestamp(cfg_in['db_path'].stat().st_mtime)
        yield None

    h5init(cfg_in, cfg_out)  # cfg_in for full path if cfg_out['db_path'] only name
    n_rows_after = 0
    #with pd.HDFStore(out_path) as store:  #, mode='w'
    for _, _ in h5_dispenser_and_names_gen(cfg_in, cfg_out, h5_names_gen):  # handles temporary db for h5_append()
        try:
            if h5remove(cfg_out['db'], cfg_in['table']):
                lf.info('previous table removed')
        except Exception as e:  # no such table?
            pass

        for st, en in pairwise(i_burst):
            cfg_out['log']['fileName'] = str(st)
            sl = slice(st, en)
            ind_ok = filter_periodic_spike(df.iloc[sl, i_col], ax=ax)

            # Filtering
            bad_p = df.loc[ind_ok, p_name] < MIN_P
            n_bad = bad_p.sum()
            if n_bad:
                lf.info('filtering {} > {}: deleting {} values in frame {}',
                        p_name, MIN_P, n_bad, pattern_log_dt.format(*df.index[[0,-1]]))
                ind_ok = ind_ok[~bad_p]
                if not ind_ok.size:
                    continue
                # df.loc[bad_p, p_name] = np.NaN

            # save result

            h5_append(cfg_out, df.loc[ind_ok], cfg_out['log'])
            n_rows_after += ind_ok.size

    # Temporary db to compressed db with pandas index
    if n_rows_after:  # check needed because ``ptprepack`` in h5index_sort() not closes hdf5 source if it not finds data
        failed_storages = h5move_tables(cfg_out)
        print('Ok.', end=' ')
        h5index_sort(cfg_out, out_storage_name=f"{cfg_out['db_path'].stem}-resorted.h5", in_storages=failed_storages)

    lf.info(f'Removed {n_rows_before - n_rows_after} rows. Saved {n_rows_after} rows to {cfg_out["db_path"]}...')


if __name__ == '__main__':
    main()









def _test_filter_periodic_spike():
    """
    not implemented
    :return:
    """
    data = np.zeros(60)
    di_period = 5
    i_spike = np.arange(1, di_period, data.size)
    i_spike[3:5] += 1
    i_spike[6:10] -= 1
    data[i_spike] = -1
    ser = pd.Series(data)

    out = filter_periodic_spike(
        ser, dp_down_min=0.5, di_period_min=di_period-1, di_period_max=di_period+1, n_bad_start=0
        )




