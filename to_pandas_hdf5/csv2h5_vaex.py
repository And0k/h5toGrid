#!/usr/bin/env python3
# coding:utf-8
"""
  Author:  Andrey Korzh <ao.korzh@gmail.com>
  Purpose: Convert (multiple) csv and alike text files to pandas hdf5 store with
           addition of log table
  Created: 26.02.2016
  Modified: 20.12.2019
"""
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple, Union

import numpy as np
import pandas as pd


def read_csv(nameFull: Sequence[Union[str, Path]],
             cfg_in: Mapping[str, Any]) -> Tuple[Union[pd.DataFrame, dd.DataFrame], dd.Series]:
    """
    Reads csv in dask DataFrame
    Calls cfg_in['fun_proc_loaded'] (if specified)
    Calls time_corr: corrects/checks Time (with arguments defined in cfg_in fields)
    Sets Time as index
    :param nameFull: list of file names
    :param cfg_in: contains fields for arguments of dask.read_csv correspondence:

        names=cfg_in['cols'][cfg_in['cols_load']]
        usecols=cfg_in['cols_load']
        error_bad_lines=cfg_in['b_raise_on_err']
        comment=cfg_in['comments']

        Other arguments corresponds to fields with same name:
        dtype=cfg_in['dtype']
        delimiter=cfg_in['delimiter']
        converters=cfg_in['converters']
        skiprows=cfg_in['skiprows']
        blocksize=cfg_in['blocksize']

        Also cfg_in has filds:
            dtype_out: numpy.dtype, which "names" field used to detrmine output columns
            fun_proc_loaded: None or Callable[
            [Union[pd.DataFrame, np.array], Mapping[str, Any], Optional[Mapping[str, Any]]],
             Union[pd.DataFrame, pd.DatetimeIndex]]
            If it returns pd.DataFrame then it also must has attribute:
                meta_out: Callable[[np.dtype, Iterable[str], Mapping[str, dtype]], Dict[str, np.dtype]]

            See also time_corr() for used fields



    :return: tuple (a, b_ok) where
        a:      dask dataframe with time index and only columns listed in cfg_in['dtype_out'].names
        b_ok:   time correction reszult bulean array
    """
    try:
        try:
            # raise ValueError('Temporary')
            ddf = dd.read_csv(
                nameFull, dtype=cfg_in['dtype_raw'], names=cfg_in['cols'],
                delimiter=cfg_in['delimiter'], skipinitialspace=True, usecols=cfg_in['dtype'].names,
                # cfg_in['cols_load'],
                converters=cfg_in['converters'], skiprows=cfg_in['skiprows'],
                error_bad_lines=cfg_in['b_raise_on_err'], comment=cfg_in['comments'],
                header=None, blocksize=cfg_in['blocksize'])  # not infer

            # , engine='python' - may help load bad file

            # index_col=False  # force pandas to _not_ use the first column as the index (row names) - no in dask
            # names=None, squeeze=False, prefix=None, mangle_dupe_cols=True,
            # engine=None, true_values=None, false_values=None, skipinitialspace=False,
            #     nrows=None, na_values=None, keep_default_na=True, na_filter=True, verbose=False,
            #     skip_blank_lines=True, parse_dates=False, infer_datetime_format=False, keep_date_col=False,
            #     date_parser=None, dayfirst=False, iterator=False, chunksize=None, compression='infer',
            #     thousands=None, decimal=b'.', lineterminator=None, quotechar='"', quoting=0,
            #     escapechar=None, encoding=None, dialect=None, tupleize_cols=None,
            #      warn_bad_lines=True, skipfooter=0, skip_footer=0, doublequote=True,
            #     delim_whitespace=False, as_recarray=None, compact_ints=None, use_unsigned=None,
            #     low_memory=True, buffer_lines=None, memory_map=False, float_precision=None)
        except ValueError as e:
            l.error('dask lib can not load data {}: {}. Trying pandas lib...'.format(e.__class__, '\n==> '.join(
                [m for m in e.args if isinstance(m, str)])))
            for i, nf in enumerate(nameFull):
                df = pd.read_csv(
                    nf, dtype=cfg_in['dtype_raw'], names=cfg_in['cols'], usecols=cfg_in['dtype'].names,
                    # cfg_in['cols_load'],
                    delimiter=cfg_in['delimiter'], skipinitialspace=True, index_col=False,
                    converters=cfg_in['converters'], skiprows=cfg_in['skiprows'],
                    error_bad_lines=cfg_in['b_raise_on_err'], comment=cfg_in['comments'],
                    header=None)
                if i > 0:
                    raise NotImplementedError('list of files => need concatenate data')
            ddf = dd.from_pandas(df, chunksize=cfg_in['blocksize'])  #
    except Exception as e:  # for example NotImplementedError if bad file
        msg = '{}: {} - Bad file. skip!\n'.format(e.__class__, '\n==> '.join([
            m for m in e.args if isinstance(m, str)]))
        ddf = None
        if cfg_in['b_raise_on_err']:
            l.error(msg + '%s\n Try set [in].b_raise_on_err= False\n', e)
            raise (e)
        else:
            l.exception(msg)
    if __debug__:
        l.debug('read_csv initialised')
    if ddf is None:
        return None, None

    meta_time = pd.Series([], name='Time', dtype='M8[ns]')  # np.dtype('datetime64[ns]')
    meta_time_index = pd.DatetimeIndex([], dtype='datetime64[ns]', name='Time')
    meta_df_with_time_col = cfg_in['cols_load']

    # Process ddf and get date in ISO string or numpy standard format
    cfg_in['file_stem'] = Path(nameFull[0]).stem  # may be need in func below to extract date
    try:
        date_delayed = None
        try:
            if not getattr(cfg_in['fun_proc_loaded'], 'meta_out', None) is None:
                # fun_proc_loaded will return modified data. Go to catch it
                # todo: find better condition
                raise TypeError

            # ddf_len = len(ddf)
            # counts_divisions = list(range(1, int(ddf_len / cfg_in.get('decimate_rate', 1)), cfg_in['blocksize']))
            # counts_divisions.append(ddf_len)
            #
            # date_delayed = delayed(cfg_in['fun_proc_loaded'], nout=1)(ddf, cfg_in)
            # date = dd.from_delayed(date_delayed, meta=meta_time_index, divisions=ddf.index.divisions)
            # date = dd.from_dask_array(date.values, index=ddf.index)

            date = ddf.map_partitions(lambda *args, **kwargs: pd.Series(
                cfg_in['fun_proc_loaded'](*args, **kwargs)), cfg_in, meta=meta_time)  # meta_time_index
            # date = date.to_series()
        except (TypeError, Exception) as e:
            # fun_proc_loaded retuns tuple (date, a)
            changing_size = False  # ? True  # ?
            if changing_size:
                date_delayed, a = delayed(cfg_in['fun_proc_loaded'], nout=2)(ddf, cfg_in)
                # if isinstance(date, tuple):
                #     date, a = date
                # if isinstance(a, pd.DataFrame):
                #     a_is_dask_df = False
                # else:chunksize=cfg_in['blocksize']
                ddf_len = len(ddf)
                counts_divisions = list(range(1, int(ddf_len / cfg_in.get('decimate_rate', 1)), cfg_in['blocksize']))
                counts_divisions.append(ddf_len)
                ddf = dd.from_delayed(a, divisions=(0, counts_divisions))
                date = dd.from_delayed(date_delayed, meta=meta_time_index, divisions=counts_divisions)
                date = dd.from_dask_array(date.values, index=ddf.index)
                # date = dd.from_pandas(date.to_series(index=), chunksize=cfg_in['blocksize'], )
                # _pandas(date, chunksize=cfg_in['blocksize'], name='Time')
            else:  # getting df with time col
                meta_out = cfg_in['fun_proc_loaded'].meta_out(cfg_in['dtype']) if callable(
                    cfg_in['fun_proc_loaded'].meta_out) else None
                ddf = ddf.map_partitions(cfg_in['fun_proc_loaded'], cfg_in, meta=meta_out)
                date = ddf.Time
    except IndexError:
        print('no data?')
        return None, None
        # add time shift specified in configuration .ini

    n_overlap = 2 * int(np.ceil(cfg_in['fs'])) if cfg_in.get('fs') else 50
    # reset_index().set_index('index').
    meta2 = {'Time': 'M8[ns]', 'b_ok': np.bool8}

    #     pd.DataFrame(columns=('Time', 'b_ok'))
    # meta2.time = meta2.time.astype('M8[ns]')
    # meta2.b_ok = meta2.b_ok.astype(np.bool8)

    def time_corr_df(t, cfg_in):
        """convert tuple returned by time_corr() to dataframe"""
        return pd.DataFrame.from_dict(OrderedDict(zip(meta2.keys(), time_corr(t, cfg_in))))
        # return pd.DataFrame.from_items(zip(meta2.keys(), time_corr(t, cfg_in)))
        # pd.Series()

    # date.rename('time').to_series().reset_index().compute()
    # date.to_series().repartition(divisions=ddf.divisions[1])

    '''
    def time_corr_ar(t, cfg_in):
        """convert tuple returned by time_corr() to dataframe"""
        return np.array(time_corr(t, cfg_in))
        #return pd.DataFrame.from_items(zip(meta2.keys(), time_corr(t, cfg_in)))
        # pd.Series()
    da.overlap.map_overlap(date.values, time_corr_ar, depth=n_overlap)
    '''

    l.info('time correction in %s blocks...', date.npartitions)
    df_time_ok = date.map_overlap(time_corr_df, before=n_overlap, after=n_overlap, cfg_in=cfg_in, meta=meta2)
    # .to_series()
    # if __debug__:
    #     c = df_time_ok.compute()
    # tim = date.compute().get_values()
    # tim, b_ok = time_corr(tim, cfg_in)

    # return None, None
    # if len(ddf) == 1:  # size
    #     ddf = ddf[np.newaxis]

    # npartitions = ddf.npartitions
    # ddf = ddf.reset_index().set_index('index')
    # col_temp = set(ddf.columns).difference(cfg_in['dtype_out'].names).pop()

    # ddf.index is not unique!
    # if col_temp:
    #      # ddf[col_temp].compute().is_unique # Index.is_monotonic_increasing()
    #     # ddf[col_temp] = ddf[col_temp].map_partitions(lambda s, t: t[s.index], tim, meta=meta)
    try:
        df_time_ok = df_time_ok.persist()

    except Exception as e:
        l.exception('Can not speed up by persist')
        # # something that can trigger error to help it identificate ???
        # date = date.persist()
        # df_time_ok = df_time_ok.compute()
        df_time_ok = time_corr_df(
            (date_delayed if date_delayed is not None else date).compute(), cfg_in=cfg_in)
        # raise pass

    # df_time_ok.compute(scheduler='single-threaded')
    if isinstance(df_time_ok, dd.DataFrame):
        nbad_time = len(df_time_ok['b_ok']) - df_time_ok['b_ok'].sum().compute()
        if nbad_time:
            nonzero = []
            for b_ok in df_time_ok['b_ok'].fillna(0).ne(True).partitions:
                b = b_ok.compute()
                if b.any():
                    nonzero.extend(b.to_numpy().nonzero()[0])
        l.info('Removing %d bad time values: %s%s', nbad_time, nonzero[:20],
               ' (shows first 20)' if nbad_time > 20 else '')

        df_time_ok.Time = df_time_ok.Time.where(df_time_ok['b_ok'])
        # try:  # catch exception not works
        #     # if not interpolates (my condition) use simpler method:
        #     df_time_ok.Time = df_time_ok.Time.map_overlap(pd.Series.interpolate, before=n_overlap, after=n_overlap,
        #                                                   inplace=True, meta=meta_time)  # method='linear', - default
        # except ValueError:
        df_time_ok.Time = df_time_ok.Time.map_overlap(pd.Series.fillna, before=n_overlap, after=n_overlap,
                                                      method='ffill', inplace=False, meta=meta_time)
    else:
        nbad_time = len(df_time_ok['b_ok']) - df_time_ok['b_ok'].sum()
        l.info('Removing %d bad time values: %s%s', nbad_time,
               df_time_ok['b_ok'].fillna(0).ne(True).to_numpy().nonzero()[0][:20],
               ' (shows first 20)' if nbad_time > 20 else '')

        df_time_ok.loc[df_time_ok['b_ok'], 'Time'] = pd.NaT
        try:
            df_time_ok.Time = df_time_ok.Time.interpolate(
                inplace=False)  # inplace=True - not works, method='linear', - default
        except ValueError:  # if not interpolates (my condition) use simpler method:
            df_time_ok.Time = df_time_ok.Time.fillna(method='ffill', inplace=True)

    if nbad_time:
        # # dask get IndexingError: Unalignable boolean Series provided as indexer (index of the boolean Series and of the indexed object do not match):
        # ddf_out = ddf.loc[df_time_ok['b_ok'], list(cfg_in['dtype_out'].names)].set_index(
        #    df_time_ok.loc[df_time_ok['b_ok'], 'Time'], sorted=True)

        # so we have done interpolate that helps this:
        ddf_out = ddf.loc[:, list(cfg_in['dtype_out'].names)].set_index(df_time_ok['Time'])  # , sorted=True
        ddf_out = ddf_out.loc[df_time_ok['b_ok'], :]
    else:
        # print('data loaded shape: {}'.format(ddf.compute(scheduler='single-threaded').shape))  # debug only
        ddf_out = ddf.loc[:, list(cfg_in['dtype_out'].names)].set_index(df_time_ok['Time'], sorted=True)

    # if isinstance(df_time_ok, dd.DataFrame) else df_time_ok['Time'].compute()
    # **({'sorted': True} if a_is_dask_df else {}
    # [cfg_in['cols_load']]
    # else:
    #     col_temp = ddf.columns[0]
    #     b = ddf[col_temp]
    #     b[col_temp] = b[col_temp].map_partitions(lambda s, t: t[s.index], tim, meta=meta)
    #     ddf = ddf.reset_index().set_index('index').set_index(b[col_temp], sorted=True).loc[:, list(cfg_in['dtype_out'].names)]

    # date = pd.Series(tim, index=ddf.index.compute())  # dd.from_dask_array(da.from_array(tim.get_values(),chunks=ddf.divisions), 'Time', index=ddf.index)
    # date = dd.from_pandas(date, npartitions=npartitions)
    # ddf = ddf.loc[:, list(cfg_in['dtype_out'].names)].set_index(date, sorted=True)

    # ddf = ddf.loc[:, list(cfg_in['dtype_out'].names)].compute()
    # ddf.set_index(tim, inplace=True)
    # ddf = dd.from_pandas(ddf, npartitions=npartitions)

    logger = logging.getLogger("dask")
    logger.addFilter(lambda s: s.getMessage() != "Partition indices have overlap.")
    # b_ok = df_time_ok['b_ok'].to_dask_array().compute() if isinstance(
    #     df_time_ok, dd.DataFrame) else df_time_ok['b_ok'].to_numpy()

    # b_ok_ds= df_time_ok.set_index('Time')['b_ok']
    return ddf_out  # , b_ok_ds
