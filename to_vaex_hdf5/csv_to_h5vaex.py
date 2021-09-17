#!/usr/bin/env python3
# coding:utf-8
"""
  Author:  Andrey Korzh <ao.korzh@gmail.com>
  Purpose: Convert (multiple) csv and alike text files to vaex hdf5 store
  Created: 10.06.2020
  Modified:
"""

from pathlib import PurePath, Path
from typing import Any, Callable, Iterator, Iterable, Mapping, Dict, Optional, Set, Sequence, BinaryIO, TextIO, Tuple, \
    Union
import logging
import glob

import h5py
import numpy as np
import pandas as pd
import vaex, vaex.ml as vml
# import pyarrow as pa
import re
import itertools
from collections import defaultdict
# import threading
# lock = threading.Lock()

# my:
from utils2init import init_file_names, Ex_nothing_done, set_field_if_no, cfg_from_args, my_argparser_common_part, \
    this_prog_basename, init_logging, open_csv_or_archive_of_them, LoggingStyleAdapter, FakeContextIfOpen, standard_error_info

try:
    # this need for determine_messytables_types() - initial step that can done in special python environment
    import messytables
except ImportError:
    class messytables():
        """
        Dummy class to prevent python parser error
        """

        def __init__(self):
            return None

        def CSVTableSet(self, file):
            return None

        def headers_guess(self, rowset):
            return None

        def headers_processor(self, headers):
            return None

        def offset_processor(self, offset):
            return None

        def types_processor(self, types):
            return None

        class types():
            TYPES = None




if __name__ == '__main__':
    lf = None  # see main(): l = init_logging(logging, None, cfg['program']['log'], cfg['program']['verbose'])
else:
    lf = LoggingStyleAdapter(logging.getLogger(__name__))


def h5pandas_to_vaex_file_names(file_in: Union[None, str, PurePath] = None, path_out_str: Optional[str] = None):
    """
    Names to Prepare h5pandas_to_vaex_combine

    :param file_in: path of input pandas (pytables) hdf5 file. Optional if other argument is not None.
    :param path_out_str: path of output vaex hdf5 file. Should have two suffixes like ".vaex.h5". Optional if other argument is not None.
    :return:
        tmp_save_pattern: pattern to save temporary vaex files: replaces * in tmp_search_pattern with '{:09d}' to use with str.format(pattern, temporary file number),
        tmp_search_pattern: pattern to search temporary vaex files,
        path_out_str: '*.vaex.hdf5' file name where result will be stored - onfly if path_out_str was None
    """

    path_with_stem = Path((path_out_str.rsplit('.', 2) if path_out_str else str(file_in).rsplit('.', 1))[0])
    path_out_parent = path_with_stem.parent

    # temporary data chunks will be writen each in separate file with easy searchable name:
    tmp_search_pattern_name = f'{path_with_stem}_*.vaex.hdf5'
    tmp_search_pattern = str(path_out_parent / tmp_search_pattern_name)
    tmp_save_pattern = tmp_search_pattern.replace('*', '{:09d}')

    if path_out_str:
        return tmp_save_pattern, tmp_search_pattern
    else:
        path_out_str = str(  # vaex not supports pathlib.Path
            (path_out_parent / tmp_search_pattern_name.replace('*', '')).with_suffix('.h5')
            )
        return tmp_save_pattern, tmp_search_pattern, path_out_str


def h5pandas_to_vaex_combine(tmp_search_pattern: str,
                             path_out_str: str,
                             check_files_number: int = None,
                             del_found_tmp_files: bool = False,
                             **export_hdf5_args) -> int:
    """
    Combine many vaex.hdf5 files to one
    :param tmp_search_pattern:
    :param path_out_str: path argument of vaex.dataframe.export_hdf5()
    :param check_files_number: if not None must be equl to number of found files
    :param del_found_tmp_files: not implemented feature
    :param export_hdf5_args, dict, optional. Note that here default of 'progress' is set to True
    :return: number of tmp files found
    """
    # Find files
    hdf5_list = glob.glob(tmp_search_pattern)
    hdf5_list.sort()
    # hdf5_list_array = np.array(hdf5_list)

    # Check files existence
    if Path(path_out_str).is_file():
        lf.warning('Overwriting {:s}!', path_out_str)
    if check_files_number:
        assert len(hdf5_list) == check_files_number, "Incorrect number of files"
        lf.info('Combining {:d} found {:s} files to {:s}', check_files_number, tmp_search_pattern, PurePath(path_out_str).name)
    else:
        check_files_number = len(hdf5_list)
        lf.info('Combining {:s} to {:s}', tmp_search_pattern, PurePath(path_out_str).name)
    master_df = vaex.open_many(hdf5_list)
    try:
        master_df.export_hdf5(**{'path': path_out_str, 'progress': True, **export_hdf5_args})
    except AttributeError as e:
        # , progress=True gets AttributeError: 'ProgressBar' object has no attribute 'stime0'
        lf.debug('Try install progressbar2')
        pass

    # delete tmp files found
    if del_found_tmp_files:
        # will not work, todo: do only when export finished (use custom progress func?)
        try:
            i = 0
            for i, path_tmp in enumerate(hdf5_list):
                Path(path_tmp).unlink()  # remove file
        except Exception:
            lf.exception('Combined {0:d} but removed {i:d} temporary vaex.hdf5 files:', check_files_number, i=i)
        else:
            lf.info('Combined and removed {0:d} files.', check_files_number)
    else:
        lf.info('Combined {:d} files ({:s}), they remains', check_files_number, tmp_search_pattern)
    return check_files_number


def h5pandas_to_vaex(file_in: Union[None, str, PurePath], del_found_tmp_files=False):
    """
    Pandas hdf5 to vaex.hdf5 conversion: saves tmp files, then searches and combines them.
    :param file_in: pandas hdf5 file
    :return:
    Uses this module functions:
        h5pandas_to_vaex_file_names()
        h5pandas_to_vaex_combine()
    """
    tmp_save_pattern, tmp_search_pattern, path_out_str = h5pandas_to_vaex_file_names(file_in)
    chunksize = 500000  # will get x00 MB files

    ichunk = 0
    for chunk in pd.read_hdf(file_in, 'csv', chunksize=chunksize):  # , where='a < someval'
        df = vaex.from_pandas(chunk)
        df.export_hdf5(tmp_save_pattern.format(ichunk))
        ichunk += 1
        print(ichunk, end=' ')

    h5pandas_to_vaex_combine(tmp_search_pattern, path_out_str,
                             check_files_number=ichunk, del_found_tmp_files=del_found_tmp_files)


def determine_messytables_types(file_handle, types=messytables.types.TYPES):
    """

    :param file_handle: file handle opened in binary mode
    :return: (headers, types, row_set)
    """

    # Load a file object:
    table_set = messytables.CSVTableSet(file_handle)

    # If you aren't sure what kind of file it is
    # table_set = messytables.any_tableset(file_handle)

    # A table set is a collection of tables:
    row_set = table_set.tables[0]

    # A row set is an iterator over the table, but it can only
    # be run once. To peek, a sample is provided:
    print(next(row_set.sample))

    # guess header names and the offset of the header:
    offset, headers = messytables.headers_guess(row_set.sample)
    row_set.register_processor(messytables.headers_processor(headers))

    # add one to begin with content, not the header:
    row_set.register_processor(messytables.offset_processor(offset + 1))

    # guess column types:
    types = messytables.type_guess(row_set.sample, types, strict=True)

    # and tell the row set to apply these types to
    # each row when traversing the iterator:
    row_set.register_processor(messytables.types_processor(types))

    # now run some operation on the data:
    return headers, types, row_set


def determine_numpy_types(file_handle, headers=None, messytables_types=None, num_values_enough=100, str_nans=None,
                          **read_csv_args) -> Tuple[Dict[str, str], Dict[str, Any], Dict[str, str]]:
    """
    Determines numpy types
    :param file_handle: file handle opened in binary mode
    :param headers: optional, list of column names
    :param messytables_types: optional, list of column types
    If headers or messytables_types is None then they are determinined using determine_messytables_types(...).
    :param num_values_enough: we use this minimum number of values to determine its type
    :param str_nans: set of strings to treat as NaN, default {'Unknown'}
    :param read_csv_args: dict of pandas.read_csv() args. Defaults here is set: {
         filepath_or_buffer_in: file_handle.name,
         chunksize: 10000,
         delimiter: '\t'}.
    :return: tuple of
    - dtype: dict of numpy types determined for each column: short str representation without alignment information,
    - dates_formats:
    - statistics:
        'has_nans': bool numpy array of lengh number of columns,
        'unique_vals': dict for each column: empty set or set of unique values if its number found was small

    """
    read_csv_args_with_defaults = {
        'filepath_or_buffer': file_handle.name,
        'chunksize': 10000,
        'delimiter': '\t'}
    read_csv_args_with_defaults.update(read_csv_args)

    if None in [headers, messytables_types]:
        (headers, messytables_types, row_set) = determine_messytables_types(file_handle)
    # types = type_guess(), row_set.register_processor(...) executed

    if str_nans is None:
        str_nans = {'NULL', 'Unknown', '--unknown--'}

    num_values_tested = 0
    lengths_of_str_types = {h: 0 for h, typ in zip(headers, messytables_types)}
    # output variable, we fill it with minimal string/numeric types for now:
    types = {
        h: ('S1' if isinstance(typ, messytables.types.DateType) or h.endswith('Date')  # we will parse dates separately
            else None if isinstance(typ, messytables.types.StringType)  # may be numeric
        else 'int8')
        for h, typ in zip(headers, messytables_types)
        }  # we will replace Nones
    statistics = {
        'has_nans': np.zeros(len(headers), np.bool),
        'unique_vals': {h: set() for h in headers}  # will insert values if only if < 10 unique in chunk
        }
    print(f'Determine types in text data by analyzing {read_csv_args_with_defaults["chunksize"]}-rows length chunks:')
    try:
        for ichunk, chunk in enumerate(pd.read_csv(**read_csv_args_with_defaults)):
            notna = chunk.notna()
            num_values_in_chunk = notna.sum()
            statistics['has_nans'] |= num_values_in_chunk.values < chunk.shape[0]  # any nans
            # todo count unique only, but check for category types if there is small possible options
            # to set df['col3'] = df['col3'].astype('category')
            num_values_tested += num_values_in_chunk
            if (num_values_tested > num_values_enough).all():
                break
            print(ichunk, end=', ')

            # Correct type by pandas
            any_values = num_values_in_chunk > 0  # if no data nobody can determine type
            types_of_cols_with_vals = pd.Series(messytables_types, index=chunk.columns)[any_values]
            for (col, typ) in types_of_cols_with_vals.items():
                str_by_pandas = (chunk[col].dtype == 'object')  # in ('object', 'string_', 'unicode_')  or pd.api.types.is_string_dtype(chunk[col])
                # Pandas ever determined col type as a string?
                if str_by_pandas or np.dtype(types[col]).kind == 'S':
                    if not str_by_pandas:
                        chunk[col] = chunk[col].astype(str)
                    # Finding minimum string type
                    is_nan_strings = chunk[col].isin(str_nans)
                    have_nan_strings = is_nan_strings.any()
                    if have_nan_strings:
                        num_values_tested[col] += ((~is_nan_strings).sum() - num_values_in_chunk[col])
                        if is_nan_strings.all():
                            continue
                    lengths_of_str_types[col] = int(
                        max(  # int() is needed to use 'd' formatting below else need use :.0f
                            lengths_of_str_types[col],
                            (chunk.loc[~is_nan_strings, col] if have_nan_strings else chunk[col]).str.len().max()
                            ))
                    # print(f'{col}[{typ}]: {lengths_of_str_types[col]}')   # 'object' can have NaNs so converting to str is needed
                    types[col] = f'S{lengths_of_str_types[col]:d}'  #
                else:
                    # Finding minimum numeric type for notna values by numpy
                    # If messytables determined String (it can if it has only None values) we try to make integer with new values
                    downcast = None if isinstance(typ, messytables.types.FloatType) else 'integer'
                    type_in_chunk = pd.to_numeric(chunk[col][notna[col]],
                                                  downcast=downcast).dtype  # or np.promote_types('f4', 'f8') and np.min_scalar_type?
                    types[col] = np.result_type(types[col], type_in_chunk) if types[
                        col] else type_in_chunk  # or np.find_common_type?
                    # print(f'{col}[{typ}]: {}')

            for col, n in chunk.loc[:, any_values].nunique().items():
                if n < 10:
                    statistics['unique_vals'][col].update(set(chunk[col].unique()))

        # may be better set str types separately:
        # # insert str types with lengths determined
        # for col, typ in enumerate(types):
        #     if lengths_of_str_types[col] > 0:
        #         types[col] = f'S{lengths_of_str_types[col]:.0f}'

    except Exception as e:
        lf.exception('Pandas reading error')

    dtype = {h: np.dtype(t).str[1:] for h, t in types.items()}  # compressed representation
    # formats for post conversion to dates where messytables determined date types
    dates_formats = {h: typ.format for h, typ in zip(headers, messytables_types) if
                     isinstance(typ, messytables.types.DateType)}

    return dtype, dates_formats, statistics

    # csv converters to 'M8[ns]'


def coerce_to_exact_dtype(s: Sequence, dtype, replace_bad=None):
    """
    Con version with handling errors including Overflow to specific numpy dtype
    :param s:
    :param dtype:
    :return:
    """
    to_integer = np.dtype(dtype).kind == 'i'
    if replace_bad is None:
        if pd.api.types.is_string_dtype(s.dtype):
            replace_bad = '-1' if to_integer else 'NaN'
        else:
            replace_bad = -1 if to_integer else np.nan

    iinfo = np.iinfo(dtype)

    # converting to floats
    nums = pd.to_numeric(s,
                         errors='coerce')  # Not converting to integers here with downcast='integer' because if any too big for int64 then it not works so we use astype() below

    def test_big_vals(nums):
        """
        Test that nums is within iinfo limits
        :param nums: numpy array
        :return: bool numpy array
        Note switches off and restores warning on nans. Other way:
        ok = np.zeros_like(nums, dtype=np.bool8)
        not_na = ~np.isnan(nums)
        nums_not_na = nums[not_na]
        """

        save_invalid = np.geterr()['invalid']  # save floating-point errors handling
        np.seterr(invalid='ignore')  # switch off warning on nans
        ok = np.logical_and(iinfo.min < nums, nums < iinfo.max)
        np.seterr(invalid=save_invalid)  # restore floating-point errors handling
        return ok

    ok = test_big_vals(nums.values)
    nums = s.where(ok, other=replace_bad).astype(
        dtype)  # no float error conversion for big int values like in to_numeric()
    return nums


def csv_to_h5(
        read_csv_args,
        to_hdf_args,
        dates_formats: Mapping[str, str],
        correct_fun: Tuple[None, bool, Callable[[pd.DataFrame], None]] = None,
        processing: Optional[Mapping[Tuple[Tuple[str], Tuple[str]], Callable[[Any], Any]]] = None,
        out_cols: Optional[Sequence] = None,
        continue_row=False,
        vaex_format: Optional[bool]=None
        ):
    """
    Read csv and write to hdf5
    :param read_csv_args: dict, must have keys:
        filepath_or_buffer, chunksize
    :param to_hdf_args:
        path_or_buf: default = read_csv_args['filepath_or_buffer'].with_suffix('vaex.h5' if vaex_format else '.h5')
        mode: default = 'w' if not continue_row else 'a',
        key: hdf5 group name in hdf5 file where store data
        ...
    :param dates_formats:
        column: csv column name wich need to be convert from str to DateTime,
        date_format: date formats
    :param processing: dict with
        keys: ((_input cols_), (_output cols_)) and
        values: function(_input cols_) that will return _output cols_
    :param out_cols: default is all excluding columns that in inputs but not in output of custom param:processing
    :param continue_row: csv row number (excluding header) to start with shifting index.
    If output file exist and continue_row = True then continue converting starting from row equal to last index in it,
    useful to continue after program interrupting or csv appending. If not exist then start from row 0 giving it index 0.
    If continue_row = integer then start from this row, giving starting index = continue_row
    :param correct_fun: function applied to each chunk returned by read_csv() which is a frame of column data of type str
    :param vaex_format: bool how to write chunks:
    - True: to many vaex hdf5 files. They at end will be converted to single vaex hdf5 file
    - False: appending to single pandas hdf5 table
    - None: evaluates to True if to_hdf_args['path_or_buf'] has next to last suffix ".vaex" else to False

    :return:
    """
    if to_hdf_args.get('path_or_buf'):
        if vaex_format is None:
            vaex_format = Path(str(to_hdf_args['path_or_buf']).strip()).suffixes[:-1] == ['.vaex']
    else:  # give default name to output file
        to_hdf_args['path_or_buf'] = Path(read_csv_args['filepath_or_buffer']).with_suffix(
            f'{".vaex" if vaex_format else ""}.h5'
            )

    # Deal with vaex/pandas storing difference
    if vaex_format:
        open_for_pandas_to_hdf = None
        tmp_save_pattern, tmp_search_pattern = h5pandas_to_vaex_file_names(
            path_out_str=str(to_hdf_args['path_or_buf'])
            )
        ichunk = None
    else:
        def open_for_pandas_to_hdf(path_or_buf):
            return pd.HDFStore(
                to_hdf_args['path_or_buf'],
                to_hdf_args.get('mode', 'a' if continue_row else 'w')
                )

    # Find csv row to start
    msg_start = f'Converting in chunks of {read_csv_args["chunksize"]} rows.'
    if continue_row is True:  # isinstance(continue_same_csv, bool)
        try:
            if vaex_format:

                hdf5_list = glob.glob(tmp_search_pattern)
                if len(hdf5_list):      # continue interrupted csv_to_h5()
                    hdf5_list.sort()
                    file_last = hdf5_list[-1]
                    lf.info('Found {:d} temporary files, continue from index found in last file', len(hdf5_list))
                    "table/columns/index"
                else:                   # add next csv data
                    file_last = to_hdf_args['path_or_buf']
                with h5py.File(file_last, mode='r') as to_hdf_buf:
                    continue_row = to_hdf_buf['table/columns/index/data'][-1] + 1
            else:
                with pd.HDFStore(to_hdf_args['path_or_buf'], mode='r') as to_hdf_buf:
                    continue_row = to_hdf_buf.select(to_hdf_args['key'], columns=[], start=-1).index[-1] + 1
        except (OSError) as e:
            msg_start += ' No output file.'
            continue_row = None
        except KeyError as e:
            msg_start += ' No data in output file.'
            continue_row = None
        else:
            msg_start += ' Starting from next to last loaded csv row:'
    elif continue_row:
        msg_start += ' Starting from specified csv data row:'
    if continue_row:
        lf.info('{:s} {:d}...', msg_start, continue_row)
        read_csv_args['skiprows'] = read_csv_args.get('skiprows', 0) + continue_row
    else:
        lf.info('{:s} begining from csv row 0, giving it index 0...', msg_start)

    dtypes = read_csv_args['dtype']

    # Set default output cols if not set
    if out_cols is None and processing:
        # we will out all we will have except processing inputs if they are not mentioned in processing outputs
        cols_in_used = set()
        cols_out_used = set()
        for (c_in, c_out) in processing.keys():
            cols_in_used.update(c_in)
            cols_out_used.update(c_out)
        cols2del = cols_in_used.difference(cols_out_used)
        out_cols = dtypes.keys()
        for col in cols2del:
            del out_cols[col]
    cols_out_used = set(out_cols if out_cols is not None else dtypes.keys())

    # Group cols for conversion by types specified
    str_cols = []
    int_and_nans_cols = []
    other_cols = []
    for col, typ in dtypes.items():
        if out_cols and col not in cols_out_used:
            continue
        kind = typ[0]
        (str_cols if kind == 'S' else
         int_and_nans_cols if kind == 'I' else
         other_cols).append(col)

    str_not_dates = list(set(str_cols).difference(dates_formats.keys()))
    min_itemsize = {col: int(dtypes[col][1:]) for col in str_not_dates}

    # Read csv, process, write hdf5
    with open(read_csv_args['filepath_or_buffer'], 'r') as read_csv_buf, \
            FakeContextIfOpen(open_for_pandas_to_hdf, to_hdf_args['path_or_buf']) as to_hdf_buf:
        read_csv_args.update({
            'filepath_or_buffer': read_csv_buf,
            'memory_map': True,
            'dtype': 'string'  # switch off read_csv dtypes convertion (because if it fails it is hard to correct:
            })  # to read same csv place by pandas)
        to_hdf_args.update({
            'path_or_buf': to_hdf_buf,
            'format': 'table',
            'data_columns': True,
            'append': True,
            'min_itemsize': min_itemsize
            })
        # rows_processed = 0
        # rows_in_chunk = read_csv_args['chunksize']

        for ichunk, chunk in enumerate(pd.read_csv(**read_csv_args)):
            if continue_row:
                if chunk.size == 0:
                    ichunk = np.ceil(continue_row / read_csv_args['chunksize']).astype(int) - 1
                    break  # continue_row is > data rows
                else:
                    chunk.index += continue_row

            lf.extra['id'] = f'chunk start row {chunk.index[0]:d}'
            if ichunk % 10 == 0:
                print(f'{ichunk}', end=' ')
            else:
                print('.', end='')

            if correct_fun:
                correct_fun(chunk)

            # Convert to user specified types

            # 1. dates str to DateTime
            for col, f in dates_formats.items():
                # the convertion of 'bytes' to 'strings' is needed for pd.to_datetime()
                try:
                    chunk[col] = pd.to_datetime(chunk[col], format=f)
                except ValueError as e:
                    lf.error(
                        'Conversion to datetime("{:s}" formatted as "{:s}") {:s} -> '
                        'Replacing malformed strings by NaT...', col, f, standard_error_info(e))
                    chunk[col] = pd.to_datetime(chunk[col], format=f, exact=False, errors='coerce')

            # 2. str to numeric for other_cols and int_and_nans_cols (which is limited support pandas extension dtypes)
            # but we use numpy types instead replasing nans by -1 to able write to hdf5
            chunk[other_cols] = chunk[other_cols].fillna('NaN')  # <NA> to numpy recognized eq meaning string
            chunk[int_and_nans_cols] = chunk[int_and_nans_cols].fillna('-1')
            for col in (int_and_nans_cols + other_cols):  # for col, typ in zip(nans.columns, chunk[nans.columns].dtypes):
                typ = dtypes[col]
                if col in int_and_nans_cols:
                    is_integer = True
                    typ = f'i{typ[1:]}'  # typ.numpy_dtype
                else:
                    is_integer = np.dtype(typ).kind == 'i'
                try:
                    chunk[col] = chunk[col].astype(typ)
                    continue
                except (ValueError, OverflowError) as e:
                    # Cleaning. In case of OverflowError we do it here to prevent ValueError while handling of OverflowError below.
                    pattern_match = r'^[\d]$' if is_integer else r'^-?[\d.]$'
                    ibad = ~chunk[col].str.match(pattern_match)
                    rep_val = '-1' if is_integer else 'NaN'
                    # ibad = np.flatnonzero(chunk[col] == re.search(r'(?:")(.*)(?:")', e.args[0]).group(1), 'ascii')
                    lf.error('Conversion {:s}("{:s}") {:s} -> replacing {:d} values not maching pattern "{:s}" with "{'
                             ':s}" and again...', typ, col, standard_error_info(e), ibad.sum(), pattern_match, rep_val)
                    chunk.loc[ibad, col] = rep_val
                    # astype(str).replace(regex=True, to_replace=r'^.*[^\d.].*$', value=
                try:
                    chunk[col] = chunk[col].astype(typ)
                except (OverflowError,
                        ValueError) as e:  # May be bad value from good symbols: r'^\d*\.\d*\.+\d*$' but instead checking it we do coerce_to_exact_dtype() on ValueError here too
                    lf.error('Conversion {:s}("{:s}") {:s} -> Replacing malformed strings and big numbers'
                    ' by NaN ...', typ, col, standard_error_info(e))
                    chunk[col] = coerce_to_exact_dtype(chunk[col], dtype=typ)

            # Limit big strings length and convert StringDtype to str to can save by to_hdf()
            for col, max_len in min_itemsize.items():  # for col, typ in zip(nans.columns, chunk[nans.columns].dtypes):
                chunk[col] = chunk[col].str.slice(stop=max_len)  # apply(lambda x: x[:max_len]) not handles <NA>
            chunk[str_not_dates] = chunk[str_not_dates].astype(str)

            # Apply specified data processing
            if processing:
                for (cols_in, c_out), fun in processing.items():
                    cnv_result = fun(chunk[list(cols_in)])
                    chunk[list(c_out)] = cnv_result

            # # Bad rows check
            # is_different = chunk['wlaWID'].fillna('') != chunk['wlaAPIHartStandard'].fillna('')
            # if is_different.any():
            #     i_bad = np.flatnonzero(is_different.values)
            #     lf.debug('have wlaWID != wlaAPIHartStandard in rows {:s}', chunk.index[i_bad])
            #     # chunk= chunk.drop(chunk.index[i_bad])   # - deleting
            #     pass

            # Check unique index
            # if chunk['wlaWID'].duplicated()

            try:
                if vaex_format:
                    df = vaex.from_pandas(chunk if out_cols is None else chunk[out_cols])
                    df.export_hdf5(tmp_save_pattern.format(ichunk))
                else:  # better to move this command upper and proc. by vaex instead of pandas
                    (chunk if out_cols is None else chunk[out_cols]).to_hdf(**to_hdf_args)
                #rows_processed += rows_in_chunk  # think we red always the same length exept last which length value will not be used

            except Exception as e:
                lf.exception('write error')
                pass
        try:
            del lf.extra['id']
        except KeyError:
            lf.info('was no more data rows to read')

    # If vaex store was specified then we have chunk files that we combine now by export_hdf5():
    if vaex_format:
        h5pandas_to_vaex_combine(tmp_search_pattern, str(to_hdf_args['path_or_buf']), check_files_number=ichunk+1)


def csv_to_h5_vaex(read_csv_args, to_hdf_args, dates_formats: Mapping[str, str],
              correct_fun: Tuple[None, bool, Callable[[pd.DataFrame], None]] = None,
              processing: Optional[Mapping[Tuple[Tuple[str], Tuple[str]], Callable[[Any], Any]]] = None,
              out_cols: Optional[Sequence] = None, continue_row=False):
    """
    Read csv and write to hdf5
    :param read_csv_args: dict, must have keys:
        filepath_or_buffer, chunksize
    :param to_hdf_args:
        vaex_format: bool how to write chanks:
            True: to many vaex hdf5 files. They at end will be converted to single vaex hdf5 file
            False: appending to single pandas hdf5 table
        path_or_buf: default = read_csv_args['filepath_or_buffer'].with_suffix('vaex.h5' if vaex_format else '.h5')
        mode: default = 'w' if not continue_row else 'a',
        key: hdf5 group name in hdf5 file where store data
        ...
    :param dates_formats:
        column: csv column name wich need to be convert from str to DateTime,
        date_format: date formats
    :param processing: dict with
        keys: ((_input cols_), (_output cols_)) and
        values: function(_input cols_) that will be used returning _output cols_
    :param out_cols: default is all excluding columns that in inputs but not in output of custom param:processing
    :param continue_row: csv row number (excluding header) to start with shifting index.
    If output file exist and continue_row = True then continue converting starting from row equal to last index in it,
    useful to continue after program interrupting or csv appending. If not exist then start from row 0 giving it index 0.
    If continue_row = integer then start from this row, giving starting index = continue_row
    :param correct_fun: function applied to each chunk (which is a frame of column data of type str) immediately after reading by read_csv()
    :return:
    """
    from astropy.io import ascii

    if not to_hdf_args.get('path_or_buf'):  # give default name to output file
        to_hdf_args['path_or_buf'] = Path(read_csv_args['filepath_or_buffer']).with_suffix('.vaex.h5')

    # prepare vaex/pandas storing
    open_for_pandas_to_hdf = None
    tmp_save_pattern, tmp_search_pattern = h5pandas_to_vaex_file_names(path_out_str=str(to_hdf_args['path_or_buf']))
    ichunk = None

    # find csv row to start
    msg_start = f'Converting in chunks of {read_csv_args["chunksize"]} rows.'
    if continue_row is True:  # isinstance(continue_same_csv, bool)
        try:
            hdf5_list = glob.glob(tmp_search_pattern)
            if len(hdf5_list):      # continue interrupted csv_to_h5()
                hdf5_list.sort()
                file_last = hdf5_list[-1]
                lf.info('Found {:d} temporary files, continue from index found in last file', len(hdf5_list))
                "table/columns/index"
            else:                   # add next csv data
                file_last = to_hdf_args['path_or_buf']
            with h5py.File(file_last, mode='r') as to_hdf_buf:
                continue_row = to_hdf_buf['table/columns/index/data'][-1]
        except (OSError) as e:
            msg_start += ' No output file.'
            continue_row = None
        except KeyError as e:
            msg_start += ' No data in output file.'
            continue_row = None
        else:
            msg_start += ' Starting from last csv row in output file:'
    elif continue_row:
        msg_start += ' Starting from specified csv data row:'
    if continue_row:
        lf.info('{:s} {:s}...', msg_start, continue_row)
        read_csv_args['skiprows'] = read_csv_args.get('skiprows', 0) + continue_row
    else:
        lf.info('{:s} Beging from csv row 0, giving it index 0...', msg_start)

    dtypes = read_csv_args['dtype']

    # Set default output cols
    if out_cols is None and processing:
        cols_in_used = set()
        cols_out_used = set()
        for (c_in, c_out) in processing.keys():
            cols_in_used.update(c_in)
            cols_out_used.update(c_out)
        cols2del = cols_in_used.difference(cols_out_used)
        out_cols = dtypes.keys()
        for col in cols2del:
            del out_cols[col]
    cols_out_used = set(out_cols if out_cols is not None else dtypes.keys())

    # prepare conversion to user specified types
    str_cols = []
    int_and_nans_cols = []
    other_cols = []
    for col, typ in dtypes.items():
        if out_cols and col not in cols_out_used:
            continue
        kind = typ[0]
        (str_cols if kind == 'S' else
         int_and_nans_cols if kind == 'I' else
         other_cols).append(col)

    str_not_dates = list(set(str_cols).difference(dates_formats.keys()))
    min_itemsize = {col: int(dtypes[col][1:]) for col in str_not_dates}

    with open(read_csv_args['filepath_or_buffer'], 'r') as read_csv_buf, \
            FakeContextIfOpen(open_for_pandas_to_hdf, to_hdf_args['path_or_buf']) as to_hdf_buf:
        read_csv_args.update({
            'filepath_or_buffer': read_csv_buf,
            'memory_map': True,
            'dtype': 'string'  # switch off read_csv dtypes convertion (because if it fails it is hard to correct:
            })  # to read same csv place by pandas)
        to_hdf_args.update({
            'path_or_buf': to_hdf_buf,
            'format': 'table',
            'data_columns': True,
            'append': True,
            'min_itemsize': min_itemsize
            })
        rows_processed = 0
        rows_in_chunk = read_csv_args['chunksize']

        # alternative to pd.read_csv(**read_csv_args) but without dataframes
        tbls = ascii.read(read_csv_buf, format='csv', guess=False, delimiter=read_csv_args['delimiter'],
                          data_start=read_csv_args['skiprows'], names=read_csv_args['names'],
                          fast_reader={'chunk_size': read_csv_args['chunksize'],
                                       'chunk_generator': True})
        # other variant could be vaex.from_ascii but it have no chunk option

        # ...
