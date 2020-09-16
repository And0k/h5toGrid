# -*- coding: utf-8 -*-
import pytest
import sys
from pathlib import Path
#from to_pandas_hdf5.csv2h5 import *  # main as csv2h5, __file__ as file_csv2h5, read_csv

# import imp; imp.reload(csv2h5)

drive_d = Path('D:/' if sys.platform == 'win32' else '/mnt/D')  # allows to run on both my Linux and Windows systems:
scripts_path = drive_d.joinpath('Work/_Python3/And0K/h5toGrid/scripts')
sys.path.append(str(Path(scripts_path).parent.resolve()))  # os.getcwd()

path_data = Path(r'd:\Work\_Python3\hartenergy-find_formation_names\test\200202_1st750rows.tsv')
path_data_full = Path(r'd:\Work\_Python3\hartenergy-find_formation_names\data\HartEnergy\wells_US_all.tsv')

from to_vaex_hdf5.csv_to_h5vaex import *  #to_pandas_hdf5.

import logging
from rarfile import BadRarFile

#from omegaconf import DictConfig
#import hydra
#import messytables

#from clize import run, converters, parameters
#from sigtools.wrappers import decorator

import numpy as np
import pandas as pd

VERSION = '0.0.1'
l = logging.getLogger(__name__)


@pytest.fixture(params=[None, '.zip', '.rar'])
def data_path_with_different_ext(request):
    if request.param:
        return path_data.with_suffix(request.param)
    return path_data


@pytest.mark.skip(reason="passed")
def test_open_csv_or_archive_of_them(data_path_with_different_ext):

    #filename: Union[Path, Iterable[Union[Path, str]]]

    line_count: int = 0
    try:
        for csv_file in open_csv_or_archive_of_them(data_path_with_different_ext):
            for line in csv_file:  # readlines
                line_count += 1
    except BadRarFile:
        if line == b"bsdtar: Error opening archive: Failed to open '--'\r\n":
            Exception("could not find unrar in the environment?")
    assert line_count == 750    # csv file should have this number of lines


@pytest.fixture(scope="module")
def get_file_handle(request):
    created_gens = []

    def _get_file_handle(path):
        # uses this module constant path_data
        #path_data = getattr(request.module, 'path_data', "default path_data variable's value")
        gen_csv_handles = open_csv_or_archive_of_them(path, binary_mode=True)
        handle = next(gen_csv_handles)
        created_gens.append(gen_csv_handles)
        return handle

    yield _get_file_handle

    # force an immediate generator cleanup triggering closing handles immediately
    for generator in created_gens:
        generator.close()


@pytest.mark.parametrize('paths', [path_data])
def test_determine_messytables_types(paths, get_file_handle):
    file_handle = get_file_handle(paths)
    (headers, messytables_types, row_set) = determine_messytables_types(file_handle)   # types = type_guess(), row_set.register_processor(...) executed

    # # To peek, a sample af iterator over the table is provided
    # print(next(row_set.sample))


@pytest.mark.skip(reason="passed")
@pytest.mark.parametrize('paths', [path_data_full])
def test_determine_numpy_types(paths, get_file_handle):
    file_handle = get_file_handle(paths)
    dtype, cols_with_nans, dates_formats = determine_numpy_types(file_handle)
    assert dtype == dtype_data_full

@pytest.mark.parametrize('source_type', ['str', 'string', None])
@pytest.mark.parametrize('typ', ['i4', 'i8'])  #'i1', 'i2',
def test_coerce_to_exact_dtype(typ, source_type):
    def f_astype(s, srs_type=source_type):
        return s if srs_type is None else s.astype(str).astype(srs_type) if srs_type=='string' else s.astype(srs_type)

    typ_bites = np.dtype(typ).itemsize * 8
    bytes_take = range(typ_bites - 5, typ_bites + 5)
    ser_in = pd.Series((2**(n - 1) for n in bytes_take), index=bytes_take, name='max_values_for_bites')  # excluding one bit for sign
    ser_out = coerce_to_exact_dtype(f_astype(ser_in), dtype=typ)
    assert (ser_out == ser_in).loc[:(typ_bites - 1)].all()
    assert (ser_out.loc[typ_bites:]==-1).all()

    if typ=='i4':
        # Bad value test
        ser_in_str = ser_in.astype(str)
        ser_in_str.iloc[0] = "Hi! I'm bad value!"
        ser_out = coerce_to_exact_dtype(f_astype(ser_in_str), dtype=typ)
        right_out = ser_in
        right_out.iloc[0] = -1
        right_out.loc[typ_bites:] = -1
        assert (ser_out == right_out).all()

        # Bad value test


#@pytest.mark.skip(reason="long time test")
def test_h5pandas_to_vaex_combine(csv_path=path_data_full):
    n_tmp_files = 7
    found_tmp_files = h5pandas_to_vaex_combine(
        tmp_search_pattern=str(csv_path.with_name('wells_US_all_00000000?.vaex.hdf5')),
        path_out_str=str(path_data_full.with_suffix('.vaex.h5')),
        check_files_number=n_tmp_files,
        del_found_tmp_files=True)
    assert n_tmp_files==found_tmp_files


@pytest.mark.skip(reason="long time test")
def test_h5pandas_to_vaex():
    file = r'c:\Users\and0k\AppData\Local\Temp\wells_US_all.h5'
    h5pandas_to_vaex(file)
    pass