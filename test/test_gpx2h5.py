import pytest
from pathlib import Path
import logging

from to_pandas_hdf5.gpx2h5 import *

l = logging.getLogger(__name__)


# @pytest.fixture(params=[None, '.zip', '.rar'])
# def data_path_with_different_ext(request):
#     if request.param:
#         return path_data.with_suffix(request.param)
#     return path_data
path_db= Path(r'temp\test_gpx2h5.h5')
path_data= Path('data')

def test_main():
    main(['',
            '--db_path', str(path_db.absolute()),
            '--path', str((path_data / r'*.gpx').absolute()),
            '--tables_list', ',navigation,',  # skip waypoints
            '--table_prefix', r'',
            # '--min_date', '2019-07-17T14:00:00',
            '--b_interact', '0',
            ])