import pytest
from pathlib import Path
import logging

from to_pandas_hdf5.CTD_calc import *

l = logging.getLogger(__name__)


# @pytest.fixture(params=[None, '.zip', '.rar'])
# def data_path_with_different_ext(request):
#     if request.param:
#         return path_data.with_suffix(request.param)
#     return path_data

path_db= (Path(r'data') / '200520_Nord3-nav&ctd_loaded.h5').absolute()  # Path(__file__).with_suffix('.h5').name
#path_data= Path('data')
device = 'CTD_Idronaut_OS310'
def test_main():
    main(['ini/CTD_calc-find_runs.ini',
        '--db_path', str(path_db),
        '--tables_list', f'{device}',
        '--min_samples', '50',  # fs*depth/speed = 200: if fs = 10Hz for depth 20m
        '--min_dp', '9',
        # '--b_keep_minmax_of_bad_files', 'True',
        # '--b_skip_if_up_to_date', 'True',
        '--b_interact', 'False',  # to can check with True value in pytest need to run with -s command line option
        # todo: check it. If False need delete all previous result of CTD_calc() or set min_time > its last log time
        # '--out.tables_list', '',
        ])