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

from to_vaex_hdf5.h5tocsv import *

import logging

import numpy as np
import pandas as pd

VERSION = '0.0.1'
l = logging.getLogger(__name__)


# @pytest.mark.skip(reason="passed")
@pytest.mark.parametrize('names', [
    ['49001,15,20', '49001,15,20', '49050,65', '50000', '50000,3', '49024', '49001,15,20', '50000,3']
    ])
def test_rep_comma_sep_items(names):
    out = rep_comma_sep_items(names)
    assert out == ['49001', '49015', '49050', '50000', '50000', '49024', '49020', '50003']