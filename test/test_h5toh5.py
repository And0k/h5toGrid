# datafile = '/mnt/D/Work/_Python3/And0K/h5toGrid/test/csv2h5/data/inclin_Kondrashov.txt'
import sys
import unittest

from to_pandas_hdf5.csv2h5 import *  # main as csv2h5, __file__ as file_csv2h5, read_csv

# import imp; imp.reload(csv2h5)

path_cruise = '/mnt/D/Work/_Python3/And0K/h5toGrid/test/csv2h5/data'  # r'd:\workData\_source\BalticSea\180418_Svetlogorsk'
scripts_path = '/mnt/D/Work/_Python3/And0K/h5toGrid/scripts'  # to find ini
test_path = Path('/mnt/D/Work/_Python3/And0K/h5toGrid/test')
sys.path.append(str(Path(test_path).parent.resolve()))

# g.es(sys.argv[0])
# sys.argv[0] = scripts_path                                    # to can write log to ".\log" directory
# os.chdir(os.path.dirname(scripts_path))
from to_pandas_hdf5.h5toh5 import *


class test_1(unittest.TestCase):
    def setUp(self):
        self.cfg = {'in': {
            'path': r'd:\workData\BalticSea\181005_ABP44\181005_ABP44.h5',
            'tables': ['.*inclinometers/incl.*']
            }}

    def test_h5find_tables(self):
        with pd.HDFStore(self.cfg['in']['path'], mode='r') as store:
            tables_found1 = h5find_tables(store, self.cfg['in']['tables'][0])
            tables_found2 = h5find_tables(store, 'incl.*', '.*inclinometers')  # splitted arguments for same path
        assert tables_found1 == tables_found2


if __name__ == '__main__':
    unittest.main()
