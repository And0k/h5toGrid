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
from datetime import datetime
from to_pandas_hdf5.h5_dask_pandas import *


class test_1(unittest.TestCase):
    def setUp(self):
        self.a = dd.from_pandas(pd.DataFrame(
            {'Sal': range(0, 40, 5), 'Temp': range(10, 50, 5)}, index=pd.date_range(
                start='2018-04-18T11:00:00', end='2018-04-18T18:00:00', periods=8, tz='UTC')),
            # pd.to_datetime(['2018-04-18T11:00:00', ...
            npartitions=1)
        self.cfg_filter = {'date_min': datetime.strptime('2018-04-18T13:00:00', '%Y-%m-%dT%H:%M:%S'),
                           'max_Temp': 40
                           }

    def test_filterGlobal_minmax(self):
        out = filterGlobal_minmax(self.a, tim=None, cfg_filter=self.cfg_filter).compute()
        assert all(out[out].index == pd.date_range(
            start='2018-04-18T14:00:00', end='2018-04-18T16:00:00', periods=3, tz='UTC'))


if __name__ == '__main__':
    unittest.main()
