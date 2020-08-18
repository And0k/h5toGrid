import pytest

from to_pandas_hdf5.csv2h5_vaex import *

@pytest.mark.skip(reason="clise was not installed")
def test_with_prog_config():
    @with_prog_config
    def testing(cfg):
        print(cfg)
        return cfg

#test_with_prog_config()