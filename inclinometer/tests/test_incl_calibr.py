import pytest
import sys
from inclinometer.incl_calibr_hy import *
from inclinometer.incl_calibr_hy import __file__ as path_incl_calibr_hy


path_db = ''


@pytest.fixture()
def sys_argv_like_no_test():
    # compensate for pytest args
    sys_argv_save = sys.argv
    sys.argv = [path_incl_calibr_hy] # sys_argv_save[-1:]
    print('Running path:', sys.argv[0])
    yield None
    sys.argv = sys_argv_save


def test_main(sys_argv_like_no_test):
    """
    Real calculation. todo: make test data and config
    :param sys_argv_like_no_test:
    :return:
    """
    cfg = main_call([
        r'--config-path=scripts\ini',
        r'--config-name=201219incl_load-caliblab.yml'  # can not miss "yml" as opposed to "yaml"
        #r'--config-path=scripts\ini\201219incl_load-caliblab.yml'
        # f'input.db_path="{path_db}"',
        ],
        #fun=main
    )




def test_main_info(sys_argv_like_no_test):
    cfg = main_call([
        r'--info defaults-tree'
        ],
    )