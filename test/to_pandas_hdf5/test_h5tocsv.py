import os, sys
import pytest
from to_vaex_hdf5.h5tocsv import *
import to_vaex_hdf5.cfg_dataclasses

from omegaconf import OmegaConf, DictConfig, MISSING
from hydra.experimental import initialize, compose

path_db = Path(__file__).parent.parent / 'data/200520_Nord3-nav&ctd_loaded.h5'
tbl_data = 'CTD_Idronaut_OS310'

def cfg_initialized_with_hydra() -> DictConfig:
    """
    Initialize will add config_path (the config search path within the context). Cnfig path is relative to the file calling initialize (this file)
    The module with configs should be importable. it needs to have a __init__.py (can be empty).
    """

    # cs = to_vaex_hdf5.cfg_dataclasses.hydra_cfg_store(cs_store_name, cs_store_group_options)

    # check config file data
    with initialize(config_path='cfg'): #"../../to_vaex_hdf5/ini"
        # config is relative to a module
        cfg = compose(config_name=cs_store_name)  # overrides=["app.user=test_user"]
        assert cfg == {'input': {'db_path': 'data/data_out.h5'}}
        # default groups loaded:

    # @pytest.fixture()
    return cfg


def test_main_init() -> None:
    """
    - finds input files
    - asks user to proceed if need
    :return:
    """

    sys_argv_save = sys.argv
    sys.argv = sys.argv[1:] + [f'input.db_path="{path_db}"',
                # f'--config-dir={Path(__file__).parent}/cfg',  # contents not read
                #f'--config-path=cfg',
                #f'--config-name={cs_store_name}'
                ]

    cfg = {}
    cfg_by_hydra_ = None

    @hydra.main(config_name=cs_store_name)  # adds config store cs_store_name data/structure to :param config
    def test_cs(cfg_by_hydra: ConfigType) -> None:
        nonlocal cfg
        nonlocal cfg_by_hydra_
        cfg_by_hydra_ = cfg_by_hydra
        cfg = to_vaex_hdf5.cfg_dataclasses.main_init(cfg_by_hydra, cs_store_name)
        cfg = to_vaex_hdf5.cfg_dataclasses.main_init_input_file(cfg, cs_store_name)
    test_cs()

    assert cfg_by_hydra_['input']['db_path'] == str(path_db)
    assert cfg['in']['db_path'] == path_db
    sys.argv = sys_argv_save


def test_main() -> None:
    i_cruise = 55
    text_file_name_add = f'E090005O2_AI_{i_cruise}_H10_'

    # compensate for pytest args
    sys_argv_save = sys.argv
    sys.argv = sys.argv[1:]

    main_call([
        f'input.db_path="{path_db}"',
        f'input.tables=[{tbl_data}]',
        f'input.tables_log=[{tbl_data}/logRuns]',
        # f'out.text_path',
        f'out.text_date_format="%Y-%m-%dT%H:%M:%S"',
        f'out.text_float_format="%.6g"',
        f'out.file_name_fun="f\'{text_file_name_add}{{i+1:0>2}}.csv\'"',
        f'out.file_name_fun_log="\'{text_file_name_add}POS.csv\'"',
        # rec_num;identific;station;Pres;Temp;cond;Sal;O2%;O2ppm;sigmaT;soundV
        f'+out.cols_log={{rec_num: "i + 1", identific: "i + 1", station: "{int(i_cruise)*1000+1} + i" , LONG: Lon_st, LAT: Lat_st, DATE: index}}',
        ''.join([
            f'+out.cols={{rec_num: "i + 1", identific: "i_log_row + 1", station: "{int(i_cruise)*1000+1} + i_log_row", ',
            ', '.join([p if ':' in p else f'{p}: {p}' for p in 'Pres;Temp:Temp90;Cond;Sal;O2;O2ppm;SigmaT;SoundVel'.split(';')]),
            '}'
            ]),
        'out.sep=";"'
        ])
    sys.argv = sys_argv_save

    # for group in cs_store_group_options:
    #     assert group in cfg
    #     # hydra allows set parameter
    #     cfg['filter']['max']['Sal'] = 0
    #     # assert cfg == {
    #     #     "filter": {"user": "test_user", "num1": 10, "num2": 20},
    #     #     "input": {"host": "localhost", "port": 3306},
    #     # }

