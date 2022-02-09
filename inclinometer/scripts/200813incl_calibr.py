#!/usr/bin/env python3
# coding:utf-8
"""
  Author:  Andrey Korzh <ao.korzh@gmail.com>
  Purpose: Calibrate AB SIO RAS inclinometers: step1. magnetometer and accelerometer, step2. Water velocity(Incl.angle)
  Created: 20.08.2020
  Modified: 20.01.2022
"""
import sys
from pathlib import Path
from yaml import safe_dump as yaml_safe_dump

# import my scripts
drive_d = Path('D:/' if sys.platform == 'win32' else '/mnt/D')  # allows to run on both my Linux and Windows systems:
scripts_path = drive_d.joinpath('Work/_Python3/And0K/h5toGrid/scripts')
sys.path.append(str(Path(scripts_path).parent.resolve()))  # os.getcwd()

import to_vaex_hdf5.cfg_dataclasses as cfg_d
from inclinometer.incl_calibr_hy import main as incl_calibr_hy


step = 2  # 1: lab # 2: tank

path_db_raw = Path(
    r'd:\WorkData\_experiment\inclinometer\_Schukas\210603_lab\_raw\220128.raw.h5'
    )
db_in = str(path_db_raw).replace('\\', '/')

if step == 1:
    # Setting of hydra.searchpath to cruise specific config dir: "{path_db_raw.parent}/cfg_proc"
    # within inclinometer/cfg/incl_h5clc_hy.yaml - Else it will be not used and hydra will only warn
    # with open(r'D:\Work\_Python3\And0K\h5toGrid\inclinometer\cfg\incl_calibr_hy.yaml', 'w') as f:
    #     yaml_safe_dump({
    #         'defaults': ['base_incl_h5clc_hy', '_self_'],
    #         'hydra': {'searchpath': [f'file://{path_db_raw.parent}/cfg_proc'.replace('\\', '/')]}
    #         }, f)


    cfg = cfg_d.main_call([
        # r'--config-path=scripts\cfg',
        # fr'--config-path={path_db_raw.parent}/cfg_proc'.replace('\\', '/'),  # not works
        f'in.db_path="{db_in}"',
        'in.prefix=incl_b',
        f'out.db_paths=["{db_in}"]'
        # can not miss "yml" as opposed to "yaml":
        # r'--config-name=220128incl_load#b-caliblab.yml'
        # r'# 220111incl_load#d-caliblab.yml'
        ],
        fun=incl_calibr_hy
    )

    #incl_calibr(['cfg/201219incl_load-caliblab.yml'])
    """
    200813incl_calibr-lab-b.yml'])
    """


if step == 2:
    """ ### Coefs to convert inclination to |V| and zero calibration (not heading) ###

    Note: Execute after updating Veusz data file with previous step results. You should:
    - update coefficients in hdf5 store that vsz imports (done in previous step)
    - recalculate calibration coefficients: zeroing (may be in vsz: zeroing interval in it must be set) and fit Velocity
    - save vsz
    Note: Updates Vabs coefs and zero calibration in source for vsz (but this may not affect the Vabs coefs in vsz
    because of zero calibration is in vsz too).
    """
    from utils2init import path_on_drive_d
    from inclinometer.h5from_veusz_coef import main as h5from_veusz_coef
    from inclinometer.h5inclinometer_coef import h5copy_coef
    # from veuszPropagate import __file__ as file_veuszPropagate


    probes = range(26, 31)  # range(1, 31) [23,30,32] 17,18 [3,12,15,19,1,13,14,16] [1,4,5,7,11,12]  # [4,5,11,12]   #[29, 30, 33]  # [3, 14, 15, 16, 19]
    #channels_list = ['M', 'A']  # []

    # Output coefficients here:
    # 1. stand data (same as input for 1st step) - set no args if it is same as db_path_tank
    db_path_calibr_scalling = path_on_drive_d(
        r'd:\WorkData\_experiment\inclinometer\_Schukas\210603_lab\_raw\220128.raw.h5'
        # r'D:\WorkData\~configuration~\inclinometr\190710incl.h5'
        # r'd:\WorkData\_experiment\inclinometer\_Schukas\210603_lab\_raw\210603incl.h5'
        )
    # 2. tank data
    db_path_tank = path_on_drive_d(  # path to load calibration data: newer first
        r'd:\WorkData\_experiment\inclinometer\_Schukas\200807_tank[b01-b30]\200807_calibr-tank-b.h5'
        # r'd:\WorkData\_experiment\inclinometer\220112_stand_D01\_raw\220112.raw.h5'
        # r'd:\WorkData\_experiment\inclinometer\210331_tank[4,5,9,10,11,19,28,33,36,37,38]\210331incl.h5'
        # r'd:\WorkData\_experiment\inclinometer\_Schukas\200807_tank[b01-b30]\200807_calibr-tank-b.h5'
        )
    #vsz_substr_not_in_tbl = 'tank@'
    tbl_prefix = 'incl_b'  # 'i_d'
    vsz_data = {'veusze': None}
    for i, probe in enumerate(probes):
        # incl_calibr not supports multiple time_ranges so calculate one by one probe
        tbl = f'{tbl_prefix}{probe:0>2}'  # note: regex result from veusz name by re_tbl_from_vsz_name below must be same
        # f'incl_b{probe:0>2}'
        vsz_path = db_path_tank.with_name(
            tbl
            # f'i_{vsz_substr_not_in_tbl}d{probe:0>2}'
            # {db_path_tank.stem}
            ).with_suffix('.vsz')
        vsz_data = h5from_veusz_coef([
            'empty.yml',  #str(Path(file_veuszPropagate).with_name('veuszPropagate.ini')),
            '--data_yield_prefix', 'Inclination',
            '--path', str(vsz_path),
            '--pattern_path', str(vsz_path),
            '--widget', '/fitV(incl)/grid1/graph/fit_t/values',
            # '/fitV(force)/grid1/graph/fit1/values',
            '--data_for_coef', 'max_incl_of_fit_t',
            '--out.path', str(db_path_tank),
            #'--re_sub_tbl_from_vsz_name', vsz_substr_not_in_tbl,
            '--channels_list', 'M,A',
            '--b_update_existed', 'True',  # to not skip.
            '--export_pages_int_list', '',  #4 0 = all
            '--b_interact', 'False',
            '--b_execute_vsz', 'True',
            '--return', '<embedded_object>',  # reuse to not bloat memory
            ],
            veusze=vsz_data['veusze'])

        if db_path_calibr_scalling and vsz_data is not None:
            # if step == 3:
            # to 1st db too
            # l = init_logging(logging, None)
            print(f"Adding coefficients to {db_path_calibr_scalling}/{tbl} from {db_path_tank}")
            h5copy_coef(db_path_tank, db_path_calibr_scalling, tbl, ok_to_replace_group=True)

            vsz_data['veusze'].Close()
            try:
                vsz_data['veusze'].WaitForClose()
            except AttributeError:  # already 'NoneType' => closed ok
                pass
        else:
            vsz_data = {'veusze': None}
