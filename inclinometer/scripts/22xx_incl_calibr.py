#!/usr/bin/env python3
# coding:utf-8
"""
  Author:  Andrey Korzh <ao.korzh@gmail.com>
  Purpose: Calibrate AB SIO RAS inclinometers: step1. magnetometer and accelerometer, step2. velocity(incl.angle)
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

import cfg_dataclasses as cfg_d
import inclinometer.incl_calibr_hy as incl_calibr_hy
from inclinometer.h5inclinometer_coef import h5copy_coef
from utils2init import st, path_on_drive_d, init_logging


st.start = 35
st.end = 45  # if = st.start then process one step.
# Possible values:
# 10: lab,
# 15: copy lab coef to tank,
# 20: tank,
# 35: copy tank coef where need


path_db_raw = path_on_drive_d(
    # r'd:\WorkData\_experiment\inclinometer\_type_b\230117_stand@ib26,28-30\_raw\230117.raw.h5'
    # r'd:\WorkData\_experiment\inclinometer\_type_b\230117_stand@ib26,28-30\_raw\230117.raw.h5'
    # r'd:\WorkData\_experiment\_2018\inclinometer\181003_compas\181004.raw.h5'
    # r'd:\WorkData\_experiment\_2018\inclinometer\181003_compas\181003compas.h5'
    # r'd:\WorkData\_experiment\inclinometer\_Schukas\210603_lab\_raw\220128.raw.h5'
    )

if st(10, 'Magnetometer and accelerometer calibration'):

    db_in = str(path_db_raw).replace('\\', '/')

    # Setting of hydra.searchpath to cruise specific config dir: "{path_db_raw.parent}/cfg_proc" probes config directory
    # within inclinometer/cfg/incl_h5clc_hy.yaml - Else it will be not used and hydra will only warn
    path_cfg_default = (lambda p: p.parent / 'cfg' / p.name)(Path(incl_calibr_hy.__file__)).with_suffix('.yaml')
    with path_cfg_default.open('w') as f:
        yaml_safe_dump({
            'defaults': ['my_incl_calibr', '_self_'],  # 'base_incl_calibr_hy', {'override input': 'input_mod'}
            # 'hydra': {'searchpath': [path_db_raw.with_name("cfg_proc").as_uri().replace('///', '//')]}  # .as_posix()
            # file://d:/WorkData/_experiment/inclinometer/_type_b/230117_stand%40ib26%2C28-30/_raw/cfg_proc
            # file://d:/WorkData/_experiment/inclinometer/_type_b/230117_stand_ib26_28_30/_raw/cfg_proc
            'hydra': {'searchpath': [f'file://{path_db_raw.with_name("cfg_proc").as_posix()}']}  #.as_uri().replace('///', '//')
            }, f)  # not .as_uri() which adds "file://" as it also replaces "," and "@" but this is not that hydra need
        f.flush()

    sys_argv_save = sys.argv.copy()
    cfg = cfg_d.main_call([
        # r'--config-path=scripts\cfg',
        # fr'--config-path={path_db_raw.parent}/cfg_proc'.replace('\\', '/'),  # not works
        f'in.db_path="{db_in}"',
        'in.prefix=incl',   # incl_b
        'in.probes=[26,28,29,30]',   # 19, 6,7,8,9,10,11,12,16,
        'in.channels=[A, M]',  # [M]
        f'out.db_paths=["{db_in}"]',
        #'input=input_mod',
        #f'+probes=inclinometers',  # see config probes directory
        f'filter.A.offsets=[4,2]',    # less filter
        f'filter.M.offsets=[5,3]',  # less filter
        # can not miss "yml" as opposed to "yaml":
        # r'--config-name=220128incl_load#b-caliblab.yml'
        # r'# 220111incl_load#d-caliblab.yml'
        # '--info', 'searchpath'
        # '--cfg', 'job'  # hydra debug
        ],
        fun=incl_calibr_hy.main
        )
    sys.argv = sys_argv_save
    #incl_calibr(['cfg/201219incl_load-caliblab.yml'])
    """
    200813incl_calibr-lab-b.yml'])
    """

if st.end >= 15:
    # 1. tank data
    db_path_tank = path_on_drive_d(  # path to load calibration data: newer first
        r'd:\WorkData\_experiment\inclinometer\_type_b\230109_tube@ib26,28-30\_raw\230109.raw.h5'
        # r'd:\WorkData\_experiment\_2018\inclinometer\181004_tank[1-20]\_raw\181004.raw.h5'
        # r'd:\WorkData\_experiment\inclinometer\220525_tank\_raw\220525.raw.h5'
        # r'd:\WorkData\_experiment\inclinometer\_Schukas\200807_tank[b01-b30]\200807_calibr-tank-b.h5'
        # r'd:\WorkData\_experiment\inclinometer\220112_stand_D01\_raw\220112.raw.h5'
        # r'd:\WorkData\_experiment\inclinometer\210331_tank[4,5,9,10,11,19,28,33,36,37,38]\210331incl.h5'
        # r'd:\WorkData\_experiment\inclinometer\_Schukas\200807_tank[b01-b30]\200807_calibr-tank-b.h5'
        )
    probes = [28, 26, 29, 30]   # [2,3,6,7,8,9,10,11,12,16,19]
    tbl_prefix = 'incl_b'  # 'incl', 'i_d'

    if st(15, 'Copy laboratory calibration coefficients to other experiments databases'):
        for i, pnum in enumerate(probes):
            # incl_calibr not supports multiple time_ranges so calculate one by one pnum
            tbl = f'{tbl_prefix}{pnum:0>2}'
            print(f'Copying {tbl} coefficients from {path_db_raw}')
            h5copy_coef(path_db_raw, h5file_dest=db_path_tank, tbl=tbl, ok_to_replace_group=True)

    # For next steps

    # 2. copy coefs to:
    # (Usually needed copy to stand data - see input for 1st step and to common db. Else if stand db is same as
    # db_path_tank.or/and no need copy to other db - set it to empty list.
    db_path_copy = [path_on_drive_d(p) for p in [
            r'd:\WorkData\BalticSea\_Pregolya,Lagoon\221103@ib26,28-30\_raw\221103.raw.h5'
            # r'e:\WorkData\BalticSea\181005_ABP44\inclinometer\_raw\181017.raw.h5',
            # r'e:\WorkData\BalticSea\181005_ABP44\inclinometer\_raw\181022.raw.h5',
            # r'd:\WorkData\_experiment\_2018\inclinometer\181003_compas\181004.raw.h5'
            # r'd:\WorkData\~configuration~\inclinometer\190710incl.h5',
            # r'd:\WorkData\_experiment\inclinometer\_Schukas\210603_lab\_raw\220128.raw.h5' # 210603incl.h5'
            # r'd:\WorkData\BalticSea\220505_D6\inclinometers\_raw\220505.raw.h5'
            # r'd:\WorkData\_experiment\inclinometer\_Schukas\210603_lab\_raw\220128.raw.h5' # 210603incl.h5'
            ]
        ]

if st(20, 'Coefficients to convert inclination to |V| and zero calibration (not heading)'):
    """
    Note: Execute after updating Veusz data file with previous step results. You should:
    - update coefficients in hdf5 store that vsz imports (done in previous step)
    - recalculate calibration coefficients: zeroing (may be in vsz: zeroing interval in it must be set) and fit Velocity
    - save vsz
    Note: Updates Vabs coefs and zero calibration in data source for vsz (but this may not affect the Vabs coefs in vsz
    because of zero calibration is in vsz too).
    """
    from inclinometer.h5from_veusz_coef import main as h5from_veusz_coef

    # from veuszPropagate import __file__ as file_veuszPropagate

    # [36,37,38,3,4]  # range(26, 31)  # range(1, 31) [23,30,32] 17,18 [3,12,15,19,1,13,14,16] [1,4,5,7,11,12]  # [4,5,11,12]   #[29, 30, 33]  # [3, 14, 15, 16, 19]
    #channels_list = ['M', 'A']  # []

    # Output coefficients here:

    vsz_substr_not_in_tbl = '^[^@]+'  # 'tank@'
    vsz_data = {'veusze': None}
    for i, pnum in enumerate(probes):
        # incl_calibr not supports multiple time_ranges so calculate one by one pnum
        tbl = f'{tbl_prefix}{pnum:0>2}'  # note: regex result from veusz name by re_tbl_from_vsz_name below must be same
        # f'incl_b{pnum:0>2}'
        vsz_path = (
            db_path_tank.parent / 'vsz(range=1h)' / fr'*@ib{pnum:0>2}g'  # 230109_1404_13min@ib29g.vsz
            # f'{vsz_substr_not_in_tbl}i{pnum:0>2}'
            # tbl
            # f'i_{vsz_substr_not_in_tbl}d{pnum:0>2}'
            # {db_path_tank.stem}
            ).with_suffix('.vsz')
        vsz_data = h5from_veusz_coef([
            # str(Path(file_veuszPropagate).with_name('veuszPropagate.ini')),
            '--data_yield_prefix', 'Inclination',
            '--path', str(vsz_path),
            '--pattern_path', str(vsz_path),
            '--widget', '/fitV(incl)/grid1/graph/fit_t/values',
            # '/fitV(force)/grid1/graph/fit1/values',
            '--data_for_coef', 'max_incl_of_fit_t',
            '--out.path', str(db_path_tank),
            #'--re_match_tbl_from_vsz_name', f'[^_@\d]+_?\d+',
            '--re_sub_tbl_from_vsz_name', '^.*',  # r'\D+',
            '--to_sub_tbl_from_vsz_name', tbl,  # tbl_prefix
            '--channels_list', '',  # 'M,A',
            '--b_update_existed', 'True',  # to not skip.
            '--export_pages_int_list', '',  #4 0 = all
            '--b_interact', 'False',
            '--b_execute_vsz', 'True',
            '--return', '<embedded_object>',  # reuse to not bloat memory
            ],
            veusze=vsz_data['veusze'])

        def any_inside(v):
            if isinstance(v, list):
                return any(any_inside(e) for e in v)
            try:
                return any(v)
            except ValueError:  # The truth value of an array with more than one element is ambiguous
                return v.any()

        if vsz_data is not None:
            if any(any_inside(v) for k, v in vsz_data.items() if k != 'veusze'):
                for db in db_path_copy:
                    # if step == 3:
                    # to 1st db too
                    # l = init_logging('')
                    print(f'Copy coefficients to {db}/{tbl} from {db_path_tank}')
                    h5copy_coef(db_path_tank, db, tbl, ok_to_replace_group=True)

            vsz_data['veusze'].Close()
            try:
                vsz_data['veusze'].WaitForClose()
            except AttributeError:  # already 'NoneType' => closed ok
                pass
        else:
            vsz_data = {'veusze': None}


if st(35, f'Copy calibration coefficients from {db_path_tank} to other databases'):

    init_logging(logger=__name__)
    db_path_copy.append(path_db_raw)  # db_path_copy = [path_db_raw]
    db_path_copy.append(r'd:\Work\_Python3\And0K\h5toGrid\inclinometer\tests\data\inclinometer\incl#b.h5')

    for db in db_path_copy:
        print('to', db)
        for i, pnum in enumerate(probes):
            # incl_calibr not supports multiple time_ranges so calculate one by one pnum
            tbl = f'{tbl_prefix}{pnum:0>2}'
            print(f'Copying {tbl}')
            h5copy_coef(db_path_tank, h5file_dest=db, tbl=tbl, ok_to_replace_group=True)
