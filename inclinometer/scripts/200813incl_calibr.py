import sys
from pathlib import Path
# import my scripts
drive_d = Path('D:/' if sys.platform == 'win32' else '/mnt/D')  # allows to run on both my Linux and Windows systems:
scripts_path = drive_d.joinpath('Work/_Python3/And0K/h5toGrid/scripts')
sys.path.append(str(Path(scripts_path).parent.resolve()))  # os.getcwd()

from inclinometer.incl_calibr import main as incl_calibr


step = 2  # 1


if step == 1:
    incl_calibr(['ini/201219incl_load-caliblab.yml'])
    """
    200813incl_calibr-lab-b.yml'])
    """


if step == 2:
    """ ### Coefs to convert inclination to |V| and zero calibration (not heading) ###

    Note: Execute after updating Veusz data with previous step results. You should
    - update coefficients in hdf5 store that vsz imports (done in previous step)
    - recalculate calibration coefficients: zeroing (automaitic if done in same vsz) and fit Velocity
    - save vsz
    Note: Updates Vabs coefs and zero calibration in source for vsz, but this should not affect the Vabs coefs in vsz
    because of zero calibration in vsz too and because it not uses Vabs coefs.
    """
    import numpy as np
    from utils2init import path_on_drive_d
    from inclinometer.h5from_veusz_coef import main as h5from_veusz_coef
    from inclinometer.h5inclinometer_coef import h5copy_coef
    from veuszPropagate import __file__ as file_veuszPropagate


    probes = np.arange(10,40)  #[23,30,32] 17,18 [3,12,15,19,1,13,14,16] [1,4,5,7,11,12]  # [4,5,11,12]   #[29, 30, 33]  # [3, 14, 15, 16, 19]
    channels_list = ['M', 'A']  # []

    # Output coefficients here:
    # 1. stand data (same as input for 1st step)
    db_path_calibr_scalling = path_on_drive_d(
        r'D:\WorkData\~configuration~\inclinometr\190710incl.h5'
        # r'd:\WorkData\_experiment\inclinometer\200807_Schukas\mag_components_calibration\200807_calibr-lab#b.h5'
        )
    # 2. tank data
    db_path_tank = path_on_drive_d(  # path to load calibration data: newer first
        r'd:\WorkData\_experiment\inclinometer\210331_tank[4,5,9,10,11,19,28,33,36,37,38]\210331incl.h5'
        # r'd:\WorkData\_experiment\inclinometer\200807_Schukas\200807_calibr-tank-b.h5'
        )

    vsz_data = {'veusze': None}
    for i, probe in enumerate(probes):
        # incl_calibr not supports multiple timeranges so calculate one by one probe
        tbl = f'incl{probe:0>2}'  # note: regex result from veusz name by re_tbl_from_vsz_name below must be same
        # f'incl_b{probe:0>2}'
        vsz_path = db_path_tank.with_name(f'{tbl}.vsz')  # {db_path_tank.stem}
        vsz_data = h5from_veusz_coef(
            [str(Path(file_veuszPropagate).with_name('veuszPropagate.ini')),
            '--data_yield_prefix', 'Inclination',
            '--path', str(vsz_path),
            '--pattern_path', str(vsz_path),
            '--widget', '/fitV(incl)/grid1/graph/fit_t/values',
            # '/fitV(force)/grid1/graph/fit1/values',
            '--data_for_coef', 'max_incl_of_fit_t',
            '--out.path', str(db_path_tank),
            '--re_tbl_from_vsz_name', '\D*\d*',
            '--channels_list', 'M,A',
            '--b_update_existed', 'True',  # to not skip.
            '--export_pages_int_list', '',  #4 0 = all
            '--b_interact', 'False',
            '--b_execute_vsz', 'True',
            '--return', '<embedded_object>',  # reuse to not bloat memory
            ],
            veusze=vsz_data['veusze'])


        if vsz_data is not None:
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
