import sys
from pathlib import Path
# import my scripts
drive_d = Path('D:/' if sys.platform == 'win32' else '/mnt/D')  # allows to run on both my Linux and Windows systems:
scripts_path = drive_d.joinpath('Work/_Python3/And0K/h5toGrid/scripts')
sys.path.append(str(Path(scripts_path).parent.resolve()))  # os.getcwd()

from inclinometer.incl_calibr import main as incl_calibr


step = 2

if step == 1:
    incl_calibr(['ini/200813incl_calibr-lab-b.yml'])

if step == 2:
    import numpy as np
    from utils2init import path_on_drive_d
    from inclinometer.h5from_veusz_coef import main as h5from_veusz_coef
    from inclinometer.h5inclinometer_coef import h5copy_coef
    from veuszPropagate import __file__ as file_veuszPropagate


    probes = np.arange(1,30)  # [23,30,32] 17,18 [3,12,15,19,1,13,14,16] [1,4,5,7,11,12]  # [4,5,11,12]   #[29, 30, 33]  # [3, 14, 15, 16, 19]
    channels_list = ['M', 'A']  # []

    # stand data - input for 1st step
    db_path_calibr_scalling = path_on_drive_d(
        r'd:\WorkData\_experiment\_2019\inclinometer\200807_Schukas\mag_components_calibration\200807_calibr-lab-b.h5'
        )
    # tank data - used to output coefficients in both steps
    db_path_tank = path_on_drive_d(  # path to load calibration data: newer first
        r'd:\WorkData\_experiment\_2019\inclinometer\200807_Schukas\200807_calibr-tank-b.h5')

    """ ### Coefs to convert inclination to |V| and zero calibration (except of heading) ###

    Note: Execute after updating Veusz data with previous step results. You should
    - update coefficients in hdf5 store that vsz imports (done in previous step)
    - recalculate calibration coefficients: zeroing (automaitic if done in same vsz) and fit Velocity
    - save vsz
    Note: Updates Vabs coefs and zero calibration in source for vsz, but this should not affect the Vabs coefs in vsz
    because of zero calibration in vsz too and because it not uses Vabs coefs.

    tables_list = [f'incl{probe:0>2}' for probe in probes]
    print(f"Write to {db_imput_to_vsz_path}")
    for itbl, tbl in enumerate(tables_list, start=1):
        for channel in channels_list:
            (col_str, coef_str) = channel_cols(channel)
            h5copy_coef(db_path_calibr_scalling, db_imput_to_vsz_path, tbl,
                        dict_matrices={'//coef//' + coef_str + '//A': coefs[tbl][channel]['A'],
                                       '//coef//' + coef_str + '//C': coefs[tbl][channel]['b']})
    """

    for i, probe in enumerate(probes):  # incl_calibr not supports multiple timeranges so calculate one by one probe
        # tank data - used to output coefficients in both steps
        db_path_tank = path_on_drive_d(  # path to load calibration data: newer first
            r'd:\WorkData\_experiment\_2019\inclinometer\200610_tank_ex[4,5,7,9,10,11][3,12,13,14,15,16,19]\200610_tank.h5'

        # f'190711incl{probe:0>2}.vsz', '190704incl[0-9][0-9].vsz'
        vsz_path = db_path_tank.with_name(f'incl{probe:0>2}_.vsz')  # {db_path_tank.stem}
        h5from_veusz_coef([str(Path(file_veuszPropagate).with_name('veuszPropagate.ini')),
                           '--data_yield_prefix', 'Inclination',
                           '--path', str(vsz_path),
                           '--pattern_path', str(vsz_path),
                           '--widget', '/fitV(incl)/grid1/graph/fit_t/values',
                           # '/fitV(force)/grid1/graph/fit1/values',
                           '--data_for_coef', 'max_incl_of_fit_t',
                           '--output_files.path', str(db_path_tank),
                           '--re_tbl_from_vsz_name', '\D*\d*',
                           '--channels_list', 'M,A',
                           '--b_update_existed', 'True',  # to not skip.
                           '--export_pages_int_list', '0',  # 0 = all
                           '--b_interact', 'False'
                           ])
        # if step == 3:
        # to 1st db too
        # l = init_logging(logging, None)
        l.info(f"Adding coefficients to {db_path_calibr_scalling}/{tbl} from {db_path_tank}")
        h5copy_coef(db_path_tank, db_path_calibr_scalling, tbl, ok_to_replace_group=True)
