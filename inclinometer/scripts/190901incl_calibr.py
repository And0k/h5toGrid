#!/usr/bin/env python3
# coding:utf-8
"""
  Author:  Andrey Korzh <ao.korzh@gmail.com>
  Purpose: Calibration of AB SIO RAS inclinometers
  Created: 26.02.2019
  Modified: 15.12.2019

  Step 2: update/create vsz of calibration in tank after executing 1st step: fit coef (again) and save vsz!
"""
import logging
import sys
from pathlib import Path

import pandas as pd

# my
sys.path.append(str(Path(__file__).parent.parent))  # to can import utils2init
from utils2init import path_on_drive_d
from inclinometer.incl_calibr import str_range, main as incl_calibr, zeroing_azimuth
from inclinometer.incl_h5clc import h5_names_gen
from inclinometer.h5inclinometer_coef import h5copy_coef
from inclinometer.h5from_veusz_coef import main as h5from_veusz_coef
from veuszPropagate import __file__ as file_veuszPropagate

l = logging.getLogger(__name__)

probes = [4,5,7,9,10,11,3,12,13,14,15,16,19] # [23,30,32] 17,18 [3,12,15,19,1,13,14,16] [1,4,5,7,11,12]  # [4,5,11,12]   #[29, 30, 33]  # [3, 14, 15, 16, 19]
channels_list = ['M', 'A']  # []

# stand data - input for 1st step
db_path_calibr_scalling = path_on_drive_d(
    r'd:\WorkData\_experiment\inclinometer\190710_compas_calibr-byMe\190710incl.h5'
    )

r"""
d:\WorkData\_experiment\_2018\inclinometer\181004_tank[1-20]\181004_KTIz.h5
    # r'd:/workData/_experiment/_2019/inclinometer/190704_tank_ex2/190704incl.h5'
    # r'd:\WorkData\_experiment\inclinometer\190711_tank\190711incl.h5'
    # r'd:\WorkData\_experiment\inclinometer\190711_tank\190711incl.h5'
    # r'/mnt/D/workData/_experiment/_2019/inclinometer/190704_tank_ex2/190704incl.h5'
"""

step = 2  # one step for one program's run
# ---------------------------------------------------------------------------------------------
for i, probe in enumerate(probes):  # incl_calibr not supports multiple time_ranges so calculate one by one probe
    # tank data - used to output coefficients in both steps
    db_path_tank = path_on_drive_d(  # path to load calibration data: newer first
        r'd:\WorkData\_experiment\inclinometer\200610_tank_ex[4,5,7,9,10,11][3,12,13,14,15,16,19]\200610_tank.h5' if probe in
[4,5,7,9,10,11,3,12,13,14,15,16,19] else
    r'd:\WorkData\_experiment\inclinometer\200117_tank[23,30,32]\200117_tank.h5' if probe in
[23,30,32] else
        r'd:\WorkData\_experiment\inclinometer\191106_tank_ex[1,13,14,16][3,12,15,19]\191106_tank_ex2.h5' if probe in
[3,12,15,19] else
        r'd:\WorkData\_experiment\inclinometer\191106_tank_ex[1,13,14,16][3,12,15,19]\191106_tank_ex1.h5' if probe in
[1,13,14,16] else
        r'd:\WorkData\_experiment\inclinometer\190711_tank[1,4,5,7,11,12]\190711incl.h5' if probe in
[1, 4, 5, 7,11,12] else
        r'd:\WorkData\_experiment\inclinometer\190704_tank_ex2[12,22,27,28,30,31,35]\190704incl.h5' if probe in
[22,27,28,30,31,35] else  # 12,
        r'd:\WorkData\_experiment\inclinometer\190704_tank_ex1[21,23,24,25,26,29,32,34]\190704incl.h5' if probe in
[21,23,24,25,26,29,32,34] else
        r'd:\WorkData\_experiment\_2018\inclinometer\181004_tank[1-20]\181004_KTIz.h5'
        # old DB with inverted M like new
        )  # r'd:\WorkData\_experiment\_2018\inclinometer\180406_tank[9]\180406_incl09.h5'

    tbl = f'incl{probe:0>2}'
    if step == 1:
        # Soft magnetic at the stand
        time_ranges = {
            23: ['2019-07-10T17:29:00', '2019-07-10T17:47:56'],
            32: ['2019-07-10T18:46:05', '2019-07-10T18:58:32'],
            30: ['2019-07-09T18:37:00', '2019-07-09T18:44:00', '2019-07-09T19:00:00', '2019-07-09T19:20:00'],

            29: ['2019-07-09T19:23:20', '2019-07-10T19:33:00'],
            33: ['2019-07-12T12:33:10', '2019-07-12T12:47:00'],

            12: ['2019-07-11T18:07:30', '2019-07-11T18:15:44', '2019-07-11T18:16:36', '2019-07-11T18:24:22'],
            5: ['2019-07-11T18:30:11', '2019-07-11T18:46:28'],
            4: ['2019-07-11T17:25:30', '2019-07-11T17:39:30'],
            11: ['2019-07-11T17:46:10', '2019-07-11T18:02:00'],

            3: ['2019-09-03T19:36:20', '2019-09-03T19:49:20'],
            14: ['2019-09-02T14:07:10', '2019-09-02T14:50:00'],
            15: ['2019-09-02T14:32:43', '2019-09-02T15:05:00'],
            16: ['2019-09-03T19:03:17', '2019-09-03T19:21:07'],
            19: ['2019-09-03T18:54:18', '2019-09-03T19:20:00'],

            1: ['2019-07-11T18:51:24', '2019-07-11T19:06:10'],
            13: ['2019-10-15T16:41:30', '2019-10-15T16:49:18', '2019-10-15T16:50:34', '2019-10-15T16:58:22'],
            7: ['2019-07-11T16:54:35', '2019-07-11T17:15:20'],
            9: ['2019-12-20T16:17:30', '2019-12-20T16:33:00', '2019-12-20T16:35:32', '2019-12-20T16:45:30'],
            10: ['2019-12-23T17:00:40', '2019-12-23T17:28:10'],
            }

        incl_calibr([
            '',
            '--db_path', str(db_path_calibr_scalling),
            '--channels_list', ','.join(channels_list),  # 'M,', Note: empty element cause calc of accelerometer coef.
            '--tables_list', tbl,
            '--time_range_list', str_range(time_ranges, probe),
            # '--time_range_nord_list', str_range(time_ranges_nord, probe),
            '--out.db_path', str(db_path_tank),  # save here addititonally
            ])

    if step == 2:
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
                h5copy_coef(db_path_copy, db_imput_to_vsz_path, tbl,
                            dict_matrices={'//coef//' + coef_str + '//A': coefs[tbl][channel]['A'],
                                           '//coef//' + coef_str + '//C': coefs[tbl][channel]['b']})
        """
        # f'190711incl{probe:0>2}.vsz', '190704incl[0-9][0-9].vsz'
        vsz_path = db_path_tank.with_name(f'incl{probe:0>2}_.vsz')  # {db_path_tank.stem}
        h5from_veusz_coef([str(Path(file_veuszPropagate).with_name('veuszPropagate.ini')),
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
                           '--export_pages_int_list', '0',  # 0 = all
                           '--b_interact', 'False'
                           ])
        # if step == 3:
        # to 1st db too
        # l = init_logging('')
        l.info(f"Adding coefficients to {db_path_calibr_scalling}/{tbl} from {db_path_tank}")
        h5copy_coef(db_path_tank, db_path_calibr_scalling, tbl, ok_to_replace_group=True)

    if step == 3:
        time_ranges_nord = {
            1: ['2019-07-11T18:48:35', '2019-07-11T18:49:20'],
            #  7: ['2019-07-11T16:53:40', '2019-07-11T16:54:10'], ???
            # 30: ['2019-07-09T17:54:50', '2019-07-09T17:55:22'],
            4: ['2019-07-11T17:22:15', '2019-07-11T17:23:08'],
            5: ['2019-07-11T18:27:10', '2019-07-11T18:27:48'],
            9: ['2019-12-20T16:58:30', '2019-12-20T16:59:15'],
            10: ['2019-12-23T17:32:35', '2019-12-23T17:33:27'],
            11: ['2019-07-11T17:41:44', '2019-07-11T18:42:48'],
            12: ['2019-07-11T18:04:46', '2019-07-11T18:05:36'],
            14: ['2019-09-02T14:01:41', '2019-09-02T14:02:15'],  # todo
            16: ['2019-09-03T19:22:20', '2019-09-03T19:22:54'],

            }
        if time_ranges_nord.get(probe):
            # Recalc zeroing_azimuth with zeroed scaling coefs
            cfg_in = {
                'tables': [tbl],
                'db_path': db_path_tank,
                'time_range_nord': time_ranges_nord[probe]}
            with pd.HDFStore(db_path_calibr_scalling, mode='r') as store:
                for tbl, coefs in h5_names_gen(cfg_in):
                    del coefs['azimuth_shift_deg']  # to calculate shift of uncorrected data
                    # Calculation:
                    dict_matrices = {'//coef//H//azimuth_shift_deg': zeroing_azimuth(
                        store, tbl, time_ranges_nord[probe], coefs, cfg_in)}
            h5copy_coef(None, db_path_tank, tbl, dict_matrices=dict_matrices)
            h5copy_coef(db_path_tank, db_path_calibr_scalling, tbl, ok_to_replace_group=True)
        else:
            l.warning('Inlab time_ranges_nord not defined')
