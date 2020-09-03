#!/usr/bin/env python3
# coding:utf-8
"""
  Author:  Andrey Korzh <ao.korzh@gmail.com>
  Purpose: Update inclinometr Vabs coef in hdf5 tables
  Created: 01.03.2019

Load coefs from Veusz fitting of force(inclination) to velocity data to hdf5 coef table
"""

import logging
import re
import sys
from pathlib import Path
from datetime import datetime
import h5py
import numpy as np
import pandas as pd

# my
sys.path.append(str(Path(__file__).parent.parent.resolve()))
from utils2init import cfg_from_args, this_prog_basename, init_file_names, init_logging, Ex_nothing_done, dir_from_cfg,\
    set_field_if_no, \
    path_on_drive_d
import veuszPropagate
from inclinometer.h5inclinometer_coef import h5copy_coef
from inclinometer.incl_calibr import channel_cols

if __name__ != '__main__':
    l = logging.getLogger(__name__)
else:
    l = None  # will set in main()


def atoi(text):
    return int(text) if text.isdigit() else text


def digits_first(text):
    return (lambda tdtd: (atoi(tdtd[1]), tdtd[0]))(re.split('(\d+)', text))


def main(new_arg=None):
    """
    Note: if vsz data source have 'Ag_old_inv' variable then not invert coef. Else invert to use in vsz which not invert coefs
    :param new_arg:
    :return:
    """
    global l
    p = veuszPropagate.my_argparser()
    p_groups = {g.title: g for g in p._action_groups if
                g.title.split(' ')[-1] != 'arguments'}  # skips special argparse groups
    p_groups['in'].add(
        '--channels_list',
        help='channels needed zero calibration: "magnetometer" or "M" for magnetometer and any else for accelerometer, use "M, A" for both, empty to skip '
        )
    p_groups['in'].add(
        '--widget',
        help='path to Veusz widget property which contains coefficients. For example "/fitV(force)/grid1/graph/fit1/values"'
        )
    p_groups['in'].add(
        '--data_for_coef', default='max_incl_of_fit_t',
        help='Veusz data to use as coef. If used with widget then this data is appended to data from widget'
        )

    p_groups['output_files'].add(
        '--output_files.path',
        help='path to db where write coef')
    p_groups['output_files'].add(
        '--re_tbl_from_vsz_name',
        help='regex to extract hdf5 table name from to Veusz file name (last used "\D*\d*")'
        # ? why not simly specify table name?
        )
    # todo:  "b_update_existed" arg will be used here for exported images. Check whether False works or prevent open vsz

    cfg = cfg_from_args(p, new_arg)

    if not Path(cfg['program']['log']).is_absolute():
        cfg['program']['log'] = str(
            Path(__file__).parent.joinpath(cfg['program']['log']))  # l.root.handlers[0].baseFilename
    if not cfg:
        return
    if new_arg == '<return_cfg>':  # to help testing
        return cfg

    l = init_logging(logging, None, cfg['program']['log'], cfg['program']['verbose'])
    veuszPropagate.l = l
    print('\n' + this_prog_basename(__file__), 'started', end=' ')
    if cfg['output_files']['b_images_only']:
        print('in images only mode.')
    try:
        print('Output pattern ')
        # Using cfg['output_files'] to store pattern information
        if not Path(cfg['in']['pattern_path']).is_absolute():
            cfg['in']['pattern_path'] = str(cfg['in']['path'].parent.joinpath(cfg['in']['pattern_path']))
        set_field_if_no(cfg['output_files'], 'path', cfg['in']['pattern_path'])
        cfg['output_files'] = init_file_names(cfg['output_files'], b_interact=cfg['program']['b_interact'])
    except Ex_nothing_done as e:
        print(e.message, ' - no pattern')
        return  # or raise FileNotFoundError?
    try:
        print(end='Data ')
        cfg['in'] = init_file_names(cfg['in'], b_interact=False)  # do not bother 2nd time
    except Ex_nothing_done as e:
        print(e.message)
        return  # or raise FileNotFoundError?
    if not cfg['output_files']['export_dir']:
        cfg['output_files']['export_dir'] = Path(cfg['output_files']['path']).parent
    if cfg['program']['before_next'] and 'restore_config' in cfg['program']['before_next']:
        cfg['in_saved'] = cfg['in'].copy()
    # cfg['loop'] = asyncio.get_event_loop()
    # cfg['export_timeout_s'] = 600
    dir_from_cfg(cfg['output_files'], 'export_dir')

    veuszPropagate.load_vsz = veuszPropagate.load_vsz_closure(cfg['program']['veusz_path'])
    gen_veusz_and_logs = veuszPropagate.load_to_veusz(veuszPropagate.ge_names(cfg), cfg, None)

    names_get = ['Inclination_mean_use1', 'logVext1_m__s']  # \, 'Inclination_mean_use2', 'logVext2_m__s'
    names_get_fits = ['fit']  # , 'fit2'
    vsz_data = {n: [] for n in names_get}
    for n in names_get_fits:
        vsz_data[n] = []

    # prepare collecting all coef in text also
    names_get_txt_results = ['fit1result']  # , 'fit2result'
    txt_results = {n: {} for n in names_get_txt_results}

    i_file = 0
    for veusze, log in gen_veusz_and_logs:
        if not veusze:
            continue
        i_file += 1
        print(i_file)
        if cfg['output_files']['re_tbl_from_vsz_name']:
            table = cfg['output_files']['re_tbl_from_vsz_name'].match(log['out_name']).group()
        else:
            table = re.sub('^[\d_]*', '', log['out_name'])  # delete all first digits (date part)

        for n in names_get:
            vsz_data[n].append(veusze.GetData(n)[0])
        for n in [cfg['in']['data_for_coef']]:
            vsz_data[n] = list(veusze.GetData(n)[0])

        # Save velocity coefficients into //{table}//coef//Vabs{i} where i - fit number enumeretad from 0
        for i, name_out in enumerate(names_get_fits):  # ['fit1', 'fit2']
            coef = veusze.Get(
                cfg['in']['widget'])  # veusze.Root['fitV(inclination)']['grid1']['graph'][name_out].values.val
            if 'a' in coef:
                coef_list = [coef[k] for k in ['d', 'c', 'b', 'a'] if k in coef]
            else:
                coef_list = [coef[k] for k in sorted(coef.keys(), key=digits_first)]
            if cfg['in']['data_for_coef']:
                coef_list += vsz_data[cfg['in']['data_for_coef']]

            vsz_data[name_out].append(coef_list)
            h5copy_coef(None, cfg['output_files']['path'], table,
                        dict_matrices={f'//coef//Vabs{i}': coef_list,
                                       f'//coef//date': np.float64(
                                           [np.NaN, np.datetime64(datetime.now()).astype(np.int64)])})
            # h5savecoef(cfg['output_files']['path'], path=f'//{table}//coef//Vabs{i}', coef=coef_list)
            txt_results[names_get_txt_results[i]][table] = str(coef)

        # Zeroing matrix - calculated in Veusz by rotation on old0pitch old0roll
        Rcor = veusze.GetData('Rcor')[0]  # zeroing angles tuned by "USEcalibr0V_..." in Veusz Custom definitions

        if len(cfg['in']['channels']):
            l.info('Applying zero calibration matrix of peach = {} and roll = {} degrees'.format(
                np.rad2deg(veusze.GetData('old0pitch')[0][0]),
                np.rad2deg(veusze.GetData('old0roll')[0][0])
                ))
            with h5py.File(cfg['output_files']['path'], 'a') as h5:
                for channel in cfg['in']['channels']:
                    (col_str, coef_str) = channel_cols(channel)
                    # h5savecoef(cfg['output_files']['path'], path=f'//{table}//coef//Vabs{i}', coef=coef_list), dict_matrices={'//coef//' + coef_str + '//A': coefs[tbl][channel]['A'], '//coef//' + coef_str + '//C': coefs[tbl][channel]['b']})

                    # Currently used inclinometers have electronics rotated on 180deg. Before we inserted additional
                    # rotation operation in Veusz by inverting A_old. Now we want iclude this information in database coef only.
                    try:  # Checking that A_old_inv exist
                        A_old_inv = veusze.GetData('Ag_old_inv')
                        is_old_used = True  # Rcor is not account for electronic is rotated.
                    except KeyError:
                        is_old_used = False  # Rcor is account for rotated electronic.

                    if is_old_used:  # The rotation is done in vsz (A_old in vsz is inverted) so need rotate it back to
                        # use in vsz without such invertion

                        # Rotate on 180 deg (note: this is not inversion)
                        A_old_inv = h5[f'//{table}//coef//{coef_str}//A'][...]
                        A_old = np.dot(A_old_inv, [[1, 0, 0], [0, -1, 0], [0, 0, -1]])  # adds 180 deg to roll
                    else:
                        A_old = h5[f'//{table}//coef//{coef_str}//A'][...]
                    # A_old now accounts for rotated electronic

                    A = np.dot(Rcor, A_old)
                    h5copy_coef(None, h5, table,
                                dict_matrices={f'//coef//{coef_str}//A': A})

        # veusze.Root['fitV(inclination)']['grid1']['graph2'][name_out].function.val
        print(vsz_data)
        veuszPropagate.export_images(veusze, cfg['output_files'], f"_{log['out_name']}",
         b_skip_if_exists=not cfg['output_files']['b_update_existed'])

        # vsz_data = veusz_data(veusze, cfg['in']['data_yield_prefix'])
        # # caller do some processing of data and gives new cfg:
        # cfgin_update = yield(vsz_data, log)  # to test run veusze.Save('-.vsz')
        # cfg['in'].update(cfgin_update)  # only update of cfg.in.add_custom_expressions is tested
        # if cfg['in']['add_custom']:
        #     for n, e in zip(cfg['in']['add_custom'], cfg['in']['add_custom_expressions']):
        #         veusze.AddCustom('definition', n, e, mode='replace')
        # #cor_savings.send((veusze, log))
        #
        #
        #
        #

    # veusze.Save(str(path_vsz_save), mode='hdf5')  # veusze.Save(str(path_vsz_save)) saves time with bad resolution
    print(f'Ok')
    print(txt_results)
    for n in names_get:
        pd.DataFrame.from_dict(dict(zip(list(txt_results['fit1result'].keys()), vsz_data[n]))
                               ).to_csv(
            Path(cfg['output_files']['path']).with_name(f'average_for_fitting-{n}.txt'), sep='\t',
            header=txt_results['fit1result'].keys, mode='a')


if __name__ == '__main__':
    cfg_out_db_path = path_on_drive_d(
        r'd:\WorkData\_experiment\_2019\inclinometer\190711_tank\190711incl.h5'
        # r'/mnt/D/workData/_experiment/_2019/inclinometer/190704_tank_ex2/190704incl.h5'
        )
    # r'd:\WorkData\_experiment\_2019\inclinometer\190704\190704incl.h5'
    # r'd:\workData\BalticSea\190713_ABP45\inclinometer\190816incl.h5'
    # 190711incl.h5   cfg['output_files']['db_path']

    cfg_in = {
        'path': cfg_out_db_path.with_name('190711incl12.vsz'),  # '190704incl[0-9][0-9].vsz'
        # r'd:\WorkData\_experiment\_2019\inclinometer\190704\190704incl[0-9][0-9].vsz',
        # r'd:\WorkData\_experiment\_2019\inclinometer\190711_tank\190711incl[0-9][0-9].vsz',
        # r'd:\WorkData\_experiment\_2018\inclinometr\181004_KTI\incl09.vsz'
        'widget': '/fitV(force)/grid1/graph/fit1/values'
        }
    # , 'output_files': {    'db_path': }

    # if not cfg_out_db_path.is_absolute():
    #     cfg_out_db_path = Path(cfg_in['path']).parent / cfg_out_db_path
    # d:\workData\_experiment\_2018\inclinometr\180731_KTI\*.vsz

    main([str(Path(veuszPropagate.__file__).with_name('veuszPropagate.ini')),
          '--data_yield_prefix', 'Inclination',
          '--path', str(cfg_in['path']),
          '--pattern_path', str(cfg_in['path']),
          '--output_files.path', str(cfg_out_db_path),
          '--channels_list', 'M,A',

          '--b_update_existed', 'True',  # to not skip.
          '--export_pages_int_list', '',
          ])
