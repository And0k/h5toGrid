from os import path as os_path

import numpy as np
import pandas as pd

from h5toGpx import init_gpx_symbols_fun, save_to_gpx, my_argparser as h5toGpx_parser
from to_pandas_hdf5.csv_specific_proc import convertNumpyArrayOfStrings
from to_pandas_hdf5.h5toh5 import h5select
# my:
from utils2init import init_file_names, cfg_from_args, this_prog_basename, Ex_nothing_done, standard_error_info


def my_argparser():
    p = h5toGpx_parser()
    # Append arguments with my common options:
    s = [g for g in p._action_groups if g.title == 'in'][0]
    s.add('--path', default='.',  # nargs=?,
             help='path to source file(s) to parse. Use patterns in Unix shell style')
    return p


def ge_names(cfg):
    for in_full in cfg['in']['paths']:
        # ifile += 1
        inFE = os_path.basename(in_full)
        inF = os_path.splitext(inFE)[0].encode('ascii')
        yield inF


def filename2date(inF):
    """

    :param inF: base name of source data file (20171015_090558p1) yyyymmdd_HHMMSS*
    :param cfg: dict with keys:
        out:
            dir, paths - pattern path
            b_images_only, b_update_existed - see command line arguments,
        in:
            import_method, header if import_method == 'ImportFile',
            add_custom
    :param g:
    :return:
    """
    a = np.array(inF, dtype={'yyyy': ('a4', 0), 'mm': ('a2', 4), 'dd': ('a2', 6), 'HH': ('a2', 9), 'MM': ('a2', 11),
                             'SS': ('a2', 13)})
    try:
        date = np.array(
            a['yyyy'].astype(np.object) + b'-' + a['mm'].astype(
                np.object) + b'-' + a['dd'].astype(np.object) + b'T' +
            a['HH'].astype(np.object) + b':' +
            a['MM'].astype(np.object) + b':' +
            a['SS'].astype(np.object), '|S19', ndmin=1)
        # date = b'%(yyyy)b-%(mm)b-%(dd)bT%(HH)02b-%(MM)02b-%(SS)02b' % a
    except Exception as e:
        print('Can not convert date: ', standard_error_info(e))
        raise e
    return convertNumpyArrayOfStrings(date, 'datetime64[ns]')


# , '--path', r'd:\workData\BalticSea\171003_ANS36\Baklan\2017*p1d5.txt',
def main(new_arg=None):
    new_arg = [r'.\h5toGpx_CTDs.ini',
               '--db_path', r'd:\workData\BalticSea\170614_ANS34\170614Strahov.h5',
               '--path', r'd:\workData\BalticSea\170614_ANS34\Baklan\2017*p1d5.txt',
               '--gpx_names_fun_format', '+{:02d}',
               '--gpx_symbols_list', "'Navaid, Orange'"
               ]  # 'db_path', r'd:\workData\BalticSea\171003_ANS36\171003Strahov.h5'
    cfg = cfg_from_args(my_argparser(), new_arg)
    if not cfg:
        return
    if new_arg == '<return_cfg>':  # to help testing
        return cfg
    print('\n' + this_prog_basename(__file__), 'started', end=' ')

    if not cfg['out']['path'].is_absolute():
        cfg['out']['path'] = cfg['in']['db_path'].parent / cfg['out']['path']  # set relative to cfg['in']['db_path']

    try:
        print(end='Data ')
        cfg['in']['paths'], cfg['in']['nfiles'], cfg['in']['path'] = init_file_names(**cfg['in'])  # may interact
    except Ex_nothing_done as e:
        print(e.message)
        return  # or raise FileNotFoundError?

    itbl = 0
    # compile functions if defined in cfg or assign default
    gpx_symbols = init_gpx_symbols_fun(cfg['out'])
    gpx_names_funs = ["i+1"]
    gpx_names_fun = eval(compile("lambda i, row: '{}'.format({})".format(
        cfg['out']['gpx_names_fun_format'],
        gpx_names_funs[itbl]), [], 'eval'))

    tim = filename2date([f for f in ge_names(cfg)])

    with pd.HDFStore(cfg['in']['db_path'], mode='r') as storeIn:
        # dfL = storeIn[tblD + '/logFiles']
        nav2add = h5select(storeIn, cfg['in']['table_nav'], ['Lat', 'Lon', 'DepEcho'], tim)[0]
        rnav_df_join = nav2add.assign(itbl=itbl)  # copy/append on first/next cycle
        # Save to gpx waypoints

        # if 'gpx_names_funs' in cfg['out'] and \
        #     len(cfg['out']['gpx_names_funs'])>itbl:
        #
        #     gpx_names = eval(compile('lambda i: str({})'.format(
        #         cfg['out']['gpx_names_funs'][itbl]), [], 'eval'))
        #
        save_to_gpx(rnav_df_join[-len(nav2add):], cfg['out']['path'].with_name('fileNames'),
                    gpx_obj_namef=gpx_names_fun, waypoint_symbf=gpx_symbols, cfg_proc=cfg['process'])


if __name__ == '__main__':
    main()
