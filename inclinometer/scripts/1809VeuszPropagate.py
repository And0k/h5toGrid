import sys
from pathlib import Path

drive_d = 'D:' if sys.platform == 'win32' else '/mnt/D'  # allows to run on both my Linux and Windows systems:
scripts_path = Path(drive_d + '/Work/_Python3/And0K/h5toGrid/scripts')
sys.path.append(str(Path(scripts_path).parent.resolve()))  # os.getcwd()

import pandas as pd
# my
import veuszPropagate
from utils2init import Ex_nothing_done, this_prog_basename

# ###################################################################################
path_cruise = Path(drive_d + '/workData/_experiment/_2018/inclinometr/180905_clockCheck/raw')


def main():
    print('\n' + this_prog_basename(__file__), end=' started. ')

    import_pattern = """
        ImportFileHDF5(u'180905_clockCheck.h5', [
        u'/{probe}/coef/G/A', u'/{probe}/coef/G/C', u'/{probe}/coef/H/A', u'/{probe}/coef/H/C', u'/{probe}/table'],
        linked=True,
        namemap={{u'/{probe}/coef/G/A': u'Ag_old', u'/{probe}/coef/G/C': u'Cg', u'/{probe}/coef/H/A': u'Ah_old',
                 u'/{probe}/coef/H/C': u'Ch', u'/{probe}/table/Ax': u'countsAx',
                 u'/{probe}/table/Ay': u'countsAy', u'/{probe}/table/Az': u'countsAz',
                 u'/{probe}/table/Mx': u'countsMx', u'/{probe}/table/My': u'countsMy',
                 u'/{probe}/table/Mz': u'countsMz'}}, renames={{u'Temp': u'sT'}})
        """.replace('{{', '{{{{').replace('}}', '}}}}')

    cfg = veuszPropagate.main([
        Path(veuszPropagate.__file__).parent.with_name('veuszPropagate.ini'),
        '--data_yield_prefix', '-',
        '--path', str(path_cruise.joinpath(r'incl*.txt')),
        '--pattern_path', str(path_cruise.with_name('180905_1320incl01.vsz')),
        # '--import_method', 'ImportFileCSV',
        '--before_next', 'restore_config',
        # '--add_custom_list',
        # 'USEtime',
        # '--add_custom_expressions_list',
        # """
        # "[['2018-09-05T13:19:55', '2018-09-05T13:20:25']]"
        # """,
        '--export_pages_int_list', '6,7',
        '--b_interact', '0',
        '--b_images_only', 'True'
        ])
    if not cfg:
        return 1
    # veuszPropagate.load_vsz = cfg['load_vsz']

    log_times = pd.to_datetime([
        '05.09.2018 13:20',  # 05.09.2018 13:20-00 первое качание
        '07.09.2018 15:48',  # 07.09.2018 15:48-00 второе качание
        '11.09.2018 12:59'  # 11.09.2018 12:59-00 третье качание
        ], dayfirst=True)
    t_log_list = pd.to_datetime(log_times)

    dt_to_st = pd.Timedelta(seconds=-5)
    dt_to_en = pd.Timedelta(seconds=30)
    custom_expressions_use_time = ["[['{:%Y-%m-%dT%H:%M:%S}', '{:%Y-%m-%dT%H:%M:%S}']]".format(
        t_log + dt_to_st, t_log + dt_to_en) for t_log in t_log_list]
    # u"[['2018-09-05T13:19:55', '2018-09-05T13:20:25']]",
    # u"[['2018-09-07T15:47:55', '2018-09-07T15:48:25']]",
    # u"[['2018-09-11T13:58:55', '2018-09-11T13:59:55']]",

    cor_savings = veuszPropagate.co_savings(cfg);
    cor_savings.send(None)

    for t_log, expr in zip(t_log_list, custom_expressions_use_time):
        name_add_time = '_{:%y%m%d_%H%M}-{:%H%M}'.format(t_log + dt_to_st, t_log + dt_to_en)
        print('Processing group {}...'.format(name_add_time))

        def f_file_name(file_name):
            p = Path(file_name)
            return str(p.with_name(p.stem + name_add_time + p.suffix))

        gen_veusz_and_logs = veuszPropagate.load_to_veusz(veuszPropagate.ge_names(cfg, f_file_name), cfg)
        cor_send_data = veuszPropagate.co_send_data(gen_veusz_and_logs, cfg, cor_savings)
        cfgin_update = None
        while True:  # for vsz_data, log in cor_send_data.send(cfgin_update):
            try:
                vsz_data, log = cor_send_data.send(cfgin_update)
            except (GeneratorExit, StopIteration, Ex_nothing_done):
                print('ok>')
                break
            # except Exception as e:
            #     print('There are error: ', standard_error_info(e))
            #     break  # continue
            probe = log['fileName'][:log['fileName'].find('_')]
            if not cfg['output_files']['b_images_only']:
                cfgin_update = {
                    'add_custom': ['USEtime'],
                    'add_custom_expressions': [expr],
                    'eval': [import_pattern.format(probe=probe)]}


if __name__ == '__main__':
    main()
