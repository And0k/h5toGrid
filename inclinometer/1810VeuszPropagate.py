import sys
from pathlib import Path

drive_d = 'D:/' if sys.platform == 'win32' else '/mnt/D/'  # allows to run on both my Linux and Windows systems:
scripts_path = Path(drive_d).joinpath('Work/_Python3/And0K/h5toGrid/scripts')
sys.path.append(str(Path(scripts_path).parent.resolve()))  # os.getcwd()

import pandas as pd
# my
import veuszPropagate
from utils2init import Ex_nothing_done, this_prog_basename
from to_pandas_hdf5.h5toh5 import h5find_tables

# user's constants ############################################################
path_db = Path(drive_d).joinpath(
    r'workData\BalticSea\181005_ABP44\181005_ABP44.h5'
    # 'WorkData/_experiment/_2018/inclinometr/181004_KTI/181004_KTI.h5'
    )
str_date = '181022'


# noinspection Annotator
def main():
    print('\n' + this_prog_basename(__file__), end=' started. ')

    import_pattern = """
ImportFileHDF5(u'../181005_ABP44.h5', [u'/181017inclinometers/incl03/coef', u'/181017inclinometers/incl03/table'], linked=True, namemap={u'/181017inclinometers/incl03/coef/G/A': u'Ag_old_inv', u'/181017inclinometers/incl03/coef/G/C': u'Cg', u'/181017inclinometers/incl03/coef/H/A': u'Ah_old_inv', u'/181017inclinometers/incl03/coef/H/C': u'Ch', u'/181017inclinometers/incl03/table/Ax': u'countsAx', u'/181017inclinometers/incl03/table/Ay': u'countsAy', u'/181017inclinometers/incl03/table/Az': u'countsAz', u'/181017inclinometers/incl03/table/Mx': u'countsMx', u'/181017inclinometers/incl03/table/My': u'countsMy', u'/181017inclinometers/incl03/table/Mz': u'countsMz', u'/181017inclinometers/incl03/table/Temp': u'sT'}, renames={u'Vabs0': u'kVabs'})
""".replace('{', '{{{{').replace('}', '}}}}').replace('incl03', '{probe}')

    cfg = veuszPropagate.main([
        Path(veuszPropagate.__file__).parent.with_name('veuszPropagate.ini'),
        '--data_yield_prefix', '-',
        '--path', str(path_db),  # use for custom loading from db and some source is required
        '--tables_list', 'incl{}',
        '--pattern_path', str(path_db.parent.joinpath(fr'inclinometer\*.vsz')),  # {str_date}incl03
        '--before_next', 'restore_config',
        # '--add_custom_list',
        # 'USEtime',
        # '--add_custom_expressions_list',
        # """
        # "[['2018-10-03T17:23:00', '2018-10-03T18:25:00']]"
        # """,
        '--b_update_existed', 'True',
        '--export_pages_int_list', '0',  # ''4   ',  #''
        '--b_interact', '0',
        '--b_images_only', 'True'
        ])
    if not cfg:
        return 1
    cor_savings = veuszPropagate.co_savings(cfg);
    cor_savings.send(None)

    # Custom loading from db
    cfg['in']['db_parent_path'] = f'{str_date}inclinometers'
    if not 'out' in cfg:
        cfg['out'] = {}
    cfg['out']['f_file_name'] = lambda tbl: f'{str_date}{tbl}'

    def ge_names(cfg, f_file_name=lambda x: x):
        """
        Replasing for veuszPropagate.ge_names() to use tables instead files
        :param cfg:
        :return:
        """
        with pd.HDFStore(cfg['in']['path'], mode='r') as store:
            if len(cfg['in']['tables']) == 1:
                cfg['in']['tables'] = h5find_tables(store, cfg['in']['tables'][0],
                                                    parent_name=cfg['in']['db_parent_path'])
        for tbl in cfg['in']['tables']:
            # if int(tbl[-2:]) in {5,9,10,11,14,20}:
            yield f_file_name(tbl)

    # def f_file_name(file_name):
    #     p = Path(file_name)
    #     return str(p.with_name(p.stem + name_add_time + p.suffix))

    gen_veusz_and_logs = veuszPropagate.load_to_veusz(ge_names(cfg, cfg['out']['f_file_name']), cfg)
    cor_send_data = veuszPropagate.co_send_data(gen_veusz_and_logs, cfg, cor_savings)
    cfgin_update = None
    while True:  # for vsz_data, log in cor_send_data.send(cfgin_update):
        try:
            vsz_data, log = cor_send_data.send(cfgin_update)
            # will delete cfg['in']['tables']
        except (GeneratorExit, StopIteration, Ex_nothing_done):
            print('ok>')
            break
        # i_ = log['fileName'].rfind('incl')
        probe = log['fileName'].replace(str_date,
                                        '')  # re.sub('^[\d_]*', '', in_file.stem),  # use last digits (date part) log['fileName'][:(i_)]
        if not cfg['output_files']['b_images_only']:
            cfgin_update = {
                'add_custom': ['USEtime'],
                # 'add_custom_expressions': [expr],
                'eval': [import_pattern.format(probe=probe)]}


if __name__ == '__main__':
    main()
