import sys
from pathlib import Path

drive_d = 'D:/' if sys.platform == 'win32' else '/mnt/D/'  # allows to run on both my Linux and Windows systems:
scripts_path = Path(drive_d).joinpath('Work/_Python3/And0K/h5toGrid/scripts')
sys.path.append(str(Path(scripts_path).parent.resolve()))  # os.getcwd()

# my
import veuszPropagate
from utils2init import Ex_nothing_done, this_prog_basename
from to_pandas_hdf5.h5toh5 import h5.log_names_gen  # , h5.find_tables

# user's constants ############################################################
path_db = Path(drive_d).joinpath(
    r'workData\BalticSea\181005_ABP44\181005_ABP44.h5')


#########################################################################
def main():
    print('\n' + this_prog_basename(__file__), end=' started. ')

    cfg = veuszPropagate.main([
        Path(veuszPropagate.__file__).parent.with_name('veuszPropagate.ini'),
        '--data_yield_prefix', '-',
        '--path', str(path_db),  # use for custom loading from db and some source is required
        '--pattern_path', str(path_db.parent.joinpath(r'CTD_S&S48M#1253\profiles\181005_1810-1813.vsz')),
        # '--before_next', 'restore_config',
        # '--add_custom_list',
        # 'USEtime',
        # '--add_custom_expressions_list',
        # """
        # "[['2018-10-03T17:23:00', '2018-10-03T18:25:00']]"
        # """,
        '--b_update_existed', 'True',
        # '--export_pages_int_list', '',
        '--b_interact', '0',
        # '--b_images_only', 'True'
        ])
    if not cfg:
        return 1

    # Custom loading from db
    cfg['in']['table_log'] = '/CTD_SST_48M/logRuns'
    cfg['in']['db_path'] = path_db

    gen_veusz_and_logs = veuszPropagate.load_to_veusz(h5.log_names_gen(cfg['in']), cfg)
    cor_savings = veuszPropagate.co_savings(cfg);
    cor_savings.send(None)
    cor_send_data = veuszPropagate.co_send_data(gen_veusz_and_logs, cfg, cor_savings)
    cfgin_update = None
    while True:  # for vsz_data, log in cor_send_data.send(cfgin_update):
        try:
            vsz_data, log = cor_send_data.send(cfgin_update)
            # will delete cfg['in']['tables']
        except (GeneratorExit, StopIteration, Ex_nothing_done):
            print(f"{datetime.now():%Y-%m-%d %H:%M:%S} Ok>")
            break

        custom_expressions_use_time = "[['{Index:%Y-%m-%dT%H:%M:%S}', '{DateEnd:%Y-%m-%dT%H:%M:%S}']]".format_map(
            cfg['log_row'])
        if not cfg['out']['b_images_only']:
            cfgin_update = {
                'add_custom': ['USE_timeRange'],
                'add_custom_expressions': [custom_expressions_use_time]}


if __name__ == '__main__':
    main(
        # '--min_time', '2018-10-20T03:28:00+00:00',
        )
