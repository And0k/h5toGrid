from inclinometer.incl_load import main as incl_load
# incl_load([
#     'cfg/210603incl_load#b-caliblab.yml',
# ])



incl_load([
    'cfg/200907incl_load#b.yml',
    '--raw_subdir', '{prefix:}{number:0>2}.ZIP',
    #'--min_date_dict', '2: 2020-09-09T15:30:00',
    '--step_start_int', '2',
    #'--aggregate_period_s_int_list', 'None,2,600,7200',  #'',  #'None,2,600,7200',
    '--aggregate_period_s_not_to_text_int_list', '',  #  '2,600,7200',
    '--split_period', '',
    #'--probes_int_list', '2',  #'3,4,8,9,11,14,17,21,22', #'5,7,19,24', #'11', #
    #'--dask_scheduler', 'synchronous'

# '../tests/data/inclinometer/200813incl_load#b-sent.yml'
# 'cfg/200813incl_load#b-paths_relative_to_scripts.yml', #'--step_start_int', '2'
# 'cfg/200813incl_load-calibtank#b20.yml'
# 'cfg/200813incl_load-caliblab-b.yml'
# 'cfg/200813incl_load-calibtank-b.yml'
])

# import sys
# from pathlib import Path
# import my scripts
# drive_d = Path('D:/' if sys.platform == 'win32' else '/mnt/D')  # allows to run on both my Linux and Windows systems:
# module_path = drive_d.joinpath('Work/_Python3/And0K/h5toGrid')
# sys.path.append(str(Path(module_path).resolve()))  # os.getcwd()