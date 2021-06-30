# import my scripts
from inclinometer.incl_load import main as incl_load



for st_en in [
    '15:56-16:35',
    '14:38-15:28',
    '13:16-13:49',
    '12:28-13:01',
    '10:11-11:44']:
    st, en = st_en.split('-')
    incl_load([
        'cfg/210519incl_load_Pregolya,Lagoon.yml',
        '--min_date_dict', f'0: 2021-05-19T{st}:00',
        '--max_date_dict', f'0: 2021-05-19T{en}:50',  # '16: 2020-10-21T16:10:33, 15: 2020-10-21T17:46:05',
        '--step_start_int', '2',  # '50', #'2',
        '--step_end_int', '2',  # '2',
        '--dask_scheduler', 'synchronous'
        ])


# incl_load([
#     # 'cfg/incl_load_201202_BalticSpit.yml',
#     'cfg/210519incl_load_Pregolya,Lagoon.yml',
#     '--min_date_dict', '0: 2020-12-18T13:28:00',
#     '--max_date_dict', '0: 2020-12-18T15:29:50',  # '16: 2020-10-21T16:10:33, 15: 2020-10-21T17:46:05',
#
#     # # wavegage:
#     # 'cfg/incl_load_201202_BalticSpit#w.yml',
#
#     # probe with 1800s burst period:
#     # '--probes_int_list', '04, 14, 36, 37, 38',  # '28, 33',  #3, 5, 9, 10, 11, 15, 19'
#     # '--aggregate_period_s_int_list', '2, 600, 1800, 7200',  # '',  # None,  300,  3600
#     #'--aggregate_period_s_not_to_text_int_list', 'None, 2',       # default=None, use 2 for speedup. 300, 7200, ,3600  # exclude too long saving op
#     '--step_start_int', '2',  # '50', #'2',
#     '--step_end_int',   '2',  # '2',
#
#      #'--load_timeout_s_float', '0'
#     # '--dask_scheduler', 'distributed'  # 'synchronous'  #   may be not affect incl_h5clc
# ])


"""    
    'cfg/200901incl_load.yml',
    '--path_cruise', r'd:\WorkData\_experiment\inclinometer\190710_compas_calibr-byMe',
#
#r'd:\WorkData\BalticSea\_Pregolya,Lagoon\200918-inclinometer'
#r'd:\WorkData\BalticSea\200915_Pionerskiy',
# r'd:\workData\BalticSea\200630_AI55',
    '--probes_int_list', '36,37,38',  #'3,15,16,10,4', 2,'10,15',
    '--raw_subdir', '201218', #'*.ZIP'
   # 'cfg/200901incl_load#wavegege.yml',
   # '--dask_scheduler', 'synchronous',
    '--step_start_int', '1', '--step_end_int', '1',
    '--min_date_dict', '0: 2020-12-18T13:28:00',    # '0: 2020-09-15T14:00:00',  10: 2020-09-10T10:00:00, 15: 2020-09-10T10:00:00
    '--max_date_dict', '0: 2020-12-18T15:29:50',    # '16: 2020-10-21T16:10:33, 15: 2020-10-21T17:46:05',
    # '--time_start_utc_dict', '15: 2020-09-15T06:42:00',
    # '--dt_from_utc_days_dict', '10: -7',
   # '--aggregate_period_s_not_to_text_int_list', 'None,2,600,7200', #,3600
   
   # wavegage:
    '--raw_pattern', "*{prefix:}_V{number:0>2}*.[tT][xX][tT]",
    '--aggregate_period_s_int_list', 'None, 2, 600, 3600',   
   
# 'cfg/200901incl_load.yml',
# 'cfg/200813incl_load-calibtank2#b.yml'
# 'cfg/200813incl_load-calibtank#b20.yml'
# 'cfg/200813incl_load-caliblab-b.yml'
# 'cfg/200813incl_load-calibtank-b.yml'
"""