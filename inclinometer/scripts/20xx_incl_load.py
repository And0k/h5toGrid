# import my scripts
from inclinometer.incl_load import main as incl_load

b_wavegauge = not True  # False

# import warnings
# warnings.filterwarnings("error")

# for st_en in [
#     '15:56-16:35',
#     '14:38-15:28',
#     '13:16-13:49',
#     '12:28-13:01',
#     '10:11-11:44']:
#     st, en = st_en.split('-')
#     incl_load([
#         'cfg/210519incl_load_Pregolya,Lagoon.yml',
#         '--min_date_dict', f'0: 2021-05-19T{st}:00',
#         '--max_date_dict', f'0: 2021-05-19T{en}:50',  # '16: 2020-10-21T16:10:33, 15: 2020-10-21T17:46:05',
#         '--step_start_int', '2',  # '50', #'2',
#         '--step_end_int', '2',  # '2',
#         '--dask_scheduler', 'synchronous'
#         ])

#for raw_subdir in ['get210827@i4,11,36,37,38,w6_Симагин_rar']: #'get210813@i4,11,36,37,38,w6_rar',
raw_subdir = None \
    # '@i09_off211019_1637.zip'
    #'get210827.rar'
    # '@i14,15w1,4_get210929.zip'

path_cruise = \
    r'd:\workData\BalticSea\201202_BalticSpit_inclinometer\211008E15m@i11,36,37,38,w2'
    # r'd:\workData\BalticSea\201202_BalticSpit_inclinometer\210618P7.5m@i9,14,15,19w1,2,4'
    # r'd:\workData\BalticSea\201202_BalticSpit_inclinometer\210827@P10m,P15m-i14,15,w1,4'
    #
if True:
    if b_wavegauge:
        incl_load([
            'cfg/incl_load_201202_BalticSpit#w.yml',
            # '--db_path', r'd:\workData\BalticSea\201202_BalticSpit_inclinometer\210726@i4,5,11,36,37,38,w2,5,6\_raw\210726incl.h5',
            '--path_cruise', path_cruise,
            '--raw_subdir', raw_subdir,  # '210611_get210710[w2].rar',
            # '--min_date_dict', '0: 2021-06-10T12:15:00',
            #
            # probe with 1800s burst period:
            #   # '28, 33',  #3, 5, 9, 10, 11, 15, 19'
            '--probes_int_list', '2, 4',  # '1, 4',
                # '--aggregate_period_s_int_list', 'None',
            # '--aggregate_period_s_int_list', 'None, 2, 300, 3600',  #', 'None2, 600, 7200',
            # 'None, 2, 600, 1800, 7200',  # '',  # None,  300,  3600
            #    '--aggregate_period_s_not_to_text_int_list', 'None, 2',
            # default=None, use 2 for speedup. 300, 7200, ,3600  # exclude too long saving op
            # '--step_start_int', '1',  # '50', #'2',
            # '--step_end_int', '1',
            #
            #     # '--load_timeout_s_float', '0'
            '--dask_scheduler', 'threads'  #synchronous distributed'  # may be not affect incl_h5clc
            ])
    else:
        incl_load([
        'cfg/incl_load_201202_BalticSpit.yml',
        '--path_cruise', path_cruise,
        #'--db_coefs', r'd:\workData\BalticSea\201202_BalticSpit_inclinometer\210922@E15m-i19,36,37,38,w2\_raw\210922raw-.h5',  # 210921raw_zero.h5
        '--raw_subdir', raw_subdir,
        #'cfg/210519incl_load_Pregolya,Lagoon.yml',
        '--probes_int_list', '36,37,38',  #11,, 9,19, 15 '19, 36, 37, 38',  # '04, 11, 36, 37, 38',  #,
        #'--time_start_utc_dict', '14: 2021-08-27T18:49',  # last: 14187000 row ~01.10.2021 13:04
        # '--aggregate_period_s_int_list', 'None',  # '600, 7200',  # 2,
        # '--aggregate_period_s_not_to_text_int_list', 'None, 2, 600, 7200',
        '--min_date_dict', '0: 2021-10-08T14:00',
        #'--max_date_dict', '0: 2020-12-18T15:29:50',  # '16: 2020-10-21T16:10:33, 15: 2020-10-21T17:46:05',
        # '--step_start_int', '1',
        # '--step_end_int', '1',
        '--dask_scheduler', 'threads'  #, 'synchronous'
        ])

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
   
   # wavegauge:
    '--raw_pattern', "*{prefix:}_V{number:0>2}*.[tT][xX][tT]",
    '--aggregate_period_s_int_list', 'None, 2, 600, 3600',   
   
# 'cfg/200901incl_load.yml',
# 'cfg/200813incl_load-calibtank2#b.yml'
# 'cfg/200813incl_load-calibtank#b20.yml'
# 'cfg/200813incl_load-caliblab-b.yml'
# 'cfg/200813incl_load-calibtank-b.yml'
"""