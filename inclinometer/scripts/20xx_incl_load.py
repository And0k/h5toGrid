# import my scripts
from inclinometer.incl_load import main as incl_load

incl_load([
    'ini/200901incl_load.yml',
    '--path_cruise', r'd:\WorkData\BalticSea\_Pregolya,Lagoon\200918-inclinometer',
#r'd:\WorkData\BalticSea\_Pregolya,Lagoon\200918-inclinometer'
#r'd:\WorkData\BalticSea\200915_Pionerskiy',
# r'd:\workData\BalticSea\200630_AI55',
    '--probes_int_list', '14,23',  #'3,15,16,10,4', 2,'10,15',
    '--raw_subdir', '*.ZIP',
   # 'ini/200901incl_load#wavegege.yml',
   # '--dask_scheduler', 'synchronous',
    '--step_start_int', '3', '--step_end_int', '3',
    '--min_date_dict', '0: 2020-09-19T09:00:00',  #'0: 2020-09-15T14:00:00',  10: 2020-09-10T10:00:00, 15: 2020-09-10T10:00:00
    #  '--max_date_dict', '3: 2020-10-21T14:46:50, 16: 2020-10-21T16:10:33, 15: 2020-10-21T17:46:05',
    # '--time_start_utc_dict', '15: 2020-09-15T06:42:00',
    # '--dt_from_utc_days_dict', '10: -7',
   # '--aggregate_period_s_not_to_text_int_list', 'None,2,600,7200', #,3600
# 'ini/200901incl_load.yml',
# 'ini/200813incl_load-calibtank2#b.yml'
# 'ini/200813incl_load-calibtank#b20.yml'
# 'ini/200813incl_load-caliblab-b.yml'
# 'ini/200813incl_load-calibtank-b.yml'
])