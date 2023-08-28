# import my scripts
from inclinometer.incl_load import main as incl_load

incl_load([
    '--path_cruise', r"d:\WorkData\BalticSea\230423inclinometer_Zelenogradsk",
    # d:\WorkData\_experiment\inclinometer\200117_tank[23,30,32]",
    '--min_date', '0: 2023-04-23T10:52:57',
    '--max_date', '0: 2023-04-28T16:30:00',
    # '--path_cruise', r"d:\workData\_experiment\inclinometer\210331_tank[4,5,9,10,11,19,28,33,36,37,38]",
    # '--min_date_dict', '0: 2021-03-31T13:40:00, 19: 2021-03-31T14:49:00, 15: 2021-03-31T14:49:00, 28: 2021-03-31T14:49:00, 33: 2021-03-31T14:49:00',
    # '--max_date_dict', '0: 2021-03-31T14:38:50, 19: 2021-03-31T16:17:50, 15: 2021-03-31T16:17:50, 28: 2021-03-31T16:17:50, 33: 2021-03-31T16:17:50',  #, 4: 2021-03-31T15:00:00
    # '--step_start_int', '1',
    '--step_end_int',   '1',
    # '--db_coefs', r'd:\WorkData\_experiment\inclinometer\190710_compas_calibr-byMe\190710incl.h5',
    '--probes_int_list', '28,33',
    # '--dt_from_utc_seconds_dict', '0: -860'
])


# commad line example:
"""  
d: && cd d:\Work\_Python3\And0K\h5toGrid && python -m inclinometer.incl_load --path_cruise "d:\workData\_experiment\inclinometer\210331_tank[4,5,9,10,11,19,28,33,36,37,38]" --time_start_utc_dict "4: 2021-03-30T18:51:06" --step_end 1
"""
