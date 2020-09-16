# import my scripts
from inclinometer.incl_load import main as incl_load

incl_load([
'ini/200901incl_load.yml', '--step_start', '2'
# 'ini/200813incl_load-calibtank2#b.yml'
# 'ini/200813incl_load-calibtank#b20.yml'
# 'ini/200813incl_load-caliblab-b.yml'
# 'ini/200813incl_load-calibtank-b.yml'
])