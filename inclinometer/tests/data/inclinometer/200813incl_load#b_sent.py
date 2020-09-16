
import sys
from pathlib import Path
# import my scripts
from inclinometer.incl_load import main as incl_load

incl_load(['200813incl_load#b-sent.yml'
# '200813incl_load#b-sent.yml', '--step_start_int', '2'
])