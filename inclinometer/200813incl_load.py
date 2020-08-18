
import sys
from pathlib import Path
# import my scripts
drive_d = Path('D:/' if sys.platform == 'win32' else '/mnt/D')  # allows to run on both my Linux and Windows systems:
module_path = drive_d.joinpath('Work/_Python3/And0K/h5toGrid')
sys.path.append(str(Path(module_path).resolve()))  # os.getcwd()

from inclinometer.incl_load import main as incl_load

#incl_load([])
incl_load([
'ini/200813incl_load-calibtank2#b.yml'
# 'ini/200813incl_load-calibtank#b20.yml'
# 'ini/200813incl_load-caliblab-b.yml'
# 'ini/200813incl_load-calibtank-b.yml'
])