import sys
from pathlib import Path

drive_d = 'D:' if sys.platform == 'win32' else '/mnt/D'  # to run on my Linux/Windows systems both
scripts_path = Path(drive_d + '/Work/_Python3/And0K/h5toGrid/scripts')
sys.path.append(str(Path(scripts_path).parent.resolve()))
# my funcs
from SMS2GPX.SMS2GPX import main as SMS2GPX

path_cruise = Path(r'd:\WorkData\BalticSea\191018_Trackers')

# Extract navigation data at runs/starts to GPX tracks. Useful to indicate where no nav?
SMS2GPX(['',
         '--path', str(path_cruise / 'sms.xml'),
         '--contacts_names', 'Tracker1, Tracker2, Tracker3',
         '--date_min', '18.10.2019 09:00:00'
         ])
