from to_pandas_hdf5.csv_specific_proc import rep_in_file, correct_txt
from pathlib import Path
import sys
sys.argv = [sys.argv[0], r'd:\WorkData\BalticSea\220505_D6\meteo_GMX500\_raw\220505@GMX500.csv']

print('Correcting csv', sys.argv[1])
filename_in = Path(sys.argv[1])

file_in = correct_txt(
    filename_in,
    binary_mode=True,
    mod_file_name=lambda x: f'{x}@GMX500.csv',
    sub_str_list=[b'^Q?(?P<use>([^,]*,){17}[^,]*)', b'^.+']
    )
