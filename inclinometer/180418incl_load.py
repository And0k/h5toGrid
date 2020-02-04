import sys
from pathlib import Path

drive_d = Path('D:/' if sys.platform == 'win32' else '/mnt/D')  # allows to run on both my Linux and Windows systems:
scripts_path = drive_d.joinpath('Work/_Python3/And0K/h5toGrid/scripts')
sys.path.append(str(Path(scripts_path).parent.resolve()))  # os.getcwd()
import re

from to_pandas_hdf5.csv2h5 import main as csv2h5
from to_pandas_hdf5.csv_specific_proc import correct_baranov_txt

block_10Mbt_str = '10000000'
# g.es(sys.argv[0])
# sys.argv[0] = scripts_path                                    # to can write log to ".\log" directory
# os.chdir(os.path.dirname(scripts_path))
if False:  # True: #g.unitTesting:  #
    # import project_root_path
    # g.cls()
    from to_pandas_hdf5.csv_specific_proc import *

if True:  # Faluse:  #    #   # Real data
    # Cruise
    path_cruise = drive_d.joinpath(
        r'd:\workData\BalticSea\191119_Filino'
        # d:\workData\BalticSea\181005_ABP44\inclinometr
        # r"WorkData\_experiment\_2018\inclinometr\181004inclinometer\_raw"
        # '/workData/_experiment/_2018/inclinometr/180905_clockCheck/raw')
        # '/mnt/D/workData/_experiment/_2018/inclinometr/180828_time/raw')
        # '/mnt/D/workData/_experiment/_2018/inclinometr/180731_КТИ/raw')
        # '/mnt/D/workData/_experiment/_2018/inclinometr/180726_КТИ/raw'
        # '/mnt/D/workData/BalticSea/180418_Svetlogorsk/inclinometer' (r'd:\workData\BalticSea\180418_Svetlogorsk')
        )
    path_db = 'incl'  # '/mnt/D/workData/_experiment/_2018/inclinometr/180731_КТИ/180731incl.h5'
    # '/mnt/D/workData/_experiment/_2018/inclinometr/180726_КТИ/180726incl.h5'
    # '/mnt/D/WorkData/BalticSea/180418_Svetlogorsk/inclinometer/180418inclPres.h5'

    if True:  # False:  # Baranov format
        probes = range(1, 20)
        dir_raw_data = path_cruise / r'inclinometer\_raw\pressure'  # V,P_txt/1D/180510_1545inclPres14.txt _source/180418inclPres11.txt
        for probe in probes:
            source_pattern = f'*W*{probe:0>2}*.[tT][xX][tT]'  # remove some * if load other probes!
            find_in_dir = dir_raw_data.glob
            source_found = list(find_in_dir(source_pattern))
            if not source_found:  # if have only output files of correct_*_txt() then just use them
                source_found = find_in_dir(f'w{probe:0>2}.txt')
                correct_fun = lambda x: x
            else:
                correct_fun = correct_baranov_txt

            for in_file in source_found:
                in_file = correct_fun(in_file)
                if not in_file:
                    continue
                csv2h5([scripts_path / 'ini/csv_Baranov_inclin.ini',
                        '--path', str(in_file),  #
                        '--blocksize_int', block_10Mbt_str,
                        '--table', re.sub('^[\d_]*', '', in_file.stem),  # delete all first digits (date part)
                        '--db_path', path_db,
                        '--log', str(scripts_path / 'log/csv_Baranov_inclin.log'),
                        '--fs_float', '10',  # 'fs_old_method_float'
                        '--b_interact', '0'
                        ])

    if False:  # True:  # True:  # Kondrashov format
        probes = [3, 9, 10,
                  16]  # [k for k in range(1, 21)]   # [1,6,8,11, 18] #3 [1,6,8,10,11,12,15,18,19] #[4, 6, 8, 9]
        for probe in probes:
            in_file = path_cruise / f'INKL_{probe:03d}.TXT'  # '_source/incl_txt/180510_INKL10.txt' # r'_source/180418_INKL09.txt'

            in_file_cor = correct_kondrashov_txt(in_file)
            csv2h5([scripts_path / 'ini/csv_Kondrashov_inclin.ini',
                    '--path', str(in_file_cor),
                    '--blocksize_int', block_10Mbt_str,
                    '--table', re.sub('^inkl_0', 'incl',
                                      re.sub('^[\d_]*', '', in_file.stem).lower()),
                    '--date_min', '17.10.2018 14:30:00',  # ; UTC, not output data < date_min
                    '--date_max', '18.10.2018 07:40:00',  # ; UTC, not output data > date_max
                    '--db_path', path_db,
                    '--log', str(scripts_path / 'log/csv2h5_Kondrashov_inclin.log'),
                    '--b_raise_on_err', '0',  # ?! need fo this file only?
                    '--b_interact', '0',
                    '--fs_float', '4.8'
                    ])
else:
    # Test with small file
    path_cruise = Path(
        '/mnt/D/Work/_Python3/And0K/h5toGrid/test/csv2h5/data')  # r'd:\workData\_source\BalticSea\180418'
    path_db = str(path_cruise / '180418incl09.h5')
    if False:  # True:  #
        csv2h5([scripts_path / 'ini/csv_Baranov_inclin.ini',
                '--path', str(path_cruise / 'inclinpres_Baranov_180418.txt'),
                '--blocksize_int', block_10Mbt_str,
                '--db_path', str(path_db),
                '--log', str(scripts_path / 'log/csv_Baranov_inclin.log')
                ])

    if True:  # True:  #
        csv2h5([scripts_path / 'ini/csv_Kondrashov_inclin.ini',
                '--path', str(path_cruise / 'inclin_Kondrashov_180430.txt'),
                '--blocksize_int', '1000',
                '--db_path', str(path_db),
                '--log', str(scripts_path / 'log/csv2h5_Kondrashov_inclin.log')
                ])
        Path(path_db)  # /mnt/D/Work/_Python3/And0K/h5toGrid/test/csv2h5/data/180418incl09.h5
