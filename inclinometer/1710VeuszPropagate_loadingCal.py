import re
from os import path as os_path

import numpy as np

from utils2init import this_prog_basename
# my
from veuszPropagate import main as veuszPropagate, __file__ as file_veuszPropagate


# ###################################################################################

def main():
    print('\n' + this_prog_basename(__file__), end=' started. ')
    # try:
    #     cfg['in']= init_file_names(cfg['in'])
    # except Ex_nothing_done as e:
    #     print(e.message)
    #     return()

    # gen_names = ge_names(cfg)

    cfg = veuszPropagate([
        os_path.join(os_path.dirname(file_veuszPropagate), 'veuszPropagate_incl.ini'),
        # '--path', r'd:\workData\BalticSea\171003_ANS36\inclinometr\171017\171017#??.TXT',  # windows
        '--path', r'/mnt/D/workData/BalticSea/171003Strahov/inclinometr/171017/171017#??.TXT',
        # in Linux match the case is important
        '--pattern_path', '171017#01.vsz',
        '--log', os_path.join(os_path.dirname(file_veuszPropagate), 'logs/viewsPropagate.log'),
        '--data_yield_prefix', 'Hxyz',
        '--eval_list',
        """
        "ImportFileCSV(u'{nameRFE}', blanksaredata=True, encoding='ascii', headermode='1st', linked=True, dsprefix='counts', rowsignore=2, skipwhitespace=True)", 
        "TagDatasets(u'source', [u'countsAx', u'countsAy', u'countsAz', u'countsBattery', u'countsDay', u'countsHour', u'countsMinute', u'countsMonth', u'countsMx', u'countsMy', u'countsMz', u'countsSecond', u'countsTemp', u'countsYear'])"
        """,
        # '--import_method', 'ImportFileCSV',
        '--add_custom_list', 'Ag_old',  # , Ch',
        '--add_custom_expressions_list',
        # """
        # 'float64([[1,0,0],[0,1,0],[0,0,1]])/16384.0'
        # """
        None,
        '--before_next_list', 'restore_config',  # 'Close(), '
        '--export_pages_int_list', '1,3,4,5',  #
        '--veusz_path', '/usr/lib64/python3.6/site-packages/veusz-2.1.1-py3.6-linux-x86_64.egg/veusz',
        # '/home/korzh/Python/other_sources/veusz/veusz',
        '-V', 'DEBUG'])
    # os_path.dirname( #
    if not cfg:
        exit(0)

    file_cal_pattern = os_path.join(cfg['in']['dir'], '171121zeroing/INKL_{:03}_data.txt')
    iFile = cfg['in']['start_file']  # inclinometers are numbered from 1
    cfgin_update = None
    while True:
        iFile += 1
        try:
            d, log = cfg['co_send_data'].send(cfgin_update)

        except (GeneratorExit, StopIteration):
            print('ok>')
            break
        except Exception as e:
            print('There are error ', e.__class__, '!',
                  '\n==> '.join([a for a in e.args if isinstance(a, str)]), sep='')
            continue

        i = int(log['fileName'].split('#')[1])

        Hxyz = d['Hxyz']
        # Hxyz = np.column_stack((a['Mx'], a['My'], a['Mz']))[slice(*iUseTime.flat)].T
        if len(Hxyz) < 3 or not np.prod(Hxyz.shape):  # 3 is ok but may be empty
            print('\nNo data from Veusz!\n')
            bBad = True
        else:
            file_data = file_cal_pattern.format(i)
            with open(file_data) as f:
                Ag_str = f.read()
            Ag_str = re.sub(r'((?<=\d)([ ]+))|(?=\n)', r',\1', Ag_str)
            Ag = np.float64(eval(Ag_str))
            Ag_str = 'float64({})'.format(Ag_str)
            if Ag[0, 0] > 10:
                print('\nBad calibration!\n')
                bBad = True
            else:
                bBad = False

        if bBad:
            print('using default coef!')
            Ag_str = 'float64([[1,0,0],[0,1,0],[0,0,1]])/16384.0'

        """
        AddCustom('definition', u'Ch', u'float64([[60,-160,-650]])')
        AddCustom('definition', u'Ah_old', u'float64([[50,0,0],\n[0,65,0],\n[0,0,90]])*1e-4')
        """

        # calibrate_plot(Hxyz, Ah, b)
        # A_str, b_str = coef2str(Ah, b)
        # b_str= 'float64([{}])'.format(b_str)
        if not bBad:
            print('calibration coefficient loaded ({}): '.format(os_path.basename(file_data)),
                  'A = \n', Ag_str)
        cfgin_update = {'add_custom_expressions': [Ag_str]}


if __name__ == '__main__':
    main()
