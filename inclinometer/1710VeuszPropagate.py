from os import path as os_path

import numpy as np
from .incl_calibr import calibrate, coef2str

from utils2init import this_prog_basename, standard_error_info
# my
from veuszPropagate import main as veuszPropagate, __file__ as file_veuszPropagate

cfg = {'in': {
    'path': r'd:\workData\BalticSea\171003_ANS36\inclinometr\171015_intercal_on_board\#*.TXT',

    }}


# ###################################################################################
def main():
    print('\n' + this_prog_basename(__file__), end=' started. ')
    # try:
    #     cfg['in']= init_file_names(cfg['in'])
    # except Ex_nothing_done as e:
    #     print(e.message)
    #     return()

    # gen_names = ge_names(cfg)
    gen_data = veuszPropagate([os_path.join(os_path.dirname(
        file_veuszPropagate), 'veuszPropagate.ini'),
        '--data_yield_prefix', 'Hxyz', '--path', cfg['in']['path'], '--pattern_path',
        r'd:\workData\BalticSea\171003_ANS36\inclinometr\171015_intercal_on_board\~pattern~.vsz',
        '--eval_list',
        """
        'ImportFileCSV(u"{nameRFE}", blanksaredata=True, encoding="ascii", headermode="1st", linked=True, dsprefix=u"counts", rowsignore=2, skipwhitespace=True)',
        "TagDatasets(u'source', [u'countsAx', u'countsAy', u'countsAz', u'countsBattery', u'countsDay', u'countsHour', u'countsMinute', u'countsMonth', u'countsMx', u'countsMy', u'countsMz', u'countsSecond', u'countsTemp', u'countsYear'])"
        """,
        # '--import_method', 'ImportFileCSV',
        '--add_custom_list', 'Ah_old, Ch',
        '--add_custom_expressions_list',
        """
        'float64([[ 1, 0, 0],\
          [0, 1, 0],\
          [0, 0, 1]])\
        ',
        
        'float64([[0, 0, 0]])'
        """, '--before_next', 'restore_config',
        '--export_pages_int_list', '1'])

    cfgin_update = None
    while True:
        try:
            d = gen_data.send(cfgin_update)
        except (GeneratorExit, StopIteration):
            print('ok>')
            break
        except Exception as e:
            print('There are error ', standard_error_info(e))
            continue

        Hxyz = d['Hxyz']
        # Hxyz = np.column_stack((a['Mx'], a['My'], a['Mz']))[slice(*iUseTime.flat)].T
        if len(Hxyz) < 3 or not np.prod(Hxyz.shape):  # 3 is ok but may be empty
            print('\nNo data from Veusz!\n')
            bBad = True
        else:
            Ah, b = calibrate(Hxyz)
            if Ah[0, 0] > 10:
                print('\nBad calibration!\n')
                bBad = True
            else:
                bBad = False
        if bBad:
            print('use 1st coef!')
            b = np.float64([[46, -166, -599]])
            Ah = np.float64([[0.0054, -0.0001, -0.0001],
                             [-0.0001, 0.0069, -0.0001],
                             [-0.0001, -0.0001, 0.0089]])
        """
        AddCustom('definition', u'Ch', u'float64([[60,-160,-650]])')
        AddCustom('definition', u'Ah_old', u'float64([[50,0,0],\n[0,65,0],\n[0,0,90]])*1e-4')
        """

        # calibrate_plot(Hxyz, Ah, b)
        A_str, b_str = coef2str(Ah, b)
        # b_str= 'float64([{}])'.format(b_str)
        if not bBad:
            print('calibration coefficients calculated:',
                  '\nA = \n', A_str, '\nb = \n', b_str)
        cfgin_update = {'add_custom_expressions': [A_str, b_str]}


if __name__ == '__main__':
    main()
