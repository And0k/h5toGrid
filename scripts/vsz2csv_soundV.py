#!/usr/bin/env python
# coding:utf-8

"""
  Author:  Andrey Korzh <ao.korzh@gmail.com>
  Purpose: load filtered(?) ctd data from Veusz profile
  calc parameters (SoundV) and save to csv
"""

# gsw.sound\_speed(SA,t,p)

# my functions:
# gsw.sound\_speed(SA,t,p)
import csv
import re

from gsw import Sstar_from_SP, sound_speed_t_exact

from to_pandas_hdf5.csv_specific_proc import deg_min_float_as_text2deg
# my functions:
from veuszPropagate import load_vsz_closure

cfg = {'in': {}, 'out': {}, 'program': {}}

# User input ##########################################################
cfg['in']['path'] = \
    r'd:\workData\BalticSea\171003_ANS36\CTD_Idronaut#494\171016_0859.vsz'
# r'd:\workData\BalticSea\170614_ANS34\CTD_Idronaut#494\170707_0602.vsz'
str = '5516.059082,N, 1636.309937,E'
# convert coordinates to float (degrees)
str_list = re.split(',[NE][, ]*', str)
cfg['in']['lat'] = deg_min_float_as_text2deg(float(str_list[0]))  # 55+39.33/60
cfg['in']['lon'] = deg_min_float_as_text2deg(float(str_list[1]))  # 19+22.68/60
print('Coordinates: lat={lat}, lon={lon}'.format_map(cfg['in']))

cfg['out']['header'] = 'Pres,SoundV'
cfg['out']['header'] = 'Pres,SoundV'

cfg['out']['precision'] = [3, 2]
cfg['out']['name_add'] = '_SoundV.txt'

cfg['program']['veusz_path'] = u'C:\\Program Files (x86)\\Veusz'  # directory of Veusz
load_vsz = load_vsz_closure(cfg['program']['veusz_path'])

# ----------------------------------------------------------------------
g, ctd = load_vsz(cfg['in']['path'], prefix='CTD')

# Sal = SP_from_C(ctd['Cond'], ctd['Temp']/1.00024, ctd['Pres']) # T68 -> T90
# SA= SA_from_rho_t_exact(rho, t, p)
SA = Sstar_from_SP(ctd['Sal'], ctd['Pres'], lat=cfg['in']['lat'],
                   lon=cfg['in']['lon'])  # SA=S* for Baltic
ctd['SoundV'] = sound_speed_t_exact(SA, ctd['Temp'], ctd['Pres'])

header = [col_name.strip() for col_name in cfg['out']['header'].split(',')]
# header_b= [h.encode('ascii') for h in header]
with open(cfg['in']['path'][:-4] + cfg['out']['name_add'],
          'w', newline='') as f:  # 'wb',
    writer = csv.writer(f, delimiter='\t')  # ,delimiter=' '
    writer.writerow(header)
    for tup in zip(*[ctd[col_name].round(n_digits) for col_name, n_digits \
                     in zip(header, cfg['out']['precision'])]):
        writer.writerow(tup)
print('ok')
