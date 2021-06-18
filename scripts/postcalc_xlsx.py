# @+leo-ver=5-thin
# @+node:korzh.20180601111551.1: * @file postcalc_xlsx.py
# @+others
# @+node:korzh.20180601111755.1: ** o2% to ppm
# from gsw.conversions import t90_from_t68, CT_from_t
import re
import numpy as np
import pandas as pd

if False:
    # @+others
    # @+node:korzh.20180601111926.1: *3* gsw Solubility - not implemented
    # coord df.Coord.apply(lambda x: re.split(' *[NE ] *', x))
    coord_parts = df.Coord.apply(lambda x: [np.float64(y) for y in re.split(' *[NE ] *', x) if len(y)])
    df['Lat'] = coord_parts.apply(lambda x: x[0] + x[1] / 60)
    df['Lon'] = coord_parts.apply(lambda x: x[2] + x[3] / 60)
    df.drop(columns='Coord', inplace=True)

    Temp90 = gsw.conversions.t90_from_t68(df['T'])
    # out['Sal'] = SP_from_C(out['Cond'], out['Temp'], out['Pres'])  # recalc here?
    SA = gsw.SA_from_SP(df['S'], df['P'], lat=df['Lat'], lon=df['Lon'])  # Absolute Salinity  [g/kg]

    conservative_temperature = gsw.conversions.CT_from_t(SA, Temp90, df['P'])

    # SA= gsw.Sstar_from_SP(df['S'], df['P'], lat=df['Lat'], lon=df['Lon']) # SA=S* for Baltic
    df['pt'] = gsw.pt_from_CT(SA, df['T'])
    Solubility = gsw.O2sol_SP_pt(df.S, df['pt'])
    # @-others

dat_file = '/mnt/D/workData/BalticSea/171003_ANS36/_doc/bottom_layer_data.xlsx'
print('reading {}'.format(dat_file))
df = pd.read_excel(dat_file, header=0)
# df.columns 
# Index(['St', 'Date', 'Tim at Ship', 'Unnamed: 3', 'Time', 'Dist_sect_km', 'Т',
#        'S', 'Dens', 'О2%', 'area', 'Coord', 'P'],

dDist = np.diff(df.Dist_sect_km)
breakes = np.pad(
    np.flatnonzero(np.logical_or(dDist == 0, np.abs(dDist) > 10)),
    mode='constant', constant_values=(0, len(df.P)), pad_width=1
    )
# array([36])
for st, en in zip(breakes[:-1], breakes[1:]):
    df.iloc[st:en, df.columns == 'P'] = df.iloc[st:en].set_index('Dist_sect_km').P.interpolate().values

from seawater import satO2

Solubility = satO2(df.S.values, df['T'].values)  # salinity [psu (PSS-78)], temperature [℃ (ITS-68)],

df['O2ppm'] = df['О2%'] * Solubility / 100
# @-others
# @-leo
