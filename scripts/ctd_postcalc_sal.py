# @+leo-ver=5-thin
# @+node:korzh.20180601111551.1: * @file postcalc_xlsx.py
# @+others
# @+node:korzh.20180601111755.1: ** o2% to ppm
# from gsw.conversions import t90_from_t68, CT_from_t
import re
import gsw
import numpy as np
import pandas as pd
# from numpy.polynomial import Polynomial
from numpy.polynomial.polynomial import polyval

# not need really:
lon_kaliningrad, lat_kaliningrad = 20.5, 54.7  # Координаты Калининграда
lon_gulf, lat_gulf = 28.0, 60.0  # Примерные координаты Финского залива

def calculate_salinity(c, t, p, lon=None, lat=None):
    """
    Расчет абсолютной солености с использованием gsw

    c: электропроводность (мСм/см)
    t: температура in-situ (°C)
    p: давление (дбар)
    lon: долгота (градусы)
    lat: широта (градусы)

    Возвращает:
    SA: абсолютная соленость (г/кг)
    """
    # расчет практической солености
    SP = gsw.SP_from_C(c, t, p)

    # # Расчет абсолютной солености
    # SA = gsw.SA_from_SP(SP, p, lon, lat)

    return SP


# CTD SAIV

# 0.45355 + 0.9784 * x + 0.00042699 * x^2 - 3.6304e-06 * x^3
coefC_SD208 = [
    0.45569, 0.97683, 0.00049753, -4.3801e-06
]  # [0.45355, 0.9784, 0.00042699, -3.6304e-06]
data = pd.read_csv(
    r"d:\WorkData\_experiment\CTD\240821@Idro316,SAIV_ref=SST48Mc\SD208_CST.txt",
    delimiter="\t",
    encoding="ascii",
    skip_blank_lines=True,  # Veusz 'blanksaredata' is True, but pandas treats blank lines as NaN by default
    skipinitialspace=True,
)
Sref = data["S_ctd48"].where(data["S_sal"].isna(), other=data["S_sal"])

C_mSm = polyval(data['C_SD208'], coefC_SD208)

# Calculate salinity
SP = calculate_salinity(
    c=C_mSm, t=data["T_SD208"], p=0, lon=lon_kaliningrad, lat=lat_kaliningrad
)
print(pd.DataFrame({"dS_err": (SP - Sref).values}, index=Sref))


# CTD Idronaut

# -0.0055809 + 1.0033 * x - 0.00018094 * x^2 + 2.609e-06 * x^3
coefC_I316 = [-0.0055809, 1.0033, -0.00018094, 2.609e-06]
coefT_I316 = [-0.0041328, 1.0002, 5.7624e-05]
data = pd.read_csv(
    r"d:\WorkData\_experiment\CTD\240821@Idro316,SAIV_ref=SST48Mc\Idronaut316#494_CST.txt",
    delimiter="\t",
    encoding="ascii",
    skip_blank_lines=True,  # Veusz 'blanksaredata' is True, but pandas treats blank lines as NaN by default
    skipinitialspace=True,
)
Sref = data["S_ctd48"].where(data["S_sal"].isna(), other=data["S_sal"])

C_mSm = polyval(data['C_I316'], coefC_I316)
t = polyval(data["T_I316"], coefT_I316)

# Calculate salinity
SP = calculate_salinity(
    c=C_mSm, t=t, p=0, lon=lon_kaliningrad, lat=lat_kaliningrad
)
print(pd.DataFrame({"dS_err": (SP - Sref).values}, index=Sref))


# Check sensitivity to errors

SP - calculate_salinity(c=C_mSm - 0.001, t=t - 0.005, p=0)


C_kaliningrad = 0.5  # мСм/см (пример значения для речной воды)
C_gulf = 2.5  # мСм/см (пример значения для Финского залива)
t = 15  # °C
p = 10  # дбар



SA_gulf = calculate_salinity(C_gulf, t, p, lon_gulf, lat_gulf)

print(f"Абсолютная соленость речной воды в Калининграде: {SA_kaliningrad:.4f} г/кг")
print(f"Абсолютная соленость воды в Финском заливе: {SA_gulf:.4f} г/кг")

# Дополнительные расчеты для анализа
density_kaliningrad = gsw.rho(SA_kaliningrad, t, p)
density_gulf = gsw.rho(SA_gulf, t, p)

print(f"Плотность воды в Калининграде: {density_kaliningrad:.2f} кг/м³")
print(f"Плотность воды в Финском заливе: {density_gulf:.2f} кг/м³")



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
