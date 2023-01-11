from pathlib import Path
import pandas as pd
import numpy as np

dir_in = r'd:\WorkData\BalticSea\211029_ABP48\_subproduct\weighted_average_CTD'
out_params = ['Temp', 'Sal']
out = []
for f in (dir_in:=Path(dir_in)).glob('АБП*.txt'):
    print(f.name, '...')
    if f.name == 'АБП48004.txt':
        df = pd.read_csv(
            f, skiprows=5,  header=0, delimiter=',', encoding_errors='replace', index_col=['Pres'],
            names='Date Time Pres Temp Sal Sigma Turb SoundV'.split(), usecols='Date Time Pres'.split() + out_params,
            parse_dates=[[0, 1]], dayfirst=True, skipinitialspace=True
            )
    else:
        df = pd.read_csv( #  Date       Time     Pres     Temp     Cond      Sal      O2%    O2ppm
            f, skiprows=1, header=0, delimiter=r'\s+', encoding_errors='replace', index_col=['Pres'],
            names='Date Time Pres Temp Cond Sal O2% O2ppm'.split(), usecols='Date Time Pres'.split() + out_params,
            parse_dates=[[0, 1]]  #, skipinitialspace=True
            )
    t_start = df['Date_Time'].values[0]
    df = df.drop(columns='Date_Time').sort_index().groupby(df.index).mean()
    uniform_pres = np.arange(*df.index[[0, -1]], 0.1) # pressure step = 0.1
    mean_params = [np.interp(uniform_pres, df.index, df[param].values).mean() for param in out_params]
    #df_interp = df.reindex(uniform_pres).interpolate(method='linear')  #df_interp.mean().to_list()
    out.append([str(t_start), f.name] + df.index[[0, -1]].to_list() + mean_params)
df_out = pd.DataFrame.from_records(out, columns=['TimeStart', 'File', 'PresStart', 'PresEnd'] + out_params, index='File')
df_out.to_csv(dir_in / 'out_weighted_mean.csv', sep='\t', float_format='%.5g')