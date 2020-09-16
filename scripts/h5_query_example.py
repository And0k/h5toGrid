import pandas as pd

from utils2init import ini2dict

cfg = ini2dict(r'D:\Work\_Python3\_projects\PyCharm\h5toGrid\CTD_calc.ini')
tblD = cfg['out']['table'][0]
tblL = tblD + '/logFiles'
store = pd.HDFStore(r'd:\WorkData\_source\AtlanticOcean\161113_Strahov\161113_Strahov.h5')
dfL = store[tblL]
qstr_trange_pattern = "index>=Timestamp('{}') & index<=Timestamp('{}')"
r = [r for n, r in enumerate(dfL.itertuples()) if n == 3][0]  # dfL.iloc[3], r['Index']= dfL.index[3]
qstr = qstr_trange_pattern.format(r.Index, r.DateEnd)
Dat = store.select(tblD, qstr)

store.close()
