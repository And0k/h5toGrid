# @+leo-ver=5-thin
# @+node:korzh.20180606191031.1: * @file /mnt/D/Work/_Python3/And0K/h5toGrid/scripts/mean_on_csv.py
# @+<<declarations>>
# @+node:korzh.20180606191204.1: ** <<declarations>>
import sys

from to_pandas_hdf5.csv_specific_proc import *

try:
    __file__
except NameError as e:
    __file__ = '/mnt/D/Work/_Python3/And0K/h5toGrid/scripts/incl_proc180418.py'

path_h5toGrid = Path(__file__).parent.parent
sys.path.append(str(path_h5toGrid))

from to_pandas_hdf5.csv2h5 import *  # main as csv2h5, __file__ as file_csv2h5, read_csv


# @-<<declarations>>
# @+others
# @+node:korzh.20180520173058.1: ** plot2vert
def plot2vert(x, y=None, y_new=None, title='', b_show_diff=True, ylabel='P, dBar'):
    """
    :param y: points
    :param y_new: line
    example:
        plot2vert(dfcum.index, dfcum.P, y_filt, 'Loaded {} points'.format(dfcum.shape[0]), b_show_diff=False)
    """
    ax1 = plt.subplot(211 if b_show_diff else 111)
    ax1.set_title(title)
    if y is None:
        b_show_diff = False
    else:
        plt.plot(x, y, '.b')  # ,  label='data'
    if y_new is None:
        b_show_diff = False
    else:
        plt.plot(x, y_new, 'g--')  # , label='fit'
    ax1.grid(True)
    plt.ylabel(ylabel)
    plt.legend()

    if b_show_diff:
        plt.setp(ax1.get_xticklabels(), visible=False)

        ax2 = plt.subplot(212, sharex=ax1)  # share x only
        plt.plot(x, y_new - y, '.r', label='fit')
        plt.setp(ax2.get_xticklabels(), fontsize=6)
        ax2.grid(True)
        plt.ylabel('error, dBar')
        plt.xlabel('codes')

    plt.show()


# @+node:korzh.20180606095915.1: ** mean_on_csv
path_data = '/mnt/D/workData/_experiment/_2018/inclinometr/180605Pcalibr/inclPres11/*.txt'
# '/mnt/D/workData/_experiment/_2018/inclinometr/180605Pcalibr/inclPres14/*.txt'

cfg = main([
    str(path_h5toGrid / 'scripts' / 'cfg' / 'csv_chain_Baranov.ini'),
    '--path', path_data,
    '--b_interact', 'False',
    '--header', 'yyyy(text),mm(text),dd(text),HH(text),MM(text),SS(text),P,X1,X2',
    '--return', '<return_cfg_step_gen_names_and_log>',
    '--log', str(path_h5toGrid / 'scripts' / 'log' / 'csv2h5_inclin_Kondrashov.log'),
    '--b_incremental_update', 'False'  # becouse we not use store at all
    ])
print(cfg)
cfg_out = cfg['out']
# cfg['in']['fun_proc_loaded'].visualize()
stat = []
i = 0
name_value = []
tim_start = []
for path_csv in cfg['in']['gen_names_and_log'](cfg['out']):
    a = read_csv(path_csv, **cfg['in'])
    # , tim, b_ok
    # # filter
    # if not np.all(b_ok):
    #     # delete interpolated time values - very possible bad rows
    #     a = a.merge(pd.DataFrame(index=np.flatnonzero(b_ok)))
    #     tim = tim[~b_ok]

    df = a[['P']].compute()
    tim_start.append(tim[0])
    stat.append(df.describe())
    name_value.append(re.sub('[^\d.]', '', path_csv.stem))  # del all leters
    try:
        plot2vert(df.index, df.P, stat[-1].loc['mean', 'P'] + np.zeros_like(df.index),
                  'Fitting "{}" {} points'.format(path_csv.stem, df.shape[0]))
    except Exception as e:
        print(e)
    # Save last time to can filter next file
    cfg['in']['time_last'] = tim[-1]  # date[-1]

df_export = pd.DataFrame({'Load': np.float64(name_value),
                          'code': np.float64([s.loc['mean', 'P'] for s in stat])},
                         index=tim_start)

path_export = path_csv.parent.with_suffix('.dat')
df_export.to_csv(path_export, index_label='DateTime_UTC', date_format='%Y-%m-%dT%H:%M:%S', sep='\t')
# @-others
# @-leo
