"""
replacement re expressions (pattern, replacement):
(?!=TagDatasets)?

u'([^']+)'  $1
AddCustom\('definition', (\S+) *\(([^\(]+)\),(.+)\)    $1 = lambda $2:$3
AddCustom\('definition', (\S+),(.+)\)   $1 = $2
SetData(?:2D|)Expression\((\S+), (.+), linked=True\)    $1 = $2
DatasetPlugin\('NumbersToText', \{'ds_out': ([^,]+), 'ds_in': ([^,]+),\s*'format':\s*(.+)\}\)   $1 = '$3'.format($2)
"""

from scripts.incl_calibr import calibrate, calibrate_plot

from to_pandas_hdf5.csv2h5 import init_input_cols
# my:
from to_pandas_hdf5.csv_specific_proc import convertNumpyArrayOfStrings

"""
file contents:
Year,Month,Day,Hour,Minute,Second,Ax,Ay,Az,Mx,My,Mz,Battery,Temp
2017,10,15,14,29,18,-128,48,-11696,-118,-198,160,9.64,23.00
"""
cfg = {'in': {
    'path': r'd:\workData\BalticSea\171003_ANS36\inclinometr\171015_intercal_on_board\#20.TXT',
    'header': 'yyyy(text),mm(text),dd(text),HH(text),MM(text),SS(text),Ax,Ay,Az,Mx,My,Mz,Battery,Temp',
    'delimiter': ',',
    'skiprows': 13
    }}

# Prepare cpecific format loading and writing
cfg['in'] = init_input_cols(cfg['in'])


# cfg['output_files']['names'] = np.array(cfg['in']['dtype'].names)[cfg['in']['cols_loaded_save_b']]
# cfg['output_files']['formats'] = [cfg['in']['dtype'].fields[n][0] for n in cfg['output_files']['names']]
# cfg['output_files']['dtype'] = np. \
#     dtype({'formats': cfg['output_files']['formats'], 'names': cfg['output_files']['names']})

def fun_proc_loaded(a, cfg_in):
    """
    Specified prep&proc of navigation data from program "Supervisor":
    - Time calc: gets time in current zone

    :param a: numpy record array. Will be modified inplace.
    :param cfg_in: dict
    :return: numpy 'datetime64[ns]' array

    Example input:
    a = {
    'yyyy': b"2017", 'mm': b"10", 'dd': b"14",
    'HH': b'59','MM': b'59', 'SS': b'59'}
    """

    # Baranov format specified proc
    # Time calc: gets string for time in current zone
    try:
        date = np.array(
            a['yyyy'].astype(np.object) + b'-' + a['mm'].astype(
                np.object) + b'-' + a['dd'].astype(np.object) + b'T' +
            np.char.rjust(a['HH'], 2, ﬁllchar=b'0').astype(np.object) + b':' +
            np.char.rjust(a['MM'], 2, ﬁllchar=b'0').astype(np.object) + b':' +
            np.char.rjust(a['SS'], 2, ﬁllchar=b'0').astype(np.object), '|S19', ndmin=1)
        # date = b'%(yyyy)b-%(mm)b-%(dd)bT%(HH)02b-%(MM)02b-%(SS)02b' % a
    except Exception as e:
        print('Can not convert date!')
        raise
    return convertNumpyArrayOfStrings(date, 'datetime64[ns]')


def ImportFileCSV(file_name, **varargs):
    a = np.loadtxt(file_name, dtype=cfg['in']['dtype'], delimiter=cfg['in']['delimiter'],
                   usecols=cfg['in']['cols_load'], converters=cfg['in']['converters'],
                   skiprows=cfg['in']['skiprows'])
    # Process a and get date date in ISO standard format
    try:
        date = fun_proc_loaded(a, cfg['in'])
    except IndexError as e:

        print('no data!')
    return a, date


a, stime = ImportFileCSV(cfg['in']['path'])

# ###################################################################################
from numpy import *  # for Veusz document functions

iUse = np.int64([[0, -1]])
strTimeUse = [['2017-10-15T15:37:00', '2017-10-15T19:53:00']]
iCalibr0V = np.int64([[20188, 21438]])  # array([[91295, 97601]])
nAveragePrefer = 1200  # 128
DISP_TimeZoom = [['2017-10-08T12:20:00', '2017-10-08T13:20:00']]
vecScale = 1

maxGsumMinus1 = 0.3  # 0.1
TimeShiftedFromUTC_sec = 0
fdate = lambda DateCols: np.int64(
    array(map('{0:02.0f}-{1:02.0f}-{2:02.0f}'.format, *DateCols), dtype='datetime64[D]') - np.datetime64(
        '2009-01-01')) * (24 * 3600)
fPitch = lambda Gxyz: -np.arctan2(Gxyz[0, :], np.sqrt(np.sum(np.square(Gxyz[1:, :]), 0)))
fRoll = lambda Gxyz: np.arctan2(Gxyz[1, :], Gxyz[2,
                                            :])  # np.arctan2(Gxyz[1,:], np.sqrt(np.sum(np.square(Gxyz[(0,2),:]), 0)))  #=np.arctan2(Gxyz[1,:], np.sqrt(np.square(Gxyz[0,:])+np.square(Gxyz[2,:])) )
fInclination = lambda Gxyz: np.arctan2(np.sqrt(np.sum(np.square(Gxyz[:-1, :]), 0)), Gxyz[2, :])
fHeading = lambda H, p, r: np.arctan2(H[2, :] * np.sin(r) - H[1, :] * np.cos(r),
                                      H[0, :] * np.cos(p) + (H[1, :] * np.sin(r) + H[2, :] * np.cos(r)) * np.sin(p))
fG = lambda Axyz, Ag, Cg: dot(Ag.T, (Axyz - Cg[0, :]).T)
fGi = lambda Ax, Ay, Az, Ag, Cg, i: dot(Ag.T, (column_stack((Ax, Ay, Az))[slice(*i)] - Cg[0, :]).T)
fVabs_old = lambda Gxyz, kVabs: polyval(kVabs.flat, np.sqrt(np.tan(fInclination(Gxyz))))
fbinningClip = lambda x, bin2_iStEn, bin1_nAverage: np.nanmean(reshape(x[slice(*bin2_iStEn)], (-1, bin1_nAverage)), 1)
fbinning = lambda x, bin1_nAverage: np.nanmean(reshape(x, (-1, bin1_nAverage)), 1)
repeat3shift1 = lambda A2: [A2[t:(len(A2) - 2 + t)] for t in range(3)]
median3cols = lambda a, b, c: np.where(a < b, np.where(c < a, a, np.where(b < c, b, c)),
                                       np.where(a < c, a, np.where(c < b, b, c)))
median3 = lambda x: np.hstack((np.NaN, median3cols(*repeat3shift1(x)), np.NaN))
rep2mean = lambda x, bOk: interp(arange(len(x)), flatnonzero(bOk), x[bOk], np.NaN, np.NaN)
fForce2Vabs_fitted = lambda x: np.where(x > 2, 2, np.where(x < 1, 0.25 * x, 0.25 * x + 0.3 * (x - 1) ** 4))
fIncl2Force = lambda incl: np.sqrt(np.tan(incl))
fVabs = lambda Gxyz, kVabs: fForce2Vabs_fitted(fIncl2Force(fInclination(Gxyz)))
f = lambda fun, *args: fun(*args)
positiveInd = lambda i, L: np.int32(np.where(i < 0, L - i, i))
minInterval = lambda iLims1, iLims2, L: f(
    lambda iL1, iL2: transpose([max(iL1[:, 0], iL2[:, 0]), min(iL1[:, -1], iL2[:, -1])]), positiveInd(iLims1, L),
    positiveInd(iLims2, L))
fStEn2bool = lambda iStEn, length: np.hstack(
    [(ones(iEn2iSt, dtype=np.bool8) if b else zeros(iEn2iSt, dtype=np.bool8)) for iEn2iSt, b in np.vstack((diff(
        np.hstack((0, iStEn.flat, length))), np.hstack(
        (np.repeat([(False, True)], size(iStEn, 0), 0).flat, False)))).T])
TimeShift_Log_sec = 60

# ImportFileCSV('171008#01.TXT', blanksaredata=True, encoding='ascii', headermode='1st', linked=True, dsprefix='counts', rowsignore=2, skipwhitespace=True)

# 'filter'
# maxT =  np.NaN
# bP = np.logical_not(a['Temp'] < maxT) # RuntimeWarning: invalid value encountered in less
bP = np.ones_like(a['Temp'], np.bool8)
iStEn_auto = atleast_2d((np.searchsorted(bP, 1), len(bP) - np.searchsorted(flipud(bP), 1)))
iUseTime = np.searchsorted(stime, [array(s, 'datetime64[s]') for s in array(strTimeUse)]) if len(
    strTimeUse) > 0 else iUse
iUseC = minInterval(iStEn_auto, minInterval(np.int32(iUse), iUseTime, len(stime)), len(stime))

# 'source'
dstime = diff(stime)
i_burst = nonzero(dstime > (dstime[0] if dstime[1] > dstime[0] else dstime[1]) * 2)[0] + 1
i_burst1St = i_burst[np.searchsorted(i_burst, iUseC[0, 0])] if len(i_burst) > 0 else iUseC[0, 0]
mean_burst_size = i_burst[1] - i_burst[0] if len(i_burst) > 0 else diff(iUse[0, :])

bin0_nAveragePreferCoef = f(lambda x, y: np.floor_divide(x, y) if x > y else 1 / np.floor_divide(y, x), mean_burst_size,
                            nAveragePrefer)
bin1_nAverage = np.int32(np.floor_divide(mean_burst_size, bin0_nAveragePreferCoef))
bin1_nAveragePrefer_testErr = mean_burst_size - bin1_nAverage * bin0_nAveragePreferCoef
bin2_iStEn = np.hstack((i_burst1St, iUseC[0, -1] - (iUseC[0, -1] - i_burst1St) % bin1_nAverage))
bin3_sliceMid = np.hstack([bin2_iStEn + np.int32(np.floor_divide(bin1_nAverage, 2)), bin1_nAverage])
bin3_timeMid = stime[slice(*bin3_sliceMid)]
bin3_dtime = diff(np.hstack((bin3_timeMid, stime[bin2_iStEn[1] - 1])))
bin3_timeSt = stime[slice(*append(bin2_iStEn, bin1_nAverage))]

time_BurstStarts = stime[np.int32(i_burst[logical_and(bin2_iStEn[0] <= i_burst, i_burst < bin2_iStEn[1])])]
time = stime[slice(*bin2_iStEn)]
tEnd = time[-1]
tStart = time[0]

# 'zeroing'
Ah_old = np.float64([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # /500.0
Ag_old = np.float64([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) / 16384.0
Cg = np.float64([[0, 0, 0]])
Ch = np.float64([[0, 0, 0]])

Gxyz0old = [[np.NaN, np.NaN, np.NaN]] if nansum(isnan(iCalibr0V)) > 0 else fG(column_stack((np.nanmean(
    a['Ax'][slice(*iCalibr0V[0, :])]), np.nanmean(a['Ay'][slice(*iCalibr0V[0, :])]), np.nanmean(
    a['Az'][slice(*iCalibr0V[0, :])]))), Ag_old, Cg)
old1pitch = -fPitch(Gxyz0old)
old1roll = -fRoll(Gxyz0old)

# 'coeficient'

Ag = transpose(dot(
    dot([[np.cos(old1pitch), 0, -np.sin(old1pitch)],
         [0, 1, 0],
         [np.sin(old1pitch), 0, np.cos(old1pitch)]],

        [[1, 0, 0],
         [0, np.cos(old1roll), np.sin(old1roll)],
         [0, -np.sin(old1roll), np.cos(old1roll)]]),

    transpose(Ag_old)))

Ah = transpose(dot(
    dot([[np.cos(old1pitch), 0, -np.sin(old1pitch)], [0, 1, 0], [np.sin(old1pitch), 0, np.cos(old1pitch)]],
        [[1, 0, 0], [0, np.cos(old1roll), np.sin(old1roll)], [0, -np.sin(old1roll), np.cos(old1roll)]]),
    transpose(Ah_old)))
Gxyz0 = fGi(a['Ax'], a['Ay'], a['Az'], Ag, Cg, np.ravel(np.int32(iCalibr0V)))
Gxyz0mean = transpose([np.nanmean(Gxyz0, 1)])
kVabs = np.float64([[0.361570991503], [0]])

# 'mainparam'
Gxyz = fGi(a['Ax'], a['Ay'], a['Az'], Ag, Cg, bin2_iStEn)
Hxyz = fGi(a['Mx'], a['My'], a['Mz'], Ah, Ch, bin2_iStEn)
GsumMinus1 = np.sqrt(np.sum(np.square(Gxyz), 0)) - 1
HsumMinus1 = np.sqrt(np.sum(np.square(Hxyz), 0)) - 1
sPitch = fPitch(Gxyz)
sRoll = fRoll(Gxyz)
sTmin = min(a['Temp'])

Tfilt = interp(arange(len(bP)), flatnonzero(bP), median3(a['Temp'][np.bool8(bP)]), np.NaN, np.NaN)
T = Tfilt[slice(*bin2_iStEn)]
T_bin = fbinningClip(Tfilt, bin2_iStEn, bin1_nAverage)
Vabs = fVabs_old(np.where(np.abs(GsumMinus1) > maxGsumMinus1, np.NaN, Gxyz), kVabs)
Vdir = np.degrees(np.arctan2(np.tan(sRoll), np.tan(sPitch)) + fHeading(Hxyz, sPitch, sRoll))

# DatasetPlugin('PolarToCartesian', {'x_out': Vn, 'units': np.degrees, 'r_in': Vabs, 'y_out': Ve, 'theta_in': Vdir})
Vn = Vabs * np.cos(radians(Vdir))
Ve = Vabs * np.sin(radians(Vdir))
# np.degrees(np.arctan2(x1, x2[, out]))

VeBin = fbinning(Ve, bin1_nAverage)
VnBin = fbinning(Vn, bin1_nAverage)
VabsBin = absolute(VeBin + 1j * VnBin)
VabsD_Bin = fbinning(Vabs, bin1_nAverage)
VdirBin = np.arctan2(VeBin, VnBin) * (180 / pi)
VeBinCum = cumsum(VeBin * bin3_dtime)
VnBinCum = cumsum(VnBin * bin3_dtime)

# 'statistics'
max_VabsBin = np.round(np.nanmax(VabsBin), 3)
min_VabsBin = np.round(np.nanmin(VabsBin), 3)
mean_T = np.nanmean(T)
mean_VeBin = np.nanmean(VeBin)
mean_VnBin = np.nanmean(VnBin)
mean_VdirBin = (np.degrees(np.arctan2(mean_VeBin, mean_VnBin))) % 360
mean_VabsBin = np.abs(mean_VnBin + 1j * mean_VeBin)
mid_VabsBin = np.round((min_VabsBin + max_VabsBin) / 2, 1 + np.int64(-log10((max_VabsBin - min_VabsBin) / 2)))

# 'display'
LegMinMax = ['min', '', 'max', ]
LegV = np.hstack((min_VabsBin, mid_VabsBin, max_VabsBin))
LegY = linspace(0.8, 0.96, size(LegV))
ones_lenNBursts = ones_like(time_BurstStarts)
# DatasetPlugin('Concatenate', {'ds_out': mean_VabsVdirBin, 'ds_in': (mean_VabsBin, mean_VdirBin)})

# stime = fdate([a['Year'], a['Month'], a['Day']]) + 3600*a['Hour']+ 60*a['Minute'] + a['Second'] + TimeShiftedFromUTC_sec

txtEnd = '%VDd.%VDm.%VDY %VDH:%VDM'.format(tEnd)
txtStartTitle = 'Inclinometr %VDd.%VDm.%VDY %VDH:%VDM -'.format(tStart)
txt_mean_T = 'mean(T) = %.2f\xb0C'.format(mean_T)
txt_mean_Vabs = '%.3gm/s'.format(mean_VabsBin)
txt_mean_Vdir = '%.2f\xb0'.format(mean_VdirBin)

d_HGsumMinus1 = f(lambda Lbin, g, h, n: Lbin(np.sqrt((Lbin(g, n) - g) ** 2 + (Lbin(h, n) - h) ** 2), n),
                  (lambda x, n: np.repeat(fbinning(x, n), n)), GsumMinus1, HsumMinus1, np.int32(bin1_nAverage))
if len(i_burst) > 0:
    dt = diff(time[[0, np.int32(mean_burst_size)]]) / mean_burst_size
    time_streched = np.array(np.arange(*np.int64([time[0], time[-1], dt])), 'datetime64[ns]')
    # arange(time[0], time[-1], dt) # not works with numpy 'datetime64[ns]'
else:
    time_streched = time

iDisp_TimeZoom = np.searchsorted(stime, np.ravel(array(DISP_TimeZoom, 'datetime64[s]')))
iDisp_TimeZoom_burst1St = i_burst[np.searchsorted(i_burst, iDisp_TimeZoom[0])] if len(i_burst) > 0 else iDisp_TimeZoom[
    0]
iDisp_TimeZoom_bin_iStEn = np.hstack(
    (iDisp_TimeZoom_burst1St, iDisp_TimeZoom[1] - (iDisp_TimeZoom[1] - iDisp_TimeZoom_burst1St) % bin1_nAverage))
iDisp_TimeZoom_bursts = i_burst[slice(*np.searchsorted(i_burst, np.ravel(iDisp_TimeZoom_bin_iStEn)))]
iDisp_TimeZoom_inbinStEn = np.searchsorted(bin3_timeSt, stime[np.int64(iDisp_TimeZoom_bin_iStEn)])
print('Veusz data loaded Ok>')

A, b = calibrate(Hxyz)
calibrate_plot(Hxyz, A, b)
print('calibration coefficients calculated:', '\nA:\n', '[{}]'.format(
    '\n'.join(['[{}],'.format(','.join(str(A[i, j]) for j in range(A.shape[1]))) for i in range(A.shape[0])])),
      '\nb:\n',
      '[{}]'.format(','.join(str(bi) for bi in b.flat)))

"""
TagDatasets(u'binning', [u'T_bin', u'VabsBin', u'VabsD_Bin', u'VdirBin', u'VeBin', u'VeBinCum', u'VnBin', u'VnBinCum', u'bin0_nAveragePreferCoef', u'bin1_nAverage', u'bin1_nAveragePrefer_testErr', u'bin2_iStEn', u'bin3_dtime', u'bin3_sliceMid', u'bin3_timeMid', u'bin3_timeSt', u'i_burst1St'])
TagDatasets(u'coeficient', [u'Ag', u'Ag_old', u'Ah', u'Cg', u'kVabs'])
TagDatasets(u'display', [u'LegMinMax', u'LegV', u'LegY', u'iDisp_TimeZoom', u'iDisp_TimeZoom_bin_iStEn', u'iDisp_TimeZoom_burst1St', u'iDisp_TimeZoom_bursts', u'iDisp_TimeZoom_inbinStEn', u'ones_lenNBursts', u'time_BurstStarts', u'time_streched', u'txtEnd', u'txtStartTitle', u'txt_mean_T'])
TagDatasets(u'filter', [u'GsumMinus1', u'HsumMinus1', u'Tfilt', u'bP', u'iStEn_auto', u'iUseC', u'iUseTime'])
TagDatasets(u'mainparam', [u'Gxyz', u'Hxyz', u'T', u'Vabs', u'Vdir', u'Ve', u'Vn', u'dstime', u'sPitch', u'sRoll', u'tEnd', u'tStart', u'time'])
TagDatasets(u'statistics', [u'max_VabsBin', u'mean_T', u'mean_VabsBin', u'mean_VabsVdirBin', u'mean_VdirBin', u'mean_VeBin', u'mean_VnBin', u'mid_VabsBin', u'min_VabsBin', u'sTmin', u'txt_mean_Vabs', u'txt_mean_Vdir'])
TagDatasets(u'unlink4(Vel,diagram)only', [u'LegV', u'Vabs', u'VabsBin', u'VabsD_Bin', u'Vdir', u'VdirBin', u'VeBinCum', u'VnBinCum', u'bin3_timeMid', u'time', u'txtEnd', u'txtStartTitle', u'txt_mean_Vabs', u'txt_mean_Vdir'])
TagDatasets(u'zeroing', [u'Gxyz0', u'Gxyz0mean', u'Gxyz0old', u'd_HGsumMinus1', u'old1pitch', u'old1roll'])
"""
