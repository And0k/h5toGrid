# @+leo-ver=5-thin
# @+node:korzh.20180518223923.1: * @file /mnt/D/Work/_Python3/And0K/h5toGrid/scripts/fitting.py
# @+<<declarations>>
# @+node:korzh.20180519100658.1: ** <<declarations>>
# messytablesi
# messytablesi
import gsw
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# @+others
# @+node:korzh.20180520173058.1: *3* plot2vert
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


# @-others
# @-<<declarations>>
# @+others
# @+node:korzh.20180519124610.1: ** load_calibr_by_load
def load_calibr_by_load(file_csv, colon=50):
    """
    Load calibration data from file file_csv which must have header with param Load.
    coumn Load is converted to float such that it increased on :colon: then all 'A' values will be zero
     
    :param file_csv: str, path
    :param colon: waith to add to Load
    load_data('/mnt/D/workData/_experiment/_2018/inclinometr/180416Pcalibr/Pcalibr_log.dat')
    """
    csv = pd.read_csv(file_csv, header=0, sep='\s+')  # , na_values=
    # pci
    csv.Load = csv.Load.where(lambda x: x != 'A', -colon).astype(float) + colon
    csv.dropna(inplace=True)  # csv.code.isna()

    return csv


# @+node:korzh.20180519153749.1: ** process
def psig_to_dbar(psig):
    """
    psig = pound/square inch [gauge]
    1 lb (force) = 0.4536 * 9.81 Newtons
    1 sq.in. = (2.54)^2 * 10^-4 sq.m
    
    So the conversion of 1 psi to N/m^2 is
    (0.4536 * 9.81)/2.54^2 * 1e4 = 6897. N/m^2 or Pa
    1 Pa = 1e-4 dBar
    """
    return psig * (0.45359 * gsw.grav(54, 0) / 2.54 ** 2)  # sw_g(54,0)*unitsratio('cm', 'inch')^2


if __name__ == '__main__':
    csv = load_calibr_by_load(file_csv)
    x = csv.code
    y = psig_to_dbar(csv.Load)  # Pet

    # result:
    fit = np.polyfit(x, y, 3)
    # array([ 3.67255551e-16,  1.93432957e-10,  1.20038165e-03, -1.66400137e+00])
    #         6.45620065e-14 -7.69655284e-09  1.02516917e-03 -2.90061592e+00
    y_new = np.polyval(fit, x)  # to check by plot

# @+node:korzh.20180519220058.1: ** plot

plot2vert(x, y, y_new - y, 'Fitting "{}" {} points'.format(file_csv.stem, x.shape[0]))

if False:
    ax1 = plt.subplot(211)
    ax1.set_title('Fitting')
    plt.plot(x, y, '.b', label='data')
    plt.plot(x, y_new, 'g--', label='fit')
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.grid(True)
    plt.ylabel('Pet, dBar')
    plt.legend()

    ax2 = plt.subplot(212, sharex=ax1)  # share x only
    plt.plot(x, y_new - y, '.r', label='fit')
    plt.setp(ax2.get_xticklabels(), fontsize=6)
    ax2.grid(True)
    plt.ylabel('error, dBar')
    plt.xlabel('codes')

    plt.show()
# @+node:korzh.20180519100658.2: ** example of nonlinear fitting
# example of nonlinear fitting
if False:
    def func(x, a, b, c):
        return a * np.exp(-b * x) + c


    # define the data to be fit with some noise
    xdata = np.linspace(0, 4, 50)
    y = func(xdata, 2.5, 1.3, 0.5)
    y_noise = 0.2 * np.random.normal(size=xdata.size)
    ydata = y + y_noise
    plt.plot(xdata, ydata, 'b-', label='data')

    # Fit for the parameters a, b, c of the function func
    popt, pcov = curve_fit(func, xdata, ydata)
    plt.plot(xdata, func(xdata, *popt), 'r-', label='fit')

    # Constrain the optimization to the region of 0 < a < 3, 0 < b < 2 and 0 < c < 1:
    popt, pcov = curve_fit(func, xdata, ydata, bounds=(0, [3., 2., 1.]))
    plt.plot(xdata, func(xdata, *popt), 'g--', label='fit-with-bounds')

    # see also:  
    # https://lmfit.github.io/lmfit-py/intro.html  
    # https://pythonhosted.org/PyModelFit/index.html
# @-others
# @@language python
# @@tabwidth -4
# @-leo
