"""
Wavelet filtering
"""
from matplotlib import pyplot as plt
import numpy as np
from numba import njit
from numba.core.withcontexts import objmode_context
import pywt

from filters import mad


def waveletSmooth(y, wavelet="db4", level=1, ax=None, label=None, x=None, color='k'):
    """
    Wavelet smoothing
    :param y:
    :param wavelet: #'db8'
    :param level: 11
    :param ax:
    :param label:
    :param x: where to plot y
    :return: y, ax

    waveletSmooth(NDepth.values.flat, wavelet='db8', level=11, ax=ax, label='Depth')
    """
    if level <= 0:
        return y, ax

    # calculate the wavelet coefficients
    mode = "antireflect"
    coeff = pywt.wavedec(y, wavelet, mode=mode)  # , mode="per", level=10
    # calculate a threshold: changing this threshold also changes the behavior
    sigma = mad(coeff[-level])
    # mad = median(Wj-1, k-median(Wj-1, k)) /0.6475  # Universal freshold
    uthresh = sigma * np.sqrt(2 * np.log(len(y)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode="garotte") for i in coeff[1:])  # soft

    # reconstruct the signal using the thresholded coefficients
    out = pywt.waverec(coeff, wavelet, mode=mode)
    if out.size > y.size:  # ?
        out = out[:y.size]

    if __debug__ and label:
        if x is None:
            x = np.arange(len(y))
        if ax is None:
            plt.style.use('bmh')
            f, ax = plt.subplots()
            ax.plot(x, y, color='r', alpha=0.5, label=label + ' sourse')
        ax.plot(x, out, color=color, label='{}^{}({})'.format(wavelet, level, label))
        ax.set_xlim((0, len(y)))
    else:
        ax = None
    return out, ax


if __name__ == '__main__':
    from scipy import stats

    """
    Generate the y and get the coefficients using the multilevel discrete wavelet transform.
    """
    #
    # db8 = pywt.Wavelet('db8')
    # scaling, wavelet, x = db8.wavefun()

    np.random.seed(12345)
    blck = blocks(np.linspace(0, 1, 2 ** 11))
    nblck = blck + stats.norm().rvs(2 ** 11)

    true_coefs = pywt.wavedec(blck, 'db8', level=11, mode='per')
    noisy_coefs = pywt.wavedec(nblck, 'db8', level=11, mode='per')

    # Plot the true coefficients and the noisy ones.
    fig, axes = plt.subplots(2, 1, figsize=(9, 14), sharex=True)

    _ = coef_pyramid_plot(true_coefs[1:], ax=axes[0])  # omit smoothing coefs
    axes[0].set_title("True Wavelet Detail Coefficients")

    _ = coef_pyramid_plot(noisy_coefs[1:], ax=axes[1])
    axes[1].set_title("Noisy Wavelet Detail Coefficients")

    fig.tight_layout()

    """
    Apply soft thresholding using the universal threshold
    """
    # robust estimator of the standard deviation of the finest level detail coefficients:
    sigma = mad(noisy_coefs[-1])
    uthresh = sigma * np.sqrt(2 * np.log(len(nblck)))

    denoised = noisy_coefs[:]

    denoised[1:] = (pywt.thresholding.soft(i, value=uthresh) for i in denoised[1:])

    """
    Recover the signal by applying the inverse discrete wavelet transform to the thresholded coefficients
    """
    signal = pywt.waverec(denoised, 'db8', mode='per')

    fig, axes = plt.subplots(1, 2, sharey=True, sharex=True,
                             figsize=(10, 8))
    ax1, ax2 = axes

    ax1.plot(signal)
    ax1.set_xlim(0, 2 ** 10)
    ax1.set_title("Recovered Signal")
    ax1.margins(.1)

    ax2.plot(nblck)
    ax2.set_title("Noisy Signal")

    for ax in fig.axes:
        ax.tick_params(labelbottom=False, top=False, bottom=False, left=False,
                       right=False)

    fig.tight_layout()


@njit
def doppler(x):
    """
    Parameters
    ----------
    x : array-like
        Domain of x is in (0,1]

    """
    if not np.all((x >= 0) & (x <= 1)):
        with objmode():
            raise ValueError("Domain of doppler is x in (0,1]")
    return np.sqrt(x * (1 - x)) * np.sin((2.1 * np.pi) / (x + .05))


@njit
def blocks(x):
    """
    Piecewise constant function with jumps at t.

    Constant scaler is not present in Donoho and Johnstone.
    """
    K = lambda x: (1 + np.sign(x)) / 2.
    t = np.array([[.1, .13, .15, .23, .25, .4, .44, .65, .76, .78, .81]]).T
    h = np.array([[4, -5, 3, -4, 5, -4.2, 2.1, 4.3, -3.1, 2.1, -4.2]]).T
    return 3.655606 * np.sum(h * K(x - t), axis=0)


@njit
def bumps(x):
    """
    A sum of bumps with locations t at the same places as jumps in blocks.
    The heights h and widths s vary and the individual bumps are of the
    form K(t) = 1/(1+|x|)**4
    """
    K = lambda x: (1. + np.abs(x)) ** -4.
    t = np.array([[.1, .13, .15, .23, .25, .4, .44, .65, .76, .78, .81]]).T
    h = np.array([[4, 5, 3, 4, 5, 4.2, 2.1, 4.3, 3.1, 2.1, 4.2]]).T
    w = np.array([[.005, .005, .006, .01, .01, .03, .01, .01, .005, .008, .005]]).T
    return np.sum(h * K((x - t) / w), axis=0)


@njit
def heavisine(x):
    """
    Sinusoid of period 1 with two jumps at t = .3 and .72
    """
    return 4 * np.sin(4 * np.pi * x) - np.sign(x - .3) - np.sign(.72 - x)


def coef_pyramid_plot(coefs, first=0, scale='uniform', ax=None):
    """
    Shows common diagnostic plot of the wavelet coefficients

    Parameters
    ----------
    coefs : array-like
        Wavelet Coefficients. Expects an iterable in order Cdn, Cdn-1, ...,
        Cd1, Cd0.
    first : int, optional
        The first level to plot.
    scale : str {'uniform', 'level'}, optional
        Scale the coefficients using the same scale or independently by
        level.
    ax : Axes, optional
        Matplotlib Axes instance

    Returns
    -------
    Figure : Matplotlib figure instance
        Either the parent figure of `ax` or a new pyplot.Figure instance if
        `ax` is None.
    """

    if ax is None:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)  #, axisbg='lightgrey'
    else:
        fig = ax.figure

    n_levels = len(coefs)
    n = 2 ** (n_levels - 1)  # assumes periodic

    if scale == 'uniform':
        biggest = [np.max(np.abs(np.hstack(coefs)))] * n_levels
    else:
        # multiply by 2 so the highest bars only take up .5
        biggest = [np.max(np.abs(i)) * 2 for i in coefs]

    for i in range(first, n_levels):
        x = np.linspace(2 ** (n_levels - 2 - i), n - 2 ** (n_levels - 2 - i), 2 ** i)
        ymin = n_levels - i - 1 + first
        yheight = coefs[i] / biggest[i]
        ymax = yheight + ymin
        ax.vlines(x, ymin, ymax, linewidth=1.1)

    ax.set_xlim(0, n)
    ax.set_ylim(first - 1, n_levels)
    ax.yaxis.set_ticks(np.arange(n_levels - 1, first - 1, -1))
    ax.yaxis.set_ticklabels(np.arange(first, n_levels))
    ax.tick_params(top=False, right=False, direction='out', pad=6)
    ax.set_ylabel("Levels", fontsize=14)
    ax.grid(True, alpha=.85, color='white', axis='y', linestyle='-')
    ax.set_title('Wavelet Detail Coefficients', fontsize=16,
                 position=(.5, 1.05))
    fig.subplots_adjust(top=.89)

    return fig