import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
# from scipy.fft import fft, fftfreq, fftshift, highpass
from scipy.signal import kaiserord, lfilter, firwin, freqz


import json
pd.set_option('display.max_columns', None)  # for better display tables
pd.set_option('display.width', None)
cfg = {'in': {
    'path': r'd:\WorkData\BalticSea\231121_ABP54\inclinometer@i37,38,58-60\231121.proc_noAvg.h5',
    'table': ['i37', 'i58', 'i59'],  # ['i37', 'i38', 'i58', 'i59', 'i60'],
    'cols': ['u', 'v'],
    }
}

device_dir = Path(cfg['in']['path']).parent

# Load devices info for headers to results
b_info_devices_json_found = (device_dir / 'info_devices.json').is_file()
device_info = {}
if b_info_devices_json_found:
    with Path(device_dir / 'info_devices.json').open(encoding='utf8') as f:
        device_info_loaded = json.load(f)
    for pid_cur in cfg['in']['table']:
        try:
            pid_info = device_info_loaded[pid_cur]
        except KeyError:
            # If fist letter was not 'i' then add it
            if pid_cur[0] == 'i':
                continue
            try:
                pid_info = device_info_loaded[f'i{pid_cur}']
            except KeyError:
                continue
        device_info[pid_cur] = dict(zip('pbdsc', (
            lambda p, b, bd, s, lat=None, lon=None:
                [p, b, None if b is None else round(b - bd, 1), s] + ([(lat, lon)] if lat else [])
        )(*pid_info)))
    print('info_devices.json loaded ok')

file_intervals = r'd:\WorkData\BalticSea\231121_ABP54\meteo\ECMWF\..vsz\intervals_selected.txt'
d = []
with Path(file_intervals).open('rt') as hf:
    hf.readline()
    for line in hf.readlines():
        d.append(datetime.fromisoformat(line.removesuffix('\n')))
    print(len(d), 'intervals loaded ok')

# Filter out frequencies below:
freq_min = 1/60

plot_results = False  # plot graph on each iteration

def main():
    mean_std = {t: {} for t in d}
    mean_std_filt = {t: {} for t in d}
    mdfs = []
    mdfs_filt = []
    val_sum_prev = 0

    sample_rate_prev = None
    qstr_trange_pattern = "index>='{}' & index<='{}'"
    with pd.HDFStore(cfg['in']['path'], mode='r') as storeIn:
        for tbl in cfg['in']['table']:
            for t_st in d:
                qstr = qstr_trange_pattern.format(t_st, t_st + timedelta(minutes=5))
                df = storeIn.select(tbl, where=qstr, columns=cfg['in']['cols'])
                try:
                    sample_rate = 1 / (df.index[1] - df.index[0]).total_seconds()
                except IndexError:
                    print('!', end='', flush=True)
                    continue
                if sample_rate_prev != sample_rate:
                    print('Creating filter for sample rate:', sample_rate)
                    sample_rate_prev = sample_rate
                    # Create_hf to filter signal with the FIR filter by scipy.signal.lfilter.
                    taps, beta = create_hf_filter(cutoff_hz=freq_min, sample_rate=sample_rate)
                    N = len(taps)
                val_sum = 0 
                for col in cfg['in']['cols']:
                    # removing mean since it is not interested and can lead to filtering errors
                    mean = df[col].mean()
                    df[col] -= mean
                    
                    std = df[col].std()
                    mean_std[t_st][col] = std
                    val_sum += std

                    hp = lfilter(taps, 1, df[col])
                    # The first N-1 samples are "corrupted" by the initial conditions
                    mean_std_filt[t_st][col] = hp[(N-1):].std()
                    if plot_results:
                        show_result(np.arange(len(df)) / sample_rate, hp, df[col], sample_rate=sample_rate, N=len(taps))
                print('>' if val_sum_prev > val_sum else '<', end='', flush=True)
                val_sum_prev = val_sum

            mdf = pd.DataFrame.from_dict(mean_std, orient='index')
            mdf_filt = pd.DataFrame.from_dict(mean_std_filt, orient='index')
            mdfs.append(mdf)
            mdfs_filt.append(mdf_filt)
            print(f"{tbl} processed")
    for df_list in [mdfs, mdfs_filt]:
        dfc = pd.concat(df_list, keys=cfg['in']['table'], names=['Depth', 'Time'], axis=1)
        dfc = dfc.rename(columns={pid: f"{device_info[pid]['d']}m" for pid in cfg['in']['table']}).round(3).sort_index(axis=1)
        print(dfc)

    print('ok>')


def create_hf_filter(cutoff_hz=freq_min, sample_rate=1, show_filter=False):
    """High pass filter
    cutoff_hz: The cutoff frequency of the filter.
    """

    # The Nyquist rate of the signal.
    nyq_rate = sample_rate / 2

    # The desired width of the transition from pass to stop,
    # relative to the Nyquist rate.  We'll design the filter
    # with a 5 Hz transition width.
    width = 5/nyq_rate

    # The desired attenuation in the stop band, in dB.
    ripple_db = 60

    # Compute the order and Kaiser parameter for the FIR filter.
    N, beta = kaiserord(ripple_db, width)

    # Use firwin with a Kaiser window to create a highpass (by pass_zero=False) FIR filter.
    taps = firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta), pass_zero=False)
    
    if show_filter:
        """Plot the filter coefficients.
        """
        #------------------------------------------------
        # Plot the FIR filter coefficients.
        #------------------------------------------------

        figure(1)
        plot(taps, 'bo-', linewidth=2)
        title('Filter Coefficients (%d taps)' % N)
        grid(True)

        #------------------------------------------------
        # Plot the magnitude response of the filter.
        #------------------------------------------------

        figure(2)
        clf()
        w, h = freqz(taps, worN=8000)
        plot((w/pi)*nyq_rate, absolute(h), linewidth=2)
        xlabel('Frequency (Hz)')
        ylabel('Gain')
        title('Frequency Response')
        ylim(-0.05, 1.05)
        grid(True)

        # Upper inset plot.
        ax1 = axes([0.42, 0.6, .45, .25])
        plot((w/pi)*nyq_rate, absolute(h), linewidth=2)
        xlim(0,8.0)
        ylim(0.9985, 1.001)
        grid(True)

        # Lower inset plot
        ax2 = axes([0.42, 0.25, .45, .25])
        plot((w/pi)*nyq_rate, absolute(h), linewidth=2)
        xlim(12.0, 20.0)
        ylim(0.0, 0.0025)
        grid(True)

    return taps, beta

# plot_results = True
if plot_results:
    import matplotlib.pyplot as plt
    from pylab import figure, clf, plot, xlabel, ylabel, xlim, ylim, title, grid, axes, show

    def show_result(t, x, filtered_x, sample_rate, N=len(taps)):
        """Plot the original and filtered signals."""
        delay = 0.5 * (N-1) / sample_rate

        figure(3)
        # Plot the original signal.
        plot(t, x)
        # Plot the filtered signal, shifted to compensate for the phase delay.
        plot(t-delay, filtered_x, 'r-', linewidth=1)
        
        # Plot just the "good" part of the filtered signal.  The first N-1
        # samples are "corrupted" by the initial conditions.
        plot(t[N-1:]-delay, filtered_x[N-1:], 'g', linewidth=1)

        xlabel('t')
        grid(True)

        show()

if __name__ == '__main__':
    main()
