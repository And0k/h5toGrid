from pathlib import Path
import numpy as np
import scipy.signal as sp
import pandas as pd
from pandas.tseries.frequencies import to_offset
from datetime import datetime
from re import sub, match
from contextlib import contextmanager
# my
import plot
import sys_path_tcm
from tcm.incl_h5spectrum import h5_velocity_by_intervals_gen, df_interp, init_psd_nc_file
from tcm.utils2init import standard_error_info

if __debug__:
    import matplotlib
    matplotlib.rcParams['axes.linewidth'] = 1.5
    matplotlib.rcParams["lines.linewidth"] = 0.5

    matplotlib.rcParams['figure.figsize'] = (16, 7)
    matplotlib.rcParams['font.size'] = 12
    matplotlib.rcParams["axes.xmargin"] = 0.001  # (default: 0.05)
    matplotlib.rcParams["axes.ymargin"] = 0.01
    try:
        matplotlib.use(
            "Qt5Agg"
        )  # must be before importing plt (raises error after although documentation sed no effect)
    except ImportError:
        pass
    from matplotlib import pyplot as plt, dates as mdates, ticker as mticker
    from matplotlib.colors import LogNorm
    matplotlib.interactive(True)

plt.style.use('bmh')
plt.rcParams["lines.linewidth"] = 0.5

def get_trend(signal):
    """Trend including mean from the signal"""
    x = np.arange(signal.size)
    coef_trend = np.polyfit(x, signal, 1)
    trend = np.polyval(coef_trend, x)  # Trend line
    return trend


def add_custom_freq_labels(
    ax,
    freq_edges=None,
    side="x",
    disp_periods=[
        to_offset(dt)
        for dt in [
            "5s",
            "30s",
            "1min",
            "5min",
            "10min",
            "30min",
            "1h",
            "2h",
            "6h",
            "12h",
            "1D",
            "2D",
            "4D",
            "8D",
            # "10D",
        ]
    ],
):
    """
    Add custom labels on the left side
    """

    disp_freq = 1e9 / np.array([dt.nanos for dt in disp_periods])
    if freq_edges:
        b_disp_in_range = (freq_edges[0] < disp_freq) & (disp_freq <= freq_edges[-1])
        disp_freq = disp_freq[b_disp_in_range]
        disp_periods_labels = [
            f"1{dt.freqstr}" if dt.n == 1 else f"{dt.freqstr}"
            for dt, b in zip(disp_periods, b_disp_in_range)
            if b
        ]
    else:
        disp_periods_labels = [
            f"1{dt.freqstr}" if dt.n == 1 else f"{dt.freqstr}"
            for dt in disp_periods
        ]

    text_args = {
        "fontsize": 12,
        "color": "blue",
    }
    if side=="x":
        text_args.update(
            {
                "transform": ax.get_xaxis_transform(),
                "ha": "center",
                "va": "bottom",
            }
        )
        def text(fr, lb):
            return ax.text(fr, 0, lb, **text_args)
    else:
        text_args.update(
            {
                "transform": ax.get_yaxis_transform(),
                "ha": "right",
                "va": "center",
            }
        )
        def text(fr, lb):
            return ax.text(0, fr, lb, **text_args)

    tx = [text(fr, lb) for fr, lb in zip(disp_freq, disp_periods_labels)]
    return tx


def format_to_timedelta(x):
    if not x:
        return ""
    td = pd.Timedelta(x, "s")

    components = td.components
    units = [
        ("D", "days"),
        ("h", "hours"),
        ("min", "minutes"),
        ("s", "seconds"),
        ("ms", "milliseconds"),
        ("us", "microseconds"),
        ("ns", "nanoseconds"),
    ]

    unit_parts = [
        f"{getattr(components, attr)}{abbrev}"
        for abbrev, attr in units
        if getattr(components, attr) > 0
    ]
    unit_str = " ".join(unit_parts) if unit_parts else "0s"

    total_seconds = td.total_seconds()
    if total_seconds >= 86400:
        return unit_str
    elif total_seconds >= 1:
        std_str = str(td).split(".")[0]
        return unit_str if len(unit_str) < len(std_str) else std_str
    else:
        return unit_str


# Define major tick frequencies
class CustomMajorLocator(mticker.FixedLocator):

    # Corresponding units
    units = {
        1: 'seconds',
        10: 'seconds',
        60: 'minutes',
        600: 'minutes',
        3600: 'hours',
        36000: 'hours',
        86400: 'days',
        864000: 'days'
    }

    def __init__(self):
        # Define time intervals in seconds
        time_intervals = list(self.units.keys())
        super().__init__(time_intervals)

    # def tick_values(self, vmin, vmax):
    #     # Get all predefined freqs
    #     ticks = self.locs
    #     # Filter ticks within the view limits
    #     ticks = [tick for tick in ticks if vmin <= tick <= vmax]
    #     return ticks
    # secax.xaxis.set_major_locator(CustomMajorLocator())


class CustomMinorLocator(mticker.LogLocator):
    def __init__(self, major_intervals=list(CustomMajorLocator.units), units=CustomMajorLocator.units):
        super().__init__()
        self.major_intervals = major_intervals
        self.units=units

    def tick_values(self, vmin, vmax):
        # todo: check why hangs

        # Find the major interval that vmin falls into

        # Ensure vmin is not less than the first major interval
        vmin = max(vmin, self.major_intervals[0])

        # Find the index of the major interval just less than or equal to vmin
        index = np.searchsorted(self.major_intervals, vmin, side="right") - 1
        if index < 0:
            index = 0

        # Get the unit based on the major interval
        unit = self.units[self.major_intervals[index]]

        # Determine the step for minor ticks based on the unit
        if unit == "seconds":
            step = 1
        elif unit == "minutes":
            step = 10
        elif unit == "hours":
            step = 60
        elif unit == "days":
            step = 600
        else:
            step = 1
        # Generate minor ticks
        minor_ticks = []
        for i in range(int(vmin), int(vmax), step):
            minor_ticks.append(i)
        return minor_ticks



class InvertedLogTimeLocator(mticker.LogLocator):
    # for secax.xaxis.set_major_locator(InvertedLogTimeLocator())
    def __init__(self, bases=(1, 60, 3600, 86400)):
        self.bases = bases
        super().__init__()

    def tick_values(self, vmin, vmax):
        # Frequencies corresponding to 1s, 1m, 1h, 1d
        freqs = [1 / 86400, 1 / 3600, 1 / 60, 1]
        # Filter frequencies within the current view limits
        freqs = [f for f in freqs if vmin <= f <= vmax]
        return np.array(freqs)


class CustomLocator(mticker.FixedLocator):
    freqs = [1 / 86400, 1 / 3600, 1 / 60, 1]

    def __init__(self, locs=freqs, nbins=None):
        super().__init__(locs)

    def tick_values(self, vmin, vmax):
        # Frequencies corresponding to 1s, 1m, 1h, 1d
        freqs = [1 / 86400, 1 / 3600, 1 / 60, 1]
        # Filter frequencies within the current view limits
        freqs = [f for f in freqs if vmin <= f <= vmax]
        return np.array(freqs)
    # secax.xaxis.set_major_locator(CustomLocator())



# Custom Locator to handle inverted values
class InvertedDateLocator(mdates.AutoDateLocator):
    def tick_values(self, vmin, vmax):
        # Invert the range for tick generation
        inv_vmin, inv_vmax = 1 / vmax, 1 / vmin
        # Generate ticks using the inverted range
        ticks = super().tick_values(*mdates.num2date([inv_vmin, inv_vmax]))
        # Invert the ticks back to the original scale
        return 1 / ticks


class LogTimeLocator(mticker.LogLocator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def tick_values(self, vmin, vmax):
        # Get potential tick positions using AutoDateLocator
        auto_locator = InvertedDateLocator()  # mdates.AutoDateLocator
        auto_ticks = auto_locator.tick_values(vmin, vmax)

        # Generate logarithmic tick positions within the data range
        log_ticks = np.logspace(np.log10(vmin), np.log10(vmax), num=10)
        # Find the closest auto_ticks to log_ticks
        closest_indices = [np.argmin(np.abs(auto_ticks - tick)) for tick in log_ticks]

        final_ticks = auto_ticks[closest_indices]
        return final_ticks



def inv_prop(x):
    # inverse proportional function for the spectrum's frequncy / time axis
    return 1 / x


def apply_notch_filter(
    signal: np.ndarray, fs: float, freq: float, bandwidth: float
) -> np.ndarray:
    """
    Design and apply a notch filter to the input signal.

    :param signal: Input 1D signal (non-stationary).
    :param fs: Sampling frequency in Hz.
    :param freq: Target frequency to suppress in Hz.
    :param bandwidth: Bandwidth around freq in Hz for rejection.
    :return: Filtered signal.
    """
    # Design the notch filter

    # f0_norm = freq / (fs / 2)  # Normalize target frequency
    # bw_norm = bandwidth / (fs / 2)  # Normalize bandwidth

    b, a = sp.iirnotch(freq, Q=freq / bandwidth, fs=fs)

    # Apply the filter to the signal
    filtered_signal = sp.filtfilt(b, a, signal)
    return filtered_signal

def gen_separate_probes(file_csv):
    df_in = pd.read_csv(file_csv, index_col="Time", date_format="ISO8601", dtype=np.float32)
    time_edges = df_in.index[[0,-1]]
    print(f"Loaded data length: {df_in.shape[0]} points, {time_edges}")
    n_t_cols = int(df_in.columns.size/2)
    temp_cols = df_in.columns[:n_t_cols]
    press_cols = df_in.columns[n_t_cols:]
    yield (
        df_in[temp_cols].rename(
            columns={
                c: f"t({v}m)"
                for c, (col_p, v) in zip(
                    temp_cols, df_in[press_cols].mean().round().astype(int).items()
                )
            },
        )
        ,
        "t",
        ""
    )
    if False:
        for col, ser in df_in[t_cols].items():
            tbl = col
            data_name = ""  # f'{tbl}/PSD_{start_end[0]}{data_name_suffix}'
            yield (ser.to_frame(), tbl, data_name)


# #########################################################################

if __name__ == "__main__":
    db_path_in = Path(r"C:\Work\_\t-chain\240625@TCm1,2.csv")
    # db_path_in = Path(r"C:\Work\_\t-chain\240625isolines(t)@TCm1,2.h5")
    param_letter = "t"

    # FIR path
    filter_path = Path(
        r"C:\Work\Python\AB_SIO_RAS\h5toGrid\scripts\cfg\Fourier_filters\band_stop10-15h_filter_fc=1min^-1.npz"
    )  # Numerator and Denominator coefficients: filt_coef["ba"]:
    filt_coef = np.load(filter_path)

    # set None to not save figure:
    fig_save_suffix = "v1"  # "no_detrand" #
    print("Start calculation to save", fig_save_suffix, "data")
    cfg = {
        "in": {"fs": 0.016666666666666666, "db_path": db_path_in, "tables": [".*"]},
        "out": {
            "db_path": db_path_in.with_suffix(".filtered.nc"),  # comment to not save to *.nc
            "table": "psd",
        },
        "proc": {"fmin": 1 / (3600 * 24), "fmax": 1 / (120)},
    }
    detrend = "linear"

    # # Stop band center frequency
    period_sb_h = 13.8  # 14
    # Stop band filter parameters to remove it
    fsb_center = 1 / (period_sb_h * 3600)
    # Bandwidth in Hz (for edge periods ± 1 hour)
    fsb_range = [1 / ((period_sb_h + 1) * 3600), 1 / ((period_sb_h - 1) * 3600)]
    bw = np.diff(fsb_range).item()


b_plot = True

# Downsample the signal if need
fs_desired = 1 / 3600  # None  # resolution  # for 1-min to 1-hour set: 1 / 3600
if fs_desired and not fs_desired == cfg["in"]["fs"]:
    downsample_factor = int(cfg["in"]["fs"] / fs_desired)
    fs = fs_desired
else:
    fs = cfg["in"]["fs"]

# Output data time range variables initialization
time_good_min, time_good_max = pd.Timestamp.max, pd.Timestamp.min

fig = None
nc_root = None

out = {}  # filtered time series
psd = {}  # original time series PSD
stats = {}
# ".h5" or ".csv"
for df, tbl, dataname in (
    gen_separate_probes(db_path_in)
    if db_path_in.suffix == ".csv"
    else h5_velocity_by_intervals_gen(cfg, cfg["out"])
):
    cols = df.columns.to_list()
    print(f"{tbl}:", cols)
    bad_source_index = df.index.values
    df, bads = df_interp(df, cfg["in"]["fs"], cols=cols, method="pchip")
    for i1_col, (col, bad_source) in enumerate(bads.items(), start=1):
        if fs_desired:
            # Downsample
            signal = sp.decimate(df[col].values, downsample_factor, zero_phase=True)
            signal_bads_to_0 = sp.decimate(
                np.where(bads[col], 0, df[col].values), downsample_factor, zero_phase=True,
            )
            df_decimated_len = signal.size
            _ = np.zeros(df_decimated_len, dtype=bool)
            _[bads[col][::downsample_factor][:df_decimated_len]] = True
            bads[col] = _
            time_index = df.index[::downsample_factor]
            # Compute edges for time intervals (use midpoint-based decimation logic):
            # time_edges has one more element than the downsampled signal,
            # representing the boundaries of each time interval.
            time_edges = pd.date_range(
                start=time_index[0],
                end=time_index[-1] + pd.Timedelta(seconds=1 / fs_desired),
                periods=len(signal) + 1,
            )
        else:
            signal = df[col].values
            time_index = df.index
            time_edges = time_index[[0, -1]]

        # signal mean without initial nan regions and mean of interpolated signal
        signal_mean0 = (
            signal_bads_to_0.mean() if fs_desired else df[col].values[~bads[col]].mean().item()
        )
        signal_mean = signal.mean()

        # To approximate filtering procedure to calculation of PSD, but not completely
        # as on calculation of PSD the detrending is performed on each FFT interval
        if detrend == "constant":
            trend = signal_mean
        elif detrend == "linear":
            trend = get_trend(signal)
        else:
            raise NotImplementedError('Can not detrend with "{detrend}" function')
        s_detrended = signal - trend

        # variant 1: notch IIR filter
        signal_notch = apply_notch_filter(s_detrended, fs, fsb_center, bw) + trend

        # variant 2: bandstop FIR filter
        try:
            signal_fir = sp.filtfilt(*filt_coef["ba"], s_detrended) + trend
        except ValueError:  #  The length of the input vector x must be greater than padlen
            # too big filter for decimated singal: filter original then decimate
            trend_orig = get_trend(df[col].values) if detrend == "linear" else trend
            signal_fir = (
                sp.filtfilt(*filt_coef["ba"], df[col].values - trend_orig) + trend_orig
            )
            signal_fir = sp.decimate(signal_fir, downsample_factor, zero_phase=True)
        # sp.lfilter(*filt_coef["ba"], s_detrended) + trend

        # # PSD Spectrum
        # Welch's method with windowing and zero-padding
        # Number of samples per segment
        nperseg_min = int(period_sb_h * 3600 * fs)
        nperseg = 2 ** int(np.log2(signal.size / 10))
        print(
            "Selecting Number of samples per segment between",
            nperseg_min,
            f"and {signal.size}:",
            nperseg,
        )
        nfft = 2 * nperseg  # Zero-padding for more points (for visually better graph)

        # Original signal spectrum
        freq, psd[col] = sp.welch(
            signal + trend, fs=fs, nperseg=nperseg, nfft=nfft, detrend=detrend
        )

        # check we can continue to use frequncy from previous iteration to save one PSD dataframe for all cols
        if "freq" in psd:
            assert np.allclose(psd["freq"], freq)
        else:
            psd["freq"] = freq


        # # Plotting results
        if b_plot:
            fig, (ax1, ax2) = plt.subplots(2, 1)
            ax1.set_title("Original and filtered time series and their magnitude spectrum")
            ax1.clear()

            # ## Time series
            ax1.plot(time_index, np.where(bads[col], np.nan, signal), "r", label="Original")
            if isinstance(trend, np.ndarray):
                ax1.plot(time_index, trend, "k", label="Global trend")
            ax1.plot(
                time_index,
                np.where(bads[col], np.nan, signal_notch),
                "g",
                label=f"{period_sb_h}h period notched",
                marker="",
            )
            ax1.plot(
                time_index, np.where(bads[col], np.nan, signal_fir), "b", label="FIR"
            )
            ax1.legend()

            # ## PSD axes
            # Better spectrum x-axis (must be done before set axes log)
            # - Move the original bottom spine to the top
            ax2.spines["bottom"].set_position(
                ("axes", 1.0)
            )  # Move the bottom spine to the top
            # Invert the tick direction for the bottom spine since it's now at the top
            ax2.xaxis.tick_top()
            # remove the original top spine to avoid confusion
            ax2.spines["top"].set_visible(False)
            secax = ax2.secondary_xaxis("bottom", functions=(inv_prop, inv_prop))

            # ## PSD plot
            # plots in DB:
            # ax2.psd(
            #     signal + trend,
            #     Fs=fs,
            #     NFFT=nfft,
            #     detrend="linear",
            #     noverlap=int(nfft / 2),
            #     sides="onesided",
            #     label="Original",
            #     color="k",
            # )
            # We plot in PSD units

            # default params:
            # noverlap = nperseg // 2      # Overlap between segments (e.g., 50%)
            # window = hann(nperseg)       # Hann window
            # return_onesided=True, scaling='density'

            b_log_x = True
            plot.vs_freq(
                pd.DataFrame(
                    {
                        "Original": psd[col],
                        f"{period_sb_h}h period notched": sp.welch(
                            signal_notch,
                            fs=fs,
                            nperseg=nperseg,
                            nfft=nfft,
                            detrend=detrend,
                        )[1],
                        "FIR": sp.welch(
                            signal_fir,
                            fs=fs,
                            nperseg=nperseg,
                            nfft=nfft,
                            detrend=detrend,
                        )[1],
                    },
                    index=psd["freq"],  # index_name="freq"
                ),
                ax2,
                legend_title=sub(r"(\d)p(\d)", r"\1.\2", col).replace(
                    "t_", "t="
                ),  # "t, °C",
                b_log_x=b_log_x,
                path_dir=None,  # save later  # db_path_in.parent
                ylabel_prefix=f"PSD({param_letter})",
                save_name_parts=(
                    "gray_subdir?",
                    "ylabel_prefix",
                    f"{param_letter}(t)@t-chain",
                    "log(x)?",
                    ".png",
                ),
                colors=["red", "green", "blue", "black"],
                no_labels_till_save=False,
                plot_kwargs={"marker": ""},
            )

            # ax2.magnitude_spectrum(signal + trend, Fs=fs, color="r", scale="dB", label="Original")
            # ax2.magnitude_spectrum(signal_notch, Fs=fs, color="g", linestyle=':', scale="dB", label=f"{period_sb_h}h period notched")
            # ax2.magnitude_spectrum(signal_fir, Fs=fs, color="b", linestyle=':', scale="dB", label="FIR") # Filtered Signal
            if param_letter=="z":
                ax2.set_ylim([1, 8e6])
            else:
                ax2.set_ylim([0.1, 1e5])

            #  use same (previous used) limits to not depend on decimation
            freq_edges = [cfg["in"]["fs"] / (1024 * 32), fsb_range[-1] * 20]
            # default lims: [fs / nfft, fsb_range[-1] * 20]

            ax2.set_xlim(*freq_edges)
            secax.set_xlim(*freq_edges)

            secax.xaxis.set_major_locator(CustomMajorLocator())
            # secax.xaxis.set_major_locator(InvertedDateLocator())
            # Define a custom formatter for the bottom axis to show 1/x
            secax.xaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, pos: format_to_timedelta(x))  # secax.xaxis.base
            )
            # Set minor ticks: works only in interactive mode
            secax.xaxis.set_minor_locator(CustomMinorLocator())
            secax.xaxis.set_minor_formatter(mticker.NullFormatter())  # helps with log axis issues
            add_custom_freq_labels(ax2, freq_edges)

            # plt.tight_layout()
            plt.show()

            if fig_save_suffix:  # dbstop
                try:
                    fig.savefig(
                        db_path_in.with_name(
                            "{}@t-chain-{}{}.png".format(
                                f"{param_letter}({col.split('_')[-1]}))"
                                if param_letter != "t"
                                else col,
                                f"d{downsample_factor}-" if fs_desired else "",
                                fig_save_suffix,
                            )
                        ),
                        dpi=300,
                        bbox_inches="tight",
                    )
                except Exception as e:
                    print(f'Can not save fig: {standard_error_info(e)}')
            plt.close(fig.number)
        if cfg["out"].get("db_path"):
            out[col] = np.where(bads[col], np.nan, signal_notch)


# # Save data

if cfg["out"].get("db_path"):
    df_out = pd.DataFrame(
        out,
        index=time_index,
    )

    table_prefix = match(r"^\d*([^@\(]+)", cfg["in"]["db_path"].stem).group(1)
    if table_prefix.isdigit():
        table_prefix = param_letter
    file_out = cfg["out"]["db_path"].with_name(
        "".join([
            cfg["out"]["db_path"].stem,
            f"d{downsample_factor}-" if fs_desired else ""
        ])
    ).with_suffix(".filtered.h5")
    print("Saving to", cfg["out"]["db_path"], end=f' to "{table_prefix}..."\n')
    table = "".join([table_prefix, f"d{downsample_factor}" if fs_desired else "", "_filtered"])
    with pd.HDFStore(file_out, mode="a") as store:
        print(fr'filtered nan-interpolated signal to "{table}"')
        df_out.to_hdf(
            store,
            key=table,
            append=True,
            data_columns=True,
            format="table",
            index=False,
            # dropna=
        )

    # ## Save PSD
    file_psd = cfg["out"]["db_path"].with_name("{}_PSD_Welch.h5".format(cfg["out"]["db_path"].name.split(".")[0]))
    b_file_exist = file_psd.is_file()
    table = table_prefix + "_psd"
    with pd.HDFStore(file_psd, mode="a") as store:
        if b_file_exist and store.get_node(table):
            print(f"skipping to save {file_psd.name} / {table} - table exist")
        else:
            df_psd = pd.DataFrame.from_records(
                psd,
                index="freq",
            )
            print(r'original signal\'s PSD to "{} / {}"'.format(file_psd, table))
            df_psd.to_hdf(
                store,
                key=table,
                append=True,
                data_columns=True,
                format="table",
                index=False,
                # dropna=
            )

print("OK>")
