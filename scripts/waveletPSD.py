from pathlib import Path
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
from pandas.tseries.frequencies import to_offset
from datetime import datetime
import pywt  # PyWavelets (PyWT)
from tcm.incl_h5spectrum import h5_velocity_by_intervals_gen, df_interp, init_psd_nc_file
from tcm.utils2init import standard_error_info
from re import sub

if __debug__:
    import matplotlib
    matplotlib.rcParams['axes.linewidth'] = 1.5
    matplotlib.rcParams['figure.figsize'] = (16, 7)
    matplotlib.rcParams['font.size'] = 12
    try:
        matplotlib.use(
            "Qt5Agg"
        )  # must be before importing plt (raises error after although documentation said no effect)
    except ImportError:
        pass
    from matplotlib import pyplot as plt, dates as mdates
    from matplotlib.ticker import LogLocator, LogFormatter
    from matplotlib.colors import LogNorm
    # from matplotlib import rc

    # rc('text',usetex=True)
    # rc('text.latex', preamble=r'\usepackage{color}')
    matplotlib.interactive(True)

plt.style.use('bmh')

db_path_in = Path(r"C:\Work\_\t-chain\240625isolines(t)@TCm1,2.h5")


cfg = {
    "in": {"fs": 0.016666666666666666, "db_path": db_path_in, "tables": [".*"]},
    "out": {
        # "db_path": db_path_in.with_suffix(".WT.nc"),  # comment to not save to *.nc
        "table": "scalogram",
    },
    "proc": {"fmin": 1 / (5 * 24 * 3600), "fmax": 1 / (120)},
}

def find_regions_with_more_ones(b_arr, window_size, min_sum=None):
    if min_sum is None:  # ones are more frequent than zeros
        min_sum = window_size / 2
    # Create a sliding window view of the array
    windows = sliding_window_view(b_arr, window_shape=window_size)

    # Count the number of 1s in each window
    count_ones = np.sum(windows, axis=1)

    # Check where 1s are more frequent than min_sum
    regions = count_ones > min_sum
    # Pad the regions array to match the input size
    pad_size = window_size // 2
    output = np.pad(
        regions,
        (pad_size, window_size - pad_size - 1),
        mode="edge",
        # constant_values=0,
    )
    return output


def cwt_with_nans(data, scales_rel, wavelet_obj=None, nans=None):
    # Initialize coefficients array
    coefficients = np.zeros((len(scales_rel), len(data)), dtype=complex)
    if nans is None:
        nans = np.isnan(data)
    # Compute CWT with NaN handling
    for i, scale in enumerate(scales_rel):
        width = wavelet_width(scale, wavelet_name)
        half_width = int(np.ceil(width / 2))
        for t in range(len(data)):
            start = max(0, t - half_width)
            end = min(len(data), t + half_width)
            segment = data[start:end]
            nan_ratio = np.isnan(segment).mean()
            if nan_ratio < 0.5:  # Threshold: less than 50% NaNs
                segment = np.nan_to_num(segment)  # Replace NaNs with zeros
                coefficients[i, t] = (
                    segment
                    * np.conj(wavelet_obj.wavefun(scale=scale, length=len(segment))[0])
                ).sum()
            else:
                coefficients[i, t] = np.nan
    return coefficients

disp_periods = [
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
        # "10D",
    ]
]


def plot_wavelet_spectrogram(
    magnitudes: np.ndarray,
    freq_edges,
    time_edges,
    title="",
    file=None,
    fig=None,
    time_format="%m/%d",
    vmin_clip=None,
    vmax_clip=None,
    b_log_freq=True,
    **imshow_kwargs,  # '%Y-%m-%d %H:%M'
):
    """
    Display a data matrix in logarithmically y-values scale
    :param magnitudes: data which rows correspond to equally or logarithmically spaced y if `b_log_freq`
    :param freq_edges: 1D numpy array, which 1st and last values are y coordinates of edges of data.
    :param time_edges: _description_
    :param title: figure header
    :param file: file path to save figure
    :param vmin_clip: clip autofinded vmin to this v min value
    :param vmax_clip: clip autofinded vmax to this v max value
    :param fig: figure object with previous for faster drawing with same scale by reuse axes
    :param time_format: _description_, defaults to '%m/%d'
    :return: figure object
    """
    cbar = None
    if b_log_freq:  # data calculated at logarithmically spaced y
        # Data already in log scale, so to return to it from displaying scale ax_y there is path back
        def inverse(y):
            """
            Linear to log scale with clipping preventing "divide by zero encountered in log10" message
            :param y:
            :return:
            """
            return np.log10(np.where(y > 0, y, 1e-100))
    else:
        def inverse(y):
            return y

    if fig:
        ax_y, ax, *_ = fig.axes[:2]
        if b_log_freq:
            ax, ax_y = ax_y, ax
        # Remove the AxesImage (imshow) from the given axes
        for artist in ax.get_children():
            if isinstance(artist, matplotlib.image.AxesImage):
                cbar = artist.colorbar  # to reuse
                artist.remove()
                # break
    else:
        if b_log_freq:  # data calculated at logarithmically spaced y
            _, ax = plt.subplots()
            # Define log y-axis ax_y on the left side that depends on linear ax' y-axis

            def forward(y):  # log scale to linear
                return np.float_power(10, y)  #  10**y not supports negative y

            ax.tick_params(axis="y", right=True)  # to work after copied?
            ax_y = ax.secondary_yaxis("left", functions=(forward, inverse))
            # Hide the y-axis. It need only to plot image on it
            ax.yaxis.set_visible(False)

            # Set the tick positions and labels for the secondary y-axis
            ax_y.yaxis.set_major_locator(LogLocator(numticks=10))  # Automatically place major ticks
            ax_y.yaxis.set_major_formatter(LogFormatter())  # Format ticks as powers of 10
            ax_y.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10))
        else:  # data calculated at equally spaced y
            _, ax = plt.subplots()
            ax_y = ax
            ax.set_yscale("log")
        ax_y.grid(True, linestyle="--", zorder=10)  # color="blue",

        # y-axis titles
        ax.text(-0.05, 0.98, r"f, Hz", transform=ax.transAxes, fontsize=14)
        ax.text(-0.05, 0.94, r"dt", transform=ax.transAxes, fontsize=14, color='darkblue', rotation=10)


        # adjust ticks visibility
        ax.spines["left"].set_visible(False)
        ax_y.set_zorder(100)
        for spine in ax.spines.values():
            spine.set_zorder(0)
        ax.tick_params(axis="x", top=True) # zorder=10 not works
        # ax_y.tick_params(axis="y", right=True)  # - makes outside ticks what I don't want
        # Set tick positions explicitly
        # Ticks on both top and bottom
        # ax_y.yaxis.set_ticks_position("both")  # Ticks on both left and right - don't works


        # ax_y.spines["left"].set_position(("outward", 0))
        # ax_y.tick_params(axis="y", direction="inout", length=10, pad=-5)  # , width=1.5



        # Time period labels on y-axis
        disp_freq = 1e9 / np.array([dt.nanos for dt in disp_periods])
        b_disp_in_range = (freq_edges[0] < disp_freq) & (disp_freq <= freq_edges[-1])
        disp_freq = disp_freq[b_disp_in_range]
        disp_periods_labels = [
            f"1{dt.freqstr}" if dt.n == 1 else f"{dt.freqstr}"
            for dt, b in zip(disp_periods, b_disp_in_range) if b
        ]

        # Add custom labels on the left side
        tx = [
            ax.text(
                0,  # x
                fr,
                lb,
                transform=ax.get_yaxis_transform(),  # Use +999 for x, data coordinates for y
                # fontsize=12,
                rotation=10,
                color="darkblue",
                ha="right",
                va="center",

            )
            for fr, lb in zip(inverse(disp_freq), disp_periods_labels)
        ]

        # x-axis. Convert from timestamps to human-readable datetime labels

        # Set major ticks to be at the start of each week (Monday)
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter(time_format))



    vmax = np.nanmax(magnitudes)
    if vmax_clip is not None and vmax > vmax_clip:
        vmax = vmax_clip
    vmin = np.nanmin(magnitudes) or 10 ** round(np.log10(vmax) - 7)  # make > 0 requirement for log scale
    if vmin_clip is not None and vmin < vmin_clip:
        vmin = vmin_clip

    im = ax.imshow(
        magnitudes,
        extent=[*time_edges[[0, -1]], *inverse((freq_edges[[0, -1]]))],
        # vmin=magnitudes.min(),
        # vmax=magnitudes.max(),
        **imshow_kwargs
        or {
            "cmap": "Spectral_r",  # nipy_spectral/rainbow/jet
            "aspect": "auto",
            "origin": "lower",
        },
        norm=LogNorm(vmin=vmin, vmax=vmax),
        interpolation="nearest",  # Reduce interpolation artifacts
    )
    if not cbar:
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        # Add the colorbar title at y=1
        cbar.ax.text(
            0.5, 1.05, "|W(z)|²",
            transform=cbar.ax.transAxes,  # Use axes coordinates
            ha='center', va='bottom', fontsize=14
        )  #
    else:
        cbar.update_normal(im)
        # Manually set the colorbar attribute of new im to can find it next time function calling
        im.colorbar = cbar

    plt.xlabel("Time")
    if title:
        plt.title(title, fontsize=14)
    plt.show()

    if file:  # dbstop
        try:
            ax.figure.savefig(file, dpi=300, bbox_inches="tight")
            print(file.name, "saved")
        except Exception as e:
            print(f'Can not save fig: {standard_error_info(e)}')
    return ax.figure

b_plot = True
b_dwt = False  # not tested
b_log_freq = True  # other not implemented (I'm to silly)

# Downsample the signal if need
fs_desired = None  # resolution  # for 1-min to 1-hour set: 1 / 3600
if fs_desired and not fs_desired == cfg["in"]["fs"]:
    import scipy.signal as sp
    downsample_factor = int(cfg["in"]["fs"] / fs_desired)
    fs = fs_desired
else:
    fs = cfg["in"]["fs"]

if not b_dwt:
    # User input:
    num_scales = 400
    if b_log_freq:
        frequencies = np.logspace(
            np.log10(cfg["proc"]["fmin"]), np.log10(cfg["proc"]["fmax"]), num=num_scales
        )
    else:
        frequencies = np.linspace(
            cfg["proc"]["fmin"], cfg["proc"]["fmax"], num=num_scales
        )

    # Morlet Wavelet
    bw, cf = 1.5, 1.0  # bandwidth frequency and center frequency for cmor family
    # bw, cf = 2, 0.5  # For resolving low frequencies, adjust cw to 0.3–0.5 and keep bw at around 1.5–2.0
    # bw, cf = 1.5, 1.2

    # A higher bw results in a wavelet that is more localized in time (narrower time support)
    wavelet = f"cmor{bw}-{cf}"  # frequency2scale not works for "fbsp1-{bw}-{cf}"

    # fbsp name should take the form fbspM-B-C where M is the spline order and B, C are floats representing the bandwidth frequency and center frequency, respectively (example: fbsp1-1.5-1.0).

    # scales relative to sampling frequncy for pywt

    scales_rel = pywt.frequency2scale(wavelet, frequencies / fs, precision=8)
    # cf / (frequencies * fs)

    # frequencies
    freqs = pywt.scale2frequency(wavelet, scales_rel) * fs

# open and start saving to NetCDF
if cfg["out"].get("db_path"):
    nc_root, nc_psd = init_psd_nc_file(**cfg["out"], dt_interval=cfg["proc"].get("dt_interval"))
    nc_psd.createDimension('freq', len(freqs))
    # nv_... - variables to be used as ``NetCDF variables``
    nv_freq = nc_psd.createVariable('freq', 'f4', ('freq',), zlib=True)
    nv_freq[:] = freqs
else:
    nc_root = None

# Output data time range variables initialization
time_good_min, time_good_max = pd.Timestamp.max, pd.Timestamp.min

fig = None
for df, tbl, dataname in h5_velocity_by_intervals_gen(cfg, cfg['out']):
    cols = df.columns.to_list()
    print(f"{tbl}:", cols)

    bad_source_index = df.index.values
    df, bads = df_interp(df, cfg["in"]["fs"], cols=cols, method="pchip")
    out = {}
    for col, bad_source in bads.items():
        if fs_desired:
            # Downsample
            signal = sp.decimate(df[col].values, downsample_factor, zero_phase=True)

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
            time_edges = df.index[[0, -1]]

        if b_dwt:
            # Perform DWT

            wavelet = 'db4'  # Discrete wavelet
            max_level = pywt.dwt_max_level(len(signal), wavelet)
            coeffs = pywt.wavedec(signal, wavelet, level=max_level, mode="symmetric")
            # Prepare data for imshow
            coef = np.array([
                np.pad(c, (0, len(signal) - len(c)), mode='constant', constant_values=np.nan)
                for c in coeffs[1:]
            ])
            # arr, coeff_slices = pywt.coeffs_to_array(coeffs)
            scales_rel = np.power(2, np.arange(1, coef.shape[0] + 1))
        else:
            # масштабы и их частоты
            t_range = np.diff(df.index[[0, -1]].values).astype("m8[s]").astype(int).item()  # total_seconds

            # min scale to get range from Nyquist Frequency = fs/scale[0] < fs/2
            # max scale ≈ T_wavelet / t_range = 2*pi / (fs * t_range)
            # For Morlet wavelet: T_wavelet ≈ k / cf, where k ≈ 1
            scale_end = t_range / (1 / cf)
            assert(scale_end > scales_rel[0])

            if True:
                coef, _ = pywt.cwt(
                    signal,
                    scales_rel,
                    wavelet,
                    # sampling_period=1 / fs,
                    method="fft",  # faster for big data
                )
                # mark as bad on top on maximum frequencies side, skipping single bad values
                # i_bad = np.searchsorted(
                #     df.index.values,
                #     bad_source_index[
                #         bad_source & np.movavg(np.int8(bad_source == 0), to_end=1).astype(np.bool_)
                #     ],
                # )
                b_bad = np.zeros(df.index.size, dtype=np.bool)
                i_bad = np.searchsorted(df.index.values, bad_source_index[bad_source])

                if i_bad.any():
                    i_bad = i_bad[i_bad < coef.shape[1]]
                    b_bad[i_bad] = True
                    b_bad = find_regions_with_more_ones(b_bad, 7)
                    coef[int(num_scales / 2):, b_bad] = np.nan  # set to NaN several values to be visible
                # np.testing.assert_almost_equal(
                #     _, freqs, decimal=7, err_msg="", verbose=True
                # )  # assert (freqs_ == freqs).all()  # not works
            else:
                coef = cwt_with_nans(signal, scales_rel, pywt.ContinuousWavelet(wavelet))

        out[col] = np.abs(coef)**2
        if nc_root:
            if tbl not in nc_psd.groups:
                nc_tbl = nc_psd.createGroup(tbl)
                cols = df.columns
                for c in cols:
                    nc_tbl.createVariable(c, 'f4', ('time', 'freq',), zlib=True)
            nc_tbl.variables[col][:, :] = np.abs(coef).T
            # Update overall min and max
            if time_good_min.to_numpy() > df.index[0].to_numpy():
                # to_numpy('<M8[ns]') get values to avoid tz-naive/aware comparing restrictions
                time_good_min = df.index[0]
            if time_good_max.to_numpy() < df.index[-1].to_numpy():
                time_good_max = df.index[-1]

        if b_plot:
            file_stem = f'Scalogram(z({col.replace("z_t", "t=")}))_{wavelet.replace(".", "p")}'
            title = "{} °C isoline scalogram (wavelet: {})".format(
                sub(r"_t(\d)", r"|t=\1", sub(r"(\d)p(\d)", r"\1.\2", col)),
                f"{'CMOR' if wavelet.startswith('cmor') else wavelet.split('-')[0].upper()}"
                f" with bandwidth {bw}, center frequency {cf}"
                if wavelet.startswith(("cmor", "fbsp1"))
                else wavelet.upper(),
            )
            fig = plot_wavelet_spectrogram(
                out[col],
                freqs,
                time_edges=time_edges,
                title=title,
                file=(db_path_in.parent / file_stem).with_suffix(".png"),
                fig=fig,
                vmin_clip=5e-4,
                vmax_clip=1e4,
                b_log_freq=b_log_freq,
            )

if nc_root:
    nc_psd.variables['time_good_min'][:] = np.array(time_good_min.value, 'M8[ns]')
    nc_psd.variables['time_good_max'][:] = np.array(time_good_max.value, 'M8[ns]')
    nc_root.close()
print("OK>")

if False:
    # ['db_path'], mode='r') as store
    #     for tbl in names_gen(cfg['in'], cfg_out):  # old: h5.names_gen?
    #         # Get data in ranges
    #         for df0, start_end in gen_loaded(tbl):



    # np.arange(2, scale_end)
    freqs = pywt.scale2frequency(wavelet, scales_rel) * fs  # ~(fs / scales_rel)