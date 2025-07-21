# %%
# import json
from datetime import datetime, timedelta
from itertools import dropwhile, groupby
import numpy as np
import pandas as pd
import scipy.signal as sp
from pathlib import Path
from typing import Mapping, Optional
import re

import statsmodels.api as sm  # .stats.api as sms
from statsmodels.tsa import stattools as smt

# Plotting libraries and settings
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import LogLocator, AutoMinorLocator, MultipleLocator
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.axes_size import Fixed, Scaled

# Custom functions
from scripts import stat_wind
import plot
import func_vsz as fv
from scripts.stat_adv_tcm import fit_angular_regression, fit_regression

# Matplotlib settings
try:
    matplotlib.use(
        "Qt5Agg"
    )  # must be before importing plt (raises error after although documentation said no effect)
except ImportError:
    pass
matplotlib.rcParams["axes.linewidth"] = 1.5
matplotlib.rcParams["figure.figsize"] = (16, 7)
matplotlib.rcParams["font.size"] = 12
matplotlib.rcParams["axes.xmargin"] = 0.001  # (default: 0.05)
matplotlib.rcParams["axes.ymargin"] = 0.01
matplotlib.interactive(True)
plt.ion()

# %%
# Data paths
# dir with raw data - to save our results above
path_raw_adv = Path(
    r"B:\WorkData\Cruises(older)\_KaraSea\220906_AMK89-1\ADV_Sontek\_raw,to_txt,vsz"
)
# previously prepared data - to load
path_db = Path(
    r"B:\WorkData\Cruises(older)\_KaraSea\220906_AMK89-1\ADV_Sontek\st7440A,7441A@ADV+TCM.h5"
)

adv_str = "ADV"
tcm_str = "TCM"
str_abs, str_dir = "|V|", "Vdir"
n_params = 2  # number of (above) parameters to plot
lang = "Ru"

# %%
# Load from DB
print(f"Loading from {path_db}...", end="")
dfs = {}  # stations dict with values of all devices data merged into one dataframe
with pd.HDFStore(path_db, mode="r") as store:
    nodes = sorted(store.root.__members__)  # , key=number_key
    print(nodes)
    st_prefix = "st"
    stations = [node[len(st_prefix):] for node in nodes if node.startswith(st_prefix)]
    # dfs that is needed for regression plots
    for st in stations:
        dfs[st] = store[f"st{st}"].rename(
            lambda col: col.replace("Vabs", "|V|"), axis="columns"
        ).filter(regex="[^_]+_(ADV4|TCM5)")
dt = np.diff(dfs[stations[0]].index[:2]).astype("m8[s]").astype(int).item()
print(f"Loaded {stations} data with period: {dt}s")


# %%
print("Plotting")
b_plot_source = True  # False
if b_plot_source:
    ## 3.1 Plot source on stations
    for st, df in dfs.items():
        # # Convert dates to days for color points and label colorbar
        # days_start = df.index[0].floor("d")
        # df["days"] = (df.index - days_start).total_seconds() / 86400

        cols_prm = [c for c in df.columns if c.startswith(str_abs)]
        len_str_prm_prefix = len(str_abs) + 1
        devs = [
            c[len_str_prm_prefix:] for c in cols_prm
            # if c[len_str_prm_prefix:].startswith(("ADV4", "TCM5"))
        ]
        devs.sort(
            key=lambda c: (
                lambda b_tcm: (not b_tcm) * 10000  # > max depth to sort TCM first
                + int(c[len(tcm_str if tcm_str else adv_str) : -1])  # sort depth
            )(c.startswith(tcm_str))
        )
        print(st, devs)
        n_rows = len(devs) * n_params  #  = df.columns.size if no new cols

        fig, axes = plt.subplots(nrows=n_rows, sharex=True)
        for i_dev, dev in enumerate(devs):
            for i_prm, (str_prm, str_unit) in enumerate(
                [(str_abs, "{m}/{s}".format_map(fv.I)), (str_dir, "°")]
            ):
                i_plot = i_prm * len(devs) + i_dev  # i_dev * n_params + i_prm
                ax = axes[i_plot]
                ax.plot(
                    df.index,
                    df[f"{str_prm}_{dev}"],
                    label=dev,
                    color="red" if str_prm == str_abs else "blue",
                )
                if i_plot == n_rows - 1:
                    ax.set_xlabel("Time")
                ax.set_ylabel(", ".join([str_prm, str_unit]))
                ax.grid(True, linestyle="--")
                ax.legend(title=st if i_plot == 0 else None, loc="upper right")

                # cols_prm = [c for c in df.columns if c.startswith(str_prm)]
                # len_str_prm_prefix = len(str_prm) + 1
                # devs = [c[len_str_prm_prefix:] for c in cols_prm]

                # icol_x = devs.index([dev for dev in devs if dev.startswith(tcm_str)][0])
                # # ADV devices
                # col_prm_y = cols_prm[icol_x]

    plt.show()
    print("source data plotted")

# %%
# Statistics for all devices
b_save_stat = False
for st in stations:
    print(f"st{st}:")
    df_stats, df_stats_corr = stat_wind.get_stat(dfs[st], v="V")
    print(df_stats_corr)
    if b_save_stat:
        stat_wind.save_stat(
            df_stats,
            df_stats_corr,
            path_base=path_raw_adv.with_name(f"st{st}_avg-dec={int(dt)}s"),
            v="V",
        )
    print("-"*30)

# %%
## 3.3 Plot regressions
dt = np.diff(next(iter(dfs.values())).index[:2].to_numpy("M8[s]")).astype(int).item()
print(f"Plot regressions for data is decimated to dt={int(dt)}s")
# Plot regressions
ncols = 2  # = number of parameters
wspace, hspace, cbar_space, cbar_height = 0.1, 0.1, 0.03, 0.8

# %%

# Figure per station
for st, df in dfs.items():
    # # Convert dates to days for color points and label colorbar
    # days_start = df.index[0].floor("d")
    # df["days"] = (df.index - days_start).total_seconds() / 86400
    cols_abs = [c for c in df.columns if c.startswith(str_abs)]
    len_str_prm_prefix = len(str_abs) + 1
    devs = [c[len_str_prm_prefix:] for c in cols_abs]
    nrows = len(devs) - 1  # devices for y axes (other one is on x)
    fig, axes = plot.create_uniform_subplots(ncols, nrows, figsize=(12, 6 * nrows), constrained_layout=True)
    axs, cbars = [], []

    # Axes column per parameter
    for i_prm, (str_prm, str_unit) in enumerate(
        [(str_abs, "m/s"), (str_dir, "°")]
        ):
        # Debug: i_prm, str_prm = 0, str_abs
        # st, df = next(iter(dfs.items()))

        cols_prm = [c for c in df.columns if c.startswith(str_prm)] if str_prm != str_abs else cols_abs
        print(st, devs)
        icol_y = devs.index([dev for dev in devs if dev.startswith(tcm_str)][0])
        col_prm_y = cols_prm[icol_y]

        # color by |V| measured by TCM
        col_clr = col_prm_y if str_prm == "|V|" else col_prm_y.replace("Vdir", "|V|")

        print(f"{tcm_str}:", col_prm_y, "colored by", col_clr)
        plot.regression(
            df[cols_prm if col_clr in cols_prm else cols_prm + [col_clr]],
            col_prm_y=col_prm_y,
            predict_fun=(fit_angular_regression if col_prm_y.startswith("Vdir") else fit_regression),
            axes=axes[i_prm],
            col_clr=col_clr,
            str_unit=str_unit,
            clr_units="{m}/{s}".format_map(fv.I),
            lang=lang,
        )

    # Automatically adjust data limits to fill axes
    for ax_col in axes:
        min_x0, max_x1 = 1e9, -1e9
        for ax, axc in ax_col:
            ax_pos = ax.get_position()
            if ax_pos.x0 < min_x0:
                min_x0 = ax_pos.x0
            if ax_pos.x1 > max_x1:
                max_x1 = ax_pos.x1

        for ax, axc in ax_col:
            ax.autoscale(enable=True, tight=True)
            ax.apply_aspect()
            plot.force_equal_ticks(ax, select_fun=min)

    try:  # Stop here to manually extend width of figure if graphs not fit!
        file_fig = path_raw_adv.with_name(f"Reg_{st}@{','.join(devs)}_avg-dec={int(dt)}s.png")
        fig.savefig(
            file_fig,
            dpi=300,
            bbox_inches="tight",
        )
        print(file_fig.name, "saved")
    except Exception as e:
        print(f"Can not save fig to {file_fig}: {e}")

# %%
# Проверка на стационарность

# Data per station
for st, df in dfs.items():
    # # Convert dates to days for color points and label colorbar
    # days_start = df.index[0].floor("d")
    # df["days"] = (df.index - days_start).total_seconds() / 86400
    cols_abs = [c for c in df.columns if c.startswith(str_abs)]
    len_str_prm_prefix = len(str_abs) + 1
    devs = [c[len_str_prm_prefix:] for c in cols_abs]
    # Axes column per parameter
    for i_prm, (str_prm, str_unit) in enumerate([(str_abs, "m/s"), (str_dir, "°")]):
        # Debug: i_prm, str_prm = 0, str_abs
        # st, df = next(iter(dfs.items()))
        cols_prm = (
            [c for c in df.columns if c.startswith(str_prm)]
            if str_prm != str_abs else cols_abs
        )
        print("\n", st, devs)
        for col in cols_prm:
            for test_fun_name, test_fun in [('ADF', smt.adfuller), ("KPSS", smt.kpss)]:
                is_adf = test_fun_name=="ADF"
                result = test_fun(df[col][df[col].notna()], regression="c")
                print(col, f'{test_fun_name} test:')
                print(f'{test_fun_name} Statistic: {result[0]}')
                print(f'p-value: {result[1]}', "- not stationary!" if (result[1] < 0.05) ^ is_adf else "")
                print('Critical Values:')
                for key, value in result[4 if is_adf else 3].items():
                    print(f'   {key}: {value}')
# %%
