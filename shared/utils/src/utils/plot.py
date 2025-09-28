import numpy as np
import pandas as pd
import re
from typing import Mapping, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import LogLocator, AutoMinorLocator, MultipleLocator
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.axes_size import Fixed, Scaled
try:
    matplotlib.use(
        "Qt5Agg"
    )  # must be before importing plt (raises error after although documentation said no effect)
except ImportError:
    pass

lang = "Ru"


def vs_freq(
    psd: pd.DataFrame,
    ax: matplotlib.axes.Axes,
    legend_title: str,
    b_log_x: bool = True,
    path_dir: Optional[Path | str] = None,
    ylabel_prefix: str = "PSD(u,v)",
    colors=["red", "blue", "green", "black"],
    linestyles=None,
    save_name_parts=(
        "gray_subdir?",
        "ylabel_prefix",
        "@",
        "legend_titles",
        "log(x)?",
        ".png",
    ),
    no_labels_till_save=True,
    b_gray=False,
    lang=lang,
    plot_kwargs={}
):
    """
    Plot `psd` data columns vs index, treated as Power Spectral Density (PSD) and frequency.
    Customizable styling options:
    - multiple subplots with shared y-axis labels
    - For grayscale plots (b_gray=True), different line styles are used to distinguish between series
    - allows to save figure with organized filename based on plot characteristics

    psd: DataFrame containing PSD data with frequency as index and columns for different variables
    ax: The axes object to plot on
    legend_title: Title for the plot legend
    b_log_x: If True, x-axis will be in log scale (default: True)
    path_dir: optional directory path to save the plot. If None, plot won't be saved (default: None)
    ylabel_prefix: optional Prefix for y-axis label (default: "PSD(u,v)")
    colors : list or str, optional
        Colors for plot lines. If string, same color used for all lines (default: ["red", "blue", "green", "black"])
    linestyles : list or str, optional
        Line styles for plot. If None, defaults based on b_gray parameter (default: None)
    save_name_parts : tuple, optional
        Parts to construct save filename (default: ("gray_subdir?", "ylabel_prefix", "@", "legend_titles", "log(x)?", ".png"))
    no_labels_till_save : bool, optional
        If True, delays adding x-labels until saving (default: True)
    b_gray : bool, optional
        If True, uses grayscale palette (default: True)
    """
    n = len(psd.columns)
    if b_gray:
        if linestyles is None:
            # (0, (10, 5)) - Long Dashes comparing to "--"
            linestyles = ["-", (0, (10, 5)), "-.", ":"]
        if colors is None:
            colors = ["black"] * n
        elif isinstance(colors, str):
            colors = [colors] * n
        else:
            colors = [color / n for color in range(n)]  # 0 is black
    elif linestyles is None:
        linestyles = ["-"] * n
    elif isinstance(linestyles, str):
        linestyles = [linestyles] * n

    for i, (col, clr, l_style) in enumerate(zip(psd.columns, colors, linestyles)):
        clr_is_float_gray = isinstance(
            clr, float
        )  # replace lightness of gray with alpha
        ax.plot(
            psd[col].index,
            psd[col],  # , detrend='linear'
            label=col,
            color="black" if clr_is_float_gray else clr,
            linestyle=l_style,
            alpha=(1 - clr if clr_is_float_gray else 1)
            if b_gray
            else (0.5 if b_log_x else 0.9),
            **plot_kwargs,
        )
        # ax.psd(
        #     data1,
        #     FS=fs,
        #     NFFT=nfft,
        #     detrend="linear",
        #     noverlap=int(nfft / 2),
        #     sides="onesided",
        #     label=col,
        #     color=clr
        # )  # plots in DB on y. We plot in PSD units

    ax.set_yscale("log")
    ax.yaxis.set_minor_locator(
        LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10)
    )
    if b_log_x:
        ax.set_xscale("log")
        ax.xaxis.set_minor_locator(
            LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10)
        )
    else:
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.minorticks_on()
    # ax.tick_params(axis="both", which="minor", length=4, width=2, color="r")
    ax.grid(True, linestyle="--")
    # Length with increased length to fit long dashes used for gray palette
    ax.legend(title=legend_title, **{"handlelength": 3} if b_gray else {})
    ylabel = ", ".join([ylabel_prefix, "(м/с)²/Гц" if lang == "Ru" else "(m/s)²/Hz"])
    if not no_labels_till_save:
        ax.set_xlabel(", ".join(["f", "Гц" if lang == "Ru" else "Hz"]))
        ax.set_ylabel(ylabel)

    if path_dir:
        save_name = "".join(
            ("grayscale/" if b_gray else "")
            if part == "gray_subdir?"

            else re.sub(r"\|([^|]+)\|", r"\1abs", ylabel_prefix)
            if part == "ylabel_prefix"
            else ",".join(
                axis.get_legend().get_title().get_text() for axis in ax.figure.axes
            )
            if part == "legend_titles"
            else ("-log(x)" if b_log_x else "")
            if part == "log(x)?"
            else part
            for part in save_name_parts
        )
        fig = ax.figure

        # fig.tight_layout()  # Layout so plots do not overlap
        if no_labels_till_save:
            ax.set_xlabel(", ".join(["f", "Гц" if lang == "Ru" else "Hz"]))
            # Adjust the layout to ensure no spacing between axes
            fig.subplots_adjust(hspace=0.03)
            # Add a centered y-title on the left side of the figure
            min_y0, max_y1 = 1e9, -1e9
            for ax in fig.axes:
                ax_pos = ax.get_position()
                if ax_pos.y0 < min_y0:
                    min_y0 = ax_pos.y0
                if ax_pos.y1 > max_y1:
                    max_y1 = ax_pos.y1
                    ax_top = ax
                ax.spines["top"].set_visible(False)
                # ax.autoscale(enable=True, axis="both", tight=True)
            # Calculate the vertical bounds of all subplots
            center_y = (min_y0 + max_y1) / 2
            # Determine x position (left of the leftmost subplot)
            x_position = (
                ax.get_position().x0 - 0.05
            )  # Adjust this value to position the label
            fig.text(
                x_position,
                center_y,
                ylabel,
                rotation=90,
                va="center",
                ha="center",
                fontsize=12,
            )
            # fig.text(0, 0.5, ylabel, rotation=90, ha='right', va='center', size=12)
            ax_top.spines["top"].set_visible(True)
        try:
            fig.savefig(
                path_dir / save_name,
                dpi=300,
                bbox_inches="tight",
            )
            print(save_name, "saved")
        except Exception as e:
            print(f'Can not save figure to "{path_dir / save_name}":', e)
        try:
            fig.canvas.manager.set_window_title(save_name)
        except Exception as e:
            print(f'Can not set window title to "{save_name}":', e)

    elif no_labels_till_save:
        ax.tick_params(axis="x", labelbottom=False)


def fig_size(
    ncols,
    nrows,
    subplot_width=7,
    aspect_ratio=1,
    wspace=0.1,
    hspace=0.1,
    cbar_space=0.1,
):
    """
    Calculate the figure size
    Each subplot has width/height = 1 (aspect ratio)
    Total width = ncols * (subplot_width + cbar_width) + (ncols-1) * wspace * subplot_width
    Total height = nrows * subplot_height + (nrows-1) * hspace * subplot_height
    :param wspace: Space between columns
    :param hspace: Space between rows
    :param cbar_space: Space for the colorbar relative to subplot width
    """
    subplot_height = subplot_width / aspect_ratio
    total_width = (
        ncols * (subplot_width + cbar_space * subplot_width)
        + (ncols - 1) * wspace * subplot_width
    )
    total_height = nrows * subplot_height + (nrows - 1) * hspace * subplot_height
    return total_width, total_height


def figure_height_for_same_axes_height(fig, n_axes_out):
    """
    Calculate new figure height preserving original axis heights,
    assuming that all axes in the new figure will have similar decorations
    (same font sizes, similar titles, etc.)

    Parameters:
    - fig: Original Matplotlib figure
    - n_axes_out: Number of rows in new figure

    Returns:
    - New figure height in inches
    """
    n_axes_in = len(fig.axes)
    if n_axes_in == 0:
        raise ValueError("Original figure contains no axes")

    # Get original figure parameters
    fig_height = fig.get_size_inches()[1]
    hspace = fig.subplotpars.hspace  # Fraction of subplot height

    # Calculate height ratio factor
    height_ratio = (n_axes_out + hspace * (n_axes_out - 1)) / (
        n_axes_in + hspace * (n_axes_in - 1)
    )

    return fig_height * height_ratio


def force_equal_ticks(ax, select_fun=min):
    # Force a draw to compute initial ticks
    ax.figure.canvas.draw()  # draw()

    # Get current ticks and calculate steps
    x_ticks, y_ticks = ax.get_xticks(), ax.get_yticks()

    x_step = np.abs(x_ticks[1] - x_ticks[0]) if len(x_ticks) > 1 else np.inf
    y_step = np.abs(y_ticks[1] - y_ticks[0]) if len(y_ticks) > 1 else np.inf
    min_step = select_fun(x_step, y_step)

    # Set both axes to use the minimal step
    ax.xaxis.set_major_locator(MultipleLocator(min_step))
    ax.yaxis.set_major_locator(MultipleLocator(min_step))

    # Optional: Adjust limits if necessary (not typically needed)
    # ax.relim()
    # ax.autoscale_view()

    # Redraw to apply changes
    ax.figure.canvas.draw_idle()


def create_equal_subplots(
    ncols,
    nrows_per_col,
    figsize=(8, 6),
    cbar_width=0.2,
    cbar_pad=0.05,
    margin=1,
    margin_left=2,
    constrained_layout=False,
):
    """
    Create a grid of subplots with specified layout constraints.

    Parameters:
    - ncols: Number of columns.
    - nrows_per_col: List of integers specifying rows per column.
    - figsize: Optional figure size (width, height) in inches.
    - cbar_width: Width of each colorbar in inches.
    - cbar_pad: Padding between subplot and colorbar in inches.
    - margin: Fixed margin around the figure in inches.

    Returns:
    - fig: Matplotlib figure instance.
    - axes: Nested list of axes (columns then rows).
    """
    fig = plt.figure(figsize=figsize, constrained_layout=constrained_layout)

    # Convert margin from inches to figure fractions
    fig_width_in, fig_height_in = fig.get_size_inches()
    left = (margin_left or margin) / fig_width_in
    right = 1 - (margin / fig_width_in)
    bottom = margin / fig_height_in
    top = 1 - (margin / fig_height_in)

    # Use Fixed size class for inches-based dimensions
    cbar_size = Fixed(cbar_width)  # Width in inches
    cbar_padding = Fixed(cbar_pad)  # Padding in inches

    # Create subfigures for each column with equal widths
    subfigs = fig.subfigures(1, ncols, wspace=0, width_ratios=[1] * ncols)

    axes = []
    for col_idx, (subfig, nrows) in enumerate(zip(subfigs, nrows_per_col)):
        col_axes = []
        # Create gridspec inside subfigure for rows
        gs = subfig.add_gridspec(nrows, 1, hspace=0)

        for row_idx in range(nrows):
            ax = subfig.add_subplot(gs[row_idx])
            ax.set_aspect("equal")
            ax.set_adjustable("datalim")

            # Add colorbar with fixed size and padding
            divider = make_axes_locatable(ax)
            # cax = divider.append_axes("right", size=cbar_width, pad=cbar_pad)  # bar height as axis
            cax = divider.append_axes("right", size=cbar_size, pad=cbar_padding)
            col_axes.append((ax, cax))

        axes.append(col_axes)

    # Adjust figure margins
    if not constrained_layout:
        fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
    return fig, axes


def create_uniform_subplots(
    ncols,
    nrows,
    figsize=None,
    cbar_width=0.2,
    cbar_pad=0.005,
    margin=0.1,
    margin_left=0.5,
    margin_right=0.5,
    label_pad=0.1,
    hspace=0,
    wspace=0,
    constrained_layout=False,
):
    """
    Create subplots with strict layout control and equal aspect ratios.

    Parameters:
    - ncols: Number of columns (all with same number of rows)
    - nrows: Number of rows per column
    - figsize: Figure size in inches (width, height)
    - cbar_width: Colorbar width in inches
    - cbar_pad: Padding between plot and colorbar in inches
    - margin: Figure margin in inches
    - label_pad: Space reserved for axes labels in inches

    Returns:
    - fig: Figure object
    - axes: Nested list of axes (columns then rows).
    rows are grops of main axes + its colorbar axes
    """

    if figsize is None:
        figsize = (8, 6)

    fig = plt.figure(figsize=figsize, constrained_layout=constrained_layout)

    # Convert measurements to figure fractions
    fig_w, fig_h = fig.get_size_inches()
    margin_left = (margin_left or margin) / fig_w
    margin_right = ((margin_right or margin) + cbar_width + cbar_pad) / fig_w
    margin_bottom = (margin + label_pad) / fig_h
    margin_top = (margin + label_pad) / fig_h

    # Create main gridspec
    gs = fig.add_gridspec(
        nrows,
        ncols,
        left=margin_left,
        right=1 - margin_right,
        bottom=margin_bottom,
        top=1 - margin_top,
        hspace=hspace,
        wspace=wspace,
    )

    # Use Fixed size class for inches-based dimensions
    cbar_size = Fixed(cbar_width)  # Width in inches
    cbar_padding = Fixed(cbar_pad)  # Padding in inches

    axes = []
    for col in range(ncols):
        col_axes = []
        for row in range(nrows):
            if row == 0:  # Create main axis
                # First row, create the subplot without sharing x-axis
                ax = fig.add_subplot(gs[row, col])
            else:
                # Share the x-axis with the first row
                ax = fig.add_subplot(gs[row, col], sharex=ax)

            ax.set_aspect("equal", adjustable="datalim")
            # Create colorbar axis
            divider = make_axes_locatable(ax)
            # cax = divider.append_axes(
            #     "right", size=f"{cbar_width}in", pad=f"{cbar_pad}in"
            # )
            cax = divider.append_axes("right", size=cbar_size, pad=cbar_padding)
            ax.set_anchor("W")  # Align plots to left (for colorbar space)
            col_axes.append((ax, cax))

        axes.append(col_axes)

    # # Equalize column widths and row heights
    # for col_axes in axes:
    #     for ax, cax in col_axes:
    #         ax.set_anchor("W")  # Align plots to left (for colorbar space)

    return fig, axes


def number_to_size(x, x_min=100, x_max=100000, y_min=1, y_max=10):
    """
    Monotonic function that transforms x value in range from 100 to 100000
    to y value from 10 to 1. It changes faster at small x and always > 1 for x > 1.
    """
    # Apply a logarithmic transformation to x to make it change faster at small x
    log_x = np.log(x)

    # Apply a linear transformation to log_x to scale it to the range [0, 1]
    scaled_log_x = (log_x - np.log(x_min)) / (np.log(x_max) - np.log(x_min))

    # Apply a linear transformation to scaled_log_x to scale it to the range [y_max, y_min]
    y = y_max - scaled_log_x * (y_max - y_min)

    # Ensure y is always > 1 for x > 1 and never > 15
    y = np.clip(y, 1, 15)

    return y


def regression(
    df,
    col_prm_y,
    axes,
    str_unit,
    predict_fun,
    col_clr="days",
    lang=lang,
    clr_units=None
):
    """
    Creates regression plots comparing two parameters with time-based coloring.
    This function generates scatter plots with regression lines, comparing a dependent
    variable against independent variables while color-coding points based on temporal
    information.

    df : Input data with columns to plot
    col_prm_y : str
        Column name for y-axis data
    axes : list of tuple
        List of (main_axis, colorbar_axis) tuples
    str_unit : str
        Units label for axes
    predict_fun : callable
        Function to generate predicted values
    col_clr : str, optional
        Column name for color coding (default: 'days')
    lang : str, optional
        Language for labels (default: global lang, 'En' for English, other for Russian).
    clr_units : str or datetime, optional
        Units for colorbar. If col_clr is 'days', this is start date
    Notes
    -----
    - For directional data (columns starting with 'Vdir'), uses angular regression.
    - For other data, uses Ordinary Least Squares (OLS) regression.
    - Creates scatter plots with regression lines and adds colorbar showing temporal progression.
    - Points are colored based on the 'days' column using viridis colormap.
    - Plots maintain equal aspect ratio and include grid lines.
    Note: Function modifies the provided axes in-place.
    """
    cols_prm_x = [c for c in df.columns if c not in (col_prm_y, col_clr)]
    for i_ax_row, (col_prm_x, (ax, axc)) in enumerate(zip(cols_prm_x, axes)):
        b_ok = ~df[[col_prm_x, col_prm_y]].isna().any(axis=1)
        isorted_x = df.index[b_ok][np.argsort(df.loc[b_ok, col_prm_x])]
        y_predicted = predict_fun(df.loc[isorted_x, [col_prm_x, col_prm_y]])

        scatter = ax.scatter(
            df[col_prm_x].values,
            df[col_prm_y].values,
            c=df[col_clr],
            cmap="viridis",
            s=number_to_size(df.shape[0]),
            alpha=0.5,
        )

        ax.plot(df.loc[isorted_x, col_prm_x].values, y_predicted, color="r")
        ax.autoscale(enable=True, tight=True)
        ax.set_aspect("equal", "box")
        ax.set_xlabel(", ".join([col_prm_x, str_unit]))
        ax.set_ylabel(", ".join([col_prm_y, str_unit]))
        ax.grid(True, linestyle="--")
        # Add a colorbar
        cbar = ax.figure.colorbar(scatter, cax=axc, orientation="vertical")
        if clr_units:
            if col_clr == "days":
                clr_units = (
                    f"Days from {clr_units:%d.%m.%Y}"
                    if lang == "En"
                    else f"Дней от {clr_units:%d.%m.%Y}"
                )
                cbar.set_label(clr_units)
            else:
                cbar.set_label(", ".join([col_clr, clr_units]))