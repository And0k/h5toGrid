# %%

import netCDF4
import numpy as np
from pathlib import Path, PurePath
import pandas as pd
import re
from collections import defaultdict
from datetime import datetime, timedelta
from itertools import tee
import difflib
from scipy.interpolate import LinearNDInterpolator, CloughTocher2DInterpolator  # , RectBivariateSpline
from sklearn.preprocessing import MinMaxScaler
from typing import Any, Callable, Dict, Optional, Sequence, Mapping, Tuple, Union
from operator import sub
from utils.logging_config import setup_logging
from veusz_helpers.common import func_vsz as fv

logger = setup_logging(__name__, console_format_args={"name": False, "datefmt": "%H:%M:%S"})

def load_t_chain_text_output(path: PurePath):
    """
    :param path: path to output file or path to .7z archive with path inside the archive (with archive as dir)
    :raises FileNotFoundError:
    :return: (df, file_path): loaded Dataframe and path to file (equal to the path if path is not an arhive
        path)
    """
    try:
        i, arc_path = next((i, x) for i, x in enumerate(path.parents) if x.suffix == ".7z")
    except StopIteration:
        file_path = path
    else:
        import py7zr
        inner_file = Path(*path.parts[-i - 1 :])

        temp_dir = arc_path.parent / "~tmp"
        file_path = temp_dir / inner_file
        if not file_path.is_file():
            with py7zr.SevenZipFile(arc_path, mode="r") as zip:
                # Extract only the specific file to the temporary directory
                zip.extract(path=temp_dir, targets=[inner_file.as_posix()], recursive=True)
            if not file_path.exists():
                raise FileNotFoundError(f"File {inner_file} not found in archive")

    df = pd.read_csv(
        file_path,
        sep=",",
        header=0,
        index_col=0,
        date_format="ISO8601",
        dtype=np.float32,
        skipinitialspace=True,
    )
    return df, file_path


def create_extrap_func(x, y) -> Callable[[Sequence[float]], np.ndarray]:
    """Create extrapolation function"""

    # slopes for extrapolation
    left_slope = (y[1] - y[0]) / (x[1] - x[0])
    right_slope = (y[-1] - y[-2]) / (x[-1] - x[-2])

    def f(xi: np.ndarray):
        xi = np.asarray(xi)
        out = np.empty_like(xi, float)
        left = xi < x[0]
        right = xi > x[-1]
        inside = ~(left | right)
        out[inside] = np.interp(xi[inside], x, y)
        out[left] = y[0] + left_slope * (xi[left] - x[0])
        out[right] = y[-1] + right_slope * (xi[right] - x[-1])
        return out

    return f

# %%  Functions to save to NetCDF

def create_dar(
    data=None,
    coords=None,
    interp_dt: Union[None, np.timedelta64, timedelta] = None,
    bin_dt=None,
    bin_dz=None,
    attrs={},
):
    """
    Helper function to create NetCDF DataArray
    :param data: if None then will be zeros for coords, which must be defined
    :param coords: keys must be in order of data dimensions. If None then coordinates will be data integer indexes of each dimension
    :param interp_dt:
    :return: DataArray
    """
    import xarray as xr

    if data is None:
        data = np.zeros([len(v) for v in coords.values()])
    else:
        if coords is None:
            coords = {c: np.arange(0, n) for c, n in zip("xyz", data.shape)}
    # try:
    #     # Add dimensions for single lat/lon
    #     if (
    #         not coords["lat"].data.shape
    #         and not coords["lon"].data.shape
    #         and data.ndim == len(coords) - 2
    #     ):
    #         # Calculate new shape: (original[0], original[1], 1, 1, rest...)
    #         new_shape = data.shape[:2] + (1, 1) + data.shape[2:]
    #         data = data.reshape(new_shape)
    # except KeyError:
    #     pass

    try:
        name = attrs.pop("name")
    except KeyError:
        name = None
    dar = xr.DataArray(
        data,
        dims=[k for k, v in coords.items() if (not isinstance(v, xr.DataArray)) or v.data.shape],
        coords=coords,
        name=name,
        attrs=attrs,
    )

    if interp_dt:
        if isinstance(interp_dt, np.timedelta64):  # to timedelta:
            interp_dt = interp_dt.astype("m8[s]").item()
        dar = dar.resample(time=interp_dt).interpolate("linear")
    if bin_dt or bin_dz:
        if bin_dt:
            if isinstance(bin_dt, np.timedelta64):  # to timedelta:
                bin_dt = bin_dt.astype("m8[s]").item()
            dar_out = dar.resample(time=bin_dt).mean()
            # dim=['z', 'time'] .groupby_bins(bins, group='time', precision=15)
        else:
            dar_out = dar
        if bin_dz:
            dz = np.diff(coords["z"][:2]).item()
            dz_bin = dz * (bin_dz + 1)
            n_bins = len(coords["z"]) // bin_dz
            bins = coords["z"][0] + np.cumsum([dz_bin] * n_bins) - dz_bin - dz / 2
            # bins = coords['z'][::n_bins]
            # bins = len(coords['z'])//bin_dz
            # np.arange(coords['z'][0], coords['z'][-1] + bin_dz/2, bin_dz)
            dar_out = dar_out.groupby_bins(bins=bins, group="z", restore_coord_dims=True).mean()
            coord_z_new = np.float64([bin.mid for bin in dar_out["z_bins"].values])
            # transpose back (why above have transposed data?)  # restore_coord_dims not works?
            if dar_out.coords.keys() != dar.coords.keys():
                # dar_out = dar_out.transpose()
                dar_out = xr.DataArray(
                    dar_out.data,
                    dims=coords.keys(),
                    coords={key: coord_z_new if key == "z" else dar_out.coords[key] for key in coords.keys()},
                    name=name, attrs=attrs
                )
    else:
        dar_out = dar

    # Assigning encoding controls on-disk format:
    dar_out["time"].encoding.update({
        "units": "days since 1899-12-30 00:00:00",  # Excel's days, also used in Surfer
        "calendar": "proleptic_gregorian",
        "dtype": "float64",
    })
    return dar_out


def save_nc_for_surfer(
    time: np.ndarray,
    y: np.ndarray,
    out: Mapping[str, np.ndarray],
    path_base: Path,
    dt: Union[None, list, np.timedelta64],
    dy=None,
    not_interp_keys = {},  # todo: auto detect
    stem_sfx="",
    lat=None,
    lon=None,
    attrs={}
):
    """
    Saves each `out` 2D item to separate NetCDF file

    :param time:
    :param y:
    :param out:
    :param path_base:
    :param dt: if many then average and save file for each dt and dy
    :param dz: -//-
    """
    import xarray as xr
    # from veusz_helpers.common import func_vsz as fv

    b_have_v_dekart = "u" in out

    try:
        from dask import da
        da_np = da if isinstance(out["u"] if b_have_v_dekart else next(iter(out.values())), da.Array) else np
    except ImportError:
        da_np = np

    if "Vabs" in out and not b_have_v_dekart:
        from tcm.incl_h5clc_hy import polar2dekart

        out["v"], out["u"] = polar2dekart(out["Vabs"], out["Vdir"])
        del out["Vabs"], out["Vdir"]
        out["Vabs"] = out["Vdir"] = None  # to the end of ``out``


    # Time for Surfer
    # (will be calculated before saving of 1st 2d dataset and then assigned to all next datasets)
    dt = dt if isinstance(dt, list) else [dt]
    dy = dy if isinstance(dy, list) else [dy] * len(dt)
    for i_used_dt, (dt, dy) in enumerate(zip(dt, dy)):
        ds_saved = {}
        # time_coord_converted = (
        #     None  # we set value after creating of ds because dt and time should have same units
        # )

        str_dt = fv.str_dt(
            dt.astype("m8[s]") if isinstance(dt, np.timedelta64) else np.timedelta64(dt, "s"),
            lang="en"
        )

        def get_file_name(param):
            return (
            f"{path_base.name}_{param}_{f'dt={str_dt}' if str_dt else ''}{f',dz={dy}' if dy else ''}"
            f"{stem_sfx}.nc"
        )

        coords_lat_lon = (
            {
                "latitude": xr.DataArray(lat, attrs={"standard_name": "latitude", "units": "degrees_north"}),
                "longitude": xr.DataArray(lon, attrs={"standard_name": "longitude", "units": "degrees_east"}),
            }
            if lat is not None
            else {}
        )
        coords = {
            "z": y,
            "time": ("time", time, {"standard_name": "time"}),
            **coords_lat_lon,
        }
        for name, data in out.items():
            if name == "Vabs":
                data = da_np.sqrt(ds_saved["u"] ** 2 + ds_saved["v"] ** 2)
                dt_cur = dz_cur = None  # do not interp/bin 2nd time
                coords = ds_saved["u"].coords  # use interp/binned coord
            elif name == "Vdir":  # must run after 'Vabs' in this cycle
                data = da_np.degrees(da_np.arctan2(ds_saved["u"], ds_saved["v"])) % 360
                # leave dt_cur and coords same as for Vabs
            elif name in not_interp_keys:
                dt_cur = dz_cur = None
            else:
                dt_cur = dt
                dz_cur = dy
            if len(data.shape) != 2:
                continue
            print(name, end=", ")  # **{("bin_dt" if i_used_dt > 0 else "interp_dt"): dt_cur}?
            dar = create_dar(data, coords=coords, interp_dt=dt_cur, bin_dz=dz_cur, attrs=attrs.get(name, {}))
            # .chunk({"x": 100, "y": 100, "time": -1})

            # Save Vabs and Vdir only after interp/bining of u and v in create_xrds(). Skip saving u and v
            if name in ("u", "v"):
                ds_saved[name] = dar
                continue
            # Not need manually convert:
            # if time_coord_converted is None:
            #     # to Excel time
            #     time_coord_converted = (dar.coords["time"] - np.datetime64("1899-12-30T00:00:00")).astype(
            #         "M8[ns]"
            #     ).values.astype("f8") / (24 * 3600e9)
            # dar["time"] = time_coord_converted  # changes dar.coords['time']
            xr.Dataset({name: dar}).to_netcdf(
                path_base.parent / get_file_name(name),
                mode="w",
                format="NETCDF4_CLASSIC",
            )

        logger.info(
            f"Exported 2D datasets to {get_file_name('{param}')} "
            f"files as NetCDF grids for param = {list(out.keys())}"
        )

# %% to del
def save_to_nc_del():
    track2cube.convert_cubes_depth2height(cubes)
    col_axis_for_surfer = "height"
    col_axis_vals_for_surfer = -np.array(range(0, 100, 10))  # need to be regular for Surfer
    for cube in cubes:
        if cube.ndim == 2:

            # Regrid
            good_cols_range = np.flatnonzero(~np.isnan(cube.data.filled(np.nan)).all(axis=0))[[0,-1]]
            good_depth_range = cube.coord(col_axis_for_surfer).points[good_cols_range]
            if gt(*good_depth_range):  # reverse ranges
                good_depth_range = good_depth_range[::-1]
            col_vals_use = np.array(col_axis_vals_for_surfer)
            col_vals_use = col_vals_use[
                (good_depth_range[0] - 2 <= col_vals_use) & (col_vals_use <= good_depth_range[-1] + 2)
            ]
            cube = cube.interpolate([(col_axis_for_surfer, col_vals_use)], iris.analysis.Linear())

            # remove bounds to for Surfer (not needed)
            cube.coord("time").bounds = None
            iris.util.squeeze(cube)  # remove size 1 coords
            if len(cube.coords(dim_coords=True)) < 2:
                coord = cube.coords(new_coords[0], dim_coords=False)[0]  #, contains_dimension=new_coords[0]
                iris.util.promote_aux_coord_to_dim_coord(cube, coord)
                cube.transpose([1, 0])
                #, contains_dimension=coord_dim[0])
            else:
                pass # already 2D

            track2cube.nc_save(
                [cube], file.parent, file_stem=file.stem, b_separately=True, print_before="* files: "
            )


def plot_2d_my(X, Y, Z, time_range, y_overlay=None, title="", lbl_x="", lbl_y="", lbl_z="", lbl_overlay=""):
    """
    Create contour plot with O2ppm contours and y_st overlay

    :param X:
    :param Y:
    :param Z:
    :param time_range:
    :param y_overlay:
    :return: axis
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm
    fig, ax = plt.subplots(figsize=(16, 4))

    x = X[:, 0]  # norm_time(df.index.astype(int).values)

    # Define contour levels
    contour_levels = list(range(11))

    # Create contour plot
    contour = ax.contour(X, Y, Z, levels=contour_levels, colors="blue", linewidths=0.8)
    ax.clabel(contour, inline=True, fontsize=8, fmt="%1.0f")

    # Fill contours with a light color for better visibility
    contourf = ax.contourf(X, Y, Z, levels=contour_levels, cmap="Blues", alpha=0.3)

    # Overlay y_overlay (starting depths at min temperature)
    if y_overlay is not None:
        ax.plot(x, y_overlay, "r-", linewidth=0.5, label=lbl_overlay)

    # Add colorbar
    cbar = plt.colorbar(contourf, ax=ax, pad=0.001)  # pad (in axis width units) - to move it closer to axis
    cbar.set_label(lbl_z, rotation=270, labelpad=2)

    # Set labels and title
    ax.set_xlabel(lbl_x)
    ax.set_ylabel(lbl_y)
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.invert_yaxis()  # show depth increasing downward

    # Set x-axis to show non-normalized time values
    ax.set_xlim(*x[[0, -1]])

    # For displaying actual time values at the bottom, we can use the original time values
    # `time_ticks = np.linspace(*time_range[[0, -1]].to_list(), num=10)` code replacement to support datetime
    num = 10
    start, end = time_range[[0, -1]].to_numpy("M8[s]")
    step = (end - start) // (num - 1)
    time_ticks = np.arange(start, end + step / 2, step)

    time_labels = [datetime(time_ticks).strftime("%m-%d") for t in time_ticks]
    ax.set_xticks(np.linspace(*x[[0, -1]], num=10))
    ax.set_xticklabels(time_labels)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return ax


def plot_2d(
    X,
    Y,
    Z,
    time_range,
    y_lims=None,
    y_overlay: Optional[Sequence[np.ndarray] | np.ndarray] = None,
    title="",
    lbl_x="",
    lbl_y="",
    lbl_z="",
    lbl_overlay: Optional[Sequence[str] | str] = "",
    dt_tick=None,
    contour_levels=None,
    cmap="rainbow",
    b_show=True,
    b_diverging=False



):
    """
    Create contour plot with O2ppm contours and y_st overlay

    :param X: 2D array of normalized x-coordinates (normalized time values, typically 0-1)
    :param Y: 2D array of y-coordinates (depth values)
    :param Z: 2D array of z-coordinates (contour values)
    :param time_range: pandas DatetimeIndex with [min, max] datetime values corresponding to X extent
    :param y_lims:
    :param y_overlay: optional 1D array for overlay line plot or list of them
    :param title: plot title string
    :param lbl_x: x-axis label string
    :param lbl_y: y-axis label string
    :param lbl_z: z-axis (colorbar) label string
    :param lbl_overlay: label for overlay line / or list of labels for each of line
    :param dt_tick: timedelta, string ('D', 'h', 'min'), or None for automatic tick spacing
    :param b_show:
    :return: matplotlib axis object
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.ticker import MaxNLocator
    from matplotlib import contour
    from matplotlib.colors import TwoSlopeNorm  # DivergingNorm

    fig, ax = plt.subplots(figsize=(16, 4))

    x = X[:, 0]  # normalized time values

    # # Define contour levels
    # contour_levels = list(range(int(np.nanmin(Z)), int(np.nanmax(Z))))

    # Более плотную сетку уровней
    # Используем np.linspace между каждой парой соседних уровней contour_levels
    fine_levels = []
    n_div = 10  # Создаем n_div новых уровней между contour_levels[i] и contour_levels[i+1]
    if contour_levels is None:
        contour_levels = MaxNLocator(nbins=7).tick_values(Z.min(), Z.max())
    contour_levels_lims = contour_levels[:: len(contour_levels) - 1]

    for i in range(len(contour_levels) - 1):
        # Всего точек: n_div (между) + 1 (начальная), но endpoint=False, чтобы не дублировать
        sub_levels = np.linspace(contour_levels[i], contour_levels[i+1], num=n_div, endpoint=False)
        fine_levels.extend(sub_levels)
    fine_levels.append(contour_levels[-1])  # не теряем последний уровень

    # # Fill contours with a light color for better visibility
    # contourf = ax.contourf(
    #     X,
    #     Y,
    #     Z,
    #     cmap=cmap,
    #     alpha=0.5,
    #     # в 2 раза реже чем fine_levels, но не совпадает с main contours
    #     levels=fine_levels,
    #     linewidths=0,
    # )

    # Fill contours with a light color for better visibility
    v_lims_dict = {} if contour_levels is None else dict(zip(("vmin", "vmax"), contour_levels_lims))
    contourf = ax.imshow(  # contourf
        Z.T,
        extent=[*x[[0, -1]], *Y[0, [0, -1]]],
        **({} if b_diverging else v_lims_dict),
        cmap=cmap,
        alpha=0.5,
        origin="lower",  # если координаты увеличиваются вверх (обычно так у X, Y)
        aspect="auto",  # чтобы не искажать пропорции
        **({"norm": TwoSlopeNorm(**v_lims_dict, vcenter=0)} if b_diverging else {}),
    )
    # Add minor contours
    contour = ax.contour(
        X,
        Y,
        Z,
        colors="blue" if lbl_z.lower().startswith("o") else "black",
        linewidths=0.1,
        # в 2 раза реже чем fine_levels, но не совпадает с main contours
        levels=np.float64(fine_levels[::2])[(np.arange(len(fine_levels)) % n_div)[::2] > 0],
    )
    # Add major contours
    all_integers = (
        False if contour_levels is None else np.all(np.equal(contour_levels, np.floor(contour_levels)))
    )
    contour = ax.contour(
        X,
        Y,
        Z,
        colors="blue" if lbl_z.lower().startswith("o") else "black",
        linewidths=0.5,
        levels=contour_levels,
    )
    ax.clabel(contour, inline=True, fontsize=8, fmt="%1.0f" if all_integers else "%1.1f")

    # Overlay `y_overlay`s (initially for starting depths at min temperature)
    if y_overlay is not None:
        clr = ["r", '#00FF00', 'm']
        for i_ovr, (lbl_ovr, y_ovr) in (
            [[0, (lbl_overlay, y_overlay)]]
            if isinstance(y_overlay, np.ndarray) or isinstance(lbl_overlay, str)
            else enumerate(zip(lbl_overlay, y_overlay))
        ):
            ax.plot(x, y_ovr, color=clr[i_ovr % len(clr)], linewidth=0.75, label=lbl_ovr)

    # Add colorbar
    cbar = plt.colorbar(contourf, ax=ax, pad=0.001)
    cbar.ax.set_title(lbl_z) #, pad=20
    # cbar.set_label(lbl_z, rotation=270, labelpad=5)

    # Set labels and title
    ax.set_xlabel(lbl_x)
    ax.set_ylabel(lbl_y)
    ax.set_title(title)
    ax.legend(loc="upper right")

    ax.set_ylim(*(y_lims if y_lims is not None else Y[0, [0, -1]]))
    ax.invert_yaxis()

    # Set x-axis limits to normalized values
    ax.set_xlim(*x[[0, -1]])

    # Configure time axis
    start_time, end_time = time_range[[0, -1]].astype("M8[s]")
    total_duration = end_time - start_time

    # Determine major tick locator and formatter
    if dt_tick is None:
        # Auto-determine based on duration
        duration_seconds = total_duration.item().total_seconds()

        if duration_seconds < 3600:  # < 1 hour
            major_locator = mdates.MinuteLocator(interval=10)
            minor_locator = mdates.MinuteLocator(interval=2)
            formatter = mdates.DateFormatter("%H:%M")
        elif duration_seconds < 86400:  # < 1 day
            major_locator = mdates.HourLocator(interval=2)
            minor_locator = mdates.HourLocator()
            formatter = mdates.DateFormatter("%H:%M")
        elif duration_seconds < 604800:  # < 1 week
            major_locator = mdates.DayLocator()
            minor_locator = mdates.HourLocator(byhour=[0, 6, 12, 18])
            formatter = mdates.DateFormatter("%m-%d")
        elif duration_seconds < 2592000:  # < 30 days
            major_locator = mdates.DayLocator(interval=3)
            minor_locator = mdates.DayLocator()
            formatter = mdates.DateFormatter("%m-%d")
        else:
            major_locator = mdates.WeekdayLocator()
            minor_locator = mdates.DayLocator()
            formatter = mdates.DateFormatter("%m-%d")
    else:
        # User-specified dt_tick
        if isinstance(dt_tick, str):
            unit_map = {
                "D": (mdates.DayLocator(), mdates.HourLocator(byhour=[0, 6, 12, 18])),
                "d": (mdates.DayLocator(), mdates.HourLocator(byhour=[0, 6, 12, 18])),
                "h": (mdates.HourLocator(), mdates.MinuteLocator(interval=15)),
                "H": (mdates.HourLocator(), mdates.MinuteLocator(interval=15)),
                "m": (mdates.MinuteLocator(), mdates.MinuteLocator(interval=1)),
                "min": (mdates.MinuteLocator(), mdates.MinuteLocator(interval=1)),
            }
            major_locator, minor_locator = unit_map.get(dt_tick, (mdates.DayLocator(), mdates.HourLocator()))
        elif isinstance(dt_tick, (timedelta, np.timedelta64)):
            if isinstance(dt_tick, np.timedelta64):
                dt_seconds = dt_tick / np.timedelta64(1, "s")
            else:
                dt_seconds = dt_tick.total_seconds()

            if dt_seconds < 3600:  # < 1 hour
                interval = int(dt_seconds // 60)
                major_locator = mdates.MinuteLocator(interval=max(1, interval))
                minor_locator = mdates.MinuteLocator()
            elif dt_seconds < 86400:  # < 1 day
                interval = int(dt_seconds // 3600)
                major_locator = mdates.HourLocator(interval=max(1, interval))
                minor_locator = mdates.HourLocator()
            else:
                interval = int(dt_seconds // 86400)
                major_locator = mdates.DayLocator(interval=max(1, interval))
                minor_locator = mdates.DayLocator()

        # Set formatter based on total duration
        duration_seconds = total_duration.total_seconds()
        if duration_seconds < 86400:
            formatter = mdates.DateFormatter("%H:%M")
        else:
            formatter = mdates.DateFormatter("%m-%d")

    # Generate tick positions in datetime space
    dummy_ax = fig.add_subplot(111, frame_on=False)
    dummy_ax.set_xlim(mdates.date2num(start_time), mdates.date2num(end_time))
    dummy_ax.xaxis.set_major_locator(major_locator)
    dummy_ax.xaxis.set_minor_locator(minor_locator)

    major_tick_times = mdates.num2date(dummy_ax.xaxis.get_majorticklocs())
    minor_tick_times = mdates.num2date(dummy_ax.xaxis.get_minorticklocs())
    dummy_ax.remove()

    # Map datetime ticks to normalized x positions using linear interpolation
    # time_range contains [start, end], x contains [x_min, x_max]
    start_num = mdates.date2num(start_time)
    end_num = mdates.date2num(end_time)

    major_tick_nums = mdates.date2num(major_tick_times)
    minor_tick_nums = mdates.date2num(minor_tick_times)

    # Linear mapping: normalized_x = x_min + (time - start) / (end - start) * (x_max - x_min)
    major_tick_positions = x[0] + (major_tick_nums - start_num) / (end_num - start_num) * (x[-1] - x[0])
    minor_tick_positions = x[0] + (minor_tick_nums - start_num) / (end_num - start_num) * (x[-1] - x[0])

    # Set ticks on actual axis
    ax.set_xticks(major_tick_positions)
    ax.set_xticks(minor_tick_positions, minor=True)
    ax.set_xticklabels([formatter(mdates.date2num(t)) for t in major_tick_times])

    ax.grid(True, alpha=0.3, which="major")
    ax.grid(True, alpha=0.15, which="minor")
    plt.tight_layout()
    if b_show:
        plt.show()
    return ax


def create_interpolator(time: np.ndarray, y, Z, x_scaler, method="LinearND"):
    """
    Create interpolator function t = f(time_norm, pres)
    time_norm is X values: a `time` converted to seconds resolution and to int then scaled with `x_scaler`.
    :param time_nc:
    :param y_nc:
    :param z_nc:
    :param scaler: not needed (as we use rescale=True) but is kept for compatibility of output interpolator
    :param method: Interpolator method:
    - LinearND (default), will also be used if method=None is specified
    - CloughTocher gives a smoother surface, but is still locally based on triangulation - by
    default.
    :return: interpolator, metadata dict about axes used in interpolator
    """
    z = Z.ravel()
    b_ok = np.isfinite(z)
    n_bad = z.size - b_ok.sum()

    time_nc_norm = x_scaler(time.astype("M8[s]").astype(int).reshape(-1, 1)).flatten()
    xy_ok = np.column_stack([c.ravel()[b_ok] for c in np.meshgrid(time_nc_norm, y, indexing="ij")])

    meta = {
        k: dict(zip(("min", "max"), lims))
        for k, lims in zip(["x", "y"], zip(xy_ok.min(axis=0), xy_ok.max(axis=0)))
    }
    if n_bad:
        logger.info(
            f"Removed {n_bad}/{z.size} ({100 * n_bad / z.size:.1f}%) bad values\n"
            f"New y limits: {meta['y']}"
        )

    nc_interpolator = (
        LinearNDInterpolator if method is None or method == "LinearND" else CloughTocher2DInterpolator
    )(xy_ok, z[b_ok])  #, rescale=True
    return nc_interpolator, meta


def isort_points_in_circle(points: Sequence[Tuple[float, float]], f_istart_point=None, b_clockwise=True):
    """
    Sorts points to minimize path length using nearest-neighbor algorithm.

    Creates an ordering that approximates a short closed path through all points
    by greedily selecting the nearest unvisited point at each step.

    Parameters
    ----------
    points : array-like, shape (N, 2)
        List or array of points in format [[x1, y1], [x2, y2], ...]
    f_istart_point : callable, optional
        Function that takes points array and returns the index of the starting point.
        If None, uses the point with minimum y-coordinate (minimum x if tied).
    b_clockwise : bool, default=True
        If True, prefer clockwise direction when choosing between equidistant points;
        if False, prefer counter-clockwise direction.

    Returns
    -------
    sorted_indices : ndarray, shape (N,)
        Indices defining a short path through all points.
    """
    ix, iy = 0, 1
    points = np.asarray(points, dtype=float)

    if points.size == 0:
        return np.array([], dtype=int)

    n_points = len(points)

    if n_points == 1:
        return np.array([0])

    # Define the start point function if not provided
    if f_istart_point is None:

        def f_istart_point(points):
            """Returns index of point with minimum y-coordinate (minimum x if tied)."""
            iy_min = np.where(points[:, iy] == np.min(points[:, iy]))[0]
            start_idx = iy_min[np.argmin(points[iy_min, ix])]
            return start_idx

    i_st = f_istart_point(points)

    # Nearest neighbor algorithm
    sorted_indices = [i_st]
    visited = np.zeros(n_points, dtype=bool)
    visited[i_st] = True

    current_idx = i_st
    center = np.mean(points, axis=0)  # Used for tie-breaking with angular preference

    for _ in range(n_points - 1):
        current_point = points[current_idx]

        # Find distances to all unvisited points
        # unvisited_mask = ~visited
        distances = np.sum((points - current_point) ** 2, axis=1)
        distances[visited] = np.inf  # Exclude visited points

        # Find nearest point(s)
        min_dist = np.min(distances)
        nearest_candidates = np.where(np.abs(distances - min_dist) < 1e-10)[0]

        if len(nearest_candidates) == 1:
            next_idx = nearest_candidates[0]
        else:
            # Tie-breaking: use angular preference for clockwise/counter-clockwise
            current_vec = current_point - center
            current_angle = np.arctan2(current_vec[iy], current_vec[ix])

            candidate_vecs = points[nearest_candidates] - center
            candidate_angles = np.arctan2(candidate_vecs[:, iy], candidate_vecs[:, ix])

            # Compute angular difference (normalized to [0, 2π))
            angle_diffs = np.mod(candidate_angles - current_angle, 2 * np.pi)

            if b_clockwise:
                # Choose point with smallest positive angle (clockwise = decreasing angle in math coords)
                # We want the point that's "most clockwise" = largest angle or smallest from opposite side
                next_idx = nearest_candidates[np.argmin(angle_diffs)]
            else:
                # Counter-clockwise: choose largest angle
                next_idx = nearest_candidates[np.argmax(angle_diffs)]

        sorted_indices.append(next_idx)
        visited[next_idx] = True
        current_idx = next_idx

    return np.array(sorted_indices)


# to del
def isort_points_in_circle_bad(points: Sequence[Tuple[float, float]], f_istart_point=None, b_clockwise=True):
    """
    Sorts points in a circle
    Сортирует точки по кругу
    начиная с точки с минимальным y (и минимальным x при равенстве y).

    Параметры:
        points : array-like, shape (N, 2)
            Список или массив точек в формате [[x1, y1], [x2, y2], ...]
        f_istart_point:
        b_clockwise: по часовой стрелке (True) или наоборот,

    Возвращает:
        sorted_indices : ndarray, shape (N,)
            Индексы для сортировки точек.
    """
    ix, iy = 0, 1
    points = np.asarray(points, dtype=float)

    if points.size == 0:
        return points

    # Найдём центр масс (можно заменить на другую точку, если нужно)
    center = np.mean(points, axis=0)

    # Вычислим углы относительно центра
    vectors = points - center
    angles = np.arctan2(vectors[:, iy], vectors[:, ix])

    if f_istart_point is None:

        def f_istart_point(points):
            """индекс точки с минимальным y (и минимальным x при равенстве)"""
            iy_min = np.where(points[:, iy] == np.min(points[:, iy]))[0]
            start_idx = iy_min[np.argmin(points[iy_min, ix])]
            return start_idx

    i_st = f_istart_point(points)

    # Угол стартовой точки
    angle_start = angles[i_st]

    # Сдвинем углы так, чтобы стартовая точка имела угол 0 приводя углы к диапазону [0, 2π)
    shifted_angles = np.mod(angles - angle_start, 2 * np.pi)

    # Для сортировки по часовой стрелке — сортируем по убыванию угла
    # (т.к. atan2 даёт положительные углы против часовой стрелки)
    sorted_indices = np.argsort((-1)**b_clockwise*shifted_angles)

    return sorted_indices


def lat_lon_from_cmems_nc_filestem(stem):
    """
    Extract latitude and longitude from filename ending with coordinates like 19.15E_55.22N

    Args:
        filename (str): The filename to extract coordinates from

    Returns:
        tuple: (latitude, longitude) as floats, or None if no match found
    """
    pattern = r"([+-]?\d+\.\d+)([EW])_([+-]?\d+\.\d+)([NS])$"
    match = re.search(pattern, stem)
    if match:
        lon_value = float(match.group(1))
        lon_dir = match.group(2)
        lat_value = float(match.group(3))
        lat_dir = match.group(4)

        # Convert to proper signed coordinates
        longitude = lon_value if lon_dir == "E" else -lon_value
        latitude = lat_value if lat_dir == "N" else -lat_value

        return latitude, longitude

    return None


def nc_load(path_section: Path, variables):

    path_section = Path(path_section)
    f = netCDF4.Dataset(path_section)

    # Extract dimensions from 1st variable
    z_nc = f.variables[variables[0]]

    # Get dimensions
    dims_list = z_nc.get_dims()
    dims = {d.name: d.size for d in dims_list}

    z_nc = z_nc[:].filled(fill_value=np.nan)
    if len(dims) > 2:
        z_nc = z_nc.squeeze(axis=(-2, -1))

    time_name, lat_name, lon_name = "time", "latitude", "longitude"
    if time_name in dims:
        if len(dims) > 2:  # 3D or 4D: time, [depth, ] latitude, longitude
            time_name, *_, lat_name, lon_name = dims.keys()
            if _:
                if len(_) > 1:
                    logger.warning("Don't know what is extra dimensions in {_}, taking only 1st for y")
                y_name = _[0]
            # latitudes = f.variables[lat_name][:]
            # longitudes = f.variables[lon_name][:]
        else:
            y_name = next(name for name in dims.keys() if name != time_name)
    time_var = f.variables[time_name]
    time_nc = netCDF4.num2date(
        time_var[:], time_var.units, only_use_cftime_datetimes=False, only_use_python_datetimes=True
    ).filled(fill_value=np.datetime64('NaT')).astype("M8[s]")

    y_nc = f.variables[y_name][:].data  # filled(fill_value=np.nan)

    attrs = {
        var: {
            a: f.variables[var].getncattr(a) for a in f.variables[var].ncattrs()
        } for var in variables
    }

    # Get meta (we can also get from file name which is like "phy_anfc_PT1H-i_thetao-so-sob_19.15E_55.22N.nc")
    attrs.update({
        abbr: round(f.variables[k][0].data.item(), 4) for abbr, k in [("lat", lat_name), ("lon", lon_name)]
    })
    return time_nc, y_nc, z_nc, attrs


##############################################################################################################
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # %% t-chain

    path_cruise = Path(r"F:\WorkData\BalticSea\240616_ABP56@i,t-chain")
    path_save = path_cruise / "_post_proc=o2_from_t"

    # t-chain  temperature time section
    path_section_t_chain = Path(
        r"F:\WorkData\BalticSea\240616_ABP56@i,t-chain\t-chain\text_output.7z\text_output\240625@TCm1,2.csv"
    )

    # CMEMS data directories (can be used to construct relative paths)

    # cmems_root_dir = Path(r"f:\WorkData\BalticSea\CMEMS")
    # cmems_project_dir = cmems_root_dir / "240616_ABP56(t-chain)"

    # dir for pure CMEMS processing outputs (same name as parent to fast copy and publish)
    # cmems_save_dir = cmems_project_dir / cmems_project_dir.name


    # CMEMS netCDF4 files with temperature time section
    dir_cmems_point_time_sections = Path(  # (project_dir.glob(path_cruise.name.split("@")[0]))
        r"F:\WorkData\BalticSea\CMEMS\240616_ABP56(t-chain)\2024-06-25..09-05\(0.50-91.31m)points_of_~same_depth(so,sob,thetao;o2,o2b)@cmems_mod_bal"
    )
    cmems_filename_glob = "phy_anfc_PT1H-i_thetao-so-sob_*"
    # r"F:\WorkData\BalticSea\CMEMS\240616_ABP56(t-chain)\240616_ABP56(t-chain)\so,sob,thetao,V,wo(time,depth)\1_240625_0000_thetao.nc"


    # Define constants for column names
    col_O = "O2ppm"
    col_T = "Temp"
    col_P = "Pres"

    # Load saved regular t-chain grid
    grid_tc = np.load(path_save / "240625_2000_grid_Temp(time, Pres).npz")  # "T", "t_ns", "p_dbar"
    shape_orig = grid_tc["T"].shape  # (time, pres)
    time_step_s_tc = sub(*grid_tc["t_ns"][1::-1]).astype("m8[s]").item().total_seconds()
    time_range = grid_tc["t_ns"][[0, -1]]
    y_min = np.nanmin(grid_tc["p_dbar"])
    y_max = np.nanmax(grid_tc["p_dbar"])
    z_min = np.nanmin(grid_tc["T"])
    z_max = np.nanmax(grid_tc["T"])
    logger.info(
        "\n".join([
            f"Loaded 2d data time range: {time_range[0]} - {time_range[1]}, shape: {shape_orig}, "
            f"step: {time_step_s_tc}s",
            f"{col_P} range: [{y_min:g}, {y_max:g}]",
            f"{col_T} range: [{z_min:g}, {z_max:g}]",
        ])
    )

    # Starting depths at min(col_T) from grid
    iy_st = np.nanargmin(grid_tc["T"], axis=1)
    y_st = grid_tc["p_dbar"][iy_st]
    z_st = grid_tc["T"][np.arange(shape_orig[0]), iy_st]
    y_st_min = y_st.min()
    logger.info(f"Starting min and max depths for min({col_T}): ({y_st_min:g}, {y_st.max():g})")

    logger.info(f"min({col_T}): {z_st} corresponds to ")
    logger.info(f"- depths: {y_st}")

    loaded = np.load(path_save / "240706_1122_points_O2ppm(Temp).npz")
    f_t_to_o1 = create_extrap_func(loaded["x"], loaded["y"])


    o_st = f_t_to_o1(z_st)
    logger.info(f"- ~max {col_O}: {o_st}")

    # Prepare common coordinates and transformation to scale all dataset time to same coordinate units
    t_int = grid_tc["t_ns"].astype('M8[s]').astype(int)

    scaler = MinMaxScaler()
    scaler.fit(t_int.reshape(-1, 1))  # Only fit once to able apply same scaler to all next data

    x_tc = scaler.transform(t_int.reshape(-1, 1))  # Reshape to column
    X_tc, Y_tc = np.meshgrid(x_tc, grid_tc["p_dbar"], indexing="ij")


    # %% CMEMS data
    variables = ["thetao"]  # put variable with max dimensions first (dim. will be determined from it)

    # Sort points
    path_cmems_point_time_sections = list(dir_cmems_point_time_sections.glob(cmems_filename_glob))
    points_lon_lat = [lat_lon_from_cmems_nc_filestem(p.stem)[::-1] for p in path_cmems_point_time_sections]
    ipoints = isort_points_in_circle(points_lon_lat, b_clockwise=False)
    path_cmems_point_time_sections = [path_cmems_point_time_sections[i] for i in ipoints]

    # Process time sections for all points
    for i1point, path_section in enumerate(path_cmems_point_time_sections, start=1):
        # Open netCDF4 file
        time_nc, y_nc, z_nc, meta = nc_load(path_section, variables)

        # Check that sufficient CMEMS data time range is loaded
        assert time_nc[0] <= grid_tc["t_ns"][0]
        assert time_nc[-1] >= grid_tc["t_ns"][-1]

        # Interpolate to common coordinates

        # interpolator function t = f(time_norm, pres)
        nc_interpolator, lims = create_interpolator(
            time_nc, y_nc, z_nc, scaler.transform, method="CloughTocher"
        )
        b_y_ok = (lims["y"]["min"] <= Y_tc[0, :]) & (Y_tc[0, :] <= lims["y"]["max"])
        Y_ok = Y_tc[:, b_y_ok]
        X_ok = X_tc[:, b_y_ok]
        # Calculate 2D array col_T for same regular time, pres as t-chain to compare with later
        # Apply the interpolator to get temperature values on the regular grid
        T_nc = nc_interpolator(X_ok, Y_ok)  # regular 2D

        # Starting depths at min(col_T) from grid
        iy_st_nc = np.nanargmin(T_nc, axis=1)  # all(iy_st_nc == 16)
        y_st_nc = Y_ok[0, iy_st_nc]
        iy_st_nc_min = iy_st_nc.min()
        y_st_nc_min = y_st_nc[iy_st_nc_min]  # z_nc_min = T_nc[np.arange(shape_orig[1]), iy_st_nc]

        O_nc = f_t_to_o1(T_nc[:, iy_st_nc_min:])
        o_st_nc = O_nc[np.arange(shape_orig[0]), iy_st_nc - iy_st_nc_min]

        # original CMEMS data resolution
        time_step_s_nc = sub(*time_nc[1::-1]).astype("m8[s]").item().total_seconds()
        logger.info(
            f"Point {i1point}. Interpolated CMEMS {path_section.name} data "
            f"from {time_step_s_nc} to {time_step_s_tc} time resolution.\n"
            f"Starting min and max depths for min({col_T}): {y_st_nc_min:g}..{y_st_nc.max():g}"
            f"~max {col_O}: {o_st_nc}"
        )

        meta_str = f"{i1point} ({meta['lat']}, {meta['lon']})"
        meta_str_short = f"{i1point}({meta['lat']:2.2f},{meta['lon']:2.2f})"
        for lbl_z_cur, z_cur in [(col_T, T_nc), (col_O, O_nc)]:
            ax = plot_2d(
                X=X_ok[:, iy_st_nc_min:] if lbl_z_cur == col_O else X_ok,
                Y=Y_ok[:, iy_st_nc_min:] if lbl_z_cur == col_O else Y_ok,
                Z=z_cur,
                time_range=grid_tc["t_ns"][[0, -1]],
                y_lims = [49.930626, 83.81112],
                y_overlay=Y_tc[np.arange(shape_orig[0]), iy_st_nc],
                title=f"CMEMS point {i1point} ({meta['lat']}, {meta['lon']}) {lbl_z_cur} contours",
                lbl_x="Time, m-d",
                lbl_y="Depth, m",
                lbl_z=f"{lbl_z_cur}",
                lbl_overlay=f"min({col_T}) depth",  # f"min({col_T}) depth overlay",
                contour_levels=range(12) if lbl_z_cur == col_O else np.arange(3.4, 8, 0.2),  # 3.5, 8, 0.5
                cmap="rainbow_r" if lbl_z_cur == col_O else "rainbow",
                b_show=False
            )
            path_save_fig = path_section.with_name(
                f"{meta_str_short}_{lbl_z_cur}@CMEMS.png"  # _with_y(min({col_T}))
            )
            ax.figure.savefig(
                path_save_fig,
                format="png",
                dpi=300,
                transparent=False,
            )   # , bbox_inches="tight"
            logger.info(f"Plot saved to {path_save_fig}")
            plt.close(ax.figure)

        # Save NetCDF grid for Surfer here
        dir_save_nc = dir_cmems_point_time_sections.with_name("_interp_to_t-chain_regular")
        stem_sfx = "@CMEMS"
        b_save_to_netcdf = not any(dir_save_nc.glob(f"{meta_str_short}*{stem_sfx}.nc"))
        if b_save_to_netcdf:
            save_nc_for_surfer(
                time=grid_tc["t_ns"],
                y=-Y_ok[0, :],  # grid_tc["p_dbar"],
                out={"T": T_nc.T, "O": O_nc.T},  # puts y 1st for Surfer, as required by save_nc_for_surfer()
                path_base=dir_save_nc / meta_str_short,
                dt=np.timedelta64(int(time_step_s_tc), "s"),
                not_interp_keys={"T", "O"},
                stem_sfx=stem_sfx,
                attrs={
                    "T": {
                        "standard_name": "sea_water_potential_temperature",
                        "units": "degree_Celsius",
                        "name": "temperature",
                    },
                    "O": {"standard_name": "mass_concentration_of_oxygen_in_sea_water", "units": "mg l-1"},
                },
                lat=meta["lat"],
                lon=meta["lon"],
            )




# %%
