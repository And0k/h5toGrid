#!/usr/bin/python3.7+
"""
Convert NetCDF data that was downloaded by download_copernicus.py to csv or grd
Input file can be, for example, loaded from dataset-bal-analysis-forecast-phy-monthlymeans dataset and have variables:
uo,vo,mlotst,so,sob

NAME    STANDARD NAME       UNITS
bottomT sea water potential temperature at sea floor, degrees C
mlotst  ocean mixed layer thickness defined by sigma theta, m
siconc  sea ice area fraction   1
sithick sea ice thickness, m
sla     sea surface height above sea level, m
so      sea water salinity * 0.001
sob     sea water salinity at sea floor * 0.001
thetao  sea water potential temperature, degree Celsius
uo      eastward sea water velocity, m/s
vo      northward sea water velocity, m/s
"""
import re
import numpy as np
from pathlib import Path
import pandas as pd
from collections import defaultdict

def constant_factory(val):
    def default_val():
        return val
    return default_val


def calc_iso_surface(
    v3d: np.ndarray,
    v_isosurface,
    zi: np.ndarray,
    interp_order: int = 1,
    weight_power: float = 1,
    dv_max: float = 20,
    dy_ambiguity: int = 1,
) -> np.array:
    """
    weighted average to compute the iso-surface
    :param v3d: with zi axes last
    :param v_isosurface: sequence of values of isolines
    :param zi: 1d or 2d array of z values corresponded to v3d data last dimension(s)
    :param interp_order: number of points to one side of requested
    :param weight_power: 1 for linear interpolation
    :param dy_ambiguity: 1 for exclude any ambiguity else eyes of this height - 1 (in bins) is allowed
    :return z2d: 2D array of shape v3d.shape[:-1] of nearest z values

    """
    # v3d = np.moveaxis(np.atleast_3d(v3d), -1, 0)
    # v3d.shape

    # Add top & bot edges
    dv = v3d[None, ...] - np.array(v_isosurface)[:, None, None, None]
    try:
        dv_filled = dv.filled(np.nan)
    except AttributeError:
        dv_filled = dv
        # dv_filled.fill(np.nan)
    shape_collect = [len(v_isosurface)] + list(v3d.shape[:-1]) + [interp_order * 2]
    # collect nearest values with positive dv and negative separately to:
    arg_nearest = np.zeros(shape_collect, dtype=int)
    # collect its weights and mask to:
    z_weight = np.ma.zeros(shape_collect, dtype=float)




    # Check if closest points on each side is not adjacent points (also deletes where some side have no points?)
    if isinstance(dy_ambiguity, float):
        # Find positions of minimum abs(dv) values for negative dv and positive dv separately
        for proc_dv_sign, sl_collect in [(-1, slice(0, interp_order)), (1, slice(interp_order, None))]:
            # mask outside current (positive/negative) side of dv
            dv_ma = np.ma.masked_outside(proc_dv_sign * dv_filled, 0, dv_max)  # good if 0 < output < dv_max
            # find nearest elements of current side
            arg = dv_ma.argsort(axis=-1, kind='mergesort')[..., :interp_order]   # use stable algorithm

            # keep elements of current side by mask with weights
            arg_nearest[..., sl_collect] = arg

            dv_sorted = np.take_along_axis(dv_ma, arg, axis=-1)
            z_weight[..., sl_collect] = 1 / dv_sorted ** weight_power

        b_ambiguous = (
            abs(
                np.diff(arg_nearest[..., (interp_order - 1) : (interp_order + 1)], axis=-1)
            )[..., 0] > dy_ambiguity
        )
    elif dy_ambiguity == 'last':
        # Find the first elements that are less than isoline, previous elements should be bigger
        dv = dv[..., ::-1]  # invert to search from last
        b_ambiguous = np.zeros(shape_collect[:-1], dtype=bool)
        b_negative = np.empty_like(v3d)
        i_first = np.empty(v3d.shape[:-1], dtype=int)
        b_bad = np.empty(v3d.shape[:-1], dtype=bool)
        for i_iso in np.arange(len(v_isosurface)):
            b_negative[...] = dv[i_iso, ...] < 0
            i_first[...] = np.argmax(b_negative, axis=-1)
            arg_nearest[i_iso, ..., 0] = i_first

            # Mark bad indexes which are 0 because of no cross of isoline value
            b_bad[...] = i_first == 0
            b_bad[b_bad] = b_negative[b_bad, :].all(axis=-1) | ~b_negative[b_bad, :].any(axis=-1)
            b_ambiguous[i_iso, b_bad] = True

            i_first -= 1
            arg_nearest[i_iso, ..., 1] = i_first

        # recover after inversion
        dv = dv[..., ::-1]                               # original dv
        arg_nearest = (v3d.shape[-1] - 1) - arg_nearest  # indexes for original dv

        arg_nearest_clipped = np.clip(arg_nearest, 0, v3d.shape[-1]-1)
        for proc_dv_sign, sl_collect in [(-1, slice(0, interp_order)), (1, slice(interp_order, None))]:
            z_weight[..., sl_collect] = (
                1 / np.take_along_axis(dv, arg_nearest_clipped[..., sl_collect], axis=-1)
                ** weight_power
            )
        z_weight[arg_nearest_clipped!=arg_nearest] = 0


    # # set output to NaN if from some side have no points:
    # b_outside = ~arg_nearest[..., (interp_order - 1):(interp_order + 1)].all(axis=-1)
    if zi.ndim == 1:
        z_nearest = zi[arg_nearest]
    else:
        z_nearest = np.take_along_axis(np.tile(zi[None, None,...], (1, 1, 2)), arg_nearest, axis=-1)
    if b_ambiguous is not None:
        z_nearest[b_ambiguous, :] = np.nan
    # average
    return np.ma.average(z_nearest, weights=abs(z_weight), axis=-1)
    # np.ma.masked_array(z_nearest, z_weight.mask)


##############################################################################################################
if __name__ == "__main__":
    file_path = r"C:\Work\_\t-chain\240625@TCm1,2.csv"
    # r"D:\WorkData\BalticSea\240616_ABP56\t-chain\240625@TCm1,2.csv"

    path_in = Path(file_path)
    output_dir = path_in.parent
    # put variable with max dimensions first (they will be determined from it)
    variables = ['t']

    # make "x: lambda x" be default value
    variables_apply_coef = defaultdict(
        constant_factory(lambda x: x),
        {'depth': lambda x: -x}
    )  # 'so': lambda x: x*1e3, 'sob': lambda x: x, *1e3

    var_short_names = variables
    z_iso = {'t': np.arange(4.5, 7.5 + 0.001, step=0.5)}

    b_2d_to_txt = False     # Save to text / grid
    interp_order = 1  # weighted mean between 2 nearest points


    df_in = pd.read_csv(path_in, index_col="Time", date_format="ISO8601", dtype=np.float32)


    time_edges = df_in.index[[0,-1]]
    print(f"Loaded data length: {df_in.shape[0]} points, {df_in} {time_edges}")
    v2d_in = df_in.to_numpy() # .loc[:, [col for col in]]
    v2d, z2d = np.split(v2d_in, 2, axis=1)

    i_v_min = v2d.argmin(axis=1)


    # # mask of all data to the min values from the top
    # mask = np.arange(v2d.shape[1])[None, :] < i_v_min[:, None]
    # mask &= v2d <

    for v_name in variables:
        out = calc_iso_surface(
            np.moveaxis(np.atleast_3d(v2d), -1, 0),
            z_iso[v_name],
            z2d,
            interp_order=interp_order,
            weight_power=1,
            dv_max=20,
            dy_ambiguity="last",
        )  # v2d[..., None]
        out = out[:, 0, :].filled(np.nan)
        df_out = pd.DataFrame(
            -out.T,
            columns=[f"z({v_name}={z})" for z in z_iso[v_name]],
            index=df_in.index,
        )

        # Save...
        pref, sfx = path_in.stem.split("@", 1)
        if sfx:
            sfx = f'@{sfx}'
        path_out = output_dir / f"{pref}isolines({v_name}){sfx}.tsv"

        # Save to hdf5
        df_out.rename(
            columns={
                c: re.sub("[)=]|.0", "", re.sub("[(]", "_", c))
                .replace("(", "_")
                .replace(".", "p")
                for c in df_out.columns
            },
        ).to_hdf(
            path_out.with_suffix(".h5"),
            key=f"isolines_{v_name}",
            format="table",
            data_columns=True,
            append=False,
            # index=False,
        )

        # Save to text file
        # Convert the datetime index to Excel serial time
        df_out.index = (df_out.index - pd.Timestamp("1899-12-30")) / pd.Timedelta("1D")
        df_out.to_csv(
            path_out,
            index_label="DateTime_UTC",
            date_format="%Y-%m-%dT%H:%M:%S",  # .%f
            sep='\t'
        )

        df_out.iloc[(df_out.iloc[:, 0] < -80).values, 0]
    # for tp in df_in.itertuples(index=False, name=None):
    #     p_out = tp.


if b_2d_to_txt:
    def save2d(v2d, t, v_name):
        file_name = f'{t:%y%m%d_%H%M}{v_name}.csv'
        file_path = output_dir / file_name
        print(f'Writing to {file_name}')
        df = pd.DataFrame(v2d, index=latitudes, columns=longitudes)
        df.to_csv(file_path)

else:
    from gs_surfer import save_grd

    x_min, x_max = longitudes[[0, -1]]
    y_min, y_max = latitudes[[0, -1]]
    x_resolution = np.diff(longitudes[:2]).item()
    y_resolution = np.diff(latitudes[:2]).item()
    # check grid is ok
    np.testing.assert_almost_equal(x_resolution, (x_max - x_min) / (longitudes.size - 1), decimal=5)
    np.testing.assert_almost_equal(y_resolution, (y_max - y_min) / (latitudes.size - 1), decimal=5)  # 6 works too not 7

    def save2d(v2d, t, v_name):
        file_name = f'{t:%y%m%d_%H%M}{v_name}.grd'
        file_path = output_dir / file_name
        print(f'Writing to {file_name}')
        save_grd(np.flipud(v2d.filled(np.nan)), x_min, y_max, x_resolution, y_resolution, file_path)


if method == 'file_for_each_time':      # METHOD 1
    # Write 2D each time
    Path.mkdir(output_dir, parents=True, exist_ok=True)
    for v_name in variables:
        v = f.variables[v_name]                                                             # netcdf var
        v = variables_apply_coef[v_name](v[:, :, :, :] if v.ndim > 3 else v[:, :, :])    # numpy var
        if v.ndim > 3:
            # Extract data v_name as a 3D numpy array, calculate and write 2D isosurfaces
            z_values = variables_apply_coef['depth'](f.variables[expver_name[0]][:])  # depth values
            for i, t in enumerate(times):
                v3d = np.moveaxis(v[i, ...], 0, -1)
                for z_iso in z_isosurface[v_name]:
                    v2d = calc_iso_surface(
                        v3d, v_isosurface=z_iso, zi=z_values, interp_order=interp_order
                    )
                    save2d(v2d, t, f'z({v_name}={z_iso})')
        else:
            # Extract data v_name as a 2D pandas DataFrame and write it to CSV
            for i, t in enumerate(times):
                v2d = v[i, ...]
                save2d(v2d, t, v_name)


        print('saved to', filename)



print(f"{datetime.now():%Y-%m-%d %H:%M:%S} Ok>")
