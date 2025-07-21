# %%
from datetime import datetime
from typing import Optional, Tuple, Sequence
from pathlib import Path
import pandas as pd
import xarray as xr
# import matplotlib.pyplot as plt
import numpy as np
try:
    import cftime  # testing requirement for downloading NCEP data
except ImportError:
    print("!pip install cftime", "required")
    import cftime
try:
    import netCDF4  # testing requirement for downloading NCEP data
except ImportError:
    print("!pip install netCDF4", "required")
    import netCDF4

from utils import interp_to_point

xr_backend = "h5netcdf"  # not for NCEP data



def load_era5_wind_data(
    file_path: str, lat_range: Tuple[float, float], lon_range: Tuple[float, float]
) -> xr.Dataset:
    """
    Loads ERA5 wind data from a NetCDF file for a specified geographical range.

    :param file_path: Path to the ERA5 NetCDF file.
    :param lat_range: Tuple containing the minimum and maximum latitude (min_lat, max_lat).
    :param lon_range: Tuple containing the minimum and maximum longitude (min_lon, lon_max).
    :return: Xarray Dataset containing the filtered ERA5 wind data.
    """
    # Open the NetCDF file using xarray
    ds = xr.open_dataset(file_path, decode_timedelta=True)

    # Find latitude and longitude coordinate names and select data within ranges
    ds_subset = ds
    for coord_name, range_val, name_options in [
        ("latitude", lat_range, ["latitude", "lat"]),
        ("longitude", lon_range, ["longitude", "lon"]),
    ]:
        found_coord = None
        for name_option in name_options:
            if name_option in ds_subset.coords:
                found_coord = name_option
                break
        if found_coord and range_val:
            ds_subset = ds_subset.sel({found_coord: slice(range_val[0], range_val[1])})
        elif range_val:
            print(f"Warning: {coord_name.capitalize()} coordinate not found in dataset.")

    return ds_subset


def get_nc_file_content_if_matches(
    out_path: Path,
    lat: float,
    lon: float,
    date_range: Sequence[str],
    vars_list: list,
    tolerance=0.25,
    xr_backend=xr_backend,
) -> Optional[xr.Dataset]:
    """
    Checks if a local NetCDF file exists at the given path and matches the requested parameters.

    :param out_path: Path to the potential existing NetCDF file or directory.
    :param lat: latitude of the point to check.
    :param lon: longitude of the point to check.
    :param date_range: [start, end] date in "YYYY-MM-DD" to check.
    :param vars_list: list of variables to check for.
    :param tolerance: lat/lon tolerance
    :return: Xarray Dataset if a matching file exists, otherwise None.
    """
    # Determine the actual file path if out_path is a directory
    if out_path.is_dir():
        # Compose a filename based on input parameters
        out_filename = (
            f"{lon:.6g}E_{lat:.6g}N_{date_range[0]}-{date_range[-1]}({','.join(sorted(vars_list))}).nc"
        )
        file_path = out_path / out_filename
    else:
        file_path = out_path

    print(f"Checking for existing file: {file_path}")
    if not file_path.exists() or not file_path.is_file():
        print(f"File not found at '{file_path}'.")
        return None

    try:
        # Open the existing file to check its contents
        with xr.open_dataset(file_path, engine=xr_backend, decode_timedelta=True) as existing_ds:
            # Check if coordinates match the input
            coord_match = True
            for name_options, value in [
                (("lat", "latitude"), lat),
                (("lon", "longitude"), lon),
            ]:  # , (("lat", "latitude"), lat), (("lon", "longitude"), lon)]:
                found_coord = False
                for name_option in name_options:
                    if name_option in existing_ds.coords:
                        existing_coords = existing_ds[name_option].to_numpy()
                        if len(existing_coords) == 1 and abs(existing_coords.item() - value) < tolerance:
                            found_coord = True
                        elif len(existing_coords) == 4:  # Check for 4 nearest points
                            # This is a simplification; a more robust check would involve
                            # verifying if the 4 points are indeed the nearest neighbors
                            # around the target lat/lon within a certain radius/grid spacing.
                            # For this modification, we assume if there are 4 points,
                            # they represent the nearest neighbors.
                            found_coord = True
                        else:
                            print(f"existed {name_option} coordinates size {len(existing_coords)} != 1 or 4")

                        if not found_coord:
                            print(
                                f"existed {name_option} coordinates do not match the expected configuration for {value}"
                            )
                        break
                else:
                    print(f"not found {name_options} coordinates in file!")
                if not found_coord:
                    coord_match = False
                    break

            # Check time range
            time_match = False
            if "time" in existing_ds.coords:
                # Check if the existing time range exactly matches the requested range
                existing_time_range = [
                    pd.to_datetime(existing_ds["time"].values.min()),
                    pd.to_datetime(existing_ds["time"].values.max()),
                ]
                requested_time_range = [pd.to_datetime(t) for t in date_range]
                time_match = all(
                    requested_t == existing_t
                    for requested_t, existing_t in zip(requested_time_range, existing_time_range)
                )
                if not time_match:
                    print(
                        f"existed time range {existing_time_range} not matches required "
                        f"{requested_time_range}"
                    )
            # Check if required variables exist
            vars_match = all(v in existing_ds.variables for v in vars_list)

            if coord_match and time_match and vars_match:
                print(f"Existing file '{file_path}' matches input parameters. Returning existing data.")
                return existing_ds  # Return the existing dataset

            print(f"Existing file '{file_path}' does not match input parameters.")
    except Exception as e:
        print(f"Error checking existing file '{file_path}': {e}. Proceeding to download.")

    return None


def output_file_dir_path(out: Path, lat, lon, start, end, vars_list) -> Tuple[Path, Path | None]:
    """
    Determines the output file path and creates the necessary directory if 'out' is a directory.
    If 'out' is a file path, it ensures the parent directory exists.

    :param out: Desired output path. If a directory, a filename will be composed. If a file path, it will be used directly.
    :param lat: latitude used for composing filename if 'out' is a directory.
    :param lon: longitude used for composing filename if 'out' is a directory.
    :param start: start date string used for composing filename if 'out' is a directory.
    :param end: end date string used for composing filename if 'out' is a directory.
    :param vars_list: list of variables used for composing filename if 'out' is a directory.
    :return: Tuple of (file_path, dir_path_created).
        - file_path: The determined path to the output NetCDF file.
        - dir_path_created: The path to the directory that was created, or None if no directory was created.
    """
    if out is None:
        return None, None
    if isinstance(out, str):
        out = Path(out)

    if out.is_dir():
        # Compose a filename based on input parameters
        if not isinstance(lon, float) or not isinstance(lat, float):
            try:
                out_filename = "".join([
                    "" if "NCEP" in out.name else "wind@NCEP-CFSv2_",
                    "area({:g}-{:g}N,{:g}-{:g}E)".format(*lat.round(2), *lon.round(2)),
                ])
            except Exception as e:
                print("lat and lon are not float arrays!", e)
                raise ValueError(
                    f"Latitude and longitude must be floats. Have: ({lat}, {lon}) of type {type(lat)}")
        else:
            out_filename = f"{lon:.6g}E_{lat:.6g}N_{start}-{end}"
        out_filename = f'{out_filename}({",".join(sorted(vars_list))}).nc'
        out_dir_created, file_path = out, out / out_filename
    else:
        file_path = out
        out_dir_created = out.parent
    # Ensure the parent directory of the output file exists
    if out_dir_created.exists():
        out_dir_created = None
    else:
        out_dir_created.mkdir(parents=True, exist_ok=True)
    return file_path, out_dir_created


def load_ncep_cfs_local(
    file_path: Path,
    lat_range: Optional[Tuple[float, float]] = None,
    lon_range: Optional[Tuple[float, float]] = None,
    xr_backend="scipy",
) -> Optional[xr.Dataset]:
    """
    Loads NCEP CFSv2 wind data from a NetCDF file or a directory of NetCDF files
    for a specified geographical range.

    :param file_path: Path to the NCEP CFSv2 NetCDF file or directory.
    :param lat_range: Tuple containing the minimum and maximum latitude (min_lat, max_lat).
        If None, no latitude filtering on 'lat'/'latitude' vars is applied. ,
    :param lon_range: Tuple containing the minimum and maximum longitude (lon_min, lon_max).
        If None, no longitude filtering on 'lon'/'longitude' vars is applied.
    :return: Xarray Dataset containing the filtered and combined NCEP CFSv2 wind data,
        or None if no files found or required variables are missing.
    """
    n_files = 1
    if file_path.is_dir():
        # If file_path is a directory, open multiple files
        # Assuming all .nc files in the directory should be combined
        nc_files = sorted(file_path.glob("*.nc"))
        n_files = len(nc_files)
        if nc_files:
            print(f"Loading {n_files} files found in directory: {nc_files}")
        else:
            print(f"No NetCDF files found in directory: {file_path}")
            return None
        if n_files > 1:
            ds = xr.open_mfdataset(nc_files, combine="by_coords", decode_timedelta=True)
        else:
            # If only one file in the directory, treat it as a single file case
            file_path = nc_files[0]
            ds = xr.open_dataset(file_path, decode_timedelta=True)  # , engine=xr_backend
    elif file_path.is_file():
        # If file_path is a single file, open it
        print(f"Loading single file: {file_path}")
        ds = xr.open_dataset(file_path, decode_timedelta=True)  # , engine=xr_backend
    else:
        print(f"Error: Input path does not exist or is not a file or directory: {file_path}")
        return None

    # Select data within the specified latitude and longitude ranges if provided
    ds_subset = ds
    for name_options, range_val in [(("lat", "latitude"), lat_range), (("lon", "longitude"), lon_range)]:
        if range_val is not None:
            found_coord = False
            for name_option in name_options:
                try:
                    ds_subset = ds_subset.sel({name_option: slice(range_val[0], range_val[1])})
                    found_coord = True
                    break
                except KeyError:
                    continue
            if not found_coord:
                print(f"Warning: {name_options} coordinates not found in dataset.")

    # Select required variables
    required_vars = [
        "U_GRD_L103",
        "V_GRD_L103",
        "latitude",
        "longitude",
        "lat",
        "lon",
        "wnd10mu",
        "wnd10mv",
    ]
    # Update to check for any of the required variables before filtering
    available_vars = [var for var in required_vars if var in ds_subset.variables]
    if available_vars:
        # Select only the available required variables
        ds_subset = ds_subset[available_vars]
    else:
        print(f"Warning: None of the required variables {required_vars} found in dataset.")
        return None

    # Ensure time coordinate is datetime
    if "time" in ds_subset.coords:
        ds_subset["time"] = pd.to_datetime(ds_subset["time"])

    return ds_subset


def convert_time_to_seconds_since_epoch(ds: xr.Dataset, epoch: str = "1970-01-01 00:00:00") -> xr.Dataset:
    """
    Converts the time coordinate of an xarray Dataset to seconds since a specified epoch.

    :param ds: The xarray Dataset with a time coordinate.
    :param epoch: The reference epoch as a string (e.g., '1970-01-01 00:00:00').
    :return: A new xarray Dataset with the time coordinate in seconds since the epoch.
    :raises ValueError: If the dataset does not have a 'time' coordinate.
    """
    if "time" not in ds.coords:
        raise ValueError("Dataset must have a 'time' coordinate.")

    # Ensure the time coordinate is in datetime objects
    if not pd.api.types.is_datetime64_any_dtype(ds["time"].dtype):
        # Attempt to convert to datetime, coercing errors
        try:
            ds["time"] = ds["time"].astype("datetime64[ns]")
        except Exception as e:
            print(f"Warning: Could not convert time coordinate to datetime64[ns]: {e}")
            print("Attempting conversion using pandas.to_datetime with errors='coerce'")
            try:
                ds["time"] = pd.to_datetime(ds["time"].values, errors="coerce")
            except Exception as e_pd:
                raise ValueError(f"Could not convert time coordinate to datetime objects: {e_pd}") from e_pd

    # Define the epoch datetime object
    epoch_dt = pd.to_datetime(epoch)

    # Calculate the time difference in seconds as a floating-point number
    time_diff_seconds = (ds["time"].to_numpy().astype("datetime64[ns]") - np.datetime64(epoch_dt)).astype(
        np.float64
    ) / 1e9

    # Create the units string dynamically
    units_string = f"seconds since {epoch_dt:%Y-%m-%d %H:%M:%S}"

    # Create a new DataArray for the time coordinate with updated attributes
    new_time_coord = xr.DataArray(
        time_diff_seconds,
        coords={"time": time_diff_seconds},
        dims=("time",),
        name="time",
        attrs={
            "units": units_string,
            "calendar": "standard",  # or 'gregorian' or 'proleptic_gregorian'
            "standard_name": "time",
            "long_name": "Time",
        },
    )

    # Replace the old time coordinate with the new one
    new_dataset = ds.copy()
    new_dataset["time"] = new_time_coord
    new_dataset = new_dataset.set_coords("time")
    return new_dataset


def save_dataset_with_time_encoding_hangs(
    ds: xr.Dataset, file_path: Path, epoch: str = "1970-01-01 00:00:00"
):
    """
    Handles both cftime and datetime64 time values with robust epoch parsing
    """
    # Create a copy to avoid modifying original
    ds_to_save = ds.copy()

    if "time" in ds_to_save.coords:
        time_values = ds_to_save.time.values

        # Handle cftime objects
        if isinstance(time_values[0], cftime.datetime):
            # Parse epoch using pandas for robustness
            epoch_dt = pd.to_datetime(epoch)
            epoch_cf = cftime.DatetimeGregorian(
                epoch_dt.year, epoch_dt.month, epoch_dt.day, epoch_dt.hour, epoch_dt.minute, epoch_dt.second
            )

            # Calculate seconds since epoch
            time_sec = np.array([(t - epoch_cf).total_seconds() for t in time_values])
        elif pd.api.types.is_datetime64_any_dtype(time_values):
            # Handle numpy datetime64
            epoch_dt = np.datetime64(pd.to_datetime(epoch))
            time_sec = (time_values - epoch_dt) / np.timedelta64(1, "s")
        else:
            raise TypeError(f"Unsupported time type: {type(time_values[0])}")

        # Convert to int32 and verify range
        if time_sec.max() > 2147483647 or time_sec.min() < -2147483648:
            raise ValueError("Time values exceed int32 range")
        time_int32 = time_sec.astype("int32")

        # Update time coordinate
        ds_to_save = ds_to_save.assign_coords(time=("time", time_int32))

        # Set required attributes
        ds_to_save.time.attrs.update({"units": f"seconds since {epoch}", "calendar": "standard"})

        # Encoding only for dtype
        encoding = {"time": {"dtype": "int32"}}
    else:
        encoding = {}

    # Save with classic format
    ds_to_save.to_netcdf(file_path, format="NETCDF4_CLASSIC", encoding=encoding)


def save_dataset_with_time_encoding(
    ds: xr.Dataset,
    file_path: Path,
    epoch: Optional[str] = "1970-01-01 00:00:00",  # Made epoch optional
    format="NETCDF4",  # Changed default format
    **kwargs,
):
    """
    Save dataset with flexible time encoding options:
    - If the time coordinate is already numeric, it is saved as is with its original attributes.
    - If the time coordinate is datetime objects, it is converted to seconds since the specified epoch (default)
      or kept in original source units if epoch is None (requires preserved attributes).
    This version simplifies the time encoding logic to avoid potential hangs.
    """
    ds_to_save = ds.copy()
    encoding = {}

    if "time" in ds_to_save.coords:
        time_var = ds_to_save.time

        # Check if time is already numeric
        if np.issubdtype(time_var.dtype, np.number):
            print("Saving time as numeric values as is (inferred from dtype).")
            # Preserve original units and calendar if they exist
            if "units" in time_var.attrs:
                encoding["time"] = {"units": time_var.attrs["units"]}
            if "calendar" in time_var.attrs:
                if "time" in encoding:
                    encoding["time"]["calendar"] = time_var.attrs["calendar"]
                else:
                    encoding["time"] = {"calendar": time_var.attrs["calendar"]}

            # Set dtype explicitly to match the numeric values in the dataset
            if "time" not in encoding:
                encoding["time"] = {}
            encoding["time"]["dtype"] = str(time_var.dtype)

        # If time is not numeric, it's assumed to be datetime objects or needing decoding
        elif pd.api.types.is_datetime64_any_dtype(time_var.dtype) or isinstance(
            time_var.values[0], cftime.datetime
        ):
            # If epoch is None, try to save with original attributes
            if epoch is None and "units" in time_var.attrs and "calendar" in time_var.attrs:
                print("Saving time using original source units and calendar.")
                encoding["time"] = time_var.attrs.copy()  # Copy attributes
            else:
                # Default to converting to seconds since epoch if epoch is specified or attributes are missing
                epoch_str = epoch if epoch is not None else "1970-01-01 00:00:00"
                print(f"Converting time to seconds since '{epoch_str}' for saving.")
                try:
                    epoch_dt = pd.to_datetime(epoch_str)
                except Exception as e:
                    print(
                        f"Warning: Invalid epoch '{epoch_str}'. Using default '1970-01-01 00:00:00'. Error: {e}"
                    )
                    epoch_str = "1970-01-01 00:00:00"
                    epoch_dt = pd.to_datetime(epoch_str)

                # Calculate seconds since epoch
                if isinstance(time_var.values[0], cftime.datetime):
                    seconds_since_epoch = cftime.date2num(
                        time_var.values,
                        f"seconds since {epoch_str}",
                        calendar="standard",  # Use standard calendar for saving
                    )
                elif pd.api.types.is_datetime64_any_dtype(time_var):
                    epoch_np = np.datetime64(epoch_dt)
                    seconds_since_epoch = (time_var.values - epoch_np) / np.timedelta64(1, "s")
                else:
                    print(
                        f"Warning: Unsupported time type {type(time_var.values[0])} for conversion to seconds. Skipping time encoding."
                    )
                    # Remove time encoding if type is unsupported
                    if "time" in encoding:
                        del encoding["time"]
                    pass  # Skip encoding for this type
                    # We won't raise an error here, just warn and try to save without specific time encoding.

                # If conversion was successful, set encoding and update coordinate
                if "time" not in encoding:  # Only proceed if time encoding wasn't skipped
                    if seconds_since_epoch.max() > 2147483647 or seconds_since_epoch.min() < -2147483648:
                        print("Warning: Time values exceed int32 range. Saving as float64.")
                        time_sec = seconds_since_epoch.astype("float64")
                        encoding["time"] = {"dtype": "float64"}
                    else:
                        time_sec = seconds_since_epoch.astype("int32")
                        encoding["time"] = {"dtype": "int32"}

                    # Update dataset with the converted time values and attributes
                    ds_to_save = ds_to_save.assign_coords(time=("time", time_sec))
                    ds_to_save.time.attrs.update({
                        "units": f"seconds since {epoch_str}",
                        "calendar": "standard",
                    })
                    print("Dataset updated for saving with time encoding.")

        else:
            print(
                f"Warning: Time coordinate has an unexpected dtype {time_var.dtype}. Saving without specific time encoding."
            )
            # Remove time encoding if dtype is unexpected
            if "time" in encoding:
                del encoding["time"]

    # Filter encoding to only include allowed parameters for the chosen engine/format
    backend = kwargs.get("engine", "netcdf4")  # Default to netcdf4 if not specified
    if format == "NETCDF4_CLASSIC":
        backend = "netcdf4"  # NETCDF4_CLASSIC format uses netcdf4 engine implicitly

    allowed_enc = set()
    if backend == "netcdf4":
        allowed_enc = {
            "blosc_shuffle",
            "chunksizes",
            "quantize_mode",
            "significant_digits",
            "zlib",
            "endian",
            "contiguous",
            "_FillValue",
            "szip_coding",
            "fletcher32",
            "shuffle",
            "szip_pixels_per_block",
            "compression",
            "least_significant_digit",
            "dtype",
            "complevel",
            "units",
            "calendar",  # Added units and calendar
        }
    elif backend == "h5netcdf":
        allowed_enc = {
            "blosc_shuffle",
            "compression_opts",
            "chunksizes",
            "quantize_mode",
            "significant_digits",
            "zlib",
            "endian",
            "contiguous",
            "_FillValue",
            "szip_coding",
            "fletcher32",
            "shuffle",
            "szip_pixels_per_block",
            "compression",
            "dtype",
            "complevel",  # Removed units and calendar
        }
    else:  # For other engines, be more lenient or specific as needed
        print(f"Warning: Unknown backend '{backend}'. Using a broad set of allowed encoding parameters.")
        allowed_enc = {
            "blosc_shuffle",
            "compression_opts",
            "chunksizes",
            "quantize_mode",
            "significant_digits",
            "zlib",
            "endian",
            "contiguous",
            "_FillValue",
            "szip_coding",
            "fletcher32",
            "shuffle",
            "szip_pixels_per_block",
            "compression",
            "dtype",
            "complevel",
            "units",
            "calendar",
            "chunks",
        }

    encoding = {
        var_name: {k: v for k, v in enc.items() if k in allowed_enc} for var_name, enc in encoding.items()
    }

    print(f"Saving dataset to {file_path} with encoding: {encoding}, format: {format}, kwargs: {kwargs}")

    try:
        ds_to_save.to_netcdf(file_path, encoding=encoding, format=format, **kwargs)
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S}: Saved {file_path.parent}/{file_path.name}")
    except Exception as e:
        print(f"Error saving dataset to {file_path} with encoding {encoding}, format {format}, {kwargs}: {e}")
        # Attempt to save without any encoding if an error occurs
        print("Attempting to save without any encoding due to previous error.")
        try:
            ds_to_save.to_netcdf(file_path, format=format, **kwargs)
            print(
                f"{datetime.now():%Y-%m-%d %H:%M:%S}: Saved {file_path.parent}/{file_path.name} without encoding."
            )
        except Exception as e_no_enc:
            print(f"Error saving dataset to {file_path} even without encoding: {e_no_enc}")
            raise  # Re-raise the original error if saving without encoding also fails


def save_dataset_with_time_encoding_bad(ds: xr.Dataset, file_path: Path, epoch: str = "1970-01-01 00:00:00"):
    """
    Saves an xarray Dataset to a NetCDF file with specific encoding for the time variable
    to handle potential issues with int64 time values.

    The time coordinate is converted to seconds since a specified epoch and saved as int32.

    :param ds: The xarray Dataset to save.
    :param file_path: The path to save the NetCDF file.
    :param epoch: The reference epoch as a string (e.g., '1970-01-01 00:00:00').
    """
    encoding = {}
    if "time" in ds.coords and pd.api.types.is_datetime64_any_dtype(ds["time"].dtype):
        # Define the epoch datetime object
        epoch_dt = pd.to_datetime(epoch)

        # Create the units string dynamically
        units_string = f"seconds since {epoch_dt:%Y-%m-%d %H:%M:%S}"

        # Specify encoding for the time variable
        encoding["time"] = {"units": units_string, "dtype": "int32", "calendar": "standard"}

    # Save the dataset with the specified encoding
    ds.to_netcdf(file_path, format="NETCDF4_CLASSIC", encoding=encoding)


def load_and_join_ncep_local(
    ncep_path: Path, b_save_standard_time: bool = True
) -> Tuple[xr.Dataset | None, Path | None]:
    """
    Loads data from a local NetCDF file or joins data from multiple NetCDF files
    in a directory, and optionally saves the combined dataset to a single file
    in the parent directory with standard time encoding.

    :param ncep_path: Path to a single NetCDF file or a directory containing split NCEP CFSv2 NetCDF files.
    :param b_save_standard_time: If True, convert time to seconds since epoch as int32 before saving.
    :return: Tuple of (dataset, output_file_path).
        - dataset: The loaded and/or joined xarray Dataset, or None if no data was loaded.
        - output_file_path: The path to the saved NetCDF file (if saved), or None.
    """
    out_file_path = None
    ds = None

    if not ncep_path.exists():
        print(f"Error: Input path does not exist: {ncep_path}")
        return None, None

    print(
        "loading and converting",
        f"{'joining data from directory' if ncep_path.is_dir() else ''} {ncep_path}...",
    )
    ds = load_ncep_cfs_local(ncep_path)
    if ds is None:
        print(f"No data found in directory: {ncep_path}")
        return None, None

    # Get scalar values for lat, lon, start, and end
    coord_names = ("lat", "lon")
    lat_lon = [
        (
            (ds[c].values[0] if bool(ds[c].shape) else ds[c].values).item() if ds[c].size == 1 else
            ds[c].values[[0, -1]]
        ) if c in ds.coords else np.nan for c in coord_names
    ]
    if "time" in ds.coords and ds["time"].size > 0:
        start_end = [pd.to_datetime(v).strftime("%y%m%d") for v in ds["time"][[0, -1]].values]
    else:
        start_end = ["unknown", "start"]
    out_file_path, out_dir_created = output_file_dir_path(
        ncep_path.parent,  # Save the joined file in the parent directory of the split files
        *lat_lon,
        *start_end,
        [
            k.replace("_GRD_L103", "").replace("wnd10m", "")
            for k in ds.variables.keys()
            if k not in coord_names and k != "time"
        ],  # Updated variable name replacement
    )
    try:
        if b_save_standard_time:
            print("Saving data with converted time to NetCDF4 {}".format(out_file_path), end="...")
            save_dataset_with_time_encoding(ds, out_file_path)
        else:
            print("Saving data to NetCDF4 {}".format(out_file_path), end="...")
            save_dataset_with_time_encoding(ds, out_file_path, epoch=None)
        print("ok")
    except Exception as save_e:
        print(f"\nError load_and_join_ncep_local() saving data to {out_file_path}: {save_e}")
        out_file_path = None  # Indicate saving failed
    return ds, out_file_path


def select_nearest_grid_points(ds: xr.Dataset, lat: float, lon: float, num_points: int = 1) -> xr.Dataset:
    """
    Selects the nearest grid points (1 or 4) to the given latitude and longitude.

    :param ds: The xarray Dataset.
    :param lat: The target latitude.
    :param lon: The target longitude.
    :param num_points: The number of nearest points to select (1 or 4).
    :return: An xarray Dataset with the selected grid points.
    :raises ValueError: If num_points is not 1 or 4, or if lat/lon coordinates are not found.
    """
    if num_points not in [1, 4]:
        raise ValueError("num_points must be 1 or 4.")

    # Define coordinate names and target values in lists
    coord_names_options = [["lat", "latitude"], ["lon", "longitude"]]
    target_coords = [lat, lon]
    coord_names = [None, None]  # To store the actual coordinate names found in ds

    # Find the actual coordinate names in the dataset
    for i, name_options in enumerate(coord_names_options):
        for name_option in name_options:
            if name_option in ds.coords:
                coord_names[i] = name_option
                break

    if None in coord_names:
        raise ValueError("Latitude or longitude coordinates not found in the dataset.")

    if num_points == 1:
        # Select the single nearest point using a dictionary comprehension
        selection = {coord_name: target_coord for coord_name, target_coord in zip(coord_names, target_coords)}
        return ds.sel(selection, method="nearest", tolerance=0.5)
    else:  # num_points == 4
        # Find the indices of the nearest latitude and longitude
        coords_data = [ds[coord_names[0]].values, ds[coord_names[1]].values]
        nearest_indices = [np.abs(coords_data[i] - target_coords[i]).argmin() for i in range(2)]

        # Determine the indices for the 2x2 grid using lists and zip
        indices_to_sel = [[], []]  # [[lat_indices], [lon_indices]]

        for i, (nearest_idx, coords) in enumerate(zip(nearest_indices, coords_data)):
            # Select the nearest coordinate and potentially the next one to bracket the target
            if nearest_idx > 0 and coords[nearest_idx - 1] < target_coords[i]:
                indices_to_sel[i].append(nearest_idx - 1)
            indices_to_sel[i].append(nearest_idx)
            if nearest_idx < len(coords) - 1 and coords[nearest_idx + 1] > target_coords[i]:
                indices_to_sel[i].append(nearest_idx + 1)

            # Ensure we have at least two unique indices, taking the first two if more are found
            indices_to_sel[i] = sorted(list(set(indices_to_sel[i])))[:2]

        # Check if we have exactly two indices for both latitude and longitude
        if len(indices_to_sel[0]) == 2 and len(indices_to_sel[1]) == 2:
            # Create the selection dictionary using a comprehension and zip
            selection = {coord_name: indices for coord_name, indices in zip(coord_names, indices_to_sel)}
            return ds.isel(selection)
        else:
            raise ValueError(
                f"Could not select a 2x2 grid around ({lat}, {lon}). Selected {len(indices_to_sel[0])} latitudes and {len(indices_to_sel[1])} longitudes."
            )


def download_cfs_reanalysis(
    lat: float,
    lon: float,
    date_range: Sequence[str],
    vars_list: list = ["wnd10mu", "wnd10mv"],  # Updated variable names
    out: Optional[Path] = None,
    xr_backend="netcdf4",  # Note: "netCDF4" must be installed
    base_url: str = "http://apdrc.soest.hawaii.edu:80/dods/public_data/CFSv2/hourly_timeseries_analysis",
) -> xr.Dataset:
    """
    Download 10m wind vector components from CFSv2 reanalysis via OPeNDAP from APDRC,
    automating URL construction for the given date range.

    Saves the dataset as loaded without additional time conversion.

    :param lat: latitude of the point
    :param lon: longitude of the point
    :param date_range: Sequence of two strings, start and end dates in YYYY-MM-DD.
    :param vars_list: list of variables to download: default - u/v wind at 10m.
    :param out: output NetCDF filename (optional) to save locally. If it is a directory, then
    automatically name the output file and check for existing matching files.
    :param base_url: by default - APDRC CFSv2 hourly timeseries analysis OPeNDAP URL
    :return: Xarray Dataset containing the filtered NCEP CFSv2 wind data.
    """
    print(f"Starting download_cfs_reanalysis for {lat}, {lon}, dates: {date_range}")

    # Determine the output file path and create directory if necessary
    out_file, out_dir_created = output_file_dir_path(out, lat, lon, *date_range, vars_list)

    if out_dir_created:
        print(f"Output directory created: {out_dir_created}")

    # Check for existing file using the determined file path
    existing_ds = get_nc_file_content_if_matches(out_file, lat, lon, date_range, vars_list)
    if existing_ds is not None:
        print(f"Existing file found and matches parameters: {out_file}. Returning existing data.")
        return existing_ds

    # Construct URLs for variables
    urls = [f"{base_url}/{var}" for var in vars_list]
    print(f"Constructed URLs: {urls}")

    # Convert date strings to cftime.DatetimeGregorian objects for time subsetting logic
    start_date, end_date = [
        cftime.DatetimeGregorian(*map(int, date_str.split("-"))) for date_str in date_range[:2]
    ]
    print(f"Converted date range to cftime objects: {start_date} to {end_date}")

    print(f"Attempting to open datasets from {base_url}...")
    ds_subset = None  # Initialize ds_subset to None

    # Define preprocess function to handle time and space subsetting
    def preprocess(ds):
        """Subset both time AND space before full decoding."""
        print("Preprocessing dataset...")
        # 1. Time subsetting (numeric comparison)
        time_var = ds["time"]
        # Explicitly handle ambiguous reference date string
        units = time_var.attrs.get("units", "days since 0001-1-1 00:00:0.0")  # Pad with zeros
        calendar = time_var.attrs.get("calendar", "standard")
        print(f"  Units: {units}, Calendar: {calendar}")
        time_vals = time_var.values.astype(np.float64)
        start_days, end_days = [cftime.date2num(t, units, calendar=calendar) for t in [start_date, end_date]]
        time_mask = (time_vals >= start_days) & (time_vals <= end_days)
        print(f"  Time mask created. Selected {time_mask.sum()} out of {len(time_mask)} time steps.")

        # Apply time subsetting
        ds_time_subset = ds.isel(time=time_mask)

        # 2. Space selection (nearest neighbor or 4 nearest)
        print(f"  Selecting nearest point(s) for ({lat}, {lon})...")
        ds_sub = None  # Initialize ds_sub within preprocess

        try:
            # Use the refactored select_nearest_grid_points which uses the desired style
            ds_sub = select_nearest_grid_points(ds_time_subset, lat, lon, num_points=4)
            print(
                f"  Selected 4 nearest points in preprocess. Selected latitudes: {ds_sub['lat'].values}, longitudes: {ds_sub['lon'].values}"
            )
        except Exception as e:
            print(f"  Error selecting nearest grid points in preprocess: {e}")
            ds_sub = None  # Ensure ds_sub is None if spatial selection fails

        if ds_sub is None:
            print("Spatial selection failed in preprocess.")
            return None  # Return None if spatial selection fails

        # Keep time as numeric within preprocess, don't decode here.
        return ds_sub

    # Open and process all datasets using open_mfdataset with the preprocess function
    try:
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S}: Attempting open_mfdataset with preprocess...")
        ds_subset = xr.open_mfdataset(
            urls,
            engine=xr_backend,  # Use the passed xr_backend
            decode_times=False,  # Keep False to prevent automatic decoding before preprocess
            preprocess=preprocess,
            combine="by_coords",
            # parallel=True  # Enable parallel loading if multiple files - crashes Golab
        )
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S}: open_mfdataset with preprocess complete.")

        # Select the specified variables (already done in preprocess, but good to re-confirm)
        if ds_subset is not None:
            ds_subset = ds_subset[vars_list]

        print("Finished dataset processing.")

    except Exception as e:
        print(f"An error occurred during data download and processing: {e}")
        ds_subset = None  # Ensure ds_subset is None if any error occurs

    if out_file and ds_subset is not None:  # Check if ds_subset is not None before saving
        print(f"Saving processed data to {out_file}...")
        try:
            # Save the dataset keeping original units if time is datetime, which is indicated by
            # epoch=None or if time is numeric which should be if preprocess() has been used
            save_dataset_with_time_encoding(
                ds_subset,
                out_file,
                epoch=None,  # Let the function infer how to save time
                format=None,  # Let save_dataset_with_time_encoding decide format or use default
                # engine="h5netcdf"  hangs
            )
            print("Data saved successfully.")
        except Exception as save_e:
            print(f"Error saving data to {out_file}: {save_e}")

    print("download_cfs_reanalysis() finished.")
    return ds_subset


# %% Analyze

def calculate_mean_wind_speed(dataset: xr.Dataset) -> Optional[xr.DataArray]:
    """
    Calculates the mean wind speed from a dataset containing 'u' and 'v' wind components.

    Assumes 'u' and 'v' are the variable names for the zonal and meridional wind components.
    Calculates wind speed as sqrt(u^2 + v^2).

    :param dataset: Xarray Dataset containing 'u' and 'v' wind components.
    :return: Xarray DataArray containing the calculated mean wind speed over all dimensions, or None if required variables are missing.
    """
    # Calculate wind speed from u and v components
    # Check for common wind variable names
    u_var = None
    v_var = None
    wind_vars_options = [('u', 'v'), ('U_GRD_L103', 'V_GRD_L103')]

    for u_name, v_name in wind_vars_options:
        if u_name in dataset.variables and v_name in dataset.variables:
            u_var = u_name
            v_var = v_name
            break # Found a matching pair

    if u_var and v_var:
        wind_speed = (dataset[u_var]**2 + dataset[v_var]**2)**0.5
        # Calculate the mean wind speed over all dimensions
        mean_speed = wind_speed.mean()
        return mean_speed
    else:
        print("Warning: Dataset does not contain required wind component variables (u/v or U_GRD_L103/V_GRD_L103) for wind speed calculation.")
    return None


def compare_mean_wind_speeds(era5_mean_speed: xr.DataArray, ncep_mean_speed: xr.DataArray):
    """
    Compares the mean wind speeds from two different datasets.

    :param era5_mean_speed: Mean wind speed from ERA5 data.
    :param ncep_mean_speed: Mean wind speed from NCEP CFSv2 data.
    """
    print(f"Mean wind speed (ERA5): {era5_mean_speed.values} m/s")
    print(f"Mean wind speed (NCEP CFSv2): {ncep_mean_speed.values} m/s")

    # Calculate the difference
    difference = era5_mean_speed - ncep_mean_speed
    print(f"Difference (ERA5 - NCEP CFSv2): {difference.values} m/s")


if __name__ == "__main__":
    # Main constants and settings

    # Set:
    # - directory of meteo data
    # - lat_st, lon_st
    # - use_date_range - Last day will be included.
    # Old: If previous text data exist you can net [] to load from last loaded data to now
    project_output_dir_base, lat_st, lon_st, use_date_range = (
        r"B:\WorkData\BalticSea\230825_Kulikovo@ADCP,ADV,i,tr\meteo",
        54.99, 20.3, ["2023-08-25", "2023-09-10"],  # 54.9896, 20.29972
    )

    try:  # Google Colab + Drive Specific
        from google.colab import drive, userdata
        project_relative_dir = "Oceano/data"

        mount_path = Path("/content/drive")  # commonly used Google Colab system mount path
        google_drive_base_dir = mount_path / "MyDrive"  # /Colab Notebooks

        # Attempt to import and mount Google Drive and set project relative to google_drive_base_dir
        # Check if the mount point already exists and is not empty
        if mount_path.exists and any(mount_path.glob("*")):
            # drive.flush_and_unmount()

            # # Rename the existing directory to avoid overwriting
            # os.rename(mount_path, '/content/drive_old/')
            # print(f"Existing '{mount_path}' renamed to '/content/drive_old/'")
            # !rm -rf /content/drive_old
            pass
        else:
            drive.mount(str(mount_path))  # mounts here my Gogle disk as "MyDrive"
        # Construct the full path for the output directory within Google Drive
        project_output_dir_base = google_drive_base_dir / project_relative_dir
        print(f"Google Drive mounted. Base project output directory: {project_output_dir_base}")
    except ImportError:  # Fallback to a local directory if not in Colab or Drive mount fails
        print("Google Colab modules not found. Assuming local execution.")
        print("in the Colab environment output will be saved locally and may be lost.")
        project_output_dir_base = Path(project_output_dir_base)   # Original local path structure
    except Exception as e:
        print(f"Error mounting Google Drive: {e} as {mount_path}. Try again?")
        raise e
    project_output_dir_base.mkdir(exist_ok=True)


    b_test = False  # True
    if not b_test:
        # --- User Input ---
        ncep_path = project_output_dir_base / "NCEP_CFSv2" / "splitted"
    else:
        print("Starting test download with minimal data...")
        # Use a very short date range for the test
        test_date_range = ["2023-08-25", "2023-08-26"] # Just two days

        # Define a test output path, perhaps in a temporary directory or a 'test' subfolder
        ncep_path = project_output_dir_base / "NCEP_CFSv2" / "test" / "test_wind_uv10.nc"

    if ncep_path:
        if False:
            # Load NCEP CFSv2 data for the specified coordinates
            ncep_speed = download_cfs_reanalysis(
                lat=lat_st, lon=lon_st, date_range=use_date_range,
                out=ncep_path  # "wind_uv10.nc"
            )
            if ncep_speed is not None:
                print("NCEP CFSv2 data loaded successfully:")
                print(ncep_speed)
            else:
                print("Failed to load NCEP CFSv2 data.")

        # saving with correct format
        ds, out_file_path = load_and_join_ncep_local(ncep_path)

        path_interp = interp_to_point(out_file_path, lat_st, lon_st)
        print(path_interp, "saved")
