
import logging
import math
from pathlib import Path
import re
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    Iterable,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    List,
    Union,
    TypeVar,
)
import xml.etree.ElementTree as ET
import numpy as np
import xarray as xr
from zipfile import ZipFile
from io import IOBase
import os
from tempfile import NamedTemporaryFile



l = logging.getLogger(__name__)

def safe_netcdf_atomic(ds, path: Path, format="NETCDF4_CLASSIC", engine="netcdf4", **kwargs) -> None:
    """Safely safe a NetCDF file using atomic overwrite.
    :param format: Default: "NETCDF4_CLASSIC" ensures compatibility of output data with Veusz import HDF5
    format
    :param kwargs: other `to_netcdf()` parameters dict
    """

    # temp file in the same dir as original (prevents WinError 17)
    with NamedTemporaryFile(prefix=path.stem, suffix=".nc", dir=os.path.dirname(path), delete=False) as tmp:
        tmp_path = tmp.name
        ds.to_netcdf(tmp_path, mode="w", format=format, engine=engine, **kwargs)
    os.replace(tmp_path, path)  # atomic overwrite, same disk only
    # os.remove(tmp_path)  # replace removes

def is_angular(var: xr.DataArray) -> bool:
    """
    Проверяет, является ли переменная угловой (по её атрибуту units).
    """
    units = var.attrs.get("units", "").lower()
    return any(x in units for x in ["degree", "degrees_east", "degrees_north"])


def interp_angle(da, new_coords, method="linear"):
    """
    :param da: исходный DataArray с угловыми значениями в градусах
    :param new_coords: {name: val}
    - name: имя координаты, по которой проводится интерполяция (например, 'time' или 'longitude').
    - val: новые значения координаты, на которые необходимо интерполировать данные
    :param method: метод интерполяции
    :return: _description_
    """
    # Check if all variables in the dataset have angular units
    for var in da.data_vars:
        units = da[var].attrs.get("units", "").lower()
        if not any(x in units for x in ["degree", "degrees_east", "degrees_north"]):
            l.warning(f"Variable {var} does not have angular units ({units}), skipping angular interpolation")
            return da  # Return the original data if not all variables are angular
    # Only proceed if there are variables to interpolate and they are all angular
    if len(da.data_vars) == 0:
        # If no variables, return as is
        return da
    # Преобразуем углы в радианы
    radians = np.deg2rad(da)
    # Представляем как комплексные числа на единичной окружности
    complex_repr = xr.apply_ufunc(np.exp, 1j * radians, dask="allowed")
    # Интерполируем действительную и мнимую части отдельно (xarray и scipy не поддерживают интерполяцию по комплексным значениям)
    real_interp = complex_repr.real.interp(new_coords, method=method)
    imag_interp = complex_repr.imag.interp(new_coords, method=method)
    # Восстанавливаем комплексные числа
    complex_interp = xr.apply_ufunc(lambda x, y: x + 1j * y, real_interp, imag_interp, dask="allowed")
    # Вычисляем угол и преобразуем обратно в градусы
    angle_interp = xr.apply_ufunc(np.angle, complex_interp, dask="allowed")
    degrees = np.rad2deg(angle_interp) % 360
    return degrees


def interp_to_point(path_loaded: str, lat: float, lon: float, backend="h5netcdf") -> None:
    """
    Interpolate the data to the point (lat, lon) and save to "NETCDF4_CLASSIC" format file having same
    NetCDF settings (supported subset) under same name in which old coordinates replaced to
    "-to_{new coordinates}"
    :param path_loaded: str or Path object of the original NetCDF file.
    """
    # открыть файл
    path_loaded = Path(path_loaded)

    # check data
    def is_between(p, less, bigger):
        return (less < p) & (p < bigger)

    # possible coordinate names
    coord_names_options = [["lat", "latitude"], ["lon", "longitude"]]

    target_coords = [lat, lon]

    new_coords: Dict[str, float] = {}
    with xr.open_dataset(path_loaded, engine=backend) as ds:
        # Creating dict with the actual coordinate names in the dataset
        for i, name_options in enumerate(coord_names_options):
            found_name = None
            for name_option in name_options:
                if name_option in ds.coords:
                    found_name = name_option
                    break

            if found_name is None:
                raise ValueError("Latitude or longitude coordinates not found in the dataset.")
            new_coords[found_name] = target_coords[i]
        for k, v in new_coords.items():
            ds = ds.sortby(k)
            assert is_between(v, *ds[k].values[[0, -1]]), f"{k}: {v} is not between {ds[k].values}"

        # Separate variables into angular and others
        angular_vars = [v for v in ds.data_vars if is_angular(ds[v])]
        other_vars = [v for v in ds.data_vars if v not in angular_vars]

        ds_to_merge = []
        # Interpolate 'others' normally
        if any(other_vars):
            ds_to_merge.append(ds[other_vars].interp(**new_coords, method="linear"))

        # Interpolate 'angular' variables
        if any(angular_vars):
            ds_to_merge.append(interp_angle(ds[angular_vars], new_coords, method="linear"))

    # Merge the results
    if not ds_to_merge:
        raise ValueError("No variables found to interpolate or merge.")
    ds_interp = xr.merge(ds_to_merge)

    # Save
    add_str = f"-to_{lon:.6g}E_{lat:.6g}N"
    path_new_stem, n_rep = re.subn(
        r"_(\d{1,3}\.\d+[EN][_-]){2,4}",
        f"{add_str}_",
        path_loaded.stem,
    )
    if not n_rep:
        path_new_stem = f"{path_loaded.stem}{add_str}"

    path_new = path_loaded.with_name(f"{path_new_stem}.nc")
    # not supported keys in selected output format
    not_classic = {"szip", "zstd", "bzip2", "blosc", "preferred_chunks", "coordinates"}
    not_classic.add("chunksizes")  # can not exceed dimensions
    encoding = {
        var_name: {k: v for k, v in var.encoding.items() if k not in not_classic}
        for var_name, var in ds.variables.items()
    }
    # Attempt to save using NETCDF4_CLASSIC format with careful encoding
    try:

        safe_netcdf_atomic(ds_interp, path_new, encoding=encoding)
        return path_new
    except ValueError as e:
        l.exception("Bad encoding parameters or other ValueError during save. Falling back...")
        # Fallback to safe_netcdf_atomic if to_netcdf fails
    except Exception as e:
        l.exception(f"An unexpected error occurred during saving: {e}")
        # Final fallback for any other unexpected error
    safe_netcdf_atomic(ds_interp, path_new)
    l.info(f"interpolated data saved to: {path_new}")
    # to_csv(path_new.with_suffix(".csv"))
    return path_new


def extract_zip_to_named_dir(zip_path: str | Path, target_dir=None, dry_run=False) -> Path:
    """
    Extracts all files from the ZIP archive into a directory named after the archive itself.

    :param zip_path: path to the .zip file
    :return: path to the directory where contents were extracted
    """
    zip_path = Path(zip_path).resolve()
    if target_dir is None:
        target_dir = zip_path.with_suffix("")  # remove .zip

    if not dry_run:
        with ZipFile(zip_path) as zf:
            zf.extractall(target_dir)

    return target_dir


def h5_format(file: Union[str, Path, List[Union[str, Path]]], backend="h5netcdf", **meta: Mapping[str, Any]):
    """
    Format and add metadata to HDF5/NetCDF files. If input file is .grib file then writes to .nc file, if
    input file is .nc, then overwrites input file

    :param file: NetCDF or .grib file path or list of such files paths
    :param **meta: {attribute_name: value} metadata to add to netcdf file (i.e. to dataset)
    :param backend: _description_, defaults to "h5netcdf"
    """
    files = [Path(file)] if isinstance(file, (Path, str)) else [Path(f) for f in file]
    done = False
    for file_path in files:
        grib_to_netcdf = file_path.suffix == ".grib"
        with xr.open_dataset(file_path, engine="cfgrib" if grib_to_netcdf else backend) as ds:  # decode_cf=False
            try:
                lat = ds.latitude.values
                lon = ds.longitude.values
                print(
                    "Downloaded grid centers: ",
                    ", ".join([f"{_:.5f}" for _ in lat]),
                    "°N; ",
                    ", ".join([f"{_:.5f}" for _ in lon]),
                    "°E. ",
                    sep="",
                    end="",
                )
            except Exception as e:
                print(f"not found existed lat/lon in {file}?", e)

            for k, v in meta.items():
                if ds.attrs.get(k) == v:
                    print("-", end="")
                    continue
                print(".", end="")
                done = True
                ds.attrs[k] = v

        if done or grib_to_netcdf:
            try:
                safe_netcdf_atomic(
                    ds.sel(latitude=lat, longitude=lon) if grib_to_netcdf else ds,
                    file_path.with_suffix(".nc")
                )
            except PermissionError as e:
                print(f"{file_path.with_suffix('.nc').name}:", "Permission denied")
                done = False
            except Exception as e:
                print(f"{file_path.with_suffix('.nc').name}:", e)
                done = False
    print(f"Attributes: {meta} saved" if done else "no attributes saved")


def grid_aligned_bbox(
    lat: float, lon: float, delta: float = 0.25, extend: float = 0
) -> Tuple[float, float, float, float]:
    """
    Generate ECMWF-style area bounding box aligned to ERA5 grid.

    :param lat: center latitude in degrees (-90 to 90)
    :param lon: center longitude in degrees (-180 to 180 or 0 to 360)
    :param delta: grid resolution (default = 0.25 for ERA5)
    :param extend: продлевает диагональ
    :return: tuple (north, west, south, east), each aligned to grid
    """
    # ensure lon in [0, 360)
    lon = lon % 360

    # align lat and lon to nearest lower grid point
    lat0 = (math.floor if lat < 0 else int)(lat / delta) * delta
    lon0 = int(lon / delta) * delta

    # create bounding box with +1 grid cell in both directions
    north = lat0 + delta + extend
    south = lat0
    west = lon0
    east = lon0 + delta + extend

    # ECMWF expects [N, W, S, E] with descending latitude
    return round(north, 5), round(west, 5), round(south, 5), round(east, 5)








class ReverseTxt(IOBase):
    """
    Edited source from https://stackoverflow.com/a/51750850/2028147
    An example
    rev = ReverseTxt(filename)
    for i, line in enumerate(rev):
        print("{0}: {1}".format(i, line.strip()))
    """

    def __init__(self, filename, headers=0, **kwargs):
        """

        :param filename:
        :param headers:
        :param kwargs: args to call open(filename, **kwargs)
        """
        self.fp = open(filename, **kwargs)
        self.headers = headers
        self.reverse = self.reversed_lines()
        self.end_position = -1
        self.current_position = -1

    def readline(self, size=-1):
        if self.headers > 0:
            self.headers -= 1
            raw = self.fp.readline(size)
            self.end_position = self.fp.tell()
            return raw

        raw = next(self.reverse)
        if self.current_position > self.end_position:
            return raw

        raise StopIteration

    def reversed_lines(self):
        """Generate the lines of file in reverse order."""
        part = ""
        for block in self.reversed_blocks():
            block = block + part
            block = block.split("\n")
            block.reverse()
            part = block.pop()
            if block[0] == "":
                block.pop(0)

            for line in block:
                yield line + "\n"

        if part:
            yield part

    def reversed_blocks(self, blocksize=0xFFFF):
        """Generate blocks of file's contents in reverse order."""
        file = self.fp
        file.seek(0, os.SEEK_END)
        here = file.tell()
        while 0 < here:
            delta = min(blocksize, here)
            here -= delta
            file.seek(here, os.SEEK_SET)
            self.current_position = file.tell()
            yield file.read(delta)


def extract_coordinates_from_gpx(gpx_path: Path, waypoints_re: str = None) -> Optional[Dict[str, Dict[str, float]]]:
    """Extract coordinates from a .gpx file.

    Args:
        gpx_path: Path to .gpx file
        waypoints_re: Regular expression to filter waypoints by name

    Returns:
        Dict of {waypoint_name: {"lat": lat, "lon": lon}} or None if no coordinates found
    """
    l.info(f"Extracting coordinates from {gpx_path}")
    # Define namespace
    ns = {"gpx": "http://www.topografix.com/GPX/1/1"}

    points = {}
    try:
        tree = ET.parse(gpx_path)
        root = tree.getroot()

        # Try to find waypoints first
        waypoints = root.findall(".//gpx:wpt", ns)
        for waypoint in waypoints:
            # Get the name of the waypoint if it exists
            name_elem = waypoint.find("gpx:name", ns)
            waypoint_name = name_elem.text if name_elem is not None else f"waypoint_{len(points)}"

            # If waypoints_re is provided, check if the waypoint name matches the pattern
            if waypoints_re and not re.match(waypoints_re, waypoint_name):
                continue
            lat = float(waypoint.get("lat"))
            lon = float(waypoint.get("lon"))
            points[waypoint_name] = {"lat": lat, "lon": lon}
            l.debug(f"Found coordinates in waypoint {waypoint_name}: lat={lat}, lon={lon}")

        # Check if any coordinates were found
        if not points:
            l.warning(f"No waypoints found in {gpx_path} matching pattern {waypoints_re}")
            return None

        l.debug(f"Successfully read {len(points)} waypoints from {gpx_path}")
        return points
    except Exception as e:
        l.error(f"Error reading {gpx_path}: {e}")
        return None
