
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
import numpy as np
import xarray as xr
from zipfile import ZipFile
from io import IOBase
import os
from tempfile import NamedTemporaryFile

l = logging.getLogger(__name__)

def safe_netcdf_atomic(ds, path: Path) -> None:
    """Safely safe a NetCDF file using atomic overwrite."""

    # temp file in the same dir as original (prevents WinError 17)
    with NamedTemporaryFile(prefix=path.stem, suffix=".nc", dir=os.path.dirname(path), delete=False) as tmp:
        tmp_path = tmp.name
        ds.to_netcdf(tmp_path, mode="w")
    # try:
    os.replace(tmp_path, path)  # atomic overwrite, same disk only
    os.remove(tmp_path)
    # except Exception as e:
    #     print

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


def h5_format(file, backend="h5netcdf", **meta: Mapping[str, any]):
    files = [file] if isinstance(file, (Path, str)) else file
    done = False
    for file in files:
        grib_to_netcdf = file.suffix == ".grib"
        with xr.open_dataset(file, engine="cfgrib" if grib_to_netcdf else backend) as ds:  # decode_cf=False
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
                    file.with_suffix(".nc")
                )
            except PermissionError as e:
                print(f"{file.with_suffix('.nc').name}:", "Permission denied")
                done = False
            except Exception as e:
                print(f"{file.with_suffix('.nc').name}:", e)
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
    assert all([da[var].attrs.get("units", "").lower().startswith("degree") for var in da.data_vars])
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
    """
    # открыть файл
    path_loaded = Path(path_loaded)

    # check data
    def is_between(p, less, bigger):
        return (less < p) & (p < bigger)

    with xr.open_dataset(path_loaded, engine=backend) as ds:
        # # Creating dict with correct names like {'latitude': [lat], 'longitude': [lon]}

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

        new_coords = {k: [v] for k, v in zip(coord_names, target_coords)}
        for k, v in zip(coord_names, target_coords):
            ds = ds.sortby(k)
            assert is_between(v, *ds[k].values[[0, -1]]), f"{k}: {v} is not between {ds[k].values}"


        # разделяем переменные на угловые и остальные
        angular_vars = [v for v in ds.data_vars if is_angular(ds[v])]
        other_vars   = [v for v in ds.data_vars if v not in angular_vars]

        ds_to_merge = []
        # интерполируем «остальные» стандартно
        if any(other_vars):
            ds_to_merge.append(ds[other_vars].interp(**new_coords, method="linear"))

        # интерполируем «угловые»
        if any(angular_vars):
            ds_to_merge.append(interp_angle(ds[angular_vars], new_coords, method="linear"))

        # kwargs_period = {"period": 360}  # чтобы xarray передал этот параметр в numpy.interp
        # ds[angular_vars].interp(**new_coords, method="linear", kwargs=kwargs_period)

    # объединяем результаты
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
    try:
        ds_interp.to_netcdf(path_new, format="NETCDF4_CLASSIC", engine="netcdf4", encoding=encoding)
    except ValueError as e:
        l.exception("Bad encoding parameters? - removing encoding")
        ds_interp.to_netcdf(path_new, format="NETCDF4_CLASSIC", engine="netcdf4")
    l.info(f"interpolated data saved to: {path_new}")
    # to_csv(path_new.with_suffix(".csv"))
    return path_new


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
