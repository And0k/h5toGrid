#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate wave parameters from pressure sensor data.

This program:
1. Calculates spectrograms using spectr_clc.main()
2. Loads spectrograms and calculates wave parameters for each spectrum
3. Calculates H1/3 from time domain for quality control
4. Saves spectrograms and wave parameters to NetCDF files

Author: Wave Parameters Processing
Created: 2025
"""
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple
import copy
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4

import tcm.spectr_clc as spectr_clc
import tcm.wave_params_from_pres as wave_params_from_pres
from utils.init import LoggingStyleAdapter
from hdf5_pandas import h5

# Initialize logger with str.format() style support
lf = LoggingStyleAdapter(__name__)


# Shared spectral analysis parameters (single source of truth)
spectral_analysis = {
    # Frequency range for spectral analysis (Hz)
    "fmin": 0.04,
    "fmax": 0.5,
    # Length of each interval in minutes (duration for each spectral calculation)
    "dt_interval": np.timedelta64(60, 'm'),
    # Overlap ratio between consecutive intervals [0, 1)
    # 0 = no overlap, 0.5 = 50% overlap
    "overlap": 0,  # data is in 15 minute burst at each hour (0.5 overlap will produce same data twice)
}

# Configuration dictionary that replicates spectr_clc configuration structure
# This cfg is used throughout the code for all configuration parameters
cfg = {
    "spectr_clc": {
        "in": {
            "db_path": "",  # Will be overridden by command line or main function
            "tables": ["w.*"],  # Wave gauge tables pattern (list for h5_names_gen)
            "chunksize": 50000,
            "min_date": datetime.fromisoformat("2019-01-01T00:00:00"),
            "min_Pressure": -1e15,
            "max_Pressure": None,
            "fs": None,
            "dt_hole_warning": np.timedelta64(60, "m").item(),
        },
        "filter": {
            "min_dict": None,
            "max_dict": None,
        },
        "out": {
            "db_path": "",  # Will be overridden by command line or main function
            "table": "psd",
        },
        "proc": {
            "time_intervals_center_list": None,
            "calc_version": "trigonometric(incl)",
            "max_incl_of_fit_deg": None,
            # Reference shared spectral parameters
            **spectral_analysis,
        },
        "program": {
            "return": "<end>",
            "b_interact": 0,  # Non-interactive mode
            "verbose": "INFO",
            "log": None,
        },
    },
    "wave_params": {
        # Reference shared spectral parameters
        **spectral_analysis,
        # Sensor height above bottom in meters
        "sensor_height": 2.0,
        # Sea depths for each wave gauge (depth at sensor position)
        # Based on task description: depths 10, 4, 4, 4, 4 for w01, w02, w04, w05, w06
        "sea_depths": {
            "w01": 10.0,
            "w02": 4.0,
            "w04": 4.0,
            "w05": 4.0,
            "w06": 4.0,
        },
        # Maximum number of missing spectra allowed before raising error
        "max_missing_spectra": 100,
    },
    "program": {
        "log": "log/wave_params_run.log",
        "verbose": "INFO",
    },
}


def _build_spectr_clc_config(
    cfg: Dict,
    h5_path: Path,
    psd_path: Path,
    tables: Optional[list] = None,
    columns: Optional[list] = None,
    table_stats: Optional[Dict[str, int]] = None,
) -> Dict:
    """
    Update configuration dictionary cfg to call spectr_clc.main()

    Args:
        h5_path: Path to input HDF5 store
        psd_path: Path to output NetCDF file with spectrograms
        cfg: configuration dictionary required for spectr_clc.main() that
        needs to be updated (see "spectr_clc" field of main config of this module)
        tables: Optional list of tables to process (overrides cfg["in"]["tables"])
        columns: Optional list of columns to load (defaults to ["Pressure"])
        table_stats: Optional statistics from check_spectra_existence() containing
            min_date and max_date to override cfg["in"] "min_date" and "max_date" fields

    Returns:
        Configuration dictionary for spectr_clc functions

    """
    spectr_cfg = copy.deepcopy(cfg)
    spectr_cfg["in"]["db_path"] = h5_path
    spectr_cfg["in"]["columns"] =  columns if columns is not None else ["Pressure"]
    if tables is not None:
        spectr_cfg["in"]["tables"] = tables
    spectr_cfg["out"]["db_path"] = psd_path

    # Use table-specific min_date and max_date from statistics if available
    # This prevents h5_velocity_by_intervals_gen() from generating empty intervals
    if table_stats is not None:
        for k in ["min_date", "max_date"]:
            if (d:=table_stats.get(k)) is not None:
                spectr_cfg["in"][k] = d
    return spectr_cfg


def get_sampling_frequency(h5_path: Path, table_name: str) -> float:
    """
    Determine sampling frequency from HDF5 table.

    Args:
        h5_path: Path to HDF5 store
        table_name: Name of table to check

    Returns:
        Sampling frequency in Hz
    """
    with pd.HDFStore(str(h5_path), mode="r") as store:
        df = store[table_name]
        if len(df) < 2:
            raise ValueError(
                f"Table {table_name} has insufficient data to determine sampling frequency"
            )
        # Calculate time difference between consecutive samples
        dt = (df.index[1] - df.index[0]).total_seconds()
        fs = 1.0 / dt
    return fs


def get_wave_tables(h5_path: Path, cfg: Dict) -> list:
    """
    Get list of wave gauge tables from source HDF5.

    Centralized function to avoid code duplication across check_spectra_existence(),
    calculate_wave_parameters(), and calculate_h13_time_domain().

    Args:
        h5_path: Path to HDF5 store
        cfg: Main configuration dictionary

    Returns:
        List of wave gauge table names
    """
    with pd.HDFStore(str(h5_path), mode="r") as store:
        wave_tables = h5.find_tables(store, cfg["spectr_clc"]["in"]["tables"])

    if not wave_tables:
        lf.warning("No wave gauge tables found in {}", h5_path)

    return wave_tables


def _validate_nc_table(nc_tbl, table_name: str) -> Tuple[bool, int]:
    """
    Validate NetCDF table structure and count existing spectra.

    Args:
        nc_tbl: NetCDF table group
        table_name: Table name for logging

    Returns:
        Tuple of (is_valid, existing_count)
    """
    if "time_start" not in nc_tbl.variables or "time_end" not in nc_tbl.variables:
        lf.warning("Missing time variables in NetCDF table: {}", table_name)
        return False, 0
    existing_starts = nc_tbl.variables["time_start"][:]
    return True, len(existing_starts)


def _count_missing_intervals(
    expected_intervals: list,
    existing_starts: np.ndarray,
    existing_ends: np.ndarray,
) -> int:
    """
    Count how many expected intervals are missing from existing spectra.

    Args:
        expected_intervals: List of (start, end) pairs for expected intervals
        existing_starts: Array of existing start times
        existing_ends: Array of existing end times

    Returns:
        Number of missing intervals
    """
    missing_count = 0
    for expected_start, expected_end in expected_intervals:
        expected_start_ns = expected_start.astype("datetime64[ns]").astype(np.int64)
        expected_end_ns = expected_end.astype("datetime64[ns]").astype(np.int64)

        found = any(
            abs(existing_start - expected_start_ns) < 1e9 and
            abs(existing_end - expected_end_ns) < 1e9
            for existing_start, existing_end in zip(existing_starts, existing_ends)
        )
        if not found:
            missing_count += 1
    return missing_count


def check_spectra_existence(
    psd_path: Path,
    h5_path: Path,
    cfg: Dict,
    fast_check: bool = False,
) -> Tuple[bool, Dict[str, Dict[str, int]]]:
    """
    Check if spectra already exist for all source tables and intervals.

    Compares existing spectra in NetCDF file with expected intervals based on
    source HDF5 tables and configuration parameters (dt_interval, overlap).
    Uses spectr_clc.h5_velocity_by_intervals_gen() to generate expected intervals
    to avoid duplicating interval generation logic.

    Args:
        psd_path: Path to NetCDF file with spectrograms
        h5_path: Path to input HDF5 store
        cfg: Configuration dictionary containing spectral analysis parameters
        fast_check: If True, only check tables and date ranges without detailed
            interval matching. If tables and date ranges are valid, calculate wave
            parameters and raise error if missing spectra exceed max_missing_spectra.

    Returns:
        Tuple of (all_spectra_exist, statistics):
            - all_spectra_exist: True if all expected spectra are present
            - statistics: Dictionary with statistics per table:
                {'table_name': {'total': int, 'existing': int, 'missing': int,
                'min_date': pd.Timestamp, 'max_date': pd.Timestamp}}
    """

    statistics = {}

    if not psd_path.exists():
        lf.info("Spectra file does not exist: {}", psd_path)
        return False, statistics

    wave_tables = get_wave_tables(h5_path, cfg)
    if not wave_tables:
        return False, statistics

    try:
        with netCDF4.Dataset(psd_path, mode="r") as nc_root:
            if "psd" not in nc_root.groups:
                lf.info("No 'psd' group found in NetCDF file")
                return False, statistics

            nc_psd = nc_root.groups["psd"]
            for table_name in wave_tables:
                tbl_normalized = table_name.replace("incl", "_i")

                # Get actual min_date and max_date from source HDF5 table
                with pd.HDFStore(str(h5_path), mode="r") as store:
                    statistics[tbl_normalized] = {
                        "min_date": store.select(table_name, columns=[], stop=1).index[0].tz_localize(None),
                        "max_date": store.select(table_name, columns=[], start=-1).index[0].tz_localize(None),
                        "total": 0,
                        "existing": 0,
                        "missing": 0,
                    }
                try:
                    nc_tbl = nc_psd.groups[tbl_normalized]
                except KeyError:
                    lf.info("No spectra found for table: {}", table_name)
                    continue
                is_valid, existing_count = _validate_nc_table(nc_tbl, table_name)
                if not is_valid:
                    statistics[tbl_normalized].update({"total": 0, "existing": 0, "missing": 0})
                    continue


                if fast_check:
                    statistics[tbl_normalized].update({
                        "total": existing_count,
                        "existing": existing_count,
                        "missing": 0
                    })
                    lf.info("Fast check: Table {} has {} spectra", table_name, existing_count)
                    continue

                spectr_cfg = _build_spectr_clc_config(
                    cfg["spectr_clc"], h5_path, psd_path, tables=[table_name],
                    table_stats=statistics[tbl_normalized]
                )
                existing_starts = nc_tbl.variables["time_start"][:]
                existing_ends = nc_tbl.variables["time_end"][:]
                expected_intervals = [
                    df_interval.index[[0, -1]].values
                    for df_interval, _, _ in spectr_clc.h5_velocity_by_intervals_gen(
                        cfg=spectr_cfg, cfg_out=spectr_cfg["out"]
                    )
                    if df_interval is not None and len(df_interval) > 0
                ]

                expected_count = len(expected_intervals)
                missing_count = _count_missing_intervals(
                    expected_intervals, existing_starts, existing_ends
                )

                statistics[tbl_normalized].update({
                    "total": expected_count,
                    "existing": existing_count,
                    "missing": missing_count,
                })

                if missing_count > 0:
                    lf.info(
                        "Table {}: {}/{} spectra missing",
                        table_name, missing_count, expected_count
                    )

    except Exception as e:
        lf.exception("Error checking spectra existence")
        return False, statistics

    if fast_check:
        lf.info("Fast check completed: tables and date ranges validated")
        return True, statistics

    all_spectra_exist = all(
        stats["missing"] == 0 for stats in statistics.values() if stats["total"] > 0
    )
    lf.info(
        "All spectra already exist" if all_spectra_exist else "Some spectra are missing"
    )
    return all_spectra_exist, statistics


def calculate_wave_parameters(
    h5_path: Path,
    psd_path: Path,
    fs: float,
    cfg: Dict,
    max_missing_spectra: int = 100,
    statistics: Optional[Dict[str, Dict[str, int]]] = None,
) -> Path:
    """
    Calculate wave parameters from spectrograms.

    Args:
        h5_path: Path to input HDF5 store
        psd_path: Path to NetCDF file with spectrograms
        fs: Sampling frequency in Hz
        cfg: Configuration dictionary
        max_missing_spectra: Maximum number of missing spectra allowed before
            raising an error (used when fast_check=True in check_spectra_existence).
        statistics: Optional statistics from check_spectra_existence().
            If provided, used to calculate total expected intervals without regenerating them.

    Returns:
        Path to output NetCDF file with wave parameters

    Raises:
        RuntimeError: If number of missing spectra exceeds max_missing_spectra
    """
    lf.info("Calculating wave parameters from spectrograms: {}", psd_path)

    wave_tables = get_wave_tables(h5_path, cfg)
    if not wave_tables:
        lf.warning("No wave gauge tables found in {}", h5_path)
        return None
    lf.info("Found {} wave gauge tables: {}", len(wave_tables), wave_tables)

    spectr_cfg = _build_spectr_clc_config(cfg["spectr_clc"], h5_path, psd_path)
    results = {}
    if statistics is None:
        statistics = {}
    for table_name in wave_tables:
        tbl_normalized = table_name.replace("incl", "_i")

        sea_depth = cfg["wave_params"]["sea_depths"].get(tbl_normalized)
        if sea_depth is None:
            lf.warning("No depth defined for table {}, skipping", table_name)
            continue

        lf.info("Processing table: {} (depth: {} m)", table_name, sea_depth)

        try:
            spectr_cfg = _build_spectr_clc_config(
                cfg["spectr_clc"], h5_path, psd_path, tables=[table_name],
                table_stats=statistics[tbl_normalized]
            )
            n_expected = statistics[tbl_normalized]["total"]
        except KeyError:
            # Build spectr_cfg with table-specific dates to avoid empty intervals
            n_expected = sum(
                1
                for df_interval, _, _ in spectr_clc.h5_velocity_by_intervals_gen(
                    cfg=spectr_cfg, cfg_out=spectr_cfg["out"]
                )
                if df_interval is not None and len(df_interval) > 0
            )

        group_path = f'psd/{tbl_normalized}'
        try:
            # Load table-specific data
            with netCDF4.Dataset(psd_path, format="NETCDF4") as nc_root:
                gr_spectrogram = nc_root[group_path]
                time = gr_spectrogram.variables["time"]
                time = netCDF4.num2date(time[:], units=time.units, calendar=time.calendar)
                psd_pres = gr_spectrogram.variables["Pressure"][:] * 1e8  # dBar²/Hz -> Pa²/Hz
                try:
                    freq = gr_spectrogram.variables["freq"][:]
                except KeyError:
                    # Load frequency from psd group level (not table-specific)
                    freq = nc_root["psd"].variables["freq"][:]
                    assert freq.size == psd_pres.shape[1], "NetCDF variables sizes check"

            # with xr.open_dataset(
            #     psd_path, group=group_path, decode_times={"time_good_range": False}
            # ) as ds_spectrogram:
            #     print(ds_spectrogram["time_good_range"])
            #     # time_start = ds_spectrogram["time_start"].values
            #     # time_end = ds_spectrogram["time_end"].values
            #     time = ds_spectrogram["time"].values

            n_existing = len(time)
            n_missing = n_expected - n_existing
            if n_missing > 0:
                if n_missing > max_missing_spectra:
                    raise RuntimeError(
                        f"Too many missing spectra: {n_missing} missing (max allowed: {max_missing_spectra}). "
                        f"Delete {psd_path} and restart to recalculate all spectra."
                    )

                lf.warning(
                    "{} spectra missing out of {} expected (within tolerance of {})",
                    n_missing, n_expected, max_missing_spectra
                )

        except OSError as e:
            lf.warning("No valid spectrogram found for table {}, skipping: {}", table_name, e)
            continue

        n_intervals = psd_pres.shape[0]
        hs_values = np.full(n_intervals, np.nan)
        tm_minus1_values = np.full(n_intervals, np.nan)

        transfer = wave_params_from_pres.pressure_response_correction(
            freq=freq,
            water_depth=sea_depth,
            sensor_height_above_bed=cfg["wave_params"]["sensor_height"],
        )
        for i, psd_interval in enumerate(psd_pres):
            psd_surface = wave_params_from_pres.compute_surface_elevation_spectrum(
                transfer=transfer,
                psd=psd_interval,
            )

            m0, m_minus1 = wave_params_from_pres.spectral_moments(
                freq=freq,
                psd=psd_surface,
            )

            hs_values[i] = 4.0 * np.sqrt(m0)
            tm_minus1_values[i] = m_minus1 / m0 if m0 > 0 else np.nan

        results[tbl_normalized] = {
            "time": time,
            "sea_surface_wave_significant_height": hs_values,
            "sea_surface_wave_energy_period": tm_minus1_values,
            "sea_depth": sea_depth,
        }

        lf.info("  Calculated {} wave parameters for table {}", n_intervals, table_name)



    lf.info("Calculating H1/3 from time domain for QC")
    for table_name, data in results.items():
        sea_depth = data["sea_depth"]
        lf.info("Calculating H1/3 for table: {}", table_name)

        h13_values, h13_time = calculate_h13_time_domain(
            h5_path=h5_path,
            table_name=table_name,
            sea_depth=sea_depth,
            fs=fs,
            cfg=cfg,
            psd_path=psd_path,
            table_stats=statistics.get(table_name),
        )

        # Add H1/3 to results with its own time coordinates
        if len(h13_values) > 0:
            results[table_name]["sea_surface_wave_significant_height_time_domain"] = (
                h13_values
            )
            results[table_name]["time_time_domain"] = h13_time
            lf.info(
                "  Calculated {} H1/3 values for table {}",
                len(h13_values),
                table_name,
            )
        else:
            results[table_name]["sea_surface_wave_significant_height_time_domain"] = (
                np.full(len(data["time"]), np.nan)
            )
            results[table_name]["time_time_domain"] = data["time"]
            lf.warning("  No valid H1/3 values calculated for table {}", table_name)

    # Save wave parameters to NetCDF
    output_path = h5_path.with_name(f"{h5_path.stem}_wave_params.nc")
    save_wave_parameters(results, output_path, cfg)

    lf.info("Wave parameters calculation completed: {}", output_path)
    return output_path


def calculate_h13_time_domain(
    h5_path: Path,
    table_name: str,
    sea_depth: float,
    fs: float,
    cfg: Dict,
    psd_path: Path,
    table_stats: Optional[Dict[str, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate H1/3 from time domain for a single table for quality control.

    Uses h5_velocity_by_intervals_gen() to iterate through intervals with
    same overlapping intervals as spectr_clc.main().

    Args:
        h5_path: Path to input HDF5 store
        table_name: Name of table to process
        sea_depth: Sea depth for this table
        fs: Sampling frequency in Hz
        cfg: Configuration dictionary
        psd_path: Path to spectrograms output file
        table_stats: Optional statistics from check_spectra_existence() containing
            min_date and max_date for the table. Used to avoid generating
            empty intervals in h5_velocity_by_intervals_gen().

    Returns:
        Tuple of (h13_values, time_values):
            - h13_values: Array of H1/3 values for each interval
            - time_values: Array of time coordinates for each interval (midpoint)
    """
    # Build spectr_clc config for interval generation with table-specific dates
    spectr_cfg = _build_spectr_clc_config(
        cfg["spectr_clc"], h5_path, psd_path, tables=[table_name], table_stats=table_stats
    )

    gen = spectr_clc.h5_velocity_by_intervals_gen(
        cfg=spectr_cfg,
        cfg_out=spectr_cfg["out"],
    )

    h13_values = []
    time_values = []

    for i, (df_interval, tbl, data_name) in enumerate(gen):
        if df_interval is None or len(df_interval) < 2:
            h13_values.append(np.nan)
            # Use NaN for time when interval is invalid
            time_values.append(np.datetime64("NaT"))
            continue

        # Calculate interval midpoint as time coordinate
        interval_start, interval_end = df_interval.index[[0, -1]].to_numpy()
        time_values.append(interval_start + (interval_end - interval_start) / 2)

        # Interpolate to regular grid
        df_interp_result, bads = spectr_clc.df_interp(df_interval, fs=fs)
        if df_interp_result is None or len(df_interp_result) < 2:
            h13_values.append(np.nan)
            continue

        # Convert pressure time series (dBar) to surface elevation time series
        # Using shallow-water approximation suitable for sensors near seabed
        elevation, water_column = wave_params_from_pres.pressure_to_elevation_linear_shallow_simple(
            water_pressure_dbar=df_interp_result["Pressure"].values,
            depth=sea_depth,
        )

        # Calculate H1/3 from time domain
        h13_values.append(
            wave_params_from_pres.h13_from_time_domain(
                eta=elevation,
                fs=fs,
            )
        )

    return np.array(h13_values), np.array(time_values)


def _add_common_global_attributes(
    ds: xr.Dataset,
    output_path: Path,
    cfg: Dict,
    title: str,
    method: str,
    additional_attrs: Optional[Dict] = None,
) -> xr.Dataset:
    """
    Add common global attributes to xarray Dataset.

    Centralized function to avoid code duplication when adding standard
    CF-compliant attributes to datasets.

    Args:
        ds: xarray Dataset to add attributes to
        output_path: Path to source HDF5 file for reference
        cfg: Configuration dictionary
        title: Dataset title
        method: Method description
        additional_attrs: Optional additional attributes to add

    Returns:
        Dataset with added attributes
    """
    ds.attrs["title"] = title
    ds.attrs["institution"] = "Shirshov Institute of Oceanology"
    ds.attrs["source"] = str(output_path.with_suffix(".h5"))
    ds.attrs["history"] = f"Created by {Path(__file__).name}"
    ds.attrs["Conventions"] = "CF-1.8"
    ds.attrs["method"] = method

    # Add spectral analysis parameters if available
    for param_name in ["fmin", "fmax"]:
        if param_name in cfg["wave_params"]:
            ds.attrs[param_name] = cfg["wave_params"][param_name]
    if "dt_interval" in cfg["spectr_clc"]["proc"]:
        # Convert timedelta64 to string for NetCDF attribute compatibility
        ds.attrs["dt_interval"] = str(cfg["spectr_clc"]["proc"]["dt_interval"])

    # Add additional attributes if provided
    if additional_attrs:
        ds.attrs.update(additional_attrs)

    return ds


def save_wave_parameters(results: Dict, output_path: Path, cfg: Dict) -> None:
    """
    Save wave parameters to NetCDF file with separate groups for each table.

    Saves each table as a separate NetCDF group containing both spectral domain
    parameters (Hs, Tm-1) and time domain parameters (H1/3) in subgroups.
    This structure allows easy access to parameters per table.

    Args:
        results: Dictionary with wave parameters for each table
        output_path: Path to output NetCDF file
        cfg: Configuration dictionary
    """
    import netCDF4

    param_name_h1_3 = "sea_surface_wave_significant_height_time_domain"

    # Use context manager to handle NetCDF file operations
    with netCDF4.Dataset(output_path, mode="w", format="NETCDF4") as nc_root:
        # Add global attributes to root
        nc_root.setncattr("title", "Wave Parameters from Pressure Sensors")
        nc_root.setncattr("institution", "Shirshov Institute of Oceanology")
        nc_root.setncattr("source", str(output_path.with_suffix(".h5")))
        nc_root.setncattr("history", f"Created by {Path(__file__).name}")
        nc_root.setncattr("Conventions", "CF-1.8")

        # Add spectral analysis parameters
        for param_name in ["fmin", "fmax"]:
            if param_name in cfg["wave_params"]:
                nc_root.setncattr(param_name, cfg["wave_params"][param_name])
        if "dt_interval" in cfg["spectr_clc"]["proc"]:
            nc_root.setncattr(
                "dt_interval", str(cfg["spectr_clc"]["proc"]["dt_interval"])
            )

        # Save each table as a separate group
        for table_name, data in results.items():
            # Create spectral domain group for this table
            nc_group = nc_root.createGroup(table_name)

            # Add table-specific attributes
            nc_group.setncattr("table_name", table_name)
            nc_group.setncattr(
                "sensor_height_above_bottom", cfg["wave_params"]["sensor_height"]
            )
            nc_group.setncattr("sea_depth", data["sea_depth"])
            nc_group.setncattr("method", "Spectral analysis from pressure spectrograms")

            # Create time dimension
            time_dim = nc_group.createDimension("time", len(data["time"]))
            time_var = nc_group.createVariable(
                "time", "f8", ("time",), fill_value=np.nan
            )
            # Convert datetime64 to numeric (seconds since 1970-01-01)
            time_var[:] = data["time"].astype("datetime64[s]").astype(int)
            time_var.setncattr("standard_name", "time")
            time_var.setncattr("units", "seconds since 1970-01-01")

            # Create spectral domain variables
            for var_name in [
                "sea_surface_wave_significant_height",
                "sea_surface_wave_energy_period",
            ]:
                var = nc_group.createVariable(
                    var_name, "f8", ("time",), fill_value=np.nan
                )
                var[:] = data[var_name]
                if var_name == "sea_surface_wave_significant_height":
                    var.setncattr("units", "m")
                    var.setncattr("standard_name", "sea_surface_wave_significant_height")
                    var.setncattr("method", "spectral_moment_m0")
                elif var_name == "sea_surface_wave_energy_period":
                    var.setncattr("units", "s")
                    var.setncattr("standard_name", "sea_surface_wave_energy_period")

            lf.info("Saved spectral domain for table: {}", table_name)

            # Save time domain to separate subgroup if available
            if param_name_h1_3 in data:
                h13_values = data[param_name_h1_3]
                time_h13 = data.get("time_time_domain", data["time"])

                nc_time_domain = nc_group.createGroup("time_domain")
                nc_time_domain.setncattr("table_name", table_name)
                nc_time_domain.setncattr(
                    "sensor_height_above_bottom", cfg["wave_params"]["sensor_height"]
                )
                nc_time_domain.setncattr("sea_depth", data["sea_depth"])
                nc_time_domain.setncattr(
                    "method", "Time domain analysis from pressure time series"
                )
                nc_time_domain.setncattr(
                    "description",
                    "H1/3 calculated from time domain for quality control. "
                    "Note: time dimension may differ from spectral domain due to "
                    "different interval processing.",
                )

                # Create time dimension for time domain
                time_dim_td = nc_time_domain.createDimension("time", len(time_h13))
                time_var_td = nc_time_domain.createVariable(
                    "time", "f8", ("time",), fill_value=np.nan
                )
                time_var_td[:] = time_h13.astype("datetime64[s]").astype(int)
                time_var_td.setncattr("standard_name", "time")
                time_var_td.setncattr("units", "seconds since 1970-01-01")

                # Create H1/3 variable
                h13_var = nc_time_domain.createVariable(
                    param_name_h1_3, "f8", ("time",), fill_value=np.nan
                )
                h13_var[:] = h13_values
                h13_var.setncattr("units", "m")
                h13_var.setncattr("standard_name", "sea_surface_wave_significant_height")
                h13_var.setncattr("method", "zero_upcrossing")

                lf.info("Saved time domain for table: {}", table_name)

    lf.info("Wave parameters saved to: {}", output_path)


def main(
    h5_path: Optional[Path] = None,
    fs: Optional[float] = None,
    min_date: Optional[str] = None,
    max_date: Optional[str] = None,
    fast_check: bool = True,
) -> None:
    """
    Main function to calculate wave parameters from pressure sensor data.

    Args:
        h5_path: Path to input HDF5 store with pressure data
        fs: Sampling frequency in Hz (if None, determined from data)
        min_date: Optional minimum date string (e.g., '2023-01-01T00:00:00')
        max_date: Optional maximum date string (e.g., '2023-12-31T23:59:59')
        fast_check: If True, use fast check for spectra existence (default: True).
            Fast check only verifies tables and date ranges, skipping detailed interval
            matching. Missing spectra are detected during wave parameter calculation.
    """
    lf.info("=" * 70)
    lf.info("Wave Parameters Processing Started")
    lf.info("=" * 70)

    if h5_path is None:
        h5_path = Path(
            "B:/Cruises/BalticSea/201202_BalticSpit/inclinometer/"
            "201202P1-5,I1-2@i3,5,9,10,11,15,19,23,28,30,32,33,w1-6/"
            "201202P1-5,I1-2.proc_noAvg.h5"
        )
        lf.warning("Using default path: {}", h5_path)

    if not h5_path.exists():
        raise FileNotFoundError(f"Input HDF5 file not found: {h5_path}")

    if fs is None:
        fs = get_sampling_frequency(h5_path, "w01")

    psd_path = h5_path.with_name(f"{h5_path.stem}_proc_psd.nc")

    cfg_with_overrides = copy.deepcopy(cfg)
    if min_date:
        cfg_with_overrides["spectr_clc"]["in"]["min_date"] = min_date
    if max_date:
        cfg_with_overrides["spectr_clc"]["in"]["max_date"] = max_date

    all_spectra_exist, statistics = check_spectra_existence(
        psd_path=psd_path,
        h5_path=h5_path,
        cfg=cfg_with_overrides,
        fast_check=fast_check,
    )

    if statistics:
        lf.info("=" * 70)
        lf.info("Spectrum Existence Statistics:")
        lf.info("=" * 70)
        for table_name, stats in statistics.items():
            lf.info(
                "  Table {}: total={}, existing={}, missing={}, "
                "min_date={}, max_date={}",
                table_name,
                stats["total"],
                stats["existing"],
                stats["missing"],
                stats["min_date"],
                stats["max_date"],
            )
        lf.info("=" * 70)

    if not all_spectra_exist:
        lf.info("Calculating spectra for missing intervals...")
        spectr_kwargs = copy.deepcopy(cfg["spectr_clc"])
        spectr_kwargs["in"]["db_path"] = h5_path
        spectr_kwargs["in"]["fs"] = fs
        if min_date:
            spectr_kwargs["in"]["min_date"] = min_date
        if max_date:
            spectr_kwargs["in"]["max_date"] = max_date
        spectr_kwargs["out"]["db_path"] = psd_path
        spectr_kwargs["program"]["b_interact"] = 0

        spectr_clc_config_path = Path(__file__).parent / "cfg" / "spectr_clc.yaml"
        spectr_clc.main([str(spectr_clc_config_path)], **spectr_kwargs)
        lf.info("Spectrograms saved to: {}", psd_path)
    else:
        lf.info("Using existing spectra from: {}", psd_path)

    wave_params_path = calculate_wave_parameters(
        h5_path=h5_path,
        psd_path=psd_path,
        fs=fs,
        cfg=cfg,
        max_missing_spectra=cfg["wave_params"]["max_missing_spectra"],
        statistics=statistics,
    )

    lf.info("=" * 70)
    lf.info("Wave Parameters Processing Completed")
    lf.info("=" * 70)
    lf.info("Spectrograms saved to: {}", psd_path)
    lf.info("Wave parameters saved to: {}", wave_params_path)


if __name__ == "__main__":
    main()
