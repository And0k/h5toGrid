#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Remove humps from spectra in NetCDF file.

This script loads spectrograms from a NetCDF file created by spectr_clc,
removes humps from each spectrum using filt_humps.remove_hump(),
and saves the cleaned spectrograms to a new NetCDF file.

Author: Wave Parameters Processing
Created: 2025
"""
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import netCDF4

import tcm.filt_humps as filt_humps
from utils.init import LoggingStyleAdapter

# Initialize logger with str.format() style support
lf = LoggingStyleAdapter(__name__)


# ============================================================================
# CONFIGURATION PARAMETERS - Single source of truth for all settings
# ============================================================================

CONFIG = {
    # Input/Output file paths
    "input": {
        # Path to NetCDF file with spectrograms (output from spectr_clc)
        "B:/Cruises/BalticSea/201202_BalticSpit/inclinometer/201202P1-5,I1-2@i3,5,9,10,11,15,19,23,28,30,32,33,w1-6/201202P1-5,I1-2.proc_noAvg_proc_psd.nc"
        "psd_path": None,
    },
    "output": {
        # Path to output NetCDF file with cleaned spectrograms
        # Default: input_psd_path.with_suffix('_filtered.nc')
        "output_path": None,
    },
    # Spectral processing parameters - match filt_humps.remove_hump signature
    "spectral": {
        # Frequency range for hump search [Hz]
        "f_hump_min": 0.15,
        "f_hump_max": 0.25,
        # Baseline estimation function (direct function reference)
        "baseline_fn": filt_humps.baseline_als,
        # Baseline function parameters (passed to selected baseline function)
        "baseline_kw": {
            # For 'als' method
            "lam": 1e5,
            "p": 0.01,
            "niter": 10,
            # For 'poly' method (alternative)
            # "deg": 3,
            # "peak_threshold": 0.1,
            # For 'min' method (alternative)
            # "window": 50,
            # For 'snip' method (alternative)
            # "niter": 10,
            # "m": 5,
        },
        # Peak detection threshold relative to corrected-spectrum max
        "prominence_factor": 0.1,
        # Apply SNR mask after hump removal
        "apply_snr_mask": False,
        # SNR threshold for noise masking
        "snr_threshold": 5.0,
        # Minimum frequency for SNR mask
        "fmin": 0.04,
    },
    # Tables to process
    "tables": {
        # List of table names to process (None = process all tables matching pattern)
        "tables_list": ["w01"],
        # Table name pattern for filtering (e.g., "w.*" for all wave gauges)
        "table_pattern": "w.*",
    },
    # Program behavior
    "program": {
        # Logging configuration
        "log": "log/remove_humps.log",
        "verbose": "INFO",
        # Return baselines from remove_hump for analysis
        "return_baselines": False,
    },
}


# ============================================================================
# Helper functions
# ============================================================================

def get_wave_tables(nc_path: Path, table_pattern: str = "w.*") -> list:
    """
    Get list of wave gauge tables from NetCDF file.

    Args:
        nc_path: Path to NetCDF file
        table_pattern: Pattern to match table names (default: "w.*")

    Returns:
        List of table names matching the pattern
    """
    import re

    with netCDF4.Dataset(nc_path, mode="r") as nc_root:
        if "psd" not in nc_root.groups:
            lf.warning("No 'psd' group found in NetCDF file: {}", nc_path)
            return []

        nc_psd = nc_root.groups["psd"]
        pattern = re.compile(table_pattern)
        tables = [tbl for tbl in nc_psd.groups.keys() if pattern.match(tbl)]

    if not tables:
        lf.warning("No tables matching pattern '{}' found in {}", table_pattern, nc_path)

    return tables


def process_table(
    nc_path: Path,
    table_name: str,
    freq: np.ndarray,
    cfg: Dict,
) -> Tuple[Optional[np.ndarray], Optional[list], Optional[np.ndarray]]:
    """
    Process a single table: remove humps from all spectra.

    Args:
        nc_path: Path to input NetCDF file
        table_name: Name of table to process
        freq: Frequency array (Hz)
        cfg: Configuration dictionary

    Returns:
        Tuple of (cleaned_spectrogram, hump_info, baselines):
            - cleaned_spectrogram: (n_spectra, n_freqs) array or None if error
            - hump_info: List of per-spectrum hump information or None
            - baselines: (n_spectra, n_freqs) array or None if not requested
    """
    try:
        with netCDF4.Dataset(nc_path, mode="r") as nc_root:
            # Load spectrogram data
            psd_pres = nc_root[f"psd/{table_name}"].variables["Pressure"][:]
            lf.info(
                "Loaded {} spectra from table '{}' (shape: {})",
                psd_pres.shape[0],
                table_name,
                psd_pres.shape,
            )

    except (OSError, KeyError) as e:
        lf.error("Failed to load spectrogram from table '{}': {}", table_name, e)
        return None, None, None

    # Remove humps from spectrogram
    lf.info("Removing humps from table '{}'...", table_name)
    cleaned, hump_info, baselines = filt_humps.remove_hump(
        psd_pres,
        freq,
        **cfg["spectral"],
        return_baselines=cfg["program"]["return_baselines"],
    )

    # Log summary statistics
    lf.info(
        "Table '{}': {}/{} spectra had humps removed",
        table_name,
        sum(1 for h in hump_info if h["hump_position"] is not None),
        len(hump_info),
    )

    return cleaned, hump_info, baselines


def save_cleaned_spectrogram(
    nc_input_path: Path,
    nc_output_path: Path,
    table_name: str,
    cleaned_spectrogram: np.ndarray,
    hump_info: list,
    freq: np.ndarray,
    baselines: Optional[np.ndarray] = None,
    cfg: Dict = None,
) -> None:
    """
    Save cleaned spectrogram to output NetCDF file.

    Creates output file if it doesn't exist, opens in append mode otherwise.
    Copies all variables from input table and replaces Pressure data with cleaned version.

    Args:
        nc_input_path: Path to input NetCDF file
        nc_output_path: Path to output NetCDF file
        table_name: Name of table
        cleaned_spectrogram: Cleaned spectrogram data
        hump_info: List of hump information dicts
        freq: Frequency array
        baselines: Optional baseline array to save
        cfg: Configuration dictionary for metadata
    """
    # Open input file to copy structure
    with netCDF4.Dataset(nc_input_path, mode="r") as nc_in:
        gr_in = nc_in[f"psd/{table_name}"]

        # Open output file (create if doesn't exist)
        mode = "a" if nc_output_path.exists() else "w"

        with netCDF4.Dataset(nc_output_path, mode=mode, format="NETCDF4") as nc_out:
            # Create root and psd group if new file
            if mode == "w":
                nc_out.setncattr("title", "Cleaned Spectrograms (Humps Removed)")
                nc_out.setncattr("institution", "Shirshov Institute of Oceanology")
                nc_out.setncattr("source", str(nc_input_path))
                nc_out.setncattr("history", f"Created by {Path(__file__).name}")
                nc_out.setncattr("Conventions", "CF-1.8")
                nc_out.setncattr("processing", "Hump removal using filt_humps")
                nc_out.setncattr("f_hump_min", cfg["spectral"]["f_hump_min"])
                nc_out.setncattr("f_hump_max", cfg["spectral"]["f_hump_max"])
                nc_out.setncattr("baseline_fn", cfg["spectral"]["baseline_fn"].__name__)
                nc_psd = nc_out.createGroup("psd")
            else:
                try:
                    nc_psd = nc_out.groups["psd"]
                except KeyError:
                    nc_psd = nc_out.createGroup("psd")

            # Create table group if it doesn't exist
            try:
                gr_out = nc_psd.groups[table_name]
            except KeyError:
                gr_out = nc_psd.createGroup(table_name)

            # Copy dimensions from input
            for dim_name in gr_in.dimensions:
                if dim_name not in gr_out.dimensions:
                    gr_out.createDimension(
                        dim_name,
                        dim.size if not (dim := gr_in.dimensions[dim_name]).isunlimited() else None,
                    )

            # Copy variables from input (except Pressure which we replace)
            for var_name, var_in in gr_in.variables.items():
                if var_name == "Pressure":
                    continue

                if var_name not in gr_out.variables:
                    var_out = gr_out.createVariable(
                        var_name,
                        var_in.dtype,
                        var_in.dimensions,
                        zlib=var_in.filters() is not None,
                        fill_value=getattr(var_in, "_FillValue", None),
                    )
                    for attr_name in var_in.ncattrs():
                        var_out.setncattr(attr_name, var_in.getncattr(attr_name))
                    var_out[:] = var_in[:]

            # Create cleaned Pressure variable
            var_pres = gr_in.variables["Pressure"]
            pres_out = gr_out.createVariable(
                "Pressure",
                var_pres.dtype,
                var_pres.dimensions,
                zlib=var_pres.filters() is not None,
                fill_value=getattr(var_pres, "_FillValue", np.nan),
            )
            for attr_name in var_pres.ncattrs():
                pres_out.setncattr(attr_name, var_pres.getncattr(attr_name))
            pres_out.setncattr("processing", "Hump removed using filt_humps.remove_hump()")
            pres_out[:] = cleaned_spectrogram

            # Save baselines if requested
            if baselines is not None:
                var_baseline = gr_out.createVariable(
                    "baseline",
                    "f8",
                    ("time", "freq"),
                    zlib=True,
                    fill_value=np.nan,
                )
                var_baseline.setncattr("long_name", "Baseline used for hump removal")
                var_baseline.setncattr("units", var_pres.getncattr("units", "dBar^2/Hz"))
                var_baseline[:] = baselines

    # Save hump information as attributes
    gr_out.setncattr("n_spectra_with_humps", sum(1 for h in hump_info if h["hump_position"] is not None))
    gr_out.setncattr("n_spectra_total", len(hump_info))

    lf.info("Saved cleaned spectrogram for table '{}' to {}", table_name, nc_output_path)


# ============================================================================
# Main processing function
# ============================================================================

def main(cfg: Dict) -> None:
    """
    Main function to remove humps from spectra in NetCDF file.

    Args:
        cfg: Configuration dictionary with all parameters
    """
    lf.info("=" * 70)
    lf.info("Hump Removal Processing Started")
    lf.info("=" * 70)

    # Validate input path
    if not (psd_path := Path(cfg["input"]["psd_path"])).exists():
        raise FileNotFoundError(f"Input NetCDF file not found: {psd_path}")

    # Set output path if not specified
    cfg["output"]["output_path"] = (
        Path(output_path) if (output_path := cfg["output"]["output_path"])
        else psd_path.with_suffix("_filtered.nc")
    )

    lf.info("Input NetCDF file: {}", psd_path)
    lf.info("Output NetCDF file: {}", cfg["output"]["output_path"])

    # Get tables to process
    tables = (
        tables_list
        if (tables_list := cfg["tables"]["tables_list"])
        else get_wave_tables(psd_path, cfg["tables"]["table_pattern"])
    )
    lf.info(
        "Processing specified tables: {}" if tables_list
        else "Found {} tables matching pattern '{}': {}",
        tables if tables_list else (len(tables), cfg["tables"]["table_pattern"], tables),
    )

    if not tables:
        lf.warning("No tables to process. Exiting.")
        return

    # Get frequency array from first table
    with netCDF4.Dataset(psd_path, mode="r") as nc_root:
        try:
            freq = nc_root[f"psd/{tables[0]}"].variables["freq"][:]
        except KeyError:
            # Try loading frequency from psd group level
            freq = nc_root["psd"].variables["freq"][:]

    lf.info("Frequency range: {:.4f} - {:.4f} Hz ({} points)", freq[0], freq[-1], len(freq))
    lf.info("Hump search range: {:.4f} - {:.4f} Hz", cfg["spectral"]["f_hump_min"], cfg["spectral"]["f_hump_max"])

    # Process each table
    total_spectra = 0
    total_humps = 0

    for table_name in tables:
        lf.info("-" * 70)
        lf.info("Processing table: {}", table_name)

        cleaned, hump_info, baselines = process_table(psd_path, table_name, freq, cfg)

        if cleaned is None:
            lf.warning("Skipping table '{}' due to error", table_name)
            continue

        # Save cleaned spectrogram
        save_cleaned_spectrogram(
            psd_path,
            cfg["output"]["output_path"],
            table_name,
            cleaned,
            hump_info,
            freq,
            baselines,
            cfg,
        )

        # Update statistics
        total_spectra += len(hump_info)
        total_humps += sum(1 for h in hump_info if h["hump_position"] is not None)

    # Final summary
    lf.info("=" * 70)
    lf.info("Hump Removal Processing Completed")
    lf.info("=" * 70)
    lf.info("Total spectra processed: {}", total_spectra)
    lf.info("Total humps removed: {}", total_humps)
    lf.info("Output saved to: {}", cfg["output"]["output_path"])


if __name__ == "__main__":
    # Initialize logging
    from utils.init import init_logging

    init_logging(lf, CONFIG["program"]["log"], CONFIG["program"]["verbose"])

    # Run main function with configuration
    main(CONFIG)
