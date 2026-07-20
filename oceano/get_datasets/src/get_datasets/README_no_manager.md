# Data Downloading Scripts

This directory contains Python scripts designed for downloading various types of environmental data from different sources, including Copernicus Marine Service (CMEMS) and ECMWF ERA5 reanalysis.

Note: some more structured and modular downloading scripts are in `with_manager` directory. That scripts utilize the Hydra configuration framework for flexible parameter management, they includes `DownloadHistoryManager` for managing the logging history and parameters of the download attempt (see `with_manager/README.md` for more details).

## Utility Functions in `d_utils.py`

*   **`d_utils.py`**: This module provides a collection of core utility functions used by multiple downloading scripts. These include:
    *   `safe_netcdf_atomic()`: Safely saves NetCDF files with atomic overwrite.
    *   `extract_zip_to_named_dir()`: Extracts ZIP archives into a directory named after the archive.
    *   `h5_format()`: Formats and adds metadata to HDF5/NetCDF files.
    *   `grid_aligned_bbox()`: Generates ECMWF-style bounding boxes aligned to the ERA5 grid.
    *   `is_angular()`: Checks if a variable is angular based on its units.
    *   `interp_angle()`: Interpolates angular data (e.g., wind direction).
    *   `interp_to_point()`: Interpolates gridded data to a specific geographical point.
    *   `ReverseTxt`: A utility class for reading text files in reverse.


## Scripts Description

### `download_copernicus_points.py`

*   **Purpose**: Downloads CMEMS data for multiple predefined geographical points and datasets using the `copernicusmarine` API. This script is designed for batch downloading of point-specific time series data.
*   **Workflow**:
    1.  Loads dataset IDs, variables, and a list of target points from the Hydra configuration.
    2.  Iterates through each dataset and each point.
    3.  For each combination, it calls the `copernicusmarine.subset()` function to download data for the single point.
    4.  Handles potential API errors (e.g., `requests.exceptions.JSONDecodeError`).
*   **Execution**:
    ```bash
    python scripts/downloading/download_copernicus_points.py
    ```
    Configuration parameters are loaded from `scripts/downloading/cfg/copernicus.yaml` under the `copernicus.point_wind` (for points and variables) and `copernicus.region` (for general settings like depth, compression) sections.
*   **Expected Output**: Multiple NetCDF files, each containing data for a specific point and dataset, saved in the configured output directory.


### `download_ecmwf.py`

*   **Purpose**: Downloads ERA5 reanalysis data from the ECMWF Climate Data Store (CDS) using the `cdsapi` library. It supports downloading various meteorological and wave parameters for a specified area and time range.
*   **Workflow**:
    1.  Loads configuration from Hydra, including dataset, variables, product types, and geographical/temporal settings.
    2.  Determines the output directory and filename based on configuration.
    3.  Checks for existing downloaded files to avoid re-downloading.
    4.  Groups variables by their typical spatial resolution (e.g., 0.25x0.25 degrees for atmospheric, 0.5x0.5 degrees for wave data).
    5.  Uses `cdsapi.Client` to retrieve data from the CDS, handling different resolutions and potential proxy settings.
    6.  Extracts data from downloaded ZIP archives (if `download_format` is `zip`).
    7.  Formats the downloaded NetCDF files using `d_utils.h5_format()`.
    8.  Optionally converts NetCDF files to CSV using `netcdf2csv.py`.
    9.  Interpolates the gridded data to a specific point using `d_utils.interp_to_point()`.
*   **Execution**:
    ```bash
    python scripts/downloading/download_ecmwf.py
    ```
    Configuration parameters are loaded from `scripts/downloading/cfg/ecmwf.yaml` and `base.yaml`.
*   **Expected Output**: Downloaded NetCDF files (and optionally CSV files) in the specified output directory (e.g., `data/downloaded/ECMWF/`), containing ERA5 reanalysis data.

### `download_ncep.py`

*   **Purpose**: Downloads NCEP CFSv2 reanalysis wind data via OPeNDAP from the APDRC server. It can download data for a specific point or a region and handles time encoding for NetCDF saving.
*   **Workflow**:
    1.  Loads configuration from Hydra, including the base OPeNDAP URL, variables, and geographical/temporal settings.
    2.  Determines the output file path and checks for existing matching files.
    3.  Constructs OPeNDAP URLs for the requested variables and date range.
    4.  Uses `xarray.open_mfdataset()` with a `preprocess` function to subset data in time and space efficiently.
    5.  Selects the nearest grid points (1 or 4) to the target location.
    6.  Saves the processed dataset to a NetCDF file, handling time encoding to ensure compatibility.
    7.  Optionally loads and joins split NCEP files from a local directory.
    8.  Interpolates the data to a specific point using `d_utils.interp_to_point()`.
*   **Execution**:
    ```bash
    python scripts/downloading/download_ncep.py
    ```
    Configuration parameters are loaded from `scripts/downloading/cfg/ncep.yaml` and `base.yaml`.
*   **Expected Output**: Downloaded and processed NetCDF files in the specified output directory (e.g., `data/downloaded/NCEP_CFSv2/`), containing NCEP CFSv2 wind reanalysis data.

### `netcdf2csv.py`

*   **Purpose**: A standalone utility script to convert NetCDF files (specifically those downloaded from ECMWF) into CSV format. It supports different output methods based on how the data should be structured in the CSV.
*   **Workflow**:
    1.  Opens a NetCDF file using `netCDF4.Dataset`.
    2.  Extracts time, latitude, longitude, and specified variables.
    3.  Converts the data into a pandas DataFrame.
    4.  Saves the DataFrame to a CSV file using one of three methods:
        *   `file_for_each_time`: Creates a separate CSV file for each time step.
        *   `one_file`: Creates a single CSV file with time, lat, lon, and all variable values.
        *   `file_for_each_coord`: Creates a separate CSV file for each geographical coordinate, containing time series data for all variables at that point.
*   **Execution**:
    ```bash
    python scripts/downloading/netcdf2csv.py --file_path <path_to_netcdf_file> [options]'
    ```
    This script is typically called internally by `download_ecmwf.py` but can be run independently.
*   **Expected Output**: One or more CSV files containing the converted data, depending on the chosen method.
