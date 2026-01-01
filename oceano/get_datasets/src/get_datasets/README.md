# Data Downloading Scripts

Scripts designed for downloading various types of environmental data from different sources, including Copernicus Marine Service (CMEMS) and ECMWF ERA5 reanalysis. The scripts utilize the Hydra configuration framework for flexible parameter management. The logging history and parameters of the download attempt (success or failure) is managed by `DownloadHistoryManager`.

The main functionality has been unified into a single `download_copernicus.py` script that can handle both point and region downloads based on configuration. It also supports GPX file input for multiple coordinate points.

### Configuration Structure
Hydra allows you define your download tasks in dedicated YAML configuration files in the external user configuration directory separating general settings from project-specific ones:

*   **`cfg/base.yaml`**: Contains general settings applicable across all downloading tasks, such as the base directory for saving downloaded data (`base.local_path`) and the path to the download history file (`base.history_file`).
*   **`cfg/copernicus.yaml`**: Defines configurations specific to Copernicus Marine Service data downloads. It now includes a `defaults` section that allows composing project-specific configurations from the `projects/` subdirectory.
*   **`cfg/projects/` Directory**: This is where you define individual download projects. Each project gets its own subdirectory (e.g., `cfg/projects/kulikovo/`, `cfg/projects/mariculture_1/`).
    *   Within each project subdirectory, you create YAML files named after the type of download (e.g., `point_wind.yaml`, `region.yaml`). These files contain the specific parameters for that project and download type.
    *   **Example Project Structure**:
        ```
        cfg/
        ├── base.yaml
        ├── copernicus.yaml
        └── projects/
            ├── kulikovo/
            │   └── point_wind.yaml
            ├── abp56/
            │   └── point_wind.yaml
            ├── mariculture_1/
            │   └── region.yaml
            └── ...
        ```
    *   **Content of Project YAMLs**: Each project YAML file should start with `# @package projects.<project_name>` to ensure its content is composed correctly under the `projects` key in the final configuration. The parameters within these files directly define the settings for the specific download type (e.g., `point_wind` or `region`).

The path to the external `cfg/` user configuration directory (currently `/oceano/get_datasets/cfg`) is defined in the package hydra configuration `/oceano/get_datasets/src/get_datasets/cfg/base.yaml

### Organizing Your Settings

To define a new download task:
1.  Create a new subdirectory under `cfg/projects/` for your project (e.g., `cfg/projects/my_new_project/`).
2.  Inside your project directory, create a YAML file corresponding to the script you want to use (e.g., `point_wind.yaml` for `download_copernicus_point.py`, or `region.yaml` for `download_copernicus_region.py`).
3.  Populate this YAML file with the necessary parameters for your download, following the structure defined in the respective script's `main` function. Remember to add `# @package projects.<your_project_name>` at the top of the file.

## Running Scripts

You specify which project configuration to load using command-line overrides.

### Single Project Execution

To run a script for a specific project, use the `projects=<project_name>/<config_type>` override with the unified script:

    ```bash
    pixi run python -m get_datasets.download_copernicus projects=abp56_tchain/point
    ```

*   **For point-based downloads** (formerly `download_copernicus_point.py`):
    ```bash
    python download_copernicus.py projects=kulikovo/point_wind
    ```
    or
    ```bash
    python download_copernicus.py projects=abp56/point_wind
    ```

*   **For region-based downloads** (formerly `download_copernicus_region.py`):
    ```bash
    python download_copernicus.py projects=mariculture_1/region
    ```
    (Replace `mariculture_1` with `mariculture_2`, `mariculture_3`, `abp56_tchain`, or `inflow` for other region projects.)

### Multi-run Execution (for testing or batch processing)

Hydra's multi-run feature allows you to run a script multiple times with different configurations. This is particularly useful for testing all defined projects or performing batch downloads.

To run all `point_wind` projects:
```bash
python download_copernicus.py --multirun projects=kulikovo/point_wind,abp56/point_wind
```

To run all `region` projects:
```bash
python download_copernicus.py --multirun projects=mariculture_1/region,mariculture_2/region,mariculture_3/region,abp56_tchain/region,inflow/region
```

## Architecture Overview

The downloading system is organized around a centralized configuration managed by Hydra, and a set of specialized Python scripts for each data source. Common utilities are consolidated to promote code reuse and maintainability.

*   **`cfg/` Directory**: This directory holds all the YAML configuration files for the downloading scripts.
    *   `base.yaml`: Contains general settings applicable across all downloading tasks.
    *   `copernicus.yaml`: Defines general Copernicus-specific configurations and acts as a composer for project-specific configurations.
    *   `projects/`: Contains subdirectories for each defined download project, with project-specific configuration files (e.g., `kulikovo/point_wind.yaml`).
    *   `ncep.yaml`: Contains configurations for NCEP CFSv2 reanalysis data downloads.  (not used, todo: implement)
    *   `ecmwf.yaml`: Stores configurations for ECMWF ERA5 data downloads.  (not used, todo: implement)



*   **`manager.py`**: Manages the logging and retrieval of download history, ensuring that successful and failed download attempts are recorded.

*   **Download Scripts**: Each script is responsible for interacting with a specific data source or performing a specific type of download. They leverage Hydra for configuration and the `utils.py` module for common data handling tasks.

## Scripts Description

### `download_copernicus_point.py`

*   **Purpose**: Downloads CMEMS wind data for a small region around a specified point and then interpolates the data to the exact point. This is useful for obtaining time series data at precise locations.
*   **Workflow**:
    1.  Loads configuration from Hydra, including dataset ID, variables, and interpolation delta.
    2.  Defines a bounding box around the target latitude and longitude using the `interpolation_delta`.
    3.  Uses the `copernicusmarine` library to download the regional subset of data.
    4.  Interpolates the downloaded gridded data to the exact target point using `utils.interp_to_point()`.
    5.  Saves the interpolated data as a NetCDF file.
    6.  Logs the download and interpolation process.
*   **Execution**:
    ```bash
    python download_copernicus_point.py projects=<project_name>/point_wind
    ```
    (e.g., `projects=kulikovo/point_wind` or `projects=abp56/point_wind`)
*   **Expected Output**: A NetCDF file containing interpolated wind data for the specified point in the designated output directory (e.g., `data/downloaded/CMEMS_Wind_Kulikovo/`), and history logs.



### `download_copernicus_region.py`

*   **Purpose**: Downloads CMEMS data for a specified geographical region using the `copernicusmarine` API. This script is ideal for retrieving gridded data over a larger area.
*   **Workflow**:
    1.  Loads dataset IDs, variables, bounding box coordinates, date range, and depth parameters from the Hydra configuration.
    2.  Calls the `copernicusmarine.subset()` function to download the regional data.
    3.  Handles potential API errors.
*   **Execution**:
    ```bash
    python download_copernicus_region.py projects=<project_name>/region
    ```
    (e.g., `projects=mariculture_1/region` or `projects=inflow/region`)
*   **Expected Output**: NetCDF files containing gridded data for the specified region, saved in the configured output directory.

### `download_copernicus_ftp.py`

*   **Purpose**: Downloads Copernicus Marine Service (CMEMS) data via FTP. This script is suitable for retrieving large datasets or specific collections that are available through the CMEMS FTP service.
*   **Workflow**:
    1.  Retrieves CMEMS credentials from Hydra configuration.
    2.  Downloads index files from the CMEMS FTP server to identify available data.
    3.  Reads and merges information from the index files.
    4.  Filters the available data based on specified spatial (bounding box) and temporal (date range) criteria, and optionally by parameters.
    5.  Downloads the selected NetCDF files from the FTP server.
*   **Execution**:
    ```bash
    python download_copernicus_ftp.py
    ```
    Configuration parameters are loaded from `cfg/copernicus.yaml` under the `copernicus.ftp` section.
*   **Expected Output**: Downloaded NetCDF files in the specified local directory (e.g., `data/downloaded/CMEMS_FTP_Data/`), and log entries in the download history file.


## Utility Functions in `utils.py`

Detailed descriptions of functions in `utils.py` in directory above are provided in the Readme in that dir.

## Running Tests

To run the tests for the downloading scripts, you can use pytest:

```bash
cd scripts/downloading/with_manager
pytest test/
```

This will run all the tests in the `test/` directory. The tests will:

1. Set up a clean test environment for each test function
2. Run the download scripts with different project configurations
3. Verify that the download history is correctly recorded
4. Check that the expected NetCDF files are created

You can also run individual test files:

```bash
pytest test/test_download_copernicus_point.py
pytest test/test_download_copernicus_region.py
```

Or run a specific test function:

```bash
pytest test/test_download_copernicus_point.py::test_point_wind_kulikovo
```
