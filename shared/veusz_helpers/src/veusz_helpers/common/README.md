# Veusz plugins: pattern loader

This tool selects data files, determines which data parts to load, and defines how to display them in Veusz based on file path/name and data files. When a user opens a .vsz file containing only commands that call vsz_loader.py, the loader takes control of the execution logic.

## How It Works

The system operates by parsing the .vsz filename to extract time range and device information. The vsz_loader.py module uses this information to locate and load appropriate data files (HDF5, CSV, NetCDF) with the correct time range and parameters. The data is then visualized using Veusz with the help of supporting modules like vsz_drawer.py, which creates graphs and plots.

Key functions involved in the process:
1. `get_info_from_filename()` - parses the filename to extract time range and device information
2. `veusz_load_hdf5()` - loads HDF5 data into Veusz based on time range and device specifications
3. `veusz_load_hdf5_ecmwf()` and `veusz_load_hdf5_cmems()` - specialized functions for loading meteorological data
4. `prepare_draw_tcm()` - prepares processed TCM (inclinometer) data for drawing
5. `load_info_json()` - loads device metadata from JSON configuration files
6. `get_path_in_parents()` - searches for configuration files in parent directories

## Configuration

Users can overwrite vsz_loader.py, or the modules it uses (like vsz_drawer_cfg.py for configuration and vsz_drawer.py for visualization) to achieve specific program behavior in different directories.

## Directory Structure and Special Keywords

The system uses special directory names and structure patterns to determine data processing behavior:

### Special Directory Names and Their Effects

- **`_raw`**: When present in the path hierarchy, the system prioritizes raw data loading and changes search behavior to locate raw data files in `_raw` subdirectories
- **`inclinometer`**: Directories with this name trigger specific TCM (inclinometer) data processing logic
- **`txt`**: Directories with `txt` in their name modify the data source search logic to try load text-based data
- **`vsz`**: Directories with `vsz` in their name affect search parent logic for locating data files
- **`meteo/ECMWF`**: Directories with these names are specifically searched for meteorological data files
- **`processed` and `raw`**: These directory names may be used to distinguish between different processing states of data files

### Directory Navigation Keywords in .vsz File Names
- **`vsz` prefix**: Indicates that data files for *.vsz files inside are on same level as the `vsz*` dir itself
- **`..vsz` prefix**: Indicates the number of parent directory levels to navigate up when searching for data files (e.g., `..vsz` means to look one level up)
- **`vsz({dir})` pattern**: Points to a sibling directory named `{dir}` where data files are located
- **`vsz(dir={dir})` pattern**: Alternative syntax for specifying the data directory location

### Recommended Directory Structure

The system works best with a structured directory layout:

```
project_root/
├── 230825_Kulikovo@ADCP,ADV,i,tr/          # Date and location data
│   ├── raw/                                  # Raw data files
│   ├── processed.h5                         # data file/database
│   ├── inclinometer/                        # Inclinometer data (special keyword)
│   │   ├── info_devices.json               # Device configuration
│   │   └── vsz/                            # .vsz files per time range
│   │       ├── 230830_0000-10_1200@windECMWF.vsz    # Load 10 days of wind searching its data in meteo/ECMWF
│   │       └── 230830_0000@i91.vsz         # Load single day for inclinometer i91 from processed.h5
│   ├── meteo/                               # Meteorological data (special keyword)
│   │   └── ECMWF/                          # ECMWF NetCDF files
│   │       └── area(54.75-55.0N,20.25-20.5E)/
│   └── _raw/                                # Raw data directory (special keyword)
```

## Usage

Copy the 449-byte file `scripts\231229_2201@i91.vsz` to a directory containing your data or in a subdirectory that allows data discovery by renaming them like `{yymmdd_HHMM}@{device_ids}.vsz`. This enables the system to find data in known formats named according to known devices.
Open (run) your correctly named 449-byte .vsz file with Veusz. For debugging: run vsz_loader.py with the full path as an argument to this file.

## .vsz File Naming Patterns & Device Parsing

### File Naming Structure
The .vsz filename structure `{yymmdd_HHMM}@{device_ids}.vsz` enables automated data discovery:

- **Date and time**: `{yymmdd_HHMM}` specifies when the data collection started (YYMMDD for year-month-day, HHMM for hour-minute)
- **Device identifiers**: `{device_ids}` specifies which devices to load data for
- **Time ranges**: Use `{start_time}-{end_time}` for ranges (e.g., `230830_0000-10_1200` for a 10-day period)
- **Duration**: Use `dt=duration` to specify duration from a start time (e.g., `230830_0000dt=1h` for 1 hour)

### Device Name Parsing Rules
- Underscores after "i" or zeros before first other number are not considered significant:
  - `i_b27` → `ib27`
  - `i_p06` → `ip06`
  - `i_03` → `i03`
  - `i_p06` → `ip6`
- Complex patterns:
  - Semicolon-separated groups: `i3,4,15,19,37,38;ib27-30,ip6.vsz` - separates different device groups
  - Range (numeric only) expansion: `27-30` → `["27", "28", "29", "30"]`
  - Parentheses support: `i(38,37,59,60,58)` → `i38,i37,i59,i60,i58`

## Key `vsz_loader` Functions

- get_info_from_filename - extracts time range and device information from the basename of the .vsz file
- veusz_load_hdf5 - loads HDF5 data to Veusz based on specified time range and device IDs
- veusz_load_hdf5_tcm_raw - loads raw TCM (inclinometer) data from HDF5 files
- veusz_load_hdf5_ctd_profile - loads CTD profile data from HDF5 files
- veusz_load_hdf5_ecmwf - loads ECMWF meteorological data from NetCDF files
- veusz_load_hdf5_cmems - loads CMEMS oceanographic data from NetCDF files
- veusz_load_csv_gmx500 - loads GMX 500 data from CSV files
- veusz_load_csv_ecmwf - loads ECMWF data from CSV files
- prepare_draw_tcm - prepares processed TCM data for visualization
- load_info_json - loads device information from JSON configuration files
- get_path_in_parents - searches for specified file in parent directories
- get_fun_load_end_ext - determines the appropriate loading function based on device type and file extension
- add_months - adds months to a numpy datetime64 object
- search_time_range_indexes - finds time range indexes for data slicing
- bool2ranges - converts boolean array to ranges ignoring short intervals between
- zone_to_seconds_offset - converts time zone string to seconds offset
- _info_json_item_array_to_dict - converts JSON array item to dictionary format

## `vsz_loader` Dependent Modules

- vsz_drawer - visualization module for creating plots and graphs in Veusz
- vsz_drawer_cfg - configuration module for drawer parameters and settings
- func_vsz - support module with various utility functions for Veusz operations

## Data Format Support

The system supports various data formats and loading mechanisms:

- **HDF5 files**: For various instruments (TCM, CTD, navigation)
- **NetCDF files**: For meteorological data (ECMWF, CMEMS)
- **CSV files**: For specific instruments (GMX500)
- **Configurable Visualization**: Uses vsz_drawer and vsz_drawer_cfg to create customizable plots and graphs

## Potential Issues and Areas for Improvement

1. **Missing Error Handling**: Some functions may fail unexpectedly when data is missing or in unexpected formats. Better error handling and fallback mechanisms are needed.

2. **Hardcoded Values**: Several functions use hardcoded values that might not be appropriate for all datasets. These should be made configurable.

3. **Type Handling**: The system handles different device types (i, w, tr, ECMWF, CMEMS) with complex string manipulations that could be simplified with proper data structures.

4. **Data Validation**: Limited validation of input data which could lead to processing failures with unexpected data formats.

5. **Numpy Array Handling**: There are instances where numpy arrays are used where scalars are expected, causing deprecation warnings. These need to be fixed.

6. **Path Handling**: The system assumes specific directory structures which might not be present in all environments.

7. **Custom Definitions Integration**: The tight coupling with Veusz Custom Definitions requires the code to be executed in specific contexts, limiting reusability.

8. **Incomplete File Loading**: Some loader functions are marked as incomplete or not implemented (e.g., loading with time ranges for CSV functions).

9. **Coordinate Logic**: The logic for finding the nearest meteorological data based on coordinates could be more robust.

10. **Decimation Logic**: The data decimation feature implementation may need review for correctness and performance.

# vsz files naming rules

## Data Filenames Pattern Support
`{yymmdd_HHMM}@{Device Names}.vsz'

### Device Names in vsz file name
underscores after "i" or zeros before first other number are not considered significant:
- `i_b27` → `ib27`
- `i_p06` → `ip06`
- `i_03` → `i03`
- `i_p06` → `ip6`

### Complex Patterns
Semicolon-Separated Groups:
- `i3,4,15,19,37,38;ib27-30,ip6.vsz` - separates different device groups

Range (numeric only) Expansion:
- `27-30` → `["27", "28", "29", "30"]`

Parentheses Support:
- `i(38,37,59,60,58)` → `i38,i37,i59,i60,i58`
- `i_b(27,28,29,30)` → `i_b27,i_b28,i_b29,i_b30`
- `i(38,37);ip(06);ib(27,28)` - supports mixed prefixes with parentheses

# Format to load metadata (from files named `info_devices.json`)

records format:
```json
{
№ прибора: [
  точка,
  глубина моря,
  глубина прибора от дна,
  символ модификации-конструкции,
  координаты (широта, долгота),

  ],
...
}
```
символ модификации-конструкции:
o	⯯ – отрицательной плавучести (с наиболее широким диапазоном)
o	⯭ – положительной плавучести, с широким диапазоном (с большим поплавком)
o	⭡ – положительной плавучести, с узким диапазоном измеряемой скорости течения
