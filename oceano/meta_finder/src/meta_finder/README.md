# Metadata Processor (meta_finder)

This tool collects metadata from files in Cruises directories.
Currently instruments supported: inclinometers (or tilt current meters, TCM), wave gauges, and combined device (inclinometer with pressure sensor) made by AB SIO RAS

## Usage

### Metadata collection process

Run the main script (`python collect.py` or in pixi environment: `pixi run collect`). This will generate two files with timestamp in their names in the `meta` directory relative to the current working directory:
1. `meta/{yymmdd_HHMM}_files_TCM.tsv` - List of all processed files
2. `meta/{yymmdd_HHMM}_meta_TCM.tsv` - Table with extracted metadata

### Command switches

Control discovery/use of metadata using other command switches.
Use `--interactive` or `-i` to prompt for each Config setting in order, entering `*` to use defaults for all remaining settings.
Configure whether to create info_devices@meta_finder.yaml files from existing data structures in devices data directories (using `--create-info-files` switch). Also available (via pixi) through `pixi run create-info-files` that runs `collect --create-info-files --no-from-data"`: create info_devices@meta_finder.yaml files from existing data structures, discovering device entries from data file names and structures without extracting metadata from the data files themselves.

## Configuration

The program uses a dataclass-based configuration system with command-line arguments. The configuration is defined in the `Config` dataclass in `config.py`.

### Key configuration parameters

- `search_dirs`: List of directories to search for cruise data (hardcoded default usually is not what you want: specify explicitly or overwrite with `cruise-dir` parameter)
- `cruise-dir`: Specifies a single cruise directory to process, overriding the `--search-dirs` option (default: None). This is useful for debugging specific cruise directories without processing all search directories (See Standard cruise directory structure below)
- `create_info_files`: Whether to create or update info_devices@meta_finder.yaml files (default: False)
- `from_data`: Whether to extract metadata from data files. When True, extracts metadata (time ranges, device info, etc.) from data files and combines with existing metadata. When False, only uses metadata from existing metadata files (default: True)
- `extract_hdf5_times`: Whether to extract time metadata from HDF5 files following the priority order (when text_output files are not available): *.proc_noAvg.h5, *.proc.h5, and _raw/*.h5 (default: True)
- `extract_hdf5_coef_dates`: Whether to extract coefficient dates from HDF5 files (default: False)
- `max_burst_time_detection`: Maximum time in seconds for burst detection analysis (default: 10800 = 3 hours). The code reads lines 1 and 20 to calculate time interval, then computes how many lines are needed to cover this time span.
- `default_text_file_averaging`: Default averaging value for text files that don't specify averaging (default: 2.0001 seconds to treat them lower priority files for extracting time than files with, usually sufficient, 2s averaging)
- `device_dir_pattern`: Regex pattern for identifying standard device directories. Matches device keywords (inclinometer, incl, tcm, wavegauge, wave_gauge, pres, @i[0-9]) or device types (i, w, incl, wg) after separators (_, @, #, digits, -). Device types require comma, semicolon, or end-of-string anchor. The code adds `^` anchor when needed for matching at start of string. Used by `find_device_dirs()` and `add_dataset_name()` to identify device directories and build dataset names.

### Other configuration parameters

- `output_dir`: Output directory for generated files (default: None, which uses the "meta" directory relative to the current working directory)
- `raw_hdf5_cols`: Set of columns to trigger extraction of corresponding info from RAW HDF5/MAT files (from _raw/*.h5 and _raw/*.mat). Options include "coef_date" and "raw_date_range" (default: {"coef_date", "raw_date_range"})
- `logging_level`: Global logging level setting. Can be specified as a string (DEBUG, INFO, WARNING, ERROR, CRITICAL) or numeric values (default: INFO, which is 20)
- `output_format`: Output formats to generate (default: ["tsv"]). Currently not useful to change: skip saving *.tsv possible only
- `overwrite_bad_devs_in_info_files`: Controls selective updating of individual device entries in info files: only devices with all empty values ("?", "-", or "") get updated with new information, while preserving existing non-placeholder values; preserves the order of devices in info_devices@meta_finder.yaml during updates (default: True)
- `cache_files_number`: Cache configuration for file reading to minimize redundant file access (default: 2000)
- `temp_dir`: Temporary directory settings. If not specified, defaults to `src/meta_finder/temp` (default: None)

All configuration parameters can be set via command-line arguments, with optional support for loading from a JSON config file using the `--config` option.

### Two-Run Requirement

When processing cruise directories for the first time (or when `info_devices@meta_finder.yaml` files don't exist), the program performs device discovery from data files and creates YAML files with placeholder values. However, **time data extraction only occurs for devices that are already listed in existing YAML metadata files** to review and manually edit device lists before time extraction.

This means you typically need to run the program **twice**:

1. **First run** (with `--create-info-files`):
   - Discovers all devices from data files (text_output, _raw, HDF5)
   - Creates `info_devices@meta_finder.yaml` files with all discovered devices
   - Devices have placeholder values ("?") for time fields
   - Program will show warning: "NEW DEVICES FOUND - REQUIRES SECOND RUN"

2. **Second run** (with `--create-info-files`):
   - Reads existing `info_devices@meta_finder.yaml` files
   - Extracts time data from data files for devices listed in YAML
   - Updates devices with actual time ranges (time_st, time_en, burst_dt, bursts_t)
   - Preserves all other metadata from the first run

## Default and test environments

Project package currently is set up using pixi in development editable mode (see pyproject.toml), so modules should be imported directly without needing to modify sys.path. The project uses default and "test" environment with additional `pytest`, `pytest-mock` and `pytest-arraydiff` packages for testing.

### Test Infrastructure

All tests are in the `tests/` directory.
The test suite includes a global output directory fixture that standardizes where test data read and output files are written:

- the test data directories are defined in test_data/Cruises by the tests/common_test_data_setup.py. If the .setup_complete marker file exists, so the fixture is not creating the new test directories (remove the marker file once you modified setup)

- **Global Output Directory**: All tests write their output files (like `{yymmdd_HHMM}_files_TCM.tsv` and `{yymmdd_HHMM}_meta_TCM.tsv`) to `test_data/meta_temp/` directory
- **Session-Level Cleanup**: The fixture automatically cleans up all files at the beginning of the test session to ensure a clean test environment
- **Session-Level Logging**: All log files are redirected to `test_data/meta_temp/logs/` for the entire test session to keep them centralized with other test outputs. For this use centralized logging configuration:
```python
from meta_finder.logging_config import setup_logging
logger = setup_logging(__name__, console_level=logging.DEBUG, file_level=logging.DEBUG)
```
- **Consistent Naming**: Test files have been renamed to follow a consistent pattern reflecting the specific functions they test

### Running Tests

To run the test suite, use the pixi "test" environment to ensure proper module imports and dependencies.

- Run all tests: `pixi run -e test python -m pytest tests/`
- Run specific test file: `pixi run -e test python -m pytest tests/test_combined_file_processing.py`
- Run tests with verbose output: `pixi run -e test python -m pytest tests/ -v`
- Skip debug test files: `pixi run -e test python -m pytest tests/ -k "not debug"`
- Run specific test file for create_info_files: `pixi run -e test python -m pytest tests/test_create_info_files_unit.py -v`

To see log messages when running tests:
```bash
pixi run -e test python -m pytest tests/your_test_file.py -v -s
```

## How It Works

1. **Cruise Directory Discovery**: The program scans configured search directories to find cruise directories using the pattern `YYMMDD_{cruise_name}` (`find_cruise_directories()`).

2. **Device Directory Discovery**: For each cruise directory, the program identifies device subdirectories using `find_device_dirs()`, which matches:
   - Device keywords (inclinometer, incl, tcm, wavegauge, wave_gauge, pres, @i[0-9])
   - Device types (i, w, incl, wg) when followed by comma, semicolon, or end of string
   - The cruise directory itself if its name contains device identifiers and no device subdirectories exist
   - Date-named subdirectories (6-digit format) if they exist and no other device subdirectories are found

3. **Metadata File Discovery**: In each device directory, the program looks for device metadata files in priority order:
   - First tries to read `info_devices@meta_finder.yaml`
   - Falls back to `info_devices.yaml` (replaces JSON if present)
   - Finally falls back to `info_devices.json` (deprecated)

4. **Data File Discovery**: For each device directory, the program discovers all available data files using `discover_datafiles_for_all_dev_in_dev_dir()` which:
   - Searches text_output directories and archives for data files (.tsv, .txt, .csv)
   - Uses fallback mechanism to extract device IDs from subdirectory names when filenames don't contain device information
   - Looks for files in _raw directories that match device naming patterns
   - Includes HDF5 files (proc_noAvg, proc, raw) if HDF5 extraction is enabled
   - Creates a mapping of device IDs to their associated data file paths

5. **Navigation File Discovery**: The program searches for `.gpx` navigation files in `*navigation*` or `*map*` subdirectories within device directories and cruise directories (`find_navigation_files()`).

6. **Metadata Association and Extraction**: The program uses `get_absent_meta()` to associate devices with data files and extract temporal information:
   - Creates device data structure containing metadata and data_paths for all devices found
   - Gets prioritized data sources for time extraction for each device using the `sort_data_paths()` function
   - Tries to extract time metadata from prioritized data sources until successful, supporting both text files and HDF5 files
   - Updates device metadata with time information while preserving all data paths for each device
   - Ensures all available data files are included for each device regardless of missing higher-priority files

7. **JSON and Data File Metadata Merging (Field-Level Preservation)**:
   When extracting metadata from data files, the system follows a **preserve-if-valid** policy for individual metadata fields:
   - Metadata file values are **preserved** when data file extraction returns placeholder values (`?`, `""`, `-`, `None`)
   - Only fields with valid extracted data overwrite metadata file values
   - This ensures existing burst parameters (`burst_dt`, `bursts_t`) and time ranges (`time_st`, `time_en`) are not lost when extraction fails or returns empty values
   - Example: If JSON has `burst_dt: 120` but extraction returns `burst_dt: ""`, the value `120` is preserved

8. **Creating Combined File Comments**: Special comments are generated for devices in combined data files to indicate which devices are represented in each file.

9. **Extracting coordinates from GPX as fallback**: If coordinate metadata is missing.

10. **Output Generation**: The program creates two output files in the `meta` directory:
   - `meta/{yymmdd_HHMM}_files_TCM.tsv`: List of all processed files organized by cruise, sorted by cruise name and device ID (`write_files_list()`)
   - `meta/{yymmdd_HHMM}_meta_TCM.tsv`: Tab-separated table with consolidated metadata for all devices, sorted by cruise name and device ID (`write_metadata_table()`)

The internal data structure organizes information as follows:
```json
cruise_path: {
    path of metadata file found (info_devices@meta_finder.yaml, info_devices.yaml, or info_devices.json as fallback): {
        device_name: {
          "data_paths": {(text_output_dir_path, data_file_relative_path): dataname_metadata},
          "gpx": (path to device gpx dir or, if not found, to cruise gpx dir),
          **metadata_from_json
        }
      }
    }
```

where
- `dataname_metadata` - metadata extracted from text output data file name
- `(text_output_dir_path, data_file_relative_path)` - keys sorted with described priority

Log files (`meta/{yymmdd_HHMM}_{meta_finder or mf_{abbreviated config}}.log`) contain WARNING and ERROR level messages.

## Key Functions

The implementation includes functions to replace the complex `get_absent_meta` function:

- `get_all_data_files_for_device_dir(device_dir)` - Find all data files in a device directory and organize them by device ID, including text output files (with fallback to subdirectory names and device validation), raw directory files, and HDF5 files if fallback is enabled
- `add_all_data_paths(meta_in, device_dir, cruise_name)` - Create device data structure containing metadata and data_paths for all devices found, ensuring all available data files are included regardless of missing higher-priority files
- `get_prioritized_data_sources_for_time_extraction(devices_data)` - Get prioritized data sources for time extraction for each device using the `sort_data_paths` function for proper prioritization
- `extract_time_metadata_from_prioritized_sources(device_id, prioritized_sources, ...)` - Try to extract time metadata from prioritized data sources until successful, with support for both text files and HDF5 files
- `update_device_metadata_with_time_info(devices_data, ...)` - Update devices_data with time metadata extracted from prioritized data sources while preserving all data paths for each device
- `get_absent_meta(meta_in, device_dir, ...)` - Search for metadata typically present in info_devices@meta_finder.yaml files (or info_devices.yaml/info_devices.json as fallback) but absent in input metadata dict, ensuring all available data files are included for each device regardless of whether higher-priority files are missing
- `process_all_metadata(cruise_and_its_dev_dirs, ...)` - Process all metadata from Cruises directories, which inherently handles the bug where missing text_output and .proc.h5 files cause other files to be excluded from data_paths

### create_info_files Module

The `create_info_files.py` module creates `info_devices@meta_finder.yaml` files for cruises with inclinometer or wavegauge directories. Key features include:
- Scanning cruise directories for device subdirectories
- Discovering devices from text_output files, _raw directories, or HDF5 files (with fallback support)
- Extracting time range information from data files
- Creating info_devices@meta_finder.yaml with default "?" values for missing metadata
- Selectively updating device entries (preserving their order) and their values: only those devices that have all empty values ("?", "-", or "") get updated with new information, while preserving existing non-placeholder values for the same devices
- Deduplication: if the merged content is identical to `info_devices.yaml` (after normalizing device IDs and comparing station data), the `@meta_finder.yaml` file is not written; if a pre-existing `@meta_finder.yaml` duplicates `info_devices.yaml`, it is deleted

## File Finding Functions

- `file_finder.find_cruise_directories(search_dirs)` - Finds all cruise directories in the specified search directories.
- `file_finder.find_device_dirs(cruise_dir)` - Finds device subdirectories in a cruise directory using `device_dir_pattern` which matches:
  - Device keywords (inclinometer, incl, tcm, wavegauge, wave_gauge, pres, @i[0-9])
  - Device types (i, w, incl, wg) when followed by comma, semicolon, or end of string
  Returns only matching subdirectories, or the cruise directory itself if it matches and no subdirectories exist, or dated subdirectories (6-digit format) if they exist within matched directories and no other device subdirectories are found.
- `file_finder.extract_devices_from_text_output(text_output_dir)` - extracts device IDs from data file names or content. Returns dictionary mapping device IDs to their associated files. Directory_path and archives treated seamlessly. Finds all data text files (.tsv, .txt, .csv) INSIDE. Returns a dictionary with values of the form {text_output_path or text_output_archive in posix format: [list of relative file paths inside in posix format]}
- `file_finder.discover_datafiles_for_all_dev_in_dev_dir(device_dir)` - Function to discover data files for all devices in a device directory

### Metadata Extraction Functions

- `metadata_extractor.read_metadata_files(json_path)` - Extracts metadata from metadata files in priority order: info_devices@meta_finder.yaml, info_devices.yaml (replaces JSON if present), or info_devices.json (deprecated).
- `data_proc_funcs.extract_time_info_from_text_file()` - Extracts start, end time from text data file. When called with a specified averaging interval, also extracts burst information (burst_dt and bursts_t) by analyzing gaps in timestamps. Supports time-split files (e.g., `140228_1300-28_2304@i03.txt`, `140301_0000-01_2304@i03.txt`) - automatically finds first/last files and combines their time ranges.
- `data_proc_funcs.extract_time_ranges_from_combined_file()` - Extracts time ranges for each device from combined data file. When averaging interval is provided, also extracts burst information.
- `parse_data_file_name.parse_filename_for_metadata(filename)` - Parses filename to extract device ID and averaging interval. Supports parentheses in filenames.
- `metadata_extractor.extract_coordinates_from_gpx(gpx_path, points)` - Extracts coordinates from `.gpx` file and adds them to existing points structure. Processes waypoints in GPX file and updates latitude/longitude for corresponding points. Returns updated points structure with coordinates or None if no points found.

### Archive Processing Functions

- Functions in `utils_sys.py` for direct archive reading:
  - `utils_sys.read_first_last_lines(archive_path, inner_file, skip_header)`: Reads first and last lines of a file inside a ZIP or 7z archive without full extraction
  - `utils_sys.list_archive_recursive(archive_path)`: Recursively lists all files/folders in ZIP or 7z archive, returning a flat list of all items including nested files

### Data Processing Functions

- `data_processor.get_h5_type_and_priority(file_path)` - Determines the type and priority of an HDF5 file based on its filename. Returns a tuple of (type, priority) where type is one of 'proc_noAvg', 'proc', or 'raw', and priority is an integer (1 for proc_noAvg, 2 for proc, 3 for raw).
- `data_processor.sort_input_dirs(input_dirs)` - Sorts input directories by priority order for processing. Returns sorted list of directories with higher priority first.
- `data_processor.sort_data_paths(data_paths, device_ids)` - Sorts data paths by priority for time extraction. Takes a dictionary of data paths and list of device IDs, returns sorted list of (priority, (dir_path, rel_path)) tuples sorted by priority.
- `collect.process_all_metadata()` - Main function for processing all metadata. Finds all cruise directories, info_devices*.json metadata files, text files and navigation files, then processes them for metadata extraction. Uses `get_absent_meta()` for associating devices with data files and extracting time information. Processes both single device files and combined files with data for multiple devices. Builds dataset names from cruise directory names using `add_dataset_name()`. After data association, performs JSON and text file metadata merging, as well as GPX file processing for coordinates.
- `parse_cruise_dir_name.add_dataset_name(device_dir, cruise_dir, ...)` - Builds a unique dataset name from cruise and device directory names, combining an optional date prefix with the cruise name and optional device suffix. The cruise name is the pure expedition name extracted from the directory (without date prefix or device identifiers). The resulting dataset name may include date components for disambiguation.
  - For cruise names without digits, uses `/` separator between date prefix and name (e.g., `201202_BalticSpit` -> `2012/BalticSpit`)
  - For cruise names with digits, uses `_` separator or no separator (e.g., `230616_Kulikovo` -> `2306/Kulikovo`)
  - For duplicates with different dates, adds more date components (YYMM or YYMMDD) for differentiation
  - For device directories with dates, extracts date from device subdirectory (e.g., `201202_BalticSpit/inclinometers/211008P7.5@i04` -> `2110/BalticSpit/P7.5`).

### File Writing Functions

- `file_writer.write_files_list(json_metadata, out_path: Path, write_1st_paths=False)` - Writes list of all collected pathsto `{yymmdd_HHMM}_files_TCM.tsv` (or only first (highest priority) data file path per device if `write_1st_paths=True`).
- `file_writer.write_metadata_table(metadata_list, meta_tcm_path, write_1st_paths=True)` - Writes metadata table to `{yymmdd_HHMM}_meta_TCM.tsv`. When `write_1st_paths=True` (default), writes only the first data file path in a `data_file_path` column; otherwise writes all paths in a `data_paths` column.
- `file_writer.find_latest_meta_file(output_dir)` - Finds the latest existing metadata file by pattern `meta_TCM_*.tsv` in the specified directory.
- `file_writer.load_existing_metadata(meta_file_path)` - Loads existing metadata from specified file for use when filling missing values.

### Helper Functions for Processing File Names

- `parse_data_file_name.expand_device_range(range_str)` - Expands device ranges, e.g. '27-30' to ['27', '28', '29', '30']. Also supports ranges with prefixes, like 'ib27-30' → ['ib27', 'ib28', 'ib29', 'ib30'].
- `parse_data_file_name.parse_device_group(group)` - Processes a group of devices separated by commas and ranges. Supports parentheses and complex patterns, such as 'i(38,37,59,60,58)' → ['i38', 'i37', 'i59', 'i60', 'i58'].
- `parse_data_file_name.parse_filename_for_metadata(filename)` - Parses filename to extract device ID and averaging interval, using a regular expression to match the pattern '{datetime}bin{interval}s{separator}{devices}.tsv'. Also recognizes combined data files without device suffixes (e.g., '{datetime}bin{interval}s.tsv') and files with device prefixes (e.g., '{datetime}bin{interval}s@i.tsv').

### Data Extracting Functions

- `data_proc_funcs.extract_time_info_from_text_file(dir_archive, rel_path, averaging_interval)` - Extracts time metadata from text data file, including time ranges and burst information (burst_dt and bursts_t) when averaging interval is provided.
- `data_proc_funcs.extract_time_ranges_from_combined_file(file_path, device_ids)` - Extracts time ranges for each device from combined data file containing data for multiple devices. Analyzes file header to determine columns corresponding to each device and finds start and end of data for each device. Handles both named column files (e.g., "Vabs (i01)") and combined files without device suffixes in column names or with suffixes containing multiple devices and creates special comments for them (for comment). Returns a dictionary with time ranges for devices and information about combined columns.
- `collect.get_absent_meta(meta_in, device_dir, ...)` - Creates content for saving metadata files. This function associates devices with data files and extracts temporal information, creating device data structure containing metadata and data_paths for all devices found.

Most functions with an input path argument accept pathlib.Path objects.

## Project Structure

The project is organized as follows:
```
src/
├── meta_finder/
│   ├── __init__.py
│   ├── config.py              # Configuration settings
│   ├── metadata_extractor.py  # Metadata extraction from JSON and GPX files
│   ├── file_finder.py         # File discovery in directory structures
│   ├── file_writer.py         # Output file generation
│   ├── data_processor.py      # High-level data processing orchestration
│   ├── data_proc_funcs.py     # Specific file processing functions
│   ├── collect.py # Functions for associating metadata and data files
│   ├── parse_data_file_name.py # Filename parsing functions
│   ├── utils_sys.py           # System-level utilities for archive processing
│   ├── collect.py       # Main application entry point
│   └── README.md              # This file
tests/                         # Test suite for TCM Metadata Processor
```

## Output Format

The `meta/{yymmdd_HHMM}_meta_TCM.tsv` file contains the following columns:
- `setup_name`: Dataset name — a unique identifier combining the cruise (expedition) name with an optional date prefix and device suffix for disambiguation (built by `add_dataset_name()`)
- `device_id`: Normalized identifier of the device (e.g., i7, w1)
- `point`: Station name or point number
- `sea_depth`: Sea depth at the deployment point (m)
- `height_above_bottom`: Device height relative to the seabed (m)
- `lat`: Latitude of the deployment point or ?
- `lon`: Longitude of the deployment point or ?
- `time_st`: Start date of processed data with accuracy to seconds
- `time_en`: End date of processed data with accuracy to seconds
- `burst_dt`: Device active interval in seconds (if device works periodically)
- `bursts_t`: Device operation period in seconds (if device works periodically)
- `data_file_path`: Path to the one processed text file on disk
- `quality`: quality: if two symbols: +- corresponds to absolute values and direction
- `comment`: Special comments including combined device information and GPX file paths

Format:
- Fields not found by the program are filled with "?" sign
- Values of burst_dt and bursts_t equal to "-", mean that the device operated normally without periodic shutdown,
  (such operation is not marked in the original metadata file or these fields are empty)
- Coordinates are recorded in degrees, however, if they were not found, then instead of `lat` and `lon`
  "?" is written, and paths to all found gpx files are written in the `comment` field with the prefix "GPX:"
- For combined data columns (e.g., Vabs_i05_14) in the `comment` field "{device1}+{device2} output" is written

## Cruises files structure

Supported also combined directory structures:
- `YYMMDD_{cruise_name}/{device_type}/info_devices.yaml` (standard)
- `YYMMDD_{cruise_name}/info_devices.yaml`
- `YYMMDD_{cruise_name}/inclinometers/{YYMMDD}*/info_devices.yaml`
Standard directory structure:
```
B:/WorkData/BalticSea/
└── YYMMDD_{cruise_name}/
    ├── *inclinometer* or *wavegauge*/
    │   ├── info_devices@meta_finder.yaml (fallback to info_devices.yaml, then info_devices.json)
    │   ├── text_output/
    │   │   └── *.tsv, *.txt, *.csv
    │   ├── text_output.zip
    │   └── text_output.7z
    └── *navigation* or *map*/
        └── *.gpx
```

## Device Discovery Priority and Return Format

The system discovers devices through multiple methods with the following processing approach for text output files, which are prioritized based on several criteria:

1. **Averaging interval priority**: Files with lower averaging intervals (binning seconds) have higher priority (2s files are prioritized over 600s files, which are prioritized over 7200s files)
2. **Files without averaging information**: Files without averaging information in their names are treated as having the configured default averaging value from config.default_text_file_averaging (typically 2.0001 seconds), allowing them to be sorted normally based on this value
3. **Specificity**: Dedicated files (for specific devices) have higher priority than combined files
4. **Number of devices**: Files with fewer devices mentioned have higher priority
5. **Number of unmatched devices**: Files with fewer devices not present in the metadata files have higher priority

The overall data processing priority order is:

1. **Data File Name Parsing**: Primary source of device information from text output file names in the `text_output` directory or archives. File names follow specific patterns that encode device identifiers:
   - `{datetime}bin{interval}s[@_#]{devices}.tsv` (original pattern with 2-5 digit time)
   - `{date}#{devices}-bin{interval}s.{ext}` (hash separator with binning)
   - `{date}#{devices}.{ext}` (hash separator without binning)
   - `{date}_{time}_{device}.{ext}` (simple device pattern)
   - `{date}_{time}@{device}.{ext}` (@ separator without binning)

   When text_output directory contains archives (.zip, .7z), the system looks for data files INSIDE those archives. The system processes both regular directories and archives in the text_output directory with the following priority:
   - Directories have higher priority than archives
   - Archives are sorted by name for consistent ordering
   - All directories and archives are treated equally during processing

2. **Raw Directory Search**: Secondary source when no text output files are found. The system searches for files in the `_raw` directory or archives matching these patterns:
   - Files with device type: `{prefix}?{type}{model}{number}.{ext}` where `{type}` is `i` or `w`, `{prefix}` can be `@`, `#`, `_`, or `-`
   - Files without device type (requires `@` or `#` prefix): `[@#]{number}.{ext}` - defaults to device type 'i' (inclinometer)
   - Supported extensions: `.txt`, `.tsv`, `.csv`, `.h5`, `.hdf5`, `.mat`
   - Examples: `#W1_130510.txt` → w1, `#1.txt` → i1, `i1.txt` → i1, `130510#W1_130510.txt` → w1

3. **HDF5/MAT Fallback Support**: Final fallback when text_output and raw files are not available. The system automatically falls back to HDF5/MAT files following this priority order (see "HDF5 files" section for details on expected structure). Works for all device types (inclinometers, wavegauges, etc.):
   - `device_dir/*.proc_noAvg.h5` - Processed HDF5 files without averaging
   - `device_dir/*.proc.h5` - Processed HDF5 files with averaging information (averaging extracted from filename pattern "bin{averaging_seconds}")
   - `device_dir/_raw/*.h5` and `device_dir/_raw/*.mat` - Raw HDF5 and MATLAB files (treated with raw priority)

   Also searches subdirectories containing whole word "h5" in their names (e.g., `h5_files`, `processed_h5`). Files in `*_raw` and its h5 subdirectories are treated as raw. Files in other h5 subdirectories are categorized by filename suffix.

The system skips HDF5 extraction when time metadata is already available in metadata files. If the metadata files already contain meaningful time information (values that are NOT placeholders like "?", "-", or empty strings) for `time_st`, `time_en`, or `coef_date` fields, the system will preserve the existing metadata and skip the potentially time-consuming HDF5 extraction process.

4. **Time Range Extraction**: For each discovered device, the system extracts start and end time information from the corresponding data files. Files that do not contain valid timestamp data in the expected format (YYYY-MM-DD HH:MM:SS) are ignored and not associated with devices.

Note: Directory names are only parsed to identify device directories (those containing "inclinometer" or "wavegauge"), not to extract specific device identifiers. Device identifiers are always extracted from data file names or content. The functions `extract_devices_from_text_output` and `discover_all_devices` return dictionaries mapping device IDs to their associated files in the format of (directory_path, file_path) tuples.

## Data Files

### Device Names

Data files usually include device names in the file name which is an abbreviation of the device type, model and number. Underscores after type ("i" for inclinometer) or zeros before the first other number are not considered significant:
- `i_b27` → `ib27`
- `i_p06` → `ip06`
- `i_03` → `i03`
- `i_p06` → `ip6`

### Device ID Patterns

Device IDs follow specific patterns defined in `config.py`:

- **device_type_pattern**: `r"incl|wg|[iw]"` - matches device type without number
  - Examples: `i`, `w`, `ib`, `ip`, `wb`, `wp`, `incl`, `wg`

- **device_model_pattern**: `r"[bp]?"` - optional model suffix
  - Examples: `b` (for ib, ip), `p` (for wb, wp)

- **device_number_pattern**: `r"\d+"` - device number

- **device_id_pattern**: `fr"(?:{device_type_pattern})(?:{device_model_pattern})?{device_number_pattern}"` - matches complete device IDs
  - Examples: `i01`, `i02`, `w01`, `w02`, `p01`, `ib27`, `ip6`, `incl01`, `wg02`

- **device_id_complex_pattern**: `r"\d*[@#_-]?(?P<type>{device_type_pattern})(?:ncl|nkl)?_?(?P<model>{device_model_pattern})0*(?P<number>{device_number_pattern})"` - for extracting device IDs with optional prefixes and separators
  - Uses named capturing groups for type, model, and number (constructed using building blocks)
  - Used for HDF5 group names and file names in `_raw` directory
  - Examples: `#W1_130510.txt` → `w1`, `#1.txt` → `i1`, `130510#W1_130510.txt` → `w1`

- **devices_in_text_output_files_pattern**: Flexible pattern for text output filenames
  - Supports device numbers without type prefix (e.g., `07,23,30,32` → `i07,i23,i30,i32`)
  - Handles comma-separated lists, ranges, and complex patterns

**Note**: Default patterns automatically stop at first non-matching character, allowing suffixes/comments after valid device IDs. See [README_advanced.md - Suffixes After Valid Device IDs](README_advanced.md#suffixes-after-valid-device-ids) for details.

### Fallback Mechanism for Device ID Extraction

When filenames don't contain device information, the system uses a fallback mechanism:

1. **Primary**: Extract device IDs from filename using `parse_filename_for_metadata()`
2. **Secondary**: If filename contains generic device types (`*`, `i`, `w`, `p`), read file content to resolve actual device IDs from column names
3. **Fallback**: If no devices found from content, extract device ID from subdirectory name using `_extract_device_id_from_subdirectory_name()`

Example fallback scenario:
```
text_output/130510/
├── i02/          # Device ID in subdirectory name
│   └── 130510_1100-10_2319.txt  # Filename doesn't contain device ID
└── W03/          # Device ID in subdirectory name
    └── 130510_1100-10_2319.txt  # Filename doesn't contain device ID
```

In this case, the system extracts `i02` and `w03` from subdirectory names and associates the files with these devices.

## Text Data Filenames

Pattern support:
`{Date}{separator}{bin{seconds}s}_{Device Names}.{ext}' - standard, but old data may have devices before seconds:
`{Date}#{devices}-bin{seconds}s.{ext}`

Any part except Date, which can be in `yymmdd_HHMM`, `yymmdd_HH` or `yymmdd` format, may be omitted. Separator - "@", "_" or "#". Seconds - binning interval (float).

Complex Patterns for Device Names is supported:

- Semicolon-Separated Groups separates different device groups: `i3,4,15,19,37,38;ib27-30,ip6.tsv`
- Range (numeric only) Expansion: `27-30` → `["27", "28", "29", "30"]`
- Parentheses Support:
  - `i(38,37,59,60,58).tsv` → `i38,i37,i59,i60,i58.tsv`
  - `i_b(27,28,29,30).tsv` → devices: `i_b27, i_b28, i_b29, i_b30`
  - `i(38,37);ip(06);ib(27,28).tsv` - supports mixed prefixes with parentheses

### Examples

`191210#07,23,30,32-bin300s.zip` → devices: i7, i23, i30, i32, bin = 300s
`200113_0000_i13.csv` → device: i13
`200113_00@i13.csv` → device: i13

#### Abbreviated generalizations
- Combined data files (e.g., '191108_120bin600s.tsv')
- Files with device prefixes (e.g., '191108_1200bin600s@i.tsv')

## Supported Metadata format in devices metadata files

Metadata files must be in yaml (or json but deprecated) format which contains "device_id": [array] structures

```json
{
  "device_id": [
    точка,  # 1
    глубина моря,  # 2
    глубина прибора от дна,  # 3
    символ модификации-конструкции,  # 4
    координаты: широта,  # 5
    координаты: долгота,  # 6
    время начала хороших данных,  # 7
    время конца хороших данных,  # 8
    burst_dt,  # 9
    bursts_t,  # 10
    комментарий  # 11
  ],
  ...
}
```

### "device_id"
devices id
- If it has leading spaces then it will be ignored (no data yet)
- Device IDs ending with one or more underscores (e.g., `i10_`, `i10__`, `i10___`) are not normalized and treated as different devices in the program, however underscore suffixes are only to store additional metadata to same device (`i10`).
This should be expressed in yaml format with one more level instead of modifying device_id (not tested in json):
`{device_id: {station_id: [array]}}`


### formats for elements in metadata arrays in devices metadata files
Contains records in the following order (lists for each device should contain at least the first 4 elements):

1. **Variable Array Lengths**: Arrays can have different lengths, from minimum 4 elements (point, sea_depth, height_above_bottom, modification_symbol) up to 11+ elements. Missing elements beyond the provided length will be filled with default values ("?" for most fields, "" for burst_dt and bursts_t).

2. **Flexible Time Formats**: Time fields (positions 6 and 7) accept various ISO time formats:
   - `YYYY-MM-DD HH:MM:SS` (with space separator)
   - `YYYY-MM-DDTHH:MM:SS` (with T separator)
   - `YYYY-MM-DDTHH:MM` (without seconds)
   - `YYYY-MM-DDTHH:MM:SS.ffffff` (with microseconds)
   - Example formats: `"2019-12-10T14:22"`, `"2019-12-26 12:40"`, `"2019-12-10T14:22:00"`

3. The system handles `null` values in any position of the metadata array, treating them as missing/unknown values.

4. Fields can contain either numeric values or string representations, which will be handled appropriately during processing.


### Metadata Comments

The `comment` field (11th element in the metadata array) may contain user comments from the JSON file. During data processing, the system combines comments from JSON with automatically generated comments (e.g., comments about combined devices or paths to GPX files), separating them with a semicolon. This allows preserving both user notes and automatically extracted metadata in the resulting `{yymmdd_HHMM}_meta_TCM.tsv` file.

Modification-symbol meanings:
- o ⯯ – negative buoyancy (with the widest range)
- o ⯭ – positive buoyancy, with wide range (with large float)
- o ⭡ – positive buoyancy, with narrow range of measured current velocity
- o ⤉ - positive buoyancy, with rod
- o ↑ - positive buoyancy, symbol not used

## Data format in `text_output` directories/archives

# Tilt current meter (inclinometer) data files contain the following titled data columns (with tab separator):
- Time – Kaliningrad time, `yyyy-mm-dd HH:MM:SS.ffffff`, (in older versions the column might also be named `index`)
- Vabs – absolute value of current velocity, m/s,
- Vdir – direction of current velocity, ° in geographic coordinate system,
- v – north component of current velocity, m/s,
- u – east component of current velocity, m/s,
- Inclination – inclinometer angle, °,
- Temp – temperature of inclinometer processor, °C.

# Wave gauge data files contain the following titled data columns (with tab separator):
- Time – Kaliningrad time, `yyyy-mm-dd HH:MM:SS.ffffff`,
- Pressure – pressure, dBar.

# Data files combining data from multiple devices

Usually contain a smaller set of columns for each device, data columns have device name suffix. Example:
```
index  v_i03  u_i03  Temp_i03  v_i04  u_i04  Temp_i04
```

## HDF5 and MAT files

MAT files (.mat) in `_raw` directories are processed as the same as HDF5 files with raw priority. They use the same HDF5 processing functions and structure.

### Non .proc.h5 files contain groups:
- device_id (can be normalized or not normalized, e.g., i07, w01, inclinometer07, @i07)
- table (data table) with columns by parameter names
- coef (only in raw directory files or with .raw.h5 suffix) - group with coefficients and their metadata

#### proc_noAvg.h5
```
/
├── i63/  # device_id_proc (normalized device name)
│   ├── table (columns: ['index', 'v', 'u', 'inclination', 'Battery', 'Temp'])
│   └── logFiles/
│       └── table (columns: ['index', 'fileName', 'fileChangeTime', 'DateEnd', 'DateProc'])
```

#### raw.h5
```
/
├── incl63/  # device_id_raw (not normalized device name) # normalized(device_id_raw) = normalized(device_id_proc)
│   ├── table (columns: ['index', 'Ax', 'Ay', 'Az', 'Mx', 'My', 'Mz', 'Battery', 'Temp'])
│   ├── coef/
│   └── logFiles/
│       └── table (columns: ['index', 'fileName', 'fileChangeTime', 'DateEnd', 'DateProc'])
```

### .proc_Avg.h5 files contain groups:
- device_id with averaging bin (e.g., `i04bin2s`, `i03bin600s`) - device IDs are in group names, not in column names
- table (data table) with parameter name columns WITHOUT device_id suffixes (columns like "Vabs", "v", "u", etc.)

Structure example:
```
/
├── i04bin2s/  # device_id with averaging bin
│   └── table (columns: ['index', 'Vabs', 'Vdir', 'v', 'u', 'Inclination', 'Temp'])
├── i05bin2s/  # another device with same averaging
│   └── table (columns: ['index', 'Vabs', 'Vdir', 'v', 'u', 'Inclination', 'Temp'])
└── i03bin600s/  # device with different averaging
    └── table (columns: ['index', 'Vabs', 'Vdir', 'v', 'u', 'Inclination', 'Temp'])
```

### .proc.h5 files contain groups:
- averaging_bin (contains averaging information in name, e.g., bin600s)
- table (data table) with parameter name columns containing device_id suffixes, where device IDs are embedded in column names like "Vabs_i03", "v_i04", etc.

## Advanced Documentation

For detailed information about device name patterns, HDF5 file structures, and advanced configuration options, see [README_advanced.md](README_advanced.md).
