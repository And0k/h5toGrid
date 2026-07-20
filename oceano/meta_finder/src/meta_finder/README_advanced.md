# Advanced Documentation

This document contains detailed information about device name patterns, HDF5 file structures, and advanced configuration options that have been moved from the main README to keep it more concise.

## Device Names and Patterns

**NOTE: This section describes default patterns defined in `config.py`. Modifying these patterns is NOT recommended as it may break device identification.**

### Default Device ID Patterns (from config.py)

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

### Device Name Normalization

Data files usually include device names in file name which is an abbreviation of device type, model and number. Underscores after type ("i" for inclinometer) or zeros before the first other number are not considered significant:

- `i_b27` → `ib27`
- `i_p06` → `ip06`
- `i_03` → `i03`
- `i_p06` → `ip6`

### Suffixes After Valid Device IDs

Default regex patterns automatically stop at first non-matching character, allowing suffixes/comments after valid device IDs without affecting operation.

**Patterns used:**
- `device_id_pattern` (for standard IDs)
- `r'@([a-z]+[0-9]+)'` (for @ prefix in subdirectories)

**Examples:**
- `@i03-comment` → extracts `i03`, ignores `-comment`
- `i02-data` → extracts `i02`, ignores `-data`
- `w05_test` → extracts `w05`, ignores `_test`

This allows adding descriptive suffixes to device names without breaking identification.

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

## HDF5 and MAT Files

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

### .proc.h5 files contain groups:
- averaging_bin (contains averaging information in name, e.g., bin600s)
- table (data table) with parameter name columns containing device_id suffixes, where device IDs are embedded in column names like "Vabs_i03", "v_i04", etc.

## Metadata format in devices metadata files

Contains records in the following order (lists for each device should contain at least the first 4 elements):
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

### Supported Metadata Formats

The system supports various formats for metadata arrays in devices metadata files:

1. **Variable Array Lengths**: Arrays can have different lengths, from minimum 4 elements (point, sea_depth, height_above_bottom, modification_symbol) up to 11+ elements. Missing elements beyond the provided length will be filled with default values ("?" for most fields, "" for burst_dt and bursts_t).

2. **Flexible Time Formats**: Time fields (positions 6 and 7) accept various ISO time formats:
   - `YYYY-MM-DD HH:MM:SS` (with space separator)
   - `YYYY-MM-DDTHH:MM:SS` (with T separator)
   - `YYYY-MM-DDTHH:MM` (without seconds)
   - `YYYY-MM-DDTHH:MM:SS.ffffff` (with microseconds)
   - Example formats: `"2019-12-10T14:22"`, `"2019-12-26 12:40"`, `"2019-12-10T14:22:00"`

3. The system handles `null` values in any position of the metadata array, treating them as missing/unknown values.

4. Fields can contain either numeric values or string representations, which will be handled appropriately during processing.

### Special JSON Metadata Format for Trailing Underscore Devices

- Device IDs ending with one or more underscores (e.g., `i10_`, `i10__`, `i10___`) are not normalized and treated as different devices in the program, however underscore suffixes are only to store additional metadata to
  - same device (`i10`)

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

## Advanced Configuration Options

### HDF5 Extraction Control

The `extract_hdf5_times` configuration parameter controls whether to extract time metadata from HDF5 files following a priority order:
- `extract_hdf5_coef_dates`: Whether to extract coefficient dates from HDF5 files (default: False)
- `raw_hdf5_cols`: Set of columns to trigger extraction of corresponding info from RAW HDF5/MAT files (from _raw/*.h5 and _raw/*.mat). Options include "coef_date" and "raw_date_range" (default: {"coef_date", "raw_date_range"})

### Device Discovery Priority Details

The system discovers devices through multiple methods with the following processing approach for text output files, which are prioritized based on several criteria:

1. **Averaging interval priority**: Files with lower averaging intervals (binning seconds) have higher priority (2s files are prioritized over 600s files, which are prioritized over 7200s files)
2. **Files without averaging information**: Files without averaging information in their names are treated as having the configured default averaging value from config.default_text_file_averaging (typically 2.0001 seconds), allowing them to be sorted normally based on this value
3. **Specificity**: Dedicated files (for specific devices) have higher priority than combined files
4. **Number of devices**: Files with fewer devices mentioned have higher priority
5. **Number of unmatched devices**: Files with fewer devices not present in JSON metadata have higher priority

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

The system skips HDF5 extraction when time metadata is already available in JSON files. If JSON metadata files already contain meaningful time information (values that are NOT placeholders like "?", "-", or empty strings) for `time_st`, `time_en`, or `coef_date` fields, the system will preserve the existing JSON metadata and skip the potentially time-consuming HDF5 extraction process.

### Time Range Extraction

For each discovered device, the system extracts start and end time information from the corresponding data files. Files that do not contain valid timestamp data in the expected format (YYYY-MM-DD HH:MM:SS) are ignored and not associated with devices.

### JSON and Data File Metadata Merging (Field-Level Preservation)

When extracting metadata from data files, the system follows a **preserve-if-valid** policy for individual metadata fields:
- JSON metadata values are **preserved** when data file extraction returns placeholder values (`?`, `""`, `-`, `None`)
- Only fields with valid extracted data overwrite JSON metadata
- This ensures existing burst parameters (`burst_dt`, `bursts_t`) and time ranges (`time_st`, `time_en`) are not lost when extraction fails or returns empty values
- Example: If JSON has `burst_dt: 120` but extraction returns `burst_dt: ""`, the value `120` is preserved

### Combined File Comments

Special comments are generated for devices in combined data files to indicate which devices are represented in each file.

## Dataset Name Construction

The `add_dataset_name()` function builds unique dataset names from cruise and device directory names. The **cruise name** is the pure expedition name extracted from the directory (without date prefix or device identifiers, e.g., `BalticSpit`, `Kulikovo`, `ABP53`). The resulting **dataset name** (output column `setup_name`) combines an optional date prefix with the cruise name and optional device suffix:

- For cruise names without digits, uses `/` separator between date prefix and name (e.g., `201202_BalticSpit` -> `2012/BalticSpit`)
- For cruise names with digits, uses `_` separator or no separator (e.g., `230616_Kulikovo` -> `2306/Kulikovo`)
- For duplicates with different dates, adds more date components (YYMM or YYMMDD) for differentiation
- For device directories with dates, extracts date from device subdirectory (e.g., `201202_BalticSpit/inclinometers/211008P7.5@i04` -> `2110/BalticSpit/P7.5`).

## Technical Limitations

### Time Precision

Текущая реализация извлекает только время с точностью до секунд, обрезая доли секунд из исходных данных. Для сохранения полной точности требуется модификация функции `extract_time_info_from_text_file` в `data_proc_funcs.py` для сохранения полного формата временных меток.

(Translation: Current implementation extracts time only with precision to seconds, truncating fractions of seconds from source data. To preserve full precision, modification of the `extract_time_info_from_text_file` function in `data_proc_funcs.py` is required to save the full time format.)

## VSZ Configuration File Loading

Loading metadata from `*.vsz`:
- coordinates from veusz commands like `AddCustom("*device*", "{"coord": (Lat, Lon)})"`
- time from
  - CustomDefinition variable "USEtime_{device_ID}" where device_ID comes from filename if device_ID="_"
  - configuration .yaml file by its path in commands like
  - `ImportFileCSV('../_raw/cfg_proc/probes/inclinometers.yaml', delimiter="'", headermode='none', linked=True, dsprefix='log_intervals', renames={'log_intervalscol1': 'area_time_none', 'log_intervalscol2': 'area_time_start', 'log_intervalscol3': 'area_time_comment'}, rowsignore=8, skipwhitespace=True)`
