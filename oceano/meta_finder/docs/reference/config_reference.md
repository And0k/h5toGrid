# Configuration Reference

Authoritative spec for all configuration fields. Defined in [`config.py`](../../src/meta_finder/config.py) (`Config` dataclass).

## `search_dirs`

List of directories to search for cruise data.

- **Type:** `Tuple[Path, ...]`
- **Default:** hardcoded paths (usually not what you want — specify explicitly or use `--cruise-dir`)
- **CLI:** `--search-dirs`

## `cruise_dir`

Specifies a single cruise directory to process, overriding `search_dirs`.

- **Type:** `Optional[List[Path]]`
- **Default:** `None`
- **CLI:** `--cruise-dir`

## `create_info_files`

Whether to create or update `info_devices@meta_finder.yaml` files.

- **Type:** `bool`
- **Default:** `False`
- **CLI:** `--create-info-files`

When True, the program scans cruise directories for device subdirectories, discovers devices from text_output files, _raw directories, or HDF5 files, and creates/updates `info_devices@meta_finder.yaml` with default "?" values for missing metadata.

## `from_data`

Whether to extract metadata from data files.

- **Type:** `bool`
- **Default:** `True`
- **CLI:** `--from-data` / `--no-from-data`

When True, extracts metadata (time ranges, device info, etc.) from data files and combines with existing info-file metadata. When False, only uses metadata from existing info-files without extracting from data files.

## `extract_hdf5_times`

Whether to extract time metadata from HDF5 files following the priority order.

- **Type:** `bool`
- **Default:** `True`
- **CLI:** `--extract-hdf5-times` / `--no-extract-hdf5-times`

When text_output files are not available, extracts from: `*.proc_noAvg.h5`, `*.proc.h5`, and `_raw/*.h5`.

## `extract_hdf5_coef_dates`

Whether to extract coefficient dates from HDF5 files.

- **Type:** `bool`
- **Default:** `False`
- **CLI:** `--extract-hdf5-coef-dates` / `--no-extract-hdf5-coef-dates`

## `max_burst_time_detection`

Maximum time in seconds for burst detection analysis.

- **Type:** `int`
- **Default:** `10800` (3 hours)
- **CLI:** `--max-burst-time-detection`

The code reads lines 1 and 20 to calculate time interval, then computes how many lines are needed to cover this time span.

## `default_text_file_averaging`

Default averaging value for text files that don't specify averaging.

- **Type:** `float`
- **Default:** `2.0001`
- **CLI:** `--default-text-file-averaging`

Files without averaging information in their names are treated as having this value, allowing them to be sorted normally. The value 2.0001 is slightly lower priority than 2s files.

## `device_dir_pattern`

Regex pattern for identifying standard device directories.

- **Type:** `str`
- **Default:** compiled from `ptn_device_dir_keywords` and `ptn_device_dir_sep`
- **CLI:** `--device-dir-pattern`

Matches device keywords (`inclinometer`, `incl`, `tcm`, `wavegauge`, `wave_gauge`, `pres`, `@i[0-9]`) or device types (`i`, `w`, `incl`, `wg`) after separators (`_`, `@`, `#`, digits, `-`). Device types require comma, semicolon, or end-of-string anchor.

## `output_format`

Output formats to generate.

- **Type:** `List[str]`
- **Default:** `["tsv"]`
- **CLI:** `--output-format`

## `output_dir`

Output directory for generated files.

- **Type:** `Optional[Path]`
- **Default:** `None` (uses "meta" directory relative to current working directory)
- **CLI:** `--output-dir`

## `overwrite_bad_devs_in_info_files`

Controls selective updating of individual device entries in info files.

- **Type:** `bool`
- **Default:** `True`
- **CLI:** `--overwrite-bad-devices-in-info-files` / `--no-overwrite-bad-devices-in-info-files`

Only devices with all empty values ("?", "-", or "") get updated with new information, while preserving existing non-placeholder values and device order.

## `raw_hdf5_cols`

Set of columns to trigger extraction of corresponding info from RAW HDF5/MAT files.

- **Type:** `set`
- **Default:** `{"coef_date", "raw_date_range"}`
- **CLI:** `--raw-hdf5-cols`

Options: "coef_date", "raw_date_range".

## `logging_level`

Global logging level setting.

- **Type:** `Union[str, int]`
- **Default:** `INFO` (20)
- **CLI:** `--logging-level`

Can be specified as a string (DEBUG, INFO, WARNING, ERROR, CRITICAL) or numeric value.

## `cache_files_number`

Cache configuration for file reading to minimize redundant file access.

- **Type:** `int`
- **Default:** `2000`
- **CLI:** `--cache-files-number`

## `temp_dir`

Temporary directory settings.

- **Type:** `Optional[Path]`
- **Default:** `None` (defaults to `src/meta_finder/temp`)
- **CLI:** `--temp-dir`

## Metadata file fields

The metadata file fields and their array indices are defined in [`io_info_files.py`](../../src/meta_finder/io_info_files.py) (`info_devices_field_names_extended`). See [Array element order](io_formats.md#array-element-order) in the Input / Output Format Specification for the full field table.

## See also

- [Configuration Guide](../user_guide/configuration.md)
- [Input / Output Format Specification](io_formats.md)
