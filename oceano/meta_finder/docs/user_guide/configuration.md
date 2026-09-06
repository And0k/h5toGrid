# Configuration Guide

The program uses a dataclass-based configuration system with command-line arguments.
For the exact configuration field spec, see [Configuration Reference](../reference/config_reference.md).
For implementation details, see [CLI Internals](../project_developer_guide/CLI.md).

## Key configuration parameters

- `search_dirs`: List of directories to search for cruise data (hardcoded default usually is not what you want: specify explicitly or overwrite with `cruise-dir` parameter)
- `cruise_dir`: Specifies a single cruise directory to process, overriding the `--search-dirs` option (default: None). Useful for debugging specific cruise directories.
- `create_info_files`: Whether to create or update `info_devices@meta_finder.yaml` files (default: False)
- `from_data`: Whether to extract metadata from data files. When True, extracts metadata (time ranges, device info, etc.) from data files and combines with existing metadata. When False, only uses metadata from existing metadata files (default: True)
- `extract_hdf5_times`: Whether to extract time metadata from HDF5 files following the priority order (when text_output files are not available): `*.proc_noAvg.h5`, `*.proc.h5`, and `_raw/*.h5` (default: True)
- `extract_hdf5_coef_dates`: Whether to extract coefficient dates from HDF5 files (default: False)
- `max_burst_time_detection`: Maximum time in seconds for burst detection analysis (default: 10800 = 3 hours). The code reads lines 1 and 20 to calculate time interval, then computes how many lines are needed to cover this time span.
- `default_text_file_averaging`: Default averaging value for text files that don't specify averaging (default: 2.0001 seconds to treat them lower priority files for extracting time than files with, usually sufficient, 2s averaging)
- `device_dir_pattern`: Regex pattern for identifying standard device directories. Matches device keywords (`inclinometer`, `incl`, `tcm`, `wavegauge`, `wave_gauge`, `pres`, `@i[0-9]`) or device types (`i`, `w`, `incl`, `wg`) after separators (`_`, `@`, `#`, digits, `-`). Device types require comma, semicolon, or end-of-string anchor.

## Other configuration parameters

- `output_dir`: Output directory for generated files (default: None, which uses the "meta" directory relative to the current working directory)
- `raw_hdf5_cols`: Set of columns to trigger extraction of corresponding info from RAW HDF5/MAT files (from `_raw/*.h5` and `_raw/*.mat`). Options include "coef_date" and "raw_date_range" (default: `{"coef_date", "raw_date_range"}`)
- `logging_level`: Global logging level setting. Can be specified as a string (DEBUG, INFO, WARNING, ERROR, CRITICAL) or numeric values (default: INFO, which is 20)
- `output_format`: Output formats to generate (default: `["tsv"]`)
- `overwrite_bad_devs_in_info_files`: Controls selective updating of individual device entries in info files: only devices with all empty values ("?", "-", or "") get updated with new information, while preserving existing non-placeholder values; preserves the order of devices in `info_devices@meta_finder.yaml` during updates (default: True)
- `cache_files_number`: Cache configuration for file reading to minimize redundant file access (default: 2000)
- `temp_dir`: Temporary directory settings. If not specified, defaults to `src/meta_finder/temp` (default: None)

All configuration parameters can be set via command-line arguments, with optional support for loading from a JSON config file using the `--config` option.

## Two-Run Requirement

When processing cruise directories for the first time (or when `info_devices@meta_finder.yaml` files don't exist), the program performs device discovery from data files and creates YAML files with placeholder values. However, **time data extraction only occurs for devices that are already listed in existing YAML metadata files** to review and manually edit device lists before time extraction.

This means you typically need to run the program **twice**:

1. **First run** (with `--create-info-files --no-from-data`):
   - Discovers all devices from data files (text_output, _raw, HDF5)
   - Creates `info_devices@meta_finder.yaml` files with all discovered devices
   - Devices have placeholder values ("?") for time fields

2. **Second run** (with `--create-info-files`):
   - Reads existing `info_devices@meta_finder.yaml` files
   - Extracts time data from data files for devices listed in YAML
   - Updates devices with actual time ranges (time_st, time_en, burst_dt, bursts_t)
   - Preserves all other metadata from the first run

## See also

- [CLI Guide](cli.md)
- [Input / Output Guide](input_output.md)
- [Processing Guide](processing.md)
- [Configuration Reference](../reference/config_reference.md) — exact config field spec
