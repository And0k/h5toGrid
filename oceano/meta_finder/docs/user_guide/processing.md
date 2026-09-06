# Processing Guide

Describes how meta_finder discovers devices, extracts metadata, and generates output.
For implementation details (function names, module architecture), see
[CLI Internals](../project_developer_guide/CLI.md).

## Processing pipeline

The main processing pipeline follows these stages:

1. **Cruise Directory Discovery**: Scan configured search directories for cruise directories using the pattern `YYMMDD_{cruise_name}`.

2. **Device Directory Discovery**: For each cruise directory, identify device subdirectories by matching:
   - Device keywords: `inclinometer`, `incl`, `tcm`, `wavegauge`, `wave_gauge`, `pres`, `@i[0-9]`
   - Device types (`i`, `w`, `incl`, `wg`) when followed by comma, semicolon, or end of string
   - The cruise directory itself if its name contains device identifiers and no device subdirectories exist
   - Date-named subdirectories (6-digit format) if they exist and no other device subdirectories are found

3. **Metadata File Discovery**: In each device directory, look for device metadata files in priority order:
   - First tries to read `info_devices@meta_finder.yaml`
   - Falls back to `info_devices.yaml` (replaces JSON if present)
   - Finally falls back to `info_devices.json` (deprecated)

4. **Data File Discovery**: For each device directory, discover all available data files:
   - Searches `text_output` directories and archives for data files (`.tsv`, `.txt`, `.csv`)
   - Uses fallback mechanism to extract device IDs from subdirectory names when filenames don't contain device information
   - Looks for files in `_raw` directories that match device naming patterns
   - Includes HDF5 files (proc_noAvg, proc, raw) if HDF5 extraction is enabled
   - Creates a mapping of device IDs to their associated data file paths

5. **Navigation File Discovery**: Search for `.gpx` navigation files in `*navigation*` or `*map*` subdirectories within device directories and cruise directories.

6. **Metadata Association and Extraction**: Associate devices with data files and extract temporal information:
   - Creates device data structure containing metadata and data_paths for all devices found
   - Gets prioritized data sources for time extraction for each device
   - Tries to extract time metadata from prioritized data sources until successful, supporting both text files and HDF5 files
   - Updates device metadata with time information while preserving all data paths for each device

7. **JSON and Data File Metadata Merging (Field-Level Preservation)**:
   When extracting metadata from data files, the system follows a **preserve-if-valid** policy for individual metadata fields:
   - Metadata file values are **preserved** when data file extraction returns placeholder values (`?`, `""`, `-`, `None`)
   - Only fields with valid extracted data overwrite metadata file values
   - This ensures existing burst parameters (`burst_dt`, `bursts_t`) and time ranges (`time_st`, `time_en`) are not lost when extraction fails or returns empty values

8. **Creating Combined File Comments**: Special comments are generated for devices in combined data files to indicate which devices are represented in each file.

9. **Extracting coordinates from GPX as fallback**: If coordinate metadata is missing.

10. **Output Generation**: Create two output files in the `meta` directory:
    - `meta/{yymmdd_HHMM}_files_TCM.tsv`: List of all processed files organized by cruise
    - `meta/{yymmdd_HHMM}_meta_TCM.tsv`: Tab-separated table with consolidated metadata for all devices

## Device discovery priority

The system discovers devices through multiple methods with the following processing approach for text output files, which are prioritized based on several criteria:

1. **Averaging interval priority**: Files with lower averaging intervals (binning seconds) have higher priority (2s files are prioritized over 600s files, which are prioritized over 7200s files)
2. **Files without averaging information**: Files without averaging information in their names are treated as having the configured default averaging value (typically 2.0001 seconds)
3. **Specificity**: Dedicated files (for specific devices) have higher priority than combined files
4. **Number of devices**: Files with fewer devices mentioned have higher priority
5. **Number of unmatched devices**: Files with fewer devices not present in the metadata files have higher priority

### Time range extraction priority

For each discovered device, the system extracts start and end time information from the corresponding data files in this order:

1. **Data File Name Parsing**: Primary source of device information from text output file names
2. **Raw Directory Search**: Secondary source when no text output files are found
3. **HDF5/MAT Fallback Support**: Final fallback when text_output and raw files are not available

The system skips HDF5 extraction when time metadata is already available in metadata files. If the metadata files already contain meaningful time information for `time_st`, `time_en`, or `coef_date` fields, the system will preserve the existing metadata and skip the potentially time-consuming HDF5 extraction process.

## Multiple intervals

For devices with multiple deployment intervals, the metadata file uses a nested YAML structure. The TSV output creates a **separate row for each interval**. See [Multiple intervals](../reference/io_formats.md#multiple-intervals) in the format specification for the exact YAML structure and behavior.

## See also

- [Getting Started](getting_started.md)
- [Input / Output Guide](input_output.md)
- [Configuration Guide](configuration.md)
- [Input / Output Format Specification](../reference/io_formats.md)
- [Multiple Intervals Handling](../project_developer_guide/multiple_intervals.md)
