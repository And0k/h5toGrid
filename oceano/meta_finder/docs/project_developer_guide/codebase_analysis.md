# Codebase Analysis

Comprehensive analysis of the meta_finder codebase, mapping each module to its functionality and corresponding test coverage.

## Source Modules & Functionality

### 1. `config.py`

**Functionality:** Configuration management, CLI arguments, constants, and application settings
- Defines global constants and configuration parameters
- Handles command-line argument parsing
- Manages application-wide settings and defaults

**Test Coverage:**
- `test_command_line_args.py` — Tests CLI argument parsing
- `test_interactive_config.py` — Tests interactive configuration features

### 2. `parse_data_file_name.py`

**Functionality:** Filename parsing, device ID extraction, and pattern matching
- Parses device IDs from filenames and directory names
- Handles complex device patterns (ranges, groups, special characters)
- Extracts metadata from filename components
- Normalizes device IDs across different formats

**Test Coverage:**
- `test_parse_filename_patterns.py` — Tests pattern matching logic
- `test_device_extraction.py` — Tests device ID extraction from various sources
- `test_device_id_normalization_fix.py` — Tests device ID normalization
- `test_duplicate_device_id_fix.py` — Tests duplicate device ID handling

### 3. `parse_cruise_dir_name.py`

**Functionality:** Cruise directory name parsing
- Extracts cruise information from directory names
- Handles various naming conventions
- Validates cruise directory structure
- Builds dataset names from cruise and device directory names

**Test Coverage:**
- `test_extract_cruise_name.py` — Tests cruise name extraction
- Integrated in `test_device_directory_logic.py`

### 4. `file_finder.py`

**Functionality:** File discovery, device directory finding, and path resolution
- Discovers device directories in cruise structures
- Implements file searching algorithms
- Handles exclusion patterns and filters
- Resolves device patterns from file paths
- Extracts devices from text output directories and archives

**Test Coverage:**
- `test_device_directory_logic.py` — Tests directory discovery logic
- `test_file_finder_gpx_filtering.py` — Tests GPX file filtering
- `test_excluded_dirs.py` — Tests directory exclusion logic
- `test_file_finder_fix.py` — Tests file finder fixes

### 5. `metadata_extractor.py`

**Functionality:** Metadata file reading and parsing
- Reads `info_devices@meta_finder.yaml`, `info_devices.yaml`, and `info_devices.json`
- Extracts device metadata from YAML/JSON structures
- Handles multiple intervals (underscore suffixes)
- Normalizes time formats
- Extracts coordinates from GPX files

**Test Coverage:**
- `test_read_metadata_files.py` — Tests metadata file reading
- `test_info_devices_yaml_parsing.py` — Tests YAML parsing
- `test_multiple_intervals_handling.py` — Tests multi-interval devices

### 6. `data_processor.py`

**Functionality:** High-level data processing orchestration
- Sorts data paths by priority for time extraction
- Determines HDF5 file types and priorities
- Orchestrates data source selection

**Test Coverage:**
- `test_sort_data_paths.py` — Tests data path sorting
- `test_h5_type_and_priority.py` — Tests HDF5 file classification

### 7. `data_proc_funcs.py`

**Functionality:** Specific file processing functions
- Extracts time metadata from text files
- Extracts time ranges from combined files
- Handles time-split files (multiple files for same device)
- Analyzes burst patterns in timestamps

**Test Coverage:**
- `test_extract_time_info.py` — Tests time extraction from text files
- `test_extract_time_ranges_combined.py` — Tests combined file time extraction

### 8. `hdf5_processor.py`

**Functionality:** HDF5/MAT file metadata extraction
- Extracts device IDs from HDF5 group names
- Extracts time ranges from HDF5 tables
- Extracts coefficient dates from HDF5 coef groups
- Handles both normalized and non-normalized group names

**Test Coverage:**
- `test_hdf5_extraction.py` — Tests HDF5 metadata extraction
- `test_hdf5_coef_date.py` — Tests coefficient date extraction

### 9. `io_info_files.py`

**Functionality:** YAML/JSON info file I/O
- Reads and writes `info_devices@meta_finder.yaml` files
- Handles multiple intervals (underscore suffixes)
- Converts between list/tuple and dict formats
- Preserves device order during updates

**Test Coverage:**
- `test_io_info_files.py` — Tests info file I/O
- `test_multiple_intervals_handling.py` — Tests multi-interval handling

### 10. `file_writer.py`

**Functionality:** Output file generation
- Writes `{yymmdd_HHMM}_files_TCM.tsv` — list of processed files
- Writes `{yymmdd_HHMM}_meta_TCM.tsv` — consolidated metadata table
- Handles multi-interval devices (one row per interval)
- Computes quality indicators from data file paths

**Test Coverage:**
- `test_file_writer.py` — Tests TSV output generation
- `test_write_metadata_table.py` — Tests metadata table writing

### 11. `collect.py`

**Functionality:** Main application entry point and processing orchestration
- Orchestrates the entire metadata collection pipeline
- Associates devices with data files
- Extracts temporal information from prioritized sources
- Handles multi-interval devices
- Manages field-level metadata merging

**Test Coverage:**
- `test_collect.py` — Tests main collection pipeline
- `test_get_absent_meta.py` — Tests metadata association
- `test_process_all_metadata.py` — Tests full metadata processing

### 12. `create_info_files.py`

**Functionality:** `info_devices@meta_finder.yaml` creation and updating
- Scans cruise directories for device subdirectories
- Discovers devices from text_output files, _raw directories, or HDF5 files
- Extracts time range information from data files
- Creates/updates info files with default "?" values
- Selectively updates device entries (preserving order and non-placeholder values)
- Deduplication against existing `info_devices.yaml`

**Test Coverage:**
- `test_create_info_files_unit.py` — Tests info file creation
- `test_create_info_files_integration.py` — Tests full workflow

### 13. `utils_sys.py`

**Functionality:** System-level utilities for archive processing
- Reads first and last lines of files inside ZIP or 7z archives
- Recursively lists all files/folders in archives
- Handles archive path resolution

**Test Coverage:**
- `test_utils_sys.py` — Tests archive utilities

### 14. `parse_data_file_name.py`

**Functionality:** Filename parsing and device ID extraction
- Parses device IDs from filenames using regex
- Expands device ranges (e.g., `27-30` → `27,28,29,30`)
- Processes semicolon-separated groups
- Handles parentheses and complex patterns
- Normalizes device IDs

**Test Coverage:**
- `test_parse_filename_patterns.py` — Tests pattern matching
- `test_device_id_normalization_fix.py` — Tests normalization
- `test_duplicate_device_id_fix.py` — Tests duplicate handling

### 15. `parse_cruise_dir_name.py`

**Functionality:** Cruise directory name parsing and dataset naming
- Extracts cruise names from directory names
- Builds unique dataset names with date prefixes
- Handles duplicate cruise names with different dates
- Extracts dates from device subdirectories

**Test Coverage:**
- `test_extract_cruise_name.py` — Tests cruise name extraction
- `test_add_dataset_name.py` — Tests dataset name construction

---

### 16. `similarity.py`

**Functionality:** Vendored hierarchical folder-name similarity (zero external deps)
- Pure-python port of the `match_dirs` name-similarity used by `post_processing`
- `hierarchical_weighed_similarity()` + `HIGH/LOW_CONFIDENCE_THRESHOLD`

### 17. `post_processing/`

**Functionality:** Formally integrated post-collection utilities (shipped in wheel)
- `path_checker.py` — TSV path availability + renamed-folder mapping
- `path_checker_main.py` — `check-paths` CLI entry (`python -m meta_finder.post_processing.path_checker_main`)
- `check_device_dirs.py` — device-dir availability report + symlink hierarchy (`check-device-dirs`)

See [Path Checker Guide](../user_guide/path_checker.md).

## See also

- [CLI Internals](CLI.md)
- [Workflow Tree](workflow_tree.md)
- [Multiple Intervals Handling](multiple_intervals.md)
- [HDF5 Functionality](hdf5_functionality.md)
