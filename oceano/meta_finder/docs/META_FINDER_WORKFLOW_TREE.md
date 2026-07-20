# Meta Finder Workflow Tree & Execution Branches

## Overview
This document maps the complete workflow tree of the meta_finder application, showing all possible execution branches and when each functionality is invoked.

## Main Entry Points

### 1. Command Line Interface (`collect.py`)
```
main() → parse_command_line_args() → initialize_config()
├── collect() workflow
│   ├── file_finder.discover_device_dirs()
│   ├── collect.process_all_metadata()
│   │   ├── collect.get_absent_meta()
│   │   │   ├── file_finder.extract_devices_from_text_output()
│   │   │   ├── collect.get_all_data_files_for_device_dir()
│   │   │   └── collect.add_all_data_paths()
│   │   └── collect.update_device_metadata_with_time_info()
│   │       ├── collect.get_prioritized_data_sources_for_time_extraction()
│   │       └── collect.extract_time_metadata_from_prioritized_sources()
│   └── file_writer.write_output_files()
└── file_writer.generate_output()
```

### 2. Programmatic API
```
collect.process_cruise_directories()
├── file_finder.discover_device_dirs()
└── collect.process_all_metadata()
    └── [same as above]
```

## Execution Branches & Triggers

### A. Device Discovery Branch
**Triggered:** When processing cruise directories
```
file_finder.discover_device_dirs()
├── Input: top_search_dirs, input_dirs
├── Condition: Directory contains device-related patterns
│   ├── Pattern: inclinometer|incl|tcm|wavegauge|wave_gauge|pres|@i[0-9]?
│   ├── Pattern: ptn_device_dir_keywords
│   └── Pattern: ptn_device_dir_sep
├── Action: Scan subdirectories for device patterns
├── Output: Dictionary mapping cruise dirs to device dirs
└── Used by: process_all_metadata()
```

### B. Metadata Extraction Branch
**Triggered:** When `from_data=True` or when info files are missing
```
collect.get_absent_meta()
├── Input: meta_in (existing metadata), device_dir
├── Condition: Missing metadata in existing info files
├── Sub-branches:
│   ├── file_finder.extract_devices_from_text_output()
│   │   ├── Trigger: Text output files exist
│   │   ├── Action: Parse device IDs from filenames/content
│   │   └── Output: Device IDs and data paths
│   ├── collect.get_all_data_files_for_device_dir()
│   │   ├── Trigger: Need to find all data files
│   │   ├── Actions:
│   │   │   ├── file_finder.discover_datafiles_for_all_dev_in_dev_dir()
│   │   │   ├── hdf5_processor.find_hdf5_files()
│   │   │   └── file_finder.find_raw_directory_files()
│   │   └── Output: Organized data paths by device
│   └── collect.add_all_data_paths()
│       ├── Trigger: Need to associate data files with devices
│       └── Action: Build data_paths structure
└── Output: Complete metadata with data paths
```

### C. Time Extraction Branch
**Triggered:** When time metadata is needed
```
collect.update_device_metadata_with_time_info()
├── Input: devices_data (with data_paths)
├── Condition: extract_hdf5_times=True OR text files available
├── Sub-branches:
│   ├── collect.get_prioritized_data_sources_for_time_extraction()
│   │   ├── Priority order: text_output > proc.h5 > raw.h5 > other
│   │   └── Output: Prioritized list of data sources per device
│   └── collect.extract_time_metadata_from_prioritized_sources()
│       ├── Sub-sub-branches:
│       │   ├── data_proc_funcs.extract_time_info_from_text_file()
│       │   │   ├── Trigger: Text file available
│       │   │   ├── Actions:
│       │   │   │   ├── data_proc   _funcs.extract_time_range_from_file()
│       │   │   │   ├── data_proc_funcs.extract_burst_info_from_file()
│       │   │   │   └── data_proc_funcs.calculate_burst_statistics()
│       │   │   └── Output: time_st, time_en, burst_dt, bursts_t
│       │   ├── hdf5_processor.extract_time_range_from_hdf5_table()
│       │   │   ├── Trigger: HDF5 file available
│       │   │   └── Output: Time range from HDF5
│       │   └── hdf5_processor.extract_all_coef_dates_from_hdf5_files()
│       │       ├── Trigger: Coefficient dates needed
│       │       └── Output: Coef dates from HDF5
│       └── Output: Updated metadata with time information
└── Output: devices_data with time metadata
```

### D. File Creation Branch
**Triggered:** When `create_info_files=True` and files don't exist
```
create_info_files.update_devices_meta_file()
├── Input: device_dir, content
├── Condition: info_devices@meta_finder.json doesn't exist OR overwrite enabled
├── Sub-branches:
│   ├── Check existing file content
│   │   ├── io_info_files.all_vals_empty()
│   │   │   ├── True: Overwrite with new content
│   │   │   └── False: Merge selectively
│   │   └── create_info_files._merge_device_metadata()
│   │       ├── Action: Selective merge of placeholder values
│   │       └── Preserves: Non-placeholder existing values
│   ├── Format content → create_info_files._format_for_devices_meta_file()
│   │   ├── Convert dict to list format (11 elements)
│   │   ├── Handle nested dict structures (multiple intervals)
│   │   └── Convert datetime objects to strings
│   └── Write file → io_info_files.write_metadata_file()
│       ├── Sub-sub-branches:
│       │   ├── io_info_files.write_devices_meta_json()
│       │   │   ├── Trigger: File extension is .json
│       │   │   ├── Action: Write in JSON format
│       │   │   └── Uses: atomic_write() decorator
│       │   ├── io_info_files.write_devices_meta_yaml()
│       │   │   ├── Trigger: File extension is .yaml/.yml
│       │   │   ├── Action: Write in YAML format
│       │   │   ├── Uses: io_info_files.save_to_yaml_format()
│       │   │   └── Handles: Nested dict structures, comments, formatting
│       │   └── io_info_files.write_devices_meta_txt() (if implemented)
│       └── Output: Created info_devices@meta_finder.[json|yaml]
└── Result: Returns True if file was created/updated, False otherwise
```

### E. HDF5 Processing Branch
**Triggered:** When HDF5 files are present and extraction enabled
```
[find_and_process_hdf5_files()] (in various modules)
├── Input: device_dir, extract_hdf5_times, extract_hdf5_coef_dates
├── Condition: HDF5 files exist (.h5, .hdf5, .mat)
├── Sub-branches:
│   ├── hdf5_processor.find_hdf5_files()
│   │   ├── Action: Scan for HDF5 files in device directory
│   │   ├── Filters: extensions_hdf5 from config
│   │   └── Excludes: Raw subdirectories if configured
│   ├── hdf5_processor.extract_time_range_from_hdf5_table()
│   │   ├── Trigger: extract_hdf5_times=True
│   │   ├── Actions:
│   │   │   ├── Open HDF5 file
│   │   │   ├── Read time columns
│   │   │   ├── Calculate burst statistics
│   │   │   └── Handle time zone conversions
│   │   └── Output: time_st, time_en, burst_dt, bursts_t
│   ├── hdf5_processor.extract_all_coef_dates_from_hdf5_files()
│   │   ├── Trigger: extract_hdf5_coef_dates=True
│   │   ├── Action: Extract coefficient dates from HDF5 attributes
│   │   └── Output: coef_date metadata
│   └── hdf5_processor.extract_time_ranges_from_hdf5_combined()
│       ├── Trigger: Combined device HDF5 files
│       ├── Action: Handle multiple devices in single HDF5
│       └── Output: Time ranges for each device
└── Integration: Results fed back into main metadata structure
```

### F. Output Generation Branch
**Triggered:** When processing completes and outputs are needed
```
file_writer.write_output_files()
├── Input: processed metadata, output format specifications
├── Condition: Output generation enabled
├── Sub-branches:
│   ├── file_writer.write_metadata_table()
│   │   ├── Trigger: TSV output format requested
│   │   ├── Actions:
│   │   │   ├── Flatten nested metadata structure
│   │   │   ├── Handle multiple intervals per device
│   │   │   ├── Apply column formatting
│   │   │   └── Write to meta_TCM_*.tsv file
│   │   └── Output: TSV file with complete metadata
│   ├── [write_individual_device_files()] (if implemented)
│   │   └── Action: Create separate files per device
│   └── [write_summary_statistics()] (if implemented)
│       └── Action: Generate processing statistics
└── Result: Output files written to specified directory
```

### G. Validation & Error Handling Branches

#### Path Validation
```
[path_validate_device_paths()] (in path_checker.py)
├── Trigger: Before processing each device directory
├── Actions:
│   ├── Check directory exists
│   ├── Validate path format
│   ├── Verify read permissions
│   └── Sanitize path components
└── Continues: Only if path is valid
```

#### File Format Validation
```
io_info_files.read_metadata_file()
├── Trigger: Reading existing metadata files
├── Sub-branches:
│   ├── JSON format → json.load() with encoding retry
│   ├── YAML format → yaml.load() with encoding retry
│   └── TXT format → custom parsing
├── Error handling: Multiple encoding attempts
└── Output: Parsed metadata dictionary
```

#### Data Validation
```
[validate_extracted_data()] (in various modules)
├── Trigger: After data extraction
├── Actions:
│   ├── Check time ranges are valid
│   ├── Validate burst calculations
│   ├── Verify device ID formats
│   └── Confirm data consistency
└── Error handling: Log warnings, continue processing
```

## Conditional Execution Paths

### Configuration-Dependent Branches
- `create_info_files`: Controls info file creation branch
- `from_data`: Controls metadata extraction from data files
- `extract_hdf5_times`: Controls HDF5 time extraction
- `extract_hdf5_coef_dates`: Controls HDF5 coefficient date extraction
- `output_format`: Determines output generation format
- `overwrite_bad_devs_in_info_files`: Controls file overwrite behavior

### Data-Dependent Branches
- Available file types determine which extraction methods are used
- Device patterns in filenames determine parsing methods
- Existing metadata content determines merge vs. overwrite behavior
- Data quality determines fallback mechanisms used

## Integration Points

### Cross-Module Dependencies
1. `collect.py` → `file_finder.py`: Device directory discovery
2. `collect.py` → `data_proc_funcs.py`: Time extraction
3. `collect.py` → `hdf5_processor.py`: HDF5 processing
4. `collect.py` → `create_info_files.py`: File creation
5. `collect.py` → `file_writer.py`: Output generation
6. `create_info_files.py` → `io_info_files.py`: File I/O operations
7. `file_finder.py` → `parse_data_file_name.py`: Filename parsing
8. `data_proc_funcs.py` → `parse_data_file_name.py`: Device ID extraction

This workflow tree shows all possible execution paths and when each functionality is invoked based on configuration, input data, and runtime conditions.