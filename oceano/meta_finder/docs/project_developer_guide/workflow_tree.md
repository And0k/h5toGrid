# Workflow Tree

Maps the complete workflow tree of the meta_finder application, showing all possible execution branches and when each functionality is invoked.

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
collect.update_device_metadata_with_time_info()
├── Input: devices_data, device_dir
├── Condition: from_data=True AND device has no valid time metadata
├── Action: Extract time from prioritized data sources
│   ├── Source 1: text_output files (highest priority)
│   ├── Source 2: _raw directory files
│   └── Source 3: HDF5/MAT files (lowest priority)
└── Output: Updated devices_data with time_st, time_en, burst_dt, bursts_t
```

### C. HDF5 Extraction Branch

**Triggered:** When text_output and raw files are not available, and `extract_hdf5_times=True`

```
hdf5_processor.extract_metadata_from_hdf5()
├── Input: device_dir, device_id
├── Condition: No text/raw files found AND extract_hdf5_times=True
├── Action: Scan HDF5 files in priority order
│   ├── *.proc_noAvg.h5 (priority 1)
│   ├── *.proc.h5 (priority 2)
│   ├── *.proc_Avg.h5 (priority 2)
│   ├── _raw/*.h5 (priority 3)
│   └── _raw/*.mat (priority 3)
└── Output: time_st, time_en, coef_date (if available)
```

### D. Info File Creation Branch

**Triggered:** When `create_info_files=True`

```
create_info_files.update_devices_meta_file()
├── Input: cruise_dir, device_dirs
├── Condition: create_info_files=True
├── Action:
│   ├── Scan for existing info_devices@meta_finder.yaml
│   ├── Discover devices from data files
│   ├── Create/update YAML with placeholder values
│   └── Preserve existing non-placeholder values
└── Output: info_devices@meta_finder.yaml files
```

### E. Output Generation Branch

**Triggered:** Always (at end of processing)

```
file_writer.write_output_files()
├── Input: all_devices, cruise_data
├── Action:
│   ├── write_files_list() → {yymmdd_HHMM}_files_TCM.tsv
│   └── write_metadata_table() → {yymmdd_HHMM}_meta_TCM.tsv
└── Output: Two TSV files in meta directory
```

## Data Flow

```
Cruise Directory
    │
    ▼
Device Directory Discovery
    │
    ▼
Metadata File Reading (YAML/JSON)
    │
    ▼
Data File Discovery (text_output, _raw, HDF5)
    │
    ▼
Device-Data Association
    │
    ▼
Time Extraction (prioritized sources)
    │
    ▼
Field-Level Metadata Merging
    │
    ▼
Output Generation (TSV files)
```

## Conditional Branches

| Condition | Branch | Action |
|-----------|--------|--------|
| `create_info_files=True` | Info File Creation | Create/update `info_devices@meta_finder.yaml` |
| `from_data=True` | Data Extraction | Extract time metadata from data files |
| No text_output files | HDF5 Fallback | Extract from HDF5/MAT files |
| Existing time metadata | Skip Extraction | Preserve existing values |
| Multi-interval device | Multi-Row Output | Create one TSV row per interval |
| `overwrite_bad_devs_in_info_files=True` | Selective Update | Only update empty-value devices |

## See also

- [CLI Internals](CLI.md)
- [Processing Guide](../user_guide/processing.md)
- [Codebase Analysis](codebase_analysis.md)
- [Multiple Intervals Handling](multiple_intervals.md)
