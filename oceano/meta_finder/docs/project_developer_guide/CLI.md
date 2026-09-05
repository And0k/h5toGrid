# CLI Internals

## Module Architecture

```
src/meta_finder/
├── __init__.py
├── config.py              ← Configuration settings, CLI arguments, constants
├── collect.py             ← Main application entry point, process_all_metadata()
├── file_finder.py         ← File discovery in directory structures
├── metadata_extractor.py  ← Metadata extraction from YAML/JSON and GPX files
├── data_processor.py      ← High-level data processing orchestration
├── data_proc_funcs.py     ← Specific file processing functions (time extraction)
├── file_writer.py         ← Output file generation (TSV)
├── hdf5_processor.py      ← HDF5/MAT file metadata extraction
├── io_info_files.py       ← YAML/JSON info file I/O
├── parse_data_file_name.py ← Filename parsing and device ID normalization
├── parse_cruise_dir_name.py ← Cruise directory name parsing and dataset naming
├── create_info_files.py   ← info_devices@meta_finder.yaml creation/updating
├── similarity.py          ← vendored hierarchical folder-name similarity (no external deps)
├── post_processing/       ← formally integrated utilities (shipped in wheel)
│   ├── path_checker.py      ← TSV path availability + renamed-folder mapping
│   ├── path_checker_main.py ← `check-paths` CLI (`python -m meta_finder.post_processing.path_checker_main`)
│   └── check_device_dirs.py ← device-dir availability report + symlink hierarchy (`check-device-dirs`)
├── utils_sys.py           ← System-level utilities for archive processing
└── README.md              ← Legacy (content redistributed to docs/)
```

## Entry point

`collect.py` — main entry point using `argparse` for CLI parsing.

### CLI

```bash
# Process all cruise directories
pixi run collect

# Process a specific cruise directory
pixi run collect --cruise-dir "B:/Cruises/BalticSea/250415_ABP60"

# Create/update info_devices@meta_finder.yaml files
pixi run collect --create-info-files --no-from-data

# Extract time metadata from data files
pixi run collect --create-info-files

# Interactive mode
pixi run collect --interactive
```

### Flow

1. `main()` → `parse_command_line_args()` → `initialize_config()`
2. `collect()` workflow:
   - `file_finder.discover_device_dirs()`
   - `collect.process_all_metadata()`
     - `collect.get_absent_meta()`
       - `file_finder.extract_devices_from_text_output()`
       - `collect.get_all_data_files_for_device_dir()`
       - `collect.add_all_data_paths()`
     - `collect.update_device_metadata_with_time_info()`
       - `collect.get_prioritized_data_sources_for_time_extraction()`
       - `collect.extract_time_metadata_from_prioritized_sources()`
   - `file_writer.write_output_files()`

### Programmatic API

```python
from meta_finder.collect import process_cruise_directories

process_cruise_directories(
    top_search_dirs=[Path("B:/Cruises/BalticSea")],
    create_info_files=True,
    from_data=True,
)
```

## Key Functions

### File discovery

- `file_finder.find_cruise_directories(search_dirs)` — Finds all cruise directories in the specified search directories.
- `file_finder.find_device_dirs(cruise_dir)` — Finds device subdirectories in a cruise directory.
- `file_finder.extract_devices_from_text_output(text_output_dir)` — Extracts device IDs from data file names or content.
- `file_finder.discover_datafiles_for_all_dev_in_dev_dir(device_dir)` — Discovers data files for all devices in a device directory.

### Metadata extraction

- `metadata_extractor.read_metadata_files_to_dict(json_path)` — Extracts metadata from metadata files in priority order: `info_devices@meta_finder.yaml`, `info_devices.yaml`, or `info_devices.json`.
- `data_proc_funcs.extract_time_info_from_text_file(dir_archive, rel_path, averaging_interval)` — Extracts start, end time from text data file. Also extracts burst information when averaging interval is provided.
- `data_proc_funcs.extract_time_ranges_from_combined_file(file_path, device_ids)` — Extracts time ranges for each device from combined data file.
- `hdf5_processor.extract_time_range_from_hdf5_table()` — Extracts time metadata from HDF5 tables.

### Data processing

- `data_processor.sort_data_paths(data_paths, device_ids)` — Sorts data paths by priority for time extraction.
- `data_processor.get_h5_type_and_priority(file_path)` — Determines the type and priority of an HDF5 file.
- `collect.get_absent_meta(meta_in, device_dir, ...)` — Creates content for saving metadata files.
- `collect.process_all_metadata(cruise_and_its_dev_dirs, ...)` — Main function for processing all metadata.

### File writing

- `file_writer.write_files_list(json_metadata, out_path, write_1st_paths)` — Writes list of all collected paths to `{yymmdd_HHMM}_files_TCM.tsv`.
- `file_writer.write_metadata_table(metadata_list, meta_tcm_path, write_1st_paths)` — Writes metadata table to `{yymmdd_HHMM}_meta_TCM.tsv`.

### Helper functions

- `parse_data_file_name.parse_filename_for_metadata(filename)` — Parses filename to extract device ID and averaging interval.
- `parse_data_file_name.normalize_device_id(device_id)` — Normalizes device ID (removes underscores, lowercases, etc.).
- `parse_cruise_dir_name.add_dataset_name(device_dir, cruise_dir, ...)` — Builds a unique dataset name from cruise and device directory names.
- `utils_sys.read_first_last_lines(archive_path, inner_file, skip_header)` — Reads first and last lines of a file inside ZIP or 7z archive.

## Data structures

The internal data structure organizes information as follows:

```python
cruise_path: {
    path of metadata file found: {
        device_name: {
            "data_paths": {(text_output_dir_path, data_file_relative_path): dataname_metadata},
            "gpx": (path to device gpx dir or cruise gpx dir),
            **metadata_from_json
        }
    }
}
```

where:
- `dataname_metadata` — metadata extracted from text output data file name
- `(text_output_dir_path, data_file_relative_path)` — keys sorted with described priority

## See also

- [Getting Started](../user_guide/getting_started.md)
- [Processing Guide](../user_guide/processing.md)
- [Codebase Analysis](codebase_analysis.md)
- [Workflow Tree](workflow_tree.md)
