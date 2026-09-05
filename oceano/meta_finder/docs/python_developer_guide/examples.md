# Examples

Usage examples for the meta_finder API.

## Basic usage

```python
from pathlib import Path
from meta_finder.collect import process_cruise_directories

# Process all cruise directories
process_cruise_directories(
    top_search_dirs=[Path("B:/Cruises/BalticSea")],
    create_info_files=False,
    from_data=True,
)
```

## Creating info files

```python
from meta_finder.create_info_files import update_devices_meta_file
from pathlib import Path

# Create/update info_devices@meta_finder.yaml for a specific cruise
update_devices_meta_file(
    cruise_dir=Path("B:/Cruises/BalticSea/250415_ABP60"),
    from_data=False,  # Only create placeholders
)
```

## Extracting time metadata

```python
from meta_finder.data_proc_funcs import extract_time_info_from_text_file
from pathlib import Path

# Extract time info from a text file
time_info = extract_time_info_from_text_file(
    dir_archive=Path("text_output"),
    rel_path="191108_1200bin600s@i03.tsv",
    averaging_interval=600.0,
)
print(time_info)
# {'time_st': '2019-11-08 12:00:00', 'time_en': '2019-11-08 18:00:00', ...}
```

## Parsing filenames

```python
from meta_finder.parse_data_file_name import parse_filename_for_metadata, normalize_device_id

# Parse a filename
meta = parse_filename_for_metadata("191210#07,23,30,32-bin300s.zip")
print(meta)
# {'datetime': '191210', 'averaging_interval': 300, 'devices': ['i7', 'i23', 'i30', 'i32']}

# Normalize a device ID
normalize_device_id("i_03")  # Returns 'i3'
normalize_device_id("i_b27")  # Returns 'ib27'
```

## Sorting data paths

```python
from meta_finder.data_processor import sort_data_paths

# Sort data paths by priority
data_paths = {
    "i03": {
        ("text_output", "191108_1200bin600s@i03.tsv"): {"averaging_interval": 600},
        ("text_output", "191108_1200bin2s@i03.tsv"): {"averaging_interval": 2},
    }
}
sorted_paths = sort_data_paths(data_paths, ["i03"])
# Higher priority (2s) file comes first
```

## See also

- [CLI Guide](../user_guide/cli.md)
- [Configuration Guide](../user_guide/configuration.md)
- [Processing Guide](../user_guide/processing.md)
