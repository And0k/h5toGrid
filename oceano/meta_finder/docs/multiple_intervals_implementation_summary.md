# Multiple Intervals Handling Implementation Summary

## Problem Statement

Devices with multiple time intervals (represented with underscore suffixes like `i10`, `i10_`, `i10__`) were being incorrectly handled. The system was outputting the same last time for all intervals in the TSV file instead of showing the correct time range for each interval.

**Example Issue:**
```yaml
"i10":
  0: ["", 85, 0, "⭡", 55.8940, 19.0899, "2018-10-17 16:30:00", "2018-10-18 07:15:00"]
  1: ["", 85, 0, "⭡", 55.8940, 19.0899, "2018-10-22T12:03:00", "2018-10-27T06:47:28"]
```

**Incorrect Output:**
```
ABP44	i10		85	0	55.894	19.0899	2018-10-17 16:30:00	2018-10-27 06:47:26	...
ABP44	i10_		85	0	55.894	19.0899	2018-10-22 12:03:00	2018-10-27 06:47:28	...
```

**Expected Output:**
```
ABP44	i10		85	0	55.894	19.0899	2018-10-17 16:30:00	2018-10-18 07:15:00	...
ABP44	i10		85	0	55.894	19.0899	2018-10-22 12:03:00	2018-10-27 06:47:28	...
```

## Solution Approach

The solution implements a three-layer approach:

1. **Metadata Reading Layer** ([`io_info_files.py`](src/meta_finder/io_info_files.py:267-345)): Flatten multiple intervals at metadata reading stage by converting devices with underscore suffixes into a single device entry with an `intervals` list

2. **Processing Layer** ([`collect.py`](src/meta_finder/collect.py:820-865)): Update time extraction logic to handle the new intervals structure

3. **Output Layer** ([`file_writer.py`](src/meta_finder/file_writer.py:131-221)): Modify TSV writer to create one row per interval with correct time values

## Implementation Details

### 1. Metadata Reading Layer (`io_info_files.py`)

Modified [`_ungroup_devices_with_underscore_suffixes`](src/meta_finder/io_info_files.py:267-345) to:
- Detect devices with multiple intervals (multiple underscore suffixes for same base name)
- Create an `intervals` list containing time-specific data for each interval
- Extract non-time fields (sea_depth, lat, lon, etc.) from the base device
- Handle both list/tuple and dict formats for device metadata

**Key Changes:**
- Added tuple/list to dict conversion for compatibility with existing YAML format
- Separated time fields from non-time fields in the intervals structure
- Maintained backward compatibility with single-interval devices

### 2. Processing Layer (`collect.py`)

Modified [`update_device_metadata_with_time_info`](src/meta_finder/collect.py:820-865) to:
- Check if device has an `intervals` structure
- Update time fields in the first interval for multi-interval devices
- Update time fields directly in the device entry for single-interval devices
- Only update fields when extracted values are not placeholders

**Key Changes:**
- Conditional logic to handle both single and multi-interval devices
- Preserves the interval structure while updating time information

### 3. Output Layer (`file_writer.py`)

Modified [`write_metadata_table`](src/meta_finder/file_writer.py:92-243) to:
- Check if device has an `intervals` list
- Create one TSV row per interval for multi-interval devices
- Copy non-time fields from device metadata to each interval row
- Copy `data_paths` from device to each interval row
- Add `interval_index` field to track interval number

**Key Changes:**
- Helper function `_process_device_entry` to avoid code duplication (DRY principle)
- Proper ordering of operations: populate `all_devices` first, then process data paths
- Maintains backward compatibility with single-interval devices

## Data Structure

### Multi-Interval Device Structure
```python
{
    "i10": {
        "point": "",
        "sea_depth": 85,
        "height_above_bottom": 0,
        "modification_symbol": "⭡",
        "lat": 55.8940,
        "lon": 19.0899,
        "burst_dt": "",
        "bursts_t": "",
        "comment": "",
        "coef_date": "",
        "time_raw_st": "",
        "time_raw_en": "",
        "intervals": [
            {
                "time_st": "2018-10-17 16:30:00",
                "time_en": "2018-10-18 07:15:00",
                "burst_dt": "",
                "bursts_t": "",
                "coef_date": "",
                "time_raw_st": "",
                "time_raw_en": "",
            },
            {
                "time_st": "2018-10-22 12:03:00",
                "time_en": "2018-10-27 06:47:28",
                "burst_dt": "",
                "bursts_t": "",
                "coef_date": "",
                "time_raw_st": "",
                "time_raw_en": "",
            },
        ],
        "data_paths": {},
        "cruise": "ABP44",
    }
}
```

### Single-Interval Device Structure
```python
{
    "i07": {
        "point": "",
        "sea_depth": 80,
        "height_above_bottom": 10,
        "modification_symbol": "⭡",
        "lat": 57.3456,
        "lon": 21.2345,
        "burst_dt": "",
        "bursts_t": "",
        "comment": "",
        "coef_date": "",
        "time_raw_st": "",
        "time_raw_en": "",
        "time_st": "2018-10-20 10:00:00",
        "time_en": "2018-10-25 15:00:00",
        "data_paths": {},
        "cruise": "ABP44",
    }
}
```

## Field Classification

### Time Fields (vary per interval)
- `time_st` - Start time
- `time_en` - End time
- `burst_dt` - Burst datetime
- `bursts_t` - Burst time
- `coef_date` - Coefficient date
- `time_raw_st` - Raw start time
- `time_raw_en` - Raw end time

### Non-Time Fields (shared across intervals)
- `point` - Point identifier
- `sea_depth` - Sea depth
- `height_above_bottom` - Height above bottom
- `modification_symbol` - Modification symbol
- `lat` - Latitude
- `lon` - Longitude
- `comment` - Comment

## Files Modified

1. **[`src/meta_finder/io_info_files.py`](src/meta_finder/io_info_files.py:267-345)**
   - Modified `_ungroup_devices_with_underscore_suffixes` to flatten multiple intervals
   - Added tuple/list to dict conversion for compatibility
   - Separated time and non-time fields in intervals structure

2. **[`src/meta_finder/collect.py`](src/meta_finder/collect.py:820-865)**
   - Modified `update_device_metadata_with_time_info` to handle intervals structure
   - Added conditional logic for single vs multi-interval devices

3. **[`src/meta_finder/file_writer.py`](src/meta_finder/file_writer.py:92-243)**
   - Modified `write_metadata_table` to create one row per interval
   - Added `_process_device_entry` helper function
   - Fixed ordering bug (populate devices before processing data paths)

## Backward Compatibility

The implementation maintains full backward compatibility:
- Single-interval devices continue to work as before
- YAML files with underscore suffixes are automatically converted to intervals structure
- Existing TSV output format is preserved for single-interval devices
- No changes required to existing YAML files

## Testing

Created comprehensive test suite in [`tests/test_multiple_intervals_handling.py`](tests/test_multiple_intervals_handling.py):
- Test ungrouping creates proper intervals structure
- Test each interval has correct time values
- Test single interval devices remain unchanged
- Test TSV output creates multiple rows for devices with intervals
- Test non-time fields are preserved across interval rows
- Test interval_index field is present for multi-interval devices

## Bug Fixes

1. **Ordering Bug in [`file_writer.py`](src/meta_finder/file_writer.py:131-221)**
   - Fixed: Data paths processing loop was running before devices were populated
   - Solution: Moved data paths processing after device population

2. **Tuple Handling Bug in [`io_info_files.py`](src/meta_finder/io_info_files.py:267-345)**
   - Fixed: `base_metadata` could be a tuple/list, causing AttributeError
   - Solution: Added tuple/list to dict conversion before processing

## Usage Examples

### YAML Input (existing format)
```yaml
i10:
  0: ["", 85, 0, "⭡", 55.8940, 19.0899, "2018-10-17 16:30:00", "2018-10-18 07:15:00"]
  1: ["", 85, 0, "⭡", 55.8940, 19.0899, "2018-10-22T12:03:00", "2018-10-27T06:47:28"]
```

### Internal Representation (after ungrouping)
```python
{
    "i10": {
        "sea_depth": 85,
        "lat": 55.8940,
        "lon": 19.0899,
        "intervals": [
            {"time_st": "2018-10-17 16:30:00", "time_en": "2018-10-18 07:15:00"},
            {"time_st": "2018-10-22 12:03:00", "time_en": "2018-10-27 06:47:28"}
        ]
    }
}
```

### TSV Output
```
ABP44	i10		85	0	55.894	19.0899	2018-10-17 16:30:00	2018-10-18 07:15:00	...
ABP44	i10		85	0	55.894	19.0899	2018-10-22 12:03:00	2018-10-27 06:47:28	...
```

## Notes

- The implementation follows DRY (Don't Repeat Yourself) principle
- Code is documented following PEP standards with clear intent descriptions
- Defensive error handling is used throughout
- Long expressions are split to fit line-length wrapping
- Imports are organized at the top of files
- Constants are defined and reused instead of hardcoding
