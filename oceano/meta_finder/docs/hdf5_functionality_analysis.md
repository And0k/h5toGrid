# HDF5 Functionality Analysis

## Overview
The HDF5 functionality is designed to extract time range information from HDF5 files. This serves as an alternative data source for metadata processing in the TCM Metadata Processor.

## Test Results with Real Files

### Directory Structure
Tested with HDF5 files from: `D:/WorkData/BalticSea/250415_ABP60@i,t-chain/inclinometer`

### File Types Found
- **proc_noAvg**: `250415.proc_noAvg.h5` (Processed file without averaging)
- **raw**: `250415.raw.h5` (Raw data file in `_raw` subdirectory)
- **Priority Order**: The system correctly follows the priority order: `proc_noAvg` → `proc` → `raw`

### Device Extraction
- **Successfully extracted**: Device ID `i63` from both files
- **proc_noAvg file**: Device found in group `/i63`
- **raw file**: Device found in group `/incl63` (normalized to `i63`)
- **Device extraction regex**: Working correctly with the pattern `@?(?P<type>[iwp])(?:ncl|nkl)?_?(?P<model>[bp]?)0*(?P<number>\d+)`

### File Structure Analysis
#### proc_noAvg.h5
```
/
├── i63/
│   ├── table (columns: ['index', 'v', 'u', 'inclination', 'Battery', 'Temp'])
│   └── logFiles/
│       └── table (columns: ['index', 'fileName', 'fileChangeTime', 'DateEnd', 'DateProc'])
```

#### raw.h5
```
/
├── incl63/
│   ├── table (columns: ['index', 'Ax', 'Ay', 'Az', 'Mx', 'My', 'Mz', 'Battery', 'Temp'])
│   ├── coef/
│   └── logFiles/
│       └── table (columns: ['index', 'fileName', 'fileChangeTime', 'DateEnd', 'DateProc'])
```

### Time Range Extraction Issue
- **Expected**: Time ranges extracted from the `index` column (which contains datetime data)
- **Actual**: No time ranges extracted (returns empty dictionary)
- **Root cause**: The extraction logic may not be properly accessing the nested table structure

## Functionality Breakdown

### 1. `find_hdf5_files()`
✅ **Working correctly**: Properly identifies files in priority order
- Finds `*.proc_noAvg.h5` files in main directory
- Finds `*.h5` files in `_raw` subdirectory

### 2. `extract_devices_from_hdf5_groups()`
✅ **Working correctly**: Successfully extracts device IDs from group names
- Handles both `i63` and `incl63` patterns
- Applies proper normalization

### 3. `extract_time_range_from_hdf5_table()`
⚠️ **Partially working**: Can identify time columns but may not be extracting data correctly
- Correctly identifies `index` as the time column
- May have issues with data extraction from the specific HDF5 structure

### 4. `extract_time_ranges_from_hdf5_combined()`
⚠️ **Partially working**: Designed for combined tables with multiple device columns
- Not applicable for this dataset structure (devices are in separate groups)

### 5. `extract_metadata_from_hdf5()`
⚠️ **Needs improvement**: Priority-based extraction is working but time range extraction fails
- Follows correct priority order: proc_noAvg → proc → raw
- Opens files correctly but fails to extract time ranges

## Potential Issues & Recommendations

### Issue 1: Nested Group Structure
The current logic may not properly navigate the nested HDF5 structure where:
- Groups contain subgroups (e.g., `/i63/table` instead of just `/table`)
- The extraction logic may need to look deeper in the hierarchy

### Issue 2: Time Column Data Access
The time data exists in the `index` column, but the extraction may fail due to:
- Data type mismatches
- Time format incompatibilities
- PyTables API usage issues

### Issue 3: Table Path Resolution
The system may need to better handle table paths like `/i63/table` instead of just `/table`.

### Issue 4: Timestamp Conversion (Root Cause Identified)
The debug output shows that time range extraction IS working, but the timestamps are being incorrectly converted:
- Raw values: `174566089300000000` (nanosecond timestamps, representing April 2025)
- Converted to: `5317793529-11-05 07:06:40` (incorrect year ~55 million AD)
- Expected: Around April 2025 based on the file naming convention `250415` (April 15, 2025)

The issue is in the timestamp conversion logic in `extract_time_range_from_hdf5_table()` function where the line:
```python
start_time = str(time_values[0].astype('datetime64[s]'))
```
is incorrectly converting nanosecond timestamps as if they were second timestamps.

## Solution
The function needs to detect if the timestamps are in nanoseconds and convert them appropriately by dividing by 1e9 before conversion.

## Fix Implemented
I have implemented a fix that:
1. Added a helper function `_convert_timestamps()` that checks the `index_kind` attribute from the table
2. If `index_kind` contains `datetime64[ns]`, it properly converts nanosecond timestamps by dividing by 1e9
3. Updated both `extract_time_range_from_hdf5_table()` and `extract_time_ranges_from_hdf5_combined()` to use the helper function
4. This follows the DRY principle by centralizing the timestamp conversion logic

## Results After Fix
After implementing the fix, the debug output shows correct timestamps:
- Raw values: `1745660893000000000` nanosecond timestamps
- Converted to: `2025-04-26 11:48:13` (correct date in 2025)
- The function now properly extracts time ranges from HDF5 files

## Performance Notes
- **Speed**: The HDF5 processing is relatively fast
- **Memory usage**: Efficient with file handling using context managers
- **Error handling**: Good logging and exception handling in place

## Conclusion
The HDF5 fallback functionality has a solid foundation with proper file discovery and device extraction, but the time range extraction component needs refinement to properly handle the nested structure of real-world HDF5 files. The system correctly identifies files and devices but fails to extract the actual time ranges from the tables.

## Next Steps
1. Debug the time extraction function to handle nested group structures
2. Add better error logging to identify where exactly the extraction fails
3. Test with additional HDF5 file structures to ensure robustness