# Comparison of filter_local_with_file_name_settings Functions

## Overview

This document explains the differences between two functions in the csv2h5 module:
1. `filter_local_with_file_name_settings_old` - The original implementation
2. `filter_local_with_file_name_settings` - The enhanced implementation with directional filtering

## Function Signatures

Both functions have the same signature:
```python
def filter_local_with_file_name_settings(d: Union[pd.DataFrame, dd.DataFrame],
                                        cfg: Mapping[str, Any],
                                        path_csv: PurePath) -> Union[pd.DataFrame, dd.DataFrame]
```

## Key Differences

### 1. Directional Filtering Support

**Old Function (`filter_local_with_file_name_settings_old`)**:
- Does not support directional filtering
- Any file name containing `up_` or `down_` prefixes will be treated as regular column names
- Filters the entire column regardless of directional prefixes

**New Function (`filter_local_with_file_name_settings`)**:
- Supports directional filtering with `up_` and `down_` prefixes
- `up_` - Filters from the pressure maximum to the end of the data
- `down_` - Filters from the start of the data to the pressure maximum

### 2. Implementation Approach

**Old Function**:
- Directly parses the file name using regex `[-_,;([]no_`
- Manually handles the filtering logic

**New Function**:
- Uses helper functions for better separation of concerns:
  - `_parse_bad_columns_and_direction_from_filename()` - Parses file name and extracts direction
  - `_apply_column_filter()` - Applies the filtering logic
  - `_apply_column_filter_to_dataframe()` - Handles filtering for individual DataFrames

## File Name Parsing Rules

Both functions parse file names looking for patterns like:
```
{separator}no_[up_|down_]{Name1[, Name2...]}
```

Where:
- **Separators**: `;`, `(`, `)`, `[`, `]` (can be around the expression)
- **Column separators**: `-`, `_`, `,`, or space
- **Special case**: 'Ox' is treated as 'O2' and 'O2ppm'
- **Directional prefixes**: 'up_' or 'down_' to filter only ascending/descending pressure rows

## Directional Filtering Logic

The new function implements directional filtering based on pressure data:

1. **Find pressure maximum**: Locate the time index of maximum pressure value
2. **Apply directional filter**:
   - `up_` direction: Set specified columns to NaN from pressure maximum time to the end
   - `down_` direction: Set specified columns to NaN from start to pressure maximum time

### Example

Given pressure data: `[1, 2, 3, 4, 5, 4, 3, 2, 1]` (maximum at index 4)

- `up_Temp` would set Temp values at indices 4-8 to NaN
- `down_Temp` would set Temp values at indices 0-4 to NaN

## Test Cases

The test suite includes cases for:

1. **Non-directional filtering**: Both functions should produce identical results
2. **Directional filtering**: New function should implement directional logic, old function should filter entire columns
3. **Edge cases**: Empty DataFrames, missing pressure columns, single values, etc.

## When to Use Each Function

- **Use the new function** (`filter_local_with_file_name_settings`) when you need directional filtering capability
- **The old function** (`filter_local_with_file_name_settings_old`) can be used for backward compatibility, but lacks directional filtering

## Migration Path

When replacing the old function with the new one:
1. All existing non-directional filtering will work identically
2. Files with directional prefixes will now have proper directional filtering applied
3. No changes needed for existing code that doesn't use directional filtering