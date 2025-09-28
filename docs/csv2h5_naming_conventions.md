# HDF5_PANDAS.CSV2H5 Naming Conventions and HDF5 Structure

## Overview

This document describes the naming conventions, expected inputs, and resulting HDF5 structure for the `csv2h5.py` module.
which allows efficient storage and retrieval of time-series scientific data while maintaining metadata about the source files and processing history.

### Special Features
- Automatic time zone conversion to UTC
- Data filtering based on min/max values
- Duplicate detection and removal
- Incremental updates for processing new files
- Separator rows to distinguish between different source files


## Naming Conventions

### Input File Naming
- Pattern: `YYMMDD_HHMMSS_*.TOB` (based on the glob pattern in the example)
- Files are typically organized in directories by device type
- Raw data files are often in `_raw` subdirectories

### Configuration File Naming

- Located in `scripts/cfg/` or `shared/hdf5_pandas/cfg/csv2h5_ini/`
- Follows pattern: `csv_DEVICE_TYPE.ini`
- Examples: `csv_CTD_SST.ini`, `csv_CTD_Schuka.ini`


### Table Naming Conventions
- Device-specific names: `CTD_SST_CTD90`, `CTD_SST_90M`
- Log tables follow pattern: `{device_name}/logRuns` or `{device_name}/logFiles`

## HDF5 Structure

### Overall Organization
The resulting HDF5 file follows this structure:
```
filename.h5
├── device_name/
│   ├── table_name/
│   │   ├── data (main data table)
│   │   └── logRuns/ or logFiles/ (metadata table)
│   └── other_device_tables/
└── navigation/
    └── sectionsCTD_routes/ (if applicable)
```

### Main Data Table Structure
- Indexed by datetime (Time column)
- Columns for each measurement parameter
- Data types preserved from input (float, text, etc.)
- Optional separator rows with NaN values

### Log Table Structure
The log table contains metadata about each processed file:
- `fileName`: Relative path and name of the source file
- `fileChangeTime`: Modification time of the source file
- `Date0`: Start time of data in the file
- `DateEnd`: End time of data in the file
- `rows`: Number of rows processed
- `DateProc`: Processing timestamp

### Example Structure from Step 10
```
221116_AMK91.h5
└── CTD_SST_CTD90/
    ├── table (main data)
    │   ├── Time (index)
    │   ├── Pres
    │   ├── Temp
    │   ├── Sal
    │   ├── Turb
    │   ├── Trans
    │   ├── Cond
    │   └── other columns...
    └── logRuns/
        ├── Time (index)
        ├── fileName
        ├── fileChangeTime
        ├── Date0
        ├── DateEnd
        ├── rows
        └── DateProc
```

### Data Types
- Time columns: `datetime64[ns, UTC]`
- Measurement columns: `float64` (typically)
- Text columns: Fixed-width strings
- Index: Always datetime-based for time series data

## Expected Inputs

### Data Format Expectations
- Text files with delimited columns
- Header row defining column names and types
- Time data that can be parsed or converted to datetime
- Numeric data for scientific measurements

### Command Line Arguments
The module accepts arguments in the form of:
1. Configuration file path (e.g., `cfg/csv_CTD_SST.ini`)
2. Key-value pairs as command line arguments (e.g., `--path`, `--header`, etc.)

### Configuration File Structure
Configuration files use INI format with sections:
- `[in]`: Input parameters
- `[filter]`: Filtering parameters
- `[out]`: Output parameters
- `[program]`: Program behavior parameters

### Input Fields
Based on the example configuration and code analysis:

#### [in] Section
- `path`: File path pattern to match input files
- `header`: Column names with optional type specifiers (text), (float), (time)
- `delimiter_chars`: Character(s) separating columns
- `skiprows_integer`: Number of header rows to skip
- `dt_from_utc_hours`: Time zone adjustment

#### [out] Section
- `table`: Name of the HDF5 table to create
- `base`: Base name for output files
- `b_insert_separator`: Whether to insert separator rows between files
