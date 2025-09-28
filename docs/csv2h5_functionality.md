# HDF5_PANDAS.CSV2H5 Functionality

## Overview

The `csv2h5.py` module is part of the `hdf5_pandas` package and is designed to convert CSV and similar text files into pandas HDF5 stores with the addition of log tables. It's a powerful tool for processing scientific data files and storing them in an efficient, queryable format.

## Main Function Workflow

The main function in `csv2h5.py` follows this sequence when processing text data and saving it to HDF5 store:

### 1. Configuration and Initialization
- Parses command-line arguments and configuration files
- Initializes input column definitions using `init_input_cols()`
- Sets up output database paths and table names using `h5.out_init()`

### 2. File Processing Loop
For each input file:
1. **Data Loading**: Uses `read_csv()` function which:
   - Loads CSV data using dask.dataframe for efficient handling of large files
   - Applies custom processing functions if specified
   - Performs time correction and validation

2. **Data Filtering**: Applies global and local filters using:
   - `set_filterGlobal_minmax()` for range-based filtering
   - `filter_local_with_file_name_settings()` for file name-based filtering

3. **Data Storage**: Appends processed data to HDF5 store using:
   - `h5.append()` function which saves both data and metadata

### 3. Post-Processing
- Sorts and packs data tables using `h5.move_tables()`
- Creates indexes for efficient querying

## Key Components

### Data Loading (`read_csv`)
The `read_csv()` function is responsible for:
- Reading CSV files using dask for memory-efficient processing
- Handling different data types (float, text, time)
- Applying time corrections and validations
- Setting the time column as the index

### Data Storage (`h5.append`)
The `h5.append()` function:
- Stores the processed dataframe in the specified HDF5 table
- Creates a corresponding log table with metadata
- Handles chunked storage for large datasets

### Configuration
The module supports extensive configuration through:
- Command-line arguments
- Configuration files (INI format)
- Direct parameter passing

## Data Flow for Text to HDF5 Storage

1. **Input**: Text files (CSV or similar format)
2. **Processing**:
   - Parse according to header/column specifications
   - Apply type conversions and time corrections
   - Filter data based on configured criteria
3. **Storage**:
   - Save data in HDF5 table with time-based indexing
   - Create metadata log table with file information
   - Maintain data integrity through sorting and indexing

This approach allows for efficient storage and retrieval of time-series data while maintaining metadata about the source files.