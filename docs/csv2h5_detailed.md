# CSV to HDF5 Conversion Process

## Overview

The `csv2h5.py` module is the core component responsible for converting various text-based data formats into HDF5 format for efficient storage and analysis. This document details the conversion process, configuration options, and usage patterns.

## Core Functionality

### Main Conversion Pipeline

1. **File Discovery**: Locate and validate input files based on path patterns
2. **Format Detection**: Determine CSV dialect and structure
3. **Data Loading**: Read data using pandas/dask with appropriate parsing options
4. **Preprocessing**: Apply format-specific transformations
5. **Filtering**: Apply data quality filters
6. **Storage**: Write data to HDF5 with metadata

### Key Components

#### Configuration Parser
The system uses a sophisticated configuration system that:
- Supports INI-style configuration files
- Allows command-line parameter overrides
- Performs automatic type conversion based on parameter names
- Supports complex data structures (lists, dictionaries)

#### Data Loading Engine
- Uses pandas for smaller datasets
- Uses dask for larger-than-memory datasets
- Supports various CSV dialects and text formats
- Handles fixed-width and delimited formats

#### Processing Pipeline
- Applies instrument-specific processing functions
- Performs time corrections and validations
- Calculates derived parameters
- Filters data based on quality criteria

## Configuration System

### Parameter Naming Conventions

The system uses naming conventions to automatically determine data types:

| Suffix/Prefix | Data Type | Example |
|---------------|-----------|---------|
| `_list` | List | `cols_not_save_list` |
| `_int`, `_integer` | Integer | `skiprows_integer` |
| `_float` | Float | `min_Pres_float` |
| `_bool`, `_b` | Boolean | `b_incremental_update_bool` |
| `_date`, `_time` | DateTime | `min_date` |
| `_dict` | Dictionary | `filter_dict` |
| `dt_` | TimeDelta | `dt_from_utc_hours` |

### Configuration Sections

#### [in] - Input Configuration
Controls data input and parsing:
- `path`: File path pattern for input files
- `header`: Column names and types
- `delimiter_chars`: Field separator characters
- `skiprows_integer`: Number of header rows to skip
- `max_text_width`: Maximum width for text fields

#### [filter] - Data Filtering
Defines quality control filters:
- `min_{param}`: Minimum acceptable value
- `max_{param}`: Maximum acceptable value
- `fun_{param}`: Custom filter function

#### [out] - Output Configuration
Controls HDF5 output structure:
- `table`: Table name in HDF5 file
- `b_insert_separator`: Add separator rows between files
- `b_reuse_temporary_tables`: Reuse existing temporary data

#### [program] - Program Behavior
Controls execution behavior:
- `log`: Log file path
- `verbose`: Logging verbosity level
- `b_interact`: Interactive mode

## Data Processing Pipeline

### 1. File Input and Validation

The system processes files in batches, with support for:
- Wildcard patterns for file selection
- Recursive directory searching
- File existence and readability checks
- Interactive confirmation for large file sets

### 2. Format Detection and Parsing

Based on configuration parameters:
- `delimiter_chars`: Determines field separator
- `header`: Defines column names and types
- `skiprows_integer`: Specifies header rows to skip
- Custom parsing functions for complex formats

### 3. Instrument-Specific Processing

The system supports specialized processing for different instrument types through the `fun_proc_loaded` parameter:

```python
# Example for SST CTD data
fun_proc_loaded = loaded_sst
```

Processing functions are defined in `csv_specific_proc.py` and handle:
- Time column combination and formatting
- Unit conversions
- Data validation
- Format-specific corrections

### 4. Data Filtering

Multiple filtering mechanisms:
- Range filtering (`min_`, `max_` parameters)
- Custom function filtering (`fun_` parameters)
- Global filtering for entire datasets
- Local filtering that preserves data structure

### 5. HDF5 Storage

Data is stored in a structured format:
- Main data table with time index
- Metadata table with processing information
- Optional log tables for tracking file processing

## Usage Patterns

### Basic Usage

```bash
python csv2h5.py config.ini
```

### Command Line Overrides

```bash
python csv2h5.py config.ini --path "data/*.csv" --table "CTD_data"
```

### Python API Usage

```python
from hdf5_pandas.csv2h5 import main as csv2h5

csv2h5([
    'config.ini',
    '--path', 'data/*.csv',
    '--table', 'CTD_SST',
    '--b_interact', '0'
])
```

### Advanced Configuration

```python
csv2h5([
    'config.ini',
    '--path', 'data/*.csv',
    '--table', 'CTD_SST',
    '--b_interact', '0'
], **{
    'in': {
        'fun_proc_loaded': loaded_sst,
        'csv_specific_param': {
            'Temp_fun': lambda x: np.polyval([-1.102460295e-05, 1.00018, 0.037725], x),
            'Cond_fun': lambda x: np.polyval([-0.000666294, 1.0279, -0.140743], x)
        }
    }
})
```

## Instrument-Specific Processing

### CTD Data Processing

#### SST CTD
- Combines separate Date and Time columns
- Applies calibration coefficients
- Calculates derived parameters (Salinity using GSW)

#### Idronaut CTD
- Handles specific date/time format
- Processes multi-column time data
- Applies instrument-specific corrections

#### Schuka CTD
- Custom date parsing
- Special handling for missing data
- Format-specific quality checks

### Navigation Data

#### GPX Data
- Converts waypoints, tracks, and routes
- Handles coordinate transformations
- Preserves metadata and descriptions

#### Supervisor Format
- Custom navigation data processing
- Time synchronization with other sensors
- Coordinate system handling

## Performance Considerations

### Memory Management
- Uses dask for out-of-core processing
- Configurable chunk sizes
- Temporary file management
- Efficient data types

### Parallel Processing
- Dask-based parallelization
- Chunked data processing
- Memory-efficient operations
- Progress tracking

### Optimization Strategies
- Pre-sorting for faster indexing
- Efficient data type selection
- Compression options
- Index creation strategies

## Error Handling

### Data Validation
- Type checking and conversion
- Range validation
- Time sequence validation
- Missing data handling

### Recovery Mechanisms
- Temporary file management
- Incremental updates
- Duplicate detection
- Error logging and reporting

### Common Issues and Solutions

#### Time Zone Problems
- Use `dt_from_utc_hours` parameter
- Apply time corrections in processing functions
- Validate time sequences

#### Data Type Mismatches
- Use type suffixes in configuration
- Apply custom parsing functions
- Validate data during loading

#### Memory Issues
- Use dask for large files
- Adjust chunk sizes
- Process files in smaller batches

## Extending the System

### Adding New Instrument Support

1. Create processing function in `csv_specific_proc.py`
2. Add configuration template in `cfg/csv2h5_ini/`
3. Register function in processing pipeline
4. Test with sample data

### Custom Processing Functions

```python
@meta_out_fields(keys_del={'Date', 'Time'}, add_before={'Time': 'M8[ns]'})
def loaded_custom_instrument(a: Union[pd.DataFrame, np.ndarray],
                           cfg_in: Mapping[str, Any],
                           csv_specific_param: Optional[Mapping[str, Any]] = None) -> pd.DataFrame:
    """
    Custom processing for new instrument type
    """
    # Combine date and time columns
    date = pd.to_datetime(a['Date'], format='%d.%m.%Y') + \
           pd.to_timedelta(a['Time'])

    # Apply custom transformations
    # ...

    return a.assign(Time=date)
```

### Configuration Templates

Create new INI files in `cfg/csv2h5_ini/` following the established patterns:

```ini
[in]
path = data/custom_instrument/*.dat
header = Date(text),Time(text),Param1(float),Param2(float)
delimiter_chars = \t

[filter]
min_Param1 = 0
max_Param1 = 100

[out]
table = Custom_Instrument_Data

[program]
log = logs/custom_processing.log
```

## Best Practices

### Configuration Management
- Use descriptive configuration file names
- Document parameter purposes
- Version control configuration files
- Test configurations with sample data

### Data Quality
- Validate input data formats
- Implement appropriate filters
- Log data quality metrics
- Handle edge cases explicitly

### Performance Optimization
- Choose appropriate chunk sizes
- Use efficient data types
- Minimize temporary file usage
- Monitor memory consumption

### Maintenance
- Keep documentation updated
- Test with various data formats
- Monitor processing logs
- Profile performance regularly