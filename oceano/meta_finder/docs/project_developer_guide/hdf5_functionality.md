# HDF5 Functionality

Analysis and documentation of HDF5/MAT file processing in meta_finder.

## Overview

The HDF5 functionality extracts time range information from HDF5 files as an alternative data source for metadata processing. MAT files (`.mat`) in `_raw` directories are processed the same as HDF5 files with raw priority.

## Supported file types

| Type | Priority | Description |
|------|----------|-------------|
| `proc_noAvg` | 1 | Processed HDF5 without averaging |
| `proc` | 2 | Processed HDF5 with averaging |
| `proc_Avg` | 2 | Processed HDF5 with groups by device |
| `raw` | 3 | Raw HDF5/MAT files in `_raw/` |

## File structure

### proc_noAvg.h5

```
/
├── i63/  # device_id_proc (normalized device name)
│   ├── table (columns: ['index', 'v', 'u', 'inclination', 'Battery', 'Temp'])
│   └── logFiles/
│       └── table (columns: ['index', 'fileName', 'fileChangeTime', 'DateEnd', 'DateProc'])
```

### raw.h5

```
/
├── incl63/  # device_id_raw (not normalized device name)
│   ├── table (columns: ['index', 'Ax', 'Ay', 'Az', 'Mx', 'My', 'Mz', 'Battery', 'Temp'])
│   ├── coef/
│   └── logFiles/
│       └── table (columns: ['index', 'fileName', 'fileChangeTime', 'DateEnd', 'DateProc'])
```

### proc.h5

```
/
├── bin600s/  # averaging bin
│   └── table (columns: ['index', 'Vabs_i03', 'v_i04', ...])
```

### proc_Avg.h5

```
/
├── i04bin2s/  # device_id with averaging bin
│   └── table (columns: ['index', 'Vabs', 'v', 'u', 'Inclination', 'Temp'])
├── i05bin2s/
│   └── table (columns: ['index', 'Vabs', 'v', 'u', 'Inclination', 'Temp'])
└── i03bin600s/
    └── table (columns: ['index', 'Vabs', 'v', 'u', 'Inclination', 'Temp'])
```

## Device extraction

Device IDs are extracted from HDF5 group names using the pattern:

```
\d*[@#_-]?(?P<type>[iwp])(?:ncl|nkl)?_?(?P<model>[bp]?)0*(?P<number>\d+)
```

Both normalized (`i63`) and non-normalized (`incl63`) group names are supported.

## Time range extraction

Time ranges are extracted from the `index` column of HDF5 tables. The extraction logic:

1. Opens the HDF5 file and locates the device group
2. Reads the `table` dataset
3. Identifies the time column (typically `index`)
4. Extracts first and last valid timestamps

## Coefficient extraction

Coefficient dates are extracted from the `coef` group when `extract_hdf5_coef_dates=True` or `coef_date` is in `raw_hdf5_cols`.

## Skip conditions

HDF5 extraction is skipped when time metadata is already available in info files. If `time_st`, `time_en`, or `coef_date` already contain meaningful values (not placeholders like "?", "-", or empty strings), the existing metadata is preserved.

## Implementation

- `hdf5_processor.py` — HDF5/MAT extraction functions
- `data_processor.py` — `get_h5_type_and_priority()` classifies files
- `collect.py` — orchestrates extraction via `extract_time_metadata_from_prioritized_sources()`

## See also

- [Input / Output Format Specification](../reference/io_formats.md)
- [Processing Guide](../user_guide/processing.md)
- [Codebase Analysis](codebase_analysis.md)
