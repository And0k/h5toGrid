# Input / Output Guide

This guide describes the data formats accepted and produced by meta_finder.
For the authoritative format contracts, see [Input / Output Format Specification](../reference/io_formats.md).

## Input data

The program discovers and processes data from three main sources:

### Text data files (text_output)

Tab-separated text files containing processed instrument data:

**Tilt current meter (inclinometer) data files contain the following titled data columns (with tab separator):**
- Time – Kaliningrad time, `yyyy-mm-dd HH:MM:SS.ffffff`, (in older versions the column might also be named `index`)
- Vabs – absolute value of current velocity, m/s
- Vdir – direction of current velocity, ° in geographic coordinate system
- v – north component of current velocity, m/s
- u – east component of current velocity, m/s
- Inclination – inclinometer angle, °
- Temp – temperature of inclinometer processor, °C

**Wave gauge data files contain the following titled data columns (with tab separator):**
- Time – Kaliningrad time, `yyyy-mm-dd HH:MM:SS.ffffff`
- Pressure – pressure, dBar

**Data files combining data from multiple devices** usually contain a smaller set of columns for each device; data columns have device name suffix. Example:

```
index  v_i03  u_i03  Temp_i03  v_i04  u_i04  Temp_i04
```

### HDF5 and MAT files

MAT files (.mat) in `_raw` directories are processed as the same as HDF5 files with raw priority.

**Non .proc.h5 files contain groups:**
- `device_id` (can be normalized or not normalized, e.g., `i07`, `w01`, `inclinometer07`, `@i07`)
- `table` (data table) with columns by parameter names
- `coef` (only in raw directory files or with `.raw.h5` suffix) - group with coefficients and their metadata

**.proc_noAvg.h5:**
```
/
├── i63/  # device_id_proc (normalized device name)
│   ├── table (columns: ['index', 'v', 'u', 'inclination', 'Battery', 'Temp'])
│   └── logFiles/
│       └── table (columns: ['index', 'fileName', 'fileChangeTime', 'DateEnd', 'DateProc'])
```

**.raw.h5:**
```
/
├── incl63/  # device_id_raw (not normalized device name)
│   ├── table (columns: ['index', 'Ax', 'Ay', 'Az', 'Mx', 'My', 'Mz', 'Battery', 'Temp'])
│   ├── coef/
│   └── logFiles/
│       └── table (columns: ['index', 'fileName', 'fileChangeTime', 'DateEnd', 'DateProc'])
```

**.proc.h5 files contain groups:**
- averaging_bin (contains averaging information in name, e.g., `bin600s`)
- `table` (data table) with parameter name columns containing device_id suffixes

## Cruise directory structure

The program expects the following standard directory structure:

```
B:/WorkData/BalticSea/
└── YYMMDD_{cruise_name}/
    ├── *inclinometer* or *wavegauge*/
    │   ├── info_devices@meta_finder.yaml  ← priority 1 (auto-generated)
    │   ├── info_devices.yaml              ← priority 2
    │   ├── info_devices.json              ← priority 3 (deprecated)
    │   ├── text_output/
    │   │   └── *.tsv, *.txt, *.csv
    │   ├── text_output.zip
    │   ├── text_output.7z
    │   ├── *.proc_noAvg.h5
    │   ├── *.proc.h5
    │   ├── *.proc_Avg.h5
    │   └── _raw/
    │       ├── *.h5, *.mat
    │       └── *.txt, *.tsv, *.csv
    └── *navigation* or *map*/
        └── *.gpx
```

Combined structures are also supported:
- `YYMMDD{cruise_name}/{device_type}/info_devices.yaml`
- `YYMMDD{cruise_name}/info_devices.yaml` (metadata in root)
- `YYMMDD{cruise_name}/inclinometers/{YYMMDD}*/info_devices.yaml` (nested dated subdirectories)

## Output files

The program creates two output files in the `meta` directory:

| File | Description |
|------|-------------|
| `{yymmdd_HHMM}_files_TCM.tsv` | List of all processed files organized by cruise |
| `{yymmdd_HHMM}_meta_TCM.tsv` | Tab-separated table with consolidated metadata for all devices |

### Metadata table columns

| Column | Description |
|--------|-------------|
| `setup_name` | Dataset name — unique identifier combining cruise name with optional date prefix and device suffix |
| `device_id` | Normalized device identifier (e.g., i7, w1) |
| `point` | Station name or point number |
| `sea_depth` | Sea depth at deployment point (m) |
| `height_above_bottom` | Device height relative to seabed (m) |
| `lat` | Latitude of deployment point or ? |
| `lon` | Longitude of deployment point or ? |
| `time_st` | Start date of processed data (seconds precision) |
| `time_en` | End date of processed data (seconds precision) |
| `burst_dt` | Device active interval in seconds (periodic mode) |
| `bursts_t` | Device operation period in seconds (periodic mode) |
| `data_file_path` | Path to the one processed text file on disk |
| `quality` | Quality indicator: `+-` for amplitude+direction, `+` for processed, `-` for missing, `?` for raw |
| `comment` | Comments including combined device info and GPX paths |
| `modification_symbol` | Device modification/construction symbol |
| `coef_date` | Calibration coefficient date (from HDF5) |
| `time_raw_st` | Start of raw data (from HDF5/MAT in `_raw/`) |
| `time_raw_en` | End of raw data (from HDF5/MAT in `_raw/`) |

### Output format rules

- Fields not found by the program are filled with "?" sign
- Values of `burst_dt` and `bursts_t` equal to "-" mean the device operated normally without periodic shutdown
- Coordinates are recorded in degrees; if not found, "?" is written for `lat` and `lon`, and paths to all found GPX files are written in the `comment` field with prefix "GPX:"
- For combined data columns (e.g., `Vabs_i05_14`), the comment field contains "{device1}+{device2} output"

## See also

- [Getting Started](getting_started.md)
- [Processing Guide](processing.md)
- [Input / Output Format Specification](../reference/io_formats.md) — authoritative format contracts
- [Configuration Reference](../reference/config_reference.md)
