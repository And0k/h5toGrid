# Input / Output Format Specification

Authoritative contracts for data formats accepted and produced by meta_finder.
Implementation: [`file_finder.py`](../../src/meta_finder/file_finder.py) (discovery),
[`parse_data_file_name.py`](../../src/meta_finder/parse_data_file_name.py) (identity),
[`data_proc_funcs.py`](../../src/meta_finder/data_proc_funcs.py) (time extraction),
[`file_writer.py`](../../src/meta_finder/file_writer.py) (output).

## Input formats

### Text data files

Tab-separated text files in `text_output/` directories or archives (`.zip`, `.7z`).

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

MAT files (`.mat`) in `_raw` directories are processed as the same as HDF5 files with raw priority.

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
- `table` (data table) with parameter name columns containing device_id suffixes, where device IDs are embedded in column names like "Vabs_i03", "v_i04", etc.

**.proc_Avg.h5 files contain groups:**
- device_id with averaging bin (e.g., `i04bin2s`, `i03bin600s`) - device IDs are in group names, not in column names
- `table` (data table) with parameter name columns WITHOUT device_id suffixes (columns like "Vabs", "v", "u", etc.)

Structure example:
```
/
├── i04bin2s/  # device_id with averaging bin
│   └── table (columns: ['index', 'Vabs', 'Vdir', 'v', 'u', 'Inclination', 'Temp'])
├── i05bin2s/  # another device with same averaging
│   └── table (columns: ['index', 'Vabs', 'Vdir', 'v', 'u', 'Inclination', 'Temp'])
└── i03bin600s/  # device with different averaging
    └── table (columns: ['index', 'Vabs', 'Vdir', 'v', 'u', 'Inclination', 'Temp'])
```

## File name parsing

`parse_filename_for_metadata()` in [`parse_data_file_name.py`](../../src/meta_finder/parse_data_file_name.py) extracts device identity from filenames.

### Supported filename patterns

| Pattern | Example | Extracted data |
|---------|---------|----------------|
| `{datetime}bin{interval}s[@_#]{devices}.tsv` | `191108_1200bin600s@i03.tsv` | devices: i03, bin: 600s |
| `{date}#{devices}-bin{interval}s.{ext}` | `191210#07,23,30,32-bin300s.zip` | devices: i07,i23,i30,i32, bin: 300s |
| `{date}#{devices}.{ext}` | `191210#i03.tsv` | devices: i03 |
| `{date}_{time}_{device}.{ext}` | `200113_0000_i13.csv` | device: i13 |
| `{date}_{time}@{device}.{ext}` | `200113_00@i13.csv` | device: i13 |

Any part except Date (which can be in `yymmdd_HHMM`, `yymmdd_HH` or `yymmdd` format) may be omitted. Separator: "@", "_" or "#". Seconds: binning interval (float).

### Complex device name patterns

- **Semicolon-separated groups** separate different device groups: `i3,4,15,19,37,38;ib27-30,ip6.tsv`
- **Range expansion**: `27-30` → `["27", "28", "29", "30"]`
- **Parentheses**: `i(38,37,59,60,58).tsv` → `i38,i37,i59,i60,i58`
- **Mixed prefixes**: `i_b(27,28,29,30).tsv` → devices: `i_b27, i_b28, i_b29, i_b30`

### Device ID normalization

Data files usually include device names in file name which is an abbreviation of device type, model and number. Underscores after type ("i" for inclinometer) or zeros before the first other number are not considered significant:

| Original | Normalized | Note |
|----------|-----------|------|
| `i_b27` | `ib27` | Underscore after type removed |
| `i_p06` | `ip06` | Underscore between type and model removed |
| `i_03` | `i03` | Leading zero removed |
| `W01` | `w1` | Lowercased |
| `INCL01` | `i01` | Expanded type shortened |
| `i03_` | `i03_` | Trailing underscore NOT removed (additional entry) |

### Default device ID patterns

Defined in [`config.py`](../../src/meta_finder/config.py):

- **device_type_pattern**: `r"incl|wg|[iw]"` - matches device type without number
- **device_model_pattern**: `r"[bp]?"` - optional model suffix
- **device_number_pattern**: `r"\d+"` - device number
- **device_id_pattern**: `fr"(?:{device_type_pattern})(?:{device_model_pattern})?{device_number_pattern}"` - matches complete device IDs
- **device_id_complex_pattern**: `r"\d*[@#_-]?(?P<type>{device_type_pattern})(?:ncl|nkl)?_?(?P<model>{device_model_pattern})0*(?P<number>{device_number_pattern})"` - for extracting device IDs with optional prefixes and separators
- **devices_in_text_output_files_pattern**: Flexible pattern for text output filenames

## Device discovery priority

### Data file name parsing priority

1. **Averaging interval priority**: Files with lower averaging intervals have higher priority (2s > 600s > 7200s)
2. **Files without averaging information**: Treated as having the configured default averaging value (2.0001s)
3. **Specificity**: Dedicated files (for specific devices) > combined files
4. **Number of devices**: Files with fewer devices > files with more devices
5. **Number of unmatched devices**: Files with fewer unmatched devices > files with more unmatched devices

### Source priority order

1. **Data File Name Parsing** (text_output directory or archives)
2. **Raw Directory Search** (`_raw/` directory)
3. **HDF5/MAT Fallback** (when text_output and raw files are not available)

For HDF5 files:
- `*.proc_noAvg.h5` — priority 2 (highest)
- `*.proc_Avg.h5` — priority 3
- `*.proc.h5` — priority 4
- `_raw/*.h5` — priority 5
- `_raw/*.mat` — priority 6 (lowest)

## Output formats

### `meta_TCM.tsv`

Tab-separated table with consolidated metadata. See [Input / Output Guide](../user_guide/input_output.md) for column descriptions.

### `files_TCM.tsv`

Text list of paths grouped by cruise:

```
path/to/device/dir/cruise1
path/to/data/file1.tsv
path/to/data/file2.tsv

path/to/device/dir/cruise2
path/to/data/file3.tsv
```

## Metadata file format

The system reads metadata files in priority order:
1. `info_devices@meta_finder.yaml`
2. `info_devices.yaml`
3. `info_devices.json` (deprecated)

### YAML format

```yaml
# Instrument_ID: [Point, Sea_depth, H_above_bot, Symbol, Lat, Lon, Time_st, Time_en, Burst_dt, Bursts_t, Comment]
i03: ["P1", 15, 0.5, "⯯", 54.7123, 19.8456, "2023-05-08 12:00:00", "2023-06-15 08:30:00", -, -, ""]
w01: ["P1", 15, 5, "", 54.7123, 19.8456, "2023-05-08 12:00:00", "2023-06-15 08:30:00", -, -, ""]
```

### Array element order

| Index | Field | Description | Required |
|-------|-------|-------------|----------|
| 0 | `point` | Station name/number | Yes |
| 1 | `sea_depth` | Sea depth (m) | Yes |
| 2 | `height_above_bottom` | Height above bottom (m) | Yes |
| 3 | `modification_symbol` | Modification symbol | Yes |
| 4 | `lat` | Latitude (degrees) | No |
| 5 | `lon` | Longitude (degrees) | No |
| 6 | `time_st` | Start time | No |
| 7 | `time_en` | End time | No |
| 8 | `burst_dt` | Active interval (s) | No |
| 9 | `bursts_t` | Operation period (s) | No |
| 10 | `comment` | Comment | No |
| 11 | `coef_date` | Calibration date (from HDF5) | No |
| 12 | `time_raw_st` | Raw data start | No |
| 13 | `time_raw_en` | Raw data end | No |

- **Minimum array length:** 4 elements (first 4 required fields).
- **Missing elements** filled with defaults: `?` for most fields, `""` for `burst_dt`, `bursts_t`, `coef_date`, `time_raw_st`, `time_raw_en`.
- **`null`/`~` values** in YAML treated as missing.
- **Leading space in device IDs** (e.g., `" i10"`) ignored — convention for devices not yet collected.

### Multiple intervals

For devices with multiple deployment intervals, use nested structure:

```yaml
i03:
  0: ["P1", 15, 0.5, ?, 54.7, 19.8, "2023-05-08 12:00:00", "2023-05-15 08:00:00", -, -, ""]
  1: ["P2", 20, 1.0, ?, 54.9, 20.1, "2023-06-01 10:00:00", "2023-06-10 14:00:00", 120, 600, ""]
```

## Time formats

Time fields accept various ISO formats:

| Format | Example |
|--------|---------|
| `YYYY-MM-DD HH:MM:SS` | `2019-12-10 14:22:00` |
| `YYYY-MM-DDTHH:MM:SS` | `2019-12-10T14:22:00` |
| `YYYY-MM-DDTHH:MM` | `2019-12-10T14:22` |
| `YYYY-MM-DD HH:MM` | `2019-12-26 12:40` |
| `YYYY-MM-DDTHH:MM:SS.ffffff` | `2019-12-10T14:22:00.123456` |

All formats are normalized to `YYYY-MM-DD HH:MM:SS` on read.

## Modification symbols

| Symbol | Meaning |
|--------|---------|
| `⯯` | Negative buoyancy (widest range) |
| `⯭` | Positive buoyancy, wide range (large float) |
| `⭡` | Positive buoyancy, narrow range of measured current velocity |
| `⤉` | Positive buoyancy, with rod |
| `↑` | Positive buoyancy, symbol not used |

## See also

- [Input / Output Guide](../user_guide/input_output.md)
- [Configuration Reference](config_reference.md)
- [Processing Guide](../user_guide/processing.md)
