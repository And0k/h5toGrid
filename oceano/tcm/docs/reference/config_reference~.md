# Configuration Schema and Device Metadata Reference

## CLI keys outside the typed configuration (not in YAML)

| Field | Purpose |
|-------|---------|
| `path_field` = — | Data/config search path — required first CLI argument (optional for the GUI, where the search field sets it)[↓](#path_field) |
| `+input.min_date` = `None` | Processing interval start (an alternative to the `input.time_ranges` start — merged with it when that is set too). |
| `+input.max_date` = `None` | Processing interval end (an alternative to the `input.time_ranges` end — merged with it when that is set too). |

### `path_field`

Search path for raw data / their processing configs.

#### Detailed
Processing configurations will be created, if they don't already exist, in the `cfg_proc/run/` subdirectory.
Output files will be in the directory above (if the `_raw` subdirectory is in the path, then above it).

##### Important

Enter the absolute path to the:
- **directory** to search for data with the glob `*i*.txt`, or
- raw file(s) via **glob** (`*i*.txt`) / **regex** (`i.*\.txt`) of file-names, or
- config(s) (must end in **`.yaml`**) — load ready configs directly from the `cfg_proc/run/` subfolderEnter the absolute search path - existing configuration/s (if end in **`.yaml`**) from the raw files subfolder `cfg_proc/run/`

Expected [input data layout](io_formats.md#directory-layout):
```text
├── _raw\            ← raw data (`.txt`/`.csv`/`.h5`/`.nc`) — REQUIRED
├── cfg_proc\
│   └── run\         ← per-probe YAML configs (auto-generated on first scan)
```

## YAML file and command-line configuration fields (typed configuration via Hydra/OmegaConf)

YAML configs live at `cfg_proc/run/{yymmdd_hhmm}@pcid[-comment].yaml` inside the raw data directory, where
- `yymmdd_hhmm` — timestamp from `input.time_ranges[0]` or row files 1st row (absent when not found);
- pcid — canonical identificator, e.g. `i3.txt` → `i03`;
- `-comment` — the suffix with its separator "-" (if any) is left unchanged.

All fields are defined in `tcm/schema.py` via the `Config` dataclass and registered groups
(`input`, `out`, `filter`, `program`).

Every run YAML starts with `# @package _global_` so Hydra merges it into the top-level Config.

## `input` — Data source & its initial processing parameters

| Field = Default | Purpose |
|-----------------|---------|
| `path` = — | Absolute path to the data file. Determines probe identity (pcid)[↓](#inputpath) |
| `tables` = `['incl*']` | Table groups in binary (NC/HDF5) input; glob allowed (`incl*` = all inclinometer groups). Auto-derived from the filename for text input. |
| `ids` = `None` | Process only these probe IDs (e.g. `[i01, i_p02]`). Re-run a single problematic probe without touching others. |
| `prefix` = `'I*[_0]'` | Filename prefix filter for CSV file discovery. |
| `text_type` = `None` | Column layout variant (`i`, `p`, `b`, `d`, `w`). Auto-detected from file header; override here if detection fails. |
| `text_line_regex` = `None` | Custom regex for raw text line parsing. Only needed when auto-detection fails on unusual file formats. |
| `coefs` = see [§coefs](#inputcoefs--calibration-coefficients) | Calibration coefficients and its metadata. Loaded from the calibration file on first run for a specific probe. |
| `date_to_from` = `None` | Two timestamps of one moment — [true, instrument reading]: the difference becomes the clock offset `dt_from_utc`. Fill when the instrument clock is off. |
| `dt_from_utc` = `0` | Offset from UTC in seconds. Set the timezone to convert instrument time to UTC. |
| `time_ranges` = `None` | Processing time window `[start, end, …]` in ISO format. Auto-filled from the data on first run — narrow it to process the needed period, give several pairs to skip the gaps between them[↓](#inputtime_ranges) |
| `min` = `{}` | Hard lower bounds on raw sensor values. Rows outside bounds are **removed entirely** (not just NaN'd). `M` expands to `Mx`/`My`/`Mz`[↓](#input-min-max) |
| `max` = `{}` | Hard upper bounds on raw sensor values. Same `M` expansion as `min`[↓](#inputmax) |
| `corr_time_mode` = `True` | Integer-second timestamp handling: `True` = snap to sub-second grid, `None` = mask-only, `"delete_inversions"` = clean but keep timestamps. |
| `corr_time_outlier_threshold_s` = `0.6` | Spike detection sensitivity (seconds). Lower = stricter. Samples deviating more than this from neighbors are flagged. |
| `dt_interp_between` = `1.5` | Minimum gap (seconds) to distinguish a real data hole from jitter within a burst. |
| `max_incl_of_fit_deg` = `None` | Extreme tilt angle° where the velocity curve switches to the linear tangent at Θ_last. Overrides the last element of `kVabs` — see [§Velocity computation](../methodology/velocity.md). |
| `calc_version` = `'trigonometric(incl)'` | Velocity calculation method. `trigonometric(incl)` (formulas (1)–(3)) is standard; other variants are experimental — see [§Velocity computation](../methodology/velocity.md). |
| `dt_hole_warning` = `600` | Alert threshold for data gaps (seconds). Gaps larger than this trigger a warning. `None` disables. |
| `fs_rounding` = `100` | Round estimated sampling frequency to the nearest multiple of this value. 0 = exact estimation. |
| `tables_log` = `['{}/logFiles']` | NC log group name template (`{}` → table name). |

Only `coefs` is non-optional — all other fields fall back to their defaults.
Field types: [`ConfigIn_InclProc` dataclass](../../src/tcm/schema.py)
(load-stage + calib: `ConfigInCalib_InclProc`).

### `input.path`
Absolute path to the data file.  The filename **determines probe identity** (pcid):
the pipeline extracts the leading type letter (`i` for inclinometer, `w` for wave gauge),
an optional model letter (`p`, `b`, `d`), and the probe number — e.g. `i_01.txt` → pcid
`i01`, `i_p05_data.txt` → pcid `i_p05`.  A wrong filename maps to the wrong table and
wrong coefficients.



### `input.time_ranges`
Limit data processing to one or more time windows. Auto-populated from
data edge rows on first run.

#### Detailed
Pairwise list `[start, end]`: (`[n1, k1, n2, k2, ...]`) of disjoint intervals in ISO format. Filled in at the outermost data rows upon first run. Narrow to process the desired period, specify multiple rows for gaps between them.

- The interval end is processed inclusively. If the end of the pair is not specified, everything up to the end is taken.

- When working with NetCDF, when there is already data (re-run), only new data will be appended:
Old data will only be overwritten if you changed the coefficients or parameters and explicitly indicated the need for overwriting using `out.overwrite_db`; otherwise, the program will stop. [Re-run behavior](config_tuning.md#re-run-behavior).

### `input.min` {#input-min-max}
Hard lower bound on raw sensor values — load-stage **DROP**: entire rows outside
are removed. If you need to preserve lines, use `filter.min`/`max`.

> The shorthand `M` for `Mx`, `My`, `Mz` is automatically expanded at config compose time (can be used on the command line or in yaml configuration). If individual axes are also specified, they take precedence.

```yaml
input:
  max: {M: 500}           # → Mx=500, My=500, Mz=500
  max: {Mx: 500, My: 400} # explicit overrides M for Mx, My; Mz=500 from M
```

### `input.max`
Upper mirror of `min` — same load-stage **DROP** and `M` expansion.
## `input.coefs` — Calibration coefficients
### Detailed
Loaded from the coefficient file and copied into each per-probe YAML on first run.
Edit these to update a probe's calibration — changes are persisted automatically.
[Azimuth calibration and re-run behavior](config_tuning.md).

### Table. Parameters and metadata of coefficients

| Field = Default | Physical meaning |
|-----------------|------------------|
| `Ag` = `[[1.73e-3,0,0],[0,1.73e-3,0],[0,0,1.73e-3]]` | Accelerometer scale matrix: `G = Ag @ (Axyz − Cg)` |
| `Cg` = `[10, 10, 10]` | Accelerometer bias vector |
| `Ah` = Identity | Magnetometer scale matrix: `H = Ah @ (Mxyz − Ch)` |
| `Ch` = `[10, 10, 10]` | Magnetometer bias vector |
| `Rz` = Identity | Sensor-to-instrument alignment rotation applied after calibration |
| `kVabs` = `[10, −10, −10, −3, 3, 70]` | Velocity polynomial `Vabs(inclination)`, formula (3) — see [§Velocity computation](../methodology/velocity.md) |
| `P_t` = `None` | Pressure–temperature 2‑D polynomial for `p`‑type probes; when set it supersedes `P`/`PBattery`/`PTemp`[↓](#inputcoefsp_t) |
| `P` = `[0, 1]` | Auxiliary sensor #1 linear correction: `y = P[0] + P[1]·x` |
| `PBattery` = `[0, 1]` | Battery voltage linear correction |
| `PTemp` = `[0, 1]` | Temperature linear correction |
| `azimuth_shift_deg` = `180` | Azimuth° correction — converts tilt direction from sensor to geographic coordinates; compensates magnetometer sign inversion at load time. See [Azimuth calibration](config_tuning.md#azimuth-calibration)[↓](#inputcoefsazimuth_shift_deg) |
| `dates` = `{}` | Per‑component calibration dates |
| `date` = `None` | Overall calibration date |
| `path` = `tcm/cfg/coef/calibration.h5` | Coefficient source — a directory of per-probe YAMLs or a single HDF5/NC/YAML file[↓](#inputcoefs_path) |

Field types and shapes: [`ConfigInCoefs_InclProc` dataclass](../../src/tcm/schema.py).
Resolution priority (own config > `input.coefs.path` file > bundled `yaml_export/` >
dataclass defaults): see [§Coefficient source priority](io_formats.md#coefficient-source-priority).

### `input.coefs.azimuth_shift_deg`
Azimuth° correction — converts tilt direction from sensor to geographic coordinates.

#### Detailed
**Azimuth calibration**: `input.calib.time_ranges_azimuth` specifies an interval where the
instrument was tilted in a **known direction** (e.g. known Northward tilt).
The pipeline computes the azimuth shift from calibrated mag+accel unit vectors
and writes `azimuth_shift_deg` to the per-probe YAML.

**Layering**: `input.calib.azimuth_add` (manual offset, degrees) and
`input.calib.coordinates` (magnetic declination, current date) are applied
**on top of** the data-computed azimuth.

### `input.coefs.P_t`
Temperature-compensated pressure polynomial — converts raw pressure counts and
temperature into physical pressure.

#### Detailed
2-D polynomial `polyval2d(u, t, P_t)` in raw pressure counts `u`
(`P`/`P_counts`) and temperature `t` (`Temp`); `P_t[i][j]` multiplies
`u^i·t^j` (six coefficients of total degree ≤ 2). Computed pressure is as
calibrated — referenced to standard atmospheric pressure P0 = 10.1325 dbar.
Formula and provenance: [§Pressure computation](../methodology/pressure.md).


### `input.coefs.path` {#inputcoefs_path}
Coefficient source — a directory of per-probe YAMLs or a single YAML file. Or NetCDF/HDF5: the group {g} matching the probe is selected.

#### Detailed

Path to the configuration file. Must contain `input.coefs` coefficients. Possible:

- single coefficient source file: HDF5 (`.h5`), NetCDF4 (`.nc`) or coefficient YAML config (`.yaml`).
- directory of per-probe YAMLs must contain files `{g}.yaml`, where {g} is the probe identifier (`incl_{model#}.yaml`)
- missing/incorrect path — allowed if all required parameters are already set manually (they have priority over file data), another attempt will be made to find the needed coefficients from the bundled `yaml_export/`: [priority chain](io_formats.md#coefficient-source-priority).


## `input.calib` — Process-stage calibration correction

Applied at process stage, after loading


| Field = Default | Physical meaning |
|-----------------|------------------|
| `g0xyz` = `None` | User-defined gravity reference vector. When set, overrides `Rz` with a computed rotation[↓](#inputcalibg0xyz) |
| `time_ranges_zeroing` = `[]` | Intervals where the instrument hung level. Pipeline computes a rotation to align sensor Z with gravity (`Rz`)[↓](#inputcalibtime_ranges_zeroing) |
| `time_ranges_azimuth` = `[]` | Intervals where the instrument was tilted in a known direction. Pipeline calibrates the azimuth shift (`azimuth_shift_deg`) from mag+accel unit vectors[↓](#inputcalibtime_ranges_azimuth) |
| `coordinates` = `None` | Station `[Lat, Lon]` — enables magnetic declination correction (true-north velocity directions)[↓](#inputcalibcoordinates) |
| `azimuth_add` = `0` | Manual azimuth° fine-tuning, added on top of the data-calibrated shift[↓](#inputcalibazimuth_add) |
Field types: [`ConfigInCalib_InclProc` dataclass](../../src/tcm/schema.py).
Applied by :func:`tcm._xr.coefs.prepare_coefs`.


### `input.calib.g0xyz`

#### Detailed
Raw accelerometer vector `[Ax, Ay, Az]` measured at known zero tilt. When set,
it **overrides** any existing `Rz` — computes rotation to align sensor Z with
gravity directly, bypassing `time_ranges_zeroing`.

### `input.calib.time_ranges_zeroing`
Intervals where the instrument hung level — pipeline computes the `Rz` rotation
aligning the sensor Z-axis with gravity; written back to the probe YAML.

#### Detailed
The instrument hangs plumb. The pipeline averages accelerometer data over the
window and computes the rotation matrix `Rz`. Alternative:
[`input.calib.g0xyz`](#inputcalibg0xyz) (raw accel vector at known zero tilt)
overrides any data-computed `Rz`.

### `input.calib.time_ranges_azimuth`
Intervals where the instrument was tilted in a **known direction** — pipeline
computes `azimuth_shift_deg` from calibrated mag+accel unit vectors; written
back to the probe YAML.

#### Detailed
The azimuth computation uses calibrated unit vectors only (no velocity/magnitude
calculation), so it does not depend on `kVabs` or inclination-to-magnitude
coefficients.

### `input.calib.coordinates`
Station `[Lat, Lon]` in decimal degrees — enables magnetic declination
correction, converting velocity directions from magnetic to true north.
Declination is evaluated for the current date at the station location.

#### Detailed
Applied on top of the data-computed azimuth shift together with
`input.calib.azimuth_add` — see
[`input.coefs.azimuth_shift_deg`](#inputcoefsazimuth_shift_deg) for the layering
order.

### `input.calib.azimuth_add`

#### Detailed
Layering: `azimuth_add` (manual offset, degrees) and `coordinates` (magnetic
declination via `pygeomag`) are applied **after** the data-computed azimuth
shift, before velocity direction is resolved — see
[`input.coefs.azimuth_shift_deg`](#inputcoefsazimuth_shift_deg).

```yaml
input:
  calib:
    time_ranges_zeroing: ["2026-06-25T17:23:30", "2026-06-25T17:25:00"]
    time_ranges_azimuth: ["2026-06-25T17:23:30", "2026-06-25T17:25:00"]
    coordinates: [54.70, 20.51]   # Kaliningrad
    azimuth_add: 2.5              # manual fine-tune
```




## `out` — Output configuration

| Field = Default | Purpose |
|-----------------|---------|
| `db_path` = `None` | Combined multi-probe output (`.proc.nc`) — all probes merged along a `probe` dimension. |
| `avg_db_path` = `None` | Per-probe binned output (`.proc_Avg.nc`) — one group per probe per bin size. |
| `not_joined_db_path` = `None` | Per-probe full-resolution output (`.proc_noAvg.nc`) — every sample preserved. |
| `raw_db_path` = `None` | Raw data archive (`.raw.nc`) — unprocessed readings + calibration coefficients for later reprocessing. |
| `table` = `''` | Override output table name. When non-empty, replaces the auto-derived pcid for text-file suffixes and HDF5 group names. |
| `tables_log` = `['{}/logFiles']` | NC log group name template(s) for output storage (`{}` → table name). |
| `dt_bins` = `[0, 2, 600, 3600, 7200]` | Time averaging bins (seconds). `0` = full resolution. Multiple values produce separate outputs (e.g. `[0, 600, 3600]` = full + 10 min + 1 h). |
| `dt_bins_min_save_text` = `1` | Minimum averaging bin (s) for the text export (`*.TSV`). Full-resolution data is saved only when this is 0. |
| `split_period` = `''` | Split output into time blocks (e.g. `'1D'` = daily files). Empty = single continuous output. |
| `text_path` = `'text_output'` | Directory for TSV output files. Created automatically if missing. |
| `text_date_format` = `'%Y-%m-%d %H:%M:%S.%f'` | Date format string for TSV timestamps. |
| `text_columns` = `[]` | Filter which columns appear in TSV output. Empty = all available. Columns listed but absent from a particular output are silently skipped. |
| `b_all_to_one_col` = `False` | Multi-probe layout: `False` = interleave columns (`v_i01, u_i01, …`), `True` = stack rows. |
| `b_overwrite_text` = `True` | Overwrite existing TSV files. Set `False` to keep previous exports. |
| `b_split_by_time_ranges` = `False` | Split output by `time_ranges` boundaries — each interval gets its own file. |
| `b_del_temp_db` = `False` | Delete temporary HDF5 files after processing. |
| `overwrite_db` = `None` | NC overwrite strategy. `None` = append-only (safe). `"splice"` = replace overlapping, keep rest. `"trim"` = delete outside `time_ranges`. `"export"` = TSV only, no NC writes. See [§overwrite_db](#outoverwrite_db)[↓](#outoverwrite_db) |

Non-optional: `dt_bins` and `text_path`. Field types:
[`ConfigOut_InclProc` dataclass](../../src/tcm/schema.py).

### `out.overwrite_db`

Controls how the pipeline handles existing processed output when re-running.

| `overwrite_db` | Params changed? | `time_ranges` vs existing | Behavior |
|:---:|:---:|:---:|---|
| `None` | No | subset | **Skip NC** — export TSV only |
| `None` | No | extends | **Append** — append new tail only |
| `None` | Yes | extends | **Append + warn** — keep existing, append new |
| `None` | Yes | contained | **Error** — suggest `out.overwrite_db=splice` |
| `"splice"` | — | subset | **Splice** — keep outside, replace inside with reprocessed |
| `"splice"` | — | extends | **Splice** — keep outside, replace/append inside |
| `"splice"` | — | None | **Splice** — reprocess all from source |
| `"trim"` | — | subset | **Trim** — delete outside `time_ranges`, no reprocessing |
| `"trim"` | — | extends | **Trim + append** — trim existing, process/append new |
| `"export"` | — | any | **Export only** — block NC writes, export TSV |

A re-run of the computation (when NetCDF read/write available) with changed parameters inside a
previously written window goes through `"splice"` only.
Full contract (append positions, log-based dedup) —
[re-run behavior](config_tuning.md#re-run-behavior).

## `filter` — Process-stage quality thresholds

`filter` is **process-stage**: values exceeding thresholds become NaN (rows
**kept**, not dropped).  Contrast with `input.min`/`max` — **load-stage DROP**
that removes entire rows.  Same key names may appear in both namespaces with
different semantics.

| Field = Default | Purpose |
|-----------------|---------|
| `min` = `{}` | Lower thresholds: readings below the threshold become NaN (rows kept). |
| `max` = `{'g_minus_1': 1, 'h_minus_1': 8}` | Upper thresholds: readings above the threshold become NaN (rows kept)[↓](#filtermax) |
| `bad_p_at_bursts_starts_period` = `''` | Pressure burst cleanup period in pandas format (e.g. `'1h'`). Nulls the first 2 samples per burst to remove startup artifacts. Empty disables. |

Field types: [`ConfigFilter_InclProc` dataclass](../../src/tcm/schema.py).

### `filter.max`
Upper bounds on process-computed columns — values beyond become NaN (rows kept).

#### Detailed
Threshold keys address process-computed columns: `g_minus_1 = ∥Gxyz∥ − 1`
(gravity magnitude deviation), `h_minus_1 = ∥Hxyz∥ − 1` (magnetic magnitude
deviation); `M` expands to `Mx`/`My`/`Mz` exactly as in `input.min`/`max`.

### Calibration filter extensions (`filter/calib`)

When the calibration entry point uses `filter: calib`, the filter group adds
typed despike overrides:

| Field | Type | Purpose |
|-------|------|---------|
| `blocks` | `List[int]` | Apex despike block sizes (default `[21, 7]`) |
| `offsets` | `List[float]` | Apex despike offset thresholds |
| `std_smooth_sigma` | `float` | Apex despike smoothing sigma |
| `A` | `ConfigFilterChannel` | Per-axis overrides for accelerometer |
| `M` | `ConfigFilterChannel` | Per-axis overrides for magnetometer |
| `no_works_noise` | `Dict[str, float]` | Noise threshold per channel |

## `proc` — Per-entry-point processing parameters (optional)

`proc` is an **optional** group per entry point. Processing entry has none.

| Entry point | `proc` option | Dataclass | Purpose |
|-------------|--------------|-----------|---------|
| Processing | *(none)* | — | All processing params live in `out.dt_bins` + `input.calc_version` |
| Calibration | `calib` | `ConfigProcCalib` | Maps to `PipelineConfig` fields |
| Spectrum | `spectrum` | `ConfigProcSpectrum` | **Reserved** — spectrum module not ported yet |

## `program` — Runtime flags

| Field = Default | Purpose |
|-----------------|---------|
| `b_interact` = `False` | Confirmation prompt before creating directories; when disabled, proceeds without prompting. |
| `log` = `''` | Log file path base (without extension); empty = auto-named under `cfg_proc/log/`. |
| `verbose` = `'INFO'` | Console log verbosity. `'DEBUG'` for troubleshooting, `'INFO'` for normal runs. |
| `use_h5` = `'auto'` | Binary I/O policy. `auto` = use if available, skip silently. `off` = disable NC/HDF5. `require` = error if unavailable. `prefer` = warn and fall back. |
| `return_` = `'<end>'` | How far to run the processing (e.g. `'<saved_raw>'` — to verify the raw data ingestion). See [Phase-stopping](config_tuning.md#phase-stopping). |

Field types: [`ConfigProgram` dataclass](../../src/tcm/schema.py).

## `metadata` — Device deployment metadata (per-probe `info_devices.yaml`)

Fields describe the instrument deployment and are saved to `info_devices.yaml` above the
raw data directory — the journal accompanying the data.

| Field = Default | Indices | Purpose |
|-----------------|---------|---------|
| `point` = `?` | 0 | Station point identifier |
| `symbol` = `?` | 3 | Modification / instrument symbol (e.g. `↟`) |
| `sea_depth` = `?` | 1 | Sea depth (m) |
| `h_above` = `?` | 2 | Height above bottom (m) |
| `lat` = `?` | 4 | Latitude, decimal degrees |
| `lon` = `?` | 5 | Longitude, decimal degrees |
| `time_range` = `?` | 6, 7 | The start and end of the correct operation of the device at the station[↓](#metadatatime_range) |
| `burst_dt` = `?` | 8 | Duration of the active (continuous) recording (s) when working with interruptions |
| `bursts_t` = `?` | 9 | Recording start period (s) when working with interruptions |
| `comment` = `?` | 10 | Free-form comment |


### Detailed

[Device deployment metadata](../user_guide/meta_finder.md#device-metadata-file), stored in `info_devices.yaml` as an 11-element array or less: a trailing NaN after the 8th element is not written. `?, -, "", ~` are equivalent to NaN, written in YAML as `~`.

`?, -, "", ~` are equivalents of missing data (NaN — written to YAML as `~`; an all-NaN tail after the 8th element is not written at all).

> In the GUI the rows are paired: `point, symbol | sea depth, h_above | lat, lon | time_range | burst_dt/t | comment`. You can specify your own save path, not the one from which metadata is loaded when searching for data. `time_range` ↔ `input.time_ranges[[0,-1]]` are bidirectionally synced where unset, on scan.

### `metadata.path`

Path to the deployment metadata: loaded from `info_devices.yaml` when scaned for data.

#### Detailed
By default, changes are saved to the same location they were downloaded from: `info_devices.yaml` in the parent directory of the `_raw` directory or the raw data file directory, if not in a (sub)directory of `_raw`. Don't change the path if you want to save metadata changes in the file that is automatically loaded during scan. Write - on run the processing.
### `metadata`



### `metadata.time_range`

#### Detailed
Does not affect the current processing interval — a record for the deployment journal
`info_devices.yaml` only.
On scan only (not a processing run): when the `input.time_ranges` in the config file is
unset or incomplete, its missing ends are updated from this record. Details —
[deployment metadata](../user_guide/meta_finder.md#tcm-gui-editor-for-metadata-records).

### `program.return_`

#### Detailed
| `return_` value | Stops after | Typical use |
|:---|:---|:---|
| `<cfg_from_args>` | Config composition (no I/O) | Scan input, generate missing configs |
| `<saved_coefs>` | Coef persistence only | Zeroing/azimuth → save coefs, stop |
| `<saved_raw>` | Raw NC save | Verify raw ingestion |
| `<saved_noavg>` | No-avg output | Diagnostic without full binning |
| `<saved_all>` | All binned NC writes | Skip combined output |
| `<end>` (default) | Full pipeline | Normal processing |

## See also
> **Behavior tuning** (phase-stopping, time correction modes, column order,
> `overwrite_db`, azimuth calibration, YAML examples) — in [config tuning](config_tuning.md).
> Internals — in [CLI internals](../project_developer_guide/CLI.md).