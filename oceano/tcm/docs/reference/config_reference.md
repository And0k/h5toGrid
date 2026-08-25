# Config YAML Field Reference

Each run YAML (`cfg_proc/run/{source_stem}.yaml`) is a structured Hydra/OmegaConf config.
All fields are defined in `tcm/schema.py` via the `Config` dataclass and registered groups
(`input`, `out`, `filter`, `program`).

Every run YAML starts with `# @package _global_` so Hydra merges it into the top-level Config.

> **Behavior tuning & decision tables** (phase-stopping, time correction modes,
> column order, `overwrite_db`, azimuth calibration, YAML examples) are in
> [config_tuning.md](config_tuning.md).  Implementation internals live in
> [../project_developer_guide/CLI.md](../project_developer_guide/CLI.md).

## `input` — Data source & parameters

| Field = Default | Purpose |
|-----------------|---------|
| `path` = — | Absolute path to the data file. Determines probe identity (pcid). |
| `tables` = `['incl*']` | Table groups in binary (NC/HDF5) input; glob allowed (`incl*` = all inclinometer groups). Auto-derived from the filename for text input. |
| `ids` = `None` | Process only these probe IDs (e.g. `[i01, i_p02]`). Re-run a single problematic probe without touching others. |
| `prefix` = `'I*[_0]'` | Filename prefix filter for CSV file discovery. |
| `text_type` = `None` | Column layout variant (`i`, `p`, `b`, `d`, `w`). Auto-detected from file header; override here if detection fails. |
| `text_line_regex` = `None` | Custom regex for raw text line parsing. Only needed when auto-detection fails on unusual file formats. |
| `coefs` = see [§coefs](#inputcoefs--calibration-coefficients) | Calibration coefficients — the heart of measurement accuracy. Auto-loaded on first run; edit here to fine-tune a specific probe. |
| `coefs_path` = `tcm/cfg/coef/calibration.h5` | Coefficient source — a directory of per-probe YAMLs or a single HDF5/NC/YAML file; see the dir/file details below. |
| `date_to_from` = `None` | Two timestamps `[real_time, raw_time]` — their offset becomes `dt_from_utc`. |
| `dt_from_utc` = `0` | UTC offset in seconds. Set to your timezone to convert instrument time to UTC. |
| `min_date` = `None` | Convenience shorthand for `time_ranges` — automatically merged. |
| `max_date` = `None` | Convenience shorthand for `time_ranges` — automatically merged. |
| `time_ranges` = `None` | Time window for processing `[start, end, …]` in ISO format. Auto-populated from data on first run — narrow it to focus on specific periods. |
| `min` = `{}` | Hard lower bounds on raw sensor values. Rows outside bounds are **removed entirely** (not just NaN'd). `M` expands to `Mx`/`My`/`Mz`. |
| `max` = `{}` | Hard upper bounds on raw sensor values. Same `M` expansion as `min`. |
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

### `input.path` <mode>probe</mode>
Absolute path to the data file.  The filename **determines probe identity** (pcid):
the pipeline extracts the leading type letter (`i` for inclinometer, `w` for wave gauge),
an optional model letter (`p`, `b`, `d`), and the probe number — e.g. `i_01.txt` → pcid
`i01`, `i_p05_data.txt` → pcid `i_p05`.  A wrong filename maps to the wrong table and
wrong coefficients.

### `input.path` <mode>search</mode>
supports glob (`*i*.txt`), regex (`i.*\.txt`), or directory.
`.yaml` suffix filters existing configs by stem.

#### Detailed
The search path anchors data + config discovery. Accepted forms:
- **directory** (e.g. `B:\Cruises\BalticSea\`) — scan for raw data files and a
  `cfg_proc/run/` subfolder; expects a `_raw`/`proc` layout (below).
- **glob** (`*i*.txt`) / **regex** (`i.*\.txt`) — match data files by name.
- **`.yaml`** path — load existing configs directly, skip discovery.

Expected directory layout ([see also](io_formats.md#directory-layout)):
```text
├── _raw\            ← raw data files (.txt/.csv/.h5/.nc) — REQUIRED
├── cfg_proc\
│   └── run\         ← per-probe YAML configs (auto-generated on first scan)
```

### `input.coefs_path` <mode>dir</mode>
Directory of per-probe coefficient YAMLs — the pipeline reads `{tbl}.yaml`
(e.g. `incl_p05.yaml`) following the `input.coefs` structure. No file for the
probe here → the next source of the [resolution
chain](io_formats.md#coefficient-source-priority) is used (bundled
`yaml_export/`, then defaults); values already set in the probe's own config
take priority over this directory.

### `input.coefs_path` <mode>file</mode>
Single coefficient source file: HDF5 (`.h5`), NetCDF4 (`.nc`), or
exported YAML (`.yaml`).  All probes share the file — the pipeline
selects the group by table name.  Comma-separated paths are accepted
(fallback chain, first match wins).

### `input.time_ranges`
Time window for processing — restricts to data within it; auto-populated from
data edge rows on first run.

#### Detailed
Two-element list `[start, end]` in ISO format (`"YYYY-MM-DDTHH:MM:SS"`).
Multiple pairs are accepted (`[s1, e1, s2, e2, ...]`) for disjoint intervals.

**Interaction with `overwrite_db`**: when `overwrite_db=None` and
`time_ranges` is a subset of existing NC data, the pipeline checks stored
`/param_spans/{tbl}` parameters — if they changed, `ValueError` is raised
with a unified diff. Pass `out.overwrite_db=splice` to force reprocessing.
When `time_ranges` extends beyond existing data, only the new tail is appended.

**GUI hover** (compared live to `info_devices`): _matches_ (`kept`) when equal, _broader than_ when extending beyond either end (warning tint), _differs_ when narrowed or shifted — the status bar text is recomputed on every hover/edit, never cached from the scan.  See [GUI internals](../project_developer_guide/GUI.md#coef_sheetpy--composition-root--treeeditrow-space-core).

**End-bound semantics**: end values are **inclusive** in the config. Internally
they are converted to exclusive bounds (whole-second ends get +1 s) to prevent
boundary data loss from CF float64 precision drift.

### `input.min` {#input-min-max}
Hard lower bound on raw sensor values — load-stage **DROP**: entire rows outside
are removed. Contrast `filter.min`/`filter.max` — process-stage NaN-out, rows kept.

#### Detailed
`M` is a shorthand for `Mx`, `My`, `Mz`. If `M` is set but individual axes
are not, the value is copied to all three:

```yaml
input:
  max: {M: 500}           # → Mx=500, My=500, Mz=500
  max: {Mx: 500, My: 400} # explicit overrides M for Mx, My; Mz=500 from M
```

The `M` shorthand expands automatically at config compose time.

### `input.max`
Upper mirror of `min` — same load-stage **DROP** and `M` expansion.

## `input.calib` — Process-stage calibration correction

Applied at process stage, after loading, by :func:`tcm._xr.coefs.prepare_coefs`.
Field types: [`ConfigInCalib_InclProc` dataclass](../../src/tcm/schema.py).

| Field = Default | Physical meaning |
|-----------------|------------------|
| `g0xyz` = `None` | User-defined gravity reference vector. When set, overrides `Rz` with a computed rotation. |
| `time_ranges_zeroing` = `[]` | Intervals where the instrument hung level. Pipeline computes a rotation to align sensor Z with gravity (`Rz`). |
| `time_ranges_azimuth` = `[]` | Intervals where the instrument was tilted in a known direction. Pipeline calibrates the azimuth shift (`azimuth_shift_deg`) from mag+accel unit vectors. |
| `coordinates` = `None` | Station `[Lat, Lon]` — enables magnetic declination correction (true-north velocity directions). |
| `azimuth_add` = `0` | Manual azimuth° fine-tuning, added on top of the data-calibrated shift. |

### `input.calib.g0xyz`
User-defined gravity reference vector.

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
Manual azimuth° fine-tuning, added on top of the data-calibrated shift.

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

## `input.coefs` — Calibration coefficients

Loaded from the coefficient file and copied into each per-probe YAML on first run.
Edit these to update a probe's calibration — changes are persisted automatically.

| Field = Default | Physical meaning |
|-----------------|------------------|
| `Ag` = `[[1.73e-3,0,0],[0,1.73e-3,0],[0,0,1.73e-3]]` | Accelerometer scale matrix: `G = Ag @ (Axyz − Cg)` |
| `Cg` = `[10, 10, 10]` | Accelerometer bias vector |
| `Ah` = Identity | Magnetometer scale matrix: `H = Ah @ (Mxyz − Ch)` |
| `Ch` = `[10, 10, 10]` | Magnetometer bias vector |
| `Rz` = Identity | Sensor-to-instrument alignment rotation applied after calibration |
| `kVabs` = `[10, −10, −10, −3, 3, 70]` | Velocity polynomial `Vabs(inclination)`, formula (3) — see [§Velocity computation](../methodology/velocity.md) |
| `P_t` = `None` | Pressure–temperature 2‑D polynomial for `p`‑type probes; when set it supersedes `P`/`PBattery`/`PTemp`. |
| `P` = `[0, 1]` | Auxiliary sensor #1 linear correction: `y = P[0] + P[1]·x` |
| `PBattery` = `[0, 1]` | Battery voltage linear correction |
| `PTemp` = `[0, 1]` | Temperature linear correction |
| `azimuth_shift_deg` = `180` | Azimuth° correction — converts tilt direction from sensor to geographic coordinates; compensates magnetometer sign inversion at load time. See [Azimuth calibration](config_tuning.md#azimuth-calibration). |
| `dates` = `{}` | Per‑component calibration dates |
| `date` = `None` | Overall calibration date |

Field types and shapes: [`ConfigInCoefs_InclProc` dataclass](../../src/tcm/schema.py).
Resolution priority (own config → `coefs_path` file → bundled `yaml_export/` →
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

## `out` — Output configuration

| Field = Default | Purpose |
|-----------------|---------|
| `db_path` = `None` | Combined multi-probe output (`.proc.nc`) — all probes merged along a `probe` dimension. |
| `avg_db_path` = `None` | Per-probe binned output (`.proc_Avg.nc`) — one group per probe per bin size. |
| `not_joined_db_path` = `None` | Per-probe full-resolution output (`.proc_noAvg.nc`) — every sample preserved. |
| `raw_db_path` = `None` | Raw data archive (`.raw.nc`) — unprocessed readings + calibration coefficients for later reprocessing. |
| `table` = `''` | Override output table name. When non-empty, replaces the auto-derived pcid for text-file suffixes and HDF5 group names. |
| `tables_log` = `['{}/logFiles']` | NC log group name template(s) for output storage (`{}` → table name). |
| `b_incremental_update` = `True` | Incremental-append mode of the HDF5 pipeline — replaced in the NC pipeline by the `overwrite_db` decision logic. |
| `b_overwrite` = `False` | HDF5-pipeline overwrite flag — replaced by `overwrite_db`; unused by the NC pipeline. |
| `dt_bins` = `[0, 2, 600, 3600, 7200]` | Time averaging bins (seconds). `0` = full resolution. Multiple values produce separate outputs (e.g. `[0, 600, 3600]` = full + 10 min + 1 h). |
| `dt_bins_min_save_text` = `1` | Minimum bin size (seconds) for TSV export. Bin=0 skipped when >0. Set to 0 to include full-resolution data in text output. |
| `split_period` = `''` | Split output into time blocks (e.g. `'1D'` = daily files). Empty = single continuous output. |
| `text_path` = `'text_output'` | Directory for TSV output files. Created automatically if missing. |
| `text_date_format` = `'%Y-%m-%d %H:%M:%S.%f'` | Date format string for TSV timestamps. |
| `text_columns` = `[]` | Filter which columns appear in TSV output. Empty = all available. Columns listed but absent from a particular output are silently skipped. |
| `b_all_to_one_col` = `False` | Multi-probe layout: `False` = interleave columns (`v_i01, u_i01, …`), `True` = stack rows. |
| `b_overwrite_text` = `True` | Overwrite existing TSV files. Set `False` to keep previous exports. |
| `b_split_by_time_ranges` = `False` | Split output by `time_ranges` boundaries — each interval gets its own file. |
| `b_del_temp_db` = `False` | Delete temporary HDF5 files after processing. |
| `overwrite_db` = `None` | NC overwrite strategy. `None` = append-only (safe). `"splice"` = replace overlapping, keep rest. `"trim"` = delete outside `time_ranges`. `"export"` = TSV only, no NC writes. See [§overwrite_db](#outoverwrite_db). |

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

When processing parameters changed and `overwrite_db=None`, the pipeline
compares stored `/param_spans/{tbl}` interval table to the current values
and raises `ValueError` with a unified diff. See
[config_tuning.md](config_tuning.md) for the full contract including
incremental append positions and log-based dedup.

## `filter` — Process-stage quality thresholds

`filter` is **process-stage**: values exceeding thresholds become NaN (rows
**kept**, not dropped).  Contrast with `input.min`/`max` — **load-stage DROP**
that removes entire rows.  Same key names may appear in both namespaces with
different semantics.

| Field = Default | Purpose |
|-----------------|---------|
| `min` = `{}` | Lower bounds: values with `\|col\| < min[col]` set to NaN. |
| `bad_p_at_bursts_starts_period` = `''` | Pressure burst cleanup period (e.g. `'1h'`). Nulls first 2 samples per burst to remove startup artifacts. Empty disables. |
| `max` = `{'g_minus_1': 1, 'h_minus_1': 8}` | Upper bounds. `M` expands to `Mx`/`My`/`Mz`. |

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
| `return_` = `'<end>'` | Pipeline exit point. Run partial processing for debugging (e.g. `'<saved_raw>'` to verify data ingestion). See [Phase-stopping](config_tuning.md#phase-stopping). |

Field types: [`ConfigProgram` dataclass](../../src/tcm/schema.py).

## `metadata` — Device deployment metadata (per-probe `info_devices.yaml`)

Paired GUI rows ↔ 11-array indices (see `tcm/_meta_pairs.py:PAIRS`).

| Field = Default | Indices | Purpose |
|-----------------|---------|---------|
| `path` = — | — | Device file path (directory of `info_devices.yaml`) — browseable |
| `point` = `?` | 0 | Station point identifier |
| `symbol` = `?` | 3 | Modification / instrument symbol (e.g. `↟`) |
| `sea_depth` = `?` | 1 | Sea depth (m) |
| `h_above` = `?` | 2 | Height above bottom (m) |
| `lat` = `?` | 4 | Latitude, decimal degrees |
| `lon` = `?` | 5 | Longitude, decimal degrees |
| `time_range` = `?` | 6, 7 | Deployment interval `[time_st, time_en]` ISO |
| `burst_dt` = `?` | 8 | Burst sampling dt (s) |
| `bursts_t` = `?` | 9 | Burst interval T (s) |
| `comment` = `?` | 10 | Free-form comment |

In the GUI the paired rows show `point, symbol | sea depth, h_above | lat, lon | time_range | burst_dt/t | comment` with gray example placeholders (e.g. `P3, 7.5, 54.62`) that vanish on edit — identical to `CellPlaceholder` for dates. `?, -, "", ~ (null)` are placeholders; required `0..time_en(7)` writes `~` when placeholder, optional tail `8..10` is trimmed if empty. `time_range` ↔ `input.time_ranges[[0,-1]]` bidirectionally synced.

### `metadata.path`
Directory containing `info_devices.yaml` (parent of `_raw`). Click to browse — same floating editor as `input.path`.

### `metadata.point`
Station / point identifier (deployment location name).

### `metadata.symbol`
Modification / instrument symbol (e.g. `↟`).

### `metadata.sea_depth`
Sea depth at deployment point (m).

### `metadata.h_above`
Height above bottom (m).

### `metadata.lat`
Latitude, decimal degrees.

### `metadata.lon`
Longitude, decimal degrees.

### `metadata.time_range`
Deployment time interval — start and end timestamps (`YYYY-MM-DDTHH:MM:SS`).

#### Detailed
When `info_devices.yaml` provides `time_range` but the run YAML's `input.time_ranges` is missing or has <2 elements, absent ends are filled from device metadata. Device `time_range` is edited in the `metadata` node; `time_ranges` remains the processing window.

### `metadata.burst_dt`
Burst sampling dt (s).

### `metadata.bursts_t`
Bursts interval T (s).

### `metadata.comment`
Free-form comment for the deployment.

### `program.return_`
Pipeline exit point — run partial processing for debugging.

#### Detailed
| `return_` value | Stops after | Typical use |
|:---|:---|:---|
| `<cfg_from_args>` | Config composition (no I/O) | Scan input, generate missing configs |
| `<saved_coefs>` | Coef persistence only | Zeroing/azimuth → save coefs, stop |
| `<saved_raw>` | Raw NC save | Verify raw ingestion |
| `<saved_noavg>` | No-avg output | Diagnostic without full binning |
| `<saved_all>` | All binned NC writes | Skip combined output |
| `<end>` (default) | Full pipeline | Normal processing |
