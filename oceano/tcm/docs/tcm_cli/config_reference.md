# Config YAML Field Reference

Each run YAML (`cfg_proc/run/{source_stem}.yaml`) is a structured Hydra/OmegaConf config.
All fields are defined in `tcm/schema.py` via the `Config` dataclass and registered groups
(`input`, `out`, `filter`, `program`).

Every run YAML starts with `# @package _global_` so Hydra merges it into the top-level Config.

> **Behavior tuning & decision tables** (phase-stopping, time correction modes,
> column order, `overwrite_db`, azimuth calibration, YAML examples) are in
> [config_tuning.md](config_tuning.md).  Implementation internals live in
> [how_it_works.md](how_it_works.md).

## `input` — Data source & parameters

| Field | Type | Default | Required | Purpose |
|-------|------|---------|----------|---------|
| `path` | `str` | — | **Yes** | Absolute path to the data file. Determines probe identity (pcid). |
| `tables` | `List[str]` | `['incl*']` | No | Data groups to read — accepts regex (`incl*` matches all inclinometer groups). Auto-derived from filename for CSV. |
| `ids` | `List[str]` | `None` | No | Process only these probe IDs (e.g. `[i01, i_p02]`). Re-run a single problematic probe without touching others. |
| `prefix` | `str` | `'I*[_0]'` | No | Filename prefix filter for CSV file discovery. |
| `text_type` | `str` | `None` | No | Column layout variant (`i`, `p`, `b`, `d`, `w`). Auto-detected from file header; override here if detection fails. |
| `text_line_regex` | `str` | `None` | No | Custom regex for raw text line parsing. Only needed when auto-detection fails on unusual file formats. |
| `coefs` | `ConfigInCoefs_InclProc` | defaults (see [§coefs](#inputcoefs--calibration-coefficients)) | **Yes** | Calibration coefficients — the heart of measurement accuracy. Auto-loaded on first run; edit here to fine-tune a specific probe. |
| `coefs_path` | `str` | `tcm/cfg/coef/calibration.h5` | No | Path to coefficient source file (HDF5 or YAML directory). Falls back to bundled `cfg/coef/yaml_export/` silently when missing. |
| `date_to_from` | `List[Any]` | `None` | No | Two-point time correction `[real_time, raw_time]`. Maps raw instrument timestamps to real-world time via linear interpolation. |
| `dt_from_utc` | `int` | `0` | No | UTC offset in seconds. Set to your timezone to convert instrument time to UTC. |
| `min_date` | `str` | `None` | No | Convenience shorthand for `time_ranges` — automatically merged. |
| `max_date` | `str` | `None` | No | Convenience shorthand for `time_ranges` — automatically merged. |
| `time_ranges` | `List[str]` | `None` | No | Time window for processing `[start, end, …]` in ISO format. Auto-populated from data on first run — narrow it to focus on specific periods. |
| `min` | `Dict[str, float]` | `{}` | No | Hard lower bounds on raw sensor values. Rows outside bounds are **removed entirely** (not just NaN'd). `M` expands to `Mx`/`My`/`Mz`. |
| `max` | `Dict[str, float]` | `{}` | No | Hard upper bounds on raw sensor values. Same `M` expansion as `min`. |
| `corr_time_mode` | `[bool, str, None]` | `True` | No | Integer-second timestamp handling: `True` = snap to sub-second grid, `None` = mask-only, `"delete_inversions"` = clean but keep timestamps. |
| `corr_time_outlier_threshold_s` | `float` | `0.6` | No | Spike detection sensitivity (seconds). Lower = stricter. Samples deviating more than this from neighbors are flagged. |
| `dt_interp_between` | `float` | `1.5` | No | Minimum gap (seconds) to distinguish a real data hole from jitter within a burst. |
| `coordinates` | `List[float]` | `None` | No | Station `[Lat, Lon]` for magnetic declination — enables true-north velocity directions. |
| `time_ranges_zeroing` | `List[str]` | `[]` | No | Time intervals where the instrument was level. Pipeline computes a zeroing rotation to remove sensor misalignment. |
| `time_ranges_azimuth` | `List[str]` | `[]` | No | Time intervals where the instrument was tilted in a known direction. Pipeline calibrates azimuth correction from this data. |
| `azimuth_add` | `float` | `0` | No | Manual azimuth fine-tuning (degrees), added on top of the data-calibrated shift. |
| `max_incl_of_fit_deg` | `float` | `None` | No | Inclination angle (degrees) above which the velocity curve flattens — used to calibrate the kVabs polynomial for extreme tilts. |
| `calc_version` | `str` | `'trigonometric(incl)'` | No | Velocity calculation method. `trigonometric(incl)` is standard; other variants are experimental. |
| `dt_hole_warning` | `int` | `600` | No | Alert threshold for data gaps (seconds). Gaps larger than this trigger a warning. `None` disables. |
| `fs_rounding` | `int` | `100` | No | Round estimated sampling frequency to the nearest multiple of this value. 0 = exact estimation. |
| `tables_log` | `List[str]` | `['{}/logFiles']` | No | NC log group name template. Default `{}/logFiles` works for standard layouts. |

> **Field detail sections** — When a field's meaning depends on context (per-probe
> processing vs. input specification), the table cell stays minimal and detailed
> documentation goes into `###` subsections tagged with a **mode**:
>
> | Mode | Content regime | Example consumer |
> |------|---------------|------------------|
> | `<mode>probe</mode>` | Per-probe processing meaning — what the field does, how it affects results | GUI coef hover, popup |
> | `<mode>search</mode>` | Input specification patterns — glob, regex, directory, YAML | GUI path field, CLI help |
>
> The GUI compares its current context to the `<mode>` tag and selects matching
> content.  Add new modes as `### \`field.path\` <mode>value</mode>` subheadings;
> the parser recognizes any `[a-z_]+` value.  Keep table cells to one sentence.

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
Top PathField = CLI first positional argument — anchors data + config discovery.

Accepted forms:
- **directory** (e.g. `B:\Cruises\BalticSea\`) — pipeline scans for raw data
  files and a `cfg_proc/run/` subfolder; expects a `_raw`/`proc` layout (below).
- **glob** (`*i*.txt`) / **regex** (`i.*\.txt`) — match data files by name.
- **`.yaml`** path — load pre-built configs directly, skip discovery.

Expected directory layout:
```
{path}\
├── _raw\            ← raw data files (.txt/.csv/.h5/.nc) — REQUIRED
├── cfg_proc\
│   └── run\         ← per-probe YAML configs (auto-generated on first scan)
├── proc\            ← pipeline NC output (created on Run)
└── text_output\     ← TSV export (created on Run)
```

Common errors:
- `FileNotFoundError: No input files found matching …` — `_raw` missing or
  empty; point at the **root** cruise directory, not a leaf.
- `SystemExit` from `_print_usage_error` — path not resolved; verify absolute
  and the directory exists.

### `input.coefs_path` <mode>dir</mode>
Directory of per-probe YAML coefficient files. The pipeline resolves
`{tbl}.yaml` where `tbl` is the probe table name derived from the
probe's id (e.g. `incl03`, `incl_p05`, `incl_b12`). Each YAML file
must follow the `input.coefs` structure with at least one calibration
field (`Ag`, `Cg`, `Ah`, `Ch`, `Rz`, `kVabs`, `azimuth_shift_deg`).

Missing probe file → falls back to dataclass defaults silently.
For configs in human readable (YAML) format see bundled `cfg/coef/yaml_export/`.

### `input.coefs_path` <mode>file</mode>
Single coefficient source file: HDF5 (`.h5`), NetCDF4 (`.nc`), or
exported YAML (`.yaml`).  All probes share the file — the pipeline
selects the group by table name.  Comma-separated paths are accepted
(fallback chain, first match wins).

## `input.coefs` — Calibration coefficients

Loaded from the coefficient file and copied into each per-probe YAML on first run.
Edit these to update a probe's calibration — changes are persisted automatically.

| Field | Type | Default | Physical meaning |
|-------|------|---------|------------------|
| `Ag` | 3×3 float | `[[1.73e-3,0,0],[0,1.73e-3,0],[0,0,1.73e-3]]` | Accelerometer scale matrix: `G = Ag @ (Axyz − Cg)` |
| `Cg` | 3‑float | `[10, 10, 10]` | Accelerometer bias vector |
| `Ah` | 3×3 float | Identity | Magnetometer scale matrix: `H = Ah @ (Mxyz − Ch)` |
| `Ch` | 3‑float | `[10, 10, 10]` | Magnetometer bias vector |
| `Rz` | 3×3 float | Identity | Sensor-to-instrument alignment rotation applied after calibration |
| `kVabs` | 6‑float | `[10, −10, −10, −3, 3, 70]` | Velocity polynomial: `Vabs(inclination)` |
| `P` | 2‑float | `[0, 1]` | Auxiliary sensor #1 linear correction: `y = P[0] + P[1]·x` |
| `PBattery` | 2‑float | `[0, 1]` | Battery voltage linear correction |
| `PTemp` | 2‑float | `[0, 1]` | Temperature linear correction |
| `azimuth_shift_deg` | `float` | `180` | Azimuth correction (degrees) — converts tilt direction from sensor to geographic coordinates. Default `180°` compensates magnetometer sign inversion at load time. See [Azimuth calibration](config_tuning.md#azimuth-calibration). |
| `g0xyz` | 3‑float | `None` | User-defined gravity reference vector. When set, overrides `Rz` with a computed rotation. |
| `dates` | `Dict[str, str]` | `{}` | Per‑component calibration dates |
| `date` | `str` | `None` | Overall calibration date |

Pressure probes (`p`‑type) use `P_t` (2‑D polynomial) instead of the scalar
`P`/`PBattery`/`PTemp` triples. When `P_t` is present, those scalars are silently ignored.

## `out` — Output configuration

| Field | Type | Default | Required | Purpose |
|-------|------|---------|----------|---------|
| `db_path` | `str` | `None` | No | Combined multi-probe output (`.proc.nc`) — all probes merged along a `probe` dimension. |
| `avg_db_path` | `str` | `None` | No | Per-probe binned output (`.proc_Avg.nc`) — one group per probe per bin size. |
| `not_joined_db_path` | `str` | `None` | No | Per-probe full-resolution output (`.proc_noAvg.nc`) — every sample preserved. |
| `raw_db_path` | `str` | `None` | No | Raw data archive (`.raw.nc`) — unprocessed readings + calibration coefficients for later reprocessing. |
| `table` | `str` | `''` | No | Override output table name. When non-empty, replaces the auto-derived pcid for text-file suffixes and HDF5 group names. |
| `dt_bins` | `List[int]` | `[0, 2, 600, 3600, 7200]` | **Yes** | Time averaging bins (seconds). `0` = full resolution. Multiple values produce separate outputs (e.g. `[0, 600, 3600]` = full + 10 min + 1 h). |
| `dt_bins_min_save_text` | `int` | `1` | No | Minimum bin size (seconds) for TSV export. Bin=0 skipped when >0. Set to 0 to include full-resolution data in text output. |
| `split_period` | `str` | `''` | No | Split output into time blocks (e.g. `'1D'` = daily files). Empty = single continuous output. |
| `text_path` | `str` | `'text_output'` | **Yes** | Directory for TSV output files. Created automatically if missing. |
| `text_date_format` | `str` | `'%Y-%m-%d %H:%M:%S.%f'` | No | Date format string for TSV timestamps. |
| `text_columns` | `List[str]` | `[]` | No | Filter which columns appear in TSV output. Empty = all available. Columns listed but absent from a particular output are silently skipped. |
| `b_all_to_one_col` | `bool` | `False` | No | Multi-probe layout: `False` = interleave columns (`v_i01, u_i01, …`), `True` = stack rows. |
| `b_overwrite_text` | `bool` | `True` | No | Overwrite existing TSV files. Set `False` to keep previous exports. |
| `b_split_by_time_ranges` | `bool` | `False` | No | Split output by `time_ranges` boundaries — each interval gets its own file. |
| `b_del_temp_db` | `bool` | `False` | No | Delete temporary HDF5 files after processing. |
| `overwrite_db` | `str \| None` | `None` | No | NC overwrite strategy. `None` = append-only (safe). `"splice"` = replace overlapping, keep rest. `"trim"` = delete outside `time_ranges`. `"export"` = TSV only, no NC writes. See [`overwrite_db` behavior](config_tuning.md#overwrite_db-behavior). |

## `filter` — Process-stage quality thresholds

`filter` is **process-stage**: values exceeding thresholds become NaN (rows
**kept**, not dropped).  Contrast with `input.min`/`max` — **load-stage DROP**
that removes entire rows.  Same key names may appear in both namespaces with
different semantics.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `min` | `Dict[str, float]` | `{}` | Lower bounds: values with `\|col\| < min[col]` set to NaN. |
| `max` | `Dict[str, float]` | `{'g_minus_1': 1, 'h_minus_1': 8}` | Upper bounds. `M` expands to `Mx`/`My`/`Mz`. |
| `bad_p_at_bursts_starts_period` | `str` | `''` | Pressure burst cleanup period (e.g. `'1h'`). Nulls first 2 samples per burst to remove startup artifacts. Empty disables. |

`g_minus_1 = ∥Gxyz∥ − 1` (gravity magnitude deviation),
`h_minus_1 = ∥Hxyz∥ − 1` (magnetic magnitude deviation).

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

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `return_` | `str` | `'<end>'` | Pipeline exit point. Run partial processing for debugging (e.g. `'<saved_raw>'` to verify data ingestion). See [Phase-stopping](config_tuning.md#phase-stopping). |
| `dask_scheduler` | `str` | `''` | Dask execution backend: `'synchronous'` for debugging, `'threads'` for production. |
| `sleep_s` | `float` | `0.5` | Pause (seconds) between probes. Increase if memory pressure is high during multi-probe runs. |
| `verbose` | `str` | `'INFO'` | Console log verbosity. `'DEBUG'` for troubleshooting, `'INFO'` for normal runs. |
| `use_h5` | `str` | `'auto'` | Binary I/O policy. `auto` = use if available, skip silently. `off` = disable NC/HDF5. `require` = error if unavailable. `prefer` = warn and fall back. |
