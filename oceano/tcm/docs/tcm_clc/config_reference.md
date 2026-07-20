# Config YAML Field Reference

Each run YAML (`cfg_proc/run/{source_stem}.yaml`) is a structured Hydra/OmegaConf config.
All fields are defined in `tcm/config.py` via the `Config` dataclass and registered groups
(`input`, `out`, `filter`, `program`).

Every run YAML starts with `# @package _global_` so Hydra merges it into the top-level Config.

## `input` — Data source & parameters

| Field | Type | Default | Required | Purpose |
|-------|------|---------|----------|---------|
| `path` | `str` | — | **Yes** | File path, glob, or regex pattern. Interpreted as glob or regex automatically (see [Pattern interpretation](#pattern-interpretation)). **Auto‑generated** in per-probe YAMLs; points to corrected file when available. |
| `tables` | `List[str]` | `['incl.*']` | No | HDF5 table names (regex allowed). For CSV, set to the raw table name derived from pcid. |
| `prefix` | `str` | `'I*[_0]'` | No | Filename prefix for CSV discovery. |
| `text_type` | `str` | `None` | No | Column layout variant: `i`, `p`, `b`, `d`, `w`. Auto‑derived from filename model. See [Text type → column layout](#text-type--column-layout). |
| `text_line_regex` | `str` | `None` | No | Override regex for raw text correction. |
| `coefs` | `ConfigInCoefs_InclProc` | defaults (see [§coefs](#inputcoefs--calibration-coefficients)) | **Yes** | Calibration coefficients. **Auto‑loaded** from coefficient file; user edits per‑probe. |
| `coefs_path` | `str` | `tcm/cfg/coef/calibration.h5` | No | Path to HDF5 coefficients file or YAML export dir. The sibling `cfg/coef/yaml_export/` dir is **always** appended as a silent fallback (noh5 / `dist/tcm_clc_txt` packaging). |
| `date_to_from` | `List[Any]` | `None` | No | Time shift: `[real_time, raw_time]` — two points to linearly map timestamps. |
| `dt_from_utc` | `int` | `0` | No | UTC offset in seconds. |
| `min_date` | `str` | `None` | No | **Sugar** — folded into `time_ranges` at compose. Not in structured schema. |
| `max_date` | `str` | `None` | No | **Sugar** — folded into `time_ranges` at compose. Not in structured schema. |
| `time_ranges` | `List[str]` | `None` | No | **Source of truth** for time window. Explicit intervals `[start, end, …]`. **Auto‑populated** from first/last data row on initial generation. |
| `min` | `Dict[str, float]` | `{}` | No | **Load-stage DROP**: raw-column lower bounds. Rows with `col < min[col]` are **dropped**. `M` shorthand expanded to `Mx`/`My`/`Mz` at compose. |
| `max` | `Dict[str, float]` | `{}` | No | **Load-stage DROP**: raw-column upper bounds. Same expansion as `min`. |
| `corr_time_mode` | `[bool, str, None]` | `True` | No | Time correction mode (moved from `filter`). See [Time correction modes](#time-correction-modes). |
| `corr_time_outlier_threshold_s` | `float` | `0.6` | No | Spike/backward threshold for `_correct_time()` (**seconds**; moved from `filter`). |
| `dt_interp_between` | `float` | `1.5` | No | Gap threshold for interpolation between bursts (**seconds**; moved from `filter`). |
| `coordinates` | `List[float]` | `None` | No | `[Lat, Lon]` for magnetic declination. |
| `time_ranges_zeroing` | `List[str]` | `[]` | No | Intervals for tilt zeroing (``Rz`` rotation). |
| `time_ranges_azimuth` | `List[str]` | `[]` | No | Intervals for tilt direction azimuth (``azimuth_shift_deg`` from mag+accel unit vectors). |
| `azimuth_add` | `float` | `0` | No | Manual azimuth offset (degrees). |
| `max_incl_of_fit_deg` | `float` | `None` | No | Inclination (deg) where Vabs curve flattens; used in calibration. |
| `calc_version` | `str` | `'trigonometric(incl)'` | No | Vabs calculation variant. |
| `dt_hole_warning` | `int` | `600` | No | Warn if max data gap > this (**seconds**). `None` disables. |
| `fs_rounding` | `int` | `100` | No | Frequency estimation rounding target (0 disables). |
| `tables_log` | `List[str]` | `['{}/logFiles']` | No | NC log group name template (overrides hardcoded `"logFiles"`). |

### Pattern interpretation

`input.path` is automatically classified as **glob** or **regex**:

| Condition | Mode | Example input | Effective regex |
|-----------|------|---------------|-----------------|
| Invalid regex (compilation fails) | glob | `*[0bdp]*.txt` | `.*?[0bdp].*?\.txt` |
| Valid regex, extension dot **unescaped** | glob | `file?.txt` | `file.\.txt` |
| Valid regex, extension dot **escaped** (`\.`) | regex | `i.*\.txt` | `i.*\.txt` |
| `path` is a directory | default regex `i.*\.txt` | `_raw/` | `i.*\.txt` |

The "extension dot" is the last `.` before a suffix containing no further dots.
Glob conversion: `*` → `.*?`, `?` → `.`, all dots → `\.` (all case-insensitive).

**Directory mode**: when `path` points to a directory, the default regex `i.*\.txt`
matches any inclinometer `.txt` file. Corrected `@`-prefixed files are always found
independently — `@?i.*\.txt` and `i.*\.txt` produce identical results because the
`@` prefix is stripped before pattern matching.

See `how_it_works.md` (§Discovery) for the implementation in `csv_load._pattern_to_regex()`.

## `input.coefs` — Calibration coefficients

Copied from global file (`calibration.h5` or YAML export) into each per‑probe YAML.
User edits these to update a probe's calibration.

| Field | Type | Default | Physical meaning |
|-------|------|---------|------------------|
| `Ag` | 3×3 float | `[[1.73e-3,0,0],[0,1.73e-3,0],[0,0,1.73e-3]]` | Accelerometer scale matrix: `G = Ag @ (Axyz − Cg)` |
| `Cg` | 3‑float | `[10, 10, 10]` | Accelerometer bias vector |
| `Ah` | 3×3 float | Identity | Magnetometer scale matrix: `H = Ah @ (Mxyz − Ch)` |
| `Ch` | 3‑float | `[10, 10, 10]` | Magnetometer bias vector |
| `Rz` | 3×3 float | Identity | Combined alignment/rotation matrix applied after `Ag`/`Ah` |
| `kVabs` | 6‑float | `[10, −10, −10, −3, 3, 70]` | Polynomial: `Vabs(inclination)` |
| `P` | 2‑float | `[0, 1]` | Auxiliary sensor #1 linear: `y = P[0] + P[1]·x` |
| `PBattery` | 2‑float | `[0, 1]` | Battery linear correction |
| `PTemp` | 2‑float | `[0, 1]` | Temperature linear correction |
| `azimuth_shift_deg` | `float` | `180` | Correction converting tilt direction from sensor to geographic coordinates (degrees). Default `180` compensates the magnetometer sign inversion applied at load time (``invert_magnetometer`` in ``csv_load.py`` → ``Mxyz`` negated in ``format_loaded.py``).  See [Azimuth calibration](#azimuth-calibration). |
| `g0xyz` | 3‑float | `None` | Alternative gravity vector for zeroing (overrides `Rz` if set) |
| `dates` | `Dict[str, str]` | `{}` | Per‑component calibration dates |
| `date` | `str` | `None` | Overall calibration date |

### Coefficient loading priority

1. `input.coefs` in YAML (highest — user edits live here)
2. File at `input.coefs_path` (HDF5 or YAML directory)
3. Sibling `cfg/coef/yaml_export/` directory (silent noh5 fallback)
4. Dataclass defaults (lowest)

When the HDF5 file is missing (e.g. `dist/tcm_clc_txt` packaging), the chain
falls through to `yaml_export/{tbl}.yaml` silently — no user intervention needed.

Pressure probes (`p`‑type) use `P_t` (2‑D polynomial) instead of the legacy
scalar `P`/`PBattery`/`PTemp` triples. When `P_t` is loaded or overridden,
those scalar defaults are silently ignored — no warning about "not redefined".

## `out` — Output configuration

| Field | Type | Default | Required | Purpose |
|-------|------|---------|----------|---------|
| `db_path` | `str` | `None` | No | `.proc.h5` path (HDF5 mode; null when h5py unavailable). |
| `not_joined_db_path` | `str` | `None` | No | `.proc_noAvg.h5` path. |
| `raw_db_path` | `str` | `None` | No | `.raw.h5` path. |
| `table` | `str` | `''` | No | Output table name override. When non-empty, overrides the pcid derived from `input.path` for text-file suffixes. The raw value is also used as HDF5 table name (not the derived pcid). |
| `dt_bins` | `List[int]` | `[0, 2, 600, 3600, 7200]` | **Yes** | Averaging bins (seconds → timedelta). `0` = no averaging.  **noh5 dist** default: `[0, 3600]` (no-avg + 1h only). |
| `dt_bins_min_save_text` | `int` | `1` | No | Minimum bin size to save text output. Bin=0 skipped when >0.  **noh5 dist** default: `0` (no-avg TSV enabled). |
| `split_period` | `str` | `''` | No | Pandas offset string to split output blocks (e.g. `'1D'`). |
| `text_path` | `str` | `'text_output'` | **Yes** | Text output directory. |
| `text_date_format` | `str` | `'%Y-%m-%d %H:%M:%S.%f'` | No | Date format in text files. |
| `text_columns` | `List[str]` | `[]` | No | Column filter; empty = all calculated columns. |
| `b_all_to_one_col` | `bool` | `False` | No | Concatenate columns; if true, probes are stacked row‑wise instead. |
| `b_overwrite_text` | `bool` | `True` | No | Overwrite existing text files. |
| `b_split_by_time_ranges` | `bool` | `False` | No | Split output by `time_ranges`. |
| `b_del_temp_db` | `bool` | `False` | No | Delete temporary HDF5 after processing. |

## `filter` — Process-stage NaN-out thresholds

`filter` is **process-stage**: values exceeding thresholds are set to NaN
(rows **preserved**, not dropped).  Contrast with `input.min`/`max` which
is **load-stage DROP** (rows removed).  Same key names may appear in both
namespaces — semantically distinct.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `min` | `Dict[str, float]` | `{}` | Lower bounds: values with `\|col\| < min[col]` set to NaN. |
| `max` | `Dict[str, float]` | `{'g_minus_1': 1, 'h_minus_1': 8}` | Upper bounds. `M` shorthand expanded to `Mx`/`My`/`Mz`. |
| `bad_p_at_bursts_starts_period` | `str` | `''` | Pandas offset alias (e.g. `'1h'`) for pressure burst NaN-out — nulls first 2 points per burst period. Empty disables. |

`g_minus_1 = ∥Gxyz∥ − 1` (gravity magnitude deviation), `h_minus_1 = ∥Hxyz∥ − 1` (magnetic magnitude deviation).

### Stage classification: `input` vs `filter`

| Stage | Namespace | Action | Operates on |
|-------|-----------|--------|-------------|
| **Load** (DROP) | `input.min` / `input.max` | Rows removed | Raw columns (`Ax`…`Mz`, `P_counts`) |
| **Load** (window) | `input.time_ranges` | Rows outside window removed | `time` coordinate |
| **Process** (NaN-out) | `filter.min` / `filter.max` | Values → NaN, rows kept | Computed columns (`g_minus_1`, `h_minus_1`) |
| **Process** (pressure) | `filter.bad_p_at_bursts_starts_period` | First-2 per burst → NaN | `Pressure` |

### Calibration filter extensions (`filter/calib`)

When the calibration entry point uses `filter: calib`, the filter group adds
typed despike overrides (mirrors `_dask_legacy/incl_calibr_hy.ConfigFilter`):

| Field | Type | Purpose |
|-------|------|---------|
| `blocks` | `List[int]` | Apex despike block sizes (default `[21, 7]`) |
| `offsets` | `List[float]` | Apex despike offset thresholds |
| `std_smooth_sigma` | `float` | Apex despike smoothing sigma |
| `A` | `ConfigFilterChannel` | Per-axis overrides for accelerometer |
| `M` | `ConfigFilterChannel` | Per-axis overrides for magnetometer |
| `no_works_noise` | `Dict[str, float]` | Noise threshold per channel (`is_works()` — reserved, wire deferred) |

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
| `return_` | `str` | `'<end>'` | Controls how far the pipeline runs before stopping. See [Phase-stopping](#phase-stopping). |
| `dask_scheduler` | `str` | `''` | `'synchronous'`, `'threads'`, `'processes'`, `'distributed'`. |
| `sleep_s` | `float` | `0.5` | Sleep between probes to manage memory. |
| `verbose` | `str` | `'INFO'` | Log level. |
| `force_reprocess` | `bool` | `False` | Override time-range containment for **processed** NC writes (noAvg/binned). Does NOT affect coef persistence (always overwrites in-place). Accepted via `+force_reprocess=True`. |
| `use_h5` | `bool` or `None` | `None` | Control HDF5/NC I/O. `None` auto-detects from `h5py`/`netCDF4` availability. `True` forces enable (errors if unavailable). `False` forces disable (warnings on skipped operations). Used in noh5 builds to suppress HDF5 writes gracefully. |

## Decision tables and behavior tuning

### Phase-stopping

`program.return_` controls how far the pipeline runs before stopping:

| Value | Stops after | Output produced |
|-------|-------------|-----------------|
| `'<end>'` (default) | Full processing | All NC + TSV |
| `'<saved_raw>'` | Coef persistence + raw NC save | `*.raw.nc` with coefs + log |
| `'<saved_noavg>'` | noAvg NC write | `*.proc_noAvg.nc` with per-probe groups |
| `'<saved_all>'` | All NC writes | `*.proc_noAvg.nc` + `*.proc.nc` (no combined output) |
| `'<cfg_from_args>'` | Config composition | Config dict returned |
| `'<gen_names_and_log>'` | Config generation | YAML files written |

**Typical use**: debug partial output without waiting for full processing.
For example, `program.return_='<saved_raw>'` to verify raw data ingestion.

> **noh5 note**: `<saved_raw>` persists coefs to NC when h5py is available, or to
> the run YAML in noh5 mode. Raw data cannot be saved to NC without pytables, but
> coef changes ARE written regardless. Coefficients always overwrite in-place;
> `force_reprocess` does NOT affect coef persistence — it only controls whether
> processed outputs (noAvg/binned) are re-generated when the time range is already
> covered. See [Updating Coefficients via Zeroing](../README.md#updating-coefficients-via-zeroing)
> for the coef persistence matrix.

### Time correction modes

`filter.corr_time_mode` controls how the pipeline handles integer-second timestamps:

| Value | Behavior |
|-------|----------|
| `True` (default) | **Snap-to-grid**: detects sampling frequency from data, assigns regular sub-second timestamps (e.g. 100 ms at 10 Hz). Backward jumps, spikes, and overlong runs removed first (outlier + trim steps), then the clean subset is snapped. |
| `None` / `False` | **Mask-only**: removes backward/spike samples via `b_ok` mask but does NOT snap. For integer-second N Hz data, N-1 samples per second are removed → collapses to 1 Hz. |
| `"delete_inversions"` | Runs full outlier pipeline (trim + spike + backward removal) but timestamps unchanged. Non-monotone positions masked. |

**Config fields affecting time correction** (from `cfg.filter`):

| Field | Default | Effect on `_correct_time` |
|-------|---------|---------------------------|
| `corr_time_mode` | `True` | Snap-to-grid vs mask-only vs delete_inversions |
| `dt_interp_between` | `1.5s` | Minimum gap to detect a real hole (vs jitter within a segment) |
| `corr_time_outlier_threshold_s` | `0.6s` | Spike/backward detection threshold |

**Edge-row detection**: When data has ≤ 2 time values (config generation mode),
`time_corr()` skips `_correct_time()` entirely regardless of mode, avoiding
misleading "freq unknown → defaulting to 1Hz" warnings.

**Diagnostics**: `save_time_corr_diagnostics()` and `plot_time_corr_diagnostics()`
produce NPZ arrays with action bitmask per sample:

| Bit | Constant | Meaning |
|-----|----------|---------|
| `0x01` | `ACT_TRIM` | Overlong run, sample dropped |
| `0x02` | `ACT_SPIKE` | Bilateral outlier, dropped |
| `0x04` | `ACT_BACKWARD` | HWM backward section, dropped |
| `0x08` | `ACT_HOLE` | Data gap > dt_hole starts here |
| `0x10` | `ACT_ALARM` | Segment snap RMS exceeds threshold |
| `0x20` | `ACT_NOT_MONO` | Non-monotone after snap, masked |
| `0x40` | `ACT_OUT_OF_RANGE` | Excluded by time_ranges |

### Filter expansion

`cfg.filter.max` and `cfg.filter.min` support `M` as a shorthand for `Mx`,
`My`, `Mz`. If `M` is set but `Mx`/`My`/`Mz` are not, the value is copied
to all three axes.

```yaml
# Equivalent configurations:
filter:
  max: {M: 5}           # → Mx=5, My=5, Mz=5

# Explicit (overrides M expansion):
filter:
  max: {Mx: 5, My: 4, Mz: 6}
```

### Column order

Output columns are ordered to match legacy convention:

```text
v, u, inclination, Vabs, Vdir          ← first (velocity/direction group)
Pressure, Temp, Battery, ...           ← remaining sensor variables
```

For combined multi-probe TSV, each probe's columns are interleaved per-probe:
`v_i01, u_i01, inclination_i01, v_i02, u_i02, ...` (axis=1 concatenation).
When `b_all_to_one_col=True`, probes are stacked row-wise instead.

### Text type → column layout

`text_type` determines which columns are read from raw CSV files:

| `text_type` | Columns read |
|-------------|-------------|
| `i`, `b`, `""` | `[Ax, Ay, Az, Mx, My, Mz, Battery, Temp]` |
| `p`, `d` | `[Ax, Ay, Az, Mx, My, Mz, P_counts, Battery, Temp]` |
| `w` | `[Battery, Temp]` (no inertial sensors) |

The column layout is auto-detected from the file header via
`csv_load.format_parts_select_raw(file_path)` and falls back to
`format_parts_select(text_type)` when auto-detection fails.
`text_type` is derived from the filename model (first character of pcid),
and can be overridden via `input.text_type` in YAML or CLI.

### Incremental append positions

When appending data to an existing NC group, `_time_range_overlap()` classifies
the position of new data relative to existing data:

| Position | Condition | Write strategy |
|----------|-----------|----------------|
| `AFTER` | `new_min > ex_max` | h5py `resize()` — O(1), no re-read |
| `BEFORE` | `new_max < ex_min` | `_prepend_nc_group()` — h5py resize + chunkwise shift, O(chunk) memory |
| `CONTAINED` | `new_min >= ex_min && new_max <= ex_max` | skip (no write) |
| `OVERLAP_TAIL` | `new_max > ex_max && new_min <= ex_max` | trim new via `ex_ns[-1]` + `_append_to_nc_group()`, O(new) memory |
| `OVERLAP_HEAD` | `new_min < ex_min && new_max <= ex_max` | trim new + `_prepend_nc_group()`, O(chunk) memory |

**Key behavior**: existing data is never modified. On overlap, the new data's
overlapping portion is trimmed and a warning is logged.

### Log-based dedup

`check_file_vs_log(cur, existing_log)` returns a 3-way decision controlling
how a source file is appended:

| Decision | Condition | Action |
|----------|-----------|--------|
| `SKIP` | same `fileName`, `cur.fileChangeTime ≤ existing` | Skip entirely — file not modified |
| `RESUME` | same `fileName`, `cur.fileChangeTime > existing` | Append only tail after existing last time — file was updated |
| `NEW_FILE` | no matching `fileName` in log | Full position compare + append (see [Incremental append positions](#incremental-append-positions)) |

**RESUME details**: when the same source file was updated (newer mtime),
only data after the existing last timestamp is appended. The log is updated
with two rows: original start and new tail end (both with the new
`fileChangeTime`).

### Re-run behavior

On re-processing the same input data, each NC output type handles idempotency
differently:

| Output | Dedup mechanism | Re-run effect |
|--------|----------------|---------------|
| `*.raw.nc` | Log table (`check_file_vs_log`) | **SKIP** — same fileName + mtime → no write |
| `*.proc.nc` (binned) | `store_processed_incremental` (time-range containment) | **SKIP** — new range ⊂ existing → no write |
| `*.proc_noAvg.nc` | `store_processed_incremental` (time-range containment) | **SKIP** — new range ⊂ existing → no write |
| Combined groups | `_combine_probes` → `store_processed(mode="a")` | **Overwrite** — always rewrites from per-probe groups |

Time-range containment uses ``ex_ns.min()``/``ex_ns.max()`` (not ``[0]``/``[-1]``)
because time may be unsorted when multiple stems are appended in discovery order.

## Per-file run YAMLs (`@package _global_`)

Each source file gets its own YAML at `cfg_proc/run/{source_stem}.yaml`, starting with
`# @package _global_` so Hydra merges it into the top-level Config.

```yaml
# @package _global_
input:
  path: "/abs/path/to/@i_01.txt"
  tables: ["incl01"]
  coefs:
    Ag: [[1,0,0],[0,1,0],[0,0,1]]
out:
  dt_bins: [0, 2, 600]
  text_path: "text_output"
filter: {}
```

Override any top-level field (`input`, `out`, `filter`, `program`). The `# @package _global_`
directive tells Hydra to merge this YAML's contents at the Config root rather than under a
`run` namespace.

## Minimal viable config

Auto‑generated, user edits `coefs` and `time_ranges`:

```yaml
# @package _global_
input:
  path: "/abs/path/to/@i_01.txt"
  tables: ["incl01"]
  coefs:
    Ag: [[1,0,0],[0,1,0],[0,0,1]]
    Cg: [0,0,0]
    Ah: [[1,0,0],[0,1,0],[0,0,1]]
    Ch: [0,0,0]
    kVabs: [1,0,0,0,0,0]
    Rz: [[1,0,0],[0,1,0],[0,0,1]]
out:
  dt_bins: [0, 2, 600]
  text_path: "text_output"
filter: {}
```

**Required user edits** after initial generation:
- `input.coefs` — replace defaults with actual calibration values
- `input.time_ranges` — optionally restrict processing window

### Azimuth calibration

`azimuth_shift_deg` is the correction converting the **tilt direction** (azimuth
of the inclinometer's lean) from sensor coordinates to geographic coordinates.
The `Vdir` formula computes the tilt azimuth via `G × H` (gravity × magnetic
field) cross product, then adds `azimuth_shift_deg` to get degrees from North.

Default is `180°` to compensate the magnetometer sign inversion applied at load
time (``invert_magnetometer`` in ``csv_load.py`` → ``Mxyz`` negated).

| `azimuth_shift_deg` | Tilt direction reported as |
|---|---|
| `0` | North |
| `90` | East |
| `180` (default) | South |
| `270` | West |

**Manual**: set the value directly in `input.coefs.azimuth_shift_deg`.

**From data** (`time_ranges_azimuth`): record data while the instrument is tilted
in a known direction, then set `input.time_ranges_azimuth` to that interval. The
pipeline computes the shift via `orientation.azimuth_shift()` and writes it to
the coefficients (YAML in noh5, NC in full env).

**Corrections** applied on top: `azimuth_add` (manual offset) and magnetic
declination from `coordinates` + `data_date`.