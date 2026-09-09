# Config Tuning — Decision Tables & Behavior

Decision tables, behavior tuning, and YAML examples for the processing pipeline.
Field definitions are in [Configuration Schema and Device Metadata Reference](config_reference.md); implementation
internals are in [CLI Internals](../project_developer_guide/CLI.md).

## Phase-stopping

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

> **When h5py is unavailable**: `<saved_raw>` persists coefs to NC when h5py is
> available, or to the run YAML otherwise. Raw data cannot be saved to NC without
> pytables, but coef changes ARE written regardless. Coefficients always overwrite
> in-place; `out.overwrite_db` does NOT affect coef persistence — it only controls
> whether processed outputs (noAvg/binned) are re-generated when the time range is
> already covered. See [Updating Coefficients via
> Zeroing](../user_guide/configuration.md#updating-coefficients-via-zeroing) for
> the coef persistence matrix.

## Time correction modes

`input.corr_time_mode` controls how the pipeline handles integer-second timestamps:

| Value | Behavior |
|-------|----------|
| `True` (default) | **Snap-to-grid**: detects sampling frequency from data, assigns regular sub-second timestamps (e.g. 100 ms at 10 Hz). Backward jumps, spikes, and overlong runs removed first (outlier + trim steps), then the clean subset is snapped. |
| `None` / `False` | **Mask-only**: removes backward/spike samples via `b_ok` mask but does NOT snap. For integer-second N Hz data, N-1 samples per second are removed → collapses to 1 Hz. |
| `"delete_inversions"` | Runs full outlier pipeline (trim + spike + backward removal) but timestamps unchanged. Non-monotone positions masked. |

**Config fields affecting time correction** (from `input`):

| Field | Default | Effect on `_correct_time` |
|-------|---------|---------------------------|
| `corr_time_mode` | `True` | Snap-to-grid vs mask-only vs delete_inversions |
| `dt_interp_between` | `1.5s` | Minimum gap to detect a real hole (vs jitter within a segment) |
| `corr_time_outlier_threshold_s` | `0.6s` | Spike/backward detection threshold |

See [§Time correction](../project_developer_guide/CLI.md#time-correction) for the correction pipeline internals,
diagnostics bitmask, and edge-row detection behavior.

## Inverted time_ranges

A `time_ranges` pair with `start > end` matches nothing (`t >= s & t < e` is
empty), so keeping it would silently filter 100% of the data out. Behavior:

| Condition | Behavior |
|-----------|----------|
| Scan finds inverted extraction pair (non-monotonic file) | **Warning** (`Inverted time edges [...] — repaired to parsed-rows span [...] (min at line N, max at line M)`) + YAML stores min/max over timestamp-parseable rows from the extractor's full scan (loose + archive members), not what Run will load after correction |
| Scan finds inverted pair with no readable file (metadata sync, unreadable file) | **Warning/error**, pair kept verbatim — `main_init` strips it at Run so the probe degrades to a full load |
| Inverted pair still reaches load (e.g. hand-edited YAML) | **Warning** (`Inverted time_ranges ... ignored ... — full-file load`) + pair dropped; actual file min/max logged after load |
| Full load still yields no data (`None` or 0 rows) | **Error** (`No data loaded for {pcid} — processing aborted`) → probe marked **failed**, never `ok` |

Open bounds (`None`/`NaT`) are never inverted and pass through unchanged.

> In the GUI, ``input.time_ranges`` / ``metadata.time_range`` date cells that
> break ascending order are red-flagged (`check: "sorted"`, same error color
> as a non-existent `input.path`) — fix the order before Run.

## Config filtering

Two parameters control which run YAMLs are processed, both using the same
glob/regex auto-detection as `input.path` discovery.  The key distinction:
**CLI values** may be patterns; **YAML stored values** are always resolved
absolute paths to concrete data files.

| Parameter | Source | Filters against | When set | Config generation |
|-----------|--------|-----------------|----------|-------------------|
| `input.path` (not directory) | CLI pattern (glob/regex/concrete) | YAML's resolved `input.path` **filename** | After generation, only YAMLs whose resolved path filename matches are kept | **Runs normally** — generates configs for source files matching the pattern |
| `input.yaml_path` | CLI pattern (glob/regex) | YAML **filename stem** or **full name** (matches both `stem` and `stem.yaml`) | Only existing YAMLs whose stem matches are processed | **Skipped entirely** — no new configs are created |

When both are set, both filters apply (AND logic): a config must match both
`input.path` and `yaml_path` to be included.

**Dry-run**: combine with `program.return_=<cfg_from_args>` to list matching
configs without processing any data:

```bash
# List all configs (no generation, no processing)
python scripts/tcm_proc.py "_raw" input.yaml_path="*" program.return_=<cfg_from_args>

# List configs matching a data file pattern (generation runs, then filter)
python scripts/tcm_proc.py "_raw/@i_p5*.TXT" program.return_=<cfg_from_args>

# List configs matching a YAML stem pattern (no generation)
python scripts/tcm_proc.py "_raw" input.yaml_path="*@i_p5*" program.return_=<cfg_from_args>
```

## Pattern interpretation

`input.path` is automatically classified as **glob** or **regex**:

| Condition | Mode | Example input | Effective regex |
|-----------|------|---------------|-----------------|
| Invalid regex (compilation fails) | glob | `*[0bdp]*.txt` | `.*?[0bdp].*?\.txt` |
| Valid regex, extension dot **unescaped** | glob | `file?.txt` | `file.\.txt` |
| Valid regex with `|` or `(...)` wrapper | regex | `(a\|b).txt` | `(a\|b).txt` |
| Valid regex, extension dot **escaped** (`\.`) | regex | `i.*\.txt` | `i.*\.txt` |
| `path` is a directory | default regex `i.*\.txt` | `_raw/` | `i.*\.txt` |

The "extension dot" is the last `.` before a suffix containing no further dots.
Glob conversion: `*` → `.*?`, `?` → `.`, all dots → `\.` (all case-insensitive).

**Directory mode**: when `path` points to a directory, the default regex `i.*\.txt`
matches any inclinometer `.txt` file. Corrected `@`-prefixed files are always found
independently — `@?i.*\.txt` and `i.*\.txt` produce identical results because the
`@` prefix is stripped before pattern matching.

See [§Discovery](../project_developer_guide/CLI.md#discovery) for the implementation in `csv_load._pattern_to_regex()`.

## Column order

Output columns follow this ordering:

```text
v, u, inclination                          ← persisted in NC (velocity/direction group)
Pressure, Temp, Battery, ...               ← remaining sensor variables
```

**Vabs/Vdir save policy**:

| Output | Vabs/Vdir | inclination |
|--------|-----------|-------------|
| NC files (`*.proc_noAvg.nc`, `*.proc_Avg.nc`, `*.proc.nc`) | **not saved** | saved |
| Per-probe TSV | computed on-the-fly from `v`/`u` | saved |
| Combined TSV (`@joined.tsv`) | **not saved** | **not saved** |

`Vabs = hypot(v, u)`, `Vdir = degrees(arctan2(u, v))` — exact inverse of
`polar2dekart`.  The on-the-fly computation is in `physical.add_vabs_vdir()`.

### Available `text_columns` values

`text_columns` filters which columns appear in TSV output.  Empty (default)
writes **all available** columns for the given output type.  Columns listed
but absent from a particular output are silently skipped.

| Column | Per-probe TSV | Combined TSV | Notes |
|--------|:---:|:---:|-------|
| `v` | ✓ | ✓ | North velocity component |
| `u` | ✓ | ✓ | East velocity component |
| `Vabs` | ✓ | — | On-the-fly from `v`/`u`; requires `kVabs ≠ None` |
| `Vdir` | ✓ | — | On-the-fly from `v`/`u`; requires `kVabs ≠ None` |
| `inclination` | ✓ | — | Sensor tilt angle (degrees) |
| `Pressure` | ✓ | ✓ | When `P_t` coefficients provided |
| `Temp` | ✓ | ✓ | Temperature (if present in raw data) |

Example: `text_columns: [v, u, Vabs, Vdir]` — produces four columns in
per-probe TSV; in combined TSV only `v` and `u` appear (Vabs/Vdir skipped).

For combined multi-probe TSV, each probe's columns are interleaved per-probe:
`v_i01, u_i01, v_i02, u_i02, ...` (axis=1 concatenation; no inclination).
When `b_all_to_one_col=True`, probes are stacked row-wise instead.

## Text type → column layout

`text_type` determines which columns are read from raw CSV files:

| `text_type` | Columns read |
|-------------|-------------|
| `i`, `b`, `""` | `[Ax, Ay, Az, Mx, My, Mz, Battery, Temp]` |
| `p`, `d` | `[Ax, Ay, Az, Mx, My, Mz, P_counts, Battery, Temp]` |
| `w` | `[Battery, Temp]` (no inertial sensors) |

Auto-detected from the file header via `csv_load.format_parts_select_raw(file_path)`;
falls back to `format_parts_select(text_type)` when auto-detection fails.
`text_type` is derived from the filename model (first character of pcid),
and can be overridden via `input.text_type` in YAML or CLI.

## Re-run behavior

On re-processing the same input data, each NC output type handles idempotency
differently:

| Output | Dedup mechanism | Re-run effect |
|--------|----------------|---------------|
| `*.raw.nc` | Log table (`check_file_vs_log`) | **SKIP** — same fileName + mtime → no write |
| `*.proc.nc` (binned) | `store_processed_incremental` (time-range containment) | **SKIP** — new range ⊂ existing → no write |
| `*.proc_noAvg.nc` | `store_processed_incremental` (time-range containment) | **SKIP** — new range ⊂ existing → no write |
| Combined groups | `_combine_probes` → `store_processed(mode="a")` | **Overwrite** — always rewrites from per-probe groups |

Time-range containment uses `ex_ns.min()`/`ex_ns.max()` (not `[0]`/`[-1]`)
because time may be unsorted when multiple stems are appended in discovery order.

### Incremental append positions

When appending data to an existing NC group, the position of new data relative
to existing data determines the write strategy:

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

### `/param_spans/{tbl}` interval table

Every processed NC file stores processing parameters as an HDF5 sibling group
`/param_spans/{tbl}`.  Each processing run appends a new interval; duplicate
params are not recorded.

The interval table has:

- coord `start` (`datetime64[ns]`) — interval boundaries
- var `params` (str) — sorted key=value text per interval
- var `meta` (str JSON) — metadata per interval

Interval *i* covers `[start[i], start[i+1])` or `[start[i], ∞)` if last.
Written **after** the data write so the NC file already exists.

On re-run with data already covered, only the **latest** interval entry is
compared to the current value (ignoring `input.time_ranges` lines) — a
mismatch raises `ValueError` with a unified diff.

The params text is built by `_build_filter_params_text()` (`processing.py`)
and contains **all** resolved parameters that affect processed output, sorted
by key:

| Prefix | Source | Config section | Fields |
|--------|--------|----------------|--------|
| `filter.` | Process-stage NaN-out thresholds | `filter` (`ConfigFilter_InclProc`) | `min.<var>`, `max.<var>`, `bad_p_at_bursts_starts_period` |
| `input.` | Load-stage window / thresholds | `input` (`ConfigIn_InclProc`) | `time_ranges`, `min.<var>`, `max.<var>`, `dt_min_binning_proc` |
| `coef.` | Prepared coefficients | `coefs` dict (post `prepare_coefs()`) | All keys except `dates`, `Rz` |

The comparison on skip strips `input.time_ranges` lines before comparing —
so changing only the time window (e.g. narrowing `time_ranges`) does NOT
trigger a `ValueError`; only filter or coefficient changes do.  The full
(including `input.time_ranges`) text is still stored for diagnostic purposes.

**Example** `/param_spans/{tbl}` with two intervals:

| `start` | `params` |
|---|---|
| `2024-01-15T10:00:00` | `coef.Ag=[[1. 0. 0.] ...]` · `filter.max.Ax=5` · `input.time_ranges=[…]` |
| `2024-02-01T09:30:00` | `coef.Ag=[[1. 0. 0.] ...]` · `filter.max.Ax=5` · `input.time_ranges=[…]` |

## `overwrite_db` behavior

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

### Absent text files

When the text file referenced by `input.path` no longer exists on disk, the
pipeline can still load from `*.raw.nc` via the **raw NC fast-path** — provided:

1. `*.raw.nc` exists and its time range covers `time_ranges`, AND
2. the NC's `/{tbl}/logFiles` group contains a `fileName` entry matching
   the source file (`{parent_dir_name}/{stem}` format, first 255 chars).

When both conditions hold, configs are **not** marked stale and the pipeline
loads `ds_raw` + coefs from `*.raw.nc` as if the text file were present.
Phase 4 (raw NC save) is skipped since the data already resides in the NC.

If neither the text file nor a matching raw NC log entry exists, the config is
marked stale and `FileNotFoundError` is raised during processing (caught by
`process_loading_yaml`).

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

Minimal viable config, azimuth calibration, and filter expansion are
documented in [the configuration guide](../user_guide/configuration.md).
