# tcm_clc — Inclinometer Data Processing Pipeline

CLI tool for processing inclinometer survey data from raw CSV/HDF5 files to
velocity, inclination, direction, and pressure outputs.

## Quick Start

```bash
# Process all probes in a data directory (_raw is just an recommended name — use any path)
python -m scripts.tcm_clc "D:/data/experiment/_raw"

# Process specific probes only from any txt files that contains `i`
python -m scripts.tcm_clc "_raw/*i*.txt" input.ids=[i01,i_p02]

# Override any other field described in the config_reference.md, for example:
python -m scripts.tcm_clc "_raw/*i*.txt" filter.corr_time_mode=false out.text_path=./results

# List overrides — wrap in single quotes to protect brackets from the shell:
python -m scripts.tcm_clc "_raw" \
  'input.time_ranges=["2024-01-01T00:00:00","2024-01-02T00:00:00"]'

```

> **PowerShell note**: wrap each override containing ``[""]`` or ``<>`` in single
> quotes to prevent shell interpretation: ``'program.return_="<cfg_from_args>"'``.

The first positional argument is a **path to scan**: directory, glob, or regex.
A plain directory scans for `i*.txt` (case-insensitive default); wildcards like
`*i*.txt` use glob rules; escaped dots like `i.\\.txt` trigger regex interpretation
(see ``_pattern_to_regex`` in `docs/tcm_clc/how_it_works.md`).
The name `_raw` is just a convention — your data can live anywhere.

### Common workflows

**First run on new data**:
On first run the program creates a `cfg_proc/` directory inside `_raw/` and
auto-generates per-probe config YAMLs in `cfg_proc/run/` (calibration
coefficients, time ranges). Edit these to set the correct calibration values,
then re-run:

```bash
# Phase 1: generate configs (auto-detected from data)
python -m scripts.tcm_clc "_raw/i*.txt"

# Edit cfg_proc/run/@i_01.yaml — replace default coefs with actual values
# Then re-run to process:
python -m scripts.tcm_clc "_raw/i*.txt"
```

**Re-process with tweaked coefficients**: edit the probe's YAML, re-run.

**Batch re-run after adding data files**: the tool appends incrementally
to existing `*.raw.nc` files. Re-running is safe — overlapping data is
detected and skipped.

## Pipeline Overview

The pipeline processes raw sensor data through three high-level stages:

1. **Discovery & Config Generation** — scans raw data directory for probe
   files, groups them by identity, generates per-probe YAML configs with
   auto-detected time ranges and calibration placeholder values. Configs
   are preserved between runs — only generate additional configs for new raw text files.

2. **Loading & Preparation** — loads raw data, resolves calibration coefficients
   from various sources according to priority (saved per-probe YAML wins highest)
   across YAML, HDF5 coefficient file or the bundled `yaml_export/` directory,
   applies azimuth correction and auto-zeroing rotation from data within
   user-specified time windows.

3. **Processing & Export** — computes velocity projections (`u`, `v`),
   absolute speed (`Vabs`), direction (`Vdir`), sensor inclination, and optional
   pressure. Averages over configurable bin intervals (default in the full
   distribution: none, 2 s, 600 s, 3600 s, 7200 s). Saves to NetCDF4 with
   per-probe groups and exports TSV text files.

## What You Get

Source, configuration and logs (all relative to the **data directory**, which serves as
the working root. ``_raw`` is a **conventional** subdirectory name — any name works,
but the code looks for ``_raw`` first when resolving relative output paths.)

```text
data_dir/_raw/        ← raw files in instrument dir (conventional name)
    i_01.txt
    @i_01.txt           ← corrected (auto-generated; identical to i_01.txt if nothing to correct)
    cfg_proc/           ← Hydra config directory (created inside _raw/)
        config.yaml     ← optional primary config (overwrites bundled defaults)
        run/            ← Per-probe config (coefficients, time ranges)
            @i_01.yaml  — edit this
        log/            ← program and hydra logs output directory
```
After processing, the pipeline produces these output data files

| File | Contents |
|------|----------|
| `*.raw.nc` | Raw data + calibration coefficients (incremental append) |
| `*.proc_noAvg.nc` | Non-averaged processed output (per-probe groups) |
| `*.proc.nc` | Binned processed output (per-probe groups for each bin) |
| `text_output/{timestamp}@{pcid}.tsv` | Tab-separated text exports (binned only) |

When multiple probes are processed in one run, combined groups with a `probe`
dimension are written to the same NC files and combined TSV files.

## Probe Identity (pcid)

Each probe is identified by a canonical string (`pcid`) that links its raw data,
coefficients, config file, and output:

- `i01` — inclinometer probe 01
- `i_p02` — inclinometer model-p probe 02
- `w01` — wave probe 01

The pcid maps bidirectionally to config YAML stems (`@i_01`) and raw data
filenames; when saving to NetCDF4/HDF5 it maps to the table name (`incl01`),
and when several probes are stored in one table or file it becomes the column
name.

## Configuration

Four configuration groups control the pipeline behavior. Override any field
on the CLI or in per-probe YAML files:

| Group | What it controls | Key fields you'll likely touch |
|-------|------------------|-------------------------------|
| `input` | Source files, calibration, time ranges | `path`, `ids`, `coefs`, `time_ranges`, `coordinates` |
| `out` | Output format, binning, text export | `dt_bins`, `text_path`, `split_period` |
| `filter` | Data quality thresholds | `min`, `max`, `corr_time_mode` |
| `program` | Runtime behavior | `return_`, `verbose` |

For the full field reference, see `docs/tcm_clc/config_reference.md`.

### Per-probe YAMLs

Each probe's config lives in `cfg_proc/run/{prefix}@{pcid}-{comment}.yaml`.

- The `{prefix}` (e.g. `260624_1730`) is auto-generated from the data's first
  timestamp as `%y%m%d_%H%M`.  If no timestamp is available at generation time
  (rare edge case), the file is simply `@{pcid}.yaml` without a prefix.
- Everything before the **last** `@` is metadata — ignored for probe identity.
  The `-{comment}` suffix after the pcid stem is also stripped
  (see `tcm.format.stem_to_pcid`).
- So `@i_01.yaml`, `260624_1730@i_01.yaml`, `@i_01-extra.yaml` all resolve to
  the same probe `i01`.

Configs are auto-generated from file discovery and **never overwritten** on
subsequent runs — only missing configs are created, and stale ones (whose
`input.path` references a deleted file) are regenerated. This means:

- **`input.path` is the real link** between a YAML and its source data file.
  The YAML filename (`{datestamp}@{stem}.yaml`) is metadata only — the
  pipeline reads `input.path` to locate the data, not the YAML name.
- **Renaming a YAML** creates a **duplicate** for the same probe → the pipeline
  processes each YAML independently.  No data is lost: output NC uses incremental
  append (overlapping time ranges are skipped), so both runs write to the same
  file without row duplication.  However, coefs are written twice (last YAML wins),
  and if `+force_reprocess=True` is set, the second run overwrites the first run's
  processed data.  Remove the obsolete copy to avoid redundant work.
- **Stale configs** (whose `input.path` references a deleted file) are warned
  about but **never auto-deleted**.  When a source file is renamed or moved,
  the old YAML becomes stale and a new one is generated — the old one remains
  and must be removed manually.
- **Deduplication**: when regenerating configs, the pipeline checks if any
  existing YAML for the same normalized pcid already has a valid `input.path`.
  If so, no new YAML is created.  This prevents duplicates when the same probe
  has differently-formatted filenames (e.g. `i_090` vs `i90`).
- **Generating only configs** (no processing):
  `python -m scripts.tcm_clc "_raw/i*.txt" 'program.return_="<cfg_from_args>"'`.

Hand-editable:

```yaml
# @package _global_
input:
  path: "/abs/path/to/@i_01.txt"
  coefs:
    Ag: [[0.00173,0,0],[0,0.00173,0],[0,0,0.00173]]
    Cg: [10,10,10]
  time_ranges: ["2023-01-15T10:00:00", "2023-01-15T14:30:00"]
out:
  dt_bins: [0, 2, 600]
  text_path: "text_output"
```

Typical user edits: replace default `coefs` with actual calibration values,
narrow `time_ranges` to the deployment period, set `coordinates` for magnetic
declination correction.

## Calibration Coefficients

### Coefficient resolution priority (highest to lowest)

1. `input.coefs` in the per-probe YAML — edit here for probe-specific calibration
2. `coefs_path` — shared HDF5 or YAML coefficient file
3. `tcm/cfg/coef/yaml_export/` directory (bundled distribution fallback)
4. Defaults from the configuration dataclass

When the HDF5 file is missing (e.g. `dist/tcm_clc_txt` packaging without `calibration.h5`),
coefficients are loaded automatically from the `yaml_export/` directory — no extra
configuration needed.

### Where coefficient files live (`yaml_export/`)

Each probe's coefficients live in a YAML file named after its **table name**
(derived from `pcid` via `pcid_to_raw_name()`):

| pcid | Table name | Coef file |
|------|-----------|-----------|
| `i01` | `incl01` | `yaml_export/incl01.yaml` |
| `i_p05` | `incl_p05` | `yaml_export/incl_p05.yaml` |
| `i_b03` | `incl_b03` | `yaml_export/incl_b03.yaml` |

To edit: open the YAML, modify values under `input.coefs`, save. The YAML
structure matches the per-probe config format — you can copy coef blocks
directly into `cfg_proc/run/@i_XX.yaml` to override specific probes.

Pressure probes (`p`‑type) use `P_t` (2‑D pressure‑temperature polynomial) instead of
the legacy scalar `P`/`PBattery`/`PTemp` triples. When `P_t` is present in the loaded
coefficients or user overrides, those scalar defaults are silently ignored.

After loading, the pipeline applies:
- **Azimuth correction** from magnetic declination (requires `coordinates` + calibration date)
- **Auto-zeroing rotation** from data within `time_ranges_zeroing` windows
- **Azimuth zeroing** from data within `time_ranges_azimuth` windows (optional)
- **Gravity zeroing matrix** to level the sensor

Ellipsoid fitting — deriving new calibration coefficients from raw measurements
— is handled by the `calibration/` package (`run.py` → `run_calibration`).
See `docs/calibration/calibration_wiki.md` for the method and `docs/tcm_clc/how_it_works.md`
for integration details. Requires the **full** environment (scipy, matplotlib, h5py).

### Updating Coefficients via Zeroing

Two independent zeroing operations, each with its own time window:

| Parameter | What | How | Writes |
|---|---|---|---|
| `time_ranges_zeroing` | Tilt rotation | `orientation.zeroing_rotation()` on accelerometer data | `Rz` |
| `time_ranges_azimuth` | Tilt direction azimuth | `orientation.azimuth_shift()` on mag+accel unit vectors | `azimuth_shift_deg` |

The azimuth computation uses calibrated unit vectors only (no velocity/magnitude
calculation), so it does not depend on ``kVabs`` or inclination-to-magnitude
coefficients.

If `azimuth_add` or `coordinates` are also set, their corrections are applied
**on top of** the data-computed azimuth (e.g. manual offset or magnetic
declination).

**Where updated coefs are persisted** depends on the input source and environment:

| Source | h5py available | Updated coefs → |
|--------|:---:|-----------------|
| CSV/TXT | Yes | `*.raw.nc` file (`/{tbl}/coef/` group) |
| NC/HDF5 | Yes | Source NC file (overwritten in-place) |
| Any | No (noh5) | Run YAML (`cfg_proc/run/@i_01.yaml` under `input.coefs`) |

When coefs are written to NC, the log shows how many datasets were overwritten:
```
Coefs saved to ...//incl01: 12 datasets (2 overwritten)
```

**Typical zeroing workflow**:

```bash
# From text CSV — generates config, computes Rz, proceeds with processing
python -m scripts.tcm_clc "_raw/*i*.txt" \
  'input.time_ranges_zeroing=["2026-06-25T17:23:30","2026-06-25T17:25:00"]' \
  'input.time_ranges=["2026-06-25T17:23:30","2026-06-25T17:25:00"]'

# From binary NC — specify tables
python -m scripts.tcm_clc "260624.raw.nc" \
  'input.tables=["incl_p05"]' \
  'input.time_ranges_zeroing=["2026-06-25T17:23:30","2026-06-25T17:25:00"]' \
  'input.time_ranges=["2026-06-25T17:23:30","2026-06-25T17:25:00"]'
```

This computes `Rz` from the zeroing interval, overwrites it in the source NC,
and then proceeds with full processing. To stop after the coef write (without
processing data), add `program={"return_": "<saved_raw>"}` — see below.

### Zeroing with `g0xyz`

Instead of computing `Rz` from a data interval, you can supply `g0xyz` — the raw
accelerometer vector `[Ax, Ay, Az]` measured when the instrument was at zero tilt
(e.g. hanging plumb). The pipeline computes the rotation matrix that maps this
reference vector to vertical:

```yaml
input:
  coefs:
    g0xyz: [100.5, 50.2, 980.1]   # raw accelerometer at zero tilt
```

When `g0xyz` is set, it **overrides** any existing `Rz` in the coefficients — the
computed rotation replaces it. The rotation is applied before `fG()` calibration,
so all downstream velocity/inclination/direction values use the corrected frame.

`g0xyz` and `time_ranges_zeroing` serve different purposes:

| Method | Input | What it computes | Typical use |
|--------|-------|-----------------|-------------|
| `g0xyz` | Single raw accel vector at known zero tilt | Rotation to align sensor Z with gravity | Lab calibration, known plumb reference |
| `time_ranges_zeroing` | Data interval with instrument at rest | Mean tilt rotation from multiple samples | Field zeroing, post-deployment correction |
| `time_ranges_azimuth` | Data interval at known tilt direction | Azimuth shift from mag+accel unit vectors | Field azimuth calibration |

Both write the result to `Rz` in the coefficients. If both are set, `g0xyz` takes
precedence (computed in `prepare_coefs` after `time_ranges_zeroing`, so it overwrites).

### Azimuth Calibration (Tilt Direction)

`azimuth_shift_deg` corrects the **tilt direction** (azimuth of the inclinometer's
lean) from sensor coordinates to geographic coordinates. Default is `180°` to
compensate the magnetometer sign inversion applied at load time (``invert_magnetometer``
in ``csv_load.py`` → ``Mxyz`` negated in ``format_loaded.py``).

Physical meaning: an inclinometer measures the **azimuth of tilt direction**
(where the current comes from), not the instrument's compass heading. The
magnetometer determines North direction, the accelerometer defines the tilt
plane. The `Vdir` formula uses the cross product `G × H` (gravity × magnetic
field) to find the tilt direction in sensor coordinates, then `azimuth_shift_deg`
converts to degrees from North.

#### Calibration Procedure

Full calibration (call examples in `scripts/26xx_calibr_incl_hy.py`):

1. **Lab** (ellipsoid fit) — requires **full environment** (scipy, h5py):
   run `calibration.run.run_calibration` on magnetometer + accelerometer data
   → yields `Ag, Cg, Ah, Ch`. In noh5 these are must be set in configuration.

2. **Tilt zeroing** (`time_ranges_zeroing`) — instrument hangs plumb, data
   recorded → computes `Rz` (rotation matrix aligning sensor Z with gravity).

3. **Velocity calibration** (in tank/flume) — determines `kVabs` (inclination-
   to-velocity polynomial).

4. **Azimuth calibration** (`time_ranges_azimuth`) — instrument tilted in a
   **known direction** (e.g. known to be tilted Northward), data recorded.
   Pipeline computes azimuth shift and writes `azimuth_shift_deg` to YAML.

#### YAML Configuration

```yaml
# All calibrations at once:
input:
  time_ranges_zeroing: ["2026-06-25T17:23:30", "2026-06-25T17:25:00"]   # Rz
  time_ranges_azimuth: ["2026-06-25T17:23:30", "2026-06-25T17:25:00"]   # azimuth_shift_deg
  time_ranges: ["2026-06-25T17:23:30", "2026-06-25T17:25:00"]           # data filter
  coefs:
    azimuth_shift_deg: 180   # default (compensates invert_magnetometer we do by default)
```

**Layering on top of computed azimuth**: `azimuth_add` (manual offset, degrees)
and `coordinates` + `data_date` (magnetic declination via `pygeomag`).

### Phased Processing (`return_`)

The `program.return_` field controls how far the pipeline runs before stopping.
Useful for debugging, discovery, or coef-only updates:

| `return_` value | Stops after | Typical use |
|:---|---|---|
| `<cfg_from_args>` | Config composition (no I/O) | Scan input, generate missing configs (existing preserved), stop |
| `<gen_names_and_log>` | Config generation | Write YAML files, stop |
| `<saved_coefs>` | Coef persistence only | Zeroing/azimuth → save coefs to YAML (with backup) + NC, stop before raw NC save |
| `<saved_raw>` | Coef persistence + raw NC save | Verify raw ingestion, or zeroing-only |
| `<saved_noavg>` | No-avg processed output | Diagnostic without full binning |
| `<saved_all>` | All binned NC writes | Skip combined output |
| `<end>` (default) | Full pipeline | Normal processing |

**Discover-only** (generate configs, don't process):
```bash
python -m scripts.tcm_clc "_raw/i*.txt" 'program.return_="<cfg_from_args>"'
```

**Zeroing-only** (compute Rz, persist coefs, stop — no data processing):
```bash
# Text CSV source — coefs saved to YAML (with backup) + raw NC if available
python -m scripts.tcm_clc "_raw/*i*.txt" \
  'input.time_ranges_zeroing=["2026-06-25T17:23:30","2026-06-25T17:25:00"]' \
  'input.time_ranges=["2026-06-25T17:23:30","2026-06-25T17:25:00"]' \
  'program.return_="<saved_coefs>"'

# Binary NC source
python -m scripts.tcm_clc "260624.raw.nc" \
  'input.tables=["incl_p05"]' \
  'input.time_ranges_zeroing=["2026-06-25T17:23:30","2026-06-25T17:25:00"]' \
  'input.time_ranges=["2026-06-25T17:23:30","2026-06-25T17:25:00"]' \
  'program.return_="<saved_coefs>"'

# To also save raw data to NC, use <saved_raw> instead of <saved_coefs>
```

> **Note**: `<saved_raw>` persists coefs to NC when h5py is available, or to the
> run YAML in noh5 mode. In noh5 mode, raw data cannot be saved to NC (no pytables),
> but coef changes ARE written to the YAML.

### `force_reprocess`

By default, the pipeline skips re-processing if the output NC already covers the
input time range (incremental update). Pass `+force_reprocess=True` to override
this containment check:

```bash
python -m scripts.tcm_clc "_raw/i*.txt" +force_reprocess=True
```

`force_reprocess` affects **processed** NC writes only (`*.proc_noAvg.nc`,
`*.proc.nc`) — it does NOT affect coefficient persistence (coefs always overwrite
in-place). Use it when filter parameters, coefficients, or input window changed
and you need to re-bin existing data.

When skipping, the pipeline compares stored `_run_params` (filter + window +
coefficient text) to the current values and emits a unified-diff warning on
mismatch — so you see exactly what changed.

See `config_reference.md` for the full decision table.

## Time Correction

Raw instrument timestamps are recorded at integer-second resolution. With N Hz
sampling, N consecutive rows share the same second. The pipeline corrects this
by detecting the sampling frequency, removing anomalies (spikes, backward jumps),
and snapping remaining data to a regular time grid.

Three modes (set via `filter.corr_time_mode`):
- **Snap-to-grid** (default) — full correction pipeline
- **Mask-only** — removes bad timestamps but keeps original values
- **Delete inversions** — removes anomalies only

See `docs/tcm_clc/config_reference.md` for mode details and config fields.

## Input Data Requirements

### File naming → coefficient mapping

The pipeline derives each probe's **pcid** from its filename, then maps pcid to a
**table name** which selects the correct coefficient file.  Understanding this chain
is essential: wrong filename → wrong table → wrong coefficients.

| Filename | pcid | Table | Coef file in `yaml_export/` |
|----------|------|-------|-----------------------------|
| `i_01.txt` | `i01` | `incl01` | `incl01.yaml` |
| `i_p05.txt` | `i_p05` | `incl_p05` | `incl_p05.yaml` |
| `INKL_P05_0_v_trube.TXT` | `i_p05` | `incl_p05` | `incl_p05.yaml` |
| `incl_b03.txt` | `i_b03` | `incl_b03` | `incl_b03.yaml` |
| `I_P01_001.txt` | `i_p01` | `incl_p01` | `incl_p01.yaml` |
| `w01.txt` | `w01` | `w01` | — |

The mapping rules (implemented in `tcm/format.py`):
1. The first `i` or `w` (case-insensitive) is captured as the **probe type**. Everything before it is ignored (e.g. the digits in `30967_i90.txt`)
2. Common instrument-name suffixes after the type letter (`nkl`, `ncl`) are consumed but ignored — they are part of the raw instrument name, not probe identity
3. Model letter after type: `p`, `b`, `d` (or none for plain inclinometers)
4. Probe number: leading zeros stripped, then re-padded to ≥2 digits (`090` → `90`, `001` → `01`)
5. Everything after the number is a **comment suffix** — stripped for identity, preserved in corrected filename
6. Table name = `incl` + model + number (e.g. `i_p05` → `incl_p05`)

**Normalization examples** (all resolve to the same probe `i90` / table `incl90`):

| Filename | consumed prefix | type | suffix | model | number | comment | pcid |
|----------|----------------|------|--------|-------|--------|---------|------|
| `INKL_090.TXT` | — | `i` | `nkl_` | — | `90` | — | `i90` |
| `INCL_090.TXT` | — | `i` | `ncl_` | — | `90` | — | `i90` |
| `I_090.TXT` | — | `i` | `_` | — | `90` | — | `i90` |
| `i90.txt` | — | `i` | — | — | `90` | — | `i90` |
| `30967_i90.txt` | `30967_` | `i` | — | — | `90` | — | `i90` |
| `INKL_090_переход.TXT` | — | `i` | `nkl_` | — | `90` | `_переход` | `i90` |

The `I`/`i` in `INKL_` / `INCL_` / `I_` is the **probe type**, not a prefix.
Only `nkl`/`ncl` (instrument model name) and `_` (separator) are consumed and ignored.

**Key rules for naming source files**:
- Leading zeros in probe numbers are **ignored**: `090`, `0090`, and `90` all resolve to pcid `i90`.
- The **comment suffix** (everything after the number, e.g. `_переход`, `_v_trube`) does not affect probe identity.  Two files with the same number but different comments share the same pcid and coefficients — they are treated as different source files for the same probe.
- The **model letter** (`p`, `b`, `d`) is significant: `i_p05` and `i05` are **different** probes with different coefficient files.
- Corrected files (prefixed with `@`) are matched identically — `@i_01.txt` and `i_01.txt` resolve to the same probe.  The `@` prefix is stripped before matching.

**Verification**: run with `program.return_="<cfg_from_args>"` to see which
pcid/table the pipeline assigns without processing:

```bash
python -m scripts.tcm_clc "_raw/i*.txt" 'program.return_="<cfg_from_args>"'
```

The log shows: `Coefs for i_p05: paths=[...], date=2023-08-13` — confirming
the probe identity and which coef file was loaded.

### Required columns

Raw CSV/TXT files must contain at minimum (header auto-detected):

| Probe type | `text_type` | Required columns | Optional |
|------------|:-----------:|------------------|----------|
| Inclinometer | `i`, `b` | `Ax, Ay, Az, Mx, My, Mz` | `Battery, Temp` |
| Pressure | `p`, `d` | `Ax, Ay, Az, Mx, My, Mz, P_counts` | `Battery, Temp` |
| Wave gauge | `w` | `Battery, Temp` | — |

Override auto-detection with `input.text_type` if the header is non-standard.

## Console Messages

During processing, the pipeline logs key messages at INFO and WARNING levels.
Below are the most important ones to watch for.

**Coefficient messages** (one per probe at INFO):

| Message pattern | Meaning |
|----------------|---------|
| `Coefs for i_p05: paths=[...], date=2023-08-13` | Coefficients loaded; `paths` shows the source chain (yaml_export, coefs_path, etc.) |
| `Coefs prepared: with new rotation to direction averaged on configured time_ranges_zeroing interval` | Zeroing rotation computed from data interval |
| `Coefs prepared: with new rotation to user defined zero point (g0xyz)` | Zeroing rotation from `g0xyz` reference vector |
| `Coefs unchanged for i_p05 — skipping write` | No coef changes detected; nothing written to disk |
| `Overwrote coefs [Rz] in 260624.raw.nc` | Changed coefs written back to NC source file |
| `Updated coefs [Rz] in @i_p05.yaml` | Tilt zeroing rotation written to run YAML (noh5 mode) |
| `Coefs saved to ...//incl_p05: 12 datasets` | Full coefs written to raw NC file |
| `Zeroing data -> no-op: time_ranges_zeroing ... not in current data range` | Warning: zeroing interval has no data — check your time ranges |
| `Zeroing azimuth in interval ... (N points): azimuth shift=X°` | Azimuth computed from data interval |

**Config discovery and sync** (at INFO):

| Message pattern | Meaning |
|----------------|---------|
| `Config generation: regenerating N stale configs` | Source files changed; configs regenerated |
| `Discovered N file groups from {path}` | Source files found during discovery |
| `{pcid}: info_devices [{start}, {end}]` | Time range from metadata for this probe |
| `  written to {stem}.yaml` | Time range written from metadata to config |
| `  already configured but broader than metadata: {stem}.yaml [...]` | Warning: config has wider range than metadata |

**Time correction** (at WARNING only when anomalies exceed thresholds):

| Message pattern | Meaning |
|----------------|---------|
| `time correction: N/M monotone; X% removed; correction [min, max]s` | Clean correction — no action needed |
| `time correction: N/M monotone (in-range=K); X% removed (spikes=S, backward=B); ...; A pts > alarm Thr` | Warning: significant time anomalies — check diagnostics |
| `diagnostics {path} saved (N events): HOLE=..., ALARM=...` | Diagnostics NPZ saved for detailed analysis |

**Pipeline summary** (at INFO):

| Message pattern | Meaning |
|----------------|---------|
| `Done — N probes: i90, i67 ok` | All requested probes processed successfully |
| `Done — N probes: i90 ok | 1 skipped (i64)` | Some probes skipped (not in input.ids or no source data) |
| `Done — N probes: i90 ok | 1 failed (i67)` | Some probes failed — check errors above |

Log file location: `cfg_proc/log/{timestamp}/processing.log` (inside the data directory).
Full log includes DEBUG-level detail (per-stem checks, snap RMS, segment counts, etc.).

## Related Documentation

| File | Audience | Content |
|------|----------|---------|
| `docs/tcm_clc/how_it_works.md` | Programmers | Internal architecture, code references, data flow |
| `docs/tcm_clc/config_reference.md` | All users | Complete config field reference, decision tables, tuning examples |
| `docs/calibration/calibration_wiki.md` | Programmers | Ellipsoid fitting method, calibration math |
| `docs/tcm_clc/build_tcm_clc_txt[Ru].md` | Builders | noh5 distribution build instructions (Russian) |
| `docs/tcm_clc/readme_noh5[Ru].md` | End users | Russian version of this guide (noh5 distribution) |
| `_dask_legacy/docs/tcm_clc/README.md` | Programmers | Legacy HDF5/dask pipeline guide |
