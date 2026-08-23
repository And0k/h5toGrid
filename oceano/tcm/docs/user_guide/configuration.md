# Configuration Guide

The pipeline is configured through a layered YAML system built on
[Hydra](https://hydra.cc/) and OmegaConf. This guide covers the practical
aspects of editing configuration. For the exhaustive field reference, see
[Config Reference](../reference/config_reference.md). For behavior decision
tables, see [Config Tuning](../reference/config_tuning.md).

## Config system overview

Four configuration groups control the pipeline:

| Group | What it controls | Key fields |
|-------|------------------|------------|
| `input` | Source files, calibration, time ranges | `path`, `ids`, `coefs`, `time_ranges`, `calib` |
| `out` | Output format, binning, text export | `dt_bins`, `text_path`, `overwrite_db` |
| `filter` | Data quality thresholds | `min`, `max`, `corr_time_mode` |
| `program` | Runtime behavior | `return_`, `verbose` |

Configuration is resolved by merging (highest priority last):

1. Bundled defaults (dataclass definitions)
2. `cfg_proc/config.yaml` — optional primary config (overrides bundled defaults)
3. `cfg_proc/run/@{pcid}.yaml` — per-probe YAMLs (override any top-level field)
4. CLI overrides — applied during config composition

## Per-probe YAML files

Each probe's config lives in `cfg_proc/run/{prefix}@{pcid}-{comment}.yaml`.

- The `{prefix}` (e.g. `260624_1730`) is auto-generated from the data's first
  timestamp. If no timestamp is available, the file is simply `@{pcid}.yaml`.
- Everything before the **last** `@` is metadata — ignored for probe identity.
  The `-{comment}` suffix after the pcid stem is also stripped.
- So `@i_01.yaml`, `260624_1730@i_01.yaml`, `@i_01-extra.yaml` all resolve to
  the same probe `i01`.

Configs are auto-generated from file discovery and **never overwritten** on
subsequent runs — only missing configs are created, and stale ones (whose
`input.path` references a deleted file) are regenerated.

**`input.path` is the real link** between a YAML and its source data file.
The YAML filename is metadata only — the pipeline reads `input.path` to locate
the data, not the YAML name.

**Deduplication**: when regenerating configs, the pipeline checks if any
existing YAML for the same normalized pcid already has a valid `input.path`.
If so, no new YAML is created.

## Minimal viable config

The simplest useful per-probe YAML:

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

The `# @package _global_` directive tells Hydra to merge this YAML at the
Config root rather than under a `run` namespace. Without it, the fields would
be nested under `run.input.*` instead of `input.*`.

Typical user edits: replace default `coefs` with actual calibration values,
narrow `time_ranges` to the deployment period, set `input.calib.coordinates`
for magnetic declination correction.

## Filter expansion (M shorthand)

`input.min`/`max` (load-stage DROP) and `filter.min`/`max` (process-stage NaN-out)
both support `M` as a shorthand that expands to `Mx`, `My`, `Mz`:

```yaml
input:
  max:
    M: 5       # equivalent to Mx: 5, My: 5, Mz: 5
filter:
  min:
    M: 0.001   # equivalent to min.Mx: 0.001, min.My: 0.001, min.Mz: 0.001
```

This avoids repeating the same threshold for all three magnetometer axes.

## Azimuth calibration

`azimuth_shift_deg` corrects the **tilt direction** (azimuth of the inclinometer's
lean) from sensor coordinates to geographic coordinates. Default is `180°` to
compensate the magnetometer sign inversion applied at load time.

Physical meaning: an inclinometer measures the **azimuth of tilt direction**
(where the current comes from), not the instrument's compass heading. The
magnetometer determines North direction, the accelerometer defines the tilt
plane.

### Calibration procedure

1. **Lab** (ellipsoid fit) — requires full environment (scipy, h5py):
   run `calibration.run.run_calibration` on magnetometer + accelerometer data
   → yields `Ag, Cg, Ah, Ch`.

2. **Tilt zeroing** (`time_ranges_zeroing`) — instrument hangs plumb, data
   recorded → computes `Rz` (rotation matrix aligning sensor Z with gravity).

3. **Velocity calibration** (in tank/flume) — determines `kVabs` (inclination-
   to-velocity polynomial).

4. **Azimuth calibration** (`time_ranges_azimuth`) — instrument tilted in a
   **known direction** (e.g. known to be tilted Northward), data recorded.
   Pipeline computes azimuth shift and writes `azimuth_shift_deg` to YAML.

### YAML configuration

```yaml
# All calibrations at once:
input:
  time_ranges: ["2026-06-25T17:23:30", "2026-06-25T17:25:00"]           # data filter
  coefs:
    azimuth_shift_deg: 180   # default (compensates magnetometer inversion)
  calib:
    time_ranges_zeroing: ["2026-06-25T17:23:30", "2026-06-25T17:25:00"]   # Rz
    time_ranges_azimuth: ["2026-06-25T17:23:30", "2026-06-25T17:25:00"]   # azimuth_shift_deg
    coordinates: [54.70, 20.51]   # Kaliningrad (magnetic declination)
    azimuth_add: 2.5              # manual fine-tune
```

**Layering**: `azimuth_add` (manual offset, degrees) and `coordinates` + `data_date`
(magnetic declination via `pygeomag`) are applied **on top of**
the data-computed azimuth.

## Updating coefficients via zeroing

Two independent zeroing operations, each with its own time window:

| Parameter | What | How | Writes |
|---|---|---|---|
| `input.calib.time_ranges_zeroing` | Tilt rotation | `orientation.zeroing_rotation()` on accelerometer data | `Rz` |
| `input.calib.time_ranges_azimuth` | Tilt direction azimuth | `orientation.azimuth_shift()` on mag+accel unit vectors | `azimuth_shift_deg` |

The azimuth computation uses calibrated unit vectors only (no velocity/magnitude
calculation), so it does not depend on `kVabs` or inclination-to-magnitude
coefficients.

### Zeroing with `g0xyz`

Instead of computing `Rz` from a data interval, you can supply `g0xyz` — the raw
accelerometer vector `[Ax, Ay, Az]` measured when the instrument was at zero tilt:

```yaml
input:
  calib:
    g0xyz: [100.5, 50.2, 980.1]   # raw accelerometer at zero tilt
```

When `g0xyz` is set, it **overrides** any existing `Rz` in the coefficients.

| Method | Input | What it computes | Typical use |
|--------|-------|-----------------|-------------|
| `input.calib.g0xyz` | Single raw accel vector at known zero tilt | Rotation to align sensor Z with gravity | Lab calibration, known plumb reference |
| `input.calib.time_ranges_zeroing` | Data interval with instrument at rest | Mean tilt rotation from multiple samples | Field zeroing, post-deployment correction |
| `input.calib.time_ranges_azimuth` | Data interval at known tilt direction | Azimuth shift from mag+accel unit vectors | Field azimuth calibration |

### Coefficient persistence

Where updated coefs are persisted depends on the input source and environment:

| Source | h5py available | Updated coefs → |
|--------|:---:|-----------------|
| CSV/TXT | Yes | `*.raw.nc` file (`/{tbl}/coef/` group) |
| NC/HDF5 | Yes | Source NC file (overwritten in-place) |
| Any | No | Run YAML (`cfg_proc/run/@i_01.yaml` under `input.coefs`) |

When coefs are written to NC, the log shows how many datasets were overwritten:
```
Coefs saved to ...//incl01: 12 datasets (2 overwritten)
```

### Typical zeroing workflow

```bash
# From text CSV — generates config, computes Rz, proceeds with processing
tcm_proc.exe "_raw/*i*.txt" \
  'input.calib.time_ranges_zeroing=["2026-06-25T17:23:30","2026-06-25T17:25:00"]' \
  'input.time_ranges=["2026-06-25T17:23:30","2026-06-25T17:25:00"]'

# Zeroing-only (compute Rz, persist coefs, stop — no data processing):
tcm_proc.exe "_raw/*i*.txt" \
  'input.calib.time_ranges_zeroing=["2026-06-25T17:23:30","2026-06-25T17:25:00"]' \
  'input.time_ranges=["2026-06-25T17:23:30","2026-06-25T17:25:00"]' \
  'program.return_="<saved_coefs>"'
```

## Coefficient resolution priority

Defined in [Config Reference §`input.coefs`](../reference/config_reference.md#inputcoefs--calibration-coefficients)
(`#### Detailed` block). From highest to lowest:

1. `input.coefs` in the per-probe YAML
2. `coefs_path` — shared coefficient file
3. Bundled `yaml_export/` fallback
4. Dataclass defaults

## See also

- [Config Reference](../reference/config_reference.md) — all YAML fields with descriptions
- [Config Tuning](../reference/config_tuning.md) — behavior decision tables, overwrite_db
- [Processing](processing.md) — pipeline stages, zeroing workflow
- [I/O Formats](../reference/io_formats.md) — coefficient source priority, NC storage
