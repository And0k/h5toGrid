# Processing Guide

Overview of the data processing pipeline — from raw sensor readings to
binned output files. For behavior contracts and decision tables, see
[Config Tuning](../reference/config_tuning.md).

## Pipeline stages

The processing pipeline applies these stages in order:

1. **Filter local** — NaN-out on raw columns where `filter.min`/`max` thresholds
   are exceeded (process-stage NaN-out; distinct from `input.min`/`max` which
   DROP rows at load time)
2. **Calibration** — apply gain/offset matrices (`Ag`, `Cg`, `Ah`, `Ch`),
   compute velocity from inclination (`kVabs`), convert polar → Cartesian
3. **Pressure** — 2-D polynomial (`P_t`) + burst filtering
4. **Binning** — `resample(time=dt_bin).mean()` with NaN threshold on valid-sample
   count

Coefficient application order: `prepare_coefs` (zeroing rotation) → `fG(Ag,Cg)`
→ `fInclination` → `v_abs_from_incl(kVabs)` → `azimuth_shift_deg` → `polar2dekart`.

## Zeroing workflow

Zeroing aligns the sensor frame with gravity and geographic north. Two
independent operations:

| Operation | Config field | Output coefficient |
|-----------|-------------|-------------------|
| Tilt zeroing | `input.calib.time_ranges_zeroing` | `Rz` (rotation matrix) |
| Azimuth calibration | `input.calib.time_ranges_azimuth` | `azimuth_shift_deg` |

**Tilt zeroing**: the instrument hangs plumb in a known orientation. The
pipeline averages accelerometer data over the specified time window and
computes a rotation matrix (`Rz`) that aligns the sensor Z-axis with gravity.

**Azimuth calibration**: the instrument is tilted in a **known direction**
(e.g. facing North). The pipeline computes the azimuth offset from
magnetometer + accelerometer unit vectors and writes `azimuth_shift_deg`.

**Alternative — `input.calib.g0xyz`**: supply a raw accelerometer vector measured at
known zero tilt. The pipeline computes the rotation directly, overriding
any `Rz` from `input.calib.time_ranges_zeroing`.

See [Configuration §Updating coefficients via zeroing](configuration.md#updating-coefficients-via-zeroing)
for the full workflow and persistence details.

## Time correction

Raw instrument timestamps are recorded at integer-second resolution. With N Hz
sampling, N consecutive rows share the same second. The pipeline corrects this
by detecting the sampling frequency, removing anomalies (spikes, backward jumps),
and snapping remaining data to a regular time grid.

Three modes (set via `filter.corr_time_mode`):

| Mode | Behavior |
|------|----------|
| `True` (default) | **Snap-to-grid**: full correction — detect frequency, remove anomalies, snap to regular sub-second timestamps |
| `None` / `False` | **Mask-only**: remove backward/spike samples but keep original timestamps. For integer-second N Hz data, N-1 samples per second are removed → collapses to 1 Hz |
| `"delete_inversions"` | Remove anomalies only (outlier + trim), timestamps unchanged. Non-monotone positions masked |

**Config fields affecting time correction**:

| Field | Default | Effect |
|-------|---------|--------|
| `corr_time_mode` | `True` | Snap-to-grid vs mask-only vs delete_inversions |
| `dt_interp_between` | `1.5s` | Minimum gap to detect a real hole (vs jitter within a segment) |
| `corr_time_outlier_threshold_s` | `0.6s` | Spike/backward detection threshold |

See [Config Tuning §Time correction modes](../reference/config_tuning.md#time-correction-modes)
for details.

## Phased processing

The `program.return_` field controls how far the pipeline runs before stopping.
Useful for debugging, discovery, or coefficient-only updates:

| `return_` value | Stops after | Typical use |
|:---|---|---|
| `<cfg_from_args>` | Config composition (no I/O) | Scan input, generate missing configs, stop |
| `<gen_names_and_log>` | Config generation | Write YAML files, stop |
| `<saved_coefs>` | Coef persistence only | Zeroing/azimuth → save coefs, stop before processing |
| `<saved_raw>` | Coef persistence + raw NC save | Verify raw ingestion |
| `<saved_noavg>` | No-avg processed output | Diagnostic without full binning |
| `<saved_all>` | All binned NC writes | Skip combined output |
| `<end>` (default) | Full pipeline | Normal processing |

See [CLI Guide §Phased processing](cli.md#phased-processing-return_) for
examples.

## Combined multi-probe output

When multiple probes are processed in one run, the pipeline reads per-probe
binned groups from `*.proc_Avg.nc`, merges them along a `probe` dimension,
and writes combined groups to `*.proc.nc`.

| Output | Group pattern | Content |
|--------|--------------|---------|
| `*.proc.nc` | `/{probe_type}_bin{N}s/` | All probes, binned, with `probe` dimension |
| TSV | `{ts}bin{N}s@{pcid1},{pcid2}.tsv` | Combined tab-separated text |

**Key rules**:
- Only **distinct** pcids are combined — multiple source files for the same
  pcid are deduplicated
- Non-averaged (`dt_bin=0`) data is **never combined** — per-probe only
- Combined TSV column order: `v_i01, u_i01, v_i02, u_i02, ...`
  (Vabs/Vdir/inclination excluded from combined TSV)

## Re-run behavior

The pipeline handles idempotency automatically:

| Output | Re-run behavior |
|--------|----------------|
| `*.raw.nc` | **SKIP** — same fileName + mtime detected via log table |
| `*.proc_Avg.nc` | **SKIP** — new time range ⊂ existing range |
| `*.proc_noAvg.nc` | **SKIP** — new time range ⊂ existing range |
| Combined groups | **Overwrite** — always rewrites from per-probe groups |

If processing parameters changed (coefs, filter thresholds), the pipeline
raises `ValueError` with a unified diff showing what changed. Pass
`out.overwrite_db=splice` to force reprocessing.

See [Config Tuning §overwrite_db behavior](../reference/config_tuning.md#overwrite_db-behavior)
for the full decision matrix.

## Output column order

Output columns follow this ordering:

```text
v, u, inclination                          ← persisted in NC (velocity/direction group)
Pressure, Temp, Battery, ...               ← remaining sensor variables
```

**Vabs/Vdir save policy**:

| Output | Vabs/Vdir | inclination |
|--------|:---------:|:-----------:|
| NC files | not saved | saved |
| Per-probe TSV | computed on-the-fly | saved |
| Combined TSV | not saved | not saved |

## See also

- [Config Tuning](../reference/config_tuning.md) — overwrite_db contracts, incremental append
- [Configuration](configuration.md) — config system, zeroing setup
- [Console Messages](console_messages.md) — log messages, exit codes
- [I/O Formats](../reference/io_formats.md) — output file structure
