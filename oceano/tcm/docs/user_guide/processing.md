# Processing Guide

How the `processing.run` entry point turns raw data into NetCDF/TSV output:
entry points → per-config pipeline → the parameters that branch it. Each step
links to the authoritative contract instead of restating it
([Config Tuning](../reference/config_tuning.md) for decision tables,
[Configuration Guide](configuration.md) for setup, [Config Reference](../reference/config_reference.md)
for field definitions, [CLI Internals](../project_developer_guide/CLI.md) for implementation).

## Entry points

- **`processing.run`** — data-directory path: discover source files, generate
  missing/stale YAMLs, filter, then dispatch each config to `run_processing`.
  The `input.path` value selects the mode:
  - **directory / glob / regex** → discover files and generate configs
  - **`…/(stem).yaml`** → process only existing configs matching the stem filter,
    skipping generation ([Config filtering](../reference/config_tuning.md#config-filtering))
  - **concrete data file** → match against each YAML's resolved `input.path`
- **`processing.run_processing`** — one per-probe YAML config; runs the flow below.
  Also the direct entry for programmatic calls, e.g.
  `cli.call_in_raw_dir(processing.run, input={"path": …, "yaml_path": …})`
  ([usage examples](../python_developer_guide/examples.py)).
- **`calibration.run.run_calibration`** — lab ellipsoid fit producing
  `Ag`/`Cg`/`Ah`/`Ch`; a **separate pipeline** writing to HDF5 only —
  [Calibration developer guide](../python_developer_guide/calibration.md).
- **Binary input** (`.nc`/`.h5`): `run()` calls `run_processing` per table,
  skipping discovery, config generation, and device-metadata sync entirely.

## Pipeline flow (per config, `run_processing`)

1. **Resolve probe identity** — `pcid` from the `input.path` stem or `input.tables[0]`
   ([`input.path`](../reference/config_reference.md#inputpath)).
2. **Phase 1 — Load** — chunked CSV parse + time correction, or direct NC/HDF5.
   *Branches*: [time-correction mode](#time-correction-mode), raw-NC fast-path.
3. **Phase 2 — Compute coefs** — `prepare_coefs()` turns `input.calib` + raw data
   into `Rz` / `azimuth_shift_deg` ([Calibration & coefficients](#calibration--coefficients)).
4. **Phase 3 — Persist coefs** — changed `input.coefs` written to NC and/or run YAML;
   `input.calib` is consumed ([Coefficient persistence](#coefficient-persistence)).
5. **Phase 4 — Process** — `_xr.physical.process()`: apply `Ag`/`Cg`/`Ah`/`Ch`/`Rz`,
   `v_abs_from_incl(kVabs)`, pressure `P_t`, then binning
   ([Pipeline stages](#pipeline-stages)).
6. **Phase 5 — Persist outputs** — raw/noAvg/binned NC + TSV, governed by
   [`out.overwrite_db`](#overwrite_db-mode) and
   [re-run behavior](../reference/config_tuning.md#re-run-behavior).
7. **Phase 6 — Combine** — `_combine_probes()` merges per-probe groups along `probe`
   ([Combined multi-probe output](#combined-multi-probe-output)).

### Pipeline stages

Inside `physical.process()`:

1. **Filter local** — NaN-out above `filter.min`/`max` (rows kept), unlike
   `input.min`/`max` which DROP rows at load.
2. **Calibration** — apply `Ag`/`Cg`/`Ah`/`Ch`; velocity from inclination (`kVabs`);
   polar → Cartesian. Order: `Rz` → `fG(Ag,Cg)` → `fInclination` →
   `v_abs_from_incl(kVabs)` → `azimuth_shift_deg` → `polar2dekart`.
3. **Pressure** — 2-D polynomial (`P_t`) + burst filtering.
4. **Binning** — `resample(time=dt_bin).mean()` with a valid-sample NaN threshold.

### Fast paths

- **Raw NC** — text source + `*.raw.nc` covers `time_ranges` + matching `fileName`
  log entry → load data/coefs from NC, skip parsing and the raw-NC write
  ([absent text files](../reference/config_tuning.md#absent-text-files)).
- **Trim** — `overwrite_db="trim"` + `time_ranges` ⊂ existing → trim, re-export TSV,
  no reprocessing.

## Parameters that branch the pipeline

- `program.return_` — stop at a checkpoint (config → coefs → raw → noAvg → all → full):
  [Phase-stopping](../reference/config_tuning.md#phase-stopping).
- `out.overwrite_db` — how existing NC is reused on re-run:
  [overwrite_db behavior](../reference/config_tuning.md#overwrite_db-behavior).
- `input.corr_time_mode` — integer-second timestamp handling:
  [Time correction modes](#time-correction-mode).
- `input.path` ending in `.yaml` — skip generation, process existing configs:
  [Config filtering](../reference/config_tuning.md#config-filtering).
- `input.ids` — restrict to the listed probes.
- h5py availability (`policy.io().h5`) — where computed coefs can be persisted:
  [Coefficient persistence](#coefficient-persistence).
- `out.overwrite_db="export"` — block NC writes, TSV only.

## Calibration & coefficients

### `input.calib` → `input.coefs`

`input.calib` fields are **user triggers**; `prepare_coefs()` (`_xr/coefs.py`)
computes the coefficients from raw data:

| `input.calib` | Computes | Depends on |
|---------------|----------|------------|
| `g0xyz` | `Rz` | raw accel vector at known zero tilt; overrides file/data `Rz` |
| `time_ranges_zeroing` | `Rz` | mean accelerometer tilt over the window |
| `time_ranges_azimuth` | `azimuth_shift_deg` | calibrated mag+accel unit vectors (no `kVabs` dependency) |
| `coordinates` | declination added to `azimuth_shift_deg` | `pygeomag` at station/date |
| `azimuth_add` | manual offset added to `azimuth_shift_deg` | user degrees |

Computation order: base `azimuth_shift_deg` → `time_ranges_azimuth` override →
layer `azimuth_add`/`coordinates` → `time_ranges_zeroing` `Rz` (or `g0xyz` override).
Layering and formulas: [Updating coefficients via zeroing](configuration.md#updating-coefficients-via-zeroing),
[velocity methodology](../methodology/velocity.md#azimuth-shift-psi_shift).

### Coefficient persistence

`input.calib` triggers are **one-shot**: after changed coefficients are successfully
persisted, the block is **consumed** (dropped from the run YAML); only `input.coefs`
persist. Changed entries are stamped into `input.coefs.dates`; `input.coefs.date`
becomes the latest calibration timestamp.

| Source | h5py | Updated coefs → |
|--------|:---:|-----------------|
| NC/HDF5 | yes | source NC, in-place |
| NC/HDF5 | no | run YAML `input.coefs` |
| CSV/TXT | yes | `*.raw.nc` + mirrored to run YAML |
| CSV/TXT | no | run YAML `input.coefs` |

Failed runs (no changed coefs, or no write target) keep the trigger for retry.
Full rules, logs, and backup naming: [Configuration §Coefficient persistence](configuration.md#coefficient-persistence).

**GUI instant-apply**: `g0xyz`, `coordinates`, `azimuth_add` carry a ☑ that computes
the same `prepare_coefs` result in memory; the YAML syncs on Run —
[GUI Guide §Edit coefficients](gui.md#2-edit-coefficients).

## Time correction mode

`input.corr_time_mode` selects `True` snap-to-grid (default), mask-only
(`None`/`False`), or `"delete_inversions"`. Fields and behavior:
[Config Tuning §Time correction modes](../reference/config_tuning.md#time-correction-modes).

## `overwrite_db` mode

`None` (extend only), `"splice"` (reprocess inside `time_ranges`), `"trim"` (delete
outside, no reprocess), `"export"` (TSV only). Decision matrix:
[Config Tuning §overwrite_db behavior](../reference/config_tuning.md#overwrite_db-behavior).

## Combined multi-probe output

- `*.proc.nc` gains `/{probe_type}_bin{N}s/` groups with a `probe` dimension;
  combined TSV is `{ts}bin{N}s@{pcid1},{pcid2}.tsv`.
- Only **distinct** pcids combine; `dt_bin=0` is per-probe only; combined groups
  always overwrite.

## See also

- [Configuration Guide](configuration.md) — config system, per-probe YAML
- [Config Tuning](../reference/config_tuning.md) — behavior decision tables
- [Config Reference](../reference/config_reference.md) — YAML field definitions
- [CLI Guide](cli.md) · [GUI Guide](gui.md) — task-oriented usage
