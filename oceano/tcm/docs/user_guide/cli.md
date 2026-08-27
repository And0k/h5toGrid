# CLI Guide

Task-oriented guide for the `tcm_proc` command-line interface.
For the exhaustive argument specification, see [CLI Reference](../reference/cli.md).

## Basic invocation

```bash
# Process all probes in a data directory
tcm_proc.exe "D:/data/_raw"

# Process specific probes only
tcm_proc.exe "_raw/*i*.txt" 'input.ids=[i01,i_p02]'

# Override any config field
tcm_proc.exe "_raw/*i*.txt" filter.corr_time_mode=false out.text_path=./results
```

The first positional argument is a **path to scan**: directory, glob, or regex.
See [CLI Reference §Pattern classification](../reference/cli.md#pattern-classification)
for the auto-detection rules.

## First run on new data

```bash
# Phase 1: generate configs (auto-detected from data)
tcm_proc.exe "_raw/i*.txt"

# Edit cfg_proc/run/@i_01.yaml — replace default coefs with actual values
# Then re-run to process:
tcm_proc.exe "_raw/i*.txt"
```

## Phased processing (`return_`)

The `program.return_` field controls how far the pipeline runs before stopping:

| `return_` value | Stops after | Typical use |
|:---|---|---|
| `<cfg_from_args>` | Config composition (no I/O) | Scan input, generate missing configs, stop |
| `<gen_names_and_log>` | Config generation | Write YAML files, stop |
| `<saved_coefs>` | Coef persistence only | Zeroing/azimuth → save coefs, stop before processing |
| `<saved_raw>` | Coef persistence + raw NC save | Verify raw ingestion, or zeroing-only |
| `<saved_noavg>` | No-avg processed output | Diagnostic without full binning |
| `<saved_all>` | All binned NC writes | Skip combined output |
| `<end>` (default) | Full pipeline | Normal processing |

**Discover-only** (generate configs, don't process):
```bash
tcm_proc.exe "_raw/i*.txt" 'program.return_="<cfg_from_args>"'
```

**Zeroing-only** (compute Rz, persist coefs, stop — no data processing):
```bash
tcm_proc.exe "_raw/*i*.txt" \
  'input.calib.time_ranges_zeroing=["2026-06-25T17:23:30","2026-06-25T17:25:00"]' \
  'input.time_ranges=["2026-06-25T17:23:30","2026-06-25T17:25:00"]' \
  'program.return_="<saved_coefs>"'
```

## Coefficient editing workflow

1. Run once to generate configs: `tcm_proc.exe "_raw"`
2. Edit `cfg_proc/run/@i_XX.yaml` — replace default `coefs` with actual values
3. Re-run: `tcm_proc.exe "_raw"` — pipeline picks up edited YAMLs

**Re-process with tweaked coefficients**: edit the probe's YAML, re-run.

**Batch re-run after adding data files**: the tool appends incrementally
to existing `*.raw.nc` files. Re-running is safe — overlapping data is
detected and skipped.

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

See [Config Tuning](../reference/config_tuning.md) for the full decision tables.

## `overwrite_db` modes

Controls how the pipeline handles existing output when re-running:

| Mode | Description |
|:---:|---|
| `None` (default) | **Extend-only** — append new data; never reprocess or trim |
| `"splice"` | **Always reprocess** — keep data outside `time_ranges`, replace inside |
| `"trim"` | **Trim-only** — delete data outside `time_ranges`, no reprocessing |
| `"export"` | **Export-only** — block NC writes, export TSV only |

```bash
tcm_proc.exe "_raw/i*.txt" out.overwrite_db=splice
```

See [Config Reference §overwrite_db](../reference/config_reference.md#outoverwrite_db)
for the full decision matrix.

## See also

- [CLI Reference](../reference/cli.md) — exhaustive argument spec
- [Input / Output Format Specification](../reference/io_formats.md) — input/output format contracts
- [Configuration Guide](configuration.md) — config system guide
- [Processing Guide](processing.md) — pipeline stages, zeroing, time correction
