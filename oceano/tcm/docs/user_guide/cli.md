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
tcm_proc.exe "_raw/*i*.txt" input.corr_time_mode=false out.text_path=./results
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

`program.return_` stops the pipeline at a checkpoint — from config composition
(`<cfg_from_args>`) through coef persistence (`<saved_coefs>`) to the full run
(`<end>`, default). Full value table and phase order:
[Config Tuning §Phase-stopping](../reference/config_tuning.md#phase-stopping).

**Discover-only** (generate configs, don't process):
```bash
tcm_proc.exe "_raw/i*.txt" 'program.return_="<cfg_from_args>"'
```

**Zeroing-only** (compute Rz, persist coefs, stop):
[Configuration §Typical zeroing workflow](configuration.md#typical-zeroing-workflow).

## Coefficient editing workflow

1. Run once to generate configs: `tcm_proc.exe "_raw"`
2. Edit `cfg_proc/run/@i_XX.yaml` — replace default `coefs` with actual values
3. Re-run: `tcm_proc.exe "_raw"` — pipeline picks up edited YAMLs

**Re-process with tweaked coefficients**: edit the probe's YAML, re-run.

**Batch re-run after adding data files**: the tool appends incrementally
to existing `*.raw.nc` files. Re-running is safe — overlapping data is
detected and skipped.

## Re-run behavior

Re-runs are idempotent: `*.raw.nc` skips on same fileName+mtime; binned NCs
skip when the new range is contained in the existing one; combined groups
always overwrite. A `ValueError` with a unified diff is raised when the new
range is contained but processing parameters changed — pass
`out.overwrite_db=splice` to force reprocessing. Full rules:
[Config Tuning §Re-run behavior](../reference/config_tuning.md#re-run-behavior).

## `overwrite_db` modes

Controls how the pipeline handles existing output when re-running:
`None` extends (default), `"splice"` reprocesses inside `time_ranges`,
`"trim"` deletes outside without reprocessing, `"export"` blocks NC writes.

```bash
tcm_proc.exe "_raw/i*.txt" out.overwrite_db=splice
```

Decision matrix: [Config Tuning §overwrite_db behavior](../reference/config_tuning.md#overwrite_db-behavior).

## See also

- [CLI Reference](../reference/cli.md) — exhaustive argument spec
- [Input / Output Format Specification](../reference/io_formats.md) — input/output format contracts
- [Configuration Guide](configuration.md) — config system guide
- [Processing Guide](processing.md) — pipeline stages, zeroing, time correction
