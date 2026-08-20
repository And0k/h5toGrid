# Getting Started

## What is tcm

**tcm** (Tilt Current Meter) processes raw data files from AB SIO RAS inclinometer
probes into physical-unit outputs: velocity, inclination, direction, and pressure.
It supports both CLI (`tcm_proc`) and GUI (`tcm_gui`) frontends.

## Distribution types

| Distribution | Contents | Invocation |
|-------------|----------|------------|
| **Compiled (full)** | `tcm_proc.exe` + `tcm_gui.exe` — HDF5/NetCDF + GUI + all deps | Run exe directly |
| **Compiled (noh5)** | `tcm_proc.exe` — no h5py/pytables, text-only output | Run exe directly |
| **From source** | Full Python package | `python -m tcm.scripts.tcm_proc` / `python -m tcm_gui` |

The noh5 distribution omits h5py, pytables, scipy, and matplotlib. All processing
works identically — only NC/HDF5 output and ellipsoid fitting are unavailable.

## Quick start

```bash
# 1. Point at your data directory
tcm_proc.exe "D:/data/_raw"

# 2. Edit generated configs (replace default coefs with actual calibration)
#    Files are in: D:/data/_raw/cfg_proc/run/@i_XX.yaml

# 3. Re-run to process
tcm_proc.exe "D:/data/_raw"
```

On first run the program creates `cfg_proc/` inside the data directory and
auto-generates per-probe config YAMLs in `cfg_proc/run/`. Edit these to set
the correct calibration values, then re-run.

## Workflow overview

```
┌─────────────────────────────────────────────────────────────┐
│  1. Discovery & Config Generation                           │
│     Scan raw data → group by probe identity (pcid)          │
│     Generate per-probe YAML configs (idempotent)            │
├─────────────────────────────────────────────────────────────┤
│  2. Loading & Preparation                                   │
│     Load raw data (txt/csv/nc/h5)                           │
│     Resolve calibration coefficients (priority chain)       │
│     Apply azimuth correction, auto-zeroing rotation         │
├─────────────────────────────────────────────────────────────┤
│  3. Processing & Export                                     │
│     Compute velocity projections, inclination, pressure     │
│     Average over configurable bin intervals                 │
│     Save to NetCDF4 (per-probe groups) + TSV text files     │
└─────────────────────────────────────────────────────────────┘
```

## What you get

```text
data_dir/_raw/        ← raw files (conventional name)
    i_01.txt
    @i_01.txt           ← corrected (auto-generated)
    cfg_proc/           ← Hydra config directory
        run/            ← Per-probe config YAMLs — edit these
        log/            ← Program and hydra logs
    *.raw.nc            ← Raw data + coefficients (incremental)
    *.proc_noAvg.nc     ← Non-averaged processed output
    *.proc_Avg.nc       ← Binned processed output
    *.proc.nc           ← Combined multi-probe output
    text_output/        ← TSV text exports
```

## Next steps

| Task | Guide |
|------|-------|
| Use GUI | [GUI Guide](gui.md) |
| Use CLI | [CLI Guide](cli.md) |
| Understand input/output | [Input/Output](input_output.md) |
| Edit configuration | [Configuration](configuration.md) |
| Processing details | [Processing](processing.md) |
| Console messages | [Console Messages](console_messages.md) |
| Config field spec | [Config Reference](../reference/config_reference.md) |
| Config behavior tables | [Config Tuning](../reference/config_tuning.md) |
| CLI argument spec | [CLI Reference](../reference/cli.md) |
| I/O format spec | [I/O Formats](../reference/io_formats.md) |
