# CLI Reference

Authoritative specification of the `tcm_proc` command-line interface.
Implementation: [`scripts/tcm_proc.py`](../../scripts/tcm_proc.py),
[`cli.py`](../../src/tcm/cli.py).

## Invocation

```bash
# Compiled distribution (recommended)
tcm_proc.exe <data_path> [OVERRIDES...]

# From source
python -m tcm.scripts.tcm_proc <data_path> [OVERRIDES...]
```

The compiled `tcm_proc.exe` is built via PyInstaller. In the noh5 distribution
it runs without h5py/pytables; in the full distribution it includes HDF5/NetCDF
support.

## Positional argument

The first positional argument is `data_path` — a **path to scan**: directory,
glob, or regex.

| Input form | Behavior |
|------------|----------|
| Directory (e.g. `"_raw/"`) | Scans for `i*.txt` (case-insensitive default); a single `_raw` is processed, a parent with several `_raw` anchors is listed for selection (GUI) or rejected with a hint (CLI: pick one anchor) |
| Glob (e.g. `"*i*.txt"`) | Glob matching — wildcards trigger glob mode |
| Regex (e.g. `"i.*\.txt"`) | Regex matching — escaped dots, `\|`, `(...)` trigger regex mode |
| `.yaml` path | Loads pre-built configs directly, skips discovery |
| `.nc` / `.h5` path | Binary input — skips text discovery, processes directly |

### Pattern classification

`input.path` is automatically classified as glob or regex:

| Condition | Mode | Example input | Effective regex |
|-----------|------|---------------|-----------------|
| Invalid regex (compilation fails) | glob | `*[0bdp]*.txt` | `.*?[0bdp].*?\.txt` |
| Valid regex, extension dot **unescaped** | glob | `file?.txt` | `file.\.txt` |
| Valid regex with `\|` or `(...)` wrapper | regex | `(a\|b).txt` | `(a\|b).txt` |
| Valid regex, extension dot **escaped** (`\.`) | regex | `i.*\.txt` | `i.*\.txt` |
| `path` is a directory | default regex `i.*\.txt` | `_raw/` | `i.*\.txt` |

Glob conversion: `*` → `.*?`, `?` → `.`, all dots → `\.` (all case-insensitive).

> **Regex quoting**: any regex pattern (containing `|`, `(`, `)`, `\`, `[`, …)
> **must** be quoted — these characters are shell metacharacters and will break
> the command or silently alter the pattern otherwise.

### Drop-on-shortcut

Windows passes the raw path as `sys.argv[1]` when a file/folder is dropped on
the exe. Commas, backslashes, quotes in the path are handled automatically —
`input.path` is injected directly into `DictConfig` via OmegaConf merge,
bypassing Hydra's ANTLR override parser entirely.

## Hydra CLI flags

```bash
tcm_proc --help           # all config fields and overrides
tcm_proc --cfg job        # show the composed config without running
tcm_proc --info           # Hydra internals (plugins, search path, defaults)
tcm_proc --hydra-help     # Hydra-specific flags only
tcm_proc --version        # version info
```

Hydra flags work **without** a data path — no `_raw/` directory is needed.
They cause Hydra to print information and exit before the pipeline starts.

## Overrides

Any config field can be overridden on the CLI:

```bash
# Process specific probes only
tcm_proc.exe "_raw/*i*.txt" 'input.ids=[i01,i_p02]'

# Override filter and output settings
tcm_proc.exe "_raw/*i*.txt" filter.corr_time_mode=false out.text_path=./results

# Restrict time window
tcm_proc.exe "_raw" \
  'input.time_ranges=["2024-01-01T00:00:00","2024-01-02T00:00:00"]'

# Phase-stopping: generate configs only (no processing)
tcm_proc.exe "_raw/i*.txt" 'program.return_="<cfg_from_args>"'

# overwrite_db mode
tcm_proc.exe "_raw/i*.txt" out.overwrite_db=splice
```

> **PowerShell note**: wrap each override containing `[""]` or `<>` in single
> quotes to prevent shell interpretation: `'program.return_="<cfg_from_args>"'`.

Overrides from the CLI are applied during config composition. For text-file
processing, per-probe YAML files in `cfg_proc/run/` layer on top of CLI
overrides. For binary inputs (NC/HDF5), dict overrides are the sole config
source.

## Exit codes

| Code | Meaning |
|------|---------|
| `0` | Success — all requested probes processed |
| `Ex_nothing_done` | No probes were processed (no matching configs, all stale, or phase-stopped before data) |
| `1` | Unhandled exception |

The pipeline logs a summary line at INFO:

| Message | Meaning |
|---------|---------|
| `Done — N probes: i90, i67 OK` | All requested probes processed successfully |
| `Done — N probes: i90 OK \| 1 skipped (i64)` | Some probes skipped (not in `input.ids` or no source data) |
| `Done — N probes: i90 OK \| 1 failed (i67)` | Some probes failed — check errors above |

## Environment differences

| Feature | Full distribution | noh5 distribution |
|---------|-------------------|-------------------|
| HDF5/NetCDF I/O | Available (h5py, netCDF4) | Not available |
| `*.raw.nc` output | Written | Skipped |
| `*.proc_*.nc` output | Written | Skipped |
| Combined output (`*.proc.nc`) | Written | Skipped |
| TSV export | Available | Available |
| Coef persistence | NC file (in-place) | Run YAML |
| `program.return_=<saved_raw>` | Saves raw data to NC | Saves coefs to YAML only |
| `dt_bins_min_save_text` default | `1` (no-avg skipped) | `0` (no-avg TSV enabled) |
| `use_h5` default | `'auto'` (detects libraries) | `'off'` (set by runtime hook) |

Log file location: `cfg_proc/log/{timestamp}/processing.log` (inside the data
directory). When `program.return_` is non-default (e.g. `<cfg_from_args>`),
the file is named `processing-{sanitized}.log`.
