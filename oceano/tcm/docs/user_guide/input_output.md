# Input / Output Guide

Practical guide to what goes in and what comes out.
For the exhaustive format specification, see [I/O Formats Reference](../reference/io_formats.md).

## Input formats

The pipeline accepts three input formats, auto-detected from file extension:

| Extension | What happens |
|-----------|-------------|
| `.txt`, `.csv` | CSV loading with header auto-detection |
| `.raw.nc` | Native xarray read; coefficients from `/{tbl}/coef/` group |
| `.raw.h5` | HDF5 read via pytables; coefficients from `/{tbl}/coef/` group |

Both `.raw.nc` and `.raw.h5` carry embedded coefficients — the processing
pipeline runs identically regardless of input format. `.raw.h5` requires
pytables (`TABLES_AVAILABLE`).

## Probe identity (pcid)

Each probe is identified by a canonical string (`pcid`) that links raw data,
coefficients, config, and output:

- `i01` — inclinometer probe 01
- `i_p02` — inclinometer model-p probe 02
- `w01` — wave probe 01

The pcid maps bidirectionally to config YAML stems (`@i_01`) and raw data
filenames; when saving to NetCDF4/HDF5 it maps to the table name (`incl01`).

### File naming rules

The pipeline derives pcid from the filename:

1. The first `i` or `w` (case-insensitive) is the **probe type**
2. Instrument-name suffixes (`nkl`, `ncl`) are consumed but ignored
3. Model letter: `p`, `b`, `d` (or none)
4. Probe number: leading zeros stripped, re-padded to ≥2 digits
5. Everything after the number is a **comment suffix** — stripped for identity

**Normalization examples** (all resolve to the same probe `i90`):

| Filename | pcid |
|----------|------|
| `INKL_090.TXT` | `i90` |
| `I_090.TXT` | `i90` |
| `i90.txt` | `i90` |
| `30967_i90.txt` | `i90` |
| `INKL_090_переход.TXT` | `i90` |

**Key rules**:
- Leading zeros are ignored: `090`, `0090`, `90` all → `i90`
- Comment suffixes don't affect identity: `_переход`, `_v_trube` are stripped
- Model letter is significant: `i_p05` ≠ `i05`
- `@`-prefixed files match identically: `@i_01.txt` = `i_01.txt`

**Verification**: run with `program.return_="<cfg_from_args>"` to see which
pcid/table the pipeline assigns without processing.

## Output files

| File | Contents |
|------|----------|
| `*.raw.nc` | Raw data + calibration coefficients (incremental append) |
| `*.proc_noAvg.nc` | Non-averaged processed output (per-probe groups) |
| `*.proc_Avg.nc` | Binned processed output (per-probe groups for each bin) |
| `*.proc.nc` | Combined multi-probe output (probe dimension) + combined TSV |
| `text_output/{timestamp}@{pcid}.tsv` | Tab-separated text exports (per-probe, binned only) |

Per-probe binned data → `*.proc_Avg.nc` (one group per pcid per bin interval).
When multiple probes are processed in one run, combined groups (with a `probe`
dimension) → `*.proc.nc`. Non-averaged (`dt_bin=0`) data is never combined.

## See also

- [I/O Formats Reference](../reference/io_formats.md) — full format contracts
- [Configuration](configuration.md) — config system, YAML editing
- [Processing](processing.md) — pipeline stages, binning, combined output
