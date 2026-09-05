# Input / Output Format Specification

Authoritative contracts for data formats accepted and produced by the pipeline.
Implementation: [`_xr/io.py`](../../src/tcm/_xr/io.py) (`load_raw`),
[`_xr/dataset.py`](../../src/tcm/_xr/dataset.py) (`open_nc`, `open_csv_chunks`),
[`format.py`](../../src/tcm/format.py) (probe identity).

## Input formats

`load_raw()` is the single entry point for all input. Format is auto-detected
from `input.path` suffix. Extension sets: `EXT_NC = {".nc"}`, `EXT_HDF5 = {".h5"}`
in [`_constants.py`](../../src/tcm/_constants.py).

| Extension | Backend | Coefs source | Notes |
|-----------|---------|-------------|-------|
| `.txt`, `.csv`, `.dat` | `open_csv_chunks()` | `None` (separate `coefs.path`) | Chunked `pd.read_csv`; progressive concat |
| `.raw.nc`, `.nc`, `.nc4` | `open_nc()` | `/{tbl}/coef/` group | `xr.open_dataset(group=tbl)` — native xarray |
| `.raw.h5`, `.h5`, `.hdf5` | `open_hdf5()` | `/{tbl}/coef/` group | `pd.HDFStore` → `DataFrame` → `xr.Dataset.from_dataframe()`; requires `TABLES_AVAILABLE` |

Both NC and HDF5 extract coefficients from `/{tbl}/coef/` within the same file.
All downstream processing is format-agnostic — `load_raw()` returns
`(ds_raw, coefs_from_file)` regardless of source.

### CSV/TXT

Chunked loading via `csv_load.load_from_csv_gen()` (streaming
`pd.read_csv(chunksize=blocksize)`). Each chunk is converted to `xr.Dataset`
by `open_csv_chunks()` and progressively concatenated in `load_raw()` — each
chunk is released after merge.

Header auto-detection via `csv_load.format_parts_select_raw(file_path)`;
falls back to `format_parts_select(text_type)`. Override via `input.text_type`.

#### Required columns

| Probe type | `text_type` | Required columns | Optional |
|------------|:-----------:|------------------|----------|
| Inclinometer | `i`, `b` | `Ax, Ay, Az, Mx, My, Mz` | `Battery, Temp` |
| Pressure | `p`, `d` | `Ax, Ay, Az, Mx, My, Mz, P_counts` | `Battery, Temp` |
| Wave gauge | `w` | `Battery, Temp` | — |

#### Text type → column layout

| `text_type` | Columns read |
|-------------|-------------|
| `i`, `b`, `""` | `[Ax, Ay, Az, Mx, My, Mz, Battery, Temp]` |
| `p`, `d` | `[Ax, Ay, Az, Mx, My, Mz, P_counts, Battery, Temp]` |
| `w` | `[Battery, Temp]` (no inertial sensors) |

### Raw NC

`open_nc()` calls `xr.open_dataset(path, group=tbl, engine="netcdf4")`.
Coefficients are read from `/{tbl}/coef/` group hierarchy (see
[Coefficient NC storage](#coefficient-nc-storage)).

### Raw HDF5

`open_hdf5()` opens via `pd.HDFStore`, reads the table into a `DataFrame`,
then converts to `xr.Dataset.from_dataframe()`. Coefficients are read from
`/{tbl}/coef/` group (same layout as NC). Requires `pytables`
(`TABLES_AVAILABLE` must be `True`).

## File name parsing

`format.parse_name(name)` in [`format.py`](../../src/tcm/format.py) extracts
probe identity parts. Three regex steps (first match wins):

1. **Regular stems**: `[^iw]*(i|w)(nkl|ncl|_?)(b|d|p|)_?0*(\d{1,4})(?P<comment>.*)`
   — The first `i`/`w` is the **probe type** (not a prefix). `nkl`/`ncl` and
   `_` after the type are consumed but ignored (instrument-name suffix).
   Leading zeros before the number are consumed by `0*`. `comment` captures
   everything after the number.
2. **Glob patterns**: broader model capture with `chars3` for glob reconstruction.
3. **Fallback**: `voln_v*` → wave gauge.

### Normalization rules

1. Everything before the first `i`/`w` (case-insensitive) is ignored
   (e.g. `30967_i90.txt` → prefix `30967_` stripped)
2. Common instrument-name suffixes after the type letter (`nkl`, `ncl`) are
   consumed but ignored
3. Model letter after type: `p`, `b`, `d` (or none for plain inclinometers)
4. Probe number: leading zeros stripped, then re-padded to ≥2 digits
   (`090` → `90`, `001` → `01`)
5. Everything after the number is a **comment suffix** — stripped for identity,
   preserved in corrected filename
6. Table name = `incl` + model + number (e.g. `i_p05` → `incl_p05`)

### Normalization examples

| Filename | consumed prefix | type | suffix | model | number | comment | pcid |
|----------|----------------|------|--------|-------|--------|---------|------|
| `INKL_090_переход.TXT` | — | `i` | `nkl_` | — | `90` | `_переход` | `i90` |
| `30967_i90.txt` | `30967_` | `i` | — | — | `90` | — | `i90` |
| `INKL_P05_0_v_trube.TXT` | — | `i` | `nkl_` | `p` | `5` | `_0_v_trube` | `i_p05` |

### Key naming rules

- Leading zeros in probe numbers are **ignored**: `090`, `0090`, and `90` all
  resolve to pcid `i90`
- **Comment suffix** (everything after the number, e.g. `_переход`, `_v_trube`)
  does not affect probe identity. Two files with the same number but different
  comments share the same pcid and coefficients
- **Model letter** (`p`, `b`, `d`) is significant: `i_p05` and `i05` are
  **different** probes with different coefficient files
- Corrected files (prefixed with `@`) are matched identically — `@i_01.txt`
  and `i_01.txt` resolve to the same probe. The `@` prefix is stripped before
  matching

### Corrected file naming

[`mod_name()`](../../src/tcm/format_loaded.py) normalizes raw filenames to
`@{type}_{model}{number}-{comment}.{ext}` — leading zeros stripped
(`INKL_090` → `@i_90.TXT`), trailing suffix preserved as `-comment`
(`INKL_P05_0_v_trube` → `@i_p5-0_v_trube.TXT`).

The corrected stem is **not** the canonical pcid (`i_90`, `i_p5` vs `i90`,
`i_p05`: `_` always present, number unpadded) — normalize via
`to_pcid_from_name(stem_to_pcid(stem))` before comparing.

### pcid ↔ raw table name

[`to_pcid_from_name()`](../../src/tcm/format.py) normalizes any name form
(filename stem, raw table, output column) to the canonical pcid;
[`pcid_to_raw_name()`](../../src/tcm/format.py) maps the canonical pcid to
the raw/noAvg DB table name: `incl` + model + number (the `_` separator
dropped), wave gauges unchanged. Per-probe coef file in `yaml_export/` is
named after the raw table — `{raw_table}.yaml` (inclinometers only; wave
gauges have no coefs).

Config and coefs search compares **canonical pcids only**: any corrected-file
stem normalizes first (`i_90` ≡ `i90`, `i_1` ≡ `i01`), while the `-comment`
suffix stays significant where backup copies must remain distinguishable
([`cli._pcid_key()`](../../src/tcm/cli.py)).

| Canonical pcid | Raw table | `yaml_export/` coef file |
|--------|-----------|--------------------------|
| `i90` | `incl90` | `incl90.yaml` |
| `i_p05` | `incl_p05` | `incl_p05.yaml` |
| `w01` | `w01` | — |

## Coefficient source priority

Highest to lowest:

1. `input.coefs` in the per-probe YAML — probe-specific calibration
2. `coefs.path` — shared HDF5 or YAML coefficient file
3. `tcm/cfg/coef/yaml_export/` directory (bundled distribution fallback)
4. Defaults from the configuration dataclass

When the HDF5 file is missing (e.g. distribution without `calibration.h5`),
coefficients are loaded automatically from `yaml_export/` — no extra
configuration needed.

Implementation: [`incl_calc.coefs.get_coefs_from_cfg()`](../../src/tcm/incl_calc/coefs.py)
builds the three-tier chain, driven by
[`config_yaml.prep_cfg_for_probe()`](../../src/tcm/config_yaml.py). HDF5 paths
are gated on `H5_AVAILABLE` — when h5py is not installed the h5 candidate is
silently skipped.

### Coefficient NC storage

Coefs are stored in `/{tbl}/coef/` groups within `*.raw.nc` files:

```
├─ G
│  ├─ A  (3×3 float64)        — accelerometer gain
│  └─ C  (3 float64)          — accelerometer offset
├─ H
│  ├─ A  (3×3 float64)        — magnetometer gain
│  ├─ C  (3 float64)          — magnetometer offset
│  └─ azimuth_shift_deg (scalar) — azimuth correction
│
├─ Vabs0 (6 float64)           — velocity polynomial
├─ P_t   (3×3 float64)         — pressure-temperature polynomial (optional)
├─ i     (scalar int)          — probe serial number
└─ date  (string attr)         — calibration date
```

Write: `save_coefs_to_nc()` in [`_xr/coefs.py`](../../src/tcm/_xr/coefs.py) —
converts raw coefs dict to flat `{h5_path: value}` via `_coefs_to_h5_dict()`,
then delegates the h5py write to `h5inclinometer_coef.h5copy_coef()`.
Read: `load_coefs_from_nc()` — traverses `/{tbl}/coef/` group hierarchy.
Both are idempotent (overwrite in-place).

## Output files

All paths resolved by [`paths.PathLayout`](../../src/tcm/paths.py) from structural
anchors (`proc_dir`, `raw_dir`). The data directory serves as the working root.

### Directory layout

Above every `_raw/` sits the cruise/device hierarchy discovered via
`meta_finder` (`find_device_dirs` + `is_valid_device_dir`): cruise root →
dated cruise dirs → device dirs (keyword or `@id` suffix) → `_raw/`.
Excluded: `DOC`/`GRIDDING`/`CTD_…` (no keyword/id match), `bad`, `test…`,
`*-`. Full user-facing rules:
[user_guide/meta_finder.md](../user_guide/meta_finder.md); function map:
[meta_finder Integration](meta_finder_integration.md).

```text
data_dir/
├── _raw/               ← raw data and its processing config anchor dir (conventional name; any name works)
│   ├── i_01.txt        ← raw files (original)
│   ├── @i_01.txt       ← raw files (corrected, auto-generated)
│   ├── cfg_proc/       ← Hydra config directory
│   │   ├── config.yaml ← optional primary config
│   │   ├── run/        ← per-probe YAML configs
│   │   └── log/        ← program and hydra logs
│   ├── *.raw.nc        ← raw data archive
│   ├── text_output/    ← TSV output
│   ├── *.proc_noAvg.nc ←
│   ├── *.proc_Avg.nc   ← NC output
│   └── *.proc.nc       ←
```

### Output NC files

| File | Contents | Groups |
|------|----------|--------|
| `*.raw.nc` | Raw data + calibration coefficients (incremental append) | `/{tbl}/`, `/{tbl}/coef/`, `/{tbl}/logFiles/`, `/{tbl}/param_spans/` |
| `*.proc_noAvg.nc` | Non-averaged processed output | `/{pcid}/` per probe |
| `*.proc_Avg.nc` | Binned processed output | `/{pcid}bin{N}s/` per probe per bin |
| `*.proc.nc` | Combined multi-probe output (probe dimension) | `/{probe_type}_bin{N}s/` |

Per-probe binned data is written to `*.proc_Avg.nc` (one group per pcid per
bin interval). Non-averaged (`dt_bin=0`) data lives in `*.proc_noAvg.nc`.

### Combined multi-probe output

When multiple probes are processed in one run, `_combine_probes()` reads from
`*.proc_Avg.nc` and writes combined groups (with a `probe` dimension) to
`*.proc.nc`. Only **distinct** pcids are combined — multiple stems for the
same pcid are deduplicated via `dict.fromkeys()`.
Non-averaged (`dt_bin=0`) data is **never combined**.

| Output | Group | Content |
|--------|-------|---------|
| `*.proc.nc` | `/{probe_type}_bin{N}s/` | All probes, binned, `probe` dim |
| TSV | `{ts}bin{N}s@{pcid1},{pcid2}.tsv` | Combined tab-separated text |

### Output TSV files

| File | Contents |
|------|----------|
| `text_output/{timestamp}@{pcid}.tsv` | Per-probe, binned only |
| `text_output/{timestamp}bin{N}s@{pcid}.tsv` | Per-probe, specific bin |
| `text_output/{timestamp}bin{N}s@{pcid1},{pcid2}.tsv` | Combined multi-probe |

Minimum bin size for TSV export: `out.dt_bins_min_save_text` (default 1 s).
`dt_bin=0` (no-avg) is skipped when threshold > 0.

### Column order

```text
v, u, inclination                          ← persisted in NC
Pressure, Temp, Battery, ...               ← remaining sensor variables
```

**Vabs/Vdir save policy**:

| Output | Vabs/Vdir | inclination |
|--------|-----------|-------------|
| NC files (`*.proc_noAvg.nc`, `*.proc_Avg.nc`, `*.proc.nc`) | **not saved** | saved |
| Per-probe TSV | computed on-the-fly from `v`/`u` | saved |
| Combined TSV (`@joined.tsv`) | **not saved** | **not saved** |

`Vabs = hypot(v, u)`, `Vdir = degrees(arctan2(u, v))`.

### Available `text_columns` values

`text_columns` filters which columns appear in TSV output. Empty (default)
writes **all available** columns for the given output type.

| Column | Per-probe TSV | Combined TSV | Notes |
|--------|:---:|:---:|-------|
| `v` | yes | yes | North velocity component |
| `u` | yes | yes | East velocity component |
| `Vabs` | yes | — | On-the-fly from `v`/`u`; requires `kVabs != None` |
| `Vdir` | yes | — | On-the-fly from `v`/`u`; requires `kVabs != None` |
| `inclination` | yes | — | Sensor tilt angle (degrees) |
| `Pressure` | yes | yes | When `P_t` coefficients provided |
| `Temp` | yes | yes | Temperature (if present in raw data) |

For combined multi-probe TSV, each probe's columns are interleaved per-probe:
`v_i01, u_i01, v_i02, u_i02, ...` (axis=1 concatenation; no inclination).
When `b_all_to_one_col=True`, probes are stacked row-wise instead.

### On-disk encoding

All NC write paths apply:

1. **`_downcast_float32(ds)`** — `float64` → `float32` for every data variable.
   Coordinates left unchanged.
2. **`_compression_encoding(ds)`** — `zlib=True, complevel=9, shuffle=True,
   fletcher32=True, dtype="float32"` for every numeric data variable.
   String/datetime variables excluded.

No `scale_factor` / `add_offset` is set — avoids implicit dtype change on read.

### On-disk time encoding

CF-standard `float64 seconds since 1970-01-01` with calendar
`proleptic_gregorian`. Effective resolution: ~100 ns for current timestamps.

| On-disk | In-memory | Encoder | Decoder |
|---------|-----------|---------|---------|
| `float64` seconds, attr `units="seconds since 1970-01-01"` | `datetime64[ns]` | `_dt_ns_to_cf()` | `_cf_to_dt_ns()` |

### Log table

`/{tbl}/logFiles` group in the NC4 file:

| Variable | On-disk | In-memory | Description |
|----------|---------|-----------|-------------|
| `Date0` (dim) | `float64` s | `datetime64[ns]` | start time of data chunk |
| `fileName` | `S255` | `str` | source file name |
| `fileChangeTime` | `float64` s | `datetime64[ns]` | source file mtime |
| `DateEnd` | `float64` s | `datetime64[ns]` | end time of data chunk |
| `DateProc` | `float64` s | `datetime64[ns]` | processing timestamp |
