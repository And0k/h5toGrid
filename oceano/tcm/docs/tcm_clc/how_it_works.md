# How It Works — Internal Architecture

## Module Architecture

```
scripts/tcm_clc.py          ← thin CLI entry point (@hydra.main)
tcm/
    cli.py                  ← parse_data_path, _build_hydra_argv, _prepare_overrides, safe_cfg_dir, main_fun, call_in_raw_dir, process_loading_yaml
    processing.py           ← run() orchestrator, run_processing(), _combine_probes()
    config.py               ← Hydra structured config dataclasses + ConfigStore registration
    config_yaml.py          ← gen_metadata(), save_config_to_yaml(), stale detection
    metadata.py             ← device metadata I/O (get_path_in_parents, load_file_meta, extract_devices_info)
                             extracted from veusz_helpers.common.metadata (no Veusz dependency)
    csv_load.py             ← CSV file discovery (search_csv_files) and correction
    format.py               ← probe identity mapping (pcid, pcid_from_parts, parse_name)
    paths.py                ← PathLayout — declarative, lazy path resolver
    _constants.py           ← RAW_DIR_NAME, version info, optional-dependency flags
    to_omegaconf.py         ← utils
    utils2init.py           ← LoggingStyleAdapter, directory helpers
    incl_calc/
        coefs.py            ← coefficient loading/preparation, get_coefs()
        calc.py             ← pure numpy math kernels (Layer 0)
    _xr/
        coefs.py            ← NC coefs I/O + prepare_coefs (zeroing, azimuth)
        physical.py         ← velocity/pressure/binning pipeline (process())
        storage.py          ← netCDF persistence (incremental append, log table)
        calc.py             ← xr.apply_ufunc wrappers around incl_calc/calc.py
        dataset.py          ← open_csv_chunks, open_nc, merge_probes
        io.py               ← ds_to_csv, open_hdf5
        filters.py          ← data quality filter application
        calibration/        ← standalone calibration pipeline
            calibrate.py    ← ellipsoid fitting (Li & Griffiths quadric-form, pure numpy)
                            ← weighted fit via moments.py for uneven angular coverage
            moments.py      ← sample-weighting scheme for Li-Griffiths fitting
            spatial_binning.py ← 3-D bin avg on sphere (θ+φ), replaces azimuth-only bin_avg
            filtering.py    ← per-channel despiking (despike_channels)
            pipeline.py     ← iterative fit→reject loop (calibrate_pipeline)
            visualization.py← 3-D ellipsoid / channel diagnostic plots
            run.py          ← entry point: run_calibration() programmatic API + CLI via main()
            orientation.py   ← zero-tilt zeroing, heading reference, azimuth_shift

### Module Boundary Rules

- `_xr/` must never import from `_dask_legacy`
- `_dask_legacy` is optional — `csv_load.py` imports it lazily (try/except)
- Functions inside each namespace use clean names (folder is the namespace)
- `incl_calc/calc.py` — pure numpy kernels (Layer 0)

### Unified loading

`_xr/io.py::load_raw()` is the single entry point for all input formats.
Both `processing._load_single` and `calibration.run.run_calibration`
delegate to it — no format-specific code outside `load_raw`:

    .nc / .nc4  →  _xr/dataset.open_nc()      (group-based, coefs from /{tbl}/coef/)
    .h5 / .hdf5 →  _xr/io.open_hdf5()         (pandas HDFStore, coefs via load_coefs)
    .txt / .csv →  _xr/dataset.open_csv()      (csv_load pipeline, no embedded coefs)
    _dask_legacy/           ← legacy dask.dataframe pipeline (optional, lazy import)
```

### Directory resolution

Two-layer architecture in `tcm/paths.py`:

**Layer 1 — Anchor discovery** (stateless functions):

| Function | Purpose | Returns |
|----------|---------|---------|
| `find_dir_raw(path_in)` | Walk ancestors for `_raw/` (case-insensitive) | `_raw/` dir or `None` |
| `_infer_proc_dir(path_in)` | Walk up to digit/inclinometer parent | inferred `proc_dir` or `path_in.parent` |
| `find_dir_raw_absolute(path_in)` | CLI bootstrap anchor (`os.chdir`, `cfg_proc/` search) | `_raw/` dir or fallback (always valid, never `None`) |

`find_dir_raw_absolute` resolution:
1. `find_dir_raw` → `_raw/` ancestor found → return it
2. Not found → warn, return *path_in* if directory else its parent

**Layer 2 — Output paths** (`PathLayout`):
Uses Layer 1 primitives internally for anchor detection — never re-implements
ancestor scanning.  `PathLayout._resolve_anchors` applies this order:
1. `find_dir_raw` → `_raw/` found: `raw_dir = _raw`, `proc_dir = parent`
2. `.proc`/`.proc_noAvg` suffix: `proc_dir = parent`, `raw_dir = parent/_raw`
3. Fallback: `_infer_proc_dir` → digit/inclinometer ancestor or `path_in.parent`

Layer 1 and Layer 2 fallbacks may differ when `_raw/` is absent — by design,
they serve different purposes (CLI CWD vs output path roots).

## Entry point

`scripts/tcm_clc.py` — thin CLI caller using `@hydra.main`.

### CLI

```bash
# ── text files (discovery via cfg_proc/run/*.yaml) ──
# Process all discovered probes
python scripts/tcm_clc.py "_raw/*i*.txt"

# Override any config field
python scripts/tcm_clc.py "_raw/*i*.txt" input.ids=[i01,i_p02]
python scripts/tcm_clc.py "_raw/*i*.txt" out.text_path=./results filter.corr_time_mode=false

# Drop-on-shortcut: Windows passes the raw path as sys.argv[1].
# Commas, backslashes, quotes in the path are handled automatically —
# input.path is injected directly into DictConfig via OmegaConf merge,
# bypassing Hydra's ANTLR override parser entirely.
```

**Binary inputs** (NC/HDF5) are processed via ``call_in_raw_dir()`` directly with
dict overrides — they skip the ``cfg_proc/run/`` discovery sweep:

```python
cli.call_in_raw_dir(
    processing.run,
    input={"path": "260624.raw.nc", "tables": ["incl_p05"],
           "time_ranges": ["2026-06-25T17:23:30", "2026-06-25T17:25:00"]},
)
```

### Flow

1. `cli.parse_data_path(sys.argv)` → extract **first run of consecutive non-flag, non-`key=value` arguments** as the data path (joined with commas to reconstruct paths split by shell comma-handling, e.g. `@i,t-chain`), remaining CLI args returned unchanged
2. `paths.find_dir_raw_absolute(path_in)` → `data_dir`
3. `os.chdir(data_dir)` — all relative paths resolve against data directory
4. Build `sys.argv` for Hydra via `_build_hydra_argv(data_dir)`: only `--config-dir <data_dir>/cfg_proc` (if exists) — targets Hydra's **argparse** layer which natively handles commas, backslashes, colons, parentheses, brackets, braces, equals signs, and other ANTLR special characters. ``input.path`` is injected via ``_prepare_overrides()`` into the overrides dict, then merged into ``DictConfig`` by :func:`hydra_main` — the path string **never passes through Hydra's ANTLR override parser**.
5. `main_fun(processing.run, config_name="config")` — if no dict overrides, uses `@hydra.main(config_name="config", config_path=pkg://tcm.cfg.cfg_proc)` to compose the full `Config`
   (logging, run.dir, resolvers all active). If dict overrides are provided, composes defaults first via `@hydra.main`, then merges overrides on top via `OmegaConf.merge`.
6. `processing.run(cfg)` — canonical orchestrator: discover → generate configs → process.
   For binary inputs this step **branches**: calls ``run_processing`` directly per table,
   skipping the text-only config discovery/generation/sync chain.

### Log output

Hydra writes logs to `{data_dir}/cfg_proc/log/{timestamp}/tcm_clc.log` (configured
in `cfg_proc/config.yaml` via `hydra.run.dir: ./cfg_proc/log/${now:...}`).
Since `os.chdir(data_dir)` is called before `@hydra.main`, all relative paths
resolve inside the data directory.
Hydra's own output (`config.yaml`, `overrides.yaml`) goes to the same directory.

**Logging level policy** (console = INFO, file = DEBUG):

| Level | Used for |
|-------|----------|
| `WARNING` | Anomalies requiring user attention: stale/orphan configs, time correction quality issues (non-monotone, alarm threshold exceeded), raw file correction needed |
| `INFO` | Pipeline milestones: discovery summary, config generation, coefs loaded per probe, data loaded, TSV/NC saved, final summary |
| `DEBUG` | Diagnostic detail: per-stem stale checks, existing time_ranges matches, bilateral spike timestamps, segment counts, snap RMS, per-file load progress |

Key conventions:
- **One consolidated warning** per probe for time correction quality (combines monotone count, spike %, alarm pts)
- **One INFO** per probe for coefs (path + date), not duplicate "Loaded N coefs" + "Coefs for {pcid}" messages
- **Stale config** detection logs per-stem details at DEBUG; the caller (`processing.run`) logs the summary set at WARNING
- **FileNotFoundError** in `process_loading_yaml` is caught separately with context about likely stale config

### Why `@hydra.main` and not Compose API

The pipeline uses `@hydra.main` (not `initialize_config_module` + `hydra.compose`)
because Compose API only provides config resolution — it does **not** set up the
Hydra runtime environment. The following features require `@hydra.main`:

- **Logging**: `@hydra.main` configures log handlers, `hydra.run.dir`, and
  `colorlog` — without it, nothing is logged (not even to console)
- **Resolvers**: `${now:...}`, `${hydra:job.name}`, `${hydra.job.num}` only
  work inside a running Hydra application
- **Runtime state**: `hydra.runtime.cwd`, output directory creation, sweep
  configuration — all require `@hydra.main` initialization
- **Validation**: MISSING-field checks and type validation happen at
  composition time inside `@hydra.main`, not in bare `compose()`

### Dict overrides via `main_fun()`

`main_fun()` supports passing a hierarchical dict as `overrides` that layers
**on top of** Hydra-composed defaults. This is used by calibration and other
entry points that need to inject config programmatically (not via CLI).

When `overrides` is provided:
1. `@hydra.main` composes defaults normally (ConfigStore + `sys.argv`)
2. A wrapper intercepts the composed `DictConfig`, strips struct via
   `to_container` → `create`, then `OmegaConf.merge` the overrides on top
3. The merged config is passed to the task function

When `overrides` is `None` (default): standard `@hydra.main` path — composes
and dispatches with no modification.

### Per-probe YAML loading via `cli.process_loading_yaml()`

Per-probe configs (`cfg_proc/run/*.yaml` with `@package _global_`) are loaded
by `cli.process_loading_yaml()` — a shared loop extracted from both
`processing.run()` and the calibration pipeline. For each YAML:

```python
cfg_dc = OmegaConf.load(yaml_path)      # load per-probe YAML
cfg_dc = OmegaConf.merge(base_cfg, cfg_dc)  # merge on top of composed cfg
process_fun(cfg_dc)                      # call the processing function
```

This works because `base_cfg` from `@hydra.main` has **all** groups
(input, out, filter, program) fully resolved — no MISSING sentinels —
so the merge preserves defaults without manual fixups.

**Duplicate handling**: when multiple YAMLs resolve to the same pcid
(e.g. `@i_01.yaml` and `260613_1200@i_01.yaml`), `get_existed_cfgs` returns
`{pcid: [stem1, stem2]}` and `process_loading_yaml` iterates **all** stems.
Both `run_processing` calls write to the same output files — no data
duplication occurs because NC incremental append skips overlapping time
ranges (see [Re-run behavior](#re-run-behavior)). Coefs are written twice
(last YAML wins). Combined output deduplicates via `dict.fromkeys()`.

**Stem validation**: before calling `process_fun`, the YAML stem (after
stripping the last `@`) is compared with the `input.path` file stem (after
stripping `@`). If they differ, the YAML is skipped — this catches
manually-copied configs (e.g. `@i_01_backup.yaml`) whose stem no longer
matches the data file they reference.

### `call_in_raw_dir()` — entry point for non-processing pipelines

`call_in_raw_dir(fun, yaml_path=None, **kwargs)` bootstraps the Hydra
runtime for any entry point (calibration, etc.):

1. Resolves `data_dir` from `input.path` (in kwargs or `sys.argv`)
2. `os.chdir(data_dir)` + injects `--config-dir <data_dir>/cfg_proc`
   (if exists) — targets Hydra's argparse layer (natively handles special chars)
3. Collects non-`main_fun` kwargs as override dicts
4. Optionally loads a per-probe YAML via `yaml_path=` (merged as base;
   explicit kwargs win on top)
5. Calls `main_fun(fun, overrides=...)`

```python
cli.call_in_raw_dir(
    run.main, config_name="config",
    yaml_path=path_db_raw.parent / "cfg_proc" / "run" / "230811_1622@i_p5-маг.yaml",
    input={"path": path_db_raw, "tables": tables_raw, "channels": ["M", "A"]},
    out={"db_paths": [db_in]},
)
```

The `yaml_path` param loads the YAML via `OmegaConf.load` and uses it as
the **base** for dict overrides — explicit `**kwargs` win on top via
`OmegaConf.merge`.

## Data structures

### pcid (Probe output Column ID)

Canonical probe identifier: `i{N:02d}`, `i_{model}{N:02d}`, or `w{N:02d}`.

| Example | Constructed by (`tcm/format.py`) |
|---------|---------------------------------|
| `i01` | `pcid_from_parts(type="i", model="", number=1)` |
| `i_p02` | `pcid_from_parts(type="i", model="p", number=2)` |
| `w01` | `pcid_from_parts(type="", model="w", number=1)` |

The inverse `pcid_to_raw_name()` maps `i01` → `incl01` (for HDF5/NC group names).

## Input routing

`processing._load_single(cfg, pcid)` auto-detects format from `cfg.input.path` suffix:

| Suffix | Handler (`tcm/`) | Coefs source |
|--------|------------------|-------------|
| `.csv`, `.txt`, `.dat` | `_xr.dataset.open_csv_chunks()` | separate `coefs_path` file |
| `.h5`, `.hdf5` | `_xr.io.open_hdf5()` → `(ds, coefs)` | `/{tbl}/coef/` group in same file |
| `.nc`, `.nc4` | `_xr.dataset.open_nc()` → `(ds, coefs)` | `/{tbl}/coef/` group in same file |

Returns `(ds_raw, coefs_from_file)`. CSV yields `None` coefs; HDF5/NC extract
coefs from the file's `/{tbl}/coef/` group (see [Coefficient NC storage](#coefficient-nc-storage)).

File-level coefs are merged into config-sourced coefs in `run_processing()`
with config overrides winning on conflict.

## Discovery

Discovery is performed by `config_yaml.gen_metadata()` which calls `csv_load.search_csv_files()` internally.

### Path pattern classification

`_pattern_to_regex(name)` in `csv_load.py` classifies `input.path` into glob or regex.
The decision table for pattern interpretation is in
`config_reference.md` (§Pattern interpretation). Implementation:

1. Tries `re.compile(name)` — on failure → glob via `_glob_to_regex()`
2. On success, checks if the last dot before extension is escaped (`\.`)
3. If not escaped → still glob. The "extension dot" is `name.rfind('.')`;
   if `name[pos-1] != '\\'` → glob

Directory shortcut: `path_in.is_dir()` → `_DIR_DEFAULT_REGEX` (`i.*\.txt`,
case-insensitive). `search_csv_files` strips the `@` prefix before matching,
so `@?-prefixed` and unprefixed patterns find corrected files identically.

### `search_csv_files(path_in)` → `{(model, number): [paths]}`

Scans `path_in.parent` (or `path_in` itself if directory) for files matching
the pattern. Groups by `(model, number)` identity extracted via
`format.probe_from_name()`. When both `@`-prefixed (corrected) and raw versions
exist for the same identity, only the corrected version is returned.

### File pairing (corrected/raw)

Handled internally by `csv_load.search_csv_files()`. When both `@`-prefixed
(corrected) and raw versions exist for the same probe identity, only the
corrected version is returned.  Matching uses canonical stem via
`format_loaded.mod_name()` — strips `@` prefix and normalises the name
(e.g. `INKL_P05_0_v_trube` → `i_p5-0_v_trube`).

`correct_raw_files()` is called downstream on the already-filtered list:
`@`-prefixed files pass through unchanged; raw files get corrected and
saved as `@`-prefixed copies.

### Multi-table discovery for HDF5/NC

When `cfg.input.tables` contains a glob pattern (e.g. `["incl*"]`),
`_discover_tables()` in `config_yaml.py` scans the HDF5/NC file for
top-level groups matching the pattern — `pd.HDFStore.keys()` for HDF5,
`h5py.File.keys()` for NC — expanding the glob to concrete table names.

## Config system

### Type conversion via `main_init()`

After Hydra composition, **every** entry point calls `cli.main_init(cfg)` which
converts the `DictConfig` to a plain `dict` with resolved types.  This is the
**single conversion point** — downstream code never needs to re-convert.

The conversion chain: `main_init` → `ini2dict` → `type_fix` (per key).

Key name-driven conversions by `type_fix` (`utils2init.py`):

| Prefix/Suffix | Converts to | Name change | Example |
|---|---|---|---|
| `dt_*` (prefix) | `timedelta` | suffix stripped if unit name, else kept | `dt_hole_warning=600` → `timedelta(600s)` |
| `*_path` / `path_*` | `Path` | kept | `path="/raw/i.txt"` → `Path("/raw/i.txt")` |
| `*_date` / `*_time` | `datetime` | suffix stripped | `min_date="2024-01-01"` → `datetime(2024,1,1)` |
| `*_int` / `*_integer` | `int` | suffix stripped | `count_int="5"` → `5` |
| `*_float` | `float` | suffix stripped | `ratio_float="1.5"` → `1.5` |
| `*_list` / `*_names` | `list` | suffix stripped, comma-split | `ids_list="a,b"` → `["a","b"]` |
| `*_dict` | `dict` | suffix stripped, colon-split | `cfg_dict="k:v"` → `{"k":"v"}` |
| `min_*` / `max_*` | `float` (catch-all) | kept | `min_Mx="0.1"` → `0.1` |

**Important `dt_*` detail**: ALL `dt_*`-prefixed keys become `timedelta`, even
when the suffix is not a recognised duration unit (e.g. `dt_hole_warning`,
`dt_bins`).  The default unit is `seconds`.  Consumers must handle `timedelta`
values — use `val.total_seconds()` to extract numeric seconds.

After `ini2dict`, `main_init` also:
- Expands `M` shorthand → `Mx/My/Mz` in min/max dicts (`sugar_expand_m`)
- Merges `min_date`/`max_date` into `time_ranges` (`sugar_condense_lim_date`)
- Resolves output paths via `PathLayout` and copies them to the returned dict

**Downstream rule**: after `main_init`, `cfg` is a plain `dict` — use `[]` /
`.get()` access, never OmegaConf attribute access.

### Hydra search path

`main()` adds `--config-dir <data_dir>/cfg_proc` (if exists) via Hydra's
argparse layer (bypasses ANTLR) so that per-project run YAMLs in `cfg_proc/run/`
are discoverable by Hydra.

- **Bundled config** — `tcm/cfg/cfg_proc/` package resolves `config.yaml` and
  structured config groups (input/out/filter/program) from the package
- **`{data_dir}/cfg_proc`** — per-project `run/*.yaml` on disk (searchpath override)

### Config generation via `gen_metadata()`

`config_yaml.gen_metadata(cfg, input_paths)`:
1. Locate CSV files for each input path and merges results. Logs "Discovered N
   file groups" summary. Raises `FileNotFoundError` when no files match
2. Loads edge rows (first/last) via `csv_load.load_from_csv_gen(return_="first_last_row")`
3. Loads coefficients via `_xr.coefs.prep_cfg_for_probe()`
4. Sets `cfg1["input"]["time_ranges"]` from edge-row timestamps (under `input`
   to survive `to_omegaconf_merge_compatible(cfg1, Config)` filtering)
5. Yields `(cfg1, (False, pcid, None))` per source file

`save_config_to_yaml()` iterates `gen_metadata()` and writes each to
`cfg_proc/run/{source_stem}.yaml` with `# @package _global_` header.

**Deduplication**: before writing, scans existing YAMLs for the same
normalized pcid (via `stem_to_pcid` → `probe_from_name` → `pcid_from_parts`).
If any existing YAML already references a valid `input.path` file for the
same pcid, the new config is skipped.  This prevents duplicate configs when
the same probe has differently-formatted names (e.g. `i_090` vs `i90`,
different comments after `-`).

Edge-row data (≤ 4 time values) is too short for frequency estimation.
`time_corr()` detects this and skips `_correct_time()` entirely, avoiding
misleading "freq unknown → defaulting to 1Hz" warnings.

#### Linking run YAMLs to raw input files

Each run YAML is linked to a corrected raw input file through **two mechanisms**:

1. **YAML filename stem**: `save_config_to_yaml()` names the YAML as
   `{datestamp}@{corrected_stem}.yaml` — the stem after `@` matches the
   corrected input file stem.  The `datestamp` prefix metadata from
   `input.time_ranges[0]` is not used in processing, it is for user only.

2. **`input.path` inside the YAML**: `prep_cfg_for_probe()` writes the
   absolute corrected-file path into `cfg1["input"]["path"]`.
   `find_stale_cfgs()` validates this path still exists.

| Run YAML stem | Corrected input file | Raw input file | pcid |
|---|---|---|---|
| `260624_1255@i_p5-press` | `@i_p5-press.TXT` | `i_p05-press.TXT` | `i_p05` |
| `230811_1622@i_p5-маг` | `@i_p5-маг.TXT` | `i_p05-маг.TXT` | `i_p05` |
| `260625_1708@i_p5-0_v_trube` | `@i_p5-0_v_trube.TXT` | `i_p05_0_v_trube.TXT` | `i_p05` |
| `@i_p01-001` | `@i_p01-001.TXT` | `I_P01_001.txt` | `i_p01` |

**Bridging logic** (all in `format.py`):
- `stem_to_pcid(stem)` isolates the probe identity stripping `{prefix}@` and then `-{comment}` suffix.
- `probe_from_name(pcid_stem)` → `parse_name()` regex extracts `(model, number)`.

**Stem validation** (`processing.run()`): before processing, the YAML stem
(after stripping `{datestamp}@`) is compared with the `input.path` file stem
(after stripping `@`).  If they differ, the YAML is skipped with a warning.
This prevents manually-copied or renamed configs (e.g. `@i_p1 — копия.yaml`)
from being silently used as valid configs — even though `probe_from_name()`
would resolve the correct pcid, the raw stem mismatch catches the discrepancy.

### Device-metadata `time_ranges` sync

After config generation, `processing.run()` calls
`config_yaml.sync_yamls_devmeta_and_hydra(dev_dir, dir_cfgs, cfgs_existed)`
to push `time_ranges` from `info_devices.yaml/.json` into any run YAML that
lacks them. Idempotent: configs with existing `input.time_ranges` are skipped.
Missing metadata file is handled EAFP (logged at DEBUG, returns cleanly).

The sync function logs at INFO per probe:
```
  i90: info_devices [2025-12-04T18:00:19, 2026-03-12T22:11:03]
    written to 251204_1800@i_090.yaml          # written from metadata
```

When all configs already have time_ranges matching metadata, only the
`info_devices` line appears (per-stem "already configured" → DEBUG).
When a config's existing range is **broader** than metadata, a WARNING is
emitted showing the broader range — this signals a mismatch that may need
manual review.

Uses `tcm.metadata` — local extraction of `get_path_in_parents`,
`load_file_meta`, `extract_devices_info` from `veusz_helpers.common.metadata`.
The original module's `func_vsz` dependency (Veusz registry access at import
time) made it unusable in the frozen distribution.

### `get_existed_cfgs(dir_cfgs)` → `{pcid: [stems, …]}`

Defined in `tcm/config_yaml.py`. Globs `*.yaml` in `dir_cfgs`, derives pcid
from stem via `format.probe_from_name`, keeps all stems per pcid.

### `find_stale_cfgs(cfgs_existed, dir_cfgs)` → stale pcid set

Defined in `tcm/config_yaml.py`. Opens each YAML, reads `input.path`. If the
referenced file doesn't exist → pcid is stale. No `ProbeFiles` dependency —
checks YAML content directly. Per-stem details logged at DEBUG; the caller
(`processing.run`) logs the summary set at WARNING level. After regeneration,
`processing.run` re-checks and only warns about **still-stale** pcids that
are in `input.ids` — orphans not in `input.ids` are reported as skipped.

**Stale configs are never deleted** — only warned about.  The user decides
how to handle them (manual cleanup, re-pointing `input.path`, etc.).

### `has_run_yamls(dir_run)` → bool

Defined in `tcm/config_yaml.py`. Returns True when `cfg_proc/run/` contains
at least one YAML.

## Processing

### `processing.run(cfg)` — canonical orchestrator

Accepts a Hydra-composed `DictConfig` (from `@hydra.main`). All groups
(input, out, filter, program) are fully resolved — no MISSING sentinels.

**Binary inputs (NC/HDF5)**: if ``input.path`` suffix is non-text (``.nc``, ``.h5``, …),
config discovery/generation is **skipped** — ``run()`` calls ``run_processing()``
directly, once per ``input.tables`` entry (if present), or once inferring pcid from
path stem.  These files carry their own coefs (``/{tbl}/coef/`` group) and are not
part of the per-text-file config sweep.

**Text inputs (CSV/TXT)**: the full discovery pipeline runs as below.

1. **Config generation** (idempotent): ``config_yaml.save_config_to_yaml(cfg, ...)``
   is called when:
   - Stale configs exist (source file deleted), OR
   - No configs exist, OR
   - New source files are found that have no config yet — detected via
     ``csv_load.search_csv_files()`` for **both** wildcard mode and specific
     ``input.ids`` (when requested IDs lack configs but source data may exist).
   After generation, orphan configs (no matching source file) produce a
   **warning only** — configs are never auto-deleted.
2. **Device-metadata sync**: `sync_yamls_devmeta_and_hydra(dir_raw.parent, dir_cfgs, cfgs_existed)`
   pushes `time_ranges` from `info_devices.yaml` into run YAMLs that lack them.
3. **Resolve which configs to process** (by `input.ids` or all).
   When specific IDs are requested and some have no config after discovery
   (source data truly absent), ``ValueError`` is raised — the error is now
   justified because config generation was already attempted in step 1.
4. `cli.process_loading_yaml(run_processing, base_cfg=cfg, dir_cfgs=..., cfgs=..., n_cfgs_existed=...)`
   — loads each YAML, merges on top of `cfg`, validates stem match, calls `run_processing(cfg_dc)`.
   Returns `(processed_pcids, failed_pcids, last_cfg)`.
5. Combined output: `_combine_probes()` merges per-probe groups with `probe`
   dim (if >1 probe)
6. Terminal log: `"Done — {n_ok} probes: {pcids} ok | {n_skipped} skipped ({pcids}) | {n_failed} failed ({pcids})"`
   Counting is by **distinct probe** (not by YAML attempts): a probe that fails
   on one YAML but succeeds on another is considered successful.

### `run_processing(cfg)` — single-file processing

1. Derive probe identity (pcid) and table name (tbl):
   - **Binary input** (NC/HDF5 suffix): pcid ← ``input.tables[0]`` via
     ``format.to_pcid_from_name()``, tbl ← ``tables[0]`` as-is.
     ``run()`` pins ``tables=[single_tbl]`` per call when iterating.
   - **Text input** (CSV/TXT): pcid ← path stem via ``format.stem_to_pcid()``
     → ``format.to_pcid_from_name()``, tbl ← ``pcid_to_raw_name(pcid)``.
2. Resolve output paths via `paths.PathLayout.from_cfg()` + `layout.apply_to_cfg(cfg.out)`
3. **Phase 1 — Load coefs**: `get_coefs_from_cfg()` + merge file coefs + HDF5 auto-migrate
   (extract coefs from legacy `.raw.h5` if `.raw.nc` absent)
4. **Phase 2 — Calc/update coefs**: `prepare_coefs()` — zeroing rotation from
   `time_ranges_zeroing`, azimuth correction from `time_ranges_azimuth` (data-driven
   tilt direction) and/or `azimuth_add`/`coordinates` (manual/declination)
5. **Phase 3 — Save coefs**: write changed coefs to NC file (NC source or raw_db_path)
   or run YAML (noh5). NC-source coefs overwrite in-place (bypasses data-skip guard).
   Changed coefs are **always** mirrored to the run YAML when `yaml_path` exists
   (not just in noh5 mode) — keeps the config readable.  Before first modification,
   `update_coefs_in_run_yaml` creates a timestamped backup
   (`-backupYYMMDD_HHMMSS.yaml`); subsequent updates reuse the same backup.
6. **Phase 4 — Save data**: append raw data to `*.raw.nc` via `nc_incremental_update`
   (skip for NC sources). Runs after Phase 3 because coefs may be in the same NC file.
7. Process + persist via `_process_and_persist()` — passes `coef_zeroing_matrix`
   to `_xr.physical.process()`

### Phase-stopping (`program.return_`)

Controls how far the pipeline runs before stopping. See `config_reference.md`
(§Phase-stopping) for the complete decision table with return values.

`<cfg_from_args>` causes `main_init` to return the raw `DictConfig` before
type conversion — `run_processing` receives an unconverted config, so no data
is loaded and no output is written. `process_loading_yaml` still dispatches
to `run_processing` for each config (the early exit is inside `main_init`,
not before dispatch). Existing user-edited configs are **not** overwritten —
`save_config_to_yaml` is only called when configs are stale, missing, or new
source files appear (see `processing.run()` lines 174–196).

`<saved_coefs>` stops `run_processing` after Phase 3 (coef persistence to
YAML with backup + NC), before Phase 4 (raw NC save) and data processing.
`<saved_raw>` stops `run_processing` after Phase 4 (coef persistence + raw
NC save). `<saved_noavg>` and `<saved_all>` stop after the corresponding
NC write in `_process_and_persist`.

### Time correction

The inclinometer firmware timestamps at integer-second resolution via
`Y,M,D,H,M,S` columns. At 10 Hz sampling, 10 consecutive rows share the
same second (e.g. `2000,1,1,0,14,12`). `time_corr()` in `tcm/utils_time_corr.py`
(called inside `csv_load.load_from_csv_gen`) corrects this.

The correction pipeline (`_correct_time`):
1. Subset to in-range data — out-of-range data never enters HWM or bilateral
2. `_trim_overlong_runs` — prevents parking-scan drift for near-integer freq
3. `_bilateral_check` — isolated spikes (O(n) per iteration)
4. `_hwm_check` — sustained backward sections (O(n))
5. `_find_hole_edges` — segment boundaries on clean subset
6. `_snap_to_grid` — `g(k) = origin + k·dt_step` per segment

Three modes are available via `filter.corr_time_mode` — see `config_reference.md`
(§Time correction) for the mode decision table, config fields, and examples.

### Filter stages (load vs process)

Filtering is split into two distinct stages by namespace (see `config_reference.md` §Stage classification):

1. **Load-stage** (`input.min`/`input.max` + `input.time_ranges`) — rows **dropped**. Applied in `_xr/io.py::load_raw()` for **all** sources (NC/HDF5: `apply_load_time_ranges` + `filter_global_minmax`; CSV: `filter_global_minmax` only, time_ranges applied by `time_corr`).

2. **Process-stage** (`filter.min`/`filter.max`) — values → NaN, rows **preserved**. Applied in `_xr/physical.py::process()` via `filter_local()`. `bad_p_at_bursts_starts_period` NaN-out in `calc_pressure()`. `g_minus_1`/`h_minus_1` NaN-out in `calc_velocity()`.

### Filter expansion

`cfg.input.min`/`max` (load-stage DROP) and `cfg.filter.min`/`max` (process-stage NaN-out)
both support `M` as a shorthand for `Mx`, `My`, `Mz`. Expansion runs at compose time via
`_xr/filters.expand_m_shorthand()`. The expansion logic and defaults are in `config_reference.md`
(§Filter expansion).

### Processing pipeline stages

`_xr/physical.py::process()` (line 194) applies the following stages in order:

1. **filter_local** — NaN-out on raw columns where `cfg_filter.min`/`max` thresholds exceeded
2. **calc_velocity** — calibration (`fG`/`fInclination`), `g_minus_1` NaN-out on computed `GsumMinus1`, `v_abs_from_incl`, `h_minus_1` NaN-out on computed `HsumMinus1`, `polar2dekart`
3. **calc_pressure** — `polyval2d` + `bad_p_at_bursts_starts_period` (first-2-per-burst NaN-out)
4. **binning** — `resample(time=dt_bin).mean()` with NaN threshold on valid-sample count.  When data is large (≥ 100 K rows), a persistent `TqdmCallback` is registered in `process()` and passed to each `binning()` call; the dataset is chunked along `time` (1 M rows) and `.compute()` materialises the dask graph so task-level progress is shown in a single bar shared across all bins (`_RESAMPLE_CHUNK_N`, `tqdm_cb` parameter).

Coefficient application order (inside `calc_velocity`):
`prepare_coefs` (zeroing rotation) → `fG(Ag,Cg)` → `fInclination` → `v_abs_from_incl(kVabs, calc_version)` → `azimuth_shift_deg` → `polar2dekart`

### Column order

Output columns are ordered to match legacy convention — see `config_reference.md`
(§Column order) for the exact ordering specification.

### Text type → column layout

`csv_load.format_parts_select_raw(file_path)` auto-detects columns from the
file header. `format_parts_select(text_type)` is the fallback. The column
variants per `text_type` are documented in `config_reference.md`
(§Text type → column layout).

### Output persistence

`_process_and_persist()` writes each bin result to shared NC files using
**per-probe groups** (not per-probe files). PathLayout resolves the paths
at the top of `run_processing()`.  When the result dataset is still
dask-backed (e.g. the no-avg output), a per-bin `TqdmCallback` wraps the
NC write so `to_netcdf` triggers `.compute()` with task-level progress
(`processing.py` — `_is_dask` / `_nc_ctx` guard).

| dt_bin | Target file | Group | Example |
|--------|-------------|-------|---------|
| `0` (no-avg) | `not_joined_db_path` (`*.proc_noAvg.nc`) | `/{pcid}/` | `/i01/` |
| `>0` (binned) | `db_path` (`*.proc.nc`) | `/{pcid}bin{bin_s}s/` | `/i01bin600s/` |

Fallback: when PathLayout fails (no `_raw/` ancestor), output goes to
`cfg.out.dir / @{pcid}.nc` — per-probe files only as last resort.

**CSV/TSV export**: each binned result (`dt_bin >= dt_bins_min_save_text`) is
exported to `{text_path}/{timestamp}{suffix}@{pcid}.tsv`.  Dev default:
threshold ≥ 1 s → no-avg (dt_bin=0) skipped.  **noh5 dist** default:
`dt_bins_min_save_text=0` → no-avg TSV enabled alongside 1 h bin.

**Logging directory**: configured in `cfg_proc/config.yaml` via
`hydra.run.dir: ./cfg_proc/log/${now:...}`. Since `os.chdir(data_dir)` is called
before `@hydra.main`, logs go to `{data_dir}/cfg_proc/log/{timestamp}/`.

Per-probe groups in `*.proc.nc`: each bin result is stored as a NetCDF4 group
(`/{pcid}/` for no-avg, `/{pcid}bin{bin_s}s/` for binned) within a shared file.
`merge_probes()` concatenates along a `probe` dimension:

```python
ds_combined = xr.concat(per_probe_datasets, dim="probe").assign_coords(probe=pcids)
```

### Batch mode

A run YAML may contain a top-level `files` list. When present,
`run_processing()` iterates `cfg.files`, loads each, concatenates, and processes
once. Implemented in `_run_batch()` (`tcm/processing.py`).

Multi-file concatenation trusts input file order — data flows through in the
order files are discovered (or listed in `cfg.files`). The NC incremental append
layer handles overlap/position logic downstream.

### Combined multi-probe output

When multiple probes are processed in one run, `_combine_probes()` in
`processing.py` merges per-probe groups into combined groups with a
`probe` dimension.  Only **distinct** pcids are combined — multiple
stems (source files) for the same pcid are deduplicated via
`dict.fromkeys()` to avoid spurious probe duplicates in the combined output.
Skipped when `H5_AVAILABLE` is `False` (noh5 environment — no netCDF4 backend).

| Output | Group | Content |
|--------|-------|---------|
| `*.proc_noAvg.nc` | `/{probe_type}/` | All probes, no-avg, `probe` dim |
| `*.proc.nc` | `/{probe_type}_bin{N}s/` | All probes, binned, `probe` dim |
| TSV | `{ts}bin{N}s@{pcid1},{pcid2}.tsv` | Combined tab-separated text |

*`probe_type`* is the short probe prefix derived from the first pcid
(e.g. `"i"` for inclinometers, `"w"` for wave gauges).

Column order in combined TSV: `v_i01, u_i01, inclination_i01, v_i02, ...`
(axis=1 concatenation unless `b_all_to_one_col=True`).

### Re-run behavior

On re-processing the same input data, each NC output type handles idempotency differently:

| Output | Write function | Re-run behavior |
|--------|---------------|----------------|
| `*.raw.nc` | `nc_incremental_update` (log dedup) | **SKIP** — same fileName + mtime detected via log table |
| `*.proc.nc` (binned) | `store_processed_incremental` (time-range check) | **SKIP** — new time range ⊂ existing range |
| `*.proc_noAvg.nc` (no-avg) | `store_processed_incremental` (time-range check) | **SKIP** — new time range ⊂ existing range |
| Combined groups | `_combine_probes` → `store_processed(..., mode="a")` | **Overwrites** in-place — re-reads per-probe groups, re-concatenates, re-writes |

`store_processed_incremental` only checks time-range containment, not data content.
If filter params, coefficients, or input window change, the stored ``_run_params``
attribute is diff-compared to the current values — a WARNING with unified diff
is emitted, but the data is still skipped. Delete the `*.proc.nc`
file or pass `+force_reprocess=True` to force re-processing.

### Incremental update

`store_processed_incremental()` in `_xr/storage.py` checks if the target group
already has data covering the time range before writing. Skips if new data is
fully contained in existing range — avoids duplicates on re-run. See
`config_reference.md` (§Incremental append positions) for the position-aware
append strategy decision table and `config_reference.md` (§Log-based dedup)
for the log-based skip/resume/new-file decision table.

### `process_inmemory(ds, coefs, ...)` — standalone API

For callers/tests that already have a Dataset in hand. Defined in
`tcm/processing.py`. Pass-through to `_xr.physical.process` with optional
persistence.

## Raw NC storage

Raw data and coefficients are persisted in separate phases (see [run_processing](#run_processingcfg--single-file-processing)):

**Coefficient persistence (Phase 3)** — runs for ALL source types:
- **NC source**: changed coefs (from zeroing/azimuth) are written back to the
  source NC file directly.  Before writing, `ds_raw` is materialised into
  memory (`load()`) and the netCDF4 file handle released (`close()`) so h5py
  can open in append mode.  When coefs are unchanged, no memory is allocated
  and no file handle is touched.  `OSError` (e.g. file locked by another
  reader) is caught and logged as a warning — processing continues with the
  in-memory data.
- **CSV/HDF5 source + h5py**: all coefs written to `raw_db_path` NC on first
  creation; only changed coefs on re-run.
- **noh5 mode**: changed coefs written to the run YAML (`cfg_proc/run/*.yaml`)
  via `config_yaml.update_coefs_in_run_yaml()`.

**Data persistence (Phase 4)** — CSV/HDF5 sources only:
1. Checks `cfg.out.raw_db_path` — if absent, no raw NC storage
2. Releases xarray's netCDF4 read handle on the source Dataset (`ds_raw.load();
   ds_raw.close()`) before the h5py write — Windows HDF5 mandatory locking
   prohibits opening a file that already has a read-only handle.  Same pattern
   as the binary-source coef write guard (Phase 3).
3. Appends data via `nc_incremental_update()` (see [Incremental append (NC)](#incremental-append-nc))
4. Skips entirely for NC sources (data already in the file)

**HDF5 auto-migrate** (Phase 1b) — runs before `prepare_coefs`:
- If `*.raw.h5` exists but no `*.raw.nc`, extracts coefs from the HDF5 file
  via `incl_calc.coefs.load_coefs()` and merges them into the coefs dict
  before zeroing calculation. Migration is one-way — subsequent runs read
  from NC; the HDF5 file is preserved but no longer written to.

When coefs are written, the log distinguishes new from overwritten datasets:
```
Coefs saved to ...//incl01: 12 datasets (2 overwritten)
```

## Incremental append (NC)

`_xr/storage.py` — replaces the HDF5 `append_through_temp_db_gen` (temp file +
ptrepack). NC4 unlimited dimensions support native append — no temp file needed.

**Core principle: never re-sort**. Data is appended or prepended in the
order it arrives. If overlapping timestamps are detected, the new data
is **trimmed** (existing data always preserved) and a warning is logged.

### Position-aware append

`_time_range_overlap(new_min, new_max, ex_min, ex_max)` classifies where
new data lands relative to existing. The position classification and write
strategy are documented in `config_reference.md` (§Incremental append positions).

All write strategies are **O(1) in existing data size** — no full-dataset
read or `xr.concat` is ever used:

- **AFTER / OVERLAP_TAIL**: trim new data using existing time range (already
  in hand from h5py), then `_append_to_nc_group()` (h5py resize + write).
  No existing data read needed — O(new_data) memory.
- **BEFORE / OVERLAP_HEAD**: `_prepend_nc_group()` — h5py resize + chunkwise
  shift of existing rows right, then write new rows at index 0.
  O(chunk) memory, where chunk default is 50 000 rows.
- **Resize fallback**: `_rebuild_and_append()` — when h5py `resize()` fails
  (non-extendable datasets), builds a new group with `maxshape=(None,)`,
  copies old data chunk-wise from h5py, then appends new data.
  O(chunk) memory.

After any h5py resize (`_h5py_extend_group`, `_prepend_nc_group`),
dimension scales (`make_scale`/`attach_scale`) are re-attached so
`xr.open_dataset(engine="netcdf4")` can read the group back.
Before the combine step, `ensure_dim_scales()` repairs scales across all
groups in a file — defensive against files corrupted by older runs.
On Windows, HDF5 mandatory file locks can cause `OSError` when opening
a file whose netCDF4 handle was recently closed; `ensure_dim_scales`
catches both `RuntimeError` and `OSError` gracefully (warning + continue).

### Log-based dedup + resume

`check_file_vs_log(cur, existing_log)` in `_xr/storage.py` returns a 3-way
decision (SKIP, RESUME, NEW_FILE). The decision table is in
`config_reference.md` (§Log-based dedup).

**RESUME mode** (`_resume_append()`): when the same source file was updated
(newer mtime), only data after the existing last timestamp is appended.
The log is updated with two rows: original start and new tail end (both
marked with the new `fileChangeTime`).

**Overlap warning**: when new data overlaps existing data *and* the file
is a different source (not a resume), `append_to_nc()` logs a warning
and trims the overlapping portion of the new data — existing data is
never modified or removed.

### nc_incremental_update flow

`nc_incremental_update(ds_new, nc_path, tbl, file_meta)`:
1. Read existing log via `read_nc_log()` → `xr.Dataset`
2. `check_file_vs_log(cur, log)`:
   - `SKIP` → return `False`
   - `RESUME` → `_resume_append()` (trim tail + `_append_to_nc_group` + update log)
3. `NEW_FILE` → `_append_positional()`:
   - Read existing time range via h5py
   - `_time_range_overlap()` → classify position
   - On overlap: **warn** + trim new data
   - `append_to_nc()` → optimal write strategy per position
4. Append log record (one row per source file)

### h5py-only file I/O

All file operations in `append_to_nc()` and `_write_dataset_to_nc_group()`
use **h5py exclusively**. Mixing xarray's `to_netcdf` (netCDF4 backend) with
h5py on the same file within one process causes stale HDF5 file-handle entries
in xarray's LRU file-manager cache — the netCDF4 library fails with
`OSError: [Errno -103] NetCDF: Can't write file` or `OSError: Unable to
synchronously open file (file is already open for read-only)` when reopening
after h5py.

**Rule**: never hold an xarray/netCDF4 handle on a file while h5py opens it
for writing.  This applies to:

| Pipeline | Strategy | Code |
|----------|----------|------|
| Processing (Phase 3) | When `changed_coefs` is non-empty: `ds_raw.load()` + `ds_raw.close()` to release the read-only netCDF4 handle, then `save_coefs_to_nc` opens with h5py in append mode.  When coefs unchanged: no memory overhead, no file handle touched.  `OSError` caught and logged as warning. | ``processing.py`` Phase 3 |
| Processing (Phase 4) | Before `nc_incremental_update`: `ds_raw.load()` + `ds_raw.close()` to release the read-only handle (same pattern as Phase 3).  On Windows, HDF5 mandatory locking blocks ANY open (even read) when another handle exists. | ``processing.py`` Phase 4 |
| Calibration (`run_calibration`) | Calls `ds.close()` before `h5copy_coef` writes to the **same** NC file | `calibration/run.py:241` |
| Storage (`append_to_nc`) | Uses h5py exclusively for writes; read via `autoclose=True` | `_xr/storage.py` |

The read step uses `xr.open_dataset(engine="netcdf4", autoclose=True)` to
release the netCDF4 handle immediately on context exit. The delete + write
steps use h5py only. Dimension scales are applied so `xr.open_dataset` can
read h5py-written groups back with proper dimension names.

### On-disk time encoding

All h5py time datasets (both data groups and log table) use CF-standard
encoding — defined once in `_xr/storage.py`:

```python
_EPOCH_NS: int = np.datetime64("1970-01-01", "ns").astype(np.int64)
_CF_TIME_UNITS = "seconds since " + str(...)
_CF_CALENDAR = "proleptic_gregorian"
```

| On-disk | In-memory | Encoder | Decoder |
|---------|-----------|---------|---------|
| `float64` seconds, attr `units="seconds since 1970-01-01"` | `datetime64[ns]` | `_dt_ns_to_cf()` | `_cf_to_dt_ns()` |

Datasets created with `maxshape=(None,)` along the time axis so
`_append_to_nc_group()` can `resize()` them for O(1) tail-appends.

_write_time_ds(grp, name, data)` writes the dataset + attributes in one call.
`_cf_to_dt_ns(raw, units)` is backward-compatible: it inspects dtype and handles
both CF `float64` seconds (parsing the epoch from the *units* string) and legacy
`int64` nanoseconds (from files written before this unification). This ensures
`store_processed_incremental`, `_read_nc_group_as_dataset`, `append_to_nc`, and
`read_nc_log` all decode identically.

#### Float64-seconds precision limit

CF-standard `float64 seconds since 1970-01-01` has **~100 ns** effective
resolution for current timestamps (~1.76 × 10⁹ s uses 10 integer digits,
leaving 5–6 for the fraction).  Two `datetime64[ns]` values that differ
by < 100 ns are distinct in memory but **collapse to the same float64**
on disk.

This matters when `_snap_to_grid` produces timestamps differing by a
few nanoseconds (interpolation jitter at segment boundaries).  The
values are unique at `datetime64[ns]` precision so all in-memory
assertions pass, but the netCDF round-trip merges them into duplicates.

`store_processed_incremental` applies an **O(n) monotonicity dedup**
before writing: it checks `np.diff(float64_seconds) > 0` and drops
non-monotone positions.  This is the write-boundary guard.

The constant `storage.DT_CF_NS` (200 ns) is the single source of truth
for the CF encoding resolution.  It is used by:

- `store_processed_incremental`: monotonicity dedup before NC write
- `utils_time_corr._resolve_freq`: caps estimated frequency to
  `NS / DT_CF_NS` so that `_snap_to_grid` never produces a grid
  finer than the encoding can represent

### On-disk data encoding (dtype, compression)

All NC write paths apply two transforms before persisting:

1. **`_downcast_float32(ds)`** — `float64` → `float32` for every data variable.
   Coordinates (e.g. `time`) are left unchanged.  Float32 is sufficient for
   all measured geophysical data and halves on-disk size.

2. **`_compression_encoding(ds)`** — returns per-variable encoding dicts with
   `zlib=True, complevel=9, shuffle=True, fletcher32=True, dtype="float32"`
   for every numeric data variable.  String/datetime variables are excluded.

   For h5py-written groups (`_write_dataset_to_nc_group`, `_rebuild_and_append`),
   the equivalent h5py parameters are `compression="gzip", compression_opts=9,
   shuffle=True, fletcher32=True`.

No `scale_factor` / `add_offset` is ever set — avoids implicit dtype change on
read (xarray defaults to the on-disk dtype, not the scaled dtype).

### Battery variable

`Battery` is only relevant in raw NC files (`*.raw.nc`).  All processed
outputs (`store_processed`, `store_processed_incremental`, `save_netcdf`,
TSV export) strip it via `_drop_battery(ds)` before writing.  The raw
write paths (`store_raw`, `_write_dataset_to_nc_group` for raw incremental)
preserve Battery unchanged.

### Log table

`/{tbl}/logFiles` group in the NC4 file. Same on-disk encoding as data groups
(CF-standard float64 seconds):

| Variable | On-disk | In-memory | Description |
|----------|---------|-----------|-------------|
| `Date0` (dim) | `float64` s | `datetime64[ns]` | start time of data chunk |
| `fileName` | `S255` | `str` | source file name |
| `fileChangeTime` | `float64` s | `datetime64[ns]` | source file mtime |
| `DateEnd` | `float64` s | `datetime64[ns]` | end time of data chunk |
| `DateProc` | `float64` s | `datetime64[ns]` | processing timestamp |

`read_nc_log(nc_path, tbl)` → `xr.Dataset` with dim `Date0`.
`write_nc_log(nc_path, tbl, log)` — idempotent overwrite of the log group.
`keep_recorded_nc(cur_meta, log)` — xr-native dedup check.

## Path resolution

`paths.PathLayout` — declarative, lazily-evaluated path resolver (Layer 2).
Uses `find_dir_raw` and `_infer_proc_dir` (Layer 1) for anchor detection.
Resolves output paths from structural anchors (`proc_dir`, `raw_dir`).

### NC SCHEMA

```python
SCHEMA = {
    "raw_db":        ("raw_dir",  ".raw.nc",       True),
    "db":            ("proc_dir", ".proc.nc",      True),
    "not_joined_db": ("proc_dir", ".proc_noAvg.nc", True),
    "text":          ("proc_dir", "",               False),
}
```

Resolution hierarchy (`resolve(entity_name)`):
1. **Absolute path** — used as-is
2. **Relative path** — resolved relative to `proc_dir`
3. **Auto-generation** — stem from `RAW_DIR_NAME`-anchored directory + SCHEMA suffix

`_constants.RAW_DIR_NAME` (`"_raw"`) is the single source of truth for the
raw-data directory name — all source and test code imports this constant
instead of hardcoding the string.

`select_input_db()` checks `.nc` first, falls back to `.h5` — enables
transparent migration from HDF5 to NC without config changes.

## CSV correction

`csv_load.correct_raw_files()` → `(corrected_paths, params)`:

- Skips files already starting with `@`
- Raw files → `csv_specific_proc.correct_txt()`:
  - Determines column regex from `text_type` or file header
  - Preserves last 3 header rows
  - Filters bad lines
  - Output: `@i_p5-0_v_trube.txt` format (`@{pcid}-{comment}.{ext}`)
- Returns `(paths, params)` where *params* contains `header`, `dtype`,
  `skiprows`, `text_line_regex` from the **first** successful detection.
  Empty dict when all files were `@`-prefixed (no correction needed).

Detection flow in `load_from_csv_gen`:
1. `correct_raw_files(paths_csv, text_type, user_regex)` → `(paths, params)`
2. If `params` is empty (all `@`-files): `config_text_params(text_type, paths_csv[0])` reads
   the header from the first corrected file as fallback.
3. **Per-file detection**: each file is loaded with its own `config_text_params` +
   `init_input_cols` call.  Different files in the same probe group may have
   different column counts (e.g. older 15-col vs newer 16-col pressure probes).

This ensures detection happens **exactly once per file**: either during correction
(raw files exist) or from the corrected file (all `@`-prefixed).  No per-text_type
caching — each file's column layout is resolved independently.

### Header-vs-data column count guard

`format_parts_select_raw()` compares the header column count against the first
data row's column count.  When the header has more columns (e.g. a previously
botched correction stripped trailing columns from data), the header is truncated
to match the data.  This prevents `ValueError` in `init_input_cols` and
`ParserError` in pandas `read_csv`.

## Coefficient loading

`processing.get_coefs_from_cfg()` builds a three-tier fallback chain:
1. `input.coefs` in YAML (highest priority)
2. `coefs_path` (HDF5 `calibration.h5` or YAML directory)
3. Sibling `cfg/coef/yaml_export/` directory — **always** appended as final fallback
   (silently used in noh5 / `dist/tcm_clc_txt` packaging where the `.h5` file was pruned)

The same chain is mirrored in `_xr/coefs.prep_cfg_for_probe()`.
HDF5 paths are gated on `H5_AVAILABLE` (`tcm._constants`) — when h5py is not
installed the h5 candidate is silently skipped (no message).  When h5py IS
installed but the file is missing, a WARNING is emitted (genuinely unexpected).
YAML loads log INFO only when loaded coefs contain **new** keys not already in
the user overrides; otherwise the load is logged at DEBUG.

`incl_calc/coefs.get_coefs()`:
- Merges from all sources; override keys from config win
- Converts lists to `np.float64`
- Computes `date` as max of all component dates
- Dispatches on file suffix: directory → YAML, `.yaml`/`.yml` → direct,
  `.nc` → `load_coefs_from_nc()` (lazy import), else → `h5py`
- **`P_t` supersedes `P`/`PBattery`/`PTemp`**: for pressure probes (`*p*`) the 2-D
  polynomial `P_t` replaces the legacy scalar triples. When `P_t` is defined (loaded
  or overridden), the missing `P`/`PBattery`/`PTemp` defaults are silently ignored —
  no warning about "not redefined from current run config".

### Coefficient NC storage

Coefs are stored in `/{tbl}/coef/` groups within `*.raw.nc` files:

```
/{tbl}/coef/G/A   (3×3 float64)        — accelerometer gain
/{tbl}/coef/G/C   (3 float64)           — accelerometer offset
/{tbl}/coef/H/A   (3×3 float64)        — magnetometer gain
/{tbl}/coef/H/C   (3 float64)           — magnetometer offset
/{tbl}/coef/Vabs0 (6 float64)           — pressure calibration polynomial
/{tbl}/coef/H/azimuth_shift_deg (scalar)— azimuth correction
/{tbl}/coef/P_t   (3×3 float64)         — rotation matrix (optional)
/{tbl}/coef/i     (scalar int)          — probe serial number
/{tbl}/coef/date  (string attr)         — calibration date
```

Write: `save_coefs_to_nc(nc_path, tbl, coefs, pcid=)` in `_xr/coefs.py` —
converts raw coefs dict to flat `{h5_path: value}` via `_coefs_to_h5_dict()`,
then delegates the h5py write to `h5inclinometer_coef.h5copy_coef()`
(handles str/bool→dtype, shape mismatch→delete+recreate, NaN masking,
`timestamp` attrs, `True`→ISO-date in `dates`).
Read: `load_coefs_from_nc(nc_path, tbl)` — traverses
`/{tbl}/coef/` group hierarchy and reconstructs the coefs dict. Both are
idempotent (overwrite in-place).

Calibration writes coefs via legacy `h5inclinometer_coef.h5copy_coef()`
which uses the same h5py approach. When the target NC file is already open
by xarray (e.g. loaded via `load_raw`), call `ds.close()` before writing —
see [h5py-only file I/O](#h5py-only-file-io).

### Coefs preparation (xr-native)

`_xr.coefs.prepare_coefs()`:
1. Accepts `coefs`, `ds_raw`, and keyword args `time_ranges_zeroing`,
   `time_ranges_azimuth`, `azimuth_add`, `coordinates`, `data_date`
2. Azimuth correction via `get_coef_azimuth_shift()` (if `azimuth_shift_deg` in coefs)
3. If `time_ranges_zeroing`: tilt zeroing → `coef_zeroing_rotation_from_data()` → `Rz`
4. If `time_ranges_azimuth`: tilt direction azimuth → `coef_azimuth_from_data()` → `azimuth_shift_deg`
   (uses `orientation.azimuth_shift()` on calibrated unit vectors, no kVabs dependency)
5. `get_coef_zeroing_matrix(**coefs)` — computes rotation from `g0xyz` (if set)
   or returns existing `Rz` (if non-identity). `g0xyz` takes precedence.
6. Returns `(coefs_merged, coef_zeroing_matrix, dates, msg)`

`get_coef_zeroing_matrix(Rz, g0xyz, Ag, Cg)`:
- `g0xyz` set → `to_unit_vector(g0xyz.reshape(3,1), SensorCalibration(Cg, Ag))`
  then `rotate(zenith, [0,0,1])`. Returns `(R, "with new rotation to user defined zero point (g0xyz)")`.
- `g0xyz` unset, `Rz` non-identity → returns `(Rz, "")`.
- Both unset or `Rz` identity → returns `(None, "")`.

**Shape fix** (2026-07): `g0xyz` from YAML arrives as flat `(3,)` array but
`to_unit_vector` expects `(3, N)`. The fix reshapes to `(3, 1)` before the call.
Without this, numpy broadcasting silently produces `(3, 3)` and the subsequent
`np.cross` in `rotate()` fails with "incompatible dimensions".

No `dask.dataframe` dependency — uses `xr.sel` + numpy.

## File name parsing

`format.parse_name(name)` in `tcm/format.py` extracts probe identity parts.
Three regex steps (first match wins):

1. **Regular stems**: `[^iw]*(i|w)(nkl|ncl|_?)(b|d|p|)_?0*(\d{1,4})(?P<comment>.*)`
   — The first `i`/`w` is the **probe type** (not a prefix). `nkl`/`ncl` and
   `_` after the type are consumed but ignored (instrument-name suffix).
   Leading zeros before the number are consumed by `0*`. `comment` captures
   everything after the number.
2. **Glob patterns**: broader model capture with `chars3` for glob reconstruction.
3. **Fallback**: `voln_v*` → wave gauge.

`format_loaded.mod_name()` normalizes raw filenames to
**`@{pcid}-{comment}.{ext}`** format:
- Leading zeros in probe number stripped (`i_056` → `i_56`)
- Trailing suffix preserved as `-` delimited comment
  (`INKL_P05_0_v_trube` → `@i_p5-0_v_trube.TXT`)
- No comment → plain pcid (`i_01` → `@i_1.TXT`, `incl_b03` → `@i_b3.TXT`)

`format.stem_to_pcid_stem(stem)` strips `@` prefix and `-comment` suffix,
returning just the pcid stem (e.g. `@i_p5-0_v_trube` → `i_p5`).
Used by `get_existed_cfgs`, `processing.run`, and validation logic to
derive pcid from corrected filenames.

`probe_from_name()` wraps `parse_name()` → `(model_str, number_int)`.
The `comment` group is ignored — only structural identity fields matter.
Leading zeros in probe number are consumed by the regex (`_?0*` in `chars2`)
and `pcid_from_parts` re-pads to ≥2 digits, so `INKL_090` and `I_90` both
normalize to pcid `i90`.

**Normalization is significant for config deduplication**: `save_config_to_yaml`
checks if any existing YAML for the same normalized pcid already references a
valid `input.path` before writing.  This prevents duplicate configs when the
same probe has differently-formatted filenames (e.g. `i_090` vs `i90`,
different comment suffixes).