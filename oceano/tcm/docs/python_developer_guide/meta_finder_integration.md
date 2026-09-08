# meta_finder Integration

Comprehensive map of every `meta_finder` capability reused by `tcm`.
User-facing summary: [Organizing TCM data and metadata per setup](../user_guide/meta_finder.md).

## Reused functions

| `meta_finder` function | Module | Purpose in `tcm` | Called from |
|---|---|---|---|
| `find_device_dirs(root)` | `file_finder.py` | Filtered 1-level device-dir enumeration (`ptn_dir_exclude` + `ptn_device_dir_search` + `is_valid_device_dir`) | `tcm/anchors.py`, `tcm/search.py` |
| `is_valid_device_dir(dir)` | `file_finder.py` | Validity gate: `_raw/`/`text_output/` subdir or `*_raw.zip`/`.7z` present | `tcm/anchors.py`, `tcm/search.py` |
| `find_raw_files_recursive(raw, ptn)` | `file_finder.py` | Per-anchor file enumeration incl. archives via `utils_sys.gen_from_archive` | `tcm/search.py` |
| `extract_time_info_from_text_file(dir, rel, averaging_interval, …)` | `data_proc_funcs.py` | Single time-range + burst source for `input.time_ranges` and burst overlay (no temp extraction, no csv edge loading) | `tcm/config_yaml.py`, `tcm/bursts.py` |
| `read_file_lines_universal(dir, rel, …)` | `data_proc_funcs.py` | Head/tail line sampling (lines 1 & 20 for interval estimate); optional `encoding`/`sep`/`skip_header` hints | via `extract_time_info_…` |
| `parse_datetime_from_row` | `data_proc_funcs.py` | Row timestamp parsing for edges, interval estimate, burst gaps (raw 6-col / ISO / `+HHMM` fix / serial fallback) | via `extract_time_info_…` |
| `_run_edge` / `_validated_tail_line` | `data_proc_funcs.py` | Timestamp-validated edge tolerance: top-50 window scanned in reverse, bottom-10 forward, contiguous-run semantics | via `extract_time_info_…` |
| `read_zip_member_head_tail` / `decode_bytes` | `utils_sys.py` | Single-pass ZIP member streaming without extraction; `utf-8-sig` → `cp1251` → `utf-8` decode order | via archive tail/head reads |
| `read_metadata_file` / `write_metadata_file` | `io_info_files.py` | `info_devices.yaml`/`.json` I/O | `tcm/bursts.py` |
| `_merge_device_metadata` | `create_info_files.py` | Non-destructive merge of burst autofill into existing metadata | `tcm/bursts.py` |
| `gen_from_archive` | `utils_sys.py` | Archive member listing without extraction | via `find_raw_files_recursive` |
| `config.extensions_*`, `ptn_*` | `config.py` | Extension sets (`.zip`/`.7z`, text, HDF5) and device patterns — imported, never copied | `tcm/search.py`, `tcm/anchors.py` |

Single sources: `tcm/anchors.py::collect_anchors` (discovery),
`tcm/search.py::search_csv_files_recursive` (enumeration),
`tcm/bursts.py` (GET on scan / WRITE on run). No unfiltered `rglob`
anywhere on the discovery path.

## Configuring search patterns

Device discovery patterns (`device_dir_pattern`, `ptn_device_dir_keywords`,
`ptn_device_dir_sep`) and extension sets (`extensions_archive`, `extensions_text`,
`extensions_hdf5`) are defined in
[meta_finder config.py](../../../meta_finder/src/meta_finder/config.py) and
imported by `tcm`. To customize which directories are recognized as device
directories or which file extensions are processed, see
[meta_finder config reference](../../../meta_finder/docs/reference/config_reference.md#device_dir_pattern)
and override via `meta_finder`'s CLI (`--device-dir-pattern`) or Hydra config.

## Data flow

```text
input.path (cruise root | device dir | _raw | file)
  → csv_load.search_csv_files (shallow iterdir, trigger-logged)
  → miss + dir ⇒ search_csv_files_recursive
      → find_device_dirs ⇒ per-_raw find_raw_files_recursive
      ⇒ {(model, number): [loose | archive-composite paths]}
  → config_yaml.gen_metadata: one extract_time_info call per file
      (loose or archive member) ⇒ input.time_ranges + burst_dt/bursts_t;
      no temp extraction, no csv edge loading on the metadata path
  → save_config_to_yaml → {yymmdd_hhmm}@pcid[-comment].yaml
  → sync_yamls_devmeta_and_hydra (info_devices time fill)
  → bursts.fill_missing_bursts on Run (indices 8–9) — persisted BEFORE
      data compute so a cancelled/failed run still saves autofilled bursts;
      failure is logged and never blocks processing (Scan stays read-only)
```

## Contracts reused verbatim

- Archive extensions `.zip`/`.7z` (`config.extensions_archive` ↔ `tcm._constants.ARCHIVE_EXTS`).
- Directory exclusion `.*-(?:\.|$)`, `^bad$`, `^test[^.]*$`.
- Device keywords `inclinometers?|incl|tcm|wave_?gau?ges?|pressure|pres|@i[0-9]?`
  plus device-id suffix grammars (complex `i3,5,9,w1-6`, ranges, parentheses).
- Metadata array layout `[point, sea_depth, h_above, symbol, lat, lon,
  time_st, time_en, burst_dt, bursts_t, comment, …]`; file priority
  `info_devices@meta_finder.yaml` > `info_devices.yaml` > `.json`.
- Full specs: [meta_finder I/O formats](../../../meta_finder/docs/reference/io_formats.md),
  [meta_finder config reference](../../../meta_finder/docs/reference/config_reference.md).

## Log origins (tcm-visible)

| Message | Origin | Level |
|---|---|---|
| `Have read lines (max: …) from …` | `data_proc_funcs.read_file_lines_universal` | INFO (burst path) |
| `Skipped N/M bad lines, …` | `data_proc_funcs._extract_burst_info_from_lines` | INFO |
| `Time extraction is not successful from …` | `data_proc_funcs.extract_time_info_from_text_file` | WARNING — single reader, no fallback chain; tcm logs `Time extraction failed for … (unified reader)` per file |
| `Found N _raw anchors …` / `Anchor … has no files …` | `tcm/anchors.py`, `tcm/search.py` | INFO |

`tcm`-side rows: [Console Messages](../user_guide/console_messages.md).

## Implementation sources

- [`anchors.py`](../../src/tcm/anchors.py), [`search.py`](../../src/tcm/search.py),
  [`bursts.py`](../../src/tcm/bursts.py),
  [`config_yaml.py` (archive branch)](../../src/tcm/config_yaml.py),
  [`csv_load.py::search_csv_files`](../../src/tcm/csv_load.py)
- [`file_finder.py`](../../../meta_finder/src/meta_finder/file_finder.py),
  [`data_proc_funcs.py`](../../../meta_finder/src/meta_finder/data_proc_funcs.py),
  [`config.py`](../../../meta_finder/src/meta_finder/config.py)
