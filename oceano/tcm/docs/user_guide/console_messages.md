# Console Messages

Guide to exit codes, log levels, and the messages the pipeline produces during
processing. For the exhaustive CLI spec, see [CLI Reference](../reference/cli.md).

## Exit codes

| Code | Meaning |
|------|---------|
| `0` | Success — all requested probes processed |
| `Ex_nothing_done` | No probes were processed (no matching configs, all stale, or phase-stopped before data) |
| `1` | Unhandled exception |

## Pipeline summary messages

The pipeline logs a summary line at INFO on completion:

| Message | Meaning |
|---------|---------|
| `Done — N probes: i90, i67 OK` | All requested probes processed successfully |
| `Done — N probes: i90 OK \| 1 skipped (i64)` | Some probes skipped (not in `input.ids` or no source data) |
| `Done — N probes: i90 OK \| 1 failed (i67)` | Some probes failed — check errors above |

## Log levels

| Level | What it shows |
|-------|---------------|
| `DEBUG` | Per-stem checks, snap RMS, segment counts, coef load details |
| `INFO` | Pipeline progress — config discovery, coef loading, stage transitions, summary |
| `WARNING` | Time correction anomalies, stale configs, file-not-found, zeroing interval has no data |

Log file location: `cfg_proc/log/{timestamp}/processing.log` (inside the data
directory). When `program.return_` is non-default (e.g. `<cfg_from_args>`),
the file is named `processing-{sanitized}.log`.

Full log includes DEBUG-level detail. Set `program.verbose` to control the
console output level.

## Coefficient messages

One per probe at INFO level:

| Message pattern | Meaning |
|----------------|---------|
| `Coefs for i_p05: paths=[...], date=2023-08-13` | Coefficients loaded; `paths` shows the source chain |
| `Coefs prepared: with new rotation to direction averaged on configured time_ranges_zeroing interval` | Zeroing rotation computed from data interval |
| `Coefs prepared: with new rotation to user defined zero point (g0xyz)` | Zeroing rotation from `g0xyz` reference vector |
| `Coefs unchanged for i_p05 — skipping write` | No coef changes detected; nothing written to disk |
| `Overwrote coefs [Rz] in 260624.raw.nc` | Changed coefs written back to NC source file |
| `Updated coefs [Rz] in @i_p05.yaml` | Tilt zeroing rotation written to run YAML |
| `Coefs saved to ...//incl_p05: 12 datasets` | Full coefs written to raw NC file |
| `Zeroing data -> no-op: time_ranges_zeroing ... not in current data range` | **Warning**: zeroing interval has no data — check your time ranges |
| `Zeroing azimuth in interval ... (N points): azimuth shift=X°` | Azimuth computed from data interval |

## Config discovery messages

At INFO level:

| Message pattern | Meaning |
|----------------|---------|
| `Worker scan trigger={path}` | Scan entry — the original path before any anchor resolution |
| `Scan list for parent trigger={path} → N anchors` | Parent directory resolved to N `_raw` anchors (dropdown filled) |
| `Processing trigger={path} dir_raw={path}` | Single-anchor run entry (run handles one `_raw` only) |
| `Config generation: regenerating N stale configs` | Source files changed; configs regenerated |
| `Discovered N file groups from {path}` | Source files found during discovery |
| `{pcid}: info_devices [{start}, {end}]` | Time range from metadata for this probe |
| `  written to {stem}.yaml` | Time range written from metadata to config |
| `  already configured but broader than metadata: {stem}.yaml [...]` | **Warning**: config has wider range than metadata |
| `Ignoring N config(s) (whitespace in name): ...` | **Warning**: YAML name contains whitespace; kept on disk, not processed |
| `Skipping N config(s) — YAML stem ≠ input.path (manual copy?): ...` | **Warning**: config stem identity ≠ its `input.path` (renamed/manual copy); kept on disk, not processed |
| `Orphan configs (input.path points to not existing file): ... — ignored!` | **Warning**: stale config — source file missing; kept on disk, not processed (configs are never auto-deleted) |

## Time correction messages

At WARNING level only when anomalies exceed thresholds:

| Message pattern | Meaning |
|----------------|---------|
| `Burst for {pcid}: burst_dt={} bursts_t={}` | Burst gaps detected (`>max(10, 2·avg)`); stored to `info_devices.yaml`, not `input` |
| `Time extraction failed for {pcid} (both TCM and meta_finder)` | **Warning**: neither TCM edge rows nor meta_finder time info yielded a range |
| `time correction: N/M monotone; X% removed; correction [min, max]s` | Clean correction — no action needed |
| `time correction: N/M monotone (in-range=K); X% removed (spikes=S, backward=B); ...; A pts > alarm Thr` | **Warning**: significant time anomalies — check diagnostics |
| `diagnostics {path} saved (N events): HOLE=..., ALARM=...` | Diagnostics NPZ saved for detailed analysis |


## Common issues

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `Zeroing data -> no-op` | `time_ranges_zeroing` doesn't overlap with data | Check time ranges match your data interval |
| `ValueError` with param diff on re-run | Processing parameters changed since last run | Use `out.overwrite_db=splice` to reprocess |
| `Ex_nothing_done` exit | No matching configs or all stale | Check `input.path` pattern, verify source files exist |
| Config has wider range than metadata | YAML `time_ranges` wider than file metadata | Narrow `time_ranges` to actual deployment period |
| `FileNotFoundError` during processing | Source file deleted, stale config remains | Remove stale YAML from `cfg_proc/run/` |


## meta_finder log origins (tcm-visible)

| Message | Origin | Level |
|---|---|---|
| `Have read lines (max: …) from …` | `data_proc_funcs.read_file_lines_universal` | INFO (burst path) |
| `Skipped N/M bad lines, …` | `data_proc_funcs._extract_burst_info_from_lines` | INFO |
| `Time extraction is not successful from …` | `data_proc_funcs.extract_time_info_from_text_file` | WARNING — single reader, no fallback chain; tcm logs `Time extraction failed for … (unified reader)` per file |
| `Found N _raw anchors …` / `Anchor … has no files …` | `tcm/anchors.py`, `tcm/search.py` | INFO |


## See also

- [CLI Reference](../reference/cli.md) — exhaustive argument spec, exit codes
- [CLI Guide](cli.md) — task-oriented CLI usage
- [Processing Guide](processing.md) — pipeline stages, re-run behavior details
