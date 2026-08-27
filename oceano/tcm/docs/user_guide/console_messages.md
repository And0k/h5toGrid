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
| `Config generation: regenerating N stale configs` | Source files changed; configs regenerated |
| `Discovered N file groups from {path}` | Source files found during discovery |
| `{pcid}: info_devices [{start}, {end}]` | Time range from metadata for this probe |
| `  written to {stem}.yaml` | Time range written from metadata to config |
| `  already configured but broader than metadata: {stem}.yaml [...]` | **Warning**: config has wider range than metadata |

## Time correction messages

At WARNING level only when anomalies exceed thresholds:

| Message pattern | Meaning |
|----------------|---------|
| `time correction: N/M monotone; X% removed; correction [min, max]s` | Clean correction — no action needed |
| `time correction: N/M monotone (in-range=K); X% removed (spikes=S, backward=B); ...; A pts > alarm Thr` | **Warning**: significant time anomalies — check diagnostics |
| `diagnostics {path} saved (N events): HOLE=..., ALARM=...` | Diagnostics NPZ saved for detailed analysis |

## Re-run behavior

On re-processing the same input data, each output type handles idempotency
differently:

| Output | Re-run behavior |
|--------|----------------|
| `*.raw.nc` | **SKIP** — same fileName + mtime detected via log table |
| `*.proc_Avg.nc` | **SKIP** — new time range ⊂ existing range |
| `*.proc_noAvg.nc` | **SKIP** — new time range ⊂ existing range |
| Combined groups | **Overwrite** — always rewrites from per-probe groups |

If processing parameters changed (coefs, filter thresholds), the pipeline
raises `ValueError` with a unified diff showing what changed. Pass
`out.overwrite_db=splice` to force reprocessing.

### `overwrite_db` decision matrix

| `overwrite_db` | Params changed? | `time_ranges` vs existing | Behavior |
|:---:|:---:|:---:|---|
| `None` | No | subset | **Skip NC** — export TSV only |
| `None` | No | extends | **Append** — append new tail only |
| `None` | Yes | extends | **Append + warn** — keep existing, append new |
| `None` | Yes | contained | **Error** — suggest `out.overwrite_db=splice` |
| `"splice"` | — | subset | **Splice** — keep outside, replace inside |
| `"splice"` | — | extends | **Splice** — keep outside, replace/append inside |
| `"splice"` | — | None | **Splice** — reprocess all from source |
| `"trim"` | — | subset | **Trim** — delete outside `time_ranges` |
| `"trim"` | — | extends | **Trim + append** — trim existing, process new |
| `"export"` | — | any | **Export only** — block NC writes, export TSV |

See [Config Tuning](../reference/config_tuning.md) for the full contracts.

## Common issues

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `Zeroing data -> no-op` | `time_ranges_zeroing` doesn't overlap with data | Check time ranges match your data interval |
| `ValueError` with param diff on re-run | Processing parameters changed since last run | Use `out.overwrite_db=splice` to reprocess |
| `Ex_nothing_done` exit | No matching configs or all stale | Check `input.path` pattern, verify source files exist |
| Config has wider range than metadata | YAML `time_ranges` wider than file metadata | Narrow `time_ranges` to actual deployment period |
| `FileNotFoundError` during processing | Source file deleted, stale config remains | Remove stale YAML from `cfg_proc/run/` |

## See also

- [CLI Reference](../reference/cli.md) — exhaustive argument spec, exit codes
- [CLI Guide](cli.md) — task-oriented CLI usage
- [Processing Guide](processing.md) — pipeline stages, re-run behavior details
