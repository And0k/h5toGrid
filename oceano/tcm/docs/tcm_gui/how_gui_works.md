# How the GUI works — Internal Architecture

Optional Tkinter frontend wrapping `tcm.cli.call_in_raw_dir` in a background
thread.  No custom CLI parsing — Hydra handles all config keys natively via
`sys.argv` (see [CLI usage](../tcm_clc/README.md#quick-start)).

## Architecture

| File | Purpose |
|------|---------|
| `app.py` | Tk root, layout §1–5, 300 ms polling, argv prefill |
| `worker.py` | Background thread: `call_in_raw_dir` for Scan and Run |
| `coef_sheet.py` | tksheet treeview for per-config coefficient editing |
| `progress_bridge.py` | `GuiTqdm` (tqdm replacement) + module-level runtime injection |
| `log_bridge.py` | `QueueHandler` (consecutive dedup) → `ScrolledText` drain |
| `runtime.py` | Shared state: queues, `ProgressState`, `PauseGate` |

## Data flow

### Scan

```
Browse / Enter input.path
  → app._clear_log (flush queue + clear ScrolledText)
  → worker._scan (thread)
    → call_in_raw_dir(processing.run, return_="<cfg_from_args>")
      → processing.run: discovery → gen_metadata → process_loading_yaml
        → run_processing: main_init → return DictConfig (early exit)
        → collected [(stem, yaml_path, DictConfig)]
      → return (processed_pcids, failed_pcids, last_cfg, collected)
    → result_queue.put(("scan_ok", result))
  → app._poll_results → _on_scan_ok
    → one tab per config (stem) with ConfigSheet (clean snapshot taken)
```

### Run

```
Edit coefs in tabs → tab title gets "*" (dirty indicator, polled 300ms)
  → click Run
    → app._clear_log (flush queue + clear ScrolledText)
    → app._write_coefs per tab (skips tabs where is_dirty == False)
      → config_yaml.update_coefs_in_run_yaml(yaml_path, patch)
      → cs.mark_clean() → removes "*"
    → worker._run (thread)
      → call_in_raw_dir(processing.run,
          input={path: data_path, yaml_path: "(stem1|stem2)"})
        → processing.run: yaml_path filter → process_loading_yaml
          → run_processing: full pipeline (load → coefs → process → persist)
          → return (processed_pcids, failed_pcids, last_cfg, collected)
      → result_queue.put(("run_ok", result))
    → app._poll_results → _on_run_done → reset bars
```

### Pause / Resume

```
Click Run while processing → PauseGate
  → pause(): gate.clear()
    → QueueHandler.emit: gate.wait() blocks → log freeze
    → GuiTqdm.update: gate.wait() blocks → dask task freeze
  → resume(): gate.set() → both unblock
```

## Key decisions

| Decision | Why |
|---|---|
| `hydra_main` in thread, not Compose API | logging, resolvers, runtime state require `@hydra.main` |
| QueueHandler inside `_wrap(fun)` | Hydra `dictConfig` resets handlers; inject after |
| `_runtime` is module-level, not `threading.local` | `TqdmCallback` creates `GuiTqdm` in dask worker threads |
| `return_="<cfg_from_args>"` for Scan | pipeline does discovery + gen_metadata, returns configs without processing |
| `input.yaml_path` for Run | documented regex filter, skip discovery, only selected configs |
| `_run` uses minimal `sys.argv` | YAML files are sole config source; `original_argv` overrides not re-applied |
| `PauseGate` in log + tqdm, not pipeline | pipeline code untouched; pause on next tick |
| `_COEF_SHAPES` dict in coef_sheet | single source for tree structure; `None` → empty cells by shape |
| `meta_date_cols` explicit, not `_is_date` | metadata dates override alignment; `_is_date` only in edit validation |
| dirty tracking via `_snap` tuple in ConfigSheet | snapshot `(coefs, dates, path)` after load; `is_dirty` compares current vs snap |
| `"*"` on tab title (300 ms poll) | visual feedback for unsaved edits; removed by `mark_clean()` after write |
| `_write_coefs` skips clean tabs | avoids redundant timestamped backups identical to existing YAML |
| `_clear_log` on scan/run start | prevents cross-operation message accumulation in ScrolledText |
| `QueueHandler` consecutive dedup | drops records with identical `(funcName, message)` in a row |

## Pipeline patches (minimal)

| File | What | Lines |
|---|---|---|
| `cli.py::call_in_raw_dir` | `exit_on_error=False` → raise instead of `sys.exit(1)` | ~3 |
| `processing.py` | `TqdmCallback(tqdm_class=get_tqdm_class() or tqdm)` in `_process_and_persist` | ~2 |
| `processing.py::process_loading_yaml` | collect `(stem, yaml_path, result)` as 4th element of return | ~5 |
| `processing.py::run_processing` | `Stage` enum + `_stage()` ticks at stage boundaries | ~15 |
| `csv_load.py::_pattern_to_regex` | `if '|' in name: return name` (regex alternation) | ~2 |

## Progress

Upper bar (config-level): stage-aware ticks via `processing.Stage` enum —
load → coefs → proc → NC write (per bin) → TSV write (per bin).
Each config occupies 100 units; active stages share the scale evenly.
NC stages counted only when `use_h5` is `True`.

Lower bar (stage-level): dask task progress via `GuiTqdm` injected into
`TqdmCallback(tqdm_class=GuiTqdm)` in `processing.py`.

## CLI integration

GUI accepts CLI args: first positional = data path (prefills GUI entry, auto-scans), then
`key=value` = Hydra overrides passed verbatim via `sys.argv`
(see [`cli.py` internals](../tcm_clc/how_it_works.md#entry-point) for
`call_in_raw_dir`, `parse_data_path`, `hydra_main`)
Example: `python -m tcm_gui "D:/data/_raw/@i_p1.TXT" "input.ids=[i90]"`.

`App.__init__` stores `self._original_argv`.  For **Scan**, Worker resets
`sys.argv = list(original_argv)` so Hydra composes with launch-time overrides.
For **Run**, Worker uses a minimal `sys.argv` (script name only) —
the YAML files are the sole config source, `original_argv` overrides are not
re-applied.
