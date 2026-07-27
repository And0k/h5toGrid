# How the GUI works — Internal Architecture

Optional Tkinter frontend wrapping `tcm.cli.call_in_raw_dir` in a background
thread.  No custom CLI parsing — Hydra handles all config keys natively via
`sys.argv` (see [CLI usage](../tcm_clc/README.md#quick-start)).

## Architecture

| File | Purpose |
|------|---------|
| `app.py` | Tk root, layout §1–5, 300 ms polling, argv prefill |
| `worker.py` | Background thread: `call_in_raw_dir` for Scan and Run |
| `coef_sheet.py` | tksheet treeview: type-aware widgets (checkbox/dropdown/align), node + metadata bg; row-geometry-free styling via `_row_map()` |
| `_path_field.py` | 1×1 tksheet as path field — frame-anchored hover button, column-width tracking via `<Configure>`, inward `editor_place` during edit |
| `_browse_button.py` | `BrowseOverlay` (widget core + `pending` state), `BrowseButtonManager` (sheet-edit policy + injectable `editor_place`), `SheetHoverBinder` (MT motion → overlay show/hide with pending-aware veto), `bind_hover_browse` (Entry legacy) |
| `_cell_spec.py` | Hydra dataclass → ``CellSpec`` (bool/enum/text/number/date) for cell rendering |
| `const.py` | Centralized colors & styles: `DEFAULT_FG`, `BLUE_FG`, `FG_DEFAULT`, `FUNC_COLOR`, `FRAME_BG_FALLBACK`, `ENTRY_BG_FALLBACK`, `TAG_COLORS`; resolved-theme helpers `resolved_frame_bg()`, `resolved_entry_bg()`; `tk_color_to_hex` / `tk_color_to_rgb` / `tk_font_family` |
| `cli_cfg.py` | `CFG_DEFAULTS` (config-tree defaults) + `COEF_SHAPES` (auto-derived) + `COEFS_TYPE` — all derived from `Config` via `get_type_hints`, no per-section imports |
| `progress_bridge.py` | `GuiTqdm` (tqdm replacement) + module-level runtime injection |
| `log_bridge.py` | `install()` once at App startup → root logger captures GUI-thread AND worker logs → `QueueHandler` (consecutive dedup + emit-time text freeze) → `ScrolledText` drain |
| `_rtf_clipboard.py` | `Ctrl+C` on log → RTF + plain text on clipboard (colors preserved) |
| `runtime.py` | Shared state: queues, `ProgressState` (with one-shot `clear_and_reset`/`consume_clear`), `PauseGate`, persistent `queue_handler` reference |

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
| `QueueHandler` installed once at App startup; re-attached in `_wrap` after Hydra `dictConfig` | Hydra's ``logging.config.dictConfig`` replaces **all** root handlers with ``[console, file]`` each worker task, removing the ``QueueHandler`` from root.  ``_wrap.wrapped`` (running *after* dictConfig) re-adds it so both worker-thread and GUI-main-thread logs (e.g. ``_reload_coefs`` triggered by treeview interaction) reach the ScrolledText.  ``reset_dedup()`` per task prevents the first record of a new task from being swallowed as a "duplicate" of the previous task's tail |
| `_runtime` is module-level, not `threading.local` | `TqdmCallback` creates `GuiTqdm` in dask worker threads |
| `return_="<cfg_from_args>"` for Scan | pipeline does discovery + gen_metadata, returns configs without processing |
| `input.yaml_path` for Run | documented regex filter, skip discovery, only selected configs |
| `_run` uses minimal `sys.argv` | YAML files are sole config source; `original_argv` overrides not re-applied |
| `PauseGate` in log + tqdm, not pipeline | pipeline code untouched; pause on next tick |
| `COEF_SHAPES` auto-derived in `cli_cfg.py` | `infer_coef_shapes()` walks `ConfigInCoefs_InclProc` fields: shape from default value structure (when not `None`) or `Annotated` metadata; `P_t` annotated `(3,3)` since default is `None` |
| `cli_cfg` derives section types from `Config` | single `Config` import + `get_type_hints()` → `_SECTION_TYPES` dict; `COEFS_TYPE` extracted from `Config.input.coefs` field; no per-section imports needed |
| `meta_date_cols` explicitly, not `as_date` | metadata dates override alignment; `as_date` only in edit validation |
| `_meta[iid]["path"]` — dotted Hydra path | `_ins()` computes `parent_path + "." + text`; array children override with explicit correct paths (e.g. `input.coefs.Ag[0]`, not doubled `input.coefs.Ag.Ag[0]`) |
| `_meta[iid]["parent"]` backlink | set in `_ins()` from `parent_iid`; enables `_node_at_default` recursive walk |
| **Two-row system** — internal vs display | `_row_map()` → **internal** rows (all items, even collapsed) for cell-API calls ;  `_walk(visible=True)` → **display** rows (collapsed items compressed out) for event decoding |
| `_row_map()` → `get_row_from_iid` / fallback walk | When `get_row_from_iid` fails, walks ALL items (depth-first) — used for API calls that don't know about collapse |
| `_walk(visible=True)` → visible-only DFS | `sh.get_children` with `_is_open` check (tksheet `MT.treeview` truth, `_meta["open"]` fallback); collapsed subtrees yield no rows; feeds `_vis` in `_rebuild_row_caches`, used by `_iid_at_row` + `_apply_styles` node-fg updates |
| open-state bookkeeping ; `_item_hook` wraps `sh.item()` | tksheet 7.6's getter doesn't expose `"open"` key; every `open_` set call (ours + tksheet arrow toggles) is recorded in `_meta[iid]["open"]` |
| `_iid_at_row(r)` = visible-only lookup | `self._vis[r]` — O(1) indexing into the display-row tuple built by `_rebuild_row_caches`; ignores collapsed children |
| gray foreground for default values | `_apply_default_fg()` → `_default_for_cell(iid,m,j)` → `CFG_DEFAULTS` via `default_for_path`; works for ALL config sections (input, out, filter, program), not just coefs |
| `_default_for_cell` rejects dict results | non-leaf paths (e.g. `"input"`) return `NO_DEFAULT`; `input`-type cells append `.path` to resolve the input.path field |
| `_fg_default` — theme foreground color | resolved once from `TFrame` foreground via `const.tk_color_to_hex`; applied explicitly (never `fg=None`, which is a per-key merge no-op in tksheet 7.x) |
| `_apply_edit_value` / `_apply_default_fg` use `overwrite=False` | edit-time restylers pass only `fg` to `highlight_cells`; `overwrite=False` preserves the `bg` that `_apply_styles` set (input.row data cells keep button-face after edits) |
| **Input row styling** | node label: button-face bg + normal black `FG_DEFAULT` (never blue/gray toggle); all data cells: button-face bg via `highlight_cells` across `total_columns()`.  Other rows: button-face bg + `BLUE_FG`/`_fg_default` node fg as before |
| **PathField styling** | `ENTRY_BG_FALLBACK` (white) bg + `FG_DEFAULT` (black) fg + **bold** font (sheet-wide, 1×1 cell); entry-field silhouette distinct from the gray coef_sheet cells |
| **Blue node labels** → subtree unchanged | `_node_at_default(id)` recurs: every leaf value matches its config dataclass default; `const.BLUE_FG = "#0055CC"` on index canvas |
| `_on_end_edit` → cascade toggle | Gray/clear fg per cell **+** walk ancestral tree labels (blue/standard); `_fg_default` used for clear side (not `fg=None`) |
| dirty tracking via `_data_snapshot` | `tuple(tuple(str(val) for val in row) for row in sheet)` covers ALL editable cells (not just coefs); `is_dirty` compares current vs snap |
| `"*"` on tab title (300 ms poll) | visual feedback for unsaved edits; removed by `mark_clean()` after write |
| `_write_coefs` skips clean tabs | avoids redundant timestamped backups identical to existing YAML |
| `_clear_log` on scan/run start | prevents cross-operation message accumulation in ScrolledText |
| `_clear_status` one-shot flag (not blanket clear) | original `_poll_progress` set `_status.set("")` every 300 ms when `tot == 0`, wiping "Ready", "Done …", and hover hints.  `clear_and_reset()` (worker, at probe start) + `consume_clear()` (GUI, once) replaces continuous clearing with a single event per probe boundary.  `_path_hovering` guard defers consumption while hover is active |
| `QueueHandler` consecutive dedup | drops equivalent records (same msg at same call site), registered by `funcName+msg` key.  **freezes** the rendered text onto the `LogRecord` (`rec.msg = text; rec.args = ()`) at emit time so deferred `drain`-time `getMessage()` cannot be corrupted by the mutable `Message` reused across log calls in `LoggingStyleAdapter`.  Mirrors Hydra's `job_logging/colorlog` formatter, which renders `record.getMessage()` once synchronously. |
| `Ctrl+C` → RTF + plain on clipboard | `_rtf_clipboard.copy_rich` serializes tag-colored `ScrolledText`; pywin32 absent → plain fallback |
| `config.Config` + `config.Return` passed to `load()` | structured-config root + `StrEnum` for `program.return_` dropdown |
| `_cell_spec_for` → bool/enum/text/number | walks dataclass tree via `spec_for_path`; `bool` → checkbox, `Enum` → dropdown, `str`/`Path` → left-align |
| node column bg = header bg | `highlight_cells(canvas="index")` in `_apply_styles`; `resolved_frame_bg()` (TFrame background) for all rows including `input` |
| metadata row bg up to last date cell | all cells from col 0 through last `meta_date_cols` entry share the bg |
| ordering `_apply_open()` → `_row_map()` → `_apply_styles()` → `_apply_default_fg()` | invariant: build tree → set open states → compute row map → apply styles → gray defaults → redraw |
| `date` independent of `max_col` | coefs parent has `max_col=0`; styling in dedicated section before `max_col` loop |
| PathField = 1×1 Sheet, not Entry | cell-behavior parity: double-click/keypress edit, Enter commit, Esc undo; Entry can't grow these |
| `SheetHoverBinder` extracted from ConfigSheet | three MT binds + churn veto reusable by PathField and any future sheet-hover site |
| `_hover_resolve` uses `_iid_of_row` cache | O(1) lookup on every `<Motion>` event; rebuilt in `load()` (stable between loads) |
| `_hover_resolve` publishes status for ALL rows | not just browse rows; `_clear_status` is a separate `<Leave>` bind |
| `_hover_resolve` handles identify_row API drift | tries `identify_row(event)` first (7.x), falls back to `identify_row(event.y)` (older) |

## Type-aware cell rendering (full mode)

When `Shift` is held at startup, `ConfigSheet.load()` receives the full
`Config` dataclass as `config_root`.  Each cell's type is resolved via
`_cell_spec.spec_for_path(config_root, path, Return)`:

| CellSpec.kind | Rendering | Example fields |
|---|---|---|
| `"bool"` | tksheet checkbox | `program.b_interact`, `out.b_incremental_update` |
| `"enum"` | tksheet dropdown | `program.return_` (7 `Return` values) |
| `"text"` | left-aligned | `input.path`, `out.text_path`, `program.log` |
| `"number"` | right-aligned (default) | `input.azimuth_add`, coefs matrices |
| `"date"` | right-aligned | `datetime` fields |

The `path` stored in `_meta[iid]["path"]` is the dotted Hydra path (e.g.
`"program.return_"`, `"out.dt_bins"`).  Resolution walks the dataclass
tree using `dataclasses.fields` + `get_type_hints(include_extras=True)`.
`Annotated`, `Optional`, and `Union` are unwrapped by `_cell_spec._unwrap`.

## Floating browse button (`_browse_button.py`)

A `…📁` / `…📄` button that appears next to the active editor (tksheet
TextEditor or `ttk.Entry`) and writes the selected path to **column 0** of
the target row, regardless of which column the user clicked.

### Two usage sites

| Site | Trigger | Target |
|------|---------|--------|
| `_path_field.py` §1 — data path | `SheetHoverBinder` motion policy | PathField's single cell `(0, 0)` via `set_cell_data` + `_notify` |
| `coef_sheet.py` — config tree | `_on_begin_edit_cell` for rows with `meta["browse"] = True` (`input`, `coefs_path`) | tksheet cell `(row, 0)` via `set_cell_data` |

### Create / destroy lifecycle

Button is **created** in `_acquire_and_place` (after retry-polling finds the
editor) and **destroyed** in `detach()`.  A persistent widget whose
`in_=editor` master is destroyed by tksheet would survive at stale canvas
coordinates — `place_forget()` alone is insufficient.

### Three-layer teardown (tksheet path)

| Layer | Signal | Covers |
|-------|--------|--------|
| 1 | `end_edit_cell` → `detach()` | Normal Enter / click-away close |
| 2 | `<Destroy>` on editor → `detach()` | Tree-arrow toggle, `load()` rebuild |
| 3 | `_update_icon` polling (80 ms) | Editor reuse without destroy; detects `get_text_editor_widget() is not self._editor` |

`detach()` is also called unconditionally at the start of every
`_on_begin_edit_cell` — ensures the previous button is destroyed before
any new edit, even on non-path rows.

### Idempotent reset (F3)

`attach()` opens with `detach()`, which cancels pending retry/icon jobs
and destroys any live button from a previous cycle.  Prevents button
accumulation when the user rapidly switches between cells.

### Shift-aware icon

`_is_shift_pressed()` polls `GetAsyncKeyState(0x10)` every 80 ms while the
button is visible.  Default label `…📁` (directory), Shift label `…📄` (files).

### Focus prevention

The button overrides `focus_set` to no-op and sets `takefocus=False`.
Clicking it must not steal focus from the TextEditor (tksheet closes the
editor on `<FocusOut>`).  `<Button-1>` returns `"break"` to suppress
default focus-change behavior.

## PathField (`_path_field.py`)

A 1×1 tksheet posing as the top-level path field.  Visually an Entry
(white `ENTRY_BG_FALLBACK` background, bold font, headers/index/grid
hidden), contractually a cell — double-click/keypress edit, Enter
commit, Esc undo, all native tksheet.  Normal black `FG_DEFAULT`
foreground (not `BLUE_FG`).  Reuses the floating-button stack with
PathField-specific placement: frame-anchored hover button
(geometry-managed, no canvas arithmetic) and inward `editor_place`
during edit.

### Geometry: frame-anchored hover + column tracking

The hover button is parented to the `PathField` frame (not the sheet
canvas), placed via `place(in_=self, relx=1.0, anchor="e")`.  This
decouples the button from cell width — `relx=1.0` tracks every resize
with zero bindings.  The column width is kept in sync with the frame
via `<Configure>` → `column_width(0, event.width - 2)`, so the editor
spans the field exactly and its right border stays visible.

The during-edit button uses an injectable `editor_place` lambda passed
to `BrowseButtonManager`, also anchored inward (`anchor="e", x=-2`) so
it occupies the same strip as the hover button — no glyph jump on
hover → edit → hover transitions.

After edit ends (`_on_end_edit`), `_ov.schedule_show(_place_kw())`
re-arms the hover button immediately — pointer is over the field by
construction, so no mouse move is needed.

### Design decision: why not Entry?

An Entry can't grow cell-behavior parity (double-click/keypress edit, Enter
commit, Esc undo).  A 1×1 Sheet with headers/index hidden is visually an
Entry and contractually a cell — tksheet has no "detached cell" primitive,
but this is indistinguishable from one, and everything built for the config
tree drops in.

### Build-verify traps (all degrade silently via `suppress`)

| Trap | Status |
|------|--------|
| `show_header`/`show_index` as `set_options` keys vs constructor kwargs | Constructor kwargs work in 7.6; `set_options` keys tested as fallback |
| `set_height` existence | Fallback: size PathField from the app's geometry manager |
| `attach(0, 0)` vs `(row, col, iid=None)` signature | Compatible — `iid` is optional in `BrowseButtonManager.attach` |
| Single-click-to-edit | One extra `<Button-1>` bind if double-click feels wrong for a field |
| `column_width(col, w)` runtime method | Verified in tksheet 7.6 source; fallback: `set_options(column_width=…) + redraw()` |

### Esc-cancel is silent by construction

`_pre_edit` snapshot compare swallows unchanged values — the `_on_end_edit`
handler only fires `_notify` when the value actually changed.

## SheetHoverBinder (`_browse_button.py`)

Motion policy on a sheet's MT canvas → overlay show/hide.  Extracts the
three MT binds (`<Motion>`, `<Leave>`, `<MouseWheel>`) and the churn-veto
logic from ConfigSheet into a reusable class.

### Design

```
SheetHoverBinder(sheet, overlay, resolve)
  <Motion>     → resolve(event) → place_kw | None
                 same place_kw → cancel_hide (churn veto)
                 different place_kw → schedule_show
                 None → hide
  <Leave>      → schedule_hide (pointer-check vetoes over button)
  <MouseWheel> → hide (viewport shifted → button displaced)
```

The *resolve* callback owns all business logic: row identification
(API drift: 7.x takes event object, older takes y), status-bar
publishing, browse gating.  The binder owns only mechanical
show/hide/churn.  All binds use `add="+"` — never replace tksheet's
own MT handlers (a replacing `<MouseWheel>` bind kills scrolling).

### ConfigSheet usage

```python
self._hover_binder = SheetHoverBinder(self.sh, self._hover_ov, self._hover_resolve)
# Status-bar clear on leave — separate bind (binder handles overlay hide)
self.sh.MT.bind("<Leave>", lambda _: self._clear_status(), add="+")
```

`_hover_resolve` uses `_iid_of_row` cache (rebuilt in `load()`) and
publishes status for any row (not just browse rows).  `_clear_status`
is a separate `<Leave>` binding — the binder's `<Leave>` only handles
overlay hide.

`Ctrl+C` on the log `ScrolledText` calls `copy_rich` from
[`_rtf_clipboard.py`](`_rtf_clipboard.py`), which walks all tag boundaries,
maps each tag's `foreground` to an 8-bit RGB via `winfo_rgb`, and emits a
single `{\cfN …}` segment per slice into an RTF `\\colortbl`.
Both `CF_UNICODETEXT` (plain, fallback target) and `CF_RTF` are placed on the
clipboard so Word / Outlook keep colors while plain-text targets degrade
gracefully.  When `pywin32` is unavailable, falls back to
`widget.clipboard_append(plain)`.

## Pipeline patches (minimal)

| File | What | Lines |
|---|---|---|
| `cli.py::call_in_raw_dir` | `exit_on_error=False` → raise instead of `sys.exit(1)` | ~3 |
| `processing.py` | `TqdmCallback(tqdm_class=get_tqdm_class() or tqdm)` in `_process_and_persist` | ~2 |
| `processing.py::process_loading_yaml` | collect `(stem, yaml_path, result)` as 4th element of return | ~5 |
| `processing.py::run_processing` | `Stage` enum + `_stage()` ticks at stage boundaries + `progress_stage.clear_and_reset()` at probe start | ~15 |
| `physical.py` | `get_tqdm_class() or tqdm` for binning loop → GUI stage bar | ~3 |
| `io.py` | `get_tqdm_class() or tqdm` for CSV write progress | ~3 |
| `csv_load.py::_pattern_to_regex` | `if '|' in name: return name` (regex alternation) | ~2 |

## Progress

Two independent progress bars + a status text line, all driven by
`ProgressState` snapshots read every 300 ms in `App._poll_progress`.

### Upper bar — config-level (overall)

Stage-aware ticks via [`processing.Stage`](../tcm_clc/how_it_works.md#stage-context-stage_ctxpy)
enum — load → coefs → proc → NC write (per bin) → TSV write (per bin).
Each config occupies 100 units; active stages share the scale evenly.
NC stages counted only when `use_h5` is `True`.  Ticks are emitted by
`processing._stage()` which writes to `progress_overall`.

### Lower bar — stage-level

Per-item progress via `GuiTqdm` — a tqdm replacement that routes
`(n, total, desc)` to `progress_stage` (read by `App._poll_progress`).

Two code paths activate `GuiTqdm`:
- **Binning loop** (`physical.py`): `get_tqdm_class() or tqdm` — iterates bins,
  updating `progress_stage` per bin. GUI active → `GuiTqdm`, CLI → terminal tqdm.
- **Dask NC write** (`processing.py`): `TqdmCallback(tqdm_class=GuiTqdm)` —
  task-level progress during `.compute()` on dask arrays.

`physical.py` and `io.py` import `get_tqdm_class` with `try/except ImportError`
fallback (same pattern as `processing.py`'s `progress_bridge` import).

### Status bar text — one-shot clear signal

The status `StringVar` is **not** owned by `_poll_progress`.  Explicit setters
control it:

| Setter | When | Text |
|--------|------|------|
| `App.__init__` | startup | `"Ready"` |
| `_on_run_done` | run completion | `"Done — {pct}% ({ok}/{n} ok)"` |
| `_on_path_hover_in` | mouse enters path field | hover hint |
| `_poll_progress` (`tot > 0`) | active `GuiTqdm` | stage `desc` |
| `_poll_progress` (clear flag) | probe boundary | `""` (one-shot) |

`_poll_progress` only writes `""` when `progress_stage.consume_clear()`
returns `True` — a one-shot flag set by `ProgressState.clear_and_reset()`
at each probe boundary in `processing.run_processing`.  This prevents
aggressive clearing of "Ready", "Done …", and hover hints during idle
and inter-probe gaps, while still wiping stale stage text from the
previous probe.

Decision matrix:

| `progress_stage.tot` | `_clear_status` | `_path_hovering` | Action |
|---|---|---|---|
| `> 0` | any | `False` | show `desc` |
| `> 0` | any | `True` | preserve (hover active) |
| `0` | `True` | `False` | **clear** — flag consumed |
| `0` | `True` | `True` | preserve — flag deferred |
| `0` | `False` | any | **preserve** — no-op |

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
