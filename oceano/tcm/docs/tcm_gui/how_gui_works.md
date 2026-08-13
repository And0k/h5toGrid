# How the GUI works — Internal Architecture

Optional Tkinter frontend wrapping `tcm.cli.call_in_raw_dir` in a background
thread.  No custom CLI parsing — Hydra handles all config keys natively via
`sys.argv` (see [CLI usage](../tcm_cli/README.md#quick-start)).

## Architecture

| File | Purpose |
|------|---------|
| `app.py` | Tk root, layout §1–6, 300 ms polling, argv prefill, `_initial_scan` flag (immediate overlay show), `_prog_floater` overlay (400 ms delay for run), z-order `<Motion>` bind; manual `ttk.Frame` + `tk.Text` + `ttk.Scrollbar` log container (replaces `ScrolledText` for ttk-styled scrollbar); `_log_autoscroll` flag + `<MouseWheel>`/`<Button-4/5>` bindings for scroll-aware auto-follow; `_cfg_state: ScanStage` enum drives dual-purpose label at row=1; Run button floats via `place(in_=self._main)`; page stack + `tkraise()` (no Notebook) |
| `md_label.py` | `MarkdownLabel` (`tk.Text` subclass): Tk renderer for Markdown AST from `_md_parse`; `_current` holds parsed `Block` tuple (not raw text) — `set_text(text, raw=False)` parses Markdown by default so `STR["{role}.status"]` with `**bold**` renders bold; `raw=True` bypasses parsing for paths/keys that interpolate untrusted content (e.g. `tab.status` after `.format(path=...)`); `rerender()` replays `_render` without reparse; `mark_font_ready()` (enables auto-sizing without resizing), `fit_to_height` (rescales + enables), dynamic width (`_fit_width` via font metrics, `wrap="none"` → `wrap="word"`), auto-height (`_fit_height` on `<Configure>`), table tab-stop alignment |
| `_md_parse.py` | Pure Markdown parser (zero Tk dependency): `parse_inline()`, `parse_markdown()`, `split_table_row()`; AST types `Heading`/`Paragraph`/`CodeBlock`/`Table`/`Inline` |
| `worker.py` | Background thread: `call_in_raw_dir` for Scan and Run |
| `states.py` (tcm) | `ScanStage(StrEnum)` — scan lifecycle labels (`DEFAULT`, `SCAN`, `DONE`); `Stage(StrEnum)` — per-probe processing phase labels; value = display text |
| `coef_sheet.py` | tksheet treeview: type-aware widgets (checkbox/dropdown/align), node + metadata bg; row-geometry-free styling via `_row_map()`; floated `PathField` hover-edit on browse rows |
| `_path_field.py` | 1×1 tksheet for display + `ttk.Entry` overlay for editing — frame-anchored hover button, column-width tracking via `<Configure>` |
| `_browse_button.py` | `BrowseOverlay` (widget core + `pending` state + `on_status`/`status_hint` hover-to-status-bar wiring), `BrowseButtonManager` (sheet-edit policy + injectable `editor_place`), `SheetHoverBinder` (MT motion → overlay show/hide with pending-aware veto), `bind_hover_browse` (Entry legacy) |
| `_cell_spec.py` | Hydra dataclass → ``CellSpec`` (bool/enum/text/number/date) for cell rendering |
| `_help.py` | Auto-extract config-cell help from ``config_reference.md`` tables (``HelpEntry``, ``help_for_path``, ``parse_reference``); index-stripping for arrays (``Ag[0]`` → ``Ag``); mode-tagged `###` sections with `####` detail sub-blocks; per-lang cache (`_CACHE` dict, not `lru_cache`); `detail=` kwarg for `#### Detailed` blocks |
| `const.py` | Immutable user settings (`UI_SCALE`, `FONT_SCALE`, `TTK_THEME`, `COLOR_MODE`); `UIScale` (sets `tk scaling = platform × UI_SCALE` for uniform geometry scaling + named font multiplier via `FONT_SCALE`; `font()` returns scaled `TkDefaultFont` copy; `set_font(*widgets)` applies per-widget copies to any widget); `configure_ui` (ttk theme selection); `tk_font_family` |
| `theme.py` | Mutable runtime state: color globals (`FUNC_COLOR`, `FG_DEFAULT`, `BLUE_FG`, `DEFAULT_FG`, `FRAME_BG_FALLBACK`, `ENTRY_BG_FALLBACK`, `CELL_NON_DATA_BG`); `THEME`; `TAG_COLORS`; `widget_meta` registry; `STR` i18n surface; `get_widget_meta` (callable-resolving); `apply_theme_defaults` (dark/light via `COLOR_MODE` or Windows registry); `_apply_ttk_dark` (clam + dark ttk.Style); `_opt_into_dark_titlebar` (`DwmSetWindowAttribute`); `tk_color_to_rgb`/`tk_color_to_hex`; `resolved_frame_bg`/`resolved_entry_bg` |
| `cli_cfg.py` | `CFG_DEFAULTS` (config-tree defaults) + `COEF_SHAPES` (auto-derived) + `COEFS_TYPE` — all derived from `Config` via `get_type_hints`, no per-section imports |
| `progress_bridge.py` | `GuiTqdm` (tqdm replacement) + module-level runtime injection + `set_cfg`/`get_cfg` per-config attribution + `stage_desc` feeding both `progress_overall` and `ProgressBank` |
| `log_bridge.py` | `install()` once at App startup → root logger captures GUI-thread AND worker logs → `QueueHandler` (consecutive dedup + emit-time text freeze) → `tk.Text` drain |
| `_rtf_clipboard.py` | `Ctrl+C` on log → RTF + HTML + plain text on clipboard (colors preserved) |
| `runtime.py` | Shared state: queues, `ProgressState` (with one-shot `clear_and_reset`/`consume_clear`), `ProgressBank`, `PauseGate`, persistent `queue_handler` reference |
| `_tab_rail.py` | Vertical tab rail: progress column + tab column, configs stacked top→down.  Replaces ttk.Notebook entirely — page stack + `tkraise()` for zero-theme page switching.  Hover via `on_hover` callback, per-cell fill animation (lerp), content-based vertical sizing (waterfill on shortage, even split on extreme shortage, capped grow on surplus) |
| `progress_bank.py` | Per-configuration progress: fixed stage weights → overall fraction.  Thread-safe: workers mutate under lock, the GUI polls `snapshot_all()`.  States: pending/running/done/error; `canon_stage()` maps free-form descriptions to canonical stages |

## Data flow

### Scan

```
Browse / Enter input.path
  → app._clear_log (flush queue + clear tk.Text log)
  → app._scan → worker._scan (thread)
    → call_in_raw_dir(processing.run,
        input={path: live-path-field},
        return_="<cfg_from_args>")
       → processing.run:
           if input.path.suffix in (.yaml, .yml):
               yaml_path = path.stem → skip discovery → filter by stem
           else: discovery → gen_metadata → process_loading_yaml
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
    → app._clear_log (flush queue + clear tk.Text log)
    → app._write_coefs per tab (skips tabs where is_dirty == False)
      → config_yaml.update_coefs_in_run_yaml(yaml_path, patch)
      → cs.mark_clean() → removes "*"
    → worker._run (thread)
      → call_in_raw_dir(processing.run,
          input={path: "<dir>/cfg_proc/run/(stem1|stem2).yaml"})
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
| `QueueHandler` installed once at App startup; re-attached in `_wrap` after Hydra `dictConfig` | Hydra's ``logging.config.dictConfig`` replaces **all** root handlers with ``[console, file]`` each worker task, removing the ``QueueHandler`` from root.  ``_wrap.wrapped`` (running *after* dictConfig) re-adds it so both worker-thread and GUI-main-thread logs (e.g. ``_reload_coefs`` triggered by treeview interaction) reach the log ``tk.Text`` widget.  ``reset_dedup()`` per task prevents the first record of a new task from being swallowed as a "duplicate" of the previous task's tail |
| `_runtime` is module-level, not `threading.local` | `TqdmCallback` creates `GuiTqdm` in dask worker threads |
| `return_="<cfg_from_args>"` for Scan | pipeline does discovery + gen_metadata, returns configs without processing |
| `ScanStage(StrEnum)` for cfg state labels | value = display text; same pattern as `Stage` for processing phases; drives `_overall_lbl` at row=1 via `progress_overall.desc` during scan, `_cfg_state` attr when idle |
| `progress_overall.set()` at scan boundaries | `tick()` is per-probe (100-units math); scan-level uses direct `set()` same as `progress_stage` updates already in `processing.run()` |
| `TabRail` replaces ttk.Notebook entirely | page stack + `tkraise()` for zero-theme page selection; rail owns selection, progress, dirty state — no Notebook API remains |
| `ProgressBank` per-config tracking | stage weights (Processing=60%) → fractional fill; `run_start`/`finish` called from pipeline per-config; `stage_start`/`inner` fed via `progress_bridge` |
| `set_cfg(stem)` in `process_loading_yaml` | per-config attribution: called before `process_fun(cfg)` so `stage_desc` and `GuiTqdm` ticks land in the correct bank cell; `set_cfg(None)` reset in `_wrap.wrapped` at task start |
| `bank.finish(stem)` in `process_loading_yaml` | per-config completion: `finally` block calls `bank.finish(stem, ok)` right after each config succeeds/fails — rail cells transition independently during sequential processing |
| `_on_rail_hover` callback | rail hover → `_nb_hovering` guard + status bar yaml path display; `on_hover=None` default for headless/test usage |
| Run button `place(in_=self._main)` on root | floats at main area bottom-right with scrollbar margin; `lift()` on `<Configure>` for z-order |
| Dirty tabs → rail cell `*` via `set_dirty` | rail is the sole dirty indicator; no notebook tabs remain |
| `input.path` `.yaml` suffix for Run | `cfg_proc/run/(stem1\|stem2).yaml` → `path_in.stem` becomes regex filter, skip discovery, only selected configs |
| `_run` / `_scan` use minimal `sys.argv` | launch-time positional path stripped via `parse_data_path`; data path fed as `input.path` override, `key=value` overrides preserved for scan only — YAML files for run are the sole config source |
| `PauseGate` in log + tqdm, not pipeline | pipeline code untouched; pause on next tick |
| `COEF_SHAPES` auto-derived in `cli_cfg.py` | `infer_coef_shapes()` walks `ConfigInCoefs_InclProc` fields: shape from default value structure (when not `None`) or `Annotated` metadata; `P_t` annotated `(3,3)` since default is `None` |
| `cli_cfg` derives section types from `Config` | single `Config` import + `get_type_hints()` → `_SECTION_TYPES` dict; `COEFS_TYPE` extracted from `Config.input.coefs` field; no per-section imports needed |
| `meta_date_cols` explicitly, not `as_date` | metadata dates override alignment; `as_date` only in edit validation |
| `_meta[iid]["path"]` — dotted Hydra path | `_ins()` computes `parent_path + "." + text`; array children override with explicit correct paths (e.g. `input.coefs.Ag[0]`, not doubled `input.coefs.Ag.Ag[0]`) |
| `_meta[iid]["parent"]` backlink | set in `_ins()` from `parent_iid`; enables `_node_at_default` recursive walk |
| **Two-row system** — internal vs display | `_row_map()` → **internal** rows (all items, even collapsed) for cell-API calls ;  `_walk(visible=True)` → **display** rows (collapsed items compressed out) for event decoding.  `_on_begin_edit_cell` always converts display→internal via `_internal_row(iid)` before `get_cell_data` — prevents reading wrong cell when ancestors are collapsed |
| `_row_map()` → `get_row_from_iid` / fallback walk | When `get_row_from_iid` fails, walks ALL items (depth-first) — used for API calls that don't know about collapse |
| `_walk(visible=True)` → visible-only DFS | `sh.get_children` with `_is_open` check (tksheet `MT.treeview` truth, `_meta["open"]` fallback); collapsed subtrees yield no rows; feeds `_vis` in `_rebuild_row_caches`, used by `_iid_at_row` + `_apply_styles` node-fg updates |
| open-state bookkeeping ; `_item_hook` wraps `sh.item()` | tksheet 7.6's getter doesn't expose `"open"` key; every `open_` set call (ours + tksheet arrow toggles) is recorded in `_meta[iid]["open"]` |
| `_iid_at_row(r)` = visible-only lookup | `self._vis[r]` — O(1) indexing into the display-row tuple built by `_rebuild_row_caches`; ignores collapsed children |
| gray foreground for default values | `_apply_default_fg()` → `_default_for_cell(iid,m,j)` → `CFG_DEFAULTS` via `default_for_path`; works for ALL config sections (input, out, filter, program), not just coefs |
| `_default_for_cell` rejects dict results | non-leaf paths (e.g. `"input"`) return `NO_DEFAULT`; `input`-type cells append `.path` to resolve the input.path field |
| `_fg_default` — theme foreground color | `theme.FG_DEFAULT` (set by `apply_theme_defaults`); applied explicitly (never `fg=None`, which is a per-key merge no-op in tksheet 7.x) |
| `_apply_edit_value` / `_apply_default_fg` use `overwrite=False` | edit-time restylers pass only `fg` to `highlight_cells`; `overwrite=False` preserves the `bg` that `_apply_styles` set (input.row data cells keep button-face after edits) |
| **Input row styling** | node label: button-face bg + normal black `FG_DEFAULT` (never blue/gray toggle); all data cells: button-face bg via `highlight_cells` across `total_columns()`.  Other rows: button-face bg + `BLUE_FG`/`_fg_default` node fg as before |
| **PathField styling** | `theme.ENTRY_BG_FALLBACK` bg + `theme.FG_DEFAULT` fg + **bold** font (sheet-wide, 1×1 cell); left-aligned by default (`align="w"`), on hover tksheet widget shrinks to `frame − btn_w` via `place(width=…)` + right-aligned (`align="e"`) — same `_field_place_kw` geometry as ConfigSheet; entry-field silhouette distinct from the gray coef_sheet cells; **editing via `ttk.Entry` overlay** (veto tksheet's `tk.Text`), `justify="right"` |
| **Blue node labels** → subtree at default | `_node_at_default(iid)`: own cells (the node's `max_col`) AND every child subtree must match its config dataclass default; `theme.BLUE_FG = "#0055CC"` on index canvas. `len` is array-shape metadata, NOT `max_col` — a `1d` parent (`max_col=0`+`len=n`, e.g. `kVabs`) holds date columns only and defers entirely to its child row, so it turns blue iff the child row's cells match the dataclass default |
| `_on_end_edit` → cascade toggle | Gray/clear fg per cell **+** walk ancestral tree labels (blue/standard); `_fg_default` used for clear side (not `fg=None`) |
| dirty tracking via `_data_snapshot` | `tuple(tuple(str(val) for val in row) for row in sheet)` covers ALL editable cells (not just coefs); `is_dirty` compares current vs snap |
| `"*"` on tab title (300 ms poll) | visual feedback for unsaved edits; removed by `mark_clean()` after write |
| `_write_coefs` skips clean tabs | avoids redundant timestamped backups identical to existing YAML |
| `_clear_log` on scan/run start | prevents cross-operation message accumulation in log ``tk.Text`` widget |
| `_log_autoscroll` flag + scroll bindings | persistent flag (not `yview()` threshold — `see("end")` yields ~0.91–0.98, never 1.0); starts `True`, cleared by `<MouseWheel>`/`<Button-4/5>` when `after_idle` check finds `yview()[1] < 0.90`, restored when user scrolls back to bottom; `_poll_logs` calls `see("end")` only when flag is `True` |
| `_clear_status` one-shot flag (not blanket clear) | original `_poll_progress` set `_status.set("")` every 300 ms when `tot == 0`, wiping "Ready", "Done …", and hover hints.  `clear_and_reset()` (worker, at probe start) + `consume_clear()` (GUI, once) replaces continuous clearing with a single event per probe boundary.  `_path_hovering` guard defers consumption while hover is active |
| `QueueHandler` consecutive dedup | drops equivalent records (same msg at same call site), registered by `funcName+msg` key.  **freezes** the rendered text onto the `LogRecord` (`rec.msg = text; rec.args = ()`) at emit time so deferred `drain`-time `getMessage()` cannot be corrupted by the mutable `Message` reused across log calls in `LoggingStyleAdapter`.  Mirrors Hydra's `job_logging/colorlog` formatter, which renders `record.getMessage()` once synchronously. |
| `Ctrl+C` → RTF + HTML + plain on clipboard | `_rtf_clipboard.copy_rich` serializes tag-colored log ``tk.Text`` widget; bound on root `<<Copy>>` (not `<Control-c>`), writes CF_RTF + HTML Format + CF_UNICODETEXT, retries `OpenClipboard` on contention, falls back to plain text on `ImportError` or exhausted retry |
| `config.Config` + `config.Return` passed to `load()` | structured-config root + `StrEnum` for `program.return_` dropdown |
| `_cell_spec_for` → bool/enum/text/number | walks dataclass tree via `spec_for_path`; `bool` → checkbox, `Enum` → dropdown, `str`/`Path` → left-align |
| node column bg = header bg | `highlight_cells(canvas="index")` in `_apply_styles`; `resolved_frame_bg()` (TFrame background) for all rows including `input` |
| metadata row bg up to last date cell | all cells from col 0 through last `meta_date_cols` entry share the bg |
| ordering `_apply_open()` → `_row_map()` → `_apply_styles()` → `_apply_default_fg()` | invariant: build tree → set open states → compute row map → apply styles → gray defaults → redraw |
| `date` independent of `max_col` | coefs parent has `max_col=0`; styling in dedicated section before `max_col` loop |
| PathField = 1×1 Sheet for display, `ttk.Entry` for editing | Display: left-aligned default; on hover tksheet widget shrinks (`place(width=frame−btn_w)`) + right-align + `xview_moveto(1.0)` — filename ends before button.  Edit: `ttk.Entry` is inherently single-line (no wrapping), native horizontal scroll, cursor always visible.  tksheet's `tk.Text` editor cannot disable wrapping (`table_wrap` is display-only).  Veto via `return None` from `begin_edit_cell` callback |
| `SheetHoverBinder` extracted from ConfigSheet | three MT binds + churn veto reusable by PathField and any future sheet-hover site |
| `_hover_resolve` uses `_iid_of_row` cache | O(1) lookup on every `<Motion>` event; rebuilt in `load()` (stable between loads) |
| `_hover_resolve` publishes status for ALL rows | not just browse rows; status clears at `<Leave>` |
| `_hover_resolve` handles identify_row API drift | tries `identify_row(event)` first (7.x), falls back to `identify_row(event.y)` (older) |
| tree column hover on RI canvas | `_on_tree_motion` bound to `self.sh.RI`; shows section-level help; separate from MT data-cell hover |
| `_status_source` tracks hover canvas | `"tree"` (RI) / `"data"` (MT) — re-publishes status on source change for same row |
| `_any_hovering` property | combines `_path_hovering`, `_nb_hovering`, `_chrome_hovering`, `_browse_hovering` — single guard against poll clobbering |
| `_bind_chrome_hover` wires status to Run/progress/labels | `<Motion>`/`<Leave>` on all registered chrome widgets; skips `_path_field` + `nb` (own handlers) |
| **Floated PathField on browse rows** | one reusable `PathField` for text + separate `BrowseOverlay` for button; intent-delayed (120 ms); focus strictly opt-in; full edit parity free; button stays at sheet right edge while field text stops at button's left edge; `_do_field_hide` vetoes hide during `f._editing`; `_on_field_edit_end` → `_restore_hover_placement` (show button first, `update_idletasks`, then `f.place` at shortened width) |
| `_field_iid` survives hide | `_hide_hover_field` keeps `_field_iid` — `PathField._notify` queues via `after_idle`, so a commit in flight still writes to its row; `_hover_btn` (browse button) is hidden separately |

## Dark / light theme architecture

Three layers cooperate to render the entire GUI in a consistent dark or light
palette.  `theme.apply_theme_defaults(root)` runs once at startup (before any
widget is created) and orchestrates all three.

Startup flow:
```
App.__init__
  → UIScale(root)                 # tk scaling = platform × UI_SCALE + named font scaling
  → configure_ui(root)            # ttk theme selection (native/clam)
  → apply_theme_defaults(root)    # detect theme → mutate globals → ttk.Style → root.bg
  → _build()                      # widgets created with correct const values
    → PathField(sheet uses ENTRY_BG_FALLBACK at construction)
    → ConfigSheet created on scan
      → __init__: change_theme("dark") if THEME == "dark"
```

### UI_SCALE / FONT_SCALE — independent settings

`UI_SCALE` and `FONT_SCALE` are independent user-facing multipliers,
both defaulting to `1.0` (no change from platform defaults).

`UI_SCALE` scales **all** Tk geometry by setting `tk scaling` to
`platform_scaling × UI_SCALE`.  This affects every widget, padding, font,
and measurement uniformly — no per-widget configuration needed.

`FONT_SCALE` is an additional multiplier on named fonts only
(`TkDefaultFont`, `TkTextFont`, `TkMenuFont`, `TkHeadingFont`).
`FONT_SCALE=1.0` is a no-op.

| Setting | Effect |
|---------|--------|
| `UI_SCALE = 1.0` | all Tk geometry unchanged |
| `UI_SCALE = 1.5` | all Tk geometry × 1.5 (widgets, padding, fonts, measurements) |
| `FONT_SCALE = 1.0` | named font sizes unchanged |
| `FONT_SCALE = 1.2` | named font sizes × 1.2 (on top of UI_SCALE) |

Tk measurements returned by `dlineinfo`, `bbox`, `count ypixels` etc.
are already in the scaled coordinate system — no multiplier needed.

| Layer | What it styles | Mechanism |
|---|---|---|
| **theme globals** | Log tags, per-cell highlights, log ``tk.Text`` bg/fg, `MarkdownLabel` bg/fg, `tk.Frame`/`tk.Label` bg/fg | `_DARK` / `_LIGHT` palettes → `setattr` on theme module globals (`FUNC_COLOR`, `DEFAULT_FG`, `BLUE_FG`, `FG_DEFAULT`, `FRAME_BG_FALLBACK`, `ENTRY_BG_FALLBACK`, `CELL_NON_DATA_BG`, `THEME`) + `TAG_COLORS.update()` |
| **ttk.Style** | All `ttk.Frame`, `ttk.Label`, `ttk.Button`, `ttk.Entry`, `ttk.Notebook`, `ttk.Progressbar` | `theme._apply_ttk_dark(root)` → switches to ``clam`` theme (native themes ``vista``/``xpnative`` ignore ``Style().configure()`` for rendering), then ``ttk.Style().configure()`` with bg/fg from theme globals + ``style.map()`` for active/selected states; root window ``bg`` set directly |
| **tksheet** | Sheet canvas (table, header, index, scrollbars, selection) | `ConfigSheet.__init__` calls `self.sh.change_theme("dark")` when `theme.THEME == "dark"` + `scrollbar_theme_inheritance="default"` so tksheet's canvas scrollbars inherit the same ttk theme as `App.Vertical.TScrollbar`; `PathField` uses explicit `table_bg`/`table_fg` from theme at construction (no `change_theme` needed — headers/index/scrollbars hidden) |

### Widget-specific notes

| Widget | bg/fg source |
|---|---|
| `tk.Text` + `ttk.Scrollbar` (log) | `bg=theme.ENTRY_BG_FALLBACK`, `fg=theme.FG_DEFAULT`, `insertbackground=theme.FG_DEFAULT`; manual container replaces `ScrolledText` to get a real `ttk.Scrollbar` |
| Log scrollbar | `ttk.Scrollbar` with `style="App.Vertical.TScrollbar"` — matches tksheet via shared theme inheritance (`scrollbar_theme_inheritance="default"`) |
| `MarkdownLabel` (status) | `background=theme.FRAME_BG_FALLBACK`, `foreground=theme.FG_DEFAULT` |
| `tk.Frame` + `tk.Label` (prog_floater) | `bg=theme.FRAME_BG_FALLBACK`, `fg=theme.FG_DEFAULT` |
| `ConfigSheet` (tksheet) | `change_theme("dark")` + `scrollbar_theme_inheritance="default"` in `__init__`; `_apply_styles` uses `theme.resolved_frame_bg()` + `theme.FG_DEFAULT` |
| `PathField` (1×1 tksheet) | `table_bg=theme.ENTRY_BG_FALLBACK`, `table_fg=theme.FG_DEFAULT` — set at construction |
| `ttk.Entry` (PathField editor) | inherits from `ttk.Style("TEntry")` dark configuration |
| Root window + title bar | `root.configure(bg=...)` + `GetAncestor(winfo_id(), GA_ROOT)` to get real toplevel HWND (Tk's `winfo_id()` returns a child widget, not the DWM-controlled frame) + `DwmSetWindowAttribute(DWMWA_USE_IMMERSIVE_DARK_MODE=TRUE)` via `ctypes.WinDLL("dwmapi")` (Win32 only, Win11 22000+) |
| `ttk.Notebook` + tabs | `style.configure("TNotebook.Tab", ...)` + `style.map` for selected state |
| `ttk.Button` (Run) | `style.configure("TButton", ...)` + `style.map` for active/pressed |
| **Scrollbars** | Log `ttk.Scrollbar` + tksheet internal scrollbars | `App.Vertical.TScrollbar` ttk style configured in `_apply_ttk_dark` (dark) / default theme (light); tksheet uses `scrollbar_theme_inheritance="default"` so its canvas scrollbars inherit the same ttk theme as the App; log uses manual `ttk.Frame` + `tk.Text` + `ttk.Scrollbar` instead of `ScrolledText` (which uses an unstyled classic `tk.Scrollbar`) |

## Help system architecture

Two independent help sources — one per widget category:

| Source | Widgets | Key derivation | i18n mechanism |
|---|---|---|---|
| `STR` (``const.py``) | Chrome widgets (`self._path_lbl`, `_path_field`, `_overall_lbl`, `_run_btn`, `_status_lbl`) + dynamic tabs | Attribute name → role → ``STR["{role}.tooltip"]`` / ``STR["{role}.status"]`` | Replace ``STR`` dict wholesale at build for target language |
| ``_help.py`` (``config_reference.md``) | Config cells (``_meta[iid]["path"]`` keys) | ``help_for_path(strip_index(path)).short`` | Replace ``config_reference_<lang>.md`` |

### Chrome widgets: auto-registration

``App._register_chrome_help()`` runs once at the end of ``_build()``.  It walks
``vars(self)`` for all ``tk.Misc`` instances whose attribute name (minus the
leading ``_``) has entries in ``STR``.  For each match:

* ``tooltip`` = ``STR["{role}.tooltip"]`` (static string).
* ``status``  = ``STR["{role}.status"]`` (static string) **or** a bound method
  (``self._run_btn_status`` for Run) returning the live caption.  Dynamic
  ``status`` is stored as a ``Callable[[], str]`` in ``widget_meta``; the
  ``get_widget_meta`` resolver invokes it at hover-time, so it sees the current
  application state (busy/paused) and the current language (``STR``)
  simultaneously.

Widgets with no matching STR keys get no help — the loop skips them.

Dynamic tabs (created per config in ``_add_page``) don't have `self._*` names,
so the auto-role loop can't find them.  ``_add_page`` calls
``set_widget_meta(frame, status=STR["tab.status"].format(path=rel))`` directly
using the STR template (``{path}`` = yaml path relative to the data directory).
Tab hover is wired via ``TabRail._on_hover`` → ``App._on_rail_hover`` which reads
``get_widget_meta(frame, "status")`` and writes to ``_status_lbl``.  Because
``{path}`` is a filesystem path that may contain special characters,
``_on_rail_hover`` calls ``set_text(status, raw=True)`` to bypass Markdown
parsing — the template itself stays literal (no ``**bold**`` there).

Static ``STR["{role}.status"]`` strings (defined by chrome widgets — e.g.
``path_field.status``) **do** go through the Markdown parser by default at
``_on_chrome_hover`` / ``_on_path_hover_in``, so ``**bold**`` segments in
those STR entries render bold.

### MetaValue: callable status support

``widget_meta`` stores ``MetaValue = str | Callable[[], str]``.  The
``get_widget_meta`` getter resolves callables at read time:

```python
val = widget_meta.get(widget, {}).get(key, default)
return val() if callable(val) else val
```

Static ``str`` values pass through unchanged.  Dynamic status callables are
never stored pre-resolved — they close over ``STR`` and/or ``self``, so each
hover-time read reflects the live state and language.

### Config cells: doc-driven hover (no widget_meta needed)

Config cells have ``_meta[iid]["path"]`` (dotted Hydra path) — that IS the
help key.  ``coef_sheet._publish_status`` calls ``help_for_path(path)``,
which returns a ``HelpEntry(short, body)`` parsed once from the tables in
``config_reference.md``:

1. Parser walks lines, tracking code-fence state and ``## `section` `` headings
   (``input``, ``input.coefs``, ``out``, ``filter``, ``program``).
2. Inside a config-group section, every markdown table row whose first cell is
   a backtick-quoted identifier (``| `field` | … | description |``) emits
   ``HelpEntry(path="{section}.{field}", short=<last cell>, body={})``.
3. **Mode-tagged sections** — ``### `field.path` <mode>value</mode>``
   subheaders accumulate detail content into ``body[mode]``.  See
   [Mode-tagged sections](#mode-tagged-sections-in-config_referencemd) below.
4. CamelCase field names (``Ag``, ``Cg``, ``Rz``) parse identically to
   lowercase Hydra names.
5. ``_DOC_PATH`` resolves to ``config_reference.md`` at
   ``{tcm_root.parent}/docs/tcm_cli/config_reference.md``; absent file →
   empty cache → no hover text (graceful degradation).
6. Array indices stripped at lookup time: ``Ag[0]`` / ``Ag[1][2]`` → ``Ag``.

Fallback chain in ``_publish_status``:
``hover_status[ident]`` → ``help_for_path(candidate).short`` → ``key`` / ``label`` / ``path``.
No ``set_widget_meta`` calls on config cells — the entire chain is read-only
from the parsed doc.

**Tree column vs data cells**: the tree column renders on tksheet's RI (Row
Index) canvas, which is separate from the MT canvas.  ``_on_tree_motion``
(bound to ``RI``) always resolves the section-level path as-is (e.g.
``help_for_path("input")`` → "Data source & parameters").  ``_publish_status``
(bound to ``MT``) tries relocated-field paths FIRST for parent rows:

| Row | Candidate order (first match wins) | Result |
|---|---|---|
| ``input`` node (data cell) | ``input.path``, ``input`` | "File path, glob, or regex pattern…" |
| ``input.coefs`` parent (date cell) | ``input.coefs.date``, ``input.coefs.path``, ``input.coefs`` | "Overall calibration date" |
| ``input.coefs.Ag`` | ``input.coefs.Ag.path``, ``input.coefs.Ag`` | "Accelerometer scale matrix…" |

``_status_source`` (``"tree"`` / ``"data"`` / ``None``) tracks which canvas owns
the current status so moving between tree column and data cell on the SAME row
triggers a re-publish.

### Mode-tagged sections in `config_reference.md`

When a field's meaning depends on context (per-probe processing vs. input
specification), the table cell stays minimal and detailed documentation goes
into ``###`` subsections tagged with a **mode**:

```markdown
## `input` — Data source & parameters
| Field | Type | Default | Required | Purpose |
|-------|------|---------|----------|---------|
| `path` | `str` | — | **Yes** | Absolute path to the data file. Determines probe identity (pcid). |

### `input.path` <mode>probe</mode>
Absolute path to the data file.  The filename **determines probe identity** (pcid):
the pipeline extracts the leading type letter...

### `input.path` <mode>search</mode>
supports glob (`*i*.txt`), regex (`i.*\.txt`), or directory.
`.yaml` suffix filters existing configs by stem.
```

**Heading syntax**: ``### `dotted.field.path` <mode>value</mode>``
(``</>`` shorthand also accepted).  Regex:
``^###\s+`([A-Za-z_]\w*(?:\.\w+)*)`\s+<mode>([a-z_]+)</(?:mode)?>``

**Parser behavior**:
- ``_FIELD_MODE_HEAD`` is checked **before** ``_ANY_HEADING`` so ``###``
  mode-tagged headers don't close the parent ``##`` section.
- Each ``###`` subheader opens accumulation under ``HelpEntry.body[mode]``.
- Next ``###`` or ``##`` closes the previous accumulation.
- Fields without mode-tagged content get ``body={}``.

**Consumer API** — ``help_for_path(path, *, mode=None)``:

| Call | Return |
|------|--------|
| ``help_for_path("input.path")`` | ``HelpEntry(body={"probe": "...", "search": "..."})`` |
| ``help_for_path("input.path", mode="probe")`` | ``HelpEntry(body="...")`` — probe content only |
| ``help_for_path("input.path", mode="search")`` | ``HelpEntry(body="...")`` — search content only |

**Mode names** — content regimes, not environment labels:

| Mode | Content regime | Example consumer |
|------|---------------|------------------|
| ``probe`` | Per-probe processing meaning — what the field does, how it affects results | GUI coef hover, popup |
| ``search`` | Input specification patterns — glob, regex, directory, YAML | GUI path field, CLI help |

To add a new mode: (1) add a ``### `field.path` <mode>new_mode</mode>``
subsection in ``config_reference.md``; (2) call ``help_for_path(path,
mode="new_mode")`` in the consumer.  The parser recognizes any ``[a-z_]+``
mode value — no code changes needed in ``_help.py``.

## i18n architecture

All user-visible strings are centralized in `str.yaml` (loaded once at startup
via `const.load_str()`, cached).  No hardcoded display text in app.py,
worker.py, _browse_button.py, _path_field.py, _tab_rail.py, or
coef_sheet.py — every label, button text, dialog title, status message,
context menu item, and format template is read from the cached STR dict
at runtime.

### String categories in `str.yaml`

| Category | Key pattern | Example |
|----------|-------------|---------|
| Window chrome | `window.*`, `default_page.*` | `window.title: "TCM"` |
| Chrome tooltips / hover status | `{role}.tooltip`, `{role}.status` | `run.start: "Start processing"` |
| Browse dialog titles | `dialog.*` | `dialog.data_dir: "Browse data path"`, `dialog.filter_search: "Data & configs"` |
| Browse button labels | `browse.*` | `browse.dir_label: "…📁"` |
| Sheet context menu | `sheet.*` | `sheet.insert_col: "Insert column"` |
| Button labels | `run_btn.*` | `run_btn.pause: "Pause"` |
| Status / progress text | `status.*` | `status.ready: "Ready"` |
| Completion template | `overall_lbl.done_detail` | `" - Done {pct}% ({ok}/{n} ok)"` |
| Error prefixes | `error.*` | `error.scan: "Scan: {p}"` |
| PathField placeholder | `path_field.placeholder` / `path_field.placeholder_shift` | `"D:/data/_raw/"` / `"D:/data/_raw/(i*raw_file1[.]txt\|i*raw_file2[.]txt)"` |

### Locale switching

`LANG` in `const.py` controls language selection (same pattern as
`COLOR_MODE`):

| Value | Behavior |
|-------|----------|
| `"auto"` (default) | Detect from OS locale via `locale.getlocale()` → two-letter code (e.g. `"ru"`). Falls back to `"en"`. |
| `"en"`, `"ru"`, etc. | Explicit language code — loads `str_{lang}.yaml` |

Resolution: `const.resolve_lang()` → cached two-letter code.
Loading: `const.load_str()` → `str_{lang}.yaml` if exists, else `str.yaml`.
Result is cached — all modules calling `load_str()` get the same dict.

Modules that need i18n: `app.py`, `worker.py`, `_browse_button.py`,
`_path_field.py`, `_tab_rail.py`, `coef_sheet.py` — each imports
`load_str` from `const`.

To add a language: create `str_{lang}.yaml` with the same keys as `str.yaml`.

### PathField placeholder

When the path field is empty, a dim-gray placeholder example is shown
(`path_field.placeholder` from str.yaml — simple directory hint).  Holding
**Shift** swaps to `path_field.placeholder_shift` (advanced pattern syntax);
releasing Shift restores the simple one.  The swap only applies when the
placeholder is visible (cell empty) and no edit is active — it does not
alter `_has_placeholder` state.

`path_field.status` and `path_field.status_shift` are two separate i18n keys:
normal hover status and Shift-held status (YAML config loading hint).  PathField
reads both from ``STR`` at init; `_on_shift_press`/`_on_shift_release` swap the
status bar text alongside the placeholder.

The placeholder clears on first keystroke or double-click edit, and
reappears when the field is committed empty.  `PathField.get()` returns
`""` while the placeholder is visible — the placeholder text is never
treated as user input.

### YAML config selection in PathField

When the PathField value ends with `.yaml`/`.yml`, `processing.run` detects
the suffix, derives `yaml_path` from `path_in.stem`, and skips data
discovery — loading only matching configs into tabs.  No GUI-side plumbing
needed — the detection happens in the CLI layer.

The `input.yaml_path` config field has been removed.  Callers pass
``input.path=<dir>/cfg_proc/run/(stem1|stem2).yaml`` instead.  The
``.yaml`` suffix must be **outside** the alternation (``Path.stem`` strips
it); ``Path("(file1[.]yaml|file2[.]yaml)").stem`` gives
``"(file1[.]yaml|file2"`` — broken.  Correct: ``(file1|file2).yaml``.

See [config_reference.md `input.path`](../tcm_cli/config_reference.md#input--data-source--parameters)
for the CLI equivalent.

### `states.py` enum values

`ScanStage` and `Stage` in `tcm/states.py` are `StrEnum` whose values are the
display text shown in progress bars and the overall label.  These are **code
constants** — to translate them, either change the enum values directly or add
a lookup layer in `str.yaml` keyed by `"{EnumName}.{value}"` with fallback to
the enum value.

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
| `coef_sheet.py` — in-sheet edit | `_on_begin_edit_cell` for rows with `meta["browse"] = True` (`input`, `coefs_path`) | tksheet cell `(row, 0)` via `set_cell_data` |
| `coef_sheet.py` — hover-edit | intent-delayed `PathField` (text) + separate `BrowseOverlay` (button at right edge) over browse rows | `_hover_write` → `set_cell_data` + `_apply_edit_value` + `coefs_path` notify |

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

### Status-bar hint on hover

`BrowseOverlay` accepts `on_status: Callable[[str], None]` and
`status_hint: str` parameters.  On `<Enter>` the button calls
`on_status(status_hint)` to show the Shift hint in the GUI status bar;
on `<Leave>` it calls `on_status("")` to clear.  `_make_button` binds
both events with `add="+"`.  All three creation sites pass the callback:

| Site | `on_status` source | `status_hint` source |
|------|-------------------|---------------------|
| `PathField` (§1 data path) | `App._on_browse_status` | `STR["browse_btn.status"]` via `PathField.__init__` |
| `BrowseButtonManager` (in-sheet edit) | `App._on_browse_status` | `STR["browse_btn.status"]` via `BrowseButtonManager.__init__` |
| `ConfigSheet._ensure_hover_field` (hover-edit) | `lambda text: self.on_hover_status(text, True)` | `tcm_gui.theme.STR["browse_btn.status"]` |

`App._on_browse_status` sets `_browse_hovering = bool(text)` — included in
`_any_hovering` — so the 300 ms poll does not clobber the hint during
progress-stage boundary clear signals.

### Focus prevention

The button overrides `focus_set` to no-op and sets `takefocus=False`.
Clicking it must not steal focus from the TextEditor (tksheet closes the
editor on `<FocusOut>`).  `<Button-1>` returns `"break"` to suppress
default focus-change behavior.

## PathField (`_path_field.py`)

A 1×1 tksheet for **display** + a `ttk.Entry` overlay for **editing**.

**Display mode**: column=4096, `align="w"` by default — long paths
show the beginning (directory).  On hover the tksheet widget shrinks
to `frame − button` via `place(width=…)` (same pattern as
ConfigSheet's `_field_place_kw`) and switches to `align="e"` with
`xview_moveto(1.0)` — filename ends right before the browse button.
`ENTRY_BG_FALLBACK` background, bold font, all chrome hidden.  Normal
black `FG_DEFAULT` foreground (not `BLUE_FG`).  The `align` parameter
defaults to `"w"` (standalone); ConfigSheet passes `align="e"` for the
floated field.

**Edit mode**: `_on_begin_edit` returns `None` to **veto** tksheet's
built-in `tk.Text` editor, then places a `ttk.Entry` filling the
PathField frame (`relx=0, rely=0, relwidth=1, relheight=1`).
The Entry has `justify="right"` so the cursor starts at the filename
end.  Enter commits, Esc cancels — same contract as a tksheet cell.

### Why `ttk.Entry` instead of tksheet's `tk.Text` editor

tksheet's editor is a `tk.Text` widget that defaults to `wrap="char"`.
There is no tksheet API to set `wrap="none"` on the editor.  Options
tried and rejected:

| Approach | Problem |
|----------|---------|
| `table_wrap=""` on Sheet | Controls display rendering only, not the editor widget |
| `after(ms)` callback to set `wrap="none"` | Races with tksheet's `update_idletasks` / redraw; unreliable |
| Monkey-patch `MT.open_text_editor` | `wrap="none"` + right-justify: `see("insert")` can't scroll right-justified content past the left edge |
| Constrain column + left-align during edit | `wrap="char"` still wraps regardless of alignment |

`ttk.Entry` is inherently single-line — no wrapping, native horizontal
scroll, cursor always visible.  Vetoing the tksheet editor via
`return None` from the `begin_edit_cell` callback is the documented
tksheet mechanism for custom editor implementations.

### Edit lifecycle

```
_double-click / keypress_
  → tksheet fires begin_edit_cell
  → _on_begin_edit:
      _editing = True
      _pre_edit = get()           # snapshot for Esc-undo
      _ov.hide()                  # hide browse overlay (+ restore left-align)
      _on_begin_edit_cb()         # coef_sheet: cancel hide, hide btn, expand field
      _open_entry()               # ttk.Entry fills PathField frame
      return None                 # VETO tksheet's tk.Text editor

_Enter_ → _commit_entry:
      read Entry value
      destroy Entry
      _editing = False
      _on_end_edit_cb()           # coef_sheet: _restore_hover_placement
      set_cell_data(0, 0, value)  # if not cancel
      redraw + scroll if right-aligned (floated field)
      _notify(value)              # if changed

_Escape_ → _commit_entry(cancel=True):
      destroy Entry
      _editing = False
      _on_end_edit_cb()           # coef_sheet: _restore_hover_placement
      no _notify (pre_edit == current)
```

### `<Configure>` guard

`_on_configure` skips during `_editing`.  When `_hovering`, re-places
the tksheet at `width = max(frame − btn_w, 50)` (resize tracking) and
scrolls right.  When right-aligned but not hovering (floated field),
calls `_scroll_to_right()`.  When left-aligned (default standalone),
no-op.

### Hover widget shrink

The overlay's `schedule_show` and `hide` are patched to call
`_switch_to_hover_shrink()` / `_restore_default_layout()`.  On hover:
switch tksheet from `pack` to `place(width=max(frame−btn_w, 50))` +
`table_align("e")` + `_scroll_to_right()`.  On leave: `place_forget`
→ `pack(fill="both", expand=True)` + `table_align("w")`.  Same
geometry as ConfigSheet's ``_field_place_kw``.

The hover button is parented to the `PathField` frame (not the sheet
canvas), placed via `place(in_=self, relx=1.0, x=-2, anchor="ne")`.
`relx=1.0` tracks every resize with zero bindings.

### `cancel_edit()` public API

Destroys the Entry if open (no-op otherwise).  Used by
`ConfigSheet._hide_hover_field` and `_show_hover_field` to hand off
between rows cleanly.

## Floated PathField in ConfigSheet (`coef_sheet.py`)

One reusable `PathField` instance (text surface) + a separate `BrowseOverlay`
(button at the sheet's right edge) float over browse rows
(`input`, `coefs_path`) on hover, replacing the former single `BrowseOverlay`.

### Intent-delayed show / hide

```
<Motion> on browse row → _schedule_field_show (120 ms)
  → _show_hover_field: cancel_edit, _field_iid = iid, .set(val), place, lift
<Motion> off browse row → _schedule_field_hide (120 ms)
  → _do_field_hide: editing veto OR pointer-in-field veto → _hide_hover_field
<Leave> → schedule_field_hide + clear status UNLESS pointer is inside the field
<MouseWheel> → immediate _hide_hover_field
load() / _tree_shape_changed / _on_begin_edit_cell → immediate _hide_hover_field
```

`_do_field_hide` checks `f._editing` **before** the pointer-in-field
test — during Entry editing the field must stay mapped regardless of
pointer position (the Entry fills the PathField frame but the pointer
may drift outside its bounds).  After edit ends, `_on_field_edit_end`
→ `_restore_hover_placement` immediately shrinks the field back to
hover width and re-shows the browse button.

### Status bar preservation

When the pointer transitions from MT to the floated field, `<Leave>` fires
on MT but `_on_sheet_leave` checks `_pointer_in_field()` first — if the
pointer landed on the field, the status bar text is preserved (the pointer
is still conceptually on the same row).

### Focus is opt-in

`place` / `set` / `lift` never focus.  `startup_focus=False` (pinned in
`PathField.__init__`) prevents the sheet from stealing focus on creation.
The first focus arrives only from the user's click, which then enters
native click-to-edit via tksheet's normal edit binding.

### Full edit parity (free)

Because the surface IS a `PathField`, all its capabilities arrive for free:
Enter commit → `_hover_write` (set_cell_data + `_apply_edit_value` +
`coefs_path` notify).  Esc undo → `_commit_entry(cancel=True)`.  Browse
button → separate `BrowseOverlay` at the sheet's right edge (Shift toggles
dir/file).  PathField's own internal browse overlay is suppressed — only
the reparented button is active.  Editing uses `ttk.Entry` overlay (not
tksheet's `tk.Text`) — same `justify="right"` behavior as the top path
field.

### In-sheet edit fallback

A click landing before the 120 ms intent window edits in place through
tksheet's native cell editor + `BrowseButtonManager` (the existing
`_on_begin_edit_cell` / `_on_end_edit_cell` pipeline).  Both paths
converge at `_hover_write` / `_on_end_edit_cell` for the commit.

### Commit-race closure

`_hide_hover_field` deliberately keeps `_field_iid` — `PathField._notify`
queues through `after_idle`, so a commit already queued when the field is
hidden still writes to its row.

### Edit lifecycle on the overlay

When the user clicks the floated PathField to edit:

1. `_on_begin_edit` (PathField) fires → sets `_editing = True` → calls
   `_on_begin_edit_cb` → `ConfigSheet._on_field_edit_start`:
   cancels pending hide job, hides browse button, expands field to
   full width (`_field_full_width_kw`), forces geometry via
   `update_idletasks`.
2. PathField constrains column to frame width, opens `ttk.Entry`,
   returns `None` (vetoes tksheet editor).
3. Enter/Esc → `_commit_entry` → `_editing = False` → calls
   `_on_end_edit_cb` → `ConfigSheet._on_field_edit_end` →
   `_restore_hover_placement`: shrinks field to hover width
   (`_field_place_kw`), re-shows browse button.
4. `_commit_entry` continues: restores column to 4096, scrolls right.

`_do_field_hide` vetoes hide while `_editing` is `True` — prevents
`<Leave>` (armed when pointer stepped onto Entry) from unmapping the
field mid-edit.

### Width rule

Field text area ends where the browse button starts — `browse_button_width`
(measured once, cached as `_btn_w_cache`) is subtracted from the data strip
width.  Minimum 50 px fallback.  The browse button itself is a separate
`BrowseOverlay` parented to `self.sh` and placed at the sheet's right edge
(same position as the old hover overlay).  Implemented in `_field_place_kw`
+ `_btn_place_kw`.

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

ConfigSheet no longer uses `SheetHoverBinder` directly — it owns the
hover lifecycle through intent-delayed `PathField` floats
(`_schedule_field_show` / `_schedule_field_hide`).  `SheetHoverBinder`
remains in `_path_field.py` for the top-level path field's own hover
button.

`_hover_resolve` uses `_iid_of_row` cache (rebuilt in `load()`) and
publishes status for any row (not just browse rows).  `_on_sheet_leave`
triggers `_schedule_field_hide` (pointer-check vetoes over field).

`Ctrl+C` on the log `tk.Text` widget calls `copy_rich` from
[`_rtf_clipboard.py`](`_rtf_clipboard.py`), which walks all tag boundaries,
maps each tag's `foreground` to an 8-bit RGB via `winfo_rgb`, and emits a
single `{\cfN …}` segment per slice into an RTF `\\colortbl`.
Three formats are placed on the clipboard:

- **`CF_UNICODETEXT`** — plain text fallback for all targets
- **`CF_RTF`** — Word / Outlook preserve foreground colors
- **`HTML Format`** — CopyQ and other clipboard managers that prefer HTML over RTF

When `pywin32` is unavailable, falls back to `widget.clipboard_append(plain)`.

**Binding placement.** The binding is `self.root.bind("<<Copy>>", ...)`,
NOT `<Control-c>`.  Tk maps `<Control-Key-c>` → `<<Copy>>` via
`event add` at the C level — a `<Control-c>` binding is dead code on real
keypresses.  `_log` is `state='disabled'` → never gets keyboard focus →
widget-scoped binding would never fire.  Root `<<Copy>>` fires for any
focused widget.

`tksheet` binds `<Control-c>` (not `<<Copy>>`) on its canvas, so its
`ctrl_c` handler runs for `<Control-c>` dispatches but **not** for real
Ctrl+C keypresses (which generate `<<Copy>>`).  The `<<Copy>>` event
propagates through bindtags (widget → Canvas class → root → all) to our
root handler.  Python's `event_generate("<<Copy>>")` on a tksheet canvas
does NOT propagate (a tkinter quirk); Tcl-level `event generate` does.
Tests use Tcl-level dispatch to match reality.

`App._on_copy_rich` checks `_log.tag_ranges("sel")`: if the user has a
mouse selection on the disabled log, it serves RTF + HTML + plain via
`copy_rich` and returns `"break"` (suppresses the default Text/Entry
`<<Copy>>` so the focused widget doesn't overwrite the clipboard with plain
text); otherwise it returns `None` and the focused widget (e.g. `_path_field`
ttk.Entry) keeps its normal copy behaviour.

**Hardening.** `copy_rich` builds RTF + HTML payloads BEFORE touching the OS
clipboard (a failed build leaves the prior clipboard contents intact rather
than empty-but-nothing).  `win32clipboard.OpenClipboard` is retried with
backoff — it can raise `pywintypes.error` ("Отказано в доступе"/Access denied)
briefly while another viewer holds the clipboard or while Tk's own idle-time
clipboard propagation is in flight (`widget.update()` is called each retry to
drain Tk's pending clipboard writes).  On exhausted retries or `ImportError`
(no `pywin32`), falls back to Tk's plain-text clipboard so the user still gets
text — never an unhandled exception from Ctrl+C.

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

## TabRail (`_tab_rail.py`)

Vertical tab rail replacing the native ttk.Notebook tab row.  Two columns
share one vertical extent per config:

```
[progress column (PROG_W)] [tab column (TAB_W)]
  thin vertical fill          rotated label (angle=90, reads bottom→up)
  grows top→down              selection accent on notebook-facing edge
  color by state              dirty `*` suffix, ✔ on done
```

### Sizing policy

Ideal height = rotated label length + padding.  Three modes:
- **normal** — height by content;
- **surplus** — capped proportional grow (`min(GROW_CAP, 35%)`), remainder empty below;
- **shortage** — selected keeps ideal, inactive waterfill-compress to `MIN_H`;
- **extreme shortage** (``H < n * MIN_H``) — even split, no floor; selected gets
  a remainder pixel.  Every tab stays visible regardless of window size.

### Interaction

Tab column click → `_on_select(stem)` → `App._select_tab` → `frame.tkraise()`.
Progress column is display-only (`e.x >= scaled(PROG_W)` check).  Hover
highlights cell face; `_on_hover` callback updates status bar.

## Progress

Three layers: per-config fills on the rail, overall description in `_overall_lbl`,
and stage-level floater overlay — all driven by snapshots read every 300 ms in
`App._poll_progress`.

### Per-config fills — `ProgressBank` → `TabRail`

`ProgressBank` tracks each config's processing state (pending → running → done
/error) with stage-weighted fractional progress.  Stage weights are defined in
`progress_bank.WEIGHTS` (Processing dominates at 60%).  `canon_stage()` maps
free-form `stage_desc` text to canonical stages by 4-letter prefix.

Feeding:
- `cli.process_loading_yaml` calls `progress_bridge.set_cfg(stem)` per config
  so subsequent `stage_desc` / `GuiTqdm` ticks land in the correct bank cell.
- `stage_desc(desc)` (called at each processing phase boundary) updates
  `progress_overall` AND calls `bank.stage_start(current_cfg, canon_stage(desc))`.
- `GuiTqdm.update(n)` calls `bank.inner(current_cfg, n, total)` alongside
  `progress_stage.set`.

Rendering: `App._poll_progress` reads `bank.snapshot_all()` →
`rail.set_state(cfg, state, frac)` per config.  Aggregate % appended to
`_overall_lbl` text during run.  On completion, `bank.finish(stem, ok)` →
rail cell shows full fill (done) or error tint.

### Stage-level overlay — `progress_stage` + `GuiTqdm`

Per-item progress via `GuiTqdm` — a tqdm replacement that routes
`(n, total, desc)` to `progress_stage` (read by `App._poll_progress`).

Two code paths activate `GuiTqdm`:
- **Binning loop** (`physical.py`): `get_tqdm_class() or tqdm` — iterates bins,
  updating `progress_stage` per bin. GUI active → `GuiTqdm`, CLI → terminal tqdm.
- **Dask NC write** (`processing.py`): `TqdmCallback(tqdm_class=GuiTqdm)` —
  task-level progress during `.compute()` on dask arrays.

`physical.py` and `io.py` import `get_tqdm_class` with `try/except ImportError`
fallback (same pattern as `processing.py`'s `progress_bridge` import).

### Status bar layout (§6)

Two independent overlays on `root`, both at the bottom edge:

```
root
│
├── row=0: path field (§1)
│
├── row=1: overall status label (§2)
│   └── f1 (ttk.Frame)
│       ├── column=0: _overall_lbl (ttk.Label, text=_cfg_state enum value)
│       │   ScanStage.DEFAULT → "Default configuration"
│       │   ScanStage.DONE → completion text + aggregated % from bank
│       │   During run: stage_desc text + " — 62%" from bank snapshot
│       └── column=1: _prog_status (ttk.Label, anchor="e")
│           Current processing stage description (separate from tab rail).
│           Cleared when no stage is active.
│
├── row=2: main area (§3)
│   └── _main (ttk.Frame)
│       ├── column=0: _rail (TabRail) — vertical tab rail
│       │   ├── progress column (PROG_W) — per-config fills top→down
│       │   └── tab column — rotated labels, selection accent, dirty `*`
│       └── column=1: _stack (ttk.Frame) — page stack, tkraise() switching
│   └── _run_btn (ttk.Button, place(in_=self._main, relx=1.0, rely=1.0, anchor="se"))
│       floats at main area bottom-right with scrollbar margin
│
├── row=3: log (§5, weight=1)
│   └── _log_frame (ttk.Frame)
│       ├── _log (tk.Text, row=0, weight=1)   ← scrolling log
│       └── _log_vbar (ttk.Scrollbar, row=0 col=1)
│
│   (error tooltip is rendered in _status_lbl — no separate row)
│
├── place(rely=1.0, relx=0.0, anchor="sw")  ← bottom-left
│   └── _status_lbl (MarkdownLabel)
│       wrap="none" by default; switches to wrap="word" only if
│       content exceeds window width (_fit_width via font metrics).
│       Width contracts to text width.
│       Height = exact pixel via place_configure (dlineinfo walk).
│       Font: ui.font() copy of TkDefaultFont (same as _log via set_font).
│       mark_font_ready() enables auto-sizing; does NOT crush to bar height.
│
└── place(relx=1.0, rely=1.0, anchor="se")  ← bottom-right, hidden by default
    └── _prog_floater (tk.Frame, bg=match root)
        ├── _prog_stage_text (tk.Label, anchor="e", right-aligned)
        └── _prog_stage (ttk.Progressbar, length=220)
```

**Z-order competition**: `<Motion>` on root → `_status_lbl.lift()`.
`_prog_floater.lift()` on each `tot > 0` poll update.  `_run_btn.lift()`
on root `<Configure>`.  Last `lift()` wins.

**Show delay**: `_prog_floater` is shown via `root.after(400, _show_prog_floater)`
— avoids flashing for very short operations.  Cancelled if `tot` drops to 0
before the delay fires.

### Status bar text — one-shot clear signal

The status `StringVar` is **not** owned by `_poll_progress`.  Explicit setters
control it:

| Setter | When | Widget | Text |
|--------|------|--------|------|
| `_fit_status_font` | startup, no CLI args | `_status_lbl` | `"Ready"` (calls `mark_font_ready()` — does NOT crush font to bar height; status auto-grows for multi-line content) |
| `_fit_status_font` | startup, CLI args | `_prog_floater` | `"Loading…"` (immediate, no delay) |
| `_on_path_changed` | browse button / Enter | `_prog_floater` | `"Loading…"` (immediate, no delay) |
| `processing.run` | scan phases | `_prog_floater` | "Discovering…", "Generating…", "Composing {stem}…" |
| `process_loading_yaml` | per-config | `_prog_floater` | "Composing {stem}…" (via `_pb`) |
| `_on_scan_ok` | scan completion | `_status_lbl` | `"Ready"` |
| `_on_run_done` | run completion | `_status_lbl` | `"Done — {pct}% ({ok}/{n} ok)"` |
| `_on_path_hover_in` | mouse enters path field | `_status_lbl` | hover hint |
| `_poll_progress` (clear flag) | probe boundary | `_status_lbl` | `""` (one-shot) |

`_poll_progress` only writes `""` when `progress_stage.consume_clear()`
returns `True` — a one-shot flag set by `ProgressState.clear_and_reset()`
at each probe boundary in `processing.run_processing`.  This prevents
aggressive clearing of "Ready", "Done …", and hover hints during idle
and inter-probe gaps, while still wiping stale stage text from the
previous probe.

Decision matrix for `_status_lbl` clearing (``_any_hovering`` = ``_path_hovering``
or ``_nb_hovering`` or ``_chrome_hovering is not None`` or ``_browse_hovering``).  Stage `desc` is
written to `_prog_stage_text` (overlay) unconditionally — `_any_hovering`
only gates `_status_lbl` clearing.

| `progress_stage.tot` | `_clear_status` | `_any_hovering` | Action on `_status_lbl` |
|---|---|---|---|
| `> 0` | any | any | no-op — `desc` goes to overlay |
| `0` | `True` | `False` | **clear** — flag consumed |
| `0` | `True` | `True` | preserve — flag deferred |
| `0` | `False` | any | **preserve** — no-op |

### Chrome widget hover bindings

``_bind_chrome_hover`` (called once at end of ``_build``) adds
``<Motion>``/``<Leave>`` bindings to every widget registered by
``_register_chrome_help`` — Run button, progress bars, labels.  Widgets with
their own dedicated hover handling (``_path_field``, ``nb``, ``_log``) are skipped.

``_on_chrome_hover`` reads ``get_widget_meta(w, "status")`` (static string or
live callable) and writes to ``self._status``.  ``_on_chrome_leave`` clears
``_chrome_hovering`` so the poll cycle can resume writing status.

### Hover-hide for stage status + floater

``<Enter>`` on ``_prog_status``, ``_prog_floater``, ``_prog_stage_text``, or
``_prog_stage`` fires ``_on_status_enter``, which:
1. Sets ``_status_hovering = True``
2. Records the union bbox of visible status widgets (``_status_bounds``)
3. ``grid_remove()``s ``_prog_status`` + ``place_forget()``s ``_prog_floater``

Hidden widgets can't fire ``<Leave>``, so root ``<Motion>`` +
``_on_motion_check_hover`` detects leave: when the mouse exits the recorded
bbox, ``_status_hovering`` resets to ``False``.  ``_poll_progress`` then
restores both on the next 300 ms tick (if ``tot > 0``).

``winfo_ismapped()`` is not used for place-managed widgets (unreliable before
window realization and on withdrawn test roots); ``place_info()`` /
``grid_info()`` are used instead.

### Error tooltip in `_status_lbl`

``_show_tip`` sets ``_tip_active = True`` and renders markdown directly in
``_status_lbl``.  While active, ``_set_status`` is a no-op — all chrome-hover,
poll, and log-motion status updates are suppressed.  Dismissed by ``_hide_tip``
on: new scan/run (``_clear_log`` / ``_on_scan_ok`` / ``_on_run_done``), path
change (``_on_path_changed``), ``<Escape>`` (root binding), or cell edit begin
(``ConfigSheet.on_edit_begin`` → ``_hide_tip``; fired from
``_on_begin_edit_cell`` and ``_on_field_edit_start``, excluding the Top
PathField which is a ``PathField``, not a ``ConfigSheet``).

## CLI integration

GUI accepts CLI args: first positional = data path (prefills GUI entry, auto-scans), then
`key=value` = Hydra overrides passed verbatim via `sys.argv`
(see [`cli.py` internals](../tcm_cli/how_it_works.md#entry-point) for
`call_in_raw_dir`, `parse_data_path`, `hydra_main`)
Example: `python -m tcm_gui "D:/data/_raw/@i_p1.TXT" "input.ids=[i90]"`.

`App.__init__` stores `self._original_argv`.  Both **Scan** and **Run** feed
the **live path-field value** to `call_in_raw_dir` as `input.path` (OmegaConf
merge — bypasses Hydra's ANTLR parser, the documented safe channel for paths
carrying `@`/`:`/`,` as in `D:/data/_raw/@i_p1.TXT`).  `Worker._setup` strips
the positional path from `original_argv` via `cli.parse_data_path`, so only
the launch-time `key=value` overrides remain in `sys.argv` and survive
rescans after a GUI browse selection.  Without this, the stale startup
positional would leak into Hydra's override parser (the `@`-crash guarded by
`TestGuiAtSignFilename`).  For **Run**, `original_argv` is `["__main__"]`:
the user-edited YAML files are the sole config source.
