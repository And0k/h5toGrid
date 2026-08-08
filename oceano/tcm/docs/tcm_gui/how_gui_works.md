# How the GUI works — Internal Architecture

Optional Tkinter frontend wrapping `tcm.cli.call_in_raw_dir` in a background
thread.  No custom CLI parsing — Hydra handles all config keys natively via
`sys.argv` (see [CLI usage](../tcm_clc/README.md#quick-start)).

## Architecture

| File | Purpose |
|------|---------|
| `app.py` | Tk root, layout §1–6, 300 ms polling, argv prefill, `_initial_scan` flag (immediate overlay show), `_prog_floater` overlay (400 ms delay for run), z-order `<Motion>` bind; manual `ttk.Frame` + `tk.Text` + `ttk.Scrollbar` log container (replaces `ScrolledText` for ttk-styled scrollbar); `_log_autoscroll` flag + `<MouseWheel>`/`<Button-4/5>` bindings for scroll-aware auto-follow |
| `md_label.py` | `MarkdownLabel` (`tk.Text` subclass): Tk renderer for Markdown AST from `_md_parse`; font scaling (`fit_to_height`), dynamic width (`_fit_width` via font metrics, `wrap="none"` → `wrap="word"`), auto-height (`_fit_height` on `<Configure>`), table tab-stop alignment |
| `_md_parse.py` | Pure Markdown parser (zero Tk dependency): `parse_inline()`, `parse_markdown()`, `split_table_row()`; AST types `Heading`/`Paragraph`/`CodeBlock`/`Table`/`Inline` |
| `worker.py` | Background thread: `call_in_raw_dir` for Scan and Run |
| `coef_sheet.py` | tksheet treeview: type-aware widgets (checkbox/dropdown/align), node + metadata bg; row-geometry-free styling via `_row_map()`; floated `PathField` hover-edit on browse rows |
| `_path_field.py` | 1×1 tksheet for display + `ttk.Entry` overlay for editing — frame-anchored hover button, column-width tracking via `<Configure>` |
| `_browse_button.py` | `BrowseOverlay` (widget core + `pending` state), `BrowseButtonManager` (sheet-edit policy + injectable `editor_place`), `SheetHoverBinder` (MT motion → overlay show/hide with pending-aware veto), `bind_hover_browse` (Entry legacy) |
| `_cell_spec.py` | Hydra dataclass → ``CellSpec`` (bool/enum/text/number/date) for cell rendering |
| `_help.py` | Auto-extract config-cell help from ``config_reference.md`` tables (``HelpEntry``, ``help_for_path``, ``parse_reference``); index-stripping for arrays (``Ag[0]`` → ``Ag``); ``lru_cache``-memoized loader |
| `const.py` | Centralized colors & styles; `apply_ui_scale` (DPI + named fonts); `apply_theme_defaults` (Windows dark/light registry → all `FUNC_COLOR`, `TAG_COLORS`, `FG_DEFAULT`, `DEFAULT_FG`, `BLUE_FG`, `FRAME_BG_FALLBACK`, `ENTRY_BG_FALLBACK`, `CELL_NON_DATA_BG`, `THEME`); `_apply_ttk_dark` (clam theme + ttk.Style dark configure + `App.Vertical.TScrollbar` scrollbar style); `_opt_into_dark_titlebar` (`DwmSetWindowAttribute(DWMWA_USE_IMMERSIVE_DARK_MODE)` via `GetAncestor(GA_ROOT)` for real toplevel HWND); `widget_meta` registry; `STR` i18n surface; `get_widget_meta` (callable-resolving) |
| `cli_cfg.py` | `CFG_DEFAULTS` (config-tree defaults) + `COEF_SHAPES` (auto-derived) + `COEFS_TYPE` — all derived from `Config` via `get_type_hints`, no per-section imports |
| `progress_bridge.py` | `GuiTqdm` (tqdm replacement) + module-level runtime injection |
| `log_bridge.py` | `install()` once at App startup → root logger captures GUI-thread AND worker logs → `QueueHandler` (consecutive dedup + emit-time text freeze) → `tk.Text` drain |
| `_rtf_clipboard.py` | `Ctrl+C` on log → RTF + plain text on clipboard (colors preserved) |
| `runtime.py` | Shared state: queues, `ProgressState` (with one-shot `clear_and_reset`/`consume_clear`), `PauseGate`, persistent `queue_handler` reference |

## Data flow

### Scan

```
Browse / Enter input.path
  → app._clear_log (flush queue + clear tk.Text log)
  → worker._scan (thread)
    → call_in_raw_dir(processing.run,
        input={path: live-path-field}, return_="<cfg_from_args>")
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
    → app._clear_log (flush queue + clear tk.Text log)
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
| `QueueHandler` installed once at App startup; re-attached in `_wrap` after Hydra `dictConfig` | Hydra's ``logging.config.dictConfig`` replaces **all** root handlers with ``[console, file]`` each worker task, removing the ``QueueHandler`` from root.  ``_wrap.wrapped`` (running *after* dictConfig) re-adds it so both worker-thread and GUI-main-thread logs (e.g. ``_reload_coefs`` triggered by treeview interaction) reach the log ``tk.Text`` widget.  ``reset_dedup()`` per task prevents the first record of a new task from being swallowed as a "duplicate" of the previous task's tail |
| `_runtime` is module-level, not `threading.local` | `TqdmCallback` creates `GuiTqdm` in dask worker threads |
| `return_="<cfg_from_args>"` for Scan | pipeline does discovery + gen_metadata, returns configs without processing |
| `input.yaml_path` for Run | documented regex filter, skip discovery, only selected configs |
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
| `_fg_default` — theme foreground color | `const.FG_DEFAULT` (set by `apply_theme_defaults`); applied explicitly (never `fg=None`, which is a per-key merge no-op in tksheet 7.x) |
| `_apply_edit_value` / `_apply_default_fg` use `overwrite=False` | edit-time restylers pass only `fg` to `highlight_cells`; `overwrite=False` preserves the `bg` that `_apply_styles` set (input.row data cells keep button-face after edits) |
| **Input row styling** | node label: button-face bg + normal black `FG_DEFAULT` (never blue/gray toggle); all data cells: button-face bg via `highlight_cells` across `total_columns()`.  Other rows: button-face bg + `BLUE_FG`/`_fg_default` node fg as before |
| **PathField styling** | `const.ENTRY_BG_FALLBACK` bg + `const.FG_DEFAULT` fg + **bold** font (sheet-wide, 1×1 cell); right-aligned (`align="e"`): long paths show filename at the right edge; entry-field silhouette distinct from the gray coef_sheet cells; **editing via `ttk.Entry` overlay** (veto tksheet's `tk.Text`), `justify="right"` |
| **Blue node labels** → subtree unchanged | `_node_at_default(id)` recurs: every leaf value matches its config dataclass default; `const.BLUE_FG = "#0055CC"` on index canvas |
| `_on_end_edit` → cascade toggle | Gray/clear fg per cell **+** walk ancestral tree labels (blue/standard); `_fg_default` used for clear side (not `fg=None`) |
| dirty tracking via `_data_snapshot` | `tuple(tuple(str(val) for val in row) for row in sheet)` covers ALL editable cells (not just coefs); `is_dirty` compares current vs snap |
| `"*"` on tab title (300 ms poll) | visual feedback for unsaved edits; removed by `mark_clean()` after write |
| `_write_coefs` skips clean tabs | avoids redundant timestamped backups identical to existing YAML |
| `_clear_log` on scan/run start | prevents cross-operation message accumulation in log ``tk.Text`` widget |
| `_log_autoscroll` flag + scroll bindings | persistent flag (not `yview()` threshold — `see("end")` yields ~0.91–0.98, never 1.0); starts `True`, cleared by `<MouseWheel>`/`<Button-4/5>` when `after_idle` check finds `yview()[1] < 0.90`, restored when user scrolls back to bottom; `_poll_logs` calls `see("end")` only when flag is `True` |
| `_clear_status` one-shot flag (not blanket clear) | original `_poll_progress` set `_status.set("")` every 300 ms when `tot == 0`, wiping "Ready", "Done …", and hover hints.  `clear_and_reset()` (worker, at probe start) + `consume_clear()` (GUI, once) replaces continuous clearing with a single event per probe boundary.  `_path_hovering` guard defers consumption while hover is active |
| `QueueHandler` consecutive dedup | drops equivalent records (same msg at same call site), registered by `funcName+msg` key.  **freezes** the rendered text onto the `LogRecord` (`rec.msg = text; rec.args = ()`) at emit time so deferred `drain`-time `getMessage()` cannot be corrupted by the mutable `Message` reused across log calls in `LoggingStyleAdapter`.  Mirrors Hydra's `job_logging/colorlog` formatter, which renders `record.getMessage()` once synchronously. |
| `Ctrl+C` → RTF + plain on clipboard | `_rtf_clipboard.copy_rich` serializes tag-colored log ``tk.Text`` widget; pywin32 absent → plain fallback |
| `config.Config` + `config.Return` passed to `load()` | structured-config root + `StrEnum` for `program.return_` dropdown |
| `_cell_spec_for` → bool/enum/text/number | walks dataclass tree via `spec_for_path`; `bool` → checkbox, `Enum` → dropdown, `str`/`Path` → left-align |
| node column bg = header bg | `highlight_cells(canvas="index")` in `_apply_styles`; `resolved_frame_bg()` (TFrame background) for all rows including `input` |
| metadata row bg up to last date cell | all cells from col 0 through last `meta_date_cols` entry share the bg |
| ordering `_apply_open()` → `_row_map()` → `_apply_styles()` → `_apply_default_fg()` | invariant: build tree → set open states → compute row map → apply styles → gray defaults → redraw |
| `date` independent of `max_col` | coefs parent has `max_col=0`; styling in dedicated section before `max_col` loop |
| PathField = 1×1 Sheet for display, `ttk.Entry` for editing | Display: cell-behavior parity (right-align, overflow).  Edit: `ttk.Entry` is inherently single-line (no wrapping), native horizontal scroll, cursor always visible.  tksheet's `tk.Text` editor cannot disable wrapping (`table_wrap` is display-only).  Veto via `return None` from `begin_edit_cell` callback |
| `SheetHoverBinder` extracted from ConfigSheet | three MT binds + churn veto reusable by PathField and any future sheet-hover site |
| `_hover_resolve` uses `_iid_of_row` cache | O(1) lookup on every `<Motion>` event; rebuilt in `load()` (stable between loads) |
| `_hover_resolve` publishes status for ALL rows | not just browse rows; status clears at `<Leave>` |
| `_hover_resolve` handles identify_row API drift | tries `identify_row(event)` first (7.x), falls back to `identify_row(event.y)` (older) |
| tree column hover on RI canvas | `_on_tree_motion` bound to `self.sh.RI`; shows section-level help; separate from MT data-cell hover |
| `_status_source` tracks hover canvas | `"tree"` (RI) / `"data"` (MT) — re-publishes status on source change for same row |
| `_any_hovering` property | combines `_path_hovering`, `_nb_hovering`, `_chrome_hovering` — single guard against poll clobbering |
| `_bind_chrome_hover` wires status to Run/progress/labels | `<Motion>`/`<Leave>` on all registered chrome widgets; skips `_path_field` + `nb` (own handlers) |
| **Floated PathField on browse rows** | one reusable `PathField` for text + separate `BrowseOverlay` for button; intent-delayed (120 ms); focus strictly opt-in; full edit parity free; button stays at sheet right edge while field text stops at button's left edge; `_do_field_hide` vetoes hide during `f._editing`; `_on_field_edit_end` → `_restore_hover_placement` (show button first, `update_idletasks`, then `f.place` at shortened width) |
| `_field_iid` survives hide | `_hide_hover_field` keeps `_field_iid` — `PathField._notify` queues via `after_idle`, so a commit in flight still writes to its row; `_hover_btn` (browse button) is hidden separately |

## Dark / light theme architecture

Three layers cooperate to render the entire GUI in a consistent dark or light
palette.  `const.apply_theme_defaults(root)` runs once at startup (before any
widget is created) and orchestrates all three.

Startup flow:
```
App.__init__
  → apply_ui_scale(root)          # DPI + named fonts
  → apply_theme_defaults(root)    # detect theme → mutate globals → ttk.Style → root.bg
  → _build()                      # widgets created with correct const values
    → PathField(sheet uses ENTRY_BG_FALLBACK at construction)
    → ConfigSheet created on scan
      → __init__: change_theme("dark") if THEME == "dark"
```

| Layer | What it styles | Mechanism |
|---|---|---|
| **const globals** | Log tags, per-cell highlights, log ``tk.Text`` bg/fg, `MarkdownLabel` bg/fg, `tk.Frame`/`tk.Label` bg/fg | `_DARK` / `_LIGHT` palettes → `setattr` on module globals (`FUNC_COLOR`, `DEFAULT_FG`, `BLUE_FG`, `FG_DEFAULT`, `FRAME_BG_FALLBACK`, `ENTRY_BG_FALLBACK`, `CELL_NON_DATA_BG`, `THEME`) + `TAG_COLORS.update()` |
| **ttk.Style** | All `ttk.Frame`, `ttk.Label`, `ttk.Button`, `ttk.Entry`, `ttk.Notebook`, `ttk.Progressbar` | `_apply_ttk_dark(root)` → switches to ``clam`` theme (native themes ``vista``/``xpnative`` ignore ``Style().configure()`` for rendering), then ``ttk.Style().configure()`` with bg/fg from const globals + ``style.map()`` for active/selected states; root window ``bg`` set directly |
| **tksheet** | Sheet canvas (table, header, index, scrollbars, selection) | `ConfigSheet.__init__` calls `self.sh.change_theme("dark")` when `const.THEME == "dark"` + `scrollbar_theme_inheritance="clam"` so tksheet's canvas scrollbars match the `App.Vertical.TScrollbar` ttk style; `PathField` uses explicit `table_bg`/`table_fg` from const at construction (no `change_theme` needed — headers/index/scrollbars hidden) |

### Widget-specific notes

| Widget | bg/fg source |
|---|---|
| `tk.Text` + `ttk.Scrollbar` (log) | `bg=const.ENTRY_BG_FALLBACK`, `fg=const.FG_DEFAULT`, `insertbackground=const.FG_DEFAULT`; manual container replaces `ScrolledText` to get a real `ttk.Scrollbar` |
| Log scrollbar | `ttk.Scrollbar` with `style="App.Vertical.TScrollbar"` — matches tksheet via shared `clam` theme inheritance |
| `MarkdownLabel` (status) | `background=const.FRAME_BG_FALLBACK`, `foreground=const.FG_DEFAULT` |
| `tk.Frame` + `tk.Label` (prog_floater) | `bg=const.FRAME_BG_FALLBACK`, `fg=const.FG_DEFAULT` |
| `ConfigSheet` (tksheet) | `change_theme("dark")` + `scrollbar_theme_inheritance="clam"` in `__init__`; `_apply_styles` uses `const.resolved_frame_bg()` + `const.FG_DEFAULT` |
| `PathField` (1×1 tksheet) | `table_bg=const.ENTRY_BG_FALLBACK`, `table_fg=const.FG_DEFAULT` — set at construction |
| `ttk.Entry` (PathField editor) | inherits from `ttk.Style("TEntry")` dark configuration |
| Root window + title bar | `root.configure(bg=...)` + `GetAncestor(winfo_id(), GA_ROOT)` to get real toplevel HWND (Tk's `winfo_id()` returns a child widget, not the DWM-controlled frame) + `DwmSetWindowAttribute(DWMWA_USE_IMMERSIVE_DARK_MODE=TRUE)` via `ctypes.WinDLL("dwmapi")` (Win32 only, Win11 22000+) |
| `ttk.Notebook` + tabs | `style.configure("TNotebook.Tab", ...)` + `style.map` for selected state |
| `ttk.Button` (Run) | `style.configure("TButton", ...)` + `style.map` for active/pressed |
| **Scrollbars** | Log `ttk.Scrollbar` + tksheet internal scrollbars | `App.Vertical.TScrollbar` ttk style configured in `_apply_ttk_dark` (dark) / default clam (light); tksheet uses `scrollbar_theme_inheritance="clam"` so its canvas scrollbars inherit the same ttk theme; log uses manual `ttk.Frame` + `tk.Text` + `ttk.Scrollbar` instead of `ScrolledText` (which uses an unstyled classic `tk.Scrollbar`) |

## Help system architecture

Two independent help sources — one per widget category:

| Source | Widgets | Key derivation | i18n mechanism |
|---|---|---|---|
| `STR` (``const.py``) | Chrome widgets (`self._path_lbl`, `_path_field`, `_cfg_lbl`, `_run_btn`, `_status_lbl`) + dynamic tabs | Attribute name → role → ``STR["{role}.tooltip"]`` / ``STR["{role}.status"]`` | Replace ``STR`` dict wholesale at build for target language |
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
Tab hover is wired via ``<Motion>`` / ``<Leave>`` bindings on ``self.nb`` that
use ``nb.identify(x, y)`` + ``nb.index(f"@{x},{y}")`` to find the tab frame
and read its status from ``widget_meta``.

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
   ``HelpEntry(path="{section}.{field}", short=<last cell>)``.
3. CamelCase field names (``Ag``, ``Cg``, ``Rz``) parse identically to
   lowercase Hydra names.
4. ``_DOC_PATH`` resolves to ``config_reference.md`` at
   ``{tcm_root.parent}/docs/tcm_clc/config_reference.md``; absent file →
   empty cache → no hover text (graceful degradation).
5. Array indices stripped at lookup time: ``Ag[0]`` / ``Ag[1][2]`` → ``Ag``.

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

### Focus prevention

The button overrides `focus_set` to no-op and sets `takefocus=False`.
Clicking it must not steal focus from the TextEditor (tksheet closes the
editor on `<FocusOut>`).  `<Button-1>` returns `"break"` to suppress
default focus-change behavior.

## PathField (`_path_field.py`)

A 1×1 tksheet for **display** + a `ttk.Entry` overlay for **editing**.

**Display mode**: column=4096, `align="e"`, `xview_moveto(1.0)` — long
paths show the filename at the right edge.  White `ENTRY_BG_FALLBACK`
background, bold font, all chrome hidden.  Normal black `FG_DEFAULT`
foreground (not `BLUE_FG`).

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
      _ov.hide()                  # hide browse overlay
      _on_begin_edit_cb()         # coef_sheet: cancel hide, hide btn, expand field
      constrain column to frame   # display column is 4096; edit needs frame-width
      _open_entry()               # ttk.Entry fills PathField frame
      return None                 # VETO tksheet's tk.Text editor

_Enter_ → _commit_entry:
      read Entry value
      destroy Entry
      _editing = False
      _on_end_edit_cb()           # coef_sheet: _restore_hover_placement
      set_cell_data(0, 0, value)  # if not cancel
      restore column to 4096
      _scroll_to_right()          # filename visible again
      _notify(value)              # if changed

_Escape_ → _commit_entry(cancel=True):
      destroy Entry
      _editing = False
      _on_end_edit_cb()           # coef_sheet: _restore_hover_placement
      restore column, scroll right
      no _notify (pre_edit == current)
```

### `<Configure>` guard

`_on_configure` only calls `_scroll_to_right()` when `_editing` is
`False`.  During edit the column is constrained — scrolling right would
move the editor off-screen.

### Geometry: frame-anchored hover + column tracking

The hover button is parented to the `PathField` frame (not the sheet
canvas), placed via `place(in_=self, relx=1.0, anchor="e")`.  This
decouples the button from cell width — `relx=1.0` tracks every resize
with zero bindings.  The column width is kept in sync with the frame
via `<Configure>` → `column_width(0, event.width - 2)`, so the editor
spans the field exactly and its right border stays visible.

After edit ends (`_on_end_edit`), `_ov.schedule_show(_place_kw())`
re-arms the hover button immediately — pointer is over the field by
construction, so no mouse move is needed.

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

### Status bar layout (§6)

Two independent overlays on `root`, both at the bottom edge:

```
root (no f4 — removed)
│
├── place(rely=1.0, relx=0.0, anchor="sw")  ← bottom-left
│   └── _status_lbl (MarkdownLabel)
│       wrap="none" by default; switches to wrap="word" only if
│       content exceeds window width (_fit_width via font metrics).
│       Width contracts to text width.  Height auto-grows via _fit_height.
│
└── place(relx=1.0, rely=1.0, anchor="se")  ← bottom-right, hidden by default
    └── _prog_floater (tk.Frame, bg=match root)
        ├── _prog_stage_text (tk.Label, anchor="e", right-aligned)
        └── _prog_stage (ttk.Progressbar, length=220)
```

**Z-order competition**: `<Motion>` on root → `_status_lbl.lift()`.
`_prog_floater.lift()` on each `tot > 0` poll update.  Last `lift()` wins.

**Show delay**: `_prog_floater` is shown via `root.after(400, _show_prog_floater)`
— avoids flashing for very short operations.  Cancelled if `tot` drops to 0
before the delay fires.

### Status bar text — one-shot clear signal

The status `StringVar` is **not** owned by `_poll_progress`.  Explicit setters
control it:

| Setter | When | Widget | Text |
|--------|------|--------|------|
| `_fit_status_font` | startup, no CLI args | `_status_lbl` | `"Ready"` |
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
or ``_nb_hovering`` or ``_chrome_hovering is not None``).  Stage `desc` is
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
their own dedicated hover handling (``_path_field``, ``nb``) are skipped.

``_on_chrome_hover`` reads ``get_widget_meta(w, "status")`` (static string or
live callable) and writes to ``self._status``.  ``_on_chrome_leave`` clears
``_chrome_hovering`` so the poll cycle can resume writing status.

## CLI integration

GUI accepts CLI args: first positional = data path (prefills GUI entry, auto-scans), then
`key=value` = Hydra overrides passed verbatim via `sys.argv`
(see [`cli.py` internals](../tcm_clc/how_it_works.md#entry-point) for
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
