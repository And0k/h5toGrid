# GUI Architecture

Optional Tkinter frontend wrapping `tcm.cli.call_in_raw_dir` in a background
thread.  No custom CLI parsing — Hydra handles all config keys natively via
`sys.argv` (see [CLI entry point](../../../readme.md)).

Companion pages: [GUI Widgets](widgets.md) · [GUI Help System](help_system.md) ·
[GUI Key Decisions with Rationale and Regression Notes](decisions.md) — index: [GUI Internals](_index.md).

## Architecture

File - Purpose
--------------

### `app.py`

Tk root, layout §1–6, 300 ms polling, argv prefill, startup placement via `App.GEOMETRY` → `const.fit_to_workarea` (size clamped + centered in the taskbar-excluded work area, chrome re-fit on `<Map>` — never off-screen at start); the root is `withdraw()`n right after creation and `_build()` runs unmapped — `fit_to_workarea`'s `deiconify()` is the FIRST show (no top-left default-size blink; same pattern as `AboutDialog`), `Alt+Arrows` nudges the window 10px (`Shift` ×10) via `const.nudge_window` — a shell drag can't carry the title bar above the screen top (OS clamps ALL apps' interactive drags there; programmatic moves aren't limited), `_initial_scan` flag (immediate stage-row show), collapsible §2 status row (`_overall_lbl` + `_prog_stage` + `_prog_stage_text` gridded together, 400 ms delay for run); manual `ttk.Frame` + `tk.Text` + `ttk.Scrollbar` log container (replaces `ScrolledText` for ttk-styled scrollbar); `_log_autoscroll` flag + `<MouseWheel>`/`<Button-4/5>` bindings for scroll-aware auto-follow; `_cfg_state: ScanStage` enum drives dual-purpose label at row=1; Run button floats via `place(in_=self._main)` (root `<Configure>` stacking via `_raise_overlays`: Run above pages, open anchor list above Run); page stack + `tkraise()` (no Notebook); **§1 search path row**: `path_lbl` + `path_field` + vertical separator + `_button_bar` frame (extensible container) with `?` help button (opens `AboutDialog`); `_set_cfg_ui_disabled` — inert look (dim rail + caption) in simple mode until configs exist: enabled by `_on_path_changed`/`_on_scan_ok`, re-dimmed by `_on_scan_error` when `not _yaml_paths`; root `<F1>` dispatch (`_on_f1_help` + `_within` master walk): focus inside `_path_field` → `path_field` anchor, else mouse over ``_status_lbl`` → ``_status_lbl_f1_anchor`` forged at display time (``_apply_status``/``_show_dwell_tip``/``_show_tip`` keep anchor in sync with what is shown), else the focused (or current) page's `ConfigSheet._f1_anchor`, else the localized readme (`_about.local_readme`) — one binding for all tabs (per-sheet bindings fired once per opened page)

### `md_label.py`

`MarkdownLabel` (`tk.Text` subclass): Tk renderer for Markdown AST from `_md_parse`; `_current` holds parsed `Block` tuple (not raw text) — `set_text(text, raw=False, base=None)` parses Markdown by default so `STR["{role}.status"]` with `**bold**` renders bold; `raw=True` bypasses parsing for paths/keys that interpolate untrusted content (e.g. `tab.status` after `.format(path=...)`) **and disables bare-path auto-linking** (`_autolink` flag) — a path revealed in the status bar (e.g. a log-hover target) must display as plain text and never re-link into a hyperlink; `rerender()` replays `_render` without reparse; `mark_font_ready()` (enables auto-sizing without resizing), `fit_to_height` (rescales + enables), dynamic width (`_fit_width` via font metrics, `wrap="none"` → `wrap="word"`), auto-height (`_fit_height` on `<Configure>`), table tab-stop alignment.  **Inline links are first-class**: ``[text](url)`` spans carry the URL as their span tag; `_insert_inline` styles them with the ``link`` tag (`theme.LINK_FG` + underline, raised above base tags), the tag's `<Enter>`/`<Leave>` bindings switch the widget cursor to `hand2`, and `<Button-1>` forwards `(url, base)` to the `on_link` callback (App and the About dialog pass `open_md_link`, no lambdas); link ranges are recorded in `_links` and `link_at(x, y)` resolves the URL under a point.  `set_text(..., base=…)` stores the source doc's directory so relative link targets resolve on click — dwell/error tips and every debounced status apply (`_apply_status`) pass `_help.doc_path().parent`.  **Bare filesystem paths auto-link**: link-free spans are split by `_path_spans` using `tcm_gui._allowed_paths` — absolute paths beneath the `allowed_dir` provider value (App passes `lambda: self._path_field.get()`; `''`/`None` → any absolute path) become link spans: files (2–4 alnum extension) display shrunk to the file name, directories (existence-checked) in full — the full path rides in a `file:` URI tag; explicit `[text](url)` spans pass through untouched; path links register in `_links` like parser links (clicks + Ctrl+C hyperlink export).  **Link hover row**: `set_hover_line(text)` appends the link target as a **plain** raw span below the rendered text — paths never re-enter Markdown parsing (``_raw``-style names intact) and are **not re-linked** (no `link` tag, no click), the row survives same-content re-renders (font rescale), but `set_text` clears it on any *new* content; `''` removes it.  `bind_link_hover(widget, show)` / `make_link_hover_handler` (extracted from the About dialog) publish the URL under the pointer; the status-label use keeps the row for in-widget motion (`clear_on_off_link=False`), the About/log uses clear off-link.

### `_md_parse.py`

Pure Markdown parser (zero Tk dependency): `parse_inline()`, `parse_markdown()`, `split_table_row()`; AST types `Heading`/`Paragraph`/`CodeBlock`/`Table`/`Inline`.  Inline ``[text](url)`` links parse recursively — the **URL tag unions onto every nested span** of the link text (``[`code`](url)`` → ``(code, {code, url})``: code font under one clickable link) — so the renderer tells styles apart (their own font variants) from links (any other tag), and URLs are never measured as text.

### `worker.py`

Background thread: `call_in_raw_dir` for Scan and Run.
In simplified mode (`Runtime.full_mode`, set from `App._full_mode`) it injects
`out = SIMPLE_OUT_DEFAULTS` (`cli_cfg.py` — `dt_bins=[0]`,
`dt_bins_min_save_text=0`) into both compositions, so Scan writes those values
into the generated run YAMLs and Run re-applies them over YAMLs lacking the
override.  Skipped when `sys.argv[1:]` (the surviving launch-time overrides,
see [CLI integration](#cli-integration)) mentions `dt_bins` — CLI args win —
and in full mode, which keeps the `schema.ConfigOut_InclProc` defaults.

### `states.py` (tcm)

`ScanStage(StrEnum)` — scan lifecycle labels (`DEFAULT`, `SCAN`, `DONE`), value = `scan_stage.*` i18n key in `str.yaml`; `Stage(StrEnum)` — per-probe processing phase labels, value = display text

### `coef_sheet.py` — composition root + tree/edit/row-space core

Wires the tksheet widget, builds the tree from a config dict and owns row-space resolution (`_row_map` internal vs display), item-hook open-state oracle, edit lifecycle (`_on_edit`/`_on_begin_edit_cell`/`_on_end_edit_cell`) and dirty tracking — **two independent flags** (separate Run writes): `is_dirty` covers coefs (values + dates + `input.path`; `_data_snapshot()` vs `_snap`, both iid-keyed, retaken together by `load()`/`mark_clean()`/`_rebuild_metadata_rows()`), and `is_metadata_dirty` covers the metadata node.  `_data_snapshot()` skips `is_metadata*` rows so a metadata edit never marks the coefs dirty (no spurious run-YAML rewrite).  It captures the date cell (tksheet col `_DATE_PH_COL`) of date-only coef rows (`coefs`/2d/1d parents, `max_col=0`) via `_cell_str`, so a coef-date edit marks the tab dirty.  `is_metadata_dirty()` is True when metadata rows differ from the snapshot **or** when the metadata was autofilled from an absent `info_devices.yaml` (`_metadata_unsaved` set in `_build_metadata` when `md_list is None` **and** the derived `metadata` path is non-empty; the old guard `any(not is_placeholder(v) …)` missed the empty-stub case with no `time_ranges` to seed, and the unconditional `autofilled` kept the default page — no `input.path` → `_path==""` — dirty).  A new stub for a real cruise is dirty from load so the tab shows `*` in the top-left corner (`_apply_metadata_dirty_label` is now called at the end of `load()`/`_rebuild_metadata_rows()` after the snapshots — previously it only ran on edit, so an autofilled dirty on load never showed `metadata*`) and `App._write_metadata`'s dirty branch picks it up (the `file_absent` fallback remains as safety); the default page stays clean.  `App._poll_dirty_tabs()` forwards `is_dirty` and `is_metadata_dirty` separately to `TabRail.set_dirty(dc, dm)` — both members must be **called**; a bare bound method is always truthy, so referencing `is_metadata_dirty` without `()` would store the method object in rail state and render `*` on every tab forever.  Composes three mixins (external imports unchanged): `_sheet_tint.SheetTintMixin`, `_sheet_status.SheetHoverMixin`, `_sheet_styles.SheetStylesMixin`.  **Metadata node** — top-level `metadata <info_devices.yaml>` (browseable path row, `check=exists`, `browse=True`) + 6 paired children (`point,symbol|sea depth,h_above|lat,lon|time_range|burst_dt/t|comment` from `tcm/_meta_pairs.PAIRS` 11-array).  Empty cells show gray example ghosts via `CellPlaceholder` (`_meta_pairs.EXAMPLES`: `P3, 7.5, ↟, 54.62, 19.84, 2026-07-11T12:20:12, 60, 600, deployment note` — `get_edited_metadata` reads ghost as `""` → `~`), `time_range` columns are `has_date` validated.  Public getters (`get_edited_input_path`/`is_path_valid`/`get_edited_metadata`) read via `_cell_str` — ghost never leaks as real data; deletion via the floated field commits `""` and the ghost is restored by `_restore_placeholder`.  **Since metadata-node split:** the whole node moved to the `MetadataNodeMixin` in `tcm_gui/_sheet_metadata_node.py` (module docstring is the contract) — `coef_sheet.ConfigSheet` mixes it in and calls `_init_metadata_node()`; the model is `_setups = [[num, 11-array], …]` with `_metadata` a property alias for the first array (scan/App call sites unchanged).  A single interval renders flat paired rows; several (nested `info_devices.yaml` entry) render autonumbered `setup` sublevels labelled by station keys.  `App._write_metadata` writes `get_edited_metadata_map()` (all station keys, preserving nesting) instead of one sid derived from stem order.  The sheet's built-in *Insert rows above/below* (`MT.rc_add_rows`) is intercepted via the mixin's `split_setup` (metadata root or `setup` node selected; `_rc_sel_iid` recorded in `_on_cell_select` as the rc target oracle): the copy is pinned to the shared boundary (`above` ⇒ `time_range[1] := time_range[0]`, `below` ⇒ `[0] := [1]`) and numbered with the next free integer.  Menu integration is patch-once in `tcm_gui/_sheet_popup.py` (`install_menu_patch`: sort bindings off, append-at-end extras, per-popup *Insert setup N above/below* relabel via the `MT/RI.extra_rc_func` pre-popup hooks, i18n `sheet.insert_setup_above/below`); each split coalesces into one native-chronology undo step through `tcm_gui/_sheet_undo.py::GroupUndoBridge` (`_snapshot_group`/`_restore_group` protocol, no second undo system).

**Cross-tab metadata sync** — when several tabs reference the same device file + probe id (pcid), a metadata mutation on any one of them fans out to all same-identity peers instantly.  Identity key: ``(device_path, pcid)`` — the *edited* root cell (browsing a file on one tab detaches it) and the canonical probe id (different file or different probe ⇒ independent configs).  ``App._on_metadata_changed`` resolves the key via ``_metadata_identity`` and reads the source's edited cell values via ``get_edited_metadata_setups`` (NOT ``src._setups`` — the internal model is only updated on structural changes, while cell edits live in the tksheet), then calls each peer's ``apply_metadata_setups(setups, dirty=src.is_metadata_dirty())`` — the dirty flag propagates so a clean peer becomes dirty when it receives an edit, and ``_apply_metadata_dirty_label`` refreshes the ``metadata*`` tree label; no echo via ``_sync_guard``.  Publication points: value edit (``_on_end_edit_cell`` skips coef-only edits via ``_meta_snap`` hash guard; root-path browse is skipped — its reload notifies after rebuild), split (``split_setup``), native undo/redo of value edits (``_sheet_undo.GroupUndoBridge._hook``), group undo/redo of splits (``_undo_group``/``_redo_group``), and browse reload (``_reload_metadata_from``).  Each tab keeps its own undo history — peers receive only the resulting state, not the chronology.  Coef-only edits never publish (hash guard); absent-file autofills on one tab propagate to peers (deterministic last-edit-wins, matching ``_write_metadata`` merge).

### `_sheet_tint.py` — defaults, tint, placeholders, live sync

`SheetTintMixin`: `_cell_str` (ghost→`""`), `_own_cols` (scalar vs `time_ranges` sizing), `_time_ranges_iid`/`_sheet_time_ranges` (live sheet window), `_time_ranges_relation`/`_time_ranges_detail` (equal/broader/differs recomputed on every hover/edit — no scan-time cache), `_metadata_time_range_default` (`time_ranges[0,-1]` for cell and node tint), `_default_for_cell` (metadata empty→`""`, calib empty-list→`""` per cell — empty means at-default; **coefs path cell** → `input.coefs.path` dataclass default so it tints gray at default and the validation pass restores `CELL_DEFAULT_VAL_FG`; **metadata root path cell** → computed `device_dir/info_devices.yaml` so an existing default-path cell tints gray, a missing one turns red via `_apply_validations` `check=exists`; **input path cell** → the loaded run-YAML `input.path` (no schema default) so clearing restores the loaded value and an unchanged cell tints gray), `_node_at_default` (ghost-aware, empty metadata subtree → blue; locator cells skipped so node labels reflect data, not the pointer), `_apply_default_fg`/`_apply_edit_value`/`_apply_end_edit_style`, placeholders `_placeholder_for`/`_apply_placeholders`/`_restore_placeholder`/`_on_editor_closed`, sync tint `apply_time_ranges_sync_status`/`_apply_time_ranges_tint`.  `_PH_BY_FIELD` registry lives here.  **input.path** gray tint (in `_apply_styles._apply_validations`): additionally gray when the input file stem matches the page config stem per `format.pcid_key` — a same-identity differently-spelled file (e.g. `i_p05` ≡ `i_p5`) still reads as default; a `-backup`/renamed file or a different probe does not (comment must match, see `docs/reference/io_formats.md`).  **empty→default** (in `_apply_edit_value`): committing an empty cell that has a non-empty default auto-assigns the default via `set_cell_data` (Enter on empty fills default), untracking any ghost placeholder first so the written default reads as real data; no reload callbacks fire (reloads only on explicit browse or non-empty manual edit).

### `_sheet_status.py` — hover status + floated PathField overlay

`SheetHoverMixin`: status publication (`_publish_status`/`_clear_status`, tree vs data `_help_candidates`/`_resolve_detail`/`_coefs_status_hint`/Shift), F1 anchor resolution (`_f1_anchor` — selected row via `sh.tree_selected`, or if nothing selected the mouse inside the dwell tooltip widget via `_pointer_in_field`; `_f1_anchor_for_iid` shared resolution for status-label hover; `_help_candidates` fan-out + `meta["parent"]` walk, dispatched by `App._on_f1_help`), floated browse-row overlay (`_ensure_hover_field`/`_show_hover_field`/`_field_place_kw`/`_btn_place_kw`/`_pointer_in_field`/`_hover_read`/`_hover_write`/`_hover_btn_status`/`_restore_hover_placement`/schedule/hide) — `PathField` empty commits propagate to `_hover_write` which writes `""` + ghost, motion branch refreshes `f.set(_hover_read())`.

### `_sheet_styles.py` — alignment/widgets, node fg, cell validation

`SheetStylesMixin`: `_apply_open`, `_cell_spec_for` (numeric metadata via `_meta_pairs.NUMERIC_IDXS`), `_apply_styles` (tree fg blue/black, browse bg, date align), `_apply_validations` (red `check:"exists"` via `_cell_str` — ghost skipped, sentinel-aware; `check:"sorted"` date rows via `_validate_dates` — red on unparseable (`_cell_spec.as_date`) or order-breaking cells, valid cells restore gray/warning/normal), `_path_exists` (``~`` + glob).

### `_path_field.py`

1×1 tksheet for display + `ttk.Entry` overlay for editing — frame-anchored hover button, column-width tracking via `<Configure>`. Parent `scan_list` attaches the inherent numbered dropdown (`_numbered_dropdown.py`) to cell `(0, 0)`; the ordinal is shown in the existing path caption (`_path_lbl`), not a second widget. `_on_begin_edit` yields arrow-band clicks to tksheet (`_is_dropdown_expand` — same band `b1_release` checks) so the list expands; double-click / Return / typing keep the custom Entry. Structural invariant arrow-visible ⟺ no Entry open: edit start restores full-width layout (arrow scrolls off-screen), binder motion neither re-shrinks nor re-shows while `_editing`, expand restores the double-click-like full layout too (full width + `_orig_hide`, but keeping the edit text right-aligned like the custom Entry — only the list itself stays left-aligned; a frozen shrunk layout hid the value) and `_patched_hide` skips restore while `dropdown.open`. The caption toggles the list (`toggle_dropdown` — close when open, else expand). Every tksheet editor/dropdown close (Esc, FocusOut, click-away) re-scrolls the viewport to the value (`_patched_hide_editor_dropdown` — opening scrolls the 4096 px column left while right-aligned text stays far right, which showed a blank field until the next hover).

### `_numbered_dropdown.py`

Library-only (no demo): `NumberedPathDropdown(sheet, row, col, paths, number_label, default_text, on_select)` — display `N. path`, cell keeps the bare path. `set_paths()` on `scan_list`, `refresh()` after programmatic `set()` (bypasses `end_edit_cell`), selection strips `N. ` and rescans that anchor via `App._on_anchor_dropdown_select`. Attach uses `edit_data=False` (never rewrites the cell); selection writes the stripped path back into `event.value` (tksheet commits it after `selection_function`); `end_edit_cell` chains a prior handler (single slot); `set_paths([])` drops the dropdown. The list itself is tksheet's inherent dropdown with the overflow patch (`_dropdown_overflow.py`), opened via arrow click, path-caption click (`hand2` cursor once anchors arrive) or Up/Down on the cell (`PathField.expand_dropdown`, tksheet nav unbound on the 1×1 cell); the pick commits with `redraw=True` so the field updates synchronously.

### `_dropdown_overflow.py`

Instance-level monkey patch for tksheet dropdowns on tiny sheets: the path
field is 1×1 (~18 px) and tksheet embeds the list inside the canvas viewport
(measured list window: 1 px — opens but invisible). `enable_dropdown_overflow`
keeps tksheet's `Dropdown` intact and only replaces the canvas embedding with
`place()` in the sheet's toplevel (per-instance `MethodType` patch of
`open/hide/refresh_positions/newline_binding`, no installed-package edits).
Explicit geometry: not clipped by the sheet (always downward); width
`max(visible cell width, widest item + padding)` capped at the screen (the
*visible* cell width is the floor — a 4096 px column would span screens);
text always left-aligned; height fits *all* items (no six-item / 500 px cap),
capped at the window bottom (a `place()` child is clipped by its window —
short list whole, tall list scrolls);
recalculated after zoom/resize/editing — plus a forced row repaint after
placement. Editor arrows navigate the list: Up/Down/Prior/Next step the
highlight (`_step_open_list`, highlight-only like mouse hover — Return still
commits the editor text) and navigation-key releases skip filter research
(`_dropdown_editor_key_release`, else every release would snap back to the
text match).

### `_browse_button.py`

`BrowseOverlay` (widget core + `pending` state + `on_status`/`status_hint` hover-to-status-bar wiring), `BrowseButtonManager` (sheet-edit policy + injectable `editor_place`), `SheetHoverBinder` (MT motion → overlay show/hide with pending-aware veto), `bind_hover_browse` (Entry legacy).  Dialog `initialdir`: the current value when it is an existing directory (previous `askdirectory` pick opens AT itself, not its parent — regression: `_raw` reopen landed one level up), else its parent dir.

### `_cell_spec.py`

Hydra dataclass → ``CellSpec`` (bool/enum/text/number/date) for cell rendering

### `_about.py`

About dialog: modal `tk.Toplevel` shown only when ready (`withdraw()` → build → single centering `geometry` → `deiconify()` → `_refit()` → `grab_set()` — no top-left corner flash). System title carries the runtime statuses via `about.title` template (`{name} — {mode}, HDF5: {h5}`). Two widgets — `_header` (`MarkdownLabel`: metadata as separate list items: description paragraph, Version, Company, Copyright, clickable repo URL; the meta label's links are plain markdown syntax — `[repo](repo_url)`, a clickable Company link `[company](company_url)` (`_company_label`), docs `[internet](docs_url) / [local](readme path)`, and the copyright's ``<email>`` auto-converted to a clickable `mailto:` link whose percent-encoded target is the RFC 5322 mailbox `Display Name <email>` (`_mailto_link` via `_EMAIL_LINK` — brackets dropped, the mail client shows the name; scheme routed by `open_md_link`'s external whitelist, `link_display` percent-decodes other URIs for hover) — styled, hover-cursored and clicked by `MarkdownLabel` itself (`on_link=open_md_link`), and the hover URL lands in the main status bar via `_on_header_motion` → `label.link_at(x, y)`);
docs widget `_docs_view` (`ttk.Treeview`, `show="tree"`, style `Docs.Treeview`): directory-nested parents (each filesystem folder is a node; a folder holding `_index.md` links that row to the index with no separate `_index.md` leaf, its label is the index title), doc titles as leaves, only the first level expanded by default (deeper folders start collapsed). Tree styling: themed like the main window — `background`/`fieldbackground` = `theme.ENTRY_BG_FALLBACK`, `foreground` = `theme.FG_DEFAULT` (folder nodes are NOT the gray default), selected row = `mix_hex(entry_bg, NODE_DEFAULT_VALS_FG, 0.3)`; row font = ⅔ of the theme `Treeview` font (`_docs_font`), doc rows link-blue (`tag "doc"` via `theme.LINK_FG` — leaves and `_index.md`-linked parents alike; plain folder containers keep the default fg); `rowheight = linespace + 2` synced to that font so descenders ("g", "p") never clip against the next row. All theme colors read at BUILD time via the `theme.X` module attribute — `from .theme import NAME` would freeze the light palette bound before `apply_theme_defaults` mutates the globals.
**Manual word-wrap** (`_populate(width)`): treeview has no native row wrap AND Tk 8.6 items have no per-row `-height` → each extra wrapped line is its own continuation item (`iid = f"{path}#n"`); `_wrap_px` is a greedy word-wrap against `font.measure` with depth-aware budget `width − (_ICON_PX + depth × _INDENT_PX)`. `_iid_path` maps EVERY row segment (leaf + linked parent, first + continuations) to its file path → click/hover resolve uniformly; plain folders (no `_index.md`) only toggle expand.
**Auto scrollbar**: `yscrollcommand=_on_tree_yview` inspects the `(first, last)` fractions (Tk's own pattern) — `_docs_vbar` appears only while the tree can scroll and is `place`d in the tree's right padding strip (`_place_vbar`), so the tree box itself never shifts.
**Screen fit**: all clamping uses `const.work_area()` — per-monitor Win32 `MonitorFromWindow` + `GetMonitorInfo` `rcWork` (taskbar excluded; `SPI_GETWORKAREA` only knows the primary monitor), never `winfo_screenheight`. The main window uses `const.fit_to_workarea(root, *GEOMETRY)` at start (clamp + center + `maxsize`). The dialog stays vertically CENTERED on the work area: `__init__` centers the initial `_W×_H`; `_refit` re-centers on every fitted-height change (`y = top + max(need_gap, 0) // 2`) reading the position via `_pos()` (`wm_geometry()` — the same coordinate space `geometry()` writes; `winfo_x/y` semantics differ per platform); capped content pins to the work-area top. `_fit_label_height` +1-unit growth shifts the window up by `unit // 2` — fixed-y growth would drift the bottom off-center and past the screen. Header px via `_content_px` (tree collapsed to 1 row during measurement): request growing heights (`n_display + 8/16/24`), take the last display line's `dlineinfo` bottom (spacing tags included — never predicted from fonts); growth that stops increasing `winfo_height` = pack-squeeze → bail at the allocation. Docs px = display rows × row px calibrated from two `height` settings (DPI-safe). Chrome px = `_PADS` only (no separator between header and tree — the tree is visually distinct; never the window spare, which would feed back per reflow). `<Configure>` → `_on_resize`: width change >100px → full `_refit` (titles re-wrap at the new width); height-only change → re-glue the vbar. **Hover → main-window status bar**: `on_status` callback injected by `App._on_help` (`_set_status(msg, raw=True)`); header `<Motion>` shows the URL under the pointer (the label's `link_at`), tree `<Motion>` shows the hovered leaf's file path AND switches the widget cursor to `hand2` on leaves (`_on_tree_motion` with a state-deduped `tree.configure(cursor=...)`, `_on_tree_leave` restores it) — folders keep the default cursor; `<Leave>` clears, `_hover_status` dedups motion storms, dialog `<Destroy>` clears; `App` restores `status.ready` on the dialog's `<Destroy>`.
All chrome strings AND the meta VALUES are i18n via `STRINGS` `about.*` keys (`about.meta.description`/`about.meta.company` override build meta; `version_meta.json` stays EN for the exe version info; `©` copyright renders bare).
`parse_markdown` merges consecutive lines into one paragraph, so each header field must be its own block. `<<Copy>>` → `copy_rich` from the _header. `discover_docs` returns `(folder, title, path)` 3-tuples with titles Markdown-stripped via `_md_parse.plain_text` (treeview rows are single-font — `` `ticks` `` inside a ``# `` heading must not display literally), then `_lang_filter` keeps only the app language (`resolve_lang`): `en` → drop suffixed stems (`_lang_parts` splits the `_ru`-style `_([a-z]{2})$` suffix); other langs → per base name prefer the `_{lang}` version, else the unsuffixed original, else any translation. `_docs_tree()`/`_readme_doc_order()` follow the *UI-language* readme (``readme_Ru.md`` etc., English base as fallback) so translation-only files keep their subsection position (regression: the RU-only `tcm_gui_walkthrough_Ru.md`, 3rd in `readme_Ru.md` but absent from `readme.md`, used to sink to the tree tail); the same `_readmes` helper serves `local_readme()`. `_docs_tree()` builds the directory-nested hierarchy from real paths (`_readme_doc_order()` supplies sibling order only); text click on a title or linked parent opens `open_md_link(path)` while the disclosure indicator (`Treeitem.indicator` via `_click_target()`/`_hover_path()`) keeps native expand/collapse with a normal cursor — row text shows `hand2` — a localhost HTTP server serves the document to the system default browser, rendered client-side by vendored marked.js + MathJax (see [`tcm_gui/browser/`](../../../src/tcm_gui/browser/) below)

### `tcm_gui/browser/` — documentation browser

Local document browser subsystem. One localhost HTTP server (`127.0.0.1:<random-port>`) starts on first `open(path)` and is reused afterwards. Serves four document classes — markdown (marked.js + MathJax TeX — math tokenizers `MATH_INLINE`/`MATH_BLOCK` in [`web/viewer.js`](../../../src/tcm_gui/browser/web/viewer.js) extract `$…$`/`$$…$$`/`$`…`$` TeX spans *before* CommonMark parsing so `_`, `*`, `[]` inside formulas stay literal, then MathJax typesets the emitted spans; raw HTML is not sanitized, which is acceptable only because the viewer serves repo-controlled documents inside `allowed_roots` — revisit with DOMPurify if untrusted sources ever appear), source/text (highlight.js, `#L42` line anchors), images (`/api/asset`, native MIME) and external links (handed to the OS browser). Source files cover only repo-present languages (`server._SOURCE_LANG` mirrors [`web/viewer.js`](../../../src/tcm_gui/browser/web/viewer.js)).

Split: [`browser.py`](../../../src/tcm_gui/browser/browser.py) = public `DocumentationBrowser` + singleton (`get_documentation_browser()`); [`server.py`](../../../src/tcm_gui/browser/server.py) = `_Handler`/`_Server`, document classification and static locations; `web/` = first-party viewer page ([`index.html`](../../../src/tcm_gui/browser/web/index.html), [`viewer.js`](../../../src/tcm_gui/browser/web/viewer.js), [`viewer.css`](../../../src/tcm_gui/browser/web/viewer.css), bundled with the package). `_Handler` routes: viewer page at `/` `/index.html` `/viewer.js` `/viewer.css`, JSON `{kind, file, language, content}` at `/api/document?file=<path>` (415 for unsupported kinds), image bytes at `/api/asset`, vendored runtime under `/assets/` (suffix whitelist + root containment). The third-party runtime is generated into [`_build/browser-runtime/`](../../../_build/browser-runtime/) by [`browser/vendor.mjs`](../../../browser/vendor.mjs) (pixi `browser-runtime`; see that script header for the file map and the MathJax newcm font-package constraints — the stub must stay vendored AND `loader.paths["mathjax-newcm"]` pins it, else fonts fall back to the jsdelivr CDN). The frozen build bundles [`_build/browser-runtime/`](../../../_build/browser-runtime/) → `_build/browser-runtime` as PyInstaller datas; `open()` guards a missing runtime with `pixi run -e bin-optim-tcm browser-runtime`.  `open(path, anchor=…)` appends the URL-encoded `#anchor` fragment — the viewer's startup reads `location.hash` and scrolls to the GitHub-slug heading / `L42` line.  Module-level `open_md_link(url, base=None)` is the single dispatcher for every markdown link rendered in the GUI: whitelisted external schemes (`http/https/mailto/ftp/ftps/file:`) → `open_os_target` — the shared OS-associated-application opener (`os.startfile` / `open` / `xdg-open`; `file:` URIs are unquoted to their filesystem path first, UNC netloc restored); anything else is a local doc path (relative ones resolved against `base` — the source `.md`'s directory) → `DocumentationBrowser.open(path, anchor)`; failures are logged, never raised. Only files within `allowed_roots` (default: `resource_root()` — the whole `tcm` package, since docs cross-link files above `docs/`) are served. Browser-side JavaScript handles relative links (Windows paths) and anchors (`slugify`/`addHeadingIds` assign GitHub-style heading ids — `{#explicit-id}` suffix honored, punctuation dropped, each space → `-`, dupes get `-N`; `navigationSerial` guards stale fetches) — no new server starts for in-page navigation. Panel visibility is toggled by inline `display:"block"` (never `""`, which would fall back to the stylesheet's hidden state and blank the page).

### `_help.py`

Auto-extract config-cell help from [`config_reference.md`](../../reference/config_reference.md) tables (``HelpEntry``, ``help_for_path``, ``parse_reference``, ``doc_path``); index-stripping for arrays (``Ag[0]`` → ``Ag``); mode-tagged `###` sections with `####` detail sub-blocks; per-lang cache (`_CACHE` dict, not `lru_cache`); `detail=` kwarg for `#### Detailed` blocks.  ``doc_path()`` resolves the localized source doc — its directory is also the `base` for relative markdown links inside rendered bodies (dwell/error tooltips)

### `const.py`

Immutable user settings (`UI_SCALE`, `FONT_SCALE`, `TTK_THEME`, `COLOR_MODE`); `UIScale` (sets `tk scaling = platform × UI_SCALE` for uniform geometry scaling + named font multiplier via `FONT_SCALE`; `font()` returns scaled `TkDefaultFont` copy; `set_font(*widgets)` applies per-widget copies to any widget); `configure_ui` (ttk theme selection); `work_area` (per-monitor taskbar-excluded rect `(left, top, right, bottom)`, full screen fallback); `fit_to_workarea` (clamp + center; `<Map>` pass re-fits with measured chrome — Tk's `geometry` is the CLIENT rect while the work area bounds the OUTER window — and writes SIZE-ONLY geometry: an explicit `+x+y` stays stored in Tk which re-applies it on every later content-resize, teleporting the window back to its startup spot after the user moved it; its `deiconify()` doubles as the main window's first show — the root stays `withdraw()`n during `_build()` so the default 200×200 top-left mapping never reaches the screen); `tk_font_family`

### `theme.py`

Mutable runtime state: color globals (`FUNC_COLOR`, `FG_DEFAULT`, `NODE_DEFAULT_VALS_FG`, `CELL_DEFAULT_VAL_FG`, `LINK_FG`, `LINK_SEL_FG`, `CODE_FG`, `CODE_BG`, `CODE_SEL_FG`, `CODE_SEL_BG`, `FRAME_BG_FALLBACK`, `ENTRY_BG_FALLBACK`, `CELL_NON_DATA_BG`); `THEME`; `TAG_COLORS`; `widget_meta` registry; `STR` i18n surface; `get_widget_meta` (callable-resolving); `apply_theme_defaults` (dark/light via `COLOR_MODE` or Windows registry); `_apply_ttk_dark` (clam + dark ttk.Style); `_opt_into_dark_titlebar` (`DwmSetWindowAttribute`); `tk_color_to_rgb`/`tk_color_to_hex`; `resolved_frame_bg`/`resolved_entry_bg`

### `cli_cfg.py`

`CFG_DEFAULTS` (config-tree defaults) + `COEF_SHAPES` (auto-derived) + `COEFS_TYPE` — all derived from `Config` via `get_type_hints`, no per-section imports

### `progress_bridge.py`

`GuiTqdm` (tqdm replacement) + module-level runtime injection + `set_cfg`/`get_cfg` per-config attribution + `stage_desc` feeding both `progress_overall` and `ProgressBank`

### `log_bridge.py`

`install()` once at App startup → root logger captures GUI-thread AND worker logs → `QueueHandler` (consecutive dedup + emit-time text freeze) → `tk.Text` drain.  `emit` degrades gracefully: a record whose message cannot survive stdlib `%`-formatting (a `{}`-style string on a plain logger, or a stray `%` in user content) still reaches the queue with its raw text — a logging bug must never crash the GUI callback that produced it (regression: the About dialog died this way from `theme.py`'s `{:#x}` debug call).  `drain` also renders `exc_info` records' full traceback (`traceback.format_exception`) so worker-side `lf.exception(...)`/`exception(...)` errors surface their exact location (file:line of each frame) in the GUI log instead of only in the console.  **`LogText`** (the App log widget) renders message/traceback chunks through `insert_linked(text, tags, root)` — bare paths under the `root` allowed dir (App passes the path field's current value) become links via `tcm_gui._allowed_paths`: files display shrunk to the file name, directories (existence-checked on disk) in full.  Link chunks carry the shared `loglink` visual tag plus a per-target data tag `link:<file: URI>` — the "tag IS the URL" convention of `_md_parse` spans, so `link_url_at(index)` is a one-line `tag_names` lookup with no index bookkeeping; clicks route through `open_md_link` → the OS-associated application.  `drain` duck-types `insert_linked` (plain fallback for dummy widgets).  Log hover: `_on_log_motion` checks `link_at(x, y)` first — the status bar shows the full target (`link_display` decodes `file:` URIs; files render shrunk in the log, so hover reveals the path) and wins over the row status; `_on_log_status_fade` skips clearing while a link is hovered

### `_rtf_clipboard.py`

`Ctrl+C` on log → RTF + HTML + plain text on clipboard (colors preserved; `MarkdownLabel` and `LogText` links → RTF `HYPERLINK` fields / HTML anchors via their `link_url_at` hooks)

### `runtime.py`

Shared state: queues, `ProgressState` (with one-shot `clear_and_reset`/`consume_clear`), `ProgressBank`, `PauseGate`, persistent `queue_handler` reference

### `_tab_rail.py`

Vertical tab rail: progress column + tab column, configs stacked top→down.  Replaces ttk.Notebook entirely — page stack + `tkraise()` for zero-theme page switching.  Hover via `on_hover` callback, per-cell fill animation (lerp), content-based vertical sizing (waterfill on shortage, even split on extreme shortage, capped grow on surplus).  `set_disabled` — inert rail: dim text, hidden selection accent, click veto, no `hand2` cursor (simple mode before the first successful scan; survives `clear()`/`add_tab` rebuilds).  Tab order == `_on_scan_ok` collected order (== `cfgs` dict order); the first tab is selected right after all pages exist — never inside `_add_page` (a page gridded later stacks ABOVE an earlier `tkraise()`'d one, which made the visible page the LAST tab while the rail highlighted the first)

### `progress_bank.py`

Per-configuration progress: fixed stage weights → overall fraction.  Thread-safe: workers mutate under lock, the GUI polls `snapshot_all()`.  States: pending/running/done/error; `canon_stage()` maps free-form descriptions to canonical stages |

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
     → after ALL pages exist: _select_tab(first tab) — rail indicator + raised page
```

### Run

```
Edit coefs in tabs → tab title gets "*" (dirty indicator, polled 300ms)
  → click Run
    → app._clear_log (flush queue + clear tk.Text log)
    → app._write_coefs per tab (skips tabs where is_dirty == False)
      → config_yaml.update_run_yaml(yaml_path, patch)
      → cs.mark_clean() → removes "*"
    → worker._run (thread)
      → call_in_raw_dir(processing.run,
          input={path: "<dir>/cfg_proc/run/(stem1|stem2).yaml"})
        → processing.run: yaml_path filter → process_loading_yaml
          → run_processing: full pipeline (load → coefs → process → persist)
          → return (processed_pcids, failed_pcids, last_cfg, collected)
      → result_queue.put(("run_ok", result))
    → app._poll_results → _on_run_done → reset bars → reload processed tabs
```

### Post-Run tab reload

`_reload_tabs.reload_tabs_after_run` rebuilds each successfully processed tab from its
updated run YAML. `compose_reload_cfg` merges the on-disk YAML over the scan-time sheet config
and drops consumed `input.calib`; tabs with unsaved edits or unprocessed/failed probes are
skipped. Full mode backfills structured defaults and device metadata is re-resolved, while stale
`sync_status` is omitted until the next scan.

### Pause / Resume

```
Click Run while processing → PauseGate
  → pause(): gate.clear()
    → QueueHandler.emit: gate.wait() blocks → log freeze
    → GuiTqdm.update: gate.wait() blocks → dask task freeze
  → resume(): gate.set() → both unblock
```

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
| **theme globals** | Log tags, per-cell highlights, log ``tk.Text`` bg/fg, `MarkdownLabel` bg/fg, `tk.Frame`/`tk.Label` bg/fg | `_DARK` / `_LIGHT` palettes → `setattr` on theme module globals (`FUNC_COLOR`, `CELL_DEFAULT_VAL_FG`, `NODE_DEFAULT_VALS_FG`, `LINK_FG`, `LINK_SEL_FG`, `CODE_FG`, `CODE_BG`, `CODE_SEL_FG`, `CODE_SEL_BG`, `FG_DEFAULT`, `FRAME_BG_FALLBACK`, `ENTRY_BG_FALLBACK`, `CELL_NON_DATA_BG`, `THEME`) + `TAG_COLORS.update()` |
| **ttk.Style** | All `ttk.Frame`, `ttk.Label`, `ttk.Button`, `ttk.Entry`, `ttk.Notebook`, `ttk.Progressbar` | `theme._apply_ttk_dark(root)` → switches to ``clam`` theme (native themes ``vista``/``xpnative`` ignore ``Style().configure()`` for rendering), then ``ttk.Style().configure()`` with bg/fg from theme globals + ``style.map()`` for active/selected states; root window ``bg`` set directly |
| **tksheet** | Sheet canvas (table, header, index, scrollbars, selection) | `ConfigSheet.__init__` calls `self.sh.change_theme("dark")` when `theme.THEME == "dark"` + `scrollbar_theme_inheritance="default"` so tksheet's canvas scrollbars inherit the same ttk theme as `App.Vertical.TScrollbar`; `PathField` uses explicit `table_bg`/`table_fg` from theme at construction (no `change_theme` needed — headers/index/scrollbars hidden) |

### Widget-specific notes

| Widget | bg/fg source |
|---|---|
| `tk.Text` + `ttk.Scrollbar` (log) | `bg=theme.ENTRY_BG_FALLBACK`, `fg=theme.FG_DEFAULT`, `insertbackground=theme.FG_DEFAULT`; manual container replaces `ScrolledText` to get a real `ttk.Scrollbar` |
| Log scrollbar | `ttk.Scrollbar` with `style="App.Vertical.TScrollbar"` — matches tksheet via shared theme inheritance (`scrollbar_theme_inheritance="default"`) |
| `MarkdownLabel` (status) | `background=theme.FRAME_BG_FALLBACK`, `foreground=theme.FG_DEFAULT` |
| `ttk.Label` + `ttk.Progressbar` (§2 status row) | inherits from `ttk.Style` (TLabel/TProgressbar) — no explicit colors |
| `ConfigSheet` (tksheet) | `change_theme("dark")` + `scrollbar_theme_inheritance="default"` in `__init__`; `_apply_styles` uses `theme.resolved_frame_bg()` + `theme.FG_DEFAULT` |
| `PathField` (1×1 tksheet) | `table_bg=theme.ENTRY_BG_FALLBACK`, `table_fg=theme.FG_DEFAULT` — set at construction |
| `ttk.Entry` (PathField editor) | inherits from `ttk.Style("TEntry")` dark configuration |
| Root window + title bar | `root.configure(bg=...)` + `GetAncestor(winfo_id(), GA_ROOT)` to get real toplevel HWND (Tk's `winfo_id()` returns a child widget, not the DWM-controlled frame) + `DwmSetWindowAttribute(DWMWA_USE_IMMERSIVE_DARK_MODE=TRUE)` via `ctypes.WinDLL("dwmapi")` (Win32 only, Win11 22000+) |
| `ttk.Notebook` + tabs | `style.configure("TNotebook.Tab", ...)` + `style.map` for selected state |
| `ttk.Button` (Run) | `style.configure("TButton", ...)` + `style.map` for active/pressed |
| **Scrollbars** | Log `ttk.Scrollbar` + tksheet internal scrollbars | `App.Vertical.TScrollbar` ttk style configured in `_apply_ttk_dark` (dark) / default theme (light); tksheet uses `scrollbar_theme_inheritance="default"` so its canvas scrollbars inherit the same ttk theme as the App; log uses manual `ttk.Frame` + `tk.Text` + `ttk.Scrollbar` instead of `ScrolledText` (which uses an unstyled classic `tk.Scrollbar`) |

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
| Sheet context menu | `sheet.*` | `sheet.insert_col: "Add column"` (append-only extras are *Add*, never *Insert*) |
| Button labels | `run_btn.*` | `run_btn.pause: "Pause"` |
| Status / progress text | `status.*` | `status.ready: "Ready"` |
| Completion template | `overall_lbl.done_detail` | `" - Done {pct}% ({ok}/{n} ok)"` |
| Error prefixes | `error.*` | `error.scan: "Scan: {p}"` |
| PathField placeholder | `path_field.placeholder` / `path_field.placeholder_files` | `"D:/data"` / `"D:/data/_raw/(i*raw_file1[.]txt\|i*raw_file2[.]txt)"` |

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
**Shift** swaps to `path_field.placeholder_files` (advanced pattern syntax);
releasing Shift restores the simple one.  The swap only applies when the
placeholder is visible (cell empty) and no edit is active — it does not
alter `_has_placeholder` state.

`path_field.status` / `path_field.status_shift` are doc+STR: PathField reads
the config_reference `path_field` general short (pre-``####``) and appends
STR suffixes `path_field.status.dirs` (default) / `path_field.status.files`
(Shift-held variant) at init (`_status_body` — same augmentation as
`time_ranges.hover.*`); `_on_shift_press`/`_on_shift_release` swap the
status bar text alongside the placeholder. See [help_system.md](help_system.md).

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

See [config_reference.md `input.path`](../../reference/config_reference.md#input--data-source--its-initial-processing-parameters)
for the CLI equivalent.

### `states.py` enum values

`Stage` and `ScanStage` in `tcm/states.py` are `StrEnum` whose values travel
through `progress_overall` / `progress_stage` and are translated by the GUI
(`_translate_desc` → `_S.get(desc, desc)`, free-form text passes through):

| Enum | Value | Example |
|------|-------|---------|
| `Stage` | backend-owned display text | `LOAD = "load"` |
| `ScanStage` | `scan_stage.*` key in `str.yaml` — display text lives there only | `SCAN = "scan_stage.scan"` |

## Type-aware cell rendering

`ConfigSheet.load()` always receives the full `Config` dataclass as
`config_root` (both modes).  Each cell's type is resolved via
`_cell_spec.spec_for_path(config_root, path, Return)`:

| CellSpec.kind | Rendering | Example fields |
|---|---|---|
| `"bool"` | tksheet checkbox | `program.b_interact`|
| `"enum"` | tksheet dropdown | `program.return_` (7 `Return` values) |
| `"text"` | left-aligned | `input.path`, `out.text_path`, `program.log` |
| `"number"` | right-aligned (default) | `input.calib.azimuth_add`, coefs matrices |
| `"date"` | right-aligned | `datetime` fields |

The `path` stored in `_meta[iid]["path"]` is the dotted Hydra path (e.g.
`"program.return_"`, `"out.dt_bins"`).  Resolution walks the dataclass
tree using `dataclasses.fields` + `get_type_hints(include_extras=True)`.
`Annotated`, `Optional`, and `Union` are unwrapped by `_cell_spec._unwrap`.

## Full mode (Shift at startup)

When `Shift` is held at startup, `App._full_mode = True` and
`ConfigSheet.load()` is called with `full=True`.  The difference:

| Aspect | Simple mode (`full=False`) | Full mode (`full=True`) |
|--------|---------------------------|------------------------|
| Row builder | `_build_coefs` — only `input` section | `_build_full` — all config sections (missing sections/leaves backfilled from structured defaults via `cli_cfg.ensure_full_cfg`; `full_default_cfg()` feeds the pre-scan placeholder; internal keys — `_`-prefixed, `defaults`, `hydra` — never become rows) |
| Visible sections | `input.path`, `time_ranges`, `coefs` (coef source `input.coefs.path` shown inline in the `coefs` row), `calib` | `input`, `out`, `filter`, `program` |
| Editing before scan | Read-only (default page) | Editable |
| Config rail + `_overall_lbl` before scan | Dimmed + click-ignored (`set_disabled` / `_set_cfg_ui_disabled`) | Active |
| Editing after scan | Editable | Editable |
| Overlays before scan | Hidden (hover PathField, browse buttons) | Visible |
| Overlays after scan | Visible | Visible |

Both modes use identical type-aware cell rendering (see above).  The only
difference is **which rows are built**, not how cells are rendered.

Full-mode edits to `out`/`filter`/`program` persist on Run — and since the
generic reader (`tcm_gui/_sheet_patch.py`) covers every non-skipped section,
`input.calib`/`input.time_ranges` edits persist in simple mode too:
`ConfigSheet.get_edited_full()` (thin delegate to `build_patch`) reads generic
rows back into a minimal changed-vs-defaults patch (empty = at-default,
omitted), and `App._write_coefs` merges it into the run YAML via
`config_yaml.update_run_yaml` (deep-merge, backup + `# @package _global_`
header preserved).  Typing comes from the Hydra structured-config dataclass
(`_leaf_kind`/`_elem_kind` via `resolve_dataclass_field`) — `out.dt_bins:
list[int]` round-trips as `int` (strict `parse_int_strict` drops float
spellings instead of writing `600.0`); `check:"sorted"` date rows are
validated + canonicalized before writing (`iso_secs` — ISO `T`-separated
seconds; a row with an unparseable cell is skipped with a warning, and
`ConfigSheet.is_dates_valid` gates the Run button — see
[config_tuning §Inverted time_ranges](../../reference/config_tuning.md#inverted-time_ranges));
`input.path`/`input.coefs`
(matrix/date/`dates` machinery), `metadata*` and `has_date` cells keep their
dedicated write paths (`is_skipped`).  `update_coefs_in_run_yaml` keeps its flat
`{coef: values}` contract and delegates to the same writer.

Containers are never editable: `_ins` defaults `max_col=0` ("no values on this
row"), so every edit gate (`_on_edit`, `_on_begin_edit_cell`, `_on_cell_select`)
rejects cells on section roots, dict parents (`input.calib`, `input.min`) and
2-D parents; only data rows state their width explicitly.  Before this default,
a container rendered all 6 columns editable but the generic reader dropped the
typed values (the old `"["`/`is_string` heuristic) — the cells looked live and
were silently lost on Run.  Since a disabled row can no longer be edited,
double-clicking its cells expands/collapses it (`_redirect_overflow_double` →
`_is_container_row` → `sh.item(iid, open_=…)` with `undo=False`, the same
mechanism as tksheet's own tree-arrow click); date-only rows keep their
date-editor routing and `browse` rows their overflow redirect.  For every
parent node, a single click on its tree-view **label** also toggles
expand/collapse (`_on_tree_col_click`, bound on the RI canvas after tksheet's
own `b1_release` so the arrow keeps its native toggle and there is no double
toggle); clicking a leaf's label is inert.

In readonly mode (`set_readonly(True)`), `_on_begin_edit_cell` vetoes
editing and `_on_sheet_motion` suppresses hover overlays.  Calling
`set_readonly(True)` also tears down any active overlays immediately.  The
same startup branch calls `_set_cfg_ui_disabled(True)` — the rail and the
centered `_overall_lbl` (fg `theme.CELL_DEFAULT_VAL_FG`) join the inert look until a
new search (`_on_path_changed`) or a successful scan (`_on_scan_ok`) enables
them; a failed scan re-dims only while `_yaml_paths` is empty (tabs from an
earlier successful scan stay active).


## Progress

Three layers: per-config fills on the rail, overall description in `_overall_lbl`,
and the collapsible stage row — all driven by snapshots read every 300 ms in
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

**Terminal states are final**: `stage_start` only acts on `pending`/`running`
cells.  Post-loop phases (h5 `combine`) still carry the last config's
attribution — without the guard their `stage_desc` would flip a `done` cell
back to `running` (fill regresses ~1.0 → 0.8), stalling the last config's
bar below 100% (`finish` already ran per-config and never repeats).
`run()` additionally detaches attribution via `progress_bridge.set_cfg(None)`
before combine (see [CLI Internals](../CLI.md) — Combine
attribution detach).

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

### Status bar layout (§2 + §6)

One gridded status row at row=1 plus one bottom-left overlay:

```
root
│
├── row=0: path field (§1)
│
├── row=1: status row (§2) — COLLAPSES when the stage progress is inactive
│   └── f1 (ttk.Frame)
│       ├── column=0 (weight=1): _overall_lbl (ttk.Label)
│       │   Spans the whole row when collapsed (anchor="center" — caption
│       │   centers across the full width).  anchor="w" while expanded.
│       │   ScanStage.DEFAULT → scan_stage.default → "Default configuration"
│       │   ScanStage.DONE → completion text + aggregated % from bank
│       │   During run: stage_desc text + " — 62%" from bank snapshot
│       ├── column=1: _prog_stage (ttk.Progressbar, length=220)      ┐ gridded
│       └── column=2: _prog_stage_text (ttk.Label, anchor="w")        ┘ together
│           Current stage description (or appended error line).  Both are
│           grid_remove'd when tot == 0 (and at build) — column 0 then
│           re-expands and the overall caption re-centers.
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
└── place(rely=1.0, relx=0.0, anchor="sw")  ← bottom-left
    └── _status_lbl (MarkdownLabel)
        wrap="none" by default; switches to wrap="word" only if
        content exceeds window width (_fit_width via font metrics).
        Width contracts to text width.
        Height = exact pixel via place_configure (dlineinfo walk).
        Font: ui.font() copy of TkDefaultFont (same as _log via set_font).
        mark_font_ready() enables auto-sizing; does NOT crush to bar height.
        (error tooltip is rendered here too)
```

**Expand / collapse** (replaces the former bottom-right floater + its z-order
management — gridded widgets never compete with the `_status_lbl` overlay):
`_show_stage_progress()` grids bar + text into columns 1–2 and flips
`_overall_lbl` to `anchor="w"`; `_hide_stage_progress()` `grid_remove`s both
and restores `anchor="center"`.  Column 0 keeps `weight=1` in both states, so
the caption centers over the *entire* row when collapsed — the geometry
manager expresses the two layouts, not per-text `justify` tricks.

**Show delay**: the row is shown via `root.after(400, _show_prog_stage)`
— avoids flashing for very short operations.  Cancelled if `tot` drops to 0
before the delay fires.

### Status bar text — one-shot clear signal

The status `StringVar` is **not** owned by `_poll_progress`.  Explicit setters
control it:

| Setter | When | Widget | Text |
|--------|------|--------|------|
| `_fit_status_font` | startup, no CLI args | `_status_lbl` | `"Ready"` (calls `mark_font_ready()` — does NOT crush font to bar height; status auto-grows for multi-line content) |
| `_fit_status_font` | startup, CLI args | §2 stage row | `"Loading…"` (immediate, no delay) |
| `_on_path_changed` | browse button / Enter | §2 stage row | `"Loading…"` (immediate, no delay) |
| `processing.run` | scan phases | §2 stage row | "Discovering…", "Generating…", "Composing {stem}…" |
| `process_loading_yaml` | per-config | §2 stage row | "Composing {stem}…" (via `_pb`) |
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

### Hover-hide for the stage progress row

Shown always while active; the row collapses only while the pointer is over
the bar **or** the stage text, or the user starts an editing interaction.
Root ``<Motion>`` (``_on_status_motion``) checks both widgets' live bboxes
(``_pointer_inside``) — bar and text hide together as one group
(``_stage_hovering``).

1. Pointer over a gridded stage widget → ``_stage_hovering`` is set and both
   widgets are ``grid_remove()``d (overall caption re-centers).
2. User starts editing → ``_hide_progress_widgets()`` sets the flag and
   collapses the row immediately.  Triggered by: tksheet cell edit begin
   (``ConfigSheet.on_edit_begin`` → fired from ``_on_begin_edit_cell`` and
   from the overlay-field hook ``_on_field_edit_start``), top PathField
   edit begin (``on_begin_edit``), PathField browse click
   (``on_browse_click``), and ConfigSheet browse button click
   (``BrowseButtonManager`` → ``BrowseOverlay.on_click``).  It also sets
   ``_status_hovering`` so hover hints don't replace the label while an
   editor holds focus; the freeze is released on edit-end
   (``on_end_edit`` → ``App._unfreeze_status`` — editor close funnels through
   ``_on_editor_closed`` / ``_commit_entry`` → ``_on_field_edit_end``).
3. Once hidden, the row STAYS collapsed after the pointer leaves — restoration
   is exclusively programmatic: ``_poll_progress`` compares the stage snapshot
   to ``_stage_last`` and clears the flag on change (progress advance / new
   stage), then re-grids via the regular branches.  Pointer leave alone never
   re-shows.

**Why Motion, not ``<Enter>``**: ``_poll_progress`` re-grids the row
mid-motion with the pointer already inside it — Tk only fires ``<Enter>`` on a
boundary crossing, so the widgets can never "catch" the mouse that way
(flicker: disappears while the mouse moves, reappears when it stops).  Root
``<Motion>`` re-evaluates live bounds on every event, so the hide triggers
only when the pointer is genuinely over the widgets and never on motion
elsewhere.

**Error row hides too**: ``_error_active`` does NOT veto hover-hide — the
error row collapses when the pointer is over it, exactly like live progress.
While active, the stage snapshot is frozen (``tot == 0``), so it stays hidden
until a fresh scan/run changes the snapshot (or expands it explicitly); the
error itself persists until cleared by ``_clear_log`` / ``_on_scan_ok`` /
``_on_run_done`` / ``_on_path_changed``.

``_show_stage_progress()`` is the single expansion point; it also clears
``_stage_hovering`` — any explicit expansion is a programmatic activation.
Used by the initial scan, ``_show_prog_stage``, ``_on_path_changed`` and
``_surface_error``.

``winfo_ismapped()`` is not used (unreliable before window realization and on
withdrawn test roots); ``grid_info()`` (via the ``_stage_shown`` property) is
used instead.

## CLI integration

GUI accepts CLI args: first positional = data path (prefills GUI entry, auto-scans), then
`key=value` = Hydra overrides passed verbatim via `sys.argv`
(see [`cli.py` internals](../CLI.md#entry-point) for
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
the user-edited YAML files are the sole config source, plus the simplified-mode
`out` binning defaults ([`worker.py`](#workerpy)).
