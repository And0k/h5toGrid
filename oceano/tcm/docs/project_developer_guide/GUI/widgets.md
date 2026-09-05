# GUI Widgets

Interactive overlay widgets floating above the tksheet surfaces — browse
button, path field, hover binder, tab rail, rich clipboard.

Companion pages: [GUI Architecture](architecture.md) ·
[GUI Help System](help_system.md) ·
[GUI Key Decisions with Rationale and Regression Notes](decisions.md) —
index: [GUI Internals](_index.md).

## Floating browse button (`_browse_button.py`)

A `…📁` / `…📄` button that appears next to the active editor (tksheet
TextEditor or `ttk.Entry`) and writes the selected path to **column 0** of
the target row, regardless of which column the user clicked.

### Two usage sites

| Site | Trigger | Target |
|------|---------|--------|
| `_path_field.py` §1 — data path | `SheetHoverBinder` motion policy | PathField's single cell `(0, 0)` via `set_cell_data` + `_notify` |
| `coef_sheet.py` — in-sheet edit | `_on_begin_edit_cell` for rows with `meta["browse"] = True` (`input`, `coefs.path`) | tksheet cell `(row, 0)` via `set_cell_data` |
| `coef_sheet.py` — hover-edit | intent-delayed `PathField` (text) + separate `BrowseOverlay` (button at right edge) over browse rows | `_hover_write` → `set_cell_data` + `_apply_edit_value` + `coefs.path` notify |

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

`BrowseOverlay` accepts `on_status: Callable[[str], None]`,
`status_hint: str`, and `status_hint_files: str` parameters.
`_resolve_hint()` is mode-aware: when the button is in file mode
(``_files_only`` or Shift held) and ``status_hint_files`` is set,
it returns the files-specific hint; otherwise the default (dir) hint.
On `<Enter>` the button calls `on_status(_resolve_hint())` to show
the hint in the GUI status bar; on `<Leave>` it calls
`on_status("")` to clear.  `_make_button` binds both events with
`add="+"`.  All three creation sites pass the callback:

| Site | `on_status` source | `status_hint` / `status_hint_files` |
|------|-------------------|--------------------------------------|
| `PathField` (§1 data path) | `App._on_browse_status` | dir: `STR["browse_btn.status"]`, files: `STR["browse_btn.status_files"]` |
| `BrowseButtonManager` (in-sheet edit) | `App._on_browse_status` | `STR["browse_btn.status"]` (dir-mode hint) |
| `ConfigSheet._ensure_hover_field` (hover-edit) | `lambda text: self.on_hover_status(text, True)` | files-only: `STR["browse_btn.status_files"]` |

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
black `FG_DEFAULT` foreground (not `NODE_DEFAULT_VALS_FG`).  The `align` parameter
defaults to `"w"` (standalone); ConfigSheet passes `align="e"` for the
floated field.

**Edit mode**: `_on_begin_edit` returns `None` to **veto** tksheet's
built-in `tk.Text` editor, then places a `ttk.Entry` filling the
PathField frame (`relx=0, rely=0, relwidth=1, relheight=1`).
The Entry has `justify="right"` so the cursor starts at the filename
end.  Enter commits, Esc cancels — same contract as a tksheet cell.
Exception: an arrow-band click on a dropdown cell (`_is_dropdown_expand`)
returns the cell text instead — yielding to tksheet's list, no Entry.

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
      _is_dropdown_expand?       # click + pointer in arrow band + dropdown kwargs
        → return get()           # YIELD: tksheet opens its editor + list, no Entry
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
_FocusOut_ → _commit_entry       # click-away / focus loss commits — same
                                # contract as a tksheet cell (its editor
                                # commits on FocusOut); ``_entry is None``
                                # guard makes destroy-triggered re-entry safe
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

## Numbered anchor dropdown (`_numbered_dropdown.py`)

Inherent tksheet dropdown on the §1 path cell `(0, 0)` — no separate
combobox. `NumberedPathDropdown(sheet, 0, 0, paths, number_label=_path_lbl,
default_text=…, on_select=…)` shows `1. …\_raw`, `2. …\_raw`, … while the
cell keeps the unnumbered path. The ordinal lives in the existing path
caption: `selected/total` on exact dropdown match, default caption otherwise.
`set_paths()` replaces the list on parent `scan_list`; `refresh()` re-reads
the cell after programmatic `set()` (which bypasses `end_edit_cell`).
Dropdown selection strips the `N. ` prefix, updates the number, and calls
`on_select(path)` → path-field `set` + rescan of that single `_raw`.
Manual typing via the `ttk.Entry` overlay refreshes the number in
`App._on_path_changed`. No `Demo` — the module is library-only.

Display/value split mechanics (tksheet writes display strings back, so each
is neutralized): attach passes `edit_data=False` (attaching never rewrites
the cell with the first numbered value); selection writes the stripped path
into `event["value"]` (`close_dropdown_window` commits it after
`selection_function` returns); `end_edit_cell` chains the previously bound
handler (`extra_bindings` is a single slot — `PathField._on_end_edit` must
survive); `set_paths([])` calls `del_dropdown` so the arrow offers no stale
anchors. Click-to-expand needs the PathField yield above: tksheet's
`open_dropdown_window(state="normal")` gates the list on `open_text_editor`
succeeding, and the veto made every arrow click open the Entry instead.
Openers beyond the arrow: caption click (toggles — `PathField.toggle_dropdown`;
`hand2` once anchors arrive) and
Up/Down (`PathField.expand_dropdown` drives tksheet's `"rc"` opener;
tksheet's own Up/Down are unbound on the 1×1 cell since their `"break"`
swallows later handlers — stock never routes editor arrows to its list, so
the patch steps the highlight itself (`_step_open_list`) and skips filter
research on navigation releases); the pick commits with `redraw=True` so the
field updates synchronously.
Overlay coexistence (the open Entry used to swallow all arrow clicks —
self-perpetuating): edit start restores full-width layout via the patched
hide (arrow scrolls off-screen, so the full-frame Entry never covers it);
binder motion returns None while `_editing` (no re-shrink under the open
Entry, no button); expand restores the same full layout (tksheet's editor
opens over the cell with the value visible) and calls `_orig_hide`
(button off); `_patched_hide`
skips restore while `dropdown.open` (no canvas jump under the open list).
Every tksheet editor/dropdown close (Esc, FocusOut, click-away) re-scrolls
the viewport to the value end/start (`_patched_hide_editor_dropdown` —
opening scrolls the wide column left while the shrunk layout keeps
right-aligned text at the far right, which blanked the field until the
next hover masked it).
Floated ConfigSheet fields are unaffected (null overlay, never a dropdown).
The list itself escapes via `_dropdown_overflow.enable_dropdown_overflow`
(tksheet embeds it in the 1-row canvas — measured 1 px tall): width
`max(visible field width, widest path + padding)` capped at the screen,
always left-aligned, height for all items capped at the window bottom.
Regression coverage: `test_numbered_dropdown_expand.py`.

## Floated PathField in ConfigSheet (`coef_sheet.py`)

One reusable `PathField` instance (text surface) + a separate `BrowseOverlay`
(button at the sheet's right edge) float over browse rows
(`input`, `coefs.path`) on hover, replacing the former single `BrowseOverlay`.

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

`_hide_hover_field` unmaps the field **then** `cancel_edit()`s an open
Entry — unmap-first makes `_restore_hover_placement`'s `winfo_ismapped`
guard skip the place/show round-trip just undone.  Without the cancel
(regression: double-click a browse row, then double-click any other row)
an orphaned Entry kept `_editing=True` forever and every
`_editing`-guarded path (`_show_hover_field`, `_do_field_hide`,
`_on_sheet_motion`) early-returned — the browse button + field overlay
never reappeared until a new scan rebuilt the sheet.

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
`coefs.path` notify).  Esc undo → `_commit_entry(cancel=True)`.  Browse
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
   fires `on_edit_begin` (status-freeze parity with cell editing),
   cancels pending hide job, hides browse button, expands field to
   full width (`_field_full_width_kw`), forces geometry via
   `update_idletasks`.
2. PathField constrains column to frame width, opens `ttk.Entry`,
   returns `None` (vetoes tksheet editor).
3. Enter/Esc → `_commit_entry` → `_editing = False` → calls
   `_on_end_edit_cb` → `ConfigSheet._on_field_edit_end` →
   fires `on_edit_end` (App unfreezes hover status — also reached via
   `cancel_edit()` from `_hide_hover_field` teardown) →
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

## Rich clipboard (`_rtf_clipboard.py`)

`Ctrl+C` on the log `tk.Text` widget calls `copy_rich` from
[`_rtf_clipboard.py`](`_rtf_clipboard.py`).  `_segments` walks all tag
boundaries, maps each tag's `foreground` to an 8-bit RGB via `winfo_rgb`, and
resolves link URLs duck-typed through `MarkdownLabel.link_url_at` (widgets
without the hook — e.g. the log — simply carry no URL).  `build_rtf` emits a
single `{\cfN …}` run per span into an RTF `\colortbl`; link spans wrap in
`{\field{\*\fldinst{HYPERLINK "url"}}{\fldrslt …}}` (+`\ul`) so Word keeps
them clickable, `build_html` wraps them in `<a href>`.  Three formats are
placed on the clipboard:

- **`CF_UNICODETEXT`** — plain text fallback for all targets
- **`CF_RTF`** — Word / Outlook preserve foreground colors and links
- **`HTML Format`** — CopyQ and other clipboard managers that prefer HTML over RTF

Word may still show its paste-options flyout defaulting to *Merge Formatting*
— that is Word's own "Pasting from other programs" setting (set it to *Keep
Source Formatting* once via *Set Default Paste…*); the RTF itself is
well-formed (fonttbl+`\deff0`, `\cfN` runs, `\uN?` unicode escapes).

When `pywin32` is unavailable, falls back to `widget.clipboard_append(plain)`.

**Binding placement.** The binding is `self.root.bind("<<Copy>>", ...)`,
NOT `<Control-c>`.  Tk maps `<Control-Key-c>` → `<<Copy>>` via
`event add` at the C level — a `<Control-c>` binding is dead code on real
keypresses.  `_log` is `state='disabled'` → never gets keyboard focus →
widget-scoped binding would never fire.  Root `<<Copy>>` fires for any
focused widget.

**Layout-independent shortcuts (`keyboard.py`).** Tk's `<<Copy>>` /
`<<SelectAll>>` / … only fire for Latin keysyms.  On Cyrillic/Greek layouts
the physical key produces a different character, so the shortcut is broken
for `tksheet`, `_log` RTF, and the About dialog.  `LayoutIndependentShortcuts`
(bound once via `bind_all("<KeyPress>", ...)` in `App.__init__`, so every
widget is covered) detects the physical key by its platform `keycode`
(win32 VK / X11 codes in `keyboard._KEYCODES`) and re-emits the semantic
virtual event on the focused widget: Ctrl+A → `<<SelectAll>>`, Ctrl+C →
`<<Copy>>`, Ctrl+X → `<<Cut>>`, Ctrl+V → `<<Paste>>`, Ctrl+Z → `<<Undo>>`,
Ctrl+Y → `<<Redo>>`, Ctrl+F → `<<Find>>` (app-level hook, no default Tk
target).  Latin layouts pass through (`keysym` already Latin) so Tk handles
them unchanged — no double-fire; non-Latin returns `"break"` so the
translated keysym leaks nowhere.

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

**Tests.** The `TestRtfClipboard` tests that write the real OS clipboard carry
`@pytest.mark.clipboard` and are deselected by default
(`addopts -m "not clipboard"` in `pyproject.toml`) — each run would otherwise
fill clipboard-manager history (CopyQ / Win+V) with test junk.  Run them
explicitly with `pytest -m clipboard`.  Builder-level tests (`build_rtf` /
`build_html`, incl. link preservation) are pure and always run.

## TabRail (`_tab_rail.py`)

Vertical tab rail replacing the native ttk.Notebook tab row.  Two columns
share one vertical extent per config:

```
[progress column (PROG_W)] [tab column (TAB_W)]
  thin vertical fill          rotated label (angle=90, reads bottom→up)
  grows top→down              selection accent on notebook-facing edge
  color by state              dirty `*` in top-left corner (separate text;
                              config-only → config bg, meta-only → meta bg,
                              both → FG), ✔ on done
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
