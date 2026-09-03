# GUI Help System

**Content doctrine**: status / dwell text = purpose + what editing the field
affects + when/why to edit. Provenance narration not impotant;
only effects of the user's actions belong in help text. GUI behavior
(live comparisons, tints) is shown intuitively at viewing time — never
narrated in help text, and so not to gui.md. Links of user-relevant behavior → user_guide pages
(e.g. configuration.md), to the CLI docs - are encouraged, but not to the development docs

`config_reference*.md` is mandatory — no STR fallback for doc-driven texts.

Two independent help sources — one per widget category:

| Source | Widgets | Key derivation | i18n mechanism |
|---|---|---|---|
| `STR` (``const.py``) | Chrome widgets (`self._path_lbl`, `_path_field`, `_overall_lbl`, `_run_btn`) + dynamic tabs | Attribute name → role (aliased via `_CHROME_ALIAS`) → ``STR["{role}.tooltip"]`` / ``STR["{role}.status"]`` | Replace ``STR`` dict wholesale at build for target language |
| ``_help.py`` (``config_reference.md``) | Config cells (``_meta[iid]["path"]`` keys) + PathField statuses | ``section_body_short(help_for_path(strip_index(path)))`` | Replace ``config_reference_<lang>.md`` |

Companion pages: [GUI Architecture](architecture.md) ·
[GUI Widgets](widgets.md) ·
[GUI Key Decisions with Rationale and Regression Notes](decisions.md) —
index: [GUI Internals](_index.md).

## Chrome widgets: auto-registration

``App._register_chrome_help()`` runs once at the end of ``_build()``.  It walks
``vars(self)`` for all ``tk.Misc`` instances whose attribute name (minus the
leading ``_``), after ``_CHROME_ALIAS`` resolution, has entries in ``STR``.

**`_CHROME_ALIAS`** — chrome role → help source: ``{"path_lbl": "path_field"}``.
The search-path label shares the PathField's help — one source, no duplicated
STR keys.  For aliased roles:

* ``tooltip`` = ``STR["{src}.tooltip"]``.
* ``status`` is **doc + STR** = the config_reference ``path_field`` general short (pre-``####``) plus ``STR["path_field.status.dirs"]`` (default) / ``STR["path_field.status.files"]`` (Shift) — see ``_path_field._status_body`` (same augmentation as ``time_ranges.hover.*``).

For every other match:

* ``tooltip`` = ``STR["{role}.tooltip"]`` (static string).
* ``status``  = ``STR["{role}.status"]`` (static string) **or** a bound method
  (``self._run_btn_status`` for Run) returning the live caption.  Dynamic
  ``status`` is stored as a ``Callable[[], str]`` in ``widget_meta``; the
  ``get_widget_meta`` resolver invokes it at hover-time, so it sees the current
  application state (disabled/busy/paused) and the current language (``STR``)
  simultaneously.  The Run button's dynamic status reports *why* it is
  disabled (no configurations vs. invalid paths) before falling through to
  the busy/paused/ready captions.

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

Static ``STR["{role}.status"]`` strings **do** go through the Markdown parser
by default at ``_on_chrome_hover`` / ``_on_path_hover_in``, so ``**bold**``
segments render bold — same for the doc+STR ``path_field`` statuses
(``help_for_path("path_field", mode="dirs").body`` + ``STR["path_field.status.*"]``).

## MetaValue: callable status support

``widget_meta`` stores ``MetaValue = str | Callable[[], str]``.  The
``get_widget_meta`` getter resolves callables at read time:

```python
val = widget_meta.get(widget, {}).get(key, default)
return val() if callable(val) else val
```

Static ``str`` values pass through unchanged.  Dynamic status callables are
never stored pre-resolved — they close over ``STR`` and/or ``self``, so each
hover-time read reflects the live state and language.

## Config cells: doc-driven hover (no widget_meta needed)

Config cells have ``_meta[iid]["path"]`` (dotted Hydra path) — that IS the
help key.  ``_sheet_status._publish_status`` (via `ConfigSheet`) calls
``section_body_short(help_for_path(path))``, which returns the ``###`` section
lead-in text (the "section status below the table") when present, falling back
to the table-row last cell.  The text is cleaned before display:
``[↓](#anchor)`` links (arrow-only, no useful display text) are stripped, and
when H5 is unavailable (``_constants.H5_AVAILABLE`` is ``False``), lines
mentioning ``HDF5`` or ``NetCDF`` are dropped from status/tooltip text.  Both
cleanups run once at parse time in ``_help._clean_text`` — every display path
benefits automatically.  ``help_for_path`` returns a ``HelpEntry(short, body)``
parsed once from the tables in ``config_reference.md``:

1. Parser walks lines, tracking code-fence state and ``## `section` `` headings
   (``input``, ``input.coefs``, ``out``, ``filter``, ``program``).
2. Inside a config-group section, every markdown table row whose first cell is
   a backtick-quoted identifier (``| `field` = default | … | description |`` —
   the joined ``Field = Default`` column) emits
   ``HelpEntry(path="{section}.{field}", short=<last cell>, body={})``.
3. **Field detail sections** — ``### `field.path` `` subheaders (the
   `<mode>` tag is optional — context-dependent fields only) accumulate
   detail content into ``body[mode]``; a modeless section is stored under
   ``body[_NO_MODE]``.  Authoring contract: [§Field detail sections in
   doc_authoring.md](doc_authoring.md#structure).
   - **New**: `#### <mode>value</mode>` nested under a `### \`field.path\``
     heading inherits the parent field path — equivalent to a separate
     `### \`field.path\` <mode>value</mode>` heading but nests the mode
     detail under the general field description.  The general description
     (stored under the modeless key `_NO_MODE`) is accessible via
     `help_general_for_path()` for field-associated error messages.
4. CamelCase field names (``Ag``, ``Cg``, ``Rz``) parse identically to
   lowercase Hydra names.
5. ``_DOC_PATH`` resolves to ``config_reference.md`` at
   ``{tcm_root.parent}/docs/reference/config_reference.md``; absent file →
   empty cache → no hover text (graceful degradation).
6. Array indices stripped at lookup time: ``Ag[0]`` / ``Ag[1][2]`` → ``Ag``.

Fallback chain in ``_publish_status`` (now in ``_sheet_status``):
``section_body_short(help_for_path(candidate))`` → ``key`` / ``label`` / ``path`` (no `hover_status` cache — `time_ranges` detail is live via `_time_ranges_detail`).
No ``set_widget_meta`` calls on config cells — the entire chain is read-only
from the parsed doc.

**Text cleaning** — ``_help._clean_text`` runs once at parse time over every
assembled text (table-row cells, section lead-in, detail blocks, post-heading
paragraphs, section subtitles) so all display paths benefit:

* ``[↓](#anchor)`` markdown links are stripped — the arrow is a doc-internal
  "see detailed section below" marker with no useful display text in a status
  bar or tooltip.
* When ``_constants.H5_AVAILABLE`` is ``False`` (no h5py/pytables), lines
  containing ``HDF5`` or ``NetCDF`` (case-insensitive) are dropped — keeps
  status/tooltip text relevant to the no-h5 distribution without separate
  doc variants.

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
| ``metadata`` burst pair (``burst_dt/t``) | ``metadata.burst_dt``, ``metadata.bursts_t`` | both cells fan like ``,``-pairs — ``/`` splits too, shortened ``t`` maps to the ``bursts_t`` key |

``_status_source`` (``"tree"`` / ``"data"`` / ``None``) tracks which canvas owns
the current status so moving between tree column and data cell on the SAME row
triggers a re-publish.

## Field detail sections in `config_reference.md`

When a field needs more than the table cell, its detailed documentation goes
into a ``### `field.path` `` section; when the meaning depends on the consumer
context, one section per context, tagged with a **mode** — authoring rules
and examples live in [doc_authoring.md](doc_authoring.md#structure).  This
section documents the parser and consumer internals.

**Heading syntax**: ``### `dotted.field.path` `` optionally followed by
``<mode>value</mode>`` (``</>`` shorthand accepted) and an explicit ``{#id}``.
Regex (mode group optional):
``^###\s+`([A-Za-z_]\w*(?:\.\w+)*)`(?:\s+<mode>([a-z_]+)</(?:mode)?>)?``

New: ``#### <mode>value</mode>`` under a ``### `dotted.field.path` `` heading
inherits the parent field path:
``^####\s+<mode>([a-z_]+)</(?:mode)?>``
Its child detail uses ``#####`` (one level deeper):
``^#####\s+(?P<tag>.+?)\s*$``

**Parser behavior**:
- ``_FIELD_MODE_HEAD`` is checked **before** ``_ANY_HEADING`` — a ``###``
  field section (tagged or modeless) does not close the parent ``##`` section.
- Each ``###`` subheader opens accumulation under ``HelpEntry.body[mode]``;
  a modeless heading stores under the implicit key ``_NO_MODE = "detail"``.
- A field-level ``####`` block still open when a ``###`` section opens is
  flushed first — the section body never leaks into the previous field's detail.
- ``#### <Tag>`` sub-blocks (e.g. ``#### Detailed``) nest inside the active
  ``###`` section — they do NOT close it.  A section carrying any ``####``
  block is stored as ``_ModeBody(short=<pre-#### lines>, details={tag: body})``;
  without ``####`` it stays a plain ``str``.
- **Bare `### Detailed`** (no backticks) under a `## ` section starts a mode
  for the current section with tag `"Detailed"` — its content becomes the
  section's tooltip (stored as the mode body's `short`).  Unlike a
  `` ### `path` `` heading it does **not** close the section, so subsequent
  `` ### `` path blocks still parse.  Detected before the generic heading
  check, so it does not close the section.
- **Post-heading paragraph**: the paragraph between a `## ` heading and its
  table is captured as the section's `short` (falls back to the subtitle).
- **Citation blockquote** (`>`): a line starting with `>` finalizes the current
  accumulation (post-heading paragraph or mode body) and is itself discarded.
  Subsequent headings still start new sections.
- **New**: ``#### <mode>mode</mode>`` under a ``### `field.path` `` heading
  closes the current mode (saving the general description under ``_NO_MODE``)
  and opens a new mode with the inherited field path (level 4).  Its child
  detail uses ``#####`` (e.g. ``##### Detailed``).  This is equivalent to a
  separate ``### `field.path` <mode>mode</mode>`` heading (whose child is
  ``#### Detailed``) but keeps the general description.  A ``####`` heading
  after a ``#### <mode>`` section is a sibling, not a detail — it closes the
  mode, so `#### Detailed` under `#### <mode>` must be `##### Detailed`.
- Next ``###`` or ``##`` (or a `####` sibling of a `#### <mode>` section)
  closes the previous accumulation.
- Fields without ``###`` sections get ``body={}``.

**Consumer API** — ``help_for_path(path, *, mode=None, detail=None)``:

| Call | Return |
|------|--------|
| ``help_for_path("input.path")`` | ``HelpEntry(body={"detail": _ModeBody(...)})`` — modeless section |
| ``help_for_path("path_field")`` | ``HelpEntry(body={"detail": _ModeBody(short="Search path...", details={"Detailed": "...", "Important": "..."})})`` — non-Hydra, modeless only; STR ``path_field.status.dirs/files`` supplies GUI dirs/files variants |
| ``help_for_path("path_field", mode="dirs")`` | ``HelpEntry(body="...")`` — general short (fallback to ``_NO_MODE``) — GUI appends ``STR["path_field.status.dirs"]`` via ``_path_field._status_body`` for hover status |
| ``help_for_path("path_field", mode="dirs", detail="Detailed")`` | ``HelpEntry(body="...")`` — the ``#### Detailed`` (general) block body — dwell tooltip (both dirs/files fallback to same) |
| ``help_for_path("program.return_", mode=_NO_MODE, detail="Detailed")`` | ``HelpEntry(body="...")`` — a modeless section's Detailed block |
| ``help_for_path("path_field", detail="Unknown")`` | ``HelpEntry(body="")`` — unknown detail → empty (caller no-ops) |

**Non-Hydra sections**: ``path_field`` (GUI search path) is not a schema field
— it registers via the explicit ``{"metadata", "path_field"}`` membership of
``_FIELD_SECTIONS``.  No ``## `` heading or table row is required: the first
``### `path_field` `` heading auto-opens the section and creates the entry
(the anchor is the heading slug minus the ``<mode>`` tail; mode bodies attach
only to existing entries).  Its general short (``_NO_MODE``) plus STR suffixes
``path_field.status.dirs/files`` feed the PathField hover statuses
(``_path_field._status_body``) and the ``path_lbl`` chrome status (same concat);
the scan/run error hint shows the ``#### Important`` detail of the general
section (via ``help_general_for_path``), while ``#### Detailed`` feeds the
dwell tooltip.

**New API** — ``help_general_for_path(path)`` returns the general (modeless)
description for a config path — the ``### `field` `` short body before any
``#### <mode>`` or ``### `field` <mode>mode</mode>`` section.  Used for
field-associated error messages (e.g. ``FileNotFoundError`` on a failed
data/config search) where the mode-specific body is irrelevant.

Likewise ``##`` section headings accept plain titles (backticks optional):
a heading registers a field section iff its name is in ``_FIELD_SECTIONS``,
so general prose headings (CLI keys, "See also") pass through unregistered.

To add a new mode: (1) add a ``### `field.path` <mode>new_mode</mode>``
subsection in ``config_reference.md``; (2) call ``help_for_path(path,
mode="new_mode")`` in the consumer.

## Field-level `#### Detailed` blocks (no `###` section)

Fields may also carry a ``#### Detailed`` block directly under the ``##
section`` heading — **after** all table rows.  All current docs use explicit
``### `field` `` sections instead; the field-level form remains parser-supported:

**Critical placement rule**: the ``####`` heading breaks the markdown table —
any table rows after it are parsed as body text, not field rows. The content
is associated with the **last field row** before the block (tracked via
``last_field_path``) and stored as ``ModeBody(short="", details={tag: body})``
under the sentinel key ``_FIELD_DETAIL = "_"``.

**Consumer** — ``_resolve_detail`` in ``coef_sheet.py`` scans every section of
the field (mode-tagged, modeless, field-level) and returns the first
``#### Detailed`` body, or the content of a bare ``### Detailed`` block:

```python
if (e := _help.help_for_path(path)) and isinstance(e.body, Mapping):
    for tag, val in e.body.items():
        if isinstance(val, _help.ModeBody):
            if tag == "Detailed":
                return val.short
            if d := val.details.get("Detailed"):
                return str(d)
return ""
```

``#### Detailed`` (and bare ``### Detailed``) is the **only** body that arms the
dwell tooltip — section short bodies and group prose never do.

## Error tooltip in `_status_lbl`

``_show_tip`` sets ``_tip_active = True`` and renders markdown directly in
``_status_lbl``.  While active, ``_set_status`` is a no-op — all chrome-hover,
poll, and log-motion status updates are suppressed.  The short error line
lives in ``_prog_stage_text`` (§2 row), which cannot overlap the bottom-left
tooltip — the former floater z-order dance (``_lift_status_z``) is gone.
Dismissed by
``_hide_tip`` on: new scan/run (``_clear_log`` / ``_on_scan_ok`` /
``_on_run_done``), path change (``_on_path_changed``), ``<Escape>`` (root
binding), or cell edit begin (``ConfigSheet.on_edit_begin`` → ``_hide_tip``;
fired from ``_on_begin_edit_cell`` and ``_on_field_edit_start``, excluding the
Top PathField which is a ``PathField``, not a ``ConfigSheet``).

``_hide_tip`` also clears dwell tooltip state (``_dwell_active``,
``_dwell_widget``, pending job) — both tooltip types share the same
``_status_lbl`` overlay and dismissal triggers.  It releases the
``_status_hovering`` hold unconditionally (before the no-op guard): clearing
the label collapses it to ~0 width under a stationary pointer, and Tk emits
no ``<Leave>`` for a widget shrinking beneath a cursor — a stale hold would
otherwise keep ``_apply_status`` blocked while detailed hints still render.

## Dwell tooltip (_DWELL_MS delay hover → detailed help)

When the mouse stays in a widget area for :attr:`App._DWELL_MS`,
a detailed tooltip is shown in ``_status_lbl``.  Unlike error tooltips
(``_tip_active``), dwell tooltips disappear when the mouse leaves the widget.

**Trigger**: each hover-enter event (``<Motion>`` on a new chrome widget,
``<Enter>`` on PathField, sheet cell change) calls :meth:`_arm_dwell` with
the widget's tooltip text.  The method schedules a single ``after(_DWELL_MS, ...)``
callback.  Subsequent motion within the same widget does NOT reset the timer
— ``_dwell_widget`` tracks the arming widget and only re-arms on change.

**Content**:

| Widget category | Dwell text source |
|---|---|
| Chrome widgets | ``widget_meta[w]["tooltip"]`` (``STR["{role}.tooltip"]``) |
| PathField | ``help_for_path("path_field", detail="Detailed")`` (``mode="dirs"/"files"`` fallback to same) — ``#### Detailed`` (general) (fallback to ``STR["path_field.tooltip"]``) |
| ConfigSheet cells | :meth:`ConfigSheet._resolve_detail` — ``####/##### Detailed`` blocks only from ``config_reference.md`` (mode ``probe`` → mode ``search`` → field-level).  Mode short bodies and parent-group prose never arm the dwell — a tooltip exists ⟺ the field carries a ``Detailed`` block (regression: every coef row showed the ``input.coefs`` group text) |
| Log | ``STR["log.tooltip"]`` (if defined) |

**Suppression**: while ``_dwell_active`` is True, the debounced
:meth:`_apply_status` (``_STATUS_SETTLE_MS`` = 0.3 s after the latest
:meth:`_set_status`) clears it — motion within the same widget publishes no new
status (dedup), so the tip persists; switching to another row/widget replaces
it after 0.3 s.

**Dismissal**: the tip stays while hovered; a dismissal trigger (``<Leave>``
on the widget, hover-enter on a different widget, a debounced status switch —
cursor moved to another sheet row) only **schedules** the clear after
``_DWELL_HIDE_MS`` (3 s linger — reading / clicking links); the debounced
switch re-queues itself right after the clear, so the new status takes over
exactly when the tip goes away.  A re-show (``_show_dwell_tip``) cancels the
pending clear and takes the label.  Hard clears are immediate: ``<Escape>``
(root binding), ``_hide_tip()`` (new scan/run/path change/edit begin) and
``_show_tip()`` (error precedence, via ``_clear_dwell_now``).  ``_hide_tip``
/``_clear_dwell_now`` also release the ``_status_hovering`` pointer hold so a
collapsed-label dismissal can't freeze normal status (see *Error tooltip*).
The linger
countdown **pauses while the pointer is on the status label itself**
(``_on_status_enter``/``_on_status_leave`` hold the tip; ``_apply_status``
also yields while ``_status_hovering``).  The status label carries no
``status_lbl.*`` STR keys — hovering it must not replace the tip it renders.

**Architecture** — anchor is forged at **display time**, not hover time, so it always
matches what is currently shown in ``_status_lbl`` (short status vs dwell
tooltip for different rows can otherwise diverge):

```
_on_chrome_hover(A) → _cancel_dwell() + _set_status(A.status, A.anchor) + _arm_dwell(A.tooltip, A.anchor)
  → _DWELL_MS timer fires → _show_dwell_tip(A.tooltip, A.anchor) → _dwell_active = True (stays while hovered)
_on_chrome_leave(A) → _cancel_dwell() → clear scheduled after _DWELL_HIDE_MS (3 s linger)
_on_chrome_hover(B) → _cancel_dwell() + _set_status(B.status, B.anchor) + _arm_dwell(B.tooltip, B.anchor)
  → _apply_status fires after _STATUS_SETTLE_MS (0.3 s) — dwell still owns the label:
    clear scheduled at _DWELL_HIDE_MS, the switch re-queues itself right after it (anchor threaded too)
```

``_set_status(text, anchor)`` → debounced ``_apply_status(text, anchor)`` sets
``_status_lbl_f1_anchor = anchor`` only when the text is actually rendered;
``_arm_dwell(text, anchor)`` freezes ``anchor`` via the ``after`` lambda so
``_show_dwell_tip(text, anchor)`` applies the anchor for the tooltip that is
actually shown; ``_show_tip(text, anchor)`` does the same for error tooltips.
``_hide_tip`` / ``_clear_dwell_now`` clear the anchor when the label is emptied.
Chrome/path/log handlers also mirror ``_status_lbl_f1_anchor = f1`` immediately
for F1 before the debounce fires (same value ``_apply_status`` will re-apply).

ConfigSheet cells: ``_publish_status`` resolves ``_hover_detail`` (detailed
body from ``config_reference.md``) and ``_f1_anchor_for_iid(iid)``, passes both
through ``on_hover_status`` → ``App._on_cell_status`` → ``_set_status(msg, anchor)`` +
``_arm_dwell(detail, anchor)``.

## F1 — doc browser for the focused/selected widget

One root-level binding (``App._on_f1_help``; pages bind nothing — per-sheet
bindings fired once per opened tab).  Resolution order:

1. **Top path field** — keyboard focus inside ``_path_field``'s subtree
   (``App._within`` master walk) → ``help_for_path("path_field").anchor``.
2. **Status label** — mouse over ``_status_lbl`` (shows both the short status
   and the dwell tooltip) → ``_status_lbl_f1_anchor`` for **what is currently
   displayed**.  The anchor is forged at **display time**: hover handlers
   (``_on_cell_status``, ``_on_chrome_hover``, ``_on_path_hover_in``,
   ``_on_log_motion``) resolve the anchor and pass it to
   ``_set_status(text, anchor)`` / ``_arm_dwell(text, anchor)``; the anchor is
   applied only when the text is actually rendered —
   ``_apply_status(text, anchor)`` for short status (debounced, re-queued
   after dwell linger with the same anchor), ``_show_dwell_tip(text, anchor)``
   for dwell tooltips (anchor frozen via the ``after`` lambda), and
   ``_show_tip(text, anchor)`` for error tooltips.  While a dwell tooltip for
   row A is still showing, hovering row B does NOT make F1 open B — the
   displayed A tooltip keeps anchor A until it clears.  Cleared by
   ``_hide_tip`` / ``_clear_dwell_now`` when the label is emptied.
3. **Sheet row** — the page owning the focus (else the current page) asks
   ``ConfigSheet._f1_anchor``: target = the selected row
   (``sh.tree_selected`` — current selection box's iid); if nothing selected
   but the mouse is inside the dwell tooltip widget (``_hover_field``), use
   that row (``_status_iid``).  Mouse over other sheet elements is not
   tracked for F1 — nothing selected and no tooltip hover falls through to
   the readme.  Anchors resolve through the same
   ``_help_candidates`` fan-out as hover status (paired metadata rows try
   every split label), then walk ``meta["parent"]`` — child rows of an
   undocumented node inherit their ancestor's section (``Ag[0]`` →
   ``input.coefs.Ag`` → the ``input.coefs`` group).
   ``_f1_help_candidates`` reorders the fan-out for F1: a metadata row's own
   path (``input.time_ranges``) is tried before ``metadata.time_ranges`` so
   F1 opens the field's section, while status text still prefers
   ``metadata.*`` first.
4. **nothing is selected/tooltip-hovered or nothing documents
   the target** — then **Readme** using ``_about.local_readme()``
   (localized by ``resolve_lang``, base ``readme.md`` fallback) opens instead.

Anchors are section-heading slugs — ``_help._slug`` mirrors
``viewer.js::slugify``; ``{#explicit-id}`` wins, field rows inherit the
section anchor.  The doc MUST be the same localized file the entries were
parsed from — ``doc_path()`` without a lang always serves English, and a
localized anchor then finds no element (page opens, never scrolls).
Complex formulas live on methodology pages: the ``#### Detailed`` bodies
link there (e.g.
[Pressure computation from the `P_t` polynomial](../../methodology/pressure.md)) because the Tk
``MarkdownLabel`` tooltip renders plain text only — the browser typesets them
with MathJax.
