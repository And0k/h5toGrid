# GUI Help System

Two independent help sources — one per widget category:

| Source | Widgets | Key derivation | i18n mechanism |
|---|---|---|---|
| `STR` (``const.py``) | Chrome widgets (`self._path_lbl`, `_path_field`, `_overall_lbl`, `_run_btn`) + dynamic tabs | Attribute name → role → ``STR["{role}.tooltip"]`` / ``STR["{role}.status"]`` | Replace ``STR`` dict wholesale at build for target language |
| ``_help.py`` (``config_reference.md``) | Config cells (``_meta[iid]["path"]`` keys) | ``help_for_path(strip_index(path)).short`` | Replace ``config_reference_<lang>.md`` |

Companion pages: [GUI Architecture](GUI_architecture.md) ·
[GUI Widgets](GUI_widgets.md) ·
[GUI Key Decisions with Rationale and Regression Notes](GUI_decisions.md) —
index: [GUI Internals](_index.md).

## Chrome widgets: auto-registration

``App._register_chrome_help()`` runs once at the end of ``_build()``.  It walks
``vars(self)`` for all ``tk.Misc`` instances whose attribute name (minus the
leading ``_``) has entries in ``STR``.  For each match:

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

Static ``STR["{role}.status"]`` strings (defined by chrome widgets — e.g.
``path_field.status``) **do** go through the Markdown parser by default at
``_on_chrome_hover`` / ``_on_path_hover_in``, so ``**bold**`` segments in
those STR entries render bold.

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
help key.  ``_sheet_status._publish_status`` (via `ConfigSheet`) calls ``help_for_path(path)``,
which returns a ``HelpEntry(short, body)`` parsed once from the tables in
``config_reference.md``:

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
4. CamelCase field names (``Ag``, ``Cg``, ``Rz``) parse identically to
   lowercase Hydra names.
5. ``_DOC_PATH`` resolves to ``config_reference.md`` at
   ``{tcm_root.parent}/docs/reference/config_reference.md``; absent file →
   empty cache → no hover text (graceful degradation).
6. Array indices stripped at lookup time: ``Ag[0]`` / ``Ag[1][2]`` → ``Ag``.

Fallback chain in ``_publish_status`` (now in ``_sheet_status``):
``help_for_path(candidate).short`` → ``key`` / ``label`` / ``path`` (no `hover_status` cache — `time_ranges` detail is live via `_time_ranges_detail`).
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
- Next ``###`` or ``##`` closes the previous accumulation.
- Fields without ``###`` sections get ``body={}``.

**Consumer API** — ``help_for_path(path, *, mode=None, detail=None)``:

| Call | Return |
|------|--------|
| ``help_for_path("input.path")`` | ``HelpEntry(body={"probe": "...", "search": _ModeBody(...)})`` |
| ``help_for_path("input.path", mode="probe")`` | ``HelpEntry(body="...")`` — probe content (no #### → str) |
| ``help_for_path("input.path", mode="search")`` | ``HelpEntry(body="...")`` — search short body (pre-#### lines only) |
| ``help_for_path("input.path", mode="search", detail="Detailed")`` | ``HelpEntry(body="...")`` — the ``#### Detailed`` block body |
| ``help_for_path("program.return_", mode=_NO_MODE, detail="Detailed")`` | ``HelpEntry(body="...")`` — a modeless section's Detailed block |
| ``help_for_path("input.path", mode="search", detail="Unknown")`` | ``HelpEntry(body="")`` — unknown detail → empty (caller no-ops) |

To add a new mode: (1) add a ``### `field.path` <mode>new_mode</mode>``
subsection in ``config_reference.md``; (2) call ``help_for_path(path,
mode="new_mode")`` in the consumer.

## Field-level `#### Detailed` blocks (legacy, no `###` section)

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
``#### Detailed`` body:

```python
if (e := _help.help_for_path(path)) and isinstance(e.body, Mapping):
    for val in e.body.values():
        if isinstance(val, _help.ModeBody) and (d := val.details.get("Detailed")):
            return str(d)
return ""
```

``#### Detailed`` is the **only** body that arms the dwell tooltip — section
short bodies and group prose never do.

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
| PathField | ``STR["path_field.tooltip"]`` |
| ConfigSheet cells | :meth:`ConfigSheet._resolve_detail` — ``#### Detailed`` blocks only from ``config_reference.md`` (mode ``probe`` → mode ``search`` → field-level).  Mode short bodies and parent-group prose never arm the dwell — a tooltip exists ⟺ the field carries a ``Detailed`` block (regression: every coef row showed the ``input.coefs`` group text) |
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

**Architecture**:

```
_on_chrome_hover(A) → _cancel_dwell() + _set_status(A.status) + _arm_dwell(A.tooltip)
  → _DWELL_MS timer fires → _show_dwell_tip(A.tooltip) → _dwell_active = True (stays while hovered)
_on_chrome_leave(A) → _cancel_dwell() → clear scheduled after _DWELL_HIDE_MS (3 s linger)
_on_chrome_hover(B) → _cancel_dwell() + _set_status(B.status) + _arm_dwell(B.tooltip)
  → _apply_status fires after _STATUS_SETTLE_MS (0.3 s) — dwell still owns the label:
    clear scheduled at _DWELL_HIDE_MS, the switch re-queues itself right after it
```

ConfigSheet cells: ``_publish_status`` resolves ``_hover_detail`` (detailed
body from ``config_reference.md``) and passes it through ``on_hover_status``
→ ``App._on_cell_status`` → ``_arm_dwell(detail)``.

## F1 — doc browser at the hovered row's heading

``ConfigSheet._on_f1_help`` (toplevel ``<F1>`` binding): the hovered row
(``_status_iid``) resolves its config path → ``help_for_path(path).anchor``
(section heading slug — ``_help._slug`` mirrors ``viewer.js::slugify``;
``{#explicit-id}`` wins, field rows inherit the section anchor) →
``get_documentation_browser().open(doc_path(resolve_lang()), anchor=…)``.
The doc MUST be the same localized file the entries were parsed from —
``doc_path()`` without a lang always serves English, and a localized anchor
then finds no element (page opens, never scrolls).  Complex formulas
live on methodology pages: the ``#### Detailed`` bodies link there (e.g.
[Pressure computation from the `P_t` polynomial](../methodology/pressure.md)) because the Tk
``MarkdownLabel`` tooltip renders plain text only — the browser typesets them
with MathJax.
