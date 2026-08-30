# GUI Internals

Optional Tkinter frontend wrapping `tcm.cli.call_in_raw_dir` in a background
thread.  No custom CLI parsing — Hydra handles all config keys natively via
`sys.argv` (see [CLI entry point](../../../readme.md)).

The internals are split into focused pages:

- [GUI Architecture](architecture.md) — module map, data flow (Scan / Run / Pause), dark/light theme, i18n, type-aware cell rendering, full mode, pipeline patches, progress plumbing, CLI integration
- [GUI Widgets](widgets.md) — browse button, PathField, floated PathField, SheetHoverBinder, rich clipboard, TabRail
- [GUI Help System](help_system.md) — chrome auto-registration, doc-driven config-cell hover, `config_reference.md` parser internals, error / dwell tooltips, F1 doc browser
- [GUI Key Decisions with Rationale and Regression Notes](decisions.md)

## Development log

Task-planning records of GUI changes live next to this page:

- [done/](done/) — completed task records (incl. this page split: `improve.md`)
- [todo/](todo/) — open planning (`plan_journal.md`)
