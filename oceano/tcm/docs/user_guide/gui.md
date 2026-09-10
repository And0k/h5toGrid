# GUI Guide

Tkinter frontend for the `tcm` processing pipeline with live progress.
Edit calibration coefficients visually, process multiple probes, and monitor
progress in real time.

## Launching

```bash
# Compiled distribution (recommended)
tcm_gui.exe "D:/data/_raw"

# From source
python -m tcm_gui "D:/data/_raw"

# With a specific file and probe filter
tcm_gui.exe "D:/data/_raw/@i_p1.TXT" "input.ids=[i90, i67]"
```

The GUI runs in both distribution variants. In the noh5 (text-only) build it
works identically — template hints and tooltips describing HDF5/NetCDF fields
are filtered out, and no NC/HDF5 output is produced
(see [Getting Started — Distribution types](getting_started.md)).

## Workflow

### 1. Scan — discover configs

Enter a data path in the top field (or click **Dir...** / **Files...**) and press
Enter. The tool discovers run YAMLs in `cfg_proc/run/` and creates one tab
**per config**.

> **Config vs probe**: one probe (e.g. `i_p05`) can have multiple configs —
> each representing a different deployment period with its own `time_ranges`,
> coefficients, and `input.path`. Each config gets its own tab.

Tabs appear top-to-bottom in the backend config-scan order (configs of each
probe sorted alphabetically by file name; probe groups in scan order) — Run
processes configs in exactly the same top-to-bottom order.  The first (top)
tab is selected after every scan, and the shown tree always corresponds to
the selected tab.

If the search fails (the path cannot be resolved), the top search field turns
**red**; a new search or a successful scan restores the normal color.

### Parent paths and the anchor dropdown

A cruise root (e.g. `B:\Cruises\BalticSea`) contains many `_raw` anchors.
The scan first tries the entered path shallowly; on a miss it enumerates
device directories via `meta_finder` filter and fills the **inherent
dropdown of the path field** with all anchors (`1. …\_raw`, `2. …\_raw`, …).
The caption left of the field shows the selected anchor's **number** (e.g.
`3`); any manually typed path that is not in the list restores the default
caption. The first anchor is auto-selected for tab-fill (`cfg_proc/run`
lives per `_raw`), and any other anchor is one dropdown pick away — picking
rescans that single `_raw`. Folder rules, exclusions, and burst/time sourcing:
[Organizing TCM data and metadata per setup](meta_finder.md).

### 2. Edit coefficients

Each tab shows a treeview with the config's parameters:
- **`metadata`** — device deployment file path (`info_devices.yaml` parent of `_raw`) — always editable with browse; children are `point, symbol | sea depth, h_above | lat, lon | time_range | burst_dt/t | comment` per probe deployment, with gray example hints (`P3`, `7.5`, `↟`, `60`, `600`, `2026-07-11T12:20:12`, `deployment note`) that vanish on edit; `metadata*` (asterisk) marks unsaved edits to the device file — see [Config Reference](../reference/config_reference.md#metadata--device-deployment-metadata-per-probe-infodevicesyaml)
- **`input.path`** — data file path
- **`input.coefs.path`** — calibration coefficients source file (HDF5 or YAML) — row `path` under `input.coefs` (hidden when `coefs` collapsed)
- **`input.time_ranges`** — time window for this deployment
- **`input.coefs`** — calibration coefficients (Ag, Cg, Ah, Ch, Rz, kVabs, P, etc.)

#### Insert rows

The sheet's *Insert rows above/below* (`right-click`) splits an interval into a copy pinned to the shared boundary — **above** sets the copy's `time_range[1] = time_range[0]`, **below** sets `time_range[0] = time_range[1]` — and numbers the new `setup` with the next free integer (`1` when splitting a flat single interval; existing flat rows first move into node `0`).
On a `metadata`/`setup` target the entries read *Insert setup N above/below* (`N` = the number the copy will take; sorting entries are removed from the sheet menus. Each split is a single native *Undo* step (no second undo system — other edits undo through tksheet as before). Paired rows are fixed (insert denied there and on top-level nodes — no insertion ever creates a top-level node); *Delete* applies only to self-added rows/columns (`Add row` parents a child under the selection, `Add column` appends at the end); a read-only sheet disables the whole context menu.


Click a cell to edit. Date fields are validated (`YYYY-MM-DD` format).
Numeric fields reject non-numeric input.

**Date validation**: date cells of `input.time_ranges` and
`input.calib.time_ranges_*` turn **red** when the value cannot be parsed
(ISO `2024-01-15T10:30:00` / `2024-01-15 10:30:00` or `15.01.2024` are
accepted) or breaks the ascending order of the sequence.  Run stays disabled
until `input.time_ranges` cells parse again, and Run writes every date cell in
the canonical ISO `T`-form regardless of the spelling you typed; a row with a
red (unparsable) cell keeps its stored YAML value instead. `metadata.time_range`
is journal-only and never gates Run — see
[Config Tuning §Inverted time_ranges](../reference/config_tuning.md#inverted-time_ranges).


**Path validation**: the `input.path` cell turns **red** when the path does
not exist on disk (and the Run button is disabled).  `input.coefs.path` is
**optional** — coefficients may be entered manually — but a missing file is
still flagged with red text so you can see it at a glance; it does not block
Run.  The **`metadata`** row's path cell behaves identically (red when the
device file is missing, browse works even when the file does not yet exist —
a new `info_devices.yaml` is created on Run).

**Default tint**: cells at their dataclass defaults appear dim gray (`#999`),
edited cells are black — same rule for `metadata` (`time_range` defaults to the
current `input.time_ranges[[0, -1]]` — copy-paste there grays instantly, live) and for `input.calib` date lists (`time_ranges_zeroing`/`time_ranges_azimuth`: empty means at-default).  The leaf node label is blue when its entire subtree is at defaults, black otherwise; an empty metadata child (all ghosts) is therefore blue.

**Deleting a path**: clear the floated `input.path` / `metadata` field and the cell empties, the dim placeholder returns and the path reads as `""` — a deleted path never resurrects on the next hover.

**Time window hover**: hovering `input.time_ranges` shows the live relation to `info_devices` — _matches_ (equal), _broader than_ (warning tint), _differs_ (narrowed/shifted) — recomputed on every hover/edit, not a stale scan-time message.

**Shift-click** at startup (or hold Shift while clicking Browse) to load the
full config tree (all sections: `input`, `out`, `filter`, `program` — missing
values are filled with defaults). Edits to any section are saved to the run
YAMLs on Run, same as coefficients.

**Instant apply** (data-independent triggers only): `input.calib.g0xyz`,
`input.calib.coordinates` and `input.calib.azimuth_add` each carry a ☑ cell
right of their values. Each trigger applies fully on its own — `g0xyz`
computes `Rz`, `coordinates` shifts `azimuth_shift_deg` by declination,
`azimuth_add` adds its offset — and clears only its own cells.
Empty triggers show a disabled ☑ (in sync); complete input enables it (pending).
Clicking it computes with the same pipeline calls as
Run, updates the coefs cells in-sheet and clears the trigger — all in memory,
YAML syncs on the next Run. `time_ranges_zeroing` / `time_ranges_azimuth` need
data windows and stay Run-time only. A partial/non-numeric trigger reddens its
tree label and disables Run until completed or cleared — same precedent as an
invalid `input.path`. See [Configuration §Updating coefficients
via zeroing](configuration.md#updating-coefficients-via-zeroing).

Without Shift (simplified mode) the `out` binning is fixed to full resolution —
`dt_bins = [0]`, `dt_bins_min_save_text = 0` — unless the launch command line
overrides them (e.g. `out.dt_bins=[0,600]`; see the [CLI guide](cli.md)).

### 3. Run — process data

Click **Run**. The tool:
1. Saves edited coefficients to YAML files (timestamped backup created)
2. Processes all configs with edited YAMLs
3. Shows progress live: the **left rail** attaches a vertical progress fill to
   each tab (config-level stage weights → overall fraction), and the **status
   row** shows the current stage bar + description

### 4. Monitor

- **Left rail (progress column)**: per-config fill, aligned with each tab —
  stages load → coefs → process → NC write → TSV write
- **Status row**: current stage progress (bar + description) — the same stage
  the pipeline is on (load / coefs / process / save…)
- **Log panel**: all pipeline log messages with color-coded severity. File
  paths in log lines render as links — files show their file name, directories
  the full path; hovering a link shows the full path in the status bar, and
  clicking one opens it with its OS-associated application (Ctrl+C copies a
  selection with clickable hyperlinks). Hovering a link in the status bar
  itself shows the full path in a row below the status text
- **Status bar**: current stage description; preserves "Ready" at idle and
  "Done — X%" after completion (cleared only at probe boundaries). Bare file
  paths in statuses render as links — clicking one opens the file with its
  OS-associated application

### 5. Pause / Resume

Click **Run** again while processing to **Pause**. Click again to **Resume**.
Pause freezes both logging and progress updates at the next checkpoint.

## Browse modes

| Button | Action |
|--------|--------|
| **Dir...** | Select a data directory → auto-scan. Reopens at the last selected folder, not its parent |
| **Files...** (Shift held) | Select specific data files → regex pattern → auto-scan |

## Coefficient reload

Edit the **`path`** row under `input.coefs` to load coefficients from a
different file (HDF5 or YAML) — hidden when the `coefs` section is collapsed. Press Enter or click Browse to reload.
You can also leave `path` empty or pointing to a missing file and type
coefficients manually in the `input.coefs` section — the Run button stays
available (only `input.path` gates it).

## Help

- **Hover** a row: the status bar shows the field's short help; staying on the
  row opens a detailed tooltip (auto-closes after 3 s; status switches settle
  in 0.3 s so quick pointer passes don't flicker).
- **F1**: opens the documentation browser — at the selected row's section in
  the config reference (child rows open their parent's section), at the
  `path_field` section when the path entry has focus, or the readme when
  nothing specific applies. Complex formulas live on methodology pages
  (e.g. pressure `P_t`) and are linked from the tooltip.

## CLI reference

```
tcm_gui.exe [PATH] [OVERRIDES...]
```

| Argument | Example | Purpose |
|----------|---------|---------|
| `PATH` | `"D:/data/_raw"` | Data path (prefills entry, auto-scans) |
| `OVERRIDES` | `"input.ids=[i90]"` | Hydra overrides passed to pipeline |

Overrides from the CLI are applied during **Scan** (config discovery).
During **Run**, only the YAML files (edited by the user) are used — CLI
overrides are not re-applied.

## See also

- [Organizing TCM data and metadata per setup](meta_finder.md) — folder structure, device dirs, bursts
- [Configuration Schema and Device Metadata Reference](../reference/config_reference.md) — all YAML fields with hover tooltips
- [GUI Internals](../project_developer_guide/GUI/_index.md) — programmer-facing architecture
