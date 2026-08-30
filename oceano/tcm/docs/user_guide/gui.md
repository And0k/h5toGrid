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

The GUI requires the full distribution (h5py, scipy, matplotlib, numba).

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

### 2. Edit coefficients

Each tab shows a treeview with the config's parameters:
- **`metadata`** — device deployment file path (`info_devices.yaml` parent of `_raw`) — always editable with browse; children are `point, symbol | sea depth, h_above | lat, lon | time_range | burst_dt/t | comment` per probe deployment, with gray example hints (`P3`, `7.5`, `↟`, `60`, `600`, `2026-07-11T12:20:12`, `deployment note`) that vanish on edit; `metadata*` (asterisk) marks unsaved edits to the device file — see [Config Reference](../reference/config_reference.md#metadata--device-deployment-metadata-per-probe-infodevicesyaml)
- **`input.path`** — data file path
- **`input.coefs_path`** — calibration coefficients source file (HDF5 or YAML)
- **`input.time_ranges`** — time window for this deployment
- **`input.coefs`** — calibration coefficients (Ag, Cg, Ah, Ch, Rz, kVabs, P, etc.)

Click a cell to edit. Date fields are validated (`YYYY-MM-DD` format).
Numeric fields reject non-numeric input.

**Path validation**: the `input.path` cell turns **red** when the path does
not exist on disk (and the Run button is disabled).  `input.coefs_path` is
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
full config tree (all sections: `out`, `filter`, `program`).

### 3. Run — process data

Click **Run**. The tool:
1. Saves edited coefficients to YAML files (timestamped backup created)
2. Processes all configs with edited YAMLs
3. Shows progress in the upper bar (per-config stages) and lower bar (dask tasks)

### 4. Monitor

- **Upper bar**: config-level progress — load → coefs → process → NC write → TSV write
- **Lower bar**: stage-level dask task progress
- **Log panel**: all pipeline log messages with color-coded severity
- **Status bar**: current stage description; preserves "Ready" at idle and
  "Done — X%" after completion (cleared only at probe boundaries)

### 5. Pause / Resume

Click **Run** again while processing to **Pause**. Click again to **Resume**.
Pause freezes both logging and dask task progress at the next checkpoint.

## Browse modes

| Button | Action |
|--------|--------|
| **Dir...** | Select a data directory → auto-scan. Reopens at the last selected folder, not its parent |
| **Files...** (Shift held) | Select specific data files → regex pattern → auto-scan |

## Coefficient reload

Edit the **`coefs_path`** row under `input` to load coefficients from a
different file (HDF5 or YAML). Press Enter or click Browse to reload.
You can also leave `coefs_path` empty or pointing to a missing file and type
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

- [Configuration Schema and Device Metadata Reference](../reference/config_reference.md) — all YAML fields with hover tooltips
- [GUI Internals](../project_developer_guide/GUI/_index.md) — programmer-facing architecture
