# TCM GUI — Inclinometer Data Processing with Live Progress

Tkinter frontend for the `tcm` processing pipeline.  Edit calibration
coefficients visually, process multiple probes, and monitor progress in
real time.

## Quick start

```bash
# Launch with a data path (auto-scans on startup)
python -m tcm_gui "D:/data/_raw"

# Launch with a specific file and probe filter
python -m tcm_gui "D:/data/_raw/@i_p1.TXT" "input.ids=[i90, i67]"
```

## Workflow

### 1. Scan — discover configs

Enter a data path in the top entry (or click **Dir…** / **Files…**) and press
Enter.  The tool discovers run YAMLs in `cfg_proc/run/` and creates one tab
**per config**.

> **Config vs probe**: one probe (e.g. `i_p05`) can have multiple configs —
> each representing a different deployment period with its own `time_ranges`,
> coefficients, and `input.path`.  Each config gets its own tab.

### 2. Edit coefficients

Each tab shows a treeview with the config's parameters:
- **`input.path`** — data file path
- **`input.time_ranges`** — time window for this deployment
- **`input.coefs`** — calibration coefficients (Ag, Cg, Ah, Ch, Rz, kVabs, P, etc.)

Click a cell to edit.  Date fields are validated (`YYYY-MM-DD` format).
Numeric fields reject non-numeric input.

**Shift-click** at startup (or hold Shift while clicking Browse) to load the
full config tree (all sections: `out`, `filter`, `program`).

### 3. Run — process data

Click **Run**.  The tool:
1. Saves edited coefficients to YAML files (timestamped backup created)
2. Processes all configs with edited YAMLs
3. Shows progress in the upper bar (per-config stages) and lower bar (dask tasks)

### 4. Monitor

- **Upper bar**: config-level progress — load → coefs → process → NC write → TSV write
- **Lower bar**: stage-level dask task progress
- **Log panel**: all pipeline log messages with color-coded severity
- **Status bar**: current stage description

### 5. Pause / Resume

Click **Run** again while processing to **Pause**.  Click again to **Resume**.
Pause freezes both logging and dask task progress at the next checkpoint.

## Browse modes

| Button | Action |
|--------|--------|
| **Dir…** | Select a data directory → auto-scan |
| **Files…** (Shift held) | Select specific data files → regex pattern → auto-scan |

## Coefficient reload

Edit the **`coefs_path`** entry at the top of each tab to load coefficients
from a different file (HDF5 or YAML).  Press Enter or click Browse to reload.

## CLI reference

```
python -m tcm_gui [PATH] [OVERRIDES...]
```

| Argument | Example | Purpose |
|----------|---------|---------|
| `PATH` | `"D:/data/_raw"` | Data path (prefills entry, auto-scans) |
| `OVERRIDES` | `"input.ids=[i90]"` | Hydra overrides passed to pipeline |

Overrides from the CLI are applied during **Scan** (config discovery).
During **Run**, only the YAML files (edited by the user) are used — CLI
overrides are not re-applied.

## See also

- [Config field reference](../tcm_clc/config_reference.md) — all YAML fields
- [Pipeline architecture](../tcm_clc/how_it_works.md) — internal design
- [GUI internals](how_gui_works.md) — programmer-facing architecture
