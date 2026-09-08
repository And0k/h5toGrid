# Organizing TCM data and metadata per setup

`tcm` reuses the `meta_finder` package for everything that happens **before**
processing: finding cruise/device directories, enumerating raw files (including
inside `.zip`/`.7z`), extracting deployment time ranges and burst gaps, and
reading/writing `info_devices.yaml`. You never call `meta_finder` directly —
but its folder-structure expectations decide what the GUI/CLI finds. For the
full function-level map, see [meta_finder Integration](../reference/meta_finder_integration.md).

## Expected folder structure


```text
B:\Cruises\BalticSea\                        ← cruise root (top search dir)
├── 140228_Sambian@ADCP,ADV,i\               ← cruise dir (dated prefix)
│   └── inclinometer-Yantarniy\               ← device dir (keyword + ids)
│       ├── _raw\                             ← raw files anchor
│       │   ├── i3.txt / @i3.txt
│       │   └── _raw.zip                      ← archive with i3.txt, i4.txt, …
│       └── cfg_proc\run\                     ← generated per-probe YAMLs
├── 181005_ABP44\inclinometer\_raw\           ← another device anchor
└── DOC\GRIDDING\_raw\                        ← ignored (not a device dir)
```

Rules:

- A **cruise dir** starts with a date (`YYMMDD_…`, e.g. `140228_Sambian…`).
- A **device dir** name contains a keyword (`inclinometer`, `incl`, `tcm`,
  `wave gauge`, `pressure`, `@i…`) or a device-id suffix
  (`…@i3,5,9,w1-6`, `…@i,t-chain`), and is **valid** only with a `_raw/`
  (or `text_output/`) subdir or a `*_raw.zip`/`.7z` archive.
- **Excluded**: `DOC`, `GRIDDING`, `CTD_…` (no keyword/id match), plus names
  ending with `-`, `bad`, `test…`.

## What a parent scan does

Entering a cruise root (e.g. `B:\Cruises\BalticSea`) runs shallow discovery
first; on a miss the filtered device-dir enumeration collects every `_raw`
anchor with matching files (29 anchors in the example above), fills the path
field's inherent numbered dropdown, auto-selects the first anchor for
tab-fill, and leaves the rest one pick away. `processing.run` itself always
handles exactly one `_raw` — see [Configuration](configuration.md).

## Time ranges and bursts

For each probe `meta_finder` samples lines 1 and 20 to estimate the sampling
interval, then scans up to 3 h (`max_burst_time_detection`) for gaps larger
than `max(10 s, 2·avg)`. `tcm` takes `time_ranges` from its own edge rows
first (authoritative) and overlays `burst_dt`/`bursts_t` from this gap
detection; both-failed extractions stay at `WARNING` — never hidden. Bursts
are deployment metadata (`info_devices.yaml` indices 8–9), not `input`.

## Device metadata file

Intention: concise deployment journal, human editable and machine readable.

One per device dir, `info_devices.yaml` (fallback `.json`) holds
- one `array` per probe: point, depth, coordinates, burst settings, the period of correct operation. Or
- same `array` with `setup_id` keys `{setup_id: array}` if there was several setups/deployment intervals.

`info_devices.yaml` location: next to the raw data directory - in device data directory (parent of `_raw`)

Full spec: [meta_finder I/O formats](../../../meta_finder/docs/reference/io_formats.md#metadata-file-format).


### TCM GUI editor for metadata records

`metadata` group (under groups of `configuration`, with different background tint) edits `array` record; it is saved back on **Run** and goes into the output files.

Several deployment intervals of one device (a nested `info_devices.yaml`
entry — see the [meta_finder I/O formats](../../../meta_finder/docs/reference/io_formats.md#multiple-intervals))
are shown as autonumbered **`setup`** sublevels under `metadata`, each
labelled by its station key. A single interval stays flat (no `setup` level).
To split an interval into two adjacent ones, `right-click` the `metadata`
node or a `setup` node and use the split entries (the sheet's built-in
*Insert rows* command is intercepted here — elsewhere it inserts plain rows).
On a `metadata`/`setup` target the entries read **Insert setup N above/below**,
where `N` is the number the copy will take:

| Action | Copy placement | Copy `time_range` |
|--------|----------------|-------------------|
| Insert setup **N above** | above the selected interval | `[1] := [0]` (copy shares the start) |
| Insert setup **N below** | below the selected interval | `[0] := [1]` (copy shares the end) |

The new `setup` is numbered with the next free integer (`1` when splitting a
flat single interval — the existing flat rows first move into node `0`).
Each split is one step of the sheet's native *Undo* (`Ctrl+Z`); sorting
entries are removed from the sheet context menus.

Row protection: the six paired rows under each interval are fixed — *Insert*
above/below is disabled there (denied attempts ring the system bell), and no
new top-level node can ever be created by insertion. *Delete rows*/*Delete
columns* stay enabled only for rows/columns you added yourself: **Add row**
parents a child under the selected node (never top-level; the metadata subtree
is off-limits — split is the only way to add there) and **Add column** appends
at the end. In simplified mode before data loads the sheet is read-only and
every context-menu entry is disabled.

**`time_range` vs `input.time_ranges`** — two different things:

| Field | Meaning | Effect |
|-------|---------|--------|
| `metadata.time_range` | Journal record: start/end of the correct operation of the device at the station | Never changes a window that is already set |
| `input.time_ranges` | The processing window of the run | Everything outside is skipped |

The journal record helps only when the window is not set yet: on **Scan**
(config discovery), if `input.time_ranges` is empty or incomplete, its
missing ends are filled from `time_range`. How the window got its current
contents — journal, data edges, or manual typing — makes no difference:
**Run** never fills or overrides the window, it uses it exactly as it stands
in the YAML (empty window = the whole record is processed). So if the
journal record is wrong, correct it **before** clearing/emptying the window;
a filled window is yours alone.

Field descriptions: [Reference §`metadata`](../reference/config_reference.md#metadata--device-deployment-metadata-per-probe-infodevicesyaml).


## File names and normalization

Device-ID normalization (`i_p06` → `ip06`, `I_P01_001.txt` → `i_p01`,
`i3.txt` → `i03`) is shared logic — details:
[meta_finder I/O formats](../../../meta_finder/docs/reference/io_formats.md#device-id-normalization),
[tcm I/O formats](../reference/io_formats.md#file-name-parsing).

## See also

- [meta_finder Integration](../reference/meta_finder_integration.md) — every reused function, data flow, log origins
- [meta_finder Getting Started](../../../meta_finder/docs/user_guide/getting_started.md) — standalone collection workflow
- [meta_finder Console Messages](../../../meta_finder/docs/user_guide/console_messages.md) — message catalog
- [GUI Guide](gui.md) — anchor dropdown and scan flow
- [Console Messages](console_messages.md) — tcm-side log rows
