# Input / Output Guide

This guide describes the data formats accepted and produced by meta_finder.
For the authoritative format contracts, see [Input / Output Format Specification](../reference/io_formats.md).
For implementation details, see [CLI Internals](../project_developer_guide/CLI.md).

## Input data

The program discovers and processes data from three main sources: text data files, HDF5/MAT files, and metadata files. For authoritative format contracts (column layouts, HDF5 group structures, file priorities), see [Input / Output Format Specification](../reference/io_formats.md).

## Cruise directory structure

The program expects the following standard directory structure:

```
B:/WorkData/BalticSea/
└── YYMMDD_{cruise_name}/
    ├── *inclinometer* or *wavegauge*/
    │   ├── info_devices@meta_finder.yaml  ← priority 1 (auto-generated)
    │   ├── info_devices.yaml              ← priority 2
    │   ├── info_devices.json              ← priority 3 (deprecated)
    │   ├── text_output/
    │   │   └── *.tsv, *.txt, *.csv
    │   ├── text_output.zip
    │   ├── text_output.7z
    │   ├── *.proc_noAvg.h5
    │   ├── *.proc.h5
    │   ├── *.proc_Avg.h5
    │   └── _raw/
    │       ├── *.h5, *.mat
    │       └── *.txt, *.tsv, *.csv
    └── *navigation* or *map*/
        └── *.gpx
```

Combined structures are also supported:
- `YYMMDD{cruise_name}/{device_type}/info_devices.yaml`
- `YYMMDD{cruise_name}/info_devices.yaml` (metadata in root)
- `YYMMDD{cruise_name}/inclinometers/{YYMMDD}*/info_devices.yaml` (nested dated subdirectories)

## Output files

The program creates two output files in the `meta` directory:

| File | Description |
|------|-------------|
| `{yymmdd_HHMM}_files_TCM.tsv` | List of all processed files organized by cruise |
| `{yymmdd_HHMM}_meta_TCM.tsv` | Tab-separated table with consolidated metadata for all devices |

### Metadata table columns

See [Metadata table description](metadata_table.md) for the full column reference (names, descriptions, and data sources).

### Output format rules

- Fields not found by the program are filled with "?" sign
- Values of `burst_dt` and `bursts_t` equal to "-" mean the device operated normally without periodic shutdown
- Coordinates are recorded in degrees; if not found, "?" is written for `lat` and `lon`, and paths to all found GPX files are written in the `comment` field with prefix "GPX:"
- For combined data columns (e.g., `Vabs_i05_14`), the comment field contains "{device1}+{device2} output"

## See also

- [Getting Started](getting_started.md)
- [Processing Guide](processing.md)
- [Input / Output Format Specification](../reference/io_formats.md) — authoritative format contracts
- [Configuration Reference](../reference/config_reference.md)
