# Getting Started

## What is meta_finder

**meta_finder** collects metadata from files in cruise directories for AB SIO RAS inclinometers (tilt current meters), wave gauges, and combined devices (inclinometer with pressure sensor). It scans data directories, discovers devices, extracts time ranges and other metadata, and produces consolidated TSV tables.

## Quick start

```bash
# 1. First run: discover devices and create info_devices@meta_finder.yaml files
pixi run collect --create-info-files --no-from-data

# 2. Review and edit the generated YAML files in each device directory
#    Add missing station names, depths, coordinates, etc.

# 3. Second run: extract time metadata from data files
pixi run collect --create-info-files
```

On first run the program creates `info_devices@meta_finder.yaml` files in each device directory with placeholder values. After manual review, re-run to extract actual time data.

## Workflow overview

```
┌─────────────────────────────────────────────────────────────┐
│  1. Cruise Directory Discovery                              │
│     Scan configured directories for YYMMDD_{cruise_name}/   │
├─────────────────────────────────────────────────────────────┤
│  2. Device Directory Discovery                              │
│     Find inclinometer/ wavegauge subdirectories            │
├─────────────────────────────────────────────────────────────┤
│  3. Metadata File Discovery                                 │
│     Read info_devices@meta_finder.yaml / .yaml / .json     │
├─────────────────────────────────────────────────────────────┤
│  4. Data File Discovery                                     │
│     Find text_output, _raw, HDF5 files per device          │
├─────────────────────────────────────────────────────────────┤
│  5. Time Extraction                                         │
│     Extract time ranges from prioritized data sources      │
├─────────────────────────────────────────────────────────────┤
│  6. Output Generation                                       │
│     Write meta_TCM.tsv + files_TCM.tsv                     │
└─────────────────────────────────────────────────────────────┘
```

## What you get

```text
cruise_dir/
├── inclinometers/ or wavegauges/
│   ├── info_devices@meta_finder.yaml  ← generated/updated metadata
│   ├── info_devices.yaml              ← user metadata (preserved)
│   ├── text_output/
│   │   └── *.tsv, *.txt, *.csv
│   ├── _raw/
│   │   └── *.h5, *.mat, *.txt
│   └── *.proc_noAvg.h5, *.proc.h5
└── navigation/ or map/
    └── *.gpx

meta/                                    ← output directory
├── {yymmdd_HHMM}_files_TCM.tsv          ← list of processed files
└── {yymmdd_HHMM}_meta_TCM.tsv          ← consolidated metadata table
```

## Next steps

| Task | Guide |
|------|-------|
| Use CLI | [CLI Guide](cli.md) |
| Configure | [Configuration Guide](configuration.md) |
| Understand I/O | [Input / Output Guide](input_output.md) |
| Processing details | [Processing Guide](processing.md) |
| Check moved/renamed data | [Path Checker](../../src/meta_finder/post_processing/README.md) |
| Console messages | [Console Messages](console_messages.md) |
| Output format spec | [Input / Output Format Specification](../reference/io_formats.md) |
| Config fields | [Configuration Reference](../reference/config_reference.md) |
| Internals | [CLI Internals](../project_developer_guide/CLI.md) |
