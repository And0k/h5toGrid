# CLI Guide

Task-oriented guide for the `meta_finder` command-line interface.
For the configuration reference, see [Configuration Reference](../reference/config_reference.md).

## Basic invocation

```bash
# Process all cruise directories
pixi run collect

# Process a specific cruise directory
pixi run collect --cruise-dir "B:/Cruises/BalticSea/250415_ABP60"

# Create/update info_devices@meta_finder.yaml files
pixi run collect --create-info-files --no-from-data

# Extract time metadata from data files
pixi run collect --create-info-files
```

## First run on new data

When processing cruise directories for the first time (or when `info_devices@meta_finder.yaml` files don't exist), the program performs device discovery from data files and creates YAML files with placeholder values. However, **time data extraction only occurs for devices that are already listed in existing YAML metadata files** to review and manually edit device lists before time extraction.

This means you typically need to run the program **twice**:

1. **First run** (with `--create-info-files --no-from-data`):
   - Discovers all devices from data files (text_output, _raw, HDF5)
   - Creates `info_devices@meta_finder.yaml` files with all discovered devices
   - Devices have placeholder values ("?") for time fields
   - Program will show warning: "NEW DEVICES FOUND - REQUIRES SECOND RUN"

2. **Second run** (with `--create-info-files`):
   - Reads existing `info_devices@meta_finder.yaml` files
   - Extracts time data from data files for devices listed in YAML
   - Updates devices with actual time ranges (time_st, time_en, burst_dt, bursts_t)
   - Preserves all other metadata from the first run

## Command switches

| Switch | Description |
|--------|-------------|
| `--interactive`, `-i` | Prompt for each Config setting in order; enter `*` to use defaults for all remaining settings |
| `--create-info-files` | Create or update `info_devices@meta_finder.yaml` files from existing data structures |
| `--no-from-data` | Skip extracting metadata from data files; only use existing metadata files |
| `--cruise-dir` | Specify a single cruise directory to process, overriding `--search-dirs` |
| `--config` | Load configuration from a JSON config file |

## Re-run behavior

The pipeline handles idempotency automatically:

| Output | Re-run behavior |
|--------|----------------|
| `info_devices@meta_finder.yaml` | **Updated** — only empty-value devices get new information |
| `meta_TCM.tsv` | **Overwritten** — regenerated from current state |
| `files_TCM.tsv` | **Overwritten** — regenerated from current state |

When `overwrite_bad_devs_in_info_files=True` (default), only devices with all empty values ("?", "-", or "") get updated with new information, while preserving existing non-placeholder values.

## See also

- [Getting Started](getting_started.md)
- [Configuration Guide](configuration.md) — config system guide
- [Input / Output Guide](input_output.md) — data formats
- [Processing Guide](processing.md) — workflow internals
- [Configuration Reference](../reference/config_reference.md) — exact config field spec
- [Input / Output Format Specification](../reference/io_formats.md) — format contracts
