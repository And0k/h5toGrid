# Console Messages

Log files (`meta/{yymmdd_HHMM}_{meta_finder or mf_{abbreviated config}}.log`) contain WARNING and ERROR level messages.

## Logging levels

| Level | Used for |
|-------|----------|
| `WARNING` | Anomalies requiring user attention: new devices found, missing metadata, extraction failures |
| `INFO` | Pipeline milestones: discovery summary, files processed, metadata written |
| `DEBUG` | Diagnostic detail: per-device extraction, file matching, priority sorting |

## Key messages

### Two-run requirement

When new devices are discovered during the first run, the program emits:

```
WARNING: NEW DEVICES FOUND - REQUIRES SECOND RUN
```

This indicates that `info_devices@meta_finder.yaml` files have been created with placeholder values and a second run is needed to extract actual time metadata.

### Stale configs

When existing metadata files reference data files that no longer exist:

```
WARNING: Stale configs detected for devices: i01, i03
```

### HDF5 extraction

When HDF5 extraction is skipped because metadata already exists:

```
DEBUG: Skipping HDF5 extraction for i01 — time metadata already present
```

## Test logging

For centralized logging in tests, use:

```python
from utils.logging_config import setup_logging
logger = setup_logging(__name__, console_level=logging.DEBUG, file_level=logging.DEBUG)
```

All test logs are written to `test_data/meta_temp/logs/` during the test session.

## See also

- [Getting Started](getting_started.md)
- [Processing Guide](processing.md)
- [Configuration Guide](configuration.md)
