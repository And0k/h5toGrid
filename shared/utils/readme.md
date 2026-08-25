# `utils` — shared monorepo utilities

## Logging (unified, two-layer)

All monorepo logging concentrates here; subprojects keep only declarative configs
(e.g. tcm's `cfg/cfg_proc/hydra/job_logging/colorlog.yaml`).

### `logging_config.py` — handler/setup layer
Configures handlers, formatters and the root logger. Cooperates with hydra
`dictConfig` instead of fighting it: `setup_logging()` attaches console/file
handlers **only** when the root logger is unconfigured (`force=True` overrides) —
library imports can never hijack a host app's colorlog console or queue handlers.

- `setup_logging(name, ..., log_file_dir, force, use_custom_logger, package_prefix)` — console + optional timestamped UTF-8 file; no-op on an already-configured root
- `CustomLogger` — real call-site reporting (skips internals frames, exception-origin lineno, same-module `caller>callee` display)
- `SafeStringFormatter` / `get_formatter()` — UTF-8 sanitization, own-package prefix stripping, VSCode-clickable tracebacks (`File "p:123"`)
- `colored_formatter()` — single colorlog-based colored formatter (plain fallback)
- `load_yaml_logging(path)` — apply a declarative logging YAML via `dictConfig`, hydra optional
- `add_file_handler(logger, dir, name)` — attach-only file logging (never touches console handlers)
- `init_logging()` — legacy basicConfig-based script logging (moved from log_init.py)

### `log_init.py` — message-style layer
Used at call sites: `lf = LoggingStyleAdapter(__name__)`.
- `Message` — lazy `{}`-style (str.format) records
- `LoggingStyleAdapter` — `{}`-style logging over stdlib `%`-style core
- `LoggingContextFilter` — correct caller funcName/lineno through adapter layers
- `LoggingFilter_DuplicatesOption` — suppress repeats (`{"filter_same": n}`) or append repeat counters (`{"add_same_counter": n}`)
- `my_logging(name)` — adapter + duplicate filter in one call

### Consumers
- meta_finder: plain `logging.getLogger(__name__)` per module; its CLIs call `setup_logging(...)` once at entry (importing `meta_finder.*` has no logging side effects — regression-tested in `oceano/meta_finder/tests/test_logging_no_side_effects.py`)
- get_datasets: `from utils.logging_config import setup_logging`

---

# Key name-driven conversions by `type_fix` (`utils/init.py`)

| Prefix/Suffix | Converts to | Name change | Example |
|---|---|---|---|
| `dt_*` (prefix) | `timedelta` | suffix stripped if unit name, else kept | `dt_hole_warning=600` → `timedelta(600s)` |
| `*_path` / `path_*` | `Path` | kept | `path="/raw/i.txt"` → `Path("/raw/i.txt")` |
| `*_date` / `*_time` | `datetime` | suffix stripped | `min_date="2024-01-01"` → `datetime(2024,1,1)` |
| `*_int` / `*_integer` / `*_index` | `int` | suffix stripped | `count_int="5"` → `5` |
| `*_float` | `float` | suffix stripped | `ratio_float="1.5"` → `1.5` |
| `*_bool` / `*_b` | `bool` | suffix stripped | `flag_b="True"` → `True` |
| `*_list` / `*_names` | `list` | suffix stripped, comma-split | `ids_list="a,b"` → `["a","b"]` |
| `*_dict` | `dict` | suffix stripped, colon-split | `cfg_dict="k:v"` → `{"k":"v"}` |
| `min_*` / `max_*` / `fixed_*` / `float_*` (catch-all) | `float` | kept | `min_Mx="0.1"` → `0.1` |

Note: ALL `dt_*`-prefixed keys become `timedelta`, even
when the suffix is not a recognised duration unit (e.g. `dt_hole_warning`,
`dt_bins`).  The default unit is `seconds`.  Consumers must handle `timedelta`
values — use `val.total_seconds()` to extract numeric seconds.
