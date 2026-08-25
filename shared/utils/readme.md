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
