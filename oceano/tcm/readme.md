<img src="docs/images/logo.png" alt="TCM" align="right" width="256">

[←Russian](readme_Ru.md)


# TCM Data Processing

Convert raw data files measured by AB SIO RAS tilt current meters (TCM) to
text (CSV/TSV) and NetCDF data files with physical units (velocity,
inclination, direction, and pressure outputs) optionally with filtering/averaging.
Provides both a CLI pipeline (`tcm_proc`) and a GUI with live progress (`tcm_gui`).

## Distribution types

- **exe** — `tcm_proc.exe`, `tcm_gui.exe` for the operating system
  corresponding to the compilation environment (Windows 10 64 bit).
- **Source** — `python -m tcm.scripts.tcm_proc` in an installed Python
  environment with the required packages (see `pyproject.toml`).

### Execution variants

The source code or a compiled distribution works in either of two runtime
configurations, depending on the environment:

- **Full** — h5py, scipy, matplotlib, numba: complete NC/HDF5 I/O.
- **noh5** — no h5py/pytables: TSV output only, coefficients persisted to YAML.

## Documentation

### [User guides](docs/user_guide/)

- [Getting Started](docs/user_guide/getting_started.md)
- [GUI Guide](docs/user_guide/gui.md)
- [CLI Guide](docs/user_guide/cli.md)
- [Input / Output Guide](docs/user_guide/input_output.md)
- [Configuration Guide](docs/user_guide/configuration.md)
- [Processing Guide](docs/user_guide/processing.md)
- [Organizing TCM data and metadata per setup](docs/user_guide/meta_finder.md)
- [Console Messages](docs/user_guide/console_messages.md)
- [Описание журнала работы tcm_proc](docs/user_guide/log_description_ru.md)

### [Reference](docs/reference/)

Exact, authoritative specs.

- [Configuration Schema and Device Metadata Reference](docs/reference/config_reference.md)
- [Config Tuning — Decision Tables & Behavior](docs/reference/config_tuning.md)
- [CLI Reference](docs/reference/cli.md)
- [Input / Output Format Specification](docs/reference/io_formats.md)
- [meta_finder Integration](docs/reference/meta_finder_integration.md)

### [Methodology](docs/methodology/)

Theory behind the algorithms, not usage.

- [Calibration Wiki: Theory and Method](docs/methodology/calibration_wiki.md)
- [Algorithm of frequency estimation `utils_time_corr._estimate_freq_np`](docs/methodology/estimate_freq_logic.md)
- [Velocity computation from accelerometer and magnetometer](docs/methodology/velocity.md)
- [Pressure computation from the `P_t` polynomial](docs/methodology/pressure.md)

### [Python developer guide](docs/python_developer_guide/)

- [Magnetometer/Accelerometer Calibration — Usage Guide](docs/python_developer_guide/calibration.md)
- [Examples](docs/python_developer_guide/examples.py)

### [Project developer guide](docs/project_developer_guide/)

Internal architecture and build instructions.

- [CLI Internals](docs/project_developer_guide/CLI.md)

#### [GUI Internals](docs/project_developer_guide/GUI/_index.md)

- [GUI Architecture](docs/project_developer_guide/GUI/architecture.md)
- [GUI Widgets](docs/project_developer_guide/GUI/widgets.md)
- [GUI Help System](docs/project_developer_guide/GUI/help_system.md)
- [GUI Key Decisions with Rationale and Regression Notes](docs/project_developer_guide/GUI/decisions.md)

- [Documentation Authoring Contract](docs/project_developer_guide/doc_authoring.md)
- [Сборка дистрибутивов `tcm_proc` и `tcm_gui`](docs/project_developer_guide/build_tcm_clc_txt_Ru.md)
