# Обработка данных инклинометров

<p align="center">
  <img src="docs/images/logo.png" alt="TCM" width="128">
</p>

Преобразование сырых данных, измеренных инклинометрами течений (TCM)
АО ИО РАН, в текстовые файлы (CSV/TSV) и NetCDF с физическими величинами
(скорость, наклон, направление и давление) с возможной фильтрацией и
усреднением. Включает интерфейс командной строки (CLI) `tcm_proc` и графический
интерфейс (GUI) с отслеживанием прогресса `tcm_gui`.

## Типы дистрибутива

- **exe** — `tcm_proc.exe`, `tcm_gui.exe` для операционной системы,
  соответствующей среде компиляции (Windows 10 64 bit).
- **Исходный код** — `python -m tcm.scripts.tcm_proc` в установленной среде
  Python с необходимыми пакетами (см. `pyproject.toml`).

### Варианты выполнения

Исходный код или собранный дистрибутив работает в одной из двух конфигураций
среды исполнения:

- **Полный** — h5py, scipy, matplotlib, numba: полный ввод-вывод NC/HDF5.
- **noh5** — без h5py/pytables: только TSV, коэффициенты тоже сохраняются только в открываемом любым редактором YAML формате ().

## Документация

### [Руководства пользователя](docs/user_guide/)

- [Getting Started](docs/user_guide/getting_started.md)
- [GUI Guide](docs/user_guide/gui.md)
- [CLI Guide](docs/user_guide/cli.md)
- [Input / Output Guide](docs/user_guide/input_output.md)
- [Configuration Guide](docs/user_guide/configuration.md)
- [Processing Guide](docs/user_guide/processing.md)
- [Organizing TCM data and metadata per setup](docs/user_guide/meta_finder.md)
- [Console Messages](docs/user_guide/console_messages.md)
- [Описание журнала работы tcm_proc](docs/user_guide/log_description_ru.md)

### [Справочные материалы](docs/reference/)

Точные, авторитетные спецификации.

- [Справочник по полям схемы конфигурации и метаданным устройств](docs/reference/config_reference_Ru.md)
- [Config Tuning — Decision Tables & Behavior](docs/reference/config_tuning.md)
- [CLI Reference](docs/reference/cli.md)
- [Input / Output Format Specification](docs/reference/io_formats.md)
- [meta_finder Integration](docs/reference/meta_finder_integration.md)

### [Методология](docs/methodology/)

Теория алгоритмов, а не их использование.

- [Calibration Wiki: Theory and Method](docs/methodology/calibration_wiki.md)
- [Algorithm of frequency estimation `utils_time_corr._estimate_freq_np`](docs/methodology/estimate_freq_logic.md)
- [Вычисление скорости течения по данным акселерометра и магнетометра](docs/methodology/velocity_Ru.md)
- [Вычисление давления по полиному `P_t`](docs/methodology/pressure_Ru.md)

### [Руководство для Python-разработчиков](docs/python_developer_guide/)

- [Калибровка магнитометра/акселерометра: сценарий использования и ограничения](docs/python_developer_guide/calibration_Ru.md)
- [Примеры](docs/python_developer_guide/examples.py)

### [Руководство для разработчиков проекта](docs/project_developer_guide/)

Внутренняя архитектура и инструкции по сборке.

- [CLI Internals](docs/project_developer_guide/CLI.md)
#### [GUI Internals](docs/project_developer_guide/GUI/_index.md)
- [GUI Architecture](docs/project_developer_guide/GUI/architecture.md)
- [GUI Widgets](docs/project_developer_guide/GUI/widgets.md)
- [GUI Help System](docs/project_developer_guide/GUI/help_system.md)
- [GUI Key Decisions with Rationale and Regression Notes](docs/project_developer_guide/GUI/decisions.md)

- [Правила создания документации](docs/project_developer_guide/doc_authoring_Ru.md)
- [Сборка минимального дистрибутива `tcm_proc`](docs/project_developer_guide/build_tcm_clc_txt_Ru.md)
