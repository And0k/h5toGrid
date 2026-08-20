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

- [Начало работы](docs/user_guide/getting_started.md)
- [Руководство по GUI](docs/user_guide/gui.md)
- [Руководство по CLI](docs/user_guide/cli.md)
- [Ввод / Вывод](docs/user_guide/input_output.md)
- [Конфигурация](docs/user_guide/configuration.md)
- [Обработка](docs/user_guide/processing.md)
- [Сообщения консоли](docs/user_guide/console_messages.md)
- [Описание журнала работы tcm_proc](docs/user_guide/log_description_ru.md)

### [Справочные материалы](docs/reference/)

Точные, авторитетные спецификации.

- [Справочник полей конфигурации YAML](docs/reference/config_reference_Ru.md)
- [Настройка поведения](docs/reference/config_tuning.md)
- [Спецификация CLI](docs/reference/cli.md)
- [Форматы ввода / вывода](docs/reference/io_formats.md)

### [Методология](docs/methodology/)

Теория алгоритмов, а не их использование.

- [Калибровка: теория и метод](docs/methodology/calibration_wiki.md)
- [Алгоритм `_estimate_freq_np` (режим B)](docs/methodology/estimate_freq_logic.md)

### [Руководство для Python-разработчиков](docs/python_developer_guide/)

- [Калибровка магнитометра/акселерометра: сценарий использования и ограничения](docs/python_developer_guide/calibration_Ru.md)
- [Примеры](docs/python_developer_guide/examples.py)

### [Руководство для разработчиков проекта](docs/project_developer_guide/)

Внутренняя архитектура и инструкции по сборке.

- [Архитектура CLI](docs/project_developer_guide/CLI.md)
- [Архитектура GUI](docs/project_developer_guide/GUI.md)
- [Контракт написания документации](docs/project_developer_guide/doc_authoring.md)
- [Сборка минимального дистрибутива `tcm_proc`](docs/project_developer_guide/build_tcm_clc_txt_Ru.md)
