# Внутреннее устройство CLI

## Архитектура модуля

```
src/meta_finder/
├── __init__.py
├── config.py              ← Настройки конфигурации, аргументы CLI, константы
├── collect.py             ← Основная точка входа приложения, process_all_metadata()
├── file_finder.py         ← Обнаружение файлов в структурах директорий
├── metadata_extractor.py  ← Извлечение метаданных из YAML/JSON и GPX файлов
├── data_processor.py      ← Высокоуровневая оркестрация обработки данных
├── data_proc_funcs.py     ← Конкретные функции обработки файлов (извлечение времени)
├── file_writer.py         ← Генерация выходных файлов (TSV)
├── hdf5_processor.py      ← Извлечение метаданных HDF5/MAT файлов
├── io_info_files.py       ← Ввод/вывод YAML/JSON info файлов
├── parse_data_file_name.py ← Анализ имён файлов и нормализация идентификаторов устройств
├── parse_cruise_dir_name.py ← Анализ имён директорий экспедиций и построение имён наборов данных
├── create_info_files.py   ← Создание/обновление info_devices@meta_finder.yaml
├── utils_sys.py           ← Системные утилиты для обработки архивов
└── README.md              ← Устаревший (содержание перераспределено в docs/)
```

## Точка входа

`collect.py` — основная точка входа, использующая `argparse` для разбора CLI.

### CLI

```bash
# Обработать все директории экспедиций
pixi run collect

# Обработать конкретную директорию экспедиции
pixi run collect --cruise-dir "B:/Cruises/BalticSea/250415_ABP60"

# Создать/обновить info_devices@meta_finder.yaml файлы
pixi run collect --create-info-files --no-from-data

# Извлечь временные метаданные из файлов данных
pixi run collect --create-info-files

# Интерактивный режим
pixi run collect --interactive
```

### Flow

1. `main()` → `parse_command_line_args()` → `initialize_config()`
2. `collect()` workflow:
   - `file_finder.discover_device_dirs()`
   - `collect.process_all_metadata()`
     - `collect.get_absent_meta()`
       - `file_finder.extract_devices_from_text_output()`
       - `collect.get_all_data_files_for_device_dir()`
       - `collect.add_all_data_paths()`
     - `collect.update_device_metadata_with_time_info()`
       - `collect.get_prioritized_data_sources_for_time_extraction()`
       - `collect.extract_time_metadata_from_prioritized_sources()`
   - `file_writer.write_output_files()`

### Программный API

```python
from meta_finder.collect import process_cruise_directories

process_cruise_directories(
    top_search_dirs=[Path("B:/Cruises/BalticSea")],
    create_info_files=True,
    from_data=True,
)
```

## Ключевые функции

### Обнаружение файлов

- `file_finder.find_cruise_directories(search_dirs)` — Находит все директории экспедиций в указанных директориях поиска.
- `file_finder.find_device_dirs(cruise_dir)` — Находит поддиректории устройств в директории экспедиции.
- `file_finder.extract_devices_from_text_output(text_output_dir)` — Извлекает идентификаторы устройств из имён файлов данных или содержимого.
- `file_finder.discover_datafiles_for_all_dev_in_dev_dir(device_dir)` — Обнаруживает файлы данных для всех устройств в директории устройства.

### Извлечение метаданных

- `metadata_extractor.read_metadata_files_to_dict(json_path)` — Извлекает метаданные из файлов метаданных в порядке приоритета: `info_devices@meta_finder.yaml`, `info_devices.yaml`, или `info_devices.json`.
- `data_proc_funcs.extract_time_info_from_text_file(dir_archive, rel_path, averaging_interval)` — Извлекает начальное, конечное время из текстового файла данных. Также извлекает информацию о burst-режиме, когда задан интервал осреднения.
- `data_proc_funcs.extract_time_ranges_from_combined_file(file_path, device_ids)` — Извлекает временные диапазоны для каждого устройства из комбинированного файла данных.
- `hdf5_processor.extract_time_range_from_hdf5_table()` — Извлекает временные метаданные из таблиц HDF5.

### Обработка данных

- `data_processor.sort_data_paths(data_paths, device_ids)` — Сортирует пути данных по приоритету для извлечения времени.
- `data_processor.get_h5_type_and_priority(file_path)` — Определяет тип и приоритет HDF5 файла.
- `collect.get_absent_meta(meta_in, device_dir, ...)` — Создаёт содержимое для сохранения файлов метаданных.
- `collect.process_all_metadata(cruise_and_its_dev_dirs, ...)` — Основная функция для обработки всех метаданных.

### Запись файлов

- `file_writer.write_files_list(json_metadata, out_path, write_1st_paths)` — Записывает список всех собранных путей в `{yymmdd_HHMM}_files_TCM.tsv`.
- `file_writer.write_metadata_table(metadata_list, meta_tcm_path, write_1st_paths)` — Записывает таблицу метаданных в `{yymmdd_HHMM}_meta_TCM.tsv`.

### Вспомогательные функции

- `parse_data_file_name.parse_filename_for_metadata(filename)` — Разбирает имя файла для извлечения идентификатора устройства и интервала осреднения.
- `parse_data_file_name.normalize_device_id(device_id)` — Нормализует идентификатор устройства (удаляет подчёркивания, приводит к нижнему регистру и т.д.).
- `parse_cruise_dir_name.add_dataset_name(device_dir, cruise_dir, ...)` — Строит уникальное имя набора данных из имён директорий экспедиции и устройства.
- `utils_sys.read_first_last_lines(archive_path, inner_file, skip_header)` — Читает первые и последние строки файла внутри ZIP или 7z архива.

## Структуры данных

Внутренняя структура данных организует информацию следующим образом:

```python
cruise_path: {
    path of metadata file found: {
        device_name: {
            "data_paths": {(text_output_dir_path, data_file_relative_path): dataname_metadata},
            "gpx": (path to device gpx dir or cruise gpx dir),
            **metadata_from_json
        }
    }
}
```

где:
- `dataname_metadata` — метаданные, извлечённые из имени файла text output данных
- `(text_output_dir_path, data_file_relative_path)` — ключи, отсортированные с описанным приоритетом

## См. также

- [Руководство по началу работы](../user_guide/getting_started_Ru.md)
- [Руководство по обработке](../user_guide/processing_Ru.md)
- [Анализ кодовой базы](codebase_analysis_Ru.md)
- [Дерево workflow](workflow_tree_Ru.md)
