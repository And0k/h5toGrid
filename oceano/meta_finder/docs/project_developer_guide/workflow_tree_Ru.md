# Дерево workflow

Отображает полное дерево workflow приложения meta_finder, показывая все возможные ветки выполнения и когда каждая функциональность вызывается.

## Основные точки входа

### 1. Интерфейс командной строки (`collect.py`)

```
main() → parse_command_line_args() → initialize_config()
├── collect() workflow
│   ├── file_finder.discover_device_dirs()
│   ├── collect.process_all_metadata()
│   │   ├── collect.get_absent_meta()
│   │   │   ├── file_finder.extract_devices_from_text_output()
│   │   │   ├── collect.get_all_data_files_for_device_dir()
│   │   │   └── collect.add_all_data_paths()
│   │   └── collect.update_device_metadata_with_time_info()
│   │       ├── collect.get_prioritized_data_sources_for_time_extraction()
│   │       └── collect.extract_time_metadata_from_prioritized_sources()
│   └── file_writer.write_output_files()
└── file_writer.generate_output()
```

### 2. Программный API

```
collect.process_cruise_directories()
├── file_finder.discover_device_dirs()
└── collect.process_all_metadata()
    └── [same as above]
```

## Ветки выполнения и триггеры

### A. Ветка обнаружения устройств

**Триггер:** При обработке директорий экспедиций

```
file_finder.discover_device_dirs()
├── Ввод: top_search_dirs, input_dirs
├── Условие: Директория содержит паттерны, связанные с устройствами
│   ├── Паттерн: inclinometer|incl|tcm|wavegauge|wave_gauge|pres|@i[0-9]?
│   ├── Паттерн: ptn_device_dir_keywords
│   └── Паттерн: ptn_device_dir_sep
├── Действие: Сканировать поддиректории на предмет паттернов устройств
├── Вывод: Словарь, отображающий директории экспедиций на директории устройств
└── Используется: process_all_metadata()
```

### B. Ветка извлечения метаданных

**Триггер:** Когда `from_data=True` или когда info-файлы отсутствуют

```
collect.update_device_metadata_with_time_info()
├── Ввод: devices_data, device_dir
├── Условие: from_data=True И устройство не имеет валидных временных метаданных
├── Действие: Извлечь время из приоритизированных источников данных
│   ├── Источник 1: text_output файлы (наивысший приоритет)
│   ├── Источник 2: _raw директория файлы
│   └── Источник 3: HDF5/MAT файлы (наименьший приоритет)
└── Вывод: Обновлённые devices_data с time_st, time_en, burst_dt, bursts_t
```

### C. Ветка извлечения HDF5

**Триггер:** Когда text_output и raw файлы недоступны, и `extract_hdf5_times=True`

```
hdf5_processor.extract_metadata_from_hdf5()
├── Ввод: device_dir, device_id
├── Условие: Нет text/raw файлов И extract_hdf5_times=True
├── Действие: Сканировать HDF5 файлы в порядке приоритета
│   ├── *.proc_noAvg.h5 (приоритет 1)
│   ├── *.proc.h5 (приоритет 2)
│   ├── *.proc_Avg.h5 (приоритет 2)
│   ├── _raw/*.h5 (приоритет 3)
│   └── _raw/*.mat (приоритет 3)
└── Вывод: time_st, time_en, coef_date (если доступно)
```

### D. Ветка создания info-файлов

**Триггер:** Когда `create_info_files=True`

```
create_info_files.update_devices_meta_file()
├── Ввод: cruise_dir, device_dirs
├── Условие: create_info_files=True
├── Действие:
│   ├── Сканировать на существующие info_devices@meta_finder.yaml
│   ├── Обнаруживать устройства из файлов данных
│   ├── Создавать/обновлять YAML со значениями-заполнителями
│   └── Сохранять существующие не-заполнительные значения
└── Вывод: info_devices@meta_finder.yaml файлы
```

### E. Ветка генерации вывода

**Триггер:** Всегда (в конце обработки)

```
file_writer.write_output_files()
├── Ввод: all_devices, cruise_data
├── Действие:
│   ├── write_files_list() → {yymmdd_HHMM}_files_TCM.tsv
│   └── write_metadata_table() → {yymmdd_HHMM}_meta_TCM.tsv
└── Вывод: Два TSV файла в meta директории
```

## Поток данных

```
Директория экспедиции
    │
    ▼
Обнаружение директории устройств
    │
    ▼
Чтение файла метаданных (YAML/JSON)
    │
    ▼
Обнаружение файлов данных (text_output, _raw, HDF5)
    │
    ▼
Ассоциация устройств с данными
    │
    ▼
Извлечение времени (приоритизированные источники)
    │
    ▼
Слияние метаданных на уровне полей
    │
    ▼
Генерация вывода (TSV файлы)
```

## Условные ветки

| Условие | Ветка | Действие |
|---------|-------|----------|
| `create_info_files=True` | Создание info-файлов | Создать/обновить `info_devices@meta_finder.yaml` |
| `from_data=True` | Извлечение данных | Извлечь временные метаданные из файлов данных |
| Нет text_output файлов | HDF5 fallback | Извлечь из HDF5/MAT файлов |
| Существующие временные метаданные | Пропустить извлечение | Сохранить существующие значения |
| Устройство с множественными интервалами | Вывод с множественными строками | Создать одну строку TSV на интервал |
| `overwrite_bad_devs_in_info_files=True` | Выборочное обновление | Обновлять только устройства с пустыми значениями |

## См. также

- [CLI Internals](CLI_Ru.md)
- [Руководство по обработке](../user_guide/processing_Ru.md)
- [Анализ кодовой базы](codebase_analysis_Ru.md)
- [Обработка множественных интервалов](multiple_intervals_Ru.md)
