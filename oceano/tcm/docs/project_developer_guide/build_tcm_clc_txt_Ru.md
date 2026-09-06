# Сборка дистрибутивов `tcm_proc` и `tcm_gui`

**tcm_proc** — самодостаточный exe-файл для обработки данных инклинометров
(AB SIO RAS TCM) **без** зависимостей от HDF5 (h5py, pytables) и Intel MKL.
Вместо MKL используется OpenBLAS.

Собранный exe не требует установленного Python — все упаковано в один каталог
с помощью PyInstaller.

---

## 1. Требования

-   **ОС:** Windows 10/11 64-bit
-   **Инструмент сборки:** [pixi](https://pixi.sh/latest/) (менеджер окружений conda + pip)
-   **Размер на диске после сборки:** ~280 МБ

---

## 2. Сборка на Windows

### 2.1. Окружение `noh5-tcm`

Используется выделенное pixi-окружение **`noh5-tcm`** (solve-group `noh5`),
в котором все зависимости BLAS/LAPACK собраны с OpenBLAS, а Intel MKL отсутствует.

Окружение собирается из двух features (`pyproject.toml`):

| Feature | Ключевые пакеты | Назначение |
|---------|----------------|------------|
| `noh5` | `libblas`/`libcblas`/`liblapack` (build=`*openblas*`), `pyinstaller`, `pyinstaller-hooks-contrib` | BLAS без MKL, инструмент сборки |
| `tcm` | `python>=3.11`, `dask-core>=2024.4`, `xarray>=2026.7`, `absioras-tcm` (editable), `pygeomag>=1.1.0` | Вычислительное ядро |

Ключевые пакеты окружения:

| Пакет              | Назначение                           |
| ------------------ | ------------------------------------ |
| `python >=3.11`    | Разрешается до 3.14 (`noh5-tcm` использует 3.14.6) |
| `numpy`            | Численные расчёты                    |
| `pandas`           | Табличные данные                     |
| `xarray`           | Многомерные массивы                  |
| `dask-core`        | Параллельные вычисления (только ядро, без `dask.dataframe`) |
| `hydra-core`       | Конфигурация CLI                     |
| `hydra-colorlog`   | Цветной лог                          |
| `pygeomag`         | Геомагнитные модели (IGRF)           |
| `libblas`/`libcblas`/`liblapack` | OpenBLAS-вариа                      |
| `numba`            | JIT-оптимизации                      |

**Важно:** `noh5-tcm` **не включает** `h5py`, `pytables`, `scipy`, `matplotlib` —
эти пакеты остаются в `bin-optim-tcm` / `ocean`-features.

Отдельно указаны build-строки для BLAS-пакетов:

```toml
libblas   = { version = "*", build = "*openblas*" }
libcblas  = { version = "*", build = "*openblas*" }
liblapack = { version = "*", build = "*openblas*" }
```

Это гарантирует, что `libblas.dll`, `libcblas.dll`, `liblapack.dll` будут
обёртками над `openblas.dll`, а не над `mkl_rt.3.dll`.

### 2.2. Запуск сборки

Через pixi-задачу (рекомендуется):
```bash
pixi run -e noh5-tcm build-tcm-clc-txt
```

Задача объявлена в feature `noh5` (нужен pyinstaller), поэтому доступна в
окружениях, включающих его (`noh5-tcm`, `bin-optim-tcm`, …), и устанавливает `BUILD_MODE=manual`:
```toml
[tool.pixi.feature.noh5.tasks.build-tcm-clc-txt]
cmd = "python oceano/tcm/scripts/build/build_tcm_proc.py"
env = { BUILD_MODE = "manual" }
```

`BUILD_MODE=manual` — версия генерируется из текущей даты (`YYYY.MM`).
При `BUILD_MODE=auto` (задача `build-tcm-clc-txt-auto`) версия читается из
`oceano/tcm/scripts/build/version_meta.json`.

Или напрямую через скрипт-обёртку:
```bash
pixi run -e noh5-tcm python oceano/tcm/scripts/build/build_tcm_proc.py
```

Каталоги вывода: по умолчанию `build/` и `dist/` создаются внутри `oceano/tcm/`.
Аргумент `--build-root` (передаётся задаче как её аргумент) выносит тяжёлый
вывод на другой диск — папка получает суффикс окружения, поэтому сборки разных
env не пересекаются:
```bash
pixi run -e noh5-tcm build-tcm-clc-txt --build-root B:\Temp
# → B:\Temp\tcm_proc-env=noh5-tcm\dist\tcm_proc\tcm_proc.exe
```

Скрипт-обёртка (`scripts/build/build_tcm_proc.py`):
1. Записывает/читает `version_meta.json` (версия, продукт, URL репозитория/документации)
2. Запускает `generate_version_info` — создаёт `version_info.txt` с Windows-ресурсами
3. Вызывает PyInstaller с spec-файлом

`version_meta.json` — единый источник метаданных сборки:
- Используется PyInstaller для Windows file properties (CompanyName, FileDescription, LegalCopyright, ProductName и др.)
- Упаковывается в exe и читается runtime для окна «О программе»
- URL репозитория получается программно из `git remote origin.url`
- URL документации автоформируется как `{repo_url}/tree/{branch}/oceano/tcm/docs`

Результат: `dist/tcm_proc/tcm_proc.exe` + сопутствующие файлы.

### 2.3. Структура spec-файла

`scripts/build/tcm_proc.spec` — конфигурация PyInstaller. Ключевые моменты:

-   **Точка входа:** `scripts/tcm_proc.py`
-   **Явно добавленные DLL** (из `_ENV_LIB_BIN`):
    `openblas.dll`, `libcblas.dll`, `libblas.dll`, `liblapack.dll`,
    `libmpdec-4.dll`, `liblzma.dll`, `libexpat.dll`, `ffi-8.dll`,
    `yaml.dll`, `sqlite3.dll`, `libzmq-mt-4_3_5.dll`,
    `tbb12.dll`, `tbbmalloc.dll`, `tbbmalloc_proxy.dll`.
-   **Фильтрация бинарников** (post-analysis, `_should_keep_binary`):
    из собранного набора удаляются всё, содержащее в имени:
    `mkl_`, `.h5`, `.hdf5`, `pyarrow`, `parquet.dll`, `libzstd.dll`,
    `_zstd`, `botocore`, `certifi`, `charset_normalizer`, `google_crc32c`,
    `numcodecs`, `zstd`, `lz4`.
    А также `.pyd`-расширения pyarrow (`_parquet`, `_orc`, `_dataset`,
    `_fs`, `_gcsfs`, `_s3fs`, `_flight`, `_gandiva`, `_acero`, и др. —
    полный список в `_exclude_pyarrow_pyd`).
-   **Данные** (дерево зеркалит dev-репозиторий — `_MEIPASS` ≙ корень репо, поэтому
    кросс-ссылки документации `../../…` → проект и `../../../…` → `oceano/` работают
    в заморозке так же, как в dev):
    - Исходный код `src/tcm/` → `oceano/tcm/src/tcm/`
    - Исходники editable-пакетов — repo-относительные пути из
      `spec_common.FIRST_PARTY_PKGS` переносятся как есть (дерево зеркалит dev-репозиторий):
      `shared/utils/src/utils/`, `shared/veusz_helpers/src/veusz_helpers/`,
      `oceano/meta_finder/src/meta_finder/`
      (без `veuszPropagate`, тестов, `copy/`, `descript.ion`, `AGENTS.md`, `*-.py`,
      `__pycache__`)
    - Документация: `docs/` → `oceano/tcm/docs/`, docs проектов-сиблингов
      (meta_finder) → `oceano/meta_finder/docs/` (кроме `todo.md`,
      `potential_functionality_and_improvement.md`)
    - Конфиги Hydra: `collect_data_files("hydra", subdir="conf")` +
      `collect_data_files("hydra_plugins.hydra_colorlog")`
    - Данные `pygeomag` (через `collect_data_files`)
    - `__init__.py` в `hydra/conf/` и `hydra_plugins/hydra_colorlog/conf/`
      для корректной работы `importlib.resources`
    - METADATA пакетов `pandas` и `numpy`
-   **Исключения pure-Python:**
    - Все модули HDF5: `h5py`, `tables`, `pytables`, `hdf5`, `tcm.h5*`,
      `tcm.incl_h5*`, `tcm.incl_calibr_hy`
    - Все MKL-модули (`mkl`, `mkl_rt`, `mkl_core`, `mkl_intel_thread`, и т.д.)
    - `pyarrow` целиком + его транзитивные зависимости (`botocore`, `certifi`,
      `charset_normalizer`, `google_crc32c`, `numcodecs`, `zstandard`)
    - Тяжёлые библиотеки: `scipy`, `matplotlib`, `bokeh`, `sklearn`,
      `IPython`, `jupyter*`, `PIL`, `lxml`, `openpyxl`, `cryptography`,
      `tkinter`, `sphinx`, `pytest`, `setuptools`, `pip`, и др.
    - `distributed` (планировщик dask)
    - Модули `tcm.*`, `tcm_gui.*`, `utils.*`, `veusz_helpers.*`, `meta_finder.*`
      исключаются из `pure` и собираются только из `datas` (исходники): статически
      видимая цепочка импортов обрывается на datas-коде `tcm` — Analysis не видит ни
      `from utils import …` / `import meta_finder` внутри него, ни ленивых импортов
      datas-модулей, поэтому только `datas` гарантируют все сабмодули (дублирование
      при заморозке исключено). Документация пакетов идёт туда же
      (`oceano/<proj>/docs/`, см. выше).

    Роль `hiddenimports` сужается до модулей, достижимых **только** через
    datas-код: `http.server`, `webbrowser`, `tksheet` в `tcm_gui.spec` — их
    импортируют `browser/server.py` и лист коэффициентов из исключённого из
    `pure` `tcm_gui.*`, поэтому Analysis их не видит: транзитивные сторонние
    зависимости datas-пакетов:
    `tcm_proc.py` → `tcm.cli`, `tcm_gui.py` → `tcm_gui.app`)
    импортируется явно, поэтому Analysis их видит.
    Все first-party пакеты собираются как datas, благодаря чему ссылки браузера
    документации на `…/src/tcm/*.py`, `…/src/meta_finder/*.py` работают.

-   **Runtime hooks:**
    - `rthook_repo_layout.py` — первым добавляет `shared/` и `oceano/*/src`
      в `sys.path`, чтобы импорты datas-пакетов (`tcm`, `utils`, …) находили
      зеркальное дерево
    - `rthook_hydra_pkg.py` — patch argparse для Python 3.14 + регистрация
      плагинов Hydra (см. § 2.4)
    - `rthook_noh5_bins.py` — переопределение `out/base` в ConfigStore
      (см. § 2.5)

### 2.4. Runtime hook (`rthook_hydra_pkg.py`)

Выполняет две задачи:

**① Python 3.14 argparse-patch (до регистрации Hydra):**

Начиная с Python 3.14, `argparse.HelpFormatter._get_help_string` вызывает
`_check_help`, который проверяет `'%' not in help_string`. Если значение
`help=` — не строка (например, lazy-doc wrapper от dask), возникает
`ValueError: badly formed help string`. Хук перехватывает
`_get_help_string` и приводит non-string значения к `str()`.

**② Регистрация плагинов Hydra (для pkg:// и file:// схем):**

```python
from hydra.core.plugins import Plugins
p = Plugins.instance()

from hydra._internal.core_plugins.importlib_resources_config_source import ImportlibResourcesConfigSource
from hydra._internal.core_plugins.file_config_source import FileConfigSource
from hydra._internal.core_plugins.structured_config_source import StructuredConfigSource
from hydra._internal.core_plugins.basic_launcher import BasicLauncher
from hydra._internal.core_plugins.basic_sweeper import BasicSweeper

for cls in [ImportlibResourcesConfigSource, FileConfigSource,
            StructuredConfigSource, BasicLauncher, BasicSweeper]:
    p.register(cls)

try:
    from hydra_plugins.hydra_colorlog.colorlog import HydraColorlogSearchPathPlugin
    p.register(HydraColorlogSearchPathPlugin)
except ImportError:
    pass
```

Без этого хука exe выдавал ошибку:
`No config source registered for schema pkg`.

### 2.5. Runtime hook (`rthook_noh5_bins.py`)

Переопределяет умолчания усреднения для текстового вывода в собранном
дистрибутиве — **только** при сборке в окружении `noh5-tcm`:

```python
import tcm.schema  # форсирует регистрацию ConfigStore с dev-умолчаниями
from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()
cs.store(
    group="out",
    name="base",
    node=tcm.schema.ConfigOut_InclProc(
        dt_bins=[0, 3600],            # без усреднения + 1 час
        dt_bins_min_save_text=0,      # разрешить TSV для bin=0
    ),
    provider="noh5-rthook",
)
```

**Результат:** в собранном `tcm_proc.exe` выходные текстовые файлы
формируются с двумя настройками усреднения:

| dt_bin | Описание | TSV-файл |
|--------|----------|----------|
| `0` | Без усреднения (raw → physical) | `{ts}@i_01.tsv` |
| `3600` | Усреднение 1 час | `{ts}bin3600s@i_01.tsv` |

Для сравнения, в dev-окружении (где доступны h5py/pytables) умолчания
остаются `[0, 2, 600, 3600, 7200]` с `dt_bins_min_save_text=1`.

---

## 3. Сборка GUI (`tcm_gui`)

**tcm_gui** — самодостаточный exe-файл с Tkinter-интерфейсом для обработки
данных инклинометров. В отличие от `tcm_proc`, GUI включает:
- Графический интерактивный редактор коэффициентов (`tksheet`)
- Визуальный контроль прогресса обработки
- Поддержку HDF5 (чтение/запись через `h5py`)
- Поддержку `scipy`, `matplotlib`, `numba` (из `bin-optim` feature)

### 3.1. Окружение `bin-optim-tcm`

Используется pixi-окружение **`bin-optim-tcm`** (solve-group `noh5`),
которое объединяет features `noh5` + `bin-optim` + `tcm` + `test` + `browser`:

| Feature      | Ключевые пакеты                           | Назначение                    |
| ------------ | ----------------------------------------- | ----------------------------- |
| `noh5`       | OpenBLAS, `pyinstaller`                   | BLAS без MKL, инструмент сборки |
| `bin-optim`  | `h5py`, `scipy`, `matplotlib`, `numba`, `xarray`, `netcdf4` | Полный вычислительный стек    |
| `tcm`        | `dask-core`, `absioras-tcm`, `pygeomag`   | Ядро обработки TCM            |
| `test`       | `pytest`, `pytest-mock`                   | Тестирование                  |
| `browser`    | `nodejs >=20`                             | Генерация runtime браузера документации |

**Важно:** `bin-optim-tcm` **включает** `h5py`, `scipy`, `matplotlib` —
в отличие от `noh5-tcm`, эти пакеты доступны и собираются в дистрибутив.

### 3.2. Запуск сборки

Через pixi-задачу (рекомендуется):
```bash
pixi run -e bin-optim-tcm build-tcm-gui
```

Задача объявлена в feature `tk-gui` — доступна только в окружениях с tksheet
(`bin-optim-tcm`, `noh5-tcm-gui`); в остальных (напр. `noh5-tcm`) pixi сразу не
найдёт задачу, а `tcm_gui.spec` дополнительно проверяет импорт `tksheet`
(сборка без него даёт exe с `ModuleNotFoundError: tksheet` на старте):
```toml
[tool.pixi.feature.tk-gui.tasks.build-tcm-gui]
cmd = "python oceano/tcm/scripts/build/build_tcm_gui.py"
env = { BUILD_MODE = "manual" }
depends-on = ["browser-runtime"]
```

Перед сборкой выполняется зависимость **`browser-runtime`** — `npm ci` по
зафиксированному `browser/package-lock.json` и копирование минимального набора
файлов браузера документации в `_build/browser-runtime/` (см. `browser/vendor.mjs`).
Явное обновление зависимостей runtime — `pixi run -e bin-optim-tcm browser-runtime-update`
(поднимает до `@latest` и пересобирает `_build/browser-runtime`).

Или напрямую:
```bash
pixi run -e bin-optim-tcm python oceano/tcm/scripts/build/build_tcm_gui.py
```

Каталоги вывода — как в §2.2: по умолчанию внутри `oceano/tcm/`, `--build-root`
переносит их (суффикс окружения добавляется автоматически):
```bash
pixi run -e bin-optim-tcm build-tcm-gui --build-root B:\Temp
# → B:\Temp\tcm_gui-env=bin-optim-tcm\dist\tcm_gui\tcm_gui.exe
```

Скрипт-обёртка (`scripts/build/build_tcm_gui.py`):
1. Записывает/читает `version_meta.json` с GUI-метаданными:
   - `product`: `"tcm_gui"`
   - `description`: `"AB SIO RAS' TCM inclinometer data processor GUI..."`
   - `internal_name`: `tcm_gui.exe`
   - `original_filename`: `tcm_gui\__main__.py`
2. Запускает `generate_version_info` — создаёт `version_info.txt`
3. Вызывает PyInstaller с `tcm_gui.spec`

Результат: `dist/tcm_gui/tcm_gui.exe` + сопутствующие файлы.

### 3.3. Структура spec-файла

`scripts/build/tcm_gui.spec` — конфигурация PyInstaller для GUI.
Ключевые отличия от `tcm_proc.spec`:

| Параметр            | `tcm_proc`                 | `tcm_gui`                          |
| ------------------- | ----------------------------- | ---------------------------------- |
| Точка входа         | `scripts/tcm_proc.py`          | `src/tcm_gui/__main__.py`          |
| `console`           | `True`                        | **`False`** (оконное приложение)   |
| tkinter             | Исключён                      | **Включён**                        |
| `h5py`              | Исключён                      | **Включён**                        |
| `scipy`, `matplotlib` | Исключён                    | **Включён**                        |
| `tksheet`           | —                             | **Включён** (hidden import)        |
| `rthook_noh5_bins`  | Включён                       | **Не используется**                |
| Данные `src/tcm_gui`| —                             | **Включён**                        |
| Бинарные фильтры    | `.h5`, `.hdf5` исключены      | `.h5`, `.hdf5` **не** исключены   |
| Pure-модули         | `tcm.*`, `utils.*`, `veusz_helpers.*`, `meta_finder.*` исключены | `tcm.*`, `utils.*`, `veusz_helpers.*`, `meta_finder.*` исключены |

Общая логика (фильтрация бинарников, сборка данных) вынесена в
`scripts/build/spec_common.py` и используется обоими spec-файлами.

Ресурсы браузера документации попадают в дистрибутив так: first-party страница
(`index.html`/`viewer.js`/`viewer.css`) лежит внутри пакета —
`src/tcm_gui/browser/web`, поэтому упаковывается вместе с `src/tcm_gui`;
сгенерированный runtime добавляется отдельным datas-элементом
`(_build/browser-runtime → "_build/browser-runtime")`. Подкаталоги `todo/`
документации исключаются из сборки (`spec_common.should_keep_data`).

Зеркальное дерево (§2.3) делает браузер документации консистентным без
конвертаций путей: `DOC_DIR` вычисляется одной формулой для dev и заморозки
(`PROJECT_ROOT.parents[1]/docs` → `oceano/tcm/docs`), `readme*.md` лежат рядом
с docs (`DOC_DIR.parent` → `oceano/tcm/`), поэтому относительные ссылки
документации `../../…` (проект: `src/…`, `scripts/…`) и `../../../…` (`oceano/…`,
в т.ч. docs meta_finder) разрешаются одинаково в обоих режимах.

### 3.4. Runtime hook (`rthook_hydra_pkg.py`)

GUI использует тот же хук регистрации Hydra-плагинов, что и CLI
(см. § 2.4). Хук `rthook_noh5_bins.py` **не используется** —
GUI показывает полные умолчания конфигурации (`[0, 2, 600, 3600, 7200]`),
пользователь редактирует их в интерактивном режиме.

### 3.5. Windows file properties

Оба дистрибутива (`tcm_proc.exe` и `tcm_gui.exe`) имеют
Windows file properties (версия, описание, копирайт), генерируемые
из `scripts/build/version_info.template` через `generate_version_info.py`.
Параметризованные поля шаблона:

| Поле                 | CLI значение                          | GUI значение                          |
| -------------------- | ------------------------------------- | ------------------------------------- |
| `FileDescription`    | `AB SIO RAS' TCM raw data processing CLI...` | `AB SIO RAS' TCM inclinometer data processor GUI...` |
| `InternalName`       | `tcm_proc.exe`                     | `tcm_gui.exe`                         |
| `OriginalFilename`   | `tcm\scripts\tcm_proc.py`              | `tcm_gui\__main__.py`                 |
