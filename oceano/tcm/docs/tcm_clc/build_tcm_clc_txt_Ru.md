# Сборка минимального дистрибутива `tcm_clc_txt`

**tcm_clc_txt** — самодостаточный exe-файл для обработки данных инклинометров
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

Используется выделенное pixi-окружения `noh5-tcm` или `bin-optim-tcm`, в которых все зависимости
BLAS/LAPACK собраны с OpenBLAS, а Intel MKL отсутствует.

Ключевые пакеты окружения (из `pyproject.toml`):

| Пакет          | Назначение                           |
| -------------- | ------------------------------------ |
| `python 3.12`  | Фиксировано `<3.13` (совместимость с hydra) |
| `numpy`        | Численные расчёты                    |
| `pandas`       | Табличные данные                     |
| `hydra-core`   | Конфигурация CLI                     |
| `hydra-colorlog` | Цветной лог                          |
| `libblas`/`libcblas`/`liblapack` | OpenBLAS-вариа    |

| `dask-core`    | Параллельные вычисления              |

| `numba`        | JIT-оптимизации                      |

Отдельно указаны build-строки для BLAS-пакетов:

```toml
libblas   = { version = "*", build = "*openblas*" }
libcblas  = { version = "*", build = "*openblas*" }
liblapack = { version = "*", build = "*openblas*" }
```

Это гарантирует, что `libblas.dll`, `libcblas.dll`, `liblapack.dll` будут
обёртками над `openblas.dll`, а не переключателями на `mkl_rt.3.dll`.

`dask` с его `pyarrow` - Работа с pandas через dask.dataframe - исключены

### 2.2. Запуск сборки

Активировать окружение и запустить PyInstaller:
```bash
pixi run -e noh5-tcm pyinstaller --noconfirm scripts\build\tcm_clc_txt.spec
```

Или через batch-скрипт:

```cmd
scripts\build_tcm_clc_txt.bat
```

Результат: `dist/tcm_clc_txt/tcm_clc_txt.exe` + сопутствующие файлы.

### 2.3. Структура spec-файла

`scripts/build/tcm_clc_txt.spec` — конфигурация PyInstaller. Ключевые моменты:

-   **Точка входа:** `scripts/tcm_clc.py`
-   **Явно добавленные DLL:** `openblas.dll`, `libcblas.dll`, `libblas.dll`, `liblapack.dll`,
    а также системные: `libmpdec-4.dll`, `yaml.dll`, `sqlite3.dll`, `libzmq-mt-4_3_5.dll`,
    `tbb12.dll`, `tbbmalloc.dll`, и др.
-   **Фильтрация бинарников:** из собранного набора удаляются все
    `mkl_*`, `arrow_flight*`, `gandiva*`, `libiomp5md*` (т.е. MKL-специфичные
    и неиспользуемые библиотеки Arrow).
-   **Данные:** конфиги `tcm/cfg/`, `tcm/cfg/coef/`, файлы конфигурации dask
    (`dask.yaml`, `distributed.yaml` и схемы), METADATA пакетов pandas/numpy/pyarrow.
-   **Hydra conf:** автоматический сбор данных `hydra/conf/` и
    `hydra_plugins.hydra_colorlog` через `collect_data_files`, плюс явное
    добавление `__init__.py` для корректной работы `importlib.resources`.
-   **Runtime hooks:**
    - `rthook_hydra_pkg.py` — вручную регистрирует все core-плагины Hydra
      (ImportlibResourcesConfigSource, FileConfigSource, StructuredConfigSource,
      BasicLauncher, BasicSweeper) и цветовой плагин, т.к. `pkgutil.walk_packages`
      не работает в замороженном exe.
    - `rthook_noh5_bins.py` — переопределяет `out/base` в ConfigStore:
      `dt_bins=[0, 3600]`, `dt_bins_min_save_text=0` (см. § 2.5).

### 2.4. Runtime hook (`rthook_hydra_pkg.py`)

```python
from hydra.core.plugins import Plugins
p = Plugins.instance()

# Core ConfigSources — без них hydra не найдёт pkg:// и file:// схемы
from hydra._internal.core_plugins.importlib_resources_config_source import ImportlibResourcesConfigSource
from hydra._internal.core_plugins.file_config_source import FileConfigSource
from hydra._internal.core_plugins.structured_config_source import StructuredConfigSource
from hydra._internal.core_plugins.basic_launcher import BasicLauncher
from hydra._internal.core_plugins.basic_sweeper import BasicSweeper

for cls in [ImportlibResourcesConfigSource, FileConfigSource,
            StructuredConfigSource, BasicLauncher, BasicSweeper]:
    p.register(cls)

# Цветной лог
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
import tcm.config  # форсирует регистрацию ConfigStore с dev-умолчаниями
from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()
cs.store(
    group="out",
    name="base",
    node=tcm.config.ConfigOut_InclProc(
        dt_bins=[0, 3600],            # без усреднения + 1 час
        dt_bins_min_save_text=0,      # разрешить TSV для bin=0
    ),
    provider="noh5-rthook",
)
```

**Результат:** в собранном `tcm_clc_txt.exe` выходные текстовые файлы
формируются с двумя настройками усреднения:

| dt_bin | Описание | TSV-файл |
|--------|----------|----------|
| `0` | Без усреднения (raw → physical) | `{ts}@i_01.tsv` |
| `3600` | Усреднение 1 час | `{ts}bin3600s@i_01.tsv` |

Для сравнения, в dev-окружении (где доступны h5py/pytables) умолчания
остаются `[0, 2, 600, 3600, 7200]` с `dt_bins_min_save_text=1`.
