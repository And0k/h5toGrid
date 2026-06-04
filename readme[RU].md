### Установка

pixi

установите менеджер окружений pixi и запустите pixi run pytnon ...

В проекте используется pixi с конфигурацией в pyproject.toml: как видно из него пакет требует наличия Python c библиотеками dask, pandas, numpy и других - готовьте место на диске > 4Гб

conda
Для установки также можно (раньше рекомендовал) установить менеджер пакетов и окружений [miniconda](https://conda.io/miniconda.html) или microconda.

С conda нужные пакеты при наличии интернета устанавливаются [командой](https://docs.conda.io/projects/conda/en/latest/commands/create.html):
```cmd
conda env create --force --file py3.10x64h5togrid.yml
```
из директории с [py3.10x64h5togrid.yml](py3.10x64h5togrid.yml) или пишите полный путь к нему.

При этом создается окружение `py3.10x64h5togrid`, в котором может запускаться пакет программ. Папка с установленными пакетами по умолчанию будет располагаться в c:\Users\ _имя пользователя_\conda\envs\.


### Дополнительные настройки

Для работы с rar-архивами путь к unrar.exe должен содержатся в системной переменной `PATH` (или добавьте его в код пакета определив `rarfile.UNRAR_TOOL`)

Для возможности расчета магнитного склонения используется библиотека [wmm2020](https://github.com/space-physics/wmm2020), для работы которой в Windows необходима установка и содержание в переменной `PATH` пути к исполняемым файлам [cmake](https://cmake.org/) и компилятору С++, но не Microsoft Visual C++, например, [mingw-w64](https://sourceforge.net/projects/mingw-w64)