### Установка

Пакет требует наличия Python c библиотеками dask, pandas, numpy и других. Для установки рекомендую установить менеджер пакетов и окружений [miniconda](https://conda.io/miniconda.html). Нужные пакеты при наличии интернета устанавливаются [командой](https://docs.conda.io/projects/conda/en/latest/commands/create.html):
```cmd
conda env create --force --file py3.10x64h5togrid.yml
```
из директории с [py3.10x64h5togrid.yml](py3.10x64h5togrid.yml) или пишите полный путь к нему.

При этом создается окружение `py3.10x64h5togrid`, в котором может запускаться пакет программ. В результате размер папки с установленными пакетами превышает 4Гб. По умолчанию она располагается в c:\Users\ _имя пользователя_\conda\envs\.

Для работы с rar-архивами путь к unrar.exe должен содержатся в системной переменной `PATH` (или добавьте его в код пакета определив `rarfile.UNRAR_TOOL`)

Для возможности расчета магнитного склонения используется библиотека [wmm2020](https://github.com/space-physics/wmm2020), для работы которой в Windows необходима установка и содержание в переменной `PATH` пути к исполняемым файлам [cmake](https://cmake.org/) и компилятору С++, но не Microsoft Visual C++, например, [mingw-w64](https://sourceforge.net/projects/mingw-w64)