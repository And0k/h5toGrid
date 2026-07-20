# Описание журнала работы tcm_clc (noh5-режим)

Разбор INFO/WARNING сообщений из лога обработки. Для понимания сокращений см. [README.md](README.md), [config_reference.md](config_reference.md), [how_it_works.md](how_it_works.md). Отладочные (DEBUG) сообщения здесь не рассматриваются — они в файле `cfg_proc/log/{timestamp}/processing.log`.

---

## Этап 1: Сканирование и генерация конфигов

```
14:49:48|run|Config generation: regenerating 1 stale configs
```

При запуске `processing.run()` обнаружены устаревшие конфиги — YAML-файлы в `cfg_proc/run/`, чьё поле `input.path` указывает на несуществующий файл. Генерация запущена для их обновления. См. `config_yaml.find_stale_cfgs()`.

```
14:49:48|search_csv_files|Files for 3 probes found (1 raw suppressed by corrected counterparts):
i64: @i_064.TXT
i67: I_067.TXT
i90: INKL_090.TXT
```

`csv_load.search_csv_files()` нашёл файлы для трёх датчиков. «1 raw suppressed» — для одного датчика существовал файл без префикса `@`, но обнаружен его `@`-вариант (уже обработанный), поэтому «сырой» скрыт. Список показывает, какие файлы будут использованы. Нормализация имени в pcid описана в [README.md §Probe Identity](README.md#probe-identity-pcid).

```
14:49:48|gen_metadata|Discovered 3 file groups from {path}
```

`config_yaml.gen_metadata()` сгруппировал найденные файлы по identity `(model, number)`. Группа = один датчик, может содержать несколько файлов-сегментов (если запись прерывалась и возобновлялась).

```
14:49:48|correct_raw_files|No raw files to correct for i64 (all already @-prefixed)
```

При генерации конфигов (edge-row режим) проверяется коррекция CSV-файлов. Все файлы уже имеют префикс `@` — этап пропущен. См. [how_it_works.md §CSV correction](how_it_works.md#csv-correction).

---

## Этап 1.1: Загрузка коэффициентов (генерация конфигов)

```
14:49:48|loaded_tcm|Time 25-12-04 18:17 – 05-02 16:51 (2 values) converted
14:49:48|prep_cfg_for_probe|Coefs for i64: paths=[calibration.h5, yaml_export], date=2025-06-23 12:50:24
```

Для каждого датчика при генерации конфига читаются первая и последняя строки CSV для получения диапазона времени (2 значения → `time_ranges`). Затем загружаются калибровочные коэффициенты. `paths` — цепочка источников (см. [config_reference.md §Coefficient loading priority](config_reference.md#coefficient-loading-priority)). `date` — дата калибровки из найденного набора коэффициентов.

Аналогично для i67 и i90 — каждый со своим диапазоном и датой.

---

## Этап 1.2: Коррекция CSV-файлов

```
14:49:48|rep_in_file|preliminary correcting csv file I_067.TXT by removing irregular rows, writing to @i_67.TXT
14:50:04|correct_txt|2 bad lines deleted
14:50:04|correct_raw_files|Corrected 1 file for i67
```

Файл `I_067.TXT` — «сырой» (без `@`). Запущена коррекция: чтение построчно, фильтрация строк, не проходящих по regex столбцов (см. `text_line_regex` в [config_reference.md §Text type → column layout](config_reference.md#text-type--column-layout)). Удалено 2 бракованные строки, результат записан как `@i_67.TXT`.

```
14:50:04|rep_in_file|preliminary correcting csv file INKL_090.TXT by removing irregular rows, writing to @i_90.TXT
14:50:19|correct_raw_files|Corrected 1 file for i90
```

Аналогично для `INKL_090.TXT` → `@i_90.TXT`. Оба файла `i64` уже были `@`-префиксированы на предыдущих запусках.

---

## Этап 1.3: Синхронизация с метаданными устройств

```
14:50:19|sync_yamls_devmeta_and_hydra|Loading date range from "info_devices" metadata file
```

`config_yaml.sync_yamls_devmeta_and_hydra()` загружает файл `info_devices.yaml` (или `.json`) из родительской директории (папка рейса). Содержит записанные диапазоны времени работы устройств (поле `"r"` в терминологии `_meta_array_to_dict()`). См. `tcm/metadata.py`.

```
14:50:19|sync_yamls_devmeta_and_hydra|  i90: info_devices [2025-12-05T04:30:20, 2026-03-12T22:11:03]
14:50:19|sync_yamls_devmeta_and_hydra|    already configured but broader than metadata: 251204_1800@i_090.yaml [2025-12-04T18:00:19, 2026-03-12T22:11:03]
```

Для i90: файл `info_devices` содержит диапазон `[2025-12-05T04:30:20, 2026-03-12T22:11:03]`. Но YAML-конфиг `251204_1800@i_090.yaml` уже имеет `time_ranges` `[2025-12-04T18:00:19, ...]` — начало **раньше** метаданных на ~10 часов. Конфиг не изменён (его диапазон шире). Предупреждение — повод проверить, корректен ли диапазон. Аналогичные предупреждения для i64 и i67 — у всех конфиги уже настроены с более широким диапазоном, чем в метаданных.

```
14:50:19|run|Orphan configs (stale config(s): i67: 251204_1823@i_067.yaml) — run with specific input.ids to ignore
```

Найден YAML-конфиг `251204_1823@i_067.yaml` для `i67`, но его `input.path` ссылается на несуществующий файл. Это «orphan config» (устаревший). Предупреждение указывает **имя файла** — его нужно удалить вручную или запустить с явным указанием `input.ids`. При запуске `input.ids=[i90, i67]` этот orphan не блокирует обработку — i67 будет обработан через другой конфиг (см. ниже).

---

## Этап 2: Обработка данных

```
14:50:19|process_loading_yaml|Running 2 probes (of 3 available), 3 stems
14:50:19|process_loading_yaml|[1/3] probe i90 (from "251204_1800@i_90.yaml")
```

`cli.process_loading_yaml()` запускает обработку. 2 датчика (i90, i67) из 3 доступных (i64 не в `input.ids`). 3 YAML-стема — для i67 два конфига (`@i_067.yaml` и `@i_67.yaml`), для i90 один. Обработка идёт по порядку stem'ов.

```
TCM processing started.
```

Внутри `cli.main_init()` — баннер запуска. Печатается в начале обработки каждого probe.

---

### i90: Коррекция и загрузка

```
14:50:18|correct_raw_files|No raw files to correct for i90 (all already @-prefixed)
14:50:51|loaded_tcm|Time 25-12-04 18:00 – 03-12 22:11 (4100020 values) converted
```

Файл `@i_90.TXT` уже обработан (префикс `@`). Загружено 4 100 020 строк данных.

```
14:50:51|_estimate_freq_np|freq=5.01914Hz found from 835533 floored 1s-runs: 5smp/s at 91.3% + 6smp/s at 1.8%
```

Функция `_estimate_freq_np()` в `utils_time_corr.py` определила частоту дискретизации. Алгоритм (режим B — floored 1s): группирует строки с одинаковым целым значением секунды, считает длину каждой группы (количество отсчётов в секунду). Доминирует 5 отсчётов/с (91.3%), 6 отсчётов/с — редкие (1.8%). Средневзвешенная: `0.913×5 + 0.018×6 ≈ 5.019`. См. `_estimate_freq_np` docstring в `utils_time_corr.py`.

---

### i90: Временна коррекция

```
14:50:52|_remove_outliers_combined|bilateral spikes: 6038 at ['2025-12-04T20:50:40', ...]
```

Этап удаления выбросов `_remove_outliers_combined()`. Двусторонний тест (bilateral): точка — выброс, если её отклонение от ожидаемого положения превышает порог `corr_time_outlier_threshold_s` (по умолчанию 0.6 с) **и** слева, **и** справа. Найдено 6038 выбросов. Времена — первые 10 моментов начала выбросов.

```
14:50:52|_correct_time|Removed 0.1% = (0 overlong + 6038 spike + 0 backward)/4100020
```

Итог удаления: 0.1% данных. Три категории:
- **overlong** (0) — слишком длинные серии одинаковых секунд (drift); обрезаются
- **spike** (6038) — двусторонние выбросы; удалены
- **backward** (0) — резкие обратные скачки (HWM); не обнаружены

```
14:50:55|_snap_to_grid|6038 interpolated outlier position(s) non-monotone after snap (masked)
```

После удаления выбросов оставшиеся точки интерполируются и «привязываются к сетке» (`_snap_to_grid`): каждому сегменту назначается равномерная сетка `t_k = origin + k·dt_step`. 6038 интерполированных позиций после привязки нарушают монотонность (получили одинаковое или «заднее» время из-за ограничения точности float64 CF-кодирования — см. [how_it_works.md §Float64-seconds precision limit](how_it_works.md#float64-seconds-precision-limit)). Эти точки замаскированы (исключены).

```
14:50:55|_correct_time|time correction: 4093982/4100020 monotone (in-range=4100020); 0.1% removed (spikes=6038, backward=0); correction [-0.099, 1.092]s; 386621 pts > alarm 0.80s
```

Итоговая сводка коррекции времени (WARNING — превышен порог alarm):
- **4093982/4100020** — монотонных точек из общего числа
- **in-range=4100020** — все точки в пределах `time_ranges` (ничего не отброшено окном)
- **0.1% removed** — итого удалено
- **correction [-0.099, 1.092]s** — диапазон смещения точек при привязке к сетке (от −0.1 с до +1.1 с)
- **386621 pts > alarm 0.80s** — количество точек, где сдвиг превышает порог тревоги. Порог = `(round(freq) − 1) × dt_step + ε` = `(5−1)×0.2 + ε ≈ 0.8 с`. Большое число (9.4%) типично для данных с целочисленным разрешением секунд при 5 Гц — каждая N-я точка в секунде сдвигается на ~1 с. Это **не** ошибка, а нормальное поведение snap-to-grid.

```
14:50:55|save_time_corr_diagnostics|diagnostics {path}/dt@i_90.npz saved (406796 events): HOLE=14137, ALARM=386621, NOT_MONO=6038, SPIKE=6038
```

Сохранён NPZ-файл с диагностикой. Содержит индексы, битовые маски действий и сдвиги для всех «событийных» позиций. Битовые флаги (см. `DiagBit` в `utils_time_corr.py`):
- **HOLE** (14137) — границы пропусков данных > `dt_interp_between`
- **ALARM** (386621) — точки с коррекцией > порога 0.80 с
- **NOT_MONO** (6038) — немонотонные после snap (те же, что и SPIKE — интерполированные позиции)
- **SPIKE** (6038) — удалённые двусторонние выбросы

---

### i90: Отбрасывание строк вне диапазона

```
14:50:59|csv_process|Values (6038) outside range: 2025-12-04 18:41:24, 2025-12-04 18:51:38, ... (first 10 distinct)
```

После коррекции времени часть строк попала за пределы `time_ranges` (или была отброшена фильтрами `input.min`/`input.max`). Показаны первые 10 уникальных моментов. Функция `csv_process()` в `csv_load.py`.

---

### i90: Загрузка и сохранение

```
14:51:08|load_raw|Loaded .txt: @i_90.TXT (8 vars, 4093982 rows)
```

Загрузка завершена: 8 переменных (`Ax, Ay, Az, Mx, My, Mz, Battery, Temp`), 4 093 982 строки.

```
14:51:18|_process_and_persist|Saved TSV i90 to 251204_1800bin2s@i_90.tsv
14:51:18|_process_and_persist|Saved TSV i90 to 251204_1800bin600s@i_90.tsv
14:51:18|_process_and_persist|Saved TSV i90 to 251204_1800bin3600s@i_90.tsv
14:51:18|_process_and_persist|Saved TSV i90 to 251204_1800bin7200s@i_90.tsv
```

`_process_and_persist()` — физическая обработка (калибровка → скорость → направление → усреднение) и экспорт TSV. Четыре файла с разным bin: 2 с, 600 с (10 мин), 3600 с (1 ч), 7200 с (2 ч). Имя: `{timestamp}bin{N}s@{pcid}.tsv`. См. [README.md §What You Get](README.md#what-you-get).

---

### i67: Первый конфиг — ошибка

```
14:51:18|process_loading_yaml|[2/3] probe i67 (from "251204_1823@i_067.yaml")
TCM processing started.
14:51:18|correct_raw_files|No raw files to correct for i64 (all already @-prefixed)
14:51:18|process_loading_yaml|[2/3] i67: source file missing — likely stale config ({path}\@i_067.TXT). Delete old YAML or run with correct input.ids
```

YAML `251204_1823@i_067.yaml` ссылается на `@i_067.TXT`, но файл не найден. Обработка пропущена. Это **тот самый** orphan config из предупреждения выше. Второй конфиг `251204_1823@i_67.yaml` (с другим именем файла, но тем же pcid i67) будет обработан следующим.

---

### i67: Второй конфиг — обработка

```
14:51:18|process_loading_yaml|[3/3] probe i67 (from "251204_1823@i_67.yaml")
TCM processing started.
14:51:57|loaded_tcm|Time 25-12-04 18:23 – 04-03 11:50 (5000000 values) converted
14:51:58|_estimate_freq_np|freq=5.01788Hz found from 1019496 floored 1s-runs: 5smp/s at 91.3% + 6smp/s at 1.7%
```

Загрузка `@i_67.TXT` — 5 000 000 строк. Частота 5.018 Гц (аналогично i90).

```
14:51:58|_remove_outliers_combined|bilateral spikes: 7300 at [...]
14:51:59|_correct_time|Removed 0.1% = (0 overlong + 7300 spike + 0 backward)/5000000
14:52:02|_snap_to_grid|7300 interpolated outlier position(s) non-monotone after snap (masked)
14:52:03|_correct_time|time correction: 4992700/5000000 monotone (in-range=5000000); 0.1% removed (spikes=7300, backward=0); correction [-0.096, 1.096]s; 504669 pts > alarm 0.80s
14:52:03|save_time_corr_diagnostics|diagnostics .../dt@i_67.npz saved (529210 events): HOLE=17241, ALARM=504669, NOT_MONO=7300, SPIKE=7300
14:52:09|csv_process|Values (7300) outside range: ... (first 10 distinct)
```

Первый сегмент i67: аналогично i90. 7300 выбросов, 10.1% alarm-точек.

---

### i67: Второй сегмент (данные после перерыва)

```
14:52:22|loaded_tcm|Time 26-04-03 11:50 – 04-20 23:41 (730690 values) converted
14:52:22|_estimate_freq_np|freq=5.02354Hz found from 148650 floored 1s-runs: 5smp/s at 91.4% + 6smp/s at 2.2%
```

Файл `@i_67.TXT` содержит два сегмента с разрывом (2025-12 → 2026-04). Второй сегмент: 730 690 строк.

```
14:52:22|_remove_outliers_combined|bilateral spikes: 1093 at [...]
14:52:22|_correct_time|Removed 0.1% = (0 overlong + 1093 spike + 0 backward)/730740
14:52:23|_snap_to_grid|1093 interpolated outlier position(s) non-monotone after snap (masked)
14:52:23|_correct_time|time correction: 729647/730740 monotone (in-range=730740); 0.1% removed (spikes=1093, backward=0); correction [-0.098, 1.091]s; 61206 pts > alarm 0.80s
14:52:23|save_time_corr_diagnostics|diagnostics .../dt@i_67.npz saved (64818 events): HOLE=2519, ALARM=61206, NOT_MONO=1093, SPIKE=1093
```

Второй сегмент i67: 1093 выброса, 8.4% alarm-точек. Малый объём (730K) — короче первого сегмента (5M).

```
14:52:23|csv_process|Values (1093) outside range: ... (first 10 distinct)
14:52:24|load_raw|Loaded .txt: @i_67.TXT (8 vars, 5722297 rows)
```

После объединения сегментов и фильтрации: 5 722 297 строк.

```
14:52:41|_process_and_persist|Saved TSV i67 to 251204_1823bin2s@i67.tsv
14:52:41|_process_and_persist|Saved TSV i67 to 251204_1820bin600s@i67.tsv
14:52:41|_process_and_persist|Saved TSV i67 to 251204_1800bin3600s@i67.tsv
14:52:41|_process_and_persist|Saved TSV i67 to 251204_1800bin7200s@i67.tsv
```

Экспорт i67: четыре TSV. Обратите внимание: `bin2s` использует метку `1823` (начало данных i67), а `bin600s` — `1820`, `bin3600s`/`bin7200s` — `1800`. Метка берётся от первого `datetime` в усреднённом результате.

---

## Этап 3: Итог

```
14:52:41|run|Done — 2 probes: i90, i67 ok
```

Оба запрошенных датчика обработаны успешно. i64 не обрабатывался (не в `input.ids`). i67 потерпел неудачу на первом конфиге (`@i_067.TXT` — файл отсутствует), но успешно обработан через второй конфиг (`@i_67.TXT`). Счётчик по **distinct pcid**: i67 = ok (хотя один stem failed, другой succeeded).

---

## Сводка по ключевым параметрам обработки

| Параметр | i90 | i67 (сегмент 1) | i67 (сегмент 2) |
|----------|-----|-----------------|-----------------|
| Строк | 4 100 020 | 5 000 000 | 730 690 |
| Частота | 5.019 Гц | 5.018 Гц | 5.024 Гц |
| Выбросы (spike) | 6038 (0.1%) | 7300 (0.1%) | 1093 (0.1%) |
| Alarm точки | 386 621 (9.4%) | 504 669 (10.1%) | 61 206 (8.4%) |
| Диапазон коррекции | [−0.099, 1.092] с | [−0.096, 1.096] с | [−0.098, 1.091] с |
| Выходные TSV | 4 файла (2/600/3600/7200 с) | 4 файла | — (объединены с сегментом 1) |
