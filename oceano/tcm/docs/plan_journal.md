# Feature: журнал состояния обработки (stage journal)

## Цель

Пайплайн и GUI говорят на одном языке состояния: границы этапов и подуровней пишутся **в сам лог** марками `[## …]` / `[### …]` силами пайплайна, без GUI и без сайдкар-файлов. Из этого следуют три свойства:

1. **Консольный прогон самодокументируем** — журнал состояний возникает всегда, GUI опционален.
2. **GUI восстанавливает любой прогон постфактум** — один проход по логу даёт полную картину: какие пробы/этапы завершены, что прервано, где были ошибки.
3. **Живая обработка видна как структура**, а не как стена текста: дерево этапов, вкладки-прогресс, точечная подгрузка сегментов.

## Проблемы, которые это снимает

| Было | Стало |
|---|---|
| Лог на 100k строк нечитаем — только скролл | Оглавление (дерево этапов) + клик = подгрузка сегмента с диска |
| Консольный прогон не оставляет структурированного следа | Марки пишутся сеттерами состояния; GUI не нужен для записи |
| N конфиг × этапов × чанков/бинов — каша | Дерево на пробу; вкладки = прогресс по конфигам |
| «Где упало и почему» — поиск по тексту | Раскраска по max level (ERROR красит и подуровень, и этап); клик по узлу → его строки |
| Статус этапа виден только после завершения | Граница этапа логируется в момент входа (с деталями: файл, чанк) |

## Пользовательские сценарии

**1. Консоль → GUI (главный).** Прогон запущен ночью из консоли. Утром открыт GUI → деревья всех проб восстановлены из лога (этапы ✔, последний — «прервано», если нет строки `Done`), в журнале — хвост последнего run, заголовок с датой. Никаких действий — «как будто только что закончил».

**2. Live-обработка.** Нажат Run: вкладки наливаются прогрессом (каждая — своя конфигурация, вместе — общая полоса), активная вкладка переключается на обрабатываемую конфигурацию, дерево раскрывает текущий этап, строки подуровней обновляются на месте: `read → chunk 2/3`, `binning → bin600s 3/5`.

**3. Расследование.** Клик по этапу → из файла читаются **только его строки**; клик по подуровню → его записи по всем чанкам/бинам. WARNING/ERROR узлы окрашены, эскалация поднимается до родителя. Esc / «live» — возврат к хвосту.

**4. Чистый экран.** Журнал свёрнут в строку-тикер до старта обработки; debug-строки скрыты по умолчанию (чекбокс включает — строки не удаляются, только прячутся).

## Состав

| Часть | Что даёт пользователю |
|---|---|
| Марки `[## …]`/`[### …]` в логе | Журнал = единственный источник состояния; работает без GUI |
| Дерево этапов на вкладку | Оглавление пробы: pending серый, текущий раскрыт, ошибки красные |
| Полоса вкладок | Кнопки переключения + прогресс каждой конфигурации без лишнего виджета |
| LogDock | Свёрнутый журнал с тикером, debug-тумблер, сегменты по клику |
| Restore | GUI после консоли показывает состояние без повторного запуска |

## Принципы userability

- **Слоистость информации**: вкладка (доля) → этап (статус) → подуровень (чанк/бин) → строки лога (детали). Каждый слой — один клик.
- **Одно состояние — один вызов**: `set_stage(1, "load", "Loading %s", name)` даёт лог-запись, контекст, марку и дерево; вызывающий не собирает строки вручную.
- **Управление пользователя уважается**: раскрытые вручную узлы программа не закрывает; закрытые пользователем — снова её.
- **Ничего лишнего на экране по умолчанию**: debug спрятан, журнал свёрнут, дети дерева — только живые; история не теряется, а раскрывается.
- **Деградация честная**: нет марок (старый лог) → GUI показывает хвост без семантики; прерванный прогон → последний этап помечен «прервано», не «завершено».


---


# Handoff: журнал состояния (##-марки) + stage GUI

## 1. Состояние работ

| Слой | Статус |
|---|---|
| `tcm/stage_ctx.py` (контекст + марки + `tick`) | **реализован** |
| `tcm/journal.py` (Reader) | черновик финализирован ниже, **не создан** |
| `tests/test_journal.py` | готов ниже, **не создан** |
| `tcm_gui/log_bridge.py` — 2 правки | **не внесены** (§5) |
| Пайплайн: точки вызова сеттеров | **не внесены** (§6) |
| GUI: `_stage_tree.py`, `_log_dock.py`, `_tab_strip.py`, проводка `app.py`, дополнения `const.py` | черновики **не созданы** (§7) |

Цель: прогон из консоли оставляет в логе границы состояний; GUI, открытый позже, восстанавливает полную картину (дерево этапов, сегменты лога по клику). GUI опционален — запись не зависит от него.

## 2. Контракт (согласован — не ломать)

**Марки границ** — ставит `StageContextFilter`, потребляет взведённый сеттерами `_cv_fresh` ровно один раз на смену состояния:

```
[# probe i90 1/2]                          первая запись после set_probe (редка)
[## probe i90 1/2 stage 1 load]            первая запись после set_stage
[### probe i90 1/2 stage 1 load / read]    первая запись после set_sublevel
[probe i90 1/2 stage 3 proc]               WARNING+ внутри контекста, без #
```

- Сеттер с `details` эмитит граничную запись сам: `set_stage` — INFO, `set_sublevel` — DEBUG (`funcName` вызывающего через `stacklevel=2`, имя логгера `tcm.stage_ctx`). Без `details` марка садится на следующую естественную запись.
- Префикс — **контракт идентичности** для `journal.Reader`: `probe {id} [{pi}[.{ci}]/{np}[.{nc}]] [stage {n} {name}] [/ {sub}]`. Формат `_build_prefix` не менять без смены парсера.
- Атрибуты записи для читателей: `stage_prefix` (всегда), `stage_fresh` (0–3), `boundary_msg` (чистый текст граничной записи, до prepend).
- Подуровень (`###`) — под-этап (read/time_corr/merge, ядра proc); чанк/бин — не уровень, а статус строки подуровня (из `boundary_msg` либо из `progress_stage.desc`).
- Этап видим, если породил ≥1 запись; `combine` (sn=0) идёт без `stage N` — Reader её не сегментирует (осознанно).
- Исход прогона — best-effort из терминальной строки `Done — {n_ok} probes: {pcids} ok | {n_skipped} skipped ({pcids}) | {n_failed} failed ({pcids})`.

**Механика очереди (учесть обязательно):** `QueueHandler` дедуплицирует по `(funcName, msg)` и морозит текст при emit. Чтобы границы с одинаковым текстом из одной функции не схлопывались, `[##/### prefix]` должен попасть в текст **до** freeze → `StageContextFilter` обязан стоять и на QueueHandler (§5), а повторный prepend исключён guard'ом `_stage_ctx_done`.

## 3. `tcm/journal.py` — создан

## 4. `tests/test_journal.py` — создать

Обе записи сеттеров (`tcm.stage_ctx`) и свободные записи идут через один FileHandler на логгере `tcm` — как в продакшене фильтр стоит на хендлере:

```python
"""Reader: segments from [## …] / [### …] boundaries; WARNING marks don't break them."""
import logging
from pathlib import Path

from tcm import journal, stage_ctx

_FMT = logging.Formatter("%(asctime)s|%(name)s|%(levelname)s|%(message)s", "%H:%M:%S")
_LOG = logging.getLogger("tcm")            # parent of tcm.stage_ctx — catches setters
_l = logging.getLogger("tcm.test.journal")


def _run_log(tmp: Path, fn) -> journal.Reader:
    """Emit fn()'s records through a hydra-formatted file → Reader over it."""
    log_dir = tmp / "cfg_proc" / "log" / "2026-07-29_14-15-03"
    log_dir.mkdir(parents=True)
    fh = logging.FileHandler(log_dir / "tcm_proc.log", encoding="utf-8")
    fh.setFormatter(_FMT)
    fh.addFilter(stage_ctx.StageContextFilter())
    _LOG.addHandler(fh)
    _LOG.setLevel(logging.DEBUG)
    try:
        fn()
    finally:
        fh.close()
        _LOG.removeHandler(fh)
        stage_ctx.clear()
    return journal.Reader.open(tmp)


def test_segments(tmp_path):
    def fn():
        stage_ctx.set_probe("i90", 1, 1, 2, 2)
        stage_ctx.set_stage(1, "load", "Loading data for i90…")
        _l.debug("Skipping i67 — covered")        # без марки
        _l.warning("Sparse region detected")      # [probe i90 1/2 stage 1 load]
        stage_ctx.set_stage(3, "proc", "Processing i90")
        _l.info("Done — 1 probes: i90 ok | 0 skipped () | 0 failed ()")

    rd = _run_log(tmp_path, fn)
    p = rd.probes["i90"]
    assert p.idx == "1/2" and p.last == 3
    assert set(p.stages) == {1, 3}
    assert p.stages[1].name == "load"
    assert p.stages[1].b == p.stages[3].a         # передача диапазона на границе
    assert rd.ended and p.ok is True
    seg1 = list(rd.segment("i90", 1))
    assert any("Skipping i67" in l for l in seg1)
    assert any("Sparse region" in l for l in seg1)   # WARNING внутри сегмента
    assert not any("Processing i90" in l for l in seg1)


def test_sublevels(tmp_path):
    """set_stage без деталей сразу перед set_sublevel — этап открывается из ###-префикса."""
    def fn():
        stage_ctx.set_probe("i01")
        stage_ctx.set_stage(1, "load")
        for k in (1, 2):
            stage_ctx.set_sublevel("read", "chunk %d/2", k)
            stage_ctx.set_sublevel("merge")
            _l.debug("merged")

    rd = _run_log(tmp_path, fn)
    p = rd.probes["i01"]
    assert 1 in p.stages                          # этап открыт из ###-границы
    assert [s.name for s in p.subs[1]] == ["read", "merge", "read", "merge"]
    assert len(list(rd.segment("i01", 1, "read"))) >= 2   # объединение вхождений


def test_interrupted(tmp_path):
    def fn():
        stage_ctx.set_probe("i90")
        stage_ctx.set_stage(1, "load", "Loading…")
        # убит — строки Done нет
    rd = _run_log(tmp_path, fn)
    assert not rd.ended and rd.probes["i90"].ok is None
```

## 5. `tcm_gui/log_bridge.py` — две согласованные правки

```python
def install(q: Queue, gate, level: int = logging.DEBUG) -> QueueHandler:
    """…существующий docstring + абзац ниже…

    StageContextFilter attached here so the freeze captures the already
    prepended [## / ### prefix] — consecutive dedup keys stay distinct
    for boundary records with identical text from one function.
    """
    from tcm import stage_ctx
    h = QueueHandler(q, gate)
    h.setLevel(level)
    h.addFilter(stage_ctx.StageContextFilter())
    logging.getLogger().addHandler(h)
    return h
```

И **обязательно в паре** — убрать собственную вставку префикса в `drain` (блок `if prefix and rec.levelno >= logging.WARNING: w.insert(...)`): при фильтре на QueueHandler текст уже заморожен вместе с `[prefix]`, иначе в GUI префикс удваивается.

## 6. Пайплайн — точки вызова (не внесены)

| Где | Вызов |
|---|---|
| `cli.process_loading_yaml`, цикл по YAML | `set_probe(pcid, probe_idx, cfg_idx, n_probes, n_cfgs, stem_idx=k, n_cfgs_total=n)` |
| `processing.run_processing`, начало пробы | `set_stage_plan(n_active)` — число активных этапов (load, coefs, proc + NC/TSV per bin при `use_h5`) |
| `processing.run_processing`, границы фаз | `set_stage(1, "load", "Loading %s", path.name)` и т.д. (фазы 1–4 по how_it_works §run_processing) |
| `_xr/dataset.py::load_raw` + `csv_load.load_from_csv_gen` + вход в `time_corr` | `set_sublevel("read", "chunk %d/%d", k, n)`, `set_sublevel("time_corr")`, `set_sublevel("merge")` |
| `_xr/physical.py::process` | `set_sublevel("filter_local")`, `set_sublevel("calc_velocity")`, `set_sublevel("calc_pressure")`, `set_sublevel("binning", "%d bins", len(bins))` |
| `processing._load_batch`, цикл файлов | `set_sublevel(stem)` на файл |

Прогресс чанков (нижняя полоса) — существующий канал: `GuiTqdm(load_from_csv_gen(...), total=n, desc="load")`, одна строка.

## 7. GUI — состав фазы

### `_stage_tree.py` создан

### `_log_dock.py` (спецификация)

Свёрнутая панель журнала вместо голого ScrolledText (§5 app): бар 28px (`const.scaled`) — кнопка `Journal ▴/▾`, метка run-даты (`Run #{Reader.latest}`), **тикер последней строки** (жив даже свёрнутым), чекбокс `debug`; ниже — ScrolledText (теги `TAG_COLORS` + `func` + `debug` с `elide=True` по умолчанию — переключатель_flip elide, строки не удаляются). Методы: `flip`, `expand_once` (однократно, при первом run), `render(rec)` — тело нынешнего `drain` по записи + тикер, `show_static(lines, note)` — сегмент из Reader (потолок ~5000 строк + пометка), `live()` — возврат к хвосту, `at_bottom()`, `see_end()`, `set_date(s)`. Ctrl+C → `copy_rich`.

### `_tab_strip.py` (спецификация)

ttk-вкладки не красятся поштучно и не растягиваются на ширину → `style.layout("Bare.TNotebook", [("Notebook.client", {"sticky": "nswe"})])` + свой `tk.Canvas` над Notebook: `rebuild(names)`, ячейки `w // n` (границы fill = границы вкладки → вместе читаются общей полосой), fill ∝ frac, текст центрирован, подчёркивание 3px выбранной, hover, клик → `nb.select`. Прогресс — арифметика существующего `progress_overall` (100 единиц на конфиг, порядок страниц == порядку обработки):

```
frac_k  = clamp(cur_o / 100 - k, 0, 1)      # k — индекс вкладки
running = min(int(cur_o / 100), n - 1)      # desc_o → подпись этой вкладки
```

Состояния: `frac ≤ 0` — серая, `< 1` — идёт, `≥ 1` — ✔; провал — из `failed` в `_on_run_done` (красная точка). Грязная правка — `set_dirty` вместо переименования `nb.tab`. Палитра — `const.strip_palette` (производные от темы), пиксельная геометрия — `const.scaled`, анимация: лерп fill + блик на фронте у running (свой тик ~60 мс, цели приходят из `_poll_progress`).

### `app.py` — дельты проводки

- `_build`: strip над `nb` (Bare), §4 без `_prog_all` (полоса на вкладках), §5 → `LogDock`.
- `_add_page`: `ttk.PanedWindow(orient="horizontal")` — `StageTree(pcid=format.stem_to_pcid(stem))` + фрейм ConfigSheet; `self._trees[stem] = tree`. Несколько stem одного pcid возможны (дубликаты) — деревья получают одни записи, допустимо.
- `_poll_logs`: `drain` расщепить на `take(q) -> list[rec]` + `render(w, rec)`; цикл: `dock.render(rec)` + `tree.observe(rec)` всем деревьям; `see_end` при `at_bottom`.
- `_poll_progress`: нижняя полоса как сейчас; strip по формулам выше; `set_live_progress(desc, cur, tot)` дереву активной вкладки при `tot > 0`.
- `_poll_dirty_tabs`: `strip.set_dirty(stem, cs.is_dirty)` (переименование `nb.tab` убрать).
- `_on_run`: `dock.expand_once()`, `self._follow = True`, `tree.reset()` всем.
- Follow: смена running-вкладки при `follow` — программный `nb.select` под guard'ом; ручное `<<NotebookTabChanged>>` (без guard) → `follow = False`.
- `_on_scan_ok`: после перестройки страниц — `strip.rebuild(stems)` и restore: `data_dir = paths.find_dir_raw_absolute(Path(self._path_field.get()))`, `rd = Reader.open(data_dir)`; при `rd`: `dock.set_date(f"Run #{rd.latest}")`, каждому stem `tree.apply_snap(rd.probes[pcid], rd.ended)`, состояния strip из снапшота, хвост лога в dock (`rd.segment` последнего этапа или tail файла).
- `_on_tree_pick(pcid, num, sub)`: кэшированный Reader → `dock.show_static(rd.segment(pcid, num, sub))`; для running-этапа — `dock.live()`.
- `_on_run_done`: strip в done/failed по спискам результата.

### `const.py` — дополнения

`mix_hex(a, b, t)`, `scaled(px) = round(px * UI_SCALE)`, `strip_palette(widget)` — токены track/run/done/edge/sel/dim, производные от `resolved_frame_bg` + `BLUE_FG`/`DEFAULT_FG` (никаких новых hex в виджетах).

## 8. Факты среды (проверено документацией)

- Лог: `{data_dir}/cfg_proc/log/{timestamp}/tcm_proc.log`; файл — `asctime|name|levelname|message` (simple), консоль — colorlog `funcName|message`; консоль INFO, файл DEBUG. `data_dir` = результат `paths.find_dir_raw_absolute(path_in)`.
- Hydra `dictConfig` каждый воркер-таск заменяет root-хендлеры; `worker._wrap.wrapped` пере-прикрепляет QueueHandler + `reset_dedup()`.
- Этапы: 1 load, 2 coefs, 3 proc, 4+ NC/TSV per bin, combine (sn=0, не per-probe).
- stem → pcid: `format.stem_to_pcid`; страница keyed by stem, записи — by pcid.
- `App.POLL = 300 мс`; `ProgressState` — `progress_overall`/`progress_stage` + PauseGate в обоих каналах.

## 9. Правила кодовой базы

- Логгеры: `_l` — printf, `_lf` — format-string style.
- Docstring: зачем + контракт + как взаимодействовать; без самооценки.
- Одно понятие — одно имя (не вводить синонимов: WINDOW, а не DEPTH/K; маркер, а не разделитель).
- Цвет — только `const.py`; EAFP > проверок; walrus/itertools/распаковка > ручных циклов; виджеты регистрировать в `widget_meta`.
- Общение — русский, плотно, без повторов уже согласованного.

## 10. Открытые пункты (разобрать до реализации)

1. **`tick()` vs `set_stage()`**: оба устанавливают stage-контекст, причём `tick` кладёт в `stage_num` свой счётчик. В `run_processing` нужна одна конвенция — рекомендую: контекст+граница через `set_stage(num, name, details)`, прогресс либо перенести в `set_stage` (по `stage_num`), либо строго парировать `set_stage`+`tick()`; смешать без соглашения нельзя.
2. Проверить прикрепление `StageContextFilter` в `cfg_proc/hydra/job_logging/colorlog.yaml` — фильтр должен стоять на console/file хендлерах (иначе `##` не попадёт в файл).
3. Правки §5 вносить только парой (фильтр на QueueHandler + убрать вставку префикса в `drain`).
4. Убедиться, что порядок страниц == порядку обработки (нужно арифметике frac strip'а).

Чек-лист после реализации: тесты §4 зелёные → консольный прогон без GUI → `Reader.open` видит этапы/подуровни/исход → GUI после консольного прогона восстанавливает деревья и сегменты → WARNING в файле и GUI с одним префиксом → дедуп не ест границы чанков.
