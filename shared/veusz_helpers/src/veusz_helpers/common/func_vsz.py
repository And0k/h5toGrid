"""
Note: Annotations not used as not supported in Veusz 3.2
"""

import builtins
import contextlib
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray
from logging import info, warning, exception
import operator
from itertools import zip_longest
from typing import Any, Mapping, Optional, Callable
from datetime import datetime
from subprocess import check_output
import sys
from pathlib import Path
from winreg import HKEY_CURRENT_USER, OpenKey, QueryValueEx

with OpenKey(HKEY_CURRENT_USER, "SOFTWARE\\veusz.org\\veusz") as _:
    lang = "ru" if QueryValueEx(_, "ui_english")[0] == "False" else "en"



def f_safe(fun_get_var: Callable[[str], Any], *args, err_out=None, debug=False, **kwargs):
    try:
        return fun_get_var(*args, **kwargs)
    except Exception as e:
        if debug:
            try:
                arg_in = ", ".join(args)
            except Exception:
                arg_in = ""
            exception(arg_in)
        return err_out


def fself(fun, a, *args):
    """Useful when need multi line function that is not exist here and you don't want create one here"""
    fun(a, *args)
    return a


def sl(x) -> slice:
    """
    Slice
    :param x:
    - if number of x elements > 3 takes 1st and last only. Todo: course the indexing to iterate through pairs: possible?
    - ravel x if elements < 2 dimensions
    """
    try:
        x = np.ravel(x)
    except ValueError:
        pass  # setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions.
    args = [None if np.isnan(i) else int(i) for i in x]
    try:
        return slice(*args)
    except TypeError:
        return slice(args[0], args[-1])


# Conversion date from datetime64 to Veusz and back:
dt64s2vsz = lambda dt64: np.float64(dt64) - 1230768000  # 1230768000=int32(datetime64('2009-01-01T00:00:00'))")
vsz2dt64s = lambda t_vsz: np.array(np.int32(t_vsz) + 1230768000, "M8[s]")  # to offset from 1970-01-01


def try_(fun, *args, **kwargs):
    try:
        return(fun(*args, **kwargs))
    except Exception as e:
        warning(e)


def c1(s):
    """Capitalize 1st letter only"""
    return f"{s[0].upper()}{s[1:]}"


en2ru = {
    "#": "№",
    "CTD": "СТД",
    "Day": "День",
    "E": "в.д.",
    "Hz": "Гц",
    "N": "с.ш.",
    "Oxygen concentration": "Концентрация кислорода",
    "TCM": "инклинометр",
    "TCM ": "ИИТ",  # alternative variant
    "Time": "Время",
    "V": "В",
    "a.": "а.",
    "b.": "б.",
    "c.": "в.",
    "accelerometer": "акселерометр",
    "accuracy": "погрешность",
    "after": "после",
    "all": "все",
    "along": "вдоль",
    "and": "и",
    "anchor": "якорь",
    "axis": "ось",
    "azimuth": "азимут",
    "as per": "согласно",
    "at bot.": "у дна",  # not works, so any points are
    "at bottom": "у дна",
    "at": "на",
    "atmospheric": "атмосферное",
    "avg. bin": "ячейка уср.",  # not works with points
    "averaging bin": "ячейка усреднения",
    "avg": "уср",
    "band": "полоса",
    "band-pass": "полосовой",
    "between": "между",
    "bin": "ячейка",
    "blow to": "куда дует",
    "blow from": "откуда дует",
    "bot": "нижн",
    "bottom": "дно",
    "buoy": "буй",
    "by": "по",
    "by data": "по данным",
    "by reanalysis": "по реанализу",
    "by inclinometer": "по инклинометру",
    "by inclinometers": "по инклинометрам",
    "by temperature sensor": "по датчику температуры",
    "by temperature sensor chain": "по термокосе",
    "by tilt current meter": "по инклинометру",
    "by tilt current meters": "по инклинометрам",
    "by weather station": "с метеостанции",
    "calibration": "градуировка",
    "central": "центральный",
    "chain": "цепь",
    "cm": "см",
    "component": "составляющая",
    "concentration": "концентрация",
    "conductivity": "электропроводность",
    "correspondingly": "соответственно",
    "corrected": "исправленный",
    "counts": "отсчеты",
    "current": "течение",
    "current rose": "роза течений",
    "current velocity": "скорость течения",
    "current direction": "направление течения",
    "dBar": "дБар",
    "data": "данные",
    "depth": "глубина",
    "device": "прибор",
    "dev": "пр",
    "direction": "направление",
    "displacement": "смещение",
    "dissolved": "растворенный",
    "downcast": "опускания",
    "each": "каждого",
    "every": "каждые",
    "error": "ошибка",
    "for": "для",
    "g": "г",
    "h": "ч",
    "heaving": "волнение",
    "in": "в",
    "inclination": "наклон",
    "inclinometer": "инклинометрический измеритель",
    "including": "включая",
    "index": "№",
    "interpolation": "интерполяция",
    "interval": "интервал",
    "irrelevant": "ненужный",
    "isoline": "изолиния",
    "isolines": "изолинии",
    "isotherm": "изотерма",
    "its": "его",
    "fit": "аппроксимация",
    "fitted": "аппроксимированный",
    "flow to": "куда течет",
    "filter": "фильтр",
    "filtering": "фильтрация",
    "frequency": "частота",
    "junked": "отфильтровано",
    "kg": "кг",
    "km": "км",
    "l": "л",
    "latitude": "широта",
    "Legend": "Условные обозначения",
    "longitude": "долгота",
    "lowest": "нижн.",
    "m": "м",
    "magnitude": "модуль",
    "magnitude1": "абсолютное значение",
    "magnetometer": "магнитометр",
    "manufacture": "производитель",
    "mean": "среднее",
    "measured": "измеренный",
    "measuredж": "измеренная",
    "measured by inclinometerж": "измеренная инклинометром",
    "measured by inclinometersж": "измеренная инклинометрами",
    "measured by tilt current meterж": "измеренная инклинометром",
    "measured by tilt current metersж": "измеренная инклинометрами",
    "measurement": "измерение",
    "mm": "мм",
    "mmol": "ммоль",
    "model": "модель",
    "month": "месяц",
    "more": "более",
    "moving average": "скользящее среднее",
    "mS": "мСм",
    "mV": "мВ",
    "mg": "мг",
    "name": "название",
    "narrowband": "узкополосный",
    "no": "не",
    "no ": "без ",
    "not": "не",
    "normalized": "нормированные",
    "north": "северный",
    "notation": "обозначение",
    "of": "",
    "of current": "течения",
    "of current velocity": "скорости течения",
    "of isoline": "изолинии",
    "of microprocessor": "микропроцессора",
    "of temperature": "температуры",
    "of temperature sensor": "датчика температуры",
    "of band pass filter": "полосового фильтра",
    "of buoy": "буя",
    "on": "на",
    "on point": "на точку",
    "orig": "исх",
    "original": "исходно",
    "oxygen": "кислород",
    "packet": "пакет",
    "packet start": "начало пакета",
    "parameter": "параметр",
    "past": "от",
    "perpendicular ": "перпендикулярно ",
    "point": "точка",
    "polynom": "полином",
    "pos": "полож",
    "power spectrum density": "спектральная плотность мощности",
    "probe": "датчик",
    "progressive vector diagram": "годограф",
    "pressure": "давление",
    "pressure force": "сила давления",
    "psu": "епс",
    "reanalysis": "реанализ",
    "relative to": "относительно",
    "residual error": "остаточная погрешность",
    "residual errors": "остаточные погрешности",
    "resulting": "полученный",
    "Root": "Корень",
    "raw": "исходные",
    "run": "пуск",
    "s": "c",
    "salinity": "соленость",
    "sampling": "съем данных",
    "sat": "нас",
    "seaward": "со стороны моря",
    "sea depth": "глубина",
    "sedimentary trap": "седиментационная ловушка",
    "selected": "выбранный",
    "sensor": "датчик",
    "Mean Square Error": "среднеквадратичная ошибка",
    "smoothed": "сглажено",
    "std": "CKO",
    "shift": "сдвиг",
    "shore": "берег",
    "shoreward": "со стороны берега",
    "source": "исходный",
    "speed": "скорость",
    "spec": "из спецификации",
    "st": "ст",
    "temperature": "температура",
    "temperature sensor": "датчик температуры",
    "temperature sensor chain": "термокоса",
    "then": "затем",
    "tracker": "трекер",
    "tilt current meter": "инклинометрический измеритель",
    "time": "время",
    "time resolution": "временное разрешение",
    "their": "их",
    "to the shore": "берегу",
    "top": "верх",
    "total precipitation": "сумма осадков",
    "units": "ед.",
    "upcast": "поднятия",
    "useful": "нужный",
    "used": "использовался",
    "used-": "исп.",
    "velocity": "скорость",
    "velocity magnitude": "модуль скорости",
    "vectors": "вектора",
    "vector components": "составляющие вектора",
    "vector projections": "проекции вектора",
    "zeroing": "калибровка нуля",
    "wave gauge": "волнограф",
    "wave": "волнение",
    "weather station": "метеостанция",
    "wider": "шире",
    "wind": "ветер",
    "wind rose": "роза ветров",
    "wind speed": "скорость ветра",
    "with": "с",
}


class DictKeyIfNoVal(dict):
    """
    Class that returns:
    - key if key not in dict with exceptions below,
    - if key[0] is:
        - in range of Cyrillic characters: if En return '' else return key as is
        - equal '_': if En return other chars else split key, translate each word and return combined result words in reverse order. Before translation all "_" in all words are replaced with " " so it "_" can be used to group words with space before translation.
        - is upper, then translates lower word and if success returns translation with 1st capitalized char
    - if key ends with '_' and translation is on (i.e. class is not empty) then returns ''
    else key with last '_' is replaced with ' '. Thus, putting English word with '_' suffix without space to next word allows to remove translating word with space.
    - if key ends with Russian, except 'ж', then removes Russian suffix, and spaces counting spaces Nsp. Output will be translated word except Nsp last letters + Russian suffix.
    - if key ends with digit or 'ж' and class is not empty then tries to return corresponding value from class as usual
    else key without last letter. Letter is used to add additional translation (feminin if ж).

    from translation.
    """

    def __getitem__(self, key):
        b_translate = builtins.bool(self)
        if b_translate:
            out = self.get(key)
            if out is not None:
                # Simple translation was successful
                return out

        # Check 1st character
        char0 = key[0]
        if 0x0400 <= ord(char0) <= 0x04FF:
            # if in range of Cyrillic characters then keep key as is if Ru else delete
            return key if b_translate else ""
        elif char0 == "_":
            return (
                " ".join((self.__getitem__(k) for k in reversed(key[1:].split()))) if b_translate else key[1:]
            )
        elif char0.isupper():
            out = self.get(key)
            if out:
                return c1(out)

        # Check/replace last characters
        i = -1
        char = key[i]
        if char == "_":  # replace in `self` last '_' with ' '
            return "" if b_translate else key[:i] + " "
        elif char.isdigit() or char == "ж":
            return key if b_translate else key[:-1]
        elif 0x0400 <= ord(char) <= 0x04FF:
            # if in range of Cyrillic characters then replace last characters with them if Ru
            # if char in ('а', 'е', 'и', 'х', 'я'):
            i_prev = i - 1
            while 0x0400 <= ord(key[i_prev]) <= 0x04FF:
                i_prev -= 1
                i -= 1
            chars_add = key[i:]

            # remove spaces increasing number of returned characters
            i_remove = i
            # print(key[i_prev], key[:i], i)
            while key[i_prev] == " ":
                i_prev -= 1
                i -= 1
                i_remove += 1

            # English word cleaned
            key = key[:i]  # en_word
            if not b_translate:
                return key

            # Translation
            out = self.get(key)
            if out is not None:
                # Simple translation is successful
                return f"{out[: i_remove or None]}{chars_add}"

            if key[-1] == "s":
                out = self.get(key[:-1])
                return f"{pl(out)[: i_remove or None]}{chars_add}"
        elif char == "s":  # translate without last "s" then make plural
            out = self.get(key[:-1])
            if out:
                return plru(out)
        return key


I = DictKeyIfNoVal(en2ru if lang == "ru" else {})


def plru(text):
    """Make russian word `text` to plural"""
    if len(text) <= 1:
        return text
    before, last = (text[:-1], text[-1])
    match last:
        # last 1 chars dependance
        case "к":
            last = "ки"
        case "р":
            last = "ры"
        case "т":
            last = "ты"
        case "ф":
            last = "фы"
        case "ц":
            last = "цы"
        case "ь":
            last = "и"
        case "а":
            last = "ы"
        # case "й":
        #     last = "и"
        case _:
            # last 2 chars dependance
            before, last = (text[:-2], text[-2:])
            match last:
                case "ая":
                    last = "ие"
                case "ие":
                    last = "ия"
                case "ий":
                    last = "ие"
                case "ка":
                    last = "ках"
                case "ке":
                    last = "ках"
                case "ль":
                    last = "ли"
                case "на":
                    last = "ны"
                    return f"{before}{last}".replace("на глубины", "на глубинах", 1)
                case "ом":
                    last = "ами"
                # case "ма":
                #     last = "мы"
                case _:
                    return text
    return f"{before}{last}"


def pl(text, lang=lang, add_s=True, split=False):
    """
    plural
    :param text:
    :param lang:
    :param add_s: switch mode "each word" / "one word"
    - if True then for each word:
        - if lang is not "ru": adds "s" to each word of length > 1
        - else tries replace known Russian suffixes to plural
    - else replaces "{s}" with "s", then if lang is "ru" replaces known Russian suffixes removing "s" to plural # to do: call plru() for each word ended with {s}
    Russian suffixes
    :return:
    """
    is_ru = lang and lang != "en"
    if add_s:
        if is_ru:
            text_out = " ".join((plru(t) for t in text.split(split))) if split else plru(text)
        else:
            text_out = " ".join((f"{t}s" for t in text.split(split))) if split else f"{text}s"
    else:
        text_out = text.format(s="s")
        if is_ru:
            text_out = (
                text_out.format_map(I)
                .replace("иеs", "ия")
                .replace("ийs", "ие")
                .replace("льs", "ли")
                .replace("фs", "фы")
                .replace("цs", "цы")
            )
    return text_out


# functions useful to translate abbreviations

_RU = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя"
_EN = "ABVGDEËŽZIJKLMNOPRSTUFHCČŠŜʺYʹÈÛÂabvgdeëžzijklmnoprstufhcčšŝʺyʹèûâ"
_RU2EN, _EN2RU = (str.maketrans(*src_dst) for src_dst in ((_RU, _EN), (_EN, _RU)))


def translit_ru_en(s: str) -> str:  # need?
    return s.translate(_RU2EN)


def translit_en_ru(s: str) -> str:
    return s.translate(_EN2RU)


def str_date_unit_fmt(dt, next_fmt, lang=lang):
    """
    Display label of recommended time Units (You need manually set axis properties in accordance to it)
    :param dt: switching interval: 70min, ~2D, 10D, 50D, 2.5*Y  (this definition may not be in sync with code)
    """
    return (
        [
            ("мм:сс", ":%Y-%m-%d %H:00"),
            ("Время", ":%Y-%m-%d"),
            ("День, время", ":%Y-%m"),
            ("День", ":%Y-%m"),
            ("Месяц/день", ":%Y"),
            ("Месяц", ":%Y"),
        ]
        if lang == "ru"
        else [
            ("MM:SS", ":%Y-%m-%d %H:00"),
            ("Time", ":%Y-%m-%d"),
            ("Day, time", ":%Y-%m"),
            ("Day", ":%Y-%m"),
            ("Month/day", ":%Y"),
            ("Month", ":%Y"),
        ]
    )[
        np.fmax(
            np.searchsorted(np.int32([70, 50 * 60, 240 * 60, 1200 * 60, 2.5 * 525600]), dt // 60)
            + next_fmt,
            0,
        )
    ]


def str_date_unit(t_se_vsz, lang=lang, **kwargs):
    """
    kwargs:
    - next_fmt:
    - lang: if 'ru' then русские else English unit names. Default: 'en'
    """
    t_se = vsz2dt64s(np.array(t_se_vsz) + [1, -1]).tolist()
    # - adds 1s shifts is useful for unit intervals starting from round unit make it the same for start and end:
    # this then will be detected and kept only one

    dt = -operator.sub(*t_se_vsz)
    unit, fmt = str_date_unit_fmt(dt, kwargs.get("next_fmt", 0), lang)
    st, en = [f"{{{fmt}}}".format(t) for t in t_se]
    unit_minutes = unit.startswith(("MM", "мм"))
    if st == en or unit_minutes:
        s = st
    else:
        idiff = [i for i, (left, right) in enumerate(zip(st, en)) if left != right][0]
        s_diff = st[idiff:]
        if "-" not in s_diff and " " not in s_diff:  # can remove part from en before isplit
            isplit = st[:idiff].rfind(" ")
            if isplit != -1:
                s = "\u2009–\u2009".join((st, en[isplit + 1 :]))
            else:
                isplit = st[:idiff].rfind("-")
                if isplit != -1 and int(en[isplit + 1 :]) - int(st[isplit + 1 :]) == 1:
                    s = ",\u2009".join((st, en[isplit + 1 :]))
                else:
                    s = "\u2009–\u2009".join((st, en))
        else:
            s = "\u2009–\u2009".join((st, en))
    preposition = (" {past} " if unit_minutes else " {of_}").format_map(I)
    return "".join([unit, preposition, s])


def str_date_unit_nl(*args, allow3rows=False, no_blank_at_end=False, **kwargs):
    """
    Make label shorter for short graphs splitting row (with aligning digits by inserting ``blank``)
    """
    s = str_date_unit(*args, **kwargs)
    blank = "\\color{transparent}{\u2009–}"
    if "–" in s and allow3rows:
        return s.replace("of", f"of{blank}\\\\").replace("–\u2009", "–\\\\") + (
            "" if no_blank_at_end else blank
        )
    else:
        return s.replace("2", "\\\\2", 1)


def str_date_unit_with_suffix(t_range, str_zone, **kwargs):
    """
    Used in Veusz Custom Definition as:
    str_date_u = (
    lambda ax, t_span_var, **kwargs:
    str_date_unit_with_suffix([f(lambda l: l if l!='Auto' else t, SETTING(f'{ax}/{lim:s}')) for lim, t in zip(('min', 'max'), DATA(f'{t_span_var}'))], str_zone='UTC+02:00', lang=LANG({'default': 'en', 'ru': 'ru'}), **kwargs)
    """
    b_nl = kwargs.pop("b_nl", False)
    higher = kwargs.pop("higher", False)
    str_date_unit_result = (str_date_unit_nl if b_nl else str_date_unit)(
        t_range, no_blank_at_end=str_zone, **kwargs
    )
    if str_zone and ":" not in str_date_unit_result:
        str_zone = ""
    return (
        f"{str_date_unit_result}{(chr(92) * 2 if higher else chr(8201))}"
        f"{('^' if str_date_unit_nl and str_zone else '')}{str_zone}{chr(92) * 2 * (higher - 1)}"
    )


def decompose_duration(dt_s: float | int) -> tuple[int, int, int, int, int, int, int]:
    """
    Decompose scalar seconds into chronological tuple:
    (years, months, days, hours, minutes, seconds, microseconds).

    Employs cascading `divmod` for O(1) zero-allocation projection.
    Resolves month/year ambiguity via Gregorian mean constants.
    """

    # Gregorian mean invariants
    _Y = 365.2425       # Mean year in days
    _M = _Y / 12        # Mean month in days

    total_us = int(round(dt_s * 1_000_000))

    s, us = divmod(total_us, 1_000_000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)

    y, d = divmod(d, _Y)
    mo, d = divmod(d, _M)

    return [int(y), int(mo), int(d), h, m, s, us]


def str_dt(dt_s: int | float | NDArray, lang=lang):
    """
    Time interval to readable string
    :param dt_s: time, s
    """
    a = decompose_duration(dt_s.item() if hasattr(dt_s, "item") else dt_s)
    # np.array(np.int64(dt_s * 1000000), "M8[us]").item()
    # a = np.int16(s.timetuple()[:6]) - [1970, 1, 1, 0, 0, 0]
    # if ~np.any(a):
    #     a = [0, 0, 0, 0, 0, 0, np.round(s.microsecond * 1e-06, 3)]
    if any(round_to_s:=a[:6]):
        a = round_to_s
    else:
        a[6] *= 1e-06  # to s
    out = " ".join([
        f"{d}{w}"
        for d, w in zip(
            a,
            ["лет", "месяцев", "дней", "ч", "мин", "с", "с"]
            if lang == "ru"
            else ["years", "months", "days", "h", "min", "s", "s"],
        )
        if d
    ])
    return out.strip()


def day_sfx(d):
    return {1: "st", 2: "nd", 3: "rd"}.get(d % 20, "th") if lang != "ru" else ""


def str_time_range(
    st: datetime,
    en: datetime,
    date_format: str = "%d.%m.%Y",
    time_format: str = "%H:%M",
    str_zone: str = "",
    sep: str = "\u2009",
    sep_interval: str = "\u2009–\u2009",
):
    """
    Time range string without repeating not changed time units in date format
    :param st:
    :param en:
    :param date_format:  After %d there may be {sfx} that will be replaced to appropriate day suffix
    :param str_zone: time text suffix
    :param sep: output white space separator between date and time: default - unicode "\u2009" (short space)
    :param sep_interval: separator between 1st and last datetime
    :return:
    """
    if not isinstance(st, datetime):
        return ""
    str_st_date = f"{st:{date_format}}"
    str_en_date = f"{en:{date_format}}"
    if "{sfx}" in date_format:
        str_st_date = str_st_date.replace("{sfx}", day_sfx(st.day))
        str_en_date = str_en_date.replace("{sfx}", day_sfx(en.day))
        if "%e" in date_format:
            str_st_date = str_st_date.replace(". ", ".").replace("- ", "-").strip()
            str_en_date = str_en_date.replace(". ", ".").replace("- ", "-").strip()
    b_ddate = str_st_date != str_en_date
    if b_ddate:  # - Keep only different parts
        i_split = 0  # previous date part separator index
        b_inc = "%d" == date_format[:2]  # date parts (units) in increased order
        for i, (left, right) in enumerate(
            zip(reversed(str_st_date), reversed(str_en_date)) if b_inc else zip(str_st_date, str_en_date)
        ):
            if left in ".- \\":
                i_split = i
            elif left != right:
                if i_split:
                    if b_inc:
                        str_st_date = f"{str_st_date[slice(0, -i_split - 1)]}"
                    else:
                        str_en_date = f"{str_en_date[slice(i_split + 1, None)]}"
                break
        str_en_date = [str_en_date]
    else:
        str_en_date = []
    st, en = [[f"{t:{time_format}}"] for t in (st, en)] if time_format else [[], []]
    return (
        sep_interval.join([sep.join(d_t) for d_t in ([str_st_date] + st, str_en_date + en) if d_t]) + str_zone
    )


def str_deg_min(degfloat, strpattern="{:d}°\u2009{:0.4f}'", *args):
    """equiv. old variant: strpattern % (trunc(degfloat), abs(degfloat - trunc(degfloat))*60)"""
    part_rem, part_trunc = np.modf(degfloat)
    return strpattern.format(int(part_trunc), np.abs(part_rem) * 60, *args)


def str_deg_min_join(degfloat, strpattern="{:d}°\u2009{:0.4f}'", add_strs="NE", joiner=", "):
    return joiner.join((str_deg_min(d, strpattern, a) for d, a in zip(degfloat, add_strs)))


def row_jumps_if_small_dx(x, dx_min):
    """
    Changes row if distance to previous element on row is too small:
    for preventing overlapping of many close graphical elements if they would be placed on one row
    """
    p_prev = [-np.inf] * 10  # can distribute to 10 rows maximum
    x_row = []
    for p in x:
        for row in range(len(p_prev)):
            if p - p_prev[row] > dx_min:
                x_row.append(row)
                p_prev[row] = p
                break
    return x_row


def sum_of_scaled_gaussians(x, peaks=[60, 120], sigmas=[30, 30], scales=[1, 1, 1]):
    """
    Sum of scaled gaussians and base shifts useful to create weights array of x
    :param x: input array
    :param peaks: Центры пиков
    :param sigma: std of norm distibutin: область без резкого спада widths = sigma*2
    :param scales: for elements which has no pair in peaks the addition term will be constant equal to that
    scale
    Note: for generating weights for hue (range=360) peaks x=[60, 120](цветов, которые лучше различаются)
    with sigma=30 output will be ≈0 in > than half of range if call with all arguments of same length and
    equal scales.
    :
    """
    y = 0
    for mu, sigma, scale in zip_longest(peaks, sigmas, scales):
        term = scale if mu is None else scale * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        y += term
    # Нормализация
    y /= y.sum()
    return y


def hsv_to_rgb(hues, s=1.0, v=1.0):
    """Convert HSV to RGB (vectorized for performance)"""
    hues %= 360
    h = hues / 60.0
    c = v * s
    m = v - c
    x = c * (1 - np.abs(h % 2 - 1))
    z = 0
    sector = np.floor(h).astype(int) % 6
    choices = ([c, x, z, z, x, c], [x, c, c, x, z, z], [z, z, x, c, c, x])
    rgb = np.stack([np.choose(sector, ch) for ch in choices], axis=-1)
    rgb += m if isinstance(m, float) else m[:, None]
    rgb *= 255
    return np.clip(rgb, 0, 255).astype(np.uint8)


def colors_of_hue_range(n, exclude_range=[0, 0], s=1.0, v=1.0, weight=False, out_format="#rgb"):
    """
    Generate n vivid colors with with hue range exclusion
    :param n: Number of colors to generate
    :param exclude_range: [exclude_start, exclude_end]: Start and end of hue range to exclude (degrees).
    If `exclude_end < exclude_start` then excluded range wraps through 360, => we will just generate from exclude_end to exclude_start
    :param s: Saturation (0-1), default 1.0 for vivid colors
    :param v: Value (0-1), default 1.0 for vivid colors
    :param weight: if not Falsy, modulate perceptual density of output:
    - True: increase in green/yellow regions
    - callable: your weight function on hue
    :param out_format: "#rgb" / "list_hsv" / other for rgb_colors array
    """
    if n <= 0:
        return []

    total_range = 360
    exclude_start, exclude_end = [lim % total_range for lim in exclude_range]

    # hues avoiding exclusion zone
    n_generate = np.fmax(1000, n * 100) if weight else n
    b_wrap_out = exclude_start <= exclude_end
    if b_wrap_out:
        hues_lin = (
            np.linspace(exclude_end, exclude_start + total_range, n_generate, endpoint=False) % total_range
        )
    else:
        hues_lin = np.linspace(exclude_start, exclude_end, n_generate, endpoint=False)

    if weight:
        # Apply perceptual adjustment
        if weight is True:
            # Calculate perceptual weights with peaks at 60° and 120° (Yellow & Green)
            # (to denser sampling from  generated range in green/yellow regions):
            # using sum of 2 gaussian distribution exp(-0.5 * ((x - mu) / sigma)**2):
            def weight(h):  # peaks=[60, 120]
                return sum_of_scaled_gaussians(h, peaks=[50, 310], sigmas=[30, 20], scales=[1, 1, 1])

            # or (emphasizing 60°-120° range): Flattened peak around greens
            # weights = exp(-0.5 * ((hues_lin - 90) / 45) ** 4)

        weights = weight(hues_lin)
        # Select n hues
        cdf = np.cumsum(weights)
        sample_points = np.linspace(0, cdf[-1], n, endpoint=False)
        indices_selected = np.searchsorted(cdf, sample_points)
        hues = hues_lin[indices_selected]
    else:
        hues = hues_lin
    # Gradient in brightness for visual separation from 40% to 80%
    if callable(v):
        v = v(hues / total_range)

    # Convert to RGB using existing hsv_to_rgb
    rgb_colors = hsv_to_rgb(hues, v=v)
    if out_format == "#rgb":
        return [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in rgb_colors]
    elif out_format == "list_hsv":
        return [hues, 1, v]
    else:
        return rgb_colors


def dim_bug_cor(dim):
    """former BugDimCorr"""
    if len(dim) > 1:
        d_edges = np.array(dim)[[0, -1]]
        return d_edges - operator.sub(*d_edges) * np.array([0.25, -0.25])
    else:
        warning("len(dim) < 1")
        return dim + [-1e-08, 1e-08]


def i_positive(i, lim):
    """Return positive indices within the limit."""
    ii = np.int64(i)
    return np.where(ii < 0, lim + ii, ii)


def i_ranges(
    t,
    t_ranges,  # : Union[list[Union[list[Union[str, np.datetime64, int]], int]], np.ndarray]
    t_shift_s=0,
    t_units="ns",
):
    """
    Find indexes of (useful) time ranges specified by t_ranges
    :param t: raw datetime64[ns] array or veusz time (in this case you should add 1230768000 to t_shift_s)
    :param t_ranges: iterable of ranges (iterables of 2 values) or single values. Values may be:
    - datetime64[ns] time - to search its integer index in `t` or
    - integer index - to return such elements without changes or
    - mix of described above
    :param t_shift_s: time shift [s], used to find indexes of t shifted on this value (after adding to t)
    :param t_units: units of t: can be 's', 'ns', ...
    return: list of 2-el. sequences ranges indexes
    """
    if (builtins.any if isinstance(t_ranges, list) else np.any)(t_ranges):
        t = np.int64(t)
    else:
        return [[0, t.size]]

    def to_two_elements(x):
        return x if len(x) == 2 else [x, x + 1] if len(x) == 1 else [0, t.size]

    out = [
        to_two_elements([
            i_or_t
            if isinstance(i_or_t, int)
            else np.searchsorted(
                t, np.int64(np.array(i_or_t, f"datetime64[{t_units}]") - np.timedelta64(t_shift_s, "s"))
            )
            for i_or_t in time_iter
        ])
        for time_iter in t_ranges
    ]
    return out


def i_use(t, time_iter, t_shift_s=0, t_units="ns"):
    """
    Same as i_ranges() but for t_ranges replaced with time_iter that can be 1D or 2D.
    :param t: raw datetime64[t_units] array or its directly converted to float64 version
    :param time_iter: 1D or 2D values (in that case will be taken 1st 1D el. only) of same type as
    `t_ranges `parameter of i_ranges()
    :param t_shift_s: time shift [s], used to find indexes of t shifted on this value (after adding to t)
    :param t_units:
    :return: 1D array of indexes of time_iter in t or [0, t.size]
    """
    t = np.int64(t)
    time_iter = time_iter[0] if len(np.shape(time_iter)) > 1 else time_iter
    dtime_shift = np.timedelta64(t_shift_s, "s").astype(f"m8[{t_units}]")
    out = [
        i_positive(x, t.size)
        if isinstance(x, int)
        else t.size
        if x is None
        else np.searchsorted(t, np.int64(np.array(x, f"M8[{t_units}]") - dtime_shift))
        for x in time_iter
    ]
    if len(out):
        n_out = -operator.sub(*out)
        if n_out <= 1:
            warning(
                "Souse time range: [{}] is completely out of user selected time range ({})!".format(
                    ", ".join(
                        (
                            f'"{ti}"'
                            for ti in np.array(t[[0, -1]] + np.int64(dtime_shift), f"M8[{t_units}]").astype(
                                "M8[s]"
                            )
                        )
                    ),
                    time_iter,
                )
            )
        return out
    else:
        return [0, t.size]


def lim_range_min(range, min_range):
    dat_min, dat_max = range
    lim_min, lim_max = min_range
    if dat_min < lim_min:
        return [dat_min, dat_min - operator.sub(*min_range)]
    elif dat_max > lim_max:
        return [dat_max + operator.sub(*min_range), dat_max]
    else:
        return min_range


def min_range(range1, range2, l=np.nan):
    """accounts for negative indexes"""
    st, en = i_positive([np.take(range1, [0, -1]), np.take(range2, [0, -1])], l).T
    return [np.transpose([np.max(st), np.min(en)])]


def min_range_2d(*args):
    """The min_range_2d_no_check(se1, se2) replace. Still no check negative indexes"""
    a_any = [a for a in args if np.any(a)]
    if a_any:
        return np.atleast_2d([np.nanmax([a[:, 0] for a in a_any]), np.nanmin([a[:, -1] for a in a_any])])
    else:
        return [[]]


def max_range(range1, range2):
    """Maximum range from input ranges limits"""
    st1, en1 = range1
    st2, en2 = range2
    return np.append(np.fmin(st1, st2), np.fmax(en1, en2))


def ceil_log(x, div=2):
    """Good maximum for axis limit to display x values
    >> ceil_log(x) for x in [0.01, 0.013, 0.017, 0.2, 0.8, 1.2, 1.8, 2.2, 2.8, 11, 44]]
    ... [0.01, 0.015, 0.02, 0.2, 0.8, 1.5, 2.0, 2.5, 3.0, 15.0, 45.0]
    >> ceil_log(x, 1) for x in [0.01, 0.013, 0.017, 0.2, 0.8, 1.2, 1.8, 2.2, 2.8, 11, 44]]
    ... [0.01, 0.02, 0.02, 0.2, 0.8, 2.0, 2.0, 3.0, 3.0, 20.0, 50.0]
    """
    r = div * 10 ** (-np.floor(np.log10(x)))
    return np.ceil(x * r) / r


def power_ceil(x):
    return np.int32(np.floor(np.log10(x)))


def round_ceil_signed(x, n: float = None):
    """n: if positive should be float (not numpy type like float32)"""
    s = np.sign(x)
    abs_x = np.absolute(x)
    p = 10 ** (-(power_ceil(abs_x) if n is None else n))
    return s * np.ceil(abs_x * p) / p


def shift_or_extend_lims(lim_in, x_in, e=None, scale=1):
    """Shifts range to include x if can, else extend range"""
    lim = np.float64(lim_in)
    x = np.float64(x_in)
    x_range = -operator.sub(*x).item()
    if x_range > 0:
        if e is None:
            # add 0.05 * x-range rounded to 1 decade
            e = 10 ** (int(np.floor(np.log10(x_range))) - 1) / 2
        dl = max_range(lim, (x + [-e, e]) * scale) - lim
        lim_out = (
            max_range(lim + dl[0], x * scale)
            if dl[0]
            else max_range(lim + dl[1], x * scale)
            if dl[1]
            else lim
        )
        return lim_out
    else:
        return lim


def min2nan(a, minLim):
    return np.where(a < minLim, np.nan, a)


def max2nan(a, maxLim):
    return np.where(a > maxLim, np.nan, a)


def minmax2nan(a, minLim, maxLim):
    return np.where((a < minLim) | (a > maxLim), np.nan, a)


def movavg_1d(a, n):
    if n > 2 and np.size(a) >= n:
        n_m2 = np.int32(n // 2) + 1
        n_p2 = np.int32((1 + n) // 2)
        cum_a = np.cumsum(a)
        diff_width_n = cum_a[np.int32(n) :] - cum_a[: -np.int32(n)]
        return np.hstack((
            np.cumsum(a[:n_m2]) / np.arange(1, n_m2 + 1),
            diff_width_n / n,
            np.cumsum(a[:-n_p2:-1])[::-1] / np.arange(n_p2 - 1, 0, -1),
        ))
    elif n == 2:
        return np.hstack((a[0], (a[1:] + a[:-1]) / 2.0))
    else:
        return a


def moving_std(data: np.ndarray, window: int):

    # Compute cumulative sums
    cumsum1 = np.cumsum(data)
    cumsum2 = np.cumsum(data**2)

    # Calculate sums over the moving window
    sum_window = cumsum1[window:] - cumsum1[:-window]
    sum2_window = cumsum2[window:] - cumsum2[:-window]

    # Calculate means and variances
    mean_window = sum_window / window
    var_window = sum2_window / window - mean_window**2

    # Return the standard deviation
    std_window = np.sqrt(var_window)

    # Use numpy.pad to pad edges with the first and last valid standard deviation
    std_full = np.pad(std_window, (window - 1, len(data) - len(std_window) - (window - 1)), mode="edge")
    return std_full

    # moving_std = sqrt(convolve(data**2, ones(window), 'valid') / window -
    #     convolve(data, ones(window), 'valid')**2 / window**2)


def rep2mean(x, b_ok=None, left=None, right=None):
    if b_ok is None:
        b_ok = np.isfinite(x)
        return np.interp(np.arange(len(x)), np.flatnonzero(b_ok), x[b_ok])
    return np.interp(np.arange(len(x)), np.flatnonzero(b_ok), x[b_ok], left, right)


def rep2mean_dir2rad(x, b_ok):
    return np.interp(np.arange(len(x)), np.flatnonzero(b_ok), np.radians(x[b_ok]), period=2 * np.pi)


def rep2prev(x, b_ok=None):
    if b_ok is None:
        b_ok = np.isfinite(x)
    ok = np.asarray(b_ok, dtype=np.int8)
    i_before_nan = np.flatnonzero(np.diff(ok) < 0)
    i_last_nan = np.flatnonzero(np.diff(ok) > 0)
    if i_before_nan.size and i_last_nan.size:
        # Prepare closed NaN regions to have equal values on edges:
        # delete edges for which opposite edge is open end delete them
        if np.isnan(x[0]):  # 1st NaNs have no finite value before
            del i_before_nan[0]
        if i_before_nan.size:  # last NaNs have no finite value after
            if np.isnan(x[-1]):
                del i_last_nan[0]
            out = x.copy()
            out[i_last_nan] = out[i_before_nan]
            b_ok[i_last_nan] = True
    out = np.interp(np.arange(len(x)), np.flatnonzero(b_ok), out[b_ok])
    return out


# @njit
def b1spike(a, max_spike=0):
    """
    Single spike detection
    Note: change of a at edge bigger than max_spike is treated as spike too
    :param a:
    :param max_spike:
    :return: boolean array of where is spike in a
    """
    b_single_spike_1 = lambda bad_u, bad_d: np.logical_or(
        np.logical_and(np.append(bad_d, True), np.append(True, bad_u)),  # spike to down
        np.logical_and(np.append(bad_u, True), np.append(True, bad_d)),
    )  # spike up
    diff_x = np.diff(a)
    return b_single_spike_1(diff_x < -max_spike, diff_x > max_spike)


def bin_avg(a: np.ndarray, edges, st=0):
    """
    Bin average of ``a[st:after_last]`` inside edges where ``after_last = int(edges[-1])`` - index of the end of the last interval
    :param a: numpy ndarray
    :param edges: ``a`` indexes relative to ``st`` to calculate mean(a) between
    :param st: ``edges`` origin in ``a`` indexes
    """
    edges_int = np.int32(edges)
    starts, after_last = np.array_split(edges_int, [-1])
    return (
        np.add.reduceat(a[st : after_last.item()], starts - st if st else starts) / np.ediff1d(edges_int)
    )[:, None if a.ndim > 1 else ...]


def bin_std(a: np.ndarray, edges: np.ndarray, st=0):
    """
    Bin average of ``a[st:after_last]`` inside edges where ``after_last = int(edges[-1])`` - index of the end of the last interval
    :param a: numpy ndarray
    :param edges: ``a`` indexes relative to ``st`` to calculate mean(a) between
    :param st: ``edges`` origin in ``a`` indexes
    """
    edges_int = np.int32(edges)
    starts, after_last = np.array_split(edges_int, [-1])

    # Calculate sums over intervals
    sum_bin = np.add.reduceat(a[st : after_last.item()], starts - st if st else starts)
    sum_sq = np.add.reduceat(a[st : after_last.item()] ** 2, starts - st if st else starts)

    n = np.ediff1d(edges_int)  # bin length
    variance = (sum_sq - sum_bin**2 / n) / n

    # Return the standard deviation
    return np.sqrt(variance)[:, None if a.ndim > 1 else ...]


def bool2ranges(b_ok, min_range, min_range_bad=None, pressure=None):
    """
    Get changing edges ignoring short intervals between (set ``min_range``s 0 to not ignore), set:
    - ``min_range``     larger to del wider data,
    - ``min_range_bad`` larger to del wider spikes.
    :param b_ok:
    :param min_range:     min number of elements in intervals where b_ok is True
    :param min_range_bad: min number of elements in intervals where b_ok is False
    :return:
    """
    d_ok = np.diff(b_ok, prepend=False, append=False)
    edges = np.flatnonzero(d_ok != 0)
    n_rows = np.diff(edges)
    # Delete too short bad data intervals
    if pressure is None:
        b_del_bad_interval = n_rows[1::2] < (min_range_bad or min_range)
    else:
        dp = np.abs(np.diff(pressure[edges]))
        b_del_bad_interval = n_rows[1::2] + dp[1::2] < (min_range_bad or min_range)
    if b_del_bad_interval.any():
        edges = edges[np.hstack((True, ~np.repeat(b_del_bad_interval, 2), True))]
        n_rows = np.diff(edges)
    # Delete too short good data intervals
    if pressure is None:
        b_del_good_interval = n_rows[::2] < min_range
    else:
        dp = np.abs(np.diff(pressure[edges]))
        b_del_good_interval = n_rows[::2] - dp[::2] < min_range
    if b_del_good_interval.any():
        edges = edges[~np.repeat(b_del_good_interval, 2)]
    return edges


def ranges2bool(st_en, length):
    a = np.zeros(length, np.bool_)
    for se in np.int32(st_en):
        a[slice(*se)] = True
    return a


def put_nans(x, st_ends):
    """
    x: any dimensional array of size n
    st_ends: integer array (n, 2) of starts and ends
    return x copy with NaNs from st to end along last axis
    """
    xc = x.copy()
    for se in st_ends:
        xc[..., slice(*se)] = np.nan
    return xc


def put_in_nans(x, st_ends, x_len=None):
    if x_len is None:
        x_len = x.shape[1]
    elif x_len != x.shape[1] or st_ends[0, 0] != 0:
        xc = np.empty((x.shape[0], x_len)) + np.nan()
        xc[..., st_ends[0, 0] : x_len] = x
    else:
        xc = x.copy()
    for se in zip([0] + st_ends[:, 1].tolist(), st_ends[:, 0].tolist() + [x_len]):
        xc[..., slice(*se)] = np.nan
    return xc


def at_or_other(arr, index=0, other=0):
    """
    index: index of ``arr`` element to return it if ``arr`` is not empty/zero
    other: return value if ``arr`` is empty/zero

    Useful when, for example, arr is a result of flatnonzero() and we need 1st (or last) value of it, but if it is empty we replace it to ``other``
    """
    return arr[index] if arr.size else other


def in_ranges(arr, fun, st_ends, where_no_fun_out=None):
    """
    Return array having elements: fun(arr[start:end]) where start and end are elements of ``st_ends``
    fun: function(arr, other=0) if where_no_fun_out not None else function(arr)
    st_ends: int32
    where_no_fun_out: int32 array of st_ends length, int, or None. Set to not None to use as 2nd argument of fun.
    Note: if not None and start >= end then element will be assigned to ``where_no_fun_out`` (or its element if array)
    and fun() will not be executed.
    Example:
    # index of max dens in pres_range before imin_ddens
    v.in_ranges(dens, nanargmax, st_ends)
    """
    if where_no_fun_out is None:
        where_no_fun_out = np.nan

        def fun_in(se):
            return fun(arr[slice(*se)])

    elif hasattr(where_no_fun_out, "__len__"):

        def fun_in(se, other):
            return fun(arr[slice(*se)], other=other)

        return np.int32([
            fun_in(se, other) if operator.lt(*se) else other for se, other in zip(st_ends, where_no_fun_out)
        ])
    else:

        def fun_in(se):
            return fun(arr[slice(*se)], other=where_no_fun_out)

    return [fun_in(se) if operator.lt(*se) else where_no_fun_out for se in st_ends]


def i_before(pres, pres_range, st_ends):
    """
    Start index of ``pres_range`` in ``pres`` before "ends" in ``st_ends``
    :param pres:
    :param pres_range:
    :param st_ends:
    :return:
    """
    p_inv = -np.flip(pres)
    st_ends_inv = pres.size - np.fliplr(st_ends)
    i_st = pres.size - (
        np.flip(
            in_ranges(
                p_inv,
                lambda d, other: np.searchsorted(d, d[0] + pres_range),
                st_ends_inv,
                where_no_fun_out=pres.size,
            )
        )
        + st_ends_inv[:, 0]
    )
    return i_st


def bad_bot_by_diff(
    x: np.ndarray, fun, i_en, i_st=None, p_range=None, p: np.ndarray = None, speed: np.ndarray = 1
):
    """
    Identify and replace "bad" data segments in `x` based on a detection function `fun`,
    using pressure-based constraints (if provided) and velocity scaling (`speed`).

    This function locates intervals defined by `i_st` (start) and `i_en` (end), then
    searches within each interval for points where `fun(x)` returns True. If `p_range`
    and `p` are provided, the search window is adjusted to be within `[p[i_en] - p_range, p[i_en]]`.

    The detected "bad" regions are then replaced with NaNs.

    x : ndarray
        The signal array to filter (e.g., sensor readings).
    fun : Callable[[ndarray], ndarray]
        A function that takes a segment of `ediff1d(x, to_begin=0)/speed` and returns a boolean mask or
        indices indicating positions of "bad" data.
    i_en : ndarray
        End indices of intervals to analyze. Must be 1D.
    i_st : Optional[Union[int, ndarray]], default None
        Start indices of intervals. If None, defaults to:
        - `0` if `p` is not provided,
        - Otherwise, the first finite index in `p` followed by `i_en[:-1] + 1`.
        If an integer is given, it is used as a fixed offset from `i_en[:-1]`.
    p_range : Optional[float], default None
        If provided, defines the pressure range backward from `p[i_en]` to define
        the search window: `[p[i_en] - p_range, p[i_en]]`. Requires `p` to be provided.
    p : Optional[ndarray], default None
        Pressure (or coordinate) array corresponding to `x`. Required if `p_range` is set.
    speed : ndarray, default 1
        Velocity or scaling factor applied to differences in `x`. Should have length `len(x) - 1`.
        Used to normalize the difference signal before applying `fun`.

    :return: A copy of `x` with identified "bad" segments replaced by NaN.

    Original lambda call:
    (lambda st_ends: st_ends[:,0] + int32([searchsorted(_Pres[slice(*se)], p_max - 2) for se, p_max in zip(st_ends, fd2_Pmax)])(int32(column_stack((d04_ist_en[:,0], fd1_iPmax))) )
    """
    have_p = p is not None
    st_ends = np.int32(
        np.column_stack((
            i_st
            if hasattr(i_st, "__len__")
            else np.append(
                np.flatnonzero(np.isfinite(p))[0] if have_p else 0, i_en[:-1] + (1 if i_st is None else i_st)
            ),
            i_en,
        ))
    )

    # Adjust search start points based on pressure range (if applicable)
    if have_p:
        st_ends[:, 0] = i_before(p, p_range, st_ends)

    # Search fun() points and replace starting search points with them
    st_ends[:, 0] += in_ranges(
        np.ediff1d(x, to_begin=0) / speed,
        lambda xr, other: at_or_other(np.flatnonzero(fun(xr)), 0, other=other),
        st_ends,
        where_no_fun_out=x.size,
    )
    return put_nans(x, st_ends)


def bad_top_by_diff(
    x: np.ndarray, fun, i_en, i_st=None, p_range=None, p: np.ndarray = None, speed: np.ndarray = 1
):
    """
    :param x: _description_
    :param fun: _description_
    :param i_st: _description_, defaults to None
    :param i_en: _description_

    :param p_range: _description_, defaults to None
    :param p: _description_, defaults to None
    :param speed: _description_, defaults to 1
    Example:
    # v.bad_bot_by_diff(CTD_Sal_f, lambda x: (x<-0.1)| (x>5), CTDends, i_st=CTDstarts, p_range=1, p=CTD_Pres, speed=CTDspeedDown_MA)
    # lambda x: (x<-0.001)|(x>0.01)
    """
    x_len = x.size
    return np.flip(
        bad_bot_by_diff(
            np.flip(x),
            fun,
            i_en=x_len - i_st,
            i_st=x_len - i_en,
            p_range=p_range,
            p=-np.flip(p) if p is not None else None,
            speed=-np.flip(speed) if speed is not None else None,
        )
    )


def loop_filt(p, i_st: int = 1) -> np.ndarray:
    """
    Remove loops from CTD profile using global accumulate from anchor.

    Algorithm:
    1. Downward pass (anchor to end): keeps points strictly greater than all previous
    2. Upward pass (anchor to start): keeps points strictly less than all previous

    Anchor point itself is included only if it passes monotonicity checks

    :param p: sequence of depth values from CTD profile
    :param i_st: anchor index to start monotonic selection
    :return: Boolean mask indicating which points to keep (True = keep)

    Examples:
    >>> p = array([1, 2, 4, 3, 1, 5, 6, 7])
    >>> p[loop_filt(p, 3)]  # doctest: +ELLIPSIS
    Downward pass indexes: ...
    ...
    array([1, 2, 4, 5, 6, 7])
    >>> p[loop_filt(p, 7)]  # doctest: +ELLIPSIS
    Downward pass indexes: ...
    ...
    array([1, 5, 6, 7])

    """
    n = len(p)
    mask = np.zeros(n, dtype=np.bool)
    if i_st > 0:
        # Downward pass: anchor to end
        if i_st < n:
            p_down = p[i_st - 1 :]
            max_acc = np.maximum.accumulate(p_down)[:-1]
            mask[i_st:] = p_down[1:] > max_acc
            i_down = np.flatnonzero(mask[i_st:])
            p0down = p_down[i_down[0] + 1] if i_down.size else None
        else:
            p0down = None

        # Upward pass: anchor to start
        p_up = p[i_st::-1]
        # If there is a selected point down, we limit the anchor to it
        if p0down is not None:
            p_up[0] = np.fmax(p_up[0], p0down)
        min_acc = np.minimum.accumulate(p_up)[:-1]
        mask[i_st - 1 :: -1] = p_up[1:] < min_acc
    elif i_st < n:
        # Downward pass: anchor to end
        max_acc = np.maximum.accumulate(p)
        mask[1:] = p[1:] > max_acc[:-1]
        i_down = np.flatnonzero(mask)
        mask[0] = p[0] < p[i_down[0]] if i_down.size else True
    return mask


def filt_ctd_surface(
    c: np.ndarray,
    pres: np.ndarray,
    *,
    dt: Optional[float] = None,
    max_depth: float = 3.0,
    win: int = 7,
    k_mad: float = 5.0,
    k_d: float = 8.0,
) -> np.ndarray:
    """
    Boolean quality mask for removing air-bubble-contaminated salinity - **not tested**
    (or conductivity) measurements near the surface.


    :param max_depth: only samples with pres[i] ≤ max_depth are eligible for rejection.
    :param pres: pressure, used also to remove any data acquired before the (last) minimum pressure, i.e.
     good[i] = False for all i < argmin(pres)

    :param c: parameter to filter - use conductivity (preferred) if available else use salinity
    :param win: odd window to filter `c` by calculating local robust reference (level) for each eligible
    index i with a full window:
        median(c)_i = median{c_j | j ∈ [i−h, i+h]} - rolling median,
        MAD(c)_i = median{|c_j − median(c)_i| | j ∈ [i−h, i+h]} - rolling MAD,
        where h = (win−1)/2
        i = 0..N−1 index the samples.

    Reject sample if it satisfies either:
    1. Level-based spike condition (asymmetric):
    A sample i is considered contaminated if
        c_i < median(c)_i − k_mad · MAD(c)_i
    Only negative deviations are tested, reflecting the physical signature of air bubbles (conductivity loss).

    2. Dynamic spike (derivative) condition (if dt is provided):
        v_i < median(v)_i − k_d · MAD(v)_i,
    where
        v_i = (c_i − c_{i−1}) / dt - discrete time derivative
        median and MAD of v computed using the same rolling window
    This detects sharp negative transitions even for wide spikes.

    Notes
    -----
    - No assumption is made about monotonic pressure, cast direction, or existence of a continuous downcast.
    - No explicit wet/dry pressure threshold is used.
    - Spike length is not constrained: both short and long air events are rejected purely by robust statistics.
    """
    n = c.size
    good = np.ones(n, dtype=np.bool)
    i0 = int(np.argmin(pres))
    good[:i0] = False
    good &= pres <= max_depth
    if win % 2 == 0:
        raise ValueError("win must be odd")
    h = win // 2
    good[:h] = False
    good[-h:] = False
    xw = sliding_window_view(c, win)
    med = np.median(xw, axis=1)
    mad = np.median(np.abs(xw - med[:, None]), axis=1)
    mad[mad == 0] = np.nan
    idx = np.arange(h, n - h)
    level_spike = c[idx] < med - k_mad * mad
    if dt is not None:
        dxdt = np.diff(c, prepend=c[0]) / dt
        dw = sliding_window_view(dxdt, win)
        dmed = np.median(dw, axis=1)
        dmad = np.median(np.abs(dw - dmed[:, None]), axis=1)
        dmad[dmad == 0] = np.nan
        deriv_spike = dxdt[idx] < dmed - k_d * dmad
    else:
        deriv_spike = np.zeros_like(level_spike)
    bad = level_spike | deriv_spike
    good[idx[bad]] = False
    return good


def f_iBadBefore(arr, fun_bad, i_en, to_end=50):
    """
    to_end: int32
    """
    st_ends = np.int32(np.column_stack((i_en - to_end, i_en)))
    return (
        np.int32([at_or_other(np.flatnonzero(fun_bad(arr[slice(*se)])), 0, to_end) for se in st_ends])
        - to_end
    )


def bad_bot_by_diff_old(x, fun_bad, i_en=None, speed=1):
    """
    old variant: fBotNansDiff = lambda(x,fun_bad,i_en):
    fPutNans(x, f_iBadBefore(x, fun_bad, i_en)                    , i_en)
    fPutNans(x, f_iBadBefore(diff(x)/speed[:-1], funBad, i_en, 50), i_en)
    where f_iBadBefore(arr,funBad,i_en,to_end) = int32([append(flatnonzero(funBad(arr[int32(En-to_end):int32(En)])), to_end)[0] for En in i_en]) - to_end
    """
    return put_nans(x, f_iBadBefore(np.diff(x) / speed[:-1], fun_bad, i_en), i_en)


def grid(z, y, grid_y, st, en, reverse: builtins.bool = False):
    return np.transpose([
        np.interp(grid_y, y[s:e], z[s:e], np.nan, np.nan) if np.any(z[s:e]) else np.nan + grid_y
        for s, e in (
            zip(np.int32(st)[::-1], np.int32(en)[::-1]) if reverse else zip(np.int32(st), np.int32(en))
        )
    ])


def i_whole_time(time, dt, dt_shift: int = 0):
    """
    :param time: numpy array of time values
    :param dt: units str to get f'M8[{dt}]' date type directly or seconds to find it
    :param dt_shift: shift, s
    :return: time array converted to dt units shifted on dt_shift
    """
    if not isinstance(dt, str):
        unit_sym, unit_dt = (("s", 1), ("m", 60), ("h", 3600), ("D", 24 * 3600))[
            np.searchsorted([60, 3600, 24 * 3600], dt)
        ]
        dt = f"{np.fmax(dt // unit_dt, 1)}{unit_sym}"
    return np.array(np.int64(time + dt_shift), "M8[s]").astype(f"M8[{dt}]")


def i_whole_time_intervals(time, dt, dt_shift=0, dt_burst=0):
    """
    Returns indexes where time values are nearest to whole and edges indexes,
    Makes 1st and last intervals > dt/2
    Note: NaNs near edges will lead to big intervals near start and finish. So delete NaNs before use
    :param time:
    :param dt:
    :param dt_shift: to find indexes of shifted time. 1st and last intervals lengths will be constrained relative to original time
    :param dt_burst: if not 0 then allow smaller 1st interval: makes it > dt_burst/2
    :return:
    """
    ind = np.flatnonzero(np.diff(np.int8(i_whole_time(time, dt, dt_shift)))) + 1
    try:
        _slice = slice(
            1 if 2 * np.subtract(*time[[ind[0], 0]]) < (dt_burst or dt) else 0,
            -1 if 2 * np.subtract(*time[[-1, ind[-1]]]) < dt else len(time),
        )
        return np.hstack([0, ind[_slice], len(time) - 1])
    except IndexError:
        return 0


def stretch_time(t, packet_st):
    """
    Stretch time values between packets starts
    last interval: to packet_st[-1]
    :param t: numpy array of time values
    :param packet_st: Sequence of packets starts time
    return: stretched time array
    """
    o = np.copy(t)
    inds_found_st = np.searchsorted(t, packet_st)
    inds_found_en = inds_found_st[1:] + 1
    inds_found_en[-1] = t.size
    for ind_s, ind_e, *se in zip(inds_found_st[:-1], inds_found_en, packet_st[:-1], packet_st[1:]):
        o[np.arange(ind_s, ind_e, dtype=np.int32)] = np.linspace(*se, ind_e - ind_s)
    return o


def i_closest(a, values):
    """
    a: numpy array
    values: should be sorted
    https://stackoverflow.com/a/46184652/2028147
    """
    # get insert positions
    idxs = np.searchsorted(a, values, side="left")

    # find indexes where previous index is closer
    prev_idx_is_less = (idxs == len(a)) | (
        np.fabs(values - a[np.maximum(idxs - 1, 0)]) < np.fabs(values - a[np.minimum(idxs, len(a) - 1)])
    )
    idxs[prev_idx_is_less] -= 1
    return idxs


def rose_table(v_abs, v_dir, rose_bins, nsectors=32):
    sector = 360.0 / nsectors
    angles = np.arange(-sector / 2, 360.0 + sector, sector, dtype=float)
    t = np.histogram2d(x=v_abs, y=v_dir % 360, bins=[rose_bins, angles], normed=False)[0]
    t2 = np.column_stack((np.nansum(t[:, [0, -1]], axis=1), t[:, 1:-1]))
    np.info(t)
    return np.flipud(np.cumsum(np.flipud(t2 * 100 / np.nansum(t2)), axis=0))


# For inclinometers


def rotate(r_from, r_to):
    """
    Rotation matrix to rotate 1st vector to second
    """
    if np.all(r_from == r_to):
        return np.eye(3)
    a = np.ravel(r_from)
    b = np.ravel(r_to)
    c = np.cross(a, b)
    scc_cross_ab = np.float64([[0, -c[2], c[1]], [c[2], 0, -c[0]], [-c[1], c[0], 0]])
    return np.eye(3) + scc_cross_ab + np.linalg.matrix_power(scc_cross_ab, 2) * (1 - a @ b) / np.sum(c**2)


def mag_dec(lat, lon, time_iso, depth=0):
    """
        Returns magnetic declination using wmm2020 library

    :param lat, lon: coordinates in degrees WGS84
    :param time_iso: # like '2020-09-20'
    :param depth: in km (negative below sea surface)
    """
    run_commands = f'''c:/Users/and0k/conda/Scripts/activate.bat py3.10x64h5togrid && python -c "from datetime import datetime; import wmm2020 as wmm; _year_fraction = lambda date: (lambda start: date.year + float(date.toordinal() - start) / (datetime(date.year + 1, 1, 1).toordinal() - start))(datetime(date.year, 1, 1).toordinal()); mag = wmm.wmm({lat}, {lon}, {depth}, _year_fraction(datetime.fromisoformat('{time_iso}'))); print(mag.decl.item(0))"'''
    decl_str = check_output(
        f"c:/Users/and0k/conda/Scripts/activate.bat py3.10x64h5togrid && {run_commands}",
        shell=True,
        text=True,
    )
    return float(decl_str)


def wrap_dir(x, disp_central_dir=180):
    """Wrap directions around central direction."""
    dir0 = disp_central_dir - 180
    return (x - dir0) % 360 + dir0


def wrap_dir_unwrap180(x, disp_central_dir, pass_over):
    s = wrap_dir(x, disp_central_dir)
    bs = np.abs(s - disp_central_dir) < 180 - pass_over
    return np.where(
        (np.abs(np.interp(np.arange(len(s)), np.flatnonzero(bs), s[bs]) - s) > 90) & ~bs,  # , period=360
        np.where(s < disp_central_dir, s + 360, s - 360),
        s,
    )


def norm_field(raw3d, coef_a2d, coef_c, raw3d_helps_recover=None):
    """

    :param raw3d: data
    :param coef_c: shift part of coef
    :param coef_a2d: multiplier part of coef
    If some diagonal element is zero, then corresponding channel data will be recovered
    :param raw3d_helps_recover: used if need to recover raw3d data channel by copying its sign
    # todo: get initial sign such that most of raw3d_helps_recover sign will match recovering channel data
    :return:
    """
    if coef_c.ndim < 2:
        coef_c = coef_c.reshape(-1, 1)
    # Apply coefs
    s = np.dot(coef_a2d, raw3d - coef_c)

    # If gain for some channel is zero
    i_ch_bad = np.flatnonzero(coef_a2d.diagonal() == 0)
    if i_ch_bad.size:
        # Recover channel
        i_ch_ok = [i for i in range(3) if i != i_ch_bad]
        s[i_ch_bad] = np.square(1 - (s[i_ch_ok] ** 2).sum(axis=0))
        if (s[i_ch_bad].imag != 0).any():
            s[i_ch_bad] = s[i_ch_bad].real

        # Sign of recovering channel
        if raw3d_helps_recover is not None:
            s[i_ch_bad] *= np.sign(raw3d_helps_recover[i_ch_bad])
            return s

        # Select sign in inverse gradient points so that we minimise changes of gradient
        s_dif = np.ediff1d(s[i_ch_bad], to_begin=0)
        # where gradient reverse
        b_reversed = s_dif < 0
        b_reversed &= np.append(~b_reversed[1:], False)
        # Keep reversed where diff(s_dif) of inverted signal will be less than of not inverted (where gradient reversed)
        # diff of reversed signal (after b_reversed points)
        _ = s[i_ch_bad, b_reversed] + s[i_ch_bad, np.roll(b_reversed, 1)]
        # diff in point before gradient reverse:
        s_dif_prev = s_dif[np.roll(b_reversed, -1)]
        # select sign of signal with minimum change of |gradient|
        b_reversed[b_reversed] = np.abs(s_dif_prev - _) < np.abs(s_dif_prev - s_dif[b_reversed])

        # Consequently invert signal in b_reversed points
        s_rev_sign = np.zeros_like(s_dif)
        s_rev_sign[0] = 1
        n_reversed = np.sum(b_reversed)
        s_rev_sign[b_reversed] = np.tile([-2, 2], int(np.ceil(n_reversed / 2)))[:n_reversed]
        s_rev_sign = np.cumsum(s_rev_sign)
        s[i_ch_bad] = s_rev_sign * s[i_ch_bad]
    return s


def xy_or_y(x, y, use_x_if=lambda x: np.bool(x), f_xy=operator.add):
    return f_xy(x, y) if (use_x_if(x) if callable(use_x_if) else use_x_if) else y


def xy_or_x(x, y, use_y_if=lambda y: np.bool(y), f_xy=operator.add):
    return f_xy(x, y) if (use_y_if(y) if callable(use_y_if) else use_y_if) else x


def xy_sel(x, y, use_x_if=lambda x: np.bool(x), use_y_if=lambda y: np.bool(y), f_xy=operator.add, nothing=""):
    use_x = use_x_if(x) if callable(use_x_if) else use_x_if
    use_y = use_y_if(y) if callable(use_y_if) else use_y_if
    if use_x:
        return f_xy(x, y) if use_y else x
    else:
        return y if use_y else nothing


# For CTD "zabor"


def cor_where_run_p(arr, dict_i_p, inds, pres, fun=lambda x: np.nan):
    """
    Apply function to array elements within specified pressure ranges for each run.

    Parameters:
    - arr: input array to modify
    - dict_i_p: {run#: [min_pressure, max_pressure]}
    - inds: array assigning each element of `arr` to a run number
    - pres: pressure values
    - fun: function to apply (default: set to nan)
    """
    ac = arr.copy()
    for i, (p_st, p_en) in dict_i_p.items():
        b = (inds == i) & (p_st < pres) & (pres < p_en)
        ac[b] = fun(arr[b])
    return ac


def where_run_p(arr, dict_i_p, inds, pres, fun=np.nanmean, val_for_fun_of_empty=None):
    """Same args as for cor_where_run_p()"""
    ac = []
    for i, (p_st, p_en) in dict_i_p.items():
        b = (inds == i) & (p_st < pres) & (pres < p_en)
        ac.append(fun(arr[b]) if b.any() else val_for_fun_of_empty)
    return ac


def zaborrunsselect(use_ranges, use_runs_in_used_range, runs_st, runs_lengths, data_indexes, time_shift_s=0):
    """
        Select specific data runs from multiple sources based on time ranges and run indices.
    :param use_ranges: Time ranges to use from joined source data
    :param use_runs_in_used_range: Run indices to select (supports int, negative indexing, ranges)
        Examples: [[0, -3], -2, -1] or [[0, None]] (None means all remaining)
    :param runs_st: Run start time indexes in joined source data
    :param runs_lengths: Length of each run
    :param data_indexes: Time index of joined source data
    :param time_shift_s: Time shift in seconds
    :return: 2D array with columns `[idatasel_st, idatasel_en, idata_st_n, idata_en, j_table]`
    where:
    - prefixes: `idatasel` - selected combined data, `idata` - raw data,
    - suffixes: `en` - runs ends, `st` - starts, `st_n` - next runs starts;
    - j_table: device table index.

    # Example run:
    import builtins; from logging import info, warning
    USEranges__ = [['2023-05-09T17:54:23', '2023-05-11T23:38:43']]
    USEruns_in_used_range = [[0, 5], -1]
    TimeShiftedFromUTC_s = 0
    l0index = GetData('l0index')[0]
    l0rows = GetData('l0rows')[0]
    l0rows_filtered = GetData('l0rows_filtered')[0]
    t = GetData('l0/CTD_SST_48Mc#1253/table/index')[0]
    sl = lambda x: slice(*int32(ravel(x)))
    #def i_ranges():...
    zabor_runs_edges(
        [USEranges__], USEruns_in_used_range,
        runs_st=(l0index,), runs_lengths=(l0rows + l0rows_filtered,), data_indexes=(t,), time_shift_s=TimeShiftedFromUTC_s
    )
    # or better:
     use_ranges, use_runs_in_used_range, runs_st, runs_lengths, data_indexes, time_shift_s = [
        USEranges__], USEruns_in_used_range, (l0index,), (l0rows + l0rows_filtered,), (t,), TimeShiftedFromUTC_s
    """
    # Runs' time ranges and indexes from log tables
    n_st_i = []
    for i, (r_st, r_n, tranges) in enumerate(zip(runs_st, runs_lengths, use_ranges)):
        i_log_use = i_ranges(r_st, tranges, time_shift_s - 10, t_units="ns")
        # next start:
        n_st_i.append([r_n[sl(i_log_use)], r_st[sl(i_log_use)], np.full(np.ediff1d(i_log_use).item(), i)])

    # device index, run' start time, run' end time, run' up end time (= next run start):
    n_st_j = np.vstack(n_st_i)
    # sort by run start time
    j_sort = np.argsort(n_st_j[1, :])

    # List of selected runs
    i_runs = [
        i
        for se in use_runs_in_used_range
        for i in (
            [se]
            if isinstance(se, int)
            else range(*[n_st_j.shape[1] if j is None else j + n_st_j.shape[1] if j < 0 else j for j in se])
        )
    ]
    # same effect expression: sum([[se] if isinstance(se, int) else
    # list(range(*[((i + n_st_j.shape[1]) if i < 0 else i) for i in se])) for se in use_runs_in_used_range])

    n_st_j_use = n_st_j[:, j_sort[i_runs]]
    # warning('zabor runs indexes selected: %s', repr(arange(n_st_j.shape[1])[j_sort[i_runs]]))
    j_use = n_st_j_use[-1, :]

    # Data runs indexes
    # - in raw data
    idata_st = np.hstack([
        np.searchsorted(data_indexes[int(j.item(0))], starts)
        for starts, j in np.hsplit(n_st_j_use[1:, :], np.flatnonzero(np.diff(j_use)) + 1)
    ])  # if i > 0 else n_st_j_use[1:, :].T
    idata_en = idata_st + n_st_j_use[0, :]

    # - in selected runs we will combine (except 1st = 0)
    idatasel_en = np.cumsum(n_st_j_use[0, :])
    idatasel_st = np.append(0, idatasel_en[:-1])
    out = np.column_stack([idatasel_st, idatasel_en, idata_st, idata_en, j_use])
    # warning('zabor_runs_edges() result has shape %s: %s', repr(out.shape), repr(out))
    return out


def _DS_(bit, part):
    """failed correction of Veusz bug of dimension query
    bit, part will be 'rose_table' 'data'
    """

    def f_out(*args):
        return
        # return 2D array cause fatal error:
        # globals()[bit](*args)
        # array([[0]])
        # [[]]

    return f_out


def load_cash(
    vars,
    dtypes=("M8[s]",),
    index="Time_UTC",
    file="~cash.csv",
    usecols=None,
    delimiter="\t",
    fun_out=lambda index, loaded_index, loaded: np.atleast_2d(
        loaded[np.isin(np.int32(loaded_index), np.int32(index))].view("f8")
    ),
):
    """

    :param vars:
    :param dtypes:
    :param index:
    :param file:
    :param usecols:
    :param fun_out:
    :return:
    """
    try:
        index_col, index_val = next(iter(index.items())) if isinstance(index, Mapping) else (index, None)
        n_formats_skipped = len(vars) - len(dtypes)
        if n_formats_skipped:
            dtypes = tuple(dtypes) + ("f8",) * n_formats_skipped
        data_exist = np.loadtxt(
            file,
            skiprows=1,
            usecols=usecols,
            delimiter=delimiter,
            dtype=np.dtype({
                "names": tuple(vars),
                "formats": ["f8" if dtyp.startswith("M") else dtyp for dtyp in dtypes],
            }),
            converters={
                i: lambda x: dt64s2vsz(np.datetime64(x, "s"))
                for i, dtyp in enumerate(dtypes)
                if dtyp.startswith("M")
            },
        )
    except:
        exception("Error in load_cash(%s))!", file)
        return
    if index_val is None:
        return data_exist
    return fun_out(index_val, data_exist[index_col], data_exist)


def parse_toon(toon_text):
    """
        Parse TOON format text into structured Python data.

        Args:
            toon_text: String containing TOON format data

        Returns:
            dict: Parsed data with column definitions and record arrays
        Note: the function is for this subformat:
    toon_data = '''@cols = {site,H,z,mod,lat,lon,t0,t1,bdt,bt}
    i10[1]@cols: "",,,?,58.1716,10.76,2017-06-19T15:00,2017-06-21T05:35,240,1800
    i11[1]@cols: "",,,A,59.0032,11.4201,2018-07-02T09:15,2018-07-03T18:40,300,1800
    i20[3]@cols: "",,,?,55.3083,15.6383,2017-06-25T08:00,2017-06-26T15:05,240,1800
    "",,,?,55.2152,17.0235,2017-06-28T11:33,2017-07-01T11:05,240,1800
    "",,,?,54.9910,18.1042,2017-07-03T10:20,2017-07-05T09:50,240'''
    """
    import re
    lines = toon_text.strip().split("\n")

    # Extract column definition
    cols_match = re.match("@cols\\s*=\\s*\\{([^}]+)\\}", lines[0])
    cols = [c.strip() for c in cols_match.group(1).split(",")]

    # Parse data records
    records = {}
    current_id = None
    current_data = []
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue

        # Check for record ID
        id_match = re.match("(\\w+)\\[\\d+\\]@cols:", line)
        if id_match:
            if current_id:
                records[current_id] = np.array(current_data, dtype=object)
            current_id = id_match.group(1)
            current_data = []
            line = line[id_match.end() :].strip()

        # Parse CSV row
        values = [v.strip().strip('"') or None for v in line.split(",")]
        current_data.append(values[: len(cols)])
    if current_id:
        records[current_id] = np.array(current_data, dtype=object)
    return {"columns": cols, "records": records}


def fmt_3_digits_after_dot(x):
    if isinstance(x, str):
        return x
    # elif isinstance(x, datetime):
    #     return x.strftime("%Y-%m-%d %H:%M:%S.%f")
    return ("%.3f" % x).rstrip("0").rstrip(".") if x > 0.1 else f"{x:.3g}"


def format_2d_array(data, formatters=()):
    """
    Function to format a 2D numpy array with a list of formatter functions
    data: numpy array or list of columns to write
    """
    b_array = isinstance(data, np.ndarray)
    n_elements = (
        ((data.shape[1] if data.ndim > 1 else 1) if data.dtype.names is None else len(data.dtype.names))
        if b_array
        else len(data)
    )
    if len(formatters) < n_elements:
        formatters = list(formatters)
        formatters += [fmt_3_digits_after_dot] * (n_elements - len(formatters))
    elif len(formatters) != n_elements:
        raise ValueError("Number of formatters must be <= number of columns.")
    formatted_data = []
    prev_formatter = None
    # Apply each formatter to its corresponding column
    for i, formatter in enumerate(formatters):
        data_cur = data[:, i] if b_array else data[i]
        # application of the formatter
        if prev_formatter != formatter:
            if isinstance(formatter, str):
                formatter = f"{{:{formatter}}}".format
                if isinstance(data_cur[0], np.datetime64):
                    fmatter = formatter
                    formatter = lambda el: fmatter(el.item())
        try:
            formatted_data.append([formatter(el) for el in data_cur])
        except Exception:
            raise ValueError(
                "Error apply formatting to {} at col{}: {}!".format(
                    data_cur,
                    i,
                    f"{formatter}.format" if isinstance(formatter, str) else formatter
                )
            )

        prev_formatter = formatter
    return np.array(formatted_data).T


def save2text(
    vars,
    dtypes=(),
    formats=(),
    delimiter="\t",
    file=None,
    file_sfx="_out.tsv",
    skip_if_exist=True,
    fun_get=lambda x: x,  # : Callable[[Any], Any]
    fun_before_compare=np.int32,
):
    """
    Save data to a text file.

    :param vars: variables to save. Must be of type for which `list(vars)` returns Sequnce[str] of var names
    - Mapping[str, Any]: column names to data values. Ensure the values are of correct types
    - Sequence[str]: values will be obtained with `fun_get()` of each var - useful if csv headers matches existed names
    :param dtypes: input data types for the `vars`. If their number is less than `vars` then remained are float64.
    Default: empty tuple. Use any numpy array types: "M8[s]", "f8'...
    :param formats: output format specifiers for the `vars` - standard ("f", "s", ...) or custom formatting functions.
    :param delimiter: delimiter for the output file.
    :param file: file path to save the data. Default: `sys.argv[1]` with suffix replaced with `file_sfx`
    :param file_sfx: will be appended to basename to get output file name if `file` is `None`
    :param skip_if_exist: behavior when the output file exists:
    - True: does nothing if output file exists.
    - column_name: **append** file if fun_get(f'~{skip_if_exist}{file_stem}') not contains
      val=fun_get(skip_if_exist), with header if former is falsy. Assume fun_get is a Veusz DATA() GUI
      function else skip_if_exist should be a one item dict: {column_name: val}
    - False: overwrite
    :param fun_get: function used for each `var` to retrieve its data.
    :param fun_before_compare: function to preprocess data before comparison. Skip saving cur data is in
    previously saved file data
    :return: number of data rows saved.

    Example
    save2text(
        ('bin2_P__','bin2_Vabs__','bin2_Vdir__'),
        'bin2_t0st__', '	',
        lambda x: (warning(x), DATA(x))[1]
    )
    """
    if file is None:
        parent, basename = (lambda p: (p.parent, p.name))(Path(sys.argv[1]))
        file = parent / (basename.rsplit(".")[0] + file_sfx)
    else:
        file = Path(file)
    data_exist = False

    n_formats_skipped = len(vars) - len(dtypes)
    if n_formats_skipped:
        dtypes = tuple(dtypes) + ("f8",) * n_formats_skipped

    def from_vsz(v, fmt):
        return vsz2dt64s(fun_get(v)) if fmt.startswith("M") else fun_get(v)

    if skip_if_exist:
        if isinstance(skip_if_exist, builtins.bool):
            if file.is_file():
                info(f"skipping saving to existed file {file}...")
                return -1  # file exist
        elif file.is_file():
            file_stem = file.stem
            if isinstance(skip_if_exist, dict):
                skip_if_exist, data_exist = next(iter(skip_if_exist.items()))
            else:
                data_exist = None
            var_name = skip_if_exist.strip()
            try:
                # Get data corresponding to `skip_if_exist` column header
                # 1. data that we are going to write
                if skip_if_exist in vars:
                    with contextlib.suppress(TypeError):  # may already have `var_name` right
                        var_name = vars[skip_if_exist]
                data_new = fun_get(var_name)
                if data_exist == "loadtxt":
                    var_icol = tuple(vars).index(skip_if_exist)
                    dtyp_col = dtypes[var_icol]
                    data_exist = np.loadtxt(
                        file, skiprows=1, dtype=dtyp_col, usecols=[var_icol], delimiter=delimiter
                    )
                    if dtyp_col.startswith("M"):
                        data_exist = dt64s2vsz(data_exist)
                elif data_exist is None:
                    data_exist = fun_get(f"~{var_name}{file_stem}")
                data_new_cmp = fun_before_compare(data_new)
                data_exist_cmp = fun_before_compare(data_exist)
                b_new = ~np.isin(data_new_cmp, data_exist_cmp, assume_unique=True)
                if not b_new.any():
                    return -2
                else:
                    warning(
                        f"{data_new}->{data_new_cmp}: not in {data_exist}->{data_exist_cmp}! Saving%s...",
                        "" if b_new.all() else f" {np.flatnonzero(b_new)} items",
                    )
            except Exception:
                data_exist = False
                exception("Error in save2text(%s))!", file.name)
                warning(f"~{var_name}{file_stem}: {data_exist}")
                warning(f"column_name: {fun_get(var_name)}")
                return
    warning(f"Saving {vars} to {file}:\ndtypes={dtypes}, type: {type(dtypes)}")
    # 1 get values
    vars_vals = (
        vars.values()
        if isinstance(vars, dict)
        else [from_vsz(v, fmt) for v, fmt in zip(vars, dtypes)]
        if fun_get
        else vars
    )
    el_size = next(iter(vars_vals)).size
    try:
        # 2. broadcast scalars
        val_arr = [np.repeat(v, el_size) if np.isscalar(v) else v for v, fmt in zip(vars_vals, dtypes)]
        n_formats_skipped = len(vars) - len(formats)
        if len(formats) < len(dtypes):
            formats = list(formats) + [
                fmt_3_digits_after_dot if dtyp.startswith("f") else "s"
                for dtyp in dtypes[-n_formats_skipped:]
            ]
        with file.open(mode="a" if skip_if_exist else "w") as f:
            if data_exist is False:
                np.savetxt(f, np.atleast_2d(list(vars)), "%s", delimiter=delimiter)
            str_array = format_2d_array(val_arr, formatters=formats)
            warning(f"formats: {formats},\narray(str): {str_array}\n")
            np.savetxt(f, str_array, fmt="%s", delimiter=delimiter)
        warning("File %s: %s", "saved" if data_exist is False else "appended", str(file))
    except Exception:
        exception("Error in save2text(%s)!", file.name)
        warning(f"Creating record array from: {val_arr} and saving is failed!\nTypes: {dtypes}")
        warning(f"{fun_get}(vars={vars}) gives: {[fun_get(n) for n in vars_vals]}")
        return
    return str_array.shape[1]


def nc_var_scale_offcet(file_nc, var_path: str):
    """
    :param file_nc: _description_
    :param var_path: /path/to/variable
    :return: _description_
    """
    import h5py

    with h5py.File(file_nc, "r") as f:
        var = f[var_path]
        scale = var.attrs.get("scale_factor")
        offset = var.attrs.get("add_offset")
    return (scale, offset)


def dx_dy_dist_bearing(lon1, lat1, lon2, lat2):
    """
    Distance and bearing between two points
    :param lon1: degrees, 1st point "lon"-coordinate(s)
    :param lat1: degrees, 1st point "lat"-coordinate(s)
    :param lon2: degrees, 2nd point "lon"-coordinate(s)
    :param lat2: degrees, 2nd point "lat"-coordinate(s)
    :return: array with 4 columns:(s)
     - dx: m, distance along "lon" coord line
     - dy: m, distance along "lat" coord line
     - dist: m, distance between points
     - bearing: degrees, in the range ``[-180, 180]``
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    # radius of the earth in m
    R = 6371000
    klon = np.cos((lat2 + lat1) / 2)
    dx = R * klon * dlon
    dy = R * dlat
    d = np.hypot(dx, dy)
    angle = np.arctan2(dx, dy)
    # or use atan2[(sin Δλ ⋅ cos φ₂), (cos φ₁ ⋅ sin φ₂ − sin φ₁ ⋅ cos φ₂ ⋅ cos Δλ)] from https://www.omnicalculator.com/other/azimuth
    return np.column_stack((dx, dy, d, np.degrees(angle)))


def skewness(data):
    n = len(data)
    mean_ = np.mean(data)
    std_ = np.std(data, ddof=1)  # Set ddof=1 for sample standard deviation
    third_moment = ((data - mean_) ** 3).sum() / n
    out = third_moment / std_**3
    return out


def kurtosis(data):
    """
    Excess kurtosis (where the kurtosis of the normal distribution is zero).
    If you want the Fisher kurtosis (which would be three for a normal distribution), you can simply remove the - 3
    """
    n = len(data)
    mean_ = np.mean(data)
    std_ = np.std(data, ddof=1)
    fourth_moment = ((data - mean_) ** 4).sum() / n
    out = fourth_moment / std_**4 - 3  # Subtract 3 for excess kurtosis
    return out


### Functions copied from https://pypi.org/project/seawater/3.3/ to calculate potential density (sw_pden()) ###
###############################################################################################################

T68conv = lambda t90: t90 * 1.00024
T90conv = lambda t68: t68 / 1.00024


def ptmp(s, t, p, pr=0):
    """Calculates potential temperature as per UNESCO 1983 report.

    Parameters
    ----------
    s(p) : array_like
           salinity [psu (PSS-78)]
    t(p) : array_like
           temperature [:math:°C (ITS-90)]
    p : array_like
        pressure [db].
    pr : array_like
        reference pressure [db], default = 0

    Returns
    -------
    pt : array_like
         potential temperature relative to PR [:math:°C (ITS-90)]

    Examples
    --------
    >>> import seawater as sw
    >>> t = T90conv([[0, 0, 0, 0, 0, 0],
    ...              [10, 10, 10, 10, 10, 10],
    ...              [20, 20, 20, 20, 20, 20],
    ...              [30, 30, 30, 30, 30, 30],
    ...              [40, 40, 40, 40, 40, 40]])
    >>> s = [[25, 25, 25, 35, 35, 35],
    ...      [25, 25, 25, 35, 35, 35],
    ...      [25, 25, 25, 35, 35, 35],
    ...      [25, 25, 25, 35, 35, 35],
    ...      [25, 25, 25, 35, 35, 35]]
    >>> p = [0, 5000, 10000, 0, 5000, 10000]
    >>> sw.T68conv(sw.ptmp(s, t, p, pr=0))
    array([[  0.        ,  -0.30614418,  -0.96669485,   0.        ,
             -0.3855565 ,  -1.09741136],
           [ 10.        ,   9.35306331,   8.46840949,  10.        ,
              9.29063461,   8.36425752],
           [ 20.        ,  19.04376281,  17.94265   ,  20.        ,
             18.99845171,  17.86536441],
           [ 30.        ,  28.75124632,  27.43529911,  30.        ,
             28.72313484,  27.38506197],
           [ 40.        ,  38.46068173,  36.92544552,  40.        ,
             38.44979906,  36.90231661]])

    References
    ----------
    .. [1] Fofonoff, P. and Millard, R.C. Jr UNESCO 1983. Algorithms for
    computation of fundamental properties of seawater. UNESCO Tech. Pap. in
    Mar. Sci., No. 44, 53 pp.  Eqn.(31) p.39.
    http://unesdoc.unesco.org/images/0005/000598/059832eb.pdf

    .. [2] Bryden, H. 1973. New Polynomials for thermal expansion, adiabatic
    temperature gradient and potential temperature of sea water.
    Deep-Sea Res. Vol20,401-408. doi:10.1016/0011-7471(73)90063-6

    Modifications: 92-04-06. Phil Morgan.
                   99-06-25. Lindsay Pender, Fixed transpose of row vectors.
                   03-12-12. Lindsay Pender, Converted to ITS-90.
    """

    def adtg(s, t, p):
        """Calculates adiabatic temperature gradient as per UNESCO 1983 routines.
        References
        ----------
        .. [1] Fofonoff, P. and Millard, R.C. Jr UNESCO 1983. Algorithms for
        computation of fundamental properties of seawater. UNESCO Tech. Pap. in
        Mar. Sci., No. 44, 53 pp.
        http://unesdoc.unesco.org/images/0005/000598/059832eb.pdf

        .. [2] Bryden, H. 1973. New Polynomials for thermal expansion, adiabatic
        temperature gradient and potential temperature of sea water. Deep-Sea Res.
        Vol20,401-408. doi:10.1016/0011-7471(73)90063-6

        Modifications: 93-04-22. Phil Morgan.
                       99-06-25. Lindsay Pender, Fixed transpose of row vectors.
                       03-12-12. Lindsay Pender, Converted to ITS-90.
        """
        T68 = T68conv(t)
        a = [3.5803e-05, 8.5258e-06, -6.836e-08, 6.6228e-10]
        b = [1.8932e-06, -4.2393e-08]
        c = [1.8741e-08, -6.7795e-10, 8.733e-12, -5.4481e-14]
        d = [-1.1351e-10, 2.7759e-12]
        e = [-4.6206e-13, 1.8676e-14, -2.1687e-16]
        return (
            a[0]
            + (a[1] + (a[2] + a[3] * T68) * T68) * T68
            + (b[0] + b[1] * T68) * (s - 35)
            + (c[0] + (c[1] + (c[2] + c[3] * T68) * T68) * T68 + (d[0] + d[1] * T68) * (s - 35)) * p
            + (e[0] + (e[1] + e[2] * T68) * T68) * p * p
        )

    # Theta1.
    del_P = pr - p
    del_th = del_P * adtg(s, t, p)
    th = T68conv(t) + 0.5 * del_th
    q = del_th
    sqrt2 = np.sqrt(2)

    # Theta2.
    del_th = del_P * adtg(s, T90conv(th), p + 0.5 * del_P)
    th = th + (1 - 1 / sqrt2) * (del_th - q)
    q = (2 - sqrt2) * del_th + (-2 + 3 / sqrt2) * q

    # Theta3.
    del_th = del_P * adtg(s, T90conv(th), p + 0.5 * del_P)
    th = th + (1 + 1 / sqrt2) * (del_th - q)
    q = (2 + sqrt2) * del_th + (-2 - 3 / sqrt2) * q

    # Theta4.
    del_th = del_P * adtg(s, T90conv(th), p + del_P)
    return T90conv(th + (del_th - 2 * q) / 6)


def dens0(s, t):
    """Density of Sea Water at atmospheric pressure.

    Parameters
    ----------
    s(p=0) : array_like
             salinity [psu (PSS-78)]
    t(p=0) : array_like
             temperature [:math:°C (ITS-90)]

    Returns
    -------
    dens0(s, t) : array_like
                  density  [kg m :sup:`3`] of salt water with properties
                  (s, t, p=0) 0 db gauge pressure

    Examples
    --------
    Data from UNESCO Tech. Paper in Marine Sci. No. 44, p22
    >>> import seawater as sw
    >>> s = [0, 0, 0, 0, 35, 35, 35, 35]
    >>> t = T90conv([0, 0, 30, 30, 0, 0, 30, 30])
    >>> sw.dens0(s, t)
    array([  999.842594  ,   999.842594  ,   995.65113374,   995.65113374,
            1028.10633141,  1028.10633141,  1021.72863949,  1021.72863949])

    References
    ----------
    .. [1] Fofonoff, P. and Millard, R.C. Jr UNESCO 1983. Algorithms for
    computation of fundamental properties of seawater. UNESCO Tech. Pap. in
    Mar. Sci., No. 44, 53 pp.  Eqn.(31) p.39.
    http://unesdoc.unesco.org/images/0005/000598/059832eb.pdf

    .. [2] Millero, F.J. and  Poisson, A. International one-atmosphere equation
    of state of seawater. Deep-Sea Res. 1981. Vol28A(6) pp625-629.
    doi:10.1016/0198-0149(81)90122-9

    Modifications: 92-11-05. Phil Morgan.
                   03-12-12. Lindsay Pender, Converted to ITS-90.
    """
    T68 = T68conv(t)

    def smow():
        """Density of Standard Mean Ocean Water (Pure Water) using EOS 1980."""
        a = (999.842594, 0.06793952, -0.00909529, 0.0001001685, -1.120083e-06, 6.536332e-09)
        return a[0] + (a[1] + (a[2] + (a[3] + (a[4] + a[5] * T68) * T68) * T68) * T68) * T68

    # UNESCO 1983 Eqn.(13) p17.
    b = (0.824493, -0.0040899, 7.6438e-05, -8.2467e-07, 5.3875e-09)
    c = (-0.00572466, 0.00010227, -1.6546e-06)
    d = 0.00048314
    return (
        smow()
        + (b[0] + (b[1] + (b[2] + (b[3] + b[4] * T68) * T68) * T68) * T68) * s
        + (c[0] + (c[1] + c[2] * T68) * T68) * s * np.sqrt(s)
        + d * s**2
    )


def dens(s, t, p):
    """Density of Sea Water using UNESCO 1983 (EOS 80) polynomial.

    Parameters
    ----------
    s(p) : array_like
           salinity [psu (PSS-78)]
    t(p) : array_like
           temperature [:math:°C (ITS-90)]
    p : array_like
        pressure [db].

    Returns
    -------
    dens : array_like
           density  [kg m :sup:`3`]

    Examples
    --------
    Data from Unesco Tech. Paper in Marine Sci. No. 44, p22.
    >>> import seawater as sw
    >>> s = [0, 0, 0, 0, 35, 35, 35, 35]
    >>> t = T90conv([0, 0, 30, 30, 0, 0, 30, 30])
    >>> p = [0, 10000, 0, 10000, 0, 10000, 0, 10000]
    >>> sw.dens(s, t, p)
    array([  999.842594  ,  1045.33710972,   995.65113374,  1036.03148891,
            1028.10633141,  1070.95838408,  1021.72863949,  1060.55058771])

    References
    ----------
    .. [1] Fofonoff, P. and Millard, R.C. Jr UNESCO 1983. Algorithms for
    computation of fundamental properties of seawater. UNESCO Tech. Pap. in
    Mar. Sci., No. 44, 53 pp.  Eqn.(31) p.39.
    http://unesdoc.unesco.org/images/0005/000598/059832eb.pdf

    .. [2] Millero, F.J., Chen, C.T., Bradshaw, A., and Schleicher, K. A new
    high pressure equation of state for seawater. Deap-Sea Research., 1980,
    Vol27A, pp255-264. doi:10.1016/0198-0149(80)90016-3

    Modifications: 92-11-05. Phil Morgan.
                   99-06-25. Lindsay Pender, Fixed transpose of row vectors.
                   03-12-12. Lindsay Pender, Converted to ITS-90.
    """
    T68 = T68conv(t)

    def seck(s, p=0):
        """Secant Bulk Modulus (K) of Sea Water using Equation of state 1980.
        UNESCO polynomial implementation.
        """
        # Compute compression terms.
        p = p / 10.0  # Convert from db to atmospheric pressure units.

        # Pure water terms of the secant bulk modulus at atmos pressure.
        # UNESCO Eqn 19 p 18.
        h = [3.239908, 0.00143713, 0.000116092, -5.77905e-07]
        AW = h[0] + (h[1] + (h[2] + h[3] * T68) * T68) * T68


        k = [8.50935e-05, -6.12293e-06, 5.2787e-08]
        BW = k[0] + (k[1] + k[2] * T68) * T68
        e = [19652.21, 148.4206, -2.327105, 0.01360477, -5.155288e-05]
        KW = e[0] + (e[1] + (e[2] + (e[3] + e[4] * T68) * T68) * T68) * T68

        # Sea water terms of secant bulk modulus at atmos. pressure.
        j0 = 0.000191075
        i = [0.0022838, -1.0981e-05, -1.6078e-06]
        sqrt_s = np.sqrt(s)
        A = AW + (i[0] + (i[1] + i[2] * T68) * T68 + j0 * sqrt_s) * s
        m = [-9.9348e-07, 2.0816e-08, 9.1697e-10]
        B = BW + (m[0] + (m[1] + m[2] * T68) * T68) * s  # Eqn 18.
        f = [54.6746, -0.603459, 0.0109987, -6.167e-05]
        g = [0.07944, 0.016483, -0.00053009]
        K0 = (
            KW
            + (f[0] + (f[1] + (f[2] + f[3] * T68) * T68) * T68 + (g[0] + (g[1] + g[2] * T68) * T68) * sqrt_s)
            * s
        )  # Eqn 16.
        return K0 + (A + B * p) * p  # Eqn 15.`

    # UNESCO 1983. Eqn..7  p.15.
    densP0 = dens0(s, t)
    K = seck(s, p)
    p = p / 10.0  # Convert from db to atm pressure units.
    return densP0 / (1 - p / K)


def sw_pden(s, t90, p, pr):
    return dens(s, ptmp(s, t90, p, pr), pr)

### Experiments ###

# Solubility bad functions (todo: correct or delete)


def oxygen_solubility(t, S):
    """
    Alternative: O2sol or O2sol_SP_pt
    Oxygen solubility after Garcia and Gordon (1992)
    This function is an implementation of the Computation of Oxygen Solubility equation, as specified in Seabird Application Note 64, Appendix A.
    :param t: temperature, degrees celsius
    :param S: Salinity, PSU
    :return oxsol: Oxygen solubility
    """
    a = [3.88767, -0.256847, 4.94457, 4.0501, 3.22014, 2.00907]
    b = [-0.00817083, -0.010341, -0.00737614, -0.00624523]
    c0 = -4.88682e-07
    Ts = np.log((298.15 - t) / (273.15 + t))
    oxsol = np.exp(np.polyval(a, Ts) + S * np.polyval(b, Ts) + c0 * S**2)
    return oxsol


def oxygen_solubility_scor(t, S, P=0, p_atm=1013.25):
    """
    Oxygen solubility according to recommendations by SCOR WG 142 "Quality Control Procedures
    for Oxygen and Other Biogeochemical Sensors on Floats and Gliders"

    :param t: temperature in °C
    :param S: salinity (PSS-78)
    :param P: hydrostatic pressure in dBar (default: 0 dBar)
    :param p_atm: atmospheric (air) pressure in mBar (default: 1013.25 mBar)
    :return: Oxygen solubility in µmol/L
    Note: to convert to mg/L multiply to the molar weight of O2: 0.0319988 mg/µmol (31.9988 g/mol (CIAAW 2015))

    From Matlab function O2conc=O2stoO2c(O2sat,t,S,P,p_atm) of
    convert oxygen saturation (O2sat: oxygen saturation in %) to
    molar oxygen concentration (in umol L-1):
    O2conc = O2sat / oxygen_solubility_scor(...)
    by Henry Bittig (Laboratoire d'Ocйanographie de Villefranche-sur-Mer, France
    bittig@obs-vlfr.fr 28.10.2015, 19.04.2018: v1.1, fixed typo in B2 exponent)
    22.11.2022
    """

    # Scaled temperature
    t_k = t + 273.15
    # for use in TCorr and SCorr
    t_sca = np.log((298.15 - t) / t_k)

    # Saturated water vapor in mBar
    pH2Osat = 1013.25 * np.exp(24.4543 - 67.4509 * (100 / t_k) - 4.8489 * np.log(t_k / 100) - 0.000544 * S)

    # Temperature correction part from Garcia and Gordon (1992), Benson and Krause (1984) refit mL(STP) L-1; and conversion from mL(STP) L-1 to umol L-1
    TCorr = 44.6596 * np.exp(np.polyval([3.88767, -0.256847, 4.94457, 4.0501, 3.22014, 2.00907], t_sca))
    # Salinity correction part from Garcia and Gordon (1992), Benson and Krause (1984) refit ml(STP) L-1
    Scorr = np.exp(
        S * np.polyval([-0.00817083, -0.010341, -0.00737614, -0.00624523], t_sca) - 4.88682e-07 * S**2
    )
    # Molar volume of O2 in m3 mol-1 Pa dBar-1 (Enns et al. 1965)
    Vm = 0.317
    # Universal gas constant in J mol-1 K-1
    R = 8.314
    return 100 * (TCorr * Scorr) * (p_atm - pH2Osat) / (1013.25 - pH2Osat) / np.exp(Vm * P / (R * t_k))
