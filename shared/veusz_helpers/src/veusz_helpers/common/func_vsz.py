"""
Note: Annotations not used as not supported in Veusz 3.2
"""
import builtins
import contextlib
from numpy import *
from logging import info, warning, exception
import operator
from itertools import zip_longest
from typing import Mapping  # Any, Callable, Union
from subprocess import check_output

import sys
from pathlib import Path

from winreg import HKEY_CURRENT_USER, OpenKey, QueryValueEx
with OpenKey(HKEY_CURRENT_USER, r"SOFTWARE\veusz.org\veusz") as _:
    lang = 'ru' if QueryValueEx(_, "ui_english")[0] == 'False' else 'en'  # or ast.literal_eval

# from importlib import util
# from itertools import dropwhile
# from functools import lru_cache
# @lru_cache
# def import_file(path, module_name):
#     fpy = lambda p: (p / module_name).with_suffix('.py')
#     file_py = fpy(next(dropwhile(lambda p: not fpy(p).is_file(), path.parents)))
#     spec = util.spec_from_file_location(module_name, file_py)
#     mod = util.module_from_spec(spec)
#     spec.loader.exec_module(mod)
#     warning(f'Loading {mod}')
#     return mod

def fself(fun, a, *args):
    # Useful when need multi line function that is not exist here and you don't want create one here
    fun(a, *args)
    return a


def sl(x) -> slice:
    """
    Slice
    :param x:
    - if number of x elements > 3 takes 1st and last only. Todo: course the indexing to iterate through pairs: possible?
    - ravel x if elements < 2 dimensions
    """
    args = [(None if isnan(i) else int(i)) for i in ravel(x)]
    try:
        return slice(*args)  # slice(*int32(ravel(x)))   # or lambda *x
    except TypeError:
        return slice(args[0], args[-1])


# Conversion date from datetime64 to Veusz and back:
dt64s2vsz = lambda dt64: float64(dt64) - 1230768000  # 1230768000=int32(datetime64('2009-01-01T00:00:00'))")
vsz2dt64s = lambda t_vsz: array(int32(t_vsz) + 1230768000, 'M8[s]')


def try_(fun, *args, **kwargs):
    try:
        retutn(fun(*args, **kwargs))
    except e:
        warning(e)


def c1(s):
    """Capitalize 1st letter only"""
    return f'{s[0].upper()}{s[1:]}'


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
    "after": "после",
    "along": "вдоль",
    "and": "и",
    "anchor": "якорь",
    "azimuth": "азимут",
    "at bot.": "у дна",  # not works, so any points are
    "at bottom": "у дна",
    "at": "на",
    "atmospheric": "атмосферное",
    "avg. bin": "ячейка уср.",  # not works with points
    "averaging bin": "ячейка усреднения",
    "avg": "уср",
    "band": "полоса",
    "band-pass": "полосовой",
    "bin": "ячейка",
    "blow to": "куда дует",
    "blow from": "откуда дует",
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
    "cm": "см",
    "component": "составляющая",
    "concentration": "концентрация",
    "correspondingly": "соответственно",
    "corrected": "исправленный",
    "counts": "отсчеты",
    "current": "течение",
    "current rose": "роза течений",
    "current velocity": "скорость течения",
    "dBar": "дБар",
    "data": "данные",
    "depth": "глубина",
    "device": "прибор",
    "dev": "пр",
    "direction": "направление",
    "displacement": "смещение",
    "downcast": "опускания",
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
    "magnitude": "абсолютное значение",
    "magnetometer": "магнитометр",
    "mean": "среднее",
    "measured": "измеренный",
    "measuredж": "измеренная",
    "measured by inclinometerж": "измеренная инклинометром",
    "measured by inclinometersж": "измеренная инклинометрами",
    "measured by tilt current meterж": "измеренная инклинометром",
    "measured by tilt current metersж": "измеренная инклинометрами",
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
    "normalized": "нормированные",
    "north": "северный",
    "notation": "обозначение",
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
    "pos": "полож",
    "power spectrum density": "спектральная плотность мощности",
    "progressive vector diagram": "годограф",
    "pressure": "давление",
    "pressure force": "сила давления",
    "psu": "епс",
    "reanalysis": "реанализ",
    "relative to": "относительно",
    "residual error": "остаточная погрешность",
    "run": "пуск",
    "s": "c",
    "salinity": "соленость",
    "sampling": "съем данных",
    "sat": "нас",
    "seaward": "со стороны моря",
    # "sea bed": "дно моря",
    "sea depth": "глубина",
    "sedimentary trap": "седиментационная ловушка",
    "sensor": "датчик",
    # "sea surface": "поверхность моря",
    "smoothed": "сглажено",
    "std": "CKO",
    "shore": "берег",
    "shoreward": "со стороны берега",
    "speed": "скорость",
    "st": "ст",
    "temperature": "температура",
    "temperature sensor": "датчик температуры",
    "temperature sensor chain": "термокоса",
    "then": "затем",
    "tracker": "трекер",
    "tilt current meter": "инклинометрический измеритель",
    "time": "время",
    "time resolution": "временное разрешение",
    "to the shore": "берегу",
    "total precipitation": "сумма осадков",
    "units": "ед.",
    "useful": "нужный",
    "velocity": "скорость",
    "velocity magnitude": "модуль скорости",
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
# 'down':'вниз', 'up':'вверх'


class DictKeyIfNoVal(dict,):
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

        b_translate = builtins.bool(self)  # if not self some clearing still may be needed
        if b_translate:
            out = self.get(key)
            if out is not None:
                # Simple translation was successful
                return out

        # Check 1st character
        char0 = key[0]
        if 0x0400 <= ord(char0) <= 0x04FF:
            # if in range of Cyrillic characters then keep key as is if Ru else delete
            return key if b_translate else ''
        elif char0 == '_':
            return (
                " ".join(self.get(k.replace("_", " "), k) for k in reversed(key[1:].split()))
                if b_translate
                else key.replace("_", " ")
            )
        elif char0.isupper():
            out = self.get(key.lower())
            if out:
                return c1(out)

        # Check/replace last characters
        i = -1
        char = key[i]
        if char == '_':  # replace in `self` last '_' with ' '
            return '' if b_translate else key[:i] + ' '
        elif char.isdigit() or char == 'ж':
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
            en_word = key[:i]
            if not self:
                return en_word

            # Translation
            out = self.get(en_word)
            # print(en_word, i, out, i_remove)
            if out is not None:
                # Simple translation is successful
                return f"{out[:(i_remove or None)]}{chars_add}"

            if en_word[-1] == 's':
                out = self.get(en_word[:-1])
                # print(en_word, out, pl(out), i_remove)
                return f"{pl(out)[:(i_remove or None)]}{chars_add}"

        elif char == "s":  # translate without last "s" then make plural
            out = self.get(key[:-1])
            if out:
                return plru(out)

            # if chars == 'ая':
            #     en_word = key[:-2]
            #     return f"{self.get(en_word, en_word)[:-2]}{chars}" if b_translate else en_word
            # en_word = key[:-1]
            # return f"{self.get(en_word, en_word)[:-1]}{char}" if b_translate else en_word

        return key  # Fallback to Key


I = DictKeyIfNoVal(en2ru if lang == 'ru' else {})

def plru(text):
    "Make russian word `text` to plural"
    if len(text) <= 1:
        return text
    before, last = text[:-1], text[-1]
    match last:
        # last 1 chars dependance
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
            before, last = text[:-2], text[-2:]
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
                    return f"{before}{last}".replace("на глубины", "на глубинах", 1)  # todo: find where used
                case "ом":
                    last = "ами"
                # case "ма":
                #     last = "мы"
                case _:
                    return text
    return f"{before}{last}"


def pl(text, lang=lang, add_s=True, split=False):  # , n
    """
    plural
    :param text:
    :param lang:
    :param add_s: switch mode "each word" / "one word"
    - if True then for each word:
        - if lang is not "ru": adds "s" to each word of length > 1
        - else tries replace known Russian suffixes to plural
    - else replaces "{s}" with "s", then if lang is "ru" replaces known Russian suffixes + "s" to plural
    Russian suffixes
    :return:
    """
    is_ru = lang and lang != 'en'  # ru
    if add_s:
        if is_ru:
            text_out = ' '.join(plru(t) for t in text.split(split)) if split else plru(text)
        else:
            text_out = ' '.join(f'{t}s' for t in text.split(split)) if split else f'{text}s'
    else:
        text_out = text.format(s='s')
        if is_ru:  # plural ru
            text_out = (
                text_out.format_map(I).replace('иеs', 'ия').replace('ийs', 'ие').replace('льs', 'ли')
                .replace('фs', 'фы').replace('цs', 'цы')
                )
    return text_out


# Display label of recommended time Units (You need manually set axis properties in accordance to it)
def str_date_unit_fmt(dt, next_fmt, lang=lang):
    return ([
        ('мм:сс', ':%Y-%m-%d %H:00'),
        ('Время', ':%Y-%m-%d'),
        ('День, время', ':%Y-%m'),
        ('День', ':%Y-%m'),
        ('Месяц/день', ':%Y'),
        ('Месяц', ':%Y')
    ] if lang == 'ru' else [
        ('MM:SS', ':%Y-%m-%d %H:00'),
        ('Time', ':%Y-%m-%d'),
        ('Day, time', ':%Y-%m'),
        ('Day', ':%Y-%m'),
        ('Month/day', ':%Y'),
        ('Month', ':%Y')
    ])[fmax(searchsorted(int32([70, 50*60, 240*60, 1200*60, 2.5*525600]), int(dt)//60) + next_fmt, 0)]  # min
    # this comment may not be in sync: 70min, ~2D, 10D, 50D, 2.5*Y


def str_date_unit(t_se_vsz, lang=lang, **kwargs):
    """
    kwargs:
    - next_fmt:
    - lang: if 'ru' then русские else English unit names. Default: 'en'
    """
    t_se = vsz2dt64s(array(t_se_vsz) + [1, -1]).tolist()
    # - adds 1s shifts is useful for unit intervals starting from round unit make it the same for start and end:
    # this then will be detected and kept only one

    dt = int(diff(t_se_vsz))
    unit, fmt = str_date_unit_fmt(dt, kwargs.get('next_fmt', 0), lang)
    st, en = [f'{{{fmt}}}'.format(t) for t in t_se]
    unit_minutes = unit.startswith(('MM', 'мм'))
    if st == en or unit_minutes:
        s = st
    else:
        idiff = [i for i, (left, right) in enumerate(zip(st, en)) if left != right][0]
        s_diff = st[idiff:]
        if '-' not in s_diff and ' ' not in s_diff:  # can remove part from en before isplit
            isplit = st[:idiff].rfind(' ')
            if isplit != -1:
                s = '\u2009–\u2009'.join((st, en[isplit+1:]))
            else:
                isplit = st[:idiff].rfind('-')
                if isplit != -1 and int(en[isplit+1:]) - int(st[isplit+1:]) == 1:
                    s = ',\u2009'.join((st, en[isplit+1:]))
                else:
                    # s = '{st[:isplit]}{st[isplit:]}\u2009–\u2009{en[isplit:]}'
                    s = "\u2009–\u2009".join((st, en))
        else:
            s = "\u2009–\u2009".join((st, en))

    # print("str_date_unit st, en:", st, en)
    preposition = (' {past} ' if unit_minutes else ' {of_}').format_map(I)
    return ''.join([unit, preposition, s])


def str_date_unit_nl(*args, allow3rows=False, no_blank_at_end=False, **kwargs):
    """
    Make label shorter for short graphs splitting row (with aligning digits by inserting ``blank``)
    """
    s = str_date_unit(*args, **kwargs)
    blank = '\\color{transparent}{\u2009–}'
    if "–" in s and allow3rows:
        return s.replace("of", rf"of{blank}\\").replace("–\u2009", r"–\\") + (
            "" if no_blank_at_end else blank
        )
    else:
        return s.replace('2', r'\\2', 1)
        # - instead of s.replace('of', r'of\\').replace('after', r'after\\').replace('после', r'после\\') ...


def str_date_unit_with_suffix(t_range, str_zone, **kwargs):
    """
    Used in Veusz Custom Definition as:
    str_date_u = (
    lambda ax, t_span_var, **kwargs:
    str_date_unit_with_suffix([f(lambda l: l if l!='Auto' else t, SETTING(f'{ax}/{lim:s}')) for lim, t in zip(('min', 'max'), DATA(f'{t_span_var}'))], str_zone='UTC+02:00', lang=LANG({'default': 'en', 'ru': 'ru'}), **kwargs)
    """
    b_nl = kwargs.pop('b_nl', False)
    higher = kwargs.pop('higher', False)
    str_date_unit_result=(str_date_unit_nl if b_nl else str_date_unit)(
            t_range,
            no_blank_at_end=str_zone,
            **kwargs
        )
    return (
        f'{str_date_unit_result}{chr(92)*2 if higher else chr(8201)}'
        f'{"^" if str_date_unit_nl and str_zone else ""}{str_zone}{chr(92)*2*(higher - 1)}'
    )

def str_dt(dt, lang=lang):
    """Time interval to readable string"""
    s = array(dt*1000000, 'M8[us]').item()
    a = int16(s.timetuple()[1:6]) - [1, 1, 0, 0, 0]
    if ~any(a):
        a = [0, 0, 0, 0, 0, round(s.microsecond * 1E-6, 3)]
    out = ' '.join([f'{d}{w}' for d, w in zip(
        a,
        ['месяцев', 'дней', 'ч', 'мин', 'с', 'с'] if lang == 'ru' else
        ['months', 'days', 'h', 'min', 's', 's']
    ) if d])
    return out.strip()
    # fDisp_read_dt(dt_var_name)', "f(f(''.join, [f'{d}{w}' for d,w in zip(
    #    f(lambda s: (f((lambda a: a if any(a) else [0,0,0,0,0,round(s.microsecond*1E-6, 3)]), int16(s.timetuple()[1:6]) - [1,1,0,0,0])), f(array(DATA(dt_var_name)*1E6, 'M8[us]').item)),
    #    ['months ', 'days ', 'h ', 'min. ', 's' ,'s']
    #    ) if d]).strip)



def day_sfx(d):
    return {1: 'st', 2: 'nd', 3: 'rd'}.get(d % 20, 'th') if lang != 'ru' else ''


def str_time_range(st, en, date_format='%d.%m.%Y', str_zone=''):
    """
    Time range string without repeating not changed time units in date format
    :param st:
    :param en:
    :param date_format:  After %d there may be {sfx} that will be replaced to appropriate day suffix
    :param str_zone: time text suffix
    :return:
    """
    str_st_date = f'{st:{date_format}}'
    str_en_date = f'{en:{date_format}}'
    if '{sfx}' in date_format:
        str_st_date = str_st_date.replace('{sfx}', day_sfx(st.day))
        str_en_date = str_en_date.replace('{sfx}', day_sfx(en.day))
        if '%e' in date_format:
            str_st_date = str_st_date.replace('. ', '.').replace('- ', '-').strip()
            str_en_date = str_en_date.replace('. ', '.').replace('- ', '-').strip()
        # day = str_st_date.split('{sfx}')[0].split('.')[-1].split('-')[-1].split('/')[-1].split(' ')[-1]

    b_ddate = str_st_date != str_en_date    # Have different dates?
    if b_ddate:                             # - Keep only different parts
        i_split = 0                         # previous date part separator index
        b_inc = '%d' == date_format[:2]     # date parts (units) in increased order
        for i, (left, right) in enumerate(
            zip(reversed(str_st_date), reversed(str_en_date)) if b_inc else
            zip(str_st_date, str_en_date)
        ):
            if left in '.- \\':
                i_split = i
            elif left != right:
                if i_split:
                    if b_inc:
                        str_st_date = f'{str_st_date[slice(0, -i_split - 1)]}'
                    else:
                        str_en_date = f'{str_en_date[slice(i_split + 1, None)]}'
                break
        str_en_date = f'{str_en_date}\u2009'
    else:
        str_en_date = ''
    return f'{str_st_date}\u2009{st:%H:%M}\u2009–\u2009{str_en_date}{en:%H:%M}{str_zone}'


def str_deg_min(degfloat, strpattern="{:d}°\u2009{:0.4f}\'", *args):
    """equiv. old variant: strpattern % (trunc(degfloat), abs(degfloat - trunc(degfloat))*60)
    """
    part_rem, part_trunc = modf(degfloat)
    return strpattern.format(int(part_trunc), abs(part_rem)*60, *args)


def str_deg_min_join(degfloat, strpattern="{:d}°\u2009{:0.4f}\'", add_strs='NE', joiner=', '):
    return joiner.join(str_deg_min(d, strpattern, a) for d, a in zip(degfloat, add_strs))


def row_jumps_if_small_dx(x, dx_min):
    """
    Changes row if distance to previous element on row is too small:
    for preventing overlapping of many close graphical elements if they would be placed on one row
    """
    p_prev = [-inf]*10  # can distribute to 10 rows maximum
    x_row = []
    for p in x:
        for row in range(len(p_prev)):
            if p - p_prev[row] > dx_min:
                x_row.append(row)
                p_prev[row] = p
                break
    return x_row


# def hsv_to_rgb(h, s, v):
#     """
#         Convert HSV values to RGB using manual calculations.
#         :param h: Hue (0-360)
#         :param s: Saturation (0-1)
#         :param v: Value/Brightness (0-1)
#         :return: RGB tuple
#     """
#     h %= 360
#     # s = clip(s, 0, 1).item()
#     # v = clip(v, 0, 1).item()
#     if v > 1:
#         v = 1
#     elif v < 0:
#         v = 0
#     if s > 1:
#         s = 1
#     elif s <= 0:
#         r = g = b = int(v * 255)
#         return (r, g, b)

#     c = v * s
#     h_prime = h / 60
#     x = c * (1 - abs(h_prime % 2 - 1))
#     m = v - c

#     if h_prime < 1:
#         r, g, b = c, x, 0
#     elif h_prime < 2:
#         r, g, b = x, c, 0
#     elif h_prime < 3:
#         r, g, b = 0, c, x
#     elif h_prime < 4:
#         r, g, b = 0, x, c
#     elif h_prime < 5:
#         r, g, b = x, 0, c
#     else:
#         r, g, b = c, 0, x

#     r = int((r + m) * 255)
#     g = int((g + m) * 255)
#     b = int((b + m) * 255)
#     return (r, g, b)


# def colors_of_hue_range(n: int, exclude_hue_start: int = 0, exclude_hue_end: int = 360):
#     """
#     Distribute hues evenly in the HSV color space while avoiding specified hue range
#     and adjusting saturation and value for vividness and perceptual separation.
#     :param n: number of colors to generate
#     :param exclude_hue_start: start of excluded hue range (0-360)
#     :param exclude_hue_end: end of excluded hue range (0-360)
#     :return: list of hex color codes in "#RRGGBB" format

#     Example: generate excluding blue (210-270 degrees) range:
#     >>> colors_of_hue_range(n, exclude_hue_start=210, exclude_hue_end=270)

#     """
#     colors = []
#     total_hue_range = 360
#     excluded_range = (exclude_hue_end - exclude_hue_start) % total_hue_range
#     available_hue_range = total_hue_range - excluded_range
#     step = available_hue_range / n
#     in_excluded_range = (
#         (lambda angle: exclude_hue_end < angle < exclude_hue_start)
#         if exclude_hue_start <= exclude_hue_end else
#         (lambda angle: (angle > exclude_hue_start or angle <= exclude_hue_end))
#     )
#     for i in range(n):
#         hue = i * step
#         if in_excluded_range(hue):
#             hue += excluded_range
#             hue %= 360

#         s = 1.0  # High saturation for vividness
#         v = 0.4 + i * 0.4 / (n - 1)  # gradient in brightness for visual separation from 40% to 80%

#         r, g, b = hsv_to_rgb(hue, s, v)
#         hex_color = "#{:02x}{:02x}{:02x}".format(r, g, b)
#         colors.append(hex_color)

#     return colors



# 30.06.2025

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
        term = scale if mu is None else scale * exp(-0.5 * ((x - mu) / sigma) ** 2)
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
    x = c * (1 - abs(h % 2 - 1))
    z = 0
    sector = floor(h).astype(int) % 6
    choices = (
        [c, x, z, z, x, c],
        [x, c, c, x, z, z],
        [z, z, x, c, c, x]
    )
    rgb = stack([choose(sector, ch) for ch in choices], axis=-1)
    rgb += (m if isinstance(m, float) else m[:, None])
    rgb *= 255
    return clip(rgb, 0, 255).astype(uint8)


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
    n_generate = fmax(1000, n * 100) if weight else n
    b_wrap_out = exclude_start <= exclude_end
    if b_wrap_out:
        hues_lin = (
            linspace(exclude_end, exclude_start + total_range, n_generate, endpoint=False)
        ) % total_range
    else:
        hues_lin = linspace(exclude_start, exclude_end, n_generate, endpoint=False)

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
        cdf = cumsum(weights)
        sample_points = linspace(0, cdf[-1], n, endpoint=False)
        indices_selected = searchsorted(cdf, sample_points)
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
        d_edges = array(dim)[[0,-1]]
        return d_edges + diff(d_edges) * array([0.25, -0.25])
    else:
        warning('len(dim) < 1')
        return dim + [-0.00000001, 0.00000001]


def i_positive(i, lim):
    """Return positive indices within the limit."""
    ii = int64(i)
    return where(ii < 0, lim + ii, ii)


def i_ranges(
        t, t_ranges,  # : Union[list[Union[list[Union[str,datetime64,int]], int]], ndarray]
        t_shift_s=0, t_units='ns'):
    """
    Find indexes of (useful) time ranges specified by t_ranges
    :param t: raw datetime64[ns] array
    :param t_ranges: iterable of ranges (iterables of 2 values) or single values. Values may be:
    - datetime64[ns] time - to search its integer index in `t` or
    - integer index - to return such elements without changes or
    - mix of described above
    :param t_shift_s: time shift [s], used to find indexes of t shifted on this value (after adding to t)
    :param t_units: units of t: can be 's', 'ns', ...
    return: list of 2-el. sequences ranges indexes
    """
    if (builtins.any if isinstance(t_ranges, list) else any)(t_ranges):
        t = int64(t)
    else:
        return [[0, t.size]]

    def to_two_elements(x):
        return x if len(x) == 2 else [x, x+1] if len(x) == 1 else [0, t.size]

    out = [to_two_elements([
            i_or_t if isinstance(i_or_t, int) else
            searchsorted(t, int64(array(i_or_t, f'datetime64[{t_units}]') - timedelta64(t_shift_s, 's')))
            for i_or_t in time_iter
        ]) for time_iter in t_ranges
    ]
    return out


def i_use(t, time_iter, t_shift_s=0, t_units='ns'):
    """
    Same as i_ranges() but for t_ranges replaced with time_iter that can be 1D or 2D.
    :param t: raw datetime64[t_units] array or its directly converted to float64 version
    :param time_iter: 1D or 2D values (in that case will be taken 1st 1D el. only) of same type as
    `t_ranges `parameter of i_ranges()
    :param t_shift_s: time shift [s], used to find indexes of t shifted on this value (after adding to t)
    :param t_units:
    :return: 1D array of indexes of time_iter in t or [0, t.size]
    """
    t = int64(t)
    time_iter = time_iter[0] if len(shape(time_iter)) > 1 else time_iter
    dtime_shift = timedelta64(t_shift_s, 's').astype(f'm8[{t_units}]')
    out = [
        i_positive(x, t.size) if isinstance(x, int) else t.size if x is None else
        searchsorted(t, int64(array(x, f'M8[{t_units}]') - dtime_shift))
        for x in time_iter
    ]
    if len(out):
        n_out = diff(out)
        if n_out <= 1:
            warning("Souse time range: [{}] is completely out of user selected time range ({})!".format(
                ', '.join(f'"{ti}"' for ti in array(
                        t[[0, -1]] + int64(dtime_shift), f'M8[{t_units}]'
                    ).astype('M8[s]')),
                time_iter
            ))
        return out
    else:
        return [0, t.size]


def min_range(range1, range2, l=nan):
    """accounts for negative indexes"""
    st, en = i_positive([take(range1, [0, -1]), take(range2, [0, -1])], l).T
    return [transpose([max(st), min(en)])]


def min_range_2d(*args):
    """The min_range_2d_no_check(se1, se2) replace. Still no check negative indexes"""
    a_any = [a for a in args if any(a)]
    if a_any:
        return atleast_2d([nanmax([a[:, 0] for a in a_any]), nanmin([a[:, -1] for a in a_any])])
    else:
        return [[]]


def max_range(range1, range2):
    """Maximum range from input ranges limits"""
    st1, en1 = range1
    st2, en2 = range2
    return append(fmin(st1, st2), fmax(en1, en2))


def ceil_log(x, div=2):
    """Good maximum for axis limit to display x values
    >> ceil_log(x) for x in [0.01, 0.013, 0.017, 0.2, 0.8, 1.2, 1.8, 2.2, 2.8, 11, 44]]
    ... [0.01, 0.015, 0.02, 0.2, 0.8, 1.5, 2.0, 2.5, 3.0, 15.0, 45.0]
    >> ceil_log(x, 1) for x in [0.01, 0.013, 0.017, 0.2, 0.8, 1.2, 1.8, 2.2, 2.8, 11, 44]]
    ... [0.01, 0.02, 0.02, 0.2, 0.8, 2.0, 2.0, 3.0, 3.0, 20.0, 50.0]
    """
    r = div * 10 ** -floor(log10(x))
    return ceil(x * r) / r


def power_ceil(x):
    return int32(floor(log10(x)))


def round_ceil_signed(x, n:float = None):
    """n: if positive should be float (not numpy type like float32)"""
    s = sign(x)
    abs_x = absolute(x)
    p = 10**-(power_ceil(abs_x) if n is None else n)
    return s * ceil(abs_x * p) / p


def shift_or_extend_lims(lim_in, x_in, e=None, scale=1):
    """Shifts range to include x if can, else extend range
    """
    lim = float64(lim_in)
    x = float64(x_in)
    x_range = diff(x).item()
    if x_range > 0:
        if e is None:
            # add 0.05 * x-range rounded to 1 decade
            e = 10**(int(floor(log10(x_range))) - 1)/2
        dl = max_range(lim, (x + [-e, e])*scale) - lim
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
    return where(a < minLim, nan, a)


def max2nan(a, maxLim):
    return where(a > maxLim, nan, a)


def minmax2nan(a, minLim, maxLim):
    return where((a < minLim) | (a > maxLim), nan, a)


def movavg_1d(a, n):
    if n>2 and size(a) >= n:
        n_m2 = int32(n//2) + 1
        n_p2 = int32((1 + n)//2)
        cum_a = cumsum(a)
        diff_width_n = cum_a[int32(n):] - cum_a[:-int32(n)]
        return hstack((
            cumsum(a[:n_m2])/arange(1, n_m2+1),
            diff_width_n/n,
            cumsum(a[:-n_p2:-1])[::-1]/arange(n_p2-1, 0, -1)
        ))
    elif n == 2:
        return hstack((a[0], (a[1:]+a[:-1]) / 2.0))
    else:
        return a

def moving_std(data: ndarray, window: int):

    # Compute cumulative sums
    cumsum1 = cumsum(data)
    cumsum2 = cumsum(data**2)

    # Calculate sums over the moving window
    sum_window = cumsum1[window:] - cumsum1[:-window]
    sum2_window = cumsum2[window:] - cumsum2[:-window]

    # Calculate means and variances
    mean_window = sum_window / window
    var_window = (sum2_window / window) - (mean_window**2)

    # Return the standard deviation
    std_window = sqrt(var_window)

    # Use numpy.pad to pad edges with the first and last valid standard deviation
    std_full = pad(
        std_window,
        (window - 1, len(data) - len(std_window) - (window - 1)),
        mode='edge'
    )
    return std_full

    # moving_std = sqrt(convolve(data**2, ones(window), 'valid') / window -
    #     convolve(data, ones(window), 'valid')**2 / window**2)


def rep2mean(x, b_ok=None, left=None, right=None):
    if b_ok is None:
        b_ok = isfinite(x)
        return interp(arange(len(x)), flatnonzero(b_ok), x[b_ok])
    return interp(arange(len(x)), flatnonzero(b_ok), x[b_ok], left, right)


def rep2mean_dir2rad(x, b_ok):
    # unwrap(),
    return interp(arange(len(x)), flatnonzero(b_ok), radians(x[b_ok]), period=2*pi)

def rep2prev(x, b_ok=None):
    if b_ok is None:
        b_ok = isfinite(x)
    ok = asarray(b_ok, dtype=int8)
    i_before_nan = flatnonzero(diff(ok) < 0)
    i_last_nan = flatnonzero(diff(ok) > 0)
    if i_before_nan.size and i_last_nan.size:
        # Prepare closed NaN regions to have equal values on edges:
        # delete edges for which opposite edge is open end delete them
        if isnan(x[0]):  # 1st NaNs have no finite value before
            del i_before_nan[0]
        if i_before_nan.size:  # last NaNs have no finite value after
            if isnan(x[-1]):
                del i_last_nan[0]
            out = x.copy()
            out[i_last_nan] = out[i_before_nan]
            b_ok[i_last_nan] = True
    out = interp(arange(len(x)), flatnonzero(b_ok), out[b_ok])
    return out


#@njit
def b1spike(a, max_spike=0):
    """
    Single spike detection
    Note: change of a at edge bigger than max_spike is treated as spike too
    :param a:
    :param max_spike:
    :return: boolean array of where is spike in a
    """
    b_single_spike_1 = lambda bad_u, bad_d: logical_or(
        logical_and(append(bad_d, True), append(True, bad_u)),  # spike to down
        logical_and(append(bad_u, True), append(True, bad_d)))  # spike up
    diff_x = diff(a)
    return b_single_spike_1(diff_x < -max_spike, diff_x > max_spike)


def bin_avg(a: ndarray, edges, st=0):
    """
    Bin average of ``a[st:after_last]`` inside edges where ``after_last = int(edges[-1])`` - index of the end of the last interval
    :param a: numpy ndarray
    :param edges: ``a`` indexes relative to ``st`` to calculate mean(a) between
    :param st: ``edges`` origin in ``a`` indexes
    """
    edges_int = int32(edges)
    starts, after_last = array_split(edges_int, [-1])
    return (add.reduceat(
        a[st:after_last.item()], (starts - st) if st else starts
    ) / ediff1d(edges_int))[:, None if a.ndim > 1 else ...]


def bin_std(a: ndarray, edges: ndarray, st=0):
    """
    Bin average of ``a[st:after_last]`` inside edges where ``after_last = int(edges[-1])`` - index of the end of the last interval
    :param a: numpy ndarray
    :param edges: ``a`` indexes relative to ``st`` to calculate mean(a) between
    :param st: ``edges`` origin in ``a`` indexes
    """
    edges_int = int32(edges)
    starts, after_last = array_split(edges_int, [-1])

    # Calculate sums over intervals
    sum_bin = add.reduceat(a[st : after_last.item()], (starts - st) if st else starts)
    sum_sq = add.reduceat(a[st : after_last.item()] ** 2, (starts - st) if st else starts)

    n = ediff1d(edges_int)  # bin length
    variance = (sum_sq - sum_bin**2 / n) / n

    # Return the standard deviation
    return sqrt(variance)[:, None if a.ndim > 1 else ...]


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
    d_ok = diff(b_ok, prepend=False, append=False)
    edges = flatnonzero(d_ok != 0)
    n_rows = diff(edges)
    # Delete too short bad data intervals
    if pressure is None:
        b_del_bad_interval = n_rows[1::2] < (min_range_bad or min_range)
    else:
        dp = abs(diff(pressure[edges]))
        b_del_bad_interval = (n_rows[1::2] + dp[1::2]) < (min_range_bad or min_range)
    if b_del_bad_interval.any():
        edges = edges[hstack((True, ~repeat(b_del_bad_interval, 2), True))]
        n_rows = diff(edges)
    # Delete too short good data intervals
    if pressure is None:
        b_del_good_interval = n_rows[::2] < min_range
    else:
        dp = abs(diff(pressure[edges]))
        b_del_good_interval = (n_rows[::2] - dp[::2]) < min_range
    if b_del_good_interval.any():
        edges = edges[~repeat(b_del_good_interval, 2)]

    return edges


def ranges2bool(st_en, length):
    a = zeros(length, bool_)
    for se in int32(st_en):
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
        xc[..., slice(*se)] = nan
    return xc


def put_in_nans(x, st_ends, x_len=None):
    if x_len is None:
        x_len = x.shape[1]
    elif x_len != x.shape[1] or st_ends[0,0] != 0:
        xc = empty((x.shape[0], x_len)) + nan()
        xc[..., st_ends[0, 0]:x_len] = x
    else:
        xc = x.copy()

    for se in zip([0] + st_ends[:, 1].tolist(), st_ends[:, 0].tolist() + [x_len]):
        xc[..., slice(*se)] = nan
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
        where_no_fun_out = nan

        def fun_in(se):
            return fun(arr[slice(*se)])

    elif hasattr(where_no_fun_out, '__len__'):
        def fun_in(se, other):
            return fun(arr[slice(*se)], other=other)

        return int32([
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

    p_inv = -flip(pres)
    st_ends_inv = pres.size - fliplr(st_ends)
    i_st = pres.size - (flip(in_ranges(
        p_inv,
        lambda d, other: searchsorted(d, d[0] + pres_range),
        st_ends_inv,
        where_no_fun_out=pres.size
    )) + st_ends_inv[:, 0])
    return i_st



def bad_bot_by_diff(
        x: ndarray, fun, i_en, i_st=None, p_range=None, p: ndarray = None, speed: ndarray = 1
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
        A function that takes a segment of `ediff1d(x, to_begin=0)/speed` and returns a boolean mask or indices
        indicating positions of "bad" data.
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

    # # Build initial start-end intervals
    # if i_st is None:
    #     # Default: start at first finite point in p (if available), else 0
    #     # Then use i_en[:-1] + 1 as preceding ends
    #     first_finite = flatnonzero(isfinite(p))[0] if have_p else 0
    #     st_ends = append(first_finite, i_en[:-1] + 1)
    # elif hasattr(i_st, "__len__"):
    #     st_ends = asarray(i_st, dtype=int32)
    # else:
    #     # i_st is a scalar offset
    #     st_ends = i_en[:-1] + i_st
    #     st_ends = append(i_st, st_ends)  # prepend single value

    st_ends = int32(column_stack((
        i_st if hasattr(i_st, '__len__') else append(
            flatnonzero(isfinite(p))[0] if have_p else 0,
            i_en[:-1] + (1 if i_st is None else i_st)
            ),
        i_en
        ))
    )

    # Adjust search start points based on pressure range (if applicable)
    if have_p:
        st_ends[:, 0] = i_before(p, p_range, st_ends)

    # Search fun() points and replace starting search points with them
    st_ends[:, 0] += in_ranges(
        ediff1d(x, to_begin=0) / speed,  # diff(x)/speed[:-1]
        lambda xr, other: at_or_other(flatnonzero(fun(xr)), 0, other=other),
        st_ends,
        where_no_fun_out=x.size,  # something > diff(st_ends)
    )
    return put_nans(x, st_ends)


def bad_top_by_diff(
        x: ndarray, fun, i_en, i_st=None, p_range=None, p: ndarray = None, speed: ndarray = 1
):
    """
    :param x: _description_
    :param fun: _description_
    :param i_st: _description_, defaults to None
    :param i_en: _description_

    :param p_range: _description_, defaults to None
    :param p: _description_, defaults to None
    :param speed: _description_, defaults to 1
    """
    # v.bad_bot_by_diff(CTD_Sal_f, lambda x: (x<-0.1)| (x>5), CTDends, i_st=CTDstarts, p_range=1, p=CTD_Pres, speed=CTDspeedDown_MA)
    # lambda x: (x<-0.001)|(x>0.01)
    x_len = x.size
    return flip(bad_bot_by_diff(
        flip(x),
        fun,
        i_en=x_len - i_st,
        i_st=x_len - i_en,
        p_range=p_range,
        p=-flip(p) if p is not None else None,
        speed=-flip(speed) if speed is not None else None,
    ))


# 2 old functions:
def f_iBadBefore(arr, fun_bad, i_en, to_end=50):
    """
    to_end: int32
    """
    st_ends = int32(column_stack((i_en - to_end, i_en)))
    return int32([at_or_other(flatnonzero(fun_bad(arr[slice(*se)])), 0, to_end) for se in st_ends]) - to_end


def bad_bot_by_diff_old(x, fun_bad, i_en=None, speed=1):
    """
    old variant: fBotNansDiff = lambda(x,fun_bad,i_en):
    fPutNans(x, f_iBadBefore(x, fun_bad, i_en)                    , i_en)
    fPutNans(x, f_iBadBefore(diff(x)/speed[:-1], funBad, i_en, 50), i_en)
    where f_iBadBefore(arr,funBad,i_en,to_end) = int32([append(flatnonzero(funBad(arr[int32(En-to_end):int32(En)])), to_end)[0] for En in i_en]) - to_end
    """
    return put_nans(x, f_iBadBefore(diff(x)/speed[:-1], fun_bad, i_en), i_en)


def grid(z, y, grid_y, st, en, reverse: builtins.bool = False):
    return transpose([
        (interp(grid_y, y[s:e], z[s:e], nan, nan) if any(z[s:e]) else nan + grid_y) for s, e in
        (zip(int32(st)[::-1], int32(en)[::-1]) if reverse else zip(int32(st), int32(en)))
        ])


def i_whole_time(time, dt, dt_shift: int = 0):
    """
    :param time: numpy array of time values
    :param dt: units str to get f'M8[{dt}]' date type directly or seconds to find it
    :param dt_shift: shift, s
    :return: time array converted to dt units shifted on dt_shift
    """
    if not isinstance(dt, str):
        unit_sym, unit_dt = (
            ('s', 1), ('m', 60), ('h', 3600), ('D', 24*3600)
        )[searchsorted([60, 3600, 24*3600], dt)]
        dt = f'{fmax(dt // unit_dt, 1)}{unit_sym}'

    return array(int64(time + dt_shift), 'M8[s]').astype(f'M8[{dt}]')
    #??? why earlier: array(array((time + dt_shift) // max(dt // unit_dt, 1), 'M8[s]'), f'M8[{unit_sym}]')


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
    # Coarse resolution to dt and then find positions where it changes
    ind = flatnonzero(diff(int8(i_whole_time(time, dt, dt_shift)))) + 1  # whole time indexes
    try:
        _slice = slice(
            ( 1 if 2*subtract(*time[[ind[0],   0]]) < (dt_burst or dt) else 0        ),
            (-1 if 2*subtract(*time[[-1, ind[-1]]]) <  dt              else len(time))
        )
        return hstack([0, ind[_slice], len(time) - 1])
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
    o = copy(t)
    inds_found_st = searchsorted(t, packet_st)
    inds_found_en = inds_found_st[1:] + 1
    inds_found_en[-1] = t.size

    for ind_s, ind_e, *se in zip(inds_found_st[:-1], inds_found_en, packet_st[:-1], packet_st[1:]):
        o[arange(ind_s, ind_e, dtype=int32)] = linspace(*se, ind_e - ind_s)
    return o

# old version
# def stretch_time(t, packet_st):
#     o = copy(t)
#     for se in zip(packet_st, append(packet_st[1:], t[-1])):
#         ind = searchsorted(t, se)
#         if ind[-1] < t.size:
#             ind[-1] += 1
#         o[arange(*ind, dtype=int32)] = linspace(*se, diff(ind).item())
#     return o


def i_closest(a, values):
    """
    a: numpy array
    values: should be sorted
    https://stackoverflow.com/a/46184652/2028147
    """

    # get insert positions
    idxs = searchsorted(a, values, side="left")

    # find indexes where previous index is closer
    prev_idx_is_less = (idxs == len(a)) | (fabs(values - a[maximum(idxs-1, 0)]) < fabs(values - a[minimum(idxs, len(a)-1)]))
    idxs[prev_idx_is_less] -= 1

    return idxs


def rose_table(v_abs, v_dir, rose_bins, nsectors=32):
    sector = 360./nsectors
    angles = arange(-sector / 2, 360. + sector, sector, dtype=float)
    t = histogram2d(x=v_abs, y=v_dir%360, bins=[rose_bins, angles], normed=False)[0]
    t2 = column_stack((nansum(t[:, [0,-1]], axis=1), t[:, 1:-1]))
    info(t)
    return flipud(cumsum(flipud(t2*100/nansum(t2)), axis=0))
    # AddCustom('definition', 'f_rose_table(v_abs, v_dir)', 'f(lambda t2: flipud(cumsum(flipud(t2*100/nansum(t2)), axis=0)), f(lambda t: column_stack((nansum(t[:, [0,-1]], axis=1), t[:, 1:-1])), histogram2d(x=v_abs, y=(v_dir + (0 if Rose_blow_to else 180))%360, bins=[Rose_bins, f(lambda angle: arange(-angle / 2, 360. + angle, angle, dtype=float), 360./Rose_nsector)], normed=False)[0]))')


# for inclinometers

def rotate(r_from, r_to):
    """
    Rotation matrix to rotate 1st vector to second
    """
    if all(r_from == r_to):
        return eye(3)

    a = ravel(r_from)
    b = ravel(r_to)

    c = cross(a, b)
    scc_cross_ab = float64([
        [0, -c[2], c[1]],
        [c[2], 0, -c[0]],
        [-c[1], c[0], 0]
    ])
    return eye(3) + scc_cross_ab + linalg.matrix_power(scc_cross_ab, 2) * (1 - a @ b) / sum(c**2)


def mag_dec(lat, lon, time_iso, depth=0):
    """
        Returns magnetic declination using wmm2020 library

    :param lat, lon: coordinates in degrees WGS84
    :param time_iso: # like '2020-09-20'
    :param depth: in km (negative below sea surface)
    """
    run_commands = f'c:/Users/and0k/conda/Scripts/activate.bat py3.10x64h5togrid && python -c "from datetime import datetime; import wmm2020 as wmm; year_fraction = lambda date: (lambda start: date.year + float(date.toordinal() - start) / (datetime(date.year + 1, 1, 1).toordinal() - start))(datetime(date.year, 1, 1).toordinal()); mag = wmm.wmm({lat}, {lon}, {depth}, year_fraction(datetime.fromisoformat(\'{time_iso}\'))); print(mag.decl.item(0))"'
    decl_str = check_output(
        f'c:/Users/and0k/conda/Scripts/activate.bat py3.10x64h5togrid && {run_commands}',
        shell=True, text=True
    )
    return float(decl_str)


def wrap_dir(x, disp_central_dir=180):
    """Wrap directions around central direction."""
    dir0 = disp_central_dir - 180
    return (x - dir0) % 360 + dir0


def wrap_dir_unwrap180(x, disp_central_dir, pass_over):
    s = wrap_dir(x, disp_central_dir)
    bs = abs(s - disp_central_dir) < 180 - pass_over
    return where(
        (abs(interp(arange(len(s)), flatnonzero(bs), s[bs]) - s) > 90) & ~bs,  # , period=360
        where(s < disp_central_dir, s + 360, s - 360),
        s
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
    s = dot(coef_a2d, raw3d - coef_c)

    # If gain for some channel is zero
    i_ch_bad = flatnonzero(
        coef_a2d.diagonal() == 0
    )  # diagonal elements: coef_a2d[[0, 1, 2], [0, 1, 2]]
    if i_ch_bad.size:
        # Recover channel
        i_ch_ok = [i for i in range(3) if i != i_ch_bad]
        s[i_ch_bad] = square(1 - (s[i_ch_ok] ** 2).sum(axis=0))
        if (s[i_ch_bad].imag != 0).any():
            s[i_ch_bad] = s[i_ch_bad].real

        # Sign of recovering channel
        if raw3d_helps_recover is not None:
            s[i_ch_bad] *= sign(raw3d_helps_recover[i_ch_bad])
            return s

        # Select sign in inverse gradient points so that we minimise changes of gradient
        s_dif = ediff1d(s[i_ch_bad], to_begin=0)
        # where gradient reverse
        b_reversed = s_dif < 0  # signal decreases
        b_reversed &= append(
            ~b_reversed[1:], False
        )  # and signal increases in next point
        # Keep reversed where diff(s_dif) of inverted signal will be less than of not inverted (where gradient reversed)
        # diff of reversed signal (after b_reversed points)
        _ = s[i_ch_bad, b_reversed] + s[i_ch_bad, roll(b_reversed, 1)]
        # diff in point before gradient reverse:
        s_dif_prev = s_dif[roll(b_reversed, -1)]
        # select sign of signal with minimum change of |gradient|
        b_reversed[b_reversed] = abs(s_dif_prev - _) < abs(
            s_dif_prev - s_dif[b_reversed]
        )

        # Consequently invert signal in b_reversed points
        s_rev_sign = zeros_like(s_dif)
        s_rev_sign[0] = 1
        n_reversed = sum(b_reversed)
        s_rev_sign[b_reversed] = tile([-2, 2], int(ceil(n_reversed / 2)))[:n_reversed]
        s_rev_sign = cumsum(s_rev_sign)
        s[i_ch_bad] = s_rev_sign * s[i_ch_bad]
    return s


def xy_or_y(x, y, use_x_if=lambda x: bool(x), f_xy=operator.add):
    return f_xy(x, y) if (use_x_if(x) if callable(use_x_if) else use_x_if) else y

def xy_or_x(x, y, use_y_if=lambda y: bool(y), f_xy=operator.add):
    return f_xy(x, y) if (use_y_if(y) if callable(use_y_if) else use_y_if) else x

def xy_sel(x, y, use_x_if=lambda x: bool(x), use_y_if=lambda y: bool(y), f_xy=operator.add, nothing=""):
    use_x = (use_x_if(x) if callable(use_x_if) else use_x_if)
    use_y = (use_y_if(y) if callable(use_y_if) else use_y_if)
    if use_x:
        return f_xy(x, y) if use_y else x
    else:
        return y if use_y else nothing

# For CTD "zabor"

def cor_where_run_p(arr, dict_i_p, inds, pres, fun=(lambda x: nan)):
    """
    Applies fun() to parts of arr' copy and returns it. Part is found for each ``dict_i_p`` item
    which key equal to ``inds`` and ``pres`` between its values.
    arr: parameter to change by apply fun on its parts (one part per run)
    dict_i_p: {run#: [minP, maxP], ...} - run number, its min and max pressure where apply fun
    inds: array (of arr size) which each element assigned to run number it belongs
    pres: pressure
    fun: function to apply, default: set elements to nan
    """
    ac = arr.copy()
    for i, (p_st, p_en) in dict_i_p.items():
        b = (inds == i) & (p_st < pres) & (pres < p_en)
        ac[b] = fun(arr[b])
        # putmask(ac, b , fun(arr[b])) is bad if some NaNs!
    return ac


def where_run_p(arr, dict_i_p, inds, pres, fun=nanmean, val_for_fun_of_empty=None):
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
        i_log_use = i_ranges(r_st, tranges, time_shift_s - 10, t_units='ns')
        # next start:
        # r_next = r_st[sl(int32(i_log_use) + 1)]
        n_st_i.append([
            r_n[sl(i_log_use)],
            r_st[sl(i_log_use)],
            # r_next if i_log_use[-1][-1] < len(r_st) else append(r_next, r_n[-1]),
            full(ediff1d(i_log_use).item(), i)
        ])

    # device index, run' start time, run' end time, run' up end time (= next run start):
    n_st_j = vstack(n_st_i)
    # sort by run start time
    j_sort = argsort(n_st_j[1, :])

    # List of selected runs
    i_runs = [
        i for se in use_runs_in_used_range for i in (
            [se] if isinstance(se, int) else
            range(*[(n_st_j.shape[1] if j is None else (j + n_st_j.shape[1]) if j < 0 else j) for j in se])
        )
    ]
    # same effect expression: sum([[se] if isinstance(se, int) else
    # list(range(*[((i + n_st_j.shape[1]) if i < 0 else i) for i in se])) for se in use_runs_in_used_range])

    n_st_j_use = n_st_j[:, j_sort[i_runs]]
    # warning('zabor runs indexes selected: %s', repr(arange(n_st_j.shape[1])[j_sort[i_runs]]))
    j_use = n_st_j_use[-1, :]

    # Data runs indexes
    # - in raw data
    idata_st = hstack([searchsorted(data_indexes[int(j.item(0))], starts) for starts, j in (
        hsplit(n_st_j_use[1:, :], flatnonzero(diff(j_use)) + 1)
    )])  # if i > 0 else n_st_j_use[1:, :].T
    idata_en = idata_st + n_st_j_use[0, :]

    # - in selected runs we will combine (except 1st = 0)
    idatasel_en = cumsum(n_st_j_use[0, :])
    idatasel_st = append(0, idatasel_en[:-1])

    out = column_stack([idatasel_st, idatasel_en, idata_st, idata_en, j_use])
    #warning('zabor_runs_edges() result has shape %s: %s', repr(out.shape), repr(out))
    return out


# for correct bug in 2D expression definition (not works)

def _DS_(bit, part):
    """ failed correction of Veusz bug of dimension query
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
    vars, dtypes=('M8[s]',), index='Time_UTC', file='~cash.csv', usecols=None, delimiter='\t',
    fun_out=lambda index, loaded_index, loaded: atleast_2d(loaded[isin(int32(loaded_index), int32(index))].view('f8'))
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
        n_formats_skipped = (len(vars) - len(dtypes))
        if n_formats_skipped:
            dtypes = tuple(dtypes) + ('f8',) * n_formats_skipped
        data_exist = loadtxt(
            file, skiprows=1, usecols=usecols, delimiter=delimiter,
            dtype=dtype({
                'names': tuple(vars), 'formats': ['f8' if dtyp.startswith('M') else dtyp for dtyp in dtypes]
            }),
            converters={i: (lambda x: dt64s2vsz(datetime64(x, 's'))) for i, dtyp in enumerate(dtypes) if dtyp.startswith('M')}
        )
        # index_icol = tuple(vars).index(index_col)
        # print(index_val)
        # print(data_exist[index_col])
        # print(data_exist)
    except:
        exception('Error in load_cash(%s))!', file)
        return
    if index_val is None:
        return data_exist
    return fun_out(index_val, data_exist[index_col], data_exist)


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
    b_array = isinstance(data, ndarray)
    # Set/check the number of formatters is same as the number of columns
    n_elements = (
        (
            (data.shape[1] if data.ndim > 1 else 1) if data.dtype.names is None else len(data.dtype.names)
        )
        if b_array else len(data)
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
        # Vectorized application of the formatter
        if prev_formatter != formatter:
            vec_formatter = vectorize(
                f"{{:{formatter}}}".format
                if isinstance(formatter, str)
                else formatter
            )

        formatted_data.append(vec_formatter(data[:, i] if b_array else data[i]))
        prev_formatter = formatter

    return array(formatted_data).T


def save2text(
    vars,
    dtypes=(),
    formats=(),
    delimiter="\t",
    file=None,
    file_sfx="_out.tsv",
    skip_if_exist=True,
    fun_get=lambda x: x,  # : Callable[[Any], Any]
    fun_before_compare=int32,
):
    """
    Save data to a text file.

    :param vars: variables to save. Must be of type for which `list(vars)` returns Sequnce[str] of var names
    - Mapping[str, Any]: column names to data values
    - Sequnce[str]: values will be obtained with `fun_get()` of each var
    :param dtypes: data types for the `vars`. If their number is less than `vars` then remained are float64.
    Default: empty tuple. Use any numpy array types: "M8[s]", "f8'...
    :param formats: format specifiers for the `vars` - standard ("f", "s", ...) or custom formatting
        functions.
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
        'bin2_t0st__', '\t',
        lambda x: (warning(x), DATA(x))[1]
    )
    """

    if file is None:
        parent, basename = (lambda p: (p.parent, p.name))(Path(sys.argv[1]))
        file = parent / (basename.rsplit('.')[0] + file_sfx)
    else:
        file = Path(file)
    data_exist = False

    n_formats_skipped = (len(vars) - len(dtypes))
    if n_formats_skipped:
        dtypes = tuple(dtypes) + ('f8',)*n_formats_skipped
    def from_vsz(v, fmt):
        return vsz2dt64s(fun_get(v)) if fmt.startswith("M") else fun_get(v)

    warning(f'Saving {vars} to {file}...')
    if skip_if_exist:
        if isinstance(skip_if_exist, builtins.bool):
            if file.is_file():
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
                # 2. data loaded earlier
                if data_exist == 'loadtxt':
                    var_icol = tuple(vars).index(skip_if_exist)
                    dtyp_col = dtypes[var_icol]
                    data_exist = loadtxt(
                        file, skiprows=1, dtype=dtyp_col, usecols=[var_icol], delimiter=delimiter
                    )
                    if dtyp_col.startswith('M'):
                        data_exist = dt64s2vsz(data_exist)
                elif data_exist is None:
                    data_exist = fun_get(f'~{var_name}{file_stem}')

                data_new_cmp = fun_before_compare(data_new)
                data_exist_cmp = fun_before_compare(data_exist)
                b_new = ~isin(data_new_cmp, data_exist_cmp, assume_unique=True)
                if not b_new.any():  # , kind='table' is not supported yet
                    return -2  # data exist
                else:
                    warning(f'{data_new}->{data_new_cmp}: not in {data_exist}->{data_exist_cmp}! Saving%s...',
                    '' if b_new.all() else f' {flatnonzero(b_new)} items'
                    )
            except Exception:
                data_exist = False
                exception('Error in save2text(%s))!', file.name)
                warning(f'~{var_name}{file_stem}: {data_exist}')
                warning(f'column_name: {fun_get(var_name)}')
                return
    # from numpy.core.records import fromarrays
    warning(f"dtypes={dtypes}, type: {type(dtypes)}")
    # 1 get values
    vars_vals = (
        vars.values()
        if isinstance(vars, dict)
        else [from_vsz(v, fmt) for v, fmt in zip(vars, dtypes)]
        if fun_get
        else vars
    )
    el_size = next(iter(vars_vals)).size
    #if len(v) for v in val_arr):
    #else:
    #    warning([])
    try:
        # 2. broadcast scalars
        val_arr = [
            repeat(v, el_size) if isscalar(v) else v
            for v, fmt in zip(vars_vals, dtypes)
        ]
        # data_records = fromarrays(val_arr, dtype=dtype({'names': tuple(vars), 'formats': dtypes}))
        n_formats_skipped = (len(vars) - len(formats))
        if len(formats) < len(dtypes):
            formats = list(formats) + [
                fmt_3_digits_after_dot if dtyp.startswith("f") else "s"
                for dtyp in dtypes[-n_formats_skipped:]
            ]



        with file.open(mode= 'a' if skip_if_exist else 'w') as f:
            if data_exist is False:
                savetxt(f, atleast_2d(list(vars)), '%s', delimiter=delimiter)
            # savetxt(f, data_records, fmt=formats, delimiter=delimiter)  # delimiter.join(formats)
            str_array = format_2d_array(val_arr, formatters=formats)
            warning(f"formats: {formats},\narray: {str_array}\n")
            savetxt(
                f,
                str_array,
                fmt="%s",
                delimiter=delimiter,
            )
        warning("File %s: %s", 'saved' if data_exist is False else 'appended', file)
        # print("File %s: %s" % ('saved' if data_exist is False else 'appended', file))
    except Exception:
        exception('Error in save2text(%s)!', file.name)
        warning(f'Creating record array from: {val_arr} and saving is failed!\nTypes: {dtypes}')
        warning(f'{fun_get}(vars={vars}) gives: {[fun_get(n) for n in vars_vals]}')
        return
    return str_array.shape[1]  # rows written


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
    return scale, offset


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
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    R = 6371000  # radius of the earth in m
    klon = cos((lat2 + lat1) / 2)
    dx = R * klon * dlon
    dy = R * dlat

    d = hypot(dx, dy)
    # angle = arctan2(dlat, dlon)
    angle = arctan2(dx, dy)  # or use atan2[(sin Δλ ⋅ cos φ₂), (cos φ₁ ⋅ sin φ₂ − sin φ₁ ⋅ cos φ₂ ⋅ cos Δλ)] from https://www.omnicalculator.com/other/azimuth

    return column_stack((dx, dy, d, degrees(angle)))


def skewness(data):
    n = len(data)
    mean_ = mean(data)
    std_ = std(data, ddof=1)  # Set ddof=1 for sample standard deviation
    third_moment = ((data - mean_)**3).sum() / n
    out = third_moment / (std_**3)
    return out

def kurtosis(data):
    """
    Excess kurtosis (where the kurtosis of the normal distribution is zero).
    If you want the Fisher kurtosis (which would be three for a normal distribution), you can simply remove the - 3
    """
    n = len(data)
    mean_ = mean(data)
    std_ = std(data, ddof=1)
    fourth_moment = ((data - mean_)**4).sum() / n
    out = fourth_moment / (std_**4) - 3  # Subtract 3 for excess kurtosis
    return out



### Functions copied from https://pypi.org/project/seawater/3.3/ to calculate potential density (sw_pden()) ###
###############################################################################################################

T68conv = lambda t90: t90*1.00024
T90conv = lambda t68: t68/1.00024


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

        a = [3.5803e-5, 8.5258e-6, -6.836e-8, 6.6228e-10]
        b = [1.8932e-6, -4.2393e-8]
        c = [1.8741e-8, -6.7795e-10, 8.733e-12, -5.4481e-14]
        d = [-1.1351e-10, 2.7759e-12]
        e = [-4.6206e-13, 1.8676e-14, -2.1687e-16]
        return (a[0] + (a[1] + (a[2] + a[3] * T68) * T68) * T68 +
                (b[0] + b[1] * T68) * (s - 35) +
                ((c[0] + (c[1] + (c[2] + c[3] * T68) * T68) * T68) +
                 (d[0] + d[1] * T68) * (s - 35)) * p +
                (e[0] + (e[1] + e[2] * T68) * T68) * p * p)

    # Theta1.
    del_P = pr - p
    del_th = del_P * adtg(s, t, p)
    th = T68conv(t) + 0.5 * del_th
    q = del_th
    sqrt2 = sqrt(2)

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
        """Density of Standard Mean Ocean Water (Pure Water) using EOS 1980.
        """
        a = (999.842594, 6.793952e-2, -9.095290e-3, 1.001685e-4, -1.120083e-6,
             6.536332e-9)
        return (a[0] + (a[1] + (a[2] + (a[3] + (a[4] + a[5] * T68) * T68) * T68) *
                T68) * T68)



    # UNESCO 1983 Eqn.(13) p17.
    b = (8.24493e-1, -4.0899e-3, 7.6438e-5, -8.2467e-7, 5.3875e-9)
    c = (-5.72466e-3, 1.0227e-4, -1.6546e-6)
    d = 4.8314e-4
    return (smow() + (b[0] + (b[1] + (b[2] + (b[3] + b[4] * T68) * T68) *
            T68) * T68) * s + (c[0] + (c[1] + c[2] * T68) * T68) * s *
            sqrt(s) + d * s ** 2)


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
        # h0 = -0.1194975
        h = [3.239908, 1.43713e-3, 1.16092e-4, -5.77905e-7]
        AW = h[0] + (h[1] + (h[2] + h[3] * T68) * T68) * T68

        # k0 = 3.47718e-5
        k = [8.50935e-5, -6.12293e-6, 5.2787e-8]
        BW = k[0] + (k[1] + k[2] * T68) * T68

        # e0 = -1930.06
        e = [19652.21, 148.4206, -2.327105, 1.360477e-2, -5.155288e-5]
        KW = e[0] + (e[1] + (e[2] + (e[3] + e[4] * T68) * T68) * T68) * T68

        # Sea water terms of secant bulk modulus at atmos. pressure.
        j0 = 1.91075e-4
        i = [2.2838e-3, -1.0981e-5, -1.6078e-6]
        sqrt_s = sqrt(s)
        A = AW + (i[0] + (i[1] + i[2] * T68) * T68 + j0 * sqrt_s) * s

        m = [-9.9348e-7, 2.0816e-8, 9.1697e-10]
        B = BW + (m[0] + (m[1] + m[2] * T68) * T68) * s  # Eqn 18.

        f = [54.6746, -0.603459, 1.09987e-2, -6.1670e-5]
        g = [7.944e-2, 1.6483e-2, -5.3009e-4]
        K0 = (KW + (f[0] + (f[1] + (f[2] + f[3] * T68) * T68) * T68 +
                    (g[0] + (g[1] + g[2] * T68) * T68) * sqrt_s) * s)  # Eqn 16.
        return K0 + (A + B * p) * p  # Eqn 15.`

    # UNESCO 1983. Eqn..7  p.15.
    densP0 = dens0(s, t)
    K = seck(s, p)
    p = p / 10.  # Convert from db to atm pressure units.
    return densP0 / (1 - p / K)


def sw_pden(s, t90, p, pr):
    return dens(s, ptmp(s, t90, p, pr), pr)


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
    c0 = -0.000000488682

    Ts = log((298.15 - t) / (273.15 + t))
    oxsol = exp(polyval(a, Ts) + S * polyval(b, Ts) + c0 * S**2)
    return oxsol


def oxygen_solubility_scor(t, S, P=0, p_atm=1013.25):
    """
    Oxygen solubility according to recommendations by SCOR WG 142 "Quality Control Procedures
    for Oxygen and Other Biogeochemical Sensors on Floats and Gliders"

    :param t: temperature in °C
    :param S: salinity (PSS-78)
    :param P: hydrostatic pressure in dBar (default: 0 dBar)
    :param p_atm: atmospheric (air) pressure in mBar (default: 1013.25 mBar)
    :return: Oxygen solubility in µmol L-1
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
    t_sca = log((298.15 - t) / t_k)

    # Saturated water vapor in mBar
    pH2Osat = 1013.25 * (exp(24.4543-(67.4509*(100 / t_k))-(4.8489*log((t_k / 100)))-0.000544 * S))

    # Temperature correction part from Garcia and Gordon (1992), Benson and Krause (1984) refit mL(STP) L-1; and conversion from mL(STP) L-1 to umol L-1
    TCorr   = 44.6596 * exp(polyval([3.88767 , -0.256847, 4.94457, 4.0501, 3.22014, 2.00907], t_sca))
    # Salinity correction part from Garcia and Gordon (1992), Benson and Krause (1984) refit ml(STP) L-1
    Scorr   = exp(S * polyval([-8.17083e-3, -1.03410e-2, -7.37614e-3, -6.24523e-3], t_sca) - 4.88682e-7 * S ** 2)
    # Molar volume of O2 in m3 mol-1 Pa dBar-1 (Enns et al. 1965)
    Vm      = 0.317
    # Universal gas constant in J mol-1 K-1
    R       = 8.314

    return 100 * (TCorr * Scorr) * (p_atm - pH2Osat) / (1013.25 - pH2Osat) / exp(Vm * P / (R * t_k))
