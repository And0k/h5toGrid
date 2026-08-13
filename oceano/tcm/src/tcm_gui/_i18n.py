"""
i18n string loader
Chrome i18n strings (:data:`STR`) are loaded from :file:`str{_lang}.yaml`
"""

from __future__ import annotations

import locale
import logging
import re
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from types import MappingProxyType

import yaml

from tcm_gui.const import LANG

_l = logging.getLogger(__name__)
_LANG_SEP = re.compile(r"[._\-]")
_LANGS = frozenset({"en", "ru"})  # actually supported translations


def _lang_code(raw: str | None) -> str:
    """First supported two-letter code from locale/setting text."""
    code = _LANG_SEP.split((raw or "").strip().casefold(), 1)[0][:2]
    return code if code in _LANGS else "en"


def _detect_raw() -> str:
    try:
        return locale.getlocale()[0] or ""
    except Exception:
        return ""


@cache
def resolve_lang() -> str:
    """Resolved two-letter language code."""
    return _lang_code(_detect_raw() if LANG == "auto" else LANG)


@cache
def load_str() -> Mapping[str, str]:
    """Cached i18n mapping loader."""

    lang = resolve_lang()
    root = Path(__file__).parent

    if not (path := root / f"str_{lang}.yaml").is_file():
        path = root / "str.yaml"

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a YAML mapping")

    if not all(isinstance(k, str) and isinstance(v, str) for k, v in data.items()):
        raise TypeError(f"{path} must contain only string keys and string values")

    _l.debug("i18n: loaded %s (lang=%s)", path.name, lang)
    return MappingProxyType(data)


STRINGS: Mapping[str, str] = load_str()
