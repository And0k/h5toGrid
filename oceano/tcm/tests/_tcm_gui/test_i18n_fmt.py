"""Headless regression for :func:`tcm_gui._i18n.fmt_status`.

Color markup (``{#name}…{/}``) shares braces with format fields — plain
``.format`` on a template combining both crashes with
``KeyError: '#name'`` (regression: user-colored
``time_ranges.hover.broader`` killed hover with ``KeyError:
'#sheet_warning'``). Genuine field typos must still raise.
"""

from __future__ import annotations

import pytest

from tcm_gui._i18n import fmt_status


def test_markup_spans_survive_format():
    out = fmt_status("шире чем в info_devices [{s}, {e}] — {#sheet_warning}проверьте{/}", s="a", e="b")
    assert out == "шире чем в info_devices [a, b] — {#sheet_warning}проверьте{/}", f"out={out!r}"


def test_plain_template_unchanged():
    assert fmt_status("Run: {p}", p="x") == "Run: x"
    assert fmt_status("no fields") == "no fields"


def test_genuine_typo_still_raises():
    with pytest.raises(KeyError):
        fmt_status("{s} — {e}", s="a")
