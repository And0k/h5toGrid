"""TDD tests for MarkdownLabel status bar behavior.

Verifies:
1. Status label contracts to text width (short text → narrow, long → window)
2. Empty content → width ≈ 0
3. Long content is fully visible (bbox(last) not None)
4. Height adapts when window resizes
"""

from __future__ import annotations

import sys
import time

import pytest
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk

_mod = sys.modules[__name__]
_mod._root = None


@pytest.fixture(autouse=True, scope="module")
def _tk_root():
    try:
        r = tk.Tk()
        r.withdraw()
    except tk.TclError:
        pytest.skip("Tk not available")
        return
    _mod._root = r
    yield
    r.destroy()
    _mod._root = None


def _wait_settled(root, lbl, timeout_ms=300):
    """Wait for deferred _fit_width / _fit_height steps to complete."""
    deadline = time.monotonic() + timeout_ms / 1000
    while time.monotonic() < deadline:
        root.update()
        time.sleep(0.01)
    root.update()


def _build_status_bar(root):
    """Build §6 layout: label overlaid bottom-left on root (no f4)."""
    from tcm_gui.const import UIScale, configure_ui
    from tcm_gui.md_label import MarkdownLabel

    for w in root.winfo_children():
        w.destroy()

    root.geometry("1100x800")
    root.deiconify()
    root.attributes("-alpha", 0)  # invisible but geometry resolves

    ui = UIScale(root, font_scale=1.0)
    configure_ui(root)

    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(2, weight=2)
    root.grid_rowconfigure(4, weight=1)

    lbl = MarkdownLabel(root, font=ui.font())
    lbl.place(rely=1.0, relx=0.0, anchor="sw", x=4, y=-4)

    # Standalone progress bar for font-scaling measurement.
    prog = ttk.Progressbar(root, mode="determinate", length=220)
    prog.place(relx=1.0, rely=1.0, anchor="se", x=-8, y=-4)

    for _ in range(20):
        root.update()

    bar_h = prog.winfo_reqheight()
    if bar_h > 4:
        lbl.fit_to_height(bar_h)

    lbl.set_text("Ready", raw=True)
    _wait_settled(root, lbl)

    root.withdraw()
    root.attributes("-alpha", 1)

    return lbl, prog


class TestContraction:
    """Label width contracts to text content."""

    def test_short_text_narrower_than_window(self):
        root = _mod._root
        lbl, _ = _build_status_bar(root)
        win_w = root.winfo_width()
        lbl_w = lbl.winfo_width()
        assert lbl_w < win_w // 2, f"short text: label {lbl_w}px should be < half window {win_w}px"

    def test_empty_collapses(self):
        root = _mod._root
        lbl, _ = _build_status_bar(root)
        lbl.set_text("")
        _wait_settled(root, lbl)
        assert lbl.winfo_width() <= 2, f"empty label should collapse, got {lbl.winfo_width()}px"

    def test_re_show_after_clear(self):
        """Hover show → clear → hover show again must not stay collapsed."""
        root = _mod._root
        lbl, _ = _build_status_bar(root)
        lbl.set_text("")
        _wait_settled(root, lbl)
        assert lbl.winfo_width() <= 2
        # Re-show (simulates second hover).
        lbl.set_text("Hover hint text", raw=True)
        _wait_settled(root, lbl)
        assert lbl.winfo_width() > 10, f"re-show after clear: label {lbl.winfo_width()}px should be > 10px"

    def test_long_text_fills_window(self):
        root = _mod._root
        lbl, _ = _build_status_bar(root)
        lbl.set_text("word " * 50)
        _wait_settled(root, lbl)
        win_w = root.winfo_width()
        lbl_w = lbl.winfo_width()
        assert lbl_w >= win_w - 20, f"long text: label {lbl_w}px should be ≈ window {win_w}px"


class TestContentNotCut:
    LONG_HELP = (
        "Source of truth for time window. Explicit intervals "
        "[start, end, ...]. Auto-populated from first/last data "
        "row on initial generation."
    )

    def test_long_help_visible(self):
        root = _mod._root
        lbl, _ = _build_status_bar(root)
        lbl.set_text(self.LONG_HELP)
        _wait_settled(root, lbl)
        last = lbl.index("end-1c")
        bbox = lbl.bbox(last)
        assert bbox is not None, (
            f"last char CUT: h={lbl.cget('height')}, px={lbl.winfo_width()}x{lbl.winfo_height()}"
        )

    def test_multi_paragraph_visible(self):
        root = _mod._root
        lbl, _ = _build_status_bar(root)
        lbl.set_text("First.\n\nSecond.\n\nThird.")
        _wait_settled(root, lbl)
        last = lbl.index("end-1c")
        bbox = lbl.bbox(last)
        assert bbox is not None, f"last char CUT: h={lbl.cget('height')}"

    def test_height_grows_for_wrapping_text(self):
        root = _mod._root
        lbl, _ = _build_status_bar(root)
        lbl.set_text("**" + "word " * 50 + "**")
        _wait_settled(root, lbl)
        wh = lbl.winfo_height()
        if wh <= 1:
            pytest.skip("withdrawn window — dlineinfo unavailable")
        assert wh > lbl._fonts["plain"].metrics("linespace")


class TestResizeAdapts:
    def test_narrow_window_grows_height(self):
        root = _mod._root
        lbl, _ = _build_status_bar(root)

        medium = "word " * 25
        lbl.set_text(medium)
        _wait_settled(root, lbl)
        h_wide = lbl.winfo_height()
        if h_wide <= 1:
            pytest.skip("withdrawn window — geometry unavailable")

        root.geometry("600x800")
        lbl.rerender()  # re-render at new width (same content, different layout)
        _wait_settled(root, lbl)
        h_narrow = lbl.winfo_height()
        if h_narrow <= 1:
            pytest.skip("withdrawn window — geometry not propagated")

        root.geometry("1100x800")
        assert h_narrow >= h_wide, f"narrow({h_narrow}) < wide({h_wide})"


class TestStatusLinks:
    """Markdown links in status-bar text are rendered as clickable link spans."""

    def test_link_tag_and_url_recorded(self):
        root = _mod._root
        lbl, _ = _build_status_bar(root)
        lbl.set_text("Ожидаемая структура ([подробнее](io_formats.md#directory-layout))")
        content = lbl.get("1.0", "end-1c")
        assert content == "Ожидаемая структура (подробнее)", f"url must not render: {content!r}"
        pos = content.index("подробнее")
        assert "link" in lbl.tag_names(f"1.{pos}"), (
            f"link tag missing at link span: {lbl.tag_names(f'1.{pos}')!r}"
        )
        assert len(lbl._links) == 1, f"recorded links: {lbl._links!r}"
        _start, _end, url = lbl._links[0]
        assert url == "io_formats.md#directory-layout", f"recorded url: {url!r}"

    def test_plain_text_no_link_tag(self):
        root = _mod._root
        lbl, _ = _build_status_bar(root)
        lbl.set_text("Ready", raw=True)
        assert "link" not in lbl.tag_names("1.0"), f"unexpected link tag: {lbl.tag_names('1.0')!r}"
