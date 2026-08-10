"""Tests for const.py: UIScale, configure_ui, widget_meta registry, and STR content dict.

Covers :class:`UIScale`, :func:`configure_ui`, :func:`set_widget_meta`,
:func:`get_widget_meta`, the :data:`widget_meta` dictionary,
callable-resolving ``get_widget_meta``, and the :data:`STR` i18n content table.
"""

from __future__ import annotations

import tkinter as tk
from unittest.mock import MagicMock

import pytest
from tcm_gui.const import set_widget_meta
import tcm_gui.theme

# ── UI scaling ────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def _tk_root():
    """Module-scoped Tk root for scaling tests.

    If a Tk root was already created and destroyed in this process session,
    ``tk.Tk()`` will raise ``TclError`` — return ``None`` so tests skip.
    """
    try:
        root = tk.Tk()
        root.withdraw()
        yield root
        root.destroy()
    except tk.TclError:
        yield None


class TestUIScale:
    """UIScale sets tk scaling and configures named fonts."""

    def test_sets_tk_scaling(self, _tk_root):
        """UIScale(ui_scale=1.5) sets tk scaling to platform × 1.5."""
        if _tk_root is None:
            pytest.skip("Tk unavailable — Tcl interpreter already destroyed")
        from tcm_gui.const import UIScale

        platform = _tk_root.tk.call("tk", "scaling")
        UIScale(_tk_root, ui_scale=1.5)
        actual = _tk_root.tk.call("tk", "scaling")
        expected = platform * 1.5
        assert actual == pytest.approx(expected, abs=0.01), (
            f"tk scaling after UIScale(ui_scale=1.5): expected ~{expected}, got {actual}"
        )

    def test_font_returns_copy(self, _tk_root):
        """font() returns a copy, not the original named font."""
        if _tk_root is None:
            pytest.skip("Tk unavailable — Tcl interpreter already destroyed")
        import tkinter.font as tkfont

        from tcm_gui.const import UIScale

        ui = UIScale(_tk_root, font_scale=1.0)
        f = ui.font()
        original = tkfont.nametofont("TkDefaultFont")
        assert f is not original, "font() should return a copy, not the original"
        assert f.cget("family") == original.cget("family"), (
            f"font() family mismatch: expected {original.cget('family')}, got {f.cget('family')}"
        )

    def test_font_identity_at_default(self, _tk_root):
        """font() with font_scale=1.0 returns copy with same size."""
        if _tk_root is None:
            pytest.skip("Tk unavailable — Tcl interpreter already destroyed")
        import tkinter.font as tkfont

        from tcm_gui.const import UIScale

        ui = UIScale(_tk_root, font_scale=1.0)
        f = ui.font()
        original_size = tkfont.nametofont("TkDefaultFont").cget("size")
        assert f.cget("size") == original_size, (
            f"font() size at scale=1.0: expected {original_size}, got {f.cget('size')}"
        )

    def test_font_scales_named_font(self, _tk_root):
        """UIScale(font_scale=1.2) configures named fonts to 1.2× platform size."""
        if _tk_root is None:
            pytest.skip("Tk unavailable — Tcl interpreter already destroyed")
        import tkinter.font as tkfont

        from tcm_gui.const import UIScale

        names = ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont")
        saved = {n: tkfont.nametofont(n).cget("size") for n in names}
        try:
            ui = UIScale(_tk_root, font_scale=1.2)
            default = tkfont.nametofont("TkDefaultFont")
            expected = round(saved["TkDefaultFont"] * 1.2)
            assert default.cget("size") == expected, (
                f"TkDefaultFont at scale=1.2: expected {expected}, got {default.cget('size')}"
            )
            f = ui.font()
            assert f.cget("size") == expected, (
                f"font() copy at scale=1.2: expected {expected}, got {f.cget('size')}"
            )
        finally:
            for n, s in saved.items():
                tkfont.nametofont(n).configure(size=s)

    def test_font_size_diff_after_scale(self, _tk_root):
        """size_diff is applied after font_scale."""
        if _tk_root is None:
            pytest.skip("Tk unavailable — Tcl interpreter already destroyed")
        import tkinter.font as tkfont

        from tcm_gui.const import UIScale

        ui = UIScale(_tk_root, font_scale=1.0)
        f = ui.font(size_diff=2)
        original_size = tkfont.nametofont("TkDefaultFont").cget("size")
        assert f.cget("size") == original_size + 2, (
            f"font(size_diff=2) at scale=1.0: expected {original_size + 2}, got {f.cget('size')}"
        )


class TestSetFont:
    """UIScale.set_font applies the scaled TkDefaultFont to widgets."""

    def test_text_receives_default_font_not_fixed(self, _tk_root):
        """set_font overrides tk.Text implicit TkFixedFont with TkDefaultFont."""
        if _tk_root is None:
            pytest.skip("Tk unavailable — Tcl interpreter already destroyed")
        import tkinter.font as tkfont

        from tcm_gui.const import UIScale

        ui = UIScale(_tk_root, font_scale=1.0)
        txt = tk.Text(_tk_root)
        # Before set_font: tk.Text defaults to TkFixedFont.
        default_family = tkfont.nametofont("TkDefaultFont").cget("family")
        fixed_family = tkfont.nametofont("TkFixedFont").cget("family")
        assert txt.cget("font") == "TkFixedFont", (
            f"fresh tk.Text font should be 'TkFixedFont', got {txt.cget('font')!r}"
        )
        # Apply the GUI font.
        ui.set_font(txt)
        applied = tkfont.Font(font=txt.cget("font"))
        assert applied.cget("family") == default_family, (
            f"set_font: expected family {default_family!r}, got {applied.cget('family')!r}"
        )
        assert applied.cget("family") != fixed_family, (
            f"set_font should override TkFixedFont ({fixed_family!r}), still got it"
        )
        txt.destroy()

    def test_set_font_gives_each_widget_own_copy(self, _tk_root):
        """Two widgets get independent Font objects — mutating one is safe."""
        if _tk_root is None:
            pytest.skip("Tk unavailable — Tcl interpreter already destroyed")
        import tkinter.font as tkfont

        from tcm_gui.const import UIScale

        ui = UIScale(_tk_root, font_scale=1.0)
        t1 = tk.Text(_tk_root)
        t2 = tk.Text(_tk_root)
        ui.set_font(t1, t2)
        f1 = tkfont.Font(font=t1.cget("font"))
        f2 = tkfont.Font(font=t2.cget("font"))
        # Different objects (each gets its own copy).
        assert str(f1) != str(f2), "set_font should give each widget its own Font object"
        # Mutating one must not affect the other.
        orig_size = f2.cget("size")
        f1.configure(size=6)
        assert f2.cget("size") == orig_size, (
            f"mutating t1 font should not affect t2: expected size {orig_size}, got {f2.cget('size')}"
        )
        t1.destroy()
        t2.destroy()

    def test_set_font_size_diff(self, _tk_root):
        """set_font(size_diff=-1) shrinks the applied font by 1 point."""
        if _tk_root is None:
            pytest.skip("Tk unavailable — Tcl interpreter already destroyed")
        import tkinter.font as tkfont

        from tcm_gui.const import UIScale

        ui = UIScale(_tk_root, font_scale=1.0)
        base_size = tkfont.nametofont("TkDefaultFont").cget("size")
        txt = tk.Text(_tk_root)
        ui.set_font(txt, size_diff=-1)
        applied = tkfont.Font(font=txt.cget("font"))
        assert applied.cget("size") == base_size - 1, (
            f"set_font(size_diff=-1): expected {base_size - 1}, got {applied.cget('size')}"
        )
        txt.destroy()


class TestLogStatusFontMatch:
    """Reproduce the actual App._build() font setup and verify _log ≡ _status_lbl."""

    def test_log_and_status_share_font_family_and_size(self, _tk_root):
        """_log (via set_font) and MarkdownLabel (via font=ui.font()) must match."""
        if _tk_root is None:
            pytest.skip("Tk unavailable — Tcl interpreter already destroyed")
        import tkinter.font as tkfont

        from tcm_gui.const import UIScale
        from tcm_gui.md_label import MarkdownLabel

        ui = UIScale(_tk_root, font_scale=1.0)

        # Replicate App._build() §5 + §6 exactly.
        log = tk.Text(_tk_root, wrap="word")
        ui.set_font(log)

        status = MarkdownLabel(_tk_root, font=ui.font())

        log_font = tkfont.Font(font=log.cget("font"))
        status_font = status._fonts["plain"]  # the base font for rendered text

        assert log_font.cget("family") == status_font.cget("family"), (
            f"family mismatch — log={log_font.cget('family')!r}, status={status_font.cget('family')!r}"
        )
        assert log_font.cget("size") == status_font.cget("size"), (
            f"size mismatch — log={log_font.cget('size')}, status={status_font.cget('size')}"
        )
        log.destroy()
        status.destroy()

    def test_fonts_match_after_mark_font_ready_lifecycle(self, _tk_root):
        """Fonts stay matched after mark_font_ready + rerender (the real _fit_status_font flow)."""
        if _tk_root is None:
            pytest.skip("Tk unavailable — Tcl interpreter already destroyed")
        import tkinter.font as tkfont

        from tcm_gui.const import UIScale
        from tcm_gui.md_label import MarkdownLabel

        ui = UIScale(_tk_root, font_scale=1.0)

        log = tk.Text(_tk_root, wrap="word")
        ui.set_font(log)

        status = MarkdownLabel(_tk_root, font=ui.font())
        # Replicate _fit_status_font lifecycle: mark_font_ready + rerender.
        status.mark_font_ready()
        status.set_text("Ready", raw=True)

        log_font = tkfont.Font(font=log.cget("font"))
        status_font = status._fonts["plain"]

        assert log_font.cget("family") == status_font.cget("family"), (
            f"post-lifecycle family mismatch — log={log_font.cget('family')!r}, "
            f"status={status_font.cget('family')!r}"
        )
        assert log_font.cget("size") == status_font.cget("size"), (
            f"post-lifecycle size mismatch — log={log_font.cget('size')}, status={status_font.cget('size')}"
        )
        log.destroy()
        status.destroy()


class TestConfigureUI:
    """configure_ui selects ttk theme based on TTK_THEME policy."""

    def test_sets_clam_when_policy_clam(self, _tk_root, monkeypatch):
        """configure_ui sets clam when TTK_THEME='clam'."""
        if _tk_root is None:
            pytest.skip("Tk unavailable — Tcl interpreter already destroyed")
        import tcm_gui.const as const_mod
        from tcm_gui.const import configure_ui

        monkeypatch.setattr(const_mod, "TTK_THEME", "clam")
        style = configure_ui(_tk_root)
        assert style.theme_use() == "clam", (
            f"configure_ui with TTK_THEME='clam': expected 'clam', got {style.theme_use()!r}"
        )

    def test_sets_vista_when_policy_native(self, _tk_root, monkeypatch):
        """configure_ui sets vista on Windows when TTK_THEME='native'."""
        if _tk_root is None:
            pytest.skip("Tk unavailable — Tcl interpreter already destroyed")
        import sys

        import tcm_gui.const as const_mod
        from tcm_gui.const import configure_ui

        monkeypatch.setattr(const_mod, "TTK_THEME", "native")
        style = configure_ui(_tk_root)
        if sys.platform == "win32":
            assert style.theme_use() == "vista", (
                f"configure_ui with TTK_THEME='native' on Windows: expected 'vista', "
                f"got {style.theme_use()!r}"
            )


# ── widget_meta registry ─────────────────────────────────────────────────────


class TestWidgetMeta:
    """``set_widget_meta`` / ``get_widget_meta`` store and retrieve metadata."""

    def setup_method(self):
        """Clear widget_meta before each test to prevent state leakage."""
        from tcm_gui.const import widget_meta

        widget_meta.clear()

    def test_set_and_get_by_widget(self):
        """set_widget_meta stores metadata by widget key (mock)."""
        from tcm_gui.const import get_widget_meta

        btn = MagicMock(spec=tk.Button)
        set_widget_meta(btn, status="Click me", tooltip="A button")
        assert get_widget_meta(btn, "status") == "Click me", (
            "widget_meta: status mismatch for widget instance"
        )
        assert get_widget_meta(btn, "tooltip") == "A button", (
            "widget_meta: tooltip mismatch for widget instance"
        )

    def test_set_and_get_by_string(self):
        """set_widget_meta stores metadata by string identifier."""
        from tcm_gui.const import get_widget_meta

        set_widget_meta("input.coefs_path", status="Path to coefficients file")
        assert get_widget_meta("input.coefs_path", "status") == "Path to coefficients file", (
            "widget_meta: status mismatch for string identifier"
        )

    def test_missing_key_returns_default(self):
        """get_widget_meta returns default for missing key."""
        from tcm_gui.const import get_widget_meta

        assert get_widget_meta("nonexistent", "tooltip", "fallback") == "fallback", (
            "widget_meta: missing key should return default"
        )

    def test_missing_widget_returns_default(self):
        """get_widget_meta returns default for unknown widget."""
        from tcm_gui.const import get_widget_meta

        assert get_widget_meta("no_such_widget", "status") == "", (
            "widget_meta: missing widget should return empty string default"
        )

    def test_multiple_keys_per_widget(self):
        """set_widget_meta supports multiple keys per widget."""
        from tcm_gui.const import get_widget_meta

        lbl = MagicMock(spec=tk.Label)
        set_widget_meta(lbl, status="Measured bottom temperature", tooltip="Temp in °C", help="...")
        assert get_widget_meta(lbl, "status") == "Measured bottom temperature", (
            "widget_meta: multi-key status mismatch"
        )
        assert get_widget_meta(lbl, "tooltip") == "Temp in °C", "widget_meta: multi-key tooltip mismatch"
        assert get_widget_meta(lbl, "help") == "...", "widget_meta: multi-key help mismatch"

    def test_overwrite_replaces_all_keys(self):
        """Second set_widget_meta call replaces the entire dict for that widget."""
        from tcm_gui.const import get_widget_meta

        btn = MagicMock(spec=tk.Button)
        set_widget_meta(btn, status="first")
        set_widget_meta(btn, tooltip="second")  # replaces — status is gone
        assert get_widget_meta(btn, "tooltip") == "second", "widget_meta: overwrite should store new keys"
        assert get_widget_meta(btn, "status", "GONE") == "GONE", (
            "widget_meta: overwrite should remove old keys"
        )

    def test_mixed_widget_and_string_keys(self):
        """widget_meta supports both widget instances and string keys simultaneously."""
        from tcm_gui.const import get_widget_meta

        btn = MagicMock(spec=tk.Button)
        set_widget_meta(btn, status="Widget status")
        set_widget_meta("input.path", status="String status")
        assert get_widget_meta(btn, "status") == "Widget status", (
            "widget_meta: mixed keys — widget status mismatch"
        )
        assert get_widget_meta("input.path", "status") == "String status", (
            "widget_meta: mixed keys — string status mismatch"
        )

    def test_callable_status_resolved_at_read_time(self):
        """get_widget_meta resolves a ``Callable[[], str]`` status live."""
        from tcm_gui.const import get_widget_meta

        state = {"n": 0}

        def counter() -> str:
            state["n"] += 1
            return f"status-{state['n']}"

        btn = MagicMock(spec=tk.Button)
        set_widget_meta(btn, status=counter)
        assert get_widget_meta(btn, "status") == "status-1", "callable status should be invoked once per read"
        assert get_widget_meta(btn, "status") == "status-2", (
            "second read should invoke callable again (live state)"
        )

    def test_callable_tooltip_resolved(self):
        """``Callable`` value also works for ``tooltip`` — resolved at read time."""
        from tcm_gui.const import get_widget_meta

        lbl = MagicMock(spec=tk.Label)
        set_widget_meta(lbl, tooltip=lambda: "live tooltip")
        assert get_widget_meta(lbl, "tooltip") == "live tooltip", (
            "callable tooltip should be resolved at read"
        )

    def test_string_status_passthrough_unchanged(self):
        """Plain ``str`` status is returned verbatim — no invocation."""
        from tcm_gui.const import get_widget_meta

        lbl = MagicMock(spec=tk.Label)
        set_widget_meta(lbl, status="static text")
        assert get_widget_meta(lbl, "status") == "static text", "string status should pass through unchanged"

    def test_missing_callable_field_returns_default(self):
        """Default returned when a callable-stored widget lacks the queried key."""
        from tcm_gui.const import get_widget_meta

        btn = MagicMock(spec=tk.Button)
        set_widget_meta(btn, tooltip="btn tip")  # only tooltip, no status
        assert get_widget_meta(btn, "status", "fallback") == "fallback", (
            "missing callable key should return default string"
        )


# ── Theme detection ─────────────────────────────────────────────────────────


class TestThemeDetection:
    """``_detect_windows_theme`` returns 'dark' or 'light'; ``apply_theme_defaults`` mutates colors."""

    def test_detect_returns_valid_theme(self):
        from tcm_gui.theme import _detect_windows_theme

        result = _detect_windows_theme()
        assert result in ("dark", "light"), f"_detect_windows_theme()={result!r}, expected 'dark' or 'light'"

    def test_apply_dark_configures_ttk_style(self, _tk_root, monkeypatch):
        if _tk_root is None:
            pytest.skip("Tk unavailable")
        from tkinter import ttk

        from tcm_gui import const
        from tcm_gui.theme import apply_theme_defaults

        # Force dark mode regardless of actual system theme.
        monkeypatch.setattr(tcm_gui.theme, "_detect_windows_theme", lambda: "dark")
        apply_theme_defaults(_tk_root)
        style = ttk.Style()
        # Dark mode switches to "clam" (native themes ignore style configure).
        assert style.theme_use() == "clam", f"ttk theme={style.theme_use()!r}, expected 'clam' for dark mode"
        assert style.lookup("TFrame", "background") == tcm_gui.theme.FRAME_BG_FALLBACK, (
            f"TFrame bg={style.lookup('TFrame', 'background')!r} ≠ {tcm_gui.theme.FRAME_BG_FALLBACK!r}"
        )
        assert style.lookup("TEntry", "fieldbackground") == tcm_gui.theme.ENTRY_BG_FALLBACK, (
            f"TEntry fieldbg={style.lookup('TEntry', 'fieldbackground')!r} ≠ {tcm_gui.theme.ENTRY_BG_FALLBACK!r}"
        )
        assert style.lookup("TLabel", "foreground") == tcm_gui.theme.FG_DEFAULT, (
            f"TLabel fg={style.lookup('TLabel', 'foreground')!r} ≠ {tcm_gui.theme.FG_DEFAULT!r}"
        )
