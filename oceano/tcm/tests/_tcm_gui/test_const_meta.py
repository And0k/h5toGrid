"""Tests for const.py: UI scaling, widget_meta registry, and STR content dict.

Covers :func:`apply_ui_scale`, :func:`set_widget_meta`, :func:`get_widget_meta`,
the :data:`widget_meta` dictionary, callable-resolving ``get_widget_meta``,
and the :data:`STR` i18n content table.
"""

from __future__ import annotations

import tkinter as tk
from unittest.mock import MagicMock

import pytest

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


class TestApplyUiScale:
    """``apply_ui_scale`` sets tk scaling and configures named fonts."""

    def test_sets_tk_scaling(self, _tk_root):
        """apply_ui_scale calls tk scaling with UI_SCALE."""
        if _tk_root is None:
            pytest.skip("Tk unavailable — Tcl interpreter already destroyed")
        from tcm_gui.const import UI_SCALE, apply_ui_scale

        apply_ui_scale(_tk_root)
        actual = _tk_root.tk.call("tk", "scaling")
        # Tk may nudge the value slightly (e.g. 1.0 → 1.00049… from DPI rounding).
        assert float(actual) == pytest.approx(UI_SCALE, abs=0.01), (
            f"apply_ui_scale: tk scaling mismatch — expected ~{UI_SCALE}, got {actual}"
        )

    def test_named_font_size(self, _tk_root):
        """apply_ui_scale sets TkDefaultFont size to FONT_SIZE."""
        if _tk_root is None:
            pytest.skip("Tk unavailable — Tcl interpreter already destroyed")
        import tkinter.font as tkfont

        from tcm_gui.const import FONT_SIZE, apply_ui_scale

        apply_ui_scale(_tk_root)
        default_font = tkfont.nametofont("TkDefaultFont")
        assert default_font.cget("size") == FONT_SIZE, (
            f"apply_ui_scale: TkDefaultFont size mismatch — expected {FONT_SIZE}, "
            f"got {default_font.cget('size')}"
        )

    def test_text_font_size(self, _tk_root):
        """apply_ui_scale sets TkTextFont size to FONT_SIZE."""
        if _tk_root is None:
            pytest.skip("Tk unavailable — Tcl interpreter already destroyed")
        import tkinter.font as tkfont

        from tcm_gui.const import FONT_SIZE, apply_ui_scale

        apply_ui_scale(_tk_root)
        text_font = tkfont.nametofont("TkTextFont")
        assert text_font.cget("size") == FONT_SIZE, (
            f"apply_ui_scale: TkTextFont size mismatch — expected {FONT_SIZE}, got {text_font.cget('size')}"
        )

    def test_calls_tk_scaling_with_value(self):
        """apply_ui_scale calls root.tk.call('tk', 'scaling', UI_SCALE) via mock."""
        from tcm_gui.const import UI_SCALE, apply_ui_scale

        mock_root = MagicMock()
        mock_root.tk.call.return_value = UI_SCALE
        apply_ui_scale(mock_root)
        mock_root.tk.call.assert_any_call("tk", "scaling", UI_SCALE)


# ── widget_meta registry ─────────────────────────────────────────────────────


class TestWidgetMeta:
    """``set_widget_meta`` / ``get_widget_meta`` store and retrieve metadata."""

    def setup_method(self):
        """Clear widget_meta before each test to prevent state leakage."""
        from tcm_gui.const import widget_meta

        widget_meta.clear()

    def test_set_and_get_by_widget(self):
        """set_widget_meta stores metadata by widget key (mock)."""
        from tcm_gui.const import get_widget_meta, set_widget_meta

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
        from tcm_gui.const import get_widget_meta, set_widget_meta

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
        from tcm_gui.const import get_widget_meta, set_widget_meta

        lbl = MagicMock(spec=tk.Label)
        set_widget_meta(lbl, status="Measured bottom temperature", tooltip="Temp in °C", help="...")
        assert get_widget_meta(lbl, "status") == "Measured bottom temperature", (
            "widget_meta: multi-key status mismatch"
        )
        assert get_widget_meta(lbl, "tooltip") == "Temp in °C", "widget_meta: multi-key tooltip mismatch"
        assert get_widget_meta(lbl, "help") == "...", "widget_meta: multi-key help mismatch"

    def test_overwrite_replaces_all_keys(self):
        """Second set_widget_meta call replaces the entire dict for that widget."""
        from tcm_gui.const import get_widget_meta, set_widget_meta

        btn = MagicMock(spec=tk.Button)
        set_widget_meta(btn, status="first")
        set_widget_meta(btn, tooltip="second")  # replaces — status is gone
        assert get_widget_meta(btn, "tooltip") == "second", "widget_meta: overwrite should store new keys"
        assert get_widget_meta(btn, "status", "GONE") == "GONE", (
            "widget_meta: overwrite should remove old keys"
        )

    def test_mixed_widget_and_string_keys(self):
        """widget_meta supports both widget instances and string keys simultaneously."""
        from tcm_gui.const import get_widget_meta, set_widget_meta

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
        from tcm_gui.const import get_widget_meta, set_widget_meta

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
        from tcm_gui.const import get_widget_meta, set_widget_meta

        lbl = MagicMock(spec=tk.Label)
        set_widget_meta(lbl, tooltip=lambda: "live tooltip")
        assert get_widget_meta(lbl, "tooltip") == "live tooltip", (
            "callable tooltip should be resolved at read"
        )

    def test_string_status_passthrough_unchanged(self):
        """Plain ``str`` status is returned verbatim — no invocation."""
        from tcm_gui.const import get_widget_meta, set_widget_meta

        lbl = MagicMock(spec=tk.Label)
        set_widget_meta(lbl, status="static text")
        assert get_widget_meta(lbl, "status") == "static text", "string status should pass through unchanged"

    def test_missing_callable_field_returns_default(self):
        """Default returned when a callable-stored widget lacks the queried key."""
        from tcm_gui.const import get_widget_meta, set_widget_meta

        btn = MagicMock(spec=tk.Button)
        set_widget_meta(btn, tooltip="btn tip")  # only tooltip, no status
        assert get_widget_meta(btn, "status", "fallback") == "fallback", (
            "missing callable key should return default string"
        )


# ── STR content dict (i18n surface) ─────────────────────────────────────────


class TestSTR:
    """``STR`` dict provides the stable i18n key surface for chrome widgets."""

    def test_has_path_field_keys(self):
        from tcm_gui.const import STR

        assert "path_field.tooltip" in STR, "path_field.tooltip missing from STR"
        assert "path_field.status" in STR, "path_field.status missing from STR"

    def test_has_run_keys(self):
        from tcm_gui.const import STR

        for key in ("run.tooltip", "run.start", "run.pause", "run.resume"):
            assert key in STR, f"{key} missing from STR"

    def test_has_tab_template(self):
        from tcm_gui.const import STR

        assert "tab.status" in STR, "tab.status template missing from STR"
        assert "{path}" in STR["tab.status"], (
            f"tab.status must contain {{path}} for format(); got {STR['tab.status']!r}"
        )

    def test_all_values_are_strings(self):
        from tcm_gui.const import STR

        non_str = {k: type(v).__name__ for k, v in STR.items() if not isinstance(v, str)}
        assert not non_str, f"STR values must all be str (content layer); non-str keys: {non_str}"
