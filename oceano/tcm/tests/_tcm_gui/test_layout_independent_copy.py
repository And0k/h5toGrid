"""Layout-independent shortcuts: physical Ctrl+letter keys must trigger their
Tk virtual events regardless of the active keyboard layout (Cyrillic, …).

On non-Latin layouts Tk's ``<<Copy>>`` / ``<<SelectAll>>`` / … never fire
because the physical key produces a different ``keysym``.
:class:`keyboard.LayoutIndependentShortcuts` detects the physical key by its
platform ``keycode`` (see :mod:`tcm_gui.keyboard`) and re-emits the virtual
event on the event widget.

These tests simulate non-Latin keypresses with synthetic ``<KeyPress>``
events whose ``keycode`` is the physical key but whose ``keysym`` is NOT the
Latin letter — exactly what Windows/X11 deliver on a Cyrillic layout.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import pytest

from tcm_gui.keyboard import LayoutIndependentShortcuts


def _keycode(ws: str, latin: str) -> int:
    """Physical *latin* letter's keycode on windowing system *ws*."""
    return next(kc for kc, (_, exp) in LayoutIndependentShortcuts._KEYCODES[ws].items() if exp == latin)


def _ctrl_event(widget: tk.Misc, keycode: int, keysym: str) -> tk.Event:
    ev = tk.Event()
    ev.keycode = keycode
    ev.keysym = keysym
    ev.state = 0x0004  # Control mask
    ev.widget = widget
    return ev


def _handler(ws: str = "win32") -> LayoutIndependentShortcuts:
    """Bare handler with a fixed keycode table (no Tk root needed)."""
    h = LayoutIndependentShortcuts.__new__(LayoutIndependentShortcuts)
    h._keycodes = LayoutIndependentShortcuts._KEYCODES[ws]
    return h


# ── unit tests for the handler logic ─────────────────────────────────────────


class TestOnKeyPress:
    @pytest.mark.parametrize(
        ("latin", "virtual"),
        [
            ("a", "<<SelectAll>>"),
            ("c", "<<Copy>>"),
            ("x", "<<Cut>>"),
            ("v", "<<Paste>>"),
            ("z", "<<Undo>>"),
            ("y", "<<Redo>>"),
            ("f", "<<Find>>"),
        ],
    )
    def test_non_latin_triggers_virtual(self, latin, virtual):
        """Physical key + non-Latin keysym → virtual event + ``'break'``."""
        h = _handler()
        target = tk.Text.__new__(tk.Text)
        generated: list[str] = []
        target.event_generate = lambda seq, **kw: generated.append(seq)  # type: ignore[method-assign]

        ret = h._on_key_press(_ctrl_event(target, _keycode("win32", latin), "Cyrillic_ef"))

        assert ret == "break"
        assert generated == [virtual]

    def test_latin_passes_through(self):
        """Latin keysym → None so Tk handles it natively (no double-fire)."""
        h = _handler()
        target = tk.Text.__new__(tk.Text)
        generated: list[str] = []
        target.event_generate = lambda seq, **kw: generated.append(seq)  # type: ignore[method-assign]

        assert h._on_key_press(_ctrl_event(target, _keycode("win32", "c"), "c")) is None
        assert h._on_key_press(_ctrl_event(target, _keycode("win32", "c"), "C")) is None
        assert generated == []

    def test_other_key_ignored(self):
        target = tk.Text.__new__(tk.Text)
        generated: list[str] = []
        target.event_generate = lambda seq, **kw: generated.append(seq)  # type: ignore[method-assign]

        ev = _ctrl_event(target, _keycode("win32", "c") + 1000, "Cyrillic_a")
        assert _handler()._on_key_press(ev) is None
        assert generated == []

    def test_no_control_ignored(self):
        h = _handler()
        target = tk.Text.__new__(tk.Text)
        generated: list[str] = []
        target.event_generate = lambda seq, **kw: generated.append(seq)  # type: ignore[method-assign]

        ev = _ctrl_event(target, _keycode("win32", "a"), "Cyrillic_ef")
        ev.state = 0  # no Control
        assert h._on_key_press(ev) is None
        assert generated == []

    def test_x11_table(self):
        """X11 keycodes route the same way (e.g. physical C = 54)."""
        h = _handler("x11")
        target = tk.Text.__new__(tk.Text)
        generated: list[str] = []
        target.event_generate = lambda seq, **kw: generated.append(seq)  # type: ignore[method-assign]

        assert _keycode("x11", "c") == 54
        assert h._on_key_press(_ctrl_event(target, 54, "Cyrillic_es")) == "break"
        assert generated == ["<<Copy>>"]


# ── integration: real Tk, real binding ───────────────────────────────────────


@pytest.mark.gui
class TestLayoutIndependentShortcutsIntegration:
    """End-to-end: synthetic non-Latin Ctrl+C reaches the copy handler."""

    @pytest.fixture(scope="class")
    @classmethod
    def _tk_root(cls):
        try:
            root = tk.Tk()
            root.withdraw()
            yield root
            root.destroy()
        except tk.TclError:
            yield None

    def test_keypress_binding_installed(self, _tk_root):
        """``LayoutIndependentShortcuts`` installs a ``bind_all("<KeyPress>")`` handler."""
        if _tk_root is None:
            pytest.skip("Tk unavailable")
        kbd = LayoutIndependentShortcuts(_tk_root)
        try:
            assert any("<KeyPress>" in b or "<Key>" in b for b in _tk_root.bind_all())
        finally:
            kbd.destroy()

    def test_non_latin_ctrl_c_reaches_copy_handler(self, _tk_root):
        """Synthetic non-Latin Ctrl+C on a focused Entry fires ``<<Copy>>``.

        Tk on Windows can't synthesize an event with a fake keysym (it needs a
        keysym that maps to a real keycode), so we invoke the bound handler
        directly with a mock event — exactly what Tk would do when the physical
        C key produces a non-Latin keysym on a Cyrillic/Greek layout.
        """
        if _tk_root is None:
            pytest.skip("Tk unavailable")

        from tcm_gui.app import App

        log = tk.Text(_tk_root, state="disabled")
        log.pack()
        copied: list[str] = []
        try:
            kbd = LayoutIndependentShortcuts(_tk_root)

            def _on_copy(ev):
                copied.append("copy")
                return None

            # Install handlers the way App does.
            app = App.__new__(App)
            app._log = log
            app._status_lbl = tk.Text(_tk_root)
            _tk_root.bind("<<Copy>>", app._on_copy_rich, add="+")
            _tk_root.bind("<<Copy>>", _on_copy, add="+")

            ent = ttk.Entry(_tk_root, width=20)
            ent.insert(0, "test-data")
            ent.pack()
            ent.focus_set()
            _tk_root.update()

            # Simulate Ctrl+C on a Cyrillic layout: physical C + Cyrillic keysym.
            ws = _tk_root.tk.call("tk", "windowingsystem")
            ev = _ctrl_event(ent, _keycode(ws, "c"), "Cyrillic_es")
            kbd._on_key_press(ev)
            _tk_root.update_idletasks()
            _tk_root.update()

            assert "copy" in copied, "<<Copy>> never fired for non-Latin Ctrl+C"
        finally:
            kbd.destroy()
            ent.destroy()
            log.destroy()
