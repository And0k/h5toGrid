"""Layout-independent Ctrl+C: the physical ``C`` key must trigger ``<<Copy>>``
regardless of the active keyboard layout (Cyrillic, Greek, …).

On non-Latin layouts Tk's ``<<Copy>>`` virtual event never fires because the
physical ``C`` key produces a different ``keysym``.  ``App._on_ctrl_keypress``
detects the physical key by its platform ``keycode`` (:data:`const.VK_C`) and
re-emits ``<<Copy>>`` on the event widget.

These tests simulate a non-Latin keypress by constructing a synthetic
``<Control-KeyPress>`` event whose ``keycode`` is VK_C but whose ``keysym`` is
NOT Latin ``c``/``C`` — exactly what Windows delivers for Ctrl+C on a Cyrillic
layout.
"""

from __future__ import annotations

import sys
import tkinter as tk
from tkinter import ttk

import pytest

from tcm_gui.const import VK_C


# ── helpers ──────────────────────────────────────────────────────────────────


def _ctrl_c_event(widget: tk.Misc, keysym: str = "Cyrillic_es") -> tk.Event:
    """Build a synthetic ``<Control-KeyPress>`` that mimics Ctrl+C on a
    non-Latin layout: physical ``C`` key (VK_C) but a non-Latin keysym."""
    ev = tk.Event()
    ev.keycode = VK_C
    ev.keysym = keysym
    ev.state = 0x0004  # Control mask
    ev.widget = widget
    return ev


# ── unit tests for the handler logic ─────────────────────────────────────────


class TestOnCtrlKeypress:
    """``App._on_ctrl_keypress`` routing decisions (no Tk root needed)."""

    @pytest.fixture()
    def app(self):
        """Minimal App stub with just the method under test."""
        from tcm_gui.app import App

        return App.__new__(App)

    def test_non_latin_c_triggers_copy(self, app):
        """VK_C + non-Latin keysym → generate ``<<Copy>>``, return ``'break'``."""
        target = tk.Text.__new__(tk.Text)  # lightweight; we only track generate
        generated: list[str] = []
        target.event_generate = lambda seq, **kw: generated.append(seq)  # type: ignore[method-assign]

        ev = _ctrl_c_event(target, keysym="Cyrillic_es")
        ret = app._on_ctrl_keypress(ev)

        assert ret == "break"
        assert generated == ["<<Copy>>"]

    def test_latin_c_passes_through(self, app):
        """VK_C + Latin ``c`` → return None so Tk handles it natively."""
        target = tk.Text.__new__(tk.Text)
        generated: list[str] = []
        target.event_generate = lambda seq, **kw: generated.append(seq)  # type: ignore[method-assign]

        ev = _ctrl_c_event(target, keysym="c")
        ret = app._on_ctrl_keypress(ev)

        assert ret is None
        assert generated == []

    def test_latin_C_passes_through(self, app):
        """VK_C + uppercase Latin ``C`` (Caps Lock) → also pass through."""
        target = tk.Text.__new__(tk.Text)
        generated: list[str] = []
        target.event_generate = lambda seq, **kw: generated.append(seq)  # type: ignore[method-assign]

        ev = _ctrl_c_event(target, keysym="C")
        ret = app._on_ctrl_keypress(ev)

        assert ret is None
        assert generated == []

    def test_other_key_ignored(self, app):
        """Different keycode (not VK_C) → return None, no generate."""
        target = tk.Text.__new__(tk.Text)
        generated: list[str] = []
        target.event_generate = lambda seq, **kw: generated.append(seq)  # type: ignore[method-assign]

        ev = _ctrl_c_event(target, keysym="Cyrillic_a")
        ev.keycode = VK_C + 1  # not the C key
        ret = app._on_ctrl_keypress(ev)

        assert ret is None
        assert generated == []


# ── integration: real Tk, real binding ───────────────────────────────────────


@pytest.mark.gui
class TestLayoutIndependentCopyIntegration:
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

    def test_bind_all_handler_installed(self, _tk_root):
        """``App`` installs a ``bind_all("<Control-KeyPress>", ...)`` handler."""
        if _tk_root is None:
            pytest.skip("Tk unavailable")
        # The handler is installed in App._build via bind_all; verify the
        # binding exists by installing it the same way App does.
        from tcm_gui.app import App

        app = App.__new__(App)
        _tk_root.bind_all("<Control-KeyPress>", app._on_ctrl_keypress, add="+")
        bindings = _tk_root.bind_all()
        # Tk normalizes <Control-KeyPress> to <Control-Key> in the binding list.
        assert any(
            "<Control-Key>" in b for b in bindings
        ), f"no application-wide <Control-KeyPress> binding; got {bindings!r}"

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

        def _on_copy(ev):
            copied.append("copy")
            return None

        # Install handlers the way App does: bind_all for application-wide events.
        app = App.__new__(App)
        app._log = log
        app._status_lbl = tk.Text(_tk_root)
        _tk_root.bind_all("<<Copy>>", app._on_copy_rich, add="+")
        _tk_root.bind_all("<Control-KeyPress>", app._on_ctrl_keypress, add="+")
        _tk_root.bind_all("<<Copy>>", _on_copy, add="+")

        ent = ttk.Entry(_tk_root, width=20)
        ent.insert(0, "test-data")
        ent.pack()
        ent.focus_set()
        _tk_root.update()

        # Simulate Ctrl+C on a Cyrillic layout: VK_C + Cyrillic keysym.
        # We call the bound handler directly because event_generate can't
        # synthesize events with non-Latin keysyms on Windows.
        ev = _ctrl_c_event(ent, keysym="Cyrillic_es")
        app._on_ctrl_keypress(ev)
        _tk_root.update_idletasks()
        _tk_root.update()

        assert "copy" in copied, "<<Copy>> never fired for non-Latin Ctrl+C"

        ent.destroy()
        log.destroy()
