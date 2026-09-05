"""Layout-independent Ctrl+letter shortcuts (``keyboard.py``).

Tk's ordinary bindings such as ``<Control-a>`` / ``<Control-c>`` match the
translated keysym, so on a non-Latin layout (Cyrillic, Greek, …) the physical
A/C/V/X/Z/Y/F key produces another keysym and the normal binding never fires.

:class:`LayoutIndependentShortcuts` fixes this application-wide: a single
``bind_all("<KeyPress>")`` handler (``add="+"`` — existing bindings kept)
detects the physical key by ``event.keycode`` and re-emits the semantic Tk
virtual event on the focused widget, which then performs its own operation:

    ============= ================= ======================================
    physical key  virtual event   handled by
    ============= ================= ======================================
    Ctrl+A        ``<<SelectAll>>``  Entry / Text / ScrolledText / tksheet
    Ctrl+C        ``<<Copy>>``       Entry / Text / tksheet + App RTF hook
    Ctrl+X        ``<<Cut>>``        Entry / Text / tksheet
    Ctrl+V        ``<<Paste>>``      Entry / Text / tksheet
    Ctrl+Z        ``<<Undo>>``       Text (``undo=True``) / tksheet
    Ctrl+Y        ``<<Redo>>``       Text (``undo=True``) / tksheet
    Ctrl+F        ``<<Find>>``       app-level hook (no default Tk target)
    ============= ================= ======================================

``<<Find>>`` is a custom virtual event — plain Tk widgets ignore it; the app
may bind it later. Emitting it here keeps the physical-key detection in one
place.

English layouts pass through untouched (``keysym`` already Latin → return
``None`` so the widget's native binding runs — no double-fire). Non-English
layouts get the virtual event plus ``"break"`` so the translated keysym does
not leak into ordinary bindings.

Platform notes — ``event.keycode`` is the platform keycode, not a Tk
abstraction (X11 values are the standard-PC ``xev`` codes)::

    win32:  A=0x41 C=0x43 F=0x46 V=0x56 X=0x58 Y=0x59 Z=0x5A (Windows VK)
    x11:    A=38 C=54 F=41 V=55 X=53 Y=29 Z=52

Only ``win32`` and ``x11`` are supported; anything else raises instead of
installing wrong bindings.
"""

from __future__ import annotations

import tkinter as tk


class LayoutIndependentShortcuts:
    """See module docstring — install once per Tk root, ``destroy()`` to remove."""

    # Tk's state bit for the Control modifier.
    _CONTROL_MASK = 0x0004

    # windowing system → keycode → (virtual event, expected Latin keysym).
    _KEYCODES: dict[str, dict[int, tuple[str, str]]] = {
        "win32": {
            0x41: ("<<SelectAll>>", "a"),  # VK_A
            0x43: ("<<Copy>>", "c"),  # VK_C
            0x46: ("<<Find>>", "f"),  # VK_F
            0x56: ("<<Paste>>", "v"),  # VK_V
            0x58: ("<<Cut>>", "x"),  # VK_X
            0x59: ("<<Redo>>", "y"),  # VK_Y
            0x5A: ("<<Undo>>", "z"),  # VK_Z
        },
        "x11": {
            38: ("<<SelectAll>>", "a"),  # physical A
            54: ("<<Copy>>", "c"),  # physical C
            41: ("<<Find>>", "f"),  # physical F
            55: ("<<Paste>>", "v"),  # physical V
            53: ("<<Cut>>", "x"),  # physical X
            29: ("<<Redo>>", "y"),  # physical Y
            52: ("<<Undo>>", "z"),  # physical Z
        },
    }

    def __init__(self, root: tk.Misc) -> None:
        """Install the handler on the ``all`` bind tag (``add="+"`` appends)."""
        self.root = root
        self.windowing_system = root.tk.call("tk", "windowingsystem")

        try:
            self._keycodes = self._KEYCODES[self.windowing_system]
        except KeyError as exc:
            raise RuntimeError(f"Unsupported Tk windowing system: {self.windowing_system!r}") from exc

        self._funcid = root.bind_all("<KeyPress>", self._on_key_press, add="+")

    def _on_key_press(self, event: tk.Event) -> str | None:
        """Translate physical Ctrl+letter into the widget's virtual event."""
        # Ignore key presses without Control held.
        if not event.state & self._CONTROL_MASK:
            return None

        if (hit := self._keycodes.get(event.keycode)) is None:
            return None
        virtual_event, expected_keysym = hit

        # English layout: native binding already handles it — don't double-fire.
        if event.keysym.lower() == expected_keysym:
            return None

        # Non-English layout: invoke the widget's semantic operation, swallow
        # the translated keysym so no ordinary binding sees it.
        event.widget.event_generate(virtual_event)
        return "break"

    def destroy(self) -> None:
        """Remove only this instance's application-wide binding."""
        if self._funcid is not None:
            self.root.unbind_all("<KeyPress>", self._funcid)
            self._funcid = None
