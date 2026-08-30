"""Test F1 help resolution when mouse is over the status label.

Regression test: F1 should open the help for the row whose help is currently
shown in the status label, even after the sheet's _status_iid has been cleared
(on mouse leave).
"""

from __future__ import annotations

import sys
import tkinter as tk

import pytest

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


class _FakeConfigSheet:
    """Minimal ConfigSheet stand-in for F1 resolution tests."""

    def __init__(self, status_iid=None):
        self._status_iid = status_iid
        self._meta = {}
        self._f1_anchor_calls = []

    def _f1_anchor_for_iid(self, iid):
        self._f1_anchor_calls.append(iid)
        return f"anchor_for_{iid}" if iid else ""


class _FakeApp:
    """Minimal App stand-in for F1 resolution tests."""

    def __init__(self, root):
        self.root = root
        self._pointer_on_status_label = False  # simulates pointer over status label
        self._status_lbl_f1_anchor = None  # F1 anchor for what's shown in status label
        self._path_field = None
        self._pages = {}
        self._current = None

    def _within(self, widget, ancestor):
        if widget is None or ancestor is None:
            return False
        return widget is ancestor

    def _on_f1_help(self, _event=None):
        """Simplified F1 handler — returns the anchor that would be opened."""
        try:
            w = self.root.focus_get()
        except Exception:
            w = None

        if self._within(w, self._path_field):
            return "path_field_anchor"
        elif self._pointer_on_status_label and self._status_lbl_f1_anchor:
            # Mouse over the status label — use the stored anchor.
            return self._status_lbl_f1_anchor
        else:
            return "readme"


class TestF1StatusLabel:
    """F1 resolution when mouse is over the status label."""

    def test_f1_uses_stored_anchor_when_hovering_status(self):
        """F1 uses the stored anchor when mouse is over the status label.

        Simulates the real flow: _on_cell_status computes the anchor from the
        hovered row and stores it directly; the sheet's _status_iid is then
        cleared on mouse-leave, but the stored anchor persists for F1.
        """
        app = _FakeApp(_mod._root)
        cs = _FakeConfigSheet(status_iid="row_123")

        # Simulate _on_cell_status: compute and store the resolved anchor.
        app._status_lbl_f1_anchor = cs._f1_anchor_for_iid(cs._status_iid)

        # Simulate mouse entering the status label
        app._pointer_on_status_label = True

        # Simulate the sheet's _status_iid being cleared (mouse left the sheet)
        cs._status_iid = None

        # F1 should use the stored anchor, not the cleared _status_iid
        anchor = app._on_f1_help()
        assert anchor == "anchor_for_row_123"
        assert cs._f1_anchor_calls == ["row_123"]

    def test_f1_readme_when_not_hovering_status(self):
        """F1 opens readme when mouse is not over the status label."""
        app = _FakeApp(_mod._root)
        cs = _FakeConfigSheet(status_iid="row_123")

        # Store the anchor (as if status was published)
        app._status_lbl_f1_anchor = cs._f1_anchor_for_iid(cs._status_iid)

        # Mouse is NOT over the status label
        app._pointer_on_status_label = False

        anchor = app._on_f1_help()
        assert anchor == "readme"

    def test_f1_readme_when_no_stored_anchor(self):
        """F1 opens readme when no anchor is stored (status label empty)."""
        app = _FakeApp(_mod._root)

        # No anchor stored (status label is empty or showing something else)
        app._status_lbl_f1_anchor = None
        app._pointer_on_status_label = True

        anchor = app._on_f1_help()
        assert anchor == "readme"

    def test_stored_anchor_cleared_on_hide_tip(self):
        """_hide_tip clears the stored anchor."""
        from tcm_gui.app import App

        ns = type("NS", (), {})()
        ns.root = _mod._root
        ns._tip_active = True
        ns._dwell_active = True
        ns._dwell_widget = None
        ns._status_hovering = True
        ns._status_lbl_f1_anchor = "anchor_for_row_123"
        ns._dwell_job = None
        ns._dwell_hide_job = None
        ns._labels = []
        ns._status_lbl = type("L", (), {"set_text": lambda self, t, raw=False, base=None: ns._labels.append(t)})()
        ns._cancel_dwell_job = App._cancel_dwell_job.__get__(ns)
        ns._cancel_dwell_hide_job = App._cancel_dwell_hide_job.__get__(ns)
        ns._hide_tip = App._hide_tip.__get__(ns)

        ns._hide_tip()

        assert ns._status_lbl_f1_anchor is None

    def test_stored_anchor_cleared_on_clear_dwell_now(self):
        """_clear_dwell_now clears the stored anchor."""
        from tcm_gui.app import App

        ns = type("NS", (), {})()
        ns.root = _mod._root
        ns._dwell_active = True
        ns._status_hovering = True
        ns._status_lbl_f1_anchor = "anchor_for_row_123"
        ns._dwell_job = None
        ns._dwell_hide_job = None
        ns._labels = []
        ns._status_lbl = type("L", (), {"set_text": lambda self, t, raw=False, base=None: ns._labels.append(t)})()
        ns._cancel_dwell_job = App._cancel_dwell_job.__get__(ns)
        ns._cancel_dwell_hide_job = App._cancel_dwell_hide_job.__get__(ns)
        ns._clear_dwell_now = App._clear_dwell_now.__get__(ns)

        ns._clear_dwell_now()

        assert ns._status_lbl_f1_anchor is None
