"""Windowed-entry stream sinks — `app.main` never runs with ``None`` console streams.

Regression: PyInstaller ``--noconsole`` dists expose no console —
``sys.stdout``/``sys.stderr`` are ``None``, and hydra's ``JobReturn.return_value``
writes to ``sys.stderr`` before re-raising a *failed* job's exception, so the
resulting ``AttributeError: 'NoneType' object has no attribute 'write'`` replaced
the real scan error (see `app.main`).
"""

import os
import sys


class _FakeApp:
    def __init__(self, argv):
        self.argv = argv

    def run(self):
        pass


def test_none_streams_replaced_by_devnull(monkeypatch):
    from tcm_gui import app as app_mod

    monkeypatch.setattr(app_mod, "App", _FakeApp)
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)

    app_mod.main(["tcm_gui", "some/path"])

    assert getattr(sys.stdout, "name", "") == os.path.basename(os.devnull)
    assert getattr(sys.stderr, "name", "") == os.path.basename(os.devnull)
    print("probe")  # EAFP — raises AttributeError if stdout were still None


def test_existing_streams_kept(monkeypatch):
    from tcm_gui import app as app_mod

    monkeypatch.setattr(app_mod, "App", _FakeApp)
    out, err = sys.stdout, sys.stderr

    app_mod.main(["tcm_gui"])

    assert sys.stdout is out  # console attached → untouched
    assert sys.stderr is err
