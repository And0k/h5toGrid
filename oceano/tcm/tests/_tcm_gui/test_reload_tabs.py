"""Post-Run reload composition and tab guards without requiring Tk."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tcm_gui import _reload_tabs as reload_tabs

pytestmark = pytest.mark.gui


class _ReloadSheet:
    """Minimal ConfigSheet stand-in capturing reload arguments."""

    def __init__(self, cfg: dict, *, dirty: bool = False, metadata_dirty: bool = False):
        self._cfg = cfg
        self.is_dirty = dirty
        self._metadata_dirty = metadata_dirty
        self.loaded = None

    def is_metadata_dirty(self) -> bool:
        return self._metadata_dirty

    def load(self, cfg: dict, **kwargs) -> None:
        self.loaded = (cfg, kwargs)


def _reload_app(yaml_paths: dict, *, full_mode: bool = False) -> SimpleNamespace:
    """App stand-in with only the attributes used by reload helpers."""
    return SimpleNamespace(
        _yaml_paths=yaml_paths,
        _full_mode=full_mode,
        _pages={},
        _load_device_meta=lambda: (None, None, None),
    )


@pytest.mark.parametrize(
    ("prev_cfg", "disk", "expected_calib", "expected_ag", "test_description"),
    [
        pytest.param(
            {
                "input": {
                    "coefs": {"Ag": "old"},
                    "calib": {"g0xyz": [1.0, 2.0, 3.0]},
                },
                "out": {"dt_bins": [0]},
                "_page_stem": "@i_01",
            },
            {"input": {"coefs": {"Ag": "new"}}},
            False,
            "new",
            "Consumed calib disappears while disk coefs and scan-time sections win",
            id="consumed-calib",
        ),
        pytest.param(
            {
                "input": {
                    "coefs": {"Ag": "old"},
                    "calib": {"azimuth_add": 2.5},
                },
                "filter": {"max": {"M": 5.0}},
                "_page_stem": "@i_01",
            },
            {"input": {"coefs": {"Ag": "new"}, "calib": {"azimuth_add": 2.5}}},
            True,
            "new",
            "Calib present on disk is retained alongside unrelated scan-time sections",
            id="retained-calib",
        ),
    ],
)
def test_compose_reload_cfg(prev_cfg, disk, expected_calib, expected_ag, test_description):
    """Reload merge keeps scan context, applies disk values, and handles calib."""
    merged = reload_tabs.compose_reload_cfg(prev_cfg, disk)

    assert ("calib" in merged["input"]) is expected_calib, (
        f"{test_description}: calib presence mismatch, got {merged['input'].get('calib')!r}"
    )
    assert merged["input"]["coefs"]["Ag"] == expected_ag, (
        f"{test_description}: disk coefs lost, got {merged['input']['coefs']!r}"
    )
    assert merged["_page_stem"] == "@i_01", (
        f"{test_description}: scan-time page stem lost, got {merged.get('_page_stem')!r}"
    )


def test_reload_processed_clean_tab(tmp_path):
    """A clean processed tab rebuilds from disk with consumed calib removed."""
    stem = "@i_01"
    yaml_path = tmp_path / f"{stem}.yaml"
    yaml_path.write_text(
        "# @package _global_\n"
        "input:\n"
        "  path: 'data/@i_01.txt'\n"
        "  coefs:\n"
        "    Rz: [[1,0,0],[0,1,0],[0,0,1]]\n"
        "    date: '2024-06-13T12:00:05'\n",
        encoding="utf-8",
    )
    prev_cfg = {
        "input": {
            "path": "data/@i_01.txt",
            "coefs": {"Rz": "old"},
            "calib": {"g0xyz": [1.0, 2.0, 3.0]},
        },
        "_page_stem": stem,
    }
    sheet = _ReloadSheet(prev_cfg)
    app = _reload_app({stem: yaml_path})
    test_description = "Post-Run reload applies stamped YAML and clears consumed triggers"

    assert reload_tabs.reload_tab_after_run(app, stem, sheet, {"i01"}) is True
    assert sheet.loaded is not None, f"{test_description}: clean processed tab did not reload"
    merged, kwargs = sheet.loaded
    assert "calib" not in merged["input"], (
        f"{test_description}: consumed calib survived reload, got {merged['input'].get('calib')!r}"
    )
    assert merged["input"]["coefs"]["date"] == "2024-06-13T12:00:05", (
        f"{test_description}: stamped date lost, got {merged['input']['coefs']!r}"
    )
    assert kwargs["metadata"] is None and kwargs["metadata_path"] is None, (
        f"{test_description}: reload must omit unavailable metadata, got {kwargs!r}"
    )
    assert "sync_status" not in kwargs, (
        f"{test_description}: post-run sync status must not be reused, got {kwargs!r}"
    )


@pytest.mark.parametrize(
    ("dirty", "metadata_dirty", "processed", "test_description"),
    [
        pytest.param(
            True,
            False,
            {"i01"},
            "In-flight config edits are preserved instead of being overwritten",
            id="dirty-config",
        ),
        pytest.param(
            False,
            True,
            {"i01"},
            "In-flight metadata edits are preserved instead of being overwritten",
            id="dirty-metadata",
        ),
        pytest.param(
            False,
            False,
            {"i02"},
            "Failed or unprocessed probes retain their pre-Run sheet",
            id="unprocessed-probe",
        ),
    ],
)
def test_reload_skips_guarded_tabs(tmp_path, dirty, metadata_dirty, processed, test_description):
    """Dirty and unprocessed tabs are left untouched after Run."""
    stem = "@i_01"
    yaml_path = tmp_path / f"{stem}.yaml"
    yaml_path.write_text("# @package _global_\ninput:\n  path: 'data/@i_01.txt'\n", encoding="utf-8")
    sheet = _ReloadSheet({"input": {}}, dirty=dirty, metadata_dirty=metadata_dirty)
    app = _reload_app({stem: yaml_path})

    assert reload_tabs.reload_tab_after_run(app, stem, sheet, processed) is False
    assert sheet.loaded is None, f"{test_description}: guarded tab was reloaded"
