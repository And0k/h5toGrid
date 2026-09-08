"""Split/nesting model of the ``metadata`` node — ``MetadataNodeMixin``.

Style: test_metadata_sheet.py (``__new__`` + mocked sheet harness).
Reference: docs/reference/io_formats.md (*Multiple intervals*) +
docs/project_developer_guide/GUI/architecture.md.

Unit-level (no Tk) coverage, against the real model:
* single → multi: ``split_setup`` moves flat paired rows into a ``0`` node
  and numbers the copy ``1``;
* boundary pinning: ``above`` ⇒ ``time_range[1] := time_range[0]``,
  ``below`` ⇒ ``time_range[0] := time_range[1]``;
* nested load: dict station keys → ``setup`` sublevels (numeric preserved,
  non-numeric renumbered);
* multi-setup split keeps adjacency and bumps the autonumber;
* write-back returns the full ``{station_key: 11-array}`` map.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import tcm_gui.coef_sheet as coef_sheet
from tcm.schema import Config, Return

_PAIRS_REF = [
    "point, symbol",
    "sea depth, h_above",
    "lat, lon",
    "time_range",
    "burst_dt/t",
    "comment",
]


def _single_interval_arr():
    return ["P1", 15.0, 0.5, "↟", 54.71, 19.84, "2026-07-11T12:00:00", "2026-07-20T08:30:00", 60, 600, ""]


def _harness(cfg=None):
    """Minimal ``__new__``-constructed ``ConfigSheet`` (real mixin model)."""
    mock_sh = MagicMock()
    mock_sh.total_columns.return_value = 6
    mock_sh.total_rows.return_value = 0
    mock_sh.get_children.return_value = []
    mock_sh.get_cell_data.return_value = ""
    mock_sh.tag_names.return_value = []
    mock_sh.winfo_rgb.return_value = (0, 0, 0)
    items: dict = {}
    kids: dict = {}

    def _insert(**kw):
        iid = f"iid_{kw.get('text', 'x')}_{len(items)}"
        parent = kw.get("parent") or ""
        kids.setdefault(parent, []).append(iid)
        items[iid] = {"values": kw.get("values", ()), "text": kw.get("text", "")}
        return iid

    def _item(iid=None, **kw):
        if iid is None:
            return {}
        if kw:
            if iid in items and "text" in kw:
                items[iid]["text"] = kw["text"]
            return {}
        return items.get(iid, {})

    mock_sh.insert.side_effect = _insert
    mock_sh.item.side_effect = _item
    mock_sh.get_children.side_effect = lambda parent="": list(kids.get(parent or "", ()))

    cfg = cfg or {"input": {"path": "dummy.txt", "coefs": {}}}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(coef_sheet, "Sheet", lambda *a, **k: mock_sh)
        cs = coef_sheet.ConfigSheet.__new__(coef_sheet.ConfigSheet)
        cs.sh = mock_sh
        cs._meta = {}
        cs._nv = 6
        cs._full = False
        cs._cfg = cfg
        cs._config_root = Config
        cs._return_enum = Return
        cs._snap = ()
        cs._snap_meta = ()
        cs._fg_default = "#000000"
        cs._int_row_of = {}
        cs._vis = ()
        cs._col_resize = MagicMock()
        cs.on_edit_begin = None
        cs._readonly = False
        cs._page_stem = ""
        cs._metadata_path = "device_dir/info_devices.yaml"
        cs._ph = MagicMock()
        cs._ph.has.return_value = False
        cs._rebuild_row_caches = MagicMock()
        cs._apply_open = MagicMock()
        cs._apply_styles = MagicMock()
        cs._apply_default_fg = MagicMock()
        cs._apply_placeholders = MagicMock()
        cs._apply_validations = MagicMock()
        cs._apply_metadata_dirty_label = MagicMock()
        # wire mixin model state (cleared before a split-specific seed)
        cs._setups = None
        cs._next_setup_num = 0
        cs._cur_setup = 0
        cs._metadata_unsaved = False
        cs._rc_sel_iid = None
    return cs, mock_sh


def test_build_single_interval_flat():
    """One interval ⇒ paired rows directly under ``metadata`` (no setup level)."""
    cs, _ = _harness()
    cs._metadata = _single_interval_arr()
    cs._build_metadata()
    root = next(i for i, m in cs._meta.items() if m.get("is_metadata_root"))
    kids = [m.get("label") for iid, m in cs._meta.items() if m.get("parent") == root]
    assert kids == _PAIRS_REF
    assert not any(m.get("is_setup") for m in cs._meta.values())


def test_nested_load_creates_setup_sublevels():
    """Several intervals (nested device entry) ⇒ ``setup`` sublevels shown."""
    cs, _ = _harness()
    cs._metadata = {
        "0": _single_interval_arr(),
        "1": ["P2", 20.0, 1.0, "↟", 54.9, 20.1, "2026-06-01T10:00:00", "2026-06-10T14:00:00", 120, 600, ""],
    }
    cs._build_metadata()
    root = next(i for i, m in cs._meta.items() if m.get("is_metadata_root"))
    setup_iids = [iid for iid, m in cs._meta.items() if m.get("parent") == root and m.get("is_setup")]
    assert len(setup_iids) == 2
    assert [m["setup_idx"] for iid, m in cs._meta.items() if m.get("is_setup")] == [0, 1]
    for sid in setup_iids:
        row_labels = [
            m.get("label") for iid, m in cs._meta.items() if m.get("parent") == sid and m.get("is_metadata")
        ]
        assert row_labels == _PAIRS_REF
    assert cs._metadata == _single_interval_arr(), "_metadata alias is the first interval"


def test_split_fresh_single_interval():
    """Root split: flat rows move into ``0``; copy lands above in ``1``.

    "Insert above" ≈ tksheet's insert-row-above: the new setup (the copy) is
    the sibling above the current one — physical order ``[1(copy), 0(orig)]``.
    """
    cs, _ = _harness()
    cs._metadata = _single_interval_arr()
    cs._build_metadata()
    cs._rebuild_metadata_rows = lambda: None
    first_iid = next(iter(cs._meta))
    assert cs.split_setup(above=True, ref=cs._meta[first_iid])
    assert [n for n, _ in cs._setups] == [1, 0]
    _1, copy = cs._setups[0]
    _0, orig = cs._setups[1]
    # above ⇒ copy shares origin start as both ends
    assert copy[6] == orig[6] and copy[7] == orig[6]
    assert cs.is_metadata_dirty()


def test_split_below_pins_start_boundary():
    cs, _ = _harness()
    cs._metadata = _single_interval_arr()
    cs._build_metadata()
    cs._rebuild_metadata_rows = lambda: None
    cs.split_setup(above=False)
    _0, orig = cs._setups[0]
    _1, copy = cs._setups[1]
    assert [n for n, _ in cs._setups] == [0, 1]
    assert copy[7] == orig[7] and copy[6] == orig[7]  # below ⇒ copy origin[0] := origin[1]


def test_split_multi_setup_adjacent_autonumber():
    """Existing setup selected: copy is the next sibling, autonumber bumped."""
    cs, _ = _harness()
    cs._metadata = {"0": _single_interval_arr(), "1": _single_interval_arr()}
    cs._build_metadata()
    cs._rebuild_metadata_rows = lambda: None
    sel_ref = next(m for m in cs._meta.values() if m.get("is_setup") and m.get("setup_idx") == 1)
    cs.split_setup(above=False, ref=sel_ref)
    assert [n for n, _ in cs._setups] == [0, 1, 2]
    assert cs._setups[1][1][7] == cs._setups[2][1][6]  # below-copy origin[0] := origin[1]


def test_non_numeric_keys_renumber_sequentially():
    """Non-numeric station keys → autonumbered 0..n-1 in file order."""
    cs, _ = _harness()
    cs._metadata = {"a": _single_interval_arr(), "b": _single_interval_arr()}
    cs._build_metadata()
    assert [n for n, _ in cs._setups] == [0, 1]


def test_get_edited_metadata_map_full():
    """Write-back map preserves every interval's station key."""
    cs, _ = _harness()
    cs._metadata = {"0": _single_interval_arr(), "3": _single_interval_arr()}
    cs._build_metadata()
    cs._metadata_unsaved = True
    result = cs.get_edited_metadata_map()
    assert set(result) == {"0", "3"}
    assert result["0"][6] == _single_interval_arr()[6]


def test_autofill_burst_seeds_stub():
    """autofill_burst creates a stub and fills burst fields (indices 8, 9)."""
    cs, _ = _harness()
    tr = ["2026-07-11T12:00:00", "2026-07-20T08:30:00"]
    assert cs.autofill_burst(60, 600, time_ranges=tr)
    assert cs._setups[0][1][8] == 60 and cs._setups[0][1][9] == 600
    assert cs._setups[0][1][6] == tr[0] and cs._setups[0][1][7] == tr[1]
    assert cs._metadata_unsaved


def test_autofill_burst_noop_when_filled():
    cs, _ = _harness()
    cs._metadata = _single_interval_arr()
    cs._build_metadata()
    assert not cs.autofill_burst(99, 999), "filled burst pair must not rewrite"


@pytest.mark.parametrize(
    ("anchor", "calls_insert"),
    [
        ("hydra", True),
        ("leaf", False),
        ("setup", False),
        ("root", False),
        (None, False),
    ],
    ids=["hydra-child", "paired-leaf", "setup-node", "meta-root", "no-selection"],
)
def test_add_row_child_only_parents_hydra(anchor, calls_insert, test_description="child-add routing"):
    """Extras ``Add row`` parents a tracked child under hydra nodes — never metadata, never top-level."""
    cs, mock_sh = _harness()
    cs._metadata = _single_interval_arr()
    cs._build_metadata()
    anchors = {
        "hydra": {"parent": "input", "key": "path"},
        "leaf": {"is_metadata": True, "setup_idx": 0, "parent": "root", "label": "time_range"},
        "setup": {"is_setup": True, "setup_idx": 0, "parent": "root"},
        "root": {"is_metadata_root": True, "parent": ""},
    }
    if anchor is None:
        cs._rc_sel_iid = None
    else:
        cs._meta["anchor"] = anchors[anchor]
        cs._rc_sel_iid = "anchor"
    mock_sh.insert.reset_mock()
    new_iid = cs._add_row_child()
    if not calls_insert:
        assert new_iid is None, f"{test_description}: {anchor} must deny"
        mock_sh.insert.assert_not_called()
        return
    assert new_iid is not None, f"{test_description}: hydra parent must insert"
    assert mock_sh.insert.call_args.kwargs.get("parent") == "anchor", (
        f"{test_description}: child must parent under selection — {mock_sh.insert.call_args}"
    )
    assert new_iid in cs._user_row_set(), f"{test_description}: child must be tracked deletable"
