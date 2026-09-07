"""Existing-menu patch + group undo — labels, sort removal, native chronology.

Style: no-Tk fakes (``FakeMenu``/``FakeMT`` duck-type the tksheet surface the
patch touches); sketch API ``MetadataTree → GroupUndoBridge → install_menu_patch``.
Reference: ``tcm_gui/_sheet_popup.py`` + ``tcm_gui/_sheet_undo.py`` docstrings.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import tcm_gui._sheet_popup as popup
from tcm_gui._sheet_popup import SORT_BINDINGS, install_menu_patch, refresh_insert_labels, target_label
from tcm_gui._sheet_undo import (
    GROUP_EVENT,
    GroupUndoBridge,
    MetadataTree,
    _VetoNativeUndo,
    suspend_native_pushes,
)

_STR_DIR = Path(__file__).resolve().parents[2] / "src" / "tcm_gui"


class FakeMenu:
    """Minimal ``tk.Menu`` surface: command entries with get/set label."""

    def __init__(self, labels):
        self._labels = list(labels)

    def index(self, arg):
        return len(self._labels) - 1 if arg == "end" and self._labels else None

    def type(self, i):
        return "command"

    def entrycget(self, i, opt):
        assert opt == "label", f"unexpected option {opt}"
        return self._labels[i]

    def entryconfig(self, i, **kw):
        self._labels[i] = kw["label"]

    @property
    def labels(self):
        return list(self._labels)


class FakeMT:
    """Native undo-stack surface (lists + hook slot, no Tk)."""

    def __init__(self):
        self.undo_stack: list = []
        self.redo_stack: list = []
        self.extra_begin_ctrl_z_func = None
        self.undo_enabled = True

    def purge_redo_stack(self):
        self.redo_stack = []


class FakeSheet:
    """Sheet surface for ``install_menu_patch`` (sort switch + extras + menus)."""

    def __init__(self):
        self.MT = FakeMT()
        self.RI = FakeMT()  # only needs the extra_rc_func slot
        self.RI.extra_rc_func = None
        self.MT.extra_rc_func = None
        self.disabled: list | None = None
        self.extras: dict = {}
        self.inserted_rows = 0

    def disable_bindings(self, bindings):
        self.disabled = list(bindings)

    def popup_menu_add_command(self, label, func):
        self.extras[label] = func

    def insert_row(self):
        self.inserted_rows += 1


class FakeHost:
    """ConfigSheet-shaped metadata owner (model + snapshot protocol, no Tk)."""

    def __init__(self, ref=None):
        self.sh = FakeSheet()
        self._setups = [[0, [None] * 11]]
        self._next_setup_num = 1
        self._cur_setup = 0
        self._metadata_unsaved = False
        self._ref = ref or {"is_metadata_root": True}
        self._meta = {"iid": self._ref}
        self._rc_sel_iid = "iid"
        self._undo_bridge = None
        self.split_calls: list = []

    def _rc_meta_ref(self):
        return self._ref

    def split_setup(self, *, above):
        self.split_calls.append(above)
        return True

    def _insert_col_at_end(self, _event=None):
        pass

    def _snapshot_group(self):
        import copy

        return (copy.deepcopy(self._setups), self._next_setup_num, self._cur_setup, self._metadata_unsaved)

    def _restore_group(self, snap):
        import copy

        setups, nxt, cur, unsaved = snap
        self._setups = copy.deepcopy(setups)
        self._next_setup_num, self._cur_setup, self._metadata_unsaved = nxt, cur, unsaved


@pytest.mark.parametrize(
    "fname",
    ["str.yaml", "str_ru.yaml"],
    ids=["en", "ru"],
)
def test_str_keys_carry_target_placeholder(fname, test_description="i18n split keys"):
    """Both chrome string tables define the split labels with a ``{target}`` slot."""
    data = yaml.safe_load((_STR_DIR / fname).read_text(encoding="utf-8"))
    for key in ("sheet.insert_setup_above", "sheet.insert_setup_below"):
        assert key in data, f"{test_description}: {fname} misses {key}"
        assert "{target}" in data[key], f"{test_description}: {fname}:{key} has no " + "{target} slot"


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        ({"is_metadata_root": True}, "1"),
        ({"is_setup": True, "setup_idx": 0}, "1"),
        ({"is_metadata": True, "setup_idx": 0}, ""),
        ({"key": "input"}, ""),
        (None, ""),
    ],
    ids=["meta-root", "setup-node", "paired-row", "plain-node", "no-selection"],
)
def test_target_label_only_on_meta_or_setup(ref, expected, test_description="split-label anchor"):
    """The relabel anchor is the ``metadata`` root / ``setup`` node — its next autonumber."""
    tree = MetadataTree(FakeHost(ref if ref is not None else {"key": "input"}))
    if ref is None:
        tree.host._ref = None
        tree.host._meta = {}
        tree.host._rc_sel_iid = "missing"
    assert target_label(tree) == expected, (
        f"{test_description}: ref={ref!r} — expected {expected!r}, got {target_label(tree)!r}"
    )


def test_render_insert_labels_interpolates_target(test_description="label rendering"):
    """Rendered labels name the number the split copy will take."""
    monkey_S = {"sheet.insert_setup_above": "ABOVE {target}", "sheet.insert_setup_below": "BELOW {target}"}
    orig = popup._S
    popup._S = monkey_S
    try:
        above, below = popup.render_insert_labels(MetadataTree(FakeHost()))
    finally:
        popup._S = orig
    assert (above, below) == ("ABOVE 1", "BELOW 1"), f"{test_description}: got {(above, below)!r}"


def test_render_insert_labels_native_when_no_anchor(test_description="native fallback"):
    """Paired rows and plain nodes keep tksheet's native labels (``(None, None)``)."""
    tree = MetadataTree(FakeHost({"is_metadata": True, "setup_idx": 0}))
    assert popup.render_insert_labels(tree) == (None, None), f"{test_description}: expected native labels"


def test_refresh_relabels_and_restores(test_description="per-popup refresh"):
    """Above/below entries follow the selection; other entries are never touched."""
    menu = FakeMenu(["Cut", "Insert rows above", "Insert rows below", "Delete rows"])
    tree = MetadataTree(FakeHost())
    monkey_S = {"sheet.insert_setup_above": "ABOVE {target}", "sheet.insert_setup_below": "BELOW {target}"}
    orig = popup._S
    popup._S = monkey_S
    try:
        assert refresh_insert_labels(menu, tree) is True, (
            f"{test_description}: first refresh must change labels"
        )
        assert menu.labels[1:3] == ["ABOVE 1", "BELOW 1"], f"{test_description}: got {menu.labels!r}"
        assert refresh_insert_labels(menu, tree) is False, (
            f"{test_description}: repeat refresh must be a no-op"
        )
        plain = MetadataTree(FakeHost({"key": "input"}))
        assert refresh_insert_labels(menu, plain) is True, f"{test_description}: leaving must restore natives"
        assert menu.labels == ["Cut", "Insert rows above", "Insert rows below", "Delete rows"], (
            f"{test_description}: got {menu.labels!r}"
        )
    finally:
        popup._S = orig


def test_disable_sort_menus_covers_all_sort_bindings(test_description="sort removal"):
    """Sort entries go off through binding flags (menus are rebuilt per popup)."""
    sheet = FakeSheet()
    assert popup.disable_sort_menus(sheet) is True, f"{test_description}: expected True"
    assert sheet.disabled is not None and set(sheet.disabled) == set(SORT_BINDINGS), (
        f"{test_description}: got {sheet.disabled!r}"
    )


def test_install_registers_extras_and_hooks_once(test_description="one-time install"):
    """Extras land in the existing menu dicts; hooks are set once and survive re-install."""
    host = FakeHost()
    assert install_menu_patch(host) is True, f"{test_description}: install must succeed"
    assert host.sh.disabled is not None and set(host.sh.disabled) == set(SORT_BINDINGS), (
        f"{test_description}: sort not disabled — {host.sh.disabled!r}"
    )
    assert len(host.sh.extras) == 2, f"{test_description}: extras missing — {sorted(host.sh.extras)!r}"
    assert host._undo_bridge is not None, f"{test_description}: bridge not linked"
    mt_hook, ri_hook = host.sh.MT.extra_rc_func, host.sh.RI.extra_rc_func
    assert mt_hook is not None and ri_hook is not None, f"{test_description}: pre-popup hooks missing"
    assert install_menu_patch(host) is True, f"{test_description}: re-install must succeed"
    assert host.sh.MT.extra_rc_func is mt_hook and host.sh.RI.extra_rc_func is ri_hook, (
        f"{test_description}: hooks must not double-wrap"
    )


def test_install_accepts_explicit_bridge(test_description="sketch API"):
    """``MetadataTree(sheet, rows) → GroupUndoBridge → install_menu_patch`` wiring."""
    host = FakeHost()
    meta = MetadataTree(host, rows=[[0, [None] * 11]])
    bridge = GroupUndoBridge(meta)
    assert install_menu_patch(meta, bridge) is True, f"{test_description}: install must succeed"
    assert host._undo_bridge is bridge, f"{test_description}: explicit bridge must be linked"


def test_group_coalesces_fragments_into_one_marker(test_description="group push"):
    """Rebuild fragments pushed inside ``group()`` are discarded; one marker remains."""
    host = FakeHost()
    bridge = GroupUndoBridge(MetadataTree(host))
    host._undo_bridge = bridge
    with bridge.group():
        host._setups.append([1, [None] * 11])
        host.sh.MT.undo_stack.append({"name": "add_rows", "data": {}})  # rebuild fragment
    (marker,) = host.sh.MT.undo_stack
    assert marker["name"] == GROUP_EVENT, f"{test_description}: got {marker!r}"
    assert marker["data"]["before"][0] == [[0, [None] * 11]], f"{test_description}: bad before snapshot"
    assert [n for n, _ in marker["data"]["after"][0]] == [0, 1], f"{test_description}: bad after snapshot"


def test_undo_redo_roundtrip_through_native_hook(test_description="native chronology"):
    """A top marker vetoes native inversion and restores snapshots; redo mirrors it."""
    host = FakeHost()
    bridge = GroupUndoBridge(MetadataTree(host))
    with bridge.group():
        host._setups.append([1, [None] * 11])
        host._next_setup_num = 2
    hook = host.sh.MT.extra_begin_ctrl_z_func
    assert bridge.can_undo() and not bridge.can_redo(), f"{test_description}: bad initial state"
    with pytest.raises(_VetoNativeUndo):
        hook({"eventname": "begin_undo"})
    assert [n for n, _ in host._setups] == [0], f"{test_description}: undo must restore before"
    assert bridge.can_redo(), f"{test_description}: marker must reach the redo stack"
    with pytest.raises(_VetoNativeUndo):
        hook({"eventname": "begin_redo"})
    assert [n for n, _ in host._setups] == [0, 1], f"{test_description}: redo must restore after"
    assert bridge.can_undo() and not bridge.can_redo(), f"{test_description}: marker must return"


def test_hook_delegates_native_entries(test_description="delegate otherwise"):
    """Ordinary stack tops pass through to the previous hook / native path (no veto)."""
    host = FakeHost()
    seen: list = []
    host.sh.MT.extra_begin_ctrl_z_func = seen.append  # pre-existing hook
    GroupUndoBridge(MetadataTree(host))  # chains, not replaces
    host.sh.MT.undo_stack.append({"name": "end_edit_table", "data": {"eventname": "end_edit_table"}})
    assert host.sh.MT.extra_begin_ctrl_z_func({"eventname": "begin_undo"}) is None, (
        f"{test_description}: native entries must not veto"
    )
    assert seen and seen[0]["eventname"] == "begin_undo", f"{test_description}: prev hook must run"
    assert len(host.sh.MT.undo_stack) == 1, f"{test_description}: native entry must stay put"


def test_failed_restore_keeps_marker(test_description="failing restore"):
    """A restore error never loses the chronology entry (and still vetoes native)."""
    host = FakeHost()
    bridge = GroupUndoBridge(MetadataTree(host))
    with bridge.group():
        host._setups.append([1, [None] * 11])

    def _boom(_snap):
        raise RuntimeError("boom")

    host._restore_group = _boom
    with pytest.raises(_VetoNativeUndo):
        host.sh.MT.extra_begin_ctrl_z_func({"eventname": "begin_undo"})
    assert len(host.sh.MT.undo_stack) == 1, f"{test_description}: marker must be kept"
    assert not host.sh.MT.redo_stack, f"{test_description}: nothing may reach redo"


def test_suspend_discards_fragments_keeps_redo(test_description="suspension"):
    """Scratch-stack swap drops inner pushes and pins the redo deque."""
    mt = FakeMT()
    mt.undo_stack.append({"name": "old", "data": {}})
    mt.redo_stack.append({"name": "r", "data": {}})
    with suspend_native_pushes(mt):
        mt.undo_stack.append({"name": "frag", "data": {}})
        mt.redo_stack = []
    assert [e["name"] for e in mt.undo_stack] == ["old"], f"{test_description}: fragments must go"
    assert [e["name"] for e in mt.redo_stack] == ["r"], f"{test_description}: redo must survive"


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        ({"is_metadata_root": True}, ("meta", None)),
        ({"is_setup": True, "setup_idx": 2}, ("setup", 2)),
        ({"is_metadata": True, "setup_idx": 1}, ("row", 1)),
        ({"key": "input"}, None),
    ],
    ids=["root", "setup", "row", "other"],
)
def test_facade_selected_ref(ref, expected, test_description="facade protocol"):
    """``MetadataTree.selected_ref`` normalizes host dicts to the sketch protocol."""
    assert MetadataTree(FakeHost(ref)).selected_ref() == expected, (
        f"{test_description}: ref={ref!r} — got {MetadataTree(FakeHost(ref)).selected_ref()!r}"
    )


def test_facade_insert_delegates_split(test_description="facade insert"):
    """``insert(above)`` routes to the host split; delete stays a native no-op."""
    tree = MetadataTree(FakeHost())
    assert tree.insert(above=False) is True, f"{test_description}: insert must delegate"
    assert tree.host.split_calls == [False], f"{test_description}: got {tree.host.split_calls!r}"
    assert tree.delete_selected_setup() is False, f"{test_description}: no setup deletion exists"
