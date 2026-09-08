"""Existing-menu patch + group undo + row-mutation authorization.

Style: no-Tk fakes (``FakeMenu``/``FakeMT`` duck-type the tksheet surface the
patch touches); sketch API ``MetadataTree → GroupUndoBridge → install_menu_patch``.
Reference: ``tcm_gui/_sheet_popup.py`` + ``tcm_gui/_sheet_undo.py`` docstrings.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import tcm_gui._sheet_popup as popup
from tcm_gui._sheet_popup import (
    OFF_BINDINGS,
    SORT_BINDINGS,
    RowPolicy,
    SheetGuard,
    install_menu_patch,
    refresh_insert_labels,
    refresh_menu_state,
    target_label,
)
from tcm_gui._sheet_undo import (
    GROUP_EVENT,
    GroupUndoBridge,
    MetadataTree,
    _VetoNativeUndo,
    suspend_native_pushes,
)

_STR_DIR = Path(__file__).resolve().parents[2] / "src" / "tcm_gui"


class FakeMenu:
    """Minimal ``tk.Menu`` surface: command entries with get/set label + state."""

    def __init__(self, labels):
        self._labels = list(labels)
        self._states = ["normal"] * len(self._labels)

    def index(self, arg):
        return len(self._labels) - 1 if arg == "end" and self._labels else None

    def type(self, i):
        return "command"

    def entrycget(self, i, opt):
        assert opt in ("label", "state"), f"unexpected option {opt}"
        return self._labels[i] if opt == "label" else self._states[i]

    def entryconfig(self, i, **kw):
        if "label" in kw:
            self._labels[i] = kw["label"]
        if "state" in kw:
            self._states[i] = kw["state"]

    @property
    def labels(self):
        return list(self._labels)

    def state_of(self, label):
        return self._states[self._labels.index(label)]


class FakeMT:
    """Native undo-stack surface (lists + hook slot, no Tk)."""

    def __init__(self):
        self.undo_stack: list = []
        self.redo_stack: list = []
        self.extra_begin_ctrl_z_func = None
        self.extra_end_insert_rows_rc_func = None
        self.extra_end_insert_cols_rc_func = None
        self.undo_enabled = True
        self.popup_menu_loc = None
        self.selected_cols: list = []

    def purge_redo_stack(self):
        self.redo_stack = []

    def get_selected_cols(self):
        return list(self.selected_cols)

    def datacn(self, c):
        return int(c)


class FakeCH:
    def __init__(self):
        self.popup_menu_loc = None


class FakeSheet:
    """Sheet surface for ``install_menu_patch`` (sort switch + extras + menus)."""

    def __init__(self):
        self.MT = FakeMT()
        self.MT.CH = FakeCH()
        self.RI = FakeMT()  # only needs the extra_rc_func slot
        self.RI.extra_rc_func = None
        self.MT.extra_rc_func = None
        self.disabled: list | None = None
        self.extras: dict = {}
        self.inserted_rows = 0
        self.selected_rows: list = []
        self.existing: set = set()
        self.bells = 0
        self.n_cols = 6

    def disable_bindings(self, bindings):
        self.disabled = list(bindings)

    def popup_menu_add_command(self, label, func):
        self.extras[label] = func

    def insert_row(self):
        self.inserted_rows += 1

    def get_selected_rows(self):
        return list(self.selected_rows)

    def exists(self, iid):
        return iid in self.existing

    def bell(self):
        self.bells += 1

    def total_columns(self):
        return self.n_cols


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
        self._row_policy = None
        self._row_guard = None
        self._guard_suspended = 0
        self._user_rows: set = set()
        self._user_col_base = None
        self._readonly = False
        self._loading = False
        self.vis = ["iid"]
        self.rebuilt = 0
        self.cleared = 0
        self.split_calls: list = []

    def _walk(self, parent="", *, visible=False):
        return iter(list(self.vis)) if visible else iter(())

    def _rc_meta_ref(self):
        return self._ref

    def split_setup(self, *, above):
        self.split_calls.append(above)
        return True

    def _insert_col_at_end(self, _event=None):
        pass

    def _add_row_child(self, _event=None):
        pass

    def _rebuild_row_caches(self):
        self.rebuilt += 1

    def _clear_status(self):
        self.cleared += 1

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
    """Unsafe entries go off through binding flags (menus are rebuilt per popup)."""
    sheet = FakeSheet()
    assert popup.disable_unsafe_menus(sheet) is True, f"{test_description}: expected True"
    assert sheet.disabled is not None and set(sheet.disabled) == set(OFF_BINDINGS), (
        f"{test_description}: got {sheet.disabled!r}"
    )
    assert set(SORT_BINDINGS) <= set(sheet.disabled), f"{test_description}: sort must stay off"


def test_install_registers_extras_and_hooks_once(test_description="one-time install"):
    """Extras land in the existing menu dicts; hooks are set once and survive re-install."""
    host = FakeHost()
    assert install_menu_patch(host) is True, f"{test_description}: install must succeed"
    assert host.sh.disabled is not None and set(host.sh.disabled) == set(OFF_BINDINGS), (
        f"{test_description}: unsafe bindings not disabled — {host.sh.disabled!r}"
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


# ------------------------------------------------------------- RowPolicy
def _policy_host(ref=None, **kw):
    """FakeHost with a ``RowPolicy`` bound (mirrors ``install_menu_patch`` linking)."""
    host = FakeHost(ref)
    for k, v in kw.items():
        setattr(host, k, v)
    tree = MetadataTree(host)
    return host, RowPolicy(tree)


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        ({"is_metadata_root": True}, "split"),
        ({"is_setup": True, "setup_idx": 1}, "split"),
        ({"is_metadata": True, "setup_idx": 0}, "deny"),
        ({"parent": "", "key": "input"}, "deny"),
        ({"parent": "input", "key": "path"}, "native"),
        ({"parent": "coefs", "key": "Ag"}, "native"),
        (None, "deny"),
    ],
    ids=["meta", "setup", "leaf", "top", "hydra-child", "hydra-leaf", "no-target"],
)
def test_insert_permission_matrix(ref, expected, test_description="insert verdicts"):
    """Split only on meta/setup; leaves + top nodes deny; hydra children go native."""
    host, pol = _policy_host(ref if ref is not None else {"key": "input"})
    if ref is None:
        host._ref, host._meta, host._rc_sel_iid = None, {}, "missing"
    assert pol.insert_permission(pol.oracle_ref()) == expected, (
        f"{test_description}: ref={ref!r} — got {pol.insert_permission(pol.oracle_ref())!r}"
    )


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        ({"is_metadata_root": True}, False),
        ({"is_setup": True, "setup_idx": 0}, False),
        ({"is_metadata": True, "setup_idx": 0}, False),
        ({"parent": "", "key": "input"}, True),
        ({"parent": "input", "key": "path"}, True),
        ({"parent": "coefs", "key": "Ag"}, True),
    ],
    ids=["meta", "setup", "leaf", "top", "hydra-child", "hydra-leaf"],
)
def test_child_add_only_outside_metadata(ref, expected, test_description="child-add gate"):
    """Extras ``Add row`` parents under hydra/top/user nodes — never the metadata subtree."""
    host, pol = _policy_host(ref)
    assert pol.can_child_add() is expected, f"{test_description}: ref={ref!r}"
    host._readonly = True
    assert pol.can_child_add() is False, f"{test_description}: readonly must deny"


def test_user_rows_insert_deny_but_deletable(test_description="delete-only-yours"):
    """Extras-added rows: insert denied, delete allowed iff the selection is all-user."""
    host, pol = _policy_host({"parent": "input", "key": "path"})
    host._user_rows = {"u1", "u2"}
    host.sh.existing = {"u1", "u2", "iid"}
    host.vis = ["u1", "u2"]
    host._rc_sel_iid = "u1"
    assert pol.kind_of("u1") == "user", f"{test_description}: kind must be user"
    assert pol.insert_permission(("user", None)) == "deny", f"{test_description}: user insert denied"
    host.sh.selected_rows = [0, 1]
    assert pol.can_delete_rows() is True, f"{test_description}: all-user selection must delete"
    host.sh.selected_rows = [1]
    host.vis = ["u1", "iid"]
    assert pol.can_delete_rows() is False, f"{test_description}: mixed selection must not delete"
    host.sh.selected_rows = []
    assert pol.can_delete_rows() is False, f"{test_description}: empty selection must not delete"


def test_delete_columns_only_trailing_appends(test_description="column gate"):
    """``Delete columns`` passes only for appended trailing columns (``>= base``)."""
    host, pol = _policy_host({"parent": "", "key": "input"})
    assert pol.can_delete_columns() is False, f"{test_description}: no base → deny"
    host._user_col_base = 6
    host.sh.MT.selected_cols = [6, 7]
    assert pol.can_delete_columns() is True, f"{test_description}: trailing cols must delete"
    host.sh.MT.selected_cols = [5, 6]
    assert pol.can_delete_columns() is False, f"{test_description}: mixed cols must not delete"
    host.sh.MT.selected_cols = []
    host.sh.MT.CH.popup_menu_loc = 6
    assert pol.can_delete_columns() is True, f"{test_description}: popup col counts as target"


def test_readonly_denies_everything(test_description="readonly sheet"):
    """Simplified mode without data — every verdict denies."""
    host, pol = _policy_host()
    host._readonly = True
    assert pol.insert_permission(("setup", 0)) == "deny", f"{test_description}: insert"
    assert pol.can_delete_rows() is False, f"{test_description}: delete rows"
    assert pol.can_delete_columns() is False, f"{test_description}: delete cols"
    assert pol.can_child_add() is False, f"{test_description}: child add"


# ------------------------------------------------------------- SheetGuard
def test_guard_denies_with_bell_and_allows_hydra(test_description="method guard"):
    """Wrapped mutation methods bell+``None`` on deny, pass through on allow."""
    host, pol = _policy_host({"parent": "input", "key": "path"})
    tree = MetadataTree(host)
    host.sh.real_insert = lambda *a, **k: "inserted"
    host.sh.real_delete = lambda *a, **k: "deleted"
    guard = SheetGuard.__new__(SheetGuard)
    guard._tree, guard._policy, guard._host, guard._patched = tree, pol, host, []
    guard._wrap(host.sh, "real_insert", pol.can_native_insert)
    guard._wrap(host.sh, "real_delete", pol.can_delete_rows)
    assert host.sh.real_insert() == "inserted", f"{test_description}: hydra insert must pass"
    assert host.sh.real_delete() is None, f"{test_description}: hydra delete must deny"
    assert host.sh.bells == 1, f"{test_description}: deny must bell"
    host._loading = True  # internal rebuilds bypass
    assert host.sh.real_delete() == "deleted", f"{test_description}: loading must bypass"
    host._loading = False
    with guard.suspended():
        assert host.sh.real_delete() == "deleted", f"{test_description}: suspension must bypass"


class StubOwner:
    """tksheet-shaped method surface for guard rule tests."""

    def __init__(self):
        self.calls: list = []

    def rc_add_columns(self, *a, **k):
        self.calls.append("cols")
        return "cols"

    def delete_rows(self, *a, **k):
        self.calls.append("rows")
        return "rows"

    def insert_rows(self, *a, **k):
        self.calls.append("insert")
        return "insert"


def test_guard_leaves_tree_construction_open(test_description="programmatic insert"):
    """``Sheet.insert_rows`` is never a guard rule — tree building (fixtures included) keeps working."""
    host, pol = _policy_host({"parent": "input", "key": "path"})
    guard = SheetGuard(MetadataTree(host), pol)
    assert guard._rule("del_rows").__func__ is pol.can_delete_rows.__func__, (
        f"{test_description}: row delete gate"
    )
    assert guard._rule("del_columns").__func__ is pol.can_delete_columns.__func__, (
        f"{test_description}: col delete gate"
    )
    assert guard._never() is False, f"{test_description}: native col insert always denied"
    stub = StubOwner()
    guard._patch_owner(stub, (("rc_add_columns", guard._never), ("delete_rows", pol.can_delete_rows)))
    assert stub.rc_add_columns() is None and stub.calls == [], f"{test_description}: col insert denied"
    assert stub.insert_rows() == "insert", f"{test_description}: bare insert untouched"


def test_guard_skips_missing_and_double_wrap(test_description="guard robustness"):
    """Unknown tksheet versions degrade gracefully; double-wrap is a no-op."""
    host, pol = _policy_host()
    guard = SheetGuard(MetadataTree(host), pol)
    guard._patch_owner(
        host.sh, (("insert_row", pol.can_native_insert), ("no_such_method", pol.can_delete_rows))
    )
    before = list(guard._patched)
    guard._patch_owner(
        host.sh, (("insert_row", pol.can_native_insert), ("no_such_method", pol.can_delete_rows))
    )
    assert guard._patched == before, f"{test_description}: re-patch must not duplicate"


# ------------------------------------------------------------- menu state
def _menu4(labels=("Cut", "Insert rows above", "Insert rows below", "Delete rows", "Delete columns")):
    return FakeMenu(list(labels))


@pytest.mark.parametrize(
    ("ref", "above_state", "below_state"),
    [
        ({"is_metadata_root": True}, "normal", "normal"),
        ({"is_setup": True, "setup_idx": 0}, "normal", "normal"),
        ({"is_metadata": True, "setup_idx": 0}, "disabled", "disabled"),
        ({"parent": "", "key": "input"}, "disabled", "disabled"),
        ({"parent": "input", "key": "path"}, "normal", "normal"),
    ],
    ids=["meta", "setup", "leaf", "top", "hydra"],
)
def test_menu_insert_state_per_kind(ref, above_state, below_state, test_description="menu insert state"):
    """Above/below enabled only for split targets and hydra children — never leaves/tops."""
    host, pol = _policy_host(ref)
    menu = _menu4()
    assert refresh_menu_state(menu, MetadataTree(host), pol) is True, f"{test_description}: must change"
    # Above/below sit at fixed indexes (locale-independent); split targets relabel them.
    assert menu.entrycget(1, "state") == above_state, f"{test_description}: above ({menu.labels[1]!r})"
    assert menu.entrycget(2, "state") == below_state, f"{test_description}: below ({menu.labels[2]!r})"
    if ref.get("is_metadata_root") or ref.get("is_setup"):
        assert menu.labels[1] != "Insert rows above", f"{test_description}: split must relabel"
        assert menu.labels[2] != "Insert rows below", f"{test_description}: split must relabel"
    assert menu.state_of("Delete rows") == "disabled", f"{test_description}: delete stays off"


def test_menu_delete_rows_enabled_for_user_selection(test_description="menu delete state"):
    """``Delete rows`` flips on only when the selection is all-user."""
    host, pol = _policy_host({"parent": "input", "key": "path"})
    host._user_rows, host.sh.existing = {"u1"}, {"u1"}
    host.vis, host._rc_sel_iid, host.sh.selected_rows = ["u1"], "u1", [0]
    menu = _menu4()
    refresh_menu_state(menu, MetadataTree(host), pol)
    assert menu.state_of("Delete rows") == "normal", f"{test_description}: user selection must enable"
    host.sh.selected_rows = []
    host._rc_sel_iid = "missing"
    host.vis = []
    refresh_menu_state(menu, MetadataTree(host), pol)
    assert menu.state_of("Delete rows") == "disabled", f"{test_description}: empty must disable"


def test_menu_fully_disabled_when_readonly(test_description="readonly menu"):
    """Simplified mode without data — every entry, including navigation-adjacent ones, disabled."""
    host, pol = _policy_host()
    host._readonly = True
    menu = _menu4()
    refresh_menu_state(menu, MetadataTree(host), pol)
    for label in menu.labels:
        assert menu.state_of(label) == "disabled", f"{test_description}: {label!r} must be disabled"


def test_menu_native_column_entries_always_disabled(test_description="append-only columns"):
    """Left/right column inserts never run — the ``Add column`` extra is the only path."""
    host, pol = _policy_host({"parent": "input", "key": "path"})
    menu = FakeMenu(["Insert columns left", "Insert columns right", "Delete columns", "Add column"])
    refresh_menu_state(menu, MetadataTree(host), pol)
    assert menu.state_of("Insert columns left") == "disabled", f"{test_description}: left"
    assert menu.state_of("Insert columns right") == "disabled", f"{test_description}: right"
    assert menu.state_of("Add column") == "normal", f"{test_description}: extra stays"


def test_structure_hook_rebuilds_and_clears(test_description="insert follow-up"):
    """Post-insert hook hides overlays, clears stale status and rebuilds row caches."""
    host = FakeHost()
    tree = MetadataTree(host)
    host._hide_hover_field = lambda: setattr(host, "hidden", True)
    assert popup._install_structure_hook(tree) is True, f"{test_description}: install"
    assert popup._install_structure_hook(tree) is True, f"{test_description}: re-install no-op"
    host.sh.MT.extra_end_insert_rows_rc_func({})
    assert host.rebuilt >= 1 and host.cleared == 1 and host.hidden is True, (
        f"{test_description}: rebuilt={host.rebuilt} cleared={host.cleared}"
    )


def test_add_extras_use_add_labels(test_description="extras rename"):
    """Extras register under Add (append) labels in both chrome string tables."""
    for fname, col, row in (
        ("str.yaml", "Add column", "Add row"),
        ("str_ru.yaml", "Добавить столбец", "Добавить строку"),
    ):
        data = yaml.safe_load((_STR_DIR / fname).read_text(encoding="utf-8"))
        assert data.get("sheet.insert_col") == col, f"{test_description}: {fname} col"
        assert data.get("sheet.insert_row") == row, f"{test_description}: {fname} row"
