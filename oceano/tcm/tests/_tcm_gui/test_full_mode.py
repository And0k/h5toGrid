"""Full (non-simplified) GUI mode — all Config sections visible, editable, persisted.

Regression: a Shift-started GUI showed only ``input`` — the placeholder page used
``default_cfg()`` (input-only) and ``_build_full`` rendered whatever keys the
dict carried, leaking internal keys (``_page_stem``) as rows. Edits to
``out``/``filter``/``program`` additionally never reached the run YAML.
"""

from __future__ import annotations

import sys

import pytest

from tcm import config_yaml, schema
from tcm_gui.cli_cfg import SIMPLE_OUT_DEFAULTS, default_cfg, ensure_full_cfg, full_default_cfg
from tcm_gui.runtime import Runtime
from tcm_gui.worker import Worker

_ALL_SECTIONS = ("input", "out", "filter", "program")


@pytest.mark.gui
class TestFullDefaults:
    @pytest.mark.parametrize(
        ("factory", "expected", "test_description"),
        [
            pytest.param(
                default_cfg,
                ["input"],
                "simple-mode placeholder stays input-only",
                id="simple-stays-input-only",
            ),
            pytest.param(
                full_default_cfg,
                list(_ALL_SECTIONS),
                "full-mode placeholder carries every Config section",
                id="full-has-all-sections",
            ),
        ],
    )
    def test_section_keys(self, factory, expected, test_description):
        """Section keys of default cfgs — {test_description}."""
        assert sorted(factory()) == sorted(expected), (
            f"{test_description}: keys={sorted(factory())} != {sorted(expected)}"
        )

    def test_ensure_full_cfg_backfills_thin_dict(self):
        """Input-only dict gains all sections; existing values are preserved."""
        cfg = {"input": {"path": "D:/data/_raw/i_01.txt"}}
        out = ensure_full_cfg(cfg)
        assert sorted(out) == sorted(_ALL_SECTIONS), f"sections={sorted(out)}"
        assert out["input"]["path"] == "D:/data/_raw/i_01.txt", "input.path must survive backfill"
        assert out["out"].get("text_path") is not None, "out leaves must be filled with defaults"

    def test_ensure_full_cfg_keeps_overrides(self):
        """Pre-set values win over structured defaults."""
        cfg = {"input": {"path": "p"}, "out": {"text_path": "custom"}}
        out = ensure_full_cfg(cfg)
        assert out["out"]["text_path"] == "custom", f"text_path={out['out']['text_path']!r} overwritten"


@pytest.mark.gui
class TestSimplifiedOutDefaults:
    """Simplified (Shift-less) mode injects ``out`` binning defaults at composition.

    `Worker._out_overrides` feeds `SIMPLE_OUT_DEFAULTS` into Scan/Run — full
    resolution only — unless the launch CLI already overrides a `dt_bins` key
    or the GUI started in full mode.
    """

    @staticmethod
    def _overrides(full_mode: bool) -> dict:
        rt = Runtime()
        rt.full_mode = full_mode
        return Worker(rt)._out_overrides()

    def test_defaults_pinned(self):
        assert SIMPLE_OUT_DEFAULTS == {"dt_bins": [0], "dt_bins_min_save_text": 0}

    def test_injected_when_simplified(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["tcm_gui"])
        assert self._overrides(False) == {"out": SIMPLE_OUT_DEFAULTS}

    def test_skipped_when_full_mode(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["tcm_gui"])
        assert self._overrides(True) == {}, "full mode must keep schema defaults"

    @pytest.mark.parametrize("argv", [["out.dt_bins=[0,2,600]"], ["+out.dt_bins_min_save_text=2"]])
    def test_skipped_when_cli_overrides(self, monkeypatch, argv):
        monkeypatch.setattr(sys, "argv", ["tcm_gui", *argv])
        assert self._overrides(False) == {}, f"CLI override must win: {argv}"


def _load_full_sheet(root, cfg):
    """Real ConfigSheet with full-mode tree; caller must destroy ``sheet.sh``."""
    from tcm_gui.coef_sheet import ConfigSheet

    sheet = ConfigSheet(root)
    sheet.sh.pack(fill="both", expand=True)
    sheet.load(cfg, full=True, config_root=schema.Config, return_enum=schema.Return)
    root.update_idletasks()
    return sheet


@pytest.mark.gui
class TestFullTree:
    def test_all_sections_rendered(self, _session_tk_root):
        """Full tree contains input/out/filter/program rows — {stem} placeholder."""
        import tkinter as tk

        if _session_tk_root is None:
            pytest.skip("Tk not available")
        root = _session_tk_root
        try:
            root.deiconify()
        except tk.TclError:
            pytest.skip("Tk not available")
        sheet = _load_full_sheet(root, full_default_cfg())
        try:
            tops = {m.get("path", "").split(".", 1)[0] for m in sheet._meta.values() if m.get("path")}
            missing = [s for s in _ALL_SECTIONS if s not in tops]
            assert not missing, f"full tree missing sections: {missing}"
        finally:
            sheet.sh.destroy()

    def test_thin_cfg_renders_all_sections(self, _session_tk_root):
        """Backfilled input-only cfg still renders every section (scan-thin-YAML path)."""
        import tkinter as tk

        if _session_tk_root is None:
            pytest.skip("Tk not available")
        root = _session_tk_root
        try:
            root.deiconify()
        except tk.TclError:
            pytest.skip("Tk not available")
        sheet = _load_full_sheet(root, ensure_full_cfg(default_cfg()))
        try:
            tops = {m.get("path", "").split(".", 1)[0] for m in sheet._meta.values() if m.get("path")}
            missing = [s for s in _ALL_SECTIONS if s not in tops]
            assert not missing, f"backfilled tree missing sections: {missing}"
        finally:
            sheet.sh.destroy()

    def test_internal_keys_not_rendered(self, _session_tk_root):
        """``_page_stem``/``hydra``-style keys never become tree rows."""
        import tkinter as tk

        if _session_tk_root is None:
            pytest.skip("Tk not available")
        root = _session_tk_root
        try:
            root.deiconify()
        except tk.TclError:
            pytest.skip("Tk not available")
        cfg = full_default_cfg()
        cfg["_page_stem"] = "i_01"
        cfg["hydra"] = {"run": {"dir": "x"}}
        sheet = _load_full_sheet(root, cfg)
        try:
            bad = [
                m.get("path")
                for m in sheet._meta.values()
                if (m.get("path") or "").split(".", 1)[0] in ("_page_stem", "hydra", "defaults")
            ]
            assert not bad, f"internal keys rendered as rows: {bad}"
        finally:
            sheet.sh.destroy()

    def test_clean_load_has_empty_patch(self, _session_tk_root):
        """Defaults-only load → ``get_edited_full()`` is empty (at-default omitted)."""
        import tkinter as tk

        if _session_tk_root is None:
            pytest.skip("Tk not available")
        root = _session_tk_root
        try:
            root.deiconify()
        except tk.TclError:
            pytest.skip("Tk not available")
        sheet = _load_full_sheet(root, full_default_cfg())
        try:
            assert sheet.get_edited_full() == {}, (
                f"clean load must give empty patch, got {sheet.get_edited_full()}"
            )
        finally:
            sheet.sh.destroy()

    def test_edited_out_cell_in_patch(self, _session_tk_root):
        """Edited ``out.text_path`` cell lands in the patch; untouched sections stay out."""
        import tkinter as tk

        if _session_tk_root is None:
            pytest.skip("Tk not available")
        root = _session_tk_root
        try:
            root.deiconify()
        except tk.TclError:
            pytest.skip("Tk not available")
        sheet = _load_full_sheet(root, full_default_cfg())
        try:
            iid = next(i for i, m in sheet._meta.items() if m.get("path") == "out.text_path")
            sheet.sh.set_cell_data(sheet._internal_row(iid), 0, "custom_out", redraw=False)
            patch = sheet.get_edited_full()
            assert patch == {"out": {"text_path": "custom_out"}}, f"unexpected patch: {patch}"
        finally:
            sheet.sh.destroy()


@pytest.mark.gui
class TestFullWriteBack:
    def test_update_run_yaml_merges_sections(self, tmp_path):
        """Generic patch merges; untouched sections and the header survive."""
        yp = tmp_path / "run.yaml"
        yp.write_text(
            "# @package _global_\ninput:\n  path: 'D:/d/_raw/i_01.txt'\nout:\n  text_path: 'text_output'\n",
            encoding="utf-8",
        )
        config_yaml.update_run_yaml(yp, {"filter": {"max": {"g_minus_1": 2}}, "out": {"text_path": "custom"}})
        with yp.open(encoding="utf-8") as f:
            data = config_yaml._ry(write=False).load(f)
        assert data["input"]["path"] == "D:/d/_raw/i_01.txt", "input.path must be preserved"
        assert data["out"]["text_path"] == "custom", f"out not merged: {data['out']}"
        assert data["filter"]["max"]["g_minus_1"] == 2, f"filter not merged: {data.get('filter')}"
        assert yp.read_text(encoding="utf-8").startswith("# @package _global_"), "package header lost"

    def test_update_coefs_keeps_flat_contract(self, tmp_path):
        """Flat coefs mapping still lands under ``input.coefs`` (no ``input`` nesting)."""
        yp = tmp_path / "run.yaml"
        yp.write_text("# @package _global_\ninput:\n  path: 'p'\n", encoding="utf-8")
        config_yaml.update_coefs_in_run_yaml(yp, {"Ag": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]})
        with yp.open(encoding="utf-8") as f:
            data = config_yaml._ry(write=False).load(f)
        assert data["input"]["coefs"]["Ag"][0][0] == 1, f"coefs misplaced: {data['input'].get('coefs')}"
        assert "input" not in (data["input"].get("coefs") or {}), "nested input.input.coefs regression"

    def test_app_write_coefs_persists_full_sections(self, _session_tk_root, tmp_path):
        """``App._write_coefs`` in full mode persists out edits alongside coefs."""
        import tkinter as tk

        if _session_tk_root is None:
            pytest.skip("Tk not available")
        root = _session_tk_root
        try:
            root.deiconify()
        except tk.TclError:
            pytest.skip("Tk not available")
        from tcm_gui.app import App

        yp = tmp_path / "i_01.yaml"
        yp.write_text("# @package _global_\ninput:\n  path: 'D:/d/_raw/i_01.txt'\n", encoding="utf-8")
        sheet = _load_full_sheet(root, full_default_cfg())
        try:
            iid = next(i for i, m in sheet._meta.items() if m.get("path") == "out.text_path")
            sheet.sh.set_cell_data(sheet._internal_row(iid), 0, "custom_out", redraw=False)
            assert sheet.is_dirty, "edited sheet must be dirty"
            app = App.__new__(App)
            app._yaml_paths = {"i_01": yp}
            app._full_mode = True
            app._write_coefs("i_01", sheet)
            with yp.open(encoding="utf-8") as f:
                data = config_yaml._ry(write=False).load(f)
            assert data["out"]["text_path"] == "custom_out", f"out edit lost: {data.get('out')}"
            assert not sheet.is_dirty, "sheet must be clean after write"
        finally:
            sheet.sh.destroy()

    def test_dt_bins_roundtrip_stays_int(self, _session_tk_root):
        """``out.dt_bins: list[int]`` sheet edit lands in the patch as ``int`` (not ``float``)."""
        import tkinter as tk

        if _session_tk_root is None:
            pytest.skip("Tk not available")
        root = _session_tk_root
        try:
            root.deiconify()
        except tk.TclError:
            pytest.skip("Tk not available")
        sheet = _load_full_sheet(root, full_default_cfg())
        try:
            rows = [(i, m) for i, m in sheet._meta.items() if m.get("path") == "out.dt_bins"]
            assert rows, "out.dt_bins row missing from full tree"
            iid, m = rows[0]
            r = sheet._internal_row(iid)
            for j, v in enumerate(["0", "600"]):
                sheet.sh.set_cell_data(r, j, v, redraw=False)
            for j in range(2, int(m.get("max_col") or 6)):
                sheet.sh.set_cell_data(r, j, "", redraw=False)
            patch = sheet.get_edited_full()
            assert patch["out"]["dt_bins"] == [0, 600], f"unexpected patch: {patch}"
            assert all(type(x) is int for x in patch["out"]["dt_bins"]), f"int lost: {patch}"
        finally:
            sheet.sh.destroy()

    def test_calib_roundtrip_via_generic(self, _session_tk_root):
        """``input.calib.*`` rows go through the generic reader with ``sections=None``."""
        import tkinter as tk

        if _session_tk_root is None:
            pytest.skip("Tk not available")
        root = _session_tk_root
        try:
            root.deiconify()
        except tk.TclError:
            pytest.skip("Tk not available")
        cfg = full_default_cfg()
        cfg["input"]["calib"] = {"g0xyz": [1.0, 2.0, 3.0], "azimuth_add": 1.5}
        sheet = _load_full_sheet(root, cfg)
        try:
            by_path = {m.get("path"): i for i, m in sheet._meta.items() if m.get("path")}
            assert "input.calib.g0xyz" in by_path, "calib g0xyz row missing"
            iid = by_path["input.calib.azimuth_add"]
            sheet.sh.set_cell_data(sheet._internal_row(iid), 0, "5", redraw=False)
            patch = sheet.get_edited_full(None)
            assert patch["input"]["calib"]["azimuth_add"] == 5.0, f"unexpected patch: {patch}"
        finally:
            sheet.sh.destroy()


class TestContainerNotEditable:
    """Container rows (section roots, dict parents) reject edits at every gate."""

    _ROOTS = ("out", "filter", "program")

    def test_container_meta_max_col_zero(self, _session_tk_root):
        """Full-mode section roots render as non-editable rows (max_col=0)."""
        import tkinter as tk

        if _session_tk_root is None:
            pytest.skip("Tk not available")
        root = _session_tk_root
        try:
            root.deiconify()
        except tk.TclError:
            pytest.skip("Tk not available")
        sheet = _load_full_sheet(root, full_default_cfg())
        try:
            for sec in self._ROOTS:
                iid = next(i for i, m in sheet._meta.items() if m.get("path") == sec)
                assert sheet._meta[iid]["max_col"] == 0, f"{sec} root must be non-editable"
        finally:
            sheet.sh.destroy()

    def test_on_begin_edit_vetoes_container(self, _session_tk_root):
        """``_on_begin_edit_cell`` refuses to open an editor on a container cell."""
        import tkinter as tk
        from types import SimpleNamespace

        if _session_tk_root is None:
            pytest.skip("Tk not available")
        root = _session_tk_root
        try:
            root.deiconify()
        except tk.TclError:
            pytest.skip("Tk not available")
        sheet = _load_full_sheet(root, full_default_cfg())
        try:
            for sec in self._ROOTS:
                iid = next(i for i, m in sheet._meta.items() if m.get("path") == sec)
                r = sheet._internal_row(iid)
                ev = SimpleNamespace(row=r, column=0)
                assert sheet._on_begin_edit_cell(ev) is None, f"{sec} container editor must not open"
        finally:
            sheet.sh.destroy()

    def test_on_edit_rejects_container_value(self, _session_tk_root):
        """``_on_edit`` rejects a typed value on a container cell (no silent drop)."""
        import tkinter as tk
        from types import SimpleNamespace

        if _session_tk_root is None:
            pytest.skip("Tk not available")
        root = _session_tk_root
        try:
            root.deiconify()
        except tk.TclError:
            pytest.skip("Tk not available")
        sheet = _load_full_sheet(root, full_default_cfg())
        try:
            for sec in self._ROOTS:
                iid = next(i for i, m in sheet._meta.items() if m.get("path") == sec)
                r = sheet._internal_row(iid)
                ev = SimpleNamespace(row=r, column=0, value="x", eventname="cell_edited")
                assert sheet._on_edit(ev) is None, f"{sec} container value must be REJECTED"
        finally:
            sheet.sh.destroy()

    def test_container_predicate(self, _session_tk_root):
        """``_is_container_row`` covers section roots/dict parents, never browse/date rows."""
        import tkinter as tk

        if _session_tk_root is None:
            pytest.skip("Tk not available")
        root = _session_tk_root
        try:
            root.deiconify()
        except tk.TclError:
            pytest.skip("Tk not available")
        sheet = _load_full_sheet(root, full_default_cfg())
        try:
            by_path = {m.get("path"): i for i, m in sheet._meta.items() if m.get("path")}
            assert sheet._is_container_row(sheet._meta[by_path["out"]]), "section root is a container"
            assert sheet._is_container_row(sheet._meta[by_path["filter"]]), "section root is a container"
            browse = next(m for m in sheet._meta.values() if m.get("browse"))
            assert not sheet._is_container_row(browse), "browse row edits"
            ag_parent = next(m for m in sheet._meta.values() if m.get("key") == "Ag")
            assert not sheet._is_container_row(ag_parent), "2d date parent keeps date routing"
        finally:
            sheet.sh.destroy()

    def test_double_click_toggles_container(self, _session_tk_root):
        """Double-click on a disabled row's cells expands/collapses it (editor stays vetoed)."""
        import tkinter as tk
        from types import SimpleNamespace

        if _session_tk_root is None:
            pytest.skip("Tk not available")
        root = _session_tk_root
        try:
            root.deiconify()
        except tk.TclError:
            pytest.skip("Tk not available")
        sheet = _load_full_sheet(root, full_default_cfg())
        try:
            iid = next(i for i, m in sheet._meta.items() if m.get("path") == "out")
            mt = sheet.sh.MT
            r = sheet._vis.index(iid)  # row_positions is DISPLAY-row indexed
            assert not sheet._is_open(iid), "section root starts collapsed"
            ev = SimpleNamespace(
                x=int(mt.col_positions[0]) + 2,
                y=int(mt.row_positions[r]) + 2,
                state=0,
            )
            sheet._redirect_overflow_double(ev)
            assert sheet._is_open(iid), "double-click must EXPAND the container"
            assert not getattr(mt.text_editor, "open", False), "editor must stay vetoed"
            # Top-level row — its own y is stable (children open BELOW it)
            sheet._redirect_overflow_double(ev)
            assert not sheet._is_open(iid), "second double-click must COLLAPSE it back"
        finally:
            sheet.sh.destroy()

    def test_tree_label_click_toggles_parent(self, _session_tk_root):
        """Single click on a parent node's tree label (not the arrow) toggles it."""
        import tkinter as tk
        from types import SimpleNamespace

        if _session_tk_root is None:
            pytest.skip("Tk not available")
        root = _session_tk_root
        try:
            root.deiconify()
        except tk.TclError:
            pytest.skip("Tk not available")
        sheet = _load_full_sheet(root, full_default_cfg())
        try:
            ri = sheet.sh.RI
            iid = next(i for i, m in sheet._meta.items() if m.get("path") == "out")
            assert list(sheet.sh.get_children(iid)), "section root must have children"
            r = sheet._vis.index(iid)
            # Click on the label area (right of the arrow glyph) — same y band.
            ev = SimpleNamespace(
                x=int(ri.current_width) - 4,
                y=int(sheet.sh.MT.row_positions[r]) + 2,
            )
            assert not sheet._is_open(iid), "section root starts collapsed"
            sheet._on_tree_col_click(ev)
            assert sheet._is_open(iid), "tree-label click must EXPAND the parent"
            sheet._on_tree_col_click(ev)
            assert not sheet._is_open(iid), "second click must COLLAPSE it back"
        finally:
            sheet.sh.destroy()

    def test_tree_label_click_leaf_inert(self, _session_tk_root):
        """Clicking a leaf's tree label does not toggle (nothing to expand)."""
        import tkinter as tk
        from types import SimpleNamespace

        if _session_tk_root is None:
            pytest.skip("Tk not available")
        root = _session_tk_root
        try:
            root.deiconify()
        except tk.TclError:
            pytest.skip("Tk not available")
        sheet = _load_full_sheet(root, full_default_cfg())
        try:
            ri = sheet.sh.RI
            # Expand `out` first so its leaf children are reachable & visible.
            parent = next(i for i, m in sheet._meta.items() if m.get("path") == "out")
            sheet.sh.item(parent, open_=True, undo=False)
            sheet._rebuild_row_caches()
            leaf = next(i for i, m in sheet._meta.items() if m.get("path") == "out.text_path")
            assert not list(sheet.sh.get_children(leaf)), "text_path is a leaf"
            r = sheet._vis.index(leaf)
            ev = SimpleNamespace(
                x=int(ri.current_width) - 4,
                y=int(sheet.sh.MT.row_positions[r]) + 2,
            )
            # Leaf has no children → handler must be a no-op (no exception, no toggle).
            sheet._on_tree_col_click(ev)
        finally:
            sheet.sh.destroy()


@pytest.mark.gui
class TestInstantApplyBoxes:
    def test_boxes_synced_on_defaults(self, _session_tk_root):
        """Default calib (no triggers) renders boxes, all synced."""
        import tkinter as tk

        if _session_tk_root is None:
            pytest.skip("Tk not available")
        root = _session_tk_root
        try:
            root.deiconify()
        except tk.TclError:
            pytest.skip("Tk not available")
        sheet = _load_full_sheet(root, full_default_cfg())
        try:
            boxes = sheet._apply_boxes
            assert set(boxes) == {"g0xyz", "coordinates", "azimuth_add"}, f"boxes={sorted(boxes)}"
            assert boxes["g0xyz"]["pending"] is False, "empty g0xyz must be synced"
            assert boxes["coordinates"]["pending"] is False, "empty coords must be synced"
            assert boxes["azimuth_add"]["pending"] is False, "default add must be synced"
        finally:
            sheet.sh.destroy()

    def test_boxes_pending_on_trigger_data(self, _session_tk_root):
        """g0xyz values flip only its own box to pending; others stay synced."""
        import tkinter as tk

        if _session_tk_root is None:
            pytest.skip("Tk not available")
        root = _session_tk_root
        try:
            root.deiconify()
        except tk.TclError:
            pytest.skip("Tk not available")
        cfg = full_default_cfg()
        cfg["input"]["calib"] = {**cfg["input"]["calib"], "g0xyz": [1.0, 2.0, 3.0]}
        sheet = _load_full_sheet(root, cfg)
        try:
            assert sheet._apply_boxes["g0xyz"]["pending"] is True, "g0xyz data must pend"
            assert sheet._apply_boxes["coordinates"]["pending"] is False, "coords must stay synced"
            assert sheet._apply_boxes["azimuth_add"]["pending"] is False, "add must stay synced"
        finally:
            sheet.sh.destroy()

    def test_add_box_independent_of_coordinates(self, _session_tk_root):
        """azimuth_add flips only its own box; coordinates stay synced."""
        import tkinter as tk

        if _session_tk_root is None:
            pytest.skip("Tk not available")
        root = _session_tk_root
        try:
            root.deiconify()
        except tk.TclError:
            pytest.skip("Tk not available")
        cfg = full_default_cfg()
        cfg["input"]["calib"] = {**cfg["input"]["calib"], "azimuth_add": 2.5}
        sheet = _load_full_sheet(root, cfg)
        try:
            assert sheet._apply_boxes["azimuth_add"]["pending"] is True, "add data must pend"
            assert sheet._apply_boxes["coordinates"]["pending"] is False, "coords must stay synced"
            assert sheet._apply_boxes["g0xyz"]["pending"] is False, "g0xyz must stay synced"
        finally:
            sheet.sh.destroy()

    def test_boxes_render_without_calib_key(self, _session_tk_root):
        """Consumed YAML (no input.calib) still renders triggers + synced boxes."""
        import tkinter as tk

        if _session_tk_root is None:
            pytest.skip("Tk not available")
        root = _session_tk_root
        try:
            root.deiconify()
        except tk.TclError:
            pytest.skip("Tk not available")
        cfg = full_default_cfg()
        cfg["input"].pop("calib", None)
        sheet = _load_full_sheet(root, cfg)
        try:
            assert sheet._calib_iid("input.calib.g0xyz") is not None, "g0xyz row must render"
            assert set(sheet._apply_boxes) == {"g0xyz", "coordinates", "azimuth_add"}, "boxes must exist"
        finally:
            sheet.sh.destroy()

    def test_each_box_anchors_on_own_row(self, _session_tk_root):
        """Every apply box lives on its own trigger row."""
        import tkinter as tk

        if _session_tk_root is None:
            pytest.skip("Tk not available")
        root = _session_tk_root
        try:
            root.deiconify()
        except tk.TclError:
            pytest.skip("Tk not available")
        sheet = _load_full_sheet(root, full_default_cfg())
        try:
            for kind, path in (
                ("g0xyz", "input.calib.g0xyz"),
                ("coordinates", "input.calib.coordinates"),
                ("azimuth_add", "input.calib.azimuth_add"),
            ):
                iid = sheet._calib_iid(path)
                assert iid is not None, f"{path} row must render"
                assert sheet._apply_boxes[kind]["iid"] == iid, f"{kind} box must anchor on own row"
        finally:
            sheet.sh.destroy()

    def test_partial_trigger_stays_disabled(self, _session_tk_root):
        """Partial g0xyz is incomplete — box disabled, no misleading apply."""
        import tkinter as tk

        if _session_tk_root is None:
            pytest.skip("Tk not available")
        root = _session_tk_root
        try:
            root.deiconify()
        except tk.TclError:
            pytest.skip("Tk not available")
        cfg = full_default_cfg()
        cfg["input"]["calib"] = {**cfg["input"]["calib"], "g0xyz": ["", "", "1"]}
        sheet = _load_full_sheet(root, cfg)
        try:
            box = sheet._apply_boxes["g0xyz"]
            assert box["state"] == "incomplete", f"partial must be incomplete: {box}"
            assert box["pending"] is False, "partial must not enable"
            assert sheet.calib_blocking() is True, "partial must block Run"
            g0 = sheet._calib_iid("input.calib.g0xyz")
            assert sheet._node_has_error(g0) is True, "trigger row must flag error"
            assert sheet._node_has_error(sheet._calib_iid("input.calib")) is True, "error propagates up"
        finally:
            sheet.sh.destroy()

    def test_clean_triggers_not_blocking(self, _session_tk_root):
        """Empty or complete triggers never block Run."""
        import tkinter as tk

        if _session_tk_root is None:
            pytest.skip("Tk not available")
        root = _session_tk_root
        try:
            root.deiconify()
        except tk.TclError:
            pytest.skip("Tk not available")
        sheet = _load_full_sheet(root, full_default_cfg())
        try:
            assert sheet.calib_blocking() is False, "defaults must not block"
            assert sheet._node_has_error(sheet._calib_iid("input.calib")) is False
        finally:
            sheet.sh.destroy()
