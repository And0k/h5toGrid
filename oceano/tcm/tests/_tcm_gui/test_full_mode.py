"""Full (non-simplified) GUI mode — all Config sections visible, editable, persisted.

Regression: a Shift-started GUI showed only ``input`` — the placeholder page used
``default_cfg()`` (input-only) and ``_build_full`` rendered whatever keys the
dict carried, leaking internal keys (``_page_stem``) as rows. Edits to
``out``/``filter``/``program`` additionally never reached the run YAML.
"""

from __future__ import annotations

import pytest

from tcm import config_yaml, schema
from tcm_gui.cli_cfg import default_cfg, ensure_full_cfg, full_default_cfg

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
