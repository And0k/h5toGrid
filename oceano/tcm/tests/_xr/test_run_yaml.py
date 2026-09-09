"""Tests for tcm/config_yaml.py — run YAML generation and loading."""

from __future__ import annotations

import re

import pytest

from tcm._constants import RAW_DIR_NAME
from tcm.config_yaml import find_stale_cfgs, get_existed_cfgs, save_config_to_yaml


@pytest.mark.xr
class TestSaveRunYaml:
    """save_config_to_yaml writes to cfg_proc/run/{source_stem}.yaml."""

    @pytest.mark.parametrize(
        "stem",
        [
            pytest.param("@i_01", id="corrected-stem"),
        ],
    )
    def test_creates_file_at_project_level(self, tmp_path, mocker, stem):
        """YAML written to project_dir/cfg_proc/run/ (parent of _raw/)."""
        raw_dir = tmp_path / RAW_DIR_NAME
        raw_dir.mkdir()
        src_file = raw_dir / f"{stem}.txt"
        src_file.write_text("dummy\n")

        cfg = {
            "input": {
                "path": src_file,
                "tables": ["incl*"],
                "coefs": {"path": None},
                "coefs": {},
                "corr_time_mode": True,
            },
            "filter": {},
            "out": {"dt_bins": [0], "table": ""},
        }
        cfg1 = {
            "input": {"path": str(src_file), "coefs": {}, "corr_time_mode": True},
            "out": {"dt_bins": [0]},
            "filter": {},
        }
        mocker.patch(
            "tcm.config_yaml.gen_metadata",
            return_value=iter([(cfg1, (False, "i_01", None))]),
        )

        save_config_to_yaml(cfg, [src_file])

        # Configs at _raw/cfg_proc/run/ (not project_dir)
        run_dir = raw_dir / "cfg_proc" / "run"
        assert run_dir.is_dir(), f"cfg_proc/run/ not created at {run_dir}"
        yaml_files = list(run_dir.glob("*.yaml"))
        assert len(yaml_files) >= 1, "No YAML files created"
        for f in yaml_files:
            assert not re.search(r"_\d{8}_\d{6}", f.stem), (
                f"Filename {f.name} contains full timestamp — expected stem-only naming"
            )

    def test_dated_name_when_time_ranges(self, tmp_path, mocker):
        """cfg1 with time_ranges[0] → filename ``{yymmdd_hhmm}@{pcid}.yaml``."""
        raw_dir = tmp_path / RAW_DIR_NAME
        raw_dir.mkdir()
        src_file = raw_dir / "@i_01.txt"
        src_file.write_text("dummy\n")

        cfg = {
            "input": {
                "path": src_file,
                "tables": ["incl*"],
                "coefs": {"path": None},
                "coefs": {},
                "corr_time_mode": True,
            },
            "filter": {},
            "out": {"dt_bins": [0], "table": ""},
        }
        cfg1 = {
            "input": {
                "path": str(src_file),
                "coefs": {},
                "time_ranges": ["2025-07-05T14:19:00", "2025-07-05T15:00:00"],
                "corr_time_mode": True,
            },
            "out": {"dt_bins": [0]},
            "filter": {},
        }
        mocker.patch(
            "tcm.config_yaml.gen_metadata",
            return_value=iter([(cfg1, (False, "i_01", None))]),
        )

        save_config_to_yaml(cfg, [src_file])

        run_dir = raw_dir / "cfg_proc" / "run"
        yaml_files = list(run_dir.glob("*.yaml"))
        assert len(yaml_files) == 1
        assert yaml_files[0].name == "250705_1419@i_01.yaml"

    def test_undated_name_when_no_time_ranges(self, tmp_path, mocker):
        """cfg1 without time_ranges → filename ``@{pcid}.yaml`` (no date prefix)."""
        raw_dir = tmp_path / RAW_DIR_NAME
        raw_dir.mkdir()
        src_file = raw_dir / "@i_01.txt"
        src_file.write_text("dummy\n")

        cfg = {
            "input": {
                "path": src_file,
                "tables": ["incl*"],
                "coefs": {"path": None},
                "coefs": {},
                "corr_time_mode": True,
            },
            "filter": {},
            "out": {"dt_bins": [0], "table": ""},
        }
        cfg1 = {
            "input": {"path": str(src_file), "coefs": {}, "corr_time_mode": True},
            "out": {"dt_bins": [0]},
            "filter": {},
        }
        mocker.patch(
            "tcm.config_yaml.gen_metadata",
            return_value=iter([(cfg1, (False, "i_01", None))]),
        )

        save_config_to_yaml(cfg, [src_file])

        run_dir = raw_dir / "cfg_proc" / "run"
        yaml_files = list(run_dir.glob("*.yaml"))
        assert len(yaml_files) == 1
        assert yaml_files[0].name == "@i_01.yaml"

    def test_comment_from_raw_device_name(self, tmp_path, mocker):
        """Raw (non-canonical) device file name suffix → ``-{comment}`` in config name.

        ``INKL_P05_маг.TXT`` has no ``-`` separator — the comment comes from the
        parse_name suffix after the pcid (separators stripped), so files of one
        probe get distinct names instead of colliding on ``@{pcid}.yaml``.
        """
        raw_dir = tmp_path / RAW_DIR_NAME
        raw_dir.mkdir()
        src_file = raw_dir / "INKL_P05_маг.TXT"
        src_file.write_text("dummy\n")

        cfg = {
            "input": {"path": src_file, "tables": ["incl*"], "coefs": {}, "corr_time_mode": True},
            "filter": {},
            "out": {"dt_bins": [0], "table": ""},
        }
        cfg1 = {
            "input": {"path": str(src_file), "coefs": {}, "corr_time_mode": True},
            "out": {"dt_bins": [0]},
            "filter": {},
        }
        mocker.patch(
            "tcm.config_yaml.gen_metadata",
            return_value=iter([(cfg1, (False, "i_p05", None))]),
        )

        save_config_to_yaml(cfg, [src_file])

        yaml_files = list((raw_dir / "cfg_proc" / "run").glob("*.yaml"))
        assert [f.name for f in yaml_files] == ["@i_p05-маг.yaml"]

    def test_stale_configs_kept(self, tmp_path, mocker):
        """Stale configs of any identity are never deleted; regeneration
        writes a fresh file alongside (same-name overwrite only)."""
        raw_dir = tmp_path / RAW_DIR_NAME
        raw_dir.mkdir()
        src_file = raw_dir / "INKL_P05_маг.TXT"
        src_file.write_text("dummy\n")
        run_dir = raw_dir / "cfg_proc" / "run"
        run_dir.mkdir(parents=True)
        # Stale: same identity (i_p05, маг), input.path gone (old machine copy)
        (run_dir / "230811_1622@i_p5-маг.yaml").write_text("input:\n  path: D:/nonexistent/@i_p5-маг.TXT\n")
        # Stale: different identity (i_p05, 0_v_trube) — no data file, must stay
        (run_dir / "260625_1708@i_p5-0_v_trube.yaml").write_text(
            "input:\n  path: D:/nonexistent/@i_p5-0_v_trube.TXT\n"
        )

        cfg = {
            "input": {"path": raw_dir, "tables": ["incl*"], "coefs": {}, "corr_time_mode": True},
            "filter": {},
            "out": {"dt_bins": [0], "table": ""},
        }
        cfg1 = {
            "input": {"path": str(src_file), "coefs": {}, "corr_time_mode": True},
            "out": {"dt_bins": [0]},
            "filter": {},
        }
        mocker.patch(
            "tcm.config_yaml.gen_metadata",
            return_value=iter([(cfg1, (False, "i_p05", None))]),
        )

        save_config_to_yaml(cfg, [raw_dir])

        stems = {f.stem for f in run_dir.glob("*.yaml")}
        assert "@i_p05-маг" in stems, f"{stems=!r}"  # regenerated
        assert "230811_1622@i_p5-маг" in stems, f"{stems=!r}"  # stale same identity kept
        assert "260625_1708@i_p5-0_v_trube" in stems, f"{stems=!r}"  # stale other identity kept

    def test_identity_mismatch_does_not_block_regen(self, tmp_path, mocker):
        """A valid config of another identity pointing at the same file
        (comment-less stem) does not suppress the correctly-named config."""
        raw_dir = tmp_path / RAW_DIR_NAME
        raw_dir.mkdir()
        src_file = raw_dir / "INKL_P05_маг.TXT"
        src_file.write_text("dummy\n")
        run_dir = raw_dir / "cfg_proc" / "run"
        run_dir.mkdir(parents=True)
        # Valid path, mismatched (comment-less) stem — leftover after rename
        (run_dir / "260624_1255@i_p05.yaml").write_text(f"input:\n  path: {src_file.as_posix()}\n")

        cfg = {
            "input": {"path": raw_dir, "tables": ["incl*"], "coefs": {}, "corr_time_mode": True},
            "filter": {},
            "out": {"dt_bins": [0], "table": ""},
        }
        cfg1 = {
            "input": {"path": str(src_file), "coefs": {}, "corr_time_mode": True},
            "out": {"dt_bins": [0]},
            "filter": {},
        }
        mocker.patch(
            "tcm.config_yaml.gen_metadata",
            return_value=iter([(cfg1, (False, "i_p05", None))]),
        )

        save_config_to_yaml(cfg, [raw_dir])

        stems = {f.stem for f in run_dir.glob("*.yaml")}
        assert "@i_p05-маг" in stems, f"{stems=!r}"  # correctly-named config generated
        assert "260624_1255@i_p05" in stems, f"{stems=!r}"  # old mismatched kept (never deleted)

    def test_whitespace_name_ignored_everywhere(self, tmp_path, mocker):
        """Whitespace in a YAML stem → ignored: it neither suppresses
        regeneration nor counts as a valid config."""
        raw_dir = tmp_path / RAW_DIR_NAME
        raw_dir.mkdir()
        src_file = raw_dir / "INKL_P05_маг.TXT"
        src_file.write_text("dummy\n")
        run_dir = raw_dir / "cfg_proc" / "run"
        run_dir.mkdir(parents=True)
        # whitespace-named with a VALID path — must still be ignored for dedup
        (run_dir / "@i_p05-маг - Copy.yaml").write_text(f"input:\n  path: {src_file.as_posix()}\n")
        # whitespace-named with a MISSING path — ignored, not tracked as stale either
        (run_dir / "@i_p05-маг — копия.yaml").write_text("input:\n  path: D:/nonexistent/x.TXT\n")

        cfg = {
            "input": {"path": raw_dir, "tables": ["incl*"], "coefs": {}, "corr_time_mode": True},
            "filter": {},
            "out": {"dt_bins": [0], "table": ""},
        }
        cfg1 = {
            "input": {"path": str(src_file), "coefs": {}, "corr_time_mode": True},
            "out": {"dt_bins": [0]},
            "filter": {},
        }
        mocker.patch(
            "tcm.config_yaml.gen_metadata",
            return_value=iter([(cfg1, (False, "i_p05", None))]),
        )

        save_config_to_yaml(cfg, [raw_dir])

        stems = {f.stem for f in run_dir.glob("*.yaml")}
        assert "@i_p05-маг" in stems, f"{stems=!r}"  # generated despite valid-path copy
        assert "@i_p05-маг - Copy" in stems, f"{stems=!r}"  # copies never deleted
        assert "@i_p05-маг — копия" in stems, f"{stems=!r}"


@pytest.mark.xr
class TestGetExistedCfgs:
    """get_existed_cfgs reads from cfg_proc/run/."""

    @pytest.mark.parametrize(
        ("files", "expected_count"),
        [
            pytest.param(["@i_01.yaml", "@i_02.yaml"], 2, id="two-configs"),
            pytest.param([], 0, id="empty-dir"),
        ],
    )
    def test_get_existed_cfgs(self, tmp_path, files, expected_count):
        run_dir = tmp_path / "cfg_proc" / "run"
        run_dir.mkdir(parents=True)
        for fname in files:
            (run_dir / fname).write_text("input:\n  path: test.txt\n")

        cfgs = get_existed_cfgs(run_dir)
        assert len(cfgs) >= expected_count

    def test_whitespace_stems_excluded(self, tmp_path):
        """Whitespace-named stems (GUI backups) are invisible to discovery —
        a lone backup does not count as an existing config."""
        run_dir = tmp_path / "cfg_proc" / "run"
        run_dir.mkdir(parents=True)
        (run_dir / "@i_01.yaml").write_text("input:\n  path: test.txt\n")
        (run_dir / "@i_01 - backup260908_150251.yaml").write_text("input:\n  path: test.txt\n")

        cfgs = get_existed_cfgs(run_dir)
        stems = [s for ss in cfgs.values() for s in ss]
        assert "@i_01" in stems and not any(" " in s for s in stems), f"{cfgs=!r}"

    def test_find_stale_cfgs(self, tmp_path):
        run_dir = tmp_path / "cfg_proc" / "run"
        run_dir.mkdir(parents=True)
        (run_dir / "@i_01.yaml").write_text("input:\n  path: /nonexistent/file.txt\n")

        cfgs = get_existed_cfgs(run_dir)
        stale = find_stale_cfgs(cfgs, run_dir)
        assert "i01" in stale

    @pytest.mark.parametrize(
        ("fname", "expected_pcid"),
        [
            pytest.param("@i_01.yaml", "i01", id="plain-@-prefix"),
            pytest.param("250705_1419@i_01.yaml", "i01", id="dated-prefix"),
            pytest.param("250705_1419@i_p02.yaml", "i_p02", id="dated-prefix-p-model"),
            pytest.param("anything@i_05.yaml", "i05", id="arbitrary-prefix"),
        ],
    )
    def test_pcid_extraction_ignores_prefix_before_at(self, tmp_path, fname, expected_pcid):
        """``@`` delimiter: anything before it is not significant for pcid."""
        run_dir = tmp_path / "cfg_proc" / "run"
        run_dir.mkdir(parents=True)
        (run_dir / fname).write_text("input:\n  path: test.txt\n")

        cfgs = get_existed_cfgs(run_dir)
        assert expected_pcid in cfgs, f"Expected pcid {expected_pcid} from {fname}, got {list(cfgs)}"
