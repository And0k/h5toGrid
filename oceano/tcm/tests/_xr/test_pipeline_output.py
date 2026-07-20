"""Tests for pipeline output wiring — PathLayout, shared NC files, TSV export.

TDD: tests define expected behavior; code must satisfy them.
See ``.kilo/plans/1782426545504-pipeline-output-wiring-plan.md``.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import pytest
from omegaconf import DictConfig, OmegaConf

from tcm._constants import RAW_DIR_NAME
from tcm.processing import run_processing
from _xr.conftest import _mock_pipeline

# ---------------------------------------------------------------------------
# T1: PathLayout resolves output paths
# ---------------------------------------------------------------------------


@pytest.mark.xr
class TestPathLayoutWiring:
    """PathLayout.from_cfg + apply_to_cfg populates cfg.out paths."""

    def test_resolves_raw_db_path(self, pipeline_env, mock_pipeline, mocker):
        """cfg.out.raw_db_path is set after run_processing."""
        mock_pipeline(pipeline_env.cfg, pipeline_env, mocker)
        run_processing(pipeline_env.cfg)
        assert pipeline_env.cfg.out.raw_db_path is not None
        assert Path(pipeline_env.cfg.out.raw_db_path).is_absolute()

    def test_resolves_not_joined_db_path(self, pipeline_env, mock_pipeline, mocker):
        """cfg.out.not_joined_db_path lives in proc_dir, not raw_dir."""
        mock_pipeline(pipeline_env.cfg, pipeline_env, mocker)
        run_processing(pipeline_env.cfg)
        njp = pipeline_env.cfg.out.get("not_joined_db_path")
        assert njp is not None, "not_joined_db_path not resolved"
        njp = Path(njp)
        assert RAW_DIR_NAME not in njp.parts, (
            f"not_joined_db_path should not be inside {RAW_DIR_NAME}: {njp}"
        )

    def test_resolves_text_path(self, pipeline_env, mock_pipeline, mocker):
        """cfg.out.text_path is set after run_processing."""
        mock_pipeline(pipeline_env.cfg, pipeline_env, mocker)
        run_processing(pipeline_env.cfg)
        assert pipeline_env.cfg.out.get("text_path") is not None


# ---------------------------------------------------------------------------
# T2: Shared NC files with per-probe groups
# ---------------------------------------------------------------------------


@pytest.mark.xr
class TestSharedNCOutput:
    """Data goes into shared files with HDF5 groups, not per-probe files."""

    @pytest.mark.parametrize(
        ("dt_bins", "expected_file", "expected_group"),
        [
            pytest.param([0], "noavg_path", "i01", id="noavg-only"),
            pytest.param([0, 600], "noavg_path", "i01", id="noavg-with-bins"),
            pytest.param([0, 600], "avg_path", "i01bin600s", id="binned-in-avg"),
        ],
    )
    def test_group_in_correct_file(
        self, pipeline_env, mock_pipeline, mocker, dt_bins, expected_file, expected_group,
    ):
        """Each bin writes to the correct shared file with the correct group."""
        pipeline_env.cfg.out.dt_bins = dt_bins
        mock_pipeline(pipeline_env.cfg, pipeline_env, mocker)
        run_processing(pipeline_env.cfg)

        target = getattr(pipeline_env, expected_file)
        assert target.exists(), f"{expected_file} not at {target}"
        with h5py.File(target, "r") as f:
            assert expected_group in f, (
                f"Expected group '{expected_group}', got: {list(f.keys())}"
            )

    def test_no_per_probe_nc_files(self, pipeline_env, mock_pipeline, mocker):
        """Must NOT create @i01.nc — data goes into shared files."""
        pipeline_env.cfg.out.dt_bins = [0]
        mock_pipeline(pipeline_env.cfg, pipeline_env, mocker)
        run_processing(pipeline_env.cfg)
        per_probe = list(pipeline_env.proc_dir.glob("@*.nc"))
        assert not per_probe, (
            f"Per-probe NC files must not be created — use shared proc_noAvg.nc: {per_probe}"
        )

    def test_raw_nc_has_coefs_group(self, pipeline_env, mock_pipeline, mocker):
        """raw.nc has /incl01/coef/G/A dataset."""
        pipeline_env.cfg.out.dt_bins = [0]
        mock_pipeline(pipeline_env.cfg, pipeline_env, mocker)
        run_processing(pipeline_env.cfg)

        with h5py.File(pipeline_env.raw_db_path, "r") as f:
            assert "incl01/coef" in f, (
                f"Coefs group missing from raw.nc, got: {list(f.keys())}"
            )


# ---------------------------------------------------------------------------
# T3: CSV/TSV export to text_path
# ---------------------------------------------------------------------------


@pytest.mark.xr
class TestCSVExport:
    """TSV files written to text_path for bins ≥ dt_bins_min_save_text."""

    @pytest.mark.parametrize(
        ("dt_bins", "expected_suffixes"),
        [
            pytest.param([0, 2], {"bin2s@i01.tsv"}, id="noavg-plus-2s"),
            pytest.param([0, 600], {"bin600s@i01.tsv"}, id="noavg-plus-600s"),
            pytest.param([0, 2, 600], {"bin2s@i01.tsv", "bin600s@i01.tsv"}, id="all-bins"),
        ],
    )
    def test_tsv_created_for_each_bin(
        self, pipeline_env, mock_pipeline, mocker, dt_bins, expected_suffixes,
    ):
        """Each binned result (dt_bin > 0) produces a TSV in text_path.

        The no-avg bin (dt_bin=0) is skipped because dt_bins_min_save_text
        defaults to 1 second.
        """
        pipeline_env.cfg.out.dt_bins = dt_bins
        mock_pipeline(pipeline_env.cfg, pipeline_env, mocker)
        run_processing(pipeline_env.cfg)

        text_dir = Path(pipeline_env.cfg.out.get("text_path"))
        assert text_dir.exists(), f"text_path not created: {text_dir}"
        actual_names = {f.name for f in text_dir.glob("*.tsv")}
        for suffix in expected_suffixes:
            assert any(n.endswith(suffix) for n in actual_names), (
                f"TSV ending with '{suffix}' not found; got: {actual_names}"
            )

    def test_small_bins_skipped(self, pipeline_env, mock_pipeline, mocker):
        """dt_bins_min_save_text=10 → bin2s skipped, bin600s present."""
        pipeline_env.cfg.out.dt_bins = [0, 2, 600]
        pipeline_env.cfg.out.dt_bins_min_save_text = 10
        mock_pipeline(pipeline_env.cfg, pipeline_env, mocker)
        run_processing(pipeline_env.cfg)

        text_dir = Path(pipeline_env.cfg.out.get("text_path"))
        names = {f.name for f in text_dir.glob("*.tsv")} if text_dir.exists() else set()
        assert not any("bin2s@" in n for n in names), (
            f"bin2s TSV should be skipped (dt_bins_min_save_text=10): {names}"
        )
        assert any("bin600s@" in n for n in names), (
            f"bin600s TSV should be present: {names}"
        )


# ---------------------------------------------------------------------------
# T5: Multi-probe shared file
# ---------------------------------------------------------------------------


@pytest.mark.xr
class TestMultiProbeSharedFile:
    """Multiple probes write into the same shared NC file."""

    def test_two_probes_share_noavg_file(self, pipeline_env, mocker):
        """i01 and i02 → both groups in same proc_noAvg.nc."""

        # Process probe 1
        _mock_pipeline(pipeline_env.cfg, pipeline_env, mocker)
        run_processing(pipeline_env.cfg)

        # Process probe 2 (same data, different pcid)
        cfg2 = DictConfig(OmegaConf.to_container(pipeline_env.cfg, resolve=True))
        cfg2.input.path = str(pipeline_env.raw_dir / "@i_02.txt")
        (pipeline_env.raw_dir / "@i_02.txt").write_text(
            pipeline_env.csv_file.read_text(encoding="utf-8"), encoding="utf-8",
        )
        # Re-mock with updated cfg2
        env2 = type(pipeline_env)(**{
            **pipeline_env.__dict__,
            "cfg": cfg2,
        })
        _mock_pipeline(cfg2, env2, mocker)
        run_processing(cfg2)

        with h5py.File(pipeline_env.noavg_path, "r") as f:
            groups = list(f.keys())
            assert "i01" in groups, f"i01 group missing: {groups}"
            assert "i02" in groups, f"i02 group missing: {groups}"
