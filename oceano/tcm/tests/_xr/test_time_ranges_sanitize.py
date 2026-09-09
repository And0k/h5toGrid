"""Tests for inverted time_ranges fallback — sanitize, mask primitive, fail-fast guard.

Covers: utils_time_corr.sanitize_time_ranges (drop start > end pairs, keep open
bounds, multi-pair partial, odd-length padding), make_range_mask inverted-pair
primitive (all-False — the reason the fallback exists), and run_processing
raising FileNotFoundError on None / 0-row data (never reported as ok).
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tcm import utils_time_corr
from tcm.utils_time_corr import make_range_mask, sanitize_time_ranges


@pytest.mark.xr
class TestSanitizeTimeRanges:
    @pytest.mark.parametrize(
        ("ranges", "exp_valid", "exp_dropped", "case"),
        [
            pytest.param(
                ["2000-01-01T00:07:06", "2000-01-01T00:14:12"],
                ["2000-01-01T00:07:06", "2000-01-01T00:14:12"],
                [],
                "ordered pair passes through",
                id="valid-kept",
            ),
            pytest.param(
                ["2000-01-01T00:14:12", "2000-01-01T00:07:06"],
                [],
                [("2000-01-01T00:14:12", "2000-01-01T00:07:06")],
                "report case: start > end must not filter",
                id="inverted-dropped",
            ),
            pytest.param(
                [None, "2000-01-01T00:14:12"],
                [None, "2000-01-01T00:14:12"],
                [],
                "open start is never inverted",
                id="open-start-kept",
            ),
            pytest.param(
                ["2000-01-01T00:07:06", None],
                ["2000-01-01T00:07:06", None],
                [],
                "open end is never inverted",
                id="open-end-kept",
            ),
            pytest.param(
                ["2000-01-01T00:14:12", "2000-01-01T00:07:06", "2000-01-02T00:00:00", "2000-01-02T01:00:00"],
                ["2000-01-02T00:00:00", "2000-01-02T01:00:00"],
                [("2000-01-01T00:14:12", "2000-01-01T00:07:06")],
                "valid pairs survive alongside inverted",
                id="multi-pair-partial",
            ),
            pytest.param(
                ["2000-01-01T00:07:06"],
                ["2000-01-01T00:07:06", None],
                [],
                "mirrors make_range_mask padding",
                id="odd-length-padded",
            ),
            pytest.param([], [], [], "no filter configured", id="empty"),
            pytest.param(None, [], [], "no filter configured", id="none"),
        ],
    )
    def test_sanitize(self, ranges, exp_valid, exp_dropped, case):
        """sanitize_time_ranges splits valid vs inverted pairs without mutating input."""
        valid, dropped = sanitize_time_ranges(ranges)
        assert valid == exp_valid, f"{case}: valid mismatch — {valid=!r}, {exp_valid=!r}"
        assert dropped == exp_dropped, f"{case}: dropped mismatch — {dropped=!r}, {exp_dropped=!r}"

    def test_equal_bounds_kept(self):
        """Zero-width pair (start == end) is not inverted — downstream decides."""
        valid, dropped = sanitize_time_ranges(["2000-01-01T00:07:06", "2000-01-01T00:07:06"])
        assert dropped == [], f"equal bounds must not drop — {dropped=!r}"
        assert valid == ["2000-01-01T00:07:06", "2000-01-01T00:07:06"], f"kept as-is — {valid=!r}"


@pytest.mark.xr
class TestMakeRangeMaskInverted:
    def test_inverted_pair_matches_nothing(self):
        """Inverted pair yields all-False mask — documents why the fallback must drop it."""
        t_ns = np.array(["2000-01-01T00:07:00", "2000-01-01T00:10:00"], dtype="M8[ns]").view(np.int64)
        mask = make_range_mask(t_ns, ["2000-01-01T00:14:12", "2000-01-01T00:07:06"])
        assert not mask.any(), f"inverted pair must match nothing — {mask=!r}"


@pytest.mark.xr
class TestExtractedTimeRanges:
    """config_yaml.extracted_time_ranges: pure normalizer — the extractor self-repairs inverted edges."""

    def test_valid_pair_normalized(self):
        """Ordered extraction passes through with space→T normalization."""
        from tcm.config_yaml import extracted_time_ranges

        result = extracted_time_ranges("2000-01-01 00:07:06", "2000-01-01 00:14:12", "i_p01", "@i_p1.TXT")
        assert result == ["2000-01-01T00:07:06", "2000-01-01T00:14:12"], f"kept as-is — {result=!r}"

    def test_inverted_pair_kept_for_main_init(self):
        """Residual inverted pair (repair impossible) is kept verbatim for main_init to strip at Run."""
        from tcm.config_yaml import extracted_time_ranges

        result = extracted_time_ranges("2000-01-01 00:14:12", "2000-01-01 00:07:06", "i_p01", "@i_p1.TXT")
        assert result == ["2000-01-01T00:14:12", "2000-01-01T00:07:06"], f"kept verbatim — {result=!r}"

    def test_missing_end_returns_none(self):
        """Half-open extraction leaves time_ranges absent (full-file load downstream)."""
        from tcm.config_yaml import extracted_time_ranges

        assert extracted_time_ranges("2000-01-01 00:07:06", None, "i_p01", "@i_p1.TXT") is None


@pytest.mark.xr
class TestSyncInvertedMetadata:
    """sync_yamls_devmeta_and_hydra keeps inverted info_devices ranges (main_init strips at Run)."""

    @staticmethod
    def _write_run_yaml(dir_cfgs, stem):
        from tcm.config_yaml import _ry

        yp = dir_cfgs / f"{stem}.yaml"
        with yp.open("w", encoding="utf-8") as f:
            _ry().dump({"input": {}}, f)
        return yp

    def test_inverted_metadata_kept(self, tmp_path, mocker):
        """Inverted metadata range is written verbatim (parsed-rows span needs a file read, not a swap)."""
        from tcm import config_yaml, metadata as _metadata

        dir_cfgs = tmp_path / "run"
        dir_cfgs.mkdir()
        stem = "000101_0014@i_p01"
        yp = self._write_run_yaml(dir_cfgs, stem)
        mocker.patch.object(_metadata, "get_path_in_parents", return_value=tmp_path / "info_devices.yaml")
        mocker.patch.object(_metadata, "load_file_meta", return_value={})
        mocker.patch.object(
            _metadata,
            "extract_devices_info",
            return_value={"ip01": {"r": ["2000-01-01 00:14:12", "2000-01-01 00:07:06"]}},
        )
        result = config_yaml.sync_yamls_devmeta_and_hydra(tmp_path, dir_cfgs, {"i_p01": [stem]})
        assert result[stem]["status"] == "written", f"verbatim write recorded — {result=!r}"
        with yp.open(encoding="utf-8") as f:
            tr = (config_yaml._ry(write=False).load(f) or {}).get("input", {}).get("time_ranges")
        assert tr == ["2000-01-01T00:14:12", "2000-01-01T00:07:06"], f"YAML keeps pair — {tr=!r}"


@pytest.mark.xr
class TestRunProcessingEmptyGuard:
    def test_none_data_raises(self, pipeline_env, mock_pipeline, mocker):
        """load_raw → (None, None) fails the probe instead of reporting ok."""
        from tcm._xr import io as _xr_io
        from tcm.processing import run_processing

        mock_pipeline(pipeline_env.cfg, pipeline_env, mocker)
        mocker.patch.object(_xr_io, "load_raw", return_value=(None, None))
        with pytest.raises(FileNotFoundError, match="No data loaded"):
            run_processing(pipeline_env.cfg)

    def test_zero_row_data_raises(self, pipeline_env, mock_pipeline, mocker):
        """load_raw → 0-row Dataset fails the probe (NC/minmax-emptied case)."""
        from tcm._xr import io as _xr_io
        from tcm.processing import run_processing

        mock_pipeline(pipeline_env.cfg, pipeline_env, mocker)
        mocker.patch.object(_xr_io, "load_raw", return_value=(xr.Dataset(), None))
        with pytest.raises(FileNotFoundError, match="No data loaded"):
            run_processing(pipeline_env.cfg)
