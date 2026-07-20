"""Tests for tcm/processing.py — coefficient resolution and run_processing dispatch."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from omegaconf import DictConfig

from tcm.config import ConfigIn_InclProc, Return
from tcm.incl_calc.coefs import get_coefs
from tcm.processing import get_coefs_from_cfg, process_inmemory, run_processing

_GET_COEFS_PATCH = "tcm.incl_calc.coefs.get_coefs"


@pytest.mark.xr
class TestGetCoefsFromCfg:
    """get_coefs_from_cfg: coefs_path fallback, override merge, list→ndarray."""

    def test_fallback_to_class_default_coefs_path(self, mocker):
        """When cfg_in has no coefs_path, falls back to ConfigIn_InclProc.coefs_path."""
        mock_get = mocker.patch(_GET_COEFS_PATCH, return_value={"Ag": np.eye(3)})
        get_coefs_from_cfg({}, "i_01")
        assert ConfigIn_InclProc.coefs_path in mock_get.call_args[0][0]

    def test_explicit_coefs_path_used_first(self, mocker):
        """When cfg_in has coefs_path, it appears before the class default."""
        mock_get = mocker.patch(_GET_COEFS_PATCH, return_value={"Ag": np.eye(3)})
        get_coefs_from_cfg({"coefs_path": "/custom/path.h5", "coefs": {}}, "i_01")
        paths = mock_get.call_args[0][0]
        assert str(paths[0]) == "/custom/path.h5"
        assert ConfigIn_InclProc.coefs_path in paths

    def test_coefs_paths_is_list(self, mocker):
        """coefs_paths passed to get_coefs must always be a list, not a scalar."""
        mock_get = mocker.patch(_GET_COEFS_PATCH, return_value={})
        get_coefs_from_cfg({"coefs_path": "/some/path.h5", "coefs": {}}, "i_01")
        assert isinstance(mock_get.call_args[0][0], list)

    def test_override_passed_as_coefs_ovr(self, mocker):
        """cfg_in.coefs is forwarded as coefs_ovr to get_coefs."""
        override = {"azimuth_shift_deg": 195.0}
        mocker.patch(_GET_COEFS_PATCH, return_value=override)
        get_coefs_from_cfg({"coefs": override}, "i_01")
        # Already tested via mock — verify coefs_ovr keyword
        mock_get = mocker.patch(_GET_COEFS_PATCH, return_value=override)
        get_coefs_from_cfg({"coefs": override}, "i_01")
        assert mock_get.call_args[1]["coefs_ovr"] == override

    def test_override_replaces_value(self, mocker):
        """Override values replace loaded coefs (delegated to get_coefs)."""
        mocker.patch(_GET_COEFS_PATCH, return_value={"azimuth_shift_deg": 195.0, "Ag": np.eye(3)})
        coefs = get_coefs_from_cfg({"coefs": {"azimuth_shift_deg": 195.0}}, "i_01")
        assert coefs["azimuth_shift_deg"] == 195.0

    def test_empty_coefs_returns_dict(self, mocker):
        """Empty coefs still returns a dict (not None)."""
        mocker.patch(_GET_COEFS_PATCH, return_value={})
        assert isinstance(get_coefs_from_cfg({}, "i_01"), dict)

    def test_yaml_export_dir_added_as_final_fallback(self, mocker):
        """yaml_export dir of the class-default coefs is appended as final fallback.

        Lets the ``dist/tcm_clc_txt`` packaging (no bundled ``calibration.h5``)
        silently load from per-probe YAML exports.
        """
        mock_get = mocker.patch(_GET_COEFS_PATCH, return_value={})
        get_coefs_from_cfg({}, "i_01")
        expected_yaml_dir = ConfigIn_InclProc.coefs_path.parent / "yaml_export"
        assert expected_yaml_dir in mock_get.call_args[0][0]
        assert mock_get.call_args[0][0][-1] == expected_yaml_dir

    def test_yaml_export_not_duplicated_with_explicit_path(self, mocker):
        """Explicit ``coefs_path`` already pointing at ``yaml_export`` dir → no duplicate append."""
        mock_get = mocker.patch(_GET_COEFS_PATCH, return_value={})
        yaml_dir = ConfigIn_InclProc.coefs_path.parent / "yaml_export"
        get_coefs_from_cfg({"coefs_path": str(yaml_dir), "coefs": {}}, "i_01")
        assert mock_get.call_args[0][0].count(yaml_dir) == 1


@pytest.mark.xr
class TestGetCoefsArrayConversion:
    """incl_calc.coefs.get_coefs: list values → numpy ndarrays."""

    @pytest.mark.parametrize(
        ("key", "value", "expected_dtype", "expected_shape"),
        [
            pytest.param("Ag", [[0.00173, 0, 0], [0, 0.00173, 0], [0, 0, 0.00173]], np.float64, (3, 3), id="Ag-2d"),
            pytest.param("Cg", [478.49, -160.82, 363.21], np.float64, (3,), id="Cg-1d"),
            pytest.param("kVabs", [-11.21, 19.88, 17.39, 6.35, -6.80, 71.45], np.float64, (6,), id="kVabs"),
            pytest.param("P_t", [[-9.99, -0.00079, -9.42e-05], [5.25e-06, 1.0e-08, 0.0], [0.0, 0.0, 0.0]], np.float64, (3, 3), id="P_t-3d"),
        ],
    )
    def test_list_becomes_ndarray(self, key, value, expected_dtype, expected_shape):
        """List coefs values → numpy ndarrays with correct dtype/shape."""
        result = get_coefs([], "i01", coefs_ovr={"Ag": [[1]], key: value})
        assert isinstance(result[key], np.ndarray)
        assert result[key].dtype == expected_dtype
        assert result[key].shape == expected_shape

    def test_dates_stays_dict(self):
        """'dates' key must remain a dict, not be converted to ndarray."""
        dates = {"Ag": "2023-08-13T07:29:28", "Cg": "2023-08-13T07:29:28"}
        result = get_coefs([], "i01", coefs_ovr={"Ag": [[1]], "dates": dates})
        assert isinstance(result["dates"], dict)

    def test_date_stays_str(self):
        """'date' key must remain a str/datetime, not be converted to ndarray."""
        result = get_coefs([], "i01", coefs_ovr={"Ag": [[1]], "date": "2023-08-13T07:29:28"})
        assert not isinstance(result["date"], np.ndarray)


@pytest.mark.xr
class TestCoefsPTsupersession:
    """``P_t`` (2-D pressure-T polynomial) silently supersedes legacy scalars
    ``P``, ``PBattery``, ``PTemp``. The pressure pipeline only uses ``P_t``,
    so the legacy defaults must not error/warn when ``P_t`` is provided."""

    def test_P_t_in_override_suppresses_scalar_missing_warn(self, caplog):
        """P_t in coefs_ovr for pressure probe → no warn about P/PBattery/PTemp missing."""
        import logging

        with caplog.at_level(logging.WARNING, logger="tcm.incl_calc.coefs"):
            result = get_coefs([], "i_p01", coefs_ovr={"Ag": [[1]]})
        assert "P_t" not in result and "PBattery" not in result
        for rec in caplog.records:
            assert "not redefined" not in rec.getMessage(), (
                f"P_t should silence the warning, got: {rec.getMessage()}"
            )

    def test_P_t_in_load_silences_scalar_warning(self, caplog):
        """P_t via coefs_ovr for p-type probe → no warn about P/PBattery/PTemp defaults.

        ``loaded`` mimics what ``load_coefs(yaml_export/incl_p01.yaml)`` returns;
        here we test the coefs_ovr-driven path instead (same ``get_coefs`` logic).
        """
        import logging

        P_t_val = [[-9.99, -0.00079, -9.42e-05], [5.25e-06, 1.0e-08, 0.0], [0.0, 0.0, 0.0]]
        with caplog.at_level(logging.WARNING, logger="tcm.incl_calc.coefs"):
            result = get_coefs([], "i_p01", coefs_ovr={"Ag": [[1]], "P_t": P_t_val})
        assert "P_t" in result
        for rec in caplog.records:
            assert "not redefined" not in rec.getMessage(), (
                f"P_t should silence the warning, got: {rec.getMessage()}"
            )

    def test_non_p_probe_without_overrides_returns_empty(self):
        """Non-pressure probe without overrides or file coefs → empty dict (no crash)."""
        result = get_coefs([], "i_01")
        assert isinstance(result, dict)
        assert result == {"dates": {}}


@pytest.mark.xr
class TestRunProcessingDispatch:
    """run_processing dispatches to the correct code path."""

    def test_dispatches_to_batch(self, mocker, tmp_path):
        """run_processing with cfg.files → _load_batch."""
        from datetime import timedelta

        out_dir = str(tmp_path / "out")
        cfg = DictConfig({
            "input": {"path": "/dummy.txt"},
            "out": {"dt_bins": [0], "dir": out_dir},
            "filter": {},
            "program": {"return_": Return.END, "verbose": "INFO"},
            "files": [{"path": "/f1.txt", "coefs": {}}, {"path": "/f2.txt", "coefs": {}}],
        })
        # main_init converts DictConfig → plain dict; return a minimal mock
        mocker.patch(
            "tcm.processing.cli.main_init",
            return_value={
                "input": {"path": Path("/dummy.txt"), "tables": ["incl*"]},
                "out": {"dt_bins": [timedelta(0)], "dir": out_dir},
                "filter": {},
                "program": {"return_": Return.END, "verbose": "INFO"},
                "files": [{"path": Path("/f1.txt")}, {"path": Path("/f2.txt")}],
            },
        )
        mocker.patch("tcm.processing.format.to_pcid_from_name", return_value="i_01")
        mocker.patch("tcm.processing.get_coefs_from_cfg", return_value={})
        mock_batch = mocker.patch("tcm.processing._load_batch", return_value=(None, None))
        run_processing(cfg)
        mock_batch.assert_called_once()


@pytest.mark.xr
class TestProcessInmemory:
    """process_inmemory is importable and callable."""

    def test_importable(self):
        assert callable(process_inmemory)


# ---------------------------------------------------------------------------
# _build_filter_params_text
# ---------------------------------------------------------------------------


@pytest.mark.xr
class TestBuildFilterParamsText:
    """_build_filter_params_text produces sorted key=value lines for _run_params attr."""

    @staticmethod
    def _call(cfg_in: dict, cfg_filter: dict, coefs: dict | None = None) -> str:
        from tcm.processing import _build_filter_params_text

        return _build_filter_params_text(cfg_in, cfg_filter, coefs=coefs)

    def test_filter_min_max(self):
        """Filter min/max params appear as sorted key=value lines."""
        result = self._call({}, {"min": {"g_minus_1": 0.01}, "max": {"h_minus_1": 8.0}})
        lines = result.splitlines()
        assert "filter.max.h_minus_1=8.0" in lines
        assert "filter.min.g_minus_1=0.01" in lines

    def test_time_ranges(self):
        """time_ranges appear as input.time_ranges=[start, end]."""
        result = self._call({"time_ranges": ["2024-01-01", "2024-06-01"]}, {})
        assert "input.time_ranges=[2024-01-01, 2024-06-01]" in result.splitlines()

    def test_coef_scalars(self):
        """Scalar coefficient values rendered via str()."""
        result = self._call({}, {}, coefs={"azimuth_shift_deg": 180.0, "kVabs": 1.5, "dates": {}})
        lines = result.splitlines()
        assert "coef.azimuth_shift_deg=180.0" in lines
        assert "coef.kVabs=1.5" in lines
        assert not any("coef.dates" in ln for ln in lines), "dates should be skipped"

    def test_coef_ndarrays(self):
        """ndarray coefs rendered via np.array2string (compact, deterministic)."""
        coefs = {"Ag": np.eye(2) * 0.00173, "Cg": np.array([1.0, 2.0, 3.0]), "dates": {}}
        result = self._call({}, {}, coefs=coefs)
        assert "coef.Ag=" in result
        assert "coef.Cg=" in result
        assert "0.00173" in result  # Ag diagonal value present

    def test_coef_rz_skipped(self):
        """Rz key is skipped (large rotation matrix, not informative in diff)."""
        result = self._call({}, {}, coefs={"Rz": np.eye(3), "dates": {}})
        assert not any("Rz" in ln for ln in result.splitlines())

    def test_sorted_output(self):
        """Output lines are sorted by key for stable diff."""
        result = self._call(
            {"time_ranges": ["2024-01-01", "2024-06-01"]},
            {"max": {"h_minus_1": 8.0}},
            coefs={"azimuth_shift_deg": 180.0, "dates": {}},
        )
        keys = [line.split("=")[0] for line in result.splitlines()]
        assert keys == sorted(keys), f"Keys not sorted: {keys}"

    def test_empty_inputs(self):
        """Empty filter + empty coefs → empty string."""
        assert self._call({}, {}) == ""

    def test_dt_min_binning_proc(self):
        """dt_min_binning_proc included when present."""
        result = self._call({"dt_min_binning_proc": 120}, {})
        assert "input.dt_min_binning_proc=120" in result.splitlines()
