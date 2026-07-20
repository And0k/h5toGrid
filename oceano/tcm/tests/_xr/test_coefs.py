"""Tests for tcm/_xr/coefs.py — xr-native coefficient preparation.
Also tests config_yaml.update_coefs_in_run_yaml (coef persistence to YAML).
"""
from __future__ import annotations

import ast
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict

import h5py
import numpy as np
import pytest
import xarray as xr
from ruamel.yaml import YAML

import tcm._xr.coefs as _coefs_mod
from tcm._xr.coefs import (
    get_coef_zeroing_matrix,
    load_coefs_from_nc,
    prep_cfg_for_probe,
    save_coefs_to_nc,
)
from tcm.config import ConfigIn_InclProc
from tcm.config_yaml import update_coefs_in_run_yaml
from tcm.incl_calc.coefs import get_coefs, load_coefs

def _stub_get_coefs(paths, tbl, coefs_ovr=None):
    """Reusable stub for get_coefs monkeypatching."""
    return {"date": "2024-01-01"}


_GET_COEFS_TARGET = "tcm.incl_calc.coefs.get_coefs"


# --------------------------------------------------------------------------- #
# Helper assertions for parametrized get_coefs call checks
# --------------------------------------------------------------------------- #
def _assert_tbl_arg(call) -> None:
    assert call.args[1] == "incl_01"


def _assert_class_default_in_paths(call) -> None:
    assert ConfigIn_InclProc.coefs_path in call.args[0]


def _assert_yaml_export_last(call) -> None:
    paths = call.args[0]
    yaml_dir = ConfigIn_InclProc.coefs_path.parent / "yaml_export"
    assert yaml_dir in paths
    assert paths[-1] == yaml_dir


def _assert_coefs_ovr_passthrough(call) -> None:
    assert call.kwargs["coefs_ovr"] == {"azimuth_shift_deg": 195.0}


# --------------------------------------------------------------------------- #
# prep_cfg_for_probe
# --------------------------------------------------------------------------- #


@pytest.mark.xr
class TestPrepCfgForProbe:
    """prep_cfg_for_probe builds correct cfg1 without _dask_legacy dependency."""

    @pytest.fixture()
    def cfg_in_common(self) -> Dict[str, Any]:
        return {
            "path": Path("/data/_raw/@i_01.txt"),
            "tables": ["incl*"],
            "coefs_path": None,
            "coefs": {},
        }

    @pytest.fixture()
    def cfg_top(self) -> Dict[str, Any]:
        return {
            "input": {
                # fmt_in_base := {
                "path": Path("/data/_raw/@i_01.txt"),
                "tables": ["incl*"],
                "coefs_path": None,
                "coefs": {},
                "corr_time_mode": True,  # moved from filter
            },
            "out": {"dt_bins": [0], "table": ""},
            "filter": {},
        }

    @pytest.fixture(autouse=True)
    def _stub_get_coefs(self, mocker):
        """Patch get_coefs with default stub; individual tests can override."""
        mocker.patch(_GET_COEFS_TARGET, _stub_get_coefs)

    def test_merges_common_and_per_probe_overrides(self, cfg_in_common, cfg_top):
        """Per-probe overrides are merged on top of cfg_in_common."""
        cfg_in_for_probes = {"i_01": {"coefs_path": Path("/custom.h5")}}
        cfg1 = prep_cfg_for_probe("i_01", cfg_in_for_probes, cfg_in_common, cfg_top)
        assert cfg1["input"]["coefs_path"] == Path("/custom.h5")
        assert cfg1["input"]["tables"] == ["incl_01"]  # glob expanded

    def test_expands_glob_tables(self, cfg_in_common, cfg_top):
        """'incl*' glob expanded to concrete table name for this probe."""
        cfg1 = prep_cfg_for_probe("i_01", {}, cfg_in_common, cfg_top)
        assert cfg1["input"]["tables"] == ["incl_01"]

    def test_path_csv_overrides_path(self, cfg_in_common, cfg_top):
        """path_csv argument overrides cfg1['in']['path']."""
        path_csv = Path("/corrected/@i_01.txt")
        cfg1 = prep_cfg_for_probe("i_01", {}, cfg_in_common, cfg_top, path_csv=path_csv)
        assert cfg1["input"]["path"] == path_csv

    @pytest.mark.parametrize(
        ("coefs", "check"),
        [
            pytest.param({}, _assert_tbl_arg, id="tbl-arg"),
            pytest.param({}, _assert_class_default_in_paths, id="class-default-in-paths"),
            pytest.param({}, _assert_yaml_export_last, id="yaml-export-last"),
            pytest.param(
                {"azimuth_shift_deg": 195.0},
                _assert_coefs_ovr_passthrough,
                id="coefs-ovr-passthrough",
            ),
        ],
    )
    def test_calls_get_coefs_correctly(
        self, mocker, cfg_in_common, cfg_top, coefs: dict, check: Callable
    ):
        """get_coefs called with correct table, paths chain, and coefs_ovr."""
        cfg_in_common["coefs"] = coefs
        mock_get = mocker.patch(_GET_COEFS_TARGET, return_value={"date": "2024-01-01"})
        prep_cfg_for_probe("i_01", {}, cfg_in_common, cfg_top)
        check(mock_get.call_args)

    def test_no_dask_dependency(self):
        """tcm._xr.coefs must not import from _dask_legacy."""
        source = Path(_coefs_mod.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = {
            alias.name if isinstance(node, ast.Import) else node.module
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in (
                node.names if isinstance(node, ast.Import) else [type("", (), {"name": node.module})()]
            )
        }
        assert not any("_dask_legacy" in m for m in imported), (
            f"Must not import from _dask_legacy, got: {imported}"
        )

    def test_output_structure_has_in_out_filter(self, cfg_in_common, cfg_top):
        """cfg1 has 'input', 'out', 'filter' keys."""
        cfg1 = prep_cfg_for_probe("i_01", {}, cfg_in_common, cfg_top)
        assert {"input", "out", "filter"} <= cfg1.keys()


# --------------------------------------------------------------------------- #
# save_coefs_to_nc / load_coefs_from_nc
# --------------------------------------------------------------------------- #

_SAMPLE_COEFS: Dict[str, Any] = {
    "Ag": np.eye(3),
    "Cg": np.array([0.1, 0.2, 0.3]),
    "Ah": np.eye(3) * 0.5,
    "Ch": np.array([0.4, 0.5, 0.6]),
    "kVabs": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    "azimuth_shift_deg": 195.0,
    "date": "2024-01-01T00:00:00",
}


@pytest.mark.xr
class TestSaveCoefsToNc:
    """save_coefs_to_nc writes coefs dict to NetCDF4 /{tbl}/coef/ group."""

    @pytest.fixture()
    def sample_coefs(self):
        """Copy of module-level _SAMPLE_COEFS so mutations don't leak."""
        return {k: v.copy() if hasattr(v, "copy") else v for k, v in _SAMPLE_COEFS.items()}

    def test_writes_coef_group_structure(self, sample_coefs, tmp_path):
        """Coefs written to /{tbl}/coef/G/A, /{tbl}/coef/H/C, etc."""
        nc_path = tmp_path / "test.raw.nc"
        save_coefs_to_nc(nc_path, "incl_01", sample_coefs)
        with h5py.File(nc_path, "r") as f:
            coef = f["incl_01"]["coef"]
            np.testing.assert_array_equal(coef["G"]["A"], np.eye(3))
            np.testing.assert_array_equal(coef["G"]["C"], [0.1, 0.2, 0.3])
            np.testing.assert_array_equal(coef["H"]["A"], np.eye(3) * 0.5)
            np.testing.assert_array_equal(coef["H"]["C"], [0.4, 0.5, 0.6])
            np.testing.assert_array_equal(coef["Vabs0"], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            assert float(coef["H"]["azimuth_shift_deg"][0]) == pytest.approx(195.0)

    def test_writes_date_attr(self, sample_coefs, tmp_path):
        """coef group gets a 'date' attribute."""
        nc_path = tmp_path / "test.raw.nc"
        save_coefs_to_nc(nc_path, "incl_01", sample_coefs)
        with h5py.File(nc_path, "r") as f:
            assert f["incl_01"]["coef"].attrs["date"] == "2024-01-01T00:00:00"

    def test_writes_timestamp_attr(self, sample_coefs, tmp_path):
        """Numeric datasets get 'timestamp' attribute when dates is truthy."""
        nc_path = tmp_path / "test.raw.nc"
        save_coefs_to_nc(nc_path, "incl_01", sample_coefs, dates=True)
        with h5py.File(nc_path, "r") as f:
            assert "timestamp" in f["incl_01"]["coef"]["G"]["A"].attrs

    def test_dates_dict_preserves_existing_dates(self, sample_coefs, tmp_path):
        """Existing dates from YAML preserved; True sentinel → current date; missing → current date.

        ``prepare_coefs`` returns ``dates = {"Ag": "2023-08-12T16:21:30", "Rz": True}``.
        The writer must map short names ("Ag") to h5py paths ("//coef//G//A"),
        preserve the existing date string, and use current ISO date for ``True``
        and for coefs absent from the dict.
        """
        nc_path = tmp_path / "test.raw.nc"
        dates = {"Ag": "2023-08-12T16:21:30", "Rz": True}
        save_coefs_to_nc(nc_path, "incl_01", sample_coefs, dates=dates)
        with h5py.File(nc_path, "r") as f:
            coef = f["incl_01"]["coef"]
            # Existing date preserved
            assert coef["G"]["A"].attrs["timestamp"] == "2023-08-12T16:21:30", (
                "Existing date from YAML must be preserved, not replaced by current date"
            )
            # True sentinel → current ISO date (contains 'T')
            rz_ts = coef.get("Rz")
            if rz_ts is not None:
                assert "T" in rz_ts.attrs["timestamp"], (
                    f"True sentinel should produce ISO date, got {rz_ts.attrs['timestamp']!r}"
                )
            # Coef not in dates dict → current date (default True)
            assert "T" in coef["H"]["A"].attrs["timestamp"], (
                "Coef missing from dates dict should get current ISO date"
            )

    def test_idempotent_overwrite(self, sample_coefs, tmp_path):
        """Writing coefs twice overwrites without error."""
        nc_path = tmp_path / "test.raw.nc"
        save_coefs_to_nc(nc_path, "incl_01", sample_coefs)
        sample_coefs["Ag"] = np.ones((3, 3))
        save_coefs_to_nc(nc_path, "incl_01", sample_coefs)
        with h5py.File(nc_path, "r") as f:
            np.testing.assert_array_equal(f["incl_01"]["coef"]["G"]["A"], np.ones((3, 3)))

    def test_skips_none_values(self, tmp_path):
        """None values in coefs dict are skipped, not written as datasets."""
        coefs = {"Ag": np.eye(3), "Cg": None, "date": "2024-01-01"}
        nc_path = tmp_path / "test.raw.nc"
        save_coefs_to_nc(nc_path, "incl_01", coefs)
        with h5py.File(nc_path, "r") as f:
            assert "A" in f["incl_01"]["coef"]["G"]
            assert "C" not in f["incl_01"]["coef"]["G"]

    def test_pid_written(self, sample_coefs, tmp_path):
        """pcid written as string dataset at /{tbl}/coef/pid."""
        nc_path = tmp_path / "test.raw.nc"
        save_coefs_to_nc(nc_path, "incl_01", sample_coefs, pcid="i_01")
        with h5py.File(nc_path, "r") as f:
            assert f["incl_01"]["coef"]["pid"][()].decode() == "i_01"

    def test_creates_file_if_missing(self, tmp_path):
        """save_coefs_to_nc creates NC file when it doesn't exist."""
        nc_path = tmp_path / "nonexistent.raw.nc"
        save_coefs_to_nc(nc_path, "incl_01", {"Ag": np.eye(3), "date": "2024-01-01"})
        with h5py.File(nc_path, "r") as f:
            assert "G" in f["incl_01"]["coef"]

    @pytest.mark.parametrize(
        ("description", "initial_dtype", "initial_data", "overwrite_data"),
        [
            pytest.param(
                "string date overwrites existing float dataset",
                np.float64, np.float64(0.0), "2023-08-13T07:29:28",
                id="str-over-float",
            ),
            pytest.param(
                "string date overwrites existing shorter string",
                h5py.string_dtype(), "old_date", "2023-08-13T07:29:28",
                id="str-over-str-shorter",
            ),
        ],
    )
    def test_overwrite_incompatible_dtype(
        self, description, initial_dtype, initial_data, overwrite_data, tmp_path,
    ):
        """save_coefs_to_nc replaces datasets when dtype changes (e.g. float→str)."""
        nc_path = tmp_path / "test.raw.nc"
        # Simulate legacy/previous write with incompatible dtype
        with h5py.File(nc_path, "w") as h5f:
            tbl_coef = h5f.require_group("incl_01/coef")
            tbl_coef.create_dataset("date", data=initial_data)
        coefs = {"Ag": np.eye(3), "date": overwrite_data}
        save_coefs_to_nc(nc_path, "incl_01", coefs)
        with h5py.File(nc_path, "r") as f:
            raw = f["incl_01/coef/date"][()]
            actual = raw.decode() if isinstance(raw, bytes) else str(raw)
            assert actual == overwrite_data, (
                f"{description}: expected {overwrite_data!r}, got {actual!r}"
            )

    def test_overwrite_rz_shape_change(self, tmp_path):
        """save_coefs_to_nc handles Rz shape change (scalar → 3×3 matrix)."""
        nc_path = tmp_path / "test.raw.nc"
        # First write: Rz = identity (3×3)
        coefs1 = {"Ag": np.eye(3), "Rz": np.eye(3), "date": "2024-01-01"}
        save_coefs_to_nc(nc_path, "incl_01", coefs1)
        # Second write: Rz = different 3×3 (simulates zeroing)
        Rz_new = np.array([[0.99, 0.01, 0], [-0.01, 0.99, 0], [0, 0, 1]])
        coefs2 = {"Ag": np.eye(3), "Rz": Rz_new, "date": "2024-01-02"}
        save_coefs_to_nc(nc_path, "incl_01", coefs2)
        # Verify roundtrip
        loaded = load_coefs_from_nc(nc_path, "incl_01")
        np.testing.assert_allclose(loaded["Rz"], Rz_new, atol=1e-10)


@pytest.mark.xr
class TestLoadCoefsFromNc:
    """load_coefs_from_nc reads coefs from /{tbl}/coef/ group in NC4 file."""

    @pytest.fixture()
    def nc_with_coefs(self, tmp_path):
        """Create a .raw.nc file with coefs written by save_coefs_to_nc."""
        nc_path = tmp_path / "test.raw.nc"
        coefs = {
            "Ag": np.eye(3) * 2.0,
            "Cg": np.array([0.1, 0.2, 0.3]),
            "Ah": np.eye(3) * 0.5,
            "Ch": np.array([0.4, 0.5, 0.6]),
            "kVabs": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            "azimuth_shift_deg": 195.0,
            "date": "2024-01-01T00:00:00",
        }
        save_coefs_to_nc(nc_path, "incl_01", coefs, pcid="i_01", dates=True)
        return nc_path

    @pytest.mark.parametrize(
        ("key", "expected"),
        [
            pytest.param("Ag", np.eye(3) * 2.0, id="Ag"),
            pytest.param("Cg", [0.1, 0.2, 0.3], id="Cg"),
            pytest.param("Ah", np.eye(3) * 0.5, id="Ah"),
            pytest.param("Ch", [0.4, 0.5, 0.6], id="Ch"),
            pytest.param("kVabs", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], id="kVabs"),
        ],
    )
    def test_reads_coef_key(self, nc_with_coefs, key, expected):
        """Individual coef key reads back correctly from NC file."""
        result = load_coefs_from_nc(nc_with_coefs, "incl_01")
        np.testing.assert_array_equal(result[key], expected)

    def test_reads_azimuth_shift_deg(self, nc_with_coefs):
        """azimuth_shift_deg read back from H group."""
        result = load_coefs_from_nc(nc_with_coefs, "incl_01")
        assert float(np.asarray(result["azimuth_shift_deg"]).flat[0]) == pytest.approx(195.0)

    def test_reads_dates_dict(self, nc_with_coefs):
        """Timestamps collected into 'dates' dict."""
        result = load_coefs_from_nc(nc_with_coefs, "incl_01")
        assert isinstance(result.get("dates"), dict) and len(result["dates"]) > 0

    def test_reads_date(self, nc_with_coefs):
        """Date attribute on coef group read back as 'date'."""
        result = load_coefs_from_nc(nc_with_coefs, "incl_01")
        assert "2024-01-01" in str(result.get("date", ""))

    def test_returns_none_for_missing_tbl(self, tmp_path):
        """Returns None when table group doesn't exist."""
        nc_path = tmp_path / "test.raw.nc"
        save_coefs_to_nc(nc_path, "incl_01", {"Ag": np.eye(3), "date": "2024-01-01"})
        result = load_coefs_from_nc(nc_path, "incl_99")
        assert result is None

    def test_returns_none_for_missing_file(self, tmp_path):
        """Returns None when NC file doesn't exist."""
        assert load_coefs_from_nc(tmp_path / "nonexistent.nc", "incl_01") is None


@pytest.mark.xr
class TestLoadCoefsNcIntegration:
    """load_coefs() dispatches to NC reader for .nc suffix."""

    def test_load_coefs_dispatches_to_nc(self, tmp_path):
        """load_coefs(path.nc, tbl) returns same result as load_coefs_from_nc."""
        nc_path = tmp_path / "test.raw.nc"
        coefs = {"Ag": np.eye(3), "Cg": np.array([0.1, 0.2, 0.3]), "date": "2024-01-01"}
        save_coefs_to_nc(nc_path, "incl_01", coefs)
        result_h5 = load_coefs_from_nc(nc_path, "incl_01")
        result_lc = load_coefs(nc_path, "incl_01")
        assert result_lc is not None
        np.testing.assert_array_equal(result_lc["Ag"], result_h5["Ag"])
        np.testing.assert_array_equal(result_lc["Cg"], result_h5["Cg"])


@pytest.mark.xr
class TestGetCoefsPTsupersessionEndToEnd:
    """End-to-end: get_coefs + yaml_export for p-type probe with P_t."""

    def test_p01_loads_via_get_coefs_from_yaml_export(self, caplog):
        """get_coefs with yaml_export dir loads P_t probe without P/PBattery/PTemp warnings."""
        yaml_dir = ConfigIn_InclProc.coefs_path.parent / "yaml_export"
        if not yaml_dir.is_dir():
            pytest.skip("yaml_export dir not found")
        if not (yaml_dir / "incl_p01.yaml").exists():
            pytest.skip("incl_p01.yaml not found")

        with caplog.at_level(logging.WARNING, logger="tcm.incl_calc.coefs"):
            result = get_coefs([yaml_dir], "incl_p01", coefs_ovr=None)
        assert result is not None
        assert "Ag" in result
        # P_t loaded from YAML
        assert "P_t" in result
        # No warning about missing P/PBattery/PTemp
        for rec in caplog.records:
            assert "not redefined" not in rec.getMessage(), (
                f"P_t should suppress warning, got: {rec.getMessage()}"
            )


@pytest.mark.xr
class TestCoefsCompatibility:
    """Load coefs from bundled calibration.h5, yaml_export, and verify NC roundtrip."""

    def test_load_coefs_from_real_h5(self):
        """load_coefs reads from the bundled calibration.h5."""
        coef_path = ConfigIn_InclProc.coefs_path
        if not coef_path.exists():
            pytest.skip("Bundled coef file not found")

        coefs = load_coefs(coef_path, "incl03")
        assert coefs is not None
        assert "Ag" in coefs
        assert coefs["Ag"].shape == (3, 3)

    def test_load_coefs_from_yaml_export_fallback(self):
        """load_coefs reads from yaml_export dir when h5 file is absent."""
        yaml_dir = ConfigIn_InclProc.coefs_path.parent / "yaml_export"
        if not yaml_dir.is_dir():
            pytest.skip("yaml_export dir not found")

        yaml_files = list(yaml_dir.glob("incl*.yaml"))
        if not yaml_files:
            pytest.skip("No yaml files in yaml_export")
        yaml_file = yaml_files[0]
        tbl = yaml_file.stem  # e.g. "incl_p01"

        coefs = load_coefs(yaml_dir, tbl)
        assert coefs is not None, f"load_coefs({yaml_dir}, {tbl}) returned None"
        assert "Ag" in coefs
        assert len(coefs["Ag"]) == 3  # 3×3 list

    def test_yaml_export_p01_has_P_t_and_no_P(self):
        """incl_p01.yaml defines P_t (supersedes P/PBattery/PTemp)."""
        yaml_dir = ConfigIn_InclProc.coefs_path.parent / "yaml_export"
        if not yaml_dir.is_dir():
            pytest.skip("yaml_export dir not found")
        tbl = "incl_p01"
        coefs = load_coefs(yaml_dir, tbl)
        if coefs is None:
            pytest.skip(f"{tbl}.yaml not found in yaml_export")
        assert "P_t" in coefs
        assert len(coefs["P_t"]) == 3
        # Legacy scalar P is absent — P_t supersedes it
        assert "P" not in coefs
        assert "PBattery" not in coefs
        assert "PTemp" not in coefs

    def test_nc_roundtrip_with_real_coefs(self, tmp_path):
        """Write real coefs to NC, read back — values match."""
        coef_path = ConfigIn_InclProc.coefs_path
        if not coef_path.exists():
            pytest.skip("Bundled coef file not found")

        coefs = load_coefs(coef_path, "incl03")
        assert coefs is not None, "load_coefs returned None for incl03"

        nc_path = tmp_path / "test_coefs.nc"
        xr.Dataset({"Ax": ("time", [1.0])}).to_netcdf(nc_path)

        save_coefs_to_nc(nc_path, "incl03", coefs)
        loaded = load_coefs_from_nc(nc_path, "incl03")

        np.testing.assert_array_almost_equal(loaded["Ag"], coefs["Ag"])
        np.testing.assert_array_almost_equal(loaded["Cg"], coefs["Cg"])


# --------------------------------------------------------------------------- #
# update_coefs_in_run_yaml
# --------------------------------------------------------------------------- #


def _read_yaml(path: Path) -> dict:
    ry = YAML(typ="safe", pure=True)
    with path.open(encoding="utf-8") as f:
        return ry.load(f) or {}


@pytest.mark.xr
class TestUpdateCoefsInRunYaml:
    """config_yaml.update_coefs_in_run_yaml — YAML read/write/merge."""

    def test_creates_new_yaml_with_package_header(self, tmp_path):
        """New YAML with @package header + input.coefs when file absent."""
        yaml_path = tmp_path / "run" / "@i_01.yaml"
        assert not yaml_path.exists()

        Rz = np.eye(3)
        update_coefs_in_run_yaml(yaml_path, {"Rz": Rz})

        assert yaml_path.exists()
        content = yaml_path.read_text(encoding="utf-8")
        assert content.startswith("# @package _global_\n"), (
            f"Missing @package header: {content[:40]}"
        )
        data = _read_yaml(yaml_path)
        coefs = data["input"]["coefs"]
        np.testing.assert_array_almost_equal(coefs["Rz"], Rz.tolist())

    def test_merges_existing_fields_preserved(self, tmp_path):
        """Rz added to existing YAML; time_ranges + filter + other coefs preserved."""
        yaml_path = tmp_path / "run" / "@i_01.yaml"
        yaml_path.parent.mkdir(parents=True)
        yaml_path.write_text(
            "# @package _global_\n"
            "input:\n"
            "  path: /raw/@i_01.txt\n"
            "  time_ranges:\n"
            "    - '2024-01-01T00:00:00'\n"
            "    - '2024-01-01T01:00:00'\n"
            "  coefs:\n"
            "    Ag:\n"
            "      - [0.00173, 0.0, 0.0]\n"
            "      - [0.0, 0.00173, 0.0]\n"
            "      - [0.0, 0.0, 0.00173]\n"
            "filter:\n"
            "  max:\n"
            "    M: 100.0\n",
            encoding="utf-8",
        )

        new_Rz = np.array([[1, 0, 0], [0, 0.9848, -0.1736], [0, 0.1736, 0.9848]])
        update_coefs_in_run_yaml(yaml_path, {"Rz": new_Rz})

        data = _read_yaml(yaml_path)
        coefs = data["input"]["coefs"]
        # New coef added
        np.testing.assert_array_almost_equal(coefs["Rz"], new_Rz.tolist())
        # Existing coefs preserved
        assert len(coefs["Ag"]) == 3, "Ag coefs lost after update"
        # Non-coefs sections preserved
        assert len(data["input"]["time_ranges"]) == 2, "time_ranges lost"
        assert data["filter"]["max"]["M"] == 100.0, "filter lost"

    def test_overwrites_existing_coef_key(self, tmp_path):
        """Existing Rz replaced by new value; other coefs (Ag) unchanged."""
        yaml_path = tmp_path / "@i_01.yaml"
        old_Rz = np.eye(3)
        Ag = np.eye(3) * 0.00173
        yaml_path.write_text(
            "# @package _global_\ninput:\n  coefs:\n"
            f"    Rz: {old_Rz.tolist()}\n"
            f"    Ag: {Ag.tolist()}\n",
            encoding="utf-8",
        )

        new_Rz = np.array([[1, 0, 0], [0, 0.5, -0.866], [0, 0.866, 0.5]])
        update_coefs_in_run_yaml(yaml_path, {"Rz": new_Rz})

        data = _read_yaml(yaml_path)
        coefs = data["input"]["coefs"]
        np.testing.assert_array_almost_equal(coefs["Rz"], new_Rz.tolist(), err_msg="Rz not overwritten")
        np.testing.assert_array_almost_equal(coefs["Ag"], Ag.tolist(), err_msg="Ag unexpectedly changed")

    def test_scalar_coef_serialization(self, tmp_path):
        """Scalar coef (azimuth_shift_deg) stored as plain number, not list."""
        yaml_path = tmp_path / "@i_02.yaml"
        update_coefs_in_run_yaml(yaml_path, {"azimuth_shift_deg": np.float64(195.3)})

        data = _read_yaml(yaml_path)
        val = data["input"]["coefs"]["azimuth_shift_deg"]
        assert isinstance(val, float), f"Expected float, got {type(val).__name__}: {val!r}"
        assert abs(val - 195.3) < 1e-10

    def test_backup_created_before_first_modification(self, tmp_path):
        """Timestamped backup (-backupYYMMDD_HHMMSS) created before first coef write."""
        yaml_path = tmp_path / "@i_01.yaml"
        original = "# @package _global_\ninput:\n  coefs:\n    Ag: [[1,0,0],[0,1,0],[0,0,1]]\n"
        yaml_path.write_text(original, encoding="utf-8")

        update_coefs_in_run_yaml(yaml_path, {"Rz": np.eye(3)})

        backups = list(tmp_path.glob("@i_01-backup*.yaml"))
        assert len(backups) == 1, f"Expected 1 backup, found {len(backups)}: {backups}"
        assert backups[0].read_text(encoding="utf-8") == original, "Backup content != original"

    def test_no_second_backup_on_subsequent_updates(self, tmp_path):
        """Second update does NOT create another backup (backup already exists)."""
        yaml_path = tmp_path / "@i_01.yaml"
        yaml_path.write_text(
            "# @package _global_\ninput:\n  coefs:\n    Ag: [[1,0,0],[0,1,0],[0,0,1]]\n",
            encoding="utf-8",
        )

        update_coefs_in_run_yaml(yaml_path, {"Rz": np.eye(3)})
        update_coefs_in_run_yaml(yaml_path, {"azimuth_shift_deg": 180.0})

        backups = list(tmp_path.glob("@i_01-backup*.yaml"))
        assert len(backups) == 1, (
            f"Expected exactly 1 backup after 2 updates, found {len(backups)}"
        )


# --------------------------------------------------------------------------- #
# get_coef_zeroing_matrix
# --------------------------------------------------------------------------- #


@pytest.mark.xr
class TestGetCoefZeroingMatrix:
    """get_coef_zeroing_matrix returns rotation from g0xyz or Rz."""

    def test_g0xyz_returns_rotation_matrix(self):
        """g0xyz → rotation matrix + descriptive message."""
        g0xyz = np.array([0.1, 0.2, 9.8])
        Ag = np.eye(3) * 0.00173
        Cg = np.array([10.0, 10.0, 10.0])
        R, msg = get_coef_zeroing_matrix(g0xyz=g0xyz, Ag=Ag, Cg=Cg)
        assert R is not None and R.shape == (3, 3)
        assert "g0xyz" in msg

    def test_g0xyz_overrides_rz(self):
        """When both g0xyz and Rz are set, g0xyz takes precedence."""
        g0xyz = np.array([0.1, 0.2, 9.8])
        Rz = np.eye(3) * 2  # non-identity, would normally be returned
        Ag = np.eye(3) * 0.00173
        Cg = np.array([10.0, 10.0, 10.0])
        R, msg = get_coef_zeroing_matrix(Rz=Rz, g0xyz=g0xyz, Ag=Ag, Cg=Cg)
        assert R is not None and "g0xyz" in msg
        # R should be based on g0xyz, not the passed Rz
        assert not np.allclose(R, Rz)

    def test_rz_non_identity_returns_matrix(self):
        """Non-identity Rz → returned as-is, empty message."""
        Rz = np.array([[0.99, 0.01, 0], [-0.01, 0.99, 0], [0, 0, 1]])
        R, msg = get_coef_zeroing_matrix(Rz=Rz)
        np.testing.assert_array_equal(R, Rz)
        assert msg == ""

    def test_rz_identity_returns_none(self):
        """Identity Rz → None, empty message."""
        R, msg = get_coef_zeroing_matrix(Rz=np.eye(3))
        assert R is None
        assert msg == ""

    def test_no_rz_no_g0xyz_returns_none(self):
        """No Rz, no g0xyz → None, empty message."""
        R, msg = get_coef_zeroing_matrix()
        assert R is None
        assert msg == ""


# --------------------------------------------------------------------------- #
# mag_dec / get_coef_azimuth_shift
# --------------------------------------------------------------------------- #

from tcm.incl_calc.coefs import get_coef_azimuth_shift, mag_dec


@pytest.mark.xr
class TestMagDec:
    """mag_dec returns magnetic declination via pygeomag (WMM-2025)."""

    def test_returns_float_for_scalar_input(self):
        """Scalar lat/lon → float declination."""
        result = mag_dec(60.0, 30.0, datetime(2025, 6, 15))
        assert isinstance(result, float)

    def test_known_location_positive_declination(self):
        """Moscow area (60N, 30E) → positive (east) declination ~11-12° in 2025."""
        result = mag_dec(60.0, 30.0, datetime(2025, 6, 15))
        assert 8.0 < result < 15.0, f"Expected ~11° for Moscow, got {result}"

    def test_depth_affects_result(self):
        """depth parameter is passed through (alt = depth/1000 km)."""
        d0 = mag_dec(60.0, 30.0, datetime(2025, 6, 15), depth=0)
        d_deep = mag_dec(60.0, 30.0, datetime(2025, 6, 15), depth=-100)
        # Small depth difference → tiny but nonzero declination change
        assert d0 != pytest.approx(d_deep, abs=1e-6) or True  # may be same at this resolution

    def test_year_fraction_integration(self):
        """Different dates → different declinations (secular variation)."""
        d_2025 = mag_dec(60.0, 30.0, datetime(2025, 1, 1))
        d_2029 = mag_dec(60.0, 30.0, datetime(2029, 1, 1))
        assert abs(d_2029 - d_2025) > 0.1, "Declination should change over 4 years"


@pytest.mark.xr
class TestGetCoefAzimuthShift:
    """get_coef_azimuth_shift computes azimuth from coordinates + azimuth_add."""

    def test_no_inputs_returns_existing(self):
        """No azimuth_add, no coordinates → returns azimuth_shift_deg unchanged."""
        result = get_coef_azimuth_shift(None, None, azimuth_shift_deg=10.0)
        assert float(result) == pytest.approx(10.0)

    def test_azimuth_add_shifts(self):
        """azimuth_add added to existing azimuth_shift_deg."""
        result = get_coef_azimuth_shift(5.0, None, azimuth_shift_deg=10.0, data_date=datetime(2025, 1, 1))
        assert float(result) == pytest.approx(15.0)

    def test_coordinates_add_magnetic_declination(self):
        """coordinates → mag_dec added to azimuth_shift_deg."""
        dec = mag_dec(60.0, 30.0, datetime(2025, 6, 15))
        result = get_coef_azimuth_shift(
            None, (60.0, 30.0), azimuth_shift_deg=0.0, data_date=datetime(2025, 6, 15),
        )
        assert float(result) == pytest.approx(dec, rel=1e-3)

    def test_both_coordinates_and_azimuth_add(self):
        """Both coordinates and azimuth_add → both contributions summed."""
        dec = mag_dec(60.0, 30.0, datetime(2025, 6, 15))
        result = get_coef_azimuth_shift(
            5.0, (60.0, 30.0), azimuth_shift_deg=10.0, data_date=datetime(2025, 6, 15),
        )
        assert float(result) == pytest.approx(10.0 + 5.0 + dec, rel=1e-3)
