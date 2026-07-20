"""Integration tests — discovery → config generation → stale detection.

Uses pytest fixtures + monkeypatch (no unittest.mock.patch).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from omegaconf import DictConfig

from tcm import _constants, config_yaml as cfg_mod, csv_load
from tcm._constants import RAW_DIR_NAME
from tcm.config import Return
from tcm.config_yaml import (
    find_stale_cfgs,
    get_existed_cfgs,
    save_config_to_yaml,
    update_coefs_in_run_yaml,
)
from tcm.processing import run_processing


@pytest.fixture()
def raw_dir(tmp_path):
    """Create _raw directory with test CSV files."""
    d = tmp_path / RAW_DIR_NAME
    d.mkdir()
    for name in ("i_01.txt", "i_02.txt"):
        (d / name).write_text(
            "header1\nheader2\nheader3\n"
            "2024,06,13,12,00,00,100,200,300,400,500,600,25.0\n"
            "2024,06,13,12,00,01,101,201,301,401,501,601,25.1\n"
            "2024,06,13,12,00,02,102,202,302,402,502,602,25.2\n",
            encoding="utf-8",
        )
    return d


def _make_edge_df():
    idx = pd.to_datetime(["2024-06-13 12:00:00", "2024-06-13 12:00:02"]).tz_localize("UTC")
    return pd.DataFrame({"Ax": [100, 102]}, index=idx)


def _patch_discovery(monkeypatch, raw_dir, n_probes=1):
    """Monkeypatch search_csv_files and load_from_csv_gen."""

    files = sorted(raw_dir.glob("i_*.txt"))
    search_result = {("i", i + 1): [f] for i, f in enumerate(files[:n_probes])}
    monkeypatch.setattr(csv_load, "search_csv_files", lambda p: search_result)

    def fake_load(csv_files_dict, cfg_in, return_="first_last_row"):
        for (model, number), paths in csv_files_dict.items():
            yield _make_edge_df(), (f"i{number:02d}", f"i{number:02d}", paths[0])

    monkeypatch.setattr(csv_load, "load_from_csv_gen", fake_load)


def _base_cfg(raw_dir, path=None):
    return {
        "input": {
            "path": path or raw_dir / "i_01.txt", "tables": ["incl*"],
            "coefs_path": None, "coefs": {},
            "text_type": None, "text_line_regex": None, "prefix": None, "dt_from_utc": 0,
            "corr_time_mode": None,  # moved from filter
        },
        "filter": {},
        "out": {"dt_bins": [0], "table": ""},
    }


@pytest.mark.xr
class TestIntegrationDiscovery:
    """gen_metadata + save_config_to_yaml + find_stale_cfgs."""

    @pytest.mark.parametrize("n_probes,expected_pcids", [
        pytest.param(1, {"i01"}, id="single-probe"),
        pytest.param(2, {"i01", "i02"}, id="multiple-probes"),
    ])
    def test_gen_metadata_discovers_probes(self, raw_dir, monkeypatch, n_probes, expected_pcids):

        _patch_discovery(monkeypatch, raw_dir, n_probes)
        # Patch on the coefs object that config_yaml imported (module-level import)
        monkeypatch.setattr(
            cfg_mod.coefs, "prep_cfg_for_probe",
            lambda pcid, *a, **kw: {"input": {"path": raw_dir / f"{pcid}.txt"}, "out": {}, "filter": {}},
        )

        path = raw_dir / "i_01.txt" if n_probes == 1 else raw_dir / "*i*.txt"
        results = list(cfg_mod.gen_metadata(_base_cfg(raw_dir, path), [path]))
        pcids = {pcid for _, (_, pcid, _) in results}
        assert pcids == expected_pcids

    def test_gen_metadata_yields_time_ranges(self, raw_dir, monkeypatch):

        _patch_discovery(monkeypatch, raw_dir, 1)
        monkeypatch.setattr(
            cfg_mod.coefs, "prep_cfg_for_probe",
            lambda pcid, *a, **kw: {"input": {"path": raw_dir / f"{pcid}.txt"}, "out": {}, "filter": {}},
        )

        results = list(cfg_mod.gen_metadata(_base_cfg(raw_dir), [raw_dir / "i_01.txt"]))
        cfg1, _ = results[0]
        assert cfg1["input"].get("time_ranges") is not None
        assert len(cfg1["input"]["time_ranges"]) == 2

    def test_save_and_find_stale_cycle(self, raw_dir, monkeypatch):

        raw_path = raw_dir / "i_01.txt"

        fake_cfg1 = {
            "input": {"path": raw_path, "tables": ["incl*"], "text_type": None,
                      "corr_time_mode": None},
            "out": {"dt_bins": [0], "table": ""},
            "filter": {},
            "coefs": {},
        }
        monkeypatch.setattr(
            cfg_mod, "gen_metadata",
            lambda cfg, paths, **kw: iter([(fake_cfg1, (False, "i01", None))]),
        )

        cfg = _base_cfg(raw_dir)
        save_config_to_yaml(cfg, [raw_path])

        # Configs at _raw/cfg_proc/run/
        run_dir = raw_dir / "cfg_proc" / "run"
        cfgs = get_existed_cfgs(run_dir)
        assert "i01" in cfgs

        # Valid → no stale
        assert len(find_stale_cfgs(cfgs, run_dir)) == 0

        # Delete source → stale
        raw_path.unlink()
        assert "i01" in find_stale_cfgs(cfgs, run_dir)


# ---------------------------------------------------------------------------
# Pipeline integration: CSV → raw.nc → proc.nc
# ---------------------------------------------------------------------------


@pytest.mark.xr
class TestPipelineIntegration:
    """Full pipeline: load_raw → prepare_coefs → _process_and_persist → proc.nc."""

    @pytest.fixture()
    def csv_file(self, tmp_path):
        """Synthetic inclinometer CSV (10 rows, 1 Hz)."""
        raw = tmp_path / RAW_DIR_NAME
        raw.mkdir()
        f = raw / "@i_01.txt"
        lines = [
            "yyyy,mm,dd,HH,MM,SS,Ax,Ay,Az,Mx,My,Mz,Battery,Temp",
            *[
                f"2024,01,01,00,{i // 60:02d},{i % 60:02d},"
                f"{100 + i},{200 + i},{300 + i},{-400 - i},{-500 - i},{-600 - i},{12.5},{25.0}"
                for i in range(10)
            ],
        ]
        f.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return f

    @pytest.fixture()
    def synthetic_ds(self):
        """Synthetic Dataset with known sensor data (bypasses CSV parser).

        Time index is tz-naive (UTC implied) — matches CSV-produced datasets
        and netCDF4 compatibility.
        """
        n = 100
        rng = np.random.default_rng(42)
        time = pd.date_range("2024-01-01", periods=n, freq="s")
        # ~10° tilt, ~45° heading — same geometry as conftest._make_sensor_data
        Ax = np.full(n, np.sin(np.radians(10.0)))
        Ay = np.zeros(n)
        Az = np.full(n, np.cos(np.radians(10.0)))
        Mx = np.full(n, np.cos(np.radians(45.0)) * np.cos(np.radians(10.0)))
        My = np.full(n, np.sin(np.radians(45.0)))
        Mz = np.full(n, -np.cos(np.radians(45.0)) * np.sin(np.radians(10.0)))
        return xr.Dataset(
            {
                "Ax": ("time", Ax), "Ay": ("time", Ay), "Az": ("time", Az),
                "Mx": ("time", Mx), "My": ("time", My), "Mz": ("time", Mz),
                "Battery": ("time", rng.uniform(12, 13, n)),
                "Temp": ("time", rng.uniform(24, 26, n)),
            },
            coords={"time": time},
        )

    def _make_cfg(self, tmp_path, csv_file):
        """DictConfig for run_processing."""
        out_dir = tmp_path / "out"
        out_dir.mkdir(exist_ok=True)
        return DictConfig({
            "input": {
                "path": str(csv_file),
                "coefs_path": None,
                "coefs": {},
                "tables": ["incl*"],
                "text_type": None,
                "text_line_regex": None,
                "prefix": None,
                "dt_from_utc": 0,
                "corr_time_mode": None,  # moved from filter
            },
            "out": {
                "dt_bins": [0],
                "dir": str(out_dir),
                "raw_db_path": str(out_dir / "i_01.raw.nc"),
            },
            "filter": {},
        })

    def _known_coefs(self):
        """Standard coefs dict for pipeline tests."""
        return {
            "Ag": np.eye(3) * 0.00173, "Cg": np.zeros(3),
            "Ah": np.eye(3), "Ch": np.zeros(3),
            "kVabs": np.array([1.0, 0.0, 0.5]),
            "azimuth_shift_deg": 180.0,
        }

    def test_csv_to_proc_nc(self, pipeline_env, mock_pipeline, mocker):
        """CSV input → raw.nc (with coefs) → shared proc_noAvg.nc with group."""
        import h5py

        pipeline_env.cfg.out.dt_bins = [0]
        mock_pipeline(pipeline_env.cfg, pipeline_env, mocker)
        run_processing(pipeline_env.cfg)

        # Verify raw.nc created with coefs group
        assert pipeline_env.raw_db_path.exists(), "raw.nc not created"
        with h5py.File(pipeline_env.raw_db_path, "r") as f:
            assert "incl01/coef" in f, (
                f"coefs group missing from raw.nc, got: {list(f.keys())}"
            )

        # Verify shared proc_noAvg.nc created with probe group
        assert pipeline_env.noavg_path.exists(), (
            f"proc_noAvg.nc not created; proc_dir: {list(pipeline_env.proc_dir.iterdir())}"
        )
        with h5py.File(pipeline_env.noavg_path, "r") as f:
            assert "i01" in f, (
                f"Probe group 'i01' missing from proc_noAvg.nc, got: {list(f.keys())}"
            )
        ds = xr.open_dataset(pipeline_env.noavg_path, group="i01")
        try:
            assert any(v in ds.data_vars for v in ("v", "Vabs", "Vdir")), (
                f"Expected velocity vars in proc_noAvg.nc, got: {list(ds.data_vars)}"
            )
        finally:
            ds.close()

    def test_prepare_coefs_called(self, pipeline_env, mock_pipeline, mocker):
        """prepare_coefs is wired into run_processing — coef_zeroing_matrix flows to process."""

        pipeline_env.cfg.out.dt_bins = [0]
        pipeline_env.cfg.input.time_ranges_zeroing = [
            "2024-01-01T00:00:00", "2024-01-01T00:00:05",
        ]

        mock_pipeline(pipeline_env.cfg, pipeline_env, mocker)
        mock_process = mocker.patch("tcm._xr.physical.process", wraps=None)
        mock_process.return_value = [pipeline_env.synthetic_ds]

        run_processing(pipeline_env.cfg)

        _, kwargs = mock_process.call_args
        assert "coef_zeroing_matrix" in kwargs

    def test_incremental_skip(self, pipeline_env, mock_pipeline, mocker):
        """Second run skips already-processed raw data (incremental update)."""
        import h5py

        pipeline_env.cfg.out.dt_bins = [0]
        mock_pipeline(pipeline_env.cfg, pipeline_env, mocker)

        run_processing(pipeline_env.cfg)
        run_processing(pipeline_env.cfg)  # second run — should not crash

        # raw.nc still exists and has data
        assert pipeline_env.raw_db_path.exists()
        with h5py.File(pipeline_env.raw_db_path, "r") as f:
            assert "incl01/coef" in f


# ---------------------------------------------------------------------------
# Coefficient persistence: NC-source overwrite + noh5 YAML fallback
# ---------------------------------------------------------------------------


@pytest.mark.xr
class TestCoefPersistence:
    """run_processing + zeroing → coefs persisted to NC or YAML."""

    @pytest.fixture()
    def nc_source_env(self, pipeline_env, tmp_path):
        """Create a *.raw.nc with initial coefs (simulates pre-existing NC source)."""
        from tcm._xr.coefs import save_coefs_to_nc

        # Ensure use_h5 is enabled for fixture setup (NC write).
        _constants.use_h5_set(True)
        raw_nc = pipeline_env.raw_db_path
        # Write minimal NC structure + initial coefs
        ds = pipeline_env.synthetic_ds.copy()
        ds.to_netcdf(raw_nc)
        initial_coefs = dict(pipeline_env.coefs)
        initial_coefs["Rz"] = np.eye(3)  # initial Rz = identity
        save_coefs_to_nc(raw_nc, "incl01", initial_coefs, pcid="i01")

        # Point cfg.input.path to the NC file (NC-source mode)
        pipeline_env.cfg.input.path = str(raw_nc)
        pipeline_env.cfg.input.tables = ["incl01"]
        return pipeline_env

    def test_nc_source_coef_overwrite(self, nc_source_env, mock_pipeline, mocker):
        """Source=*.raw.nc, time_ranges_zeroing → Rz written back to source NC."""
        import h5py
        from tcm._xr.coefs import load_coefs_from_nc

        env = nc_source_env
        env.cfg.input.time_ranges_zeroing = [
            "2024-01-01T00:00:00", "2024-01-01T00:00:05",
        ]
        env.cfg.out.dt_bins = [0]

        # Mock for NC-source mode: main_init returns NC path, load_raw returns synthetic data
        from tcm.paths import PathLayout
        try:
            layout = PathLayout.from_cfg(env.cfg.input, env.cfg.out)
            layout.apply_to_cfg(env.cfg.out)
        except (ValueError, OSError):
            pass

        cfg_t = {
            "input": {
                "path": Path(env.cfg.input.path),
                "tables": ["incl01"],
                "coefs_path": None, "coefs": {},
                "dt_from_utc": pd.Timedelta(0),
                "corr_time_mode": None,
                "time_ranges_zeroing": env.cfg.input.time_ranges_zeroing,
            },
            "out": {},
            "filter": {},
            "program": {"return_": Return.END},
        }
        for k in ("raw_db_path", "not_joined_db_path", "db_path", "text_path"):
            v = env.cfg.out.get(k)
            cfg_t["out"][k] = Path(v) if v is not None else None
        for k, v in env.cfg.out.items():
            if k not in cfg_t["out"]:
                cfg_t["out"][k] = v
        cfg_t["out"]["dt_bins"] = [pd.Timedelta(seconds=0)]

        mocker.patch("tcm.processing.cli.main_init", return_value=cfg_t)
        mocker.patch(
            "tcm.processing.xr_io.load_raw",
            return_value=(env.synthetic_ds, None),
        )
        mocker.patch("tcm.processing.get_coefs_from_cfg", return_value=env.coefs)

        run_processing(env.cfg)

        # Verify Rz changed in the NC file (was identity, now is zeroing rotation)
        loaded = load_coefs_from_nc(env.raw_db_path, "incl01")
        assert loaded is not None
        Rz = loaded["Rz"]
        assert Rz.shape == (3, 3)
        assert not np.allclose(Rz, np.eye(3)), (
            f"Rz still identity after zeroing — coefs not written back to NC: {Rz}"
        )

    def test_noh5_yaml_fallback(self, nc_source_env, mock_pipeline, mocker):
        """H5_AVAILABLE=False → changed coefs written to run YAML instead of NC."""
        from tcm._xr import coefs as xr_coefs_mod

        env = nc_source_env
        env.cfg.input.time_ranges_zeroing = [
            "2024-01-01T00:00:00", "2024-01-01T00:00:05",
        ]
        env.cfg.out.dt_bins = [0]

        # Create a run YAML to receive coefs
        run_dir = env.raw_dir / "cfg_proc" / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        yaml_path = run_dir / "@i_01.yaml"
        yaml_path.write_text(
            "# @package _global_\ninput:\n  coefs:\n    Ag: [[0.00173,0,0],[0,0.00173,0],[0,0,0.00173]]\n",
            encoding="utf-8",
        )

        # Monkeypatch H5_AVAILABLE to False + reset use_h5 to None (noh5 auto-skip)
        mocker.patch.object(_constants, "H5_AVAILABLE", False)
        _constants.use_h5_set(None)

        # Mock pipeline
        from tcm.paths import PathLayout
        try:
            layout = PathLayout.from_cfg(env.cfg.input, env.cfg.out)
            layout.apply_to_cfg(env.cfg.out)
        except (ValueError, OSError):
            pass

        cfg_t = {
            "input": {
                "path": Path(env.cfg.input.path),
                "tables": ["incl01"],
                "coefs_path": None, "coefs": {},
                "dt_from_utc": pd.Timedelta(0),
                "corr_time_mode": None,
                "time_ranges_zeroing": env.cfg.input.time_ranges_zeroing,
            },
            "out": {},
            "filter": {},
            "program": {"return_": Return.END},
            "_yaml_path": yaml_path,
        }
        for k in ("raw_db_path", "not_joined_db_path", "db_path", "text_path"):
            v = env.cfg.out.get(k)
            cfg_t["out"][k] = Path(v) if v is not None else None
        for k, v in env.cfg.out.items():
            if k not in cfg_t["out"]:
                cfg_t["out"][k] = v
        cfg_t["out"]["dt_bins"] = [pd.Timedelta(seconds=0)]

        mocker.patch("tcm.processing.cli.main_init", return_value=cfg_t)
        mocker.patch(
            "tcm.processing.xr_io.load_raw",
            return_value=(env.synthetic_ds, None),
        )
        mocker.patch("tcm.processing.get_coefs_from_cfg", return_value=env.coefs)
        # Mock save_coefs_to_nc to raise ImportError (h5py not available)
        mocker.patch.object(xr_coefs_mod, "save_coefs_to_nc", side_effect=ImportError("no h5py"))

        run_processing(env.cfg)

        # Verify Rz written to YAML
        from ruamel.yaml import YAML
        ry = YAML(typ="safe", pure=True)
        with yaml_path.open(encoding="utf-8") as f:
            data = ry.load(f)
        coefs = data["input"]["coefs"]
        assert "Rz" in coefs, f"Rz missing from YAML coefs after noh5 zeroing. Keys: {list(coefs)}"
        Rz = np.array(coefs["Rz"])
        assert Rz.shape == (3, 3)
        assert not np.allclose(Rz, np.eye(3)), (
            f"Rz still identity in YAML after noh5 zeroing: {Rz}"
        )
