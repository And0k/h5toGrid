"""Tests for tcm/config.py — ConfigStore registration and Config dataclass."""
from __future__ import annotations

import os

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

import tcm.config  # noqa: F401 — triggers ConfigStore registration
from tcm.config import Config, ConfigIn_InclProc

_CONFIG_YAML = (
    "defaults:\n"
    "  - input: base\n  - out: base\n  - filter: base\n  - program: base\n  - _self_\n"
)


@pytest.fixture()
def cfg_dir(tmp_path):
    """Temp dir with cfg_proc/config.yaml ready for Hydra compose."""
    d = tmp_path / "cfg_proc"
    d.mkdir()
    (d / "config.yaml").write_text(_CONFIG_YAML)
    return d


def _compose_with(cfg_dir, overrides=None):
    """Compose config from cfg_dir, restoring CWD after."""
    old_cwd = os.getcwd()
    try:
        os.chdir(str(cfg_dir.parent))
        with initialize_config_dir(config_dir=str(cfg_dir), version_base="1.3"):
            return compose(config_name="config", overrides=overrides or [])
    finally:
        os.chdir(old_cwd)


@pytest.mark.xr
class TestConfigStore:
    """ConfigStore has all required groups registered."""

    def test_config_store_registers_all_groups(self):
        """ConfigStore has input, out, filter, program groups."""
        all_keys = set(ConfigStore.instance().repo.keys())
        for group in ("input", "out", "filter", "program"):
            has_group = group in all_keys or any(k.startswith(f"{group}/") for k in all_keys)
            assert has_group, (
                f"ConfigStore missing group '{group}' — registered keys: {sorted(all_keys)}"
            )

    def test_config_defaults_compose(self, cfg_dir):
        """compose(config_name='config') produces valid DictConfig with all groups."""
        cfg = _compose_with(cfg_dir)
        assert isinstance(cfg, DictConfig)
        assert {"input", "out", "filter", "program"} <= cfg.keys()

    def test_config_override_dt_bins(self, cfg_dir):
        """CLI override out.dt_bins=[0,600] is reflected in composed config."""
        cfg = _compose_with(cfg_dir, overrides=["out.dt_bins=[0,600]"])
        assert cfg.out.dt_bins == [0, 600], (
            f"Override not reflected: expected [0, 600], got {cfg.out.dt_bins}"
        )


@pytest.mark.xr
class TestConfigDataclass:
    """Config dataclass has correct field types and defaults."""

    @pytest.mark.parametrize(
        "cls,required_fields",
        [
            pytest.param(Config, ("input", "out", "filter", "program", "defaults"), id="Config"),
            pytest.param(ConfigIn_InclProc, ("coefs", "coefs_path", "path"), id="ConfigIn"),
        ],
    )
    def test_has_required_fields(self, cls, required_fields):
        """Dataclass has all required fields."""
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        assert set(required_fields) <= field_names
