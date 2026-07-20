"""Calibration entry point — reuses shared loaders, dispatches to calibration pipeline.

Replaces ``tcm._dask_legacy.incl_calibr_hy.main()``.

The loading path is shared with :mod:`tcm.processing`:
:func:`tcm._xr.io.load_raw` — one dispatch by suffix, no reimplementation.

Two entry points
----------------
* :func:`run_calibration` — **programmatic API** (no hydra, no sys.argv).

  ::

      from tcm.calibration.run import run_calibration
      run_calibration("experiment.raw.nc", tables=["incl63", "incl64"])

* ``python -m tcm.calibration.run "_raw/*i*.txt"`` — CLI via
  :func:`_hydra_main` (same pattern as ``scripts/tcm_clc.py``).

Structured configs
------------------
All calibration dataclasses inherit from the processing base schemas in
:mod:`tcm.config` — single source of truth:

- ``ConfigInCalib(ConfigIn_InclProc)`` — adds ``time_ranges_north`` only.
  ``channels`` is NOT a structured field; read via
  ``OmegaConf.select(cfg, "input.channels", default=["M","A"])``.
- ``ConfigFilterCalib(ConfigFilter_InclProc)`` — from :mod:`tcm.config`.
  Typed ``A``/``M`` per-axis despike overrides (``ConfigFilterChannel``).
- ``ConfigProcCalib`` — from :mod:`tcm.config`.  Maps to
  :class:`tcm.calibration.pipeline.PipelineConfig` fields.
  Per-channel ``field_magnitude`` via freeform
  ``+proc.field_magnitudes.M=52000`` (not in schema).
- ``ConfigOutCalib`` — output paths (``db_paths`` only).
- ``ConfigProgramCalib(ConfigProgram)`` — adds ``dask_scheduler``.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import xarray as xr
from hydra.core.config_store import ConfigStore
from tcm import h5inclinometer_coef, utils2init, cli
from tcm._xr import io as xr_io
from tcm.calibration import filtering, pipeline, robust, visualization
from tcm.calibration.calibrate import to_unit_vector

lf = utils2init.LoggingStyleAdapter(__name__)


# =================================================================== #
# Programmatic API — no hydra, no sys.argv
# =================================================================== #

def run_calibration(
    cfg: Mapping[str, Mapping[str, Any]],
) -> Dict[str, dict]:
    """Calibrate inclinometer channels for the given tables.

    This is the **primary entry point** — called by Hydra (via
    :func:`cli.call_in_raw_dir`) and by test fixtures directly.

    Parameters
    ----------
    cfg
        Hydra-composed top-level config (``DictConfig`` or plain dict).
        ``cfg.proc`` is resolved as :class:`PipelineConfig` natively
        by Hydra (``ConfigProcCalib = PipelineConfig``).

        Supported ``+`` overrides (freeform, not in structured schema):

        ``+input.channels=[M]``
            Restrict calibration to specific channel types.
            Default ``["M", "A"]``.

        ``+proc.field_magnitudes.M=52000``
            Per-channel reference magnitude override.  When a channel
            key is present, its value replaces ``proc.field_magnitude``
            for that channel only (via :func:`dataclasses.replace`).
            Channels absent from the dict use the global
            ``field_magnitude`` (default ``1.0``).

    Returns
    -------
    dict
        ``{table: {channel: {"A": gain, "b": bias}}}`` — computed
        coefficients for all successfully calibrated tables.
    """
    cfg = cli.main_init(cfg, program_name="Calibration")

    # ── PipelineConfig + freeform +overrides from resolved dict ─────────
    proc = cfg.get("proc") or {}
    # Freeform +override: +proc.field_magnitudes.M=52000 (not in structured schema)
    ch_field_magnitudes = proc.pop("field_magnitudes", None)
    base_pc = pipeline.PipelineConfig(**{
        k: v for k, v in proc.items() if k in pipeline.PipelineConfig.__dataclass_fields__
    })

    db_paths = [Path(p) for p in cfg["out"]["db_paths"] or []]

    cfg_in = cfg["input"]
    cfg_filter = cfg["filter"]

    if not cfg_in["tables"] :
        raise ValueError("tables must list at least one table name")
    channels = list(cfg_in.get("channels", ["M", "A"]))
    (fig_dir := cfg_in["path"].parent / "images-channels_calibration").mkdir(exist_ok=True)

    lf.info(
        "Begin {:s}({:s}) for channels: {}",
        utils2init.this_prog_basename(__file__),
        ", ".join(cfg_in["tables"]),
        channels,
    )

    # Build per-channel despike overrides
    def _ch_overrides(channel: str) -> dict:
        if cfg_filter is None:
            return {}
        col_str, _ = h5inclinometer_coef.channel_cols(channel)
        ch_cfg = cfg_filter.get(col_str, {})
        return {
            axis: {
                k: v
                for k, v in {
                    "blocks": ch_cfg.get(axis, {}).get("blocks"),
                    "offsets": ch_cfg.get(axis, {}).get("offsets"),
                    "std_smooth_sigma": ch_cfg.get(axis, {}).get("std_smooth_sigma"),
                }.items()
                if v is not None
            }
            for axis in "xyz"
            if axis in ch_cfg
        }

    coefs: Dict[str, dict] = {}
    fig_filt, fig_fit = None, None
    for tbl in cfg_in["tables"]:
        # ── Load using shared infrastructure ─────────────────────────────
        lf.info("Loading table '{}' from {}", tbl, cfg_in["path"].name)
        ds, _coefs_from_file = xr_io.load_raw(cfg_in["path"], tbl=tbl, cfg_in=cfg_in)
        if ds is None or ds.sizes.get("time", 0) == 0:
            lf.error("No data for {} — skipping", tbl)
            continue

        coefs[tbl] = {}
        for channel in channels:
            print(f' channel "{channel}"', end=" ")
            data_3d = _extract_channel(ds, channel)
            if data_3d is None:
                lf.warning("Channel '{}' columns not in {} — skipping", channel, tbl)
                continue

            # ── Despike ──────────────────────────────────────────────────
            ch_cfg = _ch_overrides(channel)
            data_3d_filt, mask_good = filtering.despike_channels(data_3d, **ch_cfg)
            lf.info("despike({}): {}/{} points kept", channel, mask_good.sum(), data_3d.shape[1])

            # ── Plot filtered channels ───────────────────────────────────
            fig_filt, _ = visualization.plot_despiked_channels(
                ds["time"].values, data_3d, mask_good=mask_good, fig=fig_filt,
                fig_save_prefix=str(fig_dir / tbl) + f"-'{channel}'",
                window_title=f"channel {channel}",
            )

            # ── Calibrate pipeline (fit → reject) ────────────────────────
            pc = (  # Per-channel field_magnitude override (e.g. via +proc.field_magnitudes.M=52000)
                replace(base_pc, field_magnitude=ch_field_magnitudes[channel])
                if ch_field_magnitudes and channel in ch_field_magnitudes
                else base_pc
            )
            result = pipeline.calibrate_pipeline(data_3d_filt, pc)

            # ── Coverage diagnostic ──────────────────────────────────────
            query_dirs, density = robust.coverage_at(
                data_3d_filt, result.calibration, query_directions=int(max(200, result.n_inliers/9))
            )  # number of bins selected to get 9 points per segment (in average if ideal spread)
            n_zero = (density == 0).sum().item()
            lf.info(
                "'{}' directional coverage: {}/{}, density range=[{:.3g}, {:.3g}]",
                channel, query_dirs.shape[1] - n_zero, query_dirs.shape[1], density.min(), density.max(),
            )

            # ── Plot calibration result ──────────────────────────────────
            if (fig_fit := visualization.calibrate_plot(
                data_3d_filt, result.gain, result.bias,
                fig=fig_fit, window_title=f'{tbl} "{channel}"-channel ellipse',
                projection=pc.calibration_projection,
                field_magnitude=pc.field_magnitude,
            )) is not None:
                fig_fit.savefig(
                    fig_dir / f"{tbl} {channel}-channel ellipse.png", dpi=300, bbox_inches="tight"
                )

            # ── Uncertainty diagnostic ──────────────────────────────────
            _, uncertainty = robust.uncertainty_at(
                data_3d_filt, pc.field_magnitude, query_directions=query_dirs, weighted=pc.weighted,
            )

            # ── Plot coverage + uncertainty ──────────────────────────────
            if (fig_cov := visualization.coverage_heatmap(
                query_dirs, density,
                fig=None, projection=pc.coverage_projection,
                window_title=f"{tbl} '{channel}' coverage",
                sample_directions=to_unit_vector(data_3d_filt, result.calibration),
                uncertainty=uncertainty,
            )) is not None:
                fig_cov.savefig(fig_dir / f"{tbl} '{channel}' coverage.png", dpi=300, bbox_inches="tight")

            coefs[tbl][channel] = {"A": result.gain, "b": result.bias}

        # ── Save coefficients ────────────────────────────────────────────
        if not coefs[tbl]:
            lf.warning("No coefficients computed for {}", tbl)
            continue

        ds.close()  # Release xarray's netCDF4 file handle before h5py opens the same file for writing.
        for db_path in db_paths:
            lf.info("Writing coefs to {}/{}", db_path, tbl)
            dict_matrices = h5inclinometer_coef.dict_matrices_for_h5(coefs[tbl], tbl, channels)
            h5inclinometer_coef.h5copy_coef(None, db_path, tbl, dict_matrices=dict_matrices, dates=True)

    if not coefs:
        lf.warning("No probes calibrated from {}", cfg_in["path"])
    else:
        lf.info("Calibration complete for {} table(s)", len(coefs))
    return coefs


# --------------------------------------------------------------------------- #
# Channel extraction helper
# --------------------------------------------------------------------------- #

def _extract_channel(ds: xr.Dataset, channel: str) -> "np.ndarray | None":
    """Extract 3×N array ``(Mx,My,Mz)`` or ``(Ax,Ay,Az)`` from a Dataset.

    Returns ``None`` if any column is missing.
    """
    col_str, _ = h5inclinometer_coef.channel_cols(channel)
    cols = [f"{col_str}{c}" for c in "xyz"]
    if not all(c in ds.data_vars for c in cols):
        return None
    return np.vstack([ds[c].values for c in cols])


# ── Structured config (inherits from tcm.config base schemas) ────────────── #

from tcm.config import (  # noqa: E402 — after __future__ annotations
    ConfigFilterCalib, ConfigIn_InclProc, ConfigProcCalib, ConfigProgram,
)


@dataclass
class ConfigInCalib(ConfigIn_InclProc):
    """Calibration input — inherits all load-stage fields + adds ``time_ranges_north``.

    ``channels`` is NOT a structured field (function-default ``["M","A"]``
    in :func:`run_calibration`).  Override via ``+input.channels=[M]`` at
    the Hydra CLI — :func:`_hydra_main`` reads it via
    ``OmegaConf.select(cfg, "input.channels", default=["M","A"])``.
    """
    time_ranges_north: List[str] = field(default_factory=list)


@dataclass
class ConfigOutCalib:
    """Calibration output — where to write coefficients (simpler than processing output)."""
    db_paths: List[str] = field(default_factory=list)


@dataclass
class ConfigProgramCalib(ConfigProgram):
    """Calibration program behavior — adds ``dask_scheduler``."""
    dask_scheduler: str = "synchronous"


# ── Hydra ConfigStore registration ──────────────────────────────────────────── #

_cs_store_name = Path(__file__).stem  # "run"
_cs = ConfigStore.instance()
_cs.store(group="input", name="base", node=ConfigInCalib)
_cs.store(group="out", name="base", node=ConfigOutCalib)
_cs.store(group="filter", name="base", node=ConfigFilterCalib)
_cs.store(group="proc", name="calib", node=ConfigProcCalib)
_cs.store(group="program", name="base", node=ConfigProgramCalib)


if __name__ == "__main__":
    cli.call_in_raw_dir(run_calibration, config_name=_cs_store_name)
