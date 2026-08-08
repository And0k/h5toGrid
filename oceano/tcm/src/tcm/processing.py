"""Processing pipeline entry point for the xr-native workflow.

:func:`run` is the canonical orchestrator: discover → generate configs → process.
:func:`run_processing` processes a single run YAML.
:func:`process_inmemory` is a standalone API for callers with an existing Dataset.
"""

from __future__ import annotations

import contextlib
import re
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

import numpy as np
import tcm._xr.nc_utils
import xarray as xr
from omegaconf import DictConfig, OmegaConf
from tqdm.dask import TqdmCallback


from tcm import _constants, cli, config_yaml, format, paths, stage_ctx, utils2init, policy, schema
from tcm._xr import coefs as xr_coefs
from tcm._xr import dataset, physical, storage
from tcm._xr import io as xr_io
from tcm.incl_calc.coefs import get_coefs_from_cfg

try:
    from tcm_gui import progress_bridge
except ImportError:

    class DumbChain:
        """Attributes as no-op callables"""

        __slots__ = ()

        def __getattr__(self, _):
            return self

        def __call__(self, *_, **_kw):
            return None

    progress_bridge = DumbChain()  # fallback (GUI not installed)

lf = utils2init.LoggingStyleAdapter(__name__)

# Extensions that carry their own coefs (no text-file config discovery).
_EXT_BINARY = _constants._EXT_NC | _constants._EXT_HDF5


# ── Upper-bar stage ticks (one tick per stage boundary) ─────────────────
# ``Stage`` labels the per-probe phases; the upper bar advances within each
# probe as stages start.  Each probe occupies 100 units of the upper-bar
# scale; active stages share it evenly (NC only when ``io()`` is truthy,
# TSV only when ``text_path`` set).  Bottom bar (dask TqdmCallback) covers
# substages continuously within a stage.


class Stage(StrEnum):
    """Per-probe phase labels (value = upper-bar description text)."""

    LOAD = "load"  # xr_io.load_raw / _load_batch
    COEFS = "coefs"  # prepare_coefs + save
    PROC = "proc"  # physical.process (calc + binning)
    NC = "NC"  # store_processed_incremental (per bin), use_h5 only
    TSV = "TSV"  # xr_io.ds_to_csv (per bin), text_path only
    COMBINE = "combine"  # _combine_probes (post-loop, not per-probe)


# ---------------------------------------------------------------------------
# Config helpers — deduplicate repeated cfg.out access patterns
# ---------------------------------------------------------------------------


def _dt_bins(cfg_out) -> list[timedelta]:
    """``cfg.out.dt_bins`` normalised to ``list[timedelta]``, default ``[0]``.

    After :func:`tcm.cli.main_init`, ``dt_bins`` elements are already
    ``timedelta`` (``type_fix`` converts the ``dt_*`` prefix).  This helper
    guarantees ``timedelta`` regardless of source (int, float, or timedelta).
    """
    bins = cfg_out.get("dt_bins") or [0]
    return [b if isinstance(b, timedelta) else timedelta(seconds=int(b)) for b in bins]


def _dt_min_save(cfg_out) -> timedelta:
    """``cfg.out.dt_bins_min_save_text`` as timedelta, default 1 s.

    After ``main_init`` the value is already ``timedelta``; this helper
    handles both ``timedelta`` and raw ``int`` inputs.
    """
    val = cfg_out.get("dt_bins_min_save_text")
    return val if isinstance(val, timedelta) else timedelta(seconds=int(val or 1))


def _output_nc_paths(cfg_out) -> tuple[Path | None, Path | None, Path | None]:
    """Resolve ``(noavg_path, avg_path, combined_path)`` from config.

    - ``noavg_path``: per-probe no-averaged groups (``.proc_noAvg.nc``)
    - ``avg_path``: per-probe averaged/binned groups (``.proc_Avg.nc``)
    - ``combined_path``: combined groups with probe dimension (``.proc.nc``)
    """
    noavg = Path(p) if (p := cfg_out.get("not_joined_db_path")) else None
    avg = Path(p) if (p := cfg_out.get("avg_db_path")) else None
    combined = Path(p) if (p := cfg_out.get("db_path")) else None
    return noavg, avg, combined


def _text_date_fmt(cfg_out, bin_s: int) -> str:
    """``text_date_format`` with ``.%f`` stripped for integer-second bins > 0."""
    fmt = cfg_out.get("text_date_format", "%Y-%m-%d %H:%M:%S.%f")
    if bin_s > 0 and isinstance(fmt, str) and fmt.endswith(".%f"):
        return fmt[: -len(".%f")]
    return fmt


def _build_filter_params_text(cfg_in: dict, cfg_filter: dict, coefs: dict | None = None) -> str:
    """Build sorted text of resolved filter + window + coefficients for re-run warning.

    Used as ``_run_params`` attr on processed NC groups: on skip, stored
    text is diff-compared to current — mismatch triggers a unified diff
    warning so users know filter / coefficients / window changed.

    *cfg_in* is ``cfg["input"]`` (plain dict, post-:func:`main_init`).
    *coefs* is the prepared coefficients dict (post-:func:`prepare_coefs`);
    keys with array values render via ``np.array2string``.
    """
    params: dict[str, str] = {}
    # Filter params (process-stage NaN-out)
    if cfg_filter:
        for lim in ("min", "max"):
            if cfg_filter.get(lim):
                for k, v in sorted(cfg_filter[lim].items()):
                    params[f"filter.{lim}.{k}"] = str(v)
        if cfg_filter.get("bad_p_at_bursts_starts_period"):
            params["filter.bad_p_at_bursts_starts_period"] = cfg_filter["bad_p_at_bursts_starts_period"]
    # Window params (load-stage)
    if (tr := cfg_in.get("time_ranges")) and len(tr) >= 2:
        params["input.time_ranges"] = f"[{tr[0]}, {tr[1]}]"
    for lim in ("min", "max"):
        if cfg_in.get(lim) and isinstance(cfg_in[lim], dict):
            for k, v in sorted(cfg_in[lim].items()):
                params[f"input.{lim}.{k}"] = str(v)
    # dt_min_binning_proc affects binning → include for reproducibility.
    # After main_init/ini2dict the '_s' suffix is stripped by type_fix.
    if dt_bp := cfg_in.get("dt_min_binning_proc"):
        params["input.dt_min_binning_proc"] = str(dt_bp)
    # Coefficients (sorted keys, ndarray rendered via %g-format for stable diff
    # across magnitudes — handles very large and very small values uniformly)
    _G_FMT = "{:.8g}".format
    if coefs:
        _SKIP_COEF_KEYS = frozenset({"dates", "Rz"})
        for k, v in sorted(coefs.items()):
            if k in _SKIP_COEF_KEYS:
                continue
            if isinstance(v, np.ndarray):
                params[f"coef.{k}"] = np.array2string(
                    v.astype(float), separator=", ", formatter={"float_kind": _G_FMT}
                )
            elif isinstance(v, (int, float, np.floating, np.integer)):
                params[f"coef.{k}"] = _G_FMT(float(v))
            else:
                params[f"coef.{k}"] = str(v)

    # Sort by key for stable diff
    return "\n".join(f"{k}={params[k]}" for k in sorted(params))


# ---------------------------------------------------------------------------
# Trim fast-path helpers — used by run_processing for overwrite_db="trim"
# and by _export_tsv_from_nc for overwrite_db=False.
# ---------------------------------------------------------------------------


def _read_run_params(nc_path: str | Path, tbl: str) -> str:
    """Read the latest stored param_spans entry from *tbl* in *nc_path*.

    Returns ``""`` when the file/group is missing.
    """
    from tcm._xr.store_params import get_latest_params, read_param_spans

    return get_latest_params(read_param_spans(nc_path, tbl))


def _time_ranges_in_nc(nc_path: str | Path, tbl: str, time_ranges: list) -> bool:
    """Check whether *time_ranges* is a subset of the existing NC time extent.

    Returns ``True`` when the NC group's time range fully covers *time_ranges*
    (i.e. no data outside the stored range is requested).  Used to decide
    whether a trim fast-path is safe.
    """
    if not time_ranges or len(time_ranges) < 2:
        return True
    nc_path = Path(nc_path)
    if not nc_path.exists():
        return False
    try:
        with _constants._h5py.File(str(nc_path), "r") as f:
            if tbl not in f or "time" not in f[tbl]:
                return False
            td = f[tbl]["time"]
            if td.shape[0] == 0:
                return False
            ex_min, ex_max = float(td[0]), float(td[-1])
    except (OSError, KeyError):
        return False
    # Convert time_ranges endpoints to CF float64 seconds for comparison
    _EPOCH_NS = np.datetime64("1970-01-01", "ns").astype(np.int64)
    tr_min = (
        (np.datetime64(time_ranges[0], "ns").astype(np.int64) - _EPOCH_NS) / 1e9 if time_ranges[0] else ex_min
    )
    tr_max = (
        (np.datetime64(time_ranges[-1], "ns").astype(np.int64) - _EPOCH_NS) / 1e9
        if time_ranges[-1]
        else ex_max
    )
    return bool(tr_min >= ex_min and tr_max <= ex_max)


def _load_raw_nc_if_covered(
    raw_nc: str | Path,
    tbl: str,
    cfg_in: dict,
    *,
    source_path: Path | None = None,
) -> tuple[xr.Dataset | None, dict | None] | None:
    """Load ``ds_raw`` + coefs from ``*.raw.nc`` when it covers ``time_ranges``.

    Returns ``(ds_raw, coefs_from_file)`` on success, or ``None`` when
    the fast-path cannot be taken (file missing, group absent, range not
    covered, log mismatch, or empty result).

    When *source_path* is given and the file does **not** exist on disk,
    verifies that the NC's ``/{tbl}/logFiles`` contains a ``fileName``
    entry matching the source file — prevents loading from an NC built
    from a different file with an overlapping time range.
    """
    if not policy.io():
        return None
    nc_path = Path(raw_nc)
    if not nc_path.exists():
        return None
    if not _time_ranges_in_nc(nc_path, tbl, cfg_in.get("time_ranges")):
        return None
    # When source file is absent, verify NC log provenance
    if (
        source_path is not None
        and not source_path.exists()
        and not _source_file_in_nc_log(nc_path, tbl, source_path)
    ):
        lf.debug("Raw NC log has no entry for {} — skipping fast-path", source_path.name)
        return None
    lf.info("Raw NC covers time_ranges — loading from {}", nc_path.name)
    ds, coefs = xr_io.load_raw(path=nc_path, tbl=tbl, cfg_in=cfg_in)
    if ds is None or ds.sizes.get("time", 0) == 0:
        lf.warning("Raw NC {} group {} is empty after time filter — falling back to text", nc_path.name, tbl)
        return None
    return ds, coefs


def _source_file_in_nc_log(nc_path: Path, tbl: str, source_path: Path) -> bool:
    """Check if ``/{tbl}/logFiles`` in *nc_path* has a ``fileName`` matching *source_path*.

    The log ``fileName`` format is ``{parent_name}/{stem}`` (first 255 chars),
    matching :func:`tcm.h5.file_name_and_time_to_record`.
    """
    log = storage.read_nc_log(nc_path, tbl)
    if log.sizes.get("Date0", 0) == 0:
        return False
    expected = f"{source_path.parent.name}/{source_path.stem}"[-255:]
    return bool((log["fileName"].values == expected).any())


def _trim_all_nc(cfg: dict, pcid: str, time_ranges: list) -> None:
    """Trim all NC output files (raw, noavg, all bins) to *time_ranges*.

    Operates on the same NC files that :func:`_process_and_persist` writes to.
    """
    tr_start = np.datetime64(time_ranges[0]) if time_ranges[0] else None
    tr_end = np.datetime64(time_ranges[-1]) if time_ranges[-1] else None
    cfg_out = cfg["out"]

    # raw NC
    if raw_nc := cfg_out.get("raw_db_path"):
        tbl = format.pcid_to_raw_name(pcid)
        storage.trim_group_to_range(raw_nc, tbl, tr_start, tr_end)

    # proc_noAvg NC
    noavg_path, avg_path, _ = _output_nc_paths(cfg_out)
    if noavg_path:
        storage.trim_group_to_range(noavg_path, pcid, tr_start, tr_end)

    # proc NC (all bins)
    if avg_path:
        for dt_bin in _dt_bins(cfg_out):
            bin_s = int(dt_bin.total_seconds())
            if bin_s > 0:
                storage.trim_group_to_range(avg_path, f"{pcid}bin{bin_s}s", tr_start, tr_end)


def _export_tsv_from_nc(cfg: dict, pcid: str) -> None:
    """Re-export TSV files from NC data after trim fast-path.

    Reads each processed NC group and writes TSV — used when data was trimmed
    but not reprocessed, so the in-memory ``ds_out`` is stale.
    """
    cfg_out = cfg["out"]
    text_path = cfg_out.get("text_path")
    if not text_path:
        return
    text_columns = cfg_out.get("text_columns") or []
    split_period = cfg_out.get("split_period") or None
    dt_min_save = _dt_min_save(cfg_out)
    noavg_path, avg_path, _ = _output_nc_paths(cfg_out)

    for dt_bin in _dt_bins(cfg_out):
        bin_s = int(dt_bin.total_seconds())
        if dt_bin < dt_min_save:
            continue
        nc_path = avg_path if bin_s > 0 and avg_path else noavg_path
        if not nc_path:
            continue
        group = f"{pcid}bin{bin_s}s" if bin_s > 0 else pcid
        try:
            ds_tsv = xr.open_dataset(nc_path, group=group, engine=_constants.nc_engine)
        except (OSError, KeyError):
            lf.debug("Trim TSV export: group {} not found in {}", group, nc_path.name)
            continue
        if ds_tsv.sizes.get("time", 0) == 0:
            ds_tsv.close()
            continue
        fmt = _text_date_fmt(cfg_out, bin_s)
        suffix_csv = f"bin{bin_s}s" if bin_s else ""
        ts = datetime.fromtimestamp(int(ds_tsv["time"].values[0]) // 1_000_000_000, timezone.utc).strftime(
            "%y%m%d_%H%M"
        )
        csv_name = f"{ts}{suffix_csv}@{pcid}.tsv"
        csv_out = Path(text_path) / csv_name
        xr_io.ds_to_csv(
            physical.add_vabs_vdir(ds_tsv),
            csv_out,
            split_period=split_period,
            text_date_format=fmt,
            text_columns=text_columns or None,
        )
        ds_tsv.close()
        lf.info("Re-exported TSV {} after trim", csv_out.name)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run(cfg: DictConfig) -> tuple[list[str], list[str], DictConfig | None, list[tuple[str, str, Any]]] | None:
    """
    Canonical pipeline: discover → generate configs → process.

    :param cfg: Hydra-composed top-level configuration (from ``@hydra.main``),
    cfg.input.path - input data path:
    - **Text ** (``.txt/.csv/.tsv``) — discovery sweep:
        1. Generate missing/stale YAML configs via :func:`config_yaml.save_config_to_yaml`.
        2. Sync ``time_ranges`` from device metadata into run YAMLs (idempotent:
        configs with existing ``input.time_ranges`` are skipped).
        3. Filter by ``input.ids``, ``yaml_path`` (YAML stem glob), and
        ``input.path`` (matches stored YAML's ``input.path`` filename).
        4. Dispatch one :func:`run_processing` per YAML via
        :func:`cli.process_loading_yaml`.
        5. Log completion summary with ok/skipped/failed counts.

    - **Binary** (``.nc/.h5``) — direct dispatch, no config sweep:
        Binary formats carry their own coefficients; skip config search/generation/sync.
        Calls :func:`run_processing` once per ``input.tables`` entry (a single
        NC/HDF5 may hold several probes). Each call pins ``tables=[tbl]`` so
        :func:`run_processing` derives the correct pcid/output group.  Use
        ``cli.call_in_raw_dir(processing.run, input={...}, yaml_path=...)`` to pass
        an explicit per-probe YAML override if needed.

    All Hydra groups (input, out, filter, program) are fully resolved — no MISSING sentinels.
    Per-probe YAMLs use ``@package _global_`` and are merged on top.

    :returns: ``(processed_pcids, failed_pcids, last_cfg, collected)`` after
    processing, or ``None`` for binary inputs. Where:
    ``collected`` is  ``[(stem, yaml_path_str, result), ...]`` — populated only when
    ``program.return_ == Return.CFG_FROM_ARGS`` (early exit before data load);
    ``result`` is the **full** per-probe config (all fields, not just non-defaults) — unlike YAMLs on disk
    which strip defaults.

    See also: :doc:`how_it_works </tcm_clc/how_it_works>`, :doc:`config_reference
    </tcm_clc/config_reference>`.
    """
    if (path_in := cfg.input.path) is None:
        raise ValueError("cfg.input.path must be provided")
    path_in = Path(path_in).absolute()

    # Binary: NC/HDF5 carry their own coefs — direct dispatch, no config sweep.
    if path_in.suffix.lower() in _EXT_BINARY:
        # One probe per table group (a single NC/HDF5 may hold several);
        # pin ``tables=[tbl]`` per call so run_processing derives correct pcid.
        tables = list(cfg.input.tables or [])
        for tbl in tables or [""]:
            cfg_pc = OmegaConf.merge(cfg, OmegaConf.create({"input": {"tables": [tbl]}})) if tables else cfg
            run_processing(cfg_pc)
        lf.info(
            "Done — processed {} table{} from {}",
            len(tables),
            "" if len(tables) == 1 else "s",
            path_in.name,
        )
        return

    ids = list(cfg.input.ids) if cfg.input.ids else None
    pcids_requested = format.normalize_probes(set(ids)) if ids else {format.PROBE_WILDCARD}

    dir_raw = paths.find_dir_raw_absolute(path_in)
    dir_cfgs = dir_raw / "cfg_proc" / "run"
    cli.safe_cfg_dir(dir_cfgs)
    cfgs_existed = config_yaml.get_existed_cfgs(dir_cfgs)

    # Scan progress — update progress_stage so the GUI overlay shows activity.
    _rt = progress_bridge.get_runtime()

    # ── Config generation (skipped when yaml_path provided) ──────────────
    if (yaml_path := OmegaConf.select(cfg, "input.yaml_path", default=None)) is None:
        # Step 1: regenerate on stale configs OR new source files missing configs.
        stale = config_yaml.find_stale_cfgs(cfgs_existed, dir_cfgs)
        regenerate = bool(stale) or not cfgs_existed
        if not regenerate:
            if _rt:
                _rt.progress_stage.set(0, 3, "Discovering files\u2026")
            try:  # lightweight: check for source files lacking config
                from tcm import csv_load

                discovered = csv_load.search_csv_files(path_in)
                disc_pcids = {format.pcid_from_parts(model=m, number=n) for m, n in discovered}
                new_pcids = disc_pcids - set(cfgs_existed)
                if new_pcids and (pcids_requested == {format.PROBE_WILDCARD} or pcids_requested & new_pcids):
                    lf.info("Source files without configs: {} — will generate", new_pcids)
                    regenerate = True
            except (FileNotFoundError, OSError):
                pass  # discovery fails → skip regeneration check
        if regenerate:
            if _rt:
                _rt.progress_stage.set(1, 3, "Generating configs\u2026")
            reason = (
                f"regenerating {len(stale)} stale config(s): {', '.join(stale)}"
                if stale
                else "no configs exist — generating from scratch"
                if not cfgs_existed
                else "new source files found"
            )
            lf.info("Config generation: {}", reason)
            config_yaml.save_config_to_yaml(cfg, [path_in])
            cfgs_existed = config_yaml.get_existed_cfgs(dir_cfgs)
        # Sync time_ranges (idempotent: configs with existing ranges are skipped).
        config_yaml.sync_yamls_devmeta_and_hydra(dir_raw.parent, dir_cfgs, cfgs_existed)
        # Warn about orphan configs pointing to non-existing files.
        still_stale = config_yaml.find_stale_cfgs(cfgs_existed, dir_cfgs) if stale else {}
        if still_stale:
            stale_pcids = set(still_stale)
            ignored = stale_pcids - pcids_requested if pcids_requested != {format.PROBE_WILDCARD} else set()
            actionable = stale_pcids - ignored
            parts = []
            if actionable:
                stale_details = "; ".join(
                    f"{pcid}: {', '.join(f'{s}.yaml' for s in stems)}"
                    for pcid, stems in still_stale.items()
                    if pcid in actionable
                )
                parts.append(f"stale config(s): {stale_details}")
            if ignored:
                parts.append(f"not in input.ids: {ignored} — will be skipped")
            lf.warning(
                "Orphan configs (input.path points to not existing file): {} — ignored!",
                "; ".join(parts),
            )
    else:
        stale = {}
        still_stale = {}

    # Step 2: select configs to run by requested pcids.
    if pcids_requested == {format.PROBE_WILDCARD}:
        cfgs_to_run = cfgs_existed
    elif pcids_requested:
        if not_found := pcids_requested - set(cfgs_existed):
            raise ValueError(
                f"Requested probes have no configs: {not_found}. Available: {sorted(cfgs_existed)}"
            )
        cfgs_to_run = {k: v for k, v in cfgs_existed.items() if k in pcids_requested}
    else:
        cfgs_to_run = {}

    # Drop stale stems so run_processing never sees them.
    if still_stale:
        cfgs_to_run = {
            pcid: [s for s in stems if s not in still_stale.get(pcid, [])]
            for pcid, stems in cfgs_to_run.items()
        }
        cfgs_to_run = {pcid: stems for pcid, stems in cfgs_to_run.items() if stems}

    # ── Filter: yaml_path pattern ↔ YAML stem ───────────────────────────
    from tcm.csv_load import _pattern_to_regex as _ptr

    if yaml_path:
        _yp_re = re.compile(_ptr(yaml_path), re.IGNORECASE)
        cfgs_to_run = {
            pcid: [s for s in stems if _yp_re.fullmatch(s) or _yp_re.fullmatch(f"{s}.yaml")]
            for pcid, stems in cfgs_to_run.items()
        }
        cfgs_to_run = {k: v for k, v in cfgs_to_run.items() if v}
        lf.debug("yaml_path filter '{}' → {} probes", yaml_path, len(cfgs_to_run))

    # ── Filter: input.path pattern ↔ stored YAML input.path ─────────────
    # A concrete file / glob narrows YAMLs whose stored input.path filename
    # matches; uses ruamel YAML directly — lightweight read, avoids
    # OmegaConf.load's full resolution overhead.
    if not path_in.is_dir():
        _ip_re = re.compile(_ptr(path_in.name), re.IGNORECASE)
        _ry = config_yaml._ry(write=False)
        filtered = {}
        for pcid, stems in cfgs_to_run.items():
            matched = []
            for s in stems:
                with contextlib.suppress(Exception):
                    y = _ry.load((dir_cfgs / f"{s}.yaml").open(encoding="utf-8"))
                    ip = (y or {}).get("input", {}).get("path", "")
                    if ip and _ip_re.fullmatch(Path(ip).name):
                        matched.append(s)
            if matched:
                filtered[pcid] = matched
        if filtered != cfgs_to_run:
            lf.debug(
                "input.path filter '{}' → {} probes (was {})",
                path_in.name,
                len(filtered),
                len(cfgs_to_run),
            )
            cfgs_to_run = filtered

    # Step 3: process each config; early-exit (CFG_FROM_ARGS) returns before data load.
    n_cfgs_total = sum(len(s) for s in cfgs_to_run.values())
    if _rt:
        _rt.progress_stage.set(0, max(n_cfgs_total, 1), "Composing configs\u2026")
    processed_pcids, failed_pcids, last_cfg, collected = cli.process_loading_yaml(
        run_processing,
        base_cfg=cfg,
        dir_cfgs=dir_cfgs,
        cfgs=cfgs_to_run,
        n_cfgs_existed=len(cfgs_existed),
    )
    if cfg["program"]["return_"] == schema.Return.CFG_FROM_ARGS:
        return processed_pcids, failed_pcids, last_cfg, collected

    # Combine distinct probes (legacy parity). Requires HDF5/netCDF4 backend.
    distinct_pcids = list(dict.fromkeys(processed_pcids))
    if len(distinct_pcids) > 1 and last_cfg is not None and policy.io():
        if not _constants.NC4_AVAILABLE:
            lf.debug("Combine skipped — netCDF4 not available")
        else:
            stage_ctx.set_stage(0, Stage.COMBINE, "Combining %d probes", len(distinct_pcids))
            _combine_probes(distinct_pcids, last_cfg)

    lf.debug(
        "Run stats: cfgs_existed={}, cfgs_to_run={}, processed={}, failed={}, distinct={}, last_cfg={}",
        len(cfgs_existed),
        len(cfgs_to_run),
        processed_pcids,
        failed_pcids,
        distinct_pcids,
        OmegaConf.select(last_cfg, "_yaml_path") if last_cfg else None,
    )
    # A probe failed on one YAML but succeeded on another = success (config, not data, failed).
    truly_failed = sorted(set(failed_pcids) - set(processed_pcids))
    skipped = (
        sorted(pcids_requested - set(cfgs_existed)) if pcids_requested != {format.PROBE_WILDCARD} else []
    )
    parts = [
        *([f"{len(distinct_pcids)} probes: {', '.join(distinct_pcids)} ok"] if distinct_pcids else []),
        *([f"{len(skipped)} skipped ({', '.join(skipped)})"] if skipped else []),
        *([f"{len(truly_failed)} failed ({', '.join(truly_failed)})"] if truly_failed else []),
    ]
    if (return_ := OmegaConf.select(cfg, "program.return_", default=None)) and return_ != schema.Return.END:
        parts.append(f"return_={return_}")
    lf.info("Done — {}", " | ".join(parts) if parts else "nothing processed")
    stage_ctx.clear()
    return processed_pcids, failed_pcids, last_cfg, collected


# ---------------------------------------------------------------------------
# Single-file processing
# ---------------------------------------------------------------------------


def run_processing(cfg: DictConfig):
    """Process one run YAML — single file or batch.

    Derives probe identity from ``input.path`` filename (text CSV) or, for
    binary inputs (NC/HDF5), from ``input.tables[0]`` (explicit table group
    pinned per call by :func:`run`).
    Resolves coefs: ``coefs_path`` → ``input.coefs`` (highest priority).
    Resolves output paths via :class:`paths.PathLayout`.
    Streams chunks, applies physical conversion + binning, persists (NC + CSV).

    Returns the merged ``DictConfig`` for early-exit modes; ``None`` for
    normal processing.
    """
    src_path = Path(cfg.input.path)
    if src_path.suffix.lower() in _EXT_BINARY and (tables := list(cfg.input.tables or [])):
        # Binary: pcid ← explicit table name (e.g. "incl_p05" → "i_p05").
        # Fall back to path-stem inference when tables is only the default
        # glob ["incl*"] — avoids pcid='*' in output groups.
        if tables == ["incl*"]:
            pcid = format.to_pcid_from_name(format.stem_to_pcid(src_path.stem))
            tbl = format.pcid_to_raw_name(pcid)
        else:
            pcid = format.to_pcid_from_name(tables[0])
            tbl = tables[0]
    else:
        # Text: pcid ← path stem (1 CSV file = 1 probe, legacy convention)
        pcid = format.to_pcid_from_name(format.stem_to_pcid(src_path.stem))
        tbl = format.pcid_to_raw_name(pcid)

    lf.debug("Loading {} data for processing...", pcid)
    # Set per-config progress base (injected by process_loading_yaml).
    # Extract early for logging context — before main_init for minimal latency.
    if (si := cfg.get("_stem_idx")) and (nc := cfg.get("_n_cfgs")):
        stage_ctx.set_probe(pcid, stem_idx=si, n_cfgs_total=nc)
    cfg = cli.main_init(cfg)
    # Early-exit: main_init returns DictConfig before ini2dict; propagate upstream.
    if not isinstance(cfg, dict):
        return cfg
    cfg_in = cfg["input"]  # already type-converted plain dict after main_init

    # Active stage plan for upper-bar ticks (NC only if io().h5; TSV if text_path).
    # Each active stage gets an equal slice of the 100-unit per-probe scale.
    _dt_bins_list = _dt_bins(cfg["out"])
    _n_bins = len(_dt_bins_list)
    _has_nc = bool(policy.io())
    _has_tsv = bool(cfg["out"].get("text_path"))
    _stages = [Stage.LOAD, Stage.COEFS, Stage.PROC]
    _stages += [Stage.NC] * _n_bins if _has_nc else []
    _stages += [Stage.TSV] * _n_bins if _has_tsv else []
    _n_active = len(_stages)
    stage_ctx.set_stage_plan(_n_active)

    # Load begins — boundary record carries data source for clarity
    _src = cfg_in.get("path", "")
    stage_ctx.set_stage(1, Stage.LOAD, "Loading %s", Path(_src).name if _src else pcid)

    # Batch mode (cfg.files exists): iterate and concatenate
    _loaded_from_raw_nc = False  # track fast-path for Phase 4 skip
    if cfg.get("files"):
        ds_raw, coefs_from_file = _load_batch(cfg, pcid)
    else:
        # Single-file mode — try raw NC fast-path first for text sources:
        # if *.raw.nc already covers time_ranges, skip text parsing entirely.
        _fast_path: tuple | None = None
        if src_path.suffix.lower() not in _EXT_BINARY and (raw_nc := cfg["out"].get("raw_db_path")):
            _fast_path = _load_raw_nc_if_covered(raw_nc, tbl, cfg_in, source_path=src_path)
        if _fast_path is not None:
            ds_raw, coefs_from_file = _fast_path
            _loaded_from_raw_nc = True
        else:
            ds_raw, coefs_from_file = xr_io.load_raw(
                tbl=tbl,
                text_type=pcid[:1] if pcid else "i",
                cfg_in=cfg_in,
            )
    stage_ctx.tick()  # load done

    # Coefs: coefs_path (file) → input.coefs (run YAML override wins)
    stage_ctx.set_stage(2, Stage.COEFS)
    coefs = get_coefs_from_cfg(cfg_in, pcid)
    if coefs_from_file:
        coefs = {**coefs, **{k: v for k, v in coefs_from_file.items() if v is not None}}
        lf.debug("Merged coefs from data file: {} extra keys", len(coefs_from_file))

    # ── Phase 1b: extract coefs from .raw.h5 if .raw.nc absent (legacy HDF5 auto-migrate)
    if (raw_nc_path := cfg["out"].get("raw_db_path")) and policy.io():
        raw_nc_path = Path(raw_nc_path)
        if not raw_nc_path.exists():
            if (h5_path := raw_nc_path.with_suffix("").with_suffix(".raw.h5")).exists():
                from tcm.incl_calc.coefs import load_coefs

                if h5_coefs := load_coefs(h5_path, tbl):
                    coefs = {**h5_coefs, **{k: v for k, v in coefs.items() if v is not None}}
                    lf.info("Auto-migrate: extracted coefs from {}", h5_path)

    # Prepare coefs: zeroing rotation, azimuth correction
    lf.debug("Preparing coefs for {}...", pcid)

    coefs_merged, coef_zeroing_matrix, dates, msg = xr_coefs.prepare_coefs(
        coefs,
        ds_raw,
        time_ranges_zeroing=cfg_in.get("time_ranges_zeroing") or None,
        time_ranges_azimuth=cfg_in.get("time_ranges_azimuth") or None,
        azimuth_add=cfg_in.get("azimuth_add") or None,
        coordinates=tuple(cfg_in["coordinates"]) if cfg_in.get("coordinates") else None,
    )
    lf.info("Coefs prepared for {}: {}", pcid, msg or "no zeroing/azimuth adjustments")
    stage_ctx.tick()  # coefs done

    # Compute run params text once — used by incremental skip, trim fast-path,
    # and passed to _process_and_persist for storage attrs.
    cfg_filter = cfg.get("filter") or {}
    run_params_text = _build_filter_params_text(cfg_in, cfg_filter, coefs=coefs_merged)
    overwrite_db = cfg["out"].get("overwrite_db")

    # ── Phase 3: Save coefs
    # Two triggers: (a) coefs changed (zeroing/azimuth), (b) raw NC being created for the first time.
    # NC sources: always write back changed coefs to source file.
    # CSV/HDF5 sources: write all coefs on first creation, or changed coefs on re-run.
    # YAML is ALWAYS updated when yaml_path exists and coefs changed (not just noh5 fallback).
    changed_coefs = {k for k, v in dates.items() if v is True}
    yaml_path = cfg.get("_yaml_path")
    coefs_to_write: dict | None = None  # filled only when write is needed
    yaml_written = False  # track whether YAML was the primary write target

    if src_path.suffix.lower() in _EXT_BINARY:
        if changed_coefs and policy.io():
            # Release xr's read-only netCDF4/HDF5 file handle before h5py opens in
            # append mode.  Only materialise when coefs actually changed — avoids
            # unnecessary memory pressure on large files.
            if ds_raw is not None:
                ds_raw.load()
                ds_raw.close()
            try:
                xr_coefs.save_coefs_to_nc(src_path, tbl, coefs_merged, pcid=pcid, dates=dates)
                lf.info("Overwrote coefs {} in {}", sorted(changed_coefs), src_path.name)
            except OSError:
                lf.warning(
                    "Could not write coefs to {} (file locked) — coefficients may "
                    "be stale; close other readers and re-run",
                    src_path.name,
                )
        elif changed_coefs and yaml_path:
            config_yaml.update_coefs_in_run_yaml(
                yaml_path,
                {k: coefs_merged[k] for k in changed_coefs},
            )
            yaml_written = True
        elif changed_coefs:
            lf.warning("Coefs changed ({}) but no write target available", sorted(changed_coefs))
        else:
            lf.debug("Coefs unchanged for {} — skipping write", pcid)
    elif policy.io() and (raw_nc := cfg["out"].get("raw_db_path")):
        # CSV source + H5: write coefs to raw_db_path.
        # Multiple probes may share one *.raw.nc — each needs its own
        # /{tbl}/coef/ group.  save_coefs_to_nc (called AFTER Phase 4)
        # is idempotent: h5copy_coef handles both create and overwrite.
        if _loaded_from_raw_nc:
            # Autoload fast-path: coefs group already exists in NC —
            # only overwrite when prepare_coefs detected a difference.
            if changed_coefs:
                coefs_to_write = coefs_merged
            else:
                lf.debug("Coefs unchanged for {} — skipping write", pcid)
        else:
            # CSV text → NC: always write (group may not exist yet).
            coefs_to_write = coefs_merged
    elif yaml_path and changed_coefs:
        # noh5 fallback: write only changed coefs to run YAML
        config_yaml.update_coefs_in_run_yaml(
            yaml_path,
            {k: coefs_merged[k] for k in changed_coefs},
        )
        yaml_written = True
    elif changed_coefs:
        lf.warning("Coefs changed ({}) but no write target available", sorted(changed_coefs))
    else:
        lf.debug("Coefs unchanged for {} — skipping write", pcid)

    # Always mirror changed coefs to YAML when available (keeps config readable).
    # Skip if YAML was already the primary write target above.
    if changed_coefs and yaml_path and not yaml_written:
        config_yaml.update_coefs_in_run_yaml(
            yaml_path,
            {k: coefs_merged[k] for k in changed_coefs},
        )

    # ── Phase 4: Save raw data (skip for NC sources — data already there,
    #   or when loaded from *.raw.nc fast-path — covers time_ranges already)
    if src_path.suffix.lower() not in _EXT_BINARY and ds_raw is not None and not _loaded_from_raw_nc:
        raw_nc_path = cfg["out"].get("raw_db_path")
        if raw_nc_path:
            # Release xarray read handle so h5py can open the same file.
            # On Windows, HDF5 uses mandatory locking — any open handle
            # (even read-only) blocks new opens (read or write).
            if ds_raw is not None:
                ds_raw.load()
                ds_raw.close()
            # Mirror tcm.h5.file_name_and_time_to_record (no tables dependency):
            file_meta = {
                "fileName": f"{src_path.parent.name}/{src_path.stem}"[-255:],
                "fileChangeTime": datetime.fromtimestamp(src_path.stat().st_mtime),
            }
            storage.nc_incremental_update(ds_raw, Path(raw_nc_path), tbl, file_meta)

    # Write coefs after data (raw NC may have been created by Phase 4)
    if coefs_to_write is not None and (raw_nc := cfg["out"].get("raw_db_path")):
        if _loaded_from_raw_nc and ds_raw is not None:
            # Autoload fast-path with changed coefs: ds_raw still holds a
            # netCDF4 read handle on the same NC file.  close() releases the
            # file lock without materialising all data into memory (unlike
            # load()+close()).  save_coefs_to_nc writes via h5py, then we
            # reopen from the same file for _process_and_persist.
            ds_raw.close()
        xr_coefs.save_coefs_to_nc(Path(raw_nc), tbl, coefs_to_write, pcid=pcid, dates=dates)
        # save_coefs_to_nc logs "Coefs saved to ..."
        if _loaded_from_raw_nc:
            # Reopen after h5py released its handle — restores lazy access
            # for _process_and_persist without full materialisation.
            ds_raw, _ = xr_io.load_raw(path=Path(raw_nc), tbl=tbl, cfg_in=cfg_in)

    # Phase-stopping: stop after coefs saved or raw data saved (before processing).
    if (return_ := cfg["program"]["return_"]) in (schema.Return.SAVED_COEFS, schema.Return.SAVED_RAW):
        lf.info("return_={} — stopping {} now", return_, pcid)
        return

    # ── overwrite_db mode dispatch (export-only / trim / splice or extend)
    raw_nc_dispatch = cfg["out"].get("raw_db_path")
    tr_dispatch = cfg_in.get("time_ranges")

    if overwrite_db == "export":
        # Export-only: block all NC writes, just export TSV from existing data
        if raw_nc_dispatch:
            _export_tsv_from_nc(cfg, pcid)
            lf.info("Export-only (overwrite_db=export) TSV for {}", pcid)
        return

    if overwrite_db == "trim":
        if tr_dispatch and raw_nc_dispatch and _time_ranges_in_nc(raw_nc_dispatch, tbl, tr_dispatch):
            _trim_all_nc(cfg, pcid, tr_dispatch)
            _export_tsv_from_nc(cfg, pcid)
            lf.info("Trimmed {} to time_ranges (overwrite_db=trim)", pcid)
            return  # no reprocess
        if not tr_dispatch:
            lf.warning("overwrite_db=trim but no time_ranges — skipping {}", pcid)
            return
        # time_ranges extends existing → fall through to _process_and_persist (append new data)

    # overwrite_db == "splice" or None → fall through to _process_and_persist

    # ── tick adapter for _process_and_persist (PROC, NC×n_bins, TSV×n_bins)
    def _tick(stage: Stage, _bin_i: int = 0, _n_bin: int = 1) -> None:
        stage_ctx.tick(stage)

    # Processing begins — start decoration at DEBUG (result INFO from _process_and_persist follows)
    stage_ctx.set_stage(3, Stage.PROC)
    lf.debug("Processing %s (%d bins)...", pcid, _n_bins)
    _process_and_persist(
        ds_raw,
        coefs_merged,
        cfg,
        pcid,
        coef_zeroing_matrix=coef_zeroing_matrix,
        tick=_tick,
        has_nc=_has_nc,
        has_tsv=_has_tsv,
        run_params_text=run_params_text,
        overwrite_db=overwrite_db,
    )


def _load_batch(
    cfg: Dict[str, Dict[str, Any]], pcid: str
) -> tuple[Optional[xr.Dataset], Optional[Dict[str, Any]]]:
    """Iterate ``cfg.files``, load each, concatenate progressively to limit peak memory."""

    ds_raw = None
    for file_cfg in cfg["files"]:
        src_path = Path(file_cfg["path"])
        if not src_path.is_file():
            lf.warning("Batch file {} does not exist — skipping", src_path)
            continue

        for ds_chunk, _meta in dataset.open_csv_chunks(
            src_path,
            text_type=pcid[:1] if pcid else "i",
            cfg_in=cfg["input"],
        ):
            ds_raw = ds_chunk if ds_raw is None else xr.concat([ds_raw, ds_chunk], dim="time")

    if ds_raw is None:
        lf.warning("No data loaded for batch {} — skipping", pcid)
        return None, None

    coefs = get_coefs_from_cfg(cfg["input"], pcid)
    return ds_raw, coefs


def _process_and_persist(
    ds_raw,
    coefs: Mapping[str, Any],
    cfg: Mapping[str, Mapping[str, Any]],
    pcid: str,
    *,
    coef_zeroing_matrix: "np.ndarray | None" = None,
    tick: "Callable[[Stage, int, int], None] | None" = None,
    has_nc: bool = False,
    has_tsv: bool = False,
    run_params_text: str | None = None,
    overwrite_db: str | None = None,
) -> None:
    """Apply physical conversion + binning, persist results.

    Saves each bin result to netCDF using shared files with per-probe groups:
    - no-avg (dt_bin=0): writes to ``not_joined_db_path`` (``.proc_noAvg.nc``)
      with group ``/{pcid}/``
    - binned (dt_bin>0): writes to ``avg_db_path`` (``.proc_Avg.nc``)
      with group ``/{pcid}bin{bin_s}s/``

    Combined output (``db_path`` = ``.proc.nc``) is written later by
    ``_combine_probes()``.

    Exports CSV/TSV for bins ≥ ``dt_bins_min_save_text`` to ``text_path``.

    *cfg* is a plain ``dict`` (post-:func:`main_init`); use ``[]`` / ``.get()``
    access, never OmegaConf attribute access.
    """

    if ds_raw is None:
        return

    cfg_in = cfg["input"]
    cfg_out = cfg["out"]
    cfg_filter = cfg.get("filter") or {}
    dt_bins = _dt_bins(cfg_out)

    # M shorthand → Mx/My/Mz expansion in process-stage NaN-out dicts (shared helper)
    if cfg_filter:
        cli.sugar_expand_m(cfg_filter)

    # Build run_params text when not precomputed by caller (process_inmemory compat)
    if run_params_text is None:
        run_params_text = _build_filter_params_text(cfg_in, cfg_filter, coefs=coefs)

    # Merge calc params into coefs (calc_velocity receives them via **coefs)
    coefs_for_calc = {**coefs}
    if cv := cfg_in.get("calc_version"):
        coefs_for_calc["calc_version"] = cv
    if mi := cfg_in.get("max_incl_of_fit_deg"):
        coefs_for_calc["max_incl_of_fit_deg"] = mi

    results = physical.process(
        ds_raw,
        coefs=coefs_for_calc,
        coef_zeroing_matrix=coef_zeroing_matrix,
        cfg_filter=cfg_filter,
        dt_bins=dt_bins,
        pcid=pcid,
        dt_min_binning_proc=(
            v
            if isinstance(v := cfg_in.get("dt_min_binning_proc"), timedelta)
            else timedelta(seconds=int(v or 2))
        ),
    )
    if tick:
        tick(Stage.PROC)  # process done

    text_path = cfg_out.get("text_path")
    dt_min_save = _dt_min_save(cfg_out)
    split_period = cfg_out.get("split_period") or None
    text_columns = cfg_out.get("text_columns") or []

    # Resolve shared output files once
    noavg_path, avg_path, _combined_path = _output_nc_paths(cfg_out)
    return_ = cfg["program"]["return_"]

    for _bin_i, (ds_out, dt_bin) in enumerate(zip(results, dt_bins)):
        if ds_out is None:
            continue
        # Battery only meaningful in raw.nc — drop from all processed outputs
        if "Battery" in ds_out.data_vars:
            ds_out = ds_out.drop_vars("Battery")
        bin_s = int(dt_bin.total_seconds())

        # --- netCDF: shared files with per-probe groups ---
        # Wrap with TqdmCallback when ds_out still has dask arrays (triggers
        # .compute() inside to_netcdf) — gives task-level progress per bin.
        _is_dask = any(ds_out[v].chunks is not None for v in ds_out.data_vars)
        _nc_label = f"bin{bin_s}s" if bin_s else "noAvg"
        if tick and has_nc:
            tick(Stage.NC, _bin_i, len(dt_bins))  # NC stage start for this bin
        with (
            TqdmCallback(
                desc=f"[{pcid}] {_nc_label} NC write",
                leave=False,
                **({"tqdm_class": _tc} if (_tc := progress_bridge.get_tqdm_class()) else {}),
            )
            if _is_dask
            else contextlib.nullcontext()
        ):
            if bin_s == 0 and noavg_path:
                # no-avg → /{pcid}/ group in *.proc_noAvg.nc (incremental skip + run-params sig)
                storage.store_processed_incremental(
                    ds_out,
                    noavg_path,
                    group=pcid,
                    filter_params=run_params_text,
                    force_reprocess=overwrite_db == "splice",
                )
                # Phase-stopping: return after noAvg save
                if return_ == schema.Return.SAVED_NOAVG:
                    lf.info("return_={} — stopping after noAvg save for {}", return_, pcid)
                    return
            elif bin_s > 0 and avg_path:
                # binned → /{pcid}bin{bin_s}s/ group in *.proc.nc (incremental skip + run-params sig)
                storage.store_processed_incremental(
                    ds_out,
                    avg_path,
                    group=f"{pcid}bin{bin_s}s",
                    filter_params=run_params_text,
                    force_reprocess=overwrite_db == "splice",
                )
            else:
                # Fallback: no PathLayout resolved paths
                out_dir = Path(cfg_out.get("dir", "./out"))
                out_path = out_dir / f"{'@' + pcid + f'_bin{bin_s}s' if bin_s else '@' + pcid}.nc"
                storage.store_processed(ds_out, out_path)

            # --- CSV/TSV export ---
            if text_path and dt_bin >= dt_min_save:
                # Shorten date format for integer-second bins (strip .%f)
                fmt = _text_date_fmt(cfg_out, bin_s)

                suffix_csv = f"bin{bin_s}s" if bin_s else ""
                # TSV source: read from NC after splice when overwrite_db="splice"
                # (spliced data may include head/tail not in ds_out)
                if overwrite_db == "splice":
                    nc_tsv_path = avg_path if bin_s > 0 and avg_path else noavg_path
                    nc_tsv_group = f"{pcid}bin{bin_s}s" if bin_s > 0 else pcid
                    try:
                        ds_tsv = xr.open_dataset(nc_tsv_path, group=nc_tsv_group, engine=_constants.nc_engine)
                    except (OSError, KeyError):
                        ds_tsv = ds_out
                else:
                    ds_tsv = ds_out
                ts = datetime.fromtimestamp(
                    int(ds_tsv["time"].values[0]) // 1_000_000_000, timezone.utc
                ).strftime("%y%m%d_%H%M")
                csv_name = f"{ts}{suffix_csv}@{pcid}.tsv"
                csv_out = Path(text_path) / csv_name

                if not cfg_out.get("b_overwrite_text", True) and csv_out.exists():
                    lf.info("TSV exists, b_overwrite_text=False — skipping {}", csv_out.name)
                else:
                    if tick and has_tsv:
                        tick(Stage.TSV, _bin_i, len(dt_bins))  # TSV stage start for this bin
                    xr_io.ds_to_csv(
                        physical.add_vabs_vdir(ds_tsv),
                        csv_out,
                        split_period=split_period,
                        text_date_format=fmt,
                        text_columns=text_columns or None,
                    )
                    lf.info("Saved TSV {} to {}", pcid, csv_out.name)
                    if ds_tsv is not ds_out:
                        ds_tsv.close()

    # Phase-stopping: return after all NC writes (before combined output)
    if return_ == schema.Return.SAVED_ALL:
        lf.info("return_={} — stopping after all NC saves for {}", return_, pcid)
        return


def _combine_probes(pcids: list[str], cfg: dict) -> None:
    """Merge per-probe groups into combined output with probe dimension.

    Reads per-probe groups from proc_noAvg.nc and proc.nc, merges along
    a new ``probe`` dimension, writes combined groups. Also writes
    combined TSV with joined pcid suffix.

    Output naming:
    - NC noAvg (proc_noAvg.nc): group ``/{probe_type}/`` (probe dim)
    - NC binned (proc.nc): group ``/{probe_type}_bin{N}s/`` (probe dim)
    - TSV: ``{ts}bin{N}s@{joined}.tsv``

    *probe_type* is the short probe prefix derived from the first pcid
    (e.g. ``"i"`` for inclinometers, ``"w"`` for wave gauges). All probes
    combined in a single call must share the same type.
    """

    cfg_out = cfg["out"]
    _noavg_path, avg_path, combined_path = _output_nc_paths(cfg_out)
    if not combined_path:
        return

    text_path = cfg_out.get("text_path")
    text_columns = cfg_out.get("text_columns") or []
    dt_bins = _dt_bins(cfg_out)
    joined = ",".join(pcids)
    probe_type = pcids[0][0] if pcids else "i"
    if any(p[0] != probe_type for p in pcids):
        lf.debug("Combined probes have mixed types — using type '{}' from first pcid", probe_type)

    # Combine binned groups from proc_Avg.nc → proc.nc
    # (noAvg is not combined — per-probe only)
    if avg_path and avg_path.exists():
        storage.ensure_dim_scales(avg_path)
        for dt_bin in dt_bins:
            bin_s = int(dt_bin.total_seconds())
            if bin_s <= 0:
                continue
            combined_group = f"/{probe_type}_bin{bin_s}s/"
            _merge_groups_to_combined(
                avg_path, pcids, combined_group, f"bin{bin_s}s", bin_s=bin_s, combined_nc_path=combined_path
            )

    # Combined TSV (for each binned result only — no noAvg combined TSV)
    if text_path and combined_path:
        dt_min_save = _dt_min_save(cfg_out)
        for dt_bin in dt_bins:
            if dt_bin < dt_min_save:
                continue
            bin_s = int(dt_bin.total_seconds())
            if bin_s <= 0:
                continue
            combined_group = f"/{probe_type}_bin{bin_s}s/"
            try:
                ds_combined = xr.open_dataset(
                    combined_path, group=combined_group, engine=_constants.nc_engine
                )
            except (AttributeError, KeyError, OSError, ValueError):
                lf.debug("Combined group {} not found or malformed — skipping TSV", combined_group)
                continue

            ts = datetime.fromtimestamp(
                int(ds_combined["time"].values[0]) // 1_000_000_000, timezone.utc
            ).strftime("%y%m%d_%H%M")
            csv_name = f"{ts}bin{bin_s}s@{joined}.tsv"
            csv_out = Path(text_path) / csv_name
            # Combined TSV: exclude Vabs/Vdir/inclination (never saved to combined)
            _drop_combined = [v for v in ("Vabs", "Vdir", "inclination") if v in ds_combined]
            xr_io.ds_to_csv(
                ds_combined.drop_vars(_drop_combined) if _drop_combined else ds_combined,
                csv_out,
                text_date_format=_text_date_fmt(cfg_out, bin_s),
                text_columns=text_columns or None,
            )
            lf.info("Saved combined TSV {} to {}", joined, csv_out.name)
            ds_combined.close()


def _combined_group_is_current(
    combined_path: Path,
    per_probe_path: Path,
    combined_grp: str,
    pcids: list[str],
    bin_s: int,
) -> bool:
    """Quick metadata check: is the combined group readable and covering all per-probe ranges?

    Uses h5netcdf for readability (catches corrupted dimension scales) and
    h5py for time-range comparison (lightweight — no full data load).

    Parameters
    ----------
    combined_path
        File containing the combined group (e.g. ``.proc.nc``).
    per_probe_path
        File containing per-probe groups (e.g. ``.proc_Avg.nc``).
        May be the same as *combined_path* when both live in one file.
    combined_grp
        NetCDF group name of the combined output.
    pcids
        Probe identifiers whose per-probe groups must be covered.
    bin_s
        Bin interval in seconds — used to construct per-probe group names.
    """
    if not combined_path.exists():
        return False

    # 1. Readability check — h5netcdf validates dimension-scale metadata
    import gc

    try:
        ds = xr.open_dataset(combined_path, group=combined_grp, engine=_constants.nc_engine)
        ds.close()
        del ds
        gc.collect()
    except (ValueError, AttributeError, KeyError, OSError):
        return False

    # 2. Time-range coverage check via h5py (reads only time datasets).
    #    Combined group lives in *combined_path*; per-probe groups live in
    #    *per_probe_path* — they are typically different files.
    _h5py = _constants._h5py
    if _h5py is None:
        return True
    try:
        # Read combined time range from the combined file
        with _h5py.File(str(combined_path), "r") as f:
            if combined_grp not in f or "time" not in f[combined_grp]:
                return False
            td_c = f[combined_grp]["time"]
            if td_c.shape[0] == 0:
                return False
            c_vals = storage.cf_to_dt_ns(
                td_c[:],
                td_c.attrs.get("units", ""),
            ).astype(np.int64)
            combined_min, combined_max = c_vals[0], c_vals[-1]

        # Read per-probe time ranges from the per-probe source file
        with _h5py.File(str(per_probe_path), "r") as f:
            for pcid in pcids:
                grp_name = f"{pcid}bin{bin_s}s" if bin_s > 0 else pcid
                if grp_name not in f or "time" not in f[grp_name]:
                    return False
                td = f[grp_name]["time"]
                if td.shape[0] == 0:
                    return False
                t_vals = storage.cf_to_dt_ns(
                    td[:],
                    td.attrs.get("units", ""),
                ).astype(np.int64)
                if t_vals[0] < combined_min or t_vals[-1] > combined_max:
                    return False
    except (OSError, KeyError, AttributeError):
        return False

    return True


def _merge_groups_to_combined(
    nc_path: Path,
    pcids: list[str],
    combined_group: str,
    label: str,
    *,
    bin_s: int = 0,
    combined_nc_path: Path | None = None,
) -> None:
    """Read per-probe groups from *nc_path*, merge with probe dimension, write combined group.

    *combined_nc_path* (default: same as *nc_path*) is where the combined
    group is written — allows per-probe data to live in a different file
    (e.g. ``.proc_Avg.nc``) from the combined output (``.proc.nc``).
    """

    combined_grp = combined_group.strip("/")
    out_path = combined_nc_path or nc_path

    # Quick metadata check — avoids loading per-probe data when the combined
    # group is already valid.  Old groups with corrupted dimension scales will
    # fail the h5netcdf readability test → rewrite.
    if _combined_group_is_current(out_path, nc_path, combined_grp, pcids, bin_s):
        lf.debug("Combined group {} is current — skipping", combined_grp)
        return

    groups_to_merge = []
    for pcid in pcids:
        grp_name = f"{pcid}bin{bin_s}s" if bin_s > 0 else pcid
        try:
            ds = xr.open_dataset(nc_path, group=grp_name, engine=_constants.nc_engine)
            groups_to_merge.append(ds.expand_dims(probe=[pcid]))
        except (AttributeError, KeyError, OSError):
            lf.debug("Group {} not found in {} — skipping for combined {}", grp_name, nc_path.name, label)
            continue

    if len(groups_to_merge) < 2:
        for ds in groups_to_merge:
            ds.close()
        return

    combined = xr.concat(groups_to_merge, dim="probe", join="outer")
    for ds in groups_to_merge:
        ds.close()

    storage.delete_h5py_group(out_path, combined_grp)
    # Use to_netcdf directly — it handles multi-dimensional data with string
    # coordinates (like "probe") natively, producing correct dimension-scale
    # metadata.  _write_dataset_to_nc_group (h5py-only) can produce malformed
    # byte-string dimension scales that h5netcdf misinterprets on read.
    combined = tcm._xr.nc_utils.strip_tz_datetime(combined)
    combined = tcm._xr.nc_utils.downcast_float32(combined)
    enc = {**tcm._xr.nc_utils.force_epoch(combined), **tcm._xr.nc_utils.compression_encoding(combined)}
    combined.to_netcdf(out_path, group=combined_grp, mode="a", engine=_constants.nc_engine, encoding=enc)
    lf.info("Combined {} to {} (probe dim with {} probes)", label, out_path.name, len(groups_to_merge))


# ---------------------------------------------------------------------------
# Standalone in-memory API
# ---------------------------------------------------------------------------


def process_inmemory(
    ds: xr.Dataset,
    coefs: dict,
    *,
    coef_zeroing_matrix: "np.ndarray | None" = None,
    dt_bins: list[timedelta] | None = None,
    out_path: Path | None = None,
    out_csv_path: Path | None = None,
    split_period: str | None = None,
) -> list[xr.Dataset]:
    """Process a single Dataset in-memory and optionally persist.

    Pass-through to :func:`_xr.physical.process` with optional persistence.
    Kept for callers/tests that already have a Dataset in hand; the main
    pipeline entry point is :func:`run`.
    """

    if dt_bins is None:
        dt_bins = [timedelta(0)]

    results = physical.process(ds, coefs=coefs, coef_zeroing_matrix=coef_zeroing_matrix, dt_bins=dt_bins)

    for ds_out, dt_bin in zip(results, dt_bins):
        if ds_out is None:
            continue
        # Battery only meaningful in raw.nc — drop from all processed outputs
        if "Battery" in ds_out.data_vars:
            ds_out = ds_out.drop_vars("Battery")
        bin_s = int(dt_bin.total_seconds())
        lf.info(
            "Bin {}s: {} time steps, {} vars",
            bin_s,
            ds_out.sizes.get("time", 0),
            len(ds_out.data_vars),
        )
        if out_path is not None:
            nc_path = out_path.with_stem(f"{out_path.stem}_bin{bin_s}s") if bin_s else out_path
            storage.store_processed(ds_out, nc_path)
        if out_csv_path is not None:
            suffix = f"_bin{bin_s}s" if bin_s else ""
            csv_path = out_csv_path.with_stem(f"{out_csv_path.stem}{suffix}")
            xr_io.ds_to_csv(physical.add_vabs_vdir(ds_out), csv_path, split_period=split_period)

    return [r for r in results if r is not None]
