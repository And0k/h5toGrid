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
import xarray as xr
from omegaconf import DictConfig, OmegaConf
from tqdm.dask import TqdmCallback




from tcm import _constants, cli, config_yaml, format, paths, utils2init
from tcm._xr import coefs as xr_coefs
from tcm._xr import dataset, physical, storage
from tcm._xr import io as xr_io
from tcm.config import Return
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
# scale; active stages share it evenly (NC only when ``use_h5_get() is True``,
# TSV only when ``text_path`` set).  Bottom bar (dask TqdmCallback) covers
# substages continuously within a stage.


class Stage(StrEnum):
    """Per-probe phase labels (value = upper-bar description text)."""
    LOAD  = "load"   # xr_io.load_raw / _load_batch
    COEFS = "coefs"  # prepare_coefs + save
    PROC  = "proc"   # physical.process (calc + binning)
    NC    = "NC"     # store_processed_incremental (per bin), use_h5 only
    TSV   = "TSV"    # xr_io.ds_to_csv (per bin), text_path only


_probe_base = {"v": 0}    # set per probe by process_loading_yaml
_probe_total = {"v": 0}   # set once by process_loading_yaml (n_cfgs × 100)


def _stage(label: str, frac: int) -> None:
    """Tick the upper bar: ``set(base + frac, probe_total, label)``."""
    if rt := progress_bridge.get_runtime():
        rt.progress_overall.set(_probe_base["v"] + frac, _probe_total["v"], label)


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


def _output_nc_paths(cfg_out) -> tuple[Path | None, Path | None]:
    """Resolve ``(noavg_path, avg_path)`` from ``not_joined_db_path``."""
    njp = cfg_out.get("not_joined_db_path")
    if njp:
        noavg = Path(njp)
        avg = noavg.parent / noavg.name.replace("_noAvg", "", 1)
        return noavg, avg
    return None, None


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
    # Coefficients (sorted keys, ndarray rendered via np.array2string)
    if coefs:
        _SKIP_COEF_KEYS = frozenset({"dates", "Rz"})
        for k, v in sorted(coefs.items()):
            if k in _SKIP_COEF_KEYS:
                continue
            if isinstance(v, np.ndarray):
                params[f"coef.{k}"] = np.array2string(v.astype(float), separator=", ")
            else:
                params[f"coef.{k}"] = str(v)

    # Sort by key for stable diff
    return "\n".join(f"{k}={params[k]}" for k in sorted(params))


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _resolve_use_h5(cfg: DictConfig) -> None:
    """Resolve ``program.use_h5`` against library availability.

    Overwrites the config field **and** :data:`_constants._use_h5` so that
    both config-aware and config-unaware code sees the same resolved value.

    Resolution rules (``use_h5`` × h5py available):

    ==============  ==========  ==========================
    use_h5 (in)  h5py avail  result
    ==============  ==========  ==========================
    ``None``        yes         ``True``  (auto-enable)
    ``None``        no          ``None``  (silent skip)
    ``True``        yes         ``True``  (user confirmed)
    ``True``        no          ``False`` + **WARNING**
    ``False``       any         ``False`` (user disabled)
    ==============  ==========  ==========================
    """
    bio = OmegaConf.select(cfg, "program.use_h5", default=None)
    if bio is None:
        if _constants.H5_AVAILABLE:
            OmegaConf.update(cfg, "program.use_h5", True)
            _constants.use_h5_set(True)
        else:
            _constants.use_h5_set(None)  # stays None → silent skip
    elif bio:
        if _constants.H5_AVAILABLE:
            _constants.use_h5_set(True)
        else:
            lf.warning(
                "use_h5=True requested but h5py unavailable — "
                "forced to False (TSV-only mode)"
            )
            OmegaConf.update(cfg, "program.use_h5", False)
            _constants.use_h5_set(False)
    else:
        # User explicitly set False — honour without extra logging.
        _constants.use_h5_set(False)


def run(cfg: DictConfig) -> dict[str, dict[str, Any]] | None:
    """Discover (text), generate configs, process — the canonical pipeline entry point.

    Accepts a Hydra-composed :class:`DictConfig` (from ``@hydra.main``).
    All groups (input, out, filter, program) are fully resolved — no MISSING
    sentinels.  Per-probe YAMLs are merged on top via
    ``OmegaConf.merge(cfg, OmegaConf.load(yaml))``.

    **Text inputs** (``.txt/.csv/.tsv``) — discovery sweep:
    1. Generate missing/stale YAML configs (``save_config_to_yaml``).
    2. Resolve which configs to process (by ``input.ids`` or all).
    3. Process each YAML via :func:`run_processing`.
    4. Log completion summary with ok/failed counts.

    **Binary inputs** (``.nc/.h5``) — direct dispatch:
    Calls :func:`run_processing` once per ``input.tables`` entry,
    skipping the entire config discovery/generation/sync chain.
    Use ``cli.call_in_raw_dir(processing.run, input={...}, yaml_path=...)``
    to pass an explicit per-probe YAML override if needed.

    :param cfg: Hydra-composed top-level configuration (from ``@hydra.main``).
    :returns: ``{input_path_str: cfg1_dict}`` from :func:`config_yaml.save_config_to_yaml`
        when ``program.return_ == Return.CFG_FROM_ARGS``; ``None`` otherwise.
        Each ``cfg1_dict`` is the **full** per-probe config (all fields, not just
        non-defaults) — unlike YAMLs on disk which strip defaults.
    """

    path_in = cfg.input.path
    if path_in is None:
        raise ValueError("cfg.input.path must be provided")
    path_in = Path(path_in).absolute()

    # ── Resolve use_h5: user preference × library availability ─────────
    # True  → proceed with NC/HDF5 I/O (no extra logging).
    # False → skip + warn at each write point.
    # None  → skip silently (env doesn't support, user didn't ask).
    _resolve_use_h5(cfg)

    # Binary formats (HDF5/NC) carry their own coefs and are not part of the
    # text-file discovery pipeline: skip config search/generation and process
    # the directly-passed config via :func:`run_processing`.  Per-probe run
    # YAMLs (``cfg_proc/run/``) and the searchpath injection are text-only;
    # pass an explicit ``yaml_path`` to :func:`cli.call_in_raw_dir` to opt in.
    if path_in.suffix.lower() in _EXT_BINARY:
        # Iterate ``input.tables`` → one probe per table group (NC/HDF5 may
        # hold several probes in one file); pin ``tables=[tbl]`` per call so
        # :func:`run_processing` derives the correct pcid and output group.
        tables = list(cfg.input.tables or [])
        for tbl in tables or [""]:
            cfg_pc = OmegaConf.merge(
                cfg, OmegaConf.create({"input": {"tables": [tbl]}})
            ) if tables else cfg
            run_processing(cfg_pc)
        lf.info(
            "Done — processed {} table{} from {}",
            len(tables), "" if len(tables) == 1 else "s", path_in.name,
        )
        return

    ids = list(cfg.input.ids) if cfg.input.ids else None
    pcids_requested = format.normalize_probes(set(ids)) if ids else {format.PROBE_WILDCARD}

    dir_raw = paths.find_dir_raw_absolute(path_in)
    dir_cfgs = dir_raw / "cfg_proc" / "run"

    cli.safe_cfg_dir(dir_cfgs)

    # ── yaml_path filter: ANY non‑None skips config generation ──────────
    yaml_path = OmegaConf.select(cfg, "input.yaml_path", default=None)

    cfgs_existed = config_yaml.get_existed_cfgs(dir_cfgs)

    if yaml_path is None:
        # Step 1: generate missing configs via gen_metadata (single discovery source)
        stale = config_yaml.find_stale_cfgs(cfgs_existed, dir_cfgs)
        # Check for source files that have no config yet (lightweight: just directory scan).
        # Runs for wildcard (all new files) and for specific IDs (only when requested IDs
        # lack configs but source data may exist) — avoids a false "no configs" error.
        regenerate = bool(stale) or not cfgs_existed
        if not regenerate:
            try:
                from tcm import csv_load
                discovered = csv_load.search_csv_files(path_in)
                disc_pcids = {format.pcid_from_parts(model=m, number=n) for m, n in discovered}
                cfg_pcids = set(cfgs_existed)
                new_pcids = disc_pcids - cfg_pcids
                if new_pcids and (
                    pcids_requested == {format.PROBE_WILDCARD} or pcids_requested & new_pcids
                ):
                    lf.info("Source files without configs: {} — will generate", new_pcids)
                    regenerate = True
            except (FileNotFoundError, OSError):
                pass  # discovery fails → skip regeneration check
        if regenerate:
            if stale:
                reason = f"regenerating {len(stale)} stale config(s): {', '.join(stale)}"
            elif not cfgs_existed:
                reason = "no configs exist — generating from scratch"
            else:
                reason = "new source files found"
            lf.info("Config generation: {}", reason)
            config_yaml.save_config_to_yaml(cfg, [path_in])
            cfgs_existed = config_yaml.get_existed_cfgs(dir_cfgs)
        # Sync time_ranges from info_devices metadata into run YAMLs lacking them.
        # Idempotent: configs with existing input.time_ranges are skipped inside.
        config_yaml.sync_yamls_devmeta_and_hydra(dir_raw.parent, dir_cfgs, cfgs_existed)
        # Warn about orphan configs after regeneration
        still_stale = config_yaml.find_stale_cfgs(cfgs_existed, dir_cfgs) if stale else {}
        if still_stale:
            stale_pcids = set(still_stale)
            ignored = (
                stale_pcids - pcids_requested
                if pcids_requested != {format.PROBE_WILDCARD} else set()
            )
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

    # Step 2: resolve which configs to run
    if pcids_requested == {format.PROBE_WILDCARD}:
        cfgs_to_run = cfgs_existed
    elif pcids_requested:
        if not_found := (pcids_requested - set(cfgs_existed)):
            raise ValueError(
                f"Requested probes have no configs: {not_found}. Available: {sorted(cfgs_existed)}"
            )
        cfgs_to_run = {k: v for k, v in cfgs_existed.items() if k in pcids_requested}
    else:
        cfgs_to_run = {}

    # Exclude stale stems (already warned above) so process_loading_yaml never sees them
    if still_stale:
        cfgs_to_run = {
            pcid: [s for s in stems if s not in still_stale.get(pcid, [])]
            for pcid, stems in cfgs_to_run.items()
        }
        cfgs_to_run = {pcid: stems for pcid, stems in cfgs_to_run.items() if stems}

    # ── Filter by yaml_path pattern (YAML stem match) ──────────────────
    from tcm.csv_load import _pattern_to_regex as _ptr

    if yaml_path:
        _yp_re = re.compile(_ptr(yaml_path), re.IGNORECASE)
        cfgs_to_run = {
            pcid: [
                s for s in stems
                if _yp_re.fullmatch(s) or _yp_re.fullmatch(f"{s}.yaml")
            ]
            for pcid, stems in cfgs_to_run.items()
        }
        cfgs_to_run = {k: v for k, v in cfgs_to_run.items() if v}
        lf.debug("yaml_path filter '{}' → {} probes", yaml_path, len(cfgs_to_run))

    # ── Filter by input.path pattern (match against YAML's input.path) ─
    # Skipped only when path_in is a directory (default discovery mode).
    # When path_in is a concrete file OR a glob/regex, only YAMLs whose
    # stored input.path filename matches the pattern are kept.
    # Uses ruamel YAML (config_yaml._ry) to avoid collision with OmegaConf.load
    # mocks in process_loading_yaml.
    if not path_in.is_dir():
        _ip_re = re.compile(_ptr(path_in.name), re.IGNORECASE)
        _ry = config_yaml._ry(write=False)
        filtered = {}
        for pcid, stems in cfgs_to_run.items():
            matched = []
            for s in stems:
                try:
                    y = _ry.load((dir_cfgs / f"{s}.yaml").open(encoding="utf-8"))
                    ip = (y or {}).get("input", {}).get("path", "")
                    if ip and _ip_re.fullmatch(Path(ip).name):
                        matched.append(s)
                except Exception:
                    continue
            if matched:
                filtered[pcid] = matched
        if filtered != cfgs_to_run:
            lf.debug(
                "input.path filter '{}' → {} probes (was {})",
                path_in.name, len(filtered), len(cfgs_to_run),
            )
            cfgs_to_run = filtered

    # Step 3: process each config
    # Early-exit (CFG_FROM_ARGS): run_processing returns DictConfig before data load.
    processed_pcids, failed_pcids, last_cfg, collected = cli.process_loading_yaml(
        run_processing, base_cfg=cfg, dir_cfgs=dir_cfgs, cfgs=cfgs_to_run, n_cfgs_existed=len(cfgs_existed)
    )

    if cfg["program"]["return_"] == Return.CFG_FROM_ARGS:
        return (processed_pcids, failed_pcids, last_cfg, collected)

    # Combined output: merge distinct probes with probe dimension (legacy parity).
    # Deduplicate preserving order (multiple stems for the same pcid do not constitute multiple probes).
    # No main_init needed: last_cfg.out already has PathLayout-resolved paths (side-effect from
    # run_processing's main_init), and _dt_bins/_dt_min_save handle raw int/str values.
    # Combined output requires HDF5/netCDF4 backend — skip when use_h5 is not True.
    distinct_pcids = list(dict.fromkeys(processed_pcids))
    if len(distinct_pcids) > 1 and last_cfg is not None:
        if _constants.use_h5_get() is True:
            if not _constants.NC4_AVAILABLE:
                lf.debug("Combine skipped — netCDF4 not available")
            else:
                _combine_probes(distinct_pcids, last_cfg)

    # Statistics on all variables before exit
    lf.debug(
        "Run stats: cfgs_existed={}, cfgs_to_run={}, processed={}, failed={}, distinct={}, "
        "last_cfg={}",
        len(cfgs_existed), len(cfgs_to_run), processed_pcids, failed_pcids, distinct_pcids,
        OmegaConf.select(last_cfg, "_yaml_path") if last_cfg else None,
    )
    # Count by distinct probe (a probe that failed on one YAML but succeeded on
    # another is considered successful — the failure was in config, not in data).
    truly_failed = sorted(set(failed_pcids) - set(processed_pcids))
    skipped = sorted(pcids_requested - set(cfgs_existed)) if pcids_requested != {format.PROBE_WILDCARD} else []
    parts = []
    if distinct_pcids:
        parts.append(f"{len(distinct_pcids)} probes: {', '.join(distinct_pcids)} ok")
    if skipped:
        parts.append(f"{len(skipped)} skipped ({', '.join(skipped)})")
    if truly_failed:
        parts.append(f"{len(truly_failed)} failed ({', '.join(truly_failed)})")
    if (return_ := OmegaConf.select(cfg, "program.return_", default=None)) and return_ != Return.END:
        parts.append(f"return_={return_}")
    lf.info("Done — {}", " | ".join(parts) if parts else "nothing processed")
    return (processed_pcids, failed_pcids, last_cfg, collected)


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

    lf.debug("Loading data for {}...", pcid)
    cfg = cli.main_init(cfg, program_name="TCM processing")
    # Early-exit: main_init returns DictConfig before ini2dict; propagate upstream.
    if not isinstance(cfg, dict):
        return cfg
    # Set per-config base for granular stage ticks (injected by process_loading_yaml).
    if (si := cfg.get("_stem_idx")) and (nc := cfg.get("_n_cfgs")):
        _probe_base["v"] = (si - 1) * 100
        _probe_total["v"] = nc * 100
    cfg_in = cfg["input"]  # already type-converted plain dict after main_init

    # Active stage plan for upper-bar ticks (NC only if use_h5 True; TSV if text_path).
    # Each active stage gets an equal slice of the 100-unit per-probe scale.
    _dt_bins_list = _dt_bins(cfg["out"])
    _n_bins = len(_dt_bins_list)
    _has_nc = _constants.use_h5_get() is True
    _has_tsv = bool(cfg["out"].get("text_path"))
    _stages = [Stage.LOAD, Stage.COEFS, Stage.PROC]
    _stages += [Stage.NC] * _n_bins if _has_nc else []
    _stages += [Stage.TSV] * _n_bins if _has_tsv else []
    _n_active = len(_stages)

    def _frac(i: int) -> int:  # boundary i of N active stages → 0..100
        return round(i * 100 / _n_active)

    # Batch mode (cfg.files exists): iterate and concatenate
    if cfg.get("files"):
        ds_raw, coefs_from_file = _load_batch(cfg, pcid)
    else:
        # Single-file mode — load data + optional coefs from file
        ds_raw, coefs_from_file = xr_io.load_raw(
            tbl=tbl,
            text_type=pcid[:1] if pcid else "i",
            cfg_in=cfg_in,
        )
    _stage(f"{pcid} {Stage.LOAD}", _frac(1))  # load done → tick 1

    # Coefs: coefs_path (file) → input.coefs (run YAML override wins)
    coefs = get_coefs_from_cfg(cfg_in, pcid)
    if coefs_from_file:
        coefs = {**coefs, **{k: v for k, v in coefs_from_file.items() if v is not None}}
        lf.debug("Merged coefs from data file: {} extra keys", len(coefs_from_file))

    # ── Phase 1b: extract coefs from .raw.h5 if .raw.nc absent (legacy HDF5 auto-migrate)
    if (raw_nc_path := cfg["out"].get("raw_db_path")) and _constants.use_h5_get() is True:
        raw_nc_path = Path(raw_nc_path)
        if not raw_nc_path.exists():
            if (h5_path := raw_nc_path.with_suffix("").with_suffix(".raw.h5")).exists():
                from tcm.incl_calc.coefs import load_coefs
                if (h5_coefs := load_coefs(h5_path, tbl)):
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
    if msg:
        lf.debug("Coefs prepared: {}", msg)
    _stage(f"{pcid} {Stage.COEFS}", _frac(2))  # coefs done → tick 2

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
        if changed_coefs and _constants.use_h5_get() is True:
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
                yaml_path, {k: coefs_merged[k] for k in changed_coefs},
            )
            yaml_written = True
        elif changed_coefs:
            lf.warning("Coefs changed ({}) but no write target available", sorted(changed_coefs))
        else:
            lf.debug("Coefs unchanged for {} — skipping write", pcid)
    elif _constants.use_h5_get() is True and (raw_nc := cfg["out"].get("raw_db_path")):
        # CSV source + H5: write all coefs to raw_db_path (first creation or changed)
        raw_nc = Path(raw_nc)
        if not raw_nc.exists() or changed_coefs:
            coefs_to_write = coefs_merged
        else:
            lf.debug("Coefs unchanged for {} — skipping write", pcid)
    elif yaml_path and changed_coefs:
        # noh5 fallback: write only changed coefs to run YAML
        config_yaml.update_coefs_in_run_yaml(
            yaml_path, {k: coefs_merged[k] for k in changed_coefs},
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
            yaml_path, {k: coefs_merged[k] for k in changed_coefs},
        )

    # ── Phase 4: Save raw data (skip for NC sources — data already there)
    if src_path.suffix.lower() not in _EXT_BINARY and ds_raw is not None:
        try:
            from tcm import h5
        except ImportError:
            lf.debug("pytables not available — skipping raw NC data save for {}", pcid)
        else:
            raw_nc_path = cfg["out"].get("raw_db_path")
            if raw_nc_path:
                # Release xarray read handle so h5py can open the same file.
                # On Windows, HDF5 uses mandatory locking — any open handle
                # (even read-only) blocks new opens (read or write).
                if ds_raw is not None:
                    ds_raw.load()
                    ds_raw.close()
                file_meta = h5.file_name_and_time_to_record(src_path)
                storage.nc_incremental_update(ds_raw, Path(raw_nc_path), tbl, file_meta)

    # Write coefs after data (raw NC may have been created by Phase 4)
    if coefs_to_write is not None and (raw_nc := cfg["out"].get("raw_db_path")):
        xr_coefs.save_coefs_to_nc(Path(raw_nc), tbl, coefs_to_write, pcid=pcid, dates=dates)
        # save_coefs_to_nc logs "Coefs saved to ..."

    # Phase-stopping: stop after coefs saved or raw data saved (before processing).
    if (return_ := cfg["program"]["return_"]) in (Return.SAVED_COEFS, Return.SAVED_RAW):
        lf.info(
            "return_={} — stopping after {} for {}",
            return_,
            "coef save" if return_ == Return.SAVED_COEFS else "raw NC save",
            pcid,
        )
        return

    lf.debug(
        "Processing {} (bins: {})...", pcid,
        ", ".join(str(int(b.total_seconds())) for b in _dt_bins(cfg["out"])),
    )
    # ── stage ticks closure for _process_and_persist (PROC, NC×n_bins, TSV×n_bins)
    _tick_idx = {"v": 2}  # already ticked LOAD(1), COEFS(2)

    def _tick(stage: Stage, bin_i: int = 0, n_bin: int = 1) -> None:
        _tick_idx["v"] += 1
        label = f"{pcid} {stage}" + (f" {bin_i+1}/{n_bin}" if n_bin > 1 else "")
        _stage(label, _frac(_tick_idx["v"]))

    _process_and_persist(
        ds_raw, coefs_merged, cfg, pcid,
        coef_zeroing_matrix=coef_zeroing_matrix, tick=_tick, has_nc=_has_nc, has_tsv=_has_tsv,
    )


def _load_batch(
    cfg: Dict[str, Dict[str, Any]], pcid: str
) -> tuple[Optional[xr.Dataset], Optional[Dict[str, Any]]]:
    """Iterate ``cfg.files``, load each, concatenate"""

    frames = []
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
            frames.append(ds_chunk)

    if not frames:
        lf.warning("No data loaded for batch {} — skipping", pcid)
        return None, None

    ds_raw = xr.concat(frames, dim="time") if len(frames) > 1 else frames[0]
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
) -> None:
    """Apply physical conversion + binning, persist results.

    Saves each bin result to netCDF using shared files with per-probe groups:
    - no-avg (dt_bin=0): writes to ``not_joined_db_path`` with group ``/{pcid}/``
    - binned (dt_bin>0): writes to ``db_path`` (derived from ``not_joined_db_path``
      by replacing ``_noAvg``) with group ``/{pcid}bin{bin_s}s/``

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

    # Build run_params text for re-run warning: sorted text of resolved filter + window + coefs
    run_params_text = _build_filter_params_text(cfg_in, cfg_filter, coefs=coefs)

    # Resolve +force_reprocess (accepted via +, not in structured schema)
    force_reprocess = bool(cfg.get("force_reprocess", False))

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
            v if isinstance(v := cfg_in.get("dt_min_binning_proc"), timedelta)
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
    noavg_path, avg_path = _output_nc_paths(cfg_out)
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
                    ds_out, noavg_path, group=pcid,
                    filter_params=run_params_text, force_reprocess=force_reprocess,
                )
                # Phase-stopping: return after noAvg save
                if return_ == Return.SAVED_NOAVG:
                    lf.info(
                        "return_={} — stopping after noAvg save for {}", Return.SAVED_NOAVG, pcid
                    )
                    return
            elif bin_s > 0 and avg_path:
                # binned → /{pcid}bin{bin_s}s/ group in *.proc.nc (incremental skip + run-params sig)
                storage.store_processed_incremental(
                    ds_out, avg_path, group=f"{pcid}bin{bin_s}s",
                    filter_params=run_params_text, force_reprocess=force_reprocess,
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
                ts = datetime.fromtimestamp(
                    int(ds_out["time"].values[0]) // 1_000_000_000, timezone.utc
                ).strftime("%y%m%d_%H%M")
                csv_name = f"{ts}{suffix_csv}@{pcid}.tsv"
                csv_out = Path(text_path) / csv_name

                if not cfg_out.get("b_overwrite_text", True) and csv_out.exists():
                    lf.info("TSV exists, b_overwrite_text=False — skipping {}", csv_out.name)
                else:
                    if tick and has_tsv:
                        tick(Stage.TSV, _bin_i, len(dt_bins))  # TSV stage start for this bin
                    xr_io.ds_to_csv(
                        ds_out,
                        csv_out,
                        split_period=split_period,
                        text_date_format=fmt,
                        text_columns=text_columns or None,
                    )
                    lf.info("Saved TSV {} to {}", pcid, csv_out.name)

    # Phase-stopping: return after all NC writes (before combined output)
    if return_ == Return.SAVED_ALL:
        lf.info("return_={} — stopping after all NC saves for {}", Return.SAVED_ALL, pcid)
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
    noavg_path, avg_path = _output_nc_paths(cfg_out)
    if not noavg_path:
        return

    text_path = cfg_out.get("text_path")
    text_columns = cfg_out.get("text_columns") or []
    dt_bins = _dt_bins(cfg_out)
    joined = ",".join(pcids)
    probe_type = pcids[0][0] if pcids else "i"
    if any(p[0] != probe_type for p in pcids):
        lf.debug("Combined probes have mixed types — using type '{}' from first pcid", probe_type)

    # Combine noAvg groups
    if noavg_path.exists():
        storage.ensure_dim_scales(noavg_path)
        _merge_groups_to_combined(noavg_path, pcids, f"/{probe_type}/", "noAvg")

    # Combine binned groups — group name ``{probe_type}_bin{bin_s}s``
    if avg_path.exists():
        storage.ensure_dim_scales(avg_path)
        for dt_bin in dt_bins:
            bin_s = int(dt_bin.total_seconds())
            if bin_s <= 0:
                continue
            combined_group = f"/{probe_type}_bin{bin_s}s/"
            _merge_groups_to_combined(
                avg_path, pcids, combined_group, f"bin{bin_s}s", bin_s=bin_s
            )

    # Combined TSV (for each binned result)
    if text_path:
        dt_min_save = _dt_min_save(cfg_out)
        for dt_bin in dt_bins:
            if dt_bin < dt_min_save:
                continue
            bin_s = int(dt_bin.total_seconds())
            # Read combined group from NC, write TSV (matches new combined group naming)
            nc_path = avg_path if bin_s > 0 else noavg_path
            combined_group = f"/{probe_type}_bin{bin_s}s/" if bin_s > 0 else f"/{probe_type}/"
            try:
                ds_combined = xr.open_dataset(nc_path, group=combined_group, engine="netcdf4")
            except (AttributeError, KeyError, OSError):
                lf.debug("Combined group {} not found — skipping TSV", combined_group)
                continue

            ts = datetime.fromtimestamp(
                int(ds_combined["time"].values[0]) // 1_000_000_000, timezone.utc
            ).strftime("%y%m%d_%H%M")
            suffix_csv = f"bin{bin_s}s" if bin_s > 0 else ""
            csv_name = f"{ts}{suffix_csv}@{joined}.tsv"
            csv_out = Path(text_path) / csv_name
            xr_io.ds_to_csv(
                ds_combined,
                csv_out,
                text_date_format=_text_date_fmt(cfg_out, bin_s),
                text_columns=text_columns or None,
            )
            lf.info("Saved combined TSV {} to {}", joined, csv_out.name)
            ds_combined.close()


def _merge_groups_to_combined(
    nc_path: Path,
    pcids: list[str],
    combined_group: str,
    label: str,
    *,
    bin_s: int = 0,
) -> None:
    """Read per-probe groups from NC, merge with probe dimension, write combined group."""

    groups_to_merge = []
    for pcid in pcids:
        grp_name = f"{pcid}bin{bin_s}s" if bin_s > 0 else pcid
        try:
            ds = xr.open_dataset(nc_path, group=grp_name, engine="netcdf4", autoclose=True)
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
    storage.store_processed(combined, nc_path, group=combined_group.strip("/"), mode="a")
    lf.info("Combined {} to {} (probe dim with {} probes)", label, nc_path.name, len(groups_to_merge))


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
            xr_io.ds_to_csv(ds_out, csv_path, split_period=split_period)

    return [r for r in results if r is not None]
