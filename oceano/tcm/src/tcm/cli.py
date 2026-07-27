"""CLI argument parsing for the tcm processing pipeline.

Extracted from ``scripts/tcm_clc.py`` to keep the entry point as a thin caller.
Hydra handles all config keys (``input.path``, ``input.ids``, ``out.*``, etc.)
natively via ``compose`` — this module only handles pre-Hydra setup.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from functools import wraps
from pathlib import Path, PurePath
from typing import Any, Callable, Dict, Mapping, Optional
from io import StringIO

# ---------------------------------------------------------------------------
# argparse compatibility for Python 3.14 — must run before Hydra builds parser
# ---------------------------------------------------------------------------
# Python 3.14's ``_check_help`` calls ``_expand_help`` on every ``help=``
# value.  If help is a non-string, non-iterable (e.g. Hydra's
# ``LazyCompletionHelp``), ``_expand_help`` raises ``TypeError`` which
# ``_check_help`` re-raises as ``ValueError``.  Patch ``_get_help_string``
# to coerce non-string help values to ``str()``.
_orig_get_help_string = argparse.HelpFormatter._get_help_string


def _patched_get_help_string(self, action) -> str | None:
    help_string = _orig_get_help_string(self, action)
    if help_string is not None and not isinstance(help_string, str):
        return str(help_string)
    return help_string


argparse.HelpFormatter._get_help_string = _patched_get_help_string

import hydra
from omegaconf import DictConfig, MissingMandatoryValue, OmegaConf

from tcm import _constants, config, paths, to_omegaconf, config_yaml
from tcm import stage_ctx
from tcm.utils2init import (
    Ex_nothing_done,
    LoggingStyleAdapter,
    ini2dict,
    standard_error_info,
    this_prog_basename,
    update_cfg_time_ranges,
)

lf = LoggingStyleAdapter(__name__)


# Default glob pattern (Windows-first: uppercase I).
_TCM_DEFAULT_GLOB_PATTERN = "*I*.txt"
DEFAULT_GLOB = f"{config.RAW_DIR_NAME}/{_TCM_DEFAULT_GLOB_PATTERN}"


def parse_data_path(argv: list[str]) -> tuple[Optional[Path], list[str]]:
    """Extract first positional arg (data path) from ``argv``, handling commas.

    Positional = non-flag, non-``key=value`` argument.  Consecutive positional
    arguments are **joined with commas** to reconstruct paths split by shell
    comma-handling (e.g. ``@i,t-chain`` split by PowerShell).  Only the first
    run of consecutive positionals is consumed as the data path; subsequent
    positionals pass through in ``remaining``.

    :param argv: ``sys.argv``-style list (includes script name at index 0).
    :returns: ``(path_in, remaining_argv)`` where ``remaining_argv`` has the
        consumed positional span stripped.  Returns ``None`` for *path_in*
        when no non-flag, non-``key=value`` argument is found — callers
        that need a concrete path should apply :data:`DEFAULT_GLOB` as
        fallback (see :func:`call_in_raw_dir`).
    """
    remaining = list(argv)

    # Collect consecutive non-flag, non-key=value args as path segments
    segments: list[str] = []
    for i, arg in enumerate(argv[1:], start=1):
        if arg.startswith("-") or "=" in arg:
            if segments:
                break  # path complete — stop at first flag/key=value after segments
            continue  # pre-path flag, skip
        segments.append(arg)

    if not segments:
        return None, remaining  # No positional found — caller applies fallback

    # Remove consumed segments from remaining argv
    for seg in segments:
        remaining.remove(seg)

    # Join with commas: shell split at commas — reinstate them to reconstruct path
    path_in = Path(",".join(segments))
    return path_in, remaining


def safe_cfg_dir(path: Path) -> Path:
    """Create *path* only if it is outside the code project.

    Universal guard for ``cfg_proc/`` subdirectories (``run/``, ``log/``, …).
    Call this instead of ``path.mkdir(parents=True, exist_ok=True)`` for any
    config/log directory derived from data paths.

    :raises SystemExit: if *path* resolves inside :data:`_constants.PROJECT_ROOT`.
    """
    resolved = path.resolve()
    if resolved == _constants.PROJECT_ROOT or _constants.PROJECT_ROOT in resolved.parents:
        print(
            f"Error: refusing to create {resolved} inside code project {_constants.PROJECT_ROOT}.\n"
            "Move your data outside the project tree.",
            file=sys.stderr,
        )
        sys.exit(1)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


_result: Any = None  # stores fun(cfg) return — @hydra.main doesn't propagate it


def hydra_main(
    fun,
    config_name: str = "config",
    config_path: str = _constants.BUNDLED_CFG_PKG,
    version_base: str = "1.3",
    overrides: Optional[Mapping[str, Any]] = None,
    *,
    exit_on_error: bool = True,
) -> Any:
    """Dispatch *fun* via Hydra, return its result (``@hydra.main`` swallows returns).

    Two code paths:

    * **No overrides** (default): standard ``@hydra.main`` — composes defaults
      from ConfigStore + ``sys.argv`` CLI overrides.
    * **With overrides**: ``@hydra.main`` composes defaults as usual, then
      a wrapper deep-merges the hierarchical *overrides* dict on top before
      calling *fun*.  CLI ``sys.argv`` overrides still apply (below the dict).

    The decorated function's return value is stored in :data:`_result` and
    returned to the caller — ``@hydra.main`` itself discards return values.

    :param fun: task function accepting one ``DictConfig`` argument.
    :param overrides: hierarchical dict to merge on top of composed defaults.
    :returns: whatever *fun* returned (``None`` if it returned nothing).
    """
    global _result
    _result = None  # reset before each dispatch

    # Force Hydra to re-raise original exceptions instead of swallowing them
    # with sys.exit(1). Without this, ``except BaseException`` below catches
    # only a bare ``SystemExit`` and the real traceback is lost.
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")

    @wraps(fun)
    def _store(cfg: DictConfig):
        """Run *fun*, stashing its return in :data:`_result`."""
        global _result
        if overrides:
            base = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
            merged = OmegaConf.merge(base, overrides)
            _result = fun(merged)
        else:
            _result = fun(cfg)
        return _result

    try:
        m_fun = hydra.main(config_name=config_name, config_path=config_path, version_base=version_base)(
            _store
        )
        m_fun()
        return _result
    except BaseException:
        lf.exception("Error. Exiting the entire process")
        if exit_on_error:
            sys.exit(1)
        raise


# Kwargs accepted by :func:`hydra_main` (excluding ``fun``) — everything else
# is treated as an override dict to merge on top of composed defaults.
_HYDRA_MAIN_PARAMS = {"config_name", "config_path", "version_base", "overrides", "exit_on_error"}


def _build_hydra_argv(data_dir: Path) -> list[str]:
    """Build Hydra argv overrides — only ``--config-dir`` (argparse layer).

    ``input.path`` is injected via :func:`_prepare_overrides` (OmegaConf merge)
    to entirely bypass Hydra's ANTLR override parser.

    ``--config-dir`` targets Hydra's **argparse** layer, which natively handles
    commas, backslashes, colons, parentheses, brackets, braces, equals signs,
    and other ANTLR special characters — no escaping or quoting needed.
    """
    cfg_dir = data_dir / "cfg_proc"
    return ["--config-dir", str(cfg_dir)] if cfg_dir.is_dir() else []


def _prepare_overrides(path_in: Path, overrides: dict) -> dict:
    """Inject ``input.path`` into *overrides* for OmegaConf merge after composition.

    This bypasses Hydra's CLI override parser entirely — the path string
    never passes through the ANTLR grammar, so commas, backslashes, quotes,
    and other special characters are handled correctly.
    """
    return OmegaConf.to_container(
        OmegaConf.merge(OmegaConf.create(overrides), {"input": {"path": path_in.as_posix()}}),
        resolve=True,
    )


def call_in_raw_dir(fun, yaml_path: Optional[Path] = None, **kwargs) -> Any:
    """Bootstrap CLI → Hydra runtime for a processing entry point.

    1. Parses the raw-data path from ``sys.argv``: 1st non-flag, non-``key=value``
       argument treated as path; resolves to nearest ``_raw/`` ancestor via
       :func:`paths.find_dir_raw_absolute` (always returns a valid directory).
    2. Changes the working directory there.
    3. Injects ``input.path`` into the *overrides* dict via
       :func:`_prepare_overrides` (OmegaConf merge — bypasses Hydra's ANTLR
       override parser, so commas, backslashes, quotes are handled correctly).
       Injects ``--config-dir`` into ``sys.argv`` via
       :func:`_build_hydra_argv` (argparse layer — natively handles ALL
       ANTLR special characters in paths).
    4. Calls *fun* via :func:`hydra_main`.

    Any keyword argument whose name is **not** a :func:`hydra_main` parameter
    (``config_name``, ``config_path``, ``version_base``, ``overrides``) is
    collected into an *overrides* dict and passed to :func:`hydra_main` as
    hierarchical config overrides that layer **on top of** Hydra-composed
    defaults — preserving all ConfigStore defaults for unspecified groups.

    Args:
        fun: A callable accepting one ``DictConfig`` argument
            (typically a ``@hydra.main``-decorated or plain function).
        yaml_path: Optional per-probe run YAML (``cfg_proc/run/<name>.yaml``).
            Loaded via :func:`OmegaConf.load` and used as the **base** for
            dict overrides — explicit ``**kwargs`` win over YAML values.
        **kwargs: ``config_name``, ``config_path``, etc. forwarded to
            :func:`hydra_main`; remaining keys (e.g. ``input={...}``) are
            treated as config-group overrides.

    Returns:
        Whatever *fun* returned (``None`` if it didn't return anything).
        ``@hydra.main`` doesn't propagate return values, so :func:`hydra_main`
        stashes the result in :data:`_result` and returns it here.

    Note:
        ConfigStore registration (structured-group dataclasses) must happen
        before ``@hydra.main`` resolves — for the processing pipeline,
        ``tcm.config`` (imported above) does this at module level.
    """
    # Separate hydra_main params from override dicts.
    hydra_main_kwargs: Dict[str, Any] = {}
    overrides: Dict[str, Any] = {}
    for k, v in kwargs.items():
        if k in _HYDRA_MAIN_PARAMS:
            hydra_main_kwargs[k] = v
        else:
            overrides[k] = v

    # When overrides dict is provided, extract input.path from it for
    # _prepare_overrides injection.  Remove from hydra_main_kwargs to avoid
    # duplicate 'overrides' in the final call.
    if "overrides" in hydra_main_kwargs:
        overrides = OmegaConf.to_container(
            OmegaConf.merge(OmegaConf.create(overrides), hydra_main_kwargs.pop("overrides")),
            resolve=True,
        )

    # Load per-probe YAML as base; explicit kwargs merge on top via OmegaConf.
    if yaml_path is not None:
        yaml_cfg = OmegaConf.load(Path(yaml_path))
        overrides_cfg = OmegaConf.create(overrides)
        overrides = OmegaConf.to_container(OmegaConf.merge(yaml_cfg, overrides_cfg), resolve=True)

    # Extract input.path: from overrides dict or from sys.argv.
    try:
        path_in = Path(overrides["input"]["path"])
    except KeyError:
        try:
            path_in = Path(hydra_main_kwargs["overrides"]["input"]["path"])
        except (KeyError, AttributeError, TypeError):
            path_in, remaining_argv = parse_data_path(sys.argv)
            if path_in is None:
                path_in = Path(DEFAULT_GLOB)  # CLI fallback when no positional arg
            path_in = path_in.resolve() if not path_in.is_absolute() else path_in
        else:
            # overrides dict provided path — keep sys.argv as-is (Worker
            # sets it to [script] + hydra_args before each call).
            remaining_argv = list(sys.argv)
    else:
        remaining_argv = list(sys.argv)

    # Resolve the nearest `_raw/` ancestor (always returns a valid dir).
    data_dir = paths.find_dir_raw_absolute(path_in)

    # Inject input.path into overrides dict (bypasses Hydra's ANTLR parser).
    overrides = _prepare_overrides(path_in, overrides)

    # Build sys.argv for @hydra.main: script name + searchpath + user options.
    # input.path is in overrides — never touches Hydra's override parser.
    remaining_argv[1:1] = _build_hydra_argv(data_dir)

    os.chdir(data_dir)

    sys.argv = remaining_argv
    return hydra_main(fun, overrides=overrides or None, **hydra_main_kwargs)


def process_loading_yaml(process_fun: Callable, base_cfg, dir_cfgs, cfgs, n_cfgs_existed):
    """Load per-probe run YAMLs, merge on top of *base_cfg*, call process_fun.

    Run YAMLs (``cfg_proc/run/*.yaml``) use ``@package _global_`` — their
    keys live at the root level.  :func:`OmegaConf.merge` applies the YAML
    as overrides on top of *base_cfg*, preserving all groups not mentioned
    in the YAML.

    **Pre-filtering**: before processing, all YAML stems are validated against
    ``input.path`` to exclude backup copies (e.g. ``i_90-backup260723.yaml``).
    One WARNING summarises all skips; the remaining valid configs drive the
    ``[idx/n_cfgs]`` numbering and GUI progress totals.

    :param process_fun: Callable(merged config)
    :param base_cfg: Hydra-composed config (from ``@hydra.main``).
    :param dir_cfgs: absolute path to the directory of run YAML.
    :param cfgs: map of pcid to its config stem from `dir_cfgs` dir
    :param n_cfgs_existed: number of existed configs for logging
    :return: processed_pcids, failed_pcids, last_cfg, collected
        where ``collected`` is ``[(stem, yaml_path_str, result), ...]``
    """
    processed_pcids: list[str] = []
    failed_pcids: list[str] = []
    last_cfg: DictConfig | None = None
    collected: list[tuple[str, str, Any]] = []
    if not cfgs:
        lf.info("No configs to run (available: {}, requested: {})", n_cfgs_existed, len(cfgs))
        return processed_pcids, failed_pcids, last_cfg, collected

    # ── Pre-filter: validate YAML stems against input.path ──────────────
    # Exclude backup copies and missing files BEFORE the processing loop so
    # that [idx/n_cfgs] numbering and GUI progress totals are correct.
    valid_cfgs: dict[str, list[str]] = {}  # pcid → [valid stems]
    loaded_cfgs: dict[str, DictConfig] = {}  # stem → merged DictConfig (cached)
    skipped_cfgs: list[tuple[str, str, str]] = []  # (stem, yaml_core, input_core)
    missing_cfgs: list[tuple[str, str]] = []  # (stem, pcid)

    for pcid, stems in cfgs.items():
        valid_stems: list[str] = []
        for stem in stems:
            yaml_path = dir_cfgs / f"{stem}.yaml"
            if not yaml_path.is_file():
                missing_cfgs.append((stem, pcid))
                continue
            cfg_dc = OmegaConf.load(yaml_path)
            cfg_dc = OmegaConf.merge(base_cfg, cfg_dc)
            OmegaConf.update(cfg_dc, "_yaml_path", yaml_path, force_add=True)
            yaml_core = stem.rsplit("@", 1)[-1]
            input_core = Path(cfg_dc.input.path).stem.rsplit("@", 1)[-1]
            if yaml_core != input_core:
                skipped_cfgs.append((stem, yaml_core, input_core))
                continue
            valid_stems.append(stem)
            loaded_cfgs[stem] = cfg_dc  # cache for processing loop
        if valid_stems:
            valid_cfgs[pcid] = valid_stems

    # Report all skips in one consolidated message
    if skipped_cfgs:
        details = "; ".join(f'"{s}" (stem "{yc}" ≠ path "{ic}")' for s, yc, ic in skipped_cfgs)
        lf.warning(
            "Skipping {} config{} — YAML stem ≠ input.path (manual copy?): {}",
            len(skipped_cfgs),
            "" if len(skipped_cfgs) == 1 else "s",
            details,
        )
    for stem, pcid in missing_cfgs:
        lf.warning("Config missing for {}: {}", pcid, dir_cfgs / f"{stem}.yaml")
        failed_pcids.append(pcid)

    cfgs = valid_cfgs
    n_cfgs = sum(len(s) for s in cfgs.values())
    n_probes = len(cfgs)

    # ── Process valid configs ───────────────────────────────────────────
    lf.info(
        "Running {} {}{} ({} config{})",
        n_probes,
        "probe" if n_probes == 1 else "probes",
        f" of {n_cfgs_existed} available" if n_probes != n_cfgs_existed else "",
        n_cfgs,
        "" if n_cfgs == 1 else "s",
    )
    stem_idx = 0
    for probe_i, (pcid, stems) in enumerate(cfgs.items(), start=1):
        for cfg_i, stem in enumerate(stems, start=1):
            stem_idx += 1
            # Reuse cached config from pre-filter pass
            cfg_dc = loaded_cfgs[stem]

            # Set stage context for logging prefix and GUI progress display
            stage_ctx.set_probe(
                pcid,
                probe_idx=probe_i,
                cfg_idx=cfg_i,
                n_probes=n_probes,
                n_cfgs=len(stems),
            )

            lf.info('[{}/{}] probe {} (from "{}")', stem_idx, n_cfgs, pcid, yaml_path.name)
            OmegaConf.update(cfg_dc, "_stem_idx", stem_idx, force_add=True)
            OmegaConf.update(cfg_dc, "_n_cfgs", n_cfgs, force_add=True)
            try:
                result = process_fun(cfg_dc)
                processed_pcids.append(pcid)
                last_cfg = cfg_dc
                if result is not None:
                    # CFG_FROM_ARGS (scan): result=DictConfig — no data processed
                    collected.append((stem, str(yaml_path), result))
            except FileNotFoundError as e:
                lf.warning(
                    "[{}/{}] {}: source file missing ({}). Delete stale YAML",
                    stem_idx,
                    n_cfgs,
                    pcid,
                    e.filename or e,
                )
                failed_pcids.append(pcid)
            except Exception:
                lf.exception("[{}/{}] Processing failed for {}", stem_idx, n_cfgs, pcid)
                failed_pcids.append(pcid)
            finally:
                gc.collect()  # release previous probe's data before loading next

    # Clear stage context after all configs processed
    stage_ctx.clear()
    return processed_pcids, failed_pcids, last_cfg, collected


def sugar_expand_m(cfg_dict: dict[str, Any]) -> None:
    """Expand shorthand keys in-place within cfg sub-dicts. Sugar expansion

    For every nested ``min`` / ``max`` dict in *cfg_dict* (found recursively
    at the first nesting level only — not deeper), replaces shorthand keys
    (e.g. ``"M"``) with concrete axis keys (``"Mx"``, ``"My"``, ``"Mz"``),
    copying the value into each new key **only when the concrete key is absent**.

    Parameters
    ----------
    cfg_dict
        Configuration dict (e.g. ``cfg.input`` or ``cfg.filter``).
        Modified in-place.
    """

    _SHORTHAND_EXPANSIONS: dict[str, tuple[str, ...]] = {
        "M": ("Mx", "My", "Mz"),  # Shorthand column keys expanded to concrete axis keys
    }

    for sub_key in ("min", "max"):
        sub = cfg_dict.get(sub_key)
        if not isinstance(sub, dict):
            continue
        for short, expansions in _SHORTHAND_EXPANSIONS.items():
            if short not in sub:
                continue
            val = sub[short]
            for concrete in expansions:
                sub.setdefault(concrete, val)
            del sub[short]  # remove shorthand after expansion


def sugar_condense_lim_date(cfg_dict: Mapping[str, Any]) -> Mapping[str, Any]:
    # """initialise program state, convert types."""
    # if cfg.input.path is None:
    #     raise ValueError("input.path must be provided")

    # # Build cfg_in: resolve input config, run sugar merge + M expansion
    # cfg_in = OmegaConf.to_container(cfg.input, resolve=True)
    update_cfg_time_ranges(
        cfg_dict,
        min_date=cfg_dict.pop("min_date", None),
        max_date=cfg_dict.pop("max_date", None),
    )
    # cfg_in["path"]
    # cfg = {
    #     "in": cfg_in,
    #     "filter": OmegaConf.to_container(cfg.filter, resolve=True) if cfg.filter else None,
    #     "proc": OmegaConf.to_container(cfg.proc, resolve=True) if OmegaConf.select(cfg, "proc") else None,
    # }
    # return cfg_dict


def main_init(
    cfg: DictConfig[str, DictConfig[str, Any]],
    program_name: str = "",
    __file__: Optional[str] = None,
) -> Mapping[str, Any]:
    """Convert Hydra ``DictConfig`` to a plain dict with resolved types and paths.

    Centralised post-Hydra bootstrap shared by all entry points
    (``run_processing``, ``run_calibration``, etc.).  Every pipeline MUST call
    this once after Hydra composition and before any downstream consumption.

    Steps (in order):
      1. logs config summary (non-empty values only, debug level)
      2. Early return for sentinel ``program.return_`` values.
      3. ``ini2dict(cfg)`` — converts ``DictConfig`` → plain ``dict``, applying
         name-driven type conversions (see :func:`utils2ini2dict`):

         - ``dt_*`` prefix → ``timedelta`` (suffix becomes the unit, default seconds)
         - ``*_path`` / ``path_*`` → ``pathlib.Path``
         - ``*_date`` / ``*_time`` → ``datetime``
         - ``*_int`` / ``*_float`` / ``*_bool`` → native types
         - ``*_list`` / ``*_names`` → ``list`` (comma-split, recursive fix)
         - ``*_dict`` → ``dict`` (colon-split, recursive fix)
         - ``min_*`` / ``max_*`` (catch-all) → ``float``

      4. Sugar expansion: ``M`` shorthand → ``Mx/My/Mz`` in min/max dicts.
      5. ``min_date``/``max_date`` → merged into ``time_ranges``.
      6. ``PathLayout`` resolves output paths (``db_path``, ``not_joined_db_path``,
         ``raw_db_path``, ``text_path``) and writes them into **both** the original
         ``cfg.out`` DictConfig **and** the returned ``cfg_t["out"]`` dict.

    Parameters
    ----------
    cfg
        Hydra-composed top-level ``DictConfig`` (all groups resolved).
    program_name
        Short label for the startup banner (e.g. ``"TCM processing"``).
    __file__
        Optional module path for the startup banner (auto-detected if omitted).

    Returns
    -------
    dict
        Plain ``dict`` with type-converted values, sugar expansions applied,
        and output paths resolved.  Downstream code should use this instead of
        the original ``DictConfig``.
    """
    lf.debug("Working directory: {}", os.getcwd())
    try:
        conf_, ignored_keys = to_omegaconf.to_omegaconf_merge_compatible(cfg, config.Config)
        conf_ignored = {
            k0: (
                {k1: v1 for k1, v1 in v0.items() if v1}
                if hasattr(v0, "items")
                else str(v0)
                if isinstance(v0, PurePath)
                else v0
            )
            for k0, v0 in cfg.items()
            if k0 in ignored_keys
        }
        ru = config_yaml._ry()
        with StringIO() as s:
            s.writelines("--- Configuration (defaults excluded) ---\n")
            ru.dump(conf_, s)
            s.writelines("--- Additional arguments ---\n")
            ru.dump(conf_ignored, s)
            msg = s.getvalue()
        lf.debug(msg)
        # OmegaConf.to_yaml({
        #     k0: ({k1: v1 for k1, v1 in v0.items() if v1} if hasattr(v0, "items") else hasattr(v0, "items") if isinstance(v0, PurePath) else v0)
        #     for k0, v0 in cfg.items()
        # })
    except MissingMandatoryValue as e:
        lf.error(standard_error_info(e))
        raise Ex_nothing_done()

    if not cfg.program.return_:
        print("Can not initialise: provide non empty program.return_ value")
        return cfg
    elif cfg.program.return_ == config.Return.CFG_FROM_ARGS:
        return cfg

    hydra.verbose = cfg.program.verbose == "DEBUG"
    print("\n" + this_prog_basename(__file__) if __file__ else program_name, end=" started. ")
    try:
        cfg_t = ini2dict(cfg)
    except MissingMandatoryValue as e:
        lf.error(standard_error_info(e))
        raise Ex_nothing_done()
    except Exception:
        lf.exception("startup error")

    sugar_expand_m(cfg_t["input"])
    sugar_condense_lim_date(cfg_t["input"])

    # Resolve output paths: raw_db_path, text_path, not_joined_db_path, db_path.
    # PathLayout operates on the original DictConfig; copy results to cfg_t["out"].
    try:
        paths.PathLayout.from_cfg(cfg.input, cfg.out).apply_to_cfg(cfg.out)
        for entity_name in paths.PathLayout.SCHEMA:
            key = f"{entity_name}_path"
            if (resolved := getattr(cfg.out, key, None)) is not None:
                cfg_t["out"][key] = resolved
    except (ValueError, OSError) as e:
        lf.debug("PathLayout resolution skipped: {}", e)

    return cfg_t
