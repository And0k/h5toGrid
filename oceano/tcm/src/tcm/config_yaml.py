"""YAML config management for probe-specific processing parameters.

Saves, loads, and validates per-file YAML configs in ``cfg_proc/run/``.
"""

from collections.abc import Iterator, Mapping, MutableMapping, Sequence
from datetime import datetime
from itertools import chain
from pathlib import Path, PurePath
from typing import Any

from omegaconf import OmegaConf

from tcm import schema, _constants, csv_load, format, metadata, paths, policy, to_omegaconf, utils2init
from tcm.incl_calc.coefs import get_coefs_from_cfg

lf = utils2init.LoggingStyleAdapter(__name__)


def has_run_yamls(dir_run: Path) -> bool:
    """Return True when ``cfg_proc/run/`` contains at least one YAML."""
    return dir_run.is_dir() and any(dir_run.glob("*.yaml"))


try:
    from ruamel.yaml import YAML
except ImportError as e:
    print(f"{e}: ruamel lib is required for yaml saving in consistent format")


def _ry(write: bool = True) -> YAML:
    """Pre-configured YAML instance for reading/writing run configs.

    :param write: If True, set block style with ``default_flow_style=False``
        and ``allow_unicode=True`` for human-readable output.  Read-only
        callers can pass ``write=False`` to skip style setup.
    """
    ry = YAML(typ="safe", pure=True)
    if write:
        ry.default_flow_style = False
        ry.allow_unicode = True
        ry.preserve_quotes = True
    return ry


def get_existed_cfgs(
    dir_cfgs: Path,
    glob: str = "*.yaml",
) -> dict[str, list[str]]:
    """Resolve config mapping ``{pcid: [stem, …]}`` from *dir_cfgs*.

    Each YAML file's stem is used directly (no timestamp extraction).
    Across different source files for the same probe (pcid), **all** are
    kept — multirun processes each independently.

    :param dir_cfgs: directory containing YAML config files (``cfg_proc/run/``).
    :param glob: glob pattern for YAML files.
    :returns: ``{pcid: [stem_sorted, …]}``.
    """
    result: dict[str, list[str]] = {}
    for f in dir_cfgs.glob(glob):
        stem = f.stem
        # Derive pcid from stem via probe_from_name (strips @ prefix and -comment suffix)
        identity = format.probe_from_name(format.stem_to_pcid(stem).lower())
        if identity:
            pcid = format.pcid_from_parts(model=identity[0], number=identity[1])
        else:
            pcid = format.stem_to_pcid(stem)  # fallback: raw stem without @ or -comment
        result.setdefault(pcid, []).append(stem)
    # Sort stems within each pcid for deterministic order
    for stems in result.values():
        stems.sort()
    return result


def sync_yamls_devmeta_and_hydra(dev_dir, dir_cfgs, cfgs: dict[str, list[str]]):
    """Load date ranges from ``info_devices.yaml/.json`` and update ``time_ranges`` in hydra configs.

    For each probe (pcid), iterates all its config stems.  When multiple configs exist,
    synchronises start time from the first config's ``time_ranges[0]`` and end time from
    the last config's ``time_ranges[-1]`` to the metadata file — using the minimum start
    and maximum end across sources.

    :param dev_dir: directory to search for ``info_devices`` metadata file.
    :param dir_cfgs: directory containing YAML config files.
    :param cfgs: ``{pcid: [cfg_stems, …]}`` from :func:`get_existed_cfgs`.
    """
    lf.info('Loading date range from "info_devices" metadata file')
    all_stems = [s for stems in cfgs.values() for s in stems]
    pcids = [
        format.pcid_from_parts(**format.parse_name(format.stem_to_pcid(v))).replace("_", "")
        for v in all_stems
    ]
    # EAFP: missing info_devices.yaml is benign — no metadata to sync, return cleanly
    # (was the silent cause of "no time_ranges recorded": raised FileNotFoundError
    # before reaching the try block, aborting the whole pipeline before any write).
    try:
        devmeta_path = metadata.get_path_in_parents(dev_dir, "info_devices.yaml", "info_devices.json")
        meta_arrays = metadata.load_file_meta(devmeta_path)
        device_info = metadata.extract_devices_info(meta_arrays, pcids)
    except FileNotFoundError:
        lf.debug("No info_devices.yaml/.json in {} — skipping time_ranges sync", dev_dir)
        return
    except Exception:
        lf.warning(
            'Failed to load or parse "info_devices" metadata from {} — skipping time_ranges sync',
            dev_dir,
            exc_info=True,
        )
        return

    ry = _ry()

    if not any(str_time_ranges_devmeta_all := {pcid: v["r"] for pcid, v in device_info.items() if "r" in v}):
        lf.info("No time records in metadata file")
        return

    try:
        for pcid, stems in cfgs.items():
            pcid_key = pcid.replace("_", "")
            if not (str_time_ranges_devmeta_pcid := str_time_ranges_devmeta_all.get(pcid_key)):
                lf.debug("  {}: no time ranges in metadata", pcid)
                continue

            time_ranges_devmeta = [
                datetime.fromisoformat(t).strftime("%Y-%m-%dT%H:%M:%S") for t in str_time_ranges_devmeta_pcid
            ]

            updated_stems: list[str] = []
            kept_stems: list[str] = []
            broader_stems: dict[str, list[str]] = {}  # stem → existing time_ranges

            for stem in stems:
                cfg_path = (dir_cfgs / stem).with_suffix(".yaml")
                try:
                    cfg_cur = ry.load(cfg_path)
                except Exception:
                    lf.warning("  Skipping {} (load error)", stem, exc_info=True)
                    continue
                if tr_existing := (cfg_cur or {}).get("input", {}).get("time_ranges"):
                    kept_stems.append(stem)
                    if tr_existing[0] < time_ranges_devmeta[0] or tr_existing[-1] > time_ranges_devmeta[-1]:
                        broader_stems[stem] = tr_existing
                    continue
                cfg_cur["input"]["time_ranges"] = time_ranges_devmeta
                ry.dump(cfg_cur, stream=cfg_path)
                updated_stems.append(stem)

            # Log what info_devices provides for this probe
            lf.info(
                "  {}: [{}, {}] in info_devices metadata file",
                pcid,
                time_ranges_devmeta[0],
                time_ranges_devmeta[-1],
            )

            if updated_stems:
                lf.info("    written to {}", ", ".join(f"{s}.yaml" for s in updated_stems))
            if kept_stems:
                if broader_stems:
                    lf.warning(
                        "  found: {} - broader than in metadata file",
                        ", ".join(f"[{tr[0]}, {tr[-1]}] in {s}.yaml" for s, tr in broader_stems.items()),
                    )
                else:
                    lf.debug("    already configured: {}", ", ".join(f"{s}.yaml" for s in kept_stems))
    except Exception:
        lf.exception('Date range job from "info_devices" metadata file failed')


def _discover_tables(path: Path, table_pattern: str) -> list[str]:
    """List groups/tables in HDF5 or NC file matching *table_pattern*.

    Patterns use glob semantics (same as text-file search in
    :func:`csv_load.search_csv_files`): ``*`` matches any characters,
    ``?`` matches one character, literal dots are escaped.
    Example: ``incl*`` matches both ``incl.05`` and ``incl_p05``.

    Returns: bare group names (no leading ``/``).
    Raises :exc:`ImportError` when the needed backend is not installed.
    """
    import re

    from tcm.csv_load import _glob_to_regex

    re_pattern = re.compile(_glob_to_regex(table_pattern))
    suffix = path.suffix.lower()
    if suffix in _constants._EXT_HDF5:
        if not _constants.TABLES_AVAILABLE:
            raise ImportError("pytables (tables) required to read HDF5 files — install or use NC/CSV input")
        import pandas as pd

        with pd.HDFStore(str(path), mode="r") as s:
            return [k.lstrip("/") for k in s.keys() if re_pattern.fullmatch(k.lstrip("/"))]
    if suffix in _constants._EXT_NC:
        policy.io().require_nc("reading NC4 groups")
        with _constants._h5py.File(path, "r") as f:
            return [k for k in f.keys() if re_pattern.fullmatch(k)]
    return []


def prep_cfg_for_probe(
    pcid: str,
    cfg_in_for_probes: Mapping[str, Any],
    cfg_in_common: Mapping[str, Any],
    cfg: Mapping[str, Any],
    path_csv: Path | None = None,
) -> MutableMapping[str, Any]:
    """Build probe-specific config with coefficients.

    Differences from legacy ``cur_cfg = legacy.incl_calc.coefs.prep_cfg_for_probe``:
    - No HDF5 raw-DB-as-coefs-source logic (handled separately).
    - coefs_paths chain: explicit ``coefs_path`` → class default only.

    :param pcid: Probe output Column ID (e.g. ``"i_01"``).
    :param cfg_in_for_probes: per-probe overrides keyed by pcid.
    :param cfg_in_common: input config common to all probes.
    :param cfg: top-level config dict (``cfg["input"]``, ``cfg["out"]``, ``cfg["filter"]``).
    :param path_csv: if set, overrides ``cfg1["input"]["path"]`` with the corrected CSV path.
    :return: ``cfg1`` dict with keys ``input``, ``out``, ``filter``, and loaded coefs.
    """

    cfg_in = {**cfg_in_common.copy(), **cfg_in_for_probes.get(pcid, {})}

    # Build coefs_paths: explicit coefs_path → class default → yaml_export dir.
    # The yaml_export fallback lets ``dist/tcm_clc_txt`` packaging (without the
    # bundled ``calibration.h5`` file) load coefs silently from exported YAMLs.
    cfg_in["coefs"] = get_coefs_from_cfg(cfg_in, pcid)

    # Override path with corrected CSV path if provided
    if path_csv is not None:
        cfg_in["path"] = path_csv

    # Expand glob "incl*" to the concrete raw table name for this probe
    if cfg_in.get("tables") and cfg_in["tables"][0] == "incl*":
        cfg_in["tables"] = [format.pcid_to_raw_name(pcid)]

    return {
        "input": cfg_in,
        "out": dict(cfg["out"]),
        "filter": dict(cfg["filter"]),
    }


def gen_metadata(
    cfg: MutableMapping[str, Any], input_paths: Sequence[Path], cfg_in_for_probes: dict = {}
) -> Iterator[tuple[dict[str, dict[str, Any]], tuple[bool, str, None]]]:
    """
    Yield per-probe metadata (config + edge time rows) for YAML export in the xarray pipeline.

    CSV mode only — HDF5 mode raises :exc:`NotImplementedError`.

    File pairing (corrected ``@``-prefixed over raw) and pcid grouping are
    handled internally by :func:`csv_load.load_from_csv_gen` — no separate
    ``discover_probes`` step needed.

    Per-file overrides come from the run YAML itself (``@package _global_``)

    :param cfg: top-level configuration dict.  ``cfg["input"]`` must contain ``path``,
        ``tables``, etc.  ``cfg["out"]["dt_bins"]`` and ``cfg["filter"]`` are also read.
    :param input_paths: resolved list of input paths (from :func:`init_file_names`).
    :param incl_calc_kwargs: forwarded (unused in metadata-only mode).
    :param cfg_in_for_probes: per-probe overrides if need
    :yields: ``(cfg1, (False, pcid, None))`` where ``cfg1`` is a probe-specific config dict
        with coefficients and optional ``time_ranges`` from edge data rows.

    :raises FileNotFoundError: when no probe files match any input path.
    """
    # Convert to plain dict — cfg["input"] may be a DictConfig backed by ConfigIn_InclProc
    # schema, which rejects extra keys like corr_time_mode.  A plain dict accepts them.
    cfg_in_input = cfg["input"]
    cfg_in_common: dict = (
        OmegaConf.to_container(cfg_in_input, resolve=True)
        if OmegaConf.is_config(cfg_in_input)
        else dict(cfg_in_input)
    )

    # HDF5/NC mode: discover table groups in the file
    if Path(cfg["input"]["path"]).suffix.lower() in _constants._EXT_HDF5 | _constants._EXT_NC:
        cfg_in_common["corr_time_mode"] = cfg["input"].get("corr_time_mode", True)
        table_patterns = cfg["input"].get("tables", ["incl*"])
        discovered: list[str] = [
            *chain.from_iterable(
                _discover_tables(path, pattern)
                for path in map(Path, input_paths)
                for pattern in table_patterns
            )
        ]
        if not discovered:
            raise FileNotFoundError(f"No table groups matching {table_patterns} in {input_paths}")
        lf.info("Discovered {} table groups: {}", len(discovered), discovered)
        for tbl in discovered:
            try:
                pcid = format.to_pcid_from_name(tbl)
                cfg1 = prep_cfg_for_probe(pcid, cfg_in_for_probes, cfg_in_common, cfg)
                cfg1["input"]["tables"] = [tbl]
                cfg1["out"]["dt_bins"] = cfg["out"].get("dt_bins", [0, 2, 600, 3600, 7200])
                for del_field in [
                    "tables",
                    "nfiles",
                    "b_del_temp_db",
                    "temp_db_path",
                    "b_incremental_update",
                ]:
                    cfg1["out"].pop(del_field, None)
                cfg1["input"].pop("dt_min_binning_proc", None)
                cfg1["input"].pop("b_insert_separator", None)
                cfg1["input"].pop("cfgFile", None)
                yield cfg1, (False, pcid, None)
            except Exception:
                lf.warning(
                    "Skipping config generation for table {:s} (will use existing config if available)",
                    tbl,
                    exc_info=True,
                )
        return

    # CSV mode: locate corrected CSV files across input_paths
    cfg_in_common["corr_time_mode"] = cfg["input"].get("corr_time_mode", True)

    # Discover + merge CSV file dicts from all input_paths
    merged: dict[tuple, list[Path]] = {}  # ``{(model, number): [paths]}`` dict of discovered file groups
    for p in input_paths:
        for key, files in csv_load.search_csv_files(p).items():
            merged.setdefault(key, []).extend(files)
    if not merged:
        raise FileNotFoundError(f"No input files found from {input_paths}")
    lf.info("Discovered {} probes (from {} data files)", len(merged), ",".join(str(s) for s in input_paths))

    # Load edge rows
    # Internally handles: stem grouping, corrected/raw pairing
    # — all the pairing logic that the old discover_probes() performed explicitly.
    cfg_merged = {**csv_load.cfg_default["in"], **cfg_in_common}
    for df_raw_edges, (ipid, pcid, path_csv) in csv_load.load_from_csv_gen(
        csv_files_dict=merged,
        cfg_in=cfg_merged,
        return_="first_last_row",
    ):
        try:
            # Configuration with coefficients for current input pcid
            cfg1 = prep_cfg_for_probe(pcid, cfg_in_for_probes, cfg_in_common, cfg, path_csv=path_csv)
            if df_raw_edges is not None:
                cfg1["input"]["time_ranges"] = [dt.isoformat() for dt in df_raw_edges.index]

            # output pcid
            if cfg["out"]["table"]:
                pcid = format.to_pcid_from_name(cfg["out"]["table"])

            cfg1["out"]["dt_bins"] = cfg["out"].get("dt_bins", [0, 2, 600, 3600, 7200])

            # Delete fields not in structured config
            for del_field in [
                "tables",
                "nfiles",
                "b_del_temp_db",
                "temp_db_path",
                "b_incremental_update",
            ]:
                cfg1["out"].pop(del_field, None)
            cfg1["input"].pop("dt_min_binning_proc", None)
            cfg1["input"].pop("b_insert_separator", None)
            cfg1["input"].pop("cfgFile", None)

            yield cfg1, (False, pcid, None)
        except Exception:
            lf.warning(
                "Skipping config generation for probe {:s} (will use existing config if available)",
                pcid,
                exc_info=True,
            )


def save_config_to_yaml(cfg: Mapping[str, Any], input_paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    """Save per-file YAML configs from gen_metadata() to ``cfg_proc/run/``.

    Each source file gets one YAML named ``{yymmdd_hhmm}@{pcid_stem}.yaml``
    when ``input.time_ranges[0]`` is determined from data; otherwise just
    ``@{pcid_stem}.yaml``.  The ``@`` delimiter isolates the date prefix
    (metadata) from the pcid stem — see :func:`format.stem_to_pcid`.
    Each YAML starts with ``# @package _global_`` so it merges into the
    top-level :class:`Config`.

    **Deduplication**: before writing, checks if any existing YAML for the
    same normalized pcid already references a valid (existing) ``input.path``
    file.  If so, the new config is skipped — avoids creating duplicate
    configs that only differ in pid formatting (e.g. ``i_090`` vs ``i90``)
    or comment suffix (e.g. ``@i_p5-press`` vs ``@i_p5``).

    :param cfg: top-level config dict.
    :param input_paths: resolved list of input paths (from :func:`init_file_names`).
    :return: mapping of ``{input_path_str: cfg1_dict}``.
    """
    out_dicts: dict[str, dict[str, Any]] = {}
    # Find config dir: try h5 paths first, fall back to text_path
    in_path = Path(cfg["input"]["path"])
    dir_cfg_proc = paths.find_dir_raw_absolute(in_path) / "cfg_proc" / "run"
    from tcm.cli import safe_cfg_dir

    safe_cfg_dir(dir_cfg_proc)

    ry = _ry()

    def path_representer(dumper, data):
        """Representer for pathlib.Path objects, converting them to strings."""
        return dumper.represent_scalar("tag:yaml.org,2002:str", str(data))

    ry.representer.add_multi_representer(PurePath, path_representer)

    # Build set of pcids that already have a valid config (input.path exists).
    # Uses the same normalization as get_existed_cfgs: stem_to_pcid → probe_from_name → pcid_from_parts.
    existing_valid: set[str] = set()
    for yaml_file in dir_cfg_proc.glob("*.yaml"):
        try:
            with yaml_file.open(encoding="utf-8") as fp:
                cfg_yaml = ry.load(fp)
            cfg_path = (cfg_yaml or {}).get("input", {}).get("path")
            if cfg_path and Path(cfg_path).expanduser().is_file():
                stem = yaml_file.stem
                identity = format.probe_from_name(format.stem_to_pcid(stem).lower())
                if identity:
                    existing_valid.add(format.pcid_from_parts(model=identity[0], number=identity[1]))
                else:
                    existing_valid.add(format.stem_to_pcid(stem))
        except Exception:
            continue

    # Iterate per-file metadata (each run YAML is independent)
    for cfg1, (probe_continues, pcid, _) in gen_metadata(cfg, input_paths):
        # Skip if an equivalent config already exists for this normalized pcid
        if pcid in existing_valid:
            lf.debug(
                "{}: skipping config generation — valid config already exists for this probe",
                pcid,
            )
            out_dicts[str(cfg1["input"]["path"])] = cfg1
            continue

        # Date stamp from time_ranges[0] → {yymmdd_hhmm}@ prefix; anything before @ is
        # metadata, not significant for probe identity (see format.stem_to_pcid).
        file_name = "".join(
            (
                [datetime.fromisoformat(t0).strftime("%y%m%d_%H%M")]
                if (time_ranges := cfg1["input"].get("time_ranges")) and (t0 := time_ranges[0])
                else []
            )
            + (
                ["@", source_stem]
                if (source_stem := Path(cfg1["input"]["path"]).stem)[0] != "@"
                else [source_stem]
            )
            + [".yaml"]
        )

        conf_, ignored_keys = to_omegaconf.to_omegaconf_merge_compatible(cfg1, schema.Config)
        lf.debug("Saving {} config: {} to {}", pcid, file_name, dir_cfg_proc)
        if ignored_keys:
            lf.debug('Removed fields "{}" not in Config', ignored_keys)

        with (dir_cfg_proc / file_name).open(encoding="utf8", mode="w") as fp:
            fp.write("# @package _global_\n")
            ry.dump(conf_, stream=fp)

        out_dicts[str(cfg1["input"]["path"])] = cfg1
    return out_dicts


def find_stale_cfgs(
    cfgs_existed: dict[str, list[str]],
    dir_cfgs: Path,
) -> dict[str, list[str]]:
    """Return pcids → stale YAML stems whose ``input.path`` file is missing.

    A config is **not** marked stale when the raw NC log has a ``fileName``
    entry matching the missing source file — the data can still be loaded
    from the raw NC fast-path.

    :param cfgs_existed: ``{pcid: [cfg_stems]}`` from :func:`get_existed_cfgs`.
    :param dir_cfgs: directory containing YAML config files.
    :returns: ``{pcid: [stale_stem, …]}`` — only pcids with at least one stale config.
    """
    ry = _ry(write=False)
    stale: dict[str, list[str]] = {}
    for pcid, stems in cfgs_existed.items():
        for stem in stems:
            yaml_path = dir_cfgs / f"{stem}.yaml"
            if not yaml_path.is_file():
                stale.setdefault(pcid, []).append(stem)
                continue
            try:
                with yaml_path.open(encoding="utf-8") as fp:
                    cfg_yaml = ry.load(fp)
                cfg_path = (cfg_yaml or {}).get("input", {}).get("path")
                if cfg_path and not Path(cfg_path).expanduser().is_file():
                    if _raw_nc_has_source(Path(cfg_path), pcid):
                        lf.debug("Config {} for {}: text absent but raw NC has log entry", stem, pcid)
                        continue
                    lf.debug("Config {} for {} references non-existent {}", stem, pcid, cfg_path)
                    stale.setdefault(pcid, []).append(stem)
            except Exception:
                lf.debug("Skipping stale check for {} (load error)", stem, exc_info=True)
    return stale


def _raw_nc_has_source(source_path: Path, pcid: str) -> bool:
    """Check if the raw NC for *pcid* has a log entry matching *source_path*.

    Resolves ``raw_db_path`` via :class:`paths.PathLayout` and reads the
    ``/{tbl}/logFiles`` group.  Returns ``True`` when the log contains a
    ``fileName`` matching the expected ``{parent_name}/{stem}`` format.
    """
    try:
        layout = paths.PathLayout(source_path)
        raw_nc = layout.raw_db
        if not raw_nc or not raw_nc.exists():
            return False
    except (ValueError, OSError):
        return False
    from tcm._xr.storage import read_nc_log

    tbl = format.pcid_to_raw_name(pcid)
    log = read_nc_log(raw_nc, tbl)
    if log.sizes.get("Date0", 0) == 0:
        return False
    expected = f"{source_path.parent.name}/{source_path.stem}"[-255:]
    return bool((log["fileName"].values == expected).any())


def update_coefs_in_run_yaml(yaml_path: Path, coefs_changed: dict[str, object]) -> None:
    """Merge changed coefficients into existing run YAML under ``input.coefs``.

    Reads the YAML, updates only the given keys, writes back.
    Non-coefs sections (time_ranges, out, filter) are preserved.
    Creates a timestamped backup (``-backupYYMMDD_HHMMSS``) before first
    modification if the YAML already exists and doesn't already have a
    backup marker. Used as noh5 fallback when ``h5py`` is unavailable,
    or to keep the run YAML in sync with computed values (e.g. zeroing Rz).

    :param yaml_path: Path to existing run YAML (created if missing).
    :param coefs_changed: Mapping of coef_name → numpy array/scalar values.
    """
    from datetime import datetime

    from tcm.to_omegaconf import to_omegaconf_compatible_types

    ry = _ry()
    existing: dict[str, Any] = {}
    if yaml_path.exists():
        # Create timestamped backup before first modification
        ts = datetime.now().strftime("%y%m%d_%H%M%S")
        backup = yaml_path.with_stem(f"{yaml_path.stem}-backup{ts}")
        if not backup.exists():
            import shutil

            shutil.copy2(yaml_path, backup)
            lf.info("Backup created: {}", backup.name)

        try:
            with yaml_path.open("r", encoding="utf-8") as f:
                existing = ry.load(f) or {}
        except Exception:
            lf.warning("Could not read {} — creating fresh", yaml_path.name)

    coefs_node = existing.setdefault("input", {}).setdefault("coefs", {})
    for k, v in coefs_changed.items():
        coefs_node[k] = to_omegaconf_compatible_types(v)

    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with yaml_path.open("w", encoding="utf-8") as f:
        f.write("# @package _global_\n")
        ry.dump(existing, f)

    lf.info("Updated coefs {} in {}", sorted(coefs_changed), yaml_path.name)
