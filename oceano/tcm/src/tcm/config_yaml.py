"""YAML config management for probe-specific processing parameters.

Saves, loads, and validates per-file YAML configs in ``cfg_proc/run/``.
"""

from collections.abc import Iterator, Mapping, MutableMapping, Sequence
from datetime import datetime
from itertools import chain
from pathlib import Path, PurePath, PurePosixPath
from typing import Any

from omegaconf import OmegaConf
from utils import log_init

from tcm import _constants, csv_load, format, metadata, paths, policy, schema, to_omegaconf
from tcm.incl_calc.coefs import get_coefs_from_cfg

lf = log_init.LoggingStyleAdapter(__name__)


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
    """Load date ranges from ``info_devices.yaml/.json`` and update ``time_ranges``.

    For each probe (pcid), iterates all its config stems.  When multiple configs exist,
    synchronises start time from the first config's ``time_ranges[0]`` and end time from
    the last config's ``time_ranges[-1]`` to the metadata file — using the minimum start
    and maximum end across sources.

    Bidirectional fill: if device metadata provides ``time_range`` [6,7] and the run
    YAML's ``input.time_ranges`` is missing or has <2 elements, absent ends are
    filled from metadata (existing elements are never overwritten).

    :param dev_dir: directory to search for ``info_devices`` metadata file.
    :param dir_cfgs: directory containing YAML config files.
    :param cfgs: ``{pcid: [cfg_stems, …]}`` from :func:`get_existed_cfgs`.
    :returns: ``SyncResult`` ``{stem: {status, meta_tr, existing_tr?}}`` or ``None``
        when no metadata file / no records. ``status`` in ``written, kept, broader``.
    """
    lf.info('Loading date range from "info_devices" metadata file')
    all_stems = [s for stems in cfgs.values() for s in stems]
    pcids = [
        format.pcid_from_parts(**format.parse_name(format.stem_to_pcid(v))).replace("_", "")
        for v in all_stems
    ]
    try:
        devmeta_path = metadata.get_path_in_parents(dev_dir, "info_devices.yaml", "info_devices.json")
        meta_arrays = metadata.load_file_meta(devmeta_path)
        device_info = metadata.extract_devices_info(meta_arrays, pcids)
    except FileNotFoundError:
        lf.debug("No info_devices.yaml/.json in {} — skipping time_ranges sync", dev_dir)
        return None
    except Exception:
        lf.warning(
            'Failed to load or parse "info_devices" metadata from {} — skipping time_ranges sync',
            dev_dir,
            exc_info=True,
        )
        return None

    ry = _ry()
    if not any(str_time_ranges_devmeta_all := {pcid: v["r"] for pcid, v in device_info.items() if "r" in v}):
        lf.info("No time records in metadata file")
        return None

    sync_result: dict[str, dict] = {}
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
            broader_stems: dict[str, list[str]] = {}
            for stem in stems:
                cfg_path = (dir_cfgs / stem).with_suffix(".yaml")
                try:
                    cfg_cur = ry.load(cfg_path)
                except Exception:
                    lf.warning("  Skipping {} (load error)", stem, exc_info=True)
                    continue
                tr_existing = (cfg_cur or {}).get("input", {}).get("time_ranges")
                # Bidirectional: device time_range → input.time_ranges for absent ends
                if time_ranges_devmeta[0] and time_ranges_devmeta[-1]:
                    if not tr_existing or len(tr_existing) < 2 or not tr_existing[0] or not tr_existing[-1]:
                        filled = list(tr_existing or [])
                        if not filled or not filled[0]:
                            filled = [time_ranges_devmeta[0]] + (filled[1:] if len(filled) > 1 else [])
                            if len(filled) == 1:
                                filled.append(time_ranges_devmeta[-1])
                        if len(filled) < 2 or not filled[-1]:
                            if len(filled) >= 2:
                                filled[-1] = time_ranges_devmeta[-1]
                            else:
                                filled.append(time_ranges_devmeta[-1])
                        if filled != (tr_existing or []):
                            cfg_cur.setdefault("input", {})["time_ranges"] = filled
                            ry.dump(cfg_cur, stream=cfg_path)
                            tr_existing = filled
                            updated_stems.append(stem)
                            sync_result[stem] = {"status": "written", "meta_tr": time_ranges_devmeta}
                            continue
                if tr_existing:
                    kept_stems.append(stem)
                    broader = (
                        tr_existing[0] < time_ranges_devmeta[0] or tr_existing[-1] > time_ranges_devmeta[-1]
                    )
                    sync_result[stem] = {
                        "status": "broader" if broader else "kept",
                        "meta_tr": time_ranges_devmeta,
                        "existing_tr": list(tr_existing),
                    }
                    if broader:
                        broader_stems[stem] = tr_existing
                    continue
                cfg_cur.setdefault("input", {})["time_ranges"] = time_ranges_devmeta
                ry.dump(cfg_cur, stream=cfg_path)
                updated_stems.append(stem)
                sync_result[stem] = {"status": "written", "meta_tr": time_ranges_devmeta}
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
    return sync_result if sync_result else None


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
    if suffix in _constants.EXT_HDF5:
        if not _constants.TABLES_AVAILABLE:
            raise ImportError("pytables (tables) required to read HDF5 files — install or use NC/CSV input")
        import pandas as pd

        with pd.HDFStore(str(path), mode="r") as s:
            return [k.lstrip("/") for k in s.keys() if re_pattern.fullmatch(k.lstrip("/"))]
    if suffix in _constants.EXT_NC:
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

    :param pcid: Probe output Column ID (e.g. ``"i_01"``).
    :param cfg_in_for_probes: per-probe overrides keyed by pcid.
    :param cfg_in_common: input config common to all probes.
    :param cfg: top-level config dict (``cfg["input"]``, ``cfg["out"]``, ``cfg["filter"]``).
    :param path_csv: if set, overrides ``cfg1["input"]["path"]`` with the corrected CSV path.
    :return: ``cfg1`` dict with keys ``input``, ``out``, ``filter``, and loaded coefs.
    """

    # Merge with deep coefs handling: per-probe coefs.path overrides common
    per_probe = cfg_in_for_probes.get(pcid, {})
    cfg_in = {**cfg_in_common.copy(), **per_probe}
    # Deep merge for coefs dict (path + overrides)
    if "coefs" in cfg_in_common or "coefs" in per_probe:
        merged_coefs = {**(cfg_in_common.get("coefs") or {}), **(per_probe.get("coefs") or {})}
        # OmegaConf containers → plain dict for consistent merging
        if OmegaConf.is_config(merged_coefs):
            merged_coefs = OmegaConf.to_container(merged_coefs, resolve=True)
        cfg_in["coefs"] = merged_coefs

    # Preserve source path before it is consumed by get_coefs_from_cfg
    _coefs_path = (cfg_in.get("coefs") or {}).get("path") if isinstance(cfg_in.get("coefs"), dict) else None

    # Build coefs dict: explicit coefs.path → class default → yaml_export dir.
    # The yaml_export fallback lets ``dist/tcm_proc`` packaging (without the
    # bundled ``calibration.h5`` file) load coefs silently from exported YAMLs.
    loaded_coefs = get_coefs_from_cfg(cfg_in, pcid)
    # Restore source path (write-only attribute for external use, not part of loaded coefs)
    if _coefs_path is not None:
        loaded_coefs["path"] = _coefs_path
    cfg_in["coefs"] = loaded_coefs

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
    cfg: MutableMapping[str, Any],
    input_paths: Sequence[Path],
    cfg_in_for_probes: dict = {},
    eager: bool = True,
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
    if Path(cfg["input"]["path"]).suffix.lower() in _constants.EXT_HDF5 | _constants.EXT_NC:
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
                    "temp_db_path",
                    "overwrite_db",
                    "b_del_temp_db",
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

    _prog_return = OmegaConf.select(cfg, "program.return_") if OmegaConf.is_config(cfg) else (cfg.get("program") or {}).get("return_")
    _is_scan = _prog_return == schema.Return.CFG_FROM_ARGS
    merged: dict[tuple, list[Path]] = {}  # ``{(model, number): [paths]}`` dict of discovered file groups
    for p in input_paths:
        try:
            try:
                discovered = csv_load.search_csv_files(p, trigger=p)
            except TypeError as _te:
                # Test mocks may not accept trigger kw
                if "trigger" in str(_te):
                    discovered = csv_load.search_csv_files(p)  # type: ignore[call-arg]
                else:
                    raise
        except FileNotFoundError:
            # Missing input path (e.g. YAML stem filter with no physical files yet, or
            # non-existent path in cfg) — skip, do not abort discovery; gen_metadata
            # will yield nothing for this path rather than raising mid-iteration.
            lf.debug("search_csv_files skipped for non-existent input_path={}", p, exc_info=True)
            continue
        for key, files in discovered.items():
            merged.setdefault(key, []).extend(files)
    if not merged:
        raise FileNotFoundError(f"No input files found from {input_paths} (trigger={input_paths[0] if input_paths else '?'})")
    lf.info(
        "Discovered {} probes (from {} data files, trigger={}, eager={})",
        len(merged),
        ",".join(str(s) for s in input_paths),
        input_paths[0] if input_paths else "?",
        eager,
    )

    # Partition by storage type: loose text, archive text (composite), loose h5
    try:
        from tcm.search import is_archive_composite, split_archive_path
    except ImportError:

        def is_archive_composite(p: Path) -> bool:  # type: ignore[no-redef]
            return False

        def split_archive_path(p: Path):  # type: ignore[no-redef]
            return None

    merged_loose: dict[tuple, list[Path]] = {}
    merged_archive: dict[tuple, list[Path]] = {}
    merged_h5: dict[tuple, list[Path]] = {}
    for _key, _files in merged.items():
        for _f in _files:
            if is_archive_composite(_f):
                merged_archive.setdefault(_key, []).append(_f)
            elif _f.suffix.lower() in (_constants.EXT_HDF5 | _constants.EXT_NC):
                if not policy.io().h5:
                    lf.debug("Skipping HDF5/NC file {} — {}", _f, policy.io().reason)
                    continue
                merged_h5.setdefault(_key, []).append(_f)
            else:
                merged_loose.setdefault(_key, []).append(_f)

    # ── H5 loose files discovered via recursive search (when dir input) ──
    if merged_h5:
        if not eager:
            # Deferred: stub without opening H5 (row-select will discover tables)
            for _key, _files in merged_h5.items():
                for path_h5 in _files:
                    try:
                        pcid_h5 = format.pcid_from_parts(model=_key[0], number=_key[1])
                        cfg1_h5 = prep_cfg_for_probe(
                            pcid_h5, cfg_in_for_probes, cfg_in_common, cfg, path_csv=path_h5
                        )
                        cfg1_h5["out"]["dt_bins"] = cfg["out"].get("dt_bins", [0, 2, 600, 3600, 7200])
                        for del_field in [
                            "tables",
                            "nfiles",
                            "temp_db_path",
                            "overwrite_db",
                            "b_del_temp_db",
                            "b_incremental_update",
                        ]:
                            cfg1_h5["out"].pop(del_field, None)
                        cfg1_h5["input"].pop("dt_min_binning_proc", None)
                        cfg1_h5["input"].pop("b_insert_separator", None)
                        cfg1_h5["input"].pop("cfgFile", None)
                        yield cfg1_h5, (False, pcid_h5, None)
                    except Exception:
                        lf.warning(
                            "Skipping config generation for h5 {:s} (will use existing config if available)",
                            str(path_h5),
                            exc_info=True,
                        )
        else:
            table_patterns = cfg["input"].get("tables", ["incl*"])
            for _key, _files in merged_h5.items():
                for path_h5 in _files:
                    try:
                        tables: list[str] = []
                        for pat in table_patterns:
                            tables.extend(_discover_tables(path_h5, pat))
                        if not tables:
                            lf.warning("No table groups matching {} in {}", table_patterns, path_h5)
                            continue
                        for tbl in tables:
                            pcid_h5 = format.to_pcid_from_name(tbl)
                            cfg1_h5 = prep_cfg_for_probe(
                                pcid_h5, cfg_in_for_probes, cfg_in_common, cfg, path_csv=path_h5
                            )
                            cfg1_h5["input"]["tables"] = [tbl]
                            cfg1_h5["out"]["dt_bins"] = cfg["out"].get("dt_bins", [0, 2, 600, 3600, 7200])
                            for del_field in [
                                "tables",
                                "nfiles",
                                "temp_db_path",
                                "overwrite_db",
                                "b_del_temp_db",
                                "b_incremental_update",
                            ]:
                                cfg1_h5["out"].pop(del_field, None)
                            cfg1_h5["input"].pop("dt_min_binning_proc", None)
                            cfg1_h5["input"].pop("b_insert_separator", None)
                            cfg1_h5["input"].pop("cfgFile", None)
                            yield cfg1_h5, (False, pcid_h5, None)
                    except Exception:
                        lf.warning(
                            "Skipping config generation for h5 {:s} (will use existing config if available)",
                            str(path_h5),
                            exc_info=True,
                        )

    # ── Archive text files — time/burst via meta_finder without extraction ──
    if merged_archive:
        if not eager:
            # Deferred: stub configs without reading archive members (row-select will fill time_ranges/burst)
            for _key, _files in merged_archive.items():
                for path_csv in _files:
                    try:
                        pcid_arc = format.pcid_from_parts(model=_key[0], number=_key[1])
                        cfg1_arc = prep_cfg_for_probe(
                            pcid_arc, cfg_in_for_probes, cfg_in_common, cfg, path_csv=path_csv
                        )
                        if cfg["out"]["table"]:
                            pcid_arc = format.to_pcid_from_name(cfg["out"]["table"])
                        cfg1_arc["out"]["dt_bins"] = cfg["out"].get("dt_bins", [0, 2, 600, 3600, 7200])
                        for del_field in [
                            "tables",
                            "nfiles",
                            "temp_db_path",
                            "overwrite_db",
                            "b_del_temp_db",
                            "b_incremental_update",
                        ]:
                            cfg1_arc["out"].pop(del_field, None)
                        cfg1_arc["input"].pop("dt_min_binning_proc", None)
                        cfg1_arc["input"].pop("b_insert_separator", None)
                        cfg1_arc["input"].pop("cfgFile", None)
                        yield cfg1_arc, (False, pcid_arc, None)
                    except Exception:
                        lf.warning(
                            "Skipping config generation for probe {:s} (will use existing config if available)",
                            format.pcid_from_parts(model=_key[0], number=_key[1]),
                            exc_info=True,
                        )
        else:
            # TCM-first time_ranges, then meta_finder burst overlay (D3: integrate, do not hide failures)
            try:
                from meta_finder.data_proc_funcs import extract_time_info_from_text_file as _eti_archive

                _has_burst_archive = True
            except ImportError:
                _eti_archive = None  # type: ignore[assignment]
                _has_burst_archive = False

            def _tcm_ranges_from_archive(dir_archive: Path, rel: PurePosixPath, _key: tuple) -> list[str] | None:
                """Extract TCM time_ranges from archive member via temp extraction + csv_load edge path."""
                import tempfile
                import zipfile

                try:
                    suffix = dir_archive.suffix.lower()
                    with tempfile.TemporaryDirectory() as td:
                        td_path = Path(td)
                        extracted: Path | None = None
                        if suffix == ".zip":
                            with zipfile.ZipFile(dir_archive) as zf:
                                data = zf.read(str(rel))
                                extracted = td_path / Path(rel).name
                                extracted.write_bytes(data)
                        elif suffix == ".7z":
                            try:
                                import py7zr  # type: ignore[import]

                                with py7zr.SevenZipFile(dir_archive, mode="r") as zf:
                                    zf.extract(path=td, targets=[str(rel)])
                                    extracted = td_path / rel
                                    if not extracted.is_file():
                                        cand = list(td_path.rglob(Path(rel).name))
                                        extracted = cand[0] if cand else None
                            except Exception:
                                lf.debug("7z extract failed for %s / %s", dir_archive, rel, exc_info=True)
                                return None
                        else:
                            return None
                        if extracted is None or not extracted.is_file():
                            return None
                        tmp_dict: dict[tuple, list[Path]] = {(_key[0], _key[1]): [extracted]}
                        cfg_merged_tmp = {**csv_load.cfg_default["in"], **cfg_in_common}
                        for df_edges, (_ipid, _pcid, _ppath) in csv_load.load_from_csv_gen(
                            csv_files_dict=tmp_dict, cfg_in=cfg_merged_tmp, return_="first_last_row"
                        ):
                            if df_edges is not None and len(df_edges.index) >= 2:
                                return [dt.isoformat() for dt in df_edges.index[:1]] + [
                                    dt.isoformat() for dt in df_edges.index[-1:]
                                ]
                            if df_edges is not None and len(df_edges.index) == 1:
                                iso = df_edges.index[0].isoformat()
                                return [iso, iso]
                            return None
                        return None
                except Exception:
                    lf.debug("TCM archive ranges failed for %s / %s", dir_archive, rel, exc_info=True)
                    return None

            for _key, _files in merged_archive.items():
                for path_csv in _files:
                    try:
                        split = split_archive_path(path_csv)
                        if split is None:
                            continue
                        dir_archive, rel = split
                        averaging_interval = cfg_in_common.get("averaging_interval")
                        if averaging_interval is None:
                            averaging_interval = 2
                        # TCM-first time_ranges
                        tcm_ranges = _tcm_ranges_from_archive(dir_archive, rel, _key)
                        info = None
                        burst_dt = bursts_t = "-"
                        t_st = t_en = None
                        if _has_burst_archive:
                            try:
                                info = _eti_archive(
                                    dir_archive, rel, averaging_interval=averaging_interval
                                )
                                if info is not None:
                                    t_st, t_en, burst_dt, bursts_t = info
                            except Exception:
                                lf.debug("extract_time_info failed for %s / %s", dir_archive, rel, exc_info=True)
                                info = None
                        pcid_arc = format.pcid_from_parts(model=_key[0], number=_key[1])
                        cfg1_arc = prep_cfg_for_probe(
                            pcid_arc, cfg_in_for_probes, cfg_in_common, cfg, path_csv=path_csv
                        )
                        # Prefer TCM ranges; fallback to meta_finder ranges when TCM absent
                        if tcm_ranges is not None:
                            cfg1_arc["input"]["time_ranges"] = tcm_ranges
                            if info is None:
                                lf.debug("TCM ranges for {}: {} (burst unavailable)", pcid_arc, tcm_ranges)
                            elif burst_dt != "-" or bursts_t != "-":
                                lf.info(
                                    "Burst for {}: burst_dt={} bursts_t={} (from {})",
                                    pcid_arc,
                                    burst_dt,
                                    bursts_t,
                                    path_csv.name if hasattr(path_csv, "name") else str(path_csv).rsplit("/", 1)[-1],
                                )
                            else:
                                lf.debug("Burst for {}: continuous (-/-) in {}", pcid_arc, path_csv)
                        elif info is not None:
                            if t_st and t_en:
                                cfg1_arc["input"]["time_ranges"] = [
                                    t.replace(" ", "T") if " " in t else t for t in (t_st, t_en)
                                ]
                            if burst_dt != "-" or bursts_t != "-":
                                lf.info(
                                    "Burst for {}: burst_dt={} bursts_t={} (from {})",
                                    pcid_arc,
                                    burst_dt,
                                    bursts_t,
                                    path_csv.name if hasattr(path_csv, "name") else str(path_csv).rsplit("/", 1)[-1],
                                )
                            else:
                                lf.debug("Burst for {}: continuous (-/-) in {}", pcid_arc, path_csv)
                        else:
                            lf.warning("Time extraction failed for {} from {} (both TCM and meta_finder)", pcid_arc, path_csv)
                        if cfg["out"]["table"]:
                            pcid_arc = format.to_pcid_from_name(cfg["out"]["table"])
                        cfg1_arc["out"]["dt_bins"] = cfg["out"].get("dt_bins", [0, 2, 600, 3600, 7200])
                        for del_field in [
                            "tables",
                            "nfiles",
                            "temp_db_path",
                            "overwrite_db",
                            "b_del_temp_db",
                            "b_incremental_update",
                        ]:
                            cfg1_arc["out"].pop(del_field, None)
                        cfg1_arc["input"].pop("dt_min_binning_proc", None)
                        cfg1_arc["input"].pop("b_insert_separator", None)
                        cfg1_arc["input"].pop("cfgFile", None)
                        yield cfg1_arc, (False, pcid_arc, None)
                    except Exception:
                        lf.warning(
                            "Skipping config generation for probe {:s} (will use existing config if available)",
                            format.pcid_from_parts(model=_key[0], number=_key[1]),
                            exc_info=True,
                        )

    # ── Loose text files — original path via csv_read_gen (handles header detection) ──
    if not merged_loose:
        if not merged_archive and not merged_h5:
            # Nothing left to yield (should have been caught earlier)
            return
        # Only archive/h5 existed — already yielded above
        if not merged_loose:
            return

    if not eager:
        # Deferred: stub configs without reading file edges/burst (filled on row-select)
        for _key, _files in merged_loose.items():
            for path_csv in _files:
                try:
                    pcid = format.pcid_from_parts(model=_key[0], number=_key[1])
                    cfg1 = prep_cfg_for_probe(pcid, cfg_in_for_probes, cfg_in_common, cfg, path_csv=path_csv)
                    cfg1["out"]["dt_bins"] = cfg["out"].get("dt_bins", [0, 2, 600, 3600, 7200])
                    for del_field in [
                        "tables",
                        "nfiles",
                        "temp_db_path",
                        "overwrite_db",
                        "b_del_temp_db",
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
                        format.pcid_from_parts(model=_key[0], number=_key[1]),
                        exc_info=True,
                    )
        return

    # Load edge rows
    # Internally handles: stem grouping, corrected/raw pairing
    # — all the pairing logic that the old discover_probes() performed explicitly.
    cfg_merged = {**csv_load.cfg_default["in"], **cfg_in_common}
    for df_raw_edges, (ipid, pcid, path_csv) in csv_load.load_from_csv_gen(
        csv_files_dict=merged_loose,
        cfg_in=cfg_merged,
        return_="first_last_row",
    ):
        try:
            # Configuration with coefficients for current input pcid
            cfg1 = prep_cfg_for_probe(pcid, cfg_in_for_probes, cfg_in_common, cfg, path_csv=path_csv)
            if df_raw_edges is not None:
                cfg1["input"]["time_ranges"] = [dt.isoformat() for dt in df_raw_edges.index]

            # Burst detection for loose files (same as archive — fills missing metadata, not input)
            try:
                from pathlib import PurePosixPath as _PP

                from meta_finder.data_proc_funcs import extract_time_info_from_text_file as _eti

                _avg = cfg_in_common.get("averaging_interval") or 2
                _info = _eti(path_csv.parent, _PP(path_csv.name), averaging_interval=_avg)
                if _info is not None:
                    _, _, _bdt, _bst = _info
                    if _bdt != "-" or _bst != "-":
                        lf.info("Burst for {}: burst_dt={} bursts_t={} (from {})", pcid, _bdt, _bst, path_csv.name)
                    else:
                        lf.debug("Burst for {}: continuous (-/-) in {}", pcid, path_csv.name)
                else:
                    lf.debug("Burst for {}: no time_info from {}", pcid, path_csv.name)
            except ImportError:
                pass
            except Exception:
                lf.debug("Burst extraction failed for {}", path_csv, exc_info=True)

            # output pcid
            if cfg["out"]["table"]:
                pcid = format.to_pcid_from_name(cfg["out"]["table"])

            cfg1["out"]["dt_bins"] = cfg["out"].get("dt_bins", [0, 2, 600, 3600, 7200])

            # Delete fields not in structured config or which we enforce to be specified explicitly
            for del_field in [
                "tables",
                "nfiles",
                "temp_db_path",
                "overwrite_db",
                "b_del_temp_db",
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


def save_config_to_yaml(
    cfg: Mapping[str, Any], input_paths: Sequence[Path], eager: bool = True
) -> dict[str, dict[str, Any]]:
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
    # Single-anchor: dir_cfg_proc derived from trigger (processing.run enforces one _raw)
    in_path = Path(cfg["input"]["path"])
    lf.info("save_config_to_yaml trigger={} input_paths={}", in_path, input_paths)
    dir_cfg_proc = paths.find_dir_raw_absolute(in_path) / "cfg_proc" / "run"

    ry = _ry()

    def path_representer(dumper, data):
        """Representer for pathlib.Path objects, converting them to strings."""
        return dumper.represent_scalar("tag:yaml.org,2002:str", str(data))

    ry.representer.add_multi_representer(PurePath, path_representer)

    # Build set of input.path strings that already have a valid config.
    # Per-file dedup (not per-pcid) so multiple files of same probe with
    # different dates each get a YAML tab.  Archive composites are valid when
    # the archive file itself exists.
    def _is_valid_input_path(p_str: str) -> bool:
        p = Path(p_str)
        try:
            from tcm.search import is_archive_composite, split_archive_path

            if is_archive_composite(p):
                sp = split_archive_path(p)
                if sp is None:
                    return False
                archive_path, _rel = sp
                return archive_path.is_file()
            return p.expanduser().is_file()
        except Exception:
            try:
                return p.expanduser().is_file()
            except Exception:
                return False

    existing_valid_paths: set[str] = set()
    if dir_cfg_proc.is_dir():
        for yaml_file in dir_cfg_proc.glob("*.yaml"):
            try:
                with yaml_file.open(encoding="utf-8") as fp:
                    cfg_yaml = ry.load(fp)
                cfg_path = (cfg_yaml or {}).get("input", {}).get("path")
                if cfg_path and _is_valid_input_path(str(cfg_path)):
                    existing_valid_paths.add(str(Path(str(cfg_path)).as_posix()))
            except Exception:
                continue

    # Iterate per-file metadata (each run YAML is independent) — eager=False defers file reads to row-select
    for cfg1, (probe_continues, pcid, _) in gen_metadata(cfg, input_paths, eager=eager):
        cfg_path_str = str(Path(str(cfg1["input"]["path"])).as_posix())
        # Per-file dedup: skip only when exact file path already has a valid YAML
        if cfg_path_str in existing_valid_paths:
            lf.debug(
                "{}: skipping config generation — valid config already exists for file {}",
                pcid,
                cfg_path_str,
            )
            out_dicts[str(cfg1["input"]["path"])] = cfg1
            continue

        # Date stamp + pcid + preserved -comment (D5: pcid canonical, comment from source stem)
        # Source stem extraction handles archive composites (inner name) and @ prefix
        _src_path = Path(str(cfg1["input"]["path"]))
        try:
            from tcm.search import is_archive_composite as _is_comp
            from tcm.search import split_archive_path as _split

            if _is_comp(_src_path):
                _sp = _split(_src_path)
                _src_stem = _sp[1].stem if _sp is not None else _src_path.stem
            else:
                _src_stem = _src_path.stem
        except Exception:
            _src_stem = Path(str(cfg1["input"]["path"])).stem
        _stem_no_at = _src_stem.lstrip("@")
        _comment = _stem_no_at.split("-", 1)[1] if "-" in _stem_no_at else ""
        _date_prefix = (
            datetime.fromisoformat(t0).strftime("%y%m%d_%H%M")
            if (time_ranges := cfg1["input"].get("time_ranges")) and (t0 := time_ranges[0])
            else ""
        )
        file_name = f"{_date_prefix + '@' if _date_prefix else '@'}{pcid}{'-' + _comment if _comment else ''}.yaml"

        # OmegaConf schema expects str for path — convert Path (posix for archive composites)
        if isinstance(cfg1["input"].get("path"), Path):
            cfg1["input"]["path"] = cfg1["input"]["path"].as_posix()
        conf_, ignored_keys = to_omegaconf.to_omegaconf_merge_compatible(cfg1, schema.Config)
        _per_file_dir = dir_cfg_proc
        lf.debug("Saving {} config: {} to {}", pcid, file_name, _per_file_dir)
        if ignored_keys:
            lf.debug('Removed fields "{}" not in Config', ignored_keys)

        # Lazy dir creation — only when a YAML is actually written.
        from tcm.cli import safe_cfg_dir

        safe_cfg_dir(_per_file_dir)
        with (_per_file_dir / file_name).open(encoding="utf8", mode="w") as fp:
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

    def _path_exists(p_str: str) -> bool:
        p = Path(p_str)
        try:
            from tcm.search import is_archive_composite, split_archive_path

            if is_archive_composite(p):
                sp = split_archive_path(p)
                if sp is None:
                    return False
                archive_path, _rel = sp
                return archive_path.is_file()
            return p.expanduser().is_file()
        except Exception:
            try:
                return p.expanduser().is_file()
            except Exception:
                return False

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
                if cfg_path and not _path_exists(str(cfg_path)):
                    if _raw_nc_has_source(Path(str(cfg_path).split(".zip/")[0].split(".7z/")[0]), pcid):
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
