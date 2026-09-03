"""Recursive discovery for tcm — rglob + archive, reused via meta_finder.

This module holds the ``rglob`` / archive logic so :mod:`tcm.csv_load`
stays thin and under the 1500-line limit (AGENTS.md).  Archive members are
stored as composite :class:`Path` ``archive.as_posix() + "/" + rel.as_posix()``
(``"/"`` not ``"!"``) — readers split at the archive suffix and delegate to
``meta_finder.utils_sys`` without extraction.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path, PurePosixPath

from utils import log_init

from tcm import _constants, format, policy
from tcm.format_loaded import mod_name

lf = log_init.LoggingStyleAdapter(__name__)

# Reuse canonical archive set — mirrors meta_finder.config.extensions_archive.
_ARCHIVE_EXTS = _constants.ARCHIVE_EXTS


def is_archive_composite(path: Path) -> bool:
    """Return True when *path* contains an archive suffix (``.zip/.7z``) + ``/``."""
    posix = path.as_posix().lower()
    return any(f"{ext}/" in posix for ext in _ARCHIVE_EXTS)


def split_archive_path(path: Path) -> tuple[Path, PurePosixPath] | None:
    """Split composite ``archive/inner`` path at first archive suffix.

    :param path: composite path ``…/archive.zip/inner/file.txt``.
    :return: ``(archive_path, rel_posix)`` or ``None`` when not composite.
    """
    posix = path.as_posix()
    lower = posix.lower()
    for ext in sorted(_ARCHIVE_EXTS):
        needle = ext + "/"
        idx = lower.find(needle)
        if idx != -1:
            archive_str = posix[: idx + len(ext)]
            rel_str = posix[idx + len(ext) + 1 :]
            if rel_str:
                return Path(archive_str), PurePosixPath(rel_str)
    return None


def search_csv_files_recursive(
    parent: Path,
    ptn: re.Pattern,
    *,
    trigger: Path | str | None = None,
) -> dict[tuple[str, int], list[Path]]:
    """Recursive discovery (text + archive text, H5 when policy allows) grouped by probe.

    Shallow miss fallback: filtered via ``meta_finder.find_device_dirs``
    (``ptn_device_dir_search`` + ``is_valid_device_dir`` + ``ptn_dir_exclude``).
    ``DOC/GRIDDING`` etc. are excluded because they fail the device filter.
    Per valid ``_raw`` enumerates via :func:`meta_finder.file_finder.find_raw_files_recursive`
    (zip+7z via ``utils_sys.gen_from_archive``). No unfiltered ``rglob``.

    :param parent: directory to search recursively (already resolved).
    :param ptn: compiled regex applied to ``name`` / ``@``-stripped name.
    :param trigger: original user-supplied path for traceability.
    :return: ``{(model, number): [Path, …]}`` grouped, deduped (``@`` wins).
    :raises FileNotFoundError: when no probe files match (includes trigger).
    """
    parent = Path(parent).expanduser().resolve()
    _trigger = trigger if trigger is not None else parent
    if not parent.exists():
        raise FileNotFoundError(f"No input files found matching {parent} (trigger={_trigger})")

    lf.debug("Recursive scan: trigger={} parent={} ptn=/{} /", _trigger, parent, ptn.pattern)

    try:
        from meta_finder.file_finder import find_device_dirs, find_raw_files_recursive, is_valid_device_dir
    except Exception as exc:  # pragma: no cover — meta_finder not installed
        raise FileNotFoundError(f"No input files found matching {parent} (recursive: meta_finder unavailable: {exc}) from trigger={_trigger}") from exc

    entries: list[tuple[Path, PurePosixPath]] = []
    if parent.is_dir() and parent.name.lower() == _constants.RAW_DIR_NAME.lower():
        # Single _raw anchor — direct enumeration, no device_dirs fan-out
        entries = find_raw_files_recursive(parent, ptn)
    else:
        # Parent/cruise — filtered device dirs only, no unfiltered rglob
        try:
            device_dirs = find_device_dirs(parent)
        except Exception as exc:
            lf.debug("find_device_dirs failed for {}: {}", parent, exc, exc_info=True)
            device_dirs = []
        if device_dirs:
            for dd in device_dirs:
                raw = Path(dd) / _constants.RAW_DIR_NAME
                if raw.is_dir():
                    before = len(entries)
                    entries.extend(find_raw_files_recursive(raw, ptn))
                    if len(entries) == before:
                        lf.info("Anchor {} has no files matching /{}/ — skip (trigger={})", raw, ptn.pattern, _trigger)
                else:
                    lf.info("Device dir {} has no _raw — skip (trigger={})", dd, _trigger)
            lf.debug(
                "Cruise-scoped scan via meta_finder: {} device dirs → {} entries (trigger={})",
                len(device_dirs),
                len(entries),
                _trigger,
            )
        elif is_valid_device_dir(parent) and (Path(parent) / _constants.RAW_DIR_NAME).is_dir():
            # Parent itself is a device dir (e.g. ``251201_ABP64@i,t-chain``)
            raw = Path(parent) / _constants.RAW_DIR_NAME
            entries = find_raw_files_recursive(raw, ptn)
        else:
            raise FileNotFoundError(f"No input files found matching {parent} (recursive, no device dirs) from trigger={_trigger}")
    # Gate H5/NC by effective policy (meta_finder may still return them).
    allowed = policy.effective_data_exts()
    # Keep only allowed data extensions; archives already filtered to text members.
    entries = [(da, rel) for da, rel in entries if rel.suffix.lower() in allowed]
    if not entries:
        raise FileNotFoundError(f"No input files found matching {parent} (recursive) from trigger={_trigger}")

    lf.debug("Recursive scan: {} candidate entries under {} trigger={}", len(entries), parent, _trigger)

    # Build (identity, composite_path, is_corrected) triples
    raw_files: list[tuple[tuple[str, int], Path, bool]] = []
    for dir_archive, rel in entries:
        # Composite path for YAML ``input.path`` — "/" separator, not "!"
        if dir_archive.is_dir():
            composite = dir_archive / rel
        else:
            composite = Path(dir_archive.as_posix() + "/" + rel.as_posix())
        is_corrected = rel.name.startswith("@")
        # probe_from_name handles "@" prefix via [^iw]* consuming it
        identity = format.probe_from_name(rel.stem.lower())
        if identity:
            raw_files.append((identity, composite, is_corrected))
        else:
            lf.debug("Skipping %s — probe_from_name failed for %s", composite, rel.name)

    if not raw_files:
        raise FileNotFoundError(f"No input files found matching {parent} (no probe identity) from trigger={_trigger}")

    # Group by identity; per-file pairing via canonical stem (mod_name normalization)
    by_identity: dict[tuple[str, int], list[Path]] = defaultdict(list)
    corr_stems: set[str] = set()
    for _, f, is_corr in raw_files:
        if is_corr:
            # For archive composites, mod_name should see inner filename only
            name_for_mod = split_archive_path(f)[1].name if is_archive_composite(f) else f.name
            _, p = mod_name(name_for_mod, add_prefix="")
            corr_stems.add(p.stem.lower())

    for identity, f, is_corr in raw_files:
        if is_corr:
            by_identity[identity].append(f)
        else:
            name_for_mod = split_archive_path(f)[1].name if is_archive_composite(f) else f.name
            _, p = mod_name(name_for_mod, add_prefix="")
            if p.stem.lower() not in corr_stems:
                by_identity[identity].append(f)

    n_suppressed = sum(1 for _, _, is_corr in raw_files if not is_corr) - sum(
        1
        for ff in by_identity.values()
        for f in ff
        if not (split_archive_path(f)[1].name if is_archive_composite(f) else f.name).startswith("@")
    )
    lf.info(
        "Files for {:d} probe{:s} found ({:d} raw suppressed by corrected counterparts) via recursive scan (trigger={}):\n{:s}",
        len(by_identity),
        "" if len(by_identity) == 1 else "s",
        n_suppressed,
        _trigger,
        "\n".join(
            "{}{}: {}".format(m, n, ", ".join(f.as_posix() for f in ff)) for (m, n), ff in by_identity.items()
        ),
    )
    return dict(by_identity)
