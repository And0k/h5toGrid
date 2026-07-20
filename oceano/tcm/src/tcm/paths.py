"""Path resolution — two-layer architecture.

Layer 1 — **Anchor discovery** (stateless functions):
  :func:`find_dir_raw`      — low-level: walk path ancestors for ``_raw/``.
  :func:`find_dir_raw_absolute` — CLI bootstrap: find ``_raw/`` or infer fallback.
  :func:`_infer_proc_dir`   — shared fallback: walk up to digit/inclinometer parent.

Layer 2 — **Output path layout** (:class:`PathLayout`):
  Declarative, lazily-evaluated resolver for SCHEMA entities (``raw_db``, ``db``,
  ``not_joined_db``, ``text``).  Uses Layer 1 primitives internally — never
  re-implements ancestor scanning.
"""

from __future__ import annotations
from functools import cached_property
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List
from omegaconf import OmegaConf, DictConfig

from tcm import _constants, utils2init

lf = utils2init.LoggingStyleAdapter(__name__)

class PathLayout:
    """Declarative, lazily-evaluated path resolver for output paths (Layer 2).

    Uses Layer 1 primitives (:func:`find_dir_raw`, :func:`_infer_proc_dir`)
    for anchor detection — never re-implements ancestor scanning.

    Instead of hard-coding logic for individual entities (e.g., ``get_db_path``),
    this class uses a SCHEMA dictionary to define how to construct any path based
    on structural anchors (``proc_dir``, ``raw_dir``).  Paths are resolved
    generically via ``resolve()`` and accessed lazily as attributes.

    Resolution hierarchy for each entity:
      1. **Absolute path** — user-provided absolute path used as-is.
      2. **Relative path** — resolved against ``proc_dir``.
      3. **Auto-generation** — stem from ``_raw``-anchored directory + SCHEMA suffix.

    Anchor resolution (``_resolve_anchors``):
      - ``_raw/`` in path → ``raw_dir = _raw``, ``proc_dir = parent``.
      - ``.proc``/``.proc_noAvg`` suffix → ``proc_dir = parent``, ``raw_dir = parent/_raw``.
      - Fallback → ``_infer_proc_dir`` walks up to digit/inclinometer ancestor.
    """

    # Schema Definition: entity_name -> (base_anchor_attribute, suffix, use_stem)
    # Entity name equal to an `config.out` name with "_path" suffix stripped
    SCHEMA: Dict[str, Tuple[str, str, bool]] = {
        "raw_db": ("raw_dir", ".raw.nc", True),
        "db": ("proc_dir", ".proc.nc", True),
        "not_joined_db": ("proc_dir", ".proc_noAvg.nc", True),
        "text": ("proc_dir", "", False),  # Directory
    }

    def __init__(self, path_in: Path | str, **user_paths: Any):
        self._path_in = Path(path_in)
        self._user_paths = user_paths
        self._cache: Dict[str, Optional[Path]] = {}

        self._resolve_anchors()

    @classmethod
    def from_cfg(cls, cfg_in: DictConfig, cfg_out: DictConfig) -> "PathLayout":
        """
        Extracting the source paths of the Hydra structured config available in SCHEMA for deferred processing
        """
        path_in = cfg_in.path

        # 2. Собираем user_paths только для ключей из SCHEMA
        user_paths = {
            key_path: val_path
            for key in cls.SCHEMA.keys()
            if (val_path := OmegaConf.select(cfg_out, key_path := f"{key}_path")) is not None
        }

        return cls(path_in=path_in, **user_paths)

    def apply_to_cfg(self, cfg_out: DictConfig):
        """
        Разрешает все ленивые пути и записывает их обратно в OmegaConf.
        """
        OmegaConf.set_struct(cfg_out, False)  # Разрешаем модификацию конфига
        for entity_name in self.SCHEMA.keys():
            resolved_path = self.resolve(entity_name)
            if resolved_path is not None:
                setattr(cfg_out, f"{entity_name}_path", resolved_path)
        OmegaConf.set_struct(cfg_out, True)  # Возвращаем строгий режим

    def _resolve_anchors(self):
        """Determines the absolute layout anchors (``proc_dir`` and ``raw_dir``).

        Resolution order:
        1. If *path_in* is relative, anchor it to the first absolute user path.
        2. ``find_dir_raw`` → ``_raw/`` found: ``raw_dir = _raw``, ``proc_dir = parent``.
        3. ``.proc``/``.proc_noAvg`` suffix: ``proc_dir = parent``, ``raw_dir = parent/_raw``.
        4. Fallback: ``_infer_proc_dir`` walks up to digit/inclinometer ancestor.
        """
        path_in = self._path_in

        # 1. If path_in is relative, find an absolute path among user_paths to anchor it
        if not path_in.is_absolute():
            for val in self._user_paths.values():
                if (p := Path(val)).is_absolute():
                    path_in = p.parent / path_in
                    break

        path_in = path_in.resolve()

        # 2. Detect anchors via shared primitives (case-insensitive _raw matching)
        if (raw := find_dir_raw(path_in)) is not None:
            self.raw_dir = raw
            self.proc_dir = raw.parent
        elif path_in.suffixes and any(sfx in path_in.suffixes for sfx in [".proc", ".proc_noAvg"]):
            self.proc_dir = path_in.parent
            self.raw_dir = self.proc_dir / _constants.RAW_DIR_NAME
        else:
            self.proc_dir = _infer_proc_dir(path_in)
            self.raw_dir = self.proc_dir / _constants.RAW_DIR_NAME

    @cached_property
    def stem(self) -> str:
        """Auto-detects the project stem from the proc_dir structure."""
        p = self.proc_dir
        name = p.name
        parent_name = p.parent.name

        raw_stem = (
            name
            if name and name[0].isdigit()
            else parent_name
            if parent_name and (parent_name[0].isdigit() or name.startswith("inclinometer"))
            else name
        )

        # Strip everything after first '_' or '@'
        return raw_stem.replace("@", "_", 1).split("_")[0]

    def resolve(self, entity_name: str) -> Optional[Path]:
        """
        Generic lazy resolver for any entity defined in SCHEMA.
        - Absolute user paths are used as-is.
        - Relative user paths are resolved against `proc_dir`.
        - "auto", True, or None triggers auto-generation from Schema.
        """
        if entity_name in self._cache:
            return self._cache[entity_name]

        if entity_name not in self.SCHEMA:
            raise ValueError(f"Unknown path entity: '{entity_name}'")

        user_val = self._user_paths.get(f"{entity_name}_path")
        result = None

        # 1. Explicit absolute path
        if user_val and Path(user_val).is_absolute():
            result = Path(user_val)

        # 2. Explicit relative path
        elif user_val and user_val not in (True, "auto"):
            result = self.proc_dir / str(user_val)

        # 3. Auto-generation
        else:
            base_attr, suffix, use_stem = self.SCHEMA[entity_name]
            base_dir = getattr(self, base_attr)

            if use_stem:
                result = (base_dir / self.stem).with_suffix(suffix)
            else:
                # Fallback for schema items that don't use the auto-stem
                result = base_dir / suffix

        # --- Entity-specific Edge Cases (kept minimal) ---
        if entity_name == "not_joined_db":
            # If the input file itself is noAvg, it becomes the not_joined_db
            if self._path_in.suffixes and ".proc_noAvg" in self._path_in.suffixes:
                result = self._path_in if self._path_in.is_absolute() else self.proc_dir / self._path_in

        self._cache[entity_name] = result
        return result

    # --- Dynamic Attribute Access ---
    def __getattr__(self, name: str) -> Any:
        if name in self.SCHEMA:
            return self.resolve(name)
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def select_input_db(self, dt_bins: List[float], dt_min_binning_proc: float) -> Path:
        """Select the correct input DB based on binning parameters.

        Checks ``.nc`` first, then falls back to ``.h5`` for legacy files.
        """
        need_load_counts = True
        p_sfx = self._path_in.suffixes[-2:-1]

        if p_sfx and p_sfx[0].startswith(".proc"):
            need_load_counts = any((not bin) or (bin <= dt_min_binning_proc) for bin in dt_bins)

        paths_to_check = ([] if need_load_counts else [self.not_joined_db]) + [self.raw_db]

        # Also add .h5 fallback variants for each path
        h5_fallbacks = [
            p.with_suffix(p.suffix.replace(".nc", ".h5"))
            for p in paths_to_check
            if p and p.suffix == ".nc"
        ]
        all_paths = paths_to_check + h5_fallbacks

        for path in all_paths:
            if path and path.is_file():
                return path

        raise FileNotFoundError(f"Not found stored data: {[p for p in all_paths if p]}")


def find_dir_raw(path_in: Path, raw_dir_name: Optional[str] = None) -> Optional[Path]:
    """Return *path_in* or its nearest ancestor whose name equals :data:`RAW_DIR_NAME` (case-insensitive).

    :param path_in: The raw-dir itself or any descendant.
    :param raw_dir_name: Override for the marker name (defaults to :data:`_constants.RAW_DIR_NAME`).
    :return: The matched raw-dir path, or ``None`` if no match.
    """
    marker = (raw_dir_name or _constants.RAW_DIR_NAME).lower()
    if path_in.name.lower() == marker:
        return path_in
    return next((p for p in path_in.parents if p.name.lower() == marker), None)


def _infer_proc_dir(path_in: Path) -> Path:
    """Infer the processing directory when no ``_raw/`` ancestor exists.

    Walks up from *path_in* looking for a directory whose name starts with a
    digit (e.g. ``250101_experiment``) or ``"inclinometer"``.  Falls back to
    ``path_in.parent`` when no such ancestor is found.

    :param path_in: Resolved absolute input path.
    :return: Inferred ``proc_dir``.
    """
    return next(
        (parent for parent in path_in.parents
         if parent.name and (parent.name[0].isdigit() or parent.name.startswith("inclinometer"))),
        path_in.parent,
    )


def find_dir_raw_absolute(path_in: Path) -> Path:
    """Find the ``_raw`` data directory and auto-infer when not standard.

    The ``_raw`` directory is the anchor for all relative paths:
    ``cfg_proc/run/`` (configs), ``cfg_proc/log/`` (Hydra logs),
    ``diagnostics/``, etc.

    Resolution order:
      1. ``find_dir_raw`` → ``_raw/`` ancestor found → return it.
      2. Not found → warn, return *path_in* if directory else its parent.

    Always returns a valid directory — never ``None``.

    .. note::
       This is the **CLI bootstrap** anchor (``os.chdir``, ``cfg_proc/`` search).
       :class:`PathLayout` uses its own ``_infer_proc_dir`` fallback for output
       paths — they may differ when ``_raw/`` is absent.
    """
    if (dir_raw := find_dir_raw(path_in)) is not None:
        return dir_raw
    lf.warning(
        "Not standard input path {}: raw data should be in {} subdirectory."
        " Using deepest dir as anchor…",
        path_in,
        _constants.RAW_DIR_NAME,
    )
    return path_in if path_in.is_dir() else path_in.parent
