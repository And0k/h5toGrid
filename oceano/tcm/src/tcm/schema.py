"""Structured configuration for the xr-native pipeline.

Defines the dataclass schema and registers it with Hydra's ConfigStore.
The top-level ``Config`` dataclass is the single source of truth for all
config groups (``input``, ``out``, ``filter``, ``program``, ``proc``).

Stage semantics
---------------
- **input** — load-stage windowing + DROP: ``time_ranges``, raw-column bounds
  ``min``/``max``, time-correction knobs.
- **filter** — process-stage NaN-out: threshold-flagging on computed columns
  (preserves row count, sets values to NaN).  Calibration adds typed despike
  sub-configs ``ConfigFilterComponent`` → ``ConfigFilterChannel`` →
  ``ConfigFilterCalib(ConfigFilter_InclProc)``.
- **proc** — optional per-entry-point processing parameters (calib: ``calib``,
  spectrum: ``spectrum``).  Processing entry has no ``proc`` group.

Legacy types (``ConfigMultiIn_InclProc``, ``ConfigOutSimple``, etc.) live in
``_dask_legacy/cfg_compat.py``.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf


class Return(StrEnum):
    """``program.return_`` constants — controls how far ``run_processing`` runs and what returns.

    Each value stops after a progressively later phase.  Downstream code
    compares against these constants instead of bare string literals.
    Inherits from :class:`str` so ``Return.END == "<end>"`` is ``True``.
    """

    CFG_FROM_ARGS = "<cfg_from_args>"  # config composition only (no I/O)
    GEN_NAMES_AND_LOG = "<gen_names_and_log>"  # config generation, stop
    SAVED_COEFS = "<saved_coefs>"  # coef persistence (YAML+NC), stop
    SAVED_RAW = "<saved_raw>"  # coefs + raw NC save, stop
    SAVED_NOAVG = "<saved_noavg>"  # no-avg processed output, stop
    SAVED_ALL = "<saved_all>"  # all binned NC writes, stop
    END = "<end>"  # full pipeline (default)


class UseH5(StrEnum):
    """Requested binary I/O mode from configuration."""

    AUTO = "auto"
    """Enable if available; skip silently otherwise."""

    OFF = "off"
    """Disable binary I/O; skipped NC operations are logged."""

    PREFER = "prefer"
    """Enable if available; warn and fall back otherwise."""

    REQUIRE = "require"
    """Enable if available; error if unavailable."""


# ---------------------------------------------------------------------------
# Structured configs
# ---------------------------------------------------------------------------


@dataclass
class ConfigInCoefs_InclProc:
    """Calibration coefficients for inclinometer processing."""

    path: Path | None = Path(__file__).with_name("cfg") / "coef" / "calibration.h5"
    Ag: list[list[float]] | None = field(
        default_factory=lambda: [[0.00173, 0, 0], [0, 0.00173, 0], [0, 0, 0.00173]],
    )
    Cg: Annotated[list[float], 3] | None = field(default_factory=lambda: [10, 10, 10])
    Ah: list[list[float]] | None = field(
        default_factory=lambda: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    )
    Ch: Annotated[list[float], 3] | None = field(default_factory=lambda: [10, 10, 10])
    Rz: list[list[float]] | None = field(
        default_factory=lambda: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    )
    kVabs: list[float] | None = field(default_factory=lambda: [10, -10, -10, -3, 3, 70])
    P_t: Annotated[list[list[float]], (3, 3)] | None = None
    P: list[float] | None = field(default_factory=lambda: [0, 1])
    PBattery: list[float] | None = field(default_factory=lambda: [0, 1])
    PTemp: list[float] | None = field(default_factory=lambda: [0, 1])
    azimuth_shift_deg: float | None = 180
    dates: dict[str, str] | None = field(  # struct=False to allow dynamic keys
        default_factory=lambda: OmegaConf.create({}, flags={"struct": False})
    )
    date: str | None = None


@dataclass
class ConfigInCalib_InclProc:
    """Inclinometer calibration coefficients process-stage correction"""

    g0xyz: Annotated[list[float], 3] | None = None
    time_ranges_zeroing: list[str] | None = field(default_factory=list)
    time_ranges_azimuth: list[str] | None = field(default_factory=list)
    coordinates: Annotated[list[float], 2] | None = None
    azimuth_add: float = 0


@dataclass
class ConfigIn_InclProc:
    """Load-stage windowing + DROP + time-correction parameters (input entity).

    ``min`` / ``max`` here apply **at load** — rows outside raw-column bounds
    are **dropped**.  The ``filter`` group's ``min`` / ``max`` are process-stage
    threshold NaN-out on computed columns (``g_minus_1``, ``h_minus_1``, …).

    Sugar fields ``min_date`` / ``max_date`` and ``M`` shorthand in ``min``/``max``
    dicts are **not** in this structured schema — they are merged into source-of-truth
    fields (``time_ranges``, ``Mx``/``My``/``Mz``) at compose time by
    :func:`tcm._xr.filters.expand_m_shorthand` and :func:`utils.init.update_cfg_time_ranges`.
    Override via ``+`` prefix: ``+input.min_date=2024-01-01``.

    ``dt_from_utc``: time shift from UTC in **seconds** (not hours).
    ``corr_time_outlier_threshold_s``: spike/backward threshold in **seconds**.
    ``fs_rounding``: rounding target for sampling frequency estimation
      (e.g. 100 rounds to nearest 1/100 Hz).  0 disables rounding.
    ``dt_hole_warning``: seconds threshold for logging a warning about data gaps
      after load.  ``None`` disables the check.
    ``tables_log``: log group name template for NC storage (default ``{}/logFiles``).
    """

    # ── source identification ──
    path: str | None = None
    tables: list[str] = field(default_factory=lambda: ["incl*"])
    ids: list[str] | None = None
    prefix: str | None = "I*[_0]"
    text_type: str | None = None
    text_line_regex: str | None = None

    # ── data windowing (load-stage DROP) ──
    time_ranges: list[str | None] | None = None  # source of truth for time window
    min: dict[str, float] | None = field(default_factory=dict)  # raw-column DROP lower bound
    max: dict[str, float] | None = field(default_factory=dict)  # raw-column DROP upper bound

    # ── sugar (NOT structured — merged at compose, accepted via +) ──
    # min_date, max_date → merged into time_ranges by update_cfg_time_ranges
    # M in min/max dicts → expanded to Mx/My/Mz by expand_m_shorthand

    # ── time correction ──
    dt_from_utc: int | None = 0  # timezone offset in seconds
    date_to_from: list[Any] | None = None  # sugar → dt_from_utc
    corr_time_mode: bool | str | None = (
        True  # moved from filter: True/'increase'=snap; 'delete_inversions'=backward mask; None/False=skip
    )
    corr_time_outlier_threshold_s: float | None = 0.6  # moved from filter; _s = seconds convention
    dt_interp_between: float | None = 1.5  # gap threshold for interpolation between bursts
    fs_rounding: int | None = 100  # frequency estimation rounding (0=disabled)
    dt_hole_warning: int | None = 600  # warn if max data gap > this seconds; None=skip

    # ── process-stage coefs ──
    coefs: ConfigInCoefs_InclProc | None = field(default_factory=ConfigInCoefs_InclProc)
    max_incl_of_fit_deg: float | None = None
    calc_version: str = "trigonometric(incl)"

    # ── process-stage calibration correction ──
    calib: ConfigInCalib_InclProc | None = field(default_factory=ConfigInCalib_InclProc)

    # ── storage wiring ──
    tables_log: list[str] = field(default_factory=lambda: ["{}/logFiles"])


@dataclass
class ConfigFilter_InclProc:
    """Process-stage NaN-out thresholds (row-preserving flagging).

    ``min`` / ``max`` here apply **at process stage** — values exceeding
    the threshold are set to NaN (rows preserved).  These operate on
    process-computed columns (``g_minus_1`` = |G| − 1, ``h_minus_1`` = |H| − 1,
    and any raw-column overrides inherited via ``Mx``/``My``/``Mz``).

    ``M`` shorthand in ``min``/``max`` dicts is expanded to ``Mx``/``My``/``Mz``
    at compose time (:func:`tcm._xr.filters.expand_m_shorthand`).

    ``bad_p_at_bursts_starts_period``: pandas offset alias (``'1h'``, ``'30min'``)
    for pressure burst NaN-out — nulls first 2 points of each pressure burst period.
    Empty string disables.
    """

    min: dict[str, float] | None = field(default_factory=dict)
    max: dict[str, float] | None = field(
        default_factory=lambda: {"g_minus_1": 1, "h_minus_1": 8},
    )
    bad_p_at_bursts_starts_period: str = ""


@dataclass
class ConfigOut_InclProc:
    """Output parameters for the xr-native pipeline.

    ``db_path``, ``not_joined_db_path``, ``raw_db_path`` are resolved by
    :class:`tcm.paths.PathLayout` at runtime — defaults are ``None``.

    ``tables_log``: log group name template for NC storage (default ``{}/logFiles``).
    ``dt_bins``: binning intervals in seconds (list of int).  0 = no-averaging output.
    ``split_period``: pandas offset alias for splitting output into separate files.
    ``text_path``: output directory for TSV/CSV files.
    ``b_split_by_time_ranges``: split output by ``input.time_ranges`` instead of ``split_period``.

    Previously inherited from ``ConfigOutSimple`` (``_dask_legacy/cfg_compat.py``)
    which carried HDF5-only dead fields in xr context — those move to legacy.
    """

    # ── HDF5/NC output paths ──
    db_path: Path | None = None
    avg_db_path: Path | None = None
    not_joined_db_path: Path | None = None
    raw_db_path: Path | None = None
    table: str = ""
    tables_log: list[str] = field(default_factory=lambda: ["{}/logFiles"])
    overwrite_db: str | None = None
    """NC overwrite mode — controls how existing data is handled.

    ``None`` (default) — extend-only: append new data with current config.
      If existing data has different config, keep and warn.

    ``"export"`` — export-only: block NC/H5 writes, only export TSV.

    ``"splice"`` — always reprocess (splice): replace existing data in
      ``time_ranges`` with new processing.  NEVER trim.

    ``"trim"`` — trim-only: trim existing data to ``time_ranges``.
      NEVER reprocess existing data.  If ``time_ranges`` extends existing
      AND new source data exists, process and append that new data.
    """

    # ── binning ──
    dt_bins: list[int] | None = field(default_factory=lambda: [0, 2, 600, 3600, 7200])
    dt_bins_min_save_text: int | None = 1

    # ── text output ──
    split_period: str = ""
    text_path: Path | None = Path("text_output")
    text_date_format: str = "%Y-%m-%d %H:%M:%S.%f"
    text_columns: list[str] = field(default_factory=list)
    b_split_by_time_ranges: bool = False
    b_all_to_one_col: bool = False
    b_del_temp_db: bool = False
    b_overwrite_text: bool = True


@dataclass
class ConfigProgram:
    """Program behaviour flags.

    :param return_: phase-stopping point — see :class:`Return` for valid values.
    :param log_: log level
    :param verbose_: one_of('CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'),
    :param use_h5: toggle binary (NC/HDF5) persistence.

        ``auto`` (default) — auto-detect from library availability at startup:

        * h5py/netCDF4 importable → binary I/O enabled.
        * h5py/netCDF4 missing    → silent skip (text-only).

        ``off`` — binary I/O disabled by user choice.  Skipped NC operations
        are logged at INFO level.

        ``prefer`` — prefer binary I/O; if unavailable, warn and fall back.

        ``require`` — binary I/O required.  If unavailable, raise
        ``ImportError`` at startup.

        See :class:`policy.IOPolicy` for the resolved runtime state.
    """

    return_: str = Return.END
    b_interact: bool = False
    log: str = ""
    verbose: str = "INFO"
    use_h5: str = UseH5.AUTO


# ---------------------------------------------------------------------------
# Calibration filter — typed per-channel despike (mirrors legacy)
# ---------------------------------------------------------------------------


@dataclass
class ConfigFilterComponent:
    """Apex despike parameters for a channel or axis.

    Mirrors ``_dask_legacy.incl_calibr_hy.ConfigFilterComponent`` exactly.
    ``None`` defaults signal "use the parent's value" during resolution.
    """

    blocks: list[int] | None = field(default_factory=lambda: [21, 7])
    offsets: list[float] | None = field(default_factory=lambda: [1.5, 2])
    std_smooth_sigma: float | None = 4


@dataclass
class ConfigFilterChannel:
    """Per-axis overrides for a single channel (e.g. ``A`` or ``M``).

    Mirrors ``_dask_legacy.incl_calibr_hy.ConfigFilterChannel`` exactly.
    Each axis (``x``, ``y``, ``z``) is an optional ``ConfigFilterComponent``
    whose ``None`` defaults inherit from this channel's parent apex.
    """

    x: ConfigFilterComponent | None = field(
        default_factory=lambda: ConfigFilterComponent(None, None, None),
    )
    y: ConfigFilterComponent | None = field(
        default_factory=lambda: ConfigFilterComponent(None, None, None),
    )
    z: ConfigFilterComponent | None = field(
        default_factory=lambda: ConfigFilterComponent(None, None, None),
    )


@dataclass
class ConfigFilterCalib(ConfigFilter_InclProc):
    """Calibration filter — inherits process-stage NaN-out + adds typed despike.

    Mirrors ``_dask_legacy.incl_calibr_hy.ConfigFilter`` exactly.
    The apex ``blocks`` / ``offsets`` / ``std_smooth_sigma`` apply to all
    channels; ``A`` / ``M`` override per channel; ``no_works_noise`` reserves
    the `is_works()` noise threshold (wire deferred — see ``dtdr-todo.md`` B).
    """

    blocks: list[int] | None = field(default_factory=lambda: [21, 7])
    offsets: list[float] | None = field(default_factory=lambda: [1.5, 2])
    std_smooth_sigma: float | None = 4
    A: ConfigFilterChannel | None = field(
        default_factory=lambda: ConfigFilterChannel(
            ConfigFilterComponent(None, None, None),
            ConfigFilterComponent(None, None, None),
            ConfigFilterComponent(None, None, None),
        ),
    )
    M: ConfigFilterChannel | None = field(
        default_factory=lambda: ConfigFilterChannel(
            ConfigFilterComponent(None, None, None),
            ConfigFilterComponent(None, None, None),
            ConfigFilterComponent(None, None, None),
        ),
    )
    no_works_noise: dict[str, float] = field(default_factory=lambda: {"M": 10, "A": 100})


# ---------------------------------------------------------------------------
# Per-entry-point ``proc`` group — optional processing parameters
# ---------------------------------------------------------------------------


# ConfigProcCalib is PipelineConfig — single source of truth lives in pipeline.py.
# Re-exported here so that Hydra ConfigStore registration and ``run.py`` imports
# both resolve from the same canonical location.
# Lazy: calibration requires scipy, absent in the noh5 (text-only) distribution.
try:
    from tcm.calibration.pipeline import PipelineConfig as ConfigProcCalib

    _HAS_CALIBRATION = True
except ImportError:
    _HAS_CALIBRATION = False
    ConfigProcCalib = None  # type: ignore[assignment,misc]


@dataclass
class ConfigProcSpectrum:
    """Spectrum processing parameters (``proc`` group → ``spectrum`` option).

    Reserved forward declaration — the spectrum module has NOT been ported
    from ``_dask_legacy/incl_h5spectrum.py``.  Schema-only for ConfigStore
    registration.  Follow-up tracked in ``dtdr-todo.md`` A.
    """

    overlap: float = 0.5
    time_intervals_center: list[str] | None = None
    dt_interval: str | None = None
    fmin: float | None = None
    fmax: float | None = None


# ---------------------------------------------------------------------------
# Top-level Config + ConfigStore registration
# ---------------------------------------------------------------------------


@dataclass
class Config:
    """Top-level Hydra configuration.

    Defaults list selects ``base`` option for each group.
    Per-file overrides come from ``run`` config group (via CLI).
    """

    defaults: list[Any] = field(
        default_factory=lambda: [
            {"input": "base"},
            {"out": "base"},
            {"filter": "base"},
            {"program": "base"},
            "_self_",
        ]
    )
    input: ConfigIn_InclProc = MISSING
    out: ConfigOut_InclProc = MISSING
    filter: ConfigFilter_InclProc = MISSING
    program: ConfigProgram = MISSING


if _HAS_CALIBRATION:

    @dataclass
    class ConfigCalib(Config):
        """Calibration entry-point — inherits processing groups + adds proc + calib filter."""

        defaults: list[Any] = field(
            default_factory=lambda: [
                {"input": "base"},
                {"out": "base"},
                {"filter": "calib"},
                {"proc": "calib"},
                {"program": "base"},
                "_self_",
            ]
        )
        filter: ConfigFilterCalib = MISSING  # overrides Config.filter type
        proc: ConfigProcCalib = MISSING  # PipelineConfig (single source of truth)


cs = ConfigStore.instance()

# Processing entry-point groups (no ``proc``)
cs.store(group="input", name="base", node=ConfigIn_InclProc)
cs.store(group="out", name="base", node=ConfigOut_InclProc)
cs.store(group="filter", name="base", node=ConfigFilter_InclProc)
cs.store(group="program", name="base", node=ConfigProgram)

# Calibration entry-point filter (inherits base filter + adds despike)
cs.store(group="filter", name="calib", node=ConfigFilterCalib)

# ``proc`` group — optional per-entry-point processing parameters
if _HAS_CALIBRATION:
    cs.store(group="proc", name="calib", node=ConfigProcCalib)
cs.store(group="proc", name="spectrum", node=ConfigProcSpectrum)  # schema-only, port deferred

# Note: the top-level ``Config`` and ``ConfigCalib`` dataclasses are NOT registered
# with ConfigStore.  Hydra resolves them from config.yaml + defaults list + structured
# group configs.  The ``Config/ConfigCalib`` classes are used directly by
# ``to_omegaconf_merge_compatible()`` in ``config_yaml.save_config_to_yaml()``.
