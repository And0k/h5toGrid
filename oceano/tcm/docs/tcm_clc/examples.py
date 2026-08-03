"""Usage examples for TCM CLI pipeline: config resolution, generation, and processing.

Run from the project root::

    pixi run python oceano/tcm/docs/tcm_clc/examples.py [data_pattern]

All examples assume a data directory with ``_raw/`` containing CSV/NC/HDF5 files
and ``cfg_proc/`` containing Hydra config files (``config.yaml``, ``run/*.yaml``).

Logging is configured by Hydra via ``cfg_proc/hydra/job_logging/colorlog.yaml``
(console: colored ``funcName|message``; file: ``asctime|name|levelname|message``).
No manual ``logging.basicConfig`` is needed — ``cli.call_in_raw_dir`` bootstraps
``@hydra.main``, which applies the ``dictConfig`` before any example code runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from tcm import cli, config_yaml, format, processing
from tcm.schema import Return


def _pcid_from_path(path: str | Path) -> str:
    """Derive probe ID from an input path stem (``@i_01.txt`` → ``i01``)."""
    stem = Path(path).stem
    if (identity := format.probe_from_name(format.stem_to_pcid(stem).lower())):
        return format.pcid_from_parts(model=identity[0], number=identity[1])
    return format.stem_to_pcid(stem)


# ---------------------------------------------------------------------------
# Example 1: Resolve per-probe configurations without processing
# ---------------------------------------------------------------------------

def resolve_configs(data_pattern: str) -> dict[str, dict[str, Any]]:
    """Return resolved per-probe config dicts for *data_pattern*.

    Uses ``program.return_=<cfg_from_args>`` so ``processing.run`` generates
    YAML configs and returns a 4-tuple
    ``(processed_pcids, failed_pcids, last_cfg, collected)`` where
    ``collected = [(stem, yaml_path, DictConfig), ...]``.  No data is loaded
    or processed; no YAML read-back is needed.

    Parameters
    ----------
    data_pattern
        Glob/regex matching data files (same as the first positional CLI arg
        to ``tcm_clc.py``), e.g. ``"_raw/*i*.txt"``.

    Returns
    -------
    dict[str, dict]
        ``{pcid: cfg_dict}`` where each value is a plain ``dict`` with keys
        ``"input"``, ``"out"``, ``"filter"`` — as produced by
        :func:`config_yaml.gen_metadata` (via :func:`config_yaml.prep_cfg_for_probe`).

        ``cfg_dict["input"]`` contains resolved fields:

        - ``path`` — absolute path to the corrected data file
        - ``coefs`` — calibration coefficients (``Ag``, ``Cg``, ``Ah``, ``Ch``, ``Rz``, ``kVabs``, …)
        - ``time_ranges`` — ``[start_iso, end_iso]`` from first/last data row
        - ``corr_time_mode`` — time correction flag
        - ``tables`` — resolved HDF5 table name or CSV column layout code
    """
    from omegaconf import OmegaConf

    result = cli.call_in_raw_dir(
        processing.run,
        input={"path": data_pattern},
        program={"return_": Return.CFG_FROM_ARGS},
    )
    # result is a 4-tuple: (processed_pcids, failed_pcids, last_cfg, collected)
    collected = result[3] if result and len(result) >= 4 else []
    return {
        _pcid_from_path(cfg_dc["input"]["path"]): OmegaConf.to_container(cfg_dc, resolve=True)
        for _stem, _yp, cfg_dc in collected
    }


# ---------------------------------------------------------------------------
# Example 2: Generate (and optionally modify) per-probe YAML run configs
# ---------------------------------------------------------------------------

def generate_and_save(
    data_pattern: str,
    *,
    modify: Optional[Callable[[dict], dict]] = None,
) -> list[Path]:
    """Generate ``cfg_proc/run/*.yaml`` for *data_pattern*, optionally modified.

    ``processing.run`` (with ``return_=cfg_from_args``) generates YAMLs via
    :func:`config_yaml.save_config_to_yaml` and returns the full per-probe
    config dicts.  If *modify* is given, each YAML on disk is read back with
    ruamel (same library as ``save_config_to_yaml``), modified, and overwritten
    — preserving the ``# @package _global_`` header and ruamel formatting.

    Parameters
    ----------
    data_pattern
        Glob/regex matching data files.
    modify
        ``modify(cfg_dict) -> modified_cfg_dict`` applied to each per-probe
        config dict read from the YAML on disk (plain ``dict`` with ``"input"``,
        ``"out"``, ``"filter"`` keys — non-default fields only, as written by
        ``save_config_to_yaml``).  Return the modified dict to overwrite the file.

    Returns
    -------
    list[Path]
        Absolute paths of the YAML files in ``cfg_proc/run/``.
    """
    cli.call_in_raw_dir(
        processing.run,
        input={"path": data_pattern},
        program={"return_": Return.CFG_FROM_ARGS},
    )

    run_dir = Path.cwd() / "cfg_proc" / "run"
    written: list[Path] = []

    for f in sorted(run_dir.glob("*.yaml")):
        if modify:
            ry = config_yaml._ry()
            with f.open(encoding="utf-8") as fp:
                cfg_dict = ry.load(fp)
            cfg_dict = modify(cfg_dict)
            with f.open("w", encoding="utf-8") as fp:
                fp.write("# @package _global_\n")
                ry.dump(cfg_dict, fp)
        written.append(f)

    return written


# ---------------------------------------------------------------------------
# Example 3: Run only existing configs without generating new ones
# ---------------------------------------------------------------------------

def run_existing(
    data_pattern: str,
    yaml_filter: str = "*",
) -> None:
    """Process only existing ``cfg_proc/run/*.yaml`` matching *yaml_filter*.

    When ``input.yaml_path`` is set (any non-None value), ``processing.run``
    **skips config generation** — no new YAMLs are created.  Only existing
    YAMLs whose stem (or full name) matches *yaml_filter* are processed.

    Parameters
    ----------
    data_pattern
        Glob/regex for data directory discovery (used to locate ``_raw/``).
    yaml_filter
        Glob/regex matched against YAML stems in ``cfg_proc/run/``.
        ``"*"`` (default) matches all.  Examples: ``"@i_01"``,
        ``"@i_(01|02)"``, ``"*_01.yaml"``.
    """
    cli.call_in_raw_dir(
        processing.run,
        input={"path": data_pattern, "yaml_path": yaml_filter},
    )


# ---------------------------------------------------------------------------
# Entry point — demo all three
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    pattern = sys.argv[1] if len(sys.argv) > 1 else "_raw/*i*.txt"

    print(f"=== Example 1: Resolve configs for {pattern!r} ===")
    cfgs = resolve_configs(pattern)
    for pcid, c in sorted(cfgs.items()):
        print(f"  {pcid}: {c['input']['path']}")
        if tr := c.get("input", {}).get("time_ranges"):
            print(f"       time_range: {tr[0]} ... {tr[-1]}")

    print("\n=== Example 2: Generate (optionally modify) YAMLs ===")
    written = generate_and_save(pattern, modify=lambda c: c)  # identity = no modification
    print(f"  Written {len(written)} YAML files")

    print("\n=== Example 3: Run only existing configs ===")
    run_existing(pattern)
    print("  Done.")
