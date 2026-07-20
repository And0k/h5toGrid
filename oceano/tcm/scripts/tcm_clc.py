"""Inclinometer data processor — thin CLI entry point.

Parses CLI arguments, resolves data directory, and delegates to
:func:`tcm.processing.run` via Hydra ``@hydra.main`` which composes
the full :class:`tcm.config.Config`.

Usage::

    # Process all discovered probes
    python scripts/tcm_clc.py "_raw/*i*.txt"

    # Specific probes only
    python scripts/tcm_clc.py "_raw/*i*.txt" input.ids=[i01,i_p02]

    # Override any config field
    python scripts/tcm_clc.py "_raw/*i*.txt" out.text_path=./results

For the legacy dask-dataframe pipeline, use ``tcm._dask_legacy.scripts.tcm_clc``.
Full user guide: :file:`docs/tcm_clc/README.md`.
"""
from tcm import processing, cli

if __name__ == "__main__":
    cli.call_in_raw_dir(processing.run)
