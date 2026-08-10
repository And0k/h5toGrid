"""Inclinometer data processor — thin CLI entry point.

Parses CLI arguments, resolves data directory, and delegates to
:func:`tcm.processing.run` via Hydra ``@hydra.main`` which composes
the full :class:`tcm.schema.Config`.

Usage::

    # Process all discovered probes
    python scripts/tcm_proc.py "_raw/*i*.txt"

    # Specific probes only
    python scripts/tcm_proc.py "_raw/*i*.txt" 'input.ids=[i01,i_p02]'

    # Override any config field
    python scripts/tcm_proc.py "_raw/*i*.txt" out.text_path=./results

    # Config-generation-only (scan) — writes processing-cfg_from_args.log
    python scripts/tcm_proc.py "_raw/*i*.txt" 'program.return_=<cfg_from_args>'

For the legacy dask-dataframe pipeline, use ``tcm._dask_legacy.scripts.tcm_proc``.
Full user guide: :file:`docs/tcm_cli/README.md`.
"""
from tcm import cli, processing

if __name__ == "__main__":
    cli.call_in_raw_dir(processing.run)
