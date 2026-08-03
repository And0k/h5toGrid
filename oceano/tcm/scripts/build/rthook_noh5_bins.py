"""Runtime hook: override ``out/base`` defaults in frozen noh5 dist.

Re-registers ``out/base`` in Hydra ConfigStore with noh5-appropriate
binning/text-save defaults:

- ``dt_bins: [0, 3600]`` — no averaging + 1 hour (instead of dev ``[0,2,600,3600,7200]``)
- ``dt_bins_min_save_text: 0`` — allow TSV export for bin=0 (no-avg)

Executed **before** the application entry point → Hydra compose picks up
the overridden defaults instead of the dataclass originals.

``tcm/schema.py`` registers ``out/base`` at module level during ``import tcm``
(which ``from tcm import processing`` triggers).  This hook forces that import
then re-registers with noh5 values — ``cs.store(group="out", name="base", ...)``
overwrites the prior registration.
"""
# Force tcm.schema import → triggers ConfigStore registration with dev defaults
import tcm.schema  # noqa: F401 — side-effect only

from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()

# Re-register out/base with noh5-appropriate defaults.
# Passing an *instance* (not type) so field values serve as Hydra defaults.
cs.store(
    group="out",
    name="base",
    node=tcm.schema.ConfigOut_InclProc(
        dt_bins=[0, 3600],
        dt_bins_min_save_text=0,
    ),
    provider="noh5-rthook",
)