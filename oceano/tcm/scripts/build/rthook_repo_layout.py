"""Runtime hook: expose repo-mirrored datas to the import system.

The frozen tree mirrors the dev repo — ``_MEIPASS`` ≙ repo root — with bundled
packages under ``shared/`` and ``oceano/<proj>/src/`` (see
``spec_common.FIRST_PARTY_PKGS`` / spec dest constants).  Prepending those parents to
``sys.path`` lets ``import tcm`` / ``tcm_gui`` / ``utils`` / ``veusz_helpers``
resolve to the source datas.  Must be listed **first** — before any hook that
imports ``tcm`` (``rthook_noh5_bins``) or composes ``pkg://`` Hydra configs.
"""

import os
import sys

_base = sys._MEIPASS
sys.path[:0] = [
    os.path.join(_base, "shared", "utils", "src"),
    os.path.join(_base, "shared", "veusz_helpers", "src"),
    os.path.join(_base, "oceano", "tcm", "src"),
    os.path.join(_base, "oceano", "tcm_gui", "src"),
    os.path.join(_base, "oceano", "meta_finder", "src"),
]
