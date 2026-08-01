"""Runtime hook: register Hydra plugins + patch argparse for Python 3.14.

Python 3.14's argparse introduced ``_check_help`` which calls
``_expand_help`` on every ``help=`` value passed to ``add_argument``.
If the help is a non-string, non-iterable object (e.g. a lazy-doc wrapper
from dask or a workflow library), Python raises
``TypeError: not a container or iterable`` when ``_expand_help`` tries
``'%' not in help_string``, which is then re-raised as
``ValueError: badly formed help string`` by ``_check_help``.

We patch ``_get_help_string`` before Hydra registers its plugins so that
any non-string help value is coerced to ``str`` — the string representation
is fine for display and prevents the crash.
"""

import os
import sys

_BASE = getattr(sys, "_MEIPASS", None) or os.path.dirname(sys.executable)
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

# ---------------------------------------------------------------------------
# argparse compatibility for Python 3.14 — must run before Hydra builds parser
# ---------------------------------------------------------------------------
import argparse as _arg

_orig_get_help_string = _arg.HelpFormatter._get_help_string  # type: ignore[attr-defined]


def _patched_get_help_string(self, action) -> str | None:
    """Coerce non-string help values to ``str()`` before argparse validates them."""
    help_string = _orig_get_help_string(self, action)
    if help_string is not None and not isinstance(help_string, str):
        return str(help_string)
    return help_string


_arg.HelpFormatter._get_help_string = _patched_get_help_string  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Hydra plugin registrations
# ---------------------------------------------------------------------------
from hydra.core.plugins import Plugins  # noqa: E402

p = Plugins.instance()

from hydra._internal.core_plugins.basic_launcher import BasicLauncher  # noqa: E402
from hydra._internal.core_plugins.basic_sweeper import BasicSweeper  # noqa: E402
from hydra._internal.core_plugins.file_config_source import FileConfigSource  # noqa: E402
from hydra._internal.core_plugins.importlib_resources_config_source import ImportlibResourcesConfigSource  # noqa: E402
from hydra._internal.core_plugins.structured_config_source import StructuredConfigSource  # noqa: E402

for cls in [
    ImportlibResourcesConfigSource,
    FileConfigSource,
    StructuredConfigSource,
    BasicLauncher,
    BasicSweeper,
]:
    p.register(cls)

try:
    from hydra_plugins.hydra_colorlog.colorlog import HydraColorlogSearchPathPlugin
    p.register(HydraColorlogSearchPathPlugin)
except ImportError:
    pass