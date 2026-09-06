import os
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from spec_common import should_keep_binary, should_keep_data

# ICU DLLs + netCDF-C stack must be dropped
for name in (
    "icudt78.dll", "icuuc78.dll", "icuin78.dll", "icuio78.dll", "icutu78.dll", "icutest78.dll",
    "netcdf.dll", "libxml2.dll", "libcurl.dll", "psl-5.dll",
):
    assert not should_keep_binary((name, "Library/bin/" + name, "BINARY")), name

# Essentials kept
for name in ("openblas.dll", "liblapack.dll", "sqlite3.dll", "tk86t.dll"):
    assert should_keep_binary((name, "Library/bin/" + name, "BINARY")), name

# No false positive from "icu" substring in data filenames (e.g. circUlation)
assert should_keep_data(("docs/circulation_overview.md", "docs/circulation_overview.md", "DATA"))
assert should_keep_data(("tcm/graphics.py", "tcm/graphics.py", "PYMODULE"))

print("filter checks OK")
