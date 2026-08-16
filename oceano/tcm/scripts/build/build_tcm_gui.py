"""Build tcm_gui distribution via PyInstaller.

Updates ``version_meta.json`` when ``BUILD_MODE=manual`` (date-based version);
``auto`` mode reads the existing JSON.  Then generates ``version_info.txt``
and invokes PyInstaller.
"""
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from generate_version_info import main as generate_version_info, write_meta

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SPEC = PROJECT_ROOT / "scripts" / "build" / "tcm_gui.spec"
SPEC_DIR = SPEC.parent

build_mode = os.environ.get("BUILD_MODE", "auto")
if build_mode == "manual":
    BASE_VERSION = datetime.now(tz=UTC).strftime("%Y.%m")
else:
    from generate_version_info import load_meta
    BASE_VERSION = load_meta(SPEC_DIR)["version"].split("+")[0]

print(f"BUILD_MODE: {build_mode},", f"BASE_VERSION: {BASE_VERSION}")

write_meta(
    BASE_VERSION,
    product="tcm_gui",
    description=(
        "AB SIO RAS' TCM inclinometer data processor GUI. "
        "Configure and run processing with interactive coefficient editor"
    ),
    internal_name="tcm_gui.exe",
    original_filename="tcm\\scripts\\tcm_gui.py",
    company_name=(
        "Atlantic Branch of Shirshov Institute of Oceanology, "
        "Russian Academy of Sciences (AB SIO RAS)"
    ),
    legal_copyright='© Andrey Korzh <"ao.korzh@gmail.com">',
    product_name="TCM calculations",
    project_root=PROJECT_ROOT,
)
generate_version_info(SPEC_DIR)

cmd = [
    sys.executable, "-m", "PyInstaller",
    "--noconfirm",
    str(SPEC),
]
print(f"Running: {' '.join(cmd)}")
result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False)
if result.returncode == 0:
    print(f"\nBuild complete: {PROJECT_ROOT / 'dist' / 'tcm_gui' / 'tcm_gui.exe'}")
sys.exit(result.returncode)
