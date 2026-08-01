"""Build tcm_gui distribution via PyInstaller. Updates VERSION (version.py) if env BUILD_MODE==manual """
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from generate_version_info import main as generate_version_info

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SPEC = PROJECT_ROOT / "scripts" / "build" / "tcm_gui.spec"
if (build_mode := os.environ.get("BUILD_MODE", "auto")) == "manual":  # auto | manual
    BASE_VERSION = datetime.now(tz=UTC).strftime("%Y.%m")
    SPEC.with_name("version.py").write_text(f'VERSION = "{BASE_VERSION}"\n', encoding="utf-8")   # update lib
else:
    from version import VERSION as BASE_VERSION

print(f"BUILD_MODE: {build_mode},", f"BASE_VERSION: {BASE_VERSION}")
generate_version_info(
    BASE_VERSION,
    description=(
        "AB SIO RAS' TCM inclinometer data processor GUI. "
        "Configure and run processing with interactive coefficient editor"
    ),
    internal_name="tcm_gui.exe",
    original_filename="tcm\\scripts\\tcm_gui.py",
)

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
