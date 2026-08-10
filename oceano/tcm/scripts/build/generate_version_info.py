"""Generate ``version_info.txt`` for PyInstaller from ``version.py`` + env suffix."""
import sys
from pathlib import Path

_suffixes = [env_name] if (env_name := Path(sys.prefix).name) != "default" else []

# CLI defaults — overridden by GUI build script via kwargs.
_DESCRIPTION = (
    "AB SIO RAS' TCM raw data processing CLI. "
    "Exports physical values (e.g. water velocity) to text"
)
_INTERNAL_NAME = "tcm_proc.exe"
_ORIGINAL_FILENAME = "tcm\\scripts\\tcm_proc.py"


def main(
    version: str,
    suffixes: list[str] = _suffixes,
    description: str = _DESCRIPTION,
    internal_name: str = _INTERNAL_NAME,
    original_filename: str = _ORIGINAL_FILENAME,
) -> None:
    """Write ``version_info.txt`` from the template, injecting *version* and metadata."""
    parts = version.split(".")
    nums = [int(x) for x in parts]
    nums.extend([0] * (4 - len(nums)))
    filevers = ", ".join(map(str, nums[:4]))
    cur_file = Path(__file__)
    template = cur_file.with_name("version_info.template").read_text(encoding="utf-8")

    cur_file.with_name("version_info.txt").write_text(
        template.format(
            VERSION="+".join([version] + suffixes),
            FILEVERS=filevers,
            DESCRIPTION=description,
            INTERNAL_NAME=internal_name,
            ORIGINAL_FILENAME=original_filename,
        ),
        encoding="utf-8",
    )
