"""Build metadata: write/read ``version_meta.json``, generate ``version_info.txt``.

Single source of truth for version + product metadata used by:
- PyInstaller ``version_info.txt`` (Windows file properties)
- Frozen exe at runtime (About dialog reads ``version_meta.json`` from ``_MEIPASS``)

DRY: build scripts call :func:`write_meta` once with per-product kwargs;
:func:`main` reads the JSON and renders the template — no parameter duplication.
"""
import json
import subprocess
import sys
from pathlib import Path

_SUFFIXES = [env] if (env := Path(sys.prefix).name) != "default" else []
_META_FILE = "version_meta.json"


# ── git remote URL ──────────────────────────────────────────────────────────


def repo_url(cwd: Path | None = None) -> str | None:
    """Read ``remote.origin.url`` from git; normalize SSH→HTTPS; strip trailing ``.git``.

    Returns ``None`` on any failure (no git, no remote, tarball build).
    """
    try:
        result = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None
        url = result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None

    # git@github.com:User/Repo.git → https://github.com/User/Repo
    if url.startswith("git@"):
        # Format: git@host:path → https://host/path
        host_part, _, path_part = url[4:].partition(":")
        url = f"https://{host_part}/{path_part}"
    return url.removesuffix(".git")


def git_branch(cwd: Path | None = None) -> str | None:
    """Read current branch name from git; fallback ``main`` on failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def docs_url(repo: str | None, branch: str | None = None) -> str | None:
    """Derive docs URL from repo: ``{repo}/tree/{branch}/oceano/tcm/docs``."""
    if not repo:
        return None
    return f"{repo}/tree/{branch or 'main'}/oceano/tcm/docs"


# ── JSON writer ─────────────────────────────────────────────────────────────


def write_meta(
    version: str,
    *,
    product: str,
    description: str,
    internal_name: str,
    original_filename: str,
    company_name: str,
    legal_copyright: str,
    product_name: str,
    suffixes: list[str] = _SUFFIXES,
    project_root: Path | None = None,
    out_dir: Path | None = None,
) -> dict:
    """Write ``version_meta.json`` next to this script (or into *out_dir*).

    Args:
        version: Base version string (e.g. ``"2026.08"``).
        product: ``"tcm_gui"`` or ``"tcm_proc"``.
        description: Windows ``FileDescription``.
        internal_name: Windows ``InternalName`` (e.g. ``"tcm_gui.exe"``).
        original_filename: Windows ``OriginalFilename``.
        company_name: Windows ``CompanyName``.
        legal_copyright: Windows ``LegalCopyright``.
        product_name: Windows ``ProductName``.
        suffixes: Env suffix list appended to version (e.g. ``["bin-optim-tcm"]``).
        project_root: For git remote lookup; defaults to this script's great-grandparent.
        out_dir: Target directory (tests); default = next to this script.

    Returns:
        The written metadata dict.
    """
    full_version = "+".join([version] + suffixes)
    parts = version.split(".")
    nums = [int(x) for x in parts]
    nums.extend([0] * (4 - len(nums)))

    repo = repo_url(project_root or Path(__file__).resolve().parent.parent.parent)
    branch = git_branch(project_root or Path(__file__).resolve().parent.parent.parent)
    meta = {
        "name": "TCM",
        "product": product,
        "version": full_version,
        "filevers": nums[:4],
        "description": description,
        "internal_name": internal_name,
        "original_filename": original_filename,
        "company_name": company_name,
        "legal_copyright": legal_copyright,
        "product_name": product_name,
        "repo_url": repo,
        "docs_url": docs_url(repo, branch),
    }
    (out_dir or Path(__file__).parent).joinpath(_META_FILE).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return meta


# ── JSON reader (used by spec files + runtime) ──────────────────────────────


def load_meta(spec_dir: Path | None = None) -> dict:
    """Read ``version_meta.json`` from *spec_dir* (default: next to this script)."""
    path = (spec_dir or Path(__file__).parent) / _META_FILE
    return json.loads(path.read_text(encoding="utf-8"))


# ── version_info.txt generator ──────────────────────────────────────────────


def main(spec_dir: Path | None = None) -> None:
    """Read ``version_meta.json`` and render ``version_info.txt`` from template."""
    meta = load_meta(spec_dir)
    cur_file = Path(__file__)
    template = cur_file.with_name("version_info.template").read_text(encoding="utf-8")

    filevers = ", ".join(map(str, meta["filevers"]))
    cur_file.with_name("version_info.txt").write_text(
        template.format(
            VERSION=meta["version"],
            FILEVERS=filevers,
            DESCRIPTION=meta["description"],
            INTERNAL_NAME=meta["internal_name"],
            ORIGINAL_FILENAME=meta["original_filename"],
            COMPANY_NAME=meta["company_name"],
            LEGAL_COPYRIGHT=meta["legal_copyright"],
            PRODUCT_NAME=meta["product_name"],
        ),
        encoding="utf-8",
    )
