"""Local documentation browser — one localhost server, four resource classes.

Architecture
------------
Tkinter link click → :meth:`DocumentationBrowser.open` → start the
localhost server once → system browser opens ``/?file=<absolute-path>`` →
JavaScript fetches ``/api/document?file=...`` and dispatches by class:

===========  =============================================================
markdown     GFM via vendored marked.js; TeX via vendored MathJax 4
             (``typesetPromise`` after every dynamic load; delimiters
             ``\\(...\\)`` inline, ``$$...$$``/``\\[...\\]`` display —
             shielded from marked's backslash-escape handling by
             ``protectMath`` placeholders in ``web/viewer.js``)
source/text  verbatim text + vendored highlight.js, ``#L42`` line anchors
images       raw bytes from ``/api/asset`` with native MIME type
external     ``http(s)/mailto/ftp/file`` links are handed to the OS handler
             (``file:`` unquoted to its filesystem path first)
===========  =============================================================

Subsequent ``open`` calls and all in-page navigation (relative links,
same-file/cross-file anchors) reuse the same server — handled browser-side
by JavaScript; back/forward is the browser's own history.

The HTTP server binds to 127.0.0.1 only.  Documents must resolve beneath
``allowed_roots`` (default: :func:`~tcm._constants.resource_root` — the
whole ``tcm`` package, since docs cross-link files above ``docs/`` — plus
:data:`~tcm._constants.REPO_ROOT` so sibling-project docs like
``meta_finder/`` are reachable).
First-party viewer files ship inside this package (``web/``); the
third-party runtime is generated into ``_build/browser-runtime`` by
``browser/vendor.mjs`` (pixi task ``browser-runtime``).  Fully offline:
no CDN.  Windows-path link resolution (``resolveLinkUrl``, GitHub-slug
heading ids) is pinned by ``tests/_tcm_gui/test_documentation_browser.py``.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path

from tcm._constants import REPO_ROOT, resource_root
from tcm_gui.browser.server import _VEND_DIR, _kind_of, _rel_error, _Server

_l = os.path.basename(__name__)  # module logger name
_lf = __import__("logging").getLogger(_l)


class DocumentationBrowser:
    """Singleton-style documentation service for the application.

    The first call to :meth:`open` starts a localhost HTTP server; subsequent
    calls reuse it.  Links inside rendered documents are handled by
    JavaScript and never start another server.

    Example::

        browser = DocumentationBrowser()
        browser.open(path_to_doc)

    Parameters
    ----------
    allowed_roots
        Filesystem roots from which documents may be served — the security
        boundary.  Default: :func:`~tcm._constants.resource_root` plus
        :data:`~tcm._constants.REPO_ROOT` (sibling-project docs like
        ``meta_finder/`` are linked from tcm docs).
    """

    def __init__(self, *, allowed_roots: list[str | os.PathLike[str]] | None = None) -> None:
        self._roots: tuple[Path, ...] = tuple(
            Path(r).resolve() for r in (allowed_roots or [resource_root(), REPO_ROOT])
        )
        self._server: _Server | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        """Return True when the local HTTP server is running."""
        return self._server is not None

    def add_root(self, root: str | os.PathLike[str]) -> None:
        """Add an approved document root at runtime."""
        path = Path(root).resolve()
        with self._lock:
            if path not in self._roots:
                self._roots = (*self._roots, path)

    def is_allowed(self, path: Path) -> bool:
        """Return True when *path* belongs to an approved root."""
        if not self._roots:
            return True
        return any(not _rel_error(path, root) for root in self._roots)

    def _ensure_server(self) -> tuple[str, int]:
        """Start the singleton server if needed and return its address."""
        with self._lock:
            if self._server is not None:
                return self._server.server_address
            server = _Server(("127.0.0.1", 0), self)
            thread = threading.Thread(target=server.serve_forever, name="doc-browser", daemon=True)
            thread.start()
            self._server = server
            self._thread = thread
            _lf.debug("Documentation server started at %s:%s", *server.server_address)
            return server.server_address

    def open(self, path: str | os.PathLike[str], *, anchor: str | None = None) -> None:
        """Open a supported local document in the system default browser.

        Starts the localhost server on the first call and reuses it afterwards.
        *anchor* — an optional heading id (GitHub-slug) or ``L42`` source-line
        anchor — becomes the ``#fragment`` of the served URL, which the viewer
        resolves and scrolls to.
        """
        file_path = Path(path).resolve()
        if not file_path.is_file():
            raise FileNotFoundError(file_path)
        if _kind_of(file_path) is None:
            raise ValueError(f"Unsupported documentation file type: {file_path}")
        if not self.is_allowed(file_path):
            raise PermissionError(f"Document is outside allowed roots: {file_path}")
        if not (_VEND_DIR / "marked" / "marked.min.js").is_file():
            raise FileNotFoundError(
                f"Browser runtime is not installed: {_VEND_DIR}\n"
                "Run: pixi run -e bin-optim-tcm browser-runtime"
            )

        host, port = self._ensure_server()
        url = f"http://{host}:{port}/?file={urllib.parse.quote(str(file_path), safe='')}"
        if anchor:
            url += f"#{urllib.parse.quote(anchor, safe='')}"
        webbrowser.open_new_tab(url)
        _lf.debug("Opened in browser: %s (%s)", file_path.name, _kind_of(file_path))

    def close(self) -> None:
        """Stop the localhost server."""
        with self._lock:
            server = self._server
            self._server = None
            self._thread = None
        if server:
            server.shutdown()
            server.server_close()
            _lf.debug("Documentation server stopped")


# ─────────────────────────────────────────────────────────────────────────────
# Markdown link dispatcher
# ─────────────────────────────────────────────────────────────────────────────

# Schemes handed to the OS handler unchanged (everything else is a local path;
# a Windows drive letter like ``C:`` can never match this whitelist).
_EXTERNAL_SCHEME = re.compile(r"^(?:https?|mailto|ftp|ftps|file):", re.IGNORECASE)


def _file_uri_path(uri: str) -> str:
    """``file:`` URI → filesystem path (UNC ``//netloc`` restored)."""
    parts = urllib.parse.urlparse(uri)
    path = urllib.request.url2pathname(parts.path)
    return rf"\\{parts.netloc}{path}" if parts.netloc else path


def link_display(url: str) -> str:
    """Human-readable link target for hover/status display: ``file:`` URI → path."""
    return _file_uri_path(url) if url.lower().startswith("file:") else url


def open_os_target(target: str) -> None:
    """Open *target* with its OS-associated application.

    Shared dispatcher for external link schemes and auto-linked filesystem
    paths; failures are logged, never raised — links live in tooltips.
    """
    try:
        if sys.platform == "win32":
            os.startfile(target)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", target])
        else:
            subprocess.Popen(["xdg-open", target])
    except OSError:
        _lf.exception("Failed to open with OS application: %s", target)


def open_md_link(url: str, base: str | os.PathLike[str] | None = None) -> None:
    """Open a markdown link target.

    Handles any ``[text](target)`` link rendered by the GUI: ``file:`` URIs
    (explicit or auto-linked bare paths, see
    :func:`tcm_gui.md_label._path_spans`) and other external schemes
    (``http(s)/mailto/ftp``) via the OS-associated application; local
    document paths (relative ones resolved against *base* — the source
    ``.md`` file's path) via :class:`DocumentationBrowser`'s localhost server,
    also shown in the OS browser; a ``#anchor`` fragment scrolls to the
    heading.  Failures are logged, never raised — links live in tooltips.
    """
    target = url.strip()
    if not target:
        return
    file_part, _, anchor = target.partition("#")
    if _EXTERNAL_SCHEME.match(file_part):
        open_os_target(_file_uri_path(file_part) if file_part.lower().startswith("file:") else target)
        return
    path = Path(file_part)
    if not path.is_absolute() and base:
        # Anchor-only link (#foo): base is the source .md file path, use it
        # directly.  Relative link (doc.md#foo): resolve against base's parent
        # directory.
        base_path = Path(base)
        path = base_path if not file_part else base_path.parent / path
    try:
        get_documentation_browser().open(path, anchor=anchor or None)
    except (OSError, ValueError):
        _lf.exception("Failed to open doc link: %s", target)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────

documentation_browser: DocumentationBrowser | None = None


def get_documentation_browser() -> DocumentationBrowser:
    """Return the application-wide :class:`DocumentationBrowser` singleton."""
    global documentation_browser
    if documentation_browser is None:
        documentation_browser = DocumentationBrowser()
    return documentation_browser
