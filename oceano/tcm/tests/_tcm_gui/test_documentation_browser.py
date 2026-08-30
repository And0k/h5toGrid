"""Regression guards for the documentation browser: viewer assets, offline
runtime locations and allowed roots.

The first-party viewer page ships inside the package (``tcm_gui/browser/web``);
the third-party runtime is generated into ``_build/browser-runtime`` by
``browser/vendor.mjs`` (pixi task ``browser-runtime``).  These tests pin the
wiring between server routes, file locations and the page markup.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tcm._constants import resource_root
from tcm_gui.browser import DocumentationBrowser, get_documentation_browser, open_md_link
from tcm_gui.browser.server import (
    _VEND_DIR,
    _VIEWER_FILES,
    _WEB_DIR,
    _dir_listing,
    _frozen_variant,
    _kind_of,
)


def _src(name: str) -> str:
    return (_WEB_DIR / name).read_text(encoding="utf-8")


def test_static_locations_follow_resource_root():
    """Dev and frozen layouts coincide: web/ inside the package, vendor under _build."""
    assert _WEB_DIR.parent.name == "browser"  # tcm_gui/browser/web
    assert _VEND_DIR == resource_root() / "_build" / "browser-runtime"
    assert all((_WEB_DIR / f).is_file() for f in ("index.html", "viewer.js", "viewer.css"))
    assert _VIEWER_FILES["/"][0] == "index.html"  # routes map onto the same dir
    assert _VIEWER_FILES["/viewer.js"][0] == "viewer.js"
    assert _VIEWER_FILES["/viewer.css"][0] == "viewer.css"
    assert get_documentation_browser is not None


@pytest.mark.skipif(not (_VEND_DIR / "marked").is_dir(), reason="browser-runtime runtime not synced")
def test_vendor_runtime_present():
    """Minimal file map served under /assets/ (browser/vendor.mjs header)."""
    for rel in (
        "marked/marked.min.js",
        "mathjax/tex-chtml.js",
        "mathjax/sre/speech-worker.js",
        "mathjax/output/fonts/mathjax-newcm/chtml.js",
        "highlight/highlight.min.js",
        "highlight/default.min.css",
    ):
        assert (_VEND_DIR / rel).is_file(), rel


def test_index_html_is_fully_offline():
    """All scripts/fonts vendored under /assets/ — the frozen EXE has no Internet."""
    h = _src("index.html")
    assert "cdn.jsdelivr" not in h

    for needle in (
        "/assets/marked/marked.min.js",
        "/assets/mathjax/tex-chtml.js",
        "/assets/highlight/highlight.min.js",
        "/assets/highlight/default.min.css",
        # MathJax newcm font package pinned locally (stub must be vendored too,
        # else the loader falls back to its jsdelivr default)
        'loader: { paths: { "mathjax-newcm": "/assets/mathjax/output/fonts/mathjax-newcm" } }',
        "typeset: false",
        '<script src="/viewer.js"></script>',
        '<link rel="stylesheet" href="/viewer.css">',
    ):
        assert needle in h, needle

    # MathJax config must precede its script; chrome stripped: browser's own
    # back/forward + tab title suffice
    assert h.index("window.MathJax") < h.index("/assets/mathjax/tex-chtml.js")
    assert 'id="toolbar"' not in h
    assert "history.back()" not in h


def test_viewer_js_document_classes():
    """markdown / source / image dispatch and the TeX-shielding pipeline."""
    js = _src("viewer.js")
    for needle in (
        "/api/document",  # JSON {kind, file, language, content}
        "/api/asset",  # image bytes, native MIME
        "function kindOf",  # JS mirror of _kind_of
        "function protectMath",  # TeX delimiters → %%MATHn%% before marked…
        "function restoreMath",  # …restored after (marked strips \ before punctuation)
        "%%MATH",
        "function renderSource",  # highlight.js view
        "function renderImage",
        "function scrollToSourceLine",  # #L42 line anchors
        "navigationSerial",  # stale-fetch guard
        "rewriteLocalImages",
        "wrapTables",
        "resolveLinkUrl",
    ):
        assert needle in js, needle


def test_viewer_js_panels_become_visible():
    """show*View must set display:"block" — "" falls back to the stylesheet's
    display:none and leaves a blank page (regression: .py files rendered empty)."""
    js = _src("viewer.js")
    for panel in ("content", "sourcePanel", "imagePanel"):
        assert f'{panel}.style.display = "block"' in js, panel
        assert f'{panel}.style.display = ""' not in js, panel


def test_viewer_js_rehighlights_source_after_back():
    """hljs must not skip the <code> it already marked on a bfcache restore —
    the data-highlighted marker is cleared before each highlight (regression:
    py coloring vanished after browser Back)."""
    js = _src("viewer.js")
    assert 'removeAttribute("data-highlighted")' in js
    # marker is stripped in renderSource BEFORE highlightElement runs
    hl = js.index('removeAttribute("data-highlighted")')
    assert js.index("highlightElement") > hl


def test_kind_of_classification():
    assert _kind_of(resource_root() / "readme.md") == "markdown"
    assert _kind_of(resource_root() / "src" / "tcm" / "__init__.py") == "source"
    assert _kind_of(resource_root() / "docs" / "x.png") == "image"
    assert _kind_of(resource_root() / "data.h5") is None
    # language table trimmed to repo-present types
    assert _kind_of(Path("x.java")) is None
    assert _kind_of(Path("x.toml")) == "source"


def test_default_root_covers_docs_cross_links():
    """``../../readme.md``-style links land above ``docs/`` → root must be the package."""
    db = DocumentationBrowser()
    assert db.is_allowed(resource_root() / "readme.md")
    assert db.is_allowed(resource_root() / "docs" / "project_developer_guide" / "GUI" / "_index.md")


@pytest.mark.skipif(
    not (_VEND_DIR / "marked" / "marked.min.js").is_file(), reason="browser-runtime runtime not synced"
)
def test_open_appends_anchor_fragment(monkeypatch):
    """``open(path, anchor=...)`` appends a URL-encoded ``#fragment`` — the
    viewer decodes ``location.hash`` and scrolls to that heading/line."""
    captured: dict[str, str] = {}
    monkeypatch.setattr(
        "tcm_gui.browser.browser.webbrowser.open_new_tab", lambda url: captured.setdefault("url", url)
    )
    db = DocumentationBrowser()
    monkeypatch.setattr(db, "_ensure_server", lambda: ("127.0.0.1", 9999))
    md = resource_root() / "readme.md"
    if not md.is_file():
        pytest.skip("readme.md not present")
    db.open(md, anchor="directory layout")
    assert captured["url"].endswith("#directory%20layout"), captured["url"]


def test_open_md_link_relative_resolves_against_base(monkeypatch):
    """A relative link ``io_formats.md#anchor`` resolves against the base file's parent."""
    calls: list[tuple[Path, str | None]] = []

    class _Fake:
        def open(self, path, anchor=None):
            calls.append((Path(path), anchor))

    monkeypatch.setattr("tcm_gui.browser.browser.get_documentation_browser", lambda: _Fake())
    open_md_link("io_formats.md#directory-layout", base="R:/docs/reference/config_reference.md")
    assert len(calls) == 1
    assert calls[0][0] == Path("R:/docs/reference/io_formats.md")
    assert calls[0][1] == "directory-layout"


def test_open_md_link_anchor_only_uses_base(monkeypatch):
    """An anchor-only link ``#anchor`` uses the base file path directly."""
    calls: list[tuple[Path, str | None]] = []

    class _Fake:
        def open(self, path, anchor=None):
            calls.append((Path(path), anchor))

    monkeypatch.setattr("tcm_gui.browser.browser.get_documentation_browser", lambda: _Fake())
    open_md_link("#input.path", base="R:/docs/reference/config_reference.md")
    assert len(calls) == 1
    assert calls[0][0] == Path("R:/docs/reference/config_reference.md")
    assert calls[0][1] == "input.path"


def test_open_md_link_absolute_path(monkeypatch):
    calls: list[tuple[Path, str | None]] = []

    class _Fake:
        def open(self, path, anchor=None):
            calls.append((Path(path), anchor))

    monkeypatch.setattr("tcm_gui.browser.browser.get_documentation_browser", lambda: _Fake())
    open_md_link("C:/docs/x.md#L42")
    assert calls == [(Path("C:/docs/x.md"), "L42")]


def test_open_md_link_external_os(monkeypatch):
    """External schemes go to the OS browser, never the doc server."""
    import os as _os

    opened: list[str] = []
    monkeypatch.setattr("tcm_gui.browser.browser.sys.platform", "win32")
    monkeypatch.setattr(_os, "startfile", opened.append, raising=False)
    open_md_link("https://example.com/x")
    assert opened == ["https://example.com/x"]


def test_open_md_link_empty_noop(monkeypatch):
    calls: list[tuple[Path, str | None]] = []

    class _Fake:
        def open(self, path, anchor=None):
            calls.append((path, anchor))

    monkeypatch.setattr("tcm_gui.browser.browser.get_documentation_browser", lambda: _Fake())
    open_md_link("   ")
    assert calls == []


def test_frozen_variant_strips_src_segment():
    """Frozen app bundles ``src/tcm`` under ``tcm/`` — dev-layout doc links
    (``../../src/tcm/…``) must lose ``src`` to resolve in ``_MEIPASS``."""
    # dev layout: .../tcm/src/tcm/format.py → frozen: .../tcm/tcm/format.py
    dev = resource_root() / "src" / "tcm" / "format.py"
    frozen = _frozen_variant(dev)
    assert frozen == resource_root() / "tcm" / "format.py"
    # src not followed by tcm, or no src at all → not applicable
    assert _frozen_variant(resource_root() / "docs" / "x.md") is None
    assert _frozen_variant(resource_root() / "src" / "other" / "y.py") is None


def test_dir_listing_immediate_children_md_only(tmp_path: Path):
    """/api/directory lists immediate children — subfolders and markdown files
    only (source excluded unless show_source), hrefs relative with trailing slash."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "user_guide").mkdir()
    (tmp_path / "docs" / "user_guide" / "getting_started.md").write_text("# Getting Started")
    (tmp_path / "docs" / "user_guide" / "config.yaml").write_text("a: 1")
    (tmp_path / "docs" / "user_guide" / "nested").mkdir()

    listing = _dir_listing(tmp_path / "docs" / "user_guide")
    assert listing["kind"] == "directory"
    assert listing["name"] == "user_guide"
    assert listing["directories"] == [{"name": "nested", "href": "nested/"}]
    assert listing["files"] == [{"name": "getting_started.md", "href": "getting_started.md"}]

    # show_source adds the yaml file; hidden tests of the default are above
    src = _dir_listing(tmp_path / "docs" / "user_guide", show_source=True)
    assert "config.yaml" in [f["name"] for f in src["files"]]


def test_viewer_js_directory_first_class():
    """Folder links are ordinary <a> with a trailing slash; the same delegated
    click handler routes them to the directory panel, and popstate restores it."""
    js = _src("viewer.js")
    for needle in (
        "/api/directory",
        "function isDirHref",  # trailing-slash → folder nav
        "function renderDirectory",
        "function navigateDirectory",
        "function resolveDirHref",
        'currentKind === "directory"',  # directory listings resolve against the folder
        'e.state.kind === "directory"',  # back/forward restores the listing
    ):
        assert needle in js, needle
    # directory panel is a hidden panel shown via display:block like the others
    idx = _src("index.html")
    assert 'id="directory-panel"' in idx
    assert 'id="directory-title"' in idx


def test_viewer_js_directory_panel_display_block():
    """directoryPanel.showDirectoryView sets display:block — same blank-page
    fix as source/image panels ("": falls back to CSS display:none)."""
    js = _src("viewer.js")
    assert 'directoryPanel.style.display = "block"' in js
    assert js.count('style.display = "none"') == 4  # all four panels hide
