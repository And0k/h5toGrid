"""Local documentation browser — serves tcm docs offline in the system browser.

Subsystem layout (the whole chain is owned by the ``browser/`` folder at the
tcm project root):

    browser/package.json + package-lock.json   npm dependency declaration & lock
    browser/vendor.mjs                         sync/update automation
    _build/browser-runtime/                    generated third-party runtime
    tcm_gui/browser/server.py                  HTTP server + static locations
    tcm_gui/browser/browser.py                 public entry (DocumentationBrowser)
    tcm_gui/browser/web/                       first-party viewer page files
"""

from tcm_gui.browser.browser import (
    DocumentationBrowser,
    documentation_browser,
    get_documentation_browser,
    link_display,
    open_md_link,
)

__all__ = [
    "DocumentationBrowser",
    "documentation_browser",
    "get_documentation_browser",
    "link_display",
    "open_md_link",
]
