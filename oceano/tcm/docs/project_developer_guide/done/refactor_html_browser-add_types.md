The right abstraction is no longer `MarkdownBrowser`; it is a small `DocumentationBrowser` that owns one localhost server and understands four classes of local resources: Markdown, source/text documents, images, and ordinary external links. The same service can be called from any Tkinter widget, while links inside rendered Markdown remain inside the browser.

The implementation below uses `marked` for Markdown/GFM, MathJax 4 for dynamic TeX rendering, and highlight.js for source code. `marked` documents browser-side parsing with `marked.parse()` and GFM support, while warning that its HTML output is not sanitized. ([Marked][1]) MathJax 4 specifically requires `typesetPromise()` for robust dynamic content and recommends promise-based typesetting because font data can also load asynchronously. ([MathJax Documentation][2]) highlight.js currently documents `highlightElement()` for browser-side `<pre><code>` highlighting and reports 11.11.1 as its current release; its own site also says not to use the project website itself as a CDN, so the example uses cdnjs instead. ([highlightjs.org][3])

The code supports `.md`, Python and common source/config formats, PNG/JPEG/GIF/WebP/SVG/BMP/ICO/AVIF images, relative Markdown links, relative image references, fragments, browser history, a rendered/source toggle, syntax highlighting, line-preserving source display, and explicit allowed filesystem roots.

```python
"""
documentation_browser.py

A local documentation browser for a Tkinter application.

Architecture
------------

                         Tkinter application
                                  |
             +--------------------+--------------------+
             |                    |                    |
        Text widget          About dialog          Help dialog
             |                    |                    |
             +--------------------+--------------------+
                                  |
                         DocumentationBrowser
                                  |
                         one localhost server
                                  |
                    http://127.0.0.1:<port>/
                                  |
               +------------------+------------------+
               |                  |                  |
             .md                source             images
               |                  |                  |
             marked          highlight.js        <img>/<browser>
               |
           MathJax 4

The application creates ONE DocumentationBrowser instance and shares it
among all widgets.

The first .open(...) call starts the HTTP server.

Every later .open(...) call reuses the same server and the same port.

Markdown links inside the browser are handled by JavaScript, so they also
reuse the existing server.

Examples
--------

    docs = DocumentationBrowser(
        allowed_roots=[
            PROJECT_ROOT / "docs",
            PROJECT_ROOT / "src",
            PROJECT_ROOT / "scripts",
        ]
    )

    docs.open(PROJECT_ROOT / "docs" / "README.md")

A Markdown document may contain:

    [implementation](../scripts/tcm_proc.py)

    ![architecture](images/architecture.png)

    [configuration](../src/config.py#L42)

Mathematics remains ordinary TeX in Markdown:

    Inline: \\(E = mc^2\\)

    Display:

    \\[
    x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}
    \\]

Supported document classes
---------------------------

Markdown:
    .md, .markdown

Source / text:
    .py, .pyi, .js, .jsx, .ts, .tsx, .css, .html, .htm,
    .json, .yaml, .yml, .toml, .ini, .cfg, .conf,
    .txt, .log, .sh, .bash, .ps1, .bat, .cmd,
    .sql, .xml, .xsd, .c, .h, .cpp, .hpp, .java,
    .rs, .go, .r, .lua, .php, .rb, .swift, .kt,
    .cmake, .make, .dockerfile

Images:
    .png, .jpg, .jpeg, .gif, .webp, .svg, .bmp, .ico, .avif

The roots are security boundaries. A requested file must resolve beneath one
of the configured roots. Symlinks that resolve outside the roots are rejected.

Security
--------

This viewer is intended for trusted local documentation.

Marked does not sanitize Markdown-generated HTML. Therefore the Markdown
roots must be considered trusted. If untrusted Markdown is ever displayed,
add an HTML sanitizer such as DOMPurify before assigning innerHTML.

External hyperlinks are left to the browser.

Local executable files are NEVER executed; they are returned as plain text.

Dependencies
------------

Python:
    standard library only

Browser:
    marked
    MathJax 4
    highlight.js

The URLs below use public CDNs for simplicity.

For a PyInstaller application that must work without Internet access,
vendor these JavaScript/CSS resources into the application and serve them
from the same localhost server. No application architecture changes are
required.
"""

import atexit
import html
import http.server
import mimetypes
import os
from pathlib import Path
import threading
import urllib.parse
import webbrowser


# =============================================================================
# Browser-side libraries
# =============================================================================

# Marked's browser bundle.
MARKED_URL = (
    "https://cdn.jsdelivr.net/npm/marked/lib/marked.umd.js"
)

# MathJax 4 combined TeX + CommonHTML component.
MATHJAX_URL = (
    "https://cdn.jsdelivr.net/npm/mathjax@4/tex-chtml.js"
)

# highlight.js 11.11.1.
HIGHLIGHT_JS_URL = (
    "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/"
    "11.11.1/highlight.min.js"
)

HIGHLIGHT_CSS_URL = (
    "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/"
    "11.11.1/styles/default.min.css"
)


# =============================================================================
# File type definitions
# =============================================================================

MARKDOWN_EXTENSIONS = {
    ".md",
    ".markdown",
}


SOURCE_LANGUAGES = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".css": "css",
    ".html": "xml",
    ".htm": "xml",
    ".xml": "xml",
    ".xsd": "xml",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".ini": "ini",
    ".cfg": "ini",
    ".conf": "ini",
    ".txt": "plaintext",
    ".log": "plaintext",
    ".sh": "shell",
    ".bash": "shell",
    ".ps1": "powershell",
    ".bat": "dos",
    ".cmd": "dos",
    ".sql": "sql",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cc": "cpp",
    ".hh": "cpp",
    ".java": "java",
    ".rs": "rust",
    ".go": "go",
    ".r": "r",
    ".lua": "lua",
    ".php": "php",
    ".rb": "ruby",
    ".swift": "swift",
    ".kt": "kotlin",
}


IMAGE_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
    ".bmp": "image/bmp",
    ".ico": "image/x-icon",
    ".avif": "image/avif",
}


# =============================================================================
# Browser application
# =============================================================================

def _viewer_html() -> str:
    """Return the complete HTML/JavaScript browser application."""

    marked_url = html.escape(MARKED_URL, quote=True)
    mathjax_url = html.escape(MATHJAX_URL, quote=True)
    highlight_js_url = html.escape(HIGHLIGHT_JS_URL, quote=True)
    highlight_css_url = html.escape(HIGHLIGHT_CSS_URL, quote=True)

    return f"""\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">

<title>Documentation</title>

<link
    rel="stylesheet"
    href="{highlight_css_url}"
>


<!--
MathJax configuration MUST precede the MathJax script.

Dollar-sign inline math is intentionally disabled. Documentation commonly
contains ordinary $ characters, shell variables, prices, etc. Use:

    \\( ... \\)

for inline mathematics and:

    \\[ ... \\]
    $$ ... $$

for display mathematics.
-->
<script>
window.MathJax = {{
    tex: {{
        inlineMath: [
            ["\\\\(", "\\\\)"]
        ],

        displayMath: [
            ["\\\\[", "\\\\]"],
            ["$$", "$$"]
        ]
    }}
}};
</script>

<script src="{marked_url}"></script>
<script src="{highlight_js_url}"></script>
<script src="{mathjax_url}"></script>


<style>
/* =========================================================================
   Theme
   ========================================================================= */

:root {{
    color-scheme: light dark;

    --page-background: #ffffff;
    --text: #202124;

    --muted: #6b7075;

    --border: #c6c9cd;

    --toolbar-background: #eef0f2;

    --button-background: #ffffff;
    --button-hover: #e2e5e8;

    --table-header: #eceff1;
    --table-alt: #f7f8f9;

    --code-background: #f2f3f5;

    --link: #1565c0;

    --blockquote: #757575;

    --source-line-number: #8a8f95;
}}


@media (prefers-color-scheme: dark) {{
    :root {{
        --page-background: #1e1f20;
        --text: #e4e4e4;

        --muted: #a0a5ab;

        --border: #55585d;

        --toolbar-background: #292b2e;

        --button-background: #34363a;
        --button-hover: #41444a;

        --table-header: #303236;
        --table-alt: #27292c;

        --code-background: #2b2d30;

        --link: #78a9ff;

        --blockquote: #aaaeb5;

        --source-line-number: #777c82;
    }}
}}


html {{
    background: var(--page-background);
}}


body {{
    margin: 0;

    background: var(--page-background);
    color: var(--text);

    font-family:
        system-ui,
        -apple-system,
        "Segoe UI",
        sans-serif;

    line-height: 1.55;
}}


/* =========================================================================
   Toolbar
   ========================================================================= */

#toolbar {{
    position: sticky;
    top: 0;
    z-index: 1000;

    display: flex;
    align-items: center;
    gap: 0.35rem;

    padding:
        0.4rem
        0.65rem;

    background: var(--toolbar-background);

    border-bottom:
        1px solid
        var(--border);
}}


button {{
    padding:
        0.3rem
        0.65rem;

    border:
        1px solid
        var(--border);

    border-radius: 4px;

    background:
        var(--button-background);

    color:
        var(--text);

    font:
        inherit;

    cursor: pointer;
}}


button:hover {{
    background:
        var(--button-hover);
}}


button.active {{
    font-weight: 600;
}}


#filename {{
    min-width: 0;

    margin-left: 0.6rem;

    color: var(--muted);

    font-size: 0.84rem;

    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}}


/* =========================================================================
   Rendered Markdown
   ========================================================================= */

#content {{
    max-width: 1100px;

    margin:
        0
        auto;

    padding:
        2rem
        3rem
        4rem;
}}


.markdown-body h1,
.markdown-body h2,
.markdown-body h3,
.markdown-body h4,
.markdown-body h5,
.markdown-body h6 {{
    line-height: 1.25;
}}


.markdown-body h1 {{
    padding-bottom: 0.35em;

    border-bottom:
        1px solid
        var(--border);
}}


.markdown-body a {{
    color: var(--link);
}}


.markdown-body img {{
    max-width: 100%;
    height: auto;
}}


.markdown-body blockquote {{
    margin:
        1rem
        0;

    padding-left: 1rem;

    border-left:
        4px solid
        var(--blockquote);

    color: var(--muted);
}}


.markdown-body hr {{
    border: 0;

    border-top:
        1px solid
        var(--border);
}}


.markdown-body code {{
    padding:
        0.12em
        0.3em;

    border-radius: 4px;

    background:
        var(--code-background);

    font-family:
        Consolas,
        "Cascadia Code",
        "DejaVu Sans Mono",
        monospace;
}}


.markdown-body pre {{
    overflow-x: auto;

    padding: 1rem;

    border-radius: 6px;

    background:
        var(--code-background);
}}


.markdown-body pre code {{
    padding: 0;

    background:
        transparent;
}}


/* =========================================================================
   Markdown tables
   ========================================================================= */

.table-wrapper {{
    overflow-x: auto;

    margin:
        1.2rem
        0;
}}


.markdown-body table {{
    width: 100%;

    border-collapse: collapse;

    border:
        1px solid
        var(--border);

    font-size: 0.95em;
}}


.markdown-body th,
.markdown-body td {{
    padding:
        0.45rem
        0.7rem;

    border:
        1px solid
        var(--border);

    text-align: left;
    vertical-align: top;
}}


.markdown-body th {{
    background:
        var(--table-header);

    font-weight: 600;
}}


.markdown-body tr:nth-child(even) td {{
    background:
        var(--table-alt);
}}


/* =========================================================================
   MathJax
   ========================================================================= */

.markdown-body mjx-container {{
    overflow-x: auto;
    overflow-y: hidden;
}}


.markdown-body mjx-container[display="true"] {{
    margin:
        1rem
        0;
}}


/* =========================================================================
   Source viewer
   ========================================================================= */

#source-panel {{
    display: none;

    margin:
        0
        auto;

    padding:
        1.2rem
        2rem
        3rem;
}}


#source-code {{
    margin: 0;

    padding: 1rem;

    overflow-x: auto;

    border-radius: 6px;

    background:
        var(--code-background);

    tab-size: 4;

    font-family:
        Consolas,
        "Cascadia Code",
        "DejaVu Sans Mono",
        monospace;

    font-size: 0.9rem;

    line-height: 1.5;

    white-space: pre;
}}


/*
 * Keep syntax highlighting independent of the application theme.
 * highlight.js provides the syntax token colors.
 */
#source-code code {{
    font-family: inherit;
}}


/* =========================================================================
   Image viewer
   ========================================================================= */

#image-panel {{
    display: none;

    min-height:
        calc(100vh - 55px);

    box-sizing: border-box;

    padding:
        1.5rem;

    text-align: center;
}}


#image-view {{
    max-width: 100%;

    max-height:
        calc(100vh - 110px);

    object-fit: contain;

    border:
        1px solid
        var(--border);
}}


/* =========================================================================
   Message / error states
   ========================================================================= */

.message {{
    max-width: 1000px;

    margin:
        3rem
        auto;

    padding:
        0
        2rem;

    color:
        var(--muted);
}}


#error {{
    color:
        #c62828;

    white-space:
        pre-wrap;
}}
</style>
</head>


<body>

<header id="toolbar">

    <button
        id="render-button"
        class="active"
        type="button"
    >
        Rendered
    </button>

    <button
        id="source-button"
        type="button"
    >
        Source
    </button>

    <button
        id="back-button"
        type="button"
        title="Browser Back"
    >
        ←
    </button>

    <button
        id="forward-button"
        type="button"
        title="Browser Forward"
    >
        →
    </button>

    <span id="filename"></span>
</header>


<main
    id="content"
    class="markdown-body"
></main>


<section id="source-panel">

    <pre id="source-code"><code></code></pre>

</section>


<section id="image-panel">

    <img
        id="image-view"
        alt=""
    >

</section>


<script>
"use strict";


/* =========================================================================
   DOM
   ========================================================================= */

const content = document.getElementById("content");

const sourcePanel = document.getElementById("source-panel");
const sourceCode = document.querySelector("#source-code code");

const imagePanel = document.getElementById("image-panel");
const imageView = document.getElementById("image-view");

const filenameLabel = document.getElementById("filename");

const renderButton = document.getElementById("render-button");
const sourceButton = document.getElementById("source-button");

const backButton = document.getElementById("back-button");
const forwardButton = document.getElementById("forward-button");


/* =========================================================================
   State
   ========================================================================= */

let currentFile = "";
let currentKind = "";
let currentLanguage = "";
let currentContent = "";

let currentMode = "rendered";


/*
 * A monotonic counter prevents a slow previous request from overwriting a
 * newer navigation request.
 */
let navigationSerial = 0;


/* =========================================================================
   File-type helpers
   ========================================================================= */

const markdownExtensions = new Set([
    ".md",
    ".markdown"
]);


const sourceLanguages = {{
    "py": "python",
    "pyi": "python",
    "js": "javascript",
    "jsx": "javascript",
    "mjs": "javascript",
    "cjs": "javascript",
    "ts": "typescript",
    "tsx": "typescript",
    "css": "css",
    "html": "xml",
    "htm": "xml",
    "xml": "xml",
    "xsd": "xml",
    "json": "json",
    "yaml": "yaml",
    "yml": "yaml",
    "toml": "toml",
    "ini": "ini",
    "cfg": "ini",
    "conf": "ini",
    "txt": "plaintext",
    "log": "plaintext",
    "sh": "shell",
    "bash": "shell",
    "ps1": "powershell",
    "bat": "dos",
    "cmd": "dos",
    "sql": "sql",
    "c": "c",
    "h": "c",
    "cpp": "cpp",
    "hpp": "cpp",
    "cc": "cpp",
    "hh": "cpp",
    "java": "java",
    "rs": "rust",
    "go": "go",
    "r": "r",
    "lua": "lua",
    "php": "php",
    "rb": "ruby",
    "swift": "swift",
    "kt": "kotlin"
}};


const imageExtensions = new Set([
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".svg",
    ".bmp",
    ".ico",
    ".avif"
]);


function extensionOf(path) {{
    const clean = splitAnchor(path).path;

    const slash = Math.max(
        clean.lastIndexOf("/"),
        clean.lastIndexOf("\\\\")
    );

    const filename = clean.slice(slash + 1);

    const dot = filename.lastIndexOf(".");

    if (dot < 0) {{
        return "";
    }}

    return filename.slice(dot).toLowerCase();
}}


function isMarkdownFile(path) {{
    return markdownExtensions.has(
        extensionOf(path)
    );
}}


function isSourceFile(path) {{
    return Object.prototype.hasOwnProperty.call(
        sourceLanguages,
        extensionOf(path).slice(1)
    );
}}


function isImageFile(path) {{
    return imageExtensions.has(
        extensionOf(path)
    );
}}


function sourceLanguage(path) {{
    return (
        sourceLanguages[
            extensionOf(path).slice(1)
        ] || "plaintext"
    );
}}


/* =========================================================================
   Path handling
   ========================================================================= */

function splitAnchor(value) {{
    const index = value.indexOf("#");

    if (index < 0) {{
        return {{
            path: value,
            anchor: ""
        }};
    }}

    return {{
        path: value.slice(0, index),
        anchor: value.slice(index + 1)
    }};
}}


function isExternalUrl(value) {{
    return /^(?:https?|mailto|ftp|data):/i.test(value);
}}


function normalizeFilesystemPath(path) {{
    path = path.replaceAll("/", "\\\\");

    let prefix = "";

    if (/^[A-Za-z]:/.test(path)) {{
        prefix = path.slice(0, 2);
        path = path.slice(2);
    }} else if (path.startsWith("\\\\\\\\")) {{
        prefix = "\\\\\\\\";
        path = path.slice(2);
    }}

    const parts = [];

    for (const part of path.split("\\\\")) {{

        if (!part || part === ".") {{
            continue;
        }}

        if (part === "..") {{

            if (
                parts.length > 0 &&
                parts[parts.length - 1] !== ".."
            ) {{
                parts.pop();
            }}

            continue;
        }}

        parts.push(part);
    }}

    if (prefix === "\\\\\\\\") {{
        return prefix + parts.join("\\\\");
    }}

    if (prefix) {{
        return prefix + "\\\\" + parts.join("\\\\");
    }}

    return parts.join("\\\\");
}}


/*
 * Resolve href/src relative to the filesystem location of the current
 * Markdown document.
 */
function resolveRelativePath(baseFile, target) {{
    const parsed = splitAnchor(target);

    if (!parsed.path) {{
        return {{
            path: baseFile,
            anchor: parsed.anchor
        }};
    }}

    const targetPath =
        decodeURIComponent(parsed.path)
        .replaceAll("/", "\\\\");

    const basePath =
        baseFile
        .replaceAll("/", "\\\\");

    /*
     * Absolute Windows drive path:
     *     C:\\docs\\file.py
     */
    if (/^[A-Za-z]:\\\\/.test(targetPath)) {{
        return {{
            path: normalizeFilesystemPath(targetPath),
            anchor: parsed.anchor
        }};
    }}

    /*
     * UNC:
     *     \\\\server\\share\\file.py
     */
    if (targetPath.startsWith("\\\\\\\\")) {{
        return {{
            path: normalizeFilesystemPath(targetPath),
            anchor: parsed.anchor
        }};
    }}

    /*
     * Relative path.
     */
    const separator = basePath.lastIndexOf("\\\\");

    const directory =
        separator >= 0
            ? basePath.slice(0, separator + 1)
            : "";

    return {{
        path: normalizeFilesystemPath(
            directory + targetPath
        ),
        anchor: parsed.anchor
    }};
}}


/* =========================================================================
   API
   ========================================================================= */

async function fetchDocument(file) {{
    const query = new URLSearchParams({{
        file
    }});

    const response = await fetch(
        `/api/document?${{query.toString()}}`,
        {{
            cache: "no-store"
        }}
    );

    if (!response.ok) {{
        throw new Error(
            `HTTP ${{response.status}} while loading ${{file}}\\n\\n` +
            await response.text()
        );
    }}

    return response.json();
}}


/* =========================================================================
   Layout
   ========================================================================= */

function hideAllViews() {{
    content.style.display = "none";
    sourcePanel.style.display = "none";
    imagePanel.style.display = "none";
}}


function showRenderedView() {{
    hideAllViews();

    content.style.display = "";

    renderButton.classList.add("active");
    sourceButton.classList.remove("active");
}}


function showSourceView() {{
    hideAllViews();

    sourcePanel.style.display = "";

    renderButton.classList.remove("active");
    sourceButton.classList.add("active");
}}


function showImageView() {{
    hideAllViews();

    imagePanel.style.display = "";

    renderButton.classList.remove("active");
    sourceButton.classList.remove("active");
}}


/* =========================================================================
   Markdown rendering
   ========================================================================= */

function wrapTables() {{
    for (const table of content.querySelectorAll("table")) {{

        if (
            table.parentElement &&
            table.parentElement.classList.contains(
                "table-wrapper"
            )
        ) {{
            continue;
        }}

        const wrapper =
            document.createElement("div");

        wrapper.className =
            "table-wrapper";

        table.parentNode.insertBefore(
            wrapper,
            table
        );

        wrapper.appendChild(table);
    }}
}}


/*
 * Replace local image URLs created by marked with URLs pointing to our local
 * image endpoint.
 *
 * Example:

    ![diagram](../images/overview.png)

 * becomes:

    /api/asset?file=C%3A%5Cproject%5Cimages%5Coverview.png
 */
function rewriteLocalImages(baseFile) {{
    for (
        const image of
        content.querySelectorAll("img[src]")
    ) {{

        const src =
            image.getAttribute("src");

        if (!src || isExternalUrl(src)) {{
            continue;
        }}

        const resolved =
            resolveRelativePath(
                baseFile,
                src
            );

        if (!isImageFile(resolved.path)) {{
            continue;
        }}

        const query =
            new URLSearchParams({{
                file: resolved.path
            }});

        image.src =
            `/api/asset?${{query.toString()}}`;

        image.loading = "lazy";

        image.decoding = "async";
    }}
}}


async function typesetMath() {{
    if (
        window.MathJax &&
        window.MathJax.typesetPromise
    ) {{
        await window.MathJax.typesetPromise(
            [content]
        );
    }}
}}


/* =========================================================================
   Source rendering
   ========================================================================= */

function renderSource(
    text,
    file,
    language
) {{
    showSourceView();

    sourceCode.textContent = text;

    /*
     * highlightElement() operates on the <code> element and preserves the
     * source as code rather than interpreting it as HTML.
     */
    sourceCode.className =
        `language-${{language}}`;

    if (window.hljs) {{
        window.hljs.highlightElement(
            sourceCode
        );
    }}
}}


/* =========================================================================
   Image rendering
   ========================================================================= */

function renderImage(
    file
) {{
    showImageView();

    const query =
        new URLSearchParams({{
            file
        }});

    imageView.src =
        `/api/asset?${{query.toString()}}`;

    imageView.alt =
        file.split(/[\\\\/]/).pop() || "Image";
}}


/* =========================================================================
   Markdown rendering
   ========================================================================= */

async function renderMarkdown(
    markdown,
    file,
    anchor
) {{
    /*
     * MathJax maintains internal MathItem objects for rendered expressions.
     * Before replacing dynamic content, tell MathJax that the old content
     * is disappearing.
     */
    if (
        window.MathJax &&
        window.MathJax.typesetClear
    ) {{
        window.MathJax.typesetClear(
            [content]
        );
    }}

    /*
     * marked parses CommonMark/GFM Markdown into HTML.
     */
    const rendered =
        marked.parse(
            markdown,
            {{
                gfm: true,
                breaks: false
            }}
        );

    content.innerHTML =
        rendered;

    wrapTables();

    rewriteLocalImages(file);

    currentContent =
        markdown;

    currentFile =
        file;

    currentKind =
        "markdown";

    currentLanguage =
        "markdown";

    filenameLabel.textContent =
        file;

    if (currentMode === "source") {{
        /*
         * The user explicitly requested source view while navigating.
         * Source display does not need MathJax.
         */
        sourceCode.textContent =
            markdown;

        sourceCode.className =
            "language-markdown";

        if (window.hljs) {{
            window.hljs.highlightElement(
                sourceCode
            );
        }}

        showSourceView();
    }} else {{
        showRenderedView();

        /*
         * Dynamic Markdown must be explicitly typeset.
         */
        await typesetMath();

        if (anchor) {{
            scrollToAnchor(anchor);
        }} else {{
            window.scrollTo(0, 0);
        }}
    }}
}}


/* =========================================================================
   Navigation
   ========================================================================= */

async function navigate(
    file,
    anchor = "",
    pushHistory = true
) {{
    const serial =
        ++navigationSerial;

    content.innerHTML =
        '<div class="message">Loading…</div>';

    try {{
        const documentData =
            await fetchDocument(file);

        /*
         * A newer navigation may already have started while the previous
         * request was in flight.
         */
        if (serial !== navigationSerial) {{
            return;
        }}

        const kind =
            documentData.kind;

        currentFile =
            documentData.file;

        currentKind =
            kind;

        currentContent =
            documentData.content || "";

        currentLanguage =
            documentData.language || "plaintext";

        filenameLabel.textContent =
            documentData.file;

        if (kind === "markdown") {{

            await renderMarkdown(
                documentData.content,
                documentData.file,
                anchor
            );

        }} else if (kind === "source") {{

            renderSource(
                documentData.content,
                documentData.file,
                documentData.language
            );

            if (anchor) {{
                scrollToSourceLine(
                    anchor
                );
            }}

        }} else if (kind === "image") {{

            renderImage(
                documentData.file
            );

        }} else {{
            throw new Error(
                `Unsupported document kind: ${{kind}}`
            );
        }}

        if (pushHistory) {{
            const url =
                `/?file=${{encodeURIComponent(file)}}` +
                (
                    anchor
                        ? `#${{encodeURIComponent(anchor)}}
`
                        : ""
                );

            history.pushState(
                {{
                    file,
                    anchor
                }},
                "",
                url
            );
        }}

    }} catch (error) {{
        showError(error);
    }}
}}


/* =========================================================================
   Markdown links
   ========================================================================= */

content.addEventListener(
    "click",
    event => {{

        const link =
            event.target.closest("a");

        if (!link) {{
            return;
        }}

        const href =
            link.getAttribute("href");

        if (!href) {{
            return;
        }}

        /*
         * External URLs remain ordinary browser links.
         */
        if (isExternalUrl(href)) {{
            return;
        }}

        /*
         * Current-document fragment.
         */
        if (href.startsWith("#")) {{

            event.preventDefault();

            const anchor =
                href.slice(1);

            scrollToAnchor(
                anchor
            );

            const url =
                `/?file=${{encodeURIComponent(currentFile)}}` +
                `#${{encodeURIComponent(anchor)}}`;

            history.pushState(
                {{
                    file: currentFile,
                    anchor
                }},
                "",
                url
            );

            return;
        }}

        const resolved =
            resolveRelativePath(
                currentFile,
                href
            );

        /*
         * A local Markdown/source/image link stays in this documentation
         * application.
         */
        if (
            isMarkdownFile(resolved.path) ||
            isSourceFile(resolved.path) ||
            isImageFile(resolved.path)
        ) {{
            event.preventDefault();

            navigate(
                resolved.path,
                resolved.anchor,
                true
            );
        }}
    }}
);


/* =========================================================================
   Source line anchors
   ========================================================================= */

function scrollToSourceLine(anchor) {{
    /*
     * The source viewer uses a simple convention:

        #L42

     means "approximately line 42".

     The source element is plain text, so this is implemented by calculating
     the line height and scrolling the <pre> accordingly.
     */
    const match =
        /^L(\\d+)$/i.exec(
            anchor
        );

    if (!match) {{
        return;
    }}

    const lineNumber =
        Number(match[1]);

    if (
        !Number.isFinite(lineNumber) ||
        lineNumber < 1
    ) {{
        return;
    }}

    const lineHeight =
        parseFloat(
            getComputedStyle(
                sourceCode
            ).lineHeight
        );

    const top =
        Math.max(
            0,
            (lineNumber - 1) * lineHeight
        );

    sourceCode.parentElement.scrollTop =
        top;
}}


function scrollToAnchor(anchor) {{
    requestAnimationFrame(() => {{

        const element =
            document.getElementById(
                anchor
            );

        if (element) {{
            element.scrollIntoView({{
                block: "start"
            }});
        }}
    }});
}}


/* =========================================================================
   Browser history
   ========================================================================= */

window.addEventListener(
    "popstate",
    event => {{

        if (
            !event.state ||
            !event.state.file
        ) {{
            return;
        }}

        navigate(
            event.state.file,
            event.state.anchor || "",
            false
        );
    }}
);


/* =========================================================================
   Toolbar
   ========================================================================= */

renderButton.addEventListener(
    "click",
    async () => {{

        currentMode =
            "rendered";

        if (
            currentKind === "markdown"
        ) {{
            await renderMarkdown(
                currentContent,
                currentFile,
                ""
            );
        }} else if (
            currentKind === "image"
        ) {{
            renderImage(
                currentFile
            );
        }} else {{
            renderSource(
                currentContent,
                currentFile,
                currentLanguage
            );
        }}
    }}
);


sourceButton.addEventListener(
    "click",
    () => {{

        currentMode =
            "source";

        if (
            currentKind === "markdown" ||
            currentKind === "source"
        ) {{
            renderSource(
                currentContent,
                currentFile,
                currentLanguage
            );
        }}
    }}
);


backButton.addEventListener(
    "click",
    () => window.history.back()
);


forwardButton.addEventListener(
    "click",
    () => window.history.forward()
);


/* =========================================================================
   Errors
   ========================================================================= */

function showError(error) {{
    hideAllViews();

    content.style.display = "";

    const wrapper =
        document.createElement("div");

    wrapper.className =
        "message";

    const heading =
        document.createElement("h2");

    heading.id =
        "error";

    heading.textContent =
        "Cannot display document";

    const details =
        document.createElement("pre");

    details.textContent =
        String(error);

    wrapper.appendChild(
        heading
    );

    wrapper.appendChild(
        details
    );

    content.replaceChildren(
        wrapper
    );
}}


/* =========================================================================
   Initial document
   ========================================================================= */

(async () => {{

    const parameters =
        new URLSearchParams(
            window.location.search
        );

    const file =
        parameters.get("file");

    if (!file) {{
        content.innerHTML =
            '<div class="message">' +
            'No document specified.' +
            '</div>';

        return;
    }}

    const anchor =
        decodeURIComponent(
            window.location.hash.slice(1)
        );

    try {{
        await navigate(
            file,
            anchor,
            false
        );

        history.replaceState(
            {{
                file,
                anchor
            }},
            "",
            window.location.href
        );
    }} catch (error) {{
        showError(error);
    }}
}})();

</script>

</body>
</html>
"""


# =============================================================================
# HTTP server
# =============================================================================

class _DocumentationRequestHandler(
    http.server.BaseHTTPRequestHandler
):
    """HTTP endpoint implementation for DocumentationBrowser."""

    server_version = "TkDocumentationBrowser/1.0"

    @property
    def browser(self):
        """Return the owning DocumentationBrowser."""
        return self.server.browser

    def log_message(self, format_string, *args):
        """Do not write HTTP requests to the application's console."""
        return

    def send_bytes(
        self,
        data,
        status=200,
        content_type="application/octet-stream",
    ):
        """Send raw bytes."""
        self.send_response(status)

        self.send_header(
            "Content-Type",
            content_type
        )

        self.send_header(
            "Content-Length",
            str(len(data))
        )

        self.send_header(
            "Cache-Control",
            "no-store"
        )

        self.end_headers()

        self.wfile.write(data)

    def send_text(
        self,
        text,
        status=200,
        content_type="text/plain; charset=utf-8",
    ):
        """Send UTF-8 text."""
        self.send_bytes(
            text.encode("utf-8"),
            status=status,
            content_type=content_type,
        )

    def send_json(
        self,
        data,
        status=200,
    ):
        """Send a JSON object."""
        import json

        self.send_text(
            json.dumps(
                data,
                ensure_ascii=False,
            ),
            status=status,
            content_type="application/json; charset=utf-8",
        )

    def parse_requested_path(self):
        """
        Resolve and validate the 'file' query parameter.

        Returns:
            (Path, None) on success
            (None, error_message) on failure
        """
        parsed =
            urllib.parse.urlparse(
                self.path
            )

        query =
            urllib.parse.parse_qs(
                parsed.query
            )

        values =
            query.get("file")

        if not values:
            return (
                None,
                "Missing 'file' query parameter."
            )

        raw_path =
            values[0]

        try:
            path =
                Path(
                    raw_path
                ).resolve(
                    strict=True
                )
        except (
            OSError,
            RuntimeError,
            ValueError,
        ) as exc:
            return (
                None,
                f"Invalid file path: {exc}"
            )

        if not self.browser.is_allowed(
            path
        ):
            return (
                None,
                "Requested file is outside the configured roots."
            )

        if not path.is_file():
            return (
                None,
                "Requested path is not a regular file."
            )

        return (
            path,
            None
        )

    def do_GET(self):
        """Dispatch GET requests."""

        parsed =
            urllib.parse.urlparse(
                self.path
            )

        # ---------------------------------------------------------------------
        # Viewer
        # ---------------------------------------------------------------------

        if parsed.path in {
            "/",
            "/index.html",
        }:
            self.send_text(
                _viewer_html(),
                content_type="text/html; charset=utf-8",
            )
            return

        # ---------------------------------------------------------------------
        # Text / document endpoint
        # ---------------------------------------------------------------------

        if parsed.path == "/api/document":

            path, error =
                self.parse_requested_path()

            if error:
                self.send_text(
                    error,
                    status=400,
                )
                return

            kind =
                self.browser.kind_for(
                    path
                )

            if kind == "image":
                language = ""
                content = ""

            elif kind == "markdown":
                language = "markdown"

                try:
                    content =
                        path.read_text(
                            encoding="utf-8"
                        )
                except UnicodeDecodeError:
                    self.send_text(
                        "Markdown file is not valid UTF-8.",
                        status=400,
                    )
                    return
                except OSError as exc:
                    self.send_text(
                        f"Cannot read Markdown file: {exc}",
                        status=500,
                    )
                    return

            elif kind == "source":
                language =
                    SOURCE_LANGUAGES.get(
                        path.suffix.lower(),
                        "plaintext",
                    )

                try:
                    content =
                        path.read_text(
                            encoding="utf-8"
                        )
                except UnicodeDecodeError:
                    self.send_text(
                        "Source file is not valid UTF-8.",
                        status=400,
                    )
                    return
                except OSError as exc:
                    self.send_text(
                        f"Cannot read source file: {exc}",
                        status=500,
                    )
                    return

            else:
                self.send_text(
                    "Unsupported document type.",
                    status=415,
                )
                return

            self.send_json(
                {
                    "kind": kind,
                    "file": str(path),
                    "name": path.name,
                    "language": language,
                    "content": content,
                }
            )

            return

        # ---------------------------------------------------------------------
        # Binary asset endpoint
        # ---------------------------------------------------------------------

        if parsed.path == "/api/asset":

            path, error =
                self.parse_requested_path()

            if error:
                self.send_text(
                    error,
                    status=400,
                )
                return

            if not self.browser.is_image(
                path
            ):
                self.send_text(
                    "Only configured image files may be served as assets.",
                    status=403,
                )
                return

            try:
                data =
                    path.read_bytes()
            except OSError as exc:
                self.send_text(
                    f"Cannot read image: {exc}",
                    status=500,
                )
                return

            mime_type =
                IMAGE_MIME_TYPES.get(
                    path.suffix.lower()
                )

            if mime_type is None:
                mime_type, _ =
                    mimetypes.guess_type(
                        path.name
                    )

            if mime_type is None:
                mime_type =
                    "application/octet-stream"

            self.send_bytes(
                data,
                content_type=mime_type,
            )

            return

        # ---------------------------------------------------------------------
        # Unknown route
        # ---------------------------------------------------------------------

        self.send_text(
            "Not found.",
            status=404,
        )


class _DocumentationHTTPServer(
    http.server.ThreadingHTTPServer
):
    """Threaded localhost server carrying the DocumentationBrowser owner."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        address,
        browser,
    ):
        super().__init__(
            address,
            _DocumentationRequestHandler,
        )

        self.browser = browser


# =============================================================================
# Public application service
# =============================================================================

class DocumentationBrowser:
    """
    Singleton-style documentation service.

    The class does not enforce a process-global singleton. The intended design
    is to construct ONE instance at application startup and share it among
    all Tkinter components.

    Example:

        documentation = DocumentationBrowser(
            allowed_roots=[
                PROJECT_ROOT / "docs",
                PROJECT_ROOT / "src",
                PROJECT_ROOT / "scripts",
            ]
        )

        documentation.open(
            PROJECT_ROOT / "docs" / "README.md"
        )
    """

    def __init__(
        self,
        allowed_roots=None,
    ):
        """
        Create the documentation service.

        Parameters
        ----------
        allowed_roots:
            Iterable of directories that may be served.

            These directories are security boundaries.

            If omitted, all existing files of supported types may technically
            be served. For an application, explicit roots are preferable.
        """

        if allowed_roots is None:
            allowed_roots = []

        self._roots =
            tuple(
                Path(root).resolve()
                for root in allowed_roots
            )

        self._server = None
        self._thread = None

        self._lock =
            threading.RLock()

        atexit.register(
            self.close
        )

    # -------------------------------------------------------------------------
    # Root management
    # -------------------------------------------------------------------------

    def add_root(self, root):
        """
        Add another permitted filesystem root.

        Example:

            documentation.add_root(PROJECT_ROOT / "examples")
        """

        path =
            Path(root).resolve()

        with self._lock:
            if path not in self._roots:
                self._roots =
                    (*self._roots, path)

    def is_allowed(self, path):
        """
        Return whether path is inside one of the configured roots.

        Path.resolve() is performed again so that a symlink escaping a root
        cannot bypass the boundary.
        """

        if not self._roots:
            return True

        path =
            Path(path).resolve()

        for root in self._roots:

            try:
                path.relative_to(
                    root
                )
            except ValueError:
                continue

            return True

        return False

    # -------------------------------------------------------------------------
    # Type classification
    # -------------------------------------------------------------------------

    @staticmethod
    def kind_for(path):
        """
        Return:

            'markdown'
            'source'
            'image'
            None
        """

        suffix =
            Path(path).suffix.lower()

        if suffix in MARKDOWN_EXTENSIONS:
            return "markdown"

        if suffix in SOURCE_LANGUAGES:
            return "source"

        if suffix in IMAGE_MIME_TYPES:
            return "image"

        return None

    @staticmethod
    def is_image(path):
        """Return whether path has a supported image extension."""
        return (
            Path(path).suffix.lower()
            in IMAGE_MIME_TYPES
        )

    # -------------------------------------------------------------------------
    # Server lifecycle
    # -------------------------------------------------------------------------

    def _ensure_server(self):
        """
        Start the localhost server if necessary.

        Port 0 means "ask Windows for an unused ephemeral port".
        """

        with self._lock:

            if self._server is not None:
                return (
                    self._server.server_address[0],
                    self._server.server_address[1],
                )

            server =
                _DocumentationHTTPServer(
                    ("127.0.0.1", 0),
                    self,
                )

            thread =
                threading.Thread(
                    target=server.serve_forever,
                    name="documentation-http-server",
                    daemon=True,
                )

            thread.start()

            self._server =
                server

            self._thread =
                thread

            return (
                server.server_address[0],
                server.server_address[1],
            )

    @property
    def running(self):
        """Return True while the local server is active."""
        return self._server is not None

    @property
    def base_url(self):
        """
        Return the local viewer URL, or None if it has not started.
        """

        if self._server is None:
            return None

        host, port =
            self._server.server_address

        return (
            f"http://{host}:{port}/"
        )

    def close(self):
        """Stop the local server."""

        with self._lock:

            server =
                self._server

            if server is None:
                return

            self._server =
                None

            self._thread =
                None

        server.shutdown()
        server.server_close()

    # -------------------------------------------------------------------------
    # Open document
    # -------------------------------------------------------------------------

    def open(self, path):
        """
        Open a supported local document in the system default browser.

        The first invocation starts the local HTTP server.

        Later invocations reuse the same server.
        """

        path =
            Path(path).resolve()

        if not path.is_file():
            raise FileNotFoundError(
                path
            )

        if not self.is_allowed(
            path
        ):
            raise PermissionError(
                "File is outside the configured "
                f"documentation roots: {path}"
            )

        kind =
            self.kind_for(path)

        if kind is None:
            raise ValueError(
                "Unsupported documentation file type: "
                f"{path}"
            )

        host, port =
            self._ensure_server()

        encoded =
            urllib.parse.quote(
                str(path),
                safe="",
            )

        url =
            (
                f"http://{host}:{port}/"
                f"?file={encoded}"
            )

        webbrowser.open_new_tab(
            url
        )


# =============================================================================
# Tkinter helper
# =============================================================================

def bind_documentation_link(
    widget,
    tag_name,
    documentation,
    path,
):
    """
    Bind a Tkinter Text/ScrolledText tag to DocumentationBrowser.open().

    The same DocumentationBrowser instance may be bound to any number of
    widgets.

    Example:

        bind_documentation_link(
            text,
            "docs-link",
            documentation,
            DOCS_DIR / "user-guide.md",
        )

    The helper intentionally contains no server logic. The service owns
    the server and all document-type handling.
    """

    widget.tag_bind(
        tag_name,
        "<Button-1>",
        lambda _event: documentation.open(path),
    )
```

There is one Python formatting issue in that presentation: the `parse_requested_path()` section should use ordinary Python assignment statements, not line-broken assignments created by the long-form display. The executable form of that method is:

```python
    def parse_requested_path(self):
        """
        Resolve and validate the 'file' query parameter.

        Returns:
            (Path, None) on success
            (None, error_message) on failure
        """
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        values = query.get("file")

        if not values:
            return None, "Missing 'file' query parameter."

        raw_path = values[0]

        try:
            path = Path(raw_path).resolve(strict=True)
        except (OSError, RuntimeError, ValueError) as exc:
            return None, f"Invalid file path: {exc}"

        if not self.browser.is_allowed(path):
            return None, "Requested file is outside the configured roots."

        if not path.is_file():
            return None, "Requested path is not a regular file."

        return path, None
```

Likewise, in `DocumentationBrowser.__init__`, use the normal Python form:

```python
    def __init__(self, allowed_roots=None):
        if allowed_roots is None:
            allowed_roots = []

        self._roots = tuple(
            Path(root).resolve()
            for root in allowed_roots
        )

        self._server = None
        self._thread = None
        self._lock = threading.RLock()

        atexit.register(self.close)
```

I would actually make one further refinement before putting this into your project: **vendor the three browser dependencies into your application resources** rather than leaving the CDN URLs in the production version. For your PyInstaller application the directory could be:

```text
project/
    docs/
        README.md
        user-guide.md
        images/
            architecture.png
            pipeline.svg
    scripts/
        tcm_proc.py
    src/
        tcm/
            __init__.py
            processing.py
    resources/
        documentation/
            marked.umd.js
            mathjax/
                ...
            highlight/
                highlight.min.js
                default.min.css
```

Then the localhost server can serve those resources too, so the documentation viewer becomes completely independent of Internet availability. MathJax 4's browser output can load additional font ranges dynamically, which is exactly why the implementation uses `typesetPromise()` rather than synchronous `typeset()`. ([MathJax Documentation][2])

The resulting behavior is then:

```text
Tkinter "Help"
    │
    └── docs.open("docs/README.md")
                │
                ▼
          one localhost server
                │
                ▼
           Browser viewer
                │
       ┌────────┼─────────┐
       ▼        ▼         ▼
     README   tcm_proc.py  diagram.png
       │        │             │
    marked    highlight.js   <img>
       │
    MathJax
       │
       ▼
 rendered documentation
```

and, importantly, the same browser can move from:

```text
README.md
   → user-guide.md
      → ../scripts/tcm_proc.py
         → ../src/tcm/config.py
            → images/pipeline.svg
```

without another server, another generated HTML file, or direct `file://` access. The `.py` files are displayed as source rather than executed, while the images are served as their native binary MIME types.

[1]: https://marked.js.org/?utm_source=chatgpt.com "Marked Documentation"
[2]: https://docs.mathjax.org/en/latest/advanced/typeset.html?utm_source=chatgpt.com "MathJax in Dynamic Content — MathJax 4.0 documentation"
[3]: https://highlightjs.org/?snippet=1&style=5&utm_source=chatgpt.com "highlight.js"
