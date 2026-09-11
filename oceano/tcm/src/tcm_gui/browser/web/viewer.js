"use strict";
const content = document.getElementById("content");
const sourcePanel = document.getElementById("source-panel");
const sourceCode = document.getElementById("source-code");
const imagePanel = document.getElementById("image-panel");
const imageView = document.getElementById("image-view");
const directoryPanel = document.getElementById("directory-panel");
const directoryTitle = document.getElementById("directory-title");
const directoryParent = document.getElementById("directory-parent");
const directoryListEl = document.getElementById("directory-list");
const fileListEl = document.getElementById("file-list");
const bs = "\\";  // one literal backslash, for building regexes from strings

/* Document classes — mirrors the Python tables in documentation_browser.py.
   Only languages present in this repository (py sources, web assets, configs). */
const markdownExts = new Set([".md", ".markdown"]);
const sourceLanguages = {
    "py": "python", "pyi": "python",
    "js": "javascript", "mjs": "javascript", "cjs": "javascript",
    "css": "css", "html": "xml", "htm": "xml", "xml": "xml",
    "json": "json", "yaml": "yaml", "yml": "yaml", "toml": "toml",
    "ini": "ini", "cfg": "ini", "txt": "plaintext", "log": "plaintext",
    "ps1": "powershell", "bat": "dos", "cmd": "dos", "sh": "shell"
};
const imageExts = new Set([".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp", ".ico", ".avif"]);

function extOf(p) {
    const name = p.split(/[\\/]/).pop().toLowerCase();
    const i = name.lastIndexOf(".");
    return i < 0 ? "" : name.slice(i);
}

function kindOf(p) {
    const ext = extOf(p);
    if (markdownExts.has(ext)) return "markdown";
    if (Object.hasOwn(sourceLanguages, ext.slice(1))) return "source";
    if (imageExts.has(ext)) return "image";
    return "";
}

let currentFile = "", currentKind = "", currentLanguage = "";
let navigationSerial = 0;  // a stale in-flight fetch must not overwrite a newer navigation

function currentUrlFile() {
    return new URLSearchParams(window.location.search).get("file");
}

function winPath(pathname) {
    /* file: URL.pathname → filesystem path (drop the / before a drive letter). */
    return pathname.replace(/^\/([A-Za-z]:)/, "$1");
}

function resolveLinkUrl(baseFile, href) {
    /* file:// URL (written by preprocessMdLinks/fixRenderedLinks) → path + anchor. */
    if (/^file:\/\//i.test(href)) {
        try {
            const u = new URL(href);
            return {file: winPath(u.pathname), anchor: decodeURIComponent(u.hash.slice(1))};
        } catch(e) { /* fall through to raw-path handling below */ }
    }
    /* Absolute Windows / UNC path → normalize separators to /. */
    if (/^[A-Za-z]:[\\/]/.test(href) || /^\\\\/.test(href))
        return {file: href.replace(/\\/g, "/"), anchor: href.includes("#") ? href.split("#").slice(1).join("#") : ""};
    /* Relative path → resolve against the current document's directory. */
    const baseDir = baseFile.replace(/\\/g, "/").replace(/\/[^/]*$/, "");
    const resolved = new URL(href, "file:///" + baseDir + "/");
    return {file: winPath(resolved.pathname), anchor: decodeURIComponent(resolved.hash.slice(1))};
}

function resolveDirHref(href) {
    /* Directory-listing hrefs are bare names relative to currentFile, which
       IS the folder — resolve against it directly (no segment to strip). */
    const base = currentFile.replace(/\\/g, "/").replace(/[\\/]+$/, "") + "/";
    return winPath(new URL(href, "file:///" + base).pathname);
}

function fixRenderedLinks(baseFile) {
    /* Rewrite scheme-less .md hrefs marked emitted verbatim against the
       document's dir (the HTTP origin knows only the server root) — covers
       inline, reference-style and autolink-embedded links alike. */
    const baseDir = baseFile.replace(/\\/g, "/").replace(/\/[^/]*$/, "");
    content.querySelectorAll("a[href]").forEach(a => {
        const href = a.getAttribute("href");
        if (!href || href.startsWith("#") || /^[A-Za-z]+:/i.test(href)) return;
        if (!/\.md$/i.test(href.split("#")[0])) return;
        a.setAttribute("href", new URL(href, "file:///" + baseDir + "/").href);
    });
}

function rewriteLocalImages(baseFile) {
    /* getAttribute returns the RAW src (the browser resolves relative srcs
       against the server root, which knows nothing of the document's dir)
       → point local images at the /api/asset endpoint. */
    content.querySelectorAll("img[src]").forEach(img => {
        const src = img.getAttribute("src");
        if (!src || /^(https?:|data:|file:)/i.test(src)) return;
        const {file} = resolveLinkUrl(baseFile, src);
        if (kindOf(file) !== "image") return;
        img.src = `/api/asset?file=${encodeURIComponent(file)}`;
        img.loading = "lazy";
        img.decoding = "async";
    });
}

function slugify(text) {
    /* GitHub heading anchor: {#explicit-id} suffix wins; else lowercase,
       strip tags, drop non-word/space/hyphen, EACH space → - (so " — " → "--"). */
    const m = text.match(new RegExp(bs + "{#([^" + bs + "{}]+)}$"));
    if (m) return m[1];
    return text.toLowerCase()
        .replace(/<[^>]*>/g, "")
        .replace(/[^\p{L}\p{N} _-]+/gu, "")
        .trim().replace(/ /g, "-");
}

function addHeadingIds() {
    /* marked emits headings without ids — slug them so same-file (#section)
       and cross-file (doc.md#section) anchors have something to land on. */
    const used = {};
    content.querySelectorAll("h1,h2,h3,h4,h5,h6").forEach(h => {
        if (h.id) return;
        const base = slugify(h.textContent);
        h.id = used[base] ? `${base}-${used[base]++}` : (used[base] = 1, base);
    });
}

function wrapTables() {
    /* Horizontal scroll on narrow windows. */
    content.querySelectorAll("table").forEach(t => {
        if (t.parentElement?.classList.contains("table-wrapper")) return;
        const w = document.createElement("div");
        w.className = "table-wrapper";
        t.parentNode.insertBefore(w, t);
        w.appendChild(t);
    });
}

/*MATH-EXT-START — marked extensions tokenizing GitHub math ($…$ / $$…$$ /
   $`…`$) before CommonMark runs: the TeX interior (_, *, [], {}, `) is
   consumed raw by the tokenizer, so Markdown can never reinterpret it, and
   the renderer emits literal-TeX spans MathJax later typesets from the DOM.
   Inline extensions run before built-in inline tokenizers (codespan etc.) at
   every position, and code spans/fences consume their own text first —
   TeX-looking text inside code stays literal.  $-inline contract (GitHub
   spacing rules, single line): opening $ followed by non-whitespace (skips
   currency), closing $ preceded by non-whitespace, content without bare $
   (escape pairs \$ allowed). */
const escapeHtml = (value) => value
    .replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");

const MATH_INLINE = {
    name: "mathInline",
    level: "inline",
    start(src) { const i = src.indexOf("$"); return i < 0 ? undefined : i; },
    tokenizer(src) {
        const bt = src.startsWith("$`") ? src.indexOf("`$", 2) : -1;   // $`…`$ —
        if (bt >= 0)                                                   // wrapper stripped
            return {type: "mathInline", raw: src.slice(0, bt + 2), text: src.slice(2, bt), display: false};
        if (!src.startsWith("$") || src.startsWith("$$")) return;
        let end = 1;                       // first unescaped closing $ on the line
        while ((end = src.indexOf("$", end)) >= 0 && src[end - 1] === "\\") end++;
        if (end < 0) return;
        const text = src.slice(1, end);
        if (!text || /\n/.test(text) || /\s/.test(text[0]) || /\s$/.test(text)) return;
        return {type: "mathInline", raw: src.slice(0, end + 1), text, display: false};
    },
    renderer(token) { return `<span class="math-inline">$${escapeHtml(token.text)}$</span>`; },
};

const MATH_BLOCK = {
    name: "mathBlock",
    level: "block",
    start(src) { const m = src.match(/^\$\$/m); return m ? m.index : undefined; },
    tokenizer(src) {
        const m = src.match(/^\$\$[^\S\n]*\n?([\s\S]*?)\n?\$\$(?:\n+|$)/);
        return m ? {type: "mathBlock", raw: m[0], text: m[1], display: true} : undefined;
    },
    renderer(token) { return `<div class="math-block">$$\n${escapeHtml(token.text)}\n$$</div>`; },
};

marked.use({gfm: true, breaks: false, extensions: [MATH_INLINE, MATH_BLOCK]});
/*MATH-EXT-END*/

function scrollToAnchor(anchor) {
    /* after typesetting — layout may have shifted; getElementById avoids
       CSS-selector interpretation of the anchor */
    requestAnimationFrame(() => document.getElementById(anchor)?.scrollIntoView({block: "start"}));
}

function scrollToSourceLine(anchor) {
    /* #L42 convention: white-space:pre keeps every row exactly line-height tall */
    const m = /^L(\d+)$/i.exec(anchor);
    if (!m) return;
    const lineHeight = parseFloat(getComputedStyle(sourceCode).lineHeight);
    const top = sourceCode.getBoundingClientRect().top + window.scrollY + (Number(m[1]) - 1) * lineHeight;
    window.scrollTo(0, Math.max(0, top - 12));  /* small breathing space */
}

function hideAllViews() {
    content.style.display = "none";
    sourcePanel.style.display = "none";
    imagePanel.style.display = "none";
    directoryPanel.style.display = "none";
}

/* "block", never "" — the stylesheet hides the panels by default,
   and clearing the inline style would fall back to that hidden state (blank page). */
function showRenderedView() {
    hideAllViews();
    content.style.display = "block";
}

function showSourceView() {
    hideAllViews();
    sourcePanel.style.display = "block";
}

function showImageView() {
    hideAllViews();
    imagePanel.style.display = "block";
}

function showDirectoryView() {
    hideAllViews();
    directoryPanel.style.display = "block";
}

/* Directory links are marked by a trailing slash (see renderDirectory); the
   delegated click handler resolves them to a folder before the file checks. */
function isDirHref(href) {
    return /[\\/]$/.test(href);
}

async function fetchDirectory(dir) {
    const r = await fetch(`/api/directory?path=${encodeURIComponent(dir)}`, {cache: "no-store"});
    if (!r.ok) throw new Error(`HTTP ${r.status} while loading directory ${dir}

${await r.text()}`);
    return r.json();
}

function renderDirectory(dir) {
    /* The server returns relative hrefs (folder names + "/", file names) —
       the browser resolves them against the current dir, so navigation is a
       pure chain of the same delegated click handler. */
    directoryTitle.textContent = dir.name || dir.path;
    directoryParent.replaceChildren();
    directoryListEl.replaceChildren();
    fileListEl.replaceChildren();
    if (dir.parent) {
        const li = document.createElement("li");
        const up = document.createElement("a");
        up.href = dir.parent;  // "../" — isDirHref → navigateDirectory
        up.textContent = "..";
        li.appendChild(up);
        directoryParent.appendChild(li);
    }
    for (const item of dir.directories) {
        const li = document.createElement("li");
        const a = document.createElement("a");
        a.href = item.href;  // "examples/"
        a.textContent = item.name + "/";
        li.appendChild(a);
        directoryListEl.appendChild(li);
    }
    for (const item of dir.files) {
        const li = document.createElement("li");
        const a = document.createElement("a");
        a.href = item.href;  // "algorithms.md"
        a.textContent = item.name;
        li.appendChild(a);
        fileListEl.appendChild(li);
    }
    showDirectoryView();
    currentFile = dir.path; currentKind = "directory"; currentLanguage = "";
}

async function navigateDirectory(dir, push = false) {
    const serial = ++navigationSerial;
    try {
        const listing = await fetchDirectory(dir);
        if (serial !== navigationSerial) return;  // superseded by a newer navigation
        renderDirectory(listing);
        document.title = (listing.name || listing.path).split(/[\\/]/).pop();
        if (push) history.pushState({file: dir, anchor: "", kind: "directory"}, "",
            `/?file=${encodeURIComponent(dir)}`);
    } catch (e) { showError(e); }
}

async function fetchDocument(file) {
    const r = await fetch(`/api/document?file=${encodeURIComponent(file)}`, {cache: "no-store"});
    if (!r.ok) throw new Error(`HTTP ${r.status} while loading ${file}

${await r.text()}`);
    return r.json();
}

async function renderMarkdown(md, file, anchor = "") {
    /* MathJax must forget the previous page before its DOM is replaced. */
    window.MathJax?.typesetClear?.([content]);
    /* marked math extensions (MATH-EXT block) keep the TeX interior raw —
       MathJax finds the literal $…$ / $$…$$ delimiters in the emitted spans. */
    content.innerHTML = marked.parse(md);
    fixRenderedLinks(file);
    rewriteLocalImages(file);
    addHeadingIds();
    wrapTables();
    currentFile = file; currentKind = "markdown"; currentLanguage = "markdown";
    showRenderedView();
    /* Content did not exist at MathJax startup → typeset per page; await
       startup first — typesetPromise is undefined until it completes. */
    if (window.MathJax) {
        await (window.MathJax.startup?.promise ?? Promise.resolve());
        await window.MathJax.typesetPromise([content]);
    }
    if (anchor) scrollToAnchor(anchor);
    else window.scrollTo(0, 0);
}

function renderSource(text, file, language, anchor = "") {
    /* textContent — verbatim source, never interpreted as HTML */
    sourceCode.textContent = text;
    sourceCode.className = `language-${language}`;
    /* bfcache restore keeps the element's data-highlighted marker → the native
       (re)highlight would be skipped on Back navigation; clear it first */
    sourceCode.removeAttribute("data-highlighted");
    window.hljs?.highlightElement(sourceCode);
    showSourceView();
    currentFile = file; currentKind = "source"; currentLanguage = language;
    if (anchor) scrollToSourceLine(anchor);
}

function renderImage(file) {
    imageView.src = `/api/asset?file=${encodeURIComponent(file)}`;
    imageView.alt = file.split(/[\\/]/).pop() || "Image";
    showImageView();
    currentFile = file; currentKind = "image";
}

async function navigate(file, anchor = "", push = false) {
    const serial = ++navigationSerial;
    if (push) content.innerHTML = '<div class="message">Loading…</div>';
    try {
        const kind = kindOf(file);
        if (kind === "image") renderImage(file);
        else {
            const d = await fetchDocument(file);
            if (serial !== navigationSerial) return;  // superseded by a newer navigation
            if (kind === "markdown") await renderMarkdown(d.content, d.file, anchor);
            else if (kind === "source") renderSource(d.content, d.file, d.language, anchor);
            else throw new Error(`Unsupported document type: ${file}`);
        }
        document.title = file.split(/[\\/]/).pop();
        if (push) history.pushState({file, anchor}, "",
            `/?file=${encodeURIComponent(file)}` + (anchor ? `#${encodeURIComponent(anchor)}` : ""));
    } catch (e) { showError(e); }
}

document.addEventListener("click", e => {
    const a = e.target.closest("a");
    if (!a) return;
    const href = a.getAttribute("href");
    if (!href) return;
    /* External links → let the browser handle them. */
    if (/^(https?:|mailto:|ftp:)/i.test(href)) return;

    /* Directory listings are ordinary <a> elements; their links resolve
       against currentFile (the listed folder), not a document directory. */
    if (currentKind === "directory") {
        if (href.startsWith("#")) return;  // none emitted; be safe
        e.preventDefault();
        const file = resolveDirHref(href);
        if (isDirHref(href)) navigateDirectory(file, true);
        else navigate(file, "", true);
        return;
    }

    const base = currentUrlFile();
    if (!base) return;
    if (href.startsWith("#")) {
        /* Same-file anchor → history entry + scroll. */
        e.preventDefault();
        const anchor = decodeURIComponent(href.slice(1));
        history.pushState({file: base, anchor}, "",
            `/?file=${encodeURIComponent(base)}#${encodeURIComponent(anchor)}`);
        currentKind === "markdown" ? scrollToAnchor(anchor) : scrollToSourceLine(anchor);
        return;
    }
    const {file, anchor} = resolveLinkUrl(base, href);
    if (isDirHref(href)) {  // folder link from a rendered document
        e.preventDefault();
        navigateDirectory(file, true);
        return;
    }
    if (!kindOf(file)) return;  // unsupported class → leave the link to the browser
    e.preventDefault();
    navigate(file, anchor, true);
});

window.addEventListener("popstate", e => {
    if (!e.state?.file) return;
    e.state.kind === "directory"
        ? navigateDirectory(e.state.file)
        : navigate(e.state.file, e.state.anchor || "");
});

function showError(err) {
    console.error(err);
    content.innerHTML = '<div class="message"><h2 id="error">Cannot display document</h2><pre>' +
        String(err).replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;") + '</pre></div>';
    hideAllViews();
    content.style.display = "block";
}

(async () => {
    const file = currentUrlFile();
    if (!file) { content.innerHTML = '<div class="message">No document was specified.</div>'; return; }
    const anchor = decodeURIComponent(window.location.hash.slice(1));
    try {
        await navigate(file, anchor);
        history.replaceState({file, anchor}, "", window.location.href);
    } catch (e) { showError(e); }
})();
