"""About dialog: version, runtime info, documentation browser.

Opened by the ``?`` button in the search path row.  Displays:
- Product name + version (from ``version_meta.json``)
- Repository URL + clickable Documentation label (→ browser)
- Discovered ``*.md`` docs from the bundled ``docs/`` directory

Runtime status (full/simple mode, HDF5 availability) lives in the
window system title, not the body.  All chrome labels AND the translatable
meta values (description, company) are i18n via :data:`STRINGS`
(``about.*`` keys); doc discovery is filtered to the application language
(:func:`_lang_filter`).

Doc titles are extracted from the first ``# `` heading of each file and
listed in a directory-nested ``ttk.Treeview`` (only the first level expanded;
a folder holding ``_index.md`` links its parent row to that file with no
separate ``_index.md`` leaf); clicking a title opens the document in the OS
browser via :func:`~tcm_gui.browser.open_md_link` — a localhost server serves
the document, rendered client-side by marked.js / MathJax / highlight.js.

Markdown note: ``parse_markdown`` joins consecutive lines into ONE
paragraph, so every distinct metadata field must be its own block
(separated by blank lines, or a ``- `` list item).
"""

from __future__ import annotations

import logging
import re
import tkinter as tk
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from tkinter import font as tkfont
from tkinter import ttk

from tcm._constants import DOC_DIR, H5_AVAILABLE, version_meta

from . import theme
from ._i18n import STRINGS as _S
from ._i18n import resolve_lang
from ._rtf_clipboard import copy_rich
from .browser import open_md_link
from .const import UIScale, work_area
from .md_label import MarkdownLabel
from .theme import _opt_into_dark_titlebar, mix_hex

_l = logging.getLogger(__name__)

_RE_HEADING = re.compile(r"^#\s+(.+?)\s*$")

# Language suffix on doc stems: ``config_reference_ru`` → ("config_reference", "ru").
_RE_LANG_SUFFIX = re.compile(r"_(?P<lang>[a-z]{2})$", re.IGNORECASE)

# Logo bundled with docs/ (readme.md references the same file).
_LOGO_FILE = DOC_DIR / "images" / "logo.png"

# Docs subdirs to exclude from discovery (internal/TODO/implemented-notes).
_EXCLUDE_DIRS = {"todo", "done"}

# Base dialog size — first guess only: _refit measures the content and
# resizes the height to it (avoids a big map→refit jump; width stays fixed).
_W, _H = 560, 720

# Vertical padding px of the packed children (see _build) — the fixed chrome
# part of the window height, with NO spare (spare would feed back into the
# _refit size and accumulate on every reflow).
_PADS = (16, 4, 4, 12)

# Px reserved before a row's text: disclosure indicator + icon, plus one
# indent level more for children (folder → leaf).  Wrap-width budget.
_ICON_PX, _INDENT_PX = 24, 24


# ── doc discovery ───────────────────────────────────────────────────────────


def discover_docs(root: Path | None = None, lang: str | None = None) -> list[tuple[str, str, Path]]:
    """Walk ``docs/`` for ``*.md`` files; return ``(folder, title, path)`` sorted by path.

    Title = first ``# `` heading; fallback = filename stem.
    Folder = parent dir name relative to docs root (``""`` for top-level files).
    Excludes ``todo/`` and other internal dirs, then filters by *lang*
    (default: :func:`resolve_lang`) — see :func:`_lang_filter`.
    """
    docs_root = root or DOC_DIR
    if not docs_root.is_dir():
        _l.warning("Docs root not found: %s", docs_root)
        return []

    results = []
    for md in sorted(docs_root.rglob("*.md")):
        # Exclude internal dirs (todo, etc.)
        rel = md.relative_to(docs_root)
        if any(part in _EXCLUDE_DIRS for part in rel.parts[:-1]):
            continue
        title = _extract_title(md) or md.stem
        results.append((rel.parts[0] if len(rel.parts) > 1 else "", title, md))
    return _lang_filter(results, lang or resolve_lang())


def _lang_parts(stem: str) -> tuple[str, str | None]:
    """Split doc stem into ``(base, lang | None)``: ``readme_noh5_Ru`` → ``("readme_noh5", "ru")``."""
    if m := _RE_LANG_SUFFIX.search(stem):
        return stem[: m.start()], m.group("lang").casefold()
    return stem, None


def local_readme() -> Path:
    """Bundled readme for :func:`resolve_lang`, else the base ``readme.md``.

    Uses the same :func:`_lang_parts` matching as doc discovery so any
    future ``readme_<lang>.md`` is picked up without further changes.
    """
    lang = resolve_lang()
    candidates = [p for p in DOC_DIR.parent.glob("readme*.md") if _lang_parts(p.stem)[1] == lang]
    return candidates[0] if candidates else DOC_DIR.parent / "readme.md"


def _lang_filter(docs: list[tuple[str, str, Path]], lang: str) -> list[tuple[str, str, Path]]:
    """Language-relevant subset of *docs* (already sorted by path).

    ``en`` → only unsuffixed files (``*_ru.md`` etc. are translations).
    Other *lang* → per base name prefer the ``_{lang}`` version; without it,
    keep the unsuffixed original; without either, any available translation.
    """
    if lang == "en":
        return [d for d in docs if _lang_parts(d[2].stem)[1] is None]

    groups: dict[tuple[str, str], list[tuple[str, str, Path]]] = defaultdict(list)
    for d in docs:
        groups[(d[0], _lang_parts(d[2].stem)[0])].append(d)

    out = []
    for group in groups.values():
        localized = [d for d in group if _lang_parts(d[2].stem)[1] == lang]
        plain = [d for d in group if _lang_parts(d[2].stem)[1] is None]
        out += localized or plain or group[:1]
    return sorted(out, key=lambda d: d[2])


def _docs_tree(docs: list[tuple[str, str, Path]], docs_root: Path = DOC_DIR) -> dict[str, object]:
    """Build directory-nested docs hierarchy for treeview display.

    Content comes solely from the actual ``folder/*.md`` structure (titles are
    each file's first ``# `` heading); ``readme.md`` supplies only sibling order.
    Folders containing ``_index.md`` store it as ``node["index"]`` so the parent
    row links directly to it instead of showing a separate ``_index.md`` leaf.

    Returns the root node ``{"index": None, "files": [...], "sub": {...}}`` where
    each sub node has the same shape.  Top-level and nested siblings sort by
    first-appearance order of their ``docs/...`` hrefs in ``readme.md``; paths
    absent there sort alphabetically after.
    """
    root: dict[str, object] = {"index": None, "files": [], "sub": {}}
    for _folder, title, path in docs:
        try:
            rel = path.relative_to(docs_root)
        except ValueError:
            continue
        node = root  # type: ignore[assignment]
        for part in rel.parts[:-1]:
            sub = node["sub"]  # type: ignore[index]
            if part not in sub:
                sub[part] = {"index": None, "files": [], "sub": {}}
            node = sub[part]
        if rel.name == "_index.md":
            node["index"] = (title, path)  # type: ignore[index]
        else:
            node["files"].append((title, path))  # type: ignore[index]
    order = _readme_doc_order(docs_root)
    inf = len(order)

    def _rel(p: Path) -> str:
        try:
            return p.relative_to(docs_root).as_posix()
        except ValueError:
            return p.name

    def _sub_min(node: dict[str, object]) -> int:
        best = inf
        if node.get("index") is not None:
            _, ip = node["index"]  # type: ignore[misc]
            best = min(best, order.get(_rel(ip), inf))
        for _, fp in node.get("files", []):  # type: ignore[union-attr]
            best = min(best, order.get(_rel(fp), inf))
        for child in node.get("sub", {}).values():  # type: ignore[union-attr]
            best = min(best, _sub_min(child))  # type: ignore[arg-type]
        return best

    def _sort_node(node: dict[str, object], prefix: tuple[str, ...]) -> None:
        files = node.get("files", [])  # type: ignore[assignment]
        files.sort(key=lambda t: (order.get(_rel(t[1]), inf), t[0].lower(), t[1].name.lower()))  # type: ignore[union-attr]

        def _sub_key(name: str) -> tuple[int, str]:
            child = node["sub"][name]  # type: ignore[index]
            own = order.get("/".join((*prefix, name)), inf)
            return (min(own, _sub_min(child)), name.lower())

        node["sub"] = dict(sorted(node["sub"].items(), key=lambda kv: _sub_key(kv[0])))  # type: ignore[index]
        for name, child in node["sub"].items():  # type: ignore[union-attr]
            _sort_node(child, (*prefix, name))  # type: ignore[arg-type]

    _sort_node(root, ())
    return root


def _readme_doc_order(docs_root: Path) -> dict[str, int]:
    """``{rel_path: index}`` of ``docs/...`` hrefs in ``readme.md`` appearance order.

    Covers file hrefs (``docs/a/b.md``) and folder hrefs (``docs/a/`` → ``a``);
    ``{#anchor}`` fragments stripped, images (``![...]``) excluded.  Readme is
    order-only — treeview content still comes from the filesystem walk.
    """
    try:
        text = (docs_root.parent / "readme.md").read_text(encoding="utf-8")
    except OSError:
        return {}
    order: dict[str, int] = {}
    for m in re.finditer(r"(?<!\!)\[[^\]]*\]\(docs/([^)]+)\)", text):
        rel = m.group(1).split("#", 1)[0].strip().rstrip("/")
        if rel and rel not in order:
            order[rel] = len(order)
    return order


def _folder_label(folder: str) -> str:
    """Humanized node label: ``tcm_cli`` → ``Tcm cli`` (doc-stem fallback too)."""
    return folder.replace("_", " ").capitalize()


def _wrap_px(text: str, measure: Callable[[str], int], max_px: int) -> list[str]:
    """Greedy word-wrap of *text* to *max_px* px (``measure`` = font.measure)."""
    lines, cur = [], ""
    for word in text.split():
        if cur and measure(f"{cur} {word}") > max_px:
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    lines.append(cur)  # flush the remainder
    return lines or [""]


def _extract_title(path: Path) -> str | None:
    """Extract the first ``# `` heading from a markdown file."""
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if m := _RE_HEADING.match(line):
                return m.group(1)
    except OSError:
        pass
    return None


def _click_target(tree: ttk.Treeview, x: int, y: int, iid_path: dict[str, str]) -> str | None:
    """Resolve a ``<Button-1>`` click to a doc path, or ``None`` when no doc opens.

    The disclosure control (``Treeitem.indicator``) keeps its native
    expand/collapse behavior — clicks there never open a document.  Clicks
    elsewhere on a row whose iid maps to a path (leaf or ``_index.md``-linked
    parent, wrap segments included) resolve to that path.  No ``"break"`` is
    returned: native row selection still applies.
    """
    item = tree.identify_row(y)
    if not item:
        return None
    try:
        element = tree.identify_element(x, y)
    except tk.TclError:
        element = ""
    if element == "Treeitem.indicator":
        return None
    return iid_path.get(item)


def _hover_path(tree: ttk.Treeview, x: int, y: int, iid_path: dict[str, str]) -> str:
    """Resolve a ``<Motion>`` position to the hovered doc path (``""`` when none).

    Mirrors :func:`_click_target`: the disclosure control (``Treeitem.indicator``)
    shows the normal cursor and no link — only row text/cell areas arm ``hand2``.
    """
    try:
        element = tree.identify_element(x, y)
    except tk.TclError:
        element = ""
    if element == "Treeitem.indicator":
        return ""
    return iid_path.get(tree.identify_row(y), "")


# ── About dialog ────────────────────────────────────────────────────────────


class AboutDialog(tk.Toplevel):
    """Modal About window: metadata header + hierarchical docs list."""

    def __init__(
        self,
        parent: tk.Misc,
        *,
        full_mode: bool,
        ui: UIScale,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.withdraw()  # build off-screen → no corner flash before centering
        self.transient(parent)
        self.resizable(True, True)
        self.minsize(440, 320)

        self._ui = ui
        self._meta = version_meta()
        self._docs = discover_docs()
        self._docs_hierarchy = _docs_tree(self._docs)
        self._fit_width = 0  # last width the labels were fitted to
        self._on_status = on_status or (lambda _msg: None)  # main status bar
        self._hover_msg = ""  # dedup across motion storms
        self._tree_hover_link = False  # hand2 while the pointer is over a doc leaf
        self._iid_path: dict[str, str] = {}  # row iid (leaf + linked parent segs) → path str

        # System title carries the two dynamic runtime statuses
        mode = _S["about.mode_full"] if full_mode else _S["about.mode_simple"]
        h5 = _S["about.h5_available"] if H5_AVAILABLE else _S["about.h5_missing"]
        self.title(_S["about.title"].format(name=self._meta.get("name", "TCM"), mode=mode, h5=h5))

        bg = theme.FRAME_BG_FALLBACK
        self.configure(bg=bg)
        _opt_into_dark_titlebar(self)  # titlebar follows the OS theme too

        self._build(bg)
        self._show_overview()

        # Close on Escape; destroyed → clear our text from the status bar
        self.bind("<Escape>", lambda _e: self.destroy())
        self.bind("<Destroy>", self._on_destroy)
        # Ctrl+C → RTF/HTML copy (same as App._log)
        self.bind("<<Copy>>", self._on_copy_rich)
        # Reflow text heights on window resize (wrap width changes)
        self.bind("<Configure>", self._on_resize, add="+")

        # Center on the SCREEN work area (taskbar excluded; _refit re-centers
        # whenever the fitted height changes); show only when fully laid out.
        self.update_idletasks()
        left, top, right, bottom = work_area(self)
        x = left + max(right - left - _W, 0) // 2
        y = top + max(bottom - top - _H, 0) // 2
        self.geometry(f"{_W}x{_H}+{x}+{y}")
        self.deiconify()
        self.update()  # pump <Configure> → text relayout at final width
        self._refit()  # exact pixel fit now that the window is mapped
        self.update()
        self.grab_set()
        self.focus_set()

    def _build(self, bg: str) -> None:
        """Build the dialog layout: logo row (centered title+description) + meta + tree.

        Title and description share the row with the logo (shrunken to its left),
        their text centered inside the widget; the metadata list below spans the
        full window.  autoheight=False: _fit_height uses place which conflicts
        with pack; wrap="word": long text wraps instead of overflowing.
        """
        self._head = tk.Frame(self, background=bg)
        self._head.pack(fill="x", padx=16, pady=(16, 2))
        self._logo = None  # kept for tests; absent → text full-width
        if _LOGO_FILE.is_file():
            self._logo_img = tk.PhotoImage(file=_LOGO_FILE)
            self._logo = tk.Label(
                self._head, image=self._logo_img, background=bg, highlightthickness=0, borderwidth=0
            )
            self._logo.pack(side="right", anchor="n", padx=(12, 0))
        self._header = MarkdownLabel(
            self._head,
            font=self._ui.font(size_diff=1),
            background=bg,
            foreground=theme.FG_DEFAULT,
            colors=theme.TAG_COLORS,
            autoheight=False,
            wrap="word",
            on_link=open_md_link,
        )
        self._header.pack(fill="both", expand=True)
        # Title + description text centered inside the shrunken widget
        self._header.tag_configure("heading", justify="center")
        self._header.tag_configure("normal", justify="center")

        # Metadata list — full width below the logo row (never squeezed)
        self._meta_lbl = MarkdownLabel(
            self,
            font=self._ui.font(size_diff=1),
            background=bg,
            foreground=theme.FG_DEFAULT,
            colors=theme.TAG_COLORS,
            autoheight=False,
            wrap="word",
            on_link=open_md_link,
        )
        self._meta_lbl.pack(fill="x", padx=16, pady=(0, 2))

        # Docs list: hierarchical treeview — directory-nested parents (only the
        # first level expanded), titles as leaves.  Row font = ⅔ of the theme Treeview
        # font, link-blue leaves; rowheight synced to the font's linespace
        # so descenders ("g", "p") never clip against the next row.
        style = ttk.Style(self)
        base = style.lookup("Treeview", "font")
        self._docs_font = tkfont.Font(font=base if isinstance(base, str) else "TkDefaultFont")
        self._docs_font.configure(size=int(self._docs_font.actual("size") / 1.5))
        # Themed like the main window: field bg, theme fg (folders are NOT
        # the gray default), selected row = a tint of the link blue.
        style.configure(
            "Docs.Treeview",
            font=self._docs_font,
            rowheight=self._docs_font.metrics("linespace") + 2,
            background=theme.ENTRY_BG_FALLBACK,
            fieldbackground=theme.ENTRY_BG_FALLBACK,
            foreground=theme.FG_DEFAULT,
            borderwidth=1,
        )
        style.map(
            "Docs.Treeview",
            background=[("selected", mix_hex(theme.ENTRY_BG_FALLBACK, theme.NODE_DEFAULT_VALS_FG, 0.3))],
            foreground=[("selected", theme.FG_DEFAULT)],
        )
        self._vbar_shown = False
        self._docs_view = ttk.Treeview(
            self, style="Docs.Treeview", show="tree", selectmode="browse", height=1
        )
        self._docs_view.tag_configure("doc", foreground=theme.LINK_FG)
        # Vbar lives in the tree's right padding strip (place, not pack):
        # the tree box itself never shifts when the bar auto-appears.
        self._docs_vbar = ttk.Scrollbar(
            self, orient="vertical", command=self._docs_view.yview, style="App.Vertical.TScrollbar"
        )
        self._docs_view.configure(yscrollcommand=self._on_tree_yview)
        self._docs_view.pack(fill="both", padx=16, pady=(4, 12))
        # No button row: Escape (bound in __init__) closes the dialog

    def _show_overview(self) -> None:
        """Render the overview: metadata list, URLs, doc tree.

        Every metadata field is its own markdown block (list item) —
        ``parse_markdown`` merges consecutive lines into one paragraph.
        Mode/HDF5 statuses live in the system title (set in ``__init__``).
        """
        meta = self._meta
        version = meta.get("version", "dev")
        if product := meta.get("product"):
            version = f"{version} ({product})"

        # Title + description share the logo row (centered within the shrunken width)
        lines = [f"# {meta.get('name', 'TCM')}", ""]
        # Translated meta value takes precedence over build meta (the JSON
        # stays EN — source of truth for the exe version info)
        if desc := _S.get("about.meta.description") or meta.get("description"):
            lines += [desc, ""]
        self._header.set_text("\n".join(lines))

        items = []  # metadata list — full width below the logo row
        items += [f"**{_S['about.version']}:** `{version}`"]
        if meta.get("product_name"):
            items.append(f"**{_S['about.product']}:** {meta['product_name']}")
        if company := _S.get("about.meta.company") or meta.get("company_name"):
            items.append(f"**{_S['about.company']}:** {company}")
        if cp := meta.get("legal_copyright"):
            # © sign is self-labeling → bare value, no "Copyright:" prefix
            items.append(cp if "©" in cp else f"**{_S['about.copyright']}:** {cp}")
        if repo := meta.get("repo_url"):
            items.append(f"**{_S['about.repository']}:** [{repo}]({repo})")
        if meta.get("docs_url"):
            # "Документация (интернет / локальная)" — each half is its own link:
            # internet → site URL; local → bundled readme (a real file path, so
            # open_md_link routes it through the documentation browser).
            items.append(
                f"**{_S['about.documentation']}** "
                f"([{_S['about.docs_internet']}]({meta['docs_url']}) / "
                f"[{_S['about.docs_local']}]({local_readme().as_posix()}))"
            )
        self._meta_lbl.set_text("\n".join(f"- {it}" for it in items))
        # Hover → main status bar shows the URL under the pointer
        self._meta_lbl.bind("<Motion>", self._on_header_motion, add="+")
        self._meta_lbl.bind("<Leave>", lambda _e: self._hover_status(""), add="+")

        # Hierarchical doc tree: directory-nested parents (a folder holding
        # _index.md links to it, no separate leaf), doc titles as leaves;
        # iid = str(path) so a click resolves straight to the file.  Unwrapped
        # until _refit knows the mapped width.
        self._populate()
        self._bind_doc_clicks(self._docs_view)
        # No fit here: pre-show width is ~1px (wrap=word would count hundreds
        # of display lines and request a huge height that squeezes pack).
        # __init__ calls _refit() once the window is mapped at final size.

    def _refit(self) -> None:
        """Fit header + doc tree, sizing the window to their content FIRST.

        Pack squeezes children whose total px request exceeds the window
        height — and squeezed text is unmeasurable (``dlineinfo`` returns
        ``None`` below the allocation).  Resizing the window to the measured
        content before fitting makes every fit run un-squeezed.  Content
        taller than the screen WORK AREA caps at it; the window stays
        vertically centered on the work area (capped → pinned to its top),
        and the squeezed tree scrolls via the auto-appearing vbar.
        """
        tree = self._docs_view
        tree.configure(height=1)  # collapse for header measurement
        # Title+description share the logo row (logo height wins if taller); the
        # full-width metadata list below is sized independently.
        hdr_px = self._content_px(self._header)
        if self._logo is not None:
            hdr_px = max(hdr_px, self._logo.winfo_reqheight())
        hdr_px += self._content_px(self._meta_lbl)

        # row px calibrated from two row counts (DPI-safe), × display lines
        tree.configure(height=2)
        tree.update()
        px2 = tree.winfo_height()
        tree.configure(height=1)
        tree.update()
        h1 = tree.winfo_height()
        row_px = px2 - h1 or h1  # squeezed fallback: 1-row height, borders incl.
        self._populate(tree.winfo_width())  # re-wrap titles to the real width
        lines = sum(1 + len(tree.get_children(f)) for f in tree.get_children(""))
        docs_px = h1 + max(lines - 1, 0) * row_px

        # chrome = pads only (no separator, no window spare → no feedback)
        left, top, right, bottom = work_area(self)
        need = min(hdr_px + docs_px + sum(_PADS), bottom - top)
        x, _y = self._pos()
        x = max(left, min(x, right - self.winfo_width()))  # stay on the monitor
        y = top + max(bottom - top - need, 0) // 2  # vertically centered
        # ±2: no px-jitter resizing; wm_geometry() is the position truth
        if abs(need - self.winfo_height()) > 2 or abs(y - _y) > 2:
            self.geometry(f"{self.winfo_width()}x{need}+{x}+{y}")
            self.update()

        # all rows visible when they fit; a capped window squeezes the tree →
        # yscrollcommand fractions auto-show the vbar (see _on_tree_yview)
        tree.configure(height=max(lines, 1))
        tree.yview_moveto(0)
        tree.update()

        self._fit_label_height(self._header, bottom=bottom)
        self._fit_label_height(self._meta_lbl, bottom=bottom)
        if self._vbar_shown:  # header growth may have shifted geometry
            self._place_vbar()

    def _pos(self) -> tuple[int, int]:
        """Current frame ``(x, y)`` per ``wm_geometry()`` — the same coordinate
        space ``geometry()`` writes (winfo_x/y semantics differ per platform)."""
        x, y = re.fullmatch(r"\d+x\d+([+-]\d+)([+-]\d+)", self.wm_geometry()).groups()
        return int(x), int(y)

    @staticmethod
    def _last_display_line(label: MarkdownLabel) -> tuple[int, str]:
        """``(count, index)`` of the last display line (wrap-aware walk)."""
        n, idx = 0, "1.0"
        while True:
            n += 1
            nxt = label.index(f"{idx} +1 displayline")
            if label.compare(nxt, ">=", "end") or nxt == idx:
                break
            idx = nxt
        return n, idx

    def _content_px(self, label: MarkdownLabel) -> int:
        """Content height in px at the current wrap width; leaves 1 unit.

        Spacing tags make content px exceed the display-line count, so px is
        measured, never predicted from fonts: request generous heights until
        ``dlineinfo`` reports the LAST display line, take its bottom.  Growth
        that stops increasing ``winfo_height`` means pack-squeeze → bail at
        the allocation (window too small; fitting can't win against pack).
        """
        label.configure(height=1)
        label.update()
        if not label.get("1.0", "end-1c") or label.winfo_width() < 10:
            return label.winfo_height()

        _n, idx = self._last_display_line(label)
        px = 0
        for h in (_n + 8, _n + 16, _n + 24):  # slack for spacing-tag px
            label.configure(height=h)
            label.update()
            if dl := label.dlineinfo(idx):
                px = dl[1] + dl[3]  # last-line bottom = content px
                break
            if label.winfo_height() == px:  # request ineffective → squeezed
                break
            px = label.winfo_height()
        label.configure(height=1)
        label.update()
        return px

    def _on_resize(self, event: tk.Event) -> None:
        """Window resize → refit at the new width; vbar stays glued to the tree."""
        if event.widget is not self:
            return
        if (w := self.winfo_width()) != self._fit_width and w > 100:
            # w>100: early <Configure> at ~1px (pre-geometry) would fit at
            # near-zero wrap width — the explicit _refit() in __init__ covers
            # the real size once mapped.
            self._fit_width = w
            self._refit()
        elif self._vbar_shown:
            self._place_vbar()  # height-only resize: re-glue the bar

    def _fit_label_height(
        self, label: MarkdownLabel, *, min_lines: int = 1, bottom: int | None = None
    ) -> None:
        """Set ``height`` (text lines) so all content shows; grow iteratively.

        The label lives in a row whose height is ``max(children)`` — while its
        requested height stays below a taller sibling (the logo), its allocated
        box is squeezed and ``dlineinfo`` clamps to that box, so both the
        widget height AND the window must grow together until ``yview`` shows
        the last display line.  px per unit is DPI-dependent, never predicted.
        """
        label.update()
        if not label.get("1.0", "end-1c") or not label.winfo_ismapped() or label.winfo_width() < 10:
            label.configure(height=min_lines)
            return

        n, idx = self._last_display_line(label)
        h = n
        while h <= n + 8:  # slack for spacing-tag px
            label.configure(height=h)
            label.update()
            if label.dlineinfo(idx):
                break
            h += 1

        # keep growing window + widget until the last display line is visible
        while label.yview()[1] < 0.9999 and h <= n + 8:
            h += 1
            label.configure(height=h)
            # px per unit ≈ alloc/h (last line draws at h ⇒ not squeezed)
            unit = max(label.winfo_height() // max(h, 1), 1)
            if bottom is not None:  # never grow past the work area bottom
                unit = min(unit, bottom - self._pos()[1] - self.winfo_height())
            if unit <= 0:
                break
            x, y = self._pos()
            self.geometry(f"{self.winfo_width()}x{self.winfo_height() + unit}+{x}+{max(0, y - unit // 2)}")
            self.update()
        label.configure(height=max(min_lines, h))

    def _on_copy_rich(self, _event: tk.Event) -> str | None:
        """``<<Copy>>``: rich copy of the selected label, else plain combined
        overview (title+description+metadata)."""
        if sel := next((w for w in (self._header, self._meta_lbl) if w.tag_ranges("sel")), None):
            copy_rich(sel)
            return "break"
        text = "\n".join(w.get("1.0", "end-1c") for w in (self._header, self._meta_lbl)).strip()
        if text:
            self.clipboard_clear()
            self.clipboard_append(text)
            return "break"
        return None

    def _on_destroy(self, event: tk.Event) -> None:
        """Dialog closed → clear our hover text from the main status bar."""
        if event.widget is self:
            self._on_status("")

    def _hover_status(self, msg: str) -> None:
        """Publish a hover msg to the main status bar (deduped across motion)."""
        if msg != self._hover_msg:
            self._hover_msg = msg
            self._on_status(msg)

    def _on_header_motion(self, event: tk.Event) -> None:
        """Header hover → status bar shows the URL under the pointer."""
        self._hover_status(event.widget.link_at(event.x, event.y) or "")

    def _on_tree_motion(self, event: tk.Event) -> None:
        """Row-text hover → hand2 + file path in status bar; indicator/plain rows: neither."""
        path = _hover_path(self._docs_view, event.x, event.y, self._iid_path)
        self._hover_status(path)
        if (link := bool(path)) != self._tree_hover_link:  # dedup cursor churn
            self._tree_hover_link = link
            self._docs_view.configure(cursor="hand2" if link else "")

    def _on_tree_leave(self, _event: tk.Event) -> None:
        """Pointer left the tree → clear status text and the link cursor."""
        self._tree_hover_link = False
        self._docs_view.configure(cursor="")
        self._hover_status("")

    def _on_tree_yview(self, first: str, last: str) -> None:
        """yscrollcommand: forward fractions + auto-show the vbar only while
        the tree can scroll (Tk's own first/last pattern for Treeview)."""
        self._docs_vbar.set(first, last)
        if (scrollable := float(first) > 0.0 or float(last) < 1.0) != self._vbar_shown:
            self._vbar_shown = scrollable
            if scrollable:
                self._place_vbar()
            else:
                self._docs_vbar.place_forget()

    def _place_vbar(self) -> None:
        """Overlay the vbar in the tree's right pad strip — tree box stays put."""
        t = self._docs_view
        self._docs_vbar.place(
            x=t.winfo_x() + t.winfo_width() + 2, y=t.winfo_y(), height=t.winfo_height(), width=14
        )

    def _populate(self, width: int = 0) -> None:
        """(Re)build the doc tree, word-wrapping titles to *width* (0 → as-is).

        Directory-nested: each folder is a parent row; a folder holding
        ``_index.md`` links that row to the index (no separate ``_index.md``
        leaf).  Tk 8.6 items have no per-row ``-height`` → each extra wrapped
        line becomes its own continuation item; ``_iid_path`` maps every row
        segment (leaf + linked parent) to its file path.  Heading-less docs
        fall back to the underscore stem → humanized (real heading titles keep
        their casing).  Only the first level is expanded; deeper folders start
        collapsed.
        """
        tree, measure = self._docs_view, self._docs_font.measure
        tree.delete(*tree.get_children(""))
        self._iid_path.clear()

        def _insert_leaf(parent: str, title: str, path: Path, depth: int) -> None:
            text = title if parent or title != path.stem else _folder_label(title)
            avail = width - (_ICON_PX + depth * _INDENT_PX) if width else 0
            segs = _wrap_px(text, measure, avail) if avail and avail > 0 else [text]
            iid = str(path)
            tree.insert(parent, "end", iid=iid, text=segs[0], tags=("doc",))
            self._iid_path[iid] = iid
            for n, seg in enumerate(segs[1:], 2):
                cont = f"{iid}#{n}"
                tree.insert(parent, "end", iid=cont, text=seg, tags=("doc",))
                self._iid_path[cont] = iid

        def _insert_node(node: dict[str, object], parent: str, depth: int) -> None:
            link_order = _readme_doc_order(DOC_DIR)
            inf = len(link_order)

            def _rel(p: Path) -> str:
                try:
                    return p.relative_to(DOC_DIR).as_posix()
                except ValueError:
                    return p.name

            def _child_min(child: dict[str, object]) -> int:
                best = inf
                if child.get("index") is not None:
                    _, ip = child["index"]  # type: ignore[misc]
                    best = min(best, link_order.get(_rel(ip), inf))
                for _, fp in child.get("files", []):  # type: ignore[union-attr]
                    best = min(best, link_order.get(_rel(fp), inf))
                for grand in child.get("sub", {}).values():  # type: ignore[union-attr]
                    best = min(best, _child_min(grand))  # type: ignore[arg-type]
                return best

            items: list[tuple[tuple[int, str], str, object]] = []
            for title, path in node.get("files", []):  # type: ignore[union-attr]
                items.append(((link_order.get(_rel(path), inf), title.lower()), "file", (title, path)))  # type: ignore[arg-type]
            for dirname, child in node.get("sub", {}).items():  # type: ignore[union-attr]
                items.append(
                    (
                        (min(_child_min(child), link_order.get(dirname, inf)), dirname.lower()),
                        "dir",
                        (dirname, child),
                    )
                )  # type: ignore[arg-type]
            items.sort(key=lambda t: t[0])
            for _, kind, data in items:
                if kind == "file":
                    title, path = data  # type: ignore[misc]
                    _insert_leaf(parent, title, path, depth)
                else:
                    dirname, child = data  # type: ignore[misc]
                    index = child.get("index")  # type: ignore[union-attr]
                    if index is not None:
                        ititle, ipath = index  # type: ignore[misc]
                        avail = width - (_ICON_PX + depth * _INDENT_PX) if width else 0
                        segs = _wrap_px(ititle, measure, avail) if avail and avail > 0 else [ititle]
                        iid = str(ipath)
                        tree.insert(parent, "end", iid=iid, text=segs[0], tags=("doc",), open=False)
                        self._iid_path[iid] = iid
                        for n, seg in enumerate(segs[1:], 2):
                            cont = f"{iid}#{n}"
                            tree.insert(parent, "end", iid=cont, text=seg, tags=("doc",))
                            self._iid_path[cont] = iid
                        _insert_node(child, iid, depth + 1)  # type: ignore[arg-type]
                    else:
                        iid = tree.insert(parent, "end", text=_folder_label(dirname), open=False)
                        _insert_node(child, iid, depth + 1)  # type: ignore[arg-type]

        root = self._docs_hierarchy
        _insert_node(root, "", 0)  # type: ignore[arg-type]
        for top in tree.get_children(""):
            tree.item(top, open=True)

    def _bind_doc_clicks(self, tree: ttk.Treeview) -> None:
        """Text click on a leaf or linked parent → open its doc; indicator toggles expand."""

        def _open(event: tk.Event) -> None:
            if path := _click_target(tree, event.x, event.y, self._iid_path):
                open_md_link(path)  # wrap segments resolve too

        tree.bind("<Button-1>", _open, add="+")
        tree.bind("<Motion>", self._on_tree_motion, add="+")
        tree.bind("<Leave>", self._on_tree_leave, add="+")
