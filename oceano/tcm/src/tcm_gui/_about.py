"""About dialog: version, runtime info, documentation browser.

Opened by the ``?`` button in the search path row.  Displays:
- Product name + version (from ``version_meta.json``)
- Repository + documentation URLs (clickable → browser)
- Discovered ``*.md`` docs from the bundled ``docs/`` directory

Runtime status (full/simple mode, HDF5 availability) lives in the
window system title, not the body.  All chrome labels AND the translatable
meta values (description, company) are i18n via :data:`STRINGS`
(``about.*`` keys); doc discovery is filtered to the application language
(:func:`_lang_filter`).

Doc titles are extracted from the first ``# `` heading of each file.
Clicking a title opens :class:`DocViewer` — a full-size zoomed window
rendering the document in a scrollable :class:`MarkdownLabel`.

Markdown note: ``parse_markdown`` joins consecutive lines into ONE
paragraph, so every distinct metadata field must be its own block
(separated by blank lines, or a ``- `` list item).
"""

from __future__ import annotations

import logging
import os
import re
import sys
import tkinter as tk
from collections import defaultdict
from pathlib import Path
from tkinter import ttk

from tcm._constants import DOC_DIR, H5_AVAILABLE, version_meta

from ._i18n import STRINGS as _S
from ._i18n import resolve_lang
from ._rtf_clipboard import copy_rich
from .const import UIScale
from .md_label import MarkdownLabel
from .theme import FG_DEFAULT, FRAME_BG_FALLBACK, TAG_COLORS, THEME

_l = logging.getLogger(__name__)

_RE_HEADING = re.compile(r"^#\s+(.+?)\s*$")

# Language suffix on doc stems: ``config_reference_ru`` → ("config_reference", "ru").
_RE_LANG_SUFFIX = re.compile(r"_(?P<lang>[a-z]{2})$", re.IGNORECASE)

# Docs subdirs to exclude from discovery (internal/TODO).
_EXCLUDE_DIRS = {"todo"}

# Base dialog size — first guess only: _refit measures the content and
# resizes the height to it (avoids a big map→refit jump; width stays fixed).
_W, _H = 560, 720

# Vertical padding px of the packed children (see _build) — the fixed chrome
# part of the window height, with NO spare (spare would feed back into the
# _refit size and accumulate on every reflow).
_PADS = (16, 4, 4, 4, 0, 12)


# ── doc discovery ───────────────────────────────────────────────────────────


def discover_docs(root: Path | None = None, lang: str | None = None) -> list[tuple[str, str, Path]]:
    """Walk ``docs/`` for ``*.md`` files; return ``(folder, title, path)`` sorted by path.

    Title = first ``# `` heading; fallback = filename stem.
    Folder = parent dir name relative to docs root (``""`` for top-level files).
    Excludes ``todo/`` and other internal dirs, then filters by *lang*
    (default: :func:`resolve_lang`) — see :func:`_lang_filter`.
    """
    docs_root = root or DOC_DIR.parent  # DOC_DIR = docs/tcm_cli → parent = docs
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


def _docs_tree(docs: list[tuple[str, str, Path]]) -> dict[str, list[tuple[str, Path]]]:
    """Group flat ``(folder, title, path)`` docs by folder for hierarchical display."""
    tree: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for folder, title, path in docs:
        tree[folder].append((title, path))
    return dict(sorted(tree.items()))


def _extract_title(path: Path) -> str | None:
    """Extract the first ``# `` heading from a markdown file."""
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if m := _RE_HEADING.match(line):
                return m.group(1)
    except OSError:
        pass
    return None


# ── About dialog ────────────────────────────────────────────────────────────


class DocViewer(tk.Toplevel):
    """Full-size (zoomed) window rendering one markdown document scrollably."""

    def __init__(self, parent: tk.Misc, *, title: str, path: Path, ui: UIScale) -> None:
        super().__init__(parent)
        self.title(title)
        self.transient(parent)
        self.geometry("900x640")
        try:
            self.state("zoomed")  # maximize (Windows)
        except tk.TclError:
            pass

        bg = FRAME_BG_FALLBACK
        self.configure(bg=bg)

        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            _l.exception("Failed to load doc: %s", path)
            text = f"*Failed to load: {path.name}*"

        # wrap="word" + fill → long docs wrap at window width; scrollbar for overflow
        content = MarkdownLabel(
            self,
            font=ui.font(),
            background=bg,
            foreground=FG_DEFAULT,
            colors=TAG_COLORS,
            autoheight=False,
            wrap="word",
        )
        vbar = ttk.Scrollbar(self, orient="vertical", command=content.yview)
        content.configure(yscrollcommand=vbar.set)
        content.pack(side="left", fill="both", expand=True)
        vbar.pack(side="right", fill="y")
        content.set_text(text)

        # Ctrl+C → RTF/HTML copy (same as App._log); Escape closes
        self._content = content
        self.bind("<<Copy>>", lambda _e: (copy_rich(self._content), "break")[1])
        self.bind("<Escape>", lambda _e: self.destroy())


class AboutDialog(tk.Toplevel):
    """Modal About window: metadata header + hierarchical docs list."""

    def __init__(self, parent: tk.Misc, *, full_mode: bool, ui: UIScale) -> None:
        super().__init__(parent)
        self.withdraw()  # build off-screen → no corner flash before centering
        self.transient(parent)
        self.resizable(True, True)
        self.minsize(440, 320)

        self._ui = ui
        self._meta = version_meta()
        self._docs = discover_docs()
        self._docs_by_folder = _docs_tree(self._docs)
        self._fit_width = 0  # last width the labels were fitted to

        # System title carries the two dynamic runtime statuses
        mode = _S["about.mode_full"] if full_mode else _S["about.mode_simple"]
        h5 = _S["about.h5_available"] if H5_AVAILABLE else _S["about.h5_missing"]
        self.title(_S["about.title"].format(name=self._meta.get("name", "TCM"), mode=mode, h5=h5))

        bg = FRAME_BG_FALLBACK
        self.configure(bg=bg)

        self._build(bg)
        self._show_overview()

        # Close on Escape
        self.bind("<Escape>", lambda _e: self.destroy())
        # Ctrl+C → RTF/HTML copy (same as App._log)
        self.bind("<<Copy>>", self._on_copy_rich)
        # Reflow text heights on window resize (wrap width changes)
        self.bind("<Configure>", self._on_resize, add="+")

        # Center on parent; show only when fully laid out
        self.update_idletasks()
        pw, ph = parent.winfo_width(), parent.winfo_height()
        px, py = parent.winfo_rootx(), parent.winfo_rooty()
        x, y = px + (pw - _W) // 2, py + (ph - _H) // 2
        self.geometry(f"{_W}x{_H}+{x}+{y}")
        self.deiconify()
        self.update()  # pump <Configure> → text relayout at final width
        self._refit()  # exact pixel fit now that the window is mapped
        self.update()
        self.grab_set()
        self.focus_set()

    def _build(self, bg: str) -> None:
        """Build the dialog layout: info header + separator + docs tree."""
        # Info header: MarkdownLabel for product/runtime metadata
        # autoheight=False: _fit_height uses place which conflicts with pack
        # wrap="word": long text (description, company) wraps instead of overflowing
        self._header = MarkdownLabel(
            self,
            font=self._ui.font(size_diff=1),
            background=bg,
            foreground=FG_DEFAULT,
            colors=TAG_COLORS,
            autoheight=False,
            wrap="word",
        )
        self._header.pack(fill="x", padx=16, pady=(16, 4))

        # Separator between the metadata header and the docs list
        ttk.Separator(self, orient="horizontal").pack(fill="x", padx=16, pady=4)

        # Docs list: hierarchical tree at half the base font size
        self._docs_lbl = MarkdownLabel(
            self,
            font=self._ui.font(size_diff=-2),
            background=bg,
            foreground=FG_DEFAULT,
            colors=TAG_COLORS,
            autoheight=False,
            wrap="word",
        )
        self._docs_lbl.pack(fill="x", padx=16, pady=(0, 12))
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

        lines = [f"# {meta.get('name', 'TCM')}", ""]
        # Translated meta value takes precedence over build meta (the JSON
        # stays EN — source of truth for the exe version info)
        if desc := _S.get("about.meta.description") or meta.get("description"):
            lines += [desc, ""]

        # All distinct metadata as separate list items (labels i18n)
        items = [f"**{_S['about.version']}:** `{version}`"]
        if meta.get("product_name"):
            items.append(f"**{_S['about.product']}:** {meta['product_name']}")
        if company := _S.get("about.meta.company") or meta.get("company_name"):
            items.append(f"**{_S['about.company']}:** {company}")
        if cp := meta.get("legal_copyright"):
            # © sign is self-labeling → bare value, no "Copyright:" prefix
            items.append(cp if "©" in cp else f"**{_S['about.copyright']}:** {cp}")
        if repo := meta.get("repo_url"):
            items.append(f"**{_S['about.repository']}:** `{repo}`")
        if docs := meta.get("docs_url"):
            items.append(f"**{_S['about.documentation']}:** `{docs}`")
        lines += [f"- {it}" for it in items]

        self._header.set_text("\n".join(lines))
        self._bind_url_clicks(self._header, meta)

        # Hierarchical doc tree (smaller font): folder as bold paragraph,
        # each title its own column-0 list item (indented lines would fold
        # into the previous item per the parser).
        doc_lines = [f"**{_S['about.documentation']}:**", ""]
        for folder, entries in self._docs_by_folder.items():
            if folder:
                doc_lines += [f"**{folder}/**"]
            doc_lines += [f"- {title}" for title, _ in entries]
            doc_lines.append("")
        self._docs_lbl.set_text("\n".join(doc_lines) if self._docs else "")
        self._bind_doc_clicks(self._docs_lbl)
        # No fit here: pre-show width is ~1px (wrap=word would count hundreds
        # of display lines and request a huge height that squeezes pack).
        # __init__ calls _refit() once the window is mapped at final size.

    def _refit(self) -> None:
        """Fit both labels, sizing the window to their content FIRST.

        Pack squeezes children whose total px request exceeds the window
        height — and squeezed text is unmeasurable (``dlineinfo`` returns
        ``None`` below the allocation).  Resizing the window to the measured
        content before fitting makes every fit run un-squeezed.
        """
        # measure with the sibling collapsed (_content_px leaves 1 unit)
        hdr_px, docs_px = self._content_px(self._header), self._content_px(self._docs_lbl)
        # chrome = paddings + separator only — never window spare (feedback)
        sep_px = next(w.winfo_height() for w in self.winfo_children() if isinstance(w, ttk.Separator))
        need = min(hdr_px + docs_px + sep_px + sum(_PADS), self.winfo_screenheight() - 60)
        if abs(need - self.winfo_height()) > 2:  # ±2: no px-jitter resizing
            self.geometry(f"{self.winfo_width()}x{need}+{self.winfo_x()}+{self.winfo_y()}")
            self.update()
        self._fit_label_height(self._header)
        self._fit_label_height(self._docs_lbl)

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
        """Window width changed → refit label heights so wrapped text reflows."""
        if event.widget is not self:
            return
        if (w := self.winfo_width()) != self._fit_width and w > 100:
            # w>100: early <Configure> at ~1px (pre-geometry) would fit at
            # near-zero wrap width — the explicit _refit() in __init__ covers
            # the real size once mapped.
            self._fit_width = w
            self._refit()

    def _bind_url_clicks(self, label: MarkdownLabel, meta: dict) -> None:
        """Bind click events on URL text ranges to open in browser."""
        # Find URL positions in the Text widget and bind <Button-1>
        for key in ("repo_url", "docs_url"):
            url = meta.get(key)
            if not url:
                continue
            # Search for the URL text in the widget
            start = "1.0"
            while True:
                pos = label.search(url, start, tk.END)
                if not pos:
                    break
                end = f"{pos}+{len(url)}c"
                label.tag_add("link", pos, end)
                start = end
        # Configure link tag
        link_color = "#0066CC" if THEME == "light" else "#66AAFF"
        label.tag_configure("link", foreground=link_color, underline=True)
        label.tag_bind("link", "<Button-1>", lambda e, m=meta: self._open_url(e, m))
        label.tag_bind("link", "<Enter>", lambda _e: label.configure(cursor="hand2"))
        label.tag_bind("link", "<Leave>", lambda _e: label.configure(cursor=""))

    def _open_url(self, event: tk.Event, meta: dict) -> None:
        """Open the clicked URL: locate the link range under the click, match its text."""
        w = event.widget
        idx = w.index(f"@{event.x},{event.y}")
        urls = {u for u in (meta.get("repo_url"), meta.get("docs_url")) if u}
        # tag_ranges → alternating (start, end) pairs of every link range
        ranges = w.tag_ranges("link")
        for start, end in zip(ranges[0::2], ranges[1::2]):
            if w.compare(start, "<=", idx) and w.compare(idx, "<", end):
                if (url := w.get(start, end)) in urls:
                    self._launch_url(url)
                return

    def _launch_url(self, url: str) -> None:
        """Open URL in default browser (Windows: os.startfile)."""
        try:
            if sys.platform == "win32":
                os.startfile(url)
            else:
                import subprocess

                subprocess.Popen(["xdg-open", url])
        except OSError:
            _l.error("Failed to open URL: %s", url)

    def _show_doc(self, path: Path) -> None:
        """Open the document in a separate full-size :class:`DocViewer` window.

        Releases this dialog's modal grab while the viewer is open (grab
        would swallow all input to the viewer window), re-grabs on close.
        """
        title = next((t for _f, t, p in self._docs if p == path), path.stem)
        self.grab_release()
        viewer = DocViewer(self, title=title, path=path, ui=self._ui)
        viewer.bind("<Destroy>", lambda e: e.widget is viewer and self._regrab())

    def _regrab(self) -> None:
        """Restore modal grab after DocViewer closed (no-op if dialog gone)."""
        try:
            self.grab_set()
        except tk.TclError:
            pass

    def _fit_label_height(self, label: MarkdownLabel, *, min_lines: int = 1) -> None:
        """Set ``height`` (text lines) so the content shows without clipping.

        Grow from the wrap-aware display-line count until ``dlineinfo``
        reports the last display line — it returns ``None`` below the widget
        bottom, the clipping condition itself (spacing tags may cost a few
        extra units of slack).  px per unit is DPI-dependent, never predicted
        from fonts.  Headroom is assumed: :meth:`_refit` sized the window to
        the content first.  Unmapped / <10px wrap width → defer (min height).
        """
        label.update()
        if not label.get("1.0", "end-1c") or not label.winfo_ismapped() or label.winfo_width() < 10:
            label.configure(height=min_lines)
            return

        n, idx = self._last_display_line(label)

        # smallest height that draws the last line
        h = n
        while h <= n + 6:
            label.configure(height=h)
            label.update()
            if label.dlineinfo(idx):
                break
            h += 1

        # trailing spacing after the last line isn't a display line → yview
        # truth; the window is exactly sized, so room for +1 must be granted
        if label.yview()[1] < 0.9999:
            # px per unit ≈ alloc/h (last line draws at h ⇒ not squeezed)
            unit = max(label.winfo_height() // max(h, 1), 1)
            self.geometry(
                f"{self.winfo_width()}x{self.winfo_height() + unit}+{self.winfo_x()}+{self.winfo_y()}"
            )
            self.update()
            h += 1
        label.configure(height=max(min_lines, h))

    def _on_copy_rich(self, _event: tk.Event) -> str | None:
        """``<<Copy>>`` on dialog → RTF/HTML from whichever widget has a selection."""
        for w in (self._header, self._docs_lbl):
            if w.tag_ranges("sel"):
                copy_rich(w)
                return "break"
        # No selection: copy the largest non-empty widget
        for w in (self._header, self._docs_lbl):
            if w.get("1.0", "end-1c").strip():
                copy_rich(w)
                return "break"
        return None

    def _bind_doc_clicks(self, label: MarkdownLabel) -> None:
        """Bind click events on doc titles to show their content."""
        link_color = "#0066CC" if THEME == "light" else "#66AAFF"
        for i, (_folder, title, path) in enumerate(self._docs):
            # Find the title text in the widget and bind it
            start = "1.0"
            while True:
                pos = label.search(title, start, tk.END)
                if not pos:
                    break
                end = f"{pos}+{len(title)}c"
                label.tag_add(f"doc_{i}", pos, end)
                start = end
            # Configure and bind the tag
            label.tag_configure(f"doc_{i}", foreground=link_color, underline=True)
            label.tag_bind(f"doc_{i}", "<Button-1>", lambda _e, p=path: self._show_doc(p))
            label.tag_bind(f"doc_{i}", "<Enter>", lambda _e: label.configure(cursor="hand2"))
            label.tag_bind(f"doc_{i}", "<Leave>", lambda _e: label.configure(cursor=""))
