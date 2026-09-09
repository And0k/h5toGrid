"""Tests for bare-filesystem-path auto-linking.

Covers:
  * ``tcm_gui._allowed_paths`` — restricted path detection (allowed-dir prefix,
    2–4 letter extensions, space termination, unrestricted fallback)
  * ``tcm_gui.md_label._path_spans`` — span expansion (file-name display,
    ``file:`` URI tags, explicit links untouched)
  * ``tcm_gui.browser.browser.open_md_link`` — ``file:`` URI / external scheme
    routing to ``open_os_target``, local paths to the doc browser
"""

from __future__ import annotations

import logging
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import pytest

from tcm.paths import link_root
from tcm_gui import _allowed_paths
from tcm_gui import log_bridge as log_bridge_mod
from tcm_gui._allowed_paths import extract_allowed_paths, path_kind
from tcm_gui.app import App
from tcm_gui.browser import browser as brows
from tcm_gui.log_bridge import LogText, drain
from tcm_gui.md_label import MarkdownLabel, _path_spans, bind_link_hover, make_link_hover_handler

ROOT = r"C:\work\data"


def _uri(p: str) -> str:
    return Path(p).as_uri()


# ── extract_allowed_paths — restricted (allowed-dir prefix) ──────────────────


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (rf"{ROOT}\run1.nc", (rf"{ROOT}\run1.nc",)),
        (rf"{ROOT}\a\b\run1.nc", (rf"{ROOT}\a\b\run1.nc",)),  # nested dirs
        (rf"{ROOT}\run1.h5", (rf"{ROOT}\run1.h5",)),  # 2-letter ext
        (f"out: {ROOT}/run1.nc done", (f"{ROOT}/run1.nc",)),  # forward separators
        (rf"see {ROOT}\run1.nc.", (rf"{ROOT}\run1.nc",)),  # trailing prose dot not consumed
        (rf"{ROOT}\archive.tar.gz", (rf"{ROOT}\archive.tar.gz",)),  # multi-dot tail
        (rf"list: {ROOT}\a.nc and {ROOT}\b.h5", (rf"{ROOT}\a.nc", rf"{ROOT}\b.h5")),
        (r"\\srv\share\f.nc", ()),  # outside ROOT
        (rf"{ROOT}X\f.nc", ()),  # prefix boundary: dataX ≠ data
        (rf"{ROOT}\f.c", ()),  # 1-letter extension
        (rf"{ROOT}\f.astro", ()),  # 5-letter extension
        (rf"{ROOT}\f.htmlx", ()),  # extension truncation guard (not a prefix of a longer word)
        (rf"{ROOT}\noext", ()),  # no extension → not a linkable file
        (rf"{ROOT}\My Dir\f.nc", ()),  # space terminates the tail
        ("just text", ()),
        ("", ()),
    ],
)
def test_extract_restricted(text: str, expected: tuple[str, ...]):
    assert extract_allowed_paths(text, ROOT) == expected


def test_spaced_allowed_dir():
    """Allowed dir is matched literally — its spaces are fine."""
    assert extract_allowed_paths(r"C:\My Data\f.nc", r"C:\My Data") == (r"C:\My Data\f.nc",)
    assert extract_allowed_paths(r"C:\My Data2\f.nc", r"C:\My Data") == ()


# ── extract_allowed_paths — unrestricted fallback (no allowed dir) ───────────


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (r"C:\data\f.nc", (r"C:\data\f.nc",)),  # drive root
        ("/home/u/f.nc", ("/home/u/f.nc",)),  # POSIX root
        (r"\\srv\share\f.nc", (r"\\srv\share\f.nc",)),  # UNC
        (r"\x\f.nc", (r"\x\f.nc",)),  # root-relative Windows
        ("relative/f.nc", ()),
        ("https://x/f.nc", ()),  # URL — guard blocks `s:/` and `//`
        ("mail me@x/f.nc", ()),
        ("just text", ()),
    ],
)
def test_extract_unrestricted(text: str, expected: tuple[str, ...]):
    assert extract_allowed_paths(text, "") == expected


# ── _path_spans (MarkdownLabel expansion) ────────────────────────────────────


class TestPathSpans:
    def test_plain_text_split(self):
        p = rf"{ROOT}\run1.nc"
        assert _path_spans(f"done: {p} ok", frozenset(), ROOT) == (
            ("done: ", frozenset()),
            ("run1.nc", frozenset({_uri(p)})),
            (" ok", frozenset()),
        )

    def test_display_is_file_name(self):
        p = rf"{ROOT}\deep\nested\x.nc"
        spans = _path_spans(p, frozenset(), ROOT)
        assert spans == (("x.nc", frozenset({_uri(p)})),)

    def test_explicit_link_untouched(self):
        span = (r"C:\x\f.nc", frozenset({"http://a/b"}))
        assert _path_spans(*span, ROOT) == (span,)

    def test_code_style_kept(self):
        """A path inside a code span keeps the style tag and gains the URI."""
        p = rf"{ROOT}\f.nc"
        assert _path_spans(p, frozenset({"code"}), ROOT) == (("f.nc", frozenset({"code", _uri(p)})),)

    def test_style_kept(self):
        p = rf"{ROOT}\f.nc"
        assert _path_spans(p, frozenset({"bold"}), ROOT) == (("f.nc", frozenset({"bold", _uri(p)})),)

    def test_tab_status_full_path_links(self):
        """``tab.status`` (raw) with a full config path: bare path → file-name link."""
        p = rf"{ROOT}\dev\_raw\cfg_proc\run\140228_1300@i03.yaml"
        assert _path_spans(f"Configuration: {p}", frozenset(), ROOT) == (
            ("Configuration: ", frozenset()),
            ("140228_1300@i03.yaml", frozenset({_uri(p)})),
        )

    def test_backtick_wrapped_path_not_linkable(self):
        """Backticks around a bare path break auto-linking — the trailing backtick
        joins the last node and ``path_kind`` rejects the odd suffix (``yaml` ``).
        ``tab.status`` must stay backtick-free so the path auto-links to its file name."""
        p = rf"{ROOT}\cfg_proc\run\a.yaml"
        assert _path_spans(f"Configuration: `{p}`", frozenset(), ROOT) == (
            (f"Configuration: `{p}`", frozenset()),
        )

    def test_no_root_matches_drive(self):
        p = r"C:\anywhere\f.nc"
        assert _path_spans(f"see {p}!", frozenset(), "") == (
            ("see ", frozenset()),
            ("f.nc", frozenset({_uri(p)})),
            ("!", frozenset()),
        )

    def test_driveless_fallback_uri(self):
        r"""``\x\f.nc`` from the unrestricted branch still yields a file: URI."""
        ((name, tags),) = _path_spans(r"\x\f.nc", frozenset(), "")
        assert name == "f.nc"
        (uri,) = tags
        assert uri.startswith("file:")


# ── open_md_link routing ─────────────────────────────────────────────────────


class _FakeBrowser:
    def __init__(self):
        self.calls: list[tuple[Path, str | None]] = []

    def open(self, path, anchor=None):
        self.calls.append((path, anchor))


class TestOpenMdLink:
    def test_file_uri_to_os_app(self, monkeypatch):
        calls = []
        monkeypatch.setattr(brows, "open_os_target", calls.append)
        brows.open_md_link("file:///C:/work/f%20x/f.nc")
        assert calls == [r"C:\work\f x\f.nc"]

    def test_file_uri_unc_to_os_app(self, monkeypatch):
        calls = []
        monkeypatch.setattr(brows, "open_os_target", calls.append)
        brows.open_md_link(Path(r"\\srv\share\f.nc").as_uri())
        assert calls == [r"\\srv\share\f.nc"]

    def test_external_scheme_to_os_app(self, monkeypatch):
        calls = []
        monkeypatch.setattr(brows, "open_os_target", calls.append)
        brows.open_md_link("https://example.com/x?q=1")
        assert calls == ["https://example.com/x?q=1"]

    def test_local_path_to_doc_browser(self, monkeypatch):
        fake = _FakeBrowser()
        monkeypatch.setattr(brows, "get_documentation_browser", lambda: fake)
        brows.open_md_link("readme.md")
        assert fake.calls == [(Path("readme.md"), None)]

    def test_anchor_resolved_against_base(self, monkeypatch):
        fake = _FakeBrowser()
        monkeypatch.setattr(brows, "get_documentation_browser", lambda: fake)
        brows.open_md_link("doc.md#heading", base="C:/docs/x.md")
        assert fake.calls == [(Path("C:/docs/x.md").parent / "doc.md", "heading")]


# ── path_kind: file vs dir vs plain ──────────────────────────────────────────


class TestPathKind:
    def test_file_suffixes(self):
        assert path_kind(r"C:\x\f.nc") == "file"
        assert path_kind(r"C:\x\f.h5") == "file"
        assert path_kind(r"C:\x\f.yaml") == "file"
        assert path_kind(r"C:\x\f.tar.gz") == "file"

    def test_odd_suffix_stays_plain(self):
        assert path_kind(r"C:\x\f.c") == ""
        assert path_kind(r"C:\x\f.astro") == ""
        assert path_kind(r"C:\x\f.htmlx") == ""

    def test_dir_requires_existence(self, tmp_path):
        (tmp_path / "sub").mkdir()
        assert path_kind(str(tmp_path / "sub")) == "dir"
        assert path_kind(str(tmp_path / "absent" / "deep")) == ""

    def test_extension_node_is_file_even_if_dir(self, tmp_path):
        """`_raw.zip`-style archive dirs carry an extension → treated as files."""
        (tmp_path / "_raw.zip").mkdir()
        assert path_kind(str(tmp_path / "_raw.zip")) == "file"

    def test_dot_and_dotdot_not_linkable(self):
        """Bare ``.`` / ``..`` must not link to the project root (status-bar bug)."""
        assert path_kind(".") == ""
        assert path_kind("..") == ""


# ── restricted detection now covers directories ──────────────────────────────


class TestRestrictedDirs:
    def test_dirs_and_files_under_root(self, tmp_path):
        root = tmp_path / "data"
        sub = root / "sub"
        sub.mkdir(parents=True)
        (root / "f.nc").touch()
        text = f"in {sub} from {root}\\f.nc"
        assert extract_allowed_paths(text, str(root)) == (str(sub), str(root / "f.nc"))

    def test_root_itself_matches(self, tmp_path):
        """The allowed root itself is now linkable (device dir must link)."""
        assert extract_allowed_paths(str(tmp_path), str(tmp_path)) == (str(tmp_path),)

    def test_spans_dir_full_file_shrunk(self, tmp_path):
        (tmp_path / "sub").mkdir()
        root = str(tmp_path)
        sub = f"{root}\\sub"
        assert _path_spans(f"in {sub} ok", frozenset(), root) == (
            ("in ", frozenset()),
            (sub, frozenset({_allowed_paths.to_uri(sub)})),
            (" ok", frozenset()),
        )


# ── LogText: linked inserts in the log widget ────────────────────────────────


class TestLogText:
    @pytest.fixture
    def log(self, _session_tk_root):
        if _session_tk_root is None:
            pytest.skip("Tk unavailable")
        w = LogText(_session_tk_root)
        yield w
        w.destroy()

    def test_file_shrunk_to_name(self, log, tmp_path):
        root = str(tmp_path)
        log.insert_linked(f"see {root}\\f.nc ok", (), root)
        assert log.get("1.0", "end-1c") == "see f.nc ok"
        (a, _b) = log.tag_ranges("loglink")
        assert log.link_url_at(f"{a}+1c") == _allowed_paths.to_uri(f"{root}\\f.nc")
        assert log.link_url_at("1.2") is None

    def test_dir_displayed_in_full(self, log, tmp_path):
        (tmp_path / "sub").mkdir()
        root = str(tmp_path)
        sub = f"{root}\\sub"
        log.insert_linked(f"in {sub} dir", (), root)
        assert log.get("1.0", "end-1c") == f"in {sub} dir"
        (a, _b) = log.tag_ranges("loglink")
        assert log.link_url_at(f"{a}+1c") == _allowed_paths.to_uri(sub)

    def test_nonexistent_dir_stays_plain(self, log, tmp_path):
        root = str(tmp_path)
        text = f"in {root}\\absent\\deep end"
        log.insert_linked(text, (), root)
        assert log.get("1.0", "end-1c") == text  # untouched — no shrinking, no link
        assert not log.tag_ranges("loglink")

    def test_open_link_routes_to_open_md_link(self, log, monkeypatch):
        calls = []
        monkeypatch.setattr(log_bridge_mod, "open_md_link", calls.append)
        monkeypatch.setattr(log, "link_url_at", lambda _index: "file:///C:/x/f.nc")
        log._open_link(SimpleNamespace(x=1, y=1))
        assert calls == ["file:///C:/x/f.nc"]


# ── drain: duck-typed insert_linked with fallback ────────────────────────────


class _DummyLog:
    def __init__(self):
        self.chunks: list[tuple[str, object]] = []

    def insert(self, _index, text, tags=None):
        self.chunks.append((text, tags))


class TestDrain:
    @staticmethod
    def _record(msg: str) -> logging.LogRecord:
        return logging.LogRecord("n", logging.INFO, "p", 1, msg, (), None)

    def test_plain_widget_fallback(self):
        q: Queue = Queue()
        q.put(self._record("hello"))
        dummy = _DummyLog()
        assert drain(q, dummy) == 1
        assert any("hello" in text for text, _tags in dummy.chunks)

    def test_linked_widget_receives_root(self):
        q: Queue = Queue()
        q.put(self._record("hello"))
        seen: list[tuple[str, object, str]] = []
        w = _DummyLog()
        w.insert_linked = lambda text, tags, root: seen.append((text, tags, root))  # type: ignore[attr-defined]
        assert drain(q, w, "C:/root") == 1
        assert seen and seen[0] == ("hello\n", "info", "C:/root")


# ── restricted detection: root itself matches (device dir links) ────────────


class TestRestrictedRootSelfMatch:
    def test_root_itself_matches_when_directory(self, tmp_path):
        """The allowed root (device dir) must link, not only paths under it."""
        device = tmp_path / "260711_Pionerskiy@i"
        raw = device / "_raw"
        raw.mkdir(parents=True)
        pat = _allowed_paths.allowed_path_re(str(device))
        # device dir itself now matches (was the bug: only _raw and below matched)
        assert [m.group(0) for m in pat.finditer(str(device))] == [str(device)]
        assert _allowed_paths.path_kind(str(device)) == "dir"

    def test_longest_match_wins_under_root(self, tmp_path):
        """Greedy: device/_raw matches the whole string, not just device."""
        device = tmp_path / "260711_Pionerskiy@i"
        raw = device / "_raw"
        raw.mkdir(parents=True)
        pat = _allowed_paths.allowed_path_re(str(device))
        text = str(raw)
        assert [m.group(0) for m in pat.finditer(text)] == [text]

    def test_root_with_commas_and_at(self, tmp_path):
        """Real device names contain @ and commas — root still matches itself."""
        device = tmp_path / "140228_Sambian@ADCP,ADV,i" / "inclinometer-Yantarniy"
        raw = device / "_raw"
        raw.mkdir(parents=True)
        pat = _allowed_paths.allowed_path_re(str(device))
        assert [m.group(0) for m in pat.finditer(str(device))] == [str(device)]
        # and the _raw child still matches as the longer string
        assert [m.group(0) for m in pat.finditer(str(raw))] == [str(raw)]


class TestLinkRoot:
    def test_inside_raw(self, tmp_path):
        raw = tmp_path / "dev" / "_raw"
        raw.mkdir(parents=True)
        assert link_root(raw) == raw.parent
        assert link_root(raw / "f.nc") == raw.parent

    def test_plain_dir_unchanged(self, tmp_path):
        assert link_root(tmp_path) == tmp_path


# ── App allowed-dir resolution ───────────────────────────────────────────────


class TestResolveAllowedDir:
    @staticmethod
    def _app(anchors: list[Path]) -> SimpleNamespace:
        return SimpleNamespace(_device_anchors=lambda: anchors)

    def test_single_device(self, tmp_path):
        dev = tmp_path / "dev"
        anchors = [dev / "_raw"]
        assert App._resolve_allowed_dir(self._app(anchors), "x") == str(dev)

    def test_multi_device_common_ancestor(self, tmp_path):
        anchors = [tmp_path / "d1" / "_raw", tmp_path / "d2" / "_raw"]
        assert App._resolve_allowed_dir(self._app(anchors), "x") == str(tmp_path)

    def test_no_anchors_falls_back_to_link_root(self, tmp_path):
        (tmp_path / "_raw").mkdir()
        assert App._resolve_allowed_dir(self._app([]), str(tmp_path / "_raw")) == str(tmp_path)

    def test_empty_field(self):
        assert App._resolve_allowed_dir(self._app([]), "") == "."

    def test_allowed_dir_cached_per_field(self, tmp_path):
        """Cache invalidates on field change — device_dir may change under it."""
        fields = iter([str(tmp_path / "a"), str(tmp_path / "b"), str(tmp_path / "b")])
        ns = self._app([])
        ns._path_field = SimpleNamespace(get=lambda: next(fields))
        ns._resolve_allowed_dir = lambda p: App._resolve_allowed_dir(ns, p)
        ns._allowed_dir_key = None
        ns._allowed_dir_val = ""
        first = App._allowed_dir(ns)
        second = App._allowed_dir(ns)
        third = App._allowed_dir(ns)
        assert (first, second, third) == (str(tmp_path / "a"), str(tmp_path / "b"), str(tmp_path / "b"))


# ── MarkdownLabel hover row (render-level line below the status text) ────────


class TestHoverLine:
    @pytest.fixture
    def label(self, _session_tk_root):
        if _session_tk_root is None:
            pytest.skip("Tk unavailable")
        w = MarkdownLabel(_session_tk_root)
        yield w
        w.destroy()

    def test_appended_raw_and_markdown_clean(self, label):
        label.set_text("Ready")
        label.set_hover_line(r"C:\a\_raw dir\f.nc")
        assert label.get("1.0", "end-1c") == rf"Ready{chr(10)}C:\a\_raw dir\f.nc"
        # Same-content re-render keeps the row (no new status written)
        label.rerender()
        assert label.get("1.0", "end-1c") == rf"Ready{chr(10)}C:\a\_raw dir\f.nc"

    def test_cleared_on_new_status_text(self, label):
        """A fresh status from another widget must clear the stale path row."""
        label.set_text("Ready")
        label.set_hover_line(r"C:\a\_raw dir\f.nc")
        assert r"C:\a\_raw dir\f.nc" in label.get("1.0", "end-1c")
        label.set_text("Other **msg**")
        assert label.get("1.0", "end-1c") == "Other msg"  # hover row gone
        label.set_hover_line("")
        assert label.get("1.0", "end-1c") == "Other msg"

    def test_no_markdown_mangling(self, label):
        label.set_text("s", raw=True)
        label.set_hover_line(r"C:\x\_y\_raw\f.nc")
        assert r"C:\x\_y\_raw\f.nc" in label.get("1.0", "end-1c")

    def test_hover_row_is_plain_not_a_link(self, label):
        """The revealed path in the status bar must not re-link into a hyperlink."""
        label.set_text("Ready", raw=True)
        label.set_hover_line(r"C:\x\_raw\f.nc")
        text = label.get("1.0", "end-1c")
        assert text.endswith(r"C:\x\_raw\f.nc")
        # no link range anywhere — the revealed path is plain text, not a hyperlink
        assert label.tag_ranges("link") == ()
        # the row index carries only the plain normal tag, no link/data tags
        row_start = f"1.0 + {len('Ready') + 1}c"
        assert label.tag_names(row_start) == ("normal",)

    def test_raw_status_path_not_autolinked(self, label):
        """A path shown as raw status (e.g. a log-hover reveal) must not be re-linked."""
        label.set_text(r"C:\data\f.nc", raw=True)
        assert label.tag_ranges("link") == ()
        # displayed verbatim (full path, not shrunk to a file name)
        assert label.get("1.0", "end-1c") == r"C:\data\f.nc"

    def test_rerender_keeps_current(self, label):
        """rerender must keep _current — _fit_width reads it (else width collapses to 1)."""
        label.set_text("Ready")
        assert label._current is not None
        label.set_hover_line(r"C:\a\f.nc")
        # The regression: _current became None → after_idle(_fit_width) → width=1
        assert label._current is not None
        pad = int(label.cget("padx")) * 2
        assert label._natural_width(label._current, pad) > 10

    def test_hover_line_switches_between_links(self, label):
        """A second hover (different target) must update the row — not no-op."""
        label.set_text("Ready")
        label.set_hover_line(r"C:\a\one.nc")
        label.set_hover_line(r"D:\b\two.nc")
        assert label.get("1.0", "end-1c") == rf"Ready{chr(10)}D:\b\two.nc"


# ── link hover handlers ──────────────────────────────────────────────────────


class TestLinkHover:
    def test_handler_dedup_and_off_link(self):
        shown: list[str] = []
        uri = "file:///C:/x/f.nc"
        w = SimpleNamespace(link_at=lambda x, y: uri if x > 0 else None)
        motion = make_link_hover_handler(w, shown.append)
        motion(SimpleNamespace(x=1, y=1))  # on link
        motion(SimpleNamespace(x=2, y=2))  # still on link — deduped
        motion(SimpleNamespace(x=-1, y=-1))  # off link
        motion(SimpleNamespace(x=-1, y=-1))  # still off — deduped
        assert shown == [uri, ""]

    def test_bind_leave_clears(self):
        shown: list[str] = []
        uri = "file:///C:/x/f.nc"
        binds: dict[str, object] = {}
        w = SimpleNamespace(
            link_at=lambda x, y: uri if x > 0 else None, bind=lambda seq, fn, add: binds.__setitem__(seq, fn)
        )
        bind_link_hover(w, shown.append)
        binds["<Motion>"](SimpleNamespace(x=1, y=1))
        binds["<Leave>"](None)
        assert shown == [uri, ""]

    def test_status_row_kept_on_off_link_motion(self):
        """Status-label mode: in-widget motion off the link keeps the row; only Leave clears."""
        shown: list[str] = []
        uri = "file:///C:/x/f.nc"
        w = SimpleNamespace(link_at=lambda x, y: uri if x > 0 else None)
        motion = make_link_hover_handler(w, shown.append, clear_on_off_link=False)
        motion(SimpleNamespace(x=1, y=1))  # on link
        motion(SimpleNamespace(x=-1, y=-1))  # off link — the row below status IS not a link
        motion(SimpleNamespace(x=2, y=2))  # still on link — deduped
        assert shown == [uri]  # no "" published — the row persists
        # The Leave binding's synthetic off-link event clears it
        leave = make_link_hover_handler(w, shown.append, clear_on_off_link=True)
        leave(SimpleNamespace(x=1, y=1))  # re-hovering sets last
        leave(SimpleNamespace(x=-1, y=-1))  # then leaving clears
        assert shown == [uri, uri, ""]
