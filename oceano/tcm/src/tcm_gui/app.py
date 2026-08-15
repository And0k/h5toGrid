"""Tk root: layout §1–5, 300 ms polling, event wiring."""

from __future__ import annotations

import ctypes
import logging
import sys
import tkinter as tk
from collections.abc import Sequence
from pathlib import Path, PurePath
from queue import Empty
from tkinter import ttk

from omegaconf import OmegaConf

import tcm_gui.theme
from tcm import cli, config_yaml, format, incl_calc, paths, schema
from tcm.states import ScanStage
from tcm_gui.cli_cfg import default_cfg

from ._browse_button import DATA_FILETYPES, SEARCH_FILETYPES, BrowseButtonManager, _is_shift_pressed
from ._help import help_for_path
from ._i18n import STRINGS as _S  # Chrome with auto-detection of OS locale if LANG=auto
from ._path_field import PathField
from ._rtf_clipboard import copy_rich
from ._tab_rail import TabRail
from .coef_sheet import ConfigSheet
from .const import (
    UIScale,
    configure_ui,
    get_widget_meta,
    set_widget_meta,
    widget_meta,
)
from .log_bridge import drain, install
from .md_label import MarkdownLabel
from .runtime import Runtime
from .theme import apply_theme_defaults
from .worker import Worker

lf = logging.getLogger(__name__)


def _tip_body(path: str, **kwargs: str) -> str:
    """Return help body text or empty string when the entry / body is absent."""
    e = help_for_path(path, **kwargs)
    return e.body if e and isinstance(e.body, str) and e.body else ""


class App:
    APP_ID = "Vendor.Product"  # todo: Fix, not hardcode here
    POLL = 300  # ms

    def __init__(self, argv: Sequence[str] | None = None) -> None:
        if sys.platform == "win32":
            shell32.SetCurrentProcessExplicitAppUserModelID(self.APP_ID)

        self.root = tk.Tk()
        self.ui = UIScale(self.root)
        configure_ui(self.root)
        self._theme = apply_theme_defaults(self.root)  # dark/light log colors
        self.root.title(_S.get("window.title", "TCM"))
        self.root.geometry("1100x800")

        # use exe icon
        self._icons: tuple[int, ...] = ()
        self.root.bind("<Destroy>", self._free_icons, add=True)
        self.set_window_icon()

        self.rt = Runtime()
        self._current: str | None = None  # currently selected page stem
        # Install the QueueHandler on the root logger once, for the lifetime
        # of the app, so log calls from the GUI main thread (e.g.
        # ``_reload_coefs`` triggered by treeview/cell interactions) reach
        # ``rt.log_queue`` → ScrolledText.  Worker's ``_wrap.wrapped``
        # re-attaches it after Hydra's ``dictConfig`` replaces root handlers,
        # so worker-thread logs also reach the queue.
        self.rt.queue_handler = install(self.rt.log_queue, self.rt.pause_gate)
        self.wk = Worker(self.rt)
        self._tip_active: bool = False  # error detail shown in _status_lbl; suppresses status updates
        self._error_active = False
        self._pages: dict[str, ConfigSheet] = {}
        self._yaml_paths: dict[str, Path] = {}
        self._tab_of: dict[str, ttk.Frame] = {}  # stem → notebook tab frame
        # Store original argv — Worker passes it to call_in_raw_dir which
        # extracts the data path via cli.parse_data_path(sys.argv) internally.
        self._original_argv = list(argv or sys.argv)

        # Configuration state — drives _overall_lbl caption transitions
        self._cfg_state = ScanStage.DEFAULT
        self._cfg_detail = ""  # suffix appended to _cfg_state in label (e.g. " - Done 100%")

        # watch Shift globally on root (Windows doesn't send Shift to widgets)
        self._full_mode = _is_shift_pressed()
        self._chrome_hovering: tk.Widget | None = None  # generic chrome hover guard
        self._browse_hovering: bool = False  # browse button Shift-hint hover guard

        self._build()
        self._status_font_fitted = False  # one-shot flag for _fit_status_font
        # Prefill path entry from CLI args — only when user explicitly provided one.
        # parse_data_path returns None when no positional arg is found (e.g.
        # ``python -m tcm_gui`` without a data path), so the GUI starts empty.
        path_in, _ = cli.parse_data_path(self._original_argv)
        self._initial_scan = path_in is not None
        if path_in is not None:
            self._path_field.set(str(path_in))
            self.root.after(100, self._scan)
        else:
            # No CLI path — show a placeholder page so the notebook isn't empty.
            self._add_page(_S.get("default_page.stem", "(default)"), default_cfg())
        self._poll()

    @property
    def _any_hovering(self) -> bool:
        """True when any hover guard is active — suppresses poll status clobber."""
        return (
            self._path_hovering
            or self._nb_hovering
            or self._chrome_hovering is not None
            or self._browse_hovering
        )

    # ── layout ──────────────────────────────────────────────────────

    def _build(self) -> None:
        r = self.root
        r.grid_rowconfigure(2, weight=2)  # notebook
        r.grid_rowconfigure(3, weight=1)  # log
        r.grid_columnconfigure(0, weight=1)

        # §1 input.path
        f0 = ttk.Frame(r)
        f0.grid(row=0, column=0, sticky="ew", padx=4, pady=2)
        f0.columnconfigure(1, weight=1)
        self._path_lbl = ttk.Label(f0, text=_S["path_lbl.tooltip"])
        self._path_lbl.grid(row=0, column=0, padx=(0, 4))
        self._path_field = PathField(
            f0,
            on_commit=self._on_path_changed,
            filetypes=SEARCH_FILETYPES,
            on_status=self._on_browse_status,
            status_hint=_S["browse_btn.status"],
            status_hint_files=_S["browse_btn.status_files"],
            on_shift=self._on_top_shift,
        )
        self._path_field.grid(row=0, column=1, sticky="ew")
        # Status message on hover — rebind on the Sheet's MT canvas
        self._path_hovering = False
        self._path_field.sh.MT.bind("<Enter>", lambda _: self._on_path_hover_in(), add="+")
        self._path_field.sh.MT.bind("<Leave>", lambda _: self._on_path_hover_out(), add="+")

        # §2 Overall status label — scan state / run-level progress text
        f1 = ttk.Frame(r)
        f1.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 0))
        f1.columnconfigure(0, weight=1)
        self._overall_lbl = ttk.Label(f1, text=self._translate_scan_stage(self._cfg_state))
        self._overall_lbl.grid(row=0, column=0, sticky="w")
        # §2b Progress status — current processing stage, right-aligned,
        # separate from the tab rail's visual fill.
        self._prog_status = ttk.Label(f1, text="", anchor="e")
        self._prog_status.grid(row=0, column=1, sticky="e", padx=(8, 0))
        self._status_hovering = False
        self._floater_hovering = False  # independent per-widget hover-hide flags
        # Last-seen stage snapshot — a change = progress advanced (see _poll_progress).
        self._stage_last: tuple[int, int, str] = (0, 0, "")

        # §3 Main area — vertical rail left, page stack right.
        # No notebook: rail owns selection entirely, tkraise() switches pages.
        self._main = ttk.Frame(r)
        self._main.grid(row=2, column=0, sticky="nsew", padx=4, pady=(0, 2))
        self._main.columnconfigure(1, weight=1)
        self._main.rowconfigure(0, weight=1)
        self._rail = TabRail(self._main, on_select=self._select_tab, on_hover=self._on_rail_hover)
        self._rail.grid(row=0, column=0, sticky="ns")
        self._stack = ttk.Frame(self._main)
        self._stack.grid(row=0, column=1, sticky="nsew")
        self._stack.rowconfigure(0, weight=1)
        self._stack.columnconfigure(0, weight=1)

        # §4 Run button — floats at main area bottom-right, parented on root for z-order
        self._run_btn = ttk.Button(r, text=_S["run_btn.text"], command=self._on_run, state="disabled")
        self._run_btn.place(in_=self._main, relx=1.0, rely=1.0, anchor="se", x=-24, y=-24)
        r.bind("<Configure>", lambda _: self._run_btn.lift(), add="+")

        # §5 Log — tk.Text + ttk.Scrollbar in a ttk.Frame (ScrolledText uses a
        # classic tk.Scrollbar that can't be styled via ttk.Style; a manual
        # container gives us a real ttk.Scrollbar matching tksheet's scrollbars).
        _log_frame = ttk.Frame(r)
        _log_frame.grid(row=3, column=0, sticky="nsew", padx=4, pady=2)
        _log_frame.grid_rowconfigure(0, weight=1)
        _log_frame.grid_columnconfigure(0, weight=1)
        self._log = tk.Text(
            _log_frame,
            height=10,
            state="disabled",
            wrap="word",
            bg=tcm_gui.theme.ENTRY_BG_FALLBACK,
            fg=tcm_gui.theme.FG_DEFAULT,
            insertbackground=tcm_gui.theme.FG_DEFAULT,
            borderwidth=0,
            highlightthickness=0,
        )
        self.ui.set_font(self._log)  # same scaled font as status label
        self._log.grid(row=0, column=0, sticky="nsew")
        self._log_vbar = ttk.Scrollbar(
            _log_frame,
            orient="vertical",
            command=self._log.yview,
            style="App.Vertical.TScrollbar",
        )
        self._log_vbar.grid(row=0, column=1, sticky="ns")
        self._log.configure(yscrollcommand=self._log_vbar.set)
        # Auto-scroll: follow new log lines unless user scrolled up manually.
        self._log_autoscroll: bool = True
        self._log.bind("<MouseWheel>", self._on_log_scroll, add="+")
        self._log.bind("<Button-4>", self._on_log_scroll, add="+")  # Linux scroll up
        self._log.bind("<Button-5>", self._on_log_scroll, add="+")  # Linux scroll down
        # Log status: show only while mouse is actively moving over the log.
        # A short timer clears the text when motion stops or pointer leaves.
        self._log_status_job: str | None = None
        self._log_status_fade_ms = 600
        self._log.bind("<Motion>", self._on_log_motion, add="+")
        self._log.bind("<Leave>", self._on_log_leave, add="+")
        for lvl, clr in tcm_gui.theme.TAG_COLORS.items():
            self._log.tag_configure(lvl, foreground=clr)
        self._log.tag_configure("func", foreground=tcm_gui.theme.FUNC_COLOR)
        # ``<<Copy>>`` is the virtual event Tk synthesises from Ctrl+C /
        # Ctrl+Ins at the C level (``<Control-Key-c>`` → ``<<Copy>>`` via
        # ``event add``).  Binding ``<Control-c>`` on root is dead code: the
        # virtual event mapping intercepts the raw keypress BEFORE any
        # ``<Control-c>`` binding sees it.  ``_log`` is ``state='disabled'``
        # so it can never take keyboard focus — a widget-scoped binding would
        # never fire.  The root ``<<Copy>>`` handler checks for a ``_log``
        # selection first; if present it serialises colored RTF via
        # :func:`copy_rich` and returns ``'break'`` to suppress further
        # propagation.  Otherwise it falls through so the focused widget
        # (e.g. ``_path_field`` ttk.Entry) keeps normal copy behaviour.
        self.root.bind("<<Copy>>", self._on_copy_rich, add="+")

        # §6 GUI status — MarkdownLabel overlaid bottom-left, dynamic width.
        self._status_lbl = MarkdownLabel(
            r,
            font=self.ui.font(),
            background=tcm_gui.theme.FRAME_BG_FALLBACK,
            foreground=tcm_gui.theme.FG_DEFAULT,
            colors=tcm_gui.theme.TAG_COLORS,
        )
        self._status_lbl.place(rely=1.0, relx=0.0, anchor="sw", x=4, y=0)

        # §6b Stage progress — overlaid bottom-right, shown only when active.
        # Text label sits above the bar inside a transparent-background frame.
        _bg = tcm_gui.theme.FRAME_BG_FALLBACK
        self._prog_floater = tk.Frame(r, bg=_bg, bd=0, highlightthickness=0)
        self._prog_stage_text = tk.Label(
            self._prog_floater,
            text="",
            anchor="e",
            justify="right",
            bg=_bg,
            fg=tcm_gui.theme.FG_DEFAULT,
            bd=0,
            highlightthickness=0,
            padx=4,
        )
        self._prog_stage_text.pack(side="top", anchor="e", fill="x")
        self._prog_stage = ttk.Progressbar(self._prog_floater, mode="determinate", length=220)
        self._prog_stage.pack(side="bottom", fill="x")
        self._prog_show_job: str | None = None  # after() id for delayed show

        # Z-order: mouse motion lifts GUI status (bottom-left) above floater —
        # except while an error is surfaced: the wide error-detail tooltip in
        # _status_lbl would bury the floater's error line between poll lifts.
        r.bind("<Motion>", self._lift_status_z, add="+")
        # Esc dismisses the error detail tooltip shown in _status_lbl.
        r.bind("<Escape>", lambda _e: self._hide_tip(), add="+")
        # Hover-hide: root <Motion> hides ONLY when the live pointer is over the
        # visible status widgets; motion anywhere else never hides them.  <Enter>
        # bindings proved unreliable — _poll_progress re-shows/lifts the floater
        # mid-motion, so <Enter> can't fire while the pointer is already inside.
        # Once hidden, only programmatic activation restores (progress advance /
        # explicit placement) — pointer leave alone never re-shows.
        r.bind("<Motion>", self._on_status_motion, add="+")

        # One pass: bind every chrome ``self._*`` widget to its help text / status
        # from STR.  No per-widget ``set_widget_meta`` calls above — role is derived
        # from the attribute name (without the leading underscore).  Widgets absent
        # from the registry get no help ("не ко всему").  Help is opt-in via STR.
        self._register_chrome_help()
        self._bind_chrome_hover()

    def _bind_chrome_hover(self) -> None:
        """Add <Motion>/<Leave> hover bindings to all chrome widgets with status.

        After ``_register_chrome_help`` populates ``widget_meta``, this method
        wires hover-to-status for every registered widget.  Widgets that already
        have dedicated hover handlers (``_path_field``, ``nb``) are skipped —
        their handlers are more specific (path hovering guard, notebook identify).
        ``_log`` is skipped: it uses its own Motion/Leave with a fade timer.
        """
        # Widgets with their own hover handling — skip to avoid conflicts.
        _skip = {self._path_field, self._rail, self._log}
        for w in widget_meta:
            if not isinstance(w, tk.Misc) or w in _skip:
                continue
            if "status" not in widget_meta[w]:
                continue
            w.bind("<Motion>", self._on_chrome_hover, add="+")
            w.bind("<Leave>", self._on_chrome_leave, add="+")

    def _on_chrome_hover(self, event: tk.Event) -> None:
        """Generic chrome hover: show widget's status text."""
        w = event.widget
        if status := get_widget_meta(w, "status"):
            self._chrome_hovering = w
            self._set_status(status)

    def _on_chrome_leave(self, _event: tk.Event) -> None:
        """Generic chrome leave: clear hover flag."""
        self._chrome_hovering = None

    def _register_chrome_help(self) -> None:
        """Bind chrome widgets to help text / status in one pass.

        Role = attribute name without the leading underscore.  ``tooltip`` is a
        static ``str`` from STR; ``status`` is either a static ``str`` from STR
        or a bound method returning the live caption (Run button: busy/paused).
        Dynamic ``status`` callables are resolved by :func:`get_widget_meta`
        at hover time — one read sees current state AND current language (STR).
        Widgets whose role has no STR entries get no binding → no help ("не ко всему").
        """
        for attr, w in vars(self).items():
            if not isinstance(w, tk.Misc):
                continue
            role = attr.lstrip("_")
            tooltip = _S.get(f"{role}.tooltip")
            status: object
            if role == "run_btn":
                # Dynamic: reflected busy / paused at hover time, not registration.
                status = self._run_btn_status
            else:
                status = _S.get(f"{role}.status")
            if tooltip is None and status is None:
                continue
            kwargs: dict[str, object] = {}
            if tooltip is not None:
                kwargs["tooltip"] = tooltip
            if status is not None:
                kwargs["status"] = status
            set_widget_meta(w, **kwargs)

    def _run_btn_status(self) -> str:
        """Dynamic Run button status: reflects disabled / busy / paused state, read live."""
        if str(self._run_btn.cget("state")) == "disabled":
            return _S["run_btn.disabled_no_pages" if not self._pages else "run_btn.disabled_invalid_path"]
        if not self.wk.busy:
            return _S["run.start"]
        return _S["run.resume"] if self.rt.pause_gate.paused else _S["run.pause"]

    def _fit_status_font(self) -> None:
        """Mark status label font as ready for auto-sizing.

        Called from ``_poll`` until the progress bar has real geometry.
        After marking ready, sets initial content and shows the progress
        overlay immediately when a CLI scan is pending.

        Does NOT resize fonts — the status label auto-grows vertically
        for multi-line content, so constraining to a single-row pixel
        budget is wrong.
        """
        bar_h = self._prog_stage.winfo_reqheight()
        if bar_h <= 4:
            return  # not yet laid out — retry on next poll
        self._status_font_fitted = True
        self._status_lbl.mark_font_ready()
        # Re-render cached content with scaled font.
        if not self._status_lbl.rerender():
            if self._initial_scan and not self._prog_floater.place_info():
                self._place_floater()
                self._prog_stage_text.config(text=_S["status.loading"])
            elif not self._initial_scan:
                self._set_status(_S["status.ready"], raw=True)

    def _on_path_hover_in(self) -> None:
        """Mouse enters PathField — show doc-driven status; Shift swaps to browse mode hint."""
        self._path_hovering = True
        # Show Shift-aware text on entry; the BrowseOverlay's 80 ms polling
        # re-publishes via _resolve_hint() for ongoing Shift toggles.
        status = self._path_field._path_status
        if _is_shift_pressed() and self._path_field._shift_status:
            status = self._path_field._shift_status
        self._set_status(status)

    def _on_path_hover_out(self) -> None:
        """Mouse leaves Entry — clear hover flag (status restored by poll)."""
        self._path_hovering = False

    def _on_browse_status(self, text: str) -> None:
        """Browse button hover: show hint on enter; on leave restore path status.

        ``text=""`` fires from BrowseOverlay ``<Leave>`` (mouse moved off the
        browse button back to the host widget).  Instead of clearing the status
        bar, restore the PathField's own hover status so the cell help persists.
        """
        self._browse_hovering = bool(text)
        if text:
            self._set_status(text)
        elif self._path_hovering:
            self._set_status(self._path_field._path_status)

    def _on_status_motion(self, event: tk.Event) -> None:
        """Hover-hide each status widget independently under the live pointer.

        Shown always; a widget hides only when the pointer is over THAT widget
        — hovering ``_prog_status`` (top row) never hides the floater and vice
        versa.  ``<Enter>`` can't "catch" the mouse here: the poll re-shows/
        lifts widgets mid-motion with the pointer already inside, so root
        ``<Motion>`` + per-event live bounds is the reliable trigger.

        Once hidden, a widget STAYS hidden after the pointer leaves —
        restoration is exclusively programmatic: progress advance
        (``_poll_progress`` clears both flags on snapshot change) or explicit
        placement (``_place_floater`` clears ``_floater_hovering``).
        """
        if (
            not self._status_hovering
            and self._prog_status.grid_info()
            and self._pointer_inside(event, self._prog_status)
        ):
            self._status_hovering = True
            self._prog_status.grid_remove()
        if (
            not self._floater_hovering
            and self._prog_floater.place_info()
            and self._pointer_inside(event, self._prog_floater)
        ):
            self._floater_hovering = True
            self._prog_floater.place_forget()

    def _place_floater(self) -> None:
        """Place + lift the stage floater — programmatic activation ends hover-hide."""
        self._floater_hovering = False
        self._prog_floater.place(relx=1.0, rely=1.0, anchor="se", x=-8, y=-4)
        self._prog_floater.lift()

    @staticmethod
    def _pointer_inside(event: tk.Event, w: tk.Widget) -> bool:
        """True when *event*'s screen coords fall inside *w*'s live bbox."""
        return (
            w.winfo_rootx() <= event.x_root <= w.winfo_rootx() + w.winfo_width()
            and w.winfo_rooty() <= event.y_root <= w.winfo_rooty() + w.winfo_height()
        )

    def _lift_status_z(self, _event: tk.Event) -> None:
        """Motion z-order: status label above floater — error floater above tooltip.

        ``_status_lbl`` grows to window width for the error-detail tooltip and
        visually buries the bottom-right floater when lifted over it.  During
        normal progress the motion lift keeps hover hints readable (the poll
        lifts the floater back on the next 300 ms tick), but while
        ``_error_active`` the floater carries the short error line — re-lift
        it immediately so it never flickers under the tooltip on mouse motion.
        """
        self._status_lbl.lift()
        if self._error_active and self._prog_floater.place_info():
            self._prog_floater.lift()

    def _on_top_shift(self, is_file: bool) -> None:
        """Top PathField Shift state changed — swap status text.

        Fires every 80 ms from the BrowseOverlay Shift poll when the state
        transitions (dir ↔ file).  Only updates when ``_path_hovering`` is
        True so irrelevant Shift events from other overlays are ignored.
        """
        if not self._path_hovering:
            return
        status = self._path_field._shift_status if is_file else self._path_field._path_status
        self._set_status(status)

    # ── §3 rail ↔ page stack sync ────────────────────────────────────

    _nb_hovering: bool = False  # set by _on_rail_hover, read by _any_hovering

    def _select_tab(self, stem: str) -> None:
        """Switch to *stem* page (rail click or programmatic).

        No events, no guards — the rail is the sole selection owner;
        tkraise() is the entire mechanism.  All pages stay mapped, so
        ConfigSheets keep their state across switches.
        """
        if (frame := self._tab_of.get(stem)) is None:
            return
        frame.tkraise()
        self._current = stem
        self._rail.set_selected(stem)

    def _on_rail_hover(self, name: str | None) -> None:
        """Rail hover callback — show yaml path in status bar."""
        if name:
            self._nb_hovering = True
            if frame := self._tab_of.get(name):
                self._set_status(get_widget_meta(frame, "status"), raw=True)
        else:
            self._nb_hovering = False

    @staticmethod
    def _fmt_multi(paths: tuple[str, ...]) -> str:
        """1 file → as-is;  N files → parent/(n1[.]ext|n2[.]ext)."""
        if len(paths) == 1:
            return paths[0]

        parent = PurePath(paths[0]).parent
        names = []
        for p in paths:
            name = PurePath(p).name
            stem, dot, ext = name.rpartition(".")
            names.append(f"{stem}[.]{ext}" if dot else name)
        return f"{parent.as_posix()}/({'|'.join(names)})"

    def _on_path_changed(self, _path: str) -> None:
        """PathField committed a new path — trigger scan with immediate overlay."""
        self._error_active = False
        self._hide_tip()
        self._initial_scan = True
        # Show progress overlay immediately (skip "Ready" → "Loading…" transition).
        self._set_status("", raw=True)
        if self._prog_floater.place_info():
            self._prog_floater.lift()
        else:
            self._place_floater()
        self._prog_stage_text.config(text=_S["status.loading"])
        self._scan()

    def _scan(self) -> None:
        path = self._path_field.get().strip()
        if path:
            # Shift+browse stores comma-separated paths; reformat as regex
            # alternation so find_dir_raw_absolute can resolve the _raw/ anchor.
            if "," in path:
                path = self._fmt_multi(tuple(path.split(",")))
                self._path_field.set(path)
            self._clear_log()
            # Live path field — not the stale startup argv — drives the scan,
            # so a GUI browse selection of ``_raw`` rescans that directory.
            # YAML auto-detection (`.yaml`/`.yml` → `input.yaml_path` filter)
            # happens in processing.run — no GUI-side plumbing needed.
            self.wk.scan(self._original_argv, path)

    # ── §2 page management ──────────────────────────────────────────

    def _add_page(self, stem: str, cfg: dict, yaml_path: Path | None = None) -> None:
        frame = ttk.Frame(self._stack)
        frame.grid(row=0, column=0, sticky="nsew")  # all pages share cell (0,0)
        self._tab_of[stem] = frame
        # Status for rail hover — relative yaml path when available.
        if yaml_path is not None and (field := self._path_field.get().strip()):
            anchor = paths.find_dir_raw_absolute(Path(field).absolute())
            try:
                rel = Path(yaml_path).relative_to(anchor)
                status_text = _S["tab.status"].format(path=rel)
            except ValueError:
                status_text = _S["tab.status"].format(path=yaml_path.name)
        else:
            status_text = _S["tab.status"].format(path=stem)
        set_widget_meta(frame, status=status_text)
        self._rail.add_tab(stem)

        cs = ConfigSheet(frame, status_hint=_S["browse_btn.status_files"])
        cs.sh.pack(fill="both", expand=True, padx=2, pady=2)
        cs._mgr = BrowseButtonManager(
            cs.sh,
            on_path_changed=lambda path: self._set_coefs_and_reload(stem, path),
            on_edit_restyler=cs._apply_edit_value,
            on_status=self._on_browse_status,
            status_hint=_S.get("sheet.input.path", ""),
            filetypes=DATA_FILETYPES,
            dir_title="",
        )
        cs.load(cfg, full=self._full_mode, config_root=schema.Config, return_enum=schema.Return)
        cs.on_hover_status = lambda msg, md=False: self._set_status(msg, raw=not md)
        cs.on_validity_change = self._update_run_btn_state
        cs.on_edit_begin = self._hide_tip
        cs._empty_area_hint = _S["empty_area.synced" if yaml_path is not None else "empty_area.unsaved"]
        self._pages[stem] = cs
        if len(self._tab_of) == 1:  # first page owns the stack
            self._select_tab(stem)

    def _set_coefs_and_reload(self, stem: str, coefs_path: str) -> None:
        """Called from ConfigSheet when ``input.coefs_path`` cell changes."""
        cs = self._pages.get(stem)
        if not cs or not coefs_path.strip():
            return
        tbl = format.pcid_to_raw_name(format.stem_to_pcid(stem))
        try:
            coefs = incl_calc.coefs.get_coefs(coefs_path.split(","), tbl)
        except Exception:
            lf.exception("Failed to load coefficients from %s", coefs_path)
            return
        cs._cfg.setdefault("input", {})["coefs"] = coefs
        cs._cfg.setdefault("input", {})["coefs_path"] = coefs_path
        cs.load(cs._cfg, full=self._full_mode, config_root=schema.Config, return_enum=schema.Return)

    # ── §3 Run / Pause / Resume ─────────────────────────────────────

    def _update_run_btn_state(self) -> None:
        """Enable Run iff at least one page exists and ALL have valid input.path."""
        ok = bool(self._pages) and all(cs.is_path_valid() for cs in self._pages.values())
        self._run_btn.config(state="normal" if ok else "disabled")

    def _on_run(self) -> None:
        if self.wk.busy:
            gate = self.rt.pause_gate
            (gate.resume if gate.paused else gate.pause)()
            self._run_btn.config(text=_S["run_btn.resume"] if gate.paused else _S["run_btn.pause"])
            return
        stems = list(self._pages)
        if not stems or not all(cs.is_path_valid() for cs in self._pages.values()):
            return
        for s, cs in self._pages.items():
            self._write_coefs(s, cs)
        self._clear_log()  # resets _error_active + hides tip
        self._cfg_detail = ""
        self._run_btn.config(text=_S["run_btn.pause"])
        self.rt.progress_bank.run_start(stems)
        self._overall_lbl.config(text=_S["status.starting"])
        self.wk.run(self._path_field.get(), stems)

    def _write_coefs(self, stem: str, cs: ConfigSheet) -> None:
        """Write edited coefs back to YAML — only if user actually changed something."""
        if not cs.is_dirty:
            return
        if not (yp := self._yaml_paths.get(stem)):
            return
        coefs, dates, path = cs._current_state()
        patch: dict = {"input": {}}
        if path:
            patch["input"]["path"] = path
        if coefs or dates:
            patch["input"]["coefs"] = {}
            if coefs:
                patch["input"]["coefs"].update(coefs)
            if dates:
                patch["input"]["coefs"]["dates"] = dates
        config_yaml.update_coefs_in_run_yaml(yp, patch)
        cs.mark_clean()

    # ── polling (300 ms) ────────────────────────────────────────────

    def _poll(self) -> None:
        if not self._status_font_fitted:
            self._fit_status_font()
        self._poll_dirty_tabs()
        self._poll_logs()
        self._poll_progress()
        self._poll_results()
        self.root.after(self.POLL, self._poll)

    def _poll_dirty_tabs(self) -> None:
        """Sync dirty indicator on rail cells."""
        for stem, cs in self._pages.items():
            self._rail.set_dirty(stem, cs.is_dirty)

    def _on_log_motion(self, _event: tk.Event) -> None:
        """Show log status text while mouse is actively moving; fade on pause."""
        if (status := get_widget_meta(self._log, "status")) is not None:
            self._set_status(status)
        # Reset the fade timer on every motion tick.
        if self._log_status_job is not None:
            self.root.after_cancel(self._log_status_job)
        self._log_status_job = self.root.after(self._log_status_fade_ms, self._on_log_status_fade)

    def _on_log_leave(self, _event: tk.Event) -> None:
        """Clear log status text immediately when pointer leaves."""
        if self._log_status_job is not None:
            self.root.after_cancel(self._log_status_job)
            self._log_status_job = None
        self._set_status("", raw=True)

    def _on_log_status_fade(self) -> None:
        """Fade timer expired — clear the log status text."""
        self._log_status_job = None
        self._set_status("", raw=True)

    def _on_log_scroll(self, _event: tk.Event) -> None:
        """Disable auto-scroll when user scrolls up; re-enable at bottom."""
        self.root.after_idle(self._check_log_scroll_position)

    def _check_log_scroll_position(self) -> None:
        """After scroll settles, toggle auto-scroll based on position."""
        y1 = self._log.yview()[1]
        # see("end") typically gives yview()[1] ≈ 0.91–0.98 depending on
        # content vs widget height.  Use 0.90 as the "near bottom" threshold.
        self._log_autoscroll = y1 > 0.90

    def _poll_logs(self) -> None:
        self._log.config(state="normal")
        if drain(self.rt.log_queue, self._log) and self._log_autoscroll:
            self._log.see("end")
        self._log.config(state="disabled")

    def _on_copy_rich(self, _event: tk.Event) -> str | None:
        """Root-level ``<<Copy>>`` → copy ``_log`` selection as RTF, else fall through.

        ``_log`` is ``state='disabled'`` and so cannot receive keyboard focus, so
        a widget-scoped ``<<Copy>>`` binding would never fire.  Tk's default
        ``<<Copy>>`` on Entry/Text copies plain text only — never RTF colors.
        This handler runs at root level so it fires after the focused widget's
        class binding (which already copied plain text to the clipboard).
        When ``_log`` carries a non-empty ``sel`` tag (mouse drag on the
        disabled text): serve RTF + plain via :func:`copy_rich` (overwriting
        the Entry's plain text) and return ``'break'`` to suppress further
        propagation.  Otherwise return ``None`` so the focused widget (e.g.
        ttk.Entry) keeps its normal copy behaviour.
        """
        if self._log.tag_ranges("sel"):
            copy_rich(self._log)
            return "break"
        return None

    # ── i18n translation helpers ────────────────────────────────────────────

    @staticmethod
    def _translate_desc(desc: str) -> str:
        """Translate backend stage description key to current locale.

        Handles simple keys (``"stage.discovering"``) and the templated
        ``composing:{stem}`` convention (single case from ``cli.py``).
        """
        if not desc:
            return desc
        if desc.startswith("composing:"):
            stem = desc.split(":", 1)[1]
            return _S.get("stage.composing_stem", "Composing {stem}\u2026").format(stem=stem)
        return _S.get(desc, desc)

    @staticmethod
    def _translate_scan_stage(stage: ScanStage) -> str:
        """Translate ScanStage enum value (a ``scan_stage.*`` key) to current locale."""
        return _S.get(stage, str(stage))

    def _poll_progress(self) -> None:
        # Snapshot both states once — avoids redundant lock acquisitions.
        cur, tot, desc = self.rt.progress_stage.snapshot()
        # Progress advanced (or stage changed) → new information ends
        # hover-hide; the branches below re-show the widgets.
        if (cur, tot, desc) != self._stage_last:
            self._stage_last = cur, tot, desc
            self._status_hovering = self._floater_hovering = False
        _cur_o, tot_o, desc_o = self.rt.progress_overall.snapshot()
        if tot > 0:
            self._prog_stage.config(maximum=tot, value=cur)
            # Only update stage text while stage is running AND no error is
            # surfaced — _surface_error appends the error to the current stage
            # text; _poll_progress must not clobber it on the next 300 ms tick.
            if cur < tot and not self._error_active:
                self._prog_stage_text.config(text=self._translate_desc(desc) or "")
            # Restore/update each widget independently of its own hover flag:
            # _prog_status (row-1-right label) + the floater (delayed show
            # avoids flashing for very short operations).
            if not self._status_hovering:
                self._prog_status.config(text=self._translate_desc(desc) or "")
                self._prog_status.grid()
            if not self._floater_hovering:
                if self._prog_floater.place_info():
                    self._prog_floater.lift()
                elif self._prog_show_job is None:
                    self._prog_show_job = self.root.after(400, self._show_prog_floater)
        else:
            # Cancel pending show if progress ended before delay.
            if self._prog_show_job is not None:
                self.root.after_cancel(self._prog_show_job)
                self._prog_show_job = None
            # Don't hide during initial scan — _fit_status_font showed it.
            # Keep floater visible while an error is surfaced so the localized
            # error text stays on screen until the next ok scan/run clears it.
            if not self._initial_scan and not self._error_active and self._prog_floater.place_info():
                self._prog_floater.place_forget()
            # Stage inactive: consume a one-shot clear signal (set at each
            # probe start via progress_stage.clear_and_reset) so stale text is
            # wiped exactly once; otherwise leave _status alone — explicit
            # setters own it ("Ready", "Done …", hover hints).
            if not self._any_hovering and self.rt.progress_stage.consume_clear():
                self._set_status("", raw=True)
            self._prog_status.config(text="")
        if tot_o > 0:
            # desc_o carries ScanStage i18n keys — translate like stage descs.
            self._overall_lbl.config(text=self._translate_desc(desc_o) or "")
        else:
            self._overall_lbl.config(text=f"{self._translate_scan_stage(self._cfg_state)}{self._cfg_detail}")
        # Per-config fills → rail; aggregate % → _overall_lbl suffix
        snaps = self.rt.progress_bank.snapshot_all()
        for cfg, (state, frac, _stage, _lvl) in snaps.items():
            self._rail.set_state(cfg, state, frac)
        if snaps and self.wk.busy:
            pct = round(100 * sum(v[1] for v in snaps.values()) / len(snaps))
            desc_o = self._translate_desc(self.rt.progress_overall.snapshot()[2])
            self._overall_lbl.config(text=f"{desc_o} \u2014 {pct}%" if desc_o else f"{pct}%")

    def _show_prog_floater(self) -> None:
        """Delayed show of stage progress overlay — skipped while the floater is hover-hidden."""
        self._prog_show_job = None
        if self._floater_hovering:
            return
        cur, tot, _desc = self.rt.progress_stage.snapshot()
        if tot > 0:
            self._prog_stage.config(maximum=tot, value=cur)
            self._place_floater()

    def _poll_results(self) -> None:
        try:
            kind, payload = self.rt.result_queue.get_nowait()
        except Empty:
            return
        {
            "scan_ok": self._on_scan_ok,
            "scan_error": self._on_scan_error,
            "run_ok": self._on_run_done,
            "run_error": self._on_run_error,
        }[kind](payload)

    def _on_scan_error(self, exc: BaseException) -> None:
        self._surface_error(exc, _S["error.scan"])

    def _on_run_error(self, exc: BaseException) -> None:
        self._run_btn.config(text=_S["run_btn.text"])
        self._surface_error(exc, _S["error.run"])

    def _surface_error(self, exc: BaseException, log_prefix: str) -> None:
        """Common error surface: log line, separator, floater text, detail tip.

        Sets ``_error_active`` so ``_poll_progress`` keeps the floater on screen
        until a fresh ok scan/run clears it.  The independently-logged exception
        (worker's ``lf.exception``) is already in ``_log``; here we add the
        localized prefix line + the markdown detail block rendered in
        ``_status_lbl`` (bottom-left overlay) via :meth:`_show_tip`.
        """
        self._error_active = True
        short = f"{type(exc).__name__}: {exc}"
        self._log_err(log_prefix.format(p=short))
        # Append the error to the current stage text (e.g. "Генерация конфигураций"
        # → "Генерация конфигураций\nError \"FileNotFoundError: …\".") so the
        # user sees both the stage that failed and the error details.
        current = self._prog_stage_text.cget("text")
        error = self._stage_error_text(exc)
        self._prog_stage_text.config(text=f"{current}\n{error}" if current else error)
        if not self._prog_floater.place_info():
            self._place_floater()
        if tip := _tip_body("input.path", mode="search", detail="Detailed"):
            self._show_tip(tip)
        else:
            self._hide_tip()

    @staticmethod
    def _stage_error_text(exc: BaseException) -> str:
        short = f"{type(exc).__name__}: {exc}"
        return _S.get("stage.error", 'Error "{msg}".').format(msg=short)

    def _set_status(self, text: str, *, raw: bool = False) -> None:
        """Set ``_status_lbl`` text, suppressed while the error tooltip is active.

        All chrome-hover, poll, and log-motion callers route through here so
        the tooltip (rendered in the same ``_status_lbl``) is never clobbered
        by a transient status update.  ``_show_tip`` / ``_hide_tip`` write to
        ``_status_lbl`` directly, bypassing this guard.
        """
        if self._tip_active:
            return
        self._status_lbl.set_text(text, raw=raw)

    def _show_tip(self, text: str) -> None:
        """Show error detail tooltip in ``_status_lbl``; suppress status updates.

        While ``_tip_active`` is True, :meth:`_set_status` is a no-op so hover
        hints and poll-driven status cannot overwrite the tooltip.  Dismissed
        by :meth:`_hide_tip` (new scan/run, path change, Esc, or cell edit).
        """
        self._tip_active = True
        self._status_lbl.set_text(text)  # markdown rendered, bypasses _set_status guard

    def _hide_tip(self) -> None:
        """Dismiss the error tooltip; resume normal status updates."""
        if not self._tip_active:
            return
        self._tip_active = False
        self._status_lbl.set_text("", raw=True)

    def _on_scan_ok(self, result) -> None:
        if not result or len(result) < 4:
            return

        self._error_active = False
        self._hide_tip()
        for frame in self._tab_of.values():
            frame.destroy()
        self._pages.clear()
        self._yaml_paths.clear()
        self._tab_of.clear()
        self._rail.clear()
        self._current = None
        for stem, yp, cfg_dc in result[3]:
            cfg = OmegaConf.to_container(cfg_dc, resolve=True)
            # Strip the technical ``CFG_FROM_ARGS`` sentinel injected by
            # ``worker._scan`` for early-exit — it is not a user config value.
            # Restore the real default so the dropdown shows the value that
            # ``processing.run`` will actually use when the user clicks Run.
            prog = cfg.get("program")
            if prog and prog.get("return_") == schema.Return.CFG_FROM_ARGS:
                prog["return_"] = str(schema.Return.END)
            self._yaml_paths[stem] = Path(yp)
            self._add_page(stem, cfg, yaml_path=Path(yp))
        self._cfg_state = ScanStage.DONE
        self._cfg_detail = ""
        self._initial_scan = False
        # Clear scan progress so _prog_floater hides on next poll.
        self.rt.progress_stage.set(0, 0, "")
        # Scan done — show "Ready" now.
        self._set_status(_S["status.ready"], raw=True)
        self._overall_lbl.config(text=self._translate_scan_stage(self._cfg_state))
        self._update_run_btn_state()

    def _on_run_done(self, result) -> None:
        self._error_active = False
        self._hide_tip()
        self._run_btn.config(text=_S["run_btn.text"])
        processed, failed = result[0], result[1]
        n = len(processed) + len(failed)
        pct = round(100 * len(processed) / n) if n else 100
        # Hide stage progress floater — completion shown in overall label
        if self._prog_floater.place_info():
            self._prog_floater.place_forget()
        self.rt.progress_stage.set(0, 0, "")
        self._cfg_state = ScanStage.DONE
        self._cfg_detail = _S["overall_lbl.done_detail"].format(pct=pct, ok=len(processed), n=n)
        self._overall_lbl.config(text=f"{self._translate_scan_stage(self._cfg_state)}{self._cfg_detail}")
        self.rt.progress_overall.set(0, 0, "")

    def _clear_log(self) -> None:
        """Flush pending queue records and clear the ScrolledText widget."""
        while True:
            try:
                self.rt.log_queue.get_nowait()
            except Empty:
                break
        self._log.config(state="normal")
        self._log.delete("1.0", "end")
        self._log.config(state="disabled")
        self._error_active = False
        self._hide_tip()

    def _log_err(self, msg: str, *, separator: bool = False) -> None:
        """Append ``msg`` as an ``error``-tagged line; optionally add a visual
        separator (``info`` tag) tying the error to the markdown detail block."""
        self._log.config(state="normal")
        self._log.insert("end", f"{msg}\n", "error")
        self._log.see("end")
        self._log.config(state="disabled")

    def run(self) -> None:
        self.root.mainloop()

    def set_window_icon(self, icon_index: int = 0) -> None:
        """Set title/taskbar icon from the executable's RT_GROUP_ICON."""
        if sys.platform != "win32":
            return

        WM_SETICON = 0x0080
        GA_ROOT = 2
        EXTRACT_FAIL = 0xFFFFFFFF

        self.root.update_idletasks()

        base = ctypes.wintypes.HWND(int(str(self.root.winfo_id()), 0))
        if not (hwnd := user32.GetAncestor(base, GA_ROOT) or base):
            return

        large, small = HICON(), HICON()

        n = shell32.ExtractIconExW(
            sys.executable,
            icon_index,
            ctypes.byref(large),
            ctypes.byref(small),
            1,
        )

        if not n or n == EXTRACT_FAIL:
            return

        # Если доступен только один размер, использовать его для обоих.
        large.value, small.value = (
            large.value or small.value,
            small.value or large.value,
        )

        if not (large.value or small.value):
            return

        new = tuple(dict.fromkeys(filter(None, (large.value, small.value))))

        for wparam, h in zip(
            (1, 0),  # ICON_BIG, ICON_SMALL
            (large.value, small.value),
            strict=False,
        ):
            if h:
                user32.SendMessageW(hwnd, WM_SETICON, wparam, h)

        # Старые иконки уничтожаются только после установки новых.
        for h in set(self._icons) - set(new):
            user32.DestroyIcon(h)

        self._icons = new

    def _free_icons(self, event: tk.Event) -> None:
        if sys.platform == "win32" and event.widget is self.root:
            for h in self._icons:
                user32.DestroyIcon(h)
            self._icons = ()


if sys.platform == "win32":
    shell32 = ctypes.WinDLL("shell32", use_last_error=True)
    user32 = ctypes.WinDLL("user32", use_last_error=True)

    LRESULT = getattr(ctypes.wintypes, "LRESULT", ctypes.c_ssize_t)
    HRESULT = getattr(ctypes.wintypes, "HRESULT", ctypes.c_long)
    HICON = getattr(ctypes.wintypes, "HICON", ctypes.c_void_p)

    shell32.ExtractIconExW.argtypes = (
        ctypes.wintypes.LPCWSTR,
        ctypes.c_int,
        ctypes.POINTER(HICON),
        ctypes.POINTER(HICON),
        ctypes.c_uint,
    )
    shell32.ExtractIconExW.restype = ctypes.c_uint

    shell32.SetCurrentProcessExplicitAppUserModelID.argtypes = (ctypes.wintypes.LPCWSTR,)
    shell32.SetCurrentProcessExplicitAppUserModelID.restype = HRESULT

    user32.SendMessageW.argtypes = (
        ctypes.wintypes.HWND,
        ctypes.wintypes.UINT,
        ctypes.wintypes.WPARAM,
        ctypes.wintypes.LPARAM,
    )
    user32.SendMessageW.restype = LRESULT

    user32.GetAncestor.argtypes = (
        ctypes.wintypes.HWND,
        ctypes.c_uint,
    )
    user32.GetAncestor.restype = ctypes.wintypes.HWND

    user32.DestroyIcon.argtypes = (HICON,)
    user32.DestroyIcon.restype = ctypes.wintypes.BOOL


def main(argv: list[str] | None = None) -> None:
    App(argv).run()
