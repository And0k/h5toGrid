"""Tk root: layout §1–5, 300 ms polling, event wiring."""

from __future__ import annotations

import copy
import ctypes
import logging
import os
import re
import sys
import traceback
import tkinter as tk
from collections.abc import Sequence
from pathlib import Path, PurePath
from queue import Empty
from tkinter import ttk
from typing import Any

from omegaconf import OmegaConf

import tcm_gui.theme
from tcm import cli, config_yaml, format, incl_calc, paths, schema, to_omegaconf
from tcm.states import ScanStage
from tcm_gui.cli_cfg import default_cfg, ensure_full_cfg, full_default_cfg

from ._about import AboutDialog, local_readme
from . import _reload_tabs as reload_tabs
from ._browse_button import DATA_FILETYPES, SEARCH_FILETYPES, BrowseButtonManager, _is_shift_pressed
from ._help import doc_path, help_for_path, help_general_for_path
from ._i18n import (
    STRINGS as _S,
    fmt_status,
    resolve_lang,
)  # Chrome with auto-detection of OS locale if LANG=auto
from ._numbered_dropdown import NumberedPathDropdown
from ._path_field import PathField
from ._rtf_clipboard import copy_rich
from ._tab_rail import TabRail
from .browser import get_documentation_browser, open_md_link
from .browser.browser import link_display
from .coef_sheet import ConfigSheet
from .const import (
    UIScale,
    configure_ui,
    fit_to_workarea,
    get_widget_meta,
    nudge_window,
    set_widget_meta,
    widget_meta,
)
from .keyboard import LayoutIndependentShortcuts
from .log_bridge import LogText, drain, install
from .md_label import MarkdownLabel, bind_link_hover
from .runtime import Runtime
from .theme import apply_theme_defaults
from .worker import Worker

lf = logging.getLogger(__name__)


def _tip_body(path: str, **kwargs: str) -> str:
    """Return help body text or empty string when the entry / body is absent."""
    e = help_for_path(path, **kwargs)
    return e.body if e and isinstance(e.body, str) and e.body else ""


# Chrome role → help source: ``path_lbl`` shares the PathField help — one source
# (config_reference ``path_field`` modes + STR tooltip), no duplicated keys.
_CHROME_ALIAS: dict[str, str] = {"path_lbl": "path_field"}


class App:
    APP_ID = "Vendor.Product"  # todo: Fix, not hardcode here
    GEOMETRY = (1100, 800)  # desired initial size — clamped to work area at start
    POLL = 300  # ms
    _DWELL_MS = 6000  # dwell tooltip delay — show detailed help after hover
    _DWELL_HIDE_MS = 1500  # dwell tooltip auto-close delay after show
    _STATUS_SETTLE_MS = 300  # status message switch/close debounce

    def __init__(self, argv: Sequence[str] | None = None) -> None:
        if sys.platform == "win32":
            shell32.SetCurrentProcessExplicitAppUserModelID(self.APP_ID)

        self.root = tk.Tk()
        self.root.withdraw()  # unmapped while building — fit_to_workarea deiconifies
        # once the geometry is final (no top-left default-size blink at start)
        self.ui = UIScale(self.root)
        configure_ui(self.root)
        # Layout-independent Ctrl+A/C/X/V/Z/Y/F — physical-key detection so
        # shortcuts work on non-Latin layouts (see keyboard.py). Installed once
        # here so every widget (Entry, Text, tksheet, dialogs) is covered.
        self._kbd = LayoutIndependentShortcuts(self.root)
        self._theme = apply_theme_defaults(self.root)  # dark/light log colors
        # Custom label style matching the config tree column tint.
        ttk.Style().configure("Overall.TLabel", background=tcm_gui.theme.CONFIG_TREE_BG)
        self.root.title(_S.get("window.title", "TCM"))
        # Alt+←/→/↑/↓ shifts the window (Shift = ×10) — a mouse drag can't
        # carry the title bar above the screen top, keyboard nudges can
        for key, d in (("Left", (-1, 0)), ("Right", (1, 0)), ("Up", (0, -1)), ("Down", (0, 1))):
            self.root.bind(f"<Alt-KeyPress-{key}>", lambda e, d=d: self._on_nudge(e, d))

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
        # Root defaults to WARNING — pre-Hydra worker probe INFO (e.g. parent
        # B:\Cruises\BalticSea trigger) would be filtered before reaching the
        # queue. Allow INFO+ from startup so parent scan_list traces survive.
        logging.getLogger().setLevel(logging.INFO)
        # Tkinter catches exceptions in callbacks itself and hands them to
        # ``report_callback_exception`` (default: stderr print only — sys.excepthook
        # never fires).  Route through logging so they reach ``_log``; frozen
        # builds have no console.  Since ``drain`` renders full tracebacks for
        # ``exc_info`` records, the hook can pass the exception object as exc_info.
        self.root.report_callback_exception = self._report_tk_exception
        self.wk = Worker(self.rt)
        self._tip_active: bool = False  # error detail shown in _status_lbl; suppresses status updates
        self._error_active = False
        # Dwell tooltip state — detailed help shown after _DWELL_MS of hover.
        # Unlike _tip_active (error), cleared on widget leave / Esc.
        self._dwell_job: str | None = None  # pending after() id
        self._dwell_active: bool = False  # dwell tip currently shown
        self._dwell_widget: tk.Widget | None = None  # widget that armed the dwell
        self._dwell_hide_job: str | None = None  # pending dwell auto-close after()
        self._status_job: str | None = None  # pending debounced _apply_status
        self._status_hovering: bool = False  # pointer on _status_lbl — hold the dwell tip
        self._pages: dict[str, ConfigSheet] = {}
        self._yaml_paths: dict[str, Path] = {}
        self._tab_of: dict[str, ttk.Frame] = {}  # stem → notebook tab frame
        self._run_forced_during_scan: bool = False
        # Store original argv — Worker passes it to call_in_raw_dir which
        # extracts the data path via cli.parse_data_path(sys.argv) internally.
        self._original_argv = list(argv or sys.argv)

        # Configuration state — drives _overall_lbl caption transitions
        self._cfg_state = ScanStage.DEFAULT
        self._cfg_detail = ""  # suffix appended to _cfg_state in label (e.g. " - Done 100%")

        # watch Shift globally on root (Windows doesn't send Shift to widgets)
        self._full_mode = _is_shift_pressed()
        self.rt.full_mode = self._full_mode  # worker injects simplified-mode `out` defaults per mode
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
            try:
                self._anchor_dropdown.refresh(str(path_in))
            except Exception:
                pass
            self.root.after(100, self._scan)
        else:
            # No CLI path — show a placeholder page so the notebook isn't empty.
            # Full mode gets every Config section (not just input) — same tree as post-scan.
            self._add_page(
                _S.get("default_page.stem", "(default)"),
                full_default_cfg() if self._full_mode else default_cfg(),
            )
            # Non-full mode: disable editing until scan finds configs.
            if not self._full_mode:
                for cs in self._pages.values():
                    cs.set_readonly(True)
                self._set_cfg_ui_disabled(True)
        # Fit within screen space minus taskbar, centered (one-shot, after the
        # window is built so the chrome-measured size is final) — no geometry
        # handler is left behind, so the user's mouse resize/move is untouched;
        # its deiconify() reveals the built window — first pixels on screen
        fit_to_workarea(self.root, *self.GEOMETRY)
        lf.debug("Root geometry %s after work-area clamp", self.root.geometry())
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

    # ── window movement ─────────────────────────────────────────────

    def _on_nudge(self, event: tk.Event, d: tuple[int, int]) -> str:
        """Alt+Arrow → shift the window by 10 px (Shift = ×10) in *d* direction."""
        step = 100 if event.state & 0x1 else 10
        nudge_window(self.root, d[0] * step, d[1] * step)
        return "break"

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
        self._path_lbl_default = _S["path_lbl.text"]
        self._path_lbl = ttk.Label(f0, text=self._path_lbl_default)
        self._path_lbl.grid(row=0, column=0, padx=(0, 4))
        self._allowed_dir_key: str | None = None  # link-root cache key (field value)
        self._allowed_dir_val = ""
        self._path_field = PathField(
            f0,
            on_commit=self._on_path_changed,
            on_begin_edit=self._hide_progress_widgets,
            on_end_edit=self._unfreeze_status,
            on_browse_click=self._hide_progress_widgets,
            filetypes=SEARCH_FILETYPES,
            on_status=self._on_browse_status,
            status_hint=_S["browse_btn.status"],
            status_hint_files=_S["browse_btn.status_files"],
            on_shift=self._on_top_shift,
        )
        self._path_field.grid(row=0, column=1, sticky="ew")
        # Numbered anchor popup on the path cell; the ordinal lives in
        # _path_lbl as "selected/total" (default text on no popup match).
        self._anchor_dropdown = NumberedPathDropdown(
            self._path_field.sh,
            0,
            0,
            [],
            number_label=self._path_lbl,
            default_text=self._path_lbl_default,
            on_select=self._on_anchor_dropdown_select,
        )
        # Up/Down on the cell opens the list (PathField.expand_dropdown).
        # The caption toggles the list — same path, no arrow aim.
        self._path_lbl.bind("<Button-1>", lambda _e: self._path_field.toggle_dropdown(), add="+")

        # Vertical separator + button bar (extensible container for future buttons)
        ttk.Separator(f0, orient="vertical").grid(row=0, column=2, sticky="ns", padx=(4, 2))
        self._button_bar = ttk.Frame(f0)
        self._button_bar.grid(row=0, column=3, sticky="e", padx=(0, 2))
        self._help_btn = ttk.Button(
            self._button_bar,
            text="?",
            width=2,
            command=self._on_help,
        )
        self._help_btn.pack(side="right")

        # Status message on hover — rebind on the Sheet's MT canvas
        self._path_hovering = False
        self._path_field.sh.MT.bind("<Enter>", lambda _: self._on_path_hover_in(), add="+")
        self._path_field.sh.MT.bind("<Leave>", lambda _: self._on_path_hover_out(), add="+")

        # §2 Status row — one line that COLLAPSES to the centered overall
        # caption while the stage progress is inactive.  _overall_lbl spans
        # column 0 (weight=1): its anchor centers the idle caption across the
        # whole row and left-aligns the text once the stage widgets (columns
        # 1–2) are gridded: overall caption, bar, stage description.
        f1 = ttk.Frame(r)
        f1.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 0))
        f1.columnconfigure(0, weight=1)
        self._status_hovering = False  # pointer on _status_lbl — hold the dwell tip
        self._status_lbl_f1_anchor: str | None = None  # F1 anchor for what's shown in _status_lbl
        self._stage_hovering = False  # pointer over the stage widgets — keep them hidden
        self._prog_show_job: str | None = None  # after() id for delayed show
        # Use mode-specific default stage text: full mode is editable, non-full is readonly.
        self._overall_lbl = ttk.Label(
            f1, text=self._default_stage_text(), anchor="center", style="Overall.TLabel"
        )
        self._overall_lbl.grid(row=0, column=0, sticky="ew")
        # §2b Stage progress — never gridded at build; only active stages grid
        # them (see _show_stage_progress), so the row starts collapsed.
        self._prog_stage = ttk.Progressbar(f1, mode="determinate", length=220)
        self._prog_stage_text = ttk.Label(f1, text="", anchor="w", justify="left")
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
        r.bind("<Configure>", lambda _e: self._raise_overlays(), add="+")

        # §5 Log — tk.Text + ttk.Scrollbar in a ttk.Frame (ScrolledText uses a
        # classic tk.Scrollbar that can't be styled via ttk.Style; a manual
        # container gives us a real ttk.Scrollbar matching tksheet's scrollbars).
        _log_frame = ttk.Frame(r)
        _log_frame.grid(row=3, column=0, sticky="nsew", padx=4, pady=2)
        _log_frame.grid_rowconfigure(0, weight=1)
        _log_frame.grid_columnconfigure(0, weight=1)
        self._log = LogText(
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
        self._log_link_hover: str = ""  # link URL under the log pointer ("" = none)
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
        # Non-Latin layouts are covered by ``LayoutIndependentShortcuts``
        # (installed in ``__init__``): it re-emits ``<<Copy>>`` from the
        # physical C key, which lands here the same way.
        self.root.bind("<<Copy>>", self._on_copy_rich, add="+")
        # F1 — context help: top path_field / focused-or-current sheet row
        # (selection first) / readme fallback.  One root binding — pages
        # bind nothing (per-sheet F1 bindings fired once per opened tab).
        self.root.bind("<F1>", self._on_f1_help, add="+")

        # §6 GUI status — MarkdownLabel overlaid bottom-left, dynamic width.
        self._status_lbl = MarkdownLabel(
            r,
            font=self.ui.font(),
            background=tcm_gui.theme.FRAME_BG_FALLBACK,
            foreground=tcm_gui.theme.FG_DEFAULT,
            colors=tcm_gui.theme.TAG_COLORS,
            on_link=open_md_link,
            # Bare data paths auto-link under the device dir / anchor parent
            allowed_dir=self._allowed_dir,
        )
        # Sheet color legend — bind markup names to actual theme colors so
        # {#sheet_*} spans in empty_area.* strings render with live values.
        self._status_lbl._colors.update(
            {
                "sheet_default": tcm_gui.theme.CELL_DEFAULT_VAL_FG,
                "sheet_placeholder": tcm_gui.theme.ghost_fg(tcm_gui.theme.ENTRY_BG_FALLBACK),
                "sheet_changed": tcm_gui.theme.FG_DEFAULT,
                "sheet_invalid": tcm_gui.theme.INVALID_FG,
                "sheet_warning": tcm_gui.theme.TAG_COLORS["warning"],
            }
        )
        self._status_lbl.place(rely=1.0, relx=0.0, anchor="sw", x=4, y=0)
        # Hovering the status text itself: pause the dwell auto-close so the
        # user can keep reading / click a link; the countdown resumes on leave.
        self._status_lbl.bind("<Enter>", self._on_status_enter, add="+")
        # Link hover → the full target in a row below the status text.  The row
        # persists for any in-widget motion (it is not itself a link) — only a
        # real Leave clears it, same persistence as the status text.
        bind_link_hover(self._status_lbl, self._show_link_hover, clear_on_off_link=False)
        self._status_lbl.bind("<Leave>", self._on_status_leave, add="+")

        # Esc dismisses the error detail tooltip shown in _status_lbl.
        r.bind("<Escape>", lambda _e: (self._hide_tip(), self._cancel_dwell(force=True)), add="+")
        # Hover-hide: root <Motion> hides ONLY when the live pointer is over the
        # visible stage widgets; motion anywhere else never hides them.  <Enter>
        # bindings proved unreliable — _poll_progress re-grids the widgets
        # mid-motion, so <Enter> can't fire while the pointer is already inside.
        # Once hidden, only programmatic activation restores (progress advance /
        # explicit placement) — pointer leave alone never re-shows.
        r.bind("<Motion>", self._on_status_motion, add="+")
        # Root <Leave> fires when the pointer exits the window from the gap
        # between widgets (root border).  Without it, a dwell tip shown while
        # hovering a widget would never start its linger if the pointer left
        # via the root border — the widget <Leave> is suppressed by the
        # root-border guard, so no one schedules the auto-close.
        r.bind("<Leave>", lambda _e: self._cancel_dwell(), add="+")

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
        """Generic chrome hover: show widget's status text; arm dwell tooltip."""
        w = event.widget
        if status := get_widget_meta(w, "status"):
            self._chrome_hovering = w
            # Widget's F1 anchor (if any) — stored for both status and dwell.
            f1 = get_widget_meta(w, "f1_anchor")
            # Widget changed → clear previous dwell before updating status.
            if self._dwell_widget is not w:
                self._cancel_dwell()
                self._dwell_widget = w
                self._set_status(status, f1)
                self._arm_dwell(get_widget_meta(w, "tooltip"), f1)
            else:
                self._set_status(status, f1)
            # Store the widget's F1 anchor if it has one (status label shows its help).
            if f1:
                self._status_lbl_f1_anchor = f1

    def _on_chrome_leave(self, _event: tk.Event) -> None:
        """Generic chrome leave: clear hover flag + cancel dwell."""
        self._chrome_hovering = None
        self._dwell_widget = None
        self._cancel_dwell()

    def _register_chrome_help(self) -> None:
        """Bind chrome widgets to help text / status in one pass.

        Role = attribute name without the leading underscore, aliased via
        :data:`_CHROME_ALIAS` (``path_lbl`` → ``path_field`` — the label shows
        the field's own help).  ``tooltip`` is a static ``str`` from STR;
        ``status`` is either a static ``str`` from STR, a bound method returning
        the live caption (Run button: busy/paused) or doc-driven
        (``path_field``: config_reference search-mode short body).
        Dynamic ``status`` callables are resolved by :func:`get_widget_meta`
        at hover time — one read sees current state AND current language (STR).
        Widgets whose role has no STR entries get no binding → no help ("не ко всему").
        """
        for attr, w in vars(self).items():
            if not isinstance(w, tk.Misc):
                continue
            role = attr.lstrip("_")
            src = _CHROME_ALIAS.get(role, role)
            tooltip = _S.get(f"{src}.tooltip")
            status: object
            if role == "run_btn":
                # Dynamic: reflected busy / paused at hover time, not registration.
                status = self._run_btn_status
            elif src == "path_field":
                # Doc general + STR suffix — identical to PathField hover status.
                base = _tip_body("path_field", mode="dirs")
                suffix = _S.get("path_field.status.dirs", "")
                status = f"{base} {suffix}".strip() if base and suffix else (base or suffix or None)
            else:
                status = _S.get(f"{src}.status")
            if tooltip is None and status is None:
                continue
            kwargs: dict[str, object] = {}
            if tooltip is not None:
                kwargs["tooltip"] = tooltip
            if status is not None:
                kwargs["status"] = status
            # F1 anchor for this chrome role (path_field uses its own entry).
            f1_src = "path_field" if src == "path_field" else src
            if f1_entry := help_for_path(f1_src):
                kwargs["f1_anchor"] = f1_entry.anchor
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
            if self._initial_scan and not self._stage_shown:
                self._show_stage_progress()
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
        self._cancel_dwell()
        f1 = e.anchor if (e := help_for_path("path_field")) else None
        self._set_status(status, f1)
        self._status_lbl_f1_anchor = f1
        # Dwell: doc-driven Detailed for the current pathField regime (dirs/files),
        # fallback to generic STR tooltip.
        _mode = "files" if _is_shift_pressed() and self._path_field._shift_status else "dirs"
        if (
            (de := help_for_path("path_field", mode=_mode, detail="Detailed"))
            and isinstance(de.body, str)
            and de.body
        ):
            _tip = de.body
        else:
            _tip = _S.get("path_field.tooltip", "")
        self._arm_dwell(_tip, f1)

    def _on_path_hover_out(self) -> None:
        """Mouse leaves Entry — clear hover flag + cancel dwell (status restored by poll)."""
        self._path_hovering = False
        self._cancel_dwell()

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

    def _hide_progress_widgets(self) -> None:
        """Collapse the stage progress row on user interaction (edit/browse start).

        Freezes hover status (``_status_hovering``) so the label does not
        switch to cell-hover hints while an editor has focus; released on
        edit-end (see :meth:`_unfreeze_status`).  Also hides the stage row via
        ``_stage_hovering`` until programmatic activation (progress advance /
        explicit placement).  Called when the user starts editing tksheet
        cells, the path field, or clicks browse.
        """
        self._status_hovering = True
        self._stage_hovering = True
        self._hide_stage_progress()

    def _unfreeze_status(self) -> None:
        """Release the edit-time status freeze (editor closed).

        Keeps ``_stage_hovering`` untouched — the stage row resumes only on
        progress advance.  Normal hover status resumes once no editor holds the
        freeze, while the true pointer hold (the status label ``<Leave>``) is
        tracked independently by ``_on_status_enter``/``_on_status_leave``.
        """
        self._status_hovering = False

    def _on_status_motion(self, event: tk.Event) -> None:
        """Hover-hide the stage widgets under the live pointer.

        Shown always; the row collapses only when the pointer is over the bar
        or the stage text.  ``<Enter>`` can't "catch" the mouse here: the poll
        re-grids the widgets mid-motion with the pointer already inside, so
        root ``<Motion>`` + per-event live bounds is the reliable trigger.

        Once hidden, the row STAYS collapsed after the pointer leaves —
        restoration is exclusively programmatic: progress advance
        (``_poll_progress`` clears the flag on snapshot change) or explicit
        placement (``_show_stage_progress``).
        """
        if (
            not self._stage_hovering
            and self._stage_shown
            and (
                self._pointer_inside(event, self._prog_stage)
                or self._pointer_inside(event, self._prog_stage_text)
            )
        ):
            self._stage_hovering = True
            self._hide_stage_progress()

    @property
    def _stage_shown(self) -> bool:
        """True while the stage progress widgets are gridded in the status row."""
        return bool(self._prog_stage.grid_info())

    def _show_stage_progress(self) -> None:
        """Grid bar + stage text into the row, left-align the overall caption.

        Programmatic activation — also clears ``_stage_hovering`` so a
        hover-hidden row re-appears.
        """
        self._stage_hovering = False
        self._prog_stage.grid(row=0, column=1, padx=(8, 4))
        self._prog_stage_text.grid(row=0, column=2, sticky="w")
        self._overall_lbl.configure(anchor="w")

    def _hide_stage_progress(self) -> None:
        """Collapse the row — overall caption alone, centered across it."""
        self._prog_stage.grid_remove()
        self._prog_stage_text.grid_remove()
        self._overall_lbl.configure(anchor="center")

    @staticmethod
    def _pointer_inside(event: tk.Event, w: tk.Widget) -> bool:
        """True when *event*'s screen coords fall inside *w*'s live bbox."""
        return (
            w.winfo_rootx() <= event.x_root <= w.winfo_rootx() + w.winfo_width()
            and w.winfo_rooty() <= event.y_root <= w.winfo_rooty() + w.winfo_height()
        )

    def _pointer_in_dwell_hierarchy(self) -> bool:
        """True when the pointer is within the dwelling widget's hierarchy.

        Covers three cases where a ``<Leave>`` on the dwelling widget must NOT
        cancel the dwell (the tip stays so the user can keep reading):

        1. Pointer is still on the dwelling widget itself — Tk can fire
           ``<Leave>`` while the pointer is at the widget's edge (premature
           leave, before the resize cursor would appear).
        2. Pointer moved to an ancestor (parent frame) — the user perceives
           this as "still in the gap", not on another interactive widget.
        3. Pointer moved to the root window background.

        The root is an ancestor of every widget, so walking up from the
        dwelling widget and checking for the pointer's widget covers all
        three cases in one pass.
        """
        if self._dwell_widget is None:
            return False
        pw = self._pointer_widget()
        if pw is None:
            return False
        return self._within(self._dwell_widget, pw)

    def _pointer_on_status_label(self) -> bool:
        """True when the pointer is over the status label or a descendant.

        The status label shows both short status and the markdown dwell
        tooltip; F1 resolves the anchor from the row whose help is shown
        there.  A ``MarkdownLabel`` is a ``tk.Text`` that may embed link
        windows — the master walk reaches ``_status_lbl`` for any of them.
        """
        pw = self._pointer_widget()
        if pw is None:
            return False
        return self._within(pw, self._status_lbl)

    def _pointer_widget(self) -> tk.Widget | None:
        """Topmost widget under the pointer, or ``None`` if outside the window."""
        x, y = self.root.winfo_pointerxy()
        return self.root.winfo_containing(x, y)

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

    def _on_help(self) -> None:
        """Open the About dialog (version, runtime info, docs browser)."""
        dlg = AboutDialog(
            self.root,
            full_mode=self._full_mode,
            ui=self.ui,
            on_status=lambda msg: self._set_status(msg, raw=True),  # hover URLs/paths
        )
        # dialog closed → restore Ready (its hover texts die with the window)
        dlg.bind(
            "<Destroy>", lambda e: e.widget is dlg and self._set_status(_S["status.ready"], raw=True), add="+"
        )

    @staticmethod
    def _within(widget, ancestor) -> bool:
        """True iff *widget* is *ancestor* or nested in it (master walk)."""
        while widget is not None:
            if widget is ancestor:
                return True
            widget = getattr(widget, "master", None)
        return False

    def _on_f1_help(self, _event=None) -> None:
        """F1 — open the doc browser for the focused/selected widget.

        Resolution: top path field (focus inside its subtree) → mouse over
        the status label (``_status_lbl`` — shows both the short status and
        the dwell tooltip; detected by pointer position at press time, not a
        tracked flag) → focused or current page's SELECTED row, or if
        nothing selected the mouse is inside the dwell tooltip widget
        (``_hover_field``) — (:meth:`ConfigSheet._f1_anchor`; child rows walk
        up to the parent's section) → localized readme when nothing is
        selected or the tooltip is not hovered.  The served doc MUST be the
        same localized file the entries were parsed from
        (``doc_path(resolve_lang())`` — exactly what ``_help._load`` reads);
        plain ``doc_path()`` always serves English and a localized anchor
        then finds no element — the page opens but never scrolls.
        """
        try:
            w = self.root.focus_get()  # KeyError on menu focus, TclError on dead widgets
        except Exception:  # noqa: BLE001 — focus anomalies degrade to no-focus
            w = None
        if self._within(w, self._path_field):
            anchor = e.anchor if (e := help_for_path("path_field")) else ""
        elif self._pointer_on_status_label() and self._status_lbl_f1_anchor:
            # Mouse over the status label (short status or dwell tooltip) — use the stored anchor.
            anchor = self._status_lbl_f1_anchor
        else:
            # Focused page owns the answer; no focus inside any sheet → current page.
            cs = next((p for p in self._pages.values() if self._within(w, p.sh)), None) or self._pages.get(
                self._current
            )
            anchor = cs._f1_anchor() if cs else ""
        try:
            br = get_documentation_browser()
            if anchor:
                br.open(doc_path(resolve_lang()), anchor=anchor)
            else:
                br.open(local_readme())
        except (OSError, ValueError):
            lf.exception("F1: failed to open documentation")

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
        # Deferred: fill time_ranges/burst for this row only (scan created stubs without reading files)
        try:
            self._lazy_fill_for_stem(stem)
        except Exception:
            pass

    def _on_rail_hover(self, name: str | None) -> None:
        """Rail hover callback — show yaml path in status bar; cancel dwell."""
        self._cancel_dwell()
        if name:
            self._nb_hovering = True
            if frame := self._tab_of.get(name):
                self._set_status(get_widget_meta(frame, "status"), raw=True)
        else:
            self._nb_hovering = False

    def _lazy_fill_for_stem(self, stem: str) -> None:
        """Deferred per-row metadata fill (scan created stubs without file reads).

        On first selection of a row whose ``input.time_ranges`` is missing and
        whose ``info_devices`` metadata lacks time_range/burst, read the file
        edges + burst in a daemon thread and patch the sheet/YAML.  Empty
        files are tolerated (no time_ranges, just log).
        """
        cs = self._pages.get(stem)
        if cs is None:
            return
        # Already has time_ranges or already scheduled?
        cfg = getattr(cs, "_cfg", None) or getattr(cs, "_data", None) or {}
        tr = (cfg.get("input", {}) or {}).get("time_ranges") if isinstance(cfg, dict) else None
        if tr and len(tr) >= 2 and tr[0] and tr[1]:
            # Has time_ranges — still check burst lazy fill via metadata sheet?
            # Burst is handled via metadata rows; if already present skip
            return
        if getattr(cs, "_lazy_pending", False):
            return
        cs._lazy_pending = True  # type: ignore[attr-defined]

        def _work() -> None:
            try:
                # Resolve data file path from sheet cfg
                path_str = ""
                try:
                    path_str = (cfg.get("input", {}) or {}).get("path", "") if isinstance(cfg, dict) else ""
                except Exception:
                    path_str = ""
                if not path_str:
                    return
                p = Path(str(path_str))
                # Use same logic as bursts/gen_metadata but for single file
                # Edge rows via csv_load (handles empty gracefully)
                time_ranges: list[str] | None = None
                bdt: Any = None
                bst: Any = None
                try:
                    from tcm import csv_load as _cl

                    # Build minimal cfg_in for edge read (use defaults + sheet cfg)
                    cfg_in = {
                        **_cl.cfg_default["in"],
                        **(cfg.get("input", {}) if isinstance(cfg, dict) else {}),
                    }
                    cfg_in["corr_time_mode"] = cfg_in.get("corr_time_mode", True)
                    # Single-file dict for load_from_csv_gen
                    # Probe identity from stem
                    try:
                        pcid = format.to_pcid_from_name(format.stem_to_pcid(stem))
                        model, num = format.probe_from_name(format.stem_to_pcid(stem).lower()) or (
                            pcid[:1],
                            int(pcid[2:]) if pcid[2:].isdigit() else 0,
                        )
                    except Exception:
                        model, num = "i", 0
                    # For archive composites, skip edge read (burst via meta_finder will handle)
                    if p.as_posix().lower().find(".zip/") != -1 or p.as_posix().lower().find(".7z/") != -1:
                        pass
                    else:
                        # Loose file — try edge read
                        try:
                            # search via single file path
                            files_dict = _cl.search_csv_files(p) if p.is_file() else {}
                            if files_dict:
                                for _k, _flist in files_dict.items():
                                    # Find the matching file
                                    for _f in _flist:
                                        if Path(_f).resolve() == p.resolve() or Path(_f).name == p.name:
                                            # Use load_from_csv_gen for this file only
                                            from collections import defaultdict as _dd

                                            single_dict = {_k: [_f]}
                                            for df_edges, (_ipid, _pcid2, _pp) in _cl.load_from_csv_gen(
                                                csv_files_dict=single_dict,
                                                cfg_in=cfg_in,
                                                return_="first_last_row",
                                            ):
                                                if df_edges is not None and len(df_edges) >= 1:
                                                    time_ranges = [dt.isoformat() for dt in df_edges.index]
                                                break
                                            break
                        except Exception:
                            lf.debug("Lazy edge read failed for {}", stem, exc_info=True)
                    # Burst via meta_finder (same as bursts.collect but single)
                    try:
                        from pathlib import PurePosixPath as _PP

                        from meta_finder.data_proc_funcs import extract_time_info_from_text_file as _eti
                        from tcm.search import is_archive_composite, split_archive_path

                        avg = cfg_in.get("averaging_interval") or 2
                        if is_archive_composite(p):
                            sp = split_archive_path(p)
                            if sp:
                                da, rel = sp
                                info = _eti(da, rel, averaging_interval=avg)
                                if info:
                                    _, _, bdt, bst = info
                        else:
                            da, rel = p.parent, _PP(p.name)
                            if da.is_dir():
                                info = _eti(da, rel, averaging_interval=avg)
                                if info:
                                    _, _, bdt, bst = info
                    except ImportError:
                        pass
                    except Exception:
                        lf.debug("Lazy burst read failed for {}", stem, exc_info=True)
                except Exception:
                    lf.debug("Lazy fill pre-check failed for {}", stem, exc_info=True)

                # Patch GUI on main thread
                def _apply() -> None:
                    try:
                        cs2 = self._pages.get(stem)
                        if cs2 is None:
                            return
                        patched = False
                        if time_ranges and len(time_ranges) >= 2:
                            # Update sheet cfg and YAML on disk if time_ranges was empty
                            try:
                                # Update in-memory cfg
                                if hasattr(cs2, "_cfg") and isinstance(cs2._cfg, dict):
                                    cur = cs2._cfg.get("input", {}).get("time_ranges")
                                    if not cur or len(cur) < 2 or not cur[0] or not cur[1]:
                                        cs2._cfg.setdefault("input", {})["time_ranges"] = time_ranges
                                        patched = True
                                # Update YAML file if exists
                                yp = self._yaml_paths.get(stem)
                                if yp and yp.is_file():
                                    try:
                                        from tcm.config_yaml import _ry as _ry2

                                        ry = _ry2()
                                        cur_y = ry.load(yp) or {}
                                        cur_tr = (cur_y.get("input", {}) or {}).get("time_ranges")
                                        if not cur_tr or len(cur_tr) < 2 or not cur_tr[0] or not cur_tr[1]:
                                            cur_y.setdefault("input", {})["time_ranges"] = time_ranges
                                            ry.dump(cur_y, stream=yp.open(encoding="utf-8", mode="w"))
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                            # Refresh sheet display
                            try:
                                cs2.load(
                                    cs2._cfg,
                                    full=self._full_mode,
                                    config_root=schema.Config,
                                    return_enum=schema.Return,
                                    metadata=getattr(cs2, "_setups", None),
                                    sync_status=getattr(cs2, "_sync_status", None),
                                    metadata_path=getattr(cs2, "_metadata_path", None),
                                )
                                cs2.sh.redraw()
                            except Exception:
                                pass
                        # Burst autofill for metadata sheet (if missing) — also when md is None (deferred scan stub)
                        if bdt is not None and bst is not None:
                            try:
                                # Seed/backfill burst pair on the first setup; dirty
                                # when anything changed so Run writes info_devices.yaml
                                if hasattr(cs2, "autofill_burst") and cs2.autofill_burst(
                                    bdt, bst, time_ranges=time_ranges
                                ):
                                    cs2._apply_metadata_dirty_label()  # type: ignore[attr-defined]
                                    cs2._apply_validations()  # type: ignore[attr-defined]
                                    cs2.sh.redraw()
                                    self._rail.set_dirty(stem, False, True)
                            except Exception:
                                pass
                    finally:
                        try:
                            cs2._lazy_pending = False  # type: ignore[attr-defined]
                        except Exception:
                            pass

                self.root.after(0, _apply)
            except Exception:
                try:
                    cs._lazy_pending = False  # type: ignore[attr-defined]
                except Exception:
                    pass

        import threading

        threading.Thread(target=_work, daemon=True).start()

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
        try:
            self._anchor_dropdown.refresh(self._path_field.get().strip())
        except Exception:
            pass
        self._error_active = False
        self._path_field.set_error(False)  # fresh search attempt clears the failure mark
        self._hide_tip()
        self._initial_scan = True
        self._set_cfg_ui_disabled(False)  # search activity — drop the inert look
        # Show the stage row immediately (skip "Ready" → "Loading…" transition).
        self._set_status("", raw=True)
        self._show_stage_progress()
        self._prog_stage_text.config(text=_S["status.loading"])
        self._scan()

    def _scan(self) -> None:
        path = self._path_field.get().strip()
        if path and "," in path:
            # Shift+browse joins ``askopenfilenames()`` with "," and we support
            # multiple paths as a regex alternative ``parent/(a|b)`` via
            # ``_fmt_multi``.  A single path may itself contain a comma
            # (e.g. cruise ``251201_ABP64@i,t-chain``) — naive ``split(",")``
            # would mangle it.  Multi-file strings are ``",".join`` of absolute
            # paths (``B:\…``); an embedded comma is followed by
            # ``t-chain\…``, not a drive.  Split only at "," before ``X:\``.
            if Path(path).exists():
                parts: tuple[str, ...] = ()
            else:
                parts = tuple(p.strip() for p in re.split(r",(?=[A-Za-z]:[\\/])", path) if p.strip())
            if len(parts) > 1 and all(Path(p).is_absolute() for p in parts):
                path = self._fmt_multi(parts)
                self._path_field.set(path)
                try:
                    self._anchor_dropdown.refresh(path)
                except Exception:
                    pass
        self._clear_log()
        # Live path field — not the stale startup argv — drives the scan,
        # so a GUI browse selection of ``_raw`` rescans that directory.
        # YAML auto-detection (`.yaml`/`.yml` → `input.yaml_path` filter)
        # happens in processing.run — no GUI-side plumbing needed.
        # An EMPTY path is forwarded as-is: it means "./" and the pipeline
        # owns the verdict (cli.call_in_raw_dir rejects anchors inside the
        # code project) — the failure surfaces like any other scan error
        # instead of freezing the "Loading…" stage.
        self.wk.scan(self._original_argv, path)
        # Enable Run as Pause during scan (even with no pages yet) so user can pause long discovery
        try:
            self._run_btn.config(state="normal", text=_S["run_btn.pause"])
            self._run_forced_during_scan = True
        except Exception:
            pass

    # ── §2 page management ──────────────────────────────────────────

    def _add_page(
        self,
        stem: str,
        cfg: dict,
        yaml_path: Path | None = None,
        metadata: list | None = None,
        sync_status: dict | None = None,
        metadata_path: str | None = None,
    ) -> None:
        frame = ttk.Frame(self._stack)
        frame.grid(row=0, column=0, sticky="nsew")  # all pages share cell (0,0)
        self._tab_of[stem] = frame
        # Status for rail hover — full yaml path, stem when no file backs the page.
        status_text = fmt_status(_S["tab.status"], path=yaml_path if yaml_path else stem)
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
            on_click=self._hide_progress_widgets,
        )
        if self._full_mode:
            # Thin run YAMLs carry only overrides — backfill structured defaults so
            # _build_full renders every section (out/filter/program), not just input.
            ensure_full_cfg(cfg)
        cfg["_page_stem"] = stem
        cs.load(
            cfg,
            full=self._full_mode,
            config_root=schema.Config,
            return_enum=schema.Return,
            metadata=metadata,
            sync_status=sync_status,
            metadata_path=metadata_path,
        )
        cs._page_stem = stem
        # Visual sync indicator on time_ranges row (broader → warning fg) + hover detail
        if sync_status:
            try:
                cs.apply_time_ranges_sync_status(sync_status)
            except Exception:
                pass
        cs.on_hover_status = lambda msg, md=False: self._on_cell_status(cs, msg, md)
        cs.on_edit_begin = lambda: (self._hide_tip(), self._hide_progress_widgets())
        cs.on_edit_end = self._unfreeze_status
        cs.on_validity_change = self._update_run_btn_state
        cs._empty_area_hint = _S[
            "empty_area.synced"
            if yaml_path is not None
            else ("empty_area.unsaved_full" if self._full_mode else "empty_area.unsaved")
        ]
        self._pages[stem] = cs
        # Cross-tab metadata sync: a metadata mutation on this sheet fans out
        # to same-identity peers (same device file + pcid).  _metadata_identity
        # resolves the key; None → no sync (indeterminate).
        cs.on_metadata_changed = lambda: self._on_metadata_changed(cs)
        cs.on_instant_apply = lambda kind, stem=stem: self._on_instant_apply(stem, kind)
        # NOTE: no _select_tab here — a page gridded later stacks ABOVE any
        # earlier tkraise()'d one (Tk sibling order), so the visible page would
        # end up the LAST tab while the rail highlights the first.  The first
        # tab is selected once, after ALL pages exist (_on_scan_ok).

    def _metadata_identity(self, cs: ConfigSheet) -> tuple[str, str | None] | None:
        """Key that decides whether two tabs share one ``info_devices.yaml``.

        ``(device_path, pcid)`` — device path is the *edited* root cell (the
        user may have browsed a different file on this tab → it detaches);
        pcid is the canonical probe id.  ``None`` when either is missing
        (no metadata, indeterminate name) — such a tab neither syncs nor
        receives.  Different file or different probe ⇒ independent configs.
        """
        path = cs.get_metadata_path().strip()
        if not path:
            return None
        stem = getattr(cs, "_page_stem", "") or ""
        pcid = format.to_pcid_from_name(format.stem_to_pcid(stem)) if stem else None
        return (path, pcid)

    def _on_metadata_changed(self, src: ConfigSheet) -> None:
        """Fan out src's metadata model to every same-identity peer.

        Peer's ``apply_metadata_setups`` rebuilds its subtree with programmatic
        refill semantics (clean, no echo).  Coef-only edits never reach here —
        ``_notify_metadata_changed``'s hash guard leaves them silent.
        """
        key = self._metadata_identity(src)
        if key is None:
            return
        # Fan-out must use edited cell values (from sheet), NOT src._setups —
        # the internal model is only updated on structural changes (splits),
        # while cell edits live in the tksheet until the next rebuild.
        edited = src.get_edited_metadata_setups()
        nums = [num for num, _ in (getattr(src, "_setups", None) or [])]
        setups = (
            [[nums[i] if i < len(nums) else i, copy.deepcopy(arr)] for i, arr in enumerate(edited)]
            if edited
            else []
        )
        # Propagate dirty flag: if source metadata is dirty, peers must be dirty too
        src_dirty = src.is_metadata_dirty()
        for stem, cs in self._pages.items():
            cs_key = self._metadata_identity(cs)
            if cs is src:
                continue
            if cs_key != key:
                continue
            try:
                cs.apply_metadata_setups(setups, dirty=src_dirty)
            except Exception:
                lf.exception("metadata sync: failed to apply to tab %s", stem)

    def _set_coefs_and_reload(self, stem: str, coefs_path: str) -> None:
        """Called from ConfigSheet when ``input.coefs`` path cell changes."""
        cs = self._pages.get(stem)
        if not cs or not coefs_path.strip():
            return
        # Page stem is a corrected-filename stem (e.g. "i_90"), not the canonical
        # pcid ("i90") — normalize before resolving the coefs table name
        tbl = format.pcid_to_raw_name(format.to_pcid_from_name(format.stem_to_pcid(stem)))
        try:
            coefs = incl_calc.coefs.get_coefs(coefs_path.split(","), tbl)
        except Exception:
            lf.exception("Failed to load coefficients from %s", coefs_path)
            return
        # get_coefs returns numpy arrays; cfg models the run YAML (plain lists)
        cs._cfg.setdefault("input", {}).setdefault("coefs", {})["path"] = coefs_path
        cs._cfg["input"]["coefs"].update(to_omegaconf.to_omegaconf_compatible_types(coefs))
        cs.load(cs._cfg, full=self._full_mode, config_root=schema.Config, return_enum=schema.Return)

    # ── §3 Run / Pause / Resume ─────────────────────────────────────

    def _update_run_btn_state(self) -> None:
        """Enable Run iff pages exist with valid input.path + date rows and no blocking calib."""
        ok = (
            bool(self._pages)
            and all(cs.is_path_valid() and cs.is_dates_valid() for cs in self._pages.values())
            and not any(cs.calib_blocking() for cs in self._pages.values())
        )
        self._run_btn.config(state="normal" if ok else "disabled")

    def _raise_overlays(self) -> None:
        """Root `<Configure>` stacking: Run above pages, open anchor list above Run.

        A blind Run lift covered a tall open dropdown after every resize —
        the list opens downward from the top row and can reach the floating
        Run button.
        """
        from contextlib import suppress

        with suppress(Exception):
            self._run_btn.lift()
        try:
            mt = self._path_field.sh.MT
            if mt.dropdown.open and mt.dropdown.window is not None:
                mt.dropdown.window.lift()
        except Exception:
            pass

    def _on_run(self) -> None:
        if self.wk.busy:
            gate = self.rt.pause_gate
            (gate.resume if gate.paused else gate.pause)()
            self._run_btn.config(text=_S["run_btn.resume"] if gate.paused else _S["run_btn.pause"])
            return
        stems = list(self._pages)
        if (
            not stems
            or not all(cs.is_path_valid() and cs.is_dates_valid() for cs in self._pages.values())
            or any(cs.calib_blocking() for cs in self._pages.values())
        ):
            return
        try:
            for s, cs in self._pages.items():
                self._write_coefs(s, cs)
            self._write_metadata()
        except Exception:
            lf.exception("Run pre-write failed")
            self._surface_error(
                __import__("sys").exc_info()[1] or Exception("pre-write failed"),
                _S.get("error.run", "Run: {p}"),
            )
            return
        self._clear_log()  # resets _error_active + hides tip
        self._cfg_detail = ""
        self._run_btn.config(text=_S["run_btn.pause"])
        self.rt.progress_bank.run_start(stems)
        self._overall_lbl.config(text=_S["status.starting"])
        self.wk.run(self._path_field.get(), stems)

    def _write_coefs(self, stem: str, cs: ConfigSheet) -> None:
        """Write sheet edits back to the run YAML — only if user changed something.

        ``input.path`` + ``input.coefs`` keep their dedicated write path (flat
        coefs contract of :func:`config_yaml.update_coefs_in_run_yaml`). Every
        other non-default leaf — ``input.calib``/``input.time_ranges`` (visible
        in simple mode too) plus ``out``/``filter``/``proc``/``program`` — comes
        from the generic :meth:`ConfigSheet.get_edited_full` (``sections=None``),
        merged by :func:`config_yaml.update_run_yaml`.
        """
        if not cs.is_dirty:
            return
        if not (yp := self._yaml_paths.get(stem)):
            return
        coefs, dates, path = cs._current_state()
        coefs_date = getattr(cs, "get_coefs_date", lambda: "")()
        coefs_node: dict = dict(coefs)
        if dates:
            coefs_node["dates"] = dates
        if coefs_date:
            coefs_node["date"] = coefs_date
        patch: dict = {}
        if path:
            patch.setdefault("input", {})["path"] = path
        if coefs_node:
            patch.setdefault("input", {})["coefs"] = coefs_node
        if callable(get_full := getattr(cs, "get_edited_full", None)):
            for sec, sub in (get_full(None) or {}).items():
                if sub:
                    patch.setdefault(sec, {}).update(sub)
        if not patch:
            cs.mark_clean()
            return
        config_yaml.update_run_yaml(yp, patch)
        cs.mark_clean()

    def _on_instant_apply(self, stem: str, kind: str) -> None:
        """Apply data-independent calib in-sheet (no YAML write — Run persists).

        Each trigger applies fully on its own and clears only its own cells:
        ``g0xyz`` computes ``Rz`` from sheet Ag/Cg/g0xyz; ``coordinates``
        shifts ``azimuth_shift_deg`` by magnetic declination; ``azimuth_add``
        adds its manual offset. Coefs stay dirty for Run pre-write.
        """
        from tcm_gui import _instant_calib as _ic

        cs = self._pages.get(stem)
        if cs is None:
            return
        tip = f"input.calib.{kind}"
        try:
            coefs = cs.get_edited_coefs()
            base = coefs.get("azimuth_shift_deg", 0) or 0
            if kind == "g0xyz":
                cells = cs.get_instant_cells("g0xyz")["g0xyz"]
                g0 = _ic.parse_g0xyz(cells)  # type: ignore[arg-type]
                if coefs.get("Ag") is None or coefs.get("Cg") is None:
                    raise KeyError("Ag/Cg coef rows absent")
                cs.write_instant_rz(_ic.rz_from_g0xyz(g0, coefs["Ag"], coefs["Cg"]))
                self._set_status(_S.get("instant.ok.rz", "Rz updated from g0xyz — saved on Run"))
            elif kind == "coordinates":
                cells = cs.get_instant_cells("coordinates")["coordinates"]
                coords = _ic.parse_coords(cells)  # type: ignore[arg-type]
                cs.write_instant_shift(_ic.shift_with_tuning(float(base), None, coords), kind)
                self._set_status(_S.get("instant.ok.coords", "azimuth_shift_deg updated — saved on Run"))
            else:
                add = _ic.parse_add(cs.get_instant_cells("azimuth_add")["azimuth_add"])  # type: ignore[arg-type]
                cs.write_instant_shift(_ic.shift_with_tuning(float(base), add, None), kind)
                self._set_status(_S.get("instant.ok.add", "azimuth_shift_deg updated — saved on Run"))
        except Exception as e:
            lf.exception("Instant apply failed")
            try:
                cs._sync_apply_boxes()
            except Exception:
                pass
            self._surface_error(e, _S.get("error.instant", "Instant apply: {p}"), tip_path=tip)

    def _device_anchors(self) -> list[Path]:
        """All anchor ``_raw`` dirs for the current path-field value.

        meta_finder discovery (multi-``_raw`` scan) with the single-anchor
        fallback; ``[]`` when the field is empty or discovery fails.
        """
        p = self._path_field.get().strip()
        if not p:
            return []
        p_path = Path(p).absolute()
        try:
            from tcm.anchors import _anchors_via_meta_finder

            if (mf := _anchors_via_meta_finder(p_path)) is not None:
                return mf
        except Exception:
            pass
        return [paths.find_dir_raw_absolute(p_path)]

    def _resolve_allowed_dir(self, p: str) -> str:
        """Root for bare-path auto-linking from the field value *p*.

        Device dir of the scanned anchors — their common parent for
        multi-device scans (narrowing to the first would drop the others);
        fallback ``paths.link_root`` — the anchor ``_raw``'s parent when the
        value points into one.
        """
        device_dirs = [a.parent for a in self._device_anchors()]
        root = (
            device_dirs[0]
            if len(device_dirs) == 1
            else paths.common_ancestor(device_dirs)
            if device_dirs
            else None
        )
        if root is None:
            root = paths.link_root(Path(p)) if p else Path()
        return str(root)

    def _allowed_dir(self) -> str:
        """Allowed dir at call time — cached per field value.

        Device_dir changes with re-browse / ``input.path`` edits, so the cache
        key is the field value itself; per-poll callers stay cheap.
        """
        p = self._path_field.get()
        if p != self._allowed_dir_key:
            self._allowed_dir_key = p
            self._allowed_dir_val = self._resolve_allowed_dir(p)
        return self._allowed_dir_val

    def _load_device_meta(self) -> tuple[dict | None, Path | None, Path | None]:
        """Resolve device dir + ``info_devices.yaml`` path + parsed content.

        Returns ``(device_meta, ddir, metadata_path)`` — all ``None`` when
        resolution fails.  Single source for scan + write paths (DRY).
        Handles parent-dir selection via filtered ``find_device_dirs``: when
        ``p`` is a cruise/parent dir containing ``_raw`` descendants, merges
        metadata from all anchors and returns the first anchor's dir/path for
        display/default-save.  ``path_field`` itself stays as user typed.
        """
        try:
            from meta_finder import io_info_files
            from meta_finder.config import DEVICES_FILE_NAME_YAML, DEVICES_FILE_NAME

            if anchors := self._device_anchors():
                p_path = Path(self._path_field.get().strip()).absolute()
                # anchors are _raw dirs; device_dir = parent
                merged: dict = {}
                first_ddir: Path | None = None
                first_cand: Path | None = None
                for anchor in anchors:
                    ddir = anchor.parent
                    if first_ddir is None:
                        first_ddir = ddir
                    for nm in (DEVICES_FILE_NAME_YAML, DEVICES_FILE_NAME):
                        cand = ddir / nm
                        if first_cand is None and ddir == first_ddir:
                            first_cand = cand
                        if cand.is_file():
                            try:
                                part = io_info_files.read_metadata_file(cand)
                            except Exception:
                                continue
                            # Merge per device, per station
                            for k, v in part.items():
                                if k not in merged:
                                    merged[k] = v
                                elif isinstance(v, dict) and isinstance(merged[k], dict):
                                    # Merge stations, preserve existing
                                    for sk, sv in v.items():
                                        if sk not in merged[k]:
                                            merged[k][sk] = sv
                            break
                if merged:
                    return merged, first_ddir, first_cand
                if first_ddir is not None:
                    if first_cand is None:
                        first_cand = first_ddir / DEVICES_FILE_NAME_YAML
                    return None, first_ddir, first_cand
                # Fallback single-anchor (legacy)
                ddir = paths.find_dir_raw_absolute(p_path).parent
                for nm in (DEVICES_FILE_NAME_YAML, DEVICES_FILE_NAME):
                    cand = ddir / nm
                    if cand.is_file():
                        return io_info_files.read_metadata_file(cand), ddir, cand
                return None, ddir, ddir / DEVICES_FILE_NAME_YAML
        except Exception:
            pass
        return None, None, None

    def _write_metadata(self) -> None:
        """Write edited ``metadata`` rows back to ``info_devices.yaml``.

        Collects per-stem 11-arrays from dirty ``metadata*`` nodes, merges into
        the device file via ``meta_finder`` (preserving other devices), and
        marks metadata clean on success.  Also writes autofilled metadata when
        the device file doesn't exist yet (dirty flag is True after load for
        autofilled metadata; the existence check covers edge cases).
        Frozen build includes ``meta_finder``
        (see ``pyproject.toml: tcm`` feature + ``tcm_gui.spec``).
        """
        # Resolve the target info_devices.yaml path(s) — parent-dir may have multiple anchors
        browsed: Path | None = next(
            (Path(cs.get_metadata_path()) for cs in self._pages.values() if cs.get_metadata_path().strip()),
            None,
        )
        # Per-anchor write when parent dir selected (multiple _raw)
        _pf = getattr(self, "_path_field", None)
        try:
            p_str = _pf.get().strip() if _pf and hasattr(_pf, "get") else ""
        except Exception:
            p_str = ""
        anchors: list[Path] = []
        if p_str and not browsed:
            try:
                from tcm.anchors import _anchors_via_meta_finder

                _mf = _anchors_via_meta_finder(Path(p_str).absolute())
                anchors = _mf if _mf is not None else []
            except Exception:
                anchors = []
        # Single-file browsed case — keep legacy single-file path
        if browsed is not None:
            device_dir, info_path = browsed.parent, browsed
            _, _, fallback = self._load_device_meta()
            existing_fallback: dict | None = None
            if fallback and fallback != info_path:
                try:
                    from meta_finder import io_info_files as _io2

                    if fallback.is_file():
                        existing_fallback = _io2.read_metadata_file(fallback)
                except Exception:
                    pass
            # Collect per-anchor if multiple anchors but browsed is single — treat as single
            if anchors and len(anchors) > 1:
                lf.debug("Browsed info file overrides multi-anchor parent — single write to {}", info_path)
            # Collect new_content for single file — dedup by actual VALUES so
            # synced tabs with identical metadata don't write duplicate setups.
            # Key: (pcid, station_id, tuple(array)) — only first occurrence wins.
            file_absent = not info_path.is_file()
            new_content_single: dict[str, dict[str, list]] = {}
            _seen: set[tuple[str, str, tuple]] = set()
            for stem, cs in self._pages.items():
                is_dirty = getattr(cs, "is_metadata_dirty", False) and cs.is_metadata_dirty()
                if not is_dirty and not file_absent:
                    continue
                try:
                    pcid = format.to_pcid_from_name(format.stem_to_pcid(stem))
                except Exception:
                    continue
                meta_map = cs.get_edited_metadata_map() if hasattr(cs, "get_edited_metadata_map") else {}
                if not meta_map:
                    continue
                for sid, arr in meta_map.items():
                    _key = (pcid, sid, tuple(arr) if arr else ())
                    if _key in _seen:
                        continue
                    _seen.add(_key)
                    new_content_single.setdefault(pcid, {})[sid] = arr
            if not new_content_single:
                return
            try:
                from meta_finder import io_info_files
                from meta_finder.create_info_files import _merge_device_metadata

                existing: dict = {}
                if existing_fallback is not None:
                    existing = dict(existing_fallback)
                if info_path.is_file():
                    try:
                        existing = io_info_files.read_metadata_file(info_path)
                    except Exception:
                        lf.warning("Failed to read %s — will overwrite", info_path, exc_info=True)
                # Merge: user-edited metadata overwrites non-placeholder; other devices preserved
                if existing:
                    merged = _merge_device_metadata(existing, new_content_single)
                    # Force dirty SIDs to user values (merge keeps existing non-placeholder)
                    for pcid, sids in new_content_single.items():
                        for sid, arr in sids.items():
                            # Ensure the merged entry reflects user's edited array
                            if pcid in merged and isinstance(merged[pcid], dict) and sid in merged[pcid]:
                                # Overwrite this sid with user's array (dirty wins)
                                merged[pcid][sid] = arr
                            elif pcid in merged:
                                # Fallback: set at top level
                                if isinstance(merged[pcid], dict):
                                    merged[pcid][sid] = arr
                                else:
                                    merged[pcid] = {sid: arr}
                            else:
                                merged[pcid] = {sid: arr}
                else:
                    merged = new_content_single
                # Write via atomic helper (tmp → move)
                io_info_files.write_metadata_file(device_dir, info_path, merged)
                lf.info("Wrote metadata for %s to %s", ", ".join(new_content_single), info_path.name)
                for cs in self._pages.values():
                    if getattr(cs, "is_metadata_dirty", False) and cs.is_metadata_dirty():
                        cs.mark_metadata_clean()
                    # Refresh ``metadata*`` → ``metadata`` label and clear red path validation
                    with __import__("contextlib").suppress(Exception):
                        cs._apply_metadata_dirty_label()
                        cs._apply_validations()  # path now exists → remove red fg
                        cs.sh.redraw()
            except Exception:
                lf.exception("Failed to write metadata to info_devices.yaml")
        else:
            # No browsed file — parent dir with single or multiple anchors
            # Use per-anchor device_dir derived from each stem's data file
            # Group new_content by anchor, dedup by VALUES so synced tabs
            # with identical metadata write only unique setup nodes.
            anchor_groups: dict[Path, dict[str, dict[str, list]]] = {}
            _seen: set[tuple[Path, str, str, tuple]] = set()
            for stem, cs in self._pages.items():
                is_dirty = getattr(cs, "is_metadata_dirty", False) and cs.is_metadata_dirty()
                # Also write autofilled when file absent
                # Determine anchor for this stem
                try:
                    pcid = format.to_pcid_from_name(format.stem_to_pcid(stem))
                except Exception:
                    continue
                # Find anchor via data file path
                try:
                    data_path_str = cs._cfg.get("input", {}).get("path", "") if hasattr(cs, "_cfg") else ""
                    anchor = paths.anchor_for_fs_path(Path(data_path_str).parent) if data_path_str else None
                    if anchor is None or not anchor.name.lower() == "_raw":
                        # Fallback to first anchor or find via lightweight
                        anchor = (
                            anchors[0]
                            if anchors
                            else paths.find_dir_raw_absolute(Path(p_str).absolute())
                            if p_str
                            else None
                        )
                    device_dir = anchor.parent if anchor and anchor.name.lower() == "_raw" else None
                except Exception:
                    device_dir = None
                if device_dir is None:
                    # Fallback single device_dir
                    _, device_dir, _ = self._load_device_meta()
                    if device_dir is None:
                        continue
                # Check dirty or file absent
                try:
                    from meta_finder.config import DEVICES_FILE_NAME_YAML as _Y

                    info_cand = device_dir / _Y
                    file_absent = not info_cand.is_file()
                except Exception:
                    file_absent = True
                if not is_dirty and not file_absent:
                    continue
                meta_map = cs.get_edited_metadata_map() if hasattr(cs, "get_edited_metadata_map") else {}
                if not meta_map:
                    continue
                # Dedup by actual VALUES so synced tabs with identical metadata
                # don't write duplicate setups. Key: (device_dir, pcid, sid, tuple(arr)).
                _grp = anchor_groups.setdefault(device_dir, {})
                for sid, arr in meta_map.items():
                    _key = (device_dir, pcid, sid, tuple(arr) if arr else ())
                    if _key in _seen:
                        continue
                    _seen.add(_key)
                    _grp.setdefault(pcid, {})[sid] = arr
            if not anchor_groups:
                return
            for device_dir, group_content in anchor_groups.items():
                try:
                    from meta_finder import io_info_files
                    from meta_finder.config import DEVICES_FILE_NAME_YAML as _Y
                    from meta_finder.create_info_files import _merge_device_metadata

                    info_path = device_dir / _Y
                    existing: dict = {}
                    if info_path.is_file():
                        try:
                            existing = io_info_files.read_metadata_file(info_path)
                        except Exception:
                            lf.warning("Failed to read %s — will overwrite", info_path, exc_info=True)
                    if existing:
                        merged = _merge_device_metadata(existing, group_content)
                        for pcid, sids in group_content.items():
                            for sid, arr in sids.items():
                                if pcid in merged and isinstance(merged[pcid], dict) and sid in merged[pcid]:
                                    merged[pcid][sid] = arr
                                elif pcid in merged and isinstance(merged[pcid], dict):
                                    merged[pcid][sid] = arr
                                else:
                                    merged[pcid] = {sid: arr}
                    else:
                        merged = group_content
                    io_info_files.write_metadata_file(device_dir, info_path, merged)
                    lf.info("Wrote metadata for %s to %s", ", ".join(group_content), info_path)
                    for cs in self._pages.values():
                        # Only mark clean for stems belonging to this device_dir
                        try:
                            dp = cs._cfg.get("input", {}).get("path", "") if hasattr(cs, "_cfg") else ""
                            ad = paths.anchor_for_fs_path(Path(dp).parent) if dp else None
                            dd = ad.parent if ad and ad.name.lower() == "_raw" else None
                            if (
                                dd == device_dir
                                and getattr(cs, "is_metadata_dirty", False)
                                and cs.is_metadata_dirty()
                            ):
                                cs.mark_metadata_clean()
                                with __import__("contextlib").suppress(Exception):
                                    cs._apply_metadata_dirty_label()
                                    cs._apply_validations()
                                    cs.sh.redraw()
                        except Exception:
                            pass
                except Exception:
                    lf.exception("Failed to write metadata to %s", device_dir)

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
        """Sync dirty indicator on rail cells — separate config vs metadata."""
        for stem, cs in self._pages.items():
            dc = cs.is_dirty
            dm = getattr(cs, "is_metadata_dirty", False) and cs.is_metadata_dirty()
            self._rail.set_dirty(stem, dc, dm)

    def _on_log_motion(self, event: tk.Event) -> None:
        """Show log status text while mouse is actively moving; fade on pause."""
        f1 = get_widget_meta(self._log, "f1_anchor")
        # Link hover wins over the row status — the status shows the full target
        # (files render shrunk in the log); kept while the pointer rests on the link.
        if url := self._log.link_at(event.x, event.y):
            self._log_link_hover = url
            self._cancel_dwell()
            self._set_status(link_display(url), raw=True)
        else:
            self._log_link_hover = ""
            # Log's F1 anchor (if any) — stored for both status and dwell.
            if (status := get_widget_meta(self._log, "status")) is not None:
                self._cancel_dwell()
                self._set_status(status, f1)
            # Store the log's F1 anchor (status label shows its help).
            if f1:
                self._status_lbl_f1_anchor = f1
        # Arm dwell with log tooltip (if defined) on first motion.
        if self._dwell_widget is not self._log:
            self._dwell_widget = self._log
            self._arm_dwell(_S.get("log.tooltip", ""), f1)
        # Reset the fade timer on every motion tick.
        if self._log_status_job is not None:
            self.root.after_cancel(self._log_status_job)
        self._log_status_job = self.root.after(self._log_status_fade_ms, self._on_log_status_fade)

    def _on_log_leave(self, _event: tk.Event) -> None:
        """Clear log status text + cancel dwell when pointer leaves."""
        if self._log_status_job is not None:
            self.root.after_cancel(self._log_status_job)
            self._log_status_job = None
        self._dwell_widget = None
        self._log_link_hover = ""
        self._cancel_dwell()
        self._set_status("", raw=True)

    def _on_log_status_fade(self) -> None:
        """Fade timer expired — clear the log status text unless a link is hovered."""
        self._log_status_job = None
        if not self._log_link_hover:
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
        if drain(self.rt.log_queue, self._log, self._allowed_dir()) and self._log_autoscroll:
            self._log.see("end")
        self._log.config(state="disabled")

    def _on_copy_rich(self, event: tk.Event) -> str | None:
        """Root-level ``<<Copy>>`` → rich-copy the selection of any text surface.

        ``_log`` is ``state='disabled'`` and ``_status_lbl`` never needs focus, so
        widget-scoped ``<<Copy>>`` bindings would never fire for them.  Tk's
        default ``<<Copy>>`` on Entry/Text copies plain text only — never RTF
        colors.  This handler runs at root level (fires via the toplevel
        bindtag whatever holds focus) after the focused widget's class binding
        (which already copied plain text).  When one of our text surfaces
        carries a ``sel`` tag — the event widget if it is one, else the first
        with a selection: serve RTF + plain via :func:`copy_rich` (overwriting
        the plain copy) and return ``'break'`` to suppress further propagation.
        Otherwise return ``None`` so the focused widget (e.g. ttk.Entry) keeps
        its normal copy behaviour.
        """
        surfaces = (self._log, self._status_lbl)
        focused = getattr(event, "widget", None)
        target = next((w for w in surfaces if focused is w), None) or next(
            (w for w in surfaces if w.tag_ranges("sel")), None
        )
        if target is not None:
            copy_rich(target)
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
            return fmt_status(_S.get("stage.composing_stem", "Composing {stem}\u2026"), stem=stem)
        return _S.get(desc, desc)

    @staticmethod
    def _translate_scan_stage(stage: ScanStage) -> str:
        """Translate ScanStage enum value (a ``scan_stage.*`` key) to current locale."""
        return _S.get(stage, str(stage))

    def _default_stage_text(self) -> str:
        """Return the appropriate default stage text based on _full_mode.

        Full mode: editable, no 'enable editing' hint.
        Non-full mode: readonly, shows 'provide a data path above to enable editing'.
        """
        key = "scan_stage.default_full" if self._full_mode else "scan_stage.default"
        return _S.get(key, "")

    def _set_cfg_ui_disabled(self, disabled: bool) -> None:
        """Inert look for rail + overall caption while no scanned configs exist.

        Simple mode only (full mode is editable before scan by design): the
        rail grays out and ignores clicks, the centered caption renders dim —
        same awaiting-a-path affordance as the readonly tksheet pages.
        """
        self._rail.set_disabled(disabled)
        self._overall_lbl.config(
            foreground=tcm_gui.theme.CELL_DEFAULT_VAL_FG if disabled else tcm_gui.theme.FG_DEFAULT
        )

    def _poll_progress(self) -> None:
        # Snapshot both states once — avoids redundant lock acquisitions.
        cur, tot, desc = self.rt.progress_stage.snapshot()
        # Progress advanced (or stage changed) → new information ends
        # hover-hide; the branches below re-show the row.
        if (cur, tot, desc) != self._stage_last:
            self._stage_last = cur, tot, desc
            self._stage_hovering = self._status_hovering = False
        _cur_o, tot_o, desc_o = self.rt.progress_overall.snapshot()
        if tot > 0:
            self._prog_stage.config(maximum=tot, value=cur)
            # Only update stage text while the stage runs AND no error is
            # surfaced — _surface_error appends the error to the current stage
            # text; _poll_progress must not clobber it on the next 300 ms tick.
            if cur < tot and not self._error_active:
                self._prog_stage_text.config(text=self._translate_desc(desc) or "")
            # Delayed show avoids flashing the row for very short operations.
            if not self._stage_hovering and not self._stage_shown and self._prog_show_job is None:
                self._prog_show_job = self.root.after(400, self._show_prog_stage)
        else:
            # Cancel pending show if progress ended before the delay.
            if self._prog_show_job is not None:
                self.root.after_cancel(self._prog_show_job)
                self._prog_show_job = None
            # Don't hide during initial scan — _fit_status_font showed it.
            # Keep the row while an error is surfaced so the localized error
            # text stays on screen until the next ok scan/run clears it.
            if not self._initial_scan and not self._error_active and self._stage_shown:
                self._hide_stage_progress()
            # Stage inactive: consume a one-shot clear signal (set at each
            # probe start via progress_stage.clear_and_reset) so stale text is
            # wiped exactly once; otherwise leave _status alone — explicit
            # setters own it ("Ready", "Done …", hover hints).
            if not self._any_hovering and self.rt.progress_stage.consume_clear():
                self._set_status("", raw=True)
        if tot_o > 0:
            # desc_o carries ScanStage i18n keys — translate like stage descs.
            self._overall_lbl.config(text=self._translate_desc(desc_o) or "")
        elif self._cfg_state == ScanStage.DEFAULT:
            # In DEFAULT state, use mode-specific text (full vs non-full).
            self._overall_lbl.config(text=self._default_stage_text())
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

    def _show_prog_stage(self) -> None:
        """Delayed show of the stage progress row — skipped while hover-hidden."""
        self._prog_show_job = None
        if self._stage_hovering:
            return
        cur, tot, _desc = self.rt.progress_stage.snapshot()
        if tot > 0:
            self._prog_stage.config(maximum=tot, value=cur)
            self._show_stage_progress()

    def _poll_results(self) -> None:
        try:
            kind, payload = self.rt.result_queue.get_nowait()
        except Empty:
            return
        {
            "scan_ok": self._on_scan_ok,
            "scan_error": self._on_scan_error,
            "scan_list": self._on_scan_list,
            "run_ok": self._on_run_done,
            "run_error": self._on_run_error,
        }[kind](payload)

    def _on_scan_list(self, payload) -> None:
        """Parent with multiple _raw anchors — list-fill only, no meta_finder/processing.

        Shows anchors count and auto-selects first anchor for tab-fill (no cfg_proc at parent).
        Keeps parent trigger visible in log before field replacement.
        """
        try:
            # Unpack (parent, anchors) or legacy [anchors]
            if isinstance(payload, tuple) and len(payload) == 2 and isinstance(payload[1], list):
                parent, anchors = payload  # type: ignore[assignment]
            else:
                parent, anchors = None, payload  # type: ignore[assignment]
            # Reset any forced Run→Pause from the lightweight list scan
            if getattr(self, "_run_forced_during_scan", False):
                try:
                    self._run_btn.config(state="disabled", text=_S["run_btn.text"])
                except Exception:
                    pass
                self._run_forced_during_scan = False
            if not anchors:
                self._on_scan_error(RuntimeError("No _raw anchors found"))
                return
            # Log parent trigger explicitly before field replacement — addresses "no logged B:\\Cruises\\BalticSea"
            if parent:
                lf.info("Scan list for parent trigger=%s → %s anchors", parent, len(anchors))
            lf.info(
                "Found %s _raw anchors (list-fill, no processing): %s",
                len(anchors),
                ", ".join(str(a) for a in anchors[:5]) + (" …" if len(anchors) > 5 else ""),
            )
            self._overall_lbl.config(text=f"Found {len(anchors)} anchors — loading first…")
            self._set_status(f"Found {len(anchors)} anchors, loading {anchors[0].parent.name}…", raw=True)
            # Auto-select first anchor for tab-fill (no meta_finder device dirs discovery)
            first_anchor = anchors[0]
            if parent:
                lf.info("Parent %s → auto-select first anchor %s for tab-fill", parent, first_anchor)
            # Feed the inherent path-cell dropdown; the number appears in _path_lbl
            try:
                self._anchor_dropdown.set_paths([str(a) for a in anchors])
                # Clickable caption only when a pick list exists (2+ anchors).
                self._path_lbl.configure(cursor="hand2" if self._anchor_dropdown.has_list else "")
            except Exception:
                pass
            # Update path_field to the anchor (so subsequent scans/tabs use anchor, not parent)
            try:
                self._path_field.set(str(first_anchor))
                self._anchor_dropdown.refresh(str(first_anchor))
            except Exception:
                pass
            # Trigger tab-fill scan for the single anchor (this will call processing.run with eager=False stubs)
            self.wk.scan(self._original_argv, str(first_anchor))
            # Re-enable Run→Pause for the upcoming tab-fill scan
            try:
                self._run_btn.config(state="normal", text=_S["run_btn.pause"])
                self._run_forced_during_scan = True
            except Exception:
                pass
        except Exception as exc:
            self._on_scan_error(exc)

    def _on_anchor_dropdown_select(self, path: str) -> None:
        """Inherent path-cell dropdown commit — rescan the picked anchor for tab-fill."""
        sel = (path or "").strip()
        if not sel:
            return
        try:
            self._path_field.set(sel)
            self._anchor_dropdown.refresh(sel)
        except Exception:
            pass
        lf.info("Anchor selected: %s", sel)
        self._overall_lbl.config(text=f"Loading {Path(sel).parent.name}…")
        self._set_status(f"Loading {Path(sel).parent.name}…", raw=True)
        self.wk.scan(self._original_argv, sel)
        try:
            self._run_btn.config(state="normal", text=_S["run_btn.pause"])
            self._run_forced_during_scan = True
        except Exception:
            pass

    def _on_scan_error(self, exc: BaseException) -> None:
        self._path_field.set_error(True)  # failed search — red fg on the search path
        self._surface_error(exc, _S["error.scan"], tip_path="path_field")
        # Reset state to DEFAULT and clear progress so _poll_progress shows default text.
        self._cfg_state = ScanStage.DEFAULT
        self._cfg_detail = ""
        self.rt.progress_overall.set(0, 0, "")
        self._overall_lbl.config(text=self._default_stage_text())
        # No configs from ANY successful scan (placeholder never enters
        # _yaml_paths) → back to the inert look until a valid path.
        if not self._full_mode and not self._yaml_paths:
            self._set_cfg_ui_disabled(True)
        # Scan failed after we forced Run→Pause for pausing — disable again
        if getattr(self, "_run_forced_during_scan", False):
            try:
                self._run_btn.config(state="disabled", text=_S["run_btn.text"])
            except Exception:
                pass
            self._run_forced_during_scan = False
        else:
            try:
                self._update_run_btn_state()
            except Exception:
                pass

    def _on_run_error(self, exc: BaseException) -> None:
        self._run_btn.config(text=_S["run_btn.text"])
        self._surface_error(exc, _S["error.run"])

    def _surface_error(self, exc: BaseException, log_prefix: str, *, tip_path: str | None = None) -> None:
        """Common error surface: log line, separator, floater text, detail tip.

        Sets ``_error_active`` so ``_poll_progress`` keeps the floater on screen
        until a fresh ok scan/run clears it.  The independently-logged exception
        (worker's ``lf.exception``) is already in ``_log``; here we add the
        localized prefix line + the markdown detail block rendered in
        ``_status_lbl`` (bottom-left overlay) via :meth:`_show_tip`.
        The dwell tip shows only for an explicit *tip_path* (scan → ``path_field``;
        instant-apply → the trigger's own doc path) or else file errors (``OSError`` →
        ``path_field`` help)
        """
        self._error_active = True
        short = f"{type(exc).__name__}: {exc}"
        self._log_err(fmt_status(log_prefix, p=short))
        # Append the error to the current stage text (e.g. "Генерация конфигураций"
        # → "Генерация конфигураций\nError \"FileNotFoundError: …\".") so the
        # user sees both the stage that failed and the error details.
        current = self._prog_stage_text.cget("text")
        error = self._stage_error_text(exc)
        self._prog_stage_text.config(text=f"{current}\n{error}" if current else error)
        self._show_stage_progress()
        tip_src = tip_path or ("path_field" if isinstance(exc, OSError) else None)
        if tip_src and (tip := help_general_for_path(tip_src)):
            f1 = e.anchor if (e := help_for_path(tip_src)) else None
            self._show_tip(tip, f1)
        else:
            self._hide_tip()

    @staticmethod
    def _stage_error_text(exc: BaseException) -> str:
        short = f"{type(exc).__name__}: {exc}"
        return fmt_status(_S.get("stage.error", 'Error "{msg}".'), msg=short)

    def _set_status(self, text: str, anchor: str | None = None, *, raw: bool = False) -> None:
        """Debounced status switch/close — applied by :meth:`_apply_status` after
        :attr:`_STATUS_SETTLE_MS`.

        All chrome-hover, poll, and config-cell callers route through here so a
        quick pointer pass does not flicker the label; the latest text wins.
        ``_show_tip`` / ``_show_dwell_tip`` write to ``_status_lbl`` directly and
        cancel the pending job, bypassing this debounce.  *anchor* is the F1 anchor
        for the text — applied together at display time so it always matches.
        """
        self._cancel_status_job()
        self._status_job = self.root.after(
            self._STATUS_SETTLE_MS,
            lambda t=text, a=anchor, r=raw: self._apply_status(t, anchor=a, raw=r),
        )

    def _cancel_status_job(self) -> None:
        """Cancel a pending debounced status apply."""
        if self._status_job is not None:
            self.root.after_cancel(self._status_job)
            self._status_job = None

    def _apply_status(self, text: str, anchor: str | None = None, *, raw: bool = False) -> None:
        """Fire after :attr:`_STATUS_SETTLE_MS` — error tooltip and a reader on
        the status label win.  While a dwell tooltip owns the label, ANY switch
        (row move on the sheet, leave) first waits out the linger window
        :attr:`_DWELL_HIDE_MS` — the tip stays readable / clickable; the new
        text is re-applied right after the clear.  A pending dwell arm from the
        new row stays untouched.  ``base`` resolves relative markdown links in
        config-reference-derived texts (mode bodies) against the doc directory."""
        self._status_job = None
        if self._tip_active or self._status_hovering:
            return
        if self._dwell_active:
            self._cancel_dwell_hide_job()
            self._dwell_hide_job = self.root.after(self._DWELL_HIDE_MS, self._clear_dwell_now)
            self._dwell_widget = None
            self._cancel_status_job()
            self._status_job = self.root.after(
                self._DWELL_HIDE_MS,
                lambda t=text, a=anchor, r=raw: self._apply_status(t, anchor=a, raw=r),
            )
            return
        self._status_lbl.set_text(text, raw=raw, base=doc_path(resolve_lang()))
        self._status_lbl_f1_anchor = anchor

    # ── status-label hover — hold the dwell tip while reading / clicking ─────

    def _on_status_enter(self, _event: tk.Event | None = None) -> None:
        """Pointer entered the status text — pause the dwell auto-close."""
        self._status_hovering = True
        self._cancel_dwell_hide_job()

    def _show_link_hover(self, url: str) -> None:
        """Link hover row — the full target below the status text ('' removes it).

        Files render shrunk to the file name in both widgets, so the hover row
        is what reveals the full path.
        """
        self._status_lbl.set_hover_line(link_display(url) if url else "")

    def _on_status_leave(self, _event: tk.Event | None = None) -> None:
        """Pointer left the status text — restart the linger countdown."""
        self._status_hovering = False
        if self._dwell_active:
            self._cancel_dwell_hide_job()
            self._dwell_hide_job = self.root.after(self._DWELL_HIDE_MS, self._clear_dwell_now)

    def _show_tip(self, text: str, anchor: str | None = None) -> None:
        """Show error detail tooltip in ``_status_lbl``; suppress status updates.

        While ``_tip_active`` is True, :meth:`_set_status` is a no-op so hover
        hints and poll-driven status cannot overwrite the tooltip.  Dismissed
        by :meth:`_hide_tip` (new scan/run, path change, Esc, or cell edit).
        Also clears any active dwell tip (error takes precedence).
        """
        self._clear_dwell_now()  # error takes precedence over dwell
        self._cancel_status_job()  # error tip shows immediately — drop pending switch
        self._tip_active = True
        # Relative links in the body resolve against config_reference_*.md.
        self._status_lbl.set_text(text, base=doc_path(resolve_lang()))
        if anchor is not None:
            self._status_lbl_f1_anchor = anchor

    def _hide_tip(self) -> None:
        """Dismiss the error tooltip; resume normal status updates.

        Also clears any active dwell tip — both are tooltip overlays in
        ``_status_lbl`` and share the same dismissal triggers (Esc, new
        scan/run, path change, cell edit begin).

        Releases the status-hover hold unconditionally (before the no-op
        guard): dismissing the overlay empties the label, which collapses to
        ~0 width under a stationary pointer — Tk emits no ``<Leave>`` for a
        widget shrinking beneath a cursor, so a stale ``_status_hovering``
        would keep ``_apply_status`` blocked even if only an empty tip was
        dismissed (regression: Esc after any detail dwell froze the normal
        status while detailed tips kept working).
        """
        self._status_hovering = False
        was_active = self._tip_active or self._dwell_active
        if not was_active:
            return
        self._tip_active = False
        self._dwell_active = False
        self._dwell_widget = None
        self._cancel_dwell_job()
        self._cancel_dwell_hide_job()
        self._status_lbl.set_text("", raw=True)
        # Clear the stored F1 anchor — status label no longer shows row help.
        self._status_lbl_f1_anchor = None

    # ── dwell tooltip ────────────────────────────────────────────────

    def _arm_dwell(self, text: str, anchor: str | None = None) -> None:
        """Schedule dwell tooltip — show *text* in ``_status_lbl`` after :attr:`_DWELL_MS`.

        Only armed when *text* is non-empty.  Cancels any pending dwell job
        first (widget changed or re-entry).  The actual show is done by
        :meth:`_show_dwell_tip` which fires from the ``after()`` callback.
        *anchor* is frozen via the ``after`` lambda so F1 resolves the section
        for what is actually shown when the tip fires.
        """
        self._cancel_dwell_job()
        if not text or self._tip_active:
            return
        self._dwell_job = self.root.after(self._DWELL_MS, lambda: self._show_dwell_tip(text, anchor))

    def _cancel_dwell_job(self) -> None:
        """Cancel a pending dwell ``after()`` job (does NOT clear an active tip)."""
        if self._dwell_job is not None:
            self.root.after_cancel(self._dwell_job)
            self._dwell_job = None

    def _cancel_dwell_hide_job(self) -> None:
        """Cancel a pending dwell auto-close ``after()`` job."""
        if self._dwell_hide_job is not None:
            self.root.after_cancel(self._dwell_hide_job)
            self._dwell_hide_job = None

    def _cancel_dwell(self, *, force: bool = False) -> None:
        """Soft-dismiss: cancel the pending arm; an ACTIVE tip lingers
        :attr:`_DWELL_HIDE_MS` (reading / clicking links) before
        :meth:`_clear_dwell_now` clears it.  A re-show (:meth:`_show_dwell_tip`)
        or a non-empty status replacement cancels the pending clear.

        When the pointer is within the dwelling widget's hierarchy (the
        widget itself, an ancestor frame, or the root background), the dwell
        is NOT cancelled — the tip stays so the user can keep reading.
        *force* bypasses this guard (Esc key)."""
        if not force and self._pointer_in_dwell_hierarchy():
            self._cancel_dwell_hide_job()  # drop any pending linger — tip stays
            return
        self._cancel_dwell_job()
        if self._dwell_active:
            self._cancel_dwell_hide_job()
            self._dwell_hide_job = self.root.after(self._DWELL_HIDE_MS, self._clear_dwell_now)
        self._dwell_widget = None

    def _clear_dwell_now(self) -> None:
        """Hard-dismiss linger + active tip; preserve pending dwell arm.

        Via ``_dwell_hide_job`` after A→B move: ``_dwell_job`` is single-slot
        current arm — cancelling kills B's dwell (A linger → B never fires
        until re-hover, reported bug). Stale impossible: next hover cancels,
        error absorbed by ``_show_dwell_tip``'s ``_tip_active`` guard. True
        hard-dismiss still via :meth:`_hide_tip`/:meth:`_cancel_dwell`.
        Also releases stale ``_status_hovering`` like :meth:`_hide_tip`
        (label collapses under pointer, no ``<Leave>``).
        """
        self._cancel_dwell_hide_job()
        if self._dwell_active:
            self._dwell_active = False
            self._status_hovering = False
            self._status_lbl.set_text("", raw=True)
            # Clear the stored F1 anchor — status label no longer shows row help.
            self._status_lbl_f1_anchor = None
        self._dwell_widget = None

    def _show_dwell_tip(self, text: str, anchor: str | None = None) -> None:
        """Fire after :attr:`_DWELL_MS` — render *text* in ``_status_lbl``.

        Suppressed while ``_tip_active`` (error tooltip takes precedence).
        Stays while hovered: dismissal triggers (leave / widget switch) only
        schedule the clear via :meth:`_cancel_dwell` — the tip lingers
        :attr:`_DWELL_HIDE_MS` so the user can keep reading or click a link.
        """
        self._dwell_job = None
        if self._tip_active:
            return
        self._dwell_active = True
        self._cancel_status_job()  # dwell takes the label — drop the pending switch
        self._cancel_dwell_hide_job()  # a deferred clear must not kill the re-shown tip
        # Relative links in the body resolve against config_reference_*.md.
        self._status_lbl.set_text(text, base=doc_path(resolve_lang()))
        if anchor is not None:
            self._status_lbl_f1_anchor = anchor

    def _on_cell_status(self, cs: ConfigSheet, msg: str, md: bool = False) -> None:
        """ConfigSheet hover callback — debounced status switch + arm dwell.

        Row switch cancels the pending dwell arm from the previous row; the
        debounced :meth:`_apply_status` replaces a still-active dwell tooltip,
        so a stale tip never outlives its row.  Anchor is forged at display
        time — passed to :meth:`_set_status` / :meth:`_arm_dwell` so it matches
        what is actually shown.
        """
        self._cancel_dwell_job()
        a = cs._f1_anchor_for_iid(cs._status_iid) if cs._status_iid is not None else None
        self._set_status(msg, a, raw=not md)
        if detail := getattr(cs, "_hover_detail", ""):
            self._arm_dwell(detail, a)

    def _on_scan_ok(self, result) -> None:
        if not result or len(result) < 4:
            return
        # Burst GET is metadata, not input — scan returns it for display only (write on Run)
        sync_result: dict | None = None
        bursts: dict[str, tuple[Any, Any]] | None = None
        if len(result) >= 6 and isinstance(result[4], dict) and isinstance(result[5], dict):
            # 5th = sync_result (dict with status), 6th = bursts {stem: (bdt,bst)}
            sync_result, bursts = result[4], result[5]
        elif len(result) >= 5 and isinstance(result[4], dict):
            sample = next(iter(result[4].values()), None) if result[4] else None
            if isinstance(sample, tuple) and len(sample) == 2:
                bursts = result[4]
            else:
                sync_result = result[4]
                if len(result) >= 6 and isinstance(result[5], dict):
                    bursts = result[5]
        # Preload device metadata for ``metadata`` node (frozen build includes meta_finder).
        # DRY: device-dir + file lookup via App helper (also used by _write_metadata).
        device_meta, ddir, metadata_path = self._load_device_meta()

        self._error_active = False
        self._path_field.set_error(False)
        self._hide_tip()
        for frame in self._tab_of.values():
            frame.destroy()
        self._pages.clear()
        self._yaml_paths.clear()
        self._tab_of.clear()
        self._rail.clear()
        self._set_cfg_ui_disabled(False)  # scanned configs exist — active from first paint
        self._current = None

        for stem, yp, cfg_dc in result[3]:
            cfg = OmegaConf.to_container(cfg_dc, resolve=True)
            prog = cfg.get("program")
            if prog and prog.get("return_") == schema.Return.CFG_FROM_ARGS:
                prog["return_"] = str(schema.Return.END)
            self._yaml_paths[stem] = Path(yp)
            md = reload_tabs.meta_for_stem(stem, device_meta)
            # Burst GET autofill for display (scan) — missing metadata only, no file write yet
            burst_filled = False
            if bursts and stem in bursts:
                bdt, bst = bursts[stem]

                def _is_ph(v: Any) -> bool:
                    return v in ("?", "-", "", None)

                if md is None:
                    md = [[0, [None] * 11]]
                    tr = cfg.get("input", {}).get("time_ranges") or []
                    if len(tr) >= 2:
                        md[0][1][6], md[0][1][7] = tr[0], tr[1]
                    md[0][1][8], md[0][1][9] = bdt, bst
                    burst_filled = True
                else:
                    arr = md[0][1]
                    if len(arr) < 11:
                        arr = list(arr) + [None] * (11 - len(arr))
                        md[0][1] = arr
                    cur_bdt = arr[8] if len(arr) > 8 else None
                    cur_bst = arr[9] if len(arr) > 9 else None
                    if _is_ph(cur_bdt) or _is_ph(cur_bst):
                        if str(cur_bdt) != str(bdt) or str(cur_bst) != str(bst):
                            arr[8], arr[9] = bdt, bst
                            burst_filled = True
            ss = sync_result.get(stem) if sync_result else None
            self._add_page(
                stem,
                cfg,
                yaml_path=Path(yp),
                metadata=md,
                sync_status=ss,
                metadata_path=str(metadata_path) if metadata_path else None,
            )
            # Mark burst-autofilled sheet dirty so Run persists it to info_devices.yaml
            if burst_filled:
                cs = self._pages.get(stem)
                if cs is not None:
                    cs._metadata_unsaved = True
                    with __import__("contextlib").suppress(Exception):
                        cs._apply_metadata_dirty_label()
                        cs._apply_validations()
                        cs.sh.redraw()
                    # Ensure rail shows metadata dirty (config stays clean)
                    self._rail.set_dirty(stem, False, True)
        if self._tab_of:  # top tab gets rail indicator + the raised (visible) page
            self._select_tab(next(iter(self._tab_of)))
        self._cfg_state = ScanStage.DONE
        self._cfg_detail = ""
        self._initial_scan = False
        # Clear scan progress so the stage row collapses on next poll.
        self.rt.progress_stage.set(0, 0, "")
        # Scan done — show "Ready" now.
        self._set_status(_S["status.ready"], raw=True)
        self._overall_lbl.config(text=self._translate_scan_stage(self._cfg_state))
        try:
            self._run_btn.config(text=_S["run_btn.text"])
        except Exception:
            pass
        self._update_run_btn_state()
        self._run_forced_during_scan = False

    def _on_run_done(self, result) -> None:
        self._error_active = False
        self._hide_tip()
        self._run_btn.config(text=_S["run_btn.text"])
        processed, failed = result[0], result[1]
        n = len(processed) + len(failed)
        pct = round(100 * len(processed) / n) if n else 100
        # Collapse the stage progress row — completion shown in overall label
        self._hide_stage_progress()
        self.rt.progress_stage.set(0, 0, "")
        self._cfg_state = ScanStage.DONE
        self._cfg_detail = fmt_status(_S["overall_lbl.done_detail"], pct=pct, ok=len(processed), n=n)
        self._overall_lbl.config(text=f"{self._translate_scan_stage(self._cfg_state)}{self._cfg_detail}")
        self.rt.progress_overall.set(0, 0, "")
        reload_tabs.reload_tabs_after_run(self, processed or [])

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

    def _report_tk_exception(self, exc, val, tb) -> None:
        """Tkinter callback exception → logging → ``_log`` (and console/file handlers).

        Tkinter passes the already-caught ``(exc, val, tb)`` — we are NOT inside
        an active exception, so ``lf.exception`` (implicit ``sys.exc_info()``)
        would lose the traceback: embed it in the message instead.
        Also surfaces in the GUI log panel via ``_surface_error`` so every
        Tk callback failure is visible to the user without needing the console.
        """
        msg = "".join(traceback.format_exception(exc, val, tb))
        lf.error("Exception in Tkinter callback\n%s", msg)
        # Mirror to GUI log + status tip — general, no per-call try/except needed.
        try:
            self._surface_error(val, _S.get("error.run", "Run: {p}"))
        except Exception:
            pass

    def _log_err(self, msg: str, *, separator: bool = False) -> None:
        """Append ``msg`` as an ``error``-tagged line; optionally add a visual
        separator (``info`` tag) tying the error to the markdown detail block."""
        self._log.config(state="normal")
        self._log.insert_linked(f"{msg}\n", "error", self._allowed_dir())
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
    # Windowed processes (pythonw / PyInstaller --noconsole) have no console:
    # sys.stdout/stderr are None and ANY stream write raises
    # AttributeError: 'NoneType' object has no attribute 'write'.  Worst
    # offender: hydra's JobReturn.return_value writes to sys.stderr before
    # re-raising a *failed* job's exception — the AttributeError raised there
    # replaces the real error, making scan failures unexplainable.  Devnull
    # sinks keep such writes harmless so real exceptions propagate (rendered
    # with tracebacks into the GUI log, see log_bridge.drain).
    sys.stdout = sys.stdout or open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115 — process-lifetime sink
    sys.stderr = sys.stderr or open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115 — process-lifetime sink
    App(argv).run()
