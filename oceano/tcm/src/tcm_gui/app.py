"""Tk root: layout §1–5, 300 ms polling, event wiring."""

from __future__ import annotations

import ctypes
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

# Chrome i18n strings — loaded via const.load_str() (auto-detects OS locale).
from .const import load_str

STR: dict[str, str] = load_str()

from ._browse_button import BrowseButtonManager, _is_shift_pressed
from ._help import help_for_path
from ._path_field import PathField
from ._rtf_clipboard import copy_rich
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
        self.root.title("TCM")
        self.root.geometry("1100x800")

        # use exe icon
        self._icons: tuple[int, ...] = ()
        self.root.bind("<Destroy>", self._free_icons, add=True)
        self.set_window_icon()

        self.rt = Runtime()
        # Install the QueueHandler on the root logger once, for the lifetime
        # of the app, so log calls from the GUI main thread (e.g.
        # ``_reload_coefs`` triggered by treeview/cell interactions) reach
        # ``rt.log_queue`` → ScrolledText.  Worker's ``_wrap.wrapped``
        # re-attaches it after Hydra's ``dictConfig`` replaces root handlers,
        # so worker-thread logs also reach the queue.
        self.rt.queue_handler = install(self.rt.log_queue, self.rt.pause_gate)
        self.wk = Worker(self.rt)
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
            self._add_page("(default)", default_cfg())
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
        self._path_lbl = ttk.Label(f0, text=STR["path_lbl.tooltip"])
        self._path_lbl.grid(row=0, column=0, padx=(0, 4))
        self._path_field = PathField(
            f0,
            on_commit=self._on_path_changed,
            on_status=self._on_browse_status,
            status_hint=STR["browse_btn.status"],
        )
        self._path_field.grid(row=0, column=1, sticky="ew")
        # Status message on hover — rebind on the Sheet's MT canvas
        self._path_hovering = False
        self._path_field.sh.MT.bind("<Enter>", lambda _: self._on_path_hover_in(), add="+")
        self._path_field.sh.MT.bind("<Leave>", lambda _: self._on_path_hover_out(), add="+")

        # §2 Overall status label + progress bar (dual-purpose: scan state / progress)
        f1 = ttk.Frame(r)
        f1.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 0))
        f1.columnconfigure(0, weight=1)  # label stretches; bar is fixed-width
        self._overall_lbl = ttk.Label(f1, text=self._cfg_state)
        self._overall_lbl.grid(row=0, column=0, sticky="w")
        self._prog_all = ttk.Progressbar(f1, mode="determinate", length=220)
        # _prog_all grid-managed on demand by _poll_progress (hidden by default)

        # §3 Notebook — full width, below the configuration label
        self.nb = ttk.Notebook(r)
        self.nb.grid(row=2, column=0, sticky="nsew", padx=4, pady=(0, 2))
        # Tab hover: show config-file status when pointer is over a tab label.
        # Tab frames store their status in widget_meta (set by _add_page);
        # _on_nb_motion reads it for the tab under the pointer.
        self.nb.bind("<Motion>", self._on_nb_motion, add="+")
        self.nb.bind("<Leave>", self._on_nb_leave, add="+")

        # §4 Run button — floats at notebook bottom-right, parented on root for z-order
        self._run_btn = ttk.Button(r, text=STR["run_btn.text"], command=self._on_run)
        self._run_btn.place(in_=self.nb, relx=1.0, rely=1.0, anchor="se", x=-24, y=-24)
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
        for lvl, clr in tcm_gui.theme.TAG_COLORS.items():
            self._log.tag_configure(lvl, foreground=clr)
        self._log.tag_configure("func", foreground=tcm_gui.theme.FUNC_COLOR)
        # ``Ctrl+C`` is bound at the ROOT level (not on ``_log``): ``_log`` is
        # ``state='disabled'`` so it can never take keyboard focus, meaning a
        # widget-scoped ``<Control-c>`` binding would never fire and the user
        # would get Tk's default ``<<Copy>>`` (plain text only) — never RTF
        # colors.  The root handler checks for a ``_log`` selection first; if
        # present it serialises colored RTF via :func:`copy_rich` and returns
        # ``'break'`` to suppress the default.  Otherwise it falls through so
        # the focused widget (e.g. ``_path_field`` ttk.Entry) keeps normal copy.
        self.root.bind("<Control-c>", self._on_copy_rich, add="+")

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

        # Z-order: mouse motion → GUI status on top.
        r.bind("<Motion>", lambda _: self._status_lbl.lift(), add="+")

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
        """
        # Widgets with their own hover handling — skip to avoid conflicts.
        _skip = {self._path_field, self.nb}
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
            self._status_lbl.set_text(status)

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
            tooltip = STR.get(f"{role}.tooltip")
            status: object
            if role == "run_btn":
                # Dynamic: reflected busy / paused at hover time, not registration.
                status = self._run_btn_status
            else:
                status = STR.get(f"{role}.status")
            if tooltip is None and status is None:
                continue
            kwargs: dict[str, object] = {}
            if tooltip is not None:
                kwargs["tooltip"] = tooltip
            if status is not None:
                kwargs["status"] = status
            set_widget_meta(w, **kwargs)

    def _run_btn_status(self) -> str:
        """Dynamic Run button status: reflects busy / paused state, read live."""
        if not self.wk.busy:
            return STR["run.start"]
        return STR["run.resume"] if self.rt.pause_gate.paused else STR["run.pause"]

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
            if self._initial_scan and not self._prog_floater.winfo_ismapped():
                self._prog_floater.place(relx=1.0, rely=1.0, anchor="se", x=-8, y=-4)
                self._prog_stage_text.config(text=STR["status.loading"])
            elif not self._initial_scan:
                self._status_lbl.set_text(STR["status.ready"], raw=True)

    def _on_path_hover_in(self) -> None:
        """Mouse enters Entry — show status hint from widget_meta registry."""
        self._path_hovering = True
        self._status_lbl.set_text(get_widget_meta(self._path_field, "status"))

    def _on_path_hover_out(self) -> None:
        """Mouse leaves Entry — clear hover flag (status restored by poll)."""
        self._path_hovering = False

    def _on_browse_status(self, text: str) -> None:
        """Browse button hover: show Shift hint, guard against poll clobber."""
        self._browse_hovering = bool(text)
        self._status_lbl.set_text(text)

    # ── §3 notebook tab hover ────────────────────────────────────────

    _nb_hovering: bool = False  # guards against poll clobbering tab status

    def _on_nb_motion(self, event: tk.Event) -> None:
        """Pointer over notebook — show config-file status when over a tab label.

        ``identify(x, y)`` returns ``"tab"`` (top edge of tab) or ``"label"``
        (deeper in tab strip) when the pointer is over a tab; ``"client"`` or
        ``""`` otherwise.  The 3-arg Tcl form ``identify tab x y`` returns the
        integer tab index directly.
        """
        try:
            tab_idx = self.nb.tk.call(str(self.nb), "identify", "tab", event.x, event.y)
        except (tk.TclError, AttributeError):
            return
        # 3-arg form returns int index or "" when not over a tab.
        if tab_idx == "" or tab_idx is None:
            if self._nb_hovering:
                self._nb_hovering = False
            return
        tabs = self.nb.tabs()
        tab_idx_int = int(tab_idx)
        if tab_idx_int >= len(tabs):
            return
        frame = self.nb.nametowidget(tabs[tab_idx_int])
        status = get_widget_meta(frame, "status")
        if not self._nb_hovering:
            self._nb_hovering = True
        self._status_lbl.set_text(status, raw=True)

    def _on_nb_leave(self, _event: tk.Event) -> None:
        """Pointer left the notebook — clear tab hover flag."""
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
        self._initial_scan = True
        # Show progress overlay immediately (skip "Ready" → "Loading…" transition).
        self._status_lbl.set_text("", raw=True)
        if self._prog_floater.winfo_ismapped():
            self._prog_floater.lift()
        else:
            self._prog_floater.place(relx=1.0, rely=1.0, anchor="se", x=-8, y=-4)
        self._prog_stage_text.config(text=STR["status.loading"])
        self._scan()

    def _scan(self) -> None:
        path = self._path_field.get().strip()
        if path:
            self._clear_log()
            # Live path field — not the stale startup argv — drives the scan,
            # so a GUI browse selection of ``_raw`` rescans that directory.
            self.wk.scan(self._original_argv, path)

    # ── §2 page management ──────────────────────────────────────────

    def _add_page(self, stem: str, cfg: dict, yaml_path: Path | None = None) -> None:
        frame = ttk.Frame(self.nb)
        self.nb.add(frame, text=stem)
        self._tab_of[stem] = frame
        # Dynamic widgets (per-config tabs) live outside App attr names, so the
        # autorole loop in _register_chrome_help can't see them — bind here.
        # Format: "Configuration cfg_proc/run/stem.yaml" relative to the
        # ``_raw`` anchor that ``processing.run`` actually uses — NOT the raw
        # field text, which may point *into* ``_raw`` (e.g. ``_raw/<cruise>``)
        # while configs live directly under ``_raw/cfg_proc/``. Mismatch makes
        # ``relative_to`` raise → fallback to bare filename (loss of context).
        if yaml_path is not None and (field := self._path_field.get().strip()):
            anchor = paths.find_dir_raw_absolute(Path(field).absolute())
            try:
                rel = Path(yaml_path).relative_to(anchor)
                status_text = STR["tab.status"].format(path=rel)
            except ValueError:
                status_text = STR["tab.status"].format(path=yaml_path.name)
        else:
            status_text = STR["tab.status"].format(path=stem)
        set_widget_meta(frame, status=status_text)

        cs = ConfigSheet(frame, status_hint=STR["browse_btn.status"])
        cs.sh.pack(fill="both", expand=True, padx=2, pady=2)
        cs._mgr = BrowseButtonManager(
            cs.sh,
            on_path_changed=lambda path: self._set_coefs_and_reload(stem, path),
            on_edit_restyler=cs._apply_edit_value,
            on_status=self._on_browse_status,
            status_hint=STR["browse_btn.status"],
        )
        cs.load(cfg, full=self._full_mode, config_root=schema.Config, return_enum=schema.Return)
        cs.on_hover_status = lambda msg, md=False: self._status_lbl.set_text(msg, raw=not md)
        cs._empty_area_hint = STR["empty_area.synced" if yaml_path is not None else "empty_area.unsaved"]
        self._pages[stem] = cs

    def _set_coefs_and_reload(self, stem: str, coefs_path: str) -> None:
        """Called from ConfigSheet when ``input.coefs_path`` cell changes."""
        cs = self._pages.get(stem)
        if not cs or not coefs_path.strip():
            return
        tbl = format.pcid_to_raw_name(format.stem_to_pcid(stem))
        coefs = incl_calc.coefs.get_coefs(coefs_path.split(","), tbl)
        cs._cfg.setdefault("input", {})["coefs"] = coefs
        cs._cfg.setdefault("input", {})["coefs_path"] = coefs_path
        cs.load(cs._cfg, full=self._full_mode, config_root=schema.Config, return_enum=schema.Return)

    # ── §3 Run / Pause / Resume ─────────────────────────────────────

    def _on_run(self) -> None:
        if self.wk.busy:
            gate = self.rt.pause_gate
            (gate.resume if gate.paused else gate.pause)()
            self._run_btn.config(text=STR["run_btn.resume"] if gate.paused else STR["run_btn.pause"])
            return
        stems = list(self._pages)
        if not stems:
            return
        for s, cs in self._pages.items():
            self._write_coefs(s, cs)
        self._clear_log()
        self._cfg_detail = ""
        self._run_btn.config(text=STR["run_btn.pause"])
        # Show a sliver on overall bar immediately — before the first stage tick
        self._prog_all.config(value=0, maximum=1)
        if not self._prog_all.winfo_ismapped():
            self._prog_all.grid(row=0, column=1, sticky="e", padx=(8, 0))
        self._overall_lbl.config(text=STR["status.starting"])
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
        """Append/remove '*' on tab titles to reflect unsaved edits."""
        for stem, cs in self._pages.items():
            if (frame := self._tab_of.get(stem)) is None:
                continue
            current = self.nb.tab(frame, "text")
            desired = f"{stem}*" if cs.is_dirty else stem
            if current != desired:
                self.nb.tab(frame, text=desired)

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
        """Root-level ``<Control-c>`` → copy ``_log`` selection as RTF, else fall through.

        ``_log`` is ``state='disabled'`` and so cannot receive keyboard focus, so
        a widget-scoped ``<Control-c>`` binding would never fire (the user would
        see Tk's default ``<<Copy>>`` — plain text only — never RTF colors).
        This handler runs at root level so the binding fires regardless of which
        widget has focus.  When ``_log`` carries a non-empty ``sel`` tag (mouse
        drag on the disabled text): serve RTF + plain via :func:`copy_rich` and
        return ``'break'`` to suppress the default ``<<Copy>>`` propagation.
        Otherwise return ``None`` so the focused widget (e.g. ttk.Entry) keeps
        its normal copy behaviour.
        """
        if self._log.tag_ranges("sel"):
            copy_rich(self._log)
            return "break"
        return None

    def _poll_progress(self) -> None:
        # Snapshot both states once — avoids redundant lock acquisitions.
        cur, tot, desc = self.rt.progress_stage.snapshot()
        cur_o, tot_o, desc_o = self.rt.progress_overall.snapshot()
        if tot > 0:
            self._prog_stage.config(maximum=tot, value=cur)
            # Only update stage text while stage is running; keep "Done" message at 100%
            if cur < tot:
                self._prog_stage_text.config(text=desc or "")
            if self._prog_floater.winfo_ismapped():
                self._prog_floater.lift()
            elif self._prog_show_job is None:
                # Delayed show — avoids flashing for very short operations.
                self._prog_show_job = self.root.after(400, self._show_prog_floater)
        else:
            # Cancel pending show if progress ended before delay.
            if self._prog_show_job is not None:
                self.root.after_cancel(self._prog_show_job)
                self._prog_show_job = None
            # Don't hide during initial scan — _fit_status_font showed it.
            if not self._initial_scan and self._prog_floater.winfo_ismapped():
                self._prog_floater.place_forget()
            # Stage inactive: consume a one-shot clear signal (set at each
            # probe start via progress_stage.clear_and_reset) so stale text is
            # wiped exactly once; otherwise leave _status alone — explicit
            # setters own it ("Ready", "Done …", hover hints).
            if not self._any_hovering and self.rt.progress_stage.consume_clear():
                self._status_lbl.set_text("", raw=True)
        if tot_o > 0:
            self._prog_all.config(maximum=tot_o, value=cur_o)
            if not self._prog_all.winfo_ismapped():
                self._prog_all.grid(row=0, column=1, sticky="e", padx=(8, 0))
            self._overall_lbl.config(text=desc_o or "")
        else:
            if self._prog_all.winfo_ismapped():
                self._prog_all.grid_remove()
            self._prog_all.config(value=0)
            self._overall_lbl.config(text=f"{self._cfg_state}{self._cfg_detail}")

    def _show_prog_floater(self) -> None:
        """Delayed show of stage progress overlay."""
        self._prog_show_job = None
        cur, tot, _desc = self.rt.progress_stage.snapshot()
        if tot > 0:
            self._prog_stage.config(maximum=tot, value=cur)
            self._prog_floater.place(relx=1.0, rely=1.0, anchor="se", x=-8, y=-4)
            self._prog_floater.lift()

    def _poll_results(self) -> None:
        try:
            kind, payload = self.rt.result_queue.get_nowait()
        except Empty:
            return
        {
            "scan_ok": self._on_scan_ok,
            "scan_error": lambda p: self._log_err(STR["error.scan"].format(p=p)),
            "run_ok": self._on_run_done,
            "run_error": lambda p: (
                self._log_err(STR["error.run"].format(p=p)),
                self._run_btn.config(text=STR["run_btn.text"]),
            ),
        }[kind](payload)

    def _on_scan_ok(self, result) -> None:
        if not result or len(result) < 4:
            return

        for tab in self.nb.tabs():
            self.nb.forget(tab)
        self._pages.clear()
        self._yaml_paths.clear()
        self._tab_of.clear()
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
        self._status_lbl.set_text(STR["status.ready"], raw=True)
        self._overall_lbl.config(text=self._cfg_state)

    def _on_run_done(self, result) -> None:
        self._run_btn.config(text=STR["run_btn.text"])
        processed, failed = result[0], result[1]
        n = len(processed) + len(failed)
        pct = round(100 * len(processed) / n) if n else 100
        # Hide stage progress floater — completion shown in overall label
        if self._prog_floater.winfo_ismapped():
            self._prog_floater.place_forget()
        self.rt.progress_stage.set(0, 0, "")
        # Reset overall progress bar; show completion in overall label
        if self._prog_all.winfo_ismapped():
            self._prog_all.grid_remove()
        self._prog_all.config(value=0)
        self._cfg_state = ScanStage.DONE
        self._cfg_detail = STR["overall_lbl.done_detail"].format(pct=pct, ok=len(processed), n=n)
        self._overall_lbl.config(text=f"{self._cfg_state}{self._cfg_detail}")
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

    def _log_err(self, msg: str) -> None:
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
