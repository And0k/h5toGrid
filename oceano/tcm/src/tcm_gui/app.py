"""Tk root: layout §1–5, 300 ms polling, event wiring."""

from __future__ import annotations


from collections.abc import Sequence
import ctypes
import sys
import tkinter as tk
from pathlib import Path, PurePath
from queue import Empty
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

from omegaconf import OmegaConf

from tcm import cli, config, config_yaml, format, incl_calc
from tcm_gui.cli_cfg import default_cfg

from ._browse_button import BrowseButtonManager
from ._path_field import PathField
from ._rtf_clipboard import copy_rich
from .coef_sheet import ConfigSheet
from .const import (
    FUNC_COLOR,
    TAG_COLORS,
    apply_ui_scale,
    get_widget_meta,
    set_widget_meta,
)
from .log_bridge import drain, install
from .runtime import Runtime
from .worker import Worker


def _shift_at_startup() -> bool:
    try:
        return bool(ctypes.windll.user32.GetAsyncKeyState(0x10) & 0x8000)
    except (AttributeError, OSError):
        return False


class App:
    APP_ID = "Vendor.Product"  # todo: Fix, not hardcode here
    POLL = 300  # ms

    def __init__(self, argv: Sequence[str] | None = None) -> None:
        if sys.platform == "win32":
            shell32.SetCurrentProcessExplicitAppUserModelID(self.APP_ID)

        self.root = tk.Tk()
        apply_ui_scale(self.root)  # global DPI + named fonts — before any widget
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

        # Configuration label state — drives _cfg_lbl caption transitions
        self._cfg_scanned = False  # True after first successful scan
        self._cfg_was_dirty = False  # True while any page has unsaved edits

        # watch Shift globally on root (Windows doesn't send Shift to widgets)
        self._full_mode = _shift_at_startup()

        self._build()
        # Prefill path entry from CLI args — only when user explicitly provided one.
        # parse_data_path returns None when no positional arg is found (e.g.
        # ``python -m tcm_gui`` without a data path), so the GUI starts empty.
        path_in, _ = cli.parse_data_path(self._original_argv)
        if path_in is not None:
            self._path_field.set(str(path_in))
            self.root.after(100, self._scan)
        else:
            # No CLI path — show a placeholder page so the notebook isn't empty.
            self._add_page("(default)", default_cfg())
        self._poll()

    # ── layout ──────────────────────────────────────────────────────

    def _build(self) -> None:
        r = self.root
        r.grid_rowconfigure(2, weight=2)  # notebook
        r.grid_rowconfigure(4, weight=1)  # log
        r.grid_columnconfigure(0, weight=1)

        # §1 input.path
        f0 = ttk.Frame(r)
        f0.grid(row=0, column=0, sticky="ew", padx=4, pady=2)
        f0.columnconfigure(1, weight=1)
        self._path_lbl = ttk.Label(f0, text="Data search path")
        self._path_lbl.grid(row=0, column=0, padx=(0, 4))
        set_widget_meta(self._path_lbl, tooltip="Path field label")
        self._path_field = PathField(f0, on_commit=self._on_path_changed)
        self._path_field.grid(row=0, column=1, sticky="ew")
        set_widget_meta(self._path_field, status=self._HOVER_MSG, tooltip="Data search path")
        # Status message on hover — rebind on the Sheet's MT canvas
        self._path_hovering = False
        self._path_field.sh.MT.bind("<Enter>", lambda _: self._on_path_hover_in(), add="+")
        self._path_field.sh.MT.bind("<Leave>", lambda _: self._on_path_hover_out(), add="+")

        # §2 Configuration status label — sits between path field and notebook tabs
        self._cfg_state = tk.StringVar(value="Default configuration")
        self._cfg_lbl = ttk.Label(r, textvariable=self._cfg_state)
        self._cfg_lbl.grid(row=1, column=0, sticky="w", padx=8, pady=(0, 0))
        set_widget_meta(self._cfg_lbl, status="Current configuration state")

        # §3 Notebook — full width, below the configuration label
        self.nb = ttk.Notebook(r)
        self.nb.grid(row=2, column=0, sticky="nsew", padx=4, pady=(0, 2))

        # §4 Run + overall progress (config-level)
        f2 = ttk.Frame(r)
        f2.grid(row=3, column=0, sticky="ew", padx=4, pady=2)
        f2.columnconfigure(2, weight=1)
        self._run_btn = ttk.Button(f2, text="Run", command=self._on_run)
        self._run_btn.grid(row=0, column=0, padx=(0, 4))
        set_widget_meta(self._run_btn, status="Start / pause / resume processing", tooltip="Run button")
        set_widget_meta(f2, status="Run controls and overall progress")
        self._prog_all_lbl = tk.StringVar(value="")
        ttk.Label(f2, textvariable=self._prog_all_lbl).grid(row=0, column=1, padx=(0, 4))
        self._prog_all = ttk.Progressbar(f2, mode="determinate")
        self._prog_all.grid(row=0, column=2, sticky="ew")

        # §5 Log
        self._log = ScrolledText(r, height=10, state="disabled", wrap="word")
        self._log.grid(row=4, column=0, sticky="nsew", padx=4, pady=2)
        for lvl, clr in TAG_COLORS.items():
            self._log.tag_configure(lvl, foreground=clr)
        self._log.tag_configure("func", foreground=FUNC_COLOR)
        self._log.bind("<Control-c>", lambda _: (copy_rich(self._log), "break")[1])

        # §6 Status bar
        f4 = ttk.Frame(r)
        f4.grid(row=5, column=0, sticky="ew", padx=4, pady=2)
        f4.columnconfigure(1, weight=1)
        self._status = tk.StringVar(value="Ready")
        self._status_lbl = ttk.Label(f4, textvariable=self._status)
        self._status_lbl.grid(row=0, column=0, padx=(0, 4))
        set_widget_meta(self._status_lbl, status="Application status messages")
        self._prog_stage = ttk.Progressbar(f4, mode="determinate")
        self._prog_stage.grid(row=0, column=1, sticky="ew")

    # ── §1 entry hover status message ────────────────────────────────

    # Canonical hover hint — also stored in widget_meta by _build() for the
    # centralized registry (future tooltip popups read the same value).
    _HOVER_MSG = "Changing data path rescans and resets all config tabs below"

    def _on_path_hover_in(self) -> None:
        """Mouse enters Entry — show status hint from widget_meta registry."""
        self._path_hovering = True
        self._status.set(get_widget_meta(self._path_field, "status", self._HOVER_MSG))

    def _on_path_hover_out(self) -> None:
        """Mouse leaves Entry — clear hover flag (status restored by poll)."""
        self._path_hovering = False

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
        """PathField committed a new path — trigger scan."""
        self._scan()

    def _scan(self) -> None:
        if self._path_field.get().strip():
            self._clear_log()
            self.wk.scan(self._original_argv)

    # ── §2 page management ──────────────────────────────────────────

    def _add_page(self, stem: str, cfg: dict) -> None:
        frame = ttk.Frame(self.nb)
        self.nb.add(frame, text=stem)
        self._tab_of[stem] = frame
        set_widget_meta(frame, status=f"Configuration tab: {stem}")

        cs = ConfigSheet(frame)
        cs.sh.pack(fill="both", expand=True, padx=2, pady=2)
        cs._mgr = BrowseButtonManager(
            cs.sh,
            on_path_changed=lambda path: self._set_coefs_and_reload(stem, path),
            on_edit_restyler=cs._apply_edit_value,
        )
        cs.load(cfg, full=self._full_mode, config_root=config.Config, return_enum=config.Return)
        cs.on_hover_status = self._status.set
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
        cs.load(cs._cfg, full=self._full_mode, config_root=config.Config, return_enum=config.Return)

    # ── §3 Run / Pause / Resume ─────────────────────────────────────

    def _on_run(self) -> None:
        if self.wk.busy:
            gate = self.rt.pause_gate
            (gate.resume if gate.paused else gate.pause)()
            self._run_btn.config(text="Pause" if gate.paused else "Resume")
            return
        stems = list(self._pages)
        if not stems:
            return
        for s, cs in self._pages.items():
            self._write_coefs(s, cs)
        self._clear_log()
        self._run_btn.config(text="Pause")
        # Show a sliver on overall bar immediately — before the first stage tick
        self._prog_all.config(value=0, maximum=1)
        self._prog_all_lbl.set("Starting…")
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
        self._poll_dirty_tabs()
        self._poll_logs()
        self._poll_progress()
        self._poll_results()
        self.root.after(self.POLL, self._poll)

    def _poll_dirty_tabs(self) -> None:
        """Append/remove '*' on tab titles to reflect unsaved edits; update cfg label."""
        any_dirty = False
        for stem, cs in self._pages.items():
            if cs.is_dirty:
                any_dirty = True
            if (frame := self._tab_of.get(stem)) is None:
                continue
            current = self.nb.tab(frame, "text")
            desired = f"{stem}*" if cs.is_dirty else stem
            if current != desired:
                self.nb.tab(frame, text=desired)
        # Transition: any dirty → "Processing configurations"; all clean → restore scan caption
        if any_dirty and not self._cfg_was_dirty:
            self._cfg_was_dirty = True
            self._cfg_state.set("Processing configurations")
        elif not any_dirty and self._cfg_was_dirty:
            self._cfg_was_dirty = False
            if self._cfg_scanned:
                self._cfg_state.set("Generated configuration for processing found data")

    def _poll_logs(self) -> None:
        at_bottom = self._log.yview()[1] > 0.99  # до вставки
        self._log.config(state="normal")
        if drain(self.rt.log_queue, self._log) and at_bottom:
            self._log.see("end")  # только если был внизу
        self._log.config(state="disabled")

    def _poll_progress(self) -> None:
        # Snapshot both states once — avoids redundant lock acquisitions.
        cur, tot, desc = self.rt.progress_stage.snapshot()
        cur_o, tot_o, desc_o = self.rt.progress_overall.snapshot()
        if tot > 0:
            self._prog_stage.config(maximum=tot, value=cur)
            if not self._path_hovering:
                self._status.set(desc or "")
        else:
            self._prog_stage.config(value=0)
            # Stage inactive: consume a one-shot clear signal (set at each
            # probe start via progress_stage.clear_and_reset) so stale text is
            # wiped exactly once; otherwise leave _status alone — explicit
            # setters own it ("Ready", "Done …", hover hints).
            if not self._path_hovering and self.rt.progress_stage.consume_clear():
                self._status.set("")
        if tot_o > 0:
            self._prog_all.config(maximum=tot_o, value=cur_o)
            self._prog_all_lbl.set(desc_o or "")
        else:
            self._prog_all.config(value=0)  # null bar when inactive
            self._prog_all_lbl.set("")  # clear label too

    def _poll_results(self) -> None:
        try:
            kind, payload = self.rt.result_queue.get_nowait()
        except Empty:
            return
        {
            "scan_ok": self._on_scan_ok,
            "scan_error": lambda p: self._log_err(f"Scan: {p}"),
            "run_ok": self._on_run_done,
            "run_error": lambda p: (self._log_err(f"Run: {p}"), self._run_btn.config(text="Run")),
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
            if prog and prog.get("return_") == config.Return.CFG_FROM_ARGS:
                prog["return_"] = str(config.Return.END)
            self._yaml_paths[stem] = Path(yp)
            self._add_page(stem, cfg)
        self._cfg_scanned = True
        self._cfg_was_dirty = False
        self._cfg_state.set("Generated configuration for processing found data")

    def _on_run_done(self, result) -> None:
        self._run_btn.config(text="Run")
        processed, failed = result[0], result[1]
        n = len(processed) + len(failed)
        pct = round(100 * len(processed) / n) if n else 100
        self._status.set(f"Done — {pct}% ({len(processed)}/{n} ok)")
        # Reset both progress bars on completion
        self._prog_stage.config(value=0)
        self._prog_all.config(value=0)
        self._prog_all_lbl.set("")
        self.rt.progress_overall.set(0, 0, "")
        self.rt.progress_stage.set(0, 0, "")

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

    shell32.SetCurrentProcessExplicitAppUserModelID.argtypes = (
        ctypes.wintypes.LPCWSTR,
    )
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
