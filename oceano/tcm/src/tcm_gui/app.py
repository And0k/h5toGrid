"""Tk root: layout §1–5, 300 ms polling, event wiring."""

from __future__ import annotations

import ctypes
import sys
import tkinter as tk
from pathlib import Path, PurePath
from queue import Empty
from tkinter import filedialog, ttk
from tkinter.scrolledtext import ScrolledText

from omegaconf import OmegaConf

from tcm import cli, config, config_yaml, format, incl_calc

from .coef_sheet import ConfigSheet
from .log_bridge import FUNC_COLOR, TAG_COLORS, drain
from .runtime import Runtime
from .worker import Worker


def _default_cfg() -> dict:
    """Return a plain dict with all ``config.ConfigIn_InclProc`` defaults wrapped as ``input`` section."""
    return {"input": OmegaConf.to_container(OmegaConf.structured(config.ConfigIn_InclProc()), resolve=True)}

def _shift_at_startup() -> bool:
    try:
        return bool(ctypes.windll.user32.GetAsyncKeyState(0x10) & 0x8000)
    except Exception:
        return False


class App:
    POLL = 300  # ms

    def __init__(self, argv: list[str] | None = None) -> None:
        self.root = tk.Tk()
        self.root.title("TCM")
        self.root.geometry("1100x800")
        self.rt = Runtime()
        self.wk = Worker(self.rt)
        self._pages: dict[str, ConfigSheet] = {}
        self._yaml_paths: dict[str, Path] = {}
        self._tab_of: dict[str, ttk.Frame] = {}  # stem → notebook tab frame
        # Store original argv — Worker passes it to call_in_raw_dir which
        # extracts the data path via cli.parse_data_path(sys.argv) internally.
        self._original_argv = list(argv or sys.argv)

        # watch Shift globally on root (Windows doesn't send Shift to widgets)
        self._shift_held = False
        self._full_mode = _shift_at_startup()
        self._browse_btns: list[ttk.Button] = []
        for ks in ("Shift_L", "Shift_R"):
            self.root.bind(f"<KeyPress-{ks}>", lambda _: self._set_shift(True))
            self.root.bind(f"<KeyRelease-{ks}>", lambda _: self._set_shift(False))

        self._build()
        self._add_page("(default)", _default_cfg())
        # Prefill path entry from CLI args (same extraction as call_in_raw_dir)
        path_in, _ = cli.parse_data_path(self._original_argv)
        if path_in:
            self._path_var.set(str(path_in))
            self.root.after(100, self._scan)
        self._poll()

    # ── layout ──────────────────────────────────────────────────────

    def _build(self) -> None:
        r = self.root
        r.grid_rowconfigure(1, weight=2)
        r.grid_rowconfigure(3, weight=1)
        r.grid_columnconfigure(0, weight=1)

        # §1 input.path
        f0 = ttk.Frame(r)
        f0.grid(row=0, column=0, sticky="ew", padx=4, pady=2)
        f0.columnconfigure(0, weight=1)
        self._path_var = tk.StringVar()
        e = ttk.Entry(f0, textvariable=self._path_var)
        e.grid(row=0, column=0, sticky="ew")
        e.bind("<Return>", lambda _: self._scan())
        e.bind("<FocusOut>", lambda _: self._scan())

        self._mk_browse(f0, self._on_browse_input).grid(row=0, column=1, padx=(4, 0))

        # §2 Notebook
        self.nb = ttk.Notebook(r)
        self.nb.grid(row=1, column=0, sticky="nsew", padx=4, pady=2)

        # §3 Run + overall progress (config-level)
        f2 = ttk.Frame(r)
        f2.grid(row=2, column=0, sticky="ew", padx=4, pady=2)
        f2.columnconfigure(2, weight=1)
        self._run_btn = ttk.Button(f2, text="Run", command=self._on_run)
        self._run_btn.grid(row=0, column=0, padx=(0, 4))
        self._prog_all_lbl = tk.StringVar(value="")
        ttk.Label(f2, textvariable=self._prog_all_lbl).grid(row=0, column=1, padx=(0, 4))
        self._prog_all = ttk.Progressbar(f2, mode="determinate")
        self._prog_all.grid(row=0, column=2, sticky="ew")

        # §4 Log
        self._log = ScrolledText(r, height=10, state="disabled", wrap="word")
        self._log.grid(row=3, column=0, sticky="nsew", padx=4, pady=2)
        for lvl, clr in TAG_COLORS.items():
            self._log.tag_configure(lvl, foreground=clr)
        self._log.tag_configure("func", foreground=FUNC_COLOR)

        # §5 Status bar
        f4 = ttk.Frame(r)
        f4.grid(row=4, column=0, sticky="ew", padx=4, pady=2)
        f4.columnconfigure(1, weight=1)
        self._status = tk.StringVar(value="Ready")
        ttk.Label(f4, textvariable=self._status).grid(row=0, column=0, padx=(0, 4))
        self._prog_stage = ttk.Progressbar(f4, mode="determinate")
        self._prog_stage.grid(row=0, column=1, sticky="ew")

    # ── §1 callbacks ────────────────────────────────────────────────

    def _set_shift(self, held: bool) -> None:
        self._shift_held = held
        txt = "Files…" if held else "Dir…"
        for b in self._browse_btns:
            try:
                b.config(text=txt)
            except tk.TclError:
                pass  # кнопка уничтожена


    def _mk_browse(self, parent, cmd) -> ttk.Button:
        """Единая фабрика Browse-кнопок: регистрация + текст по Shift."""
        b = ttk.Button(parent, text="Files…" if self._shift_held else "Dir…")
        self._browse_btns.append(b)
        b.bind("<Button-1>", lambda e: cmd())
        return b


    def _on_browse_input(self) -> None:
        if self._shift_held:
            if paths := filedialog.askopenfilenames(title="Data files"):
                self._path_var.set(self._fmt_multi(paths))
                self._scan()
        else:
            if d := filedialog.askdirectory(title="Data directory"):
                self._path_var.set(d)
                self._scan()


    def _browse_input(self, files: bool = False) -> None:
        if files:
            if paths := filedialog.askopenfilenames(title="Data files"):
                self._path_var.set(self._fmt_multi(paths))
                self._scan()
        else:
            if d := filedialog.askdirectory(title="Data directory"):
                self._path_var.set(d)
                self._scan()

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


    def _scan(self) -> None:
        if self._path_var.get().strip():
            self._clear_log()
            self.wk.scan(self._original_argv)

    # ── §2 page management ──────────────────────────────────────────

    def _add_page(self, stem: str, cfg: dict) -> None:
        frame = ttk.Frame(self.nb)
        self.nb.add(frame, text=stem)
        self._tab_of[stem] = frame
        # §2.1 coefs_path
        fcp = ttk.Frame(frame)
        fcp.pack(fill="x", padx=2, pady=2)
        fcp.columnconfigure(0, weight=1)
        cp_var = tk.StringVar(value=cfg.get("input", {}).get("coefs_path", ""))
        cpe = ttk.Entry(fcp, textvariable=cp_var)
        cpe.grid(row=0, column=0, sticky="ew")
        cpe.bind("<Return>", lambda ev: self._reload_coefs(stem, cp_var))
        cpe.bind("<FocusOut>", lambda _: self._reload_coefs(stem, cp_var))
        self._mk_browse(fcp, lambda: self._on_browse_coefs(stem, cp_var)) \
            .grid(row=0, column=1, padx=(4, 0))

        # §2.2 tksheet
        cs = ConfigSheet(frame)
        cs.sh.pack(fill="both", expand=True, padx=2, pady=2)
        cs.load(cfg, full=self._full_mode)
        self._pages[stem] = cs

    def _on_browse_coefs(self, stem: str, cp_var: tk.StringVar) -> None:
        if self._shift_held:
            if paths := filedialog.askopenfilenames(
                title="Coefficient files", filetypes=[("Coefs", "*.h5 *.nc *.yaml *.yml"), ("All", "*.*")]
            ):
                cp_var.set(",".join(paths))
        else:
            if d := filedialog.askdirectory(title="Coefficients directory"):
                cp_var.set(d)
        self._reload_coefs(stem, cp_var)

    def _browse_coefs(self, stem: str, cp_var: tk.StringVar, files: bool) -> None:
        if files:
            if paths := filedialog.askopenfilenames(
                    title="Coefficient files",
                    filetypes=[("Coefs", "*.h5 *.nc *.yaml *.yml"), ("All", "*.*")]
            ):
                cp_var.set(",".join(paths))
        else:
            if d := filedialog.askdirectory(title="Coefficients directory"):
                cp_var.set(d)
        self._reload_coefs(stem, cp_var)


    def _reload_coefs(self, stem: str, cp_var: tk.StringVar) -> None:
        cs = self._pages.get(stem)
        if not cs or not (p := cp_var.get().strip()):
            return
        tbl = format.pcid_to_raw_name(format.stem_to_pcid(stem))
        cfg = cs._cfg
        cfg.setdefault("input", {})["coefs"] = incl_calc.coefs.get_coefs(p.split(","), tbl)
        cs.load(cfg, full=self._full_mode)

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
        self.wk.run(self._path_var.get(), stems)

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
        """Append/remove '*' on tab titles to reflect unsaved edits."""
        for stem, cs in self._pages.items():
            if (frame := self._tab_of.get(stem)) is None:
                continue
            current = self.nb.tab(frame, "text")
            desired = f"{stem}*" if cs.is_dirty else stem
            if current != desired:
                self.nb.tab(frame, text=desired)

    def _poll_logs(self) -> None:
        at_bottom = self._log.yview()[1] > 0.99   # до вставки
        self._log.config(state="normal")
        if drain(self.rt.log_queue, self._log) and at_bottom:
            self._log.see("end")                   # только если был внизу
        self._log.config(state="disabled")


    def _poll_progress(self) -> None:
        cur, tot, desc = self.rt.progress_stage.snapshot()
        if tot > 0:
            self._prog_stage.config(maximum=tot, value=cur)
            self._status.set(desc or f"{cur}/{tot}")
        cur_o, tot_o, desc_o = self.rt.progress_overall.snapshot()
        if tot_o > 0:
            self._prog_all.config(maximum=tot_o, value=cur_o)
            # Composite label: config index/total + current stage desc
            self._prog_all_lbl.set(f"{cur_o}/{tot_o}  {desc_o}")
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
            self._yaml_paths[stem] = Path(yp)
            self._add_page(stem, OmegaConf.to_container(cfg_dc, resolve=True))

    def _on_run_done(self, result) -> None:
        self._run_btn.config(text="Run")
        processed, failed = result[0], result[1]
        n = len(processed) + len(failed)
        pct = round(100 * len(processed) / n) if n else 100
        self._status.set(f"Done — {pct}% ({len(processed)}/{n} ok)")
        self._prog_stage.config(value=0)
        self.rt.progress_overall.set(0, 0, "")  # null upper bar on completion

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


def main(argv: list[str] | None = None) -> None:
    App(argv).run()
