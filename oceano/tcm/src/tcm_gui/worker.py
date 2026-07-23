"""Worker thread: pipeline stages via tcm.cli.call_in_raw_dir in background."""
from __future__ import annotations
import logging
import sys
import threading
from functools import wraps

from .runtime import Runtime
from .log_bridge import install as install_qh
from .progress_bridge import GuiTqdm, set_runtime, set_tqdm_class


class Worker:
    def __init__(self, rt: Runtime) -> None:
        self.rt = rt
        self._thr: threading.Thread | None = None

    @property
    def busy(self) -> bool:
        return self._thr is not None and self._thr.is_alive()

    def scan(self, original_argv: list[str]) -> None:
        self._spawn(self._scan, original_argv)

    def run(self, data_path: str, stems: list[str]) -> None:
        self._spawn(self._run, data_path, stems)

    # ── internals ───────────────────────────────────────────────────

    def _spawn(self, target, *args) -> None:
        if self.busy:
            return
        self._thr = threading.Thread(target=target, args=args, daemon=True)
        self._thr.start()

    def _setup(self, original_argv: list[str]) -> None:
        """Clear GlobalHydra and reset sys.argv to the original CLI args.

        ``call_in_raw_dir`` extracts the data path from sys.argv via
        ``parse_data_path`` and inserts ``--config-dir`` — so we must give
        it the full original argv each time.
        """
        from hydra.core.global_hydra import GlobalHydra
        GlobalHydra.instance().clear()
        sys.argv = list(original_argv)
        set_runtime(self.rt)
        set_tqdm_class(GuiTqdm)

    def _wrap(self, fun):
        """Add QueueHandler *after* Hydra dictConfig (inside task fn)."""
        rt = self.rt
        @wraps(fun)
        def wrapped(cfg):
            h = install_qh(rt.log_queue, rt.pause_gate)
            try:
                return fun(cfg)
            finally:
                logging.getLogger().removeHandler(h)
        return wrapped

    def _scan(self, original_argv: list[str]) -> None:
        from tcm import cli, processing
        self._setup(original_argv)
        # Null overall progress bar during scan (no probe-level processing)
        self.rt.progress_overall.set(0, 0, "")
        try:
            res = cli.call_in_raw_dir(
                self._wrap(processing.run),
                config_name="config",
                program={"return_": "<cfg_from_args>"},
                exit_on_error=False,
            )
            self.rt.result_queue.put(("scan_ok", res))
        except Exception as exc:
            logging.getLogger(__name__).exception("scan failed")
            self.rt.result_queue.put(("scan_error", exc))

    def _run(self, data_path: str, stems: list[str]) -> None:
        from tcm import cli, processing
        # Minimal argv: no original CLI overrides.  YAML files (edited by user)
        # are the sole config source.  Data path passed via overrides.
        self._setup(["__main__"])
        self.rt.progress_overall.set(0, 0, "")
        try:
            res = cli.call_in_raw_dir(
                self._wrap(processing.run),
                config_name="config",
                input={"path": data_path, "yaml_path": f"({'|'.join(stems)})"},
                exit_on_error=False,
            )
            self.rt.result_queue.put(("run_ok", res))
        except Exception as exc:
            logging.getLogger(__name__).exception("run failed")
            self.rt.result_queue.put(("run_error", exc))