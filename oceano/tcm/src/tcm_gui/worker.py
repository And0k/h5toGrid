"""Worker thread: pipeline stages via tcm.cli.call_in_raw_dir in background."""

from __future__ import annotations

import logging
import sys
import threading
from functools import wraps

from ._i18n import STRINGS as _S
from .progress_bridge import GuiTqdm, set_cfg, set_runtime, set_tqdm_class
from .runtime import Runtime


class Worker:
    def __init__(self, rt: Runtime) -> None:
        self.rt = rt
        self._thr: threading.Thread | None = None

    @property
    def busy(self) -> bool:
        return self._thr is not None and self._thr.is_alive()

    def scan(self, original_argv: list[str], data_path: str) -> None:
        self._spawn(self._scan, original_argv, data_path)

    def run(self, data_path: str, stems: list[str]) -> None:
        self._spawn(self._run, data_path, stems)

    # ── internals ───────────────────────────────────────────────────

    def _spawn(self, target, *args) -> None:
        if self.busy:
            return
        self._thr = threading.Thread(target=target, args=args, daemon=True)
        self._thr.start()

    def _setup(self, original_argv: list[str]) -> None:
        """Clear GlobalHydra and reset sys.argv to the non-path CLI args.

        The data path is fed to ``call_in_raw_dir`` as an ``input.path``
        override (OmegaConf merge), never via ``sys.argv`` — Hydra's ANTLR
        override parser chokes on ``@``/``:``/``,`` in Windows paths.  Strip
        the positional path from *original_argv* so only ``key=value`` and
        flag overrides remain: those survive rescans after the GUI path
        field changes, the stale positional does not.
        """
        from hydra.core.global_hydra import GlobalHydra

        from tcm.cli import parse_data_path

        GlobalHydra.instance().clear()
        _, remaining = parse_data_path(list(original_argv))
        sys.argv = remaining
        set_runtime(self.rt)
        set_tqdm_class(GuiTqdm)

    def _wrap(self, fun):
        """Ensure the persistent QueueHandler is on the root logger, then run *fun*.

        The QueueHandler is installed once at App startup (see
        :func:`tcm_gui.app.App.__init__`).  Hydra's ``dictConfig`` (applied
        inside ``@hydra.main`` before the wrapped function runs) **replaces**
        all handlers on the root logger with ``[console, file]`` — so when the
        worker thread re-enters the wrapped function, the persistent QH may be
        gone from the root logger.  Re-attach it here so:

        - worker-pipeline logs (this thread) reach the queue, AND
        - after the worker task returns, GUI-callback logs (main thread)
          keep flowing to the queue — Hydra's handlers stay alongside it.

        ``reset_dedup()`` clears the consecutive-duplicate state so the first
        record of a new task is never swallowed as a "duplicate" of the
        previous task's tail.
        """
        rt = self.rt

        @wraps(fun)
        def wrapped(cfg):
            root = logging.getLogger()
            if rt.queue_handler is not None and rt.queue_handler not in root.handlers:
                root.addHandler(rt.queue_handler)
            if rt.queue_handler is not None:
                rt.queue_handler.reset_dedup()
            set_cfg(None)  # reset per-config attribution before a new task
            return fun(cfg)

        return wrapped

    def _scan(self, original_argv: list[str], data_path: str) -> None:
        from tcm import cli, processing

        self._setup(original_argv)
        # Null overall progress bar during scan (no probe-level processing)
        self.rt.progress_overall.set(0, 0, "")
        try:
            res = cli.call_in_raw_dir(
                self._wrap(processing.run),
                config_name="config",
                input={"path": data_path},
                program={"return_": "<cfg_from_args>"},
                exit_on_error=False,
            )
            self.rt.result_queue.put(("scan_ok", res))
        except SystemExit as exc:
            # _print_usage_error (and similar CLI paths) call sys.exit(1);
            # SystemExit is BaseException, not Exception — catch explicitly
            # so the error reaches the GUI result_queue instead of dying silently.
            logging.getLogger(__name__).error("scan exited: code %s", exc.code)
            self.rt.result_queue.put(("scan_error", exc))
        except Exception as exc:
            logging.getLogger(__name__).exception(_S.get("error.log.scan", "scan failed"))
            self.rt.result_queue.put(("scan_error", exc))

    def _run(self, data_path: str, stems: list[str]) -> None:
        from pathlib import Path

        from tcm import cli, paths, processing

        # Minimal argv: no original CLI overrides.  YAML files (edited by user)
        # are the sole config source.  Data path passed via overrides.
        self._setup(["__main__"])
        # Show a sliver on overall bar immediately (non-zero total → bar visible)
        self.rt.progress_overall.set(0, 1, _S["status.starting"])
        self.rt.progress_stage.set(0, 0, "")
        # Filter by config stems: construct ``cfg_proc/run/(stems).yaml`` path
        # so processing.run auto-detects the .yaml suffix and filters by stem.
        dir_raw = paths.find_dir_raw_absolute(Path(data_path).absolute())
        stem_path = str(dir_raw / "cfg_proc" / "run" / f"({'|'.join(stems)}).yaml")
        try:
            res = cli.call_in_raw_dir(
                self._wrap(processing.run),
                config_name="config",
                input={"path": stem_path},
                exit_on_error=False,
            )
            self.rt.result_queue.put(("run_ok", res))
        except SystemExit as exc:
            logging.getLogger(__name__).error("run exited: code %s", exc.code)
            self.rt.result_queue.put(("run_error", exc))
        except Exception as exc:
            logging.getLogger(__name__).exception(_S.get("error.log.run", "run failed"))
            self.rt.result_queue.put(("run_error", exc))
