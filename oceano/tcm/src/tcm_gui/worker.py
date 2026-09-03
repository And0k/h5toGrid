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
        from pathlib import Path

        from tcm import cli, processing
        from utils import log_init

        lf = log_init.LoggingStyleAdapter(__name__)

        self._setup(original_argv)
        # Ensure pre-Hydra probe INFO reaches the queue even if App startup level was reset
        try:
            if logging.getLogger().getEffectiveLevel() > logging.INFO:
                logging.getLogger().setLevel(logging.INFO)
        except Exception:
            pass
        # Trace trigger path for every scan
        try:
            p = Path(str(data_path)).expanduser()
            rp = p.absolute() if p.is_absolute() else (Path.cwd() / p).absolute()
            lf.info("Worker scan trigger={} rp={}", data_path, rp)
        except Exception:
            rp = Path(str(data_path))
            lf.info("Worker scan trigger={} (rp resolve failed)", data_path)

        # Data-driven parent detection: shallow hit → single-anchor,
        # shallow miss → filtered recursive via meta_finder → scan_list if >1 valid anchor.
        # No unfiltered rglob; anchor discovery lives in csv_load/search via find_device_dirs.
        try:
            p_probe = Path(str(data_path)).expanduser()
            rp_probe = p_probe.absolute() if p_probe.is_absolute() else (Path.cwd() / p_probe).absolute()
            # Probe shallow without full processing: csv_load.search_csv_files will do
            # shallow first, then filtered recursive only on miss.
            from tcm import csv_load as _cl

            # Use trigger for logging traceability
            try:
                discovered = _cl.search_csv_files(rp_probe, trigger=data_path)
            except FileNotFoundError as exc:
                # No files at all → surface as scan_error with trigger in message
                lf.info("Worker scan: no files for trigger={} → {}", data_path, exc)
                self.rt.result_queue.put(("scan_error", exc))
                return
            # discovered is {(model,num):[paths,...]} grouped over all valid anchors
            # Derive distinct anchors from composite paths
            anchors_set: set[Path] = set()
            for flist in discovered.values():
                for f in flist:
                    try:
                        from tcm.search import is_archive_composite, split_archive_path
                        from tcm.paths import anchor_for_fs_path

                        if is_archive_composite(f):
                            sp = split_archive_path(f)
                            if sp is not None:
                                dir_archive, _rel = sp
                                base = dir_archive.parent if not dir_archive.is_dir() else dir_archive
                                anchors_set.add(anchor_for_fs_path(base))
                            else:
                                anchors_set.add(anchor_for_fs_path(Path(f).parent))
                        else:
                            # Loose file — anchor is _raw
                            anchors_set.add(anchor_for_fs_path(Path(f).parent))
                    except Exception:
                        continue
            anchors = sorted(anchors_set)
            # Parent list when trigger is a directory (not _raw) and discovered anchors differ from trigger.
            # DRY: use anchors vs trigger comparison, not duplicated shallow iterdir heuristic.
            # Shallow hit on single _raw → anchors == [trigger] → not a list.
            # Cruise root shallow miss → anchors are child _raw(s) != trigger → list (even single).
            is_parent_list = False
            try:
                if rp_probe.is_dir() and rp_probe.name.lower() != "_raw" and len(anchors) >= 1:
                    if len(anchors) > 1 or anchors[0].resolve() != rp_probe.resolve():
                        is_parent_list = True
            except Exception:
                pass
            if is_parent_list:
                lf.info("Worker scan trigger={} → scan_list with {} anchors: {}", data_path, len(anchors), ", ".join(str(a) for a in anchors))
                # Pass parent explicitly so App can log it before field replacement
                self.rt.result_queue.put(("scan_list", (data_path, anchors)))
                return
            # Single-anchor or file input with shallow hit — fall through to full processing on that anchor
            # (processing.run will reuse shallow result and not re-scan parent)
        except SystemExit:
            raise
        except Exception as exc:
            # Discovery probe failed for unexpected reason — log and fall through to full processing
            # (processing.run will surface the real error with proper handling)
            lf.debug("Worker scan probe failed for trigger={}: {}", data_path, exc, exc_info=True)

        # Single-anchor or file scan — full tab-fill via processing
        self.rt.progress_overall.set(0, 0, "")
        try:
            res = cli.call_in_raw_dir(
                self._wrap(processing.run),
                config_name="config",
                input={"path": data_path},
                program={"return_": "<cfg_from_args>"},
                exit_on_error=False,
                enable_file_logging=False,
            )
            self.rt.result_queue.put(("scan_ok", res))
        except SystemExit as exc:
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
