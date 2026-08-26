"""Repro: tabs dirty right after load without user edits."""

from __future__ import annotations

import os
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from omegaconf import OmegaConf

from tcm import cli, processing, schema
from tcm._constants import RAW_DIR_NAME
from tcm.schema import Return
from tcm_gui.coef_sheet import ConfigSheet


def dump(cs: ConfigSheet, tag: str) -> None:
    cur = cs._data_snapshot()
    snap = dict(cs._snap)
    print(f"--- {tag}: is_dirty={cs.is_dirty} meta_dirty={cs.is_metadata_dirty()} "
          f"rows_snap={len(cs._snap)} rows_cur={len(cur)}")
    for iid, vals in cur:
        if snap.get(iid) != vals:
            print(f"  DIFF iid={iid!r}\n    snap={snap.get(iid)}\n    cur ={vals}")


tmp = Path(os.environ["TEMP"]) / "_repro_dirty_tmp"
raw_dir = tmp / RAW_DIR_NAME
raw_dir.mkdir(parents=True, exist_ok=True)
csv_file = raw_dir / "@i_01.txt"
import shutil

shutil.rmtree(tmp, ignore_errors=True)
raw_dir.mkdir(parents=True)
csv_file.write_text(
    "yyyy,mm,dd,HH,MM,SS,Ax,Ay,Az,Mx,My,Mz,Battery,Temp\n"
    "2024,06,13,12,00,00,100.0,200.0,300.0,400.0,500.0,600.0,12.5,25.0\n"
    "2024,06,13,12,00,01,101.0,201.0,301.0,401.0,501.0,601.0,12.5,25.0\n",
    encoding="utf-8",
)

os.chdir(tmp)
sys.argv = ["prog"]
processing.run_processing = lambda cfg_dc, *a, **k: cfg_dc  # scan early-exit (as in test_gui_actions)
result = cli.call_in_raw_dir(
    processing.run,
    overrides={
        "input": {"path": str(raw_dir / "*i*.txt")},
        "program": {"return_": Return.CFG_FROM_ARGS},
    },
    exit_on_error=False,
)
stem, yp, cfg_dc = result[3][0]
cfg = OmegaConf.to_container(cfg_dc, resolve=True)
prog = cfg.get("program") or {}
if prog.get("return_") == schema.Return.CFG_FROM_ARGS:
    prog["return_"] = str(schema.Return.END)
print("stem:", stem, "| yaml:", yp)

root = tk.Tk()
root.withdraw()
frame = ttk.Frame(root)
cs = ConfigSheet(frame)
cs.sh.pack(fill="both", expand=True)

from tcm_gui.cli_cfg import default_cfg

md = [None] * 11  # no info_devices.yaml at startup

# ── Real App reproduction: auto-scan via CLI path (Worker thread) ──
sys.argv = ["tcm_gui", str(tmp / "_raw" / "*i*.txt")]
from tcm_gui.app import App

app = App(argv=sys.argv)
root = app.root
root.withdraw()
print("initial pages:", list(app._pages))


def dump_app(tag: str) -> None:
    for stem, cs in app._pages.items():
        cur = cs._data_snapshot()
        snap = dict(cs._snap)
        print(f"[{tag}] {stem!r}: is_dirty={cs.is_dirty} meta={cs.is_metadata_dirty()} "
              f"snap_n={len(cs._snap)}")
        for iid, vals in cur:
            if snap.get(iid) != vals:
                print(f"  DIFF iid={iid!r}\n    snap={snap.get(iid)}\n    cur ={vals}")


import time
t0 = time.time()
i = 0
while time.time() - t0 < 15:
    root.update()
    i += 1
    if i % 200 == 0:
        dump_app(f"cycle{i}")
print("final pages:", list(app._pages))
dump_app("final")



root.destroy()

