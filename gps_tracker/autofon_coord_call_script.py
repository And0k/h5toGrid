from pathlib import Path
from cfg_dataclasses import main_call
from gps_tracker.autofon_coord import main

path_device = Path((
    r'd:\WorkData\_experiment\tracker\240329_Pregolya'
    # r'd:\WorkData\_experiment\tracker\240315_Devau'
    # r'd:\WorkData\BalticSea\230825_Kulikovo@ADCP,ADV,i,tr\tracker_SPOT'
).replace('\\', '/'))


cfg = {
    'dir_device': str(path_device),
    'file_raw_local': str(
        # path_device / '_raw' / 'ActivityReport.xlsx'
        path_device / '240329@sp1-3.raw.h5'  # to faster loading if reprocess
    ).replace('\\', '/'),
    'date_prefix': path_device.stem.split('_')[0],
    'date_start': '2024-03-29T10:07',  # UTC
    'date_end': '2024-03-29T13:31',    # UTC. Use "now" to not limit
    'D1': 'sp1',  # devices output names. Use sp* for SPOT
    'D2': 'sp2',
    'D3': 'sp3',
    # 'ANCHOR_D_NUM': 1,
    # 'ANCHOR_D_TYPE': 'tr'
}
# dependant fields
cfg.update({
    # 'TYPE@D': 'current@{D}ref{ANCHOR_D_NUM}'.format_map(cfg),
    'file_stem': '{date_prefix}@sp1-3'.format_map(cfg)
})

args = [  # cycle removes (possible) remaining line breaks
    a.format_map(cfg) for a in (
'input.time_interval=[{date_start}, {date_end}]',
'+input.dt_from_utc_hours={{sp.*:2}}',
'+input.alias={{{D1}:LOO1, {D2}:LOO3, {D3}:LOO4}}',
'input.path_raw_local_default="{file_raw_local}"',
# +path_raw_local {{{D3}:None, {ANCHOR_D_TYPE}{ANCHOR_D_NUM}:None}}',
'+process.anchor_coord={{{D1}:[54.696531, 20.434871], {D2}:[54.695979, 20.434749], {D3}:[54.696251, 20.436146]}}',
# anchor_tracker=[{ANCHOR_D_TYPE}{ANCHOR_D_NUM}]',
'process.anchor_depth=-7',
'out.db_path=\'{dir_device}/{file_stem}.h5\'',
'out.tables=["{D1}","{D2}","{D3}"]', # "{ANCHOR_D_TYPE}{ANCHOR_D_NUM}","{D0}",
'process.period_tracks=1D',
'process.max_dr_default=9999999',
'out.dt_bins=[5min, 20min, 5min]',  # items can not be empty
'out.dt_rollings=["", "", "35min"]'
)]
# 'out.tables=[{}]'.format(','.join([f'"{d}"' for d in device])),
# +input.path_raw_local={{{ANCHOR_D_TYPE}{ANCHOR_D_NUM}:"{file_raw_local}"}} ^

main_call(args, fun=main)


if False:
    from gps_tracker.autofon_coord import call_example, proc
    call_example()
    # without hydra still possible to run:
    proc()
    
