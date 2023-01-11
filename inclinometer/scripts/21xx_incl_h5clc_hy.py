from pathlib import Path
import sys
from yaml import safe_dump as yaml_safe_dump

import to_vaex_hdf5.cfg_dataclasses as cfg_d
import inclinometer.incl_h5clc_hy as incl_h5clc_hy

# raw data db - data that was converted from csv
path_db_raw = Path(
    r'd:\WorkData\BalticSea\_Pregolya,Lagoon\221103@ib26,28,29,30\_raw\221103.raw.h5'
    # r'd:\WorkData\KaraSea\220906_AMK89-1\inclinometer\_raw\220910.raw.h5'
    # r'e:\WorkData\BalticSea\181005_ABP44\inclinometer\_raw\181022.raw.h5'
    # r'e:\WorkData\BalticSea\181005_ABP44\inclinometer\_raw\181017.raw.h5'
    # r'd:\WorkData\BalticSea\220601_ABP49\inclinometer\_raw\220603.raw.h5'
    # r'd:\WorkData\BalticSea\220505_D6\inclinometers\_raw\220505.raw.h5'
    # r'd:\WorkData\BalticSea\_Pregolya,Lagoon\220327@i36\_raw\220327.raw.h5'
    # r'd:\WorkData\BalticSea\210924_AI59-inclinometer\_raw\210924.raw.h5'
    # r'd:\workData\BalticSea\201202_BalticSpit_inclinometer\211008P7.5,15,E15m@i04,11,14,36,37,38,w2,5\_raw\211008.raw.h5'
    # r'd:\workData\BalticSea\_Pregolya,Lagoon\210908-inclinometer\_raw\210908.raw.h5'
    # r'd:\workData\BalticSea\_Pregolya,Lagoon\211111-inclinometer\_raw\211111.raw.h5'
    # r'd:\workData\BalticSea\201202_BalticSpit_inclinometer\211008E15m@i11,36,37,38,w2\_raw\211008.raw.h5'
    # r'd:\workData\BalticSea\201202_BalticSpit_inclinometer\210618P7.5m@i9,14,15,19w1,2,4\_raw\210726.raw.h5'
    # r'd:\workData\BalticSea\201202_BalticSpit_inclinometer\210618P7.5m@i9,14,15,19w1,2,4\210618@i14,15+19+09,w2+1,4.proc_noAvg.h5'
    # r'd:\workData\BalticSea\201202_BalticSpit_inclinometer\210618P7.5m@i9,14,15,19w1,2,4\210618@i14,15+19,w2+1,4.proc_noAvg.h5'
    # r'd:\workData\BalticSea\201202_BalticSpit_inclinometer\210726P10,15,E14.5m@i4,5,9,11,36,37,38,w1,2,5,6\210726P10,15m@i5+14,9+15,w5+1,2.proc_noAvg.h5'
    # r'd:\workData\BalticSea\201202_BalticSpit_inclinometer\210726@P7.5m,P15m-i9,14,19w1,4\_raw\210726.raw.h5'
    # r'd:\workData\BalticSea\201202_BalticSpit_inclinometer\210827@P10m,P15m-i14,15,w1,4\_raw\210827.raw.h5'
    # r'd:\workData\BalticSea\201202_BalticSpit_inclinometer\210618@P7.5m-i15,w2(cal)\_raw\210618.raw.h5'
    # r'd:\workData\BalticSea\201202_BalticSpit_inclinometer\210922@E15m-i19,36,37,38,w2\_raw\210922.raw.h5'
    # r'd:\workData\BalticSea\201202_BalticSpit_inclinometer\201202@i3,5,9,10,11,15,19,23,28,30,32,33,w1-6\_raw\201202.raw.h5'
    # r'd:\workData\BalticSea\201202_BalticSpit_inclinometer\210726@i4,5,11,36,37,38,w2,5,6\_raw\210726.raw.h5'
    ).absolute()


# Setting of hydra.searchpath to cruise specific config dir: "{path_db_raw.parent}/cfg_proc" (probes config directory)
# within inclinometer/cfg/incl_h5clc_hy.yaml - Else it will be not used and hydra will only warn
path_cfg_default = (lambda p: p.parent / 'cfg' / p.name)(Path(incl_h5clc_hy.__file__)).with_suffix('.yaml')
with path_cfg_default.open('w') as f:
    yaml_safe_dump({
        'defaults': ['base_incl_h5clc_hy', '_self_'],
        'hydra': {'searchpath': [path_db_raw.with_name("cfg_proc").as_uri().replace('///', '//')]}  # .as_posix()
        }, f)
    f.flush()

"""
defaults:
  - base_incl_h5clc_hy
# - probes: wavegauges
# if installed
#  - override hydra/job_logging: colorlog
#  - override hydra/hydra_logging: colorlog
  - _self_  # just to prevent warning
hydra:
  searchpath:  # add path to cfg_proc dir
    - file://d:/workData/BalticSea/201202_BalticSpit_inclinometer/211008E15m@i11,36,37,38,w2/_raw/cfg_proc
#    - file://d:/workData/BalticSea/201202_BalticSpit_inclinometer/210618P7.5m@i9,14,15,19w1,2,4/_raw/cfg_proc
#    d:/workData/BalticSea/201202_BalticSpit_inclinometer/210827@P10m,P15m-i14,15,w1,4/_raw/cfg_proc
#    - file://d:/workData/BalticSea/201202_BalticSpit_inclinometer/210618@P7.5m-i15,w2(cal)/_raw/cfg_proc
#    - file://d:/workData/BalticSea/201202_BalticSpit_inclinometer/210922@E15m-i19,36,37,38,w2/_raw/cfg_proc/
#    - file://c:/temp/cfg_proc
#    - file://d:/workData/BalticSea/201202_BalticSpit_inclinometer/201202@i3,5,9,10,11,15,19,23,28,30,32,33,w1-6/_raw/cfg_proc
"""


db_out = None  # '"{}"'.format((path_db_raw.parent.parent / f'{path_db_raw.stem}_proc23,32;30.h5').replace('\\', '/'))

aggregate_period_s = {  # will be joined for multirun. Default: [0, 2, 300, 600, 1800, 7200]: [300, 1800] is for burst mode
    'inclinometers': [0, 2, 600, 3600, 7200],   # 0, [200]  0,  [0, 2, 600, 7200]  .  #[0, ]
    'wavegauges': [0, 2, 300, 3600],    # 0, [0, 2, 300, 3600]   #[0],
    }

# todo: Change config dir and hydra output dir. will be relative to this dir. to raw data dir.
sys_argv_save = sys.argv.copy()
# sys.argv = ["c:/temp/cfg_proc"]  #[str(path_db_raw.parent / 'cfg_proc')]  # path of yaml config for hydra (main_call() uses sys.argv[0] to add it)
db_in = str(path_db_raw).replace('\\', '/')
# db_in = str(r'e:\WorkData\BalticSea\181005_ABP44\inclinometer\181017-27.proc_noAvg.h5').replace('\\', '/')
split_avg_out_by_time_ranges = False   # True  # Run only after common .proc_noAvg.h5 saved (i.e. with aggregate_period_s=0)
# 'inclinometers wave gauges'
for probes in 'inclinometers'.split():  # 'inclinometers wavegauges', 'inclinometers_tau600T1800'
    if False:  # not False - save by time_ranges
        if split_avg_out_by_time_ranges:
            db_out = str(path_db_raw.parent.with_name(
                '_'.join([
                    path_db_raw.parent.parent.name.replace('-inclinometer', ''),
                    'ranges'  # st.replace(":", "")
                    ])
                ).with_suffix(".proc.h5")).replace('\\', '/')
            print('Saving to', db_out)
        cfg_d.main_call([
            # f'in.min_date=2021-11-11T{st}:00',
            # f'in.max_date=2021-11-11T{en}:00',
            f'in.db_path="{db_in}"',
            f'out.db_path="{db_out}"',
            'out.b_del_temp_db=True',
            'program.verbose=INFO',
            'program.dask_scheduler=threads',
            f'+probes={probes}',  # see probes config directory
            f"out.aggregate_period={','.join(f'{a}s' for a in aggregate_period_s[probes])}",
            ] + ([
                #  f"out.aggregate_period={','.join(f'{a}s' for a in aggregate_period_s[probes] if a !=0)}",
                'out.b_split_by_time_ranges=True',  # flag to split by time_ranges (defined in cfg_proc\probes\inclinometers.yaml)
                ] if split_avg_out_by_time_ranges else [
                ]) +
            (
                ["in.tables=['i.*']"] if probes == 'inclinometers' else  # ['incl(23|30|32).*']  # ['i.*']
                ["in.tables=['w.*']"]                                    # ['w0[2-6].*']         # ['w.*']
            ) + ['--multirun'], fun=incl_h5clc_hy.main)
    else:
        df = cfg_d.main_call([
            f'in.db_path="{db_in}"',
            # '++filter.time_bad_intervals=[2021-06-02T13:49, now]', # todo
            # 'input.tables=["incl.*"]', # was set in probes config directory
            f'out.db_path={db_out}',
            # f'out.table=V_incl_bin{aggregate_period_s}s',
            'out.b_del_temp_db=True',
            # f'out.text_path=text_output',
            'program.verbose=INFO',
            'program.dask_scheduler=threads',
            f'+probes={probes}',  # see probes config directory
            f"out.aggregate_period={','.join(f'{a}s' for a in aggregate_period_s[probes])}",
            # '--config-path=cfg_proc',  # Primary config module 'inclinometer.cfg_proc' not found.
            # '--config-dir=cfg_proc'  # additional cfg dir
            # 'in.min_date=2018-10-17T16:30',
            # 'in.max_date=2018-10-18T07:15',
            # Note: "*" is mandatory for regex, "incl" only used in raw files, but you can average data in processed db.
            "in.tables=['i.*']" if probes == 'inclinometers' else  # ['incl(23|30|32).*']  # ['i.*']
            "in.tables=['w.*']",                                    # ['w0[2-6].*']         # ['w.*']

            '--multirun'],
            fun=incl_h5clc_hy.main)

sys.argv = sys_argv_save
