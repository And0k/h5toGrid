#!/usr/bin/env python3
# coding:utf-8
"""
Author:  Andrey Korzh <ao.korzh@gmail.com>
Purpose: Calibrate AB SIO RAS inclinometers.
The following steps and corresponding `st.start` and `st.end` values to control execution implemented:
- 10: magnetometer and accelerometer in lab;
- 15: copy lab coef to tank;
- 20: velocity in tank;
- 35: copy tank coef where need.
Note: if st.end >= st.start then one step will be processed.

Modified: 08.07.2026
Requirement: source raw data loaded to HDF5/NC table
"""
from datetime import datetime

from tcm import cli, format, h5inclinometer_coef, processing
from tcm.utils2init import init_logging, path_on_drive_d, st
from tcm.calibration import run

st.start = 11  # 10 15 20 35
st.end = 10

## Source calibration lab/tank data path
# (in lab + in tank if you not redefine it under step 15)
path_db_raw = path_on_drive_d(
    r"B:\WorkData\experiment\inclinometer\260624@ip05-Press\_raw\260624.raw.nc"
    # r"D:\WorkData\_experiment\inclinometer\250610tank@i63,64,67,68,78,86,87\_raw\250610tank.raw.h5"
    # r"D:\WorkData\_experiment\inclinometer\250505_tube\_raw\250505.raw.h5"
    # r"F:\_\copy\AB_SIO_RAS\240527_stand,tank,tube@i61-90\_raw\240604tube.raw.h5"
    # r"d:\WorkData\_experiment\inclinometer\240527_stand,tank,tube@i61-90\_raw\240527.raw.h5"
    # r"d:\WorkData\_experiment\inclinometer\_type_b\231009_tank@iB18,25-30\_raw\231009.raw.h5"
    # r'd:\WorkData\_experiment\inclinometer\231229_stand@i58-60\_raw\231229.raw.h5'
    # r'd:\WorkData\_experiment\inclinometer\231010_stand,tank@i52-56\_raw\231010stand.raw.h5'
    # r'd:\WorkData\_experiment\inclinometer\230727_stand@i3,4\_raw\230727stand.raw.h5'
    # r'd:\WorkData\BalticSea\230423inclinometer_Zelenogradsk\_raw\230423.raw.h5'
    # r'd:\WorkData\_experiment\inclinometer\230428_stand,tank,pres@ip1-6\_raw\230428stand.raw.h5'
    # r'd:\WorkData\_experiment\inclinometer\_type_b\230117_stand@ib26,28-30\_raw\230117.raw.h5'
    # r'd:\WorkData\_experiment\_2018\inclinometer\181003_compas\181004.raw.h5'
    # r'd:\WorkData\_experiment\_2018\inclinometer\181003_compas\181003compas.h5'
    # r'd:\WorkData\_experiment\inclinometer\_Schukas\210603_lab\_raw\220128.raw.h5'
)

# Probe list and table prefix — shared across all steps
probes = ["p05"]
tables_raw = [f"{format.pcid_to_raw_name(p)}" for p in format.normalize_probes(set(probes))]

# ───────────────────────────────────────────────────────────────────────────── #
# Step 10: Magnetometer and accelerometer calibration
# ───────────────────────────────────────────────────────────────────────────── #

if st(10, "Magnetometer and accelerometer calibration of devices listed in cfg_proc/defaults"):

    db_in = str(path_db_raw).replace('\\', '/')
    cli.call_in_raw_dir(
        run.run_calibration,
        config_name="config",
        yaml_path=path_db_raw.parent / "cfg_proc" / "run" / "230811_1622@i_p5-маг.yaml",
        input={"path": path_db_raw, "tables": tables_raw, "channels": ["M", "A"]},
        out={"db_paths": [db_in]},
        proc={"coverage_projection": "sphere"},
    )

    # run._hydra_main
    # Uses the same loading infrastructure as tcm_clc.py (load_raw → open_nc/open_hdf5).
    # run.run_calibration(z
    #     path=path_db_raw,
    #     tables=tables_raw,
    #     channels=["M", "A"],
    #     db_paths=[db_in],
    # )
if st(11, "Magnetometer and accelerometer calibration of devices listed in cfg_proc/defaults"):
    time_ranges_zeroing = ["2026-06-25T17:23:30", "2026-06-25T17:25:00"]
    cli.call_in_raw_dir(
        processing.run,
        input={
            "path": path_db_raw,
            "tables": tables_raw,
            "time_ranges_zeroing": time_ranges_zeroing,
            "time_ranges": time_ranges_zeroing
        },
    )


# ───────────────────────────────────────────────────────────────────────────── #
# Step 15: Copy lab coefficients to tank database
# ───────────────────────────────────────────────────────────────────────────── #

if st.end >= 15:
    ## Tank calibration settings

    db_path_tank = path_db_raw  # modify this line if need separate tank data
    # path_on_drive_d(  # or other path to load calibration data (add newer on top of first and comment old item)
    # r'd:\WorkData\_experiment\inclinometer\231010_stand,tank@i52-56\_raw\231011tank.raw.h5'
    # r'd:\WorkData\_experiment\inclinometer\230614_tank@i3,4,15,19,28,33,37,38;В27-30\230614tank.raw.h5'
    # r'd:\WorkData\_experiment\inclinometer\230428_stand,tank,pres@ip1-6\_raw\230428tank.raw.h5'
    # r'd:\WorkData\_experiment\inclinometer\_type_b\230109_tube@ib26,28-30\_raw\230109.raw.h5'
    # r'd:\WorkData\_experiment\_2018\inclinometer\181004_tank[1-20]\_raw\181004.raw.h5'
    # r'd:\WorkData\_experiment\inclinometer\220525_tank\_raw\220525.raw.h5'
    # r'd:\WorkData\_experiment\inclinometer\_Schukas\200807_tank[b01-b30]\200807_calibr-tank-b.h5'
    # r'd:\WorkData\_experiment\inclinometer\220112_stand_D01\_raw\220112.raw.h5'
    # r'd:\WorkData\_experiment\inclinometer\210331_tank[4,5,9,10,11,19,28,33,36,37,38]\210331incl.h5'
    # r'd:\WorkData\_experiment\inclinometer\_Schukas\200807_tank[b01-b30]\200807_calibr-tank-b.h5'
    # )

    vsz_param = {
        "dir": db_path_tank.parent / "vsz(250610_1340,range=1h)_tank",  # / "vsz(range=5min)"}
        "substr_not_in_tbl": r"^[^@]+",  # 'tank@'
    }
    # db_path_tank.parent / "vsz(240604_1300,range=1h,db_stem=240604tube)" / f"tube@i{p_type}{p_num:0>2}"
    # db_path_tank.parent / "240920_tube@72,75-77,79-85,89,90" / f"[0-9]*@i{p_type}{p_num:0>2}"
    # db_path_tank.parent / f"*@i{p_type}{p_num:0>2}"  # fr'*@iB{p_num:0>2}'  @i_p
    # db_path_tank.parent / 'vsz(range=1h)' / fr'*@ib{p_num:0>2}g'  # 230109_1404_13min@ib29g.vsz
    # f'{vsz_substr_not_in_tbl}i{p_num:0>2}'
    # tbl
    # f'i_{vsz_substr_not_in_tbl}d{p_num:0>2}'
    # {db_path_tank.stem}

    if path_db_raw != db_path_tank and st(15, 'Copy laboratory calibration coefficients to other experiments databases'):
        for i, tbl in enumerate(tables_raw):
            # incl_calibr not supports multiple time_ranges so calculate one by one p_num
            print(f'Copying {tbl} coefficients from {path_db_raw}')
            h5inclinometer_coef.h5copy_coef(path_db_raw, h5file_dest=db_path_tank, tbl=tbl, ok_to_replace_group=True)

    # For next steps

    # 2. Where to copy coefs during |V| calibration:
    # (Usually needed copy to stand data - see input for 1st step and to common db. Else if stand db is same as
    # db_path_tank.or/and no need copy to other db - set it to empty list.
    db_paths_copy = [
        path_on_drive_d(p)
        for p in [
            r'C:\Work\Python\AB_SIO_RAS\tcm\tcm\cfg\coef\calibration.h5',  # default: all coefficients here
            # r"d:\WorkData\~configuration~\inclinometer\incl#b.h5",
            # r'd:\WorkData\BalticSea\230507_ABP53\inclinometer@i3,4,15,19,37,38;ib27-30,ip6\_raw\230507.raw.h5'
            # r'd:\WorkData\BalticSea\_Pregolya,Lagoon\221103@ib26,28-30\_raw\221103.raw.h5'
            # r'e:\WorkData\BalticSea\181005_ABP44\inclinometer\_raw\181017.raw.h5',
            # r'e:\WorkData\BalticSea\181005_ABP44\inclinometer\_raw\181022.raw.h5',
            # r'd:\WorkData\_experiment\_2018\inclinometer\181003_compas\181004.raw.h5'
            # r'C:\Work\Python\AB_SIO_RAS\tcm\tcm\cfg\coef\calibration.h5',
            # r'd:\WorkData\_experiment\inclinometer\_Schukas\210603_lab\_raw\220128.raw.h5' # 210603incl.h5'
            # r'd:\WorkData\BalticSea\220505_D6\inclinometers\_raw\220505.raw.h5'
            # r'd:\WorkData\_experiment\inclinometer\_Schukas\210603_lab\_raw\220128.raw.h5' # 210603incl.h5'
        ]
    ]


# ───────────────────────────────────────────────────────────────────────────── #
# Step 20: Velocity calibration and North zeroing
# ───────────────────────────────────────────────────────────────────────────── #

if st(20, 'Coefficients to convert inclination to |V| and zero calibration (not heading)'):
    """
    Note: Execute after updating Veusz data file with previous step results. You should:
    - update coefficients in hdf5 store that vsz imports (done in previous step)
    - recalculate calibration coefficients: zeroing (may be in vsz: zeroing interval in it must be set) and fit Velocity
    - save vsz
    Note: Updates Vabs coefs and zero calibration in data source for vsz (but this may not affect the Vabs coefs in vsz
    because of zero calibration is in vsz too).
    """
    from h5from_veusz_coef import main as h5from_veusz_coef
    # from utils.veuszPropagate import __file__ as file_veuszPropagate
    vsz_data = {'veusze': None}
    for i, tbl in enumerate(tables_raw):
        # incl_calibr not supports multiple time_ranges so calculate one by one p_num
        probe = tbl.replace("incl", "i")  # note: regex result from veusz name by re_tbl_from_vsz_name below must be same
        vsz_path = (vsz_param["dir"] / f"@{probe}").with_suffix(".vsz")
        vsz_data = h5from_veusz_coef(
            [
                # str(Path(file_veuszPropagate).with_name('veuszPropagate.ini')),
                '--data_yield_prefix', 'Inclination',
                '--path', str(vsz_path),
                '--pattern_path', str(vsz_path),
                '--widget', '/fitV(incl)/grid1/graph/fit_t/values',
                # '/fitV(force)/grid1/graph/fit1/values',
                '--data_for_coef', 'max_incl_of_fit_t',
                '--out.path', str(db_path_tank),
                #'--re_match_tbl_from_vsz_name', f'[^_@\d]+_?\d+',
                '--re_sub_tbl_from_vsz_name', '^.*',  # r'\D+',
                '--to_sub_tbl_from_vsz_name', tbl,  # tbl_prefix
                '--channels_list', '',  # 'M,A',
                '--b_update_existed', 'True',  # to not skip.
                '--export_pages_int_list', '4,6',  #4 0 = all
                '--b_interact', 'False',
                '--b_execute_vsz', 'True',  # not works without
                '--return', '<embedded_object>',  # reuse to not bloat memory
            ],
            veusze=vsz_data['veusze']
        )

        def any_inside(v):
            if isinstance(v, list):
                return any(any_inside(e) for e in v)
            try:
                return any(v)
            except ValueError:  # The truth value of an array with more than one element is ambiguous
                return v.any()

        if vsz_data is not None:
            if any(any_inside(v) for k, v in vsz_data.items() if k != 'veusze'):
                for db in db_paths_copy:
                    # if step == 3:
                    # to 1st db too
                    # l = init_logging('')
                    print(f'Copy coefficients to {db}/{tbl} from {db_path_tank}')
                    h5inclinometer_coef.h5copy_coef(db_path_tank, db, tbl, ok_to_replace_group=True)

            vsz_data['veusze'].Close()
            try:
                vsz_data['veusze'].WaitForClose()
            except AttributeError:  # already 'NoneType' => closed ok
                pass
        else:
            vsz_data = {'veusze': None}

#%%
tbl_prefix = 'incl'
db_paths_copy = []
if path_db_raw != db_path_tank:
    db_paths_copy.append(path_db_raw)  # db_paths_copy = [path_db_raw]
db_paths_copy.append(
    # r'C:\Work\Python\AB_SIO_RAS\h5toGrid\inclinometer\tests\data\inclinometer\incl#b.h5'
    r"D:\Cruises\BalticSea\240616_ABP56@i,t-chain\inclinometer\_raw\240625.raw.h5"
)
if db_paths_copy and st(35, f"Copy calibration coefficients from {db_path_tank} to {db_paths_copy}"):
    init_logging(logger=__name__)
    for db in db_paths_copy:
        print('to', db)
        for i, p_num in enumerate(probes):
            # incl_calibr not supports multiple time_ranges so calculate one by one p_num
            tbl = f'{tbl_prefix}{p_num:0>2}'
            try:
                print(f'Copying {tbl}', end="... ")
                h5inclinometer_coef.h5copy_coef(db_path_tank, h5file_dest=db, tbl=tbl, ok_to_replace_group=True)
            except KeyError:
                print(f"skipping {tbl} - not found")
            else:
                print('- ok')

#%%

print(f"{datetime.now():%Y-%m-%d %H:%M:%S} Ok>")
