from vsz_loader import NaT
import numpy as np
import json
import re
from itertools import groupby

import vsz_add_data


def vsz_draw_init_tcm(
    probes,
    meta_devices_all,
    db,
    time_range,
    cus,
    use_bins=None,
    b_old_format_in_h5: bool = True,
):
    """
    Load processed TCM data to draw with default drawer `vsz_drawer.py`

    Set needed inclinometers IDs for types of devices:
    - cus.USE_bursts: inclinometers working in burst mode
    - ids_i: all inclinometers (burst and continuous),
    - ids_w: wave gauges
    Note: to easier distinguish probe in veusz object names (and replace probe names later) we use in
    probe names "_"-prefix and probe number suffix formatted as 2 digits filled left with 0
    f'_{n}' for n in 'i03,i04,i19,i37,i38,i_p06,i_p05'.split(',')
    :param probes:
    :param time_range:
    :param device_dir:
    :param db_stem:
    :param cus:
    :param use_bins: bin average intervals and corresponding data names prefixes.
    defaults to {"": 2, "bin_": 600, "bin2_": 3600} [s].
    You can exclude high resolution data to load/edit faster
    :param b_old_format_in_h5: , defaults to True
    :globals: cruise_dir, ...
    :retrun: tuple:
        ids_i,
        ids_ip,
        ids_p,
        ids_w,
        ids_order,
        disp_time_range,
        use_bins,
        use_bins_w,
        bin0name,
        bin_burst_name,
        b_one_table,
    """
    format_model_part = lambda model: "" if model in ["i", ""] else f"_{model}"
    if b_old_format_in_h5:
        format_model_part_old = lambda model: "" if model in ["i", ""] else model
        ids_i_old = [  # f'_i{i}' for n in ''.split(',') if n '5,19,11,15'
                "_{type}{_model}{number:02d}".format(**probe, _model=format_model_part_old(probe["model"]))
                for pid, probe in probes["devices"].items() if pid
            ]
    ids_i = [  # f'_i{i}' for n in ''.split(',') if n '5,19,11,15'
            "_{type}{_model}{number:02d}".format(**probe, _model=format_model_part(probe["model"]))
            for pid, probe in probes["devices"].items() if pid
        ]
    _ = set(f"_{k}" for k, v in probes["devices"].items() if hasattr(v, "get") and v.get("w"))
    if _:
        print(f"Devices in burst mode found from metadata: {_}")
        cus.add_later(USE_bursts=_ & set(ids_i))  # Set which of ids_i is in burst mode  | {"_i04"}
    else:
        cus.add_later(USE_bursts=set())
        # todo: autoget and use pulse ratio
    ids_w = [pid for pid in ids_i if pid.startswith("_w")]
    if ids_w:
        ids_i = [pid for pid in ids_i if not pid.startswith("_w")]
        # f'_w{int(n):02d}' for n in ''.split(',') if n]

    print("Prepare processed data for inclinometers & wavegauges:", ids_w, ids_i)
    # Load all devices info {id: 'p','b','bd','s'} - p: point, bd, s: device's depth above bottom and type symbol

    # Get probes from `info_devices.json` config to draw in that order, after loading from `hdf5`
    ids_order = {}  # temporary dict is for uniqueness: will be converted to list
    ids = ids_i + ids_w
    meta_devices = {}
    for pid, val in meta_devices_all.items():
        if True:  # p.startswith('E'):
            for id_to_load in ids:
                id_to_load1 = id_to_load[1:]
                if pid in id_to_load1:
                    # assigning to any value - it will not be used
                    ids_order[id_to_load] = None

                    # Also adding items for stacked devices in meta_devices by combining existed items (and copy other used items)
                    idls = [s for s in re.split(r"([^\d]+\d+)_", id_to_load1) if s]
                    if len(idls) > 1:
                        # print([meta_devices[f'{id_to_load1[0]}{idl[-2:]}'] for idl in id_to_load1.split('_')])
                        meta_devices[
                                id_to_load1
                            ] = [  # combine str items, use 1st item for numbers
                                ",".join(vv)
                                if isinstance(vv[0], str) and iparam != 0
                                else vv[0]
                                for iparam, vv in enumerate(
                                    zip(
                                        *[
                                            meta_devices_all[
                                                f"{id_to_load[0]}{idl[-2:]}"
                                            ]
                                            for idl in idls
                                        ]
                                    )
                                )
                            ]  # ((p,p),(b,b),(bd,bd),(s,s))
                    else:
                        meta_devices[id_to_load1] = val  # (p,b,bd,s,*lat_lon)

                    # # helps to my translation to russian:
                    # if meta_devices[id_to_load1][0] in ('seaward point', 'shoreward point'):
                    #     meta_devices[id_to_load1][0] = '_{}'.format(meta_devices[id_to_load1][0])

    print("Info loaded from info_devices.json:", meta_devices)
    cus.add(DISPdevices_info=meta_devices, comments="p, b, bd, s, c, t, burst_w")

    ids_ip = [n for n in ids_i if n.startswith("_i_p")]
    # names different from ids_i to combine with
    ids_p = [n.replace("i_p", "p") for n in ids_ip]
    # For inclinometers with pressure sensor put pressure graphs before velocities
    ids_order = [
        i
        for n in ids_order
        for i in ([ids_p[ids_ip.index(n)], n] if n in ids_ip else [n])
    ]
    # ids_order = [i for n in ids_order for i in ([n, ids_p[ids_ip.index(n)]] if n in ids_ip else [n])]
    print("Devices draw order:", ids_order)

    # Loading

    ## Define bin averaged data to load

    if use_bins is None:
        use_bins = {"": 2, "bin_": 600, "bin2_": 3600}
    # Use raw data sampling frequency if time range is small
    if any(np.isfinite(time_range)) and len(time_range) >= 2 and time_range[-1] is not NaT:
        dtime_range_s = np.diff(time_range).astype("m8[s]").astype(int).item()
        # <=> np.subtract(*time_range[::-1]).astype(int).item()

        # Exclude too big bins for our time range
        use_bins = {
                n: dt
                for n, dt in use_bins.items()
                if dt < dtime_range_s / 2
            }
        if dtime_range_s <= 3600 or not use_bins:  # < 1H  # may be need < 10min
            use_bins = {"": 0}  # , 'bin_': 2  # 0 means raw data sampling frequency
    else:
        dtime_range_s = None
    # todo: make needed averaging in Veusz if not found in db-data

    use_bins_w = use_bins if any(ids_w) or any(ids_p) else {}

    if cus.postponed["USE_bursts"]:
        use_bins["binB_"] = 1800
        if dtime_range_s and use_bins["binB_"] and dtime_range_s < use_bins["binB_"]:
            cus.add_later(USE_bursts = set())
    cus.add_postponed()
    _ = len(use_bins)
    # Define variables containing names of minimum and burst or (else) max bin for incl/wavegauges
    if _ > 1:
        bin0name, bin_burst_name = list(use_bins)[::(_ - 1)]  # 'bin2_'
    else:
        bin0name = bin_burst_name = list(use_bins)[0]

    # - inclinometers

    # load db with one table for all devices if binned data will be needed

    b_one_table = db.stem.endswith('.proc')  # still may load other db with separate table for each device

    b_load_all_data = False
    if b_load_all_data:
        # try:
        #     ImportFileHDF5(
        #         db,
        #         [f"/i_bin{dt_s}s/table"],
        #         linked=True,
        #         namemap={
        #             f"/i_bin{dt_s}s/table/index": "t_ns",
        #             **{
        #                 f"/i_bin{dt_s}s/table/Pressure{idci}": f"P{idc}"
        #                 for idci, idc in zip(ids_ip, ids_p)
        #             },
        #             **{
        #                 f"/i_bin{dt_s}s/table/{prm}{idci}": f"{prm}{idci}"
        #                 for idci in ids_i
        #                 for prm in ("u", "v", "Temp")
        #             },
        #         },
        #         prefix=bin,
        #     )
        # except Exception as e:
        #     print("Can not load", f"/i_bin{dt_s}s/table", "from", db)
        #     raise e

        # # - wave gauges
        # if any(ids_w):
        #     # use_bins_w = {'': 2, 'bin_': 300, 'bin2_': 3600}
        #     print("Loading", ids_w, "from", db)
        #     namemap = {
        #         f"/w_bin{dt_s}s/table/{v}{'' if v == 'index' else idc}": "".join(
        #             bin,
        #             'P' if v == 'Pressure' else 't_ns' if v == 'index' else v,
        #             '_w' if v == 'index' else idc
        #         )
        #         for idc in ids_w
        #         for v in ("Pressure", "Temp", "index")
        #         for bin, dt_s in use_bins_w.items()
        #     }
        #     ImportFileHDF5(
        #         db,
        #         [f"/w_bin{dt_s}s/table" for dt_s in use_bins_w.values()],
        #         linked=True,
        #         namemap=namemap,
        #     )
        #     print(namemap)

        # try:
        #     TagDatasets(
        #         "loaded",
        #         [
        #             f"{bin}{param}{pid}"
        #             for pid in ids_i
        #             for param in ("u", "v", "Temp")
        #             for bin in use_bins
        #         ]  # inclinometers
        #         + [
        #             f"{bin}{param}{pid}"
        #             for pid in ids_w
        #             for param in (["P"] if pid in ids_p else ("P", "Temp"))
        #             for bin in use_bins_w
        #         ]  # wave gauges
        #         + [
        #             f"{bin}t_ns{sfx_w}"
        #             for sfx_w, bins in (("", use_bins), ("_w", use_bins_w))
        #             for bin in bins
        #             if pid not in ids_p and use_bins[bin]
        #         ],  # indexes
        #     )  # todo: add all f't_ns{pid}' when not use_bins[bin]
        # except Exception as e:
        #     print("TagDatasets Error: ", e)
        pass
    elif ids:
        params = (["u", "v"] if ids_i else []) + ["Temp"]
        for is_binning, ub in groupby(use_bins.items(), lambda x: x[1] > 0):
            if is_binning:  # bin > 0
                cols_name_map = {"index": "t_ns"}
                if b_one_table:
                    h5_group_to_bin_prefix = {
                        f"{'i' if ids!=ids_w else 'w'}_bin{dt_s}s": bin for bin, dt_s in ub
                    }
                    top_groups = list(h5_group_to_bin_prefix.keys())

                    # for bin, dt_s in ub.items():  # ?
                    if b_old_format_in_h5:
                        for idci, idci_old in zip(ids_i, ids_i_old):
                            for prm in params:
                                cols_name_map[f"{prm}{idci_old}"] = f"{prm}{idci}"
                    else:
                        for idci in ids_i:
                            for prm in params:
                                cols_name_map[f"{prm}{idci}"] = f"{prm}{idci}"  # need?
                    for id_p, idc in zip(ids_ip, ids_p):
                        cols_name_map[f"Pressure{id_p}"] = f"P{idc}"
                    # if any(ids_w) or any(ids_p):
                    #     # - wave gauges
                else:
                    h5_group_to_bin_prefix = {
                        f"{pid.removeprefix('_')}bin{dt_s}s": (bin, pid)
                        for bin, dt_s in ub
                        for pid in ids  # ids_i
                    }
                    top_groups = list(h5_group_to_bin_prefix.keys())
                    cols_name_map.update(
                        {
                            f"{prm}": f"{prm}"
                            for prm in params
                            # if len(probes["devices"]) > 1 else params
                        }
                    )
                    for id_p in ids_p:
                        cols_name_map[f"Pressure{id_p}"] = f"P{id_p}"
                if any(ids_w):
                    top_groups += [f"w_bin{dt_s}s" for bin, dt_s in ub]

                def f_table_cols_fmt(col, device_id):
                    """Add prefixes / suffixes for columns of each grp_d group"""
                    if b_one_table:
                        bin = h5_group_to_bin_prefix[device_id]
                        if device_id in ids_w:
                            if col == "index":  # 't_ns'?
                                prefix = ""
                                sfx = "_w"
                            else:
                                prefix = "w_"
                                sfx = ""
                        else:
                            prefix = ""
                            sfx = ""
                    else:
                        prefix = ""
                        bin, device_id = h5_group_to_bin_prefix[device_id]
                        sfx = device_id

                    return f"{prefix}{bin}{col}{sfx}"
                db_for_bin = db
            else:  # no binning (bin = 0 means this): loading from *.proc_noAvg.h5 db
                cols_name_map = {"index": "t_ns", **dict(zip(params, params))}
                if ids_p or ids_w:
                    cols_name_map["Pressure"] = "P"
                top_groups = {pid.removeprefix("_") for pid in ids}  # ids_i

                def f_table_cols_fmt(col, device_id):
                    """Add prefixes / suffixes for columns of each grp_d group"""
                    # old: col_out = col.replace("ip", "i_p").replace("ib", "i_b")
                    if col == 'P':
                        device_id = device_id.replace('i_', '')
                    return f"{col}_{device_id}"

                db_for_bin = db.with_name(
                    db.name.replace(
                        ".proc." if len(probes["devices"]) > 1  else ".proc_Avg.",
                        ".proc_noAvg.",
                        1,
                    )
                )
            existed_devs, time_range_raw, i_ranges = vsz_add_data.veusz_load_hdf5(
                db_for_bin,
                top_groups,
                grp_d={"table": "/table/"},
                cols_namemap={"table": cols_name_map},
                grp_d_rename_funs={"table": f_table_cols_fmt},
                time_range=time_range,
                time_shift_s=cus.USE_timeShift_s,
                decimation=probes.get("decimation"),
            )

    # Set not defined `time_range` elements from raw time range
    if not any(np.isfinite(time_range)):
        time_range = time_range_raw
    elif len(time_range_raw) == 2:
        time_range = np.where(np.isnat(time_range), time_range_raw, time_range)

    # Calculation
    # ###########
    if any(ids_p):
        ids_w += ids_p
    if ids_w:
        bin_max_w = list(use_bins_w)[-1]

    # separately for iclinometers and wave gauges
    for devs, sfx_w, ub, bin0 in [  # devices, w suffix for wave gauges, bins to use, name of min bin
        (ids_i, "", use_bins, bin0name),  # inclinometers
        (ids_w, "w", *((use_bins_w, next(iter(use_bins_w))) if ids_w else ([], None))),  # wave gauges
    ]:
        # Pid time suffix needed if loaded data is not of one device per type (w/i) or data is not combined
        b_t_sfx_is_pid = not b_one_table  # not (b_one_table and (bin0 or (ub and ub[bin0])))?
        t_sfx = (devs[0] if devs else None) if b_t_sfx_is_pid else ("_w" if sfx_w else "")
        for i_bin, bin in enumerate(ub):
            if i_bin == 0:
                ## Minimum/no bin
                if t_sfx is not None:
                    SetDataExpression(
                        "dt" if not bin else f"{bin[:-1]}{sfx_w}",
                        f"1E-9*min(diff({bin}t_ns{t_sfx}[:3]))",
                        linked=True,
                    )
                # Intervals for each device
                for i_pid, pid in enumerate(devs):
                    if b_t_sfx_is_pid:
                        t_sfx = pid
                    if sfx_w:
                        try:
                            _w = ""
                            ip = ids_p.index(pid)
                        except ValueError:  # pid is not in list
                            _w = "_w"
                            SetData2DExpression(
                                f"iUseAuto{pid}",
                                f"[flatnonzero(isfinite({bin0}P{pid}))[[0,-1]]]",
                                linked=True,
                            )
                            SetData2DExpression(
                                f"iu{pid}",
                                f"v.min_range_2d(atleast_2d(v.i_positive(v.i_use(t_ns_w, USEtime{pid}, "
                                f"t_shift_s=USE_timeShift_s), t_ns_w.size)), iUseAuto{pid})",
                                linked=True,
                            )
                        SetDataExpression(
                            f"mean_P{pid}",
                            f"nanmean({bin_max_w}P{pid}[sl_({bin_max_w}iu{pid if _w else ids_ip[ip]})])",
                            linked=True,
                        )
                    else:
                        SetData2DExpression(
                            f"iUseAuto{pid}",
                            f"[flatnonzero(isfinite({bin}u{pid}))[[0,-1]]]",
                            linked=True,
                        )
                        SetData2DExpression(
                            f"iu{pid}",
                            f"v.min_range_2d(atleast_2d(v.i_positive(v.i_use(t_ns{t_sfx}, USEtime{pid}, "
                            f"t_shift_s=USE_timeShift_s), t_ns{t_sfx}.size)), iUseAuto{pid})",
                            linked=True,
                        )
                        # SetData2DExpression(
                        #     f"iu{pid}",
                        #     f"searchsorted({bin}t_ns{t_sfx}, (float64(array(DISPtime, 'datetime64[ns]'))) - "
                        #     f"USE_timeShift_s*1E9) if len(DISPtime)>0 else [[0, {bin}t_ns{sfx_w}.size]]",
                        #     linked=True
                        # )
                        SetDataExpression(
                            f"time_span{pid}",
                            f"around(v.dt64s2vsz(1E-9*{bin}t_ns{t_sfx}[sl_(iu{pid})][[0, -1]] + "
                            "USE_timeShift_s) / 60) * 60",
                            linked=True
                        )
            elif devs:
                ## Binning
                for pid in devs:
                    # - other bins data:
                    t_sfx = "" if b_one_table else pid

                    if sfx_w:
                        try:
                            _w = ""
                            ip = ids_p.index(pid)
                        except ValueError:  # pid is not in list
                            _w = "_w"
                    else:
                        _w = ""
                    # Suffix is used if we've loaded not combined bin0name data:
                    t0sfx = pid if b_t_sfx_is_pid else _w

                    SetData2DExpression(
                        f"{bin}iu{pid}",
                        (
                            f"[searchsorted({bin}t_ns{t_sfx}, {bin0}t_ns{t0sfx}[int32(clip("
                            f"iu{pid}[0], 0, {bin0}t_ns{t0sfx}.size - 1))]) + int32([0, -1])]"
                        ),
                        linked=True,
                    )
                    # Mininum interval of all devs
                    SetData2DExpression(
                        f"{bin}iu_cmn{pid}",
                        (
                            f"[searchsorted({bin}t_ns{t_sfx}, "
                            "v.vsz2dt64s(time_span_i_common).astype(int)*1E9) + int32([0, -1])]"
                        ),
                        linked=True,
                    )


                    # Time for individual devices
                    SetDataExpression(
                        f"{bin}t0st{pid}",
                        f"v.dt64s2vsz(1E-9*{bin}t_ns{t_sfx}[sl_({bin}iu{pid})]) + USE_timeShift_s",
                        linked=True,
                    )

                    SetDataExpression(
                        f"{bin}Vabs{pid}",
                        f"absolute({bin}u{pid}+1j*{bin}v{pid})[sl_({bin}iu{pid})]",
                        linked=True,
                    )
                    SetDataExpression(
                        f"{bin}Vdir{pid}",
                        f"v.wrap_dir(degrees(arctan2({bin}u{pid}, {bin}v{pid})[sl_({bin}iu{pid})]), disp_central_dir)",
                        linked=True,
                    )
                SetDataExpression(
                    f"{bin[:-1]}{sfx_w}",
                    f"1E-9*min(diff({bin}t_ns{t_sfx}[:3]))",
                    linked=True,
                )
    SetDataExpression(
        "time_span_i",
        "(lambda time_st, time_en: [min(time_st), max(time_en)])(*column_stack(({})))".format(
            "".join(f"time_span{pid}, " for pid in ids_i)
        ),
        linked=True,
    )
    SetDataExpression(
        "time_span_i_common",
        "(lambda time_st, time_en: [max(time_st), min(time_en)])(*column_stack(({})))".format(
            "".join(f"time_span{pid}, " for pid in ids_i)
        ),
        linked=True,
    )
    SetDataExpression(
        "disp_time_span",
        "v.dt64s2vsz(array(DISPtime[0], 'M8[s]')) if len(DISPtime)>0 else time_span_i",
        linked=True,
    )
    # Burst data
    for pid in cus.USE_bursts:
        # Suffix is needed for bins which corresponding loaded data is not combined
        t0sfx = "" if b_one_table and use_bins[bin0name] else pid
        t_sfx = "" if b_one_table else pid
        SetData2DExpression(
            f"binB_iu{pid}",
            f"[searchsorted(binB_t_ns{t_sfx}, {bin0name}t_ns{t0sfx}[int32("
            f"clip(iu{pid}[0], 0, {bin0name}t_ns{t0sfx}.size - 1))]) + int32([0, -1])]",
            linked=True,
        )
        SetDataExpression(
            f"binB_t0st{pid}",
            f"v.dt64s2vsz(1E-9*binB_t_ns{t_sfx}[sl_(binB_iu{pid})] + USE_timeShift_s)",
            linked=True,
        )
        SetDataExpression(
            f"disp_ones_burst_st{pid}",
            f"ones_like({bin_burst_name}t0st{pid}) if '{pid}' in USE_bursts else []",
            linked=True,
        )

    SetDataExpression(
        "disp_central_dir",
        f"0  # f(lambda result: result + (-180 if result > 0 else 180), f(lambda angle, n: floor_divide(360,n)*(floor_divide(n*(angle + 180) + 180, 360)%n), mean_bin_Vdir{ids_i[0] if ids_i else 0}, 4))",
        linked=True,
    )

    # Propose default time range for vsz_drawer:
    #  time_range, for time_range > 1D rounded to hours + 1h to edges
    if any(np.isfinite(time_range)):
        disp_time_range = np.array(time_range, "M8[s]")
        t_float = disp_time_range.astype(np.float64)
        disp_time_range += np.array(
            (
                [0 if r == 0 else (a - r) for a, r in zip([-3600, 3600], t_float % 3600)]
                if np.diff(t_float) > 24 * 3600  # time range > np.timedelta64(1, "D")
                else 0
            ),
            "m8[s]",
        )
    else:
        disp_time_range = []


    return (
        ids_i,
        ids_ip,
        ids_p,
        ids_w,
        ids_order,
        disp_time_range,
        use_bins,
        use_bins_w,
        bin0name,
        bin_burst_name,
        b_one_table,
    )