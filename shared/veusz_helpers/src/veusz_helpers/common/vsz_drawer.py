from ast import literal_eval
from collections import namedtuple
from itertools import compress, takewhile
from logging import warning
from operator import add
from typing import Optional

import func_vsz as fv
import numpy as np
from dataclasses import replace

# Custom Definitions, global vars: colors, ... are defined in vsz_drawer_cfg (executed from loader)
# Here we execute code for additional functions for drawing 2D and 3D plots (must be after vsz_drawer_cfg):
exec_module_into_globals("vsz_drawer_2d", globals())


def format_TickLabels(dt=None, next_fmt=0, zoom=0, st_fmt=None, compact=False):
    """
        Best tick labels format(interval) for TickLabels (on page of standard lengh or scaled in zoom x 40)
        :param dt: [s], interval
        :param next_fmt:
        :param zoom:
        :param st_fmt:
        :return: formats[st_fmt+next_fmt:en_fmt+zoom] where st_fmt and en_fmt will be found depending on dt



        If next_fmt=0, zoom=0, then
        0    1    2    3    4    5
    ist|    returns format:               |    start         end | str_date_unit_fmt() | text
    0  |                       '%VDS'     | if        dt <= 2.5M |    'SS' after       | ':%Y-%m-%d %H:00'
    1  |                  '%VDM:%VDS'     | if 2.5M < dt <= 8.0M | 'MM:SS' after       | ':%Y-%m-%d %H:00'
    2  |             '%VDH:%VDM:%VDS'     | if 8.0M < dt <= 30 M | 'Time'      of      | ':%Y-%m-%d'
    3  |             '%VDH:%VDM'          | if 30 M < dt <= 2.0D |                     |
    4  |        '%VDd %VDH:%VDM'          | if 2.0D < dt <= 4.0D | 'Day, time' of      | ':%Y-%m'
    5  |        '%VDd^%VDH:%VDM'          | if 4.0D < dt <= 10 D |                     |
    6  |        '%VDd'                    | if 10 D < dt <= 60 D | 'Day'       of      | ':%Y-%m'
    7  |   '%VDm/%VDd'                    | if 60 D < dt <= 1  Y | 'Month/day' of      | ':%Y'
    8  |   '%VDm'                         | if 1  Y < dt <= 2.5Y | 'Month'     of      | ':%Y'
    9  |'%Y/%VDm'                         | if 2.5Y < dt <= 30 Y | ('Year, month'-todo)|
    10 |'%Y'                              | if 30 Y < dt,        | ('Year' - todo)     |
        Shifts on next_fmt, wides on zoom     :
        '%VDd^{%VDH:%VDM}_%VDb' if 2.5D < dt <= 1Y, zoom=1
        Test:
        for k in 2.5 * np.int32([1/60, 1, 24, 24*30, 3*24*30]):
            print(f'{k}:', format_TickLabels(3600*k, next_fmt=1, zoom=2))

    """
    # We will combine this possible parts:
    formats = ["%VDy/", "%VDm/", "%VDd\u2009", "%VDH:", "%VDM:", "%VDS}"]

    # Time scale:   2.5M           50M  2d    4d    10d    60d     1y      2.5y       30y
    time_scale = [  # ist
        0.0417,  # 0
        8,  # 1
        30,  # 2
        #   50 * 60,               #
        2880,  # 3
        5760,  # 4 # 4D
        14400,  # 5
        86400,  # 6 # 60D
        #   525600,                #
        #   1314000,               #
        15768000,  # 7 # 1Y
        15768000 * 2.5,
        15768000 * 30,
    ]  # Minutes
    # 0.0417, 8, 30

    # Max time scale index
    istart = np.searchsorted(time_scale, dt / 60).item() + next_fmt

    # Starting and ending ``formats`` indexes (see table in func help block)
    st_fmt0 = [5, 4, 3, 3, 2, 2, 2, 1, 1, 0, 0][istart]
    en_fmt = [5, 5, 5, 4, 4, 4, 2, 2, 1, 1, 0][istart]

    # Start overwritten?
    if not st_fmt:
        st_fmt = st_fmt0

    if zoom:
        en_fmt = max(
            min(en_fmt + zoom, 5), 1
        )  # if zoom then en_fmt may deviate from mentioned table
        # Not stop on months or on hours
        if ("m" in formats[en_fmt]) or "H" in formats[en_fmt]:
            en_fmt += 1

    # Ensure number of time parts > 2
    if st_fmt == en_fmt == 3:  # %M -> %M:%S
        en_fmt = 4
    elif st_fmt == en_fmt == 4:  # %S -> %M:%S
        st_fmt = 3
    elif en_fmt - st_fmt >= 1:
        # modify string to make it smaller
        if compact:
            if en_fmt >= 3:  # superscript time
                formats[3] = "^{%VDH:"  # start
                formats[en_fmt] = (
                    f"{formats[en_fmt].replace(':', '')}}}"  # stop superscript
                )
            if st_fmt == 1:  # subscript months
                formats[1], formats[2] = "%VDd", "_%VDb_"
    else:
        if en_fmt < st_fmt:
            en_fmt = st_fmt
    formats[en_fmt] = formats[en_fmt][:-1]  # last symbol is not used
    f = "".join(formats[st_fmt : en_fmt + 1])
    return f


# AddCustom('definition', 'fstr_date_unit_fmt(dt, next_fmt)', "[('MM:SS', ':%Y-%m-%d %H:00'),('Time', ':%Y-%m-%d'), ('Day, time', ':%Y-%m'), ('Day', ':%Y-%m'), ('Month-day', ':%Y')][searchsorted(int32([2, 24*2, 24*6, 24*60]), int(dt)//3600) + next_fmt]")
def x_datetime_ticks(zoom, cfg_plot):
    Set("mode", "datetime")
    Set(
        "TickLabels/format",
        f" {format_TickLabels(disp_dtime_range_s, next_fmt=zoom, zoom=zoom * 2)} ",
    )
    if zoom:
        # Set('TickLabels/format', '%VDd\u2009%VDH:%VDM')
        Set("autoRange", "exact")
        Set("autoMirror", False)
        Set("MajorTicks/number", 120)  # 240  80
        Set("MinorTicks/number", 900)
        Set("GridLines/width", "0.25pt")
    else:
        # Set('TickLabels/format', format_TickLabels(disp_dtime_range_s))  #  ^{%VDH:%VDM}
        Set("MajorTicks/number", int(max(0.625 * cfg_plot.graph_width, 3)))  # 24
        minor_ticks = int(2 * cfg_plot.graph_width)
        if minor_ticks < 2:
            Set("MinorTicks/hide", True)
        else:
            Set("MinorTicks/number", minor_ticks)  # 96


def insert_with_curly_braces(pattern, words, n_curly_braces=1):
    """
    Inserts words in pattern with specified number of curly braces
    :param pattern: str with curly braces ({}) which neeed to be replaced by words
    :param words: Sequence[Any]
    :param n_curly_braces: number of curly braces. Default 1 (to use in f-string or with format_map(I))
    """
    _o = "{" * n_curly_braces  # opening
    _c = "}" * n_curly_braces  # closing
    return pattern.format(*(f"{_o}{w}{_c}" for w in words))


def common_point_for_all(pids, n_curly_braces=1):
    """
    Point name from `cus.DISPdevices_info` with depth+units,
    if it is common for all devices else empty string.
    Returns "" or string with words "point"/"sea depth" and "m" in `n_curly_braces` curly braces (ready
    to further formatting number of times  equal to `n_curly_braces`, for translation of these words)
    Ignores character "_" and all after
    :param pids: probe pids list (each `pid` prepended with "_" as in `ids_*` global variables)
    :return: out, either:
    - "" if found difference before 1st "_",
    - 1st point_text, stripped on first "_" if any points with "_" else as is.

    global used: `cus.DISPdevices_info` (equal to `DISPdevices_info` Custom Definition in Veusz): its
    point and depth values for each of devices_graphs
    """
    # Find info at 1st good index (skip not existed in `devices_info` Wind id)
    for i_st, g in enumerate(pids):
        try:
            out_p, out_b = cus.DISPdevices_info[pids[i_st][1:]][:2]  # point, depth
            break
        except KeyError:
            continue
    else:
        return ""
    # Compare with info at next index
    i_st += 1
    out_ = out_p.split("_", maxsplit=1)[0]
    for g in pids[i_st:]:
        try:
            next_p, next_b = cus.DISPdevices_info[g[1:]][:2]
            # print("common_point_for_all cycle:", next_p, next_b)
            # break
        except KeyError:
            continue
        if out_b != next_b:
            return ""
        elif out_p != next_p:
            next_ = next_p.split("_", maxsplit=1)[0]
            if out_ != next_:
                return ""
            out = out_
    if out_b is None:
        return ""
    else:
        if "{" in out_:
            out_ = out_.format_map(fv.I)
        return insert_with_curly_braces(  # include depth in `pattern` argument
            *(  # `pattern`, `words`
                (f"{{}}.\u2009{out_}: {out_b}{{}}", ["st", "m"])  # point
                if out_
                else (f"{{}}: {out_b}{{}}", ["sea depth", "m"])
            ),
            n_curly_braces=n_curly_braces,
        )
        # todo: make updatable using point and depth included in `DISPdevice_info` from `DISPdevices_info`
        #  as `p` and `b`


def add_label_Title(
    sentences,
    split_before_date=False,
    split_params=WidthGrade < WidthGrades["VeryWide"],
    text2="",
    param2_trange=None,
    text_add="",
    cfg_plot=None,
    graphs_height_sum=10,
    grid_leftMargin=None,
    x=0.5,
    y_cm=None,
    str_vsz_time_range='DATA("time_span_i")',
):
    """
    Adds text + inclinometer (and param2) ranges
    :param sentences: texts or expression returning texts. in Veusz each will be formatted with I variable
    (see func_vsz.py) of current Veusz language and then each item 1st letter will be capitaliced
    :param split_before_date:
    :param split_params: slit before ``text2`` to make rows more narrow
    :param text2: same as ``text`` but will be inserted after "{text} {time_range}, "
    :param param2_trange:
    :param text_add: same as ``text`` but will be inserted after previous text as new sentence:
    inserting ". " before and Tile case of 1st letter will be automatic.
    :param graphs_height_sum:
    :param grid_horMargins_sum:
    :return:
    """
    Add("label", name="Title", autoadd=False)
    To("Title")
    label_texts_to_join = [f""" %{{{{v.c1({s}.format_map(I))}}}}%""" for s in sentences]
    if WidthGrade > WidthGrades["VeryNarrow"]:
        sep = (
            r"\\"
            if (WidthGrade == WidthGrades["Narrow"] and split_before_date)
            else "\u2009"
        )

        # # '\u2009' is not supported date format symbol, so we split date: {2:%m-%d}\u2009{2:%H:%M}'
        # # also removing special formatting symbols: {}^_
        # _ = format_TickLabels(disp_dtime_range_s).replace('%VD', '%').replace('{', '').replace('}', '').replace('_', '')
        # fmt_st_en = [_]
        # try:
        # fmt_st_en = _.replace('^', '').split('\u2009', maxsplit=1)
        # except ValueError:
        # try:
        # fmt_st_en = _.split('^', maxsplit=1)
        # except ValueError:  # : not enough values to unpack (expected 2, got 1)
        # pass
        # fmt = ['\u2009'.join([f"{{{k}:{fmt_}}}" for fmt_ in fmt_st_en]) for k in (1,2)]
        # fmt[0] = f'{fmt[0]}\u2009–'
        # fmt[1] = fmt[1].replace('%y/', '')
        # fmt_date = f'{sep}{fmt[0]}{sep}{fmt[1]}'

        # round to nearest minutes if need:
        fmt_labels = format_TickLabels(disp_dtime_range_s)
        s_time = (
            f"v.vsz2dt64s({str_vsz_time_range} + 30).astype('M8[m]')"
            if fmt_labels[-1].endswith("%M")
            else f"v.vsz2dt64s({str_vsz_time_range})"
        )
        label_texts_to_join += [sep, f"%{{{{v.str_time_range(*f({s_time}.tolist))}}}}%"]
        # %{{f(lambda t0,t1: f"{t0:%d.%m.%Y %H:%M} – {(f'{t1:%d.%m.%Y} ' if t0.date() != t1.date() else '')}{t1:%H:%M}'{str_zone}'", *f(v.vsz2dt64s(DATA('time_span__')).tolist))}}%
        if text2:
            if param2_trange:
                s_time2 = (
                    f"v.vsz2dt64s(DATA('{param2_trange}') + 30).astype('M8[m]')"
                    if fmt_labels[-1].endswith("%M")
                    else f"v.vsz2dt64s(DATA('{param2_trange}'))"
                )
                s_time2 = f"{sep}%{{{{v.str_time_range(*f({s_time2}.tolist))}}}}%"
            else:
                s_time2 = ""
            label_texts_to_join += [
                r",\\" if split_params else ", ",
                f""" %{{{{{text2}.format_map(I)}}}}%""",
                s_time2,
            ]
    sep = r"\\" if (WidthGrade != WidthGrades["VeryWide"]) else " "
    str_title = "".join(
        label_texts_to_join
        + ([rf""".{sep}%{{{{v.c1({text_add}.format_map(I))}}}}%"""] if text_add else [])
    )
    Set("label", str_title)
    label_Title_format(
        cfg_plot,
        graphs_height_sum,
        x=x,
        y_cm=y_cm,
    )
    # if WidthGrade == WidthGrades["VeryNarrow"]:
    #     Set('hide', True)
    To("..")


def label_Title_format(
    cfg_plot,
    graphs_height_sum=10,
    x=0.5,
    y_cm=None,
):
    # Set('hide', False)
    # grid_leftMargin is nearly equal to the xUnits width:
    Set(
        "xPos",
        [
            # 0.5 * (graph_width + grid_leftMargin) / (graph_width + grid_horMargins_sum)
            # if WidthGrade == WidthGrades["Narrow"]
            # else
            0.5
        ],
    )
    Set(
        "yPos",
        [
            (y_cm or (0.03 * cfg_plot.grid_bottomMargin))
            / (cfg_plot.grid_bottomMargin + graphs_height_sum)
        ],
    )
    Set("alignHorz", "centre")
    Set("margin", "1pt")
    Set("Text/size", "12pt")
    Set("Text/bold", True)
    Set("Background/transparency", 90)


x_units_nl = (
    WidthGrade == WidthGrades["Narrow"]
)  # reducing date label width using fDisp_date_u(b_nl=True)
force = False  # True


def label_xUnits_add(
    x_path: str,
    graphs_height_sum: float,
    y_cm: float = 0.03,
    nl: bool = x_units_nl,
    force: bool = force,
    cfg_plot=None,
):
    """
    Adds units label that display next largest date unit value of that used in x axis.
    Can be one or two values depending on whether this largest units values are different
    :param x_path: x axis to use its limits if they are not "Auto"
    :param graphs_height_sum: total height of all graphs on page, cm
    :param y_cm: vertical position, cm
    :param nl: use new line between start and stop values of date range
    :param force: change default to True/False to label/not label x units by this function by default
    Return added x Units string without %{{ prefix and }}% suffix
    """
    if not force:
        return False
    Add("label", name="xUnits", autoadd=False)
    To("xUnits")
    # string parts to evaluate i.e. to put between %{{ }}%
    str_units_parts = [
        "fDisp_date_u('",
        x_path,
        "', 'disp_time_span'",
        ", b_nl=True" if nl else "",
        ", allow3rows=True" if cfg_plot.graph_width < 10 else "",
        ")",
    ]
    Set("label", "".join(["%{{"] + str_units_parts + ["}}%"]))
    # Set("hide", False)
    Set("Background/transparency", 100)
    Set(
        "xPos", 1 - 0.02 / (cfg_plot.graph_width + cfg_plot.grid_horMargins_sum)
    )  # 0.259
    Set(
        "yPos", y_cm / (graphs_height_sum + cfg_plot.grid_bottomMargin)
    )  # 0.01, 0.067, 0.11
    Set("alignHorz", "right")
    Set("alignVert", "bottom")  # top
    To("..")
    return "".join(str_units_parts)


def scale_rows(
    scale_height: dict,
    scale_i_graphs: float = None,
    n_graphs: int = 0,
    n_graphs_w: int = 0,
) -> tuple:
    """
    Calculate scaling factors for graph heights while maintaining proper proportional relationships.

    This function distributes graph heights across multiple rows where different graph types
    follow different scaling rules:
    - Pressure/wavegauge graphs (prefixed with '_p', '_w', '_W') maintain fixed scale = 1
      (height = graph_h_default) regardless of other scaling parameters
    - Velocity graphs (prefixed with '_i') are scaled according to scale_height values
      multiplied by scale_i_graphs factor
    - Other graphs specified in scale_height use those scaling factors directly (relative to
      graph_h_default, independent of scale_i_graphs)
    - Other graphs not specified in scale_height get scaled by scale_height_common

    Parameters
    ----------
    scale_height : dict
        Dictionary mapping graph identifiers to scaling factors. For velocity graphs ('_i' prefixed)
        these values are multiplied by scale_i_graphs. For other graphs these values are used
        directly as scaling factors relative to graph_h_default
    scale_i_graphs : float, optional
        Scaling factor for velocity graphs, applied to velocity graphs ('_i' prefixed) in addition to any
        `scale_height` values specified for them
        Default is `axis_max["Vabs"] / v_to_graph_h`
    n_graphs : int, optional
        Total number of graphs (rows) to be displayed
    n_graphs_w : int, optional
        Number of pressure/wavegauge graphs ('_p', '_w', '_W' prefixed) which maintain
        fixed scale = 1 (height = graph_h_default)

    Returns
    -------
    graphs_height_sum : float
        Total sum of all graph heights before applying scale_height adjustments
        (calculated as graph_h_default * (n_graphs_w + (n_graphs - n_graphs_w) * scale_i_graphs))
    scale_height : dict
        Updated scale_height dictionary with velocity graphs scaled by scale_i_graphs
        and invalid graph IDs filtered out
    scale_height_common : float
        Scaling factor for graphs not specified in scale_height that ensures
        proper proportional relationships are maintained between all graph types
        (never returns None - returns graph_h_default when no scaling needed)

    Notes
    -----
    The function ensures that:
    - Pressure/wavegauge graphs maintain fixed height = graph_h_default
    - Velocity graphs are scaled by both scale_height values and scale_i_graphs factor
    - Other specified graphs use scale_height values directly (relative to graph_h_default)
    - Other unspecified graphs use common scaling to maintain proper layout proportions
    - Graph IDs not present in `ids_order` are filtered out with warning
    - When scale_height is empty, scale_height_common defaults to graph_h_default

    Globals
    -------
    graph_h_default : float
        Default height for individual graph rows
    axis_max["Vabs"] : float
        Maximum absolute velocity value for scaling calculations
    v_to_graph_h : float
        Velocity to height scaling factor
    ids_order : list
        Ordered list of all graph identifiers that must be displayed
    """

    # Set default value for scale_i_graphs if not provided
    if scale_i_graphs is None:
        scale_i_graphs = axis_max["Vabs"] / v_to_graph_h

    # Calculate total height that is not dependent on scale_height modifications
    graphs_height_sum = graph_h_default * (
        (n_graphs - n_graphs_w) * scale_i_graphs + n_graphs_w
    )

    # Filter out graph IDs that are not defined in ids_order
    not_defined = set(scale_height.keys()).difference(ids_order)
    if not_defined:
        warning(f"Not defined graphs for scaling: {not_defined}!!! -> Ignoring")
        scale_height = {k: v for k, v in scale_height.items() if k not in not_defined}

    # Return early if no scaling is specified
    if not scale_height:
        return graphs_height_sum, {}, graph_h_default

    # Apply scale_i_graphs to velocity graphs ('_i' prefixed) while keeping
    # other graphs with their specified scale values
    scale_height = {
        pid: (v if pid.startswith(("_p", "_w", "_W")) else v * scale_i_graphs)
        for pid, v in scale_height.items()
    }

    # Calculate number of fixed graphs that maintain scale = 1
    if n_graphs_w:
        # Count graphs that are pressure/wavegauge type but not specified in scale_height
        n_graphs_fixed = n_graphs_w - len(
            [1 for pid in scale_height if pid.startswith("_w")]
        )  # Only pressure/wavegauge graphs not in scale_height remain fixed
    else:
        # When no pressure/wavegauge graphs, count specified non-pressure/wavegauge graphs
        # to maintain same proportion between non-pressure/wavegauge graphs as on combined graph
        n_graphs_fixed = -len(
            [1 for pid in scale_height if pid.startswith(("_p", "_w", "_W"))]
        )

    # Calculate the sum of specified heights from scale_height
    sum_height_i = sum(scale_height.values())

    # Calculate number of graphs that will use scale_height_common
    n_scale_not_specified = n_graphs - len(scale_height) - n_graphs_fixed

    # Calculate common scale factor for unspecified graphs
    # This ensures the total height relationship is maintained
    if n_scale_not_specified > 0:
        scale_height_common = (
            graphs_height_sum / graph_h_default - sum_height_i - n_graphs_fixed
        ) / n_scale_not_specified
    else:
        # If no graphs will use common scaling, set to default value
        scale_height_common = graph_h_default

    # Remove pressure/wavegauge graphs from scale_height when n_graphs_w is 0
    # (these were only kept for proportion calculation)
    if not n_graphs_w:
        scale_height = {
            g: coef
            for g, coef in scale_height.items()
            if not g.startswith(("_p", "_w", "_W"))
        }

    print(
        f"Scaling heights: {scale_height}, other: {scale_height_common:g}",
        f"(Vabs) and fixed (not i): {n_graphs_fixed}" if n_graphs_w else "",
    )

    return graphs_height_sum, scale_height, scale_height_common


def pg_vectors(graphs, scale_height=None, cfg_plot=None):
    """
    bin_lite: Minimum bin we will draw vectors (in lite color) defined above globally
    Example
    pg_vectors(['_i03','_i10','_i33','_i09','_i28'])
    """
    # extract inclinometers
    graphs = [
        pid for pid in ids_order if pid.startswith("_i")
    ]  # or same condition: if pid in ids_i
    n_graphs = len(graphs) + bool(device_wind)
    if not n_graphs:
        return

    scale_height = {
        pid: scale for pid, scale in scale_height.items() if pid.startswith("_i")
    }
    pg_name = "_vectors"
    print(f"Page {pg_name}", end=": ")
    graphs_height_sum, scale_height, scale_height_common = scale_rows(
        scale_height,
        scale_i_graphs=k_height_vectors * axis_max["Vabs"] / v_to_graph_h,  # |0.3  0.5
        n_graphs=n_graphs,
        n_graphs_w=bool(device_wind),
    )  # will scale Wind height same as P  #  0.6 *
    cus.Disp_vectors_scale_height = {
        g: scale_height.get(g, scale_height_common) for g in graphs
    }

    grid_leftMargin = 1.2

    # on left page edge  # -0.75 if WidthGrade == WidthGrades["VeryNarrow"] else -0.045
    l_device_x = -grid_leftMargin / cfg_plot.graph_width
    leg_dev_x = 0.025

    Add("page", name=pg_name, autoadd=False)
    To(pg_name)
    Set("width", f"{cfg_plot.graph_width + cfg_plot.grid_horMargins_sum}cm")
    Set("height", f"{graphs_height_sum + cfg_plot.grid_bottomMargin:g}cm")
    # put x units above x-axis if narrow
    str_units_added = label_xUnits_add(
        f"/{pg_name}/grid1/x",
        graphs_height_sum,
        y_cm=(1.8 if WidthGrade == WidthGrades["Narrow"] and not x_units_nl else 0.06),
        cfg_plot=cfg_plot,
    )

    t = common_point_for_all(graphs)
    _, __ = ('v.pl(f"', '")') if len(graphs) > 1 else ('f"', '"')
    sentences = ([f'"{t}. "'] if t else []) + [
        _ + "{{current velocity}} {{by {info_incl['device']}}}" + __
    ]
    split_before_date = (
        not device_wind and WidthGrade < WidthGrades["Wide"]
    )  # can't split if have extra wind row
    add_label_Title(
        sentences=sentences,
        split_before_date=split_before_date,
        text2='''f"{{{info_wind['nature']}}} {{by}} {{{info_wind['device']}}}"'''
        if device_wind
        else (
            "" if any(r"\\" in s for s in sentences) or split_before_date else r"'\\'"
        ),
        # lift text if no \\ at all
        cfg_plot=cfg_plot,
        graphs_height_sum=graphs_height_sum,
        grid_leftMargin=grid_leftMargin,
        x=0.45,
    )

    Add("grid", name="grid1", autoadd=False)
    To("grid1")
    Set("rows", 3)
    Set("leftMargin", f"{grid_leftMargin:g}cm")
    Set(
        "rightMargin", f"{cfg_plot.grid_horMargins_sum - grid_leftMargin:g}cm"
    )  # old: "0.5pt"
    Set("topMargin", "0.5pt")
    Set("bottomMargin", "1.7cm")
    Set("internalMargin", "0pt")

    # if scale_height:
    #     Set('scaleRows', ([1] if device_wind else []) + [scale_height.get(g, scale_height_common) for g in graphs])
    # # change common scale_height_common to can set required specified ax_max_i = ax_scale_i*ax_max_prev
    # scale_height_common = (n_graphs - sum(scale_height.values())) / (n_graphs - len(scale_height))
    # # Derivation
    # # Total graphs sum of ranges (and height) must not change after scaling some of graphs:
    # # (sum(ax_scale_all_i) + (n_graphs - len(ax_scale_i))*ax_other_scale)*(ax_max - ax_min) = n_graphs*(ax_max - ax_min)
    # # common scale for graphs not in ax_scale_i:
    # # ax_other_scale = (n_graphs - sum(ax_scale_all_i))/(n_graphs - len(ax_scale_i))
    # Set('scaleRows', [2*scale_height.get(g, scale_height_common) for g in graphs])  # 2 is because parameter is antisymmetric

    Add("axis", name="x", autoadd=False)
    To("x")
    # Set('label', f"%{{{{fDisp_date_u('{x_path}', 'disp_time_span'{', b_nl=True' if x_units_nl else ''}{', allow3rows=True' if cfg_plot.graph_width < 10 else ''})}}}}%")
    Set("autoRange", "exact")
    Set("autoMirror", False)
    Set("direction", "horizontal")
    Set("Line/hide", True)
    Set("Label/hide", True)
    Set("TickLabels/hide", True)
    Set("MajorTicks/hide", True)
    Set("MinorTicks/hide", True)
    x_datetime_ticks(False, cfg_plot=cfg_plot)
    To("..")
    Add("axis", name="y[0,1]", autoadd=False)
    To("y[0,1]")
    Set("hide", True)
    Set("min", 0.0)
    Set("max", 1.0)
    Set("autoMirror", False)
    Set("direction", "vertical")
    Set("otherPosition", 1.0)
    Set("TickLabels/color", "#55ff00")
    Set("TickLabels/hide", True)
    Set("MajorTicks/hide", True)
    Set("MajorTicks/manualTicks", [0.5])
    Set("MinorTicks/hide", True)
    Set("GridLines/hide", False)
    To("..")
    Add("axis", name="xL", autoadd=False)
    To("xL")
    Set("hide", True)
    Set("min", 0.0)
    Set("max", 1.0)
    Set("autoMirror", False)
    Set("direction", "horizontal")
    Set("otherPosition", 0.9)
    Set("Line/color", "#00ff00")
    Set("GridLines/hide", True)
    To("..")
    Add("axis", name="yL", autoadd=False)
    To("yL")
    Set("hide", True)
    Set("min", 0.0)
    Set("max", 1.0)
    Set("autoMirror", False)
    Set("direction", "vertical")
    Set("TickLabels/color", "#55ff00")
    Set("MajorTicks/hide", True)
    Set("MajorTicks/manualTicks", [0.5])
    Set("MinorTicks/hide", True)
    Set("GridLines/hide", False)
    To("..")
    Add("graph", name=f"v", autoadd=False)
    To(f"v")
    # Legend: vector scale
    for ii, (dev, have) in enumerate([("wind", device_wind), ("incl", graphs)]):
        if not have:
            continue
        is_wind = dev == "wind"
        prefix = "WIND" if is_wind else "DISP"
        disp_split_leg = True
        if (
            disp_split_leg and device_wind
        ):  # need label "Current" only if have Wind also
            # Separated {dev} vector legend title
            Add("label", name=f"Leg_{dev}_title", autoadd=False)
            To(f"Leg_{dev}_title")
            Set(
                "label",
                "".join(
                    [
                        "%{{v.c1('{{{nature}}}'.format_map(info_",
                        dev,
                        ").format_map(I))}}%",
                    ]
                ),
            )
            Set("hide", False)
            Set(
                "xPos",
                f"float64(eval(str(SETTING('/_vectors/grid1/v/Leg_{dev}/xPos')))) - LANG({{'default': 0.07, 'ru': 0.08}})",
            )
            Set(
                "yPos",  # center with Leg_{dev} and its underlying vector symbol
                f"float64(eval(str(SETTING('/_vectors/grid1/v/Leg_{dev}/yPos')))) - {0.0525 / graphs_height_sum:.4g}",
            )
            Set("positioning", "axes")
            Set("xAxis", "xL")
            Set("yAxis", "yL")
            Set("alignHorz", "left")
            Set("alignVert", "bottom")
            Set("margin", "1pt")
            Set("Text/font", "Arial")
            Set("Text/color", "black")
            Set("Background/transparency", 50)
            To("..")

        Add("label", name=f"Leg_{dev}", autoadd=False)
        To(f"Leg_{dev}")
        Set(
            "label",
            ("%{{(rf'{" if disp_split_leg else "%{{v.c1(rf'{{nature}}\\\\{")
            + f"{prefix.title()}_leg_v:g}}\u2009{{{{units}}}}'.format_map(info_{dev}))}}}}%",
        )
        Set("hide", False)
        Set("xPos", f"LANG({{'default': {leg_dev_x}, 'ru': {leg_dev_x} + 0.01}})")
        Set("yPos", f"({prefix}_legY + {n_graphs - ii})/{n_graphs}")
        # " + ".join(
        #    [
        #        f'{"WIND_legY" if is_wind else "DISP_legY"}'*{1.75/graphs_height_sum:.4g}
        #    ]
        #    + ([f"{(n_graphs - ii - 1)}/{n_graphs}"] if (n_graphs - ii) == 1 else [])
        # ),
        # f'clip({prefix}_legY, 0.05, 0.87) + 0.03'
        Set("positioning", "axes")
        Set("xAxis", "xL")
        Set("yAxis", "yL")
        Set("alignHorz", "left")
        Set("alignVert", "bottom")
        Set("margin", "1pt")
        Set("Text/font", "Arial")
        Set("Text/color", "black")
        Set("Background/transparency", 50)
        To("..")
        # Legend vector line
        Add("line", name=f"Lv_{dev}", autoadd=False)
        To(f"Lv_{dev}")
        Set("arrowleft", "bar")
        Set("arrowright", "arrownarrow")
        Set("arrowSize", "1pt")
        Set("hide", False)
        Set("xPos", f"float64(eval(str(SETTING('/_vectors/grid1/v/Leg_{dev}/xPos'))))")
        Set(
            "yPos",
            f"float64(eval(str(SETTING('/_vectors/grid1/v/Leg_{dev}/yPos')))) - {0.0525 / graphs_height_sum:.4g}",
        )
        Set("length", f"{prefix.title()}_leg_v * {prefix}scale_page_vectors")
        Set("angle", f"zeros(size({prefix.title()}_leg_v))")
        Set("xPos2", "bin2_v")
        Set("yPos2", "bin2_u")
        Set("xAxis", "xL")
        Set("yAxis", "yL")
        Set("Line/color", "blue" if is_wind else "black")
        Set("Line/width", "1pt")
        Set("Line/transparency", 0)
        Set("Fill/color", "black")
        To("..")
        Add("rect", name=f"Lv-bg_{dev}", autoadd=False)
        To(f"Lv-bg_{dev}")
        Set(
            "xPos",
            f"float64(eval(str(SETTING('/_vectors/grid1/v/Leg_{dev}/xPos')))) + ({prefix.title()}_leg_v * {prefix}scale_page_vectors)/2",
        )
        Set(
            "yPos",
            f"float64(eval(str(SETTING('/_vectors/grid1/v/Leg_{dev}/yPos')))) - {0.0525 / graphs_height_sum:.4g}",
        )
        Set("width", f"{prefix.title()}_leg_v * {prefix}scale_page_vectors")
        Set("height", [round(0.05 / graphs_height_sum, 3)])
        Set("positioning", "axes")
        Set("xAxis", "xL")
        Set("yAxis", "yL")
        Set("Fill/color", "white")
        Set("Fill/hide", False)
        Set("Border/hide", True)
        Set("rounding", 5)
        To("..")
    Add("xy", name="l_origin_bug_corrector-don't_remove", autoadd=False)
    To("l_origin_bug_corrector-don't_remove")
    Set("xData", "[0]")
    Set("yData", "[0]")
    Set("hide", True)
    Set("xAxis", "xL")
    Set("yAxis", "yL")
    Set("MarkerLine/width", "1.5pt")
    Set("MarkerLine/hide", False)
    To("..")

    # cum_k_scale = 0
    for ii, pid in enumerate(fv.xy_or_y(["_Wind"], graphs, use_x_if=device_wind)):
        is_wind = pid == "_Wind"
        # cum_k_scale += k_scale
        cum_k_scale_vsz = "DISP_vecY0_distribute*" + fv.xy_sel(
            "1",
            f"sum(list(Disp_vectors_scale_height.values())[:{ii}])",
            use_x_if=device_wind,
            use_y_if=ii > 0,
            f_xy="({} + {})".format,
        )
        # print("cum_k_scale =", cum_k_scale)
        if not is_wind:
            Add("label", name=f"l_device{pid}", autoadd=False)
            To(f"l_device{pid}")
            Set("label", f"%{{{{DISPdevice['{pid}']}}}}%")
            Set("xPos", l_device_x)
            Set(
                "yPos",
                f"float64(eval(str(SETTING('/_vectors/grid1/v/vectors{pid}_lite/yPos')))) + DISP_legY0dev/{n_graphs}",
            )
            Set("positioning", "axes")
            Set("xAxis", "xL")
            Set("yAxis", "yL")
            Set("alignHorz", "left")
            Set("alignVert", "centre")  # top
            Set("margin", "1pt")
            Set("Text/font", "Arial")
            Set("Background/color", "#ffffc8")
            Set("Background/transparency", 70)
            To("..")

        if False:
            t0sfx = "" if use_bins[bin0name] else pid
            Add("line", name=f"vectors_2s{pid}", autoadd=False)
            To(f"vectors_2s{pid}")
            Set("arrowright", "arrownarrow")
            Set("arrowSize", "1pt")
            Set("mode", "length-angle")
            Set("hide", True)
            Set(
                "xPos",
                f"v.dt64s2vsz(1E-9*t_ns{t0sfx}{'' if b_one_table else pid}[sl_(iu{pid})]) "
                "+ USE_timeShift_s",
            )
            Set(
                "yPos",
                f"1 - {cum_k_scale_vsz} + DISP_vecY0/{n_graphs}",
            )
            Set(
                "length",
                f"absolute(u{pid}+1j*v{pid})[sl_(iu{pid})]*DISPscale_page_vectors",
            )
            Set("angle", f"degrees(arctan2(u{pid}, v{pid})[sl_(iu{pid})])-90")
            Set("xPos2", "bin2_u")
            Set("yPos2", "bin2_v")
            Set("yAxis", "y[0,1]")
            Set("Line/color", "black")
            Set("Line/width", "0.5pt")
            Set("Line/transparency", 50)
            Set("Fill/color", "black")
            To("..")

        # Vector display parameters for mean_lite, mean and background (top to bottom z-order)
        bin_main = list(use_bins)[-1]  # 'bin2_'

        # b_lite_in_fg = is_wind or (len(bin_lite) == len(bin_main) and bin_lite >= bin_main)
        bin_bg_wind = ""
        shift = round(
            np.fmin(
                disp_dtime_range_s / (cfg_plot.graph_width * 100),
                (wind_bin_average_s if is_wind else use_bins[bin_main]) / 2,
            ).item()
        )
        disp_vec_all = {  # :
            "suffix": ("_lite", "", "_bg"),
            "bin": (
                bin_main,  # Draw lite vectors (averaged by of ``bin2``) in foreground:
                bin_main,
                bin_bg_wind if is_wind else bin_lite,
            ),
            "arrow": ("none", "arrownarrow", "none"),
            "shift": ([f" + {shift}"] * 2 + [""]),
            "color": ("cyan", "blue", "yellow")
            if is_wind
            else ("red", "black", "#ff7850"),
            "transp": (30, 0, 0),
            "width": ("0.3pt", "0.7pt", "0.3pt"),
        }
        for z_order in range(3):  # mean_lite, mean and background
            if disp_vec_all["bin"][z_order] is None:
                continue
            disp_vec = namedtuple("disp_vec", disp_vec_all.keys())(
                *[v[z_order] for v in disp_vec_all.values()]
            )
            pid_for_time = "" if use_bins[disp_vec.bin] > 0 and b_one_table else pid
            Add("line", name=f"vectors{pid}{disp_vec.suffix}", autoadd=False)
            To(f"vectors{pid}{disp_vec.suffix}")
            print(
                pid_for_time,
                disp_vec.suffix,
                disp_vec.bin,
                disp_vec.arrow,
                disp_vec.shift,
                disp_vec.color,
            )
            Set("arrowright", disp_vec.arrow)
            Set("arrowSize", "1pt")
            Set("hide", False)
            Set(
                "yPos",
                (
                    f"float64(eval(str(SETTING('/_vectors/grid1/v/vectors{pid}_lite/yPos'))))"
                    if z_order
                    else f"1 - {cum_k_scale_vsz} + DISP_vecY0/{n_graphs}"
                ),
            )
            if is_wind:
                Set(
                    "xPos",
                    f"bin2_t0st_Wind + (WIND_bin_average_s - diff(time_Wind[:2]))/2{disp_vec.shift}",
                )
                Set(
                    "length",
                    "absolute(bin2_u_Wind+1j*bin2_v_Wind)*WINDscale_page_vectors",
                )
                Set("angle", "degrees(arctan2(bin2_u_Wind, bin2_v_Wind)) - 90")
            elif disp_vec.bin:  # Vabs and Vdir should be defined
                Set("xPos", f"{disp_vec.bin}t0st{pid}{disp_vec.shift}")
                Set("length", f"{disp_vec.bin}Vabs{pid}*DISPscale_page_vectors")
                Set("angle", f"{disp_vec.bin}Vdir{pid} - 90")
            else:
                Set(
                    "xPos",
                    f"v.dt64s2vsz(1E-9*t_ns{pid_for_time}[sl_(*iu{pid})]) + "
                    f"USE_timeShift_s{disp_vec.shift}",
                )
                Set("length", f"absolute(u{pid} + v{pid})*DISPscale_page_vectors")
                Set("angle", f"degrees(arctan2(u{pid}, v{pid})) - 90")
            Set("yAxis", "y[0,1]")
            Set("Line/color", disp_vec.color)
            Set("Line/width", disp_vec.width)
            Set("Line/transparency", disp_vec.transp)
            Set("Fill/color", "black")
            To("..")

        Add("xy", name=f"autorange_as_axis{pid}", autoadd=False)
        To(f"autorange_as_axis{pid}")
        Set("marker", "none")
        Set("markerSize", "2pt")
        Set("color", "darkred")
        Set("xData", "v.dt64s2vsz(int32(array(DISPtime[0], 'M8[s]')))")
        Set(
            "yData",
            f"zeros(2) + float64(eval(str(SETTING('/_vectors/grid1/v/vectors{pid}_lite/yPos'))))",
        )
        Set("hide", False)
        Set("yAxis", "y[0,1]")
        Set("PlotLine/color", "grey")
        Set("PlotLine/width", "0.5pt")
        Set("PlotLine/style", "dotted")
        Set("MarkerLine/width", "1.5pt")
        Set("MarkerLine/hide", False)
        Set("MarkerFill/hide", False)
        To("..")
    Add("axis-function", name="xShow - do not limit", autoadd=False)
    To("xShow - do not limit")
    if str_units_added:
        # Transparent word of length equal to xUnits 1st string to remove tick labels under
        Set(
            "label",
            "".join(
                [
                    r"%{{\color{transparent}{(",
                    str_units_added,
                    r").split('\\')[0])}}%}\\",
                ]
            ),
        )
    else:
        Set(
            "label",
            "".join(
                [
                    "%{{fDisp_date_u('/_vectors/grid1/x', 'disp_time_span'",
                    ", b_nl=True" if x_units_nl else "",
                    ", allow3rows=True" if cfg_plot.graph_width < 10 else "",
                    ", higher=1)}}%",
                ]
            ),
        )
    Set("linked", True)
    Set("linkedaxis", "x")
    Set("autoMirror", False)
    Set("match", "x")
    Set("Label/position", "at-maximum")
    Set("GridLines/hide", True)
    x_datetime_ticks(False, cfg_plot=cfg_plot)
    To("..")
    To("..")
    To("..")
    To("..")


# ##########################################################################################################
## prepare pg_1d() #########################################################################################
# ##########################################################################################################

disp_param = {
    "Vabs": "|V|",
    "Vdir": "Vdir,°",
    "t": "t,°C",
    "Temp": "t,°C",  # t and Temp are the same
    "u": "u",
    "v": "v",
    "u-shore": "V_{⃦ }",
    "v-shore": "V_{⊥}",
    "P": "P",
    "dP": "dP",
    "VabsWind": "|W|",  # Wabs
    "uWind": "u{_w}",  # Wu
    "vWind": "v{_w}",  # Wv
}


def get_param_expr_dict(bin="", prefix="", suffix="", wrap_dir="disp_central_dir"):
    """
    dict of expressions for parameters ('Vabs', 'Vdir', 'u', 'v', 'u-shore', 'v-shore')
    :param bin:
    :param prefix:
    :wrap_dir: falsy or str. If use default "disp_central_dir" then this var should be defined in Veusz
    """
    u = f"{prefix}{bin}u{suffix}"
    v = f"{prefix}{bin}v{suffix}"
    return {
        "Vabs": f"absolute(1j*{u} + {v})",
        "Vdir": f"v.wrap_dir(degrees(arctan2({u}, {v})), {wrap_dir})"
        if wrap_dir
        else f"degrees(arctan2({u}, {v})) % 360",
        "u": u,
        "v": v,
        "u-shore": f"{u}*cos(radians(Dir0proj)) - {v}*sin(radians(Dir0proj))",
        "v-shore": f"{v}*cos(radians(Dir0proj)) + {u}*sin(radians(Dir0proj))",
    }


is_antisymmetric = lambda p: p[1:4] not in ("abs", "dir") and p[0] in "PVWuv"


def get_y_lims_default(ax_max, param, b_wind):
    """
    Y axis limts values without account for relative scaling on page (`scale_y`)
    :param ax_max:
    :param param:
    :param b_wind:
    globals:
    - disp_param - dict having field with name equal to ``param`` having str value to label axis
    - axis_min, axis_max - dicts
    - v_to_graph_h: float
    :return: ax_max, ax_min
    """
    if param == "Vabs":  # to set velocity projections axis limits separately below
        if ax_max is None:
            ax_max = v_to_graph_h
        ax_min = axis_min.get(param, -0.01 * ax_max)
    else:
        if ax_max is None:
            # can be less than max in one side, so todo: use actual data lims
            ksc = 1  # scale axis_max["Vabs"] with sqrt(2) = 0.7071 not always good
            ax_max = axis_max.get(
                param,
                axis_max["dP"]
                if param == "P"
                else round(
                    ksc
                    * (
                        axis_max.get(
                            "VabsWind",
                            axis_max.get("Vabs", v_to_graph_h)
                            / cus.Wind_to_current_coef,
                        )
                        if b_wind
                        else axis_max.get("Vabs", v_to_graph_h)
                    ),
                    2,
                )
                if param[0] in "uv"
                else None,
            )
        if is_antisymmetric(param):
            ax_min = -ax_max
        else:
            ax_min = axis_min.get(param, 0)
    return ax_min, ax_max


# #######################################################################################################


def pg_1d(
    graphs,
    param="Vabs",
    zoom=False,
    i_show_legend=None,
    scale_height=None,
    cfg_plot=None,
):
    """
    :param graphs: device names in order. Will be used as result graphs names. w* graphs not used if param not
        "Vabs" or "t"
    :param param: str, one of params from global ``disp_param``, but if it is 'Vabs' then for w* graphs 'P' is
    used instead if contains '&' then will be splitted and pasted with different color on each graph relative
    to default axis for 1st param
    :param i_show_legend: dict {x, {VeuszParameterName: VeuszParameterValue}}, show legend only for this
    graphs, hide key if id not listed
    :param scale_height: dict {graph index: scale graph height coefficient relative to some default value} to
    change height of graphs keeping its scale, for example:
    Example:
    pg_1d(['_i03','_i10','_i33','_i09','_i28','w01'], zoom=False, i_show_legend={
        '_i01': {'vertManual': 0.2},
        '_i28': {'vertManual': 0.6},
        '_w01': {'vertManual': 0.6}
        },
        scale_height={'_i38': 1.4}
        )
    """
    pg_name = f"_zoom!{param}" if zoom else f"_{param}"
    print(f"Page {pg_name}", end=": ")

    # make legend wide by using many columns to strain it to 1 line if small vertical space
    b_wide_labels = False  # | True
    # | WidthGrade > WidthGrades["VeryNarrow"] and param != 'Vdir'  # len(graphs) > 5

    leg_cols_setting = {
        "columns": 3
    }  # good to align bin interval keys on multiparameter grahphs
    if b_wide_labels:
        replacing_wide_labels = ".replace(r'\\\\', '. ')"
    elif WidthGrade <= WidthGrades["Narrow"]:
        leg_cols_setting = {}
    replacing_wide_labels = (
        ".replace(':\u2009', r'\\\\').replace(',', r',\\\\')" if param == "Vdir" else ""
    )

    # Set device info labels to left of y axis (because we have short label text?)
    label_left_to_ax = bool(common_point_for_all(graphs))
    if param[0] in "uvV" and param != "Vdir":
        # preliminary set:
        need_p_ECMWF = param == "Vabs" and device_wind == "ECMWF"  # not bare "wind"
        # may add "and not zoom" to not draw P from ECMWF if zoom

        if device_wind and not zoom:
            graphs = ["_Wind"] + graphs
        if param[0] == "V":
            n_graphs_w = len([g for g in graphs if g.startswith(("_w", "_p"))])
            if n_graphs_w == 0:
                if need_p_ECMWF:
                    n_graphs_w = 1
                    graphs += ["_P"]
            if n_graphs_w:
                pg_name = f"{pg_name},P"
        else:
            n_graphs_w = 0
            graphs = [g for g in graphs if not g.startswith(("_w", "_p"))]

    else:  # no more parameters left for wave gauges
        n_graphs_w = 0
        graphs = [g for g in graphs if not g.startswith(("_w", "_p"))]
        if param == "t":
            param = "Temp"
            axis_min["Temp"] = axis_min["t"]
            axis_max["Temp"] = axis_max["t"]
        need_p_ECMWF = False

    grid_leftMargin = cfg_plot.grid_horMargins_sum
    grid_bottom = 0.8 if zoom else cfg_plot.grid_bottomMargin  # cm
    graphs_height_sum, scale_height, scale_height_common = scale_rows(
        scale_height if (param[0] in "uvV" and param != "Vdir") else {},
        scale_i_graphs=axis_max["Vabs"] / v_to_graph_h
        if (param[0] in "uvV" and param != "Vdir")
        else 1,
        n_graphs=len(graphs),
        n_graphs_w=n_graphs_w,
    )
    if scale_height:
        scale_heights = [
            scale_height.get(
                g, 1 if g.startswith(("_p", "_w", "_W")) else scale_height_common
            )
            for g in graphs
        ]
    else:
        scale_heights = [1] * len(graphs)

    # Show binning legend
    if WidthGrade == WidthGrades["VeryNarrow"] and not zoom:
        i_show_legend = {}
    elif i_show_legend is None:
        # Settings to put legend on 1st found graph in each of ids_i, cus.USE_bursts, ids_w graphs.

        # If graph is small then put lower by multiplying on coefficient:
        # to top need add legend header + rows heights  # [1, 0.92, 0.85, 0.79]
        k_lower = 1 if b_wide_labels else 12 / (10 + len(graphs))

        i_show_legend = dict(
            pid_sel
            for ids_check, vertManual in zip(  # legend bottom edge for normal incl, burst-mode and wavegages
                (ids_i, cus.USE_bursts, ids_w),
                (
                    0.2 / scale_heights[0]
                    if ids_i
                    else 0,  # for legend of 1st used i-graph
                    0.5,
                    -0.02,
                ),  # (-1.8, 0.5, 0.75): value < 1 will begin move legend below current graph
            )
            for pid_sel in next(  # get 1st of ids_check in graphs order
                (
                    [
                        (
                            pid,
                            {"vertManual": vertManual * k_lower, **leg_cols_setting},
                        )
                    ]
                    for pid in graphs
                    if pid in ids_check
                ),
                [],
            )
        )

    Add("page", name=pg_name, autoadd=False)
    To(pg_name)
    Set("height", f"{graphs_height_sum + grid_bottom:g}cm")
    if zoom:
        x_name = "xLong"
        cur_graph_width = 415 - cfg_plot.grid_horMargins_sum
        Set("width", "415cm")
        Set("Background/color", "white")
        Set("Background/hide", False)
        str_units_added = False
    else:  # no zoom
        x_name = "x"
        cur_graph_width = cfg_plot.graph_width
        Set("width", f"{cur_graph_width + cfg_plot.grid_horMargins_sum:g}cm")

        str_units_y_cm = (
            1.8 if WidthGrade == WidthGrades["Narrow"] and not x_units_nl else 0.3995
        )  # 0.4509
        str_units_added = label_xUnits_add(
            f"/{pg_name}/grid1/{x_name}",
            graphs_height_sum,
            y_cm=str_units_y_cm,
            force=True,
            cfg_plot=cfg_plot,
        )  # 0.06 /scale_height.get(pid, 1)

        t = common_point_for_all(graphs, n_curly_braces=2)
        _, __ = ('v.pl(f"', '")') if len(graphs) > 1 else ('f"', '"')
        split_before_date = n_graphs_w and WidthGrade < WidthGrades["Narrow"]
        sentences = ([f'f"{t}. "'] if t else []) + [
            "".join(
                [
                    _,
                    (
                        "{{current velocity}} "
                        if param != "Temp"
                        else "{{temperature}} {{of microprocessor}} "
                    ),
                    "{{by {info_incl['device']}}}",
                    (
                        ''' {LANG({'default': 'with pressure sensor', 'ru': 'c датчиком давления'})} "'''
                        if ids_p
                        else ""
                    ),
                    __,
                ]
            )
        ]
        add_label_Title(
            sentences=sentences,
            split_before_date=split_before_date,
            cfg_plot=cfg_plot,
            graphs_height_sum=graphs_height_sum,
            grid_leftMargin=grid_leftMargin,
            **(  # if we have extra wave gauges row
                {
                    "text2": '''f"{info_pres['measure']} {{by_}} {info_pres['device']}"''',
                    "param2_trange": "time_span_w",
                }
                if param == "Vabs" and len(ids_w) > len(ids_p)
                else {}
            ),
            text_add=(
                '''f"{{{info_wind['nature']}}} {{by}} {{{info_wind['device']}}}"'''
                if "_Wind" in graphs
                else ""
            ),
            y_cm=None
            if any(r"\\" in s for s in sentences) or split_before_date
            else str_units_y_cm,
            # lift text to the Units level if no \\ at all,
        )
        if info_wind_show != "''":
            Add("label", name="info_add", autoadd=False)
            To("info_add")
            Set("label", r"\bold{%{{info_wind['show']}}%}")
            Set("hide", False)
            Set("xPos", [0.075])  # 0.756
            Set("yPos", [0.55])  # 0.99
            Set("positioning", "relative")
            Set("xAxis", "xL")
            Set("yAxis", "yL")
            Set("alignHorz", "left")
            Set("alignVert", "top")
            Set("margin", "1pt")
            Set("Text/font", "Arial")
            Set("Background/transparency", 10)
            To("..")

    b_dense = disp_dtime_range_s / cur_graph_width > 100000

    Add("grid", name="grid1", autoadd=False)
    To("grid1")
    Set("rows", len(graphs))
    Set("bottomMargin", f"{grid_bottom}cm")
    Set("leftMargin", f"{cfg_plot.grid_horMargins_sum}cm")
    Set(
        "rightMargin",
        "0.5pt"
        if cfg_plot.grid_horMargins_sum == grid_leftMargin
        else f"{cfg_plot.grid_horMargins_sum - grid_leftMargin:g}cm",
    )
    Set("topMargin", "0.5pt")
    Set("internalMargin", "0pt")
    Set("scaleRows", scale_heights)

    def add_rect_blank_between(graph_name, scaling):
        Add("rect", name="blank_between", autoadd=False)
        To("blank_between")
        Set(
            "xPos",
            f'float64(SETTING("/{pg_name}/grid1/{graph_name}/blank_between/width")) / 2.4',
        )
        Set(
            "yPos",
            f'1 - float64(SETTING(f"/{pg_name}/grid1/{graph_name}/blank_between/height")) / 2',
        )
        Set(
            "width", [round(0.011 * cfg_plot.graph_width_standard / cur_graph_width, 5)]
        )  # 0.00057
        Set("height", [round(0.02 * (scale_height_common or 1) / scaling, 5)])
        Set("positioning", "relative")
        # "axes"
        # Set("xAxis", "xL")
        # Set("yAxis", "y[0,1]")
        Set("Fill/color", "white")
        Set("Fill/hide", False)
        Set("Border/hide", True)
        To("..")

    def tick_width_any(range, ticks):
        """
        gives any single digit in some power of 10, for example: 4,6,8,9
        :param range:
        :param ticks:
        :return:
        """
        bad_width = range / ticks
        decimals = np.ceil(np.log10(bad_width) - 1)
        div = 10**decimals
        return round(
            ceil(bad_width / div) * div, int(-decimals)
        )  # prevent round-off errors

    def tick_width(range, target_steps):
        temp_step = range / target_steps
        mag_pow = 10 ** np.floor(np.log10(temp_step))
        mag_msd = temp_step / mag_pow
        if mag_msd > 5:
            mag_msd = 10
        elif mag_msd > 2.5:
            mag_msd = 5
        elif mag_msd > 2:
            mag_msd = 2.5
        elif mag_msd > 1:
            mag_msd = 2
        else:
            mag_msd = int(mag_msd + 0.5)
        return mag_msd * mag_pow

    def add_axis_y(
        scale_y: float, ax_max=None, param: str = param, axis_name=None, axis_label=None
    ):
        """
        Adjust axis limits, set axis label

        :param scale_y: coefficient to scale default axis limits, not scale if None
        :param ax_max: maximum axis value, if None then will be set to some default maximum for param (see
            code)
        :param param: parameter name that determines axis properties. Should be in disp_param. Constant
            ticks/scale will be for param=Vdir.
        :param axis_name: axis name. If none (default) then name will be y{param}
        :param axis_label: label for axis exluding units for param if it is in info_...
        :returns: axis name
        globals:
        - disp_param - dict having field with name equal to ``param`` having str value to label axis
        - axis_min, axis_max - dicts
        - v_to_graph_h: float
        - is_antisymmetric(param): function
        """
        if axis_name is None:
            axis_name = f"y{param}"
        Add("axis", name=axis_name, autoadd=False)
        To(axis_name)
        if param.startswith("Vdir"):
            Set("label", disp_param.get(param, param))
            Set("datascale", 0.11111111111111)
            Set("TickLabels/scale", 9.0)
            Set("MajorTicks/number", 4)
            Set("MinorTicks/number", 8)
        else:
            b_wind = param.endswith("Wind")  # old: param[0] == "W"
            if param == "P":
                Set(
                    "label",
                    ",\\\\".join(
                        [
                            "P-❬P❭"
                            if zoom or n_graphs_w > 1
                            else "P-❬P❭\\\\\\color{magenta}{P_{a}-❬P_{a}❭}",
                            "%{{info_pres['units']}}%",
                        ]
                    ),
                )
                # - simpler to not show ECMWF label at all if many graphs as we draw ECMWF only on 1st
            else:
                units = (
                    (
                        ",{}%{{{{{}['units']}}}}% ".format(
                            "\u2009"
                            if len(param) < 5
                            else r"\\",  # new line for long param.
                            "info_wind" if b_wind else "info_incl",
                        )
                    )
                    if param[0] in "uvV"
                    else ""
                )
                Set("label", f"{axis_label or disp_param[param]}{units}")

            ax_min, ax_max = get_y_lims_default(ax_max, param, b_wind)
            if scale_y and scale_y != 1:
                b_antisymmetric = is_antisymmetric(param)
                print(param, f"axis lims (scaled to {scale_y}):", ax_min, ax_max)
                if ax_min is not None:
                    # Set min to scaled default min only if antisymmetric (else it is a constant support)
                    Set("min", ax_min * scale_y if b_antisymmetric else ax_min)
                if ax_max is not None:
                    # If not antisymmetric then we set its max to diff of its scaled default limits,
                    Set(
                        "max",
                        ax_max * scale_y - (0 if b_antisymmetric else (ax_min or 0)),
                    )

                Set("MajorTicks/number", max(int(3.9 * scale_y), 2))
                Set("MinorTicks/number", max(int(6 * scale_y), 3))
            else:
                print(param, "axis lims:", ax_min, ax_max)
                if ax_min is not None:
                    Set("min", ax_min)
                if ax_max is not None:
                    Set("max", ax_max)

            if param[0] in "uvV":
                n_ticks = 6  # number of ticks for not scaled axis to determine standard tick width
                major_step = tick_width((ax_max - ax_min), 6)  # 0.2
                Set("MinorTicks/number", int(50 * major_step))
                # Not possible to set equal ticks for axes of different length by using only *Ticks/number so this:

                # Min (from which with major_step we can cross 0)
                _ = major_step * (ax_min * (scale_y or 1) // major_step)
                ticks = np.around(
                    np.arange(_, ax_max * (scale_y or 1) + major_step, major_step), 3
                )
                Set("MajorTicks/manualTicks", ticks.tolist())

                n_ticks *= 2  # initial number of minor ticks
                while True:
                    minor_step = tick_width(ax_max, n_ticks)
                    rem = major_step % minor_step
                    if (
                        rem < 1e-6 or abs(rem - minor_step) < 1e-6
                    ):  # second because of round-off errors
                        break
                    n_ticks = n_ticks + 1
                # Min (from which with minor_step we can cross 0)
                _ = minor_step * (ax_min * (scale_y or 1) // minor_step)
                ticks = np.setdiff1d(
                    np.around(
                        np.arange(
                            _ - major_step,
                            ax_max * (scale_y or 1) + major_step,
                            minor_step,
                        ),
                        4,
                    ),
                    ticks,
                )
                Set("MinorTicks/manualTicks", ticks.tolist())
            else:
                Set("MajorTicks/number", 6)
                Set("MinorTicks/number", 12)
        Set("autoMirror", False)
        # if len(graphs) > 10:
        # Set('Label/size', '12pt')
        # Set('TickLabels/size', '12pt')
        Set("Label/atEdge", True)
        Set("Label/rotate", "90")
        Set("MinorTicks/hide", False)
        Set("MinorGridLines/hide", False)
        To("..")
        return axis_name

    # Different colors for each param on common graph
    params = param.split("&")
    clr_param_bins = {  # for use with Current/Wind as `clr_param_bins[bin if pid != "_Wind" else pid]`
        "": (("yellow", "cyan") if b_dense and not zoom else ("red", "#0088ff"))
        if "bin2_" in use_bins
        else ("red", "blue"),  # |("black", "red"),
        "bin2_": (
            ("#990000", "black" if len(params) == 1 else "#000099")
            if "" in use_bins
            else ("#ffaa00", "#00aaff")  # ?
        ),
        "_Wind": ("black", "blue"),
    }
    clr_param_light_darker_dots = (
        ["red"] if len(params) == 1 else ["#ff7777", "#77ff77"]
    )
    # Cycle adding graphs
    #####################
    dt1d = 3600 * 24 * 30  # 1D
    # param_w = param ?
    keys_shown = set()
    for ii, (pid, scaling) in enumerate(zip(graphs, scale_heights)):
        # binB - packet start interval
        binB = "binB"
        # binAB - intermediate average interval
        if pid.startswith(("_p", "_w", "_P")):
            # Pressure probe => display ``P`` instead of water velocity parameters
            binAB = (
                "bin_"
                if zoom or pid.lower().startswith("_p")
                else f"{list(use_bins_w)[-1]}"
            )  # no binB defined for wave gauges so use max bin
            binAB_color = "black"
            binB = bin_burst_name[:-1]  # bin2
            if pid.startswith("_w"):
                w_opt = "_w"
                pid_for_time = pid
            else:
                w_opt = ""
                pid_for_time = ids_ip[ids_p.index(pid)] if pid != "_P" else ""
            params_cur = (
                ["P" if p == "Vabs" else p for p in params] if n_graphs_w else params
            )
            #  param_w  # ?  changed at bottom to 'P' if param was 'Vabs'
            graph_name = params_cur[0]
        else:
            if pid in cus.USE_bursts:
                binAB = "binB_"
                binAB_color = "#0000ff" if zoom else "#00ffff"
            else:
                binAB = "bin_"
                binAB_color = "#aaaaff" if zoom else "#00ff00"  # contrasted to bin2
            w_opt = ""
            pid_for_time = pid
            params_cur = params
            graph_name = param

            if pid == "_Wind":
                # Wind data => display corresponding wind parameters instead of water velocity parameters
                params_cur = [f"{p}Wind" for p in params]
                graph_name = params_cur[0]

        y_axis_name = f"y{graph_name}"
        # +change graph_name to can reference in Veusz (due to Veusz looks at variables and ops &+-... - bug)
        graph_name = "{}{}".format(
            graph_name.replace("&", ";")
            if "&" in graph_name
            else graph_name.removesuffix("_Wind"),
            pid,
        )
        print(graph_name, end=",")

        t0sfx = w_opt if b_one_table and use_bins[bin0name] else pid_for_time

        Add("graph", name=graph_name, autoadd=False)
        To(graph_name)

        # Set colorful axis_label for multiple params ("u&v" to "{\\color{black}{u}}, {\\color{red}{v}}")
        axis_label = (
            ",\u2009".join(
                [
                    rf"\color{{{clr}}}{{{disp_param[p]}}}"
                    for p, clr in zip(
                        params_cur,
                        clr_param_bins[
                            ("bin2_" if "bin2_" in use_bins else "")
                            if pid != "_Wind"
                            else "_Wind"
                        ],
                    )
                ]
            )
            if "&" in param
            else None
        )

        Add("label", name="l_device", autoadd=False)
        To("l_device")
        Set("label", f"%{{{{DISPdevice['{pid_for_time}']{replacing_wide_labels}}}}}%")
        Set("hide", False)
        Set(
            "xPos",
            0.0011
            if zoom
            else -grid_leftMargin / cur_graph_width
            if label_left_to_ax or param == "Vdir"
            else 0.01,
        )
        # if narrow then 0.01 is usually where data start - bad place
        # print(scale_height_common, '!', (scale_height_common or 1) * graph_h_default / scaling, scaling)
        # try put under axis label: on center (near 0 for Vdir) else slitly above
        Set(
            "yPos",
            (
                1
                - ((0.25 if param == "Vdir" else 0.15) if label_left_to_ax else 0.01)
                * (scale_height_common or 1)
                / scaling
            ),
        )
        Set("positioning", "relative")
        Set("xAxis", "xL")
        Set("yAxis", "yL")
        Set("alignHorz", "left")
        Set("alignVert", "top")
        Set("margin", "1pt")
        Set("Text/font", "Arial")
        if param == "Vdir":  # WidthGrade == WidthGrades["VeryNarrow"] or
            Set("Text/size", "10pt")
        Set("Background/color", "#ffffc8")
        Set("Background/transparency", 70)
        To("..")
        if is_antisymmetric(param):
            Add("xy", name="y=0", autoadd=False)
            To("y=0")
            Set("marker", "none")
            Set("markerSize", "2pt")
            Set("xData", "disp_time_span")
            Set("yData", "[0, 0]")
            Set("yAxis", y_axis_name)
            Set("PlotLine/color", "black")
            Set("PlotLine/width", "0.5pt")
            To("..")

        if pid in i_show_legend:
            _ = f"legend{'_z' if zoom else ''}"
            Add("key", name=_, autoadd=False)
            To(_)
            if len(use_bins) == 1:
                Set("hide", True)
            if "title" not in keys_shown:
                keys_shown.add("title")
                Set("title", "%{{v.c1(I['averaging bin'])}}%")
            Set(
                "horzManual",
                0.02
                if zoom
                # 1.532685e-6*disp_dtime_range_s + (-4.057 if b_wide_labels else -3.6)  # right edge
                else (
                    (cur_graph_width - (9 if b_wide_labels else 3))
                    / cur_graph_width  # right edge minus legend width
                ),
            )  # if (len(i_show_legend) > 1 or i_show_legend[pid].get('vertManual') > 0) else -0.01
            #     move one low legend to the lower left

            Set("Background/transparency", 20)
            Set("horzPosn", "manual")
            Set("vertPosn", "manual")
            for k, v in i_show_legend[pid].items():
                Set(k, v)
            To("..")
        if ii == 0:
            Add("axis-function", name="x_show_up", autoadd=False)
            To("x_show_up")
            Set("linked", True)
            Set("linkedaxis", x_name)
            Set("mode", "datetime")
            Set("otherPosition", 1.0)
            Set("TickLabels/hide", True)
            Set("MajorTicks/hide", True)
            Set("MinorTicks/hide", True)
            To("..")
        if (
            zoom
            and pid in cus.USE_bursts
            and "bursts_stretched_comment" not in keys_shown
        ):
            keys_shown.add("bursts_stretched_comment")
            Add("label", name="l_comment", autoadd=False)
            To("l_comment")
            Set(
                "label",
                "Данные, записанные в интервалах, растянуты между началами интервалов"
                if fv.lang == "ru"
                else "Data recorded in intervals is shown stretched between starts of the intervals",
            )
            Set("xAxis", "xL")
            Set("yAxis", "yL")
            Set("xPos", [0.004])
            Set("yPos", [1.0])
            Set("Text/size", "8pt")
            Set("positioning", "axes")
            Set("alignHorz", "left")
            Set("alignVert", "top")
            Set("margin", "1pt")
            Set("Background/transparency", 20)
            To("..")

        def add_xy(param, bin, axis_name: Optional[str] = None):
            """Start of xy initialization with parameter averaged by bin2, binAB, dt
            :param bin: bin with "_" suffix or ""
            :param axis_name: yAxis Veusz parameter name. If None (default) then sets to ``f'y{param}'``
            """
            param_expr_dict = get_param_expr_dict(bin=bin, suffix=pid)
            # Clip binned parameters and express in defined terms (if needed)
            if bin and param[1:4] in ("abs", "dir"):  # already clipped
                y_data = (
                    f"{bin}{param}{pid}"
                    if pid != "_Wind"  # "Wind" suffix in dict values olnly
                    else param_expr_dict[param.removesuffix("Wind")]
                )
            else:
                if param in ("P", "Temp"):
                    y_data = f"{bin}{param}{pid}"
                else:
                    y_data = param_expr_dict[
                        param
                        if pid != "_Wind"  # "Wind" suffix in dict values olnly
                        else param.removesuffix("Wind")
                    ]
                y_data = f"({y_data})[sl_({bin}iu{pid_for_time})]"
            only_finite = f"[isfinite({y_data})]" if zoom else ""
            half_bin_add = f" + {bin.removesuffix('_')}/2" if bin else ""
            x_data = (
                (
                    f"f(lambda t: v.stretch_time(t, {binB}_t0st{pid_for_time}) if '{pid_for_time}' in "
                    f"USE_bursts else t{half_bin_add}, "
                    + (  # function
                        f"{bin}t0st{pid_for_time}"
                        if bin
                        else (
                            f"v.dt64s2vsz(1E-9*t_ns{t0sfx}[sl_(*iu{pid_for_time})]{only_finite}) + "
                            "USE_timeShift_s"
                        )
                    )
                    + ")"  # argument
                )
                if (not bin or (use_bins.get(bin, 0) < use_bins[bin_burst_name]))
                and pid != "_Wind"  # Simpler expression for condition above for Wind:
                else f"{bin}{'t0st' if bin else 'time'}{pid_for_time}{only_finite}{half_bin_add}"
            )
            xy_name = f"{param.removesuffix('_Wind')}{bin.removesuffix('_')}{pid}"
            Add("xy", name=xy_name, autoadd=False)
            To(xy_name)
            Set("xData", x_data)
            Set(
                "yData",
                (f"f(lambda x: x[isfinite(x)], {y_data})" if zoom else y_data)
                + (f" - mean_P{pid}" if param == "P" else ""),
            )
            Set("xAxis", x_name)
            Set("yAxis", axis_name or f"y{param}")
            # Add key value (if xy has no key then for possibility to easy switch it on in GUI)
            if pid not in i_show_legend or bin not in keys_shown:
                if pid in i_show_legend:
                    keys_shown.add(f"{bin[:-1]}{w_opt}")
                Set(
                    "key",
                    "".join(
                        # (["%{{f('{:%H:%M:%S}'.format, f(array("] + (
                        #         ["DATA('",
                        #             bin[:-1],
                        #             w_opt,
                        #             "')"
                        #         ] if bin else [
                        #             "diff(DATA('t_ns",
                        #             t0sfx,
                        #             "')[1:3])*1E-9"
                        #         ]) + [", 'M8[s]').item))[1:]}}%"]
                        # ) if use_bins.get(bin, None) != 0 else
                        [
                            "%{{v.str_dt(DATA('",
                            bin[:-1] or "dt",
                            w_opt,
                            "'), LANG({'default': 'en', 'ru': 'ru'}))}}%",
                        ]
                    ),
                )

        if params_cur[0] == "P":
            Add("label", name="l_mean_P", autoadd=False)
            To("l_mean_P")
            txt_mean_P = f"❬P❭=%{{{{'%.1f' % DATA('mean_P{pid}')[0]}}}}%"
            Set(
                "label",
                (r"\\" if WidthGrade == WidthGrades["VeryNarrow"] else ", ").join(
                    [
                        txt_mean_P,
                        r"\color{magenta}{❬P_{a}❭="
                        "%{{'%.1f' % (nanmean(DATA('sp')[sl_(DATA('iu_Wind')[0])])*1E-4)}}%",
                    ]
                )
                if need_p_ECMWF and device_wind
                else txt_mean_P,
            )
            Set("xAxis", "xL")
            Set("yAxis", "yL")
            Set("xPos", [0.009 if zoom else 0.01])  # 0.014 0.6
            Set("yPos", [0.046])  # 0.5
            Set("positioning", "axes")
            Set("alignHorz", "left")
            Set("alignVert", "bottom")
            Set("margin", "1pt")
            Set("Background/transparency", 20)
            To("..")
        else:
            # Parameter averaged by bin2
            if "bin2_" in use_bins:
                if not zoom:
                    for p, p_clr in zip(
                        params_cur,
                        clr_param_bins["bin2_" if pid != "_Wind" else pid][
                            -len(params_cur) :
                        ],
                    ):
                        add_xy(p, "bin2_", axis_name=y_axis_name)
                        # Add('xy', name=f'<{disp_param[param]}>bin2', autoadd=False)
                        # To(f'<{disp_param[param]}>bin2')
                        # Set('xData', f'bin2_t0st{pid}[isfinite(bin2_{param}{pid})] + bin2/2')
                        # Set('yData', f'bin2_{param}{pid}[isfinite(bin2_{param}{pid})]')

                        Set("PlotLine/color", p_clr)  # #55ff00
                        Set("MarkerLine/width", "1pt")  # '1.5pt'
                        Set("MarkerLine/hide", False)
                        To("..")

                if param == "Vabs":
                    # Rectificated velocity (not used?)
                    _ = f"❬{disp_param[param]}❭D"  # {bin2.removesuffix('_')}{pid}
                    Add("xy", name=f"<{_}>bin2", autoadd=False)

                    To(f"<{_}>bin2")
                    Set("xData", f"bin2_t0st{pid}[isfinite(bin2_VabsD{pid})] + bin2/2")
                    Set("yData", f"bin2_VabsD{pid}[isfinite(bin2_VabsD{pid})]")
                    Set("hide", True)
                    Set("key", f"{_}{pid}")
                    Set("xAxis", x_name)
                    Set("yAxis", y_axis_name)
                    Set("PlotLine/color", "cyan")
                    Set("MarkerLine/width", "1.5pt")
                    Set("MarkerLine/hide", False)
                    Set("hide", True)
                    if zoom:
                        Set("PlotLine/width", "1pt")
                        Set("PlotLine/style", "dotted-fine")
                    To("..")

        bin0name_cur = (
            list(use_bins_w)[0] if pid.startswith("_w") else bin0name
        )  # if main_param == 'P'

        # Parameter averaged by binAB
        for i, (p, p_clr) in enumerate(zip(params_cur, (binAB_color, "#ff44ff"))):
            if binAB in use_bins and binAB != bin0name_cur:
                add_xy(p, binAB, axis_name=y_axis_name)
                Set("PlotLine/color", p_clr)
                Set("MarkerLine/hide", False)
                if zoom:
                    Set("PlotLine/width", "0.5pt")
                    Set("MarkerLine/width", "1.5pt")
                b_have_draw = True
            else:
                b_have_draw = False

            if i == 0 and not zoom and need_p_ECMWF and pid.startswith(("_p", "_w")):
                To("..")
                # need_p_ECMWF = False
                Add("xy", name="p_ECMWF", autoadd=False)
                To("p_ECMWF")
                Set("xData", "time_Wind")
                Set(
                    "yData",
                    "f(lambda x: x - nanmean(x), sp[sl_(iu_Wind)])*1E-4",
                )
                Set(
                    "key",
                    "%{{f('{:%H:%M:%S}'.format, f(ndarray.item, array(diff(DATA('time_Wind')[1:3]), "
                    "'M8[s]')))[1:]}}%",
                )
                Set("xAxis", "x")
                Set("yAxis", "yP")
                Set("PlotLine/color", "magenta")
                Set("MarkerLine/hide", False)
                b_have_draw = True

            if b_have_draw:
                Set(
                    "PlotLine/style",
                    "dotted-fine" if disp_dtime_range_s > dt1d or zoom else "dash3",
                )
                To("..")

        # Parameter averaged by dt
        if bin0name_cur == "":  # minimum bin is needed
            for p, p_clr, p_clr_dot in zip(
                params_cur,
                clr_param_bins[bin0name_cur if pid != "_Wind" else pid]
                if len(params) > 1
                else ["yellow"],
                clr_param_light_darker_dots,
            ):
                add_xy(p, bin0name_cur, axis_name=y_axis_name)
                Set(
                    "marker", "none" if len(params) > 1 else "linehorz"
                )  # marker can make all lines black
                Set("markerSize", "0.01pt" if zoom else "0.1pt")
                Set("MarkerLine/color", "#ffaa00")  # "color", "darkred"
                # Add transparency if dencity of points higher than 1/pixel:
                # ~0 for disp_dtime_range_s <= 1D, and 99 old: 95 for 1 Month, always <= 99 + account for graph width
                if zoom:
                    value = (
                        100
                        * (disp_dtime_range_s - 500)
                        / (disp_dtime_range_s + 20000)
                        * cfg_plot.graph_width_standard
                        / (415 - cfg_plot.grid_horMargins_sum)
                    )
                    transparency = int(value) if value > 0 else 0
                    Set("PlotLine/color", "red" if p_clr == "yellow" else p_clr)
                    Set("PlotLine/width", "0.25pt")
                    Set(
                        "PlotLine/transparency", transparency
                    )  # int(0.8 * transparency)
                    Set("MarkerLine/width", "0.25pt")
                    Set("MarkerLine/transparency", int(0.5 * transparency))
                else:
                    # todo: transparency for intervals - {3D: line 50 (but for Vdir 90), marker 80; > for >}
                    value = (
                        50
                        * (disp_dtime_range_s - 500)
                        / (disp_dtime_range_s + 20000)
                        * min(cfg_plot.graph_width_standard / cur_graph_width, 1)
                    )
                    transparency = (
                        int(value) if value > 0 else 0
                    )  # 0.1 px line of transparency 99% is invisible so make its max transparency 30%
                    # - int(1/(5/30 if pid in cus.USE_bursts else 15/60 if pid in ids_w else 1))
                    # theoretic: 5/30, 15/60 # practic: 50/90
                    Set("PlotLine/color", p_clr)  # and pid not in cus.USE_bursts
                    Set("PlotLine/width", f"{0.1 if b_dense else 0.5}pt")  # 0.05
                    Set("PlotLine/transparency", transparency)
                    Set("marker", "dot")
                    Set("MarkerLine/width", f"{0.1 if b_dense else 0.5}pt")
                    # more transp or may be better switch it off because too many colors:
                    Set(
                        "MarkerLine/transparency",
                        int(transparency * (1 if len(params) == 1 else 1.5)),
                    )
                if pid.startswith(("_p", "_w")):
                    Set("PlotLine/color", "cyan")
                    Set("MarkerLine/color", "blue")
                else:
                    Set("MarkerLine/color", p_clr_dot if b_dense else "black")
                Set("MarkerLine/hide", False)
                To("..")

        if param == "Vdir":
            Add("xy", name="autorange", autoadd=False)
            To("autorange")
            Set("xAxis", x_name)
            Set("yAxis", y_axis_name)
            Set("xData", "disp_time_span")
            Set("yData", "disp_central_dir + float64([-180, 180])")
            Set("PlotLine/hide", True)
            Set("MarkerLine/hide", True)
            Set("MarkerFill/hide", True)
            To("..")

        if zoom and pid in (cus.USE_bursts | set(ids_w)):
            Add("bar", name="packet_start", autoadd=False)
            To("packet_start")
            Set("lengths", (f"disp_ones_burst_st{pid}",))
            Set("posn", f"{binB}_t0st{pid}")
            Set("mode", "stacked")

            if ids_w and pid == ids_w[-1]:  # or 'packet start' not in keys_shown
                # if pid in i_show_legend:
                # keys_shown.add('packet start')
                Set("keys", ("packet start",))
            Set("xAxis", "xLong")
            Set("yAxis", "y[0,1]")
            Set("barfill", 0.001)
            Set("groupfill", 0.0)
            Set("BarFill/fills", [("solid", "#ffff7f", False)])
            Set("BarLine/lines", [("solid", "0.5pt", "#ffff00", False)])
            # hide if interval ~< 0.1cm:
            Set(
                "hide",
                cur_graph_width / (dtime_range_s / use_bins.get("binB_", 1e15)) < 0.1,
            )
            To("..")

        if ii > 0:
            add_rect_blank_between(graph_name, scaling)
        # Add axis
        if pid == "_Wind":
            add_axis_y(
                scale_y=None,
                param=params_cur[0],
                axis_name=y_axis_name,
                axis_label=axis_label,
            )
            # elif param == 'Vabs' and n_graphs_w:  # will be 'P' graphs
            #     add_axis_y(scale_y=None, param='P')  # not scale default P axis
        else:  # adding axes to all graphs (can not use common axes because rect can not overlap them)
            add_axis_y(
                scale_y=None if params_cur[0] == "P" else scaling,
                param=params_cur[0],
                axis_name=y_axis_name,
                axis_label=axis_label,
            )

        # last ii
        if pid == graphs[-1]:
            if (not zoom) and param != "Vdir":
                Add("xy", name="autorange", autoadd=False)
                To("autorange")
                Set("marker", "none")
                Set("markerSize", "2pt")
                Set("color", "darkred")
                Set("xData", "v.dt64s2vsz(int32(array(DISPtime[0], 'M8[s]')))")
                Set(
                    "yData",
                    f"mean_P{pid}" if pid.startswith(("_p", "_w")) else "zeros(2)",
                )
                Set("hide", False)
                Set("xAxis", x_name)
                Set("yAxis", y_axis_name)
                Set("PlotLine/hide", True)
                To("..")

            Add("xy", name="area_selected", autoadd=False)
            To("area_selected")
            Set("xData", "repeat(atleast_2d(area_time_start).T + array([0, 5])*60, 2)")
            Set(
                "yData",
                "[e for pid in ones_like(area_time_start) for e in (nan,pid,pid,nan)]",
            )
            # Set('hide', True)
            Set("xAxis", x_name)
            Set("yAxis", "y[0,1]")
            Set("Color/points", [])
            Set("PlotLine/color", "#ffff7f")
            Set("MarkerLine/width", "1.5pt")
            Set("MarkerLine/hide", False)
            Set("FillBelow/color", "#c4a000")
            Set("FillBelow/style", "forward diagonals")
            Set("FillBelow/hide", False)
            Set("FillBelow/transparency", 50)
            Set("FillBelow/linewidth", "6pt")
            Set("FillBelow/patternspacing", "10pt")
            To("..")

            Add("line", name="area_selected_starts", autoadd=False)
            To("area_selected_starts")
            Set("arrowleft", "arrowreverse")
            Set("arrowright", "none")
            Set("arrowSize", "2pt")
            # Set('hide', True)
            Set("xPos", "area_time_start")
            Set("yPos", [0.0])
            Set("length", 1.48 if not zoom else 5)
            Set("angle", [-90.0])
            Set("positioning", "axes")
            Set("xAxis", x_name)
            Set("yAxis", "y[0,1]")
            Set("Line/color", "green")
            Set("Line/width", "1.5pt")
            Set("Line/style", "dashed")
            Set("Fill/color", "black")
            To("..")

            Add("axis-function", name=f"{x_name}Show - do not limit", autoadd=False)
            To(f"{x_name}Show - do not limit")
            Set("linked", True)
            Set("linkedaxis", x_name)
            Set("match", "x")
            Set("Label/position", "at-maximum")
            Set("GridLines/hide", True)
            Set(
                "otherPosition", -0.028 / scaling
            )  # -len(graphs)/1005 <-> -0.05, 17 <-> -0.2
            if str_units_added:
                # Transparent word of length equal to xUnits 1st string to remove tick labels under
                Set(
                    "label",
                    "".join(
                        [
                            r"\color{transparent}{%{{(",
                            str_units_added,
                            r").split('\\')[0]}}%}\\",
                        ]
                    ),
                )
            x_datetime_ticks(zoom, cfg_plot=cfg_plot)
            To("..")

        # Right axis as bare line. Adding to each graph because if common then it is showed at last graph only
        Add("axis", name="y[0,1]", autoadd=False)
        To("y[0,1]")
        Set("hide", False)
        Set("min", 0.0)
        Set("max", 1.0)
        Set("autoMirror", False)
        Set("direction", "vertical")
        Set("otherPosition", 1.0)
        Set("Label/position", "at-maximum")
        # Set("TickLabels/color", "#55ff00")
        Set("TickLabels/hide", True)
        Set("MajorTicks/hide", True)
        Set("MinorTicks/hide", True)
        Set("GridLines/hide", True)
        To("..")
        To("..")  # next graph

    Add("axis", name=x_name, autoadd=False)
    To(x_name)
    Set("Line/transparency", 50)
    Set("Label/hide", True)
    Set("TickLabels/hide", True)
    Set("MajorTicks/hide", True)
    Set("MinorTicks/hide", True)
    Set("direction", "horizontal")
    x_datetime_ticks(zoom, cfg_plot=cfg_plot)
    To("..")

    Add("axis", name="xL", autoadd=False)
    To("xL")
    Set("hide", True)
    Set("min", -0.075)
    Set("max", 415 / 28 if zoom else 0.925)
    Set("autoMirror", False)
    Set("direction", "horizontal")
    Set("otherPosition", 0.9)
    Set("Line/color", "#00ff00")
    Set("GridLines/hide", True)
    To("..")
    Add("axis", name="yL", autoadd=False)
    To("yL")
    Set("hide", True)
    Set("min", 0.0)
    Set("max", 1.0)
    Set("autoMirror", False)
    Set("direction", "vertical")
    Set("TickLabels/color", "#55ff00")
    Set("MajorTicks/hide", True)
    Set("MajorTicks/manualTicks", [0.5])
    Set("MinorTicks/hide", True)
    Set("GridLines/hide", False)
    To("..")

    # param_w = 'P' if param == 'Vabs' and n_graphs_w else param
    To("..")

    # Add('label', name='xUnits', autoadd=False)
    # To('xUnits')
    # # if not str_units_added:
    # Set(
    #     'label', f"%{{{{fDisp_date_u('/{pg_name}/grid1/{x_name}', "
    #              f"'disp_time_span'{', b_nl=True' if x_units_nl else ''}"
    #              f"{', allow3rows=True' if WidthGrade == WidthGrades["VeryNarrow"] else ''})}}}}%"
    # , higher={not zoom}
    # )
    # Set('hide', False)
    # Set('xPos', [1.0])
    # Set('yPos', [0.027])
    # Set('alignHorz', 'right')
    # Set('alignVert', 'bottom')
    # Set('margin', '0pt')
    # Set('Background/transparency', 10)
    # To('..')

    To("..")
    print()


def pg_progress(graphs, clr_by="probe", aspect=1, b_dt_big=True, cfg_plot=None):
    """
    clr_by: str, one from:
    - 'probe' or '' or None: color by probe
    - 'abs', 'dir', todo: other param
    Example:
    pg_vectors(['_i03','_i10','_i33','_i09','_i28'])
    checks globals: device_wind - wherher wind is needed
    """
    global lim_str
    if device_wind:
        graphs = ["_Wind"] + graphs
        colors_local = [clr_wind] + colors
    else:
        colors_local = colors
    graphs = [pid for pid in graphs if not pid.startswith(("_p", "_w"))]
    n_graphs = len(graphs)
    if not n_graphs:
        return
    scale_wind_more_str = (
        "*Wind_to_current_coef" if device_wind and n_graphs > 1 else ""
    )

    # (
    #     ("#00aaff " if device_wind else "") +
    #     "#ff0000 magenta #990099 darkblue green #0faa90 #119a40 #11c000 #117070 "
    #     "#ff7070 #ffc000 #bf9a40 #0faa90 green darkblue #990099 magenta #8b0002 #604020 #000000"

    # ).split()[:n_graphs]  # to preferably use last part as it darker
    # bin_dt = next(iter(use_bins.values()))

    # Set Veusz exressions for `graphs` and add correspoinding `pid`s to `have_data_for_pg_progress` if absent

    for pid, clr in zip(
        graphs, colors_local
    ):  # colors (f"disp_{bin}t_txt{pid}") for dateLabels
        bin = "bin2_" if pid == "_Wind" else "bin_"
        if pid in have_data_for_pg_progress:
            continue
        for u in "uv":
            for bin_cur in [bin, ""]:
                SetDataExpression(
                    f"{bin}{u}_cum{pid}",
                    (
                        "append(0, cumsum(f(lambda x: v.rep2mean(x, isfinite(x)), "
                        f"{bin}{u}{pid}[sl_({bin}iu_cmn{pid})])*{bin[:-1]})"
                        f"{scale_wind_more_str if pid == '_Wind' else ''})"
                    ),
                    linked=True,
                )  # *ediff1d({bin}t0st{pid}, )
            # SetDataExpression(
            #     f"{u}_cum_{pid}",
            #     "append(0, cumsum("
            #     f"{u}{pid}[sl_(iu{pid})])*{bin[:-1]}{scale_wind_more_str if pid == '_Wind' else ''})",
            #     linked=True,
            # )
        # for `Progress_lbl_dt` following is not correct if it contains parts before and after floating point

        SetDataExpression(
            f"disp_{bin}i{pid}",
            "(lambda t: v.i_whole_time_intervals(t, asscalar(diff(t[[0,-1]])/Progress_lbl_dt))[1:-1] if "
            "isinstance(Progress_lbl_dt, (int, float)) else flatnonzero(diff(int8(array(t[:-1]+1230768000, "
            f"'M8[s]').astype(f'M8[{{Progress_lbl_dt}}]')))))({bin}t0st{pid}[sl_({bin}iu_cmn{pid})]) + 1",
            linked=True,
        )
        # re_dt = '(?P<dt>\d+)\.(?P<dtp>\d*)(?P<dt_u>[YMWDhms])'
        #   f"""
        #   v.i_whole_time_intervals({bin}t0st{pid}, asscalar(diff({bin}t0st{pid}[[0,-1]])/Progress_lbl_dt))[1:-1] if isinstance(Progress_lbl_dt, (int, float)) else flatnonzero(f(lambda k, kp, unit:
        #       f(lambda arr: arr
        #       diff(int8(array({bin}t0st{pid}[:-1]+1230768000, 'M8[s]').astype(f"M8[{{(kp or k).lstrip('0')}}{{unit}}]")))
        #   ) / 10**len(kp.rstrip('0')),
        #           *(match('{re_dt}', Progress_lbl_dt).groups() if '.' in Progress_lbl_dt else ['', '', Progress_lbl_dt])
        #   )+1
        #   """

        SetDataExpression(
            f"disp_{bin}dt_same_units{pid}",
            f"around(array(1E9*({bin}t0st{pid}[int32(disp_{bin}i{pid} + {bin}iu_cmn{pid}[0,0])] - "
            "disp_time_span[0]), 'm8[ns]')/timedelta64(1, Progress_lbl_dt[-1]), 3)",
            linked=True,
        )
        SetDataExpression(
            f"disp_{bin}t{pid}",
            f"{bin}t0st{pid}[int32(disp_{bin}i{pid} + {bin}iu_cmn{pid}[0,0])]",
            linked=True,
        )

        _ = (
            format_TickLabels(
                disp_dtime_range_s, st_fmt=imax_time_unit_char, compact=True
            )
            if b_dt_big
            else "%Vg"
        )
        print("Labels format:", f'disp_{bin}dt_same_units{pid} is "{_}"')
        DatasetPlugin(
            "NumbersToText",
            {
                "ds_in": f"disp_{bin}t{pid}"
                if b_dt_big
                else f"disp_{bin}dt_same_units{pid}",
                "ds_out": f"disp_{bin}t_txt{pid}",
                "format": rf"\color{{{clr}}}{{{_}}}",
                # 'format': '%VDd_%VDb'
                # if disp_dtime_range_s > np.int32(np.timedelta64(2, 'D').astype('m8[s]')) else '%VDh:%VDm'
            },
        )
        have_data_for_pg_progress.add(pid)

    lim_str = {
        x: {lim: f"{lim}(DATA('{bin}{u}_cum{pid}'))" for lim in ("min", "max")}
        for x, u in ("xu", "yv")
    }  # sets lims for last one pid

    graph_height = 15.7
    cur_graph_width_cur = graph_height * aspect
    grid_leftMargin = 1.3
    grid_rigtMargin = cfg_plot.grid_horMargins_sum - grid_leftMargin
    map_bottomMargin = 0.8
    if clr_by == "probe":
        clr_by = None
    pg_name = f"_Vprogress{f'_clr{title(clr_by)}' if clr_by else ''}"  # first "_" means it is assembling picture
    clr_param = (
        "\\Psi" if clr_by == "dir" else "|V|" if clr_by == "abs" else ""
    )  # disp_param.get(f'V{clr_by}', '')
    clr_unit = (
        "°"
        if clr_by == "dir"
        else "\\\\\\italic{%{{info_incl['units']}}%}"
        if clr_by == "abs"
        else ""
    )
    print(f"Page {pg_name}", end=": ")

    Add("page", name=pg_name, autoadd=False)
    To(pg_name)
    Set("width", f"{cur_graph_width_cur + grid_leftMargin + grid_rigtMargin:g}cm")
    Set(
        "height",
        f"{graph_height + (cfg_plot.grid_bottomMargin - map_bottomMargin):g}cm",
    )
    # put x units above x-axis if narrow
    str_units_added = label_xUnits_add(
        f"/{pg_name}/grid1/x",
        graph_height,
        y_cm=(1.8 if WidthGrade == WidthGrades["Narrow"] and not x_units_nl else 0.06),
        cfg_plot=cfg_plot,
    )

    if n_graphs > 1:
        _ = 'v.pl(f"'
        __ = '")'
    else:
        _ = 'f"'
        __ = '"'
    t = common_point_for_all(graphs)
    add_label_Title(
        sentences=([f'"{t}. "'] if t else [])
        + [_ + "{{current velocity}} {{by {info_incl['device']}}}" + __],
        split_before_date=not device_wind and WidthGrade < WidthGrades["Wide"],
        # can't split if have extra wind row
        split_params=WidthGrade < WidthGrades["VeryWide"],
        text2='''f"{{{info_wind['nature']}}} {{by}} {{{info_wind['device']}}}"'''
        if device_wind
        else "",
        grid_leftMargin=grid_leftMargin,
        str_vsz_time_range='DATA("time_span_i_common")',
        cfg_plot=cfg_plot,
        graphs_height_sum=graph_height,  # have one graph only
    )
    if clr_by:
        Add("label", name="l_clr_header", autoadd=False)
        To("l_clr_header")
        Set("label", f"{clr_param}, {clr_unit}")
        Set("hide", n_graphs > 1)
        Set("xPos", [1.0])
        Set(
            "yPos",
            f"1.0{'5' if clr_by == 'dir' else '3'} - SETTING('/{pg_name}/grid_diagram/map0/cbar/vertManual')",
        )
        Set("alignHorz", "right")
        Set("alignVert", "bottom")
        Set("margin", "2pt")
        To("..")
    Add("grid", name="grid_diagram", autoadd=False)
    To("grid_diagram")
    Set("rows", 1)
    Set("columns", 1)
    Set("leftMargin", f"{grid_leftMargin}cm")
    Set("rightMargin", f"{grid_rigtMargin}cm")
    Set("topMargin", "0cm")
    Set("bottomMargin", f"{cfg_plot.grid_bottomMargin - map_bottomMargin:g}cm")
    Set("internalMargin", "0.2cm")
    Add("graph", name="map0", autoadd=False)
    To("map0")
    Set("leftMargin", "0.0cm")
    Set("rightMargin", "0.0cm")
    Set("bottomMargin", f"{map_bottomMargin:g}cm")
    Set("aspect", aspect)
    if clr_by:
        Add("colorbar", name="cbar", autoadd=False)
        To("cbar")
        Set("widgetName", f"V{bin}_clr{pid}")
        Set("hide", n_graphs > 1)
        Set("min", "Auto")
        Set("direction", "vertical")
        Set("otherPosition", 0.05)
        Set("TickLabels/font", "Arial")
        if clr_by == "dir":
            Set(
                "MajorTicks/manualTicks",
                [-270.0, -180.0, -90.0, 0.0, 90.0, 180.0, 270.0, 360.0],
            )
            Set("MinorTicks/hide", True)
        Set("horzPosn", "manual")
        Set("vertPosn", "manual")
        Set("height", "7cm")
        Set("horzManual", 1.072)
        Set("vertManual", 0.13 if clr_by == "dir" else 0.2)
        To("..")
        Add("label", name="labelsUnits", autoadd=False)
        To("labelsUnits")
        Set(
            "label",
            "%{{'' if isinstance(Progress_lbl_dt, (int, float)) else f\"{v.str_date_unit(DATA('time_span_i'))} {'(hours)' if 'h' in Progress_lbl_dt else ''}\"}}%",
        )
        Set(
            "label",
            "%{{'' if isinstance(Progress_lbl_dt, (int, float)) else f(lambda t: '{} after{}'.format({'m': 'Minutes', 'h': 'Hours', 'T': 'Time'}.get(Progress_lbl_dt[-1], 'T'), t.split('after')[-1]) if 'after' in t else t, fDisp_date_u(None, 'disp_time_span'))}}%",
        )
        Set("hide", False)
        Set("xPos", [0.5])
        Set("yPos", [0.99])
        Set("xAxis", "x_km")
        Set("yAxis", "y_km")
        Set("alignHorz", "centre")
        Set("alignVert", "top")
        Set("Text/size", "8pt")
        Set("Text/italic", True)
        To("..")
    if n_graphs > 1:
        Add("label", name="l_devices_header", autoadd=False)
        To("l_devices_header")
        Set(
            "label",
            "%{{{{v.c1(', '.join([{}]))}}}}%".format(
                "".join(
                    [
                        s
                        for info_var, b in (
                            ("info_wind", bool(device_wind)),
                            ("info_incl", (n_graphs - bool(device_wind) > 0)),
                        )
                        if b
                        for s in [
                            r"r'{{{nature}}} {{{{^\rightarrow_{letter}}}}}'.format_map(",
                            info_var,
                            ").format_map(I),",
                        ]
                    ]
                )
            ),
        )
        Set("xPos", [0.800 + 0.09])
        Set("yPos", [0.950 + 0.015])
        Set("positioning", "relative")
        Set("xAxis", "xL")
        Set("yAxis", "yL")
        Set("alignHorz", "centre")
        Set("alignVert", "bottom")
        Set("margin", "1pt")
        Set("Text/font", "Arial")
        Set("Text/color", "black")
        To("..")

        Add("label", name="l_device_option", autoadd=False)
        To("l_device_option")
        Set(
            "label",
            f"%{{{{f(r'\\\\'.join, [(r'\\bold{{\\color{{%s}}—}} %s' % ("
            f"SETTING(f'/{pg_name}/grid_diagram/map0/<V>line{{pid}}/PlotLine/color'), "
            f"fr'{{{{^\\rightarrow_W}}}}' if pid == '_Wind' else fr'{{{{^\\rightarrow_V}}}}'"
            + f")) for pid in {graphs}])}}}}%",
        )
        Set("xPos", [0.800])
        Set("yPos", [0.950])
        Set("positioning", "relative")
        Set("xAxis", "xL")
        Set("yAxis", "yL")
        Set("alignHorz", "left")
        Set("alignVert", "top")
        Set("margin", "1pt")
        Set("Text/font", "Arial")
        Set("Text/color", "black")
        Set("Text/bold", False)
        Set("Background/transparency", 100)
        To("..")
        Add("label", name="l_device_option_vals", autoadd=False)
        To("l_device_option_vals")
        _ = scale_wind_more_str.replace(
            "Wind_to_current_coef", "{Wind_to_current_coef}"
        )  # gets f"*{_coef}" in Veusz
        Set(
            "label",
            rf"%{{{{f(r'\\'.join,[f'{_}' if pid == '_Wind' else DISPdevice[pid] for pid in {graphs}])}}}}%",
        )
        Set(
            "xPos",
            f"SETTING('/{pg_name}/grid_diagram/map0/l_device_option/xPos')[0] + 0.066 / SETTING('/{pg_name}/grid_diagram/map0/aspect')",
        )
        Set(
            "yPos",
            f"SETTING('/{pg_name}/grid_diagram/map0/l_device_option/yPos')[0] - 0.007",
        )
        Set("positioning", "relative")
        Set("xAxis", "xL")
        Set("yAxis", "yL")
        Set("alignHorz", "left")
        Set("alignVert", "top")
        Set("margin", "1pt")
        Set("Text/font", "Arial")
        Set("Text/color", "black")
        Set("Text/bold", False)
        Set("Background/transparency", 100)
        To("..")

    Add("line", name="endsArrows_devices", autoadd=False)
    To("endsArrows_devices")
    Set("arrowleft", "none")
    Set("arrowright", "linecross")
    Set("arrowSize", "1pt")
    Set("mode", "point-to-point")
    Set("hide", False)
    for iend, sfx in [(-2, "Pos"), (-1, "Pos2")]:
        for xy, uv in [("x", "u"), ("y", "v")]:
            Set(
                f"{xy}{sfx}",
                f"[DATA(f'bin_{uv}_cum_{{i}}')[{iend}] for i in DISPdevices_info]"
                + (f" + [bin2_{uv}_cum_Wind[{iend}]]" if device_wind else ""),
            )
    Set("xAxis", "x_km")
    Set("yAxis", "y_km")
    Set("Line/hide", True)
    To("..")

    for pid, clr in zip(graphs, colors_local):
        param_expr_dict = get_param_expr_dict(bin=bin, suffix=pid)
        t0sfx = "" if use_bins[bin0name] else pid
        if pid == "_Wind":
            bin = "bin2_"
            const_suffix = pid
        else:
            bin = "bin_"
            const_suffix = ""
        Add("label", name=f"dateLabels{pid}", autoadd=False)
        To(f"dateLabels{pid}")
        Set("label", f"disp_{bin}t_txt{pid}")
        Set("hide", False)
        for x, u in ("xu", "yv"):
            Set(
                f"{x}Pos",
                f"{bin}{u}_cum{pid}[int32(disp_{bin}i{pid})]"
                + (
                    f" + 0.005*abs(diff(Progress_x_lims)/SETTING('/{pg_name}/grid_diagram/map0/y_km/datascale'))"
                    if x == "x"
                    else ""
                ),
            )
        Set("positioning", "axes")
        Set("xAxis", "x_km")
        Set("yAxis", "y_km")
        Set("margin", "1pt")
        Set("clip", True)
        Set("Text/size", "8pt")
        Set("Text/italic", True)
        Set("Background/transparency", 50)
        To("..")
        Add("line", name=f"dateArrows{pid}", autoadd=False)
        To(f"dateArrows{pid}")
        Set("arrowleft", "linearrowreverse")
        Set("arrowright", "none")
        Set("arrowSize", "1pt")
        Set("mode", "point-to-point")
        # Set('hide', clr_by != 'dir')
        # Set('length', f'{bin}Vabs{pid}*DISPscale_vec' if pid!='_Wind' else '{}{}'.format(param_expr_dict['Vabs'], scale_wind_more_str))
        # Set('angle', f'{bin}Vdir{pid}-90' if pid!='_Wind' else (param_expr_dict['Vdir'] + ' - 90'))
        for x, u in ("xu", "yv"):
            Set(f"{x}Pos", f"{bin}{u}_cum{pid}[int32(disp_{bin}i{pid})]")
            Set(
                f"{x}Pos2",
                f"{bin}{u}_cum{pid}[int32(disp_{bin}i{pid}) + (1 if (disp_{bin}i{pid}[-1] +1) < {bin}{u}_cum{pid}.shape[0] else append(ones((len(disp_{bin}i{pid})-1,),int32), 0) )]",
            )
        Set("xAxis", "x_km")
        Set("yAxis", "y_km")
        Set("Line/hide", True)
        To("..")
        Add("xy", name=f"datePoints{pid}", autoadd=False)
        To(f"datePoints{pid}")
        Set("marker", "circle")
        Set("markerSize", "0.5pt")
        for x, u in ("xu", "yv"):
            Set(f"{x}Data", f"{bin}{u}_cum{pid}[int32(disp_{bin}i{pid})]")
        Set("hide", True)  # clr_by == 'dir'
        Set("xAxis", "x_km")
        Set("yAxis", "y_km")
        Set("PlotLine/hide", True)
        Set("MarkerLine/color", "magenta")
        Set("MarkerLine/width", "1pt")
        Set("MarkerLine/hide", False)
        Set("MarkerFill/color", "white")
        Set("MarkerFill/hide", False)
        Set("Label/posnVert", "top")
        Set("Label/hide", False)
        To("..")
        Add("xy", name=f"<V>line{pid}", autoadd=False)
        To(f"<V>line{pid}")
        for x, u in ("xu", "yv"):
            Set(f"{x}Data", f"{bin}{u}_cum{pid}")
        Set("hide", False)
        Set("xAxis", "x_km")
        Set("yAxis", "y_km")
        Set("PlotLine/color", clr)
        Set("PlotLine/width", "1pt" if n_graphs > 1 else "0.25pt")
        To("..")
        if clr_by:
            Add("xy", name=f"V{bin}_clr{pid}", autoadd=False)
            To(f"V{bin}_clr{pid}")
            Set("marker", "circle")
            Set("markerSize", "2pt")
            for x, u in ("xu", "yv"):
                Set(
                    f"{x}Data",
                    f"append(nanmean({u}{pid}[sl_([iu{pid}[0,0], searchsorted("
                    f"t_ns{t0sfx}{'' if b_one_table else pid}, "
                    f"1E9*({bin}t0st{pid}[0] - USE_timeShift_s))])])*{bin[:-1]}, {bin}{u}_cum{pid})",
                )
            Set("hide", n_graphs > 1)
            Set("xAxis", "x_km")
            Set("yAxis", "y_km")
            Set(
                "scalePoints",
                (
                    f"(({bin}Vabs{pid}"
                    if pid != "_Wind"
                    else f"(({param_expr_dict['Vabs']}"
                )
                + ") + Progress_scale0) * Progress_scale"
                + (scale_wind_more_str if pid == "_Wind" else ""),
            )
            Set(
                "Color/points",
                "{}{}{}".format(
                    bin,
                    (lambda p: p if pid != "_Wind" else param_expr_dict[p])(
                        f"V{clr_by}"
                    ),
                    pid,
                ),
            )
            if clr_by == "dir":
                Set("Color/min", -180.0)
                Set("Color/max", 180.0)
                Set("MarkerLine/color", "darkmagenta")
                Set("MarkerLine/width", "0.5pt")
                Set("MarkerLine/hide", True)
                Set("MarkerFill/transparency", 30)
            else:
                Set("Color/max", axis_max["Vabs"])
            Set("PlotLine/color", "black")
            Set("PlotLine/width", "0.5pt")
            Set("PlotLine/hide", clr_by != "dir")
            Set("MarkerFill/hide", False)
            Set(
                "MarkerFill/colorMap", "colormapDir" if clr_by == "dir" else "spectrum2"
            )
            Set("MarkerFill/colorMapInvert", True)
            Set("Label/posnVert", "bottom")
            To("..")
        Add("xy", name=f"V0_bin{pid}", autoadd=False)
        To(f"V0_bin{pid}")
        for x, u in ("xu", "yv"):
            Set(
                f"{x}Data",
                "append(0, cumsum(f(lambda x: v.rep2mean(x, isfinite(x)), "
                f"{u}{pid}[sl_(iu{pid})])))*{bin.removesuffix('_')}",
            )
        Set("hide", True)
        Set("xAxis", "x_km")
        Set("yAxis", "y_km")
        Set("PlotLine/color", "magenta")
        To("..")
        Add("line", name=f"vectors{pid}", autoadd=False)
        To(f"vectors{pid}")
        Set("arrowleft", "none")
        Set("arrowright", "arrow")
        Set("arrowSize", "5pt")
        Set("mode", "point-to-point")
        Set("hide", True)
        Set(
            "length",
            f"{f'{bin}Vabs{pid}' if pid != '_Wind' else param_expr_dict['Vabs']}*DISPscale_vec",
        )
        Set(
            "angle",
            f"{bin}Vdir{pid}-90"
            if pid != "_Wind"
            else f"{param_expr_dict['Vdir']} - 90",
        )
        for x, u in ("xu", "yv"):
            Set(f"{x}Pos", f"{bin}{u}_cum{pid}[int32(disp_{bin}i{pid})]")
            Set(
                f"{x}Pos2",
                f"{bin}{u}_cum{pid}[int32(disp_{bin}i{pid}) + (1 if (disp_{bin}i{pid}[-1] +1) < {bin}{u}_cum{pid}.shape[0] else append(ones((len(disp_{bin}i{pid})-1,),int32), 0) )]",
            )
        Set("xAxis", "x_km")
        Set("yAxis", "y_km")
        Set("Line/color", "magenta")
        Set("Line/width", "1pt")
        Set("Fill/color", "magenta")
        Set("Fill/transparency", 50)
        To("..")

    # Titles at end of each progressive vector
    Add("label", name="l_ends_devices", autoadd=False)
    To("l_ends_devices")
    Set("label", "disp_devices_info_keys")

    # Shift labels in direction of last vector and clip by graph limits with margin (shift ~ to word len)
    # - along X axis
    Set(
        "xPos",  # "[DATA(f'bin_u_cum_{i}')[-1] for i in DISPdevices_info]"
        r"""(
lambda x, x_prev, lims, length, margin_chars, k: ravel(clip(
x*k - margin_chars*0.004*length*sign([x - x_prev]),
*lims + margin_chars*0.008*length*[[1], [-1]]
))/k
)(
*array([DATA(f'bin_u_cum_{i}')[-2:] for i in DISPdevices_info]).T,
atleast_2d(Progress_x_lims).T,
diff(Progress_x_lims),
int32([[len(w) for w in DATA('disp_devices_info_keys')]]) - len('\color{#660000}{st.\bold{}}'),
SETTING('/_Vprogress/grid_diagram/map0/y_km/datascale')
)""",
    )
    # - along Y axis
    Set(
        "yPos",  # "[DATA(f'bin_v_cum_{i}')[-1] for i in DISPdevices_info]"
        """(
    lambda x, x_prev, lims, length, margin_chars, k: ravel(clip(
        x * k - 0.015*length*sign([x - x_prev]),
        *lims + margin_chars*0.05*length*[[1], [-1]]
    ))/k
)(
    *array([DATA(f"bin_v_cum_{i}")[-2:] for i in DISPdevices_info]).T,
    array(
        [
            [Progress_y_min],
            (Progress_y_min + diff(Progress_x_lims) / SETTING("/_Vprogress/grid_diagram/map0/aspect")),
        ]
    ),
    diff(Progress_x_lims) / SETTING("/_Vprogress/grid_diagram/map0/aspect"),
    1,
    SETTING("/_Vprogress/grid_diagram/map0/y_km/datascale")
)""",
    )

    Set("positioning", "axes")
    Set("xAxis", "x_km")
    Set("yAxis", "y_km")
    Set("alignHorz", "centre")
    Set("alignVert", "centre")
    Set("margin", "1pt")
    # Set("clip", True)
    Set("Text/size", "14pt")
    Set("Text/italic", True)
    Set("Background/transparency", 50)
    To("..")
    Add("axis-function", name="x_km", autoadd=False)
    To("x_km")
    Set(
        "function",
        f"f(lambda x: t*diff(x) + x[0], v.max_range(Progress_x_lims, float32([{lim_str['x']['min']}, "
        f"{lim_str['x']['max']}])*SETTING('/{pg_name}/grid_diagram/map0/x_km/datascale')))",
    )
    Set("min", -7.5)
    Set("max", 0.5)
    Set("linkedaxis", "y[0,1]")
    Set("mint", 0.0)
    Set("maxt", 1.0)
    Set("datascale", 0.001)
    Set("direction", "horizontal")
    Set("Label/font", "Arial")
    Set("Label/position", "at-maximum")
    Set("MajorTicks/number", int(8 * aspect))
    Set("GridLines/hide", False)
    To("..")
    Add("axis-function", name="y_km", autoadd=False)
    To("y_km")
    Set(
        "function",
        f"t*diff(v.max_range(Progress_x_lims, float32([{lim_str['x']['min']}, "
        f"{lim_str['x']['max']}])*SETTING('/{pg_name}/grid_diagram/map0/x_km/datascale'))) / SETTING('/{pg_name}/grid_diagram/map0/aspect') + fmin(Progress_y_min, {lim_str['y']['min']}*SETTING('/{pg_name}/grid_diagram/map0/y_km/datascale'))",
    )
    Set(
        "label",
        "".join(
            [
                """\\italic{%{{f(lambda k: I['km'] if k==1e-3 else f"{I['m']}·10^{{{-log10(k):g}}}", SETTING('/""",
                pg_name,
                "/grid_diagram/map0/y_km/datascale'))}}%,\\\\%{{I['N']}}% }",
            ]
        ),
    )
    Set("linkedaxis", "y[0,1]")
    Set("mint", 0.0)
    Set("maxt", 1.0)
    Set("datascale", 0.001)
    Set("direction", "vertical")
    Set("Label/font", "Arial")
    Set("Label/atEdge", True)
    Set("Label/rotate", "90")
    Set("Label/position", "at-maximum")
    Set("MajorTicks/number", 8)
    Set("GridLines/hide", False)
    To("..")
    Add("xy", name="x=0", autoadd=False)
    To("x=0")
    Set("xData", "[-1e9, 1e9]")
    Set("yData", "zeros(2)")
    Set("xAxis", "x_km")
    Set("yAxis", "y_km")
    Set("PlotLine/color", "black")
    Set("PlotLine/width", "0.5pt")
    Set("MarkerLine/hide", True)
    Set("MarkerFill/hide", True)
    To("..")
    Add("xy", name="y=0", autoadd=False)
    To("y=0")
    Set("xData", "zeros(2)")
    Set("yData", "[-1e9, 1e9]")
    Set("xAxis", "x_km")
    Set("yAxis", "y_km")
    Set("PlotLine/color", "black")
    Set("PlotLine/width", "0.5pt")
    Set("MarkerLine/hide", True)
    Set("MarkerFill/hide", True)
    To("..")
    To("..")
    Add("axis", name="y[0,1]", autoadd=False)
    To("y[0,1]")
    Set("hide", True)
    Set("min", 0.0)
    Set("max", 1.0)
    Set("autoMirror", False)
    Set("otherPosition", 1.0)
    Set("Label/position", "at-maximum")
    Set("TickLabels/color", "#55ff00")
    Set("TickLabels/hide", True)
    Set("MajorTicks/hide", True)
    Set("MinorTicks/hide", True)
    Set("GridLines/hide", True)
    To("..")
    To("..")
    Add("graph", name="background", autoadd=False)
    To("background")
    if clr_by:
        const_suffix = "_Wind" if graphs == ["_Wind"] else ""
        Add("label", name="l_current", autoadd=False)
        To("l_current")
        Set("label", "|V|,\\\\\\italic{%{{info_incl['units']}}%}")
        Set("hide", clr_by != "dir")
        Set("xPos", [1.0])
        Set("yPos", [0.37])
        Set("xAxis", "x[0,1]")
        Set("yAxis", "y[0,1]")
        Set("alignHorz", "right")
        Set("alignVert", "bottom")
        Set("margin", "2pt")
        To("..")
        Add("xy", name="Lv", autoadd=False)
        To("Lv")
        Set("marker", "circle")
        Set("markerSize", "2pt")
        Set("xData", f"ones(len(leg_v_progress{const_suffix})) - 0.01")
        Set("yData", f"0.2 + arange(len(leg_v_progress{const_suffix}))/30")
        Set("xAxis", "x[0,1]")
        Set("yAxis", "y[0,1]")
        Set("labels", f"leg_v_progress{const_suffix}")
        Set(
            "scalePoints",
            f"(leg_v_progress{const_suffix} + Progress_scale0) * Progress_scale"
            + (scale_wind_more_str if pid == "_Wind" else ""),
        )
        if clr_by == "dir":
            Set("MarkerLine/color", "darkmagenta")
            Set("MarkerLine/width", "0.5pt")
            Set("MarkerLine/hide", True)
            Set("MarkerFill/color", "grey")
            Set("PlotLine/color", "grey")
        else:
            Set("Color/points", f"leg_v_progress{const_suffix}")
            Set("Color/min", 0.0)
            Set("Color/max", axis_max["Vabs"])
            Set("PlotLine/color", "black")
            Set("MarkerFill/colorMap", "spectrum2")
            Set("MarkerFill/colorMapInvert", True)
        Set("PlotLine/width", "0.5pt")
        Set("PlotLine/hide", False)
        Set("MarkerFill/transparency", 50)
        Set("MarkerFill/hide", False)
        Set("Label/posnHorz", "left")
        Set("Label/posnVert", "top")
        Set("Label/hide", False)
        To("..")
    Add("axis", name="x[0,1]", autoadd=False)
    To("x[0,1]")
    Set("hide", True)
    Set("min", 0.0)
    Set("max", 1.0)
    Set("autoMirror", False)
    Set("direction", "horizontal")
    Set("otherPosition", 1.0)
    Set("Label/position", "at-maximum")
    Set("TickLabels/color", "#55ff00")
    Set("TickLabels/hide", True)
    Set("MajorTicks/hide", True)
    Set("MinorTicks/hide", True)
    Set("GridLines/hide", True)
    To("..")
    Add("axis", name="y[0,1]", autoadd=False)
    To("y[0,1]")
    Set("hide", True)
    Set("min", 0.0)
    Set("max", 1.0)
    Set("autoMirror", False)
    Set("otherPosition", 1.0)
    Set("Label/position", "at-maximum")
    Set("TickLabels/color", "#55ff00")
    Set("TickLabels/hide", True)
    Set("MajorTicks/hide", True)
    Set("MinorTicks/hide", True)
    Set("GridLines/hide", True)
    To("..")
    To("..")
    Add("label", name="l_kmE", autoadd=False)
    To("l_kmE")
    Set(
        "label",
        "".join(
            [
                """\\italic{%{{f(lambda k: I['km'] if k==1e-3 else f"{I['m']}·10^{{{-log10(k):g}}}", SETTING('/""",
                pg_name,
                "/grid_diagram/map0/x_km/datascale'))}}%,\\\\%{{I['E']}}% }",
            ]
        ),
    )
    Set("hide", False)
    Set("xPos", [1.13 - 0.04 * aspect])
    Set("yPos", [0.035])
    Set("xAxis", "x_km")
    Set("yAxis", "y_km")
    Set("alignHorz", "right")
    Set("alignVert", "bottom")
    Set("margin", "0.5pt")
    Set("Background/transparency", 100)
    To("..")
    To("..")


# Run pages ######################################################################################################
if __name__ in ("__main__", "builtins"):
    # if False:
    if b_draw_progressive_vector:
        # pg_progress_3d(ids_order, b_zabor=True)
        # pg_progress_3d(ids_order, b_zabor=False)
        for clr_by in [""]:  # , 'abs', 'dir']:
            pg_progress(
                ids_order,
                clr_by=clr_by,
                aspect=progress_aspect,
                b_dt_big=b_dt_big,
                cfg_plot=cfg_plot,
            )

    if ids_i or device_wind:
        pg_vectors(ids_order, scale_height=graphs_scale_height, cfg_plot=cfg_plot)

    for param in (
        ["Vabs"] + (["u&v", "Vdir"] if ids_i or device_wind else []) + ["t"]
    ):  # u-shore v-shore
        pg_1d(
            ids_order, param=param, scale_height=graphs_scale_height, cfg_plot=cfg_plot
        )

    if False: # |ids_i : fix bugs in `bin_u_2d`, ... to use
        for param in "Vabs,dir u,v,t".split():  # u,v-shore
            pg_2d(param, ids_i_2d=ids_w + ids_i, cfg_plot=cfg_plot)

    for param in ["Vabs"] + (
        ["u", "v"] if ids_i or device_wind else []
    ):  # ' u-shore v-shore'
        pg_1d(
            ids_order,
            param=param,
            scale_height=graphs_scale_height,
            zoom=True,
            cfg_plot=cfg_plot,
        )
