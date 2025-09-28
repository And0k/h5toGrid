# Veusz document (version 3.4+)
from itertools import compress, takewhile
import numpy as np
from logging import warning
import func_vsz as fv
from enum import IntEnum
import builtins

# Common functions
# ################
AddCustom("definition", "f", "lambda fun, *args, **kwargs: fun(*args, **kwargs)")
AddCustom(
    "definition", "sl_(x)", "slice(*([None] if isnan(x).all() else int32(ravel(x))))"
)
AddCustom(
    "definition",
    "argv1",
    "argv[1] if argv[1]!='--embed-remote' else ENVIRON.get('VSZ_PATH', FILENAME())",
)
AddCustom(
    "definition",
    "fDisp_date_u(ax, t_span_var, **kwargs)",
    "v.str_date_unit_with_suffix([f(lambda l: l if l!='Auto' else t, SETTING(f'{ax}/{lim:s}')) for lim, t in zip(('min', 'max'), DATA(t_span_var))] if ax else DATA(t_span_var), str_zone=DISPdevice_info['zone'], lang=LANG({'default': 'en', 'ru': 'ru'}), next_fmt=kwargs.pop('next_fmt', 0), **kwargs)",
)

# Custom Definitions
# ##################

# Display time range and data range intervals

# Graph auto range (should be commented to use dir vsz(range=x) or all data settings):
if False:
    max_time_span_s_strings = ["2024-06-27T00:00", "2024-08-18T23:02:28"]
    print(f"Manual range set: [{max_time_span_s_strings}]")


# Old used ranges (insert [] expression in double quotes):
# ['2023-04-23T08:00', '2023-05-25T14:00']

AddCustom("definition", "DISPtime", f"[{max_time_span_s_strings}]  # graph auto range")

for t, ids_t, t_ranges in zip(
    "iw",
    (ids_i, ids_w),
    (
        {  # inclinometers
            # '14': ['2021-07-26T10:00:00', '2021-08-27T19:00:00'],
        },
        {  # wave gauges
            # '02': ['2021-09-22T13:00:00', '2021-09-28T18:00:00'],
        },
    ),
):
    for pid, t_range in {**{pid[2:]: [] for pid in ids_t}, **t_ranges}.items():
        AddCustom(
            "definition",
            f"USEtime_{t}{pid}",
            f"[{t_range}]"  # "['{}']".format("', '".join(t_range))
            if t_range
            else "DISPtime",
        )

time_range = np.array(max_time_span_s_strings, "M8[s]")
disp_dtime_range_s = int(np.ediff1d(time_range))
if not disp_dtime_range_s.size:
    disp_dtime_range_s = nan
print("Graph time interval, s:", disp_dtime_range_s)

imax_time_unit_char = len(
    list(
        takewhile(
            lambda x: x <= 0,
            [b - a for a, b in zip(*[t.timetuple() for t in time_range.tolist()])],
        )
    )
)  # (0: y, 1: m, 2: d, 3: H, 4: M, 5: S)

# Widgets parts sizes [cm]

grid_horMargins_sum = 2.4  # cm |3
grid_bottomMargin = 1.7  # cm
graph_width_standard = 28 - grid_horMargins_sum  # 25.6 cm
graph_width_scale = 1

# Graph width: constant or proportional to ``max_time_span_s_strings``
graph_width = np.fmax(graph_width_standard, graph_width_scale * 4e-06 * disp_dtime_range_s).item()
graph_width_scale = graph_width / disp_dtime_range_s
if graph_width != graph_width_standard:
    print(f"Graph width = {graph_width}, scale = {graph_width_scale} cm/s")


class WidthGrades(IntEnum):  # , boundary=CONFORM - handled by _missing_()
    # Upper bounds, cm
    VeryNarrow = 10
    Narrow = 23
    Wide = 40
    VeryWide = 100  # any value > Wide will have this value

    @classmethod
    def _missing_(cls, value):
        for member in cls:
            if member.value > value:
                return member
        else:
            return cls.Wide
        return None

    def eq(self, val):
        return self.__class__[val] == self.value

WidthGrade = WidthGrades(graph_width)
if WidthGrade == WidthGrades["VeryNarrow"]:
    print(f"Too narrow for full titles (< {WidthGrade.value}cm)")
elif WidthGrade == WidthGrades["Narrow"]:
    print(f"Slightly narrow for full titles (< {WidthGrade.value}cm)")

# disp_dtime_range_s < np.int32(np.timedelta64(int(graph_width_scale*24), 'D').astype('m8[s]'))
# disp_dtime_range_s < np.int32(np.timedelta64(60, 'D').astype('m8[s]'))

# Scaling graphs height / vector size parameters
################################################

# Velocity magnitude scaling `v_to_graph_h` m/s / {inch}. Default: 0.5

# To set axis limits use axis_max["Vabs"] instead: all vsz velocity graphs still will be in same scale. I.e.
# keep parameter the same for that vsz which velocity graphs are needed to compare.
v_to_graph_h = 0.5  # bigger value corresponds to smaller heights for same Vmax. | 0.5 0.2
# - Scale_height others relative to Velocity (0.5 to make wind graph = max height / 2)
graph_h_default = 5 * v_to_graph_h
# - will be multiplied by f(graphs_scale_height[graph_name]) / v_to_graph_h

# Graphs |V|-limits for specified `ids_i`&`ids_w` (only inclinometers will be used, relative to `axis_max[Vabs]`)
graphs_scale_height = {
    k: v   # / v_to_graph_h
    for k, v in {
        # "_i75": 0.156185118,
        # "_i72": 0.198510271,
    }.items()
}

#! Y limits (todo: get good limits from loader)  (set according to the max measurement signal)

axis_min = {"t": 5}  # |"t": -1.7, 'u': -0.15, 'v': -0.15,
# Do not set axis_min["dP"] as it drawn relative to the mean so antisymmetric: it will be negative of axis_max["dP"]

# Here prefer values which'll not be on major tick to have space for blank between it and next graph / to edge
axis_max = {
    "VabsWind": 15,  # | 20
    "t": 7,  # |11 18
    "dP": 1.5,
    # "Vabs": round(max(graphs_scale_height.values())*1.02, 2)  # |0.2, 0.25, 0.35, 0.42
    # max(abs(min(Vproj)), abs(max(Vproj)) can be between in range [sqrt(Vabs), Vabs]
}  # |Vabs': 0.25, 'u': 0.25, 'v': 0.25, 'VabsWind': 15 # |25
# Do not set axis_min/max["Vdir"] as for better ticks axis it is scaled on 40/360 (axis_max["Vdir"] = 40)
# If "Vabs" here not set, then `"Vabs"=v_to_graph_h` is used:
axis_max.setdefault("Vabs", v_to_graph_h)

AddCustom("definition", "Dir0proj", "0  # degrees, axes rotation")

# Minimum bin we will draw vectors in lite color
try:
    bin_lite = list(use_bins)[-2]  # 'bin_' # 'bin2_'
except IndexError:
    bin_lite = None

# not translate device to can put words in needed case later
AddCustom(
    "definition",
    "info_incl",
    "{'id': 'i', 'device': 'tilt current meter', 'dev': I['TCM'], 'measure': I['velocity'], "
    "'units': '{m}/{s}'.format_map(I), 'nature': 'current', 'letter': 'V', 'sampling': I['averaging bin']}",
)
AddCustom(
    "definition",
    "info_pres",
    "{'id': 'w', 'device': 'wave gauge', 'measure': I['pressure'], "
    "'units': I['dBar'], 'nature': I['pressure'], 'letter': 'P', 'sampling': I['averaging bin']}",
)
if wind_mean_uv is not None and np.isfinite(wind_mean_uv):
    device_wind_mean_abs = np.absolute(wind_mean_uv)
    device_wind_mean_dir = np.angle(wind_mean_uv, deg=True)
    info_wind_show = (
        r"'|W| = {:1.1f}{{m}}/{{s}},\\\\Wdir = {:1.1f}°'.format_map(I)".format(
            device_wind_mean_abs, (180 - device_wind_mean_dir) % 360
        )
    )
else:
    info_wind_show = "''"


AddCustom(
    "definition",
    "info_wind",
    f"{{'id': 'wind', 'device': {device_wind}, 'measure': I['velocity'], "
    "'units': '{m}/{s}'.format_map(I), 'nature': 'wind', 'letter': 'W', 'sampling': I['Sampling interval'], "
    f"'at': 'at 10m', 'show': {info_wind_show}}}",
)
if device_wind and disp_dtime_range_s <= 7200:
    device_wind = None

# Title for each device (DISPdevice)

# Determine similarities to not repeat in labels
zip_k_p_b_bd_s = list(
    zip(
        *[
            (k, *[np.nan if v_ is None else v_ for v_ in v])
            for k, v in cus.DISPdevices_info.items()
            if f"_{k}" in ids_i + ids_w
        ]
    )
)
depth_decimals = 0  # show digits after dot
b_one_depth_dev = not (
    zip_k_p_b_bd_s
    and np.ediff1d(np.around(np.subtract(*zip_k_p_b_bd_s[2:4]), depth_decimals)).any()
)
# b_one_depth_sea = (not zip_k_p_b_bd_s) or zip_k_p_b_bd_s[2][1:] == zip_k_p_b_bd_s[2][:-1]
b_one_point = (not zip_k_p_b_bd_s) or zip_k_p_b_bd_s[1][1:] == zip_k_p_b_bd_s[1][:-1]
# see also common_point_for_all()
str_pids = (  # """f'{si}{ki[int(ki.startswith("i")):]}'""" if b_one_point else (
    "[fr'{}' for ki, si in zip(k.replace('i_p', 'p').split('_'), ([''] if k.startswith('p') else s.split(',')))]".format(
        r'{ki[int(ki.startswith("i")):]}^{{\color{{blue}}{{{si}}}}}'
        if b_one_point and b_one_depth_dev
        else '{si}{ki[int(ki.startswith("i")):]}'  # numbers are not important, so we format they smaller with sign below
    )
)  # .replace('i_p', 'ip')
# nl = chr(92)

st = lambda p: (
    ""
    if b_one_point
    else "v.c1((f'{{st}}.{p}' if len(p) < 10 else f'{{{p}}}').format_map(I)).replace(' ', r'\\\\'), "
)
_ = "".join(
    [
        "{**{f'_{k}': ''.join([",
        st(p),
        # # show depth for all devices if any is not on bottom else show depth only for wave gauges:
        # ", f':\u2009{(b - bd):.0f}m'" if any(cus.DISPdevices_info[i[1:]][2] for i in (ids_i + ids_w) if i[1]=='i') else
        ""
        if b_one_depth_dev
        else f"""f": {{(b - bd):.{depth_decimals}f}}{{I['m']}}" if (b and isfinite(bd)) else '', """,
        "" if b_one_point and b_one_depth_dev else r"'^{\color{blue}{' + ",
        "','.join(",
        str_pids,
        ")" if b_one_point and b_one_depth_dev else ") + '}}'",  # if bd else ''
        "]) for k_, (p, b, bd, s, *kw) in DISPdevices_info.items() ",
        " for k in ([k_, k_.replace('i_p', 'p')] if k_.startswith('i_p') else [k_])}, ",  # need?
        "'_Wind': v.c1(I['wind'])}",  # adds same params for pressure data of ``ip``-probes
    ]
)
if WidthGrade == WidthGrades["VeryNarrow"]:  # replace ":{small whitespace}" with ",newline"
    _ = _.replace(":\u2009", ",{chr(92)}{chr(92)}")
AddCustom("definition", "DISPdevice", _)

colors = (
    fv.colors_of_hue_range(len(ids_order), exclude_hue_start=210, exclude_hue_end=270)
    if len(ids_order) > 1
    else ["black"]
)
clr_wind = "#00aaff"  # #5000df

# Device names evaluate here and save to `DataText`
# because text expressions for multiple coordinates are not supported
def disp_devices_info_key(k, p, b, bd, s, *kw):
    """_summary_
    :param k: pid
    :param p: station
    :param b: bottom
    :param bd: height above bottom
    :param s: device sign
    :return: string for display graph
    """

    # join signs with corresponding pids (also if they are grouped), removing "i"/"p" prefix from pids
    str_pid = [
        rf'{si}{ki[int(ki.startswith("i")):]}'
        for ki, si in zip(
            k.split("_"), ([""] if k.startswith("p") else s.split(","))
        )
    ]
    p_out = (
        "" if b_one_point
        else fv.c1((
            fr"{{st}}.\bold{{{{{p}}}}}" if len(p) < 10
            else f"{{{p}}}"
        ).format_map(fv.I))
    )
    # .removeprefix("{st}.".format_map(fv.I))
    d_out = "" if b is None or b_one_depth_dev else rf"\\^{(b - bd):.0f}{fv.I['m']}"
    return "".join(
        (
            p_out,
            d_out,
            # ('' if b is None or b_one_depth_sea else f"{b:.0f}{fv.I['m']}"),
            # "^{{\color{{blue}}{{{}}}}}".format(",".join(str_pid)),
        )
    )
# \color{#0087cc}{\bold{st.34}}
str_pids = [
    fr"\color{{{clr}}}{{{disp_devices_info_key(k, *v)}}}"
    for (k, v), clr in zip(cus.DISPdevices_info.items(), colors)
]  # or use DISPdevice.values()
SetDataText("disp_devices_info_keys", str_pids)


# todo: calc default value based on axis_max["Vabs"]
AddCustom("definition", "DISP_vecY0_distribute", "0.5   # |0.2  1")

DISP_vecY0_val = 0.1  # | 0.6
AddCustom(
    "definition",
    "DISP_vecY0",
    f"{DISP_vecY0_val}  # graphs shifts from bot relative to normed axis of one graph (y[0,1])",
)
AddCustom(
    "definition", "DISP_legY0dev", f"{DISP_vecY0_val + 0.35:.4f}  # device legends Y"
)
DISP_legY_val = 0.75  # <=> 0.82  # vector legend
AddCustom(
    "definition",
    "DISP_legY",
    f"{DISP_legY_val:g}  # legends shifts from bot relative to normed axis of one graph (y[0,1])",
)
AddCustom("definition", "Disp_leg_v", "0.2")
AddCustom(
    "definition",
    "DISP_LegVorig_mul",
    "2.5  # decrease size and increase legend value multiplier for original sampled vectors",
)
# prepare page "_vectors" scaling
# Vectors scale (should depend on max measurement signal and min displayed vector averaging)
DISPscale_vec_val = round(5 * (axis_max["Vabs"] + 4.2) / max(np.log(use_bins.get(bin_lite, 2)), 1), 1)
print("suggesting DISPscale_vec value:", DISPscale_vec_val)
cus.DISPscale_vec = 4  # | DISPscale_vec_val 8

AddCustom(
    "definition",
    "DISPscale_page_vectors",
    f"DISPscale_vec/{graph_width:g}  # can not calculate by Veusz gui this: float(SETTING('/_vectors/width')[:-2]) - float(SETTING('/_vectors/grid1/leftMargin')[:-2]) - Veusz not updates this value on loading",
)
AddCustom(
    "definition", "WINDscale_page_vectors", "DISPscale_page_vectors * 0.1/DISPscale_vec"
)
k_height_vectors = 1  # | 0.5 0.3: to set scale_i_graphs=k_height_vectors * axis_max["Vabs"] / v_to_graph_h

wind_bin_average_s = 3600
AddCustom("definition", "WIND_bin_average_s", str(wind_bin_average_s))
AddCustom(
    "definition",
    "Wind_timeShift_s",
    "0 if DISPdevice_info['zone']=='UTC' else 7200  # [s] add to source (UTC) time",
)  # addition to draw in our time zone
AddCustom("definition", "Wind_leg_v", "10")
AddCustom("definition", "WIND_legY", "0.1")
AddCustom("definition", "USEtime_Wind", "DISPtime  # data range")

# todo: use AddCustom('definition', 'DISP_max_u', '10')

# by default do not draw when minimum bin excluded (i.e. when no averaging that is typically when timerange too small?)
b_draw_progressive_vector = use_bins[bin0name]

b_draw_progressive_zabor = False  # True

cus.Wind_to_current_coef = 0.01  # Wind scale scaler for progressive vector and for default axis_max["VabsWind"]
if b_draw_progressive_vector:
    progress_aspect = 2  # len(x) / len(y)  | 1
    cus.Progress_x_lims = [-100, 600] if ids_order != [] else [-1290, 1290]  # | [-43, 43]
    cus.Progress_y_min = -120 if ids_order != [] else -300  # | -10
    cus.Progress_scale = "DISPscale_vec * 1  # used for: scalePoints"  # "DISPscale_vec*15/float(SETTING('/_/width')[:-2]
    cus.Progress_scale0 = 0.01
    cus.Progress_lbl_dt = (
        "'{}'  # Y,M,D,h,m,s for year, month, day... if str else divider of used data time range").format(
        "5D"
        if disp_dtime_range_s > np.int32(np.timedelta64(10, "D").astype("m8[s]"))
        else f"{(lambda x: x if x > 0 else 1)(4*(disp_dtime_range_s // (4*36000)))}h"
    )  # excludes 0 because else Veusz will crash when calc "disp_{bin}i{pid}"
    # Display labels as dates/day times if dt is big else as a number of time units from beginning:
    b_dt_big = disp_dtime_range_s > np.int32(np.timedelta64(5, "D").astype("m8[s]"))  # |False

    b_draw_progressive_vector_3d = False
    if b_draw_progressive_vector_3d:
        cus.Disp_pgs3d_lines_vert_dt_str = "'60s'"

    if b_draw_progressive_zabor:
        cus.Zabor_st_dt = "'6h'"
        cus.Zabor_range_dt = "'10s'"
        cus.Zabor_label_dt = "'2s'"
        cus.Zabor_shift = 2
        # To form surface we need connect progress lines in layers by many lines, so increase its frequency:
        cus.Zabor_vert_lines_freq = 5 * (
            np.timedelta64(1, cus.Zabor_range_dt.strip("'"))
            * (
                np.timedelta64(disp_dtime_range_s, "s")
                / np.timedelta64(1, cus.Zabor_st_dt.strip("'"))
            )
        ).astype("m8[s]").astype(int)
        # 100 if disp_dtime_range_s < np.int32(np.timedelta64(10, 'D').astype('m8[s]')) else 20


AddCustom(
    "colormap",
    "colormapAbs",
    (
        (255, 196, 255),
        (0, 0, 255),
        (0, 255, 255),
        (0, 255, 0),
        (255, 255, 0),
        (255, 0, 0),
        (64, 0, 0),
    ),
)
AddCustom(
    "colormap",
    "colormapDir",
    (
        (255, 0, 255),
        (0, 0, 255),
        (0, 255, 255),
        (0, 255, 0),
        (255, 255, 0),
        (255, 0, 0),
        (255, 0, 255),
    ),
)



if ids_i or ids_w:
    ## Depths for each device (negative), extent of 2D data, 2D data

    num_names_prev = 0  # starting index of device:
    for t in compress(
        ("i", "w"), [builtins.any(ids_i), builtins.any(ids_w) and ids_w != ids_p]
    ):  # in groups of same type
        # Vertical dimension of 2d images
        names = [pid[1:] for pid in (ids_i + ids_w) if pid[1:].startswith(t)]
        SetDataExpression(
            f"grD_ext_{t}",
            f"array([f(lambda p, b, bd, *kw: 0 if b is None else bd - b, *DISPdevices_info[k]) for k in {names}])",
            linked=True,
        )  # [::-1]
        SetData2DExpressionXYZ(
            f"Dim_DataP_{t}",
            f"v.dim_bug_cor(time_span_{t})",
            f"v.dim_bug_cor(-grD_ext_{t} if all(diff(grD_ext_{t})) else int32([{num_names_prev}, len(grD_ext_{t}) + {num_names_prev}]))  # if can not use depth then use devices list index",
            "0",
            linked=True,
        )
        num_names_prev = len(names)

    ## 2d data

    # 'bin_' or max_bin
    bin_use_2d = "bin_" if "bin_" in use_bins else list(use_bins)[-1]

    if bin_use_2d == "":
        # need 2d with max freq (from noAvg DB where data is not combined to one table)
        SetData2DExpression(
            "iu_i_min",
            "v.min_range_2d(array([DATA(f'iu_{i}')[0, :] for i in DISPdevices_info]))",
            linked=True,
        )
    # needed to draw 2d and progressive vector zabor. Todo: create only if needed
    for u in "uv":
        SetData2DExpression(
            f"{bin_use_2d}{u}_2d",
            f"zeros_like(Dim_DataP_i[0,0]) + array([DATA(f'{bin_use_2d}{u}_{{i}}')"
            f"{'[sl_(iu_i_min)]' if bin_use_2d == '' else ''} for i in DISPdevices_info])",
            linked=True,
        )




Set("width", "28cm")
Set("height", "21cm")
Set("StyleSheet/Font/font", "Arial")
Set("StyleSheet/axis/autoRange", "exact")
Set("StyleSheet/axis/autoMirror", False)
Set("StyleSheet/axis/direction", "vertical")
Set("StyleSheet/axis/Line/color", "black")
Set("StyleSheet/axis/Line/width", "1pt")
Set("StyleSheet/axis/Label/font", "Arial")
Set("StyleSheet/axis/Label/size", "14pt")
Set("StyleSheet/axis/Label/color", "black")
Set("StyleSheet/axis/Label/position", "at-maximum")
Set("StyleSheet/axis/TickLabels/font", "Arial")
Set("StyleSheet/axis/TickLabels/size", "14pt")
Set("StyleSheet/axis/TickLabels/color", "black")
Set("StyleSheet/axis/MajorTicks/width", "1pt")
Set("StyleSheet/axis/MinorTicks/width", "1pt")
Set("StyleSheet/axis/GridLines/hide", False)
Set("StyleSheet/axis-function/linked", True)
Set("StyleSheet/axis-function/Line/width", "1pt")
Set("StyleSheet/axis-function/Label/font", "Arial")
Set("StyleSheet/axis-function/TickLabels/font", "Arial")
Set("StyleSheet/axis-function/TickLabels/size", "14pt")
Set("StyleSheet/axis-function/MajorTicks/width", "1pt")
Set("StyleSheet/axis-function/MinorTicks/width", "1pt")
Set("StyleSheet/bar/ErrorBarLine/hide", True)
Set("StyleSheet/colorbar/autoRange", "exact")
Set("StyleSheet/colorbar/direction", "vertical")
Set("StyleSheet/colorbar/Line/width", "1pt")
Set("StyleSheet/colorbar/Label/font", "Arial")
Set("StyleSheet/colorbar/TickLabels/font", "Arial")
Set("StyleSheet/colorbar/MajorTicks/width", "1pt")
Set("StyleSheet/colorbar/MinorTicks/width", "1pt")
Set("StyleSheet/colorbar/Border/hide", True)
Set("StyleSheet/graph/leftMargin", "0cm")
Set("StyleSheet/graph/rightMargin", "0cm")
Set("StyleSheet/graph/topMargin", "0cm")
Set("StyleSheet/graph/bottomMargin", "0cm")
Set("StyleSheet/graph/Background/hide", True)
Set("StyleSheet/graph/Border/hide", True)
Set("StyleSheet/grid/columns", 1)
Set("StyleSheet/grid/leftMargin", "1.2cm")
Set("StyleSheet/grid/rightMargin", "0.5pt")
Set("StyleSheet/grid/internalMargin", "0pt")
Set("StyleSheet/key/Text/font", "Arial")
Set("StyleSheet/key/Text/size", "12pt")
Set("StyleSheet/key/Background/transparency", 20)
Set("StyleSheet/key/Border/hide", True)
Set("StyleSheet/key/horzPosn", "left")
Set("StyleSheet/key/vertPosn", "manual")
Set("StyleSheet/key/keyLength", "0.5cm")
Set("StyleSheet/label/yPos", [0.0])
Set("StyleSheet/label/margin", "1pt")
Set("StyleSheet/label/Text/font", "Arial")
Set("StyleSheet/label/Text/size", "14pt")
Set("StyleSheet/label/Background/hide", False)
Set("StyleSheet/line/arrowright", "arrownarrow")
Set("StyleSheet/line/arrowSize", "1pt")
Set("StyleSheet/line/positioning", "axes")
Set("StyleSheet/page/width", "28cm")
Set("StyleSheet/page/height", "18cm")
Set("StyleSheet/surface3d/Surface/color", "#377eb8")
Set("StyleSheet/xy/marker", "none")
Set("StyleSheet/xy/errorStyle", "none")
Set("StyleSheet/xy/PlotLine/width", "1pt")
Set("StyleSheet/xy/MarkerLine/hide", True)
Set("StyleSheet/xy/MarkerFill/color", "black")
Set("StyleSheet/xy/MarkerFill/hide", True)
Set("StyleSheet/xy/ErrorBarLine/hide", True)
Set("StyleSheet/xy/FillBelow/hideerror", True)
Set("StyleSheet/xy/FillAbove/hideerror", True)
Set("StyleSheet/xy/Label/font", "Arial")
Set("StyleSheet/xy/Label/hide", True)
