
# Required functions to be imported (source module not specified):
# import fv  # used as configuration object with lang attribute
# get_param_expr_dict, is_antisymmetric, label_Title_format, label_xUnits_add


def pg_2d(params, ids_i_2d, cfg_plot):
    """
    Create a 2D page with multiple parameter graphs.
    :param params: str of parameters separated by ",": 'Vabs,dir', 'u,v', must be in global ``disp_param``
    :param ids_i_2d: list of IDs for 2D plotting (default: combined ids_w and ids_i)
    :param cfg_plot: plotting configuration dataclass with the following required fields:
        - grid_horMargins_sum: float - sum of horizontal margins
        - graph_width: float - width of each graph
        - grid_bottomMargin: float - bottom margin for grid
     globals:
        - bin_use_2d: str - base bin name for 2D data
        - disp_param: dict - display parameter mappings
        - WidthGrade: str - current width grade
        - WidthGrades: dict - mapping of width grades to numeric values
        - zoom: bool - whether zoom is active
        - x_units_nl: bool - whether x units should be on new line
        - axis_max: dict - maximum values for each axis
        - axis_min: dict - minimum values for each axis
        - info_incl: dictionary with included information
        - cus: custom object with DISPdevices_info attribute
    """
    if not ids_i_2d:
        return
    expr_param = {
        **get_param_expr_dict(bin=bin_use_2d, suffix="_2d", wrap_dir="180"),
        "t_i": f"[{', '.join((f'{bin_use_2d}Temp{pid}' for pid in ids_i_2d if pid[1] in ('i', 'p')))}]",
        "t_w": f"[{', '.join((f'{bin_use_2d}Temp{pid}' for pid in ids_i_2d if pid[1]=='w'))}]",
    }
    pg_name = f"_2D_{params}"
    len_params = params.count(",") + 1
    len_graph = len(ids_i_2d) * 0.8
    grid_rightMargin = 1.5
    grid_leftMargin = cfg_plot.grid_horMargins_sum - grid_rightMargin
    x_units_by_label = True  # add x units label that is separate from x axis
    graphs_height_sum = len_params * len_graph + cfg_plot.grid_bottomMargin

    Add("page", name=f"_2D_{params}", autoadd=False)
    To(f"_2D_{params}")
    params = params.replace("u,v-shore", "u-shore,v-shore")
    Set("width", f"{cfg_plot.graph_width + grid_leftMargin + grid_rightMargin:g}cm")  # '29.5cm'
    Set(
        "height", f"{graphs_height_sum}cm"
    )  # proportional to graphs number and levels in them
    Add("grid", name="grid1", autoadd=False)
    To("grid1")
    Set("rows", 3)
    Set("leftMargin", f"{grid_leftMargin:g}cm")
    Set("rightMargin", f"{grid_rightMargin:g}cm")
    Set("topMargin", "0.5pt")
    Set("internalMargin", "1pt")
    Add("axis", name="yP", autoadd=False)
    To("yP")
    Set("hide", True)
    Set("autoRange", "exact")
    Set("autoMirror", False)
    Set("lowerPosition", 1.0)
    Set("upperPosition", 0.0)
    Set("MajorTicks/number", max(int(len_graph), 2))
    Set("MinorTicks/number", max(int(len_graph * 2), 4))
    To("..")
    Add("axis", name="x", autoadd=False)
    To("x")
    Set("direction", "horizontal")
    Set("Line/hide", True)
    Set("Label/hide", True)
    Set("TickLabels/hide", True)
    x_datetime_ticks(False, cfg_plot)
    for iparam, param in enumerate(params.split(",")):
        if param == "dir":
            param = "Vdir"
        To("..")
        Add("graph", name=param, autoadd=False)
        To(param)
        Set("Background/color", "#dcdcdc")
        Set("Background/hide", False)

        Add("label", name=f"l{param}", autoadd=False)
        To(f"l{param}")
        Set(
            "label",
            "".join(
                [
                    r"\size{+8}{^{",
                    disp_param[param],
                    r", }{\italic{{\frac %{{'{{{m}}}{{{s}}}'.format_map(I)}}%}}}"
                    if param[0] in "uvV" and param != "Vdir"
                    else "}",
                ]
            ),
        )
        Set("xPos", 1 + grid_rightMargin * 0.9 / cfg_plot.graph_width)  # rightmost  # 1.055
        Set("yPos", [0.74 - 0.3 / (1 + len_graph)])  # put down if very short
        Set("alignHorz", "right")
        Set("Background/transparency", 50)
        To("..")

        Add("colorbar", name=f"c{param}", autoadd=False)
        To(f"c{param}")
        Set("widgetName", param)
        Set("direction", "vertical")
        Set("otherPosition", 0.05)
        if param == "Vdir":
            Set("min", 0.0)
            Set("max", 40.0)
            Set("datascale", 0.11111111111111)
            Set("TickLabels/scale", 9.0)
            Set("MajorTicks/number", 4)
            Set("MinorTicks/number", 5)
        else:
            Set("MajorTicks/number", 2 if param[0] in "uv" else 4)
            Set("MinorTicks/number", 4)
        Set("horzPosn", "manual")
        Set("vertPosn", "manual")
        Set(
            "horzManual", 1 + grid_rightMargin * 0.75 / cfg_plot.graph_width
        )  # 1.53268e-6*disp_dtime_range_s - 3.24 rightmost
        Set("vertManual", 0.3 / (1 + len_graph) + 0.2)  # inversely to graph height
        To("..")

        Add("axis-function", name="yShow", autoadd=False)
        To("yShow")
        Set("linked", True)
        if iparam == 0:
            Set(
                "label",
                "%{{v.c1(r'{depth},\\\\{m}'.format_map(I) if all(diff(DATA('grD_ext_i'))) else "
                "r'{device}\\\\{index}'.format_map(I))}}%",
            )
        Set("linkedaxis", "yP")
        Set("autoMirror", False)
        Set("direction", "vertical")
        Set("lowerPosition", 1.0)
        Set("upperPosition", 0.0)
        Set("match", "yP")
        Set("Label/atEdge", True)
        Set("Label/rotate", "90")
        Set("Label/position", "at-minimum")
        Set("TickLabels/color", "black")
        Set("MajorTicks/number", 2)
        Set("MinorTicks/number", 12)
        Set("GridLines/hide", True)
        To("..")
        Add("contour", name="contour1", autoadd=False)
        To("contour1")
        Set("data", "Vabs_2D")
        Set("numLevels", 7)
        # Set('manualLevels', [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5] if param == 'Vabs' else [])
        Set("keyLevels", True)
        Set("ContourLabels/hide", False)
        To("..")
        for t in ("i", "w") if param == "t" else ("i",):
            if param == "t" and t == "w":
                Add("axis", name="yP", autoadd=False)
                To("yP")
                Set("hide", True)
                Set("autoRange", "exact")
                Set("autoMirror", False)
                Set("lowerPosition", 1.0)
                Set("upperPosition", 0.0)
                Set("MajorTicks/number", max(int(len_graph), 2))
                Set("MinorTicks/number", max(int(len_graph * 2), 4))
                To("..")
                Add("image", name=f"{param}_w", autoadd=False)
                To(f"{param}_w")
            else:
                Add("image", name=param, autoadd=False)
                To(param)
            Set(
                "data",
                f"zeros_like(Dim_DataP_{t}[0,0]) + {expr_param[f'{param}_{t}' if param=='t' else param]}{'*0.11111111111111' if param == 'Vdir' else ''}",
            )
            param_max = 0.3
            lim_max = axis_max.get(param, param_max) if param != "Vdir" else 360
            Set("min", -lim_max if is_antisymmetric(param) else axis_min.get(param, 0))
            Set("max", lim_max)
            Set(
                "colorMap",
                {"Vabs": "colormapAbs", "Vdir": "colormapDir", "t": "heat"}.get(
                    param, "blue-darkred"
                ),
            )
            if param == "t":
                Set("colorInvert", True)
            Set("yAxis", "yP")
            Set("drawMode", "default")
            To("..")
        if iparam == 0:
            Add(
                "xy",
                name="origin_and_autorange-don't_remove_if_only_xy-widget",
                autoadd=False,
            )
            To("origin_and_autorange-don't_remove_if_only_xy-widget")
            Set("marker", "none")
            Set("markerSize", "2pt")
            Set("color", "darkred")
            Set("xData", "v.dt64s2vsz(int32(array(DISPtime[0], 'M8[s]')))")
            Set("yData", "DISP_vecY0 + zeros(2)")
            Set("hide", False)
            Set("yAxis", "y[0,1]")
            Set("PlotLine/color", "grey")
            Set("PlotLine/width", "0.5pt")
            Set("PlotLine/style", "dotted")
            Set("MarkerLine/width", "1.5pt")
            Set("MarkerLine/hide", False)
            Set("MarkerFill/hide", False)
            To("..")
        elif iparam == len_params - 1:
            Add("axis-function", name="xShow - do not limit", autoadd=False)
            To("xShow - do not limit")
            Set("linked", True)
            if not x_units_by_label:
                Set(
                    "label",
                    f"%{{{{fDisp_date_u(f'/{pg_name}/grid1/x', 'disp_time_span'{', b_nl=True' if x_units_nl else ''}{', allow3rows=True' if cfg_plot.graph_width < 10 else ''}, higher={not zoom})}}}}%",
                )
            Set("linkedaxis", "x")
            # Set('otherPosition', -0.05)
            Set("match", "x")
            Set("Label/position", "at-maximum")
            Set("GridLines/hide", True)
            x_datetime_ticks(False, cfg_plot)
            To("..")
    To("..")
    To("..")
    if WidthGrade > WidthGrades["VeryNarrow"]:
        Add("label", name="Title", autoadd=False)
        To("Title")
        Set(
            "label",
            "%{{v.c1(v.pl(I[info_incl['device']]))}}% %{{v.pl('{at} {depth}'.format_map(I))}}%, %{{I['m']}}% "
            + ",\u2009".join([
                (r"\\" if WidthGrade == WidthGrades["Narrow"] else "\u2009")
                + k[1:].replace("_", "+")  # display names of combined devices joined by '+'
                + (lambda p, b, bd, *kw: "" if b is None else f": {(b - bd):.0f}")(
                    *cus.DISPdevices_info[k[1:]]
                )
                for k in ids_i_2d
                if k[1:].startswith("i")
            ]),
        )
        label_Title_format(cfg_plot, graphs_height_sum=10)
        # , grid_leftMargin=grid_leftMargin, grid_horMargins_sum=grid_horMargins_sum
        To("..")

    str_units_added = label_xUnits_add(
        f"/{pg_name}/grid1/x",
        graphs_height_sum,
        y_cm=(0.06 if WidthGrade == WidthGrades["VeryNarrow"] else 0.7),
        nl=(WidthGrade == WidthGrades["Narrow"]),
        force=x_units_by_label,
        cfg_plot=cfg_plot
    )
    # y_cm = 1 can be used because here is space in right margin
    To("..")




def pg_progress_3d(graphs, aspect=2, b_zabor=False, cfg_plot=None):
    """
    Create a 3D progressive vector diagram page.

    :param graphs: list of process IDs to plot
    :param aspect: float - aspect ratio for the graph (default: 2)
    :param b_zabor: bool - whether to use Zabor-specific settings (default: False)
    :param cfg_plot: plotting configuration dictionary with the following required fields:
        - grid_horMargins_sum: float - sum of horizontal margins
        - grid_bottomMargin: float - bottom margin for grid
    :globals:
    :param device_wind: bool - whether wind device is present
    :param Zabor_shift: float - shift value for Zabor
    :param Zabor_range_dt: str - Zabor range datetime specification
    :param zb_i_wind_st: array - wind start indices for Zabor
    :param zb_i_wind_en: array - wind end indices for Zabor
    :param iu_Wind: array - wind u-component indices
    :param dt_Wind: array - wind datetime values
    :param Wind_to_current_coef: float - wind to current conversion coefficient
    :param zb_i_st: array - Zabor start indices
    :param zb_i_en: array - Zabor end indices
    :param zb_label_i: array - Zabor label indices
    :param z_i: array - z-coordinate values
    :param Progress_x_lims: array - x-axis limits for progress plot
    :param Progress_y_min: array - minimum y values for progress plot
    :param disp_bin_i: array - display bin indices
    :param bin_t0st_i: array - bin start time indices
    :param disp_pgs3d_lines_i: array - display 3D lines indices
    :param Zabor_vert_lines_freq: int - frequency of vertical lines for Zabor
    :param cum_u: str or array - cumulative u values (calculated if not provided)
    :param cum_v: str or array - cumulative v values (calculated if not provided)
    """
    ## pg_progress_3d
    if b_zabor:
        pg_name = "Vprogress3D_zabor"
        str_xy_scale = ""
        dist_units = (
            "м" if fv.lang == "ru" else "m"
        )  # Veusz can not use "%{{I['m']}}%" in 3D graphs
        cum_u = "zb2d_u_cum"
        cum_v = "zb2d_v_cum"
    else:
        pg_name = "Vprogress3D"
        str_xy_scale = "0.001"
        dist_units = "km"
        cum_u = "bin_u_cum2d"
        cum_v = "bin_v_cum2d"

    str_mulr = lambda s: f"{s}*" if s else ""

    print(f"Page {pg_name}", end=": ")

    scale_wind_more_str = "*Wind_to_current_coef" if device_wind and len(graphs) > 1 else ""
    graphs = [pid for pid in graphs if not pid.startswith(("_p", "_w"))]

    graph_height = 10
    grid_leftMargin = 1.3
    cfg_plot = replace(cfg_plot, graph_width=graph_height * aspect)
    grid_horMargins_sum = cfg_plot.get('grid_horMargins_sum', 0)
    grid_rigtMargin = grid_horMargins_sum - grid_leftMargin
    map_bottomMargin = 0.8

    clr_param = "%{{v.c1(I['depth'])}}%"
    clr_unit = "%{{I['m']}}%"

    Add("page", name=pg_name, autoadd=False)
    To(pg_name)
    Set("width", f"{cfg_plot.graph_width + grid_leftMargin + grid_rigtMargin:g}cm")
    Set("height", f"{graph_height + (cfg_plot.grid_bottomMargin - map_bottomMargin):g}cm")
    Set("Background/hide", False)
    Add("label", name="Title", autoadd=False)
    To("Title")
    Set(
        "label",
        "%{{v.c1('{progressive vector diagram}'.format_map(I))}}%. %{{v.c1(v.pl(I['current'])) + "
        "' {{by {device}}}'.format_map(info_incl).format_map(I)}}%\u2009"
        "%{{v.str_time_range(*f(v.vsz2dt64s(DATA('time_span_i')).tolist))}}%",
    )
    Set("xPos", [0.5])
    Set("yPos", [0.00436])
    Set("alignHorz", "centre")
    Set("margin", "1pt")
    Set("Text/size", "12pt")
    Set("Text/bold", True)
    Set("Background/transparency", 90)
    To("..")
    Add("grid", name="grid_diagram", autoadd=False)
    To("grid_diagram")
    Set("leftMargin", f"{grid_leftMargin}cm")
    Set("rightMargin", f"{grid_rigtMargin}cm")
    Set("topMargin", "0cm")
    Set("bottomMargin", f"{cfg_plot.grid_bottomMargin - map_bottomMargin:g}cm")
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
    Add("scene3d", name="scene3d1", autoadd=False)
    To("scene3d1")
    Set("xRotation", 75.0)
    Set("yRotation", 17.0)
    Set("zRotation", -1.0)
    Set("distance", 8.0)
    Set("rendermode", "bsp")
    Set("rightMargin", "0cm")
    Set("topMargin", "0cm")
    Set("bottomMargin", "0.5cm")
    Set("Lighting1/color", "white")
    Set("Lighting1/intensity", 70.0)
    Set("Lighting1/x", 40.0)
    Set("Lighting1/y", -40.0)
    Set("Lighting1/z", 40.0)
    Add("graph3d", name="graph3d1", autoadd=False)
    To("graph3d1")
    Set("zSize", 0.3)
    Set("Border/color", "transparent")
    Add("axis3d", name="lon", autoadd=False)
    To("lon")
    Set("label", f"\\italic{{{dist_units}, E}}")
    Set("autoRange", "exact")
    Set("Line/transparency", 70.0)
    Set("GridLines/width", 0.25)
    Set("GridLines/style", "dotted")
    Set("GridLines/hide", False)
    To("..")
    Add("axis3d", name="lat", autoadd=False)
    To("lat")
    Set("label", f"\\italic{{{dist_units}, N }}")
    Set("autoRange", "exact")
    Set("direction", "y")
    Set("Line/transparency", 70.0)
    Set("MajorTicks/number", 3)
    Set("MinorTicks/number", 10)
    Set("GridLines/width", 0.25)
    Set("GridLines/style", "dotted")
    Set("GridLines/hide", False)
    To("..")
    Add("axis3d", name="z", autoadd=False)
    To("z")
    Set("label", "z, {}".format("м" if fv.lang == "ru" else "m"))
    Set("autoRange", "exact")
    Set("direction", "z")
    Set("Line/transparency", 70.0)
    Set("MajorTicks/number", 5)
    Set("GridLines/width", 0.25)
    Set("GridLines/style", "dotted")
    Set("GridLines/hide", False)
    To("..")
    if device_wind:
        Add("point3d", name="wind_line", autoadd=False)
        To("wind_line")
        Set("marker", "none")
        Set("markerSize", 3.0)
        if b_zabor:
            Set(
                "xData",
                "ravel([[i*Zabor_shift, *(i*Zabor_shift + min(dt_Wind, float64(timedelta64(timedelta64(1, Zabor_range_dt), 's')))*Wind_to_current_coef*cumsum(u_Wind[sl_(iu_Wind)][slice(*se)])), nan] for i, se in enumerate(int32([zb_i_wind_st, zb_i_wind_en]).T)])",
            )
            Set(
                "yData",
                "min(dt_Wind, float64(timedelta64(timedelta64(1, Zabor_range_dt), 's')))*Wind_to_current_coef*ravel([[0, *cumsum(v_Wind[sl_(iu_Wind)][slice(*se)]), nan] for se in int32([zb_i_wind_st, zb_i_wind_en]).T])",
            )
            Set(
                "zData",
                "zeros(int(sum(zb_i_wind_en - zb_i_wind_st))+zb_i_wind_st.size*2)",
            )
        else:
            Set("xData", f"append(0, {str_mulr(str_xy_scale)}bin2_u_cum_Wind)")
            Set("yData", f"append(0, {str_mulr(str_xy_scale)}bin2_v_cum_Wind)")
            Set("zData", "zeros(bin2_u_cum_Wind.size+1)")
        Set("xAxis", "lon")
        Set("yAxis", "lat")
        Set("Color/min", -0.3)
        Set("Color/max", 1.0)
        Set("PlotLine/color", "red")
        Set("PlotLine/width", 2.0)
        Set("PlotLine/reflectivity", 5.0)
        Set("PlotLine/hide", False)
        Set("PlotLine/colorMap", "parula")
        Set("PlotLine/colorMapInvert", True)
        To("..")

    Add("point3d", name="xyz=0", autoadd=False)
    To("xyz=0")
    Set("marker", "none")
    Set("markerSize", 3.0)

    minmax = lambda cum_var_str: (
        "f(lambda mm: v.round_ceil_signed(mm, int(v.power_ceil(abs(diff(mm))))), "
        f"f(lambda d: [nanmin(d), nanmax(d)], DATA('{cum_var_str}')))"
    )
    if b_zabor:
        # for u, xyData, str_xy_shift in zip((cum_u, cum_v), ('xData', 'yData'), (' + zb_shift_2d', '')):
        Set(
            "xData",
            (
                f"[*v.shift_or_extend_lims([-Zabor_shift, Zabor_shift*zb_i_st.size], "
                f"{minmax(cum_u)}, scale=1), nan, "
                "*[k for i in range(0, Zabor_shift*zb_i_st.size, Zabor_shift) for k in [i, i, nan, i, i, nan]]]"
            ),
        )
        Set(
            "yData",
            (
                f"[0, 0, nan, *[*v.shift_or_extend_lims([-Zabor_shift, Zabor_shift], "
                f"{minmax(cum_v)}, scale=1), nan, 0, 0, nan]*zb_i_st.size]"
            ),
        )
        Set("zData", "[0, 0, nan, *[0, 0, nan, z_i[-1], 0, nan]*zb_i_st.size]")
    else:
        Set(
            "xData",
            (
                "[*v.shift_or_extend_lims(Progress_x_lims, "
                f"{minmax(cum_u)}, scale={str_xy_scale or '1'}), nan, 0, 0, nan, 0, 0]"
            ),
        )
        Set(
            "yData",
            (
                f"[0, 0, nan, *v.shift_or_extend_lims(Progress_y_min + float32([0, diff(Progress_x_lims) / 2]), "
                f"{minmax(cum_v)}, scale={str_xy_scale or '1'}), nan, 0, 0]"
            ),
        )
        Set(
            "zData",
            "[0, 0, nan, 0, 0, nan, z_i[-1], 0]"
            if device_wind
            else "z_i[[0, 0, -1, 0, 0, -1, 0, -1]]",
        )
    Set("xAxis", "lon")
    Set("yAxis", "lat")
    Set("PlotLine/color", "black")
    Set("PlotLine/width", 1.5)
    Set("PlotLine/reflectivity", 0.0)
    Set("PlotLine/hide", False)
    To("..")

    Add("point3d", name="datePoints", autoadd=False)
    To("datePoints")
    Set("marker", "linevert")
    Set("markerSize", 3.0)
    if b_zabor:
        for cum_var_str, xyData, str_xy_shift in zip(
            (cum_u, cum_v), ("xData", "yData"), (" + zb_shift_2d", "")
        ):
            Set(
                xyData,
                (
                    f"ravel(column_stack((({cum_var_str}{str_xy_shift})[:, int32(ravel((zb_label_i + "
                    "arange(1, (zb_i_en[0] - zb_i_st[0] + 2)*zb_label_i.shape[1], step=zb_i_en[0] - zb_i_st[0]+2)"
                    ").T))].T, nan+ravel(zb_label_i))))"
                ),
            )
        Set(
            "zData",
            "ravel(column_stack((tile(z_i, [zb_label_i.size, 1]), nan+ravel(zb_label_i))))",
        )
    else:
        Set(
            "xData",
            f"{str_mulr(str_xy_scale)}ravel(column_stack(({cum_u}[:, int32(disp_bin_i)].T, nan+disp_bin_i)))",
        )
        Set(
            "yData",
            f"{str_mulr(str_xy_scale)}ravel(column_stack(({cum_v}[:, int32(disp_bin_i)].T, nan+disp_bin_i)))",
        )
        Set(
            "zData",
            "ravel(column_stack((tile(z_i, [disp_bin_i.size, 1]), nan+disp_bin_i)))",
        )
    Set("xAxis", "lon")
    Set("yAxis", "lat")
    Set("PlotLine/color", "red")
    Set("PlotLine/style", "dotted")
    Set("PlotLine/transparency", 70.0)
    Set("PlotLine/hide", False)
    To("..")

    Add("point3d", name="V_line", autoadd=False)
    To("V_line")
    Set("marker", "none")
    Set("markerSize", 3.0)
    if b_zabor:
        Set("xData", "ravel(zb2d_u_cum + zb_shift_2d)")
        Set("yData", "ravel(zb2d_v_cum)")
        Set("zData", "ravel(repeat(transpose([z_i]), zb2d_u_cum.shape[1], axis=1))")
        Set(
            "Color/points",
            "ravel(repeat(transpose([z_i]), zb2d_u_cum.shape[1], axis=1))/max(z_i)",
        )
    else:
        Set("xData", f"{str_mulr(str_xy_scale)}ravel(column_stack(({cum_u}, nan+z_i)))")
        Set("yData", f"{str_mulr(str_xy_scale)}ravel(column_stack(({cum_v}, nan+z_i)))")
        Set(
            "zData",
            "ravel(column_stack((repeat(transpose([z_i]), bin_t0st_i.size+1, axis=1) , nan+z_i)))",
        )
        Set(
            "Color/points",
            "ravel(column_stack((repeat(transpose([z_i])/max(z_i), bin_t0st_i.size+1, axis=1) , nan+z_i)))",
        )
    Set("xAxis", "lon")
    Set("yAxis", "lat")
    Set("Color/min", -0.3)
    Set("Color/max", 1.0)
    Set("PlotLine/color", "darkblue")
    Set("PlotLine/width", 2.0)
    Set("PlotLine/reflectivity", 5.0)
    Set("PlotLine/hide", False)
    Set("PlotLine/colorMap", "parula")
    Set("PlotLine/colorMapInvert", True)
    To("..")

    Add("point3d", name="xy=0", autoadd=False)
    To("xy=0")
    Set("marker", "none")
    Set("markerSize", 3.0)
    str_pos = (
        "*[k for i in range(0, zb_i_st.size*Zabor_shift, Zabor_shift) for k in [i, i, nan]]"
        if b_zabor
        else "0, 0, nan"
    )
    Set(
        "xData",
        "tile([*v.shift_or_extend_lims([-Zabor_shift, Zabor_shift*zb_i_st.size], "
        f"{minmax(cum_u)}, scale={str_xy_scale or '1'}), nan, {str_pos}], z_i.size)",
    )
    str_pos = (
        "*v.shift_or_extend_lims([-Zabor_shift, Zabor_shift], "
        f"{minmax(cum_v)}, scale={str_xy_scale or '1'}), nan"
    )
    repeat_str = lambda s: f"*[{s}]*zb_i_st.size" if b_zabor else s
    Set("yData", f"tile([0, 0, nan, {repeat_str(str_pos)}], z_i.size)")
    Set(
        "zData",
        f'z_i[ravel([[i, i, -1, {repeat_str("i, i, -1")}] for i in range(z_i.size)])]',
    )
    Set("xAxis", "lon")
    Set("yAxis", "lat")
    Set(
        "Color/points",
        f'repeat(z_i/max(z_i), {"3*(1 + zb_i_st.size)" if b_zabor else 6})',
    )
    Set("Color/min", -0.3)
    Set("PlotLine/color", "black")
    Set("PlotLine/width", 1.5)
    Set("PlotLine/style", "dotted")
    Set("PlotLine/reflectivity", 0.0)
    Set("PlotLine/hide", False)
    Set("PlotLine/colorMap", "parula")
    Set("PlotLine/colorMapInvert", True)
    To("..")

    Add("point3d", name="lines_3d_vert", autoadd=False)
    To("lines_3d_vert")
    Set("marker", "none")
    Set("markerSize", 3.0)
    if b_zabor:
        for u, xyData, str_xy_shift in zip(
            "uv", ("xData", "yData"), (" + zb_shift_2d", "")
        ):
            Set(
                xyData,
                (
                    "ravel(column_stack((apply_along_axis(lambda a: interp(arange(a.size*Zabor_vert_lines_freq),"
                    f"arange(a.size*Zabor_vert_lines_freq, step=Zabor_vert_lines_freq), a), 1, zb2d_{u}_cum{str_xy_shift}).T,"
                    "nan+empty(zb_shift_2d.shape[1]*Zabor_vert_lines_freq))))"
                ),
            )
        n_poins_on_layer_str = "zb2d_u_cum.shape[1]*Zabor_vert_lines_freq"
    else:
        for u, xyData in zip("uv", ("xData", "yData")):
            Set(
                xyData,
                f"0.001*ravel(column_stack(({u}_cum2d[:, int32(disp_pgs3d_lines_i)].T, nan+disp_pgs3d_lines_i)))",
            )
        n_poins_on_layer_str = "disp_pgs3d_lines_i.size"
    for prop, clr_norm_str in zip(("zData", "Color/points"), ("", "/max(z_i)")):
        Set(
            prop,
            (
                f"ravel(column_stack((tile(z_i{clr_norm_str}, [{n_poins_on_layer_str}, 1]), "
                f"nan+empty({n_poins_on_layer_str}))))"
            ),
        )
    Set("hide", False)
    Set("xAxis", "lon")
    Set("yAxis", "lat")
    Set("Color/min", -0.3)
    Set("PlotLine/color", "black")
    Set("PlotLine/style", "solid")
    # Set('PlotLine/transparency', 25.0)
    Set("PlotLine/reflectivity", 100.0)
    Set("PlotLine/hide", False)
    Set("PlotLine/colorMap", "parula")
    Set("PlotLine/colorMapInvert", True)
    To("..")

    Add("function3d", name="y=0", autoadd=False)
    To("y=0")
    Set("color", "red")
    Set("mode", "x=fn(y,z)")
    Set("fnx", "0")
    Set("fny", "0")
    Set("fnz", "0")
    Set("hide", False)
    Set("xAxis", "lon")
    Set("yAxis", "lat")
    Set("linesteps", 3)
    Set("surfacesteps", 3)
    Set("GridLine/hide", True)
    Set("Surface/color", "#377eb8")
    Set("Surface/transparency", 90.0)
    Set("Surface/hide", False)
    for pid in graphs:
        To("..")
        Add("function3d", name=f"z=z{pid}", autoadd=False)
        To(f"z=z{pid}")
        Set("color", "red")
        Set("mode", "z=fn(x,y)")
        Set("fnz", f"DISPdevice_info['{pid[(1 if pid[2].isdigit() else 2):]}']['d']")
        Set("hide", False)
        Set("xAxis", "lon")
        Set("yAxis", "lat")
        Set("linesteps", 5)
        Set("surfacesteps", 5)
        Set("GridLine/transparency", 70.0)
        Set("GridLine/hide", True)
        Set("Surface/color", "#377eb8")
        Set("Surface/transparency", 90.0)
        Set("Surface/hide", False)
    To("..")
    To("..")
    To("..")
    To("..")
    Add("graph", name="background", autoadd=False)
    To("background")
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
        """\\italic{%{{f(lambda k: I['km'] if k==1e-3 else f"I['m']·10^{{{-log10(k):g}}}", SETTING("""
        f"""'/{pg_name}/grid_diagram/map0/x_km/datascale'))}}}}%,\\\\{{I['E']}} }}""",
    )
    Set("hide", False)
    Set("xPos", [1.09])
    Set("yPos", [-0.035])
    Set("xAxis", "x_km")
    Set("yAxis", "y_km")
    Set("alignHorz", "right")
    Set("alignVert", "bottom")
    Set("margin", "0.5pt")
    Set("Background/transparency", 100)
    To("..")
    To("..")
