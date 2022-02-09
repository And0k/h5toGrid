#!/usr/bin/env python3
# coding:utf-8
"""
  Author:  Andrey Korzh <ao.korzh@gmail.com>
  Purpose: Golden Software Surfer Automation Rutines
  Created: 25.07.2020
  Modified: 25.07.2020
"""
import functools
from pathlib import Path
from typing import Any, Dict, Callable, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union
import logging
from datetime import datetime, timedelta
import re
from tempfile import TemporaryDirectory
import numpy as np

from win32com.client import constants, Dispatch, CastTo, pywintypes, gencache
# python "c:\Programs\_coding\WinPython3\python-3.5.2.amd64\Lib\site-packages\win32com\client\makepy.py" -i "c:\Program Files\Golden Software\S
# urfer 13\Surfer.exe"
# Use these commands in Python code to auto generate .py support


from utils2init import standard_error_info, LoggingStyleAdapter

lf = LoggingStyleAdapter(__name__)
Surfer = None
obj_interface = {}
temp_dir_path = r'C:\Windows\Temp'

# try:  # try get griddata_by_surfer() function reqwirements

def griddata_by_surfer(
        ctd,
        path_stem_pattern: Union[str, Path] = r'%TEMP%\xyz{}',
        margins: Union[bool, Tuple[float, float], None] = True,
        xCol: str='Lon', yCol: str='Lat', zCols: Sequence[str]= None,
        SearchEnable=True, BlankOutsideHull=1,
        DupMethod=15, ShowReport=False,  # DupMethod=15=constants.srfDupAvg
        **kwargs):
    """
    Grid by Surfer
    :param ctd: pd.DataFrame
    :param path_stem_pattern:
    :param margins: extend grid size on all directions:
      - True - use 10% if limits (xMin...) passed else use InflateHull value
      - Tuple[float, float] - use this values to add to edges. Note: limits (xMin...) args must be provided in this case
    :param zCols: z column indexes in ctd
    :param xCol: x column index in ctd
    :param yCol: y column index in ctd
    :param kwargs: other Surfer.GridData4() arguments
    :return:
    """
    global Surfer
    if not Surfer:
        gencache.EnsureModule('{54C3F9A2-980B-1068-83F9-0000C02A351C}', 0, 1, 4)
        try:
            Surfer = Dispatch("Surfer.Application")
        except pywintypes.com_error as e:
            print("Open Surfer! ", standard_error_info(e))
            try:
                Surfer = Dispatch("Surfer.Application")
            except pywintypes.com_error as e:
                print("Open Surfer! ", standard_error_info(e))
                raise
    try:
        tmpF = f"{path_stem_pattern.format('~temp')}.csv"
    except AttributeError:
        path_stem_pattern = str(path_stem_pattern)
        tmpF = f"{path_stem_pattern.format('~temp')}.csv"
    kwargs['xCol'] = ctd.dtype.names.index(xCol) + 1
    kwargs['yCol'] = ctd.dtype.names.index(yCol) + 1
    if zCols is None:
        zCols = list(ctd.dtype.names)
        zCols.remove(xCol)
        zCols.remove(yCol)

    izCols = [ctd.dtype.names.index(zCol) + 1 for zCol in zCols]  # ctd.columns.get_indexer(zCols) + 1
    np.savetxt(tmpF, ctd, header=','.join(ctd.dtype.names), delimiter=',', comments='')

    if margins:
        if isinstance(margins, bool):
            margins = [0, 0]
            for i, coord in enumerate(('x', 'y')):
                if (f'{coord}Min' in kwargs) and (f'{coord}Max' in kwargs):
                    margins[i] = (kwargs[f'{coord}Max'] - kwargs[f'{coord}Min']) / 10
                elif kwargs.get('InflateHull'):
                    margins[i] = kwargs['InflateHull']

        kwargs['xMin'] -= margins[0]
        kwargs['xMax'] += margins[0]
        kwargs['yMin'] -= margins[1]
        kwargs['yMax'] += margins[1]

        if kwargs.get('SearchRad1') is None:
            kwargs['SearchRad1'] = margins[1] * 3

        if kwargs.get('InflateHull') is None:
            kwargs['InflateHull'] = margins[1]


        # const={'srfDupAvg': 15, 'srfGridFmtS7': 3}
    # gdal_geotransform = (x_min, cfg['out']['x_resolution'], 0, -y_min, 0, -cfg['y_resolution_use'])
    for i, kwargs['zCol'] in enumerate(izCols):
        outGrd = path_stem_pattern.format(zCols[i]) + '.grd'
        try:
            Surfer.GridData4(
                Algorithm=constants.srfKriging, DataFile=tmpF, OutGrid=outGrd,
                SearchEnable=(SearchEnable and ctd.size > 3), BlankOutsideHull=BlankOutsideHull, DupMethod=DupMethod,
                ShowReport=ShowReport, **kwargs)
        except pywintypes.com_error as e:
            print(standard_error_info(e))
            if i >= 0:  # True but in debug mode you can change to not raise and continue without side effects by set i=-1
                raise
    return margins
# except Exception as e:
#
#     print('\nCan not initialiase Surfer.Application! {:s}'.format(standard_error_info(e)))
#
#
#     def griddata_by_surfer(
#             ctd, path_stem_pattern: Union[str, Path] = r'%TEMP%\xyz{}',
#             margins: Union[bool, Tuple[float, float], None] = True,
#             xCol: str='Lon', yCol: str='Lat', zCols=Sequence[str],
#             SearchEnable=True, BlankOutsideHull=1,
#             DupMethod=15, ShowReport=False,
#             **kwargs):
#         pass


def get_objpaths(re_ovr, re_shp=None, f_path=lambda p: p.name, f_text=None) -> Dict[Tuple[int, str], Any]:
    """
    In currently open document finds paths to matched objects/properties. Returns dict with keys shape (type, name) and
    values that is dict of overlays {(type, number): properties that points to data or can be changed if data changed}.

    :param re_ovr: saves only matched overlays' properties
    :param re_shp: if specified, save also properties of shapes' which name is matched to
    :param f_path: function(data_path: Path) which replaces data path in output obj_path, default: extract name
    :param f_text: function(text_obj_name: str) which replaces Text in matched name text_obj_name
    :return: obj_path of exported grids: {(shp.Type, shp.Name): {(ovrl.Type, ovr_i): data}}
    """
    doc = Surfer.ActiveDocument
    shapes = doc.Shapes
    obj_path = {}
    if shapes.Count > 0:
        for i_shp, shp in enumerate(shapes):
            if re_shp and not re.match(re_shp, shp.Name):
                continue
            if shp.Type == constants.srfShapeMapFrame:
                overlays = shp.Overlays
                if shp.Name.upper() != "ICON":  # Do not touch Icons
                    for ovr_i, ovrl in enumerate(overlays):
                        if not re.match(re_ovr, ovrl.Name):
                            continue
                        if ovrl.Type in (constants.srfShapeContourMap, constants.srfShapeImageMap):
                            ovrl = CastTo(ovrl, obj_interface[ovrl.Type])
                            data = {'GridFile': f_path(Path(ovrl.GridFile))}
                        elif ovrl.Type == constants.srfShapeVector2Grid:  # srfShapeVectorMap
                            ovrl = CastTo(ovrl, obj_interface[ovrl.Type])
                            data = {
                                'SetInputGrids': {
                                    'GridFileName1': f_path(Path(ovrl.AspectGridFile)),    # East component
                                    'GridFileName2': f_path(Path(ovrl.GradientGridFile)),  # North component
                                    'AngleSys': ovrl.AngleSystem,
                                    'CoordSys': ovrl.VecCoordSys
                                    },
                                'SetScaling': {
                                    'Type': constants.srfVSMagnitude,
                                    'Minimum': ovrl.MinMagnitude,
                                    'Maximum': ovrl.MaxMagnitude
                                    }  # also saves command to recover vector limits that is needed after grid replacing
                                    }
                        # elif:
                        else:
                            data = None
                        if data:
                            if ':' in shp.Name:  # Cansel modified Name effect of selected shape
                                shp.Deselect()
                            if (shp.Type, shp.Name) in obj_path:
                                obj_path[(shp.Type, shp.Name)][(ovrl.Type, ovr_i)] = data
                            else:
                                obj_path[(shp.Type, shp.Name)] = {(ovrl.Type, ovr_i): data}

                            if ovrl.Type in (constants.srfShapeContourMap, constants.srfShapeImageMap):
                                # also save color limits that need to recover after grid replacing
                                try:
                                    obj_path[(shp.Type, shp.Name)][(ovrl.Type, ovr_i)].update(
                                        {'ColorMap.SetDataLimits': {
                                            'DataMin': ovrl.ColorMap.DataMin,
                                            'DataMax': ovrl.ColorMap.DataMax}
                                         })
                                except AttributeError:
                                    obj_path[(shp.Type, shp.Name)][(ovrl.Type, ovr_i)].update(
                                        {'FillForegroundColorMap.SetDataLimits': {
                                            'DataMin': ovrl.FillForegroundColorMap.DataMin,
                                            'DataMax': ovrl.FillForegroundColorMap.DataMax}
                                         },
                                        # {'ColorMap.ApplyFilltoLevels': {}}  # needed?
                                        )
                                    pass  # IContourLayer now has not ColorMap property but FillForegroundColorMap

                            print(list(data.values())[0])
                        # if ovrl.Type == srfShapePostmap:
                        # if ovrl.Type == srfShapeBaseMap:
                        # else:
                        #     print('What?')

                        # if b_setNames:
                        #     ovrl.Name= File + ovrl.Name Else ovrl.Name= Left(ovrl.Name,17)
                    if (shp.Type, shp.Name) in obj_path:  # data to replace was found
                        # also save shp limits that need to recover after grid replacing
                        obj_path[(shp.Type, shp.Name)].update({'SetLimits': {key: getattr(shp, key) for key in (
                            'xMin', 'xMax', 'yMin', 'yMax')},
                            'xMapPerPU': shp.xMapPerPU,
                            'yMapPerPU': shp.yMapPerPU})

            elif shp.Type == constants.srfShapeText:
                #cast = 'IText'
                text = f_text(shp.Name) if f_text else None
                data = {'Text': text or shp.Text}

                if (shp.Type, shp.Name) in obj_path:
                    if isinstance(obj_path[(shp.Type, shp.Name)], list):
                        obj_path[(shp.Type, shp.Name)].append(data)
                    else:
                        obj_path[(shp.Type, shp.Name)] = [obj_path[(shp.Type, shp.Name)], data]
                else:
                    obj_path[(shp.Type, shp.Name)] = data
    return obj_path


def gen_objects(obj_path: Dict[Tuple[Any, Any], Any],
                srf: Optional[str] = None
                ) -> Iterator[Tuple[Tuple[Any, Union[int, str, None]], Any]]:
    """
    Finds all objects that `obj_path` points to and yields them with `obj_path`'s values
    :param obj_path: dict as returned by objpath_fill()
    :param srf: Surfer plot file name (*.srf) if str, else currently opened Surfer.Document if None
    :return: ((ovrl, ovr_i), data_dict): ovr_i - overlay/shape index with same name and type as in input obj_path
    """
    doc = Surfer.ActiveDocument if srf is None else Surfer.Documents.Open(srf)
    doc.Selection.DeselectAll()
    shapes = doc.Shapes

    def yield_obj_and_data(shape, ovr_data, shp_i=None):
        for ovr_or_shpprop, data_dict in ovr_data.items():
            if (shp_i is None) and isinstance(ovr_or_shpprop, tuple):
                ovr_t, ovr_i = ovr_or_shpprop
                ovrl = CastTo(shape.Overlays.Item(ovr_i + 1), obj_interface[ovr_t])
                yield (ovrl, ovr_i), data_dict
                continue
            # No overlays:
            yield (shape, shp_i), {ovr_or_shpprop: data_dict}

    for (shp_t, shp_n), ovr_data in obj_path.items():
        shape = shapes.Item(shp_n)
        if shape.Type == shp_t and not isinstance(ovr_data, list):
            yield from yield_obj_and_data(shape, ovr_data)
        else:
            # many objects with same Name exist
            k = 0
            for shp in shapes:
                if shp.Type == shp_t and shp.Name == shp_n:
                    yield from yield_obj_and_data(shp, ovr_data[k], k)
                    k += 1

    if srf:
        doc.Close(SaveChanges=constants.srfSaveChangesNo)


def load_my_var_grid_from_netcdf_closure_grid():

    grid = None

    def _load_my_var_grid_from_netcdf(prop: str, grid_file: str, ovr) -> str:
        """
        Selects netcdf VariableName (var) and load it to overlay.
        For *File* properties having grid_file ends with '.nc' var is deterimined basing on:
        - for Vector2Grid: it is Vdir and Vabs
        - for others: extracted name part before '.' from Overlay or Shape name if Overlay's result is longer than 5 symbols.

        If grid_file ends with f'_{var}.grd' then consider that is already has only this var i.e. not needed to be converted
        :param prop: property
        :param grid_file:
        :param ovr:
        :param grid: None or Surfer.IGrid3
        :return: grid_file (may be modified)
        Note: Loading is done through temp file becaus following not works:
        Shapes = ovr.Parent.Parent.Shapes
        Shapes.Parent.Import2(grid_file, Options='VariableName=Vdir', FilterId='netcdf') not works (because nc is not graphic file?) (also with 'VariableName=Vdir' and/or RecordIndex')
        ovr.Grid.LoadFile2(grid_file, Options='VariableName="Vabs"') - not works because this Grid property is read only

        So it is possible only load with parameter VariableName in standalone Grid by LoadFile2() save it in standard grid (with only 1 variable) and assign GridFile to it
        """
        nonlocal grid

        if isinstance(grid_file, dict):
            return {k: _load_my_var_grid_from_netcdf(k, v, ovr) for k, v in grid_file.items()}

        if not ('File' in prop and grid_file.endswith('.nc')):  # only *File* properties has actually grid_file string
            return grid_file

        # Guess variable name
        if ovr.Type == constants.srfShapeVector2Grid:
            var = ('Vdir' if prop == 'GridFileName1' else 'Vabs')
            # todo determine decart coordinates: if grid_file endswith('r') else ('Ve' if prop == 'GridFileName1' else 'Vn')

        else:
            reg = re.match(r'(?:map)?([^.\d]+)', ovr.Name)
            var = reg.group(1) if reg else ''  # .partition('.')[0]
            if not var or len(var) > 5:
                var = re.match(r'(?:map)?([^.\d]+)', ovr.Parent.Name).group(1)

        if grid_file.endswith(grid_file_end := f'_{var}.grd'):
            # grid file has one variable and will be loaded ok
            return grid_file

        # Convert extract variable from multivariable grid file and save it to new file with name ending with variable name
        if grid is None:
            grid = CastTo(Surfer.NewGrid(), 'IGrid3')  # 'IGrid2' for Surfer20
        grid.LoadFile2(grid_file, Options=f'VariableName={var}')

        grid_file_stem = Path(grid_file).stem
        new_grid_file = str(temp_dir_path / f'{grid_file_stem}{grid_file_end}')
        lf.debug('extract {:s} from {:s} grid to temporary grid', var, grid_file_stem)
        grid.SaveFile(new_grid_file, constants.srfGridFmtBinary)
        return new_grid_file

    return _load_my_var_grid_from_netcdf


load_my_var_grid_from_netcdf = load_my_var_grid_from_netcdf_closure_grid()


def objpath_data_paste(obj_path,
                       f_prop: Callable[[str, str, Any], str] = load_my_var_grid_from_netcdf
                       ):
    """
    Sets attributes of all objects that obj_path points with obj_path's values
    :param obj_path:
    :param f_prop: function to modify obj/property before applying it. If returned prop is None then skip
    property applying
    :return:
    """
    for ((obj, ovr_i), data_dict) in gen_objects(obj_path):
        # if 'AspectGridFile' in data_dict:
        #     obj.SetInputGrids(data_dict['AspectGridFile'], data_dict['GradientGridFile'])
        #     data_dict = data_dict.copy()
        #     del data_dict['AspectGridFile']
        #     del data_dict['GradientGridFile']

        for prop, val in data_dict.items():
            if f_prop:
                if (val := f_prop(prop, val, obj)) is None:
                    continue

            if isinstance(val, dict):
                # prop is a function name, val - contains function arguments
                if '.' in prop:
                    obj_method = obj
                    # function is defined in child object
                    for prop in prop.split('.'):
                        obj_method = getattr(obj_method, prop)
                else:
                    obj_method = getattr(obj, prop)
                obj_method(**val)
                continue

            setattr(obj, prop, val)


def srf_update_grids(re_ovr='*', re_shp=None, f_fname: Optional[Callable[[...], str]] = None,
                     f_text: Optional[Callable[[...], str]] = None, dir_in: Union[Path, str, None] = None,
                     srf_out: Union[Path, str, None] = None, srf_in: Union[Path, str, None] = None,
                     dry_run: bool = False, export_suffix: str = None, options: str = None, **kwargs):
    """
    Replaces all grids in opened pattern with same named grid files in path_dir_in and saves to srf and exports image
    srf file name increases to not overwrite existed files: todo rename existed files
    :param re_ovr: regular expression for names of overlays having data needed to be replaced
    :param re_shp: regular expression for names of shapes having data needed to be replaced
    :param f_fname: function(path_grd_old) that returns modified data name that will be used to replace data (grid).
    If returns absolute path then used as is the f_path in gen_objpaths
    :param f_text: function, same as in gen_objpaths
    :param dir_in: where search grids
    :param srf_out: where to save srfs made by replacing data in opened srf
    :param srf_in: path to srf to modify. If None srf must be opened before run this function
    :param dry_run: not modify srf if True
    :param export_suffix: exporting name suffix. If None then not export
    :param options: exporting Options
    :param kwargs: other exporting kwargs such as Quality for jpg
    :return: obj_path with replaced grids
    """

    doc = Surfer.ActiveDocument if srf_in is None else Surfer.Documents.Open(srf_in)
    path_dir_in = Path(dir_in or doc.Path)  # 'C:\Windows\TEMP'

    if f_fname:
        def ff_name_mod(p):
            """
            Prepend dir_in if returned path is not absolute
            :param p:
            :return:
            """
            name_grd = f_fname(p)
            return str((path_dir_in / name_grd) if not Path(name_grd).is_absolute() else name_grd)
    else:
        ff_name_mod = lambda p: str(path_dir_in / p.name)

    obj_path = get_objpaths(re_ovr, re_shp, f_path=ff_name_mod, f_text=f_text)

    if False:  # if need more complex changing of properties then just replace grids
        for (obj, ovr_i), data_dict in gen_objects(obj_path, srf_in):
            if obj.Parent.Type != 1:  # obj is overlay
                parent = obj.Parent
                shp_t = parent.Type
                shp_n = parent.Name
                ovr_t = obj.Type
                for prop, path_grd in data_dict.items():
                    if isinstance(path_grd, dict):  # not modify overlay if it is complex i.e. it has no data from some path
                        continue
                    # assign grid file
                    if prop.endswith('File'):
                        obj_grd = getattr(obj, prop[:-len('File')])
                        path_grd_old = Path(getattr(obj, prop))
                        data_dict[prop] = ff_name_mod(path_grd_old)
                        # obj_grd.SaveFile(FileName=path_grd, Format=constants.srfGridFmtS7)
                # data_dict['Name'] = name_grd  # to update shape's name
                obj_path[(shp_t, shp_n)][(ovr_t, ovr_i)] = data_dict
            else:
                shp_t = obj.Type
                shp_n = obj.Name
                for prop, val in data_dict.items():
                    if isinstance(val, dict):  # not save shape property if it is not a simple property
                        continue
                    val = getattr(obj, prop)
                    # assert (ovr_i is not None) == isinstance(obj_path[(shp_t, shp_n)], list)
                    (obj_path[(shp_t, shp_n)] if ovr_i is None else obj_path[(shp_t, shp_n)][ovr_i]).update({prop: val})

    if not dry_run:
        objpath_data_paste(
            obj_path,
            lambda prop, val, ovr: None if (
                # False
                # prop == 'SetLimits'  # or                     # skip keeping same size
                # ('File' in prop and val.endswith('.grd'))    # skip replacing 'grd'
                ) else (
                val  # load_my_var_grid_from_netcdf(prop, val, ovr)
                )
            )

    # Save srf
    path_srf_out = Path(srf_out) if srf_out else path_dir_in
    if path_srf_out.is_dir():
        path_srf_out = (doc_dir := path_srf_out) / (doc_name := doc.Name)
        doc_stem, doc_suffix = doc_name.rsplit('.', 1)
    else:
        doc_dir = path_srf_out.parent
        doc_stem, doc_suffix = path_srf_out.name.rsplit('.', 1)
    i = 0
    while path_srf_out.is_file():
        i += 1
        path_srf_out = doc_dir / f'{doc_stem}_{i}.{doc_suffix}'
    try:
        doc.SaveAs(str(path_srf_out))
    except pywintypes.pywintypes.com_error:
        lf.exception(f'Can not save to {path_srf_out}')

    if export_suffix:
        if options is None:
            options = (
                'Defaults=1,HDPI=300,VDPI=300,Quality=80' if export_suffix == '.jpg' else
                'Defaults=1,HDPI=300,VDPI=300'
            )
        doc.Export2(str(path_srf_out.with_suffix(export_suffix)), Options=options, **kwargs)

    return obj_path



def objpath_update_and_get_grids(obj_path, path_dir_out: Path, srf=None):
    """
    Exports grids of obj_path, and modifies obj_path values to point to these grids
    :param obj_path:
    :param srf:
    :param path_dir_out: outputs grids here and assigns this parent to grid file paths
    :return: obj_path of exported grids
    """

    if path_dir_out is None:
        doc = Surfer.ActiveDocument if srf is None else Surfer.Documents.Open(srf)
        path_dir_out = Path(doc.Path)  # 'C:\Windows\TEMP'
    for (obj, ovr_i), data_dict in gen_objects(obj_path, srf):
        if obj.Parent.Type != 1:  # obj is overlay
            parent = obj.Parent
            shp_t = parent.Type
            shp_n = parent.Name
            ovr_t = obj.Type
            for prop, path_grd in data_dict.items():
                if isinstance(path_grd, dict):  # not save overlay property if it is complex i.e. it is not a grid path
                    continue
                # assigns parent to grid file paths
                if prop.endswith('File'):
                    obj_grd = getattr(obj, prop[:-len('File')])
                    name_grd = Path(getattr(obj, prop)).name
                    path_grd = str(path_dir_out / name_grd)
                    if '85m' in shp_n:  # !!! Temporary
                        name_grd = name_grd.replace('R80', 'R85')
                        path_grd = str(path_dir_out / name_grd)
                        data_dict[prop] = path_grd
                    else:
                        data_dict[prop] = path_grd
                        obj_grd.SaveFile(FileName=path_grd, Format=constants.srfGridFmtS7)
            data_dict['Name'] = name_grd
            obj_path[(shp_t, shp_n)][(ovr_t, ovr_i)] = data_dict
        else:
            shp_t = obj.Type
            shp_n = obj.Name
            for prop, val in data_dict.items():
                if isinstance(val, dict):  # not save shape property if it is not a simple property
                    continue
                val = getattr(obj, prop)
                # assert (ovr_i is not None) == isinstance(obj_path[(shp_t, shp_n)], list)
                (obj_path[(shp_t, shp_n)] if ovr_i is None else obj_path[(shp_t, shp_n)][ovr_i]).update({prop: val})


    return obj_path


def objpath_tmp_grids_delete(obj_path: Mapping[Tuple[int, str], Any], path_dir_tmp: Path):
    """
    Removes temporary grid files
    :param obj_path:
    :param path_dir_tmp:
    :return:
    """
    for ((obj, ovr_i), data_dict) in gen_objects(obj_path):
        for prop, val in data_dict.items():
            if isinstance(prop, str) and prop.endswith('GridFile'):
                try:
                    path_grd = path_dir_tmp / Path(getattr(obj, prop)).name
                    path_grd.unlink()
                except Exception as e:
                    lf.warning('can not remove "{:s}"', path_grd)


def srfs_replace_pattern(path_srfs, re_ovr='*', re_shp=None,
                    path_dir_out: Union[Path, str, None] = None,
                    path_dir_tmp: Union[Path, str, None] = None):  # 'C:\Windows\TEMP'
    """
    Replaces all grids in opened pattern with extracted grids from files in path_srfs
    Note: open srf pattern before run this function!
    :param path_srfs: srfs having needed data
    :param re_ovr: regular expression for names of overlays having needed data
    :param re_shp: regular expression for names of shapes having needed data
    :param path_dir_out: where to save srfs made by replacing data in opened srf
    :param path_dir_tmp: where to save temporary grids
    :return: None
    """

    doc = Surfer.ActiveDocument
    shapes = doc.Shapes
    obj_path = get_objpaths(re_ovr, re_shp)

    path_dir_tmp = Path(path_dir_tmp or doc.Path)
    path_dir_out = Path(path_dir_out) or path_dir_tmp
    if Path(path_srfs).is_dir():
        path_srf_parent = Path(path_srfs)
        srf_pattern = '*.srf'
    else:
        path_srf_parent = Path(path_srfs).parent
        srf_pattern = Path(path_srfs).name
    srfs = list(path_srf_parent.glob(srf_pattern))
    print(f'Replacing grids in opened pattern using data from {len(srfs)} {srf_pattern} files\
    from {path_srf_parent}\n  to {path_dir_out}...')

    for i, srf in enumerate(srfs):
        print(f'{i}. {srf.stem}')
        Surfer.ScreenUpdating = False
        obj_path = objpath_update_and_get_grids(obj_path, path_dir_tmp, srf)
        objpath_data_paste(obj_path)
        
        path_doc = path_dir_out / Path(srf).name
        try:
            doc.SaveAs(str(path_doc))
        except pywintypes.pywintypes.com_error:
            lf.exception(f'Can not save to {path_doc}')
        Surfer.ScreenUpdating = True
        doc.Export2(str(path_doc.with_suffix('.png')), Options='HDPI=300,VDPI=300')
        objpath_tmp_grids_delete(obj_path, path_dir_tmp)


# ---
def export_all_grids(path_dir_out:Optional[Path]=None):
    """
    Export all grids from opened srf
    :param path_dir_out: where to save .grd files
    :return: None
    """
    doc = Surfer.ActiveDocument
    if path_dir_out is None:
        path_dir_out = Path(doc.Path)  # 'C:\Windows\TEMP'

    shapes = doc.Shapes
    # Surfer.ScreenUpdating = True
    if shapes.Count > 0:
        k= 0
        for shp in shapes:
            k += 1
            if shp.Type == constants.srfShapeMapFrame:
                overlays = shp.Overlays
                if shp.Name.upper() != "ICON":  # Do not touch Icons
                    for ovrl in overlays:
                        if ovrl.Type == constants.srfShapeContourMap:
                            ovrl = CastTo(ovrl, 'IContourLayer')
                            file_grid = (path_dir_out / Path(ovrl.GridFile).stem).with_suffix('.grd')
                            ovrl.Grid.SaveFile(FileName=str(file_grid), Format=constants.srfGridFmtS7)  #, Format:=srfGridFmtAscii / srfGridFmtS7
                        elif ovrl.Type == constants.srfShapeVectorMap:
                            ovrl = CastTo(ovrl, 'IVectorMap')
                            file_grid = (path_dir_out / Path(ovrl.AspectGridFile).stem).with_suffix('.grd')  # East component grid
                            ovrl.AspectGrid.SaveFile(FileName=str(file_grid), Format=constants.srfGridFmtS7)  #"1.grd"
                            file_grid = (path_dir_out / Path(ovrl.GradientGridFile).stem).with_suffix('.grd')  # North component grid
                            ovrl.GradientGrid.SaveFile(FileName=str(file_grid), Format=constants.srfGridFmtS7)
                        else:
                            file_grid = ''
                        if file_grid:
                            print(file_grid.stem)
                        # if ovrl.Type == srfShapePostmap:
                        # if ovrl.Type == srfShapeBaseMap:
                        # else:
                        #     print('What?')

                        # if b_setNames:
                        #     ovrl.Name= File + ovrl.Name Else ovrl.Name= Left(ovrl.Name,17)


# --- 20.02.2021

def xyz2d(xyz, b_show=False):
    """
    Reshape 2d array xyz of 3 columns (x,y,z) to 2d grid of z with size of (x, y).
    :param xyz: array of shape (N, 3) last dimension for x, y and z
    :param b_show: show matplotlib plot of result grid
    :return: (z2d, x_min, y_min, x_resolution, y_resolution) suits to input to save_2d_to_grd()
    """

    x_uniq = np.unique(xyz[:,0])
    y_uniq = np.unique(xyz[:,1])

    idx = np.lexsort(xyz[:, :-1].T, axis=0)
    if np.array_equal(idx, np.arange(xyz.shape[0])):
        # input data is conformed to grid
        idx = idx.reshape(y_uniq.size, x_uniq.size)
        x2d, y2d, z2d = xyz[idx, :].T
    else:
        print('input data is not conformed to grid')
        # I'm fairly sure there's a more efficient way of doing this...
        def get_z(xyz, x, y):
            ind = (xyz[:, (0, 1)] == (x, y)).all(axis=1)
            row = xyz[ind, :]
            return row[0, 2]

        x2d, y2d = np.meshgrid(x_uniq, y_uniq)
        z = np.array([get_z(xyz, x, y) for (x, y) in zip(np.ravel(x2d), np.ravel(y2d))])
        z2d = z.reshape(x2d.shape)

    x_min, x_max = x_uniq[[0, -1]]
    y_min, y_max = y_uniq[[0, -1]]
    x_resolution = np.diff(x2d[:2, 0]).item()
    y_resolution = np.diff(y2d[0, :2]).item()

    # check grid is ok
    assert x_min == x2d[0, 0]
    assert y_min == y2d[0, 0]
    assert x_resolution == (x_max - x_min) / (x_uniq.size - 1)
    assert y_resolution == (y_max - y_min) / (y_uniq.size - 1)

    if b_show:
        # graphics/interactivity
        if True:  # __debug__:
            import matplotlib

            matplotlib.rcParams['axes.linewidth'] = 1.5
            matplotlib.rcParams['figure.figsize'] = (16, 7)
            try:
                matplotlib.use(
                    'Qt5Agg')  # must be before importing plt (raises error after although docs said no effect)
            except ImportError:
                pass
            from matplotlib import pyplot as plt

            matplotlib.interactive(True)
            plt.style.use('bmh')

        plt.pcolormesh(x2d, y2d, z2d)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.show()

    return z2d, x_min, y_max, x_resolution, y_resolution


def save_2d_to_grd(z2d, x_min, y_max, x_resolution, y_resolution, file_grd):
    """

    :param z2d:
    :param x_min, y_max, x_resolution, y_resolution: z2d coordinates parameters
    :param file_grd: str (if without suffix then ".grd" will be added) or Path of output Surfer grid
    :return:
    """

    from grid2d_vsz import write_grd_fun

    # (M, N): the rows and columns = an image with scalar data

    gdal_geotransform = (x_min, x_resolution, 0, y_max, 0, -y_resolution)
    # [0] координата x верхнего левого угла
    # [1] ширина пиксела
    # [2] поворот, 0, если изображение ориентировано на север
    # [3] координата y верхнего левого угла
    # [4] поворот, 0, если изображение ориентировано на север
    # [5] высота пиксела
    write_grd_this_geotransform = write_grd_fun(gdal_geotransform)

    if not isinstance(file_grd, Path):
        file_grd = Path(file_grd)
        if file_grd.suffix != '.grd':
            file_grd = file_grd.with_suffix('.grd')

    write_grd_this_geotransform(file_grd, z2d)


def save_xyz_to_grd(xyz, file_grd, b_show=False):
    z2d, x_min, y_max, x_resolution, y_resolution = xyz2d(xyz, b_show)
    save_2d_to_grd(np.flipud(z2d.T), x_min, y_max, x_resolution, y_resolution, file_grd)


def txt2grd(file_txt, z_names=('u', 'v', 'Wabs'), z_icols=(2, 3, 4), b_show=False):
    """

    :param file_txt: text file to read data: lat, lon, u, v, abs...
    :param z_names: columns names for z values
    :return:
    """
    file_txt = Path(file_txt)
    file_parent = file_txt.parent
    file_no_sfx = file_txt.stem

    usecols = [0, 1]; usecols.extend(z_icols)
    xyz = np.loadtxt(file_txt, usecols=usecols)
    for z_name, i_z in zip(z_names, z_icols):
        save_xyz_to_grd(xyz[:, (1, 0, i_z)], file_parent / f'{file_no_sfx}_{z_name}.grd', b_show)  # y = lat, x = lon


def many_txt2grd(file_txt_path, z_names=('u', 'v', 'Wabs'), z_icols=(2, 3, 4), b_show=False):
    file_txt_path = Path(file_txt_path)
    for i, file_txt in enumerate(file_txt_path.parent.glob(file_txt_path.name)):
        print(i, end=', ')
        txt2grd(file_txt, z_names, z_icols)

#

def invert_bln(file_bln_in, file_bln_out=None, delimiter=','):
    """
    Only one polygon supported.
    :param file_bln_in:
    :param file_bln_out:
    :param delimiter: if not ',' useful to invert 1st column of text files of other types
    :return:
    """
    if not file_bln_out:
        p_in = Path(file_bln_in)
        file_bln_out = p_in.with_name(f'{p_in.stem}_out').with_suffix(p_in.suffix)
    with open(file_bln_in, 'rb') as f:
        header = f.readline()
        bln = np.loadtxt(f, dtype={'names': ('x', 'y'), 'formats': ('f4', 'f4')}, skiprows=0, delimiter=delimiter)
    bln['x'] = bln['x'].max() - bln['x']
    np.savetxt(file_bln_out, bln, fmt='%g', delimiter=delimiter, header=header.strip().decode('latin'), comments='', encoding='ascii')


def range_till_stop(start, stop, step):
    """
    Generates values from start to including stop and works for any type with defined '<=' and '+' operations
    :param start:
    :param stop:
    :param step:
    :return:
    """
    while start <= stop:
        yield start
        start += step


########################################################################################################################
# Run examples subfunctions ############################################################################################

def current_file_name_replace(fname, old_str, new_str):
    return fname.name.replace(old_str, new_str)
    # .replace('_thetao', '_so')
    # .replace('_Vabs', '_o2')

    # def rep(match_obj):
    #     return {
    #         '004m': '020.0m',
    #         '007m': '041.0m',
    #         '010m': '061.0m',
    #         '014m': '080.0m',
    #         }.get(gr0 := match_obj.group(0), gr0)
    # return re.sub('\d\d\dm', rep, fname.name.replace(old_str, new_str).replace('_V_', '_'))

    #
    # .replace('_Vdir.grd', '_Vdir.nc').replace('_Vabs.grd', '_Vabs.nc')

def current_text(shp_name, date):
    if shp_name == 'PositionText':
        return f'CMEMS NEMO\n{date:%d.%m.%y} 12:00UTC'
    # lambda shp_name: f'{date:%Y-%m-%d}' if shp_name == 'PositionText' else None,

# Run examples #########################################################################################################

def do():
    """
    Update grids
    """
    dir_in = Path(
        r'd:\workData\BalticSea\_other_data\_model\Copernicus\section_z\211030ABP48@CMEMS\211030ABP48V,so,thetao,sob,o2,o2b@CMEMS\V,so,thetao,sob,o2,o2b(lon,lat)_54.3583-56.0082,18.0138-21.0417'
        # r'd:\workData\BalticSea\_other_data\_model\Copernicus\section_z\211030ABP48@CMEMS\211030ABP48V,so,thetao,sob,o2,o2b@CMEMS\V,thetao,so,o2(dist,depth)'
        # r'd:\workData\BalticSea\_other_data\_model\Copernicus\section_z\211125BalticSpit@CMEMS\V(dist,depth)'
        # r'd:\workData\BalticSea\_other_data\_model\Copernicus\section_z\211125BalticSpit@CMEMS\V(lon,lat)_54.3583-56.0082,18.0138-21.0417'
        )
    old_str = '211102'
    out_prefix = dir_in.name.partition('_')[0]

    for date in range_till_stop(datetime.fromisoformat('2021-10-30'), datetime.fromisoformat('2021-11-02'),
                                timedelta(days=1)):
        new_str = f'{date:%y%m%d}'  # '211202'211125
        # if old_str == new_str:
        #     continue

        # Update grids
        srf_update_grids(
            re_shp=r'^((?:\|V\||vectors|theta|S|DO|map|PositionText).*)$',
            re_ovr=r'((?!^bathymetry.*)^.*\.(grd|nc))$',
            f_fname=lambda fname: current_file_name_replace(fname, old_str, new_str),
            f_text=functools.partial(current_text, date=date),
            dir_in=dir_in,
            srf_out=dir_in.parent / f'{out_prefix}{new_str}.srf',
            export_suffix='.jpg'
            )
        old_str = new_str


def do1():
    """
    srfs_replace_pattern() with my settings
    """
    srfs_replace_pattern(
        path_srfs=r'd:\WorkData\BalticSea\_other_data\_model\POM_GMasha\190801_80,85,90m\_srf',
        re_ovr=r'.*\d_filt_',
        re_shp='\d\dm|PositionText',
        path_dir_out=r'd:\WorkData\BalticSea\_other_data\_model\POM_GMasha\190801_80,85,90m\_srf_bold_isobaths',
        )


def do_in_surfer(f_do):
    """
    Run using Surfer
    :return:
    """
    global Surfer, temp_dir_path
    if not Surfer:
        gencache.EnsureModule('{54C3F9A2-980B-1068-83F9-0000C02A351C}', 0, 1, 4)
        Surfer = Dispatch("Surfer.Application")
        # interfaces to get specific obj methods by calling Cast(obj, interface):
        obj_interface.update(                                   # old interface in comments:
            {constants.srfShapeContourMap: 'IContourLayer',     # IContourMap
             constants.srfShapeVectorMap: 'IVectorMap',
             constants.srfShapeImageMap: 'IColorReliefLayer2',  # IImageLayer2
             constants.srfShapeVector2Grid: 'IVectorLayer',
             constants.srfShapeMapFrame: 'IMapFrame2',
             }
            )
    with TemporaryDirectory(prefix='~GS') as temp_dir:
        temp_dir_path = Path(temp_dir)
        try:
            Surfer.ScreenUpdating = False

            # do important things now!
            f_do()

        finally:
            Surfer.ScreenUpdating = True


def main():
    do_in_surfer(do)


def main1():
    """
    # Convert txt to grd
    """
    many_txt2grd(r'd:\WorkData\BalticSea\_other_data\_model\POM_GMasha\201001_ABP46\wind\*.txt')
    return


if __name__ == '__main__':
    main()


