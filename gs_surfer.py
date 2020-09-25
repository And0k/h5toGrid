#!/usr/bin/env python3
# coding:utf-8
"""
  Author:  Andrey Korzh <ao.korzh@gmail.com>
  Purpose: Golden Software Surfer Automation Rutines
  Created: 25.07.2020
  Modified: 25.07.2020
"""
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union
import logging
import re
import numpy as np

from win32com.client import constants, Dispatch, CastTo, pywintypes, gencache
# python "c:\Programs\_coding\WinPython3\python-3.5.2.amd64\Lib\site-packages\win32com\client\makepy.py" -i "c:\Program Files\Golden Software\S
# urfer 13\Surfer.exe"
# Use these commands in Python code to auto generate .py support


from utils2init import standard_error_info

l = logging.getLogger(__name__)
Surfer = None
obj_interface = {}

# try:  # try get griddata_by_surfer() function reqwirements

def griddata_by_surfer(
        ctd, path_stem_pattern: Union[str, Path] = r'%TEMP%\xyz{}',
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


def objpath_fill(re_ovr, re_shp=None) -> Dict[Tuple[int, str], Any]:
    """
    Finds paths to objects/propeerties having data, return dict of paths with values that points to data

    :param path_dir_out:
    :param re_ovr: saves only matched overlays' properties
    :param re_shp: if specified, save also matched shapes' properties
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
                            data = {'GridFile': str(Path(ovrl.GridFile).name)}
                        elif ovrl.Type == constants.srfShapeVector2Grid: #srfShapeVectorMap
                            ovrl = CastTo(ovrl, obj_interface[ovrl.Type])
                            data = {'AspectGridFile': str(Path(ovrl.AspectGridFile).name),      # East component
                                    'GradientGridFile': str(Path(ovrl.GradientGridFile).name),  # North component
                                    'SetScaling': {'Type': constants.srfVSMagnitude, 'Minimum': ovrl.MinMagnitude,
                                         'Maximum': ovrl.MaxMagnitude}  # also saves comand to recover vector limits that is needed after grid replacing
                                    }
                        #elif :
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
                                obj_path[(shp.Type, shp.Name)][(ovrl.Type, ovr_i)].update(
                                    {'ColorMap.SetDataLimits': {
                                        'DataMin': ovrl.ColorMap.DataMin,
                                        'DataMax': ovrl.ColorMap.DataMax}
                                                                })

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

            elif re_shp:
                if shp.Type == constants.srfShapeText:
                    #cast = 'IText'
                    data = {'Text': shp.Text,  # East component
                            }
                    if (shp.Type, shp.Name) in obj_path:
                        if isinstance(obj_path[(shp.Type, shp.Name)], list):
                            obj_path[(shp.Type, shp.Name)].append(data)
                        else:
                            obj_path[(shp.Type, shp.Name)] = [obj_path[(shp.Type, shp.Name)], data]
                    else:
                        obj_path[(shp.Type, shp.Name)] = data
    return obj_path


def gen_objects(obj_path, srf: Optional[str]=None) -> Iterator[Tuple[Tuple[Any, Union[int, str, None]], Any]]:
    """
    Finds obj of currently open doc or srf of all that obj_path points to and yields them with obj_path's values
    :param obj_path:
    :param srf: Surfer plot file name
    :return: ((ovrl, ovr_i), data_dict)
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
                if shp.Type==shp_t and shp.Name == shp_n:
                    yield from yield_obj_and_data(shp, ovr_data[k], k)
                    k += 1

    if srf:
        doc.Close(SaveChanges=constants.srfSaveChangesNo)


def objpath_data_paste(obj_path):
    """
    Sets attributes of all objects that obj_path points with obj_path's values
    :param obj_path:
    :return:
    """

    for ((obj, ovr_i), data_dict) in gen_objects(obj_path):
        if 'AspectGridFile' in data_dict:
            obj.SetInputGrids(data_dict['AspectGridFile'], data_dict['GradientGridFile'])
            data_dict = data_dict.copy()
            del data_dict['AspectGridFile']
            del data_dict['GradientGridFile']

        for prop, val in data_dict.items():
            if isinstance(val, dict):
                # prop is a function name, val - contains function arguments
                if '.' in prop:
                    obj_p = obj
                    # function is defined in child object
                    for prop in prop.split('.'):
                        obj_p = getattr(obj_p, prop)
                else:
                    obj_p = getattr(obj, prop)
                obj_p(**val)
                continue
            setattr(obj, prop, val)


def objpath_update_and_get_grids(obj_path, path_dir_out: Path, srf=None):
    """
    Exports grids of obj_path, and modifies obj_path values to point to this grids
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
                    path_grd.unlink()  #
                except Exception as e:
                    l.warning('can not remove "%s"', path_grd)

def paste_srfs_data(path_srfs, re_ovr='*', re_shp=None,
                    path_dir_out:Union[Path, str, None]=None,
                    path_dir_tmp:Union[Path, str, None]=None):
    """
    Replaces all grids in opened patern with exctracted grids from files in path_srfs
    Note: open srf pattern before run this function!
    :param path_srfs: srfs having needed data
    :param re_ovr: regular expression for names of overlays having needed data
    :param re_shp: regular expression for names of shapes having needed data
    :param path_dir_out: where to save srfs maked by replacing data in opened srf
    :param path_dir_tmp: where to save temporary grids
    :return: None
    """

    doc = Surfer.ActiveDocument
    shapes = doc.Shapes

    if path_dir_tmp is None:
        path_dir_tmp = Path(doc.Path)  # 'C:\Windows\TEMP'
    else:
        path_dir_tmp = Path(path_dir_tmp)

    if path_dir_out is None:
        path_dir_out = path_dir_tmp
    else:
        path_dir_out = Path(path_dir_out)

    if Path(path_srfs).is_dir():
        path_srf_parent = Path(path_srfs)
        srf_pattern = '*.srf'
    else:
        path_srf_parent = Path(path_srfs).parent
        srf_pattern = Path(path_srfs).name
    srfs = list(path_srf_parent.glob(srf_pattern))
    print(f'Replacing grids in opened pattern using data from {len(srfs)} {srf_pattern} files\nfrom {path_srf_parent}\n  to {path_dir_out}...')

    obj_path = objpath_fill(re_ovr, re_shp)

    for i, srf in enumerate(srfs):
        print(f'{i}. {srf.stem}')
        Surfer.ScreenUpdating = False
        obj_path = objpath_update_and_get_grids(obj_path, path_dir_tmp, srf)
        objpath_data_paste(obj_path)
        
        path_doc = path_dir_out / Path(srf).name
        try:
            doc.SaveAs(str(path_doc))
        except pywintypes.pywintypes.com_error:
            l.exception(f'Can not save to {path_doc}')
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




def main():
    """
    Executes paste_srfs_data() with my settings, in particular paths on my computer
    :return:
    """
    global Surfer
    if not Surfer:
        gencache.EnsureModule('{54C3F9A2-980B-1068-83F9-0000C02A351C}', 0, 1, 4)
        Surfer = Dispatch("Surfer.Application")
        # interfaces to get specific obj methods by calling Cast(obj, interface):
        obj_interface.update(                   # old interface in comments:
            {constants.srfShapeContourMap: 'IContourLayer',   # IContourMap
             constants.srfShapeVectorMap: 'IVectorMap',
             constants.srfShapeImageMap: 'IColorReliefLayer2', # IImageLayer2
             constants.srfShapeVector2Grid: 'IVectorLayer',
             constants.srfShapeMapFrame: 'IMapFrame2',
             }
            )

    paste_srfs_data(
        path_srfs=r'd:\WorkData\BalticSea\_other_data\_model\POM_GMasha\190801_80,85,90m\_srf',
        re_ovr=r'.*\d_filt_',
        re_shp='\d\dm|PositionText',
        path_dir_out=r'd:\WorkData\BalticSea\_other_data\_model\POM_GMasha\190801_80,85,90m\_srf_bold_isobaths'
        )




if __name__ == '__main__':
    main()