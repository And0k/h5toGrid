# @+leo-ver=5-thin
# @+node:korzh.20180529212530.11: * @file h5inclinometer_coef.py
# !/usr/bin/env python
# coding:utf-8
# @+others
# @+node:korzh.20180525044634.2: ** Declarations
"""
Save/modify coef in hdf5 data in "/coef" table of PyTables (pandas hdf5) store
"""
import logging
from typing import Iterable, Mapping, Union
from pathlib import Path
import h5py
import numpy as np

# my
from utils2init import FakeContextIfOpen, standard_error_info

if __name__ != '__main__':
    l = logging.getLogger(__name__)


def rot_matrix_x(c, s):
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]], np.float64)


def rot_matrix_y(c, s):
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]], np.float64)


def rot_matrix_z(c, s):
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]], np.float64)


def rotate_x(a2d, angle_degrees=None, angle_rad=None):
    """

    :param a2d:
    :param angle_degrees: roll
    :param angle_rad: roll
    :return:
    """
    if angle_rad is None:
        angle_rad = np.radians(angle_degrees)
    out2d = np.dot(rot_matrix_x(np.cos(angle_rad), np.sin(angle_rad)), a2d)
    return out2d


def rotate_y(a2d, angle_degrees=None, angle_rad=None):
    """

    :param a2d:
    :param angle_degrees: pitch because of x direction to Left
    :param angle_rad: pitch
    :return:
    """
    if angle_rad is None:
        angle_rad = np.radians(angle_degrees)
    out2d = rot_matrix_y(np.cos(angle_rad), np.sin(angle_rad)) @ a2d
    return out2d


def rotate_z(a2d, angle_degrees=None, angle_rad=None):
    """

    :param a2d:
    :param angle_degrees: yaw
    :param angle_rad: yaw
    :return:
    """
    if angle_rad is None:
        angle_rad = np.radians(angle_degrees)
    out2d = np.dot(rot_matrix_z(np.cos(angle_rad), np.sin(angle_rad)), a2d)
    return out2d


def h5savecoef(h5file_dest, path, coef):
    """

    :param coef:
    :return:

    Example:
    h5_savecoef(h5file_dest, path='//incl01//coef//Vabs', coef)
    """
    if np.any(~np.isfinite(coef)):
        l.error('NaNs in coef detected! Aborting')
    else:
        with h5py.File(h5file_dest, 'a') as h5dest:
            # or if you want to replace the dataset with some other dataset of different shape:
            # del f1['meas/frame1/data']
            try:
                h5dest.create_dataset(path, data=coef, dtype=np.float64)
                return
            except (OSError, RuntimeError) as e:
                try:
                    print(f'updating {h5file_dest}/{path}')  # .keys()
                    h5dest[path][...] = coef
                    h5dest.flush()
                    return
                except Exception as e:
                    pass  # prints error message?
                l.exception('Can not save/update coef to hdf5 %s. There are error ', h5file_dest)


# @+node:korzh.20180525125303.1: ** h5copy_coef
def h5copy_coef(h5file_source=None, h5file_dest=None, tbl=None, tbl_source=None, tbl_dest=None,
                dict_matrices: Union[Mapping[str, np.ndarray], Iterable[str], None] = None, ok_to_replace_group=False):
    """
    Copy tbl from h5file_source to h5file_dest overwriting tbl + '/coef/H/A and '/coef/H/C' with H and C if provided
    :param h5file_source: name of any hdf5 file with existed coef to copy structure
    :param h5file_dest: name of hdf5 file to paste structure
    :param dict_matrices: dict of numpy arrays - to write or list of paths to coefs (to matrices) under tbl - to copy them
    # Example save H and C: 3x3 and 1x3, rotation and shift matrices
    >>> h5copy_coef(h5file_source,h5file_dest,tbl)
            dict_matrices={'//coef//H//A': H,
                           '//coef//H//C': C})
    """

    if h5file_dest is None:
        h5file_dest = h5file_source
    if h5file_source is None:
        if h5file_dest is None:
            print('skipping: output not specified')
            return
        h5file_source = h5file_dest

    if tbl_source is None:
        tbl_source = tbl
    if tbl_dest is None:
        tbl_dest = tbl

    # class File_context:
    #     """
    #     If input is string filename then acts like usual open context manager
    #     else treat input as opened file object and do nothing
    #     """
    #
    #     def __init__(self, h5file_init):
    #         self.h5file_init = h5file_init
    #
    #     def __enter__(self):
    #         if isinstance(self.h5file_init, str):
    #             self.h5file = h5py.File(self.h5file_init, 'a')
    #             return self.h5file
    #         else:
    #             self.h5file = self.h5file_init
    #
    #     def __exit__(self, exc_type, ex_value, ex_traceback):
    #         if exc_type is None and isinstance(self.h5file_init, str):
    #             self.h5file.close()
    #         return False

    def path_h5(file):
        return Path(file.filename if isinstance(file, h5py._hl.files.File) else file)

    def save_operation(h5source=None):
        """
        update dict_matrices in h5file_dest. h5source may be used to copy from h5source
        :param h5source: opened h5py.File, if not None copy h5file_source//tbl_source//coef to h5file_dest//tbl//coef before update
        uses global:
            h5file_dest
            tbl_dest, tbl_source
            dict_matrices
        """
        nonlocal dict_matrices

        with FakeContextIfOpen(lambda f: h5py.File(f, 'a'), h5file_dest) as h5dest:
            try:
                if (h5source is None):
                    if (tbl_dest != tbl_source):
                        h5source = h5dest
                    else:
                        raise FileExistsError(f'Can not copy to itself {h5dest.filename}//{tbl_dest}')
                elif (path_h5(h5dest) == h5source and tbl_dest == tbl_source):
                    raise FileExistsError(f'Can not copy to itself {h5dest.filename}//{tbl_dest}')

                # Copy using provided paths:
                if h5source:
                    path_coef = f'//{tbl_source}//coef'
                    l.info(f'copying "coef" from {path_h5(h5source)}//{tbl_source} to {h5dest.filename}//{tbl_dest}')
                    # Reuse previous calibration structure:
                    # import pdb; pdb.set_trace()
                    # h5source.copy('//' + tbl_source + '//coef', h5dest[tbl_dest + '//coef'])
                    try:
                        h5source.copy(path_coef, h5dest[tbl_dest])
                        # h5source[tbl_source].copy('', h5dest[tbl_dest], name='coef')
                    except RuntimeError as e:  # Unable to copy object (destination object already exists)
                        replace_coefs_group_on_error(h5source, h5dest, path_coef, e)
                    except KeyError: # Unable to open object (object 'incl_b11' doesn't exist)"
                        l.warning('Creating "%s"', tbl_source)

                        try:
                            h5dest.create_group(tbl_source)
                        except (ValueError, KeyError) as e:  # already exists
                            replace_coefs_group_on_error(h5source, h5dest, tbl_source, e)
                        else:
                            h5source.copy(path_coef, h5dest[tbl_dest])

            except FileExistsError:
                if dict_matrices is None:
                    raise

            if dict_matrices:  # not is None:
                have_values = isinstance(dict_matrices, dict)
                l.info(f'updating {h5file_dest}/{tbl_dest}/{dict_matrices}')  # .keys()

                if have_values:  # Save provided values:
                    for k in dict_matrices.keys():
                        path = f'{tbl_dest}{k}'
                        data = dict_matrices[k]
                        if isinstance(dict_matrices[k], (int, float)):
                            data = np.atleast_1d(data)  # Veusz can't load 0d single values
                        try:
                            b_isnan = np.isnan(data)
                            if np.any(b_isnan):
                                l.warning('not writing NaNs: %s%s...', k, np.flatnonzero(b_isnan))
                                h5dest[path][~b_isnan] = data[~b_isnan]
                            else:
                                h5dest[path][...] = data
                        except TypeError as e:
                            l.error('Replacing dataset "%s" TypeError: %s -> recreating...', path,
                                    '\n==> '.join([a for a in e.args if isinstance(a, str)]))
                            # or if you want to replace the dataset with some other dataset of different shape:
                            del h5dest[path]
                            h5dest.create_dataset(path, data=data, dtype=np.float64)
                        except KeyError as e:  # Unable to open object (component not found)
                            l.warning('Creating "%s"', path)
                            h5dest.create_dataset(path, data=data, dtype=np.float64)
                else:
                    paths = list(dict_matrices)
                    dict_matrices = {}
                    for rel_path in paths:
                        path = tbl_source + rel_path
                        try:
                            dict_matrices[path] = h5source[path][...]
                        except AttributeError:  # 'ellipsis' object has no attribute 'encode'
                            l.error(
                                'Skip update coef: dict_matrices must be None or its items must point to matrices %s',
                                '\n==> '.join(a for a in e.args if isinstance(a, str)))
                            continue
                        h5dest[path][...] = dict_matrices[path]

                h5dest.flush()
            else:
                dict_matrices = {}

            # or if you want to replace the dataset with some other dataset of different shape:
            # del f1['meas/frame1/data']
            # h5dest.create_dataset(tbl_dest + '//coef_cal//H//A', data= A  , dtype=np.float64)
            # h5dest.create_dataset(tbl_dest + '//coef_cal//H//C', data= C, dtype=np.float64)
            # h5dest[tbl_dest + '//coef//H//C'][:] = C

    def replace_coefs_group_on_error(h5source, h5dest, path, e=None):
        if ok_to_replace_group:
            l.warning(f'Replacing group "%s"', path)
            del h5dest[path]
            h5source.copy(path, h5dest[tbl_dest])
        else:
            l.error('Skip copy coef' + (f': {standard_error_info(e)}!' if e else '!'))

    # try:
    with FakeContextIfOpen(
            (lambda f: h5py.File(f, 'r')) if h5file_source != h5file_dest else None,
            h5file_source) as h5source:
        save_operation(h5source)

    # if h5file_source != h5file_dest:
    #     with h5py.File(h5file_source, 'r') as h5source:
    #         save_operation(h5source)
    # else:
    #     save_operation()
    # except Exception as e:
    #     raise e.__class__('Error in save_operation()')

    # Confirm the changes were properly made and saved:
    b_ok = True
    with FakeContextIfOpen(lambda f: h5py.File(f, 'r'), h5file_dest) as h5dest:
        for k, v in dict_matrices.items():
            if not np.allclose(h5dest[tbl_dest + k][...], v, equal_nan=True):
                l.error(f'h5copy_coef(): coef. {tbl_dest + k} not updated!')
                b_ok = False
    if b_ok and dict_matrices:
        print('h5copy_coef() have updated coef. Ok>')


def h5_rotate_coef(h5file_source, h5file_dest, tbl):
    """
    Copy tbl from h5file_source to h5file_dest overwriting tbl + '/coef/H/A with
    previous accelerometer coef rotated on Pi for new set up
    :param h5file_source: name of any hdf5 file with existed coef to copy structure
    :param h5file_dest: name of hdf5 file to paste structure
    """
    with h5py.File(h5file_source) as h5source:
        with h5py.File(h5file_dest, "a") as h5dest:
            # Reuse previous calibration structure:
            in2d = h5source[tbl + '//coef//G//A'][...]
            out2d = rotate_x(in2d, 180)
            h5dest[tbl + '//coef//G//A'][:, :] = out2d
            h5dest.flush()

    print('h5copy_coef(): coef. updated')


