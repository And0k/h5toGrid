# Andrey Korzh, 12.08.2023
#
import logging
from datetime import datetime
from pathlib import Path
from ruamel.yaml import YAML
from typing import List
import numpy as np
from tcm import h5inclinometer_coef, format

yaml = YAML()
yaml.indent(mapping=2, sequence=4, offset=2)


# from numpy.polynomial.polynomial import polyval2d

def poly_str2list(poly_str) -> List[float]:
    """
    Converts a polynomial string like 'A + B * u - C * t - D * u**2 + E * u * t + F * t**2' to a list of
    polynomial coefficients like [A, B, C, D, E, F]
    :param poly_str: The polynomial string containing coefficients and terms represented by letters
    :return: A list of polynomial coefficients parsed from the input string
    """
    str_splitted = poly_str.split(' ')
    coefs = [float(str_splitted[0])]  # A
    v = None
    for s in str_splitted[1:]:
        if s in ('+', '-'):
            v = s
            continue
        elif v is not None:
            coefs.append(float(''.join([v, s])))  # B, C, D, E, F
        v = None
    return coefs


def coef_yaml2hdf5(path_yaml, paths_hdf5, tbl_prefix='incl'):
    """
    Reads polynomial coefficients from YAML and copies them into HDF5 databases.

    Parses termocompensated pressure coefficients via `coef_prepare_from_yaml()`,
    then writes each probe's coefficient matrix and date metadata into the specified
    HDF5 files using `h5copy_coef()`.

    :param path_yaml: Path to the YAML file containing coefficient definitions
    :param paths_hdf5: List of HDF5 file paths to write coefficients into
    :param tbl_prefix: Table name prefix for old-format probe numbering (e.g. ``'incl'`` -> ``incl01``)
    """
    print(f'Copy {path_yaml} coefficients...')
    with path_yaml.open('r') as f:
        data = yaml.load(f)
    arg_dict = coef_prepare_from_yaml(data)

    for db in paths_hdf5:
        print(f'to {db}:')
        for pnum, arg_coef in arg_dict.items():
            try:
                tbl = f'{tbl_prefix}{int(pnum):0>2}'
            except ValueError:
                # new format with probe id instead of just number (ignoring tbl_prefix argument)
                tbl = format.pcid_to_raw_name(format.to_pcid_from_name(pnum))

            print(f'{tbl} ', end='')
            h5inclinometer_coef.h5copy_coef(h5file_dest=db, tbl=tbl, **arg_coef, ok_to_replace_group=True)


def coef_prepare_from_yaml(data):
    """
    Prepare arguments with termocompensated pressure coef info for h5copy_coef()
    Note: to get formula from result coefficient_matrix, eval:
    `sympy.expand(polyval2d(sympy.Symbol('u'), sympy.Symbol('t'), coefficient_matrix))`
    todo: keep only max date data if multiple records for the same `pnum` (last is kept now)
    :param data: yaml loaded data
    :return: Dict mapping probe identifiers (str) to dicts with keys ``dict_matrices`` and ``dates``
    """
    arg_dict = {}

    def construct_dict(date_str, data_coefs, rel_path):
        """
        :param date_str: of '%y%m%d_%H%M' format with any further suffix (or shorter date possibly with ext)
        :param data_coefs: _description_
        :param rel_path: _description_
        :return: _description_
        """
        coefs = poly_str2list(data_coefs['poly'])
        # convert to matrix coefficient ready to use in numpy.polynomial.polynomial.polyval2d()
        coefficient_matrix = np.zeros((3, 3))
        coefficient_matrix.flat[[0, 3, 1, 6, 4, 2]] = coefs
        fmt_date = '%y%m%d_%H%M'
        date_in_file = date_str[:len(fmt_date)].split('. ')[0]  # len works due to all `%`-format parts lens=2
        return {
            "dict_matrices": {rel_path: coefficient_matrix},
            "dates": {rel_path: datetime.strptime(date_in_file, "%y%m%d_%H%M")},
        }

    # relative group path_
    rel_path = '//coef//P_t'

    for i, (data_header, data_coefs) in enumerate(data.items()):
        try:
            date_str, pnum = data_header.split(': ')
        except ValueError:  # new format with separate 2nd file level:
            # file supposedly starts with data date
            pnum = data_header
            for j, (date_str, data_coefs) in enumerate(data_coefs.items()):
                arg_dict[pnum] = construct_dict(date_str, data_coefs, rel_path)
            continue
        else:
            date_str = date_str.split('. ')[0]

        arg_dict[pnum] = construct_dict(date_str, data_coefs, rel_path)
    return arg_dict


def coef_save_to_yaml(data: dict, path_yaml: Path, ry=None):
    """Save per-probe coefficient dict to YAML in ``{input: {coefs: ...}}`` format.

    Converts ``dict_matrices`` (HDF5-style paths) to flat coef keys before
    serialization — matching ``export_coefs()`` output and ``load_coefs()`` input.
    Existing coefs not in *data* are preserved (update semantics).

    :param data: per-probe coef dict from ``coef_prepare_from_yaml()``,
        i.e. ``{"dict_matrices": {h5path: np.array}, "dates": {h5path: datetime}}``
    :param path_yaml: YAML output path — created or updated in-place
    :param ry: optional ``ruamel.yaml.YAML`` instance (created with reference settings if None)
    """
    if ry is None:
        ry = YAML()
        ry.indent(mapping=2, sequence=4, offset=2)
        ry.default_flow_style = False
        ry.allow_unicode = True

    # Convert HDF5 path keys (//coef//P_t) → flat coef keys (P_t),
    # mirroring the reverse of what h5copy_coef / dict_matrices_for_h5 do
    flat_coefs: dict[str, object] = {}
    for h5path, mat in data.get("dict_matrices", {}).items():
        flat_coefs[h5path.rstrip('/').rsplit('//', 1)[-1]] = mat
    flat_coefs['dates'] = {
        h5path.rstrip('/').rsplit('//', 1)[-1]: str(dt)
        for h5path, dt in data.get('dates', {}).items()
    }

    # Merge with existing YAML content
    existing_coefs: dict[str, object] = {}
    if path_yaml.exists():
        with path_yaml.open('r', encoding='utf-8') as f:
            existing = ry.load(f) or {}
        existing_coefs = existing.get('input', {}).get('coefs', {})
    existing_coefs.update(flat_coefs)

    # Serialize numpy → Python types (same as export_coefs uses)
    from scripts.export_coefs_to_yaml import serialize_for_yaml
    coefs_serializable = serialize_for_yaml(existing_coefs)

    # Wrap under 'input.coefs' key to match config structure
    doc = {'input': {'coefs': coefs_serializable}}

    path_yaml.parent.mkdir(parents=True, exist_ok=True)
    with path_yaml.open('w', encoding='utf-8') as f:
        ry.dump(doc, f)

    print(f"Exported coefficients file(s) to {path_yaml.name}")


if __name__ == '__main__':
    path_yaml = Path(
        r"B:\WorkData\experiment\inclinometer\260624@ip05-Press\_raw\vsz\coefs_260706.yaml"
        # r"D:\WorkData\_experiment\P~tc\250827@ip2\_raw\stand\vsz(txt, range=12h)\coefs_250929.yaml"
        # r'd:\WorkData\_experiment\inclinometer\230428_stand,tank,pres@ip1-6\_raw\coefs_230808.yaml'
    )
    paths_hdf5 = [Path(p) for p in (
        # r"D:\Cruises\BlackSea\250909_Katsiveli@i\_raw\coefs.h5"
        # r'd:\WorkData\_experiment\inclinometer\230428_stand,tank,pres@ip1-6\_raw\230428P.raw.h5'
        # r'd:\WorkData\_experiment\inclinometer\230428_stand,tank,pres@ip1-6\_raw\230428stand.raw.h5',

    )]
    paths_yaml_out = [
        Path(p)
        for p in (
            # r"D:\Cruises\BlackSea\250909_Katsiveli@i\_raw\coefs.yaml"
        )
    ] or [path_yaml.with_name(f"@{path_yaml.stem}.yaml")]

    if paths_hdf5:
        coef_yaml2hdf5(path_yaml, paths_hdf5, tbl_prefix="incl_p")
    if paths_yaml_out:
        print(f'Copy {path_yaml} coefficients...')
        with path_yaml.open('r') as f:
            data = yaml.load(f)
        arg_dict = coef_prepare_from_yaml(data)
        for pnum, coef_data in arg_dict.items():
            for dir_out in paths_yaml_out:
                # If dir_out is a file path, use its parent dir; if a dir, use as-is
                out_dir = dir_out if dir_out.is_dir() else dir_out.parent
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{pnum}.yaml"
                coef_save_to_yaml(coef_data, path_yaml=out_path, ry=None)
