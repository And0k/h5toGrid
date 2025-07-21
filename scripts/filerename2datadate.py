"""
Renames files with specified extensions to format that can include 1st date extracted from specified text files
Run example:
python filerename2datadate.py d:\\WorkData\\BalticSea\\230507_ABP53\\CTD_SST_CTD90\\_raw\\*.tob {path}|*.SRD|*.vsz

Andrey Korzh, 23.05.2023
"""

import argparse
from pathlib import Path
from re import match
from dateutil.parser import parse
from typing import Sequence


def rename_files(
    dates,
    masks: str|Sequence[str],
    out_name: Path,
):
    """
    :param dates: {path: date} mapping of files from which date was extracted, which `name` and `path` can be
    used in `rename_files_mask`
    :param masks: mask for files to be renamed, can include {path} variable
    :param out_name: Output file name format (without extension) with optional {date} and {name} variables

    """
    if isinstance(masks, str):
        masks = masks.split("|")
    if not masks:
        print('Nothing to do: "masks" argument is empty!')
        return

    for i1, (file_in, date) in enumerate(dates.items(), start=1):
        done_for_suffixes = []
        i_add = 1
        out_stem = out_stem0 = out_name.name.format(date=date, name=file_in.name)
        # (`out_stem0` saves original `out_stem` to easy update it in cycle if it will be required)
        for mask in masks:
            file = file_in if mask == "{path}" else file_in.with_suffix(mask.lstrip("*"))
            suffix = file.suffix
            try:
                while True:
                    try:
                        file.rename((out_name.parent / f"{out_stem}").with_suffix(suffix))
                        break
                    except FileExistsError:
                        out_stem = f"{out_stem0}~{i_add}"
                        i_add += 1
                        continue
            except FileNotFoundError:
                continue
            done_for_suffixes.append(suffix)
        print(
            i1,
            mask.stem,
            "=>",
            (f"{out_stem}{suffix}" if len(done_for_suffixes) == 1 else f"{out_stem} ({done_for_suffixes!r})"),
        )

def files_data_start_date(
    path: Path, date_regex: str, max_rows: int = 100
):
    """
    Extract first data date from text files.

    :param path: Path to the directory containing files to be renamed
    :param date_regex: Regex pattern to find date in the files
    :param max_rows: Skip file if date is not found in the first rows
    :return: dates
    """

    dates = {}
    for i1, file in enumerate(path.parent.glob(path.name), start=1):
        # Loading data
        with file.open('r', encoding='utf-8', errors='replace') as fdata:  # nameFull.open('rb')
            for il, line in enumerate(range(max_rows)):
                line = fdata.readline()
                m = match(date_regex, line)
                if (m is not None) and (m.group(1) is not None):
                    date_raw = ' '.join(
                        v for k, v in sorted(m.groupdict().items(), key=lambda key_val: key_val[0]))
                    dates[file] = parse(date_raw, dayfirst=True, yearfirst=True)
                    break
                elif not line:
                    print('File not renamed:', file, '- Can not find date data.', 'Skipped!')
                    break
            else:
                print(i1, f'No data found in first {max_rows} rows of {file}.', 'File(s) not renamed!')

    if i1:
        print("Files found:", i1)
    if not dates:
        if i1:
            print("Files found:", i1)
            raise ValueError(f"No dates found in {i1} found files")
        else:
            raise FileNotFoundError(f"No files found matched {path}")
    return dates


def filerename2datadate(
        path: Path,
        rename_files_mask: str,
        out_name: Path,
        date_regex: str,
        max_rows: int = 100
    ):
    """
    Rename files with specified extensions to a format that includes the first date extracted from specified text files.

    :param path: Path to the directory containing files to be renamed
    :param rename_files_mask: mask for files to be renamed, can include {path} variable
    :param out_name: Output file name format (without extension) with optional {date} and {name} variables
    :param date_regex: Regex pattern to find date in the files
    :param max_rows: Skip file if date is not found in the first rows
    :return: None
    """
    dates = files_data_start_date(path, date_regex, max_rows)
    rename_files(dates, rename_files_mask, out_name)


# #######################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'path', default="*.t??", type=Path, nargs='?',
        help='directory with files mask to read data (and rename if rename_files_mask not specified other names)'
    )
    parser.add_argument(
        'rename_files_mask', default='{path}', type=str, nargs='?',
        help='files mask that will be renamed with optional {path} variable for files defined by 1st argument'
    )
    parser.add_argument(
        'out_name', default='{date:%y%m%d_%H%M}', type=Path, nargs='?',
        help='output files name format (without extension) with optional {date} and {name} variables'
    )
    parser.add_argument(
        'date_regex', default=r'(?:\t|[ ]+)?(?:[^-\t /\\]+(?:\t|[ ]+))*'  # r'(?:(?:[^-./\t ]*(?:\t|[ ]+)))*' hangs!
        r'(?P<d>\d{1,4}[-./\\]\d{1,2}[-./\\]\d{1,4})([ ]*|\t)(?P<t>\d{1,2}:\d{1,2}[^ \t]*)',
        type=str, nargs='?',
        help='python regex to find date. Date will be extracted from named groups (i.e. '
             'started with ?P<name>) sorted by their name and concatenated through space using dateutil.parser'
    )
    parser.add_argument(
        'name_fun', default=r'',
        type=str, nargs='?',
        help=(
            'todo: python expression to evaluate returning value that can be included in out_name with {name} '
            'placeholder. Arguments: '
            '- i: int, input file number, '
            '- date: datetime, extracted date from text data'
        )
    )
    args = parser.parse_args()

    # If paths are not absolute then set parent dir same as of one of previous argument
    # or else 1st argument with absolute path
    args_dict = args.__dict__.copy()
    parent = None
    arg_need_parent = []
    parent_1st = None
    for k, p in args.__dict__.items():
        if not isinstance(p, Path):
            continue
        if p.is_absolute():
            parent = p.parent
        else:
            if parent:
                if not parent_1st:
                    parent_1st = parent
                args_dict[k] = parent / p
            else:
                arg_need_parent.append(k)
    # If no absolute paths found then use current dir
    if not parent_1st:
        parent_1st = Path.cwd()
    for k in arg_need_parent:
        args_dict[k] = parent_1st / args_dict[k]

    filerename2datadate(**args_dict)
