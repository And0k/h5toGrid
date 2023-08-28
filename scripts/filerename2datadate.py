"""
Renames files with specified extensions to format that can include 1st date extracted from specified text files
Run example:
python filerename2datadate.py d:\WorkData\BalticSea\230507_ABP53\CTD_SST_CTD90\_raw\*.tob {path}|*.SRD|*.vsz

Andrey Korzh, 23.05.2023
"""

import argparse
from pathlib import Path
from re import match
from dateutil.parser import parse


def filerename2datadate(path: Path, rename_files_mask: str, out_name: Path, date_regex: str, max_rows: int = 100,
                        name_fun=None):
    """
    
    :param path:
    :param rename_files_mask:
    :param out_name:
    :param date_regex:
    :param max_rows: skip file if we can not find date in this first rows
    :param name_fun: not implemented
    :return:
    """
    rename_for = rename_files_mask.split('|')
    if not rename_for:
        print('Nothing to do: "rename_files_mask" argument is empty!')
        return
    for i1, file in enumerate(path.parent.glob(path.name), start=1):
        # Loading data
        with file.open('r', encoding='utf-8', errors='replace') as fdata:  # nameFull.open('rb')
            date = None
            for il, line in enumerate(range(max_rows)):
                line = fdata.readline()
                m = match(date_regex, line)
                if (m is not None) and (m.group(1) is not None):
                    date_raw = ' '.join(v for k, v in sorted(m.groupdict().items(), key=lambda key_val: key_val[0]))
                    date = parse(date_raw, dayfirst=True, yearfirst=True)
                    break
                elif not line:
                    print('File not renamed:', file, '- Can not find date data.', 'Skipped!')
                    break
            else:
                print(i1, f'No data found in first {max_rows} rows of {file}.', 'File(s) not renamed!')
        if not date:
            continue
        
        stem = out_name.name.format(date=date)
        done_for_suffixes = []

        i_add = 1
        name_stem = stem
        for rename_f in rename_for:
            rename_f = file if rename_f == '{path}' else file.with_suffix(rename_f.lstrip('*'))
            suffix = rename_f.suffix
            try:
                while True:
                    try:
                        rename_f.rename((out_name.parent / f'{name_stem}').with_suffix(suffix))
                        break
                    except FileExistsError:
                        name_stem = f'{stem}~{i_add}'
                        i_add += 1
                        continue
            except FileNotFoundError:
                continue
            done_for_suffixes.append(suffix)

        print(
            i1, rename_f.stem, '↦', (f'{name_stem}{suffix}' if len(done_for_suffixes) == 1 else
                                     f'{name_stem} ({done_for_suffixes!r})')
        )


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
