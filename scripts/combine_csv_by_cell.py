# Andrey Korzh, 22.11.2023
"""
Combine consequent text WinRiverII files having data of same cell size, name result files by its data starting time.
Purpose: decrease number of files keeping order and data of different properties in separate files.
"""

import sys
from pathlib import Path, PurePath
from datetime import datetime
import io
from contextlib import nullcontext
from itertools import dropwhile
from typing import Callable, BinaryIO, Mapping, Optional, Sequence, TextIO, Tuple, Union
from re import subn

# my funcs
from utils2init import LoggingStyleAdapter
from to_vaex_hdf5.h5tocsv import ctd2csv

sys.path.append(r'C:\Work\Python\AB_SIO_RAS\tcm')
from tcm.csv_load import load_from_csv_gen
from tcm.csv_specific_proc import loaded_rock, century

lf = LoggingStyleAdapter(__name__)

# Configuration
n_acols = 100  # number of output a-cols (equal to output number of d-cols)
sep = ','


def align_block_rows(block, n_acols=100, sep=sep, header:str = None):
    
    """
    Get all values after "dz_m" in each row: these are a-cols followed same number of d-cols:
    1	2	3	4	5	6	7	8	    9	   10	11	    12	    13 a2 ... ak d1 d2 ... dk
    Y	M	D	h	m	s	ms	Dist_m	Bot_m	N	z1_m	dz_m	a1 a2 ... ak d1 d2 ... dk
    23	11	21	19	58	51	9	0	219.8308434	80	12.39	4	0.25
    
    Ignore rows where all a-cols is bad (equal to -32768)
    
    Determine k and if file has less values than previous then append empty vals to a and d-cols
    :param block:
    :param n_acols: number of output a-cols (equal to output number of d-cols)
    :param sep:
    :return:
    Note: Format has ``N`` value which we could use instead of finding ``n_acols_cur``
    """

    n_before_acols = 12  # 0-based index
    out_rows = [f'{header}\n'] if header else []
    n_acols_cur = None
    no_data_rows = 0
    max_acols_good = 0
    for row in block.splitlines():
        vals = row.split(',')
        if n_acols_cur is None:
            n_acols_cur = (len(vals) - n_before_acols) // 2
            n_add_acols = n_acols - n_acols_cur
            if n_acols_cur > n_acols:
                lf.warning(f'File hase more a-cols ({n_acols_cur}) than needed cols number {n_acols}. Increasing former value')
        
        # Get max number of good a-cols
        try:
            n_bad_at_end = next(dropwhile(
                lambda v: v[1] == '-32768',
                enumerate(vals[(n_before_acols + n_acols_cur - 1):(n_before_acols - 1):-1])
            ))[0]
        except StopIteration:  # => n_acols_good will be 0
            # not any(v for v in vals[n_before_acols:(n_before_acols + n_acols_cur)] if v != '-32768'):
            no_data_rows += 1
            continue
        n_acols_good = n_acols_cur - n_bad_at_end
        max_acols_good = max(max_acols_good, n_acols_good)
        
        # Get aligned rows
        out_row = sep.join(
            vals[:(n_before_acols + n_acols_cur)] + [''] * n_add_acols +
            vals[(n_before_acols + n_acols_cur):] + [''] * n_add_acols + ['\n']
        )
        out_rows.append(out_row)
    if no_data_rows:
        print(f'junk_rows:{no_data_rows}', end=', ')
    return out_rows, max_acols_good
    
    
def fun_combine_closure():

    def get_group(vals):
        return float(vals[11])  # cell size (element dz you can see in header)
    
    file_out_group_prev = None
    
    def file_name_from_data(text_1st_block, ext='tsv' if sep == '\t' else 'csv'):
        """
        Also returns header
        :param text_1st_block:
        :param ext:
        :return:
        """
        nonlocal file_out_group_prev
        vals = text_1st_block[:100].split(',')
        out_group = get_group(vals)
        if file_out_group_prev == out_group:
            return out_group, None, None  # not need calculate same out file params again
        file_out_group_prev = out_group
        time_vals = {k: int(v) for k, v in zip(['year', 'month', 'day', 'hour', 'minute'], vals[:5])}
        time_vals['year'] += int(century)*100
        t_st = datetime(**time_vals)
        out_name = f'{t_st:%y%m%d_%H%M}_cell={out_group:g}.{ext}'
        
        z0_1 = float(vals[10]) + 5.8  # depth of 1st cell
        out_header = (
            'Y M D h m s ms Dist_m Bot_m N z1_m dz_m'.split() +
            [f'{l}{round(z0_1 + k*out_group)}' for l in 'ad' for k in range(n_acols)]
        )
        return out_group, out_name, sep.join(out_header)
        
    file_outs_cur = {}
    
    def fun_combine(
            file_in: Union[str, Path, BinaryIO, TextIO],
            file_out: Optional[Path] = None,
            dir_out: Optional[PurePath] = None,
            get_file_params: Callable[[str, str], PurePath] = file_name_from_data,
            sub_str_list: Sequence[bytes] = None,
            **kwargs
    ) -> Tuple[Path, Mapping]:
        """
        
        :param file_in:
        :param file_out:
        :param dir_out:
        :param get_file_params:
        :param sub_str_list:
        :param kwargs:
        :return:
        """
        block_size = 1000000
        f_in_was_opened = isinstance(file_in, (io.TextIOBase, io.RawIOBase))
        with nullcontext if f_in_was_opened else open(file_in, 'r') as f_in:
            if f_in_was_opened:
                f_in, file_in = file_in, Path(file_in.name)
                binary_mode = isinstance(f_in, io.RawIOBase)
            else:
                binary_mode = False

            the_end = b'' if binary_mode else ''
            
            f_out = None
            if not dir_out:
                dir_out = file_in.parent
                
            for i, block in enumerate(iter(lambda: f_in.read(block_size), the_end)):
                if block == the_end:
                    break
                if f_out is None:  # 1st block
                    out_group, out_name, out_header = get_file_params(block)
                    # adding to
                    if out_name or (out_group not in file_outs_cur):
                        # - new out file
                        try:
                            file_outs_cur[out_group].close()
                        except KeyError:  # ok, new group will be created:
                            pass
                        f_out = file_outs_cur[out_group] = open(dir_out / out_name, 'w', newline=None)
                        print(').\t' if file_outs_cur else '\t', out_name, end=f'(\n{file_in.stem}: ')
                    else:
                        # - out file opened earlier
                        print(end=f'{file_in.stem}: ')
                        f_out = file_outs_cur[out_group]
                        # out_name = Path(file_outs_cur[out_group].name) if not want output None
                
                lines, n_acols_with_data = align_block_rows(block, n_acols=n_acols, header=out_header)
                for line in lines:
                    f_out.write(line)
                print(f'cells:{n_acols_with_data}, rows:{len(lines)}, ')  # required number of columns for a-colls
                # f_out.write(block)

        return out_name, out_group
    
    def fun_close():
        for f_out in file_outs_cur.values():
            f_out.close()
            
    return fun_combine, fun_close


def combine_csv_by_cell(cfg_in, cfg_out):
    """
    Load ROCK CTD text data of table format: "-0.0013 3.8218 0.1531 2023-11-21 11:53:22".
    Adds Sal, SigmaTh0.
    Saves to tab separated values files named by start and end data time.
    :return:
    """
    fun_combine, fun_close = fun_combine_closure()
    
    path = cfg_in['path']
    files = list(path.parent.glob(path.name))
    n_files = len(files)
    print(f'Loading {n_files} {path} files...')
    grouped = {}
    for i, file in enumerate(files):
        out_name, group = fun_combine(file)
        if group in grouped:
            grouped[group].append(file)
        else:
            grouped[group] = [file]
        
    fun_close()
    # --------------
    
    paths_csv_prev = None
    for itbl, pid, paths_csv, df_raw in load_from_csv_gen(cfg_in):
        if paths_csv_prev != paths_csv:
            paths_csv_prev = paths_csv
            csv_part = 0
        else:
            csv_part += 1  # next part of same csv
        


if __name__ == '__main__':
    path_cruise = Path(r'd:\WorkData\KaraSea\231110_AMK93')
    device = 'ADCP_75kHz'
    cfg_in = {
        'path':               path_cruise / device / r'_raw\LTA\AMK93[0-9][0-9]*_BTabs,dir_ASC.TXT',
        'fun_proc_loaded':    loaded_rock,
        'csv_specific_param': {
            # 'Pres_fun': lambda x: np.polyval([100, 0], x),
            # 'Sal_fun': lambda Cond, Temp90, Pres: gsw.SP_from_C(Cond, Temp90, Pres),  # not adds col!
            # 'SigmaTh_fun': lambda Sal, Temp90, Pres: sigma_th0(Sal, Temp90, Pres)     # not adds col!
        },
        'header': 'yy(text),mm(text),dd(text),HH(text),MM(text),SS(text),ms,Dist_m,Bot_m,N,z1_m,dz_m,a,d',
        'dtype': '|S2 |S2 |S2 |S2 |S2 |S2 f8 f8'.split(),
        # 'Y M D h m s ms Dist_m Bot_m N z1_m dz_m(text) Time(text)',
        'delimiter': ',',
        'nrows': 1
    }
    
    cfg_out_orig = {
        'cols': 'Y M D h m s ms Dist_m Bot_m N z1_m dz_m'.split(),
        'text_path':         path_cruise / device,
        'file_name_fun':     (lambda i_log, t_st, t_en, tbl: f'{t_st:%y%m%d_%H%M}-{t_en:%H%M}.tsv'),
        'text_date_format':  "%Y-%m-%d %H:%M:%S",
        'text_float_format': "%.4f",
        'sep':               '\t'
    }
    
    cfg_out = {
        'cols': 'Y M D h m s ms Dist_m Bot_m N z1_m dz_m '.split(),
        'text_path':         path_cruise / device,
        'file_name_fun':     (lambda i_log, t_st, t_en, tbl: f'{t_st:%y%m%d_%H%M}-{t_en:%H%M}.tsv'),
        'text_date_format':  "%Y-%m-%d %H:%M:%S",
        'text_float_format': "%.4f",
        'sep':               '\t'
    }


    combine_csv_by_cell(cfg_in, cfg_out)
