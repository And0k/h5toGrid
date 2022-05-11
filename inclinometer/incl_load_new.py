"""
Runs steps:
1. csv2h5(): save inclinometer data from Kondrashov format to DB
2. Calculate velocity and average
3. Calculate spectrograms.
4. veuszPropagate(): draw using Veusz pattern

Allows to specify some of args:
    - control execution parameters:
        :start: float, start step. Begins from step >= start
        Raw file names should be matched by regex: "[\d_]*(W|INKL)_\d\d*.txt"

    - data processing parameters:

        :path_cruise:
        :dir_incl: or :file_in:
        :path_db:
        :probes:
        ...

"""
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from functools import partial
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Tuple, Dict, List
import numpy as np
import pandas as pd
import hydra
from omegaconf import MISSING

# import my scripts
import to_vaex_hdf5.cfg_dataclasses
from to_vaex_hdf5.cfg_dataclasses import hydra_cfg_store, ConfigInHdf5_Simple, ConfigProgram, main_init, main_init_input_file

from to_pandas_hdf5.csv2h5 import main as csv2h5
from to_pandas_hdf5.csv_specific_proc import mod_incl_name, rep_in_file, correct_txt #correct_kondrashov_txt, correct_baranov_txt
from to_pandas_hdf5.h5_dask_pandas import h5q_interval2coord
from inclinometer.h5inclinometer_coef import h5copy_coef
from inclinometer.incl_calibr import dict_matrices_for_h5

import inclinometer.incl_h5clc as incl_h5clc
import inclinometer.incl_h5spectrum as incl_h5spectrum
import veuszPropagate
from utils_time import intervals_from_period # pd_period_to_timedelta
from utils2init import path_on_drive_d, init_logging, open_csv_or_archive_of_them, st, cfg_from_args, my_argparser_common_part
from utils2init import this_prog_basename, standard_error_info, LoggingStyleAdapter
from magneticDec import mag_dec


lf = LoggingStyleAdapter(__name__)
version = '0.0.1'
# @dataclass hydra_conf(hydra.conf.HydraConf):
#     run: field(default_factory=lambda: defaults)dir
hydra.output_subdir = 'cfg'
# hydra.conf.HydraConf.output_subdir = 'cfg'
# hydra.conf.HydraConf.run.dir = './outputs/${now:%Y-%m-%d}_${now:%H-%M-%S}'


@dataclass
class ConfigFilter:
    """
    "filter": excludes some data:

    no_works_noise: is_works() noise argument for each channel: excludes data if too small changes
    blocks: List[int] ='21, 7' despike() argument
    offsets: List[float] despike() argument
    std_smooth_sigma: float = 4, help='despike() argument
    """
    #Optional[Dict[str, float]] = field(default_factory= dict) leads to .ConfigAttributeError/ConfigKeyError: Key 'Sal' is not in struct
    min_date: Optional[Dict[str, str]] = field(default_factory=dict)
    max_date: Optional[Dict[str, str]] = field(default_factory=dict)
    no_works_noise: Dict[str, float] = field(default_factory=lambda: {'M': 10, 'A': 100})
    blocks: List[int] = field(default_factory=lambda: [21, 7])
    offsets: List[float] = field(default_factory=lambda: [1.5, 2])
    std_smooth_sigma: float = 4


@dataclass
class ConfigProcess:
    zeroing_pitch: float = 0  # degrees


@dataclass
class ConfigOut:
    """
    "out": all about output files:

    db_path: hdf5 store file path where to write resulting coef. Writes to tables that names configured for input data (cfg[in].tables) in this file
    """
    db_path: str = ''
    aggregate_period_s: List[Any] = field(default_factory=lambda: [None, 2, 600, 7200])


@dataclass
class ConfigInHdf5InclCalibr(ConfigInHdf5_Simple):
    """
    Same as ConfigInHdf5_Simple + specific (inclinometr calibration) data properties:
    channels: List: (, channel can be "magnetometer" or "M" for magnetometer and any else for accelerometer',
    chunksize: limit loading data in memory (default='50000')
    time_range_list: time range to use
    time_range_dict: time range to use for each inclinometer number (consisted of digits in table name)')
    time_range_nord_list: time range to zeroing north. Not zeroing Nord if not used')
    time_range_nord_dict: time range to zeroing north for each inclinometer number (consisted of digits in table name)')

    """
    path_cruise: Optional[str] = None  # not using MISSING because user can specify ConfigOut.path_db from which we can inference
    raw_subdir: Optional[str] = None
    probes: List[str] = field(default_factory=list)
    probes_prefix: Optional[str] = None
    raw_pattern: str = "*{prefix:}{number:0>2}*.[tT][xX][tT]"

    channels: List[str] = field(default_factory=lambda: ['M', 'A'])
    chunksize: int = 50000
    time_range: List[str] = field(default_factory=list)
    time_range_dict: Dict[str, str] = field(default_factory=dict)  # Dict[int, str] not supported in omegaconf
    time_range_nord: List[str] = field(default_factory=list)
    time_range_nord_dict: Dict[str, str] = field(default_factory=dict)  # Dict[int, str] not supported in omegaconf



cs_store_name = Path(__file__).stem
cs, ConfigType = hydra_cfg_store(
    cs_store_name,
    {
        'input': ['in_hdf5__incl_calibr'],  # Load the config ConfigInHdf5_InclCalibr from the config group "input"
        'out': ['out'],  # Set as MISSING to require the user to specify a value on the command line.
        'process': ['process'],
        'program': ['program'],
    },
    module=sys.modules[__name__]
    )


def device_in_out_paths(
        db_path: Optional[Path],
        path_cruise: Optional[Path],
        device_short_name,
        device_dir_pattern=None
        ):
    """
    Determines device_path and db_path. Required input args are `path_cruise` or `db_path` and device_short_name
    :param db_path:
    :param path_cruise:
    :param device_short_name:
    :param device_dir_pattern: pattern to find device data under cruise dir (use 1st matched)
    :return: device_path, db_path
    """

    # set name by 'path_cruise' name or parent if it has digits at start. priority for name  is  "*inclinometer*"
    if not db_path:
        for p in (lambda p: [p, p.parent])(path_cruise):
            m = re.match('(^[\d_]*).*', p.name)
            if m:
                break
        db_name = f"{m.group(1).strip('_')}{device_short_name}.h5"
    else:
        db_name = db_path.name

    # set 'db_path' to be in "device_path / '_raw' / cfg['out']['db_name']" from 'path_cruise'
    if device_dir_pattern is None:
        device_dir_pattern = f'*{device_short_name}*'

    device_path = next((d for d in path_cruise.glob(device_dir_pattern) if d.is_dir()), path_cruise)
    if not db_path.is_absolute():
        db_path = device_path / '_raw' / db_name
    return device_path, db_path


# config_path="ini",
@hydra.main(config_name=cs_store_name)  # adds config store cs_store_name data/structure to :param config
def main(config: ConfigType) -> None:
    """
    ----------------------------
    Save data to Pandas HDF5 store*.h5
    ----------------------------
    The store contains tables for each device and each device table contains log with metadata of recording sessions

    :param config: with fields:
    - in - mapping with fields:
      - tables_log: - log table name or pattern str for it: in pattern '{}' will be replaced by data table name
      - cols_good_data: -
      ['dt_from_utc', 'db', 'db_path', 'table_nav']
    - out - mapping with fields:
      - cols: can use i - data row number and i_log_row - log row number that is used to load data range
      - cols_log: can use i - log row number
      - text_date_format
      - file_name_fun, file_name_fun_log - {fun} part of "lambda rec_num, t_st, t_en: {fun}" string to compile function
      for name of data and log text files
      - sep

    """
    global cfg
    cfg = to_vaex_hdf5.cfg_dataclasses.main_init(config, cs_store_name)
    cfg_in = cfg.pop('input')
    cfg_in['cfgFile'] = cs_store_name
    cfg['in'] = cfg_in
    # try:
    #     cfg = to_vaex_hdf5.cfg_dataclasses.main_init_input_file(cfg, cs_store_name, )
    # except Ex_nothing_done:
    #     pass  # existed db is not mandatory

    device_path, cfg['out']['db_path'] = device_in_out_paths(
        db_path=cfg['out'].get('db_path'),
        path_cruise=cfg['in']['path_cruise'],
        device_short_name=cfg['in']['probes_prefix'],
        device_dir_pattern='*inclinometer*'
        )

    out = cfg['out']
    # h5init(cfg['in'], out)


    probes = cfg['in']['probes'] or range(1, 41)  # sets default range, specify your values before line ---
    raw_root, probe_is_incl = re.subn('INCL_?', 'INKL_', cfg['in']['probes_prefix'].upper())

    # some parameters that depends of probe type (indicated by probes_prefix)
    p_type = defaultdict(
        # baranov's format
        constant_factory(
            {'correct_fun': partial(correct_txt,
                mod_file_name= mod_incl_name,
                sub_str_list=[b'^\r?(?P<use>20\d{2}(\t\d{1,2}){5}(\t\d{5}){8}).*', b'^.+']),
             'fs': 10,
             'format': 'Baranov',
            }),
        {'incl':
            {'correct_fun': partial(correct_txt,
                mod_file_name= mod_incl_name,
                sub_str_list=[b'^(?P<use>20\d{2}(,\d{1,2}){5}(,\-?\d{1,6}){6}(,\d{1,2}\.\d{2})(,\-?\d{1,3}\.\d{2})).*',
                              b'^.+']),
             'fs': 5,
             'format': 'Kondrashov',
            },
         'voln':
            {'correct_fun': partial(correct_txt,
                mod_file_name=mod_incl_name,
                sub_str_list=[b'^(?P<use>20\d{2}(,\d{1,2}){5}(,\-?\d{1,8})(,\-?\d{1,2}\.\d{2}){2}).*', b'^.+']),

             'fs': 5,
             #'tbl_prefix': 'w',
             'format': 'Kondrashov',
            }
        })


    if st(1, 'Save inclinometer or wavegauge data from ASCII to HDF5'):
        # Note: Can not find additional not corrected files for same probe if already have any corrected in search path (move them out if need)

        i_proc_probe = 0  # counter of processed probes
        i_proc_file = 0  # counter of processed files
        # patten to identify only _probe_'s raw data files that need to correct '*INKL*{:0>2}*.[tT][xX][tT]':

        raw_parent = dir_incl / '_raw'  # raw_parent /=
        if cfg['in']['raw_subdir'] is None:
            cfg['in']['raw_subdir'] = ''

        dir_out = raw_parent / re.sub(r'[.\\/ *?]', '_', cfg['in']['raw_subdir'])
        # sub replaces multilevel subdirs to 1 level that correct_fun() can only make

        def dt_from_utc_2000(probe):
            """ Correct time of probes started without time setting. Raw date must start from  2000-01-01T00:00"""
            return (datetime(year=2000, month=1, day=1) - cfg['in']['time_start_utc'][probe]
                    ) if cfg['in']['time_start_utc'].get(probe) else timedelta(0)

        # convert cfg['in']['dt_from_utc'] keys to int

        cfg['in']['dt_from_utc'] = {int(p): v for p, v in cfg['in']['dt_from_utc'].items()}
        # convert cfg['in']['t_start_utc'] to cfg['in']['dt_from_utc'] and keys to int
        cfg['in']['dt_from_utc'].update(    # overwriting the 'time_start_utc' where already exist
            {int(p): dt_from_utc_2000(p) for p, v in cfg['in']['time_start_utc'].items()}
            )
        # make cfg['in']['dt_from_utc'][0] be default value
        cfg['in']['dt_from_utc'] = defaultdict(constant_factory(cfg['in']['dt_from_utc'].pop(0, timedelta(0))),
                                               cfg['in']['dt_from_utc'])


        for probe in probes:
            raw_found = []
            raw_pattern_file = str(Path(glob.escape(cfg['in']['raw_subdir'])) / cfg['in']['raw_pattern'].format(prefix=raw_root, number=probe))
            correct_fun = p_type[cfg['in']['probes_prefix']]['correct_fun']
            # if not archive:
            if (not re.match(r'.*(\.zip|\.rar)$', cfg['in']['raw_subdir'], re.IGNORECASE)) and raw_parent.is_dir():
                raw_found = list(raw_parent.glob(raw_pattern_file))
            if not raw_found:
                # Check if already have corrected files for probe generated by correct_txt(). If so then just use them
                raw_found = list(dir_out.glob(f"{cfg['in']['probes_prefix']}{probe:0>2}.txt"))
                if raw_found:
                    print('corrected csv file', [r.name for r in raw_found], 'found')
                    correct_fun = lambda x, dir_out: x
                elif not cfg['in']['raw_subdir']:
                    continue

            for file_in in (raw_found or open_csv_or_archive_of_them(raw_parent, binary_mode=False, pattern=raw_pattern_file)):
                file_in = correct_fun(file_in, dir_out=dir_out)
                if not file_in:
                    continue
                tbl = file_in.stem  # f"{cfg['in']['probes_prefix']}{probe:0>2}"
                # tbl = re.sub('^((?P<i>inkl)|w)_0', lambda m: 'incl' if m.group('i') else 'w',  # correct name
                #              re.sub('^[\d_]*|\*', '', file_in.stem).lower()),  # remove date-prefix if in name
                csv2h5(
                    [str(Path(__file__).parent / 'cfg' / f"csv_{'inclin' if probe_is_incl else 'wavegauge'}_{p_type[cfg['in']['probes_prefix']]['format']}.ini"),
                    '--path', str(file_in),
                    '--blocksize_int', '50_000_000',  # 50Mbt
                    '--table', tbl,
                    '--db_path', str(db_path),
                    # '--log', str(scripts_path / 'log/csv2h5_inclin_Kondrashov.log'),
                    # '--on_bad_lines', 'warn',  # ?
                    '--b_interact', '0',
                    '--fs_float', str(p_type[cfg['in']['probes_prefix']]['fs']),  #f'{fs(probe, file_in.stem)}',
                    '--dt_from_utc_seconds', str(cfg['in']['dt_from_utc'][probe].total_seconds()),
                    '--b_del_temp_db', '1',
                    ] +
                   (
                   ['--csv_specific_param_dict', 'invert_magnitometr: True'
                    ] if probe_is_incl else []
                   ),
                    **{
                    'filter': {
                         'min_date': cfg['filter']['min_date'].get(probe, np.datetime64(0, 'ns')),
                         'max_date': cfg['filter']['max_date'].get(probe, np.datetime64('now', 'ns')),  # simple 'now' works in sinchronious mode
                        }
                    }
                )

                # Get coefs:
                l.info(f"Adding coefficients to {db_path}/{tbl} from {cfg['in']['db_coefs']}")
                try:
                    h5copy_coef(cfg['in']['db_coefs'], db_path, tbl)
                except KeyError as e:  # Unable to open object (component not found)
                    l.warning('No coefs to copy?')  # write some dummy coefficients to can load Veusz patterns:
                    h5copy_coef(None, db_path, tbl, dict_matrices=dict_matrices_for_h5(tbl=tbl))
                except OSError as e:
                    l.warning('Not found DB with coefs?')  # write some dummy coefficients to can load Veusz patterns:
                    h5copy_coef(None, db_path, tbl, dict_matrices=dict_matrices_for_h5(tbl=tbl))
                i_proc_file += 1
            else:
                print('no', raw_pattern_file, end=', ')
            i_proc_probe += 1
        print('Ok:', i_proc_probe, 'probes,', i_proc_file, 'files processed.')






    cfg_in['tables'] = ['incl30']
    from inclinometer.incl_h5clc import h5_names_gen
    from inclinometer.h5inclinometer_coef import rot_matrix_x, rot_matrix_y  #rotate_x, rotate_y
    # R*[xyz]. As we next will need apply coefs Ag = Rz*Ry*Rx we can incorporate this
    # operation by precalculate it adding known angles on each axes to Rz,Ry,Rx.
    # If rotation is 180 deg, then we can add it only to Rx. Modified coef: Ag_new = Rz*Ry*R(x+180)
    # R(x+180) = Rx*Rx180 equivalent to rotate Ag.T in opposite direction:
    # Ag_new = rotate_x()

    # inclinometer changed so that applying coefs returns rotated data fiels vectors:
    # Out_rotated = Ag * In
    # We rotate it back:
    # Out = rotate(Out_rotated) =
    # after  angle after calibration to some angle P so determine angle relative to vertical
    # by rotate data vector in opposite dir: Out = Ag * R_back * In. This equivalent to have new coef by apply rotation to Ag:
    # Ag_new = Ag * R_back = (R_back.T * Ag.T).T = rotate_forward(Ag.T).T =

    # Applying calibration coef will get data in inverted basis so we need rotate it after:
    #
    # coefs['Ag'] = rotate_x(coefs['Ag'], angle_degrees=180)
    # coefs['Ah'] = rotate_x(coefs['Ah'], angle_degrees=180)

    # df_log_old, cfg_out['db'], cfg_out['b_incremental_update'] = h5temp_open(**cfg_out)
    for i1, (tbl, coefs) in enumerate(h5_names_gen(cfg_in), start=1):
        # using property of rotation around same axis: R(x, θ1)@R(x, θ2) = R(x, θ1 + θ2)
        coefs['Ag'] = coefs['Ag'] @ rot_matrix_x(np.cos(np.pi), np.sin(np.pi))
        coefs['Ah'] = coefs['Ah'] @ rot_matrix_x(np.cos(np.pi), np.sin(np.pi))
        coefs['azimuth_shift_deg'] = 180
        h5copy_coef(None, cfg['out']['db_path'], tbl,
                    dict_matrices=dict_matrices_for_h5(coefs, tbl, to_nested_keys=True))



    # Calculate velocity and average
    if st(2):
        # if aggregate_period_s is None then not average and write to *.proc_noAvg.h5 else loading from that h5 and writing to _proc.h5
        if not cfg['out']['aggregate_period_s']:
            cfg['out']['aggregate_period_s'] = [None, 2, 600, 3600 if 'w' in cfg['in']['probes_prefix'] else 7200]

        if cfg['in']['azimuth_add']:
            if 'Lat' in cfg['in']['azimuth_add']:
                from datetime import datetime
                # add magnetic declination,° for used coordinates
                # todo: get time
                azimuth_add = mag_dec(cfg['in']['azimuth_add']['Lat'], cfg['in']['azimuth_add']['Lon'], datetime(2020, 9, 10), depth=-1)
            else:
                azimuth_add = 0
            if 'constant' in cfg['in']['azimuth_add']:
                # and add constant. For example, subtruct declination at the calibration place if it was applied
                azimuth_add += cfg['in']['azimuth_add']['constant']  # add -6.65644183° to account for calibration in Kaliningrad
        for aggregate_period_s in cfg['out']['aggregate_period_s']:
            if aggregate_period_s is None:
                db_path_in = db_path
                db_path_out = db_path.with_name(f'{db_path.stem}.proc_noAvg.h5')
            else:
                db_path_in = db_path.with_name(f'{db_path.stem}.proc_noAvg.h5')
                db_path_out = f'{db_path.stem}_proc.h5'  # or separately: '_proc{aggregate_period_s}.h5'

            args = [Path(incl_h5clc.__file__).with_name(f'incl_h5clc_{db_path.stem}.yaml'),
                    # if no such file all settings are here
                    '--db_path', str(db_path_in),
                    # !   'incl.*|w\d*'  inclinometers or wavegauges w\d\d # 'incl09':
                    '--tables_list', 'incl.*' if not cfg['in']['probes'] else f"incl.*(?:{'|'.join('{:0>2}'.format(p) for p in cfg['in']['probes'])})",
                    '--aggregate_period', f'{aggregate_period_s}S' if aggregate_period_s else '',
                    '--out.db_path', str(db_path_out),
                    '--table', f'V_incl_bin{aggregate_period_s}' if aggregate_period_s else 'V_incl',
                    '--verbose', 'INFO',  #'DEBUG' get many numba messages
                    '--b_del_temp_db', '1',
                    # '--calc_version', 'polynom(force)',  # depreshiated
                    # '--chunksize', '20000',
                    # '--not_joined_db_path', f'{db_path.stem}_proc.h5',
                    ]
            # if aggregate_period_s <= 5:   # [s], do not need split csv for big average interval
            #     args += (['--split_period', '1D'])
            if aggregate_period_s is None:  # proc. parameters (if we have saved proc. data then when aggregating we are not processing)
                args += (
                    ['--max_dict', 'M[xyz]:4096',
                     # Note: for Baranov's prog 4096 is not suited
                     # '--time_range_zeroing_dict', "incl19: '2019-11-10T13:00:00', '2019-11-10T14:00:00'\n,"  # not works - use kwarg
                     # '--time_range_zeroing_list', '2019-08-26T04:00:00, 2019-08-26T05:00:00'
                     '--split_period', '1D'
                    ] if subs_made else
                    ['--bad_p_at_bursts_starts_peroiod', '1H',
                    ])
            # csv splitted by 1day (default for no avg) and monolith csv if aggregate_period_s==600
            if aggregate_period_s not in cfg['out']['aggregate_period_s_not_to_text']:  # , 300, 600]:
                args += ['--text_path', str(db_path.parent / 'text_output')]
            kwarg = {'in': {
                'min_date': cfg['filter']['min_date'][0],
                'max_date': cfg['filter']['max_date'][0],
                'time_range_zeroing':  cfg['in']['time_range_zeroing'],
                'azimuth_add': azimuth_add
                }
                }
            # If need all data to be combined one after one:
            # set_field_if_no(kwarg, 'in', {})
            # kwarg['in'].update({
            #
            #         'tables': [f'incl{i:0>2}' for i in min_date.keys() if i!=0],
            #         'dates_min': min_date.values(),  # in table list order
            #         'dates_max': max_date.values(),  #
            #         })
            # set_field_if_no(kwarg, 'out', {})
            # kwarg['out'].update({'b_all_to_one_col': 'True'})


            incl_h5clc.main(args, **kwarg)


    # Calculate spectrograms.
    if st(3):  # Can be done at any time after step 1
        def raise_ni():
            raise NotImplementedError('Can not proc probes having different fs in one run: you need to do it separately')

        args = [
            Path(incl_h5clc.__file__).with_name(f'incl_h5spectrum{db_path.stem}.yaml'),
            # if no such file all settings are here
            '--db_path', str(db_path.with_name(f'{db_path.stem}.proc_noAvg.h5')),
            '--tables_list', f"{cfg['in']['probes_prefix']}.*",  # inclinometers or wavegauges w\d\d  ## 'w02', 'incl.*',
            # '--aggregate_period', f'{aggregate_period_s}S' if aggregate_period_s else '',

            '--min_date', datetime64_str(cfg['filter']['min_date'][0]),
            '--max_date', datetime64_str(cfg['filter']['max_date'][0]),  # '2019-09-09T16:31:00',  #17:00:00
            # '--max_dict', 'M[xyz]:4096',  # use if db_path is not ends with .proc_noAvg.h5 i.e. need calc velocity
            '--out.db_path', f"{db_path.stem.replace('incl', cfg['in']['probes_prefix'])}_proc_psd.h5",
            # '--table', f'psd{aggregate_period_s}' if aggregate_period_s else 'psd',
            '--fs_float', f"{fs(probes[0], cfg['in']['probes_prefix'])}",
            # (lambda x: x == x[0])(np.vectorize(fs)(probes, prefix))).all() else raise_ni()
            #
            # '--time_range_zeroing_list', '2019-08-26T04:00:00, 2019-08-26T05:00:00'
            # '--verbose', 'DEBUG',
            # '--chunksize', '20000',
            '--b_interact', '0',
            ]
        if 'w' in cfg['in']['probes_prefix']:
            args += ['--split_period', '1H',
                     '--dt_interval_minutes', '10',  # burst mode
                     '--fmin', '0.0001',
                     '--fmax', '4'
                     ]
        else:
            args += ['--split_period', '2H',
                     '--fmin', '0.0004',  #0.0004
                     '--fmax', '1.05'
                     ]

        incl_h5spectrum.main(args)

    # Draw in Veusz
    if st(4):
        b_images_only = True  # False
        pattern_path = db_path.parent / r'vsz_5min\191119_0000_5m_incl19.vsz'  # r'vsz_5min\191126_0000_5m_w02.vsz'
        if not b_images_only:
            pattern_bytes_slice_old = re.escape(b'((5828756, 5830223, None),)')

        # Length of not adjacent intervals, s (set None to not allow)
        period = '1D'
        length = '5m'  # period  # '1D'

        dt_custom_s = pd_period_to_timedelta(length) if length != period else None  # None  #  60 * 5

        if True:
            # Load starts and assign ends
            t_intervals_start = pd.read_csv(
                cfg['in']['path_cruise'] / r'vsz+h5_proc\intervals_selected.txt',
                converters={'time_start': lambda x: np.datetime64(x, 'ns')}, index_col=0).index
            edges = (
                pd.DatetimeIndex(t_intervals_start), pd.DatetimeIndex(t_intervals_start + dt_custom_s))  # np.zeros_like()
        else:
            # Generate periodic intervals
            t_interval_start, t_intervals_end = intervals_from_period(datetime_range=np.array(
                [cfg['filter']['min_date']['0'], cfg['filter']['max_date']['0']],
                # ['2018-08-11T18:00:00', '2018-09-06T00:00:00'],
                # ['2019-02-11T13:05:00', '2019-03-07T11:30:00'],
                # ['2018-11-16T15:19', '2018-12-14T14:35'],
                # ['2018-10-22T12:30', '2018-10-27T06:30:00'],
                'datetime64[s]'), period=period)
            edges = (pd.DatetimeIndex([t_interval_start]).append(t_intervals_end[:-1]),
                     pd.DatetimeIndex(t_intervals_end))

        for i, probe in enumerate(probes):
            probe_name = f"{cfg['in']['probes_prefix']}{probe:02}"  # table name in db
            l.info('Draw %s in Veusz: %d intervals...', probe_name, edges[0].size)
            # for i_interval, (t_interval_start, t_interval_end) in enumerate(zip(pd.DatetimeIndex([t_interval_start]).append(t_intervals_end[:-1]), t_intervals_end), start=1):

            cfg_vp = {'veusze': None}
            for i_interval, (t_interval_start, t_interval_end) in enumerate(zip(*edges), start=1):

                # if i_interval < 23: #<= 0:  # TEMPORARY Skip this number of intervals
                #     continue
                if period != length:
                    t_interval_start = t_interval_end - pd.Timedelta(dt_custom_s, 's')

                try:  # skipping absent probes
                    start_end = h5q_interval2coord(
                        db_path=str(db_path),
                        table=f'/{probe_name}',
                        t_interval=(t_interval_start, t_interval_end))
                    if not len(start_end):
                        break  # no data
                except KeyError:
                    break  # device name not in specified range, go to next name

                pattern_path_new = pattern_path.with_name(f"{t_interval_start:%y%m%d_%H%M}_{length}_{probe_name}.vsz")

                # Modify pattern file
                if not b_images_only:
                    probe_name_old = re.match('.*((?:incl|w)\d*).*', pattern_path.name).groups()[0]
                    bytes_slice = bytes('(({:d}, {:d}, None),)'.format(*(start_end + np.int32([-1, 1]))), 'ascii')


                    def f_replace(line):
                        """
                        Replace in file
                        1. probe name
                        2. slice
                        """
                        # if i_interval == 1:
                        line, ok = re.subn(bytes(probe_name_old, 'ascii'), bytes(probe_name, 'ascii'), line)
                        if ok:  # can be only in same line
                            line = re.sub(pattern_bytes_slice_old, bytes_slice, line)
                        return line


                    if not rep_in_file(pattern_path, pattern_path_new, f_replace=f_replace):
                        l.warning('Veusz pattern not changed!')
                        # break
                    elif cfg_vp['veusze']:
                        cfg_vp['veusze'].Load(str(pattern_path_new))
                elif cfg_vp['veusze']:
                    cfg_vp['veusze'].Load(str(pattern_path_new))

                txt_time_range = \
                    """
                    "[['{:%Y-%m-%dT%H:%M}', '{:%Y-%m-%dT%H:%M}']]" \
                    """.format(t_interval_start, t_interval_end)
                print(f'{i_interval}. {txt_time_range}', end=' ')

                cfg_vp = veuszPropagate.main(
                    [Path(veuszPropagate.__file__).parent.with_name('veuszPropagate.ini'),
                     # '--data_yield_prefix', '-',
                     '--path', str(db_path),  # use for custom loading from db and some source is required
                     '--tables_list', f'/{probe_name}',  # 181022inclinometers/ \d*
                     '--pattern_path', str(pattern_path_new),
                     # fr'd:\workData\BalticSea\190801inclinometer_Schuka\{probe_name}_190807_1D.vsz',
                     # str(db_path.parent / dir_incl / f'{probe_name}_190211.vsz'), #warning: create file with small name
                     # '--before_next', 'restore_config',
                     # '--add_to_filename', f"_{t_interval_start:%y%m%d_%H%M}_{length}",
                     '--filename_fun', f'lambda tbl: "{pattern_path_new.name}"',
                     '--add_custom_list',
                     'USEtime',  # nAveragePrefer',
                     '--add_custom_expressions_list',
                     txt_time_range,
                     # + """
                     # ", 5"
                     # """,
                     '--b_update_existed', 'True',
                     '--export_pages_int_list', '1, 2',  # 0 for all '6, 7, 8',  #'1, 2, 3'
                     # '--export_dpi_int', '200',
                     '--export_format', 'emf',
                     '--b_interact', '0',
                     '--b_images_only', f'{b_images_only}',
                     '--return', '<embedded_object>',  # reuse to not bloat memory
                     ],
                    veusze=cfg_vp['veusze'])


def call_example(call_by_user=True):
    """
    to run from IDE or from bat-file with parameters
    --- bat file ---
    call conda.bat activate py3.7x64h5togrid
    D: && cd D:\Work\_Python3\And0K\h5toGrid
    python -c "from to_vaex_hdf5.autofon_coord import call_example; call_example()"
    ----------------
    # python -m to_vaex_hdf5.autofon_coord.call_example() not works
    :return:
    """
    # from to_vaex_hdf5.h5tocsv import main_call as h5tocsv
    path_db_in = Path(r'd:\WorkData\~configuration~\inclinometr\190710incl.h5')
    path_db_out = Path(r'd:\workData\BalticSea\201202_BalticSpit\inclinometer\_raw\201202.raw.h5')
    device = ['tr0']  # 221912
    to_vaex_hdf5.cfg_dataclasses.main_call([  # '='.join(k,v) for k,v in pairwise([   # ["2021-04-08T08:35:00", "2021-04-14T11:45:00"]'
        'in.time_range=["2021-04-08T09:00:00", "now"]',   # UTC, max (will be loaded and updated what is absent)
        f'in.db_path="{path_db_in}"',
        #'input.dt_from_utc_hours=3',
        'process.zeroing_pitch=180',
        # 'process.period_segments="2H"',
        f'out.db_path="{path_db_out}"'
        ], main)


if __name__ == '__main__':
    call_example()

    #main()