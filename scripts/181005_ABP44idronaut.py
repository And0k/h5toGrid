# -*- coding: utf-8 -*-
import sys
from pathlib import Path

drive_d = 'D:' if sys.platform == 'win32' else '/mnt/D'  # allows to run on both my Linux and Windows systems:
scripts_path = Path(drive_d + '/Work/_Python3/And0K/h5toGrid/scripts')
sys.path.append(str(Path(scripts_path).parent.resolve()))
from os import path as os_path
# my funcs
import veuszPropagate
from to_pandas_hdf5.csv2h5 import main as csv2h5
from to_pandas_hdf5.gpx2h5 import main as gpx2h5
from h5toGpx import main as h5toGpx
from grid2d_vsz import main as grid2d_vsz

path_cruise = r'd:\workData\BalticSea\181005_ABP44'  # d:\WorkData\181005_ABP44
path_in_Idronaut = os_path.join(path_cruise, r'CTD_Idronaut#493\18*[0-9].txt')
path_db = os_path.join(path_cruise, '181005_ABP44.h5')
go = False
start = 1
# ---------------------------------------------------------------------------------------------
go = True
if st(1):  # False: #
    # Draw CTD_Idronaut#493 data profiles
    veuszPropagate.main(['ini/veuszPropagate.ini',
                         '--path', os_path.join(path_cruise, r'CTD_Idronaut#493\18*[0-9].txt'),
                         '--pattern_path', os_path.join(path_cruise, r'CTD_Idronaut#493\181009_0154.vsz'),
                         '--before_next', 'restore_config',
                         '--eval_list',
                         """
                         "ImportFileCSV(u'{nameRFE}', blanksaredata=True, dateformat=u'hh:mm:ss', delimiter=u' ', encoding=u'ascii', headermode=u'1st', linked=True, dsprefix=u'i3_', renames={{u'i3_Date': u'i3_date_txt', u'i3_O2%': u'i3_O2', u'i3_Time': u'i3_time_part'}}, skipwhitespace=True)",
                         "TagDatasets(u'loaded', [u'i3_ChlA', u'i3_Cond', u'i3_Eh', u'i3_O2', u'i3_O2ppm', u'i3_Pres', u'i3_Sal', u'i3_Temp', u'i3_Turb', u'i3_date_txt', u'i3_pH', u'i3_time_part'])"
                         """,
                         # '--export_pages_int_list', '1', #'--b_images_only', 'True'
                         '--b_interact', '0',
                         # '--b_update_existed', '1'
                         ])
go = False
if st(2):  # False: #
    # Save CTD_Idronaut#493 to DB
    csv2h5([
        'ini/csv_CTD_Idronaut.ini',
        '--path', os_path.join(path_cruise, r'CTD_Idronaut#493\18*[0-9].txt'),
        '--dt_from_utc_hours', '2',
        '--header', 'Number,Date(text),Time(text),Pres,Temp,Sal,O2,O2ppm,SIGMA,Cond,Vbatt',
        '--cols_not_use_list', 'Number,SIGMA,Vbatt',
        '--delimiter_chars', '\\ \\',  # ''\s+',
        '--table', 'CTD_Idronaut#493',
        '--b_interact', '0',
        # '--b_raise_on_err', '0'
        ])

if st(3):  # False: #
    # Save navigation to DB
    gpx2h5(['',
            '--db_path', path_db,
            '--path', os_path.join(path_cruise, r'navigation\*tracks*.gpx'),
            '--tables_list', ',navigation,',  # skip waypoints
            '--table_prefix', r''])

if st(4):  # False: #
    # Extract navigation data at time station starts to GPX waypoints
    h5toGpx(['ini/h5toGpx_CTDs.ini',
             '--db_path', path_db,
             '--tables_list', 'CTD_Idronaut#493',
             '--select_from_tablelog_ranges_index', '0'
             ])

if st(5):  # False: #
    # Save waypoints/routes from manually prepared gpx to hdf5
    gpx2h5(['', '--path', os_path.join(path_cruise, r'navigation\CTD-sections=routes.gpx'),
            '--table_prefix', r'navigation/sectionsCTD'])

if st(6):  # False: #
    # Gridding
    grid2d_vsz(['ini/grid2d_vsz.ini', '--db_path', path_db,
                '--table_sections', r'navigation/sectionsCTD_routes',
                '--subdir', 'CTD-sections',
                '--data_columns_list', 'Temp, Sal, SigmaTh, O2, O2ppm'  # 'N^2'
                ])

########################################################################################

if st(10):  # False: #
    # Save gpx from treckers to DB
    gpx2h5(['',
            '--db_path', path_db,  # str(Path().with_name('trackers_temp')),
            '--path', os_path.join(path_cruise, r'navigation\*spot*.gpx'),
            '--table_prefix', r'navigation/',
            '--segments_cols_list', "time, latitude, longitude, comment",
            '--output_files.segments_cols_list', 'Time, Lat, Lon, comment',
            '--tables_list', ',,tracker{}', ])
# go = True
if st(11):  # False: #
    # Export treckers tracks to GPX tracks
    h5toGpx(['ini/h5toGpx_nav_all.ini',
             '--db_path', path_db,
             '--tables_list', 'tracker{}',
             '--select_from_tablelog_ranges_index', None,  # Export tracks
             '--gpx_names_fun_format', '{1:%y%m%d}_{0:}',  # track name of format(timeLocal, tblD_safe)
             '--gpx_names_funs_list', '"i, row.Index"',
             '--gpx_names_funs_cobined', ''
             ])
# extract all navigation tracks
if False:  # True: #
    # sys.argv[0]= argv0   os_path.join(os_path.dirname(file_h5toGpx)
    h5toGpx(['ini/h5toGpx_nav_all.ini',
             '--path_cruise', path_cruise,
             '--tables_list', 'navigation',
             '--simplify_tracks_error_m_float', '10',
             '--select_from_tablelog_ranges_index', None])
