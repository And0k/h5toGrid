from os import path as os_path

# my funcs
from grid2d_vsz import main as grid2d_vsz

path_cruise = r'd:\WorkData\BalticSea\171003_ANS36'
path_in_OS320 = os_path.join(path_cruise, r'CTD_Idronaut_OS320\txt\17????_*.txt')
path_in_SST_M48 = os_path.join(path_cruise, r'CTD_S&S_48M\CSV\17????_*.CSV')

path_db = os_path.join(path_cruise, '171003Strahov.h5')
# path0 = os_path.dirname(os_path.dirname(__file__)) #os_getcwd()
# argv0 = os_path.join(path0, 'script')


# Draw profiles
if False:  # True: #
    # CTD_Idronaut#494 data
    if False:  # True: #
        veuszPropagate('cfg/veuszPropagate.ini')

    # CTD_Idronaut_OS320 data
    if True:  # True: #
        veuszPropagate(['cfg/veuszPropagate.ini',
                        '--path', path_in_OS320,
                        '--pattern_path', r'd:\workData\BalticSea\171003_ANS36\CTD_Idronaut_OS320\171012_0816.vsz',
                        '--import_method', 'ImportFileCSV',
                        # '--b_images_only', 'True'
                        ])

# #todo: SST M48 data
# if True: #True: #
#         veuszPropagate(['veuszPropagate.ini',
#             '--path', path_in_SST_M48,
#             '--pattern_path', r'd:\workData\BalticSea\171003_ANS36\CTD_Idronaut_OS320\171012_0816.vsz',
#             '--import_method', 'ImportFileCSV',
#             #'--b_images_only', 'True'
#         ])

# to DB
if True:  # False: #

    # sys.argv[0] = os_path.join(path0, 'to_pandas_hdf5', 'script')   # to find ini
    # os_chdir(os_path.dirname(sys.argv[0]))                          # to can write log

    # CTD_Idronaut#493 (sinked)
    if False:  # True: #
        csv2h5([
            'cfg/csv_CTD_Idronaut.ini',
            '--path', os_path.join(path_cruise, r'CTD_Idronaut#493\txt\17????_????.txt'),
            '--header',
            'date(text),txtT(text),Pres(float),Temp(float),Cond(float),Sal(float),O2(float),O2ppm(float),pH(float),Eh(float),ChlA(float),Turb(float)',
            '--table', 'CTD_Idronaut#493'])  # sys.argv[1:]=

    # CTD
    if False:  # True: #
        csv2h5(['cfg/csv_CTD_Idronaut.ini',
                # '--path', 'd:\workData\BalticSea\171003_ANS36\CTD_Idronaut#494\txt\*_.txt'
                ])

    # CTD_Idronaut_OS320
    # Replace all "/20(\d\d)\.(\d*) " to "/20$1 00:00:00.$2 " in REDAS txt output before!
    if False:  # True: #
        csv2h5([
            'cfg/csv_CTD_IdrRedas.ini',
            '--path', path_in_OS320,
            '--header',
            'Time(text),Pres(float),Temp(float),Cond(float),Sal(float),SigmaT(float),O2,O2ppm,SoundVel,Lat(float),Lon(float)',
            '--delimiter_chars', '\t',
            '--table', 'CTD_Idronaut_OS320'])

    # CTD_SST_M48
    if False:  # True: #
        csv2h5([
            'cfg/csv_CTD_SST.ini',
            '--path', path_in_SST_M48,
            '--header', 'Date(text),Time(text),Pres,Temp,Sal,SIGMA,Cond,SOUND,Vbatt',
            '--cols_not_save_list', 'SIGMA,SOUND,Vbatt',
            '--delimiter_chars', ',',
            '--table', 'CTD_SST_M48'])
    # navigation
    if False:  # True: #
        csv2h5(['cfg/csv_nav_supervisor.ini'])

    # Meteo
    if False:  # True: #
        csv2h5([
            'cfg/csv_meteo.ini', '--path',  # to_pandas_hdf5/
            os_path.join(path_cruise, r"meteo\ship's_meteo_st_source\*.mxt"), '--header',
            'date(text),Time(text),t_air,Vabs_m__s,Vdir,dew_point,Patm,humidity,t_w,precipitation',
            '--coldate_integer', '0', '--coltime_integer', '1',
            '--cols_not_save_list', 't_w,precipitation',  # bad constant data
            '--delimiter_chars', ',', '--max_text_width', '12',
            '--on_bad_lines', 'warn', '--b_insert_separator', 'False',
            '--chunksize_percent_float', '500'
            ])

    # BAKLAN Matlab txt ouput
    if False:  # True: #
        csv2h5([
            'cfg/csv_Baklan.ini', '--db_path', path_db,
            '--path', os_path.join(path_cruise, r'Baklan\20*p1.txt'),
            '--delimiter_chars', '\\t',
            '--table', 'Baklan_d100'])
    if False:  # True: #
        h5toGpx(['cfg/h5toGpx_Baklan.ini', '--db_path', path_db,
                 '--select_from_tablelog_ranges_index', '0'])
    # also to get gpx is possible to execute ../scripts/filetime_to_gpx.py without csv2h5

    # Now change starts of sections and excluded stations with specified symbol using MapSource

    if False:  # True: #
        gpx2h5([
            '', '--db_path', path_db,
            '--path', os_path.join(path_cruise, r'navigation\Baklan_d100_sections.gpx'),
            '--table_prefix', r'navigation/sectionsBaklan_d100'])

# extract CTD runs (if files not splitted on runs):
# to_pandas_hdf5/CTD_calc.py

# extract navigation data at time station starts to GPX waypoints
if False:  # True: #
    # sys.argv[0]= argv0
    h5toGpx(['cfg/h5toGpx_CTDs.ini',
             '--select_from_tablelog_ranges_index', '0'
             ])

# extract all navigation tracks
if False:  # True: #
    # sys.argv[0]= argv0   os_path.join(os_path.dirname(file_h5toGpx)
    h5toGpx(['cfg/h5toGpx_nav_all.ini',
             '--path_cruise', path_cruise,
             '--tables_list', 'navigation',
             '--simplify_tracks_error_m_float', '10',
             '--select_from_tablelog_ranges_index', None])

# Load waypoints/routes to hdf5
if False:  # True: #

    # Now change starts of sections and excluded stations with specified symbol #
    # using MapSource or other GPX waypoints/routes editor                      #
    # and save result as a new GPX-sections file                                #

    gpx2h5(['',  # os_path.join(os_path.dirname(file_h5toGpx), 'Gpx2h5.ini')#'--path_cruise', path_cruise,
            '--path', os_path.join(path_cruise, r'navigation\CTD-sections=routes.gpx'),
            '--table_prefix', r'navigation/sectionsCTD'])

# Gridding
if True:  # False: #
    grid2d_vsz(['cfg/grid2d_vsz.ini', '--db_path', path_db,
                '--table_sections', r'navigation/sectionsBaklan_d100_routes',
                '--subdir', 'CTD-sections/Baklan',
                '--data_columns_list', 'Temp, Sal, SigmaTh, N^2, eps3varSh'  # 'N^2'
                ])
