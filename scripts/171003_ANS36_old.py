from os import path as os_path

# my funcs


path_cruise = r'd:\WorkData\BalticSea\171003_ANS36'
path_in_OS320 = os_path.join(path_cruise, r'CTD_Idronaut_OS320\txt\17????_*.txt')
path_in_SST_M48 = os_path.join(path_cruise, r'CTD_S&S_48M\CSV\17????_*.CSV')

path0 = os_path.dirname(os_path.dirname(__file__))  # os_getcwd()
argv0 = os_path.join(path0, 'script')

# Draw profiles
if False:  # True: #
    # CTD_Idronaut#494 data
    if False:  # True: #
        veuszPropagate([os_path.join(os_path.dirname(file_veuszPropagate), 'veuszPropagate.ini')])

    # CTD_Idronaut_OS320 data
    if True:  # True: #
        veuszPropagate([
            os_path.join(os_path.dirname(file_veuszPropagate), 'veuszPropagate.ini'),
            '--path', path_in_OS320,
            '--pattern_path', r'd:\workData\BalticSea\171003_ANS36\CTD_Idronaut_OS320\171012_0816.vsz',
            '--import_method', 'ImportFileCSV',
            # '--b_images_only', 'True'
            ])

# #todo: SST M48 data
# if True: #True: #
#         veuszPropagate([
#             os_path.join(os_path.dirname(file_veuszPropagate), 'veuszPropagate.ini'),
#             '--path', path_in_SST_M48,
#             '--pattern_path', r'd:\workData\BalticSea\171003_ANS36\CTD_Idronaut_OS320\171012_0816.vsz',
#             '--import_method', 'ImportFileCSV',
#             #'--b_images_only', 'True'
#         ])

# to DB
if False:  # True: #

    sys.argv[0] = os_path.join(path0, 'to_pandas_hdf5', 'script')  # to find ini
    os_chdir(os_path.dirname(sys.argv[0]))  # to can write log

    # CTD_Idronaut#493 (sinked)
    if False:  # True: #
        csv2h5([
            'csv_CTD_Idronaut.ini',
            '--path', os_path.join(path_cruise, r'CTD_Idronaut#493\txt\17????_????.txt'),
            '--header',
            'date(text),txtT(text),Pres(float),Temp(float),Cond(float),Sal(float),O2(float),O2ppm(float),pH(float),Eh(float),ChlA(float),Turb(float)',
            '--table', 'CTD_Idronaut#493'])  # sys.argv[1:]=

    # CTD
    if False:  # True: #
        csv2h5(['csv_CTD_Idronaut.ini',
                # '--path', 'd:\workData\BalticSea\171003_ANS36\CTD_Idronaut#494\txt\*_.txt'
                ])

    # CTD_Idronaut_OS320
    # Replace all "/20(\d\d)\.(\d*) " to "/20$1 00:00:00.$2 " in REDAS txt output before!
    if False:  # True: #
        csv2h5([
            'csv_CTD_IdrRedas.ini',
            '--path', path_in_OS320,
            '--header',
            'Time(text),Pres(float),Temp(float),Cond(float),Sal(float),SigmaT(float),O2,O2ppm,SoundVel,Lat(float),Lon(float)',
            '--delimiter_chars', '\t',
            '--table', 'CTD_Idronaut_OS320'])

    # CTD_SST_M48
    if False:  # True: #
        csv2h5([
            'csv_CTD_SST.ini',
            '--path', path_in_SST_M48,
            '--header', 'Date(text),Time(text),Pres,Temp,Sal,SIGMA,Cond,SOUND,Vbatt',
            '--cols_not_save_list', 'SIGMA,SOUND,Vbatt',
            '--delimiter_chars', ',',
            '--table', 'CTD_SST_M48'])
    # navigation
    if False:  # True: #
        csv2h5(['csv_nav_supervisor.ini'])

    # Meteo
    if False:  # True: #
        csv2h5([
            'csv_meteo.ini', '--path',  # to_pandas_hdf5/
            os_path.join(path_cruise, r"meteo\ship's_meteo_st_source\*.mxt"), '--header',
            'date(text),Time(text),t_air,Vabs_m__s,Vdir,dew_point,Patm,humidity,t_w,precipitation',
            '--coldate_integer', '0', '--coltime_integer', '1',
            '--cols_not_save_list', 't_w,precipitation',  # bad constant data
            '--delimiter_chars', ',', '--max_text_width', '12',
            '--on_bad_lines', 'warn', '--b_insert_separator', 'False',
            '--chunksize_percent_float', '500'
            ])

    # BAKLAN Matlab txt ouput
    if True:  # False: #
        csv2h5([
            'csv_CTD_IdrRedas.ini',
            '--path', os_path.join(path_cruise, r'Baklan\txt\20*p1d5.txt'),
            '--header', 'counts,Pres(float),Cond(float),Sh3i,Sh3,Sh1,Sh2i,Sh2,T1LPF71,'
                        'Temp(float),Gx_m/s^2,Gy_m/s^2,Gz_m/s^2,TP,Time(float),dC_mS/cm_/m,dT1_°C/m,Sal,SigmaTh(float),eps2varSh,eps3varSh_W/kg,Speed',
            '--delimiter_chars', '\t',
            '--table', 'CTD_Idronaut_OS320'])

    os_chdir(path0)

# extract CTD runs (if files not splitted on runs):
# to_pandas_hdf5/ctd_calc.py

# extract navigation data at time station starts to GPX waypoints
if False:  # True: #
    # sys.argv[0]= argv0
    h5toGpx([os_path.join(os_path.dirname(file_h5toGpx), 'h5toGpx_CTDs.ini'),
             '--select_from_tablelog_ranges_index', '0'
             ])

# extract all navigation tracks
if False:  # True: #
    # sys.argv[0]= argv0
    h5toGpx([os_path.join(os_path.dirname(file_h5toGpx), 'h5toGpx_nav_all.ini'),
             '--path_cruise', path_cruise,
             '--tables_list', 'navigation',
             '--simplify_tracks_error_m_float', '10',
             '--select_from_tablelog_ranges_index', None])

# Now change starts of sections and excluded stations with specified symbol
# using MapSource or other GPX waypoints/routes editor
# and save result as a new GPX-sections file

# Load waypoints/routes to hdf5
if False:  # True: #
    # sys.argv[0]= argv0 to_pandas_hdf5 /
    gpx2h5(['',  # os_path.join(os_path.dirname(file_h5toGpx), 'Gpx2h5.ini')#'--path_cruise', path_cruise,
            '--path', os_path.join(path_cruise, r'navigation\CTD-sections=routes.gpx'),
            '--table_prefix', r'navigation/CTDsections'])

# for Baklan: execute scripts/filetime_to_gpx.py

# Gridding
if False:  # True:
    grid2d_vsz()
