#@+leo-ver=5-thin
#@+node:3947220829000151.20211215072208.1: * @file download_copernicus.py
#!/usr/bin/env python
#@+others
#@+node:3947220829000151.20211215192713.1: ** Declarations (download_copernicus.py)
"""
Download data from Copernicus to NetCDF file - official motuclient configured to my needs with detecting last data date
See also
 - netcdf2csv.py to convert result to csv
 - download_ecmwf.py

python -m motuclient ^
 --motu https://nrt.cmems-du.eu/motu-web/Motu ^
 --service-id BALTICSEA_ANALYSISFORECAST_PHY_003_006-TDS ^
 --product-id dataset-bal-analysis-forecast-phy-monthlymeans ^
 --longitude-min 9.041585922241211 ^
 --longitude-max 30.208660125732422 ^
 --latitude-min 53.008296966552734 ^
 --latitude-max 65.8909912109375 ^
 --date-min "2021-04-16 00:00:00" ^
 --date-max "2021-04-16 00:00:00" ^
 --depth-min 0 ^
 --depth-max 205 ^
 --variable mlotst ^
 --variable so ^
 --variable sob ^
 --variable uo ^
 --variable vo ^
 --out-dir <OUTPUT_DIRECTORY> ^
 --out-name <OUTPUT_FILENAME> ^
 --user <USERNAME> ^
 --pwd <PASSWORD>
"""

import logging
import sys
from pathlib import Path

# from datetime import datetime
# import pandas as pd
from motuclient import motuclient

#from motuclient import main as retrieve, ERROR_CODE_EXIT
# LIBRARIES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'motu_utils')
# # Manage imports of project libraries
# if not os.path.exists(LIBRARIES_PATH):
#     sys.stderr.write('\nERROR: can not find project libraries path: %s\n\n' % os.path.abspath(LIBRARIES_PATH))
#     sys.exit(1)
# sys.path.append(LIBRARIES_PATH)
# # Import project libraries
# import utils_log
# import utils_messages
# import motu_api


#@-others
if False:
    from motuclient import initLogger, motu_api
    import motu_api

dir_save = r'd:\WorkData\BalticSea\_other_data\_model\Copernicus\section_z'
# file_nc_name = f'wind@ECMWF-ERA5_area({lat_st},{lon_st},{lat_en},{lon_en}).nc'
# {file_nc_name}

l = logging.getLogger(__name__)
l.info(f'Downloading to {dir_save}...')

# use_date_range = ['2020-09-01', '2021-03-31']  # ['2020-12-01', '2021-01-31']  # ['2018-12-01', '2018-12-31']
# date_range = [datetime.strptime(t, '%Y-%m-%d') for t in use_date_range]  # T%H:%M:%S
# file_date_prefix = '{:%y%m%d}-{:%m%d}'.format(*date_range)

# common = {
#     'class': 'ei',
#     'dataset': 'interim',
#     'date': '{}/to/{}'.format(*use_date_range),
#     'expver': '1',
#     'grid': '0.75/0.75',
#     'area': '{}/{}/{}/{}'.format(lat_st, lon_st, lat_en, lon_en),  # SWSE
#     'levtype': 'sfc',
#     'param': '165.128/166.128',
#     'stream': 'oper',
#     'format': 'netcdf'
if True:
    # retrieve Copernicus Marine Environment Monitoring Service (CMEMS) data from Motu HTTP server
    # https://github.com/ecmwf/cdsapi
    #
    # Code requires {UID}:{API Key} is in my %USERPROFILE% (C:\Users\{USER}\.cmems-copernicus.ini)

    # import urllib3
    # urllib3.disable_warnings()  prevent InsecureRequestWarning... Adding certificate verification is strongly advised.



    if False:
        start_time = datetime.now()
        initLogger()

        try:
            # we prepare options we want
            _options = load_options()

            if _options.log_level != None:
                logging.getLogger().setLevel(int(_options.log_level))

            motu_api.execute_request(_options)
        except Exception as e:
            log.error("Execution failed: %s", e)
            if hasattr(e, 'reason'):
                log.info(' . reason: %s', e.reason)
            if hasattr(e, 'code'):
                log.info(' . code  %s: ', e.code)
            if hasattr(e, 'read'):
                try:
                    log.log(utils_log.TRACE_LEVEL, ' . detail:\n%s', e.read())
                except:
                    pass

            log.debug('-' * 60)
            log.debug("Stack trace exception is detailed hereafter:")
            exc_type, exc_value, exc_tb = sys.exc_info()
            x = traceback.format_exception(exc_type, exc_value, exc_tb)
            for stack in x:
                log.debug(' . %s', stack.replace('\n', ''))
            log.debug('-' * 60)

    path_config = Path('scripts/cfg/coperinicus_api_client.ini').absolute()  # /scripts
    if not path_config.is_file():
        l.error('Error %s - config not found!', path_config)
        sys.exit(1)

    print(path_config)
    sys.argv = sys.argv[:1] + [
        '--config-file', str(Path.home() / '.cmems-copernicus.ini'),
        '--config-file', str(path_config),
        '--out-dir', dir_save
        ]
    options = motuclient.load_options()  # get input options for replace output name:
    out_name = '{date_min:%y%m%d}{product_id}_' \
        '{latitude_min:g}-{latitude_max:g},{longitude_min:g}-{longitude_max:g}.nc'.format_map(vars(options))
    sys.argv.append(f'--out-name {out_name}')
    motuclient.main()

    if False and (  # Convert to modern NetCDF format
        (path_out := (Path(dir_save) / out_name)).is_file() and not
        (path_out_cnv := path_out.with_suffix('.h5')).is_file()
       ):
        import subprocess
        subprocess.call(['nccopy', '-k', '4', str(path_out), str(path_out_cnv)])

    #(Path(dir_save) / 'CMEMS_no_name_data.nc').rename('')
    if motuclient.ERROR_CODE_EXIT != 0:  # not informative
        sys.exit(motuclient.ERROR_CODE_EXIT)






print('ok')

#@+<<data description>>
#@+node:3947220829000151.20211215192837.1: ** <<data description>>
"""
This Baltic Sea physical model product provides forecasts for the physical conditions in the Baltic Sea. The Baltic forecast is updated twice daily providing a new six days forecast. Three datasets are provided with hourly instantaneous values for sea level variations, ice concentration and thickness at the surface, and temperature, salinity and horizontal velocities for the 3D field. Additionally a dataset with 15 minutes (instantaneous) values are provided for the sea level variation and the horizontal surface currents. The product is produced by a Baltic Sea set up of the NEMOv4.0 ocean model. This product is provided at the models native grid with a resolution of 1 nautical mile in the horizontal, and up to 56 vertical depth levels. The area covers the Baltic Sea including the transition area towards the North Sea (i.e. the Danish Belts, the Kattegat and Skagerrak). The ocean model is offline coupled with Stokes drift data from the Baltic Wave forecast product (BALTICSEA_ANALYSISFORECAST_WAV_003_010).
Classification
Product ID
BALTICSEA_ANALYSISFORECAST_PHY_003_006
Published
5 October 2011
Originating centre
SMHI (Sweden)
Area
Baltic Sea
Time
FuturePresentPast
Universe
White OceanBlue Ocean
Main variables
Current velocity Salinity Sea ice Sea surface height Temperature

NAME	DESCRIPTION	STANDARD NAME	UNITS
bottomT	Sea water potential temperature at sea floor (given for depth comprise between 0 and 500m)	sea_water_potential_temperature_at_sea_floor	degrees_C
mlotst	Ocean mixed layer thickness defined by density (as in de Boyer Montegut, 2004)	ocean_mixed_layer_thickness_defined_by_sigma_theta	m
so	salinity	sea_water_salinity	1e-3
sob	Sea water salinity at sea floor	sea_water_salinity_at_sea_floor	0.001
thetao	potential temperature	sea_water_potential_temperature	degree_Celsius
uo	Eastward current	eastward_sea_water_velocity	m s-1
vo	Northward current	northward_sea_water_velocity	m s-1
sla	Sea level elevation	sea_surface_height_above_sea_level	m
siconc	Sea ice cover	sea_ice_area_fraction	1
sithick	Sea ice thickness	sea_ice_thickness	m

## BALTICSEA_REANALYSIS_PHY_003_011-TDS
python -m motuclient --motu https://my.cmems-du.eu/motu-web/Motu --service-id BALTICSEA_REANALYSIS_PHY_003_011-TDS --product-id dataset-reanalysis-nemo-dailymeans --longitude-min 18.972 --longitude-max 19.235 --latitude-min 55.841 --latitude-max 55.929 --date-min "2014-12-01 12:00:00" --date-max "2020-12-31 12:00:00" --depth-min 66.5774 --depth-max 121.7625 --variable bottomT --variable so --variable sob --variable thetao --variable uo --variable vo --out-dir <OUTPUT_DIRECTORY> --out-name <OUTPUT_FILENAME> --user <USERNAME> --pwd <PASSWORD>

python -m motuclient.motuclient --motu https://my.cmems-du.eu/motu-web/Motu --service-id BALTICSEA_REANALYSIS_BIO_003_012-TDS --product-id dataset-reanalysis-scobi-dailymeans --longitude-min 18.972 --longitude-max 19.235 --latitude-min 55.841 --latitude-max 55.929 --date-min "2014-12-01 12:00:00" --date-max "2020-12-31 12:00:00" --depth-min 66.5774 --depth-max 121.7625 --variable o2 --variable o2b --out-dir <OUTPUT_DIRECTORY> --out-name <OUTPUT_FILENAME> --user <USERNAME> --pwd <PASSWORD>

python -m motuclient --motu https://my.cmems-du.eu/motu-web/Motu --service-id BLKSEA_MULTIYEAR_PHY_007_004-TDS --product-id bs-cmcc-cur-rean-d --longitude-min 27.37 --longitude-max 41.9626 --latitude-min 40.86 --latitude-max 46.8044 --date-min "2020-05-31 12:00:00" --date-max "2020-05-31 12:00:00" --depth-min 2.501 --depth-max 2.5011 --variable uo --variable vo --out-dir <OUTPUT_DIRECTORY> --out-name <OUTPUT_FILENAME> --user <USERNAME> --pwd <PASSWORD>

# not works:
python -m motuclient.motuclient --motu https://nrt.cmems-du.eu/motu-web/Motu --service-id BLKSEA_MULTIYEAR_PHY_007_004-TDS --product-id bs-cmcc-cur-rean-d --longitude-min 33.972 --longitude-max 33.972 --latitude-min 44.37 --latitude-max 44.4 --depth-min 2.501 --depth-max 32.9969 --date-min 2019-10-29 --date-max 2019-11-18 --variable uo --variable vo --out-dir <OUTPUT_DIRECTORY> --out-name <OUTPUT_FILENAME> --user <USERNAME> --pwd <PASSWORD>
# works:
python -m motuclient.motuclient --motu https://my.cmems-du.eu/motu-web/Motu --service-id BLKSEA_MULTIYEAR_PHY_007_004-TDS --product-id bs-cmcc-cur-rean-d --longitude-min 33.972 --longitude-max 33.972 --latitude-min 44.37 --latitude-max 44.4 --date-min "2019-10-29 12:00:00" --date-max "2019-11-18 12:00:00"
--depth-min 2.501 --depth-max 32.9969  --variable uo  --variable vo  --out-dir <OUTPUT_DIRECTORY> --out-name <OUTPUT_FILENAME> --user <USERNAME> --pwd <PASSWORD>
"""
#@-<<data description>>
#@@language python
#@@tabwidth -4
#@-leo
