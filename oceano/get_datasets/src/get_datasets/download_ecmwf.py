#!/usr/bin/env python
"""
Download ECMWF ERA5 data from Climate Data Store to NetCDF file
See also
https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation
- netcdf2csv.py to convert result to csv
- download_copernicus.py
"""
import logging
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Tuple, List, Dict
import pandas as pd

import cdsapi
import urllib3
import requests

try:  # allow run from parent dir
    from scripts.downloading.netcdf2csv import main as netcdf2csv
    import scripts.downloading.utils
except ImportError:  # run from current dir
    from netcdf2csv import main as netcdf2csv
    import utils

l = logging.getLogger(__name__)


def group_variables_by_resolution(variables: List[str]) -> Dict[Tuple[float, float], List[str]]:
    """
    Groups variables into resolution groups for batch downloading.

    :param variables: list of variable names
    :return: dictionary of {(lat_res, lon_res): [variables]} groups
    """

    coarse = {"swh", "mwp", "mwd"}  # wave model: ~0.5 deg
    grouped = defaultdict(list)
    for var in variables:
        res = (0.5, 0.5) if var in coarse else (0.25, 0.25)
        grouped[res].append(var)
    return grouped


if __name__ == "__main__":

    # Set:
    # - dir_save
    # - lat_st, lon_st
    # - use_date_range - Last day will be included.
    # Old: If previous text data exist you can net [] to load from last loaded data to now
    dir_save, lat_st, lon_st, use_date_range = (
        # r"D:\WorkData\BalticSea\240616_ABP56\meteo\ECMWF", 54.744417, 19.5799, ['2024-06-25', '2024-09-10']  # t-chain
        # r'd:\WorkData\BalticSea\240827_AI68\meteo\ECMWF', 55.10090, 20.01027, ['2024-06-25', '2024-09-01'],
        # r'd:\WorkData\BalticSea\_Pregolya,Lagoon\231208@i19,ip5,6\meteo', 54.64485, 21.07382, ['2023-10-01', '2023-12-01']
        # r'd:\WorkData\BalticSea\231121_ABP54\meteo\ECMWF', 55.00100, 20.29770, ['2023-11-01', '2023-12-15']
        # r'd:\WorkData\BalticSea\220505_D6\meteo\ECMWF', 55.3266, 20.5789, ['2022-05-01', '2022-05-31']
        r"B:\WorkData\BalticSea\230825_Kulikovo@ADCP,ADV,i,tr\meteo\ECMWF",
        54.99,
        20.3,
        ["2023-08-25", "2023-09-10"],
        # r'd:\WorkData\BalticSea\230825_Kulikovo@ADCP,ADV,i,tr\meteo_ECMWF', 54.99, 20.3, ['2023-08-25', ]
        # dir_save, lat_st, lon_st, use_date_range = r'd:\WorkData\BalticSea\230616_Kulikovo@i3,4,19,37,38,p1-3,5,6\meteo', 54.95328, 20.32387, ['2023-06-15', '2023-07-25']
        # dir_save, lat_st, lon_st, use_date_range = r'd:\WorkData\BalticSea\221105_ASV54\meteo', 55.88, 19.12, ['2022-11-01', '2023-05-01']
        # dir_save, lat_st, lon_st, use_date_range = r'd:\WorkData\BalticSea\230507_ABP53\meteo', 55.922656, 19.018713, ['2023-05-01', '2023-05-31']
        # dir_save, lat_st, lon_st, use_date_range = r'd:\WorkData\KaraSea\220906_AMK89-1\meteo', 72.33385, 63.53786, ['2022-09-01', '2022-09-15']
        # dir_save, lat_st, lon_st, use_date_range = r'd:\WorkData\BalticSea\230423inclinometer_Zelenogradsk\meteo', 54.953351, 20.444820, None # ['2023-04-23', '2023-05-01']
        # r'e:\WorkData\BalticSea\181005_ABP44\meteo', 55.88333, 19.13139, ['2018-10-01', '2019-01-01']
        # r'd:\WorkData\BalticSea\220601_ABP49\meteo'
        # r'd:\workData\BalticSea\201202_BalticSpit_inclinometer', 54.615, 19.841  ## Pionersky: 54.9689, 20.2446
        # 55.32659, 20.57875  # 55.874845000, 19.116386667  # masha: 54.625, 19.875 #need: N54.61106 E19.84847
        # ['2022-05-01', '2022-05-20']
        # ['2022-06-01', '2022-06-23']  # ['2020-12-01', '2021-01-31']  # ['2018-12-01', '2018-12-31'], ['2020-09-01', '2021-09-16']
    )

    dGrid = 0  # 0.25
    lat_en = lat_st + dGrid
    lon_en = lon_st + dGrid


    if not use_date_range:
        # Get interval from last data timestamp we have to now
        with utils.ReverseTxt(
                '',
            # r'd:\WorkData\BalticSea\230423inclinometer_Zelenogradsk\meteo\230423wind@ECMWF-ERA5(N54.953,E20.445).tsv',
            # r'd:\workData\BalticSea\201202_BalticSpit_inclinometer\wind\200901wind@ECMWF-ERA5(N54.615,E19.841).tsv',
                encoding='ascii') as f_prev:
            last_line = next(f_prev)
        print('last ECMWF data found:', last_line)
        use_date_range = [last_line.split()[0], f'{datetime.utcnow():%Y-%m-%d}']
    elif len(use_date_range) <= 1 or not use_date_range[1]:
        use_date_range = [use_date_range[0] or '2020-09-01', f'{datetime.utcnow():%Y-%m-%d}']

    date_range = [datetime.strptime(t, '%Y-%m-%d') for t in use_date_range]  # T%H:%M:%S
    file_date_prefix = '{:%y%m%d}-{:%m%d}'.format(*date_range)

    l.info('Downloading interval {} - {}...', *date_range)

    dir_save = Path(dir_save)
    if not dir_save.is_dir():
        raise (FileNotFoundError(f"dir_save={dir_save}"))

    dir_name = "".join([
        file_date_prefix,
        "" if "ECMWF" in dir_save.name else "wind@ECMWF-ERA5_",
        "area({2}-{0}N,{1}-{3}E)".format(*utils.grid_aligned_bbox(lat_st, lon_st)),
    ])


    # European Centre for Medium-Range Weather Forecasts (ECMWF)
    # retrieve of ERA5 data from Climate Data Store
    # https://github.com/ecmwf/cdsapi
    #
    # Code requires {UID}:{API Key} is in my C:\Users\{USER}\.cdsapirc
    # manager = PoolManager(num_pools=2)
    proxies = {
        # 'http': 'http://127.0.0.1:28080',
        # 'https': 'http://127.0.0.1:28080',
        # 'ftp': 'http://127.0.0.1:28080'
        }
    dataset = "reanalysis-era5-single-levels"
    # 'reanalysis-cerra-single-levels'  # last "year": ["2021"]
    # s  # reanalysis-era5-land - not at sea
    variables = [
        '10m_u_component_of_wind',
        '10m_v_component_of_wind',
        'surface_pressure',
        "sea_surface_temperature", # 'skin_temperature'
        "total_precipitation",
        '10m_wind_gust_since_previous_post_processing',  # 'instantaneous_10m_wind_gust',
    ] + (
        [
            "10m_wind_direction",
            "10m_wind_speed",
            "2m_relative_humidity",
        ] if dataset == 'cerra-single-levels' else [
            'mean_wave_direction',
            'mean_wave_period',
            'peak_wave_period',
            'significant_height_of_combined_wind_waves_and_swell',
        ]
    )  # (if dataset == 'reanalysis-era5-single-levels' else [])

    product_types = (
        {
            "level_type": "surface_or_atmosphere",
            "data_type": ["reanalysis"],
            "product_type": "analysis",
        }
        if dataset == "reanalysis-cerra-single-levels"
        else {"product_type": ["reanalysis"]}
    )

    days_period_idx = pd.period_range(*use_date_range, freq='D')
    data_format = "netcdf"  # "grib",
    download_format = "zip"

    path_dest = (dir_save / dir_name)
    dir_name = dir_name.removeprefix(file_date_prefix)  # todo: move to date dir or add date to files names

    file_loaded = dir_save / dir_name / "data_stream-oper_stepType-instant.nc"  # one (main) of loaded files
    path_zip = path_dest.with_suffix(".zip") if download_format == "zip" else ""
    existed_files = [p for p in [file_loaded, path_zip] if p.is_file()]
    if any(existed_files):
        print("Destination exist:", f'"{dir_save.name}/{existed_files[0].name}"', "Repostprocessing it...")
    else:
        print(f'Downloading to "{dir_save.name}/{(path_zip or path_dest).name }"...')

        data_request = {
            **product_types,
            **{
                "variable": variables,
                "time": [f"{i:02d}:00" for i in range(24)],  # 1D resolution if comment
                "date": [f"{fd}" for fd in days_period_idx],
                # "grid": [0.25, 0.25],  # force grid?
                "area": utils.grid_aligned_bbox(lat_st, lon_st),
                "data_format": data_format,
                "download_format": download_format,
            },
        }
        print(
            f"Request area:", data_request["area"], "time range:", use_date_range
        )
        # l.info(f"")

        groups = group_variables_by_resolution(variables)
        with requests.Session() as session:
            session.proxies.update(proxies)  # not works
            c = cdsapi.Client(session=session)  # warning_callback=
            urllib3.disable_warnings()  # prevent InsecureRequestWarning... Adding certificate verification is strongly advised.  # better use warning_callback=?


            for (res_lat, res_lon), vars_in_group in groups.items():
                add_sfx = "_res=({res_lat:g}, {res_lon:g})" if res_lat != res_lon else "_res={res_lat:g}"
                data_request["area"] = utils.grid_aligned_bbox(lat_st, lon_st, res_lat, res_lon)
                c.retrieve(
                    dataset,
                    data_request, path_dest.with_name(
                        f"{path_dest.stem}{add_sfx}{(path_zip or path_dest).suffix}"
                    )
                )

    if not any(p for p in existed_files if p.suffix != ".zip"):
        utils.extract_zip_to_named_dir(path_zip, target_dir=dir_save / dir_name)

    files_loaded = [
        f
        for f in (dir_save / dir_name).glob("*.grib" if data_format == "grib" else "data_stream*.nc")
        if "-to_" not in f.stem
    ]
    utils.h5_format(
        files_loaded, **{"requested_latitude": lat_st, "requested_longitude": lon_st}, backend=None
    )
    try:
        netcdf2csv(list((dir_save / dir_name).glob("*.nc")), variables=variables)
        print(f"Data extracted and converted to csv. You can delete {path_zip.name} now")
    except NotImplementedError:
        pass

    for f in files_loaded:
        path_interp = utils.interp_to_point(f, lat_st, lon_st, backend=None)
        print(path_interp, "saved")

    # except Exception as e:  # Hard to debug with
    #     raise(e)

print(f"{datetime.now():%Y-%m-%d %H:%M:%S} Ok>")

import sys
sys.exit()
#%% Errors
# InsecureRequestWarning: Unverified HTTPS request is being made to host 'cds.climate.copernicus.eu'


if False:  # old
    from ecmwfapi import ECMWFDataServer

    server = ECMWFDataServer()
    l.info("part 1")
    server.retrieve({
        **common,
        **{
            "step": "0",
            "time": "00:00/06:00/12:00/18:00",
            "type": "an",
            "target": file_date_prefix + "analysis.nc",
        },
    })
    l.info("part 2")
    server.retrieve({
        **common,
        **{
            "step": "3/9",  # '3/6/9/12'
            "time": "00:00/12:00",
            "type": "fc",
            "target": file_date_prefix + "forecast.nc",
        },
    })


    #%% Good request from site
    request = {
        "product_type": ["reanalysis"],
        "variable": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "mean_sea_level_pressure",
            "mean_wave_direction",
            "mean_wave_period",
            "sea_surface_temperature",
            "significant_height_of_combined_wind_waves_and_swell",
        ],
        "year": ["2024"],
        "month": ["06", "07", "08", "09"],
        "day": [
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "08",
            "09",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
            "16",
            "17",
            "18",
            "19",
            "20",
            "21",
            "22",
            "23",
            "24",
            "25",
            "26",
            "27",
            "28",
            "29",
            "30",
            "31",
        ],
        "data_format": "netcdf",
        "download_format": "zip",
        "area": [54.79, 19.53, 54.69, 19.62],
    }
    c = cdsapi.Client()
    c.retrieve(
        dataset,
        request,
        (path_nc := (dir_save / dir_name)),
    ).download()

    # not needed (have done because of date err but it was in product_type):
    # "year": [f"{y:02}" for y in sorted(set(days_period_idx.year))],
    # "month": [f"{m:02}" for m in sorted(set(days_period_idx.month))],
    # "day": [f"{d:02}" for d in sorted(set(days_period_idx.day))],

    # "format": "netcdf",  # old api


    """
    import cdsapi

    dataset = "reanalysis-era5-single-levels"
    request = {
        "product_type": ["reanalysis"],
        "variable": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "mean_sea_level_pressure",
            "mean_wave_direction",
            "mean_wave_period",
            "sea_surface_temperature",
            "significant_height_of_combined_wind_waves_and_swell"
        ],
        "year": ["2024"],
        "month": [
            "06", "07", "08",
            "09"
        ],
        "day": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12",
            "13", "14", "15",
            "16", "17", "18",
            "19", "20", "21",
            "22", "23", "24",
            "25", "26", "27",
            "28", "29", "30",
            "31"
        ],
        "data_format": "netcdf",
        "download_format": "zip",
        "area": [54.79, 19.53, 54.69, 19.62]
    }

    client = cdsapi.Client()
    client.retrieve(dataset, request).download()


    ---

    c = cdsapi.Client()

    c.retrieve(
        'reanalysis-cerra-single-levels',
        {
            'format': 'netcdf',
            'variable': [
                '10m_wind_direction', '10m_wind_speed', '2m_relative_humidity',
                'skin_temperature', 'surface_pressure',
            ],
            'level_type': 'surface_or_atmosphere',
            'data_type': 'reanalysis',
            'product_type': 'analysis',
            'year': [
                '2020', '2021',
            ],
            'month': [
                '01', '02', '03',
                '04', '05', '06',
                '09', '10', '11',
                '12',
            ],
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',
            ],
            'time': [
                '00:00', '03:00', '06:00',
                '09:00', '12:00', '15:00',
                '18:00', '21:00',
            ],
        },
        'download.nc')





    c.retrieve(
        'reanalysis-cerra-single-levels',
        {
            'variable': [
                '10m_wind_direction', '10m_wind_gust_since_previous_post_processing', '10m_wind_speed',
                '2m_relative_humidity', 'skin_temperature', 'surface_pressure',
            ],
            'level_type': 'surface_or_atmosphere',
            'data_type': 'reanalysis',
            'product_type': 'forecast',
            'year': '2021',
            'month': '06',
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
            ],
            'time': [
                '00:00', '03:00', '06:00',
                '09:00', '12:00', '15:00',
                '18:00', '21:00',
            ],
            'format': 'netcdf',
            'leadtime_hour': [
                '1', '2', '3',
                '4', '5', '6',
                '9', '12', '15',
                '18', '21', '24',
                '27', '30',
            ],
        },
        'download.nc')
    """

    common = {
        'class': 'ei',
        'dataset': 'interim',
        'date': '{}/to/{}'.format(*use_date_range),
        'expver': '1',
        'grid': '0.75/0.75',
        'area': '{}/{}/{}/{}'.format(lat_st, lon_st, lat_en, lon_en),  # SWSE
        'levtype': 'sfc',
        'param': '165.128/166.128',
        'stream': 'oper',
        'format': 'netcdf'
        }