import logging
from pathlib import Path
import copernicusmarine as cm
import re
import xarray as xr  # for interpolating

from utils import interp_to_point

l = logging.getLogger(__name__)
# environment to run this script:
# conda create -n py12copernicus python=3.12 copernicusmarine jupyter_console -c conda-forge
# conda activate py12copernicus && jupyter console


def download_extending_coords(
    save_dir: str, lat: float, lon: float, date_range: list, delta: float = 0.125
) -> None:
    """
    Download and save as a NetCDF file a subset of CMEMS data for a region covering (lat,lon) ± delta,
    for enable interpolation of the data to the exact point later.

    Parameters:
    1. save_dir (str): directory to save files, e.g., "D:\\meteo\\CMEMS"
    2. lat (float): target latitude, e.g., 55.13533
    3. lon (float): target longitude, e.g., 19.76305
    4. date_range (list): [start_datetime, end_datetime], e.g., ['2024-06-25', '2024-09-05']
    5. delta (float): extension in degrees to obtain neighboring grid points (default 0.125)
    """
    path_save = Path(save_dir)
    if not path_save.is_dir():
        if path_save.parent.is_dir():
            path_save.mkdir()
        else:
            raise FileNotFoundError(f"dir_save={save_dir}")
    l.info(f"downloading CMEMS data to {save_dir}...")
    # задаем расширенный регион для получения данных для интерполяции
    min_lon = lon - delta
    max_lon = lon + delta
    min_lat = lat - delta
    max_lat = lat + delta
    file_path = cm.subset(
        dataset_id="cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H",
        variables=[
            "eastward_wind",
            "northward_wind",
            "northward_wind_sdd",
            "eastward_wind_sdd",
            "air_density",
        ],
        minimum_longitude=min_lon,
        maximum_longitude=max_lon,
        minimum_latitude=min_lat,
        maximum_latitude=max_lat,
        start_datetime=date_range[0],
        end_datetime=date_range[1],
        output_directory=save_dir,
        force_download=True,
    )
    l.info(f"Extended subset saved to {file_path}")
    # print(f"downloaded file: {file_path}")
    return file_path


if __name__ == "__main__":
    # Параметры: (directory, latitude, longitude, [start, end])
    # - dir_save: directory
    # - lat, lon: latitude, longitude
    # - date_range: Set [] to load from last loaded data to now
    dir_save__lat__lon__date_range = [
        (r'D:\WorkData\BalticSea-rc\230825_Kulikovo@ADCP,ADV,i,tr\meteo\CMEMS', 54.9896, 20.299717, ['2023-08-20', '2023-09-20'])
        # d:\WorkData\BalticSea\230825_Kulikovo@ADCP,ADV,i,tr\meteo\CMEMS54.99, 20.3,
        # (r"D:\WorkData\BalticSea\240616_ABP56-240827_AI68-copy\meteo\CMEMS", 55.13533, 19.76305, ['2024-06-25', '2024-09-05'])  # st30 i72
        # (r"D:\WorkData\BalticSea\240616_ABP56\meteo\CMEMS", 55.13533, 19.76305, ['2024-06-25', '2024-09-05'])  # st30 i72
        # (r"C:\Work\Veusz\meteo\CMEMS", 54.744417, 19.5799, ['2024-06-25', '2024-09-05'])
        # (r'd:\WorkData\BalticSea\240827_AI68\meteo\CMEMS', 55.10090, 20.01027, ['2024-06-25', '2024-09-01'])
        # (r'd:\WorkData\BalticSea\240827_AI68\meteo\CMEMS', 55.837282, 19.05611, ['2024-06-25', '2024-09-01'])
        # (r'd:\WorkData\BalticSea\_Pregolya,Lagoon\231208@i19,ip5,6\meteo', 54.64485, 21.07382, ['2023-10-01', '2023-12-01'])
        # (r'd:\WorkData\BalticSea\231121_ABP54\meteo\CMEMS', 55.8701, 19.05, ['2023-11-01', '2024-05-02'])
        # (r'd:\WorkData\BalticSea\220505_D6\meteo\CMEMS', 55.3266, 20.5789, ['2022-05-01', '2022-05-31']),

        # dir_save, lat, lon, date_range = r'd:\WorkData\BalticSea\230616_Kulikovo@i3,4,19,37,38,p1-3,5,6\meteo', 54.95328, 20.32387, ['2023-06-15', '2023-07-25']
        # dir_save, lat, lon, date_range = r'd:\WorkData\BalticSea\221105_ASV54\meteo', 55.88, 19.12, ['2022-11-01', '2023-05-01']
        # dir_save, lat, lon, date_range = r'd:\WorkData\BalticSea\230507_ABP53\meteo', 55.922656, 19.018713, ['2023-05-01', '2023-05-31']
        # dir_save, lat, lon, date_range = r'd:\WorkData\KaraSea\220906_AMK89-1\meteo', 72.33385, 63.53786, ['2022-09-01', '2022-09-15']
        # dir_save, lat, lon, date_range = r'd:\WorkData\BalticSea\230423inclinometer_Zelenogradsk\meteo', 54.953351, 20.444820, None # ['2023-04-23', '2023-05-01']
        # r'e:\WorkData\BalticSea\181005_ABP44\meteo', 55.88333, 19.13139, ['2018-10-01', '2019-01-01']
        # r'd:\WorkData\BalticSea\220601_ABP49\meteo'
        # r'd:\workData\BalticSea\201202_BalticSpit_inclinometer', 54.615, 19.841  ## Pionersky: 54.9689, 20.2446
        # 55.32659, 20.57875  # 55.874845000, 19.116386667  # masha: 54.625, 19.875 #need: N54.61106 E19.84847
        # ['2022-05-01', '2022-05-20']
        # ['2022-06-01', '2022-06-23']  # ['2020-12-01', '2021-01-31']  # ['2018-12-01', '2018-12-31'], ['2020-09-01', '2021-09-16']
    ]

    logging.basicConfig(level=logging.INFO)
    for point_param_in in dir_save__lat__lon__date_range:
        if False:  # True:  #
            responce = download_extending_coords(*point_param_in)
            path_loaded = responce.file_path
        else:
            path_loaded = Path(
                r"D:\WorkData\BalticSea-rc\230825_Kulikovo@ADCP,ADV,i,tr\meteo\CMEMS\cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H_multi-vars_20.19E-20.31E_54.94N-55.06N_2023-08-20-2023-09-20.nc"
            )
            # Path(point_param_in[0]) / "cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H_multi-vars_19.69E-19.81E_55.06N-55.19N_2024-06-25-2024-09-05.nc"
        path_interp = interp_to_point(path_loaded, *point_param_in[1:3])
        print(path_interp, "saved")

# ######################
if False:  # old my code
    if dir_save__lat__lon__date_range:
        for dir_save, lat, lon, date_range in dir_save__lat__lon__date_range:
            path_save = Path(dir_save)
            if path_save.is_dir():
                pass
            elif path_save.parent.is_dir():
                path_save.mkdir()
            else:
                raise (FileNotFoundError(f"dir_save={dir_save}"))
            l.info(f"Downloading CMEMS data to {dir_save}...")
            path_saved = cm.subset(
                dataset_id="cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H",
                # dataset_version="202207",
                variables=[
                    "eastward_wind",
                    "northward_wind",
                    "northward_wind_sdd",
                    "eastward_wind_sdd",
                    "air_density",
                ],
                minimum_longitude=lon,
                maximum_longitude=lon,
                minimum_latitude=lat,
                maximum_latitude=lat,
                start_datetime=date_range[0],
                end_datetime=date_range[1],
                output_directory=dir_save,
                force_download=True,  # don't ask save each file
            )
            print(path_saved)
            pass