import logging
from typing import List, Tuple, Dict, Any, Optional
import hydra
from omegaconf import DictConfig, OmegaConf

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "downloading"))

from utils import interp_to_point
from manager import DownloadHistoryManager

l = logging.getLogger(__name__)

try:
    import copernicusmarine as cm
except ImportError:
    l.warning("copernicusmarine library not found. Some download functionalities may be unavailable.")
    cm = None # Set to None if import fails

def download_extending_coords(
    cfg: DictConfig, save_dir: Path, lat: float, lon: float, date_range: List[str]
) -> Tuple[Optional[Path], Dict[str, Any]]:
    """
    Download and save as a NetCDF file a subset of CMEMS data for a region covering (lat,lon) ± delta,
    for enable interpolation of the data to the exact point later.

    Parameters:
    1. cfg (DictConfig): Hydra configuration object containing necessary parameters.
    2. save_dir (Path): directory to save files, e.g., Path("D:\\meteo\\CMEMS")
    3. lat (float): target latitude, e.g., 55.13533
    4. lon (float): target longitude, e.g., 19.76305
    5. date_range (List[str]): [start_datetime, end_datetime], e.g., ['2024-06-25', '2024-09-05']
    """
    delta = cfg.base.interpolation_delta  # Get delta from config
    dataset_id = cfg.copernicus.point_wind.dataset_id
    variables = list(cfg.copernicus.point_wind.variables)  # Convert OmegaConf list to Python list

    if not save_dir.is_dir():
        save_dir.mkdir(parents=True, exist_ok=True)
    l.info(f"downloading CMEMS data to {save_dir}...")

    min_lon = lon - delta
    max_lon = lon + delta
    min_lat = lat - delta
    max_lat = lat + delta

    file_path = None
    if cm: # Only attempt download if copernicusmarine is imported
        file_path = cm.subset(
            dataset_id=dataset_id,
            variables=variables,
            minimum_longitude=min_lon,
            maximum_longitude=max_lon,
            minimum_latitude=min_lat,
            maximum_latitude=max_lat,
            start_datetime=date_range[0],
            end_datetime=date_range[1],
            output_directory=str(save_dir), # copernicusmarine expects string
            force_download=True,
        )
        l.info(f"Extended subset saved to {file_path}")
    else:
        l.warning("Skipping CMEMS download: copernicusmarine library not available.")

    download_options = {
        'dataset_id': dataset_id,
        'variables': variables,
        'min_lon': min_lon,
        'max_lon': max_lon,
        'min_lat': min_lat,
        'max_lat': max_lat,
        'delta': delta
    }
    return Path(file_path) if file_path else None, download_options


@hydra.main(config_path="cfg", config_name="base", version_base=None)
def main(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO)
    history_manager = DownloadHistoryManager(Path(cfg.base.history_file))

    # The specific project configuration is now composed by Hydra
    # and available directly under cfg.copernicus.point_wind
    dir_save = Path(cfg.base.local_path) / cfg.copernicus.point_wind.dir_suffix
    date_range = cfg.copernicus.point_wind.date_range

    for point in cfg.copernicus.point_wind.points:
        lat = point.lat
        lon = point.lon

        l.info(f"Processing download for Lat: {lat}, Lon: {lon}, Date Range: {date_range}")

        try:
            # Download the extended region data
            path_loaded, download_options = download_extending_coords(
                cfg=cfg,  # Pass the config object
                save_dir=dir_save,
                lat=lat,
                lon=lon,
                date_range=date_range
            )

            if path_loaded:  # Only proceed if download was successful
                # Interpolate to the exact point
                path_interp = interp_to_point(path_loaded, lat, lon)
                l.info(f"Interpolated data saved to: {path_interp}")

                # Log the successful download
                history_manager.log_download(
                    dir_save=dir_save,
                    lat=lat,
                    lon=lon,
                    date_range=date_range,
                    options=download_options
                )
            else:
                l.warning(f"No file downloaded for {lat}, {lon}, {date_range}. Skipping interpolation and logging as 'skipped'.")
                history_manager.log_download(
                    dir_save=dir_save,
                    lat=lat,
                    lon=lon,
                    date_range=date_range,
                    options={**download_options, 'status': 'skipped', 'reason': 'copernicusmarine library not available or download failed'}
                )

        except Exception as e:
            l.error(f"Error during download or interpolation for {lat}, {lon}, {date_range}: {e}")
            # Log failure or partial download if needed
            history_manager.log_download(
                dir_save=dir_save,
                lat=lat,
                lon=lon,
                date_range=date_range,
                options={**download_options, 'status': 'failed', 'error': str(e)}
            )

    print("\n--- Download History ---")
    for entry in history_manager.get_history():
        print(entry)

if __name__ == "__main__":
    main()