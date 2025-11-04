import logging
from typing import List, Tuple, Dict, Any, Optional
import hydra
from omegaconf import DictConfig, OmegaConf

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "downloading"))

import utils
from .manager import DownloadHistoryManager

l = logging.getLogger(__name__)

try:
    import copernicusmarine as cm
except ImportError:
    l.error("copernicusmarine library not found. Download functionalities will not be unavailable!")
    cm = None # Set to None if import fails


def download_extending_coords(
    dataset_id,
    variables,
    date_range: List[str],
    coord: Tuple[float, float],
    coord_delta,
    output_directory: Path,
    **kwargs
) -> Tuple[Optional[Path], Dict[str, Any]]:
    """
    Download and save as a NetCDF file a subset of CMEMS data for a region covering (lat,lon) ± delta,
    for enable interpolation of the data to the exact point later.

    :param cfg: Hydra configuration object containing necessary parameters.
    :param save_dir: directory to save files, e.g., Path("D:\\meteo\\CMEMS")
    :param coord: target coordinate tuple (latitude, longitude), e.g., (55.13533, 19.76305)
    :param date_range: [start_datetime, end_datetime], e.g., ['2024-06-25', '2024-09-05']
    :param kwargs: other copernicusmarine.subset() download options
    """
    if not output_directory.is_dir():
        output_directory.mkdir(parents=True, exist_ok=True)
    l.info(f"downloading CMEMS data to {output_directory}...")
    lat, lon = coord
    download_options = {
        "dataset_id": dataset_id,
        "variables": list(variables),  # Ensure variables is a list
        "minimum_longitude": lon - coord_delta,
        "maximum_longitude": lon + coord_delta,
        "minimum_latitude": lat - coord_delta,
        "maximum_latitude": lat + coord_delta,
        "start_datetime": date_range[0],
        "end_datetime": date_range[-1],
        "output_directory": str(output_directory),
        **kwargs
    }


    try: # Only attempt download if copernicusmarine is imported
        file_path = Path(cm.subset(**download_options))
        l.info(f"Extended subset saved to {file_path}")
    except Exception as e:
        file_path = None
        l.exception(f"CMEMS download failed: {e} for {coord}, {date_range}: (options: {download_options})")
        download_options["error"] = str(e)

    return file_path, download_options


@hydra.main(config_path="cfg", config_name="base", version_base=None)
def main(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO)
    history_manager = DownloadHistoryManager(Path(cfg.base.history_file))

    # Check if command line contains "projects" override to determine which config to use
    overrides = OmegaConf.to_container(cfg, resolve=True)
    override_keys = list(overrides.keys()) if overrides else []
    has_projects_override = 'projects' in override_keys

    # Determine which config to use based on overrides
    if has_projects_override:
        # Load the corresponding config based on command-line projects override
        projects_config = cfg.get('projects', None)
        if projects_config and len(projects_config) > 0:
            # Use the first project config if available
            project_name = list(projects_config.keys())[0] if isinstance(projects_config, dict) else None
            if project_name and hasattr(cfg.copernicus, project_name):
                config_section = getattr(cfg.copernicus, project_name)
            else:
                # Default to point_wind if no specific project config is found
                config_section = cfg.copernicus.point_wind
        else:
            # Default to point_wind config
            config_section = cfg.copernicus.point_wind
    else:
        # Use copernicus.point_wind config when no projects override is specified
        config_section = cfg.copernicus.point_wind

    dir_save = Path(cfg.base.local_path) / config_section.dir_save
    date_range = config_section.date_range
    points = config_section.points or []
    if cfg.gpx_path:
        points_from_gpx = utils.extract_coordinates_from_gpx(cfg.gpx_path, cfg.gpx_waypoints_re).values()
        points += list(points_from_gpx)

    for i, point in enumerate(points):
        l.info(f"Processing download for Lat: {point.lat}, Lon: {point.lon}, Date Range: {date_range}")
        download_options = {}
        # Download the extended region data
        path_loaded, download_options = download_extending_coords(
            dataset_id=config_section.dataset_id,
            variables=list(config_section.variables),  # Convert OmegaConf list to Python list,
            date_range=date_range,
            coord=(point.lat, point.lon),
            coord_delta=cfg.base.interpolation_delta,
            output_directory=dir_save,
            netcdf_compression_level=cfg.copernicus.get('netcdf_compression_level'),
            force_download=True,
        )

        if path_loaded:  # Only proceed if download was successful
            # Interpolate to the exact point
            path_interp = utils.interp_to_point(path_loaded, point.lat, point.lon)
            l.info(f"Interpolated data saved to: {path_interp}")
        else:
            l.warning("No file downloaded! Skipping interpolation and logging as 'skipped'.")
            download_options.update({
                "status": "skipped"
            })
        history_manager.log_download(
            dir_save=dir_save,
            coords=[(point.lat, point.lon)],
            date_range=date_range,
            options=download_options
        )

    # print("\n--- Download History ---")
    # for entry in history_manager.get_history():
    #     print(entry)

if __name__ == "__main__":
    main()