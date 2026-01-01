import logging
import sys
from pathlib import Path
import re
from typing import List, Tuple, Dict, Any, Optional
import requests
import hydra
from omegaconf import DictConfig, OmegaConf


# sys.path.append(str(Path(__file__).parent.parent.parent / "downloading"))

from get_datasets import utils
from get_datasets.manager import DownloadHistoryManager

l = logging.getLogger(__name__)

try:
    import copernicusmarine as cm
except ImportError:
    l.error("copernicusmarine library not found. Download functionalities will not be unavailable!")
    cm = None

def extract_error_from_xml(xml_string):
    # This function seems to be for debugging/logging XML errors,
    # and 'e.doc' is not directly available here.
    # It should be called within an exception handler where 'e' is the exception object.
    # For now, I'll keep it as is, assuming it's called correctly elsewhere.
    import xml.etree.ElementTree as ET

    def print_xml_elements(element, indent=0):
        """print all xml elements"""
        # Print the tag and text of the current element
        print(' ' * indent + f"{element.tag}: {element.text}")
        # Recursively print all child elements
        for child in element:
            print_xml_elements(child, indent + 2)
    try:
        # Attempt to parse the XML string
        root = ET.fromstring(xml_string)
        return print_xml_elements(root)
    except Exception as ee:
        # Handle parsing errors
        print(
            "Error returned as xml:",
            xml_string,
            ": XML Parsing Error:" if isinstance(ee, ET.ParseError) else "some XML parse error",
            ee
        )


def get_interpolation_delta(base_delta, dataset_id, lat, lon):
    """Calculate interpolation delta based on dataset resolution and base delta"""
    if base_delta is not None and base_delta != "null":
        return base_delta

    # Use dataset-specific resolution fallbacks
    match dataset_id:
        case s if "cmems_obs-wind_glo_phy_my_l4_0.25deg" in s:
            return 0.25  # 0.25° for 0.25° dataset
        case s if "cmems_obs-wind_glo_phy_my_l4_0.125deg" in s:
            return 0.125  # 0.125° for 0.125° dataset
        case s if "cmems_mod_bal_phy_anfc" in s or "cmems" in s:
            return 0.03 # for 2 km: > 2/111 ≈ 0.018°
        case _:
            return 0.25  # default 0.25°


@hydra.main(config_path="cfg", config_name="base", version_base=None)
def main(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO)
    history_manager = DownloadHistoryManager(Path(cfg.base.history_file))

    # Check if there's a project-specific configuration to use
    # When using projects=project_name/config_name, the project config is available in cfg.projects
    projects_config = cfg.get('projects', None)
    l.info(f"Projects config exists: {projects_config is not None}")
    if projects_config:
        l.info(f"Projects config type: {type(projects_config)}")
        l.info(f"Projects config content: {projects_config}")
        l.info(f"Projects keys: {list(projects_config.keys()) if hasattr(projects_config, 'keys') else 'No keys method'}")

    if projects_config and hasattr(projects_config, 'keys') and len(list(projects_config.keys())) > 0:
        # Use the first project config if available
        project_name = next(iter(projects_config.keys()))
        config_section = projects_config[project_name]
        l.info(
            f"Using project-specific configuration: {project_name}, dir_save: {config_section.dir_save}"
        )
    else:
        project_name = "point_wind"
        config_section = projects_config[project_name]  # Default fallback
        l.warning(
            f"Fallback to default config section: {project_name},  dir_save: {config_section.dir_save}"
        )


    dir_save = Path(cfg.base.local_path) / config_section.dir_save
    if not dir_save.is_dir():
        dir_save.mkdir(parents=True, exist_ok=True)
    l.info(f"downloading CMEMS data to {dir_save}, Date Range: {config_section.date_range}...")

    download_options = {
        "start_datetime": config_section.date_range[0],
        "end_datetime": config_section.date_range[-1],
        "output_directory": str(dir_save),
        "netcdf_compression_level": cfg.copernicus.get("netcdf_compression_level"),
        # "force_download": True  # deprecated
    }

    points = config_section.get("points", [])
    gpx_path = getattr(config_section, 'gpx_path', None)
    if gpx_path:
        gpx_waypoints_re = getattr(config_section, 'gpx_waypoints_re', None)
        gpx_file_path = Path(gpx_path)
        # If the file doesn't exist at the relative path, try from the project root
        # The script is run from the project root, so we should check there first
        if not gpx_file_path.exists():
            # Try relative to the parent directory (project root)
            gpx_file_path = Path(__file__).parent.parent.parent / gpx_path  # Go up to project root (h5toGrid)
        if not gpx_file_path.exists():
            # Try relative to the project root (oceano/get_datasets)
            gpx_file_path = Path(__file__).parent.parent / gpx_path  # oceano/get_datasets level
        points_from_gpx = utils.extract_coordinates_from_gpx(gpx_file_path, gpx_waypoints_re)
        if points_from_gpx:
            points += list(points_from_gpx.values())
    if not points:
        try:
            bbox = config_section.bbox
            download_options.update({
                "minimum_longitude": bbox.lon_min,
                "maximum_longitude": bbox.lon_max,
                "minimum_latitude": bbox.lat_min,
                "maximum_latitude": bbox.lat_max,
            })
        except AttributeError as e:
            raise ValueError(
                "No points or bbox coordinates found in the configuration. Cannot proceed with download."
            ) from e

    dataset_vars = config_section.dataset_vars
    min_depth = config_section.get('depth_min', config_section.get('default_depth_min', 0.5))
    max_depth = config_section.get('depth_max', config_section.get('default_depth_max', 125))

    l.info(f"=== Download {len(dataset_vars)} datasets for {len(points)} points ===")
    paths = []
    for i1_ds, (dataset_id, variables) in enumerate(dataset_vars.items(), start=1):
        l.info(f"{i1_ds}. Processing dataset: {dataset_id}, variables: {variables}")
        download_options.update({
            "dataset_id": dataset_id,
            "variables": list(variables),  # Ensure variables is a list
            }
        )

        # Depth settings (exclude if exist for "wind" variables)
        if any("wind" in v for v in variables):
            download_options.pop("minimum_depth", None)
            download_options.pop("maximum_depth", None)
        else:
            download_options["minimum_depth"] = min_depth
            download_options["maximum_depth"] = max_depth

        for i1_p, point in enumerate(points if points else [None], start=1):
            if point: # Download the extended region data
                # Calculate interpolation delta based on dataset resolution
                delta = get_interpolation_delta(
                    config_section.get("interpolation_delta"), dataset_id, point["lat"], point["lon"]
                )
                l.info(
                    f"{i1_ds}.{i1_p}. Point (lat={point['lat']}, lon={point['lon']}). "
                    f"Download data around with delta={delta}"
                )
                download_options.update({
                    "minimum_longitude": point['lon'] - delta,
                    "maximum_longitude": point['lon'] + delta,
                    "minimum_latitude": point['lat'] - delta,
                    "maximum_latitude": point['lat'] + delta,
                })
                coords = [(point['lat'], point['lon'])] # history_manager.log_download() format
            else:  # Download region data
                coords = [
                    (config_section.bbox.lat_min, config_section.bbox.lon_min),
                    (config_section.bbox.lat_max, config_section.bbox.lon_max),
                ]

            # Create a separate copy of download_options for logging to avoid polluting the main options
            logging_messages = {}
            try:
                result = cm.subset(**download_options)
                # Handle the response - it returns a ResponseSubset object
                if hasattr(result, 'file_path'):
                    file_path = Path(result.file_path)
                    l.info(f"Extended subset saved to {file_path}")
                    path_loaded = str(file_path)
                elif hasattr(result, '__fspath__') or isinstance(result, str):
                    file_path = Path(result)
                    l.info(f"Extended subset saved to {file_path}")
                    path_loaded = str(file_path)
                else:
                    # For other response types, we may need to handle differently
                    path_loaded = str(result)  # Convert to string representation
                    l.info(f"Download result: {path_loaded}")
            except requests.exceptions.JSONDecodeError as e:
                l.error(f"JSONDecodeError during download for {dataset_id}: {e}")
                e_str = None
                if hasattr(e, 'doc'):
                    e_str = extract_error_from_xml(xml_string=e.doc)
                logging_messages["error"] = e_str or str(e)
                path_loaded = None
            except Exception as e:
                path_loaded = None
                l.exception(f"CMEMS download failed: {e} for {coords}: (options: {download_options})")
                logging_messages["error"] = str(e)

            if path_loaded:  # Only proceed if download was successful
                paths.append(path_loaded)
                if point:  # Interpolate to the exact point
                    try:
                        path_interp = utils.interp_to_point(path_loaded, point['lat'], point['lon'])
                        l.info(f"Interpolated data saved to: {path_interp}")
                    except AssertionError as e:
                        msg = {
                            "status": "Interpolation skipped: ",
                            "error": str(e),  # longitude: 19.87192223779857 is not between [19.875008]
                        }
                        logging_messages.update(msg)
                        l.warning("{status} {error}".format_map(msg))
            else:
                l.warning("No file downloaded! Skipping interpolation and logging as 'skipped'.")
                logging_messages.update({
                    "status": "skipped"
                })
            history_manager.log_download(
                dir_save=dir_save,
                coords=coords,
                date_range=config_section.date_range,
                options={**download_options, **logging_messages},
            )


    # print("\n--- Download History ---")
    # for entry in history_manager.get_history():
    #     print(entry)

if __name__ == "__main__":
    main()