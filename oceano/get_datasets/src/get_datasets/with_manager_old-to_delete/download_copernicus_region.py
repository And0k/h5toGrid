import logging
from pathlib import Path
import re
from typing import List, Tuple, Dict, Any, Optional
import requests # Added back requests import

import hydra
from omegaconf import DictConfig, OmegaConf

from .manager import DownloadHistoryManager

l = logging.getLogger(__name__)

try:
    import copernicusmarine as cm
except ImportError:
    l.warning("copernicusmarine library not found. Some download functionalities may be unavailable.")
    cm = None # Set to None if import fails

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
        print_xml_elements(root)
    except Exception as ee:
        # Handle parsing errors
        print(
            "Error returned as xml:",
            xml_string,
            ": XML Parsing Error:" if isinstance(ee, ET.ParseError) else "some XML parse error",
            ee
        )

@hydra.main(config_path="cfg", config_name="base", version_base=None)
def main(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO)
    history_manager = DownloadHistoryManager(Path(cfg.base.history_file))

    # The specific project configuration is now composed by Hydra
    # and available directly under cfg.copernicus.region
    dir_save = Path(cfg.base.local_path) / cfg.copernicus.region.dir_save
    date_range = cfg.copernicus.region.date_range
    bbox = cfg.copernicus.region.bbox
    dataset_vars = cfg.copernicus.region.dataset_vars
    min_depth = cfg.copernicus.region.get('depth_min', cfg.copernicus.region.default_depth_min)
    max_depth = cfg.copernicus.region.get('depth_max', cfg.copernicus.region.default_depth_max)
    netcdf_compression_level = cfg.copernicus.region.get('netcdf_compression_level', cfg.copernicus.region.netcdf_compression_level)

    if not dir_save.is_dir():
        dir_save.mkdir(parents=True, exist_ok=True)
    l.info(f"Downloading CMEMS data to {dir_save}...")

    err = None
    paths = []
    for dataset, vars in dataset_vars.items():
        l.info(f"Processing dataset: {dataset}, variables: {vars}")
        subset_params = {
            "dataset_id": dataset,
            "variables": list(vars), # Ensure variables is a list
            "minimum_longitude": bbox.lon_min,
            "maximum_longitude": bbox.lon_max,
            "minimum_latitude": bbox.lat_min,
            "maximum_latitude": bbox.lat_max,
            "start_datetime": date_range[0],
            "end_datetime": date_range[-1],
            "output_directory": str(dir_save),
            "netcdf_compression_level": netcdf_compression_level,
            "force_download": True,
        }

        # Exclude depth settings for "wind" variables
        if any("wind" in v for v in vars):
            subset_params.pop("minimum_depth", None)
            subset_params.pop("maximum_depth", None)
        else:
            subset_params["minimum_depth"] = min_depth
            subset_params["maximum_depth"] = max_depth

        try:
            if cm:
                p = cm.subset(**subset_params)
                paths.append(p)
                l.info(f"Downloaded {dataset} to {p}")
                history_manager.log_download(
                    dir_save=dir_save,
                    coords=[
                        (bbox.lat_min, bbox.lon_min),
                        (bbox.lat_max, bbox.lon_max)
                    ],
                    date_range=date_range,
                    options={**subset_params, 'bbox': OmegaConf.to_container(bbox), 'status': 'success'}
                )
            else:
                l.warning(f"Skipping CMEMS download for {dataset}: copernicusmarine library not available.")
                history_manager.log_download(
                    dir_save=dir_save,
                    coords=[
                        (bbox.lat_min, bbox.lon_min),
                        (bbox.lat_max, bbox.lon_max)
                    ],
                    date_range=date_range,
                    options={**subset_params, 'bbox': OmegaConf.to_container(bbox), 'status': 'skipped', 'reason': 'copernicusmarine library not available'}
                )

        except requests.exceptions.JSONDecodeError as e:
            err = e
            l.error(f"JSONDecodeError during download for {dataset}: {e}")
            history_manager.log_download(
                dir_save=dir_save,
                coords=[
                    (bbox.lat_min, bbox.lon_min),
                    (bbox.lat_max, bbox.lon_max)
                ],
                date_range=date_range,
                options={**subset_params, 'bbox': OmegaConf.to_container(bbox), 'status': 'failed', 'error': str(e)}
            )
        except Exception as e:
            l.error(f"Error during download for {dataset}: {e}")
            history_manager.log_download(
                dir_save=dir_save,
                coords=[
                    (bbox.lat_min, bbox.lon_min),
                    (bbox.lat_max, bbox.lon_max)
                ],
                date_range=date_range,
                options={**subset_params, 'bbox': OmegaConf.to_container(bbox), 'status': 'failed', 'error': str(e)}
            )

    print("\n--- Download History ---")
    for entry in history_manager.get_history():
        print(entry)

if __name__ == "__main__":
    main()