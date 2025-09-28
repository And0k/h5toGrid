from collections import namedtuple
from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import ftputil
import pandas as pd
from shapely.geometry import box, Point

import hydra
from omegaconf import DictConfig, OmegaConf

from .manager import DownloadHistoryManager

l = logging.getLogger(__name__)

Axes2d = namedtuple('axes2d', ('x', 'y'))
MinMax = namedtuple('MinMax', ('min', 'max'))
Range = namedtuple('Range', ('start', 'end'))

# Constants for index files
INDEX_FILES = [
    'bal_multiparameter_nrt_index.csv',
    'bal_multiparameter_nrt_monthly_index.csv',
    'bal_multiparameter_nrt_yearly_index.csv',
]
INDEX_PLATFORM_FILE = 'bal_multiparameter_nrt_platform.csv'


def overlaps_bbox(row, targeted_bbox_polygon):
    """ Checks if a file contains data in the specified area (targeted_bbox_polygon)
    """
    result = False
    try:
        bounding_box = box(
            float(row['geospatial_lon_min']), float(row['geospatial_lat_min']),
            float(row['geospatial_lon_max']), float(row['geospatial_lat_max'])
        )
        if targeted_bbox_polygon.intersects(bounding_box):
            result = True
    except Exception as e:
        l.warning(f"Error checking bbox overlap: {e}")
    return result


def overlaps_last_location(row, targeted_bbox_polygon):
    """ Checks if a file has been produced by a platform whose last position is within the specified area.
    """
    result = False
    try:
        location = Point(float(row['last_longitude_observation']), float(row['last_latitude_observation']))
        if targeted_bbox_polygon.contains(location):
            result = True
    except Exception as e:
        l.warning(f"Error checking last location overlap: {e}")
    return result


def overlaps_time(df, targeted_range: List[datetime]):
    """ Checks if a file contains data in the specified time range (targeted_range)
    """
    try:
        latest_start = df['time_coverage_start'].where(df['time_coverage_start'] > targeted_range[0], targeted_range[0])
        earliest_end = df['time_coverage_end'].where(df['time_coverage_end'] < targeted_range[1], targeted_range[1])
        return latest_start < earliest_end
    except Exception as e:
        l.warning(f"Error checking time overlap: {e}")
    return pd.Series(False, index=df.index)


def download_index_files(dataset_info: Dict[str, str], local_dir: Path, usr: str, pas: str):
    """
    Downloads index files from the CMEMS FTP server.
    """
    local_subdir = local_dir / dataset_info['name']
    if local_subdir.is_dir():
        l.info(f"Directory {local_subdir}/ exists: will use its files. To reload from FTP delete dir and run again.")
    else:
        local_subdir.mkdir(parents=True, exist_ok=True)
        indexes = INDEX_FILES + [INDEX_PLATFORM_FILE]
        with ftputil.FTPHost(dataset_info['host'], usr, pas) as ftp_host:
            l.info(f"Downloading to {local_subdir}/: ")
            for index in indexes:
                l.info(f"  {index}")
                remote_file = '/'.join(['Core', dataset_info['product'], dataset_info['name'], index])
                ftp_host.download(remote_file, local_subdir / index)
        l.info(f"Download of index files complete.")
    return local_subdir


def read_index_file(path2file: Path, targeted_bbox_polygon: Optional[Any]):
    """
    Index files reader. Load as pandas dataframe the file in the provided path.
    """
    l.info(f"Loading info from {path2file.name}...")

    def replace_comma_with_space(words):
        source_str = ','.join(words).replace(', ', ' & ')
        out = source_str.split(',')
        if out == words:
            l.warning(f"Ignoring row with error (can not correct):\n {out}")
            out = None
        else:
            l.info(f"Replacing comma with space in\n {out}")
        return out

    read_csv_args = {
        'skiprows': 5,
        'delimiter': ',',
        'engine': 'python',
        'on_bad_lines': replace_comma_with_space
    }

    if targeted_bbox_polygon is None:
        result = pd.read_csv(path2file, **read_csv_args)
        try:
            result = result.rename(columns={'provider_edmo_code': 'institution_edmo_code'})
        except KeyError:
            pass # Column might not exist, which is fine
        l.info(f" {len(result)} rows loaded.")
    else:
        raw_index_info = []
        chunks = pd.read_csv(path2file, **read_csv_args, chunksize=1000)
        n_rows = 0
        for chunk in chunks:
            chunk['overlaps_spatial'] = chunk.apply(
                overlaps_bbox, targeted_bbox_polygon=targeted_bbox_polygon, axis=1
            )
            chunk_use = chunk[chunk['overlaps_spatial']]
            n_rows += len(chunk)
            raw_index_info.append(chunk_use)
        result = pd.concat(raw_index_info).drop(columns=['overlaps_spatial'])
        l.info(f" {len(result)}/{n_rows} rows selected.")
    return result


def read_index_files(indexes_dir: Path, targeted_bbox_polygon: Any):
    """ Loads and merges in a single entity all the information contained on each file descriptor of a given dataset
    """
    indexPlatform = read_index_file(indexes_dir / INDEX_PLATFORM_FILE, None)
    indexPlatform.rename(columns={indexPlatform.columns[0]: 'platform_code'}, inplace=True)
    indexPlatform = indexPlatform.drop_duplicates(subset='platform_code', keep="first")

    netcdf_collections = []
    for filename in INDEX_FILES:
        indexFile = read_index_file(indexes_dir / filename, targeted_bbox_polygon)
        netcdf_collections.append(indexFile)
    netcdf_collections = pd.concat(netcdf_collections)

    netcdf_collections['netcdf'] = netcdf_collections['file_name'].str.split('/').str[-1]
    # Split 'netcdf' column to extract file_type, data_type, and platform_code
    # Using regex split to handle cases where parts might be missing or have different delimiters
    split_netcdf = netcdf_collections['netcdf'].str.extract(r'([A-Z]{2})_([A-Z]{2})_([A-Z0-9]+)_([A-Z0-9]+)')
    netcdf_collections['file_type'] = split_netcdf[0]
    netcdf_collections['data_type'] = split_netcdf[1]
    netcdf_collections['platform_code'] = split_netcdf[3] # Assuming platform_code is the 4th part

    # Further refine platform_code if it contains additional parts after '_'
    netcdf_collections['platform_code'] = netcdf_collections['platform_code'].apply(lambda x: x.split('_')[0] if isinstance(x, str) else x)

    headers = ['platform_code', 'wmo_platform_code', 'institution_edmo_code',
               'last_latitude_observation', 'last_longitude_observation', 'last_date_observation']
    result = pd.merge(netcdf_collections, indexPlatform[headers], on='platform_code')
    l.info(f"Index files merged.")
    return result


def download_copernicus_ftp_data(
    cfg: DictConfig,
    local_path: Path,
) -> Tuple[Optional[Path], Optional[Dict[str, Any]]]:
    """
    Main function to download Copernicus data via FTP.
    """
    # CMEMS credentials are handled by the copernicusmarine library's own configuration.
    # For ftputil, we need to get them from the config.
    # For now, hardcode or assume they are set in environment variables for ftputil.
    # In a real scenario, you might add them to the Hydra config or use a secure method.
    # For this task, we'll assume they are handled by the environment or a default.

    dataset_info = cfg.copernicus.ftp.dataset_info
    targeted_bbox = cfg.copernicus.ftp.default_bbox
    targeted_date_range_str = cfg.copernicus.ftp.default_date_range
    targeted_collection = cfg.copernicus.ftp.default_collection
    parameters_filter = list(cfg.copernicus.ftp.parameters_filter) if cfg.copernicus.ftp.parameters_filter else None

    # Placeholder for FTP credentials if not using copernicusmarine client for FTP
    # In a real application, these would come from a secure source or Hydra config
    # For now, using dummy values or expecting them to be set externally for ftputil
    usr = cfg.copernicus.ftp.username
    pas = cfg.copernicus.ftp.password

    targeted_bbox_polygon = box(
        targeted_bbox['lon_min'], targeted_bbox['lat_min'],
        targeted_bbox['lon_max'], targeted_bbox['lat_max']
    )
    targeted_range = [datetime.strptime(t, '%Y-%m-%dT%H:%M:%SZ') for t in targeted_date_range_str]

    indexes_dir = download_index_files(dataset_info, local_path / '_FTP_NRT_indexes', usr, pas)
    info = read_index_files(indexes_dir, targeted_bbox_polygon)
    info['time_coverage_start'] = info['time_coverage_start'].astype('M8[s]')
    info['time_coverage_end'] = info['time_coverage_end'].astype('M8[s]')

    if parameters_filter:
        info = info[info['parameters'].str.split(' ').apply(
            lambda x: any(p in x for p in parameters_filter)
        )]

    subset = info[info['file_name'].str.contains(targeted_collection)]
    subset = subset[overlaps_time(subset, targeted_range)]
    subset = subset[subset.apply(overlaps_last_location,
                                 targeted_bbox_polygon=targeted_bbox_polygon,
                                 axis=1)]

    len_subset = len(subset)
    download_options = {
        'dataset_info': OmegaConf.to_container(dataset_info, resolve=True),
        'targeted_bbox': OmegaConf.to_container(targeted_bbox, resolve=True),
        'targeted_date_range_str': targeted_date_range_str,
        'targeted_collection': targeted_collection,
        'parameters_filter': parameters_filter
    }

    if not len_subset:
        l.info(f"No data found for collection '{targeted_collection}' in the specified spatial and temporal boundary.")
        download_options['status'] = 'no_data_found'
        return local_path, download_options

    l.info(f"Downloading {len_subset} files from collection '{targeted_collection}' to {local_path}:")

    try:
        with ftputil.FTPHost(dataset_info['host'], usr, pas) as ftp_host:
            for i, remote_file in enumerate(subset['file_name']):
                remote_file_path = Path(remote_file)
                remote_file_name = remote_file_path.name
                l.info(f"  {i+1}/{len_subset}. {remote_file_name}")
                remote_file_for_ftp = '/'.join(remote_file_path.parts[2:])
                ftp_host.download(remote_file_for_ftp, local_path / remote_file_name)
        l.info(f"All {len_subset} files downloaded successfully.")
        return local_path, download_options
    except Exception as e:
        l.error(f"Error during FTP download: {e}")
        raise


@hydra.main(config_path="cfg", config_name="base", version_base=None)
def main(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO)
    history_manager = DownloadHistoryManager(Path(cfg.base.history_file))

    local_save_path = Path(cfg.base.local_path) / "CMEMS_FTP_Data" # Specific subdirectory for FTP downloads

    try:
        result = download_copernicus_ftp_data(
            cfg=cfg,
            local_path=local_save_path,
        )
        if result and result[0] is not None and result[1] is not None:
            download_dir, download_opts = result
            if download_opts is not None: # Ensure download_opts is not None
                history_manager.log_download(
                    dir_save=download_dir,
                    lat=f"[{download_opts['targeted_bbox']['lat_min']}-{download_opts['targeted_bbox']['lat_max']}]",
                    lon=f"[{download_opts['targeted_bbox']['lon_min']}-{download_opts['targeted_bbox']['lon_max']}]",
                    date_range=download_opts['targeted_date_range_str'],
                    options=download_opts
                )
            else:
                l.info("No data downloaded for the specified criteria (download_opts was None).")
                history_manager.log_download(
                    dir_save=local_save_path,
                    lat="N/A",
                    lon="N/A",
                    date_range=["N/A"],
                    options={'status': 'no_data_found'}
                )
        else:
            l.info("No data downloaded for the specified criteria.")
            history_manager.log_download(
                dir_save=local_save_path,
                lat="N/A",
                lon="N/A",
                date_range=["N/A"],
                options={'status': 'no_data_found'}
            )
    except Exception as e:
        l.error(f"Failed to download Copernicus FTP data: {e}")
        history_manager.log_download(
            dir_save=local_save_path,
            lat="N/A",
            lon="N/A",
            date_range=["N/A"],
            options={'status': 'failed', 'error': str(e)}
        )

    print("\n--- Copernicus FTP Download History ---")
    for entry in history_manager.get_history():
        print(entry)


if __name__ == '__main__':
    main()