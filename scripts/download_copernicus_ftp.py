#!/usr/bin/env python
"""
Download needed NetCDF files from Copernicus NRT FTP using index files located on FTP
"""
from configparser import ConfigParser
from pathlib import Path
from collections import namedtuple
from datetime import datetime
import os
# import warnings
# warnings.filterwarnings("ignore")
from shapely.geometry import box, Point
import ftputil
import pandas as pd

# User configuration
# Load the info defining product/dataset. (In case you want to explore any other In Situ NRT product/dataset just set
# the above definitions accordingly and you will be able to reproduce the analysis we will perform next):
dataset = {
    'host': 'nrt.cmems-du.eu',  # ftp host => nrt.cmems-du.eu for Near Real Time products
    # Names of the In Situ Near Real Time product and dataset available in the above In Situ Near Real Time product:
    # Variant1:
    'product': 'INSITU_BAL_NRT_OBSERVATIONS_013_032',
    'name': 'bal_multiparameter_nrt'
    # Variant2:
    #'product': 'INSITU_BAL_PHYBGCWAV_DISCRETE_MYNRT_013_032',
    #'name': 'cmems_obs-ins_bal_phybgcwav_mynrt_na_irr',
    # not exist: cmems_obs-ins_bal_phybgcwav_mynrt_profile
}

# Bounding box of interest
Axes2d = namedtuple('axes2d', ('x', 'y'))   # x and y are shortcats to longitude and latitude correspondingly
MinMax = namedtuple('MinMax', ('min', 'max'))
bbox = Axes2d(
    MinMax(22, 30),  # longitude limits: (min(x), max(x))
    MinMax(59, 60.5)  # latitude limits: (min(y), max(y))
)

# start/end dates we interested
targeted_range_str = '2020-10-01T00:00:00Z/2023-10-01T00:00:00Z'

# Our working dir to save files
local_path = Path(r'd:\WorkData\BalticSea\_other_data\_model\Copernicus')

targeted_collection = 'monthly'  # latest, monthly, history
# - latest: daily files from platforms (last 30 days of data)
# - monthly: monthly files from platforms (last 5 years of data)
# - history: one file per platform (all platform data)


# Global implicitly used parameters

# Files describing netCDF file collections within the above dataset of the available options:
index_files = ['index_latest.txt', 'index_monthly.txt', 'index_history.txt']
# - network of platforms contributting with files in the above collections
index_platform_file = 'index_platform.txt'

config = ConfigParser()
config.read(os.path.expandvars('%USERPROFILE%\.cmems-copernicus.ini'))  #  C:\Users\{USER}
usr = config['Main']['user']
pas = config['Main']['pwd']


def overlaps_bbox(row, targeted_bbox_polygon):
    """ Checks if a file contains data in the specified area (targeted_bbox_polygon)
    """
    result = False
    try:
        bounding_box = box(*[float(row[c]) for c in [
            'geospatial_lon_min', 'geospatial_lat_min', 'geospatial_lon_max', 'geospatial_lat_max']
            ])
        if targeted_bbox_polygon.intersects(bounding_box):  # check other rules on https://shapely.readthedocs.io/en/stable/manual.html
            result = True
    except Exception as e:
        pass
    return result


def overlaps_last_location(row, targeted_bbox_polygon):
    # Checks if a file has been produced by a platform whose last position is within the specified area (targeted_bbox)
    # see polygon.contains at https://shapely.readthedocs.io/en/stable/manual.html
    result = False
    try:
        location = Point(float(row['last_longitude_observation']), float(row['last_latitude_observation']))
        if targeted_bbox_polygon.contains(location):
            result = True
    except Exception as e:
        print(e)
    return result


Range = namedtuple('Range', ('start', 'end'))
date_format = '%Y-%m-%dT%H:%M:%SZ'

def overlaps_time(df, targeted_range):
    # Checks if a file contains data in the specified time range (targeted_range)
    try:
        # r1 = Range(*targeted_range)
        # r2 = Range(*[datetime.strptime(t, date_format) for t in row[, 'time_coverage_end']]])
        latest_start = df['time_coverage_start'].where(df['time_coverage_start'] > targeted_range[0], targeted_range[0])
        earliest_end = df['time_coverage_end'].where(df['time_coverage_end'] < targeted_range[1], targeted_range[1])
        return latest_start < earliest_end
    except Exception as e:
        print(e)
    return pd.Series(False, index=df.index)


def download_index_files(dataset, local_dir: Path):
    """
    Downloads so called `index files` - a set of files located in the CMEMS FTP server
    useful for exploring the needed product/dataset. Places them in subdir with name of dataset or
    skip if subdir exist.
    Returns path to subdir.
    """
    local_subdir = local_dir / dataset['name']
    if local_subdir.is_dir():
        print(f"Directory {local_subdir}/ exist: will use its files. To reload from FTP delete dir and run again.")
    else:
        # Provides
        indexes = index_files + [index_platform_file]
        with ftputil.FTPHost(dataset['host'], usr, pas) as ftp_host:  # connect to CMEMS FTP
            print(f"Downloading to {local_subdir}/: ")
            for index in indexes:
                print(index, end=', ')
                remote_file= '/'.join(['Core', dataset['product'], dataset['name'], index])
                ftp_host.download(remote_file, local_subdir / index)  # remote, local
        print('Ok>')
    return local_subdir


def read_index_file(path2file: Path, targeted_bbox_polygon):
    """
    Index files reader. Load as pandas dataframe the file in the provided path.
    In order to load the information contained in each of the files we will use the next function.
    """
    print('Loading info from', path2file.name, end='...')
    def replace_comma_with_space(words):
        """Correct line splitting if they are splitted by ', ': they must be splitted only if no space after ','
        """
        source_str = ','.join(words).replace(', ', ' & ')
        out = source_str.split(',')
        if out == words:
            print('Ignoring row with error (can not correct):\n', out)
            out = None
        else:
            print('Replacing comma with space in\n', out)
        return out

    read_csv_args = {
        'skiprows': 5,
        'delimiter': ',',
        'engine': 'python',
        'on_bad_lines': replace_comma_with_space
        }

    # if path2file.name == index_platform_file:
    #
    # else:
    #     read_csv_args = {
    #         'skiprows': 5,
    #         'delimiter': ',',
    #         'on_bad_lines': 'warn'
    #         }

    if targeted_bbox_polygon is None:
        result = pd.read_csv(path2file, **read_csv_args)
        try:
            result = result.rename(columns={'provider_edmo_code': 'institution_edmo_code'})
        except Exception as e:
            print(e)
        print(f' {len(result)} rows')
    else:
        raw_index_info = []
        chunks = pd.read_csv(path2file, **read_csv_args, chunksize=1000)
        n_rows = 0
        n_use = 0
        for chunk in chunks:
            chunk['overlaps_spatial'] = chunk.apply(
                overlaps_bbox, targeted_bbox_polygon=targeted_bbox_polygon, axis=1
                )
            chunk_use = chunk[chunk['overlaps_spatial']]
            n_rows += len(chunk)
            raw_index_info.append(chunk_use)
        result = pd.concat(raw_index_info).drop(columns=['overlaps_spatial'])
        print(f' {len(result)}/{n_rows} rows selected')
    return result


def read_index_files(indexes_dir: Path, targeted_bbox_polygon):
    """ Loads and merges in a single entity all the information contained on each file descriptor of a given dataset
    """
    # 1) Loading the index platform info as dataframe
    indexPlatform = read_index_file(indexes_dir / index_platform_file, None)
    indexPlatform.rename(columns={indexPlatform.columns[0]: 'platform_code'}, inplace = True)
    indexPlatform = indexPlatform.drop_duplicates(subset='platform_code', keep="first")
    # 2) Loading the index files info as dataframes
    netcdf_collections = []
    for filename in index_files:
        indexFile = read_index_file(indexes_dir / filename, targeted_bbox_polygon)
        netcdf_collections.append(indexFile)
    netcdf_collections = pd.concat(netcdf_collections)
    # 3) creating new columns: derived info
    netcdf_collections['netcdf'] = netcdf_collections['file_name'].str.split('/').str[-1]
    netcdf_collections[['file_type', 'file_type', 'data_type', 'platform_code']] = (
        netcdf_collections['netcdf'].str.split('.', 1).str[0].str.split('_', 3, expand=True)
    )
    netcdf_collections['platform_code'] = netcdf_collections['platform_code'].str.split('_', 1).str[0]
    # 4) Merging the information of all files
    headers = ['platform_code','wmo_platform_code', 'institution_edmo_code',
               'last_latitude_observation', 'last_longitude_observation', 'last_date_observation'
               ]
    result = pd.merge(netcdf_collections, indexPlatform[headers], on='platform_code')
    print('Ok>')
    return result


########################################################################################################################
if __name__ == '__main__':

    # Target area
    targeted_bbox_polygon = box(bbox.x.min, bbox.y.min, bbox.x.max, bbox.y.max)
    targeted_range = [datetime.strptime(t, date_format) for t in targeted_range_str.split('/')]
    if False:  # Let's see it on a map
        import folium
        from folium import plugins
        folium_polygon = folium.vector_layers.Polygon(locations=[(y,x) for (x,y) in targeted_bbox_polygon.exterior.coords])
        m = folium.Map(location=[bbox.y.min, bbox.x.min], zoom_start=6, min_zoom=4)
        m.add_child(folium_polygon)
        m.fit_bounds(folium_polygon.get_bounds())
        m

    # Copernicus Database collection overview
    # Download the most recent version of the index files:
    indexes_dir = download_index_files(dataset, local_path / '_FTP_NRT_indexes')
    # Load indexes into a dataframe:
    info = read_index_files(indexes_dir, targeted_bbox_polygon)
    info['time_coverage_start'] = info['time_coverage_start'].astype('M8[s]')
    info['time_coverage_end'] = info['time_coverage_end'].astype('M8[s]')
    info = info[info['parameters'].str.split(' ').apply(lambda x: ('PSAL' in x) or ('CNDC' in x))]  # Practical salinity / Electrical conductivity - see codes at Copernicus Marine in situ TAC - physical parameters list https://archimer.ifremer.fr/doc/00422/53381/
    # info = info[subset['parameters'].str.split(' ').apply(lambda x: 'PSAL' in x)]  # same result as previous

    for collection in ['latest', 'monthly', 'history']:
        subset = info[info['file_name'].str.contains(collection)]
        subset = subset[overlaps_time(subset, targeted_range)]
        subset = subset[info.apply(overlaps_last_location,
                                   targeted_bbox_polygon=targeted_bbox_polygon,
                                   axis=1
                                   )]
        len_collection = len(subset)
        if not len_collection:
            print(collection, '- no data in our spatial boundary')
            continue
        subset = info[info['file_name'].str.contains(collection)]
        print(collection, f'({len_collection}) - time coverage:', subset['time_coverage_start'].min(), '-', subset['time_coverage_end'].max())

    if False:
        info.transpose()  # Shows information just loaded above

    targeted_collection = 'history'
    subset = info[
        info['file_name'].str.contains(targeted_collection) & overlaps_time(info, targeted_range)
        # info.apply(
        #     lambda row: overlaps_time(row, targeted_range),
        #     #  & overlaps_last_location(row, targeted_bbox_polygon),
        #     axis=1
        #     )

    ]
    # subset.transpose()
    subset = subset[info.apply(overlaps_last_location,
            targeted_bbox_polygon=targeted_bbox_polygon,
            axis=1
            )]
    if len(subset):
        subset['data_type'] = subset['data_type'].map({
            'BO': 'Bottle',
            'BA': 'Data from Bathy messages on GTS',
            'DB': 'Drifting buoys',
            'DC': 'Drifting buoy reporting calculated sea water current',
            'FB': 'FerryBox',
            'MO': 'Fixed buoys or moorings',
            'TG': 'Tide gauges',
            'GL': 'Gliders',
            'ML': 'Mini logger',
            'CT': 'CTD profiles',
            'PF': 'Profiling floats vertical profiles',
            'RE': 'Recopesca',
            'RF': 'River flows',
            'SF': 'Scanfish profiles',
            'TS': 'Thermosalinograph data',
            'XB': 'XBT or XCTD profiles',
            'TE': 'Data from TESAC messages on GTS',
            'SM': 'Sea-mammals',
            'HF': 'High Frequency Radar',
            'SD': 'Saildrone',
            'VA': 'Vessel mounted ADCP',
            'XX': 'Unknown'
            })
        subset.to_csv(indexes_dir.with_name(f"{dataset['name']}-{targeted_collection}.csv"))
        # Download data files
        try:
            with ftputil.FTPHost(dataset['host'], usr, pas) as ftp_host:  # connect to CMEMS FTP
                print(f"Downloading {len(subset)} files to {local_path}: ")
                for i, remote_file in enumerate(subset['file_name']):
                    remote_file_path = Path(remote_file)
                    remote_file_name = remote_file_path.name
                    print(f'{i}.', remote_file_name)
                    remote_file = '/'.join(remote_file_path.parts[2:])
                    ftp_host.download(remote_file, local_path / remote_file_name)  # remote, local
                print('Ok>')
        except Exception as e:
            print(e)

    if False:
        # Next can be used to create the html table overview
        # Platforms
        # Any Situ NRT products/datasets collections are composed by files reported by a network of platforms located in the area.

        # Let's run the next cell to know how many platforms are contributting to this product/dataset collection with files:
        len(targeted_collection_info.groupby('platform_code').groups)

        # Let's get a closer look to this list of platforms:
        data_platforms = [{
            'code': code,
            'provider(s)': files['institution_edmo_code'].iloc[0],
            'files': {
                'nfiles': len(files),
                'feature(s)': [{
                    'code': ftype,
                    'nfiles': len(files.groupby('file_type').get_group(ftype)),
                    'sources(s)': [{
                        'code': dtype,
                        'nfiles': len(files.groupby('file_type').get_group(ftype).groupby('data_type').get_group(dtype)),
                        'parameters': [param for param in sum([
                            files.groupby('file_type').get_group(ftype).groupby('data_type').get_group(dtype)['parameters']
                                .unique().tolist()], [])[0].split(' ') if
                                       '_' not in param and param not in ['DEPH', 'PRES', '']]
                        } for dtype in files[files['file_type'] == ftype]['data_type'].unique().tolist()]
                    } for ftype in files['file_type'].unique().tolist()]
                }
            } for code, files in targeted_collection_info.groupby(['platform_code'])]
        # Let's see one in particular
        platform_code = '13001'
        selection = [platform for platform in data_platforms if platform['code'] == platform_code]