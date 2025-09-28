"""
Utility Functions
"""

from pathlib import Path
from runpy import run_module
from typing import Any, Callable, Iterable, Mapping, Optional, Tuple, Sequence, Union
from time import strptime
from datetime import datetime
from calendar import monthrange
import numpy as np
import h5py

NaT = np.datetime64("NaT")

class DumbMapping(dict):
    def __getitem__(self, key):
        return self.get(key, key)

# Time Range Processing Utilities

def process_time_range(time_range, have_timedelta=False):
    """
    Process and normalize time range inputs
    """
    if any(np.isfinite(time_range)):
        b_nat = np.isnat(time_range)
        if not all(b_nat):
            max_time_span_s_strings = [f"{t}" for t in time_range]
            print("Loading interval", max_time_span_s_strings)
            time_range = np.array(time_range, f"{'m' if have_timedelta else 'M'}8[ns]")
            if not have_timedelta:
                # Allow find maximum if NaT using np.searchsorted() correctly
                if ~np.isfinite(time_range[-1]):
                    time_range[-1] = np.datetime64(np.iinfo(np.int64).max, "ns")
        else:
            time_range = []
    else:
        time_range = []
    return time_range


def search_time_range_indexes(index, time_range, raw_time_shift_s, raw_time_units="ns"):
    """
    Find integer index slice from specified time range in HDF5 data
    """
    if any(np.isfinite(time_range)):
        have_timedelta = isinstance(time_range[0], np.timedelta64)
        if have_timedelta:
            raw_search_add = index[-1]
        else:
            if raw_time_shift_s:
                to_raw_time_units = np.timedelta64(1, raw_time_units).astype("m8[ns]").astype(int).item()
                raw_search_add = -to_raw_time_units * raw_time_shift_s
            else:
                raw_search_add = 0
        if np.isfinite(time_range[-1]):
            time_range_raw_search = np.array(time_range).astype(np.int64)
            i_range = np.fmin(
                np.searchsorted(index, time_range_raw_search + raw_search_add), len(index) - 1
            ).tolist()
        else:
            time_range_raw_search = np.array(time_range[0]).astype(np.int64)
            i_range = [np.searchsorted(index, time_range_raw_search + raw_search_add).item(), len(index) - 1]
    else:
        i_range = [0, -1]
    return i_range


# HDF5 File Operations Utilities

def validate_hdf5_columns(hdf5_group, expected_columns):
    """
    Validate that required columns exist in HDF5 group
    """
    if isinstance(hdf5_group, h5py.Group):
        existed_cols = set()
        for k, val in hdf5_group.items():
            if isinstance(val, h5py.Group):
                for k1, val1 in val.items():
                    existed_cols.add(f"{k}/{k1}")
            else:
                existed_cols.add(f"{k}")
        return existed_cols & set(expected_columns)
    else:
        return set(hdf5_group.dtype.fields.keys()) & set(expected_columns)


def create_column_mapping(cols_namemap, grp_d, device_ids):
    """
    Create standardized column mappings for HDF5 data loading
    """
    if cols_namemap is None:
        cols_namemap = {grp: DumbMapping() for grp in grp_d}
    else:
        cols_namemap = {
            grp: {s: s for s in map_or_seq} if not isinstance(map_or_seq, Mapping) else map_or_seq
            for grp, map_or_seq in cols_namemap.items()
        }
    return cols_namemap


# Dataset Tagging Utilities


def create_dataset_namemap(grp_dev, cols_namemap, grp_d_rename_funs, device_ids):
    """
    Create name mapping for imported datasets
    """
    namemap = {}
    for device_id in device_ids:
        for grp, t in grp_dev[device_id].items():
            # Implementation for creating name mappings
            pass
    return namemap


# Column Processing Utilities

def process_column_names(cols_namemap, existed_devs, device_id, grp):
    """
    Process column names according to mapping specifications
    """
    if not cols_namemap[grp]:
        return existed_devs[device_id][grp]
    else:
        return set(existed_devs[device_id][grp]).intersection(cols_namemap[grp])


def apply_column_renames(cols_namemap, grp_d_rename_funs, grp, col_name, device_id):
    """
    Apply renaming functions to column names
    """
    if grp in grp_d_rename_funs and col_name in cols_namemap[grp]:
        return grp_d_rename_funs[grp](cols_namemap[grp][col_name], device_id)
    return col_name
