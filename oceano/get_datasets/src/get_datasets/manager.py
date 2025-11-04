import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Tuple

class DownloadHistoryManager:
    """
    Manages the history of data download operations.
    History is stored in a human-readable, line-delimited JSON (JSONL) file.
    """
    def __init__(self, history_file: Path = Path("scripts/downloading/history/download_history.jsonl")):
        self.history_file = history_file
        self.history_entries: List[Dict[str, Any]] = []
        self._load_history()

    def _load_history(self):
        """Loads the download history from the JSONL file."""
        self.history_entries = []
        if self.history_file.exists():
            with open(self.history_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        # Convert run_date string back to datetime object
                        if 'run_date' in entry and isinstance(entry['run_date'], str):
                            entry['run_date'] = datetime.fromisoformat(entry['run_date'])
                        self.history_entries.append(entry)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Could not decode JSON from history file line: {line.strip()} - {e}")
        print(f"Loaded {len(self.history_entries)} history entries.")

    def _save_history(self):
        """Saves the current download history to the JSONL file."""
        self.history_file.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        with open(self.history_file, 'w', encoding='utf-8') as f:
            for entry in self.history_entries:
                # Convert datetime objects to ISO format strings for JSON serialization
                serializable_entry = entry.copy()
                if 'run_date' in serializable_entry and isinstance(serializable_entry['run_date'], datetime):
                    serializable_entry['run_date'] = serializable_entry['run_date'].isoformat()
                f.write(json.dumps(serializable_entry) + '\n')
        print(f"Saved {len(self.history_entries)} history entries to {self.history_file}.")

    def log_download(
        self,
        dir_save: Optional[Path],
        coords: List[Tuple[float, float]] = None,
        date_range: List[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        """
        Logs a new download operation.

        :param dir_save: Directory where the data was saved.
        :param coords: List of coordinate tuples [(lat, lon), ...] of the download point/region.
        :param date_range: List of two strings [start_date, end_date] for the download.
        :param options: Dictionary of additional options used for the download (e.g., dataset_id, variables).
        """
        run_date = datetime.now()
        new_entry = {
            'run_date': run_date,
            'dir_save': str(dir_save) if dir_save else None,
            'coords': coords,
            'date_range_start': date_range[0] if date_range else None,
            'date_range_end': date_range[1] if date_range else None,
            'options': options if options is not None else {}
        }
        self.history_entries.append(new_entry)
        self._save_history()
        print(f"Download logged: {new_entry}")

    def get_history(self) -> List[Dict[str, Any]]:
        """Returns the entire download history as a list of dictionaries."""
        return self.history_entries

    def find_downloads(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Finds download entries matching specified criteria.

        :param kwargs: Key-value pairs to filter the history.
        :return: List of matching download entries.
        """
        filtered_results = []
        for entry in self.history_entries:
            match = True
            for key, value in kwargs.items():
                if key == 'dir_save' and isinstance(value, Path):
                    if entry.get(key) != str(value):
                        match = False
                        break
                elif entry.get(key) != value:
                    match = False
                    break
            if match:
                filtered_results.append(entry)
        return filtered_results

if __name__ == "__main__":
    # Example Usage:
    history_manager = DownloadHistoryManager()

    # Log a sample download
    print("Logging first download...")
    history_manager.log_download(
        dir_save=Path(r"D:\WorkData\BalticSea\test_downloads\cmems_wind"),
        coords=[(55.1, 19.8)],
        date_range=['2024-01-01', '2024-01-02'],
        options={'dataset_id': 'cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H', 'variables': ['eastward_wind']}
    )

    # Log another sample download
    print("\nLogging second download...")
    history_manager.log_download(
        dir_save=Path(r"D:\WorkData\BalticSea\test_downloads\ncep_data"),
        coords=[(54.5, 20.0)],
        date_range=['2023-10-01', '2023-10-05'],
        options={'dataset_id': 'NCEP_CFSv2', 'variables': ['U_GRD_L103', 'V_GRD_L103'], 'interpolation': 'nearest'}
    )

    # Get and print full history
    print("\nFull Download History:")
    for entry in history_manager.get_history():
        print(entry)

    # Find specific downloads
    print("\nFinding downloads for coords containing (55.1, 19.8):")
    for entry in history_manager.find_downloads(coords=[(55.1, 19.8)]):
        print(entry)

    print("\nFinding downloads for dir_save=D:\\WorkData\\BalticSea\\test_downloads\\ncep_data:")
    for entry in history_manager.find_downloads(dir_save=Path(r"D:\WorkData\BalticSea\test_downloads\ncep_data")):
        print(entry)