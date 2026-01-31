#!/usr/bin/env python3
"""
MET File Generator for Marine Expedition Data

Creates *.met description files per VNIIGMI-MCD methodological guidelines.

REQUIREMENTS:
- met_config.yaml in same directory
- *POS.csv mandatory
- VNIIGMI-MCD data files (*.csv with ";" separator)
- execution environment according to requirements.txt
"""

import re
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple
import pandas as pd
from ruamel.yaml import YAML
from colorlog import ColoredFormatter
import questionary
from questionary import Style

CONFIG_FILE = Path(__file__).with_name("met_config.yaml")
logger = None

CUSTOM_STYLE = Style([
    ("qmark", "fg:#673ab7 bold"),
    ("question", "bold"),
    ("answer", "fg:#2196f3 bold"),
    ("pointer", "fg:#673ab7 bold"),
    ("highlighted", "fg:#673ab7 bold"),
    ("selected", "fg:#2196f3"),
])


def ask(questionary_func, **kwargs):
    """
    General wrapper for questionary functions with style=CUSTOM_STYLE as default.

    Args:
        questionary_func: questionary function (text, select, confirm, etc.)
        **kwargs: arguments to pass to questionary function

    Returns:
        Result of questionary.ask()
    """
    # Set default style if not provided
    if "style" not in kwargs:
        kwargs["style"] = CUSTOM_STYLE

    return questionary_func(**kwargs).ask()


def setup_logging(log_file: Path) -> None:
    """Setup colored logging to console and file."""
    global logger
    logger = logging.getLogger("METGenerator")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
        datefmt=None,
        reset=True,
        log_colors={"DEBUG": "cyan", "INFO": "green", "WARNING": "yellow", "ERROR": "red"},
    )
    console_handler.setFormatter(console_formatter)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)-8s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


def clean_input(text: str) -> str:
    """Clean user input."""
    if not text:
        return text
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip(".,;: ")


def validate_cp1251(text: str) -> Tuple[bool, str]:
    """
    Validate text can be encoded to Windows-1251.

    Returns: (is_valid, error_message)
    """
    try:
        text.encode("windows-1251")
        return (True, "")
    except UnicodeEncodeError as e:
        char = text[e.start]
        return (False, f"Символ '{char}' (U+{ord(char):04X}) не поддерживается Windows-1251")


def is_calculated_parameter(param_info: Dict) -> bool:
    """Check if parameter is calculated (not measured)."""
    calc_note = param_info.get("calculation_note", "")
    return calc_note.lower().startswith("рассчитывается")


class ConfigManager:
    """Configuration manager with structure-based access."""

    def __init__(self, config_path: Path):
        self.config_path = config_path

        self.yaml = YAML()
        self.yaml.preserve_quotes = True
        self.yaml.default_flow_style = False
        self.dict = self._load_config()
        self._modified = False
        self._undo_stack = []  # For undo support

    def _load_config(self) -> Dict:
        """Load YAML configuration."""
        if not self.config_path.exists():
            logger.error(f"Config not found: {self.config_path}")
            raise FileNotFoundError(f"Required: {self.config_path}")

        logger.debug(f"Loading config from {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return self.yaml.load(f)

    def save_config(self):
        """Save configuration with comments preserved."""
        if not self._modified:
            return

        logger.info("Saving updated configuration")
        with open(self.config_path, "w", encoding="utf-8") as f:
            self.yaml.dump(self.dict, f)

        self._modified = False

    def get(self, parts: str|List[str], default: Any = None) -> Any:
        """Get value by dot-separated path or already splitted path parts"""
        if isinstance(parts, str):
            parts = parts.split(".")
        value = self.dict

        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default

        return value

    def get_all_instruments(self) -> List[Tuple[str, str]]:
        """
        Get all available instruments.

        Returns: List of (instrument_type, instrument_key) tuples
        """
        result = []
        for type_key, type_data in self.dict.get("_instruments", {}).items():
            if not type_key.startswith("_"):
                continue

            inst_type = type_key[1:]
            for inst_key in type_data.keys():
                if not inst_key.startswith("_"):
                    result.append((inst_type, inst_key))

        return result

    def get_instrument_parameters(
        self, instrument_type: str, instrument_key: str, data_columns: Optional[List[str]] = None
    ) -> Dict[str, Dict]:
        """
        Get instrument parameters with inheritance.

        Returns parameters listed in instrument's 'parameters' with its
        properties inherited from common_parameters + instrument overrides.
        Also includes calculated parameters from common_parameters if present in data.

        Args:
            instrument_type: Type of instrument (e.g., "CTD")
            instrument_key: Key of instrument in config
            data_columns: Optional list of column names from data files

        Returns:
            Dict of parameter_key -> parameter_info
        """
        type_data = self.get(f"_instruments._{instrument_type}")

        common_params = type_data.get("common_parameters", {})
        inst_data = type_data.get(instrument_key, {})
        inst_params = inst_data.get("parameters", {})

        # Include parameters listed in instrument's parameters
        result = {}
        for param_key in inst_params.keys():
            if param_key not in common_params:
                logger.warning(f"Parameter {param_key} not in common_parameters for {instrument_type}")
                continue

            # Start with common definition
            param_info = dict(common_params[param_key])

            # Merge instrument-specific overrides
            inst_overrides = inst_params[param_key]
            if inst_overrides:
                param_info.update(inst_overrides)

            # Convert range_default/accuracy_default to range/accuracy
            if "range_default" in param_info and "range" not in param_info:
                param_info["range"] = param_info["range_default"]
            if "accuracy_default" in param_info and "accuracy" not in param_info:
                param_info["accuracy"] = param_info["accuracy_default"]

            result[param_key] = param_info

        # Include calculated parameters from common_parameters if present in data
        if data_columns:
            data_columns_lower = [c.lower() for c in data_columns]
            for param_key, param_info in common_params.items():
                # Skip if already included or not calculated
                if param_key in result or not is_calculated_parameter(param_info):
                    continue

                # Check if parameter is in data columns
                found = False
                if param_key.lower() in data_columns_lower:
                    found = True
                else:
                    for alias in param_info.get("aliases", []):
                        if alias.lower() in data_columns_lower:
                            found = True
                            break

                if found:
                    # Start with common definition
                    result_param = dict(param_info)

                    # Merge instrument-specific overrides
                    if param_key in inst_params:
                        result_param.update(inst_params[param_key])

                    # Convert range_default/accuracy_default to range/accuracy
                    if "range_default" in result_param and "range" not in result_param:
                        result_param["range"] = result_param["range_default"]
                    if "accuracy_default" in result_param and "accuracy" not in result_param:
                        result_param["accuracy"] = result_param["accuracy_default"]

                    result[param_key] = result_param

        return result

    def find_parameter_by_column(
        self, column_name: str, parameters: Dict[str, Dict]
    ) -> Optional[Tuple[str, Dict]]:
        """Find parameter by column name (case-insensitive, via key or aliases)."""
        col_lower = column_name.lower()

        for param_key, param_data in parameters.items():
            if param_key.lower() == col_lower:
                return (param_key, param_data)

            for alias in param_data.get("aliases", []):
                if alias.lower() == col_lower:
                    return (param_key, param_data)

        return None

    def get_instrument_type_info(self, instrument_type: str, param) -> Dict:
        """Get instrument type information."""
        type_data = self.get(f"_instruments._{instrument_type}")
        return type_data.get(param, instrument_type)

    def get_vessel_info(self, vessel_code: str) -> Optional[Dict]:
        """Get vessel information."""
        return

    def parse_data_directory(self, data_path: Path, data_files: List[Path]) -> Dict[str, Optional[str]]:
        """
        Parse directory and files to extract all metadata in one pass.

        Consolidates multiple parsing operations:
        - Directory path parsing for vessel/cruise/instrument info
        - Filename parsing for vessel/cruise from MCD format
        - Instrument detection from folder name

        Input:
            - data_path: directory path containing data files
            - data_files: list of data files in the directory

        Returns: dict with keys:
            - vessel_code: vessel code (short/dirty identifier, e.g., "AkFedorov")
            - cruise_number: cruise number (e.g., "26")
            - instrument_type: instrument type (e.g., "CTD", "XBT")
            - instrument_id: instrument code (short/dirty model string version, e.g., "SST90")
        """
        logger.debug(f"Parsing data directory: {data_path}")
        folder_result = {
            "vessel_code": None,
            "cruise_number": None,
            "instrument_type": None,
            "instrument_id": None,
        }

        # First, try to resolve vessel/cruise from MCD filenames (highest priority)
        for file in data_files:
            match = re.match(r"E\d+O2_([A-Za-z]+\d*)_(\d+)_", file.stem)
            if match:
                vessel_code = match.group(1) or ""
                cruise_num = match.group(2)
                vessel_info = self.get(f"_vessels.{vessel_code}")
                if vessel_info:
                    logger.info(f"Resolved vessel: {vessel_code}, cruise: {cruise_num}")
                    folder_result["vessel_code"] = vessel_code
                    folder_result["cruise_number"] = cruise_num
                    break

        # Only use directory info that is not already resolved from filenames
        if not folder_result["vessel_code"] or not folder_result["cruise_number"]:
            # Parse directory path for additional info
            parts = data_path.parts
            match_top = None
            for part in parts:
                if not match_top:
                    match_top = re.match(
                        r"(?P<date_str>\d{6}|\d{4}-\d{2}-\d{2})_"
                        r"(?P<cruise_vessel>[A-Za-z]+)(?P<cruise_number>\d*)"
                        r"(?:@(?P<instrument_after_at>.+))?",
                        part,
                    )
                    if match_top:
                        # Only use directory info if not already resolved from filenames
                        if not folder_result["vessel_code"]:
                            folder_result["vessel_code"] = match_top.group("cruise_vessel")
                        if not folder_result["cruise_number"]:
                            folder_result["cruise_number"] = match_top.group("cruise_number")
                        folder_result["instrument_id"] = match_top.group("instrument_after_at")

                # Instrument type/model
                match = re.match(r"CTD|XBT|XCTD", part, re.IGNORECASE)
                if match:
                    folder_result["instrument_type"] = match.group()
                    if "_" in part or "@" in part:
                        model_part = re.split(r"[_@]", part, 1)[1]
                        if folder_result["instrument_id"] is None:
                            folder_result["instrument_id"] = model_part

        if folder_result["instrument_id"] is None:
            folder_result["instrument_id"] = data_path.parent.name

        logger.debug(f"Parsed directory: {folder_result}")
        return folder_result

    def _detect_instrument_from_folder(
        self, instrument_type: str = None, instrument_id: str = '', **kwargs
    ) -> Optional[Tuple[str, str, Dict]]:
        """
        Detect instrument with selection if not found, with serial number handling.
        Implements automatic instrument detection
        from directory structure for section [Приборы] in our output *.met file.

        instrument_type - type of instrument
        instrument_id - part of folder name, that can contain instrument model and type ()

        Input format: folder name containing instrument model/type, e.g.:
        - "SST90" or "SST_90" - Sea&Sun Technology CTD 90M
        - "SAIV_SD208" - SAIV SD208
        - "CTD48M#1253" - CTD 48M with serial number 1253

        Patterns:
        - Direct match in folder_patterns
        - {type}_{model}
        - {model}#{serial} - extract serial separately
        - {type}_{model}#{serial}

        kwargs: not used here
        Returns: (instrument_type, instrument_key, instrument_data) or None
        """
        # Extract serial number if present
        serial = None
        folder_base = instrument_id
        if "#" in instrument_id:
            folder_base, serial = instrument_id.split("#", 1)
            logger.debug(f"Extracted serial: {serial}")

        folder_lower = folder_base.lower()

        for type_cur, type_data in self.dict.get("_instruments", {}).items():
            if not type_cur.startswith("_"):
                continue
            type_key = type_cur[1:]  # Remove "_" prefix
            if instrument_type and instrument_type != type_key:
                continue

            for inst_key, inst_data in type_data.items():
                if inst_key.startswith("_") or not isinstance(inst_data, dict):
                    continue

                # Direct pattern match - check if any pattern is in folder name
                patterns = inst_data.get("folder_patterns", [])
                for pattern in patterns:
                    if pattern.lower() in folder_lower:
                        logger.debug(f"Pattern match: {pattern} in {folder_base}")
                        return {"instrument_type": type_key, "instrument_id": inst_key, "instrument_info": inst_data}
                # Type_Model pattern - check if folder starts with type_
                if folder_lower.startswith(f"{type_key.lower()}_"):
                    # Extract model part after type_
                    model_part = folder_lower[len(f"{type_key.lower()}_") :]
                    # Check if model part matches any word in instrument key
                    for word in inst_key.lower().split():
                        if word in model_part:
                            logger.debug(f"Type_Model match: {type_key}_{word} in {folder_base}")
                            return {"instrument_type": type_key, "instrument_id": inst_key, "instrument_info": inst_data}

                # Model matching - use word boundaries for more precise matching
                # Only match if the word is a whole word or at least 4 chars
                for word in inst_key.split():
                    word_lower = word.lower()
                    if len(word_lower) >= 4:
                        # Use regex for word boundary matching
                        if re.search(rf"\b{re.escape(word_lower)}\b", folder_lower):
                            logger.debug(f"Model match: {word} in {folder_base}")
                            return {"instrument_type": type_key, "instrument_id": inst_key, "instrument_info": inst_data}
        return None


    def add_to_history(self, *parts: str, value: Any):
        """Add value to history with hierarchical structure of any depth"""
        if isinstance(value, str):
            value = clean_input(value)

        # Check Windows-1251 compatibility
        valid, error = validate_cp1251(value)  # todo auto replace bad symbols
        if not valid:
            logger.warning(f"Detected bad encoding (will be replaced on saving): {error}")

        # Store for undo
        if "_history" not in self.dict:
            self.dict["_history"] = {}
        section = self.dict["_history"]
        for i, part in enumerate(parts, start=1):
            if isinstance(section, list):
                if i == len(parts):
                    try:
                        index = section.index[value]
                    except IndexError:
                        pass
                    else:
                        if index == 0:
                            break
                        else:
                            del section[index]  # remove from old position
                    # store current value on the top of current history items level
                    section.insert(0, value)
                    self._modified = True
                    break
                try:
                    index = section.index[part]
                except IndexError:
                    section.append({part: {}})
                    index = -1

                section = section[index]
            else:
                if i == len(parts):
                    try:
                        if section[part] == value:
                            break
                    except KeyError:  # store current value
                        section[part] = value
                    else:  # store current and previous values
                        section[part] = [value, section[part]]
                    self._modified = True
                    break
            if not (section and section.get(part)):
                section[part] = {}
            section = section[part]

    def get_history(self, *parts: Sequence[str]) -> None|str|List[str|List[str]]:
        """Get history for field."""
        return self.get(["_history"] + list(parts))

    def add_instrument(self, instrument_type: str, instrument_id: str, instrument_info: Dict):
        """Add new instrument to config."""
        type_key = f"_{instrument_type}"

        if "_instruments" not in self.dict:
            self.dict["_instruments"] = {}

        if type_key not in self.dict["_instruments"]:
            logger.warning(f"Type {instrument_type} not in config")
            return

        self.dict["_instruments"][type_key][instrument_id] = instrument_info
        self._modified = True
        logger.info(f"Added instrument: {instrument_id}")


class DataAnalyzer:
    """Non-interactive data analysis and processing."""

    def __init__(self, config: ConfigManager):
        self.config = config

    def read_pos_file(self, pos_file: Path) -> Optional[pd.DataFrame]:
        """Read POS file with various encodings."""
        for encoding in ["utf-8", "cp1251", "latin1"]:
            try:
                df = pd.read_csv(pos_file, sep=";", encoding=encoding)
                logger.info(f"POS file loaded: {pos_file.name} ({encoding})")
                return df
            except Exception as e:
                logger.debug(f"Failed with {encoding}: {e}")

        logger.error(f"Cannot read POS file: {pos_file}")
        return None

    def get_all_columns_ordered(self, files: List[Path]) -> List[str]:
        """Get all unique columns in order from first file."""
        if not files:
            return []

        try:
            df = pd.read_csv(files[0], sep=";", encoding="utf-8", nrows=0)
            ordered_cols = list(df.columns)
        except:
            ordered_cols = []

        all_cols = set(ordered_cols)
        for file in files[1:]:
            try:
                df = pd.read_csv(file, sep=";", encoding="utf-8", nrows=0)
                for col in df.columns:
                    if col not in all_cols:
                        ordered_cols.append(col)
                        all_cols.add(col)
            except:
                continue

        return ordered_cols

    def detect_format(self, series: pd.Series) -> str:
        """
        Detect FC/AA format for data series.

        Implements part of MCD requirements for [Данные].

        Input: pandas Series with data values

        Output format:
            - FC(w,d) for numeric fields: width w, decimals d
            - AA(w) for text fields: width w

        Format specification:
            FC(w[,d]) – числовой формат, w≤15, d≤7
            AA(w) – символьный формат для текста
        """
        series = series.dropna()
        if len(series) == 0:
            return "AA(10)"

        # Convert to string and strip whitespace using pandas str methods
        str_series = series.astype(str).str.strip()

        # Filter out empty strings
        str_series = str_series[str_series != ""]

        if len(str_series) == 0:
            return "AA(10)"

        # Try to detect numeric format by analyzing string values directly
        # This preserves original decimal precision from CSV files

        # Check if all values are numeric using pandas str methods
        # Match optional sign, digits, optional decimal point, optional digits
        numeric_pattern = r"^-?\d*\.?\d+$"
        is_numeric_mask = str_series.str.match(numeric_pattern)

        if not is_numeric_mask.all():
            # Not all numeric - use text format
            max_len = str_series.str.len().max()
            return f"AA({max(max_len, 5)})"

        # All values are numeric - analyze format using pandas str methods
        has_negative = str_series.str.startswith("-").any()

        # Remove negative signs for length calculation
        clean_series = str_series.str.lstrip("-")

        # Split into integer and decimal parts
        split_series = clean_series.str.split(".", n=1, expand=True)
        int_parts = split_series[0]

        # Calculate max integer part length
        max_int_part = int(int_parts.str.len().max())

        # Calculate max decimals (only for values with decimal points)
        has_decimal = clean_series.str.contains(".")
        if not has_decimal.any() or len(split_series.columns) == 1:
            max_decimals = 0
        else:
            # Only process decimal parts if they exist
            dec_parts = split_series[1]
            # Remove trailing zeros from decimal parts and get max length
            dec_parts_no_zeros = dec_parts.str.rstrip("0")
            max_decimals = int(dec_parts_no_zeros.str.len().max())

        # Calculate total width: integer part + decimal point + decimals + sign
        width = max_int_part + (max_decimals + 1 if max_decimals > 0 else 0) + (1 if has_negative else 0)

        # Use minimum width of 1, not 4, to avoid overestimating small fields
        width = max(width, 1)

        return f"FC({width},{max_decimals})" if max_decimals > 0 else f"FC({width})"

    def detect_format_from_files(self, column: str, files: List[Path]) -> str:
        """
        Detect format by scanning all data files.

        Reads values as strings from CSV to preserve original decimal precision,
        avoiding floating-point conversion artifacts.
        """
        all_series = []

        for file in files:
            try:
                # Read CSV as strings to preserve original decimal precision
                df = pd.read_csv(file, sep=";", encoding="utf-8", dtype=str)
                if column in df.columns:
                    # Drop NaN values
                    series = df[column].dropna()
                    all_series.append(series)
            except:
                continue

        if not all_series:
            return "AA(10)"

        # Concatenate all series to avoid converting to list and back
        combined = pd.concat(all_series, ignore_index=True)

        return self.detect_format(combined)

    def extract_dates(self, pos_df: pd.DataFrame) -> Dict[str, str]:
        """Extract start/end dates from POS data."""
        date_col = next((c for c in pos_df.columns if "DATE" in c.upper() or "ДАТА" in c.upper()), None)

        if date_col:
            try:
                dates = pd.to_datetime(pos_df[date_col], errors="coerce").dropna()
                return {"start": dates.min().strftime("%d-%m-%Y"), "end": dates.max().strftime("%d-%m-%Y")}
            except:
                pass

        return {"start": "01.01.2024", "end": "31.12.2024"}


def determine_geographic_region(geo_config, lats: List[float], lons: List[float]) -> Dict:
    """Hierarchical cascade detection: Ocean → Sea → Gulf/Strait."""
    ocean_config = geo_config.get("Океан", {})
    ocean_options = ocean_config.get("_options", {})

    result = {}

    for ocean_name, ocean_data in ocean_options.items():
        if not isinstance(ocean_data, dict) or "bounds" not in ocean_data:
            continue

        if _any_coord_in_bounds(lats, lons, ocean_data["bounds"]):
            result["Океан"] = ocean_name
            logger.debug(f"Ocean detected: {ocean_name}")

            sea_dict = ocean_data.get("Море", {})
            for sea_name, sea_config in sea_dict.items():
                if not isinstance(sea_config, dict) or "bounds" not in sea_config:
                    continue

                if _any_coord_in_bounds(lats, lons, sea_config["bounds"]):
                    result["Море"] = sea_name
                    logger.debug(f"Sea detected: {sea_name}")

                    for region_type in ["Залив", "Пролив", "бассейн"]:
                        if region_type not in sea_config:
                            continue

                        regions_found = []
                        for region_name, region_data in sea_config[region_type].items():
                            if not isinstance(region_data, dict) or "bounds" not in region_data:
                                continue

                            if _any_coord_in_bounds(lats, lons, region_data["bounds"]):
                                regions_found.append(region_name)
                                logger.debug(f"{region_type} detected: {region_name}")

                        if regions_found:
                            result[region_type] = regions_found

                    break
            break

    return result


def _any_coord_in_bounds(lats: List[float], lons: List[float], bounds: Dict[str, float]) -> bool:
    """Check if ANY coordinate falls within bounds."""
    for lat, lon in zip(lats, lons):
        if bounds["lat_min"] <= lat <= bounds["lat_max"] and bounds["lon_min"] <= lon <= bounds["lon_max"]:
            return True
    return False


def calculate_coordinate_bounds(lats: List[float], lons: List[float]) -> str:
    """
    Calculate coordinate bounds in VNIIGMI-MCD format.

    Implements part of MCD requirements for [Привязка].

    Input: lists of latitude and longitude values in decimal degrees

    Output format: "Ш1;Ш2;Д1;Д2" where:
        - Ш1, Ш2 = latitude bounds in ЗГГММ format (sign+degrees+minutes)
        - Д1, Д2 = longitude bounds in ЗГГГММ format (sign+degrees+minutes)
        Example: "+5430;+6000;+01500;+03000"

    Format specification:
        Ш1 – широта южной границы района в виде ЗГГММ
        Ш2 – широта северной границы района в виде ЗГГММ
        Д1 – долгота западной границы района в виде ЗГГГММ
        Д2 – долгота восточной границы района в виде ЗГГГММ
    """
    lat_min, lat_max = min(lats), max(lats)
    lon_min, lon_max = min(lons), max(lons)

    def to_dgmm(value: float, is_lat: bool) -> str:
        """Convert decimal degrees to ЗГГММ or ЗГГГММ format."""
        sign = "" if value >= 0 else "-"
        abs_val = abs(value)
        degrees = int(abs_val)
        minutes = int((abs_val - degrees) * 60)
        return f"{sign}{degrees:{'02d' if is_lat else '03d'}}{minutes:02d}"

    return f"{to_dgmm(lat_min, True)};{to_dgmm(lat_max, True)};{to_dgmm(lon_min, False)};{to_dgmm(lon_max, False)}"


class InteractiveQuestionnaire:
    """Interactive data collection."""

    def __init__(self, config: ConfigManager, analyzer: DataAnalyzer):
        self.config = config
        self.analyzer = analyzer

    def _ask_with_prompt(self, *parts: str, default: str = "", **format_kwargs) -> str:
        """
        Ask question with description and history support.

        Args:
            parts: Configuration path
            default: Default value for the field
            **format_kwargs: Additional named parameters for formatting prompt_text and default

        Returns:
            User input or default value
        """
        if len(parts) > 1:
            path = '.'.join(parts)
            parts = path.split(".")
        else:
            path = parts
        field = parts[-1].removeprefix("_")
        field_config = self.config.get(parts)
        if not field_config:
            logger.debug(f"No configuration to display question for {path}")
            prompt_text = field
        else:  # Use for prompt configured `_prompt` (else `field`), print description with rest info
            try:
                prompt_text = field_config["_prompt"]
            except KeyError:
                prompt_text = field
                try:
                    desc = field_config["_description"]
                    print(f"\n💡 {desc}")
                except KeyError:
                    pass
            else:
                try:
                    desc = field_config["_description"]
                    print(f"\n💡 {field}: {desc}")
                except KeyError:
                    print(f"\n💡 {field}")

        # Format prompt_text and default with provided kwargs if any
        if format_kwargs:
            try:
                prompt_text = prompt_text.format(**format_kwargs)
                if default:
                    default = default.format(**format_kwargs)
            except (KeyError, ValueError) as e:
                logger.debug(f"Format error for {path}: {e}")

        history = self.config.get_history(*parts)
        if history:
            if isinstance(history, str):
                history = [history]
            result = ask(
                questionary.autocomplete,
                message=prompt_text + ":",
                choices=history,
                default=default or (history[0] if history else ""),
            )
        else:
            result = ask(questionary.text, message=prompt_text + ":", default=default)

        result = clean_input(result) if result else default
        if result and result != default:
            self.config.add_to_history(*parts, value=result)

        return result

    def collect_metadata(
        self, parsed_info: Dict, pos_df: pd.DataFrame, all_files: List[Path], output_filename: str
    ) -> Dict:
        """Collect all metadata through interactive questionnaire."""
        result = {"Общие характеристики": {}, "Дополнительные сведения": {}, "Структура файла данных": {}}

        # Prepare format parameters for Наименование field
        format_params = {}
        if parsed_info:
            format_params["instrument_type"] = parsed_info["instrument_type"]
            format_params["instrument"] = parsed_info["instrument_id"]

        vessel_code = parsed_info.get("vessel_code")
        if vessel_code:
            vessel_type = self.config.get(f"_vessels.{vessel_code}.type")
            format_params["vessel"] = " ".join(
                ([vessel_type] if vessel_type else []) +
                [self.config.get(f"_vessels.{vessel_code}.name", default=vessel_code)]
            )
        cruise_number = parsed_info.get("cruise_number", "")
        format_params["cruise"] = ", ".join(
            [format_params["vessel"]] + ([f"рейс {cruise_number}"] if cruise_number else [])
        )

        # General characteristics
        sec = "Общие характеристики"
        result[sec][field] = self._get_field(sec, field := "Наименование", **format_params)
        result[sec][field] = self._get_field(sec, field := "Содержание")
        result[sec]["Географический район"] = self._get_geographic_region(pos_df)
        result[sec]["Форма представления данных"] = self.config.get(
            "Общие характеристики.Форма представления данных._default"
        )

        dates = self.analyzer.extract_dates(pos_df)
        result[sec]["Дата начала"] = dates["start"]
        result[sec]["Дата окончания"] = dates["end"]

        result[sec][field] = self._select_option(sec, field := "Временное разрешение")
        result[sec][field] = self._select_option(sec, field := "Пространственное разрешение")
        result[sec][field] = self._get_field(sec, field := "Упорядочение")

        # Data elements - WITH VALIDATION
        data_files = [f for f in all_files if "POS" not in f.name and f.suffix == ".csv"]
        ordered_columns = self.analyzer.get_all_columns_ordered(data_files)
        result[sec]["Элементы данных"] = self._get_data_elements(data_files, parsed_info, ordered_columns)

        result[sec]["Словари"] = self.config.get("Общие характеристики.Словари._default")
        result[sec]["Файловая структура"] = self._generate_file_structure(all_files, output_filename)

        # Prepare format parameters for Источник field
        # Get organization name from hierarchical "Выходные реквизиты" structure
        output_reqs = self.config.get("Общие характеристики.Выходные реквизиты", {})
        format_params["organization"] = ""
        if output_reqs:
            # Get first organization key (e.g., "ФГБУН «АО ИО РАН»")
            org_keys = [k for k in output_reqs.keys() if not k.startswith("_")]
            if org_keys:
                format_params["organization"] = org_keys[0]

        result[sec]["Источник"] = self._get_field(sec, field := "Источник", **format_params)
        result[sec]["Выходные реквизиты"] = self._get_output_info(organization=format_params["organization"])

        # Additional information
        sec = "Дополнительные сведения"
        result[sec]["Приборы"] = self._generate_instruments_description(parsed_info)
        result[sec][field] = self._get_field(sec, field := "Методы обработки")
        result[sec]["Привязка"] = self._get_binding(pos_df)
        result[sec][field] = self._get_field(sec, field := "Полнота")
        result[sec]["Объем"] = self._calculate_volume(all_files)

        # File structure
        pos_files = [f for f in all_files if "POS" in f.name]
        if pos_files:
            result["Структура файла данных"][f"Данные] Таблица Признаки (главная), {pos_files[0].name}"] = (
                self._generate_pos_structure(pos_df)
            )

        if data_files:
            result["Структура файла данных"][
                "Данные] Таблица Параметры (подчиненная к таблице Признаки), "
                f"{output_filename.replace('.met', '.csv')}"
            ] = self._generate_data_structure(data_files, parsed_info)

        return result

    def _get_field(self, *parts: Sequence[str], **format_kwargs) -> str:
        """
        Get field value with default and history.

        Args:
            parts: Configuration path parts
            **format_kwargs: Additional named parameters for formatting prompt_text and default

        Returns:
            User input or default value
        """
        return self._ask_with_prompt(
            *parts, default=self.config.get(list(parts) + ["_default"], ""), **format_kwargs
        )

    def _select_option(self, section: str, field: str) -> str:
        """Select from options."""
        field_config = self.config.get(f"{section}.{field}")
        options = field_config.get("_options", [])
        prompt = field_config.get("_prompt", field)

        return ask(questionary.select, message=prompt + ":", choices=options)

    def _add_new_instrument(
        self, instrument: str, instrument_type: str = "", show_selection: bool = True, **kwargs
    ) -> Optional[Dict]:
        """
        Add new instrument or select from existing instruments.

        instrument: instrument name/identifier (from folder name)
        instrument_type: instrument type (e.g., "CTD", "XBT") if detected
        show_selection: if True, show selection dialog (for when auto-detection fails)
        kwargs: not used here
        """
        all_instruments = self.config.get_all_instruments()
        choices = [f"{inst_type}: {inst_key}" for inst_type, inst_key in all_instruments]
        choices.append("+ Добавить новый")

        if show_selection:
            selection = ask(
                questionary.select,
                message="Прибор '{}'{} не найден. Выберите:".format(
                    instrument, f" типа '{instrument_type}'" if instrument_type else ""
                ),
                choices=choices,
            )

            if selection == "+ Добавить новый":
                return self._add_new_instrument(instrument, instrument_type, show_selection=False)

            inst_type, inst_key = selection.split(": ", 1)
            inst_data = self.config.get(f"_instruments._{inst_type}")[inst_key]
            logger.info(f"Selected instrument: {inst_key} (type: {inst_type})")
            return {"instrument_type": inst_type, "instrument_id": inst_key, "instrument_info": inst_data}

        # Add new instrument
        print("\n" + "=" * 80)
        print("ДОБАВЛЕНИЕ НОВОГО ПРИБОРА")
        print("=" * 80)

        inst_name = ask(questionary.text, message="Название прибора:")
        if not inst_name:
            return None

        inst_type = ask(questionary.select, message="Тип прибора:", choices=["CTD", "XBT", "Другой"])

        manufacturer = ask(questionary.text, message="Производитель:")
        year = ask(questionary.text, message="Год выпуска:")

        # Allow user to edit folder pattern (detected from folder name)
        folder_pattern = ask(
            questionary.text, message="Шаблон папки (для автоопределения):", default=instrument
        )

        inst_data = {
            "folder_patterns": [folder_pattern] if folder_pattern else [instrument],
            "manufacturer": manufacturer,
            "year": int(year) if year and year.isdigit() else None,
            "parameters": {},
        }
        out = {"instrument_type": inst_type, "instrument_id": inst_name, "instrument_info": inst_data}
        self.config.add_instrument(**out)
        logger.info(f"Added new instrument: {inst_name}")
        return out

    def _get_geographic_region(self, pos_df: pd.DataFrame) -> Dict:
        """Get geographic region with cascade auto-detection."""
        lat_col = next((c for c in pos_df.columns if c.upper() in ["LAT", "ШИРОТА"]), None)
        lon_col = next((c for c in pos_df.columns if c.upper() in ["LON", "LONG", "ДОЛГОТА"]), None)

        if lat_col and lon_col:
            lats = pos_df[lat_col].dropna().tolist()
            lons = pos_df[lon_col].dropna().tolist()
            geo_config = self.config.get("Общие характеристики.Географический район", {})
            auto_result = determine_geographic_region(geo_config, lats, lons)

            if auto_result:
                logging.debug("АВТООПРЕДЕЛЕНИЕ ГЕОГРАФИЧЕСКОГО РАЙОНА")
                for key, value in auto_result.items():
                    if isinstance(value, list):
                        print(f"{key}: {', '.join(value)}")
                    else:
                        print(f"{key}: {value}")
                print("=" * 80)

                use_auto = ask(
                    questionary.confirm, message="Использовать автоопределенные значения?", default=True
                )

                if use_auto:
                    return auto_result

        return self._manual_geographic_selection()

    def _manual_geographic_selection(self) -> Dict:
        """Manual hierarchical selection."""
        result = {}

        geo_config = self.config.get("Общие характеристики.Географический район")
        ocean_config = geo_config["Океан"]
        ocean_options = ocean_config["_options"]
        ocean_names = list(ocean_options.keys())

        ocean = ask(questionary.select, message="Океан:", choices=ocean_names)
        result["Океан"] = ocean

        ocean_data = ocean_options[ocean]
        sea_dict = ocean_data.get("Море", {})

        if sea_dict:
            sea_names = list(sea_dict.keys())
            sea = ask(questionary.select, message="Море:", choices=sea_names)
            result["Море"] = sea

            sea_config = sea_dict[sea]
            for region_type in ["Залив", "Пролив", "бассейн"]:
                if region_type not in sea_config:
                    continue

                region_names = list(sea_config[region_type].keys())
                if region_names:
                    selected = ask(
                        questionary.checkbox,
                        message=f"{region_type} (выберите все подходящие):",
                        choices=region_names,
                    )

                    if selected:
                        result[region_type] = selected

        return result

    def _get_data_elements(
        self, files: List[Path], instrument_info: Optional[Dict], ordered_columns: Optional[List[str]] = None
    ) -> List[str]:
        """Get data elements with validation."""
        if ordered_columns is None:
            ordered_columns = self.analyzer.get_all_columns_ordered(files)

        id_cols = {
            "rec_num",
            "Rec_num",
            "НОМЗАП",
            "identific",
            "Identific",
            "ИДЕНТИФ",
            "station",
            "Station",
            "НОМСТ",
        }
        param_columns = [c for c in ordered_columns if c not in id_cols]

        params = {}
        if instrument_info:
            params = self.config.get_instrument_parameters(
                instrument_info["instrument_type"], instrument_info["instrument_id"], ordered_columns
            )

            # VALIDATION: Warn about missing measured parameters
            self._validate_measured_parameters(param_columns, params)

        # Generate descriptions
        elements = []
        for col in param_columns:
            param_result = self.config.find_parameter_by_column(col, params)

            if param_result:
                param_key, param_info = param_result
                desc = self._generate_parameter_description(param_key, param_info)
                elements.append(desc)
            else:
                elements.append(f"{col} (параметр не описан)")

        print("\n" + "=" * 80)
        print("ЭЛЕМЕНТЫ ДАННЫХ")
        print("=" * 80)
        print("Требование ВНИИГМИ-МЦД: описываем ТОЛЬКО параметры из файлов данных.")
        print("\nСгенерированные описания для найденных параметров:")
        print("-" * 80)
        for i, (col, elem) in enumerate(zip(param_columns, elements), 1):
            print(f"{i}. {col}: {elem}")
        print("-" * 80)

        edited_text = ask(
            questionary.text,
            message="Отредактируйте описания (разделяйте точкой с запятой):",
            default="; ".join(elements),
        )

        edited_elements = [e.strip() for e in edited_text.split(";") if e.strip()]

        # Validate all columns described
        described = set()
        for elem in edited_elements:
            for col in param_columns:
                if col.lower() in elem.lower():
                    described.add(col)
                    break

                param_result = self.config.find_parameter_by_column(col, params)
                if param_result:
                    _, param_info = param_result
                    for alias in param_info.get("aliases", []):
                        if alias.lower() in elem.lower():
                            described.add(col)
                            break

        missing = set(param_columns) - described
        if missing:
            logger.warning(f"Not described: {missing}")
            add_missing = ask(
                questionary.confirm,
                message=f"Добавить описания для пропущенных параметров ({', '.join(missing)})?",
                default=True,
            )

            if add_missing:
                for col in missing:
                    desc = ask(
                        questionary.text,
                        message=f"Описание для {col}:",
                        default=f"{col} (единица измерения, диапазон, точность)",
                    )
                    edited_elements.append(clean_input(desc))

        return sorted(edited_elements)

    def _generate_parameter_description(self, param_key: str, param_info: Dict) -> str:
        """Generate description for [Элементы данных] section."""
        aliases = param_info.get("aliases", [])
        name = aliases[-1] if aliases else param_key

        unit = param_info.get("unit", "")
        range_val = param_info.get("range", "")
        accuracy = param_info.get("accuracy", "")

        parts = [name]

        if unit:
            parts[0] += f" ({unit})"

        if is_calculated_parameter(param_info):
            parts.append("расчитываемый")
        else:
            if range_val:
                parts.append(f"диапазон измерений {range_val}")

        if accuracy:
            parts.append(f"точность {accuracy}")

        return ", ".join(parts)

    def _validate_measured_parameters(self, param_columns: List[str], instrument_params: Dict):
        """Warn if measured parameters from instrument are missing in data files."""
        expected_measured = [k for k, v in instrument_params.items() if not is_calculated_parameter(v)]

        if not expected_measured:
            return

        param_columns_lower = [c.lower() for c in param_columns]
        missing_measured = []

        for param_key in expected_measured:
            param_info = instrument_params[param_key]
            found = False

            if param_key.lower() in param_columns_lower:
                found = True
            else:
                for alias in param_info.get("aliases", []):
                    if alias.lower() in param_columns_lower:
                        found = True
                        break

            if not found:
                missing_measured.append(param_key)

        if missing_measured:
            param_names = [instrument_params[k].get("aliases", [k])[-1] for k in missing_measured]
            logger.warning("=" * 80)
            logger.warning("ОТСУТСТВУЮЩИЕ ИЗМЕРЯЕМЫЕ ПАРАМЕТРЫ")
            logger.warning(f"Прибор должен измерять: {', '.join(missing_measured)}")
            logger.warning(f"Не найдено в данных: {', '.join(param_names)}")
            logger.warning("=" * 80)
            print(f"\n[WARNING] Отсутствуют измеряемые параметры: {', '.join(param_names)}\n")

    def _generate_file_structure(self, files: List[Path], met_filename: str) -> List[str]:
        """Generate file structure listing."""
        lines = [f"{met_filename} - файл метаданных"]

        pos_files = [f for f in files if "POS" in f.name]
        if pos_files:
            lines.append(f"{pos_files[0].name} - файл данных признаков (дата, координаты и др.)")

        data_files = sorted([f for f in files if "POS" not in f.name and f.suffix == ".csv"])
        for i, f in enumerate(data_files, 1):
            lines.append(f"{f.name} - {i}-й файл данных")

        return lines

    def _get_output_info(self, organization: str = "") -> str:
        """
        Get output requisites with all required fields saved to history.

        According to VNIIGMI-MCD guidelines (line 75), [Выходные реквизиты] must include:
        - организация-изготовитель комплекта данных
        - ФИО ответственного за подготовку комплекта
        - информация для контактов (телефон, email)
        - дата создания комплекта

        Args:
            organization: Organization short name from hierarchical config structure

        Returns:
            Formatted string with all output requisites
        """
        parts = ["Общие характеристики", "Выходные реквизиты"]

        # Get 1st organization info from hierarchical structure if not provided
        if not organization:
            organization = self._ask_with_prompt(*parts)
            if output_reqs := self.config.get(parts):
                org_keys = [k for k in output_reqs.keys() if not k.startswith("_")]
                if org_keys:
                    organization = org_keys[0]

        # Collect all required fields
        org_manufacturer = self._get_field(*parts, organization=organization)
        parts += [organization]
        # Get structure under the organization
        fio = self._ask_with_prompt(*parts, "_ФИО")
        email = self._ask_with_prompt(*parts, "_email")
        phone = self._ask_with_prompt(*parts, "_Телефон")

        today = datetime.now().strftime("%d.%m.%Y")
        return f"{org_manufacturer}, {fio}, {phone}, {email}, {today}"


    def _generate_instruments_description(self, instrument_info: Optional[Dict]) -> Dict:
        """Generate instruments description"""
        if not instrument_info:
            desc = ask(questionary.text, message="Приборы:")
            return desc
        parts = [instrument_info["instrument_type"], instrument_info["instrument_id"]]
        inst_data = instrument_info["instrument_info"]
        try:
            parts.append(f"{inst_data['year']} года пр-ва")
        except KeyError:
            pass
        try:
            parts.append(inst_data["manufacturer"])
        except KeyError:
            pass
        return " ".join(parts)


    def _get_binding(self, pos_df: pd.DataFrame) -> str:
        """Get binding - coordinate bounds for stations."""
        lat_col = next((c for c in pos_df.columns if c.upper() in ["LAT", "ШИРОТА"]), None)
        lon_col = next((c for c in pos_df.columns if c.upper() in ["LON", "LONG", "ДОЛГОТА"]), None)

        if lat_col and lon_col:
            lats = pos_df[lat_col].dropna().tolist()
            lons = pos_df[lon_col].dropna().tolist()

            if lats and lons:
                return calculate_coordinate_bounds(lats, lons)

        return ""

    def _calculate_volume(self, files: List[Path]) -> str:
        """Calculate data volume."""
        data_files = [f for f in files if "POS" not in f.name and f.suffix == ".csv"]
        total_size = sum(f.stat().st_size for f in data_files)
        size_mb = total_size / (1024 * 1024)
        return f"{len(data_files)} файлов данных занимают {size_mb:.2f} Мбайт"

    def _generate_pos_structure(self, pos_df: pd.DataFrame) -> List[str]:
        """
        Generate POS table structure for [Данные] section.

        Implements part of MCD requirements for describing POS file structure.

        Input: DataFrame with POS data columns

        Output: list of strings in format:
            "column_name;    FC(w,d);  // description"

        Each line describes one field according to MCD format:
            - Field name (from column header)
            - Format (FC or AA with width/decimals)
            - Comment after // with full description
        """
        structure = []
        pos_fields = self.config.get("Структура файла данных.Таблица Признаки", {})

        for col in pos_df.columns:
            detected_format = self.analyzer.detect_format(pos_df[col])

            desc = col
            for field_key, field_data in pos_fields.items():
                if col in field_data.get("alternatives", []):
                    desc = field_data["description"]
                    break

            structure.append(f"{col};    {detected_format};  // {desc}")

        return structure

    def _generate_data_structure(self, data_files: List[Path], instrument_info: Optional[Dict]) -> List[str]:
        """
        Generate data table structure for [Данные] section.

        Implements part of MCD requirements for describing data file structure.

        Input:
            - data_files: list of CSV data files
            - instrument_info: optional instrument metadata

        Output: list of strings in format:
            "column_name;    FC(w,d);   // description"

        Each line describes one field according to MCD format:
            - Field name (from column header)
            - Format (FC or AA with width/decimals)
            - Comment after // with full description including units
        """
        structure = []
        ordered_columns = self.analyzer.get_all_columns_ordered(data_files)

        pos_fields = self.config.get("Структура файла данных.Таблица Признаки", {})

        params = {}
        if instrument_info:
            params = self.config.get_instrument_parameters(
                instrument_info["instrument_type"], instrument_info["instrument_id"], ordered_columns
            )

        for col in ordered_columns:
            fmt = self.analyzer.detect_format_from_files(col, data_files)

            desc = col

            # Try POS fields first
            for field_key, field_data in pos_fields.items():
                if col in field_data.get("alternatives", []):
                    desc = field_data["description"]
                    break
            else:
                # Try parameters
                param_result = self.config.find_parameter_by_column(col, params)
                if param_result:
                    _, param_info = param_result
                    aliases = param_info.get("aliases", [])
                    name = aliases[-1] if aliases else col
                    unit = param_info.get("unit", "")
                    desc = f"{name} [{unit}]" if unit else name

            structure.append(f"{col};    {fmt};   // {desc}")

        return structure


class METFileGenerator:
    """Generate .met file content."""

    def __init__(self, config: ConfigManager):
        self.config = config

    def dict_to_met_content(self, data: Dict) -> str:
        """
        Convert structured data to .met file content.

        Implements MCD file format specification.

        Input: dict with three sections:
            - 'Общие характеристики': general characteristics
            - 'Дополнительные сведения': additional information
            - 'Структура файла данных': file structure

        Output: string in Windows-1251 encoding with:
            - Section headers: *[Section Name]
            - Field headers: [Field Name]
            - Section headers start with *
        """
        lines = []

        # Section 1
        sec = "Общие характеристики"
        lines.append(f"*[{sec}]")
        general = data.get(sec, {})
        for key in [
            "Наименование",
            "Содержание",
            "Географический район",
            "Форма представления данных",
            "Дата начала",
            "Дата окончания",
            "Временное разрешение",
            "Пространственное разрешение",
            "Упорядочение",
            "Элементы данных",
            "Словари",
            "Файловая структура",
            "Источник",
            "Выходные реквизиты",
        ]:
            try:
                value = general[key]
                if not value:
                    continue
            except KeyError:
                continue
            lines.append(f"[{key}]")
            if key == "Географический район":
                for geo_type in ["Океан", "Море", "Залив", "Пролив", "бассейн"]:
                    if geo_type in value:
                        lines.append(f"[{geo_type}]")
                        value2 = value[geo_type]
                        lines.append(", ".join(value2) if isinstance(value2, list) else str(value2))
            elif key == "Элементы данных":
                for elem in value:
                    lines.append(elem)
            elif key in general:
                if isinstance(value, list):
                    lines.extend(value)
                else:
                    lines.append(str(value))

        # Section 2
        sec = "Дополнительные сведения"
        lines.append(f"*[{sec}]")
        try:
            additional = data[sec]
            instruments = additional["Приборы"]
        except KeyError:
            pass
        else:
            lines.append("[Приборы]")
            # we not print instrument types as headers ([CTD]) as it is internal classification
            if isinstance(instruments, dict):
                # not used?: {"Прибор" or inst_type: [inst_desc1, ...]}
                for inst_type, inst_list in instruments.items():
                    if isinstance(inst_list, list):
                        for inst_desc in inst_list:
                            lines.append(
                                inst_desc if inst_type == "Прибор" else " ".join([inst_type, inst_desc])
                            )
                    else:
                        lines.append(str(inst_list))
            else:
                lines.append(str(instruments))

        for key in ["Методы обработки", "Привязка", "Полнота", "Объем"]:
            if key in additional:
                lines.append(f"[{key}]")
                value2 = additional[key]
                if isinstance(value2, list):
                    lines.extend(value2)
                else:
                    lines.append(str(value2))

        # Section 3
        sec = "Структура файла данных"
        lines.append(f"*[{sec}]")
        try:
            structure = data[sec]
        except KeyError:
            pass
        else:
            for table_key in structure:
                lines.append(f"[{table_key}]")
                table_lines = structure[table_key]
                if isinstance(table_lines, list):
                    lines.extend(table_lines)
                else:
                    lines.append(str(table_lines))

        return "\r\n".join(lines) + "\r\n"

    def write_met_file(self, output_path: Path, content: str):
        """Write .met file with Windows-1251 correction and validation for warning"""
        valid, error = validate_cp1251(content)
        if not valid:
            logger.error(f"Encoding error: {error}")
            print(f"\n[ERROR] Ошибка кодировки: {error} in {content}")

        with open(output_path, "w", encoding="windows-1251", errors="replace") as f:
            f.write(content)

        logger.info(f"MET file written: {output_path}")


def main():
    """Main entry point with graceful exit."""
    global logger, ask

    parser = argparse.ArgumentParser(description="Генератор MET файлов для морских экспедиционных данных")
    parser.add_argument(
        "--data_dir_str", type=str, help="Путь к данным (если файл - использовать родительскую папку)"
    )
    parser.add_argument("-y", "--yes", action="store_true", help="Автоматически отвечать Да на все вопросы")
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("Генератор MET файлов")
    print("=" * 80 + "\n")

    # Redefine ask() function for auto-yes mode
    if args.yes:

        def ask(questionary_func, **kwargs):
            """
            Auto-yes wrapper that mimics questionary UI with Enter key press.

            Args:
                questionary_func: questionary function (text, select, confirm, etc.)
                **kwargs: arguments to pass to questionary function

            Returns:
                Default value as if Enter was pressed
            """
            # Set default style if not provided
            if "style" not in kwargs:
                kwargs["style"] = CUSTOM_STYLE

            # Get the message and default value
            message = kwargs.get("message", "")
            default = kwargs.get("default", "")

            # For select/confirm, get first choice or boolean default
            if questionary_func == questionary.select:
                choices = kwargs.get("choices", [])
                default = choices[0] if choices else ""
            elif questionary_func == questionary.confirm:
                default = kwargs.get("default", False)
                default_str = "Y/n" if default else "y/N"
            elif questionary_func == questionary.checkbox:
                default = ""
            elif questionary_func == questionary.path:
                default = "."

            # Print exactly like Questionary does (using hardcoded ANSI codes)
            if questionary_func == questionary.confirm:
                print(f"\u001b[38;5;98;1m\u001b[0m? {message}\u001b[1m\u001b[0m [{default_str}]\u001b[0m")
            else:
                print(f"\u001b[38;5;98;1m\u001b[0m? {message}\u001b[1m\u001b[0m [{default}]\u001b[0m")

            return default

    # Get data directory from command line or interactive prompt
    if args.data_dir_str:
        data_dir_str = args.data_dir_str
    else:
        data_dir_str = ask(questionary.path, message="Путь к данным:", only_directories=True)

        if not data_dir_str:
            return

    data_dir = Path(data_dir_str)

    # If path is not a dir, try to use its parent directory
    if not data_dir.is_dir():
        data_dir = data_dir.parent
        print(f"Input is not a dir, trying parent: {data_dir}")

    if not data_dir.exists():
        print("Ошибка: каталог не найден")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = data_dir / f"met_generation_{timestamp}.log"
    setup_logging(log_file)

    logger.info("=" * 80)
    logger.info(f"Directory: {data_dir}")
    logger.info("=" * 80)

    try:
        config = ConfigManager(CONFIG_FILE)

        analyzer = DataAnalyzer(config)
        questionnaire = InteractiveQuestionnaire(config, analyzer)
        generator = METFileGenerator(config)

        pos_files = list(data_dir.glob("*POS.csv"))
        csv_files = list(data_dir.glob("*.csv"))
        all_files = pos_files + csv_files
        if not pos_files:
            logger.error("No *POS.csv!")
            print("\n[WARNING] Ошибка: отсутствует файл *POS.csv!")
            return

        pos_df = analyzer.read_pos_file(pos_files[0])
        if pos_df is None:
            return

        # Determine output filename early
        # Get org code from hierarchical "Выходные реквизиты" structure
        output_reqs = config.get("Общие характеристики.Выходные реквизиты")
        org_code = ""
        if output_reqs:
            org_keys = [k for k in output_reqs.keys() if not k.startswith("_")]
            if org_keys:
                org_code = output_reqs[org_keys[0]].get("_org_code", "000000")

        # Dirty instrument type and key {'instrument_type', 'instrument_id'}: not checked comparing to config
        parsed_info = config.parse_data_directory(data_dir, all_files)

        # Searching {'instrument_type', 'instrument_id'} in config records, replacing `instrument_id` a
        # to full matched instrument name and appending 'instrument_info' field
        parsed_info.update(config._detect_instrument_from_folder(**parsed_info))
        if parsed_info:
            logger.info("Detected {instrument_type} instrument: {instrument_id}".format_map(parsed_info))
        else:
            logger.warning(
                "Instrument (of type {instrument_type}) is not auto-detected from folder: {instrument_id}".format_map(
                    parsed_info
                )
            )
            # 2. Show selection dialog with option to add new instrument
            parsed_info = questionnaire._add_new_instrument(**parsed_info, show_selection=True)

        parsed_info.setdefault("vessel_code", "VESSEL")
        parsed_info.setdefault("cruise_number", "XX")
        obs_code = config.get_instrument_type_info(parsed_info["instrument_type"], param="observation_code")
        output_filename = (
            f"E{org_code}O2_{{vessel_code}}_{{cruise_number}}_{obs_code or 'H10'}.met".format_map(parsed_info)
        )
        output_path = data_dir / output_filename
        if output_path.exists():
            overwrite = ask(
                questionary.confirm,
                message=f"Файл {output_filename} уже существует. Перезаписать?",
                default=False,
            )

            if not overwrite:
                print("Операция отменена")
                return

        # Collect metadata
        parsed_info = questionnaire.collect_metadata(parsed_info, pos_df, all_files, output_filename)

        # Generate content
        met_content = generator.dict_to_met_content(parsed_info)

        # Write
        generator.write_met_file(output_path, met_content)

        # Save config
        config.save_config()

        logger.info("=" * 80)
        logger.info("Completed successfully!")
        logger.info(f"Output: {output_path}")
        logger.info("=" * 80)

        print(f"\n[OK] Создан: {output_path}")
        print(f"  Логи: {log_file}\n")

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        print("\n\n")

        choice = ask(
            questionary.select,
            message="Прервано. Что делать с введенными данными?",
            choices=["Сохранить и выйти", "Не сохранять и выйти", "Отмена (продолжить)"],
        )

        if choice == "Сохранить и выйти":
            config.save_config()
            print("[OK] Данные сохранены\n")
        elif choice == "Не сохранять и выйти":
            print("[CANCEL] Данные не сохранены\n")
        else:
            print("Продолжение...\n")
            return main()

    except Exception as e:
        logger.exception("Fatal error")
        print(f"\n[ERROR] Ошибка: {e}\n")
        raise


if __name__ == "__main__":
    main()
else:
    logger = logging.getLogger("METGenerator")
