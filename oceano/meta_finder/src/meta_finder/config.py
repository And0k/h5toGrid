"""
Configuration settings for TCM Metadata Processor.
"""

import argparse
import json
from dataclasses import asdict, dataclass, field, fields, MISSING
import logging
import re
from pathlib import Path
from typing import Optional, List, Tuple, Union, get_origin, get_args
import sys

# Global constants for metadata file names
DEVICES_FILE_NAME = "info_devices.json"
DEVICES_FILE_NAME_YAML = "info_devices.yaml"
DEVICES_FILE_NAME_UPD = f"{DEVICES_FILE_NAME.split('.', 1)[0]}@meta_finder.yaml"


@dataclass
class Config:
    # Command line options
    top_search_dirs: Tuple[Path, ...] = field(
        default_factory=lambda: tuple(
            Path(p)
            for p in [
                # r"B:\Cruises\BlackSea"
                # r"B:\Cruises\BalticSea"
                # "F:/WorkData/BalticSea"
                # "D:/Cruises/BalticSea"
                r"B:\Cruises\BalticSea\_Pregolya,Lagoon",
                # "F:/WorkData/BalticSea/_Pregolya,Lagoon",
            ]
        ),
        metadata={"help": "Search directories for cruises"},
    )
    input_dirs: Optional[List[Path]] = field(default=None, metadata={"help": "Specific cruise/device to process (overrides top_search_dirs)"})
    create_info_files: bool = field(default=False, metadata={"help": f"Update {DEVICES_FILE_NAME_UPD} files if they exist, or create new ones"})
    from_data: bool = field(default=True, metadata={"help": "Process metadata from data files. If True, extracts metadata from data files and combines with existing info-file metadata. If False, only uses metadata from existing info-files without extracting from data files"})
    extract_hdf5_times: bool = field(
        default=True, metadata={"help": "Extract time metadata from HDF5 files if not found in text_output"}
    )
    extract_hdf5_coef_dates: bool = field(default=False, metadata={"help": "Extract dates of coefficients in HDF5 files"})
    output_format: List[str] = field(default_factory=lambda: ["tsv"], metadata={"help": "Output formats to generate"})
    output_dir: Optional[Path] = field(default=None, metadata={"help": "Output directory for generated files"})
    overwrite_bad_devs_in_info_files: bool = field(
        default=True,
        metadata={
            "help": 'Update individual device entries in info files having all empty values '
            '("?", "-", or ""); preserves existing non-placeholder values and device order'
        },
    )
    # Burst detection settings
    max_burst_time_detection: int = field(
        default=10800,  # 3 hours in seconds (5400 lines * 2 seconds = 10800 seconds)
        metadata={
            "help": "Maximum time in seconds for burst detection analysis (default: 10800 = 3 hours). "
            "The code reads lines 1 and 20 to calculate time interval, then computes how many "
            "lines are needed to cover this time span."
        },
    )
    # Global setting for RAW HDF5 columns that should be extracted from _raw/*.h5 files
    raw_hdf5_cols: set = field(default_factory=lambda: {"coef_date", "raw_date_range"}, metadata={"help":
    "Output columns to trigger extraction of corresponding info from RAW HDF5 files (from _raw/*.h5)"})
    # Global logging level setting
    logging_level: Union[int, str] = field(default=logging.INFO, metadata={"help": "Global logging level setting (DEBUG, INFO, WARNING, ERROR, CRITICAL or their numeric values)"})
    # Global default averaging value for text files that don't specify averaging
    default_text_file_averaging: float = field(default=2.0001, metadata={"help": "Global default averaging value for text files that don't specify averaging"})
    # Cache configuration for file reading to minimize redundant file access
    cache_files_number: int = field(default=2000, metadata={"help": "Cache configuration for file reading to minimize redundant file access"})
    temp_dir: Optional[Path] = field(default=None, metadata={"help": "Temporary directory settings"})
    # Exclusion patterns - directories and HDF5 files matching these patterns will be skipped
    ptn_dir_exclude: List[str] = field(
        default_factory=lambda: [r".*-(?:\.|$)", "^bad$", "^test[^.]*$"],
        metadata={
            "help": "List of regex patterns for excluding directories and HDF5 files from processing. "
            "Entries matching any pattern will be skipped. Default excludes names ending with '-' "
            "and the special 'bad' name used as a marker for invalid data"
        },
    )
    # Supported file extensions
    extensions_text: set = field(
        default_factory=lambda: {".txt", ".tsv", ".csv"},
        metadata={
            "help": "Set of supported text file extensions for data files"
        }
    )
    extensions_archive: set = field(
        default_factory=lambda: {".zip", ".7z"},
        metadata={
            "help": "Set of supported archive extensions for compressed data files"
        }
    )
    extensions_hdf5: set = field(
        default_factory=lambda: {".h5", ".hdf5", ".mat"},
        metadata={
            "help": "Set of supported HDF5/MAT file extensions for data files"
        }
    )
    # Regex patterns (case-insensitive matching will be used)
    # Device directory keywords pattern. (device directory can instead or in addition contain
    # device_ids or only device types optionally with models)
    ptn_device_dir_keywords: str = field(
        default=r"inclinometers?|incl|tcm|wave_?gau?ges?|pressure|pres|@i[0-9]?",
        metadata={
            "help": "Device directory regex pattern is used to find device directories within cruise "
            "directories, also used to take GPX files with such keywords for analysis"
        },
    )
    # Common separators used in file / directory names before device ids/keywords (if ids after keywords, can
    # be after keywords, else keywords can be keywords)
    ptn_device_dir_sep: str = field(
        default=r"(?:[_+-]?[@#]|^_)",
        metadata={
            "help": "Regex pattern matching separator characters before device names in device directories"
            "matches separator characters before device name. The code also searches directories that begins "
            "from device name without separator"
        },
    )
    # Device type pattern for matching device type only (without model or number)
    # This pattern matches only the type part of device IDs, not the model or number
    # Note: Longer prefixes (incl, wg) are tried first to avoid partial matches.
    # matches abbreviations directly (becomes "i")
    ptn_device_type: str = field(
        default="i(?:ncl|nkl|)?|wg?",  # device id known types. optional: by default normalizes to "i"
        metadata={
            "help": "Regex pattern for matching device type only (without model or number). "
            "Matches patterns like i, w, incl, wg. "
            "Used as building block for other device patterns."
        },
    )
    # Device model pattern for matching optional model suffix
    ptn_device_model: str = field(
        default=r"P(?:res)|[Dbpw]",
        metadata={
            "help": "Regex pattern for matching optional device model suffix. "
            "Matches b (bottom) or p (pressure) or empty string. "
            "Used as building block for other device patterns."
        },
    )
    # Device number pattern for matching device number
    ptn_device_num: str = field(
        default=r"\d{1,5}",  #  5 digits max: 6 digits digits can be taken as date
        metadata={
            "help": "Regex pattern for matching device number. "
            "Matches one or more digits. "
            "Used as building block for other device patterns."
        },
    )


    # Flexible device identifier pattern for extracting device IDs part in text output filenames
    # to get all device ids with `parse_data_file_name.parse_filename_for_metadata()`.
    # Device type or model in 1st position is REQUIRED to correctly identify IDs part and exclude random files
    # Note: Pattern is constructed in __post_init__ by substituting building blocks. Do not use named
    # groups because it is combined with itself in parse_filename_for_metadata()

    ptn_devices_groups_part: str = field(
        default=(
            (
                lambda t, m, n: (
                    lambda pfx, ngr: (
                        r"\(?"  # opt. parenthesis around
                        r"{first}"  # первый элемент — prefix обязателен
                        r"(?:{sep}?\)?\(?{cont})*"  # остальные через sep, опц. скобки
                        r"\)?".format(
                            first=rf"(?:(?<=[@#])|{pfx})_?(?:{ngr}|{n})",
                            sep=rf"(?:[,;]|-(?={n}))",  # "-" только если за ним не 6-значная дата
                            cont=rf"(?:{pfx}_?(?:{ngr}|{n})|{n})",
                        )
                    )
                )(  # prefix: pfx = `type(_?model)? | model`  — хотя бы одно непустое,
                    # После prefix: num, или ( num-список ), или ничего (но тогда `(?!\w)`)
                    # Prefix lookahead: за префиксом обязана идти цифра / _( / конец слова
                    pfx=rf"(?:{t}(?:_?{m})?|{m})(?=_?\d|_?\(|(?!\w))",
                    ngr=rf"\({n}(?:[,;]{n})*\)",  # num group: (38,37,...)
                )
            )(**{v[0]: f"(?:{{ptn_device_{v}}})" for v in ("type", "model", "num")})
        ),
        metadata={
            "help": "Flexible regex pattern for matching device IDs in text output filenames. "
            "Handles single devices, comma-separated lists, ranges, and complex patterns "
            "including multiple device types (e.g., i3,5,9,w1-6). "
            "Constructed in __post_init__ by substituting building blocks."
        },
    )



    # ptn_devices_groups_part: str = field(
    #     default=(
    #         r"(({ptn_device_type}(?:_?\(?{ptn_device_model})?|{ptn_device_model}){ptn_device_num}?[,;()-]?"
    #         "(?:{ptn_device_type}?_?{ptn_device_model}?{ptn_device_num}(?:[,;()-]+|\.*|$))*)+"
    #     ),
    #     metadata={
    #         "help": "Flexible regex pattern for matching device IDs in text output filenames. "
    #         "Handles single devices, comma-separated lists, ranges, and complex patterns "
    #         "including multiple device types (e.g., i3,5,9,w1-6). "
    #         "Constructed in __post_init__ by substituting building blocks."
    #     },
    # )

    # '((?P<type>(?:i(?:ncl|nkl|)?|wg?))_?\\(?(?:[bpw]|Pres|)?\\d+?[,;()-]?(?:(?:i(?:ncl|nkl|)?|wg?)?(?:[bpw]|Pres|)?\\d+[,;()-])*)+'

    # GPX file search pattern
    # Can be empty string to disable GPX search, or a pattern to filter GPX files
    # Default: ptn_device_id to prefer GPX files with device identifiers
    ptn_search_gpx: str = field(
        default="",  # Will be set in __post_init__
        metadata={
            "help": "Pattern for filtering GPX navigation files. "
            "If empty, all GPX files are included. "
            "If set, only GPX files matching the pattern are included. "
            "Default uses ptn_device_id to prefer files with device identifiers."
        }
    )

    def __post_init__(self):

        data = asdict(self)

        # Add composite building blocks
        # not optional type required for device directories if it has no `ptn_device_dir_keywords`
        self.ptn_device_type_model = "(?:{ptn_device_type})_?(?:{ptn_device_model}|)".format_map(data)

        # Device identifier pattern for matching device IDs in file names
        # Constructed from from all optional parts for further analyzing
        self.ptn_device_id = (
            "(?:{ptn_device_type}|)_?(?:{ptn_device_model}|)_?0*(?:{ptn_device_num}|)".format_map(data)
        )
        self.ptn_device_id_named_parts = (
            "(?P<type>{ptn_device_type}|)_?(?P<model>{ptn_device_model}|)_?(?P<zeros>0*)"
            "(?P<number>{ptn_device_num}|)".format_map(data)
        )

        for field_name, field_obj in self.__dataclass_fields__.items():
            # Handle all fields with default_factory that might be None
            if (
                field_obj.default_factory is not None
                and field_obj.default_factory is not MISSING
                and getattr(self, field_name) is None
            ):
                setattr(self, field_name, field_obj.default_factory())
            elif field_name.startswith("ptn_"):
                value = getattr(self, field_name)
                # Only format fields that still contain template placeholders like {ptn_device_...}.
                # Skip fields already built by lambdas that contain regex quantifiers like {1,5}.
                if isinstance(value, str) and '{ptn_' in value:
                    setattr(self, field_name, value.format_map(data))

        # Pattern for matching dated prefixes in file names.
        # Supports date ranges like 130510..0708 where ".." separates start and end dates.
        self.glob_dated_dir = "[0-9][0-9][0-9][0-9][0-9][0-9]*"
        self.ptn_dated_prefix = r"\d{4,6}(?:\.\.\d{2,6})?"
        self.ptn_time_range = "_[0-9]+(?:-[0-9]+(?:_[0-9]+)?)?" # `_{time start}-{date end}_{time end}`
        # or r"(?:_\d{2,5})?(?:-\d{2,5})?(?:_\d{4})?"

        # Derive strict variant of ptn_devices_groups_part for directory discovery.
        # The relaxed version contains (?<=[@#])| lookbehind that allows bare numbers
        # after @/# without device type prefix (needed for filename parsing like
        # "191210#07,23,30,32-bin300s.zip"), but causes false positives in directory
        # discovery (e.g., "CTD_SST_48Mc#1253" matching as a device directory).
        self.ptn_devices_groups_part_strict = re.sub(
            r'\(\?<=\[@#\]\)\|', '', self.ptn_devices_groups_part
        )

        # Pattern for matching device directories and extracting device suffixes from names.
        # Structure: sep + device + boundary
        # - sep: optional separator before device part (e.g., _, @, #, or leading _)
        # - device: either keyword (kw) optionally followed by @type_or_groups,
        #           or (when no keyword) type_model or device_groups,
        #           or a broader comma-separated list that contains at least one known
        #           device type/model item (e.g., ADCP,CTD,i,w where i and w are known)
        # - boundary: ensures match ends at delimiter or end of string
        # When kw matches, type_or_groups is optional (consumed after kw if present).
        # When kw is absent, type_or_groups requires a separator context via lookbehind
        # (?<=[@#_+-]) to prevent false matches inside arbitrary words (e.g., "P" in "ABP53").
        # Uses strict variant of device groups to require device type prefix in directory names.
        # The broader alternative accepts non-standard device names (e.g., ADCP, CTD) mixed
        # with known types, requiring at least one item to match ptn_device_type_model or
        # ptn_devices_groups_part_strict to avoid false positives on arbitrary word lists.
        self.ptn_broader_dev_list = (
            r"(?:[A-Za-z]\w*[,;])*(?:{tm}|{gr})(?:[,;][A-Za-z]\w*)*".format(
                tm=self.ptn_device_type_model, gr=self.ptn_devices_groups_part_strict
            )
        )
        # Extend keyword pattern to allow comma-separated keyword lists
        # (e.g., "inclinometer,wavegage,any_other") where at least one item must
        # be a known keyword, but unknown items are allowed before/after.
        kw_base = data["ptn_device_dir_keywords"]
        self.ptn_device_dir_keywords_comma = (
            rf"(?:[A-Za-z]\w*[,;])*(?:{kw_base})(?:[,;_@#][A-Za-z]\w*)*"
        )
        self.ptn_device_dir_search = (
            r"(?P<sep>{ptn_device_dir_sep}|)"
            r"(?P<device>(?P<kw>{kw_comma})?"
            r"_?[@#]?(?(kw)(?:(?:{tm})|(?:{gr}))?|"
            r"(?:(?<=[@#_+-])(?:(?:{tm})|(?:{gr})|(?:{broader})))))"
            r"([-_()]|$)".format(
                **data,
                kw_comma=self.ptn_device_dir_keywords_comma,
                tm=self.ptn_device_type_model,
                gr=self.ptn_devices_groups_part_strict,
                broader=self.ptn_broader_dev_list,
            )
        )

        # Set default GPX search pattern to ptn_device_id if not specified
        if not self.ptn_search_gpx:
            self.ptn_search_gpx = self.ptn_device_id

        # Special handling for temp_dir since it has additional logic
        if self.temp_dir is None:
            self.temp_dir = Path(__file__).parent / "temp"
            # Create temp directory if it doesn't exist
            self.temp_dir.mkdir(exist_ok=True)

        # Convert logging level if it's a string
        self.logging_level = self._convert_logging_level(self.logging_level)


    @classmethod
    def create_argument_parser(cls):
        """Create and return an argument parser with all the configuration options."""
        parser = argparse.ArgumentParser(
            description="Application Configuration", formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

        # Add interactive mode argument
        parser.add_argument(
            "-i", "--interactive",
            action="store_true",
            help="Interactive mode: prompt for each Config setting in order. "
                "Enter '*' to use defaults for all remaining settings."
        )

        def _get_field_help(field, default_start="", default_end=""):
            return field.metadata.get("help", f"{default_start}{field.name.replace('_', ' ')}{default_end}")

        def _handle_bool(parser, field_name, field):
            parser.add_argument(
                f"--{field_name}",
                action="store_true",
                default=field.default,
                help=_get_field_help(field, default_start="Enable ", default_end=""),
            )
            # Add --no- variant for important flags
            if field.name in {"debug", "verbose", "from_data", "create_info_files"}:
                parser.add_argument(
                    f"--no-{field_name}",
                    action="store_false",
                    dest=field.name,
                    help=_get_field_help(field, default_start="Disable ", default_end=""),
                )

        def _handle_sequence(parser, field_name, field):
            # Determine the item type from the field annotation
            item_type = str  # Default to string
            origin = get_origin(field.type)
            if origin in (list, List, tuple, Tuple):
                args = get_args(field.type)
                if args:
                    item_type = args[0]

            # Handle Optional[List[Path]] - unwrap the Optional first
            if cls._is_optional_type(field.type):
                inner_type = cls._get_optional_type(field.type)
                inner_origin = get_origin(inner_type)
                if inner_origin in (list, List, tuple, Tuple):
                    inner_args = get_args(inner_type)
                    if inner_args:
                        item_type = inner_args[0]

            # Handle the case where default might be MISSING (for dataclass fields)
            from dataclasses import MISSING
            default_val = field.default
            if default_val is MISSING:
                parser.add_argument(
                    f"--{field_name}",
                    nargs="+",
                    type=item_type,
                    help=_get_field_help(field, default_end=" values"),
                )
            else:
                parser.add_argument(
                    f"--{field_name}",
                    nargs="+",
                    type=item_type,
                    default=default_val if default_val is not None else [],
                    help=_get_field_help(field, default_end=" values"),
                )

        def _handle_path(parser, field_name, field):
            # Handle the case where default might be MISSING (for dataclass fields)
            default_val = field.default
            if default_val is MISSING:
                parser.add_argument(
                    f"--{field_name}",
                    type=Path,
                    help=_get_field_help(field, default_end=" path"),
                )
            else:
                parser.add_argument(
                    f"--{field_name}",
                    type=Path,
                    default=default_val,
                    help=_get_field_help(field, default_end=" path"),
                )

        # Type handlers mapping
        type_handlers = {
            **{t: _handle_sequence for t in (tuple, Tuple, list, List)},
            bool: _handle_bool,
            Path: _handle_path,
        }

        for fld in fields(cls):
            if fld.name == "config_file":
                continue

            field_name = fld.name.replace("_", "-")
            field_type = cls._get_type_annotation(fld.type)

            # Check for specific type handlers
            handler_key = get_origin(fld.type) or field_type
            if handler_key in type_handlers:
                type_handlers[handler_key](parser, field_name, fld)
            elif cls._is_optional_type(fld.type):
                # Check if the inner type is a sequence type (e.g., Optional[List[Path]])
                inner_type = cls._get_optional_type(fld.type)
                inner_origin = get_origin(inner_type)

                # Route to sequence handler if inner type is list/tuple
                if inner_origin in (list, List, tuple, Tuple):
                    _handle_sequence(parser, field_name, fld)
                else:
                    # Handle other optional types (Optional[Path], Optional[str], etc.)
                    default_val = fld.default
                    if default_val is MISSING:
                        parser.add_argument(
                            f"--{field_name}",
                            type=inner_type,
                            help=fld.metadata.get('help', f"{field_name.replace('-', ' ')} value"),
                        )
                    else:
                        parser.add_argument(
                            f"--{field_name}",
                            type=inner_type,
                            default=default_val,
                            help=fld.metadata.get('help', f"{field_name.replace('-', ' ')} value"),
                        )
            else:
                # Default handler for primitive types
                # Handle the case where default might be MISSING (for dataclass fields)
                default_val = fld.default
                if default_val is MISSING:
                    # For fields with MISSING default, don't show the object representation
                    if field_name == "logging-level":
                        # Special handler for logging level to accept both string and numeric values
                        parser.add_argument(
                            f"--{field_name}",
                            type=str,  # Accept string input
                            help=fld.metadata.get('help', f"{field_name.replace('-', ' ')} value"),
                        )
                    else:
                        parser.add_argument(
                            f"--{field_name}",
                            type=field_type,
                            help=fld.metadata.get('help', f"{field_name.replace('-', ' ')} value"),
                        )
                else:
                    if field_name == "logging-level":
                        # Special handler for logging level to accept both string and numeric values
                        parser.add_argument(
                            f"--{field_name}",
                            type=str,  # Accept string input
                            default=default_val,
                            help=f"{field_name.replace('-', ' ')} value",
                        )
                    else:
                        parser.add_argument(
                            f"--{field_name}",
                            type=field_type,
                            default=default_val,
                            help=f"{field_name.replace('-', ' ')} value",
                        )

        # Add config file argument
        parser.add_argument("--config", type=Path, help="Configuration file path")

        return parser

    @classmethod
    def _prompt_interactive(cls):
        """Prompt user interactively for each Config setting in order.

        Returns:
            dict: Dictionary of field names to user-provided values
        """
        config_dict = {}
        print("\n=== Interactive Configuration Mode ===")
        print("Enter values for each setting. Enter '*' to use defaults for all remaining settings.\n")

        for fld in fields(cls):
            if fld.name == "config_file":
                continue

            field_name = fld.name
            field_type = cls._get_type_annotation(fld.type)
            # help_text = fld.metadata.get("help", f"{field_name.replace('_', ' ')}")

            # Get default value for display
            default_val = fld.default
            if default_val is MISSING:
                default_val = None
            elif fld.default_factory is not None and fld.default_factory is not MISSING:
                default_val = fld.default_factory()

            # Format default for display
            default_display = str(default_val) if default_val is not None else "None"
            if isinstance(default_val, (tuple, list)):
                if len(default_val) > 3:
                    default_display = f"[{default_val[0]}, {default_val[1]}, ..., {default_val[-1]}]"
                else:
                    default_display = str(default_val)

            # Prompt user
            prompt = f"{field_name} [{default_display}]: "
            user_input = input(prompt).strip()

            # Check for wildcard to use defaults for remaining
            if user_input == "*":
                print("\nUsing defaults for all remaining settings.")
                break

            # Skip if empty input (use default)
            if not user_input:
                continue

            # Parse input based on field type
            try:
                parsed_value = cls._parse_field_value(user_input, field_type, fld)
                config_dict[field_name] = parsed_value
            except ValueError as e:
                print(f"  Warning: Invalid value '{user_input}' for {field_name}: {e}")
                print(f"  Using default: {default_display}")
                continue

        return config_dict

    @classmethod
    def _parse_field_value(cls, value: str, field_type, field_obj):
        """Parse user input value based on field type.

        Args:
            value: String value from user input
            field_type: Expected type of the field
            field_obj: Field object from dataclass

        Returns:
            Parsed value in the correct type
        """
        # Strip whitespace and newlines from input value
        value = value.strip()

        # Single converter function that handles all types dynamically
        def convert(val: str, target_type):
            """Convert value to target type with appropriate error handling."""
            # Handle boolean with special parsing
            if target_type is bool:
                val_lower = val.lower()
                if val_lower in ("true", "t", "yes", "y", "1"):
                    return True
                if val_lower in ("false", "f", "no", "n", "0"):
                    return False
                raise ValueError("Expected boolean (true/false, yes/no, 1/0)")

            # Handle types that can be directly constructed from string
            if target_type in (int, float, Path):
                try:
                    return target_type(val)
                except (ValueError, TypeError):
                    raise ValueError(f"Expected {target_type.__name__}")

            # Default: return as-is for string or other types
            return val

        # Handle logging level (special case)
        if field_obj.name == "logging_level":
            return cls._convert_logging_level(value)

        # Handle sequence types (list, tuple)
        origin = get_origin(field_obj.type)
        if origin in (list, List, tuple, Tuple):
            # Split by whitespace or commas
            items = [item.strip() for item in value.replace(",", " ").split() if item.strip()]
            if not items:
                raise ValueError("Expected at least one value")

            # Get item type and convert items
            args = get_args(field_obj.type)
            item_type = args[0] if args else str

            # Convert items using the unified converter
            items = [convert(item, item_type) for item in items]

            # Return as list or tuple based on origin
            return tuple(items) if origin in (tuple, Tuple) else items

        # Handle optional types - unwrap and parse with the inner type
        # Use field_type parameter instead of field_obj.type to avoid recursion
        if cls._is_optional_type(field_type):
            inner_type = cls._get_optional_type(field_type)
            return convert(value, inner_type)

        # Handle primitive types with direct conversion
        return convert(value, field_type)

    @classmethod
    def from_args(cls):
        parser = cls.create_argument_parser()

        args = parser.parse_args()
        config_dict = vars(args)
        config_file = config_dict.pop("config", None)
        interactive = config_dict.pop("interactive", False)

        # Handle interactive mode
        if interactive:
            # Prompt user for configuration values
            interactive_config = cls._prompt_interactive()
            # Merge interactive config with any non-None values from command line
            # Interactive values take precedence
            config_dict = {k: v for k, v in config_dict.items() if v is not None}
            config_dict.update(interactive_config)
        else:
            # Remove None values to ensure default_factory is used
            config_dict = {k: v for k, v in config_dict.items() if v is not None}

        # Load from config file if provided
        if config_file and config_file.exists():
            config_from_file = cls._load_from_file(config_file)
            # Command line/interactive values take precedence over config file
            config_from_file.update(config_dict)
            config_dict = config_from_file

        # Convert logging level if it's a string
        if 'logging_level' in config_dict:
            config_dict['logging_level'] = cls._convert_logging_level(config_dict['logging_level'])

        return cls(**config_dict)


    @staticmethod
    def _get_type_annotation(field_type):
        if hasattr(field_type, "__origin__"):
            if get_origin(field_type) in (Union, type(None)):
                args = get_args(field_type)
                return next(t for t in args if t is not type(None))
        return field_type

    @staticmethod
    def _is_optional_type(field_type):
        origin = get_origin(field_type)
        return origin == Union and type(None) in get_args(field_type)

    @staticmethod
    def _get_optional_type(field_type):
        args = get_args(field_type)
        return next(t for t in args if t is not type(None))

    @staticmethod
    def _convert_logging_level(level):
        """Convert logging level from string to numeric value."""
        if isinstance(level, str):
            level = level.upper()
            level_map = {
                'DEBUG': logging.DEBUG,
                'INFO': logging.INFO,
                'WARNING': logging.WARNING,
                'WARN': logging.WARNING,
                'ERROR': logging.ERROR,
                'CRITICAL': logging.CRITICAL,
                'FATAL': logging.CRITICAL
            }
            return level_map.get(level, logging.INFO)  # Default to INFO if invalid level name
        return level

    @staticmethod
    def _load_from_file(config_file: Path) -> dict:
        with open(config_file) as f:
            config_dict = json.load(f)

        # Convert logging level if it's provided as a string in the config file
        if 'logging_level' in config_dict:
            config_dict['logging_level'] = Config._convert_logging_level(config_dict['logging_level'])

        return config_dict


# Check if we're running pytest to avoid argument parsing during import

# Global variable to store the config instance
_config_instance = None
_config_initialized = False

def _initialize_config():
    """Initialize the config instance."""
    global _config_instance, _config_initialized

    if _config_initialized:
        return

    # Check if we're running pytest or importing for tests
    # If pytest is in sys.argv or if we're in a test collection context
    is_pytest = any('pytest' in str(arg) for arg in sys.argv)

    # Check if help is requested before parsing arguments
    if not is_pytest and ("--help" in sys.argv or "-h" in sys.argv):
        # Create parser to show help using the shared method
        parser = Config.create_argument_parser()
        # Print help and exit
        parser.print_help()
        sys.exit(0)

    if is_pytest:
        # When running tests, create a default config without parsing args
        config = Config()
    else:
        # Otherwise, parse command line arguments
        try:
            config = Config.from_args()
        except SystemExit:
            # If argument parsing fails due to invalid arguments, let it propagate
            # This allows argparse to properly exit with error code when invalid arguments are provided
            raise
        except Exception:
            # If there's any other error in parsing, use default
            config = Config()

    # Check if pytables is available to enable HDF5 functionality
    try:
        import tables
        # pytables is available, so HDF5 processing is supported
        # extract_hdf5_times controls whether to actually perform the extraction
    except ImportError:
        # If pytables is not available and extract_hdf5_times is True, show warning
        if config.extract_hdf5_times:
            print(
                "WARNING: HDF5 extraction was configured as True but pytables is not available. "
                "HDF5 extraction disabled.",
                file=sys.stderr,
            )
        # HDF5 functionality will be disabled by the lack of the library at runtime

    _config_instance = config
    _config_initialized = True

def get_config():
    """Get or create the global config instance."""
    _initialize_config()
    return _config_instance

# Create the config instance using the lazy loading approach
config = get_config()

# Expose all config attributes at module level
# This makes attributes like config.logging_level accessible as module attributes
for attr_name in dir(config):
    if not attr_name.startswith('_'):
        globals()[attr_name] = getattr(config, attr_name)
