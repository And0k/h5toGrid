"""
Filename parsing functions for TCM Metadata Processor.
"""

import re
from typing import Dict, Any, List, Optional
from .logging_config import setup_logging
from meta_finder import config
logger = setup_logging()


def normalize_device_id(
    device_id: str,
    prefix: Optional[str] = None,
    validate: bool = False,
) -> Optional[str]:
    """Normalize {letters}{num} device ID:
    - letters: lowercase, strip _, incl/inkl -> i
    - num: strip leading zeros (keep single 0)
    :param prefix:
    - None:  derive from matched letters, fallback "i"
    - "":    return number only
    - str:   overwrite letters with prefix
    :param validate: if True, return None if neither type nor model present
    :return: normalized ID, or None if no match or validation fails
    """
    if not (
        device_id
        and (d := device_id.lower().strip())
        and (m := re.match(f"^{config.ptn_device_id_named_parts}$", d, re.IGNORECASE))
    ):
        return None

    if (prefix_found := "".join(v for k in ("type", "model") if ((v := v[0]) if (v := m[k]) else ""))):
        if prefix is None:
            prefix = prefix_found
    elif validate:  # validation: neither type nor model present
        return None
    elif prefix is None:
        prefix = "i"

    num = m["number"] or (m["zeros"] and "0") or ""
    return f"{prefix}{num}"


def expand_device_range(range_str: str, prefix: str = "") -> List[str]:
    """Expand range with optional prefix. Splits on `..` or `-` and normalizes each part."""
    if not (s := range_str.strip()):
        return []
    parts = re.split(r"\.\.|-", s)
    prefix_clean = prefix.replace("_", "").lower() if prefix else None
    if len(parts) == 2:
        if prefix_clean:
            num_st_en = [int(normalize_device_id(parts[i].strip(), "")) for i in [0, -1]]
        else:  # letters in parts
            st_end = [normalize_device_id(parts[i].strip()) for i in [0, -1]]
            m_st_end = [re.match(r"^([a-z]*)(\d+)$", d) for d in st_end]
            prefix_clean = m_st_end[0].group(1)
            num_st_en = [int(m.group(2)) for m in m_st_end]
        num_st_en[1] += 1
        return [f"{prefix_clean}{i}" for i in range(*num_st_en)]
    elif len(parts) > 2:
        logger.warning(f'Range must contain 1 to 2 digits, has {len(parts)} in "{range_str}"')

    if prefix_clean:  # letters are in prefix
        dev_id = normalize_device_id(s, "")
        return [f"{prefix_clean}{dev_id}"]
    else:
        return [normalize_device_id(s)]


# Unified pattern: optional prefix, then either (content) or first,rest
re_device_group = re.compile(
    r"""
    ^(?:(?P<prefix>[a-z_]+)|(?:))        # Optional prefix
    (?:                                  # Non-capturing group for alternation
        \(                               #   Either: opening paren
        (?P<content>[^)]+)               #   Content inside
        \)                               #   Closing paren
    |                                    #   OR
        (?P<first>\d+(?:-\d+|\.\.\d+)?)  # First number (optional range)
        [,;]?                            #   Separator
        (?P<rest>.*)                     #   Rest of items
    )$
""",
    re.VERBOSE | re.IGNORECASE,
)

def parse_device_group(group: str) -> List[str]:
    """
    Parse a device group string into a list of normalized device IDs.

    Args:
        group: Device group string supporting parentheses "i(38,37)", prefix propagation
            "i3,4,5" (comma equivalent to semicolon equivalent), ranges "i3-5" ("-" equivalent to ".."), and abbreviations "incl"/"inkl" -> "i".

    Returns:
        List of normalized device IDs (lowercase, no underscores, no leading zeros).
        See normalize_device_id() for normalization details.

    Examples:
    >>> for inp, exp in [
    ...     ("i(38,37,59,60,58)", ['i38', 'i37', 'i59', 'i60', 'i58']),
    ...     ("i_b(27,28,29,30)", ['ib27', 'ib28', 'ib29', 'ib30']),
    ...     ("(i03,i04,w5)", ['i3', 'i4', 'w5']),
    ...     ("i03,i04,i05", ['i3', 'i4', 'i5']),
    ...     ("i03,4,5", ['i3', 'i4', 'i5']),
    ...     ("i03-5,6", ['i3', 'i4', 'i5', 'i6']),
    ...     ("i_b20,i_b28,29,i_b40", ['ib20', 'ib28', 'ib29', 'ib40']),
    ...     ("ib27-30", ['ib27', 'ib28', 'ib29', 'ib30']),
    ... ]:
    ...     assert parse_device_group(inp) == exp, f"Failed for {inp}"
    """
    if not (g := group.strip()):
        return []

    if (m := re_device_group.match(g)):
        items = (
            [m["first"]] + (re.split(r"[,;]", rest) if (rest:=m["rest"]) else [])
            if (content := m["content"]) is None
            else re.split("[,;]", content)
        )
        return [x for it in items for x in expand_device_range(it, prefix=m["prefix"])]

    items = re.split(r"[,;]", g)
    return [x for it in items for x in expand_device_range(it)]


def split_top_level(s: str) -> list[str]:
    """
    Split string at top-level commas/semicolons, ignoring those inside parentheses.
    Also preserves prefix inheritance by not splitting when separator is followed by digit.

    :param s: example: "a, b(5, 5, 6), c(4, 5), d" or "i_b20,i_b28,29,i_b40"
    :return: for s examples: ['a', 'b(i5, 5, 6)', 'c(4, 5)', 'd'] or ['i_b20', 'i_b28,29', 'i_b40']
    """
    tokens, depth, start = [], 0, 0
    for i, ch in enumerate(s):
        if   ch == '(': depth += 1
        elif ch == ')': depth -= 1
        elif ch in ',;' and depth == 0 and not s[i+1].isdigit():
            tokens.append(s[start:i].strip())
            start = i + 1
    return tokens + [s[start:].strip()]


def parse_device_id_groups(devices_str, validate=True):
    """
    Parse a string of device IDs into a list of normalized device IDs.

    Args:
        devices_str: A string containing device IDs, e.g., "i03,i04,i05" or "i03-5,6"
        validate: If True, checks that device IDs type or model exists

    Returns:
        List of normalized device IDs, e.g., ['i3', 'i4', 'i5'] or ['i3', 'i4', 'i5', 'i6']
    """

    devices = []
    for group in split_top_level(devices_str):
        if (gr := group.strip()):  # why need strip?
            # Normalize all parsed device IDs and validate against device pattern
            for dev_id in parse_device_group(gr):
                if (device_id := normalize_device_id(dev_id, validate=validate)) is None:
                    logger.warning(
                        "Extracted device ID '%s' %s, skipping...",
                        dev_id,
                        "has no device type or model" if validate else "does not match expected device pattern",
                    )
                    continue
                devices.append(device_id)
    return devices


def parse_filename_for_metadata(filename: str) -> Dict[str, Any]:
    """
    Parse filename to extract metadata including device IDs and averaging interval.

    For complete device name pattern definitions, see:
    - config.ptn_device_id: Matches device type followed by digits (e.g., i01, w01, ib27)
    - README.md "Device Names" and "Text Data Filenames" sections

    Handles two possible orders of device and bin interval

    Args:
        filename: The filename to parse (e.g., "130510_1100-10_2319.txt",
        "191210#07,23,30,32-bin300s.zip", "200113_0000@i13.csv", "200113@i.tsv")

    Returns:
        Dictionary containing extracted metadata with the following keys:
        - "datetime": Date and time from filename (e.g., "130510_1100-10")
        - "averaging_interval": Binning interval in seconds (0 if not specified)
        - "devices": List of normalized device IDs (e.g., ["i7", "i23", "i30", "i32"])
        or ["*"] for combined files without specific devices
        - "device_id": Single device ID if only one device found (optional)

    Examples:
        - "191210#07,23,30,32-bin300s.zip" → devices: ["i7", "i23", "i30", "i32"], bin: 300
        - "210618_1600bin7200s@i.tsv" → devices: ["i"], bin: 7200
        - "200113_0000_i13.csv" → devices: ["i13"]
        - "200113_00@i13.csv" → devices: ["i13"]
        - "130510_1100-10_2319.txt" → devices: ["*"] (no device in filename,
        should use subdirectory fallback)

        # todo: use this file date/range as rough info to check/optimize data selection further
    """
    # Pattern building blocks for filename parsing
    # Each block is a reusable component that can be combined in different ways
    # Build extension pattern from config for text and archive files
    ext_ptn = '|'.join(
        re.escape(ext.lstrip('.')) for ext in (config.extensions_text | config.extensions_archive)
    )

    # Pattern for date and optional time range: YYMMDD[_hhmm][-dd][_hhmm]
    datetime_ptn = "^(?P<datetime>(?:{ptn_dated_prefix})?(?:{ptn_time_range})?)".format_map(config.__dict__)


    # Pattern for device separator: @, #, or _ (optional when device type/model present)
    device_sep_ptn = r"[@#_]?"

    # Pattern (2 names) for device identifier: either single letter type or full device pattern
    # This handles both "i" (single letter) and "i01" (full device ID)
    device_ptn = [
        f"(?P<devices{v}>(?:(?:{{ptn_devices_groups_part}})|{{ptn_device_type_model}}))|".format_map(
            config.__dict__
        ) for v in ("", "1")
    ]
    # Pattern (2 names) for bin interval: `bin<digits>[optional decimal]s`
    bin_interval_ptn = [rf"bin(?P<averaging_interval{v}>\d+(?:\.\d+)?)s?" for v in ("", "1")]

    # Combine all patterns into comprehensive filename pattern
    # Separator is optional: device type/model prefix is sufficient to identify the device part,
    # e.g. "180418_1000inclPres11.txt" parses as datetime=180418_1000, device=inclPres11 -> i11
    pattern = "".join((
        datetime_ptn,
        r"(?:(?:",  # Start non-capturing group for device and bin interval (two possible orders)
        rf"(?:{device_sep_ptn}{device_ptn[0]})?[-_]?(?:{bin_interval_ptn[0]})?",  # Order 1: device then bin
        r")|(?:",  # OR
        rf"(?:{bin_interval_ptn[1]})?(?:{device_sep_ptn}{device_ptn[1]})?",  # Order 2: bin then device
        r"))?",  # End non-capturing group for device and bin interval
        rf"\.(?:{ext_ptn})$",  # File extension from config (text and archive files)
    ))
    if not (match := re.match(pattern, filename, re.IGNORECASE)):
        return {}

    metadata = match.groupdict()
    # Assign metadata from alternative order fields (devices1 and averaging_interval1) if needed
    if metadata.get("devices") is None:
        metadata["devices"] = metadata.get("devices1")
    if metadata.get("averaging_interval") is None:
        metadata["averaging_interval"] = metadata.get("averaging_interval1")

    # Remove fallback fields after copying to avoid cluttering logs and output
    metadata.pop("devices1", None)
    metadata.pop("averaging_interval1", None)


    # Extract interval (can be integer or decimal, e.g., "2s" or "2.5s")
    if metadata["averaging_interval"] is not None:
        interval_str = metadata["averaging_interval"]
        # Try to convert to float first (handles both integer and decimal values)
        try:
            metadata["averaging_interval"] = float(interval_str)
        except ValueError:
            # If conversion fails, set to 0 as default
            metadata["averaging_interval"] = 0

    # Extract and process devices
    if not (devices_str := metadata.get('devices')):
        # Combined file with all devices
        metadata["devices"] = ["*"]
    # elif isinstance(devices_str, list):
    #     # Devices already processed as list
    #     pass
    elif len(devices_str) == 1 and devices_str in "iwp":
        # Prefix-only (i, w, or p)
        metadata["devices"] = [devices_str]
    elif ";" in devices_str or "," in devices_str:
        # Semicolon or comma-separated devices (handle ranges and multiple device types)
        # Strip trailing dashes that may have been matched by the pattern (e.g., "07,23,30,32-")
        devices_str = devices_str.rstrip("-.")
        # Replace semicolons with commas for uniform processing
        devices_str = devices_str.replace(";", ",")
        metadata["devices"] = devices = parse_device_id_groups(devices_str)

        if len(devices) == 1:
            metadata["device_id"] = devices[0]
    elif devices_str and any(c.isalnum() for c in devices_str):
        # Single device ID
        if not (device_id := normalize_device_id(devices_str.rstrip(".-"))):
            logger.debug(f"Failed to parse device ID from {devices_str}")
            device_id = '*'
        metadata["devices"] = [device_id]
        metadata["device_id"] = device_id
    else:
        # No device info - combined file
        metadata["devices"] = ["*"]
    return metadata


def extract_device_ids_from_prefixed_name(devices_with_prefix_str: str, msg_what="") -> list[str]:
    """Extract device IDs from a prefixed name (HDF5 group, directory, etc.).

    If ``@`` is present in the input, devices are extracted only from the part
    after the first ``@`` separator.  Otherwise the optional dated prefix and
    ``[@#_-]`` separators are stripped first.

    Args:
        devices_with_prefix_str: string containing device identifiers
        msg_what: prefix for log/warning messages

    Returns:
        List of normalized device IDs, or empty list if none found.

    Examples:
        i54bin2s -> [i54]
        201202P1-5,I1-2@i3,5,w1-6 -> [i3, i5, w1, ..., w6]
        i or 1 or #1 or #1.txt -> []
    """
    # When @ or # is present, extract devices only after the first such separator
    # Remove optional dated prefix and track if separator prefix was present
    match_prefix = re.match(
        rf"^(?:[^@#]*[@#])|(?:(?:{config.ptn_dated_prefix})?[_-]?)", devices_with_prefix_str, re.IGNORECASE
    )
    devices_str = devices_with_prefix_str[match_prefix.end():] if match_prefix else devices_with_prefix_str

    # Extract the full matched device part suffix
    if not (match := re.match(config.ptn_devices_groups_part, devices_str, re.IGNORECASE)):
        return []
    dev_groups_validated = match.group()
    try:
        return parse_device_id_groups(dev_groups_validated)
    except Exception as e:
        logger.debug(
            f"Failed to parse {msg_what}devices from {devices_with_prefix_str} "
            f"(from its {dev_groups_validated} part): {e}"
        )
        return []
