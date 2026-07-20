import logging
import re
from typing import MutableMapping, Dict, Tuple, Optional
from pathlib import PurePath

from .logging_config import setup_logging
from . import config

logger = setup_logging()


# Number of date components and their cumulative character lengths
# 0 components = "" (0 chars), 1 component = YYMM (4 chars), 2 components = YYMMDD (6 chars)
_N_DATE_COMPONENTS = 2
_DATE_COMPONENT_FMTS = ("{YY}{MM}", "{DD}")
_DATE_COMPONENT_CUM_LENS = (0, 4, 6)


def _strip_date_prefix(name: str) -> str:
    """Strip leading date digits (2-6) and optional separator (_ or /) from a dataset name."""
    match = re.match(r"^(?:\d{2,6}[_/]?)?(.+)$", name)
    return match.group(1) if match else name


def parse_dated_dir(dir_name) -> Dict[str, str]:
    """Parse dated directory name and return it as a dictionary with date keys (YY, MM, DD) and rest.

    Recognized formats (tried in order):
    - YYMMDD (6 digits) — full date
    - YYMM (4 digits) — year and month only, DD defaults to empty string
    - YYYY-MM[-DD] — ISO date with optional day
    """
    for date_ptn in [
        r"(?P<YY>\d{2})(?P<MM>\d{2})(?P<DD>\d{2})(?P<rest>.*)",
        r"(?P<YY>\d{2})(?P<MM>\d{2})(?P<rest>.*)",  # YYMM only (4 digits)
        r"\d{2}(?P<YY>\d{2})-(?P<MM>\d{2})(-(?P<DD>\d{2}))?(?P<rest>.*)",  # ISO date with optional `-DD`
    ]:
        if (match := re.match(date_ptn, dir_name)):
            break
    else:
        raise ValueError("Cruise directory must start with date YYMMDD, YYMM, or YYYY-MM")
    result = match.groupdict()
    result.setdefault("DD", "")
    return result


def _iter_path_parts_with_dates(path):
    """
    Yield (part_str, date_parts_or_none) for each path component from deepest to shallowest.

    For dated parts, yields the parsed date dict. For non-dated parts, yields the raw string.
    Each path component is yielded exactly once.
    """
    for part in reversed(path.parts):
        try:
            date_parts = parse_dated_dir(part)
        except ValueError:
            date_parts = None
        yield part, date_parts


def strip_devices_sfx_and_get_type0(name: str) -> Tuple[str, str | None]:
    """
    Strip device keywords/IDs suffix from `name`.

    Search for device pattern (keyword or type+model) first, then fall back to `@`/`#` separator.
    When keyword (``kw`` group) is present, split at the separator/keyword position to strip both
    the keyword and any following device type. Return (prefix, device_type_first_char) where prefix
    is everything before the match, or (name, None) if nothing matched.
    device_type_first_char is the first letter of the matched device type, or None if not identified.
    """
    # Find all matches and prefer the one where kw (keyword) group is captured.
    # This ensures we split on keyword position when present, not just on @# position
    # where re.search would find a shorter valid match first.
    kw_match = None
    first_match = None
    for match in re.finditer(config.ptn_device_dir_search, name, re.IGNORECASE):
        if match["kw"]:
            kw_match = match
            break  # keyword match takes priority over type-only matches
        if first_match is None:
            first_match = match

    match = kw_match or first_match
    if match:
        # When kw matched, split at separator position (before keyword) to strip both
        # the keyword and any following device type after @#
        if match["kw"]:
            cut = match.start("sep") if match["sep"] else match.start("kw")
        else:
            cut = match.start(0)
        out_name = name[:cut].strip(";,._ ")
        # Extract device type: from kw match's device group, or from type-only match
        dev_t = match["device"]
        return out_name, (dev_t.lstrip("@#")[0] if dev_t else None)

    # Fall back to bare @ or # separator (e.g., "test@device" with no keyword/type match)
    if (match := re.search(r"[@#]", name)):
        out_name = name[: match.start(0)].strip(";,._ ")
        return out_name, None
    # No device pattern found — return name as-is with no device type
    return name.strip(";,._ "), None


def _build_dataset_name(date_components: list[str], right_part: str, sep_char: str | None = None) -> str:
    """Build dataset name from date prefix components and right part.

    Args:
        date_components: list of date component strings (e.g. ["21", "07"]).
        right_part: the non-date portion of the dataset name (cruise name, device comment, etc.).
        sep_char: separator between date prefix and right_part. ``None`` (default) selects
            ``"_"`` when right_part contains digits, ``""`` otherwise. Pass ``"/"`` for
            digitless cruise names. Device type fallback (``@type``) never gets a separator
            regardless of this argument.
    """
    date_prefix = "".join(date_components)
    if not right_part:
        return date_prefix
    # Device type fallback (right_part starts with "@") never gets a separator
    if right_part.startswith("@"):
        return f"{date_prefix}{right_part}" if date_prefix else right_part
    if sep_char is None:
        sep_char = "_" if any(ch.isdigit() for ch in right_part) else ""
    sep = sep_char if date_prefix else ""
    return f"{date_prefix}{sep}{right_part}"


def _get_number_of_date_parts(name: str, len_right_part: int) -> int:
    """Determine how many date components (0, 1, or 2) a dataset name uses based on its length."""
    date_len = len(name) - len_right_part
    # Account for optional separator (_ or /) between date prefix and right part
    if date_len > 0 and name[date_len - 1] in ("_", "/"):
        date_len -= 1
    try:
        return _DATE_COMPONENT_CUM_LENS.index(date_len)
    except ValueError:
        # Fallback: find the largest cumulative length that is <= date_len
        return max(i for i, cum_len in enumerate(_DATE_COMPONENT_CUM_LENS) if cum_len <= date_len)


def _infer_min_day(device_dir: PurePath, processed_meta: dict) -> Optional[str]:
    """Infer the earliest day (DD) from extracted device time metadata for a device directory.

    Scans all device entries' ``time_st`` fields and returns the minimum day string.
    Returns None when no valid day can be extracted.
    """
    for dev_data in processed_meta.get(device_dir, {}).values():
        time_st = dev_data.get("time_st", "?")
        if time_st and time_st not in ("?", "", None, "-"):
            try:
                day_str = str(time_st)[4:6]
                if day_str.isdigit():
                    return day_str
            except (IndexError, ValueError):
                pass
    return None


def add_dataset_name(
    device_dir: PurePath,
    cruise_dir: PurePath,
    used_datasets_paths: MutableMapping[str, PurePath] = None,
    processed_meta: dict = None,
) -> Tuple[str, str]:
    """
    Extract dataset name: `{[YYMM][DD]}{[_]cruise}[/{device dir comment suffix}|@{device_type}]`
    - [YYMM][DD]: date from `device_dir` or if not found then cruise dir date. The number of date parts
        increases from 0 to the same number for each dataset name that have equal next parts, that is required
        to make them unique. From 1 (`YYMM`) if cruise name (`cruise`) contain no digits.
    - cruise: cruise name extracted from `cruise_dir` name without date prefix / suffix of device dir keywords
        and device_ids (see `config.ptn_*`). The separator `_` from date parts is used only if cruise name
        contain digits.
    - opt. device dir comment suffix: device dir name without date prefix / suffix of device dir keywords and
        device_ids
    - device_type: device type from device dir name. Used only if no other parts except date. If no device
        type too, then dataset name should contain all date parts.

    Args:
        cruise_dir: cruise directory path
        device_dir: device directory path. Assumed that it is under parent of cruise_dir.
        used_datasets_paths: mapping of already used datasets names to path of device directory (first is used
        as key to remove duplicates by changing it in specific way, later is used to get dataset time and as
        outer unique key). Assumed that it is under parent of cruise_dir.

    Returns:
        Tuple of (unique dataset name, date string YYMMDD from device dir or cruise dir)

    Updates:
    - `used_datasets_paths` with
        - new dataset name and device dir path,
        - modifies each dataset name that has same right part with prefix of required number of date parts
        (YYMM, DD) to maintain unique names and same size of date prefix for these datasets.

    Examples:
    `cruise_dir`+(`used_datasets_paths` right part keys equal to result) -> result keys # case handled:
    - `201202_BalticSpit` -> `2012/BalticSpit` # cruise name without digits, "/" separator
    - `201102_BalticSpit`+(`20/BalticSpit`) -> `2011/BalticSpit`, `2012/BalticSpit` # same right part
    - `230616_Kulikovo`+(`230825/Kulikovo`) -> `230616Kulikovo`, `230825Kulikovo`  # same right part
    - `230507_ABP53_inclinometer@i3,4,15,19,37,38;ib27-30,ip6` -> `ABP53` # device dir keywords / device_ids
    - `221103@ib26,28-30` -> `2211@i` # directories without cruise name
        Device directories examples:
    - `201202_BalticSpit/inclinometer/201202P1-5,I1-2@i3,5,9,10,11,15,19,23,28,30,32,33,w1-6` -> `2012/BalticSpit/P1-5,I1-2`
    """
    if used_datasets_paths is None:
        used_datasets_paths = {}

    # Walk path parts from deepest to shallowest to extract date and named parts.
    rel_path = device_dir.relative_to(cruise_dir.parent)
    date_components: list[str] = []
    cruise_parts: list[str] = []
    dev_type: str | None = None

    for i, (part_str, date_parts_and_rest) in enumerate(_iter_path_parts_with_dates(rel_path)):
        if date_parts_and_rest:
            # First dated part found (deepest = device dir) provides the date
            if not date_components:
                date_components = [fmt.format_map(date_parts_and_rest) for fmt in _DATE_COMPONENT_FMTS]
            name_rest = date_parts_and_rest["rest"]
        else:
            name_rest = part_str

        # Strip device suffix to get the meaningful name part.
        # Skip purely numeric remainders — they are date-range endpoints (e.g. "..0708" in "130510..0708@i")
        # not meaningful cruise/device names.
        use_part, dev_type_new = strip_devices_sfx_and_get_type0(name_rest)
        if use_part and not use_part.isdigit():
            cruise_parts.append(use_part)
        if not dev_type and dev_type_new:
            dev_type = dev_type_new

    # Build the right part: cruise_parts joined with "/" or fallback to @device_type
    if (cruise_parts := list(reversed(cruise_parts))):
        right_part = "/".join(cruise_parts)
        has_no_digits = not any(ch.isdigit() for ch in cruise_parts[0])
    else:
        right_part = f"@{dev_type}" if dev_type else ""
        has_no_digits = False

    # Collect existing entries sharing the same right_part for disambiguation
    same_right: dict[str, PurePath] = {}
    min_date_parts = 0
    for name, path in used_datasets_paths.items():
        if _strip_date_prefix(name) != right_part:
            continue
        n_old = _get_number_of_date_parts(name, len(right_part))

        # Only consider entries with the same year for disambiguation
        if n_old:
            year_old = name[:2]
        else:
            # Extract year from the path's date parts
            date_dict = next(
                (
                    date_p
                    for _, date_p in _iter_path_parts_with_dates(path.relative_to(cruise_dir.parent))
                    if date_p and date_p["YY"]
                ),
                None,
            )
            year_old = date_dict["YY"] if date_dict else "00"
        if year_old != date_components[0][:2]:
            continue

        same_right[name] = path
        min_date_parts = max(min_date_parts, n_old)

    # Determine minimum date prefix length:
    # - 2 (YYMM, DD) if no cruise_parts except date or device id/keywords
    # - 1 if disambiguation needed (same_right non-empty) or if cruise name has no digits
    # - 0 otherwise (cruise name with digits, no conflicts)
    min_date_parts = max(
        2 if not cruise_parts else 1 if (has_no_digits or same_right) else 0,
        min_date_parts,
    )

    # Separator between date prefix and right_part:
    # - "/" when cruise name has no digits (cleaner visual separation, e.g. "1305/Sambian")
    # - "_" when cruise name contains digits (legacy disambiguation, e.g. "2306_AI55")
    # - never for device type fallback "@{type}" — handled inside _build_dataset_name
    sep_char = "/" if has_no_digits and bool(cruise_parts) else None

    def _displace_parent_to_min(
        entry_a_path: PurePath, entry_b_path: PurePath
    ) -> bool:
        """Try to free a colliding name by moving the parent (shallower) path to its minimal name.

        The parent is the entry with fewer path parts. Its name is rebuilt with progressively
        fewer date parts (removing DD first, then YYMM) until a free name is found.
        Returns True if displacement succeeded.
        """
        parent_path = (
            entry_a_path if len(entry_a_path.parts) <= len(entry_b_path.parts) else entry_b_path
        )
        parent_date = next(
            (
                dp
                for _, dp in _iter_path_parts_with_dates(parent_path.relative_to(cruise_dir.parent))
                if dp and dp["YY"]
            ),
            None,
        )
        if not parent_date:
            return False
        parent_old_name = next(
            (n for n, p in list(used_datasets_paths.items()) if p == parent_path), None
        )
        if not parent_old_name:
            return False
        # Try progressively shorter date prefixes: current-1, current-2, ..., 0
        parent_n = _get_number_of_date_parts(parent_old_name, len(right_part))
        for try_n in range(parent_n - 1, -1, -1):
            try_parts = [_DATE_COMPONENT_FMTS[i].format_map(parent_date) for i in range(try_n)]
            rebuilt = _build_dataset_name(try_parts, right_part, sep_char=sep_char)
            if rebuilt and rebuilt not in used_datasets_paths:
                del used_datasets_paths[parent_old_name]
                used_datasets_paths[rebuilt] = parent_path
                if parent_old_name in same_right:
                    del same_right[parent_old_name]
                same_right[rebuilt] = parent_path
                logger.info(
                    f"Displaced parent {parent_old_name!r} -> {rebuilt!r} "
                    f"(reduced to {try_n} date parts)"
                )
                return True
        return False

    def _make_unique_with_date_suffix(
        base_name: str, yymm: str, dd: str, registry: MutableMapping[str, PurePath]
    ) -> str:
        """Append date-based suffix to `base_name` until unique in `registry`.

        Uses ``/DD`` when YYMM matches the cruise (already in prefix), or
        ``/YYMMDD`` when the month differs and provides additional distinction.
        Falls back to incrementing numeric suffix if the date-based name also collides.
        """
        for suffix_parts in ([dd], [yymm, dd]):
            candidate = f"{base_name}/{''.join(suffix_parts)}"
            if candidate not in registry:
                return candidate
        suffix = 1
        while f"{base_name}_{suffix}" in registry:
            suffix += 1
        return f"{base_name}_{suffix}"

    # Increase date prefix length until the candidate name is unique.
    # After renaming same-right entries, try displacing any parent that still
    # blocks the candidate before escalating to more date parts.
    dataset_name = ""
    for n_new in range(min_date_parts, _N_DATE_COMPONENTS + 1):
        dataset_name = _build_dataset_name(date_components[:n_new], right_part, sep_char=sep_char)

        # Rename all same-right entries to use at least n_new date prefix parts.
        # When a rename would collide with a deeper (child) entry, displace the
        # shallower (parent) to fewer date parts instead of overwriting the child.
        for name, path in list(same_right.items()):
            if (n_old := _get_number_of_date_parts(name, len(right_part))) >= n_new:
                continue
            if name not in used_datasets_paths:
                continue
            # Rebuild the renamed entry with the correct date prefix length
            date_dict = next(
                (
                    date_p
                    for _, date_p in _iter_path_parts_with_dates(path.relative_to(cruise_dir.parent))
                    if date_p and date_p["YY"]
                ),
                None,
            )
            if date_dict:
                new_date_parts = [fmt.format_map(date_dict) for fmt in _DATE_COMPONENT_FMTS[:n_new]]
                renamed = _build_dataset_name(new_date_parts, right_part, sep_char=sep_char)
                entry_yymm = _DATE_COMPONENT_FMTS[0].format_map(date_dict)
                entry_dd = date_dict.get("DD", "")
            else:
                old_cum = _DATE_COMPONENT_CUM_LENS[n_old]
                new_cum = _DATE_COMPONENT_CUM_LENS[n_new]
                renamed = f"{name[:old_cum]}{'0' * (new_cum - old_cum)}{name[old_cum:]}"
                entry_yymm, entry_dd = "", ""
            # Handle collision: when renamed clashes with an existing entry
            if renamed in used_datasets_paths and used_datasets_paths[renamed] != path:
                occupant_path = used_datasets_paths[renamed]
                # If this entry is the parent (shallower), displace it to fewer
                # date parts rather than overwriting the child.
                if len(path.parts) <= len(occupant_path.parts):
                    if _displace_parent_to_min(path, occupant_path):
                        # Parent displaced — skip inserting under the colliding name.
                        continue
                    # Parent can't be displaced further — leave it at current prefix.
                    # The child keeps the longer prefix; parent keeps the shorter one.
                    continue
                # Child colliding with parent — fall back to date suffix.
                renamed = _make_unique_with_date_suffix(
                    renamed, entry_yymm, entry_dd, used_datasets_paths
                )
                logger.warning(
                    f"Rename collision for {path}: using {renamed!r} (date suffix)"
                )
            used_datasets_paths.pop(name, None)
            used_datasets_paths[renamed] = path
            if name in same_right:
                del same_right[name]
            same_right[renamed] = path
        # After renaming, try displacing any occupant that still blocks the candidate.
        # This handles the case where a parent was renamed to the same name the child needs.
        if dataset_name in used_datasets_paths:
            occupant = used_datasets_paths[dataset_name]
            if len(device_dir.parts) > len(occupant.parts):
                # Current entry is deeper (child) — displace the shallower occupant
                _displace_parent_to_min(occupant, device_dir)

        if dataset_name not in used_datasets_paths:
            break
    else:
        # All date components exhausted — try displacement, then date suffix
        if dataset_name in used_datasets_paths:
            occupant_path = used_datasets_paths[dataset_name]
            yymm, dd = date_components[0], date_components[1] if len(date_components) > 1 else ""
            if not _displace_parent_to_min(device_dir, occupant_path):
                dataset_name = _make_unique_with_date_suffix(dataset_name, yymm, dd, used_datasets_paths)
            logger.warning(
                f"All date components exhausted for {device_dir}: using {dataset_name!r}"
            )

    used_datasets_paths[dataset_name] = device_dir

    # When folder name has only YYMM (4 digits), try to infer DD from extracted time metadata.
    date_str = "".join(date_components)
    if len(date_str) == 4 and processed_meta and (min_day := _infer_min_day(device_dir, processed_meta)):
        date_str += min_day

    return dataset_name, date_str
