"""Hierarchical weighted folder-name similarity, vendored for zero external deps.

Pure-python port of ``matcher.hierarchical_weighed_similarity`` (+ thresholds)
from the sibling ``cruises_organizer/match_dirs`` tool, used by the
``post_processing`` path-availability checkers. Only the pieces those tools
actually use are vendored (name similarity + confidence thresholds).
"""

import re
from difflib import SequenceMatcher
from typing import List

HIGH_CONFIDENCE_THRESHOLD: float = 0.70
LOW_CONFIDENCE_THRESHOLD: float = 0.30
NAME_SEPARATORS: str = r"[-.@]"


def get_name_parts(name: str) -> List[str]:
    if not name:
        return []
    parts = [p for p in re.split(NAME_SEPARATORS, name) if p]
    return parts if parts else [name]


def hierarchical_weighed_similarity(s1: str, s2: str) -> float:
    """Weighted similarity of separator-split name parts, 0.0–1.0.

    Earlier parts weigh more (``1/2**i + 0.1``); missing parts score ``0.9``.
    """
    parts1, parts2 = get_name_parts(s1), get_name_parts(s2)
    if not parts1 or not parts2:
        return 0.0
    max_parts, delta = max(len(parts1), len(parts2)), 0.1
    total, scored = 0.0, 0.0
    for i in range(max_parts):
        weight = (1.0 / (2**i)) + delta
        total += weight
        if i >= len(parts1) or i >= len(parts2):
            scored += weight * (1.0 - delta)
        else:
            scored += weight * SequenceMatcher(None, parts1[i].lower(), parts2[i].lower()).ratio()
    return scored / total if total > 0 else 0.0
