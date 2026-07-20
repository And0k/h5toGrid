import re
from datetime import timedelta
from typing import Dict, Tuple, Optional
import numpy as np
from datetime import datetime
from tcm import utils2init

lf = utils2init.LoggingStyleAdapter(__name__)

PROBE_WILDCARD = "*"


def pcid_to_raw_name(pcid: str):
    """
    :param pcid: Probe output Column ID (pcid):
    # {type}_{model}{number} or if no model then {type}{number}:  may be i/w for inclinometer/wave gage.
    # Inclinometers with pressure sensor currently have only model "p"
    :return: table name in **raw** or **not averaged** data DB
    """
    return f"incl{pcid[1:]}" if pcid[0] == "i" else pcid


# def format_timedelta(dt: timedelta) -> str:
#     days = dt.days
#     str_parts = [f"{days} days"] if days else []

#     seconds_remainder = dt.seconds
#     if seconds_remainder:
#         hours, seconds_remainder = divmod(seconds_remainder, 3600)
#         minutes, seconds = divmod(seconds_remainder, 60)
#         str_parts += [f"{hours:02}:{minutes:02}:{seconds:02}"]

#     return ' '.join(str_parts) if str_parts else 'no'


def str_dt(dt_s: float, lang="en"):
    """
    Time interval to readable string
    :param dt_s: time, s
    """
    if isinstance(dt_s, float):
        s=np.array(int(dt_s * 1000000), "M8[us]")
        a = np.int16((s.item().timetuple())[:6]) - [1970, 1, 1, 0, 0, 0]
        if ~np.any(a):
            a = [0, 0, 0, 0, 0, np.round(s.microsecond * 1e-06, 3)]
    else:
        a = np.int16((datetime.min + dt_s).timetuple())[:6] - [1, 1, 1, 0, 0, 0]

    out = " ".join([
        f"{d}{w}"
        for d, w in zip(
            a,
            ["лет", "месяцев", "дней", "ч", "мин", "с"]
            if lang == "ru"
            else ["years", "months", "days", "h", "min", "s"],
        )
        if d
    ])
    return out.strip()


def pcid_from_parts(type: Optional[str] = None, model: str = None, number: str|int = None, **kwargs):
    """
    Get Probe output Column ID (pcid)
    :param type: probe type, default: 'i' (inclinometers) or "" if model is "w" (wavegauges)
    :return: pcid string:
    If type, then place "_" between type and model, except return "i{number}" if type=model="i"
    """
    if type is None:  # model is text type
        type = "" if model == "w" else "i"

    if type == "i":
        if model:
            if model == "i":
                model = _ = ""  # not repeat "i"
            else:
                _ = "_"
        else:
            _ = ""
    elif type == "":
        _ = ""
    else:
        _ = "_"
    return f"{type}{_}{model}{number:0>2}"


def to_pcid_from_name(probe_name: str | int, probe_type: Optional[str] = None):
    """
    Get Probe output Column ID (pcid). This normalized ID has type (optional if "i") and model parts joined
    with "_"
    :param probe_name: any of
    - output column name
    - raw table name
    - raw file stem search pattern/name
    - pid str of 3 chars probe id or just number
    :param probe_type: probe type ('i' for inclinometers), used if `probe_name` is digit
    :return: pcid: our standardized probe name, used in
    - output column name in Avg/noAvg db
    - column suffix in combined table
    - name in specific for this probe hydra config `ConfigInMany_InclProc`
    """
    if isinstance(probe_name, str):
        probe_name0 = probe_name[0]
        if probe_name0.isdigit():
            return f"{probe_type or 'i'}{probe_name:0>2}"
        if len(probe_name) == 3 and probe_name[1:].isdigit():  # pid to pcid
            if probe_name0 == probe_type or not probe_type and probe_name0 in ["i", "w"]:
                return probe_name
            if probe_name0.isalpha():
                return f"{probe_type or 'i'}_{probe_name}"
    else:  # isinstance(probe_name, int):
        return f"{probe_type or 'i'}{probe_name:0>2}"

    if (pattern_name_parts := parse_name(probe_name.replace('.', ''))):
        return utils2init.call_with_valid_kwargs(pcid_from_parts, **pattern_name_parts)
    else:
        return '*'


def normalize_probes(ids: set[str]) -> set[str]:
    """Convert probe identifiers to canonical pcids via :func:`to_pcid_from_name`.

    Idempotent for 3-char pcids (e.g. ``"i01"`` → ``"i01"``).
    Non-standard prefixes like ``"p01"`` get normalized to ``"i_p01"``.
    The wildcard sentinel ``{"*"}`` is preserved as-is.

    :param ids: raw probe identifiers or ``{PROBE_WILDCARD}`` for "all".
    :raises ValueError: if any identifier normalizes to ``"*"`` (unresolvable name).
    :returns: set of normalized pcids (or ``{PROBE_WILDCARD}`` if input is wildcard).
    """
    if ids == {PROBE_WILDCARD}:
        return {PROBE_WILDCARD}
    result = set()
    for raw in ids:
        pcid = to_pcid_from_name(raw)
        if pcid == PROBE_WILDCARD:
            raise ValueError(
                f"probe identifier {raw!r} resolves to wildcard — provide explicit pcid like 'i01', 'i_p02'"
            )
        # Idempotency guard: double-application must not change the result
        if to_pcid_from_name(pcid) != pcid:
            raise ValueError(
                f"to_pcid_from_name is not idempotent for {raw!r} -> {pcid!r} -> {to_pcid_from_name(pcid)!r}"
            )
        result.add(pcid)
    return result


def track_probe_closure(b_input_is_h5, b_from_processed_db=False):
    """
    Initialize track_probe function: `track_probe = track_probe_closure()`
    b_input_is_h5: bool, show in message that raw data is from db
    """
    pcid_prev = None
    pcid_part = 1

    def track_probe(pcid):
        nonlocal pcid_part, pcid_prev
        probe_continues = (pcid == pcid_prev)
        if probe_continues:
            pcid_part += 1  # next part of same csv
        else:
            pcid_prev = pcid
            pcid_part = 1

        tbl_raw = pcid_to_raw_name(pcid)
        lf.warning(
            "{:s}{:s}{:s} data loaded. Processing...",
            tbl_raw,
            ""
            if b_input_is_h5 is None
            else (" proc noAvg db" if b_from_processed_db else " raw db")
            if b_input_is_h5
            else " csv",
            "" if not pcid_part else f" (part {pcid_part})",
        )
        # lf.warning(
        #     "{: 2d}. {:s} from {:s}{:s}",
        #     ipid,
        #     pcid,
        #     tbl_in,
        #     msg,
        # )
        return probe_continues


# ---------------------------------------------------------------------------
# Probe identity from filename (extracted from csv_specific_proc)
# ---------------------------------------------------------------------------

century = b"20"


def parse_name(name: str) -> Optional[Dict[str, str]]:
    """
    Extract logical parts of inclinometer / wave gauge name/glob from source raw csv file name.

    :param name: name/glob of source raw csv file (case-insensitive).
        All chars before 1st ``i``/``w``/``*``/``[`` are ignored.
        Chars ``*`` and ``[``, if exist, will be returned in type assuming it is a glob.
        ``comment`` captures the full trailing suffix after the probe number,
        to be preserved as ``-{comment}`` in the corrected filename.
    :return: dict with fields ``type``, ``model``, ``number``, ``comment``, ``chars0``..``chars2``,
        or ``None`` if not recognized.
    """
    name = name.lower()

    # 1. Regular file stem — captures full suffix as 'comment'
    m = re.match(
        r"[^iw]*(?P<type>[iw])(?P<chars1>((?:nkl|ncl|))_?)"
        r"(?P<model>[bdp]|[0bdp]{1,4}]|)(?P<chars2>_?0*)(?P<number>\d{1,4})(?P<comment>.*)",
        name,
    )
    if m:
        m = m.groupdict()
        m["chars0"] = ""
        return m

    # 2. Glob of file stems — chars3 for glob reconstruction (no comment)
    m = re.match(
        r"[^iw\*\[]*(?P<chars0>\*?\[?)(?P<type>[iw])(?P<chars1>((?:nkl|ncl|))_?\[?)"
        r"(?P<model>[bdpw]{0,4})(?P<chars2>\]?\*?0*)(?P<number>\d{0,4})(?P<chars3>\D*)",
        name,
    )
    if m:
        return m.groupdict()

    # 3. Unusual i/w (e.g. voln_v)
    m = re.match(
        r"@?(?P<type>voln_v)(?P<chars2>\D*0*)(?P<number>\d\d)(?P<comment>.*)", name
    )
    if m:
        m = m.groupdict()
        m["chars0"] = m["chars1"] = ""
        m["type"] = "w"
        m["model"] = ""
        return m

    return None


def probe_from_name(name: str) -> Optional[Tuple[str, int]]:
    """Extract ``(model, number)`` probe identity from filename via :func:`parse_name`.

    :param name: file stem or name (case-insensitive).
    :return: ``(model, number)`` or ``None`` if not a recognized probe.
    """
    parts = parse_name(name.lower())
    if not parts:
        return None
    model = parts.get("model", "") or parts.get("type", "")
    number = int(parts["number"]) if parts.get("number") else 0
    return model, number


def stem_to_pcid(stem: str) -> str:
    """Strip non-significant prefix/suffix from a file stem → pcid-only stem.

    The ``@`` character is a delimiter: everything **before** it is metadata
    (date stamp, annotation, …) and ignored for probe identity.  Everything
    after (the pcid stem) is kept, then the ``-{comment}`` suffix is dropped.

    Forms accepted:
      ``{pcid}``             → ``{pcid}``
      ``@{pcid}``            → ``{pcid}``
      ``{datestamp}@{pcid}`` → ``{pcid}``
      ``@{pcid}-{comment}``  → ``{pcid}``

    :param stem: file stem (may carry ``{prefix}@`` and/or ``-{comment}``).
    :return: pcid-only stem — no prefix before ``@``, no ``-{comment}`` suffix.
    """
    s = stem.rsplit("@", 1)[-1]  # drop {anything}@ prefix; keep everything after last @
    return s.split("-", 1)[0] if "-" in s else s