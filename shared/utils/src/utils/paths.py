import os
import re
import io
from pathlib import Path, PurePath
from collections.abc import Mapping, Iterable, Iterator
from functools import cache
from glob import escape as glob_escape
from string import Formatter
from typing import (
    Any,
    BinaryIO,
    TextIO,
    Final
)

from utils.init import dir_create_if_need

def glob_from_format_string_v1(format_string: str) -> str:
    """
    Convert a Python format string with named placeholders to a glob pattern.

    :param format_string: Format string with named placeholders like {key:spec}
    :return: Glob pattern with wildcards replacing format placeholders
    """
    # Pattern to match format specifiers: {name[:format_spec]}
    format_pattern: Final[re.Pattern[str]] = re.compile(r"\{(?P<name>\w+)(?:\:(?P<format>[^}]*))?\}")

    def _replace_match(match: re.Match[str]) -> str:
        format_spec: str | None = match.group("format")
        if format_spec is None:
            return "*"
        # Convert common format specs to appropriate glob patterns
        if "%y%m%d_%H%M%S" in format_spec or "%Y%m%d_%H%M%S" in format_spec:
            # Date-time format: matches exactly 15 characters (yy+mm+dd+HH+MM+SS+2 underscores)
            return "[0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9]"
        elif "d" in format_spec or "f" in format_spec:
            # Numeric formats
            return "[0-9]*"
        else:
            # Default wildcard for other format specs
            return "*"

    return format_pattern.sub(_replace_match, format_string)


# More thorough version below

_STRFTIME_GLOB: Final[Mapping[str, str]] = {
    "Y": "[0-9]" * 4,
    "y": "[0-9]" * 2,
    "m": "[0-1][0-9]",
    "d": "[0-3][0-9]",
    "H": "[0-2][0-9]",
    "I": "[0-1][0-9]",
    "M": "[0-5][0-9]",
    "S": "[0-5][0-9]",
    "f": "[0-9]" * 6,
    "j": "[0-3][0-9][0-9]",
    "U": "[0-5][0-9]",
    "W": "[0-5][0-9]",
    "V": "[0-5][0-9]",
    "z": "[-+]" + "[0-9]" * 4,
    "Z": "[A-Za-z]*",
    "p": "[AP]M",
    "%": "%",
}

_INT_CLASS: Final[Mapping[str, str]] = {
    "b": "[01]",
    "d": "[0-9]",
    "i": "[0-9]",
    "o": "[0-7]",
    "x": "[0-9a-f]",
    "X": "[0-9A-F]",
    "n": "[0-9]",
}

_ALT_PREFIX: Final[Mapping[str, str]] = {
    "b": "0b",
    "o": "0o",
    "x": "0x",
    "X": "0X",
}

_STRFTIME_RE: Final[re.Pattern[str]] = re.compile(r"%(?P<directive>.)|(?P<literal>[^%]+)|(?P<percent>%)")

_SPEC_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:(?P<fill>.)?(?P<align>[<>=^]))?"
    r"(?P<sign>[-+ ])?"
    r"(?P<hash>#)?"
    r"(?P<zero>0)?"
    r"(?P<width>\d+)?"
    r"(?P<group>[,_])?"
    r"(?:\.(?P<precision>\d+))?"
    r"(?P<type>[bcdeEfFgGinosxX%])?$"
)


@cache
def _one_or_more(char_class: str) -> str:
    return f"{char_class}{char_class}*"


@cache
def _strftime_to_glob(spec: str) -> str:
    return "".join(
        _STRFTIME_GLOB.get(d, "*")
        if (d := m["directive"])
        else glob_escape(m["literal"] or m["percent"] or "")
        for m in _STRFTIME_RE.finditer(spec)
    )


@cache
def _py_spec_to_glob(spec: str) -> str:
    if not (m := _SPEC_RE.match(spec)):
        return "*"

    g = m.groupdict()
    typ = g["type"]
    width = int(g["width"] or 0)
    zero_padded = bool(g["zero"] or g["fill"] == "0")

    if typ in _INT_CLASS:
        digit = _INT_CLASS[typ]
        inner = digit[1:-1]

        if prefix := (_ALT_PREFIX.get(typ, "") if g["hash"] else ""):
            return f"[-+ ]*{prefix}{_one_or_more(digit)}"

        if sep := g["group"]:
            return f"[-+ ]*{digit}[{inner}{sep}]*"

        if width and zero_padded:
            if width == 1:
                return f"[-+ ]*{_one_or_more(digit)}"
            first = "[-+ ]" if g["sign"] in {"+", " "} else f"[{inner}-]"
            return first + digit * (width - 1)

        return f"[-+ ]*{_one_or_more(digit)}"

    if typ == "c":
        return "?"

    if typ in {"f", "F"}:
        p = int(g["precision"]) if g["precision"] is not None else 6
        return ("*.[0-9]" + "[0-9]" * (p - 1)) if p else "*"

    if typ == "%":
        p = int(g["precision"]) if g["precision"] is not None else 6
        return ("*.[0-9]" + "[0-9]" * (p - 1) + "%") if p else "*%"

    if typ in {None, "s"} and g["precision"] == "0":
        return ""

    return "*"


@cache
def _spec_to_glob(spec: str | None) -> str:
    if not spec:
        return "*"
    if spec.startswith("%"):
        return _strftime_to_glob(spec)
    return _py_spec_to_glob(spec)


@cache
def format_to_glob(pattern: str) -> str:
    return "".join(
        glob_escape(literal) + (_spec_to_glob(spec) if field is not None else "")
        for literal, field, spec, _ in Formatter().parse(pattern)
    )



def open_csv_or_archive_of_them(
    filename: PurePath | Iterable[Path | str], binary_mode=False, ptn_re="", glob="", encoding=None
) -> Iterator[TextIO | BinaryIO]:
    """
    Opens and yields files from archive with name filename or from list of filenames in context manager (autoclosing).
    Note: Allows stop iteration over files in archive by assigning True to next() in consumer of generator
    Note: to can unrar the unrar.exe must be in path or set rarfile.UNRAR_TOOL
    :param filename: archive with '.rar'/'.zip' suffix or file name or Iterable of file names
    :param ptn_re: regex pattern to target file in the archive - should include directories if need search
    inside (for example place ".*" at beginning (* can be also used at begin: will be eplaced to the correct
    regex .*))
    :param glob: Unix shell style file glob in the archive - should include directories if need search inside (for example place "*" at beginning)
    :return:
    Note: RarFile anyway opens in binary mode
    """
    read_mode = "rb" if binary_mode else "r"
    if pattern := ptn_re:
        if ptn_re[0] == "*":
            ptn_re = f".{ptn_re}"

        def fun_match(pattern, text):
            return re.match(pattern, text)

    elif pattern := glob:

        def fun_match(pattern, text):
            return fnmatch(text_file, glob)

    else:
        pattern = ""

        def fun_match(pattern, text):
            return True

    # not iterates inside many archives so if have iterator then just yield them opened
    if hasattr(filename, "__iter__") and not isinstance(filename, (str, bytes)):
        for text_file in filename:
            if not fun_match(pattern, text):
                continue
            with open(text_file, mode=read_mode, encoding=encoding) as f:
                yield f
    else:
        filename_str = (
            filename.lower()
            if isinstance(filename, str)
            else str(filename).lower()
            if isinstance(filename, PurePath)
            else ""
        )

        # Find arc_suffix ('.zip'/'.rar'/'') and pattern if it is in filename after suffix
        for arc_suffix in (".zip", ".rar"):
            if arc_suffix in filename_str:
                filename_str_no_ext, pattern_parent = filename_str.split(arc_suffix, maxsplit=1)
                if pattern_parent:
                    pattern = str(PurePath(pattern_parent[1:]) / pattern)  # ? check and comment
                    filename_str = f"{filename_str_no_ext}{arc_suffix}"
                arc_files = [Path(filename_str).resolve().absolute()]
                break
            else:
                if arc_suffix in (pattern_lower := pattern.lower()):
                    pattern_arcs, pattern_lower = pattern_lower.split(arc_suffix, maxsplit=1)
                    pattern = pattern[-len(pattern_lower.lstrip("/\\")) :]  # recover text case for pattern
                    arc_files = Path(filename_str).glob(f"{pattern_arcs}{arc_suffix}")
                    arc_files = list(arc_files)
                    if not arc_files:
                        if (arc_found := Path(filename_str) / f"{pattern_arcs}{arc_suffix}").is_file():
                            arc_files = [arc_found]
                        else:
                            print(f'"{arc_found}" not found!')
                            return None
                    break

        else:
            arc_suffix = ""

        if arc_suffix:
            if arc_suffix == ".zip":
                from zipfile import ZipFile as ArcFile
            elif arc_suffix == ".rar":
                import rarfile

                # Set Your UnRAR executable
                rarfile.UNRAR_TOOL = r"C:\Programs\_catalog\TotalCmd\Plugins\arc\Rar64.exe"
                # r"c:\Programs\_catalog\TotalCmd\Plugins\arc\UnRAR.exe"
                if not Path(rarfile.UNRAR_TOOL).is_file():
                    raise FileNotFoundError("Set Your UnRAR executable")
                ArcFile = rarfile.RarFile
                try:  # only try increase performance
                    # Configure RarFile Temp file size: keep ~1Gbit free, always take at least ~20Mbit:
                    # decrease the operations number as we are working with big files
                    io.DEFAULT_BUFFER_SIZE = max(io.DEFAULT_BUFFER_SIZE, 8192 * 16)
                    import tempfile

                    import psutil

                    rarfile.HACK_SIZE_LIMIT = max(
                        20_000_000, psutil.disk_usage(Path(tempfile.gettempdir()).drive).free - 1_000_000_000
                    )
                except Exception as e:
                    l.warning("%s: can not update settings to increase peformance", standard_error_info(e))
                read_mode = "r"  # RarFile need opening in mode 'r' (but it opens in binary_mode)
            for path_arc_file in arc_files:
                with ArcFile(str(path_arc_file), mode="r") as arc_file:
                    for text_file in arc_file.infolist():
                        arc_filename_cor_enc = None
                        if not fun_match(pattern, text_file.filename):
                            # account for possible bad russian encoding
                            arc_filename_cor_enc = text_file.filename.encode("cp437").decode("CP866")
                            if fun_match(pattern, arc_filename_cor_enc):
                                pass
                            else:
                                continue

                        with arc_file.open(text_file.filename, mode=read_mode) as f:
                            if arc_filename_cor_enc:
                                # return file object with correct encoded name and all properties same as of f
                                f.name = arc_filename_cor_enc
                            break_flag = yield (
                                f
                                if binary_mode
                                else io.TextIOWrapper(
                                    f, encoding=encoding, errors="replace", line_buffering=True
                                )
                            )  # , newline=None
                            if break_flag:
                                print(f'exiting after opening archived file "{text_file.filename}":')
                                print(arc_file.getinfo(text_file))
                                break
        else:
            if not fun_match(pattern, filename_str):
                return
            with open(filename, mode=read_mode) as f:
                yield f


def name_output_file(
    dir_path: PurePath, filenameB, filenameE=None, bInteract=True, fileSizeOvr=0
) -> tuple[PurePath, str, str]:
    """
    Depreciated!
    Name output file, rename or overwrite if output file exist.
    :param dir_path: file directoty
    :param filenameB: file base name
    :param filenameE: file extention. if None suppose filenameB is contans it
    :param bInteract: to ask user?
    :param fileSizeOvr: (bytes) bad files have this or smaller size. So will be overwrite
    :return: (path_out, sChange, msgFile):
    - path_out: PurePath, suggested output name. May be the same if bInteract=True, and user
                answer "no" (i.e. to update existed), or size of existed file <= fileSizeOvr
    - sChange: user input if bInteract else ''
    - msgFile: string about resulting output name
    """

    # filename_new= re_sub(r"[^\s\w\-\+#&,;\.\(\)']+", "_", filenameB)+filenameE

    # Rename while target exists and it hase data (otherwise no crime in overwriting)
    msgFile = ""
    m = 0
    sChange = ""
    str_add = ""
    if filenameE is None:
        filenameB, filenameE = os.path.splitext(filenameB)

    def append_to_filename(str_add):
        """
        Returns filenameB + str_add + filenameE if no file with such name in dir_path
        or its size is less than fileSizeOvr else returns None
        :param str_add: string to add to file name before extension
        :return: base file name or None
        """
        filename_new = f"{filenameB}{str_add}{filenameE}"
        full_filename_new = dir_path / filename_new
        if not full_filename_new.is_file():
            return filename_new
        try:
            if os.path.getsize(full_filename_new) <= fileSizeOvr:
                msgFile = "small target file (with no records?) will be overwrited:"
                if bInteract:
                    print('If answer "no" then ', msgFile)
                return filename_new
        except Exception:  # WindowsError
            pass
        return None

    while True:
        filename_new = append_to_filename(str_add)
        if filename_new:
            break
        m += 1
        str_add = f"_({m})"

    if (m > 0) and bInteract:
        sChange = input(
            'File "{old}" exists! Change target name to "{new}" (Y) or update existed (n)?'.format(
                old=f"{filenameB}{filenameE}", new=filename_new
            )
        )

    if bInteract and sChange in ["n", "N"]:
        # update only if answer No
        msgFile = "update existed"
        path_out = dir_path / f"{filenameB}{filenameE}"  # new / overwrite
        writeMode = "a"
    else:
        # change name if need in auto mode or other answer
        path_out = dir_path / filename_new
        if m > 0:
            msgFile += f"{str_add} added to name."
        writeMode = "w"
    dir_create_if_need(dir_path)
    return (path_out, writeMode, msgFile)