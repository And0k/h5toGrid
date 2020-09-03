#! /usr/bin/env python3.6
# -*- coding: utf-8 -*-
"""
  Purpose:  helper functions for input/output handling
  Author:   Andrey Korzh <ao.korzh@gmail.com>
  Created:  2016 - 2019
"""
# from __future__ import print_function
import sys

try:
    if sys.version_info[0] == 2:  # PY2?
        from future.standard_library import install_aliases

        install_aliases()
except Exception as e:
    print('Not found future.standard_library for install_aliases()!')
    print('So some functions in utils2init may not work')

from os import path as os_path, listdir as os_listdir, access as os_access, R_OK as os_R_OK, W_OK as os_W_OK
from ast import literal_eval
from fnmatch import fnmatch
from datetime import timedelta, datetime
from codecs import open
import configparser
import configargparse
import re
from pathlib import Path, PurePath
import io
from functools import wraps
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Iterable, Iterator, BinaryIO, TextIO, TypeVar, Tuple, Union
import logging

A = TypeVar('A')


def fallible(*exceptions, logger=None) \
        -> Callable[[Callable[..., A]], Callable[..., Optional[A]]]:
    """
    Decorator (very loosely inspired by the Maybe monad and lifting)
    :param exceptions: a list of exceptions to catch
    :param logger: pass a custom logger; None means the default logger,
                   False disables logging altogether.

    >>> @fallible(ArithmeticError)
    ... def div(a, b):
    ...     return a / b
    ... div(1, 2)
    0.5


    >>> res = div(1, 0)
    ERROR:root:called <function div at 0x10d3c6ae8> with *args=(1, 0) and **kwargs={}
    Traceback (most recent call last):
        ...
    File "...", line 3, in div
        return a / b

    >>> repr(res)
    'None'
    """

    def fwrap(f: Callable[..., A]) -> Callable[..., Optional[A]]:

        @wraps(f)
        def wrapped(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except exceptions:
                (logger or logging).exception('called %s with *args=%s and **kwargs=%s', f, args, kwargs)
                return None

        return wrapped

    return fwrap


def standard_error_info(e):
    msg_trace = '\n==> '.join((s for s in e.args if isinstance(s, str)))
    return f'{e.__class__}: {msg_trace}'


class Ex_nothing_done(Exception):
    def __init__(self, msg=''):
        self.message = f'{msg} => nothing done. For help use "-h" option'


class Error_in_config_parameter(Exception):
    pass


readable = lambda f: os_access(f, os_R_OK)
writeable = lambda f: os_access(f, os_W_OK)
l = {}


def dir_walker(root, fileMask='*', bGoodFile=lambda fname, mask: fnmatch(fname, mask),
               bGoodDir=lambda fname: True):
    """

    :param root: upper dir to start search files
    :param fileMask: mask for files to find
    :param bGoodFile: filter for files
    :param bGoodDir:  filter for dirs. If set False will search only in root dir
    :return: list of full names of files found
    """
    if root.startswith('.'):
        root = os_path.abspath(root)
    root = os_path.expanduser(os_path.expandvars(root))
    if readable(root):
        if not os_path.isdir(root):
            yield root
            return
        for fname in os_listdir(root):
            pth = os_path.join(root, fname)
            if os_path.isdir(pth):
                if bGoodDir(fname):
                    for entry in dir_walker(pth, fileMask, bGoodFile, bGoodDir):
                        yield entry
            elif readable(pth) and bGoodFile(fname, fileMask):
                yield pth


# Used in next two functions
bGood_NameEdge = lambda name, namesBadAtEdge: \
    all([name[-len(notUse):] != notUse and name[:len(notUse)] != notUse \
         for notUse in namesBadAtEdge])


def bGood_dir(dirName, namesBadAtEdge):
    if bGood_NameEdge(dirName, namesBadAtEdge):
        return True
    return False


def bGood_file(fname, mask, namesBadAtEdge, bPrintGood=True):
    # any([fname[i] == strProbe for i in range(min(len(fname), len(strProbe) + 1))])
    # in fnmatch.filter(os_listdir(root)
    if fnmatch(fname, mask) and bGood_NameEdge(fname, namesBadAtEdge):
        if bPrintGood: print(fname, end=' ')
        return True
    return False


def dir_create_if_need(str_dir: Union[str, Path]) -> Path:
    """

    :param str_dir:
    :return: Path(str_dir)
    """
    if str_dir:
        str_dir = Path(str_dir)
        if not str_dir.is_dir():
            print(f' ...making dir "{str_dir}"... ')
            try:
                str_dir.mkdir()
            except Exception as e:
                raise FileNotFoundError(f'Can make only 1 level of dir. Can not make: "{str_dir}"')
    return str_dir


def dir_from_cfg(cfg, key_dir):
    """
    If not cfg[key_dir] is absolute path then set: cfg[key_dir] = cfg['dir']+cfg[key_dir]
    Create cfg[key_dir] if need
    :param cfg: dict with keys key_dir, 'dir'
    :param key_dir: absolute path or relative (in last case it is appended to cfg['dir'])
    :return: None
    """
    key_dir_path = Path(cfg[key_dir])
    if not key_dir_path.is_absolute():
        cfg[key_dir] = Path(cfg['dir']) / cfg[key_dir]
    else:
        cfg[key_dir] = key_dir_path
    return dir_create_if_need(cfg[key_dir])


# def path2rootAndMask(pathF):
# if pathF[-1] == '\\':
# root, fname = os_path.split(pathF)
# fname= '*'
# else:
# root, fname = os_path.split(pathF)

# return(root, fname)
# fileInF_All, strProbe):
# for fileInF in fileInF_All:
# DataDirName, fname = os_path.split(fileInF)
# if all([DataDirName[-len(noDir):] != noDir for noDir in (r'\bad', r'\test')]) and \
# any([fname[i] == strProbe for i in range(min(len(fname), len(strProbe) + 1))]) \
# and fname[-4:] == '.txt' and fname[:4] != 'coef':
##fileInF = os_path.join(root, fname)
# print(fname, end=' ')
# yield (DataDirName, fname)

def first_of_paths_text(paths):
    # Get only first path from paths text
    iSt = min(paths.find(r':', 3) - 1, paths.find(r'\\', 3)) + 2
    iEn = min(paths.find(r':', iSt) - 1, paths.find(r'\\', iSt))
    return paths[iSt - 2:iEn].rstrip('\\\n\r ')


def set_field_if_no(dictlike, dictfield, value=None):
    """
    Modifies dict: sets field to value only if it not exist
    :param dictlike: dict
    :param dictfield: field
    :param value: value
    :return: None
    """
    if not dictfield in dictlike or dictlike[dictfield] is None:
        dictlike[dictfield] = value


def getDirBaseOut(mask_in_path, source_dir_words=None, replaceDir=None):
    """
    Finds 'Cruise' and 'Device' dirs. Also returns full path to 'Cruise'.
    If 'keyDir' in fileMaskIn and after 2 levels of dirs then treat next subsequence as:
    ...\\'keyDir'\\'Sea'\\'Cruise'\\'Device'\\... i.e. finds subdirs after 'keyDir'
    Else use subsequence before 'keyDir' (or from end of fileMaskIn if no ``keyDir``):
    ...\\'Sea'\\'Cruise'
    :param mask_in_path: path to analyse
    :param source_dir_words: str or list of str - "keyDir" or list of variants to find "keyDir" in priority order
    :param replaceDir: str, "dir" to replace "keyDir" in out_path
        + used instead "Device" dir if "keyDir" not in fileMaskIn
    :return: returns tuple, which contains:
    #1. out_path: full path to "Cruise" (see #2.)
        - if "replaceDir" is not None: with "keyDir" is replaced by "replaceDir"
        i.e. modify dir before "Sea"
    #2. "Cruise" dir: subdir of subdir of keyDir
        - if "keyDir" dir not found: parent dir of "Device" dir
        - if "Cruise" dir not found: parent of last subdir in fileMaskIn
    #3. "Device" dir: subdir of subdir of subdir of "keyDir"
        - if "keyDir" dir not found: "replaceDir" (or "" if "replaceDir" is None)
        - if "Cruise" dir not found: last subdir
    """
    # if isinstance(mask_in_path, PurePath):
    mask_in_str = str(mask_in_path)

    if isinstance(source_dir_words, list):
        for source_dir_word in source_dir_words:
            # Start of source_dir_word in 1st detected variant
            st = mask_in_str.find(source_dir_word, 3)  # .lower()
            if st >= 0: break
    else:
        source_dir_word = source_dir_words
        st = mask_in_str.find(source_dir_word, 3)

    if st < 0:
        print("Directory structure should be ..."
              "*{}{}'Sea'{}'Cruise'{}'Device'{}!".format(source_dir_word, os_path.sep, os_path.sep, os_path.sep,
                                                         os_path.sep))
        out_path, cruise = os_path.split(mask_in_str)
        return out_path, cruise, ("" if replaceDir is None else replaceDir)

    else:
        parts_of_path = Path(mask_in_str[st:]).parts
        if len(parts_of_path) <= 2:
            # use last dirs for "\\'Sea'\\'Cruise'\\'Device'\\'{}'"
            path_device = Path(mask_in_str[:st])
            parts_of_path = path_device.parts
            cruise, device = parts_of_path[-2:]
            out_path = path_device.parent / replaceDir if replaceDir else path_device.parent
        else:
            cruise = parts_of_path[2]  # after keyDir and Sea
            try:
                device = parts_of_path[3]
            except IndexError:
                device = ''
            if replaceDir:
                out_path = Path(mask_in_str[:st]) / replaceDir / Path.joinpath(*parts_of_path[1:3])
            else:
                out_path = Path(mask_in_str[:st]).joinpath(*parts_of_path[:3])  # cruise path

        return str(out_path), cruise, device


def cfgfile2dict(arg_source: Union[Mapping[str, Any], str, PurePath, None] = None
                 ) -> Tuple[Union[Dict, configparser.RawConfigParser], Union[str, PurePath], str]:
    """

    :param arg_source: path of *.ini file or yaml file. if None - use name of program called with
        ini extension.
    :return (config, arg_source):
        - config:
            dict loaded from yaml file if arg_source is name of file with yaml o yml extension
            configparser.RawConfigParser initialised with arg_source (has dict interface too)
        - file_path, arg_ext:
            arg_source splitted to base and ext if arg_source is not dict
            else '<dict>' and ''
    """

    def set_config():
        config = configparser.RawConfigParser(inline_comment_prefixes=(';',))  # , allow_no_value = True
        config.optionxform = lambda option: option  # do not lowercase options
        return config

    if not arg_source:
        return set_config(), '<None>', ''

    b_path = isinstance(arg_source, PurePath)
    if isinstance(arg_source, str) or b_path:
        # Load data from config file
        # Set default name of config file if it is not specified.

        if not b_path:
            arg_source = PurePath(arg_source)
        if not arg_source.is_absolute():
            arg_source = PurePath(sys.argv[0]).parent.joinpath(arg_source)
        arg_ext = arg_source.suffix
        try:
            dir_create_if_need(arg_source.parent)
        except FileNotFoundError:  # path is not constist of less than 1 level of new subdirs
            print('Ini file "{}" dir not found, continue...'.format(arg_source))
            config = {}
        else:
            if arg_ext.lower() in ['.yml', '.yaml']:
                """ lazy-import PyYAML so that we doesn't have to dependend
                    on it unless this parser is used
                """
                try:
                    from yaml import safe_load as yaml_safe_load
                except ImportError:
                    raise ImportError("Could not import yaml. "
                                      "It can be installed by running 'pip install PyYAML'")
                try:
                    with open(arg_source, encoding='utf-8') as f:
                        config = yaml_safe_load(f.read())
                except FileNotFoundError:  # path is not constist of less than 1 level of new subdirs
                    print('Ini file "{}" dir not found, continue...'.format(arg_source))
                    config = {}
            else:
                cfg_file = arg_source.with_suffix('.ini')
                # if not os_path.isfile(cfg_file):
                config = set_config()
                try:
                    with open(cfg_file, 'r', encoding='cp1251') as f:
                        config.read(cfg_file)
                except FileNotFoundError:
                    print('Ini file "{}" not found, continue...'.format(cfg_file))  # todo: l.warning
                    config = {}

    elif isinstance(arg_source, dict):
        # config = set_config()
        # config.read_dict(arg_source)  # todo: check if it is need
        config = arg_source
        arg_source = '<dict>'
        arg_ext = ''
    return config, arg_source, arg_ext


def type_fix(oname: str, opt: Any) -> Tuple[str, Any]:
    """
    Checking special words in parts of option's name splitted by '_'

    :param oname: option's name. If special prefix/suffix provided then opt's type will be converted accordingly
    :param opt: option's value, usually str that need to convert to the type specified by oname's prefix/suffix.
    For different oname's prefix/suffixes use this formattin rules:
    - 'dict': do not use curles, field separator: ',' if no '\n' in it else '\n,', key-value separator: ': ' (or ':' if no ': ')
    - 'list': do not use brackets, item separator: "'," if fist is ``'`` else '",' if fist is ``"`` else ','

    ...
    :return: (new_name, new_opt)
    """

    key_splitted = oname.split('_')
    key_splitted_len = len(key_splitted)
    if key_splitted_len < 1:
        return oname, opt
    else:
        prefix = key_splitted[0]
        suffix = key_splitted[-1] if key_splitted_len > 1 else ''
    onamec = None
    try:
        if suffix in {'list', 'names'}:  # , '_ends_with_list' -> '_ends_with'
            # parse list
            onamec = '_'.join(key_splitted[0:-1])
            if not opt:
                opt_list_in = [None]
            elif opt[0] == "'":  # split to strings separated by "'," stripping " ',\n"
                opt_list_in = [n.strip(" ',\n") for n in opt.split("',")]
            elif opt[0] == '"':  # split to strings separated by '",' stripping ' ",\n'
                opt_list_in = [n.strip(' ",\n') for n in opt.split('",')]
            else:  # split to strings separated by ','
                opt_list_in = [n.strip() for n in opt.split(',')]

            opt_list = []
            for opt_in in opt_list_in:
                onamec_in, val_in = type_fix(onamec, opt_in)  # process next suffix
                opt_list.append(val_in)
            return onamec_in, ([]
                               if opt_list_in == [None] else
                               opt_list)
            # suffix = key_splitted[-2]  # check next suffix:
            # if suffix in {'int', 'integer', 'index'}:
            #     # type of list values is specified
            #     onamec = '_'.join(key_splitted[0:-2])
            #     return onamec, [int(n) for n in opt.split(',')] if opt else []
            # elif suffix in {'b', 'bool'}:
            #     onamec = '_'.join(key_splitted[0:-2])
            #     return onamec, list(literal_eval(opt))
            # else:
            #     onamec = '_'.join(key_splitted[0:-1])
            #     if not opt:
            #         return onamec, []
            #     elif opt[0] == "'":  # split to strings separated by "'," stripping " ',\n"
            #         return onamec, [n.strip(" ',\n") for n in opt.split("',")]
            #     elif opt[0] == '"':  # split to strings separated by '",' stripping ' ",\n'
            #         return onamec, [n.strip(' ",\n') for n in opt.split('",')]
            #     else:  # split to strings separated by ','
            #         return onamec, [n.strip() for n in opt.split(',')]
        if suffix == 'dict':
            onamec = '_'.join(key_splitted[0:-1])
            if opt is None:
                return onamec, {}
            else:
                def val_type_fix(parent_name, field_name, field_value):
                    # modyfy type of field_value based on parent_name
                    _, val = type_fix(parent_name, field_value)
                    return field_name, val

                return onamec, dict([val_type_fix(onamec, *n.strip().split(': ' if ': ' in n else ':')) for n in
                                     opt.split('\n,' if '\n' in opt else ',') if len(n)])
        if prefix == 'b':
            return oname, literal_eval(opt)
        if prefix == 'time':
            return oname, datetime.strptime(opt, '%Y %m %d %H %M %S')
        if prefix == 'dt':
            onamec = '_'.join(key_splitted[:-1])
            if opt is None:
                try:
                    timedelta(**{suffix: 0})  # checking suffix
                except TypeError as e:
                    raise KeyError(e.msg) from e  # changing type to be not catched and accepted if bad suffix
            return onamec, timedelta(**{suffix: float(opt)})
        b_trig_is_prefix = prefix in {'date', 'time'}
        if b_trig_is_prefix or suffix in {'date', 'time'}:
            if b_trig_is_prefix or prefix in {'min', 'max'}:  # not strip to 'min', 'max'
                onamec = oname
                # = opt_new  #  use other temp. var instead onamec to keep name (see last "if" below)???
            else:
                onamec = '_'.join(key_splitted[0:-1])  # #oname = del suffix
                # onamec = opt_new  # will del old name (see last "if" below)???
            date_format = '%Y-%m-%dT'
            if not '-' in opt[:len(date_format)]:
                date_format = '%d.%m.%Y '
            try:  # opt has only date?
                return onamec, datetime.strptime(opt, date_format[:-1])
            except ValueError:
                time_format = '%H:%M:%S%z'[:(len(opt) - len(date_format) - 2)]  # minus 2 because 2 chars of '%Y' corresponds 4 digits of year
                try:
                    tim = datetime.strptime(opt, f'{date_format}{time_format}')
                except ValueError:
                    if opt == 'NaT':  # fallback to None
                        tim = None
                    elif opt == 'now':
                        tim = datetime.now()
                    else:
                        raise
                return onamec, tim
        if suffix in {'int', 'integer', 'index'}:
            onamec = '_'.join(key_splitted[0:-1])
            return onamec, int(opt)
        if suffix == 'float':  # , 'percent'
            onamec = '_'.join(key_splitted[0:-1])
            return onamec, float(opt)
        if suffix in {'b', 'bool'}:
            onamec = '_'.join(key_splitted[0:-1])
            return onamec, literal_eval(opt)
        if suffix == 'chars':
            onamec = '_'.join(key_splitted[0:-1])
            return onamec, opt.replace('\\t', '\t').replace('\\ \\', ' ')
        if prefix in {'fixed', 'float', 'max', 'min'}:
            # this snameion is at end because includes frequently used 'max'&'min' which not
            # nesesary for floats, so set to float only if have no other special format words
            return oname, float(opt)

        if 'path' in {suffix, prefix}:
            return oname, Path(opt)

        return oname, opt
    except (TypeError, AttributeError, ValueError) as e:
        # do not try to convert not a str
        if not isinstance(opt, str):
            return onamec if onamec else oname, opt  # onamec is replasement of oname
        else:
            raise e


def ini2dict(arg_source=None):
    """
    Loads configuration dict from *.ini file with type conversion based on keys names.
    Removes suffics type indicators but keep prefiх.
    prefiх/suffics type indicators (following/precieded with "_"):
        b
        chars - to list of chars, use string "\\ \\" to specify space char
        time
        dt (prefix only) with suffixes: ... , minutes, hours, ... - to timedelta
        list, (names - not recommended) - splitted on ',' but if first is "'" then on "'," - to allow "," char, then all "'" removed.
        If first list characters is " or ' then breaks list on " ',\n" or ' ",\n' correspondingly.

        before list can be other suffix to convert to
        int, integer, index - to integer
        float - to float

    :param arg_source: path of *.ini file. if None - use name of program called with
        ini extension.
    :return: dict - configuration parsed

    Uses only ".ini" extension nevertheless which was cpecified or was specified at all
    """

    config, arg_source, arg_ext = cfgfile2dict(arg_source)
    cfg = {key: {} for key in config}
    oname = None
    opt = None
    # convert cpecific fields data types
    try:
        for sname, sec in config.items():
            if sname[:7] == 'TimeAdd':
                d = {opt: float(opt) for opt in sec}
                cfg[sname] = timedelta(**d)
            else:
                opt_used = set()
                for oname, opt in sec.items():
                    new_name, val = type_fix(oname, opt)
                    # if new_name in sec:       # to implement this first set sec[new_name] to zero
                    #     val += sec[new_name]  #

                    if new_name in opt_used:
                        if opt is None:  # val == opt if opt is None # or opt is Nat
                            continue
                        cfg[sname][new_name] += val
                    else:
                        cfg[sname][new_name] = val
                        opt_used.add(new_name)
                # cfg[sname]= dict(config.items(sname)) # for sname in config.sections()]
    except Exception as e:  # ValueError, TypeError
        # l.exception(e)
        raise Error_in_config_parameter(
            '[{}].{} = "{}": {}'.format(sname, oname, str(opt), e.args[0])).with_traceback(e.__traceback__)
    set_field_if_no(cfg, 'in', {})
    cfg['in']['cfgFile'] = arg_source
    return cfg


def cfg_from_args(p, arg_add, **kwargs):
    """
    Split sys.argv to ``arg_source`` for loading config (data of 2nd priority) and rest (data of 1st priority)
    arg_source = sys.argv[1] if it not starts with '--' else tries find file sys.argv[0] + ``.yaml`` or else ``.ini``
    and assigns arg_source to it.
    Loaded data (configargparse object) is converted to configuration dict of dicts

    1st priority (sys.argv[2:]) will overwrite all other
    See requirements for command line arguments p
    (argument_groups (sections) is top level dict)

    :param p: configargparse object of parameters. 1st command line parameter in it
     must not be optional. Also it may be callable with arguments to init configargparse.ArgumentParser()
    :param arg_add: list of string arguments in format of command line interface defined by p to add/replace parameters
     (starting from 2nd)
    :param kwargs: dicts for each section: to overwrite values in them (overwrites even high priority values, other values remains)
    :return cfg: dict with parameters
        will set cfg['in']['cfgFile'] to full name of used config file
        '<prog>' strings in p replaces with p.prog
    see also: my_argparser_common_part()
    """

    def is_option_name(arg):
        """
        True if arg is an option name (i.e. that is not a value, which must be followed)
        :param arg:
        :return:
        """
        return (isinstance(arg, str) and arg.startswith('--'))

    args = {}  # will be conveerted to dict cfg:
    cfg = None

    if arg_add:
        argv_save = sys.argv.copy()

        # # todo: exclude arguments that are not strings from argparse processing (later we add them to cfg)
        # # iterate though values
        # for i in range(start=(1 if is_option_name(arg_add[0]) else 0), end=len(arg_add), step=2):
        #     if not isinstance(arg_add[i], str):
        #         arg_add
        sys.argv[1:] = arg_add

    skip_config_file_parsing = False  # info argument (help, version) was passed?
    if len(sys.argv) > 1 and not is_option_name(sys.argv[1]):
        # name of config file (argument source) is specified
        arg_source = sys.argv[1]
    else:
        # auto search config file (source of arguments)
        exe_path = Path(sys.argv[0])
        cfg_file = exe_path.with_suffix('.yaml')
        if not cfg_file.is_file():  # do I need check Upper case letters for Linux?
            cfg_file = exe_path.with_suffix('.yml')
            if not cfg_file.is_file():
                cfg_file = exe_path.with_suffix('.ini')
                if not cfg_file.is_file():
                    print('using default configuration')
                    cfg_file = None
        sys.argv.insert(1, str(cfg_file))
        if cfg_file:
            print('using configuration from file:', cfg_file)
        arg_source = cfg_file

    if len(sys.argv) > 2:
        skip_config_file_parsing = sys.argv[2] in ["-h", "--help", "-v", "--version"]
        if skip_config_file_parsing:
            if callable(p):
                p = p(None)
            args = vars(p.parse_args())  # will generate SystemExit

    # Load options from ini file
    config, arg_source, arg_ext = cfgfile2dict(arg_source)
    if callable(p):  # todo: replace configargparse back to argparse to get rid of this double loading of ini/yaml?
        p = p({'config_file_parser_class': configargparse.YAMLConfigFileParser
               } if arg_ext.lower() in ('.yml', '.yaml') else None)

    # Collect argument groups
    p_groups = {g.title: g for g in p._action_groups if
                g.title.split(' ')[-1] != 'arguments'}  # skips special argparse groups

    def get_or_add_sec(section_name, p_groups, sec_description=None):
        if section_name in p_groups:
            p_sec = p_groups[section_name]
        else:
            p_sec = p.add_argument_group(section_name, sec_description)
            p_groups[section_name] = p_sec
        return p_sec

    if config:
        prefix = '--'
        for section_name, section in config.items():
            try:  # now "if not isinstance(section, dict)" is not works, "if getattr(section, 'keys')" still works but "try" is more universal
                ini_sec_options = set(section)  # same as set(section.keys())
            except Exception as e:
                continue

            p_sec = get_or_add_sec(section_name, p_groups)
            p_sec_hardcoded_list = [a.dest for a in p_sec._group_actions]
            ini_sec_options_new = ini_sec_options.difference(set(p_sec_hardcoded_list))
            ini_sec_options_same = ini_sec_options.difference(ini_sec_options_new)
            # get not hardcoded options from ini:
            for option_name in ini_sec_options_new:
                if not isinstance(option_name, str):
                    continue
                try:
                    # p_sec.set_defaults(**{'--' + option_name: config.get(section_name, option_name)})
                    p_sec.add(f'{prefix}{option_name}', default=section[option_name])
                except configargparse.ArgumentError as e:
                    # Same options name but in other ini section
                    option_name_changed = f'{section_name}.{option_name}'
                    try:
                        p_sec.add(f'{prefix}{option_name_changed}', default=section[option_name])
                    except configargparse.ArgumentError as e:
                        # Changed option name was hardcoded so replase defaults defined there
                        p_sec._group_actions[p_sec_hardcoded_list.index(option_name_changed)].default = section[
                            option_name]
                        # p_sec.set_defaults(

            # overwrite hardcoded defaults from ini in p: this is how we make it 2nd priority and defaults - 3rd priority
            for option_name in ini_sec_options_same:
                p_sec._group_actions[p_sec_hardcoded_list.index(option_name)].default = section[option_name]

    # uppend arguments with my common options:
    p_sec = get_or_add_sec('program', p_groups, 'Program behaviour')
    try:
        p_sec.add_argument(
            '--b_interact', default='True',
            help='ask showing source files names before process them')
    except configargparse.ArgumentError as e:
        pass  # option already exist - need no to do anything
    try:
        p_sec.add_argument(
            '--log', default=os_path.join('log', f'{this_prog_basename()}.log'),
            help='write log if path to existed file is specified')
    except configargparse.ArgumentError as e:
        pass  # option already exist - need no to do anything
    try:
        p_sec.add_argument(
            '--verbose', '-V', type=str, default='INFO',  # nargs=1,
            choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'],
            help='verbosity of messages in log file')
    except configargparse.ArgumentError as e:
        pass  # option already exist - need no to do anything

    # os_path.join(os_path.dirname(__file__), 'empty.ini')
    sys.argv[1] = ''  # do not parse ini file by configargparse already parsed by configparser
    try:
        args = vars(p.parse_args())
    except SystemExit as e:
        if skip_config_file_parsing:
            raise

        # Bad arguments found. Error message of unrecognized arguments displayed,
        # but we continue. To remove message add arguments to p before call this func
        pass

    # args = vars(p.parse_args())

    try:
        # Collect arguments dict to groups (ini sections) ``cfg``
        cfg = {}
        for section_name, gr in p_groups.items():
            keys = args.keys() & [a.dest for a in gr._group_actions]
            cfg_section = {}
            for key in keys:
                arg_cur = args[key]
                if isinstance(arg_cur, str):
                    arg_cur = arg_cur.replace('<prog>', p.prog).strip()
                if '.' in key:
                    key = key.split('.')[1]
                cfg_section[key] = arg_cur

            cfg[section_name] = cfg_section

        if arg_ext.lower() in ('.yml', '.yaml'):
            # Remove type suffixes '_list', '_int' ... (currently removing is needed for yaml files only)
            suffixes = {'_list', '_int', '_integer', '_index', '_float', '_b', '_bool', 'date', 'chars', '_dict'}
        else:
            # change types based on prefix/suffix
            cfg = ini2dict(cfg)
            suffixes = ()

        # Convert cfg['re mask'] chields and all str (or list of str but not starting from '') chields beginning with 're_' to compiled regular expression object
        # lists in 're_' are joined to strings before compile (useful as list allows in yaml to use aliases for part of re expression)

        for key_level0, v in cfg.items():
            if key_level0 == 're_mask':  # replace all chields to compiled re objects
                for key_level1, opt in v.items():
                    cfg['re_mask'][key_level1] = re.compile(opt)
            else:
                for key_level1, opt in v.copy().items():
                    if key_level1.startswith('re_'):  # replace strings beginning with 're_' to compiled re objects
                        is_lst = isinstance(opt, list)
                        if is_lst:
                            if key_level1.endswith('_list'):  # type already converted, remove type suffix
                                new_key = key_level1[:-len('_list')]
                                cfg[key_level0][new_key] = opt
                                del cfg[key_level0][key_level1]
                                key_level1 = new_key

                            try:  # skip not list of str
                                if (not isinstance(opt[0], str)) or (not opt[0]):
                                    continue
                            except IndexError:
                                continue
                            cfg[key_level0][key_level1] = re.compile(''.join(opt))
                        else:
                            cfg[key_level0][key_level1] = re.compile(opt)
                    elif not (opt is None or isinstance(opt,
                                                        str)):  # type already converted, remove type suffixes here only
                        for ends in suffixes:
                            if key_level1.endswith(ends):
                                cfg[key_level0][key_level1[:-len(ends)]] = opt
                                del cfg[key_level0][key_level1]
                                break
                    else:
                        # type not converted, remove type suffixes and convert
                        new_name, val = type_fix(key_level1, opt)
                        if new_name == key_level1:               # if only type changed
                            cfg[key_level0][new_name] = val
                        else:
                            if not new_name in cfg[key_level0]:  # if not str (=> not default) parameter without suffixes added already
                                cfg[key_level0][new_name] = val
                            del cfg[key_level0][key_level1]

        if kwargs:
            for key_level0, kwargs_level1 in kwargs.items():
                cfg[key_level0].update(kwargs_level1)

        cfg['in']['cfgFile'] = arg_source
        # config = configargparse.parse_args(cfg)
    except Exception as e:  # IOError
        print('Configuration ({}) error:'.format(arg_add), end=' ')
        print('\n==> '.join([s for s in e.args if isinstance(s, str)]))  # getattr(e, 'message', '')
        raise (e)
    finally:
        if arg_add:  # recover argv for possible outer next use
            sys.argv = argv_save

    return (cfg)


class MyArgparserCommonPart(configargparse.ArgumentParser):
    def init(self, default_config_files=[],
             formatter_class=configargparse.ArgumentDefaultsRawHelpFormatter, epilog='',
             args_for_writing_out_config_file=["-w", "--write-out-config-file"],
             write_out_config_file_arg_help_message="takes the current command line arguments and writes them out to a configuration file the given path, then exits. But this file have no section headers. So to use this file you need to add sections manually. Sections are listed here in help message: [in], [output_files] ...",
             ignore_unknown_config_file_keys=True, version='?'):
        self.add('cfgFile', is_config_file=True,
                 help='configuration file path(s). Command line parameters will overwrites parameters specified iside it')
        self.add('--version', '-v', action='version', version=
        f'%(prog)s version {version} - (c) 2017 Andrey Korzh <ao.korzh@gmail.com>.')

        # Configuration sections

        # All argumets of type str (default for add_argument...), because of
        # custom postprocessing based of args names in ini2dict

        '''
        If "<filename>" found it will be sabstituted with [1st file name]+, if "<dir>" -
        with last ancestor directory name. "<filename>" string
        will be sabstituted with correspondng input file names.
        '''

        # p_program = p.add_argument_group('program', 'Program behaviour')
        # p_program.add_argument(
        #     '--verbose', '-V', type=str, default='INFO', #nargs=1,
        #     choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'],
        #     help='verbosity of messages in log file')
        return (self)


def my_argparser_common_part(varargs, version='?'):  # description, version='?', config_file_paths=[]
    """
    Define configuration
    :param varargs: dict, containing configargparse.ArgumentParser parameters to set
        Note: '-' in dict keys will be replaced  to '_' in ArgumentParser
    :param version: value for `version` parameter
    :return p: configargparse object of parameters
    """

    varargs.setdefault('default_config_files', [])
    varargs.setdefault('formatter_class', configargparse.ArgumentDefaultsRawHelpFormatter)
    # formatter_class= configargparse.ArgumentDefaultsHelpFormatter,
    varargs.setdefault('epilog', '')
    varargs.setdefault('args_for_writing_out_config_file', ["-w", "--write-out-config-file"])
    varargs.setdefault('write_out_config_file_arg_help_message',
                       "takes the current command line arguments and writes them out to a configuration file the given path, then exits. But this file have no section headers. So to use this file you need to add sections manually. Sections are listed here in help message: [in], [output_files] ...")
    varargs.setdefault('ignore_unknown_config_file_keys', True)

    p = configargparse.ArgumentParser(**varargs)

    p.add('cfgFile', is_config_file=True,
          help='configuration file path(s). Command line parameters will overwrites parameters specified iside it')
    p.add('--version', '-v', action='version', version=
    '%(prog)s version {version} - (c) 2019 Andrey Korzh <ao.korzh@gmail.com>.')

    # Configuration sections

    # All argumets of type str (default for add_argument...), because of
    # custom postprocessing based of args names in ini2dict

    '''
    If "<filename>" found it will be sabstituted with [1st file name]+, if "<dir>" -
    with last ancestor directory name. "<filename>" string
    will be sabstituted with correspondng input file names.
    '''

    # p_program = p.add_argument_group('program', 'Program behaviour')
    # p_program.add_argument(
    #     '--verbose', '-V', type=str, default='INFO', #nargs=1,
    #     choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'],
    #     help='verbosity of messages in log file')

    return (p)


def pathAndMask(path, filemask=None, ext=None):
    """
    Find Path & Mask
    :param path:
    :param filemask:
    :param ext:
    :return:

    # File mask can be specified in "path" (for examample full path) it has higher priority than
    # "filemask" which can include ext part which has higher priority than specified by "ext"
    # But if turget file(s) has empty name or ext than they need to be specified explisetly by ext = .(?)
    """
    path, fileN_fromCfgPath = os_path.split(path)
    if fileN_fromCfgPath:
        if '.' in fileN_fromCfgPath:
            fileN_fromCfgPath, cfg_path_ext = os_path.splitext(fileN_fromCfgPath)
            if cfg_path_ext:
                cfg_path_ext = cfg_path_ext[1:]
            else:
                cfg_path_ext = fileN_fromCfgPath[1:]
                fileN_fromCfgPath = ''
        else:  # wrong split => undo
            cfg_path_ext = ''
            path = os_path.join(path, fileN_fromCfgPath)
            fileN_fromCfgPath = ''
    else:
        cfg_path_ext = ''

    if not filemask is None:
        fileN_fromCfgFilemask, cfg_filemask_ext = os_path.splitext(filemask)
        if '.' in cfg_filemask_ext:
            if not cfg_path_ext:
                # possible use ext. from ['filemask']
                if not cfg_filemask_ext:
                    cfg_path_ext = fileN_fromCfgFilemask[1:]
                elif cfg_filemask_ext:
                    cfg_path_ext = cfg_filemask_ext[1:]

        if not fileN_fromCfgPath:
            # use name from ['filemask']
            fileN_fromCfgPath = fileN_fromCfgFilemask
    elif not fileN_fromCfgPath:
        fileN_fromCfgPath = '*'

    if not cfg_path_ext:
        # check ['ext'] exists
        if ext is None:
            cfg_path_ext = '*'
        else:
            cfg_path_ext = ext

    filemask = f'{fileN_fromCfgPath}.{cfg_path_ext}'
    return (path, filemask)


# ----------------------------------------------------------------------
def generator_good_between(i_start=None, i_end=None):
    k = 0
    if i_start is not None:
        while k < i_start:
            yield False
            k += 1
    if i_end is not None:
        while k < i_end:
            yield True
            k += 1
        while True:
            yield False
    while True:
        yield True


def init_file_names(cfg_files: MutableMapping[str, Any], b_interact=True, path_field=None):
    """
      Fill cfg_files filds of file names: {'path', 'filemask', 'ext'}
    which are not specified.
      Searches for files with this mask. Prints number of files found.
      If any - asks user to proceed and if yes returns its names list.
      Else raises Ex_nothing_done exception.

    :param cfg_files: dict with fields:
        'path', 'filemask', 'ext' - name of file with mask or it's part
        exclude_files_ends_with - additional filter for ends in file's names
        b_search_in_subdirs, exclude_dirs_ends_with - to search in dirs recursively
        start_file, end_file - exclude files before and after this values in search list result
    :param b_interact: do ask user to proceed? If false proseed silently
    :return: (namesFull, cfg_files)
        cfg_files: configuration with added (if was not) fields
    'path':,
    'filemask':,
    'nfiles': number of files found,
    'namesFull': list of full names of found files
    """
    set_field_if_no(cfg_files, 'b_search_in_subdirs', False)
    if path_field:
        cfg_files['path'] = cfg_files[path_field]
    set_cfg_path_filemask(cfg_files)

    # Filter unused directories and files
    filt_dirCur = lambda f: bGood_dir(f, namesBadAtEdge=cfg_files[
        'exclude_dirs_ends_with']) if ('exclude_dirs_ends_with' in cfg_files) else \
        lambda f: bGood_dir(f, namesBadAtEdge=(r'bad', r'test'))  # , r'\w'

    def skip_to_start_file(fun):
        if ('start_file' in cfg_files) or ('end_file' in cfg_files):
            fun_skip = generator_good_between(
                cfg_files['start_file'] if 'start_file' in cfg_files else None,
                cfg_files['end_file'] if 'end_file' in cfg_files else None)

            def call_skip(*args, **kwargs):
                return (fun(*args, **kwargs) and fun_skip.__next__())

            return call_skip
        return fun

    def skip_files_ends_with(fun):
        if 'exclude_files_ends_with' in cfg_files:
            def call_skip(*args, **kwargs):
                return fun(*args, namesBadAtEdge=cfg_files['exclude_files_ends_with'])
        else:
            def call_skip(*args, **kwargs):
                return fun(*args, namesBadAtEdge=(r'coef.txt',))
        return call_skip

    def print_file_name(fun):
        def call_print(*args, **kwargs):
            if fun(*args, **kwargs):
                print(args[0], end=' ')
                return True
            else:
                return False

        return call_print

    bPrintGood = True
    if not bPrintGood:
        print_file_name = lambda fun: fun

    @print_file_name
    @skip_files_ends_with
    @skip_to_start_file
    def filt_file_cur(fname, mask, namesBadAtEdge):
        # if fnmatch(fname, mask) and bGood_NameEdge(fname, namesBadAtEdge):
        #     return True
        # return False
        return bGood_file(fname, mask, namesBadAtEdge, bPrintGood=False)

    print('search for {} files'.format(os_path.join(os_path.abspath(
        cfg_files['dir']), cfg_files['filemask'])), end='')

    # Execute declared functions ######################################
    if cfg_files['b_search_in_subdirs']:
        print(', including subdirs:', end=' ')
        cfg_files['namesFull'] = [f for f in dir_walker(
            cfg_files['dir'], cfg_files['filemask'],
            bGoodFile=filt_file_cur, bGoodDir=filt_dirCur)]
    else:
        print(':', end=' ')
        cfg_files['namesFull'] = [os_path.join(cfg_files['dir'], f) for f in sorted(os_listdir(
            cfg_files['dir'])) if filt_file_cur(f, cfg_files['filemask'])]
    cfg_files['nfiles'] = len(cfg_files['namesFull'])

    print(end=f"\n- {cfg_files['nfiles']} found")
    if cfg_files['nfiles'] == 0:
        print('!')
        raise Ex_nothing_done
    else:
        print(end='. ')
    if b_interact:
        s = input(f"Process {'them' if cfg_files['nfiles'] > 1 else 'it'}? Y/n: ")
        if 'n' in s or 'N' in s:
            print('answered No')
            raise Ex_nothing_done
        else:
            print('wait... ', end='')

    """
    def get_vsz_full(inFE, vsz_path):
        # inFE = os_path.basename(in_full)
        inF = os_path.splitext(inFE)[0]
        vszFE = inF + '.vsz'
        return os_path.join(vsz_path, vszFE)

    def filter_existed(inFE, mask, namesBadAtEdge, bPrintGood, cfg_out):
        if cfg_out['fun_skip'].next(): return False

        # any([inFE[i] == strProbe for i in range(min(len(inFE), len(strProbe) + 1))])
        # in fnmatch.filter(os_listdir(root)
        if not cfg_out['b_update_existed']:
            # vsz file must not exist
            vsz_full = get_vsz_full(inFE, cfg_out['path'])
            if os_path.isfile(vsz_full):
                return False
        elif cfg_out['b_images_only']:
            # vsz file must exist
            vsz_full = get_vsz_full(inFE, cfg_out['path'])
            if not os_path.isfile(vsz_full):
                return False
        else:
            return bGood_file(inFE, mask, namesBadAtEdge, bPrintGood=True)


    """

    return cfg_files


# File management ##############################################################

def name_output_file(fileDir, filenameB, filenameE=None, bInteract=True, fileSizeOvr=0):
    """
    Name output file, rename or overwrite if output file exist.
    :param fileDir:   file directoty
    :param filenameB: file base name
    :param filenameE: file extention. if None suppose filenameB is contans it
    :param bInteract: to ask user?
    :param fileSizeOvr: (bytes) bad files have this or smaller size. So will be overwrite
    :return: (filePFE, sChange, msgFile):
    filePFE - suggested output name. May be the same if bInteract=True, and user
    answer "no" (i.e. to update existed), or size of existed file <= fileSizeOvr
    sChange - user input if bInteract else ''
    msgFile - string about resulting output name
    """

    # filename_new= re_sub("[^\s\w\-\+#&,;\.\(\)']+", "_", filenameB)+filenameE

    # Rename while target exists and it hase data (otherwise no crime in overwriting)
    msgFile = ''
    m = 0
    sChange = ''
    str_add = ''
    if filenameE is None:
        filenameB, filenameE = os_path.splitext(filenameB)

    def append_to_filename(str_add):
        """
        Returns filenameB + str_add + filenameE if no file with such name in fileDir
        or its size is less than fileSizeOvr else returns None
        :param str_add: string to add to file name before extension
        :return: base file name or None
        """
        filename_new = f'{filenameB}{str_add}{filenameE}'
        full_filename_new = os_path.join(fileDir, filename_new)
        if not os_path.isfile(full_filename_new):
            return filename_new
        try:
            if os_path.getsize(full_filename_new) <= fileSizeOvr:
                msgFile = 'small target file (with no records?) will be overwrited:'
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
        str_add = f'_({m})'

    if (m > 0) and bInteract:
        sChange = input('File "{old}" exists! Change target name to ' \
                        '"{new}" (Y) or update existed (n)?'.format(
            old=f'{filenameB}{filenameE}', new=filename_new))

    if bInteract and sChange in ['n', 'N']:
        # update only if answer No
        msgFile = 'update existed'
        filePFE = os_path.join(fileDir, f'{filenameB}{filenameE}')  # new / overwrite
        writeMode = 'a'
    else:
        # change name if need in auto mode or other answer
        filePFE = os_path.join(fileDir, filename_new)
        if m > 0:
            msgFile += f'{str_add} added to name.'
        writeMode = 'w'
    dir_create_if_need(fileDir)
    return (filePFE, writeMode, msgFile)


def set_cfg_path_filemask(cfg_files):
    """
    Sets 'dir' and 'filemask' of cfg_files based on its
    'path','filemask','ext' fieds ('path' or 'filemask' is required)
    :param cfg_files: dict with field 'path' or/and 'filemask' and may be 'ext'
    :return: None

    # Note. Extension may be within 'path' or in 'ext'
    """

    cfg_files['dir'], cfg_files['filemask'] = pathAndMask(*[
        cfg_files[spec] if spec in cfg_files else None for
        spec in ['path', 'filemask', 'ext']])

    if not os_path.isabs(cfg_files['dir']):
        dir_path = Path(sys.argv[0]).parent / cfg_files['dir']
        try:
            dir = dir_path.resolve()  # gets OSError "Bad file name" if do it directly for ``path`` having filemask symbols '*','?'
        except FileNotFoundError:
            dir_create_if_need(str(dir_path))
            dir = dir_path.resolve()
        cfg_files['dir'] = str(dir)
        cfg_files['path'] = str(dir / cfg_files['filemask'])


def splitPath(path, default_filemask):
    """
    Split path to (D, mask, Dlast). Enshure that mask is not empty by using default_filemask.
    :param path: file or dir path
    :param default_filemask: used for mask if path is directory
    :return: (D, mask, Dlast). If path is file then D and mask adjasent parts of path, else
    mask= default_filemask
        mask: never has slash and is never empty
        D: everything leading up to mask
        Dlast: last dir name in D
    """
    D = os_path.abspath(path)
    if os_path.isdir(D):
        mask = default_filemask
        Dlast = os_path.basename(D)
    else:
        D, mask = os_path.split(D)
        if not os_path.splitext(mask)[1]:  # no ext => path is dir, no mask provided
            Dlast = mask
            D = os_path.join(D, Dlast)
            mask = default_filemask
        else:
            Dlast = os_path.basename(D)
    return D, mask, Dlast


def prep(args, default_input_filemask='*.pdf',
         msgFound_n_ext_dir='Process {n} {ext}{files} from {dir}'):
    """
    Depreciated!!!

    :param args: dict {path, out_path}
        *path: input dir or file path
        *out_path: output dir. Can contain
    <dir_in>: will be replased with last dir name in args['path']
    <filename>: not changed, but used to plit 'out_path' such that it is not last in outD
    but part of outF
    :param default_input_filemask:
    :param msgFound_n_ext_dir:
    :return: tuple (inD, namesFE, nFiles, outD, outF, outE, bWrite2dir, msgFile):
    inD             - input directory
    namesFE, nFiles - list of input files found and list's size
    outD            - output directory
    outF, outE      - output base file name and its extension ()
    bWrite2dir      - "output is dir" True if no extension specified.
    In this case outE='csv', outF='<filename>'
    msgFile         - string about numaber of input files found
    """

    # Input files

    inD, inMask, inDlast = splitPath(args['path'], default_input_filemask)
    try:
        namesFE = [f for f in os_path.os.listdir(inD) if fnmatch(f, inMask)]
    except WindowsError as e:
        raise Ex_nothing_done(f'{e.message} - No {inMask} files in "{inD}"?')
    nFiles = len(namesFE)

    if nFiles > 1:
        msgFile = msgFound_n_ext_dir.format(n=nFiles, dir=inD, ext=inMask, files=' files')
    else:
        msgFile = msgFound_n_ext_dir.format(n='', dir=inD, ext=inMask, files='')

    if nFiles == 0:
        raise Ex_nothing_done
    else:
        # Output dir
        outD, outMask, Dlast = splitPath(args['out_path'], '*.%no%')
        # can not replace just in args['out_path'] if inDlast has dots
        Dlast = Dlast.replace('<dir_in>', inDlast)
        outD = outD.replace('<dir_in>', inDlast)

        if '<filename>' in Dlast:  # ?
            outD = os_path.dirname(outD)
            outMask = outMask.replace('*', Dlast)

        outF, outE = os_path.splitext(outMask)
        bWrite2dir = outE.endswith('.%no%')
        if bWrite2dir:  # output path is dir
            outE = '.csv'
            if not '<filename>' in outF:
                outF = outF.replace('*', '<filename>')
    if not os_path.isdir(outD):
        os_path.os.mkdir(outD)
    return inD, namesFE, nFiles, outD, outF, outE, bWrite2dir, msgFile


def this_prog_basename(path=sys.argv[0]):
    return os_path.splitext(os_path.split(path)[1])[0]


def init_logging(logging, logger_name=__name__, log_file=None, level_file='INFO', level_console=None):
    """
    Logging to file flogD/flogN.log and console with piorities level_file and levelConsole
    :param logging: logging class from logging library
    :param logger_name:
    :param log_file: name of log file. Default: & + "this program file name"
    :param level_file: 'INFO'
    :param level_console: 'WARN'
    :return: logging Logger
   
    Call example:
    l= init_logging(logging, None, None, args.verbose)
    l.warning(msgFile)
    """
    global l
    if log_file:
        if not os_path.isabs(log_file):
            # if flogD is None:
            flogD = os_path.dirname(sys.argv[0])
            log_file = os_path.join(flogD, log_file)
    else:
        # if flogD is None:
        flogD = os_path.join(os_path.dirname(sys.argv[0]), 'log')
        dir_create_if_need(flogD)
        log_file = os_path.join(flogD, f'&{this_prog_basename()}.log')  # '&' is for autoname indication

    if l:
        try:  # a bit more check that we already have logger
            l = logging.getLogger(logger_name)
        except Exception as e:
            pass
        if l and l.hasHandlers():
            l.handlers.clear()  # or if have good handlers return l
    else:
        l = logging.getLogger(logger_name)

    logging.basicConfig(filename=Path(log_file), format='%(asctime)s %(message)s', level=level_file)

    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(level_console if level_console else 'INFO' if level_file != 'DEBUG' else 'DEBUG')  # logging.WARN
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(message)s')  # %(name)-12s: %(levelname)-8s ...
    console.setFormatter(formatter)
    l.addHandler(console)
    l.propagate = True  # to default
    return l


def name_output_and_log(cfg, logging, f_rep_filemask=lambda f: f, bInteract=False):
    """
    Initialize cfg['output_files']['path'] and splits it to fields
    'path', 'filemask', 'ext'
    Initialize logging and prints message of beginning to write

    :param cfg: dict of dicts, requires fields:
        'in' with fields
            'namesFull'
        'output_files' with fields
            'out_path'

    :param logging:
    :param bInteract: see name_output_file()
    :param f_rep_filemask: function f(cfg['output_files']['path']) modifying its argument
        To replase in 'filemask' string '<File_in>' with base of cfg['in']['namesFull'][0] use
    lambda fmask fmask.replace(
            '<File_in>', os_path.splitext(os_path.basename(cfg['in']['namesFull'][0])[0] + '+')
    :return: cfg, l
    cfg with added fields:
        in 'output_files':
            'path'

            'ext' - splits 'out_path' or 'csv' if not found in 'out_path'
    """
    # find 'path' and 'ext' required for set_cfg_path_filemask()
    if cfg['output_files']['out_path']:
        cfg['output_files']['path'], cfg['output_files']['ext'] = os_path.splitext(cfg['output_files']['out_path'])
        if not cfg['output_files']['ext']:  # set_cfg_path_filemask requires
            cfg['output_files']['ext'] = '.csv'
        cfg['output_files']['path'] = f_rep_filemask(cfg['output_files']['out_path'])

        set_cfg_path_filemask(cfg['output_files'])

        # Check target exists
        cfg['output_files']['path'], cfg['output_files']['writeMode'], msg_name_output_file = name_output_file(
            cfg['output_files']['dir'], cfg['output_files']['filemask'], None,
            bInteract, cfg['output_files']['min_size_to_overwrite'])

        str_print = '{msg_name} Saving all to {out}:'.format(
            msg_name=msg_name_output_file, out=os_path.abspath(cfg['output_files']['path']))

    else:
        set_field_if_no(cfg['output_files'], 'dir', '.')
        str_print = ''

    l = init_logging(logging, None, os_path.join(
        cfg['output_files']['dir'], cfg['program']['log']), cfg['program']['verbose'])
    if str_print:
        l.warning(str_print)  # or use "a %(a)d b %(b)s", {'a':1, 'b':2}

    return cfg, l



class Message:
    def __init__(self, fmt, args):
        self.fmt = fmt
        self.args = args

    def __str__(self):
        return self.fmt.format(*self.args)

class LoggingStyleAdapter(logging.LoggerAdapter):
    """
    Uses str.format() style in logging messages. Usage:
    logger = LoggingStyleAdapter(logging.getLogger(__name__))
    also prepends message with [self.extra['id']]
    """
    def __init__(self, logger, extra=None):
        super(LoggingStyleAdapter, self).__init__(logger, extra or {})

    def process(self, msg, kwargs):
        return ('[%s] %s' % (self.extra['id'], msg) if 'id' in self.extra else msg,
                kwargs)

    def log(self, level, msg, *args, **kwargs):
        if self.isEnabledFor(level):
            msg, kwargs = self.process(msg, kwargs)
            self.logger._log(level, Message(msg, args), (), **kwargs)




class FakeContextIfOpen:
    """
    Context manager that do nothing if file is not str/PurePath or custom open function is None/False
    useful if file can be already opened file object
    """

    def __init__(self,
                 fn_open_file: Optional[Callable[[Any], Any]] = None,
                 file: Optional[Any] = None):
        """
        :param fn_open_file: any, if not bool(fn_open_file) then context manager will do nothing on exit .
        :param file: any, if not str or PurePath then context manager will do nonthing on exit
        """
        self.file = file
        self.fn_open_file = fn_open_file
        self._do_open_close = isinstance(self.file, (str, PurePath)) and self.fn_open_file

    def __enter__(self):
        """
        :return: opened handle or :param file: from __init__ if not need open
        """
        self.handle = self.fn_open_file(self.file) if self._do_open_close else self.file
        return self.handle

    def __exit__(self, exc_type, ex_value, ex_traceback):
        """
        Closes handle returned by fn_open_file() if need
        """
        if exc_type is None and self._do_open_close:
            # self.handle is not None and
            self.handle.close()
        return False


def open_csv_or_archive_of_them(filename: Union[PurePath, Iterable[Union[Path, str]]], binary_mode=False,
                                pattern='', encoding=None) -> Iterator[Union[TextIO, BinaryIO]]:
    """
    Opens and yields files from archive with name filename or from list of filenames in context manager (autoclosing).
    Note: Allows stop iteration over files in archive by assigning True to next() in consumer of generator
    Note: to can unrar the unrar.exe must be in path or set rarfile.UNRAR_TOOL
    :param filename: archive with '.rar'/'.zip' suffix or file name or Iterable of file names
    :param pattern: Unix shell style file pattern in the archive - should include directories if need search inside (for example place "*" at beginning)
    :return:
    Note: RarFile anyway opens in binary mode
    """
    read_mode = 'rb' if binary_mode else 'r'



    if hasattr(filename, '__iter__') and not isinstance(filename, (str, bytes)):
        for text_file in filename:
            if pattern and not fnmatch(text_file, pattern):
                continue
            with open(text_file, mode=read_mode) as f:
                yield f
    else:
        filename_str = (
            filename.lower() if isinstance(filename, str) else
            str(filename).lower() if isinstance(filename, PurePath) else
            '')

        # Find arc_suffix ('.zip'/'.rar'/'') and pattern if it is in filename after suffix
        for arc_suffix in ('.zip', '.rar'):
            if arc_suffix in filename_str:
                filename_str_no_ext, pattern_parent = filename_str.split(arc_suffix, maxsplit=1)
                if pattern_parent:
                    pattern = str(PurePath(pattern_parent[1:]) / pattern)
                    filename_str = filename_str_no_ext + arc_suffix
                break
        else:
            arc_suffix = ''

        if arc_suffix == '.zip':
            from zipfile import ZipFile as ArcFile
        elif arc_suffix == '.rar':
            import rarfile
            ArcFile = rarfile.RarFile
            try:  # only try increase peformance
                # Configure RarFile Temp file size: keep ~1Gbit free, always take at least ~20Mbit:
                # decrease the operations number as we are working with big files
                io.DEFAULT_BUFFER_SIZE = max(io.DEFAULT_BUFFER_SIZE, 8192 * 16)
                import tempfile, psutil
                rarfile.HACK_SIZE_LIMIT = max(20_000_000,
                                              psutil.disk_usage(Path(tempfile.gettempdir()).drive).free - 1_000_000_000
                                              )
            except Exception as e:
                l.warning('%s: can not update settings to increase peformance', standard_error_info(e))
            read_mode = 'r' # RarFile need opening in mode 'r' (but it opens in binary_mode)
        if arc_suffix:
            with ArcFile(str(Path(filename_str).resolve().absolute()), mode='r') as arc_file:
                for text_file in arc_file.infolist():
                    if pattern and not fnmatch(text_file.filename, pattern):
                        continue

                    with arc_file.open(text_file.filename, mode=read_mode) as f:
                        break_flag = yield (f if binary_mode else io.TextIOWrapper(
                            f, encoding=encoding, errors='replace', line_buffering=True))  # , newline=None
                        if break_flag:
                            print(f'exiting after openined archived file "{text_file.filename}":')
                            print(arc_file.getinfo(text_file))
                            break
        else:
            if pattern and not fnmatch(filename, pattern):
                return
            with open(filename, mode=read_mode) as f:
                yield f

def path_on_drive_d(path_str: str = '/mnt/D',
                    drive_win32: str = 'D:',
                    drive_linux: str = '/mnt/D'):
    """convert path location on my drive to current system (Linux / Windows)"""
    if path_str is None:
        return None
    linux_next_to_d = re.match(f'{drive_linux}(.*)', path_str)
    if linux_next_to_d:
        if sys.platform == 'win32':
            path_str = f'{drive_win32}{linux_next_to_d.group(1)}'
    elif sys.platform != 'win32':
        win32_next_to_d = re.match(f'{drive_win32}(.*)', path_str)
        path_str = f'{drive_linux}{win32_next_to_d.group(1)}'
    return Path(path_str)


def import_file(path: PurePath, module_name: str):
    """Import a python module from a path. 3.4+ only.

    Does not call sys.modules[full_name] = path
    """
    from importlib import util

    f = (path / module_name).with_suffix('.py')
    try:
        spec = util.spec_from_file_location(module_name, f)
        mod = util.module_from_spec(spec)

        spec.loader.exec_module(mod)
    except ModuleNotFoundError as e:  #(Exception),
        print(standard_error_info(e), '\n- Can not load module', f, 'here . Skipping!')
        mod = None
    return mod


def st(current: int, descr: Optional[str] = '') -> bool:
    """
    Says if need to execute current step.
    Note: executs >= one step beginnig from ``start``
    Attributes: start: int, end: int, go: Optional[bool] = True:
    start, end: int, step control limits
    go: skip if False

    :param current: step#
    :param descr: step description to print
    :return desision: True if start <= current <= max(start, end)): allows one step if end <= start
    True => execute current st, False => skip
    """
    if (st.start <= current <= max(st.start, st.end)) and st.go:
        msg = f'Step {current}.\t{descr}'
        print(msg)
        print('-'*len(msg))
        return True
    return False

st.start = 0
st.end = 1e9  # big value
st.go = True


def call_with_valid_kwargs(func: Callable[[Any], Any], *args, **kwargs):
    """
    Calls func with extracted valid arguments from kwargs
    inspired by https://stackoverflow.com/a/9433836
    :param func: function you're calling
    :param args:
    :param kwargs:
    :return:
    """
    valid_keys = kwargs.keys() & func.__code__.co_varnames[len(args):func.__code__.co_argcount]
    return func(*args, **{k: kwargs[k] for k in valid_keys})


"""
TRASH
# p_sec.set_defaults(**{'--' + option_name: config.get(section_name, option_name)})
# print('\n==> '.join([s for s in e.args if isinstance(s, str)]))
# if can loose sections
# par_ok, par_bad_list= p.parse_known_args()
# for a in par_bad_list:
#     if a.startswith('--'):
#         p.add(a)
"""
