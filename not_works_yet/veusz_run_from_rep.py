import logging
import re
from datetime import datetime
from os import chdir as os_chdir, getcwd as os_getcwd, environ as os_environ
from pathlib import Path, PurePath
import sys
from sys import platform as sys_platform, stdout as sys_stdout
from time import sleep
from typing import Any, Callable, Dict, Iterator, Optional, Tuple, Union

veusz_path = 'd:/workCode/veusz/veusz'

sep = ';' if sys_platform == 'win32' else ':'
# to find Veusz executable (Windows only):
os_environ['PATH'] += f'{sep}{veusz_path}'
sys.path.append(veusz_path)
sys.path.append(str(Path(veusz_path).parent))
import veusz_main  # ModuleNotFoundError: No module named 'veusz.helpers.qtloops'

# for Linux set in ./bash_profile if not in Python path yet:
# PYTHONPATH=$PYTHONPATH:$HOME/Python/other_sources/veusz
# export PYTHONPATH
# if you compile Veusz also may be you add there
# VEUSZ_RESOURCE_DIR=/usr/share/veusz
# export VEUSZ_RESOURCE_DIR

#veusz = import_file(veusz_path, 'embed')











from veusz import veusz_main
print('Ok')