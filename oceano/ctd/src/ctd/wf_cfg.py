from pathlib import Path
import re

path_cruise = Path(r"D:\Cruises\BalticSea\251201_ABP64")
path_db = (path_cruise / path_cruise / path_cruise.name.split("@", 1)[0]).with_suffix(".h5")

min_coord = "Lat:53, Lon:18.6"  # 10
max_coord = "Lat:60.55, Lon:30.3"  # includes Gulf Of Finland

# separate cruise number digits
cruise = re.match(r"(?P<year>\d\d)\d+_*(?P<vessel>\D+)(?P<num>\d+)", path_cruise.stem).groupdict()
devices = {}
