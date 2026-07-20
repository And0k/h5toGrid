import re
from pathlib import Path

# Test the pattern matching
dated_dir_pattern = "[0-9][0-9][0-9][0-9][0-9][0-9]*"
pattern = re.compile(dated_dir_pattern)

# Check if our directory matches
dir_name = "230507_ABP53_inclinometer"
match = pattern.match(dir_name)
print(f"Directory name: {dir_name}")
print(f"Pattern: {dated_dir_pattern}")
print(f"Matches: {match is not None}")

if match:
    print(f"Matched part: {match.group()}")