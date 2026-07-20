import re

def test_pattern_matching():
    """Test pattern matching for split files."""
    base_name = "230508_1551bin2s@i03.tsv"
    file_name = "230508_1651bin2s@i03.tsv"

    # Extract the pattern from the filename (everything except the timestamp part)
    # Pattern: {yymmdd_HHMM}{rest of filename}
    pattern_match = re.match(r'(\d{6}_\d{4})(.*)', base_name)
    if pattern_match:
        timestamp_part = pattern_match.group(1)
        rest_part = pattern_match.group(2)
        print(f"Pattern match: timestamp_part={timestamp_part}, rest_part={rest_part}")

        # Test matching with escaped rest_part
        escaped_pattern = r'(\d{6}_\d{4})(' + re.escape(rest_part) + r')'
        print(f"Escaped pattern: {escaped_pattern}")
        file_match = re.match(escaped_pattern, file_name)
        if file_match:
            print(f"Match with escaped pattern: {file_match.groups()}")
        else:
            print("No match with escaped pattern")

        # Test matching with unescaped rest_part
        unescaped_pattern = r'(\d{6}_\d{4})(' + rest_part + r')'
        print(f"Unescaped pattern: {unescaped_pattern}")
        file_match = re.match(unescaped_pattern, file_name)
        if file_match:
            print(f"Match with unescaped pattern: {file_match.groups()}")
        else:
            print("No match with unescaped pattern")

if __name__ == "__main__":
    test_pattern_matching()