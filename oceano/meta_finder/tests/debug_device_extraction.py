import re
from pathlib import Path
from meta_finder.parse_data_file_name import normalize_device_id

# Simulate the _extract_device_ids_from_column_name function
def _extract_device_ids_from_column_name(column_name: str):
    """
    Extract device IDs from a column name.
    For combined devices like 'i05_14', uses parse_device_group after converting '_' to ','.
    """

    # Split on first non-digit followed by underscore to get the device part
    parts = re.split(r"[^\d]_", column_name, 1)
    if len(parts) < 2:
        return []

    device_part = parts[1]

    # Check if this is a combined device (has underscores between digits)
    if "_" in device_part:
        # Convert underscores between digits to commas for parsing
        formatted, n_found = re.subn(r"(\d)_(\d)", r"\1,\2", device_part)
        if n_found:
            try:
                # Parse as a device group to get individual device IDs
                # For now, just simulate this - we'd need to import parse_device_group
                return [formatted.replace(',', '_')]  # Simplified for testing
            except:
                pass  # Fall back to normalizing if parsing fails
    # Simple device pattern, normalize it
    return [normalize_device_id(device_part)]

# Test with the actual header from the test file
header = [
    "Time",
    "Vabs_i03",
    "Vdir_i03",
    "v_i03",
    "u_i03",
    "Vabs_i04",
    "Vdir_i04",
    "v_i04",
    "u_i04",
    "Vabs_i37",
    "Vdir_i37",
    "v_i37",
    "u_i37",
    ""
]

dev_ids = ['i03', 'i04', 'i37']
device_columns = {}

print("Testing device extraction:")
print("Header:", header)
print("Requested device IDs:", dev_ids)
print()

# Normalize the requested device IDs
normalized_dev_ids = [normalize_device_id(device_id) for device_id in dev_ids]
print("Normalized requested device IDs:", normalized_dev_ids)
print()

for i, column_name in enumerate(header):
    col_device_ids = _extract_device_ids_from_column_name(column_name)
    print(f"Column {i}: '{column_name}' -> Device IDs: {col_device_ids}")

    # Check if any of the extracted device IDs match our requested device IDs
    for device_id in col_device_ids:
        normalized_device_id = normalize_device_id(device_id)
        if normalized_device_id in normalized_dev_ids:
            if normalized_device_id not in device_columns:
                device_columns[normalized_device_id] = []
            device_columns[normalized_device_id].append(i)

print()
print("Final device_columns:", device_columns)

# Check the condition that's causing the warning
any_match = any(normalized_device_id in device_columns for normalized_device_id in normalized_dev_ids)
print(f"Any requested device ID found in device_columns: {any_match}")

if not any_match and normalized_dev_ids:
    print("WARNING: No device-specific columns found!")
else:
    print("SUCCESS: Device-specific columns found!")