import pandas as pd
from pathlib import Path
from typing import Dict, Any, Union, List
from datetime import datetime
import sys
import json

def format_datetime_for_output(dt):
    """Convert datetime-like objects to ISO string format."""
    if pd.isna(dt):
        return None
    if isinstance(dt, (pd.Timestamp, datetime)):
        return dt.isoformat()
    if isinstance(dt, pd.Timestamp):
        return dt.isoformat()
    return str(dt)


def calculate_column_stats(series: pd.Series) -> List[Union[str, float, None]]:
    """Calculate statistics for a single column."""
    series = series.dropna()  # Remove NaN values for calculations

    if series.empty:
        return [None, None, None, None]

    dtype_name = str(series.dtype)

    if "datetime" in dtype_name or "date" in dtype_name:
        # For datetime columns
        try:
            min_val = format_datetime_for_output(series.min())
            max_val = format_datetime_for_output(series.max())
            # For datetime, mean and median are not meaningful, so we set them to None
            mean_val = None
            median_val = None
        except Exception:
            min_val = max_val = mean_val = median_val = None
    elif "object" in dtype_name or "string" in dtype_name:
        # For object/string columns, only min/max are meaningful
        try:
            min_val = str(series.min()) if len(series) > 0 else None
            max_val = str(series.max()) if len(series) > 0 else None
            mean_val = median_val = None
        except Exception:
            min_val = max_val = mean_val = median_val = None
    else:
        # For numeric columns
        try:
            min_val = float(series.min())
            max_val = float(series.max())
            mean_val = float(series.mean())
            median_val = float(series.median())
        except Exception:
            min_val = max_val = mean_val = median_val = None

    return [min_val, max_val, mean_val, median_val]


def analyze_hdf5_file(hdf5_path: str) -> Dict[str, Dict[str, List[Union[str, float, None]]]]:
    """Analyze HDF5 file and compute statistics for all tables and columns using pandas."""
    results = {}

    # Process the entire file in one pass using pandas HDFStore
    with pd.HDFStore(hdf5_path, mode="r") as store:
        keys = store.keys()

        # Process each key and collect statistics
        for key in keys:
            try:
                # Read the DataFrame using pandas
                df = store.get(key)

                if isinstance(df, pd.DataFrame):
                    table_stats = {}

                    # Process regular columns
                    for col in df.columns:
                        series = df[col]
                        stats = calculate_column_stats(series)
                        table_stats[col] = stats

                    # Process index if it's datetime-like
                    if isinstance(df.index, pd.DatetimeIndex):
                        idx_series = df.index.to_series()
                        stats = calculate_column_stats(idx_series)
                        table_stats["index"] = stats

                    # Use the key without leading slash as table name
                    results[key.lstrip("/")] = table_stats

            except Exception:
                # Skip non-DataFrame objects or problematic keys
                continue

    return results


def save_to_toon_format(results: Dict[str, Dict[str, List[Union[str, float, None]]]], output_path: str):
    """
    Save results in TOON format (Token-Oriented Object Notation).
    TOON is a compact, human-readable format.
    This implementation creates a text file with the requested structure.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        # Comment in file the data format we will write:
        f.write("# {таблица: {колонка: [мин, макс, среднее, медиана]}}")
        for table_idx, (table_name, columns_stats) in enumerate(results.items()):
            f.write(f"{table_name}: {{\n")  # Start table block

            # Write each column's statistics
            for col_idx, (col_name, stats) in enumerate(columns_stats.items()):
                # Format the stats list as [min, max, mean, median]
                stats_str = "["
                for j, val in enumerate(stats):
                    if val is None:
                        stats_str += "null"
                    elif isinstance(val, str):
                        stats_str += f'"{val}"'
                    else:  # float or int
                        stats_str += str(val)

                    if j < len(stats) - 1:  # Add comma if not the last element
                        stats_str += ", "

                stats_str += "]"

                # Write the column entry
                f.write(f"  {col_name}: {stats_str}")  # Indent column

                # Add comma if not the last column
                if col_idx < len(columns_stats) - 1:
                    f.write(",")
                f.write("\n")  # Newline after each column entry

            f.write("}\n")  # Close table block
            # Add an empty line between tables for better readability
            if table_idx < len(results) - 1:
                f.write("\n")


def main(input_file: str|Path, output_file: str|Path):
    """Main function to process HDF5 file and create TOON output."""
    print(f"Analyzing HDF5 file: {input_file}")
    results = analyze_hdf5_file(input_file)

    print(f"Saving results to TOON file: {output_file}")
    save_to_toon_format(results, output_file)

    print("Analysis completed. Here are results in JSON format:")
    print(json.dumps(
        results,
        indent=4,
        ensure_ascii=False,
        sort_keys=True,
        default=str,  # Handle non-serializable objects
    ))


if __name__ == "__main__":
    n = len(sys.argv)
    if n != 3:
        print("Usage: python script.py <input.h5> <output.toon>")
    if n >= 2:
        input_file = sys.argv[1]
    else:
        input_file = input("Enter argument 1: the path to the input HDF5 file:\n")
    if n >= 3:
        output_file = sys.argv[2]
    else:
        output_file = input(
            "Enter argument 2: the path for the output TOON file \n"
            "[default HDF5 file name with `.stat.toon`] extension]:\n"
        )
        if not output_file.strip():
            output_file = Path(input_file)
            output_file = output_file.with_suffix(".stat.toon")
        if output_file.is_file():
            stat = output_file.stat()
            if stat.st_size:
                out_file_date = datetime.fromtimestamp(stat.st_mtime)
                b = input(
                    f"Old (modified {out_file_date}) {output_file.name} exists. Overwrite [y]/n?",
                )
                if b and b not in ["Yy"]:
                    sys.exit(1)
                    print(f"Answered '{b}' => Skip processing")

    main(input_file, output_file)