# Path Availability Checker

## Overview

The Path Availability Checker is a tool designed to verify the existence of paths stored in TSV files (such as `meta_TCM.tsv` or `files_TCM.tsv`) and detect renamed folders using similarity matching. This is particularly useful when working with cruise data directories that may have been reorganized or renamed over time.

## Features

- **Path Existence Checking**: Verifies if paths listed in TSV files still exist on the filesystem
- **Renamed Folder Detection**: Uses similarity matching to detect folders that may have been renamed
- **Confidence-Based Mapping**: Assigns confidence levels to matches and marks uncertain mappings
- **Batch Processing**: Can process individual TSV files or entire directories
- **Configurable Similarity Thresholds**: Adjust cutoff values for match sensitivity

## How It Works

### Similarity Matching

The tool uses hierarchical weighted similarity matching, adapted from the `match_dirs` package. This algorithm:

1. Splits folder names into parts using common separators (`-`, `.`, `@`)
2. Compares corresponding parts with decreasing weights (earlier parts are more important)
3. Calculates a similarity score between 0.0 and 1.0

### Confidence Levels

- **High Confidence (≥ 0.70)**: Path maps directly to the matched path
- **Medium Confidence (0.30 - 0.70)**: Path maps to the matched path with `/?` marker
- **Low Confidence (< 0.30)**: No match found, path maps to partial path with `/?` marker

## Installation

The tool is part of the `meta_finder` package and requires no additional dependencies beyond the main package requirements.

## Usage

### Command Line Interface

#### Check a Single TSV File

```bash
# Using pixi
pixi run check-paths meta_TCM.tsv

# Using Python directly
python -m meta_finder.path_checker_main meta_TCM.tsv
```

#### Check All TSV Files in a Directory

```bash
# Using pixi
pixi run check-paths --directory ./cruises

# Using Python directly
python -m meta_finder.path_checker_main --directory ./cruises
```

#### Specify Custom Output Path

```bash
python -m meta_finder.path_checker_main meta_TCM.tsv --output path_mapping.tsv
```

#### Specify Custom Similarity Cutoff

```bash
python -m meta_finder.path_checker_main meta_TCM.tsv --cutoff 0.5
```

#### Specify Base Search Directory

```bash
python -m meta_finder.path_checker_main meta_TCM.tsv --search-dir ./Cruises
```

#### Verbose Mode

```bash
python -m meta_finder.path_checker_main meta_TCM.tsv --verbose
```

#### Quiet Mode

```bash
python -m meta_finder.path_checker_main meta_TCM.tsv --quiet
```

### Command Line Options

| Option | Short | Description |
|--------|-------|-------------|
| `--directory` | `-d` | Process all TSV files in the specified directory |
| `--output` | `-o` | Specify output path for mapping file |
| `--search-dir` | `-s` | Base directory for similarity search |
| `--cutoff` | `-c` | Minimum similarity threshold (default: 0.3) |
| `--pattern` | `-p` | Glob pattern for TSV files in directory mode (default: `*_TCM.tsv`) |
| `--verbose` | `-v` | Enable verbose logging |
| `--quiet` | `-q` | Suppress non-error logging |

## Output Format

The tool generates TSV files with two columns:

1. **Old Path**: The original path from the input TSV file
2. **New Path**: The mapped path (either the same if it exists, a similar path if found, or partial path with `/?` marker)

### Example Output

```tsv
old_path1/device1/data.txt	C:\full\path\to\old_path1\device1\data.txt
old_path2/device2/data.txt	C:\full\path\to\old_path2_renamed\device2\data.txt/?
old_path3/device3/data.txt	old_path3/?
```

## Python API

### Basic Usage

```python
from pathlib import Path
from meta_finder.path_checker import check_paths_from_tsv, MATCH_CUTOFF

# Check paths from a TSV file
tsv_file = Path("meta_TCM.tsv")
output_file = Path("path_mapping.tsv")
base_search_dir = Path("./Cruises")

check_paths_from_tsv(tsv_file, output_file, base_search_dir, MATCH_CUTOFF)
```

### Advanced Usage

```python
from pathlib import Path
from meta_finder.path_checker import (
    read_tsv_paths,
    generate_path_mapping,
    write_mapping_tsv,
    MATCH_CUTOFF,
)

# Read paths from TSV
paths = read_tsv_paths(Path("meta_TCM.tsv"))

# Generate mappings
base_dir = Path("./Cruises")
mappings = generate_path_mapping(paths, base_dir, MATCH_CUTOFF)

# Write to file
write_mapping_tsv(mappings, Path("output.tsv"))
```

### Custom Similarity Threshold

```python
from meta_finder.path_checker import check_paths_from_tsv

# Use a higher threshold for more strict matching
check_paths_from_tsv(
    tsv_file,
    output_file,
    base_search_dir,
    cutoff=0.5  # Only accept matches with 50%+ similarity
)
```

## How Similarity Matching Works

The similarity matching algorithm uses a hierarchical approach:

1. **Name Part Extraction**: Folder names are split into parts using separators (`-`, `.`, `@`)
   - Example: `cruise1@i01` → `["cruise1", "i01"]`

2. **Weighted Comparison**: Each part is compared with decreasing weight
   - First part: highest weight (most important)
   - Subsequent parts: exponentially decreasing weight

3. **Similarity Score**: Calculated as weighted average of part similarities

### Example

Comparing `cruise1@i01` with `cruise1@i02`:
- Part 1: `cruise1` vs `cruise1` → 1.0 similarity (weight: 1.0)
- Part 2: `i01` vs `i02` → 0.67 similarity (weight: 0.5)
- Overall: ~0.89 similarity (high confidence)

## Best Practices

1. **Start with Default Cutoff**: Use the default 0.3 cutoff initially, then adjust based on results
2. **Review Uncertain Mappings**: Always manually review paths marked with `/?`
3. **Use Appropriate Search Directory**: Specify the base directory where renamed folders are likely to be
4. **Process in Batches**: For large datasets, process directories rather than individual files
5. **Keep Original Files**: Always preserve the original TSV files before running the checker

## Troubleshooting

### No Matches Found

If most paths show `/?` markers:
- Lower the cutoff value (e.g., `--cutoff 0.2`)
- Verify the search directory is correct
- Check if folders have been completely removed (not just renamed)

### Too Many False Positives

If unrelated folders are being matched:
- Increase the cutoff value (e.g., `--cutoff 0.5`)
- Review the similarity algorithm for your specific naming patterns
- Consider customizing the separator patterns

### Permission Errors

If you encounter permission errors:
- Run with appropriate file system permissions
- Check directory access rights
- Use verbose mode (`--verbose`) to see detailed error messages

## Testing

The tool includes comprehensive tests:

```bash
# Run all path checker tests
pixi run -e test python -m pytest tests/test_path_checker.py -v

# Run CLI tests
pixi run -e test python -m pytest tests/test_path_checker_main.py -v
```

## Integration with meta_finder

The path checker is designed to work seamlessly with the meta_finder package:

- Uses the same logging configuration
- Follows the same code style and conventions
- Can be integrated into existing meta_finder workflows
- Compatible with TSV files generated by meta_finder

## Device Directory Checker

The `check_device_dirs.py` script extracts unique device directory paths from all `*_files_TCM.tsv` files in `meta/collection/`, checks their availability on disk, and optionally creates directory symlinks that replicate the parent hierarchy.

### Command Examples

```bash
# Print report for all *_files_TCM.tsv files
python src/meta_finder/post_processing/check_device_dirs.py

# Only the latest *_files_TCM.tsv file
python src/meta_finder/post_processing/check_device_dirs.py --latest-only

# Save report to file
python src/meta_finder/post_processing/check_device_dirs.py -o meta/collection/device_dirs_report.tsv

# Create symlinks to all device directories (replicating hierarchy)
python src/meta_finder/post_processing/check_device_dirs.py meta/collection/device_dirs_symlinks

# Combine: report to file, and create symlinks
python src/meta_finder/post_processing/check_device_dirs.py meta/collection/device_dirs_symlinks -o meta/collection/device_dirs_report.tsv
```

> **Note:** Creating symlinks on Windows requires either running as Administrator or enabling Developer Mode.

### Symlink Hierarchy

When creating symlinks, the script:

1. **Resolves targets** — uses the renamed/similar path for `RENAMED`/`UNCERTAIN` entries, or the original path for `OK`.
2. **Filters nested directories** — skips any device directory whose resolved target is inside another device directory's target (logged as a warning). This prevents creating symlinks inside existing symlinks.
3. **Finds the longest common parent** — computes the deepest shared ancestor directory among all resolved targets.
4. **Replicates hierarchy** — for each device dir, creates the symlink at `symlinks_dir / <relative_path_from_common_parent> / <target_name>`, preserving the original parent directory structure.

For example, the resulting symlinks directory structure (for given device dirs the determined common parent is `C:/Cruises`):

```
symlinks_dir/
├── Cruise1/
│   ├── DeviceA -> C:/Cruises/Cruise1/DeviceA
│   └── DeviceB -> C:/Cruises/Cruise1/DeviceB
└── Cruise2/
    └── DeviceC -> C:/Cruises/Cruise2/DeviceC
```

If fewer than 2 non-nested device dirs exist, the script falls back to creating flat symlinks directly in `symlinks_dir`.

### Report Output

The report is a TSV file with comment header explaining columns:

| Column | Description |
|--------|-------------|
| `status` | `OK`, `RENAMED`, `UNCERTAIN`, or `MISSING` |
| `device_dir` | Original device directory path from the TSV |
| `similar_path` | Suggested renamed path (when device_dir not found) |
| `score` | Similarity score between original and similar path (0.0–1.0) |
| `sources` | `*_files_TCM.tsv` files where this device_dir appears |

### Command Line Options

| Option | Description |
|--------|-------------|
| `symlinks_dir` (positional) | Create symlinks to all device directories in this directory, replicating parent hierarchy |
| `--collection-dir` | Directory containing `*_files_TCM.tsv` files (default: `meta/collection/`) |
| `--cutoff` | Minimum similarity threshold (default: 0.3) |
| `--output`, `-o` | Save report to file instead of stdout |
| `--latest-only` | Only process the latest `*_files_TCM.tsv` file |
| `--verbose`, `-v` | Enable verbose logging |

## Future Enhancements

Potential improvements for future versions:

- Support for additional similarity algorithms (Levenshtein distance, etc.)
- Interactive mode for manual confirmation of uncertain matches
- Support for batch renaming based on mappings
- Integration with version control systems
- Performance optimizations for large datasets

## Contributing

When contributing to the path checker:

1. Follow PEP 8 style guidelines
2. Add tests for new features
3. Update documentation
4. Ensure backward compatibility

## License

This tool is part of the meta_finder package and follows the same license (MIT).
