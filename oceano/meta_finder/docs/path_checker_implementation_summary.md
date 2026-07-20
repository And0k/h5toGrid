# Path Checker Implementation Summary

## Overview

Successfully implemented a path availability checker with renamed folder detection functionality for the meta_finder package. The tool reads paths from TSV files (meta_TCM.tsv or files_TCM.tsv), checks their availability, and detects renamed folders using similarity matching.

## Implementation Details

### Files Created

1. **[`src/meta_finder/path_checker.py`](src/meta_finder/path_checker.py)** - Core module with path checking and similarity detection logic
2. **[`src/meta_finder/path_checker_main.py`](src/meta_finder/path_checker_main.py)** - CLI entry point with argument parsing and execution
3. **[`tests/test_path_checker.py`](tests/test_path_checker.py)** - Comprehensive tests for core functionality (29 tests)
4. **[`tests/test_path_checker_main.py`](tests/test_path_checker_main.py)** - CLI tests (36 tests)
5. **[`src/meta_finder/README_path_checker.md`](src/meta_finder/README_path_checker.md)** - Complete documentation

### Files Modified

1. **[`pyproject.toml`](pyproject.toml)** - Added `check-paths` task to pixi tasks

## Key Features

### 1. Path Existence Checking
- Verifies if paths from TSV files exist on the filesystem
- Handles relative and absolute paths correctly
- Supports both single file and batch directory processing

### 2. Similarity-Based Renamed Folder Detection
- Reuses similarity matching functions from `match_dirs` package
- Implements hierarchical weighted similarity algorithm
- Compares folder names using common separators (`-`, `.`, `@`)
- Assigns confidence levels to matches

### 3. Confidence-Based Mapping
- **High Confidence (≥ 0.70)**: Direct mapping without markers
- **Medium Confidence (0.30 - 0.70)**: Mapping with `/?` marker
- **Low Confidence (< 0.30)**: Partial path with `/?` marker

### 4. Flexible CLI Interface
- Single file mode: Process one TSV file
- Directory mode: Process all matching TSV files in a directory
- Customizable output paths
- Configurable similarity cutoffs
- Verbose and quiet logging modes
- Custom glob patterns for file matching

## Architecture

### Module Structure

```
path_checker.py
├── read_tsv_paths()              # Read paths from TSV files
├── check_path_availability()       # Check if path exists
├── find_best_match()              # Find best matching path
├── find_similar_path_in_parent()  # Search for similar paths
├── generate_path_mapping()         # Generate old→new path mappings
├── write_mapping_tsv()           # Write mappings to TSV
├── check_paths_from_tsv()         # Main processing function
├── find_tsv_files()              # Find TSV files in directory
└── process_all_tsv_files()       # Batch processing
```

### Similarity Algorithm

The tool uses hierarchical weighted similarity matching:

1. **Name Part Extraction**: Split folder names using separators
   - Example: `cruise1@i01` → `["cruise1", "i01"]`

2. **Weighted Comparison**: Compare parts with decreasing weights
   - First part: weight = 1.0 + delta
   - Second part: weight = 0.5 + delta
   - Nth part: weight = (1/2^(n-1)) + delta

3. **Similarity Score**: Weighted average of part similarities

## Usage Examples

### Basic Usage

```bash
# Check a single TSV file
pixi run check-paths meta_TCM.tsv

# Check all TSV files in a directory
pixi run check-paths --directory ./cruises

# Specify custom output
pixi run check-paths meta_TCM.tsv --output path_mapping.tsv

# Adjust similarity threshold
pixi run check-paths meta_TCM.tsv --cutoff 0.5
```

### Python API

```python
from pathlib import Path
from meta_finder.path_checker import check_paths_from_tsv, MATCH_CUTOFF

check_paths_from_tsv(
    Path("meta_TCM.tsv"),
    Path("path_mapping.tsv"),
    Path("./Cruises"),
    MATCH_CUTOFF
)
```

## Testing

### Test Coverage

- **29 tests** for core functionality in `test_path_checker.py`
- **36 tests** for CLI functionality in `test_path_checker_main.py`
- **Total: 65 tests**, all passing

### Test Categories

1. **TSV Reading Tests**
   - Simple paths
   - Paths with comments
   - Multi-column TSV files
   - Empty files
   - Non-existent files

2. **Similarity Matching Tests**
   - Exact matches
   - Similar matches
   - No matches
   - Empty candidates

3. **Path Availability Tests**
   - Existing paths
   - Non-existent paths
   - Special characters

4. **Mapping Generation Tests**
   - All paths exist
   - All paths don't exist
   - Mixed scenarios
   - Uncertain markers

5. **CLI Tests**
   - Argument parsing
   - Validation
   - Single file processing
   - Directory processing
   - Error handling
   - Integration workflows

## Configuration

### Constants

```python
MATCH_CUTOFF = 0.3                    # Minimum similarity for matches
HIGH_CONFIDENCE_THRESHOLD = 0.70        # High confidence threshold
LOW_CONFIDENCE_THRESHOLD = 0.30         # Low confidence threshold
NAME_SEPARATORS = r"[-.@]"             # Folder name separators
TSV_DELIMITER = "\t"                    # TSV delimiter
ENCODING = "utf-8"                     # File encoding
UNCERTAIN_MARKER = "/?"                  # Uncertain match marker
```

## Integration with Existing Code

### Reused Components

- **Similarity Matching**: Imported from `match_dirs` package
  - [`hierarchical_weighed_similarity()`](C:/Work/Python/AB_SIO_RAS/cruises_organizer/match_dirs/src/matcher.py:20)
  - [`HIGH_CONFIDENCE_THRESHOLD`](C:/Work/Python/AB_SIO_RAS/cruises_organizer/match_dirs/src/matcher.py:8)
  - [`LOW_CONFIDENCE_THRESHOLD`](C:/Work/Python/AB_SIO_RAS/cruises_organizer/match_dirs/src/matcher.py:9)

- **Logging**: Uses existing [`logging_config`](src/meta_finder/logging_config.py:1) module
- **Conventions**: Follows meta_finder coding standards and PEP 8

## Output Format

### TSV Mapping File

Two-column format:
1. **Old Path**: Original path from input TSV
2. **New Path**: Mapped path (existing, similar, or partial with marker)

### Example

```tsv
cruise1/device1/data.txt	C:\full\path\to\cruise1\device1\data.txt
cruise2/device2/data.txt	C:\full\path\to\cruise2_renamed\device2\data.txt/?
cruise3/device3/data.txt	cruise3/?
```

## Error Handling

### Graceful Degradation

- **Non-existent TSV files**: Raises `FileNotFoundError` with clear message
- **Permission errors**: Logs warning and continues processing
- **Invalid arguments**: Validates and returns error messages
- **Keyboard interrupts**: Returns exit code 130
- **Unexpected errors**: Logs error and returns exit code 1

### Logging Levels

- **DEBUG**: Detailed similarity calculations (verbose mode)
- **INFO**: Processing progress and success messages
- **WARNING**: Non-critical issues (e.g., no TSV files found)
- **ERROR**: Critical errors that stop processing

## Performance Considerations

### Optimization Strategies

1. **Lazy Evaluation**: Uses generators for memory efficiency
2. **Early Termination**: Stops searching when good match found
3. **Batch Processing**: Can process multiple files in one run
4. **Caching**: Similarity scores calculated once per comparison

### Scalability

- Handles thousands of paths efficiently
- Memory usage scales linearly with input size
- Suitable for large cruise data directories

## Future Enhancements

### Potential Improvements

1. **Additional Similarity Algorithms**
   - Levenshtein distance
   - Jaccard similarity
   - Cosine similarity for vectorized names

2. **Interactive Mode**
   - Manual confirmation of uncertain matches
   - Visual diff of folder names

3. **Batch Operations**
   - Apply mappings to rename folders
   - Generate rename scripts
   - Integration with version control

4. **Performance**
   - Parallel processing for large datasets
   - Caching of similarity calculations
   - Progressive output for long-running operations

5. **Enhanced Features**
   - Support for multiple similarity metrics
   - Custom separator patterns
   - Integration with file metadata

## Documentation

### User Documentation

- **README**: [`README_path_checker.md`](src/meta_finder/README_path_checker.md)
  - Overview and features
  - Installation instructions
  - Usage examples
  - API reference
  - Troubleshooting guide

### Code Documentation

- **Docstrings**: All functions have comprehensive docstrings
- **Type Hints**: Full type annotations using Python 3.11+ syntax
- **Comments**: Intent-based comments for complex logic
- **Examples**: Usage examples in docstrings

## Best Practices

### For Users

1. **Start with default settings**: Use default 0.3 cutoff initially
2. **Review uncertain mappings**: Always check paths marked with `/?`
3. **Use appropriate search directory**: Specify where renamed folders are likely
4. **Process in batches**: Use directory mode for multiple files
5. **Backup original files**: Preserve original TSV files before processing

### For Developers

1. **Follow PEP 8**: Adhere to Python style guidelines
2. **Add tests**: Write tests for new features
3. **Update documentation**: Keep docs in sync with code
4. **Maintain backward compatibility**: Don't break existing functionality
5. **Use type hints**: Improve code clarity and IDE support

## Conclusion

The path checker implementation provides a robust, well-tested solution for verifying path availability and detecting renamed folders in cruise data directories. It integrates seamlessly with the existing meta_finder package, follows best practices, and includes comprehensive documentation and testing.

The tool is production-ready and can be used immediately to:
- Verify paths from TSV files
- Detect renamed folders using similarity matching
- Generate mapping files with confidence indicators
- Process single files or entire directories
- Customize behavior via command-line options

All 65 tests pass, ensuring reliability and correctness of the implementation.
