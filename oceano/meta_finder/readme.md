<p align="center"><img src="docs/images/logo.png" alt="Meta Finder" width="128"></p>

[Russian readme](readme_Ru.md)

# Meta Finder

Collects metadata from files in cruise directories for AB SIO RAS inclinometers (tilt current meters), wave gauges, and combined devices.

## Documentation

### [User guides](docs/user_guide/)

- [Getting Started](docs/user_guide/getting_started.md)
- [CLI Guide](docs/user_guide/cli.md)
- [Configuration Guide](docs/user_guide/configuration.md)
- [Input / Output Guide](docs/user_guide/input_output.md)
- [Processing Guide](docs/user_guide/processing.md)
- [Path Checker](src/meta_finder/post_processing/README.md)
- [Console Messages](docs/user_guide/console_messages.md)
- [Metadata Table Description](docs/user_guide/metadata_table.md)

### [Reference](docs/reference/)

Exact, authoritative specs.

- [Input / Output Format Specification](docs/reference/io_formats.md)
- [Configuration Reference](docs/reference/config_reference.md)

### [Python developer guide](docs/python_developer_guide/)

- [Examples](docs/python_developer_guide/examples.md)

### [Project developer guide](docs/project_developer_guide/)

Internal architecture and build instructions.

- [CLI Internals](docs/project_developer_guide/CLI.md)
- [Documentation Authoring Contract](../tcm/docs/project_developer_guide/doc_authoring.md)
- [Codebase Analysis](docs/project_developer_guide/codebase_analysis.md)
- [HDF5 Functionality](docs/project_developer_guide/hdf5_functionality.md)
- [Multiple Intervals Handling](docs/project_developer_guide/multiple_intervals.md)
- [Workflow Tree](docs/project_developer_guide/workflow_tree.md)

## Integration

`meta_finder` is the discovery and metadata layer for
[tcm](../tcm/) — the TCM data processing pipeline. `tcm` reuses
`meta_finder`'s device discovery, metadata I/O, and file enumeration. For
details, see [tcm's meta_finder Integration](../tcm/docs/reference/meta_finder_integration.md).
