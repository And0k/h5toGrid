# Pixi + VSCode Integration Guide

This document provides instructions on how to properly set up and use VSCode with your pixi environment.

## Quick Setup Verification

To verify that everything is working correctly:

1. Open VSCode in your project directory
2. Check that the Python interpreter is set to:
   ```
   ${workspaceFolder}/.pixi/envs/default/python.exe
   ```
3. Try running the tests in `tests/test_pixi_setup.py`
4. The tests should pass and show that they're using the pixi environment

## Running Tests

You can run tests in multiple ways:

### 1. Using VSCode's Testing Extension
- Open the Testing panel (Ctrl+Shift+T)
- You should see your tests listed
- Click on the "Run" button to execute tests

### 2. Using the Terminal
```bash
# Run all tests
.pixi/envs/default/Scripts/pytest.exe

# Run specific test file
.pixi/envs/default/Scripts/pytest.exe tests/test_pixi_setup.py

# Run with verbose output
.pixi/envs/default/Scripts/pytest.exe -v
```

### 3. Using Pixi Commands
```bash
# If you have pixi configured with test commands
pixi run test
```

## Troubleshooting

### If VSCode Doesn't Detect Tests

1. Reload the VSCode window:
   - Press `Ctrl+Shift+P`
   - Run "Developer: Reload Window"

2. Manually select the Python interpreter:
   - Press `Ctrl+Shift+P`
   - Run "Python: Select Interpreter"
   - Choose the interpreter from `${workspaceFolder}/.pixi/envs/default/python.exe`

3. Check the Python output in VSCode:
   - Press `Ctrl+Shift+P`
   - Run "Python: Show Output"
   - Look for any error messages

### If Tests Fail Due to Import Errors

Make sure your Python path includes your project directories:
- The workspace settings include `"python.analysis.extraPaths"` which should help with imports
- If you have custom module locations, add them to this setting

## Configuration Files

### Workspace Settings (.vscode/settings.json)

Contains the configuration to use the pixi environment:
- Python interpreter path
- Pytest executable path
- Test arguments
- Environment activation settings

### User Settings

Your global VSCode settings may contain Python configurations that conflict with project-specific settings. The workspace settings should override these, but if you experience issues, you might need to temporarily disable global Python settings.

## Best Practices

1. Always use the pixi environment's Python executable for running tests
2. Keep your workspace settings specific to the pixi environment
3. Use relative paths with `${workspaceFolder}` for portability
4. Regularly update your pixi environment dependencies
5. If you add new dependencies to your pixi environment, restart VSCode to ensure the changes are detected

## Verifying the Environment

To check that you're using the correct environment:

```bash
# Check Python version and path
.pixi/envs/default/Scripts/python.exe --version
.pixi/envs/default/Scripts/python.exe -c "import sys; print(sys.executable)"

# Check pytest version
.pixi/envs/default/Scripts/pytest.exe --version

# List installed packages
.pixi/envs/default/Scripts/pip.exe list
```

This should show that you're using the Python and pytest from your pixi environment.