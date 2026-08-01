"""TCM GUI — thin entry point for PyInstaller and direct execution.

Launches the Tkinter GUI for interactive inclinometer data processing.

Usage::

    python scripts/tcm_gui.py "_raw/*i*.txt"

For ``python -m tcm_gui`` invocation, use ``tcm_gui/__main__.py`` instead.
"""
from tcm_gui.app import main

if __name__ == "__main__":
    main()
