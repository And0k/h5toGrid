"""
Test file to verify that pytest is properly configured with the pixi environment.
This file can be used to check if VSCode's testing extension can detect and run tests.
"""

def test_pixi_environment():
    """
    Simple test to verify that pytest is working in the pixi environment.
    """
    # This test should always pass
    assert True

def test_python_version():
    """
    Test to check which Python version is being used.
    """
    import sys
    # Print the Python executable path to verify it's using the pixi environment
    print(f"Python executable: {sys.executable}")
    assert "pixi" in sys.executable.lower() or ".pixi" in sys.executable.lower()