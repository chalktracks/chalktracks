#!/usr/bin/env python3
"""
Test runner script for the chalk project using pytest.

This script runs all tests and provides a summary of results.
Can be run as: python -m tests.run_tests
"""

import sys
import subprocess
from pathlib import Path

# Add the project root to the path so we can import chalk modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_all_tests():
    """Run all tests using pytest."""
    
    test_dir = Path(__file__).parent
    
    # Run pytest with verbose output and coverage
    cmd = [
        sys.executable, '-m', 'pytest', 
        str(test_dir),
        '-v',  # verbose
        '--tb=short',  # short traceback format
        '--durations=10',  # show 10 slowest tests
    ]
    
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


def run_specific_test(test_module):
    """Run a specific test module using pytest."""
    
    test_dir = Path(__file__).parent
    
    # Handle both forms: test_utils and tests.test_utils
    if not test_module.startswith('test_'):
        test_module = f'test_{test_module}'
    
    test_file = test_dir / f"{test_module}.py"
    
    if not test_file.exists():
        print(f"Test file '{test_file}' does not exist")
        return 1
    
    # Run pytest on specific file
    cmd = [
        sys.executable, '-m', 'pytest', 
        str(test_file),
        '-v',  # verbose
        '--tb=short',  # short traceback format
    ]
    
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except Exception as e:
        print(f"Error running test module '{test_module}': {e}")
        return 1


if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Run specific test module
        test_module = sys.argv[1]
        exit_code = run_specific_test(test_module)
    else:
        # Run all tests
        exit_code = run_all_tests()
    
    sys.exit(exit_code)
