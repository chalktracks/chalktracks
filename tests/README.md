# Chalk Project Tests

This directory contains the test suite for the chalk line following project using **pytest**.

## Test Structure

- `test_utils.py` - Tests for utility functions (file operations, symlinks)
- `test_cli.py` - Tests for CLI command discovery and help system
- `test_commands.py` - Tests for individual command implementations
- `test_label_tool.py` - Tests for the interactive labeling tool (core functions only)
- `test_similarity_filter.py` - Tests for SSIM-based image filtering

## Running Tests

### Run All Tests
```bash
# Using pytest directly
pytest tests/ -v

# Using our custom test runner
python tests/run_tests.py

# From project root
python -m pytest tests/
```

### Run Specific Test Module
```bash
# Using pytest
pytest tests/test_utils.py -v

# Using our custom runner
python tests/run_tests.py test_utils
```

### Run Specific Test Class
```bash
pytest tests/test_utils.py::TestUtils -v
```

### Run Specific Test Method
```bash
pytest tests/test_utils.py::TestUtils::test_put_files_into_dir_symlink -v
```

### Run Tests with Coverage
```bash
pytest tests/ --cov=chalk --cov-report=term-missing
```

### Run Tests with Different Verbosity
```bash
# Quiet mode
pytest tests/ -q

# Very verbose
pytest tests/ -vv

# Show test durations
pytest tests/ --durations=10
```


### Debugging Tests

```bash
# Run single test with full output
pytest tests/test_utils.py::TestUtils::test_specific_function -v -s

# Stop on first failure
pytest tests/ -x

# Enter debugger on failure
pytest tests/ --pdb
```
