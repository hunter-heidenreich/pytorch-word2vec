# Testing Guide

This guide explains how to run tests for the modern-word2vec project.

## Setup

The project uses `uv` for dependency management and `pytest` for testing.

### Install Test Dependencies

```bash
uv sync --extra test
```

This installs:
- `pytest` - Test framework
- `pytest-cov` - Coverage reporting 
- `pytest-mock` - Mocking utilities

## Running Tests

### Run All Tests
```bash
uv run pytest
```

### Run Tests with Coverage
```bash
uv run pytest --cov=src/modern_word2vec --cov-report=term-missing --cov-report=html
```

### Run Specific Test Files
```bash
# Test only hierarchical softmax
uv run pytest tests/test_hierarchical_softmax.py tests/test_huffman_tree.py

# Test a specific test class
uv run pytest tests/test_hierarchical_softmax.py::TestHierarchicalSoftmax

# Test a specific test method
uv run pytest tests/test_hierarchical_softmax.py::TestHierarchicalSoftmax::test_forward_basic
```

### Run Tests by Markers
```bash
# Run only unit tests
uv run pytest -m unit

# Run only integration tests  
uv run pytest -m integration

# Skip slow tests
uv run pytest -m "not slow"
```

### Use the Test Runner Script
```bash
# Basic test run
./run_tests.py

# With coverage
./run_tests.py --coverage
```

## Test Structure

The test suite is organized as follows:

```
tests/
├── __init__.py                      # Test package
├── conftest.py                      # Shared fixtures and utilities
├── test_huffman_tree.py            # Tests for HuffmanTree class
├── test_hierarchical_softmax.py    # Tests for HierarchicalSoftmax class
└── test_integration.py             # Integration and performance tests
```

### Test Categories

- **Unit Tests** (`@pytest.mark.unit`): Test individual components in isolation
- **Integration Tests** (`@pytest.mark.integration`): Test component interactions
- **Slow Tests** (`@pytest.mark.slow`): Performance benchmarks and large-scale tests

## Coverage

Coverage reports are generated in:
- Terminal output (when using `--cov-report=term-missing`)
- HTML report in `htmlcov/` directory (when using `--cov-report=html`)

Target coverage is currently set to 50% to account for the gradual addition of tests.

## Key Test Areas

### HuffmanTree Tests (`test_huffman_tree.py`)
- Tree construction correctness
- Binary code generation
- Path consistency
- Edge cases (single word, empty vocab)
- Performance with large vocabularies

### HierarchicalSoftmax Tests (`test_hierarchical_softmax.py`)
- Model initialization
- Forward pass functionality
- Gradient computation
- Probability prediction
- Device compatibility
- Various vocabulary sizes

### Integration Tests (`test_integration.py`)
- Realistic training simulation
- Memory efficiency comparisons
- Performance benchmarks
- Numerical stability
- Edge case robustness

## Writing New Tests

### Test Structure
```python
import pytest
from src.modern_word2vec.your_module import YourClass

class TestYourClass:
    def test_basic_functionality(self, fixture_name):
        """Test description."""
        # Arrange
        instance = YourClass()
        
        # Act
        result = instance.method()
        
        # Assert
        assert result == expected_value
```

### Using Fixtures
Common fixtures are defined in `conftest.py`:
- `small_vocab`: Small vocabulary for quick tests
- `small_word_counts`: Word frequency counts
- `medium_vocab`: Larger vocabulary for more comprehensive tests
- `sample_embeddings`: Pre-generated embeddings

### Test Utilities
- `assert_tensor_close()`: Compare tensors with tolerance
- `assert_valid_probability_distribution()`: Verify probability properties
- `create_mock_dataset()`: Create mock dataset objects

## Best Practices

1. **Test Naming**: Use descriptive names that explain what is being tested
2. **Test Organization**: Group related tests in classes
3. **Fixtures**: Use fixtures for common setup to avoid repetition
4. **Assertions**: Use specific assertions with clear error messages
5. **Edge Cases**: Always test boundary conditions and edge cases
6. **Performance**: Mark expensive tests with `@pytest.mark.slow`
7. **Isolation**: Tests should be independent and not rely on execution order

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure the package is installed in development mode:
   ```bash
   uv sync
   ```

2. **CUDA Tests Failing**: GPU tests are skipped automatically if CUDA is unavailable

3. **Coverage Too Low**: Either write more tests or adjust the threshold in `pyproject.toml`

4. **Slow Tests**: Use `-m "not slow"` to skip performance benchmarks during development
